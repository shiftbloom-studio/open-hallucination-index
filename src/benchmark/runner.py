"""
OHI Benchmark Runner
====================

Main benchmark orchestration class that coordinates:
- Dataset loading and validation
- API health checking and warmup
- Parallel verification execution
- Metrics computation and report generation
"""

from __future__ import annotations

import asyncio
import csv
import logging
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from benchmark.config import BenchmarkConfig, get_config
from benchmark.metrics import (
    CalibrationMetrics,
    ConfusionMatrix,
    LatencyStats,
    PRCurveAnalysis,
    ROCAnalysis,
)
from benchmark.models import (
    BenchmarkCase,
    BenchmarkReport,
    ResultMetric,
    StatisticalComparison,
    StrategyReport,
)
from benchmark.analysis.statistical import (
    bootstrap_auc_ci,
    bootstrap_metric_ci,
    delong_from_results,
    mcnemar_from_results,
)
from benchmark.reporters import (
    ChartsReporter,
    ConsoleReporter,
    CSVReporter,
    JSONReporter,
    MarkdownReporter,
)

logger = logging.getLogger("OHI-Benchmark")


class OHIBenchmarkRunner:
    """
    Research-grade benchmark runner for OHI API evaluation.

    Provides comprehensive evaluation of verification strategies
    with statistical significance testing and multi-format reporting.

    Usage:
        async with OHIBenchmarkRunner(config=config) as runner:
            report = await runner.run_benchmark()
    """

    def __init__(
        self,
        config: BenchmarkConfig | None = None,
        strategies: list[str] | None = None,
        threshold: float | None = None,
        concurrency: int | None = None,
        warmup_requests: int | None = None,
        verbose: bool = False,
        console: Console | None = None,
    ) -> None:
        """
        Initialize benchmark runner.

        Args:
            config: Full configuration object (takes precedence).
            strategies: List of strategies to test (overrides config).
            threshold: Decision threshold (overrides config).
            concurrency: Parallel requests (overrides config).
            warmup_requests: Warmup count (overrides config).
            verbose: Enable verbose logging.
            console: Rich console for output.
        """
        # Start with default or provided config
        self.config = config or get_config()

        # Apply parameter overrides
        overrides: dict[str, Any] = {"verbose": verbose}
        if strategies is not None:
            overrides["strategies"] = strategies
        if threshold is not None:
            overrides["threshold"] = threshold
        if concurrency is not None:
            overrides["concurrency"] = concurrency
        if warmup_requests is not None:
            overrides["warmup_requests"] = warmup_requests

        if overrides:
            self.config = self.config.with_overrides(**overrides)

        # Runtime state
        self.console = console or Console(record=True)
        self.client: httpx.AsyncClient = None  # type: ignore[assignment]
        self.semaphore: asyncio.Semaphore = None  # type: ignore[assignment]
        self.results: list[ResultMetric] = []
        self.cases: list[BenchmarkCase] = []

        # Run metadata
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        if self.config.verbose:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            )

    async def __aenter__(self) -> "OHIBenchmarkRunner":
        """Async context manager entry."""
        headers = {}
        if self.config.api_key:
            headers["X-API-Key"] = self.config.api_key
        self.client = httpx.AsyncClient(timeout=self.config.timeout_seconds, headers=headers)
        self.semaphore = asyncio.Semaphore(self.config.concurrency)
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self.client:
            await self.client.aclose()

    # =========================================================================
    # Dataset Loading
    # =========================================================================

    def load_dataset(self) -> list[BenchmarkCase]:
        """
        Load and validate benchmark cases from CSV.

        Returns:
            List of BenchmarkCase objects.

        Raises:
            SystemExit if dataset cannot be loaded.
        """
        cases: list[BenchmarkCase] = []

        try:
            with open(self.config.dataset_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label_str = str(row.get("label", "")).strip().lower()
                    label_bool = label_str in ("true", "1", "yes")

                    cases.append(
                        BenchmarkCase(
                            id=int(row["id"]),
                            domain=row.get("domain", "general").strip().lower(),
                            difficulty=row.get("difficulty", "medium").strip().lower(),
                            label=label_bool,
                            text=row["text"].strip(),
                            notes=row.get("notes", ""),
                            hallucination_type=row.get("hallucination_type"),
                        )
                    )

            # Dataset statistics
            domains = set(c.domain for c in cases)
            difficulties = set(c.difficulty for c in cases)
            facts = sum(1 for c in cases if c.label)
            hallucinations = len(cases) - facts
            multi_claim = sum(1 for c in cases if c.is_multi_claim)

            self.console.print(
                Panel(
                    f"[bold green]✔ Dataset Loaded[/bold green]\n\n"
                    f"Total cases: [cyan]{len(cases)}[/cyan]\n"
                    f"Facts: [green]{facts}[/green] | "
                    f"Hallucinations: [red]{hallucinations}[/red]\n"
                    f"Multi-claim texts: [yellow]{multi_claim}[/yellow]\n"
                    f"Domains: {', '.join(sorted(domains))}\n"
                    f"Difficulties: {', '.join(sorted(difficulties))}",
                    title="Dataset",
                    border_style="green",
                )
            )

            self.cases = cases
            return cases

        except FileNotFoundError:
            self.console.print(
                f"[red]✘ Dataset not found: {self.config.dataset_path}[/red]"
            )
            raise SystemExit(1) from None

        except Exception as e:
            self.console.print(f"[red]✘ Error loading dataset: {e}[/red]")
            logger.exception("Dataset loading failed")
            raise SystemExit(1) from None

    # =========================================================================
    # API Health & Warmup
    # =========================================================================

    async def check_api_health(self) -> dict[str, Any]:
        """
        Verify API is accessible and healthy.

        Returns:
            Health status dictionary.
        """
        try:
            response = await self.client.get(
                self.config.api_health_url, timeout=10.0
            )

            if response.status_code == 200:
                content_type = response.headers.get("content-type", "")
                data = response.json() if "application/json" in content_type else {}
                self.console.print(
                    f"[green]✔ API health check passed[/green] "
                    f"({self.config.api_base_url})"
                )
                return {"status": "healthy", "data": data}

            self.console.print(
                f"[yellow]⚠ API returned status {response.status_code}[/yellow]"
            )
            return {"status": "degraded", "code": response.status_code}

        except httpx.ConnectError:
            self.console.print(
                f"[red]✘ Cannot connect to API at {self.config.api_base_url}[/red]"
            )
            return {"status": "unreachable", "error": "connection_failed"}

        except Exception as e:
            self.console.print(f"[red]✘ API health check failed: {e}[/red]")
            return {"status": "error", "error": str(e)}

    async def warmup(self) -> None:
        """Execute warmup requests to stabilize API performance."""
        if self.config.warmup_requests <= 0:
            return

        self.console.print(
            f"\n[dim]Warming up API with {self.config.warmup_requests} requests...[/dim]"
        )

        warmup_texts = [
            "The sky is blue.",
            "Water boils at 100 degrees Celsius at sea level.",
            "Python is a programming language.",
            "The Earth orbits the Sun.",
            "2 + 2 = 4",
        ]

        for i in range(self.config.warmup_requests):
            text = warmup_texts[i % len(warmup_texts)]
            try:
                await self.client.post(
                    self.config.api_verify_url,
                    json={
                        "text": text,
                        "strategy": self.config.strategies[0],
                        "use_cache": False,
                        "skip_decomposition": True,
                    },
                    timeout=30.0,
                )
            except Exception:
                pass  # Warmup failures are ignored

        self.console.print("[dim]Warmup complete.[/dim]")

    # =========================================================================
    # Verification Execution
    # =========================================================================

    async def _verify_single(
        self, case: BenchmarkCase, strategy: str
    ) -> ResultMetric:
        """
        Execute single verification request with rate limiting.

        Args:
            case: The benchmark case to verify.
            strategy: Verification strategy to use.

        Returns:
            ResultMetric with verification result.
        """
        async with self.semaphore:
            start_time = time.perf_counter()

            try:
                response = await self.client.post(
                    self.config.api_verify_url,
                    json={
                        "text": case.text,
                        "strategy": strategy,
                        "use_cache": self.config.use_cache,
                        "target_sources": self.config.target_sources,
                        # "skip_decomposition": True,  # Disabled to allowing full pipeline testing
                    },
                    timeout=self.config.timeout_seconds,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000

                if response.status_code != 200:
                    return ResultMetric(
                        case_id=case.id,
                        strategy=strategy,
                        expected=case.label,
                        predicted=False,
                        trust_score=0.0,
                        latency_ms=latency_ms,
                        domain=case.domain,
                        difficulty=case.difficulty,
                        is_multi_claim=case.is_multi_claim,
                        error=f"HTTP {response.status_code}: {response.text[:100]}",
                    )

                data = response.json()

                # Extract trust score (handle different response formats)
                trust_score_data = data.get("trust_score", {})
                if isinstance(trust_score_data, dict):
                    trust_score = trust_score_data.get("overall", 0.0)
                else:
                    trust_score = float(trust_score_data)

                predicted = trust_score >= self.config.threshold

                return ResultMetric(
                    case_id=case.id,
                    strategy=strategy,
                    expected=case.label,
                    predicted=predicted,
                    trust_score=trust_score,
                    latency_ms=latency_ms,
                    domain=case.domain,
                    difficulty=case.difficulty,
                    is_multi_claim=case.is_multi_claim,
                    claims_count=len(data.get("claims", [])),
                    processing_time_api_ms=data.get("processing_time_ms", 0.0),
                    response_id=str(data.get("id", "")),
                    raw_response=data if self.config.verbose else None,
                )

            except asyncio.TimeoutError:
                latency_ms = (time.perf_counter() - start_time) * 1000
                return ResultMetric(
                    case_id=case.id,
                    strategy=strategy,
                    expected=case.label,
                    predicted=False,
                    trust_score=0.0,
                    latency_ms=latency_ms,
                    domain=case.domain,
                    difficulty=case.difficulty,
                    is_multi_claim=case.is_multi_claim,
                    error="Timeout",
                )

            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.warning("Case %s failed: %s", case.id, e)
                return ResultMetric(
                    case_id=case.id,
                    strategy=strategy,
                    expected=case.label,
                    predicted=False,
                    trust_score=0.0,
                    latency_ms=latency_ms,
                    domain=case.domain,
                    difficulty=case.difficulty,
                    is_multi_claim=case.is_multi_claim,
                    error=str(e)[:200],
                )

    # =========================================================================
    # Main Benchmark Execution
    # =========================================================================

    async def run_benchmark(self) -> BenchmarkReport:
        """
        Execute the complete benchmark suite.

        Returns:
            BenchmarkReport with all results and analysis.
        """
        start_time = time.perf_counter()

        # Load data and check API
        self.load_dataset()
        api_health = await self.check_api_health()

        if api_health["status"] == "unreachable":
            self.console.print("[red]Aborting: API is unreachable.[/red]")
            raise SystemExit(1)

        await self.warmup()

        # Execute verification
        total_tasks = len(self.cases) * len(self.config.strategies)

        self.console.print(
            Panel(
                f"[bold blue]🚀 Starting Benchmark[/bold blue]\n\n"
                f"API: [cyan]{self.config.api_verify_url}[/cyan]\n"
                f"Strategies: [yellow]{', '.join(self.config.strategies)}[/yellow]\n"
                f"Threshold: [magenta]{self.config.threshold}[/magenta]\n"
                f"Concurrency: [green]{self.config.concurrency}[/green]\n"
                f"Total tests: [bold]{total_tasks}[/bold]",
                title="Benchmark Configuration",
                border_style="blue",
            )
        )

        disable_progress = self.config.no_progress or not self.console.is_terminal

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
            disable=disable_progress,
        ) as progress:
            overall_task = progress.add_task(
                "[bold cyan]Overall Progress", total=total_tasks
            )

            for strategy in self.config.strategies:
                strategy_task = progress.add_task(
                    f"[yellow]{strategy}", total=len(self.cases)
                )

                # Process in batches
                batch_size = max(self.config.concurrency * 2, 1)
                for i in range(0, len(self.cases), batch_size):
                    batch = self.cases[i : i + batch_size]
                    tasks = [self._verify_single(case, strategy) for case in batch]
                    batch_results = await asyncio.gather(*tasks)
                    self.results.extend(batch_results)

                    progress.advance(overall_task, len(batch))
                    progress.advance(strategy_task, len(batch))

                progress.remove_task(strategy_task)

        # Generate report
        total_runtime = time.perf_counter() - start_time
        report = self._generate_report(total_runtime, api_health)

        # Save reports
        self._save_reports(report)

        # Display summary
        console_reporter = ConsoleReporter(self.config.output_dir, self.console)
        console_reporter.display_summary(report)

        return report

    # =========================================================================
    # Report Generation
    # =========================================================================

    def _build_confusion_matrix(
        self, results: list[ResultMetric]
    ) -> ConfusionMatrix:
        """Build confusion matrix from results."""
        cm = ConfusionMatrix()
        for r in results:
            if r.has_error:
                continue
            if r.is_tp:
                cm.tp += 1
            elif r.is_tn:
                cm.tn += 1
            elif r.is_fp:
                cm.fp += 1
            elif r.is_fn:
                cm.fn += 1
        return cm

    def _generate_report(
        self, runtime: float, api_health: dict[str, Any]
    ) -> BenchmarkReport:
        """Generate comprehensive benchmark report."""
        report = BenchmarkReport(
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_path=str(self.config.dataset_path),
            dataset_size=len(self.cases),
            threshold_used=self.config.threshold,
            strategies_tested=list(self.config.strategies),
            total_runtime_seconds=runtime,
            api_health=api_health,
            config_snapshot=asdict(self.config),
        )

        # Generate per-strategy reports
        for strategy in self.config.strategies:
            strat_results = [r for r in self.results if r.strategy == strategy]
            valid_results = [r for r in strat_results if not r.has_error]

            # Core metrics
            cm = self._build_confusion_matrix(strat_results)
            latency = LatencyStats.compute([r.latency_ms for r in valid_results])

            # Probabilistic metrics
            if valid_results:
                y_true = np.array([1 if r.expected else 0 for r in valid_results])
                y_prob = np.array([r.trust_score for r in valid_results])
                y_pred = np.array([1 if r.predicted else 0 for r in valid_results])

                calibration = CalibrationMetrics.compute(y_true, y_prob)
                roc = ROCAnalysis.compute(y_true, y_prob)
                pr_curve = PRCurveAnalysis.compute(y_true, y_prob)

                # Bootstrap confidence intervals
                def accuracy_fn(yt: np.ndarray, yp: np.ndarray) -> float:
                    return float(np.mean(yt == yp))

                def f1_fn(yt: np.ndarray, yp: np.ndarray) -> float:
                    tp = np.sum((yp == 1) & (yt == 1))
                    fp = np.sum((yp == 1) & (yt == 0))
                    fn = np.sum((yp == 0) & (yt == 1))
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                    return float(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0)

                accuracy_ci = bootstrap_metric_ci(
                    y_true, y_pred, accuracy_fn,
                    n_iterations=self.config.bootstrap_iterations,
                    confidence_level=self.config.confidence_level,
                )
                f1_ci = bootstrap_metric_ci(
                    y_true, y_pred, f1_fn,
                    n_iterations=self.config.bootstrap_iterations,
                    confidence_level=self.config.confidence_level,
                )
                auc_ci = bootstrap_auc_ci(
                    y_true, y_prob,
                    n_iterations=self.config.bootstrap_iterations,
                    confidence_level=self.config.confidence_level,
                )
            else:
                calibration = CalibrationMetrics()
                roc = ROCAnalysis()
                pr_curve = PRCurveAnalysis()
                accuracy_ci = None
                f1_ci = None
                auc_ci = None

            # Stratified analysis
            by_domain = {
                domain: self._build_confusion_matrix(
                    [r for r in strat_results if r.domain == domain]
                )
                for domain in set(r.domain for r in strat_results)
            }

            by_difficulty = {
                diff: self._build_confusion_matrix(
                    [r for r in strat_results if r.difficulty == diff]
                )
                for diff in set(r.difficulty for r in strat_results)
            }

            # Multi-claim analysis
            multi_results = [r for r in strat_results if r.is_multi_claim]
            single_results = [r for r in strat_results if not r.is_multi_claim]

            # Error analysis
            fp_cases = sorted(
                [r for r in strat_results if r.is_fp],
                key=lambda x: x.trust_score,
                reverse=True,
            )
            fn_cases = sorted(
                [r for r in strat_results if r.is_fn],
                key=lambda x: x.trust_score,
            )

            worst_fp = [
                {
                    "case_id": r.case_id,
                    "trust_score": r.trust_score,
                    "text": next(
                        (c.text for c in self.cases if c.id == r.case_id), ""
                    )[:100],
                }
                for r in fp_cases[:5]
            ]
            worst_fn = [
                {
                    "case_id": r.case_id,
                    "trust_score": r.trust_score,
                    "text": next(
                        (c.text for c in self.cases if c.id == r.case_id), ""
                    )[:100],
                }
                for r in fn_cases[:5]
            ]

            error_count = sum(1 for r in strat_results if r.has_error)

            report.strategy_reports[strategy] = StrategyReport(
                strategy=strategy,
                confusion_matrix=cm,
                calibration=calibration,
                latency=latency,
                roc=roc,
                pr_curve=pr_curve,
                by_domain=by_domain,
                by_difficulty=by_difficulty,
                multi_claim_cm=(
                    self._build_confusion_matrix(multi_results) if multi_results else None
                ),
                single_claim_cm=(
                    self._build_confusion_matrix(single_results) if single_results else None
                ),
                worst_fp_cases=worst_fp,
                worst_fn_cases=worst_fn,
                error_count=error_count,
                error_rate=error_count / len(strat_results) if strat_results else 0.0,
                accuracy_ci=accuracy_ci,
                f1_ci=f1_ci,
                auc_ci=auc_ci,
            )

        # Find best performers
        if report.strategy_reports:
            report.best_strategy_accuracy = max(
                report.strategy_reports.keys(),
                key=lambda s: report.strategy_reports[s].confusion_matrix.accuracy,
            )
            report.best_strategy_f1 = max(
                report.strategy_reports.keys(),
                key=lambda s: report.strategy_reports[s].confusion_matrix.f1_score,
            )
            report.best_strategy_mcc = max(
                report.strategy_reports.keys(),
                key=lambda s: report.strategy_reports[s].confusion_matrix.mcc,
            )
            report.best_strategy_auc = max(
                report.strategy_reports.keys(),
                key=lambda s: (
                    report.strategy_reports[s].roc.auc
                    if report.strategy_reports[s].roc
                    else 0.0
                ),
            )
            report.lowest_hallucination_rate = min(
                report.strategy_reports.keys(),
                key=lambda s: report.strategy_reports[
                    s
                ].confusion_matrix.hallucination_pass_rate,
            )

        # Statistical comparisons between strategies
        if len(self.config.strategies) >= 2:
            report.comparisons = self._compute_comparisons()
            # Legacy format for backward compatibility
            report.mcnemar_results = self._compute_mcnemar_legacy()

        return report

    def _compute_comparisons(self) -> dict[str, StatisticalComparison]:
        """Compute all pairwise strategy comparisons."""
        comparisons: dict[str, StatisticalComparison] = {}
        strategies = list(self.config.strategies)

        for i, strat_a in enumerate(strategies):
            for strat_b in strategies[i + 1 :]:
                results_a = [r for r in self.results if r.strategy == strat_a]
                results_b = [r for r in self.results if r.strategy == strat_b]

                # McNemar test
                mcnemar = mcnemar_from_results(results_a, results_b)

                # DeLong test
                delong = delong_from_results(results_a, results_b)

                key = f"{strat_a}_vs_{strat_b}"
                comparisons[key] = StatisticalComparison(
                    strategy_a=strat_a,
                    strategy_b=strat_b,
                    mcnemar_chi2=mcnemar.chi2,
                    mcnemar_p_value=mcnemar.p_value,
                    delong_z=delong.z_statistic,
                    delong_p_value=delong.p_value,
                    auc_difference=delong.auc_diff,
                    is_significant=mcnemar.is_significant or delong.is_significant,
                )

        return comparisons

    def _compute_mcnemar_legacy(self) -> dict[str, dict[str, float]]:
        """Compute McNemar tests in legacy format."""
        results_by_case: dict[int, dict[str, bool]] = defaultdict(dict)
        for r in self.results:
            if not r.has_error:
                results_by_case[r.case_id][r.strategy] = r.is_correct

        mcnemar_results: dict[str, dict[str, float]] = {}
        strategies = list(self.config.strategies)

        for i, strat_a in enumerate(strategies):
            for strat_b in strategies[i + 1 :]:
                b = 0  # A correct, B wrong
                c = 0  # A wrong, B correct

                for preds in results_by_case.values():
                    if strat_a in preds and strat_b in preds:
                        if preds[strat_a] and not preds[strat_b]:
                            b += 1
                        elif not preds[strat_a] and preds[strat_b]:
                            c += 1

                if b + c > 0:
                    from scipy import stats as scipy_stats

                    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                    p_value = float(1 - scipy_stats.chi2.cdf(chi2, 1))
                else:
                    chi2 = 0.0
                    p_value = 1.0

                key = f"{strat_a}_vs_{strat_b}"
                mcnemar_results[key] = {
                    "chi2": chi2,
                    "p_value": p_value,
                    "significant_at_05": p_value < 0.05,
                    "b": float(b),
                    "c": float(c),
                }

        return mcnemar_results

    # =========================================================================
    # Report Saving
    # =========================================================================

    def _save_reports(self, report: BenchmarkReport) -> None:
        """Save reports in all configured formats."""
        base_filename = f"ohi_benchmark_{self.run_id}"

        for fmt in self.config.output_formats:
            if fmt == "csv":
                reporter = CSVReporter(self.config.output_dir)
            elif fmt == "json":
                reporter = JSONReporter(self.config.output_dir)
            elif fmt == "markdown":
                reporter = MarkdownReporter(self.config.output_dir)
            elif fmt == "html":
                reporter = ConsoleReporter(self.config.output_dir, self.console)
            elif fmt == "charts":
                reporter = ChartsReporter(self.config.output_dir)
            else:
                continue

            try:
                filepath = reporter.save(report, self.results, base_filename)
                logger.info("Saved %s report: %s", fmt, filepath)
            except Exception as e:
                logger.warning("Failed to save %s report: %s", fmt, e)

        # Always generate performance charts (if matplotlib available)
        try:
            charts_reporter = ChartsReporter(self.config.output_dir)
            chart_files = charts_reporter.generate_all_charts(
                report, self.results, base_filename
            )
            if chart_files:
                self.console.print(
                    f"\n[green]📊 Generated {len(chart_files)} performance charts:[/green]"
                )
                for chart_path in chart_files:
                    self.console.print(f"  [dim]→ {chart_path.name}[/dim]")
        except ImportError:
            self.console.print(
                "\n[dim]⚠ Charts not generated (matplotlib not installed)[/dim]"
            )
        except Exception as e:
            logger.warning("Failed to generate charts: %s", e)

        self.console.print(f"\n[dim]Reports saved to: {self.config.output_dir}/[/dim]")
