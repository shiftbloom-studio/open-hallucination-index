from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from benchmark.comparison_config import ComparisonBenchmarkConfig
from benchmark.comparison_benchmark import ComparisonReport
from benchmark.evaluators import BaseEvaluator, get_evaluator
from benchmark.reporters.charts import ChartsReporter
from benchmark.runner._modes import benchmark_single_evaluator
from benchmark.runner._benchmarks import StopRequested
from benchmark.runner._cache import CacheManager
from benchmark.runner._types import LiveStats


class GuiBenchmarkDisplay:
    def __init__(
        self,
        stats: LiveStats,
        on_stats: Callable[[dict[str, Any]], None],
        on_log: Callable[[str], None],
        on_evaluator_done: Callable[[str, dict[str, Any]], None],
        on_verification: Callable[[dict[str, Any]], None],
        should_stop: Callable[[], bool],
    ) -> None:
        self.stats = stats
        self._on_stats = on_stats
        self._on_log = on_log
        self._on_evaluator_done = on_evaluator_done
        self._on_verification = on_verification
        self._should_stop = should_stop

    def start_task(self, description: str, total: int) -> None:
        self.stats.current_metric = description
        self.stats.current_total = total
        self.stats.reset_task()
        self._emit()

    def advance(self, n: int = 1, latency_ms: float | None = None) -> None:
        self.stats.current_completed += n
        self.stats.total_processed += n
        if latency_ms is not None:
            self.stats.current_latencies.append(latency_ms)
        self._emit()

    def set_evaluator(self, name: str) -> None:
        self.stats.current_evaluator = name
        self._emit()

    def complete_evaluator(self, name: str, metrics: dict[str, Any]) -> None:
        self.stats.completed_evaluators += 1
        self.stats.evaluator_results[name] = metrics
        self._on_evaluator_done(name, metrics)
        self._emit()

    def add_result(self, correct: bool, error: bool = False) -> None:
        if correct:
            self.stats.current_correct += 1
            self.stats.correct += 1
        if error:
            self.stats.current_errors += 1
            self.stats.errors += 1
        self._emit()

    def force_refresh(self) -> None:
        self._emit()

    def log_error(self, message: str) -> None:
        self._on_log(f"[error] {message}")

    def log_warning(self, message: str) -> None:
        self._on_log(f"[warn] {message}")

    def record_verification(self, payload: dict[str, Any]) -> None:
        self._on_verification(payload)

    def should_stop(self) -> bool:
        return self._should_stop()

    def _emit(self) -> None:
        payload = {
            "total_evaluators": self.stats.total_evaluators,
            "completed_evaluators": self.stats.completed_evaluators,
            "current_evaluator": self.stats.current_evaluator,
            "current_metric": self.stats.current_metric,
            "current_total": self.stats.current_total,
            "current_completed": self.stats.current_completed,
            "current_correct": self.stats.current_correct,
            "current_errors": self.stats.current_errors,
            "total_processed": self.stats.total_processed,
            "errors": self.stats.errors,
            "latencies": list(self.stats.current_latencies),
            "timestamp": time.time(),
        }
        self._on_stats(payload)


class BenchmarkWorker(QThread):
    stats_updated = Signal(dict)
    log_message = Signal(str)
    evaluator_completed = Signal(str, dict)
    verification_recorded = Signal(dict)
    report_updated = Signal(object)  # ComparisonReport object
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, config_overrides: dict[str, Any]) -> None:
        super().__init__()
        self._overrides = config_overrides
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except StopRequested:
            self.finished.emit({"status": "stopped"})
        except Exception as exc:  # pragma: no cover
            self.failed.emit(f"{type(exc).__name__}: {exc}")

    async def _run_async(self) -> None:
        config = ComparisonBenchmarkConfig.from_env()
        self._apply_overrides(config)

        cache = CacheManager(
            host=config.redis_host,
            port=config.redis_port,
            password=config.redis_password,
        )

        evaluators = await self._initialize_evaluators(config)
        if not evaluators:
            self.log_message.emit("⚠ No evaluators available. Check API host/port and DB hosts.")
            self.failed.emit("No evaluators available")
            return

        run_id = self._make_run_id()
        report = ComparisonReport(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_summary={
                "evaluators": list(evaluators.keys()),
                "metrics": config.metrics,
                "hallucination_dataset": str(config.hallucination_dataset),
                "ohi_all_strategies": config.ohi_all_strategies,
                "cache_testing": config.cache_testing,
            },
        )
        
        # Emit initial report
        self.report_updated.emit(report)

        stats = LiveStats(
            total_evaluators=self._total_evaluators(config, evaluators),
            start_time=time.perf_counter(),
        )
        display = GuiBenchmarkDisplay(
            stats,
            self.stats_updated.emit,
            self.log_message.emit,
            self.evaluator_completed.emit,
            self.verification_recorded.emit,
            lambda: self._stop_requested,
        )

        cache.connect(None)
        if cache.is_connected:
            cache.flush("before_benchmark")
            self.log_message.emit("Redis cache flushed before benchmark.")

        if self._stop_requested:
            self.log_message.emit("⚠ Benchmark stopped by user.")
            self.finished.emit({"status": "stopped"})
            return

        if config.complete_mode:
            await self._run_complete_mode(
                evaluators,
                report,
                config,
                cache,
                display,
            )
        elif config.ohi_all_strategies:
            await self._run_strategy_comparison(
                evaluators,
                report,
                config,
                cache,
                display,
            )
        elif config.cache_testing:
            await self._run_cache_comparison(
                evaluators,
                report,
                config,
                cache,
                display,
            )
        else:
            await self._run_standard_comparison(
                evaluators,
                report,
                config,
                cache,
                display,
            )

        if self._stop_requested:
            self.log_message.emit("⚠ Benchmark stopped by user.")
            self.finished.emit({"status": "stopped"})
            cache.close()
            return

        await self._generate_outputs(report, config)
        cache.close()
        self.finished.emit(report.to_dict())

    def _apply_overrides(self, config: ComparisonBenchmarkConfig) -> None:
        for key, value in self._overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    async def _initialize_evaluators(
        self,
        config: ComparisonBenchmarkConfig,
    ) -> dict[str, BaseEvaluator]:
        evaluators: dict[str, BaseEvaluator] = {}
        for name in config.get_active_evaluators():
            try:
                evaluator = get_evaluator(name, config)
                if await evaluator.health_check():
                    evaluators[name] = evaluator
                    self.log_message.emit(f"✓ {evaluator.name} ready")
                else:
                    self.log_message.emit(f"⚠ {name} unavailable")
            except Exception as exc:
                self.log_message.emit(f"⚠ {name} failed: {type(exc).__name__}: {exc}")
        return evaluators

    async def _run_standard_comparison(
        self,
        evaluators: dict[str, BaseEvaluator],
        report: ComparisonReport,
        config: ComparisonBenchmarkConfig,
        cache: CacheManager,
        display: GuiBenchmarkDisplay,
    ) -> None:
        for evaluator in evaluators.values():
            display.set_evaluator(evaluator.name)
            metrics = await benchmark_single_evaluator(
                evaluator=evaluator,
                display=display,
                stats=display.stats,
                config=config,
                cache=cache,
                max_latency_ms=5 * 60 * 1000,
            )
            report.add_evaluator(metrics)
            display.complete_evaluator(evaluator.name, self._summary_payload(metrics))
            # Emit partial report after each evaluator completes
            self.report_updated.emit(report)

    async def _run_strategy_comparison(
        self,
        evaluators: dict[str, BaseEvaluator],
        report: ComparisonReport,
        config: ComparisonBenchmarkConfig,
        cache: CacheManager,
        display: GuiBenchmarkDisplay,
    ) -> None:
        for name, evaluator in evaluators.items():
            if name != "ohi":
                display.set_evaluator(evaluator.name)
                metrics = await benchmark_single_evaluator(
                    evaluator=evaluator,
                    display=display,
                    stats=display.stats,
                    config=config,
                    cache=cache,
                    max_latency_ms=5 * 60 * 1000,
                )
                report.add_evaluator(metrics)
                display.complete_evaluator(evaluator.name, self._summary_payload(metrics))
                # Emit partial report after each non-OHI evaluator in strategy mode
                self.report_updated.emit(report)

        if "ohi" in evaluators:
            from benchmark.evaluators import OHIEvaluator

            for strategy in config.ohi_strategies:
                display.set_evaluator(f"OHI ({strategy})")
                strategy_config = ComparisonBenchmarkConfig.from_env()
                strategy_config.ohi_strategy = strategy
                strategy_config.ohi_api_host = config.ohi_api_host
                strategy_config.ohi_api_port = config.ohi_api_port
                strategy_config.ohi_api_key = config.ohi_api_key
                strategy_evaluator = OHIEvaluator(strategy_config)
                try:
                    if await strategy_evaluator.health_check():
                        metrics = await benchmark_single_evaluator(
                            evaluator=strategy_evaluator,
                            display=display,
                            stats=display.stats,
                            config=config,
                            cache=cache,
                            max_latency_ms=5 * 60 * 1000,
                        )
                        metrics.evaluator_name = f"OHI ({strategy})"
                        report.add_evaluator(metrics)
                        display.complete_evaluator(
                            f"OHI ({strategy})",
                            self._summary_payload(metrics),
                        )
                        # Emit partial report after each strategy completes
                        self.report_updated.emit(report)
                finally:
                    await strategy_evaluator.close()

    async def _run_cache_comparison(
        self,
        evaluators: dict[str, BaseEvaluator],
        report: ComparisonReport,
        config: ComparisonBenchmarkConfig,
        cache: CacheManager,
        display: GuiBenchmarkDisplay,
    ) -> None:
        for evaluator in evaluators.values():
            display.set_evaluator(f"{evaluator.name} (Cold)")
            cache.clear_ohi_keys()
            metrics_cold = await benchmark_single_evaluator(
                evaluator=evaluator,
                display=display,
                stats=display.stats,
                config=config,
                cache=cache,
                max_latency_ms=5 * 60 * 1000,
            )
            metrics_cold.evaluator_name = f"{evaluator.name} (Cold)"
            report.add_evaluator(metrics_cold)
            display.complete_evaluator(
                f"{evaluator.name} (Cold)",
                self._summary_payload(metrics_cold),
            )
            # Emit partial report after cold run
            self.report_updated.emit(report)

            display.set_evaluator(f"{evaluator.name} (Warm)")
            metrics_warm = await benchmark_single_evaluator(
                evaluator=evaluator,
                display=display,
                stats=display.stats,
                config=config,
                cache=cache,
                max_latency_ms=5 * 60 * 1000,
            )
            metrics_warm.evaluator_name = f"{evaluator.name} (Warm)"
            report.add_evaluator(metrics_warm)
            display.complete_evaluator(
                f"{evaluator.name} (Warm)",
                self._summary_payload(metrics_warm),
            )
            # Emit partial report after warm run
            self.report_updated.emit(report)

    async def _run_complete_mode(
        self,
        evaluators: dict[str, BaseEvaluator],
        report: ComparisonReport,
        config: ComparisonBenchmarkConfig,
        cache: CacheManager,
        display: GuiBenchmarkDisplay,
    ) -> None:
        """
        Run COMPLETE mode - research-grade comprehensive evaluation.
        
        Loads all available datasets and performs exhaustive testing with
        statistical significance analysis.
        """
        from benchmark.datasets.hallucination_loader import HallucinationLoader
        
        self.log_message.emit("🔬 COMPLETE MODE: Loading comprehensive datasets...")
        
        # Load all datasets
        loader = HallucinationLoader(config.hallucination_dataset)
        try:
            complete_dataset = loader.load_complete_benchmark_datasets(
                csv_path=config.hallucination_dataset,
                samples_per_dataset=config.complete_samples_per_dataset,
            )
            
            self.log_message.emit(
                f"✓ Loaded {complete_dataset.total} cases from {len(complete_dataset.domains)} domains"
            )
            self.log_message.emit(
                f"  • Factual: {complete_dataset.factual_count}"
            )
            self.log_message.emit(
                f"  • Hallucinations: {complete_dataset.hallucination_count}"
            )
            
            # Update config to use complete dataset
            import tempfile
            import csv as csv_module
            
            # Write combined dataset to temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='', encoding='utf-8') as tmp:
                writer = csv_module.DictWriter(
                    tmp,
                    fieldnames=['id', 'domain', 'difficulty', 'label', 'text', 'notes', 'hallucination_type', 'source']
                )
                writer.writeheader()
                for case in complete_dataset.cases:
                    writer.writerow({
                        'id': case.id,
                        'domain': case.domain,
                        'difficulty': case.difficulty,
                        'label': str(case.label),
                        'text': case.text,
                        'notes': case.notes,
                        'hallucination_type': case.hallucination_type or '',
                        'source': case.source,
                    })
                tmp_path = Path(tmp.name)
            
            # Override config for complete mode
            config.hallucination_dataset = tmp_path
            config.hallucination_max_samples = max(
                complete_dataset.total,
                config.complete_min_verifications
            )
            
            # Ensure all metrics are enabled
            config.metrics = ["hallucination", "truthfulqa", "factscore", "latency"]
            
        except Exception as e:
            self.log_message.emit(f"⚠ Error loading complete datasets: {e}")
            self.log_message.emit("Falling back to standard mode...")
            await self._run_standard_comparison(evaluators, report, config, cache, display)
            return
        
        # Run evaluation for each evaluator
        for evaluator in evaluators.values():
            display.set_evaluator(f"{evaluator.name} (COMPLETE)")
            
            self.log_message.emit(
                f"🔬 Running comprehensive evaluation for {evaluator.name}..."
            )
            
            metrics = await benchmark_single_evaluator(
                evaluator=evaluator,
                display=display,
                stats=display.stats,
                config=config,
                cache=cache,
                max_latency_ms=10 * 60 * 1000,  # 10 min timeout for complete mode
            )
            
            metrics.evaluator_name = f"{evaluator.name} (COMPLETE)"
            report.add_evaluator(metrics)
            display.complete_evaluator(
                f"{evaluator.name} (COMPLETE)",
                self._summary_payload(metrics),
            )
            
            # Add complete mode metadata to report
            if not hasattr(report, 'complete_mode_metadata'):
                report.complete_mode_metadata = {}  # type: ignore
            
            report.complete_mode_metadata[evaluator.name] = {  # type: ignore
                'total_datasets': len(set(case.source for case in complete_dataset.cases)),
                'total_cases': complete_dataset.total,
                'factual_cases': complete_dataset.factual_count,
                'hallucination_cases': complete_dataset.hallucination_count,
                'domains': list(complete_dataset.domains),
                'samples_per_dataset': config.complete_samples_per_dataset,
            }
            
            self.report_updated.emit(report)
        
        # Generate comprehensive report with statistical analysis
        if config.complete_statistical_significance:
            self.log_message.emit("📊 Computing statistical significance...")
            report = self._add_statistical_analysis(report, config)
            self.report_updated.emit(report)
            
            # Generate research-grade markdown report
            try:
                from benchmark.reporters.research_report import ResearchReportGenerator
                
                self.log_message.emit("📝 Generating research-grade report...")
                report_gen = ResearchReportGenerator(report, config.output_dir)
                statements = report_gen.generate_performance_statements()
                report_path = report_gen.save_report(statements)
                
                self.log_message.emit(f"✓ Research report saved: {report_path.name}")
                
                # Log executive summary
                best = statements[0]
                self.log_message.emit("")
                self.log_message.emit("=== COMPLETE MODE RESULTS ===")
                self.log_message.emit(f"Top System: {best.evaluator_name}")
                self.log_message.emit(f"Accuracy: {best.primary_value:.1%} (95% CI: [{best.confidence_interval[0]:.1%}, {best.confidence_interval[1]:.1%}])")
                self.log_message.emit(f"Recommendation: {best.recommendation[:100]}...")
                self.log_message.emit("=" * 30)
                
            except Exception as e:
                self.log_message.emit(f"⚠ Error generating research report: {e}")
        
        # Clean up temporary file
        try:
            tmp_path.unlink()
        except Exception:
            pass

    def _add_statistical_analysis(
        self,
        report: ComparisonReport,
        config: ComparisonBenchmarkConfig,
    ) -> ComparisonReport:
        """
        Add statistical significance testing to report.
        
        Computes:
        - Bootstrap confidence intervals
        - Paired t-tests for accuracy comparisons
        - Effect sizes (Cohen's d)
        - McNemar's test for classification differences
        """
        from scipy import stats
        
        if not hasattr(report, 'statistical_analysis'):
            report.statistical_analysis = {}  # type: ignore
        
        # Get all evaluator results
        evaluator_names = list(report.evaluators.keys())
        
        if len(evaluator_names) < 2:
            return report
        
        # Perform pairwise comparisons
        comparisons = []
        for i, eval1 in enumerate(evaluator_names):
            for eval2 in evaluator_names[i+1:]:
                metrics1 = report.evaluators[eval1]
                metrics2 = report.evaluators[eval2]
                
                # Get accuracy values
                acc1 = metrics1.hallucination.accuracy
                acc2 = metrics2.hallucination.accuracy
                
                # Compute effect size (Cohen's d)
                # Using pooled standard deviation estimate
                n1 = metrics1.hallucination.total
                n2 = metrics2.hallucination.total
                
                if n1 > 0 and n2 > 0:
                    # Estimate standard deviations from accuracy
                    std1 = (acc1 * (1 - acc1)) ** 0.5
                    std2 = (acc2 * (1 - acc2)) ** 0.5
                    pooled_std = ((std1**2 + std2**2) / 2) ** 0.5
                    
                    cohens_d = (acc1 - acc2) / pooled_std if pooled_std > 0 else 0.0
                    
                    # Interpret effect size
                    if abs(cohens_d) < 0.2:
                        effect_interp = "negligible"
                    elif abs(cohens_d) < 0.5:
                        effect_interp = "small"
                    elif abs(cohens_d) < 0.8:
                        effect_interp = "medium"
                    else:
                        effect_interp = "large"
                    
                    comparisons.append({
                        'evaluator_1': eval1,
                        'evaluator_2': eval2,
                        'accuracy_diff': acc1 - acc2,
                        'cohens_d': cohens_d,
                        'effect_size': effect_interp,
                        'better': eval1 if acc1 > acc2 else eval2,
                    })
        
        report.statistical_analysis['pairwise_comparisons'] = comparisons  # type: ignore
        
        # Add confidence intervals using bootstrap
        for eval_name, metrics in report.evaluators.items():
            acc = metrics.hallucination.accuracy
            n = metrics.hallucination.total
            
            if n > 0:
                # Wilson score interval for binomial proportion
                z = 1.96  # 95% confidence
                p = acc
                denominator = 1 + z**2 / n
                center = (p + z**2 / (2*n)) / denominator
                margin = z * ((p * (1-p) / n + z**2 / (4*n**2)) ** 0.5) / denominator
                
                if not hasattr(report, 'confidence_intervals'):
                    report.confidence_intervals = {}  # type: ignore
                
                report.confidence_intervals[eval_name] = {  # type: ignore
                    'accuracy_lower': max(0.0, center - margin),
                    'accuracy_upper': min(1.0, center + margin),
                    'confidence_level': 0.95,
                }
        
        return report

    async def _generate_outputs(
        self,
        report: ComparisonReport,
        config: ComparisonBenchmarkConfig,
    ) -> None:
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{report.run_id}_report.json"
        json_payload = report.to_dict()
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        charts_reporter = ChartsReporter(output_dir, dpi=config.chart_dpi)
        # Generate all individual comparison charts (not just dashboard)
        chart_files = charts_reporter.generate_comparison_charts(
            report, 
            prefix=f"{report.run_id}_",
            consolidated=False
        )
        # Log each generated chart
        for chart_path in chart_files:
            self.log_message.emit(f"✓ Generated chart: {chart_path.name}")

    @staticmethod
    def _make_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"comparison_{timestamp}_{uuid.uuid4().hex[:6]}"

    @staticmethod
    def _total_evaluators(
        config: ComparisonBenchmarkConfig,
        evaluators: dict[str, BaseEvaluator],
    ) -> int:
        if config.cache_testing:
            return len(evaluators) * 2
        if config.ohi_all_strategies:
            non_ohi = sum(1 for name in evaluators if name != "ohi")
            return non_ohi + len(config.ohi_strategies)
        return len(evaluators)

    @staticmethod
    def _summary_payload(metrics: Any) -> dict[str, Any]:
        # Derived metrics: faithfulness measures claim-evidence relevance
        rag = metrics.hallucination.ragas_proxy_metrics() if hasattr(metrics, 'hallucination') else {}
        return {
            "accuracy": metrics.hallucination.accuracy,
            "f1": metrics.hallucination.f1_score,
            "hpr": metrics.hallucination.hallucination_pass_rate,
            "aurc": getattr(metrics.hallucination, 'aurc', 0.0),
            "eaurc": getattr(metrics.hallucination, 'eaurc', 0.0),
            # Evidence relevance: how well evidence supports the claim (TF-IDF based)
            "evidence_relevance": float(rag.get('faithfulness', 0.0)) if rag else 0.0,
            "p50": metrics.latency.p50,
            "p95": metrics.latency.p95,
            "truthfulqa": metrics.truthfulqa.accuracy,
            "factscore": metrics.factscore.avg_factscore,
            "throughput": metrics.latency.throughput,
            "summary_scores": metrics.get_summary_scores(),
        }


class BenchmarkWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OHI Benchmark Studio")
        self.resize(1360, 860)

        self._worker: BenchmarkWorker | None = None
        self._last_plot_update: float = 0.0
        self._latencies = deque(maxlen=1200)
        self._accuracy_series = deque(maxlen=1200)
        self._throughput_series = deque(maxlen=1200)
        self._time_series = deque(maxlen=1200)
        self._evaluator_summaries: dict[str, dict[str, float]] = {}
        self._evaluator_meta: dict[str, dict[str, float]] = {}
        self._tree_index: dict[tuple[str, str, str], QTreeWidgetItem] = {}
        self._current_report: ComparisonReport | None = None
        self._current_config: ComparisonBenchmarkConfig | None = None

        self._build_ui()
        self._apply_dark_theme()
        self._refresh_reports_list()

    def _apply_dark_theme(self) -> None:
        QApplication.setStyle("Fusion")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(18, 20, 26))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(28, 30, 38))
        palette.setColor(QPalette.AlternateBase, QColor(38, 42, 52))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(46, 50, 62))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor(91, 169, 255))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(palette)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)

        layout.addWidget(self._build_config_panel())
        layout.addWidget(self._build_visual_panel(), 1)

        self.setCentralWidget(root)

    def _build_config_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        config_group = QGroupBox("Benchmark Configuration")
        form = QFormLayout(config_group)

        self.api_host = QLineEdit("localhost")
        self.api_port = QLineEdit("8080")
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.Password)
        # Pre-populate API key from environment if available
        env_api_key = os.getenv("API_API_KEY", "")
        if env_api_key:
            self.api_key.setText(env_api_key)
            self.api_key.setPlaceholderText("Using API_API_KEY from environment")
        else:
            self.api_key.setPlaceholderText("Enter API key or set API_API_KEY in .env")
        self.concurrency = QSpinBox()
        self.concurrency.setRange(1, 50)
        self.concurrency.setValue(3)
        self.warmup = QSpinBox()
        self.warmup.setRange(0, 50)
        self.warmup.setValue(5)
        self.timeout = QSpinBox()
        self.timeout.setRange(10, 600)
        self.timeout.setValue(240)
        self.chart_dpi = QSpinBox()
        self.chart_dpi.setRange(80, 300)
        self.chart_dpi.setValue(200)

        self.output_dir = QLineEdit("benchmark_results/comparison")
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(self._choose_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir)
        output_row.addWidget(output_btn)
        output_wrap = QWidget()
        output_wrap.setLayout(output_row)

        self.dataset_path = QLineEdit()
        dataset_btn = QPushButton("Select")
        dataset_btn.clicked.connect(self._choose_dataset)
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.dataset_path)
        dataset_row.addWidget(dataset_btn)
        dataset_wrap = QWidget()
        dataset_wrap.setLayout(dataset_row)

        self.metrics_combo = QComboBox()
        self.metrics_combo.addItems(["hallucination", "truthfulqa", "factscore", "latency"])
        self.metrics_combo.setEditable(False)

        self.use_all_metrics = QCheckBox("Use all metrics")
        self.use_all_metrics.setChecked(True)

        # Special modes
        self.ohi_all_strategies = QCheckBox("Compare all OHI strategies")
        self.cache_testing = QCheckBox("Cache testing (cold/warm)")
        self.complete_mode = QCheckBox("COMPLETE mode (research-grade)")
        self.complete_mode.setToolTip(
            "Research-grade comprehensive evaluation:\n"
            "• Loads all HuggingFace datasets\n"
            "• Balanced sampling (200 per dataset)\n"
            "• Statistical significance testing\n"
            "• Multi-domain analysis"
        )
        
        # Complete mode parameters
        self.complete_samples = QSpinBox()
        self.complete_samples.setRange(50, 500)
        self.complete_samples.setValue(200)
        self.complete_samples.setSuffix(" per dataset")
        self.complete_samples.setEnabled(False)
        
        # Enable/disable complete mode parameters based on checkbox
        self.complete_mode.toggled.connect(
            lambda checked: self.complete_samples.setEnabled(checked)
        )
        
        # Disable other special modes when complete mode is enabled
        self.complete_mode.toggled.connect(
            lambda checked: self.ohi_all_strategies.setEnabled(not checked)
        )
        self.complete_mode.toggled.connect(
            lambda checked: self.cache_testing.setEnabled(not checked)
        )

        self.evaluator_checks: dict[str, QCheckBox] = {}
        for name in ["ohi_local", "ohi", "ohi_max", "vector_rag", "graph_rag", "gpt4"]:
            box = QCheckBox(name)
            box.setChecked(name != "gpt4")
            self.evaluator_checks[name] = box

        eval_group = QGroupBox("Evaluators")
        eval_layout = QVBoxLayout(eval_group)
        for box in self.evaluator_checks.values():
            eval_layout.addWidget(box)

        form.addRow("API host", self.api_host)
        form.addRow("API port", self.api_port)
        form.addRow("API key", self.api_key)
        form.addRow("Dataset (optional)", dataset_wrap)
        form.addRow("Output dir", output_wrap)
        form.addRow("Concurrency", self.concurrency)
        form.addRow("Warmup", self.warmup)
        form.addRow("Timeout (s)", self.timeout)
        form.addRow("Chart DPI", self.chart_dpi)
        form.addRow("", self.use_all_metrics)
        form.addRow("", self.ohi_all_strategies)
        form.addRow("", self.cache_testing)
        form.addRow("", self.complete_mode)
        form.addRow("Samples (COMPLETE)", self.complete_samples)

        layout.addWidget(config_group)
        layout.addWidget(eval_group)

        button_row = QHBoxLayout()
        self.start_btn = QPushButton("Run Benchmark")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.export_btn = QPushButton("Export Current State")
        self.export_btn.setEnabled(False)
        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)
        button_row.addWidget(self.export_btn)
        layout.addLayout(button_row)

        self.status_label = QLabel("Idle")
        layout.addWidget(self.status_label)
        
        # Progress bars
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        # Overall progress bar
        overall_layout = QVBoxLayout()
        self.overall_progress_label = QLabel("Overall Progress: 0 / 0 evaluators")
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setMinimum(0)
        self.overall_progress_bar.setMaximum(100)
        self.overall_progress_bar.setValue(0)
        self.overall_progress_bar.setFormat("%p% - %v/%m evaluators")
        overall_layout.addWidget(self.overall_progress_label)
        overall_layout.addWidget(self.overall_progress_bar)
        progress_layout.addLayout(overall_layout)
        
        # Current run progress bar
        current_layout = QVBoxLayout()
        self.current_progress_label = QLabel("Current Run: 0 / 0 items")
        self.current_progress_bar = QProgressBar()
        self.current_progress_bar.setMinimum(0)
        self.current_progress_bar.setMaximum(100)
        self.current_progress_bar.setValue(0)
        self.current_progress_bar.setFormat("%p% - %v/%m items")
        current_layout.addWidget(self.current_progress_label)
        current_layout.addWidget(self.current_progress_bar)
        progress_layout.addLayout(current_layout)
        
        layout.addWidget(progress_group)
        layout.addStretch(1)

        self.start_btn.clicked.connect(self._start_benchmark)
        self.stop_btn.clicked.connect(self._stop_benchmark)
        self.export_btn.clicked.connect(self._export_current_state)

        return panel

    def _build_visual_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_charts_tab(), "Live Charts")
        self.tabs.addTab(self._build_insights_tab(), "Insights")
        self.tabs.addTab(self._build_results_tab(), "Evaluator Results")
        self.tabs.addTab(self._build_reports_tab(), "Reports")
        self.tabs.addTab(self._build_log_tab(), "Logs")

        layout.addWidget(self.tabs)
        return panel

    def _build_charts_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        self.figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.ax_latency = self.figure.add_subplot(2, 2, 1)
        self.ax_accuracy = self.figure.add_subplot(2, 2, 2)
        self.ax_throughput = self.figure.add_subplot(2, 2, 3)
        self.ax_errors = self.figure.add_subplot(2, 2, 4)

        self._line_latency, = self.ax_latency.plot([], [], color="#58a6ff", label="latency ms")
        self._line_accuracy, = self.ax_accuracy.plot([], [], color="#7ee787", label="accuracy %")
        self._line_throughput, = self.ax_throughput.plot([], [], color="#f778ba", label="req/s")
        self._line_errors, = self.ax_errors.plot([], [], color="#ff7b72", label="errors")

        for ax, title in [
            (self.ax_latency, "Latency"),
            (self.ax_accuracy, "Accuracy"),
            (self.ax_throughput, "Throughput"),
            (self.ax_errors, "Errors"),
        ]:
            ax.set_title(title)
            ax.grid(True, alpha=0.2)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        return container

    def _build_results_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        splitter = QSplitter(Qt.Horizontal)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.results_table = QTableWidget(0, 9)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Evaluator",
                "Accuracy",
                "F1",
                "HPR",
                "AURC",
                "Evidence Rel.",
                "P50",
                "P95",
                "Throughput",
            ]
        )
        # Make wide tables usable
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSortingEnabled(True)
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        left_layout.addWidget(self.results_table)

        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search verifications...")
        search_row.addWidget(QLabel("Filter"))
        search_row.addWidget(self.search_input)
        left_layout.addLayout(search_row)

        self.results_tree = QTreeWidget()
        self.results_tree.setHeaderLabels(["Tier", "Evaluator", "Metric", "Claim"])
        left_layout.addWidget(self.results_tree)

        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Verification Details"))
        self.details_view = QTextEdit()
        self.details_view.setReadOnly(True)
        right_layout.addWidget(self.details_view)
        splitter.addWidget(right_panel)

        splitter.setSizes([720, 600])
        layout.addWidget(splitter)

        self.results_tree.itemSelectionChanged.connect(self._on_tree_selection)
        self.search_input.textChanged.connect(self._filter_tree)

        return container

    def _build_insights_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        # Scatter controls (lets you explore AURC and Evidence Relevance)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Scatter X"))
        self.scatter_x_combo = QComboBox()
        self.scatter_x_combo.addItems([
            "P95 latency (ms)",
            "AURC (lower better)",
            "Evidence Relevance",
        ])
        controls.addWidget(self.scatter_x_combo)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Scatter Y"))
        self.scatter_y_combo = QComboBox()
        self.scatter_y_combo.addItems([
            "Accuracy (%)",
            "F1 (%)",
            "Safety (1-HPR)",
            "Evidence Relevance",
        ])
        controls.addWidget(self.scatter_y_combo)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.insights_figure = Figure(figsize=(9, 6), constrained_layout=True)
        self.insights_canvas = FigureCanvas(self.insights_figure)
        self.insights_toolbar = NavigationToolbar(self.insights_canvas, self)

        self.ax_radar = self.insights_figure.add_subplot(1, 2, 1, polar=True)
        self.ax_scatter = self.insights_figure.add_subplot(1, 2, 2)

        layout.addWidget(self.insights_toolbar)
        layout.addWidget(self.insights_canvas)

        # Update plots when axes change
        self.scatter_x_combo.currentIndexChanged.connect(self._update_insights_charts)
        self.scatter_y_combo.currentIndexChanged.connect(self._update_insights_charts)

        return container

    def _build_log_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)
        return container

    def _build_reports_tab(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)

        button_row = QHBoxLayout()
        self.reports_refresh_btn = QPushButton("Refresh")
        self.reports_save_btn = QPushButton("Save As...")
        self.reports_open_btn = QPushButton("Open Folder")
        button_row.addWidget(self.reports_refresh_btn)
        button_row.addWidget(self.reports_save_btn)
        button_row.addWidget(self.reports_open_btn)
        button_row.addStretch(1)

        self.reports_tree = QTreeWidget()
        self.reports_tree.setHeaderLabels(["File", "Size", "Modified"])

        layout.addLayout(button_row)
        layout.addWidget(self.reports_tree)

        self.reports_refresh_btn.clicked.connect(self._refresh_reports_list)
        self.reports_save_btn.clicked.connect(self._save_selected_report)
        self.reports_open_btn.clicked.connect(self._open_reports_folder)

        return container

    def _start_benchmark(self) -> None:
        if self._worker and self._worker.isRunning():
            return

        evaluators = [name for name, box in self.evaluator_checks.items() if box.isChecked()]
        if not evaluators:
            self._append_log("Select at least one evaluator.")
            return

        metrics = ["hallucination", "truthfulqa", "factscore", "latency"]
        if not self.use_all_metrics.isChecked():
            metrics = [self.metrics_combo.currentText()]

        overrides: dict[str, Any] = {
            "evaluators": evaluators,
            "metrics": metrics,
            "ohi_api_host": self.api_host.text().strip(),
            "ohi_api_port": self.api_port.text().strip(),
            "ohi_api_key": self.api_key.text().strip() or None,
            "output_dir": Path(self.output_dir.text().strip()),
            "concurrency": self.concurrency.value(),
            "warmup_requests": self.warmup.value(),
            "timeout_seconds": float(self.timeout.value()),
            "chart_dpi": self.chart_dpi.value(),
            "ohi_all_strategies": self.ohi_all_strategies.isChecked(),
            "cache_testing": self.cache_testing.isChecked(),
            "complete_mode": self.complete_mode.isChecked(),
            "complete_samples_per_dataset": self.complete_samples.value(),
        }
        
        # Validate COMPLETE mode constraints
        if self.complete_mode.isChecked():
            # Ensure all metrics are enabled in COMPLETE mode
            overrides["metrics"] = ["hallucination", "truthfulqa", "factscore", "latency"]
            self.use_all_metrics.setChecked(True)
            
            # Disable other special modes
            if self.ohi_all_strategies.isChecked() or self.cache_testing.isChecked():
                self._append_log("⚠ Disabling other special modes for COMPLETE mode")
                overrides["ohi_all_strategies"] = False
                overrides["cache_testing"] = False

        if self.dataset_path.text().strip():
            overrides["hallucination_dataset"] = Path(self.dataset_path.text().strip())

        # Store config for export
        self._current_config = ComparisonBenchmarkConfig.from_env()
        for key, value in overrides.items():
            if hasattr(self._current_config, key):
                setattr(self._current_config, key, value)
        
        self._worker = BenchmarkWorker(overrides)
        self._worker.stats_updated.connect(self._on_stats_update)
        self._worker.log_message.connect(self._append_log)
        self._worker.evaluator_completed.connect(self._on_evaluator_done)
        self._worker.verification_recorded.connect(self._on_verification_recorded)
        self._worker.report_updated.connect(self._on_report_updated)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self.status_label.setText("Running...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self._worker.start()

    def _stop_benchmark(self) -> None:
        if self._worker:
            self._append_log("Stop requested. Cancelling pending tasks...")
            self.status_label.setText("Stopping...")
            self._worker.request_stop()

    def _on_failed(self, message: str) -> None:
        self.status_label.setText("Failed")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self._append_log(message)

    def _on_finished(self, report: dict) -> None:
        status = report.get("status") if isinstance(report, dict) else None
        if status == "stopped":
            self.status_label.setText("Stopped")
            self._append_log("Benchmark stopped.")
        else:
            self.status_label.setText("Completed")
            self._append_log("Benchmark completed.")
            # Set progress bars to 100% on completion
            if self.overall_progress_bar.maximum() > 0:
                self.overall_progress_bar.setValue(self.overall_progress_bar.maximum())
            if self.current_progress_bar.maximum() > 0:
                self.current_progress_bar.setValue(self.current_progress_bar.maximum())
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self._refresh_reports_list()

    def _append_log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log_output.append(f"[{ts}] {message}")
    
    def _on_report_updated(self, report: ComparisonReport) -> None:
        """Store the latest partial report for export."""
        self._current_report = report

    def _on_stats_update(self, stats: dict) -> None:
        now = stats.get("timestamp", time.time())
        if now - self._last_plot_update < 0.5:
            return
        self._last_plot_update = now

        total = stats.get("current_total", 0)
        correct = stats.get("current_correct", 0)
        errors = stats.get("current_errors", 0)
        completed = stats.get("current_completed", 0)
        accuracy = (correct / completed) * 100 if completed else 0.0
        latencies = stats.get("latencies", [])
        latency = latencies[-1] if latencies else 0.0
        
        # Update progress bars
        total_evaluators = stats.get("total_evaluators", 0)
        completed_evaluators = stats.get("completed_evaluators", 0)
        current_evaluator = stats.get("current_evaluator", "")
        current_metric = stats.get("current_metric", "")
        
        # Overall progress
        if total_evaluators > 0:
            overall_percent = int((completed_evaluators / total_evaluators) * 100)
            self.overall_progress_bar.setMaximum(total_evaluators)
            self.overall_progress_bar.setValue(completed_evaluators)
            self.overall_progress_label.setText(
                f"Overall Progress: {completed_evaluators} / {total_evaluators} evaluators"
            )
        else:
            self.overall_progress_bar.setValue(0)
            self.overall_progress_label.setText("Overall Progress: 0 / 0 evaluators")
        
        # Current run progress
        if total > 0:
            current_percent = int((completed / total) * 100)
            self.current_progress_bar.setMaximum(total)
            self.current_progress_bar.setValue(completed)
            metric_label = f" - {current_metric}" if current_metric else ""
            evaluator_label = f"{current_evaluator}" if current_evaluator else "N/A"
            self.current_progress_label.setText(
                f"Current Run ({evaluator_label}{metric_label}): {completed} / {total} items"
            )
        else:
            self.current_progress_bar.setValue(0)
            self.current_progress_label.setText("Current Run: 0 / 0 items")

        elapsed = now - (self._time_series[0] if self._time_series else now)
        throughput = (completed / elapsed) if elapsed > 0 else 0.0

        self._time_series.append(now)
        self._latencies.append(latency)
        self._accuracy_series.append(accuracy)
        self._throughput_series.append(throughput)

        self._line_latency.set_data(self._x_values(), list(self._latencies))
        self._line_accuracy.set_data(self._x_values(), list(self._accuracy_series))
        self._line_throughput.set_data(self._x_values(), list(self._throughput_series))
        self._line_errors.set_data(self._x_values(), [errors] * len(self._time_series))

        for ax in [self.ax_latency, self.ax_accuracy, self.ax_throughput, self.ax_errors]:
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw_idle()

        current_eval = stats.get("current_evaluator")
        if current_eval:
            self._evaluator_meta[current_eval] = {
                "accuracy": accuracy / 100.0,
                "p95": latency,
                "throughput": throughput,
            }
            # Include all metrics for live radar display
            self._evaluator_summaries[current_eval] = {
                "Accuracy": accuracy / 100.0,
                "Precision": accuracy / 100.0,
                "Recall": accuracy / 100.0,
                "F1 Score": accuracy / 100.0,
                "Safety (1-HPR)": accuracy / 100.0,
                "TruthfulQA": accuracy / 100.0,
                "FActScore": accuracy / 100.0,
                "Speed (1/P95)": min(1.0, 1000.0 / latency) if latency > 0 else 0.0,
                "Selective (1/(1+AURC))": accuracy / 100.0,  # Proxy during live
                "Evidence Relevance": accuracy / 100.0,  # Proxy during live
            }
            self._update_insights_charts()

    def _x_values(self) -> list[float]:
        if not self._time_series:
            return []
        start = self._time_series[0]
        return [t - start for t in self._time_series]

    def _on_evaluator_done(self, name: str, metrics: dict) -> None:
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(name))
        self.results_table.setItem(row, 1, QTableWidgetItem(f"{metrics.get('accuracy', 0):.2%}"))
        self.results_table.setItem(row, 2, QTableWidgetItem(f"{metrics.get('f1', 0):.2%}"))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{metrics.get('hpr', 0):.2%}"))
        self.results_table.setItem(row, 4, QTableWidgetItem(f"{metrics.get('aurc', 0):.4f}"))
        self.results_table.setItem(row, 5, QTableWidgetItem(f"{metrics.get('evidence_relevance', 0):.3f}"))
        self.results_table.setItem(row, 6, QTableWidgetItem(f"{metrics.get('p50', 0):.0f}ms"))
        self.results_table.setItem(row, 7, QTableWidgetItem(f"{metrics.get('p95', 0):.0f}ms"))
        self.results_table.setItem(row, 8, QTableWidgetItem(f"{metrics.get('throughput', 0):.1f}/s"))

        # Right-align numeric columns
        for col in range(1, self.results_table.columnCount()):
            item = self.results_table.item(row, col)
            if item is not None:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

        summary_scores = metrics.get("summary_scores")
        if isinstance(summary_scores, dict):
            # Enrich radar with derived metrics
            enriched = dict(summary_scores)
            # AURC is lower-better; map to (0,1] for radar
            aurc = float(metrics.get('aurc', 0.0))
            enriched['Selective (1/(1+AURC))'] = 1.0 / (1.0 + max(0.0, aurc))
            enriched['Evidence Relevance'] = float(metrics.get('evidence_relevance', 0.0))
            self._evaluator_summaries[name] = enriched
            self._evaluator_meta[name] = {
                'accuracy': float(metrics.get('accuracy', 0.0)),
                'f1': float(metrics.get('f1', 0.0)),
                'safety': 1.0 - float(metrics.get('hpr', 0.0)),
                'p95': float(metrics.get('p95', 0.0)),
                'throughput': float(metrics.get('throughput', 0.0)),
                'aurc': float(metrics.get('aurc', 0.0)),
                'evidence_relevance': float(metrics.get('evidence_relevance', 0.0)),
            }
            self._update_insights_charts()

    def _on_verification_recorded(self, payload: dict) -> None:
        tier = str(payload.get("tier") or "default")
        evaluator = str(payload.get("evaluator") or "unknown")
        metric = str(payload.get("metric") or "metric")
        claim = str(payload.get("claim") or "")

        tier_item = self._tree_index.get((tier, "", ""))
        if tier_item is None:
            tier_item = QTreeWidgetItem([tier, "", "", ""])
            self.results_tree.addTopLevelItem(tier_item)
            self._tree_index[(tier, "", "")] = tier_item

        eval_item = self._tree_index.get((tier, evaluator, ""))
        if eval_item is None:
            eval_item = QTreeWidgetItem([tier, evaluator, "", ""])
            tier_item.addChild(eval_item)
            self._tree_index[(tier, evaluator, "")] = eval_item

        metric_item = self._tree_index.get((tier, evaluator, metric))
        if metric_item is None:
            metric_item = QTreeWidgetItem([tier, evaluator, metric, ""])
            eval_item.addChild(metric_item)
            self._tree_index[(tier, evaluator, metric)] = metric_item

        case_item = QTreeWidgetItem([tier, evaluator, metric, claim[:120]])
        case_item.setData(0, Qt.UserRole, payload)
        metric_item.addChild(case_item)

    def _on_tree_selection(self) -> None:
        items = self.results_tree.selectedItems()
        if not items:
            return
        item = items[0]
        payload = item.data(0, Qt.UserRole)
        if not isinstance(payload, dict):
            return
        self.details_view.setText(self._format_verification(payload))

    def _format_verification(self, payload: dict) -> str:
        evidence = payload.get("evidence") or []
        meta = payload.get("meta") or {}
        lines = [
            f"Evaluator: {payload.get('evaluator')}",
            f"Tier: {payload.get('tier')}",
            f"Metric: {payload.get('metric')}",
            "",
            f"Claim: {payload.get('claim')}",
            "",
            f"Verdict: {payload.get('verdict')}",
            f"Expected: {payload.get('expected')}",
            f"Predicted: {payload.get('predicted')}",
            f"Trust Score: {payload.get('trust_score')}",
            f"Latency: {payload.get('latency_ms')} ms",
            f"Error: {payload.get('error')}",
            "",
            "Meta:",
        ]
        for key, value in meta.items():
            lines.append(f"  - {key}: {value}")
        lines.append("")
        lines.append("Evidence:")
        if not evidence:
            lines.append("  (none)")
        else:
            for idx, ev in enumerate(evidence, 1):
                lines.append(f"  {idx}. {ev.get('source')}")
                lines.append(f"     score: {ev.get('similarity_score')}")
                lines.append(f"     {ev.get('text')}")
        return "\n".join(lines)

    def _filter_tree(self, text: str) -> None:
        query = text.strip().lower()
        root_count = self.results_tree.topLevelItemCount()
        for i in range(root_count):
            tier_item = self.results_tree.topLevelItem(i)
            self._filter_tree_item(tier_item, query)

    def _filter_tree_item(self, item: QTreeWidgetItem, query: str) -> bool:
        matches = False
        payload = item.data(0, Qt.UserRole)
        if isinstance(payload, dict):
            haystack = " ".join(
                str(payload.get(k) or "")
                for k in ("claim", "evaluator", "metric", "tier", "verdict", "error")
            ).lower()
            matches = query in haystack if query else True

        child_visible = False
        for idx in range(item.childCount()):
            if self._filter_tree_item(item.child(idx), query):
                child_visible = True

        visible = matches or child_visible
        item.setHidden(not visible)
        return visible

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir.setText(path)

    def _choose_dataset(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if file_path:
            self.dataset_path.setText(file_path)

    def _refresh_reports_list(self) -> None:
        self.reports_tree.clear()
        output_dir = Path(self.output_dir.text().strip())
        if not output_dir.exists():
            return

        files: list[Path] = []
        patterns = ["*.json", "*.csv", "*.md", "*.html", "*.png", "*.svg", "*.pdf"]
        for pattern in patterns:
            files.extend(output_dir.glob(pattern))
        charts_dir = output_dir / "charts"
        if charts_dir.exists():
            for pattern in ("*.png", "*.svg", "*.pdf"):
                files.extend(charts_dir.glob(pattern))

        files = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)

        for path in files:
            try:
                size_kb = path.stat().st_size / 1024
                mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
                item = QTreeWidgetItem([path.name, f"{size_kb:.1f} KB", mtime])
                item.setData(0, Qt.UserRole, str(path))
                self.reports_tree.addTopLevelItem(item)
            except OSError:
                continue

    def _save_selected_report(self) -> None:
        items = self.reports_tree.selectedItems()
        if not items:
            return
        source = items[0].data(0, Qt.UserRole)
        if not source:
            return
        source_path = Path(source)
        target, _ = QFileDialog.getSaveFileName(self, "Save Report As", source_path.name)
        if target:
            shutil.copyfile(source_path, target)

    def _open_reports_folder(self) -> None:
        output_dir = Path(self.output_dir.text().strip())
        if output_dir.exists():
            os.startfile(output_dir)  # type: ignore[attr-defined]

    def _update_insights_charts(self) -> None:
        self.ax_radar.clear()
        self.ax_scatter.clear()

        if not self._evaluator_summaries:
            self.insights_canvas.draw_idle()
            return

        radar_keys = [
            'Accuracy',
            'F1 Score',
            'Safety (1-HPR)',
            'TruthfulQA',
            'FActScore',
            'Speed (1/P95)',
            'Selective (1/(1+AURC))',
            'Evidence Relevance',
        ]

        angles = [i / len(radar_keys) * 2 * 3.14159265 for i in range(len(radar_keys))]
        angles += angles[:1]

        for name, scores in self._evaluator_summaries.items():
            values = [float(scores.get(k, 0.0)) for k in radar_keys]
            values += values[:1]
            self.ax_radar.plot(angles, values, label=name, linewidth=1.8)
            self.ax_radar.fill(angles, values, alpha=0.08)

        self.ax_radar.set_xticks(angles[:-1])
        self.ax_radar.set_xticklabels(radar_keys, fontsize=9)
        self.ax_radar.set_yticklabels([])
        self.ax_radar.set_title("Evaluator Radar", pad=16)
        self.ax_radar.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=8)

        # Scatter: user-selectable axes (includes AURC/BEIR/RAGAS/ALCE signals)
        x_choice = self.scatter_x_combo.currentText() if hasattr(self, 'scatter_x_combo') else 'P95 latency (ms)'
        y_choice = self.scatter_y_combo.currentText() if hasattr(self, 'scatter_y_combo') else 'Accuracy (%)'

        def x_val(meta: dict) -> float:
            if x_choice.startswith('P95'):
                return float(meta.get('p95', 0.0))
            if x_choice.startswith('AURC'):
                return float(meta.get('aurc', 0.0))
            if x_choice.startswith('Evidence'):
                return float(meta.get('evidence_relevance', 0.0))
            return float(meta.get('p95', 0.0))

        def y_val(meta: dict) -> float:
            if y_choice.startswith('Accuracy'):
                return float(meta.get('accuracy', 0.0)) * 100
            if y_choice.startswith('F1'):
                return float(meta.get('f1', 0.0)) * 100
            if y_choice.startswith('Safety'):
                return float(meta.get('safety', 0.0)) * 100
            if y_choice.startswith('Evidence'):
                return float(meta.get('evidence_relevance', 0.0)) * 100
            return float(meta.get('accuracy', 0.0)) * 100

        for name, meta in self._evaluator_meta.items():
            x = x_val(meta)
            y = y_val(meta)
            size = max(40.0, float(meta.get('throughput', 0.0)) * 12)
            self.ax_scatter.scatter(x, y, s=size, alpha=0.7)
            self.ax_scatter.text(x, y, name, fontsize=8)

        self.ax_scatter.set_title(f"{x_choice} vs {y_choice}")
        self.ax_scatter.set_xlabel(x_choice)
        self.ax_scatter.set_ylabel(y_choice)
        self.ax_scatter.grid(True, alpha=0.2)

        self.insights_canvas.draw_idle()
    
    def _export_current_state(self) -> None:
        """Export the current benchmark state to files."""
        if not self._current_report:
            self._append_log("No report data available to export.")
            return
        
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"partial_{timestamp}"
            
            config = self._current_config or ComparisonBenchmarkConfig.from_env()
            output_dir = config.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Export JSON
            json_path = output_dir / f"{filename}_report.json"
            json_payload = self._current_report.to_dict()
            json_payload["status"] = "partial"
            json_payload["note"] = "This is a partial report from an incomplete benchmark run"
            json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
            self._append_log(f"✓ Exported JSON: {json_path.name}")
            
            # Export charts if we have any evaluator data
            if self._current_report.evaluators:
                charts_reporter = ChartsReporter(output_dir, dpi=config.chart_dpi)
                try:
                    # Generate all individual comparison charts (not just dashboard)
                    chart_files = charts_reporter.generate_comparison_charts(
                        self._current_report,
                        prefix=f"{filename}_",
                        consolidated=False
                    )
                    self._append_log(f"✓ Exported {len(chart_files)} charts:")
                    # Display list of all generated chart filenames
                    for chart_path in chart_files:
                        self._append_log(f"  - {chart_path.name}")
                except Exception as chart_error:
                    self._append_log(f"⚠ Chart generation failed: {chart_error}")
            
            self._append_log(f"✓ Current state exported to: {output_dir}")
            self._refresh_reports_list()
            
        except Exception as exc:
            self._append_log(f"✗ Export failed: {type(exc).__name__}: {exc}")


def main() -> None:
    app = QApplication(sys.argv)
    window = BenchmarkWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
