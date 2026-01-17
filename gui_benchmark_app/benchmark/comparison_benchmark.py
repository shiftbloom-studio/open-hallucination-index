"""
Comparison Benchmark Metrics
=============================

Extended metrics for multi-evaluator benchmark comparison:
- Hallucination Detection Metrics
- TruthfulQA Metrics (MC1, MC2, Generation)
- FActScore Metrics (Atomic Fact Precision)
- Latency Metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import math
import re

import numpy as np


# Text truncation limits for TF-IDF vectorization (to bound memory and computation)
MAX_TEXT_LENGTH = 2000  # For general contexts and responses
MAX_SENTENCE_LENGTH = 500  # For sentence-level faithfulness checks

# Context/evidence limits per sample
MAX_CONTEXTS_PER_SAMPLE = 8  # Balance between coverage and performance

# TF-IDF vectorizer feature limits
TFIDF_MAX_FEATURES_SMALL = 6000  # For per-response ALCE-style grounding
# For global RAGAS-style corpus (questions + answers + contexts + ground truths)
TFIDF_MAX_FEATURES_LARGE = 12000


@dataclass
class HallucinationMetrics:
    """
    Metrics for hallucination detection evaluation.

    Key safety metric: hallucination_pass_rate (lower is better)
    - Rate at which hallucinations are incorrectly marked as factual
    """

    total: int = 0
    correct: int = 0

    # Confusion matrix
    true_positives: int = 0  # Fact correctly identified
    true_negatives: int = 0  # Hallucination correctly identified
    false_positives: int = 0  # Hallucination marked as fact (DANGEROUS)
    false_negatives: int = 0  # Fact marked as hallucination

    # Optional per-sample signals (kept lightweight; used for AURC/BEIR/ALCE/RAGAS-style eval)
    # NOTE: These streams are purely derived from already-produced results;
    # they do not change inputs.
    confidence_scores: list[float] = field(default_factory=list, repr=False)
    correct_flags: list[bool] = field(default_factory=list, repr=False)
    retrieved_sources: list[list[str]] = field(default_factory=list, repr=False)
    relevant_sources: list[list[str]] = field(default_factory=list, repr=False)
    response_texts: list[str] = field(default_factory=list, repr=False)
    evidence_counts: list[int] = field(default_factory=list, repr=False)

    # RAG evaluation signals (optional). If present, we can compute RAGAS-style proxy metrics.
    rag_questions: list[str] = field(default_factory=list, repr=False)
    rag_answers: list[str] = field(default_factory=list, repr=False)
    rag_contexts: list[list[str]] = field(default_factory=list, repr=False)
    rag_ground_truths: list[str | None] = field(default_factory=list, repr=False)

    # -------------------------
    # Incremental update helper
    # -------------------------

    def add_sample(
        self,
        *,
        expected_is_fact: bool,
        predicted_is_fact: bool,
        confidence: float | None = None,
        retrieved_sources: list[str] | None = None,
        relevant_sources: list[str] | None = None,
        response_text: str | None = None,
        evidence_count: int | None = None,
        # RAG signals
        question: str | None = None,
        answer: str | None = None,
        contexts: list[str] | None = None,
        ground_truth: str | None = None,
    ) -> None:
        """Update confusion matrix + optional per-sample signals."""
        self.total += 1

        is_correct = predicted_is_fact == expected_is_fact
        if is_correct:
            self.correct += 1

        # Confusion matrix (positive class = "fact")
        if predicted_is_fact and expected_is_fact:
            self.true_positives += 1
        elif (not predicted_is_fact) and (not expected_is_fact):
            self.true_negatives += 1
        elif predicted_is_fact and (not expected_is_fact):
            self.false_positives += 1
        else:
            self.false_negatives += 1

        # Optional streams
        # NOTE: For AURC, only *ranking* matters; do not clamp, just sanitize.
        if confidence is None:
            conf = 0.0
        else:
            try:
                conf = float(confidence)
            except Exception:
                conf = 0.0
            if not np.isfinite(conf):
                conf = 0.0
        self.confidence_scores.append(conf)
        self.correct_flags.append(bool(is_correct))

        if retrieved_sources is not None and relevant_sources is not None:
            self.retrieved_sources.append(list(retrieved_sources))
            self.relevant_sources.append(list(relevant_sources))

        if response_text is not None:
            self.response_texts.append(str(response_text))
        if evidence_count is not None:
            self.evidence_counts.append(int(evidence_count))

        # RAG signals
        if question is not None and answer is not None and contexts is not None:
            self.rag_questions.append(str(question))
            self.rag_answers.append(str(answer))
            self.rag_contexts.append([str(c) for c in contexts])
            # ground truth is optional; use None to preserve alignment and distinguish
            # between "no ground truth provided" vs "empty string ground truth"
            self.rag_ground_truths.append(str(ground_truth) if ground_truth else None)

    @property
    def accuracy(self) -> float:
        """Overall accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP)."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 Score: Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def specificity(self) -> float:
        """Specificity (TNR): TN / (TN + FP)."""
        denom = self.true_negatives + self.false_positives
        return self.true_negatives / denom if denom > 0 else 0.0

    @property
    def hallucination_pass_rate(self) -> float:
        """
        CRITICAL SAFETY METRIC: Rate of hallucinations marked as factual.

        Lower is better. FP / (FP + TN)
        """
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def mcc(self) -> float:
        """Matthews Correlation Coefficient."""
        num = (
            self.true_positives * self.true_negatives - self.false_positives * self.false_negatives
        )
        denom = math.sqrt(
            (self.true_positives + self.false_positives)
            * (self.true_positives + self.false_negatives)
            * (self.true_negatives + self.false_positives)
            * (self.true_negatives + self.false_negatives)
        )
        return num / denom if denom > 0 else 0.0

    # -------------------------
    # Selective prediction (AURC)
    # -------------------------

    @property
    def aurc(self) -> float:
        """Area Under the Risk-Coverage curve using `confidence_scores` as confidence."""
        if not self.confidence_scores or not self.correct_flags:
            return 0.0
        return compute_aurc(self.confidence_scores, self.correct_flags)

    @property
    def eaurc(self) -> float:
        """Excess-AURC (AURC - optimal AURC for same accuracy)."""
        if not self.confidence_scores or not self.correct_flags:
            return 0.0
        return compute_eaurc(self.confidence_scores, self.correct_flags)

    # -------------------------
    # BEIR-style retrieval metrics (source match)
    # -------------------------

    def retrieval_metrics(
        self, *, ks: tuple[int, ...] = (1, 3, 5, 10, 50, 100)
    ) -> dict[str, float]:
        """Compute BEIR-style metrics from stored retrieved/relevant sources."""
        if not self.retrieved_sources or not self.relevant_sources:
            return {}
        return compute_retrieval_metrics(self.retrieved_sources, self.relevant_sources, ks=ks)

    # -------------------------
    # ALCE-style citation metrics (if response texts include citation markers)
    # -------------------------

    def alce_metrics(self) -> dict[str, float]:
        if not self.response_texts:
            return {}
        return compute_alce_style_metrics(
            self.response_texts,
            self.evidence_counts or None,
            contexts_by_sample=self.rag_contexts if self.rag_contexts else None,
        )

    # -------------------------
    # RAGAS-style proxy metrics (no external deps)
    # -------------------------

    def ragas_proxy_metrics(self) -> dict[str, float]:
        """Compute lightweight, dependency-free proxies inspired by common RAGAS metrics."""
        if not (self.rag_questions and self.rag_answers and self.rag_contexts):
            return {}
        return compute_ragas_proxy_metrics(
            questions=self.rag_questions,
            answers=self.rag_answers,
            contexts=self.rag_contexts,
            ground_truths=self.rag_ground_truths if self.rag_ground_truths else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        out: dict[str, Any] = {
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "specificity": round(self.specificity, 4),
            "hallucination_pass_rate": round(self.hallucination_pass_rate, 4),
            "mcc": round(self.mcc, 4),
            "confusion_matrix": {
                "tp": self.true_positives,
                "tn": self.true_negatives,
                "fp": self.false_positives,
                "fn": self.false_negatives,
            },
        }

        # Optional metrics (only included if we have the necessary signals)
        if self.confidence_scores and self.correct_flags:
            out["aurc"] = round(self.aurc, 6)
            out["eaurc"] = round(self.eaurc, 6)

        retrieval = self.retrieval_metrics()
        if retrieval:
            out["retrieval"] = {k: round(v, 4) for k, v in retrieval.items()}

        alce = self.alce_metrics()
        if alce:
            out["alce_style"] = {k: round(v, 4) for k, v in alce.items()}

        ragas_proxy = self.ragas_proxy_metrics()
        if ragas_proxy:
            out["ragas_proxy"] = {k: round(v, 4) for k, v in ragas_proxy.items()}

        return out


# =============================================================================
# Extra metric helpers (AURC, BEIR-style IR, ALCE-style citations, RAGAS adapter)
# =============================================================================


def compute_aurc(confidence_scores: list[float], correct_flags: list[bool]) -> float:
    """
    Compute AURC (Area Under Risk-Coverage curve) from confidence ranking.

    AURC measures selective prediction performance by computing the area under
    the curve of risk (error rate) vs. coverage (fraction of predictions retained)
    when samples are ranked by confidence.

    Notes
    -----
    - ``confidence_scores`` and ``correct_flags`` are truncated to the same length.
    - NaN confidence scores are treated as the lowest possible confidence (mapped to
      ``-np.inf``) so that, when sorted in descending order by confidence, they appear
      at the end of the ranking. This keeps all samples in the AURC calculation while
      conservatively assuming NaN scores are least reliable.
    - Positive infinity values are clamped to a large finite float so that they sort
      before all finite scores without breaking downstream arithmetic.
    """
    n = min(len(confidence_scores), len(correct_flags))
    if n <= 1:
        return 0.0

    scores = np.asarray(confidence_scores[:n], dtype=float)
    correct = np.asarray(correct_flags[:n], dtype=bool)
    # Normalize NaN/±inf scores for ranking:
    # - NaNs and -inf → -np.inf so they are treated as lowest confidence (sorted last)
    # - +inf → large finite max so it ranks highest while remaining numerically stable
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)

    order = np.argsort(scores)[::-1]
    correct_sorted = correct[order].astype(np.int64)
    correct_prefix = np.cumsum(correct_sorted)
    ks = np.arange(1, n + 1, dtype=float)
    coverage = ks / float(n)
    risk = 1.0 - (correct_prefix / ks)

    # Include (0,0)
    x = np.concatenate(([0.0], coverage))
    y = np.concatenate(([0.0], risk))
    return float(np.trapezoid(y, x))


def compute_optimal_aurc(n: int, correct_count: int) -> float:
    """Optimal AURC for a given (n, correct_count) assuming perfect ordering."""
    if n <= 1:
        return 0.0
    correct_count = max(0, min(n, int(correct_count)))

    ks = np.arange(1, n + 1, dtype=float)
    coverage = ks / float(n)
    # risk stays 0 until we start including incorrect predictions
    incorrect_included = np.maximum(0.0, ks - float(correct_count))
    risk = incorrect_included / ks

    x = np.concatenate(([0.0], coverage))
    y = np.concatenate(([0.0], risk))
    return float(np.trapezoid(y, x))


def compute_eaurc(confidence_scores: list[float], correct_flags: list[bool]) -> float:
    """Compute Excess-AURC: AURC - optimal AURC for same accuracy."""
    n = min(len(confidence_scores), len(correct_flags))
    if n <= 1:
        return 0.0
    aurc = compute_aurc(confidence_scores, correct_flags)
    correct_count = int(np.sum(np.asarray(correct_flags[:n], dtype=bool)))
    opt = compute_optimal_aurc(n, correct_count)
    return float(aurc - opt)


def _norm_id(x: str) -> str:
    x = (x or "").strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x


def compute_retrieval_metrics(
    retrieved_sources: list[list[str]],
    relevant_sources: list[list[str]],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10, 50, 100),
) -> dict[str, float]:
    """Compute BEIR-style retrieval metrics from retrieved vs relevant IDs."""
    if not retrieved_sources or not relevant_sources:
        return {}
    n = min(len(retrieved_sources), len(relevant_sources))

    ks_sorted = tuple(sorted(set(int(k) for k in ks if k > 0)))
    q = 0

    recall_sum = {k: 0.0 for k in ks_sorted}
    ndcg_sum = {k: 0.0 for k in ks_sorted}
    map_sum = {k: 0.0 for k in ks_sorted}
    precision_sum = {k: 0.0 for k in ks_sorted}
    mrr_sum = {k: 0.0 for k in ks_sorted}
    hit_rate_sum = {k: 0.0 for k in ks_sorted}

    for retrieved, relevants in zip(retrieved_sources[:n], relevant_sources[:n]):
        rel_set = {_norm_id(r) for r in relevants if str(r).strip()}
        if not rel_set:
            continue
        q += 1

        retrieved_norm = [_norm_id(r) for r in (retrieved or []) if str(r).strip()]

        for k in ks_sorted:
            topk = retrieved_norm[:k]
            hits = [1 if doc_id in rel_set else 0 for doc_id in topk]

            # Recall@k
            recall_sum[k] += sum(hits) / len(rel_set)

            # Precision@k (k is guaranteed > 0 by validation on line 375)
            precision_sum[k] += sum(hits) / k

            # HitRate@k ("success" if any relevant appears in top-k)
            hit_rate_sum[k] += 1.0 if any(hits) else 0.0

            # MRR@k
            rr = 0.0
            for i, h in enumerate(hits, start=1):
                if h:
                    rr = 1.0 / i
                    break
            mrr_sum[k] += rr

            # nDCG@k (binary relevance)
            dcg = 0.0
            for i, h in enumerate(hits, start=1):
                if h:
                    dcg += 1.0 / math.log2(i + 1)
            ideal_hits = min(k, len(rel_set))
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
            ndcg_sum[k] += (dcg / idcg) if idcg > 0 else 0.0

            # MAP@k (Mean Average Precision)
            # For each relevant document retrieved, compute precision at that rank,
            # then average over all relevant documents. The denominator is the
            # minimum of (total relevant documents, k), not the number of relevant
            # documents retrieved, which is standard for MAP calculation.
            num_rel = 0
            ap = 0.0
            for i, h in enumerate(hits, start=1):
                if h:
                    num_rel += 1
                    ap += num_rel / i
            denom = min(len(rel_set), k)
            map_sum[k] += (ap / denom) if denom > 0 else 0.0

    if q == 0:
        return {}

    out: dict[str, float] = {}
    for k in ks_sorted:
        out[f"recall@{k}"] = recall_sum[k] / q
        out[f"precision@{k}"] = precision_sum[k] / q
        out[f"ndcg@{k}"] = ndcg_sum[k] / q
        out[f"map@{k}"] = map_sum[k] / q
        out[f"mrr@{k}"] = mrr_sum[k] / q
        out[f"hit_rate@{k}"] = hit_rate_sum[k] / q
    out["queries"] = float(q)
    return out


_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def compute_alce_style_metrics(
    responses: list[str],
    evidence_counts: list[int] | None,
    *,
    contexts_by_sample: list[list[str]] | None = None,
) -> dict[str, float]:
    """ALCE-inspired citation quality signals (lightweight, no external deps)."""
    total = 0
    with_citations = 0
    total_citation_markers = 0
    total_valid_citations = 0
    total_tokens = 0
    total_sentences = 0
    cited_sentences = 0

    grounding_scores: list[float] = []

    for idx, text in enumerate(responses):
        if not text:
            continue
        total += 1
        tokens = text.split()
        total_tokens += len(tokens)

        # sentence split (simple + fast)
        sentences = [s.strip() for s in re.split(r"[\.!\?\n]+", text) if s.strip()]
        total_sentences += len(sentences)

        matches = list(_CITATION_RE.finditer(text))
        if matches:
            with_citations += 1
            total_citation_markers += len(matches)

        # sentence coverage
        if sentences:
            for s in sentences:
                if _CITATION_RE.search(s):
                    cited_sentences += 1

        # validity check (citation indexes within evidence length)
        if evidence_counts is not None and idx < len(evidence_counts):
            ev_n = int(evidence_counts[idx])
            if ev_n > 0:
                for m in matches:
                    raw = m.group(1)
                    for part in raw.split(","):
                        part = part.strip()
                        if part.isdigit():
                            n = int(part)
                            if 1 <= n <= ev_n:
                                total_valid_citations += 1

        # optional grounding proxy: cited sentence should be similar to cited contexts
        if contexts_by_sample is not None and idx < len(contexts_by_sample):
            ctx = contexts_by_sample[idx] or []
            if ctx and sentences:
                # TF-IDF vectorization per response.
                # NOTE: This could be optimized by batching all responses/contexts upfront
                # (similar to compute_ragas_proxy_metrics), but the current per-response
                # approach is simpler and handles variable contexts per sample more naturally.
                # For typical benchmark sizes (100s-1000s of responses), the performance
                # difference is expected to be negligible given the simplicity of the
                # TF-IDF implementation. If benchmarks show significant overhead, consider
                # switching to a batched approach similar to compute_ragas_proxy_metrics.
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity

                if ctx:
                    corpus = list(dict.fromkeys([c for c in ctx if c]))  # de-dup preserve order
                    corpus = [c[:MAX_TEXT_LENGTH] for c in corpus]
                    vect = TfidfVectorizer(
                        stop_words="english", max_features=TFIDF_MAX_FEATURES_SMALL
                    )
                    try:
                        X = vect.fit_transform(corpus + sentences)
                        X_ctx = X[: len(corpus)]
                        X_sent = X[len(corpus) :]
                        sim = cosine_similarity(X_sent, X_ctx)
                        for s_i, sent in enumerate(sentences):
                            if not _CITATION_RE.search(sent):
                                continue
                            # If the sentence has citations, require it to be grounded
                            # in *some* context.
                            grounding_scores.append(float(np.max(sim[s_i])))
                    except Exception:
                        # Vectorization can fail on degenerate inputs (empty strings, etc.).
                        # Silently skip this sample's grounding computation.
                        pass

    if total == 0:
        return {}

    citation_rate = with_citations / total
    citation_density = (total_citation_markers / total_tokens * 100.0) if total_tokens > 0 else 0.0
    sentence_coverage = (cited_sentences / total_sentences) if total_sentences > 0 else 0.0
    valid_ratio = total_valid_citations / max(1, total_citation_markers)

    out = {
        "citation_rate": citation_rate,
        "citation_density_per_100_tokens": citation_density,
        "sentence_citation_coverage": sentence_coverage,
        "valid_citation_ratio": valid_ratio,
        "responses": float(total),
    }

    if grounding_scores:
        out["citation_grounding_proxy"] = float(np.mean(grounding_scores))

    return out


# -------------------------
# RAGAS-style proxy metrics
# -------------------------


def _iter_flat_texts(items: Iterable[str]) -> list[str]:
    return [str(x) for x in items if str(x).strip()]


def compute_ragas_proxy_metrics(
    *,
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str | None] | None = None,
    # similarity thresholds are tuned for TF-IDF cosine values
    faithfulness_threshold: float = 0.20,
    context_relevance_threshold: float = 0.10,
) -> dict[str, float]:
    """Lightweight approximations of common RAG evaluation metrics.

    These are *proxies* meant for fast regression testing when LLM-based evaluators
    (like RAGAS proper) aren't available. They are deterministic and cheap.

    **Requirements:**
    Requires scikit-learn for TF-IDF vectorization and cosine similarity calculations.
    """
    n = min(len(questions), len(answers), len(contexts))
    if n == 0:
        return {}

    gt = ground_truths if ground_truths is not None else [None] * n
    if len(gt) < n:
        gt = list(gt) + [None] * (n - len(gt))

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Build a global vectorizer for stability across samples.
    corpus: list[str] = []
    for i in range(n):
        corpus.append(questions[i])
        corpus.append(answers[i])
        corpus.extend(_iter_flat_texts(contexts[i])[:MAX_CONTEXTS_PER_SAMPLE])
        if gt[i]:
            corpus.append(gt[i])

    corpus = [c[:MAX_TEXT_LENGTH] for c in corpus if c and c.strip()]
    if len(corpus) < 5:
        return {}

    vect = TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES_LARGE)
    X = vect.fit_transform(corpus)

    # Index mapping back into corpus
    idx = 0
    q_idx: list[int] = []
    a_idx: list[int] = []
    ctx_idx: list[list[int]] = []
    gt_idx: list[int | None] = []
    for i in range(n):
        q_idx.append(idx)
        idx += 1
        a_idx.append(idx)
        idx += 1
        ctx_i: list[int] = []
        for c in _iter_flat_texts(contexts[i])[:MAX_CONTEXTS_PER_SAMPLE]:
            ctx_i.append(idx)
            idx += 1
        ctx_idx.append(ctx_i)
        if gt[i]:
            gt_idx.append(idx)
            idx += 1
        else:
            gt_idx.append(None)

    # Per-sample computations
    answer_relevancies: list[float] = []
    context_precisions: list[float] = []
    context_recalls: list[float] = []
    faithfulnesses: list[float] = []
    context_utils: list[float] = []

    for i in range(n):
        qi = q_idx[i]
        ai = a_idx[i]
        ctxi = ctx_idx[i]

        if not ctxi:
            continue

        # Answer relevancy (Q vs A)
        answer_relevancies.append(float(cosine_similarity(X[qi], X[ai])[0, 0]))

        # Context precision proxy: fraction of contexts relevant to the question
        sims_q_ctx = cosine_similarity(X[qi], X[ctxi])[0]
        context_precisions.append(float(np.mean(sims_q_ctx >= context_relevance_threshold)))

        # Context recall proxy: does the retrieved context contain the ground-truth wording?
        gti = gt_idx[i]
        if gti is not None:
            sim_gt_ctx = cosine_similarity(X[gti], X[ctxi])[0]
            context_recalls.append(float(np.max(sim_gt_ctx)))

        # Context utilization: similarity of answer to *any* context
        sims_a_ctx = cosine_similarity(X[ai], X[ctxi])[0]
        context_utils.append(float(np.max(sims_a_ctx)))

        # Faithfulness proxy: sentence-level support in retrieved contexts
        ans = answers[i] or ""
        sentences = [s.strip() for s in re.split(r"[\.!\?\n]+", ans) if s.strip()]
        if sentences:
            # Build sentence vectors using the already-fitted vectorizer.
            # Truncate each sentence to 500 characters to bound vectorizer input and
            # focus faithfulness on local claims; this differs from general text
            # truncation (2000 chars) to emphasize sentence-level analysis.
            S = vect.transform([s[:MAX_SENTENCE_LENGTH] for s in sentences])
            sim_sent_ctx = cosine_similarity(S, X[ctxi])
            supported = np.max(sim_sent_ctx, axis=1) >= faithfulness_threshold
            faithfulnesses.append(float(np.mean(supported)))

    if not answer_relevancies:
        return {}

    out: dict[str, float] = {
        "answer_relevancy": float(np.mean(answer_relevancies)),
        "context_precision": float(np.mean(context_precisions)) if context_precisions else 0.0,
        "context_utilization": float(np.mean(context_utils)) if context_utils else 0.0,
        "faithfulness": float(np.mean(faithfulnesses)) if faithfulnesses else 0.0,
        "samples": float(len(answer_relevancies)),
    }
    if context_recalls:
        out["context_recall"] = float(np.mean(context_recalls))
    return out


@dataclass
class TruthfulQAMetrics:
    """
    Metrics for TruthfulQA evaluation.

    Measures truthfulness in adversarial question answering.
    """

    total_questions: int = 0
    correct_predictions: int = 0

    # Detailed breakdown
    mc1_correct: int = 0  # Single correct answer
    mc1_total: int = 0
    mc2_correct: int = 0  # Multiple correct answers
    mc2_total: int = 0

    # Category-wise results
    category_results: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Overall accuracy on TruthfulQA."""
        return self.correct_predictions / self.total_questions if self.total_questions > 0 else 0.0

    @property
    def mc1_accuracy(self) -> float:
        """MC1 accuracy (single correct answer)."""
        return self.mc1_correct / self.mc1_total if self.mc1_total > 0 else 0.0

    @property
    def mc2_accuracy(self) -> float:
        """MC2 accuracy (multiple correct answers)."""
        return self.mc2_correct / self.mc2_total if self.mc2_total > 0 else 0.0

    def get_category_accuracy(self, category: str) -> float:
        """Get accuracy for a specific category."""
        if category not in self.category_results:
            return 0.0
        result = self.category_results[category]
        total = result.get("total", 0)
        correct = result.get("correct", 0)
        return correct / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_questions": self.total_questions,
            "accuracy": round(self.accuracy, 4),
            "mc1_accuracy": round(self.mc1_accuracy, 4),
            "mc2_accuracy": round(self.mc2_accuracy, 4),
            "category_results": {
                cat: {
                    "accuracy": round(self.get_category_accuracy(cat), 4),
                    **result,
                }
                for cat, result in self.category_results.items()
            },
        }


@dataclass
class FActScoreMetrics:
    """
    Metrics for FActScore evaluation.

    FActScore = (# supported facts) / (# total facts)
    Measures atomic fact precision in longer-form text.
    """

    total_texts: int = 0
    total_facts: int = 0
    supported_facts: int = 0

    # Per-text scores
    scores: list[float] = field(default_factory=list)
    facts_per_text: list[int] = field(default_factory=list)

    @property
    def factscore(self) -> float:
        """Overall FActScore (atomic fact precision)."""
        return self.supported_facts / self.total_facts if self.total_facts > 0 else 0.0

    @property
    def avg_factscore(self) -> float:
        """Average FActScore across all texts."""
        return float(np.mean(self.scores)) if self.scores else 0.0

    @property
    def std_factscore(self) -> float:
        """Standard deviation of FActScores."""
        return float(np.std(self.scores)) if len(self.scores) > 1 else 0.0

    @property
    def avg_facts_per_text(self) -> float:
        """Average atomic facts extracted per text."""
        return float(np.mean(self.facts_per_text)) if self.facts_per_text else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_texts": self.total_texts,
            "total_facts": self.total_facts,
            "supported_facts": self.supported_facts,
            "factscore": round(self.factscore, 4),
            "avg_factscore": round(self.avg_factscore, 4),
            "std_factscore": round(self.std_factscore, 4),
            "avg_facts_per_text": round(self.avg_facts_per_text, 2),
        }


@dataclass
class LatencyMetrics:
    """
    Latency performance metrics.

    Comprehensive latency statistics for evaluation.
    """

    latencies_ms: list[float] = field(default_factory=list)

    @property
    def total_requests(self) -> int:
        return len(self.latencies_ms)

    @property
    def mean(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.latencies_ms)) if len(self.latencies_ms) > 1 else 0.0

    @property
    def p50(self) -> float:
        return float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0

    @property
    def p90(self) -> float:
        return float(np.percentile(self.latencies_ms, 90)) if self.latencies_ms else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0.0

    @property
    def min(self) -> float:
        return float(np.min(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def max(self) -> float:
        return float(np.max(self.latencies_ms)) if self.latencies_ms else 0.0

    @property
    def throughput(self) -> float:
        """Estimated throughput based on P50 latency (req/s)."""
        return 1000.0 / self.p50 if self.p50 > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_requests": self.total_requests,
            "mean_ms": round(self.mean, 2),
            "median_ms": round(self.median, 2),
            "std_ms": round(self.std, 2),
            "p50_ms": round(self.p50, 2),
            "p90_ms": round(self.p90, 2),
            "p95_ms": round(self.p95, 2),
            "p99_ms": round(self.p99, 2),
            "min_ms": round(self.min, 2),
            "max_ms": round(self.max, 2),
            "throughput_rps": round(self.throughput, 2),
        }


@dataclass
class EvaluatorMetrics:
    """
    Combined metrics for a single evaluator.

    Aggregates all metric types for comparison.
    """

    evaluator_name: str
    hallucination: HallucinationMetrics = field(default_factory=HallucinationMetrics)
    truthfulqa: TruthfulQAMetrics = field(default_factory=TruthfulQAMetrics)
    factscore: FActScoreMetrics = field(default_factory=FActScoreMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    def to_dict(self) -> dict[str, Any]:
        """Convert all metrics to dictionary."""
        return {
            "evaluator": self.evaluator_name,
            "hallucination": self.hallucination.to_dict(),
            "truthfulqa": self.truthfulqa.to_dict(),
            "factscore": self.factscore.to_dict(),
            "latency": self.latency.to_dict(),
        }

    def get_summary_scores(self) -> dict[str, float]:
        """
        Get summary scores for radar chart visualization.

        All scores normalized to 0-1 range where higher is better.
        """
        # Compute RAG faithfulness (measures claim-evidence relevance)
        rag = self.hallucination.ragas_proxy_metrics()

        # AURC is lower-better, map to (0,1] for radar (higher = better)
        aurc = self.hallucination.aurc
        selective_score = 1.0 / (1.0 + max(0.0, aurc)) if aurc >= 0 else 0.0

        return {
            "Accuracy": self.hallucination.accuracy,
            "Precision": self.hallucination.precision,
            "Recall": self.hallucination.recall,
            "F1 Score": self.hallucination.f1_score,
            "Safety (1-HPR)": 1.0 - self.hallucination.hallucination_pass_rate,
            "TruthfulQA": self.truthfulqa.accuracy,
            "FActScore": self.factscore.avg_factscore,
            "Speed (1/P95)": min(1.0, 1000.0 / self.latency.p95) if self.latency.p95 > 0 else 0.0,
            "Selective (1/(1+AURC))": selective_score,
            # Evidence relevance: how well evidence supports the claim
            "Evidence Relevance": float(rag.get("faithfulness", 0.0)),
        }


@dataclass
class ComparisonReport:
    """
    Complete comparison report across all evaluators.
    """

    evaluators: dict[str, EvaluatorMetrics] = field(default_factory=dict)
    run_id: str = ""
    timestamp: str = ""
    config_summary: dict[str, Any] = field(default_factory=dict)

    def add_evaluator(self, metrics: EvaluatorMetrics) -> None:
        """Add evaluator metrics to report."""
        self.evaluators[metrics.evaluator_name] = metrics

    def get_ranking(self, metric: str = "f1_score") -> list[str]:
        """
        Get evaluators ranked by a specific metric.

        Args:
            metric: Metric name (f1_score, accuracy, factscore, etc.)

        Returns:
            List of evaluator names sorted by metric (best first)
        """
        scores: list[tuple[str, float]] = []

        for name, metrics in self.evaluators.items():
            if metric == "accuracy":
                score = metrics.hallucination.accuracy
            elif metric == "f1_score":
                score = metrics.hallucination.f1_score
            elif metric == "factscore":
                score = metrics.factscore.avg_factscore
            elif metric == "truthfulqa":
                score = metrics.truthfulqa.accuracy
            elif metric == "latency":
                # Lower latency is better, so negate.
                # Note: If p95 <= 0 (e.g., no reliable latency data due to zero-division
                # safeguards in LatencyMetrics), treat this as missing data and rank
                # such evaluators last by giving them the worst possible score.
                p95 = metrics.latency.p95
                score = -p95 if p95 > 0 else float("-inf")
            elif metric == "safety":
                # Lower hallucination pass rate is better
                score = -metrics.hallucination.hallucination_pass_rate
            elif metric == "aurc":
                # Lower AURC is better
                score = -metrics.hallucination.aurc
            elif metric == "ndcg@10":
                retrieval = metrics.hallucination.retrieval_metrics(ks=(10,))
                score = float(retrieval.get("ndcg@10", 0.0))
            elif metric == "recall@10":
                retrieval = metrics.hallucination.retrieval_metrics(ks=(10,))
                score = float(retrieval.get("recall@10", 0.0))
            elif metric == "ragas_faithfulness":
                rag = metrics.hallucination.ragas_proxy_metrics()
                score = float(rag.get("faithfulness", 0.0))
            else:
                score = 0.0

            scores.append((name, score))

        # Sort descending (highest score first)
        scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scores]

    def to_dict(self) -> dict[str, Any]:
        """Convert full report to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": self.config_summary,
            "evaluators": {name: metrics.to_dict() for name, metrics in self.evaluators.items()},
            "rankings": {
                "by_f1": self.get_ranking("f1_score"),
                "by_accuracy": self.get_ranking("accuracy"),
                "by_safety": self.get_ranking("safety"),
                "by_factscore": self.get_ranking("factscore"),
                "by_latency": self.get_ranking("latency"),
                "by_aurc": self.get_ranking("aurc"),
                "by_ndcg10": self.get_ranking("ndcg@10"),
                "by_ragas_faithfulness": self.get_ranking("ragas_faithfulness"),
            },
        }

    def get_comparison_table(self) -> list[dict[str, Any]]:
        """
        Generate comparison table data for visualization.

        Returns:
            List of rows with evaluator name and all metrics.
        """
        rows: list[dict[str, Any]] = []

        for name, metrics in self.evaluators.items():
            retrieval = metrics.hallucination.retrieval_metrics(ks=(10, 100))
            alce = metrics.hallucination.alce_metrics()
            rag = metrics.hallucination.ragas_proxy_metrics()
            row = {
                "Evaluator": name,
                "Accuracy": f"{metrics.hallucination.accuracy:.1%}",
                "Precision": f"{metrics.hallucination.precision:.1%}",
                "Recall": f"{metrics.hallucination.recall:.1%}",
                "F1 Score": f"{metrics.hallucination.f1_score:.1%}",
                "Halluc. Pass Rate": f"{metrics.hallucination.hallucination_pass_rate:.1%}",
                "AURC": f"{metrics.hallucination.aurc:.4f}",
                "E-AURC": f"{metrics.hallucination.eaurc:.4f}",
                "nDCG@10": (f"{float(retrieval.get('ndcg@10', 0.0)):.3f}" if retrieval else "N/A"),
                "Recall@10": (
                    f"{float(retrieval.get('recall@10', 0.0)):.3f}" if retrieval else "N/A"
                ),
                "Recall@100": (
                    f"{float(retrieval.get('recall@100', 0.0)):.3f}" if retrieval else "N/A"
                ),
                "MRR@10": f"{float(retrieval.get('mrr@10', 0.0)):.3f}" if retrieval else "N/A",
                "Precision@10": (
                    f"{float(retrieval.get('precision@10', 0.0)):.3f}" if retrieval else "N/A"
                ),
                "Citation Rate": f"{float(alce.get('citation_rate', 0.0)):.1%}" if alce else "N/A",
                "Citation Grounding": (
                    f"{float(alce.get('citation_grounding_proxy', 0.0)):.3f}"
                    if alce and alce.get("citation_grounding_proxy") is not None
                    else "N/A"
                ),
                "RAG Faithfulness": (
                    f"{float(rag.get('faithfulness', 0.0)):.3f}" if rag else "N/A"
                ),
                "RAG Answer Rel.": (
                    f"{float(rag.get('answer_relevancy', 0.0)):.3f}" if rag else "N/A"
                ),
                "RAG Ctx Prec.": (
                    f"{float(rag.get('context_precision', 0.0)):.3f}" if rag else "N/A"
                ),
                "RAG Ctx Util.": (
                    f"{float(rag.get('context_utilization', 0.0)):.3f}" if rag else "N/A"
                ),
                "TruthfulQA": f"{metrics.truthfulqa.accuracy:.1%}",
                "FActScore": f"{metrics.factscore.avg_factscore:.1%}",
                "P50 Latency": f"{metrics.latency.p50:.0f}ms",
                "P95 Latency": f"{metrics.latency.p95:.0f}ms",
                "Throughput": f"{metrics.latency.throughput:.1f} req/s",
            }
            rows.append(row)

        return rows
