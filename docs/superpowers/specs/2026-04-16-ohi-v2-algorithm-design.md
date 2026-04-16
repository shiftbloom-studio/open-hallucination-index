---
title: OHI v2 Algorithm — Layered Stack with Probabilistic Claim Graph and Calibrated Conformal Prediction
date: 2026-04-16
status: draft (pending spec review + user approval)
scope: src/api/pipeline/ — full replacement of v1
sub_project: 1 of 3 (algorithm; followed by infrastructure + frontend)
authors: Fabian Zimber, with Claude
---

# OHI v2 Algorithm — Design Specification

## Executive summary

OHI v2 is a complete replacement of the current Open Hallucination Index verification engine. It introduces a seven-layer pipeline that transforms unstructured input text into a calibrated, joint-inferred trust assessment with provable coverage guarantees. The headline contributions are:

1. **A Probabilistic Claim Graph (PCG) with tree-reweighted belief propagation, validated by Gibbs MCMC.** Claims are not verified independently. Entailment and contradiction edges between claims (extracted by a calibrated NLI cross-encoder) propagate evidence between related claims via TRW-BP on an Ising-style log-linear graph; in `balanced`+ rigor tiers a parallel Gibbs sampler validates the variational marginals. Internal contradictions in the input are detected as a first-class hallucination signal.
2. **Per-domain split-conformal prediction with Mondrian stratification.** Every claim verdict is a calibrated probability `P(true)` plus a `[lower, upper]` 90% prediction interval, validated to maintain ≤10% miscoverage on held-out data per domain × claim-type stratum.
3. **An open active learning loop with EWC++ regularized retraining.** Public users submit feedback through a `/feedback` endpoint with a 3-concordant consensus filter; nightly jobs refit per-domain conformal calibration and fine-tune NLI heads with elastic weight consolidation to prevent catastrophic forgetting.
4. **Five-domain specialization.** General, biomedical, legal, code, and social each have their own NLI head, source-credibility prior, and calibration set, dispatched by a per-claim soft router.
5. **Open-by-default API.** No authentication; rate-limit + traffic-protection at the edge; a deterministic daily cost ceiling that gracefully translates load spikes into 503 with `Retry-After`. A public `/calibration/report` endpoint exposes empirical coverage statistics for auditability.
6. **Three rigor tiers (`fast` / `balanced` / `maximum`).** The pipeline trades latency for rigor explicitly via an API parameter — from sub-30s draft checks to multi-minute production verification to multi-hour paper-grade analyses with NLI ensembles, full-quadratic claim graphs, and Gibbs sampling sanity checks against TRW-BP marginals. There is no fixed wall-clock cap; long requests run to completion under the rigor tier they accepted.

The combination — calibrated NLI + joint inference via PCG + auditable conformal coverage + active learning with catastrophic-forgetting protection — is, to the best of our knowledge, novel in the hallucination-detection literature. It is designed to be both a publishable methodological contribution and a production-grade verification service.

## Table of contents

- [1. Goals, non-goals, success criteria](#1-goals-non-goals-success-criteria)
- [2. Architecture overview](#2-architecture-overview)
- [3. L1 — Decomposition + Evidence Retrieval](#3-l1--decomposition--evidence-retrieval)
- [4. L2 — Domain Router](#4-l2--domain-router)
- [5. L3 — NLI Cross-Encoder Layer](#5-l3--nli-cross-encoder-layer)
- [6. L4 — Probabilistic Claim Graph](#6-l4--probabilistic-claim-graph)
- [7. L5 — Conformal Prediction Layer](#7-l5--conformal-prediction-layer)
- [8. L6 — Active Learning Hook](#8-l6--active-learning-hook)
- [9. L7 — Output Assembly](#9-l7--output-assembly)
- [10. API surface](#10-api-surface)
- [11. Open access, rate limiting, traffic protection](#11-open-access-rate-limiting-traffic-protection)
- [12. Active learning loop, feedback store, retraining](#12-active-learning-loop-feedback-store-retraining)
- [13. Phased rollout](#13-phased-rollout)
- [14. Risks and mitigations](#14-risks-and-mitigations)
- [15. Open questions and tentative recommendations](#15-open-questions-and-tentative-recommendations)
- [16. Testing strategy](#16-testing-strategy)
- [17. Documentation deliverables](#17-documentation-deliverables)
- [18. Glossary and references](#18-glossary-and-references)

---

## 1. Goals, non-goals, success criteria

### Goals

- **G1. Calibrated trust scoring.** Every claim verdict is a calibrated probability `P(claim is true) ∈ [0, 1]` with a 90% conformal prediction interval, validated to maintain ≤10% miscoverage on held-out data per domain.
- **G2. Joint claim reasoning.** A Probabilistic Claim Graph (Ising-style, TRW-BP primary inference, Gibbs MCMC sanity-check) propagates evidence between related claims via NLI-derived entailment and contradiction edges. Internal contradictions are detected and surfaced as a hallucination signal.
- **G3. Domain-aware verification.** A learned per-claim router classifies into one of 5 domains; each domain owns its NLI head, source-credibility weights, and conformal calibration set.
- **G4. Active learning loop.** Public `/feedback` captures labels; nightly jobs (a) refit per-domain conformal calibration, (b) fine-tune the NLI head with EWC++ regularization, and (c) surface high-information claims for prioritized human review.
- **G5. Layered, ablation-friendly architecture.** Seven discrete layers with versioned interfaces; each is feature-flag-gated and individually replaceable.

### Non-goals (explicit, scope-protective)

- **Token-level streaming verification.** Pipeline is batch-per-document. The `/verify/stream` SSE endpoint streams *layer-completion events*, not LLM tokens.
- **Auto-correction / agentic remediation.** No `/rewrite` endpoint in v2.0. Detection + scoring + provenance only.
- **Cross-language verification.** English only.
- **Pretraining models from scratch.** We fine-tune existing checkpoints (DeBERTa-v3, etc.).
- **v1 backward compatibility.** v1 endpoints, schemas, and internals are deleted. `/api/v2/*` is the only namespace.

### Success criteria

- **Quantitative.** Beat current OHI on FActScore, TruthfulQA, HaluEval, and one domain benchmark per vertical (PubMedQA, LegalBench-Entailment, custom code fact-check eval, LIAR/MultiFC/ClimateFEVER). Target: ≥5pt F1 improvement on at least 4 of the 5 domain benchmarks relative to v1.
- **Calibration.** Expected Calibration Error (ECE) ≤ 0.05 per domain; conformal coverage ≥ 88% on held-out test set.
- **Latency.** No fixed wall-clock cap. The pipeline trades latency for rigor. Reference points (A100 inference, default `rigor=balanced`): simple inputs (1–3 claims) ≈ 10–30s; typical (5–15 claims) ≈ 1–3 min; complex (>30 claims) ≈ 5–10 min; maximum-rigor configuration on a long document may exceed 20 min and that is acceptable. Latency budgets are explicit per `rigor` tier in §2 and surfaced through the API `options.rigor` field.
- **Active learning.** ≥10% F1 improvement after 2 weeks of feedback collection on the prioritized-review subset.

---

## 2. Architecture overview

The pipeline is seven discrete layers. Each has a well-defined input/output contract, can be replaced or ablated independently, and is feature-flag-gated for staged rollout.

```
INPUT TEXT (+ optional context, domain hint)
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ L1 · Decomposition + Evidence Retrieval                      │
│   • LLM decomposer → atomic claims (SPO + type + span)        │
│   • Router → relevant adapters (Neo4j / Qdrant / MCP)         │
│   • AdaptiveCollector → evidence pool per claim               │
│   OUT: Claim[], Evidence[claim_id]                            │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ L2 · Domain Router                                           │
│   • Per-claim classifier → {general, biomed, legal, code,     │
│     social}; soft assignment when ambiguous                   │
│   • Binds claim to a DomainAdapter                            │
│   OUT: DomainTaggedClaim[]                                    │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ L3 · NLI Cross-Encoder Layer                                 │
│   • Per claim × evidence: P(entail / contradict / neutral)    │
│   • Per claim × claim: same, used for graph edges             │
│   • Self-consistency: K=3 stochastic forward passes           │
│   OUT: NLI tensors with calibrated logits + variance          │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ L4 · Probabilistic Claim Graph (PCG)                         │
│   • Nodes = claims; edges from L3 NLI                         │
│   • Loopy belief propagation, max 8 iterations or convergence │
│   OUT: Joint posterior P(c=true) per claim                    │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ L5 · Conformal Prediction Layer                              │
│   • Per-domain × per-claim-type Mondrian split conformal      │
│   • Mixture conformal for soft domain assignments             │
│   OUT: P_calibrated + [lower, upper] @ 90% coverage           │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ L6 · Active Learning Hook                                    │
│   • Compute information gain (entropy × interval width × wt)  │
│   • If above threshold → enqueue for review (fire-and-forget) │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────┐
│ L7 · Output Assembly                                         │
│   • ClaimVerdict (calibrated P + interval + provenance)       │
│   • DocumentVerdict (copula-aggregated joint probability)     │
│   OUT: VerificationResultV2                                   │
└──────────────────────────────────────────────────────────────┘
```

### Async sidecars (out-of-band)

```
POST /v2/feedback ─► FeedbackStore (Postgres + S3)
                            │
                            ▼
                   Nightly job (02:00 UTC):
                     • Refit per-domain conformal calibration
                     • Fine-tune NLI head with EWC++ regularization
                     • Recompute source-credibility priors
                     • Coverage validation gate → promote or rollback
```

### Module layout (after v1 removal)

```
src/api/pipeline/
├── decomposer.py          # L1 — refactored, chunked LLM decomposition
├── retrieval/             # L1 — split out from oracle.py
│   ├── router.py
│   ├── collector.py
│   └── source_credibility.py    # NEW
├── domain/                # L2 — NEW
│   ├── router.py
│   ├── adapters/                # general/biomed/legal/code/social
│   └── registry.py
├── nli/                   # L3 — NEW
│   ├── cross_encoder.py
│   ├── self_consistency.py
│   └── batching.py
├── pcg/                   # L4 — NEW
│   ├── graph.py
│   ├── propagation.py
│   └── potentials.py
├── conformal/             # L5 — NEW
│   ├── split_conformal.py
│   ├── mondrian.py
│   └── calibration_store.py
├── active_learning/       # L6 — NEW
│   ├── information_gain.py
│   ├── review_queue.py
│   └── retrainer.py
├── assembly/              # L7 — NEW
│   ├── claim_verdict.py
│   └── document_verdict.py
└── pipeline.py            # Orchestrator: L1 → L2 → ... → L7
```

`oracle.py` and `scorer.py` are removed. Their roles are dispersed across L3 (verification), L4 (joint inference), L5 (calibration), and L7 (assembly).

### Rigor tiers and latency profile

OHI v2 trades latency for rigor explicitly through the API `options.rigor` field. There is no fixed wall-clock cap; the system commits to *finishing* every accepted request, with the per-tier work envelope below. All numbers assume one **NVIDIA A100 40GB**, FP16, batch 32, sequence 512 tokens (DeBERTa-v3-large measured at ~110–140 forward passes/sec on this configuration; A10G fallback is ~40–60/sec, which roughly doubles wall times below).

| Tier | What changes | Use case |
|---|---|---|
| `fast` | K_self_consistency = 3, claim↔claim NLI top-K capped at 5N, PCG = TRW-BP only, no MCMC sanity, no refinement pass, no NLI ensemble | Real-time UX, draft checks |
| `balanced` (default) | K = 10, full quadratic claim↔claim NLI for N ≤ 40 (top-K capped at 10N above), TRW-BP with optional Gibbs spot-check on flagged graphs, 1 refinement pass on contradiction-flagged claims | Production default; "I want a real answer in a couple of minutes" |
| `maximum` | K = 30, full quadratic claim↔claim NLI regardless of size, TRW-BP + parallel Gibbs MCMC for full posterior validation, up to 3 refinement passes, NLI ensemble (DeBERTa-v3-large MNLI + RoBERTa-large MNLI averaged), expanded retrieval tier | High-stakes verification, research benchmarking, paper-grade results |

**Reference latencies (A100, default retrieval tier):**

| Document size | `fast` | `balanced` | `maximum` |
|---|---|---|---|
| 3 claims | ~5s | ~15s | ~90s |
| 10 claims | ~25s | ~90s | ~6 min |
| 30 claims | ~75s | ~5 min | TBD (≥20 min) |
| 100 claims | ~5 min | ~25 min | TBD (≥1 h) |

The `fast` and `balanced` numbers are derived from the per-layer breakdown below. The `maximum` tier numbers for ≥30 claims are upper-bound estimates pending Phase 0 profiling — full quadratic claim↔claim NLI (435+ pairs at N=30, 4,950 at N=100), K=30 self-consistency with checkpoint ensembling, multi-chain Gibbs with R̂ gating, and up to 3 refinement passes are all individually expensive and the empirical totals depend heavily on the eventual hardware. The `maximum` tier exists precisely so that we can publish results without latency-driven shortcuts; we commit to *finishing* every accepted `maximum` request, not to a particular wall-clock target.

**Per-layer breakdown (default `balanced` tier, 10-claim document):**

| Layer | Op count | Wall (A100) | Notes |
|---|---|---|---|
| L1 (decompose + retrieve) | 1 LLM decomp + ~300 retrievals | 10–20s | I/O bound; LLM is the long pole |
| L2 (domain routing) | 10 distilbert | 1s | Batched |
| L3 (NLI claim↔evidence, K=10) | 10 × 30 × 10 = 3,000 fwd | ~30s | ~94 batches of 32 |
| L3 (NLI claim↔claim full, K=10) | 45 unique × 10 = 450 fwd | ~5s | Full quadratic; N ≤ 40 |
| L4 (TRW-BP, up to 30 iterations) | ≤30 nodes | 5–10s | NumPy vectorized; converges quickly at this size |
| L4 refinement pass (1×) | re-retrieve flagged + re-NLI on subset | 10–15s | Often skipped if no contradictions |
| L5 + L6 + L7 | dict ops + SQS post | 1–2s | Negligible |
| **Total balanced 10-claim** | | **≈ 60–90s** | Comfortably under 2 min |

We deliberately do not budget L3 down by reducing K below 10 in `balanced`. K = 10 gives sample-variance confidence intervals that are usable as relative edge weights *and* as documented epistemic-uncertainty signals; K = 3 was a compromise driven by an obsolete latency target.

The infrastructure sub-project chooses concrete hardware. Phase 0 benchmarks both A100 and A10G on the chosen checkpoint and writes the locked numbers into the runbook.

---

## 3. L1 — Decomposition + Evidence Retrieval

### Purpose

Convert raw input text into a list of atomic, well-typed claims, each with normalized SPO and a pool of candidate evidence. This layer is the only place we touch raw text or external knowledge sources.

### Module

`src/api/pipeline/decomposer.py` + `src/api/pipeline/retrieval/`

### Interface

```python
class DecompositionService(Port):
    async def decompose(
        self,
        text: str,
        *,
        context: str | None,
        max_claims: int = 50,
    ) -> list[Claim]: ...


class RetrievalService(Port):
    async def collect(
        self,
        claim: Claim,
        *,
        tier: EvidenceTier,
        target_count: int | None,
    ) -> EvidencePool: ...
```

### Key changes vs. v1

1. **Chunked long-input decomposition.** Inputs are split at paragraph boundaries with a 200-token co-reference overlap window. Per-chunk claims merged and deduplicated by normalized form + cosine similarity ≥ 0.92.
2. **Coverage re-prompt.** If `len(claims) / len(sentences) < 0.4` and the text is fact-dense (POS-tag heuristic), re-prompt with `"You may have missed claims. Re-read and add only NEW atomic claims not in this list: <prev>"`. Single re-prompt cap.
3. **Strong normalization.** Dates → ISO 8601, numbers → SI base units, named entities → canonical Wikidata QID (where Wikidata MCP returns one). Cleaner equality semantics for L4 cross-claim joint inference.
4. **Source-credibility prior at retrieval time.** Each `Evidence` carries `source_credibility ∈ [0, 1]` and `temporal_decay_factor ∈ [0, 1]` set by the retrieval adapter.
5. **Provenance hashing.** `Evidence.fingerprint = sha256(source_uri + content_chunk)` for cross-path deduplication.

### Source-credibility table (initial)

| Source | Credibility |
|---|---|
| `peer_reviewed_journal` | 0.95 |
| `official_gov_docs` | 0.92 |
| `wikipedia_featured_article` | 0.88 |
| `mcp_curated` | 0.80 |
| `wikipedia_general` | 0.78 |
| `news_high_repute` | 0.75 |
| `qdrant_general` | 0.70 |
| `news_general` | 0.65 |
| `graph_inferred` | 0.60 |

These are starting priors. The nightly retraining job (Section 12) updates them per `(source, domain)` pair via Beta-Bernoulli posteriors with capped per-night drift (±0.05).

### Edge cases

- Empty input → `[]`, no error.
- Decomposition LLM failure → sentence-split fallback with confidence 0.4 (so L4 down-weights heavily).
- Retrieval timeout → `retrieval_status="partial"` flag; L4 marks the local potential as high-variance.

### Tests

- Unit tests on chunking, dedup, normalization, coverage re-prompt heuristic.
- Integration test against the existing Wikipedia ingestion store.
- Property-based tests on idempotency (same input → same claims modulo UUIDs).

---

## 4. L2 — Domain Router

### Purpose

For each claim, pick the right `DomainAdapter`. The adapter determines NLI head, source-credibility weights, and conformal calibration set. Per-claim routing (not per-document) handles mixed inputs natively.

### Module

`src/api/pipeline/domain/`

### Interface

```python
class DomainAdapter(Port):
    domain: Domain
    def nli_model_id(self) -> str: ...
    def source_credibility(self) -> dict[str, float]: ...
    def calibration_set_id(self) -> str: ...
    def decomposition_hints(self) -> str | None: ...
    def claim_pair_relatedness_threshold(self) -> float: ...


class DomainRouter:
    async def route(self, claim: Claim) -> DomainAssignment:
        # returns soft assignment: {domain → weight}, sums to 1
```

### The 5 domains

| Domain | Source-credibility skew | NLI fine-tune corpus | Benchmark |
|---|---|---|---|
| `general` | Wikipedia ≫ news ≫ blog | FEVER + ANLI | FActScore, TruthfulQA, HaluEval |
| `biomedical` | PubMed ≫ Cochrane ≫ Wiki health | SciFact + PubMedQA-NLI | PubMedQA, BioASQ-Factoid |
| `legal` | Court opinions ≫ statute ≫ commentary | ContractNLI + LegalBench | LegalBench-Entailment |
| `code` | Official docs ≫ stable repos ≫ SO | CodeNLI (synthesized via ingestion) | Custom Python/JS docs fact-check eval |
| `social` | Fact-checkers ≫ peer-reviewed ≫ Wiki ≫ established news ≫ social media | MultiFC + LIAR + ClimateFEVER + COVID-Fact | LIAR, MultiFC, ClimateFEVER |

### Routing model

- `distilbert-base` classifier fine-tuned on a domain-labeled mixture (synthesized from existing benchmark domains + 1k hand-labeled seed).
- Output: probability vector over 5 domains.
- **Soft assignment**: if `top1 - top2 < 0.15`, treat as soft. The `DomainAdapter` facade transparently blends NLI heads (mean of logits) and calibration sets (mixture conformal — Tibshirani 2019 weighted exchangeability).

### Cold-start strategy

Until per-domain calibration sets reach `n ≥ 200` labeled examples, the router falls back to the `general` adapter for that domain (logged + flagged). Frontend surfaces a "Limited calibration for domain X" badge.

### Edge cases

- Out-of-scope domain (poetry, opinion text) → routed to `general` with `domain_oos=True`; conformal interval auto-widens.
- Truly mixed claim ("This drug, recently approved by the SEC...") → soft assignment; PCG sees both potentials and reconciles.

### Tests

- Classification accuracy on held-out domain mix ≥ 0.85 macro-F1.
- Routing latency P95 < 100ms per claim.
- Soft-assignment mixture math verified against Tibshirani's reference implementation.

---

## 5. L3 — NLI Cross-Encoder Layer

### Purpose

Replace v1's prompt-engineered LLM classification with a calibrated, learned, probabilistic NLI classifier. Output is a `Categorical(entail, contradict, neutral)` distribution per (claim, evidence) and per (claim, claim) pair, with calibrated logits and an empirical variance estimate from self-consistency sampling.

### Module

`src/api/pipeline/nli/`

### Interface

```python
@dataclass(frozen=True)
class NLIDistribution:
    entail: float       # ∈ [0, 1], sums with others = 1
    contradict: float
    neutral: float
    variance: float     # from self-consistency K passes
    nli_model_id: str


class NLIService(Port):
    async def claim_evidence(
        self,
        claim: Claim,
        evidence: list[Evidence],
        adapter: DomainAdapter,
    ) -> list[NLIDistribution]: ...

    async def claim_claim(
        self,
        claims: list[Claim],
        adapter: DomainAdapter,
    ) -> dict[tuple[ClaimId, ClaimId], NLIDistribution]: ...
```

### Model

- **Base checkpoint:** `microsoft/deberta-v3-large` further fine-tuned on MNLI (we will use the public `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` checkpoint as the starting point — already trained on MNLI + FEVER + ANLI + LingNLI + WANLI, so it knows the entailment task and is asymmetric in the standard MNLI sense). Single shared encoder instance, kept resident on GPU.
- **Per-domain heads:** small classification heads (~5MB each) that swap on the encoder. Heads loaded lazily; only domains touched by the active document are pinned.
- **Recommendation (tentative):** revisit per-domain base encoders if Phase 3 domain F1 underperforms. SciFive / BiomedBERT-NLI are the most likely candidates for biomedical.

### Self-consistency (K configurable; default K=10)

For each (claim, evidence) pair, do K stochastic forward passes with input perturbation. `K` is set by the rigor tier: `fast`=3, `balanced`=10 (default), `maximum`=30.

The K passes are drawn (with replacement when K > the cross product of perturbation choices) from the following perturbation distribution:

1. **Lexical paraphrases of the claim** via a small T5-paraphrase model. We pre-generate up to 8 paraphrases per claim, cached. Paraphrase quality is filtered: only paraphrases with bidirectional NLI entailment ≥ 0.9 against the original are retained.
2. **Evidence sentence-window slides** (±1, ±2 sentences around the central retrieved span; up to 5 windows).
3. **Premise/hypothesis order swap** — DeBERTa-MNLI fine-tunes are asymmetric on the (premise, hypothesis) ordering, so claim-as-premise vs. claim-as-hypothesis are independent estimates.
4. **(maximum tier only) NLI checkpoint ensembling** — both DeBERTa-v3-large MNLI head and RoBERTa-large MNLI score every (claim, evidence) pair (and (claim, claim) pair). The two model outputs are combined by **softmax-probability average** (not logit average — calibration scaling is checkpoint-specific so logit average would mix incompatible scales). The averaged probability vector is then the input to the K-pass self-consistency mean. The K=30 budget in `maximum` is split evenly: 15 passes per checkpoint with the perturbation set above.

Final distribution = mean of softmax over K passes; variance = sample variance. With K = 10 (default), the variance estimate has tight enough confidence bands to function both as (a) a relative weight for L4 PCG edges and (b) a documented epistemic-uncertainty signal in the API output (`nli_self_consistency_variance` field). We still do not present this variance as a calibrated probability — that role remains exclusively L5's conformal interval, which has a real coverage guarantee.

### Claim ↔ claim NLI

Naively O(N²). Behavior is rigor-tier dependent:

- **`fast`**: bi-encoder pre-filter + top-K cap (5N pairs). Bi-encoder = `sentence-transformers/all-mpnet-base-v2`; threshold = `adapter.claim_pair_relatedness_threshold` (default 0.45 for general; 0.30 for `social` and `legal` where contradictions are often lexically dissimilar).
- **`balanced`** (default): full quadratic enumeration for `N ≤ 40` (≤ 780 unique pairs); above N=40, top-K cap at 10N. The bi-encoder is still run as a feature input to the PCG potential (high relatedness ⇒ stronger edge prior) but is *not* used to prune.
- **`maximum`**: full quadratic enumeration regardless of N. No bi-encoder pruning at any N.

The bi-encoder threshold issue (lexically dissimilar contradictions get pruned in `fast`) is acknowledged in `fast` mode and largely solved by `balanced` and above doing full enumeration up to reasonable N.

For a 30-claim document, full enumeration is `30 · 29 / 2 = 435` unique unordered pairs, well within budget at `balanced`.

**Symmetry caching**: compute each unordered pair (i, j), i < j once; the reverse direction (j, i) is a cheap second softmax on the already-encoded pair (DeBERTa-MNLI is asymmetric so both orderings carry information).

### Calibrated logits

Each per-domain head has a temperature parameter `T_d` fit on a held-out validation set via NLL minimization. Calibrated probability = `softmax(logits / T_d)`. `T_d` is refit nightly. This is *not* the conformal calibration (that's L5); it's classifier-side temperature scaling and gives well-behaved probability inputs to the PCG.

### Edge cases

- Evidence too long → truncated to 512 tokens with sliding-window aggregation (max-pool of entailment, mean-pool of contradiction).
- Claim too short (< 5 tokens) → flagged as `atomic_fragment`; L4 widens its prior.
- All evidence below relatedness threshold → emit `NLIDistribution(0.33, 0.33, 0.34, variance=high)`; L4 falls back to graph-based propagation from neighbors.

### Tests

- Calibration unit tests: temperature scaling reduces NLL vs. uncalibrated.
- Golden tests on known FEVER pairs.
- Latency benchmark: 30-claim document < 30s on a single A10G.

---

## 6. L4 — Probabilistic Claim Graph

### Purpose

Treat the document as a structured probabilistic graphical model. Per-claim posteriors are not independent — they are computed by joint inference over a graph whose edges encode logical/semantic relationships. This is what makes "the document is internally inconsistent" a first-class hallucination signal.

### Module

`src/api/pipeline/pcg/`

### Construction (Ising-style log-linear formulation)

We model the joint distribution over claim-truth assignments as a pairwise Ising model. Encode `T_c ∈ {-1, +1}` (false / true). The energy is:

```
E(T) = − ∑_c α_c · T_c   −   ∑_{(i,j) ∈ E} J_ij · T_i · T_j

P(T) ∝ exp(−E(T))
```

**1. Node prior log-odds** `α_c` from L3 claim↔evidence NLI.

Define per-evidence true/false probability mass (the `neutral` softmax mass is split evenly):

```
p_t(c, e) = entail(c, e)     + 0.5 · neutral(c, e)
p_f(c, e) = contradict(c, e) + 0.5 · neutral(c, e)
```

Then aggregate, evidence-weighted by source credibility, freshness, and NLI confidence:

```
w_e = source_credibility(e) · temporal_decay(e) · (1 − NLI_variance(c, e))

S_t = ∑_e w_e · log p_t(c, e)
S_f = ∑_e w_e · log p_f(c, e)

α_c = (S_t − S_f) / max(1, ∑_e w_e)
```

`α_c` is a well-defined real-valued log-odds. Positive = evidence favors `T_c = +1` (true); negative = favors false. Magnitude scales with the agreement of weighted evidence. Normalization by `∑ w_e` prevents documents with more retrieved evidence from automatically getting stronger priors.

**2. Edge interaction parameters** `J_ij` from L3 claim↔claim NLI.

Let `e_ij = entail(c_i, c_j)`, `r_ij = contradict(c_i, c_j)`, `n_ij = neutral(c_i, c_j)`.

```
strength_ij = max(e_ij, r_ij) − n_ij                   ∈ [-1, 1]
sign_ij     = +1 if e_ij ≥ r_ij else −1                # entail vs contradict
J_ij        = sign_ij · max(0, strength_ij) · γ · (1 − NLI_variance(c_i, c_j))
```

`γ` is a global edge-temperature hyperparameter (initial value 1.0; tuned on FEVER dev). `J_ij > 0` for entailment edges (favors same label); `J_ij < 0` for contradiction edges (favors opposite label). Weak edges (low strength) auto-decay to ~0 and contribute nothing — no need for a separate prune threshold beyond the bi-encoder pre-filter in L3.

**3. The graph is well-defined and identifiable.**

Both unary and pairwise terms are real-valued log-potentials; `P(T) ∝ exp(−E(T))` is a proper Boltzmann distribution. No normalization division-by-zero; no constant-row edge cases.

### Inference: Tree-Reweighted Belief Propagation (primary)

The graph is generally cyclic and can have strong contradiction edges. Plain loopy BP is known to oscillate or converge to non-Bethe-stationary points in such cases (Murphy, Weiss & Jordan, 1999). We use **tree-reweighted belief propagation** (Wainwright, Jaakkola & Willsky, 2005) as the primary inference algorithm, with damped loopy BP as a fallback.

**Why TRW-BP:**
- **Convergent** under broad conditions (vs. LBP's possible oscillation on adversarial graphs).
- **Provides upper bound** on the log-partition function — useful when the document-level normalization matters for L7 aggregation.
- **Interpolates between mean-field and LBP** via tree-reweighting coefficients `ρ_e ∈ [0, 1]`, allowing tuning toward more conservative inference on graphs we are less sure about.

**Update rule** (for binary nodes, message in log-space):

```
m_{i→j}^{k+1}(T_j) = (1 − δ) · m_{i→j}^k(T_j)
                   + δ · ρ_{ij}⁻¹ · log ∑_{T_i} exp(
                         α_i · T_i / ρ_i
                       + J_ij · T_i · T_j / ρ_{ij}
                       + ∑_{n ∈ N(i)\{j}} ρ_{ni} · m_{n→i}^k(T_i)
                     )

Belief: b_i(T_i) ∝ exp( α_i · T_i / ρ_i + ∑_{n ∈ N(i)} ρ_{ni} · m_{n→i}(T_i) )
```

`ρ_e` are computed once per graph from a uniform distribution over spanning trees (cheap — N log N with Kirchhoff's matrix-tree theorem). Damping δ = 0.5. Convergence criterion: max message change < 10⁻³ or 8 iterations.

**LBP fallback.** If TRW-BP fails to converge (max message change still > 10⁻² after 8 iterations), we re-run with damped LBP (δ = 0.3), accept the result if it converges, and otherwise emit `converged=False` and `algorithm_used="LBP-nonconvergent"`. The conformal layer (L5) treats non-converged claims as out-of-calibration — see L5 changes below.

### Gibbs sampling sanity check (rigor=`balanced` opt-in, `maximum` always)

To validate TRW-BP's variational approximation against actual posterior samples, we run a Gibbs MCMC chain in parallel.

**Trigger predicate (`balanced` tier):** invoke Gibbs only when the TRW-BP graph has any `|J_ij| > 0.6` (strong edges) or contains a frustrated cycle (any cycle with an odd number of contradiction edges, detected via DFS at graph construction). Otherwise skip — the variational approximation is reliable on benign graphs. **`maximum` tier:** always invoke Gibbs.

**Chain configuration:**
- **Burn-in:** 2,000 iterations.
- **Samples:** 8,000 iterations after burn-in.
- **Thinning:** 1 (no thinning; correlated samples are fine for marginal estimation).
- **Chain count:** 1 in `balanced`; 4 in `maximum` (for Gelman-Rubin diagnostic).
- **Initial state:** sampled from the TRW-BP marginals (not random; warm-starting cuts effective burn-in).
- **Scan order:** systematic sweep through nodes in claim ID order (deterministic, reproducible).
- **RNG seed:** `sha256(request_id + "gibbs")[:8]` interpreted as uint64 — fully deterministic per request, with multi-chain seeds derived as `seed + chain_index`.

**Validation rule:**
- For each node, compute the Gibbs posterior marginal as the empirical fraction of samples with `T_c = +1`.
- Compare TRW-BP marginal `b_TRW(c)` to Gibbs marginal `b_Gibbs(c)`. If `|b_TRW − b_Gibbs| > 0.10` for any node:
  - Mark that node `bp_validated = False`.
  - Use `b_Gibbs` as the authoritative belief downstream (L5, L7).
- Otherwise, keep `b_TRW` and set `bp_validated = True`.
- **When Gibbs is skipped** (benign graph in `balanced`): `bp_validated = None` — the field is `bool | None` to disambiguate "validated as agreeing" from "not validated because skipped".
- **Multi-chain diagnostic** (`maximum` only): compute Gelman-Rubin R̂ across the 4 chains; require R̂ < 1.05. If R̂ ≥ 1.05, run an additional 4,000 samples and re-check; if still failing, flag in `model_versions` as `gibbs_rhat_warning` and accept the multi-chain mean estimate.

This converts "approximate inference" from a hope into a measured, reproducible property of each individual document.

### Iterative refinement (rigor ≥ `balanced`)

After TRW-BP / Gibbs converges, we identify claims with high *evidence-belief disagreement* using a deterministic predicate over the L4 posterior `b_c` and the local evidence support / refute scores `S_t`, `S_f` (defined in §6 unary potential):

```
def needs_refinement(b_c, S_t, S_f) -> bool:
    # Joint inference moved the belief strongly against support-heavy evidence
    if b_c < 0.30 and S_t >= 2.0 * max(S_f, 0.1):
        return True
    # Joint inference moved the belief strongly toward truth despite
    # refute-heavy evidence
    if b_c > 0.70 and S_f >= 2.0 * max(S_t, 0.1):
        return True
    return False
```

Both directions are flagged because both represent the same underlying signal: PCG joint inference has overridden the local evidence, which is exactly when re-retrieving disambiguating context most often resolves the conflict. (`max(·, 0.1)` floors prevent divide-by-near-zero on claims with near-zero support or refute mass.)

For each flagged claim:
1. Re-issue the retrieval at one tier deeper (e.g., `default` → `max`) plus a query expansion using the surface forms of the contradicting neighbors in the PCG (the top-3 neighbors with strongest `|J_ij|` of opposite sign).
2. Re-run L3 NLI for the new evidence and any affected claim↔claim edges.
3. Re-run TRW-BP on the (potentially updated) graph.

`balanced` allows up to 1 refinement pass per request; `maximum` allows up to 3. Each pass adds ~10–20s on a 10-claim document. Refinement passes are bounded by both per-request count and by a "no-progress" detector (if `max_c |b_c^{pass+1} - b_c^{pass}| < 0.02`, stop early). The `DocumentVerdict.refinement_passes_executed` field exposes the realized count.

### Acknowledged limitations

- TRW-BP does not exactly recover the marginal posteriors of the joint Ising model; LBP and TRW-BP are both approximate. We claim "tractable approximation with tighter convergence guarantees than LBP, an explicit log-partition bound, and Gibbs-sampling validation in `balanced`+ tiers". For pathological graphs the Gibbs check catches the disagreement and substitutes the MCMC marginal.
- Gibbs sampling itself can mix slowly on multimodal posteriors. We use Gelman-Rubin in `maximum` tier (4 chains, R̂ < 1.05 required); in `balanced` we accept the single-chain estimate with a documented caveat.

### Output

Per claim: `PosteriorBelief(p_true, p_false, converged: bool, algorithm: str, iterations: int, edge_count: int, log_partition_bound: float)`.

Document-level: **internal consistency score** = average `KL(b_i || softmax(α_i))` over all nodes — i.e., how much the joint inference moved each node's belief from its evidence-only prior. We map this to [0, 1] via `1 − exp(−KL_avg / τ)` with `τ = log 2` chosen so a one-bit average shift maps to ~0.5. The `τ` choice is documented and kept fixed so the score is comparable across documents; it is not a calibrated probability and is presented in the API as `internal_consistency` (not `internal_consistency_probability`).

### Why this is the publishable contribution

Existing hallucination-detection systems (FActScore, SelfCheckGPT, RefChecker, ChainPoll) treat claims as i.i.d. Treating them as a structured Ising model with calibrated NLI-derived parameters and conformal-wrapped output is, to our knowledge, novel. The architecture admits natural extensions (temporal/causal edges from L1 normalized SPO; soft constraints from external knowledge graphs as additional unary terms) for follow-up work.

### Edge cases

- **Single-claim document**: graph has one node; belief = `softmax(α_c)`; trivial.
- **Disconnected components**: each runs TRW-BP independently; document-level aggregation in L7 still treats all claims jointly via the copula.
- **Non-convergence**: emit `converged=False`, exclude from calibration set (see L5), L6 high-information signal.
- **Numerical underflow**: log-space messages; final beliefs clamped to `[ε, 1−ε]`.

### Tests

- Unit tests on small synthetic graphs (chain, V-structure, triangle, frustrated cycle) with closed-form / exhaustive-enumeration ground truth.
- Convergence test: TRW-BP convergence rate on 30-claim documents from the held-out benchmark set.
- Ablation: edges removed → posteriors equal `softmax(α_c)`, modulo clamping.
- Stress test: adversarial frustrated graphs (every triangle contains an odd number of contradiction edges) — TRW-BP should converge or gracefully fall back to LBP-with-flag.

---

## 7. L5 — Conformal Prediction Layer

### Purpose

Take the joint posterior `b_i(T_i=true) ∈ [0, 1]` from L4 and wrap it with a distribution-free prediction interval, provably calibrated against held-out feedback data per domain × claim-type stratum.

### Module

`src/api/pipeline/conformal/`

### Why conformal vs. Bayesian

Bayesian credible intervals require a prior and are calibrated only if the prior is right — which it never is for foundation-model NLI outputs. Split conformal makes one assumption (exchangeability of test ↔ calibration) and gives finite-sample distribution-free coverage guarantees.

### Algorithm (per-domain split conformal with full re-scoring)

For each domain `d`:

1. **Calibration set** `D_d = {(claim_k, true_label_k) : k=1..n_d}` — claims with human feedback labels (only the labels are stored long-term; *not* the historical posteriors).
2. **Re-scoring under the current model.** Each calibration example is re-scored at deploy time with the *current* model `M_t`, producing fresh `b_k = M_t.posterior(claim_k)`. This is the key methodological commitment: scores `b_k` and the test-time `b_test` come from the *same* model, preserving exchangeability under the i.i.d. assumption on claims. (See §12 for how nightly retraining triggers a full re-scoring stage.)
3. **Nonconformity score**: `s_k = |b_k − 𝟙[true_label_k]|`.
4. **Quantile**: `q̂_d = ⌈(n_d + 1)(1 − α)⌉ / n_d` empirical quantile of `{s_k}`. For α = 0.10, the 90th percentile.
5. **Test-time interval**:
   ```
   P_lower = max(0, b_test − q̂_d)
   P_upper = min(1, b_test + q̂_d)
   ```
   Coverage guarantee: `Pr[true_label ∈ [P_lower, P_upper]] ≥ 1 − α` under exchangeability of test and calibration claims under model `M_t`.

**Why full re-scoring (and not Adaptive Conformal Inference).** ACI (Gibbs & Candès, 2021) is a valid alternative for the online-retraining setting and we mark it as a "Phase 5+ research extension". For Phase 4 we choose full re-scoring because (a) it preserves the standard split-conformal coverage guarantee with no asterisks, (b) the calibration set is small relative to inference cost (target 1k–5k examples per domain × stratum), so re-scoring all of it during nightly maintenance is tractable (~5–15 min per domain on the inference cluster), and (c) the published coverage report (`/api/v2/calibration/report`) can cite a clean theorem.

### Mondrian stratification

Stratify the calibration set by `claim_type × domain` (up to 5 domains × 8 claim types = 40 strata). Per-stratum minimum sample size: `n ≥ 50`.

**Fallback cascade with explicit semantics:**

```
For test claim with stratum (d, t):
  1. If |D_{d,t}| ≥ 50:
       use stratum-specific quantile q̂_{d,t}
       fallback_used = false
       interval = [b - q̂_{d,t}, b + q̂_{d,t}]   ← preserves coverage guarantee
  2. Else if |D_d| ≥ 200:
       use per-domain quantile q̂_d (pooled across claim types in d)
       fallback_used = "domain"
       interval = [b - q̂_d, b + q̂_d]   ← coverage guarantee holds for the pooled distribution, NOT for stratum t specifically
  3. Else:
       use global "general" quantile q̂_general
       fallback_used = "general"
       interval = [b - q̂_general, b + q̂_general]   ← out-of-domain; coverage guarantee does NOT hold
       coverage_target = null in output schema
```

When fallback fires, the API response sets `fallback_used` to the level used and (in the "general" case) sets `coverage_target = null` to signal that no calibration guarantee is being made. The frontend renders this as a "limited calibration" badge. We deliberately do *not* apply post-hoc multiplicative widening — that would invalidate the coverage promise on the rest of the data without delivering a real coverage promise on the fallback case.

### Mixture conformal for soft domain assignments

When L2 returned `{d → w_d}`, we use Tibshirani's weighted exchangeability formulation. Concretely, the mixture nonconformity quantile is computed as the `(1 − α)`-quantile of the pooled distribution of nonconformity scores reweighted by `w_d`:

```
For each calibration example k in domain d':
    weight_k = w_{d'} / |D_{d'}|         (down-weights examples from less-likely domains)
q̂_mix = weighted_quantile_{1-α}({s_k}, weights={weight_k})
```

This preserves the coverage guarantee under the standard exchangeability assumption within each domain (Tibshirani et al., 2019, "Conformal Prediction Under Covariate Shift").

### Behavior on non-converged L4 outputs

Claims with `converged=False` from L4 are **not** added to the calibration set (they don't represent the model's normal output distribution). At test time, a non-converged claim's calibrated verdict is emitted with:

- `coverage_target = null`
- `fallback_used = "non_converged"`
- The point estimate `p_true` is still emitted (best effort)
- `interval = [0.0, 1.0]` (uninformative — accurately reflects that we have no calibration story for non-converged claims)

This is the honest move: no fake guarantees, no post-hoc widening that secretly violates exchangeability.

### Coverage validation in nightly job

After each refit + re-scoring, the job runs k-fold cross-coverage on the calibration set and asserts empirical coverage ∈ **[0.88, 0.92]** for α = 0.10 (symmetric ±0.02 around the target). Outside this range gates the deployment (alarm, no auto-promote). Symmetric bounds because both under-coverage (intervals too narrow → false confidence) and over-coverage (intervals too wide → wasted information) are operationally bad.

### Output

```python
@dataclass(frozen=True)
class CalibratedVerdict:
    p_true: float
    interval_lower: float
    interval_upper: float
    coverage_target: float | None         # null when fallback_used != null
    calibration_set_id: str | None        # null when no real calibration set used
    calibration_n: int
    domain: Domain
    stratum: str                          # "biomedical:quantitative"
    fallback_used: str | None             # null | "domain" | "general" | "non_converged"
```

(This is the L5 internal type; L7 maps it into the public `ClaimVerdict` in §9 with the same field semantics.)

### Edge cases

- Calibration set < 50 → general fallback, `fallback_used=True`, frontend shows "broad interval" badge.
- Perfect-prediction calibration set → `q̂_d = 0`; interval collapses to a point. Logged as suspicious.
- L4 returned `converged=False` → conformal interval auto-widens by `+ 0.1`, capped to [0, 1].

### Tests

- Simulated coverage on synthetic data with known nonconformity distributions.
- Empirical coverage on held-out FEVER + ANLI splits within ±0.02 of target.
- Mixture conformal verified against Tibshirani's reference.

---

## 8. L6 — Active Learning Hook

### Purpose

Decide which claims, at request time, are worth showing to a human reviewer. Cheap, in-line, never blocks the response. Converts the request stream into a high-signal labeling stream that monotonically improves L3 and L5.

### Module

`src/api/pipeline/active_learning/`

### Information-gain metric

```
IG(c) = H(b_c) · interval_width(c) · domain_uncertainty_weight(c)

H(b) = -b · log b - (1-b) · log(1-b)
interval_width = P_upper - P_lower (from L5)
domain_uncertainty_weight = inverse function of |D_d|; higher for sparse domains
```

Multiplicative form: a claim is queued only if it's uncertain *and* labeling it would actually help.

### Queue policy

- `IG > 0.3` → enqueue for review (Postgres-backed, TTL 30d).
- Cap per-document at 3 enqueued claims.
- Per-domain quotas (target 60% of new labels into under-represented strata).
- Deduplication: cosine similarity > 0.95 to existing queue items merges.

### Out-of-band

L6 is fire-and-forget: posts to internal SQS queue (or in-memory queue locally) with claim + provenance + verdict; returns immediately. Background worker materializes the queue row. Adds ≤ 5ms to the request.

### Edge cases

- Queue full → drop with metric increment, never block request.
- Feedback worker down → claims accumulate on SQS; durable.
- Same claim queued twice → idempotent on `claim.normalized_form` hash.

### Tests

- Unit tests on IG calculation.
- Integration test: `b=0.5, interval_width=0.4, domain=social` always enqueues.
- Load test: 1000 RPS doesn't cause request-side latency regression.

---

## 9. L7 — Output Assembly

### Purpose

Take per-claim `CalibratedVerdict[]` from L5, the L4 internal consistency signal, and full provenance from L1–L3, and produce a single immutable `VerificationResult` ready for the API and frontend.

### Module

`src/api/pipeline/assembly/`

### Document-level aggregation: Gaussian copula joint probability

Naively averaging per-claim probabilities discards the joint structure. We compute the proper joint probability `P(all claims true)` using a Gaussian copula:

```
For each claim c:
    z_c = Φ⁻¹(b_c)               # probit transform of L4 posterior

Build correlation matrix R from L3 claim↔claim NLI:
    raw[i,j] = entail(c_i, c_j) − contradict(c_i, c_j)   ∈ [−1, +1]
    raw[i,i] = 1
    R = nearest_psd(raw)         # eigenvalue clipping: any eigenvalue < ε → ε

Document score = Φ_R(z_1, z_2, ..., z_N)
```

`Φ_R` is the multivariate Gaussian CDF with correlation `R`. This is the natural Gaussian-copula joint probability that all `N` claims are true.

**Computation.** For `N ≤ 10` the multivariate Gaussian CDF has fast, accurate quasi-Monte Carlo estimators (Genz algorithm via SciPy `scipy.stats.multivariate_normal.cdf`). For `N > 10` we approximate by Monte Carlo: draw 10,000 samples from `N(0, R)`, count the fraction where every coordinate `Z_c > z_c`. Sample size 10,000 gives standard error < 0.005 on the resulting probability — adequate for our use.

**Document-level interval.** Same Monte Carlo procedure applied to the conformal lower / upper bounds gives `[document_lower, document_upper]`:

```
document_lower = Φ_R(Φ⁻¹(P_lower_1), ..., Φ⁻¹(P_lower_N))
document_upper = Φ_R(Φ⁻¹(P_upper_1), ..., Φ⁻¹(P_upper_N))
```

This is a valid lower / upper bound on the true joint probability under the copula model (monotonicity of the multivariate CDF).

**PSD enforcement.** `R` constructed from raw NLI scores is generally not positive-semi-definite. We project to the nearest PSD matrix in Frobenius norm via eigenvalue clipping: compute `R = U Λ Uᵀ`, set every `λ_i < ε` (we use `ε = 10⁻⁴`) to `ε`, and reconstruct. The clipped matrix is then re-normalized to keep unit diagonal. The Frobenius distance from the raw matrix is reported in `model_versions` for auditability.

**Why joint-truth probability and not weakest-link.** The joint probability is the answer to the question users actually want to ask: "what is the probability this entire document is correct?" Weakest-link `min(b_c)` would discard the correlation structure and double-penalize correlated claims. The copula-derived joint truthfully encodes "if these two claims are mutually entailing, count them once; if they contradict, the joint is forced low." This is the methodologically correct aggregator.

### Internal consistency score

The L4 average `KL(b_i || φ_i)`, normalized to [0, 1] via `1 - exp(-KL_avg)`. Surfaced as a top-level field.

### Output schema (the new API contract)

```python
class ClaimVerdict(BaseModel):
    claim: Claim
    p_true: float
    interval: tuple[float, float]
    coverage_target: float | None  # null when fallback_used != null

    # Provenance & explainability
    domain: Domain
    domain_assignment_weights: dict[Domain, float]
    supporting_evidence: list[Evidence]
    refuting_evidence: list[Evidence]
    pcg_neighbors: list[ClaimEdge]
    nli_self_consistency_variance: float
    bp_validated: bool                 # True if Gibbs check agreed with TRW-BP

    # Active learning
    information_gain: float
    queued_for_review: bool

    # Calibration metadata
    calibration_set_id: str | None     # null when no real calibration set used
    calibration_n: int
    fallback_used: str | None          # null | "domain" | "general" | "non_converged"

    model_config = {"frozen": True}


class DocumentVerdict(BaseModel):
    document_score: float              # Φ_R(Φ⁻¹(b_1), ..., Φ⁻¹(b_N))
    document_interval: tuple[float, float]
    internal_consistency: float        # 1 - exp(-KL_avg/τ), τ=ln 2
    claims: list[ClaimVerdict]
    decomposition_coverage: float
    processing_time_ms: float
    rigor: Literal["fast", "balanced", "maximum"]
    refinement_passes_executed: int
    pipeline_version: str = "ohi-v2.0"
    model_versions: dict[str, str]     # includes nli, decomposer, router, calibration
    request_id: UUID

    model_config = {"frozen": True}
```

### Edge cases

- Empty document → `DocumentVerdict(document_score=1.0, claims=[], internal_consistency=1.0)`.
- All claims `UNVERIFIABLE` in L1 → document_score = mean of conformal lower bounds on uniform prior.
- Single claim → copula degenerates to that claim's `b_c`.

### Tests

- Golden tests on the exact JSON shape (regression-protected).
- Copula correctness on synthetic graphs where joint min has a closed form.
- Round-trip serialization tests.

---

## 10. API surface

### Endpoints

```
POST /api/v2/verify
POST /api/v2/verify/stream         (SSE; layer-completion events)
GET  /api/v2/verdict/{request_id}
POST /api/v2/feedback
GET  /api/v2/calibration/report
GET  /api/v2/health/{live|ready|deep}
```

The `/api/v1/*` namespace is deleted.

### `POST /api/v2/verify`

Request:
```json
{
  "text": "string, required, max 50_000 chars",
  "context": "string | null",
  "domain_hint": "general | biomedical | legal | code | social | null",
  "options": {
    "rigor": "fast | balanced | maximum",
    "tier": "local | default | max",
    "max_claims": 100,
    "include_pcg_neighbors": true,
    "include_full_provenance": true,
    "self_consistency_k": null,
    "coverage_target": 0.90
  },
  "request_id": "uuid | null"
}
```

`rigor` selects the work envelope (see §2). `self_consistency_k` overrides the rigor-tier default (null = use the tier default). `domain_hint` raises the per-domain router prior by 0.2 before softmax; per-claim routing still wins on outliers.

Response 200: `DocumentVerdict` (full schema in §9).

**Concrete example response (truncated for length):**

```json
{
  "request_id": "f4c1...e2",
  "pipeline_version": "ohi-v2.0",
  "model_versions": {
    "decomposer": "openai/gpt-4o-mini-2025-XX-XX",
    "domain_router": "ohi-router-v2-2026-04-16",
    "nli_general": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli@head-v3",
    "calibration_general:quantitative": "calib-2026-04-16-r1"
  },
  "document_score": 0.74,
  "document_interval": [0.61, 0.84],
  "internal_consistency": 0.83,
  "decomposition_coverage": 0.92,
  "processing_time_ms": 87341,
  "rigor": "balanced",
  "refinement_passes_executed": 1,
  "claims": [
    {
      "claim": { "id": "...", "text": "Einstein was born in 1879.", "claim_type": "temporal", ... },
      "p_true": 0.96,
      "interval": [0.91, 0.99],
      "coverage_target": 0.90,
      "domain": "general",
      "domain_assignment_weights": { "general": 0.94, "biomedical": 0.06 },
      "supporting_evidence": [ { "id": "...", "source_uri": "https://en.wikipedia.org/...", ... } ],
      "refuting_evidence": [],
      "pcg_neighbors": [
        { "neighbor_claim_id": "...", "edge_type": "entail", "edge_strength": 0.81 }
      ],
      "nli_self_consistency_variance": 0.012,
      "bp_validated": null,
      "information_gain": 0.04,
      "queued_for_review": false,
      "calibration_set_id": "calib-2026-04-16-r1",
      "calibration_n": 1247,
      "fallback_used": null
    }
  ]
}
```

Response 4xx: `400` invalid input, `429` rate-limit (`Retry-After`).

Response 5xx:
- `503` degraded (body includes `degraded_layers: ["L3", "L1.retrieval"]`).
- `503` budget exhausted (body includes `"OHI public budget exhausted, resets in Yh"`).
- `504` only on infrastructure timeout (e.g., upstream LLM provider unreachable for > 5 min). The pipeline does **not** itself impose a wall-clock cap — long requests run to completion under the rigor tier they accepted.

### `POST /api/v2/verify/stream` (SSE)

Same request shape. Events are emitted in pipeline order (one event per layer completion + one per claim verdict, with the claim verdicts emitted *after* PCG propagation, which is when they actually exist):

```
event: decomposition_complete
data: {"claim_count": 12, "estimated_total_ms": 90000}

event: claim_routed
data: {"claim_id": "...", "domain": "biomedical", "weights": {...}}
# (one per claim, emitted as L2 completes)

event: nli_complete
data: {"claim_evidence_pairs_scored": 360, "claim_pair_pairs_scored": 66}

event: pcg_propagation_complete
data: {"iterations": 6, "converged": true, "algorithm": "TRW-BP",
       "internal_consistency": 0.91, "gibbs_validated": true}

event: refinement_pass_complete
data: {"pass": 1, "claims_re_retrieved": 2, "marginal_max_change": 0.07}
# (zero or more, depending on rigor + flagged claims)

event: claim_verdict
data: <ClaimVerdict>   # one per claim, in claim ID order

event: document_verdict
data: <full DocumentVerdict>
```

Each `claim_verdict` event payload is a strict subset of the `ClaimVerdict` schema in §9 (no new fields, no different shapes).

**Tentative recommendation:** ship in Phase 1 alongside the synchronous `/verify`. Cheap on top of the async pipeline; high frontend value.

### `POST /api/v2/feedback`

Request:
```json
{
  "request_id": "uuid",
  "claim_id": "uuid",
  "label": "true | false | unverifiable | abstain",
  "labeler": {
    "kind": "user | expert | adjudicator",
    "id": "string, opaque",
    "credential_level": 0
  },
  "rationale": "string, max 2000 chars",
  "evidence_corrections": [
    { "evidence_id": "...", "correct_classification": "supports | refutes | irrelevant" }
  ]
}
```

Response 202: `{"feedback_id": "...", "queued": true}`.

Idempotent on `(request_id, claim_id, labeler.id)`.

### `GET /api/v2/calibration/report`

Public, unauthenticated, lightly rate-limited. The auditability surface.

```json
{
  "report_date": "2026-04-16",
  "global_coverage_target": 0.90,
  "domains": {
    "biomedical": {
      "calibration_n": 1247,
      "empirical_coverage": 0.913,
      "interval_width_p50": 0.18,
      "interval_width_p95": 0.34,
      "strata": { "biomedical:quantitative": {...} }
    }
  }
}
```

**Tentative recommendation:** also commit a static HTML snapshot to S3 nightly for human consumption — most marketable transparency artifact, costs almost nothing.

### `GET /api/v2/health/deep`

Per-layer status + latency + model versions + calibration freshness. Used by ops + the frontend status page.

### Versioning policy

- **Breaking changes** → new major (`/api/v3/*`) with 6-month deprecation.
- **Additive changes** → ship in `/api/v2/*` (frozen pydantic + `extra="ignore"` on consumers).
- **Model version changes** → bump `model_versions` in response, not the API. Callers can pin via `?model_version=...` for benchmark reproducibility.

### Explicitly NOT in v2.0

- Token-level streaming verification.
- Auto-correction `/rewrite` endpoint.
- `GET /verify`.
- Multipart uploads / file ingestion.
- WebSocket transport.

---

## 11. Open access, rate limiting, traffic protection

### Auth posture

- **No JWT, no Supabase, no user identity at the edge.** All public endpoints are unauthenticated.
- **Single internal bearer token** for trusted callers (MCP server, benchmark suite, CI, ingestion pipeline). Long random string in `Authorization: Bearer`. Bypasses public rate limits, observable in metrics.
- **Optional `X-OHI-Labeler-Token`** header on `/feedback` for expert/adjudicator labelers. Without it, feedback enters untrusted tier.

### Layered defense

```
┌─────────────────────────────────────────────────────────────┐
│ L0 · Edge (CloudFront + AWS WAF)                            │
│   Bot management, geo rate limits, L7 DDoS, smuggling       │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│ L1 · ALB / Nginx (per-IP token bucket)                      │
│   /verify, /verify/stream:    10 req/min/IP, burst 20       │
│   /feedback:                   3 req/min/IP, burst 5        │
│   /verdict/{id}, /calibration: 60 req/min/IP, burst 100     │
│   /health/*:                   unlimited                    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│ L2 · FastAPI middleware (sliding window + cost-aware)       │
│   Adaptive shedding when downstream queue depth > θ         │
│   503 Retry-After: jittered exponential backoff             │
│   Per-IP cost accounting                                    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│ L3 · Backend pressure (the actual ceiling)                  │
│   Bounded inference worker queue (e.g., 50 deep)            │
│   Daily $ ceiling on LLM/GPU spend                          │
│   Ceiling hit → /verify returns 503 "budget exhausted"      │
│   /verdict/{id}, /calibration, /health continue serving     │
└─────────────────────────────────────────────────────────────┘
```

The "downtimes are okay" stance is implemented as the L3 cost ceiling. Deterministic, auditable, transparent. No silent quality degradation, no runaway AWS bill.

### Caching as rate-limit multiplier

- **Tier 1 (Redis, ~50ms cold)**: `sha256(text + options)`, TTL 7d, full `DocumentVerdict`. Hit rate target ≥ 30% steady state.
- **Tier 2 (in-process LRU on worker, ~1ms)**: same key, TTL 1h, capacity 1000.

Cache hits don't count against the cost ceiling. **Default policy: cache hits *do* count against the per-IP rate limit** (1× weight, same as a cold request) — this deters cache-hammering DOS. Operators can flip this default via config if their threat model differs.

### Data retention and privacy

OHI accepts unauthenticated, public submissions of arbitrary text. We minimize what is stored and for how long, and surface that policy in `/api/v2/health/deep`:

| Data | Retention | Storage shape |
|---|---|---|
| Raw input text on `/verify` | **Not persisted by default.** Only the `sha256(text + options)` cache key is stored. Operator can opt in to retention via config flag for debugging. | Hash only (Redis cache key + Postgres `verifications.text_hash`) |
| `DocumentVerdict` and `ClaimVerdict[]` | 30 days in Postgres (for `/verdict/{request_id}` retrieval), then deleted | Postgres `verifications`, `claim_verdicts` |
| Claim text (extracted by L1) | Retained as long as the verdict is retained (30d), then deleted with the verdict | `claim_verdicts.claim_jsonb` |
| Feedback submissions on `/feedback` | 90 days in `feedback_pending`; promoted rows in `calibration_set` retained indefinitely (anonymized — labeler ID is hashed before write) | Postgres |
| `ip_hash` on feedback | 30 days, salted hash; not the raw IP | Postgres `feedback_pending.ip_hash` |
| Calibration set entries | Indefinite (these are the model's learned ground truth). Right-to-erasure requests can void specific (claim_id, labeler_id_hash) tuples. | Postgres `calibration_set` |
| S3 model artifacts | Indefinite for currently-deployed model versions; older versions pruned after 1 year | S3 |

**GDPR posture.** Because we accept EU traffic without auth and process potentially personal data submitted in `text` fields, we treat raw text as PII-suspect by default → not persisted. Hashes are not reversible. The frontend submission page surfaces a privacy notice clearly stating that submitted text is not retained. Calibration entries do not contain raw input text — they contain the *extracted claim* (a normalized fact statement), which is generally not PII; if a claim happens to contain PII (e.g., "John Smith of 42 Elm St lives in Berlin"), the right-to-erasure flow can void it.

**`/feedback` PII surface.** The `rationale` field is stored verbatim for 90 days, then dropped. Submitters are warned in the UI that rationale text is human-readable by adjudicators.

### Trust tiers for `/feedback`

`labeler.kind` in the request payload, validated against the labeler-token if one is presented:

| `labeler.kind` | Trigger | What happens |
|---|---|---|
| `user` (default) | No `X-OHI-Labeler-Token` header | `feedback_pending` queue. Promoted to `calibration_set` only after **3 concordant distinct labelers** within 30d (per the §12 SQL) and only when no other label has also reached 3 (otherwise the claim goes to `disputed_claims_queue` for adjudicator review). |
| `expert` (a.k.a. "trusted") | `X-OHI-Labeler-Token` matches a known expert-tier token | Writes directly to `calibration_set` with `source_tier='trusted'`. Single label sufficient. |
| `adjudicator` | Reserved bearer token (handful of accounts, manually issued) | Writes directly to `calibration_set` with `source_tier='adjudicator'`. Resolves disputed claims; overrides any prior consensus or trusted label. |

### `/feedback` anti-spam (algorithmic)

- Per-IP submission rate (3/min) + per-claim rate (10 untrusted labels/claim/day) at L1.
- Lexical-similarity dedupe on rationale (cosine ≥ 0.95).
- Honeypot fields in JSON schema.
- Optional Cloudflare Turnstile if abuse becomes measurable.

### Implications for infra sub-project

- AWS WAF rules need authoring (geo, bot, rate).
- Daily cost ceiling enforced at the worker layer; surface via CloudWatch + alarm.
- No Cognito, no Auth0 → ~30% Terraform module count saved.
- Internal bearer token in AWS Secrets Manager, rotated via `terraform apply`.

---

## 12. Active learning loop, feedback store, retraining

### Storage layout

```
Postgres (RDS prod, SQLite local dev)
─────────────────────────────────────
verifications        id, text_hash, request_id, document_verdict_jsonb,
                     model_versions_jsonb, created_at

claim_verdicts       id, verification_id, claim_jsonb, calibrated_verdict_jsonb,
                     information_gain, queued_for_review, created_at

feedback_pending     id, claim_id, label, labeler_kind, labeler_id_hash,
                     rationale, evidence_corrections_jsonb, ip_hash, created_at

calibration_set      id, claim_id, true_label, source_tier, n_concordant,
                     adjudicated_by, calibration_set_partition (domain:stratum),
                     posterior_at_label_time, model_versions_at_label_time,
                     created_at, retired_at NULL

retraining_runs      id, layer ('L3.nli' | 'L5.conformal' | 'L1.source_cred'),
                     started_at, completed_at, status, metrics_jsonb,
                     artifact_s3_uri, deployed_at NULL

S3 (MinIO local)
────────────────
ohi-artifacts/
├── nli-heads/{domain}/{date}/head.safetensors
├── calibration/{domain}/{stratum}/{date}/quantiles.json
├── source-cred/{date}/weights.json
├── retraining-reports/{run_id}.html
└── eval-snapshots/{date}/benchmark-results.jsonl
```

### Consensus & promotion

Scheduled job (every 15 min) with explicit handling of disagreement and per-labeler distinctness:

```sql
-- Step 1: aggregate distinct labelers per (claim, label)
WITH per_label AS (
    SELECT
        claim_id,
        label,
        COUNT(DISTINCT labeler_id_hash) AS n_distinct_labelers
    FROM feedback_pending
    WHERE created_at > now() - interval '30 days'
      AND labeler_kind = 'user'                         -- untrusted only
    GROUP BY claim_id, label
),
-- Step 2: detect disagreement (any claim with ≥3 labelers on more than one label)
disputed AS (
    SELECT claim_id
    FROM per_label
    WHERE n_distinct_labelers >= 3
    GROUP BY claim_id
    HAVING COUNT(DISTINCT label) >= 2
),
-- Step 3: clean consensus = exactly one label has ≥3 distinct labelers, no other label has ≥3
clean_consensus AS (
    SELECT pl.claim_id, pl.label, pl.n_distinct_labelers
    FROM per_label pl
    WHERE pl.n_distinct_labelers >= 3
      AND pl.claim_id NOT IN (SELECT claim_id FROM disputed)
)
INSERT INTO calibration_set (
    claim_id, true_label, source_tier, n_concordant,
    calibration_set_partition, posterior_at_label_time,
    model_versions_at_label_time, created_at
)
SELECT
    cc.claim_id, cc.label, 'consensus', cc.n_distinct_labelers,
    cv.calibrated_verdict_jsonb->>'stratum',
    (cv.calibrated_verdict_jsonb->>'p_true')::float,
    cv.calibrated_verdict_jsonb->'model_versions',
    now()
FROM clean_consensus cc
JOIN claim_verdicts cv ON cv.claim_id = cc.claim_id
ON CONFLICT (claim_id) DO NOTHING;

-- Step 4: log disputed claims for adjudicator queue
INSERT INTO disputed_claims_queue (claim_id, first_disputed_at)
SELECT claim_id, now() FROM disputed
ON CONFLICT (claim_id) DO NOTHING;
```

`labeler_id_hash` is required to be present and unique-per-labeler for the feedback to count toward consensus. Submissions from the same `ip_hash` within a short window are still permitted (different humans can share an IP) but they all count as the same `labeler_id_hash` if no opaque labeler ID is provided.

**Adjudication for disputed claims.** Disputed claims sit in `disputed_claims_queue`. An adjudicator reviewing the queue submits a `kind=adjudicator` label which overrides any prior consensus. Adjudicators are the small set of trusted labelers issued bearer tokens out-of-band.

**Trusted/adjudicator labels** bypass the consensus filter entirely and write directly to `calibration_set` with appropriate `source_tier` ('trusted' or 'adjudicator').

### Nightly retraining DAG (02:00 UTC)

```
1. Snapshot calibration set → S3
2. NLI head fine-tune with online EWC regularization
3. Re-score the entire calibration set under the new NLI head + new
   PCG potentials → fresh nonconformity scores
4. Per-domain conformal refit using the re-scored nonconformity values
   (parallel across strata)
5. Evidence-correction propagation:
     - For each evidence_correction in feedback last 7 days, add a synthetic
       (claim, evidence, label) NLI training pair to the next training run,
       and update source-credibility prior (Stage 9) accordingly.
6. Eval on held-out benchmarks (frozen 1000-claim regression suite + each
   domain's primary benchmark)
7. Coverage validation gate (k-fold cross-coverage on re-scored set)
8. Promote (atomic S3 manifest swap) or rollback (alarm + report)
9. Source-credibility adjustment (separate, conservative; ±0.05 max delta/night)
```

The critical change vs. the original sketch: **Stage 3 (re-scoring) is what preserves split-conformal exchangeability.** Calibration nonconformity scores must come from the same model that will produce test-time scores; re-scoring after each retrain guarantees this. Re-scoring 5,000 calibration examples per domain takes ~3–8 min per domain on the inference cluster — well within the nightly window. The eval and coverage gate (Stages 6 + 7) operate on the fresh scores, so a model that improves benchmark F1 but degrades calibration coverage is correctly rejected.

### Online EWC regularization (Stage 2 detail)

We use **online EWC** (Schwarz et al., 2018, "Progress & Compress"), not EWC++ (Chaudhry et al., 2018). The two are sometimes conflated; we follow the online-EWC formulation: a single quadratic anchor regularizing toward the previous model's parameters, weighted by an exponential moving average of the Fisher information matrix.

- Build training set: trusted + adjudicator + consensus-promoted labels last 90d. Weight each example by `max(0.5, 1 / max(0.1, posterior_at_label_time))` — up-weight examples where the previous model was wrong (low posterior on the true label), with floor and cap to prevent extreme weights.
- Loss:
  ```
  L = CrossEntropy(predictions, labels)
    + λ · Σ_θ F_θ · (θ − θ_prev)²
  ```
  - `θ_prev` = parameters of the currently-deployed head.
  - `F_θ` = exponentially-moving-average Fisher diagonal: `F_t = γ · F_{t-1} + diag(∇log p_θ_prev(D_new))`. EMA factor `γ = 0.95` (so old training data still anchors but recent data dominates).
  - `λ = 100` initial; tuned per domain based on observed forgetting on the regression suite.
- Train 1–3 epochs; early stopping on a 10% holdout.
- Save head + Fisher diagonal to S3 (the Fisher diagonal becomes `F_{t-1}` for the next run).

### Coverage gate (Stage 7 detail)

For each domain × stratum:
- k-fold cross-coverage on the *re-scored* calibration set.
- If `coverage ∉ [0.88, 0.92]` (symmetric ±0.02 around α=0.10 target) → **fail the gate**, no auto-promote.
- Frozen 1000-claim regression suite: F1 drop > 1pt on any domain → **fail the gate**.

Symmetric coverage bounds because over-coverage (intervals too wide → wasted information) and under-coverage (intervals too narrow → false confidence) are both operationally bad.

### Source-credibility adjustment (Stage 9)

- For each `(source, domain)` pair with sufficient feedback signal: empirical accuracy vs. ground truth, *plus* the count of `evidence_corrections` from Stage 5 that flipped the source's classification on a labeled claim.
- Update via Beta-Bernoulli posterior (Jeffreys prior `Beta(0.5, 0.5)`, conservative).
- **Capped change per night: ±0.05** from previous value (no source's credibility can swing wildly overnight).
- The full `(source, domain) → credibility` matrix is committed to S3 with the run_id.

### Replay buffer for ablation

Every retraining run produces a fully reproducible artifact: input snapshot, model weights, training config, eval results, all committed to S3 with `run_id`. Enables historical state reproduction and paper experiments.

### Operational stance

- **Worker hot-reload**: inference workers poll `model_versions/current.json` in S3 every 60s, atomically swap heads at next request boundary. No restart.
- **Manual rollback**: `ohi-models rollback --to <date>` resets `current.json` to a previous manifest.
- **Dry-run**: full DAG with `--dry-run` against staging data; promotes nothing, generates report. Used pre-flip and for paper experiments.

### Why this is publishable

- **Reproducibility**: every claim verdict in production traces to exact NLI head + conformal quantiles + source priors that produced it.
- **EWC++ for hallucination NLI** is, to our knowledge, novel — most fact-checking systems either retrain from scratch (forgetting risk) or never retrain (drift risk).
- **Auditable coverage gate**: the system literally cannot deploy a model that fails empirical coverage. A methodological commitment competitors don't make.

---

## 13. Phased rollout

5 phases. Each ships a working system with measurable gains over the previous.

### Phase 0 — Research foundation (prerequisite)

- Wire up benchmark harness for FActScore, TruthfulQA, HaluEval, PubMedQA, LegalBench-Entailment, custom code-fact eval, LIAR/MultiFC/ClimateFEVER.
- Capture v1 baseline numbers on every benchmark.
- Stand up Postgres + S3 (locally; AWS comes with infra sub-project).
- Deliverable: `benchmark_results/v1_baseline_2026-04-XX.jsonl`.

### Phase 1 — Foundation (replaces v1)

- L1 refactor (chunked decomposition, source-credibility prior).
- L7 output assembly with new `DocumentVerdict` / `ClaimVerdict` schema.
- Naive L5 (single global conformal quantile from synthesized labels). **In Phase 1, the `coverage_target` field in `ClaimVerdict` is emitted as `null` and `fallback_used = "general"` for all claims** — we do not claim conformal coverage until Phase 3 stands up real per-domain calibration sets. This is the honest default; we'd rather emit `null` than a fake `0.90`.
- Delete `oracle.py`, `scorer.py`, all v1 routes/schemas.
- Rate limiting middleware (Section 11).
- `/verify/stream` SSE (tentative recommendation: ship now).
- Data retention policy enforced (Section 11): no raw text persisted by default.
- Deliverable: parity-or-better with v1 on FActScore + TruthfulQA, new contract live, no auth.

### Phase 2 — Reasoning core (the publishable contribution)

- L3 NLI cross-encoder with K=10 self-consistency by default (rigor=`balanced`); K=3 in `fast`, K=30 in `maximum` (single shared head, no per-domain yet).
- L4 PCG with TRW-BP primary inference + Gibbs MCMC sanity check on flagged graphs in `balanced`+, always in `maximum`.
- Iterative PCG refinement loop (1 pass in `balanced`, 3 in `maximum`).
- Internal consistency surfaced as top-level signal.
- Refit conformal quantiles against L4 posteriors.
- Deliverable: ≥3pt F1 over Phase 1 on FActScore + TruthfulQA + HaluEval; coverage validated.

### Phase 3 — Domain awareness (product expansion)

- L2 router with 5 verticals.
- Per-domain NLI heads (cold-started by fine-tuning Phase 2 head per domain).
- Per-domain calibration sets (synthesized cold start; sprint label drive at launch).
- DomainAdapter wired through L1/L3/L5.
- Deliverable: ≥5pt F1 on at least 4 of 5 domain benchmarks vs. Phase 2 general-only baseline.

### Phase 4 — Active learning closure (the moat)

- L6 information-gain hook in request path.
- `/feedback` endpoint with consensus filter + trust tiers + adjudicator queue for disputed claims.
- Nightly retraining DAG (all 9 stages).
- Online EWC for NLI head fine-tuning (with EMA Fisher diagonal — see §12).
- Full re-scoring of calibration set after each retrain (preserves split-conformal exchangeability).
- Coverage validation gate + auto-rollback.
- Deliverable: end-to-end loop demonstrably improves F1 by ≥10% on prioritized-review subset within 2 weeks of feedback collection.

### Coordination with parallel sub-projects

- **Frontend** can begin against Phase 1 contract (mocked responses fine until Phase 2). Real PCG visualizations require Phase 2; domain badges require Phase 3; feedback UI requires Phase 4.
- **Infrastructure** can begin in parallel with Phase 1 (Terraform for VPC, ALB, Postgres, S3, basic compute). GPU-bound resources for Phase 2 are the tail dependency. Cost ceiling design (Section 11) is the main thing infra needs up front.

---

## 14. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| NLI compute cost (especially `maximum` tier with K=30 + ensemble) becomes prohibitive at scale | High | Per-IP rate limit + cost ceiling at the global level (§11). Cache (Redis + LRU) eats popular requests for free. `rigor=fast` is always available as the default for unauthenticated traffic; `maximum` is opt-in. |
| TRW-BP converges to a wrong fixed point on adversarial graphs | Medium | Gibbs sampling sanity check in `balanced`+ tiers catches divergence > 0.10 between TRW-BP and MCMC marginals; Gibbs result is then used. LBP fallback if TRW-BP doesn't converge. Frustrated-graph stress tests in CI. |
| Gibbs sampler mixes slowly on multimodal posteriors | Medium | `maximum` tier runs 4 chains with Gelman-Rubin R̂ < 1.05 gate. `balanced` accepts single-chain estimate with documented caveat in `model_versions`. |
| Iterative refinement loop runs forever on pathological inputs | Low | Hard cap on refinement passes (1 in `balanced`, 3 in `maximum`) + no-progress detector (stop when marginals move < 0.02 between passes). |
| Calibration cold start — no labels for new domains | High | Synthesize from benchmark dev splits (FEVER, ANLI, SciFact). Phase 1 emits `coverage_target=null` and `fallback_used="general"` honestly until per-domain sets exist. Adjudicator labeling sprint at Phase 3 launch. |
| Active learning poisoning — coordinated bad-faith feedback | Medium | 3-concordant **distinct labelers** consensus (the SQL in §12 enforces `COUNT(DISTINCT labeler_id_hash)`), disagreement detection with adjudicator queue, per-IP rate limit, lexical dedupe, coverage gate auto-rejects models that shift suspiciously. |
| Conformal exchangeability silently broken by retrained models | High | **Stage 3 of the nightly DAG re-scores the entire calibration set under the new model**, ensuring nonconformity scores and test-time scores share a model. Coverage gate fails the deployment if empirical coverage drifts outside [0.88, 0.92]. |
| Domain mis-routing on edge cases | Low | Soft assignment + log-and-review for low-confidence routings. Misroutes degrade to general (wider intervals, not wrong answers). |
| Cost ceiling fires during a launch / press moment | Medium | Cost ceiling is runtime config, not constant. Generous pre-launch provision. Pre-launch dashboard shows budget consumption rate. |
| Single shared encoder hits GPU memory with all 5 heads + ensemble loaded | Low | Heads loaded lazily; only active-document domains pinned. Encoder ~1.7GB FP16; heads ~5MB each. `maximum` tier ensemble (RoBERTa) adds ~1GB; both fit in A100 40GB. |
| Catastrophic forgetting in nightly NLI retraining | Medium | Online EWC regularization + frozen 1000-claim regression suite as hard gate. F1 drop > 1pt → no auto-promote. |
| Bi-encoder pre-filter prunes lexically-dissimilar contradictions in `fast` tier | Medium | `fast` mode documents this limitation; `balanced` (default) and `maximum` do full quadratic claim↔claim NLI for N ≤ 40 / always respectively. Per-domain thresholds (lower for social/legal). |
| Provenance hash collisions on near-duplicate evidence | Low | SHA-256 + content normalization. Non-issue at expected volumes. |
| Copula PSD enforcement loses information when raw NLI correlation matrix is severely non-PSD | Low | Eigenvalue-clipping projection minimizes Frobenius distance; the distance is reported in `model_versions` for auditability. If the projection moves the matrix significantly, that itself is a signal of inconsistent NLI judgments and is logged. |
| GDPR / right-to-erasure on calibration set entries containing extracted PII-bearing claims | Medium | §11 retention table + erasure endpoint (admin-only) that voids `(claim_id, labeler_id_hash)` rows and triggers a calibration refit. Documented in operations runbook. |

---

## 15. Open questions and tentative recommendations

These are decisions baked into this spec as tentative recommendations, to be reconfirmed during implementation planning.

1. **NLI base model.** *Tentative:* `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` as the shared encoder + per-domain heads; revisit in Phase 3 if domain F1 underperforms (likely candidates: SciFive / BiomedBERT-NLI for biomedical, possibly LegalBERT for legal). `maximum` rigor tier ensembles RoBERTa-MNLI alongside.
2. **LLM for decomposition.** *Tentative:* keep the existing `LLMProvider` port; defer local-vs-API decision to the infra sub-project. Algorithm is provider-agnostic.
3. **Phase deliverable accounting.** *Tentative:* Phase 2 measured against Phase 1; Phase 3 against Phase 2; Phase 4 against Phase 3. Plus a final §1 success criterion measured against the v1 baseline captured in Phase 0.
4. **Calibration cold-start synthesis.** *Tentative:* synthesize from benchmark dev sets for Phase 1/2; supervised labeling sprint with adjudicators for Phase 3 launch (≥200 labeled per domain).
5. **Public calibration dashboard.** *Tentative:* yes — also commit static HTML snapshot to S3 nightly. Most marketable transparency artifact, near-zero cost.
6. **Streaming endpoint priority.** *Tentative:* ship `/verify/stream` SSE in Phase 1 (cheap on top of async pipeline, big UX win for the frontend sub-project).
7. **Benchmark snapshot cadence.** *Tentative:* nightly during Phase 0–4, weekly after stabilization.
8. **Adaptive Conformal Inference (ACI) as alternative to nightly re-scoring.** *Tentative:* not in scope for v2.0. Full nightly re-scoring is simpler, gives a clean coverage theorem, and is operationally tractable. Revisit ACI (Gibbs & Candès 2021) for a Phase 5+ research extension if calibration sets grow beyond 50k examples per domain.
9. **Domain router validation set composition.** *Tentative:* 200 hand-labeled claims per domain (1,000 total) labeled by adjudicators during Phase 3 cold-start sprint. Sample from real production traffic captured in Phase 1/2 to control for selection bias against benchmark distributions.
10. **Bi-encoder claim-pair threshold per domain.** *Tentative:* 0.45 default, 0.30 for `social` and `legal` (lexically-dissimilar contradictions are common). Revalidated nightly against the regression suite.

---

## 16. Testing strategy

- **Unit tests**: every module, ≥80% line coverage gate. Existing pytest setup.
- **Integration tests**: end-to-end on a fixture document at the boundary of each phase deliverable.
- **Benchmark regression**: every PR touching pipeline code runs the full benchmark suite in CI; F1 drop > 1pt blocks merge.
- **Calibration validation**: nightly job's coverage gate also runs in CI on a frozen calibration snapshot.
- **Load tests**: p95 / p99 latency, cost-per-request — k6 or Locust against synthetic-traffic generator.
- **Chaos tests**: kill NLI worker mid-request (assert graceful degradation), drop calibration Postgres connection (assert in-memory cache fallback), inject malformed feedback (assert 400 + no DB write).
- **Adversarial tests**: known-poisoning patterns against `/feedback` (assert quarantine), known cyclic claim patterns against PCG (assert convergence or graceful flag).

---

## 17. Documentation deliverables

- `docs/algorithm/v2-overview.md` — public-facing layered pipeline overview.
- `docs/algorithm/v2-calibration.md` — methodology + reproducibility guide for the conformal layer.
- `docs/api/v2-reference.md` — full OpenAPI spec, regenerated from FastAPI.
- `docs/operations/active-learning-runbook.md` — inspect, override, rollback the loop.
- README updates: replace v1 mental model with v2.

(A research paper drafted from the PCG + calibrated NLI + active learning contribution is a *follow-on artifact* outside this spec's scope. It will live in a separate `research/` directory or external venue and is not on the implementation critical path.)

---

## 18. Glossary and references

### Glossary

- **Atomic claim**: a single, verifiable, propositional statement extracted from input text.
- **Conformal prediction**: distribution-free uncertainty quantification giving finite-sample coverage guarantees under exchangeability.
- **Coverage**: empirical fraction of test instances whose true label falls inside the prediction interval.
- **Domain adapter**: per-domain configuration bundle (NLI head, source priors, calibration set).
- **Online EWC**: Elastic Weight Consolidation in the online (sequential) setting using an exponential moving average of the Fisher information matrix as the regularization anchor (Schwarz et al., 2018, "Progress & Compress"). *Distinct from* "EWC++" (Chaudhry et al., 2018), which uses a different formulation; we use online EWC.
- **Information gain (L6)**: composite score quantifying how much labeling a claim would reduce model uncertainty.
- **Internal consistency score**: document-level signal derived from L4 BP message magnitude; high = reconciliation effort = potential incoherence.
- **Loopy BP (LBP)**: belief propagation run on a cyclic graph; converges to a stationary point of Bethe free energy when it converges, but on dense or strongly-connected graphs may oscillate or converge to non-Bethe-stationary points (Murphy/Weiss/Jordan 1999). Used as the L4 fallback inference algorithm.
- **TRW-BP**: tree-reweighted belief propagation (Wainwright/Jaakkola/Willsky 2005). Convex relaxation of LBP using uniformly-distributed spanning-tree edge weights `ρ_e`; convergent under broad conditions; provides an upper bound on the log-partition function. The L4 primary inference algorithm.
- **Gibbs sampling**: MCMC algorithm that samples from a joint distribution by repeatedly resampling each variable conditional on the others. Used in L4 as a sanity check on TRW-BP marginals — the variational approximation is validated against actual posterior samples per request.
- **Mondrian conformal**: split conformal stratified by a categorical attribute (here, claim_type × domain).
- **NLI**: Natural Language Inference; classification of (premise, hypothesis) into entail/contradict/neutral.
- **PCG**: Probabilistic Claim Graph; the L4 graphical model.
- **Self-consistency (L3)**: sampling multiple stochastic forward passes with input perturbations to estimate epistemic uncertainty.
- **Trust tier**: feedback labeler classification {untrusted, trusted, adjudicator}.

### References

**Belief propagation and graphical models**

- Yedidia, J. S., Freeman, W. T., & Weiss, Y. (2003). *Understanding Belief Propagation and Its Generalizations*. Exploring Artificial Intelligence in the New Millennium.
- Murphy, K. P., Weiss, Y., & Jordan, M. I. (1999). *Loopy belief propagation for approximate inference: An empirical study*. UAI. (Documents LBP non-convergence and convergence-to-wrong-fixed-point on cyclic graphs.)
- Wainwright, M. J., Jaakkola, T. S., & Willsky, A. S. (2005). *A new class of upper bounds on the log partition function*. IEEE Transactions on Information Theory. (Tree-reweighted belief propagation.)
- Geman, S., & Geman, D. (1984). *Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images*. IEEE PAMI. (Gibbs sampling.)

**Conformal prediction**

- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*.
- Papadopoulos, H. (2008). *Inductive Conformal Prediction: Theory and Application to Neural Networks*. (Original split / inductive conformal formulation.)
- Tibshirani, R. J., Foygel Barber, R., Candès, E. J., & Ramdas, A. (2019). *Conformal Prediction Under Covariate Shift*. NeurIPS. (Weighted exchangeability — basis for our mixture conformal.)
- Gibbs, I., & Candès, E. J. (2021). *Adaptive Conformal Inference Under Distribution Shift*. NeurIPS. (Cited as Phase 5+ research extension; we use full re-scoring instead.)

**Continual learning**

- Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS. (Original EWC.)
- Schwarz, J., et al. (2018). *Progress & Compress: A scalable framework for continual learning*. ICML. (Online EWC — the formulation we use.)
- Chaudhry, A., et al. (2018). *Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence*. ECCV. (EWC++ — distinct from online EWC; cited for completeness.)

**Hallucination detection and fact verification**

- Min, S., et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision*. EMNLP.
- Manakul, P., et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*. EMNLP.
- Lin, S., et al. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. ACL.
- Thorne, J., et al. (2018). *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL.

**NLI checkpoints**

- Laurer, M., van Atteveldt, W., Casas, A., & Welbers, K. (2024). *Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI*. (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` checkpoint.)

---

*End of design specification. Next step: spec review loop, then user approval, then implementation planning via the writing-plans skill.*
