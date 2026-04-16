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

1. **A Probabilistic Claim Graph (PCG) with loopy belief propagation.** Claims are not verified independently. Entailment and contradiction edges between claims (extracted by a calibrated NLI cross-encoder) propagate evidence between related claims. Internal contradictions in the input are detected as a first-class hallucination signal.
2. **Per-domain split-conformal prediction with Mondrian stratification.** Every claim verdict is a calibrated probability `P(true)` plus a `[lower, upper]` 90% prediction interval, validated to maintain ≤10% miscoverage on held-out data per domain × claim-type stratum.
3. **An open active learning loop with EWC++ regularized retraining.** Public users submit feedback through a `/feedback` endpoint with a 3-concordant consensus filter; nightly jobs refit per-domain conformal calibration and fine-tune NLI heads with elastic weight consolidation to prevent catastrophic forgetting.
4. **Five-domain specialization.** General, biomedical, legal, code, and social each have their own NLI head, source-credibility prior, and calibration set, dispatched by a per-claim soft router.
5. **Open-by-default API.** No authentication; rate-limit + traffic-protection at the edge; a deterministic daily cost ceiling that gracefully translates load spikes into 503 with `Retry-After`. A public `/calibration/report` endpoint exposes empirical coverage statistics for auditability.

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
- **G2. Joint claim reasoning.** A Probabilistic Claim Graph propagates evidence between related claims via NLI-derived entailment and contradiction edges. Internal contradictions are detected and surfaced as a hallucination signal.
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
- **Latency.** Simple inputs (1–3 claims) < 60s; typical (5–15 claims) ≤ 60s; complex (>30 claims) ≤ 120s hard cap. P95 within budget per tier.
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

### Latency budget (typical 10-claim document)

| Layer | Budget | Notes |
|---|---|---|
| L1 (decompose + retrieve) | 8s | LLM decomp + parallel evidence pulls |
| L2 (domain routing) | 1s | Lightweight classifier (~50ms × 10 claims, batched to ~500ms) |
| L3 (NLI, K=3 self-consistency) | 25s | GPU/API bound; batched cross-encoder |
| L4 (PCG + 8 BP iterations) | 5s | CPU-bound, NumPy vectorized |
| L5 (conformal) | <1s | Lookup + arithmetic |
| L6 + L7 | 1s | Assembly + queue post |
| **Total** | **~40s** | Within 60s typical budget |

Complex inputs (>30 claims) trade more time on L3 + L4 within the 120s hard cap.

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

- **Base encoder:** DeBERTa-v3-large (435M parameters), single shared instance, kept resident on GPU.
- **Per-domain heads:** small classification heads (~5MB each) that swap on the encoder. Heads loaded lazily; only domains touched by the active document are pinned.
- **Recommendation (tentative):** revisit per-domain base encoders if Phase 3 domain F1 underperforms. SciFive / BiomedBERT-NLI are the most likely candidates for biomedical.

### Self-consistency (K=3)

For each (claim, evidence) pair, do K=3 stochastic forward passes with input perturbation:

1. **Lexical paraphrase of the claim** via a small T5-paraphrase model (cached aggressively per claim).
2. **Evidence sentence-window slide** (±1 sentence around the central retrieved span).
3. **Premise/hypothesis order swap** (claim-as-hypothesis vs. claim-as-premise — DeBERTa-MNLI is asymmetric).

Final distribution = mean of softmax over K passes; variance = sample variance. The variance feeds L4 as the local-potential confidence inverse-weight.

### Claim ↔ claim NLI

The most innovative piece. Naively O(N²); optimized via:

1. **Bi-encoder pre-filter** with `sentence-transformers/all-mpnet-base-v2`; prune pairs below `adapter.claim_pair_relatedness_threshold` (default 0.45).
2. **Top-K cap**: never run cross-encoder on more than `min(N², 5N)` pairs.
3. **Symmetry caching**: compute (i, j) only with i < j, then a cheap second softmax for the reverse direction.

For a 30-claim document this yields ~150 cross-encoder calls instead of 870.

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

### Construction

1. **Nodes** = claims. Each node `c` has a binary latent variable `T_c ∈ {true, false}`.

2. **Local (unary) potentials** `φ_c(T_c)` from L3 claim↔evidence NLI:

   ```
   φ_c(T_c=true)  = ∑_e w_e · entail(c, e) + (1 - w_e) · 0.5
   φ_c(T_c=false) = ∑_e w_e · contradict(c, e) + (1 - w_e) · 0.5

   where w_e = source_credibility(e) · temporal_decay(e) · (1 - NLI_variance(c, e))
   ```

   Sources we trust more, that are fresher, and that NLI is more confident about, get more weight. Then normalize: `φ_c(·) ← φ_c(·) / Σ φ_c`.

3. **Pairwise (edge) potentials** `ψ_{ij}(T_i, T_j)` from L3 claim↔claim NLI:

   - **Entailment edge** (`entail(c_i, c_j) > 0.7`):
     ```
     ψ_{ij}(t, t) = entail
     ψ_{ij}(t, f) = 1 - entail   (penalty)
     ψ_{ij}(f, t) = neutral
     ψ_{ij}(f, f) = neutral
     ```
   - **Contradiction edge** (`contradict(c_i, c_j) > 0.7`): symmetric penalty matrix encoding "if c_i true then c_j false".
   - **Weak/neutral edges**: pruned (no edge).

4. **Edge strength** is multiplied by `(1 - variance(NLI(c_i, c_j)))` so noisy edges contribute less.

### Inference: Loopy Belief Propagation

The graph is generally cyclic. We use loopy BP with damping factor δ = 0.5 (standard for stability):

```
m_{i→j}^{k+1}(T_j) = (1-δ) · m_{i→j}^k(T_j)
                   + δ · Σ_{T_i} φ_i(T_i) · ψ_{ij}(T_i, T_j) ·
                                  ∏_{n ∈ N(i)\{j}} m_{n→i}^k(T_i)

Belief: b_i(T_i) ∝ φ_i(T_i) · ∏_{n ∈ N(i)} m_{n→i}(T_i)
```

Convergence: max message change < 10⁻³ or 8 iterations, whichever first. Messages computed in log-space to avoid underflow; final beliefs renormalized and clamped to `[ε, 1-ε]`.

### Why loopy BP

- **Exact** (junction tree): exponential in tree-width, intractable for cyclic graphs.
- **Variational** (mean-field): cleaner math, but ignores edge correlations — exactly what we want to capture.
- **Loopy BP**: converges to a stationary point of Bethe free energy (Yedidia et al., 2003); strong empirical results on similar structured-prediction tasks; interpretable enough to debug.

### Output

Per claim: `PosteriorBelief(p_true, p_false, converged: bool, iterations: int, edge_count: int)`.

Document-level: **internal consistency score** = average `KL(b_i || φ_i)` over all nodes. High KL = BP forced large updates from local evidence = something is internally inconsistent.

### Why this is the publishable contribution

Existing hallucination-detection systems (FActScore, SelfCheckGPT, RefChecker, ChainPoll) treat claims as i.i.d. Treating them as a structured graphical model with calibrated NLI edges and conformal-wrapped output is, to our knowledge, novel. The architecture admits natural extensions (temporal/causal edges from L1, soft constraints from external knowledge graphs) for follow-up work.

### Edge cases

- **Single-claim document**: PCG degenerates to local potential = posterior; trivial.
- **Disconnected components**: each runs BP independently.
- **Non-convergence**: after 8 iterations, return current beliefs and flag `converged=False`. Becomes a high-information signal for L6.
- **Numerical underflow**: log-space messages; final beliefs clamped.

### Tests

- Unit tests on small synthetic graphs (chain, V-structure, triangle) where exact posteriors are computable.
- Convergence test on 30-claim documents.
- Ablation tests (no edges → posteriors should equal local potentials, modulo normalization).

---

## 7. L5 — Conformal Prediction Layer

### Purpose

Take the joint posterior `b_i(T_i=true) ∈ [0, 1]` from L4 and wrap it with a distribution-free prediction interval, provably calibrated against held-out feedback data per domain × claim-type stratum.

### Module

`src/api/pipeline/conformal/`

### Why conformal vs. Bayesian

Bayesian credible intervals require a prior and are calibrated only if the prior is right — which it never is for foundation-model NLI outputs. Split conformal makes one assumption (exchangeability of test ↔ calibration) and gives finite-sample distribution-free coverage guarantees.

### Algorithm (per-domain split conformal)

For each domain `d`:

1. **Calibration set** `D_d = {(claim_k, true_label_k, b_k) : k=1..n_d}` — claims with human feedback labels, with their L4 posterior `b_k` recorded *at feedback time*.
2. **Nonconformity score**: `s_k = |b_k - 𝟙[true_label_k]|`.
3. **Quantile**: `q̂_d = ⌈(n_d + 1)(1 - α)⌉ / n_d` empirical quantile of `{s_k}`. For α=0.10, the 90th percentile.
4. **Test-time interval**:
   ```
   P_lower = max(0, b_test - q̂_d)
   P_upper = min(1, b_test + q̂_d)
   ```
   Coverage guarantee: `Pr[true_label ∈ [P_lower, P_upper]] ≥ 1 - α` under exchangeability.

### Mondrian stratification

Stratify the calibration set by `claim_type × domain` (5 × 8 = up to 40 strata). Each requires `n ≥ 50` for reliability; below threshold falls back to per-domain; below per-domain falls back to global `general` (logged + flagged).

### Mixture conformal for soft domain assignments

When L2 returned `{d → w_d}`, use Tibshirani's weighted exchangeability:
```
q̂_mix = quantile_{1-α}( ∑_d w_d · F_d^{-1} )
```
where `F_d` is the empirical CDF of nonconformity scores in domain `d`.

### Coverage validation in nightly job

After each refit, the job runs k-fold cross-coverage on the calibration set itself and asserts empirical coverage ∈ [α - 0.02, α + 0.05]. Outside this range gates the deployment (alarm, no auto-promote).

### Output

```python
@dataclass(frozen=True)
class CalibratedVerdict:
    p_true: float
    interval_lower: float
    interval_upper: float
    coverage_target: float = 0.90
    calibration_set_id: str
    calibration_n: int
    domain: Domain
    stratum: str               # "biomedical:quantitative"
    fallback_used: bool
```

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

### Document-level aggregation: Gaussian copula

Naively averaging per-claim probabilities discards the joint structure. We use a Gaussian copula:

1. Map each `b_c ∈ (0, 1)` to a Gaussian via probit: `z_c = Φ⁻¹(b_c)`.
2. Build the empirical claim-claim correlation matrix `R` from L3 claim↔claim entailment scores (entail correlations positive, contradict negative).
3. Sample `Z ~ N(0, R)`, transform back: `b' = Φ(Z)`.
4. Document score = expectation of `min(b'_c)` (joint truth requires *all* claims to be true; min is the natural aggregator).

This properly accounts for contradictory claims that can't both be true. Naïve averaging would give a high score to a self-contradicting document; copula correctly penalizes it.

### Internal consistency score

The L4 average `KL(b_i || φ_i)`, normalized to [0, 1] via `1 - exp(-KL_avg)`. Surfaced as a top-level field.

### Output schema (the new API contract)

```python
class ClaimVerdict(BaseModel):
    claim: Claim
    p_true: float
    interval: tuple[float, float]
    coverage_target: float = 0.90

    # Provenance & explainability
    domain: Domain
    domain_assignment_weights: dict[Domain, float]
    supporting_evidence: list[Evidence]
    refuting_evidence: list[Evidence]
    pcg_neighbors: list[ClaimEdge]
    nli_self_consistency_variance: float

    # Active learning
    information_gain: float
    queued_for_review: bool

    # Calibration metadata
    calibration_set_id: str
    calibration_n: int
    fallback_used: bool

    model_config = {"frozen": True}


class DocumentVerdict(BaseModel):
    document_score: float
    document_interval: tuple[float, float]
    internal_consistency: float
    claims: list[ClaimVerdict]
    decomposition_coverage: float
    processing_time_ms: float
    pipeline_version: str = "ohi-v2.0"
    model_versions: dict[str, str]
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
    "tier": "local | default | max",
    "max_claims": 50,
    "include_pcg_neighbors": true,
    "include_full_provenance": true,
    "self_consistency_k": 3,
    "coverage_target": 0.90
  },
  "request_id": "uuid | null"
}
```

`domain_hint` raises the per-domain prior by 0.2 before softmax; per-claim routing still wins on outliers.

Response 200: `DocumentVerdict`.

Response 4xx: `400` invalid input, `429` rate-limit (`Retry-After`).

Response 5xx:
- `503` degraded (body includes `degraded_layers: ["L3", "L1.retrieval"]`).
- `503` budget exhausted (body includes `"OHI public budget exhausted, resets in Yh"`).
- `504` exceeded 120s hard cap.

### `POST /api/v2/verify/stream` (SSE)

Same request shape; response is a sequence of named events:

```
event: decomposition_complete
data: {"claim_count": 12, "estimated_total_ms": 38000}

event: claim_routed
data: {"claim_id": "...", "domain": "biomedical", "weights": {...}}

event: claim_verdict
data: {"claim_id": "...", "p_true": 0.87, "interval": [0.78, 0.94], ...}

event: pcg_propagation_complete
data: {"iterations": 6, "converged": true, "internal_consistency": 0.91}

event: document_verdict
data: <full DocumentVerdict>
```

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

Cache hits don't count against the cost ceiling. (Configurable: do count against per-IP rate limit by default to deter cache-hammering DOS.)

### Trust tiers for `/feedback`

| Tier | Trigger | What happens |
|---|---|---|
| **Untrusted** (default) | No labeler token | `feedback_pending` queue. Enters calibration set after 3 concordant labels in 30d. |
| **Trusted** | `X-OHI-Labeler-Token` matches known token | Enters calibration set immediately. |
| **Adjudicator** | Reserved bearer token | Resolves disputes, overrides consensus, submits gold labels. |

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

Postgres trigger or scheduled job (every 15 min):

```sql
INSERT INTO calibration_set (claim_id, true_label, source_tier, n_concordant, ...)
SELECT
  claim_id, label, 'consensus', COUNT(*) AS n_concordant, ...
FROM feedback_pending
WHERE created_at > now() - interval '30 days'
GROUP BY claim_id, label
HAVING COUNT(*) >= 3
ON CONFLICT (claim_id) DO NOTHING;
```

Trusted/adjudicator labels override consensus rows on arrival.

### Nightly retraining DAG (02:00 UTC)

```
1. Snapshot calibration set → S3
2. Per-domain conformal refit (parallel across strata)
3. NLI head fine-tune with EWC++ regularization
4. Eval on held-out benchmarks
5. Coverage validation gate (k-fold cross-coverage)
6. Promote (atomic S3 manifest swap) or rollback (alarm)
7. Source-credibility adjustment (separate, conservative)
```

### EWC++ regularization (Stage 3 detail)

- Build training set: trusted + consensus-promoted labels last 90d, weighted by `1 / posterior_at_label_time` (up-weight previous-model errors).
- Loss: `L = CrossEntropy + λ · Σ_θ F_θ · (θ - θ_prev)²`, where `F_θ` is the online Fisher information matrix from the previous run (Schwarz et al., 2018 — exponential moving average of Fishers).
- Train 1–3 epochs; early stopping on 10% holdout.

### Coverage gate (Stage 5 detail)

For each domain × stratum:
- k-fold cross-coverage on the calibration set.
- If `coverage ∉ [0.88, 0.95]` → **fail the gate**, no auto-promote.
- Frozen 1000-claim regression suite: F1 drop > 1pt on any domain → **fail the gate**.

### Source-credibility adjustment (Stage 7)

- For each `(source, domain)` pair with sufficient feedback signal: empirical accuracy vs. ground truth.
- Update via Beta-Bernoulli posterior (Jeffreys prior, conservative).
- **Capped change per night: ±0.05** from previous value.

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
- Naive L5 (single global conformal quantile from synthesized labels).
- Delete `oracle.py`, `scorer.py`, all v1 routes/schemas.
- Rate limiting middleware (Section 11).
- `/verify/stream` SSE (tentative recommendation: ship now).
- Deliverable: parity-or-better with v1 on FActScore + TruthfulQA, new contract live, no auth.

### Phase 2 — Reasoning core (the publishable contribution)

- L3 NLI cross-encoder with K=3 self-consistency (single shared head, no per-domain yet).
- L4 PCG with loopy belief propagation.
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
- `/feedback` endpoint with consensus filter + trust tiers.
- Nightly retraining DAG (Stages 1–7).
- EWC++ for NLI head fine-tuning.
- Coverage validation gate + auto-rollback.
- Deliverable: end-to-end loop demonstrably improves F1 by ≥10% on prioritized-review subset within 2 weeks of feedback collection.

### Coordination with parallel sub-projects

- **Frontend** can begin against Phase 1 contract (mocked responses fine until Phase 2). Real PCG visualizations require Phase 2; domain badges require Phase 3; feedback UI requires Phase 4.
- **Infrastructure** can begin in parallel with Phase 1 (Terraform for VPC, ALB, Postgres, S3, basic compute). GPU-bound resources for Phase 2 are the tail dependency. Cost ceiling design (Section 11) is the main thing infra needs up front.

---

## 14. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| NLI compute cost balloons under scale (O(N²) cross-encoder if naive) | High | Bi-encoder pre-filter + top-K cap + relatedness threshold (§5). K capped at 3, not 10. Heavy paraphrase caching. |
| PCG fails to converge on pathological cyclic claims | Medium | Max 8 iterations + damping 0.5 + log-space messages. Non-convergence becomes L6 high-information signal, not failure. Conformal interval auto-widens. |
| Calibration cold start — no labels for new domains | High | Synthesize from existing benchmark dev splits (FEVER, ANLI, SciFact). Frontend "limited calibration" badge while `n < 200`. General fallback for sparse strata. |
| Active learning poisoning — coordinated bad-faith feedback | Medium | 3-concordant consensus + per-IP rate limit + lexical dedupe + adjudicator override. Coverage gate fails immediately if poison shifts calibration. |
| Domain mis-routing on edge cases | Low | Soft assignment + log-and-review for low-confidence routings. Misroutes degrade to general (wider intervals, not wrong answers). |
| Cost ceiling fires during a launch / press moment | Medium | Cost ceiling is runtime config, not constant. Generous pre-launch provision. Pre-launch dashboard shows budget consumption rate. |
| Single shared encoder hits GPU memory with all 5 heads loaded | Low | Heads loaded lazily; only active-document domains pinned. Encoder ~1.7GB FP16; heads ~5MB each. |
| Catastrophic forgetting in nightly NLI retraining | Medium | EWC++ regularization + frozen 1000-claim regression suite as hard gate. F1 drop > 1pt → no auto-promote. |
| Provenance hash collisions on near-duplicate evidence | Low | SHA-256 + content normalization. Non-issue at expected volumes. |
| PCG produces non-probabilistic edge cases (Yedidia counter-examples) | Low | Final beliefs renormalized; clamp to [ε, 1-ε]. Documented limitation. |

---

## 15. Open questions and tentative recommendations

These are decisions baked into this spec as tentative recommendations, to be reconfirmed during the spec review and implementation planning.

1. **NLI base model.** *Tentative:* one base encoder (DeBERTa-v3-large) with domain-specialized heads; revisit in Phase 3 if domain F1 underperforms (likely candidates: SciFive / BiomedBERT-NLI for biomedical).
2. **LLM for decomposition.** *Tentative:* keep the existing `LLMProvider` port; defer local-vs-API decision to the infra sub-project. Algorithm is provider-agnostic.
3. **Initial benchmarks vs. innovation tax.** *Tentative:* Phase 2 deliverable measured against Phase 1 (tighter, more defensible target).
4. **Calibration cold-start synthesis.** *Tentative:* synthesize from benchmark dev sets for Phase 1/2; supervised labeling sprint with adjudicators for Phase 3 launch.
5. **Public calibration dashboard.** *Tentative:* yes — also commit static HTML snapshot to S3 nightly. Most marketable transparency artifact, near-zero cost.
6. **Streaming endpoint priority.** *Tentative:* ship `/verify/stream` SSE in Phase 1 (cheap on top of async pipeline, big UX win for the frontend sub-project).
7. **Benchmark snapshot cadence.** *Tentative:* nightly during Phase 0–4, weekly after stabilization.

Override any of these during the spec review pass.

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
- `docs/algorithm/v2-pcg-paper.md` — drafted research paper on PCG + calibrated NLI contribution.
- `docs/algorithm/v2-calibration.md` — methodology + reproducibility guide for the conformal layer.
- `docs/api/v2-reference.md` — full OpenAPI spec, regenerated from FastAPI.
- `docs/operations/active-learning-runbook.md` — inspect, override, rollback the loop.
- README updates: replace v1 mental model with v2.

---

## 18. Glossary and references

### Glossary

- **Atomic claim**: a single, verifiable, propositional statement extracted from input text.
- **Conformal prediction**: distribution-free uncertainty quantification giving finite-sample coverage guarantees under exchangeability.
- **Coverage**: empirical fraction of test instances whose true label falls inside the prediction interval.
- **Domain adapter**: per-domain configuration bundle (NLI head, source priors, calibration set).
- **EWC++**: Elastic Weight Consolidation with online Fisher information (exponential moving average); regularizes against catastrophic forgetting.
- **Information gain (L6)**: composite score quantifying how much labeling a claim would reduce model uncertainty.
- **Internal consistency score**: document-level signal derived from L4 BP message magnitude; high = reconciliation effort = potential incoherence.
- **Loopy BP**: belief propagation run on a cyclic graph; converges to stationary point of Bethe free energy.
- **Mondrian conformal**: split conformal stratified by a categorical attribute (here, claim_type × domain).
- **NLI**: Natural Language Inference; classification of (premise, hypothesis) into entail/contradict/neutral.
- **PCG**: Probabilistic Claim Graph; the L4 graphical model.
- **Self-consistency (L3)**: sampling multiple stochastic forward passes with input perturbations to estimate epistemic uncertainty.
- **Trust tier**: feedback labeler classification {untrusted, trusted, adjudicator}.

### References

- Yedidia, J. S., Freeman, W. T., & Weiss, Y. (2003). *Understanding Belief Propagation and Its Generalizations*.
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a Random World*.
- Tibshirani, R. J., Foygel Barber, R., Candès, E. J., & Ramdas, A. (2019). *Conformal Prediction Under Covariate Shift*. NeurIPS.
- Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS.
- Schwarz, J., et al. (2018). *Progress & Compress: A scalable framework for continual learning* (introduces EWC++).
- Min, S., et al. (2023). *FActScore: Fine-grained Atomic Evaluation of Factual Precision*. EMNLP.
- Manakul, P., et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection*. EMNLP.
- Lin, S., et al. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods*. ACL.
- Thorne, J., et al. (2018). *FEVER: a Large-scale Dataset for Fact Extraction and VERification*. NAACL.

---

*End of design specification. Next step: spec review loop, then user approval, then implementation planning via the writing-plans skill.*
