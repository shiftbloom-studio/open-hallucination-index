"""
LLM-Based Claim Decomposer
==========================

Extracts atomic claims from unstructured text using LLM.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING
from uuid import uuid4

from interfaces.decomposition import ClaimDecomposer
from interfaces.llm import LLMMessage
from models.entities import Claim, ClaimType
from pipeline.decomposer_chunking import chunk_text
from pipeline.decomposer_normalization import (
    normalize_claim_text,
    normalize_date_expression,
    normalize_number_expression,
    resolve_entity_qids,
)

if TYPE_CHECKING:
    from interfaces.llm import LLMProvider

logger = logging.getLogger(__name__)

# System prompt for claim extraction
DECOMPOSITION_PROMPT = """You are a claim extraction system.
Your task is to decompose text into atomic, verifiable factual claims.

For each claim, extract:
1. The claim text (a single factual statement)
2. Subject (the entity the claim is about)
3. Predicate (the relationship or property)
4. Object (the value or related entity)
5. Claim type (one of: subject_predicate_object, temporal, quantitative,
   comparative, causal, definitional, existential, unclassified)

Rules:
- Each claim should be atomic (one fact per claim)
- Skip opinions, questions, and subjective statements
- Normalize entity names (e.g., "he" -> the actual person's name if known from context)
- For dates, use ISO format where possible

Output as a JSON object with a "claims" array:
```json
{
    "claims": [
        {
            "text": "The claim as a complete sentence",
            "subject": "Entity name",
            "predicate": "relationship",
            "object": "value or entity",
            "claim_type": "type",
            "confidence": 0.0-1.0
        }
    ]
}
```

Only output the JSON object, nothing else."""


class DecompositionError(Exception):
    """Error during claim decomposition."""

    pass


class LLMClaimDecomposer(ClaimDecomposer):
    """
    LLM-based claim decomposition service.

    Uses a language model to extract atomic claims from text.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_claims: int = 50,
    ) -> None:
        """
        Initialize the decomposer.

        Args:
            llm_provider: LLM provider for text processing.
            max_claims: Maximum claims to extract per request.
        """
        self._llm = llm_provider
        self._max_claims = max_claims

    async def decompose(self, text: str) -> list[Claim]:
        """
        Decompose text into a list of atomic claims.

        Args:
            text: Unstructured input text to analyze.

        Returns:
            List of extracted claims with structured representation.
        """
        return await self.decompose_with_context(text)

    async def decompose_with_context(
        self,
        text: str,
        context: str | None = None,
        max_claims: int | None = None,
    ) -> list[Claim]:
        """
        Decompose text with additional context for disambiguation.

        Task 1.3: inputs longer than ~7k characters (≈1800 tokens) are
        chunked with co-reference overlap; per-chunk claims are merged,
        deduplicated by normalized form, then each claim passes through
        the normalization pipeline (date ISO, number SI, entity QID).

        Args:
            text: Unstructured input text to analyze.
            context: Optional context (e.g., document title, topic).
            max_claims: Optional limit on number of claims to extract.

        Returns:
            List of extracted claims, normalized and deduplicated.
        """
        if not text or not text.strip():
            return []

        limit = max_claims or self._max_claims

        chunks = chunk_text(text, max_tokens=1800, overlap_tokens=200)
        if len(chunks) <= 1:
            # Single-chunk path: no merge/dedup work, keeps old behaviour
            raw = await self._decompose_chunk(text, context=context, limit=limit)
            merged = raw
        else:
            logger.info("Chunked decomposition: %d chunks (%d chars total)", len(chunks), len(text))
            per_chunk: list[list[Claim]] = []
            for ch in chunks:
                chunk_claims = await self._decompose_chunk(ch.text, context=context, limit=limit)
                # Re-base source_span to the original text offsets
                per_chunk.append(
                    [self._rebase_span(c, origin_offset=ch.start) for c in chunk_claims]
                )
            merged = self._dedupe_merge(per_chunk)[:limit]

        # Normalization pass: stamp normalized_form, try date/number
        # normalization, resolve entity QIDs (stub until MCP wired).
        return [await self._normalize_claim_obj(c) for c in merged]

    async def _decompose_chunk(self, text: str, *, context: str | None, limit: int) -> list[Claim]:
        """Single-LLM-call decomposition for one chunk. Preserves the
        original v1 behaviour for one shot."""
        user_content = f"Extract up to {limit} factual claims from the following text:\n\n{text}"
        if context:
            user_content = f"Context: {context}\n\n{user_content}"

        system_content = DECOMPOSITION_PROMPT
        messages = [
            LLMMessage(role="system", content=system_content),
            LLMMessage(role="user", content=user_content),
        ]

        try:
            response = await self._llm.complete(
                messages,
                temperature=0.0,
                max_tokens=2048,
                json_mode=True,
            )

            claims = self._parse_response(response.content, text)
            logger.info(f"Extracted {len(claims)} claims from text")
            return claims[:limit]

        except Exception as e:
            # Use warning level - fallback mode works fine, this is expected during startup
            logger.warning(f"Claim decomposition unavailable: {e}. Using single-claim fallback.")
            # Fallback: Treat the whole text as one unclassified claim
            # This ensures robustness even if LLM is down
            return [
                Claim(
                    id=str(uuid4()),
                    text=text.strip(),
                    subject="Unknown",
                    predicate="is",
                    object="Unknown",
                    claim_type=ClaimType.UNCLASSIFIED,
                    confidence=0.5,
                    context=context,
                )
            ]

    def _parse_response(self, response: str, original_text: str) -> list[Claim]:
        """Parse LLM response into Claim objects."""
        # Extract JSON from response (handle markdown code blocks)
        # Allow for unclosed code blocks (truncation) or standard blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", response, re.IGNORECASE)
        json_str = ""
        if json_match:
            candidate = json_match.group(1).strip()
            if candidate.startswith("{") or candidate.startswith("["):
                json_str = candidate

        if not json_str:
            json_str = response.strip()

        data = None

        # Try parsing full response first
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: extract the first JSON object/array substring
            start_obj = json_str.find("{")
            end_obj = json_str.rfind("}") + 1
            start_arr = json_str.find("[")
            end_arr = json_str.rfind("]") + 1

            candidate = None
            if start_obj != -1 and end_obj > start_obj:
                candidate = json_str[start_obj:end_obj]
            elif start_arr != -1 and end_arr > start_arr:
                candidate = json_str[start_arr:end_arr]

            if candidate:
                try:
                    data = json.loads(candidate)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse LLM response as JSON: {e}")
                    return self._fallback_decomposition(original_text)
            else:
                logger.warning("Failed to locate JSON in LLM response")
                return self._fallback_decomposition(original_text)

        # Normalize to list of claim dicts
        if isinstance(data, dict):
            data = data.get("claims", [])
        if not isinstance(data, list):
            logger.warning("LLM response JSON did not contain a claims list")
            return self._fallback_decomposition(original_text)

        claims = []
        for item in data:
            if not isinstance(item, dict):
                continue

            claim_text = item.get("text", "").strip()
            if not claim_text:
                continue

            # Map claim type
            claim_type_str = item.get("claim_type", "unclassified").lower()
            claim_type = self._map_claim_type(claim_type_str)

            # Find source span in original text
            source_span = None
            text_lower = original_text.lower()
            claim_lower = claim_text.lower()
            idx = text_lower.find(claim_lower[:50])  # Match first 50 chars
            if idx != -1:
                source_span = (idx, idx + len(claim_text))

            claims.append(
                Claim(
                    id=uuid4(),
                    text=claim_text,
                    claim_type=claim_type,
                    subject=item.get("subject"),
                    predicate=item.get("predicate"),
                    object=item.get("object"),
                    source_span=source_span,
                    confidence=float(item.get("confidence", 0.8)),
                    normalized_form=self._normalize_claim(claim_text),
                )
            )

        return claims

    def _map_claim_type(self, type_str: str) -> ClaimType:
        """Map string to ClaimType enum."""
        mapping = {
            "subject_predicate_object": ClaimType.SUBJECT_PREDICATE_OBJECT,
            "temporal": ClaimType.TEMPORAL,
            "quantitative": ClaimType.QUANTITATIVE,
            "comparative": ClaimType.COMPARATIVE,
            "causal": ClaimType.CAUSAL,
            "definitional": ClaimType.DEFINITIONAL,
            "existential": ClaimType.EXISTENTIAL,
        }
        return mapping.get(type_str, ClaimType.UNCLASSIFIED)

    def _normalize_claim(self, text: str) -> str:
        """Canonical text form for matching (delegates to
        ``decomposer_normalization.normalize_claim_text``)."""
        return normalize_claim_text(text)

    # ---- Task 1.3 helpers ------------------------------------------------

    @staticmethod
    def _rebase_span(claim: Claim, *, origin_offset: int) -> Claim:
        """Shift a claim's source_span back into the parent document's
        coordinate system. Chunked decomposition extracts each chunk
        independently so spans land relative to the chunk; we translate
        them to parent-document offsets before returning to the caller.
        """
        if claim.source_span is None or origin_offset == 0:
            return claim
        lo, hi = claim.source_span
        return claim.model_copy(update={"source_span": (lo + origin_offset, hi + origin_offset)})

    def _dedupe_merge(self, per_chunk_claims: list[list[Claim]]) -> list[Claim]:
        """Merge per-chunk claim lists, deduping by normalized_form.

        Phase 1 dedup is exact-string match on the canonical form; cosine-
        similarity dedup via the embedding model lands in Phase 2 along
        with the bi-encoder infrastructure (Task 2.3).
        """
        seen: dict[str, Claim] = {}
        order: list[str] = []
        for chunk_claims in per_chunk_claims:
            for c in chunk_claims:
                key = c.normalized_form or normalize_claim_text(c.text)
                if key in seen:
                    continue
                seen[key] = c
                order.append(key)
        return [seen[k] for k in order]

    async def _normalize_claim_enrich(self, claim: Claim) -> Claim:
        """Attach normalized_form + date/number/entity metadata to a
        freshly-extracted claim.

        Writes:
          - ``normalized_form`` — canonical text (always populated).
          - ``entity_qids`` — Wikidata QIDs (empty dict when MCP stub).
        Returns a new frozen Claim; the original is not mutated.
        """
        updates: dict[str, object] = {}
        if not claim.normalized_form:
            updates["normalized_form"] = normalize_claim_text(claim.text)

        # Try date + number normalization on temporal / quantitative types.
        # The normalized values are logged for now; Phase 3 can surface
        # them as additional Claim fields if we find a use case.
        if claim.claim_type == ClaimType.TEMPORAL:
            iso = normalize_date_expression(claim.text)
            if iso:
                logger.debug("Temporal claim normalized: %r -> %s", claim.text[:60], iso)
        if claim.claim_type == ClaimType.QUANTITATIVE:
            value = normalize_number_expression(claim.text)
            if value is not None:
                logger.debug("Quantitative claim normalized: %r -> %s", claim.text[:60], value)

        # Entity QIDs (stub — empty until Wikidata MCP is wired).
        entities = [claim.subject, claim.object]
        entities_clean = [e for e in entities if e]
        if entities_clean:
            qids = await resolve_entity_qids(entities_clean)
            if qids:
                updates["entity_qids"] = qids

        return claim.model_copy(update=updates) if updates else claim

    # Kept under the old private name for call-sites inside this class.
    async def _normalize_claim_obj(self, claim: Claim) -> Claim:
        return await self._normalize_claim_enrich(claim)

    def _fallback_decomposition(self, text: str) -> list[Claim]:
        """Simple sentence-based fallback when LLM parsing fails."""
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        claims = []

        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:  # Skip very short sentences
                continue

            claims.append(
                Claim(
                    id=uuid4(),
                    text=sent,
                    claim_type=ClaimType.UNCLASSIFIED,
                    confidence=0.5,  # Lower confidence for fallback
                    normalized_form=self._normalize_claim(sent),
                )
            )

        return claims

    async def health_check(self) -> bool:
        """Check if the decomposer is operational."""
        return await self._llm.health_check()
