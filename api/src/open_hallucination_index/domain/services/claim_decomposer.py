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

from open_hallucination_index.domain.entities import Claim, ClaimType
from open_hallucination_index.ports.claim_decomposer import ClaimDecomposer
from open_hallucination_index.ports.llm_provider import LLMMessage

if TYPE_CHECKING:
    from open_hallucination_index.ports.llm_provider import LLMProvider

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

Output as JSON array:
```json
[
  {
    "text": "The claim as a complete sentence",
    "subject": "Entity name",
    "predicate": "relationship",
    "object": "value or entity",
    "claim_type": "type",
    "confidence": 0.0-1.0
  }
]
```

Only output the JSON array, nothing else."""


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

        Args:
            text: Unstructured input text to analyze.
            context: Optional context (e.g., document title, topic).
            max_claims: Optional limit on number of claims to extract.

        Returns:
            List of extracted claims.
        """
        if not text or not text.strip():
            return []

        limit = max_claims or self._max_claims

        # Build the user message
        user_content = f"Extract up to {limit} factual claims from the following text:\n\n{text}"
        if context:
            user_content = f"Context: {context}\n\n{user_content}"

        system_content = (
            DECOMPOSITION_PROMPT
            + "\nRespond strictly with a JSON object containing a 'claims' list."
        )
        messages = [
            LLMMessage(role="system", content=system_content),
            LLMMessage(role="user", content=user_content),
        ]

        try:
            response = await self._llm.complete(
                messages,
                temperature=0.0,
                max_tokens=2048,
                json_mode=True,  # [NEU] Aktivieren
            )

            claims = self._parse_response(response.content, text)
            logger.info(f"Extracted {len(claims)} claims from text")
            return claims[:limit]

        except Exception as e:
            logger.error(f"Claim decomposition failed: {e}")
            raise DecompositionError(f"Failed to decompose text: {e}") from e

    def _parse_response(self, response: str, original_text: str) -> list[Claim]:
        """Parse LLM response into Claim objects."""
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON array directly
            json_str = response.strip()
            # Find the JSON array
            start = json_str.find("[")
            end = json_str.rfind("]") + 1
            if start != -1 and end > start:
                json_str = json_str[start:end]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            # Fall back to simple sentence splitting
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
        """Normalize claim text for matching."""
        # Lowercase, remove extra whitespace, remove punctuation
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[.,;:!?]$", "", normalized)
        return normalized

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
