"""
Verify Text Filters
===================

Extensible pre-filter pipeline for incoming verification requests.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx

from interfaces.llm import LLMMessage, LLMProvider

logger = logging.getLogger(__name__)

FilterAction = Literal["allow", "reject", "rewrite"]


@dataclass(frozen=True, slots=True)
class FilterDecision:
    """Outcome of a filter step."""

    action: FilterAction
    reason: str
    text: str | None = None
    stage: str = ""
    status_code: int = 400


class FilterStep(ABC):
    """Abstract filter step."""

    name: str = "filter"

    @abstractmethod
    async def evaluate(self, text: str) -> FilterDecision:
        """Evaluate input text and return a decision."""
        raise NotImplementedError


class BlacklistFilter(FilterStep):
    """Fast blacklist filter for known harmful terms."""

    name = "blacklist"

    def __init__(self, terms: list[str]) -> None:
        self._terms = [t.strip() for t in terms if t.strip()]
        escaped = [re.escape(t) for t in self._terms]
        self._pattern = re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

    async def evaluate(self, text: str) -> FilterDecision:
        if not self._terms:
            return FilterDecision(action="allow", reason="No blacklist terms configured.")

        match = self._pattern.search(text)
        if match:
            return FilterDecision(
                action="reject",
                reason="Text rejected: contains blocked harmful term.",
                stage=self.name,
                status_code=403,
            )
        return FilterDecision(action="allow", reason="Blacklist check passed.", stage=self.name)


class LLMHarmFilter(FilterStep):
    """LLM-based harmful content screening."""

    name = "llm_harm"

    def __init__(self, llm_provider: LLMProvider | None) -> None:
        self._llm_provider = llm_provider

    async def evaluate(self, text: str) -> FilterDecision:
        if self._llm_provider is None:
            return FilterDecision(action="allow", reason="LLM unavailable; skipping harm filter.")

        prompt = f"""You are a safety classifier. Determine if the following text contains harmful content
or requests that should be rejected (self-harm, violence, illegal activities, hate, or exploitation).

TEXT:
{text}

Respond with valid JSON only in this exact shape:
{{
    "harmful": false,
    "category": "",
    "reason": "short explanation"
}}
"""

        try:
            response = await self._llm_provider.complete(
                messages=[
                    LLMMessage(role="system", content="You classify harmful content conservatively."),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=0.1,
                max_tokens=256,
            )
            harmful, category, reason = _parse_harm_response(response.content)
        except Exception as e:
            logger.warning("LLM harm filter failed: %s", e)
            return FilterDecision(action="allow", reason="LLM harm filter failed; allowing.")

        if harmful:
            detail = "Text rejected: harmful content detected."
            if category:
                detail = f"{detail} Category: {category}."
            if reason:
                detail = f"{detail} {reason}"
            return FilterDecision(
                action="reject",
                reason=detail.strip(),
                stage=self.name,
                status_code=403,
            )

        return FilterDecision(action="allow", reason="Harm filter passed.", stage=self.name)


class LLMClaimFilter(FilterStep):
    """LLM-based claim detection and normalization."""

    name = "llm_claim"

    def __init__(self, llm_provider: LLMProvider | None) -> None:
        self._llm_provider = llm_provider

    async def evaluate(self, text: str) -> FilterDecision:
        if self._llm_provider is None:
            return FilterDecision(action="allow", reason="LLM unavailable; skipping claim filter.")

        prompt = f"""You are a fact-checking assistant. Decide how to handle the following input.

Rules:
- If the text is a question, extract the implied factual claim in declarative form.
- If there is no clear factual claim, or the text is nonsensical or obviously false in a trivial way,
  REJECT the text.
- Otherwise, PASS_THROUGH the original text.

TEXT:
{text}

Respond with valid JSON only in this exact shape:
{{
    "action": "EXTRACT_CLAIM|REJECT_NO_CLAIM|PASS_THROUGH",
    "claim": "",
    "reason": "short explanation"
}}
"""

        try:
            response = await self._llm_provider.complete(
                messages=[
                    LLMMessage(role="system", content="You detect claims conservatively."),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=0.1,
                max_tokens=256,
            )
            action, claim, reason = _parse_claim_response(response.content)
        except Exception as e:
            logger.warning("LLM claim filter failed: %s", e)
            return FilterDecision(action="allow", reason="LLM claim filter failed; allowing.")

        if action == "REJECT_NO_CLAIM":
            detail = "Text rejected: no clear factual claim detected."
            if reason:
                detail = f"{detail} {reason}"
            return FilterDecision(
                action="reject",
                reason=detail.strip(),
                stage=self.name,
                status_code=422,
            )

        if action == "EXTRACT_CLAIM":
            cleaned = (claim or "").strip()
            if not cleaned:
                return FilterDecision(
                    action="reject",
                    reason="Text rejected: unable to extract a clean claim.",
                    stage=self.name,
                    status_code=422,
                )
            detail = "Claim extracted and normalized for verification."
            if reason:
                detail = f"{detail} {reason}"
            return FilterDecision(
                action="rewrite",
                reason=detail.strip(),
                text=cleaned,
                stage=self.name,
            )

        return FilterDecision(action="allow", reason="Claim filter passed.", stage=self.name)


def build_default_filters(llm_provider: LLMProvider | None) -> list[FilterStep]:
    """Construct the default filter pipeline."""
    blacklist_terms = _load_blacklist_terms()
    return [
        BlacklistFilter(blacklist_terms),
        LLMHarmFilter(llm_provider),
        LLMClaimFilter(llm_provider),
    ]


def _load_blacklist_terms() -> list[str]:
    """Load blacklist terms from file, with safe fallback defaults."""
    env_path = os.getenv("VERIFY_BLACKLIST_PATH")
    env_url = os.getenv("VERIFY_BLACKLIST_URL")
    default_path = Path(__file__).resolve().parent / "blacklists" / "blacklist.txt"
    blacklist_path = Path(env_path) if env_path else default_path

    if env_url and not blacklist_path.exists():
        try:
            response = httpx.get(env_url, timeout=10.0)
            response.raise_for_status()
            blacklist_path.parent.mkdir(parents=True, exist_ok=True)
            blacklist_path.write_text(response.text, encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to download blacklist from URL: %s", e)

    if blacklist_path.exists():
        try:
            content = blacklist_path.read_text(encoding="utf-8")
            terms = [line.strip() for line in content.splitlines() if line.strip()]
            if terms:
                return terms
        except OSError as e:
            logger.warning("Failed to read blacklist file: %s", e)

    # Conservative fallback list (non-violent terms)
    return [
        "malware",
        "ransomware",
        "phishing",
        "credential",
        "exploit",
        "ddos",
        "keylogger",
        "botnet",
        "fraud",
        "scam",
    ]


def _parse_harm_response(response: str) -> tuple[bool, str, str]:
    try:
        response = _strip_json_wrappers(response)
        data = json.loads(response) if response else {}
        harmful = bool(data.get("harmful", False))
        category = str(data.get("category", ""))
        reason = str(data.get("reason", ""))
        return harmful, category, reason
    except Exception as e:
        logger.warning("Failed to parse harm response: %s", e)
        return False, "", ""


def _parse_claim_response(response: str) -> tuple[str, str, str]:
    try:
        response = _strip_json_wrappers(response)
        data = json.loads(response) if response else {}
        action = str(data.get("action", "PASS_THROUGH")).upper()
        claim = str(data.get("claim", ""))
        reason = str(data.get("reason", ""))
        return action, claim, reason
    except Exception as e:
        logger.warning("Failed to parse claim response: %s", e)
        return "PASS_THROUGH", "", ""


def _strip_json_wrappers(response: str) -> str:
    response = response.strip()
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", response, re.IGNORECASE)
    if json_match:
        candidate = json_match.group(1).strip()
        if candidate.startswith("{") or candidate.startswith("["):
            return candidate
    return response
