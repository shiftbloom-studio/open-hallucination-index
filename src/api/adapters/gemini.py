"""Native Google Gemini adapter.

We talk to the *native* Gemini API
(https://generativelanguage.googleapis.com/v1beta/models/<model>:generateContent)
instead of the OpenAI-compatible shim at /v1beta/openai/ because the shim
does not forward `safetySettings` or `generationConfig.thinkingConfig` —
OHI is a hallucination-detection product, so blocking factually-wrong-but-
safety-adjacent prompts (e.g. false claims about real people) at the LLM
layer would defeat its purpose. We set all four safety categories to
BLOCK_NONE and pass thinkingLevel: HIGH on Gemini 3 models for the best
reasoning.

The adapter implements the LLMProvider port so it swaps in for
OpenAILLMAdapter without any caller-side changes. The LLMSettings fields
it uses are:
- api_key  — Gemini API key (AIza...)
- model    — e.g. "gemini-3-flash-preview"
- timeout_seconds — request timeout

base_url is ignored; the Gemini endpoint is baked in.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
import orjson

from adapters.openai import LLMProviderError
from interfaces.llm import LLMMessage, LLMProvider, LLMResponse

if TYPE_CHECKING:
    from config.settings import LLMSettings

logger = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# OHI is a hallucination-detection service; decomposing factually-wrong
# claims about real people is the core product. Disable all four safety
# categories so Gemini doesn't silently return empty content and force us
# into the single-claim fallback path.
SAFETY_SETTINGS_BLOCK_NONE = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


def _split_messages(
    messages: list[LLMMessage],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Split OpenAI-style messages into (systemInstruction, contents).

    Gemini separates system instructions from the conversation turns; the
    OpenAI-compat adapter flattened them into a [INST] Mistral prefix,
    which is needless for native Gemini.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        elif m.role == "user":
            contents.append({"role": "user", "parts": [{"text": m.content}]})
        elif m.role == "assistant":
            # Gemini's conversation role for assistant is "model".
            contents.append({"role": "model", "parts": [{"text": m.content}]})
        # silently drop any other roles — Gemini would reject them
    system_instruction = (
        {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
    )
    return system_instruction, contents


def _build_generation_config(
    *,
    temperature: float,
    max_tokens: int | None,
    stop: list[str] | None,
    json_mode: bool,
    model: str,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        cfg["maxOutputTokens"] = max_tokens
    if stop:
        cfg["stopSequences"] = stop
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    # Gemini 3 introduces an explicit thinking budget. HIGH gives the best
    # reasoning for claim decomposition + verification. Older models
    # silently ignore this field.
    if model.startswith("gemini-3"):
        cfg["thinkingConfig"] = {"thinkingLevel": "HIGH"}
    return cfg


class GeminiLLMAdapter(LLMProvider):
    """Native Gemini API adapter."""

    def __init__(self, settings: LLMSettings) -> None:
        api_key = settings.api_key.get_secret_value()
        if not api_key or api_key == "no-key-required":
            raise LLMProviderError(
                "GeminiLLMAdapter requires LLM_API_KEY to be set to a real Gemini key"
            )
        self._model = settings.model
        self._client = httpx.AsyncClient(
            base_url=GEMINI_BASE_URL,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(settings.timeout_seconds, connect=10.0),
        )
        logger.info(f"Gemini native adapter configured; model={self._model}")

    async def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        system_instruction, contents = _split_messages(messages)
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": _build_generation_config(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
                json_mode=json_mode,
                model=self._model,
            ),
            "safetySettings": SAFETY_SETTINGS_BLOCK_NONE,
        }
        if system_instruction is not None:
            body["systemInstruction"] = system_instruction

        try:
            resp = await self._client.post(
                f"/models/{self._model}:generateContent", json=body
            )
        except httpx.HTTPError as exc:
            logger.error(f"Gemini request failed: {exc}")
            raise LLMProviderError(f"Request failed: {exc}") from exc

        if resp.status_code != 200:
            # Gemini puts errors at the top level or wrapped in a list.
            logger.error(
                "Gemini API error: %s - %s", resp.status_code, resp.text[:500]
            )
            raise LLMProviderError(
                f"Gemini API error: {resp.status_code} - {resp.text[:500]}"
            )

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            # Safety filters can still kick in via promptFeedback even with
            # BLOCK_NONE if the category isn't in our list. Surface it.
            pf = data.get("promptFeedback", {})
            raise LLMProviderError(
                f"Gemini returned no candidates; promptFeedback={pf}"
            )

        cand = candidates[0]
        parts = cand.get("content", {}).get("parts", [])
        content = "".join(p.get("text", "") for p in parts if isinstance(p, dict))

        usage: dict[str, int] | None = None
        um = data.get("usageMetadata", {})
        if um:
            usage = {
                "prompt_tokens": int(um.get("promptTokenCount", 0)),
                "completion_tokens": int(um.get("candidatesTokenCount", 0)),
                "total_tokens": int(um.get("totalTokenCount", 0)),
            }

        return LLMResponse(
            content=content,
            model=self._model,
            usage=usage,
            finish_reason=str(cand.get("finishReason", "")).lower() or None,
            raw_response=data,
        )

    async def complete_stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        system_instruction, contents = _split_messages(messages)
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": _build_generation_config(
                temperature=temperature,
                max_tokens=max_tokens,
                stop=None,
                json_mode=False,
                model=self._model,
            ),
            "safetySettings": SAFETY_SETTINGS_BLOCK_NONE,
        }
        if system_instruction is not None:
            body["systemInstruction"] = system_instruction

        try:
            async with self._client.stream(
                "POST",
                f"/models/{self._model}:streamGenerateContent?alt=sse",
                json=body,
            ) as resp:
                if resp.status_code != 200:
                    raw = await resp.aread()
                    raise LLMProviderError(
                        f"Gemini stream error {resp.status_code}: {raw[:400]!r}"
                    )
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        chunk = orjson.loads(payload)
                    except orjson.JSONDecodeError:
                        continue
                    for cand in chunk.get("candidates", []):
                        for part in cand.get("content", {}).get("parts", []):
                            text = part.get("text") if isinstance(part, dict) else None
                            if text:
                                yield text
        except httpx.HTTPError as exc:
            logger.error(f"Gemini stream failed: {exc}")
            raise LLMProviderError(f"Stream failed: {exc}") from exc

    async def health_check(self) -> bool:
        """Cheap liveness probe: list models."""
        try:
            resp = await self._client.get("/models?pageSize=1")
            return resp.status_code == 200
        except Exception as exc:
            logger.warning(f"Gemini health check failed: {exc}")
            return False

    @property
    def model_name(self) -> str:
        return self._model
