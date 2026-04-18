"""Wave 3 Stream P — NliOpenAIGpt54Adapter tests.

Canned-response stubs; ZERO live OpenAI calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

# Same sys.modules purge pattern as the D1/Stream-B tests.
_FLAT_PACKAGES = {"adapters", "interfaces", "models", "pipeline", "config", "server", "services"}
_cached_iface = sys.modules.get("interfaces")
_cached_iface_file = getattr(_cached_iface, "__file__", "") or ""
if _cached_iface is None or str(_SRC_API) not in _cached_iface_file:
    for _cached_name in list(sys.modules):
        _root = _cached_name.split(".", 1)[0]
        if _root in _FLAT_PACKAGES:
            del sys.modules[_cached_name]

import orjson  # noqa: E402

from adapters.nli_openai_gpt54 import (  # noqa: E402
    NliOpenAIGpt54Adapter,
    _parse_model_with_effort,
)


def _canned_response(label: str = "support", s: float = 0.8, r: float = 0.1, n: float = 0.1) -> object:
    """Mock of the openai Responses API response object."""
    class _Resp:
        output_text = orjson.dumps({
            "label": label,
            "supporting_score": s,
            "refuting_score": r,
            "neutral_score": n,
            "reasoning": "stub",
            "confidence": 0.9,
        }).decode("utf-8")

    return _Resp()


def test_parse_model_with_effort_suffixes() -> None:
    assert _parse_model_with_effort("gpt-5.4-xhigh") == ("gpt-5.4", "xhigh")
    assert _parse_model_with_effort("gpt-5.4-high") == ("gpt-5.4", "high")
    assert _parse_model_with_effort("gpt-5.4-medium") == ("gpt-5.4", "medium")
    assert _parse_model_with_effort("gpt-5.4") == ("gpt-5.4", None)


async def test_classify_happy_path_returns_nli_result() -> None:
    adapter = NliOpenAIGpt54Adapter(api_key="sk-test", model_with_effort="gpt-5.4-xhigh")
    with patch.object(adapter._client.responses, "create", new=AsyncMock(return_value=_canned_response())):
        result = await adapter.classify("claim a", "claim b")
    assert result.label == "support"
    assert abs(result.supporting_score - 0.8) < 1e-6
    assert abs(result.refuting_score - 0.1) < 1e-6


async def test_classify_passes_reasoning_effort() -> None:
    """Ctor with '-xhigh' suffix → reasoning={'effort': 'xhigh'} kwarg."""
    adapter = NliOpenAIGpt54Adapter(api_key="sk-test", model_with_effort="gpt-5.4-xhigh")
    mock = AsyncMock(return_value=_canned_response())
    with patch.object(adapter._client.responses, "create", new=mock):
        await adapter.classify("a", "b")
    mock.assert_called_once()
    kwargs = mock.call_args.kwargs
    assert kwargs["model"] == "gpt-5.4"
    assert kwargs["reasoning"] == {"effort": "xhigh"}


async def test_classify_fallback_on_terminal_failure() -> None:
    """Exception on every retry → returns _NEUTRAL_FALLBACK sentinel."""
    adapter = NliOpenAIGpt54Adapter(
        api_key="sk-test", model_with_effort="gpt-5.4-xhigh", max_retries=2
    )
    with patch.object(
        adapter._client.responses,
        "create",
        new=AsyncMock(side_effect=RuntimeError("transport error")),
    ):
        result = await adapter.classify("a", "b")
    assert result.reasoning == "nli_unavailable"
    assert result.confidence == 0.0
    assert result.label == "neutral"


async def test_classify_retries_then_succeeds() -> None:
    """First attempt fails, second succeeds → returns the parsed result."""
    adapter = NliOpenAIGpt54Adapter(api_key="sk-test", max_retries=3)
    mock = AsyncMock(
        side_effect=[RuntimeError("flaky"), _canned_response(label="refute", s=0.05, r=0.9, n=0.05)]
    )
    with patch.object(adapter._client.responses, "create", new=mock):
        result = await adapter.classify("a", "b")
    assert result.label == "refute"
    assert mock.call_count == 2


def test_classify_rejects_invalid_label_via_parser() -> None:
    """Malformed response (invalid label) raises ValueError inside
    _parse_result — the adapter catches and retries."""
    from adapters.nli_openai_gpt54 import _parse_result
    bad_json = '{"label": "maybe", "supporting_score": 0.5, "refuting_score": 0.5, "neutral_score": 0.0}'
    import pytest
    with pytest.raises(ValueError):
        _parse_result(bad_json)
