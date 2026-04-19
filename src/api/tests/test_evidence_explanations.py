from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from adapters.mcp_sources.mediawiki import MediaWikiAdapter
from interfaces.nli import NliResult
from models.entities import Claim, Evidence, EvidenceSource
from pipeline.pipeline import _annotate_evidence_with_nli


@pytest.mark.asyncio
async def test_mediawiki_find_evidence_prefers_matched_search_snippet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MediaWikiAdapter()
    adapter._available = True

    async def fake_search(query: str, limit: int = 5) -> list[dict[str, object]]:
        assert limit == 5
        if query == "Albert Einstein was born in Germany.":
            return [
                {
                    "title": "Albert Einstein",
                    "pageid": 736,
                    "snippet": (
                        '<span class="searchmatch">Albert Einstein</span> '
                        'was born in Ulm, Germany, in 1879.'
                    ),
                }
            ]
        return []

    async def fake_get_extract(title: str, sentences: int = 5) -> str:
        raise AssertionError(f"lead extract should not be fetched for snippet-friendly page {title!r}")

    monkeypatch.setattr(adapter, "_search", fake_search)
    monkeypatch.setattr(adapter, "_get_extract", fake_get_extract)

    claim = Claim(
        text="Albert Einstein was born in Germany.",
        subject="Albert Einstein",
    )

    evidence = await adapter.find_evidence(claim)

    assert len(evidence) == 1
    assert evidence[0].content == (
        "Albert Einstein: Albert Einstein was born in Ulm, Germany, in 1879."
    )
    assert evidence[0].structured_data is not None
    assert evidence[0].structured_data["snippet"] == (
        "Albert Einstein was born in Ulm, Germany, in 1879."
    )
    assert evidence[0].structured_data["matched_query"] == claim.text


@pytest.mark.asyncio
async def test_mediawiki_find_evidence_ranks_exact_subject_page_above_noisy_full_claim_hits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MediaWikiAdapter()
    adapter._available = True

    async def fake_search(query: str, limit: int = 5) -> list[dict[str, object]]:
        assert limit == 5
        if query == "albert einstein was born in russia":
            return [
                {
                    "title": "Albert Brooks",
                    "pageid": 3075,
                    "snippet": (
                        '<span class="searchmatch">Albert</span> Lawrence '
                        '<span class="searchmatch">Einstein</span> (born July 22, 1947)'
                    ),
                },
                {
                    "title": "Political views of Albert Einstein",
                    "pageid": 123,
                    "snippet": "German-born scientist Albert Einstein was best known for...",
                },
            ]
        if query == "Albert Einstein was born in Russia.":
            return [
                {
                    "title": "Religious and philosophical views of Albert Einstein",
                    "pageid": 23797095,
                    "snippet": "Albert Einstein's religious views have been widely studied.",
                }
            ]
        if query == "Albert Einstein was born in Russia":
            return []
        if query == "Albert Einstein":
            return [
                {
                    "title": "Albert Einstein",
                    "pageid": 736,
                    "snippet": (
                        "<span class=\"searchmatch\">Albert</span> "
                        "<span class=\"searchmatch\">Einstein</span> "
                        "(14 March 1879 – 18 April 1955) was a German-born theoretical physicist."
                    ),
                }
            ]
        raise AssertionError(f"unexpected query {query!r}")

    async def fake_get_extract(title: str, sentences: int = 5) -> str:
        raise AssertionError(f"lead extract should not be fetched for ranked result {title!r}")

    monkeypatch.setattr(adapter, "_search", fake_search)
    monkeypatch.setattr(adapter, "_get_extract", fake_get_extract)

    claim = Claim(
        text="Albert Einstein was born in Russia.",
        normalized_form="albert einstein was born in russia",
        subject="Albert Einstein",
        predicate="was born in",
        object="Russia",
    )

    evidence = await adapter.find_evidence(claim)

    assert len(evidence) == 3
    assert evidence[0].structured_data is not None
    assert evidence[0].structured_data["title"] == "Albert Einstein"
    assert evidence[0].structured_data["matched_query"] == "Albert Einstein"
    assert evidence[0].content.startswith(
        "Albert Einstein: Albert Einstein (14 March 1879"
    )


def test_annotate_evidence_with_nli_preserves_scores_and_reasoning() -> None:
    evidence = Evidence(
        id=uuid4(),
        source=EvidenceSource.WIKIPEDIA,
        content="Albert Einstein was born in Ulm, Germany, in 1879.",
        structured_data={"title": "Albert Einstein"},
        retrieved_at=datetime.now(UTC),
    )
    result = NliResult(
        label="refute",
        supporting_score=0.03,
        refuting_score=0.94,
        neutral_score=0.03,
        reasoning="The evidence states Einstein was born in Germany, not Russia.",
        confidence=0.88,
    )

    annotated = _annotate_evidence_with_nli(evidence, result)

    assert annotated.classification_confidence == pytest.approx(0.94)
    assert annotated.structured_data is not None
    assert annotated.structured_data["nli_label"] == "refute"
    assert annotated.structured_data["nli_reasoning"] == result.reasoning
    assert annotated.structured_data["bucket_score"] == pytest.approx(0.94)
    assert annotated.structured_data["nli_confidence"] == pytest.approx(0.88)
    assert annotated.structured_data["title"] == "Albert Einstein"
