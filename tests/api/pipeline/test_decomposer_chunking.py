"""Unit tests for pipeline.decomposer_chunking."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from pipeline.decomposer_chunking import (  # noqa: E402
    Chunk,
    chunk_text,
    estimate_tokens,
)


def test_empty_input_yields_empty_list() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n   ") == []


def test_single_paragraph_under_budget_yields_one_chunk() -> None:
    text = "Einstein was born in 1879. He developed special relativity in 1905."
    chunks = chunk_text(text, max_tokens=1800, overlap_tokens=200)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start == 0
    assert chunks[0].end == len(text)
    assert chunks[0].overlap_with_previous is False


def test_multiple_short_paragraphs_fit_in_one_chunk() -> None:
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = chunk_text(text, max_tokens=1800, overlap_tokens=200)
    assert len(chunks) == 1
    # Whole text preserved
    assert "Para one" in chunks[0].text
    assert "Para three" in chunks[0].text


def test_long_input_splits_into_multiple_chunks_with_overlap() -> None:
    # Build ~4x budget worth of text so at least 2 chunks are produced
    paragraphs = [
        f"Paragraph {i} with some filler content here." * 40 for i in range(20)
    ]
    text = "\n\n".join(paragraphs)
    # Small budget to force multiple chunks
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)
    assert len(chunks) >= 2, f"Expected multiple chunks, got {len(chunks)}"
    # Later chunks carry an overlap marker
    assert any(c.overlap_with_previous for c in chunks[1:])


def test_char_offsets_stay_within_input_bounds() -> None:
    text = "Sentence one. Sentence two.\n\nParagraph two here."
    chunks = chunk_text(text, max_tokens=1800, overlap_tokens=200)
    for c in chunks:
        assert 0 <= c.start <= c.end <= len(text)
        assert text[c.start : c.end] == c.text or c.overlap_with_previous


def test_overlap_preserves_tail_of_previous_chunk() -> None:
    # Force at least 2 chunks
    text = ("Alpha. " * 200) + "\n\n" + ("Beta. " * 200)
    chunks = chunk_text(text, max_tokens=50, overlap_tokens=10)
    assert len(chunks) >= 2
    # Second chunk's first text should overlap into the previous chunk's
    # tail (same characters should appear in both regions of the input)
    if chunks[1].overlap_with_previous:
        assert chunks[1].start < chunks[0].end


def test_estimate_tokens_rough_magnitudes() -> None:
    # 4 chars ≈ 1 token (rough): 400 chars should estimate ~100 tokens
    assert 80 <= estimate_tokens("x" * 400) <= 120
    # Very short strings clamp to minimum 1 token
    assert estimate_tokens("") == 1
    assert estimate_tokens("a") == 1


def test_chunk_dataclass_is_frozen() -> None:
    c = Chunk(text="hi", start=0, end=2, overlap_with_previous=False)
    import pytest

    with pytest.raises(Exception):  # FrozenInstanceError
        c.text = "bye"  # type: ignore[misc]
