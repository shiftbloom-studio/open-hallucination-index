"""Chunk-long-input helper for the L1 decomposer. Spec §3 + Task 1.3.

Splits an input text into overlapping chunks of bounded "tokens" so the
LLM's context window can handle documents of any length. The overlap
preserves co-reference across chunk boundaries (e.g. a pronoun resolved
in chunk N-1 is still visible to the decomposer in chunk N).

Token counting is a rough ``len(text) / 4`` approximation — adequate for
sizing the chunker, and avoids a mandatory ``tiktoken`` dependency. The
decomposer itself still uses the LLM provider's native tokenization for
the actual call, so the ~4-char-per-token estimate only drives chunk
boundaries, never token-budget-critical decisions.

Chunks preserve original character offsets so downstream code can still
populate ``Claim.source_span`` against the unchunked input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Roughly 4 chars per token in English — used only for chunk sizing.
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Chunk:
    """One contiguous slice of the input + its character offsets.

    ``start`` and ``end`` are indices into the *original* input text.
    ``overlap_with_previous`` is True when this chunk's leading portion
    was copied from the preceding chunk's tail for co-reference.
    """

    text: str
    start: int
    end: int
    overlap_with_previous: bool


def estimate_tokens(text: str) -> int:
    """Rough token count: 1 token ≈ 4 characters."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _split_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """Split on paragraph boundaries, returning (text, start, end) tuples.

    Paragraphs are separated by one or more blank lines. Whitespace-only
    paragraphs are dropped.
    """
    out: list[tuple[str, int, int]] = []
    pos = 0
    for match in re.split(r"\n\s*\n+", text):
        if not match:
            pos += 2  # skip the blank-line separator
            continue
        start = text.find(match, pos)
        if start < 0:
            # Shouldn't happen but guard against regex edge cases
            start = pos
        end = start + len(match)
        if match.strip():
            out.append((match, start, end))
        pos = end
    return out or [(text, 0, len(text))]


def _split_sentences(text: str, *, start_offset: int) -> list[tuple[str, int, int]]:
    """Coarse sentence-boundary split. Not perfect — good enough for chunking.

    Returns absolute-offset tuples so callers can thread character spans
    through from the top-level input.
    """
    out: list[tuple[str, int, int]] = []
    # Split at .!? followed by whitespace (keeping the terminator with the sentence)
    parts = re.split(r"(?<=[.!?])\s+", text)
    pos = 0
    for p in parts:
        if not p:
            continue
        idx = text.find(p, pos)
        if idx < 0:
            idx = pos
        start = start_offset + idx
        end = start + len(p)
        out.append((p, start, end))
        pos = idx + len(p)
    return out


def chunk_text(
    text: str,
    *,
    max_tokens: int = 1800,
    overlap_tokens: int = 200,
) -> list[Chunk]:
    """Split input text into overlapping chunks.

    Paragraphs are the primary split unit. A paragraph that would itself
    exceed ``max_tokens`` is sub-split at sentence boundaries. Consecutive
    small paragraphs are greedily joined until the combined token count
    approaches ``max_tokens``.

    Each non-first chunk starts with the last ``overlap_tokens`` worth of
    the previous chunk (measured in characters × 4, rounded outward),
    copied verbatim with its original offsets preserved in ``start``.

    The empty-string and whitespace-only cases both yield ``[]``.
    """
    if not text or not text.strip():
        return []

    max_chars = max_tokens * _CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _CHARS_PER_TOKEN

    paragraphs = _split_paragraphs(text)

    # Sub-split any single paragraph that busts the budget
    units: list[tuple[str, int, int]] = []
    for para_text, p_start, p_end in paragraphs:
        if len(para_text) <= max_chars:
            units.append((para_text, p_start, p_end))
        else:
            units.extend(_split_sentences(para_text, start_offset=p_start))

    chunks: list[Chunk] = []
    current: list[tuple[str, int, int]] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        start = current[0][1]
        end = current[-1][2]
        joined = text[start:end]
        overlap_with_prev = len(chunks) > 0 and chunks[-1].end > start
        chunks.append(
            Chunk(
                text=joined,
                start=start,
                end=end,
                overlap_with_previous=overlap_with_prev,
            )
        )
        current = []
        current_len = 0

    for unit in units:
        unit_text, u_start, u_end = unit
        if current_len + len(unit_text) + 2 > max_chars and current:
            # Close current chunk
            flush()
            # Seed the next chunk with the tail of the previous one for co-ref
            if chunks and overlap_chars > 0:
                prev = chunks[-1]
                seed_start = max(prev.start, prev.end - overlap_chars)
                if seed_start < prev.end:
                    current.append((text[seed_start : prev.end], seed_start, prev.end))
                    current_len = prev.end - seed_start
        current.append(unit)
        current_len += len(unit_text) + 2

    flush()
    return chunks
