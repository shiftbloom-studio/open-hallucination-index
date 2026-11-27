"""Small text helpers shared by the MCP source adapters."""

from __future__ import annotations

import re

_WORD_CHAR = re.compile(r"[\wÀ-ɏ]")


def strip_edge_punctuation(word: str) -> str:
    """Strip leading non-word chars and trailing non-word chars except "-".

    Linear-time equivalent of
    ``re.sub(r"^[^\\w\\u00c0-\\u024f]+|[^\\w\\u00c0-\\u024f\\-]+$", "", word)``.
    """
    start = 0
    end = len(word)
    while start < end and not _WORD_CHAR.match(word[start]):
        start += 1
    while end > start and word[end - 1] != "-" and not _WORD_CHAR.match(word[end - 1]):
        end -= 1
    return word[start:end]
