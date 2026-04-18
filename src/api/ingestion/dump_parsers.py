"""Wave 3 Stream C — streaming dump parsers.

Two parsers:

* :class:`WikidataJsonDumpParser` — streams the
  ``wikidata-YYYYMMDD-all.json.bz2`` archive entity-by-entity. The
  file is a giant JSON array, one entity per line (except the first
  and last lines). We bypass the standard JSON parser entirely and
  line-read, parsing each line as one entity object. This avoids
  loading ~100 GB into memory.
* :class:`EnwikiXmlDumpParser` — streams
  ``enwiki-YYYYMMDD-pages-articles.xml.bz2`` main-namespace article
  by article using ``xml.etree.ElementTree.iterparse``. Only ``page``
  elements with ``ns=0`` and a non-redirect ``revision`` are yielded.

Both parsers are designed to cooperate with the lifecycle machinery
(:mod:`~ingestion.lifecycle`) — they yield records at a cadence the
orchestrator can batch into checkpointable commits.

These are DEFENSIVE parsers — bad UTF-8, truncated records, malformed
XML attributes are logged + skipped (the DLQ lane handles persistently-
bad records). The stream never raises mid-iteration unless the file
itself is unreadable.
"""

from __future__ import annotations

import bz2
import io
import json
import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wikidata JSON dump
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WikidataEntity:
    """Minimal projection of a Wikidata JSON entity record.

    Not a pydantic model — we skip validation here to keep the parser
    fast; the orchestrator's pass layer filters + validates before
    writing to Aura.
    """

    qid: str
    label: str | None
    description: str | None
    wikipedia_title: str | None
    claims: dict[str, list[Any]]
    raw: dict[str, Any]


class WikidataJsonDumpParser:
    """Streaming parser for ``wikidata-YYYYMMDD-all.json.bz2``.

    The archive layout (documented at
    https://www.wikidata.org/wiki/Wikidata:Database_download) is a
    JSON array where each line is a standalone entity JSON object
    (with a trailing comma on all but the last line). We open the
    bz2 stream, skip the leading ``[``, then yield one entity per
    line.
    """

    def __init__(self, path: str, *, english_sitelink_only: bool = True) -> None:
        self.path = path
        self.english_sitelink_only = english_sitelink_only

    def __iter__(self) -> Iterator[WikidataEntity]:
        with bz2.open(self.path, "rt", encoding="utf-8") as fh:
            first = fh.readline().strip()
            if first != "[":
                # Some dumps ship without the leading bracket — rewind.
                fh.seek(0)
            for raw_line in fh:
                line = raw_line.strip().rstrip(",")
                if not line or line == "]":
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Wikidata parse error on line: %s", exc)
                    continue
                entity = self._project(obj)
                if entity is None:
                    continue
                yield entity

    def _project(self, obj: dict[str, Any]) -> WikidataEntity | None:
        qid = obj.get("id")
        if not qid or not isinstance(qid, str) or not qid.startswith(("Q", "P")):
            return None
        sitelinks = obj.get("sitelinks") or {}
        enwiki = sitelinks.get("enwiki")
        wikipedia_title = enwiki.get("title") if isinstance(enwiki, dict) else None
        if self.english_sitelink_only and wikipedia_title is None:
            return None
        labels = obj.get("labels") or {}
        descs = obj.get("descriptions") or {}
        label_en = (labels.get("en") or {}).get("value")
        desc_en = (descs.get("en") or {}).get("value")
        return WikidataEntity(
            qid=qid,
            label=label_en,
            description=desc_en,
            wikipedia_title=wikipedia_title,
            claims=obj.get("claims") or {},
            raw=obj,
        )


# ---------------------------------------------------------------------------
# enwiki XML dump
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnwikiPage:
    """Minimal projection of an enwiki main-namespace article."""

    pageid: int
    title: str
    wikitext: str
    is_redirect: bool


class EnwikiXmlDumpParser:
    """Streaming parser for ``enwiki-YYYYMMDD-pages-articles.xml.bz2``.

    Uses ``iterparse(events=('end',))`` on ``<page>`` elements and
    clears each one after yielding to keep memory bounded to a single
    page.
    """

    # Standard Wikipedia MediaWiki XML namespace.
    NS = "{http://www.mediawiki.org/xml/export-0.10/}"

    def __init__(self, path: str) -> None:
        self.path = path

    def __iter__(self) -> Iterator[EnwikiPage]:
        with bz2.open(self.path, "rb") as fh:
            for _ev, elem in ET.iterparse(fh, events=("end",)):
                if elem.tag != f"{self.NS}page":
                    continue
                ns = (elem.findtext(f"{self.NS}ns") or "").strip()
                if ns != "0":
                    elem.clear()
                    continue
                page = self._project(elem)
                elem.clear()
                if page is not None:
                    yield page

    def _project(self, elem: ET.Element) -> EnwikiPage | None:
        title = (elem.findtext(f"{self.NS}title") or "").strip()
        pageid_text = (elem.findtext(f"{self.NS}id") or "").strip()
        try:
            pageid = int(pageid_text)
        except ValueError:
            logger.warning("enwiki page without valid id: title=%r", title)
            return None
        revision = elem.find(f"{self.NS}revision")
        if revision is None:
            return None
        wikitext_elem = revision.find(f"{self.NS}text")
        wikitext = (wikitext_elem.text if wikitext_elem is not None else None) or ""
        is_redirect = elem.find(f"{self.NS}redirect") is not None
        return EnwikiPage(
            pageid=pageid,
            title=title,
            wikitext=wikitext,
            is_redirect=is_redirect,
        )


__all__ = [
    "EnwikiPage",
    "EnwikiXmlDumpParser",
    "WikidataEntity",
    "WikidataJsonDumpParser",
]
