"""Wave 3 Stream C ingestion pass regressions.

Focused tests around the data-quality improvements that materially
change what lands in Aura and Qdrant.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from ingestion.checkpoint_store import CheckpointStore
from ingestion.dump_parsers import EnwikiPage, WikidataEntity
from ingestion.lifecycle import DLQWriter, PauseController, ProgressReporter
from ingestion.pass1_entities import Pass1EntityWriter
from ingestion.pass1b_entity_vectors import Pass1bEntityVectorWriter
from ingestion.pass2_passages import PassageChunk, Pass2PassageWriter
import ingestion.pass1_entities as pass1_entities
import ingestion.pass2_passages as pass2_passages


@dataclass
class _RecordingGraph:
    calls: list[tuple[str, dict]] = field(default_factory=list)
    aura_batch: list[dict] | None = None

    async def run_cypher(self, cypher: str, params: dict):
        self.calls.append((cypher, params))
        if "UNWIND $titles AS title" in cypher:
            return [
                {"title": "Radium", "qid": "Q113145171"},
                {"title": "Physics", "qid": "Q413"},
                {"title": "Marie Curie", "qid": "Q7186"},
            ]
        if "MERGE (p:Passage {id: row.passage_id})" in cypher:
            self.aura_batch = params["batch"]
            return [{"n": len(params["batch"])}]
        return []


@dataclass
class _SequentialSelectGraph:
    calls: list[tuple[str, dict]] = field(default_factory=list)
    embedded: bool = False

    async def run_cypher(self, cypher: str, params: dict):
        self.calls.append((cypher, params))
        if (
            "RETURN e.qid AS qid, e.label AS label, e.description AS description"
            not in cypher
        ):
            if "db.create.setNodeVectorProperty" in cypher:
                self.embedded = True
            return []
        if params["top_k"] == 1 and not self.embedded:
            return [
                {
                    "qid": "Q7186",
                    "label": "Marie Curie",
                    "description": "Polish-born physicist and chemist",
                }
            ]
        return []


@dataclass
class _RecordingEmbedder:
    batches: list[list[str]] = field(default_factory=list)
    vector_dim: int = 4

    async def generate_embeddings_batch(self, texts: list[str]):
        self.batches.append(list(texts))
        return [[0.1] * self.vector_dim for _ in texts]


@dataclass
class _RecordingVectorStore:
    points: list[dict] | None = None

    async def upsert_passage_points(self, points: list[dict]) -> None:
        self.points = points


@dataclass
class _FixedTopKGraph:
    selected_qids: list[str] = field(default_factory=list)
    embedded_qids: list[str] = field(default_factory=list)
    entities: list[dict] = field(
        default_factory=lambda: [
            {
                "qid": "Q1000",
                "label": "Top entity",
                "description": "highest popularity",
                "inbound_link_count": 100,
                "embedding": False,
            },
            {
                "qid": "Q2",
                "label": "Second entity",
                "description": "second highest popularity",
                "inbound_link_count": 99,
                "embedding": False,
            },
            {
                "qid": "Q9999",
                "label": "Outside entity",
                "description": "outside target set",
                "inbound_link_count": 1,
                "embedding": False,
            },
        ]
    )

    async def run_cypher(self, cypher: str, params: dict):
        if (
            "RETURN e.qid AS qid, e.label AS label, e.description AS description"
            in cypher
        ):
            candidates = sorted(
                self.entities,
                key=lambda row: (-row["inbound_link_count"], row["qid"]),
            )[: params["top_k"]]
            rows = [
                {
                    "qid": row["qid"],
                    "label": row["label"],
                    "description": row["description"],
                }
                for row in candidates
                if not row["embedding"]
            ][: params["batch_size"]]
            self.selected_qids.extend(row["qid"] for row in rows)
            return rows
        if "db.create.setNodeVectorProperty" in cypher:
            for row in params["batch"]:
                self.embedded_qids.append(row["qid"])
                for entity in self.entities:
                    if entity["qid"] == row["qid"]:
                        entity["embedding"] = True
            return [{"n": len(params["batch"])}]
        return []


def _checkpoint_store(tmp_path, name: str) -> CheckpointStore:
    return CheckpointStore(path=str(tmp_path / f"{name}.db"))


def _dlq(tmp_path, name: str) -> DLQWriter:
    return DLQWriter(path=tmp_path / f"{name}.jsonl")


def _pause(tmp_path, name: str) -> PauseController:
    return PauseController(asyncio.Event(), tmp_path / f"{name}.pause")


def _progress(pass_name: str) -> ProgressReporter:
    return ProgressReporter(pass_name, total_estimate=None, every=1_000_000)


async def test_pass1_recomputes_inbound_link_counts_before_completion(
    tmp_path, monkeypatch
):
    graph = _RecordingGraph()
    writer = Pass1EntityWriter(
        graph_store=graph,
        checkpoint_store=_checkpoint_store(tmp_path, "pass1"),
        dlq=_dlq(tmp_path, "pass1"),
        pause_controller=_pause(tmp_path, "pass1"),
        progress=_progress("pass1"),
        batch_size=10,
    )
    entity = WikidataEntity(
        qid="Q1",
        label="Universe",
        description="totality of space and time",
        wikipedia_title="Universe",
        claims={
            "P31": [
                {
                    "mainsnak": {
                        "snaktype": "value",
                        "datavalue": {"value": {"id": "Q35120"}},
                    }
                }
            ]
        },
        raw={},
    )
    monkeypatch.setattr(
        pass1_entities, "WikidataJsonDumpParser", lambda _path: [entity]
    )

    await writer.run("ignored.json.bz2")

    assert any(
        "OPTIONAL MATCH (src:Entity)-[r]->(e)" in cypher for cypher, _ in graph.calls
    )
    row = writer._ckpt.get("pass1")
    assert row is not None
    assert row.status == "complete"
    assert row.last_committed_id == "1"


async def test_pass1b_embeds_label_with_description_context(tmp_path):
    graph = _SequentialSelectGraph()
    embedder = _RecordingEmbedder(vector_dim=3)
    writer = Pass1bEntityVectorWriter(
        graph_store=graph,
        embedding_adapter=embedder,
        checkpoint_store=_checkpoint_store(tmp_path, "pass1b"),
        dlq=_dlq(tmp_path, "pass1b"),
        pause_controller=_pause(tmp_path, "pass1b"),
        progress=_progress("pass1b"),
        top_k=1,
        batch_size=1,
        dim=3,
    )

    await writer.run()

    assert embedder.batches == [["Marie Curie\nPolish-born physicist and chemist"]]
    assert any("db.create.setNodeVectorProperty" in cypher for cypher, _ in graph.calls)


async def test_pass1b_targets_fixed_top_k_without_qid_cursor_skips(tmp_path):
    graph = _FixedTopKGraph()
    embedder = _RecordingEmbedder(vector_dim=3)
    writer = Pass1bEntityVectorWriter(
        graph_store=graph,
        embedding_adapter=embedder,
        checkpoint_store=_checkpoint_store(tmp_path, "pass1b-fixed-top-k"),
        dlq=_dlq(tmp_path, "pass1b-fixed-top-k"),
        pause_controller=_pause(tmp_path, "pass1b-fixed-top-k"),
        progress=_progress("pass1b"),
        top_k=2,
        batch_size=1,
        dim=3,
    )

    await writer.run(start_from="0")

    assert graph.embedded_qids == ["Q1000", "Q2"]
    assert "Q9999" not in graph.embedded_qids
    assert embedder.batches == [
        ["Top entity\nhighest popularity"],
        ["Second entity\nsecond highest popularity"],
    ]


async def test_pass2_flush_batch_resolves_mentions_and_embeds_with_context(tmp_path):
    graph = _RecordingGraph()
    vector_store = _RecordingVectorStore()
    embedder = _RecordingEmbedder(vector_dim=3)
    writer = Pass2PassageWriter(
        graph_store=graph,
        vector_store=vector_store,
        embedding_adapter=embedder,
        checkpoint_store=_checkpoint_store(tmp_path, "pass2"),
        dlq=_dlq(tmp_path, "pass2"),
        pause_controller=_pause(tmp_path, "pass2"),
        progress=_progress("pass2"),
        sitelink_resolver=None,
    )
    batch = [
        PassageChunk(
            passage_id="p1",
            text="Curie isolated radium and advanced modern physics.",
            article_title="Marie Curie",
            article_qid="Q7186",
            section_title="Legacy",
            section_ordinal=2,
            chunk_ordinal=0,
            token_estimate=10,
            text_hash="hash1",
            mention_titles=["Radium", "Marie Curie", "Physics", "Physics"],
        )
    ]

    await writer._flush_batch(batch)

    assert embedder.batches == [
        [
            "Article: Marie Curie\nSection: Legacy\nCurie isolated radium and advanced modern physics."
        ]
    ]
    assert graph.aura_batch == [
        {
            "passage_id": "p1",
            "text": "Curie isolated radium and advanced modern physics.",
            "article_title": "Marie Curie",
            "article_qid": "Q7186",
            "section_title": "Legacy",
            "section_ordinal": 2,
            "chunk_ordinal": 0,
            "token_estimate": 10,
            "text_hash": "hash1",
            "mention_qids": ["Q113145171", "Q413"],
        }
    ]
    assert vector_store.points == [
        {
            "id": "p1",
            "vector": [0.1, 0.1, 0.1],
            "payload": {"passage_id": "p1", "qid": "Q7186"},
        }
    ]


def test_pass2_chunker_skips_low_signal_sections_and_normalizes_mentions(tmp_path):
    writer = Pass2PassageWriter(
        graph_store=_RecordingGraph(),
        vector_store=_RecordingVectorStore(),
        embedding_adapter=_RecordingEmbedder(),
        checkpoint_store=_checkpoint_store(tmp_path, "pass2-chunker"),
        dlq=_dlq(tmp_path, "pass2-chunker"),
        pause_controller=_pause(tmp_path, "pass2-chunker"),
        progress=_progress("pass2"),
        sitelink_resolver=None,
    )
    page = EnwikiPage(
        pageid=1,
        title="Marie Curie",
        wikitext=(
            "Marie Curie studied radioactivity and chemistry.\n"
            "\n== Legacy ==\n[[Radium]] shaped modern [[Physics#History|physics]].\n"
            "\n== References ==\n[[File:Curie.jpg]] [[Category:Scientists]]\n"
        ),
        is_redirect=False,
    )

    chunks = list(writer._chunk_page(page, "Q7186"))

    assert {chunk.section_title for chunk in chunks} == {"lead", "Legacy"}
    legacy_chunk = next(chunk for chunk in chunks if chunk.section_title == "Legacy")
    assert legacy_chunk.mention_titles == ["Radium", "Physics"]


async def test_pass1_aborts_without_advancing_checkpoint_on_failed_batch(
    tmp_path, monkeypatch
):
    graph = _RecordingGraph()
    writer = Pass1EntityWriter(
        graph_store=graph,
        checkpoint_store=_checkpoint_store(tmp_path, "pass1-fail"),
        dlq=_dlq(tmp_path, "pass1-fail"),
        pause_controller=_pause(tmp_path, "pass1-fail"),
        progress=_progress("pass1"),
        batch_size=1,
    )
    entity = WikidataEntity(
        qid="Q1",
        label="Universe",
        description=None,
        wikipedia_title="Universe",
        claims={},
        raw={},
    )
    monkeypatch.setattr(
        pass1_entities, "WikidataJsonDumpParser", lambda _path: [entity]
    )

    async def _always_fail(**_kwargs):
        return False

    monkeypatch.setattr(pass1_entities, "retry_record", _always_fail)

    with pytest.raises(RuntimeError, match="Pass 1 aborted"):
        await writer.run("ignored.json.bz2")

    row = writer._ckpt.get("pass1")
    assert row is not None
    assert row.status == "aborted"
    assert row.last_committed_id is None


async def test_pass2_aborts_without_advancing_checkpoint_on_failed_flush(
    tmp_path, monkeypatch
):
    graph = _RecordingGraph()
    vector_store = _RecordingVectorStore()
    embedder = _RecordingEmbedder(vector_dim=3)

    async def _resolve_sitelink(_title: str) -> str | None:
        return "Q7186"

    writer = Pass2PassageWriter(
        graph_store=graph,
        vector_store=vector_store,
        embedding_adapter=embedder,
        checkpoint_store=_checkpoint_store(tmp_path, "pass2-fail"),
        dlq=_dlq(tmp_path, "pass2-fail"),
        pause_controller=_pause(tmp_path, "pass2-fail"),
        progress=_progress("pass2"),
        sitelink_resolver=_resolve_sitelink,
        batch_size=1,
    )
    page = EnwikiPage(
        pageid=7,
        title="Zeta",
        wikitext="Zeta has enough text for one chunk.",
        is_redirect=False,
    )
    monkeypatch.setattr(pass2_passages, "EnwikiXmlDumpParser", lambda _path: [page])

    async def _flush_fail(self, batch):
        return False

    monkeypatch.setattr(Pass2PassageWriter, "_flush_batch", _flush_fail)

    with pytest.raises(RuntimeError, match="Pass 2 aborted"):
        await writer.run("ignored.xml.bz2")

    row = writer._ckpt.get("pass2")
    assert row is not None
    assert row.status == "aborted"
    assert row.last_committed_id is None
