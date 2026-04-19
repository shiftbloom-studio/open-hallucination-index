"""Wave 3 Stream C ingestion pass regressions.

Focused tests around the data-quality improvements that materially
change what lands in Aura and Qdrant.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ingestion.checkpoint_store import CheckpointStore
from ingestion.dump_parsers import EnwikiPage, WikidataEntity
from ingestion.lifecycle import DLQWriter, PauseController, ProgressReporter
from ingestion.pass1_entities import Pass1EntityWriter
from ingestion.pass1b_entity_vectors import Pass1bEntityVectorWriter
from ingestion.pass2_passages import PassageChunk, Pass2PassageWriter
import ingestion.pass1_entities as pass1_entities


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

    async def run_cypher(self, cypher: str, params: dict):
        self.calls.append((cypher, params))
        if "RETURN e.qid AS qid, e.label AS label, e.description AS description" not in cypher:
            return []
        if params["start_qid"] is None:
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
    monkeypatch.setattr(pass1_entities, "WikidataJsonDumpParser", lambda _path: [entity])

    await writer.run("ignored.json.bz2")

    assert any(
        "OPTIONAL MATCH (src:Entity)-[r]->(e)" in cypher for cypher, _ in graph.calls
    )
    row = writer._ckpt.get("pass1")
    assert row is not None
    assert row.status == "complete"
    assert row.last_committed_id == "Q1"


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
    assert any(
        "db.create.setNodeVectorProperty" in cypher for cypher, _ in graph.calls
    )


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
