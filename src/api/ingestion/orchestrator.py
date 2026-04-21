"""Wave 3 Stream C — ingestion orchestrator CLI.

Coordinates the four passes with shared lifecycle, checkpoint, and
progress infrastructure. Entry point for :mod:`scripts.ingest.run_full_ingestion`.

CLI (delegate from the scripts/ wrapper):

    --skip-pass1 / --skip-pass1b / --skip-pass2 / --skip-pass3
    --only-integrity      run Pass 3 only (quick after-the-fact report)
    --only-range start:end
    --dry-run
    --retry-dlq
    --force-restart       clear checkpoint state before running

Runtime orchestrator is **async-first** — each pass is a coroutine that
takes shared lifecycle helpers and yields back control at batch
boundaries so SIGUSR1 pause can flush cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass

from ingestion.checkpoint_store import CheckpointStore
from ingestion.lifecycle import (
    DLQWriter,
    ProgressReporter,
    install_pause_handler,
)

logger = logging.getLogger(__name__)


@dataclass
class RunArgs:
    wikidata_path: str | None
    enwiki_path: str | None
    skip_pass1: bool
    skip_pass1b: bool
    skip_pass2: bool
    skip_pass3: bool
    only_integrity: bool
    dry_run: bool
    retry_dlq: bool
    force_restart: bool
    only_range: tuple[str | None, str | None]
    top_k_entities: int
    report_out: str | None


def _parse_args(argv: list[str] | None = None) -> RunArgs:
    p = argparse.ArgumentParser(
        prog="ohi-ingestion",
        description="Wave 3 corpus ingestion orchestrator (enwiki + Wikidata).",
    )
    p.add_argument("--wikidata", dest="wikidata_path", help="Path to wikidata-*.json.bz2")
    p.add_argument("--enwiki", dest="enwiki_path", help="Path to enwiki-*-pages-articles.xml.bz2")
    p.add_argument("--skip-pass1", action="store_true")
    p.add_argument("--skip-pass1b", action="store_true")
    p.add_argument("--skip-pass2", action="store_true")
    p.add_argument("--skip-pass3", action="store_true")
    p.add_argument("--only-integrity", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retry-dlq", action="store_true")
    p.add_argument("--force-restart", action="store_true")
    p.add_argument(
        "--only-range",
        help="start:end QID / page-title range (both sides optional)",
    )
    p.add_argument("--top-k-entities", type=int, default=1_800_000)
    p.add_argument("--report-out", default=None)
    args = p.parse_args(argv)

    start: str | None = None
    end: str | None = None
    if args.only_range:
        parts = args.only_range.split(":", 1)
        start = parts[0] or None
        end = parts[1] if len(parts) > 1 else None
    return RunArgs(
        wikidata_path=args.wikidata_path,
        enwiki_path=args.enwiki_path,
        skip_pass1=args.skip_pass1,
        skip_pass1b=args.skip_pass1b,
        skip_pass2=args.skip_pass2,
        skip_pass3=args.skip_pass3,
        only_integrity=args.only_integrity,
        dry_run=args.dry_run,
        retry_dlq=args.retry_dlq,
        force_restart=args.force_restart,
        only_range=(start, end),
        top_k_entities=args.top_k_entities,
        report_out=args.report_out,
    )


async def run(argv: list[str] | None = None) -> int:
    """Top-level runner. Delegates per-pass writes to the Pass*
    classes. Imports are lazy so the CLI stays importable in
    environments without Neo4j / Qdrant clients."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, os.environ.get("OHI_LOG_LEVEL", "INFO")),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    store = CheckpointStore()
    if args.force_restart:
        logger.warning("--force-restart: clearing all checkpoint rows")
        store.clear()

    dlq = DLQWriter()
    pause = install_pause_handler()

    if args.retry_dlq:
        raise SystemExit(
            "--retry-dlq is not wired in the Wave 3 orchestrator yet; "
            "rerun the affected pass normally from the last checkpoint or use --force-restart"
        )

    if args.dry_run:
        logger.info("--dry-run enabled; parsers will enumerate but not write")

    # Dependencies wired via DI later in this file — keep imports lazy.
    from config.dependencies import (  # noqa: PLC0415
        _cleanup_adapters,
        _initialize_adapters,
        get_evidence_collector,  # noqa: F401 — triggers lifespan init
        get_graph_store,
        get_vector_store,
    )

    await _initialize_adapters()
    try:
        if args.only_integrity:
            return await _run_pass3(args, store, dlq, pause)

        graph_store = get_graph_store()
        vector_store = get_vector_store()
        from config.dependencies import _embedding_adapter  # noqa: PLC0415

        if _embedding_adapter is None:
            raise RuntimeError("Embedding adapter did not initialize")

        if not args.skip_pass1:
            if not args.wikidata_path:
                raise SystemExit("--wikidata required unless --skip-pass1")
            await _run_pass1(args, store, dlq, pause, graph_store)

        if not args.skip_pass1b:
            await _run_pass1b(args, store, dlq, pause, graph_store, _embedding_adapter)

        if not args.skip_pass2:
            if not args.enwiki_path:
                raise SystemExit("--enwiki required unless --skip-pass2")
            await _run_pass2(args, store, dlq, pause, graph_store, vector_store, _embedding_adapter)

        if not args.skip_pass3:
            await _run_pass3(args, store, dlq, pause)
        return 0
    finally:
        await _cleanup_adapters()


async def _run_pass1(args: RunArgs, store, dlq, pause, graph_store) -> None:
    from ingestion.pass1_entities import Pass1EntityWriter  # noqa: PLC0415

    progress = ProgressReporter("pass1", total_estimate=None)
    writer = Pass1EntityWriter(
        graph_store=graph_store,
        checkpoint_store=store,
        dlq=dlq,
        pause_controller=pause,
        progress=progress,
    )
    start_from = args.only_range[0] or _existing_high_water(store, "pass1")
    await writer.run(args.wikidata_path, start_from=start_from)


async def _run_pass1b(args: RunArgs, store, dlq, pause, graph_store, embedding_adapter) -> None:
    from ingestion.pass1b_entity_vectors import Pass1bEntityVectorWriter  # noqa: PLC0415

    embedding_dim = int(getattr(embedding_adapter, "embedding_dimension", 384))
    top_k_entities = args.top_k_entities
    if embedding_dim > 384 and args.top_k_entities >= 1_800_000:
        # Keep the default footprint near the previous 384-dim budget.
        top_k_entities = max(100_000, int(args.top_k_entities * (384 / embedding_dim)))
        logger.warning(
            "Pass1b top_k reduced for embedding dim=%d: %d -> %d",
            embedding_dim,
            args.top_k_entities,
            top_k_entities,
        )

    progress = ProgressReporter("pass1b", total_estimate=top_k_entities)
    writer = Pass1bEntityVectorWriter(
        graph_store=graph_store,
        embedding_adapter=embedding_adapter,
        checkpoint_store=store,
        dlq=dlq,
        pause_controller=pause,
        progress=progress,
        top_k=top_k_entities,
        dim=embedding_dim,
    )
    start_from = args.only_range[0] or _existing_high_water(store, "pass1b")
    await writer.run(start_from=start_from)


async def _run_pass2(
    args: RunArgs, store, dlq, pause, graph_store, vector_store, embedding_adapter
) -> None:
    from ingestion.pass2_passages import Pass2PassageWriter  # noqa: PLC0415

    progress = ProgressReporter("pass2", total_estimate=None)

    async def _resolve_sitelink(title: str) -> str | None:
        rows = await graph_store.run_cypher(
            "MATCH (e:Entity {wikipedia_title: $title}) RETURN e.qid AS qid LIMIT 1",
            {"title": title},
        )
        row = list(rows or [])[:1]
        if not row:
            return None
        first = row[0]
        if isinstance(first, dict):
            return first.get("qid")
        return first[0]

    writer = Pass2PassageWriter(
        graph_store=graph_store,
        vector_store=vector_store,
        embedding_adapter=embedding_adapter,
        checkpoint_store=store,
        dlq=dlq,
        pause_controller=pause,
        progress=progress,
        sitelink_resolver=_resolve_sitelink,
    )
    start_from = args.only_range[0] or _existing_high_water(store, "pass2")
    await writer.run(args.enwiki_path, start_from=start_from)


async def _run_pass3(args: RunArgs, store, dlq, pause) -> int:
    from adapters.entity_resolver import EntityResolver  # noqa: PLC0415
    from config.dependencies import get_graph_store, get_vector_store  # noqa: PLC0415
    from ingestion.pass3_integrity import (  # noqa: PLC0415
        generate_integrity_report,
        write_integrity_report,
    )

    graph_store = get_graph_store()
    vector_store = get_vector_store()
    from config.dependencies import _embedding_adapter  # noqa: PLC0415

    resolver = EntityResolver(graph_store=graph_store, embedding_adapter=_embedding_adapter)
    golden = [
        "Marie Curie won two Nobel Prizes.",
        "Albert Einstein was born in Germany.",
        "Charles Darwin wrote On the Origin of Species.",
    ]
    report = await generate_integrity_report(
        graph_store=graph_store,
        vector_store=vector_store,
        golden_claim_set=golden,
        entity_resolver=resolver,
    )
    out = write_integrity_report(report)
    logger.info("Integrity: %s", report)
    store.set_status("pass3", "complete", notes=str(out))
    return 0


def _existing_high_water(store: CheckpointStore, pass_name) -> str | None:
    row = store.get(pass_name)
    if row is None:
        return None
    if (
        pass_name in {"pass1", "pass1b", "pass2"}
        and row.last_committed_id
        and not row.last_committed_id.isdigit()
    ):
        raise RuntimeError(
            f"{pass_name} checkpoint uses a legacy unsafe resume marker "
            f"({row.last_committed_id!r}); rerun with --force-restart once"
        )
    return row.last_committed_id


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(argv))


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "run"]
