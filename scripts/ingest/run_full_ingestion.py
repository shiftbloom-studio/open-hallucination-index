"""Wave 3 Stream C — ingestion CLI wrapper.

Bootstraps the DI container (so the ingestion orchestrator has the
same graph/vector/embedding adapters the Lambda runtime uses), then
hands off to :func:`ingestion.orchestrator.run`.

Invocation (PC mode, Fabian's machine):

    python scripts/ingest/run_full_ingestion.py \\
        --wikidata /data/wikidata-20260401-all.json.bz2 \\
        --enwiki   /data/enwiki-20260401-pages-articles.xml.bz2

Checkpoint-resumable — re-running with the same flags picks up at the
last committed record for each pass.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

# Make src/api importable when running from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_API = _REPO_ROOT / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))


async def _amain(argv: list[str] | None = None) -> int:
    # Import after sys.path is set so config.* resolves.
    from config.dependencies import _initialize_adapters  # noqa: PLC0415
    from ingestion.orchestrator import run as orchestrator_run  # noqa: PLC0415

    logging.basicConfig(
        level=getattr(logging, os.environ.get("OHI_LOG_LEVEL", "INFO")),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    await _initialize_adapters()
    return await orchestrator_run(argv)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    sys.exit(main())
