from __future__ import annotations

import sys
from pathlib import Path


def _add_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    ingestion_root = repo_root / "gui_ingestion_app" / "ingestion"
    if ingestion_root.exists():
        sys.path.insert(0, str(ingestion_root))


if __name__ == "__main__":
    _add_repo_paths()
    from ingestion.gui_app import main

    main()
