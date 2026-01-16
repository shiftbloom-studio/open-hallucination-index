from __future__ import annotations

import sys
from pathlib import Path


def _add_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if src_root.exists():
        sys.path.insert(0, str(src_root))


if __name__ == "__main__":
    _add_repo_paths()
    from ingestion.gui_app import main

    main()
