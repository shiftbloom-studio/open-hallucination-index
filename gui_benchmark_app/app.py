from __future__ import annotations

import sys
from pathlib import Path


def _add_repo_paths() -> None:
    app_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(app_root))


if __name__ == "__main__":
    _add_repo_paths()
    from benchmark.gui_app import main

    main()
