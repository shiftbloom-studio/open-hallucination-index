"""Root-level conftest — prepends the repo root to sys.path so tests under
`tests/` can import `scripts.*` without needing a packaged install.

The src/api package has its own pyproject.toml + editable install for its
own tests. This conftest is for the repo-root test tree only (infra smoke
tests and benchmark harness tests).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
