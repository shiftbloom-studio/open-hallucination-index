"""Cross-worktree import guard for ``tests/unit/``, session-scoped.

Fabian runs multiple worktrees against a single shared ``.venv`` whose
editable install points at the MAIN checkout's ``src/api`` tree. Any
test that imports ``from interfaces.*`` / ``from pipeline.*`` / etc.
without first inserting THIS worktree's ``src/api`` on ``sys.path``
would silently resolve against the main checkout — producing either a
stale Pipeline signature (missing newly-added kwargs) or, worse, a
test that passes against code that was never written in this worktree.

This conftest runs ONCE at ``tests/unit/`` collection time, before any
test file in this subtree loads. It:

1. Inserts the worktree's ``src/api`` at the front of ``sys.path`` so
   ``from interfaces.*`` resolves here first.
2. Purges every already-cached flat-namespace sibling package
   (``adapters``, ``interfaces``, ``models``, ``pipeline``, ``config``,
   ``server``, ``services``) from ``sys.modules`` — the editable
   install may have populated them during pytest's own bootstrap, and
   those entries would shadow the fresh re-resolution.

Per-file purge snippets (e.g. in
``tests/unit/adapters/test_nli_gemini.py``) remain as a belt-and-
braces second layer so the tests also work when run from directories
where this conftest isn't in scope. Those file-level purges should
guard themselves against re-purging a cache that's already from this
worktree (see ``tests/unit/pipeline/test_compute_posteriors.py`` for
the canonical guarded pattern) — otherwise a second purge re-creates
the module objects and breaks earlier test files' ``is`` identity
assertions.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_API = Path(__file__).resolve().parents[2] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

_FLAT_PACKAGES = {
    "adapters",
    "interfaces",
    "models",
    "pipeline",
    "config",
    "server",
    "services",
    "ingestion",  # Wave 3 Stream C
}
for _cached_name in list(sys.modules):
    _root = _cached_name.split(".", 1)[0]
    if _root in _FLAT_PACKAGES:
        del sys.modules[_cached_name]
