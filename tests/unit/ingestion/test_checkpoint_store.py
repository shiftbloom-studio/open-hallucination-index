"""Wave 3 Stream C — checkpoint_store unit tests.

Covers the crash-mid-batch resume invariant and the pause ↔
running state transitions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from ingestion.checkpoint_store import CheckpointStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(path=str(tmp_path / "ckpt.db"))


def test_initial_get_returns_none(store):
    assert store.get("pass1") is None


def test_start_pass_creates_row(store):
    store.start_pass("pass1")
    row = store.get("pass1")
    assert row is not None
    assert row.status == "running"
    assert row.records_processed == 0


def test_advance_updates_high_water_and_counter(store):
    store.start_pass("pass1")
    store.advance("pass1", last_committed_id="Q100", records_in_batch=42)
    row = store.get("pass1")
    assert row.last_committed_id == "Q100"
    assert row.records_processed == 42


def test_pause_then_resume_preserves_counters(store):
    store.start_pass("pass1")
    store.advance("pass1", last_committed_id="Q500", records_in_batch=100)
    store.set_status("pass1", "paused")
    row_paused = store.get("pass1")
    assert row_paused.status == "paused"
    assert row_paused.records_processed == 100

    store.start_pass("pass1")  # resume
    row_resumed = store.get("pass1")
    assert row_resumed.status == "running"
    assert row_resumed.records_processed == 100  # preserved
    assert row_resumed.last_committed_id == "Q500"


def test_increment_dlq(store):
    store.start_pass("pass2")
    store.increment_dlq("pass2", n=5)
    store.increment_dlq("pass2", n=3)
    assert store.get("pass2").dlq_size == 8


def test_clear_specific_pass(store):
    store.start_pass("pass1")
    store.start_pass("pass2")
    store.clear("pass1")
    assert store.get("pass1") is None
    assert store.get("pass2") is not None


def test_clear_all(store):
    store.start_pass("pass1")
    store.start_pass("pass2")
    store.clear()
    assert store.get("pass1") is None
    assert store.get("pass2") is None


def test_all_rows_returns_all(store):
    store.start_pass("pass1")
    store.start_pass("pass1b")
    store.start_pass("pass2")
    all_rows = store.all_rows()
    assert {r.pass_name for r in all_rows} == {"pass1", "pass1b", "pass2"}
