"""Wave 3 Stream C — lifecycle unit tests (DLQ + retry)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
if str(_SRC_API) not in sys.path:
    sys.path.insert(0, str(_SRC_API))

from ingestion.lifecycle import DLQWriter, retry_record  # noqa: E402


def test_dlq_writer_appends_and_iterates(tmp_path):
    dlq = DLQWriter(path=tmp_path / "dlq.jsonl")
    dlq.write(
        pass_name="pass1",
        record_id="Q100",
        error_class="RuntimeError",
        error_detail="boom",
    )
    dlq.write(
        pass_name="pass1",
        record_id="Q101",
        error_class="KeyError",
        error_detail="missing key",
    )
    records = list(dlq.iter_records())
    assert len(records) == 2
    assert records[0]["record_id"] == "Q100"
    assert records[1]["error_class"] == "KeyError"


def test_dlq_clear_removes_file(tmp_path):
    dlq = DLQWriter(path=tmp_path / "dlq.jsonl")
    dlq.write(pass_name="p", record_id="x", error_class="E", error_detail="d")
    assert dlq.path.exists()
    dlq.clear()
    assert not dlq.path.exists()


def test_retry_record_returns_true_on_success():
    calls = {"n": 0}

    async def _good():
        calls["n"] += 1

    ok = asyncio.run(
        retry_record(fn=_good, record_id="x", max_retries=3)
    )
    assert ok is True
    assert calls["n"] == 1


def test_retry_record_retries_then_succeeds():
    attempts = {"n": 0}

    async def _flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("flaky")

    ok = asyncio.run(
        retry_record(fn=_flaky, record_id="x", max_retries=5, base_delay_s=0.01)
    )
    assert ok is True
    assert attempts["n"] == 3


def test_retry_record_returns_false_after_exhaustion():
    async def _always_fail():
        raise RuntimeError("permanent")

    ok = asyncio.run(
        retry_record(fn=_always_fail, record_id="x", max_retries=2, base_delay_s=0.01)
    )
    assert ok is False
