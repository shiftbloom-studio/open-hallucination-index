"""Unit tests for ``src/api/server/jobs.py`` — DynamoDB + self-invoke helpers.

Stream D2 replaces the single long-running sync ``POST /verify`` with a
three-piece async handshake:

1. ``POST /api/v2/verify`` writes a *pending* record to DynamoDB and
   returns ``202 {"job_id": ...}`` fast.
2. The Lambda self-async-invokes with a synthetic APIGW v2 event whose
   body carries ``{__async_verify: True, depth, job_id, text, ticket}``.
3. The async handler drives ``pipeline.verify()`` with phase callbacks
   that ``update_item`` the record at each of D1's five natural
   boundaries; ``complete_job`` / ``fail_job`` land the terminal state.

``src/api/server/jobs.py`` is the collar that owns these three surfaces.
Every DynamoDB round-trip and the ``lambda:InvokeFunction`` call go
through this module so the boto3 client creation stays module-level
(warm-start reuse, per brief) while still being override-able from
tests via the module-private ``_get_*_client`` accessors.

All tests stub the AWS SDK. No real DynamoDB, no real Lambda, no $.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# sys.path + cross-worktree sys.modules purge (mirrors the pattern adopted by
# tests/unit/pipeline/test_compute_posteriors.py — see the note there about
# the shared .venv's editable install pointing at the MAIN checkout's
# src/api, which silently shadows this worktree's server/jobs.py otherwise).
# tests/unit/conftest.py does the session-scoped purge first; this guarded
# local purge is a second layer so this file also runs when invoked
# in isolation (e.g. pytest -q tests/unit/server/test_jobs.py).
# ---------------------------------------------------------------------------
_SRC_API = Path(__file__).resolve().parents[3] / "src" / "api"
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
}
_cached_iface = sys.modules.get("interfaces")
_cached_iface_file = getattr(_cached_iface, "__file__", "") or ""
if _cached_iface is None or str(_SRC_API) not in _cached_iface_file:
    for _cached_name in list(sys.modules):
        _root = _cached_name.split(".", 1)[0]
        if _root in _FLAT_PACKAGES:
            del sys.modules[_cached_name]


# ---------------------------------------------------------------------------
# Stubs for boto3 DynamoDB + Lambda clients.
# ---------------------------------------------------------------------------
class _StubDynamoDBClient:
    """Record put_item / get_item / update_item calls; hand back canned
    responses. Only the shapes we actually produce are validated."""

    def __init__(self) -> None:
        self.put_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self._get_responses: list[dict[str, Any]] = []

    def prime_get_item(self, response: dict[str, Any]) -> None:
        self._get_responses.append(response)

    def put_item(self, **kwargs: Any) -> dict[str, Any]:
        self.put_calls.append(kwargs)
        return {}

    def update_item(self, **kwargs: Any) -> dict[str, Any]:
        self.update_calls.append(kwargs)
        return {}

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        if self._get_responses:
            return self._get_responses.pop(0)
        return {}


class _StubLambdaClient:
    def __init__(self) -> None:
        self.invoke_calls: list[dict[str, Any]] = []

    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        self.invoke_calls.append(kwargs)
        return {"StatusCode": 202}


@pytest.fixture
def stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal env so jobs.py helpers don't KeyError on module globals."""
    monkeypatch.setenv("JOBS_TABLE_NAME", "ohi-verify-jobs-test")
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "ohi-api-test")
    monkeypatch.setenv("OHI_ASYNC_VERIFY_TTL_SECONDS", "3600")


@pytest.fixture
def stub_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_StubDynamoDBClient, _StubLambdaClient]:
    from server import jobs

    ddb = _StubDynamoDBClient()
    lam = _StubLambdaClient()
    monkeypatch.setattr(jobs, "_get_dynamodb_client", lambda: ddb)
    monkeypatch.setattr(jobs, "_get_lambda_client", lambda: lam)
    return ddb, lam


# ---------------------------------------------------------------------------
# new_job_id / new_invoke_ticket
# ---------------------------------------------------------------------------


def test_new_job_id_is_uuid4_string(stub_env: None) -> None:
    from uuid import UUID

    from server import jobs

    job_id = jobs.new_job_id()
    parsed = UUID(job_id)
    assert parsed.version == 4
    # UUIDs render canonical hex-with-dashes
    assert str(parsed) == job_id


def test_new_invoke_ticket_is_opaque_and_not_reused(stub_env: None) -> None:
    from server import jobs

    t1 = jobs.new_invoke_ticket()
    t2 = jobs.new_invoke_ticket()
    assert t1 != t2
    # secrets.token_urlsafe(32) yields ~43 url-safe chars; accept >= 32.
    assert len(t1) >= 32
    assert all(c.isalnum() or c in "-_" for c in t1)


# ---------------------------------------------------------------------------
# create_job
# ---------------------------------------------------------------------------


def test_create_job_writes_pending_record_with_all_required_fields(
    stub_env: None, stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient]
) -> None:
    ddb, _lam = stub_clients
    from server import jobs

    jobs.create_job(job_id="jid-1", text="hello", ticket="tkt-a")

    assert len(ddb.put_calls) == 1
    call = ddb.put_calls[0]
    assert call["TableName"] == "ohi-verify-jobs-test"
    item = call["Item"]
    assert item["job_id"] == {"S": "jid-1"}
    assert item["status"] == {"S": "pending"}
    assert item["phase"] == {"S": "queued"}
    assert item["text"] == {"S": "hello"}
    assert item["invoke_ticket"] == {"S": "tkt-a"}
    # timestamps are integer epoch seconds; ttl > created_at
    created_at = int(item["created_at"]["N"])
    updated_at = int(item["updated_at"]["N"])
    ttl = int(item["ttl"]["N"])
    assert created_at > 0
    assert updated_at == created_at
    assert ttl == created_at + 3600


def test_create_job_honours_custom_ttl_env(
    monkeypatch: pytest.MonkeyPatch,
    stub_env: None,
    stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient],
) -> None:
    monkeypatch.setenv("OHI_ASYNC_VERIFY_TTL_SECONDS", "60")
    ddb, _ = stub_clients
    from server import jobs

    jobs.create_job(job_id="jid-2", text="x", ticket="tkt-b")
    item = ddb.put_calls[0]["Item"]
    created_at = int(item["created_at"]["N"])
    ttl = int(item["ttl"]["N"])
    assert ttl - created_at == 60


# ---------------------------------------------------------------------------
# update_phase
# ---------------------------------------------------------------------------


def test_update_phase_updates_phase_and_updated_at(
    stub_env: None, stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient]
) -> None:
    ddb, _ = stub_clients
    from server import jobs

    jobs.update_phase("jid-1", "decomposing")

    assert len(ddb.update_calls) == 1
    call = ddb.update_calls[0]
    assert call["TableName"] == "ohi-verify-jobs-test"
    assert call["Key"] == {"job_id": {"S": "jid-1"}}
    # UpdateExpression must set phase and updated_at atomically
    ue = call["UpdateExpression"]
    assert "phase" in ue or ":phase" in ue
    assert "updated_at" in ue
    values = call["ExpressionAttributeValues"]
    # phase value present
    assert any(v == {"S": "decomposing"} for v in values.values())


# ---------------------------------------------------------------------------
# complete_job / fail_job
# ---------------------------------------------------------------------------


def test_complete_job_stores_status_done_and_serialized_result(
    stub_env: None, stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient]
) -> None:
    ddb, _ = stub_clients
    from server import jobs

    verdict = {"document_score": 0.87, "claims": [{"p_true": 0.9}]}
    jobs.complete_job("jid-1", verdict)

    assert len(ddb.update_calls) == 1
    call = ddb.update_calls[0]
    assert call["Key"] == {"job_id": {"S": "jid-1"}}
    values = call["ExpressionAttributeValues"]
    # status flipped to "done"
    assert any(v == {"S": "done"} for v in values.values())
    # result is JSON-serialised
    result_val = next(
        v["S"] for v in values.values() if isinstance(v, dict) and "S" in v and v["S"].startswith("{")
    )
    assert json.loads(result_val) == verdict
    # invoke_ticket is cleared so a replayed async-invoke cannot re-run
    assert any(v == {"S": ""} for v in values.values())


def test_fail_job_stores_status_error_and_message(
    stub_env: None, stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient]
) -> None:
    ddb, _ = stub_clients
    from server import jobs

    jobs.fail_job("jid-1", "ValueError: boom")

    assert len(ddb.update_calls) == 1
    call = ddb.update_calls[0]
    values = call["ExpressionAttributeValues"]
    assert any(v == {"S": "error"} for v in values.values())
    assert any(v == {"S": "ValueError: boom"} for v in values.values())
    assert any(v == {"S": ""} for v in values.values())  # ticket cleared


# ---------------------------------------------------------------------------
# get_job
# ---------------------------------------------------------------------------


def test_get_job_returns_none_when_item_missing(
    stub_env: None, stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient]
) -> None:
    ddb, _ = stub_clients
    from server import jobs

    # Stub returns {} i.e. no Item key
    assert jobs.get_job("nope") is None


def test_get_job_returns_parsed_pending_record(
    stub_env: None, stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient]
) -> None:
    ddb, _ = stub_clients
    ddb.prime_get_item(
        {
            "Item": {
                "job_id": {"S": "jid-7"},
                "status": {"S": "pending"},
                "phase": {"S": "queued"},
                "text": {"S": "Marie Curie"},
                "invoke_ticket": {"S": "tkt-x"},
                "created_at": {"N": "1760000000"},
                "updated_at": {"N": "1760000000"},
                "ttl": {"N": "1760003600"},
            }
        }
    )
    from server import jobs

    out = jobs.get_job("jid-7")
    assert out is not None
    assert out["job_id"] == "jid-7"
    assert out["status"] == "pending"
    assert out["phase"] == "queued"
    assert out["created_at"] == 1760000000
    assert out["updated_at"] == 1760000000


def test_get_job_parses_done_result_back_to_dict(
    stub_env: None, stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient]
) -> None:
    ddb, _ = stub_clients
    ddb.prime_get_item(
        {
            "Item": {
                "job_id": {"S": "jid-done"},
                "status": {"S": "done"},
                "phase": {"S": "assembling"},
                "invoke_ticket": {"S": ""},
                "created_at": {"N": "1760000000"},
                "updated_at": {"N": "1760000050"},
                "completed_at": {"N": "1760000050"},
                "ttl": {"N": "1760003600"},
                "result": {"S": '{"document_score": 0.42}'},
            }
        }
    )
    from server import jobs

    out = jobs.get_job("jid-done")
    assert out is not None
    assert out["status"] == "done"
    assert out["result"] == {"document_score": 0.42}
    assert out["completed_at"] == 1760000050


def test_get_job_surfaces_error_message(
    stub_env: None, stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient]
) -> None:
    ddb, _ = stub_clients
    ddb.prime_get_item(
        {
            "Item": {
                "job_id": {"S": "jid-err"},
                "status": {"S": "error"},
                "phase": {"S": "classifying"},
                "invoke_ticket": {"S": ""},
                "created_at": {"N": "1760000000"},
                "updated_at": {"N": "1760000050"},
                "completed_at": {"N": "1760000050"},
                "ttl": {"N": "1760003600"},
                "error": {"S": "NliAdapter: timeout"},
            }
        }
    )
    from server import jobs

    out = jobs.get_job("jid-err")
    assert out is not None
    assert out["status"] == "error"
    assert out["error"] == "NliAdapter: timeout"


# ---------------------------------------------------------------------------
# async_invoke_verify — synthetic APIGW v2 event + InvocationType=Event
# ---------------------------------------------------------------------------


def test_async_invoke_verify_sends_event_invocation_to_self(
    stub_env: None, stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient]
) -> None:
    _ddb, lam = stub_clients
    from server import jobs

    jobs.async_invoke_verify(
        job_id="jid-1", text="Marie Curie", ticket="tkt-a", depth=1
    )

    assert len(lam.invoke_calls) == 1
    call = lam.invoke_calls[0]
    assert call["FunctionName"] == "ohi-api-test"
    assert call["InvocationType"] == "Event"
    # Payload is a JSON-serialised synthetic APIGW v2 event so Lambda Web
    # Adapter routes it through FastAPI like any other request.
    payload = json.loads(call["Payload"])
    assert payload["version"] == "2.0"
    assert payload["rawPath"] == "/_internal/async-verify"
    assert payload["requestContext"]["http"]["method"] == "POST"
    assert payload["requestContext"]["http"]["path"] == "/_internal/async-verify"
    # body is itself JSON — Web Adapter forwards body as-is to FastAPI.
    body = json.loads(payload["body"])
    assert body == {
        "__async_verify": True,
        "depth": 1,
        "job_id": "jid-1",
        "text": "Marie Curie",
        "ticket": "tkt-a",
    }
    # isBase64Encoded must be False; otherwise Web Adapter base64-decodes
    # our already-plaintext body and corrupts it.
    assert payload["isBase64Encoded"] is False


def test_async_invoke_verify_defaults_depth_to_1(
    stub_env: None, stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient]
) -> None:
    _ddb, lam = stub_clients
    from server import jobs

    jobs.async_invoke_verify(job_id="jid-2", text="hi", ticket="tkt-b")

    payload = json.loads(lam.invoke_calls[0]["Payload"])
    assert json.loads(payload["body"])["depth"] == 1


def test_async_invoke_verify_propagates_depth_counter(
    stub_env: None, stub_clients: tuple[_StubDynamoDBClient, _StubLambdaClient]
) -> None:
    # The async handler itself will refuse depth > 1, but the helper must
    # faithfully surface whatever value it's given so the guard can fire.
    _ddb, lam = stub_clients
    from server import jobs

    jobs.async_invoke_verify(job_id="jid-3", text="hi", ticket="tkt-c", depth=2)
    body = json.loads(json.loads(lam.invoke_calls[0]["Payload"])["body"])
    assert body["depth"] == 2
