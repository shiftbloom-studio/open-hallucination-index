"""Integration test: app.py must register EdgeSecretMiddleware when env is set."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_edge_secret_middleware_registered_when_env_set(monkeypatch):
    monkeypatch.setenv(
        "OHI_CF_EDGE_SECRET_ARN",
        "arn:aws:secretsmanager:eu-central-1:1:secret:ohi/cf-edge-secret-ab",
    )

    # Patch the SecretsLoader so we don't actually call AWS
    fake_loader = MagicMock()
    fake_loader.get.return_value = "test-secret-value"

    with patch("config.secrets_loader.get_loader", return_value=fake_loader):
        from fastapi.testclient import TestClient

        # Re-import to trigger create_app() with the patched env
        import importlib
        from server import app as app_module

        importlib.reload(app_module)

        client = TestClient(app_module.app)
        r_no_header = client.get("/health/ready")
        assert r_no_header.status_code == 403
        r_correct = client.get(
            "/health/ready", headers={"X-OHI-Edge-Secret": "test-secret-value"}
        )
        # /health/ready returns 200 when providers are healthy, 503 when partial
        assert r_correct.status_code in (200, 503)

        # /health/live exempt
        r_live = client.get("/health/live")
        assert r_live.status_code in (200, 503)


def test_edge_secret_middleware_not_registered_when_env_unset(monkeypatch):
    monkeypatch.delenv("OHI_CF_EDGE_SECRET_ARN", raising=False)
    import importlib
    from server import app as app_module

    importlib.reload(app_module)

    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    r = client.get("/health/ready")
    # Should not be 403 — middleware is not active
    assert r.status_code != 403
