"""Lifecycle route contract tests for the control console API."""

from __future__ import annotations

from unittest.mock import AsyncMock


def test_lifecycle_routes_require_auth_csrf_reason_and_version(tmp_path) -> None:
    """State-changing routes should reject unauthenticated or stale requests."""

    from fastapi.testclient import TestClient

    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    supervisor = AsyncMock()
    supervisor.start_service.return_value = {
        "request_id": "request-1",
        "action": "start",
        "audit_event_id": "audit-1",
        "service": {"id": "brain", "version": 2, "actual_state": "running"},
    }
    supervisor.service_version.return_value = 2
    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    app = create_app(settings=settings, supervisor=supervisor)
    client = TestClient(app)

    unauthenticated = client.post(
        "/api/services/brain/start",
        json={"reason": "test", "expected_version": 1},
    )
    assert unauthenticated.status_code == 401

    login = client.post("/api/auth/login", json={"token": "secret"})
    assert login.status_code == 200
    csrf_token = login.json()["csrf_token"]
    csrf_header_name = login.json()["csrf_header_name"]

    missing_csrf = client.post(
        "/api/services/brain/start",
        json={"reason": "test", "expected_version": 1},
    )
    assert missing_csrf.status_code == 403

    missing_reason = client.post(
        "/api/services/brain/start",
        headers={csrf_header_name: csrf_token},
        json={"expected_version": 1},
    )
    assert missing_reason.status_code == 422

    version_mismatch = client.post(
        "/api/services/brain/start",
        headers={csrf_header_name: csrf_token},
        json={"reason": "test", "expected_version": 1},
    )
    assert version_mismatch.status_code == 409


def test_lifecycle_routes_reject_unknown_service_without_adoption(tmp_path) -> None:
    """Unknown services should not be adopted or controlled."""

    from fastapi.testclient import TestClient

    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(create_app(settings=settings))
    login = client.post("/api/auth/login", json={"token": "secret"})
    payload = login.json()

    response = client.post(
        "/api/services/external.process/start",
        headers={payload["csrf_header_name"]: payload["csrf_token"]},
        json={"reason": "test"},
    )

    assert response.status_code == 404


def test_lifecycle_routes_return_conflict_for_dependency_failures(tmp_path) -> None:
    """Lifecycle errors should reach the browser as controlled API responses."""

    from fastapi.testclient import TestClient

    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings
    from control_console.supervisor import ServiceLifecycleError

    supervisor = AsyncMock()
    supervisor.service_version.return_value = 0
    supervisor.start_service.side_effect = ServiceLifecycleError(
        "dependency brain is not running for adapter.debug",
    )
    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(
        create_app(settings=settings, supervisor=supervisor),
        raise_server_exceptions=False,
    )
    login = client.post("/api/auth/login", json={"token": "secret"})
    payload = login.json()

    response = client.post(
        "/api/services/adapter.debug/start",
        headers={payload["csrf_header_name"]: payload["csrf_token"]},
        json={"reason": "start debug"},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["message"] == (
        "dependency brain is not running for adapter.debug"
    )
