"""Debug chat route tests for brain-unavailable behavior."""

from __future__ import annotations


def _login(client):
    """Authenticate a test client and return CSRF metadata."""

    login = client.post("/api/auth/login", json={"token": "secret"})
    payload = login.json()
    return payload["csrf_header_name"], payload["csrf_token"]


def test_debug_chat_returns_brain_unavailable_without_cognition_when_stopped(
    monkeypatch,
    tmp_path,
) -> None:
    """Debug chat must not start cognition when the brain is unavailable."""

    from fastapi.testclient import TestClient

    from control_console import repository as repository_module
    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.contracts import ServiceRuntimeState
    from control_console.settings import ControlConsoleSettings

    async def application_identity(self):
        _ = self
        return {
            "status": "available",
            "character_name": "Test Character",
            "source": "character_state",
        }

    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "application_identity",
        application_identity,
    )

    class StoppedBrainSupervisor:
        def service_state(self, service_id: str):
            assert service_id == "brain"
            state = ServiceRuntimeState(
                id="brain",
                display_name="Brain service",
                kind="backend",
                actual_state="stopped",
            )
            return state

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(
        create_app(settings=settings, supervisor=StoppedBrainSupervisor()),
    )
    csrf_header_name, csrf_token = _login(client)

    response = client.post(
        "/api/debug-chat",
        headers={csrf_header_name: csrf_token},
        json={
            "channel_id": "debug",
            "user_id": "operator",
            "user_display_name": "Operator",
            "message_text": "hello",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["brain_available"] is False
    assert payload["response"] is None
    assert payload["error"]["code"] == "brain_unavailable"
    assert "message_text" not in payload["request"]

    audit_response = client.get("/api/bootstrap")
    event_types = [
        event["event_type"]
        for event in audit_response.json()["recent_audit_events"]
    ]
    assert "debug_chat_unavailable" in event_types


def test_debug_chat_uses_live_unmanaged_brain_endpoint(
    monkeypatch,
    tmp_path,
) -> None:
    """Debug chat should work when the brain endpoint is live but unmanaged."""

    from fastapi.testclient import TestClient

    from control_console import app as app_module
    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.contracts import ServiceRuntimeState
    from control_console.settings import ControlConsoleSettings

    class UnmanagedBrainSupervisor:
        def service_state(self, service_id: str):
            assert service_id == "brain"
            state = ServiceRuntimeState(
                id="brain",
                display_name="Brain service",
                kind="backend",
                actual_state="conflict",
                last_error_preview=(
                    "configured endpoint is already in use by an unmanaged process"
                ),
            )
            return state

    class FakeKazusaClient:
        def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
            _ = base_url
            _ = timeout_seconds

        async def send_debug_chat(self, request):
            return {
                "request_id": "cc-req-debug",
                "brain_available": True,
                "request": request.model_dump(mode="json"),
                "response": {"messages": [{"text": "hello operator"}]},
                "tracking_id": "tracking-1",
                "latency_ms": 12,
                "sent_at": "2026-06-17T00:00:00+00:00",
                "error": None,
            }

    monkeypatch.setattr(app_module, "KazusaClient", FakeKazusaClient)
    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(
        create_app(settings=settings, supervisor=UnmanagedBrainSupervisor()),
    )
    csrf_header_name, csrf_token = _login(client)

    response = client.post(
        "/api/debug-chat",
        headers={csrf_header_name: csrf_token},
        json={
            "channel_id": "debug",
            "user_id": "operator",
            "user_display_name": "Operator",
            "message_text": "hello",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["brain_available"] is True
    assert payload["response"]["messages"][0]["text"] == "hello operator"
