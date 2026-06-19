"""Authentication and CSRF tests for the control console."""

from __future__ import annotations


def test_missing_operator_hash_generates_logged_ephemeral_token(
    tmp_path,
) -> None:
    """Missing hash should create a one-process token visible in startup logs."""

    import logging

    from fastapi.testclient import TestClient

    from control_console.app import create_app
    from control_console.settings import ControlConsoleSettings

    settings = ControlConsoleSettings(state_dir=tmp_path)
    app = create_app(settings=settings)
    token_lines: list[str] = []

    class TokenLogHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if message.startswith("Control console access token: "):
                token_lines.append(message)

    logger = logging.getLogger("control_console.app")
    handler = TokenLogHandler(level=logging.WARNING)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    try:
        with TestClient(app) as client:
            token = app.state.generated_operator_token
            login = client.post("/api/auth/login", json={"token": token})
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    assert app.state.generated_operator_token == token
    assert login.status_code == 200
    assert token_lines == [f"Control console access token: {token}"]


def test_login_sets_session_and_csrf_and_rejects_bad_tokens(
    monkeypatch,
    tmp_path,
) -> None:
    """Login should issue a session cookie plus same-origin CSRF token."""

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

    class StoppedBrainSupervisor:
        def all_service_states(self):
            state = ServiceRuntimeState(
                id="brain",
                display_name="Brain service",
                kind="backend",
                actual_state="stopped",
            )
            return [state]

    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "application_identity",
        application_identity,
    )

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(
        create_app(settings=settings, supervisor=StoppedBrainSupervisor()),
    )

    locked_session = client.get("/api/auth/session")
    assert locked_session.status_code == 200
    assert locked_session.json() == {"authenticated": False}

    bad_login = client.post("/api/auth/login", json={"token": "wrong"})
    assert bad_login.status_code == 401

    good_login = client.post("/api/auth/login", json={"token": "secret"})
    assert good_login.status_code == 200
    payload = good_login.json()
    assert payload["operator"]["operator_id"] == "local_operator"
    assert payload["csrf_header_name"] == settings.csrf_header_name
    assert payload["csrf_token"]
    assert settings.session_cookie_name in good_login.cookies

    session = client.get("/api/auth/session")
    assert session.status_code == 200
    session_payload = session.json()
    assert session_payload["authenticated"] is True
    assert session_payload["csrf_header_name"] == settings.csrf_header_name
    assert session_payload["csrf_token"] == payload["csrf_token"]

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
