"""Service-configuration route tests for the control console."""

from __future__ import annotations


class _ConfigRouteSupervisor:
    """Small supervisor fake for config-route restart orchestration."""

    def __init__(
        self,
        *,
        napcat_state: str,
        restart_fails: bool = False,
    ) -> None:
        """Create service states with a configurable NapCat runtime state."""

        from control_console.contracts import ServiceRuntimeState

        self.restart_fails = restart_fails
        self.restart_calls: list[dict[str, str]] = []
        self._states = {
            "brain": ServiceRuntimeState(
                id="brain",
                display_name="Brain service",
                kind="backend",
                actual_state="running",
                desired_state="running",
                version=3,
            ),
            "adapter.napcat": ServiceRuntimeState(
                id="adapter.napcat",
                display_name="NapCat QQ adapter",
                kind="adapter",
                actual_state=napcat_state,
                desired_state="running" if napcat_state == "running" else "stopped",
                dependencies=["brain"],
                version=7,
            ),
        }

    def all_service_states(self):
        """Return the current fake service states."""

        states = list(self._states.values())
        return states

    def service_state(self, service_id: str):
        """Return one fake service state."""

        state = self._states[service_id]
        return state

    def service_version(self, service_id: str) -> int:
        """Return one fake service version."""

        version = self._states[service_id].version
        return version

    async def restart_service(
        self,
        *,
        service_id: str,
        operator_id: str,
        reason: str,
    ) -> dict[str, object]:
        """Record a restart and return the updated runtime state."""

        self.restart_calls.append({
            "service_id": service_id,
            "operator_id": operator_id,
            "reason": reason,
        })
        if self.restart_fails:
            from control_console.supervisor import ServiceLifecycleError

            raise ServiceLifecycleError("restart failed for test")

        current_state = self._states[service_id]
        next_state = current_state.model_copy(
            update={
                "actual_state": "running",
                "desired_state": "running",
                "version": current_state.version + 1,
            },
        )
        self._states[service_id] = next_state
        response = {
            "request_id": "cc-req-restart",
            "action": "restart",
            "audit_event_id": "cc-audit-restart",
            "service": next_state.model_dump(mode="json"),
        }
        return response


def _client_with_login(tmp_path, supervisor: _ConfigRouteSupervisor):
    """Create an authenticated client and CSRF metadata for config tests."""

    from fastapi.testclient import TestClient

    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(create_app(settings=settings, supervisor=supervisor))
    login = client.post("/api/auth/login", json={"token": "secret"})
    assert login.status_code == 200
    payload = login.json()
    return client, payload, settings


def _route_environment(monkeypatch) -> None:
    """Install complete route env defaults without reading local dotenv files."""

    routes = [
        "RELEVANCE_AGENT_LLM",
        "VISION_DESCRIPTOR_LLM",
        "MSG_DECONTEXTUALIZER_LLM",
        "RAG_PLANNER_LLM",
        "RAG_SUBAGENT_LLM",
        "WEB_SEARCH_LLM",
        "COGNITION_LLM",
        "BOUNDARY_CORE_LLM",
        "DIALOG_GENERATOR_LLM",
        "CONSOLIDATION_LLM",
        "JSON_REPAIR_LLM",
    ]
    monkeypatch.setenv("DEFAULT_LLM_MAX_COMPLETION_TOKENS", "8192")
    for route in routes:
        monkeypatch.setenv(f"{route}_BASE_URL", "http://localhost:1234/v1")
        monkeypatch.setenv(f"{route}_API_KEY", "test-key")
        monkeypatch.setenv(f"{route}_MODEL", f"{route.lower()}-qwen3")


def test_config_routes_require_auth_and_csrf(monkeypatch, tmp_path) -> None:
    """Config routes must use the same auth and CSRF boundary as lifecycle."""

    from fastapi.testclient import TestClient

    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    monkeypatch.setenv("NAPCAT_ACTIVE_GROUPS", "54369546")
    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(
        create_app(
            settings=settings,
            supervisor=_ConfigRouteSupervisor(napcat_state="stopped"),
        ),
    )
    service_id = "adapter.napcat"
    config_route = f"/api/services/{service_id}/config"

    locked_get = client.get(config_route)
    assert locked_get.status_code == 401

    login = client.post("/api/auth/login", json={"token": "secret"})
    assert login.status_code == 200
    csrf_token = login.json()["csrf_token"]
    csrf_header_name = login.json()["csrf_header_name"]

    missing_csrf = client.put(
        config_route,
        json={"reason": "test", "values": {"active_groups": ["54369546"]}},
    )
    assert missing_csrf.status_code == 403

    authenticated_get = client.get(config_route)
    assert authenticated_get.status_code == 200
    config = authenticated_get.json()
    assert config["service_id"] == "adapter.napcat"
    assert config["state"] == "default"
    assert config["fields"][0]["key"] == "active_groups"
    assert config["fields"][0]["default_source"] == "NAPCAT_ACTIVE_GROUPS"

    invalid_value = client.put(
        config_route,
        headers={csrf_header_name: csrf_token},
        json={"reason": "test", "values": {"active_groups": ["not-a-group"]}},
    )
    assert invalid_value.status_code == 422


def test_brain_model_route_api_applies_and_resets_selected_route(
    monkeypatch,
    tmp_path,
) -> None:
    """Route-specific API should reuse auth, validation, restart, and audit."""

    _route_environment(monkeypatch)
    supervisor = _ConfigRouteSupervisor(napcat_state="stopped")
    client, auth, settings = _client_with_login(tmp_path, supervisor)
    route_url = "/api/services/brain/model-routes/cognition_llm"

    snapshot_response = client.get("/api/services/brain/model-routes")
    assert snapshot_response.status_code == 200
    snapshot = snapshot_response.json()
    assert snapshot["service_id"] == "brain"
    assert snapshot["service_state"]["actual_state"] == "running"
    assert len(snapshot["routes"]) == 13
    assert "test-key" not in snapshot_response.text

    missing_csrf = client.put(
        route_url,
        json={
            "reason": "change cognition route",
            "values": {"model": "deepseek-v4-flash"},
        },
    )
    assert missing_csrf.status_code == 403

    apply_response = client.put(
        route_url,
        headers={auth["csrf_header_name"]: auth["csrf_token"]},
        json={
            "reason": "change cognition route",
            "expected_version": 3,
            "values": {
                "model": "deepseek-v4-flash",
                "max_completion_tokens": 4096,
                "thinking_enabled": True,
            },
        },
    )
    assert apply_response.status_code == 200
    payload = apply_response.json()
    assert payload["restart"]["attempted"] is True
    assert payload["restart"]["succeeded"] is True
    assert payload["route"]["effective"]["model"] == "deepseek-v4-flash"
    assert payload["route"]["effective"]["source"] == "override"
    assert supervisor.restart_calls[-1]["service_id"] == "brain"

    reset_response = client.post(
        f"{route_url}/reset",
        headers={auth["csrf_header_name"]: auth["csrf_token"]},
        json={"reason": "reset cognition route", "expected_version": 4},
    )
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["route"]["effective"]["model"] == "cognition_llm-qwen3"

    from control_console.audit import LocalAuditWriter

    event_types = [
        event.event_type
        for event in LocalAuditWriter(settings.audit_path).read_recent(limit=30)
    ]
    assert "brain_model_route_apply_requested" in event_types
    assert "brain_model_route_reset_requested" in event_types


def test_available_models_route_returns_redacted_provider_status(
    monkeypatch,
    tmp_path,
) -> None:
    """The browser should receive model ids without provider credentials."""

    import httpx

    _route_environment(monkeypatch)

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"data": [{"id": "qwen3-32b"}]})

    from control_console import brain_model_routes

    monkeypatch.setattr(
        brain_model_routes,
        "_MODEL_LIST_TRANSPORT",
        httpx.MockTransport(handler),
    )
    supervisor = _ConfigRouteSupervisor(napcat_state="stopped")
    client, _, _ = _client_with_login(tmp_path, supervisor)

    response = client.get(
        "/api/services/brain/model-routes/cognition_llm/available-models",
    )

    assert response.status_code == 200
    assert response.json() == {
        "route_key": "cognition_llm",
        "status": "available",
        "models": [{"id": "qwen3-32b", "family": "qwen"}],
        "message": None,
    }
    assert "test-key" not in response.text


def test_apply_config_to_stopped_service_stores_override_without_restart(
    monkeypatch,
    tmp_path,
) -> None:
    """Stopped services should use the override on next start without restart."""

    monkeypatch.setenv("NAPCAT_ACTIVE_GROUPS", "54369546")
    supervisor = _ConfigRouteSupervisor(napcat_state="stopped")
    client, auth, settings = _client_with_login(tmp_path, supervisor)
    service_id = "adapter.napcat"
    config_route = f"/api/services/{service_id}/config"

    response = client.put(
        config_route,
        headers={auth["csrf_header_name"]: auth["csrf_token"]},
        json={
            "reason": "change active groups",
            "expected_version": 7,
            "values": {"active_groups": ["905393941"]},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["service_id"] == "adapter.napcat"
    assert payload["restart"]["attempted"] is False
    assert payload["restart"]["succeeded"] is None
    assert payload["config"]["state"] == "override_active"
    assert payload["config"]["fields"][0]["override_value"] == ["905393941"]
    assert payload["service"]["actual_state"] == "stopped"
    assert supervisor.restart_calls == []

    from control_console.audit import LocalAuditWriter

    event_types = [
        event.event_type
        for event in LocalAuditWriter(settings.audit_path).read_recent(limit=20)
    ]
    assert "service_config_apply_requested" in event_types
    assert "service_config_applied" in event_types


def test_apply_config_to_running_service_restarts_target_service(
    monkeypatch,
    tmp_path,
) -> None:
    """Running services should restart after a valid config override is stored."""

    monkeypatch.setenv("NAPCAT_ACTIVE_GROUPS", "54369546")
    supervisor = _ConfigRouteSupervisor(napcat_state="running")
    client, auth, settings = _client_with_login(tmp_path, supervisor)
    service_id = "adapter.napcat"
    config_route = f"/api/services/{service_id}/config"

    response = client.put(
        config_route,
        headers={auth["csrf_header_name"]: auth["csrf_token"]},
        json={
            "reason": "activate test group",
            "expected_version": 7,
            "values": {"active_groups": ["54369546", "905393941"]},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["restart"]["attempted"] is True
    assert payload["restart"]["succeeded"] is True
    assert payload["service"]["actual_state"] == "running"
    assert payload["config"]["fields"][0]["effective_value"] == [
        "54369546",
        "905393941",
    ]
    assert supervisor.restart_calls == [
        {
            "service_id": "adapter.napcat",
            "operator_id": "local_operator",
            "reason": "config apply requires restart",
        },
    ]

    from control_console.audit import LocalAuditWriter

    event_types = [
        event.event_type
        for event in LocalAuditWriter(settings.audit_path).read_recent(limit=20)
    ]
    assert "service_config_restart_requested" in event_types
    assert "service_config_applied" in event_types


def test_apply_config_restart_failure_returns_apply_failed_state(
    monkeypatch,
    tmp_path,
) -> None:
    """Failed restart feedback should not look like a successful apply."""

    monkeypatch.setenv("NAPCAT_ACTIVE_GROUPS", "54369546")
    supervisor = _ConfigRouteSupervisor(
        napcat_state="running",
        restart_fails=True,
    )
    client, auth, settings = _client_with_login(tmp_path, supervisor)
    service_id = "adapter.napcat"
    config_route = f"/api/services/{service_id}/config"

    response = client.put(
        config_route,
        headers={auth["csrf_header_name"]: auth["csrf_token"]},
        json={
            "reason": "activate test group",
            "expected_version": 7,
            "values": {"active_groups": ["905393941"]},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["restart"]["attempted"] is True
    assert payload["restart"]["succeeded"] is False
    assert "restart failed for test" in payload["restart"]["reason"]
    assert payload["config"]["state"] == "apply_failed"
    assert payload["config"]["fields"][0]["override_value"] == ["905393941"]

    from control_console.audit import LocalAuditWriter

    event_types = [
        event.event_type
        for event in LocalAuditWriter(settings.audit_path).read_recent(limit=20)
    ]
    assert "service_config_restart_requested" in event_types
    assert "service_config_apply_failed" in event_types


def test_reset_config_clears_override_and_restarts_running_service(
    monkeypatch,
    tmp_path,
) -> None:
    """Reset should clear override state and restart a running target service."""

    monkeypatch.setenv("NAPCAT_ACTIVE_GROUPS", "54369546")
    supervisor = _ConfigRouteSupervisor(napcat_state="running")
    client, auth, settings = _client_with_login(tmp_path, supervisor)
    service_id = "adapter.napcat"
    config_route = f"/api/services/{service_id}/config"

    apply_response = client.put(
        config_route,
        headers={auth["csrf_header_name"]: auth["csrf_token"]},
        json={
            "reason": "activate test group",
            "values": {"active_groups": ["905393941"]},
        },
    )
    assert apply_response.status_code == 200

    reset_response = client.post(
        f"{config_route}/reset",
        headers={auth["csrf_header_name"]: auth["csrf_token"]},
        json={"reason": "return to default", "expected_version": 8},
    )

    assert reset_response.status_code == 200
    payload = reset_response.json()
    assert payload["restart"]["attempted"] is True
    assert payload["restart"]["succeeded"] is True
    assert payload["config"]["state"] == "default"
    assert payload["config"]["fields"][0]["override_value"] is None
    assert payload["config"]["fields"][0]["effective_value"] == ["54369546"]
    assert len(supervisor.restart_calls) == 2

    from control_console.audit import LocalAuditWriter

    event_types = [
        event.event_type
        for event in LocalAuditWriter(settings.audit_path).read_recent(limit=20)
    ]
    assert "service_config_reset_requested" in event_types
    assert "service_config_applied" in event_types


def test_reset_config_restart_failure_returns_apply_failed_state(
    monkeypatch,
    tmp_path,
) -> None:
    """Failed reset restart should visibly remain in failed apply state."""

    monkeypatch.setenv("NAPCAT_ACTIVE_GROUPS", "54369546")
    supervisor = _ConfigRouteSupervisor(
        napcat_state="running",
        restart_fails=True,
    )
    client, auth, settings = _client_with_login(tmp_path, supervisor)
    service_id = "adapter.napcat"
    config_route = f"/api/services/{service_id}/config"

    reset_response = client.post(
        f"{config_route}/reset",
        headers={auth["csrf_header_name"]: auth["csrf_token"]},
        json={"reason": "return to default", "expected_version": 7},
    )

    assert reset_response.status_code == 200
    payload = reset_response.json()
    assert payload["restart"]["attempted"] is True
    assert payload["restart"]["succeeded"] is False
    assert payload["config"]["state"] == "apply_failed"
    assert payload["config"]["fields"][0]["override_value"] is None
    assert payload["config"]["fields"][0]["effective_value"] == ["54369546"]

    from control_console.audit import LocalAuditWriter

    event_types = [
        event.event_type
        for event in LocalAuditWriter(settings.audit_path).read_recent(limit=20)
    ]
    assert "service_config_restart_requested" in event_types
    assert "service_config_reset_failed" in event_types
