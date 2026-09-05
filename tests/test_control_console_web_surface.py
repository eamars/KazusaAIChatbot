"""End-to-end web surface tests for the control console."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock


class _SyncBrainRunningSupervisor:
    """Minimal supervisor fake for routes that synchronously inspect brain state."""

    def service_state(self, service_id: str):
        """Return a running brain state for debug-chat route gating."""

        from control_console.contracts import ServiceRuntimeState

        assert service_id == "brain"
        state = ServiceRuntimeState(
            id="brain",
            display_name="Brain service",
            kind="backend",
            actual_state="running",
        )
        return state


class _StaticStoppedSupervisor:
    """Supervisor fake that keeps browser-surface tests off live services."""

    def all_service_states(self):
        """Return stopped built-in service states."""


        from control_console.contracts import ServiceRuntimeState

        services = [
            ("brain", "Brain service", "backend", []),
            ("adapter.discord", "Discord adapter", "adapter", ["brain"]),
            ("adapter.napcat", "NapCat QQ adapter", "adapter", ["brain"]),
            ("adapter.debug", "Debug adapter", "adapter", ["brain"]),
        ]
        states = [
            ServiceRuntimeState(
                id=service_id,
                display_name=display_name,
                kind=kind,
                actual_state="stopped",
                dependencies=dependencies,
            )
            for service_id, display_name, kind, dependencies in services
        ]
        return states

    def service_state(self, service_id: str):
        """Return one stopped service state for debug route gating."""

        states = self.all_service_states()
        for state in states:
            if state.id == service_id:
                return state
        raise KeyError(service_id)

    def service_version(self, service_id: str) -> int:
        """Return the stable version used by stopped config tests."""

        _ = self.service_state(service_id)
        return 0


def test_model_routes_use_semantic_slugs_without_private_route_metadata() -> None:
    """Expose semantic route slugs while keeping env mapping server-private."""

    from control_console import brain_model_routes

    snapshot = SimpleNamespace(fields=[])
    rows = brain_model_routes.project_brain_model_routes(snapshot, {})
    by_key = {row["route_key"]: row for row in rows}

    assert by_key["cognition"]["label"] == "Cognition"
    assert by_key["cognition"]["required"] is True
    assert by_key["cognition_support"]["label"] == "Cognition support"
    assert all("env_prefix" not in row for row in rows)
    serialized = repr(rows).lower()
    assert "cognition_v3_chain_llm" not in serialized
    assert "shared_non_core" not in serialized
    assert "sidecar" not in serialized
    assert "engine" not in serialized
    assert all(row["route_key"] not in {"v2", "v3"} for row in rows)


def _client_with_login(tmp_path, *, supervisor=None):
    """Create a test client and return authenticated CSRF metadata."""

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
    return client, payload
















def test_live_logs_static_surface_and_controls(tmp_path) -> None:
    """The shell should expose a focused shadcn-style live-log workspace."""

    client, _ = _client_with_login(tmp_path, supervisor=_StaticStoppedSupervisor())

    index = client.get("/")
    assert index.status_code == 200
    assert 'data-page-link="logs"' in index.text
    assert 'data-page="logs"' in index.text
    assert 'id="log-service-filter"' in index.text
    assert 'id="log-stream-filter"' in index.text
    assert 'id="log-text-filter"' in index.text
    assert 'id="log-highlight-filter"' in index.text
    assert 'id="log-pause"' in index.text
    assert 'id="log-clear"' in index.text
    assert 'id="log-autoscroll"' in index.text
    assert 'id="log-wrap-lines"' in index.text
    assert 'id="log-viewport"' in index.text
    assert 'id="log-table"' in index.text
    assert "log-placeholder" in index.text
    assert 'data-component="ScrollArea"' in index.text
    assert "Live logs" in index.text
    assert "Event monitor" in index.text

    script = client.get("/static/console.js")
    assert script.status_code == 200
    assert "LOG_ROW_LIMIT" in script.text
    assert "/api/logs/stream" in script.text
    assert "openLogStream" in script.text
    assert "renderLogControls" in script.text
    assert "appendLogRow" in script.text
    assert "state.logRows" in script.text
    assert "renderBufferedLogRows" in script.text
    assert "No retained rows match this filter. Watching live logs." in script.text
    assert "No retained rows for this selection. Watching live logs." in script.text
    assert '{retained: eventName === "log.snapshot"}' in script.text
    assert "state.logPaused && !options.retained && !state.pendingLogRows" in (
        script.text
    )
    assert "data-log-service" in script.text
    assert "setPage(\"logs\")" in script.text
    assert "function refreshLogStream()" in script.text
    assert 'bind("#log-service-filter", "change", refreshLogStream)' in (
        script.text
    )
    assert 'bind("#log-stream-filter", "change", refreshLogStream)' in (
        script.text
    )
    assert (
        'bind("#log-text-filter", "input", renderBufferedLogRows)'
        in script.text
    )
    assert (
        'bind("#log-highlight-filter", "input", renderBufferedLogRows)'
        in script.text
    )
    assert ".log-row:not(.log-placeholder)" in script.text
    assert "log.gap" in script.text
    assert "log.status" in script.text
    assert "log.ready" in script.text
    assert 'class="btn log-copy"' in script.text

    stylesheet = client.get("/static/console.css")
    assert stylesheet.status_code == 200
    assert ".log-toolbar" in stylesheet.text
    assert ".log-viewport" in stylesheet.text
    assert ".log-table { table-layout: fixed;" in stylesheet.text
    assert ".log-row td:last-child" in stylesheet.text
    assert ".log-copy" in stylesheet.text
    assert "inline-size: 56px" in stylesheet.text
    assert ".log-row" in stylesheet.text
    assert ".log-row.wrap" in stylesheet.text
    assert "--log-font" in stylesheet.text
    assert '"Microsoft YaHei UI"' in stylesheet.text
    assert '"Noto Sans CJK SC"' in stylesheet.text
    assert "max-height: min(58vh, 640px)" in stylesheet.text




def test_audit_api_collapses_actions_and_summarizes_views(tmp_path) -> None:
    """Audit should present one human action instead of raw JSONL machinery."""

    from control_console.audit import LocalAuditWriter

    client, _ = _client_with_login(
        tmp_path,
        supervisor=_StaticStoppedSupervisor(),
    )
    writer = LocalAuditWriter(tmp_path / "audit.jsonl")
    writer.write_event(
        event_type="service_start_requested",
        operator_id="local-operator",
        service_id="brain",
        target={"service_id": "brain", "expected_version": 2},
        previous_state={"actual_state": "stopped"},
        reason="operator requested start",
        request_id="cc-req-start-1",
    )
    writer.write_event(
        event_type="service_started",
        operator_id="local-operator",
        service_id="brain",
        target={"service_id": "brain"},
        previous_state={"actual_state": "stopped"},
        new_state={"actual_state": "running"},
        request_id="cc-req-start-1",
    )
    writer.write_event(
        event_type="lookup_view",
        operator_id="local-operator",
        target={"namespace": "entity.user"},
        request_id="cc-req-view-1",
    )
    writer.write_event(
        event_type="lookup_view",
        operator_id="local-operator",
        target={"namespace": "entity.group"},
        request_id="cc-req-view-2",
    )
    writer.write_event(
        event_type="lookup_view",
        operator_id="local-operator",
        target={"namespace": "entity.users"},
        request_id="cc-req-view-3",
    )
    writer.write_event(
        event_type="lookup_view",
        operator_id="local-operator",
        target={"namespace": "entity.groups"},
        request_id="cc-req-view-4",
    )
    writer.write_event(
        event_type="brain_model_routes_view",
        operator_id="local-operator",
        request_id="cc-req-view-5",
    )
    writer.write_event(
        event_type="brain_model_route_models_view",
        operator_id="local-operator",
        request_id="cc-req-view-6",
    )

    response = client.get("/api/audit?limit=20")

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) >= {
        "generated_at",
        "actions",
        "view_summary",
        "facets",
        "next_cursor",
    }
    assert len(payload["actions"]) == 1
    action = payload["actions"][0]
    assert action["request_id"] == "cc-req-start-1"
    assert action["action"] == "service start"
    assert action["target_label"] == "brain"
    assert action["outcome"] == "succeeded"
    assert action["event_count"] == 2
    summaries = {
        item["view"]: item["count"]
        for item in payload["view_summary"]
    }
    assert summaries == {
        "Groups": 2,
        "Services": 2,
        "Users": 2,
    }
    assert payload["facets"]["outcomes"] == {"succeeded": 1}
    rendered = repr(payload)
    assert "[object Object]" not in rendered
    assert "lookup_view" not in repr(payload["view_summary"])
    assert "'status':" not in repr(action)
    assert "audit_view" not in repr(payload["actions"])


def test_owner_directory_and_detail_routes_use_plural_safe_contracts(
    monkeypatch,
    tmp_path,
) -> None:
    """User and group pages should discover records before opening detail."""

    from control_console import repository as repository_module

    async def list_user_entities(self, *, limit: int):
        _ = self
        assert limit == 5
        return {
            "status": "available",
            "items": [{
                "display_name": "Operator",
                "accounts": [{
                    "platform": "qq",
                    "platform_user_id": "platform-user-1",
                    "display_name": "Operator",
                }],
                "account_count": 1,
                "alias_count": 0,
                "updated_at": "2026-07-27T00:00:00Z",
            }],
        }

    async def list_group_entities(self, *, limit: int):
        _ = self
        assert limit == 5
        return {
            "status": "available",
            "items": [{
                "platform": "qq",
                "group_id": "group-1",
                "channel_name": "Review group",
                "last_activity_at": "2026-07-27T00:00:00Z",
                "message_count": 12,
                "participant_count": 3,
            }],
        }

    async def lookup_user_entity(
        self,
        *,
        platform: str,
        platform_user_id: str,
        **kwargs,
    ):
        _ = self
        _ = kwargs
        assert platform == "qq"
        assert platform_user_id == "platform-user-1"
        return {
            "status": "available",
            "owner": "user",
            "identity": {
                "platform": "qq",
                "platform_user_id": "platform-user-1",
                "display_name": "Operator",
            },
            "panels": {},
            "redaction": {"internal_global_ids": "excluded"},
        }

    async def lookup_group_entity(
        self,
        *,
        platform: str,
        group_id: str,
        **kwargs,
    ):
        _ = self
        _ = kwargs
        assert platform == "qq"
        assert group_id == "group-1"
        return {
            "status": "available",
            "owner": "group",
            "identity": {"platform": "qq", "group_id": "group-1"},
            "panels": {},
            "redaction": {"raw_messages": "excluded"},
        }

    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "list_user_entities",
        list_user_entities,
        raising=False,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "list_group_entities",
        list_group_entities,
        raising=False,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "lookup_user_entity",
        lookup_user_entity,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "lookup_group_entity",
        lookup_group_entity,
    )

    client, _ = _client_with_login(
        tmp_path,
        supervisor=_StaticStoppedSupervisor(),
    )

    users = client.get("/api/entities/users?limit=5")
    user = client.get("/api/entities/users/qq/platform-user-1?limit=5")
    groups = client.get("/api/entities/groups?limit=5")
    group = client.get("/api/entities/groups/qq/group-1?limit=5")

    assert users.status_code == 200
    assert users.json()["items"][0]["display_name"] == "Operator"
    assert user.status_code == 200
    assert user.json()["identity"]["platform_user_id"] == "platform-user-1"
    assert groups.status_code == 200
    assert groups.json()["items"][0]["group_id"] == "group-1"
    assert group.status_code == 200
    assert group.json()["identity"]["group_id"] == "group-1"
    assert client.get("/api/entities/user?limit=5").status_code == 404
    assert client.get("/api/entities/group?limit=5").status_code == 404


def test_lifecycle_stop_and_restart_responses(tmp_path) -> None:
    """Stop and restart routes should expose stable web outputs."""

    supervisor = AsyncMock()
    supervisor.service_version.return_value = 0
    supervisor.start_service.return_value = {
        "request_id": "request-start",
        "action": "start",
        "audit_event_id": "audit-start",
        "service": {"id": "brain", "version": 1, "actual_state": "running"},
    }
    supervisor.stop_service.return_value = {
        "request_id": "request-stop",
        "action": "stop",
        "audit_event_id": "audit-stop",
        "service": {"id": "brain", "version": 1, "actual_state": "stopped"},
    }
    supervisor.restart_service.return_value = {
        "request_id": "request-restart",
        "action": "restart",
        "audit_event_id": "audit-restart",
        "service": {"id": "brain", "version": 2, "actual_state": "running"},
    }
    client, payload = _client_with_login(tmp_path, supervisor=supervisor)
    headers = {payload["csrf_header_name"]: payload["csrf_token"]}

    start = client.post(
        "/api/services/brain/start",
        headers=headers,
        json={"reason": "operator start", "expected_version": 0},
    )
    assert start.status_code == 200
    assert start.json()["action"] == "start"

    stop = client.post(
        "/api/services/brain/stop",
        headers=headers,
        json={"reason": "operator stop", "expected_version": 0},
    )
    assert stop.status_code == 200
    assert stop.json()["action"] == "stop"

    restart = client.post(
        "/api/services/brain/restart",
        headers=headers,
        json={"reason": "operator restart", "expected_version": 0},
    )
    assert restart.status_code == 200
    assert restart.json()["action"] == "restart"


def test_background_lookup_reports_empty_and_unavailable(
    monkeypatch,
    tmp_path,
) -> None:
    """Background telemetry route should distinguish empty from unavailable."""

    from control_console import app as app_module
    from control_console import repository as repository_module

    async def empty_jobs(*, limit: int):
        assert limit == 5
        return []

    async def empty_events(query):
        assert query.service_id == "background_work.worker"
        return []

    async def unavailable_events(query):
        assert query.service_id == "background_work.worker"
        return [{
            "source": "kazusa",
            "event_type": "event_log.unavailable",
            "status": "unavailable",
            "created_at": "2026-06-17T00:00:00+00:00",
        }]

    monkeypatch.setattr(app_module, "_read_kazusa_events", empty_events)
    monkeypatch.setattr(
        repository_module.background_work_job_store,
        "find_deliverable_background_work_jobs",
        empty_jobs,
    )
    monkeypatch.setattr(
        repository_module.background_work_job_store,
        "list_recent_background_work_jobs",
        empty_jobs,
    )
    client, _ = _client_with_login(tmp_path)

    empty = client.get("/api/lookups/background-work?limit=5")
    assert empty.status_code == 200
    empty_payload = empty.json()
    assert empty_payload["panels"]["worker_activity"]["status"] == "empty"

    monkeypatch.setattr(app_module, "_read_kazusa_events", unavailable_events)
    unavailable = client.get("/api/lookups/background-work?limit=5")
    assert unavailable.status_code == 200
    payload = unavailable.json()
    assert payload["panels"]["worker_activity"]["status"] == "unavailable"
    assert payload["panels"]["worker_activity"]["items"] == []
    assert "telemetry is unavailable" in (
        payload["panels"]["worker_activity"]["reason"]
    )






def test_main_invokes_uvicorn_with_cli_arguments(monkeypatch) -> None:
    """The console CLI should pass host, port, reload, and factory to uvicorn."""

    import sys

    from control_console import main as console_main

    calls: list[dict] = []

    def fake_run(app_ref, **kwargs):
        calls.append({"app_ref": app_ref, **kwargs})

    monkeypatch.setattr(console_main.uvicorn, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kazusa-control-console",
            "--host",
            "127.0.0.2",
            "--port",
            "8766",
            "--reload",
        ],
    )

    console_main.main()

    assert calls == [
        {
            "app_ref": "control_console.app:create_app",
            "host": "127.0.0.2",
            "port": 8766,
            "reload": True,
            "factory": True,
            "timeout_graceful_shutdown": 3,
        },
    ]


def test_character_identity_surface_and_pace_api_contract(
    tmp_path,
) -> None:
    """The authenticated browser surface should expose bounded identity growth."""

    client, login = _client_with_login(
        tmp_path,
        supervisor=_StaticStoppedSupervisor(),
    )

    index = client.get("/")
    script = client.get("/static/console.js")

    assert index.status_code == 200
    assert script.status_code == 200
    assert "Identity lineage and health" in index.text
    assert "Growth candidates and outcomes" in index.text
    assert "renderIdentityLineagePanel" in script.text
    assert "renderIdentityGrowthPanel" in script.text
    assert "renderCharacterLoadingState" in script.text
    assert "renderCharacterErrorState" in script.text
    assert "Loading character identity" in script.text
    assert "Character identity could not be loaded" in script.text
    for health_label in (
        "healthy idle",
        "waiting for evidence",
        "semantic rejection",
        "promotion ready",
        "awaiting consumption",
        "healthy active",
        "pipeline error",
        "consumption error",
    ):
        assert health_label in script.text
    assert "character_global" not in index.text
    assert "character_global" not in script.text
    assert "No character-global carry-over" not in script.text
    assert "validation.min_value" in script.text
    assert "validation.max_value" in script.text

    config = client.get("/api/services/brain/config")

    assert config.status_code == 200
    fields = {
        field["key"]: field
        for field in config.json()["fields"]
    }
    assert (
        fields["character_identity_growth_inferred_min_episodes"][
            "validation"
        ]
        == {"min_value": 2, "max_value": 8}
    )
    assert fields["character_identity_growth_enabled"][
        "restart_required"
    ] is True

    headers = {login["csrf_header_name"]: login["csrf_token"]}
    invalid = client.put(
        "/api/services/brain/config",
        headers=headers,
        json={
            "reason": "test cross-field validation",
            "values": {
                "character_identity_growth_inferred_min_episodes": 2,
                "character_identity_growth_inferred_min_local_dates": 3,
            },
        },
    )

    assert invalid.status_code == 422
    assert "cannot exceed" in invalid.json()["detail"]["message"]
