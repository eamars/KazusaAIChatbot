"""End-to-end web surface tests for the control console."""

from __future__ import annotations

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


def test_static_shell_favicon_and_generic_lookup_outputs(
    monkeypatch,
    tmp_path,
) -> None:
    """Static routes and generic lookups should return browser-safe outputs."""

    from control_console import repository as repository_module

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

    client, _ = _client_with_login(tmp_path, supervisor=_StaticStoppedSupervisor())

    index = client.get("/")
    assert index.status_code == 200
    assert "<title>not connected</title>" in index.text
    assert '<body data-theme="bright" data-auth-state="locked">' in index.text
    assert 'id="brand-name">not connected</strong>' in index.text
    assert 'id="brand-subtitle">database not connected</span>' in index.text
    assert (
        '<nav class="nav-group" aria-label="Console pages" '
        'data-auth-required="true">'
    ) in index.text
    assert (
        '<button class="nav-link" data-page-link="services" disabled '
        'aria-disabled="true">'
    ) in index.text
    assert 'data-page-link="memory"' in index.text
    assert 'data-page-link="style"' in index.text
    assert 'data-page-link="calendar"' in index.text
    assert 'data-page-link="background"' in index.text
    assert 'data-page="memory"' in index.text
    assert 'data-page="style"' in index.text
    assert 'data-page="calendar"' in index.text
    assert 'data-page="background"' in index.text
    assert 'id="memory-global-user-id"' in index.text
    assert 'id="memory-query"' in index.text
    assert 'id="refresh-memory"' in index.text
    assert 'id="memory-table"' in index.text
    assert 'id="style-global-user-id"' in index.text
    assert 'id="style-platform"' in index.text
    assert 'id="style-channel-id"' in index.text
    assert 'id="refresh-style"' in index.text
    assert 'id="style-table"' in index.text
    assert 'id="calendar-status"' in index.text
    assert 'id="refresh-calendar"' in index.text
    assert 'id="calendar-table"' in index.text
    assert 'id="background-status"' in index.text
    assert 'id="refresh-background"' in index.text
    assert 'id="background-table"' in index.text
    assert '<input class="input" id="token"' in index.text
    assert 'data-theme-choice="bright"' in index.text
    assert 'data-theme-choice="dark"' in index.text
    assert '<span class="status-dot" data-state="locked"' in index.text
    assert 'id="debug-brain-status"' in index.text
    assert 'id="debug-send"' in index.text
    assert 'id="debug-cognition-graph"' in index.text
    assert 'data-cognition-graph="debug_latest"' in index.text
    assert 'data-debug-input' in index.text
    assert 'data-component="FieldSet"' in index.text
    assert 'name="debug_mode" value="visible_reply" checked' in index.text
    assert 'name="debug_mode" value="think_only"' in index.text
    assert 'name="debug_mode" value="listen_only"' in index.text
    assert 'name="debug_modes" value="no_remember" checked' in index.text
    assert '<option value="kazusa">application event log</option>' in index.text
    assert 'id="overview-capability-table"' in index.text
    assert 'id="overview-unavailable-table"' in index.text
    assert 'id="overview-cognition-graph"' in index.text
    assert 'data-cognition-graph="overview_latest"' in index.text
    assert 'id="event-request-id"' in index.text
    assert 'id="event-tracking-id"' in index.text
    assert "Pending work" not in index.text
    assert "Inspection Surfaces" not in index.text
    assert "11 areas" not in index.text
    assert "Bounded shared and user memory lookup" not in index.text
    assert "User and group style-image lookup" not in index.text
    assert "Schedule and due-run inspection" not in index.text
    assert "Background job inspection" not in index.text
    assert "Kazusa operational events through bounded filters" not in index.text
    assert "Kazusa source not wired" not in index.text
    assert "bootstrap skeleton" not in index.text
    assert "Ready for bounded refresh" not in index.text
    assert "scheduler-owned" not in index.text
    assert "worker lifecycle" not in index.text
    assert "source refs" not in index.text
    assert "Mongo secondary" not in index.text
    assert "until brain starts" not in index.text
    assert 'id="health-brain-status"' in index.text
    assert 'id="health-cache-table"' in index.text
    assert 'id="health-runtime-table"' in index.text
    assert 'id="character-state-table"' in index.text
    assert 'id="character-growth-table"' in index.text
    assert 'id="login-form"' in index.text
    assert 'id="ui-notice"' in index.text
    assert 'aria-live="polite"' in index.text
    assert 'id="service-config-dialog"' in index.text
    assert 'role="dialog"' in index.text
    assert 'aria-modal="true"' in index.text
    assert 'data-component="Dialog"' in index.text
    assert 'id="service-config-title"' in index.text
    assert 'id="service-config-description"' in index.text
    assert 'id="service-config-fields"' in index.text
    assert 'id="service-config-restart-note"' in index.text
    assert 'id="service-config-apply"' in index.text
    assert 'id="service-config-reset"' in index.text
    assert 'id="service-config-close"' in index.text
    assert 'Apply and restart' in index.text
    assert 'class="dialog-panel service-config-panel card"' in index.text
    assert "hot apply" not in index.text.lower()
    assert "zero downtime" not in index.text.lower()
    assert "adapter.napcat" not in index.text
    assert "active_groups" not in index.text
    assert "NAPCAT_ACTIVE_GROUPS" not in index.text

    script = client.get("/static/console.js")
    assert script.status_code == 200
    assert "resumeSession()" in script.text
    assert "/api/auth/session" in script.text
    assert "lockSession()" in script.text
    assert "renderBrand(payload.application_identity" in script.text
    assert "state.csrfToken = payload.csrf_token" in script.text
    assert "renderCapabilitySummary" in script.text
    assert "renderShellStatus(payload)" in script.text
    assert "Brain endpoint already running outside the console" in (
        script.text
    )
    assert "Brain has a stale lifecycle conflict" in script.text
    assert '["conflict", "crashed", "unhealthy"].includes(status)' in script.text
    assert "function isEndpointConflict" in script.text
    assert "function isServiceHttpAvailable" in script.text
    assert "ENDPOINT_CONFLICT_MESSAGE" in script.text
    assert "function dependenciesAvailable" in script.text
    assert 'dependency.actual_state === "running" || isEndpointConflict(dependency)' in (
        script.text
    )
    assert "renderDebugAvailability()" in script.text
    assert "isServiceHttpAvailable(brainService)" in script.text
    assert 'form.get("debug_mode")' in script.text
    assert 'form.getAll("debug_modes")' in script.text
    assert "payload.debug_modes = debugModes" in script.text
    assert "delete payload.debug_mode" in script.text
    assert "function showNotice" in script.text
    assert "function runButtonAction" in script.text
    assert "alert(" not in script.text
    assert "Loading memory..." in script.text
    assert "refreshMemory" in script.text
    assert "/api/lookups/memory" in script.text
    assert "Loading interaction style..." in script.text
    assert "refreshStyle" in script.text
    assert "/api/lookups/style" in script.text
    assert "Loading calendar..." in script.text
    assert "refreshCalendar" in script.text
    assert "/api/lookups/calendar" in script.text
    assert "Loading background work..." in script.text
    assert "refreshBackground" in script.text
    assert "/api/lookups/background" in script.text
    assert "/api/character/status" in script.text
    assert "/api/character/growth" in script.text
    assert "#health-brain-status" in script.text
    assert "debugResponseBody(result)" in script.text
    assert "function renderCognitionGraph" in script.text
    assert "function renderOverviewCognitionGraph" in script.text
    assert "function renderDebugCognitionGraph" in script.text
    assert "function cognitionGraphModel" in script.text
    assert "function cognitionGraphCurrentNode" in script.text
    assert "function cognitionGraphInspectorMarkup" in script.text
    assert "function setCognitionGraphPinnedNode" in script.text
    assert "GRAPH_STALE_AFTER_MS = 10000" in script.text
    assert "Return to current" in script.text
    assert "data-graph-current-node-id" in script.text
    assert "data-graph-selected-node-id" in script.text
    assert "graph-inspector" in script.text
    assert "graph-run-summary" in script.text
    assert "function cognitionGraphStageGroups" in script.text
    assert "function cognitionGraphStageGroupMarkup" in script.text
    assert "function cognitionGraphConnectorMarkup" in script.text
    assert "graph-stage-rail" in script.text
    assert "graph-stage-group" in script.text
    assert "graph-connector" in script.text
    assert "graph-lane-row" not in script.text
    assert "graph-edge-layer" not in script.text
    assert "drawCognitionGraphEdges" not in script.text
    assert "is-current" in script.text
    assert "control.cognition_graph_invalidated" in script.text
    assert "function openServiceConfig" in script.text
    assert "function renderServiceConfigDialog" in script.text
    assert "function applyServiceConfig" in script.text
    assert "function resetServiceConfig" in script.text
    assert "data-config-service" in script.text
    assert "/config/reset" in script.text
    assert "Apply and restart" in script.text
    assert "service-config-dialog" in script.text
    assert "hot apply" not in script.text.lower()
    assert "zero downtime" not in script.text.lower()
    assert "adapter.napcat" not in script.text
    assert "active_groups" not in script.text
    assert "NAPCAT_ACTIVE_GROUPS" not in script.text
    assert "JSON.stringify(result.response)" not in script.text

    stylesheet = client.get("/static/console.css")
    assert stylesheet.status_code == 200
    assert '.status-dot[data-state="conflict"]' in stylesheet.text
    assert ".cognition-graph" in stylesheet.text
    assert ".cognition-graph-shell" in stylesheet.text
    assert ".graph-run-summary" in stylesheet.text
    assert ".graph-body" in stylesheet.text
    assert ".graph-inspector" in stylesheet.text
    assert ".graph-stage-rail" in stylesheet.text
    assert ".graph-stage-group" in stylesheet.text
    assert ".graph-connector" in stylesheet.text
    assert ".graph-branch-stack" in stylesheet.text
    assert ".graph-lane-row" not in stylesheet.text
    assert "min-width: max-content" not in stylesheet.text
    assert ".graph-edge-layer" not in stylesheet.text
    assert ".graph-node.is-current" in stylesheet.text
    assert ".graph-node.is-selected" in stylesheet.text
    assert ".graph-node .node-detail" not in stylesheet.text
    assert (
        "grid-template-columns: minmax(min(560px, 100%), 1fr) minmax(min(280px, 100%), 320px)"
        in stylesheet.text
    )
    assert ".notice[data-tone=\"danger\"]" in stylesheet.text
    assert 'body[data-auth-state="authenticated"] #login-form' in stylesheet.text
    assert "--nav-active-bg:" in stylesheet.text
    assert "--nav-active-border:" in stylesheet.text
    assert "--nav-active-indicator:" in stylesheet.text
    assert ".nav-link.active::before" in stylesheet.text
    assert "border-color: var(--nav-active-border)" in stylesheet.text
    assert "box-shadow: inset 0 0 0 1px var(--nav-active-border)" in (
        stylesheet.text
    )
    assert ".dialog-panel" in stylesheet.text
    assert "background: var(--panel)" in stylesheet.text
    assert "color: var(--ink)" in stylesheet.text
    assert "overflow: hidden" in stylesheet.text
    assert "isolation: isolate" in stylesheet.text
    assert "--shell-gutter: clamp(" in stylesheet.text
    assert "--sidebar-width: clamp(" in stylesheet.text
    assert (
        "grid-template-columns: var(--sidebar-width) minmax(0, 1fr)"
        in stylesheet.text
    )
    assert "repeat(auto-fit, minmax(min(280px, 100%), 1fr))" in (
        stylesheet.text
    )
    assert "repeat(auto-fit, minmax(min(320px, 100%), 1fr))" in (
        stylesheet.text
    )
    assert ".table-wrap { overflow: auto;" in stylesheet.text
    assert ".nav-group { display: flex;" in stylesheet.text
    assert "@media (max-width: 640px)" in stylesheet.text

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    capabilities = bootstrap.json()["page_capabilities"]
    assert capabilities["debug"]["label"] == "brain gated"
    assert capabilities["health"]["label"] == "runtime gated"

    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 200
    assert favicon.headers["content-type"] == "image/png"

    lookup = client.get("/api/lookups/style?limit=7")
    assert lookup.status_code == 200
    payload = lookup.json()
    assert payload["items"] == []
    assert payload["next_cursor"] is None
    assert payload["status"] == "needs_input"
    assert payload["redaction"]["source_run_ids"] == "excluded"
    assert payload["redaction"]["model_inputs"] == "excluded"


def test_event_stream_refresh_does_not_reconnect_stream(tmp_path) -> None:
    """Stream-triggered UI refreshes should not churn the SSE socket."""

    client, _ = _client_with_login(tmp_path, supervisor=_StaticStoppedSupervisor())

    script = client.get("/static/console.js")

    assert script.status_code == 200
    assert "bootstrap({reconnectStream: false})" in script.text
    assert "state.streamUrl === url" in script.text
    assert "return;" in script.text


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
    assert 'qs("#log-service-filter").addEventListener("change", refreshLogStream)' in (
        script.text
    )
    assert 'qs("#log-stream-filter").addEventListener("change", refreshLogStream)' in (
        script.text
    )
    assert (
        'qs("#log-text-filter").addEventListener("input", renderBufferedLogRows)'
        in script.text
    )
    assert (
        'qs("#log-highlight-filter").addEventListener("input", renderBufferedLogRows)'
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


def test_web_api_outputs_for_logs_events_audit_character_and_debug_error(
    monkeypatch,
    tmp_path,
) -> None:
    """Authenticated web API routes should return bounded JSON outputs."""

    from control_console import app as app_module
    from control_console import repository as repository_module
    import httpx

    async def read_kazusa_events(query):
        assert query.source == "kazusa"
        if query.service_id == "background_work.worker":
            rows = [{
                "source": "kazusa",
                "event_id": "event-bg-1",
                "event_type": "tick",
                "component": "background_work.worker",
                "level": "info",
                "status": "succeeded",
                "created_at": "2026-06-17T00:00:00+00:00",
            }]
        else:
            rows = [{
                "source": "kazusa",
                "event_id": "event-1",
                "event_type": "worker",
                "component": "background_work.worker",
                "level": "info",
                "status": "succeeded",
                "created_at": "2026-06-17T00:00:00+00:00",
                "human_prompt": "must not leak",
            }]
        return rows

    async def latest_character_status(self):
        _ = self
        return {"status": "empty", "items": []}

    async def global_growth_summary(self):
        _ = self
        return {"status": "empty", "items": []}

    async def lookup_due_calendar_runs(
        self,
        *,
        current_timestamp_utc: str,
        limit: int,
    ):
        _ = self
        assert current_timestamp_utc
        assert limit == 5
        return {"status": "empty", "items": [], "next_cursor": None}

    class FakeKazusaClient:
        def __init__(self, *, base_url: str, timeout_seconds: float) -> None:
            _ = base_url
            _ = timeout_seconds

        async def send_debug_chat(self, request):
            _ = request
            raise httpx.ConnectError("brain unavailable")

    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "latest_character_status",
        latest_character_status,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "global_growth_summary",
        global_growth_summary,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "lookup_due_calendar_runs",
        lookup_due_calendar_runs,
    )
    monkeypatch.setattr(
        app_module,
        "_read_kazusa_events",
        read_kazusa_events,
        raising=False,
    )
    monkeypatch.setattr(app_module, "KazusaClient", FakeKazusaClient)

    supervisor = _SyncBrainRunningSupervisor()
    client, _ = _client_with_login(tmp_path, supervisor=supervisor)

    logs = client.get("/api/logs/brain?limit=3")
    assert logs.status_code == 200
    assert logs.json() == {"items": [], "next_cursor": None}

    events = client.get("/api/events?source=kazusa&limit=5")
    assert events.status_code == 200
    events_payload = events.json()
    assert events_payload["items"][0]["source"] == "kazusa"
    assert events_payload["items"][0]["component"] == "background_work.worker"
    assert "human_prompt" not in repr(events_payload["items"])
    assert events_payload["next_cursor"] is None

    filtered_events = client.get(
        "/api/events?source=console&request_id=missing-request-id&limit=5",
    )
    assert filtered_events.status_code == 200
    assert filtered_events.json()["items"] == []

    audit = client.get("/api/audit?limit=10")
    assert audit.status_code == 200
    assert "generated_at" in audit.json()

    character = client.get("/api/character/status")
    assert character.status_code == 200
    assert character.json()["status"] in {"available", "empty", "unavailable"}

    growth = client.get("/api/character/growth")
    assert growth.status_code == 200
    assert growth.json()["status"] in {"available", "empty", "unavailable"}

    calendar = client.get("/api/lookups/calendar?limit=5")
    assert calendar.status_code == 200
    calendar_payload = calendar.json()
    assert calendar_payload["status"] in {"available", "empty", "unavailable"}
    assert calendar_payload["next_cursor"] is None

    background = client.get("/api/lookups/background?limit=5")
    assert background.status_code == 200
    background_payload = background.json()
    assert background_payload["status"] == "available"
    assert background_payload["items"][0]["component"] == "background_work.worker"
    assert background_payload["next_cursor"] is None

    debug = client.post(
        "/api/debug-chat",
        headers={},
        json={
            "channel_id": "debug",
            "user_id": "operator",
            "user_display_name": "Operator",
            "message_text": "hello",
        },
    )
    assert debug.status_code == 403

    login_payload = client.post("/api/auth/login", json={"token": "secret"}).json()
    failing_debug = client.post(
        "/api/debug-chat",
        headers={
            login_payload["csrf_header_name"]: login_payload["csrf_token"],
        },
        json={
            "channel_id": "debug",
            "user_id": "operator",
            "user_display_name": "Operator",
            "message_text": "hello",
        },
    )
    assert failing_debug.status_code == 200
    payload = failing_debug.json()
    assert payload["brain_available"] is False
    assert payload["error"]["code"] == "brain_unavailable"


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
    client, _ = _client_with_login(tmp_path)

    empty = client.get("/api/lookups/background?limit=5")
    assert empty.status_code == 200
    assert empty.json()["status"] == "empty"

    monkeypatch.setattr(app_module, "_read_kazusa_events", unavailable_events)
    unavailable = client.get("/api/lookups/background?limit=5")
    assert unavailable.status_code == 200
    payload = unavailable.json()
    assert payload["status"] == "unavailable"
    assert payload["items"][0]["event_type"] == "event_log.unavailable"


def test_auth_optional_mode_and_invalid_hash_rejections(
    monkeypatch,
    tmp_path,
) -> None:
    """Auth helpers should cover optional auth and malformed stored hashes."""

    from fastapi.testclient import TestClient

    from control_console import repository as repository_module
    from control_console.app import create_app
    from control_console.auth import verify_operator_token
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

    assert not verify_operator_token("secret", "")
    assert not verify_operator_token("secret", "not-a-valid-hash")
    assert not verify_operator_token("secret", "pbkdf2_sha256$bad$salt$digest")

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        require_auth=False,
    )
    client = TestClient(
        create_app(settings=settings, supervisor=_StaticStoppedSupervisor()),
    )

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    assert bootstrap.json()["operator"]["operator_id"] == "local_operator"

    lifecycle = client.post(
        "/api/services/external/start",
        json={"reason": "auth disabled unknown service"},
    )
    assert lifecycle.status_code == 404


def test_app_uses_live_debug_chat_timeout(monkeypatch, tmp_path) -> None:
    """The brain client timeout should allow a real local LLM chat turn."""

    from control_console import app as app_module
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    captured: dict[str, float] = {}

    class FakeKazusaClient:
        def __init__(
            self,
            *,
            base_url: str,
            timeout_seconds: float,
        ) -> None:
            """Capture the configured timeout without making network calls."""

            _ = base_url
            captured["timeout_seconds"] = timeout_seconds

    monkeypatch.setattr(app_module, "KazusaClient", FakeKazusaClient)
    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )

    app_module.create_app(settings=settings)

    assert captured["timeout_seconds"] == 120.0


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
