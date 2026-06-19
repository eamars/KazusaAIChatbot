"""FastAPI application factory for the local control console."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import asyncio
import inspect
import logging
import os
import secrets
import uuid

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import dotenv_values, find_dotenv
import httpx
from pymongo.errors import PyMongoError

from control_console.audit import LocalAuditWriter
from control_console.auth import (
    SessionStore,
    get_session_record,
    hash_operator_token,
    issue_login_response,
    require_csrf,
    require_operator,
    session_csrf_token,
    verify_operator_token,
)
from control_console.contracts import (
    ConsoleDebugChatRequest,
    ConsoleLookupQuery,
    ControlConsoleBootstrapResponse,
    ControlConsoleOperator,
    LoginRequest,
    OperationalEventQuery,
    ProcessLogQuery,
    ServiceActionRequest,
    ServiceConfigActionResponse,
    ServiceConfigApplyRequest,
    ServiceConfigResetRequest,
    ServiceConfigRestartResult,
)
from control_console.event_monitor import EventMonitor
from control_console.kazusa_client import KazusaClient, not_reported_cognition_graph
from control_console.log_store import ProcessLogStore
from control_console.process_store import ProcessStore
from control_console.redaction import redact_mapping
from control_console.repository import ControlConsoleRepository
from control_console.service_registry import load_service_registry
from control_console.service_config import (
    ServiceConfigOverrideStore,
    ServiceConfigRegistry,
    ServiceConfigValidationError,
    build_default_service_config_registry,
)
from control_console.settings import ControlConsoleSettings
from control_console.stream import SSEEventBuffer, encode_sse_event
from control_console.stream import LogStreamHub
from control_console.stream import log_keepalive_event
from control_console.stream import log_ready_event
from control_console.stream import log_snapshot_event
from control_console.stream import log_status_event
from control_console.stream import parse_log_streams
from control_console.supervisor import (
    ENDPOINT_CONFLICT_MESSAGE,
    ProcessSupervisor,
    ServiceLifecycleError,
)


STATIC_DIR = Path(__file__).parent / "static"
REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATED_OPERATOR_TOKEN_BYTES = 24
DEBUG_CHAT_TIMEOUT_SECONDS = 120.0
logger = logging.getLogger(__name__)
KazusaFindEvents = Callable[..., Awaitable[list[dict[str, Any]]]]
KAZUSA_EVENT_READER_ERRORS = (
    ImportError,
    KeyError,
    PyMongoError,
    ValueError,
)


def create_app(
    *,
    settings: ControlConsoleSettings | None = None,
    supervisor: Any | None = None,
) -> FastAPI:
    """Create the buildless control-console FastAPI app."""

    app_settings = settings or ControlConsoleSettings.from_env()
    generated_operator_token: str | None = None
    if app_settings.require_auth and not app_settings.operator_token_hash:
        generated_operator_token = secrets.token_urlsafe(
            GENERATED_OPERATOR_TOKEN_BYTES,
        )
        app_settings = app_settings.model_copy(
            update={
                "operator_token_hash": hash_operator_token(
                    generated_operator_token,
                ),
            },
        )
    app_settings.state_dir.mkdir(parents=True, exist_ok=True)
    sessions = SessionStore()
    services = load_service_registry(
        override_path=app_settings.service_registry_path,
        repo_root=REPO_ROOT,
    )
    service_config_registry = build_default_service_config_registry()
    service_config_overrides = ServiceConfigOverrideStore()

    def command_resolver(service_id: str, base_command: list[str]) -> list[str]:
        """Resolve descriptor-backed config into service start argv."""

        environment = _service_config_environment()
        try:
            rendered_command = service_config_registry.render_start_command(
                service_id=service_id,
                base_command=base_command,
                environment=environment,
                overrides=service_config_overrides,
            )
        except ServiceConfigValidationError as exc:
            raise ServiceLifecycleError(
                f"service config command overlay failed: {exc}"
            ) from exc
        return rendered_command

    audit_writer = LocalAuditWriter(app_settings.audit_path)
    log_stream_hub = LogStreamHub()
    log_store = ProcessLogStore(
        app_settings.log_dir,
        log_publisher=log_stream_hub.publish_log_line,
    )
    process_store = ProcessStore(app_settings.process_state_dir)
    app_supervisor = supervisor or ProcessSupervisor(
        services=services,
        store=process_store,
        log_store=log_store,
        audit_writer=audit_writer,
        command_resolver=command_resolver,
    )
    repository = ControlConsoleRepository()
    kazusa_client = KazusaClient(
        base_url=app_settings.brain_base_url,
        timeout_seconds=DEBUG_CHAT_TIMEOUT_SECONDS,
    )
    stream_buffer = SSEEventBuffer(max_events=100)
    stream_shutdown_event = asyncio.Event()
    latest_cognition_graph_state: dict[str, str | None] = {"run_id": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Prepare console runtime state."""

        if generated_operator_token is not None:
            logger.warning(
                "KAZUSA_CONTROL_OPERATOR_TOKEN_HASH was not set; generated "
                "an ephemeral local operator token for this console process.",
            )
            logger.warning(
                f"Control console access token: {generated_operator_token}",
            )
        stream_buffer.append(
            "control.heartbeat",
            {"generated_at": datetime.now(timezone.utc).isoformat()},
        )
        try:
            yield
        finally:
            stream_shutdown_event.set()
            if hasattr(app_supervisor, "shutdown_owned_services"):
                await app_supervisor.shutdown_owned_services(
                    operator_id="system",
                    reason="control console shutdown",
                )

    app = FastAPI(title="Control Console", lifespan=lifespan)
    app.state.generated_operator_token = generated_operator_token
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def current_operator(request: Request) -> ControlConsoleOperator:
        """FastAPI dependency for authenticated operators."""

        operator = require_operator(request, settings=app_settings, sessions=sessions)
        return operator

    def csrf_guard(request: Request) -> None:
        """FastAPI dependency for state-changing browser requests."""

        require_csrf(request, settings=app_settings, sessions=sessions)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        """Serve the static console shell."""

        path = STATIC_DIR / "index.html"
        html = path.read_text(encoding="utf-8")
        response = HTMLResponse(html)
        return response

    @app.get("/favicon.ico")
    async def favicon() -> FileResponse:
        """Serve the Kazusa avatar as the console favicon."""

        path = REPO_ROOT / "resources" / "avatar.png"
        response = FileResponse(path)
        return response

    @app.post("/api/auth/login")
    async def login(request: LoginRequest, response: Response) -> dict[str, object]:
        """Authenticate the local operator and issue CSRF metadata."""

        if not verify_operator_token(request.token, app_settings.operator_token_hash):
            audit_writer.write_event(
                event_type="auth_failed",
                operator_id="anonymous",
                target={"scope": "session"},
                request_id=f"cc-req-{uuid.uuid4().hex[:12]}",
            )
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

        payload = issue_login_response(
            response=response,
            settings=app_settings,
            sessions=sessions,
            operator_id="local_operator",
        )
        return payload

    @app.get("/api/auth/session")
    async def auth_session(request: Request) -> dict[str, object]:
        """Return current browser-session metadata without raising on lock."""

        record = get_session_record(
            request,
            settings=app_settings,
            sessions=sessions,
        )
        if record is None:
            return {"authenticated": False}

        payload: dict[str, object] = {
            "authenticated": True,
            "operator": {
                "operator_id": record.operator_id,
                "authenticated_at": record.authenticated_at.isoformat(),
            },
            "csrf_token": record.csrf_token,
            "csrf_header_name": app_settings.csrf_header_name,
        }
        return payload

    @app.get("/api/bootstrap")
    async def bootstrap(
        request: Request,
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return the initial UI snapshot."""

        states = _all_service_states(app_supervisor, services)
        brain_record = _service_state_record(states, service_id="brain")
        brain_state = _service_actual_state(brain_record)
        brain_error = _service_last_error_preview(brain_record)
        brain_http_available = _brain_http_available(
            brain_state,
            last_error_preview=brain_error,
        )
        brain_health: dict[str, Any]
        runtime_status: dict[str, Any]
        if brain_http_available:
            try:
                brain_health = await kazusa_client.get_health()
            except httpx.HTTPError as exc:
                brain_health = {
                    "status": "unavailable",
                    "reason": str(exc),
                }
            try:
                runtime_status = await kazusa_client.get_runtime_status()
            except httpx.HTTPError as exc:
                runtime_status = {
                    "status": "unavailable",
                    "reason": str(exc),
                }
        else:
            reason = f"brain service is {brain_state}"
            brain_health = {"status": "unavailable", "reason": reason}
            runtime_status = {"status": "unavailable", "reason": reason}

        raw_cache2 = brain_health.get("cache2", {})
        cache2 = raw_cache2 if isinstance(raw_cache2, dict) else {}
        latest_cognition_graph = not_reported_cognition_graph(
            source="overview_latest",
        )
        if brain_http_available:
            try:
                latest_cognition_graph = await (
                    kazusa_client.get_latest_cognition_graph()
                )
            except (AttributeError, httpx.HTTPError) as exc:
                latest_cognition_graph = not_reported_cognition_graph(
                    source="overview_latest",
                    reason=f"brain latest cognition graph unavailable: {exc}",
                )
        latest_cognition_graph_state["run_id"] = latest_cognition_graph.run_id
        overview = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "service_count": len(states),
            "brain_health": redact_mapping(brain_health),
            "runtime_status": redact_mapping(runtime_status),
            "cache2": redact_mapping(cache2),
            "latest_cognition_graph": latest_cognition_graph.model_dump(
                mode="json",
            ),
        }
        recent_audit = [
            event.model_dump(mode="json")
            for event in audit_writer.read_recent(limit=10)
        ]
        application_identity = await repository.application_identity()
        payload = ControlConsoleBootstrapResponse(
            generated_at=datetime.now(timezone.utc),
            operator=operator.model_dump(mode="json"),
            csrf_header_name=app_settings.csrf_header_name,
            csrf_token=session_csrf_token(
                request,
                settings=app_settings,
                sessions=sessions,
            ),
            application_identity=application_identity,
            services=states,
            overview=overview,
            latest_cognition_graph=latest_cognition_graph,
            recent_audit_events=recent_audit,
            event_counters={"audit": len(recent_audit), "services": len(states)},
            ui_capabilities={
                "event_stream": True,
                "debug_chat": True,
                "service_lifecycle": True,
                "lookups": True,
            },
            page_capabilities=_page_capabilities(),
            service_config_summaries=_service_config_summaries(
                states=states,
                registry=service_config_registry,
                environment=_service_config_environment(),
                overrides=service_config_overrides,
            ),
        )
        return payload.model_dump(mode="json")

    @app.post(
        "/api/services/{service_id}/start",
        dependencies=[Depends(csrf_guard)],
    )
    async def start_service(
        service_id: str,
        request: ServiceActionRequest,
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Start a registry service after auth, CSRF, and version checks."""

        response_payload = await _run_lifecycle_action(
            service_id=service_id,
            action="start",
            request=request,
            operator=operator,
            supervisor=app_supervisor,
            services=services,
        )
        stream_buffer.append(
            "control.service",
            {"service_id": service_id, "action": "start"},
        )
        return response_payload

    @app.post(
        "/api/services/{service_id}/stop",
        dependencies=[Depends(csrf_guard)],
    )
    async def stop_service(
        service_id: str,
        request: ServiceActionRequest,
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Stop a registry service."""

        response_payload = await _run_lifecycle_action(
            service_id=service_id,
            action="stop",
            request=request,
            operator=operator,
            supervisor=app_supervisor,
            services=services,
        )
        stream_buffer.append(
            "control.service",
            {"service_id": service_id, "action": "stop"},
        )
        return response_payload

    @app.post(
        "/api/services/{service_id}/restart",
        dependencies=[Depends(csrf_guard)],
    )
    async def restart_service(
        service_id: str,
        request: ServiceActionRequest,
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Restart a registry service."""

        response_payload = await _run_lifecycle_action(
            service_id=service_id,
            action="restart",
            request=request,
            operator=operator,
            supervisor=app_supervisor,
            services=services,
        )
        stream_buffer.append(
            "control.service",
            {"service_id": service_id, "action": "restart"},
        )
        return response_payload

    @app.get("/api/services/{service_id}/config")
    async def service_config(
        service_id: str,
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return a descriptor-driven config snapshot for one service."""

        request_id = f"cc-req-{uuid.uuid4().hex[:12]}"
        snapshot = _service_config_snapshot_or_http_error(
            service_id=service_id,
            services=services,
            registry=service_config_registry,
            overrides=service_config_overrides,
        )
        audit_writer.write_event(
            event_type="service_config_view",
            operator_id=operator.operator_id,
            service_id=service_id,
            target={"service_id": service_id, "field_count": len(snapshot.fields)},
            request_id=request_id,
        )
        return snapshot.model_dump(mode="json")

    @app.put(
        "/api/services/{service_id}/config",
        dependencies=[Depends(csrf_guard)],
    )
    async def apply_service_config(
        service_id: str,
        request: ServiceConfigApplyRequest,
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Store a process-local config override and restart if required."""

        response_payload = await _apply_service_config_change(
            service_id=service_id,
            request=request,
            operator=operator,
            supervisor=app_supervisor,
            services=services,
            registry=service_config_registry,
            overrides=service_config_overrides,
            audit_writer=audit_writer,
            clear_override=False,
        )
        stream_buffer.append(
            "control.service",
            {"service_id": service_id, "action": "config"},
        )
        return response_payload

    @app.post(
        "/api/services/{service_id}/config/reset",
        dependencies=[Depends(csrf_guard)],
    )
    async def reset_service_config(
        service_id: str,
        request: ServiceConfigResetRequest,
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Clear a process-local config override and restart if required."""

        response_payload = await _apply_service_config_change(
            service_id=service_id,
            request=request,
            operator=operator,
            supervisor=app_supervisor,
            services=services,
            registry=service_config_registry,
            overrides=service_config_overrides,
            audit_writer=audit_writer,
            clear_override=True,
        )
        stream_buffer.append(
            "control.service",
            {"service_id": service_id, "action": "config"},
        )
        return response_payload

    @app.get("/api/logs/stream")
    async def process_log_stream(
        request: Request,
        service_id: str = Query(default="all", max_length=80),
        streams: str = Query(default="stdout,stderr,supervisor", max_length=80),
        tail: int = Query(default=100, ge=0, le=500),
        cursor: str | None = Query(default=None, max_length=120),
        _: ControlConsoleOperator = Depends(current_operator),
    ) -> StreamingResponse:
        """Open the authenticated live process-log SSE stream."""

        if service_id != "all" and service_id not in services:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
        try:
            stream_filter = parse_log_streams(streams)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        effective_cursor = cursor
        if effective_cursor is None:
            effective_cursor = request.headers.get("last-event-id")
        response = StreamingResponse(
            _stream_process_logs(
                request=request,
                hub=log_stream_hub,
                log_store=log_store,
                supervisor=app_supervisor,
                services=services,
                service_id=service_id,
                streams=stream_filter,
                tail=tail,
                cursor=effective_cursor,
                keepalive_seconds=app_settings.sse_interval_seconds,
            ),
            media_type="text/event-stream",
        )
        return response

    @app.get("/api/logs/{service_id}")
    async def process_logs(
        service_id: str,
        limit: int = Query(default=100, ge=1, le=500),
        _: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return a bounded process-log tail."""

        query = ProcessLogQuery.model_validate({"service_id": service_id, "limit": limit})
        lines = log_store.tail(service_id=query.service_id, limit=query.limit)
        payload = {
            "items": [line.model_dump(mode="json") for line in lines],
            "next_cursor": None,
        }
        return payload

    @app.get("/api/events")
    async def events(
        source: str = "all",
        request_id: str | None = Query(default=None, max_length=120),
        tracking_id: str | None = Query(default=None, max_length=120),
        limit: int = Query(default=100, ge=1, le=200),
        _: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return a bounded merged event monitor page."""

        query = OperationalEventQuery.model_validate({
            "source": source,
            "request_id": request_id,
            "tracking_id": tracking_id,
            "limit": limit,
        })
        monitor = EventMonitor(
            read_audit_events=lambda event_query: _read_audit_events(
                event_query,
                audit_writer=audit_writer,
            ),
            read_process_logs=lambda event_query: _read_process_events(
                event_query,
                log_store=log_store,
                services=services,
            ),
            read_kazusa_events=_read_kazusa_events,
        )
        page = await monitor.query(query)
        audit_writer.write_event(
            event_type="event_view",
            operator_id=_.operator_id,
            target={
                "source": source,
                "request_id": request_id,
                "tracking_id": tracking_id,
                "limit": limit,
            },
        )
        return page.model_dump(mode="json")

    @app.get("/api/audit")
    async def audit_events(
        limit: int = Query(default=25, ge=1, le=100),
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return a bounded local audit page."""

        audit_writer.write_event(
            event_type="audit_view",
            operator_id=operator.operator_id,
            target={"limit": limit},
        )
        events = [
            event.model_dump(mode="json")
            for event in audit_writer.read_recent(limit=limit)
        ]
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "items": events,
            "next_cursor": None,
        }
        return payload

    @app.post(
        "/api/debug-chat",
        dependencies=[Depends(csrf_guard)],
    )
    async def debug_chat(
        request: ConsoleDebugChatRequest,
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Forward debug chat to the brain when the brain is running."""

        _ = operator
        request_id = f"cc-req-{uuid.uuid4().hex[:12]}"
        brain_record = _service_state_for_debug(app_supervisor, services)
        brain_state = _service_actual_state(brain_record)
        brain_error = _service_last_error_preview(brain_record)
        if not _brain_http_available(
            brain_state,
            last_error_preview=brain_error,
        ):
            audit_writer.write_event(
                event_type="debug_chat_unavailable",
                operator_id=operator.operator_id,
                target={
                    "channel_id": request.channel_id,
                    "user_id": request.user_id,
                    "message_text": request.message_text,
                },
                request_id=request_id,
            )
            payload = {
                "request_id": request_id,
                "brain_available": False,
                "request": redact_mapping(request.model_dump(mode="json")),
                "response": None,
                "tracking_id": None,
                "latency_ms": None,
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "error": {
                    "code": "brain_unavailable",
                    "message": "Brain HTTP endpoint is not available.",
                },
                "cognition_graph": not_reported_cognition_graph(
                    source="debug_latest",
                    run_id=request_id,
                    reason="debug chat did not start because brain is unavailable",
                ).model_dump(mode="json"),
            }
            return payload

        try:
            payload = await kazusa_client.send_debug_chat(request)
            stream_buffer.append(
                "control.cognition_graph_invalidated",
                {
                    "source": "debug_latest",
                    "run_id": payload.get("tracking_id") or payload.get("request_id"),
                },
            )
        except httpx.HTTPError as exc:
            audit_writer.write_event(
                event_type="debug_chat_unavailable",
                operator_id=operator.operator_id,
                target={
                    "channel_id": request.channel_id,
                    "user_id": request.user_id,
                    "message_text": request.message_text,
                },
                request_id=request_id,
            )
            payload = {
                "request_id": request_id,
                "brain_available": False,
                "request": redact_mapping(request.model_dump(mode="json")),
                "response": None,
                "tracking_id": None,
                "latency_ms": None,
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "error": {"code": "brain_unavailable", "message": str(exc)},
                "cognition_graph": not_reported_cognition_graph(
                    source="debug_latest",
                    run_id=request_id,
                    reason="debug chat failed before cognition telemetry was reported",
                ).model_dump(mode="json"),
            }
        return payload

    @app.get("/api/lookups/memory")
    async def lookup_memory(
        query: str = Query(default="", max_length=240),
        platform: str | None = Query(default=None, max_length=80),
        platform_user_id: str | None = Query(default=None, max_length=120),
        limit: int = Query(default=25, ge=1, le=100),
        _: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return a bounded memory lookup page."""

        lookup_query = ConsoleLookupQuery.model_validate({
            "query": query,
            "platform": platform,
            "platform_user_id": platform_user_id,
            "limit": limit,
        })
        audit_writer.write_event(
            event_type="lookup_view",
            operator_id=_.operator_id,
            target={
                "namespace": "memory",
                "query": lookup_query.query,
                "platform": lookup_query.platform or "",
                "has_platform_user_id": bool(lookup_query.platform_user_id),
            },
        )
        payload = await repository.lookup_memory(
            platform=lookup_query.platform or "",
            platform_user_id=lookup_query.platform_user_id or "",
            query=lookup_query.query,
            limit=lookup_query.limit,
        )
        return payload

    @app.get("/api/lookups/style")
    async def lookup_style(
        platform: str | None = Query(default=None, max_length=80),
        platform_user_id: str | None = Query(default=None, max_length=120),
        group_id: str | None = Query(default=None, max_length=120),
        limit: int = Query(default=25, ge=1, le=100),
        _: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return scoped interaction-style guidance."""

        lookup_query = ConsoleLookupQuery.model_validate({
            "platform": platform,
            "platform_user_id": platform_user_id,
            "group_id": group_id,
            "limit": limit,
        })
        audit_writer.write_event(
            event_type="lookup_view",
            operator_id=_.operator_id,
            target={
                "namespace": "style",
                "platform": lookup_query.platform or "",
                "has_platform_user_id": bool(lookup_query.platform_user_id),
                "has_group_id": bool(lookup_query.group_id),
            },
        )
        payload = await repository.lookup_interaction_style(
            platform=lookup_query.platform or "",
            platform_user_id=lookup_query.platform_user_id or "",
            platform_channel_id=lookup_query.group_id or "",
            limit=lookup_query.limit,
        )
        return payload

    @app.get("/api/lookups/calendar")
    async def lookup_calendar(
        limit: int = Query(default=25, ge=1, le=100),
        _: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return bounded due calendar-run state."""

        lookup_query = ConsoleLookupQuery.model_validate({"limit": limit})
        audit_writer.write_event(
            event_type="lookup_view",
            operator_id=_.operator_id,
            target={
                "namespace": "calendar",
                "limit": lookup_query.limit,
            },
        )
        payload = await repository.lookup_due_calendar_runs(
            current_timestamp_utc=datetime.now(timezone.utc).isoformat(),
            limit=lookup_query.limit,
        )
        return payload

    @app.get("/api/lookups/background")
    async def lookup_background_work(
        limit: int = Query(default=25, ge=1, le=100),
        _: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return recent background-work worker telemetry rows."""

        query = OperationalEventQuery.model_validate({
            "source": "kazusa",
            "service_id": "background_work.worker",
            "limit": limit,
        })
        audit_writer.write_event(
            event_type="lookup_view",
            operator_id=_.operator_id,
            target={
                "namespace": "background",
                "limit": query.limit,
            },
        )
        rows = await _read_kazusa_events(query)
        unavailable = any(
            row.get("event_type") == "event_log.unavailable"
            for row in rows
        )
        if unavailable:
            status_label = "unavailable"
            reason = "background-work event telemetry is unavailable"
        elif rows:
            status_label = "available"
            reason = ""
        else:
            status_label = "empty"
            reason = "no background-work worker events matched the lookup"
        payload = {
            "status": status_label,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "items": rows,
            "next_cursor": None,
            "reason": reason,
            "redaction": {
                "job_payloads": "excluded",
                "task_briefs": "excluded",
                "artifact_text": "excluded",
                "raw_messages": "excluded",
            },
        }
        return payload

    @app.get("/api/entities/character")
    async def character_entity(
        limit: int = Query(default=25, ge=1, le=100),
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return the owner-oriented character inspection envelope."""

        lookup_query = ConsoleLookupQuery.model_validate({"limit": limit})
        audit_writer.write_event(
            event_type="lookup_view",
            operator_id=operator.operator_id,
            target={
                "namespace": "entity.character",
                "limit": lookup_query.limit,
            },
        )
        payload = await repository.character_entity(limit=lookup_query.limit)
        return payload

    @app.get("/api/entities/user")
    async def user_entity(
        query: str = Query(default="", max_length=240),
        platform: str | None = Query(default=None, max_length=80),
        platform_user_id: str | None = Query(default=None, max_length=120),
        limit: int = Query(default=25, ge=1, le=100),
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return a platform-facing user owner inspection envelope."""

        lookup_query = ConsoleLookupQuery.model_validate({
            "query": query,
            "platform": platform,
            "platform_user_id": platform_user_id,
            "limit": limit,
        })
        audit_writer.write_event(
            event_type="lookup_view",
            operator_id=operator.operator_id,
            target={
                "namespace": "entity.user",
                "query": lookup_query.query,
                "platform": lookup_query.platform or "",
                "has_platform_user_id": bool(lookup_query.platform_user_id),
                "limit": lookup_query.limit,
            },
        )
        payload = await repository.lookup_user_entity(
            platform=lookup_query.platform or "",
            platform_user_id=lookup_query.platform_user_id or "",
            query=lookup_query.query,
            limit=lookup_query.limit,
        )
        return payload

    @app.get("/api/entities/group")
    async def group_entity(
        platform: str | None = Query(default=None, max_length=80),
        group_id: str | None = Query(default=None, max_length=120),
        limit: int = Query(default=25, ge=1, le=100),
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return a platform-facing group owner inspection envelope."""

        lookup_query = ConsoleLookupQuery.model_validate({
            "platform": platform,
            "group_id": group_id,
            "limit": limit,
        })
        audit_writer.write_event(
            event_type="lookup_view",
            operator_id=operator.operator_id,
            target={
                "namespace": "entity.group",
                "platform": lookup_query.platform or "",
                "has_group_id": bool(lookup_query.group_id),
                "limit": lookup_query.limit,
            },
        )
        payload = await repository.lookup_group_entity(
            platform=lookup_query.platform or "",
            group_id=lookup_query.group_id or "",
            limit=lookup_query.limit,
        )
        return payload

    @app.get("/api/character/status")
    async def character_status(
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return the latest bounded character status projection."""

        audit_writer.write_event(
            event_type="lookup_view",
            operator_id=operator.operator_id,
            target={"namespace": "character_status"},
        )
        payload = await repository.latest_character_status()
        return payload

    @app.get("/api/character/growth")
    async def character_growth(
        operator: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return bounded global character-growth projection."""

        audit_writer.write_event(
            event_type="lookup_view",
            operator_id=operator.operator_id,
            target={"namespace": "global_growth"},
        )
        payload = await repository.global_growth_summary()
        return payload

    @app.get("/api/lookups/{namespace}")
    async def lookup_empty(
        namespace: str,
        limit: int = Query(default=25, ge=1, le=100),
        _: ControlConsoleOperator = Depends(current_operator),
    ) -> dict[str, Any]:
        """Return safe empty lookup pages for not-yet-wired helpers."""

        audit_writer.write_event(
            event_type="lookup_view",
            operator_id=_.operator_id,
            target={"namespace": namespace, "limit": limit},
        )
        payload = await repository.empty_lookup(namespace=namespace)
        return payload

    @app.get("/api/stream")
    async def stream(
        request: Request,
        _: ControlConsoleOperator = Depends(current_operator),
    ) -> StreamingResponse:
        """Open the read-only compact SSE stream."""

        response = StreamingResponse(
            _stream_console_events(
                request=request,
                stream_buffer=stream_buffer,
                shutdown_event=stream_shutdown_event,
                supervisor=app_supervisor,
                services=services,
                kazusa_client=kazusa_client,
                latest_cognition_graph_state=latest_cognition_graph_state,
                interval_seconds=app_settings.sse_interval_seconds,
            ),
            media_type="text/event-stream",
        )
        return response

    return app


async def _stream_console_events(
    *,
    request: Request,
    stream_buffer: SSEEventBuffer,
    shutdown_event: asyncio.Event,
    supervisor: Any,
    services: dict[str, Any],
    kazusa_client: Any,
    latest_cognition_graph_state: dict[str, str | None],
    interval_seconds: float,
) -> AsyncIterator[str]:
    """Yield compact console SSE events until disconnect or shutdown."""

    last_event_id = request.headers.get("last-event-id")
    replay_events = stream_buffer.replay_after(last_event_id)
    for event in replay_events:
        yield encode_sse_event(event)
    last_stream_event_id = (
        replay_events[-1].event_id
        if replay_events
        else last_event_id
    )
    while not shutdown_event.is_set():
        if await request.is_disconnected():
            break
        current_states = _all_service_states(supervisor, services)
        current_brain_record = _service_state_record(
            current_states,
            service_id="brain",
        )
        current_brain_state = _service_actual_state(current_brain_record)
        current_brain_error = _service_last_error_preview(current_brain_record)
        if _brain_http_available(
            current_brain_state,
            last_error_preview=current_brain_error,
        ):
            latest_cognition_graph_state["run_id"] = await (
                _append_cognition_graph_invalidation_if_changed(
                    kazusa_client=kazusa_client,
                    stream_buffer=stream_buffer,
                    previous_run_id=latest_cognition_graph_state.get("run_id"),
                )
            )
        stream_buffer.append(
            "control.heartbeat",
            {"generated_at": datetime.now(timezone.utc).isoformat()},
        )
        pending_events = stream_buffer.replay_after(last_stream_event_id)
        for event in pending_events:
            yield encode_sse_event(event)
            last_stream_event_id = event.event_id
        should_stop = await _wait_for_stream_tick(
            shutdown_event=shutdown_event,
            interval_seconds=interval_seconds,
        )
        if should_stop:
            break


async def _stream_process_logs(
    *,
    request: Request,
    hub: LogStreamHub,
    service_id: str,
    streams: set[str],
    tail: int,
    cursor: str | None,
    keepalive_seconds: float,
    log_store: ProcessLogStore | None = None,
    supervisor: Any | None = None,
    services: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Yield live process-log SSE events until the browser disconnects."""

    has_persisted_snapshot = (
        cursor is None
        and log_store is not None
        and services is not None
    )
    if has_persisted_snapshot:
        for event in _initial_log_snapshot_events(
            log_store=log_store,
            services=services,
            service_id=service_id,
            streams=streams,
            tail=tail,
        ):
            yield encode_sse_event(event)

    if supervisor is not None and services is not None:
        for event in _log_status_events(
            supervisor=supervisor,
            services=services,
            service_id=service_id,
        ):
            yield encode_sse_event(event)

    if not has_persisted_snapshot:
        for event in hub.replay_after(
            cursor=cursor,
            service_id=service_id,
            streams=streams,
            tail=tail,
        ):
            yield encode_sse_event(event)

    subscription = hub.subscribe(service_id=service_id, streams=streams)
    try:
        yield encode_sse_event(log_ready_event())
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(
                    subscription.queue.get(),
                    timeout=keepalive_seconds,
                )
            except TimeoutError:
                yield encode_sse_event(log_keepalive_event())
                continue
            yield encode_sse_event(event)
    finally:
        hub.unsubscribe(subscription)


def _initial_log_snapshot_events(
    *,
    log_store: ProcessLogStore,
    services: dict[str, Any],
    service_id: str,
    streams: set[str],
    tail: int,
) -> list[Any]:
    """Return bounded retained log rows for the live-log stream."""

    if tail <= 0:
        events: list[Any] = []
        return events

    service_ids = _log_service_ids(service_id=service_id, services=services)
    selected_events = []
    for current_service_id in service_ids:
        lines = log_store.tail(service_id=current_service_id, limit=tail)
        for line in lines:
            if line.stream in streams:
                selected_events.append(log_snapshot_event(line))
    events = selected_events[-tail:]
    return events


def _log_status_events(
    *,
    supervisor: Any,
    services: dict[str, Any],
    service_id: str,
) -> list[Any]:
    """Return service availability status rows for the live-log stream."""

    states = _all_service_states(supervisor, services)
    service_ids = _log_service_ids(service_id=service_id, services=services)
    events = []
    for current_service_id in service_ids:
        state = _service_state_record(states, service_id=current_service_id)
        actual_state = _service_actual_state(state)
        last_error_preview = _service_last_error_preview(state)
        if actual_state == "conflict" and last_error_preview == ENDPOINT_CONFLICT_MESSAGE:
            events.append(
                log_status_event(
                    service_id=current_service_id,
                    status="unavailable",
                    message="logs unavailable from this console run",
                ),
            )
    return events


def _log_service_ids(*, service_id: str, services: dict[str, Any]) -> list[str]:
    """Return registry service ids visible to one log-stream request."""

    if service_id == "all":
        service_ids = list(services)
        return service_ids
    return [service_id]


async def _wait_for_stream_tick(
    *,
    shutdown_event: asyncio.Event,
    interval_seconds: float,
) -> bool:
    """Wait for the next SSE heartbeat or return when shutdown is requested."""

    if shutdown_event.is_set():
        return True

    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
    except TimeoutError:
        return False

    return True


async def _append_cognition_graph_invalidation_if_changed(
    *,
    kazusa_client: Any,
    stream_buffer: SSEEventBuffer,
    previous_run_id: str | None,
) -> str | None:
    """Append a graph invalidation event when the brain latest run changes."""

    try:
        latest_graph = await kazusa_client.get_latest_cognition_graph()
    except (AttributeError, httpx.HTTPError):
        return previous_run_id

    latest_run_id = latest_graph.run_id
    if not latest_run_id:
        return previous_run_id
    if latest_run_id and latest_run_id != previous_run_id:
        stream_buffer.append(
            "control.cognition_graph_invalidated",
            {
                "run_id": latest_run_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return latest_run_id


async def _read_no_events(_: OperationalEventQuery) -> list[dict[str, Any]]:
    """Return no events for unavailable source helpers."""

    rows: list[dict[str, Any]] = []
    return rows


async def _read_kazusa_events(
    query: OperationalEventQuery,
    *,
    find_events: KazusaFindEvents | None = None,
) -> list[dict[str, Any]]:
    """Read bounded Kazusa event-log rows for the event monitor."""

    filter_doc = _kazusa_event_filter(query)
    event_finder = find_events
    try:
        if event_finder is None:
            from kazusa_ai_chatbot.event_logging import repository as event_repository

            event_finder = event_repository.find_events
    except KAZUSA_EVENT_READER_ERRORS as exc:
        rows = [_kazusa_event_reader_unavailable(reason=str(exc))]
        return rows

    try:
        documents = await event_finder(
            filter_doc,
            sort=[("occurred_at", -1)],
            limit=query.limit,
        )
    except KAZUSA_EVENT_READER_ERRORS as exc:
        rows = [_kazusa_event_reader_unavailable(reason=str(exc))]
        return rows

    rows = [
        _project_kazusa_event(document)
        for document in list(documents)[: query.limit]
        if isinstance(document, dict)
    ]
    return rows


def _kazusa_event_filter(query: OperationalEventQuery) -> dict[str, Any]:
    """Build a bounded event-log query from operator filters."""

    filter_doc: dict[str, Any] = {}
    if query.service_id:
        filter_doc["component"] = query.service_id
    if query.event_type:
        filter_doc["event_type"] = query.event_type
    if query.level:
        filter_doc["severity"] = query.level
    if query.request_id:
        filter_doc["correlation_id"] = query.request_id
    if query.tracking_id:
        filter_doc["$or"] = [
            {"run_id": query.tracking_id},
            {"trigger_id": query.tracking_id},
            {"attempt_id": query.tracking_id},
            {"refs.ref_id": query.tracking_id},
        ]
    if query.since:
        filter_doc["occurred_at"] = {"$gte": query.since.isoformat()}
    return filter_doc


def _project_kazusa_event(document: dict[str, Any]) -> dict[str, Any]:
    """Project one event-log document into a browser-safe event row."""

    occurred_at = document.get("occurred_at")
    created_at = occurred_at or document.get("created_at", "")
    row = {
        "source": "kazusa",
        "event_id": str(document.get("event_id", "")),
        "event_family": str(document.get("event_family", "")),
        "event_type": str(document.get("event_type", "")),
        "component": str(document.get("component", "")),
        "level": str(document.get("severity", "")),
        "status": str(document.get("status", "")),
        "correlation_id": str(document.get("correlation_id", "")),
        "run_id": str(document.get("run_id", "")),
        "trigger_id": str(document.get("trigger_id", "")),
        "attempt_id": str(document.get("attempt_id", "")),
        "created_at": str(created_at),
        "duration_ms": document.get("duration_ms"),
    }
    error = document.get("error")
    if isinstance(error, dict):
        error_class = error.get("error_class")
        error_preview = error.get("error_preview")
        if error_class:
            row["error_class"] = str(error_class)
        if error_preview:
            row["message"] = str(error_preview)
    projected_row = redact_mapping({
        key: value
        for key, value in row.items()
        if value not in (None, "")
    })
    return projected_row


def _kazusa_event_reader_unavailable(*, reason: str) -> dict[str, Any]:
    """Build an explicit event-row fallback when event-log reads fail."""

    row = {
        "source": "kazusa",
        "event_type": "event_log.unavailable",
        "level": "warning",
        "status": "unavailable",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": f"Kazusa event log reader unavailable: {reason[:120]}",
    }
    projected_row = redact_mapping(row)
    return projected_row


async def _read_audit_events(
    query: OperationalEventQuery,
    *,
    audit_writer: LocalAuditWriter,
) -> list[dict[str, Any]]:
    """Read local control-console audit events for the event monitor."""

    events = audit_writer.read_recent(limit=query.limit)
    rows = [
        {
            **event.model_dump(mode="json"),
            "source": "console",
            "level": "info",
        }
        for event in events
        if _event_matches_filters(event.model_dump(mode="json"), query)
    ]
    return rows


async def _read_process_events(
    query: OperationalEventQuery,
    *,
    log_store: ProcessLogStore,
    services: dict[str, Any],
) -> list[dict[str, Any]]:
    """Read bounded process log lines as event-monitor rows."""

    service_ids = [query.service_id] if query.service_id else list(services)
    rows: list[dict[str, Any]] = []
    for service_id in service_ids:
        if service_id not in services:
            continue
        for line in log_store.tail(service_id=service_id, limit=query.limit):
            rows.append({
                "source": "process",
                "service_id": line.service_id,
                "event_type": f"log.{line.stream}",
                "level": "error" if line.stream == "stderr" else "info",
                "created_at": line.created_at.isoformat(),
                "message": line.line,
                "cursor": line.cursor,
            })
    return rows[: query.limit]


def _event_matches_filters(
    event: dict[str, Any],
    query: OperationalEventQuery,
) -> bool:
    """Return whether one audit event matches bounded UI filters."""

    if query.service_id and event.get("service_id") != query.service_id:
        return False
    if query.event_type and event.get("event_type") != query.event_type:
        return False
    if query.request_id and event.get("request_id") != query.request_id:
        return False
    if query.tracking_id and event.get("tracking_id") != query.tracking_id:
        return False
    return True


async def _run_lifecycle_action(
    *,
    service_id: str,
    action: str,
    request: ServiceActionRequest,
    operator: ControlConsoleOperator,
    supervisor: Any,
    services: dict[str, Any],
) -> dict[str, Any]:
    """Run one lifecycle action after service and version checks."""

    if service_id not in services:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    current_version_result = supervisor.service_version(service_id)
    if inspect.isawaitable(current_version_result):
        current_version = await current_version_result
    else:
        current_version = current_version_result
    if (
        request.expected_version is not None
        and request.expected_version != current_version
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"current_version": current_version},
        )

    try:
        if action == "start":
            result = await supervisor.start_service(
                service_id=service_id,
                operator_id=operator.operator_id,
                reason=request.reason,
            )
        elif action == "stop":
            result = await supervisor.stop_service(
                service_id=service_id,
                operator_id=operator.operator_id,
                reason=request.reason,
            )
        else:
            result = await supervisor.restart_service(
                service_id=service_id,
                operator_id=operator.operator_id,
                reason=request.reason,
            )
    except ServiceLifecycleError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": str(exc)},
        ) from exc
    return result


async def _apply_service_config_change(
    *,
    service_id: str,
    request: ServiceConfigApplyRequest | ServiceConfigResetRequest,
    operator: ControlConsoleOperator,
    supervisor: Any,
    services: dict[str, Any],
    registry: ServiceConfigRegistry,
    overrides: ServiceConfigOverrideStore,
    audit_writer: LocalAuditWriter,
    clear_override: bool,
) -> dict[str, Any]:
    """Apply or clear an ephemeral override and restart a running service."""

    if service_id not in services:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if not registry.has_descriptor(service_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    request_id = f"cc-req-{uuid.uuid4().hex[:12]}"
    current_version = await _service_version(
        supervisor=supervisor,
        service_id=service_id,
    )
    if (
        request.expected_version is not None
        and request.expected_version != current_version
    ):
        audit_writer.write_event(
            event_type=(
                "service_config_reset_failed"
                if clear_override
                else "service_config_apply_failed"
            ),
            operator_id=operator.operator_id,
            service_id=service_id,
            target={
                "service_id": service_id,
                "current_version": current_version,
            },
            reason="expected version mismatch",
            request_id=request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"current_version": current_version},
        )

    event_type = (
        "service_config_reset_requested"
        if clear_override
        else "service_config_apply_requested"
    )
    field_keys: list[str] = []
    if isinstance(request, ServiceConfigApplyRequest):
        field_keys = list(request.values)
    audit_writer.write_event(
        event_type=event_type,
        operator_id=operator.operator_id,
        service_id=service_id,
        target={"service_id": service_id, "field_keys": field_keys},
        reason=request.reason,
        request_id=request_id,
    )

    environment = _service_config_environment()
    previous_snapshot = _service_config_snapshot_or_http_error(
        service_id=service_id,
        services=services,
        registry=registry,
        overrides=overrides,
    )
    try:
        if clear_override:
            overrides.clear_override(service_id=service_id)
        elif isinstance(request, ServiceConfigApplyRequest):
            overrides.set_override(
                service_id=service_id,
                values=request.values,
                registry=registry,
                environment=environment,
            )
    except ServiceConfigValidationError as exc:
        audit_writer.write_event(
            event_type=(
                "service_config_reset_failed"
                if clear_override
                else "service_config_apply_failed"
            ),
            operator_id=operator.operator_id,
            service_id=service_id,
            target={"service_id": service_id, "field_keys": field_keys},
            previous_state=previous_snapshot.model_dump(mode="json"),
            reason=str(exc),
            request_id=request_id,
        )
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc)},
        ) from exc

    overrides.clear_apply_failed(service_id)
    service_state = supervisor.service_state(service_id)
    should_restart = _service_actual_state(service_state) == "running"
    restart = ServiceConfigRestartResult(
        attempted=should_restart,
        succeeded=None,
        reason="config apply requires restart",
    )
    service_payload = _service_state_payload(service_state)
    if should_restart:
        audit_writer.write_event(
            event_type="service_config_restart_requested",
            operator_id=operator.operator_id,
            service_id=service_id,
            target={"service_id": service_id},
            reason="config apply requires restart",
            request_id=request_id,
        )
        try:
            restart_result = await supervisor.restart_service(
                service_id=service_id,
                operator_id=operator.operator_id,
                reason="config apply requires restart",
            )
        except ServiceLifecycleError as exc:
            restart = ServiceConfigRestartResult(
                attempted=True,
                succeeded=False,
                reason=f"config apply requires restart: {exc}",
            )
            service_state = supervisor.service_state(service_id)
            service_payload = _service_state_payload(service_state)
            overrides.mark_apply_failed(service_id)
        else:
            restart = ServiceConfigRestartResult(
                attempted=True,
                succeeded=True,
                reason="config apply requires restart",
            )
            raw_service_payload = restart_result.get("service", service_payload)
            service_payload = raw_service_payload
            overrides.clear_apply_failed(service_id)
    else:
        overrides.clear_apply_failed(service_id)

    next_snapshot = _service_config_snapshot_or_http_error(
        service_id=service_id,
        services=services,
        registry=registry,
        overrides=overrides,
    )
    applied_event_type = "service_config_applied"
    if restart.succeeded is False:
        applied_event_type = (
            "service_config_reset_failed"
            if clear_override
            else "service_config_apply_failed"
        )
    applied_event = audit_writer.write_event(
        event_type=applied_event_type,
        operator_id=operator.operator_id,
        service_id=service_id,
        target={"service_id": service_id, "field_keys": field_keys},
        previous_state=previous_snapshot.model_dump(mode="json"),
        new_state={
            "config_state": next_snapshot.state,
            "restart": restart.model_dump(mode="json"),
        },
        reason=request.reason,
        request_id=request_id,
    )
    response = ServiceConfigActionResponse(
        service_id=service_id,
        config=next_snapshot.model_dump(mode="json"),
        service=service_payload,
        restart=restart,
        audit_event_id=applied_event.event_id,
    )
    response_payload = response.model_dump(mode="json")
    return response_payload


async def _service_version(*, supervisor: Any, service_id: str) -> int:
    """Return a service version from sync or async supervisor implementations."""

    current_version_result = supervisor.service_version(service_id)
    if inspect.isawaitable(current_version_result):
        current_version = await current_version_result
    else:
        current_version = current_version_result
    version = int(current_version)
    return version


def _service_config_snapshot_or_http_error(
    *,
    service_id: str,
    services: dict[str, Any],
    registry: ServiceConfigRegistry,
    overrides: ServiceConfigOverrideStore,
) -> Any:
    """Return a config snapshot or raise the API-level status code."""

    if service_id not in services:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    if not registry.has_descriptor(service_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    try:
        snapshot = registry.snapshot_for_service(
            service_id=service_id,
            environment=_service_config_environment(),
            overrides=overrides,
        )
    except ServiceConfigValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc)},
        ) from exc
    return snapshot


def _service_config_summaries(
    *,
    states: list[Any],
    registry: ServiceConfigRegistry,
    environment: dict[str, str],
    overrides: ServiceConfigOverrideStore,
) -> dict[str, dict[str, Any]]:
    """Return compact config state for service-card entrypoints."""

    state_ids = {
        state.get("id") if isinstance(state, dict) else getattr(state, "id", "")
        for state in states
    }
    summaries: dict[str, dict[str, Any]] = {}
    for service_id in registry.configurable_service_ids():
        if service_id not in state_ids:
            continue
        try:
            snapshot = registry.snapshot_for_service(
                service_id=service_id,
                environment=environment,
                overrides=overrides,
            )
        except ServiceConfigValidationError:
            summaries[service_id] = {
                "configurable": True,
                "state": "unavailable",
                "apply_behavior": "restart",
                "field_count": 0,
            }
            continue
        summaries[service_id] = {
            "configurable": True,
            "state": snapshot.state,
            "apply_behavior": snapshot.apply_behavior,
            "field_count": len(snapshot.fields),
        }
    return summaries


def _service_config_environment() -> dict[str, str]:
    """Build config defaults from dotenv values with process env overrides."""

    dotenv_path = find_dotenv(usecwd=True)
    dotenv_config = dotenv_values(dotenv_path) if dotenv_path else {}
    environment: dict[str, str] = {}
    for key, value in dotenv_config.items():
        if value is not None:
            environment[key] = value
    environment.update(os.environ)
    return environment


def _service_state_payload(state: Any) -> dict[str, Any]:
    """Serialize one service state model or dictionary for API output."""

    if state is None:
        payload: dict[str, Any] = {"actual_state": "unavailable"}
        return payload
    if isinstance(state, dict):
        payload = dict(state)
        return payload
    if hasattr(state, "model_dump"):
        payload = state.model_dump(mode="json")
        return payload
    payload = {"actual_state": str(state)}
    return payload


def _all_service_states(supervisor: Any, services: dict[str, Any]) -> list[Any]:
    """Return service states from the supervisor or a static fallback."""

    if hasattr(supervisor, "all_service_states"):
        states = supervisor.all_service_states()
        return states

    states = [
        {
            "id": service.id,
            "display_name": service.display_name,
            "kind": service.kind,
            "desired_state": "stopped",
            "actual_state": "stopped",
            "dependencies": list(service.dependencies),
            "version": 0,
        }
        for service in services.values()
    ]
    return states


def _service_state_record(states: list[Any], *, service_id: str) -> Any | None:
    """Return one service state record from a mixed state list."""

    for state in states:
        if isinstance(state, dict):
            state_id = state.get("id")
        else:
            state_id = getattr(state, "id", "")
        if state_id == service_id:
            return state
    return None


def _service_actual_state(state: Any | None) -> str:
    """Return the actual state from a dict, model, or missing record."""

    if state is None:
        return "unavailable"
    if isinstance(state, dict):
        actual_state = state.get("actual_state", "unknown")
    else:
        actual_state = getattr(state, "actual_state", "unknown")
    state_text = str(actual_state)
    return state_text


def _service_last_error_preview(state: Any | None) -> str:
    """Return one service state's bounded error preview."""

    if state is None:
        return ""
    if isinstance(state, dict):
        last_error_preview = state.get("last_error_preview") or ""
    else:
        last_error_preview = getattr(state, "last_error_preview", "") or ""
    error_text = str(last_error_preview)
    return error_text


def _brain_http_available(
    actual_state: str,
    *,
    last_error_preview: str = "",
) -> bool:
    """Return whether the configured brain HTTP API may be queried."""

    if actual_state == "running":
        return True
    is_available = (
        actual_state == "conflict"
        and last_error_preview == ENDPOINT_CONFLICT_MESSAGE
    )
    return is_available


def _page_capabilities() -> dict[str, dict[str, Any]]:
    """Return current sidebar capability status for the browser shell."""

    capabilities = {
        "overview": {
            "status": "ready",
            "label": "ready",
            "reason": "Service and audit summary is available after login.",
        },
        "services": {
            "status": "ready",
            "label": "lifecycle",
            "reason": "Registry lifecycle controls are available.",
        },
        "logs": {
            "status": "ready",
            "label": "live tail",
            "reason": "Console-owned process logs are available through a bounded live stream.",
        },
        "debug": {
            "status": "ready",
            "label": "brain gated",
            "reason": "Debug chat uses the existing brain /chat contract when the brain is running.",
        },
        "events": {
            "status": "ready",
            "label": "event log",
            "reason": "Local audit, process logs, and application event-log telemetry are available.",
        },
        "character": {
            "status": "partial",
            "label": "partial",
            "reason": "Character profile, state, growth, and safe learning panels are available; raw reflection output is excluded.",
        },
        "users": {
            "status": "partial",
            "label": "platform lookup",
            "reason": "Platform-facing user profile, relationship, memory, and style panels are available.",
        },
        "groups": {
            "status": "partial",
            "label": "group lookup",
            "reason": "Platform-facing group style and safe context panels are available.",
        },
        "calendar": {
            "status": "partial",
            "label": "due runs",
            "reason": "Due calendar-run inspection is available; schedule editing is not implemented.",
        },
        "background": {
            "status": "partial",
            "label": "worker events",
            "reason": "Background worker event telemetry is available; job payload browsing is not implemented.",
        },
        "health": {
            "status": "partial",
            "label": "runtime gated",
            "reason": "Brain health and runtime status are live when the brain HTTP endpoint is available.",
        },
        "audit": {
            "status": "partial",
            "label": "local only",
            "reason": "Local JSONL audit is available; Mongo audit mirroring is not implemented.",
        },
    }
    return capabilities


def _service_state_for_debug(supervisor: Any, services: dict[str, Any]) -> Any | None:
    """Return the current brain state for debug-chat availability."""

    if "brain" not in services:
        return None
    if hasattr(supervisor, "service_state"):
        state = supervisor.service_state("brain")
        return state
    return None
