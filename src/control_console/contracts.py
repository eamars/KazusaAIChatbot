"""Pydantic contracts for the local control console API."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
    CognitionRunObservationV1,
)

SERVICE_ID_PATTERN = r"^[a-z0-9][a-z0-9_.-]{0,63}$"
COGNITION_VIEW_REASON_PATTERN = r"^(?:|[a-z][a-z0-9_]{0,79})$"
SHELL_META_CHARS = frozenset({"&&", "||", ";", "|", ">", "<", "&"})
DENIED_COMMAND_EXECUTABLES = frozenset({
    "bash",
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "sh",
    "sh.exe",
    "zsh",
})
DENIED_INLINE_EXECUTION_FLAGS = frozenset({
    "-c",
    "/c",
    "-command",
    "/command",
    "-encodedcommand",
    "/encodedcommand",
    "-enc",
    "/enc",
})
SENSITIVE_HOST_FIELDS = frozenset({
    "pid",
    "process_id",
    "container_id",
    "container_name",
    "compose_project",
    "systemd_service",
    "windows_service",
    "kubernetes_name",
    "remote_host",
    "ssh_target",
})


class StrictModel(BaseModel):
    """Base model that rejects unknown API fields."""

    model_config = ConfigDict(extra="forbid")


class ServiceSpec(StrictModel):
    """Registry specification for one manageable local service."""

    id: str = Field(pattern=SERVICE_ID_PATTERN)
    display_name: str = Field(min_length=1, max_length=80)
    kind: Literal["backend", "frontend", "adapter", "worker", "support"]
    command: list[str] = Field(min_length=1, max_length=32)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list, max_length=16)
    health_url: str | None = None
    ready_match: str | None = Field(default=None, max_length=120)
    startup_timeout_seconds: float | None = Field(default=None, gt=0, le=300)
    shutdown_timeout_seconds: float | None = Field(default=None, gt=0, le=120)
    autostart: bool = False
    log_name: str | None = Field(default=None, max_length=80)

    @field_validator("command")
    @classmethod
    def _validate_command(cls, command: list[str]) -> list[str]:
        """Reject shell-shaped argv values before registry use."""

        executable = command[0].strip().lower()
        executable_name = executable.replace("\\", "/").split("/")[-1]
        if executable_name in DENIED_COMMAND_EXECUTABLES:
            raise ValueError("command executable must not be a shell interpreter")

        for part in command:
            if not part or not part.strip():
                raise ValueError("command entries must be non-empty argv parts")
            normalized_part = part.strip().lower()
            if normalized_part in DENIED_INLINE_EXECUTION_FLAGS:
                raise ValueError("command entries must not run inline code")
            if part.strip() in SHELL_META_CHARS:
                raise ValueError("command entries must not be shell operators")
            if _looks_like_shell_command(part):
                raise ValueError("command must be an argv list, not a shell string")
        return command

    @field_validator("env")
    @classmethod
    def _validate_env(cls, env: dict[str, str]) -> dict[str, str]:
        """Keep registry environment overlays bounded and name-only in APIs."""

        for key, value in env.items():
            if not re.fullmatch(r"[A-Z0-9_]{1,80}", key):
                raise ValueError("environment keys must be uppercase names")
            if len(value) > 2000:
                raise ValueError("environment values are too large")
        return env


class ServiceRuntimeState(StrictModel):
    """Current desired and observed service state."""

    id: str = Field(pattern=SERVICE_ID_PATTERN)
    display_name: str = Field(min_length=1, max_length=80)
    kind: Literal["backend", "frontend", "adapter", "worker", "support"]
    desired_state: Literal["running", "stopped"] = "stopped"
    actual_state: Literal[
        "unknown",
        "stopped",
        "starting",
        "running",
        "stopping",
        "unhealthy",
        "crashed",
        "conflict",
        "unavailable",
    ] = "stopped"
    pid: int | None = None
    generation: str | None = None
    command_fingerprint: str | None = None
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    uptime_seconds: float | None = None
    exit_code: int | None = None
    restart_count: int = 0
    health: dict[str, Any] = Field(default_factory=dict)
    dependencies: list[str] = Field(default_factory=list)
    last_event_at: datetime | None = None
    last_error_preview: str | None = None
    version: int = 0


class ConsoleCognitionObservationView(BaseModel):
    """Console availability metadata around one Brain observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    view_kind: Literal["overview_latest", "debug_latest", "self_latest"]
    availability: Literal[
        "available",
        "not_reported",
        "unavailable",
        "invalid",
    ]
    reason_code: str = Field(
        max_length=80,
        pattern=COGNITION_VIEW_REASON_PATTERN,
    )
    generated_at: datetime
    observation: CognitionRunObservationV1 | None

    @field_validator("generated_at")
    @classmethod
    def _normalize_generated_at(cls, value: datetime) -> datetime:
        """Require an aware console timestamp normalized to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generated_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_serializer("generated_at")
    def _serialize_generated_at(self, value: datetime) -> str:
        """Serialize console timestamps with the canonical terminal Z."""

        return value.astimezone(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        )

    @model_validator(mode="after")
    def _validate_availability(self) -> ConsoleCognitionObservationView:
        """Keep availability and observation payloads mutually consistent."""

        if self.availability == "available":
            if not self.observation or self.reason_code:
                raise ValueError(
                    "available observation views require an observation and no reason"
                )
            expected_kind = (
                "self_cognition"
                if self.view_kind == "self_latest"
                else "live_turn"
            )
            if self.observation.run_kind != expected_kind:
                raise ValueError("observation run kind does not match view kind")
        elif self.observation is not None or not self.reason_code:
            raise ValueError(
                "non-available observation views require a reason and no observation"
            )
        return self


class ServiceActionRequest(StrictModel):
    """Operator request for one state-changing service action."""

    reason: str = Field(min_length=1, max_length=240)
    expected_version: int | None = None


class ServiceActionResponse(StrictModel):
    """Accepted lifecycle action result."""

    request_id: str
    service: ServiceRuntimeState | dict[str, Any]
    action: Literal["start", "stop", "restart"]
    accepted_at: datetime
    audit_event_id: str


class ServiceConfigApplyRequest(StrictModel):
    """Operator request to apply descriptor-backed service config values."""

    reason: str = Field(min_length=1, max_length=240)
    expected_version: int | None = None
    values: dict[str, Any] = Field(default_factory=dict, max_length=96)


class ServiceConfigResetRequest(StrictModel):
    """Operator request to clear a process-local service config override."""

    reason: str = Field(min_length=1, max_length=240)
    expected_version: int | None = None


class ServiceConfigRestartResult(StrictModel):
    """Restart result for a config apply or reset operation."""

    attempted: bool
    succeeded: bool | None = None
    reason: str


class ServiceConfigActionResponse(StrictModel):
    """Generic config apply/reset response."""

    service_id: str
    config: dict[str, Any]
    service: ServiceRuntimeState | dict[str, Any]
    restart: ServiceConfigRestartResult
    audit_event_id: str


class BrainModelRouteValues(StrictModel):
    """Editable values for one Brain model route."""

    model: str | None = Field(default=None, max_length=200)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=65536)
    thinking_enabled: bool | None = None

    @field_validator("model")
    @classmethod
    def _validate_model(cls, model: str | None) -> str | None:
        """Reject empty model IDs before descriptor validation."""

        if model is None:
            return None
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("model must not be empty")
        return normalized_model


class BrainModelRouteApplyRequest(StrictModel):
    """Operator request to apply selected Brain route values."""

    reason: str = Field(min_length=1, max_length=240)
    expected_version: int | None = None
    values: BrainModelRouteValues


class BrainModelRouteResetRequest(StrictModel):
    """Operator request to clear one Brain route override."""

    reason: str = Field(min_length=1, max_length=240)
    expected_version: int | None = None


class BrainModelRouteActionResponse(StrictModel):
    """Selected-route apply/reset response."""

    service_id: str
    route: dict[str, Any]
    routes: list[dict[str, Any]]
    config: dict[str, Any]
    service: ServiceRuntimeState | dict[str, Any]
    restart: ServiceConfigRestartResult
    audit_event_id: str


class ControlConsoleOperator(StrictModel):
    """Authenticated local operator identity."""

    operator_id: str
    authenticated_at: datetime


class ControlAuditEvent(StrictModel):
    """Sanitized privileged action audit event."""

    event_id: str
    event_type: str
    operator_id: str
    service_id: str = ""
    target: dict[str, Any] = Field(default_factory=dict)
    previous_state: dict[str, Any] | None = None
    new_state: dict[str, Any] | None = None
    reason: str = ""
    created_at: datetime
    request_id: str
    source: Literal["local_control_console"] = "local_control_console"


class ProcessLogQuery(StrictModel):
    """Bounded process-log query."""

    service_id: str = Field(pattern=SERVICE_ID_PATTERN)
    level: str | None = Field(default=None, max_length=40)
    since: datetime | None = None
    limit: int = Field(default=100, ge=1, le=500)
    cursor: str | None = Field(default=None, max_length=120)


class ProcessLogLine(StrictModel):
    """One redacted process log line."""

    service_id: str
    stream: Literal["stdout", "stderr", "supervisor"]
    line: str
    created_at: datetime
    cursor: str


class OperationalEventQuery(StrictModel):
    """Bounded application operational-event query."""

    source: Literal["all", "kazusa"] = "all"
    service_id: str | None = Field(default=None, max_length=80)
    event_type: str | None = Field(default=None, max_length=80)
    level: str | None = Field(default=None, max_length=40)
    request_id: str | None = Field(default=None, max_length=120)
    tracking_id: str | None = Field(default=None, max_length=120)
    since: datetime | None = None
    limit: int = Field(default=100, ge=1, le=200)
    cursor: str | None = Field(default=None, max_length=120)


class OperationalEventPage(StrictModel):
    """Structured event-monitor page with dynamic result facets."""

    generated_at: datetime
    items: list[dict[str, Any]]
    facets: dict[str, dict[str, int]] = Field(default_factory=dict)
    query: dict[str, Any] = Field(default_factory=dict)
    next_cursor: str | None = None


class ConsoleDebugChatRequest(StrictModel):
    """Debug chat request forwarded to the brain service when available."""

    platform: Literal["debug"] = "debug"
    channel_id: str = Field(min_length=1, max_length=120)
    user_id: str = Field(min_length=1, max_length=120)
    user_display_name: str = Field(min_length=1, max_length=120)
    message_text: str = Field(min_length=1, max_length=4000)
    debug_modes: list[Literal["listen_only", "think_only", "no_remember"]] = (
        Field(default_factory=list, max_length=3)
    )
    envelope_overrides: dict[str, Any] = Field(default_factory=dict)


class ConsoleDebugChatResponse(StrictModel):
    """Debug chat response or safe unavailable result."""

    request_id: str
    brain_available: bool
    request: dict[str, Any]
    response: dict[str, Any] | None
    tracking_id: str | None
    trace_id: str = ""
    delivery_tracking_id: str | None = None
    llm_trace_id: str = ""
    latency_ms: int | None
    sent_at: datetime
    error: dict[str, Any] | None = None
    cognition_observation: ConsoleCognitionObservationView


class ConsoleLookupQuery(StrictModel):
    """Common bounded lookup query."""

    query: str = Field(default="", max_length=240)
    platform: str | None = Field(default=None, max_length=80)
    platform_user_id: str | None = Field(default=None, max_length=120)
    platform_channel_id: str | None = Field(default=None, max_length=120)
    channel_type: str | None = Field(default=None, max_length=40)
    participant_platform_user_id: str | None = Field(default=None, max_length=120)
    group_id: str | None = Field(default=None, max_length=120)
    episode_id: str | None = Field(default=None, max_length=120)
    status: str | None = Field(default=None, max_length=80)
    since: datetime | None = None
    until: datetime | None = None
    limit: int = Field(default=25, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=120)


class ConsoleLookupPage(StrictModel):
    """Bounded redacted lookup page."""

    generated_at: datetime
    items: list[dict[str, Any]]
    next_cursor: str | None
    redaction: dict[str, Any]


class ControlConsoleBootstrapResponse(StrictModel):
    """Full UI bootstrap snapshot."""

    generated_at: datetime
    operator: dict[str, Any]
    csrf_header_name: str
    csrf_token: str
    application_identity: dict[str, Any]
    services: list[ServiceRuntimeState]
    overview: dict[str, Any]
    health: dict[str, Any]
    latest_cognition_observation: ConsoleCognitionObservationView
    latest_self_cognition_observation: ConsoleCognitionObservationView
    recent_audit_events: list[dict[str, Any]]
    event_counters: dict[str, int]
    ui_capabilities: dict[str, bool]
    page_capabilities: dict[str, dict[str, Any]]
    service_config_summaries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    stream_url: str = "/api/stream"


class LoginRequest(StrictModel):
    """Token-based local operator login request."""

    token: str = Field(min_length=1, max_length=2048)


def _looks_like_shell_command(command_part: str) -> bool:
    """Return whether one argv entry appears to contain shell parsing."""

    stripped_part = command_part.strip()
    has_space = " " in stripped_part or "\t" in stripped_part
    has_meta = any(marker in stripped_part for marker in SHELL_META_CHARS)
    looks_like_command = has_space and (
        stripped_part.startswith(("python ", "uvicorn ", "cmd ", "powershell "))
        or has_meta
    )
    return looks_like_command
