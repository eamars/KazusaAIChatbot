"""Pydantic contracts for the Kazusa brain service API."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictInt,
    model_validator,
)

from kazusa_ai_chatbot.message_envelope import MentionEntityKind


class AttachmentIn(BaseModel):
    media_type: str = ""
    url: str = ""
    base64_data: str = ""
    description: str = ""
    size_bytes: int | None = None


class DebugModesIn(BaseModel):
    listen_only: bool = False
    think_only: bool = False
    no_remember: bool = False


class MentionIn(BaseModel):
    platform_user_id: str = ""
    global_user_id: str = ""
    display_name: str = ""
    entity_kind: MentionEntityKind = "unknown"
    raw_text: str = ""


class ReplyTargetIn(BaseModel):
    platform_message_id: str = ""
    platform_user_id: str = ""
    global_user_id: str = ""
    display_name: str = ""
    excerpt: str = ""
    derivation: str = ""


class AttachmentRefIn(AttachmentIn):
    storage_shape: str = ""


class MessageEnvelopeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    body_text: str
    raw_wire_text: str
    mentions: list[MentionIn]
    reply: ReplyTargetIn | None = None
    attachments: list[AttachmentRefIn]
    addressed_to_global_user_ids: list[str]
    broadcast: bool


class ChatRequestReceiptMetadata(TypedDict, total=False):
    """Service-internal durable receipt identity attached before queue admission."""

    conversation_row_id: str
    received_at: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str
    platform_channel_id: str = ""
    channel_type: str = "group"
    platform_message_id: str = ""
    platform_user_id: str
    platform_bot_id: str = ""
    display_name: str = ""
    channel_name: str = ""
    content_type: str = "text"
    message_envelope: MessageEnvelopeIn
    local_timestamp: str = ""
    debug_modes: DebugModesIn = Field(default_factory=DebugModesIn)

    _receipt_metadata: ChatRequestReceiptMetadata | None = None
    _console_trace_authorized: bool = PrivateAttr(default=False)


class AttachmentOut(BaseModel):
    media_type: str = ""
    url: str = ""
    base64_data: str = ""
    description: str = ""
    size_bytes: int | None = None


class OperationalErrorOut(BaseModel):
    """Machine-readable metadata for a user-visible operational response."""

    error_code: str
    status: Literal["failed", "exhausted"]
    retryable: bool
    exhausted: bool
    attempt_count: int = Field(ge=1)
    correlation_id: str
    trace_id: str
    branch_id: str = ""
    stage: str = ""


class ChatResponse(BaseModel):
    messages: list[str] = Field(default_factory=list)
    content_type: str = "text"
    attachments: list[AttachmentOut] = Field(default_factory=list)
    use_reply_feature: bool = False
    delivery_mentions: list[dict[str, Any]] = Field(default_factory=list)
    scheduled_followups: int = 0
    delivery_tracking_id: str = ""
    trace_id: str = ""
    cognition_graph: dict[str, Any] | None = None
    operational_error: OperationalErrorOut | None = None


class OpsLatestCognitionGraphResponse(BaseModel):
    cognition_graph: dict[str, Any] | None = None
    self_cognition_graph: dict[str, Any] | None = None
    cognition_chain_run: dict[str, Any] | None = None
    self_cognition_chain_run: dict[str, Any] | None = None


class DeliveryReceiptRequest(BaseModel):
    platform: str = Field(min_length=1)
    platform_channel_id: str = ""
    delivery_tracking_id: str = Field(min_length=1)
    logical_message_index: int = Field(ge=0)
    platform_message_id: str = Field(min_length=1)
    delivered_at: str = ""
    adapter: str = ""


class DeliveryReceiptResponse(BaseModel):
    status: str
    updated: bool = False


class EventRequest(BaseModel):
    platform: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Cache2AgentStatsResponse(BaseModel):
    agent_name: str
    hit_count: int
    miss_count: int
    hit_rate: float


class Cache2HealthResponse(BaseModel):
    agents: list[Cache2AgentStatsResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    db: bool
    scheduler: bool
    cache2: Cache2HealthResponse = Field(default_factory=Cache2HealthResponse)


class OpsRuntimeConfigResponse(BaseModel):
    calendar_scheduler_enabled: bool
    calendar_scheduler_poll_interval_seconds: int
    calendar_scheduler_claim_limit: int
    calendar_scheduler_lease_seconds: int
    calendar_scheduler_max_attempts: int
    reflection_cycle_enabled: bool
    self_cognition_enabled: bool
    background_work_worker_enabled: bool
    reflection_worker_interval_seconds: int
    reflection_phase_min_slot_spacing_seconds: int
    reflection_phase_max_slots_per_period: int
    reflection_phase_groups_per_slot: int
    self_cognition_worker_interval_seconds: int
    self_cognition_max_cases_per_tick: int
    background_work_worker_interval_seconds: int
    background_work_worker_claim_limit: int
    background_work_worker_lease_seconds: int
    background_work_worker_max_attempts: int
    background_work_input_char_limit: int
    background_work_output_char_limit: int


class OpsProcessStatusResponse(BaseModel):
    last_event_at: str = ""
    last_status: str = ""


class OpsWorkerStatusResponse(BaseModel):
    enabled: bool = False
    task_alive: bool = False
    last_event_at: str = ""
    last_status: str = ""


class CognitionEngineDescriptorResponse(BaseModel):
    """Selected cognition-engine configuration safe for operator status."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["cognition_engine_descriptor.v1"]
    engine_id: Literal["v2", "v3"]
    chain_model_name: str = Field(min_length=1, max_length=256)
    sidecar_model_name: str = Field(max_length=256)
    sidecar_enabled: bool
    subconscious_enabled: bool
    appraisal_group_count: StrictInt = Field(ge=0, le=6)
    chain_context_window_tokens: StrictInt = Field(ge=0, le=1_000_000)
    normal_budget_tokens: StrictInt = Field(ge=0, le=65_000)
    extended_budget_tokens: StrictInt = Field(ge=0, le=65_000)
    turn_deadline_seconds: StrictInt = Field(ge=0, le=600)

    @model_validator(mode="after")
    def validate_policy(self) -> CognitionEngineDescriptorResponse:
        """Reject descriptor combinations that cannot describe an engine."""

        if self.appraisal_group_count not in {0, 1, 2, 3, 6}:
            raise ValueError(
                "appraisal_group_count must be one of 0, 1, 2, 3, or 6",
            )

        if self.engine_id == "v2":
            if self.sidecar_model_name:
                raise ValueError("V2 descriptor sidecar model must be empty")
            if self.sidecar_enabled:
                raise ValueError("V2 descriptor sidecar must be disabled")
            if self.subconscious_enabled:
                raise ValueError(
                    "V2 descriptor subconscious mode must be disabled",
                )
            if any(
                value != 0
                for value in (
                    self.appraisal_group_count,
                    self.chain_context_window_tokens,
                    self.normal_budget_tokens,
                    self.extended_budget_tokens,
                    self.turn_deadline_seconds,
                )
            ):
                raise ValueError(
                    "V2 descriptor chain-specific numeric fields must be zero",
                )
            return self

        if self.appraisal_group_count not in {1, 2, 3, 6}:
            raise ValueError(
                "V3 descriptor appraisal_group_count must be one of 1, 2, 3, or 6",
            )
        if self.chain_context_window_tokens < 50_000:
            raise ValueError(
                "V3 descriptor context window must be at least 50000",
            )
        if self.normal_budget_tokens != 50_000:
            raise ValueError(
                "V3 descriptor normal budget must equal 50000",
            )
        if self.extended_budget_tokens != 65_000:
            raise ValueError(
                "V3 descriptor extended budget must equal 65000",
            )
        if not 30 <= self.turn_deadline_seconds <= 600:
            raise ValueError(
                "V3 descriptor turn deadline must be between 30 and 600",
            )
        if self.sidecar_enabled != bool(self.sidecar_model_name):
            raise ValueError(
                "V3 descriptor sidecar must match its model name",
            )
        if self.subconscious_enabled and not self.sidecar_enabled:
            raise ValueError(
                "V3 descriptor subconscious mode requires a sidecar",
            )
        return self


class OpsRuntimeStatusResponse(BaseModel):
    status: str
    generated_at: str
    window_hours: int
    config: OpsRuntimeConfigResponse
    process: OpsProcessStatusResponse = Field(
        default_factory=OpsProcessStatusResponse,
    )
    workers: dict[str, OpsWorkerStatusResponse] = Field(default_factory=dict)
    semantic_descriptors: dict[str, str] = Field(default_factory=dict)
    cognition_engine: CognitionEngineDescriptorResponse | None = None


class OpsLatestEventRefResponse(BaseModel):
    event_id: str = ""
    run_id: str = ""
    trigger_id: str = ""
    attempt_id: str = ""
    occurred_at: str = ""
    status: str = ""


class OpsStatsResponse(BaseModel):
    status: str
    generated_at: str
    window_hours: int
    counts: dict[str, int] = Field(default_factory=dict)
    latest: OpsLatestEventRefResponse = Field(
        default_factory=OpsLatestEventRefResponse,
    )
    semantic_descriptors: dict[str, str] = Field(default_factory=dict)


class OpsSelfCognitionStatsResponse(OpsStatsResponse):
    enabled: bool = False
    task_alive: bool = False


class RuntimeAdapterRegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str
    callback_url: str
    platform_bot_id: str = Field(min_length=1)
    shared_secret: str = ""
    timeout_seconds: float = 10.0


class RuntimeAdapterRegistrationResponse(BaseModel):
    status: str
    platform: str
    callback_url: str
    character_name: str = Field(min_length=1)
