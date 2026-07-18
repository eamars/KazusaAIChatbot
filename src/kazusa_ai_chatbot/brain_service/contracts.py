"""Pydantic contracts for the Kazusa brain service API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    cognition_graph: dict[str, Any] | None = None
    operational_error: OperationalErrorOut | None = None


class OpsLatestCognitionGraphResponse(BaseModel):
    cognition_graph: dict[str, Any] | None = None


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
    platform: str
    callback_url: str

    shared_secret: str = ""
    timeout_seconds: float = 10.0
    platform_bot_id: str = ""
    display_name: str = ""


class RuntimeAdapterRegistrationResponse(BaseModel):
    status: str
    platform: str
    callback_url: str
