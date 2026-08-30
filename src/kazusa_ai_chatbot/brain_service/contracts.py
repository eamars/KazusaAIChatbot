"""Pydantic contracts for the Kazusa brain service API."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)

from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
    CognitionRunObservationV1,
)
from kazusa_ai_chatbot.dsh_interaction.contracts import (
    DshBrainInteractionRequestV2 as CanonicalDshBrainInteractionRequestV2,
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
    cognition_graph: CognitionRunObservationV1 | None = None
    operational_error: OperationalErrorOut | None = None


class DshBrainInteractionRequestV2(BaseModel):
    """Versioned internal request accepted only from the DSH sidecar."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dsh_brain_interaction.v2"]
    interaction_id: str = Field(min_length=1)
    kind: Literal["approval", "question", "plan_review"]
    resolution_thread_id: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    activation_id: str = Field(min_length=1)
    lease_epoch: int = Field(ge=1)
    dsh_call_id: str = Field(min_length=1)
    tool_name: str | None
    operation_id: str = Field(min_length=1)
    operation_payload_digest: str = Field(min_length=1)
    arguments_digest: str = Field(min_length=1)
    transient_detail: str = Field(min_length=1, max_length=8_000)
    brain_conversation_ref: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    platform_channel_id: str = Field(min_length=1)
    global_user_id: str = Field(min_length=1)
    scope_fingerprint: str = Field(min_length=1)
    audience_fingerprint: str = Field(min_length=1)
    profile_version: str = Field(min_length=1)
    catalog_digest: str = Field(min_length=1)
    model_route_digest: str = Field(min_length=1)
    workspace_fingerprint: str = Field(min_length=1)
    policy_epoch: str = Field(min_length=1)
    issued_reference_digest: str = Field(min_length=1)
    nonce: str = Field(min_length=1)
    issued_at: str = Field(min_length=1)
    expires_at: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    mac: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interaction_identity(self) -> DshBrainInteractionRequestV2:
        """Enforce the closed transport shape before Brain admission."""

        self.to_canonical()
        if self.kind == "approval" and self.tool_name is None:
            raise ValueError("approval interaction requires tool_name")
        for field_name in (
            "operation_payload_digest",
            "arguments_digest",
            "scope_fingerprint",
            "audience_fingerprint",
            "catalog_digest",
            "model_route_digest",
            "workspace_fingerprint",
            "issued_reference_digest",
        ):
            value = getattr(self, field_name)
            if not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        return self

    def to_canonical(self) -> CanonicalDshBrainInteractionRequestV2:
        """Adapt the HTTP DTO through the single internal request contract."""

        return CanonicalDshBrainInteractionRequestV2.from_mapping(
            self.model_dump(mode="python"),
        )


class DshBrainInteractionResponseV2(BaseModel):
    """Versioned internal decision response returned to the DSH sidecar."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dsh_brain_interaction.v2"]
    interaction_id: str = Field(min_length=1)
    request_digest: str = Field(min_length=1)
    kind: Literal["approval", "question", "plan_review"]
    decision: Literal["answer", "allow_once", "reject"]
    reason: str = Field(min_length=1)
    answer: str | None = Field(default=None, max_length=2_000)
    grant: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_decision_payload(self) -> DshBrainInteractionResponseV2:
        """Enforce decision-specific fields on responses leaving Brain."""

        if self.decision == "answer" and self.answer is None:
            raise ValueError("answer is required for answer decision")
        if self.decision != "answer" and self.answer is not None:
            raise ValueError("answer is status-specific")
        if self.decision == "answer" and self.kind not in {"question", "plan_review"}:
            raise ValueError("answer is incompatible with interaction kind")
        if self.decision == "allow_once" and self.kind not in {
            "approval",
            "plan_review",
        }:
            raise ValueError("allow_once is incompatible with interaction kind")
        if self.decision == "allow_once" and self.grant is None:
            raise ValueError("allow_once response requires a deterministic grant")
        if self.decision != "allow_once" and self.grant is not None:
            raise ValueError("grant is status-specific")
        return self


class DshInteractionHealthResponseV1(BaseModel):
    """Readiness facts for the configured Brain interaction owner."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["dsh_brain_interaction_health.v1"]
    status: Literal["ready", "unavailable"]
    configured: bool
    durable_store: bool
    cognition_judge: bool
    task_resolution: DshTaskResolutionHealthV1


class DshTaskResolutionHealthV1(BaseModel):
    """Readiness identity for the shared Brain/DSH task edge."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ready", "unavailable"]
    sidecar_identity: str = Field(min_length=1)
    brain_bridge_identity: str = Field(min_length=1)


class OpsLatestCognitionGraphResponse(BaseModel):
    cognition_graph: CognitionRunObservationV1 | None = None
    self_cognition_graph: CognitionRunObservationV1 | None = None


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
