"""Source-neutral cognitive episode contracts and builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NoReturn,
    NotRequired,
    TypedDict,
    get_args,
)

from kazusa_ai_chatbot.time_boundary import (
    LocalTimeContextDoc,
    parse_storage_utc_datetime,
)

if TYPE_CHECKING:
    from kazusa_ai_chatbot.calendar_scheduler.models import CalendarRunDoc
    from kazusa_ai_chatbot.db.schemas import InternalActionLatchV1
    from kazusa_ai_chatbot.self_cognition.models import SelfCognitionCase

TriggerSource = Literal[
    "user_message",
    "internal_thought",
    "self_cognition",
    "scheduled_tick",
    "tool_result",
]

class MediaDescriptionRow(TypedDict):
    content_type: str
    description: str
    image_observation: NotRequired["ImageObservation"]


class ImageObservation(TypedDict):
    observation_origin: str
    source_message_id: str
    media_kind: str
    summary_status: Literal["available", "unavailable"]
    summary: str
    visible_text: list[str]
    salient_visual_facts: list[str]
    spatial_or_scene_facts: list[str]
    uncertainty: list[str]


class EvidenceRefV1(TypedDict, total=False):
    """Prompt-safe evidence reference attached to a canonical episode."""

    schema_version: Literal["evidence_ref.v1"]
    evidence_kind: str
    evidence_id: str
    owner: str
    excerpt: str | None
    observed_at: str | None


class PerceptV1(TypedDict, total=False):
    """Prompt-safe percept in the canonical episode envelope."""

    schema_version: Literal["percept.v1"]
    percept_kind: str
    source_kind: str
    source_id: str | None
    content: dict[str, object]
    observed_at: str


class TargetScopeV1(TypedDict, total=False):
    """Deterministic target and permission scope for one episode."""

    platform: str
    platform_channel_id: str
    channel_type: str
    current_platform_user_id: str
    current_global_user_id: str
    current_display_name: str
    target_addressed_user_ids: list[str]
    target_broadcast: bool
    permission_ref: str


class DebugControlsV1(TypedDict, total=False):
    """Deterministic controls accepted at the user-message boundary."""

    think_only: bool
    no_remember: bool
    no_visual_directives: bool


class UserMessageOriginV1(TypedDict, total=False):
    """Typed origin metadata for an adapter-normalized user message."""

    schema_version: Literal["user_message_origin.v1"]
    owner: str
    platform: str
    platform_message_id: str
    active_turn_platform_message_ids: list[str]
    active_turn_conversation_row_ids: list[str]
    debug_modes: dict[str, bool]
    correlation_id: str
    privacy_scope: str
    delivery_permission_ref: str
    created_at: str


class InternalThoughtOriginV1(TypedDict, total=False):
    """Typed origin metadata for a claimed internal action latch."""

    schema_version: Literal["internal_thought_origin.v1"]
    owner: str
    prior_episode_id: str
    action_latch_ref: str
    continuation_depth: int
    observed_context_refs: list[str]
    correlation_id: str
    privacy_scope: str
    delivery_permission_ref: str
    created_at: str


class SelfCognitionOriginV1(TypedDict, total=False):
    """Typed origin metadata for an observed self-cognition case."""

    schema_version: Literal["self_cognition_origin.v1"]
    owner: str
    source_case_kind: str
    source_case_ref: str
    observed_context_refs: list[str]
    target_scope_ref: str
    correlation_id: str
    privacy_scope: str
    delivery_permission_ref: str
    created_at: str


class ScheduledTickOriginV1(TypedDict, total=False):
    """Typed origin metadata for a claimed calendar run."""

    schema_version: Literal["scheduled_tick_origin.v1"]
    owner: str
    calendar_event_id: str
    claim_id: str
    scheduled_for_utc: str
    continuation_objective_ref: str
    target_scope_ref: str
    privacy_scope: str
    delivery_permission_ref: str
    created_at: str


class ToolResultOriginV1(TypedDict, total=False):
    """Typed origin metadata for a completed tool result."""

    schema_version: Literal["tool_result_origin.v1"]
    owner: str
    task_id: str
    task_kind: str
    result_ref: str
    completed_at: str
    target_scope_ref: str
    correlation_id: str
    privacy_scope: str
    delivery_permission_ref: str
    created_at: str


class ToolResultReadyV1(TypedDict, total=False):
    """Prompt-safe result projection ready to enter cognition."""

    schema_version: Literal["tool_result_ready.v1"]
    task_id: str
    task_kind: str
    semantic_summary: str
    artifact_text: str
    failure_text: str
    completed_at: str
    target_scope: TargetScopeV1
    evidence_refs: list[EvidenceRefV1]
    coding_run_context: dict[str, object]
    result_ref: str
    source_platform_bot_id: str
    source_character_name: str


class CognitiveEpisodeV1(TypedDict):
    """Authoritative five-source cognitive episode envelope."""

    schema_version: Literal["cognitive_episode.v1"]
    episode_id: str
    trigger_source: TriggerSource
    origin_metadata: dict[str, object]
    target_scope: dict[str, object]
    percepts: list[PerceptV1]
    evidence_refs: list[EvidenceRefV1]
    created_at: str
    privacy_scope: str
    continuation_depth: int


class TriggerSourceSpecV1(TypedDict):
    """Registry policy for one canonical episode source."""

    schema_version: Literal["trigger_source_spec.v1"]
    source_kind: str
    owner: str
    entrypoint: str
    llm_visibility: Literal["prompt_visible", "metadata_only", "hidden"]
    evidence_policy: Literal[
        "requires_evidence",
        "optional_evidence",
        "no_evidence",
    ]
    persistence_policy: Literal["ephemeral", "audited", "durable"]
    rate_limit_policy: dict[str, object]
    privacy_policy: dict[str, object]
    allowed_continuation_depth: int


class TextChatCompatibilityProjection(TypedDict):
    storage_timestamp_utc: str
    local_time_context: LocalTimeContextDoc
    user_input: str
    platform: str
    platform_channel_id: str
    channel_type: str
    platform_message_id: str
    active_turn_platform_message_ids: list[str]
    active_turn_conversation_row_ids: list[str]
    platform_user_id: str
    global_user_id: str
    user_name: str


ResponseOperationRole = Literal["self", "current_user", "other", "none"]


class DialogResponseOperation(TypedDict):
    """Model-owned response and embedded-action role ownership."""

    operation: str
    response_owner_role: ResponseOperationRole
    selection_owner_role: ResponseOperationRole
    selection_required: bool
    embedded_actor_role: ResponseOperationRole
    embedded_target_role: ResponseOperationRole


class CognitiveEpisodeValidationError(ValueError):
    """Raised when a cognitive episode is structurally invalid."""


_TRIGGER_SOURCES = frozenset(get_args(TriggerSource))
MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS = 4
MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS = 800
MAX_ROLE_EXPLICIT_CONTENT_CHARS = 1000
MAX_RESPONSE_OPERATION_CHARS = 500
ROLE_EXPLICIT_CONTENT_METADATA_KEY = "role_explicit_content"
RESPONSE_OPERATION_METADATA_KEY = "response_operation"
_RESPONSE_OPERATION_ROLES = frozenset({
    "self",
    "current_user",
    "other",
    "none",
})
_RESPONSE_OPERATION_FIELDS = frozenset(
    DialogResponseOperation.__annotations__
)
_IMAGE_OBSERVATION_LIST_FIELDS = (
    "visible_text",
    "salient_visual_facts",
    "spatial_or_scene_facts",
    "uncertainty",
)


def build_trigger_source_registry() -> dict[str, TriggerSourceSpecV1]:
    """Return the complete native registry for the five episode sources."""

    policy = {
        "schema_version": "policy_ref.v1",
        "policy_kind": "bounded_runtime",
    }
    registry: dict[str, TriggerSourceSpecV1] = {
        "user_message": {
            "schema_version": "trigger_source_spec.v1",
            "source_kind": "user_message",
            "owner": "brain_service.intake",
            "entrypoint": "service._process_queued_chat_item",
            "llm_visibility": "prompt_visible",
            "evidence_policy": "requires_evidence",
            "persistence_policy": "durable",
            "rate_limit_policy": dict(policy),
            "privacy_policy": dict(policy),
            "allowed_continuation_depth": 1,
        },
        "internal_thought": {
            "schema_version": "trigger_source_spec.v1",
            "source_kind": "internal_thought",
            "owner": "db.internal_action_latches",
            "entrypoint": (
                "self_cognition.runner.build_self_cognition_case_artifacts_async"
            ),
            "llm_visibility": "metadata_only",
            "evidence_policy": "requires_evidence",
            "persistence_policy": "durable",
            "rate_limit_policy": dict(policy),
            "privacy_policy": dict(policy),
            "allowed_continuation_depth": 1,
        },
        "self_cognition": {
            "schema_version": "trigger_source_spec.v1",
            "source_kind": "self_cognition",
            "owner": "self_cognition.sources",
            "entrypoint": (
                "self_cognition.runner.build_self_cognition_case_artifacts_async"
            ),
            "llm_visibility": "metadata_only",
            "evidence_policy": "requires_evidence",
            "persistence_policy": "durable",
            "rate_limit_policy": dict(policy),
            "privacy_policy": dict(policy),
            "allowed_continuation_depth": 1,
        },
        "scheduled_tick": {
            "schema_version": "trigger_source_spec.v1",
            "source_kind": "scheduled_tick",
            "owner": "calendar_scheduler.worker",
            "entrypoint": (
                "self_cognition.runner.build_self_cognition_case_artifacts_async"
            ),
            "llm_visibility": "metadata_only",
            "evidence_policy": "requires_evidence",
            "persistence_policy": "durable",
            "rate_limit_policy": dict(policy),
            "privacy_policy": dict(policy),
            "allowed_continuation_depth": 1,
        },
        "tool_result": {
            "schema_version": "trigger_source_spec.v1",
            "source_kind": "tool_result",
            "owner": "background_work.delivery",
            "entrypoint": "background_work.result_source.build_tool_result_episode",
            "llm_visibility": "prompt_visible",
            "evidence_policy": "requires_evidence",
            "persistence_policy": "audited",
            "rate_limit_policy": dict(policy),
            "privacy_policy": dict(policy),
            "allowed_continuation_depth": 1,
        },
    }
    return registry


def _canonical_percept(
    percept: PerceptV1,
    *,
    created_at: str,
) -> PerceptV1:
    """Validate one canonical prompt-safe percept."""

    required_fields = (
        "schema_version",
        "percept_kind",
        "source_kind",
        "source_id",
        "content",
        "observed_at",
    )
    if any(field_name not in percept for field_name in required_fields):
        raise CognitiveEpisodeValidationError(
            "canonical percept fields are incomplete"
        )
    if percept["schema_version"] != "percept.v1":
        raise CognitiveEpisodeValidationError(
            "unsupported percept schema version"
        )
    if not isinstance(percept["percept_kind"], str) or not percept[
        "percept_kind"
    ].strip():
        raise CognitiveEpisodeValidationError("percept_kind must be non-empty")
    if not isinstance(percept["source_kind"], str) or not percept[
        "source_kind"
    ].strip():
        raise CognitiveEpisodeValidationError("source_kind must be non-empty")
    if not isinstance(percept["content"], dict):
        raise CognitiveEpisodeValidationError("percept content must be an object")
    if not isinstance(percept["observed_at"], str) or not percept[
        "observed_at"
    ].strip():
        raise CognitiveEpisodeValidationError("percept observed_at is required")
    return dict(percept)


def _canonical_percepts(
    percepts: Sequence[PerceptV1],
    *,
    created_at: str,
) -> list[PerceptV1]:
    """Validate and copy a bounded sequence of canonical percepts."""

    normalized = [
        _canonical_percept(percept, created_at=created_at)
        for percept in percepts
    ]
    if not normalized:
        raise CognitiveEpisodeValidationError("episode requires a percept")
    return normalized


def _time_percept(
    local_time_context: LocalTimeContextDoc,
    *,
    created_at: str,
) -> PerceptV1:
    """Build the bounded prompt-safe local-time percept."""

    return {
        "schema_version": "percept.v1",
        "percept_kind": "local_time_context",
        "source_kind": "system_event",
        "source_id": None,
        "content": {"local_time_context": dict(local_time_context)},
        "observed_at": created_at,
    }


def _canonical_episode(
    *,
    episode_id: str,
    trigger_source: TriggerSource,
    origin_metadata: Mapping[str, object],
    target_scope: Mapping[str, object],
    percepts: Sequence[PerceptV1],
    evidence_refs: Sequence[EvidenceRefV1],
    created_at: str,
    privacy_scope: str,
    continuation_depth: int,
) -> CognitiveEpisodeV1:
    """Build and validate one canonical episode envelope."""

    registry = build_trigger_source_registry()
    source_spec = registry.get(trigger_source)
    if source_spec is None:
        raise CognitiveEpisodeValidationError(
            f"episode source is not registered: {trigger_source}"
        )
    if continuation_depth < 0 or continuation_depth > source_spec[
        "allowed_continuation_depth"
    ]:
        raise CognitiveEpisodeValidationError(
            "episode continuation depth exceeds source policy"
        )
    if not episode_id or not created_at or not privacy_scope:
        raise CognitiveEpisodeValidationError(
            "episode identity, created_at, and privacy_scope are required"
        )
    normalized_evidence = [dict(evidence_ref) for evidence_ref in evidence_refs]
    for evidence_ref in normalized_evidence:
        if evidence_ref.get("schema_version") != "evidence_ref.v1":
            raise CognitiveEpisodeValidationError(
                "unsupported evidence reference schema version"
            )
    normalized_percepts = _canonical_percepts(
        percepts,
        created_at=created_at,
    )
    episode: CognitiveEpisodeV1 = {
        "schema_version": "cognitive_episode.v1",
        "episode_id": episode_id,
        "trigger_source": trigger_source,
        "origin_metadata": dict(origin_metadata),
        "target_scope": dict(target_scope),
        "percepts": normalized_percepts,
        "evidence_refs": normalized_evidence,
        "created_at": created_at,
        "privacy_scope": privacy_scope,
        "continuation_depth": continuation_depth,
    }
    return validate_cognitive_episode_v1(episode)


def validate_cognitive_episode_v1(episode: object) -> CognitiveEpisodeV1:
    """Validate and return the exact canonical episode envelope."""

    if not isinstance(episode, Mapping):
        raise CognitiveEpisodeValidationError(
            "canonical cognitive episode must be a mapping"
        )
    required_fields = {
        "schema_version",
        "episode_id",
        "trigger_source",
        "origin_metadata",
        "target_scope",
        "percepts",
        "evidence_refs",
        "created_at",
        "privacy_scope",
        "continuation_depth",
    }
    if set(episode) != required_fields:
        raise CognitiveEpisodeValidationError(
            "canonical cognitive episode fields are not exact"
        )
    if episode["schema_version"] != "cognitive_episode.v1":
        raise CognitiveEpisodeValidationError(
            "unsupported cognitive episode schema version"
        )
    trigger_source = episode["trigger_source"]
    registry = build_trigger_source_registry()
    if trigger_source not in registry:
        raise CognitiveEpisodeValidationError(
            "cognitive episode trigger source is not registered"
        )
    if not isinstance(episode["episode_id"], str) or not episode[
        "episode_id"
    ].strip():
        raise CognitiveEpisodeValidationError("episode_id must be non-empty")
    if not isinstance(episode["origin_metadata"], Mapping):
        raise CognitiveEpisodeValidationError("origin_metadata must be an object")
    if not isinstance(episode["target_scope"], Mapping):
        raise CognitiveEpisodeValidationError("target_scope must be an object")
    raw_percepts = episode["percepts"]
    if not isinstance(raw_percepts, list) or not raw_percepts:
        raise CognitiveEpisodeValidationError("percepts must be non-empty")
    for percept in raw_percepts:
        if not isinstance(percept, Mapping):
            raise CognitiveEpisodeValidationError("percept must be an object")
        _canonical_percept(dict(percept), created_at=str(episode["created_at"]))
    raw_evidence = episode["evidence_refs"]
    if not isinstance(raw_evidence, list):
        raise CognitiveEpisodeValidationError("evidence_refs must be a list")
    for evidence_ref in raw_evidence:
        if not isinstance(evidence_ref, Mapping):
            raise CognitiveEpisodeValidationError(
                "evidence reference must be an object"
            )
        if evidence_ref.get("schema_version") != "evidence_ref.v1":
            raise CognitiveEpisodeValidationError(
                "unsupported evidence reference schema version"
            )
    created_at = episode["created_at"]
    privacy_scope = episode["privacy_scope"]
    continuation_depth = episode["continuation_depth"]
    if not isinstance(created_at, str) or not created_at.strip():
        raise CognitiveEpisodeValidationError("created_at must be non-empty")
    if not isinstance(privacy_scope, str) or not privacy_scope.strip():
        raise CognitiveEpisodeValidationError("privacy_scope must be non-empty")
    if not isinstance(continuation_depth, int) or continuation_depth < 0:
        raise CognitiveEpisodeValidationError(
            "continuation_depth must be a non-negative integer"
        )
    if continuation_depth > registry[str(trigger_source)][
        "allowed_continuation_depth"
    ]:
        raise CognitiveEpisodeValidationError(
            "continuation_depth exceeds trigger source policy"
        )
    return dict(episode)  # type: ignore[return-value]


def _origin_with_defaults(
    origin: Mapping[str, object],
    *,
    schema_version: str,
    owner: str,
    created_at: str,
) -> dict[str, object]:
    """Copy origin metadata and fill deterministic common fields."""

    result = dict(origin)
    result.setdefault("schema_version", schema_version)
    result.setdefault("owner", owner)
    result.setdefault("created_at", created_at)
    result.setdefault("privacy_scope", "private")
    result.setdefault("delivery_permission_ref", "")
    return result


def build_user_message_episode(
    *,
    episode_id: str,
    origin: UserMessageOriginV1,
    target_scope: TargetScopeV1,
    dialog_percept: PerceptV1,
    media_percepts: Sequence[PerceptV1],
    evidence_refs: Sequence[EvidenceRefV1],
    local_time_context: LocalTimeContextDoc,
    created_at: str,
    debug_controls: DebugControlsV1,
) -> CognitiveEpisodeV1:
    """Build the canonical user-message episode."""

    origin_metadata = _origin_with_defaults(
        origin,
        schema_version="user_message_origin.v1",
        owner="brain_service.intake",
        created_at=created_at,
    )
    origin_metadata["debug_controls"] = dict(debug_controls)
    bounded_media_percepts = list(media_percepts)[
        :MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS
    ]
    percepts = [dialog_percept, *bounded_media_percepts]
    percepts.append(_time_percept(local_time_context, created_at=created_at))
    return _canonical_episode(
        episode_id=episode_id,
        trigger_source="user_message",
        origin_metadata=origin_metadata,
        target_scope=target_scope,
        percepts=percepts,
        evidence_refs=evidence_refs,
        created_at=created_at,
        privacy_scope=str(origin_metadata["privacy_scope"]),
        continuation_depth=0,
    )


def build_internal_thought_episode(
    *,
    latch: InternalActionLatchV1,
    evidence_refs: Sequence[EvidenceRefV1],
    local_time_context: LocalTimeContextDoc,
    created_at: str,
    claim_token: str,
) -> CognitiveEpisodeV1:
    """Build an internal-thought episode from one active claimed latch."""

    if latch.get("status") != "claimed":
        raise CognitiveEpisodeValidationError(
            "internal-thought episode requires a claimed latch"
        )
    if latch.get("claim_token") != claim_token or not claim_token:
        raise CognitiveEpisodeValidationError(
            "internal-thought episode claim token is invalid"
        )
    try:
        if parse_storage_utc_datetime(str(latch.get("expires_at", ""))) <= (
            parse_storage_utc_datetime(created_at)
        ):
            raise CognitiveEpisodeValidationError(
                "internal-thought episode latch is expired"
            )
        if parse_storage_utc_datetime(
            str(latch.get("claim_expires_at", ""))
        ) <= parse_storage_utc_datetime(created_at):
            raise CognitiveEpisodeValidationError(
                "internal-thought episode latch claim is expired"
            )
    except ValueError as exc:
        raise CognitiveEpisodeValidationError(
            "internal-thought episode latch timestamps are invalid"
        ) from exc
    attempt_count = latch.get("attempt_count")
    max_attempts = latch.get("max_attempts")
    if not isinstance(attempt_count, int) or not isinstance(max_attempts, int):
        raise CognitiveEpisodeValidationError(
            "internal-thought episode latch attempts are invalid"
        )
    if attempt_count > max_attempts:
        raise CognitiveEpisodeValidationError(
            "internal-thought episode latch attempts are exhausted"
        )
    continuation_depth = int(latch.get("continuation_depth", -1))
    if continuation_depth > 1:
        raise CognitiveEpisodeValidationError(
            "internal-thought episode continuation depth exceeds one"
        )
    percept: PerceptV1 = {
        "schema_version": "percept.v1",
        "percept_kind": "continuation_objective",
        "source_kind": "internal_thought",
        "source_id": str(latch.get("latch_id", "")),
        "content": {
            "objective": str(latch.get("continuation_objective", "")),
        },
        "observed_at": created_at,
    }
    origin = _origin_with_defaults(
        {
            "prior_episode_id": latch.get("source_episode_id", ""),
            "action_latch_ref": latch.get("latch_id", ""),
            "continuation_depth": continuation_depth,
            "observed_context_refs": [
                str(ref.get("evidence_id", "")) for ref in evidence_refs
            ],
            "correlation_id": latch.get("source_action_attempt_id", ""),
        },
        schema_version="internal_thought_origin.v1",
        owner="db.internal_action_latches",
        created_at=created_at,
    )
    return _canonical_episode(
        episode_id=f"internal-thought:{latch['latch_id']}",
        trigger_source="internal_thought",
        origin_metadata=origin,
        target_scope=latch.get("target_scope", {}),
        percepts=[percept, _time_percept(local_time_context, created_at=created_at)],
        evidence_refs=evidence_refs,
        created_at=created_at,
        privacy_scope=str(latch.get("privacy_scope", "private")),
        continuation_depth=continuation_depth,
    )


def build_self_cognition_episode(
    *,
    case: SelfCognitionCase,
    percepts: Sequence[PerceptV1],
    evidence_refs: Sequence[EvidenceRefV1],
    local_time_context: LocalTimeContextDoc,
    created_at: str,
) -> CognitiveEpisodeV1:
    """Build the canonical ordinary self-cognition episode."""

    case_data = dict(case)
    source_ref = str(
        case_data.get("case_id")
        or case_data.get("source_id")
        or case_data.get("id")
        or ""
    )
    origin = _origin_with_defaults(
        {
            "source_case_kind": str(
                case_data.get("source_case_kind")
                or case_data.get("trigger_kind")
                or "self_cognition"
            ),
            "source_case_ref": source_ref,
            "observed_context_refs": [
                str(ref.get("evidence_id", "")) for ref in evidence_refs
            ],
            "target_scope_ref": str(case_data.get("scope_ref", "")),
            "correlation_id": str(case_data.get("correlation_id", source_ref)),
        },
        schema_version="self_cognition_origin.v1",
        owner="self_cognition.sources",
        created_at=created_at,
    )
    return _canonical_episode(
        episode_id=f"self-cognition:{source_ref or created_at}",
        trigger_source="self_cognition",
        origin_metadata=origin,
        target_scope=case_data.get("target_scope", {}),
        percepts=[*percepts, _time_percept(local_time_context, created_at=created_at)],
        evidence_refs=evidence_refs,
        created_at=created_at,
        privacy_scope=str(case_data.get("privacy_scope", "private")),
        continuation_depth=int(case_data.get("continuation_depth", 0)),
    )


def build_scheduled_tick_episode(
    *,
    case: SelfCognitionCase,
    calendar_run: CalendarRunDoc,
    percepts: Sequence[PerceptV1],
    evidence_refs: Sequence[EvidenceRefV1],
    local_time_context: LocalTimeContextDoc,
    created_at: str,
) -> CognitiveEpisodeV1:
    """Build the canonical claimed-calendar scheduled episode."""

    case_data = dict(case)
    run_data = dict(calendar_run)
    run_id = str(run_data.get("run_id") or run_data.get("id") or "")
    origin = _origin_with_defaults(
        {
            "calendar_event_id": str(run_data.get("schedule_id", "")),
            "claim_id": run_id,
            "scheduled_for_utc": str(
                run_data.get("due_at") or run_data.get("scheduled_for", "")
            ),
            "continuation_objective_ref": str(
                case_data.get("case_id") or case_data.get("source_id") or ""
            ),
            "target_scope_ref": str(case_data.get("scope_ref", "")),
        },
        schema_version="scheduled_tick_origin.v1",
        owner="calendar_scheduler.worker",
        created_at=created_at,
    )
    return _canonical_episode(
        episode_id=f"scheduled-tick:{run_id or created_at}",
        trigger_source="scheduled_tick",
        origin_metadata=origin,
        target_scope=case_data.get("target_scope", {}),
        percepts=[*percepts, _time_percept(local_time_context, created_at=created_at)],
        evidence_refs=evidence_refs,
        created_at=created_at,
        privacy_scope=str(case_data.get("privacy_scope", "private")),
        continuation_depth=int(case_data.get("continuation_depth", 0)),
    )


def build_tool_result_episode(
    *,
    result: ToolResultReadyV1,
    evidence_refs: Sequence[EvidenceRefV1],
    local_time_context: LocalTimeContextDoc,
    created_at: str,
) -> CognitiveEpisodeV1:
    """Build the canonical tool-result episode."""

    result_data = dict(result)
    task_id = str(result_data.get("task_id", ""))
    semantic_summary = str(result_data.get("semantic_summary", ""))
    artifact_text = str(result_data.get("artifact_text", ""))
    failure_text = str(result_data.get("failure_text", ""))
    if not task_id or not semantic_summary:
        raise CognitiveEpisodeValidationError(
            "tool-result episode requires task_id and semantic_summary"
        )
    origin = _origin_with_defaults(
        {
            "task_id": task_id,
            "task_kind": str(result_data.get("task_kind", "")),
            "result_ref": str(result_data.get("result_ref", task_id)),
            "completed_at": str(result_data.get("completed_at", created_at)),
            "target_scope_ref": task_id,
            "correlation_id": task_id,
            "source_platform_bot_id": str(
                result_data.get("source_platform_bot_id", "")
            ),
            "source_character_name": str(
                result_data.get("source_character_name", "")
            ),
        },
        schema_version="tool_result_origin.v1",
        owner="background_work.delivery",
        created_at=created_at,
    )
    percept: PerceptV1 = {
        "schema_version": "percept.v1",
        "percept_kind": "tool_result",
        "source_kind": "tool_result",
        "source_id": task_id,
        "content": {
            "semantic_summary": semantic_summary,
            "artifact_text": artifact_text,
            "failure_text": failure_text,
        },
        "observed_at": str(result_data.get("completed_at", created_at)),
    }
    return _canonical_episode(
        episode_id=f"tool-result:{task_id}",
        trigger_source="tool_result",
        origin_metadata=origin,
        target_scope=result_data.get("target_scope", {}),
        percepts=[percept, _time_percept(local_time_context, created_at=created_at)],
        evidence_refs=evidence_refs,
        created_at=created_at,
        privacy_scope="conversation",
        continuation_depth=0,
    )


def project_model_visible_percepts(
    episode: CognitiveEpisodeV1,
) -> list[dict[str, Any]]:
    """Project visible percepts with typed dialogue-role provenance."""

    validate_cognitive_episode_v1(episode)
    return _project_canonical_model_visible_percepts(episode)


def _project_canonical_model_visible_percepts(
    episode: CognitiveEpisodeV1,
) -> list[dict[str, Any]]:
    """Project canonical percept content without exposing deterministic ids."""

    validate_cognitive_episode_v1(episode)
    rows: list[dict[str, Any]] = []
    for percept in episode["percepts"]:
        content = percept["content"]
        if percept["percept_kind"] == "local_time_context":
            rows.append({
                "input_source": "local_time_context",
                "content": content,
            })
            continue
        row: dict[str, Any] = {
            "input_source": percept["source_kind"],
            "content": content,
        }
        if percept["source_kind"] == "dialog":
            row.update({
                "speaker_role": "current_user",
                "addressee_role": "self",
                "first_person_role": "current_user",
                "implicit_imperative_subject_role": "self",
            })
        rows.append(row)
    return rows


def attach_dialog_semantic_projection(
    episode: CognitiveEpisodeV1,
    role_explicit_content: str,
    response_operation: object | None,
) -> CognitiveEpisodeV1:
    """Attach one model-owned semantic projection to the dialog percept."""

    return _attach_canonical_dialog_semantic_projection(
        episode,
        role_explicit_content,
        response_operation,
    )


def _attach_canonical_dialog_semantic_projection(
    episode: CognitiveEpisodeV1,
    role_explicit_content: str,
    response_operation: object | None,
) -> CognitiveEpisodeV1:
    """Attach model-owned dialog meaning to a canonical dialog percept."""

    validate_cognitive_episode_v1(episode)
    normalized_content = _validate_role_explicit_content(role_explicit_content)
    normalized_operation = (
        validate_dialog_response_operation(response_operation)
        if response_operation is not None
        else None
    )
    updated_episode = deepcopy(episode)
    for percept in updated_episode["percepts"]:
        if percept["source_kind"] != "dialog":
            continue
        content = percept["content"]
        content[ROLE_EXPLICIT_CONTENT_METADATA_KEY] = normalized_content
        if normalized_operation is not None:
            content[RESPONSE_OPERATION_METADATA_KEY] = normalized_operation
        else:
            content.pop(RESPONSE_OPERATION_METADATA_KEY, None)
        return validate_cognitive_episode_v1(updated_episode)
    raise CognitiveEpisodeValidationError(
        "role-explicit content requires a canonical dialog percept"
    )


def has_model_visible_dialog_percept(episode: CognitiveEpisodeV1) -> bool:
    """Return whether an episode contains model-visible dialog input."""

    validate_cognitive_episode_v1(episode)
    return any(
        percept.get("source_kind") == "dialog"
        for percept in episode["percepts"]
    )


def project_dialog_role_explicit_content(
    episode: CognitiveEpisodeV1,
) -> str | None:
    """Return the bounded role projection from the current dialog percept."""

    validate_cognitive_episode_v1(episode)
    for percept in episode["percepts"]:
        if percept["source_kind"] == "dialog":
            return _role_explicit_content_from_metadata(percept["content"])
    return None


def project_dialog_response_operation(
    episode: CognitiveEpisodeV1,
) -> DialogResponseOperation | None:
    """Return the response-operation projection from the dialog percept."""

    validate_cognitive_episode_v1(episode)
    for percept in episode["percepts"]:
        if percept["source_kind"] == "dialog":
            return _response_operation_from_metadata(percept["content"])
    return None


def _role_explicit_content_from_metadata(
    metadata: Mapping[str, Any],
) -> str | None:
    """Validate and return optional role-explicit percept metadata."""

    if ROLE_EXPLICIT_CONTENT_METADATA_KEY not in metadata:
        return None
    return _validate_role_explicit_content(
        metadata[ROLE_EXPLICIT_CONTENT_METADATA_KEY],
    )


def _response_operation_from_metadata(
    metadata: Mapping[str, Any],
) -> DialogResponseOperation | None:
    """Validate and return optional response-operation metadata."""

    if RESPONSE_OPERATION_METADATA_KEY not in metadata:
        return None
    return validate_dialog_response_operation(
        metadata[RESPONSE_OPERATION_METADATA_KEY],
    )


def _validate_role_explicit_content(value: object) -> str:
    """Validate model-owned role meaning without interpreting its semantics."""

    if not isinstance(value, str):
        raise CognitiveEpisodeValidationError(
            "role-explicit content must be a string"
        )
    normalized_value = value.strip()
    if not normalized_value:
        raise CognitiveEpisodeValidationError(
            "role-explicit content must not be empty"
        )
    if len(normalized_value) > MAX_ROLE_EXPLICIT_CONTENT_CHARS:
        raise CognitiveEpisodeValidationError(
            "role-explicit content exceeds the prompt bound"
        )
    return normalized_value


def validate_dialog_response_operation(
    value: object,
) -> DialogResponseOperation:
    """Validate response-operation shape without interpreting its meaning."""

    if not isinstance(value, Mapping):
        raise CognitiveEpisodeValidationError(
            "response operation must be an object"
        )
    if set(value) != _RESPONSE_OPERATION_FIELDS:
        raise CognitiveEpisodeValidationError(
            "response operation fields are not exact"
        )
    operation = value["operation"]
    if not isinstance(operation, str) or not operation.strip():
        raise CognitiveEpisodeValidationError(
            "response operation text must not be empty"
        )
    normalized_operation = operation.strip()
    if len(normalized_operation) > MAX_RESPONSE_OPERATION_CHARS:
        raise CognitiveEpisodeValidationError(
            "response operation text exceeds the prompt bound"
        )
    selection_required = value["selection_required"]
    if not isinstance(selection_required, bool):
        raise CognitiveEpisodeValidationError(
            "response operation selection_required must be boolean"
        )
    role_fields = (
        "response_owner_role",
        "selection_owner_role",
        "embedded_actor_role",
        "embedded_target_role",
    )
    for field_name in role_fields:
        if value[field_name] not in _RESPONSE_OPERATION_ROLES:
            raise CognitiveEpisodeValidationError(
                f"response operation {field_name} is invalid"
            )
    if selection_required and value["selection_owner_role"] == "none":
        raise CognitiveEpisodeValidationError(
            "required response selection needs an owner"
        )
    return {
        "operation": normalized_operation,
        "response_owner_role": value["response_owner_role"],
        "selection_owner_role": value["selection_owner_role"],
        "selection_required": selection_required,
        "embedded_actor_role": value["embedded_actor_role"],
        "embedded_target_role": value["embedded_target_role"],
    }


def build_text_chat_media_description_rows(
    multimedia_input: list[Mapping[str, object]],
) -> list[MediaDescriptionRow]:
    """Project current media rows into prompt-safe episode description rows.

    Args:
        multimedia_input: Current-turn media rows that may include storage or
            adapter fields outside the episode contract.

    Returns:
        Rows containing only supported content types and stripped descriptions.
    """
    media_description_rows: list[MediaDescriptionRow] = []
    for item in multimedia_input:
        content_type = item.get("content_type")
        if not isinstance(content_type, str) or content_type == "":
            continue

        description = item.get("description")
        if not isinstance(description, str):
            continue

        if not (
            content_type.startswith("image/")
            or content_type.startswith("audio/")
        ):
            continue

        clean_description = _trim_media_description(description.strip())
        image_observation = _sanitize_image_observation(
            item.get("image_observation"),
            description=clean_description,
            content_type=content_type,
        )
        if clean_description == "" and image_observation is None:
            continue

        row: MediaDescriptionRow = {
            "content_type": content_type,
            "description": clean_description,
        }
        if image_observation is not None:
            row["image_observation"] = image_observation
        media_description_rows.append(row)
        if len(media_description_rows) >= MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS:
            break

    return media_description_rows


def build_reply_media_description_rows(
    reply_context: Mapping[str, object] | None,
) -> list[MediaDescriptionRow]:
    """Project quoted-reply image summaries into episode media rows.

    Args:
        reply_context: Service-facing reply context that may include stored
            prompt-safe attachment summaries for the replied-to message.

    Returns:
        Image media rows that preserve quoted-image availability without
        requiring raw media bytes.
    """
    if not isinstance(reply_context, Mapping):
        return_value: list[MediaDescriptionRow] = []
        return return_value

    attachments = reply_context.get("reply_attachments")
    if not isinstance(attachments, list):
        return_value = []
        return return_value

    source_message_id = _string_field(reply_context, "reply_to_message_id")
    rows: list[MediaDescriptionRow] = []
    for attachment in attachments:
        if not isinstance(attachment, Mapping):
            continue
        media_kind = _string_field(attachment, "media_kind").casefold()
        if media_kind != "image":
            continue

        description = _string_field(attachment, "description")
        summary_status = _summary_status(attachment, description)
        observation: ImageObservation = {
            "observation_origin": "quoted_reply_attachment",
            "source_message_id": source_message_id,
            "media_kind": "image",
            "summary_status": summary_status,
            "summary": description,
            "visible_text": [],
            "salient_visual_facts": [],
            "spatial_or_scene_facts": [],
            "uncertainty": [],
        }
        row: MediaDescriptionRow = {
            "content_type": "image/quoted-reply",
            "description": description,
            "image_observation": observation,
        }
        rows.append(row)

    return_value = build_text_chat_media_description_rows(rows)
    return return_value


def _sanitize_image_observation(
    value: object,
    *,
    description: str,
    content_type: str,
) -> ImageObservation | None:
    """Build a bounded image observation from structured or legacy input.

    Args:
        value: Optional structured image observation supplied by upstream.
        description: Prompt-safe image summary retained for compatibility.
        content_type: Current media content type.

    Returns:
        Bounded image observation, or ``None`` for non-image media without a
        structured visual observation.
    """
    if not content_type.startswith("image/"):
        return_value = None
        return return_value

    observation_data: Mapping[str, object] = {}
    if isinstance(value, Mapping):
        observation_data = value

    summary = _trim_media_description(
        _string_field(observation_data, "summary") or description
    )
    summary_status = _summary_status(observation_data, summary)
    if description == "" and not observation_data:
        return_value = None
        return return_value

    list_fields = {
        field_name: _string_list_field(observation_data, field_name)
        for field_name in _IMAGE_OBSERVATION_LIST_FIELDS
    }
    observation: ImageObservation = {
        "observation_origin": (
            _string_field(observation_data, "observation_origin")
            or "current_attachment"
        ),
        "source_message_id": _string_field(
            observation_data,
            "source_message_id",
        ),
        "media_kind": "image",
        "summary_status": summary_status,
        "summary": summary,
        "visible_text": list_fields["visible_text"],
        "salient_visual_facts": list_fields["salient_visual_facts"],
        "spatial_or_scene_facts": list_fields["spatial_or_scene_facts"],
        "uncertainty": list_fields["uncertainty"],
    }
    return observation


def _string_field(data: Mapping[str, object], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = _trim_media_description(value.strip())
    return return_value


def _string_list_field(
    data: Mapping[str, object],
    field_name: str,
) -> list[str]:
    value = data.get(field_name)
    if not isinstance(value, list):
        return_value: list[str] = []
        return return_value

    strings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        clean_item = _trim_media_description(item.strip())
        if clean_item:
            strings.append(clean_item)
    return strings


def _summary_status(
    data: Mapping[str, object],
    summary: str,
) -> Literal["available", "unavailable"]:
    value = data.get("summary_status")
    if value == "available" or value == "unavailable":
        return_value = value
    elif summary:
        return_value = "available"
    else:
        return_value = "unavailable"
    return return_value


def replace_text_chat_media_percepts(
    *,
    episode: CognitiveEpisodeV1,
    media_description_rows: list[MediaDescriptionRow] | None,
) -> CognitiveEpisodeV1:
    """Return a canonical user-message episode with refreshed media."""

    return _replace_canonical_media_percepts(
        episode,
        media_description_rows,
    )


def _replace_canonical_media_percepts(
    episode: CognitiveEpisodeV1,
    media_description_rows: list[MediaDescriptionRow] | None,
) -> CognitiveEpisodeV1:
    """Replace canonical media percepts after descriptor completion."""

    validate_cognitive_episode_v1(episode)
    updated_episode = deepcopy(episode)
    updated_episode["percepts"] = [
        percept
        for percept in updated_episode["percepts"]
        if percept["source_kind"]
        not in {"image_observation", "audio_observation"}
    ]
    dialog_percept = next(
        (
            percept
            for percept in updated_episode["percepts"]
            if percept["source_kind"] == "dialog"
        ),
        None,
    )
    if dialog_percept is None:
        raise CognitiveEpisodeValidationError(
            "canonical episode must include one dialog percept"
        )
    created_at = episode["created_at"]
    for index, row in enumerate(
        build_text_chat_media_description_rows(media_description_rows or [])[
            :MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS
        ],
        start=1,
    ):
        content_type = row["content_type"]
        source_kind = (
            "image_observation"
            if content_type.startswith("image/")
            else "audio_observation"
        )
        updated_episode["percepts"].insert(
            index,
            {
                "schema_version": "percept.v1",
                "percept_kind": source_kind,
                "source_kind": source_kind,
                "source_id": f"{episode['episode_id']}:media:{index}",
                "content": {
                    "content_type": content_type,
                    "description": row["description"],
                    "observation": dict(row.get("image_observation", {}))
                    if isinstance(row.get("image_observation"), Mapping)
                    else {},
                },
                "observed_at": created_at,
            },
        )
    return validate_cognitive_episode_v1(updated_episode)


def project_text_chat_compatibility_fields(
    episode: CognitiveEpisodeV1,
) -> TextChatCompatibilityProjection:
    """Project the canonical user-message episode into the RAG boundary."""

    return _project_canonical_text_chat_fields(episode)


def _project_canonical_text_chat_fields(
    episode: CognitiveEpisodeV1,
) -> TextChatCompatibilityProjection:
    """Project canonical user-message fields into the RAG boundary."""

    validate_cognitive_episode_v1(episode)
    if episode["trigger_source"] != "user_message":
        _raise_validation_error("episode must be a user_message episode")
    dialog_percept = next(
        (
            percept
            for percept in episode["percepts"]
            if percept["source_kind"] == "dialog"
        ),
        None,
    )
    if dialog_percept is None:
        _raise_validation_error("episode must include a dialog percept")
    content = dialog_percept["content"]
    user_input = str(
        content.get("semantic_text") or content.get("text") or ""
    )
    target_scope = episode["target_scope"]
    origin_metadata = episode["origin_metadata"]
    return {
        "storage_timestamp_utc": episode["created_at"],
        "local_time_context": _canonical_local_time_context(episode),
        "user_input": user_input,
        "platform": str(target_scope.get("platform", "")),
        "platform_channel_id": str(target_scope.get("platform_channel_id", "")),
        "channel_type": str(target_scope.get("channel_type", "")),
        "platform_message_id": str(
            origin_metadata.get("platform_message_id", "")
        ),
        "active_turn_platform_message_ids": list(
            origin_metadata.get("active_turn_platform_message_ids", [])
        ),
        "active_turn_conversation_row_ids": list(
            origin_metadata.get("active_turn_conversation_row_ids", [])
        ),
        "platform_user_id": str(
            target_scope.get("current_platform_user_id", "")
        ),
        "global_user_id": str(target_scope.get("current_global_user_id", "")),
        "user_name": str(target_scope.get("current_display_name", "")),
    }


def _canonical_local_time_context(
    episode: CognitiveEpisodeV1,
) -> LocalTimeContextDoc:
    """Return the canonical local-time percept as a typed context."""

    for percept in episode["percepts"]:
        if percept["percept_kind"] != "local_time_context":
            continue
        content = percept["content"]
        raw_context = content.get("local_time_context")
        if isinstance(raw_context, Mapping):
            return dict(raw_context)  # type: ignore[return-value]
    raise CognitiveEpisodeValidationError(
        "canonical episode is missing local_time_context"
    )


def _trim_media_description(description: str) -> str:
    """Clamp a media description to the episode percept character budget.

    Args:
        description: Sanitized image or audio description text.

    Returns:
        Description text no longer than the configured media percept limit.
    """
    if len(description) <= MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS:
        return description

    body_limit = MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS - len("...")
    trimmed_description = description[:body_limit].rstrip()
    return_value = f"{trimmed_description}..."
    return return_value


def _raise_validation_error(message: str) -> NoReturn:
    raise CognitiveEpisodeValidationError(message)
