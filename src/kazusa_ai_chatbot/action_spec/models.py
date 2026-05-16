"""Typed action-spec contracts and validators."""

from __future__ import annotations

from typing import Literal, TypedDict

ACTION_SPEC_VERSION = "action_spec.v1"
ACTION_SOURCE_REF_VERSION = "action_source_ref.v1"
ACTION_TARGET_VERSION = "action_target.v1"
ACTION_CONTINUATION_VERSION = "action_continuation.v1"
CAPABILITY_SPEC_VERSION = "capability_spec.v1"
EVIDENCE_REF_VERSION = "evidence_ref.v1"

ALLOWED_EVIDENCE_KINDS = frozenset(
    (
        "conversation_message",
        "memory_unit",
        "reflection_run",
        "scheduled_event",
        "tool_result",
        "system_event",
        "external_document",
    )
)
ALLOWED_SOURCE_REF_KINDS = frozenset(
    (
        "conversation_message",
        "memory_unit",
        "reflection_run",
        "scheduled_event",
        "tool_result",
        "cognitive_episode",
        "system_event",
    )
)
ALLOWED_SOURCE_RELATIONSHIPS = frozenset(
    ("basis", "target", "lineage", "result", "audit")
)
ALLOWED_TARGET_KINDS = frozenset(
    (
        "current_user",
        "current_channel",
        "user",
        "channel",
        "memory_unit",
        "cognitive_episode",
        "self",
        "none",
    )
)
ALLOWED_CONTINUATION_MODES = frozenset(
    ("none", "immediate_followup", "scheduled_followup", "background_followup")
)
ALLOWED_CAPABILITY_OWNERS = frozenset(
    ("memory_lifecycle", "orchestrator", "l3_text", "l3_image")
)
ALLOWED_COGNITION_MODES = frozenset(("deliberative", "reflex"))
ALLOWED_URGENCY_VALUES = frozenset(("now", "background", "scheduled"))
ALLOWED_VISIBILITY_VALUES = frozenset(("private", "preview", "user_visible"))

LIFECYCLE_STATUS_BY_DECISION = {
    "fulfilled": "completed",
    "abandoned": "cancelled",
    "obsolete": "archived",
    "deferred": "active",
}

PolicyRefV1 = str


class ActionValidationError(ValueError):
    """Raised when an action contract payload is structurally invalid."""


class EvidenceRefV1(TypedDict):
    """Reference to evidence that supports an action decision."""

    schema_version: Literal["evidence_ref.v1"]
    evidence_kind: Literal[
        "conversation_message",
        "memory_unit",
        "reflection_run",
        "scheduled_event",
        "tool_result",
        "system_event",
        "external_document",
    ]
    evidence_id: str
    owner: str
    excerpt: str | None
    observed_at: str | None


class ActionSourceRefV1(TypedDict):
    """Reference connecting an action to its cognitive basis or target."""

    schema_version: Literal["action_source_ref.v1"]
    ref_kind: Literal[
        "conversation_message",
        "memory_unit",
        "reflection_run",
        "scheduled_event",
        "tool_result",
        "cognitive_episode",
        "system_event",
    ]
    ref_id: str
    owner: str
    relationship: Literal["basis", "target", "lineage", "result", "audit"]
    evidence_refs: list[EvidenceRefV1]


class ActionTargetV1(TypedDict):
    """Semantic target selected by cognition for one action."""

    schema_version: Literal["action_target.v1"]
    target_kind: Literal[
        "current_user",
        "current_channel",
        "user",
        "channel",
        "memory_unit",
        "cognitive_episode",
        "self",
        "none",
    ]
    target_id: str | None
    owner: str
    scope: dict[str, object]


class ActionContinuationV1(TypedDict):
    """Optional contract for a future follow-up cognition cycle."""

    schema_version: Literal["action_continuation.v1"]
    mode: Literal[
        "none",
        "immediate_followup",
        "scheduled_followup",
        "background_followup",
    ]
    episode_type: str | None
    max_depth: int
    include_result_as: str | None


class CapabilitySpecV1(TypedDict):
    """Action-category capability registry entry."""

    schema_version: Literal["capability_spec.v1"]
    capability_kind: str
    category: Literal["action"]
    owner_module: Literal[
        "memory_lifecycle",
        "orchestrator",
        "l3_text",
        "l3_image",
    ]
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    handler_id: str
    lifecycle_hooks: list[str]
    permission_policy: PolicyRefV1
    rate_limit_policy: PolicyRefV1
    audit_policy: PolicyRefV1
    prompt_projection_policy: PolicyRefV1


class ActionSpecV1(TypedDict):
    """Materialized modality-neutral action residue from semantic requests."""

    schema_version: Literal["action_spec.v1"]
    kind: str
    cognition_mode: Literal["deliberative", "reflex"]
    source_refs: list[ActionSourceRefV1]
    target: ActionTargetV1
    params: dict[str, object]
    urgency: Literal["now", "background", "scheduled"]
    visibility: Literal["private", "preview", "user_visible"]
    deadline: str | None
    continuation: ActionContinuationV1
    reason: str


class ActionEvalResult(TypedDict):
    """Deterministic validation result for one action spec."""

    ok: bool
    action_spec: ActionSpecV1 | None
    capability: CapabilitySpecV1 | None
    idempotency_key: str | None
    handler_owner: str | None
    errors: list[str]


class SpeakParamsV1(TypedDict):
    """Params for a text-surface action realized by the L3 text handler."""

    delivery_mode: Literal[
        "visible_reply",
        "private_finalization",
        "delayed",
        "scheduled",
    ]
    execute_at: str | None
    surface_requirements: dict[str, object]


class MemoryLifecycleUpdateParamsV1(TypedDict):
    """Params for the initial user-memory lifecycle action."""

    memory_kind: Literal["user_memory_unit"]
    unit_type: Literal["active_commitment"]
    unit_id: str
    lifecycle_decision: Literal["fulfilled", "abandoned", "obsolete", "deferred"]
    due_at: str | None


class TriggerFutureCognitionParamsV1(TypedDict):
    """Params for a private request to create a later cognition episode."""

    episode_type: Literal["self_cognition"]
    trigger_at: str | None
    continuation_objective: str


def validate_evidence_ref(value: object) -> EvidenceRefV1:
    """Validate and return one evidence reference."""

    data = _require_mapping(value, "evidence_ref")
    _require_version(data, EVIDENCE_REF_VERSION)
    _require_enum(data, "evidence_kind", ALLOWED_EVIDENCE_KINDS)
    _require_non_empty_string(data, "evidence_id")
    _require_non_empty_string(data, "owner")
    _require_nullable_string(data, "excerpt")
    _require_nullable_string(data, "observed_at")
    return_value = data
    return return_value


def validate_action_source_ref(value: object) -> ActionSourceRefV1:
    """Validate and return one action source reference."""

    data = _require_mapping(value, "action_source_ref")
    _require_version(data, ACTION_SOURCE_REF_VERSION)
    _require_enum(data, "ref_kind", ALLOWED_SOURCE_REF_KINDS)
    _require_non_empty_string(data, "ref_id")
    _require_non_empty_string(data, "owner")
    _require_enum(data, "relationship", ALLOWED_SOURCE_RELATIONSHIPS)
    evidence_refs = _require_list(data, "evidence_refs")
    for evidence_ref in evidence_refs:
        validate_evidence_ref(evidence_ref)
    return_value = data
    return return_value


def validate_action_target(value: object) -> ActionTargetV1:
    """Validate and return one action target."""

    data = _require_mapping(value, "action_target")
    _require_version(data, ACTION_TARGET_VERSION)
    _require_enum(data, "target_kind", ALLOWED_TARGET_KINDS)
    _require_nullable_string(data, "target_id")
    _require_non_empty_string(data, "owner")
    scope = data.get("scope")
    if not isinstance(scope, dict):
        raise ActionValidationError("scope: expected object")
    return_value = data
    return return_value


def validate_action_continuation(value: object) -> ActionContinuationV1:
    """Validate and return one continuation contract."""

    data = _require_mapping(value, "action_continuation")
    _require_version(data, ACTION_CONTINUATION_VERSION)
    mode = _require_enum(data, "mode", ALLOWED_CONTINUATION_MODES)
    _require_nullable_string(data, "episode_type")
    max_depth = data.get("max_depth")
    if not isinstance(max_depth, int):
        raise ActionValidationError("max_depth: expected integer")
    if max_depth < 0:
        raise ActionValidationError("max_depth: expected non-negative integer")
    _require_nullable_string(data, "include_result_as")
    _validate_continuation_mode(data, str(mode), max_depth)
    return_value = data
    return return_value


def validate_capability_spec(value: object) -> CapabilitySpecV1:
    """Validate and return one action-category capability spec."""

    data = _require_mapping(value, "capability_spec")
    _require_version(data, CAPABILITY_SPEC_VERSION)
    _require_non_empty_string(data, "capability_kind")
    _require_enum(data, "category", frozenset(("action",)))
    _require_enum(data, "owner_module", ALLOWED_CAPABILITY_OWNERS)
    _require_dict_field(data, "input_schema")
    _require_dict_field(data, "output_schema")
    _require_non_empty_string(data, "handler_id")
    lifecycle_hooks = _require_list(data, "lifecycle_hooks")
    for hook in lifecycle_hooks:
        if not isinstance(hook, str) or not hook.strip():
            raise ActionValidationError("lifecycle_hooks: expected strings")
    for field_name in (
        "permission_policy",
        "rate_limit_policy",
        "audit_policy",
        "prompt_projection_policy",
    ):
        _require_non_empty_string(data, field_name)
    return_value = data
    return return_value


def validate_action_spec(value: object) -> ActionSpecV1:
    """Validate and return one action spec."""

    data = _require_mapping(value, "action_spec")
    _require_version(data, ACTION_SPEC_VERSION)
    _require_non_empty_string(data, "kind")
    cognition_mode = _require_enum(data, "cognition_mode", ALLOWED_COGNITION_MODES)
    if cognition_mode == "reflex":
        raise ActionValidationError("reflex cognition_mode is reserved")
    source_refs = _require_list(data, "source_refs")
    if not source_refs:
        raise ActionValidationError("source_refs: expected at least one source")
    for source_ref in source_refs:
        validate_action_source_ref(source_ref)
    validate_action_target(data.get("target"))
    _require_dict_field(data, "params")
    _require_enum(data, "urgency", ALLOWED_URGENCY_VALUES)
    _require_enum(data, "visibility", ALLOWED_VISIBILITY_VALUES)
    _require_nullable_string(data, "deadline")
    validate_action_continuation(data.get("continuation"))
    _require_non_empty_string(data, "reason")
    return_value = data
    return return_value


def _require_mapping(value: object, label: str) -> dict:
    """Return a dictionary payload or raise a contract error."""

    if not isinstance(value, dict):
        raise ActionValidationError(f"{label}: expected object")
    return_value = value
    return return_value


def _require_version(data: dict, expected: str) -> None:
    """Require a specific schema version on one contract object."""

    actual = data.get("schema_version")
    if actual != expected:
        raise ActionValidationError(f"schema_version: expected {expected}")


def _require_non_empty_string(data: dict, field_name: str) -> str:
    """Require one non-empty string field."""

    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ActionValidationError(f"{field_name}: expected non-empty string")
    return_value = value
    return return_value


def _require_nullable_string(data: dict, field_name: str) -> str | None:
    """Require one field to be either a string or None."""

    value = data.get(field_name)
    if value is not None and not isinstance(value, str):
        raise ActionValidationError(f"{field_name}: expected string or null")
    return_value = value
    return return_value


def _require_enum(data: dict, field_name: str, allowed: frozenset[str]) -> str:
    """Require one string field to belong to an allowed vocabulary."""

    value = data.get(field_name)
    if not isinstance(value, str) or value not in allowed:
        expected = sorted(allowed)
        raise ActionValidationError(f"{field_name}: expected one of {expected}")
    return_value = value
    return return_value


def _require_list(data: dict, field_name: str) -> list:
    """Require one list field."""

    value = data.get(field_name)
    if not isinstance(value, list):
        raise ActionValidationError(f"{field_name}: expected list")
    return_value = value
    return return_value


def _require_dict_field(data: dict, field_name: str) -> dict:
    """Require one object field."""

    value = data.get(field_name)
    if not isinstance(value, dict):
        raise ActionValidationError(f"{field_name}: expected object")
    return_value = value
    return return_value


def _validate_continuation_mode(
    data: dict,
    mode: str,
    max_depth: int,
) -> None:
    """Enforce the initial bounded continuation policy."""

    episode_type = data.get("episode_type")
    include_result_as = data.get("include_result_as")
    if mode == "none":
        if max_depth != 0:
            raise ActionValidationError("max_depth: expected 0 for no continuation")
        if episode_type is not None:
            raise ActionValidationError("episode_type: expected null")
        if include_result_as is not None:
            raise ActionValidationError("include_result_as: expected null")
        return
    if not isinstance(episode_type, str) or not episode_type.strip():
        raise ActionValidationError("episode_type: required for continuation")
    if max_depth < 1:
        raise ActionValidationError("max_depth: expected at least 1")
    if not isinstance(include_result_as, str) or not include_result_as.strip():
        raise ActionValidationError("include_result_as: required for continuation")
