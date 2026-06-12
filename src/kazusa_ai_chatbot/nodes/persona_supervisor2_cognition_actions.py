"""Deterministic bridge from cognition requests to project action specs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TypedDict

from kazusa_ai_chatbot.action_spec.models import (
    ActionSourceRefV1,
    ActionSpecV1,
    CapabilitySpecV1,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    BACKGROUND_WORK_REQUEST_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    build_initial_action_capabilities,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import CognitionState

logger = logging.getLogger(__name__)

ACTION_SPEC_CAP = 3
OPEN_GOAL_DELIVERABLE_STATUSES = ("pending", "partial", "blocked")
ALLOWED_ACTION_CAPABILITIES = frozenset((
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    BACKGROUND_WORK_REQUEST_CAPABILITY,
))


class ActionRequestV1(TypedDict, total=False):
    """Semantic action request emitted by L2d before deterministic wrapping."""

    capability: str
    decision: str
    reason: str
    detail: str


def _current_episode_source_ref() -> ActionSourceRefV1:
    """Return a stable prompt alias for the current cognitive episode."""

    source_ref: ActionSourceRefV1 = {
        "schema_version": "action_source_ref.v1",
        "ref_kind": "cognitive_episode",
        "ref_id": "current_cognitive_episode",
        "owner": "cognition_episode",
        "relationship": "basis",
        "evidence_refs": [],
    }
    return source_ref


def materialize_semantic_action_requests(
    requests: list[ActionRequestV1],
    state: CognitionState,
) -> list[ActionSpecV1]:
    """Materialize semantic action requests into project action specs."""

    action_specs = _materialize_action_specs(requests, state)
    return action_specs


def _materialize_action_specs(
    requests: list[ActionRequestV1],
    state: CognitionState,
) -> list[ActionSpecV1]:
    """Wrap semantic requests in deterministic action-spec envelopes."""

    action_specs: list[ActionSpecV1] = []
    for index, request in enumerate(requests):
        continuation_objective: str | None = None
        if request["capability"] == TRIGGER_FUTURE_COGNITION_CAPABILITY:
            continuation_objective = _future_cognition_objective(
                requests,
                future_request_index=index,
            )
        action_spec = _materialize_action_request(
            request,
            state,
            continuation_objective=continuation_objective,
        )
        if action_spec is None:
            continue
        action_specs.append(action_spec)
        if len(action_specs) >= ACTION_SPEC_CAP:
            break
    return action_specs


def _materialize_action_request(
    request: ActionRequestV1,
    state: CognitionState,
    *,
    continuation_objective: str | None = None,
) -> ActionSpecV1 | None:
    """Build one validated action spec for a selected semantic capability."""

    capability = request["capability"]
    if capability == SPEAK_CAPABILITY:
        action_spec = _build_speak_action_spec(request, state)
    elif capability == MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
        action_spec = _build_memory_lifecycle_action_spec(request, state)
    elif capability == TRIGGER_FUTURE_COGNITION_CAPABILITY:
        if _is_scheduled_future_cognition_source(state):
            logger.warning(
                "L2d dropped future-cognition request from scheduled "
                "future-cognition source"
            )
            return None
        action_spec = _build_future_cognition_action_spec(
            request,
            state,
            continuation_objective=continuation_objective,
        )
    elif capability == BACKGROUND_WORK_REQUEST_CAPABILITY:
        action_spec = _build_background_work_action_spec(request, state)
    else:
        logger.warning(f"L2d dropped unsupported action capability: {capability}")
        return None

    if action_spec is None:
        return None
    validated_spec = validate_action_spec(action_spec)
    return validated_spec


def _build_speak_action_spec(
    request: ActionRequestV1,
    state: CognitionState,
) -> dict[str, object]:
    """Build the deterministic envelope for a text-surface request."""

    delivery_mode = _delivery_mode_for_request(request, state)
    target_kind = "current_channel"
    visibility = "user_visible"
    urgency = "now"
    if delivery_mode == "private_finalization":
        target_kind = "self"
        visibility = "private"
        urgency = "background"
    elif delivery_mode == "scheduled":
        urgency = "scheduled"
    detail = _semantic_text(request, "detail")
    surface_requirements = {
        "decision": _semantic_text(request, "decision"),
        "detail": detail,
    }
    action_spec = _build_action_spec(
        kind=SPEAK_CAPABILITY,
        source_refs=[_current_episode_source_ref()],
        target={
            "schema_version": "action_target.v1",
            "target_kind": target_kind,
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        },
        params={
            "delivery_mode": delivery_mode,
            "execute_at": None,
            "surface_requirements": surface_requirements,
        },
        urgency=urgency,
        visibility=visibility,
        deadline=None,
        reason=request["reason"],
    )
    return action_spec


def _delivery_mode_for_request(
    request: ActionRequestV1,
    _state: CognitionState,
) -> str:
    """Return the text-surface mode implied by semantics and trigger context."""

    decision = _semantic_text(request, "decision")
    if decision in (
        "visible_reply",
        "private_finalization",
        "delayed",
        "scheduled",
    ):
        return decision

    return "visible_reply"


def _build_memory_lifecycle_action_spec(
    request: ActionRequestV1,
    _state: CognitionState,
) -> dict[str, object] | None:
    """Build the specialist route intent for commitment lifecycle review."""

    detail = _semantic_text(request, "detail")
    if not detail:
        detail = request["reason"]
    action_spec = _build_action_spec(
        kind=MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        source_refs=[_current_episode_source_ref()],
        target={
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "memory_lifecycle_specialist",
            "scope": {"unit_type": "active_commitment"},
        },
        params={
            "review_kind": "active_commitment_lifecycle",
            "detail": detail,
        },
        urgency="background",
        visibility="private",
        deadline=None,
        reason=request["reason"],
    )
    return action_spec


def _build_future_cognition_action_spec(
    request: ActionRequestV1,
    state: CognitionState,
    *,
    continuation_objective: str | None,
) -> dict[str, object]:
    """Build the deterministic envelope for a future cognition request."""

    if continuation_objective is None:
        continuation_objective = _semantic_text(request, "detail")
    if not continuation_objective:
        continuation_objective = request["reason"]
    action_spec = _build_action_spec(
        kind=TRIGGER_FUTURE_COGNITION_CAPABILITY,
        source_refs=[_current_episode_source_ref()],
        target={
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "orchestrator",
            "scope": _future_cognition_target_scope(state),
        },
        params={
            "episode_type": "self_cognition",
            "trigger_at": None,
            "continuation_objective": continuation_objective,
        },
        urgency="background",
        visibility="private",
        deadline=None,
        continuation=_scheduled_followup_continuation(),
        reason=request["reason"],
    )
    return action_spec


def _deterministic_work_seed(
    request: ActionRequestV1,
    state: CognitionState,
) -> str:
    """Build a deterministic task_brief from state and route reason.

    The trusted task_brief is never LLM-generated. It is built from
    the decontextualized input and the route reason already validated
    by the action request normalizer.
    """

    decontextualized = state.get("decontexualized_input")
    if isinstance(decontextualized, str) and decontextualized.strip():
        work_seed = decontextualized.strip()[:2000]
    else:
        work_seed = request.get("reason", "")
    detail = _semantic_text(request, "detail")
    if detail:
        work_seed = f"{work_seed} — {detail}"
    return work_seed


def _build_background_work_action_spec(
    request: ActionRequestV1,
    state: CognitionState,
) -> dict[str, object] | None:
    """Build the generic background-work queue action."""

    task_brief = _deterministic_work_seed(request, state)

    action_spec = _build_action_spec(
        kind=BACKGROUND_WORK_REQUEST_CAPABILITY,
        source_refs=[_current_episode_source_ref()],
        target={
            "schema_version": "action_target.v1",
            "target_kind": "current_user",
            "target_id": None,
            "owner": "background_work",
            "scope": _background_work_target_scope(state),
        },
        params={
            "task_brief": task_brief,
            "requested_delivery": "send_result_when_done",
            "max_output_chars": BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
        },
        urgency="background",
        visibility="private",
        deadline=None,
        reason=request["reason"],
    )
    return action_spec


def _background_work_target_scope(
    state: CognitionState,
) -> dict[str, object]:
    """Bind trusted source scope for durable background-work handoff."""

    episode = state.get("cognitive_episode")
    target_scope: Mapping[str, object] = {}
    origin_metadata: Mapping[str, object] = {}
    if isinstance(episode, Mapping):
        raw_target_scope = episode.get("target_scope")
        if isinstance(raw_target_scope, Mapping):
            target_scope = raw_target_scope
        raw_origin_metadata = episode.get("origin_metadata")
        if isinstance(raw_origin_metadata, Mapping):
            origin_metadata = raw_origin_metadata

    source_platform = _state_or_mapping_text(
        state,
        "platform",
        target_scope,
        "platform",
    )
    source_character_name = _character_name_for_scope(state)

    scope = {
        "source_platform": source_platform,
        "source_channel_id": _state_or_mapping_text(
            state,
            "platform_channel_id",
            target_scope,
            "platform_channel_id",
        ),
        "source_channel_type": _state_or_mapping_text(
            state,
            "channel_type",
            target_scope,
            "channel_type",
        ),
        "source_message_id": _state_or_mapping_text(
            state,
            "platform_message_id",
            origin_metadata,
            "platform_message_id",
        ),
        "source_platform_bot_id": _source_platform_bot_id_for_scope(
            state,
            origin_metadata,
        ),
        "source_character_name": source_character_name,
        "requester_global_user_id": _state_or_mapping_text(
            state,
            "global_user_id",
            target_scope,
            "current_global_user_id",
        ),
        "requester_platform_user_id": _state_or_mapping_text(
            state,
            "platform_user_id",
            target_scope,
            "current_platform_user_id",
        ),
        "requester_display_name": _state_or_mapping_text(
            state,
            "user_name",
            target_scope,
            "current_display_name",
        ),
    }
    return scope


def _future_cognition_objective(
    requests: list[ActionRequestV1],
    *,
    future_request_index: int,
) -> str:
    """Return the one-string objective for a future cognition handoff."""

    future_request = requests[future_request_index]
    continuation_objective = _semantic_text(future_request, "detail")
    if continuation_objective:
        return continuation_objective

    continuation_objective = future_request["reason"]
    return continuation_objective


def _future_cognition_target_scope(state: CognitionState) -> dict[str, object]:
    """Bind trusted source scope for the later scheduled cognition slot."""

    scope: dict[str, object] = {
        "episode_type": "self_cognition",
    }
    field_map = (
        ("platform", "source_platform"),
        ("platform_channel_id", "source_channel_id"),
        ("channel_type", "source_channel_type"),
        ("global_user_id", "source_user_id"),
        ("platform_bot_id", "source_platform_bot_id"),
    )
    for state_field, scope_field in field_map:
        field_value = _state_text(state, state_field)
        if field_value:
            scope[scope_field] = field_value

    character_name = _character_name_for_scope(state)
    if character_name:
        scope["source_character_name"] = character_name
    return scope


def _state_text(state: CognitionState, field_name: str) -> str:
    """Return one optional text value from the trusted cognition state."""

    raw_value = state.get(field_name)
    if not isinstance(raw_value, str):
        return_value = ""
        return return_value
    return_value = raw_value.strip()
    return return_value


def _mapping_text(value: Mapping[str, object], field_name: str) -> str:
    """Return one optional text value from a trusted mapping."""

    raw_value = value.get(field_name)
    if not isinstance(raw_value, str):
        return_value = ""
        return return_value
    return_value = raw_value.strip()
    return return_value


def _state_or_mapping_text(
    state: CognitionState,
    state_field_name: str,
    fallback_mapping: Mapping[str, object],
    fallback_field_name: str,
) -> str:
    """Read top-level state text, falling back to a trusted episode mapping."""

    state_value = _state_text(state, state_field_name)
    if state_value:
        return state_value

    fallback_value = _mapping_text(fallback_mapping, fallback_field_name)
    return fallback_value


def _source_platform_bot_id_for_scope(
    state: CognitionState,
    origin_metadata: Mapping[str, object],
) -> str:
    """Return the source bot id from trusted state or origin metadata."""

    platform_bot_id = _state_text(state, "platform_bot_id")
    if platform_bot_id:
        return platform_bot_id

    return_value = _mapping_text(origin_metadata, "platform_bot_id")
    return return_value


def _character_name_for_scope(state: CognitionState) -> str:
    """Return the active character name when present in trusted state."""

    character_name = _state_text(state, "character_name")
    if character_name:
        return character_name

    profile = state.get("character_profile")
    if not isinstance(profile, dict):
        return_value = ""
        return return_value

    name = profile.get("name")
    if not isinstance(name, str):
        return_value = ""
        return return_value

    return_value = name.strip()
    return return_value


def _is_scheduled_future_cognition_source(state: CognitionState) -> bool:
    """Return whether this cycle was itself started by a future-cognition slot."""

    conversation_progress = state.get("conversation_progress")
    if not isinstance(conversation_progress, dict):
        return False

    source = conversation_progress.get("source")
    is_scheduled_source = source == "scheduled_future_cognition"
    return is_scheduled_source


def _build_action_spec(
    *,
    kind: str,
    source_refs: list[ActionSourceRefV1],
    target: dict[str, object],
    params: dict[str, object],
    urgency: str,
    visibility: str,
    deadline: str | None,
    reason: str,
    continuation: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the common deterministic action-spec envelope."""

    action_continuation = continuation
    if action_continuation is None:
        action_continuation = _no_continuation()
    action_spec = {
        "schema_version": "action_spec.v1",
        "kind": kind,
        "cognition_mode": "deliberative",
        "source_refs": source_refs,
        "target": target,
        "params": params,
        "urgency": urgency,
        "visibility": visibility,
        "deadline": deadline,
        "continuation": action_continuation,
        "reason": reason,
    }
    return action_spec


def _no_continuation() -> dict[str, object]:
    """Return the default no-continuation execution contract."""

    continuation = {
        "schema_version": "action_continuation.v1",
        "mode": "none",
        "episode_type": None,
        "max_depth": 0,
        "include_result_as": None,
    }
    return continuation


def _scheduled_followup_continuation() -> dict[str, object]:
    """Return the bounded continuation contract for future cognition."""

    continuation = {
        "schema_version": "action_continuation.v1",
        "mode": "scheduled_followup",
        "episode_type": "self_cognition",
        "max_depth": 1,
        "include_result_as": "scheduled_event",
    }
    return continuation


def _semantic_text(value: Mapping[str, object], field_name: str) -> str:
    """Return one stripped semantic text field from LLM output."""

    raw_value = value.get(field_name)
    if not isinstance(raw_value, str):
        return ""
    return_value = raw_value.strip()
    return return_value
