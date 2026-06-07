"""L2d action initializer for modality-neutral action specs."""

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
from kazusa_ai_chatbot.action_router.prompt import ACTION_ROUTER_PROMPT
from kazusa_ai_chatbot.action_router.router import (
    build_action_router_payload_text,
    route_action_requests,
)
from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_GOAL_PROGRESS_VERSION,
    RESOLVER_PENDING_RESOLUTION_VERSION,
    ResolverGoalProgressV1,
    ResolverCapabilityRequestV1,
    ResolverPendingResolutionV1,
    ResolverValidationError,
    validate_resolver_goal_progress,
    validate_resolver_capability_request,
    validate_resolver_pending_resolution,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_output_contracts import (
    validate_cognition_output_contract,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import CognitionState
from kazusa_ai_chatbot.utils import get_llm, log_preview

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


def build_action_initializer_payload(
    state: CognitionState,
    capabilities: Mapping[str, CapabilitySpecV1] | None = None,
) -> str:
    """Build the current-run semantic context for L2d action selection.

    Delegates to ``action_router.payload.build_action_router_payload`` and
    serializes the result to a JSON string for the LLM human message.

    Args:
        state: Cognition state after L2c judgment.
        capabilities: Optional registry override for deterministic tests.

    Returns:
        A JSON string containing prompt-safe semantic sections.
    """

    payload_text = build_action_router_payload_text(state, capabilities)
    return payload_text


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


def _normalize_action_requests(parsed: object) -> list[ActionRequestV1]:
    """Normalize LLM-selected semantic requests before materialization."""

    normalized_requests: list[ActionRequestV1] = []
    if not isinstance(parsed, dict):
        return normalized_requests

    raw_requests = parsed.get("action_requests")
    if not isinstance(raw_requests, list):
        return normalized_requests

    for raw_request in raw_requests:
        if not isinstance(raw_request, dict):
            logger.warning("L2d dropped non-object action request")
            continue
        capability = _semantic_text(raw_request, "capability")
        reason = _semantic_text(raw_request, "reason")
        if not capability or not reason:
            logger.warning("L2d dropped action request without capability or reason")
            continue
        normalized_request: ActionRequestV1 = {
            "capability": capability,
            "reason": reason,
        }
        for optional_field in ("decision", "detail"):
            field_value = _semantic_text(raw_request, optional_field)
            if field_value:
                normalized_request[optional_field] = field_value
        normalized_requests.append(normalized_request)
        if len(normalized_requests) >= ACTION_SPEC_CAP:
            break
    return normalized_requests


def _normalize_pending_resume_action_requests(
    parsed: object,
    state: CognitionState,
) -> list[ActionRequestV1]:
    """Recover pending HIL/approval answers emitted with resolver capability."""

    pending_resume = state.get("pending_resolver_resume")
    if not isinstance(pending_resume, dict):
        action_requests = _normalize_action_requests(parsed)
        return action_requests

    pending_capability = pending_resume.get("capability_kind")
    if pending_capability not in ("human_clarification", "approval_preparation"):
        action_requests = _normalize_action_requests(parsed)
        return action_requests

    if not isinstance(parsed, dict):
        return []

    raw_requests = parsed.get("action_requests")
    if not isinstance(raw_requests, list):
        return []

    rewritten_requests: list[object] = []
    for raw_request in raw_requests:
        if not isinstance(raw_request, dict):
            rewritten_requests.append(raw_request)
            continue

        capability = _semantic_text(raw_request, "capability")
        if capability != pending_capability:
            rewritten_requests.append(raw_request)
            continue

        rewritten_request = dict(raw_request)
        rewritten_request["capability"] = SPEAK_CAPABILITY
        if not _semantic_text(rewritten_request, "decision"):
            rewritten_request["decision"] = "visible_reply"
        pending_detail = _pending_resume_surface_detail(
            pending_resume,
            pending_capability,
        )
        if pending_detail:
            rewritten_request["detail"] = pending_detail
        rewritten_requests.append(rewritten_request)

    rewritten_parsed = {"action_requests": rewritten_requests}
    action_requests = _normalize_action_requests(rewritten_parsed)
    return action_requests


def _pending_resume_speak_request(
    state: CognitionState,
) -> list[ActionRequestV1]:
    """Build a text action for an active pending HIL or approval row."""

    pending_resume = state.get("pending_resolver_resume")
    if not isinstance(pending_resume, dict):
        return []

    pending_capability = pending_resume.get("capability_kind")
    if pending_capability not in ("human_clarification", "approval_preparation"):
        return []

    pending_detail = _pending_resume_surface_detail(
        pending_resume,
        pending_capability,
    )
    if not pending_detail:
        return []

    action_request: ActionRequestV1 = {
        "capability": SPEAK_CAPABILITY,
        "decision": "visible_reply",
        "detail": pending_detail,
        "reason": "处理当前等待中的澄清或审批状态。",
    }
    action_requests = [action_request]
    return action_requests


def _pending_resume_surface_detail(
    pending_resume: Mapping[str, object],
    pending_capability: object,
) -> str:
    """Return the prompt-safe pending detail that L3 should surface."""

    pending_detail = ""
    if pending_capability == "human_clarification":
        pending_detail = _semantic_text(
            pending_resume,
            "prompt_safe_question",
        )
    elif pending_capability == "approval_preparation":
        pending_detail = _semantic_text(
            pending_resume,
            "prompt_safe_approval_summary",
        )
    return pending_detail


def _normalize_resolver_capability_requests(
    parsed: object,
) -> list[ResolverCapabilityRequestV1]:
    """Normalize L2d-selected resolver requests before graph merge."""

    normalized_requests: list[ResolverCapabilityRequestV1] = []
    if not isinstance(parsed, dict):
        return normalized_requests

    raw_requests = parsed.get("resolver_capability_requests")
    if not isinstance(raw_requests, list):
        return normalized_requests

    for raw_request in raw_requests:
        if (
            isinstance(raw_request, dict)
            and _semantic_text(raw_request, "capability_kind")
            in ALLOWED_ACTION_CAPABILITIES
        ):
            continue
        try:
            normalized_request = validate_resolver_capability_request(raw_request)
        except ResolverValidationError as exc:
            logger.warning(f"L2d dropped invalid resolver request: {exc}")
            continue
        normalized_requests.append(normalized_request)
        if len(normalized_requests) >= ACTION_SPEC_CAP:
            break
    return_value = normalized_requests
    return return_value


def _normalize_resolver_pending_resolution(
    parsed: object,
    state: CognitionState,
) -> ResolverPendingResolutionV1 | None:
    """Bind L2d's pending decision to the active deterministic pending row."""

    if not isinstance(parsed, dict):
        return_value = None
        return return_value

    raw_resolution = parsed.get("resolver_pending_resolution")
    if raw_resolution is None:
        return_value = None
        return return_value

    if not isinstance(raw_resolution, dict):
        logger.warning(
            "L2d dropped invalid pending resolver resolution: expected object"
        )
        return_value = None
        return return_value

    pending_resume = state.get("pending_resolver_resume")
    if not isinstance(pending_resume, dict):
        logger.warning(
            "L2d dropped pending resolver resolution without active pending row"
        )
        return_value = None
        return return_value

    raw_active_resume_id = pending_resume.get("resume_id")
    if (
        not isinstance(raw_active_resume_id, str)
        or not raw_active_resume_id.strip()
    ):
        logger.warning(
            "L2d dropped pending resolver resolution with invalid active pending id"
        )
        return_value = None
        return return_value

    active_resume_id = raw_active_resume_id.strip()
    raw_model_resume_id = raw_resolution.get("resume_id")
    if (
        isinstance(raw_model_resume_id, str)
        and raw_model_resume_id.strip()
        and raw_model_resume_id.strip() != active_resume_id
    ):
        logger.warning(
            "L2d ignored model-supplied pending resolver id that did not "
            "match the active pending row"
        )

    canonical_resolution = {
        "schema_version": RESOLVER_PENDING_RESOLUTION_VERSION,
        "resume_id": active_resume_id,
        "decision": raw_resolution.get("decision", ""),
        "reason": raw_resolution.get("reason", ""),
    }
    try:
        normalized_resolution = validate_resolver_pending_resolution(
            canonical_resolution,
        )
    except ResolverValidationError as exc:
        logger.warning(f"L2d dropped invalid pending resolver resolution: {exc}")
        return_value = None
        return return_value

    return_value = normalized_resolution
    return return_value


def _normalize_resolver_goal_progress(
    parsed: object,
) -> ResolverGoalProgressV1 | None:
    """Normalize L2d's optional goal-progress checklist."""

    if not isinstance(parsed, dict):
        return_value = None
        return return_value

    raw_progress = parsed.get("resolver_goal_progress")
    if raw_progress is None:
        return_value = None
        return return_value
    if not isinstance(raw_progress, dict):
        return_value = None
        return return_value

    canonical_progress = dict(raw_progress)
    canonical_progress["schema_version"] = RESOLVER_GOAL_PROGRESS_VERSION
    try:
        normalized_progress = validate_resolver_goal_progress(
            canonical_progress,
        )
    except ResolverValidationError as exc:
        logger.warning(f"L2d dropped invalid goal progress: {exc}")
        return_value = None
        return return_value

    return_value = normalized_progress
    return return_value


def _goal_progress_with_surface_requirements(
    goal_progress: ResolverGoalProgressV1 | None,
    action_specs: list[ActionSpecV1],
) -> ResolverGoalProgressV1 | None:
    """Mirror L2d's open deliverables into the visible-response checklist."""

    if goal_progress is None:
        return_value = None
        return return_value

    has_visible_speak = any(
        action_spec["kind"] == SPEAK_CAPABILITY
        and action_spec["visibility"] == "user_visible"
        for action_spec in action_specs
    )
    if not has_visible_speak:
        return_value = goal_progress
        return return_value

    requirements = list(goal_progress["final_response_requirements"])
    for deliverable in goal_progress["deliverables"]:
        if deliverable["status"] not in OPEN_GOAL_DELIVERABLE_STATUSES:
            continue
        description = deliverable["description"]
        if any(description in requirement for requirement in requirements):
            continue
        requirement = (
            f"{description}："
            f"{deliverable['note']}"
        )
        requirements.append(requirement)

    if requirements == goal_progress["final_response_requirements"]:
        return_value = goal_progress
        return return_value

    updated_progress = dict(goal_progress)
    updated_progress["final_response_requirements"] = requirements
    normalized_progress = validate_resolver_goal_progress(updated_progress)
    return_value = normalized_progress
    return return_value


def _resolver_requests_repeat_pending_capability(
    requests: list[ResolverCapabilityRequestV1],
    state: CognitionState,
) -> bool:
    """Return whether resolver requests only repeat the active pending row.

    Args:
        requests: Normalized resolver requests emitted by L2d.
        state: Cognition state that may contain a pending resolver resume.

    Returns:
        True when every request repeats the pending HIL/approval capability.
    """

    if not requests:
        return_value = False
        return return_value

    pending_resume = state.get("pending_resolver_resume")
    if not isinstance(pending_resume, dict):
        return_value = False
        return return_value

    pending_capability = pending_resume.get("capability_kind")
    if pending_capability not in ("human_clarification", "approval_preparation"):
        return_value = False
        return return_value

    repeats_pending = all(
        request["capability_kind"] == pending_capability
        for request in requests
    )
    return_value = repeats_pending
    return return_value


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


# Compatibility alias for tests and existing L2d imports.
_ACTION_INITIALIZER_PROMPT = ACTION_ROUTER_PROMPT

_action_initializer_llm = get_llm(
    temperature=0.1,
    top_p=0.7,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)


async def call_action_initializer(state: CognitionState) -> CognitionState:
    """Run L2d and return validated action specs without executing them."""

    parsed = await route_action_requests(_action_initializer_llm, state)
    resolver_capability_requests = _normalize_resolver_capability_requests(parsed)
    resolver_pending_resolution = _normalize_resolver_pending_resolution(
        parsed,
        state,
    )
    resolver_goal_progress = _normalize_resolver_goal_progress(parsed)
    has_raw_action_requests = (
        isinstance(parsed, dict)
        and isinstance(parsed.get("action_requests"), list)
        and bool(parsed["action_requests"])
    )
    repeated_pending_request = False
    if (
        resolver_capability_requests
        and _resolver_requests_repeat_pending_capability(
            resolver_capability_requests,
            state,
        )
    ):
        logger.warning(
            "L2d converted repeated pending resolver request to pending "
            "surface handling"
        )
        resolver_capability_requests = []
        repeated_pending_request = True
    if resolver_capability_requests:
        action_requests = []
        if isinstance(parsed, dict) and parsed.get("action_requests"):
            logger.warning(
                "L2d dropped action requests while resolver requests are pending"
            )
    else:
        if repeated_pending_request and has_raw_action_requests:
            action_requests = _normalize_pending_resume_action_requests(
                parsed,
                state,
            )
        elif repeated_pending_request:
            action_requests = _pending_resume_speak_request(state)
        else:
            action_requests = _normalize_action_requests(parsed)
    action_specs = _materialize_action_specs(action_requests, state)
    resolver_goal_progress = _goal_progress_with_surface_requirements(
        resolver_goal_progress,
        action_specs,
    )
    logger.debug(
        f"L2d action initializer: count={len(action_specs)} "
        f"kinds={log_preview([spec['kind'] for spec in action_specs])} "
        f"resolver_requests={len(resolver_capability_requests)} "
    )
    return_value = {
        "action_specs": action_specs,
        "resolver_capability_requests": resolver_capability_requests,
    }
    if resolver_pending_resolution is not None:
        return_value["resolver_pending_resolution"] = resolver_pending_resolution
    if resolver_goal_progress is not None:
        return_value["resolver_goal_progress"] = resolver_goal_progress
    validate_cognition_output_contract(
        stage="l2d_action_selection",
        payload=return_value,
    )
    return return_value
