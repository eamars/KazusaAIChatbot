"""L2d action initializer for modality-neutral action specs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.action_spec.models import (
    ActionSourceRefV1,
    ActionSpecV1,
    CapabilitySpecV1,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import (
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    build_initial_action_capabilities,
)
from kazusa_ai_chatbot.config import (
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ALLOWED_RESOLVER_CAPABILITIES,
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    RESOLVER_GOAL_PROGRESS_VERSION,
    RESOLVER_PENDING_RESOLUTION_VERSION,
    ResolverGoalProgressV1,
    ResolverCapabilityRequestV1,
    ResolverPendingResolutionV1,
    ResolverValidationError,
    project_pending_resume_for_cognition,
    validate_resolver_goal_progress,
    validate_resolver_capability_request,
    validate_resolver_pending_resolution,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_output_contracts import (
    validate_cognition_output_contract,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import CognitionState
from kazusa_ai_chatbot.utils import get_llm, log_preview, parse_llm_json_output

logger = logging.getLogger(__name__)

ACTION_SPEC_CAP = 3
OPEN_GOAL_DELIVERABLE_STATUSES = ("pending", "partial", "blocked")


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

    Args:
        state: Cognition state after L2c judgment.
        capabilities: Optional registry override for deterministic tests.

    Returns:
        One prompt-safe semantic string containing only dynamic turn context.
    """

    if capabilities is None:
        capabilities = build_initial_action_capabilities()
    prompt_capabilities = dict(capabilities)
    action_context = _build_action_context_text(state, prompt_capabilities)
    return action_context


def _build_action_context_text(
    state: CognitionState,
    prompt_capabilities: Mapping[str, CapabilitySpecV1],
) -> str:
    """Render the current cognition state as one model-facing context string."""

    episode = state["cognitive_episode"]
    input_sources = ", ".join(episode["input_sources"])
    evidence = _project_action_evidence(state)
    commitment_text = _commitment_context_text(
        state,
        memory_lifecycle_visible=(
            MEMORY_LIFECYCLE_UPDATE_CAPABILITY in prompt_capabilities
        ),
    )
    context_lines = [
        "当前行动上下文：",
        (
            f"触发来源：{episode['trigger_source']}；"
            f"输入来源：{input_sources}；"
            f"输出要求：{episode['output_mode']}；"
            f"场景：{state['channel_type']} 对话。"
        ),
        (
            "已形成的决定："
            f"立场={state['logical_stance']}；"
            f"意图={state['character_intent']}；"
            f"裁决={state['judgment_note']}；"
            f"内心判断={state['internal_monologue']}"
        ),
        (
            "即时感受："
            f"{state['emotional_appraisal']}；"
            f"互动潜台词：{state['interaction_subtext']}。"
        ),
        (
            "边界与社交语境："
            f"边界={_compact_context_value(state['boundary_core_assessment'])}；"
            f"距离={state['social_distance']}；"
            f"强度={state['emotional_intensity']}；"
            f"氛围={state['vibe_check']}；"
            f"关系={state['relational_dynamic']}。"
        ),
        f"当前输入摘要：{evidence['decontexualized_input']}",
        f"检索结论：{evidence['rag_answer'] or '无'}",
        commitment_text,
        f"相关记忆：{_evidence_list_text(evidence['memory_evidence'])}",
        f"对话进度：{_compact_context_value(evidence['conversation_progress'])}",
    ]
    resolver_context_text = _resolver_context_text(state)
    if resolver_context_text:
        context_lines.append(f"解析器上下文：\n{resolver_context_text}")
    group_engagement_text = _group_engagement_context_text(state)
    if group_engagement_text:
        context_lines.append(group_engagement_text)
    action_context = "\n".join(context_lines)
    return action_context


def _group_engagement_context_text(state: CognitionState) -> str:
    """Return group-channel engagement evidence for group self-cognition."""

    if not _is_group_self_cognition_context(state):
        return_value = ""
        return return_value

    raw_context = state.get("group_engagement_action_context")
    if not isinstance(raw_context, Mapping):
        return_value = "群聊参与习惯：无"
        return return_value

    raw_guidelines = raw_context.get("engagement_guidelines")
    guidelines: list[str] = []
    if isinstance(raw_guidelines, list):
        guidelines = [
            item.strip()
            for item in raw_guidelines
            if isinstance(item, str) and item.strip()
        ]
    if not guidelines:
        return_value = "群聊参与习惯：无"
        return return_value

    confidence = ""
    raw_confidence = raw_context.get("confidence")
    if isinstance(raw_confidence, str) and raw_confidence.strip():
        confidence = raw_confidence.strip()

    guideline_text = "；".join(guidelines)
    if confidence:
        return_value = f"群聊参与习惯：{guideline_text}；confidence={confidence}"
        return return_value

    return_value = f"群聊参与习惯：{guideline_text}"
    return return_value


def _is_group_self_cognition_context(state: CognitionState) -> bool:
    """Return whether L2d is selecting for a group self-cognition source."""

    if state["channel_type"] != "group":
        return_value = False
        return return_value

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        return_value = False
        return return_value

    trigger_source = episode.get("trigger_source")
    input_sources = episode.get("input_sources")
    is_internal_monologue = (
        isinstance(input_sources, list)
        and "internal_monologue" in input_sources
    )
    is_self_cognition = (
        trigger_source == "internal_thought"
        and is_internal_monologue
    )
    return is_self_cognition


def _resolver_context_text(state: CognitionState) -> str:
    """Return prompt-safe resolver recurrence context for L2d."""

    context_lines: list[str] = []
    resolver_context = state.get("resolver_context")
    if isinstance(resolver_context, str) and resolver_context.strip():
        context_lines.append(resolver_context.strip())

    pending_resume = state.get("pending_resolver_resume")
    if isinstance(pending_resume, dict):
        try:
            pending_context = project_pending_resume_for_cognition(pending_resume)
        except ResolverValidationError as exc:
            logger.warning(f"L2d dropped invalid pending resolver context: {exc}")
            pending_context = ""
        if pending_context:
            context_lines.append(pending_context)

    return_value = "\n".join(context_lines)
    return return_value


def _commitment_context_text(
    state: CognitionState,
    *,
    memory_lifecycle_visible: bool,
) -> str:
    """Return the prompt-safe commitment context without persistence IDs."""

    projected_commitments = _project_active_commitments_for_prompt(
        _raw_active_commitments(state)
    )
    if not projected_commitments:
        commitment_text = "活动承诺线索：无。"
        return commitment_text

    commitment_lines = _evidence_list_text(projected_commitments)
    if memory_lifecycle_visible:
        commitment_text = f"活动承诺线索：有；{commitment_lines}"
        return commitment_text

    commitment_text = f"活动承诺线索：当前不允许生命周期复核；{commitment_lines}"
    return commitment_text


def _evidence_list_text(entries: object) -> str:
    """Render prompt-safe evidence entries as compact prose."""

    if not isinstance(entries, list) or not entries:
        return "无"

    rendered_entries: list[str] = []
    for entry in entries:
        if isinstance(entry, dict):
            rendered_entry = _compact_mapping_text(entry)
        else:
            rendered_entry = _compact_context_value(entry)
        if rendered_entry != "无":
            rendered_entries.append(rendered_entry)

    if not rendered_entries:
        return "无"
    evidence_text = "；".join(rendered_entries)
    return evidence_text


def _compact_mapping_text(value: Mapping[str, object]) -> str:
    """Render a dictionary as compact prompt-facing prose."""

    rendered_pairs: list[str] = []
    for key, raw_value in value.items():
        rendered_value = _compact_context_value(raw_value)
        if rendered_value == "无":
            continue
        rendered_pairs.append(f"{key}={rendered_value}")

    if not rendered_pairs:
        return "无"
    mapping_text = "，".join(rendered_pairs)
    return mapping_text


def _compact_context_value(value: object) -> str:
    """Render one dynamic context value for the human message string."""

    if value is None:
        return "无"
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return "无"
        return stripped_value
    if isinstance(value, Mapping):
        mapping_text = _compact_mapping_text(value)
        return mapping_text
    if isinstance(value, list):
        list_text = _evidence_list_text(value)
        return list_text

    context_value = str(value)
    return context_value


def _project_action_evidence(state: CognitionState) -> dict[str, object]:
    """Return bounded, action-relevant evidence without raw transport IDs."""

    rag_result = state["rag_result"]
    answer = ""
    memory_evidence: list[object] = []
    if isinstance(rag_result, dict):
        raw_answer = rag_result.get("answer")
        if isinstance(raw_answer, str):
            answer = raw_answer
        raw_memory_evidence = rag_result.get("memory_evidence")
        if isinstance(raw_memory_evidence, list):
            memory_evidence = _project_memory_evidence_for_prompt(
                raw_memory_evidence
            )

    evidence = {
        "decontexualized_input": state["decontexualized_input"],
        "rag_answer": answer,
        "active_commitments": _project_active_commitments_for_prompt(
            _raw_active_commitments(state)
        ),
        "memory_evidence": memory_evidence,
        "conversation_progress": state.get("conversation_progress"),
    }
    return evidence


def _raw_active_commitments(state: CognitionState) -> list[object]:
    """Return raw active commitment rows for deterministic materialization."""

    rag_result = state["rag_result"]
    active_commitments: list[object] = []
    if not isinstance(rag_result, dict):
        return active_commitments

    raw_user_image = rag_result.get("user_image")
    if not isinstance(raw_user_image, dict):
        return active_commitments

    memory_context = raw_user_image.get("user_memory_context")
    if not isinstance(memory_context, dict):
        return active_commitments

    raw_commitments = memory_context.get("active_commitments")
    if isinstance(raw_commitments, list):
        active_commitments = raw_commitments
    return active_commitments


def _project_active_commitments_for_prompt(
    raw_commitments: list[object],
) -> list[dict[str, object]]:
    """Project active commitments without persistence identifiers."""

    projected_commitments: list[dict[str, object]] = []
    for raw_commitment in raw_commitments:
        if not isinstance(raw_commitment, dict):
            continue
        projected_commitment: dict[str, object] = {}
        for field_name in ("fact", "summary", "due_at", "due_state", "status"):
            field_value = raw_commitment.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                projected_commitment[field_name] = field_value
        if projected_commitment:
            projected_commitments.append(projected_commitment)
    return projected_commitments


def _project_memory_evidence_for_prompt(
    raw_memory_evidence: list[object],
) -> list[dict[str, object]]:
    """Project retrieved memory evidence without storage identifiers."""

    projected_evidence: list[dict[str, object]] = []
    for raw_entry in raw_memory_evidence:
        if not isinstance(raw_entry, dict):
            continue
        projected_entry: dict[str, object] = {}
        for field_name in ("summary", "fact", "excerpt", "due_at", "due_state"):
            field_value = raw_entry.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                projected_entry[field_name] = field_value
        if projected_entry:
            projected_evidence.append(projected_entry)
    return projected_evidence


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


def _normalize_misplaced_resolver_requests(
    parsed: object,
) -> list[ResolverCapabilityRequestV1]:
    """Recover resolver capability requests emitted in the action field."""

    normalized_requests: list[ResolverCapabilityRequestV1] = []
    if not isinstance(parsed, dict):
        return normalized_requests

    raw_requests = parsed.get("action_requests")
    if not isinstance(raw_requests, list):
        return normalized_requests

    for raw_request in raw_requests:
        if not isinstance(raw_request, dict):
            continue
        capability = _semantic_text(raw_request, "capability")
        if capability not in ALLOWED_RESOLVER_CAPABILITIES:
            continue
        objective = _semantic_text(raw_request, "detail")
        if not objective:
            objective = _semantic_text(raw_request, "decision")
        reason = _semantic_text(raw_request, "reason")
        if not objective or not reason:
            logger.warning(
                "L2d dropped misplaced resolver request without objective "
                "or reason"
            )
            continue
        request = {
            "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
            "capability_kind": capability,
            "objective": objective,
            "reason": reason,
            "priority": "now",
        }
        try:
            normalized_request = validate_resolver_capability_request(request)
        except ResolverValidationError as exc:
            logger.warning(
                f"L2d dropped invalid misplaced resolver request: {exc}"
            )
            continue
        normalized_requests.append(normalized_request)
        if len(normalized_requests) >= ACTION_SPEC_CAP:
            break
    return_value = normalized_requests
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


def _character_name_for_scope(state: CognitionState) -> str:
    """Return the active character name when present in trusted state."""

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


_ACTION_INITIALIZER_PROMPT = '''\
你是角色的语义行动选择层。
前序理解已经形成当前事件的立场、意图、边界判断和社交语境。
你的任务是把已经形成的行动意图整理成 0 到 3 个语义解析请求或语义动作请求。
解析请求表示当前证据、当前事实、用户澄清或审批信息还不足，必须先回到认知循环；动作请求表示当前认知循环已经可以外部化为可见表面或私有动作。
解析请求和动作请求不要混用：需要先解析时，返回 resolver_capability_requests，并让 action_requests 为空。
行动请求只描述我想做什么；不要生成最终发言文本，不要执行动作。
解析请求只描述下一步需要什么证据、事实、澄清或审批。
你还要维护 `resolver_goal_progress`：这是当前用户目标的语义进度表，不是动作请求，也不是工具参数。
它必须由本层根据当前输入、上游认知、解析器上下文和 observation 更新，供下一轮认知和 L3 保留原始目标、交付清单、依赖、已确认事实、推断和阻塞。
不要让 Python 或工具结果替你判断目标是否完成；你只输出结构化语义进度，确定性代码只负责校验和保存。

# 语言政策
- 除结构化枚举值、schema key、capability 名称、用户原文中的公开标识、URL、代码、命令、模型标签等必须保持原样的内容外，所有由你新生成的内部自由文本字段都必须使用简体中文。不要把内部 UUID、message id、platform id、channel id、pending/resume id 复制到自由文本字段。
- 用户原文、引用文本、专有名词、标题、别名、外部证据原句在需要精确保留时保持原语言；不要为了统一语言而改写。
- 不要添加翻译、双语复写或括号内解释，除非源文本本身已经包含。

# 来源识别
行动上下文会说明触发来源、输入来源和输出要求。
- `user_message` 表示当前外部用户发言或外部说话内容。
- `reflection_signal` + `reflection_artifact` 表示我自己的反思资料，不是用户输入、用户发言，也不是任何人正在对我说话。
- `internal_thought` + `internal_monologue` 表示我自己的内部观察资料，不是用户输入、用户发言，也不是任何人正在对我说话。
- 当前输入摘要、资料标题、字段名、JSON、时间戳、semantic_labels、window_summary、transport summary、model-facing metadata 不是可见发言对象；不要围绕这些结构选择 `speak`，也不要复制进 `decision`、`detail`、`reason` 等自由文本字段。

# 可选动作
- `rag_evidence` 是检索当前对话、记忆、关系或资料证据后再回到认知循环。内部思考需要回看前文、关系或记忆证据时，也使用同一个证据通道。
- `web_evidence` 是需要当前公共事实或外部资料证据后再回到认知循环。
- `human_clarification` 是缺少用户拥有的信息时，准备一个最小澄清问题后再回到认知循环。
- `approval_preparation` 是准备需要用户确认的副作用说明；它不执行提醒、调度、发送或数据库修改。
- `self_goal_resolution` 只在触发来源是 `internal_thought` 且输入来源包含 `internal_monologue` 时使用；它必须写入 `resolver_capability_requests`，绝不能写入 `action_requests.capability`。它用于收束已有内部目标，不是证据检索，也不是用户消息里的“先想一想”或“整理回复方案”。
- `speak` 是可见文字回复。选择它表示我决定把话说到当前外部频道；之后才会交给 L3/dialog 渲染为可见文本。
- `memory_lifecycle_update` 是私有活动承诺生命周期复核。只选择复核需要，不选择具体承诺、别名、数据库目标或生命周期决定。
- `trigger_future_cognition` 是私有未来认知。只在我需要等待或消费一个具体新信息后再处理具体问题、任务或承诺时选择。

# 解析器续轮原则
这些规则优先于普通选择流程：
- 如果本轮有清楚的用户目标，或解析器上下文已有 `resolver_goal_progress`、`original_goal`、`pending_resolver_resume`、`resolver_observations`，输出必须包含 `resolver_goal_progress`。
- `resolver_goal_progress.original_goal` 必须保持用户原始目标或 pending 中的 original_goal；当前补充信息只能更新约束、依赖和交付状态，不得替换原始目标。
- `resolver_goal_progress.deliverables` 必须拆出原始目标里用户实际期待看到的主要交付部分。不要只写“回答用户问题”，也不要因为当前只处理一个子目标就删除其他交付部分。
- 每个 deliverable 的 status 只能是 `pending`、`partial`、`satisfied`、`blocked`。证据不足但可以给框架时用 `partial`；必要证据、权限或用户信息无法取得时用 `blocked`；已由本轮 action detail 要求 L3 覆盖时才用 `satisfied`。
- `resolver_goal_progress.final_response_requirements` 是 L3/dialog 的交付清单。如果本轮选择 `speak`，它必须写清最终可见回答必须覆盖什么，不得少于未满足或部分满足的主要 deliverable。
- `source_backed_facts` 只能写当前 RAG、web、媒体或 resolver observation 直接支持的事实；失败、超时、空结果、只有线索或只有未确认候选时，不得把目标属性升格为已确认事实。
- 对时效性、公开来源绑定或用户明确要求核实来源的事实，必须先取得相应证据才能给具体当前断言。若 bounded 尝试后仍无足够证据，选择 `speak` 收束，并要求最终回答区分来源确认、角色推断和当前无法验证的部分。
- 如果检索答案按来源类别、证据轨道或比较对象区分结论，必须保留这些边界。某一路径未命中、只返回邻近线索或没有覆盖目标事实时，不得改写成跨来源一致、无冲突或已确认。
- 续轮能力目标要窄到一个能力调用可以完成。不要把多个对象、多个属性和多个证据路径塞进一次检索目标。
- 如果解析器上下文已有 `capability=human_clarification; status=blocked` 或 `pending_resolver_resume` 的 capability 是 `human_clarification`，本轮不要再次请求同一 blocked capability。返回一个 `speak` action_request，让 L3 只问 observation summary 或 pending question 里的那个最小缺口。
- 如果解析器上下文已有 `capability=approval_preparation; status=blocked` 或 `pending_resolver_resume` 的 capability 是 `approval_preparation`，本轮不要再次请求同一 blocked capability。返回一个 `speak` action_request，让 L3 说明待确认的副作用、影响和确认条件。
- approval preview 必须能力扎根：只说明当前系统确实能准备或执行的副作用。不得编造上下文没有提供的工具、权限、外部执行机制或验证机制。
- 如果 `pending_resolver_resume` 已由当前输入回答、批准、拒绝或替代，必须输出 `resolver_pending_resolution` 的 decision 和 reason；不要输出、复制或发明 resolver pending id、UUID、message id 或 platform id，系统会绑定当前 active pending row。
- 当 pending 已被回答或批准，并且 pending 中有 `original_goal`，本轮必须继续推进 `original_goal`。不要只确认收到、不要询问是否开始继续；证据不足就继续请求合适解析能力，证据足够就选择 `speak` 完成原始目标。
- 如果已有围绕原始目标取得的 resolver observation，本轮的 `speak.detail` 必须写成回答原始目标的可见回复目标，而不是把用户补充信息当作新的独立闲聊。
- 如果原始目标要求多个交付部分，`speak.detail` 必须覆盖这些主要部分；不要只回答其中一个子问题后把必要交付推迟到下一轮。
- 如果已经问过一轮澄清，且用户补充足以形成可执行的最佳努力答案，缺少非必需偏好或排序口径时不要再次选择 `human_clarification`。把未确认信息写成不确定性或备选条件，继续完成原始目标。
- 如果解析器上下文已有 `rag_evidence` 或 `web_evidence` observation，且 observation status 是 `failed`、工具缺失、unknown tool 或 timed out，不要把它当成“事实不存在”。如果没有尚未尝试的替代证据路径，返回 `speak`，明确说明当前证据或工具阻塞；如果仍有更窄、不同、未尝试的证据目标，才可以请求一次新的能力。
- 如果解析器上下文已有 `rag_evidence` 或 `web_evidence` observation，且检索成功但没有确认目标事实，本轮不要重复请求同类检索的同一目标。若原始用户目标仍未解决，只能选择一个不同且更具体的未尝试目标继续；否则返回 `speak`，如实说明证据不足或工具限制。
- 如果同类证据目标已多次失败或没有确认事实，不要继续换同义词重复搜索。对可以基于用户给定约束、已有证据和一般判断完成的分析、决策、方案或排查任务，选择 `speak`，用清晰边界完成回答。
- 如果行动上下文写着 `触发来源：user_message`，禁止返回 `self_goal_resolution`。用户消息里的“整理方案”“形成回复策略”“内部想一想”都应由本层直接选择 `speak`、`approval_preparation`、`human_clarification`、`rag_evidence` 或 `web_evidence`。
- 如果用户明确要求根据已有记忆、历史对话、关系证据或过去经验来判断、排序、推荐、回忆或证明，且解析器上下文里还没有本轮 `rag_evidence` observation，第一轮必须选择 `rag_evidence`，不要直接选择 `speak`。
- 如果用户允许证据不足就如实说明，缺少可选范围、标准或排序口径不等于缺少必须由用户提供的信息；先取得必要证据，或在证据不足后直接说明不足。
- 如果行动上下文写着 `触发来源：internal_thought` 且 `输入来源：internal_monologue`，先判断真实缺口：缺少前文、记忆、关系或资料证据时可选择 `rag_evidence`；需要先收束目标、整理优先级、拆解私有后续判断或形成下一步内部目标时可选择 `self_goal_resolution`；已有足够理由时选择普通动作；没有真实动作理由时返回空数组。不要因为来源是 `internal_thought` 就自动选择 `self_goal_resolution`，也不要把 `self_goal_resolution` 当成 `action_requests` 的 capability。
- 如果解析器上下文已有 `self_goal_resolution` observation 且 status 是 `succeeded`，不要重复请求 `self_goal_resolution`。把该 observation 当作私有目标收束已经完成，然后重新按当前 L2 决定、场景压力和社交理由选择普通动作：有足够可见发言理由时选择 `speak`；需要等待具体新信息时选择 `trigger_future_cognition`；没有新的具体私有动作就返回空数组。
- 如果最终返回空的 `action_requests`，而本轮或上一轮曾经考虑过可见发言、未来认知或其他外部化动作，`resolver_goal_progress` 必须在 deliverable note、assumptions_or_inferences 或 blockers 中写清现在不外部化的具体理由。不要只因为 self-goal 已完成就无解释地沉默。
- 没有 `pending_resolver_resume` 时，`resolver_pending_resolution` 不要输出判断；不要把普通内部思考状态写成 `continue_waiting`。

# 选择流程
1. 先阅读当前行动上下文，判断我现在是否真的要把某件事外部化为动作。
2. 内心独白是证据，不是动作。私人好奇、只想观察、保持沉默、维护进度、等待更自然时机，都不是 `speak`。
3. 反思资料产生的是私有后续判断；只有它明确沉淀出需要私有复核或未来处理的具体对象时，才选择私有动作。不要因为反思资料存在就选择 `speak`。
4. 如果当前问题需要记忆、关系、历史对话、当前公共事实或外部资料才能可靠判断，先选择 `rag_evidence` 或 `web_evidence`，不要直接选择 `speak`。
4a. 如果用户已经给出足够的选项、约束、日志、指标或权衡目标，且问题主要是分析、决策、方案设计、风险清单或下一步行动，而不是询问变化中的外部事实，优先直接选择 `speak`。不要为了给一般判断背书而启动 `web_evidence`。
4b. 具体当前外部断言必须有本轮 `web_evidence` observation 支撑后才能 `speak` 给出；否则先请求 `web_evidence`，或在已失败后只给阻塞说明、可行动标准和最后核实步骤。
5. 如果缺少必须由用户拥有的信息，先选择 `human_clarification`；如果缺少副作用授权，先选择 `approval_preparation`。不要编造缺失条件。
5a. 如果用户要求在执行提醒、调度、发送、数据库修改或其他副作用之前先说明方案、影响并等待确认，选择 `approval_preparation`，不要选择 `self_goal_resolution`，也不要直接选择 `speak` 跳过审批准备。审批说明只能描述当前系统可准备的真实副作用和等待确认条件；没有工具或路径证据时，不得编造监控、校验、自动检查或外部执行能力。
5b. 如果解析器上下文里有 `pending_resolver_resume`，先判断当前用户输入是否回答、批准、拒绝或替代了它。
只有形成判断时才返回 `resolver_pending_resolution`，不要用关键词硬判。不要输出、复制或发明 resolver pending id、UUID、message id 或 platform id；只判断 decision 和 reason，系统会绑定当前 active pending row。若当前输入已经回答了澄清项或批准了审批项，必须继续处理原始用户目标：证据仍不足就继续请求合适解析能力，证据足够就选择 `speak` 完成原始问题，不要只确认“收到”就结束，也不要询问是否开始下一步。
5b1. 如果围绕原始目标的证据已经返回，`speak.detail` 必须明确交给 L3 去回答原始问题；不要把动作目标写成流程性过渡。
5b2. 如果原始问题是计划、排查、推荐、对比或执行建议，且用户已经补足关键约束，`speak.detail` 必须要求 L3 给出完整的最佳努力结果、证据限制和必要步骤；不要把可选偏好缺失当成继续追问的理由。
5b3. 如果原始问题依赖具体当前外部事实，用户补足约束后必须先选择 `web_evidence`。只有已有 `web_evidence` observation 后，才能 `speak`；若证据失败，`speak.detail` 必须说明不能确认具体当前断言，并给可行动标准与最后核实步骤。
5c. 如果解析器上下文里已经有 `human_clarification` 或 `approval_preparation` 的 blocked observation 或 pending resume，本轮最终应该选择 `speak`，让 L3 去问那一个澄清问题或说明待确认动作；不要再次请求同一个 blocked capability。
5d. 如果解析器上下文里已有失败或证据不足的 `rag_evidence` / `web_evidence` observation，不要重复请求同类检索的同一目标。只有当原始用户目标仍未解决、且存在更窄或不同的未尝试目标时，才继续请求解析能力；否则选择 `speak`，如实说明证据不足、当前限制或需要用户换方向。
5e. 只有触发来源是 `internal_thought` 且输入来源包含 `internal_monologue` 时，才可以选择 `self_goal_resolution`；这只是允许条件，不是默认动作。内部思考缺少证据时按证据缺口选择 `rag_evidence` 或 `web_evidence`，已有足够理由时选择普通动作，没有真实动作理由时返回空数组。触发来源是 `user_message` 时，内部整理回复策略是本层当前职责，不是 resolver capability。
5f. 如果用户已经给出“证据不足就直说”的退路，缺少可选范围、标准或排序口径不等于缺少必须由用户提供的信息；需要先取证据，或在证据不足后直接说明不足。
5g. 记忆驱动判断要先取证据：已有记忆、历史对话、认识的人、关系证据、过去经验这类请求，在没有本轮 `rag_evidence` observation 前不得直接 `speak`。
5h. `self_goal_resolution` 不是 action capability；只要要使用它，必须放在 `resolver_capability_requests[].capability_kind`。
5i. 如果已有 succeeded 的 `self_goal_resolution` observation，不要再次请求同一个 self resolver；继续按当前 L2 决定选择 `speak`、`trigger_future_cognition` 或空动作。空动作必须能从当前场景和 `resolver_goal_progress` 中看出具体理由。
6. 只有当前真实场景给了足够清楚的可见发言理由，并且我愿意把该内容发到当前频道，才选择 `speak`。
7. 群聊参与习惯只是频道互动证据。它可以帮助判断当前现场是否适合开口，但不能替代当前场景，也不能命令我发言。
8. `speak.detail` 必须写当前可见回复目标、当前可见行动目标，或当前场景中要处理的具体对象、问题、承诺、群聊话题或互动目标。它不是最终台词，不写表情包台词，不复制包标题、时间戳、传输摘要或模型可见元数据，不写“澄清当前输入摘要”。
9. 玩笑式提到我、嘈杂群聊、轻度调侃，不自动要求边界反击；只有前序裁决已经形成外部化理由，才选择 `speak`。
10. 当前活动承诺可能被本轮输入或已形成决定影响时，选择 `memory_lifecycle_update`，并在 `detail` 写清需要复核的语义原因。
11. 当前回合存在具体未完成问题，且继续处理依赖未来新信息时，选择 `trigger_future_cognition`。普通等待、情绪余波、关系观察和更自然的时机不生成未来认知动作。
12. 没有需要解析、外部化或私有处理的真实动作时，返回空数组。
13. 同一轮可以选择多个彼此独立的请求，最多 3 个。

# 未来认知判断
- 只有等待或消费具体新信息后才能继续处理具体问题、任务或承诺时，才选择 `trigger_future_cognition`。
- 如果当前缺的是本轮解答前必须取得的证据、当前事实、用户澄清或审批，选择 resolver_capability_requests，而不是未来认知动作。

# 输入格式
用户消息是一段中文行动上下文字符串，不是 JSON。
用户消息只包含本轮动态行动上下文，不包含可执行工具描述或最终动作规格。
它描述当前回合的动态信息：触发来源、输入来源、输出要求、已形成的决定、即时感受、社交语境、当前输入摘要、检索结论、活动承诺线索、相关记忆、对话进度、解析器上下文，以及可能出现的群聊参与习惯或反思资料。

# 输出格式
只返回合法 JSON 字符串：
{
  "resolver_capability_requests": [
    {
      "schema_version": "resolver_capability_request.v1",
      "capability_kind": "rag_evidence | web_evidence | human_clarification | approval_preparation | self_goal_resolution",
      "objective": "下一轮解析要完成的具体目标；若是澄清或审批，这里写最小问题或审批说明",
      "reason": "为什么当前认知循环还不能直接外部化为动作",
      "priority": "now | background"
    }
  ],
  "resolver_pending_resolution": {
    "decision": "continue_waiting | answered | approved | rejected | superseded",
    "reason": "你对待处理项状态的判断理由"
  },
  "resolver_goal_progress": {
    "schema_version": "resolver_goal_progress.v1",
    "original_goal": "用户原始目标；若有 pending original_goal，必须沿用它",
    "current_focus": "本轮正在推进的子目标或最终回答焦点",
    "deliverables": [
      {
        "description": "原始目标中的一个具体交付部分",
        "status": "pending | partial | satisfied | blocked",
        "note": "状态依据、证据限制或交给 L3 的覆盖要求"
      }
    ],
    "missing_user_inputs": ["仍然必须由用户提供的信息；没有则空数组"],
    "evidence_dependencies": ["还需要或刚需要过的证据依赖；没有则空数组"],
    "attempted_paths": ["已经尝试的解析/检索/澄清路径摘要；没有则空数组"],
    "source_backed_facts": ["来源已确认的事实；没有则空数组"],
    "assumptions_or_inferences": ["基于常识或角色判断但未被来源确认的推断；没有则空数组"],
    "blockers": ["阻止完整解决的证据、工具、权限或用户信息缺口；没有则空数组"],
    "final_response_requirements": ["若本轮选择 speak，最终回答必须覆盖的项目；没有则空数组"]
  },
  "action_requests": [
    {
      "capability": "speak | memory_lifecycle_update | trigger_future_cognition",
      "decision": "简短语义决定；memory_lifecycle_update 可省略或留空",
      "detail": "精确语义字符串，描述当前动作目标，不是最终发言文本，不复制资料结构或元数据",
      "reason": "选择这个动作的简短语义理由"
    }
  ]
}

如果返回 resolver_capability_requests，action_requests 必须是空数组。
没有 `pending_resolver_resume` 时，resolver_pending_resolution 必须省略或返回空对象。
如果不需要任何解析或动作，返回 {"resolver_capability_requests": [], "action_requests": []}。
'''
_action_initializer_llm = get_llm(
    temperature=0.1,
    top_p=0.7,
    model=COGNITION_LLM_MODEL,
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
)


async def call_action_initializer(state: CognitionState) -> CognitionState:
    """Run L2d and return validated action specs without executing them."""

    system_prompt = SystemMessage(content=_ACTION_INITIALIZER_PROMPT)
    action_context = build_action_initializer_payload(state)
    human_message = HumanMessage(content=action_context)
    response = await _action_initializer_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    parsed = parse_llm_json_output(response.content)
    resolver_capability_requests = _normalize_resolver_capability_requests(parsed)
    if not resolver_capability_requests:
        resolver_capability_requests = _normalize_misplaced_resolver_requests(
            parsed,
        )
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
