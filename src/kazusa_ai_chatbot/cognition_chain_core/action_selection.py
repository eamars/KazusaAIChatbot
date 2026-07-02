"""Route-only action selection for the reusable cognition chain."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ALLOWED_RESOLVER_CAPABILITIES,
    RESOLVER_CAPABILITY_REQUEST_VERSION,
    ResolverValidationError,
    project_pending_resume_for_cognition,
)
from kazusa_ai_chatbot.cognition_chain_core.action_selection_prompt import (
    ACTION_ROUTER_PROMPT,
)
from kazusa_ai_chatbot.cognition_chain_core.utils import parse_llm_json_output
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMInvoker
from kazusa_ai_chatbot import llm_tracing

logger = logging.getLogger(__name__)

ACTION_REQUEST_CAP = 3
OPEN_GOAL_DELIVERABLE_STATUSES = ("pending", "partial", "blocked")
BACKGROUND_WORK_REQUEST_CAPABILITY = "background_work_request"
ACCEPTED_TASK_REQUEST_CAPABILITY = "accepted_task_request"
ACCEPTED_TASK_STATUS_CHECK_CAPABILITY = "accepted_task_status_check"
MEMORY_LIFECYCLE_UPDATE_CAPABILITY = "memory_lifecycle_update"
SPEAK_CAPABILITY = "speak"
TRIGGER_FUTURE_COGNITION_CAPABILITY = "trigger_future_cognition"
FUTURE_SPEAK_CAPABILITY = "future_speak"
ALLOWED_ACTION_CAPABILITIES = frozenset((
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
    FUTURE_SPEAK_CAPABILITY,
    ACCEPTED_TASK_REQUEST_CAPABILITY,
    ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
))

_ACCEPTED_TASK_FORBIDDEN_FIELDS = frozenset((
    "task_brief",
    "worker",
    "task_type",
    "tool_args",
    "work_kind",
    "artifact_text",
    "file_path",
))

_RESOLVER_FORBIDDEN_FIELDS = frozenset((
    "schema_version",
    "resume_id",
    "pending_row_id",
    "resolver_id",
))

_RESOLVER_AFFORDANCE_DESCRIPTIONS = {
    "approval_preparation": (
        "Prepare a minimal approval question before an allowed side effect."
    ),
    "human_clarification": (
        "Ask the user for one missing piece of information they control."
    ),
    "local_context_recall": (
        "Retrieve local/private context when the missing answer depends on "
        "Kazusa state: persona, user memory, relationship, prior conversation, "
        "commitments, profile, or a local nickname/reference."
    ),
    "public_answer_research": (
        "Investigate public/current/external/source-checkable answer gaps: "
        "public term meanings, product facts, news, docs, comparisons, or "
        "other internet-researchable facts."
    ),
    "self_goal_resolution": (
        "Resolve or prioritize an internal self-cognition goal."
    ),
}


def build_action_selection_payload(
    state: Mapping[str, object],
    capabilities: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the prompt-safe JSON payload for route-only action selection."""

    action_affordances = _available_action_affordances(state, capabilities)
    active_commitments = _project_active_commitments(state)
    if not active_commitments:
        action_affordances = [
            affordance for affordance in action_affordances
            if affordance["capability"] != MEMORY_LIFECYCLE_UPDATE_CAPABILITY
        ]

    payload: dict[str, object] = {
        "source": _build_source_section(state),
        "current_input": _build_current_input_section(state),
        "cognition": _build_cognition_section(state),
        "evidence": _build_evidence_section(state, active_commitments),
        "resolver": _build_resolver_section(state),
        "capabilities": _build_capabilities_section(action_affordances),
        "work_seed": _build_work_seed_section(state),
    }
    group_engagement = _build_group_engagement_section(state)
    if group_engagement is not None:
        payload["group_engagement"] = group_engagement
    return payload


def build_action_selection_payload_text(
    state: Mapping[str, object],
    capabilities: Mapping[str, object] | None = None,
) -> str:
    """Build the serialized prompt-safe action-selection human payload."""

    payload = build_action_selection_payload(state, capabilities)
    payload_text = json.dumps(payload, ensure_ascii=False, indent=None)
    return payload_text


def build_action_selection_messages(
    state: Mapping[str, object],
    capabilities: Mapping[str, object] | None = None,
) -> list[BaseMessage]:
    """Build the system and human messages for one action-selection call."""

    human_payload = build_action_selection_payload_text(state, capabilities)
    messages: list[BaseMessage] = [
        SystemMessage(content=ACTION_ROUTER_PROMPT),
        HumanMessage(content=human_payload),
    ]
    return messages


async def route_action_requests(
    llm: LLMInvoker,
    state: Mapping[str, object],
    *,
    config: LLMCallConfig,
    capabilities: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Call the route-selection LLM and normalize its semantic output."""

    messages = build_action_selection_messages(state, capabilities)
    started_at = time.perf_counter()
    response = await llm.ainvoke(messages, config=config)
    raw_content = getattr(response, "content", None)
    if not isinstance(raw_content, str):
        raise TypeError("Action selection LLM response content must be text")
    raw_parsed = parse_llm_json_output(raw_content)
    parsed = normalize_action_selection_output(
        raw_parsed,
        max_action_requests=_request_cap(state, "max_action_requests"),
        max_resolver_requests=_request_cap(state, "max_resolver_requests"),
    )
    parsed["semantic_action_requests"] = _filter_action_requests_to_affordances(
        parsed.get("semantic_action_requests"),
        state,
        capabilities,
    )
    trace_id = _safe_text(state.get("llm_trace_id"))
    await llm_tracing.record_llm_trace_step(
        trace_id=trace_id,
        stage_name="l2d_action_selection",
        route_name=config.route_name,
        model_name=config.model,
        messages=messages,
        response_text=raw_content,
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        duration_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
        output_state_fields=[
            "semantic_action_requests",
            "resolver_capability_requests",
            "resolver_pending_resolution",
            "resolver_goal_progress",
        ],
    )
    return parsed


def normalize_action_selection_output(
    raw: object,
    *,
    max_action_requests: int = ACTION_REQUEST_CAP,
    max_resolver_requests: int = ACTION_REQUEST_CAP,
) -> dict[str, object]:
    """Normalize raw route-selection output into core semantic contracts."""

    if not isinstance(raw, dict):
        return_value = {
            "resolver_capability_requests": [],
            "resolver_pending_resolution": None,
            "resolver_goal_progress": None,
            "semantic_action_requests": [],
        }
        return return_value

    resolver_capability_requests = _normalize_resolver_requests(
        raw.get("resolver_capability_requests"),
        max_resolver_requests=max_resolver_requests,
    )
    resolver_pending_resolution = _normalize_pending_resolution(
        raw.get("resolver_pending_resolution"),
    )
    resolver_goal_progress = _normalize_goal_progress(
        raw.get("resolver_goal_progress"),
    )
    semantic_action_requests = _normalize_semantic_action_requests(
        raw.get("action_requests"),
        max_action_requests=max_action_requests,
    )
    return_value = {
        "resolver_capability_requests": resolver_capability_requests,
        "resolver_pending_resolution": resolver_pending_resolution,
        "resolver_goal_progress": resolver_goal_progress,
        "semantic_action_requests": semantic_action_requests,
    }
    return return_value


def _normalize_resolver_requests(
    raw_requests: object,
    *,
    max_resolver_requests: int,
) -> list[dict[str, object]]:
    """Strip forbidden fields from resolver capability requests."""

    normalized: list[dict[str, object]] = []
    if not isinstance(raw_requests, list):
        return normalized

    for raw in raw_requests:
        if not isinstance(raw, dict):
            continue
        capability_kind = _text_field(raw, "capability_kind")
        if capability_kind not in ALLOWED_RESOLVER_CAPABILITIES:
            logger.warning(
                f"Action selection dropped unsupported resolver capability: "
                f"{capability_kind}"
            )
            continue
        objective = _text_field(raw, "objective")
        reason = _text_field(raw, "reason")
        priority = _text_field(raw, "priority")
        if not objective or not reason:
            logger.warning(
                "Action selection dropped resolver request without objective "
                "or reason"
            )
            continue
        if priority not in ("now", "background"):
            priority = "now"
        cleaned: dict[str, object] = {
            "schema_version": RESOLVER_CAPABILITY_REQUEST_VERSION,
            "capability_kind": capability_kind,
            "objective": objective,
            "reason": reason,
            "priority": priority,
        }
        normalized.append(cleaned)
        if len(normalized) >= max_resolver_requests:
            break
    return normalized


def _normalize_pending_resolution(raw: object) -> dict[str, object] | None:
    """Pass through pending resolution if structurally valid."""

    if not isinstance(raw, dict):
        return_value = None
        return return_value
    decision = _text_field(raw, "decision")
    if not decision:
        return_value = None
        return return_value
    return_value = {
        "decision": decision,
        "reason": _text_field(raw, "reason"),
    }
    return return_value


def _normalize_goal_progress(raw: object) -> dict[str, object] | None:
    """Pass through goal progress without model-supplied metadata."""

    if not isinstance(raw, dict):
        return_value = None
        return return_value
    cleaned = {
        key: value
        for key, value in raw.items()
        if key not in _RESOLVER_FORBIDDEN_FIELDS
    }
    return_value: dict[str, object] = cleaned
    return return_value


def _normalize_semantic_action_requests(
    raw_requests: object,
    *,
    max_action_requests: int,
) -> list[dict[str, object]]:
    """Strip forbidden fields from semantic action requests."""

    normalized: list[dict[str, object]] = []
    if not isinstance(raw_requests, list):
        return normalized

    for raw in raw_requests:
        if not isinstance(raw, dict):
            continue
        capability = _text_field(raw, "capability")
        reason = _text_field(raw, "reason")
        if not capability or not reason:
            logger.warning(
                "Action selection dropped request without capability or reason"
            )
            continue
        if capability not in ALLOWED_ACTION_CAPABILITIES:
            logger.warning(
                f"Action selection dropped unsupported action capability: "
                f"{capability}"
            )
            continue
        cleaned: dict[str, object] = {
            "capability": capability,
            "reason": reason,
            "decision": _text_field(raw, "decision"),
            "detail": _text_field(raw, "detail"),
        }
        if capability == ACCEPTED_TASK_REQUEST_CAPABILITY:
            for forbidden in _ACCEPTED_TASK_FORBIDDEN_FIELDS:
                cleaned.pop(forbidden, None)
        normalized.append(cleaned)
        if len(normalized) >= max_action_requests:
            break
    return normalized


def _build_source_section(state: Mapping[str, object]) -> dict[str, object]:
    """Build the source section from cognition state."""

    episode = state.get("cognitive_episode")
    trigger_source = ""
    input_sources = ""
    output_mode = ""
    if isinstance(episode, Mapping):
        trigger_source = _safe_text(episode.get("trigger_source"))
        raw_sources = episode.get("input_sources")
        if isinstance(raw_sources, list):
            input_sources = ", ".join(str(source) for source in raw_sources)
        output_mode = _safe_text(episode.get("output_mode"))

    section: dict[str, object] = {
        "trigger_source": trigger_source,
        "input_sources": input_sources,
        "output_mode": output_mode,
        "channel_type": _safe_text(state.get("channel_type")),
    }
    return section


def _build_current_input_section(state: Mapping[str, object]) -> dict[str, object]:
    """Build the current input section from cognition state."""

    section: dict[str, object] = {
        "decontextualized_input": _safe_text(
            state.get("decontexualized_input")
        ),
        "media_summary": _safe_text(state.get("media_summary")),
    }
    return section


def _build_cognition_section(state: Mapping[str, object]) -> dict[str, object]:
    """Build the cognition section from upstream judgment results."""

    section: dict[str, object] = {
        "logical_stance": _safe_text(state.get("logical_stance")),
        "character_intent": _safe_text(state.get("character_intent")),
        "judgment_note": _safe_text(state.get("judgment_note")),
        "internal_monologue": _safe_text(state.get("internal_monologue")),
        "emotional_appraisal": _safe_text(state.get("emotional_appraisal")),
        "interaction_subtext": _safe_text(state.get("interaction_subtext")),
        "boundary_core_assessment": _safe_mapping(
            state.get("boundary_core_assessment")
        ),
        "social_distance": _safe_text(state.get("social_distance")),
        "emotional_intensity": _safe_text(state.get("emotional_intensity")),
        "vibe_check": _safe_text(state.get("vibe_check")),
        "relational_dynamic": _safe_text(state.get("relational_dynamic")),
    }
    return section


def _build_evidence_section(
    state: Mapping[str, object],
    active_commitments: list[dict[str, object]],
) -> dict[str, object]:
    """Build the evidence section with prompt-safe projections."""

    rag_result = state.get("rag_result")
    rag_answer = ""
    memory_evidence: list[dict[str, object]] = []
    if isinstance(rag_result, dict):
        raw_answer = rag_result.get("answer")
        if isinstance(raw_answer, str):
            rag_answer = raw_answer
        raw_mem = rag_result.get("memory_evidence")
        if isinstance(raw_mem, list):
            memory_evidence = _project_memory_evidence(raw_mem)

    conversation_progress = state.get("conversation_progress")
    if not isinstance(conversation_progress, dict):
        conversation_progress = {}

    section: dict[str, object] = {
        "rag_answer": rag_answer,
        "memory_evidence": memory_evidence,
        "conversation_progress": conversation_progress,
        "active_commitment_clues": active_commitments,
    }
    return section


def _build_resolver_section(state: Mapping[str, object]) -> dict[str, object]:
    """Build the resolver section from cognition state."""

    pending_resume = None
    raw_pending = state.get("pending_resolver_resume")
    if isinstance(raw_pending, dict):
        try:
            pending_text = project_pending_resume_for_cognition(raw_pending)
        except ResolverValidationError:
            pending_text = ""
        if pending_text:
            pending_resume = pending_text

    resolver_context = ""
    raw_context = state.get("resolver_context")
    if isinstance(raw_context, str) and raw_context.strip():
        resolver_context = raw_context.strip()

    section: dict[str, object] = {
        "pending_resume": pending_resume,
        "resolver_context": resolver_context,
    }
    return section


def _build_capabilities_section(
    action_affordances: list[dict[str, object]],
) -> dict[str, object]:
    """Build the capabilities section from caller-provided affordances."""

    resolver_affordances = _project_resolver_affordances()
    section: dict[str, object] = {
        "resolver_affordances": resolver_affordances,
        "action_affordances": action_affordances,
    }
    return section


def _project_resolver_affordances() -> list[dict[str, object]]:
    """Return prompt-safe resolver capability affordances."""

    affordances: list[dict[str, object]] = []
    for capability_kind in sorted(ALLOWED_RESOLVER_CAPABILITIES):
        affordance: dict[str, object] = {
            "capability_kind": capability_kind,
            "available": True,
            "semantic_input_summary": (
                _RESOLVER_AFFORDANCE_DESCRIPTIONS.get(capability_kind, "")
            ),
        }
        affordances.append(affordance)
    return affordances


def _project_memory_evidence(
    raw_evidence: list[object],
) -> list[dict[str, object]]:
    """Project memory evidence without storage identifiers."""

    projected: list[dict[str, object]] = []
    for raw in raw_evidence:
        if not isinstance(raw, dict):
            continue
        entry: dict[str, object] = {}
        for field in ("summary", "fact", "excerpt", "due_at", "due_state"):
            value = raw.get(field)
            if isinstance(value, str) and value.strip():
                entry[field] = value
        if entry:
            projected.append(entry)
    return projected


def _project_active_commitments(
    state: Mapping[str, object],
) -> list[dict[str, object]]:
    """Project active commitments without persistence identifiers."""

    rag_result = state.get("rag_result")
    if not isinstance(rag_result, dict):
        return_value: list[dict[str, object]] = []
        return return_value
    raw_user_image = rag_result.get("user_image")
    if not isinstance(raw_user_image, dict):
        return_value = []
        return return_value
    memory_context = raw_user_image.get("user_memory_context")
    if not isinstance(memory_context, dict):
        return_value = []
        return return_value
    raw_commitments = memory_context.get("active_commitments")
    if not isinstance(raw_commitments, list):
        return_value = []
        return return_value

    projected: list[dict[str, object]] = []
    for raw in raw_commitments:
        if not isinstance(raw, dict):
            continue
        entry: dict[str, object] = {}
        for field in ("fact", "summary", "due_at", "due_state", "status"):
            value = raw.get(field)
            if isinstance(value, str) and value.strip():
                entry[field] = value
        if entry:
            projected.append(entry)
    return projected


def _build_group_engagement_section(
    state: Mapping[str, object],
) -> dict[str, object] | None:
    """Build group engagement context when applicable."""

    if _safe_text(state.get("channel_type")) != "group":
        return_value = None
        return return_value

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        return_value = None
        return return_value
    trigger_source = episode.get("trigger_source")
    input_sources = episode.get("input_sources")
    is_internal = (
        trigger_source == "internal_thought"
        and isinstance(input_sources, list)
        and "internal_monologue" in input_sources
    )
    if not is_internal:
        return_value = None
        return return_value

    raw_context = state.get("group_engagement_action_context")
    if not isinstance(raw_context, Mapping):
        return_value = {"engagement_guidelines": [], "confidence": ""}
        return return_value

    raw_guidelines = raw_context.get("engagement_guidelines")
    guidelines: list[str] = []
    if isinstance(raw_guidelines, list):
        guidelines = [
            item.strip()
            for item in raw_guidelines
            if isinstance(item, str) and item.strip()
        ]

    confidence = _safe_text(raw_context.get("confidence"))
    section: dict[str, object] = {
        "engagement_guidelines": guidelines,
        "confidence": confidence,
    }
    return section


def _build_work_seed_section(state: Mapping[str, object]) -> dict[str, object]:
    """Build the prompt-visible seed copied into materialization."""

    source_summary = _safe_text(state.get("decontexualized_input"))
    max_output_chars = state.get("background_work_output_char_limit")
    if not isinstance(max_output_chars, int) or max_output_chars < 1:
        max_output_chars = 4000
    section: dict[str, object] = {
        "accepted_task_allowed": True,
        "source_summary": source_summary,
        "max_output_chars": max_output_chars,
    }
    return section


def _available_action_affordances(
    state: Mapping[str, object],
    capabilities: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    """Return caller-supplied action affordances for prompt selection."""

    raw_affordances = state.get("available_action_affordances")
    if isinstance(raw_affordances, list):
        projected = _project_supplied_affordances(raw_affordances)
        return projected
    if capabilities is not None:
        return _project_capability_names(capabilities.keys())
    return_value: list[dict[str, object]] = []
    return return_value


def _project_supplied_affordances(
    raw_affordances: list[object],
) -> list[dict[str, object]]:
    """Normalize public action affordances without registry metadata."""

    projected: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_affordances:
        if not isinstance(raw, Mapping):
            continue
        raw_capability = _safe_text(raw.get("capability"))
        capability = _model_facing_action_capability(raw_capability)
        if capability not in ALLOWED_ACTION_CAPABILITIES or capability in seen:
            continue
        summary = raw.get("semantic_input_summary", "")
        if raw_capability != capability:
            summary = _default_action_summary(capability)
        if raw.get("available") is False:
            continue
        projected.append({
            "capability": capability,
            "available": True,
            "visibility": _safe_text(raw.get("visibility")),
            "semantic_input_summary": summary,
        })
        seen.add(capability)
    return projected


def _project_capability_names(
    capability_names: Sequence[str],
) -> list[dict[str, object]]:
    """Build generic affordance rows from semantic capability names."""

    projected: list[dict[str, object]] = []
    for capability in sorted(capability_names):
        capability = _model_facing_action_capability(capability)
        if capability not in ALLOWED_ACTION_CAPABILITIES:
            continue
        projected.append({
            "capability": capability,
            "available": True,
            "visibility": _default_action_visibility(capability),
            "semantic_input_summary": _default_action_summary(capability),
        })
    return projected


def _filter_action_requests_to_affordances(
    raw_requests: object,
    state: Mapping[str, object],
    capabilities: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    """Drop semantic actions the caller did not expose to the core."""

    if not isinstance(raw_requests, list):
        return_value: list[dict[str, object]] = []
        return return_value
    allowed_capabilities = {
        affordance["capability"]
        for affordance in _available_action_affordances(state, capabilities)
    }
    filtered: list[dict[str, object]] = []
    for raw_request in raw_requests:
        if not isinstance(raw_request, Mapping):
            continue
        if raw_request.get("capability") not in allowed_capabilities:
            continue
        filtered.append(dict(raw_request))
        if len(filtered) >= _request_cap(state, "max_action_requests"):
            break
    return filtered


def _request_cap(state: Mapping[str, object], field_name: str) -> int:
    """Return a positive per-input request cap."""

    raw_cap = state.get(field_name)
    if isinstance(raw_cap, int) and raw_cap > 0:
        return_value = raw_cap
        return return_value
    return_value = ACTION_REQUEST_CAP
    return return_value


def _default_action_visibility(capability: str) -> str:
    """Return a generic prompt visibility for a semantic capability."""

    if capability == SPEAK_CAPABILITY:
        return_value = "public"
        return return_value
    return_value = "private"
    return return_value


def _default_action_summary(capability: str) -> list[str]:
    """Return a generic semantic summary for a capability name."""

    summaries: dict[str, list[str]] = {
        SPEAK_CAPABILITY: [
            "Use when the character wants a text surface to exist.",
            "Provide the semantic surface intent, not final wording.",
        ],
        MEMORY_LIFECYCLE_UPDATE_CAPABILITY: [
            "Use when active commitments need semantic lifecycle review.",
            "Provide review_kind=active_commitment_lifecycle and a short detail.",
        ],
        TRIGGER_FUTURE_COGNITION_CAPABILITY: [
            "Use when the character wants a later private cognition cycle.",
            "Provide the semantic reason and ordinary-language timing hint.",
        ],
        FUTURE_SPEAK_CAPABILITY: [
            "Use when accepting a future reminder or delayed follow-up message.",
            "Put exact configured-local YYYY-MM-DD HH:MM time in decision.",
            "Put the semantic future-speaking objective in detail.",
        ],
        ACCEPTED_TASK_REQUEST_CAPABILITY: [
            "Use when the character accepts bounded delayed text work.",
            "Pair this private request with a visible speak acknowledgement.",
        ],
        ACCEPTED_TASK_STATUS_CHECK_CAPABILITY: [
            "Use when the user asks about already accepted delayed work.",
            "Pair this private check with a visible progress answer.",
        ],
    }
    return summaries.get(capability, [])


def _model_facing_action_capability(capability: str) -> str:
    """Map internal executable action names to model-facing semantic names."""

    if capability == BACKGROUND_WORK_REQUEST_CAPABILITY:
        return_value = ACCEPTED_TASK_REQUEST_CAPABILITY
        return return_value
    return_value = capability
    return return_value


def _safe_text(value: object) -> str:
    """Return a stripped text value or empty string."""

    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _safe_mapping(value: object) -> dict[str, object]:
    """Return a mapping copied to a plain dict, or an empty dict."""

    if isinstance(value, Mapping):
        mapping_value = dict(value)
        return mapping_value
    return_value: dict[str, object] = {}
    return return_value


def _text_field(value: Mapping[str, object], key: str) -> str:
    """Read one optional text field from model output."""

    raw_value = value.get(key)
    if not isinstance(raw_value, str):
        return_value = ""
        return return_value
    return_value = raw_value.strip()
    return return_value
