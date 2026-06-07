"""Prompt-safe JSON payload builder for the action router."""

from __future__ import annotations

from collections.abc import Mapping

from kazusa_ai_chatbot.action_spec.models import CapabilitySpecV1
from kazusa_ai_chatbot.action_spec.registry import (
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    build_initial_action_capabilities,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.config import BACKGROUND_WORK_OUTPUT_CHAR_LIMIT
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ALLOWED_RESOLVER_CAPABILITIES,
    ResolverValidationError,
    project_pending_resume_for_cognition,
)

_RESOLVER_AFFORDANCE_DESCRIPTIONS = {
    "approval_preparation": (
        "Prepare a minimal approval question before an allowed side effect."
    ),
    "human_clarification": (
        "Ask the user for one missing piece of information they control."
    ),
    "rag_evidence": (
        "Retrieve chat, memory, relationship, or local knowledge evidence."
    ),
    "self_goal_resolution": (
        "Resolve or prioritize an internal self-cognition goal."
    ),
    "web_evidence": (
        "Retrieve current public or external factual evidence."
    ),
}


def build_action_router_payload(
    state: Mapping[str, object],
    capabilities: Mapping[str, CapabilitySpecV1] | None = None,
) -> dict[str, object]:
    """Build the prompt-safe JSON payload for the action router.

    Args:
        state: Cognition state after L2c judgment.
        capabilities: Optional registry override for deterministic tests.

    Returns:
        A JSON-serializable dict containing only prompt-safe semantic sections.
    """

    if capabilities is None:
        capabilities = build_initial_action_capabilities()

    prompt_capabilities = dict(capabilities)
    active_commitments = _project_active_commitments(state)
    if not active_commitments:
        prompt_capabilities.pop(MEMORY_LIFECYCLE_UPDATE_CAPABILITY, None)

    payload: dict[str, object] = {
        "source": _build_source_section(state),
        "current_input": _build_current_input_section(state),
        "cognition": _build_cognition_section(state),
        "evidence": _build_evidence_section(state, active_commitments),
        "resolver": _build_resolver_section(state),
        "capabilities": _build_capabilities_section(prompt_capabilities),
        "work_seed": _build_work_seed_section(state),
    }
    group_engagement = _build_group_engagement_section(state)
    if group_engagement is not None:
        payload["group_engagement"] = group_engagement
    return payload


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
            input_sources = ", ".join(str(s) for s in raw_sources)
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
    capabilities: Mapping[str, CapabilitySpecV1],
) -> dict[str, object]:
    """Build the capabilities section from registry projections."""

    action_affordances = project_prompt_affordances(dict(capabilities))
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
        return []
    raw_user_image = rag_result.get("user_image")
    if not isinstance(raw_user_image, dict):
        return []
    memory_context = raw_user_image.get("user_memory_context")
    if not isinstance(memory_context, dict):
        return []
    raw_commitments = memory_context.get("active_commitments")
    if not isinstance(raw_commitments, list):
        return []

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
    """Build group engagement context when applicable.

    Returns None for non-group or non-self-cognition contexts.
    """

    if _safe_text(state.get("channel_type")) != "group":
        return None

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        return None
    trigger_source = episode.get("trigger_source")
    input_sources = episode.get("input_sources")
    is_internal = (
        trigger_source == "internal_thought"
        and isinstance(input_sources, list)
        and "internal_monologue" in input_sources
    )
    if not is_internal:
        return None

    raw_context = state.get("group_engagement_action_context")
    if not isinstance(raw_context, Mapping):
        return {"engagement_guidelines": [], "confidence": ""}

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
    """Build the prompt-visible seed copied into background-work materialization."""

    source_summary = _safe_text(state.get("decontexualized_input"))
    section: dict[str, object] = {
        "background_work_allowed": True,
        "source_summary": source_summary,
        "max_output_chars": BACKGROUND_WORK_OUTPUT_CHAR_LIMIT,
    }
    return section


def _safe_text(value: object) -> str:
    """Return a stripped text value or empty string."""

    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _safe_mapping(value: object) -> dict[str, object]:
    """Return a dict or empty dict."""

    if isinstance(value, dict):
        return value
    return_value: dict[str, object] = {}
    return return_value
