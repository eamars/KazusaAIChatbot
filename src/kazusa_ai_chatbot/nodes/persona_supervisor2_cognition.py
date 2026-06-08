"""Cognition subgraph — state definition, graph wiring, and entry-point.

Agent implementations live in the layer-specific submodules:
  - persona_supervisor2_cognition_l1  (L1 subconscious)
  - persona_supervisor2_cognition_l2   (L2a/L2b/L2c1 cognition)
  - persona_supervisor2_cognition_l2c2 (L2c2 social context)
  - persona_supervisor2_cognition_l2d  (L2d action selection)
"""
import asyncio
from collections.abc import Mapping
import logging
from typing import Any

from langgraph.graph import StateGraph, START, END

from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    merge_shared_memory_prewarm_result,
    run_first_cycle_shared_memory_prewarm,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverValidationError,
    validate_resolver_capability_request,
    validate_resolver_goal_progress,
    validate_resolver_pending_resolution,
)
from kazusa_ai_chatbot.db import build_group_engagement_action_context
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    CognitionState,
    GlobalPersonaState,
)
from kazusa_ai_chatbot.utils import build_interaction_history_recent, log_preview

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import agent functions from layer submodules
# ---------------------------------------------------------------------------
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l1 import (  # noqa: E402
    call_cognition_subconscious,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2 import (  # noqa: E402
    call_cognition_consciousness,
    call_boundary_core_agent,
    call_judgment_core_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2c2 import (  # noqa: E402
    call_social_context_appraisal,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2d import (  # noqa: E402
    call_action_initializer,
)


async def call_group_engagement_action_context_loader(
    state: CognitionState,
) -> CognitionState:
    """Load group-channel engagement guidance before action selection."""

    empty_context = {"engagement_guidelines": [], "confidence": ""}
    if not _is_group_self_cognition_state(state):
        return_value = {
            "group_engagement_action_context": empty_context,
        }
        return return_value

    context = await build_group_engagement_action_context(
        channel_type=state["channel_type"],
        platform=state["platform"],
        platform_channel_id=state["platform_channel_id"],
    )
    return_value = {
        "group_engagement_action_context": context,
    }
    return return_value


def _is_group_self_cognition_state(state: CognitionState) -> bool:
    """Return whether the current cognition state is group self-cognition."""

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


async def call_cognition_subgraph(state: GlobalPersonaState) -> GlobalPersonaState:
    """Run the cognition subgraph for one persona turn.

    Args:
        state: Persona state containing current stimulus, RAG evidence, profile
            context, and optional private residue context.

    Returns:
        Cognition state updates merged back into the persona graph.
    """

    residue_context = state.get("internal_monologue_residue_context", "")
    prewarm_task: asyncio.Task[dict[str, Any]] | None = None
    resolver_state = state.get("resolver_state")
    if (
        isinstance(resolver_state, Mapping)
        and resolver_state.get("cycle_index") == 0
    ):
        prewarm_task = asyncio.create_task(
            run_first_cycle_shared_memory_prewarm(state),
        )

    async def call_l2a_conscious_framing(
        cognition_state: CognitionState,
    ) -> CognitionState:
        """Inject private residue only into the L2a consciousness node."""

        rag_result = cognition_state["rag_result"]
        if prewarm_task is not None:
            prewarm_rag_result = await prewarm_task
            rag_result = merge_shared_memory_prewarm_result(
                rag_result,
                prewarm_rag_result,
            )

        l2a_state = dict(cognition_state)
        l2a_state["internal_monologue_residue_context"] = residue_context
        l2a_state["rag_result"] = rag_result
        result = await call_cognition_consciousness(l2a_state)
        result_with_rag = dict(result)
        result_with_rag["rag_result"] = rag_result
        return_value = result_with_rag
        return return_value

    sub_agent_builder = StateGraph(CognitionState)

    sub_agent_builder.add_node("l1_subconscious", call_cognition_subconscious)
    sub_agent_builder.add_node(
        "l2a_conscious_framing",
        call_l2a_conscious_framing,
    )
    sub_agent_builder.add_node("l2b_boundary_appraisal", call_boundary_core_agent)
    sub_agent_builder.add_node("l2c1_judgment_synthesis", call_judgment_core_agent)
    sub_agent_builder.add_node(
        "l2c2_social_context_appraisal",
        call_social_context_appraisal,
    )
    sub_agent_builder.add_node(
        "group_engagement_action_context",
        call_group_engagement_action_context_loader,
    )
    sub_agent_builder.add_node("l2d_action_selection", call_action_initializer)

    # Connect
    sub_agent_builder.add_edge(START, "l1_subconscious")
    sub_agent_builder.add_edge("l1_subconscious", "l2a_conscious_framing")
    sub_agent_builder.add_edge("l1_subconscious", "l2b_boundary_appraisal")

    sub_agent_builder.add_edge(
        ["l2a_conscious_framing", "l2b_boundary_appraisal"],
        "l2c1_judgment_synthesis",
    )
    sub_agent_builder.add_edge(
        "l2b_boundary_appraisal",
        "l2c2_social_context_appraisal",
    )

    sub_agent_builder.add_edge(
        ["l2c1_judgment_synthesis", "l2c2_social_context_appraisal"],
        "group_engagement_action_context",
    )
    sub_agent_builder.add_edge(
        "group_engagement_action_context",
        "l2d_action_selection",
    )
    sub_agent_builder.add_edge("l2d_action_selection", END)

    cognition_subgraph = sub_agent_builder.compile()

    # Get attributes
    decontexualized_input = state["decontexualized_input"]
    interaction_history_recent = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )

    initial_state: CognitionState = {
        "character_profile": state["character_profile"],
        # Inputs
        "storage_timestamp_utc": state["storage_timestamp_utc"],
        "local_time_context": state["local_time_context"],
        "user_input": state["user_input"],
        "prompt_message_context": state["prompt_message_context"],
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "channel_type": state["channel_type"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "platform_bot_id": state["platform_bot_id"],
        "chat_history_recent": interaction_history_recent,
        "reply_context": state["reply_context"],
        "indirect_speech_context": state["indirect_speech_context"],
        "channel_topic": state["channel_topic"],
        "conversation_progress": state.get("conversation_progress"),
        "promoted_reflection_context": state.get("promoted_reflection_context"),

        # From previous stages
        "decontexualized_input": decontexualized_input,
        "referents": state["referents"],
        "rag_result": state["rag_result"],
        "resolver_context": state.get("resolver_context", ""),
    }
    resolver_state = state.get("resolver_state")
    if resolver_state is not None:
        initial_state["resolver_state"] = resolver_state
    pending_resolver_resume = state.get("pending_resolver_resume")
    if pending_resolver_resume is not None:
        initial_state["pending_resolver_resume"] = pending_resolver_resume
    resolver_goal_progress = state.get("resolver_goal_progress")
    if resolver_goal_progress is not None:
        initial_state["resolver_goal_progress"] = resolver_goal_progress
    cognitive_episode = state.get("cognitive_episode")
    if cognitive_episode is not None:
        initial_state["cognitive_episode"] = cognitive_episode
    
    result = await cognition_subgraph.ainvoke(initial_state)

    # Generate outputs
    internal_monologue = result.get("internal_monologue", "")
    interaction_subtext = result.get("interaction_subtext", "")
    emotional_appraisal = result.get("emotional_appraisal", "")
    character_intent = result.get("character_intent", "")
    logical_stance = result.get("logical_stance", "")
    judgment_note = result.get("judgment_note", "")
    social_distance = result["social_distance"]
    emotional_intensity = result["emotional_intensity"]
    vibe_check = result["vibe_check"]
    relational_dynamic = result["relational_dynamic"]
    action_specs = result.get("action_specs", [])
    resolver_capability_requests = _validated_resolver_capability_requests(
        result.get("resolver_capability_requests", []),
    )
    resolver_pending_resolution = _validated_resolver_pending_resolution(
        result.get("resolver_pending_resolution"),
    )
    resolver_goal_progress = _validated_resolver_goal_progress(
        result.get("resolver_goal_progress"),
    )
    rag_result = result.get("rag_result", state["rag_result"])

    logger.info(
        f"Cognition output: stance={logical_stance} "
        f"intent={character_intent} "
        f"appraisal={log_preview(emotional_appraisal)} "
        f"subtext={log_preview(interaction_subtext)} "
        f"action_specs={log_preview(action_specs)} "
        f"resolver_capabilities="
        f"{log_preview(_resolver_request_log_rows(resolver_capability_requests))} "
        f"resolver_pending_resolution="
        f"{log_preview(_pending_resolution_log_row(resolver_pending_resolution))} "
        f"monologue={log_preview(internal_monologue)}"
    )
    logger.debug(
        f'Cognition input: input={log_preview(state["decontexualized_input"])}'
    )


    return_value = {
        "internal_monologue": internal_monologue,
        "action_specs": action_specs,
        "resolver_capability_requests": resolver_capability_requests,

        # Other data used by post-dialog consolidation.
        "interaction_subtext": interaction_subtext,
        "emotional_appraisal": emotional_appraisal,
        "character_intent": character_intent,
        "logical_stance": logical_stance,
        "judgment_note": judgment_note,
        "social_distance": social_distance,
        "emotional_intensity": emotional_intensity,
        "vibe_check": vibe_check,
        "relational_dynamic": relational_dynamic,
        "rag_result": rag_result,
    }
    if resolver_pending_resolution is not None:
        return_value["resolver_pending_resolution"] = resolver_pending_resolution
    if resolver_goal_progress is not None:
        return_value["resolver_goal_progress"] = resolver_goal_progress
    return return_value


def _validated_resolver_capability_requests(
    value: object,
) -> list[dict]:
    """Validate L2d resolver requests before returning to persona graph."""

    if value is None:
        return_value: list[dict] = []
        return return_value
    if not isinstance(value, list):
        logger.warning("Cognition dropped non-list resolver capability requests")
        return_value = []
        return return_value

    validated_requests: list[dict] = []
    for raw_request in value:
        try:
            validated_request = validate_resolver_capability_request(raw_request)
        except ResolverValidationError as exc:
            logger.warning(f"Cognition dropped invalid resolver request: {exc}")
            continue
        validated_requests.append(validated_request)
    return_value = validated_requests
    return return_value


def _validated_resolver_pending_resolution(value: object) -> dict | None:
    """Validate an optional L2d pending-resolver decision."""

    if value is None:
        return_value = None
        return return_value
    try:
        validated_resolution = validate_resolver_pending_resolution(value)
    except ResolverValidationError as exc:
        logger.warning(f"Cognition dropped invalid pending resolver decision: {exc}")
        return_value = None
        return return_value
    return_value = validated_resolution
    return return_value


def _validated_resolver_goal_progress(value: object) -> dict | None:
    """Validate L2d's optional goal-progress checklist."""

    if value is None:
        return_value = None
        return return_value
    try:
        validated_progress = validate_resolver_goal_progress(value)
    except ResolverValidationError as exc:
        logger.warning(f"Cognition dropped invalid goal progress: {exc}")
        return_value = None
        return return_value
    return_value = validated_progress
    return return_value


def _resolver_request_log_rows(requests: list[dict]) -> list[dict[str, str]]:
    """Build bounded resolver request log rows without raw prompt text."""

    rows: list[dict[str, str]] = []
    for request in requests:
        rows.append({
            "capability_kind": str(request.get("capability_kind", "")),
            "priority": str(request.get("priority", "")),
            "objective": str(request.get("objective", ""))[:120],
            "reason": str(request.get("reason", ""))[:120],
        })
    return_value = rows
    return return_value


def _pending_resolution_log_row(resolution: dict | None) -> dict[str, str]:
    """Build a small log row for pending resolver closure decisions."""

    if resolution is None:
        return_value: dict[str, str] = {}
        return return_value
    return_value = {
        "decision": str(resolution.get("decision", "")),
        "reason": str(resolution.get("reason", ""))[:120],
    }
    return return_value
