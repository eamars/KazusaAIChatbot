"""Persona graph orchestration for decontextualization, RAG, cognition, and dialog."""

import logging

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.execution import execute_action_specs_for_trace
from kazusa_ai_chatbot.action_spec.registry import (
    SPEAK_CAPABILITY,
)
from kazusa_ai_chatbot.action_spec.results import (
    action_attempt_id_from_eval_result,
    build_episode_trace,
    build_private_surface_output,
    build_text_surface_output,
)
from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.config import (
    CHAT_HISTORY_RECENT_LIMIT,
    COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS,
    COGNITION_RESOLVER_ENABLED,
    COGNITION_RESOLVER_MAX_CYCLES,
)
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    execute_resolver_capability_request,
    run_rag_evidence_for_persona_state as _run_rag_evidence_for_persona_state,
)
from kazusa_ai_chatbot.cognition_resolver.loop import call_cognition_resolver_loop
from kazusa_ai_chatbot.cognition_resolver.state import ensure_initial_resolver_inputs
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import call_cognition_subgraph
from kazusa_ai_chatbot.consolidation.core import call_consolidation_subgraph
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    call_l3_text_surface_handler,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_memory_lifecycle import (
    call_memory_lifecycle_update_handler,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import (
    call_msg_decontexualizer,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    GlobalPersonaState,
    ScopeUser,
)
from kazusa_ai_chatbot.rag.quote_aware_sequence import (
    call_quote_aware_rag_supervisor,
)
from kazusa_ai_chatbot.rag.cognitive_episode_adapter import (
    build_text_chat_rag_request,
)
from kazusa_ai_chatbot.state import IMProcessState
from kazusa_ai_chatbot.time_boundary import format_storage_utc_history_for_llm
from kazusa_ai_chatbot.utils import (
    build_interaction_history_recent,
    text_or_empty,
)

logger = logging.getLogger(__name__)

PERSONA_RAG_COMPONENT = "nodes.persona_supervisor2"


def _find_scope_user_index(
    scope_users: list[ScopeUser],
    *,
    display_name: str,
    platform_user_id: str,
    global_user_id: str,
) -> int | None:
    """Find a matching roster row by stable identity priority.

    Args:
        scope_users: Existing roster rows in insertion order.
        display_name: Clean display name for the incoming identity.
        platform_user_id: Clean platform user id for the incoming identity.
        global_user_id: Clean global user id for the incoming identity.

    Returns:
        Matching row index, or ``None`` when no stable identity matches.
    """

    if global_user_id:
        for index, scope_user in enumerate(scope_users):
            if scope_user["global_user_id"] == global_user_id:
                return_value = index
                return return_value

    if platform_user_id:
        for index, scope_user in enumerate(scope_users):
            if scope_user["platform_user_id"] == platform_user_id:
                return_value = index
                return return_value

    if display_name and not global_user_id and not platform_user_id:
        for index, scope_user in enumerate(scope_users):
            if scope_user["display_name"] == display_name:
                return_value = index
                return return_value

    return_value = None
    return return_value


def _add_scope_user(
    scope_users: list[ScopeUser],
    *,
    display_name: object,
    platform_user_id: object,
    global_user_id: object,
) -> None:
    """Add or merge one neutral identity row into the scoped-user roster.

    Args:
        scope_users: Mutable scoped-user roster.
        display_name: Raw display name from an already-loaded context source.
        platform_user_id: Raw platform user id from an already-loaded context.
        global_user_id: Raw global user id from an already-loaded context.
    """

    clean_display_name = text_or_empty(display_name)
    clean_platform_user_id = text_or_empty(platform_user_id)
    clean_global_user_id = text_or_empty(global_user_id)
    has_identity = any((
        clean_display_name,
        clean_platform_user_id,
        clean_global_user_id,
    ))
    if not has_identity:
        return

    existing_index = _find_scope_user_index(
        scope_users,
        display_name=clean_display_name,
        platform_user_id=clean_platform_user_id,
        global_user_id=clean_global_user_id,
    )
    if existing_index is None:
        scope_users.append({
            "display_name": clean_display_name,
            "platform_user_id": clean_platform_user_id,
            "global_user_id": clean_global_user_id,
            "aliases": [],
        })
        return

    scope_user = scope_users[existing_index]
    if clean_display_name:
        scope_user["display_name"] = clean_display_name
    if clean_platform_user_id and not scope_user["platform_user_id"]:
        scope_user["platform_user_id"] = clean_platform_user_id
    if clean_global_user_id and not scope_user["global_user_id"]:
        scope_user["global_user_id"] = clean_global_user_id


def _build_scope_users(
    state: IMProcessState,
    channel_history: list[dict],
) -> list[ScopeUser]:
    """Build the neutral identity roster visible to decontextualization.

    Args:
        state: Current top-level chat graph state after relevance gating.
        channel_history: Already-loaded recent channel history prepared for
            decontextualizer use.

    Returns:
        Deduplicated neutral identity rows. Rows contain only display name,
        platform id, global id, and aliases.
    """

    scope_users: list[ScopeUser] = []
    for row in channel_history:
        if not isinstance(row, dict):
            continue
        display_name = row.get("display_name")
        if not display_name:
            display_name = row.get("name")
        platform_user_id = row.get("platform_user_id")
        global_user_id = row.get("global_user_id")
        _add_scope_user(
            scope_users,
            display_name=display_name,
            platform_user_id=platform_user_id,
            global_user_id=global_user_id,
        )

    character_profile = state["character_profile"]
    _add_scope_user(
        scope_users,
        display_name=character_profile["name"],
        platform_user_id=state["platform_bot_id"],
        global_user_id=character_profile["global_user_id"],
    )
    _add_scope_user(
        scope_users,
        display_name=state["user_name"],
        platform_user_id=state["platform_user_id"],
        global_user_id=state["global_user_id"],
    )

    prompt_message_context = state["prompt_message_context"]
    for mention in prompt_message_context["mentions"]:
        if not isinstance(mention, dict):
            continue
        display_name = mention.get("display_name")
        platform_user_id = mention.get("platform_user_id")
        global_user_id = mention.get("global_user_id")
        _add_scope_user(
            scope_users,
            display_name=display_name,
            platform_user_id=platform_user_id,
            global_user_id=global_user_id,
        )

    for addressed_global_user_id in prompt_message_context[
        "addressed_to_global_user_ids"
    ]:
        _add_scope_user(
            scope_users,
            display_name="",
            platform_user_id="",
            global_user_id=addressed_global_user_id,
        )

    prompt_reply = prompt_message_context.get("reply")
    if isinstance(prompt_reply, dict):
        display_name = prompt_reply.get("display_name")
        platform_user_id = prompt_reply.get("platform_user_id")
        global_user_id = prompt_reply.get("global_user_id")
        _add_scope_user(
            scope_users,
            display_name=display_name,
            platform_user_id=platform_user_id,
            global_user_id=global_user_id,
        )

    reply_context = state["reply_context"]
    reply_display_name = reply_context.get("reply_to_display_name")
    reply_platform_user_id = reply_context.get("reply_to_platform_user_id")
    _add_scope_user(
        scope_users,
        display_name=reply_display_name,
        platform_user_id=reply_platform_user_id,
        global_user_id="",
    )

    return scope_users


def _selected_action_specs(state: GlobalPersonaState) -> list[dict]:
    """Return materialized action specs selected for the current episode."""

    raw_specs = state.get("action_specs")
    if not isinstance(raw_specs, list):
        return_value: list[dict] = []
        return return_value
    specs = [spec for spec in raw_specs if isinstance(spec, dict)]
    return specs


def _cognition_selects_text_surface(state: GlobalPersonaState) -> bool:
    """Return whether L2d selected the text surface handler."""

    return_value = (
        _first_valid_action_attempt_id(state, SPEAK_CAPABILITY) is not None
    )
    return return_value


def _empty_action_directives() -> dict:
    """Return a consolidation-safe directive shell for private-only episodes."""

    directives = {
        "contextual_directives": {},
        "linguistic_directives": {
            "rhetorical_strategy": "",
            "linguistic_style": "",
            "accepted_user_preferences": [],
            "content_anchors": [],
            "forbidden_phrases": [],
        },
        "visual_directives": {
            "facial_expression": [],
            "body_language": [],
            "gaze_direction": [],
            "visual_vibe": [],
        },
    }
    return directives


async def _action_results_for_state(
    state: GlobalPersonaState,
    *,
    executed_action_attempt_ids: set[str] | None = None,
) -> list[dict]:
    """Evaluate selected actions into traceable action results."""

    action_results = await execute_action_specs_for_trace(
        _selected_action_specs(state),
        storage_timestamp_utc=state["storage_timestamp_utc"],
        executed_action_attempt_ids=executed_action_attempt_ids,
    )
    return action_results


def _episode_trace_update(
    state: GlobalPersonaState,
    *,
    action_results: list[dict],
    surface_outputs: list[dict],
) -> dict:
    """Build trace fields for the current persona episode."""

    episode = state["cognitive_episode"]
    trace = build_episode_trace(
        episode_id=episode["episode_id"],
        trigger_source=episode["trigger_source"],
        created_at=state["storage_timestamp_utc"],
        action_specs=_selected_action_specs(state),
        action_results=action_results,
        surface_outputs=surface_outputs,
    )
    trace_update = {
        "action_results": action_results,
        "surface_outputs": surface_outputs,
        "episode_trace": trace,
    }
    return trace_update


async def call_action_subgraph(state: GlobalPersonaState) -> dict:
    """Run selected text-surface directives and dialog.

    Args:
        state: Current persona graph state.

    Returns:
        Partial state update with dialog fragments and addressed users.
    """

    surface_update = await call_l3_text_surface_handler(state)
    surface_state = dict(state)
    surface_state.update(surface_update)
    speak_attempt_id = _first_valid_action_attempt_id(
        surface_state,
        SPEAK_CAPABILITY,
    )
    result = await dialog_agent(surface_state)
    final_dialog = result["final_dialog"]
    action_results = await _action_results_for_state(
        surface_state,
        executed_action_attempt_ids=(
            {speak_attempt_id} if speak_attempt_id is not None else set()
        ),
    )
    surface_outputs = [
        build_text_surface_output(
            fragments=final_dialog,
            created_at=state["storage_timestamp_utc"],
            action_attempt_id=speak_attempt_id,
        )
    ]
    return_value = {
        "final_dialog": final_dialog,
        "target_addressed_user_ids": result["target_addressed_user_ids"],
        "target_broadcast": result["target_broadcast"],
        "mention_target_user": bool(result.get("mention_target_user", False)),
    }
    return_value.update(surface_update)
    return_value.update(_episode_trace_update(
        surface_state,
        action_results=action_results,
        surface_outputs=surface_outputs,
    ))
    return return_value


async def stage_3_no_response(state: GlobalPersonaState) -> dict:
    """Finish a private-only episode when L2d selected no text surface."""

    logger.info(
        f'Persona output short-circuited: platform={state["platform"]} '
        f'channel={state["platform_channel_id"] or "<dm>"} '
        f'user={state["global_user_id"]}'
    )
    return_value = {
        "should_respond": False,
        "final_dialog": [],
        "action_directives": _empty_action_directives(),
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "mention_target_user": False,
    }
    action_results = await _action_results_for_state(state)
    surface_outputs = []
    if "action_specs" in state:
        surface_outputs = [
            build_private_surface_output(
                summary="No visible text surface selected for this episode.",
                created_at=state["storage_timestamp_utc"],
            )
        ]
    return_value.update(_episode_trace_update(
        state,
        action_results=action_results,
        surface_outputs=surface_outputs,
    ))
    return return_value


async def run_rag_evidence_for_persona_state(
    state: GlobalPersonaState,
    *,
    agent_name: str,
    objective: str | None = None,
) -> dict:
    """Run reusable persona RAG evidence with this module's patch surface."""

    rag_result = await _run_rag_evidence_for_persona_state(
        state,
        agent_name=agent_name,
        objective=objective,
        call_rag_supervisor_func=call_quote_aware_rag_supervisor,
        record_rag_stage_event_func=event_logging.record_rag_stage_event,
        build_rag_request_func=build_text_chat_rag_request,
        component=PERSONA_RAG_COMPONENT,
    )
    return_value = rag_result
    return return_value


async def stage_1_research(state: GlobalPersonaState) -> dict:
    """Run RAG2 and project its facts into the persona payload.

    Args:
        state: Current top-level persona graph state.

    Returns:
        A partial state update containing the projected ``rag_result``.
    """
    rag_result = await run_rag_evidence_for_persona_state(
        state,
        agent_name="stage_1_research",
    )
    return_value = {
        "rag_result": rag_result,
    }
    return return_value


async def stage_1_goal_resolver(state: GlobalPersonaState) -> dict:
    """Run the cognition-preserving resolver loop after decontextualization."""

    initialized = ensure_initial_resolver_inputs(
        state,
        max_cycles=COGNITION_RESOLVER_MAX_CYCLES,
    )
    resolved_state = await call_cognition_resolver_loop(
        initialized,
        call_cognition_subgraph_func=call_cognition_subgraph,
        execute_capability_func=execute_resolver_capability_request,
        max_cycles=COGNITION_RESOLVER_MAX_CYCLES,
        capability_timeout_seconds=(
            COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS
        ),
    )
    return_value = resolved_state
    return return_value


def _route_after_cognition(state: GlobalPersonaState) -> str:
    """Route persona flow based on selected L2d text surfaces."""

    if _cognition_selects_text_surface(state):
        return_value = "respond"
    else:
        return_value = "silent"
    return return_value


def _first_valid_action_attempt_id(
    state: GlobalPersonaState,
    action_kind: str,
) -> str | None:
    """Return the first valid selected action-attempt id for one kind."""

    evaluator = ActionSpecEvaluator()
    for action_spec in _selected_action_specs(state):
        if action_spec.get("kind") != action_kind:
            continue
        eval_result = evaluator.evaluate(action_spec)
        if not eval_result["ok"]:
            continue
        attempt_id = action_attempt_id_from_eval_result(eval_result)
        if attempt_id:
            return_value = attempt_id
            return return_value
    return_value = None
    return return_value


async def persona_supervisor2(state: IMProcessState) -> dict:
    """Run persona reasoning with history scoped to the active user thread.

    Args:
        state: Top-level chat graph state after relevance gating.

    Returns:
        Dialog output and the persona-state snapshot used by background tasks.
    """

    recent_channel_history_for_decontextualizer = format_storage_utc_history_for_llm(
        state["chat_history_wide"]
    )[-CHAT_HISTORY_RECENT_LIMIT:]
    scope_users = _build_scope_users(
        state,
        recent_channel_history_for_decontextualizer,
    )
    raw_interaction_wide = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )
    interaction_history_wide = format_storage_utc_history_for_llm(
        raw_interaction_wide
    )
    interaction_history_recent = interaction_history_wide[-CHAT_HISTORY_RECENT_LIMIT:]

    async def stage_0_msg_decontexualizer(
        persona_state: GlobalPersonaState,
    ) -> dict:
        """Run decontextualization with recent channel history and identities."""

        decontextualizer_state = dict(persona_state)
        decontextualizer_state["chat_history_recent"] = (
            recent_channel_history_for_decontextualizer
        )
        result = await call_msg_decontexualizer(decontextualizer_state)
        return_value = result
        return return_value

    # Build the top level graph that connect stages
    persona_builder = StateGraph(GlobalPersonaState)
    persona_builder.add_node(
        "stage_0_msg_decontexualizer",
        stage_0_msg_decontexualizer,
    )
    if COGNITION_RESOLVER_ENABLED:
        persona_builder.add_node("stage_1_goal_resolver", stage_1_goal_resolver)
    else:
        persona_builder.add_node("stage_1_research", stage_1_research)
        persona_builder.add_node("stage_2_cognition", call_cognition_subgraph)
    persona_builder.add_node(
        "stage_2_memory_lifecycle",
        call_memory_lifecycle_update_handler,
    )
    persona_builder.add_node("stage_3_action", call_action_subgraph)  # perform action
    persona_builder.add_node("stage_3_no_response", stage_3_no_response)
    persona_builder.add_edge(START, "stage_0_msg_decontexualizer")
    if COGNITION_RESOLVER_ENABLED:
        persona_builder.add_edge(
            "stage_0_msg_decontexualizer",
            "stage_1_goal_resolver",
        )
        persona_builder.add_edge(
            "stage_1_goal_resolver",
            "stage_2_memory_lifecycle",
        )
    else:
        persona_builder.add_edge(
            "stage_0_msg_decontexualizer",
            "stage_1_research",
        )
        persona_builder.add_edge("stage_1_research", "stage_2_cognition")
        persona_builder.add_edge("stage_2_cognition", "stage_2_memory_lifecycle")
    persona_builder.add_conditional_edges(
        "stage_2_memory_lifecycle",
        _route_after_cognition,
        {
            "silent": "stage_3_no_response",
            "respond": "stage_3_action",
        },
    )
    persona_builder.add_edge("stage_3_action", END)
    persona_builder.add_edge("stage_3_no_response", END)

    
    persona_graph = persona_builder.compile()

    initial_persona_state: GlobalPersonaState = {
        # Character Related
        "character_profile": state["character_profile"],

        # Inputs
        "storage_timestamp_utc": state["storage_timestamp_utc"],
        "local_time_context": state["local_time_context"],
        "user_input": state["user_input"],
        "prompt_message_context": state["prompt_message_context"],
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "channel_type": state["channel_type"],
        "platform_message_id": state["platform_message_id"],
        "platform_user_id": state["platform_user_id"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "platform_bot_id": state["platform_bot_id"],
        "chat_history_wide": interaction_history_wide,
        "chat_history_recent": interaction_history_recent,
        "reply_context": state["reply_context"],
        "indirect_speech_context": state["indirect_speech_context"],
        "channel_topic": state["channel_topic"],
        "scope_users": scope_users,
        "conversation_episode_state": state.get("conversation_episode_state"),
        "conversation_progress": state.get("conversation_progress"),
        "promoted_reflection_context": state.get("promoted_reflection_context"),
        "internal_monologue_residue_context": state.get(
            "internal_monologue_residue_context",
            "",
        ),
        "referents": [],
        "debug_modes": state["debug_modes"],
        "should_respond": state["should_respond"],
    }
    cognitive_episode = state.get("cognitive_episode")
    if cognitive_episode is not None:
        initial_persona_state["cognitive_episode"] = cognitive_episode
    
    results = await persona_graph.ainvoke(initial_persona_state)
    
    return_value = {
        "should_respond": results["should_respond"],
        "final_dialog": results["final_dialog"],
        "target_addressed_user_ids": results["target_addressed_user_ids"],
        "target_broadcast": bool(results["target_broadcast"]),
        "mention_target_user": bool(results.get("mention_target_user", False)),
        "future_promises": [],
        "consolidation_state": results,
        "surface_outputs": results.get("surface_outputs", []),
        "action_results": results.get("action_results", []),
        "episode_trace": results.get("episode_trace"),
    }
    return return_value
