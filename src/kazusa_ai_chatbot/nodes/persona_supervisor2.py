"""Persona graph orchestration for decontextualization, RAG, cognition, and dialog."""

import logging
import time

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
from kazusa_ai_chatbot.config import CHAT_HISTORY_RECENT_LIMIT
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import call_cognition_subgraph
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator import call_consolidation_subgraph
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    call_l3_text_surface_handler,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import call_msg_decontexualizer
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import project_known_facts
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.nodes.referent_resolution import (
    should_skip_rag_for_unresolved_referents,
    unresolved_referent_reason,
)
from kazusa_ai_chatbot.rag.cognitive_episode_adapter import (
    build_text_chat_rag_request,
)
from kazusa_ai_chatbot.rag.quote_aware_sequence import (
    call_quote_aware_rag_supervisor,
)
from kazusa_ai_chatbot.state import IMProcessState
from kazusa_ai_chatbot.time_boundary import format_storage_utc_history_for_llm
from kazusa_ai_chatbot.utils import build_interaction_history_recent, log_preview

logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
PERSONA_RAG_COMPONENT = "nodes.persona_supervisor2"


def _selected_action_specs(state: GlobalPersonaState) -> list[dict]:
    """Return materialized action specs selected for the current episode."""

    raw_specs = state.get("action_specs")
    if not isinstance(raw_specs, list):
        return_value: list[dict] = []
        return return_value
    specs = [spec for spec in raw_specs if isinstance(spec, dict)]
    return specs


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic milliseconds since a start marker."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = max(0, int(elapsed * MILLISECONDS_PER_SECOND))
    return elapsed_ms


def _rag_correlation_id(state: GlobalPersonaState) -> str:
    """Build a non-content correlation id for persona RAG work."""

    platform = str(state.get("platform", ""))
    message_ref = str(state.get("platform_message_id", "") or "no-message-id")
    correlation_id = f"rag:{platform}:{message_ref}"
    return correlation_id


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


async def stage_1_research(state: GlobalPersonaState) -> dict:
    """Run RAG2 and project its facts into the persona payload.

    Args:
        state: Current top-level persona graph state.

    Returns:
        A partial state update containing the projected ``rag_result``.
    """
    started_at = time.perf_counter()
    correlation_id = _rag_correlation_id(state)
    referents = state["referents"]
    if should_skip_rag_for_unresolved_referents(referents):
        referent_reason = unresolved_referent_reason(referents)
        rag_result = project_known_facts(
            [],
            current_user_id=state["global_user_id"],
            character_user_id=state["character_profile"]["global_user_id"],
            answer="",
            unknown_slots=[],
            loop_count=0,
        )
        logger.info(
            f"RAG2 skipped output: reason={log_preview(referent_reason)}"
        )
        logger.debug(
            f'RAG2 skipped metadata: platform={state["platform"]} '
            f'channel={state["platform_channel_id"] or "<dm>"} '
            f'user={state["global_user_id"]} '
            f'query={log_preview(state["decontexualized_input"])} '
            f"rag_result={log_preview(rag_result)}"
        )
        await event_logging.record_rag_stage_event(
            component=PERSONA_RAG_COMPONENT,
            correlation_id=correlation_id,
            agent_name="stage_1_research",
            status="skipped",
            slot_count=0,
            retrieval_count=0,
            cache_hit=False,
            no_evidence=True,
            latency_ms=_elapsed_ms(started_at),
        )
        return_value = {
            "rag_result": rag_result,
        }
        return return_value

    rag_request = build_text_chat_rag_request(
        episode=state["cognitive_episode"],
        decontexualized_input=state["decontexualized_input"],
        character_profile=state["character_profile"],
        user_profile=state["user_profile"],
        prompt_message_context=state["prompt_message_context"],
        channel_topic=state["channel_topic"],
        chat_history_recent=state["chat_history_recent"],
        chat_history_wide=state["chat_history_wide"],
        reply_context=state["reply_context"],
        indirect_speech_context=state["indirect_speech_context"],
        conversation_progress=state.get("conversation_progress"),
        conversation_episode_state=state.get("conversation_episode_state"),
        promoted_reflection_context=state.get("promoted_reflection_context"),
    )
    rag_supervisor_result = await call_quote_aware_rag_supervisor(
        fresh_query=rag_request["original_query"],
        reply_context=state["reply_context"],
        character_name=rag_request["character_name"],
        context=rag_request["context"],
    )
    rag_result = project_known_facts(
        rag_supervisor_result["known_facts"],
        current_user_id=rag_request["current_user_id"],
        character_user_id=rag_request["character_user_id"],
        answer=str(rag_supervisor_result["answer"]),
        unknown_slots=rag_supervisor_result["unknown_slots"],
        loop_count=int(rag_supervisor_result["loop_count"] or 0),
    )
    trace = rag_result["supervisor_trace"]
    logger.info(
        f'RAG2 projection output: answer={log_preview(rag_result["answer"])}'
    )
    logger.debug(
        f'RAG2 projection metadata: platform={state["platform"]} '
        f'channel={state["platform_channel_id"] or "<dm>"} '
        f'user={state["global_user_id"]} '
        f'query={log_preview(state["decontexualized_input"])} '
        f'dispatched={len(trace["dispatched"])} '
        f'user_image={bool(rag_result["user_image"])} '
        f'character_image={bool(rag_result["character_image"])} '
        f'third_party_profiles={len(rag_result["third_party_profiles"])} '
        f'memory_evidence={len(rag_result["memory_evidence"])} '
        f'recall_evidence={len(rag_result["recall_evidence"])} '
        f'conversation_evidence={len(rag_result["conversation_evidence"])} '
        f'external_evidence={len(rag_result["external_evidence"])} '
        f"rag_result={log_preview(rag_result)}"
    )
    retrieval_count = (
        len(rag_result["memory_evidence"])
        + len(rag_result["recall_evidence"])
        + len(rag_result["conversation_evidence"])
        + len(rag_result["external_evidence"])
        + len(rag_result["third_party_profiles"])
    )
    await event_logging.record_rag_stage_event(
        component=PERSONA_RAG_COMPONENT,
        correlation_id=correlation_id,
        agent_name="stage_1_research",
        status="succeeded",
        slot_count=len(rag_supervisor_result["unknown_slots"]),
        retrieval_count=retrieval_count,
        cache_hit=False,
        no_evidence=retrieval_count == 0,
        latency_ms=_elapsed_ms(started_at),
    )
    return_value = {
        "rag_result": rag_result,
    }
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
        """Run decontextualization with recent full channel history only."""

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
    persona_builder.add_node("stage_1_research", stage_1_research)
    persona_builder.add_node("stage_2_cognition", call_cognition_subgraph)
    persona_builder.add_node("stage_3_action", call_action_subgraph)  # perform action
    persona_builder.add_node("stage_3_no_response", stage_3_no_response)
    persona_builder.add_edge(START, "stage_0_msg_decontexualizer")
    persona_builder.add_edge("stage_0_msg_decontexualizer", "stage_1_research")
    persona_builder.add_edge("stage_1_research", "stage_2_cognition")
    persona_builder.add_conditional_edges(
        "stage_2_cognition",
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
        "conversation_episode_state": state.get("conversation_episode_state"),
        "conversation_progress": state.get("conversation_progress"),
        "promoted_reflection_context": state.get("promoted_reflection_context"),
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
