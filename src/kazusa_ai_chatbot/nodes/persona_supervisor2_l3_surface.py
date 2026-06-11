"""Selected L3 text-surface handler."""

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverValidationError,
    project_goal_progress_for_cognition,
    project_observations_for_cognition,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    MAX_PROJECTED_RESOLVER_OBSERVATIONS,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import (
    call_content_plan_agent,
    call_interaction_style_context_loader,
    call_preference_adapter,
    call_surface_directive_collector,
    call_style_agent,
    call_visual_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    CognitionState,
    GlobalPersonaState,
)
from kazusa_ai_chatbot.utils import build_interaction_history_recent


def _selected_text_surface_intent(state: GlobalPersonaState) -> str:
    """Project the selected text action into one model-facing intent string."""

    raw_specs = state.get("action_specs")
    if not isinstance(raw_specs, list):
        return_value = ""
        return return_value

    for action_spec in raw_specs:
        if not isinstance(action_spec, dict):
            continue
        kind = action_spec.get("kind")
        if kind != SPEAK_CAPABILITY:
            continue
        intent_parts = _text_surface_intent_parts(action_spec, state)
        return_value = "；".join(intent_parts)
        return return_value

    return_value = ""
    return return_value


def _text_surface_intent_parts(
    action_spec: dict,
    state: GlobalPersonaState,
) -> list[str]:
    """Extract prompt-safe text-surface requirements from one action spec."""

    params = action_spec.get("params")
    if not isinstance(params, dict):
        params = {}
    raw_requirements = params.get("surface_requirements")
    if not isinstance(raw_requirements, dict):
        raw_requirements = {}

    intent_parts: list[str] = []
    original_goal = _resolved_pending_original_goal(state)
    if original_goal:
        intent_parts.append(f"原始目标：{original_goal}")
    goal_progress = _resolver_goal_progress_text(state)
    if goal_progress:
        intent_parts.append(f"目标进度：{goal_progress}")
    observation_context = _resolver_observations_text(state)
    if observation_context:
        intent_parts.append(f"证据观察：{observation_context}")
    intent_parts.extend(_background_artifact_acknowledgement_parts(state))
    intent_parts.extend(_background_work_acknowledgement_parts(state))

    for field_name, label in (
        ("decision", "决策"),
        ("intent", "目标"),
        ("detail", "内容要求"),
        ("tone", "语气"),
    ):
        value = raw_requirements.get(field_name)
        if isinstance(value, str) and value.strip():
            intent_parts.append(f"{label}：{value.strip()}")

    reason = action_spec.get("reason")
    if isinstance(reason, str) and reason.strip():
        intent_parts.append(f"理由：{reason.strip()}")

    return intent_parts


def _background_artifact_acknowledgement_parts(
    state: GlobalPersonaState,
) -> list[str]:
    """Return prompt-safe artifact queue acknowledgement constraints."""

    raw_results = state.get("pre_surface_action_results")
    if not isinstance(raw_results, list):
        return_value: list[str] = []
        return return_value

    parts: list[str] = []
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        if row.get("action_kind") != "background_artifact_request":
            continue
        work_kind = _row_text(row, "work_kind")
        objective_summary = _row_text(row, "objective_summary")
        acknowledgement_constraint = _row_text(
            row,
            "acknowledgement_constraint",
        )
        if not acknowledgement_constraint:
            status = _row_text(row, "status")
            queue_state = _row_text(row, "queue_state")
            if status == "pending" and queue_state == "queued":
                acknowledgement_constraint = "promise_allowed"
            else:
                acknowledgement_constraint = "promise_forbidden_explain_failure"
        part = (
            "background_artifact_request："
            f"work_kind={work_kind or 'unknown'}，"
            f"objective={objective_summary or 'unknown'}，"
            f"acknowledgement_constraint={acknowledgement_constraint}"
        )
        parts.append(part)
    return parts


def _background_work_acknowledgement_parts(
    state: GlobalPersonaState,
) -> list[str]:
    """Return prompt-safe background-work acknowledgement constraints."""

    raw_results = state.get("pre_surface_action_results")
    if not isinstance(raw_results, list):
        return_value: list[str] = []
        return return_value

    parts: list[str] = []
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        if row.get("action_kind") != "background_work_request":
            continue
        task_summary = _row_text(row, "task_summary")
        acknowledgement_constraint = _row_text(
            row,
            "acknowledgement_constraint",
        )
        if not acknowledgement_constraint:
            status = _row_text(row, "status")
            queue_state = _row_text(row, "queue_state")
            if status == "pending" and queue_state == "queued":
                acknowledgement_constraint = "promise_allowed"
            else:
                acknowledgement_constraint = "promise_forbidden_explain_failure"
        part = (
            "background_work_request："
            f"task={task_summary or 'unknown'}，"
            f"acknowledgement_constraint={acknowledgement_constraint}"
        )
        parts.append(part)
    return parts


def _row_text(row: dict, field_name: str) -> str:
    """Return one stripped text field from an action-result row."""

    value = row.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _resolver_goal_progress_text(state: GlobalPersonaState) -> str:
    """Return compact goal progress text for L3 surface selection."""

    raw_goal_progress = state.get("resolver_goal_progress")
    if not isinstance(raw_goal_progress, dict):
        resolver_state = state.get("resolver_state")
        if isinstance(resolver_state, dict):
            nested_goal_progress = resolver_state.get("goal_progress")
            if isinstance(nested_goal_progress, dict):
                raw_goal_progress = nested_goal_progress
    if not isinstance(raw_goal_progress, dict):
        return_value = ""
        return return_value
    try:
        goal_progress = project_goal_progress_for_cognition(raw_goal_progress)
    except ResolverValidationError:
        return_value = ""
        return return_value
    return_value = goal_progress.replace("\n", " / ")
    return return_value


def _resolver_observations_text(state: GlobalPersonaState) -> str:
    """Return bounded prompt-safe resolver observations for text surfacing."""

    resolver_state = state.get("resolver_state")
    if not isinstance(resolver_state, dict):
        return_value = ""
        return return_value
    raw_observations = resolver_state.get("observations")
    if not isinstance(raw_observations, list):
        return_value = ""
        return return_value
    try:
        observation_context = project_observations_for_cognition(
            raw_observations[-MAX_PROJECTED_RESOLVER_OBSERVATIONS:],
        )
    except ResolverValidationError:
        return_value = ""
        return return_value
    return_value = observation_context.replace("\n", " / ")
    return return_value


def _resolved_pending_original_goal(state: GlobalPersonaState) -> str:
    """Return the original HIL goal after a pending row has been resolved."""

    pending_resume = state.get("pending_resolver_resume")
    if not isinstance(pending_resume, dict):
        resolver_state = state.get("resolver_state")
        if isinstance(resolver_state, dict):
            nested_pending_resume = resolver_state.get("pending_resume")
            if isinstance(nested_pending_resume, dict):
                pending_resume = nested_pending_resume
    if not isinstance(pending_resume, dict):
        return_value = ""
        return return_value

    resolution = state.get("resolver_pending_resolution")
    include_goal = pending_resume.get("status") == "superseded"
    if isinstance(resolution, dict):
        include_goal = resolution.get("decision") in (
            "answered",
            "approved",
            "superseded",
        )
    if not include_goal:
        return_value = ""
        return return_value

    original_goal = pending_resume.get("prompt_safe_original_goal")
    if isinstance(original_goal, str) and original_goal.strip():
        return_value = original_goal.strip()
        return return_value

    return_value = ""
    return return_value


async def call_l3_text_surface_handler(state: GlobalPersonaState) -> dict:
    """Run L3 directive generation for a selected text surface.

    Args:
        state: Persona graph state after L2d has selected a ``speak`` action.

    Returns:
        Partial state update containing collected text-surface directives.
    """

    surface_builder = StateGraph(CognitionState)
    surface_builder.add_node(
        "l3_interaction_style_context_loader",
        call_interaction_style_context_loader,
    )
    surface_builder.add_node("l3_style_agent", call_style_agent)
    surface_builder.add_node("l3_content_plan_agent", call_content_plan_agent)
    surface_builder.add_node("l3_preference_adapter", call_preference_adapter)
    surface_builder.add_node("l3_visual_agent", call_visual_agent)
    surface_builder.add_node(
        "l4_surface_directive_collector",
        call_surface_directive_collector,
    )

    surface_builder.add_edge(START, "l3_interaction_style_context_loader")
    surface_builder.add_edge(
        "l3_interaction_style_context_loader",
        "l3_content_plan_agent",
    )
    surface_builder.add_edge("l3_content_plan_agent", "l3_visual_agent")
    surface_builder.add_edge("l3_interaction_style_context_loader", "l3_style_agent")
    surface_builder.add_edge(
        ["l3_style_agent", "l3_content_plan_agent"],
        "l3_preference_adapter",
    )
    surface_builder.add_edge(
        ["l3_preference_adapter", "l3_visual_agent"],
        "l4_surface_directive_collector",
    )
    surface_builder.add_edge("l4_surface_directive_collector", END)

    surface_graph = surface_builder.compile()
    interaction_history_recent = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )
    initial_state: CognitionState = {
        "character_profile": state["character_profile"],
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
        "decontexualized_input": state["decontexualized_input"],
        "referents": state["referents"],
        "rag_result": state["rag_result"],
        "emotional_appraisal": state["emotional_appraisal"],
        "interaction_subtext": state["interaction_subtext"],
        "internal_monologue": state["internal_monologue"],
        "character_intent": state["character_intent"],
        "logical_stance": state["logical_stance"],
        "judgment_note": state["judgment_note"],
        "social_distance": state["social_distance"],
        "emotional_intensity": state["emotional_intensity"],
        "vibe_check": state["vibe_check"],
        "relational_dynamic": state["relational_dynamic"],
    }
    cognitive_episode = state.get("cognitive_episode")
    if cognitive_episode is not None:
        initial_state["cognitive_episode"] = cognitive_episode
    selected_text_surface_intent = _selected_text_surface_intent(state)
    if selected_text_surface_intent:
        initial_state["selected_text_surface_intent"] = (
            selected_text_surface_intent
        )
    resolver_state = state.get("resolver_state")
    if resolver_state is not None:
        initial_state["resolver_state"] = resolver_state
    resolver_goal_progress = state.get("resolver_goal_progress")
    if resolver_goal_progress is not None:
        initial_state["resolver_goal_progress"] = resolver_goal_progress
    memory_lifecycle_context = state.get("memory_lifecycle_context")
    if isinstance(memory_lifecycle_context, dict):
        initial_state["memory_lifecycle_context"] = memory_lifecycle_context

    result = await surface_graph.ainvoke(initial_state)
    return_value = {
        "action_directives": result["action_directives"],
    }
    return return_value
