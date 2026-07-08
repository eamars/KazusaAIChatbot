"""Kazusa connector for selected L3 text-surface planning."""

import logging

from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY
from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    CognitionTextSurfaceInputV1,
    validate_text_surface_input,
)
from kazusa_ai_chatbot.cognition_chain_core.surface import (
    run_text_surface_planning,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverValidationError,
    project_goal_progress_for_cognition,
    project_observations_for_cognition,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    MAX_PROJECTED_RESOLVER_OBSERVATIONS,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_chain_services,
    build_text_surface_chain_input_from_global_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    GlobalPersonaState,
)
from kazusa_ai_chatbot.db import (
    build_interaction_style_context,
    empty_interaction_style_overlay,
)

logger = logging.getLogger(__name__)


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


def build_text_surface_input_from_global_state(
    state: GlobalPersonaState,
) -> CognitionTextSurfaceInputV1:
    """Project graph state into the selected text-surface core contract."""

    selected_intent = _selected_text_surface_intent_object(state)
    payload: CognitionTextSurfaceInputV1 = {
        "schema_version": "cognition_text_surface_input.v1",
        "chain_input": build_text_surface_chain_input_from_global_state(state),
        "cognition_residue": {
            "emotional_appraisal": state["emotional_appraisal"],
            "interaction_subtext": state["interaction_subtext"],
            "internal_monologue": state["internal_monologue"],
            "logical_stance": state["logical_stance"],
            "character_intent": state["character_intent"],
            "judgment_note": state["judgment_note"],
            "social_distance": state["social_distance"],
            "emotional_intensity": state["emotional_intensity"],
            "vibe_check": state["vibe_check"],
            "relational_dynamic": state["relational_dynamic"],
        },
        "selected_text_surface_intent": selected_intent,
        "pre_surface_action_results": _pre_surface_action_result_prompts(state),
        "memory_lifecycle_context": _memory_lifecycle_context_prompt(state),
        "interaction_style_context": _empty_interaction_style_context(
            state["channel_type"],
        ),
    }
    validated_payload = validate_text_surface_input(payload)
    return validated_payload


def _selected_text_surface_intent_object(
    state: GlobalPersonaState,
) -> dict[str, str]:
    """Project one selected speak action into prompt-safe structured intent."""

    raw_specs = state.get("action_specs")
    if not isinstance(raw_specs, list):
        return_value = _empty_text_surface_intent()
        return return_value

    for action_spec in raw_specs:
        if not isinstance(action_spec, dict):
            continue
        if action_spec.get("kind") != SPEAK_CAPABILITY:
            continue
        params = action_spec.get("params")
        if not isinstance(params, dict):
            params = {}
        requirements = params.get("surface_requirements")
        if not isinstance(requirements, dict):
            requirements = {}
        return_value = {
            "decision": "visible_reply",
            "original_goal": _resolved_pending_original_goal(state),
            "goal_progress_summary": _resolver_goal_progress_text(state),
            "observation_summary": _resolver_observations_text(state),
            "speak_intent": _selected_text_surface_intent(state),
            "detail": _row_text(requirements, "detail"),
            "tone": _row_text(requirements, "tone"),
            "reason": _row_text(action_spec, "reason"),
        }
        return return_value

    return_value = _empty_text_surface_intent()
    return return_value


def _empty_text_surface_intent() -> dict[str, str]:
    """Return an empty but schema-complete selected text intent."""

    return_value = {
        "decision": "visible_reply",
        "original_goal": "",
        "goal_progress_summary": "",
        "observation_summary": "",
        "speak_intent": "",
        "detail": "",
        "tone": "",
        "reason": "",
    }
    return return_value


def _pre_surface_action_result_prompts(
    state: GlobalPersonaState,
) -> list[dict[str, str]]:
    """Project pre-surface action results into the core surface contract."""

    raw_results = state.get("pre_surface_action_results")
    if not isinstance(raw_results, list):
        return_value: list[dict[str, str]] = []
        return return_value

    rows: list[dict[str, str]] = []
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        action_kind = _row_text(row, "action_kind")
        prompt_action_kind = _pre_surface_prompt_action_kind(action_kind)
        if prompt_action_kind not in (
            "accepted_task_request",
            "accepted_coding_task_request",
            "background_work_request",
            "future_speak",
            "memory_lifecycle_update",
        ):
            continue
        prompt_row = {
            "action_kind": prompt_action_kind,
            "status": _row_text(row, "status"),
            "task_summary": _task_summary_for_prompt(row),
            "objective_summary": _row_text(row, "objective_summary"),
            "acknowledgement_constraint": _row_text(
                row,
                "acknowledgement_constraint",
            ),
            "accepted_task_state": _row_text(row, "accepted_task_state"),
            "accepted_task_summary": _row_text(row, "accepted_task_summary"),
            "wait_guidance": _row_text(row, "wait_guidance"),
        }
        queue_state = _queue_state_for_prompt(row, prompt_action_kind)
        if queue_state:
            prompt_row["queue_state"] = queue_state
        rows.append(prompt_row)
    return rows


def _memory_lifecycle_context_prompt(
    state: GlobalPersonaState,
) -> dict[str, object]:
    """Project memory lifecycle context for selected L3 planning."""

    raw_context = state.get("memory_lifecycle_context")
    if not isinstance(raw_context, dict):
        return_value = {
            "active_commitment_aliases": [],
            "pending_memory_updates_summary": "",
            "recent_memory_resolution_summary": "",
        }
        return return_value
    aliases = raw_context.get("active_commitment_aliases")
    if not isinstance(aliases, list):
        aliases = []
    return_value = {
        "active_commitment_aliases": [
            alias for alias in aliases if isinstance(alias, str)
        ],
        "pending_memory_updates_summary": _row_text(
            raw_context,
            "pending_memory_updates_summary",
        ),
        "recent_memory_resolution_summary": _row_text(
            raw_context,
            "recent_memory_resolution_summary",
        ),
    }
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


def _background_work_acknowledgement_parts(
    state: GlobalPersonaState,
) -> list[str]:
    """Return prompt-safe accepted-task acknowledgement constraints."""

    raw_results = state.get("pre_surface_action_results")
    if not isinstance(raw_results, list):
        return_value: list[str] = []
        return return_value

    parts: list[str] = []
    for row in raw_results:
        if not isinstance(row, dict):
            continue
        action_kind = row.get("action_kind")
        if action_kind not in (
            "accepted_coding_task_request",
            "background_work_request",
            "future_speak",
        ):
            continue
        prompt_action_kind = _pre_surface_prompt_action_kind(str(action_kind))
        task_summary = _task_summary_for_prompt(row)
        accepted_task_state = _row_text(row, "accepted_task_state")
        wait_guidance = _row_text(row, "wait_guidance")
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
            f"{prompt_action_kind}："
            f"task={task_summary or 'unknown'}，"
            f"accepted_task_state={accepted_task_state or 'unknown'}，"
            f"acknowledgement_constraint={acknowledgement_constraint}，"
            f"wait_guidance={wait_guidance or 'unknown'}"
        )
        parts.append(part)
    return parts


def _pre_surface_prompt_action_kind(action_kind: str) -> str:
    """Map internal delayed-work action kind to prompt semantic kind."""

    if action_kind == "background_work_request":
        return_value = "accepted_task_request"
        return return_value
    return_value = action_kind
    return return_value


def _task_summary_for_prompt(row: dict) -> str:
    """Return accepted-task summary before legacy task summary."""

    accepted_summary = _row_text(row, "accepted_task_summary")
    if accepted_summary:
        return accepted_summary
    return_value = _row_text(row, "task_summary")
    return return_value


def _queue_state_for_prompt(row: dict, prompt_action_kind: str) -> str:
    """Hide queue state for accepted-task prompt rows."""

    if prompt_action_kind in ("accepted_task_request", "future_speak"):
        return_value = ""
        return return_value
    return_value = _row_text(row, "queue_state")
    return return_value


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


async def _load_interaction_style_context(
    state: GlobalPersonaState,
) -> dict[str, object]:
    """Load DB-backed style overlays before entering cognition core."""

    channel_type = state["channel_type"]
    try:
        context = await build_interaction_style_context(
            global_user_id=state["global_user_id"],
            channel_type=channel_type,
            platform=state["platform"],
            platform_channel_id=state["platform_channel_id"],
        )
    except Exception as exc:
        logger.exception(f"Interaction style context load failed: {exc}")
        context = _empty_interaction_style_context(channel_type)
    return context


def _empty_interaction_style_context(channel_type: str) -> dict[str, object]:
    """Return an empty L3-facing interaction style context."""

    context: dict[str, object] = {
        "user_style": empty_interaction_style_overlay(),
        "application_order": ["user_style"],
    }
    if channel_type == "group":
        context["group_channel_style"] = empty_interaction_style_overlay()
        context["application_order"] = ["user_style", "group_channel_style"]
    return context


async def call_l3_text_surface_handler(state: GlobalPersonaState) -> dict:
    """Run L3 directive generation for a selected text surface.

    Args:
        state: Persona graph state after L2d has selected a ``speak`` action.

    Returns:
        Partial state update containing collected text-surface directives.
    """

    surface_input = build_text_surface_input_from_global_state(state)
    surface_input["interaction_style_context"] = (
        await _load_interaction_style_context(state)
    )
    surface_input = validate_text_surface_input(surface_input)
    result = await run_text_surface_planning(
        surface_input,
        build_cognition_chain_services(),
    )
    return_value = {
        "action_directives": result["action_directives"],
    }
    return return_value
