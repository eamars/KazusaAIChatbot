"""Persona graph orchestration for decontextualization, RAG, cognition, and dialog."""

import logging
from collections.abc import Mapping

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.execution import execute_action_specs_for_trace
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
    FUTURE_SPEAK_CAPABILITY,
    SPEAK_CAPABILITY,
)
from kazusa_ai_chatbot.action_spec.results import (
    action_attempt_id_from_eval_result,
    build_private_surface_output,
    build_text_surface_output,
    build_visual_surface_output,
)
from kazusa_ai_chatbot.accepted_task import (
    load_open_coding_run_contexts_for_scope,
)
from kazusa_ai_chatbot.config import (
    COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS,
    COGNITION_RESOLVER_MAX_CYCLES,
)
from kazusa_ai_chatbot.conversation_progress import (
    GroupSceneProjectionError,
    build_group_scene_context,
    filter_group_scene_ambient_turns,
    logical_turns_as_history_rows,
    project_group_scene_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
    SceneParticipantBindingV1,
    validate_cognition_core_output,
    validate_text_surface_output,
    validate_visual_surface_output,
)
from kazusa_ai_chatbot.cognition_core_v2.prompt_budget import (
    MAX_SCENE_PARTICIPANT_BINDINGS,
)
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    execute_resolver_capability_request,
    run_rag_evidence_for_persona_state as _run_rag_evidence_for_persona_state,
)
from kazusa_ai_chatbot.cognition_resolver.loop import (
    call_cognition_resolver_loop,
)
from kazusa_ai_chatbot.cognition_resolver.guardrail import (
    current_cognition_retry_coordinator,
)
from kazusa_ai_chatbot.cognition_resolver.pending import (
    apply_pending_resolution,
    load_matching_pending_resume_into_state,
    upsert_pending_resume,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    ensure_initial_resolver_inputs,
)
from kazusa_ai_chatbot.nodes.dialog_agent import (
    StateContractError,
    dialog_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_action_availability_snapshot,
    call_cognition_subgraph,
    commit_cognition_output,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    call_l3_text_surface_handler,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_memory_lifecycle import (
    call_memory_lifecycle_update_handler,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontextualizer import (
    call_msg_decontextualizer,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    GlobalPersonaState,
    ScopeUser,
)
from kazusa_ai_chatbot.state import IMProcessState
from kazusa_ai_chatbot.time_boundary import format_storage_utc_history_for_llm
from kazusa_ai_chatbot.utils import (
    text_or_empty,
)

logger = logging.getLogger(__name__)


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


def _build_scene_participant_bindings(
    state: IMProcessState,
    scope_users: list[ScopeUser],
) -> list[SceneParticipantBindingV1]:
    """Project non-current visible group users into episode-local handles."""

    if state["channel_type"] != "group":
        return []
    current_global_user_id = text_or_empty(state["global_user_id"])
    current_platform_user_id = text_or_empty(state["platform_user_id"])
    current_display_name = text_or_empty(state["user_name"])
    character_profile = state["character_profile"]
    character_global_user_id = text_or_empty(
        character_profile["global_user_id"]
    )
    character_platform_user_id = text_or_empty(state["platform_bot_id"])
    character_display_name = text_or_empty(character_profile["name"])
    bindings: list[SceneParticipantBindingV1] = []
    for scope_user in scope_users:
        display_name = text_or_empty(scope_user.get("display_name"))
        if not display_name:
            continue
        row_global_user_id = text_or_empty(scope_user.get("global_user_id"))
        row_platform_user_id = text_or_empty(
            scope_user.get("platform_user_id")
        )
        same_global_identity = bool(row_global_user_id) and (
            row_global_user_id in {
                current_global_user_id,
                character_global_user_id,
            }
        )
        same_platform_identity = bool(row_platform_user_id) and (
            row_platform_user_id in {
                current_platform_user_id,
                character_platform_user_id,
            }
        )
        if (
            same_global_identity
            or same_platform_identity
            or display_name in {current_display_name, character_display_name}
        ):
            continue
        handle = f"p{len(bindings) + 1}"
        bindings.append({
            "handle": handle,
            "display_name": display_name,
            "entity_kind": "third_party",
        })
        if len(bindings) >= MAX_SCENE_PARTICIPANT_BINDINGS:
            break
    return bindings


def _selected_action_specs(state: GlobalPersonaState) -> list[dict]:
    """Return materialized action specs selected for the current episode."""

    raw_specs = state.get("action_specs")
    if not isinstance(raw_specs, list):
        return_value: list[dict] = []
        return return_value
    specs = [spec for spec in raw_specs if isinstance(spec, dict)]
    return specs


def _cognition_selects_text_surface(state: GlobalPersonaState) -> bool:
    """Return whether the validated V2 intention selects text speech."""

    evaluator = ActionSpecEvaluator()
    for action_spec in _selected_action_specs(state):
        if action_spec.get("kind") != SPEAK_CAPABILITY:
            continue
        if evaluator.evaluate(action_spec)["ok"]:
            return_value = True
            return return_value

    cognition_output = state.get("cognition_core_output")
    if not isinstance(cognition_output, Mapping):
        raise CognitionExecutionError(
            "validated V2 cognition output is required for surface routing"
        )
    validated_output = validate_cognition_core_output(cognition_output)
    return_value = validated_output["intention"]["route"] == "speech"
    return return_value


async def _action_results_for_state(
    state: GlobalPersonaState,
    *,
    executed_action_attempt_ids: set[str] | None = None,
) -> list[dict]:
    """Evaluate selected actions into traceable action results."""

    pre_surface_results = _pre_surface_action_results_for_state(state)
    pre_surface_attempt_ids = {
        action_attempt_id
        for row in pre_surface_results
        if (action_attempt_id := row.get("action_attempt_id"))
    }
    remaining_specs = [
        spec
        for spec in _selected_action_specs(state)
        if spec.get("kind") not in (
            ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
            FUTURE_SPEAK_CAPABILITY,
        )
        and _action_attempt_id_for_spec(spec) not in pre_surface_attempt_ids
    ]
    action_results = await execute_action_specs_for_trace(
        remaining_specs,
        storage_timestamp_utc=state["storage_timestamp_utc"],
        executed_action_attempt_ids=executed_action_attempt_ids,
        source_llm_trace_id=str(state.get("llm_trace_id") or ""),
        availability_snapshot_factory=(
            lambda _context: build_action_availability_snapshot(state)
        ),
    )
    return_value = [*pre_surface_results, *action_results]
    return return_value


async def _execute_pre_surface_action_results(
    state: GlobalPersonaState,
) -> list[dict]:
    """Execute selected non-surface actions before L3 wording begins."""

    action_specs = [
        spec
        for spec in _selected_action_specs(state)
        if spec.get("kind") not in (
            SPEAK_CAPABILITY,
            ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
            FUTURE_SPEAK_CAPABILITY,
        )
    ]
    if not action_specs:
        return []
    return await execute_action_specs_for_trace(
        action_specs,
        storage_timestamp_utc=state["storage_timestamp_utc"],
        source_llm_trace_id=str(state.get("llm_trace_id") or ""),
        availability_snapshot_factory=(
            lambda _context: build_action_availability_snapshot(state)
        ),
    )


def _action_attempt_id_for_spec(action_spec: dict) -> str:
    """Return the deterministic attempt id used to avoid a second execution."""

    eval_result = ActionSpecEvaluator().evaluate(action_spec)
    return action_attempt_id_from_eval_result(eval_result)


async def stage_2a_background_work_enqueue(
    state: GlobalPersonaState,
) -> dict:
    """Queue background requests before L3 builds acknowledgement."""

    background_specs = [
        spec
        for spec in _selected_action_specs(state)
        if spec.get("kind") in (
            ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
            FUTURE_SPEAK_CAPABILITY,
        )
    ]
    if not background_specs:
        return_value: dict[str, object] = {}
        return return_value
    if not _cognition_selects_text_surface(state):
        action_results = [
            _background_no_handoff_result(spec, state)
            for spec in background_specs
        ]
        return_value = {
            "pre_surface_action_results": action_results,
        }
        return return_value

    action_results = await execute_action_specs_for_trace(
        background_specs,
        storage_timestamp_utc=state["storage_timestamp_utc"],
        source_llm_trace_id=str(state.get("llm_trace_id") or ""),
        availability_snapshot_factory=(
            lambda _context: build_action_availability_snapshot(state)
        ),
    )
    return_value = {
        "pre_surface_action_results": action_results,
    }
    return return_value


def _background_no_handoff_result(
    action_spec: dict,
    state: GlobalPersonaState,
) -> dict:
    """Build a prompt-safe rejection when no visible acknowledgement exists."""

    action_kind = str(action_spec.get("kind", ""))
    action_attempt_id = _first_valid_action_attempt_id(state, action_kind)
    if action_attempt_id is None:
        action_attempt_id = ""
    params = action_spec.get("params")
    task_summary = ""
    if isinstance(params, dict):
        for field_name in (
            "task_brief",
            "objective_summary",
            "continuation_objective",
        ):
            value = params.get(field_name)
            if isinstance(value, str) and value.strip():
                task_summary = value.strip()
                break
    result = {
        "schema_version": "action_result.v1",
        "action_attempt_id": action_attempt_id,
        "action_kind": action_kind,
        "handler_owner": "background_work",
        "status": "failed",
        "result_summary": (
            f"{action_kind} failed: visible acknowledgement missing"
        ),
        "result_refs": [],
        "completed_at": state["storage_timestamp_utc"],
        "queue_state": "none",
        "task_summary": task_summary,
        "operational_owner": "none",
        "job_ref": "",
        "acknowledgement_constraint": "promise_forbidden_explain_failure",
    }
    return result


def _pre_surface_action_results_for_state(
    state: GlobalPersonaState,
) -> list[dict]:
    """Return pre-surface queue results already produced this episode."""

    raw_results = state.get("pre_surface_action_results")
    if not isinstance(raw_results, list):
        return_value: list[dict] = []
        return return_value
    results = [row for row in raw_results if isinstance(row, dict)]
    return results


def _episode_component_update(
    state: GlobalPersonaState,
    *,
    action_results: list[dict],
    surface_outputs: list[dict],
) -> dict:
    """Return action and surface components for the settlement owner."""

    trace_update = {
        "action_results": action_results,
        "surface_outputs": surface_outputs,
    }
    return trace_update


async def call_action_subgraph(state: GlobalPersonaState) -> dict:
    """Run selected text-surface directives and dialog.

    Args:
        state: Current persona graph state.

    Returns:
        Partial state update with dialog fragments and addressed users.
    """

    surface_state = dict(state)
    pre_surface_action_results = [
        *_pre_surface_action_results_for_state(state),
        *await _execute_pre_surface_action_results(state),
    ]
    surface_state["pre_surface_action_results"] = (
        pre_surface_action_results
    )
    surface_update = await call_l3_text_surface_handler(surface_state)
    surface_state["text_surface_input_v2"] = surface_update[
        "text_surface_input_v2"
    ]
    surface_state["text_surface_output_v2"] = surface_update[
        "text_surface_output_v2"
    ]
    speak_attempt_id = _first_valid_action_attempt_id(
        surface_state,
        SPEAK_CAPABILITY,
    )
    result = await dialog_agent(surface_state)
    final_dialog = result["final_dialog"]
    accepted_surface = result["text_surface_output_v2"]
    if not isinstance(accepted_surface, dict):
        raise StateContractError(
            "dialog result missing accepted text_surface_output_v2"
        )
    accepted_surface = validate_text_surface_output(accepted_surface)
    surface_state["text_surface_output_v2"] = accepted_surface
    surface_update["text_surface_output_v2"] = accepted_surface
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
    if "visual_surface_output_v2" in surface_update:
        visual_output = validate_visual_surface_output(
            surface_update["visual_surface_output_v2"]
        )
        surface_outputs.append(build_visual_surface_output(
            fragments=[visual_output["visual_directives"]],
            created_at=state["storage_timestamp_utc"],
        ))
    return_value = {
        "final_dialog": final_dialog,
        "target_addressed_user_ids": result["target_addressed_user_ids"],
        "target_broadcast": result["target_broadcast"],
    }
    return_value.update(surface_update)
    return_value.update(_episode_component_update(
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
        "target_addressed_user_ids": [],
        "target_broadcast": False,
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
    return_value.update(_episode_component_update(
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
    )
    return_value = rag_result
    return return_value


async def stage_1_goal_resolver(state: GlobalPersonaState) -> dict:
    """Run full resolver recurrence and commit only its final V2 state."""

    async def cognition_cycle(
        current_state: GlobalPersonaState,
    ) -> GlobalPersonaState:
        coordinator = current_cognition_retry_coordinator()
        if coordinator is None:
            update = await call_cognition_subgraph(
                current_state,
                commit=False,
            )
        else:
            update = await call_cognition_subgraph(
                current_state,
                commit=False,
                retry_coordinator=coordinator,
            )
        return_value = update
        return return_value

    action_context_state = await _load_live_action_selection_context(state)
    initialized = ensure_initial_resolver_inputs(
        action_context_state,
        max_cycles=COGNITION_RESOLVER_MAX_CYCLES,
    )
    initialized = await load_matching_pending_resume_into_state(initialized)
    resolved_state = await call_cognition_resolver_loop(
        initialized,
        call_cognition_subgraph_func=cognition_cycle,
        execute_capability_func=execute_resolver_capability_request,
        max_cycles=COGNITION_RESOLVER_MAX_CYCLES,
        capability_timeout_seconds=(
            COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS
        ),
        upsert_pending_resume_func=upsert_pending_resume,
        apply_pending_resolution_func=apply_pending_resolution,
    )
    core_output = resolved_state.get("cognition_core_output")
    if not isinstance(core_output, Mapping):
        raise ValueError("V2 resolver completed without cognition_core_output")
    state_update = core_output.get("state_update")
    if (
        isinstance(state_update, Mapping)
        and state_update.get("state_scope") == "character"
    ):
        expected_character_updated_at = resolved_state.get(
            "character_cognition_base_updated_at"
        )
        await commit_cognition_output(  # type: ignore[arg-type]
            core_output,
            expected_character_updated_at=(
                expected_character_updated_at
                if isinstance(expected_character_updated_at, str)
                else None
            ),
        )
    else:
        await commit_cognition_output(core_output)  # type: ignore[arg-type]
    resolved_state["cognition_state_committed"] = True
    return_value = dict(resolved_state)
    return return_value


async def _load_live_action_selection_context(
    state: GlobalPersonaState,
) -> GlobalPersonaState:
    """Load trusted prompt-safe coding-run contexts for one live user turn."""

    updated_state = dict(state)
    updated_state["action_selection_context"] = {"coding_runs": []}
    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        return updated_state  # type: ignore[return-value]
    if episode.get("trigger_source") != "user_message":
        return updated_state  # type: ignore[return-value]
    contexts = await load_open_coding_run_contexts_for_scope(
        source_platform=state["platform"],
        source_channel_id=state["platform_channel_id"],
        requester_global_user_id=state["global_user_id"],
        limit=3,
    )
    updated_state["action_selection_context"] = {
        "coding_runs": [dict(context) for context in contexts],
    }
    return updated_state  # type: ignore[return-value]


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

    ambient_logical_turns = state['ambient_logical_turns']
    if state['channel_type'] == 'group':
        ambient_logical_turns = filter_group_scene_ambient_turns(
            ambient_logical_turns=ambient_logical_turns,
            trigger_occurred_at=state['storage_timestamp_utc'],
        )
    interaction_logical_turns = state['interaction_logical_turns']
    ambient_history = logical_turns_as_history_rows(
        ambient_logical_turns
    )
    scope_users = _build_scope_users(
        state,
        ambient_history,
    )
    scene_participant_bindings = _build_scene_participant_bindings(
        state,
        scope_users,
    )
    public_group_scene = ''
    if state['channel_type'] == 'group':
        try:
            group_scene_context = build_group_scene_context(
                ambient_logical_turns=ambient_logical_turns,
                trigger_occurred_at=state['storage_timestamp_utc'],
                trigger_speaker_name=state['user_name'],
                trigger_body_text=state['prompt_message_context']['body_text'],
                trigger_addressed_global_user_ids=(
                    state['prompt_message_context'][
                        'addressed_to_global_user_ids'
                    ]
                ),
                trigger_reply_to_display_name=state['reply_context'].get(
                    'reply_to_display_name',
                    '',
                ),
                scope_users=scope_users,
                current_global_user_id=state['global_user_id'],
            )
            public_group_scene = project_group_scene_prompt(
                group_scene_context
            )
        except GroupSceneProjectionError as exc:
            failure = exc.as_failure()
            logger.warning(
                f'Group public-scene projection degraded: '
                f'code={failure["code"]} '
                f'protected_anchor_count={failure["protected_anchor_count"]}'
            )
            public_group_scene = ''
        except Exception as exc:
            logger.exception(
                f'Group public-scene projection failed; Cognition continues '
                f'with an empty public scene: {exc}'
            )
            public_group_scene = ''
    raw_interaction_wide = logical_turns_as_history_rows(
        interaction_logical_turns
    )
    interaction_history_wide = format_storage_utc_history_for_llm(
        raw_interaction_wide
    )
    interaction_history_recent = list(interaction_history_wide)

    async def stage_0_msg_decontextualizer(
        persona_state: GlobalPersonaState,
    ) -> dict:
        """Run decontextualization with recent channel history and identities."""

        decontextualizer_state = dict(persona_state)
        decontextualizer_state['ambient_logical_turns'] = (
            ambient_logical_turns
        )
        result = await call_msg_decontextualizer(decontextualizer_state)
        return_value = result
        return return_value

    # Build the top level graph that connect stages
    persona_builder = StateGraph(GlobalPersonaState)
    persona_builder.add_node(
        "stage_0_msg_decontextualizer",
        stage_0_msg_decontextualizer,
    )
    persona_builder.add_node("stage_1_goal_resolver", stage_1_goal_resolver)
    persona_builder.add_node(
        "stage_2_memory_lifecycle",
        call_memory_lifecycle_update_handler,
    )
    persona_builder.add_node(
        "stage_2a_background_work_enqueue",
        stage_2a_background_work_enqueue,
    )
    persona_builder.add_node("stage_3_action", call_action_subgraph)  # perform action
    persona_builder.add_node("stage_3_no_response", stage_3_no_response)
    persona_builder.add_edge(START, "stage_0_msg_decontextualizer")
    persona_builder.add_edge(
        "stage_0_msg_decontextualizer",
        "stage_1_goal_resolver",
    )
    persona_builder.add_edge(
        "stage_1_goal_resolver",
        "stage_2_memory_lifecycle",
    )
    persona_builder.add_edge(
        "stage_2_memory_lifecycle",
        "stage_2a_background_work_enqueue",
    )
    persona_builder.add_conditional_edges(
        "stage_2a_background_work_enqueue",
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
        "llm_trace_id": state.get("llm_trace_id", ""),
        "user_input": state["user_input"],
        "prompt_message_context": state["prompt_message_context"],
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "channel_type": state["channel_type"],
        "channel_name": state.get("channel_name", ""),
        "platform_message_id": state["platform_message_id"],
        "active_turn_platform_message_ids": list(
            state.get("active_turn_platform_message_ids", [])
        ),
        "active_turn_conversation_row_ids": list(
            state.get("active_turn_conversation_row_ids", [])
        ),
        "active_turn_conversation_source_refs": list(
            state.get("active_turn_conversation_source_refs", [])
        ),
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
        "scene_participant_bindings": scene_participant_bindings,
        "public_group_scene": public_group_scene,
        "conversation_episode_state": state.get("conversation_episode_state"),
        "conversation_progress": state.get("conversation_progress"),
        "ambient_logical_turns": ambient_logical_turns,
        "interaction_logical_turns": interaction_logical_turns,
        "conversation_progress_diagnostics": state.get(
            "conversation_progress_diagnostics"
        ),
        "promoted_reflection_context": state.get("promoted_reflection_context"),
        "internal_monologue_residue_context": state.get(
            "internal_monologue_residue_context",
            "",
        ),
        "past_dialog_cognition_context": state.get(
            "past_dialog_cognition_context",
            "",
        ),
        "referents": [],
        "debug_modes": state["debug_modes"],
        "should_respond": state["should_respond"],
    }
    cognitive_episode = state.get("cognitive_episode")
    if cognitive_episode is not None:
        initial_persona_state["cognitive_episode"] = cognitive_episode
    for identity_field in (
        "character_identity_revision_number",
        "character_identity_context",
        "character_identity_surface_context",
        "character_identity_projection_digest",
        "character_identity_consumer_kinds",
        "character_identity_episode_id",
        "character_identity_epistemic_core_included",
    ):
        if identity_field in state:
            initial_persona_state[identity_field] = state[identity_field]
    for turn_context_field in (
        "interaction_style_context",
        "settled_relevance_context_consumption",
    ):
        if turn_context_field in state:
            initial_persona_state[turn_context_field] = state[turn_context_field]
    
    results = await persona_graph.ainvoke(initial_persona_state)
    
    consolidation_state = dict(results)
    consolidation_state.pop("past_dialog_cognition_context", None)

    return_value = {
        "should_respond": results["should_respond"],
        "final_dialog": results["final_dialog"],
        "target_addressed_user_ids": results["target_addressed_user_ids"],
        "target_broadcast": bool(results["target_broadcast"]),
        "scope_users": results.get("scope_users", []),
        "future_promises": [],
        "cognition_core_output": results.get("cognition_core_output"),
        "cognition_state_update": results.get("cognition_state_update"),
        "cognition_state_committed": results.get(
            "cognition_state_committed",
            False,
        ),
        "consolidation_state": consolidation_state,
        "surface_outputs": results.get("surface_outputs", []),
        "action_results": results.get("action_results", []),
        "episode_trace": results.get("episode_trace"),
        "llm_trace_id": state.get("llm_trace_id", ""),
        "character_profile": results["character_profile"],
        "character_identity_revision_number": results.get(
            "character_identity_revision_number"
        ),
        "character_identity_context": results.get(
            "character_identity_context"
        ),
        "character_identity_surface_context": results.get(
            "character_identity_surface_context"
        ),
        "character_identity_projection_digest": results.get(
            "character_identity_projection_digest"
        ),
        "character_identity_consumer_kinds": results.get(
            "character_identity_consumer_kinds"
        ),
        "character_identity_episode_id": results.get(
            "character_identity_episode_id"
        ),
        "character_identity_epistemic_core_included": results.get(
            "character_identity_epistemic_core_included"
        ),
    }
    return return_value
