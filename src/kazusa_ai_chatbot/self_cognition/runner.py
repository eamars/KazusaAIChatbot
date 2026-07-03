"""Orchestrate one self-cognition tracking case."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any

from kazusa_ai_chatbot.action_spec.attempt_ledger import upsert_action_attempt
from kazusa_ai_chatbot.action_spec.execution import execute_action_specs_for_trace
from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
)
from kazusa_ai_chatbot.action_spec.results import build_episode_trace
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisode,
    validate_cognitive_episode,
)
from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS,
    COGNITION_RESOLVER_MAX_CYCLES,
)
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    execute_resolver_capability_request,
)
from kazusa_ai_chatbot.cognition_resolver.loop import call_cognition_resolver_loop
from kazusa_ai_chatbot.cognition_resolver.pending import (
    apply_pending_resolution,
    upsert_pending_resume,
)
from kazusa_ai_chatbot.nodes.dialog_agent import (
    DIALOG_USAGE_MODE_SELF_COGNITION_ACTION_CANDIDATE,
    StateContractError,
    dialog_agent,
    validate_dialog_action_directives,
)
from kazusa_ai_chatbot.consolidation.core import (
    call_consolidation_subgraph,
)
from kazusa_ai_chatbot.internal_monologue_residue import (
    load_residue_context,
    record_completed_episode_residue,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    call_cognition_subgraph,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    call_l3_text_surface_handler,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_memory_lifecycle import (
    call_memory_lifecycle_update_handler,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import (
    empty_user_memory_context,
)
from kazusa_ai_chatbot.runtime_coordination import PipelineRunHandle
from kazusa_ai_chatbot.self_cognition import models, projection, tracking
from kazusa_ai_chatbot.time_boundary import (
    build_turn_clock_from_storage_utc,
    format_storage_utc_for_llm,
)


SelfCognitionClient = Callable[[dict[str, Any]], Any]
ConsolidationBuildResult = tuple[dict[str, Any], dict[str, Any], bool]
SELF_COGNITION_PRIVATE_ACTION_KINDS = frozenset(
    (
        APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        TRIGGER_FUTURE_COGNITION_CAPABILITY,
    )
)


def build_self_cognition_case_artifacts(
    case: models.SelfCognitionCase,
    cognition_client: SelfCognitionClient | None = None,
    dialog_client: SelfCognitionClient | None = None,
    consolidation_client: SelfCognitionClient | None = None,
    *,
    apply_consolidation: bool = False,
    execute_private_actions: bool = False,
    pipeline_run_handle: PipelineRunHandle | None = None,
) -> dict[str, Any]:
    """Build one self-cognition case's tracking records in memory.

    Args:
        case: Self-cognition source data.
        cognition_client: Optional test seam for the shared cognition graph.
        dialog_client: Optional test seam for selected visible `speak` render.
        consolidation_client: Optional test seam for the shared consolidator.
        apply_consolidation: When true, call the shared consolidation seam
            with already-rendered dialog output when present.
        execute_private_actions: When true, execute selected private action
            specs through their deterministic owners.
        pipeline_run_handle: Optional cooperative cancellation handle.

    Returns:
        Artifact names mapped to JSON-like payloads or Markdown text.
    """

    artifact_payloads = asyncio.run(
        build_self_cognition_case_artifacts_async(
            case,
            cognition_client=cognition_client,
            dialog_client=dialog_client,
            consolidation_client=consolidation_client,
            apply_consolidation=apply_consolidation,
            execute_private_actions=execute_private_actions,
            pipeline_run_handle=pipeline_run_handle,
        )
    )
    return artifact_payloads


async def build_self_cognition_case_artifacts_async(
    case: models.SelfCognitionCase,
    cognition_client: SelfCognitionClient | None = None,
    dialog_client: SelfCognitionClient | None = None,
    consolidation_client: SelfCognitionClient | None = None,
    *,
    apply_consolidation: bool = False,
    execute_private_actions: bool = False,
    pipeline_run_handle: PipelineRunHandle | None = None,
) -> dict[str, Any]:
    """Async implementation for building self-cognition records in memory.

    Args:
        case: Self-cognition source data.
        cognition_client: Optional test seam for the shared cognition graph.
        dialog_client: Optional test seam for selected visible `speak` render.
        consolidation_client: Optional test seam for the shared consolidator.
        apply_consolidation: When true, call the shared consolidation seam
            with already-rendered dialog output when present.
        execute_private_actions: When true, execute selected private action
            specs through their deterministic owners.
        pipeline_run_handle: Optional cooperative cancellation handle.

    Returns:
        Artifact names mapped to JSON-like payloads or Markdown text.
    """

    if pipeline_run_handle is not None:
        pipeline_run_handle.raise_if_cancelled("before_case_artifacts")
    case_name = projection.validate_case_name(case)
    trigger_record = tracking.build_trigger_record(case)
    artifact_payloads: dict[str, Any] = {
        models.ARTIFACT_TRIGGER_RECORD: trigger_record,
    }

    if case_name == models.CASE_GROUP_NOISE_REJECTED:
        selected_route = models.ROUTE_AUDIT_ONLY
        budget = _budget(rag_calls=0, cognition_calls=0, dialog_calls=0)
        run_record = tracking.build_run_record(
            case,
            trigger_record,
            selected_route,
            budget,
        )
        route_effect = _route_effect_for_route(run_record, selected_route)
        artifact_payloads[models.ARTIFACT_RUN_RECORD] = run_record
        artifact_payloads[models.ARTIFACT_ROUTE_EFFECT] = route_effect
        artifact_payloads[models.ARTIFACT_LOOP_TRACE] = _loop_trace(
            case,
            run_record,
            route_effect,
        )
        return artifact_payloads

    if pipeline_run_handle is not None:
        pipeline_run_handle.raise_if_cancelled("before_source_packet")
    source_packet = projection.build_source_packet(case)
    rendered_packet = projection.render_source_packet_text(source_packet)
    cognition_input = {
        "source_packet": source_packet,
        "rendered_text": rendered_packet,
    }
    artifact_payloads[models.ARTIFACT_COGNITION_INPUT] = cognition_input

    active_cognition_client = cognition_client or _default_cognition_client
    if pipeline_run_handle is not None:
        pipeline_run_handle.raise_if_cancelled("before_cognition_context")
    residue_context = await _load_residue_context_for_case(case)
    cognition_state = _build_cognition_state(
        case,
        rendered_packet,
        residue_context=residue_context,
    )
    if pipeline_run_handle is not None:
        pipeline_run_handle.raise_if_cancelled("before_cognition")
    cognition_output = await _call_maybe_async(
        active_cognition_client,
        cognition_state,
    )
    if pipeline_run_handle is not None:
        pipeline_run_handle.raise_if_cancelled("after_cognition")
    if execute_private_actions:
        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled("before_private_actions")
        cognition_output = await _with_private_action_results(
            cognition_state,
            cognition_output,
        )
        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled("after_private_actions")
    artifact_payloads[models.ARTIFACT_COGNITION_OUTPUT] = cognition_output

    existing_attempts = _existing_attempts(case)
    selected_route = tracking.classify_route(case, cognition_output)
    action_attempt = None
    action_candidate = None
    dialog_output: dict[str, Any] | None = None
    dialog_calls = 0
    active_dialog_client = dialog_client or _default_dialog_client
    if selected_route == models.ROUTE_ACTION_CANDIDATE:
        action_attempt = tracking.build_action_attempt(
            case,
            trigger_record,
            existing_attempts,
        )
        selected_route = tracking.classify_route(
            case,
            cognition_output,
            action_attempt=action_attempt,
        )
        if action_attempt["status"] == models.ACTION_ATTEMPT_STATUS_CANDIDATE:
            if pipeline_run_handle is not None:
                pipeline_run_handle.raise_if_cancelled("before_dialog")
            dialog_state = await _build_dialog_state_with_text_surface(
                cognition_state,
                cognition_output,
                usage_mode=DIALOG_USAGE_MODE_SELF_COGNITION_ACTION_CANDIDATE,
            )
            dialog_output = await _call_maybe_async(
                active_dialog_client,
                dialog_state,
            )
            if pipeline_run_handle is not None:
                pipeline_run_handle.raise_if_cancelled("after_dialog")
            dialog_calls = models.DIALOG_RENDER_CALL_LIMIT
            action_text = _dialog_text(dialog_output)
            action_candidate = tracking.build_action_candidate(
                case,
                action_attempt,
                action_text,
            )
        artifact_payloads[models.ARTIFACT_ACTION_ATTEMPT] = action_attempt
        if action_candidate is not None:
            artifact_payloads[models.ARTIFACT_ACTION_CANDIDATE] = (
                action_candidate
            )

    if apply_consolidation:
        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled("before_consolidation")
        consolidation_state, dialog_output, dialog_called = (
            await _build_consolidation_ready_state(
                cognition_state,
                cognition_output,
                rendered_packet,
                dialog_output=dialog_output,
            )
        )
        if dialog_called:
            dialog_calls += models.DIALOG_RENDER_CALL_LIMIT
        active_consolidation_client = (
            consolidation_client or _default_consolidation_client
        )
        consolidation_result = await _call_maybe_async(
            active_consolidation_client,
            consolidation_state,
        )
        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled("after_consolidation")
        await record_completed_episode_residue(
            completed_state=consolidation_state,
            current_timestamp_utc=consolidation_state["storage_timestamp_utc"],
        )
        artifact_payloads[models.ARTIFACT_CONSOLIDATION_OUTCOME] = (
            tracking.build_consolidation_outcome_record(
                consolidation_state,
                consolidation_result,
            )
        )

    budget = _budget(
        rag_calls=_resolver_evidence_call_count(cognition_output),
        cognition_calls=1,
        dialog_calls=dialog_calls,
    )
    run_record = tracking.build_run_record(
        case,
        trigger_record,
        selected_route,
        budget,
    )
    route_effect = _route_effect_for_route(run_record, selected_route)
    artifact_payloads[models.ARTIFACT_RUN_RECORD] = run_record
    artifact_payloads[models.ARTIFACT_ROUTE_EFFECT] = route_effect
    artifact_payloads[models.ARTIFACT_LOOP_TRACE] = _loop_trace(
        case,
        run_record,
        route_effect,
        action_attempt=action_attempt,
        action_candidate=action_candidate,
    )
    return artifact_payloads


async def _load_residue_context_for_case(
    case: models.SelfCognitionCase,
) -> str:
    """Load prior residue for a self-cognition trigger without exposing rows."""

    target_scope = _target_scope(case)
    character_profile = _character_profile(case)
    character_id = _string_field(character_profile, "global_user_id")
    if not character_id:
        character_id = CHARACTER_GLOBAL_USER_ID
    idle_timestamp_utc = _string_field(case, "idle_timestamp_utc")
    if not idle_timestamp_utc:
        return_value = ""
        return return_value
    load_result = await load_residue_context(
        trigger_scope={
            "character_id": character_id,
            "platform": target_scope["platform"],
            "platform_channel_id": target_scope["platform_channel_id"],
            "channel_type": target_scope["channel_type"],
            "global_user_id": target_scope["user_id"] or "",
        },
        current_timestamp_utc=idle_timestamp_utc,
    )
    residue_context = load_result["internal_monologue_residue_context"]
    return residue_context


async def _default_cognition_client(state: dict[str, Any]) -> dict[str, Any]:
    """Call the shared cognition graph through the resolver loop.

    Args:
        state: Global persona state subset required by the cognition graph.

    Returns:
        Shared cognition output.
    """

    cognition_result = await call_cognition_resolver_loop(
        state,
        call_cognition_subgraph_func=call_cognition_subgraph,
        execute_capability_func=execute_resolver_capability_request,
        max_cycles=COGNITION_RESOLVER_MAX_CYCLES,
        capability_timeout_seconds=(
            COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS
        ),
        upsert_pending_resume_func=_non_persistent_pending_resume,
        apply_pending_resolution_func=_non_persistent_pending_resolution,
    )
    return cognition_result


async def _non_persistent_pending_resume(
    state: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Build a pending-resume payload without writing a ledger row."""

    pending_resume = await upsert_pending_resume(
        state,
        observation,
        upsert_action_attempt_func=_discard_action_attempt_record,
    )
    return pending_resume


async def _non_persistent_pending_resolution(
    state: dict[str, Any],
    resolution: dict[str, Any],
) -> dict[str, Any] | None:
    """Apply pending resolution against self-cognition's in-memory state only."""

    updated_row = await apply_pending_resolution(
        state,
        resolution,
        list_action_attempts_func=_empty_action_attempt_rows,
        upsert_action_attempt_func=_discard_action_attempt_record,
    )
    return updated_row


async def _discard_action_attempt_record(record: dict[str, Any]) -> None:
    """Accept a pending row callback without durable persistence."""

    del record


async def _empty_action_attempt_rows(*, limit: int) -> list[dict[str, Any]]:
    """Return no durable pending rows for internal self-cognition execution."""

    del limit
    rows: list[dict[str, Any]] = []
    return rows


async def _default_dialog_client(state: dict[str, Any]) -> dict[str, Any]:
    """Call the existing dialog renderer graph.

    Args:
        state: Global persona state merged with shared cognition output.

    Returns:
        Dialog graph result used as a local selected-speak render candidate.
    """

    dialog_result = await dialog_agent(state)
    return dialog_result


async def _default_consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
    """Call the existing post-dialog consolidator subgraph.

    Args:
        state: Self-cognition state after shared cognition and optional
            selected-speak rendering.

    Returns:
        Shared consolidator result with write metadata.
    """

    consolidation_result = await call_consolidation_subgraph(state)
    return consolidation_result


async def _call_maybe_async(
    callable_object: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call a sync or async test seam with a common awaitable contract."""

    result = callable_object(*args, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


async def _build_consolidation_ready_state(
    cognition_state: dict[str, Any],
    cognition_output: dict[str, Any],
    rendered_packet: str,
    *,
    dialog_output: dict[str, Any] | None,
) -> ConsolidationBuildResult:
    """Build same-path consolidation state from cognition and dialog output.

    Args:
        cognition_state: State originally sent into shared cognition.
        cognition_output: Shared cognition graph output.
        rendered_packet: Internal-thought evidence text used as the
            decontextualized consolidation input.
        dialog_output: Previously rendered dialog output, if the action route
            already needed it.

    Returns:
        Consolidation-ready state, dialog output, and whether a new dialog
        call was needed.
    """

    dialog_called = False
    active_dialog_output = dialog_output
    if active_dialog_output is None:
        active_dialog_output = {
            "final_dialog": [],
        }

    consolidation_state = _build_consolidation_state(
        cognition_state,
        cognition_output,
        active_dialog_output,
        rendered_packet,
    )
    return_value = (consolidation_state, active_dialog_output, dialog_called)
    return return_value


def _build_consolidation_state(
    cognition_state: dict[str, Any],
    cognition_output: dict[str, Any],
    dialog_output: dict[str, Any],
    rendered_packet: str,
) -> dict[str, Any]:
    """Merge cognition and rendered dialog payload for the consolidator."""

    consolidation_state = dict(cognition_state)
    consolidation_state.update(cognition_output)
    consolidation_state.update(dialog_output)
    consolidation_state["decontexualized_input"] = rendered_packet
    consolidation_state["final_dialog"] = _dialog_fragments(dialog_output)
    return consolidation_state


async def _with_private_action_results(
    cognition_state: dict[str, Any],
    cognition_output: dict[str, Any],
) -> dict[str, Any]:
    """Execute selected private actions before route and consolidation handling."""

    routed_output = await _with_memory_lifecycle_specialist_update(
        cognition_state,
        cognition_output,
    )
    private_specs = _private_action_specs(routed_output)
    if not private_specs:
        return_value = routed_output
        return return_value

    action_results = await execute_action_specs_for_trace(
        private_specs,
        storage_timestamp_utc=cognition_state["storage_timestamp_utc"],
        record_attempt_func=upsert_action_attempt,
    )
    updated_output = dict(routed_output)
    updated_output["action_results"] = action_results
    updated_output["episode_trace"] = _episode_trace_for_private_actions(
        cognition_state,
        updated_output,
        action_results,
    )
    return updated_output


async def _with_memory_lifecycle_specialist_update(
    cognition_state: dict[str, Any],
    cognition_output: dict[str, Any],
) -> dict[str, Any]:
    """Run lifecycle specialist when shared cognition selected the route."""

    if not _has_memory_lifecycle_route(cognition_output):
        return_value = cognition_output
        return return_value

    specialist_state = dict(cognition_state)
    specialist_state.update(cognition_output)
    specialist_update = await call_memory_lifecycle_update_handler(
        specialist_state,
    )
    if not specialist_update:
        return_value = cognition_output
        return return_value

    updated_output = dict(cognition_output)
    updated_output.update(specialist_update)
    return updated_output


def _has_memory_lifecycle_route(cognition_output: dict[str, Any]) -> bool:
    """Return whether cognition selected a lifecycle specialist route."""

    for action_spec in _action_specs(cognition_output):
        if action_spec.get("kind") == MEMORY_LIFECYCLE_UPDATE_CAPABILITY:
            return True
    return False


def _private_action_specs(cognition_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Return private non-surface actions selected by shared cognition."""

    raw_specs = cognition_output.get("action_specs")
    if not isinstance(raw_specs, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    specs: list[dict[str, Any]] = []
    for action_spec in raw_specs:
        if not isinstance(action_spec, dict):
            continue
        kind = action_spec.get("kind")
        if kind not in SELF_COGNITION_PRIVATE_ACTION_KINDS:
            continue
        if action_spec.get("visibility") != "private":
            continue
        specs.append(action_spec)
    return specs


def _action_specs(cognition_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Return materialized action specs selected by shared cognition."""

    raw_specs = cognition_output.get("action_specs")
    if not isinstance(raw_specs, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    specs = [spec for spec in raw_specs if isinstance(spec, dict)]
    return specs


def _episode_trace_for_private_actions(
    cognition_state: dict[str, Any],
    cognition_output: dict[str, Any],
    action_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build episode trace evidence for private self-cognition actions."""

    episode = cognition_state["cognitive_episode"]
    trace = build_episode_trace(
        episode_id=episode["episode_id"],
        trigger_source=episode["trigger_source"],
        created_at=cognition_state["storage_timestamp_utc"],
        action_specs=_action_specs(cognition_output),
        action_results=action_results,
        surface_outputs=[],
    )
    return trace


def _build_cognition_state(
    case: models.SelfCognitionCase,
    rendered_packet: str,
    *,
    residue_context: str = "",
) -> dict[str, Any]:
    """Build the shared cognition graph state for an idle source packet."""

    source_timestamp_utc = _string_field(case, "idle_timestamp_utc")
    turn_clock = build_turn_clock_from_storage_utc(source_timestamp_utc)
    storage_timestamp_utc = turn_clock["storage_timestamp_utc"]
    local_time_context = turn_clock["local_time_context"]
    target_scope = _target_scope(case)
    chat_history = _chat_history(case, target_scope)
    episode = _build_cognitive_episode(
        case,
        rendered_packet,
        storage_timestamp_utc=storage_timestamp_utc,
        local_time_context=local_time_context,
    )
    user_id = target_scope["user_id"] or ""
    state = {
        "character_profile": _character_profile(case),
        "storage_timestamp_utc": storage_timestamp_utc,
        "local_time_context": local_time_context,
        "user_input": models.SELF_COGNITION_INPUT_TEXT,
        "prompt_message_context": {
            "body_text": models.SELF_COGNITION_INPUT_TEXT,
            "addressed_to_global_user_ids": [],
            "broadcast": target_scope["channel_type"] == "group",
            "mentions": [],
            "attachments": [],
        },
        "cognitive_episode": episode,
        "user_multimedia_input": [],
        "platform": target_scope["platform"],
        "platform_channel_id": target_scope["platform_channel_id"],
        "channel_type": target_scope["channel_type"],
        "channel_name": "",
        "platform_message_id": f"self_cognition:{_string_field(case, 'case_id')}",
        "platform_user_id": user_id,
        "global_user_id": user_id,
        "user_name": _user_display_name(case, user_id),
        "user_profile": _user_profile(case),
        "platform_bot_id": _platform_bot_id(case),
        "chat_history_wide": chat_history,
        "chat_history_recent": chat_history,
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": _cognition_scene_topic(case),
        "conversation_progress": case.get("conversation_progress"),
        "promoted_reflection_context": case.get("promoted_reflection_context"),
        "internal_monologue_residue_context": residue_context,
        "debug_modes": {"no_visual_directives": True},
        "should_respond": False,
        "decontexualized_input": models.SELF_COGNITION_INPUT_TEXT,
        "referents": [],
        "rag_result": _rag_result(case),
        "internal_monologue": "",
        "action_directives": {},
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "",
        "logical_stance": "",
        "mood": "",
        "global_vibe": "",
        "reflection_summary": "",
        "subjective_appraisals": [],
        "affinity_delta": 0,
        "last_relationship_insight": "",
        "new_facts": [],
        "future_promises": [],
    }
    return state


def _cognition_scene_topic(case: models.SelfCognitionCase) -> str:
    """Return persona-scene topic, excluding group-review label carrier data."""

    if _string_field(case, "trigger_kind") == models.TRIGGER_GROUP_CHAT_REVIEW:
        return ""
    return _string_field(case, "channel_topic")


def _build_dialog_state(
    cognition_state: dict[str, Any],
    cognition_output: dict[str, Any],
    *,
    usage_mode: str,
) -> dict[str, Any]:
    """Merge cognition output into the dialog graph's input state."""

    dialog_state = dict(cognition_state)
    dialog_state.update(cognition_output)
    dialog_state["final_dialog"] = []
    dialog_state["dialog_usage_mode"] = usage_mode
    return dialog_state


async def _build_dialog_state_with_text_surface(
    cognition_state: dict[str, Any],
    cognition_output: dict[str, Any],
    *,
    usage_mode: str,
) -> dict[str, Any]:
    """Build dialog input, running selected L3 text directives when needed."""

    dialog_state = _build_dialog_state(
        cognition_state,
        cognition_output,
        usage_mode=usage_mode,
    )
    if _needs_text_surface_directives(dialog_state):
        surface_update = await _call_maybe_async(
            call_l3_text_surface_handler,
            dialog_state,
        )
        dialog_state.update(surface_update)
    _validate_self_cognition_dialog_state(
        dialog_state,
        usage_mode=usage_mode,
    )
    return dialog_state


def _validate_self_cognition_dialog_state(
    dialog_state: dict[str, Any],
    *,
    usage_mode: str,
) -> None:
    """Validate that self-cognition dialog has selected speak and directives.

    Args:
        dialog_state: State that will be passed to the dialog graph.
        usage_mode: Stable label describing why dialog is being rendered.

    Raises:
        StateContractError: If visible dialog is not backed by selected speak
            or the L3 directive payload is incomplete.
    """

    if not _has_selected_speak_action(dialog_state):
        raise StateContractError(
            f"self-cognition dialog state missing action_specs.speak "
            f"for usage_mode={usage_mode}"
        )
    validate_dialog_action_directives(dialog_state, usage_mode=usage_mode)


def _needs_text_surface_directives(state: dict[str, Any]) -> bool:
    """Return whether selected speak needs L3 directives before dialog."""

    if _has_collected_text_directives(state):
        return_value = False
        return return_value
    return_value = _has_selected_speak_action(state)
    return return_value


def _has_selected_speak_action(state: dict[str, Any]) -> bool:
    """Return whether L2d selected the text surface action."""

    raw_specs = state.get("action_specs")
    if not isinstance(raw_specs, list):
        return_value = False
        return return_value
    for action_spec in raw_specs:
        if not isinstance(action_spec, dict):
            continue
        kind = action_spec.get("kind")
        if kind == SPEAK_CAPABILITY:
            return_value = True
            return return_value
    return_value = False
    return return_value


def _has_collected_text_directives(state: dict[str, Any]) -> bool:
    """Return whether dialog already has collected L3 text directives."""

    action_directives = state.get("action_directives")
    if not isinstance(action_directives, dict):
        return_value = False
        return return_value
    linguistic_directives = action_directives.get("linguistic_directives")
    if not isinstance(linguistic_directives, dict):
        return_value = False
        return return_value
    contextual_directives = action_directives.get("contextual_directives")
    if not isinstance(contextual_directives, dict):
        return_value = False
        return return_value
    return_value = True
    return return_value


def _dialog_text(dialog_output: dict[str, Any]) -> str:
    """Extract candidate text from dialog graph output."""

    fragments = _dialog_fragments(dialog_output)
    text = "\n".join(fragments)
    return text


def _dialog_fragments(dialog_output: dict[str, Any]) -> list[str]:
    """Extract normalized final-dialog fragments from dialog output."""

    value = dialog_output.get("final_dialog")
    if not isinstance(value, list):
        return_value: list[str] = []
        return return_value
    fragments = [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    return_value = fragments
    return return_value


def _build_cognitive_episode(
    case: models.SelfCognitionCase,
    rendered_packet: str,
    *,
    storage_timestamp_utc: str,
    local_time_context: dict[str, str],
) -> CognitiveEpisode:
    """Represent the self-cognition source packet as an internal percept."""

    target_scope = _target_scope(case)
    user_id = target_scope["user_id"] or ""
    percept_content = json.dumps(
        {
            "residue": {
                "residue_id": f"self_cognition:{_string_field(case, 'case_id')}",
                "internal_monologue": rendered_packet,
            },
            "action_latch": {
                "status": "local_tracking",
                "outward_action": "allowed_to_be_considered",
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    episode: CognitiveEpisode = {
        "episode_id": f"self_cognition:tracking:{_string_field(case, 'case_id')}",
        "trigger_source": "internal_thought",
        "input_sources": ["internal_monologue"],
        "output_mode": "preview",
        "percepts": [
            {
                "percept_id": "self_cognition:source_packet",
                "input_source": "internal_monologue",
                "content": percept_content,
                "visibility": "model_visible",
                "metadata": {"source": "self_cognition_source_packet"},
            },
        ],
        "target_scope": {
            "platform": target_scope["platform"],
            "platform_channel_id": target_scope["platform_channel_id"],
            "channel_type": target_scope["channel_type"],
            "current_platform_user_id": user_id,
            "current_global_user_id": user_id,
            "current_display_name": _user_display_name(case, user_id),
            "target_addressed_user_ids": [user_id] if user_id else [],
            "target_broadcast": target_scope["channel_type"] == "group",
        },
        "origin_metadata": {
            "platform": target_scope["platform"],
            "platform_message_id": f"self_cognition:{_string_field(case, 'case_id')}",
            "active_turn_platform_message_ids": [],
            "active_turn_conversation_row_ids": [],
            "debug_modes": {"no_visual_directives": True},
        },
        "storage_timestamp_utc": storage_timestamp_utc,
        "local_time_context": local_time_context,
    }
    validate_cognitive_episode(episode)
    return episode


def _route_effect_for_route(
    run_record: dict[str, Any],
    route: str,
) -> dict[str, Any]:
    """Build the consumer effect for one selected route."""

    if route == models.ROUTE_ACTION_CANDIDATE:
        consumer = "local_action_candidate"
        effect_summary = (
            "Self-cognition action candidates are inspected and delivered "
            "through the dispatcher/runtime adapter bridge after dialog "
            "rendering."
        )
    elif route == models.ROUTE_PROGRESS_MAINTENANCE:
        consumer = "conversation_progress_candidate"
        effect_summary = (
            "Self-cognition keeps conversation progress visible; no write "
            "was performed."
        )
    else:
        consumer = "audit_log"
        effect_summary = (
            "Self-cognition recorded the observation only; no write was "
            "performed."
        )
    route_effect = tracking.build_route_effect(
        run_record,
        route,
        consumer,
        effect_summary,
        next_topic=models.EMPTY_ROUTE_EFFECT_NEXT_TOPIC,
    )
    return route_effect


def _loop_trace(
    case: models.SelfCognitionCase,
    run_record: dict[str, Any],
    route_effect: dict[str, Any],
    *,
    action_attempt: dict[str, Any] | None = None,
    action_candidate: dict[str, Any] | None = None,
) -> str:
    """Render a human-readable trace of the routing decision."""

    lines = [
        "# Self-Cognition Trace",
        "",
        f"- case_name: {_string_field(case, 'case_name')}",
        f"- trigger_id: {run_record['trigger_id']}",
        f"- run_id: {run_record['run_id']}",
        f"- selected_route: {run_record['selected_route']}",
        f"- consumer: {route_effect['consumer']}",
        f"- production_write: {route_effect['production_write']}",
    ]
    if action_attempt is not None:
        lines.append(f"- action_attempt_status: {action_attempt['status']}")
    if action_candidate is not None:
        lines.append("- action_candidate_written: true")
    else:
        lines.append("- action_candidate_written: false")
    trace = "\n".join(lines)
    return trace


def _budget(
    rag_calls: int,
    cognition_calls: int,
    dialog_calls: int,
) -> dict[str, int]:
    """Build local budget counters for the run record."""

    budget = {
        "rag_calls": rag_calls,
        "cognition_calls": cognition_calls,
        "dialog_calls": dialog_calls,
        "topic_limit": models.TOPIC_LIMIT,
    }
    return budget


def _resolver_evidence_call_count(cognition_output: dict[str, Any]) -> int:
    """Count resolver-selected evidence observations recorded by cognition."""

    resolver_state = cognition_output.get("resolver_state")
    if not isinstance(resolver_state, dict):
        return_value = 0
        return return_value
    observations = resolver_state.get("observations")
    if not isinstance(observations, list):
        return_value = 0
        return return_value

    retrieval_count = 0
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        capability_kind = observation.get("capability_kind")
        if capability_kind in {"local_context_recall", "public_answer_research"}:
            retrieval_count += 1
    return_value = retrieval_count
    return return_value


def _rag_result(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Return the baseline RAG projection before resolver-selected retrieval."""

    source_ref_commitments = _active_commitments_from_source_refs(case)
    memory_context = empty_user_memory_context()
    memory_context["active_commitments"] = source_ref_commitments
    return_value = {
        "answer": "",
        "user_image": {
            "user_memory_context": memory_context,
        },
        "user_memory_unit_candidates": [],
        "character_image": {},
        "third_party_profiles": [],
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "loop_count": 0,
            "unknown_slots": [],
            "dispatched": [],
        },
    }
    return return_value


def _active_commitments_from_source_refs(
    case: models.SelfCognitionCase,
) -> list[dict[str, Any]]:
    """Build deterministic active-commitment bindings from source refs."""

    trigger_kind = _string_field(case, "trigger_kind")
    if trigger_kind != models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK:
        return_value: list[dict[str, Any]] = []
        return return_value

    raw_refs = case.get("source_refs")
    if not isinstance(raw_refs, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    due_state = _string_field(case, "semantic_due_state")
    commitments: list[dict[str, Any]] = []
    for raw_ref in raw_refs:
        if not isinstance(raw_ref, dict):
            continue
        source_kind = _string_field(raw_ref, "source_kind")
        if source_kind != "user_memory_unit":
            continue
        unit_id = _string_field(raw_ref, "source_id")
        summary = _string_field(raw_ref, "summary")
        if not unit_id or not summary:
            continue
        commitment = {
            "unit_id": unit_id,
            "fact": summary,
            "summary": summary,
            "status": "active",
        }
        due_at = _string_field(raw_ref, "due_at")
        if due_at:
            local_due_at = format_storage_utc_for_llm(due_at)
            if local_due_at:
                commitment["due_at"] = local_due_at
        if due_state:
            commitment["due_state"] = due_state
        commitments.append(commitment)
    return_value = commitments
    return return_value


def _character_profile(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Return the supplied character profile or a self-cognition default."""

    value = case.get("character_profile")
    if isinstance(value, dict) and value:
        return_value = value
        return return_value

    profile = {
        "name": "active character",
        "mood": _string_field(case, "current_mood") or "neutral",
        "global_vibe": _string_field(case, "global_vibe") or "neutral",
        "reflection_summary": "",
        "personality_brief": {
            "mbti": "INFP",
            "logic": "relationship-aware and careful",
            "tempo": "measured",
            "defense": "soft boundary preservation",
            "quirks": "uses concise emotional cues",
            "taboos": "does not invent facts",
        },
        "boundary_profile": {
            "self_integrity": 0.5,
            "control_sensitivity": 0.5,
            "relational_override": 0.5,
            "control_intimacy_misread": 0.5,
            "authority_skepticism": 0.5,
            "compliance_strategy": "evade",
            "boundary_recovery": "rebound",
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.5,
            "hesitation_density": 0.5,
            "counter_questioning": 0.5,
            "softener_density": 0.5,
            "formalism_avoidance": 0.5,
            "abstraction_reframing": 0.5,
            "direct_assertion": 0.5,
            "emotional_leakage": 0.5,
            "rhythmic_bounce": 0.5,
            "self_deprecation": 0.5,
        },
    }
    return profile


def _user_profile(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Return the supplied user profile or a self-cognition default."""

    value = case.get("user_profile")
    if isinstance(value, dict) and value:
        return_value = value
        return return_value

    target_scope = _target_scope(case)
    if target_scope["user_id"]:
        return_value = {}
        return return_value

    if target_scope["channel_type"] == "group" and target_scope["user_id"] is None:
        display_name = "group audience"
    else:
        display_name = target_scope["user_id"] or "self cognition target"
    profile = {
        "affinity": models.DEFAULT_SELF_COGNITION_AFFINITY,
        "display_name": display_name,
        "last_relationship_insight": "",
    }
    return profile


def _target_scope(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Normalize the case target scope for graph state fields."""

    value = case.get("target_scope")
    if not isinstance(value, dict):
        value = {}
    platform = value.get("platform")
    platform_channel_id = value.get("platform_channel_id")
    channel_type = value.get("channel_type")
    user_id = value.get("user_id")
    scope = {
        "platform": platform if isinstance(platform, str) else "",
        "platform_channel_id": (
            platform_channel_id if isinstance(platform_channel_id, str) else ""
        ),
        "channel_type": channel_type if isinstance(channel_type, str) else "",
        "user_id": user_id if isinstance(user_id, str) else None,
    }
    return scope


def _existing_attempts(case: models.SelfCognitionCase) -> list[dict[str, Any]]:
    """Copy prior local action attempts supplied by the case file."""

    value = case.get("existing_attempts")
    if not isinstance(value, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    attempts = [
        dict(item)
        for item in value
        if isinstance(item, dict)
    ]
    return attempts


def _chat_history(
    case: models.SelfCognitionCase,
    target_scope: dict[str, Any],
) -> list[dict[str, Any]]:
    """Project visible context rows into the shared chat-history shape."""

    value = case.get("visible_context")
    if not isinstance(value, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    rows: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = _string_field(item, "role")
        body_text = _string_field(item, "body_text")
        if not body_text:
            body_text = _string_field(item, "text")
        if not role or not body_text:
            continue
        global_user_id = target_scope["user_id"] or ""
        if role == "assistant":
            global_user_id = (
                models.DEFAULT_SELF_COGNITION_ASSISTANT_GLOBAL_USER_ID
            )
        row = {
            "timestamp": format_storage_utc_for_llm(
                _string_field(item, "timestamp"),
            ),
            "role": role,
            "platform_user_id": global_user_id,
            "global_user_id": global_user_id,
            "display_name": _string_field(item, "display_name"),
            "body_text": body_text,
            "addressed_to_global_user_ids": [target_scope["user_id"]]
            if role == "assistant" and target_scope["user_id"]
            else [],
            "broadcast": False,
        }
        rows.append(row)
    return rows


def _character_name(case: models.SelfCognitionCase) -> str:
    """Read the active character name from the source profile."""

    profile = _character_profile(case)
    name = profile.get("name")
    if not isinstance(name, str):
        return_value = ""
        return return_value
    return_value = name
    return return_value


def _user_display_name(
    case: models.SelfCognitionCase,
    fallback_user_id: str,
) -> str:
    """Read the target display name with a stable fallback."""

    user_profile = _user_profile(case)
    display_name = user_profile.get("display_name")
    if isinstance(display_name, str) and display_name:
        return_value = display_name
    else:
        return_value = fallback_user_id
    return return_value


def _platform_bot_id(case: models.SelfCognitionCase) -> str:
    """Read the platform bot id used by the dialog graph."""

    value = case.get("platform_bot_id")
    if isinstance(value, str) and value:
        return_value = value
    else:
        return_value = "self_cognition_bot"
    return return_value


def _string_field(case: dict[str, Any], field_name: str) -> str:
    """Read an optional external string field safely."""

    value = case.get(field_name)
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value
    return return_value
