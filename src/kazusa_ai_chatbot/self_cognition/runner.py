"""Orchestrate one self-cognition tracking case."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.action_spec.attempt_ledger import upsert_action_attempt
from kazusa_ai_chatbot.action_spec.execution import execute_action_specs_for_trace
from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
    TRIGGER_FUTURE_COGNITION_CAPABILITY,
)
from kazusa_ai_chatbot.action_spec.results import (
    build_private_surface_output,
    build_text_surface_output,
)
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    bind_protected_chain_records,
    reset_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeV1,
    EvidenceRefV1,
    PerceptV1,
    TargetScopeV1,
    build_internal_thought_episode,
    build_scheduled_tick_episode,
    build_self_cognition_episode,
)
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    execute_resolver_capability_request,
)
from kazusa_ai_chatbot.cognition_resolver.loop import (
    call_cognition_resolver_loop,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    ensure_initial_resolver_inputs,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContractError,
    is_targetless_group_self_cognition_episode,
    validate_scheduled_future_speech_authority,
    validate_text_surface_output,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    resolve_state_scope,
)
from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS,
    COGNITION_RESOLVER_MAX_CYCLES,
)
from kazusa_ai_chatbot.consolidation.core import (
    call_consolidation_subgraph,
)
from kazusa_ai_chatbot.conversation_progress import (
    build_group_scene_context,
    project_group_scene_prompt,
)
from kazusa_ai_chatbot.db import build_interaction_style_context
from kazusa_ai_chatbot.db.self_cognition import (
    reserve_self_cognition_action_attempt,
)
from kazusa_ai_chatbot.internal_monologue_residue import (
    load_residue_context,
    record_completed_episode_residue,
)
from kazusa_ai_chatbot.nodes.dialog_agent import (
    DIALOG_USAGE_MODE_SELF_COGNITION_ACTION_CANDIDATE,
    StateContractError,
    dialog_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_action_availability_snapshot,
    call_cognition_subgraph,
    commit_cognition_output,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_actions import (
    materialize_semantic_action_requests,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    call_l3_text_surface_handler,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_memory_lifecycle import (
    call_memory_lifecycle_update_handler,
)
from kazusa_ai_chatbot.runtime_coordination import PipelineRunHandle
from kazusa_ai_chatbot.self_cognition import models, projection, tracking
from kazusa_ai_chatbot.time_boundary import (
    build_turn_clock_from_storage_utc,
    format_storage_utc_for_llm,
    normalize_storage_utc_iso,
    parse_storage_utc_datetime,
)

SelfCognitionClient = Callable[[dict[str, Any]], Any]
ConsolidationBuildResult = tuple[dict[str, Any], dict[str, Any], bool]
SELF_COGNITION_PRIVATE_ACTION_KINDS = frozenset(
    (
        APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        TRIGGER_FUTURE_COGNITION_CAPABILITY,
    )
)
logger = logging.getLogger(__name__)


def enforce_scheduled_authority_due_guard(
    case: models.SelfCognitionCase,
    now_utc: datetime,
) -> tuple[bool, list[str]]:
    """Enforce the deterministic scheduled due and identity guard.

    Before scheduled cognition starts, the run must carry a structurally valid
    authority whose trigger equals the run due time and whose trigger is not
    later than the current instant. Any failure fails closed without dialog
    or delivery.

    Args:
        case: Scheduled future-speak self-cognition case.
        now_utc: Current worker tick instant.

    Returns:
        A boolean pass flag and the bounded closed gate codes produced by the
        guard.
    """

    authority = case.get("scheduled_future_speech_authority")
    if not isinstance(authority, dict):
        return False, ["scheduled_authority_missing"]
    try:
        validate_scheduled_future_speech_authority(authority)
    except (CognitionContractError, ValueError):
        return False, ["scheduled_authority_invalid"]
    trigger_utc = authority["trigger"]["utc"]
    try:
        trigger_time = parse_storage_utc_datetime(trigger_utc)
    except ValueError:
        return False, ["scheduled_authority_invalid"]
    run_due_at = _scheduled_run_due_at(case)
    if not run_due_at:
        return False, ["scheduled_trigger_identity_mismatch"]
    try:
        run_due_time = parse_storage_utc_datetime(run_due_at)
    except ValueError:
        return False, ["scheduled_trigger_identity_mismatch"]
    if run_due_time != trigger_time:
        return False, ["scheduled_trigger_identity_mismatch"]
    if now_utc < trigger_time:
        return False, ["scheduled_due_not_reached"]
    return True, []


def is_scheduled_future_speech_case(
    case: models.SelfCognitionCase,
) -> bool:
    """Identify scheduled-speech cases by their carried authority field."""

    return (
        case.get("trigger_kind")
        == models.TRIGGER_SCHEDULED_FUTURE_COGNITION
        and "scheduled_future_speech_authority" in case
    )


def _scheduled_run_due_at(case: models.SelfCognitionCase) -> str:
    """Return the deterministic run due time carried by a scheduled case."""

    due_at = _string_field(case, "source_calendar_run_due_at")
    if due_at:
        return due_at
    source_refs = case.get("source_refs")
    if isinstance(source_refs, list) and source_refs:
        first_ref = source_refs[0]
        if isinstance(first_ref, dict):
            return _string_field(first_ref, "due_at")
    return ""


def build_self_cognition_case_artifacts(
    case: models.SelfCognitionCase,
    cognition_client: SelfCognitionClient | None = None,
    dialog_client: SelfCognitionClient | None = None,
    consolidation_client: SelfCognitionClient | None = None,
    *,
    apply_consolidation: bool = False,
    execute_private_actions: bool = False,
    pipeline_run_handle: PipelineRunHandle | None = None,
    reserve_action_attempt_func: Callable[..., Any] | None = None,
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
            reserve_action_attempt_func=reserve_action_attempt_func,
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
    reserve_action_attempt_func: Callable[..., Any] | None = None,
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

    try:
        projection.validate_case_contract(case)
    except (KeyError, TypeError, ValueError) as exc:
        raise StateContractError(
            f"self-cognition source state contract is invalid: {exc}"
        ) from exc

    if (
        case_name == models.CASE_SCHEDULED_FUTURE_COGNITION
        and is_scheduled_future_speech_case(case)
    ):
        now_utc = parse_storage_utc_datetime(
            _string_field(case, "idle_timestamp_utc")
        )
        guard_passed, guard_codes = enforce_scheduled_authority_due_guard(
            case,
            now_utc,
        )
        if not guard_passed:
            blocked_artifacts = _scheduled_guard_blocked_artifacts(
                case,
                trigger_record,
                guard_codes=guard_codes,
            )
            return blocked_artifacts

    if pipeline_run_handle is not None:
        pipeline_run_handle.raise_if_cancelled("before_source_packet")
    source_packet = projection.build_source_packet(case)
    rendered_packet = projection.render_source_packet_text(source_packet)
    state_scope, _ = resolve_state_scope(
        "self_cognition",
        _target_scope(case)["user_id"],
    )
    cognition_input = {
        "source_packet": source_packet,
        "rendered_text": rendered_packet,
        "state_scope": state_scope,
    }
    artifact_payloads[models.ARTIFACT_COGNITION_INPUT] = cognition_input

    active_cognition_client = cognition_client or _default_cognition_client
    if pipeline_run_handle is not None:
        pipeline_run_handle.raise_if_cancelled("before_cognition_context")
    residue_context = await _load_residue_context_for_case(case)
    interaction_style_context = await _prepare_interaction_style_context(
        case,
    )
    public_group_scene = _build_public_group_scene(case)
    cognition_state = _build_cognition_state(
        case,
        rendered_packet,
        residue_context=residue_context,
        public_group_scene=public_group_scene,
        interaction_style_context=interaction_style_context,
    )
    artifact_payloads[models.RUNTIME_COGNITIVE_EPISODE] = dict(
        cognition_state["cognitive_episode"]
    )
    if pipeline_run_handle is not None:
        pipeline_run_handle.raise_if_cancelled("before_cognition")
    trigger_id = _string_field(trigger_record, "trigger_id")
    if not trigger_id:
        raise StateContractError("self-cognition trigger_id is required")
    diagnostics_token = bind_protected_chain_records(
        run_id=f"self_cognition_run:{trigger_id}",
        source_kind="self_cognition",
        llm_trace_id=(
            _string_field(cognition_state, "llm_trace_id")
            or f"self_cognition_run:{trigger_id}"
        ),
        cognition_invocation_id=f"self_cognition_run:{trigger_id}",
    )
    try:
        cognition_output = await _call_maybe_async(
            active_cognition_client,
            cognition_state,
        )
    finally:
        reset_protected_chain_records(diagnostics_token)
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
    existing_attempts = _existing_attempts(case)
    response_outcome = tracking.evaluate_group_response_policy(
        case,
        cognition_output,
    )
    if response_outcome is not None:
        artifact_payloads[models.ARTIFACT_RESPONSE_OUTCOME] = dict(
            response_outcome
        )
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
        if (
            action_attempt["status"]
            == models.ACTION_ATTEMPT_STATUS_CANDIDATE
            and response_outcome is not None
        ):
            active_reserver = (
                reserve_action_attempt_func
                or reserve_self_cognition_action_attempt
            )
            reserved = await _call_maybe_async(
                active_reserver,
                _attempt_state(
                    action_attempt,
                    now=parse_storage_utc_datetime(
                        cognition_state["cognitive_episode"]["created_at"]
                    ),
                ),
            )
            if not reserved:
                action_attempt["status"] = (
                    models.ACTION_ATTEMPT_STATUS_DUPLICATE
                )
                selected_route = models.ROUTE_AUDIT_ONLY
                response_outcome = _response_outcome_update(
                    response_outcome,
                    policy_disposition=models.POLICY_DISPOSITION_REJECTED,
                    policy_reason="duplicate",
                    gate_code="duplicate_reservation",
                )
        if (
            action_attempt["status"]
            == models.ACTION_ATTEMPT_STATUS_CANDIDATE
        ):
            cognition_output = _materialize_canonical_self_speak_action(
                cognition_state,
                cognition_output,
                selected_route=selected_route,
            )
            if pipeline_run_handle is not None:
                pipeline_run_handle.raise_if_cancelled("before_dialog")
            dialog_calls = models.DIALOG_RENDER_CALL_LIMIT
            try:
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
                action_text = _dialog_text(dialog_output)
                action_candidate = tracking.build_action_candidate(
                    case,
                    action_attempt,
                    action_text,
                )
            except StateContractError:
                raise
            except (
                ValueError,
                RuntimeError,
                TimeoutError,
                ConnectionError,
                OSError,
            ):
                action_attempt["status"] = (
                    models.ACTION_ATTEMPT_STATUS_DELIVERY_FAILED
                )
                if response_outcome is not None:
                    response_outcome = _response_outcome_update(
                        response_outcome,
                        execution_disposition=(
                            models.EXECUTION_DISPOSITION_DIALOG_FAILED
                        ),
                        gate_code="dialog_failed",
                    )
                artifact_payloads[models.ARTIFACT_DISPATCH_RESULT] = {
                    "status": "dialog_failed",
                }
        artifact_payloads[models.ARTIFACT_ACTION_ATTEMPT] = action_attempt
        if action_candidate is not None:
            artifact_payloads[models.ARTIFACT_ACTION_CANDIDATE] = (
                action_candidate
            )

    cognition_output = dict(cognition_output)
    cognition_output["surface_outputs"] = _surface_outputs_for_case(
        cognition_output=cognition_output,
        dialog_output=dialog_output,
        action_attempt=action_attempt,
        created_at=cognition_state["cognitive_episode"]["created_at"],
    )
    artifact_payloads[models.ARTIFACT_COGNITION_OUTPUT] = cognition_output
    if response_outcome is not None:
        artifact_payloads[models.ARTIFACT_RESPONSE_OUTCOME] = dict(
            response_outcome
        )

    consolidation_state, dialog_output, dialog_called = (
        await _build_consolidation_ready_state(
            cognition_state,
            cognition_output,
            rendered_packet,
            dialog_output=dialog_output,
        )
    )
    artifact_payloads[models.RUNTIME_CONSOLIDATION_STATE] = (
        consolidation_state
    )

    if apply_consolidation:
        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled("before_consolidation")
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
        response_outcome=response_outcome,
    )
    route_effect = _route_effect_for_route(
        run_record,
        selected_route,
        response_outcome=response_outcome,
    )
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


def _materialize_canonical_self_speak_action(
    cognition_state: dict[str, Any],
    cognition_output: dict[str, Any],
    *,
    selected_route: str,
) -> dict[str, Any]:
    """Materialize a policy-approved canonical self-cognition speak action."""

    if selected_route != models.ROUTE_ACTION_CANDIDATE:
        return cognition_output
    core_output = cognition_output.get("cognition_core_output")
    if not isinstance(core_output, dict):
        core_output = cognition_output
    if core_output.get("schema_version") != "cognition_output.v3":
        return cognition_output
    episode = cognition_state.get("cognitive_episode")
    if not isinstance(episode, dict):
        return cognition_output
    plan = core_output.get("response_plan")
    if not isinstance(plan, Mapping):
        return cognition_output
    scheduled_speech = episode.get("trigger_source") == "scheduled_tick"
    self_response = plan.get("self_cognition_response")
    group_speech = (
        is_targetless_group_self_cognition_episode(episode)
        and isinstance(self_response, Mapping)
        and self_response.get("decision")
        == "propose_visible_reply"
    )
    if not scheduled_speech and not group_speech:
        return cognition_output
    raw_specs = cognition_output.get("action_specs")
    if "action_specs" in cognition_output and (
        not isinstance(raw_specs, list)
        or any(not isinstance(spec, Mapping) for spec in raw_specs)
    ):
        raise StateContractError("canonical self-cognition action_specs are invalid")
    if scheduled_speech and (
        not isinstance(self_response, Mapping)
        or self_response.get("decision") != "propose_visible_reply"
    ):
        return cognition_output
    if isinstance(raw_specs, list) and any(
        isinstance(spec, dict) and spec.get("kind") == SPEAK_CAPABILITY
        for spec in raw_specs
    ):
        return cognition_output
    response_goal = (
        self_response.get("response_goal")
        if isinstance(self_response, Mapping)
        else plan.get("response_goal")
    )
    response_reason = (
        self_response.get("reason")
        if isinstance(self_response, Mapping)
        else (
            core_output.get("active_character_goal", {}).get("reason", "")
            if isinstance(core_output.get("active_character_goal"), Mapping)
            else ""
        )
    )
    if not isinstance(response_goal, str) or not response_goal.strip():
        return cognition_output
    request = {
        "capability": SPEAK_CAPABILITY,
        "decision": "visible_reply",
        "detail": response_goal.strip(),
        "reason": str(response_reason or response_goal).strip(),
        "target_roles": [],
        "evidence_handles": [],
        "surface_role": "ordinary",
        "goal_continuation_ref": None,
    }
    speak_specs = materialize_semantic_action_requests(
        [request],
        cognition_state,
    )
    if len(speak_specs) != 1:
        raise StateContractError("canonical self-cognition speak action was not materialized")
    updated_output = dict(cognition_output)
    updated_output["action_specs"] = [
        *(raw_specs if isinstance(raw_specs, list) else []),
        dict(speak_specs[0]),
    ]
    return updated_output


def _scheduled_guard_blocked_artifacts(
    case: models.SelfCognitionCase,
    trigger_record: dict[str, Any],
    *,
    guard_codes: list[str],
) -> dict[str, Any]:
    """Build fail-closed artifacts when the due guard blocks a scheduled case."""

    gate_result: models.SelfCognitionScheduledGateResult = {
        "schema_version": models.SCHEDULED_GATE_RESULT_SCHEMA_VERSION,
        "disposition": models.SCHEDULED_GATE_DISPOSITION_SUPPRESSED,
        "gate_codes": list(guard_codes),
    }
    artifact_payloads: dict[str, Any] = {
        models.ARTIFACT_TRIGGER_RECORD: trigger_record,
        models.ARTIFACT_SCHEDULED_GATE_RESULT: gate_result,
    }
    run_record = tracking.build_run_record(
        case,
        trigger_record,
        models.ROUTE_AUDIT_ONLY,
        _budget(rag_calls=0, cognition_calls=0, dialog_calls=0),
    )
    route_effect = _route_effect_for_route(
        run_record,
        models.ROUTE_AUDIT_ONLY,
    )
    artifact_payloads[models.ARTIFACT_RUN_RECORD] = run_record
    artifact_payloads[models.ARTIFACT_ROUTE_EFFECT] = route_effect
    artifact_payloads[models.ARTIFACT_LOOP_TRACE] = _loop_trace(
        case,
        run_record,
        route_effect,
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
    """Call the shared cognition graph through the canonical resolver loop.

    Args:
        state: Global persona state subset required by the cognition graph.

    Returns:
        Shared cognition output.
    """

    async def cognition_cycle(
        current_state: dict[str, Any],
    ) -> dict[str, Any]:
        update = await call_cognition_subgraph(current_state, commit=False)
        return_value = dict(update)
        return return_value

    initialized = ensure_initial_resolver_inputs(
        state,  # type: ignore[arg-type]
        max_cycles=COGNITION_RESOLVER_MAX_CYCLES,
    )
    resolved_state = await call_cognition_resolver_loop(
        initialized,
        call_cognition_subgraph_func=cognition_cycle,  # type: ignore[arg-type]
        execute_capability_func=execute_resolver_capability_request,
        max_cycles=COGNITION_RESOLVER_MAX_CYCLES,
        capability_timeout_seconds=(
            COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS
        ),
    )
    core_output = resolved_state.get("cognition_core_output")
    if not isinstance(core_output, dict):
        raise StateContractError(
            "canonical self-cognition completed without cognition_core_output"
        )
    if core_output.get("schema_version") != "cognition_output.v3":
        raise StateContractError("canonical self-cognition output is invalid")
    state_projection = core_output.get("state_projection")
    if (
        isinstance(state_projection, dict)
        and state_projection.get("state_scope") == "character"
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


async def run_self_cognition_consolidation_async(
    state: dict[str, Any],
    consolidation_client: SelfCognitionClient | None = None,
) -> dict[str, Any]:
    """Run consolidation for a worker-settled self-cognition state."""

    active_client = consolidation_client or _default_consolidation_client
    result = await _call_maybe_async(active_client, state)
    if not isinstance(result, dict):
        raise StateContractError(
            "self-cognition consolidation result must be an object"
        )
    return result


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
    consolidation_state["decontextualized_input"] = rendered_packet
    consolidation_state["final_dialog"] = _dialog_fragments(dialog_output)
    return consolidation_state


def _surface_outputs_for_case(
    *,
    cognition_output: dict[str, Any],
    dialog_output: dict[str, Any] | None,
    action_attempt: dict[str, Any] | None,
    created_at: str,
) -> list[dict[str, Any]]:
    """Return canonical surface components for the worker settlement owner."""

    if isinstance(dialog_output, dict):
        raw_surface_outputs = dialog_output.get("surface_outputs")
        if isinstance(raw_surface_outputs, list):
            surface_outputs = [
                dict(surface_output)
                for surface_output in raw_surface_outputs
                if (
                    isinstance(surface_output, dict)
                    and surface_output.get("schema_version")
                    == "surface_output.v1"
                )
            ]
            if surface_outputs:
                return surface_outputs

    fragments = _dialog_fragments(dialog_output or {})
    action_attempt_id = None
    if isinstance(action_attempt, dict):
        raw_attempt_id = action_attempt.get("attempt_id")
        if isinstance(raw_attempt_id, str) and raw_attempt_id:
            action_attempt_id = raw_attempt_id
    if fragments:
        return [build_text_surface_output(
            fragments=fragments,
            created_at=created_at,
            action_attempt_id=action_attempt_id,
        )]

    action_specs = cognition_output.get("action_specs")
    if isinstance(action_specs, list) and action_specs:
        return [build_private_surface_output(
            summary="No visible text surface selected for this episode.",
            created_at=created_at,
        )]
    return [build_private_surface_output(
        summary="Self-cognition completed without a visible surface.",
        created_at=created_at,
    )]


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
        source_llm_trace_id=str(cognition_state.get("llm_trace_id") or ""),
        availability_snapshot_factory=(
            lambda _context: build_action_availability_snapshot(
                cognition_state,
            )
        ),
    )
    updated_output = dict(routed_output)
    updated_output["action_results"] = action_results
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


async def _prepare_interaction_style_context(
    case: models.SelfCognitionCase,
) -> dict[str, Any] | None:
    """Load one immutable style snapshot for a bound text-surface case."""

    if case.get("target_binding_status") == "failed":
        return_value = None
        return return_value
    if not isinstance(case.get("delivery_target"), dict):
        return_value = None
        return return_value

    target_scope = _target_scope(case)
    channel_type = target_scope["channel_type"]
    if channel_type not in {"private", "group"}:
        return_value = None
        return return_value

    try:
        snapshot = await build_interaction_style_context(
            global_user_id=target_scope["user_id"] or "",
            channel_type=channel_type,
            platform=target_scope["platform"],
            platform_channel_id=target_scope["platform_channel_id"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StateContractError(
            f"self-cognition interaction style snapshot preparation failed: "
            f"{exc}"
        ) from exc

    require_group_engagement = (
        channel_type == "group" and target_scope["user_id"] == ""
    )
    validated_snapshot = _validate_interaction_style_snapshot(
        snapshot,
        require_group_engagement=require_group_engagement,
    )
    return validated_snapshot


def _validate_interaction_style_snapshot(
    snapshot: object,
    *,
    require_group_engagement: bool,
) -> dict[str, Any]:
    """Validate the immutable snapshot fields consumed by V2 and L3."""

    if not isinstance(snapshot, dict):
        raise StateContractError(
            "self-cognition interaction style snapshot must be an object"
        )
    if snapshot.get("schema_version") != (
        "interaction_style_turn_snapshot.v1"
    ):
        raise StateContractError(
            "self-cognition interaction style snapshot schema is invalid"
        )
    application_order = snapshot.get("application_order")
    surface = snapshot.get("surface")
    if not isinstance(application_order, list):
        raise StateContractError(
            "self-cognition interaction style application order is required"
        )
    if not isinstance(surface, Mapping):
        raise StateContractError(
            "self-cognition interaction style surface projection is required"
        )
    allowed_scopes = {"user", "group_channel"}
    if any(
        not isinstance(scope_name, str) or scope_name not in allowed_scopes
        for scope_name in application_order
    ):
        raise StateContractError(
            "self-cognition interaction style scope is invalid"
        )
    if len(set(application_order)) != len(application_order):
        raise StateContractError(
            "self-cognition interaction style scopes are duplicated"
        )
    surface_scope_names = set(surface)
    if any(
        not isinstance(scope_name, str)
        or scope_name not in allowed_scopes
        or scope_name not in application_order
        for scope_name in surface_scope_names
    ) or surface_scope_names != set(application_order):
        raise StateContractError(
            "self-cognition interaction style surface scopes are invalid"
        )

    for scope_name in application_order:
        source_projection = surface.get(scope_name)
        if not isinstance(source_projection, Mapping):
            raise StateContractError(
                "self-cognition interaction style source projection is "
                "required"
            )
        overlay = source_projection.get("overlay")
        if not isinstance(overlay, Mapping):
            raise StateContractError(
                "self-cognition interaction style overlay is required"
            )
        for field_name in (
            "speech_guidelines",
            "social_guidelines",
            "pacing_guidelines",
            "engagement_guidelines",
        ):
            guidelines = overlay.get(field_name)
            if not isinstance(guidelines, list):
                raise StateContractError(
                    "self-cognition interaction style guidelines are invalid"
                )
            if any(
                not isinstance(guideline, str) or not guideline.strip()
                for guideline in guidelines
            ):
                raise StateContractError(
                    "self-cognition interaction style guideline is invalid"
                )
        confidence = overlay.get("confidence")
        if not isinstance(confidence, str):
            raise StateContractError(
                "self-cognition interaction style confidence is invalid"
            )

    if require_group_engagement:
        group_engagement_context = snapshot.get(
            "group_engagement_action_context"
        )
        if not isinstance(group_engagement_context, Mapping):
            raise StateContractError(
                "self-cognition group engagement snapshot projection is "
                "required"
            )
        engagement_guidelines = group_engagement_context.get(
            "engagement_guidelines"
        )
        confidence = group_engagement_context.get("confidence")
        if not isinstance(engagement_guidelines, list):
            raise StateContractError(
                "self-cognition group engagement guidelines are invalid"
            )
        if any(
            not isinstance(guideline, str) or not guideline.strip()
            for guideline in engagement_guidelines
        ):
            raise StateContractError(
                "self-cognition group engagement guideline is invalid"
            )
        if not isinstance(confidence, str):
            raise StateContractError(
                "self-cognition group engagement confidence is invalid"
            )
    return snapshot


def _build_public_group_scene(case: models.SelfCognitionCase) -> str:
    """Project a group review window through the canonical scene contract."""

    if _string_field(case, "trigger_kind") != models.TRIGGER_GROUP_CHAT_REVIEW:
        return_value = ""
        return return_value
    source_context = case.get("source_context")
    if not isinstance(source_context, dict):
        return_value = ""
        return return_value
    if source_context.get("context_kind") != "group_chat_review":
        return_value = ""
        return return_value

    visible_rows = _group_scene_visible_rows(case)
    if not visible_rows:
        return_value = ""
        return return_value
    ambient_rows = visible_rows[:-1]
    trigger_row = visible_rows[-1]
    ambient_logical_turns = [
        {
            "turn_id": f"self-cognition-scene:{index}",
            "occurred_at": row["timestamp"],
            "role": row["role"],
            "display_name": "",
            "fragments": [row["body_text"]],
            "conversation_row_ids": [],
            "llm_trace_id": "",
            "platform_user_id": "",
            "global_user_id": "",
            "addressed_to_global_user_ids": [],
            "broadcast": False,
            "reply_context": {},
        }
        for index, row in enumerate(ambient_rows)
    ]
    group_scene_context = build_group_scene_context(
        ambient_logical_turns=ambient_logical_turns,
        trigger_occurred_at=trigger_row["timestamp"],
        trigger_speaker_name="",
        trigger_body_text=trigger_row["body_text"],
        trigger_addressed_global_user_ids=[],
        trigger_reply_to_display_name="",
        scope_users=[],
    )
    public_group_scene = project_group_scene_prompt(group_scene_context)
    return public_group_scene


def _group_scene_visible_rows(
    case: models.SelfCognitionCase,
) -> list[models.SelfCognitionVisibleContextRow]:
    """Allowlist and chronologically order rows used for public scene input."""

    value = case.get("visible_context")
    if not isinstance(value, list):
        return_value: list[models.SelfCognitionVisibleContextRow] = []
        return return_value

    valid_rows: list[
        tuple[object, int, models.SelfCognitionVisibleContextRow]
    ] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        body_text = item.get("body_text")
        timestamp = item.get("timestamp")
        if not isinstance(body_text, str) or not body_text.strip():
            continue
        if not isinstance(timestamp, str) or not timestamp.strip():
            continue
        try:
            parsed_timestamp = parse_storage_utc_datetime(timestamp)
        except ValueError:
            continue
        safe_row: models.SelfCognitionVisibleContextRow = {
            "role": (
                item["role"]
                if isinstance(item.get("role"), str)
                else ""
            ),
            "display_name": "",
            "timestamp": normalize_storage_utc_iso(timestamp),
            "body_text": body_text.strip(),
        }
        valid_rows.append((parsed_timestamp, index, safe_row))
    valid_rows.sort(key=lambda item: (item[0], item[1]))
    return_value = [item[2] for item in valid_rows]
    return return_value


def _build_cognition_state(
    case: models.SelfCognitionCase,
    rendered_packet: str,
    *,
    residue_context: str = "",
    public_group_scene: str | None = None,
    interaction_style_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared cognition graph state for an idle source packet."""

    if public_group_scene is None:
        public_group_scene = _build_public_group_scene(case)
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
        "llm_trace_id": _string_field(case, "llm_trace_id"),
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
        "source_context": case.get("source_context"),
        "public_group_scene": public_group_scene,
        "promoted_reflection_context": case.get("promoted_reflection_context"),
        "internal_monologue_residue_context": residue_context,
        "debug_modes": {"no_visual_directives": True},
        "should_respond": False,
        "decontextualized_input": models.SELF_COGNITION_INPUT_TEXT,
        "referents": [],
        "internal_monologue": "",
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "character_intent": "",
        "logical_stance": "",
        "new_facts": [],
        "future_promises": [],
    }
    if interaction_style_context is not None:
        state["interaction_style_context"] = interaction_style_context
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
    interaction_style_context = cognition_state.get(
        "interaction_style_context"
    )
    if interaction_style_context is not None:
        dialog_state["interaction_style_context"] = interaction_style_context
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
    surface_output = dialog_state.get("text_surface_output_v2")
    if not isinstance(surface_output, dict):
        raise StateContractError(
            f"self-cognition dialog state missing text_surface_output_v2 "
            f"for usage_mode={usage_mode}"
        )
    validate_text_surface_output(surface_output)


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

    return isinstance(state.get("text_surface_output_v2"), dict)


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
) -> CognitiveEpisodeV1:
    """Build the canonical self-cognition or scheduled source episode."""

    target_scope = _target_scope(case)
    user_id = target_scope["user_id"] or ""
    trigger_kind = _string_field(case, "trigger_kind")
    source_kind = (
        "scheduled_tick"
        if trigger_kind in {
            models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
            models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
        }
        else "self_cognition"
    )
    canonical_target_scope: TargetScopeV1 = {
        "platform": target_scope["platform"],
        "platform_channel_id": target_scope["platform_channel_id"],
        "channel_type": target_scope["channel_type"],
        "current_platform_user_id": user_id,
        "current_global_user_id": user_id,
        "current_display_name": _user_display_name(case, user_id),
        "target_addressed_user_ids": [user_id] if user_id else [],
        "target_broadcast": target_scope["channel_type"] == "group",
    }
    evidence_ref: EvidenceRefV1 = {
        "schema_version": "evidence_ref.v1",
        "evidence_kind": "system_event",
        "evidence_id": _string_field(case, "case_id"),
        "owner": "self_cognition.sources",
        "excerpt": rendered_packet[:800],
        "observed_at": storage_timestamp_utc,
    }
    builder_case = dict(case)
    builder_case["target_scope"] = canonical_target_scope
    percept: PerceptV1 = {
        "schema_version": "percept.v1",
        "percept_kind": "self_cognition_context",
        "source_kind": "scheduled_event" if source_kind == "scheduled_tick" else source_kind,
        "source_id": _string_field(case, "case_id"),
        "content": {
            "semantic_text": rendered_packet,
            "trigger_kind": trigger_kind,
        },
        "observed_at": storage_timestamp_utc,
    }
    latch = case.get("internal_action_latch")
    claim_token = _string_field(case, "claim_token")
    if isinstance(latch, dict) and claim_token:
        episode = build_internal_thought_episode(
            latch=latch,
            evidence_refs=[evidence_ref],
            local_time_context=local_time_context,
            created_at=storage_timestamp_utc,
            claim_token=claim_token,
        )
    elif source_kind == "scheduled_tick":
        calendar_run = case.get("calendar_run")
        if not isinstance(calendar_run, dict):
            calendar_run = {
                "run_id": _string_field(case, "source_calendar_run_id"),
                "schedule_id": "",
                "due_at": storage_timestamp_utc,
            }
        episode = build_scheduled_tick_episode(
            case=builder_case,
            calendar_run=calendar_run,
            percepts=[percept],
            evidence_refs=[evidence_ref],
            local_time_context=local_time_context,
            created_at=storage_timestamp_utc,
        )
    else:
        episode = build_self_cognition_episode(
            case=builder_case,
            percepts=[percept],
            evidence_refs=[evidence_ref],
            local_time_context=local_time_context,
            created_at=storage_timestamp_utc,
        )
    episode["origin_metadata"]["debug_modes"] = {
        "no_visual_directives": True,
    }
    return episode


def _route_effect_for_route(
    run_record: dict[str, Any],
    route: str,
    *,
    response_outcome: Mapping[str, Any] | None = None,
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
        response_outcome=response_outcome,
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
    scheduled_authority_id = run_record.get("scheduled_authority_id")
    if isinstance(scheduled_authority_id, str) and scheduled_authority_id:
        lines.append(f"- scheduled_authority_id: {scheduled_authority_id}")
    scheduled_gate_trace = run_record.get("scheduled_gate_trace")
    if isinstance(scheduled_gate_trace, dict):
        lines.append(
            f"- scheduled_gate_disposition: "
            f"{scheduled_gate_trace.get('gate_disposition', '')}"
        )
        gate_codes = scheduled_gate_trace.get("gate_codes")
        if isinstance(gate_codes, list):
            lines.append(f"- scheduled_gate_codes: {gate_codes}")
    if action_attempt is not None:
        lines.append(f"- action_attempt_status: {action_attempt['status']}")
    if action_candidate is not None:
        lines.append("- action_candidate_written: true")
    else:
        lines.append("- action_candidate_written: false")
    trace = "\n".join(lines)
    return trace


def _response_outcome_update(
    outcome: Mapping[str, Any],
    *,
    semantic_disposition: str | None = None,
    policy_disposition: str | None = None,
    execution_disposition: str | None = None,
    policy_reason: str | None = None,
    gate_code: str | None = None,
) -> dict[str, Any]:
    """Apply one bounded runtime disposition update to response metadata."""

    updated = dict(outcome)
    if semantic_disposition is not None:
        updated["semantic_disposition"] = semantic_disposition
    if policy_disposition is not None:
        updated["policy_disposition"] = policy_disposition
    if execution_disposition is not None:
        updated["execution_disposition"] = execution_disposition
    if policy_reason is not None:
        updated["policy_reason"] = policy_reason
    raw_codes = updated.get("response_gate_codes")
    gate_codes = list(raw_codes) if isinstance(raw_codes, list) else []
    if (
        policy_disposition == models.POLICY_DISPOSITION_REJECTED
        and gate_code == "duplicate_reservation"
    ):
        gate_codes = [
            code for code in gate_codes if code != "approved_for_dialog"
        ]
    if gate_code and gate_code not in gate_codes:
        gate_codes.append(gate_code)
    updated["response_gate_codes"] = gate_codes[
        :models.RESPONSE_GATE_CODE_LIMIT
    ]
    return updated


def _attempt_state(
    action_attempt: dict[str, Any],
    *,
    now: datetime,
) -> dict[str, Any]:
    """Build the persisted reservation row for one action attempt."""

    attempt_state = dict(action_attempt)
    attempt_state["recorded_at"] = now.isoformat()
    return attempt_state


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

    observations = cognition_output.get("resolver_observations")
    if not isinstance(observations, list):
        return_value = 0
        return return_value

    retrieval_count = 0
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        capability_kind = observation.get("capability_kind")
        if capability_kind == "task_resolution_request":
            retrieval_count += 1
    return_value = retrieval_count
    return return_value


def _character_profile(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Return the supplied character profile or a self-cognition default."""

    value = case.get("character_profile")
    if isinstance(value, dict) and value:
        return_value = value
        return return_value

    profile = {
        "name": "active character",
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
    profile = {"display_name": display_name}
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
