"""Bounded recurrence controller around the preserved cognition subgraph."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from kazusa_ai_chatbot.action_spec.models import (
    ACTION_CONTINUATION_VERSION,
    ACTION_SOURCE_REF_VERSION,
    ACTION_SPEC_VERSION,
    ACTION_TARGET_VERSION,
    SurfaceRoleV1,
    validate_action_spec,
)
from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY
from kazusa_ai_chatbot.cognition_episode import (
    CognitiveEpisodeValidationError,
    GoalContinuationRefV1,
    build_goal_continuation_ref,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContractError,
    _validate_scene_context,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_CYCLE_TRACE_VERSION,
    RESOLVER_EVIDENCE_STATE_VERSION,
    RESOLVER_OBSERVATION_VERSION,
    ResolverCapabilityRequestV1,
    ResolverCycleStateV1,
    ResolverObservationV1,
    ResolverPendingResolutionV1,
    ResolverPendingResumeV1,
    ResolverValidationError,
    validate_required_resolver_evidence_dependency,
    validate_resolver_capability_request,
    validate_resolver_goal_progress,
    validate_resolver_observation,
    validate_resolver_pending_resolution,
)
from kazusa_ai_chatbot.cognition_resolver.pending import (
    apply_pending_resolution,
    upsert_pending_resume,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    append_cycle_trace,
    append_observation,
    ensure_initial_resolver_inputs,
    project_resolver_context,
    update_goal_progress,
    validate_resolver_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.past_dialog_cognition import (
    build_past_dialog_cognition_context_from_rag_result,
)

CognitionSubgraphFunc = Callable[
    [GlobalPersonaState],
    Awaitable[GlobalPersonaState],
]
CapabilityExecutorFunc = Callable[
    [ResolverCapabilityRequestV1, GlobalPersonaState],
    Awaitable[ResolverObservationV1],
]
PendingResumeUpsertFunc = Callable[
    [GlobalPersonaState, ResolverObservationV1],
    Awaitable[ResolverPendingResumeV1],
]
PendingResolutionApplyFunc = Callable[
    [GlobalPersonaState, ResolverPendingResolutionV1],
    Awaitable[object],
]

MAX_CYCLE_OBSERVATION_ID = "resolver_obs_max_cycles"
DUPLICATE_REQUEST_OBSERVATION_ID = "resolver_obs_duplicate_request"
ANSWERABLE_NOW_TERMINAL_REASON = (
    "goal answerable now; optional resolver request suppressed"
)
USER_INPUT_BLOCKER_TERMINAL_REASON = (
    "blocked user-input resolver request converted to clarification surface"
)
USER_INPUT_BLOCKER_PRIVATE_TERMINAL_REASON = (
    "blocked user-input resolver request kept private for non-user source"
)
BLOCKED_PENDING_CAPABILITIES = frozenset((
    "human_clarification",
    "approval_preparation",
))

logger = logging.getLogger(__name__)


async def call_cognition_resolver_loop(
    state: GlobalPersonaState,
    *,
    call_cognition_subgraph_func: CognitionSubgraphFunc,
    execute_capability_func: CapabilityExecutorFunc,
    max_cycles: int,
    capability_timeout_seconds: float,
    upsert_pending_resume_func: PendingResumeUpsertFunc = upsert_pending_resume,
    apply_pending_resolution_func: PendingResolutionApplyFunc = (
        apply_pending_resolution
    ),
) -> GlobalPersonaState:
    """Run cognition, deterministic capability observation, then cognition again."""

    _validate_loop_limits(max_cycles, capability_timeout_seconds)
    current_state = ensure_initial_resolver_inputs(
        state,
        max_cycles=max_cycles,
    )

    while _resolver_state(current_state)["cycle_index"] < max_cycles:
        status_before = _resolver_state(current_state)["status"]
        cognition_output = await call_cognition_subgraph_func(current_state)
        cognition_state = _merge_state(current_state, cognition_output)
        cognition_state = _sync_goal_progress_from_cognition(cognition_state)
        lifecycle_conflict = _final_lifecycle_conflict(cognition_state)
        if lifecycle_conflict is not None:
            selected_request, reason = lifecycle_conflict
            final_state = await _run_mixed_lifecycle_final_cognition(
                cognition_state,
                selected_request=selected_request,
                status_before=status_before,
                conflict_reason=reason,
                call_cognition_subgraph_func=call_cognition_subgraph_func,
                apply_pending_resolution_func=apply_pending_resolution_func,
            )
            return final_state
        resolver_state = _resolver_state(cognition_state)
        if _suppress_optional_resolver_request(cognition_state):
            final_state = await _finalize_without_capability(
                cognition_state,
                resolver_state=resolver_state,
                status_before=status_before,
                apply_pending_resolution_func=apply_pending_resolution_func,
                terminal_reason=ANSWERABLE_NOW_TERMINAL_REASON,
            )
            return_value = await _enforce_final_lifecycle_state(
                final_state,
                status_before=status_before,
                call_cognition_subgraph_func=call_cognition_subgraph_func,
                apply_pending_resolution_func=apply_pending_resolution_func,
            )
            return return_value
        selected_request = _select_immediate_request(cognition_state)

        if selected_request is None:
            final_state = await _finalize_without_capability(
                cognition_state,
                resolver_state=resolver_state,
                status_before=status_before,
                apply_pending_resolution_func=apply_pending_resolution_func,
            )
            return_value = await _enforce_final_lifecycle_state(
                final_state,
                status_before=status_before,
                call_cognition_subgraph_func=call_cognition_subgraph_func,
                apply_pending_resolution_func=apply_pending_resolution_func,
            )
            return return_value

        if _is_repeated_capability_request(selected_request, resolver_state):
            final_state = await _run_duplicate_request_final_cognition(
                cognition_state,
                selected_request=selected_request,
                resolver_state=resolver_state,
                status_before=status_before,
                call_cognition_subgraph_func=call_cognition_subgraph_func,
                apply_pending_resolution_func=apply_pending_resolution_func,
            )
            return_value = await _enforce_final_lifecycle_state(
                final_state,
                status_before=status_before,
                call_cognition_subgraph_func=call_cognition_subgraph_func,
                apply_pending_resolution_func=apply_pending_resolution_func,
            )
            return return_value

        _validate_capability_handoff(selected_request, cognition_state)
        observation = await _execute_with_timeout(
            selected_request,
            cognition_state,
            execute_capability_func=execute_capability_func,
            capability_timeout_seconds=capability_timeout_seconds,
        )
        if _is_blocked_pending_observation(observation):
            final_state = await _run_blocked_pending_final_cognition(
                cognition_state,
                selected_request=selected_request,
                observation=observation,
                status_before=status_before,
                call_cognition_subgraph_func=call_cognition_subgraph_func,
                upsert_pending_resume_func=upsert_pending_resume_func,
                apply_pending_resolution_func=apply_pending_resolution_func,
            )
            return_value = await _enforce_final_lifecycle_state(
                final_state,
                status_before=status_before,
                call_cognition_subgraph_func=call_cognition_subgraph_func,
                apply_pending_resolution_func=apply_pending_resolution_func,
            )
            return return_value

        if _is_user_input_blocker_observation(observation):
            final_state = await _run_user_input_blocker_final_cognition(
                cognition_state,
                selected_request=selected_request,
                observation=observation,
                status_before=status_before,
                call_cognition_subgraph_func=call_cognition_subgraph_func,
                apply_pending_resolution_func=apply_pending_resolution_func,
            )
            return_value = await _enforce_final_lifecycle_state(
                final_state,
                status_before=status_before,
                call_cognition_subgraph_func=call_cognition_subgraph_func,
                apply_pending_resolution_func=apply_pending_resolution_func,
            )
            return return_value

        resolver_state = append_observation(resolver_state, observation)
        resolver_state = _bind_required_evidence_dependency(
            resolver_state,
            selected_request=selected_request,
            observation=observation,
            request_ordinal=_resolver_request_ordinal(
                cognition_state,
                selected_request,
            ),
            required=(
                cognition_state.get("goal_resolution")
                == "requires_required_evidence"
            ),
        )
        if "rag_result" in observation:
            cognition_state["rag_result"] = observation["rag_result"]
            await _attach_past_dialog_cognition_from_rag_result(
                cognition_state,
                observation["rag_result"],
            )

        trace = _build_cycle_trace(
            cognition_state,
            resolver_state=resolver_state,
            cycle_index=resolver_state["cycle_index"],
            status_before=status_before,
            selected_capability_kind=selected_request["capability_kind"],
            observation_ids=[observation["observation_id"]],
            terminal_reason="capability observation appended",
        )
        resolver_state = append_cycle_trace(resolver_state, trace)
        cognition_state = _with_resolver_state(cognition_state, resolver_state)
        lifecycle_conflict = _final_lifecycle_conflict(cognition_state)
        if lifecycle_conflict is not None:
            selected_request, reason = lifecycle_conflict
            final_state = await _run_mixed_lifecycle_final_cognition(
                cognition_state,
                selected_request=selected_request,
                status_before=status_before,
                conflict_reason=reason,
                call_cognition_subgraph_func=call_cognition_subgraph_func,
                apply_pending_resolution_func=apply_pending_resolution_func,
            )
            return final_state
        current_state = cognition_state

    final_state = await _run_max_cycle_final_cognition(
        current_state,
        call_cognition_subgraph_func=call_cognition_subgraph_func,
        apply_pending_resolution_func=apply_pending_resolution_func,
    )
    return_value = await _enforce_final_lifecycle_state(
        final_state,
        status_before=_resolver_state(current_state)["status"],
        call_cognition_subgraph_func=call_cognition_subgraph_func,
        apply_pending_resolution_func=apply_pending_resolution_func,
    )
    return return_value


async def _run_mixed_lifecycle_final_cognition(
    state: GlobalPersonaState,
    *,
    selected_request: ResolverCapabilityRequestV1,
    status_before: str,
    conflict_reason: str,
    call_cognition_subgraph_func: CognitionSubgraphFunc,
    apply_pending_resolution_func: PendingResolutionApplyFunc,
) -> GlobalPersonaState:
    """Regenerate once after blocking a same-reference factual/task mix."""

    blocker = _mixed_lifecycle_observation(
        selected_request,
        state,
        stage="regeneration",
    )
    resolver_state = _resolver_state(state)
    resolver_state = _record_mixed_lifecycle_blocker(
        state,
        resolver_state=resolver_state,
        selected_request=selected_request,
        observation=blocker,
    )
    updated_resolver_state = dict(resolver_state)
    updated_resolver_state["status"] = "blocked"
    updated_resolver_state["terminal_reason"] = conflict_reason
    trace = _build_cycle_trace(
        state,
        resolver_state=updated_resolver_state,
        cycle_index=updated_resolver_state["cycle_index"],
        status_before=status_before,
        selected_capability_kind=selected_request["capability_kind"],
        observation_ids=[blocker["observation_id"]],
        terminal_reason=conflict_reason,
    )
    updated_resolver_state = append_cycle_trace(updated_resolver_state, trace)
    cognition_input = _with_resolver_state(state, updated_resolver_state)
    cognition_input["resolver_capability_requests"] = (
        _without_same_continuation_task_requests(
            cognition_input.get("resolver_capability_requests"),
            selected_request["goal_continuation_ref"],
        )
    )
    cognition_output = await call_cognition_subgraph_func(cognition_input)
    cognition_state = _merge_state(cognition_input, cognition_output)
    cognition_state = _sync_goal_progress_from_cognition(cognition_state)
    await _apply_pending_resolution_if_present(
        cognition_state,
        apply_pending_resolution_func=apply_pending_resolution_func,
    )
    final_conflict = _final_lifecycle_conflict(cognition_state)
    final_terminal_reason = "same-reference lifecycle conflict regenerated"
    if final_conflict is not None:
        fallback_request, fallback_reason = final_conflict
        fallback_blocker = _mixed_lifecycle_observation(
            fallback_request,
            cognition_state,
            stage="fail_closed",
        )
        cognition_state = _fail_closed_mixed_lifecycle_state(
            cognition_state,
            selected_request=fallback_request,
            blocker=fallback_blocker,
        )
        final_terminal_reason = (
            "same-reference lifecycle conflict failed closed after regeneration: "
            f"{fallback_reason}"
        )
    else:
        cognition_state["resolver_capability_requests"] = (
            _without_same_continuation_task_requests(
                cognition_state.get("resolver_capability_requests"),
                selected_request["goal_continuation_ref"],
            )
        )

    final_resolver_state = dict(_resolver_state(cognition_state))
    final_resolver_state["held_action_specs"] = list(
        cognition_state.get("action_specs", []),
    )
    final_resolver_state["status"] = "blocked"
    final_resolver_state["terminal_reason"] = final_terminal_reason
    final_trace = _build_cycle_trace(
        cognition_state,
        resolver_state=final_resolver_state,
        cycle_index=final_resolver_state["cycle_index"],
        status_before="blocked",
        selected_capability_kind="",
        observation_ids=[],
        terminal_reason=final_terminal_reason,
    )
    final_resolver_state = append_cycle_trace(final_resolver_state, final_trace)
    return _with_resolver_state(cognition_state, final_resolver_state)


async def _enforce_final_lifecycle_state(
    state: GlobalPersonaState,
    *,
    status_before: str,
    call_cognition_subgraph_func: CognitionSubgraphFunc,
    apply_pending_resolution_func: PendingResolutionApplyFunc,
) -> GlobalPersonaState:
    """Run the same-reference lifecycle gate before returning a final state."""

    lifecycle_conflict = _final_lifecycle_conflict(state)
    if lifecycle_conflict is None:
        return state
    selected_request, reason = lifecycle_conflict
    return await _run_mixed_lifecycle_final_cognition(
        state,
        selected_request=selected_request,
        status_before=status_before,
        conflict_reason=reason,
        call_cognition_subgraph_func=call_cognition_subgraph_func,
        apply_pending_resolution_func=apply_pending_resolution_func,
    )


def _final_lifecycle_conflict(
    state: GlobalPersonaState,
) -> tuple[ResolverCapabilityRequestV1, str] | None:
    """Find one typed same-reference surface/lifecycle conflict.

    This validator checks typed continuation references and surface roles only.
    It does not reinterpret the cognition stage's answerability or wording.
    """

    task_requests = _task_resolution_requests(state)
    resolver_state = _resolver_state(state)
    dependency = resolver_state.get("required_resolver_evidence_dependency")
    dependency_request = _request_from_required_dependency(
        resolver_state,
        dependency,
    )
    tool_result_ref = _tool_result_continuation_ref(state)

    if state.get("goal_resolution") == "answerable_now":
        output_ref = _cognition_output_continuation_ref(state)
        if output_ref is not None:
            matching_request = _request_for_continuation_ref(
                task_requests,
                output_ref,
            )
            if matching_request is not None:
                return (
                    matching_request,
                    "answerable_now cannot retain a same-reference task request",
                )

    for task_request in task_requests:
        if (
            tool_result_ref is not None
            and _same_continuation_ref(
                task_request["goal_continuation_ref"],
                tool_result_ref,
            )
        ):
            return (
                task_request,
                "tool-result continuation cannot create another task request",
            )

    action_specs = state.get("action_specs")
    if not isinstance(action_specs, list):
        return None
    continuation_surfaces = [
        (action_spec, _continuation_surface_ref(action_spec))
        for action_spec in action_specs
        if isinstance(action_spec, Mapping)
        and _continuation_surface_ref(action_spec) is not None
    ]
    for action_spec, action_ref in continuation_surfaces:
        surface_role = action_spec.get("surface_role")
        matching_request = _request_for_continuation_ref(
            task_requests,
            action_ref,
        )
        matching_dependency = (
            isinstance(dependency, Mapping)
            and _same_continuation_ref(
                dependency.get("goal_continuation_ref"),
                action_ref,
            )
        )
        fallback_request = matching_request or (
            dependency_request if matching_dependency else None
        )
        if surface_role == "factual_answer":
            if matching_request is not None:
                return (
                    matching_request,
                    "factual surface cannot coexist with a same-reference task request",
                )
            if matching_dependency and dependency.get("state") != "complete":
                if fallback_request is None:
                    raise ResolverValidationError(
                        "same-reference dependency lacks a request for fail-closed status"
                    )
                return (
                    fallback_request,
                    "factual surface cannot coexist with non-complete task evidence",
                )
            for other_spec, other_ref in continuation_surfaces:
                if other_spec is action_spec or not _same_continuation_ref(
                    other_ref,
                    action_ref,
                ):
                    continue
                if other_spec.get("surface_role") in {
                    "task_acknowledgement",
                    "task_result",
                    "task_status",
                }:
                    if fallback_request is None:
                        raise ResolverValidationError(
                            "same-reference continuation surfaces lack a task request"
                        )
                    return (
                        fallback_request,
                        "factual surface cannot coexist with another task surface",
                    )
        if (
            surface_role == "task_acknowledgement"
            and matching_dependency
            and dependency.get("state") != "pending"
        ):
            if fallback_request is None:
                raise ResolverValidationError(
                    "task acknowledgement lacks a request for fail-closed status"
                )
            return (
                fallback_request,
                "task acknowledgement requires pending task evidence",
            )
        if (
            surface_role == "task_result"
            and matching_dependency
            and dependency.get("state") not in {"complete", "partial"}
        ):
            if fallback_request is None:
                raise ResolverValidationError(
                    "task result lacks a request for fail-closed status"
                )
            return (
                fallback_request,
                "task result requires complete or partial task evidence",
            )
    return None


def _task_resolution_requests(
    state: GlobalPersonaState,
) -> list[ResolverCapabilityRequestV1]:
    """Return validated task-resolution requests from one cognition cycle."""

    raw_requests = state.get("resolver_capability_requests", [])
    if not isinstance(raw_requests, list):
        raise ResolverValidationError("resolver_capability_requests: expected list")
    requests: list[ResolverCapabilityRequestV1] = []
    for raw_request in raw_requests:
        request = validate_resolver_capability_request(raw_request)
        if request["capability_kind"] == "task_resolution_request":
            requests.append(request)
    return requests


def _request_for_continuation_ref(
    requests: list[ResolverCapabilityRequestV1],
    continuation_ref: object,
) -> ResolverCapabilityRequestV1 | None:
    """Return the one task request carrying an exact continuation reference."""

    for request in requests:
        if _same_continuation_ref(
            request["goal_continuation_ref"],
            continuation_ref,
        ):
            return request
    return None


def _request_from_required_dependency(
    resolver_state: ResolverCycleStateV1,
    dependency: object,
) -> ResolverCapabilityRequestV1 | None:
    """Rebuild a status-only request from validated dependency provenance."""

    if not isinstance(dependency, Mapping):
        return None
    observation_id = dependency.get("observation_id")
    continuation_ref = dependency.get("goal_continuation_ref")
    if not isinstance(observation_id, str) or not isinstance(
        continuation_ref,
        Mapping,
    ):
        return None
    for observation in resolver_state["observations"]:
        if observation["observation_id"] != observation_id:
            continue
        return validate_resolver_capability_request({
            "schema_version": "resolver_capability_request.v1",
            "capability_kind": "task_resolution_request",
            "objective": observation["request_objective"],
            "reason": observation["request_reason"],
            "priority": "now",
            "goal_continuation_ref": dict(continuation_ref),
        })
    return None


def _continuation_surface_ref(
    action_spec: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Return a continuation surface reference without parsing wording."""

    surface_role = action_spec.get("surface_role")
    if surface_role not in {
        "factual_answer",
        "task_acknowledgement",
        "task_result",
        "task_status",
    }:
        return None
    continuation_ref = action_spec.get("goal_continuation_ref")
    if not isinstance(continuation_ref, Mapping):
        raise ResolverValidationError(
            "continuation surface lacks goal_continuation_ref"
        )
    return continuation_ref


def _same_continuation_ref(first: object, second: object) -> bool:
    """Compare exact validated continuation-reference payloads."""

    return isinstance(first, Mapping) and isinstance(second, Mapping) and (
        dict(first) == dict(second)
    )


def _cognition_output_continuation_ref(
    state: GlobalPersonaState,
) -> Mapping[str, object] | None:
    """Read the V2 output continuation reference when it is present."""

    output = state.get("cognition_core_output")
    if not isinstance(output, Mapping):
        return None
    continuation_ref = output.get("goal_continuation_ref")
    if not isinstance(continuation_ref, Mapping):
        return None
    return continuation_ref


def _tool_result_continuation_ref(
    state: GlobalPersonaState,
) -> Mapping[str, object] | None:
    """Read tool-result lineage without inferring a replacement reference."""

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping) or episode.get("trigger_source") != "tool_result":
        return None
    origin = episode.get("origin_metadata")
    if not isinstance(origin, Mapping):
        return None
    continuation_ref = origin.get("goal_continuation_ref")
    if not isinstance(continuation_ref, Mapping):
        raise ResolverValidationError(
            "tool-result episode requires goal_continuation_ref"
        )
    return continuation_ref


def _without_same_continuation_task_requests(
    raw_requests: object,
    continuation_ref: object,
) -> list[object]:
    """Remove only the blocked task request while retaining distinct goals."""

    if not isinstance(raw_requests, list):
        raise ResolverValidationError("resolver_capability_requests: expected list")
    retained_requests: list[object] = []
    for raw_request in raw_requests:
        request = validate_resolver_capability_request(raw_request)
        if (
            request["capability_kind"] == "task_resolution_request"
            and _same_continuation_ref(
                request["goal_continuation_ref"],
                continuation_ref,
            )
        ):
            continue
        retained_requests.append(raw_request)
    return retained_requests


def _fail_closed_mixed_lifecycle_state(
    state: GlobalPersonaState,
    *,
    selected_request: ResolverCapabilityRequestV1,
    blocker: ResolverObservationV1,
) -> GlobalPersonaState:
    """Replace only the invalid same-reference surface with typed status."""

    continuation_ref = _task_continuation_ref(selected_request)
    resolver_state = _record_mixed_lifecycle_blocker(
        state,
        resolver_state=_resolver_state(state),
        selected_request=selected_request,
        observation=blocker,
    )
    state = _with_resolver_state(state, resolver_state)
    state["resolver_capability_requests"] = (
        _without_same_continuation_task_requests(
            state.get("resolver_capability_requests"),
            continuation_ref,
        )
    )
    raw_action_specs = state.get("action_specs", [])
    if not isinstance(raw_action_specs, list):
        raise ResolverValidationError("action_specs: expected list")
    retained_action_specs = [
        action_spec
        for action_spec in raw_action_specs
        if not (
            isinstance(action_spec, Mapping)
            and _same_continuation_ref(
                action_spec.get("goal_continuation_ref"),
                continuation_ref,
            )
        )
    ]
    if _should_surface_terminal_blocker(state):
        retained_action_specs.append(
            _terminal_blocker_speak_action_spec(
                state,
                selected_request,
                blocker,
            )
        )
    state["action_specs"] = retained_action_specs
    return state


def _record_mixed_lifecycle_blocker(
    state: GlobalPersonaState,
    *,
    resolver_state: ResolverCycleStateV1,
    selected_request: ResolverCapabilityRequestV1,
    observation: ResolverObservationV1,
) -> ResolverCycleStateV1:
    """Persist a lifecycle blocker while retaining existing task provenance."""

    updated_resolver_state = append_observation(resolver_state, observation)
    continuation_ref = _task_continuation_ref(selected_request)
    matching_request = _request_for_continuation_ref(
        _task_resolution_requests(state),
        continuation_ref,
    )
    if matching_request is None:
        return _mark_existing_dependency_blocked(
            updated_resolver_state,
            observation,
        )
    return _bind_required_evidence_dependency(
        updated_resolver_state,
        selected_request=matching_request,
        observation=observation,
        request_ordinal=_resolver_request_ordinal(state, matching_request),
        required=True,
    )


def _mixed_lifecycle_observation(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
    *,
    stage: str,
) -> ResolverObservationV1:
    """Record a typed task blocker without executing mixed lifecycle work."""

    resolver_state = _resolver_state(state)
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": (
            "resolver_obs_lifecycle_conflict_"
            f"{resolver_state['cycle_index']}_{stage}"
        ),
        "capability_kind": "task_resolution_request",
        "request_objective": request["objective"],
        "request_reason": request["reason"],
        "status": "failed",
        "prompt_safe_summary": (
            "The task continuation was blocked before execution because its "
            "factual surface and pending lifecycle state conflicted."
        ),
        "evidence_refs": [],
        "task_resolution_evidence_state": {
            "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
            "state": "blocked",
            "remaining_needs": [request["objective"]],
        },
        "goal_continuation_ref": _task_continuation_ref(request),
        "created_at_utc": _created_at_utc(state),
    }
    return validate_resolver_observation(observation)


async def _attach_past_dialog_cognition_from_rag_result(
    cognition_state: GlobalPersonaState,
    rag_result: object,
) -> None:
    """Attach private residual context from conversation-evidence source refs."""

    if cognition_state.get("past_dialog_cognition_context"):
        return
    if not isinstance(rag_result, dict):
        return

    character_profile = cognition_state["character_profile"]
    character_global_user_id = str(character_profile.get("global_user_id") or "")
    lookup_result = await build_past_dialog_cognition_context_from_rag_result(
        rag_result,
        character_global_user_id=character_global_user_id,
    )
    context = lookup_result["past_dialog_cognition_context"]
    if context:
        cognition_state["past_dialog_cognition_context"] = context


async def _finalize_without_capability(
    cognition_state: GlobalPersonaState,
    *,
    resolver_state: ResolverCycleStateV1,
    status_before: str,
    apply_pending_resolution_func: PendingResolutionApplyFunc,
    terminal_reason: str = "no resolver capability request",
) -> GlobalPersonaState:
    """Attach terminal trace/state when cognition does not need a capability."""

    await _apply_pending_resolution_if_present(
        cognition_state,
        apply_pending_resolution_func=apply_pending_resolution_func,
    )
    resolver_state = _resolver_state(cognition_state)
    action_specs = list(cognition_state.get("action_specs", []))
    updated_resolver_state = dict(resolver_state)
    updated_resolver_state["status"] = "terminal"
    updated_resolver_state["held_action_specs"] = action_specs
    updated_resolver_state["terminal_reason"] = terminal_reason
    trace = _build_cycle_trace(
        cognition_state,
        resolver_state=updated_resolver_state,
        cycle_index=updated_resolver_state["cycle_index"],
        status_before=status_before,
        selected_capability_kind="",
        observation_ids=[],
        terminal_reason=terminal_reason,
    )
    updated_resolver_state = append_cycle_trace(updated_resolver_state, trace)
    return_value = _with_resolver_state(cognition_state, updated_resolver_state)
    return return_value


async def _run_max_cycle_final_cognition(
    state: GlobalPersonaState,
    *,
    call_cognition_subgraph_func: CognitionSubgraphFunc,
    apply_pending_resolution_func: PendingResolutionApplyFunc,
) -> GlobalPersonaState:
    """Return one more cognition cycle with a structural max-cycle blocker."""

    resolver_state = _resolver_state(state)
    blocker = _max_cycle_observation(state, resolver_state)
    resolver_state = append_observation(resolver_state, blocker)
    resolver_state = _mark_existing_dependency_blocked(
        resolver_state,
        blocker,
    )
    updated_resolver_state = dict(resolver_state)
    updated_resolver_state["status"] = "max_cycles"
    updated_resolver_state["terminal_reason"] = "maximum resolver cycles reached"
    cognition_input = _with_resolver_state(state, updated_resolver_state)
    cognition_output = await call_cognition_subgraph_func(cognition_input)
    cognition_state = _merge_state(cognition_input, cognition_output)
    cognition_state = _sync_goal_progress_from_cognition(cognition_state)
    await _apply_pending_resolution_if_present(
        cognition_state,
        apply_pending_resolution_func=apply_pending_resolution_func,
    )
    final_resolver_state = _resolver_state(cognition_state)
    final_resolver_state = dict(final_resolver_state)
    final_terminal_reason = "maximum resolver cycles reached"
    answerable_now = _suppress_optional_resolver_request(cognition_state)
    final_selected_request = (
        None
        if answerable_now
        else _select_immediate_request(cognition_state)
    )
    if answerable_now:
        final_terminal_reason = ANSWERABLE_NOW_TERMINAL_REASON
    if final_selected_request is not None:
        cognition_state["resolver_capability_requests"] = []
        if cognition_state.get("action_specs"):
            final_terminal_reason = "maximum resolver cycles reached"
        elif _should_surface_terminal_blocker(cognition_state):
            logger.warning(
                "Resolver converted max-cycle terminal capability request to "
                "visible blocker surface"
            )
            cognition_state["action_specs"] = [
                _terminal_blocker_speak_action_spec(
                    cognition_state,
                    final_selected_request,
                    blocker,
                ),
            ]
            final_terminal_reason = (
                "maximum resolver cycles converted to terminal surface"
            )
        else:
            logger.warning(
                "Resolver kept max-cycle terminal capability request private "
                "for non-user source"
            )
            final_terminal_reason = (
                "maximum resolver cycles kept private for non-user source"
            )
    final_resolver_state["held_action_specs"] = list(
        cognition_state.get("action_specs", []),
    )
    final_resolver_state["status"] = "max_cycles"
    final_resolver_state["terminal_reason"] = final_terminal_reason
    trace = _build_cycle_trace(
        cognition_state,
        resolver_state=final_resolver_state,
        cycle_index=final_resolver_state["cycle_index"],
        status_before="max_cycles",
        selected_capability_kind="",
        observation_ids=[],
        terminal_reason=final_terminal_reason,
    )
    final_resolver_state = append_cycle_trace(final_resolver_state, trace)
    return_value = _with_resolver_state(cognition_state, final_resolver_state)
    return return_value


async def _run_duplicate_request_final_cognition(
    state: GlobalPersonaState,
    *,
    selected_request: ResolverCapabilityRequestV1,
    resolver_state: ResolverCycleStateV1,
    status_before: str,
    call_cognition_subgraph_func: CognitionSubgraphFunc,
    apply_pending_resolution_func: PendingResolutionApplyFunc,
) -> GlobalPersonaState:
    """Return one final cognition cycle after blocking repeated capability work."""

    blocker = _duplicate_request_observation(selected_request, state)
    resolver_state = append_observation(resolver_state, blocker)
    resolver_state = _mark_existing_dependency_blocked(
        resolver_state,
        blocker,
    )
    updated_resolver_state = dict(resolver_state)
    updated_resolver_state["status"] = "blocked"
    updated_resolver_state["terminal_reason"] = (
        "duplicate resolver capability request blocked"
    )
    trace = _build_cycle_trace(
        state,
        resolver_state=updated_resolver_state,
        cycle_index=updated_resolver_state["cycle_index"],
        status_before=status_before,
        selected_capability_kind=selected_request["capability_kind"],
        observation_ids=[blocker["observation_id"]],
        terminal_reason="duplicate resolver capability request blocked",
    )
    updated_resolver_state = append_cycle_trace(updated_resolver_state, trace)
    cognition_input = _with_resolver_state(state, updated_resolver_state)
    cognition_output = await call_cognition_subgraph_func(cognition_input)
    cognition_state = _merge_state(cognition_input, cognition_output)
    cognition_state = _sync_goal_progress_from_cognition(cognition_state)
    await _apply_pending_resolution_if_present(
        cognition_state,
        apply_pending_resolution_func=apply_pending_resolution_func,
    )
    final_resolver_state = _resolver_state(cognition_state)
    final_resolver_state = dict(final_resolver_state)
    final_terminal_reason = (
        "duplicate resolver capability request final cognition completed"
    )
    answerable_now = _suppress_optional_resolver_request(cognition_state)
    final_repeated_request = (
        None
        if answerable_now
        else _select_immediate_request(cognition_state)
    )
    if answerable_now:
        final_terminal_reason = ANSWERABLE_NOW_TERMINAL_REASON
    if final_repeated_request is not None:
        cognition_state["resolver_capability_requests"] = []
        if cognition_state.get("action_specs"):
            final_terminal_reason = (
                "duplicate resolver capability request final cognition "
                "completed"
            )
        elif _should_surface_terminal_blocker(cognition_state):
            logger.warning(
                "Resolver converted terminal capability request after "
                "duplicate blocking to visible blocker surface"
            )
            cognition_state["action_specs"] = [
                _terminal_blocker_speak_action_spec(
                    cognition_state,
                    final_repeated_request,
                    blocker,
                ),
            ]
            final_terminal_reason = (
                "duplicate resolver capability request converted to "
                "terminal surface"
            )
        else:
            logger.warning(
                "Resolver kept terminal capability request private after "
                "duplicate blocking for non-user source"
            )
            final_terminal_reason = (
                "duplicate resolver capability request kept private for "
                "non-user source"
            )
    final_resolver_state["held_action_specs"] = list(
        cognition_state.get("action_specs", []),
    )
    final_resolver_state["status"] = "blocked"
    final_resolver_state["terminal_reason"] = final_terminal_reason
    final_trace = _build_cycle_trace(
        cognition_state,
        resolver_state=final_resolver_state,
        cycle_index=final_resolver_state["cycle_index"],
        status_before="blocked",
        selected_capability_kind="",
        observation_ids=[],
        terminal_reason=final_terminal_reason,
    )
    final_resolver_state = append_cycle_trace(final_resolver_state, final_trace)
    return_value = _with_resolver_state(cognition_state, final_resolver_state)
    return return_value


async def _run_blocked_pending_final_cognition(
    state: GlobalPersonaState,
    *,
    selected_request: ResolverCapabilityRequestV1,
    observation: ResolverObservationV1,
    status_before: str,
    call_cognition_subgraph_func: CognitionSubgraphFunc,
    upsert_pending_resume_func: PendingResumeUpsertFunc,
    apply_pending_resolution_func: PendingResolutionApplyFunc,
) -> GlobalPersonaState:
    """Persist one pending blocker and run exactly one final cognition cycle."""

    pending_resume = await upsert_pending_resume_func(state, observation)
    observation_with_pending = dict(observation)
    observation_with_pending["pending_resume_id"] = pending_resume["resume_id"]
    normalized_observation = validate_resolver_observation(
        observation_with_pending,
    )
    resolver_state = _resolver_state(state)
    resolver_state = append_observation(resolver_state, normalized_observation)
    resolver_state = _bind_required_evidence_dependency(
        resolver_state,
        selected_request=selected_request,
        observation=normalized_observation,
        request_ordinal=_resolver_request_ordinal(state, selected_request),
        required=(
            state.get("goal_resolution")
            == "requires_required_evidence"
        ),
    )
    updated_resolver_state = dict(resolver_state)
    updated_resolver_state["status"] = pending_resume["status"]
    updated_resolver_state["pending_resume"] = pending_resume
    updated_resolver_state["terminal_reason"] = (
        f"{pending_resume['capability_kind']} pending resume created"
    )
    trace = _build_cycle_trace(
        state,
        resolver_state=updated_resolver_state,
        cycle_index=updated_resolver_state["cycle_index"],
        status_before=status_before,
        selected_capability_kind=selected_request["capability_kind"],
        observation_ids=[normalized_observation["observation_id"]],
        terminal_reason="blocked pending resume created",
    )
    updated_resolver_state = append_cycle_trace(updated_resolver_state, trace)
    cognition_input = _with_resolver_state(state, updated_resolver_state)
    cognition_input["pending_resolver_resume"] = pending_resume

    cognition_output = await call_cognition_subgraph_func(cognition_input)
    cognition_state = _merge_state(cognition_input, cognition_output)
    cognition_state = _sync_goal_progress_from_cognition(cognition_state)
    await _apply_pending_resolution_if_present(
        cognition_state,
        apply_pending_resolution_func=apply_pending_resolution_func,
    )
    final_resolver_state = _resolver_state(cognition_state)
    final_terminal_reason = "pending resume final cognition completed"
    repeated_request = _select_immediate_request(cognition_state)
    if _is_repeated_blocked_request(repeated_request, selected_request):
        logger.warning(
            "Resolver blocked capability repeated after pending resume creation"
        )
        cognition_state["resolver_capability_requests"] = []
        cognition_state["action_specs"] = []
        final_terminal_reason = "blocked capability repeated after pending resume"
    if not cognition_state.get("action_specs"):
        cognition_state["resolver_capability_requests"] = []
        cognition_state["action_specs"] = [
            _pending_resume_speak_action_spec(
                cognition_state,
                pending_resume,
                normalized_observation,
            ),
        ]
        if final_terminal_reason == "blocked capability repeated after pending resume":
            final_terminal_reason = (
                "pending resume fallback surface after repeated capability"
            )
        else:
            final_terminal_reason = "pending resume fallback surface completed"
    final_resolver_state = dict(final_resolver_state)
    final_resolver_state["held_action_specs"] = list(
        cognition_state.get("action_specs", []),
    )
    final_pending_resume = _pending_resume_from_state(cognition_state)
    if final_pending_resume is None:
        final_pending_resume = pending_resume
    final_resolver_state["pending_resume"] = final_pending_resume
    if _is_open_pending_resume(final_pending_resume):
        final_resolver_state["status"] = final_pending_resume["status"]
    else:
        final_resolver_state["status"] = "terminal"
    final_resolver_state["terminal_reason"] = final_terminal_reason
    final_trace = _build_cycle_trace(
        cognition_state,
        resolver_state=final_resolver_state,
        cycle_index=final_resolver_state["cycle_index"],
        status_before=pending_resume["status"],
        selected_capability_kind="",
        observation_ids=[],
        terminal_reason=final_terminal_reason,
    )
    final_resolver_state = append_cycle_trace(final_resolver_state, final_trace)
    return_value = _with_resolver_state(cognition_state, final_resolver_state)
    return return_value


async def _run_user_input_blocker_final_cognition(
    state: GlobalPersonaState,
    *,
    selected_request: ResolverCapabilityRequestV1,
    observation: ResolverObservationV1,
    status_before: str,
    call_cognition_subgraph_func: CognitionSubgraphFunc,
    apply_pending_resolution_func: PendingResolutionApplyFunc,
) -> GlobalPersonaState:
    """Run one final cognition pass after a typed user-input blocker."""

    resolver_state = _resolver_state(state)
    resolver_state = append_observation(resolver_state, observation)
    resolver_state = _bind_required_evidence_dependency(
        resolver_state,
        selected_request=selected_request,
        observation=observation,
        request_ordinal=_resolver_request_ordinal(state, selected_request),
        required=(
            state.get("goal_resolution")
            == "requires_required_evidence"
        ),
    )
    updated_resolver_state = dict(resolver_state)
    updated_resolver_state["status"] = "blocked"
    updated_resolver_state["terminal_reason"] = (
        "blocked user-input observation appended"
    )
    trace = _build_cycle_trace(
        state,
        resolver_state=updated_resolver_state,
        cycle_index=updated_resolver_state["cycle_index"],
        status_before=status_before,
        selected_capability_kind=selected_request["capability_kind"],
        observation_ids=[observation["observation_id"]],
        terminal_reason="blocked user-input observation appended",
    )
    updated_resolver_state = append_cycle_trace(updated_resolver_state, trace)
    cognition_input = _with_resolver_state(state, updated_resolver_state)
    if "rag_result" in observation:
        cognition_input["rag_result"] = observation["rag_result"]
        await _attach_past_dialog_cognition_from_rag_result(
            cognition_input,
            observation["rag_result"],
        )

    cognition_output = await call_cognition_subgraph_func(cognition_input)
    cognition_state = _merge_state(cognition_input, cognition_output)
    cognition_state = _sync_goal_progress_from_cognition(cognition_state)
    await _apply_pending_resolution_if_present(
        cognition_state,
        apply_pending_resolution_func=apply_pending_resolution_func,
    )
    final_resolver_state = dict(_resolver_state(cognition_state))
    final_terminal_reason = "blocked user-input final cognition completed"
    answerable_now = _suppress_optional_resolver_request(cognition_state)
    final_request = (
        None
        if answerable_now
        else _select_immediate_request(cognition_state)
    )
    if answerable_now:
        final_terminal_reason = ANSWERABLE_NOW_TERMINAL_REASON
    else:
        cognition_state["resolver_capability_requests"] = []
        if final_request is not None and _is_repeated_user_input_blocked_request(
            final_request,
            observation,
        ):
            if _should_surface_terminal_blocker(cognition_state):
                logger.warning(
                    "Resolver converted repeated user-input blocker request "
                    "to a visible clarification surface"
                )
                cognition_state["action_specs"] = [
                    _user_input_blocker_speak_action_spec(
                        cognition_state,
                        selected_request,
                        observation,
                    ),
                ]
                final_terminal_reason = USER_INPUT_BLOCKER_TERMINAL_REASON
            else:
                logger.warning(
                    "Resolver kept repeated user-input blocker request private "
                    "for non-user source"
                )
                final_terminal_reason = (
                    USER_INPUT_BLOCKER_PRIVATE_TERMINAL_REASON
                )
        if (
            not cognition_state.get("action_specs")
            and _should_surface_terminal_blocker(cognition_state)
        ):
            logger.warning(
                "Resolver converted silent user-input blocker cognition to "
                "a visible clarification surface"
            )
            cognition_state["action_specs"] = [
                _user_input_blocker_speak_action_spec(
                    cognition_state,
                    selected_request,
                    observation,
                ),
            ]
            final_terminal_reason = USER_INPUT_BLOCKER_TERMINAL_REASON

    final_resolver_state["held_action_specs"] = list(
        cognition_state.get("action_specs", []),
    )
    final_resolver_state["status"] = "blocked"
    final_resolver_state["terminal_reason"] = final_terminal_reason
    final_trace = _build_cycle_trace(
        cognition_state,
        resolver_state=final_resolver_state,
        cycle_index=final_resolver_state["cycle_index"],
        status_before="blocked",
        selected_capability_kind="",
        observation_ids=[],
        terminal_reason=final_terminal_reason,
    )
    final_resolver_state = append_cycle_trace(final_resolver_state, final_trace)
    return_value = _with_resolver_state(cognition_state, final_resolver_state)
    return return_value


def _pending_resume_speak_action_spec(
    state: GlobalPersonaState,
    pending_resume: ResolverPendingResumeV1,
    observation: ResolverObservationV1,
) -> dict[str, Any]:
    """Build the visible text action for a persisted pending row."""

    capability_kind = pending_resume["capability_kind"]
    decision = "ask_clarification"
    detail = pending_resume["prompt_safe_question"]
    if capability_kind == "approval_preparation":
        decision = "request_approval"
        detail = pending_resume["prompt_safe_approval_summary"]
    if not detail:
        detail = observation["prompt_safe_summary"]
    continuation_ref = _pending_resume_continuation_ref(state, pending_resume)

    action_spec = {
        "schema_version": ACTION_SPEC_VERSION,
        "kind": SPEAK_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": ACTION_SOURCE_REF_VERSION,
                "ref_kind": "system_event",
                "ref_id": observation["observation_id"],
                "owner": "cognition_resolver",
                "relationship": "basis",
                "evidence_refs": [],
            },
        ],
        "target": {
            "schema_version": ACTION_TARGET_VERSION,
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {
                "decision": decision,
                "detail": detail,
            },
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": {
            "schema_version": ACTION_CONTINUATION_VERSION,
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "surface_role": "task_status",
        "goal_continuation_ref": continuation_ref,
        "cognition_provenance": _resolver_speak_cognition_provenance(state),
        "reason": (
            "Resolver created a pending row and must surface its prompt-safe "
            "question or approval summary as a continuation status."
        ),
    }
    validated_spec = validate_action_spec(action_spec)
    return_value = dict(validated_spec)
    return return_value


def _resolver_speak_cognition_provenance(
    state: GlobalPersonaState,
) -> dict[str, Any]:
    """Build current-user provenance for a visible resolver fallback.

    Args:
        state: Persona state containing the resolved current global user id.

    Returns:
        The caller-owned target role and an empty evidence-handle list.

    Raises:
        ResolverValidationError: If the current global user id is missing or
            blank.
    """

    if "global_user_id" not in state:
        raise ResolverValidationError(
            "global_user_id: expected non-empty string",
        )
    global_user_id = state["global_user_id"]
    if not isinstance(global_user_id, str) or not global_user_id.strip():
        raise ResolverValidationError(
            "global_user_id: expected non-empty string",
        )
    provenance = {
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": global_user_id,
        }],
        "evidence_handles": [],
    }
    return provenance


def _pending_resume_continuation_ref(
    state: GlobalPersonaState,
    pending_resume: ResolverPendingResumeV1,
) -> GoalContinuationRefV1:
    """Derive the validated continuation identity from pending-resume lineage.

    A pending clarification or approval is a continuation surface of the
    original goal, so its reference is constructed deterministically from the
    source episode, source message, capability, and user scope. The model
    never authors the identifier.
    """

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        raise ResolverValidationError("cognitive_episode: expected mapping")
    source_episode_id = episode.get("episode_id")
    if not isinstance(source_episode_id, str) or not source_episode_id.strip():
        raise ResolverValidationError(
            "cognitive_episode.episode_id: expected non-empty string"
        )
    goal_ref = {
        "scope": "user",
        "kind": "goal",
        "entity_id": pending_resume["global_user_id"],
    }
    try:
        continuation_ref = build_goal_continuation_ref(
            source_episode_id=source_episode_id,
            source_message_id=pending_resume["source_message_id"],
            branch_id=(
                f"resolver_pending_resume:"
                f"{pending_resume['capability_kind']}"
            ),
            goal_ref=goal_ref,
        )
    except CognitiveEpisodeValidationError as exc:
        raise ResolverValidationError(
            f"pending resume goal_continuation_ref is invalid: {exc}"
        ) from exc
    return continuation_ref


def _user_input_blocker_speak_action_spec(
    state: GlobalPersonaState,
    request: ResolverCapabilityRequestV1,
    observation: ResolverObservationV1,
) -> dict[str, Any]:
    """Build a prompt-safe clarification action from a typed blocker."""

    surface_role, continuation_ref = _terminal_surface_metadata(request)
    action_spec = {
        "schema_version": ACTION_SPEC_VERSION,
        "kind": SPEAK_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": ACTION_SOURCE_REF_VERSION,
                "ref_kind": "system_event",
                "ref_id": observation["observation_id"],
                "owner": "cognition_resolver",
                "relationship": "basis",
                "evidence_refs": [],
            },
        ],
        "target": {
            "schema_version": ACTION_TARGET_VERSION,
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {
                "decision": "ask_clarification",
                "detail": observation["prompt_safe_summary"],
            },
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": {
            "schema_version": ACTION_CONTINUATION_VERSION,
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "surface_role": surface_role,
        "goal_continuation_ref": continuation_ref,
        "cognition_provenance": _resolver_speak_cognition_provenance(state),
        "reason": (
            "Resolver requires user input before the blocked capability can "
            "act and must surface a prompt-safe clarification."
        ),
    }
    validated_spec = validate_action_spec(action_spec)
    return_value = dict(validated_spec)
    return return_value


async def _apply_pending_resolution_if_present(
    state: GlobalPersonaState,
    *,
    apply_pending_resolution_func: PendingResolutionApplyFunc,
) -> None:
    """Apply L2d pending decision when present."""

    resolution = state.get("resolver_pending_resolution")
    if resolution is None:
        return
    validated_resolution = validate_resolver_pending_resolution(resolution)
    if (
        _resolution_targets_current_message_pending(state, validated_resolution)
        and not _same_message_resolution_has_terminal_action(state)
    ):
        state.pop("resolver_pending_resolution", None)
        return
    updated_row = await apply_pending_resolution_func(state, validated_resolution)
    _reflect_pending_resolution_in_state(
        state,
        validated_resolution,
        updated_row,
    )


def _reflect_pending_resolution_in_state(
    state: GlobalPersonaState,
    resolution: ResolverPendingResolutionV1,
    updated_row: object,
) -> None:
    """Reflect applied pending status in the current prompt-safe state."""

    pending_resume = _pending_resume_from_updated_row(updated_row)
    if pending_resume is None:
        pending_resume = _pending_resume_from_state(state)
    if pending_resume is None:
        return

    updated_pending = dict(pending_resume)
    if resolution["decision"] == "continue_waiting":
        updated_status = pending_resume["status"]
    elif resolution["decision"] == "superseded":
        updated_status = "superseded"
    else:
        updated_status = "closed"
    updated_pending["status"] = updated_status
    resolver_state = _resolver_state(state)
    updated_resolver_state = dict(resolver_state)
    updated_resolver_state["pending_resume"] = updated_pending
    state["pending_resolver_resume"] = updated_pending
    state["resolver_state"] = validate_resolver_state(updated_resolver_state)
    state["resolver_context"] = project_resolver_context(state["resolver_state"])


def _resolution_targets_current_message_pending(
    state: GlobalPersonaState,
    resolution: ResolverPendingResolutionV1,
) -> bool:
    """Return whether a resolution tries to answer a same-message pending row."""

    pending_resume = _pending_resume_from_state(state)
    if pending_resume is None:
        return_value = False
        return return_value
    same_resume = pending_resume.get("resume_id") == resolution["resume_id"]
    same_message = (
        pending_resume.get("source_message_id")
        == state.get("platform_message_id")
    )
    return_value = bool(same_resume and same_message)
    return return_value


def _same_message_resolution_has_terminal_action(state: GlobalPersonaState) -> bool:
    """Return whether final cognition already selected a terminal action."""

    action_specs = state.get("action_specs")
    return_value = isinstance(action_specs, list) and bool(action_specs)
    return return_value


def _is_open_pending_resume(pending_resume: dict[str, Any]) -> bool:
    """Return whether a pending resume should keep resolver waiting."""

    return_value = pending_resume.get("status") in {
        "waiting_for_user",
        "waiting_for_approval",
    }
    return return_value


def _pending_resume_from_updated_row(updated_row: object) -> dict | None:
    """Return pending resume payload from a just-updated ledger row."""

    if not isinstance(updated_row, dict):
        return_value = None
        return return_value
    pending_resume = updated_row.get("resolver_pending_resume")
    if isinstance(pending_resume, dict):
        return_value = pending_resume
        return return_value
    execution_result = updated_row.get("execution_result")
    if not isinstance(execution_result, dict):
        return_value = None
        return return_value
    pending_resume = execution_result.get("resolver_pending_resume")
    if isinstance(pending_resume, dict):
        return_value = pending_resume
        return return_value
    pending_resume = execution_result.get("pending_resume")
    if isinstance(pending_resume, dict):
        return_value = pending_resume
        return return_value
    return_value = None
    return return_value


def _pending_resume_from_state(state: GlobalPersonaState) -> dict | None:
    """Return pending resume payload already attached to the resolver state."""

    pending_resume = state.get("pending_resolver_resume")
    if isinstance(pending_resume, dict):
        return_value = pending_resume
        return return_value
    resolver_state = state.get("resolver_state")
    if not isinstance(resolver_state, dict):
        return_value = None
        return return_value
    pending_resume = resolver_state.get("pending_resume")
    if isinstance(pending_resume, dict):
        return_value = pending_resume
        return return_value
    return_value = None
    return return_value


def _sync_goal_progress_from_cognition(
    state: GlobalPersonaState,
) -> GlobalPersonaState:
    """Persist L2d's validated goal checklist into resolver state."""

    raw_goal_progress = state.get("resolver_goal_progress")
    if raw_goal_progress is None:
        return_value = state
        return return_value
    try:
        goal_progress = validate_resolver_goal_progress(raw_goal_progress)
    except ResolverValidationError as exc:
        logger.warning(f"Resolver dropped invalid goal progress: {exc}")
        updated = dict(state)
        updated.pop("resolver_goal_progress", None)
        return_value = updated
        return return_value

    resolver_state = update_goal_progress(_resolver_state(state), goal_progress)
    updated = _with_resolver_state(state, resolver_state)
    updated["resolver_goal_progress"] = goal_progress
    return_value = updated
    return return_value


def _suppress_optional_resolver_request(
    state: GlobalPersonaState,
) -> bool:
    """Apply the validated answerability decision to the loop boundary."""

    if state.get("goal_resolution") != "answerable_now":
        return_value = False
        return return_value
    state["resolver_capability_requests"] = []
    return_value = True
    return return_value


async def _execute_with_timeout(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
    *,
    execute_capability_func: CapabilityExecutorFunc,
    capability_timeout_seconds: float,
) -> ResolverObservationV1:
    """Execute one capability with a structural timeout observation."""

    try:
        observation = await asyncio.wait_for(
            execute_capability_func(request, state),
            timeout=capability_timeout_seconds,
        )
    except TimeoutError:
        observation = _timeout_observation(request, state)
    return_value = validate_resolver_observation(observation)
    return return_value


def _validate_capability_handoff(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> None:
    """Validate cognition-owned carriers before invoking a capability."""

    if request["capability_kind"] != "task_resolution_request":
        return
    scene_context = state.get("cognition_scene_context")
    if not isinstance(scene_context, Mapping):
        raise ResolverValidationError(
            "cognition_scene_context: expected canonical mapping"
        )
    try:
        _validate_scene_context(scene_context)
    except CognitionContractError as exc:
        raise ResolverValidationError(
            "cognition_scene_context: invalid canonical mapping: "
            f"{exc}"
        ) from exc


def _select_immediate_request(
    state: GlobalPersonaState,
) -> ResolverCapabilityRequestV1 | None:
    """Return the first resolver request this cycle must process.

    Priority ``now`` requests execute immediately as before.  A priority
    ``background`` task-resolution request also executes in the same cycle
    through the direct durable handoff branch without an inline specialist.
    """

    requests = state.get("resolver_capability_requests", [])
    for request in requests:
        validated_request = validate_resolver_capability_request(request)
        if validated_request["priority"] == "now":
            return_value = validated_request
            return return_value
        if (
            validated_request["priority"] == "background"
            and validated_request["capability_kind"]
            == "task_resolution_request"
        ):
            return_value = validated_request
            return return_value
    return_value = None
    return return_value


def _is_blocked_pending_observation(observation: ResolverObservationV1) -> bool:
    """Return whether an observation should become a pending resume row."""

    return_value = (
        observation["status"] == "blocked"
        and observation["capability_kind"] in BLOCKED_PENDING_CAPABILITIES
    )
    return return_value


def _is_user_input_blocker_observation(
    observation: ResolverObservationV1,
) -> bool:
    """Return whether an observation carries a typed user-input blocker."""

    return_value = (
        observation["status"] == "blocked"
        and observation.get("blocker_kind") == "requires_user_input"
    )
    return return_value


def _is_repeated_user_input_blocked_request(
    request: ResolverCapabilityRequestV1 | None,
    observation: ResolverObservationV1,
) -> bool:
    """Return whether final cognition repeats the blocked capability kind."""

    if request is None:
        return_value = False
        return return_value
    return_value = request["capability_kind"] == observation["capability_kind"]
    return return_value


def _is_repeated_blocked_request(
    request: ResolverCapabilityRequestV1 | None,
    previous_request: ResolverCapabilityRequestV1,
) -> bool:
    """Return whether final cognition repeated the same blocked capability."""

    if request is None:
        return_value = False
        return return_value
    return_value = (
        request["capability_kind"] == previous_request["capability_kind"]
        and request["capability_kind"] in BLOCKED_PENDING_CAPABILITIES
    )
    return return_value


def _is_repeated_capability_request(
    request: ResolverCapabilityRequestV1,
    resolver_state: ResolverCycleStateV1,
) -> bool:
    """Return whether a capability request repeats already attempted work."""

    for observation in resolver_state["observations"]:
        same_capability = observation["capability_kind"] == request["capability_kind"]
        same_objective = observation["request_objective"] == request["objective"]
        same_failed_capability = (
            same_capability
            and observation["status"] == "failed"
        )
        if same_capability and (same_objective or same_failed_capability):
            return_value = True
            return return_value
    return_value = False
    return return_value


def _resolver_request_ordinal(
    state: GlobalPersonaState,
    selected_request: ResolverCapabilityRequestV1,
) -> int:
    """Return the stable one-based request position selected in this cycle."""

    requests = state.get("resolver_capability_requests")
    if not isinstance(requests, list):
        raise ResolverValidationError(
            "resolver_capability_requests: expected list"
        )
    for ordinal, raw_request in enumerate(requests, start=1):
        validated_request = validate_resolver_capability_request(raw_request)
        if validated_request == selected_request:
            return_value = ordinal
            return return_value
    raise ResolverValidationError(
        "selected resolver request was not present in the cognition output"
    )


def _terminal_surface_metadata(
    request: ResolverCapabilityRequestV1,
) -> tuple[SurfaceRoleV1, GoalContinuationRefV1 | None]:
    """Return lifecycle metadata for one resolver-owned terminal surface."""

    if request["capability_kind"] == "task_resolution_request":
        return "task_status", _task_continuation_ref(request)
    return "ordinary", None


def _task_continuation_ref(
    request: ResolverCapabilityRequestV1,
) -> GoalContinuationRefV1:
    """Require the V1 task request to retain its V2 continuation identity."""

    continuation_ref = request["goal_continuation_ref"]
    if continuation_ref is None:
        raise ResolverValidationError(
            "task-resolution request requires goal_continuation_ref"
        )
    return continuation_ref


def _bind_required_evidence_dependency(
    resolver_state: ResolverCycleStateV1,
    *,
    selected_request: ResolverCapabilityRequestV1,
    observation: ResolverObservationV1,
    request_ordinal: int,
    required: bool,
) -> ResolverCycleStateV1:
    """Bind one required task request to its exact observation evidence state."""

    if (
        not required
        or selected_request["capability_kind"] != "task_resolution_request"
    ):
        return_value = resolver_state
        return return_value
    evidence_state = observation.get("task_resolution_evidence_state")
    if not isinstance(evidence_state, Mapping):
        raise ResolverValidationError(
            "task observation is missing its evidence state"
        )
    cycle_index = resolver_state["cycle_index"]
    evidence_handles = [
        f"resolver_evidence_{cycle_index}_{request_ordinal}_{index}"
        for index, evidence_ref in enumerate(
            observation["evidence_refs"],
            start=1,
        )
        if isinstance(evidence_ref.get("excerpt"), str)
        and evidence_ref["excerpt"].strip()
    ]
    dependency = {
        "schema_version": "required_resolver_evidence_dependency.v1",
        "accepted_request_handle": (
            f"resolver_request_{cycle_index}_{request_ordinal}"
        ),
        "observation_id": observation["observation_id"],
        "prompt_safe_observation_handle": (
            f"resolver_observation_{cycle_index}_{request_ordinal}"
        ),
        "capability_kind": "task_resolution_request",
        "state": evidence_state["state"],
        "evidence_handles": evidence_handles[:4],
        "remaining_needs": list(evidence_state["remaining_needs"]),
        "goal_continuation_ref": _task_continuation_ref(selected_request),
    }
    validated_dependency = validate_required_resolver_evidence_dependency(
        dependency,
    )
    updated = dict(resolver_state)
    updated["required_resolver_evidence_dependency"] = validated_dependency
    return_value = validate_resolver_state(updated)
    return return_value


def _mark_existing_dependency_blocked(
    resolver_state: ResolverCycleStateV1,
    observation: ResolverObservationV1,
) -> ResolverCycleStateV1:
    """Carry an existing required dependency into a terminal blocker state."""

    dependency = resolver_state.get("required_resolver_evidence_dependency")
    if dependency is None:
        return_value = resolver_state
        return return_value
    updated_dependency = dict(dependency)
    updated_dependency["state"] = "blocked"
    evidence_state = observation.get("task_resolution_evidence_state")
    if isinstance(evidence_state, Mapping):
        updated_dependency["remaining_needs"] = list(
            evidence_state["remaining_needs"]
        )
    updated = dict(resolver_state)
    updated["required_resolver_evidence_dependency"] = (
        validate_required_resolver_evidence_dependency(updated_dependency)
    )
    return_value = validate_resolver_state(updated)
    return return_value


def _timeout_observation(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> ResolverObservationV1:
    """Build a failed observation for a timed-out capability."""

    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": f"resolver_obs_timeout_{request['capability_kind']}",
        "capability_kind": request["capability_kind"],
        "request_objective": request["objective"],
        "request_reason": request["reason"],
        "status": "failed",
        "prompt_safe_summary": (
            f"Resolver capability timed out: {request['capability_kind']}"
        ),
        "evidence_refs": [],
        "created_at_utc": _created_at_utc(state),
    }
    if request["capability_kind"] == "task_resolution_request":
        observation["task_resolution_evidence_state"] = {
            "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
            "state": "blocked",
            "remaining_needs": [request["objective"]],
        }
        observation["goal_continuation_ref"] = _task_continuation_ref(request)
    return_value = validate_resolver_observation(observation)
    return return_value


def _duplicate_request_observation(
    request: ResolverCapabilityRequestV1,
    state: GlobalPersonaState,
) -> ResolverObservationV1:
    """Build a failed observation for an exact repeated capability request."""

    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": DUPLICATE_REQUEST_OBSERVATION_ID,
        "capability_kind": request["capability_kind"],
        "request_objective": request["objective"],
        "request_reason": request["reason"],
        "status": "failed",
        "prompt_safe_summary": (
            "Resolver blocked a duplicate capability request because the same "
            "objective, or a timed-out capability of the same kind, was already "
            "attempted in this resolver run."
        ),
        "evidence_refs": [],
        "created_at_utc": _created_at_utc(state),
    }
    if request["capability_kind"] == "task_resolution_request":
        observation["task_resolution_evidence_state"] = {
            "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
            "state": "blocked",
            "remaining_needs": [request["objective"]],
        }
        observation["goal_continuation_ref"] = _task_continuation_ref(request)
    return_value = validate_resolver_observation(observation)
    return return_value


def _should_surface_terminal_blocker(state: GlobalPersonaState) -> bool:
    """Return whether resolver-owned terminal blockers should be visible."""

    episode = state["cognitive_episode"]
    if not isinstance(episode, dict):
        raise ResolverValidationError("cognitive_episode: expected mapping")

    trigger_source = episode["trigger_source"]
    if not isinstance(trigger_source, str):
        raise ResolverValidationError(
            "cognitive_episode.trigger_source: expected string",
        )

    return_value = trigger_source == "user_message"
    return return_value


def _terminal_blocker_speak_action_spec(
    state: GlobalPersonaState,
    request: ResolverCapabilityRequestV1,
    blocker: ResolverObservationV1,
) -> dict[str, Any]:
    """Build a visible text-surface action for terminal resolver blockers."""

    surface_role, continuation_ref = _terminal_surface_metadata(request)
    detail = (
        f'围绕 objective={request["objective"]} 说明当前证据获取已经阻塞，'
        '不能给出来源确认的具体当前对象或状态；不要重复请求同一解析能力。'
        '可见回复应区分：已知约束、已由来源支持的事实、'
        '推断或常识层面的建议、当前无法确认的部分、'
        '以及用户或后续流程需要核实的最小项目。'
        '这是本轮终止收束，不得写成临时处理状态或延后承诺；'
        '必须在当前回复内给出可执行的最佳努力答案。'
        '如果原始目标或目标进度包含多部分交付、计划、路线、'
        '时间安排、风险、对比或执行步骤，必须按目标进度中的'
        'deliverables 和 final_response_requirements 收束；'
        '不要只回答最新证据或最容易回答的一个子问题。'
        '没有来源时不要补造具体当前实体、属性、实时状态、'
        '可用性或来源绑定结论。泛化说明也不能偷换成未授权的'
        '具体对象示例。'
        '不要把收束改成新的主澄清、放宽条件请求或开放式追问；'
        '如果提到可调整条件，只能作为可选退路，不能以追问结尾。'
    )
    action_spec = {
        "schema_version": ACTION_SPEC_VERSION,
        "kind": SPEAK_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": ACTION_SOURCE_REF_VERSION,
                "ref_kind": "system_event",
                "ref_id": blocker["observation_id"],
                "owner": "cognition_resolver",
                "relationship": "basis",
                "evidence_refs": [],
            },
        ],
        "target": {
            "schema_version": ACTION_TARGET_VERSION,
            "target_kind": "current_channel",
            "target_id": None,
            "owner": "l3_text",
            "scope": {"surface": "text"},
        },
        "params": {
            "delivery_mode": "visible_reply",
            "execute_at": None,
            "surface_requirements": {
                "decision": "explain terminal evidence blocker",
                "detail": detail,
            },
        },
        "urgency": "now",
        "visibility": "user_visible",
        "deadline": None,
        "continuation": {
            "schema_version": ACTION_CONTINUATION_VERSION,
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "surface_role": surface_role,
        "goal_continuation_ref": continuation_ref,
        "cognition_provenance": _resolver_speak_cognition_provenance(state),
        "reason": (
            "Resolver reached a terminal capability blocker and must surface "
            "the evidence boundary instead of looping silently."
        ),
    }
    validated_spec = validate_action_spec(action_spec)
    return_value = dict(validated_spec)
    return return_value


def _max_cycle_observation(
    state: GlobalPersonaState,
    resolver_state: ResolverCycleStateV1,
) -> ResolverObservationV1:
    """Build a structural observation when the recurrence cap is reached."""

    previous_observation = resolver_state["observations"][-1]
    observation = {
        "schema_version": RESOLVER_OBSERVATION_VERSION,
        "observation_id": MAX_CYCLE_OBSERVATION_ID,
        "capability_kind": previous_observation["capability_kind"],
        "request_objective": previous_observation["request_objective"],
        "request_reason": previous_observation["request_reason"],
        "status": "failed",
        "prompt_safe_summary": (
            "Resolver stopped because maximum resolver cycles were reached."
        ),
        "evidence_refs": [],
        "created_at_utc": _created_at_utc(state),
    }
    if previous_observation["capability_kind"] == "task_resolution_request":
        observation["task_resolution_evidence_state"] = {
            "schema_version": RESOLVER_EVIDENCE_STATE_VERSION,
            "state": "blocked",
            "remaining_needs": [previous_observation["request_objective"]],
        }
        observation["goal_continuation_ref"] = (
            previous_observation["goal_continuation_ref"]
        )
    return_value = validate_resolver_observation(observation)
    return return_value


def _build_cycle_trace(
    state: GlobalPersonaState,
    *,
    resolver_state: ResolverCycleStateV1,
    cycle_index: int,
    status_before: str,
    selected_capability_kind: str,
    observation_ids: list[str],
    terminal_reason: str,
) -> dict[str, Any]:
    """Build one prompt-safe cycle trace row from cognition outputs."""

    trace = {
        "schema_version": RESOLVER_CYCLE_TRACE_VERSION,
        "cycle_index": cycle_index,
        "status_before_cycle": status_before,
        "l1_emotional_appraisal": _state_text(state, "emotional_appraisal"),
        "l1_interaction_subtext": _state_text(state, "interaction_subtext"),
        "l2_internal_monologue_summary": _state_text(
            state,
            "internal_monologue",
        ),
        "l2_logical_stance": _state_text(state, "logical_stance"),
        "l2_character_intent": _state_text(state, "character_intent"),
        "l2_judgment_note": _state_text(state, "judgment_note"),
        "l2d_resolver_capability_requests": list(
            state.get("resolver_capability_requests", []),
        ),
        "l2d_action_specs_summary": _action_spec_summaries(
            state.get("action_specs", []),
        ),
        "selected_capability_kind": selected_capability_kind,
        "observation_ids": observation_ids,
        "final_surface_decision": _final_surface_decision(state),
        "terminal_reason": terminal_reason,
        "created_at_utc": _created_at_utc(state),
    }
    return_value = trace
    return return_value


def _action_spec_summaries(action_specs: object) -> list[str]:
    """Build bounded human-readable action-spec summaries for trace review."""

    if not isinstance(action_specs, list):
        return_value: list[str] = []
        return return_value
    summaries: list[str] = []
    for action_spec in action_specs:
        if not isinstance(action_spec, dict):
            continue
        kind = str(action_spec.get("kind", ""))
        urgency = str(action_spec.get("urgency", ""))
        visibility = str(action_spec.get("visibility", ""))
        reason = str(action_spec.get("reason", ""))
        summaries.append(
            f"kind={kind}; urgency={urgency}; visibility={visibility}; "
            f"reason={reason}"
        )
    return_value = summaries
    return return_value


def _final_surface_decision(state: GlobalPersonaState) -> str:
    """Summarize whether cognition selected a final surface."""

    action_specs = state.get("action_specs", [])
    if isinstance(action_specs, list) and action_specs:
        return_value = f"action_specs={len(action_specs)}"
        return return_value
    requests = state.get("resolver_capability_requests", [])
    if isinstance(requests, list) and requests:
        return_value = f"resolver_capability_requests={len(requests)}"
        return return_value
    return_value = "no action spec"
    return return_value


def _with_resolver_state(
    state: GlobalPersonaState,
    resolver_state: ResolverCycleStateV1,
) -> GlobalPersonaState:
    """Return state with resolver state and prompt-safe context refreshed."""

    validated_state = validate_resolver_state(resolver_state)
    updated = dict(state)
    updated["resolver_state"] = validated_state
    updated["resolver_context"] = project_resolver_context(validated_state)
    return_value = updated
    return return_value


def _merge_state(
    base_state: GlobalPersonaState,
    state_update: GlobalPersonaState,
) -> GlobalPersonaState:
    """Merge a cognition node update into the current persona state."""

    merged = dict(base_state)
    merged.update(state_update)
    return_value = merged
    return return_value


def _resolver_state(state: GlobalPersonaState) -> ResolverCycleStateV1:
    """Read and validate resolver state from persona state."""

    return_value = validate_resolver_state(state["resolver_state"])
    return return_value


def _state_text(state: GlobalPersonaState, field_name: str) -> str:
    """Return one cognition output as text for trace construction."""

    value = state.get(field_name, "")
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value
    return return_value


def _created_at_utc(state: GlobalPersonaState) -> str:
    """Read the current turn storage timestamp."""

    created_at = state.get("storage_timestamp_utc")
    if isinstance(created_at, str) and created_at.strip():
        return_value = created_at
        return return_value
    return_value = ""
    return return_value


def _validate_loop_limits(
    max_cycles: int,
    capability_timeout_seconds: float,
) -> None:
    """Validate deterministic loop caps."""

    if isinstance(max_cycles, bool) or not isinstance(max_cycles, int):
        raise ValueError("max_cycles: expected positive integer")
    if max_cycles < 1:
        raise ValueError("max_cycles: expected positive integer")
    if (
        isinstance(capability_timeout_seconds, bool)
        or not isinstance(capability_timeout_seconds, (int, float))
    ):
        raise ValueError(
            "capability_timeout_seconds: expected positive number",
        )
    if capability_timeout_seconds <= 0:
        raise ValueError(
            "capability_timeout_seconds: expected positive number",
        )
