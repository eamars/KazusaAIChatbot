"""Bounded recurrence controller around the preserved cognition subgraph."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_CYCLE_TRACE_VERSION,
    RESOLVER_OBSERVATION_VERSION,
    ResolverCapabilityRequestV1,
    ResolverCycleStateV1,
    ResolverObservationV1,
    validate_resolver_capability_request,
    validate_resolver_observation,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    append_cycle_trace,
    append_observation,
    ensure_initial_resolver_inputs,
    project_resolver_context,
    validate_resolver_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState

CognitionSubgraphFunc = Callable[
    [GlobalPersonaState],
    Awaitable[GlobalPersonaState],
]
CapabilityExecutorFunc = Callable[
    [ResolverCapabilityRequestV1, GlobalPersonaState],
    Awaitable[ResolverObservationV1],
]

MAX_CYCLE_OBSERVATION_ID = "resolver_obs_max_cycles"


async def call_cognition_resolver_loop(
    state: GlobalPersonaState,
    *,
    call_cognition_subgraph_func: CognitionSubgraphFunc,
    execute_capability_func: CapabilityExecutorFunc,
    max_cycles: int,
    capability_timeout_seconds: float,
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
        resolver_state = _resolver_state(cognition_state)
        selected_request = _select_immediate_request(cognition_state)

        if selected_request is None:
            final_state = _finalize_without_capability(
                cognition_state,
                resolver_state=resolver_state,
                status_before=status_before,
            )
            return_value = final_state
            return return_value

        observation = await _execute_with_timeout(
            selected_request,
            cognition_state,
            execute_capability_func=execute_capability_func,
            capability_timeout_seconds=capability_timeout_seconds,
        )
        resolver_state = append_observation(resolver_state, observation)
        if "rag_result" in observation:
            cognition_state["rag_result"] = observation["rag_result"]

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
        current_state = cognition_state

    return_value = await _run_max_cycle_final_cognition(
        current_state,
        call_cognition_subgraph_func=call_cognition_subgraph_func,
    )
    return return_value


def _finalize_without_capability(
    cognition_state: GlobalPersonaState,
    *,
    resolver_state: ResolverCycleStateV1,
    status_before: str,
) -> GlobalPersonaState:
    """Attach terminal trace/state when cognition does not need a capability."""

    action_specs = list(cognition_state.get("action_specs", []))
    updated_resolver_state = dict(resolver_state)
    updated_resolver_state["status"] = "terminal"
    updated_resolver_state["held_action_specs"] = action_specs
    updated_resolver_state["terminal_reason"] = "no resolver capability request"
    trace = _build_cycle_trace(
        cognition_state,
        resolver_state=updated_resolver_state,
        cycle_index=updated_resolver_state["cycle_index"],
        status_before=status_before,
        selected_capability_kind="",
        observation_ids=[],
        terminal_reason="no resolver capability request",
    )
    updated_resolver_state = append_cycle_trace(updated_resolver_state, trace)
    return_value = _with_resolver_state(cognition_state, updated_resolver_state)
    return return_value


async def _run_max_cycle_final_cognition(
    state: GlobalPersonaState,
    *,
    call_cognition_subgraph_func: CognitionSubgraphFunc,
) -> GlobalPersonaState:
    """Return one more cognition cycle with a structural max-cycle blocker."""

    resolver_state = _resolver_state(state)
    blocker = _max_cycle_observation(state, resolver_state)
    resolver_state = append_observation(resolver_state, blocker)
    updated_resolver_state = dict(resolver_state)
    updated_resolver_state["status"] = "max_cycles"
    updated_resolver_state["terminal_reason"] = "maximum resolver cycles reached"
    cognition_input = _with_resolver_state(state, updated_resolver_state)
    cognition_output = await call_cognition_subgraph_func(cognition_input)
    cognition_state = _merge_state(cognition_input, cognition_output)
    final_resolver_state = _resolver_state(cognition_state)
    final_resolver_state = dict(final_resolver_state)
    final_resolver_state["held_action_specs"] = list(
        cognition_state.get("action_specs", []),
    )
    trace = _build_cycle_trace(
        cognition_state,
        resolver_state=final_resolver_state,
        cycle_index=final_resolver_state["cycle_index"],
        status_before="max_cycles",
        selected_capability_kind="",
        observation_ids=[],
        terminal_reason="maximum resolver cycles reached",
    )
    final_resolver_state = append_cycle_trace(final_resolver_state, trace)
    return_value = _with_resolver_state(cognition_state, final_resolver_state)
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


def _select_immediate_request(
    state: GlobalPersonaState,
) -> ResolverCapabilityRequestV1 | None:
    """Return the first immediate resolver request selected by cognition."""

    requests = state.get("resolver_capability_requests", [])
    for request in requests:
        validated_request = validate_resolver_capability_request(request)
        if validated_request["priority"] == "now":
            return_value = validated_request
            return return_value
    return_value = None
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
    return_value = validate_resolver_observation(observation)
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
