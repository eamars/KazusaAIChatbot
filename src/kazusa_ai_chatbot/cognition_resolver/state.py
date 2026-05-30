"""State helpers for the cognition-preserving resolver loop."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ALLOWED_RESOLVER_STATES,
    RESOLVER_CYCLE_STATE_VERSION,
    ResolverCycleStateV1,
    ResolverCycleTraceV1,
    ResolverObservationV1,
    ResolverValidationError,
    project_observations_for_cognition,
    project_pending_resume_for_cognition,
    validate_resolver_cycle_trace,
    validate_resolver_observation,
    validate_resolver_pending_resume,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import (
    project_known_facts,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState

MAX_PROJECTED_RESOLVER_OBSERVATIONS = 4


def new_resolver_state(
    *,
    decontexualized_input: str,
    max_cycles: int,
) -> ResolverCycleStateV1:
    """Create the initial resolver state for a decontextualized user turn."""

    _require_non_empty_text(decontexualized_input, "decontexualized_input")
    validated_max_cycles = _require_positive_int(max_cycles, "max_cycles")
    return_value: ResolverCycleStateV1 = {
        "schema_version": RESOLVER_CYCLE_STATE_VERSION,
        "cycle_index": 0,
        "max_cycles": validated_max_cycles,
        "status": "running",
        "original_decontexualized_input": decontexualized_input,
        "observations": [],
        "cycle_traces": [],
        "held_action_specs": [],
        "terminal_reason": "",
    }
    return return_value


def append_observation(
    state: ResolverCycleStateV1,
    observation: ResolverObservationV1,
) -> ResolverCycleStateV1:
    """Append one normalized capability observation without mutating state."""

    normalized_state = validate_resolver_state(state)
    normalized_observation = validate_resolver_observation(observation)
    observations = list(normalized_state["observations"])
    observations.append(normalized_observation)
    updated = dict(normalized_state)
    updated["observations"] = observations
    return_value = updated
    return return_value


def append_cycle_trace(
    state: ResolverCycleStateV1,
    trace: ResolverCycleTraceV1,
) -> ResolverCycleStateV1:
    """Append one normalized cycle trace and advance the next cycle index."""

    normalized_state = validate_resolver_state(state)
    normalized_trace = validate_resolver_cycle_trace(trace)
    cycle_traces = list(normalized_state["cycle_traces"])
    cycle_traces.append(normalized_trace)
    next_cycle_index = max(
        normalized_state["cycle_index"],
        normalized_trace["cycle_index"] + 1,
    )
    updated = dict(normalized_state)
    updated["cycle_traces"] = cycle_traces
    updated["cycle_index"] = next_cycle_index
    return_value = updated
    return return_value


def build_empty_rag_result(
    *,
    current_user_id: str,
    character_user_id: str,
) -> dict[str, Any]:
    """Build the empty RAG projection shape required by cognition nodes."""

    _require_non_empty_text(current_user_id, "current_user_id")
    _require_non_empty_text(character_user_id, "character_user_id")
    rag_result = project_known_facts(
        [],
        current_user_id=current_user_id,
        character_user_id=character_user_id,
        answer="",
        unknown_slots=[],
        loop_count=0,
    )
    return_value = rag_result
    return return_value


def ensure_initial_resolver_inputs(
    state: GlobalPersonaState,
    *,
    max_cycles: int,
) -> GlobalPersonaState:
    """Ensure first-cycle cognition receives resolver state and RAG shape."""

    if not isinstance(state, dict):
        raise ResolverValidationError("state: expected object")
    initialized = dict(state)
    rag_result = initialized.get("rag_result")
    if rag_result is None:
        current_user_id = _require_state_text(initialized, "global_user_id")
        character_user_id = _require_character_user_id(initialized)
        initialized["rag_result"] = build_empty_rag_result(
            current_user_id=current_user_id,
            character_user_id=character_user_id,
        )
    elif not isinstance(rag_result, dict):
        raise ResolverValidationError("rag_result: expected object")

    resolver_state = initialized.get("resolver_state")
    if resolver_state is None:
        decontexualized_input = _require_state_text(
            initialized,
            "decontexualized_input",
        )
        resolver_state = new_resolver_state(
            decontexualized_input=decontexualized_input,
            max_cycles=max_cycles,
        )
    else:
        resolver_state = validate_resolver_state(resolver_state)

    initialized["resolver_state"] = resolver_state
    initialized["resolver_context"] = project_resolver_context(resolver_state)
    return_value = initialized
    return return_value


def project_resolver_context(
    state: ResolverCycleStateV1,
    *,
    max_projected_observations: int = MAX_PROJECTED_RESOLVER_OBSERVATIONS,
) -> str:
    """Project bounded resolver status and observations for cognition."""

    normalized_state = validate_resolver_state(state)
    validated_limit = _require_positive_int(
        max_projected_observations,
        "max_projected_observations",
    )
    lines = [
        (
            f"resolver_state: status={normalized_state['status']}; "
            f"cycle_index={normalized_state['cycle_index']}; "
            f"max_cycles={normalized_state['max_cycles']}; "
            f"terminal_reason={normalized_state['terminal_reason']}"
        ),
    ]
    observations = normalized_state["observations"][-validated_limit:]
    observation_context = project_observations_for_cognition(observations)
    if observation_context:
        lines.append(f"resolver_observations:\n{observation_context}")
    pending_resume = normalized_state.get("pending_resume")
    pending_context = project_pending_resume_for_cognition(pending_resume)
    if pending_context:
        lines.append(pending_context)
    return_value = "\n".join(lines)
    return return_value


def validate_resolver_state(value: object) -> ResolverCycleStateV1:
    """Validate the deterministic resolver state shape."""

    data = _require_mapping(value, "resolver_state")
    _require_version(data, RESOLVER_CYCLE_STATE_VERSION)
    cycle_index = _require_non_negative_int(data.get("cycle_index"), "cycle_index")
    max_cycles = _require_positive_int(data.get("max_cycles"), "max_cycles")
    status = _require_state_enum(data, "status")
    original_input = _require_state_text(data, "original_decontexualized_input")
    raw_observations = _require_list(data, "observations")
    observations = [
        validate_resolver_observation(observation)
        for observation in raw_observations
    ]
    raw_traces = _require_list(data, "cycle_traces")
    cycle_traces = [
        validate_resolver_cycle_trace(trace)
        for trace in raw_traces
    ]
    held_action_specs = list(_require_list(data, "held_action_specs"))
    terminal_reason = _require_string(data, "terminal_reason")
    normalized: ResolverCycleStateV1 = {
        "schema_version": RESOLVER_CYCLE_STATE_VERSION,
        "cycle_index": cycle_index,
        "max_cycles": max_cycles,
        "status": status,
        "original_decontexualized_input": original_input,
        "observations": observations,
        "cycle_traces": cycle_traces,
        "held_action_specs": held_action_specs,
        "terminal_reason": terminal_reason,
    }
    pending_resume = data.get("pending_resume")
    if pending_resume is not None:
        normalized["pending_resume"] = validate_resolver_pending_resume(
            pending_resume,
        )
    return_value = normalized
    return return_value


def _require_mapping(value: object, label: str) -> dict:
    """Return a dictionary payload or raise a resolver validation error."""

    if not isinstance(value, dict):
        raise ResolverValidationError(f"{label}: expected object")
    return_value = value
    return return_value


def _require_version(data: dict, expected: str) -> None:
    """Require one schema version string."""

    actual = data.get("schema_version")
    if actual != expected:
        raise ResolverValidationError(f"schema_version: expected {expected}")


def _require_state_text(data: dict, field_name: str) -> str:
    """Require a non-empty text field from a state mapping."""

    value = data.get(field_name)
    _require_non_empty_text(value, field_name)
    return_value = value
    return return_value


def _require_string(data: dict, field_name: str) -> str:
    """Require one string field, allowing an empty string."""

    value = data.get(field_name)
    if not isinstance(value, str):
        raise ResolverValidationError(f"{field_name}: expected string")
    return_value = value
    return return_value


def _require_non_empty_text(value: object, field_name: str) -> None:
    """Require one non-empty string argument."""

    if not isinstance(value, str) or not value.strip():
        raise ResolverValidationError(f"{field_name}: expected non-empty string")


def _require_positive_int(value: object, field_name: str) -> int:
    """Require one positive integer argument."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ResolverValidationError(f"{field_name}: expected positive integer")
    if value < 1:
        raise ResolverValidationError(f"{field_name}: expected positive integer")
    return_value = value
    return return_value


def _require_non_negative_int(value: object, field_name: str) -> int:
    """Require one non-negative integer argument."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ResolverValidationError(f"{field_name}: expected non-negative integer")
    if value < 0:
        raise ResolverValidationError(f"{field_name}: expected non-negative integer")
    return_value = value
    return return_value


def _require_state_enum(data: dict, field_name: str) -> str:
    """Require one resolver-state enum string."""

    value = data.get(field_name)
    if not isinstance(value, str) or value not in ALLOWED_RESOLVER_STATES:
        expected = sorted(ALLOWED_RESOLVER_STATES)
        raise ResolverValidationError(f"{field_name}: expected one of {expected}")
    return_value = value
    return return_value


def _require_list(data: dict, field_name: str) -> list:
    """Require one list field."""

    value = data.get(field_name)
    if not isinstance(value, list):
        raise ResolverValidationError(f"{field_name}: expected list")
    return_value = value
    return return_value


def _require_character_user_id(state: dict) -> str:
    """Read the active character's global user id from state."""

    character_profile = state.get("character_profile")
    if not isinstance(character_profile, dict):
        raise ResolverValidationError("character_profile: expected object")
    character_user_id = character_profile.get("global_user_id")
    _require_non_empty_text(character_user_id, "character_profile.global_user_id")
    return_value = character_user_id
    return return_value
