"""V1-compatible entrypoint for the validation-local cognition core."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    CognitionChainInputV1,
    CognitionChainOutputV1,
    CognitionChainServices,
    validate_cognition_chain_input,
    validate_cognition_chain_output,
)
from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    select_semantic_actions,
)
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    select_final_branches,
    select_preliminary_branches,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    BranchDefinition,
    BranchResult,
    EmotionActivation,
    LocalMotivationalState,
    LocalStateKey,
    WorkspaceResult,
)
from kazusa_ai_chatbot.cognition_core_v2.dependency_graph import (
    build_dependency_graph,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    LOCAL_STATE_STORE,
    capture_validation_event,
    record_diagnostic,
)
from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
    EMOTION_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.emotion_derivation import (
    derive_emotion_activations,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import run_goal_branch
from kazusa_ai_chatbot.cognition_core_v2.output_projection import project_v1_output
from kazusa_ai_chatbot.cognition_core_v2.parallel_executor import (
    execute_dependency_graph,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    appraise_semantic_sources,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    proposals_from_semantic_propositions,
)
from kazusa_ai_chatbot.cognition_core_v2.workspace import (
    admit_branch_bids,
    integrate_workspace,
)


MAX_BRANCH_PROMPT_CHARS = 24000


async def run_cognition_chain(
    input_payload: CognitionChainInputV1,
    services: CognitionChainServices,
) -> CognitionChainOutputV1:
    """Validate V1 input and return the initial V2 validation projection.

    Args:
        input_payload: Prompt-safe public V1 cognition input.
        services: Existing V1 service bindings reserved for V2 semantic stages.

    Returns:
        A structurally validated V1 cognition output with a V2 trace marker.
    """

    validated_input = validate_cognition_chain_input(input_payload)
    capture_validation_event(
        "facade_input",
        {
            "episode_id": _episode_id(validated_input),
            "scope_fingerprint": resolve_local_state_fingerprint(validated_input),
        },
    )
    if _is_upstream_unambiguous(validated_input):
        output = _empty_v1_output(validated_input)
        validated_output = validate_cognition_chain_output(output)
        capture_validation_event(
            "facade_output",
            {"output": validated_output, "deterministic_shortcut": True},
        )
        return validated_output

    state_key = _resolve_local_state_key(validated_input)
    before_state = await LOCAL_STATE_STORE.snapshot(state_key)
    capture_validation_event("state_before", {"state": before_state})
    warnings: list[str] = []
    evidence = _project_evidence(validated_input)
    allowed_root_ids = {
        definition.causal_inputs[0]
        for definition in EMOTION_DEFINITIONS.values()
    }
    active_root_ids = {
        root_id
        for root_id, strength in _root_strengths(before_state).items()
        if root_id in allowed_root_ids and strength > 0.0
    }
    try:
        propositions = await appraise_semantic_sources(
            evidence,
            allowed_root_ids,
            services,
            active_root_ids,
        )
    except Exception as exc:
        services.logger.error(f"V2 semantic appraisal failed: {exc}")
        propositions = []
        warnings.append(f"semantic appraisal failed: {exc}")
    proposals = proposals_from_semantic_propositions(
        propositions,
        before_state.state_version,
    )
    if proposals:
        current_state = await LOCAL_STATE_STORE.apply_proposals(state_key, proposals)
    else:
        current_state = before_state
    capture_validation_event(
        "state_transition",
        {"proposals": proposals, "state_after_reduction": current_state},
    )
    root_strengths = _root_strengths(current_state)
    activations = derive_emotion_activations(current_state, root_strengths)
    current_state = await LOCAL_STATE_STORE.set_derived_activations(
        state_key,
        activations,
    )
    capture_validation_event(
        "emotion_derivation",
        {"activations": activations, "state_after_derivation": current_state},
    )
    preliminary_branches = select_preliminary_branches(activations)
    branches = select_final_branches(preliminary_branches, activations)
    dependency_graph = build_dependency_graph(branches)
    capture_validation_event(
        "dependency_graph",
        {
            "branch_definitions": branches,
            "dependencies": {
                branch_id: list(definition.dependencies)
                for branch_id, definition in dependency_graph.definitions.items()
            },
        },
    )
    semantic_state = _semantic_state_projection(activations)

    async def run_branch(definition: BranchDefinition) -> BranchResult:
        """Run one dependency-ready goal branch with only semantic local state."""

        branch_result = await run_goal_branch(
            definition,
            semantic_state,
            evidence,
            services,
        )
        return branch_result

    execution = await execute_dependency_graph(dependency_graph, run_branch)
    warnings.extend(execution.warnings)
    capture_validation_event(
        "branch_execution",
        {
            "results": execution.results,
            "warnings": execution.warnings,
            "started_at": execution.started_at,
            "ended_at": execution.ended_at,
            "maximum_concurrency": execution.maximum_concurrency,
        },
    )
    admitted_bids = admit_branch_bids(execution.results)
    try:
        workspace = await integrate_workspace(admitted_bids, services)
    except Exception as exc:
        services.logger.error(f"V2 workspace integration failed: {exc}")
        workspace = WorkspaceResult(
            selected_bid_id=None,
            public_intention="",
            internal_summary="workspace integration failed",
            suppressed_bid_ids=(),
        )
        warnings.append(f"workspace integration failed: {exc}")
    capture_validation_event(
        "workspace_result",
        {"admitted_bids": admitted_bids, "workspace": workspace},
    )
    try:
        action_requests = await select_semantic_actions(
            workspace,
            validated_input,
            services,
        )
    except Exception as exc:
        services.logger.error(f"V2 action selection failed: {exc}")
        action_requests = []
        warnings.append(f"action selection failed: {exc}")
    capture_validation_event(
        "action_selection_result",
        {"action_requests": action_requests},
    )
    output = project_v1_output(
        activations,
        workspace,
        action_requests,
        warnings,
    )
    validated_output = validate_cognition_chain_output(output)
    capture_validation_event(
        "facade_output",
        {"output": validated_output, "warnings": warnings},
    )
    record_diagnostic(
        {
            "scope_fingerprint": state_key.target_scope_fingerprint,
            "state_version_before": before_state.state_version,
            "state_version_after": current_state.state_version,
            "transition_count": len(proposals),
            "activated_emotions": sorted(
                emotion_id
                for emotion_id, activation in activations.items()
                if activation.activation > 0.0
            ),
            "branch_ids": sorted(execution.results),
            "warnings": warnings,
        }
    )
    return validated_output


def resolve_local_state_fingerprint(input_payload: Mapping[str, object]) -> str:
    """Return a stable one-way fingerprint for the V1 semantic target scope."""

    episode = input_payload["episode"]
    if not isinstance(episode, Mapping):
        raise TypeError("validated episode must be a mapping")
    scope_summary = episode["target_scope_summary"]
    if not isinstance(scope_summary, str):
        raise TypeError("validated target scope summary must be text")
    fingerprint = hashlib.sha256(scope_summary.encode("utf-8")).hexdigest()
    return fingerprint


def _resolve_local_state_key(input_payload: Mapping[str, object]) -> LocalStateKey:
    """Build one V2-owned state key from stable V1 scope identifiers."""

    character = input_payload["character"]
    current_user = input_payload["current_user"]
    episode = input_payload["episode"]
    if (
        not isinstance(character, Mapping)
        or not isinstance(current_user, Mapping)
        or not isinstance(episode, Mapping)
    ):
        raise TypeError("validated V1 identity sections must be mappings")
    state_key = LocalStateKey(
        character_global_id=_required_text(character, "character_global_id"),
        current_user_global_id=_required_text(current_user, "global_user_id"),
        trigger_source=_required_text(episode, "trigger_source"),
        target_scope_fingerprint=resolve_local_state_fingerprint(input_payload),
    )
    return state_key


def _episode_id(input_payload: Mapping[str, object]) -> str:
    """Read the V1 episode identity used to correlate a diagnostic capture."""

    episode = input_payload["episode"]
    if not isinstance(episode, Mapping):
        raise TypeError("validated episode must be a mapping")
    episode_id = _required_text(episode, "episode_id")
    return episode_id


def _is_upstream_unambiguous(input_payload: Mapping[str, object]) -> bool:
    """Honor an upstream fact that the current episode needs no appraisal."""

    episode = input_payload["episode"]
    if not isinstance(episode, Mapping):
        raise TypeError("validated episode must be a mapping")
    origin_summary = episode["origin_summary"]
    if not isinstance(origin_summary, str):
        raise TypeError("validated origin summary must be text")
    result = origin_summary == "unambiguous greeting"
    return result


def _project_evidence(input_payload: Mapping[str, object]) -> list[dict[str, str]]:
    """Collect bounded prompt-safe current evidence without semantic filtering."""

    current_event = input_payload["current_event"]
    episode = input_payload["episode"]
    if not isinstance(current_event, Mapping) or not isinstance(episode, Mapping):
        raise TypeError("validated V1 event sections must be mappings")
    source_summaries = [
        {
            "source_ref": _required_text(episode, "episode_id"),
            "summary": _required_text(current_event, "decontextualized_input"),
        },
    ]
    evidence = input_payload["evidence"]
    if not isinstance(evidence, Mapping):
        raise TypeError("validated evidence must be a mapping")
    for index, row in enumerate(evidence["memory_evidence"]):
        if not isinstance(row, Mapping):
            continue
        content = row.get("content")
        if isinstance(content, str) and content:
            source_summaries.append({
                "source_ref": f"memory:{index}",
                "summary": content,
            })
    total_chars = 0
    bounded_summaries: list[dict[str, str]] = []
    for source in source_summaries:
        remaining_chars = MAX_BRANCH_PROMPT_CHARS - total_chars
        if remaining_chars < 1:
            break
        summary = source["summary"][:remaining_chars]
        bounded_summaries.append({"source_ref": source["source_ref"], "summary": summary})
        total_chars += len(summary)
    return bounded_summaries


def _root_strengths(state: LocalMotivationalState) -> dict[str, float]:
    """Read local root activation fields for deterministic emotion derivation."""

    root_strengths: dict[str, float] = {}
    for entity_mapping_name in (
        "drives",
        "goals",
        "bonds",
        "threats",
        "standards",
        "incidents",
        "epistemic_state",
    ):
        entity_mapping = getattr(state, entity_mapping_name)
        for entity_ref, entity in entity_mapping.items():
            activation = entity.get("activation")
            if isinstance(activation, (int, float)):
                root_strengths[entity_ref] = float(activation)
    return root_strengths


def _semantic_state_projection(
    activations: Mapping[str, EmotionActivation],
) -> dict[str, str]:
    """Project activation state into qualitative branch prompts without floats."""

    projected = {
        emotion_id: _qualitative_activation(activation.activation)
        for emotion_id, activation in activations.items()
        if activation.activation > 0.0
    }
    return projected


def _qualitative_activation(activation: float) -> str:
    """Translate normalized state into a local-model-safe semantic descriptor."""

    if activation >= 0.75:
        return "strong and current"
    if activation >= 0.4:
        return "present and current"
    return "fading but unresolved"


def _required_text(value: Mapping[str, object], key: str) -> str:
    """Read a required public text field after V1 boundary validation."""

    field_value = value[key]
    if not isinstance(field_value, str):
        raise TypeError(f"validated {key} must be text")
    return field_value


def _empty_v1_output(input_payload: Mapping[str, object]) -> dict[str, object]:
    """Project an initial no-activation V2 state onto the public V1 schema."""

    output = {
        "schema_version": "cognition_chain_output.v1",
        "cognition_residue": {
            "emotional_appraisal": "no causally active emotion",
            "interaction_subtext": "no additional interpersonal pressure established",
            "internal_monologue": "no unresolved motive is active",
            "logical_stance": "preserve the current evidence boundary",
            "character_intent": "await a grounded cognition bid",
            "judgment_note": "no causal transition was proposed",
            "social_distance": "unchanged",
            "emotional_intensity": "none",
            "vibe_check": "unassessed",
            "relational_dynamic": "unchanged",
        },
        "semantic_action_requests": [],
        "resolver_capability_requests": [],
        "chain_trace": {
            "stage_order": ["v2"],
            "selected_actions_summary": "",
            "resolver_summary": "",
            "warnings": [],
        },
    }
    return output
