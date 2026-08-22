"""Run V3 cognition through bounded cold and recurrence chains.

Cold turns run configured appraisal groups first, reduce their accepted rows
into the final state, and then build the ordinary and active goal bids from
that state. Workspace collapse remains conditional; action planning runs
after the accepted collapse and receives the same final-state carriers and
transcript. Recurrence retains its carried relational decision and resolver
tail while sharing the prompt-safe projection owners.

The deterministic head, state reduction, relationship maintenance, workspace
collapse, action planning, authorization, and output assembly use the
canonical protocol owners, so the public contract remains exactly
``CognitionCoreInputV2``/``CognitionCoreOutputV2``. Each semantic producer has
its bounded attempt ledger and typed fail-closed contract.

Accepted appraisal content bridges into native state through code-owned rows:
propositions attach to their source evidence row's candidate event root so the
native materializer resolves them through causal-event provenance, except
terminal outcome propositions whose subject binds to the unique lifecycle-
eligible entity of the asserted kind. Axis deltas translate into exact native
increment rows only when their axis matches exactly one permitted concrete
path for the stage's authorized evidence domain; every unbound or ambiguous
delta is dropped with a deterministic warning instead of reaching the native
reducer. Role assignments remain part of accepted V2 semantic products and
retain their canonical handle ordering through the appraisal bridge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import partial
from typing import Any
from uuid import uuid4

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot import db, event_logging, llm_tracing
from kazusa_ai_chatbot.cognition_core_v3.registry import (
    branch_order_key,
    select_preliminary_branches,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS,
    SELF_COGNITION_RESPONSE_DECISION_VALUES,
    SELF_COGNITION_RESPONSE_PARTICIPATION_VALUES,
    SELF_COGNITION_RESPONSE_TEXT_LIMIT,
    ActionBidV2,
    BranchDefinition,
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionExecutionError,
    is_targetless_group_self_cognition_episode,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_core_v3.facade_helpers import (
    _apply_final_relationship_maintenance,
    _bids_with_live_goals,
    _bind_pending_resolution,
    _branch_context,
    _build_cognition_observability,
    _cognition_elapsed_seconds,
    _deduplicate_diagnostics_warnings,
    _elapsed_ms,
    _episode_updated_at,
    _fact_without_producer,
    _goal_for_branch,
    _goal_projection,
    _mark_cognition_partial_failures,
    _native_relationship_context,
    _ordinary_relational_decision,
    _raise_for_unrecoverable_required_branch_failures,
    _reduce_appraisals_with_isolation,
    _resolver_progress,
    _selected_bid,
    _workspace_current_event,
    _workspace_goal_contexts,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    V2_APPRAISAL_TOTAL_ATTEMPTS,
    V2_MODEL_TOTAL_ATTEMPTS,
    V2AttemptBudgetExhausted,
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    current_v2_attempt_ledger,
    record_v2_attempt_disposition,
    record_v2_branch_disposition,
    reserve_v2_model_attempt,
    reset_v2_attempt_ledger,
    snapshot_v2_attempt_ledger,
)
from kazusa_ai_chatbot.cognition_shared.output_projection import (
    build_state_update,
    default_expression_policy,
    project_affect,
    project_relationship,
)
from kazusa_ai_chatbot.cognition_core_v3.execution_types import (
    BranchFailure,
    ParallelExecutionResult,
)
from kazusa_ai_chatbot.cognition_core_v3.semantic_source_planner import (
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_shared.state_projection import (
    PromptProjectionV2,
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_shared.state_reducers import (
    apply_state_update,
    create_deterministic_goals,
)
from kazusa_ai_chatbot.cognition_core_v3 import anchor as v3_anchor
from kazusa_ai_chatbot.cognition_core_v3 import authorization as v3_authorization
from kazusa_ai_chatbot.cognition_core_v3 import prompt as v3_prompt
from kazusa_ai_chatbot.cognition_core_v3.action_selection import (
    _authorization_repair_message,
    _materialize_action_requests,
    _materialize_resolver_requests,
    _self_cognition_target_handles,
    apply_stance_suppression,
    authorize_action_requests,
    authorize_resolver_requests,
    derive_action_route,
    accepted_at_utc_from_episode,
    validate_action_plan_decision,
    settle_resolver_outcome,
)
from kazusa_ai_chatbot.cognition_core_v3.appraisal import (
    reduce_appraisal_stage_output,
)
from kazusa_ai_chatbot.cognition_core_v3.budget import (
    CognitionContextLimitError,
    ContextBudgetLedger,
    ContextBudgetPlan,
)
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    CognitionChainServicesV3,
)
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    CHAIN_LEDGER_DEFAULTS,
    CHAIN_SIDECAR_DEFAULTS,
    bind_chain_sidecar_state,
    bind_protected_chain_records,
    current_chain_scope,
    record_accepted_transcript,
    record_chain_step,
    record_chain_system_head,
    record_current_sidecar_aggregate,
    record_degradation_marker,
    record_registered_step,
    record_session_event,
    record_sidecar_aggregate,
    record_token_ledger,
    record_warning_codes,
    reset_protected_chain_records,
    snapshot_protected_chain_records,  # noqa: F401
)
from kazusa_ai_chatbot.cognition_core_v3.execution import (
    AttemptLedger,
    SerialChainHarness,
    SerialQuestionResult,
    TurnDeadlineExceeded,
    check_turn_deadline,
    config_for_turn_deadline,
    invoke_lane_scoped_json_repair,
    invoke_serial_question_with_repair,
)
from kazusa_ai_chatbot.cognition_core_v3.goal_cognition import (
    GOAL_BID_EVIDENCE_HANDLE_LIMIT,
    ORDINARY_GOAL_KIND,
    validate_goal_bid_draft,
    selection_goal_draft_to_goal_bid,
    project_conversation_progress_evidence,
    project_required_selection_operations,
    salvage_active_goal_group_output,
    validate_active_goal_group_output,
    validate_recurrence_ordinary_goal_bid_draft,
    validate_selection_goal_draft,
)
from kazusa_ai_chatbot.cognition_core_v3.workspace import (
    collapse_authoritative_relational_bid,
    validate_workspace_partition,
)
from kazusa_ai_chatbot.cognition_core_v3.lane import (
    SidecarAdmissionLedger,
    SidecarCoordinator,
    SidecarInvocationState,
    primary_lane_coordinator,
    sidecar_lane_coordinator,
)
from kazusa_ai_chatbot.cognition_core_v3.registry import (
    APPRAISAL_FAMILY_ORDER,
    APPRAISAL_STAGE_FAMILIES,
    SERIAL_CHAIN_STEPS,
)
from kazusa_ai_chatbot.cognition_core_v3.session import (
    ChainSessionRegistry,
    ChainSessionV1,
    SessionContractError,
    advance_session_after_output,
    build_session_key,
    create_cold_session,
    reattach_or_rebuild,
)
from kazusa_ai_chatbot.cognition_core_v3.subconscious import (
    try_validate_l1_residue,
)
from kazusa_ai_chatbot.cognition_core_v3.transcript import (
    ChainMessageV1,
    ChainTranscriptV1,
)
from kazusa_ai_chatbot.cognition_core_v3.workspace import (
    collapse_single_bid,
    fallback_partition_envelope,
    materialize_partition,
    prepare_partition,
    validate_complete_bids,
)
from kazusa_ai_chatbot.cognition_episode import (
    build_goal_continuation_ref,
    project_dialog_response_operation,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    CURRENT_TURN_RELATIONAL_WILLINGNESS_VERSION,
)
from kazusa_ai_chatbot.config import (
    AUDIT_LOG_TTL_DAYS,
    COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS,
    COGNITION_RESOLVER_MAX_CYCLES,
    COGNITION_V3_APPRAISAL_STAGE_LAYOUT,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig
from kazusa_ai_chatbot.llm_tracing import failure_capsule
from kazusa_ai_chatbot.logging_retention import expiry_from_storage_iso
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso
from kazusa_ai_chatbot.utils import parse_llm_json_output

_PROVIDER_EXCEPTIONS = (
    OpenAIError,
    httpx.HTTPError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
)

_APPRAISAL_COMPLETION_CAP = 4_096
_GOAL_COMPLETION_CAP = 8_192
_WORKSPACE_COMPLETION_CAP = 2_048
_ACTION_COMPLETION_CAP = 8_192
_SIDECAR_COMPLETION_CAP = 1_024
_JSON_REPAIR_COMPLETION_CAP = 8_192
_V2_OWNER_COMPLETION_CAPS = {
    "semantic_appraisal": _APPRAISAL_COMPLETION_CAP,
    "goal_bid_structure": _GOAL_COMPLETION_CAP,
    "workspace_collapse": _WORKSPACE_COMPLETION_CAP,
    "action_planning": _ACTION_COMPLETION_CAP,
}
_CANONICALIZE_EXCEPTIONS = (AttributeError, KeyError, TypeError, ValueError)
_LOGGER = logging.getLogger(__name__)


def _bounded_stage_config(
    config: LLMCallConfig,
    *,
    stage_name: str,
    completion_cap: int,
) -> LLMCallConfig:
    """Apply a code-owned cap without increasing the configured lane cap."""

    configured_cap = config.max_completion_tokens
    bounded_cap = configured_cap
    if isinstance(configured_cap, int) and not isinstance(
        configured_cap,
        bool,
    ):
        bounded_cap = min(configured_cap, completion_cap)
    return replace(
        config,
        stage_name=stage_name,
        max_completion_tokens=bounded_cap,
    )


def _json_repair_cap(attempt_coordinates: Mapping[str, object]) -> int:
    """Return the failed primary owner's bounded repair cap."""

    stage_name = attempt_coordinates.get("producing_stage")
    if not isinstance(stage_name, str):
        return _JSON_REPAIR_COMPLETION_CAP
    owner_cap = _V2_OWNER_COMPLETION_CAPS.get(
        stage_name,
        _JSON_REPAIR_COMPLETION_CAP,
    )
    return min(owner_cap, _JSON_REPAIR_COMPLETION_CAP)

_STAGE_CONFIG_FIELDS: dict[str, str] = {
    "event_agency": "appraisal_event_agency_config",
    "moral_identity": "appraisal_moral_identity_config",
    "relationship_social": "appraisal_relationship_social_config",
    "epistemic_comparison_memory": (
        "appraisal_epistemic_comparison_memory_config"
    ),
    "existential_drive": "appraisal_existential_drive_config",
    "goal_threat_outcome": "appraisal_goal_threat_outcome_config",
}

_CHAIN_SESSION_REGISTRY = ChainSessionRegistry()


def _goal_authoritative_state(
    *,
    final_state: Mapping[str, Any],
    current_goal_kind: str,
    final_projection: PromptProjectionV2,
) -> dict[str, Any]:
    """Project the post-I1 state facts whose first consumer is G1a."""

    projected_payload = final_projection.payload
    matter_projections = {
        field_name: [dict(row) for row in projected_payload.get(field_name, [])]
        for field_name in ("goals", "threats", "events", "knowledge_gaps")
    }
    current_goal = _goal_for_branch(final_state, current_goal_kind)
    authoritative_state = {
        "current_goal": _goal_projection(current_goal, current_goal_kind),
        "matter_projections": matter_projections,
        "relationship_projection": project_relationship(
            final_state.get("relationship")
        ),
        "affect": project_affect(
            final_state.get("affect_activations", []),
            final_state,
        ),
    }
    return authoritative_state


def _goal_continuity_context(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build one bounded continuity carrier shared by each goal question."""

    engagement_context = payload["group_engagement_action_context"]
    if not isinstance(engagement_context, Mapping):
        raise CognitionExecutionError(
            "goal continuity engagement context is invalid",
            error_code="goal_continuity_context_invalid",
            stage="branch_cognition",
            safe_checkpoint="pre_state_commit",
        )
    continuity_context = {
        "private_continuity_context": payload[
            "private_continuity_context"
        ],
        "past_dialog_cognition_context": payload[
            "past_dialog_cognition_context"
        ],
        "group_engagement_action_context": {
            "engagement_guidelines": list(
                engagement_context["engagement_guidelines"]
            ),
            "confidence": engagement_context["confidence"],
        },
    }
    return continuity_context


def _goal_dialogue_role_bindings(
    observation_context: Mapping[str, object],
) -> list[dict[str, str]]:
    """Copy normalized dialogue role rows into each goal question carrier."""

    conversation_frame = observation_context["conversation_frame"]
    if not isinstance(conversation_frame, Mapping):
        raise CognitionExecutionError(
            "goal dialogue role context is invalid",
            error_code="goal_dialogue_role_context_invalid",
            stage="branch_cognition",
            safe_checkpoint="pre_state_commit",
        )
    role_bindings = conversation_frame["dialogue_role_bindings"]
    if not isinstance(role_bindings, list):
        raise CognitionExecutionError(
            "goal dialogue role context is invalid",
            error_code="goal_dialogue_role_context_invalid",
            stage="branch_cognition",
            safe_checkpoint="pre_state_commit",
        )
    copied_bindings = [dict(binding) for binding in role_bindings]
    return copied_bindings


def _appraisal_role_assignment_handles_by_evidence(
    observation_context: Mapping[str, object],
    evidence: Sequence[Mapping[str, object]],
) -> dict[str, list[str]]:
    """Bind role-assignment authority to explicit current-event evidence."""

    conversation_frame = observation_context.get("conversation_frame")
    if not isinstance(conversation_frame, Mapping):
        raise CognitionExecutionError(
            "appraisal conversation frame is invalid",
            error_code="semantic_appraisal_role_context_invalid",
            stage="semantic_appraisal",
            safe_checkpoint="pre_state_commit",
        )
    dialogue_bindings = conversation_frame.get("dialogue_role_bindings")
    participant_bindings = conversation_frame.get("participant_bindings")
    if not isinstance(dialogue_bindings, list) or not isinstance(
        participant_bindings,
        list,
    ):
        raise CognitionExecutionError(
            "appraisal role bindings are invalid",
            error_code="semantic_appraisal_role_context_invalid",
            stage="semantic_appraisal",
            safe_checkpoint="pre_state_commit",
        )
    explicit_handles = {
        binding[field_name]
        for binding in dialogue_bindings
        if isinstance(binding, Mapping)
        for field_name in (
            "speaker_handle",
            "addressee_handle",
            "first_person_handle",
            "implicit_imperative_subject_handle",
            "second_person_handle",
        )
        if isinstance(binding.get(field_name), str)
    }
    explicit_handles.update(
        binding["handle"]
        for binding in participant_bindings
        if isinstance(binding, Mapping)
        and isinstance(binding.get("handle"), str)
    )
    role_domains: dict[str, list[str]] = {}
    for evidence_row in evidence:
        handle = evidence_row.get("evidence_handle")
        if not isinstance(handle, str):
            raise CognitionExecutionError(
                "appraisal evidence handle is invalid",
                error_code="semantic_appraisal_role_context_invalid",
                stage="semantic_appraisal",
                safe_checkpoint="pre_state_commit",
            )
        authority = evidence_row.get("authority")
        if authority in {"current_event", "current_episode"}:
            role_domains[handle] = sorted(explicit_handles)
        else:
            role_domains[handle] = []
    return role_domains


def _active_goal_roster_row(
    *,
    definition: BranchDefinition,
    final_state: Mapping[str, Any],
    evidence_handles: Sequence[str],
    role_handles: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Build one compact G1b roster row with validator-owned domains."""

    goal = _goal_for_branch(final_state, definition.goal_kind)
    return {
        "branch_id": definition.branch_id,
        "goal_kind": definition.goal_kind,
        "goal_projection": _goal_projection(goal, definition.goal_kind),
        "branch_intent_guidance": definition.branch_intent_guidance,
        "action_tendencies": list(definition.action_tendencies),
        "allowed_evidence_handles": list(evidence_handles),
        "allowed_role_handles": sorted(role_handles),
    }


def _post_i1_goal_roster(
    final_state: Mapping[str, Any],
) -> list[BranchDefinition]:
    """Build the next G1 roster from canonical post-I1 goal statuses."""

    active_definitions = select_preliminary_branches(final_state["goals"])
    roster = [
        replace(definition, dependencies=(), dependency_options=())
        for definition in active_definitions
    ]
    return roster


def _selected_bid_handle(
    bid: Mapping[str, Any] | None,
    bid_handles: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """Resolve one selected branch to its precomputed stable ``bN`` handle."""

    if bid is None:
        return None
    branch_id = bid.get("branch_id")
    for handle, candidate in bid_handles.items():
        if candidate.get("branch_id") == branch_id:
            return handle
    raise CognitionExecutionError(
        "selected bid is absent from the stable handle index",
        error_code="workspace_bid_handle_missing",
        stage="workspace_collapse",
        safe_checkpoint="pre_state_commit",
        retryable=False,
    )


def _appraisal_relation_context(
    projection_payload: Mapping[str, Any],
    scene_context: Mapping[str, Any],
) -> dict[str, object]:
    """Build the relation-owned context consumed only by A2 appraisal."""

    relation_context: dict[str, object] = {
        "character_constraints": dict(
            projection_payload["character_constraints"]
        ),
        "character_operational_context": dict(
            projection_payload.get("character_operational_context", {})
        ),
        "relationship_projection": dict(
            projection_payload.get("relationship", {})
        ),
        "current_affect": list(projection_payload.get("affect", [])),
    }
    if "character_sleep_phase" in scene_context:
        relation_context["character_sleep_phase"] = scene_context[
            "character_sleep_phase"
        ]
    return relation_context


def _materialize_goal_bid(
    definition: BranchDefinition,
    state: Mapping[str, Any],
    role_bindings: Mapping[str, Mapping[str, str]],
    local_state: Mapping[str, Any],
) -> ActionBidV2:
    """Map one accepted goal draft into the exact V2 action bid shape.

    The goal reference derivation and target-role materialization mirror the
    V2 branch handler; selection mode maps its bounded fields onto the V2
    intention triple while carrying the code-bound operation, and ordinary
    drafts carry their validated or carrier-materialized relational stance.
    """

    goal_kind = definition.goal_kind
    goal = _goal_for_branch(state, goal_kind)
    if goal is None:
        goal_ref: dict[str, str] = {
            "scope": state["state_scope"],
            "kind": "goal",
            "entity_id": f"goal:{goal_kind}:episode",
        }
    else:
        goal_ref = {
            "scope": state["state_scope"],
            "kind": "goal",
            "entity_id": goal["entity_id"],
        }
    target_roles = [
        dict(role_bindings[handle])
        for handle in local_state["target_role_handles"]
    ]
    if "selection" in local_state:
        selection_text = local_state["selection"]
        bid: ActionBidV2 = {
            "branch_id": definition.branch_id,
            "goal_ref": goal_ref,
            "intention": selection_text,
            "desired_outcome": selection_text,
            "concrete_detail": selection_text,
            "reason": local_state["reason"],
            "private_monologue": local_state["private_monologue"],
            "target_roles": target_roles,
            "evidence_handles": list(local_state["evidence_handles"]),
            "expected_consequences": list(
                local_state["expected_consequences"]
            ),
            "confidence": local_state["confidence"],
        }
        bid["selected_response_operation"] = dict(
            local_state["selected_response_operation"]
        )
    else:
        bid = {
            "branch_id": definition.branch_id,
            "goal_ref": goal_ref,
            "intention": local_state["intention"],
            "desired_outcome": local_state["desired_outcome"],
            "concrete_detail": local_state["concrete_detail"],
            "reason": local_state["reason"],
            "private_monologue": local_state["private_monologue"],
            "target_roles": target_roles,
            "evidence_handles": list(local_state["evidence_handles"]),
            "expected_consequences": list(
                local_state["expected_consequences"]
            ),
            "confidence": local_state["confidence"],
        }
        if "selected_response_operation" in local_state:
            bid["selected_response_operation"] = dict(
                local_state["selected_response_operation"]
            )
    if (
        goal_kind == ORDINARY_GOAL_KIND
        and "relational_willingness" in local_state
    ):
        bid["relational_willingness"] = dict(
            local_state["relational_willingness"]
        )
    return bid


def _validate_and_materialize_selection_goal_draft(
    parsed: dict[str, object],
    *,
    evidence_handles: set[str],
    role_handles: set[str],
    selection_operations: Sequence[Mapping[str, Any]],
    episode_handles: set[str] | None,
    require_relational_willingness: bool,
) -> dict[str, Any]:
    """Use the canonical V2 selection validator and materializer for G1a."""

    required_evidence_handles: set[str] = {
        operation["evidence_handle"]
        for operation in selection_operations
    }
    validated_draft = validate_selection_goal_draft(
        parsed,
        evidence_handles=evidence_handles,
        role_handles=role_handles,
        required_evidence_handles=required_evidence_handles,
        required_operations=selection_operations,
        episode_handles=episode_handles,
        require_relational_willingness=require_relational_willingness,
        maximum_evidence_handles=max(
            GOAL_BID_EVIDENCE_HANDLE_LIMIT,
            len(required_evidence_handles),
        ),
    )
    return selection_goal_draft_to_goal_bid(
        validated_draft,
        branch_id=ORDINARY_GOAL_KIND,
        include_relational_willingness=require_relational_willingness,
    )


def _synthesize_branch_execution(
    definitions: Sequence[BranchDefinition],
    bids_by_kind: Mapping[str, ActionBidV2],
    unavailable_kinds: Mapping[str, str],
    ledger: AttemptLedger,
) -> ParallelExecutionResult:
    """Synthesize one wave's branch execution from complete-bid join results.

    Available definitions contribute materialized bids keyed by V2 branch id and
    unavailable ones contribute typed failure records, so the protected
    required-branch escalation and observability helpers run verbatim over the
    synthesized result. Every started definition keeps a started_at entry even
    when its chain failed; synthesized waves record no per-branch timing, so
    overlap and dependency wait stay at zero rather than inventing measurements.
    Synthesized failures carry no exception identity: the engine's protected
    consumers read only the typed error code from these records.
    """

    branch_id_by_kind = {
        definition.goal_kind: definition.branch_id for definition in definitions
    }
    results: dict[str, Any] = {}
    started_at: dict[str, float] = {
        definition.branch_id: 0.0 for definition in definitions
    }
    failed_ids: set[str] = set()
    failure_records: dict[str, BranchFailure] = {}
    for goal_kind, bid in bids_by_kind.items():
        results[branch_id_by_kind[goal_kind]] = bid
    for goal_kind, error_code in unavailable_kinds.items():
        branch_id = branch_id_by_kind[goal_kind]
        failed_ids.add(branch_id)
        failure_records[branch_id] = BranchFailure(
            branch_id=branch_id,
            error_code=error_code,
            stage="goal_cognition",
            attempt_count=_goal_branch_attempt_count(
                branch_id,
                fallback=ledger.attempts_used(goal_kind),
            ),
            safe_checkpoint="final_reduction",
            retryable=False,
            exception_class=None,
        )
    return ParallelExecutionResult(
        results=results,
        warnings=[],
        started_at=started_at,
        ended_at={},
        maximum_concurrency=len(definitions),
        failed_branch_ids=failed_ids,
        failure_records=failure_records,
    )


def _synthesize_and_recover_goal_phase(
    definitions: Sequence[BranchDefinition],
    bids: Sequence[ActionBidV2],
    branch_failures: Mapping[str, str],
    ledger: AttemptLedger,
) -> ParallelExecutionResult:
    """Join one goal phase and apply the canonical required-branch rule."""

    definitions_by_branch = {
        definition.branch_id: definition
        for definition in definitions
    }
    bids_by_kind: dict[str, ActionBidV2] = {}
    for bid in bids:
        branch_id = bid["branch_id"]
        definition = definitions_by_branch.get(branch_id)
        if definition is not None:
            bids_by_kind[definition.goal_kind] = bid
    unavailable_kinds = {
        definition.goal_kind: branch_failures.get(
            definition.branch_id,
            "goal_bid_unavailable",
        )
        for definition in definitions
        if definition.goal_kind not in bids_by_kind
    }
    execution = _synthesize_branch_execution(
        definitions,
        bids_by_kind,
        unavailable_kinds,
        ledger,
    )
    _raise_for_unrecoverable_required_branch_failures(
        execution,
        definitions,
    )
    return execution


async def _invoke_goal_branch_for_phase_recovery(
    **kwargs: Any,
) -> SerialQuestionResult | None:
    """Return a goal result while preserving context failure for phase join."""

    try:
        return await invoke_serial_question_with_repair(**kwargs)
    except CognitionContextLimitError:
        return None


def _goal_branch_attempt_count(branch_id: str, *, fallback: int) -> int:
    """Read actual V2 goal reservations for one stable branch."""

    snapshot = snapshot_v2_attempt_ledger()
    if snapshot is None:
        return fallback
    attempts = snapshot.get("attempts", [])
    if not isinstance(attempts, Sequence):
        return fallback
    return sum(
        1
        for attempt in attempts
        if isinstance(attempt, Mapping)
        and attempt.get("producing_stage") == "goal_bid_structure"
        and attempt.get("branch_id") == branch_id
    )


_TERMINAL_PROPOSITION_KINDS = {
    "goal_completed": "goal",
    "event_completed": "event",
    "event_repaired": "event",
    "threat_resolved": "threat",
    "knowledge_answered": "knowledge_gap",
}


def _build_i1_notice(
    *,
    accepted_count: int,
    rejected_count: int,
    state_scope: str,
) -> str:
    """Build one deterministic bounded I1 state notice for the next question."""

    if accepted_count < 0 or rejected_count < 0:
        raise ValueError("I1 counts must be non-negative")
    if state_scope not in {"user", "character"}:
        raise ValueError("I1 state_scope must be user or character")
    notice = (
        f"确定性归约完成：接受 {accepted_count} 项，拒绝 "
        f"{rejected_count} 项；状态域 {state_scope}。"
    )
    if len(notice) > 600:
        raise ValueError("I1 notice exceeds the bounded 600-character ceiling")
    return notice


def _build_serial_initial_context(
    input_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the serial cold-chain deterministic head and first packet."""

    payload = validate_cognition_core_input(input_payload)
    previous_state = validate_cognition_state(payload["mutable_state"])
    updated_at = _episode_updated_at(payload["episode"])
    elapsed_seconds = _cognition_elapsed_seconds(previous_state, updated_at)
    fact_pairs = [
        (fact["producer"], _fact_without_producer(fact))
        for fact in payload["direct_facts"]
    ]
    reducer_relationship_context = _native_relationship_context(
        payload.get("relationship_context"),
    )
    preliminary_state = apply_state_update(
        previous_state,
        direct_facts=fact_pairs,
        elapsed_seconds=elapsed_seconds,
        updated_at=updated_at,
        character_constraints=payload["character_constraints"],
        relationship_context=reducer_relationship_context,
    )
    preliminary_state = create_deterministic_goals(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        relationship_context=reducer_relationship_context,
        evidence=payload["evidence"],
        updated_at=updated_at,
    )
    preliminary_state = validate_cognition_state(preliminary_state)
    projection = project_state_for_prompt(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        relationship_context=payload.get("relationship_context"),
        character_operational_context=payload.get(
            "character_operational_context",
        ),
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
    )
    questions = plan_semantic_questions(
        payload["evidence"],
        preliminary_state,
        projection.handle_to_ref,
    )
    identity = projection.identity_by_question.get("goal_cognition")
    if not isinstance(identity, Mapping):
        raise CognitionExecutionError(
            "serial chain requires canonical goal-cognition identity",
            error_code="serial_identity_missing",
            stage="pre_state_commit",
            safe_checkpoint="pre_state_commit",
            retryable=False,
        )
    system_content = v3_anchor.build_system_head(identity)
    observation_context = v3_prompt.build_observation_context(
        projection_payload=projection.payload,
        scene_context=payload["scene_context"],
        episode=payload["episode"],
        evidence=payload["evidence"],
        direct_facts=payload["direct_facts"],
        handle_to_ref=projection.handle_to_ref,
    )
    return {
        "payload": payload,
        "previous_state": previous_state,
        "preliminary_state": preliminary_state,
        "projection": projection,
        "questions": questions,
        "system_content": system_content,
        "observation_context": observation_context,
        "relation_context": _appraisal_relation_context(
            projection.payload,
            payload["scene_context"],
        ),
    }


_L1_SUBCONSCIOUS_PROMPT = '''You are the advisory L1 subconscious for a character cognition system.
Read only the supplied current percept, qualitative affect bands, bounded
boundary summary, and evidence handles. Return an immediate non-binding
reaction. You cannot create facts, evidence, stance, willingness, permissions,
actions, resolver work, or a response route. Salience hints must be drawn only
from supplied_evidence_handles.

# Output Format
Return one JSON object with exactly these fields:
- schema_version: the literal "l1_residue.v1"
- emotional_appraisal: 1..120 character advisory text
- interaction_subtext: 0..200 character advisory text
- salience_hints: duplicate-free list of at most four supplied evidence handles
- risk_flags: duplicate-free list of closed advisory risk labels
'''


async def _run_l1_sidecar(
    *,
    packet: str,
    supplied_evidence_handles: Sequence[str],
    services: CognitionChainServicesV3,
    coordinator: SidecarCoordinator,
    invocation_state: SidecarInvocationState,
    deadline_monotonic: float | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run one optional L1 request without granting it semantic authority.

    Args:
        packet: Fresh bounded L1 input packet with no transcript or evidence
            text.
        supplied_evidence_handles: Exact turn handles L1 may cite as salience.
        services: Injected V3 routes and model invoker.
        coordinator: The serialized sidecar lane owner.
        invocation_state: Invocation-local sidecar diagnostics and task state.

    Returns:
        A validated advisory residue plus no warning, or no residue plus a
        bounded warning code for an optional-stage failure.
    """

    sidecar_lane = services.sidecar_lane
    if sidecar_lane is None:
        raise RuntimeError("L1 execution requires an injected sidecar lane")
    messages = [
        SystemMessage(content=_L1_SUBCONSCIOUS_PROMPT),
        HumanMessage(content=packet),
    ]
    try:
        async with coordinator.claim(
            stream_kind="l1",
            invocation_state=invocation_state,
            deadline_monotonic=deadline_monotonic,
        ):
            response = await services.llm.ainvoke(
                messages,
                config=config_for_turn_deadline(
                    _bounded_stage_config(
                        sidecar_lane,
                        stage_name="L1",
                        completion_cap=_SIDECAR_COMPLETION_CAP,
                    ),
                    deadline_monotonic,
                ),
            )
    except _PROVIDER_EXCEPTIONS:
        return None, "sidecar_l1_unavailable"

    parsed = parse_llm_json_output(
        str(getattr(response, "content", "")),
        deterministic_only=True,
    )
    residue = try_validate_l1_residue(
        parsed,
        supplied_evidence_handles=supplied_evidence_handles,
    )
    if residue is None:
        return None, "sidecar_l1_malformed"
    validated_residue = dict(residue)
    return validated_residue, None


def _start_l1_sidecar(
    *,
    payload: Mapping[str, Any],
    projection: Any,
    services: CognitionChainServicesV3,
    coordinator: SidecarCoordinator | None,
    admissions: SidecarAdmissionLedger | None,
    invocation_state: SidecarInvocationState | None,
    deadline_monotonic: float | None,
) -> asyncio.Task[tuple[dict[str, Any] | None, str | None]] | None:
    """Start the one optional L1 task before the primary lane is claimed.

    Args:
        payload: Canonical input already validated at the public boundary.
        projection: Prompt-safe V2 state projection for this turn.
        services: Injected V3 service bundle.
        coordinator: The optional V3 sidecar coordinator.
        admissions: The optional invocation-local sidecar admission authority.
        invocation_state: The optional invocation-local sidecar task owner.

    Returns:
        The active L1 task when enabled, or ``None`` when L1 is unavailable.

    The caller yields once after registration so the advisory L1 provider
    request begins before the primary lane is claimed.  That yield does not
    await sidecar completion or alter its admission and deadline rules.
    """

    if (
        not services.subconscious_enabled
        or coordinator is None
        or admissions is None
        or invocation_state is None
    ):
        return None
    identity = projection.identity_by_question["goal_cognition"]
    boundaries = identity["boundaries"]
    affect_bands = projection.payload["affect"]
    if not isinstance(boundaries, Mapping) or not isinstance(
        affect_bands,
        list,
    ):
        raise CognitionExecutionError(
            "L1 requires prompt-safe boundary and affect projections",
            error_code="serial_identity_missing",
            stage="pre_state_commit",
            safe_checkpoint="pre_state_commit",
            retryable=False,
        )
    supplied_evidence_handles = [
        row["evidence_handle"]
        for row in payload["evidence"]
    ]
    packet = v3_prompt.build_l1_subconscious_packet(
        episode=payload["episode"],
        affect_bands=affect_bands,
        boundary_summary=boundaries,
        supplied_evidence_handles=supplied_evidence_handles,
    )
    try:
        check_turn_deadline(deadline_monotonic)
    except TurnDeadlineExceeded:
        return None
    admissions.reserve_l1()
    task = asyncio.create_task(
        _run_l1_sidecar(
            packet=packet,
            supplied_evidence_handles=supplied_evidence_handles,
            services=services,
            coordinator=coordinator,
            invocation_state=invocation_state,
            deadline_monotonic=deadline_monotonic,
        )
    )
    invocation_state.register_l1_task(task)
    return task


async def _yield_after_l1_sidecar_start(
    task: asyncio.Task[tuple[dict[str, Any] | None, str | None]] | None,
) -> None:
    """Yield one loop turn after a registered L1 task starts."""

    if task is not None:
        await asyncio.sleep(0)


def _take_ready_l1_residue(
    task: asyncio.Task[tuple[dict[str, Any] | None, str | None]] | None,
    warnings: list[str],
) -> tuple[dict[str, Any] | None, bool]:
    """Read an already-finished L1 task without waiting for sidecar work."""

    if task is None or not task.done():
        return None, False
    try:
        residue, warning = task.result()
    except asyncio.CancelledError:
        return None, True
    except Exception:  # noqa: BLE001 - optional advisory task boundary
        warnings.append("sidecar_l1_unavailable")
        return None, True
    if warning is not None:
        warnings.append(warning)
    return residue, True


async def _await_l1_task_for_drain(
    task: asyncio.Task[tuple[dict[str, Any] | None, str | None]],
) -> tuple[str | None, bool]:
    """Drain an L1 task while classifying child cancellation separately."""

    try:
        _residue, warning = await task
    except asyncio.CancelledError:
        return None, True
    except Exception:  # noqa: BLE001 - optional advisory task boundary
        return "sidecar_l1_unavailable", False
    return warning, False


async def _drain_l1_sidecar(
    task: asyncio.Task[tuple[dict[str, Any] | None, str | None]] | None,
    *,
    invocation_state: SidecarInvocationState | None,
    warnings: list[str],
) -> None:
    """Cancel and drain L1 during owned cleanup after primary work finishes."""

    if task is None:
        return
    current_task = asyncio.current_task()
    initial_cancelling = (
        current_task.cancelling()
        if current_task is not None
        else 0
    )
    if not task.done():
        if invocation_state is not None:
            invocation_state.record_cancellation(task)
        task.cancel()
        warnings.append("sidecar_l1_dropped")
    drain_task = asyncio.create_task(_await_l1_task_for_drain(task))
    outer_cancelled = False
    while True:
        try:
            warning, _ = await asyncio.shield(drain_task)
            break
        except asyncio.CancelledError:
            outer_cancelled = True
            continue
    if warning is not None:
        warnings.append(warning)
    if (
        outer_cancelled
        or (
            current_task is not None
            and current_task.cancelling() > initial_cancelling
        )
    ):
        raise asyncio.CancelledError


async def _repair_v3_serial_candidate(
    raw_output: str,
    candidate_id: str,
    attempt_coordinates: Mapping[str, object],
    *,
    services: CognitionChainServicesV3,
    coordinator: SidecarCoordinator,
    admissions: SidecarAdmissionLedger,
    invocation_state: SidecarInvocationState,
    warnings: list[str],
    deadline_monotonic: float | None,
) -> dict[str, object] | None:
    """Run one admitted chain repair and retain only its canonical object."""

    sidecar_lane = services.sidecar_lane
    if sidecar_lane is None:
        return None
    l1_preempted_before_repair = invocation_state.l1_preempted_by_repair
    repaired = await invoke_lane_scoped_json_repair(
        raw_output=raw_output,
        candidate_id=candidate_id,
        attempt_coordinates=attempt_coordinates,
        llm=services.llm,
        config=_bounded_stage_config(
            sidecar_lane,
            stage_name="json_repair",
            completion_cap=_json_repair_cap(attempt_coordinates),
        ),
        coordinator=coordinator,
        admissions=admissions,
        invocation_state=invocation_state,
        deadline_monotonic=deadline_monotonic,
    )
    l1_warning = invocation_state.consume_l1_warning()
    if l1_warning is not None:
        warnings.append(l1_warning)
    if (
        not l1_preempted_before_repair
        and invocation_state.l1_preempted_by_repair
    ):
        warnings.append("sidecar_l1_preempted_by_repair")
    if repaired is None:
        warnings.append("sidecar_json_repair_failed")
    return repaired


async def _invoke_v3_sidecar_authorizer(
    *,
    services: CognitionChainServicesV3,
    coordinator: SidecarCoordinator,
    admissions: SidecarAdmissionLedger,
    invocation_state: SidecarInvocationState,
    cycle_index: int,
    warnings: list[str],
    messages: list[object],
    candidate_handles: list[str],
    stage_name: str,
    output_state_fields: list[str],
    prompt_cap: int,
    runtime_capability_limits: Sequence[str],
    deadline_monotonic: float | None,
) -> dict[str, bool]:
    """Run X1 or X2 through the V3 sidecar with deny-all exhaustion.

    Args:
        services: Injected V3 model routes.
        coordinator: Serialized sidecar lane owner.
        admissions: Invocation-local X1/X2 and repair admission authority.
        invocation_state: Invocation-local sidecar task and diagnostics state.
        cycle_index: Canonical resolver cycle for the authorization attempt.
        warnings: Bounded output warnings populated on optional failure.
        messages: Fresh authorization packet assembled by the existing owner.
        candidate_handles: Exact candidate handles requiring boolean coverage.
        stage_name: Canonical V2 authorization producer stage.
        output_state_fields: Existing trace contract fields for this owner.
        prompt_cap: Existing bounded prompt ceiling for this owner.
        runtime_capability_limits: Trusted runtime limits for X1 repair text.

    Returns:
        Exact boolean coverage for every candidate, with deny-all on any
        provider, repair, or replacement exhaustion.
    """

    if stage_name not in {"action_authorization", "resolver_authorization"}:
        raise ValueError("V3 sidecar authorization stage is invalid")
    if not output_state_fields or prompt_cap <= 0:
        raise ValueError("V3 sidecar authorization trace contract is invalid")
    sidecar_lane = services.sidecar_lane
    if sidecar_lane is None:
        return {handle: False for handle in candidate_handles}

    stream_kind = (
        "action_authorization"
        if stage_name == "action_authorization"
        else "resolver_authorization"
    )
    current_messages = list(messages)
    base_messages = list(messages)
    base_prompt_chars = sum(
        len(str(message.content))
        for message in base_messages
    )
    if base_prompt_chars > prompt_cap:
        return {handle: False for handle in candidate_handles}

    for attempt_number in range(1, V2_MODEL_TOTAL_ATTEMPTS + 1):
        try:
            config_for_turn_deadline(sidecar_lane, deadline_monotonic)
        except _PROVIDER_EXCEPTIONS:
            warnings.append(f"sidecar_{stage_name}_deadline_exhausted")
            break
        try:
            coordinates = reserve_v2_model_attempt(
                stage=stage_name,
                branch_id=f"cycle:{cycle_index}",
                local_attempt=attempt_number,
            )
        except V2AttemptBudgetExhausted:
            warnings.append(f"sidecar_{stage_name}_attempt_exhausted")
            break

        if stage_name == "action_authorization":
            admissions.reserve_action_authorization(
                cycle_index=cycle_index,
                attempt_coordinates=coordinates,
            )
        else:
            admissions.reserve_resolver_authorization(
                cycle_index=cycle_index,
                attempt_coordinates=coordinates,
            )
        attempt_config = _bounded_stage_config(
            sidecar_lane,
            stage_name=(
                stage_name
                if attempt_number == 1
                else f"{stage_name}.repair"
            ),
            completion_cap=_SIDECAR_COMPLETION_CAP,
        )
        try:
            async with coordinator.claim(
                stream_kind=stream_kind,
                invocation_state=invocation_state,
                deadline_monotonic=deadline_monotonic,
            ):
                attempt_config = config_for_turn_deadline(
                    attempt_config,
                    deadline_monotonic,
                )
                response = await services.llm.ainvoke(
                    current_messages,
                    config=attempt_config,
                )
        except _PROVIDER_EXCEPTIONS:
            record_v2_attempt_disposition(
                coordinates,
                disposition=(
                    "exhausted"
                    if attempt_number == V2_MODEL_TOTAL_ATTEMPTS
                    else "regenerate"
                ),
            )
            warnings.append(f"sidecar_{stage_name}_unavailable")
            current_messages = list(base_messages)
            continue

        response_text = str(getattr(response, "content", ""))
        parsed = parse_llm_json_output(
            response_text,
            deterministic_only=True,
        )
        if not parsed:
            parsed = await _repair_v3_serial_candidate(
                response_text,
                f"{stage_name}:attempt:{attempt_number}",
                coordinates,
                services=services,
                coordinator=coordinator,
                admissions=admissions,
                invocation_state=invocation_state,
                warnings=warnings,
                deadline_monotonic=deadline_monotonic,
            ) or {}
        try:
            decisions = v3_authorization.validate_authorization_decisions(
                parsed,
                candidate_handles=candidate_handles,
            )
        except (TypeError, ValueError) as exc:
            record_v2_attempt_disposition(
                coordinates,
                disposition=(
                    "exhausted"
                    if attempt_number == V2_MODEL_TOTAL_ATTEMPTS
                    else "regenerate"
                ),
            )
            warnings.append(f"sidecar_{stage_name}_malformed")
            if attempt_number == V2_MODEL_TOTAL_ATTEMPTS:
                break
            repair_message = _authorization_repair_message(
                response_text=response_text,
                contract_error=str(exc),
                candidate_handles=candidate_handles,
                runtime_capability_limits=runtime_capability_limits,
            )
            if (
                base_prompt_chars + len(str(repair_message.content))
                > prompt_cap
            ):
                break
            current_messages = [*base_messages, repair_message]
            continue

        record_v2_attempt_disposition(
            coordinates,
            disposition=(
                "accepted" if attempt_number == 1 else "recovered"
            ),
        )
        if stage_name == "action_authorization":
            admissions.finish_action_authorization(cycle_index=cycle_index)
        else:
            admissions.finish_resolver_authorization(cycle_index=cycle_index)
        return decisions

    if stage_name == "action_authorization":
        admissions.finish_action_authorization(cycle_index=cycle_index)
    else:
        admissions.finish_resolver_authorization(cycle_index=cycle_index)
    denied = {handle: False for handle in candidate_handles}
    return denied


async def _authorize_serial_action_decision(
    *,
    action_decision: Mapping[str, Any] | None,
    primary_bid: ActionBidV2 | None,
    bid_handles: Mapping[str, ActionBidV2],
    action_handles: Mapping[str, Mapping[str, Any]],
    resolver_handles: Mapping[str, Mapping[str, Any]],
    payload: Mapping[str, Any],
    services: CognitionChainServicesV3,
    coordinator: SidecarCoordinator | None,
    admissions: SidecarAdmissionLedger | None,
    invocation_state: SidecarInvocationState | None,
    warnings: list[str],
    deadline_monotonic: float | None,
) -> dict[str, Any]:
    """Authorize one P1 proposal set without changing planner semantics.

    P1 remains the sole semantic producer of action and resolver proposals.
    This function applies the existing deterministic stance and answerability
    rules, runs X1 before X2 when a sidecar exists, and materializes only rows
    with exact positive boolean authorizations.
    """

    decision = {
        "action_requests": [],
        "resolver_requests": [],
        "goal_resolution": "blocked",
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
    }
    if action_decision is not None:
        decision.update(action_decision)
    if primary_bid is None:
        if decision["action_requests"] or decision["resolver_requests"]:
            warnings.append("sidecar_authorization_denied")
        return decision

    decision, suppressed = apply_stance_suppression(decision, primary_bid)
    cycle_index = payload.get("resolver_cycle_index", 0)
    if not isinstance(cycle_index, int) or isinstance(cycle_index, bool):
        raise CognitionExecutionError(
            "resolver cycle index is invalid for sidecar authorization",
            error_code="serial_identity_missing",
            stage="action_planning",
            safe_checkpoint="final_reduction",
            retryable=False,
        )
    authorizer = None
    if (
        coordinator is not None
        and admissions is not None
        and invocation_state is not None
    ):
        authorizer = partial(
            _invoke_v3_sidecar_authorizer,
            services=services,
            coordinator=coordinator,
            admissions=admissions,
            invocation_state=invocation_state,
            cycle_index=cycle_index,
            warnings=warnings,
            deadline_monotonic=deadline_monotonic,
        )

    if suppressed or not decision["action_requests"]:
        authorized_action_rows: list[dict[str, Any]] = []
        if admissions is not None:
            admissions.finish_action_authorization(cycle_index=cycle_index)
    elif authorizer is None:
        authorized_action_rows = []
        warnings.append("sidecar_authorization_denied")
    else:
        authorized_action_rows = await authorize_action_requests(
            action_requests=decision["action_requests"],
            bid_handles=bid_handles,
            evidence=payload["evidence"],
            action_handles=action_handles,
            runtime_capability_limits=payload.get(
                "runtime_capability_limits",
                [],
            ),
            authorization_executor=authorizer,
        )
    action_requests = _materialize_action_requests(
        authorized_action_rows,
        bid_handles,
        action_handles,
    )

    if (
        suppressed
        or decision["goal_resolution"] == "answerable_now"
        or not decision["resolver_requests"]
    ):
        authorized_resolver_rows: list[dict[str, Any]] = []
        if admissions is not None:
            admissions.finish_resolver_authorization(cycle_index=cycle_index)
    elif authorizer is None:
        authorized_resolver_rows = []
        warnings.append("sidecar_authorization_denied")
    else:
        authorized_resolver_rows = await authorize_resolver_requests(
            resolver_requests=decision["resolver_requests"],
            bid_handles=bid_handles,
            evidence=payload["evidence"],
            resolver_handles=resolver_handles,
            resolver_context=payload["resolver_context"],
            authorization_executor=authorizer,
        )

    source_message_id = ""
    origin_metadata = payload["episode"].get("origin_metadata")
    if isinstance(origin_metadata, Mapping):
        platform_message_id = origin_metadata.get("platform_message_id")
        if isinstance(platform_message_id, str):
            source_message_id = platform_message_id
    goal_continuation_ref = build_goal_continuation_ref(
        source_episode_id=payload["episode"]["episode_id"],
        source_message_id=source_message_id,
        branch_id=primary_bid["branch_id"],
        goal_ref=primary_bid["goal_ref"],
    )
    materialized_resolver_rows = _materialize_resolver_requests(
        authorized_resolver_rows,
        bid_handles,
        resolver_handles,
        goal_continuation_ref,
    )
    resolver_requests, goal_resolution = settle_resolver_outcome(
        decision,
        suppressed=suppressed,
        action_requests_materialized=len(action_requests),
        resolver_requests_materialized=materialized_resolver_rows,
    )
    action_owner_denied = (
        bool(decision["action_requests"]) and not action_requests
    )
    resolver_owner_denied = (
        bool(decision["resolver_requests"]) and not resolver_requests
    )
    result = dict(decision)
    result["action_requests"] = action_requests
    result["resolver_requests"] = list(resolver_requests)
    result["goal_resolution"] = goal_resolution
    if (
        (action_owner_denied or resolver_owner_denied)
        and goal_resolution != "answerable_now"
    ):
        result["resolver_goal_progress"] = None
    return result


def _validate_active_goal_group(
    parsed: dict[str, object],
    *,
    roster: Sequence[Mapping[str, object]],
    evidence_handles: set[str],
    role_handles: set[str],
) -> object:
    """Validate one ordered active-branch group against its fixed roster."""

    return validate_active_goal_group_output(
        parsed,
        branch_roster=roster,
        evidence_handles=evidence_handles,
        role_handles=role_handles,
    )


def _salvage_exhausted_active_group(
    *,
    raw_output: str | None,
    question: v3_prompt.ChainQuestion,
    roster: Sequence[Mapping[str, object]],
    evidence_handles: set[str],
    role_handles: set[str],
    harness: SerialChainHarness,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Retain only independently valid siblings after structural exhaustion."""

    parsed = parse_llm_json_output(
        raw_output or "",
        deterministic_only=True,
    )
    rows, failures = salvage_active_goal_group_output(
        parsed,
        branch_roster=roster,
        evidence_handles=evidence_handles,
        role_handles=role_handles,
    )
    for row in rows:
        record_v2_branch_disposition(
            branch_id=str(row["branch_id"]),
            disposition="recovered_by_sibling",
        )
    for branch_id, error_code in failures.items():
        record_v2_branch_disposition(
            branch_id=branch_id,
            disposition="unrecoverable",
            error_code=error_code,
        )
    if rows:
        accepted_projection = json.dumps(
            {"bids": rows},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        harness.append_question(v3_prompt.build_question_message(question))
        harness.accept_answer(
            accepted_projection,
            {
                "question": question.contract_name,
                "typed_product": {"bids": rows},
            },
        )
    return rows, failures


def _reduce_serial_appraisals(
    *,
    reduction_state: Mapping[str, Any],
    appraisal_rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    projection: Any,
    updated_at: str,
    elapsed_seconds: int,
    reducer_relationship_context: Mapping[str, Any] | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    list[dict[str, Any]],
]:
    """Reduce accepted appraisal rows through the canonical V3 owners."""

    (
        final_state,
        appraisal_results,
        reduction_failures,
        comparison_results,
        accepted_relationship_deltas,
    ) = _reduce_appraisals_with_isolation(
        reduction_state,
        appraisal_rows,
        payload["evidence"],
        projection.handle_to_ref,
        updated_at=updated_at,
        character_constraints=payload["character_constraints"],
        relationship_context=reducer_relationship_context,
    )
    final_state = _apply_final_relationship_maintenance(
        final_state,
        episode=payload["episode"],
        elapsed_seconds=elapsed_seconds,
        accepted_relationship_deltas=accepted_relationship_deltas,
        direct_facts=payload["direct_facts"],
    )
    if final_state["state_scope"] == "user":
        final_state = apply_state_update(
            final_state,
            elapsed_seconds=0,
            updated_at=updated_at,
            character_constraints=payload["character_constraints"],
            relationship_context=reducer_relationship_context,
        )
        final_state = create_deterministic_goals(
            final_state,
            character_constraints=payload["character_constraints"],
            relationship_context=reducer_relationship_context,
            evidence=payload["evidence"],
            updated_at=updated_at,
            reconcile_salience_gated_goals=True,
        )
        final_state = validate_cognition_state(final_state)
    return (
        final_state,
        appraisal_results,
        reduction_failures,
        comparison_results,
        accepted_relationship_deltas,
    )


def _build_serial_output(
    *,
    payload: Mapping[str, Any],
    previous_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
    questions: Sequence[Mapping[str, Any]],
    appraisal_results: Mapping[str, Any],
    appraisal_failures: Mapping[str, str],
    comparison_results: Mapping[str, Any],
    preliminary_branches: Sequence[BranchDefinition],
    all_bids: Sequence[ActionBidV2],
    ordinary_bid: ActionBidV2 | None,
    workspace_collapse: Mapping[str, Any] | None,
    action_decision: Mapping[str, Any] | None,
    branch_failures: Mapping[str, str] | None,
    harness: SerialChainHarness,
    warnings: list[str],
    stage_status: dict[str, str],
    started_at: float,
) -> CognitionCoreOutputV2:
    """Project one validated serial-chain terminal result onto V2 output."""

    eligible_bids, stale_branch_ids = _bids_with_live_goals(
        list(all_bids) if all_bids else (
            [ordinary_bid] if ordinary_bid is not None else []
        ),
        final_state,
    )
    warnings.extend(
        f"stale_goal_bid_dropped:{branch_id}"
        for branch_id in stale_branch_ids
    )
    if not eligible_bids:
        raise CognitionExecutionError(
            "ordinary_response",
            error_code="ordinary_response_unavailable",
            stage="branch_cognition",
            safe_checkpoint="final_reduction",
        )
    relational_decision = _ordinary_relational_decision(eligible_bids)
    collapse = (
        collapse_authoritative_relational_bid(
            eligible_bids,
            relational_decision,
        )
        if relational_decision is not None
        and relational_decision["applicability"] == "relationship_sensitive"
        else workspace_collapse
        if workspace_collapse is not None
        else collapse_single_bid(eligible_bids[0])
        if len(eligible_bids) == 1
        else fallback_partition_envelope(eligible_bids)
    )
    if (
        relational_decision is not None
        and relational_decision["applicability"] == "relationship_sensitive"
    ):
        warnings.append("authoritative_relational_willingness")
    primary_bid = collapse.get("primary_bid")
    supporting_bids = collapse.get("supporting_bids", [])
    stage_status["branch_cognition"] = "completed"
    stage_status["workspace_collapse"] = "completed"

    if action_decision is None:
        action_decision = {
            "action_requests": [],
            "resolver_requests": [],
            "goal_resolution": "blocked",
            "resolver_pending_resolution": None,
            "resolver_goal_progress": None,
        }
    stage_status["action_planning"] = "completed"

    episode_operation = project_dialog_response_operation(payload["episode"])
    if (
        primary_bid is None
        and episode_operation is not None
        and episode_operation["selection_required"]
    ):
        raise CognitionExecutionError(
            "required selection has no admitted bid",
            error_code="required_selection_without_admitted_bid",
            stage="workspace_collapse",
            safe_checkpoint="final_reduction",
        )

    if primary_bid is not None:
        source_message_id = ""
        origin_metadata = payload["episode"].get("origin_metadata")
        if isinstance(origin_metadata, Mapping):
            platform_message_id = origin_metadata.get("platform_message_id")
            if isinstance(platform_message_id, str):
                source_message_id = platform_message_id
        goal_continuation_ref = build_goal_continuation_ref(
            source_episode_id=payload["episode"]["episode_id"],
            source_message_id=source_message_id,
            branch_id=primary_bid["branch_id"],
            goal_ref=primary_bid["goal_ref"],
        )
        self_cognition_response = action_decision.get(
            "self_cognition_response"
        )
        intention: dict[str, Any] = {
            "selected_branch_id": primary_bid["branch_id"],
            "route": derive_action_route(
                episode=payload["episode"],
                primary_bid=primary_bid,
                action_requests=action_decision["action_requests"],
                resolver_requests=action_decision["resolver_requests"],
                self_cognition_response=(
                    self_cognition_response
                    if isinstance(self_cognition_response, Mapping)
                    else None
                ),
                goal_resolution=action_decision["goal_resolution"],
                goal_continuation_ref=goal_continuation_ref,
                required_resolver_evidence_dependency=payload.get(
                    "required_resolver_evidence_dependency"
                ),
            ),
            "intention": primary_bid["intention"],
            "target_roles": list(primary_bid["target_roles"]),
            "reason": primary_bid["reason"],
            "goal_continuation_ref": goal_continuation_ref,
        }
        if "selected_response_operation" in primary_bid:
            intention["selected_response_operation"] = dict(
                primary_bid["selected_response_operation"]
            )
    else:
        intention = {
            "selected_branch_id": "",
            "route": "silence",
            "intention": "",
            "target_roles": [],
            "reason": "",
            "goal_continuation_ref": {
                "schema_version": "goal_continuation_ref.v1",
                "source_episode_id": payload["episode"]["episode_id"],
                "source_message_id": "",
                "branch_id": "",
                "goal_ref": {
                    "scope": final_state["state_scope"],
                    "kind": "goal",
                    "entity_id": "",
                },
            },
        }

    admitted_bid = _selected_bid(intention, primary_bid, supporting_bids)
    affect = project_affect(final_state["affect_activations"], final_state)
    relationship = project_relationship(final_state.get("relationship"))
    expression_policy = default_expression_policy(
        intention["route"],
        affect,
        selected_branch_id=intention.get("selected_branch_id"),
        activations=final_state["affect_activations"],
    )
    selected_bid_reason = (
        admitted_bid["reason"]
        if admitted_bid is not None
        else "没有有依据的目标候选"
    )
    frozen_preliminary_branches = [
        replace(definition, dependencies=(), dependency_options=())
        for definition in preliminary_branches
    ]
    preliminary_bids_by_kind = {
        definition.goal_kind: bid
        for definition in frozen_preliminary_branches
        for bid in eligible_bids
        if bid["branch_id"] == definition.branch_id
    }
    preliminary_execution = _synthesize_branch_execution(
        frozen_preliminary_branches,
        preliminary_bids_by_kind,
        {
            definition.goal_kind: (
                (branch_failures or {}).get(
                    definition.branch_id,
                    "goal_bid_unavailable",
                )
            )
            for definition in frozen_preliminary_branches
            if definition.goal_kind not in preliminary_bids_by_kind
        },
        harness.ledger,
    )
    cognition_observability = _build_cognition_observability(
        questions=questions,
        appraisal_results=appraisal_results,
        appraisal_failures=appraisal_failures,
        preliminary_branches=frozen_preliminary_branches,
        preliminary_execution=preliminary_execution,
        final_branches=[],
        final_execution=None,
        collapse=collapse,
        selected_bid_reason=selected_bid_reason,
        diagnostics={
            "selected_question_count": len(questions),
            "dispatched_question_count": len(questions),
            "selected_branch_count": len(frozen_preliminary_branches),
            "dispatched_branch_count": len(frozen_preliminary_branches),
            "completed_branch_count": len(preliminary_bids_by_kind),
            "failed_branch_count": (
                len(frozen_preliminary_branches)
                - len(preliminary_bids_by_kind)
            ),
            "overlap_ms": 0,
            "dependency_wait_ms": 0,
            "total_ms": _elapsed_ms(started_at),
        },
        relational_willingness=relational_decision,
    )
    diagnostics = {
        "run_id": str(payload["episode"].get("episode_id", "episode")),
        "stage_status": stage_status,
        "selected_question_count": len(questions),
        "dispatched_question_count": len(questions),
        "selected_branch_count": 1,
        "dispatched_branch_count": 1,
        "completed_branch_count": 1,
        "failed_branch_count": 0,
        "overlap_ms": 0,
        "dependency_wait_ms": 0,
        "total_ms": _elapsed_ms(started_at),
        "warnings": _deduplicate_diagnostics_warnings(warnings),
    }
    output: dict[str, Any] = {
        "schema_version": "cognition_core_output.v2",
        "intention": intention,
        "goal_continuation_ref": intention["goal_continuation_ref"],
        "supporting_bids": [],
        "state_update": build_state_update(
            previous_state,
            final_state,
            comparison_results,
        ),
        "affect_projection": affect,
        "action_requests": action_decision["action_requests"],
        "resolver_requests": action_decision["resolver_requests"],
        "goal_resolution": action_decision["goal_resolution"],
        "resolver_pending_resolution": _bind_pending_resolution(
            action_decision["resolver_pending_resolution"],
            payload.get("pending_resolver_resume"),
        ),
        "resolver_goal_progress": action_decision["resolver_goal_progress"],
        "resolver_progress": _resolver_progress(
            action_decision["resolver_requests"]
        ),
        "selected_bid_reason": selected_bid_reason,
        "private_monologue": (
            admitted_bid["private_monologue"]
            if admitted_bid is not None
            else "当前角色没有有依据的行动理由。"
        ),
        "expression_policy": expression_policy,
        "diagnostics": diagnostics,
        "cognition_observability": cognition_observability,
    }
    if admitted_bid is not None:
        output["admitted_bid"] = admitted_bid
    if relationship is not None:
        output["relationship_projection"] = relationship
    if relational_decision is not None:
        output["relational_willingness"] = dict(relational_decision)
    if is_targetless_group_self_cognition_episode(payload["episode"]):
        response_status = action_decision.get(
            "self_cognition_response_contract_status"
        )
        if isinstance(response_status, str):
            output["self_cognition_response_contract_status"] = response_status
        response = action_decision.get("self_cognition_response")
        if isinstance(response, Mapping):
            output["self_cognition_response"] = dict(response)
    return validate_cognition_core_output(output)


def _session_ttl_seconds(services: CognitionChainServicesV3) -> float:
    """Return the exact resolver-capability plus turn-deadline TTL."""

    return (
        COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS
        * COGNITION_RESOLVER_MAX_CYCLES
        + services.turn_deadline_seconds
        + 30
    )


def _expected_relational_willingness_carrier(
    payload: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the exact recurrence carrier stored beside one cold output."""

    current_carrier = payload.get("current_turn_relational_willingness")
    if isinstance(current_carrier, Mapping):
        return dict(current_carrier)
    decision = output.get("relational_willingness")
    episode_id = payload["episode"].get("episode_id")
    if not isinstance(decision, Mapping) or not isinstance(episode_id, str):
        return None
    return {
        "schema_version": CURRENT_TURN_RELATIONAL_WILLINGNESS_VERSION,
        "episode_id": episode_id,
        "branch_id": ORDINARY_GOAL_KIND,
        "decision": dict(decision),
    }


def _self_cognition_response_context(
    payload: Mapping[str, Any],
    current_episode_evidence_handles: Sequence[str],
) -> dict[str, Any] | None:
    """Build the required P1 contract carrier for targetless group turns."""

    if not is_targetless_group_self_cognition_episode(payload["episode"]):
        return None
    decision_order = ("stay_silent", "propose_visible_reply")
    participation_order = (
        "direct_address",
        "explicit_character_reference",
        "grounded_scene_intervention",
    )
    return {
        "required_fields": list(
            v3_prompt.SELF_COGNITION_RESPONSE_REQUIRED_FIELDS
        ),
        "allowed_decisions": [
            value
            for value in decision_order
            if value in SELF_COGNITION_RESPONSE_DECISION_VALUES
        ],
        "allowed_evidence_handles": sorted(
            current_episode_evidence_handles
        ),
        "allowed_semantic_target_handles": _self_cognition_target_handles(
            payload["scene_context"]
        ),
        "allowed_participation_basis_values": [
            value
            for value in participation_order
            if value in SELF_COGNITION_RESPONSE_PARTICIPATION_VALUES
        ],
        "response_goal_max_chars": SELF_COGNITION_RESPONSE_TEXT_LIMIT,
        "reason_max_chars": SELF_COGNITION_RESPONSE_TEXT_LIMIT,
    }


def _rehydrate_serial_harness(
    *,
    session: ChainSessionV1,
    services: CognitionChainServicesV3,
    deadline_monotonic: float | None,
) -> SerialChainHarness:
    """Rehydrate only the exact accepted transcript and session products."""

    if len(session.accepted_messages) != len(session.accepted_products) * 2:
        raise SessionContractError(
            "session transcript and accepted products are inconsistent"
        )
    messages: list[ChainMessageV1] = []
    for index, message in enumerate(session.accepted_messages):
        if (
            not isinstance(message, tuple)
            or len(message) != 2
            or message[0] not in {"human", "assistant"}
            or not isinstance(message[1], str)
            or not message[1].strip()
        ):
            raise SessionContractError("session transcript row is invalid")
        expected_role = "human" if index % 2 == 0 else "assistant"
        if message[0] != expected_role:
            raise SessionContractError("session transcript order is invalid")
        messages.append(ChainMessageV1(role=message[0], content=message[1]))
    if any(
        not isinstance(product, Mapping)
        for product in session.accepted_products
    ):
        raise SessionContractError("session product row is invalid")
    transcript = ChainTranscriptV1(
        messages=tuple(messages),
        accepted_products=tuple(session.accepted_products),
        attempt_ledger=dict(session.attempt_ledger),
        token_ledger=dict(session.token_ledger),
        reanchor_used=session.reanchor_used,
        deadline_monotonic=deadline_monotonic,
    )
    stored_extension_used = bool(
        session.token_ledger.get("extension_used")
    )
    stored_active_ceiling = session.token_ledger.get(
        "active_total_ceiling_tokens",
        65_000 if stored_extension_used else 50_000,
    )
    if stored_active_ceiling not in {50_000, 65_000}:
        stored_active_ceiling = 50_000
    return SerialChainHarness(
        transcript=transcript,
        ledger=AttemptLedger(
            {
                "serial_appraisal": V2_APPRAISAL_TOTAL_ATTEMPTS,
                "serial_goal_ordinary": V2_MODEL_TOTAL_ATTEMPTS,
                "serial_action_plan": V2_MODEL_TOTAL_ATTEMPTS,
            },
            _counts=dict(session.attempt_ledger),
        ),
        budget=ContextBudgetLedger(
            ContextBudgetPlan(
                serving_window_tokens=services.chain_lane.context_window_tokens,
            ),
            active_total_ceiling_tokens=stored_active_ceiling,
            extension_used=stored_extension_used,
            reanchor_used=session.reanchor_used,
        ),
    )


def _collapse_for_action_plan(
    *,
    all_bids: Sequence[ActionBidV2],
    final_state: Mapping[str, Any],
    workspace_collapse: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[ActionBidV2]]:
    """Build the deterministic pre-P1 collapse from revised live bids."""

    eligible_bids, _stale_branch_ids = _bids_with_live_goals(
        all_bids,
        final_state,
    )
    if not eligible_bids:
        raise CognitionExecutionError(
            "ordinary_response",
            error_code="ordinary_response_unavailable",
            stage="branch_cognition",
            safe_checkpoint="final_reduction",
        )
    relational_decision = _ordinary_relational_decision(eligible_bids)
    collapse = (
        collapse_authoritative_relational_bid(
            eligible_bids,
            relational_decision,
        )
        if relational_decision is not None
        and relational_decision["applicability"] == "relationship_sensitive"
        else dict(workspace_collapse)
        if workspace_collapse is not None
        else collapse_single_bid(eligible_bids[0])
        if len(eligible_bids) == 1
        else fallback_partition_envelope(eligible_bids)
    )
    return collapse, eligible_bids


async def _run_appraisal_stages(
    *,
    questions_by_family: Mapping[str, Mapping[str, Any]],
    evidence_handles: Sequence[str],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    harness: SerialChainHarness,
    system_content: str,
    services: CognitionChainServicesV3,
    warnings: list[str],
    observation_context: Mapping[str, object] | None,
    relation_context: Mapping[str, object],
    l1_residue: Mapping[str, Any] | None,
    l1_observed: bool,
    interludes: Sequence[Mapping[str, object]],
    attempt_owner: str,
    branch_prefix: str,
    stage_prefix: str,
    deterministic_only: bool,
    json_repair_callback: Any,
    deadline_monotonic: float | None,
    role_assignment_handles_by_evidence: Mapping[
        str, Sequence[str]
    ] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str], bool, bool, bool]:
    """Run fixed A1/A2 appraisal requests with direct family recovery."""

    appraisal_rows: list[dict[str, Any]] = []
    appraisal_failures: dict[str, str] = {}
    context_limited = False
    first_carriers_pending = (
        observation_context is not None
        or l1_residue is not None
        or bool(interludes)
    )
    effective_role_domains = role_assignment_handles_by_evidence
    if role_assignment_handles_by_evidence is not None:
        mutable_role_domains = {
            evidence_handle: set(handles)
            for evidence_handle, handles in (
                role_assignment_handles_by_evidence.items()
            )
        }
        for question in questions_by_family.values():
            non_identity_handles = {
                handle
                for handle in question.get(
                    "permitted_role_assignment_handles",
                    (),
                )
                if handle not in {"self", "current_user"}
                and handle_to_ref.get(handle, {}).get("kind") in {
                    "goal",
                    "relationship",
                    "standard",
                    "object",
                }
            }
            for evidence_handle in question.get("evidence_handles", ()):
                mutable_role_domains.setdefault(evidence_handle, set()).update(
                    non_identity_handles
                )
        effective_role_domains = {
            evidence_handle: sorted(handles)
            for evidence_handle, handles in mutable_role_domains.items()
        }

    def question_with_role_domains(
        question: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Attach one prompt-safe evidence role domain to a family row."""

        enriched_question = dict(question)
        if effective_role_domains is not None:
            enriched_question[
                "permitted_role_assignment_handles_by_evidence"
            ] = {
                evidence_handle: list(
                    effective_role_domains.get(evidence_handle, ())
                )
                for evidence_handle in question.get("evidence_handles", ())
            }
        return enriched_question

    def record_family_failure(
        family: str,
        disposition: str,
    ) -> None:
        """Retain one typed omission and its bounded warning."""

        appraisal_failures.setdefault(
            f"q:{family}",
            APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
        )
        warnings.append(
            f"semantic_appraisal_family_exhausted:{disposition}:{family}"
        )

    for stage_name, registered_families in APPRAISAL_STAGE_FAMILIES:
        planned_families = tuple(
            family
            for family in registered_families
            if family in questions_by_family
        )
        if not planned_families:
            continue

        try:
            check_turn_deadline(deadline_monotonic)
        except TurnDeadlineExceeded:
            for family in planned_families:
                record_family_failure(family, "deadline")
            continue

        stage_question_kwargs = {
            "planned_questions": [
                question_with_role_domains(questions_by_family[family])
                for family in planned_families
            ],
            "stage_name": stage_name,
            "l1_residue": l1_residue if first_carriers_pending else None,
        }
        if stage_name == "A2":
            stage_question_kwargs["relation_context"] = relation_context
        stage_question = v3_prompt.build_appraisal_stage_question(
            **stage_question_kwargs
        )
        grouped_branch_ids = tuple(
            f"{branch_prefix}_{stage_name}_{family}"
            for family in planned_families
        )
        grouped_interludes = interludes if first_carriers_pending else ()
        grouped_observation_context = (
            observation_context if first_carriers_pending else None
        )
        recovery_observation_pending = grouped_observation_context is not None
        recovery_l1_pending = first_carriers_pending and l1_residue is not None
        recovery_l1_residue = l1_residue if recovery_l1_pending else None

        def stage_validator(
            parsed: dict[str, object],
            *,
            stage_families: tuple[str, ...] = planned_families,
        ) -> object:
            return reduce_appraisal_stage_output(
                parsed,
                planned_families=stage_families,
                questions_by_family=questions_by_family,
                evidence_handles=evidence_handles,
                handle_to_ref=handle_to_ref,
                role_assignment_handles_by_evidence=(
                    effective_role_domains
                ),
            )

        grouped_config = _bounded_stage_config(
            services.chain_lane,
            stage_name=f"{stage_prefix}{stage_name}",
            completion_cap=_APPRAISAL_COMPLETION_CAP,
        )
        try:
            grouped_result = await invoke_serial_question_with_repair(
                harness=harness,
                system_content=system_content,
                llm=services.llm,
                config=grouped_config,
                question=stage_question,
                validator=stage_validator,
                attempt_limit=1,
                observation_context=grouped_observation_context,
                interludes=grouped_interludes,
                attempt_owner=attempt_owner,
                v2_stage="semantic_appraisal",
                v2_branch_ids=grouped_branch_ids,
                deterministic_only=deterministic_only,
                json_repair_callback=json_repair_callback,
                deadline_monotonic=deadline_monotonic,
            )
        except CognitionContextLimitError:
            context_limited = True
            warnings.append("semantic_appraisal_context_limit")
            for family in planned_families:
                record_family_failure(family, "context")
            continue

        if grouped_result.validated is not None:
            if not isinstance(grouped_result.validated, list):
                raise CognitionExecutionError(
                    "appraisal stage validator returned a non-list",
                    error_code="semantic_appraisal_result_invalid",
                    stage="semantic_appraisal",
                    safe_checkpoint="pre_state_commit",
                )
            appraisal_rows.extend(grouped_result.validated)
            first_carriers_pending = False
            continue

        grouped_disposition = grouped_result.disposition.kind
        if grouped_disposition not in {
            "structural_exhausted",
            "provider_exhausted",
        }:
            for family in planned_families:
                record_family_failure(family, grouped_disposition)
            continue

        for family in planned_families:
            try:
                check_turn_deadline(deadline_monotonic)
            except TurnDeadlineExceeded:
                record_family_failure(family, "deadline")
                continue
            singleton_question_kwargs = {
                "planned_questions": [
                    question_with_role_domains(questions_by_family[family])
                ],
                "stage_name": stage_name,
                "l1_residue": (
                    recovery_l1_residue if recovery_l1_pending else None
                ),
            }
            if stage_name == "A2":
                singleton_question_kwargs["relation_context"] = relation_context
            singleton_question = v3_prompt.build_appraisal_stage_question(
                **singleton_question_kwargs
            )
            singleton_branch_id = f"{branch_prefix}_{stage_name}_{family}"

            def singleton_validator(
                parsed: dict[str, object],
                *,
                selected_family: str = family,
            ) -> object:
                return reduce_appraisal_stage_output(
                    parsed,
                    planned_families=(selected_family,),
                    questions_by_family=questions_by_family,
                    evidence_handles=evidence_handles,
                    handle_to_ref=handle_to_ref,
                    role_assignment_handles_by_evidence=(
                        effective_role_domains
                    ),
                )

            singleton_config = _bounded_stage_config(
                services.chain_lane,
                stage_name=f"{stage_prefix}{stage_name}.{family}",
                completion_cap=_APPRAISAL_COMPLETION_CAP,
            )
            try:
                singleton_result = await invoke_serial_question_with_repair(
                    harness=harness,
                    system_content=system_content,
                    llm=services.llm,
                    config=singleton_config,
                    question=singleton_question,
                    validator=singleton_validator,
                    attempt_limit=1,
                    observation_context=(
                        grouped_observation_context
                        if recovery_observation_pending
                        else None
                    ),
                    interludes=(
                        grouped_interludes
                        if recovery_observation_pending or recovery_l1_pending
                        else ()
                    ),
                    attempt_owner=attempt_owner,
                    v2_stage="semantic_appraisal",
                    v2_branch_ids=(singleton_branch_id,),
                    v2_local_attempt_start=2,
                    deterministic_only=deterministic_only,
                    json_repair_callback=json_repair_callback,
                    deadline_monotonic=deadline_monotonic,
                )
            except CognitionContextLimitError:
                context_limited = True
                record_family_failure(family, "context")
                continue
            if singleton_result.validated is not None:
                if not isinstance(singleton_result.validated, list):
                    raise CognitionExecutionError(
                        "appraisal family validator returned a non-list",
                        error_code="semantic_appraisal_result_invalid",
                        stage="semantic_appraisal",
                        safe_checkpoint="pre_state_commit",
                    )
                appraisal_rows.extend(singleton_result.validated)
                first_carriers_pending = False
                recovery_observation_pending = False
                recovery_l1_pending = False
            else:
                record_family_failure(family, singleton_result.disposition.kind)

    accepted_families = {
        str(row["question_id"]).removeprefix("q:")
        for row in appraisal_rows
    }
    for family in APPRAISAL_FAMILY_ORDER:
        if family in questions_by_family and family not in accepted_families:
            appraisal_failures.setdefault(
                f"q:{family}",
                APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
            )
    return (
        appraisal_rows,
        appraisal_failures,
        context_limited,
        l1_observed,
        first_carriers_pending,
    )


async def _run_reattached_resolver_tail(
    *,
    payload: Mapping[str, Any],
    services: CognitionChainServicesV3,
    session: ChainSessionV1,
    session_registry: ChainSessionRegistry,
    started_at: float,
    l1_task: asyncio.Task[tuple[dict[str, Any] | None, str | None]] | None,
    sidecar_coordinator: SidecarCoordinator | None,
    sidecar_admissions: SidecarAdmissionLedger | None,
    sidecar_state: SidecarInvocationState | None,
    deadline_monotonic: float | None,
    initial_warnings: Sequence[str] = (),
    primary_queue_wait_ms: int = 0,
    primary_in_flight_at_start: int = 0,
) -> CognitionCoreOutputV2:
    """Run the bounded R-tail from one exact accepted chain-session prefix."""

    previous_state = validate_cognition_state(payload["mutable_state"])
    updated_at = _episode_updated_at(payload["episode"])
    elapsed_seconds = _cognition_elapsed_seconds(previous_state, updated_at)
    reducer_relationship_context = _native_relationship_context(
        payload.get("relationship_context"),
    )
    new_observation = payload["evidence"][-1]
    projection = project_state_for_prompt(
        previous_state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        relationship_context=payload.get("relationship_context"),
        character_operational_context=payload.get(
            "character_operational_context",
        ),
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
    )
    recurrence_observation_context = v3_prompt.build_observation_context(
        projection_payload=projection.payload,
        scene_context=payload["scene_context"],
        episode=payload["episode"],
        evidence=payload["evidence"],
        direct_facts=payload["direct_facts"],
        handle_to_ref=projection.handle_to_ref,
    )
    dialogue_role_bindings = _goal_dialogue_role_bindings(
        recurrence_observation_context,
    )
    appraisal_role_domains = (
        _appraisal_role_assignment_handles_by_evidence(
            recurrence_observation_context,
            payload["evidence"],
        )
    )
    questions = plan_semantic_questions(
        [new_observation],
        previous_state,
        projection.handle_to_ref,
    )
    questions_by_family = {
        question["question_kind"]: question
        for question in questions
    }
    identity = projection.identity_by_question.get("goal_cognition")
    if not isinstance(identity, Mapping):
        raise SessionContractError(
            "session recurrence requires canonical goal-cognition identity"
        )
    if not isinstance(session.last_output, Mapping):
        raise SessionContractError("session recurrence has no prior output")
    relational_carrier = _expected_relational_willingness_carrier(
        payload,
        session.last_output,
    )
    if not isinstance(relational_carrier, Mapping):
        raise SessionContractError(
            "session recurrence has no prior relational carrier"
        )
    carried_relational_willingness = relational_carrier.get("decision")
    if not isinstance(carried_relational_willingness, Mapping):
        raise SessionContractError(
            "session recurrence has no prior relational willingness"
        )
    system_content = v3_anchor.build_system_head(identity)
    harness = _rehydrate_serial_harness(
        session=session,
        services=services,
        deadline_monotonic=deadline_monotonic,
    )
    harness.system_content = system_content
    record_chain_system_head(system_content)
    harness.primary_queue_wait_ms = primary_queue_wait_ms
    harness.primary_in_flight_at_start = primary_in_flight_at_start
    new_evidence_handle = new_observation["evidence_handle"]
    warnings = ["session_reattached", *initial_warnings]
    json_repair_callback = (
        partial(
            _repair_v3_serial_candidate,
            services=services,
            coordinator=sidecar_coordinator,
            admissions=sidecar_admissions,
            invocation_state=sidecar_state,
            warnings=warnings,
            deadline_monotonic=deadline_monotonic,
        )
        if (
            sidecar_coordinator is not None
            and sidecar_admissions is not None
            and sidecar_state is not None
        )
        else None
    )
    stage_status: dict[str, str] = {
        "input_validation": "completed",
        "deterministic_preliminary": "completed",
        "semantic_appraisal": "skipped",
        "final_reduction": "skipped",
        "branch_cognition": "skipped",
        "workspace_collapse": "skipped",
        "action_planning": "skipped",
    }
    evidence_ref = new_observation["evidence_ref"]
    resolver_interlude = {
        "resolver_observation": {
            "evidence_handle": new_evidence_handle,
            "source_kind": evidence_ref["source_kind"],
            "semantic_summary": evidence_ref["semantic_summary"],
            "semantic_text": new_observation["semantic_text"],
            "authority": new_observation["authority"],
        }
    }
    l1_residue, l1_observed = _take_ready_l1_residue(
        l1_task,
        warnings,
    )
    (
        appraisal_rows,
        appraisal_failures,
        appraisal_context_limited,
        l1_observed,
        observation_context_pending,
    ) = await _run_appraisal_stages(
        questions_by_family=questions_by_family,
        evidence_handles=(new_evidence_handle,),
        handle_to_ref=projection.handle_to_ref,
        harness=harness,
        system_content=system_content,
        services=services,
        warnings=warnings,
        observation_context=None,
        relation_context=_appraisal_relation_context(
            projection.payload,
            payload["scene_context"],
        ),
        l1_residue=l1_residue,
        l1_observed=l1_observed,
        interludes=(resolver_interlude,),
        role_assignment_handles_by_evidence=appraisal_role_domains,
        attempt_owner="serial_appraisal",
        branch_prefix="resolver_delta_appraisal",
        stage_prefix="R.",
        deterministic_only=services.sidecar_lane is None,
        json_repair_callback=json_repair_callback,
        deadline_monotonic=deadline_monotonic,
    )
    first_carriers_accepted = not observation_context_pending
    goal_l1_residue = l1_residue if not first_carriers_accepted else None
    goal_interludes = (resolver_interlude,)
    if first_carriers_accepted:
        goal_interludes = ()
        l1_residue = None
    (
        final_state,
        appraisal_results,
        reduction_failures,
        comparison_results,
        _accepted_relationship_deltas,
    ) = _reduce_serial_appraisals(
        reduction_state=previous_state,
        appraisal_rows=appraisal_rows,
        payload=payload,
        projection=projection,
        updated_at=updated_at,
        elapsed_seconds=elapsed_seconds,
        reducer_relationship_context=reducer_relationship_context,
    )
    appraisal_failures.update(reduction_failures)
    warnings.extend(
        f"semantic_appraisal_failed:{error_code}"
        for error_code in reduction_failures.values()
    )
    stage_status["semantic_appraisal"] = (
        "degraded" if appraisal_context_limited else "completed"
    )
    stage_status["final_reduction"] = "completed"

    final_projection = project_state_for_prompt(
        final_state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        relationship_context=payload.get("relationship_context"),
        character_operational_context=payload.get(
            "character_operational_context",
        ),
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
    )
    revision_roster = _post_i1_goal_roster(final_state)
    goal_branch_context = _branch_context(
        final_projection,
        final_state,
        payload["evidence"],
        scene_context=payload["scene_context"],
        private_continuity_context=payload["private_continuity_context"],
        past_dialog_cognition_context=payload[
            "past_dialog_cognition_context"
        ],
        group_engagement_action_context=payload[
            "group_engagement_action_context"
        ],
    )
    goal_role_bindings = goal_branch_context["_role_bindings"]
    prompt_role_bindings = {
        handle: {
            "role": binding.get("role", ""),
            "entity_kind": binding.get("entity_kind", ""),
        }
        for handle, binding in goal_role_bindings.items()
    }
    evidence_handles = [row["evidence_handle"] for row in payload["evidence"]]
    episode_evidence_handles = {
        row["evidence_handle"]
        for row in payload["evidence"]
        if row["evidence_ref"]["source_kind"]
        in CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS
    }
    selection_operations = project_required_selection_operations(
        payload["evidence"]
    )
    progress_evidence = (
        project_conversation_progress_evidence(payload["evidence"])
        if selection_operations
        else []
    )
    ordinary_definition = next(
        definition
        for definition in revision_roster
        if definition.goal_kind == ORDINARY_GOAL_KIND
    )
    ordinary_goal = _goal_for_branch(final_state, ORDINARY_GOAL_KIND)
    continuity_context = _goal_continuity_context(payload)
    authoritative_state = _goal_authoritative_state(
        final_state=final_state,
        current_goal_kind=ORDINARY_GOAL_KIND,
        final_projection=final_projection,
    )
    if not l1_observed and goal_l1_residue is None:
        goal_l1_residue, l1_observed = _take_ready_l1_residue(
            l1_task,
            warnings,
        )
        if not l1_observed and l1_task is not None:
            if sidecar_state is not None:
                sidecar_state.record_cancellation(l1_task)
            l1_task.cancel()
            warnings.append("sidecar_l1_dropped")
    ordinary_question = v3_prompt.ChainQuestion(
        contract_name="ordinary_goal_bid.v1",
        payload=v3_prompt.build_goal_question_payload(
            goal_kind=ORDINARY_GOAL_KIND,
            goal_projection=_goal_projection(
                ordinary_goal,
                ORDINARY_GOAL_KIND,
            ),
            evidence_handles=evidence_handles,
            action_tendencies=list(ordinary_definition.action_tendencies),
            branch_intent_guidance="",
            role_bindings=prompt_role_bindings,
            selection_operations=list(selection_operations),
            progress_evidence=list(progress_evidence),
            authoritative_state=authoritative_state,
            continuity_context=continuity_context,
            current_episode_evidence_handles=episode_evidence_handles,
            dialogue_role_bindings=dialogue_role_bindings,
            carried_relational_willingness=carried_relational_willingness,
            l1_residue=goal_l1_residue,
        ),
    )

    def ordinary_validator(parsed: dict[str, object]) -> object:
        if selection_operations:
            return _validate_and_materialize_selection_goal_draft(
                parsed,
                evidence_handles=set(evidence_handles),
                role_handles=set(goal_role_bindings),
                selection_operations=selection_operations,
                episode_handles=None,
                require_relational_willingness=False,
            )
        return validate_recurrence_ordinary_goal_bid_draft(
            parsed,
            carried_relational_willingness=carried_relational_willingness,
            evidence_handles=set(evidence_handles),
            role_handles=set(goal_role_bindings),
        )

    ordinary_result = await _invoke_goal_branch_for_phase_recovery(
        harness=harness,
        system_content=system_content,
        llm=services.llm,
        config=_bounded_stage_config(
            services.chain_lane,
            stage_name="R.G1a",
            completion_cap=_GOAL_COMPLETION_CAP,
        ),
        question=ordinary_question,
        validator=ordinary_validator,
        attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
        interludes=goal_interludes,
        attempt_owner="serial_goal_ordinary",
        v2_stage="goal_bid_structure",
        v2_branch_ids=(ORDINARY_GOAL_KIND,),
        deterministic_only=services.sidecar_lane is None,
        json_repair_callback=json_repair_callback,
        deadline_monotonic=deadline_monotonic,
    )
    if ordinary_result is None:
        warnings.append("ordinary_goal_context_limit")
    ordinary_local = (
        ordinary_result.validated
        if ordinary_result is not None
        else None
    )
    ordinary_bid = (
        _materialize_goal_bid(
            ordinary_definition,
            final_state,
            goal_role_bindings,
            ordinary_local,
        )
        if ordinary_local is not None
        else None
    )
    phase_branch_failures: dict[str, str] = {}
    if ordinary_bid is None:
        phase_branch_failures[ordinary_definition.branch_id] = (
            "ordinary_response_unavailable"
        )
    elif selection_operations:
        ordinary_bid["relational_willingness"] = dict(
            carried_relational_willingness
        )
    all_bids: list[ActionBidV2] = (
        [ordinary_bid] if ordinary_bid is not None else []
    )

    active_roster = [
        _active_goal_roster_row(
            definition=definition,
            final_state=final_state,
            evidence_handles=evidence_handles,
            role_handles=goal_role_bindings,
        )
        for definition in revision_roster
        if definition.goal_kind != ORDINARY_GOAL_KIND
    ]
    if active_roster:
        active_question = v3_prompt.ChainQuestion(
            contract_name="active_goal_bid_group.v1",
            payload=v3_prompt.build_active_goal_group_question_payload(
                roster=active_roster,
                evidence_handles=evidence_handles,
                role_handles=goal_role_bindings,
                continuity_context=continuity_context,
                dialogue_role_bindings=dialogue_role_bindings,
            ),
        )
        try:
            active_result = await invoke_serial_question_with_repair(
                harness=harness,
                system_content=system_content,
                llm=services.llm,
                config=_bounded_stage_config(
                    services.chain_lane,
                    stage_name="R.G1b",
                    completion_cap=_GOAL_COMPLETION_CAP,
                ),
                question=active_question,
                validator=partial(
                    _validate_active_goal_group,
                    roster=active_roster,
                    evidence_handles=set(evidence_handles),
                    role_handles=set(goal_role_bindings),
                ),
                attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
                attempt_owner="serial_goal_active",
                v2_stage="goal_bid_structure",
                v2_branch_ids=tuple(
                    row["branch_id"] for row in active_roster
                ),
                deterministic_only=services.sidecar_lane is None,
                json_repair_callback=json_repair_callback,
                deadline_monotonic=deadline_monotonic,
            )
            active_rows = active_result.validated
            active_failures: dict[str, str] = {}
            if (
                active_rows is None
                and active_result.disposition.kind == "structural_exhausted"
            ):
                salvaged_rows, active_failures = (
                    _salvage_exhausted_active_group(
                        raw_output=active_result.raw_output,
                        question=active_question,
                        roster=active_roster,
                        evidence_handles=set(evidence_handles),
                        role_handles=set(goal_role_bindings),
                        harness=harness,
                    )
                )
                active_rows = salvaged_rows or None
                phase_branch_failures.update(active_failures)
                warnings.extend(
                    f"{error_code}:{branch_id}"
                    for branch_id, error_code in active_failures.items()
                )
        except CognitionContextLimitError:
            active_rows = None
            warnings.append("active_goal_group_context_limit")
        if active_rows is None:
            warnings.append("v3_chain_unavailable:active_goal_group")
        else:
            definitions_by_branch = {
                definition.branch_id: definition
                for definition in revision_roster
            }
            for row in active_rows:
                all_bids.append(
                    _materialize_goal_bid(
                        definitions_by_branch[row["branch_id"]],
                        final_state,
                        goal_role_bindings,
                        row,
                    )
                )

    phase_execution = _synthesize_and_recover_goal_phase(
        revision_roster,
        all_bids,
        phase_branch_failures,
        harness.ledger,
    )
    warnings.extend(phase_execution.warnings)
    live_bids, stale_branch_ids = _bids_with_live_goals(
        all_bids,
        final_state,
    )
    if not live_bids:
        raise CognitionExecutionError(
            "ordinary_response",
            error_code="ordinary_response_unavailable",
            stage="branch_cognition",
            safe_checkpoint="final_reduction",
        )
    ordered_bids = validate_complete_bids(live_bids)
    partition_request = prepare_partition(
        ordered_bids,
        _workspace_current_event(payload["evidence"]),
        _workspace_goal_contexts(ordered_bids, final_state),
    )
    bid_handles = partition_request.handles
    bid_index = partition_request.prompt_payload["bid_index"]
    action_handles = {
        f"a{index}": dict(row)
        for index, row in enumerate(payload["available_actions"], start=1)
    }
    resolver_handles = {
        f"r{index}": dict(row)
        for index, row in enumerate(
            payload["available_resolver_capabilities"],
            start=1,
        )
    }
    relational_decision = _ordinary_relational_decision(live_bids)
    relationship_sensitive = (
        relational_decision is not None
        and relational_decision["applicability"] == "relationship_sensitive"
    )
    i2_interlude = {
        "notice_kind": "I2",
        "complete_bid_count": len(live_bids),
        "stale_branch_ids": sorted(stale_branch_ids),
        "workspace_required": (
            len(live_bids) >= 2 and not relationship_sensitive
        ),
        "bid_index": {
            handle: dict(row)
            for handle, row in bid_index.items()
        },
    }

    workspace_collapse: dict[str, Any] | None = None
    i2_consumed_by_workspace = False
    if len(live_bids) >= 2 and not relationship_sensitive:
        workspace_question = v3_prompt.ChainQuestion(
            contract_name="workspace_partition.v1",
            payload=partition_request.prompt_payload,
        )
        try:
            partition_result = await invoke_serial_question_with_repair(
                harness=harness,
                system_content=system_content,
                llm=services.llm,
                config=_bounded_stage_config(
                    services.chain_lane,
                    stage_name="R.W1",
                    completion_cap=_WORKSPACE_COMPLETION_CAP,
                ),
                question=workspace_question,
                validator=partial(
                    validate_workspace_partition,
                    handles=set(partition_request.handles),
                ),
                attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
                interludes=(i2_interlude,),
                attempt_owner="serial_workspace",
                v2_stage="workspace_collapse",
                v2_branch_ids=("workspace",),
                deterministic_only=services.sidecar_lane is None,
                json_repair_callback=json_repair_callback,
                deadline_monotonic=deadline_monotonic,
            )
            partition = partition_result.validated
        except CognitionContextLimitError:
            partition = None
            warnings.append("workspace_collapse_context_limit")
        i2_consumed_by_workspace = partition is not None
        workspace_collapse = (
            materialize_partition(partition_request.handles, partition)
            if partition is not None
            else fallback_partition_envelope(ordered_bids)
        )

    action_collapse, eligible_bids = _collapse_for_action_plan(
        all_bids=all_bids,
        final_state=final_state,
        workspace_collapse=workspace_collapse,
    )
    primary_bid = action_collapse.get("primary_bid")
    supporting_bids = action_collapse.get("supporting_bids", [])
    self_cognition_response_required = (
        is_targetless_group_self_cognition_episode(payload["episode"])
    )
    self_cognition_context = _self_cognition_response_context(
        payload,
        episode_evidence_handles,
    )
    action_question = v3_prompt.ChainQuestion(
        contract_name="action_plan.v1",
        payload=v3_prompt.build_action_plan_question_payload(
            primary_bid_handle=_selected_bid_handle(
                primary_bid,
                bid_handles,
            ),
            supporting_bid_handles=[
                _selected_bid_handle(bid, bid_handles)
                for bid in supporting_bids
            ],
            bid_index=bid_index,
            action_index=action_handles,
            resolver_index=resolver_handles,
            resolver_context=payload["resolver_context"],
            runtime_capability_limits=payload.get(
                "runtime_capability_limits",
                [],
            ),
            current_goal_progress=payload.get("resolver_goal_progress"),
            required_resolver_evidence_dependency=payload.get(
                "required_resolver_evidence_dependency"
            ),
            evidence=payload["evidence"],
            self_cognition_response_context=self_cognition_context,
        ),
    )

    accepted_at_utc = accepted_at_utc_from_episode(
        payload["episode"]
    )

    def action_validator(parsed: dict[str, object]) -> object:
        return validate_action_plan_decision(
            parsed,
            bid_handles=bid_handles,
            action_handles=action_handles,
            resolver_handles=resolver_handles,
            current_goal_progress=payload.get("resolver_goal_progress"),
            required_resolver_evidence_dependency=payload.get(
                "required_resolver_evidence_dependency"
            ),
            runtime_capability_limits=payload.get(
                "runtime_capability_limits",
                [],
            ),
            self_cognition_response_required=self_cognition_response_required,
            evidence=payload["evidence"],
            target_handles=_self_cognition_target_handles(
                payload["scene_context"]
            ),
            accepted_at_utc=accepted_at_utc,
        )

    try:
        action_result = await invoke_serial_question_with_repair(
            harness=harness,
            system_content=system_content,
            llm=services.llm,
            config=_bounded_stage_config(
                services.chain_lane,
                stage_name="R.P1",
                completion_cap=_ACTION_COMPLETION_CAP,
            ),
            question=action_question,
            validator=action_validator,
            attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
            interludes=() if i2_consumed_by_workspace else (i2_interlude,),
            attempt_owner="serial_action_plan",
            v2_stage="action_planning",
            v2_branch_ids=("action_plan",),
            deterministic_only=services.sidecar_lane is None,
            json_repair_callback=json_repair_callback,
            deadline_monotonic=deadline_monotonic,
        )
        action_decision = action_result.validated
    except CognitionContextLimitError:
        if self_cognition_response_required:
            raise CognitionExecutionError(
                "required self-cognition response was unavailable",
                error_code="self_cognition_response_unavailable",
                stage="action_planning",
                safe_checkpoint="final_reduction",
            )
        action_decision = None
        warnings.append("action_planning_context_limit")
    if self_cognition_response_required and action_decision is None:
        raise CognitionExecutionError(
            "required self-cognition response was unavailable",
            error_code="self_cognition_response_unavailable",
            stage="action_planning",
            safe_checkpoint="final_reduction",
        )
    await _drain_l1_sidecar(
        l1_task,
        invocation_state=sidecar_state,
        warnings=warnings,
    )
    action_decision = await _authorize_serial_action_decision(
        action_decision=action_decision,
        primary_bid=primary_bid,
        bid_handles=bid_handles,
        action_handles=action_handles,
        resolver_handles=resolver_handles,
        payload=payload,
        services=services,
        coordinator=sidecar_coordinator,
        admissions=sidecar_admissions,
        invocation_state=sidecar_state,
        warnings=warnings,
        deadline_monotonic=deadline_monotonic,
    )
    validated_output = _build_serial_output(
        payload=payload,
        previous_state=previous_state,
        final_state=final_state,
        questions=questions,
        appraisal_results=appraisal_results,
        appraisal_failures=appraisal_failures,
        comparison_results=comparison_results,
        preliminary_branches=revision_roster,
        all_bids=all_bids,
        ordinary_bid=ordinary_bid,
        workspace_collapse=workspace_collapse,
        action_decision=action_decision,
        branch_failures=phase_branch_failures,
        harness=harness,
        warnings=warnings,
        stage_status=stage_status,
        started_at=started_at,
    )
    token_ledger = dict(harness.transcript.token_ledger or {})
    token_ledger.update({
        "active_total_ceiling_tokens": (
            harness.budget.active_total_ceiling_tokens
        ),
        "extension_available": int(harness.budget.extension_available),
        "extension_used": int(harness.budget.extension_used),
        "reanchor_used": int(harness.transcript.reanchor_used),
    })
    advanced_session = advance_session_after_output(
        session=session,
        payload=payload,
        output=validated_output,
        accepted_messages=harness.transcript.to_messages(),
        accepted_products=harness.transcript.accepted_products,
        current_roster=tuple(
            branch_id
            for branch_id in sorted(
                {bid["branch_id"] for bid in eligible_bids},
                key=branch_order_key,
            )
        ),
        attempt_ledger=dict(harness.ledger._counts),
        token_ledger=token_ledger,
        reanchor_used=harness.transcript.reanchor_used,
    )
    advanced_session.expires_monotonic = (
        advanced_session.last_used_monotonic + _session_ttl_seconds(services)
    )
    session_registry.put(advanced_session)
    _record_harness_observability(
        harness=harness,
        sidecar_state=sidecar_state,
        session_event="reattached",
    )
    return validated_output

def _fallback_chain_identity(
    input_payload: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    """Derive non-empty direct-engine correlation without raw content."""

    ledger = current_v2_attempt_ledger()
    ledger_invocation_id = (
        ledger.cognition_invocation_id
        if ledger is not None
        else ""
    )
    episode = input_payload.get("episode")
    episode_id = "episode"
    platform = ""
    correlation_id = ""
    if isinstance(episode, Mapping):
        candidate_episode_id = episode.get("episode_id")
        if isinstance(candidate_episode_id, str) and candidate_episode_id.strip():
            episode_id = candidate_episode_id.strip()
        origin = episode.get("origin_metadata")
        if isinstance(origin, Mapping):
            candidate_correlation_id = origin.get("correlation_id")
            if (
                isinstance(candidate_correlation_id, str)
                and candidate_correlation_id.strip()
            ):
                correlation_id = candidate_correlation_id.strip()
        target_scope = episode.get("target_scope")
        if isinstance(target_scope, Mapping):
            candidate_platform = target_scope.get("platform")
            if isinstance(candidate_platform, str):
                platform = candidate_platform.strip().lower()
    run_id = ledger_invocation_id or correlation_id or episode_id
    source_kind = "debug" if platform == "debug" else "unknown"
    trace_id = llm_tracing.current_trace_id().strip() or run_id
    return run_id, source_kind, trace_id, run_id


def _ensure_chain_scope(
    input_payload: Mapping[str, Any],
) -> tuple[object | None, object]:
    """Bind a fresh producer scope while retaining ambient identity."""

    (
        fallback_run_id,
        fallback_source_kind,
        fallback_trace_id,
        fallback_invocation_id,
    ) = _fallback_chain_identity(input_payload)
    ambient_scope = current_chain_scope()
    run_id = (
        ambient_scope.run_id
        if ambient_scope is not None and ambient_scope.run_id
        else fallback_run_id
    )
    source_kind = (
        ambient_scope.source_kind
        if ambient_scope is not None and ambient_scope.source_kind != "unknown"
        else fallback_source_kind
    )
    trace_id = (
        ambient_scope.llm_trace_id
        if ambient_scope is not None and ambient_scope.llm_trace_id
        else fallback_trace_id
    )
    invocation_id = (
        ambient_scope.cognition_invocation_id
        if ambient_scope is not None and ambient_scope.cognition_invocation_id
        else fallback_invocation_id
    )
    scope_token = bind_protected_chain_records(
        run_id=run_id,
        source_kind=source_kind,
        llm_trace_id=trace_id,
        cognition_invocation_id=invocation_id,
    )
    scope = current_chain_scope()
    if scope is None:
        raise RuntimeError("V3 chain diagnostics scope failed to bind")
    scope.chain_run_id = f"cogchain_{uuid4().hex}"
    scope.started_at_utc = storage_utc_now_iso()
    scope.started_monotonic = time.perf_counter()
    return scope_token, scope


def _record_harness_observability(
    *,
    harness: SerialChainHarness,
    sidecar_state: SidecarInvocationState | None,
    session_event: str,
) -> None:
    """Project accepted transcript and lane-owned aggregates to the scope."""

    record_accepted_transcript(
        harness.transcript.to_messages(),
        system_content=harness.system_content,
    )
    token_ledger = dict(harness.transcript.token_ledger or {})
    token_ledger.update({
        "declared_context_window_tokens": (
            harness.budget.plan.serving_window_tokens
        ),
        "normal_total_ceiling_tokens": (
            harness.budget.plan.normal_total_ceiling_tokens
        ),
        "extended_total_ceiling_tokens": (
            harness.budget.plan.extended_total_ceiling_tokens
        ),
        "active_total_ceiling_tokens": harness.budget.active_total_ceiling_tokens,
        "extension_available": harness.budget.extension_available,
        "extension_used": harness.budget.extension_used,
        "reanchor_used": harness.transcript.reanchor_used,
    })
    record_token_ledger(token_ledger)
    if sidecar_state is not None:
        sidecar_diagnostics = sidecar_state.diagnostics()
        record_sidecar_aggregate(sidecar_diagnostics)
        queue_wait_ms = int(
            sidecar_diagnostics.get("sidecar_queue_wait_ms_total", 0)
        )
        max_in_flight = int(
            sidecar_diagnostics.get("sidecar_max_in_flight", 0)
        )
        for stream_kind, counter_name in (
            ("l1", "l1_stream_count"),
            ("json_repair", "json_repair_call_count"),
            ("action_authorization", "action_auth_attempt_count"),
            ("resolver_authorization", "resolver_auth_attempt_count"),
        ):
            stream_count = int(sidecar_diagnostics.get(counter_name, 0))
            if stream_count <= 0:
                continue
            record_chain_step({
                "step_id": f"sidecar:{stream_kind}",
                "stage_kind": stream_kind,
                "lane_kind": "sidecar",
                "sidecar_stream_kind": stream_kind,
                "status": "accepted",
                "attempt_count": stream_count,
                "queue_wait_ms": queue_wait_ms,
                "in_flight_at_start": max_in_flight,
                "disposition": "accepted",
                "parse_status": "deterministic",
                "cache_class": "sidecar",
            })
    record_session_event(session_event)
    if harness.transcript.reanchor_used:
        record_session_event("reanchored")


def _record_scope_terminal_projection(
    *,
    output: Mapping[str, Any] | None,
    terminal_disposition: str,
) -> None:
    """Add deterministic registered markers and bounded warning projections."""

    if output is not None:
        diagnostics = output.get("diagnostics")
        if isinstance(diagnostics, Mapping):
            warnings = diagnostics.get("warnings")
            if isinstance(warnings, Sequence) and not isinstance(
                warnings,
                (str, bytes, bytearray),
            ):
                warning_values = [
                    warning
                    for warning in warnings
                    if isinstance(warning, str)
                ]
                record_warning_codes(warning_values)
                for warning in warning_values:
                    record_degradation_marker(warning)
            stage_status = diagnostics.get("stage_status")
        else:
            stage_status = None
    else:
        stage_status = None

    active_scope = current_chain_scope()
    if active_scope is None:
        return
    actual_step_ids = {
        str(step.get("step_id"))
        for step in active_scope.steps
        if isinstance(step, Mapping)
    }
    status_by_step = {
        "A1": "semantic_appraisal",
        "A2": "semantic_appraisal",
        "G1a": "branch_cognition",
        "G1b": "branch_cognition",
        "W1": "workspace_collapse",
        "P1": "action_planning",
    }
    for step_id in SERIAL_CHAIN_STEPS:
        if step_id in {"I1", "I2"} or step_id not in actual_step_ids:
            current_status = (
                stage_status.get(status_by_step.get(step_id, ""), "skipped")
                if isinstance(stage_status, Mapping)
                else "skipped"
            )
            if step_id in {"I1", "I2"}:
                current_status = "completed"
            marker_status = "accepted" if current_status == "completed" else "skipped"
            record_registered_step(
                step_id=step_id,
                stage_kind=step_id,
                status=marker_status,
                disposition=(
                    "accepted"
                    if marker_status == "accepted"
                    else terminal_disposition
                ),
            )
    record_chain_step({
        "step_id": "O",
        "stage_kind": "output",
        "status": "accepted" if output is not None else "failed",
        "disposition": terminal_disposition,
        "parse_status": "deterministic",
        "cache_class": "deterministic",
    })


async def _best_effort_chain_writer(
    writer_name: str,
    writer: Any,
    **kwargs: Any,
) -> None:
    """Invoke one observability writer without affecting cognition output."""

    try:
        await asyncio.wait_for(writer(**kwargs), timeout=0.5)
    except Exception as exc:  # noqa: BLE001 - writer boundary is non-propagating
        _LOGGER.warning(
            "V3 chain observability writer %s failed: %s",
            writer_name,
            exc.__class__.__name__,
        )


def _scope_messages_for_trace(scope: object) -> list[object]:
    """Convert accepted transcript rows to protected trace messages."""

    messages: list[object] = []
    for role, content in scope.accepted_messages:
        if role == "system":
            messages.append(SystemMessage(content=content))
        elif role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def _build_chain_run_document(
    *,
    scope: object,
    services: CognitionChainServicesV3,
    terminal_disposition: str,
    completed_at: str,
) -> dict[str, object]:
    """Build the closed sanitized ``cognition_chain_run.v2`` document."""

    ledger = {
        key: scope.token_ledger.get(key, default)
        for key, default in CHAIN_LEDGER_DEFAULTS.items()
    }
    sidecar = {
        key: scope.sidecar.get(key, default)
        for key, default in CHAIN_SIDECAR_DEFAULTS.items()
    }
    warning_codes = list(scope.warning_codes)[:32]
    started_at = scope.started_at_utc or completed_at
    document = {
        "schema_version": "cognition_chain_run.v2",
        "chain_run_id": scope.chain_run_id,
        "engine": "v3",
        "run_id": scope.run_id,
        "llm_trace_id": scope.llm_trace_id,
        "cognition_invocation_id": scope.cognition_invocation_id,
        "source_kind": scope.source_kind,
        "chain_model_name": services.chain_lane.model,
        "sidecar_model_name": (
            services.sidecar_lane.model
            if services.sidecar_lane is not None
            else ""
        ),
        "subconscious_enabled": services.subconscious_enabled,
        "appraisal_stage_layout": COGNITION_V3_APPRAISAL_STAGE_LAYOUT,
        "started_at": started_at,
        "completed_at": completed_at,
        "terminal_disposition": terminal_disposition,
        "steps": [dict(step) for step in scope.steps[:96]],
        "ledger": ledger,
        "sidecar": sidecar,
        "session_events": list(scope.session_events)[:16],
        "degradation_markers": list(scope.degradation_markers)[:32],
        "warning_codes": warning_codes,
        "expires_at": expiry_from_storage_iso(
            completed_at,
            ttl_days=AUDIT_LOG_TTL_DAYS,
        ),
    }
    return document


async def _persist_chain_observability(
    *,
    scope: object,
    services: CognitionChainServicesV3,
    terminal_disposition: str,
    duration_ms: int,
) -> None:
    """Write protected, event, and sanitized DB records at the run boundary."""

    record_current_sidecar_aggregate()
    completed_at = storage_utc_now_iso()
    steps = [dict(step) for step in scope.steps[:96]]
    prompt_chars_total = sum(
        int(step.get("prompt_chars", 0))
        for step in steps
        if isinstance(step.get("prompt_chars", 0), int)
    )
    new_suffix_chars_total = sum(
        int(step.get("new_suffix_chars", 0))
        for step in steps
        if isinstance(step.get("new_suffix_chars", 0), int)
    )
    prefix_share_ratio = (
        max(0.0, min(1.0, 1.0 - (new_suffix_chars_total / prompt_chars_total)))
        if prompt_chars_total
        else 0.0
    )
    token_ledger = scope.token_ledger
    sidecar = scope.sidecar
    deadline_ms = max(
        0,
        int(services.turn_deadline_seconds * 1000),
    )
    deadline_consumption_ratio = (
        max(0.0, min(1.0, duration_ms / deadline_ms))
        if deadline_ms
        else 0.0
    )
    await _best_effort_chain_writer(
        "protected_transcript",
        llm_tracing.record_cognition_chain_transcript,
        trace_id=scope.llm_trace_id,
        run_id=scope.run_id,
        messages=_scope_messages_for_trace(scope),
        steps=steps,
        terminal_disposition=terminal_disposition,
        chain_model_name=services.chain_lane.model,
        sidecar_model_name=(
            services.sidecar_lane.model
            if services.sidecar_lane is not None
            else ""
        ),
    )
    await _best_effort_chain_writer(
        "event_log",
        event_logging.record_cognition_chain_event,
        run_id=scope.run_id,
        cognition_invocation_id=scope.cognition_invocation_id,
        terminal_disposition=terminal_disposition,
        chain_model_name=services.chain_lane.model,
        sidecar_model_name=(
            services.sidecar_lane.model
            if services.sidecar_lane is not None
            else ""
        ),
        step_count=len(steps),
        repair_count=sum(
            int(step.get("repair_count", 0))
            for step in steps
            if isinstance(step.get("repair_count", 0), int)
        ),
        cold_start_count=1 if "cold" in scope.session_events else 0,
        prompt_chars_total=prompt_chars_total,
        new_suffix_chars_total=new_suffix_chars_total,
        prefix_share_ratio=prefix_share_ratio,
        max_estimated_prompt_tokens=int(
            token_ledger.get("max_estimated_prompt_tokens", 0)
        ),
        max_reserved_completion_tokens=int(
            token_ledger.get("max_reserved_completion_tokens", 0)
        ),
        max_estimated_total_context_tokens=int(
            token_ledger.get("max_estimated_total_context_tokens", 0)
        ),
        active_total_ceiling_tokens=int(
            token_ledger.get("active_total_ceiling_tokens", 0)
        ),
        extension_available=bool(token_ledger.get("extension_available", False)),
        extension_used=bool(token_ledger.get("extension_used", False)),
        reanchor_used=bool(token_ledger.get("reanchor_used", False)),
        session_disposition=(scope.session_events[-1] if scope.session_events else ""),
        duration_ms=duration_ms,
        deadline_ms=deadline_ms,
        deadline_consumption_ratio=deadline_consumption_ratio,
        l1_stream_count=int(sidecar.get("l1_stream_count", 0)),
        json_repair_call_count=int(sidecar.get("json_repair_call_count", 0)),
        action_auth_attempt_count=int(sidecar.get("action_auth_attempt_count", 0)),
        resolver_auth_attempt_count=int(sidecar.get("resolver_auth_attempt_count", 0)),
        sidecar_queue_wait_ms_total=int(sidecar.get("queue_wait_ms_total", 0)),
        sidecar_max_in_flight=int(sidecar.get("max_in_flight", 0)),
        l1_preempted_by_repair=bool(sidecar.get("l1_preempted_by_repair", False)),
        sidecar_cancellation_count=int(sidecar.get("cancellation_count", 0)),
        warning_codes=list(scope.warning_codes)[:32],
    )
    chain_document = _build_chain_run_document(
        scope=scope,
        services=services,
        terminal_disposition=terminal_disposition,
        completed_at=completed_at,
    )
    await _best_effort_chain_writer(
        "database",
        db.save_cognition_chain_run,
        document=chain_document,
    )


async def _best_effort_persist_chain_observability(
    *,
    scope: object,
    services: CognitionChainServicesV3,
    terminal_disposition: str,
    duration_ms: int,
) -> None:
    """Keep producer projection failures outside the cognition result path."""

    try:
        await _persist_chain_observability(
            scope=scope,
            services=services,
            terminal_disposition=terminal_disposition,
            duration_ms=duration_ms,
        )
    except Exception as exc:  # noqa: BLE001 - observability is non-propagating
        _LOGGER.warning(
            "V3 chain observability projection failed: %s",
            exc.__class__.__name__,
        )


async def run_cognition(
    input_payload: CognitionCoreInputV2,
    services: CognitionChainServicesV3,
) -> CognitionCoreOutputV2:
    """Run the serialized V3 cognition chain with failure-only replay capture.

    The shell owns the invocation-wide attempt ledger, binds one protected
    chain-record scope when none is already active, and captures terminal or
    partial failures through the protected failure capsule. The semantic path
    is ``_run_serial_cognition``: one append-only primary lane in the fixed
    serial order with the V2 deterministic substrate preserved after the cold
    primary chain.
    """

    deadline_monotonic = time.monotonic() + services.turn_deadline_seconds
    ledger_token = None
    if current_v2_attempt_ledger() is None:
        ledger_token = bind_v2_attempt_ledger(
            create_v2_attempt_ledger(),
            graph_attempt=1,
        )
    records_token, scope = _ensure_chain_scope(input_payload)
    observability_started_at = time.perf_counter()
    try:
        session = failure_capsule.begin_failure_capsule(
            trace_id=scope.llm_trace_id,
            entrypoint="cognition_core_v3.run_cognition",
            input_payload=input_payload,
        )
        try:
            output = await _run_serial_cognition(
                input_payload,
                services,
                deadline_monotonic,
            )
        except Exception as exc:
            failure_capsule.mark_failure(
                session,
                failure_kind="terminal_failure",
                stage_name="cognition_core_v3.run_cognition",
                details={},
                exception=exc,
            )
            failure_capsule.finish_failure_capsule(
                session,
                outcome="terminal_failure",
                exception=exc,
                attempt_ledger=snapshot_v2_attempt_ledger(),
            )
            _record_scope_terminal_projection(
                output=None,
                terminal_disposition="terminal failure",
            )
            await _best_effort_persist_chain_observability(
                scope=scope,
                services=services,
                terminal_disposition="terminal failure",
                duration_ms=max(
                    0,
                    int((time.perf_counter() - observability_started_at) * 1000),
                ),
            )
            raise

        _mark_cognition_partial_failures(session, output)
        failure_capsule.finish_failure_capsule(
            session,
            outcome=None,
            attempt_ledger=snapshot_v2_attempt_ledger(),
        )
        output_warnings = output.get("diagnostics", {}).get("warnings", [])
        has_warnings = (
            isinstance(output_warnings, Sequence)
            and not isinstance(output_warnings, (str, bytes, bytearray))
            and bool(output_warnings)
        )
        terminal_disposition = (
            "accepted-degraded" if has_warnings else "complete"
        )
        _record_scope_terminal_projection(
            output=output,
            terminal_disposition=terminal_disposition,
        )
        await _best_effort_persist_chain_observability(
            scope=scope,
            services=services,
            terminal_disposition=terminal_disposition,
            duration_ms=max(
                0,
                int((time.perf_counter() - observability_started_at) * 1000),
            ),
        )
        return output
    finally:
        if ledger_token is not None:
            reset_v2_attempt_ledger(ledger_token)
        if records_token is not None:
            reset_protected_chain_records(records_token)


def _create_sidecar_invocation_runtime(
    services: CognitionChainServicesV3,
) -> tuple[
    SidecarCoordinator | None,
    SidecarAdmissionLedger | None,
    SidecarInvocationState | None,
]:
    """Create the one optional sidecar authority for a cognition invocation."""

    if services.sidecar_lane is None:
        return None, None, None
    sidecar_state = SidecarInvocationState()
    bind_chain_sidecar_state(sidecar_state)
    return (
        sidecar_lane_coordinator(services.llm, services.sidecar_lane),
        SidecarAdmissionLedger(),
        sidecar_state,
    )


def _build_recurrence_l1_projection(
    payload: Mapping[str, Any],
) -> Any:
    """Build the prompt-safe state projection required before recurrence waits."""

    previous_state = validate_cognition_state(payload["mutable_state"])
    return project_state_for_prompt(
        previous_state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        relationship_context=payload.get("relationship_context"),
        character_operational_context=payload.get(
            "character_operational_context",
        ),
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
    )


async def _run_serial_cognition(
    input_payload: CognitionCoreInputV2,
    services: CognitionChainServicesV3,
    deadline_monotonic: float | None = None,
) -> CognitionCoreOutputV2:
    """Run one V3 invocation with session ownership before FIFO primary work."""

    started_at = time.perf_counter()
    validated_input = validate_cognition_core_input(input_payload)
    episode_id = str(validated_input["episode"].get("episode_id", "episode"))
    owner_identity = (
        f"{services.chain_lane.base_url}|{services.chain_lane.model}"
    )
    session_key = build_session_key(
        episode_id=episode_id,
        state_scope=validated_input["state_scope"],
        owner_identity=owner_identity,
    )
    primary_coordinator = primary_lane_coordinator(
        services.llm,
        services.chain_lane,
    )
    (
        sidecar_coordinator,
        sidecar_admissions,
        sidecar_state,
    ) = _create_sidecar_invocation_runtime(services)
    cold_warnings: list[str] = []
    store_cold_session = True
    cycle_index = validated_input.get("resolver_cycle_index", 0)
    if cycle_index > 0:
        claim = _CHAIN_SESSION_REGISTRY.claim(session_key)
        if claim.session is None:
            record_session_event("rebuilt")
            cold_warnings.append("session_rebuilt_session_miss")
        elif not claim.claim_token:
            cold_warnings.append(claim.disposition)
            store_cold_session = False
        else:
            l1_task = None
            try:
                reattachment = reattach_or_rebuild(
                    session=claim.session,
                    payload=validated_input,
                )
                if reattachment.reattached:
                    recurrence_projection = _build_recurrence_l1_projection(
                        validated_input,
                    )
                    l1_task = _start_l1_sidecar(
                        payload=validated_input,
                        projection=recurrence_projection,
                        services=services,
                        coordinator=sidecar_coordinator,
                        admissions=sidecar_admissions,
                        invocation_state=sidecar_state,
                        deadline_monotonic=deadline_monotonic,
                    )
                    await _yield_after_l1_sidecar_start(l1_task)
                    try:
                        async with primary_coordinator.claim(
                            deadline_monotonic=deadline_monotonic,
                        ) as lane_claim:
                            tail_warnings: list[str] = []
                            if lane_claim.queue_wait_ms > 0:
                                tail_warnings.append("primary_lane_queued")
                            return await _run_reattached_resolver_tail(
                                payload=validated_input,
                                services=services,
                                session=claim.session,
                                session_registry=_CHAIN_SESSION_REGISTRY,
                                started_at=started_at,
                                l1_task=l1_task,
                                sidecar_coordinator=sidecar_coordinator,
                                sidecar_admissions=sidecar_admissions,
                                sidecar_state=sidecar_state,
                                deadline_monotonic=deadline_monotonic,
                                initial_warnings=tail_warnings,
                                primary_queue_wait_ms=lane_claim.queue_wait_ms,
                                primary_in_flight_at_start=(
                                    lane_claim.in_flight_at_start
                                ),
                            )
                    except SessionContractError:
                        record_session_event("rebuilt")
                        cold_warnings.append("session_rebuilt_session_invalid")
                else:
                    record_session_event("rebuilt")
                    cold_warnings.append(
                        "session_rebuilt_input_divergence:"
                        f"{reattachment.divergent_field}"
                    )
            finally:
                await _drain_l1_sidecar(
                    l1_task,
                    invocation_state=sidecar_state,
                    warnings=cold_warnings,
                )
                _CHAIN_SESSION_REGISTRY.release(
                    session_key,
                    claim.claim_token,
                )

    context = _build_serial_initial_context(validated_input)
    l1_task = _start_l1_sidecar(
        payload=context["payload"],
        projection=context["projection"],
        services=services,
        coordinator=sidecar_coordinator,
        admissions=sidecar_admissions,
        invocation_state=sidecar_state,
        deadline_monotonic=deadline_monotonic,
    )
    try:
        await _yield_after_l1_sidecar_start(l1_task)
        async with primary_coordinator.claim(
            deadline_monotonic=deadline_monotonic,
        ) as lane_claim:
            lane_warnings = list(cold_warnings)
            if lane_claim.queue_wait_ms > 0:
                lane_warnings.append("primary_lane_queued")
            return await _run_cold_serial_cognition(
                context=context,
                services=services,
                started_at=started_at,
                episode_id=episode_id,
                owner_identity=owner_identity,
                store_cold_session=store_cold_session,
                cold_warnings=lane_warnings,
                l1_task=l1_task,
                sidecar_coordinator=sidecar_coordinator,
                sidecar_admissions=sidecar_admissions,
                sidecar_state=sidecar_state,
                deadline_monotonic=deadline_monotonic,
                primary_queue_wait_ms=lane_claim.queue_wait_ms,
                primary_in_flight_at_start=lane_claim.in_flight_at_start,
            )
    finally:
        await _drain_l1_sidecar(
            l1_task,
            invocation_state=sidecar_state,
            warnings=cold_warnings,
        )


async def _run_cold_serial_cognition(
    *,
    context: Mapping[str, Any],
    services: CognitionChainServicesV3,
    started_at: float,
    episode_id: str,
    owner_identity: str,
    store_cold_session: bool,
    cold_warnings: Sequence[str],
    l1_task: asyncio.Task[tuple[dict[str, Any] | None, str | None]] | None,
    sidecar_coordinator: SidecarCoordinator | None,
    sidecar_admissions: SidecarAdmissionLedger | None,
    sidecar_state: SidecarInvocationState | None,
    deadline_monotonic: float | None,
    primary_queue_wait_ms: int = 0,
    primary_in_flight_at_start: int = 0,
) -> CognitionCoreOutputV2:
    """Run one cold primary sequence while its FIFO lane remains owned."""

    payload = context["payload"]
    previous_state = context["previous_state"]
    preliminary_state = context["preliminary_state"]
    projection = context["projection"]
    questions = context["questions"]
    system_content = context["system_content"]
    observation_context = context["observation_context"]
    dialogue_role_bindings = _goal_dialogue_role_bindings(observation_context)
    appraisal_role_domains = _appraisal_role_assignment_handles_by_evidence(
        observation_context,
        payload["evidence"],
    )
    relation_context = context["relation_context"]
    updated_at = _episode_updated_at(payload["episode"])
    elapsed_seconds = _cognition_elapsed_seconds(
        previous_state,
        updated_at,
    )
    reducer_relationship_context = _native_relationship_context(
        payload.get("relationship_context"),
    )
    evidence_handles = [row["evidence_handle"] for row in payload["evidence"]]
    episode_evidence_handles = {
        row["evidence_handle"]
        for row in payload["evidence"]
        if row["evidence_ref"]["source_kind"]
        in CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS
    }
    questions_by_family = {
        question["question_kind"]: question
        for question in questions
    }

    harness = SerialChainHarness(
        transcript=ChainTranscriptV1(deadline_monotonic=deadline_monotonic),
        ledger=AttemptLedger({
            "serial_appraisal": V2_APPRAISAL_TOTAL_ATTEMPTS,
            "serial_goal_ordinary": V2_MODEL_TOTAL_ATTEMPTS,
            "serial_action_plan": V2_MODEL_TOTAL_ATTEMPTS,
        }),
        budget=ContextBudgetLedger(
            ContextBudgetPlan(
                serving_window_tokens=services.chain_lane.context_window_tokens,
            )
        ),
        primary_queue_wait_ms=primary_queue_wait_ms,
        primary_in_flight_at_start=primary_in_flight_at_start,
    )
    harness.system_content = system_content
    record_chain_system_head(system_content)

    warnings = list(cold_warnings)
    json_repair_callback = (
        partial(
            _repair_v3_serial_candidate,
            services=services,
            coordinator=sidecar_coordinator,
            admissions=sidecar_admissions,
            invocation_state=sidecar_state,
            warnings=warnings,
            deadline_monotonic=deadline_monotonic,
        )
        if (
            sidecar_coordinator is not None
            and sidecar_admissions is not None
            and sidecar_state is not None
        )
        else None
    )
    stage_status: dict[str, str] = {
        "input_validation": "completed",
        "deterministic_preliminary": "completed",
        "semantic_appraisal": "skipped",
        "final_reduction": "skipped",
        "branch_cognition": "skipped",
        "workspace_collapse": "skipped",
        "action_planning": "skipped",
    }
    all_bids: list[ActionBidV2] = []
    workspace_collapse: dict[str, Any] | None = None
    action_decision: Mapping[str, Any] | None = None

    l1_residue, l1_observed = _take_ready_l1_residue(
        l1_task,
        warnings,
    )
    (
        appraisal_rows,
        appraisal_failures,
        appraisal_context_limited,
        l1_observed,
        observation_context_pending,
    ) = await _run_appraisal_stages(
        questions_by_family=questions_by_family,
        evidence_handles=evidence_handles,
        handle_to_ref=projection.handle_to_ref,
        harness=harness,
        system_content=system_content,
        services=services,
        warnings=warnings,
        observation_context=observation_context,
        relation_context=relation_context,
        l1_residue=l1_residue,
        l1_observed=l1_observed,
        interludes=(),
        role_assignment_handles_by_evidence=appraisal_role_domains,
        attempt_owner="serial_appraisal",
        branch_prefix="cold_appraisal",
        stage_prefix="",
        deterministic_only=services.sidecar_lane is None,
        json_repair_callback=json_repair_callback,
        deadline_monotonic=deadline_monotonic,
    )
    first_carriers_accepted = not observation_context_pending
    if first_carriers_accepted:
        observation_context = None
        l1_residue = None

    (
        final_state,
        appraisal_results,
        reduction_failures,
        comparison_results,
        _accepted_relationship_deltas,
    ) = _reduce_serial_appraisals(
        reduction_state=preliminary_state,
        appraisal_rows=appraisal_rows,
        payload=payload,
        projection=projection,
        updated_at=updated_at,
        elapsed_seconds=elapsed_seconds,
        reducer_relationship_context=reducer_relationship_context,
    )
    appraisal_failures.update(reduction_failures)
    warnings.extend(
        f"semantic_appraisal_failed:{error_code}"
        for error_code in reduction_failures.values()
    )
    stage_status["semantic_appraisal"] = (
        "degraded" if appraisal_context_limited else "completed"
    )
    stage_status["final_reduction"] = "completed"

    final_projection = project_state_for_prompt(
        final_state,
        character_constraints=payload["character_constraints"],
        character_identity_context=payload["character_identity_context"],
        relationship_context=payload.get("relationship_context"),
        character_operational_context=payload.get(
            "character_operational_context",
        ),
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
    )
    revision_roster = _post_i1_goal_roster(final_state)
    definitions_by_branch = {
        definition.branch_id: definition
        for definition in revision_roster
    }
    goal_branch_context = _branch_context(
        final_projection,
        final_state,
        payload["evidence"],
        scene_context=payload["scene_context"],
        private_continuity_context=payload[
            "private_continuity_context"
        ],
        past_dialog_cognition_context=payload[
            "past_dialog_cognition_context"
        ],
        group_engagement_action_context=payload[
            "group_engagement_action_context"
        ],
    )
    goal_role_bindings = goal_branch_context["_role_bindings"]
    prompt_role_bindings = {
        handle: {
            "role": binding.get("role", ""),
            "entity_kind": binding.get("entity_kind", ""),
        }
        for handle, binding in goal_role_bindings.items()
    }
    selection_operations = project_required_selection_operations(
        payload["evidence"]
    )
    progress_evidence = (
        project_conversation_progress_evidence(payload["evidence"])
        if selection_operations
        else []
    )
    ordinary_definition = next(
        definition
        for definition in revision_roster
        if definition.goal_kind == ORDINARY_GOAL_KIND
    )
    ordinary_goal = _goal_for_branch(final_state, ORDINARY_GOAL_KIND)
    continuity_context = _goal_continuity_context(payload)
    authoritative_state = _goal_authoritative_state(
        final_state=final_state,
        current_goal_kind=ORDINARY_GOAL_KIND,
        final_projection=final_projection,
    )
    goal_l1_residue = l1_residue
    if not l1_observed and goal_l1_residue is None:
        goal_l1_residue, l1_observed = _take_ready_l1_residue(
            l1_task,
            warnings,
        )
        if not l1_observed and l1_task is not None:
            if sidecar_state is not None:
                sidecar_state.record_cancellation(l1_task)
            l1_task.cancel()
            warnings.append("sidecar_l1_dropped")
    i1_interlude = {
        "notice_kind": "state_transition",
        "stage": "I1",
        "accepted_count": len(appraisal_rows),
        "rejected_count": len(appraisal_failures),
        "notice": _build_i1_notice(
            accepted_count=len(appraisal_rows),
            rejected_count=len(appraisal_failures),
            state_scope=final_state["state_scope"],
        ),
    }
    ordinary_question = v3_prompt.ChainQuestion(
        contract_name="ordinary_goal_bid.v1",
        payload=v3_prompt.build_goal_question_payload(
            goal_kind=ORDINARY_GOAL_KIND,
            goal_projection=_goal_projection(
                ordinary_goal,
                ORDINARY_GOAL_KIND,
            ),
            evidence_handles=evidence_handles,
            action_tendencies=list(ordinary_definition.action_tendencies),
            branch_intent_guidance="",
            role_bindings=prompt_role_bindings,
            selection_operations=list(selection_operations),
            progress_evidence=list(progress_evidence),
            authoritative_state=authoritative_state,
            continuity_context=continuity_context,
            current_episode_evidence_handles=episode_evidence_handles,
            dialogue_role_bindings=dialogue_role_bindings,
            l1_residue=goal_l1_residue,
        ),
    )

    def ordinary_validator(parsed: dict[str, object]) -> object:
        if selection_operations:
            return _validate_and_materialize_selection_goal_draft(
                parsed,
                evidence_handles=set(evidence_handles),
                role_handles=set(goal_role_bindings),
                selection_operations=selection_operations,
                episode_handles=episode_evidence_handles,
                require_relational_willingness=True,
            )
        return validate_goal_bid_draft(
            parsed,
            evidence_handles=set(evidence_handles),
            role_handles=set(goal_role_bindings),
            require_relational_willingness=True,
            episode_handles=episode_evidence_handles,
        )

    ordinary_result = await _invoke_goal_branch_for_phase_recovery(
        harness=harness,
        system_content=system_content,
        llm=services.llm,
        config=_bounded_stage_config(
            services.chain_lane,
            stage_name="G1a",
            completion_cap=_GOAL_COMPLETION_CAP,
        ),
        question=ordinary_question,
        validator=ordinary_validator,
        attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
        observation_context=observation_context,
        interludes=(i1_interlude,),
        attempt_owner="serial_goal_ordinary",
        v2_stage="goal_bid_structure",
        v2_branch_ids=(ORDINARY_GOAL_KIND,),
        deterministic_only=services.sidecar_lane is None,
        json_repair_callback=json_repair_callback,
        deadline_monotonic=deadline_monotonic,
    )
    if ordinary_result is None:
        warnings.append("ordinary_goal_context_limit")
    ordinary_local = (
        ordinary_result.validated
        if ordinary_result is not None
        else None
    )
    ordinary_bid = (
        _materialize_goal_bid(
            ordinary_definition,
            final_state,
            goal_role_bindings,
            ordinary_local,
        )
        if ordinary_local is not None
        else None
    )
    phase_branch_failures: dict[str, str] = {}
    if ordinary_bid is None:
        phase_branch_failures[ordinary_definition.branch_id] = (
            "ordinary_response_unavailable"
        )
    else:
        all_bids.append(ordinary_bid)

    active_roster = [
        _active_goal_roster_row(
            definition=definition,
            final_state=final_state,
            evidence_handles=evidence_handles,
            role_handles=goal_role_bindings,
        )
        for definition in revision_roster
        if definition.goal_kind != ORDINARY_GOAL_KIND
    ]
    if active_roster:
        active_question = v3_prompt.ChainQuestion(
            contract_name="active_goal_bid_group.v1",
            payload=v3_prompt.build_active_goal_group_question_payload(
                roster=active_roster,
                evidence_handles=evidence_handles,
                role_handles=goal_role_bindings,
                continuity_context=continuity_context,
                dialogue_role_bindings=dialogue_role_bindings,
            ),
        )
        try:
            active_result = await invoke_serial_question_with_repair(
                harness=harness,
                system_content=system_content,
                llm=services.llm,
                config=_bounded_stage_config(
                    services.chain_lane,
                    stage_name="G1b",
                    completion_cap=_GOAL_COMPLETION_CAP,
                ),
                question=active_question,
                validator=partial(
                    _validate_active_goal_group,
                    roster=active_roster,
                    evidence_handles=set(evidence_handles),
                    role_handles=set(goal_role_bindings),
                ),
                attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
                observation_context=observation_context,
                attempt_owner="serial_goal_active",
                v2_stage="goal_bid_structure",
                v2_branch_ids=tuple(
                    row["branch_id"] for row in active_roster
                ),
                deterministic_only=services.sidecar_lane is None,
                json_repair_callback=json_repair_callback,
                deadline_monotonic=deadline_monotonic,
            )
            active_rows = active_result.validated
            active_failures: dict[str, str] = {}
            if (
                active_rows is None
                and active_result.disposition.kind == "structural_exhausted"
            ):
                salvaged_rows, active_failures = (
                    _salvage_exhausted_active_group(
                        raw_output=active_result.raw_output,
                        question=active_question,
                        roster=active_roster,
                        evidence_handles=set(evidence_handles),
                        role_handles=set(goal_role_bindings),
                        harness=harness,
                    )
                )
                active_rows = salvaged_rows or None
                phase_branch_failures.update(active_failures)
                warnings.extend(
                    f"{error_code}:{branch_id}"
                    for branch_id, error_code in active_failures.items()
                )
        except CognitionContextLimitError:
            active_rows = None
            warnings.append("active_goal_group_context_limit")
        if active_rows is None:
            warnings.append("v3_chain_unavailable:active_goal_group")
        else:
            for row in active_rows:
                all_bids.append(
                    _materialize_goal_bid(
                        definitions_by_branch[row["branch_id"]],
                        final_state,
                        goal_role_bindings,
                        row,
                    )
                )

    phase_execution = _synthesize_and_recover_goal_phase(
        revision_roster,
        all_bids,
        phase_branch_failures,
        harness.ledger,
    )
    warnings.extend(phase_execution.warnings)
    eligible_bids, stale_branch_ids = _bids_with_live_goals(
        all_bids,
        final_state,
    )
    warnings.extend(
        f"stale_goal_bid_dropped:{branch_id}"
        for branch_id in stale_branch_ids
    )
    if not eligible_bids:
        raise CognitionExecutionError(
            "ordinary_response",
            error_code="ordinary_response_unavailable",
            stage="branch_cognition",
            safe_checkpoint="final_reduction",
        )
    ordered_bids = validate_complete_bids(eligible_bids)
    partition_request = prepare_partition(
        ordered_bids,
        _workspace_current_event(payload["evidence"]),
        _workspace_goal_contexts(ordered_bids, final_state),
    )
    bid_handles = partition_request.handles
    bid_index = partition_request.prompt_payload["bid_index"]
    action_handles = {
        f"a{index}": dict(row)
        for index, row in enumerate(payload["available_actions"], start=1)
    }
    resolver_handles = {
        f"r{index}": dict(row)
        for index, row in enumerate(
            payload["available_resolver_capabilities"],
            start=1,
        )
    }
    relational_decision = _ordinary_relational_decision(eligible_bids)
    relationship_sensitive = (
        relational_decision is not None
        and relational_decision["applicability"] == "relationship_sensitive"
    )
    i2_interlude = {
        "notice_kind": "I2",
        "complete_bid_count": len(eligible_bids),
        "stale_branch_ids": sorted(stale_branch_ids),
        "workspace_required": (
            len(eligible_bids) >= 2 and not relationship_sensitive
        ),
        "bid_index": {
            handle: dict(row)
            for handle, row in bid_index.items()
        },
    }

    if len(eligible_bids) >= 2 and not relationship_sensitive:
        workspace_question = v3_prompt.ChainQuestion(
            contract_name="workspace_partition.v1",
            payload=partition_request.prompt_payload,
        )

        def workspace_validator(
            parsed: dict[str, object],
        ) -> object:
            return validate_workspace_partition(
                parsed,
                set(partition_request.handles),
            )

        try:
            partition_result = await invoke_serial_question_with_repair(
                harness=harness,
                system_content=system_content,
                llm=services.llm,
                config=_bounded_stage_config(
                    services.chain_lane,
                    stage_name="W1",
                    completion_cap=_WORKSPACE_COMPLETION_CAP,
                ),
                question=workspace_question,
                validator=workspace_validator,
                attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
                observation_context=observation_context,
                interludes=(i2_interlude,),
                attempt_owner="serial_workspace",
                v2_stage="workspace_collapse",
                v2_branch_ids=("workspace",),
                deterministic_only=services.sidecar_lane is None,
                json_repair_callback=json_repair_callback,
                deadline_monotonic=deadline_monotonic,
            )
            partition = partition_result.validated
        except CognitionContextLimitError:
            partition = None
            warnings.append("workspace_collapse_context_limit")
        workspace_collapse = (
            materialize_partition(
                partition_request.handles,
                partition,
            )
            if partition is not None
            else fallback_partition_envelope(ordered_bids)
        )

    action_collapse, eligible_bids = _collapse_for_action_plan(
        all_bids=all_bids,
        final_state=final_state,
        workspace_collapse=workspace_collapse,
    )
    primary_bid = action_collapse.get("primary_bid")
    supporting_bids = action_collapse.get("supporting_bids", [])
    self_cognition_response_required = (
        is_targetless_group_self_cognition_episode(payload["episode"])
    )
    self_cognition_context = _self_cognition_response_context(
        payload,
        episode_evidence_handles,
    )
    action_question = v3_prompt.ChainQuestion(
        contract_name="action_plan.v1",
        payload=v3_prompt.build_action_plan_question_payload(
            primary_bid_handle=_selected_bid_handle(
                primary_bid,
                bid_handles,
            ),
            supporting_bid_handles=[
                _selected_bid_handle(bid, bid_handles)
                for bid in supporting_bids
            ],
            bid_index=bid_index,
            action_index=action_handles,
            resolver_index=resolver_handles,
            resolver_context=payload["resolver_context"],
            runtime_capability_limits=payload.get(
                "runtime_capability_limits",
                [],
            ),
            current_goal_progress=payload.get("resolver_goal_progress"),
            required_resolver_evidence_dependency=payload.get(
                "required_resolver_evidence_dependency"
            ),
            evidence=payload["evidence"],
            self_cognition_response_context=self_cognition_context,
        ),
    )

    accepted_at_utc = accepted_at_utc_from_episode(
        payload["episode"]
    )

    def action_validator(
        parsed: dict[str, object],
        *,
        bid_handles: Mapping[str, object] = bid_handles,
        action_handles: Mapping[str, object] = action_handles,
        resolver_handles: Mapping[str, object] = resolver_handles,
    ) -> object:
        return validate_action_plan_decision(
            parsed,
            bid_handles=bid_handles,
            action_handles=action_handles,
            resolver_handles=resolver_handles,
            current_goal_progress=payload.get("resolver_goal_progress"),
            required_resolver_evidence_dependency=payload.get(
                "required_resolver_evidence_dependency"
            ),
            runtime_capability_limits=payload.get(
                "runtime_capability_limits",
                [],
            ),
            self_cognition_response_required=self_cognition_response_required,
            evidence=payload["evidence"],
            target_handles=_self_cognition_target_handles(
                payload["scene_context"]
            ),
            accepted_at_utc=accepted_at_utc,
        )

    try:
        action_result = await invoke_serial_question_with_repair(
            harness=harness,
            system_content=system_content,
            llm=services.llm,
            config=_bounded_stage_config(
                services.chain_lane,
                stage_name="P1",
                completion_cap=_ACTION_COMPLETION_CAP,
            ),
            question=action_question,
            validator=action_validator,
            attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
            observation_context=observation_context,
            interludes=() if len(eligible_bids) >= 2 and not relationship_sensitive else (i2_interlude,),
            attempt_owner="serial_action_plan",
            v2_stage="action_planning",
            v2_branch_ids=("action_plan",),
            deterministic_only=services.sidecar_lane is None,
            json_repair_callback=json_repair_callback,
            deadline_monotonic=deadline_monotonic,
        )
        action_decision = action_result.validated
    except CognitionContextLimitError:
        if self_cognition_response_required:
            raise CognitionExecutionError(
                "required self-cognition response was unavailable",
                error_code="self_cognition_response_unavailable",
                stage="action_planning",
                safe_checkpoint="final_reduction",
            )
        action_decision = None
        warnings.append("action_planning_context_limit")

    if self_cognition_response_required and action_decision is None:
        raise CognitionExecutionError(
            "required self-cognition response was unavailable",
            error_code="self_cognition_response_unavailable",
            stage="action_planning",
            safe_checkpoint="final_reduction",
        )
    await _drain_l1_sidecar(
        l1_task,
        invocation_state=sidecar_state,
        warnings=warnings,
    )
    action_decision = await _authorize_serial_action_decision(
        action_decision=action_decision,
        primary_bid=primary_bid,
        bid_handles=bid_handles,
        action_handles=action_handles,
        resolver_handles=resolver_handles,
        payload=payload,
        services=services,
        coordinator=sidecar_coordinator,
        admissions=sidecar_admissions,
        invocation_state=sidecar_state,
        warnings=warnings,
        deadline_monotonic=deadline_monotonic,
    )
    if action_decision is None:
        action_decision = {
            "action_requests": [],
            "resolver_requests": [],
            "goal_resolution": "blocked",
            "resolver_pending_resolution": None,
            "resolver_goal_progress": None,
        }
    primary_bid = action_collapse.get("primary_bid")
    validated_output = _build_serial_output(
        payload=payload,
        previous_state=previous_state,
        final_state=final_state,
        questions=questions,
        appraisal_results=appraisal_results,
        appraisal_failures=appraisal_failures,
        comparison_results=comparison_results,
        preliminary_branches=revision_roster,
        all_bids=all_bids,
        ordinary_bid=ordinary_bid,
        workspace_collapse=workspace_collapse,
        action_decision=action_decision,
        branch_failures=phase_branch_failures,
        harness=harness,
        warnings=warnings,
        stage_status=stage_status,
        started_at=started_at,
    )

    if store_cold_session:
        token_ledger = dict(harness.transcript.token_ledger or {})
        token_ledger.update({
            "active_total_ceiling_tokens": (
                harness.budget.active_total_ceiling_tokens
            ),
            "extension_available": int(harness.budget.extension_available),
            "extension_used": int(harness.budget.extension_used),
            "reanchor_used": int(harness.transcript.reanchor_used),
        })
        cold_session = create_cold_session(
            payload=payload,
            episode_id=episode_id,
            owner_identity=owner_identity,
            accepted_messages=harness.transcript.to_messages(),
            accepted_products=harness.transcript.accepted_products,
            current_roster=tuple(
                branch_id
                for branch_id in sorted(
                    {bid["branch_id"] for bid in eligible_bids},
                    key=branch_order_key,
                )
            ),
            attempt_ledger=dict(harness.ledger._counts),
            token_ledger=token_ledger,
            last_output=validated_output,
            expected_relational_willingness=(
                _expected_relational_willingness_carrier(
                    payload,
                    validated_output,
                )
            ),
            ttl_seconds=_session_ttl_seconds(services),
        )
        _CHAIN_SESSION_REGISTRY.put(cold_session)

    _record_harness_observability(
        harness=harness,
        sidecar_state=sidecar_state,
        session_event="cold",
    )
    return validated_output
