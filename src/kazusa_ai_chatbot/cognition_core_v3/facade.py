"""V3 public cognition engine over the unchanged V2 substrate.

The engine runs one parallel first wave of registry-ordered appraisal chains
plus isolated goal chains for every preliminary branch kind, reduces the
accepted prefix into a provisional state, runs the fresh canonical terminal
outcome stage on that reduction, applies final reduction and relationship
maintenance once, then reactivates dependency-ready branch kinds before the
complete-bid join. Every wave shares one invocation-wide attempt ledger whose
per-stage caps mirror the V2 model attempt policy, so global arithmetic stays
exact while chain failures stay isolated.

The deterministic head (elapsed update, preliminary goals, prompt projection,
question planning), final reduction, relationship maintenance, workspace
collapse, action planning, and output assembly ride on the unchanged V2
substrate helpers, so the public contract remains exactly
``CognitionCoreInputV2``/``CognitionCoreOutputV2``. Goal chains fail closed per
kind with a typed error code; required-branch escalation, partial failure
surfacing, and protected replay capture reuse the V2 behavior verbatim.

Accepted appraisal content bridges into native state through code-owned rows:
propositions attach to their source evidence row's candidate event root so the
native materializer resolves them through causal-event provenance, except
terminal outcome propositions whose subject binds to the unique lifecycle-
eligible entity of the asserted kind. Axis deltas translate into exact native
increment rows only when their axis matches exactly one permitted concrete
path for the stage's authorized evidence domain; every unbound or ambiguous
delta is dropped with a deterministic warning instead of reaching the native
reducer. Role assignments stay empty because the producer contract carries no
role data, recorded as a documented parity gap in the package README.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import replace
from functools import partial
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
    branch_order_key,
    select_final_branches,
    select_preliminary_branches,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CURRENT_EPISODE_EVIDENCE_SOURCE_KINDS,
    ActionBidV2,
    BranchDefinition,
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionExecutionError,
    is_targetless_group_self_cognition_episode,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import (
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
    _reduce_appraisals_with_isolation,
    _resolver_progress,
    _selected_bid,
    _workspace_current_event,
    _workspace_goal_contexts,
)
from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    V2_APPRAISAL_TOTAL_ATTEMPTS,
    V2_MODEL_TOTAL_ATTEMPTS,
    V2AttemptBudgetExhausted,
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    current_v2_attempt_ledger,
    record_v2_attempt_disposition,
    reserve_v2_model_attempt,
    reset_v2_attempt_ledger,
    snapshot_v2_attempt_ledger,
)
from kazusa_ai_chatbot.cognition_core_v2.output_projection import (
    build_state_update,
    default_expression_policy,
    project_affect,
    project_relationship,
)
from kazusa_ai_chatbot.cognition_core_v2.parallel_executor import (
    BranchFailure,
    ParallelExecutionResult,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_state_update,
    create_deterministic_goals,
)
from kazusa_ai_chatbot.cognition_core_v3 import anchor as v3_anchor
from kazusa_ai_chatbot.cognition_core_v3 import prompt as v3_prompt
from kazusa_ai_chatbot.cognition_core_v3.action_selection import (
    _authorization_repair_message,
    _materialize_action_requests,
    _materialize_resolver_requests,
    _self_cognition_target_handles,
    _validate_authorization_decisions,
    apply_stance_suppression,
    authorize_action_requests,
    authorize_resolver_requests,
    derive_action_route,
    settle_resolver_outcome,
    validate_action_plan_decision,
)
from kazusa_ai_chatbot.cognition_core_v3.appraisal import (
    reduce_grouped_appraisal_output,
)
from kazusa_ai_chatbot.cognition_core_v3.budget import (
    ContextBudgetLedger,
    ContextBudgetPlan,
)
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    CognitionChainServicesV3,
)
from kazusa_ai_chatbot.cognition_core_v3.execution import (
    AttemptLedger,
    SerialChainHarness,
    TurnDeadlineExceeded,
    check_turn_deadline,
    config_for_turn_deadline,
    invoke_lane_scoped_json_repair,
    invoke_serial_question_with_repair,
)
from kazusa_ai_chatbot.cognition_core_v3.goal_cognition import (
    GOAL_BID_EVIDENCE_HANDLE_LIMIT,
    ORDINARY_GOAL_KIND,
    materialize_selection_goal_draft,
    project_conversation_progress_evidence,
    project_goal_evidence_row,
    project_required_selection_operations,
    validate_active_goal_group_output,
    validate_goal_bid_draft,
    validate_recurrence_ordinary_goal_bid_draft,
    validate_selection_goal_draft,
)
from kazusa_ai_chatbot.cognition_core_v3.lane import (
    SidecarAdmissionLedger,
    SidecarCoordinator,
    SidecarInvocationState,
    primary_lane_coordinator,
    sidecar_lane_coordinator,
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
    collapse_authoritative_relational_bid,
    collapse_single_bid,
    fallback_partition_envelope,
    materialize_partition,
    prepare_partition,
    validate_complete_bids,
    validate_partition,
)
from kazusa_ai_chatbot.cognition_episode import (
    build_goal_continuation_ref,
    project_dialog_response_operation,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    CURRENT_TURN_RELATIONAL_WILLINGNESS_VERSION,
)
from kazusa_ai_chatbot.config import (
    COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS,
    COGNITION_RESOLVER_MAX_CYCLES,
)
from kazusa_ai_chatbot.llm_tracing import failure_capsule
from kazusa_ai_chatbot.utils import parse_llm_json_output

_PROVIDER_EXCEPTIONS = (
    OpenAIError,
    httpx.HTTPError,
    ConnectionError,
    OSError,
    RuntimeError,
    TimeoutError,
)
_CANONICALIZE_EXCEPTIONS = (AttributeError, KeyError, TypeError, ValueError)

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

_CURRENT_PROTECTED_CHAIN_SCOPE: ContextVar[
    list[dict[str, Any]] | None
] = ContextVar("cognition_v3_protected_chain_records", default=None)

_CHAIN_SESSION_REGISTRY = ChainSessionRegistry()


def bind_protected_chain_records() -> Token:
    """Bind a fresh per-trace protected chain record scope.

    The bound scope owns every trace record appended while it stays active in
    the current execution context; ``run_cognition`` binds one when none is
    already active so concurrent traces keep isolated record sets. Returns the
    reset token for ``reset_protected_chain_records``.
    """

    return _CURRENT_PROTECTED_CHAIN_SCOPE.set([])


def snapshot_protected_chain_records() -> tuple[dict[str, Any], ...]:
    """Return the active scope's protected chain trace records in order.

    Records carry full internal content (raw model output and parsed
    candidates); callers project one record through
    ``project_protected_chain_failure`` or ``project_protected_chain_result``
    before exposing it publicly. Returns an empty tuple when no scope is bound.
    """

    records = _CURRENT_PROTECTED_CHAIN_SCOPE.get()
    return tuple(records) if records is not None else ()


def reset_protected_chain_records(token: Token) -> None:
    """Drop the protected chain record scope bound by ``token``."""

    _CURRENT_PROTECTED_CHAIN_SCOPE.reset(token)


def _goal_semantic_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Filter the branch context into prompt-safe goal-tail content.

    Mirrors the V2 semantic-context contract: private underscore keys and the
    separately rendered fields (evidence, projection, role summaries, appraisal
    summaries) are excluded; character constraints drop their judgment field
    and scene context drops its raw role labels so role facts stay handle-only.
    """
    filtered = {
        key: value
        for key, value in context.items()
        if not str(key).startswith("_")
        and key
        not in {
            "evidence",
            "goal_projection",
            "role_summaries",
            "appraisal_summaries",
        }
    }
    constraints = filtered.get("character_constraints")
    if isinstance(constraints, Mapping):
        filtered["character_constraints"] = {
            key: value
            for key, value in constraints.items()
            if key != "personality_judgment"
        }
    scene_context = filtered.get("scene_context")
    if isinstance(scene_context, Mapping):
        filtered["scene_context"] = {
            key: value
            for key, value in scene_context.items()
            if key not in {"character_role", "current_user_role"}
        }
    return filtered


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
    return materialize_selection_goal_draft(
        validated_draft,
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
            attempt_count=ledger.attempts_used(goal_kind),
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
    sections = v3_prompt.build_first_packet_sections(
        projection_payload=projection.payload,
        scene_context=payload["scene_context"],
        episode=payload["episode"],
        direct_facts=payload["direct_facts"],
        available_actions=payload["available_actions"],
        available_resolver_capabilities=payload[
            "available_resolver_capabilities"
        ],
        resolver_context=payload["resolver_context"],
    )
    return {
        "payload": payload,
        "previous_state": previous_state,
        "preliminary_state": preliminary_state,
        "projection": projection,
        "questions": questions,
        "system_content": system_content,
        "first_sections": sections,
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
Return JSON only.
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
                    replace(sidecar_lane, stage_name="L1"),
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
    if warning is not None:
        warnings.append(warning)
    return residue, True


async def _drain_l1_sidecar(
    task: asyncio.Task[tuple[dict[str, Any] | None, str | None]] | None,
    *,
    invocation_state: SidecarInvocationState | None,
    warnings: list[str],
) -> None:
    """Cancel and drain L1 during owned cleanup after primary work finishes."""

    if task is None:
        return
    if not task.done():
        if invocation_state is not None:
            invocation_state.record_cancellation(task)
        task.cancel()
        warnings.append("sidecar_l1_dropped")
    try:
        _residue, warning = await task
    except asyncio.CancelledError:
        return
    if warning is not None:
        warnings.append(warning)


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
        config=replace(sidecar_lane, stage_name="json_repair"),
        coordinator=coordinator,
        admissions=admissions,
        invocation_state=invocation_state,
        deadline_monotonic=deadline_monotonic,
    )
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
        attempt_config = replace(
            sidecar_lane,
            stage_name=(
                stage_name
                if attempt_number == 1
                else f"{stage_name}.repair"
            ),
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
            decisions = _validate_authorization_decisions(
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
            services=services,
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
            services=services,
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


def _build_serial_step_payloads(
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Build current serial question payloads from the deterministic head."""

    payload = context["payload"]
    projection = context["projection"]
    preliminary_state = context["preliminary_state"]
    evidence_handles = [row["evidence_handle"] for row in payload["evidence"]]
    branch_context = _branch_context(
        projection,
        preliminary_state,
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
    preliminary_branches = select_preliminary_branches(
        preliminary_state["goals"]
    )
    ordinary_definition = next(
        (
            definition
            for definition in preliminary_branches
            if definition.goal_kind == ORDINARY_GOAL_KIND
        ),
        None,
    )
    goal_kind = ORDINARY_GOAL_KIND
    goal = _goal_for_branch(preliminary_state, goal_kind)
    selection_operations = project_required_selection_operations(
        payload["evidence"]
    )
    progress_evidence = (
        project_conversation_progress_evidence(payload["evidence"])
        if selection_operations else []
    )
    role_bindings = branch_context["_role_bindings"]
    role_summaries = branch_context["role_summaries"]
    prompt_role_bindings = {
        handle: {
            "role": binding.get("role", ""),
            "entity_kind": binding.get("entity_kind", ""),
        }
        for handle, binding in role_bindings.items()
    }
    partitioned_handles = (
        {
            operation["evidence_handle"]
            for operation in selection_operations
        }
        | {
            row["evidence_handle"]
            for row in progress_evidence
        }
        if selection_operations
        else set()
    )
    tail_rows = (
        [
            row
            for row in payload["evidence"]
            if row["evidence_handle"] not in partitioned_handles
        ]
        if selection_operations
        else list(payload["evidence"])
    )
    ordinary_payload = v3_prompt.build_goal_question_payload(
        goal_kind=goal_kind,
        goal_projection=_goal_projection(goal, goal_kind),
        evidence_handles=evidence_handles,
        action_tendencies=list(
            ordinary_definition.action_tendencies
            if ordinary_definition is not None
            else ()
        ),
        branch_intent_guidance="",
        role_bindings=prompt_role_bindings,
        role_summaries=role_summaries,
        semantic_context=_goal_semantic_context(branch_context),
        appraisal_summaries=[],
        evidence_rows=[
            project_goal_evidence_row(row)
            for row in tail_rows
        ],
        selection_operations=list(selection_operations),
        progress_evidence=list(progress_evidence),
    )
    active_roster: list[Mapping[str, object]] = [
        {
            "branch_id": definition.branch_id,
            "goal_kind": definition.goal_kind,
            "branch_intent_guidance": (
                definition.branch_intent_guidance
            ),
            "action_tendencies": list(definition.action_tendencies),
        }
        for definition in preliminary_branches
        if definition.goal_kind != ORDINARY_GOAL_KIND
    ]
    workspace_payload = v3_prompt.build_workspace_question_payload(
        bids=[],
        current_event=_workspace_current_event(payload["evidence"]),
        goal_contexts=_workspace_goal_contexts([], preliminary_state),
    )
    action_payload = v3_prompt.build_action_plan_question_payload(
        primary_bid=None,
        supporting_bids=[],
        available_actions=payload["available_actions"],
        available_resolvers=payload["available_resolver_capabilities"],
        resolver_context=payload["resolver_context"],
        runtime_capability_limits=payload.get(
            "runtime_capability_limits",
            [],
        ),
        current_goal_progress=payload.get("resolver_goal_progress"),
        required_resolver_evidence_dependency=payload.get(
            "required_resolver_evidence_dependency"
        ),
    )
    return {
        "ordinary_goal_payload": ordinary_payload,
        "active_branch_roster": active_roster,
        "workspace_payload": workspace_payload,
        "action_plan_payload": action_payload,
    }


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
    """Reduce accepted appraisal rows through the unchanged V2 owners."""

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
    preliminary_execution = _synthesize_branch_execution(
        frozen_preliminary_branches,
        {
            ORDINARY_GOAL_KIND: ordinary_bid,
        } if ordinary_bid is not None else {},
        {},
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
            "completed_branch_count": 1 if ordinary_bid is not None else 0,
            "failed_branch_count": 0 if ordinary_bid is not None else 1,
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
            extension_used=bool(session.token_ledger.get("extension_used")),
            reanchor_used=session.reanchor_used,
        ),
    )


def _revision_roster(
    *,
    session: ChainSessionV1,
    final_state: Mapping[str, Any],
    appraisal_rows: Sequence[Mapping[str, Any]],
) -> list[BranchDefinition]:
    """Build the stable prior-plus-new final roster in V2 registry order."""

    prior: dict[str, BranchDefinition] = {}
    for branch_id in session.current_roster:
        definition = DEFAULT_BRANCH_DEFINITIONS.get(branch_id)
        if definition is None:
            raise SessionContractError("session roster branch is invalid")
        prior[branch_id] = definition
    question_ids = []
    for row in appraisal_rows:
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            continue
        question_ids.append(
            question_id
            if question_id.startswith("q:")
            else f"q:{question_id}"
        )
    final_branches = select_final_branches(
        select_preliminary_branches(final_state["goals"]),
        final_state["goals"],
        question_ids=question_ids,
    )
    for definition in final_branches:
        prior.setdefault(definition.branch_id, definition)
    prior.setdefault(
        ORDINARY_GOAL_KIND,
        DEFAULT_BRANCH_DEFINITIONS[ORDINARY_GOAL_KIND],
    )
    return [
        replace(definition, dependencies=(), dependency_options=())
        for definition in sorted(
            prior.values(),
            key=lambda definition: branch_order_key(definition.branch_id),
        )
    ]


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
    relational_carrier = session.expected_relational_willingness
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
    new_evidence_handle = new_observation["evidence_handle"]
    appraisal_rows: list[dict[str, Any]] = []
    appraisal_failures: dict[str, str] = {}
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
    delta_questions = v3_prompt.build_grouped_appraisal_questions(
        planned_questions=questions,
        group_count=services.appraisal_group_count,
        l1_residue=l1_residue,
    )
    for index, question in enumerate(delta_questions, start=1):
        planned_families = [
            str(row["family"])
            for row in question.payload["questions"]
        ]

        def appraisal_validator(
            parsed: dict[str, object],
            *,
            planned_families: list[str] = planned_families,
        ) -> object:
            return reduce_grouped_appraisal_output(
                parsed,
                planned_families=planned_families,
                questions_by_family=questions_by_family,
                evidence_handles={new_evidence_handle},
                handle_to_ref=projection.handle_to_ref,
            )

        validated, _raw = await invoke_serial_question_with_repair(
            harness=harness,
            system_content=system_content,
            llm=services.llm,
            config=replace(
                services.chain_lane,
                stage_name=f"R.A{index}",
            ),
            question=question,
            validator=appraisal_validator,
            attempt_limit=V2_APPRAISAL_TOTAL_ATTEMPTS,
            interludes=(resolver_interlude,) if index == 1 else (),
            attempt_owner="serial_appraisal",
            v2_stage="semantic_appraisal",
            v2_branch_id=f"resolver_delta_appraisal_{index}",
            deterministic_only=services.sidecar_lane is None,
            json_repair_callback=json_repair_callback,
            deadline_monotonic=deadline_monotonic,
        )
        if validated is None:
            for family_name in planned_families:
                appraisal_failures[f"q:{family_name}"] = (
                    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
                )
            continue
        appraisal_rows.extend(validated)

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
    stage_status["semantic_appraisal"] = "completed"
    stage_status["final_reduction"] = "completed"

    revision_roster = _revision_roster(
        session=session,
        final_state=final_state,
        appraisal_rows=appraisal_rows,
    )
    goal_branch_context = _branch_context(
        projection,
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
    role_summaries = goal_branch_context["role_summaries"]
    prompt_role_bindings = {
        handle: {
            "role": binding.get("role", ""),
            "entity_kind": binding.get("entity_kind", ""),
        }
        for handle, binding in goal_role_bindings.items()
    }
    evidence_handles = [row["evidence_handle"] for row in payload["evidence"]]
    selection_operations = project_required_selection_operations(
        payload["evidence"]
    )
    progress_evidence = (
        project_conversation_progress_evidence(payload["evidence"])
        if selection_operations
        else []
    )
    partitioned_handles = (
        {
            operation["evidence_handle"]
            for operation in selection_operations
        }
        | {
            row["evidence_handle"]
            for row in progress_evidence
        }
        if selection_operations
        else set()
    )
    goal_evidence_rows = [
        project_goal_evidence_row(row)
        for row in payload["evidence"]
        if row["evidence_handle"] not in partitioned_handles
    ]
    ordinary_definition = next(
        definition
        for definition in revision_roster
        if definition.goal_kind == ORDINARY_GOAL_KIND
    )
    ordinary_goal = _goal_for_branch(final_state, ORDINARY_GOAL_KIND)
    goal_l1_residue = None
    if not l1_observed:
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
            role_summaries=role_summaries,
            semantic_context=_goal_semantic_context(goal_branch_context),
            appraisal_summaries=[],
            evidence_rows=goal_evidence_rows,
            selection_operations=list(selection_operations),
            progress_evidence=list(progress_evidence),
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

    ordinary_local, _raw = await invoke_serial_question_with_repair(
        harness=harness,
        system_content=system_content,
        llm=services.llm,
        config=replace(services.chain_lane, stage_name="R.G1a"),
        question=ordinary_question,
        validator=ordinary_validator,
        attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
        attempt_owner="serial_goal_ordinary",
        v2_stage="goal_bid_structure",
        v2_branch_id=ORDINARY_GOAL_KIND,
        deterministic_only=services.sidecar_lane is None,
        json_repair_callback=json_repair_callback,
        deadline_monotonic=deadline_monotonic,
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
    if ordinary_bid is None:
        raise CognitionExecutionError(
            "ordinary_response",
            error_code="ordinary_response_unavailable",
            stage="branch_cognition",
            safe_checkpoint="final_reduction",
        )
    if selection_operations:
        ordinary_bid["relational_willingness"] = dict(
            carried_relational_willingness
        )
    all_bids: list[ActionBidV2] = [ordinary_bid]

    active_roster = [
        {
            "branch_id": definition.branch_id,
            "goal_kind": definition.goal_kind,
            "branch_intent_guidance": definition.branch_intent_guidance,
            "action_tendencies": list(definition.action_tendencies),
        }
        for definition in revision_roster
        if definition.goal_kind != ORDINARY_GOAL_KIND
    ]
    if active_roster:
        active_question = v3_prompt.ChainQuestion(
            contract_name="active_goal_bid_group.v1",
            payload=v3_prompt.build_active_goal_group_question_payload(
                roster=active_roster,
                evidence_handles=evidence_handles,
                semantic_context=_goal_semantic_context(goal_branch_context),
            ),
        )
        active_rows, _raw = await invoke_serial_question_with_repair(
            harness=harness,
            system_content=system_content,
            llm=services.llm,
            config=replace(services.chain_lane, stage_name="R.G1b"),
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
            v2_branch_id="active_goal_group",
            deterministic_only=services.sidecar_lane is None,
            json_repair_callback=json_repair_callback,
            deadline_monotonic=deadline_monotonic,
        )
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
    }

    workspace_collapse: dict[str, Any] | None = None
    i2_consumed_by_workspace = False
    if len(live_bids) >= 2 and not relationship_sensitive:
        ordered_bids = validate_complete_bids(live_bids)
        partition_request = prepare_partition(
            ordered_bids,
            _workspace_current_event(payload["evidence"]),
            _workspace_goal_contexts(ordered_bids, final_state),
        )
        workspace_question = v3_prompt.ChainQuestion(
            contract_name="workspace_partition.v1",
            payload=v3_prompt.build_workspace_question_payload(
                bids=ordered_bids,
                current_event=_workspace_current_event(payload["evidence"]),
                goal_contexts=_workspace_goal_contexts(
                    ordered_bids,
                    final_state,
                ),
            ),
        )
        partition, _raw = await invoke_serial_question_with_repair(
            harness=harness,
            system_content=system_content,
            llm=services.llm,
            config=replace(services.chain_lane, stage_name="R.W1"),
            question=workspace_question,
            validator=partial(
                validate_partition,
                handles=partition_request.handles,
            ),
            attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
            interludes=(i2_interlude,),
            attempt_owner="serial_workspace",
            v2_stage="workspace_collapse",
            v2_branch_id="workspace",
            deterministic_only=services.sidecar_lane is None,
            json_repair_callback=json_repair_callback,
            deadline_monotonic=deadline_monotonic,
        )
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
    action_question = v3_prompt.ChainQuestion(
        contract_name="action_plan.v1",
        payload=v3_prompt.build_action_plan_question_payload(
            primary_bid=primary_bid,
            supporting_bids=supporting_bids,
            available_actions=payload["available_actions"],
            available_resolvers=payload["available_resolver_capabilities"],
            resolver_context=payload["resolver_context"],
            runtime_capability_limits=payload.get(
                "runtime_capability_limits",
                [],
            ),
            current_goal_progress=payload.get("resolver_goal_progress"),
            required_resolver_evidence_dependency=payload.get(
                "required_resolver_evidence_dependency"
            ),
        ),
    )
    bid_handles = {
        f"b{index}": bid
        for index, bid in enumerate(
            sorted(
                eligible_bids,
                key=lambda bid: branch_order_key(bid["branch_id"]),
            ),
            start=1,
        )
    }
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
            self_cognition_response_required=is_targetless_group_self_cognition_episode(
                payload["episode"]
            ),
            evidence=payload["evidence"],
            target_handles=_self_cognition_target_handles(
                payload["scene_context"]
            ),
            accepted_at_utc="",
        )

    action_decision, _raw = await invoke_serial_question_with_repair(
        harness=harness,
        system_content=system_content,
        llm=services.llm,
        config=replace(services.chain_lane, stage_name="R.P1"),
        question=action_question,
        validator=action_validator,
        attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
        interludes=() if i2_consumed_by_workspace else (i2_interlude,),
        attempt_owner="serial_action_plan",
        v2_stage="action_planning",
        v2_branch_id="action_plan",
        deterministic_only=services.sidecar_lane is None,
        json_repair_callback=json_repair_callback,
        deadline_monotonic=deadline_monotonic,
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
        harness=harness,
        warnings=warnings,
        stage_status=stage_status,
        started_at=started_at,
    )
    token_ledger = dict(session.token_ledger)
    token_ledger["extension_used"] = int(harness.budget.extension_used)
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
    return validated_output

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
    records_token = None
    if _CURRENT_PROTECTED_CHAIN_SCOPE.get() is None:
        records_token = bind_protected_chain_records()
    try:
        session = failure_capsule.begin_failure_capsule(
            trace_id=llm_tracing.current_trace_id(),
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
            raise

        _mark_cognition_partial_failures(session, output)
        failure_capsule.finish_failure_capsule(
            session,
            outcome=None,
            attempt_ledger=snapshot_v2_attempt_ledger(),
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
    return (
        sidecar_lane_coordinator(services.llm, services.sidecar_lane),
        SidecarAdmissionLedger(),
        SidecarInvocationState(),
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
                            )
                    except SessionContractError:
                        cold_warnings.append("session_rebuilt_session_invalid")
                else:
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
) -> CognitionCoreOutputV2:
    """Run one cold primary sequence while its FIFO lane remains owned."""

    payload = context["payload"]
    previous_state = context["previous_state"]
    preliminary_state = context["preliminary_state"]
    projection = context["projection"]
    questions = context["questions"]
    system_content = context["system_content"]
    first_packet_sections = context["first_sections"]
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
    step_payloads = _build_serial_step_payloads(context)
    goal_branch_context = _branch_context(
        projection,
        preliminary_state,
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
    preliminary_branches = [
        replace(definition, dependencies=(), dependency_options=())
        for definition in select_preliminary_branches(
            preliminary_state["goals"]
        )
    ]
    definitions_by_kind = {
        definition.goal_kind: definition
        for definition in preliminary_branches
    }
    sequence = v3_prompt.build_serial_question_sequence(
        planned_questions=questions,
        group_count=services.appraisal_group_count,
        ordinary_goal_payload=step_payloads["ordinary_goal_payload"],
        active_branch_roster=step_payloads["active_branch_roster"],
        workspace_payload=step_payloads["workspace_payload"],
        action_plan_payload=step_payloads["action_plan_payload"],
    )

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
    )

    appraisal_rows: list[dict[str, Any]] = []
    appraisal_failures: dict[str, str] = {}
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
    l1_observed = False
    all_bids: list[ActionBidV2] = []
    workspace_collapse: dict[str, Any] | None = None
    bid_handles: dict[str, ActionBidV2] = {}
    action_handles: dict[str, Mapping[str, Any]] = {}
    resolver_handles: dict[str, Mapping[str, Any]] = {}
    action_decision: Mapping[str, Any] | None = None

    for step_id, question in sequence:
        if step_id in {"A1", "A2"}:
            if step_id == "A1":
                l1_residue, l1_observed = _take_ready_l1_residue(
                    l1_task,
                    warnings,
                )
                if l1_residue is not None:
                    question = v3_prompt.ChainQuestion(
                        contract_name=question.contract_name,
                        payload={
                            **question.payload,
                            "l1_residue": l1_residue,
                        },
                    )
            planned_families = [
                str(row["family"])
                for row in question.payload["questions"]
            ]

            def appraisal_validator(
                parsed: dict[str, object],
                *,
                planned_families: list[str] = planned_families,
            ) -> object:
                return reduce_grouped_appraisal_output(
                    parsed,
                    planned_families=planned_families,
                    questions_by_family=questions_by_family,
                    evidence_handles=evidence_handles,
                    handle_to_ref=projection.handle_to_ref,
                )

            validated, _raw = await invoke_serial_question_with_repair(
                harness=harness,
                system_content=system_content,
                llm=services.llm,
                config=replace(services.chain_lane, stage_name=step_id),
                question=question,
                validator=appraisal_validator,
                attempt_limit=V2_APPRAISAL_TOTAL_ATTEMPTS,
                first_packet_sections=first_packet_sections,
                attempt_owner="serial_appraisal",
                v2_stage="semantic_appraisal",
                v2_branch_id=f"cold_appraisal_{step_id}",
                deterministic_only=services.sidecar_lane is None,
                json_repair_callback=json_repair_callback,
                deadline_monotonic=deadline_monotonic,
            )
            if validated is None:
                for family_name in planned_families:
                    appraisal_failures[f"q:{family_name}"] = (
                        APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
                    )
                continue
            appraisal_rows.extend(validated)

        elif step_id == "G1a":
            goal_l1_residue = None
            if not l1_observed:
                goal_l1_residue, l1_observed = _take_ready_l1_residue(
                    l1_task,
                    warnings,
                )
                if not l1_observed and l1_task is not None:
                    if sidecar_state is not None:
                        sidecar_state.record_cancellation(l1_task)
                    l1_task.cancel()
                    warnings.append("sidecar_l1_dropped")
            if goal_l1_residue is not None:
                question = v3_prompt.ChainQuestion(
                    contract_name=question.contract_name,
                    payload={
                        **question.payload,
                        "l1_residue": goal_l1_residue,
                    },
                )
            selection_operations = step_payloads["ordinary_goal_payload"][
                "selection_operations"
            ]
            role_handles = set(goal_role_bindings)

            def ordinary_validator(
                parsed: dict[str, object],
                *,
                selection_operations: list[Mapping[str, object]] = selection_operations,
                role_handles: set[str] = role_handles,
            ) -> object:
                if selection_operations:
                    return _validate_and_materialize_selection_goal_draft(
                        parsed,
                        evidence_handles=set(evidence_handles),
                        role_handles=role_handles,
                        selection_operations=selection_operations,
                        episode_handles=episode_evidence_handles,
                        require_relational_willingness=True,
                    )
                return validate_goal_bid_draft(
                    parsed,
                    goal_kind=ORDINARY_GOAL_KIND,
                    evidence_handles=set(evidence_handles),
                    role_handles=role_handles,
                )

            ordinary_local, _raw = await invoke_serial_question_with_repair(
                harness=harness,
                system_content=system_content,
                llm=services.llm,
                config=replace(services.chain_lane, stage_name=step_id),
                question=question,
                validator=ordinary_validator,
                attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
                first_packet_sections=first_packet_sections,
                attempt_owner="serial_goal_ordinary",
                v2_stage="goal_bid_structure",
                v2_branch_id=ORDINARY_GOAL_KIND,
                deterministic_only=services.sidecar_lane is None,
                json_repair_callback=json_repair_callback,
                deadline_monotonic=deadline_monotonic,
            )
            ordinary_bid = (
                _materialize_goal_bid(
                    definitions_by_kind[ORDINARY_GOAL_KIND],
                    preliminary_state,
                    goal_role_bindings,
                    ordinary_local,
                )
                if ordinary_local is not None
                else None
            )
            if ordinary_bid is None:
                raise CognitionExecutionError(
                    "ordinary_response",
                    error_code="ordinary_response_unavailable",
                    stage="branch_cognition",
                    safe_checkpoint="final_reduction",
                )
            all_bids.append(ordinary_bid)

        elif step_id == "G1b":
            roster = step_payloads["active_branch_roster"]
            if not roster:
                continue
            roster_value = roster
            evidence_handles_value = set(evidence_handles)
            role_handles_value = set(goal_role_bindings)
            active_validator = partial(
                _validate_active_goal_group,
                roster=roster_value,
                evidence_handles=evidence_handles_value,
                role_handles=role_handles_value,
            )

            active_rows, _raw = await invoke_serial_question_with_repair(
                harness=harness,
                system_content=system_content,
                llm=services.llm,
                config=replace(services.chain_lane, stage_name=step_id),
                question=question,
                validator=active_validator,
                attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
                first_packet_sections=first_packet_sections,
                attempt_owner="serial_goal_active",
                v2_stage="goal_bid_structure",
                v2_branch_id="active_goal_group",
                deterministic_only=services.sidecar_lane is None,
                json_repair_callback=json_repair_callback,
                deadline_monotonic=deadline_monotonic,
            )
            if active_rows is None:
                warnings.append("v3_chain_unavailable:active_goal_group")
                continue
            for row in active_rows:
                branch_id = row["branch_id"]
                definition = next(
                    definition
                    for definition in preliminary_branches
                    if definition.branch_id == branch_id
                )
                all_bids.append(
                    _materialize_goal_bid(
                        definition,
                        preliminary_state,
                        goal_role_bindings,
                        row,
                    )
                )

        elif step_id == "W1":
            relational_decision = _ordinary_relational_decision(all_bids)
            relationship_sensitive = (
                relational_decision is not None
                and relational_decision["applicability"]
                == "relationship_sensitive"
            )
            if len(all_bids) < 2 or relationship_sensitive:
                continue
            ordered_bids = validate_complete_bids(all_bids)
            partition_request = prepare_partition(
                ordered_bids,
                _workspace_current_event(payload["evidence"]),
                _workspace_goal_contexts(
                    ordered_bids,
                    preliminary_state,
                ),
            )
            workspace_question = v3_prompt.ChainQuestion(
                contract_name="workspace_partition.v1",
                payload=v3_prompt.build_workspace_question_payload(
                    bids=ordered_bids,
                    current_event=_workspace_current_event(
                        payload["evidence"]
                    ),
                    goal_contexts=_workspace_goal_contexts(
                        ordered_bids,
                        preliminary_state,
                    ),
                ),
            )

            def workspace_validator(
                parsed: dict[str, object],
                *,
                handles: Mapping[str, Mapping[str, object]] = partition_request.handles,
            ) -> object:
                return validate_partition(parsed, handles)

            partition, _raw = await invoke_serial_question_with_repair(
                harness=harness,
                system_content=system_content,
                llm=services.llm,
                config=replace(services.chain_lane, stage_name=step_id),
                question=workspace_question,
                validator=workspace_validator,
                attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
                first_packet_sections=first_packet_sections,
                attempt_owner="serial_workspace",
                v2_stage="workspace_collapse",
                v2_branch_id="workspace",
                deterministic_only=services.sidecar_lane is None,
                json_repair_callback=json_repair_callback,
                deadline_monotonic=deadline_monotonic,
            )
            workspace_collapse = (
                materialize_partition(
                    partition_request.handles,
                    partition,
                )
                if partition is not None
                else fallback_partition_envelope(ordered_bids)
            )

        elif step_id == "P1":
            bid_handles = {
                "b1": ordinary_bid,
            } if ordinary_bid is not None else {}
            action_handles = {
                f"a{index}": dict(row)
                for index, row in enumerate(
                    payload["available_actions"],
                    start=1,
                )
            }
            resolver_handles = {
                f"r{index}": dict(row)
                for index, row in enumerate(
                    payload["available_resolver_capabilities"],
                    start=1,
                )
            }

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
                    current_goal_progress=payload.get(
                        "resolver_goal_progress"
                    ),
                    required_resolver_evidence_dependency=payload.get(
                        "required_resolver_evidence_dependency"
                    ),
                    runtime_capability_limits=payload.get(
                        "runtime_capability_limits",
                        [],
                    ),
                    self_cognition_response_required=is_targetless_group_self_cognition_episode(
                        payload["episode"]
                    ),
                    evidence=payload["evidence"],
                    target_handles=_self_cognition_target_handles(
                        payload["scene_context"]
                    ),
                    accepted_at_utc="",
                )

            action_decision, _raw = await invoke_serial_question_with_repair(
                harness=harness,
                system_content=system_content,
                llm=services.llm,
                config=replace(services.chain_lane, stage_name=step_id),
                question=question,
                validator=action_validator,
                attempt_limit=V2_MODEL_TOTAL_ATTEMPTS,
                first_packet_sections=first_packet_sections,
                attempt_owner="serial_action_plan",
                v2_stage="action_planning",
                v2_branch_id="action_plan",
                deterministic_only=services.sidecar_lane is None,
                json_repair_callback=json_repair_callback,
                deadline_monotonic=deadline_monotonic,
            )

    # Canonical post-processing for the common one-bid path.
    (
        final_state,
        appraisal_results,
        reduction_failures,
        comparison_results,
        accepted_relationship_deltas,
    ) = _reduce_appraisals_with_isolation(
        preliminary_state,
        appraisal_rows,
        payload["evidence"],
        projection.handle_to_ref,
        updated_at=updated_at,
        character_constraints=payload["character_constraints"],
        relationship_context=reducer_relationship_context,
    )
    appraisal_failures.update(reduction_failures)
    warnings.extend(
        f"semantic_appraisal_failed:{error_code}"
        for error_code in reduction_failures.values()
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
    stage_status["semantic_appraisal"] = "completed"
    stage_status["final_reduction"] = "completed"

    eligible_bids, stale_branch_ids = _bids_with_live_goals(
        all_bids if all_bids else ([ordinary_bid] if ordinary_bid is not None else []),
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
    preliminary_branches = [
        replace(definition, dependencies=(), dependency_options=())
        for definition in select_preliminary_branches(
            preliminary_state["goals"]
        )
    ]
    preliminary_bids_by_kind = {
        definition.goal_kind: bid
        for definition in preliminary_branches
        for bid in all_bids
        if bid["branch_id"] == definition.branch_id
    }
    preliminary_execution = _synthesize_branch_execution(
        preliminary_branches,
        preliminary_bids_by_kind,
        {},
        harness.ledger,
    )
    cognition_observability = _build_cognition_observability(
        questions=questions,
        appraisal_results=appraisal_results,
        appraisal_failures=appraisal_failures,
        preliminary_branches=preliminary_branches,
        preliminary_execution=preliminary_execution,
        final_branches=[],
        final_execution=None,
        collapse=collapse,
        selected_bid_reason=selected_bid_reason,
        diagnostics={
            "selected_question_count": len(questions),
            "dispatched_question_count": len(questions),
            "selected_branch_count": len(preliminary_branches),
            "dispatched_branch_count": len(preliminary_branches),
            "completed_branch_count": len(preliminary_bids_by_kind),
            "failed_branch_count": 0,
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
    validated_output = validate_cognition_core_output(output)

    if store_cold_session:
        token_ledger = {
            "extension_used": int(harness.budget.extension_used),
        }
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

    return validated_output
