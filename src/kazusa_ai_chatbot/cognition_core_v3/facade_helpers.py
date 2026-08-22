"""Canonical V3 deterministic state and observability helper owners."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot.cognition_core_v3.execution_types import (
    BranchFailure,
    ParallelExecutionResult,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    BranchDefinition,
    ActionBidV2,
    CognitionContractError,
    CognitionCoreOutputV2,
    CognitionExecutionError,
    CognitionObservabilityV2,
    GroupEngagementActionContextV2,
    RelationalWillingnessV2,
    SemanticAppraisalResultV2,
    ROLE_ENTITY_KINDS,
    validate_action_bid,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    record_v2_branch_disposition,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    CognitionStateError,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_shared.state_reducers import (
    apply_relationship_maintenance,
    apply_semantic_appraisals,
    apply_state_update,
    create_deterministic_goals,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_PENDING_RESOLUTION_VERSION,
    ResolverValidationError,
    validate_resolver_pending_resolution,
    validate_resolver_pending_resume,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime
from kazusa_ai_chatbot.llm_tracing import failure_capsule

MAX_APPRAISAL_REJECTION_ERROR_CHARS = 500
AUTHORITATIVE_RELATIONAL_COLLAPSE_REASON = (
    "authoritative_relational_stance_preserved_ordinary_response"
)
_VALIDATION_EVENTS: ContextVar[list[dict[str, object]] | None] = ContextVar(
    "cognition_core_v3_validation_events", default=None
)

def capture_validation_event(event_id: str, payload: Mapping[str, object]) -> None:
    """Capture bounded local validation metadata for the current V3 trace."""

    events = _VALIDATION_EVENTS.get()
    if events is not None:
        events.append({"event_id": event_id, "payload": dict(payload)})

def _deduplicate_diagnostics_warnings(
    warnings: Sequence[str],
) -> list[str]:
    """Keep diagnostic warnings in first-seen order without duplicates."""

    unique_warnings: list[str] = []
    for warning in warnings:
        if warning not in unique_warnings:
            unique_warnings.append(warning)
    return unique_warnings

def _mark_cognition_partial_failures(
    session: failure_capsule.FailureCapsuleSession | None,
    output: CognitionCoreOutputV2,
) -> None:
    """Mark surfaced appraisal and branch failures from validated output."""

    observability = output["cognition_observability"]
    failed_appraisals = [
        {
            "question_kind": appraisal["question_kind"],
            "failure_code": appraisal.get("failure_code", ""),
        }
        for appraisal in observability["appraisals"]
        if appraisal["status"] == "failed"
    ]
    if failed_appraisals:
        failure_capsule.mark_failure(
            session,
            failure_kind="failed_appraisal",
            stage_name="semantic_appraisal",
            details={"appraisals": failed_appraisals},
        )

    failed_branches = [
        {
            "phase": branch["phase"],
            "branch_index": branch["branch_index"],
            "goal_kind": branch["goal_kind"],
            "failure_code": branch.get("failure_code", ""),
        }
        for branch in observability["branches"]
        if branch["status"] == "failed"
    ]
    if failed_branches:
        failure_capsule.mark_failure(
            session,
            failure_kind="failed_branch",
            stage_name="goal_cognition",
            details={"branches": failed_branches},
        )

def _raise_for_unrecoverable_required_branch_failures(
    execution: ParallelExecutionResult,
    definitions: Sequence[BranchDefinition],
) -> None:
    """Continue complete sibling bids or escalate an unrecoverable failure."""

    required_failures = sorted(
        definition.branch_id
        for definition in definitions
        if (
            definition.required
            and definition.branch_id in execution.failed_branch_ids
        )
    )
    if not required_failures:
        return

    complete_sibling_exists = False
    for bid in execution.results.values():
        try:
            validate_action_bid(bid)
        except CognitionContractError:
            continue
        complete_sibling_exists = True
        break
    if complete_sibling_exists:
        for branch_id in required_failures:
            warning = f"required_branch_recovered_by_valid_bid:{branch_id}"
            if warning not in execution.warnings:
                execution.warnings.append(warning)
            failure = execution.failure_records.get(branch_id)
            record_v2_branch_disposition(
                branch_id=branch_id,
                disposition="recovered_by_sibling",
                error_code=(failure.error_code if failure is not None else ""),
            )
        return

    failed_names = ", ".join(required_failures)
    primary_failure = execution.failure_records.get(required_failures[0])
    record_v2_branch_disposition(
        branch_id=required_failures[0],
        disposition="exhausted",
        error_code=(
            primary_failure.error_code
            if primary_failure is not None
            else "internal_invariant"
        ),
    )
    error = CognitionExecutionError(
        f"required cognition branch failed: {failed_names}",
        error_code=(
            primary_failure.error_code
            if primary_failure is not None
            else "internal_invariant"
        ),
        branch_id=(
            primary_failure.branch_id
            if primary_failure is not None
            else required_failures[0]
        ),
        stage=(
            primary_failure.stage
            if primary_failure is not None
            else "cognition_branch"
        ),
        attempt_count=(
            primary_failure.attempt_count
            if primary_failure is not None
            else 1
        ),
        safe_checkpoint=(
            primary_failure.safe_checkpoint
            if primary_failure is not None
            else "unknown"
        ),
        retryable=(
            primary_failure.retryable
            if primary_failure is not None
            else False
        ),
    )
    if primary_failure is not None and primary_failure.exception is not None:
        raise error from primary_failure.exception
    raise error

def _resolver_progress(
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Describe whether this result requires an episode-local resolver cycle."""

    if not requests:
        return {
            "status": "not_requested",
            "semantic_summary": "没有选择知识解析能力",
        }
    capabilities = sorted({
        str(request.get("capability", ""))
        for request in requests
        if request.get("capability")
    })
    capability_text = ", ".join(capabilities) or "bounded resolver work"
    return {
        "status": "pending",
        "semantic_summary": f"resolver evidence requested: {capability_text}",
    }

def _bind_pending_resolution(
    semantic_choice: object,
    pending_resume: object,
) -> dict[str, Any] | None:
    """Bind one model-owned decision to the active deterministic pending row."""

    if semantic_choice is None:
        return_value = None
        return return_value
    if not isinstance(semantic_choice, Mapping):
        raise CognitionExecutionError("pending resolution choice is invalid")
    if pending_resume is None:
        raise CognitionExecutionError(
            "pending resolution selected without an active pending row"
        )
    try:
        validated_pending = validate_resolver_pending_resume(pending_resume)
    except ResolverValidationError as exc:
        raise CognitionExecutionError(
            f"active pending resolver row is invalid: {exc}"
        ) from exc
    if validated_pending["status"] not in {
        "waiting_for_user",
        "waiting_for_approval",
    }:
        raise CognitionExecutionError("pending resolution targets a closed row")
    resolution = {
        "schema_version": RESOLVER_PENDING_RESOLUTION_VERSION,
        "resume_id": validated_pending["resume_id"],
        "decision": semantic_choice["decision"],
        "reason": semantic_choice["reason"],
    }
    try:
        validated_resolution = validate_resolver_pending_resolution(resolution)
    except ResolverValidationError as exc:
        raise CognitionExecutionError(
            f"pending resolution choice is invalid: {exc}"
        ) from exc
    return_value = dict(validated_resolution)
    return return_value

def _branch_context(
    projection: Any,
    state: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    appraisal_results: Sequence[Mapping[str, Any]] = (),
    *,
    scene_context: Mapping[str, Any],
    private_continuity_context: str,
    past_dialog_cognition_context: str = "",
    group_engagement_action_context: (
        GroupEngagementActionContextV2 | None
    ) = None,
) -> dict[str, Any]:
    """Build semantic branch context and retain handle bindings privately."""

    if group_engagement_action_context is None:
        group_engagement_action_context = {
            "engagement_guidelines": [],
            "confidence": "",
        }
    context = dict(projection.payload)
    role_bindings: dict[str, dict[str, str]] = {}
    role_summaries: dict[str, str] = {}
    for handle, ref in projection.handle_to_ref.items():
        role_summaries[handle] = _role_summary(
            handle,
            ref,
            scene_context=scene_context,
        )
        if handle == "self":
            role_bindings[handle] = {
                "role": "actor",
                "entity_kind": "character",
                "entity_id": "character:global",
            }
        elif handle == "current_user" and state.get("owner_user_id"):
            role_bindings[handle] = {
                "role": "target",
                "entity_kind": "user",
                "entity_id": state["owner_user_id"],
            }
        elif ref["kind"] in ROLE_ENTITY_KINDS:
            role_bindings[handle] = {
                "role": _role_label(handle, ref),
                "entity_kind": ref["kind"],
                "entity_id": ref["entity_id"],
            }
    context["role_summaries"] = role_summaries
    context["_role_bindings"] = role_bindings
    context["appraisal_summaries"] = [
        {
            "question_id": result["question_id"],
            "explanation": result["explanation"],
            "propositions": [
                proposition["semantic_value"]
                for proposition in result["propositions"]
            ],
        }
        for result in appraisal_results
    ]
    context["scene_context"] = dict(scene_context)
    context["private_continuity_context"] = private_continuity_context
    context["past_dialog_cognition_context"] = (
        past_dialog_cognition_context
    )
    context["group_engagement_action_context"] = {
        "engagement_guidelines": list(
            group_engagement_action_context["engagement_guidelines"]
        ),
        "confidence": group_engagement_action_context["confidence"],
    }
    del evidence
    return context

def _workspace_current_event(
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Project authoritative current episode evidence for bid relevance."""

    current_event = [
        {
            "handle": str(row["evidence_handle"]),
            "source_kind": str(row["evidence_ref"]["source_kind"]),
            "semantic_text": str(row["semantic_text"]),
        }
        for row in evidence
        if row["evidence_ref"]["source_kind"] == "episode"
    ]
    return current_event

def _bids_with_live_goals(
    bids: Sequence[ActionBidV2],
    state: Mapping[str, Any],
) -> tuple[list[ActionBidV2], list[str]]:
    """Keep bids whose non-ordinary persistent goal still exists in state."""

    live_goal_ids = {
        str(goal["entity_id"])
        for goal in state["goals"]
    }
    retained_bids: list[ActionBidV2] = []
    dropped_branch_ids: list[str] = []
    for bid in bids:
        if bid["branch_id"] == "ordinary_response":
            retained_bids.append(bid)
            continue
        goal_id = str(bid["goal_ref"]["entity_id"])
        if goal_id in live_goal_ids:
            retained_bids.append(bid)
            continue
        dropped_branch_ids.append(bid["branch_id"])
    return retained_bids, dropped_branch_ids

def _workspace_goal_contexts(
    bids: Sequence[ActionBidV2],
    state: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    """Resolve each non-ordinary bid to bounded persistent-goal provenance."""

    goals_by_id = {
        str(goal["entity_id"]): goal
        for goal in state["goals"]
    }
    goal_contexts: dict[str, dict[str, object]] = {}
    for bid in bids:
        if bid["branch_id"] == "ordinary_response":
            continue
        goal_id = bid["goal_ref"]["entity_id"]
        goal = goals_by_id[goal_id]
        goal_contexts[goal_id] = {
            "goal_handle": goal_id,
            "goal_kind": goal["goal_kind"],
            "description": goal["description"],
            "status": goal["status"],
            "salience": goal["salience"],
            "importance": goal["importance"],
            "progress": goal["progress"],
            "obstruction": goal["obstruction"],
            "urgency": goal["urgency"],
        }
    return goal_contexts

def _goal_for_branch(
    state: Mapping[str, Any],
    goal_kind: str,
) -> Mapping[str, Any] | None:
    """Choose the stable active goal for one branch."""

    return next(
        (
            goal for goal in state["goals"]
            if goal.get("goal_kind") == goal_kind
            and goal.get("status") in {"pursuing", "blocked"}
        ),
        None,
    )

def _goal_projection(goal: Mapping[str, Any] | None, goal_kind: str) -> dict[str, str]:
    """Project a goal into semantic branch context without ids or numbers."""

    if goal is None:
        return {"goal_kind": goal_kind, "lifecycle": "本事件中的普通回应"}
    lifecycle_labels = {
        "pursuing": "进行中",
        "blocked": "受阻，等待解决",
        "satisfied": "已完成",
        "failed": "失败，等待恢复",
        "abandoned": "已放下",
    }
    return {
        "goal_kind": str(goal["goal_kind"]),
        "description": str(goal["description"]),
        "lifecycle": lifecycle_labels.get(
            str(goal["status"]),
            str(goal["status"]),
        ),
    }

def _role_summary(
    handle: str,
    ref: Mapping[str, str],
    *,
    scene_context: Mapping[str, Any] | None = None,
) -> str:
    """Describe a local role handle without exposing its backing identity."""

    if handle == "self":
        return _scene_role_label(scene_context, "character_role", "当前角色")
    if handle == "current_user":
        return _scene_role_label(scene_context, "current_user_role", "当前用户")
    if handle.startswith("p") and isinstance(scene_context, Mapping):
        bindings = scene_context.get("participant_bindings")
        if isinstance(bindings, list):
            for binding in bindings:
                if not isinstance(binding, Mapping):
                    continue
                if binding.get("handle") != handle:
                    continue
                display_name = binding.get("display_name")
                if isinstance(display_name, str) and display_name.strip():
                    return f"{handle}={display_name.strip()}（群聊其他参与者）"
        return f"{handle}=群聊其他参与者"
    if handle.startswith("ce"):
        return "当前事件候选"
    if handle.startswith("ct"):
        return "当前威胁候选"
    if handle.startswith("ck"):
        return "当前知识缺口候选"
    kind_labels = {
        "goal": "目标",
        "threat": "威胁",
        "event": "事件",
        "knowledge_gap": "知识缺口",
        "relationship": "关系",
        "drive": "驱动力",
        "standard": "规范",
        "meaning": "意义",
    }
    return f"当前{kind_labels.get(ref['kind'], '语义')}上下文"

def _scene_role_label(
    scene_context: Mapping[str, Any] | None,
    field_name: str,
    fallback: str,
) -> str:
    """Read a bounded Chinese role label from the scene projection."""

    if isinstance(scene_context, Mapping):
        value = scene_context.get(field_name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback

def _role_label(handle: str, ref: Mapping[str, str]) -> str:
    """Choose a semantic role label for an internal target binding."""

    if handle == "self":
        return "actor"
    if handle == "current_user":
        return "target"
    return {
        "goal": "affected_goal",
        "relationship": "affected_relationship",
        "user": "target",
        "character": "actor",
        "third_party": "target",
    }.get(ref["kind"], "object")

def _selected_bid(
    intention: Mapping[str, Any],
    primary: ActionBidV2 | None,
    supporting: Sequence[ActionBidV2],
) -> ActionBidV2 | None:
    """Copy the exact bid selected by route selection."""

    selected_id = intention.get("selected_branch_id")
    for bid in [primary, *supporting]:
        if bid is not None and bid["branch_id"] == selected_id:
            return bid
    if selected_id is None:
        return None
    raise CognitionExecutionError("route selected a bid outside the admitted set")

def _ordinary_relational_decision(
    bids: Sequence[ActionBidV2],
) -> RelationalWillingnessV2 | None:
    """Copy the ordinary branch's exact typed stance without arbitration."""

    for bid in bids:
        if bid["branch_id"] != "ordinary_response":
            continue
        decision = bid.get("relational_willingness")
        if isinstance(decision, Mapping):
            return decision  # type: ignore[return-value]
    return None

def _build_cognition_observability(
    *,
    questions: Sequence[Mapping[str, Any]],
    appraisal_results: Sequence[Mapping[str, Any]],
    appraisal_failures: Mapping[str, str],
    preliminary_branches: Sequence[BranchDefinition],
    preliminary_execution: ParallelExecutionResult,
    final_branches: Sequence[BranchDefinition],
    final_execution: ParallelExecutionResult | None,
    collapse: Mapping[str, Any],
    selected_bid_reason: str,
    diagnostics: Mapping[str, Any],
    relational_willingness: RelationalWillingnessV2 | None,
) -> CognitionObservabilityV2:
    """Build semantic branch evidence without exposing prompt-local handles."""

    appraisal_by_question = {
        result["question_id"]: result
        for result in appraisal_results
        if isinstance(result, Mapping)
    }
    appraisals: list[dict[str, Any]] = []
    for question in questions:
        question_id = question["question_id"]
        result = appraisal_by_question.get(question_id)
        observation: dict[str, Any] = {
            "question_kind": question["question_kind"],
            "semantic_question": question["semantic_question"],
            "status": (
                "failed"
                if question_id in appraisal_failures
                else "completed" if result is not None else "not_reported"
            ),
        }
        if result is not None:
            observation["explanation"] = result["explanation"]
            observation["propositions"] = [
                {
                    "proposition_kind": proposition["proposition_kind"],
                    "semantic_value": proposition["semantic_value"],
                }
                for proposition in result["propositions"]
            ]
            observation["deltas"] = [
                {
                    "delta": delta["delta"],
                    "reason": delta["reason"],
                }
                for delta in result["deltas"]
            ]
        if question_id in appraisal_failures:
            observation["failure_code"] = appraisal_failures[question_id]
        appraisals.append(observation)

    primary_branch_id = collapse["primary_branch_id"]
    supporting_branch_ids = set(collapse["supporting_branch_ids"])
    suppressed_branch_ids = set(collapse["suppressed_branch_ids"])
    branches: list[dict[str, Any]] = []
    branch_index = 0
    execution_groups = (
        ("preliminary", preliminary_branches, preliminary_execution),
        ("final", final_branches, final_execution),
    )
    for phase, definitions, execution in execution_groups:
        if execution is None:
            continue
        for definition in definitions:
            branch_index += 1
            bid = execution.results.get(definition.branch_id)
            if isinstance(bid, Mapping):
                status = "completed"
            elif definition.branch_id in execution.failed_branch_ids:
                status = "failed"
            else:
                status = "not_reported"
            if definition.branch_id == primary_branch_id:
                selection = "primary"
            elif definition.branch_id in supporting_branch_ids:
                selection = "supporting"
            elif definition.branch_id in suppressed_branch_ids:
                selection = "suppressed"
            else:
                selection = "unselected"
            observation = {
                "phase": phase,
                "branch_index": branch_index,
                "goal_kind": definition.goal_kind,
                "status": status,
                "selection": selection,
            }
            if isinstance(bid, Mapping):
                for field_name in (
                    "intention",
                    "desired_outcome",
                    "concrete_detail",
                    "reason",
                    "private_monologue",
                    "confidence",
                ):
                    observation[field_name] = bid[field_name]
                observation["expected_consequences"] = bid[
                    "expected_consequences"
                ]
            else:
                failure = execution.failure_records.get(definition.branch_id)
                if failure is not None:
                    observation["failure_code"] = failure.error_code
            branches.append(observation)

    execution_observation = {
        "selected_question_count": diagnostics["selected_question_count"],
        "dispatched_question_count": diagnostics["dispatched_question_count"],
        "selected_branch_count": diagnostics["selected_branch_count"],
        "dispatched_branch_count": diagnostics["dispatched_branch_count"],
        "completed_branch_count": diagnostics["completed_branch_count"],
        "failed_branch_count": diagnostics["failed_branch_count"],
        "maximum_concurrency": max(
            preliminary_execution.maximum_concurrency,
            final_execution.maximum_concurrency if final_execution else 0,
        ),
        "overlap_ms": diagnostics["overlap_ms"],
        "dependency_wait_ms": diagnostics["dependency_wait_ms"],
        "total_ms": diagnostics["total_ms"],
    }
    index_by_branch_id = {
        definition.branch_id: index
        for index, definition in enumerate(
            [*preliminary_branches, *final_branches],
            start=1,
        )
    }
    primary_index = index_by_branch_id.get(primary_branch_id)
    supporting_indices = [
        index_by_branch_id[branch_id]
        for branch_id in collapse["supporting_branch_ids"]
        if branch_id in index_by_branch_id
    ]
    suppressed_indices = [
        index_by_branch_id[branch_id]
        for branch_id in collapse["suppressed_branch_ids"]
        if branch_id in index_by_branch_id
    ]
    collapse_selection_reason = selected_bid_reason
    if (
        relational_willingness is not None
        and relational_willingness["applicability"] == "relationship_sensitive"
    ):
        collapse_selection_reason = AUTHORITATIVE_RELATIONAL_COLLAPSE_REASON
    return_value: CognitionObservabilityV2 = {
        "execution": execution_observation,
        "appraisals": appraisals,
        "branches": branches,
        "collapse": {
            "primary_branch_index": primary_index,
            "supporting_branch_indices": supporting_indices,
            "suppressed_branch_indices": suppressed_indices,
            "selection_reason": collapse_selection_reason,
        },
    }
    if relational_willingness is not None:
        return_value["relational_willingness"] = dict(
            relational_willingness
        )
    return return_value

def _semantic_relief_transitions(
    prior_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    results: Sequence[SemanticAppraisalResultV2],
    evidence: Sequence[Mapping[str, Any]],
    handle_to_ref: Mapping[str, Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Project accepted threat-pressure reductions into relief causes."""

    evidence_by_handle = {
        row["evidence_handle"]: row["evidence_ref"]
        for row in evidence
    }
    transition_evidence: dict[str, Mapping[str, Any]] = {}
    for result in results:
        for delta in result["deltas"]:
            path = delta["target_path"].split(".")
            if len(path) != 3 or path[0] != "threats":
                continue
            if path[2] != "residual_pressure":
                continue
            target_ref = handle_to_ref.get(path[1])
            if target_ref is None or target_ref["kind"] != "threat":
                continue
            evidence_ref = evidence_by_handle.get(delta["evidence_handles"][0])
            if evidence_ref is not None:
                transition_evidence[target_ref["entity_id"]] = evidence_ref

    current_threats = {
        threat["entity_id"]: threat
        for threat in current_state["threats"]
    }
    transitions: list[dict[str, Any]] = []
    for prior_threat in prior_state["threats"]:
        entity_id = prior_threat["entity_id"]
        current_threat = current_threats.get(entity_id)
        evidence_ref = transition_evidence.get(entity_id)
        if current_threat is None or evidence_ref is None:
            continue
        prior_pressure = prior_threat["residual_pressure"]
        current_pressure = current_threat["residual_pressure"]
        if prior_pressure < 40 or prior_pressure - current_pressure < 20:
            continue
        transitions.append({
            "root_ref": {
                "scope": prior_state["state_scope"],
                "kind": "threat",
                "entity_id": entity_id,
            },
            "prior": {
                "status": prior_threat["status"],
                "residual_pressure": prior_pressure,
            },
            "current": {
                "status": current_threat["status"],
                "residual_pressure": current_pressure,
            },
            "evidence_ref": dict(evidence_ref),
            "salience": prior_threat["salience"],
        })
    return transitions

def _fact_without_producer(fact: Mapping[str, Any]) -> dict[str, Any]:
    """Strip the routing producer before passing a fact to the reducer."""

    result = dict(fact)
    result.pop("producer", None)
    return result

def _native_relationship_context(
    relationship_context: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Keep prompt-only relationship projections out of native reducers."""

    if (
        isinstance(relationship_context, Mapping)
        and relationship_context.get("schema_version")
        == "relationship_operational_context.v1"
    ):
        return None
    return relationship_context

def _episode_updated_at(episode: Mapping[str, Any]) -> str:
    """Project the canonical episode storage timestamp into native UTC-Z."""

    value = episode["created_at"]
    try:
        parsed = parse_storage_utc_datetime(value)
    except (TypeError, ValueError) as exc:
        raise CognitionContractError(
            "episode created_at is invalid"
        ) from exc
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _episode_interaction_date_utc(episode: Mapping[str, Any]) -> str:
    """Derive the canonical UTC interaction date from the episode carrier."""

    return _episode_updated_at(episode)[:10]

def _trusted_relationship_facts(
    direct_facts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pass guarded producer facts to relationship maintenance."""

    trusted: list[dict[str, Any]] = []
    for fact in direct_facts:
        row = dict(_fact_without_producer(fact))
        row["producer"] = fact["producer"]
        trusted.append(row)
    return trusted

def _apply_final_relationship_maintenance(
    state: Mapping[str, Any],
    *,
    episode: Mapping[str, Any],
    elapsed_seconds: int,
    accepted_relationship_deltas: Sequence[Mapping[str, Any]],
    direct_facts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the one final relationship-maintenance transaction."""

    if state["state_scope"] != "user":
        return dict(state)
    return apply_relationship_maintenance(
        state,
        source_episode_id=episode["episode_id"],
        interaction_date_utc=_episode_interaction_date_utc(episode),
        elapsed_seconds=elapsed_seconds,
        accepted_relationship_deltas=accepted_relationship_deltas,
        trusted_facts=_trusted_relationship_facts(direct_facts),
    )

def _elapsed_seconds(previous: str, current: str) -> int:
    """Return non-negative elapsed seconds between two UTC values."""

    try:
        previous_dt = datetime.fromisoformat(previous.replace("Z", "+00:00"))
        current_dt = datetime.fromisoformat(current.replace("Z", "+00:00"))
    except ValueError:
        return 0
    if previous_dt.tzinfo is None or current_dt.tzinfo is None:
        return 0
    return max(0, int((current_dt - previous_dt).total_seconds()))

def _cognition_elapsed_seconds(
    state: Mapping[str, Any],
    current: str,
) -> int:
    """Return elapsed evolution allowed for the cognition state scope."""

    if state["state_scope"] == "character":
        return 0
    return _elapsed_seconds(state["updated_at"], current)

def _reduce_appraisals_with_isolation(
    state: Mapping[str, Any],
    results: Sequence[SemanticAppraisalResultV2],
    evidence: Sequence[Mapping[str, Any]],
    handle_to_ref: Mapping[str, Mapping[str, str]],
    *,
    updated_at: str,
    character_constraints: Mapping[str, Any] | None,
    relationship_context: Mapping[str, Any] | None,
) -> tuple[
    dict[str, Any],
    list[SemanticAppraisalResultV2],
    dict[str, str],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Validate each appraisal through a cumulative accepted-prefix reduction.

    Args:
        state: Validated preliminary cognition state.
        results: Producer-validated semantic appraisal results.
        evidence: Typed evidence available to the appraisal batch.
        handle_to_ref: Private prompt-handle bindings for native reduction.
        updated_at: Canonical timestamp for the final state transaction.
        character_constraints: Read-only character constraints for affect and
            deterministic goal derivation.
        relationship_context: Native relationship context for affect and goal
            derivation.

    Returns:
        The last finalized state, accepted results, rejected question
        diagnostics, and comparison rows produced only by accepted reductions.
    """

    updated_state = dict(state)
    accepted_results: list[SemanticAppraisalResultV2] = []
    failures: dict[str, str] = {}
    comparison_results: list[dict[str, Any]] = []
    accepted_relationship_deltas: list[dict[str, Any]] = []
    for result in [None, *results]:
        candidate_results = list(accepted_results)
        if result is not None:
            candidate_results.append(result)
        candidate_comparisons: list[dict[str, Any]] = []
        finalization_step = "apply_semantic_appraisals"
        try:
            appraisal_application = apply_semantic_appraisals(
                state,
                candidate_results,
                evidence,
                handle_to_ref,
                candidate_comparisons,
            )
            candidate_state = appraisal_application["updated_state"]
            finalization_step = "_semantic_relief_transitions"
            relief_transitions = _semantic_relief_transitions(
                state,
                candidate_state,
                candidate_results,
                evidence,
                handle_to_ref,
            )
            finalization_step = "apply_state_update"
            candidate_state = apply_state_update(
                candidate_state,
                updated_at=updated_at,
                character_constraints=character_constraints,
                relationship_context=relationship_context,
                transition_contexts=relief_transitions,
            )
            finalization_step = "create_deterministic_goals"
            candidate_state = create_deterministic_goals(
                candidate_state,
                character_constraints=character_constraints,
                relationship_context=relationship_context,
                evidence=evidence,
                updated_at=updated_at,
            )
            finalization_step = "validate_cognition_state"
            candidate_state = validate_cognition_state(candidate_state)
        except CognitionStateError as exc:
            if result is None:
                raise
            error_code = "semantic_appraisal_reduction_rejected"
            exception_text = str(exc)[:MAX_APPRAISAL_REJECTION_ERROR_CHARS]
            failures[result["question_id"]] = error_code
            failure_capsule.mark_current_failure(
                failure_kind="semantic_appraisal_reduction_failure",
                stage_name="semantic_appraisal_reduction",
                details={
                    "question_id": result["question_id"],
                    "failure_code": error_code,
                    "finalization_step": finalization_step,
                    "exception_text": exception_text,
                },
                exception=exc,
            )
            capture_validation_event(
                "semantic_appraisal_reduction",
                {
                    "question_id": result["question_id"],
                    "status": "rejected",
                    "error_code": error_code,
                    "finalization_step": finalization_step,
                    "error": exception_text,
                },
            )
            continue
        updated_state = candidate_state
        comparison_results = candidate_comparisons
        accepted_relationship_deltas = appraisal_application[
            "accepted_delta_receipts"
        ]
        if result is not None:
            accepted_results.append(result)
    return (
        updated_state,
        accepted_results,
        failures,
        comparison_results,
        accepted_relationship_deltas,
    )

def _elapsed_ms(started_at: float) -> int:
    """Return a bounded integer duration for protected diagnostics."""

    return max(0, int((time.perf_counter() - started_at) * 1000))

__all__ = [
    "BranchFailure",
    "ParallelExecutionResult",
    "_apply_final_relationship_maintenance",
    "_bids_with_live_goals",
    "_bind_pending_resolution",
    "_branch_context",
    "_build_cognition_observability",
    "_cognition_elapsed_seconds",
    "_deduplicate_diagnostics_warnings",
    "_elapsed_ms",
    "_episode_updated_at",
    "_fact_without_producer",
    "_goal_for_branch",
    "_goal_projection",
    "_mark_cognition_partial_failures",
    "_native_relationship_context",
    "_ordinary_relational_decision",
    "_raise_for_unrecoverable_required_branch_failures",
    "_reduce_appraisals_with_isolation",
    "_resolver_progress",
    "_selected_bid",
    "_workspace_current_event",
    "_workspace_goal_contexts",
    "capture_validation_event",
]
