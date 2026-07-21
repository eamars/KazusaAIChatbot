"""Public V2 cognition facade with explicit preliminary and final phases."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.action_selection import plan_actions
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    select_final_branches,
    select_preliminary_branches,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionBidV2,
    BranchDefinition,
    CognitionCoreInputV2,
    CognitionCoreOutputV2,
    CognitionCoreServicesV2,
    CognitionContractError,
    CognitionContextLimitError,
    CognitionExecutionError,
    SemanticAppraisalResultV2,
    validate_cognition_core_input,
    validate_cognition_core_output,
)
from kazusa_ai_chatbot.cognition_core_v2.dependency_graph import (
    build_dependency_graph,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    capture_validation_event,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import run_goal_cognition
from kazusa_ai_chatbot.cognition_core_v2.output_projection import (
    build_state_update,
    default_expression_policy,
    project_affect,
    project_relationship,
)
from kazusa_ai_chatbot.cognition_core_v2.parallel_executor import (
    ParallelExecutionResult,
    execute_dependency_graph,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    appraise_semantic_question,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    plan_semantic_questions,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    ROLE_ENTITY_KINDS,
    validate_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_semantic_appraisals,
    apply_state_update,
    create_deterministic_goals,
)
from kazusa_ai_chatbot.cognition_core_v2.workspace import collapse_bids
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_PENDING_RESOLUTION_VERSION,
    ResolverValidationError,
    validate_resolver_pending_resolution,
    validate_resolver_pending_resume,
)
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime


async def run_cognition(
    input_payload: CognitionCoreInputV2,
    services: CognitionCoreServicesV2,
) -> CognitionCoreOutputV2:
    """Run the bounded two-phase cognition pipeline."""

    started_at = time.perf_counter()
    payload = validate_cognition_core_input(input_payload)
    previous_state = validate_cognition_state(payload["mutable_state"])
    updated_at = _episode_updated_at(payload["episode"])
    elapsed_seconds = _cognition_elapsed_seconds(previous_state, updated_at)
    warnings: list[str] = []
    stage_status: dict[str, str] = {
        "input_validation": "completed",
        "deterministic_preliminary": "skipped",
        "semantic_appraisal": "skipped",
        "final_reduction": "skipped",
        "branch_cognition": "skipped",
        "workspace_collapse": "skipped",
        "action_planning": "skipped",
    }

    fact_pairs = [
        (fact["producer"], _fact_without_producer(fact))
        for fact in payload["direct_facts"]
    ]
    preliminary_state = apply_state_update(
        previous_state,
        direct_facts=fact_pairs,
        elapsed_seconds=elapsed_seconds,
        updated_at=updated_at,
        character_constraints=payload["character_constraints"],
        relationship_context=payload.get("relationship_context"),
    )
    preliminary_state = create_deterministic_goals(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        relationship_context=payload.get("relationship_context"),
        evidence=payload["evidence"],
        updated_at=updated_at,
    )
    preliminary_state = validate_cognition_state(preliminary_state)
    projection = project_state_for_prompt(
        preliminary_state,
        character_constraints=payload["character_constraints"],
        relationship_context=payload.get("relationship_context"),
        evidence=payload["evidence"],
    )
    questions = plan_semantic_questions(
        payload["evidence"],
        preliminary_state,
        projection.handle_to_ref,
    )
    stage_status["deterministic_preliminary"] = "completed"

    preliminary_branches = [
        replace(definition, dependencies=(), dependency_options=())
        for definition in select_preliminary_branches(preliminary_state["goals"])
    ]
    preliminary_graph = build_dependency_graph(preliminary_branches)
    branch_context = _branch_context(
        projection,
        preliminary_state,
        payload["evidence"],
        scene_context=payload["scene_context"],
        private_continuity_context=payload["private_continuity_context"],
    )

    appraisal_tasks = [
        asyncio.create_task(
            appraise_semantic_question(
                question,
                payload["evidence"],
                projection,
                services,
            )
        )
        for question in questions
    ]
    if appraisal_tasks:
        stage_status["semantic_appraisal"] = "completed"
    preliminary_branch_task = asyncio.create_task(
        execute_dependency_graph(
            preliminary_graph,
            _branch_handler(
                branch_context,
                preliminary_state,
                payload,
                services,
            ),
        )
    )
    preliminary_execution, appraisal_collection = await asyncio.gather(
        preliminary_branch_task,
        _collect_appraisals(appraisal_tasks, questions),
    )
    appraisal_results, appraisal_warnings = appraisal_collection
    warnings.extend(appraisal_warnings)
    _raise_for_failed_required_branches(
        preliminary_execution,
        preliminary_branches,
    )
    warnings.extend(preliminary_execution.warnings)
    stage_status["branch_cognition"] = "completed"

    comparison_results: list[dict[str, Any]] = []
    final_state = preliminary_state
    if appraisal_results:
        final_state = apply_semantic_appraisals(
            final_state,
            appraisal_results,
            payload["evidence"],
            projection.handle_to_ref,
            comparison_results,
        )
    relief_transitions = _semantic_relief_transitions(
        preliminary_state,
        final_state,
        appraisal_results,
        payload["evidence"],
        projection.handle_to_ref,
    )
    final_state = apply_state_update(
        final_state,
        updated_at=updated_at,
        character_constraints=payload["character_constraints"],
        relationship_context=payload.get("relationship_context"),
        transition_contexts=relief_transitions,
    )
    final_state = create_deterministic_goals(
        final_state,
        character_constraints=payload["character_constraints"],
        relationship_context=payload.get("relationship_context"),
        evidence=payload["evidence"],
        updated_at=updated_at,
    )
    final_state = validate_cognition_state(final_state)
    stage_status["final_reduction"] = "completed"
    final_projection = project_state_for_prompt(
        final_state,
        character_constraints=payload["character_constraints"],
        relationship_context=payload.get("relationship_context"),
        evidence=payload["evidence"],
    )

    successful_questions = {result["question_id"] for result in appraisal_results}
    final_branches = select_final_branches(
        preliminary_branches,
        final_state["goals"],
        successful_questions,
    )
    new_branch_definitions = [
        definition
        for definition in final_branches
        if definition.branch_id not in preliminary_graph.definitions
    ]
    final_execution = await execute_dependency_graph(
        build_dependency_graph(
            new_branch_definitions,
            external_dependencies=successful_questions,
        ),
        _branch_handler(
            _branch_context(
                final_projection,
                final_state,
                payload["evidence"],
                appraisal_results,
                scene_context=payload["scene_context"],
                private_continuity_context=payload[
                    "private_continuity_context"
                ],
            ),
            final_state,
            payload,
            services,
        ),
        completed_external_dependencies=successful_questions,
    ) if new_branch_definitions else None
    if final_execution is not None:
        _raise_for_failed_required_branches(
            final_execution,
            new_branch_definitions,
        )
        warnings.extend(final_execution.warnings)

    bids: list[ActionBidV2] = list(preliminary_execution.results.values())
    if final_execution is not None:
        bids.extend(final_execution.results.values())
    bids = [bid for bid in bids if isinstance(bid, Mapping)]
    generated_bids = list(bids)
    try:
        collapse = await collapse_bids(bids, services) if bids else _empty_collapse()
    except Exception as exc:
        raise CognitionExecutionError(f"workspace collapse failed: {exc}") from exc
    primary_bid = collapse.get("primary_bid")
    supporting_bids = collapse.get("supporting_bids", [])
    try:
        action_plan = await plan_actions(
            primary_bid=primary_bid,
            supporting_bids=supporting_bids,
            episode=payload["episode"],
            evidence=payload["evidence"],
            available_actions=payload["available_actions"],
            available_resolvers=payload["available_resolver_capabilities"],
            resolver_context=payload["resolver_context"],
            services=services,
            current_goal_progress=payload.get("resolver_goal_progress"),
        )
    except CognitionExecutionError:
        raise
    except Exception as exc:
        raise CognitionExecutionError(f"action planning failed: {exc}") from exc
    intention = action_plan["intention"]
    action_requests = action_plan["action_requests"]
    resolver_requests = action_plan["resolver_requests"]
    pending_resolution = _bind_pending_resolution(
        action_plan["resolver_pending_resolution"],
        payload.get("pending_resolver_resume"),
    )
    stage_status["workspace_collapse"] = "completed"
    stage_status["action_planning"] = "completed"

    admitted_bid = _selected_bid(intention, primary_bid, supporting_bids)
    affect = project_affect(final_state["affect_activations"], final_state)
    relationship = project_relationship(final_state.get("relationship"))
    expression_policy = default_expression_policy(
        intention["route"],
        affect,
        selected_branch_id=intention.get("selected_branch_id"),
        activations=final_state["affect_activations"],
    )
    output: dict[str, Any] = {
        "schema_version": "cognition_core_output.v2",
        "intention": intention,
        "supporting_bids": [
            bid for bid in supporting_bids
            if admitted_bid is None or bid["branch_id"] != admitted_bid["branch_id"]
        ],
        "state_update": build_state_update(
            previous_state,
            final_state,
            comparison_results,
        ),
        "affect_projection": affect,
        "action_requests": action_requests,
        "resolver_requests": resolver_requests,
        "goal_resolution": action_plan["goal_resolution"],
        "resolver_pending_resolution": pending_resolution,
        "resolver_goal_progress": action_plan["resolver_goal_progress"],
        "resolver_progress": _resolver_progress(resolver_requests),
        "selected_bid_reason": (
            admitted_bid["reason"]
            if admitted_bid is not None
            else "没有有依据的目标候选"
        ),
        "private_monologue": (
            admitted_bid["private_monologue"]
            if admitted_bid is not None
            else "当前角色没有有依据的行动理由。"
        ),
        "expression_policy": expression_policy,
        "diagnostics": {
            "run_id": str(payload["episode"].get("episode_id", "episode")),
            "stage_status": stage_status,
            "selected_question_count": len(questions),
            "dispatched_question_count": len(appraisal_tasks),
            "selected_branch_count": (
                len(preliminary_branches) + len(new_branch_definitions)
            ),
            "dispatched_branch_count": (
                len(preliminary_execution.started_at)
                + (len(final_execution.started_at) if final_execution else 0)
            ),
            "completed_branch_count": (
                len(preliminary_execution.results)
                + (len(final_execution.results) if final_execution else 0)
            ),
            "failed_branch_count": (
                len(preliminary_execution.failed_branch_ids)
                + (len(final_execution.failed_branch_ids) if final_execution else 0)
            ),
            "overlap_ms": max(
                preliminary_execution.overlap_ms,
                final_execution.overlap_ms if final_execution else 0,
            ),
            "dependency_wait_ms": max(
                preliminary_execution.dependency_wait_ms,
                final_execution.dependency_wait_ms if final_execution else 0,
            ),
            "total_ms": _elapsed_ms(started_at),
            "warnings": warnings,
        },
    }
    if admitted_bid is not None:
        output["admitted_bid"] = admitted_bid
    if relationship is not None:
        output["relationship_projection"] = relationship
    all_branch_definitions = [
        *preliminary_branches,
        *new_branch_definitions,
    ]
    capture_validation_event(
        "dependency_graph",
        {
            "branch_definitions": [
                {
                    "branch_id": definition.branch_id,
                    "dependencies": list(definition.dependencies),
                    "action_tendencies": list(definition.action_tendencies),
                    "required": definition.required,
                    "goal_kind": definition.goal_kind,
                }
                for definition in all_branch_definitions
            ],
        },
    )
    capture_validation_event(
        "branch_execution",
        {
            "maximum_concurrency": max(
                preliminary_execution.maximum_concurrency,
                final_execution.maximum_concurrency if final_execution else 0,
            ),
            "generated_bids": generated_bids,
            "eligible_bids": bids,
            "failed_branch_ids": sorted({
                *preliminary_execution.failed_branch_ids,
                *(
                    final_execution.failed_branch_ids
                    if final_execution is not None
                    else set()
                ),
            }),
        },
    )
    capture_validation_event(
        "emotion_derivation",
        {
            "state_before_derivation": preliminary_state,
            "state_after_derivation": final_state,
            "affect_projection": affect,
        },
    )
    capture_validation_event(
        "workspace_selection",
        {
            "appraisal_results": appraisal_results,
            "comparison_results": comparison_results,
            "final_intention": intention,
        },
    )
    return validate_cognition_core_output(output)


def _raise_for_failed_required_branches(
    execution: ParallelExecutionResult,
    definitions: Sequence[BranchDefinition],
) -> None:
    """Prevent required branch failures from becoming semantic silence."""

    required_failures = sorted(
        definition.branch_id
        for definition in definitions
        if (
            definition.required
            and definition.branch_id in execution.failed_branch_ids
        )
    )
    if required_failures:
        failed_names = ", ".join(required_failures)
        primary_failure = execution.failure_records.get(required_failures[0])
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


def _branch_handler(
    semantic_context: Mapping[str, Any],
    state: Mapping[str, Any],
    payload: Mapping[str, Any],
    services: CognitionCoreServicesV2,
):
    """Create an isolated branch handler with immutable semantic inputs."""

    async def handle(definition: Any) -> ActionBidV2:
        if definition.branch_id == "ordinary_response" and not payload["evidence"]:
            return None  # type: ignore[return-value]
        goal = _goal_for_branch(state, definition.goal_kind)
        if goal is None:
            goal_ref = {
                "scope": state["state_scope"],
                "kind": "goal",
                "entity_id": f"goal:{definition.goal_kind}:episode",
            }
        else:
            goal_ref = {
                "scope": state["state_scope"],
                "kind": "goal",
                "entity_id": goal["entity_id"],
            }
        context = dict(semantic_context)
        context["goal_projection"] = _goal_projection(goal, definition.goal_kind)
        return await run_goal_cognition(
            definition,
            goal_ref,
            context,
            payload["evidence"],
            services,
        )

    return handle


def _branch_context(
    projection: Any,
    state: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    appraisal_results: Sequence[Mapping[str, Any]] = (),
    *,
    scene_context: Mapping[str, Any],
    private_continuity_context: str,
) -> dict[str, Any]:
    """Build semantic branch context and retain handle bindings privately."""

    context = dict(projection.payload)
    role_bindings: dict[str, dict[str, str]] = {}
    role_summaries: dict[str, str] = {}
    for handle, ref in projection.handle_to_ref.items():
        role_summaries[handle] = _role_summary(
            handle,
            ref,
            scene_context=scene_context,
        )
        if ref["kind"] in ROLE_ENTITY_KINDS:
            role_bindings[handle] = {
                "role": _role_label(handle, ref),
                "entity_kind": ref["kind"],
                "entity_id": ref["entity_id"],
            }
        elif handle == "self":
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
    del evidence
    return context


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


def _empty_collapse() -> dict[str, Any]:
    """Return the deterministic no-bid collapse envelope."""

    return {
        "primary_bid": None,
        "supporting_bids": [],
        "competing_bids": [],
    }


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


async def _collect_appraisals(
    tasks: Sequence[asyncio.Task[SemanticAppraisalResultV2]],
    questions: Sequence[Mapping[str, Any]],
) -> tuple[list[SemanticAppraisalResultV2], list[str]]:
    """Collect independent appraisals while isolating failed question slots."""

    if not tasks:
        return [], []
    collected = await asyncio.gather(*tasks, return_exceptions=True)
    results: list[SemanticAppraisalResultV2] = []
    warnings: list[str] = []
    for question, result in zip(questions, collected, strict=True):
        if isinstance(result, Exception):
            if isinstance(result, CognitionContextLimitError):
                raise result
            warnings.append(f"{question['question_id']} appraisal failed: {result}")
        else:
            results.append(result)
    return results, warnings


def _elapsed_ms(started_at: float) -> int:
    """Return a bounded integer duration for protected diagnostics."""

    return max(0, int((time.perf_counter() - started_at) * 1000))
