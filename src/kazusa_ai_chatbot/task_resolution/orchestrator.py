"""Bounded semantic specialist selection for task-resolution sessions."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from importlib import import_module
from time import monotonic
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    BACKGROUND_WORK_LLM_API_KEY,
    BACKGROUND_WORK_LLM_BASE_URL,
    BACKGROUND_WORK_LLM_MAX_COMPLETION_TOKENS,
    BACKGROUND_WORK_LLM_MODEL,
    BACKGROUND_WORK_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_tracing.failure_capsule import mark_current_failure
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    MAX_TASK_RESOLUTION_DISPATCHES,
    MAX_TASK_RESOLUTION_ORCHESTRATOR_CALLS,
    MAX_TASK_RESOLUTION_SPECIALIST_INVOCATIONS,
    TASK_SPECIALISTS,
    TaskResolutionCheckpointV1,
    TaskResolutionContractError,
    TaskResolutionExecutionContextV1,
    TaskResolutionResultV1,
    TaskSpecialistResultV1,
    validate_task_resolution_checkpoint,
    validate_task_resolution_execution_context,
    validate_task_resolution_result,
    validate_task_specialist_result,
)
from kazusa_ai_chatbot.task_resolution.state import (
    build_specialist_request,
    consume_started_dispatch_as_unavailable,
    has_attempted_specialist,
    mark_pending_dispatch_started,
    normalize_started_dispatch_ledger,
    record_orchestrator_call,
    record_specialist_result,
    remaining_dispatch_budget,
    result_from_checkpoint,
    select_pending_dispatch,
    specialist_invocation_count,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


ORCHESTRATOR_PAYLOAD_CHAR_CAP = 24000
MINIMUM_INLINE_DISPATCH_SECONDS = 1.0

TASK_ORCHESTRATOR_PROMPT = '''\
Choose one compatible specialist for one bounded semantic task node.

The task session needs evidence or a bounded specialist result. Select only one
specialist and one semantic subgoal. Do not choose timing, persistence, queue
work, delivery, approval, tool parameters, repository paths, database work, or
final wording. Specialists own their own low-level arguments and refusals.

# Specialist ownership
- local_context: local/private conversation, memory, relationship, profile, and session context evidence.
- public_research: public, current, external, or source-bound research evidence.
- coding: repository analysis or the existing coding-run lifecycle through its public API.
- text_computation: supplied-text rewrite/summary, bounded code snippets, or deterministic caller-supplied numeric expressions.

# Output Format
Return exactly this JSON object:
{
  "specialist": "one listed candidate",
  "subgoal": "short semantic subgoal",
  "coding_objective_mode": "none | read_only | propose_patch"
}

Use coding_objective_mode="none" for local_context, public_research, and
text_computation. Use "read_only" or "propose_patch" only for coding.
'''

_task_orchestrator_llm = LLInterface()
_task_orchestrator_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="BACKGROUND_WORK_LLM",
    base_url=BACKGROUND_WORK_LLM_BASE_URL,
    api_key=BACKGROUND_WORK_LLM_API_KEY,
    model=BACKGROUND_WORK_LLM_MODEL,
    temperature=0.1,
    top_p=0.7,
    top_k=None,
    max_completion_tokens=BACKGROUND_WORK_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=BACKGROUND_WORK_LLM_THINKING_ENABLED),
)

logger = logging.getLogger(__name__)

TaskSpecialistHandler = Callable[
    [dict[str, object], TaskResolutionExecutionContextV1],
    Awaitable[TaskSpecialistResultV1],
]
TaskResolutionCheckpointPersistFunc = Callable[
    [TaskResolutionCheckpointV1, TaskResolutionResultV1 | None],
    Awaitable[None],
]


async def run_task_orchestrator(
    checkpoint: TaskResolutionCheckpointV1,
    execution_context: TaskResolutionExecutionContextV1,
    *,
    inline_deadline: float | None,
    checkpoint_persist_func: TaskResolutionCheckpointPersistFunc | None = None,
    prior_result: TaskResolutionResultV1 | None = None,
) -> TaskResolutionResultV1:
    """Run a bounded task session until terminal result or inline defer.

    Args:
        checkpoint: Validated persistent session state to continue.
        execution_context: Trusted prompt-safe context for specialist adapters.
        inline_deadline: Monotonic deadline for foreground execution, or
            ``None`` for background continuation.
        checkpoint_persist_func: Background-only callback that durably records
            call-count, selected, started, and completed checkpoint snapshots
            while its worker lease remains active. Foreground callers leave it
            unset.
        prior_result: Previous durable deferred result from the same session.
            It preserves completed subgoals and prompt-safe coding context
            across process and lease recovery.

    Returns:
        One terminal or deferred prompt-safe task-resolution result.
    """

    current = normalize_started_dispatch_ledger(checkpoint)
    context = validate_task_resolution_execution_context(execution_context)
    completed_subgoals, coding_run_context = _prior_result_state(
        current,
        prior_result,
    )
    while not current["terminal_status"]:
        pending_dispatch = current["pending_dispatch"]
        if pending_dispatch is not None:
            if pending_dispatch["phase"] == "started":
                current = consume_started_dispatch_as_unavailable(current)
                await _persist_checkpoint_progress(
                    checkpoint_persist_func,
                    current,
                    completed_subgoals=completed_subgoals,
                    coding_run_context=coding_run_context,
                )
                break

            if (
                pending_dispatch["specialist"] == "coding"
                and inline_deadline is not None
            ):
                return _deferred_result(
                    current,
                    completed_subgoals=completed_subgoals,
                    coding_run_context=coding_run_context,
                )
            remaining_seconds = _remaining_seconds(inline_deadline)
            if remaining_seconds is not None and (
                remaining_seconds < MINIMUM_INLINE_DISPATCH_SECONDS
            ):
                return _deferred_result(
                    current,
                    completed_subgoals=completed_subgoals,
                    coding_run_context=coding_run_context,
                )
            try:
                current = mark_pending_dispatch_started(current)
            except TaskResolutionContractError:
                current = _terminalize_without_continuation(
                    current,
                    status="failed",
                )
                await _persist_checkpoint_progress(
                    checkpoint_persist_func,
                    current,
                    completed_subgoals=completed_subgoals,
                    coding_run_context=coding_run_context,
                )
                break
            await _persist_checkpoint_progress(
                checkpoint_persist_func,
                current,
                completed_subgoals=completed_subgoals,
                coding_run_context=coding_run_context,
            )
            request = build_specialist_request(current)
            specialist = pending_dispatch["specialist"]
            handler = specialist_handler(specialist)
            try:
                specialist_result = await _run_specialist_with_deadline(
                    handler,
                    request,
                    context,
                    remaining_seconds=remaining_seconds,
                )
            except TimeoutError:
                specialist_result = {
                    "schema_version": "task_specialist_result.v1",
                    "specialist": specialist,
                    "status": "temporarily_unavailable",
                    "evidence": [],
                    "completed_subgoals": [],
                    "remaining_needs": list(current["remaining_needs"]),
                    "reason": "The inline task deadline elapsed before completion.",
                    "retryable": True,
                }
            try:
                validated_result = validate_task_specialist_result(
                    specialist_result,
                )
                current = record_specialist_result(current, validated_result)
            except TaskResolutionContractError as exc:
                mark_current_failure(
                    failure_kind="task_specialist_contract_error",
                    stage_name="task_resolution_orchestrator",
                    details={
                        "specialist": specialist,
                        "task_node_id": pending_dispatch["task_node_id"],
                        "error": str(exc),
                    },
                    exception=exc,
                )
                logger.error(
                    f"Task specialist candidate failed contract validation: {exc}"
                )
                current = _terminalize_without_continuation(
                    current,
                    status="failed",
                )
                await _persist_checkpoint_progress(
                    checkpoint_persist_func,
                    current,
                    completed_subgoals=completed_subgoals,
                    coding_run_context=coding_run_context,
                )
                break
            completed_subgoals.extend(validated_result["completed_subgoals"])
            result_coding_context = validated_result.get("coding_run_context")
            if isinstance(result_coding_context, dict):
                coding_run_context = dict(result_coding_context)
            await _persist_checkpoint_progress(
                checkpoint_persist_func,
                current,
                completed_subgoals=completed_subgoals,
                coding_run_context=coding_run_context,
            )
            if current["terminal_status"]:
                break
            if (
                validated_result["status"] == "temporarily_unavailable"
                and inline_deadline is not None
            ):
                return _deferred_result(
                    current,
                    completed_subgoals=completed_subgoals,
                    coding_run_context=coding_run_context,
                )
            continue

        if remaining_dispatch_budget(current) == 0:
            current = _terminalize_without_continuation(current)
            break
        if (
            current["orchestrator_call_count"]
            >= MAX_TASK_RESOLUTION_ORCHESTRATOR_CALLS
        ):
            current = _terminalize_without_continuation(current)
            break
        remaining_seconds = _remaining_seconds(inline_deadline)
        if remaining_seconds is not None and (
            remaining_seconds < MINIMUM_INLINE_DISPATCH_SECONDS
        ):
            return _deferred_result(
                current,
                completed_subgoals=completed_subgoals,
                coding_run_context=coding_run_context,
            )

        candidate_specialists = _eligible_specialists(current)
        if not candidate_specialists:
            current = _terminalize_without_continuation(current)
            break
        current = record_orchestrator_call(current)
        await _persist_checkpoint_progress(
            checkpoint_persist_func,
            current,
            completed_subgoals=completed_subgoals,
            coding_run_context=coding_run_context,
        )
        try:
            selection = await select_next_specialist(
                current,
                context,
                candidate_specialists=candidate_specialists,
            )
            selection = _validate_specialist_selection(
                selection,
                candidate_specialists,
            )
        except TaskResolutionContractError:
            if (
                current["orchestrator_call_count"]
                >= MAX_TASK_RESOLUTION_ORCHESTRATOR_CALLS
            ):
                current = _terminalize_without_continuation(current)
                await _persist_checkpoint_progress(
                    checkpoint_persist_func,
                    current,
                    completed_subgoals=completed_subgoals,
                    coding_run_context=coding_run_context,
                )
                break
            continue
        current = select_pending_dispatch(
            current,
            specialist=selection["specialist"],
            subgoal=selection["subgoal"],
            coding_objective_mode=selection["coding_objective_mode"],
        )
        await _persist_checkpoint_progress(
            checkpoint_persist_func,
            current,
            completed_subgoals=completed_subgoals,
            coding_run_context=coding_run_context,
        )

    status = current["terminal_status"] or "unavailable"
    return _terminal_result(
        current,
        status=status,
        summary=_terminal_summary(status),
        completed_subgoals=completed_subgoals,
        coding_run_context=coding_run_context,
    )


async def select_next_specialist(
    checkpoint: TaskResolutionCheckpointV1,
    execution_context: TaskResolutionExecutionContextV1,
    *,
    candidate_specialists: list[str],
) -> dict[str, str]:
    """Ask the bounded orchestrator stage for one next semantic specialist.

    Tests may replace this module-level function to exercise deterministic
    state transitions.  Production callers use the static prompt and bounded
    human payload declared in this module.
    """

    validated_checkpoint = normalize_started_dispatch_ledger(checkpoint)
    validate_task_resolution_execution_context(execution_context)
    active_node = _active_node(validated_checkpoint)
    payload = {
        "semantic_objective": validated_checkpoint["semantic_objective"],
        "active_node": {
            "node_id": active_node["node_id"],
            "objective": active_node["objective"],
        },
        "evidence": [
            {
                "specialist": evidence["specialist"],
                "summary": evidence["summary"],
                "provenance_refs": list(evidence["provenance_refs"]),
                "limitations": list(evidence["limitations"]),
            }
            for evidence in validated_checkpoint["evidence"]
        ],
        "remaining_needs": list(validated_checkpoint["remaining_needs"]),
        "attempted_specialists": list(
            validated_checkpoint["attempted_specialists"]
        ),
        "candidate_specialists": candidate_specialists,
    }
    prompt_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(prompt_text) > ORCHESTRATOR_PAYLOAD_CHAR_CAP:
        raise TaskResolutionContractError(
            "orchestrator payload exceeds prompt-safe character cap"
        )
    response = await _task_orchestrator_llm.ainvoke(
        [
            SystemMessage(content=TASK_ORCHESTRATOR_PROMPT),
            HumanMessage(content=prompt_text),
        ],
        config=_task_orchestrator_llm_config,
    )
    parsed = parse_llm_json_output(
        response.content,
        expected_output_format=(
            '{"specialist": "...", "subgoal": "...", '
            '"coding_objective_mode": "none | read_only | propose_patch"}'
        ),
    )
    selection = _validate_specialist_selection(parsed, candidate_specialists)
    return selection


def specialist_handler(specialist: str) -> TaskSpecialistHandler:
    """Resolve one registered specialist handler through its public module."""

    module_names = {
        "local_context": "kazusa_ai_chatbot.task_resolution.specialists.local_context",
        "public_research": "kazusa_ai_chatbot.task_resolution.specialists.public_research",
        "coding": "kazusa_ai_chatbot.task_resolution.specialists.coding",
        "text_computation": "kazusa_ai_chatbot.task_resolution.specialists.text_computation",
    }
    function_names = {
        "local_context": "resolve_with_local_context",
        "public_research": "resolve_with_public_research",
        "coding": "resolve_with_coding",
        "text_computation": "resolve_with_text_computation",
    }
    module_name = module_names.get(specialist)
    function_name = function_names.get(specialist)
    if module_name is None or function_name is None:
        raise TaskResolutionContractError("specialist: unsupported value")
    module = import_module(module_name)
    handler = getattr(module, function_name)
    return handler


async def _run_specialist_with_deadline(
    handler: TaskSpecialistHandler,
    request: dict[str, object],
    execution_context: TaskResolutionExecutionContextV1,
    *,
    remaining_seconds: float | None,
) -> TaskSpecialistResultV1:
    """Run one public specialist operation under the remaining inline budget."""

    if remaining_seconds is None:
        result = await handler(request, execution_context)
        return result
    try:
        result = await asyncio.wait_for(
            handler(request, execution_context),
            remaining_seconds,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError("task specialist exceeded inline deadline") from exc
    return result


def _eligible_specialists(checkpoint: TaskResolutionCheckpointV1) -> list[str]:
    """Return specialists that still satisfy deterministic session limits."""

    active_node = _active_node(checkpoint)
    candidates: list[str] = []
    for specialist in sorted(TASK_SPECIALISTS):
        if has_attempted_specialist(
            checkpoint,
            task_node_id=active_node["node_id"],
            specialist=specialist,
        ):
            continue
        invocation_count = specialist_invocation_count(
            checkpoint,
            task_node_id=active_node["node_id"],
            specialist=specialist,
        )
        if invocation_count >= MAX_TASK_RESOLUTION_SPECIALIST_INVOCATIONS:
            continue
        if invocation_count == 1 and not _second_invocation_is_eligible(
            checkpoint,
            task_node_id=active_node["node_id"],
            specialist=specialist,
        ):
            continue
        candidates.append(specialist)
    return candidates


def _second_invocation_is_eligible(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    task_node_id: str,
    specialist: str,
) -> bool:
    """Allow one revisit only after persisted retryable or new evidence state.

    ``temporarily_unavailable`` is validated as a retryable operational result
    before it reaches the trace ledger.  A subsequent specialist dispatch that
    adds evidence for this node also changes the canonical handler input and
    therefore permits one bounded revisit.  The calculation relies only on
    validated checkpoint state, so a queue or process retry cannot create a
    blind semantic retry.
    """

    previous_dispatch = _latest_specialist_dispatch(
        checkpoint,
        task_node_id=task_node_id,
        specialist=specialist,
    )
    if previous_dispatch is None:
        return False
    if previous_dispatch["result_status"] == "temporarily_unavailable":
        return True
    return _has_material_new_evidence(
        checkpoint,
        task_node_id=task_node_id,
        specialist=specialist,
        previous_dispatch_index=previous_dispatch["dispatch_index"],
    )


def _latest_specialist_dispatch(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    task_node_id: str,
    specialist: str,
) -> Mapping[str, object] | None:
    """Return the most recent validated trace row for one specialist pair."""

    matching_rows = [
        row
        for row in checkpoint["trace_summary"]
        if row["task_node_id"] == task_node_id
        and row["specialist"] == specialist
    ]
    if not matching_rows:
        return None
    return matching_rows[-1]


def _has_material_new_evidence(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    task_node_id: str,
    specialist: str,
    previous_dispatch_index: object,
) -> bool:
    """Detect evidence added by a later different-specialist dispatch.

    Evidence is included in every subsequent specialist request.  A later
    trace from another specialist plus its node evidence therefore represents a
    material input change without storing unbounded raw payloads or adding a
    compatibility field to the exact checkpoint schema.
    """

    if not isinstance(previous_dispatch_index, int):
        raise TaskResolutionContractError(
            "trace_summary.dispatch_index: expected integer"
        )
    later_specialists = {
        row["specialist"]
        for row in checkpoint["trace_summary"]
        if row["dispatch_index"] > previous_dispatch_index
        and row["task_node_id"] == task_node_id
        and row["specialist"] != specialist
    }
    if not later_specialists:
        return False
    return any(
        evidence["task_node_id"] == task_node_id
        and evidence["specialist"] in later_specialists
        for evidence in checkpoint["evidence"]
    )


def _validate_specialist_selection(
    value: object,
    candidate_specialists: list[str],
) -> dict[str, str]:
    """Validate one stage-owned specialist and semantic subgoal choice."""

    if not isinstance(value, Mapping):
        raise TaskResolutionContractError("orchestrator selection: expected object")
    if set(value) != {
        "specialist",
        "subgoal",
        "coding_objective_mode",
    }:
        raise TaskResolutionContractError("orchestrator selection: invalid fields")
    specialist = value["specialist"]
    subgoal = value["subgoal"]
    coding_objective_mode = value["coding_objective_mode"]
    if not isinstance(specialist, str) or specialist not in candidate_specialists:
        raise TaskResolutionContractError(
            "orchestrator selection: specialist is unavailable"
        )
    if not isinstance(subgoal, str) or not subgoal.strip():
        raise TaskResolutionContractError("orchestrator selection: invalid subgoal")
    if not isinstance(coding_objective_mode, str):
        raise TaskResolutionContractError(
            "orchestrator selection: invalid coding_objective_mode"
        )
    if specialist == "coding":
        if coding_objective_mode not in {"read_only", "propose_patch"}:
            raise TaskResolutionContractError(
                "orchestrator selection: coding requires an objective mode"
            )
    elif coding_objective_mode != "none":
        raise TaskResolutionContractError(
            "orchestrator selection: non-coding requires none mode"
        )
    normalized = {
        "specialist": specialist,
        "subgoal": subgoal.strip()[:1200],
        "coding_objective_mode": coding_objective_mode,
    }
    return normalized


def _active_node(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the active node projection for prompt-safe dispatch selection."""

    active_node_id = checkpoint["active_node_id"]
    nodes = checkpoint["nodes"]
    for node in nodes:
        if node["node_id"] == active_node_id:
            return node
    raise TaskResolutionContractError("active_node_id: missing from checkpoint")


def _remaining_seconds(inline_deadline: float | None) -> float | None:
    """Return the remaining foreground deadline without changing session state."""

    if inline_deadline is None:
        return None
    remaining = inline_deadline - monotonic()
    return remaining


def _prior_result_state(
    checkpoint: TaskResolutionCheckpointV1,
    prior_result: TaskResolutionResultV1 | None,
) -> tuple[list[str], dict[str, object]]:
    """Restore durable result projections for a resumed nonterminal session."""

    if prior_result is None:
        return [], {}
    validated_result = validate_task_resolution_result(prior_result)
    if validated_result["status"] != "deferred":
        raise TaskResolutionContractError(
            "prior_result: expected deferred task-resolution result"
        )
    persisted_checkpoint = validate_task_resolution_checkpoint(
        validated_result["checkpoint"],
    )
    if persisted_checkpoint["session_id"] != checkpoint["session_id"]:
        raise TaskResolutionContractError(
            "prior_result: checkpoint session does not match resume checkpoint"
        )
    completed_subgoals = list(validated_result["completed_subgoals"])
    coding_run_context = dict(validated_result["coding_run_context"])
    return completed_subgoals, coding_run_context


async def _persist_checkpoint_progress(
    checkpoint_persist_func: TaskResolutionCheckpointPersistFunc | None,
    checkpoint: TaskResolutionCheckpointV1,
    *,
    completed_subgoals: list[str],
    coding_run_context: dict[str, object],
) -> None:
    """Persist a lease-owned checkpoint before or after a bounded operation."""

    if checkpoint_persist_func is None:
        return
    snapshot = _checkpoint_snapshot_or_none(
        checkpoint,
        completed_subgoals=completed_subgoals,
        coding_run_context=coding_run_context,
    )
    await checkpoint_persist_func(checkpoint, snapshot)


def _checkpoint_snapshot_or_none(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    completed_subgoals: list[str],
    coding_run_context: dict[str, object],
) -> TaskResolutionResultV1 | None:
    """Build a result snapshot only when the checkpoint can be deferred."""

    if checkpoint["terminal_status"]:
        return _checkpoint_result(
            checkpoint,
            completed_subgoals=completed_subgoals,
            coding_run_context=coding_run_context,
        )
    if checkpoint["dispatch_count"] >= MAX_TASK_RESOLUTION_DISPATCHES:
        return None
    if (
        checkpoint["orchestrator_call_count"]
        >= MAX_TASK_RESOLUTION_ORCHESTRATOR_CALLS
        and checkpoint["pending_dispatch"] is None
    ):
        return None
    return _checkpoint_result(
        checkpoint,
        completed_subgoals=completed_subgoals,
        coding_run_context=coding_run_context,
    )


def _deferred_result(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    completed_subgoals: list[str],
    coding_run_context: dict[str, object],
) -> TaskResolutionResultV1:
    """Return the validated durable handover for a nonterminal checkpoint."""

    return result_from_checkpoint(
        checkpoint,
        status="deferred",
        prompt_safe_summary="The task needs durable continuation.",
        completed_subgoals=completed_subgoals,
        coding_run_context=coding_run_context,
    )


def _terminalize_without_continuation(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    status: str | None = None,
) -> TaskResolutionCheckpointV1:
    """Close a session when no further specialist selection is allowed."""

    current = normalize_started_dispatch_ledger(checkpoint)
    pending_dispatch = current["pending_dispatch"]
    if pending_dispatch is not None and pending_dispatch["phase"] == "started":
        current = consume_started_dispatch_as_unavailable(current)
        if status is None:
            return current
    updated = deepcopy(current)
    if updated["pending_dispatch"] is not None:
        active_node = _active_node(updated)
        active_node["status"] = "blocked"
        updated["pending_dispatch"] = None
    if status is None:
        status = "partial" if updated["evidence"] else "unavailable"
    updated["terminal_status"] = status
    return validate_task_resolution_checkpoint(updated)


def _checkpoint_result(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    completed_subgoals: list[str],
    coding_run_context: dict[str, object],
) -> TaskResolutionResultV1:
    """Build the durable result snapshot paired with one checkpoint update."""

    terminal_status = checkpoint["terminal_status"]
    if terminal_status:
        result = _terminal_result(
            checkpoint,
            status=terminal_status,
            summary=_terminal_summary(terminal_status),
            completed_subgoals=completed_subgoals,
            coding_run_context=coding_run_context,
        )
        return result
    result = result_from_checkpoint(
        checkpoint,
        status="deferred",
        prompt_safe_summary="The task needs durable continuation.",
        completed_subgoals=completed_subgoals,
        coding_run_context=coding_run_context,
    )
    return result


def _terminal_result(
    checkpoint: TaskResolutionCheckpointV1,
    *,
    status: str,
    summary: str,
    completed_subgoals: list[str],
    coding_run_context: dict[str, object],
) -> TaskResolutionResultV1:
    """Build one terminal public result from an already-settled checkpoint."""

    result = result_from_checkpoint(
        checkpoint,
        status=status,
        prompt_safe_summary=summary,
        completed_subgoals=completed_subgoals,
        coding_run_context=coding_run_context,
    )
    return result


def _terminal_summary(status: str) -> str:
    """Describe the terminal task state without queue or worker vocabulary."""

    summaries = {
        "resolved": "The task resolved with validated evidence.",
        "partial": "The task produced validated evidence with remaining limitations.",
        "needs_user_input": "The task needs additional user-provided information.",
        "approval_required": "The task requires approval before it can continue.",
        "unavailable": "The task could not obtain a compatible available specialist.",
        "failed": "The task could not complete its bounded resolution path.",
    }
    summary = summaries.get(status)
    if summary is None:
        raise TaskResolutionContractError("status: unsupported terminal value")
    return summary
