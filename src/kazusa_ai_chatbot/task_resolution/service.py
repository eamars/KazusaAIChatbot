"""Public inline and resume entrypoints for task-resolution sessions."""

from __future__ import annotations

from time import monotonic

from kazusa_ai_chatbot.accepted_task.lifecycle import (
    create_or_return_active_accepted_task,
    mark_accepted_task_enqueue_failed,
    mark_accepted_task_pending,
)
from kazusa_ai_chatbot.accepted_task.models import AcceptedTaskCreateRequest
from kazusa_ai_chatbot.background_work.jobs import (
    enqueue_background_work_request,
)
from kazusa_ai_chatbot.background_work.models import (
    BACKGROUND_WORK_REQUESTED_DELIVERY,
    TASK_ORCHESTRATOR_WORKER,
    TASK_ORCHESTRATOR_WORKER_PAYLOAD_VERSION,
    BackgroundWorkQueueRequest,
    BackgroundWorkQueueResult,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ResolverCapabilityRequestV2,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    TaskResolutionCheckpointV1,
    TaskResolutionContractError,
    TaskResolutionExecutionContextV1,
    TaskResolutionResultV1,
    validate_task_resolution_checkpoint,
    validate_task_resolution_execution_context,
    validate_task_resolution_result,
)
from kazusa_ai_chatbot.task_resolution.orchestrator import run_task_orchestrator
from kazusa_ai_chatbot.task_resolution.state import (
    create_task_resolution_checkpoint,
    result_from_checkpoint,
)


MINIMUM_INLINE_BUDGET_SECONDS = 1.0
MAXIMUM_INLINE_BUDGET_SECONDS = 120.0


async def resolve_task_inline(
    request: ResolverCapabilityRequestV2,
    execution_context: TaskResolutionExecutionContextV1,
    *,
    inline_budget_seconds: float,
) -> TaskResolutionResultV1:
    """Run one authorized task-resolution request within a foreground budget.

    Args:
        request: Authorized V2 `task_resolution_request` capability row.
        execution_context: Trusted prompt-safe context for specialist adapters.
        inline_budget_seconds: Configured foreground wall-clock budget.

    Returns:
        A terminal result when work finishes inline, otherwise a deferred result
        carrying the same checkpoint for durable continuation.
    """

    _validate_inline_budget(inline_budget_seconds)
    context = validate_task_resolution_execution_context(execution_context)
    checkpoint = create_task_resolution_checkpoint(request, context)
    deadline = monotonic() + inline_budget_seconds
    result = await run_task_orchestrator(
        checkpoint,
        context,
        inline_deadline=deadline,
    )
    return result


async def start_task_resolution_in_background(
    request: ResolverCapabilityRequestV2,
    execution_context: TaskResolutionExecutionContextV1,
    *,
    source_trigger_source: str,
    source_platform_bot_id: str,
    requester_display_name: str,
    source_llm_trace_id: str = "",
) -> TaskResolutionResultV1:
    """Enter the durable handoff path directly without an inline specialist.

    The initial checkpoint is created and immediately materialized through the
    same accepted-task, pending-transition, and idempotent queue promotion used
    by deferred inline work.  The returned deferred result carries only the
    initial empty checkpoint, so no partial content is invented before the
    worker resumes.

    Args:
        request: Authorized V2 `task_resolution_request` capability row.
        execution_context: Trusted prompt-safe context for specialist adapters.
        source_trigger_source: Canonical episode trigger source for identity.
        source_platform_bot_id: Adapter bot id required by accepted tasks.
        requester_display_name: Adapter user display name for accepted tasks.

    Returns:
        The validated deferred result whose checkpoint is now durable.

    Raises:
        TaskResolutionContractError: When checkpoint, accepted-task, pending,
            or queue durability cannot be established.
    """

    context = validate_task_resolution_execution_context(execution_context)
    checkpoint = create_task_resolution_checkpoint(request, context)
    deferred_result = result_from_checkpoint(
        checkpoint,
        status="deferred",
        prompt_safe_summary="The task needs durable continuation.",
        completed_subgoals=[],
        coding_run_context={},
    )
    await promote_deferred_task_resolution(
        deferred_result,
        context,
        source_trigger_source=source_trigger_source,
        source_platform_bot_id=source_platform_bot_id,
        requester_display_name=requester_display_name,
        source_llm_trace_id=source_llm_trace_id,
    )
    return deferred_result


async def resume_task_resolution(
    checkpoint: TaskResolutionCheckpointV1,
    execution_context: TaskResolutionExecutionContextV1,
) -> TaskResolutionResultV1:
    """Resume the exact persisted checkpoint without resetting any counters."""

    validated_checkpoint = validate_task_resolution_checkpoint(checkpoint)
    context = validate_task_resolution_execution_context(execution_context)
    result = await run_task_orchestrator(
        validated_checkpoint,
        context,
        inline_deadline=None,
    )
    return result


async def promote_deferred_task_resolution(
    result: TaskResolutionResultV1,
    execution_context: TaskResolutionExecutionContextV1,
    *,
    source_trigger_source: str,
    source_platform_bot_id: str,
    requester_display_name: str,
    source_llm_trace_id: str = "",
) -> BackgroundWorkQueueResult:
    """Materialize one deferred inline session as the same durable v2 task.

    The accepted-task identity uses only the validated semantic objective and
    trusted requester/conversation scope. The checkpoint and trusted execution
    context remain internal to the v2 job and are never returned to cognition.
    Existing active tasks are reused only when their stored goal continuation
    reference matches the incoming validated context reference.
    """

    validated_result = validate_task_resolution_result(result)
    if validated_result["status"] != "deferred":
        raise TaskResolutionContractError(
            "task-resolution promotion requires a deferred result"
        )
    context = validate_task_resolution_execution_context(execution_context)
    checkpoint = validate_task_resolution_checkpoint(
        validated_result["checkpoint"],
    )
    _validate_deferred_result_context_binding(
        validated_result,
        checkpoint,
        context,
    )
    _require_promotion_text(source_trigger_source, "source_trigger_source")
    _require_promotion_text(source_platform_bot_id, "source_platform_bot_id")
    _require_promotion_text(requester_display_name, "requester_display_name")
    create_request: AcceptedTaskCreateRequest = {
        "task_kind": "task_resolution",
        "semantic_objective": checkpoint["semantic_objective"],
        "accepted_task_summary": checkpoint["semantic_objective"],
        "goal_continuation_ref": context["goal_continuation_ref"],
        "requested_delivery": BACKGROUND_WORK_REQUESTED_DELIVERY,
        "max_output_chars": context["max_output_chars"],
        "source_trigger_source": source_trigger_source.strip(),
        "source_platform": context["platform"],
        "source_channel_id": context["channel_id"],
        "source_channel_type": context["channel_type"],
        "source_message_id": context["source_message_id"],
        "source_platform_bot_id": source_platform_bot_id.strip(),
        "source_character_name": context["character_name"],
        "requester_global_user_id": context["requester_global_user_id"],
        "requester_platform_user_id": context["requester_platform_user_id"],
        "requester_display_name": requester_display_name.strip(),
        "storage_timestamp_utc": context["current_timestamp_utc"],
    }
    create_result = await create_or_return_active_accepted_task(create_request)
    accepted_task = create_result["task"]
    active_state = str(accepted_task.get("state", ""))
    if create_result["status"] == "already_active":
        stored_ref = accepted_task.get("goal_continuation_ref")
        if stored_ref != context["goal_continuation_ref"]:
            raise TaskResolutionContractError(
                "existing accepted task continuation reference does not "
                "match the incoming lineage"
            )
    if create_result["status"] == "already_active" and active_state not in {
        "enqueueing",
        "pending",
    }:
        return {
            "status": "pending",
            "job_id": str(accepted_task.get("executor_ref", "")),
            "job_ref": "",
            "accepted_task_id": str(accepted_task["accepted_task_id"]),
            "task_identity_key": str(accepted_task["task_identity_key"]),
            "accepted_task_summary": str(
                accepted_task["accepted_task_summary"],
            ),
            "acknowledgement_constraint": "promise_allowed",
            "wait_guidance": "non_numeric_wait",
            "result_summary": "The accepted task is already continuing.",
        }

    accepted_task_id = str(accepted_task["accepted_task_id"])
    if active_state == "pending":
        job_id = str(accepted_task.get("executor_ref", "")).strip()
        if not job_id:
            raise TaskResolutionContractError(
                "pending task-resolution continuation is missing its job id"
            )
    else:
        job_id = f"job-{accepted_task_id.removeprefix('task-')}"
        pending_task = await mark_accepted_task_pending(
            accepted_task_id=accepted_task_id,
            executor_ref=job_id,
            updated_at=context["current_timestamp_utc"],
        )
        if pending_task is None:
            await mark_accepted_task_enqueue_failed(
                accepted_task_id=accepted_task_id,
                failure_summary="Task continuation could not become pending.",
                updated_at=context["current_timestamp_utc"],
            )
            raise TaskResolutionContractError(
                "task-resolution durable promotion did not reach pending state"
            )
    queue_request: BackgroundWorkQueueRequest = {
        "job_id": job_id,
        "source_action_attempt_id": checkpoint["session_id"],
        "source_llm_trace_id": source_llm_trace_id.strip(),
        "idempotency_key": f"background_work:{accepted_task_id}",
        "accepted_task_id": accepted_task_id,
        "task_identity_key": accepted_task["task_identity_key"],
        "semantic_objective": checkpoint["semantic_objective"],
        "goal_continuation_ref": context["goal_continuation_ref"],
        "requested_worker": TASK_ORCHESTRATOR_WORKER,
        "worker_payload": {
            "schema_version": TASK_ORCHESTRATOR_WORKER_PAYLOAD_VERSION,
            "operation": "resume_task_resolution",
            "checkpoint": dict(checkpoint),
            "coding_request": None,
        },
        "task_execution_context": dict(context),
        "source_platform": context["platform"],
        "source_channel_id": context["channel_id"],
        "source_channel_type": context["channel_type"],
        "source_message_id": context["source_message_id"],
        "source_platform_bot_id": source_platform_bot_id.strip(),
        "source_character_name": context["character_name"],
        "requester_global_user_id": context["requester_global_user_id"],
        "requester_platform_user_id": context["requester_platform_user_id"],
        "requester_display_name": requester_display_name.strip(),
        "requested_delivery": BACKGROUND_WORK_REQUESTED_DELIVERY,
        "max_output_chars": context["max_output_chars"],
        "storage_timestamp_utc": context["current_timestamp_utc"],
    }
    try:
        queue_result = await enqueue_background_work_request(queue_request)
    except (ValueError, RuntimeError) as exc:
        await mark_accepted_task_enqueue_failed(
            accepted_task_id=accepted_task["accepted_task_id"],
            failure_summary="Task continuation could not be made durable.",
            updated_at=context["current_timestamp_utc"],
        )
        raise TaskResolutionContractError(
            "task-resolution durable promotion failed"
        ) from exc
    if queue_result["job_id"] != job_id:
        await mark_accepted_task_enqueue_failed(
            accepted_task_id=accepted_task_id,
            failure_summary="Task continuation returned an invalid job id.",
            updated_at=context["current_timestamp_utc"],
        )
        raise TaskResolutionContractError(
            "task-resolution durable promotion returned an unexpected job id"
        )
    return queue_result


def _validate_deferred_result_context_binding(
    result: TaskResolutionResultV1,
    checkpoint: TaskResolutionCheckpointV1,
    context: TaskResolutionExecutionContextV1,
) -> None:
    """Require promotion to preserve its original goal and scene lineage."""

    if result["semantic_objective"] != checkpoint["semantic_objective"]:
        raise TaskResolutionContractError(
            "deferred result objective does not match task checkpoint"
        )
    if result["scene_context"] != checkpoint["scene_context"]:
        raise TaskResolutionContractError(
            "deferred result scene does not match task checkpoint"
        )
    if result["goal_continuation_ref"] != checkpoint[
        "goal_continuation_ref"
    ]:
        raise TaskResolutionContractError(
            "deferred result continuation reference does not match task checkpoint"
        )
    if checkpoint["scene_context"] != context["scene_context"]:
        raise TaskResolutionContractError(
            "task checkpoint scene does not match execution context"
        )
    if checkpoint["goal_continuation_ref"] != context[
        "goal_continuation_ref"
    ]:
        raise TaskResolutionContractError(
            "task checkpoint continuation reference does not match execution context"
        )


def _validate_inline_budget(inline_budget_seconds: float) -> None:
    """Validate the configurable foreground wall-clock bound."""

    if not isinstance(inline_budget_seconds, float):
        raise TaskResolutionContractError(
            "inline_budget_seconds: expected float"
        )
    if (
        inline_budget_seconds < MINIMUM_INLINE_BUDGET_SECONDS
        or inline_budget_seconds > MAXIMUM_INLINE_BUDGET_SECONDS
    ):
        raise TaskResolutionContractError(
            "inline_budget_seconds: expected value between 1.0 and 120.0"
        )


def _require_promotion_text(value: object, field_name: str) -> None:
    """Require trusted source text that is absent from the execution context."""

    if not isinstance(value, str) or not value.strip():
        raise TaskResolutionContractError(f"{field_name}: expected non-empty text")
