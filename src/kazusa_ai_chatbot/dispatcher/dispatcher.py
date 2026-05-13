"""Validation, deduplication, and scheduler persistence for tool calls."""

from __future__ import annotations

import logging
import time

from kazusa_ai_chatbot import event_logging, scheduler
from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.dispatcher.evaluator import ToolCallEvaluator
from kazusa_ai_chatbot.dispatcher.pending_index import PendingTaskIndex
from kazusa_ai_chatbot.dispatcher.task import DispatchContext, DispatchResult, RawToolCall

logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
DISPATCHER_COMPONENT = "dispatcher.dispatcher"
DISPATCHER_SCHEDULER_COMPONENT = "dispatcher.scheduler"
SCHEDULED_EVENTS_COLLECTION = "scheduled_events"


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic milliseconds since a start marker."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = max(0, int(elapsed * MILLISECONDS_PER_SECOND))
    return elapsed_ms


def _dispatch_correlation_id(ctx: DispatchContext) -> str:
    """Build a non-content correlation id for one dispatcher call."""

    message_ref = ctx.source_message_id or "no-message-id"
    correlation_id = f"dispatch:{ctx.source_platform}:{message_ref}"
    return correlation_id


def _rejection_code(reason: str) -> str:
    """Map a validation reason to a privacy-safe rejection code."""

    if "duplicate task" in reason:
        code = "duplicate_task"
    elif "scheduler write failed" in reason:
        code = "scheduler_write_failed"
    elif "no adapters registered" in reason:
        code = "no_adapters_registered"
    elif "unknown_platform" in reason:
        code = "unknown_platform"
    elif "unknown or unavailable tool" in reason:
        code = "unknown_tool"
    elif "unparseable execute_at" in reason:
        code = "unparseable_execute_at"
    elif "target_channel_type required" in reason:
        code = "missing_target_channel_type"
    elif "missing required field" in reason:
        code = "missing_required_field"
    elif "expected one of" in reason:
        code = "invalid_enum"
    elif "expected" in reason:
        code = "invalid_type"
    else:
        code = "validation_failed"
    return code


def _rejection_codes(reason: str) -> list[str]:
    """Return stable rejection codes from a combined validation reason."""

    parts = [
        part.strip()
        for part in reason.split(";")
        if part.strip()
    ]
    if not parts:
        parts = [reason]
    codes = [_rejection_code(part) for part in parts]
    return codes


def _adapter_available_from_codes(codes: list[str]) -> bool:
    """Return whether dispatcher rejection codes indicate adapter availability."""

    unavailable_codes = {"no_adapters_registered", "unknown_platform"}
    adapter_available = not any(code in unavailable_codes for code in codes)
    return adapter_available


class TaskDispatcher:
    """Turn raw tool calls into scheduled tasks through one unified path."""

    def __init__(
        self,
        evaluator: ToolCallEvaluator,
        pending_index: PendingTaskIndex,
    ) -> None:
        self._evaluator = evaluator
        self._pending_index = pending_index

    async def dispatch(
        self,
        raw_calls: list[RawToolCall],
        ctx: DispatchContext,
        *,
        instruction: str = "",
    ) -> DispatchResult:
        """Validate, deduplicate, and schedule raw tool calls.

        Args:
            raw_calls: Raw tool calls emitted by the consolidator LLM.
            ctx: Source-side dispatch context used for substitution and
                permissions.
            instruction: Optional natural-language trace tag carried onto tasks.

        Returns:
            ``DispatchResult`` containing scheduled tasks and rejections.
        """

        scheduled: list[tuple] = []
        rejected: list[tuple[RawToolCall, str]] = []
        seen_signatures: set[str] = set()
        correlation_id = _dispatch_correlation_id(ctx)

        for raw in raw_calls:
            evaluation = self._evaluator.evaluate(raw, ctx)
            if not evaluation.ok or evaluation.task is None:
                reason = "; ".join(evaluation.errors)
                rejected.append((raw, reason))
                codes = _rejection_codes(reason)
                await event_logging.record_dispatcher_event(
                    component=DISPATCHER_COMPONENT,
                    action_kind=raw.tool,
                    validation_status="rejected",
                    adapter_available=_adapter_available_from_codes(codes),
                    status="rejected",
                    rejection_codes=codes,
                    correlation_id=correlation_id,
                    severity="warning",
                )
                continue

            task = evaluation.task
            if instruction.strip():
                task.tags.append(instruction.strip())

            signature = self._pending_index.signature_for(task)
            if signature in seen_signatures or self._pending_index.contains(task):
                reason = "duplicate task"
                rejected.append((raw, reason))
                await event_logging.record_dispatcher_event(
                    component=DISPATCHER_COMPONENT,
                    action_kind=task.tool,
                    validation_status="duplicate",
                    adapter_available=True,
                    status="rejected",
                    rejection_codes=["duplicate_task"],
                    correlation_id=correlation_id,
                    severity="warning",
                )
                continue

            event_doc = task.to_scheduler_doc(ctx)
            scheduler_write_started_at = time.perf_counter()
            try:
                event_id = await scheduler.schedule_event(event_doc)
            except DatabaseOperationError as exc:
                logger.exception(
                    f"Failed to persist scheduled task for tool {task.tool}: {exc}"
                )
                await event_logging.record_database_operation_event(
                    component=DISPATCHER_SCHEDULER_COMPONENT,
                    collection=SCHEDULED_EVENTS_COLLECTION,
                    operation_kind="insert_scheduled_event",
                    status="failed",
                    idempotency_result=f"exception:{exc.__class__.__name__}",
                    latency_ms=_elapsed_ms(scheduler_write_started_at),
                    correlation_id=correlation_id,
                    severity="warning",
                )
                await event_logging.record_dispatcher_event(
                    component=DISPATCHER_COMPONENT,
                    action_kind=task.tool,
                    validation_status="accepted",
                    adapter_available=True,
                    status="failed",
                    rejection_codes=["scheduler_write_failed"],
                    correlation_id=correlation_id,
                    severity="warning",
                )
                rejected.append((raw, "scheduler write failed"))
                continue

            await event_logging.record_database_operation_event(
                component=DISPATCHER_SCHEDULER_COMPONENT,
                collection=SCHEDULED_EVENTS_COLLECTION,
                operation_kind="insert_scheduled_event",
                status="succeeded",
                idempotency_result="inserted",
                latency_ms=_elapsed_ms(scheduler_write_started_at),
                document_ref=event_id,
                correlation_id=correlation_id,
            )
            await event_logging.record_dispatcher_event(
                component=DISPATCHER_COMPONENT,
                action_kind=task.tool,
                validation_status="accepted",
                adapter_available=True,
                status="scheduled",
                scheduled_event_ids=[event_id],
                correlation_id=correlation_id,
            )
            self._pending_index.add(event_id, task)
            seen_signatures.add(signature)
            scheduled.append((task, event_id))

        return_value = DispatchResult(scheduled=scheduled, rejected=rejected)
        return return_value
