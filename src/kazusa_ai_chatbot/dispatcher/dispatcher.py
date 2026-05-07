"""Validation, deduplication, and scheduler persistence for tool calls."""

from __future__ import annotations

import logging

from kazusa_ai_chatbot import scheduler
from kazusa_ai_chatbot.db import DatabaseOperationError
from kazusa_ai_chatbot.dispatcher.evaluator import ToolCallEvaluator
from kazusa_ai_chatbot.dispatcher.pending_index import PendingTaskIndex
from kazusa_ai_chatbot.dispatcher.task import DispatchContext, DispatchResult, RawToolCall

logger = logging.getLogger(__name__)


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
            ctx: Source-side dispatch context used for defaulting.
            instruction: Optional natural-language trace tag carried onto tasks.

        Returns:
            ``DispatchResult`` containing scheduled tasks and rejections.
        """

        scheduled: list[tuple] = []
        rejected: list[tuple[RawToolCall, str]] = []
        seen_signatures: set[str] = set()

        for raw in raw_calls:
            evaluation = self._evaluator.evaluate(raw, ctx)
            if not evaluation.ok or evaluation.task is None:
                rejected.append((raw, "; ".join(evaluation.errors)))
                continue

            task = evaluation.task
            if instruction.strip():
                task.tags.append(instruction.strip())

            signature = self._pending_index.signature_for(task)
            if signature in seen_signatures or self._pending_index.contains(task):
                rejected.append((raw, "duplicate task"))
                continue

            event_doc = task.to_scheduler_doc(ctx)
            try:
                event_id = await scheduler.schedule_event(event_doc)
            except DatabaseOperationError as exc:
                logger.exception(
                    f"Failed to persist scheduled task for tool {task.tool}: {exc}"
                )
                rejected.append((raw, "scheduler write failed"))
                continue

            self._pending_index.add(event_id, task)
            seen_signatures.add(signature)
            scheduled.append((task, event_id))

        return_value = DispatchResult(scheduled=scheduled, rejected=rejected)
        return return_value
