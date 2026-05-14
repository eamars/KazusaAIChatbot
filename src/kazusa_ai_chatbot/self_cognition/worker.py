"""Process-local idle worker for self-cognition cycles."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.config import (
    SELF_COGNITION_MAX_CASES_PER_TICK,
    SELF_COGNITION_TRACKING_DIR,
    SELF_COGNITION_WORKER_INTERVAL_SECONDS,
)
from kazusa_ai_chatbot import db, event_logging
from kazusa_ai_chatbot.dispatcher import TaskDispatcher
from kazusa_ai_chatbot.self_cognition import handoff, models, runner
from kazusa_ai_chatbot.self_cognition import sources as source_collectors

logger = logging.getLogger(__name__)

_WORKER_STOP_TIMEOUT_SECONDS = 5.0
_ATTEMPT_HISTORY_LIMIT = 1000
_SAFE_PATH_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class SelfCognitionWorkerResult:
    """Outcome counters for one self-cognition worker tick."""

    processed_count: int = 0
    dispatched_count: int = 0
    rejected_count: int = 0
    skipped_count: int = 0
    deferred: bool = False
    defer_reason: str = ""
    artifact_paths: list[str] = field(default_factory=list)


@dataclass
class SelfCognitionWorkerHandle:
    """Process-local worker task and stop signal owned by service lifespan."""

    task: asyncio.Task
    stop_event: asyncio.Event


def start_self_cognition_worker(
    *,
    is_primary_interaction_busy: Callable[[], bool],
    dispatcher: TaskDispatcher,
    character_profile_provider: Callable[[], dict[str, Any]],
    output_root: str | Path = SELF_COGNITION_TRACKING_DIR,
) -> SelfCognitionWorkerHandle:
    """Start the process-local self-cognition worker loop.

    Args:
        is_primary_interaction_busy: Service load probe.
        dispatcher: Existing task dispatcher for outbound handoff.
        character_profile_provider: Callable returning current character state.
        output_root: Compatibility path passed only to injected test seams.

    Returns:
        Worker handle used for shutdown.
    """

    stop_event = asyncio.Event()
    task = asyncio.create_task(
        _self_cognition_worker_loop(
            stop_event=stop_event,
            is_primary_interaction_busy=is_primary_interaction_busy,
            dispatcher=dispatcher,
            character_profile_provider=character_profile_provider,
            output_root=output_root,
        )
    )
    handle = SelfCognitionWorkerHandle(task=task, stop_event=stop_event)
    logger.info("Self-cognition worker started")
    return handle


async def stop_self_cognition_worker(
    handle: SelfCognitionWorkerHandle,
) -> None:
    """Stop a self-cognition worker handle created by the public starter."""

    handle.stop_event.set()
    task = handle.task
    try:
        await asyncio.wait_for(task, timeout=_WORKER_STOP_TIMEOUT_SECONDS)
    except TimeoutError:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    logger.info("Self-cognition worker stopped")


async def run_self_cognition_worker_tick(
    *,
    output_root: str | Path,
    dispatcher: TaskDispatcher | None,
    now: datetime,
    is_primary_interaction_busy: Callable[[], bool],
    character_profile: dict[str, Any] | None = None,
    collect_cases_func: Callable[..., Any] | None = None,
    run_case_func: Callable[..., Any] | None = None,
    dispatch_candidate_func: Callable[..., Any] | None = None,
    read_attempts_func: Callable[..., Any] | None = None,
    record_attempt_func: Callable[..., Any] | None = None,
    max_cases: int = SELF_COGNITION_MAX_CASES_PER_TICK,
) -> SelfCognitionWorkerResult:
    """Run one bounded self-cognition worker tick.

    Args:
        output_root: Compatibility path passed only to injected test seams.
        dispatcher: Existing task dispatcher, or `None` for no handoff.
        now: Current worker tick time.
        is_primary_interaction_busy: Service load probe.
        character_profile: Current character state snapshot.
        collect_cases_func: Optional test seam for source collection.
        run_case_func: Optional test seam for self-cognition case projection.
        dispatch_candidate_func: Optional test seam for dispatcher handoff.
        read_attempts_func: Optional test seam for prior attempt reads.
        record_attempt_func: Optional test seam for attempt persistence.
        max_cases: Maximum cases to process in this tick.

    Returns:
        Tick counters and production outcome metadata.
    """

    if is_primary_interaction_busy():
        result = SelfCognitionWorkerResult(
            deferred=True,
            defer_reason="primary interaction busy",
        )
        await _record_worker_tick_event(result)
        return result

    active_profile = character_profile or {}
    cases = await _collect_cases(
        now=now,
        character_profile=active_profile,
        max_cases=max_cases,
        collect_cases_func=collect_cases_func,
    )
    if not cases:
        result = SelfCognitionWorkerResult(skipped_count=1)
        await _record_worker_tick_event(result)
        return result

    root = Path(output_root)
    result = SelfCognitionWorkerResult()
    active_dispatch_candidate = (
        dispatch_candidate_func or handoff.dispatch_action_candidate
    )
    active_read_attempts = read_attempts_func or (
        db.list_self_cognition_action_attempts
    )
    active_record_attempt = record_attempt_func or (
        db.upsert_self_cognition_action_attempt
    )

    for case in cases[:max_cases]:
        if is_primary_interaction_busy():
            result.deferred = True
            result.defer_reason = "primary interaction busy"
            break
        prior_attempts = await _call_maybe_async(
            active_read_attempts,
            limit=_ATTEMPT_HISTORY_LIMIT,
        )
        case_for_run = _case_with_prior_attempts(case, prior_attempts)
        output_dir = _case_output_dir(root, now=now, case=case_for_run)
        if run_case_func is None:
            artifact_payloads = await runner.build_self_cognition_case_artifacts_async(
                case_for_run,
                apply_consolidation=True,
            )
        else:
            artifact_payloads = await _call_maybe_async(
                run_case_func,
                case_for_run,
                output_dir,
            )
        result.processed_count += 1
        dispatch_status = await _handle_case_action_outputs(
            case=case_for_run,
            artifact_payloads=artifact_payloads,
            dispatcher=dispatcher,
            now=now,
            dispatch_candidate_func=active_dispatch_candidate,
            record_attempt_func=active_record_attempt,
            result=result,
        )
        await _record_self_cognition_event_from_artifacts(
            case=case_for_run,
            artifact_payloads=artifact_payloads,
            dispatch_status=dispatch_status,
        )

    await _record_worker_tick_event(result)
    return result


async def _self_cognition_worker_loop(
    *,
    stop_event: asyncio.Event,
    is_primary_interaction_busy: Callable[[], bool],
    dispatcher: TaskDispatcher,
    character_profile_provider: Callable[[], dict[str, Any]],
    output_root: str | Path,
) -> None:
    """Run self-cognition scheduling ticks until stopped."""

    while not stop_event.is_set():
        try:
            character_profile = character_profile_provider()
            await run_self_cognition_worker_tick(
                output_root=output_root,
                dispatcher=dispatcher,
                now=datetime.now(timezone.utc),
                is_primary_interaction_busy=is_primary_interaction_busy,
                character_profile=character_profile,
            )
        except Exception as exc:
            logger.exception(f"Self-cognition worker tick failed: {exc}")
            await event_logging.record_runtime_error_event(
                component="self_cognition.worker",
                error_class=type(exc).__name__,
                error_preview=str(exc),
                stack_fingerprint="self_cognition_worker_tick",
                top_frame_module=__name__,
                recovered=True,
            )
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=SELF_COGNITION_WORKER_INTERVAL_SECONDS,
            )
        except TimeoutError:
            continue


async def _collect_cases(
    *,
    now: datetime,
    character_profile: dict[str, Any],
    max_cases: int,
    collect_cases_func: Callable[..., Any] | None,
) -> list[models.SelfCognitionCase]:
    """Collect worker cases through the default source collector or a seam."""

    if collect_cases_func is not None:
        cases = await _call_maybe_async(
            collect_cases_func,
            now=now,
            max_cases=max_cases,
        )
    else:
        cases = await source_collectors.collect_self_cognition_cases(
            now=now,
            character_profile=character_profile,
            max_cases=max_cases,
        )
    return cases


async def _handle_case_action_outputs(
    *,
    case: models.SelfCognitionCase,
    artifact_payloads: dict[str, Any],
    dispatcher: TaskDispatcher | None,
    now: datetime,
    dispatch_candidate_func: Callable[..., Any],
    record_attempt_func: Callable[..., Any],
    result: SelfCognitionWorkerResult,
) -> str:
    """Record action attempts and optional dispatcher handoff for one case."""

    action_attempt = artifact_payloads.get(models.ARTIFACT_ACTION_ATTEMPT)
    if not isinstance(action_attempt, dict):
        return_value = "not_requested"
        return return_value

    action_candidate = artifact_payloads.get(models.ARTIFACT_ACTION_CANDIDATE)
    if not isinstance(action_candidate, dict):
        attempt_state = _attempt_state(
            action_attempt,
            dispatch_result=None,
            now=now,
        )
        await _call_maybe_async(record_attempt_func, attempt_state)
        return_value = "not_requested"
        return return_value

    if dispatcher is None:
        dispatch_result = _not_requested_dispatch_result(action_attempt)
    else:
        dispatch_result = await _call_maybe_async(
            dispatch_candidate_func,
            case,
            action_attempt,
            action_candidate,
            dispatcher,
            now=now,
        )

    if dispatch_result["status"] == "accepted":
        result.dispatched_count += 1
    elif dispatch_result["status"] == "rejected":
        result.rejected_count += 1
    await event_logging.record_dispatcher_event(
        component="self_cognition.worker",
        action_kind=models.ACTION_KIND_SEND_MESSAGE,
        validation_status=str(dispatch_result["status"]),
        adapter_available=bool(dispatcher is not None),
        status=str(dispatch_result["status"]),
        scheduled_event_ids=[
            str(event_id)
            for event_id in dispatch_result["scheduled_event_ids"]
        ],
        rejection_codes=[
            str(rejection)
            for rejection in dispatch_result["rejections"]
        ],
        attempt_id=str(action_attempt["attempt_id"]),
    )

    attempt_state = _attempt_state(
        action_attempt,
        dispatch_result=dispatch_result,
        now=now,
    )
    await _call_maybe_async(record_attempt_func, attempt_state)
    return_value = str(dispatch_result["status"])
    return return_value


async def _record_worker_tick_event(result: SelfCognitionWorkerResult) -> None:
    """Record one sanitized self-cognition worker tick summary."""

    if result.deferred:
        status = "deferred"
    elif result.processed_count == 0 and result.skipped_count > 0:
        status = "skipped"
    else:
        status = "completed"
    await event_logging.record_worker_event(
        event_type="tick",
        component="self_cognition.worker",
        worker_name="self_cognition",
        enabled=True,
        dry_run=False,
        run_kind="self_cognition_tick",
        status=status,
        processed_count=result.processed_count,
        succeeded_count=result.dispatched_count,
        failed_count=result.rejected_count,
        skipped_count=result.skipped_count,
        deferred=result.deferred,
        defer_reason=result.defer_reason,
    )


async def _record_self_cognition_event_from_artifacts(
    *,
    case: models.SelfCognitionCase,
    artifact_payloads: dict[str, Any],
    dispatch_status: str,
) -> None:
    """Mirror self-cognition records into event logging."""

    trigger_record = artifact_payloads.get(models.ARTIFACT_TRIGGER_RECORD)
    run_record = artifact_payloads.get(models.ARTIFACT_RUN_RECORD)
    if not isinstance(trigger_record, dict) or not isinstance(run_record, dict):
        return
    action_attempt = artifact_payloads.get(models.ARTIFACT_ACTION_ATTEMPT)
    if not isinstance(action_attempt, dict):
        action_attempt = {}
    consolidation_outcome = artifact_payloads.get(
        models.ARTIFACT_CONSOLIDATION_OUTCOME
    )
    if not isinstance(consolidation_outcome, dict):
        consolidation_outcome = None
    budget = run_record["budget"]
    await event_logging.record_self_cognition_event(
        component="self_cognition.worker",
        case_id=str(case.get("case_id") or ""),
        trigger_kind=str(trigger_record["trigger_kind"]),
        selected_route=str(run_record["selected_route"]),
        output_mode=str(run_record["output_mode"]),
        budget={
            "rag_calls": int(budget["rag_calls"]),
            "cognition_calls": int(budget["cognition_calls"]),
            "dialog_calls": int(budget["dialog_calls"]),
            "topic_limit": int(budget["topic_limit"]),
        },
        dispatch_status=dispatch_status,
        status=str(run_record["status"]),
        trigger_id=str(trigger_record["trigger_id"]),
        run_id=str(run_record["run_id"]),
        attempt_id=str(action_attempt.get("attempt_id") or ""),
        consolidation_outcome=consolidation_outcome,
    )


def _case_with_prior_attempts(
    case: models.SelfCognitionCase,
    prior_attempts: list[dict[str, Any]],
) -> models.SelfCognitionCase:
    """Copy a case and prepend prior attempts for duplicate checks."""

    case_for_run: models.SelfCognitionCase = dict(case)
    existing_attempts = case.get("existing_attempts")
    if not isinstance(existing_attempts, list):
        existing_attempts = []
    case_for_run["existing_attempts"] = [
        *prior_attempts,
        *[
            attempt
            for attempt in existing_attempts
            if isinstance(attempt, dict)
        ],
    ]
    return case_for_run


def _attempt_state(
    action_attempt: dict[str, Any],
    *,
    dispatch_result: dict[str, Any] | None,
    now: datetime,
) -> dict[str, Any]:
    """Build the persisted state row for one action attempt."""

    attempt_state = dict(action_attempt)
    attempt_state["recorded_at"] = now.isoformat()
    if dispatch_result is None:
        return attempt_state

    dispatch_status = dispatch_result["status"]
    attempt_state["dispatch_status"] = dispatch_status
    attempt_state["scheduled_event_ids"] = list(
        dispatch_result["scheduled_event_ids"]
    )
    if dispatch_status == "accepted":
        attempt_state["status"] = models.ACTION_ATTEMPT_STATUS_SCHEDULED
    elif dispatch_status == "rejected":
        attempt_state["status"] = models.ACTION_ATTEMPT_STATUS_HELD
    return attempt_state


def _not_requested_dispatch_result(action_attempt: dict[str, Any]) -> dict[str, Any]:
    """Build a dispatch artifact for a candidate when no dispatcher is supplied."""

    dispatch_result = {
        "attempt_id": action_attempt["attempt_id"],
        "idempotency_key": action_attempt["idempotency_key"],
        "production_handoff": False,
        "dispatcher_called": False,
        "scheduled_event_ids": [],
        "rejections": ["dispatcher unavailable"],
        "status": "not_requested",
    }
    return dispatch_result


def _case_output_dir(
    root: Path,
    *,
    now: datetime,
    case: models.SelfCognitionCase,
) -> Path:
    """Build a stable local artifact directory for one worker case."""

    timestamp_slug = now.strftime("%Y%m%dT%H%M%SZ")
    case_id = str(case.get("case_id") or case.get("case_name") or "case")
    safe_case_id = _SAFE_PATH_COMPONENT_PATTERN.sub("_", case_id).strip("_")
    if not safe_case_id:
        safe_case_id = "case"
    output_dir = root / timestamp_slug / safe_case_id
    return output_dir


async def _call_maybe_async(
    callable_object: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call a sync or async test seam with a common awaitable contract."""

    value = callable_object(*args, **kwargs)
    if inspect.isawaitable(value):
        value = await value
    return value
