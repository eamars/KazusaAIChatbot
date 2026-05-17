"""Process-local idle worker for self-cognition cycles."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.config import (
    SELF_COGNITION_MAX_CASES_PER_TICK,
    SELF_COGNITION_TRACKING_DIR,
    SELF_COGNITION_WORKER_INTERVAL_SECONDS,
)
from kazusa_ai_chatbot import db, event_logging
from kazusa_ai_chatbot.nodes.dialog_agent import StateContractError
from kazusa_ai_chatbot.self_cognition import models, runner
from kazusa_ai_chatbot.self_cognition import sources as source_collectors
from kazusa_ai_chatbot.time_boundary import storage_utc_now

logger = logging.getLogger(__name__)

_WORKER_STOP_TIMEOUT_SECONDS = 5.0
_ATTEMPT_HISTORY_LIMIT = 1000
_SAFE_PATH_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class SelfCognitionWorkerResult:
    """Outcome counters for one self-cognition worker tick."""

    processed_count: int = 0
    failed_count: int = 0
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
    character_profile_provider: Callable[[], dict[str, Any]],
    output_root: str | Path = SELF_COGNITION_TRACKING_DIR,
) -> SelfCognitionWorkerHandle:
    """Start the process-local self-cognition worker loop.

    Args:
        is_primary_interaction_busy: Service load probe.
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
    now: datetime,
    is_primary_interaction_busy: Callable[[], bool],
    character_profile: dict[str, Any] | None = None,
    collect_cases_func: Callable[..., Any] | None = None,
    run_case_func: Callable[..., Any] | None = None,
    read_attempts_func: Callable[..., Any] | None = None,
    record_attempt_func: Callable[..., Any] | None = None,
    complete_scheduled_event_func: Callable[..., Any] | None = None,
    claim_scheduled_event_func: Callable[..., Any] | None = None,
    max_cases: int = SELF_COGNITION_MAX_CASES_PER_TICK,
) -> SelfCognitionWorkerResult:
    """Run one bounded self-cognition worker tick.

    Args:
        output_root: Compatibility path passed only to injected test seams.
        now: Current worker tick time.
        is_primary_interaction_busy: Service load probe.
        character_profile: Current character state snapshot.
        collect_cases_func: Optional test seam for source collection.
        run_case_func: Optional test seam for self-cognition case projection.
        read_attempts_func: Optional test seam for prior attempt reads.
        record_attempt_func: Optional test seam for attempt persistence.
        complete_scheduled_event_func: Optional test seam for source slot
            completion.
        claim_scheduled_event_func: Optional test seam for atomically claiming
            source scheduler rows before processing.
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
    active_read_attempts = read_attempts_func or (
        db.list_self_cognition_action_attempts
    )
    active_record_attempt = record_attempt_func or (
        db.upsert_self_cognition_action_attempt
    )
    active_complete_scheduled_event = (
        complete_scheduled_event_func or db.mark_scheduled_event_completed
    )
    active_claim_scheduled_event = (
        claim_scheduled_event_func or db.claim_pending_scheduled_event_running
    )

    for case in cases[:max_cases]:
        if is_primary_interaction_busy():
            result.deferred = True
            result.defer_reason = "primary interaction busy"
            break
        claimed = await _claim_scheduled_future_cognition_event(
            case,
            claim_scheduled_event_func=active_claim_scheduled_event,
        )
        if not claimed:
            result.skipped_count += 1
            continue
        prior_attempts = await _call_maybe_async(
            active_read_attempts,
            limit=_ATTEMPT_HISTORY_LIMIT,
        )
        case_for_run = _case_with_prior_attempts(case, prior_attempts)
        output_dir = _case_output_dir(root, now=now, case=case_for_run)
        try:
            if run_case_func is None:
                artifact_payloads = (
                    await runner.build_self_cognition_case_artifacts_async(
                        case_for_run,
                        apply_consolidation=True,
                        execute_private_actions=True,
                    )
                )
            else:
                artifact_payloads = await _call_maybe_async(
                    run_case_func,
                    case_for_run,
                    output_dir,
                )
        except StateContractError as exc:
            result.failed_count += 1
            await event_logging.record_runtime_error_event(
                component="self_cognition.worker",
                error_class="StateContractError",
                error_preview=str(exc),
                stack_fingerprint="self_cognition_case_state_contract",
                top_frame_module=__name__,
                recovered=True,
            )
            continue
        result.processed_count += 1
        dispatch_status = await _handle_case_action_outputs(
            artifact_payloads=artifact_payloads,
            now=now,
            record_attempt_func=active_record_attempt,
        )
        await _record_self_cognition_event_from_artifacts(
            case=case_for_run,
            artifact_payloads=artifact_payloads,
            dispatch_status=dispatch_status,
        )
        await _complete_scheduled_future_cognition_event(
            case_for_run,
            complete_scheduled_event_func=active_complete_scheduled_event,
        )

    await _record_worker_tick_event(result)
    return result


async def _self_cognition_worker_loop(
    *,
    stop_event: asyncio.Event,
    is_primary_interaction_busy: Callable[[], bool],
    character_profile_provider: Callable[[], dict[str, Any]],
    output_root: str | Path,
) -> None:
    """Run self-cognition scheduling ticks until stopped."""

    while not stop_event.is_set():
        try:
            character_profile = character_profile_provider()
            await run_self_cognition_worker_tick(
                output_root=output_root,
                now=storage_utc_now(),
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
    artifact_payloads: dict[str, Any],
    now: datetime,
    record_attempt_func: Callable[..., Any],
) -> str:
    """Record private action attempts for one case without scheduling output."""

    action_attempt = artifact_payloads.get(models.ARTIFACT_ACTION_ATTEMPT)
    if not isinstance(action_attempt, dict):
        return_value = "not_requested"
        return return_value

    attempt_state = _attempt_state(action_attempt, now=now)
    await _call_maybe_async(record_attempt_func, attempt_state)
    return_value = "not_requested"
    return return_value


async def _record_worker_tick_event(result: SelfCognitionWorkerResult) -> None:
    """Record one sanitized self-cognition worker tick summary."""

    if result.deferred:
        status = "deferred"
    elif result.processed_count == 0 and result.failed_count > 0:
        status = "failed"
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
        failed_count=result.failed_count,
        skipped_count=result.skipped_count,
        deferred=result.deferred,
        defer_reason=result.defer_reason,
    )


async def _claim_scheduled_future_cognition_event(
    case: models.SelfCognitionCase,
    *,
    claim_scheduled_event_func: Callable[..., Any],
) -> bool:
    """Atomically claim a source scheduled cognition slot before processing."""

    if case.get("trigger_kind") != models.TRIGGER_SCHEDULED_FUTURE_COGNITION:
        return_value = True
        return return_value

    event_id = case.get("source_scheduled_event_id")
    if not isinstance(event_id, str) or not event_id:
        return_value = False
        return return_value

    claimed = await _call_maybe_async(claim_scheduled_event_func, event_id)
    return_value = bool(claimed)
    return return_value


async def _complete_scheduled_future_cognition_event(
    case: models.SelfCognitionCase,
    *,
    complete_scheduled_event_func: Callable[..., Any],
) -> None:
    """Mark a source scheduled cognition slot consumed after processing."""

    if case.get("trigger_kind") != models.TRIGGER_SCHEDULED_FUTURE_COGNITION:
        return

    event_id = case.get("source_scheduled_event_id")
    if not isinstance(event_id, str) or not event_id:
        return

    await _call_maybe_async(complete_scheduled_event_func, event_id)


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
    now: datetime,
) -> dict[str, Any]:
    """Build the persisted state row for one action attempt."""

    attempt_state = dict(action_attempt)
    attempt_state["recorded_at"] = now.isoformat()
    return attempt_state


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
