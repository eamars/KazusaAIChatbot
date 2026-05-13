"""Process-local idle worker for self-cognition cycles."""

from __future__ import annotations

import asyncio
import inspect
import json
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
from kazusa_ai_chatbot.dispatcher import TaskDispatcher
from kazusa_ai_chatbot.self_cognition import artifacts, handoff, models, runner
from kazusa_ai_chatbot.self_cognition import sources as source_collectors

logger = logging.getLogger(__name__)

_WORKER_STOP_TIMEOUT_SECONDS = 5.0
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
        output_root: Local tracking root.

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
    max_cases: int = SELF_COGNITION_MAX_CASES_PER_TICK,
) -> SelfCognitionWorkerResult:
    """Run one bounded self-cognition worker tick.

    Args:
        output_root: Local tracking root for artifacts and ledger.
        dispatcher: Existing task dispatcher, or `None` for no handoff.
        now: Current worker tick time.
        is_primary_interaction_busy: Service load probe.
        character_profile: Current character state snapshot.
        collect_cases_func: Optional test seam for source collection.
        run_case_func: Optional test seam for self-cognition case execution.
        dispatch_candidate_func: Optional test seam for dispatcher handoff.
        max_cases: Maximum cases to process in this tick.

    Returns:
        Tick counters and artifact paths.
    """

    if is_primary_interaction_busy():
        result = SelfCognitionWorkerResult(
            deferred=True,
            defer_reason="primary interaction busy",
        )
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
        return result

    root = Path(output_root)
    result = SelfCognitionWorkerResult()
    active_run_case = run_case_func or runner.run_self_cognition_case_async
    active_dispatch_candidate = (
        dispatch_candidate_func or handoff.dispatch_action_candidate
    )

    for case in cases[:max_cases]:
        if is_primary_interaction_busy():
            result.deferred = True
            result.defer_reason = "primary interaction busy"
            break
        prior_attempts = artifacts.read_action_attempt_ledger(root)
        case_for_run = _case_with_prior_attempts(case, prior_attempts)
        output_dir = _case_output_dir(root, now=now, case=case_for_run)
        paths = await _call_maybe_async(
            active_run_case,
            case_for_run,
            output_dir,
        )
        result.processed_count += 1
        result.artifact_paths.extend(str(path) for path in paths.values())
        await _handle_case_action_outputs(
            case=case_for_run,
            output_dir=output_dir,
            paths=paths,
            dispatcher=dispatcher,
            now=now,
            dispatch_candidate_func=active_dispatch_candidate,
            result=result,
            output_root=root,
        )

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
    output_dir: Path,
    paths: dict[str, str],
    dispatcher: TaskDispatcher | None,
    now: datetime,
    dispatch_candidate_func: Callable[..., Any],
    result: SelfCognitionWorkerResult,
    output_root: Path,
) -> None:
    """Record action attempts and optional dispatcher handoff for one case."""

    attempt_path = paths.get(models.ARTIFACT_ACTION_ATTEMPT)
    if not attempt_path:
        return

    action_attempt = _read_json(attempt_path)
    candidate_path = paths.get(models.ARTIFACT_ACTION_CANDIDATE)
    if not candidate_path:
        ledger_attempt = _ledger_attempt(
            action_attempt,
            dispatch_result=None,
            now=now,
        )
        artifacts.append_action_attempt_ledger(output_root, ledger_attempt)
        return

    action_candidate = _read_json(candidate_path)
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

    dispatch_paths = artifacts.write_tracking_artifacts(
        output_dir,
        {models.ARTIFACT_DISPATCH_RESULT: dispatch_result},
    )
    result.artifact_paths.extend(str(path) for path in dispatch_paths.values())
    if dispatch_result["status"] == "accepted":
        result.dispatched_count += 1
    elif dispatch_result["status"] == "rejected":
        result.rejected_count += 1

    ledger_attempt = _ledger_attempt(
        action_attempt,
        dispatch_result=dispatch_result,
        now=now,
    )
    artifacts.append_action_attempt_ledger(output_root, ledger_attempt)


def _case_with_prior_attempts(
    case: models.SelfCognitionCase,
    prior_attempts: list[dict[str, Any]],
) -> models.SelfCognitionCase:
    """Copy a case and prepend local ledger attempts for duplicate checks."""

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


def _ledger_attempt(
    action_attempt: dict[str, Any],
    *,
    dispatch_result: dict[str, Any] | None,
    now: datetime,
) -> dict[str, Any]:
    """Build the local ledger row for one action attempt."""

    ledger_row = dict(action_attempt)
    ledger_row["recorded_at"] = now.isoformat()
    if dispatch_result is None:
        return ledger_row

    dispatch_status = dispatch_result["status"]
    ledger_row["dispatch_status"] = dispatch_status
    ledger_row["scheduled_event_ids"] = list(dispatch_result["scheduled_event_ids"])
    if dispatch_status == "accepted":
        ledger_row["status"] = models.ACTION_ATTEMPT_STATUS_SCHEDULED
    elif dispatch_status == "rejected":
        ledger_row["status"] = models.ACTION_ATTEMPT_STATUS_HELD
    return ledger_row


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


def _read_json(path: str | Path) -> dict[str, Any]:
    """Read one JSON artifact written by the self-cognition runner."""

    content = Path(path).read_text(encoding="utf-8")
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("self-cognition artifact must contain an object")
    return payload


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
