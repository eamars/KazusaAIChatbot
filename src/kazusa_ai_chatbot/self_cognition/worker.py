"""Process-local idle worker for self-cognition cycles."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
from kazusa_ai_chatbot.calendar_scheduler import repository as calendar_repository
from kazusa_ai_chatbot.config import (
    CALENDAR_SCHEDULER_LEASE_SECONDS,
    CALENDAR_SCHEDULER_MAX_ATTEMPTS,
    SELF_COGNITION_MAX_CASES_PER_TICK,
    SELF_COGNITION_WORKER_INTERVAL_SECONDS,
)
from kazusa_ai_chatbot import db, event_logging
from kazusa_ai_chatbot.dispatcher.adapter_iface import AdapterRegistry
from kazusa_ai_chatbot.nodes.dialog_agent import StateContractError
from kazusa_ai_chatbot.self_cognition.delivery import (
    SelfCognitionDeliveryResult,
    deliver_selected_speak,
)
from kazusa_ai_chatbot.self_cognition import models, runner
from kazusa_ai_chatbot.self_cognition import sources as source_collectors
from kazusa_ai_chatbot.time_boundary import storage_utc_now

logger = logging.getLogger(__name__)

_WORKER_STOP_TIMEOUT_SECONDS = 5.0
_ATTEMPT_HISTORY_LIMIT = 1000
_CALENDAR_LEASE_OWNER = "self_cognition_worker"
_CALENDAR_TRIGGER_KIND_BY_SELF_COGNITION_TRIGGER = {
    models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK: (
        calendar_models.TRIGGER_COMMITMENT_DUE_COGNITION
    ),
    models.TRIGGER_SCHEDULED_FUTURE_COGNITION: (
        calendar_models.TRIGGER_FUTURE_COGNITION
    ),
}


@dataclass
class SelfCognitionWorkerResult:
    """Outcome counters for one self-cognition worker tick."""

    processed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    deferred: bool = False
    defer_reason: str = ""


@dataclass
class SelfCognitionWorkerHandle:
    """Process-local worker task and stop signal owned by service lifespan."""

    task: asyncio.Task
    stop_event: asyncio.Event


def start_self_cognition_worker(
    *,
    is_primary_interaction_busy: Callable[[], bool],
    character_profile_provider: Callable[[], dict[str, Any]],
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None = None,
) -> SelfCognitionWorkerHandle:
    """Start the process-local self-cognition worker loop.

    Args:
        is_primary_interaction_busy: Service load probe.
        character_profile_provider: Callable returning current character state.
        adapter_registry_provider: Callable returning the live adapter registry.

    Returns:
        Worker handle used for shutdown.
    """

    stop_event = asyncio.Event()
    task = asyncio.create_task(
        _self_cognition_worker_loop(
            stop_event=stop_event,
            is_primary_interaction_busy=is_primary_interaction_busy,
            character_profile_provider=character_profile_provider,
            adapter_registry_provider=adapter_registry_provider,
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
    now: datetime,
    is_primary_interaction_busy: Callable[[], bool],
    character_profile: dict[str, Any] | None = None,
    collect_cases_func: Callable[..., Any] | None = None,
    run_case_func: Callable[..., Any] | None = None,
    read_attempts_func: Callable[..., Any] | None = None,
    record_attempt_func: Callable[..., Any] | None = None,
    complete_calendar_run_func: Callable[..., Any] | None = None,
    claim_calendar_run_func: Callable[..., Any] | None = None,
    skip_calendar_run_func: Callable[..., Any] | None = None,
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None = None,
    max_cases: int = SELF_COGNITION_MAX_CASES_PER_TICK,
) -> SelfCognitionWorkerResult:
    """Run one bounded self-cognition worker tick.

    Args:
        now: Current worker tick time.
        is_primary_interaction_busy: Service load probe.
        character_profile: Current character state snapshot.
        collect_cases_func: Optional test seam for source collection.
        run_case_func: Optional test seam for self-cognition case projection.
        read_attempts_func: Optional test seam for prior attempt reads.
        record_attempt_func: Optional test seam for attempt persistence.
        complete_calendar_run_func: Optional test seam for source run
            completion.
        claim_calendar_run_func: Optional test seam for atomically claiming
            source calendar runs before processing.
        skip_calendar_run_func: Optional test seam for terminal source run
            skips.
        adapter_registry_provider: Optional live adapter registry provider.
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
        result = SelfCognitionWorkerResult(
            skipped_count=1,
            defer_reason="no eligible source cases",
        )
        await _record_worker_tick_event(result)
        return result

    result = SelfCognitionWorkerResult()
    active_read_attempts = read_attempts_func or (
        db.list_self_cognition_action_attempts
    )
    active_record_attempt = record_attempt_func or (
        db.upsert_self_cognition_action_attempt
    )
    active_complete_calendar_run = (
        complete_calendar_run_func
        or calendar_repository.mark_calendar_run_completed
    )
    active_claim_calendar_run = (
        claim_calendar_run_func or calendar_repository.claim_calendar_run
    )
    active_skip_calendar_run = (
        skip_calendar_run_func or calendar_repository.mark_calendar_run_skipped
    )
    adapter_registry = _adapter_registry_for_tick(adapter_registry_provider)

    for case in cases[:max_cases]:
        if is_primary_interaction_busy():
            result.deferred = True
            result.defer_reason = "primary interaction busy"
            break
        claimed = await _claim_source_calendar_run(
            case,
            now=now,
            claim_calendar_run_func=active_claim_calendar_run,
        )
        if not claimed:
            result.skipped_count += 1
            continue
        source_calendar_skip_reason = case.get("source_calendar_skip_reason")
        if (
            isinstance(source_calendar_skip_reason, str)
            and source_calendar_skip_reason
        ):
            result.skipped_count += 1
            await _skip_source_calendar_run(
                case,
                now=now,
                skip_calendar_run_func=active_skip_calendar_run,
                reason=source_calendar_skip_reason,
            )
            continue
        if _target_binding_failed(case):
            result.skipped_count += 1
            await _record_target_binding_failed_event(case)
            await _skip_source_calendar_run(
                case,
                now=now,
                skip_calendar_run_func=active_skip_calendar_run,
                reason="target_binding_failed",
            )
            continue
        prior_attempts = await _call_maybe_async(
            active_read_attempts,
            limit=_ATTEMPT_HISTORY_LIMIT,
        )
        case_for_run = _case_with_prior_attempts(case, prior_attempts)
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
            case=case_for_run,
            artifact_payloads=artifact_payloads,
            now=now,
            record_attempt_func=active_record_attempt,
            adapter_registry=adapter_registry,
        )
        await _record_self_cognition_event_from_artifacts(
            case=case_for_run,
            artifact_payloads=artifact_payloads,
            dispatch_status=dispatch_status,
        )
        await _complete_source_calendar_run(
            case_for_run,
            now=now,
            complete_calendar_run_func=active_complete_calendar_run,
        )

    await _record_worker_tick_event(result)
    return result


async def _self_cognition_worker_loop(
    *,
    stop_event: asyncio.Event,
    is_primary_interaction_busy: Callable[[], bool],
    character_profile_provider: Callable[[], dict[str, Any]],
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None,
) -> None:
    """Run self-cognition scheduling ticks until stopped."""

    while not stop_event.is_set():
        try:
            character_profile = character_profile_provider()
            await run_self_cognition_worker_tick(
                now=storage_utc_now(),
                is_primary_interaction_busy=is_primary_interaction_busy,
                character_profile=character_profile,
                adapter_registry_provider=adapter_registry_provider,
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
    now: datetime,
    record_attempt_func: Callable[..., Any],
    adapter_registry: AdapterRegistry | None,
) -> str:
    """Record or deliver selected action outputs for one case."""

    action_attempt = artifact_payloads.get(models.ARTIFACT_ACTION_ATTEMPT)
    if not isinstance(action_attempt, dict):
        return_value = "not_requested"
        return return_value

    attempt_status = str(action_attempt.get("status") or "")
    if attempt_status == models.ACTION_ATTEMPT_STATUS_DUPLICATE:
        attempt_state = _attempt_state(action_attempt, now=now)
        await _call_maybe_async(record_attempt_func, attempt_state)
        return_value = "duplicate_suppressed"
        return return_value
    if attempt_status == models.ACTION_ATTEMPT_STATUS_HELD:
        attempt_state = _attempt_state(action_attempt, now=now)
        await _call_maybe_async(record_attempt_func, attempt_state)
        return_value = "held"
        return return_value

    action_candidate = artifact_payloads.get(models.ARTIFACT_ACTION_CANDIDATE)
    delivery_target = case.get("delivery_target")
    if (
        attempt_status == models.ACTION_ATTEMPT_STATUS_CANDIDATE
        and not isinstance(action_candidate, dict)
        and isinstance(delivery_target, dict)
    ):
        delivery_result: SelfCognitionDeliveryResult = {
            "status": "delivery_failed",
            "conversation_message_id": None,
            "delivery_tracking_id": None,
            "adapter_message_id": None,
            "failure_reason": "empty_text",
        }
        attempt_state = _attempt_state(
            _action_attempt_with_delivery_result(
                action_attempt,
                delivery_result,
            ),
            now=now,
        )
        await _call_maybe_async(record_attempt_func, attempt_state)
        return_value = delivery_result["status"]
        return return_value
    if (
        attempt_status != models.ACTION_ATTEMPT_STATUS_CANDIDATE
        or not isinstance(action_candidate, dict)
        or not isinstance(delivery_target, dict)
    ):
        attempt_state = _attempt_state(action_attempt, now=now)
        await _call_maybe_async(record_attempt_func, attempt_state)
        return_value = "not_requested"
        return return_value

    delivery_result = await deliver_selected_speak(
        text=str(action_candidate.get("text") or ""),
        delivery_target=delivery_target,
        character_profile=_character_profile(case),
        adapter_registry=adapter_registry,
        now=now,
        reply_to_msg_id=_optional_text(action_candidate.get("reply_to_msg_id")),
        delivery_mentions=_delivery_mentions(action_candidate),
    )
    attempt_state = _attempt_state(
        _action_attempt_with_delivery_result(
            action_attempt,
            delivery_result,
        ),
        now=now,
    )
    await _call_maybe_async(record_attempt_func, attempt_state)
    return_value = delivery_result["status"]
    return return_value


def _action_attempt_with_delivery_result(
    action_attempt: dict[str, Any],
    delivery_result: SelfCognitionDeliveryResult,
) -> dict[str, Any]:
    """Apply terminal delivery metadata to an action-attempt row."""

    attempt_state = dict(action_attempt)
    result_status = delivery_result["status"]
    if result_status == "delivery_failed":
        attempt_state["status"] = models.ACTION_ATTEMPT_STATUS_DELIVERY_FAILED
    else:
        attempt_state["status"] = result_status
    attempt_state["dispatch_status"] = result_status
    if delivery_result["conversation_message_id"]:
        attempt_state["conversation_message_id"] = (
            delivery_result["conversation_message_id"]
        )
    if delivery_result["delivery_tracking_id"]:
        attempt_state["delivery_tracking_id"] = (
            delivery_result["delivery_tracking_id"]
        )
    if delivery_result["adapter_message_id"]:
        attempt_state["adapter_message_id"] = delivery_result[
            "adapter_message_id"
        ]
    if delivery_result["failure_reason"]:
        attempt_state["failure_reason"] = delivery_result["failure_reason"]
    return attempt_state


def _adapter_registry_for_tick(
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None,
) -> AdapterRegistry | None:
    """Resolve the live adapter registry once for a worker tick."""

    if adapter_registry_provider is None:
        return_value = None
        return return_value
    return_value = adapter_registry_provider()
    return return_value


def _target_binding_failed(
    case: models.SelfCognitionCase,
) -> bool:
    """Return whether a worker case must stop before cognition."""

    if case.get("target_binding_status") == "failed":
        return_value = True
        return return_value
    if not isinstance(case.get("delivery_target"), dict):
        return_value = True
        return return_value
    return_value = False
    return return_value


async def _record_target_binding_failed_event(
    case: models.SelfCognitionCase,
) -> None:
    """Record a skipped production case whose target could not be bound."""

    failure = case.get("target_binding_failure")
    if not isinstance(failure, dict):
        failure = _missing_delivery_target_failure(case)
    await event_logging.record_self_cognition_event(
        component="self_cognition.worker",
        case_id=str(case.get("case_id") or ""),
        trigger_kind=str(case.get("trigger_kind") or ""),
        selected_route="not_started",
        output_mode="none",
        budget={
            "rag_calls": 0,
            "cognition_calls": 0,
            "dialog_calls": 0,
            "topic_limit": 0,
        },
        dispatch_status="target_binding_failed",
        status="target_binding_failed",
        trigger_id="",
        run_id="",
        attempt_id="",
        consolidation_outcome=None,
        target_binding_failure=failure,
    )


def _missing_delivery_target_failure(
    case: models.SelfCognitionCase,
) -> models.SelfCognitionTargetBindingFailure:
    """Build audit metadata for a worker-owned missing target failure."""

    target_scope = case.get("target_scope")
    if not isinstance(target_scope, dict):
        target_scope = {}
    source_refs = case.get("source_refs")
    if isinstance(source_refs, list) and source_refs:
        raw_source_ref = source_refs[0]
    else:
        raw_source_ref = {}
    if not isinstance(raw_source_ref, dict):
        raw_source_ref = {}

    source_id = _optional_text(raw_source_ref.get("source_id"))
    case_id = _optional_text(case.get("case_id"))
    source_ref = source_id or case_id or ""
    target_global_user_id = _optional_text(target_scope.get("user_id"))
    target_platform_user_id = _optional_text(
        target_scope.get("platform_user_id")
    )
    failure: models.SelfCognitionTargetBindingFailure = {
        "status": "target_binding_failed",
        "reason": "missing_delivery_target",
        "platform": _optional_text(target_scope.get("platform")) or "",
        "source_ref": source_ref,
        "source_platform_channel_id": (
            _optional_text(target_scope.get("platform_channel_id")) or ""
        ),
        "source_channel_type": (
            _optional_text(target_scope.get("channel_type")) or ""
        ),
        "target_global_user_id": target_global_user_id,
        "target_platform_user_id": target_platform_user_id,
    }
    return failure


def _delivery_mentions(action_candidate: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Read optional delivery mention metadata from an action candidate."""

    value = action_candidate.get("delivery_mentions")
    if not isinstance(value, list):
        return_value = None
        return return_value
    mentions = [
        dict(item)
        for item in value
        if isinstance(item, dict)
    ]
    return mentions


def _character_profile(case: models.SelfCognitionCase) -> dict[str, Any]:
    """Return the case character profile used by delivery metadata."""

    value = case.get("character_profile")
    if isinstance(value, dict):
        profile = value
    else:
        profile = {}
    return profile


def _optional_text(value: object) -> str | None:
    """Return stripped optional text from a local artifact field."""

    if not isinstance(value, str):
        return_value = None
        return return_value
    clean_value = value.strip()
    if clean_value:
        return_value = clean_value
    else:
        return_value = None
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


async def _claim_source_calendar_run(
    case: models.SelfCognitionCase,
    *,
    now: datetime,
    claim_calendar_run_func: Callable[..., Any],
) -> bool:
    """Atomically claim a source calendar cognition run before processing."""

    self_cognition_trigger_kind = case.get("trigger_kind")
    calendar_trigger_kind = (
        _CALENDAR_TRIGGER_KIND_BY_SELF_COGNITION_TRIGGER.get(
            self_cognition_trigger_kind,
        )
    )
    if calendar_trigger_kind is None:
        return_value = True
        return return_value

    run_id = case.get("source_calendar_run_id")
    if not isinstance(run_id, str) or not run_id:
        return_value = (
            self_cognition_trigger_kind
            != models.TRIGGER_SCHEDULED_FUTURE_COGNITION
        )
        return return_value

    claimed = await _call_maybe_async(
        claim_calendar_run_func,
        run_id,
        trigger_kind=calendar_trigger_kind,
        current_timestamp_utc=now.isoformat(),
        lease_owner=_CALENDAR_LEASE_OWNER,
        lease_duration_seconds=CALENDAR_SCHEDULER_LEASE_SECONDS,
        max_attempts=CALENDAR_SCHEDULER_MAX_ATTEMPTS,
    )
    return_value = bool(claimed)
    return return_value


async def _complete_source_calendar_run(
    case: models.SelfCognitionCase,
    *,
    now: datetime,
    complete_calendar_run_func: Callable[..., Any],
) -> None:
    """Mark a source calendar cognition run consumed after processing."""

    calendar_trigger_kind = (
        _CALENDAR_TRIGGER_KIND_BY_SELF_COGNITION_TRIGGER.get(
            case.get("trigger_kind"),
        )
    )
    if calendar_trigger_kind is None:
        return

    run_id = case.get("source_calendar_run_id")
    if not isinstance(run_id, str) or not run_id:
        return

    await _call_maybe_async(
        complete_calendar_run_func,
        run_id,
        lease_owner=_CALENDAR_LEASE_OWNER,
        storage_timestamp_utc=now.isoformat(),
        result={
            "status": "self_cognition_processed",
            "case_name": str(case.get("case_name") or ""),
        },
    )


async def _skip_source_calendar_run(
    case: models.SelfCognitionCase,
    *,
    now: datetime,
    skip_calendar_run_func: Callable[..., Any],
    reason: str,
) -> None:
    """Mark an unprocessable source calendar run skipped."""

    calendar_trigger_kind = (
        _CALENDAR_TRIGGER_KIND_BY_SELF_COGNITION_TRIGGER.get(
            case.get("trigger_kind"),
        )
    )
    if calendar_trigger_kind is None:
        return

    run_id = case.get("source_calendar_run_id")
    if not isinstance(run_id, str) or not run_id:
        return

    await _call_maybe_async(
        skip_calendar_run_func,
        run_id,
        lease_owner=_CALENDAR_LEASE_OWNER,
        storage_timestamp_utc=now.isoformat(),
        reason=reason,
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
    now: datetime,
) -> dict[str, Any]:
    """Build the persisted state row for one action attempt."""

    attempt_state = dict(action_attempt)
    attempt_state["recorded_at"] = now.isoformat()
    return attempt_state


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
