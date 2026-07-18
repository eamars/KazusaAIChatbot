"""Process-local idle worker for self-cognition cycles."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from kazusa_ai_chatbot.action_spec.results import (
    has_consolidatable_output,
)
from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
from kazusa_ai_chatbot.calendar_scheduler import repository as calendar_repository
from kazusa_ai_chatbot.config import (
    CALENDAR_SCHEDULER_LEASE_SECONDS,
    CALENDAR_SCHEDULER_MAX_ATTEMPTS,
    SELF_COGNITION_MAX_CASES_PER_TICK,
    SELF_COGNITION_WORKER_INTERVAL_SECONDS,
)
from kazusa_ai_chatbot import db, event_logging
from kazusa_ai_chatbot.brain_service.post_turn import (
    build_post_turn_lifecycle_record,
    settle_runtime_episode_trace,
)
from kazusa_ai_chatbot.internal_monologue_residue import (
    record_completed_episode_residue,
)
from kazusa_ai_chatbot.dispatcher.adapter_iface import AdapterRegistry
from kazusa_ai_chatbot.nodes.dialog_agent import StateContractError
from kazusa_ai_chatbot.runtime_coordination import (
    PipelineCancelled,
    PipelineCoordinator,
    PipelineRunHandle,
    PipelineScope,
)
from kazusa_ai_chatbot.self_cognition.delivery import (
    SelfCognitionDeliveryResult,
    deliver_selected_speak,
)
from kazusa_ai_chatbot.self_cognition import models, runner, tracking
from kazusa_ai_chatbot.self_cognition import sources as source_collectors
from kazusa_ai_chatbot.time_boundary import (
    storage_utc_now,
    storage_utc_now_iso,
)

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
    latest_cognition_graph_publisher: Callable[[dict[str, Any]], Any] | None = None,
    should_pause_for_affect_settling: Callable[..., Any] | None = None,
    pipeline_coordinator: PipelineCoordinator | None = None,
) -> SelfCognitionWorkerHandle:
    """Start the process-local self-cognition worker loop.

    Args:
        is_primary_interaction_busy: Service load probe.
        character_profile_provider: Callable returning current character state.
        adapter_registry_provider: Callable returning the live adapter registry.
        latest_cognition_graph_publisher: Optional telemetry publisher for
            the most recent self-cognition graph snapshot.
        should_pause_for_affect_settling: Optional service-level probe used to
            pause source collection while daily affect settling is pending.
        pipeline_coordinator: Optional runtime coordinator used to admit and
            cancel scoped background self-cognition runs.

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
            latest_cognition_graph_publisher=latest_cognition_graph_publisher,
            should_pause_for_affect_settling=should_pause_for_affect_settling,
            pipeline_coordinator=pipeline_coordinator,
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
    fail_calendar_run_func: Callable[..., Any] | None = None,
    skip_calendar_run_func: Callable[..., Any] | None = None,
    defer_calendar_run_func: Callable[..., Any] | None = None,
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None = None,
    latest_cognition_graph_publisher: Callable[[dict[str, Any]], Any] | None = None,
    should_pause_for_affect_settling: Callable[..., Any] | None = None,
    pipeline_coordinator: PipelineCoordinator | None = None,
    pipeline_run_handle: PipelineRunHandle | None = None,
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
        fail_calendar_run_func: Optional test seam for terminal source run
            failures.
        skip_calendar_run_func: Optional test seam for terminal source run
            skips.
        defer_calendar_run_func: Optional test seam for source run deferral.
        adapter_registry_provider: Optional live adapter registry provider.
        latest_cognition_graph_publisher: Optional telemetry publisher for
            successful self-cognition artifacts.
        should_pause_for_affect_settling: Optional service-level pause probe
            called before source collection.
        pipeline_coordinator: Optional runtime coordinator used to start
            scoped background runs for standalone worker cases.
        pipeline_run_handle: Optional caller-owned background run handle.
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

    if should_pause_for_affect_settling is not None:
        should_pause = await _call_maybe_async(
            should_pause_for_affect_settling,
            now=now,
        )
        if should_pause:
            result = SelfCognitionWorkerResult(
                deferred=True,
                defer_reason="daily affect settling pending",
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
    if collect_cases_func is None:
        try:
            internal_latch_claim = await db.claim_due_internal_action_latch(
                worker_id=_CALENDAR_LEASE_OWNER,
                now=now.isoformat(),
            )
        except Exception as exc:
            logger.warning(
                "Internal-action latch claim failed; continuing source tick: %s",
                exc,
            )
            internal_latch_claim = None
        if internal_latch_claim is not None:
            cases = [
                _case_from_internal_action_latch(
                    internal_latch_claim,
                    character_profile=active_profile,
                    now=now,
                ),
                *cases,
            ]
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
    active_fail_calendar_run = (
        fail_calendar_run_func or calendar_repository.mark_calendar_run_failed
    )
    active_skip_calendar_run = (
        skip_calendar_run_func or calendar_repository.mark_calendar_run_skipped
    )
    active_defer_calendar_run = (
        defer_calendar_run_func
        or calendar_repository.mark_calendar_run_deferred
    )
    adapter_registry = _adapter_registry_for_tick(adapter_registry_provider)

    for case in cases[:max_cases]:
        if is_primary_interaction_busy():
            result.deferred = True
            result.defer_reason = "primary interaction busy"
            break
        active_pipeline_handle = pipeline_run_handle
        owns_pipeline_handle = False
        if active_pipeline_handle is None and pipeline_coordinator is not None:
            pipeline_scope = _pipeline_scope_from_case(case)
            if pipeline_scope is not None:
                admission = await pipeline_coordinator.start_run(
                    scope=pipeline_scope,
                    owner="self_cognition.worker",
                    precedence="background",
                    run_kind=str(case.get("trigger_kind") or "self_cognition"),
                )
                if not admission.admitted:
                    result.deferred = True
                    result.defer_reason = admission.defer_reason or ""
                    break
                active_pipeline_handle = admission.handle
                owns_pipeline_handle = active_pipeline_handle is not None

        enter_run_phase = False
        try:
            claimed = await _claim_source_calendar_run(
                case,
                now=now,
                claim_calendar_run_func=active_claim_calendar_run,
            )
            if not claimed:
                result.skipped_count += 1
                continue
            source_calendar_skip_reason = case.get(
                "source_calendar_skip_reason"
            )
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
                await _fail_internal_action_latch_for_case(
                    case,
                    now=now,
                    error_code="target_binding_failed",
                )
                await _skip_source_calendar_run(
                    case,
                    now=now,
                    skip_calendar_run_func=active_skip_calendar_run,
                    reason="target_binding_failed",
                )
                continue
            enter_run_phase = True
        except Exception as exc:
            logger.exception(
                f"Self-cognition case pre-run failed: {exc}"
            )
            result.failed_count += 1
            await event_logging.record_runtime_error_event(
                component="self_cognition.worker",
                error_class=type(exc).__name__,
                error_preview=str(exc),
                stack_fingerprint="self_cognition_case_pre_run",
                top_frame_module=__name__,
                recovered=True,
            )
            continue
        finally:
            await _release_owned_pipeline_handle(
                active_pipeline_handle,
                owns_pipeline_handle=(
                    owns_pipeline_handle and not enter_run_phase
                ),
            )
        case_for_run = case
        try:
            if active_pipeline_handle is not None:
                active_pipeline_handle.raise_if_cancelled(
                    "before_self_cognition_case",
                )
            prior_attempts = await _call_maybe_async(
                active_read_attempts,
                limit=_ATTEMPT_HISTORY_LIMIT,
            )
            case_for_run = _case_with_prior_attempts(case, prior_attempts)
            if run_case_func is None:
                artifact_payloads = (
                    await runner.build_self_cognition_case_artifacts_async(
                        case_for_run,
                        apply_consolidation=False,
                        execute_private_actions=True,
                        pipeline_run_handle=active_pipeline_handle,
                    )
                )
            else:
                artifact_payloads = await _call_run_case_func(
                    run_case_func,
                    case_for_run,
                    pipeline_run_handle=active_pipeline_handle,
                )
            _validate_worker_v2_cognition_result(
                artifact_payloads,
                required=run_case_func is None,
            )
            if active_pipeline_handle is not None:
                active_pipeline_handle.raise_if_cancelled(
                    "before_action_outputs",
                )
            dispatch_status = await _handle_case_action_outputs(
                case=case_for_run,
                artifact_payloads=artifact_payloads,
                now=now,
                record_attempt_func=active_record_attempt,
                adapter_registry=adapter_registry,
                pipeline_run_handle=active_pipeline_handle,
            )
            await _settle_and_finish_self_cognition_episode(
                artifact_payloads=artifact_payloads,
                dispatch_status=dispatch_status,
                settled_at=storage_utc_now_iso(),
                pipeline_run_handle=active_pipeline_handle,
            )
            await _record_self_cognition_event_from_artifacts(
                case=case_for_run,
                artifact_payloads=artifact_payloads,
                dispatch_status=dispatch_status,
            )
            await _publish_latest_cognition_graph(
                artifact_payloads,
                publisher=latest_cognition_graph_publisher,
            )
            await _complete_source_calendar_run(
                case_for_run,
                now=now,
                complete_calendar_run_func=active_complete_calendar_run,
            )
            await _consume_internal_action_latch_for_case(
                case_for_run,
                now=now,
            )
        except PipelineCancelled as exc:
            result.deferred = True
            result.defer_reason = exc.cancellation.reason
            await _defer_source_calendar_run(
                case_for_run,
                now=now,
                defer_calendar_run_func=active_defer_calendar_run,
                reason=exc.cancellation.reason,
            )
            await _release_internal_action_latch_for_case(
                case_for_run,
                now=now,
                reason=exc.cancellation.reason,
            )
            break
        except StateContractError as exc:
            logger.exception(
                f"Self-cognition case state contract failed: {exc}"
            )
            result.failed_count += 1
            await _fail_source_calendar_run(
                case_for_run,
                now=now,
                fail_calendar_run_func=active_fail_calendar_run,
                error=str(exc),
            )
            await _fail_internal_action_latch_for_case(
                case_for_run,
                now=now,
                error_code="state_contract",
            )
            await event_logging.record_runtime_error_event(
                component="self_cognition.worker",
                error_class="StateContractError",
                error_preview=str(exc),
                stack_fingerprint="self_cognition_case_state_contract",
                top_frame_module=__name__,
                recovered=True,
            )
            continue
        except Exception as exc:
            logger.exception(
                f"Self-cognition case processing failed: {exc}"
            )
            result.failed_count += 1
            await _fail_source_calendar_run(
                case_for_run,
                now=now,
                fail_calendar_run_func=active_fail_calendar_run,
                error=str(exc),
            )
            await _release_internal_action_latch_for_case(
                case_for_run,
                now=now,
                reason=type(exc).__name__,
            )
            await event_logging.record_runtime_error_event(
                component="self_cognition.worker",
                error_class=type(exc).__name__,
                error_preview=str(exc),
                stack_fingerprint="self_cognition_case_processing",
                top_frame_module=__name__,
                recovered=True,
            )
            continue
        finally:
            await _release_owned_pipeline_handle(
                active_pipeline_handle,
                owns_pipeline_handle=owns_pipeline_handle,
            )
        result.processed_count += 1

    await _record_worker_tick_event(result)
    return result


def _validate_worker_v2_cognition_result(
    artifact_payloads: dict[str, Any],
    *,
    required: bool,
) -> None:
    """Enforce character-scoped commit completion before worker delivery."""

    output = artifact_payloads.get(models.ARTIFACT_COGNITION_OUTPUT)
    if not isinstance(output, dict):
        if required and models.ARTIFACT_COGNITION_INPUT in artifact_payloads:
            raise StateContractError("self-cognition V2 output is missing")
        return
    core_output = output.get("cognition_core_output")
    if not isinstance(core_output, dict):
        if required:
            raise StateContractError("self-cognition V2 core output is missing")
        return
    state_update = core_output.get("state_update")
    if not isinstance(state_update, dict):
        raise StateContractError("self-cognition V2 state update is missing")
    if state_update.get("state_scope") != "character":
        raise StateContractError(
            "self-cognition V2 state update must use character scope"
        )
    if output.get("cognition_state_committed") is not True:
        raise StateContractError(
            "self-cognition V2 state was not committed before delivery"
        )


async def _settle_and_finish_self_cognition_episode(
    *,
    artifact_payloads: dict[str, Any],
    dispatch_status: str,
    settled_at: str,
    pipeline_run_handle: PipelineRunHandle | None,
) -> dict[str, Any] | None:
    """Settle one self-cognition episode before its persistence consumers run."""

    episode = artifact_payloads.get(models.RUNTIME_COGNITIVE_EPISODE)
    if not isinstance(episode, dict):
        return None
    if pipeline_run_handle is not None:
        pipeline_run_handle.raise_if_cancelled("before_episode_settlement")

    cognition_output = artifact_payloads.get(
        models.ARTIFACT_COGNITION_OUTPUT,
    )
    if not isinstance(cognition_output, dict):
        cognition_output = {}
    action_specs = _dict_rows(cognition_output.get("action_specs"))
    action_results = _dict_rows(cognition_output.get("action_results"))
    surface_outputs = _dict_rows(cognition_output.get("surface_outputs"))
    action_attempt = artifact_payloads.get(models.ARTIFACT_ACTION_ATTEMPT)
    if isinstance(action_attempt, dict):
        action_results = _append_delivery_action_result(
            action_specs=action_specs,
            action_results=action_results,
            action_attempt=action_attempt,
            dispatch_status=dispatch_status,
            settled_at=settled_at,
        )

    visible_surface = _has_visible_surface(surface_outputs)
    delivery_result = artifact_payloads.get(models.ARTIFACT_DISPATCH_RESULT)
    delivery_correlation = _delivery_correlation(
        visible_surface=visible_surface,
        delivery_result=delivery_result,
    )
    terminal_status = _terminal_status(
        visible_surface=visible_surface,
        action_results=action_results,
        dispatch_status=dispatch_status,
    )
    trace = await settle_runtime_episode_trace(
        episode=episode,
        graph_result={
            "cognition_output": cognition_output,
            "action_specs": action_specs,
            "action_results": action_results,
            "surface_outputs": surface_outputs,
            "terminal_status": terminal_status,
            "delivery_correlation": delivery_correlation,
        },
        response_dialog=[],
        delivery_tracking_id=str(
            delivery_correlation.get("tracking_id") or ""
        ),
        settled_at=settled_at,
    )
    artifact_payloads[models.RUNTIME_EPISODE_TRACE] = trace

    consolidation_state = artifact_payloads.get(
        models.RUNTIME_CONSOLIDATION_STATE,
    )
    if not isinstance(consolidation_state, dict):
        consolidation_state = {}
    consolidation_state = dict(consolidation_state)
    consolidation_state["episode_trace"] = trace
    consolidation_state["action_specs"] = action_specs
    consolidation_state["action_results"] = action_results
    consolidation_state["surface_outputs"] = surface_outputs
    artifact_payloads[models.RUNTIME_CONSOLIDATION_STATE] = consolidation_state

    if has_consolidatable_output(trace):
        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled("before_consolidation")
        consolidation_result = (
            await runner.run_self_cognition_consolidation_async(
                consolidation_state,
            )
        )
        artifact_payloads[models.ARTIFACT_CONSOLIDATION_OUTCOME] = (
            tracking.build_consolidation_outcome_record(
                consolidation_state,
                consolidation_result,
            )
        )
        await record_completed_episode_residue(
            completed_state=consolidation_state,
            current_timestamp_utc=str(episode["created_at"]),
        )

    delivery_tracking_id = str(
        delivery_correlation.get("tracking_id") or ""
    )
    lifecycle_record = build_post_turn_lifecycle_record(
        source_episode_id=str(episode["episode_id"]),
        delivery_tracking_id=delivery_tracking_id,
        action_specs=action_specs,
        action_results=action_results,
        error_codes=_lifecycle_error_codes(delivery_result),
        created_at=str(episode["created_at"]),
    )
    await db.upsert_post_turn_lifecycle_record(lifecycle_record)
    return trace


def _dict_rows(value: object) -> list[dict[str, Any]]:
    """Copy dictionary rows from an optional runtime component list."""

    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, dict)]


def _has_visible_surface(surface_outputs: list[dict[str, Any]]) -> bool:
    """Return whether a component list contains a deliver-now visible surface."""

    return any(
        output.get("visibility") == "user_visible"
        and output.get("delivery_intent") == "deliver_now"
        for output in surface_outputs
    )


def _delivery_correlation(
    *,
    visible_surface: bool,
    delivery_result: object,
) -> dict[str, Any]:
    """Build immutable delivery correlation from the dispatcher result."""

    if not visible_surface:
        return {
            "schema_version": "delivery_correlation.v1",
            "delivery_intent": "do_not_deliver",
            "tracking_id": "",
            "receipt_status": "not_applicable",
            "receipt_ref": "",
        }
    result = delivery_result if isinstance(delivery_result, dict) else {}
    tracking_id = result.get("delivery_tracking_id")
    tracking_id = tracking_id if isinstance(tracking_id, str) else ""
    status = result.get("status")
    if status == "sent" and tracking_id:
        receipt_status = "pending"
    elif status == "delivery_failed":
        receipt_status = "failed"
    else:
        receipt_status = "unknown"
    failure_reason = result.get("failure_reason")
    receipt_ref = failure_reason if isinstance(failure_reason, str) else ""
    return {
        "schema_version": "delivery_correlation.v1",
        "delivery_intent": "deliver_now",
        "tracking_id": tracking_id,
        "receipt_status": receipt_status,
        "receipt_ref": receipt_ref,
    }


def _terminal_status(
    *,
    visible_surface: bool,
    action_results: list[dict[str, Any]],
    dispatch_status: str,
) -> str:
    """Choose a trace terminal status from settled deterministic outcomes."""

    if visible_surface:
        if dispatch_status in {"delivery_failed", "not_requested"}:
            return "failed"
        return "completed_visible"
    statuses = {
        str(result.get("status") or "")
        for result in action_results
    }
    if statuses and statuses <= {"failed", "rejected", "cancelled"}:
        return "failed"
    if "scheduled" in statuses or "pending" in statuses:
        return "scheduled"
    if action_results:
        return "completed_action"
    return "completed_private"


def _append_delivery_action_result(
    *,
    action_specs: list[dict[str, Any]],
    action_results: list[dict[str, Any]],
    action_attempt: dict[str, Any],
    dispatch_status: str,
    settled_at: str,
) -> list[dict[str, Any]]:
    """Append the worker-owned visible speak outcome when it is absent."""

    attempt_id = action_attempt.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        return action_results
    if any(result.get("action_attempt_id") == attempt_id for result in action_results):
        return action_results
    speak_spec = next(
        (
            spec
            for spec in action_specs
            if spec.get("kind") == "speak"
        ),
        None,
    )
    if speak_spec is None:
        return action_results
    continuation = speak_spec.get("continuation")
    if not isinstance(continuation, dict):
        continuation = {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        }
    status = "executed" if dispatch_status == "sent" else "failed"
    result = {
        "schema_version": "action_result.v1",
        "action_attempt_id": attempt_id,
        "action_kind": "speak",
        "handler_owner": "self_cognition.worker",
        "status": status,
        "visibility": str(speak_spec.get("visibility") or "user_visible"),
        "result_summary": (
            "Self-cognition speech delivered."
            if status == "executed"
            else "Self-cognition speech delivery failed."
        ),
        "result_refs": [],
        "continuation": continuation,
        "completed_at": settled_at if status == "executed" else None,
    }
    return [*action_results, result]


def _lifecycle_error_codes(delivery_result: object) -> list[str]:
    """Project dispatcher failure metadata into lifecycle error codes."""

    if not isinstance(delivery_result, dict):
        return []
    failure_reason = delivery_result.get("failure_reason")
    if isinstance(failure_reason, str) and failure_reason:
        return [failure_reason]
    return []


async def _self_cognition_worker_loop(
    *,
    stop_event: asyncio.Event,
    is_primary_interaction_busy: Callable[[], bool],
    character_profile_provider: Callable[[], dict[str, Any]],
    adapter_registry_provider: Callable[[], AdapterRegistry | None] | None,
    latest_cognition_graph_publisher: Callable[[dict[str, Any]], Any] | None,
    should_pause_for_affect_settling: Callable[..., Any] | None = None,
    pipeline_coordinator: PipelineCoordinator | None = None,
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
                latest_cognition_graph_publisher=(
                    latest_cognition_graph_publisher
                ),
                should_pause_for_affect_settling=(
                    should_pause_for_affect_settling
                ),
                pipeline_coordinator=pipeline_coordinator,
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


async def _release_owned_pipeline_handle(
    pipeline_run_handle: PipelineRunHandle | None,
    *,
    owns_pipeline_handle: bool,
) -> None:
    """Release a coordinator handle only when this worker admitted it."""

    if not owns_pipeline_handle or pipeline_run_handle is None:
        return
    await pipeline_run_handle.__aexit__(None, None, None)


def _pipeline_scope_from_case(
    case: models.SelfCognitionCase,
) -> PipelineScope | None:
    """Build a cancellation scope from a self-cognition target scope."""

    target_scope = case.get("target_scope")
    if not isinstance(target_scope, dict):
        return_value = None
        return return_value

    platform = _optional_text(target_scope.get("platform"))
    channel_id = _optional_text(target_scope.get("platform_channel_id"))
    channel_type = _optional_text(target_scope.get("channel_type"))
    if platform is None or channel_id is None or channel_type is None:
        return_value = None
        return return_value

    scope = PipelineScope(
        platform=platform,
        platform_channel_id=channel_id,
        channel_type=channel_type,
    )
    return scope


async def _call_run_case_func(
    run_case_func: Callable[..., Any],
    case: models.SelfCognitionCase,
    *,
    pipeline_run_handle: PipelineRunHandle | None,
) -> dict[str, Any]:
    """Call a case-runner seam while preserving its declared call shape."""

    signature = inspect.signature(run_case_func)
    parameters = signature.parameters
    accepts_var_keyword = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_var_keyword:
        result = await _call_maybe_async(
            run_case_func,
            case=case,
            pipeline_run_handle=pipeline_run_handle,
        )
        return result

    if "pipeline_run_handle" in parameters:
        if "case" in parameters:
            result = await _call_maybe_async(
                run_case_func,
                case=case,
                pipeline_run_handle=pipeline_run_handle,
            )
            return result
        result = await _call_maybe_async(
            run_case_func,
            case,
            pipeline_run_handle=pipeline_run_handle,
        )
        return result

    if "case" in parameters:
        result = await _call_maybe_async(run_case_func, case=case)
        return result

    result = await _call_maybe_async(run_case_func, case)
    return result


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


def _case_from_internal_action_latch(
    claim: dict[str, Any],
    *,
    character_profile: dict[str, Any],
    now: datetime,
) -> models.SelfCognitionCase:
    """Build the worker case envelope for one claimed internal latch."""

    latch = claim["latch"]
    raw_scope = latch.get("target_scope")
    scope = raw_scope if isinstance(raw_scope, dict) else {}
    platform = str(scope.get("platform") or "")
    channel_id = str(scope.get("platform_channel_id") or "")
    channel_type = str(scope.get("channel_type") or "private")
    user_id = str(
        scope.get("current_global_user_id")
        or scope.get("user_id")
        or ""
    )
    platform_user_id = str(
        scope.get("current_platform_user_id")
        or scope.get("platform_user_id")
        or user_id
    )
    display_name = str(
        scope.get("current_display_name")
        or scope.get("display_name")
        or user_id
    )
    character_name = str(character_profile.get("name") or "Character")
    source_bot_id = str(scope.get("source_platform_bot_id") or "")
    delivery_target = {
        "schema_version": "self_cognition_delivery_target.v1",
        "platform": platform,
        "platform_channel_id": channel_id,
        "channel_type": "private" if channel_type == "private" else "group",
        "target_global_user_id": user_id or None,
        "target_platform_user_id": platform_user_id or None,
        "source_kind": (
            "target_private_channel"
            if channel_type == "private"
            else "self_cognition_source_channel"
        ),
        "source_ref": str(latch.get("source_episode_id") or ""),
        "source_platform_channel_id": channel_id,
        "source_channel_type": (
            "private" if channel_type == "private" else "group"
        ),
        "source_message_id": str(latch.get("source_episode_id") or ""),
        "source_global_user_id": user_id or None,
        "source_platform_bot_id": source_bot_id,
        "source_character_name": character_name,
        "guild_id": None,
        "bot_permission_role": "self_cognition",
        "fallback_reason": "",
    }
    evidence_refs = latch.get("evidence_refs")
    source_refs = [
        {
            "source_kind": "internal_thought",
            "source_id": str(latch.get("source_episode_id") or ""),
            "summary": str(latch.get("continuation_objective") or ""),
        }
    ]
    if isinstance(evidence_refs, list):
        source_refs.extend(
            {
                "source_kind": "internal_thought_evidence",
                "source_id": str(ref.get("evidence_id") or ""),
                "summary": str(ref.get("excerpt") or ""),
            }
            for ref in evidence_refs
            if isinstance(ref, dict)
        )
    case: models.SelfCognitionCase = {
        "case_name": models.CASE_PRIVATE_NO_ACTION,
        "case_id": f"internal-thought:{latch.get('latch_id', '')}",
        "idle_timestamp_utc": now.isoformat(),
        "trigger_kind": "internal_thought",
        "target_scope": {
            "platform": platform,
            "platform_channel_id": channel_id,
            "channel_type": channel_type,
            "user_id": user_id or None,
            "platform_user_id": platform_user_id or None,
            "display_name": display_name,
        },
        "source_refs": source_refs,
        "actionability": "private",
        "visible_context": [],
        "delivery_mention_users": [],
        "existing_attempts": [],
        "character_profile": character_profile,
        "user_profile": {},
        "platform_bot_id": source_bot_id,
        "channel_topic": "",
        "promoted_reflection_context": {},
        "budget": {
            "rag_calls": 0,
            "cognition_calls": 1,
            "dialog_calls": 1,
            "topic_limit": 0,
        },
        "source_calendar_run_id": "",
        "source_calendar_skip_reason": "",
        "cognition_source": {
            "source_kind": "internal_thought",
            "source_id": str(latch.get("latch_id") or ""),
        },
        "source_action_attempt_id": str(
            latch.get("source_action_attempt_id") or ""
        ),
        "delivery_target": delivery_target,
        "target_binding_status": (
            "bound" if platform and channel_id else "failed"
        ),
        "internal_action_latch": latch,
        "claim_token": str(claim.get("claim_token") or ""),
    }
    return case


def _internal_latch_claim_parts(
    case: models.SelfCognitionCase,
) -> tuple[dict[str, Any], str] | None:
    """Return latch and claim token when this case owns a latch claim."""

    latch = case.get("internal_action_latch")
    claim_token = case.get("claim_token")
    if not isinstance(latch, dict) or not isinstance(claim_token, str):
        return None
    if not claim_token:
        return None
    return latch, claim_token


async def _consume_internal_action_latch_for_case(
    case: models.SelfCognitionCase,
    *,
    now: datetime,
) -> None:
    """Consume a claimed internal latch after its episode settles."""

    parts = _internal_latch_claim_parts(case)
    if parts is None:
        return
    latch, claim_token = parts
    await db.consume_internal_action_latch(
        latch_id=str(latch["latch_id"]),
        claim_token=claim_token,
        consumed_episode_id=str(case.get("case_id") or ""),
        now=now.isoformat(),
    )


async def _release_internal_action_latch_for_case(
    case: models.SelfCognitionCase,
    *,
    now: datetime,
    reason: str,
) -> None:
    """Release a retryable technical latch failure."""

    parts = _internal_latch_claim_parts(case)
    if parts is None:
        return
    latch, claim_token = parts
    await db.release_internal_action_latch(
        latch_id=str(latch["latch_id"]),
        claim_token=claim_token,
        retry_at=(now + timedelta(seconds=60)).isoformat(),
        error_code=reason[:80] or "technical_failure",
        now=now.isoformat(),
    )


async def _fail_internal_action_latch_for_case(
    case: models.SelfCognitionCase,
    *,
    now: datetime,
    error_code: str,
) -> None:
    """Mark a claimed latch as a typed terminal failure."""

    parts = _internal_latch_claim_parts(case)
    if parts is None:
        return
    latch, claim_token = parts
    await db.fail_internal_action_latch(
        latch_id=str(latch["latch_id"]),
        claim_token=claim_token,
        error_code=error_code[:80],
        now=now.isoformat(),
    )


async def _handle_case_action_outputs(
    *,
    case: models.SelfCognitionCase,
    artifact_payloads: dict[str, Any],
    now: datetime,
    record_attempt_func: Callable[..., Any],
    adapter_registry: AdapterRegistry | None,
    pipeline_run_handle: PipelineRunHandle | None = None,
) -> str:
    """Record or deliver selected action outputs for one case."""

    action_attempt = artifact_payloads.get(models.ARTIFACT_ACTION_ATTEMPT)
    if not isinstance(action_attempt, dict):
        artifact_payloads[models.ARTIFACT_DISPATCH_RESULT] = {
            "status": "not_requested",
        }
        return_value = "not_requested"
        return return_value

    attempt_status = str(action_attempt.get("status") or "")
    if attempt_status == models.ACTION_ATTEMPT_STATUS_DUPLICATE:
        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled(
                "before_action_attempt_persistence",
            )
        attempt_state = _attempt_state(action_attempt, now=now)
        await _call_maybe_async(record_attempt_func, attempt_state)
        artifact_payloads[models.ARTIFACT_DISPATCH_RESULT] = {
            "status": "duplicate_suppressed",
        }
        return_value = "duplicate_suppressed"
        return return_value
    if attempt_status == models.ACTION_ATTEMPT_STATUS_HELD:
        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled(
                "before_action_attempt_persistence",
            )
        attempt_state = _attempt_state(action_attempt, now=now)
        await _call_maybe_async(record_attempt_func, attempt_state)
        artifact_payloads[models.ARTIFACT_DISPATCH_RESULT] = {
            "status": "held",
        }
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
        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled(
                "before_action_attempt_persistence",
            )
        attempt_state = _attempt_state(
            _action_attempt_with_delivery_result(
                action_attempt,
                delivery_result,
            ),
            now=now,
        )
        await _call_maybe_async(record_attempt_func, attempt_state)
        artifact_payloads[models.ARTIFACT_DISPATCH_RESULT] = dict(
            delivery_result
        )
        return_value = delivery_result["status"]
        return return_value
    if (
        attempt_status != models.ACTION_ATTEMPT_STATUS_CANDIDATE
        or not isinstance(action_candidate, dict)
        or not isinstance(delivery_target, dict)
    ):
        if pipeline_run_handle is not None:
            pipeline_run_handle.raise_if_cancelled(
                "before_action_attempt_persistence",
            )
        attempt_state = _attempt_state(action_attempt, now=now)
        await _call_maybe_async(record_attempt_func, attempt_state)
        artifact_payloads[models.ARTIFACT_DISPATCH_RESULT] = {
            "status": "not_requested",
        }
        return_value = "not_requested"
        return return_value

    if pipeline_run_handle is not None:
        pipeline_run_handle.raise_if_cancelled("before_dispatch")
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
    artifact_payloads[models.ARTIFACT_DISPATCH_RESULT] = dict(delivery_result)
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


async def _defer_source_calendar_run(
    case: models.SelfCognitionCase,
    *,
    now: datetime,
    defer_calendar_run_func: Callable[..., Any],
    reason: str,
) -> None:
    """Requeue a claimed source calendar run after cooperative cancellation."""

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
        defer_calendar_run_func,
        run_id,
        lease_owner=_CALENDAR_LEASE_OWNER,
        storage_timestamp_utc=now.isoformat(),
        reason=reason,
    )


async def _fail_source_calendar_run(
    case: models.SelfCognitionCase,
    *,
    now: datetime,
    fail_calendar_run_func: Callable[..., Any],
    error: str,
) -> None:
    """Mark a claimed source calendar run failed after case processing fails."""

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
        fail_calendar_run_func,
        run_id,
        lease_owner=_CALENDAR_LEASE_OWNER,
        storage_timestamp_utc=now.isoformat(),
        error=error,
        retryable=False,
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


async def _publish_latest_cognition_graph(
    artifact_payloads: dict[str, Any],
    *,
    publisher: Callable[[dict[str, Any]], Any] | None,
) -> None:
    """Publish self-cognition telemetry without changing run outcome."""

    if publisher is None:
        return
    try:
        await _call_maybe_async(publisher, artifact_payloads)
    except Exception as exc:
        logger.exception(
            f"Self-cognition latest graph publication failed: {exc}"
        )
        await event_logging.record_runtime_error_event(
            component="self_cognition.worker",
            error_class=type(exc).__name__,
            error_preview=str(exc),
            stack_fingerprint="self_cognition_latest_graph_publication",
            top_frame_module=__name__,
            recovered=True,
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
