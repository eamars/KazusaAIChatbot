"""Daily deterministic sleep recovery for persistent character cognition."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from kazusa_ai_chatbot import db
from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    apply_sleep_recovery as apply_sleep_recovery_state,
)
from kazusa_ai_chatbot.config import (
    AFFECT_SETTLING_WAKE_PREP_MINUTES,
    CHARACTER_SLEEP_LOCAL_PERIOD,
    REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME,
)
from kazusa_ai_chatbot.db.schemas import CharacterReflectionRunDoc
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.reflection_cycle.models import (
    AFFECT_SETTLING_PROMPT_VERSION,
    REFLECTION_RUN_KIND_DAILY_AFFECT_SETTLING,
    REFLECTION_STATUS_DRY_RUN,
    REFLECTION_STATUS_FAILED,
    REFLECTION_STATUS_SKIPPED,
    REFLECTION_STATUS_SUCCEEDED,
    ReflectionWorkerResult,
)
from kazusa_ai_chatbot.reflection_cycle import repository
from kazusa_ai_chatbot.time_boundary import (
    local_time_context_from_storage_utc,
    normalize_storage_utc_iso,
    parse_storage_utc_datetime,
    storage_utc_now_iso,
)


LOCAL_CLOCK_TEXT_LENGTH = 5
LOCAL_CLOCK_SEPARATOR_INDEX = 2
LOCAL_CLOCK_HOUR_END_INDEX = 2
LOCAL_CLOCK_MINUTE_START_INDEX = 3
MAX_LOCAL_HOUR = 23
MAX_LOCAL_MINUTE = 59
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
MINUTES_PER_DAY = HOURS_PER_DAY * MINUTES_PER_HOUR
LOCAL_DATE_END_INDEX = 10
LOCAL_TIME_START_INDEX = 11
LOCAL_TIME_END_INDEX = 16

AFFECT_SETTLING_AFTER_PROMOTION_GRACE_MINUTES = 15
AFFECT_SETTLING_WAKE_DEFER_GRACE_MINUTES = 15
logger = logging.getLogger(__name__)

def compute_affect_settling_due_local_time(
    *,
    sleep_local_period: str,
    promotion_run_after_local_time: str,
    after_promotion_grace_minutes: int,
    wake_prep_minutes: int,
) -> str:
    """Return the local clock time when daily affect settling becomes due."""

    _sleep_start_minutes, sleep_end_minutes = _sleep_period_bounds(
        sleep_local_period,
    )
    promotion_minutes = _local_clock_minutes(promotion_run_after_local_time)
    due_minutes = max(
        promotion_minutes + after_promotion_grace_minutes,
        sleep_end_minutes - wake_prep_minutes,
    )
    due_time = _minutes_to_clock(due_minutes)
    return due_time


def validate_affect_settling_timing(
    *,
    sleep_local_period: str = CHARACTER_SLEEP_LOCAL_PERIOD,
    promotion_run_after_local_time: str = REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME,
    after_promotion_grace_minutes: int = (
        AFFECT_SETTLING_AFTER_PROMOTION_GRACE_MINUTES
    ),
    wake_prep_minutes: int = AFFECT_SETTLING_WAKE_PREP_MINUTES,
    wake_defer_grace_minutes: int = AFFECT_SETTLING_WAKE_DEFER_GRACE_MINUTES,
) -> None:
    """Fail fast when affect settling cannot run inside the wake window.

    Empty sleep period disables the affect-settling schedule through the
    shared sleep-period contract.
    """

    if not sleep_local_period:
        return

    _sleep_start_minutes, sleep_end_minutes = _sleep_period_bounds(
        sleep_local_period,
    )
    promotion_minutes = _local_clock_minutes(promotion_run_after_local_time)
    due_minutes = max(
        promotion_minutes + after_promotion_grace_minutes,
        sleep_end_minutes - wake_prep_minutes,
    )
    latest_minutes = sleep_end_minutes + wake_defer_grace_minutes
    if due_minutes > latest_minutes:
        raise ValueError(
            "AFFECT_SETTLING due time cannot be later than sleep end plus "
            "wake defer grace"
        )


def local_datetime_is_in_affect_settling_window(
    local_datetime: str,
    *,
    sleep_local_period: str = CHARACTER_SLEEP_LOCAL_PERIOD,
    promotion_run_after_local_time: str = REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME,
    after_promotion_grace_minutes: int = (
        AFFECT_SETTLING_AFTER_PROMOTION_GRACE_MINUTES
    ),
    wake_prep_minutes: int = AFFECT_SETTLING_WAKE_PREP_MINUTES,
    wake_defer_grace_minutes: int = AFFECT_SETTLING_WAKE_DEFER_GRACE_MINUTES,
) -> bool:
    """Return whether the local timestamp is inside the affect-settling window."""

    if not sleep_local_period:
        return False

    _sleep_start_minutes, sleep_end_minutes = _sleep_period_bounds(
        sleep_local_period,
    )
    promotion_minutes = _local_clock_minutes(promotion_run_after_local_time)
    due_minutes = max(
        promotion_minutes + after_promotion_grace_minutes,
        sleep_end_minutes - wake_prep_minutes,
    )
    latest_minutes = sleep_end_minutes + wake_defer_grace_minutes
    current_clock = local_datetime[LOCAL_TIME_START_INDEX:LOCAL_TIME_END_INDEX]
    current_minutes = _local_clock_minutes(current_clock)
    if latest_minutes >= MINUTES_PER_DAY and current_minutes < due_minutes:
        current_minutes += MINUTES_PER_DAY
    return_value = due_minutes <= current_minutes <= latest_minutes
    return return_value


def settling_local_date_for_due_affect_settling(
    local_datetime: str,
    *,
    sleep_local_period: str = CHARACTER_SLEEP_LOCAL_PERIOD,
    promotion_run_after_local_time: str = REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME,
    after_promotion_grace_minutes: int = (
        AFFECT_SETTLING_AFTER_PROMOTION_GRACE_MINUTES
    ),
    wake_prep_minutes: int = AFFECT_SETTLING_WAKE_PREP_MINUTES,
) -> str:
    """Return the sleep-ending date due for worker execution, if any."""

    if not sleep_local_period:
        return_value = ""
        return return_value

    _sleep_start_minutes, sleep_end_minutes = _sleep_period_bounds(
        sleep_local_period,
    )
    promotion_minutes = _local_clock_minutes(promotion_run_after_local_time)
    due_minutes = max(
        promotion_minutes + after_promotion_grace_minutes,
        sleep_end_minutes - wake_prep_minutes,
    )
    current_clock = local_datetime[LOCAL_TIME_START_INDEX:LOCAL_TIME_END_INDEX]
    current_minutes = _local_clock_minutes(current_clock)
    due_clock_minutes = due_minutes % MINUTES_PER_DAY
    if due_minutes >= MINUTES_PER_DAY and current_minutes < due_clock_minutes:
        current_minutes += MINUTES_PER_DAY
    if current_minutes < due_minutes:
        return_value = ""
        return return_value
    return_value = local_datetime[:LOCAL_DATE_END_INDEX]
    return return_value


async def should_pause_self_cognition_for_affect_settling(
    *,
    now: datetime,
) -> bool:
    """Return whether self-cognition should pause for pending affect settling."""

    now_utc = normalize_storage_utc_iso(now.isoformat())
    local_time_context = local_time_context_from_storage_utc(now_utc)
    local_datetime = local_time_context["current_local_datetime"]
    if not local_datetime_is_in_affect_settling_window(local_datetime):
        return_value = False
        return return_value

    settling_local_date = local_datetime[:LOCAL_DATE_END_INDEX]
    run_id = repository.daily_affect_settling_run_id(
        settling_local_date=settling_local_date,
    )
    existing = await repository.reflection_run_by_id(run_id)
    return_value = not _affect_settling_doc_blocks_retry(existing)
    return return_value


async def run_daily_affect_settling(
    *,
    settling_local_date: str,
    dry_run: bool,
    enable_character_state_write: bool,
    character_state_refresh_callback: Callable[[], Any] | None = None,
) -> ReflectionWorkerResult:
    """Run one persistent daily affect-settling pass."""

    return await _run_daily_sleep_recovery(
        settling_local_date=settling_local_date,
        dry_run=dry_run,
        enable_character_state_write=enable_character_state_write,
        character_state_refresh_callback=character_state_refresh_callback,
    )


def sleep_recovery(
    state: dict[str, Any],
    *,
    local_date_key: str,
    elapsed_sleep_seconds: int,
    started_at: str,
    completed_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply deterministic character recovery and build its audit artifact."""

    recovered_state = apply_sleep_recovery_state(
        state,
        elapsed_sleep_seconds=elapsed_sleep_seconds,
        updated_at=completed_at,
    )
    changed_paths = _changed_state_paths(state, recovered_state)
    artifact = {
        "status": "completed",
        "local_date_key": local_date_key,
        "state_scope": "character",
        "elapsed_sleep_seconds": elapsed_sleep_seconds,
        "changed_paths": changed_paths,
        "semantic_recovery_summary": _recovery_summary(
            state,
            recovered_state,
        ),
        "started_at": started_at,
        "completed_at": completed_at,
    }
    return recovered_state, artifact


async def _run_daily_sleep_recovery(
    *,
    settling_local_date: str,
    dry_run: bool,
    enable_character_state_write: bool,
    character_state_refresh_callback: Callable[[], Any] | None,
) -> ReflectionWorkerResult:
    """Run one idempotent deterministic sleep-recovery pass."""

    result = ReflectionWorkerResult(
        run_kind=REFLECTION_RUN_KIND_DAILY_AFFECT_SETTLING,
        dry_run=dry_run,
    )
    run_id = repository.daily_affect_settling_run_id(
        settling_local_date=settling_local_date,
    )
    result.run_ids.append(run_id)
    existing = await repository.reflection_run_by_id(run_id)
    if _affect_settling_doc_blocks_retry(existing):
        result.skipped_count = 1
        result.defer_reason = "daily sleep recovery already completed"
        return result

    result.processed_count = 1
    state = await db.get_character_cognition_state()
    started_at = storage_utc_now_iso()
    elapsed_sleep_seconds = _configured_sleep_interval_seconds()
    recovered_state, artifact = sleep_recovery(
        state,
        local_date_key=settling_local_date,
        elapsed_sleep_seconds=elapsed_sleep_seconds,
        started_at=started_at,
        completed_at=storage_utc_now_iso().replace("+00:00", "Z"),
    )
    output = {
        "sleep_recovery": artifact,
        "retryable": False,
    }
    if dry_run:
        await _persist_affect_settling_run(
            settling_local_date=settling_local_date,
            status=REFLECTION_STATUS_DRY_RUN,
            source_run_ids=[],
            output=output,
            validation_warnings=[],
        )
        result.skipped_count = 1
        return result
    if not enable_character_state_write:
        await _persist_affect_settling_run(
            settling_local_date=settling_local_date,
            status=REFLECTION_STATUS_SKIPPED,
            source_run_ids=[],
            output={
                "skip_reason": "character_state_write_disabled",
                **output,
            },
            validation_warnings=[],
        )
        result.skipped_count = 1
        return result

    try:
        await db.replace_character_cognition_state(recovered_state)
    except DatabaseOperationError as exc:
        logger.exception(f"Daily sleep recovery persistence failed: {exc}")
        await _persist_affect_settling_run(
            settling_local_date=settling_local_date,
            status=REFLECTION_STATUS_FAILED,
            source_run_ids=[],
            output={
                "retryable": True,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            validation_warnings=[],
            error=type(exc).__name__,
        )
        result.failed_count = 1
        result.validation_warnings.append(type(exc).__name__)
        return result

    await _persist_affect_settling_run(
        settling_local_date=settling_local_date,
        status=REFLECTION_STATUS_SUCCEEDED,
        source_run_ids=[],
        output=output,
        validation_warnings=[],
    )
    try:
        await _call_refresh_callback(character_state_refresh_callback)
    except Exception as exc:
        logger.exception(
            f"Affect-settling runtime state refresh failed: {exc}"
        )
        await event_logging.record_runtime_error_event(
            component="reflection_cycle.affect_settling",
            error_class=type(exc).__name__,
            error_preview=str(exc),
            stack_fingerprint="affect_settling_runtime_state_refresh",
            top_frame_module=__name__,
            recovered=True,
            run_id=run_id,
        )
    result.succeeded_count = 1
    return result


def _elapsed_seconds_since(value: Any, now_value: str) -> int:
    """Return non-negative elapsed seconds between two storage timestamps."""

    if not isinstance(value, str) or not value:
        return 0
    try:
        elapsed = (
            parse_storage_utc_datetime(now_value)
            - parse_storage_utc_datetime(value)
        )
    except (TypeError, ValueError):
        return 0
    return max(0, int(elapsed.total_seconds()))


def _configured_sleep_interval_seconds() -> int:
    """Return the configured sleep-window duration, independent of state age."""

    start_text, end_text = CHARACTER_SLEEP_LOCAL_PERIOD.split("-", 1)
    start_hour, start_minute = (int(value) for value in start_text.split(":"))
    end_hour, end_minute = (int(value) for value in end_text.split(":"))
    start_minutes = start_hour * MINUTES_PER_HOUR + start_minute
    end_minutes = end_hour * MINUTES_PER_HOUR + end_minute
    if end_minutes <= start_minutes:
        end_minutes += MINUTES_PER_DAY
    return (end_minutes - start_minutes) * MINUTES_PER_HOUR


def _changed_state_paths(
    before: dict[str, Any],
    after: dict[str, Any],
) -> list[str]:
    """List changed transient paths without embedding a state snapshot."""

    paths: list[str] = []
    for drive_id, drive in before.get("drives", {}).items():
        updated_drive = after.get("drives", {}).get(drive_id, {})
        if drive.get("pressure") != updated_drive.get("pressure"):
            paths.append(f"drives.{drive_id}.pressure")
    for field_name in ("goals", "threats", "active_events", "knowledge_gaps"):
        before_entities = {
            entity.get("entity_id"): entity
            for entity in before.get(field_name, [])
            if isinstance(entity, dict)
        }
        after_entities = {
            entity.get("entity_id"): entity
            for entity in after.get(field_name, [])
            if isinstance(entity, dict)
        }
        for entity_id, entity in before_entities.items():
            updated_entity = after_entities.get(entity_id, {})
            for axis in ("salience", "residual_pressure"):
                if entity.get(axis) != updated_entity.get(axis):
                    paths.append(f"{field_name}.{entity_id}.{axis}")
    for before_activation in before.get("affect_activations", []):
        if not isinstance(before_activation, dict):
            continue
        activation_id = before_activation.get("activation_id")
        updated_activation = next(
            (
                activation
                for activation in after.get("affect_activations", [])
                if isinstance(activation, dict)
                and activation.get("activation_id") == activation_id
            ),
            {},
        )
        if before_activation.get("score") != updated_activation.get("score"):
            paths.append(f"affect_activations.{activation_id}.score")
    return sorted(paths)


def _recovery_summary(
    before: dict[str, Any],
    after: dict[str, Any],
) -> str:
    """Summarize deterministic recovery semantics without raw values."""

    changed_paths = _changed_state_paths(before, after)
    if not changed_paths:
        return "Sleep recovery completed with no transient state changes."
    categories = []
    if any(path.startswith("drives.") for path in changed_paths):
        categories.append("drive pressure was reduced")
    if any(path.endswith(".salience") for path in changed_paths):
        categories.append("transient causal salience was settled")
    if any(
        path.startswith("threats.") and path.endswith(".residual_pressure")
        for path in changed_paths
    ):
        categories.append("threat pressure was recovered")
    if any(path.startswith("affect_activations.") for path in changed_paths):
        categories.append("derived affect activation was recomputed")
    return "Sleep recovery " + "; ".join(categories) + "."

async def _persist_affect_settling_run(
    *,
    settling_local_date: str,
    status: str,
    source_run_ids: list[str],
    output: dict[str, Any],
    validation_warnings: list[str],
    error: str = "",
) -> None:
    """Persist one affect-settling audit row."""

    document = repository.build_daily_affect_settling_run_document(
        settling_local_date=settling_local_date,
        prompt_version=AFFECT_SETTLING_PROMPT_VERSION,
        source_run_ids=source_run_ids,
        output=output,
        status=status,
        attempt_count=1,
        validation_warnings=validation_warnings,
        error=error,
    )
    await repository.upsert_run(document)


async def _call_refresh_callback(
    callback: Callable[[], Any] | None,
) -> None:
    """Call an optional sync or async runtime refresh callback."""

    if callback is None:
        return
    value = callback()
    if inspect.isawaitable(value):
        await value


def _affect_settling_doc_blocks_retry(
    document: CharacterReflectionRunDoc | None,
) -> bool:
    """Return whether an existing affect run should block another attempt."""

    if document is None:
        return False
    status = str(document.get("status", "") or "")
    if status == REFLECTION_STATUS_SUCCEEDED:
        return True
    if status != REFLECTION_STATUS_SKIPPED:
        return False
    output = document.get("output")
    if not isinstance(output, dict):
        return False
    return_value = output.get("retryable") is False
    return return_value


def _sleep_period_bounds(sleep_local_period: str) -> tuple[int, int]:
    """Parse exact sleep period bounds into local minutes."""

    parts = sleep_local_period.split("-", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("sleep_local_period must use HH:MM-HH:MM")
    start_minutes = _local_clock_minutes(parts[0])
    end_minutes = _local_clock_minutes(parts[1])
    return_value = (start_minutes, end_minutes)
    return return_value


def _local_clock_minutes(value: str) -> int:
    """Parse exact ``HH:MM`` text into minutes after local midnight."""

    if (
        len(value) != LOCAL_CLOCK_TEXT_LENGTH
        or value[LOCAL_CLOCK_SEPARATOR_INDEX] != ":"
    ):
        raise ValueError("local clock value must use HH:MM")
    hour_text = value[:LOCAL_CLOCK_HOUR_END_INDEX]
    minute_text = value[LOCAL_CLOCK_MINUTE_START_INDEX:]
    if not hour_text.isdecimal() or not minute_text.isdecimal():
        raise ValueError("local clock value must use HH:MM")
    hour = int(hour_text)
    minute = int(minute_text)
    if hour > MAX_LOCAL_HOUR or minute > MAX_LOCAL_MINUTE:
        raise ValueError("local clock value must use HH:MM")
    return_value = (hour * MINUTES_PER_HOUR) + minute
    return return_value


def _minutes_to_clock(minutes: int) -> str:
    """Return ``HH:MM`` text for local minutes, preserving local-day clock."""

    clock_minutes = minutes % MINUTES_PER_DAY
    hour = clock_minutes // MINUTES_PER_HOUR
    minute = clock_minutes % MINUTES_PER_HOUR
    return_value = f"{hour:02d}:{minute:02d}"
    return return_value


validate_affect_settling_timing()
