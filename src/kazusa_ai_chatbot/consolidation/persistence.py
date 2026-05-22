"""Consolidator persistence and scheduling helpers."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from kazusa_ai_chatbot.config import (
    AFFINITY_DECREMENT_BREAKPOINTS,
    AFFINITY_INCREMENT_BREAKPOINTS,
    AFFINITY_RAW_DEAD_ZONE,
)
from kazusa_ai_chatbot.consolidation.target import (
    CHARACTER_TARGET_ALIAS,
    GROUP_CHANNEL_TARGET_ALIAS,
    USER_TARGET_ALIAS,
    ConsolidationTargetPlan,
    ConsolidationTargetValidationError,
    validate_write_intent,
)
from kazusa_ai_chatbot.consolidation.group_channel import (
    persist_group_channel_style_image,
)
from kazusa_ai_chatbot.db import (
    DatabaseOperationError,
    get_character_runtime_state,
    update_affinity,
    update_last_relationship_insight,
    upsert_character_self_image,
    upsert_character_state,
)
from kazusa_ai_chatbot.consolidation.images import (
    _update_character_image,
)
from kazusa_ai_chatbot.consolidation.memory_units import (
    update_user_memory_units_from_state,
)
from kazusa_ai_chatbot.consolidation.origin_policy import (
    build_consolidation_write_policy,
)
from kazusa_ai_chatbot.consolidation.schema import (
    ConsolidatorState,
)
from kazusa_ai_chatbot.rag.cache2_events import CacheInvalidationEvent
from kazusa_ai_chatbot.time_boundary import (
    local_llm_datetime_to_storage_utc_iso,
    parse_storage_utc_datetime,
)
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.utils import (
    text_or_empty,
)

logger = logging.getLogger(__name__)


def _target_plan(state: ConsolidatorState) -> ConsolidationTargetPlan:
    """Return the deterministic target plan attached before persistence."""

    target_plan = state["consolidation_target_plan"]
    return target_plan


def _write_intent_is_allowed(
    state: ConsolidatorState,
    *,
    target_alias: str,
    write_lane: str,
    payload: dict,
) -> bool:
    """Validate one target/lane pair without calling persistence helpers."""

    target_plan = _target_plan(state)
    try:
        validate_write_intent(
            {
                "target_alias": target_alias,
                "write_lane": write_lane,
                "payload": payload,
            },
            target_plan,
        )
    except ConsolidationTargetValidationError as exc:
        logger.debug(
            f"db_writer: write intent denied before persistence: {exc}"
        )
        return_value = False
    else:
        return_value = True
    return return_value


def _group_channel_style_image_payload(
    state: ConsolidatorState,
) -> dict[str, object] | None:
    """Validate an optional group-channel style image payload."""

    raw_payload = state.get("group_channel_style_image")
    if raw_payload is None:
        return_value = None
        return return_value
    if not isinstance(raw_payload, dict):
        raise ConsolidationTargetValidationError(
            "group_channel_style_image: expected dict"
        )
    if not raw_payload:
        return_value = None
        return return_value

    overlay = raw_payload.get("overlay")
    if not isinstance(overlay, dict) or not overlay:
        raise ConsolidationTargetValidationError(
            "group_channel_style_image.overlay: expected non-empty dict"
        )
    raw_source_ids = raw_payload.get("source_reflection_run_ids")
    if not isinstance(raw_source_ids, list):
        raise ConsolidationTargetValidationError(
            "group_channel_style_image.source_reflection_run_ids: expected list"
        )

    source_reflection_run_ids: list[str] = []
    for source_id in raw_source_ids:
        if not isinstance(source_id, str):
            raise ConsolidationTargetValidationError(
                "group_channel_style_image.source_reflection_run_ids: "
                "expected string items"
            )
        clean_source_id = source_id.strip()
        if clean_source_id:
            source_reflection_run_ids.append(clean_source_id)

    payload: dict[str, object] = {
        "overlay": dict(overlay),
        "source_reflection_run_ids": source_reflection_run_ids,
    }
    return payload


def process_affinity_delta(current_affinity: int, raw_delta: int) -> int:
    """Scale a raw affinity delta by direction-specific breakpoints.

    Args:
        current_affinity: Current affinity score (0-1000).
        raw_delta: Raw delta from the relationship recorder (-5..+5).

    Returns:
        Scaled delta with sign preserved.
    """
    if raw_delta == 0:
        return 0

    if abs(raw_delta) <= AFFINITY_RAW_DEAD_ZONE:
        return 0

    if raw_delta > 0:
        breakpoints = AFFINITY_INCREMENT_BREAKPOINTS
    else:
        breakpoints = AFFINITY_DECREMENT_BREAKPOINTS

    scaling_factor = 1.0
    for i in range(len(breakpoints) - 1):
        x1, y1 = breakpoints[i]
        x2, y2 = breakpoints[i + 1]

        if x1 <= current_affinity <= x2:
            if x2 == x1:
                scaling_factor = y1
            else:
                scaling_factor = y1 + (current_affinity - x1) * (y2 - y1) / (x2 - x1)
            break

    return_value = int(round(raw_delta * scaling_factor, 0))
    return return_value


def _default_future_promise_due_time(storage_timestamp_utc: str) -> str:
    """Return the immediate fallback due time for untimed future promises.

    Args:
        storage_timestamp_utc: Turn storage timestamp used as the reference
            clock.

    Returns:
        Storage UTC timestamp for now or the next minute boundary.
    """

    reference_time = parse_storage_utc_datetime(storage_timestamp_utc)
    if reference_time.second == 0 and reference_time.microsecond == 0:
        return_value = reference_time.isoformat()
        return return_value

    next_minute = reference_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
    return_value = next_minute.isoformat()
    return return_value


def _normalize_due_time_to_utc(value: str) -> str:
    """Normalize a structured promise time into storage UTC format.

    Args:
        value: Exact local ``YYYY-MM-DD HH:MM`` from the harvester.

    Returns:
        Storage UTC string for persistence.

    Raises:
        ValueError: If ``value`` is not exact configured-local minute text.
    """
    stripped_value = value.strip()
    utc_value = local_llm_datetime_to_storage_utc_iso(stripped_value)

    return_value = utc_value
    return return_value


def _normalize_future_promises(
    future_promises: list[dict],
    *,
    storage_timestamp_utc: str,
) -> list[dict]:
    """Normalize actionable promise due times into UTC storage format.

    Args:
        future_promises: Raw promise rows from the harvester.
        storage_timestamp_utc: Turn storage timestamp used to resolve
            immediate fallbacks.

    Returns:
        Promise rows with ``future_promise`` due times normalized. Rows with
        malformed explicit due times are dropped instead of receiving a
        replacement time.
    """

    fallback_due_time = _default_future_promise_due_time(
        storage_timestamp_utc
    )
    normalized: list[dict] = []

    for promise in future_promises:
        normalized_promise = dict(promise)
        commitment_type = text_or_empty(normalized_promise.get("commitment_type"))
        due_time = normalized_promise.get("due_time")
        due_time_is_missing = (
            due_time is None
            or (isinstance(due_time, str) and not due_time.strip())
        )

        if commitment_type == "future_promise" and due_time_is_missing:
            normalized_promise["due_time"] = fallback_due_time
        elif isinstance(due_time, str) and due_time.strip():
            try:
                normalized_promise["due_time"] = _normalize_due_time_to_utc(
                    due_time
                )
            except ValueError as exc:
                logger.debug(
                    f"Dropping promise with invalid due_time "
                    f"{due_time!r}: {exc}"
                )
                continue

        normalized.append(normalized_promise)

    return normalized


async def db_writer(state: ConsolidatorState) -> dict:
    storage_timestamp_utc = state["storage_timestamp_utc"]
    global_user_id = state.get("global_user_id", "")
    user_name = state.get("user_name", "")

    metadata = dict(state.get("metadata", {}) or {})
    write_log: dict[str, bool] = {}
    cache_invalidated: list[str] = []
    origin_policy = build_consolidation_write_policy(
        origin=state["consolidation_origin"],
    )

    # ── Step 1: character_state (mood / vibe / reflection) ──────────
    mood = state.get("mood", "")
    global_vibe = state.get("global_vibe", "")
    reflection_summary = state.get("reflection_summary", "")
    character_state_allowed = _write_intent_is_allowed(
        state,
        target_alias=CHARACTER_TARGET_ALIAS,
        write_lane="character_state",
        payload={
            "mood": mood,
            "global_vibe": global_vibe,
            "reflection_summary": reflection_summary,
        },
    )
    if origin_policy["character_state"]["allowed"] and character_state_allowed:
        try:
            await upsert_character_state(
                mood=mood,
                global_vibe=global_vibe,
                reflection_summary=reflection_summary,
                updated_at_utc=storage_timestamp_utc,
            )
            write_log["character_state"] = True
        except DatabaseOperationError as exc:
            logger.exception(f"db_writer: failed to upsert character_state: {exc}")
            write_log["character_state"] = False
    else:
        write_log["character_state"] = False

    # ── Step 2: last relationship insight ───────────────────────────
    last_relationship_insight = state.get("last_relationship_insight", "")
    relationship_insight_allowed = _write_intent_is_allowed(
        state,
        target_alias=USER_TARGET_ALIAS,
        write_lane="relationship_insight",
        payload={"last_relationship_insight": last_relationship_insight},
    )
    if global_user_id and last_relationship_insight:
        if (
            origin_policy["relationship_insight"]["allowed"]
            and relationship_insight_allowed
        ):
            try:
                await update_last_relationship_insight(
                    global_user_id,
                    last_relationship_insight,
                )
                write_log["relationship_insight"] = True
            except DatabaseOperationError as exc:
                logger.exception(
                    f"db_writer: failed to update_last_relationship_insight: {exc}"
                )
                write_log["relationship_insight"] = False
        else:
            write_log["relationship_insight"] = False

    # ── Step 3: unified user-memory units ────────────────────────────
    user_memory_units_allowed = _write_intent_is_allowed(
        state,
        target_alias=USER_TARGET_ALIAS,
        write_lane="user_memory_units",
        payload={
            "new_facts": state.get("new_facts") or [],
            "future_promises": state.get("future_promises") or [],
        },
    )
    if origin_policy["user_memory_units"]["allowed"] and user_memory_units_allowed:
        future_promises = _normalize_future_promises(
            state.get("future_promises") or [],
            storage_timestamp_utc=storage_timestamp_utc,
        )
        normalized_state = {
            **state,
            "future_promises": future_promises,
        }
    else:
        future_promises = []
        normalized_state = state

    if origin_policy["user_memory_units"]["allowed"] and user_memory_units_allowed:
        try:
            memory_unit_results = await update_user_memory_units_from_state(
                normalized_state
            )
        except Exception as exc:
            logger.exception(f"db_writer: failed to update user_memory_units: {exc}")
            memory_unit_results = []
            write_log["user_memory_units"] = False
        else:
            write_log["user_memory_units"] = bool(memory_unit_results)
            metadata["user_memory_unit_results"] = memory_unit_results
    else:
        write_log["user_memory_units"] = False

    # ── Step 4: affinity (direction-scaled) ─────────────────────────
    affinity_allowed = _write_intent_is_allowed(
        state,
        target_alias=USER_TARGET_ALIAS,
        write_lane="affinity",
        payload={"affinity_delta": state.get("affinity_delta", 0) or 0},
    )
    raw_affinity_delta = state.get("affinity_delta", 0) or 0
    user_affinity_score: int | None = None
    processed_affinity_delta = 0
    if global_user_id:
        if origin_policy["affinity"]["allowed"] and affinity_allowed:
            user_profile = state["user_profile"]
            user_affinity_score = int(user_profile["affinity"])
            processed_affinity_delta = process_affinity_delta(
                user_affinity_score,
                raw_affinity_delta,
            )
            try:
                await update_affinity(global_user_id, processed_affinity_delta)
                write_log["affinity"] = True
            except DatabaseOperationError as exc:
                logger.exception(f"db_writer: failed to update_affinity: {exc}")
                write_log["affinity"] = False
        else:
            write_log["affinity"] = False

    if user_affinity_score is not None:
        logger.debug(
            f"User {user_name}(@{global_user_id}) affinity "
            f"{user_affinity_score} "
            f"-> {user_affinity_score + processed_affinity_delta}"
        )

    # ── Step 5: group-channel image ─────────────────────────────────
    has_group_channel_target = False
    for target in _target_plan(state)["targets"]:
        if target["target_alias"] == GROUP_CHANNEL_TARGET_ALIAS:
            has_group_channel_target = True
            break

    group_channel_style_payload = _group_channel_style_image_payload(state)
    if group_channel_style_payload is not None:
        group_channel_style_allowed = _write_intent_is_allowed(
            state,
            target_alias=GROUP_CHANNEL_TARGET_ALIAS,
            write_lane="group_channel_style_image",
            payload=group_channel_style_payload,
        )
    else:
        group_channel_style_allowed = False

    if (
        group_channel_style_payload is not None
        and origin_policy["group_channel_style_image"]["allowed"]
        and group_channel_style_allowed
    ):
        try:
            await persist_group_channel_style_image(
                platform=state["platform"],
                platform_channel_id=state["platform_channel_id"],
                overlay=group_channel_style_payload["overlay"],
                source_reflection_run_ids=group_channel_style_payload[
                    "source_reflection_run_ids"
                ],
                storage_timestamp_utc=storage_timestamp_utc,
            )
            write_log["group_channel_style_image"] = True
        except DatabaseOperationError as exc:
            logger.exception(
                f"db_writer: failed to persist group_channel_style_image: {exc}"
            )
            write_log["group_channel_style_image"] = False
    elif has_group_channel_target:
        write_log["group_channel_style_image"] = False

    # ── Step 6: character image ──────────────────────────────────────
    character_image_allowed = _write_intent_is_allowed(
        state,
        target_alias=CHARACTER_TARGET_ALIAS,
        write_lane="character_self_image",
        payload={"reflection_summary": state.get("reflection_summary", "")},
    )
    if origin_policy["character_image"]["allowed"] and character_image_allowed:
        async def _update_character_image_from_runtime_state() -> dict | None:
            """Build character image using the DB-current self-image base."""

            if not state["reflection_summary"]:
                return None

            runtime_state = await get_character_runtime_state()
            runtime_self_image = runtime_state.get("self_image")
            if isinstance(runtime_self_image, dict):
                existing_image = runtime_self_image
            else:
                existing_image = {}

            character_image = await _update_character_image(
                state,
                storage_timestamp_utc=storage_timestamp_utc,
                existing_image=existing_image,
            )
            return character_image

        image_results = await asyncio.gather(
            _update_character_image_from_runtime_state(),
            return_exceptions=True,
        )
        character_image_result = image_results[0]

        if isinstance(character_image_result, Exception):
            logger.error(
                f"db_writer: failed to update character_image: "
                f"{character_image_result}",
                exc_info=(
                    type(character_image_result),
                    character_image_result,
                    character_image_result.__traceback__,
                ),
            )
            write_log["character_image"] = False
        elif character_image_result is not None:
            try:
                await upsert_character_self_image(character_image_result)
                write_log["character_image"] = True
            except DatabaseOperationError as exc:
                logger.exception(
                    f"db_writer: failed to upsert_character_self_image: {exc}"
                )
                write_log["character_image"] = False
    else:
        write_log["character_image"] = False

    # ── Step 7: Cache2 invalidation events (after persistence) ──────
    evicted_total = 0
    if origin_policy["cache_invalidation"]["allowed"]:
        runtime = get_rag_cache2_runtime()
        events: list[CacheInvalidationEvent] = []

        if global_user_id and (
            write_log.get("affinity")
            or write_log.get("relationship_insight")
            or write_log.get("user_memory_units")
        ):
            events.append(CacheInvalidationEvent(
                source="user_profile",
                platform=state["platform"],
                platform_channel_id=state["platform_channel_id"],
                global_user_id=global_user_id,
                storage_timestamp_utc=storage_timestamp_utc,
                reason="consolidator: user_profile",
            ))

        if write_log.get("character_state") or write_log.get("character_image"):
            events.append(CacheInvalidationEvent(
                source="character_state",
                reason="consolidator: character_state",
            ))

        for event in events:
            evicted_total += await runtime.invalidate(event)
        cache_invalidated = [event.source for event in events]
    metadata["cache_evicted_count"] = evicted_total

    metadata.update({
        "write_success": write_log,
        "cache_invalidated": cache_invalidated,
        "affinity_before": user_affinity_score,
        "affinity_delta_processed": processed_affinity_delta,
        "consolidation_target_plan": _target_plan(state),
    })

    logger.debug(
        f"db_writer summary: user={user_name} global_user={global_user_id} "
        f"writes={write_log} cache_invalidated={cache_invalidated} "
        f"affinity_before={user_affinity_score} "
        f"affinity_delta={processed_affinity_delta}"
    )

    return_value = {"metadata": metadata}
    return return_value
