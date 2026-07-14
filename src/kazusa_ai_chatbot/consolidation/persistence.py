"""Consolidator persistence and scheduling helpers."""

from __future__ import annotations

import logging
from datetime import timedelta

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
from kazusa_ai_chatbot.consolidation.character_self_guidance import (
    persist_character_self_guidance_from_state,
)
from kazusa_ai_chatbot.db import (
    DatabaseOperationError,
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


def _consolidation_lane_is_enabled(
    state: ConsolidatorState,
    lane: str,
) -> bool:
    """Return whether a lane-router-selected lane should write."""

    enabled_lanes = state.get("enabled_consolidation_write_lanes")
    if not isinstance(enabled_lanes, (list, tuple, set)):
        return_value = False
        return return_value
    return_value = lane in enabled_lanes
    return return_value


def _consolidation_lane_was_routed(
    state: ConsolidatorState,
    lane: str,
) -> bool:
    """Return whether the lane-router explicitly accepted a lane."""

    enabled_lanes = state.get("enabled_consolidation_write_lanes")
    if not isinstance(enabled_lanes, (list, tuple, set)):
        return_value = False
        return return_value
    return_value = lane in enabled_lanes
    return return_value


def _any_consolidation_lane_enabled(
    state: ConsolidatorState,
    lanes: set[str],
) -> bool:
    """Return whether any one lane from a group is enabled."""

    enabled_lanes = state.get("enabled_consolidation_write_lanes")
    if not isinstance(enabled_lanes, (list, tuple, set)):
        return_value = False
        return return_value
    enabled_lane_set = set(enabled_lanes)
    return_value = bool(enabled_lane_set & lanes)
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
        value: Exact local ``YYYY-MM-DD HH:MM`` from a lane specialist.

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
        future_promises: Raw promise rows from lane specialist output.
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

    # ── Step 1: unified user-memory units ────────────────────────────
    user_memory_units_allowed = _write_intent_is_allowed(
        state,
        target_alias=USER_TARGET_ALIAS,
        write_lane="user_memory_units",
        payload={
            "new_facts": state.get("new_facts") or [],
            "future_promises": state.get("future_promises") or [],
        },
    )
    user_memory_lane_enabled = _any_consolidation_lane_enabled(
        state,
        {"user_memory_units", "active_commitment"},
    )
    if (
        origin_policy["user_memory_units"]["allowed"]
        and user_memory_units_allowed
        and user_memory_lane_enabled
    ):
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

    if (
        origin_policy["user_memory_units"]["allowed"]
        and user_memory_units_allowed
        and user_memory_lane_enabled
    ):
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

    # ── Step 2: group-channel image ─────────────────────────────────
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
        and _consolidation_lane_is_enabled(state, "interaction_style_image")
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

    # ── Step 3: character self-guidance memory ──────────────────────
    character_self_guidance_payload = state.get("character_self_guidance")
    has_character_self_guidance_payload = isinstance(
        character_self_guidance_payload,
        dict,
    ) and bool(character_self_guidance_payload)
    character_self_guidance_routed = _consolidation_lane_was_routed(
        state,
        "character_self_guidance",
    )
    if has_character_self_guidance_payload:
        character_self_guidance_allowed = _write_intent_is_allowed(
            state,
            target_alias=CHARACTER_TARGET_ALIAS,
            write_lane="character_self_guidance",
            payload=character_self_guidance_payload,
        )
    else:
        character_self_guidance_allowed = False

    if (
        has_character_self_guidance_payload
        and origin_policy["character_self_guidance"]["allowed"]
        and character_self_guidance_allowed
        and _consolidation_lane_is_enabled(state, "character_self_guidance")
    ):
        try:
            character_self_guidance_result = (
                await persist_character_self_guidance_from_state(state)
            )
        except Exception as exc:
            logger.exception(
                f"db_writer: failed to persist character_self_guidance: {exc}"
            )
            write_log["character_self_guidance"] = False
        else:
            write_log["character_self_guidance"] = (
                character_self_guidance_result is not None
            )
            if character_self_guidance_result is not None:
                metadata["character_self_guidance_result"] = (
                    character_self_guidance_result
                )
    elif character_self_guidance_routed:
        write_log["character_self_guidance"] = False

    # ── Step 8: Cache2 invalidation events (after persistence) ──────
    evicted_total = 0
    if origin_policy["cache_invalidation"]["allowed"]:
        runtime = get_rag_cache2_runtime()
        events: list[CacheInvalidationEvent] = []

        if global_user_id and write_log.get("user_memory_units"):
            events.append(CacheInvalidationEvent(
                source="user_profile",
                platform=state["platform"],
                platform_channel_id=state["platform_channel_id"],
                global_user_id=global_user_id,
                storage_timestamp_utc=storage_timestamp_utc,
                reason="consolidator: user_profile",
            ))

        if write_log.get("character_self_guidance"):
            events.append(CacheInvalidationEvent(
                source="character_self_guidance",
                reason="consolidator: character_self_guidance",
            ))

        for event in events:
            evicted_total += await runtime.invalidate(event)
        cache_invalidated = [event.source for event in events]
    metadata["cache_evicted_count"] = evicted_total

    metadata.update({
        "write_success": write_log,
        "cache_invalidated": cache_invalidated,
        "consolidation_target_plan": _target_plan(state),
    })

    logger.debug(
        f"db_writer summary: user={user_name} global_user={global_user_id} "
        f"writes={write_log} cache_invalidated={cache_invalidated}"
    )

    return_value = {"metadata": metadata}
    return return_value
