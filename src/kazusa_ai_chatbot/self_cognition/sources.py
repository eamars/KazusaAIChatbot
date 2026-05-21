"""Visible and actionable source collectors for the self-cognition worker."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from datetime import datetime
from typing import Any

from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED,
    SELF_COGNITION_TRIGGER_GROUP_CHAT_REVIEW_ENABLED,
)
from kazusa_ai_chatbot.consolidation.target import SYNTHETIC_USER_IDS
from kazusa_ai_chatbot.db import (
    get_conversation_history,
    get_user_profile,
    list_due_future_cognition_events,
    query_active_commitment_memory_units,
)
from kazusa_ai_chatbot.reflection_cycle.activity_windows import (
    GroupActivityWindow,
    build_group_activity_windows,
)
from kazusa_ai_chatbot.reflection_cycle.selector import collect_reflection_inputs
from kazusa_ai_chatbot.self_cognition import models
from kazusa_ai_chatbot.time_boundary import (
    normalize_storage_utc_iso,
    parse_storage_utc_datetime,
)
from kazusa_ai_chatbot.utils import text_or_empty

_CHARACTER_PROFILE_FIELDS = frozenset(
    (
        "name",
        "mood",
        "global_vibe",
        "reflection_summary",
        "personality_brief",
        "boundary_profile",
        "linguistic_texture_profile",
    )
)


async def collect_self_cognition_cases(
    *,
    now: datetime,
    character_profile: dict[str, Any],
    max_cases: int,
) -> list[models.SelfCognitionCase]:
    """Collect bounded worker cases from enabled production source types.

    Args:
        now: Current worker tick time.
        character_profile: Current character state snapshot.
        max_cases: Maximum cases this worker tick may process.

    Returns:
        Self-cognition cases ready for the runner.
    """

    cases: list[models.SelfCognitionCase] = []
    scheduled_cases = await collect_scheduled_future_cognition_cases(
        now=now,
        character_profile=character_profile,
        max_cases=max_cases,
    )
    cases.extend(scheduled_cases)

    remaining_cases = max_cases - len(cases)
    if SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED:
        if remaining_cases > 0:
            commitment_cases = await collect_active_commitment_cases(
                now=now,
                character_profile=character_profile,
                max_cases=remaining_cases,
            )
            cases.extend(commitment_cases)
    selected_cases = cases[:max_cases]
    return selected_cases


async def collect_group_chat_review_cases(
    *,
    now: datetime,
    character_profile: dict[str, Any],
    max_cases: int,
    collect_reflection_inputs_func: Callable[..., Any] | None = None,
) -> list[models.SelfCognitionCase]:
    """Build active group-review cases from reflection activity windows.

    Args:
        now: Current reflection tick time.
        character_profile: Current character state snapshot.
        max_cases: Maximum group-review cases for this tick.
        collect_reflection_inputs_func: Optional reflection input seam.

    Returns:
        Source-channel-bound self-cognition cases for group review.
    """

    if not SELF_COGNITION_TRIGGER_GROUP_CHAT_REVIEW_ENABLED:
        return_value: list[models.SelfCognitionCase] = []
        return return_value

    input_collector = collect_reflection_inputs_func or collect_reflection_inputs
    input_set = await input_collector(
        lookback_hours=3,
        now=now,
        allow_fallback=False,
    )
    window_start = parse_storage_utc_datetime(input_set.effective_start)
    window_end = parse_storage_utc_datetime(input_set.effective_end)
    windows: list[GroupActivityWindow] = []
    for scope in input_set.selected_scopes:
        if scope.channel_type != "group":
            continue
        scope_windows = build_group_activity_windows(
            scope=scope,
            window_start=window_start,
            window_end=window_end,
            now=now,
            character_global_user_id=(
                text_or_empty(character_profile.get("global_user_id"))
                or CHARACTER_GLOBAL_USER_ID
            ),
            platform_bot_id=text_or_empty(
                character_profile.get("platform_bot_id"),
            ),
        )
        windows.extend(scope_windows)
    windows.sort(key=lambda item: item.window_start, reverse=True)

    cases = await collect_group_review_cases(
        now=now,
        character_profile=character_profile,
        windows=windows,
        max_cases=max_cases,
    )
    return cases


async def collect_group_review_cases(
    *,
    now: datetime,
    character_profile: dict[str, Any],
    windows: list[GroupActivityWindow],
    max_cases: int,
) -> list[models.SelfCognitionCase]:
    """Build group-review cases from precomputed activity windows."""

    cases: list[models.SelfCognitionCase] = []
    for window in windows:
        if len(cases) >= max_cases:
            break
        if not window.visible_context:
            continue
        case = _build_group_review_case(
            window,
            character_profile=character_profile,
            now=now,
        )
        binding = await resolve_self_cognition_delivery_target(
            platform=window.platform,
            source_platform_channel_id=window.platform_channel_id,
            source_channel_type=window.channel_type,
            source_message_id=_window_source_message_id(window),
            source_ref=window.source_id,
            source_global_user_id=None,
            source_platform_bot_id=text_or_empty(
                character_profile.get("platform_bot_id"),
            ),
            source_character_name=_profile_character_name(character_profile),
            guild_id=None,
            bot_permission_role="user",
            target_global_user_id=None,
            target_platform_user_id=None,
        )
        _attach_binding(case, binding)
        cases.append(case)

    return cases


async def collect_scheduled_future_cognition_cases(
    now: datetime,
    character_profile: dict[str, Any],
    max_cases: int,
    list_due_events_func: Callable[..., Any] | None = None,
    get_latest_private_channel_func: Callable[..., Any] | None = None,
    get_user_profile_func: Callable[..., Any] | None = None,
) -> list[models.SelfCognitionCase]:
    """Build worker cases from due scheduled future-cognition slots.

    Args:
        now: Current worker tick time.
        character_profile: Current character state snapshot.
        max_cases: Maximum due slots to project.
        list_due_events_func: Optional test seam for scheduled-slot reads.
        get_latest_private_channel_func: Optional test seam for private-channel
            target lookup.
        get_user_profile_func: Optional test seam for real user profile reads.

    Returns:
        Prompt-safe self-cognition cases for the standard worker runner.
    """

    if max_cases <= 0:
        return_value: list[models.SelfCognitionCase] = []
        return return_value

    current_now_utc = parse_storage_utc_datetime(now.isoformat())
    due_events_reader = list_due_events_func or list_due_future_cognition_events
    profile_reader = get_user_profile_func or get_user_profile
    raw_events = due_events_reader(
        current_timestamp_utc=current_now_utc.isoformat(),
        limit=max_cases,
    )
    if inspect.isawaitable(raw_events):
        raw_events = await raw_events

    if raw_events is None:
        return_value = []
        return return_value
    events = raw_events if isinstance(raw_events, list) else list(raw_events)

    cases: list[models.SelfCognitionCase] = []
    for event in events:
        if len(cases) >= max_cases:
            break
        if not isinstance(event, dict):
            continue
        if not _is_due_future_cognition_event(event, current_now_utc):
            continue
        source_user_id = _non_synthetic_user_id(event.get("source_user_id"))
        if source_user_id:
            user_profile = await _call_maybe_async(
                profile_reader,
                source_user_id,
            )
        else:
            user_profile = None
        case = _build_scheduled_future_cognition_case(
            event,
            character_profile=character_profile,
            now=current_now_utc,
            user_profile=user_profile,
        )
        if case is None:
            continue
        await _attach_scheduled_delivery_binding(
            case,
            event,
            get_latest_private_channel_func=get_latest_private_channel_func,
        )
        cases.append(case)

    return cases


async def collect_active_commitment_cases(
    *,
    now: datetime,
    character_profile: dict[str, Any],
    max_cases: int,
    list_active_commitments_func: Callable[..., Any] | None = None,
    get_conversation_history_func: Callable[..., Any] | None = None,
    get_user_profile_func: Callable[..., Any] | None = None,
    get_latest_private_channel_func: Callable[..., Any] | None = None,
) -> list[models.SelfCognitionCase]:
    """Build due-check cases from active commitment memory units.

    Args:
        now: Current worker tick time.
        character_profile: Current character state snapshot.
        max_cases: Maximum active commitments to project.
        list_active_commitments_func: Optional test seam for commitment reads.
        get_conversation_history_func: Optional test seam for visible context.
        get_user_profile_func: Optional test seam for user profile reads.
        get_latest_private_channel_func: Optional test seam for private-channel
            target lookup.

    Returns:
        Bounded self-cognition cases with recent visible context.
    """

    active_commitment_reader = (
        list_active_commitments_func or query_active_commitment_memory_units
    )
    history_reader = get_conversation_history_func or get_conversation_history
    profile_reader = get_user_profile_func or get_user_profile

    units = await active_commitment_reader(
        current_timestamp_utc=now.isoformat(),
        limit=max_cases,
    )
    cases: list[models.SelfCognitionCase] = []
    for unit in units:
        if len(cases) >= max_cases:
            break
        global_user_id = text_or_empty(unit.get("global_user_id"))
        due_at = text_or_empty(unit.get("due_at"))
        if not global_user_id or not due_at:
            continue
        due_state = _due_state(due_at, now)
        if not due_state:
            continue
        rows = await history_reader(
            global_user_id=global_user_id,
            limit=models.SOURCE_VISIBLE_CONTEXT_LIMIT,
        )
        if not rows:
            continue
        user_profile = await profile_reader(global_user_id)
        case = _build_active_commitment_case(
            unit,
            rows,
            user_profile=user_profile,
            character_profile=character_profile,
            now=now,
            due_state=due_state,
        )
        await _attach_active_commitment_delivery_binding(
            case,
            unit,
            rows[-1],
            get_latest_private_channel_func=get_latest_private_channel_func,
        )
        cases.append(case)

    return cases


async def resolve_self_cognition_delivery_target(
    *,
    platform: str,
    source_platform_channel_id: str | None,
    source_channel_type: str | None,
    source_message_id: str | None,
    source_ref: str,
    source_global_user_id: str | None,
    source_platform_bot_id: str | None,
    source_character_name: str | None,
    guild_id: str | None,
    bot_permission_role: str | None,
    target_global_user_id: str | None,
    target_platform_user_id: str | None,
    get_latest_private_channel_func: Callable[..., Any] | None = None,
) -> (
    models.SelfCognitionDeliveryTarget
    | models.SelfCognitionTargetBindingFailure
):
    """Bind a production self-cognition case to a deterministic send target.

    Args:
        platform: Source and target platform key.
        source_platform_channel_id: Original source channel, if known.
        source_channel_type: Original source channel type.
        source_message_id: Original source message id, if known.
        source_ref: Stable case/source reference used for audit.
        source_global_user_id: Global user id from source context.
        source_platform_bot_id: Platform id of the active character.
        source_character_name: Runtime display name of the active character.
        guild_id: Optional platform guild/server scope.
        bot_permission_role: Permission role carried into dispatcher context.
        target_global_user_id: Semantic target user's global id.
        target_platform_user_id: Semantic target user's platform id.
        get_latest_private_channel_func: Deprecated compatibility seam. The
            source-channel target policy does not call it.

    Returns:
        Bound delivery target or an auditable binding failure.
    """

    clean_platform = text_or_empty(platform)
    clean_source_ref = text_or_empty(source_ref)
    clean_source_channel_id = text_or_empty(source_platform_channel_id)
    clean_source_channel_type = text_or_empty(source_channel_type)
    clean_source_message_id = (
        text_or_empty(source_message_id)
        or f"self_cognition:{clean_source_ref}"
    )
    clean_source_global_user_id = (
        text_or_empty(source_global_user_id)
        or text_or_empty(target_global_user_id)
        or None
    )
    clean_target_global_user_id = text_or_empty(target_global_user_id) or None
    clean_target_platform_user_id = (
        text_or_empty(target_platform_user_id) or None
    )

    if not clean_platform:
        failure = _target_binding_failure(
            reason="missing_platform",
            platform=clean_platform,
            source_ref=clean_source_ref,
            source_platform_channel_id=clean_source_channel_id,
            source_channel_type=clean_source_channel_type,
            target_global_user_id=clean_target_global_user_id,
            target_platform_user_id=clean_target_platform_user_id,
        )
        return failure
    if not clean_source_channel_id:
        failure = _target_binding_failure(
            reason="missing_delivery_target",
            platform=clean_platform,
            source_ref=clean_source_ref,
            source_platform_channel_id=clean_source_channel_id,
            source_channel_type=clean_source_channel_type,
            target_global_user_id=clean_target_global_user_id,
            target_platform_user_id=clean_target_platform_user_id,
        )
        return failure
    if clean_source_channel_type not in ("private", "group"):
        failure = _target_binding_failure(
            reason="missing_delivery_target",
            platform=clean_platform,
            source_ref=clean_source_ref,
            source_platform_channel_id=clean_source_channel_id,
            source_channel_type=clean_source_channel_type,
            target_global_user_id=clean_target_global_user_id,
            target_platform_user_id=clean_target_platform_user_id,
        )
        return failure

    target = _delivery_target(
        platform=clean_platform,
        platform_channel_id=clean_source_channel_id,
        channel_type=clean_source_channel_type,
        target_global_user_id=clean_target_global_user_id,
        target_platform_user_id=clean_target_platform_user_id,
        source_kind="self_cognition_source_channel",
        source_ref=clean_source_ref,
        source_platform_channel_id=clean_source_channel_id,
        source_channel_type=clean_source_channel_type,
        source_message_id=clean_source_message_id,
        source_global_user_id=clean_source_global_user_id,
        source_platform_bot_id=text_or_empty(source_platform_bot_id),
        source_character_name=(
            text_or_empty(source_character_name) or "active character"
        ),
        guild_id=guild_id,
        bot_permission_role=text_or_empty(bot_permission_role) or "user",
        fallback_reason="",
    )
    return target


def _build_group_review_case(
    window: GroupActivityWindow,
    *,
    character_profile: dict[str, Any],
    now: datetime,
) -> models.SelfCognitionCase:
    """Project one group activity window into a self-cognition case."""

    window_start = normalize_storage_utc_iso(window.window_start.isoformat())
    window_end = normalize_storage_utc_iso(window.window_end.isoformat())
    semantic_labels = dict(window.semantic_labels)
    case: models.SelfCognitionCase = {
        "case_name": models.CASE_GROUP_CHAT_REVIEW,
        "case_id": f"group_activity_window:{window.source_id}",
        "idle_timestamp_utc": now.isoformat(),
        "last_evidence_timestamp_utc": window.last_evidence_timestamp_utc,
        "trigger_kind": models.TRIGGER_GROUP_CHAT_REVIEW,
        "semantic_due_state": None,
        "actionability": "active_group_review_same_channel_no_fallback",
        "target_scope": {
            "platform": window.platform,
            "platform_channel_id": window.platform_channel_id,
            "channel_type": "group",
            "user_id": None,
        },
        "source_refs": [dict(source_ref) for source_ref in window.source_refs],
        "visible_context": [dict(row) for row in window.visible_context],
        "group_activity_window": {
            "source": "reflection_activity_window",
            "window_start": window_start,
            "window_end": window_end,
            "semantic_labels": semantic_labels,
        },
        "conversation_progress": {
            "source": "reflection_activity_window",
            "window_start": window_start,
            "window_end": window_end,
            "activity_labels": semantic_labels,
        },
        "character_profile": _project_character_profile(character_profile),
        "user_profile": {
            "affinity": models.DEFAULT_DRY_RUN_AFFINITY,
            "display_name": "group audience",
            "last_relationship_insight": "",
        },
        "current_mood": text_or_empty(character_profile.get("mood")),
        "global_vibe": text_or_empty(character_profile.get("global_vibe")),
    }
    return case


def _is_due_future_cognition_event(
    event: dict[str, Any],
    now: datetime,
) -> bool:
    """Return whether a scheduler row is an eligible future-cognition slot."""

    if event.get("status") != "pending":
        return_value = False
        return return_value
    if event.get("tool") != "trigger_future_cognition":
        return_value = False
        return return_value

    execute_at = text_or_empty(event.get("execute_at"))
    if not execute_at:
        return_value = False
        return return_value
    try:
        execute_time = parse_storage_utc_datetime(execute_at)
    except ValueError:
        return_value = False
        return return_value

    return_value = execute_time <= now
    return return_value


async def _attach_scheduled_delivery_binding(
    case: models.SelfCognitionCase,
    event: dict[str, Any],
    *,
    get_latest_private_channel_func: Callable[..., Any] | None,
) -> None:
    """Attach target binding metadata to a scheduled source case."""

    source_user_id = _non_synthetic_user_id(event.get("source_user_id"))
    binding = await resolve_self_cognition_delivery_target(
        platform=text_or_empty(event.get("source_platform")),
        source_platform_channel_id=text_or_empty(event.get("source_channel_id")),
        source_channel_type=text_or_empty(event.get("source_channel_type")),
        source_message_id=text_or_empty(event.get("source_message_id")),
        source_ref=text_or_empty(case.get("case_id")),
        source_global_user_id=source_user_id,
        source_platform_bot_id=text_or_empty(
            event.get("source_platform_bot_id")
        ),
        source_character_name=text_or_empty(
            event.get("source_character_name")
        ),
        guild_id=_optional_text(event.get("guild_id")),
        bot_permission_role=text_or_empty(event.get("bot_role")),
        target_global_user_id=source_user_id,
        target_platform_user_id=None,
        get_latest_private_channel_func=get_latest_private_channel_func,
    )
    _attach_binding(case, binding)


async def _attach_active_commitment_delivery_binding(
    case: models.SelfCognitionCase,
    unit: dict[str, Any],
    latest_row: dict[str, Any],
    *,
    get_latest_private_channel_func: Callable[..., Any] | None,
) -> None:
    """Attach target binding metadata to an active-commitment case."""

    binding = await resolve_self_cognition_delivery_target(
        platform=text_or_empty(latest_row.get("platform")),
        source_platform_channel_id=text_or_empty(
            latest_row.get("platform_channel_id")
        ),
        source_channel_type=text_or_empty(latest_row.get("channel_type")),
        source_message_id=text_or_empty(latest_row.get("platform_message_id")),
        source_ref=text_or_empty(case.get("case_id")),
        source_global_user_id=text_or_empty(unit.get("global_user_id")),
        source_platform_bot_id=text_or_empty(case.get("platform_bot_id")),
        source_character_name=_character_name(case),
        guild_id=_optional_text(latest_row.get("guild_id")),
        bot_permission_role=text_or_empty(latest_row.get("bot_role")) or "user",
        target_global_user_id=text_or_empty(unit.get("global_user_id")),
        target_platform_user_id=text_or_empty(latest_row.get("platform_user_id")),
        get_latest_private_channel_func=get_latest_private_channel_func,
    )
    _attach_binding(case, binding)


def _attach_binding(
    case: models.SelfCognitionCase,
    binding: (
        models.SelfCognitionDeliveryTarget
        | models.SelfCognitionTargetBindingFailure
    ),
) -> None:
    """Store target binding or failure metadata on the case."""

    if binding.get("status") == "target_binding_failed":
        failure = binding
        case["target_binding_status"] = "failed"
        case["target_binding_failure"] = failure
    else:
        target = binding
        case["target_binding_status"] = "bound"
        case["delivery_target"] = target


def _target_binding_failure(
    *,
    reason: str,
    platform: str,
    source_ref: str,
    source_platform_channel_id: str,
    source_channel_type: str,
    target_global_user_id: str | None,
    target_platform_user_id: str | None,
) -> models.SelfCognitionTargetBindingFailure:
    """Build an auditable target-binding failure payload."""

    failure: models.SelfCognitionTargetBindingFailure = {
        "status": "target_binding_failed",
        "reason": reason,
        "platform": platform,
        "source_ref": source_ref,
        "source_platform_channel_id": source_platform_channel_id,
        "source_channel_type": source_channel_type,
        "target_global_user_id": target_global_user_id,
        "target_platform_user_id": target_platform_user_id,
    }
    return failure


def _delivery_target(
    *,
    platform: str,
    platform_channel_id: str,
    channel_type: str,
    target_global_user_id: str | None,
    target_platform_user_id: str | None,
    source_kind: str,
    source_ref: str,
    source_platform_channel_id: str,
    source_channel_type: str,
    source_message_id: str,
    source_global_user_id: str | None,
    source_platform_bot_id: str,
    source_character_name: str,
    guild_id: str | None,
    bot_permission_role: str,
    fallback_reason: str,
) -> models.SelfCognitionDeliveryTarget:
    """Build a normalized delivery target payload."""

    target: models.SelfCognitionDeliveryTarget = {
        "schema_version": "self_cognition_delivery_target.v1",
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "target_global_user_id": target_global_user_id,
        "target_platform_user_id": target_platform_user_id,
        "source_kind": source_kind,
        "source_ref": source_ref,
        "source_platform_channel_id": source_platform_channel_id,
        "source_channel_type": source_channel_type,
        "source_message_id": source_message_id,
        "source_global_user_id": source_global_user_id,
        "source_platform_bot_id": source_platform_bot_id,
        "source_character_name": source_character_name,
        "guild_id": guild_id,
        "bot_permission_role": bot_permission_role,
        "fallback_reason": fallback_reason,
    }
    return target


def _character_name(case: models.SelfCognitionCase) -> str:
    """Read the case character name for delivery audit metadata."""

    profile = case.get("character_profile")
    if not isinstance(profile, dict):
        return_value = "active character"
        return return_value
    return_value = text_or_empty(profile.get("name")) or "active character"
    return return_value


def _optional_text(value: object) -> str | None:
    """Return a stripped string or ``None`` for optional source metadata."""

    clean_value = text_or_empty(value)
    if clean_value:
        return_value = clean_value
    else:
        return_value = None
    return return_value


def _non_synthetic_user_id(value: object) -> str:
    """Return a user id only when the value is not a provenance label."""

    user_id = text_or_empty(value)
    if user_id.casefold() in SYNTHETIC_USER_IDS:
        user_id = ""
    return user_id


async def _call_maybe_async(
    callable_object: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call a sync or async source seam with a common awaitable contract."""

    value = callable_object(*args, **kwargs)
    if inspect.isawaitable(value):
        value = await value
    return value


def _build_scheduled_future_cognition_case(
    event: dict[str, Any],
    *,
    character_profile: dict[str, Any],
    now: datetime,
    user_profile: object = None,
) -> models.SelfCognitionCase | None:
    """Project one due future-cognition slot into a normal worker case."""

    args = event.get("args")
    if not isinstance(args, dict):
        return_value = None
        return return_value
    if args.get("episode_type") != "self_cognition":
        return_value = None
        return return_value

    continuation_objective = text_or_empty(args.get("continuation_objective"))
    if not continuation_objective:
        return_value = None
        return return_value
    event_id = text_or_empty(event.get("event_id"))
    if not event_id:
        return_value = None
        return return_value

    execute_at = text_or_empty(event.get("execute_at"))
    source_action_attempt_id = text_or_empty(
        args.get("source_action_attempt_id")
    )
    source_platform = text_or_empty(event.get("source_platform"))
    source_channel_id = text_or_empty(event.get("source_channel_id"))
    source_channel_type = text_or_empty(event.get("source_channel_type"))
    target_user_id = _non_synthetic_user_id(event.get("source_user_id"))
    source_platform_bot_id = text_or_empty(event.get("source_platform_bot_id"))
    if not target_user_id and source_channel_type == "group":
        display_name = "group audience"
    else:
        display_name = target_user_id or "scheduled follow-up"
    source_refs = _scheduled_future_cognition_source_refs(
        args,
        continuation_objective=continuation_objective,
        execute_at=execute_at,
    )
    safe_continuation = _safe_continuation(args.get("continuation"))
    if target_user_id:
        if isinstance(user_profile, dict):
            case_user_profile = dict(user_profile)
        else:
            case_user_profile = {}
    else:
        case_user_profile = {}

    case: models.SelfCognitionCase = {
        "case_name": models.CASE_SCHEDULED_FUTURE_COGNITION,
        "case_id": event_id,
        "idle_timestamp_utc": now.isoformat(),
        "last_evidence_timestamp_utc": (
            _scheduled_event_evidence_timestamp_utc(event)
        ),
        "trigger_kind": models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
        "semantic_due_state": models.DUE_STATE_DUE_NOW,
        "actionability": "scheduled_private_followup_ready_no_direct_contact",
        "target_scope": {
            "platform": source_platform or "orchestrator",
            "platform_channel_id": source_channel_id,
            "channel_type": source_channel_type or "internal",
            "user_id": target_user_id or None,
            "display_name": display_name,
        },
        "source_refs": source_refs,
        "visible_context": [],
        "conversation_progress": {
            "source": "scheduled_future_cognition",
            "continuation_objective": continuation_objective,
            "continuation": safe_continuation,
        },
        "character_profile": _project_character_profile(character_profile),
        "user_profile": case_user_profile,
        "current_mood": text_or_empty(character_profile.get("mood")),
        "global_vibe": text_or_empty(character_profile.get("global_vibe")),
        "rag_query": continuation_objective,
        "platform_bot_id": source_platform_bot_id,
        "source_scheduled_event_id": event_id,
        "source_action_attempt_id": source_action_attempt_id,
    }
    return case


def _scheduled_future_cognition_source_refs(
    args: dict[str, Any],
    *,
    continuation_objective: str,
    execute_at: str,
) -> list[models.SelfCognitionSourceRef]:
    """Build prompt-safe source references for a scheduled cognition slot."""

    source_refs: list[models.SelfCognitionSourceRef] = [
        {
            "source_kind": "scheduled_event",
            "source_id": "scheduled_future_cognition_slot",
            "due_at": execute_at or None,
            "summary": continuation_objective,
        }
    ]

    raw_source_refs = args.get("source_refs")
    if not isinstance(raw_source_refs, list):
        return source_refs

    for index, raw_ref in enumerate(raw_source_refs):
        if not isinstance(raw_ref, dict):
            continue
        source_ref = {
            "source_kind": _safe_action_source_kind(raw_ref),
            "source_id": f"action_source_ref:{index}",
            "due_at": None,
            "summary": _safe_action_source_summary(raw_ref),
        }
        source_refs.append(source_ref)
    return source_refs


def _safe_action_source_kind(raw_ref: dict[str, Any]) -> str:
    """Project an action source kind without leaking storage identifiers."""

    ref_kind = text_or_empty(raw_ref.get("ref_kind"))
    if ref_kind:
        return_value = f"action_{ref_kind}"
    else:
        return_value = "action_source_ref"
    return return_value


def _safe_action_source_summary(raw_ref: dict[str, Any]) -> str:
    """Summarize action source metadata without raw ids or excerpts."""

    parts: list[str] = []
    for field_name in ("owner", "relationship", "ref_kind"):
        value = text_or_empty(raw_ref.get(field_name))
        if value:
            parts.append(f"{field_name}={value}")
    if not parts:
        return_value = "prior action source reference"
        return return_value
    return_value = "; ".join(parts)
    return return_value


def _safe_continuation(value: object) -> dict[str, Any]:
    """Project bounded continuation metadata for model-visible context."""

    if not isinstance(value, dict):
        return_value: dict[str, Any] = {}
        return return_value

    safe_value: dict[str, Any] = {}
    mode = value.get("mode")
    if isinstance(mode, str):
        safe_value["mode"] = mode
    return safe_value


def _window_source_message_id(window: GroupActivityWindow) -> str:
    """Choose a stable source message id for delivery audit metadata."""

    if window.source_message_refs:
        message_id = text_or_empty(window.source_message_refs[-1].get("message_id"))
        if message_id:
            return_value = message_id
            return return_value
    return_value = window.source_id
    return return_value


def _profile_character_name(character_profile: dict[str, Any]) -> str:
    """Read the active character name from a profile snapshot."""

    name = text_or_empty(character_profile.get("name"))
    if not name:
        name = "active character"
    return name


def _scheduled_event_evidence_timestamp_utc(event: dict[str, Any]) -> str:
    """Choose the visible timestamp for a scheduled slot case."""

    execute_at = text_or_empty(event.get("execute_at"))
    if execute_at:
        return execute_at
    created_at = text_or_empty(event.get("created_at"))
    return created_at


def _build_active_commitment_case(
    unit: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    user_profile: dict[str, Any],
    character_profile: dict[str, Any],
    now: datetime,
    due_state: str,
) -> models.SelfCognitionCase:
    """Project one active commitment and recent transcript into a case."""

    latest_row = rows[-1]
    unit_id = text_or_empty(unit.get("unit_id"))
    due_at = text_or_empty(unit.get("due_at"))
    fact = text_or_empty(unit.get("fact"))
    case_name = models.CASE_COMMITMENT_PAST_DUE
    actionability = "past_due_commitment_contact_socially_available"
    if due_state == models.DUE_STATE_FUTURE_DUE:
        case_name = models.CASE_COMMITMENT_BEFORE_DUE
        actionability = "future_commitment_progress_visible_no_contact_required"

    case: models.SelfCognitionCase = {
        "case_name": case_name,
        "case_id": f"active_commitment:{unit_id}:{due_at}",
        "idle_timestamp_utc": now.isoformat(),
        "last_evidence_timestamp_utc": _last_evidence_timestamp_utc(
            unit,
            rows,
        ),
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "semantic_due_state": due_state,
        "actionability": actionability,
        "target_scope": {
            "platform": text_or_empty(latest_row.get("platform")),
            "platform_channel_id": text_or_empty(
                latest_row.get("platform_channel_id")
            ),
            "channel_type": text_or_empty(latest_row.get("channel_type")),
            "user_id": text_or_empty(unit.get("global_user_id")),
            "platform_user_id": text_or_empty(
                latest_row.get("platform_user_id")
            ),
            "display_name": text_or_empty(latest_row.get("display_name")),
        },
        "source_refs": [
            {
                "source_kind": "user_memory_unit",
                "source_id": unit_id,
                "due_at": due_at,
                "summary": fact,
            }
        ],
        "visible_context": _visible_context(rows),
        "character_profile": _project_character_profile(character_profile),
        "user_profile": dict(user_profile),
        "current_mood": text_or_empty(character_profile.get("mood")),
        "global_vibe": text_or_empty(character_profile.get("global_vibe")),
    }
    return case


def _due_state(due_at: str, now: datetime) -> str:
    """Classify a due timestamp relative to the worker tick."""

    try:
        due_time = parse_storage_utc_datetime(due_at)
    except ValueError:
        return_value = ""
        return return_value
    if due_time > now:
        return_value = models.DUE_STATE_FUTURE_DUE
    elif due_time == now:
        return_value = models.DUE_STATE_DUE_NOW
    else:
        return_value = models.DUE_STATE_PAST_DUE
    return return_value


def _last_evidence_timestamp_utc(
    unit: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    """Choose the most recent visible or memory timestamp for a case."""

    latest_row = rows[-1]
    row_timestamp_utc = text_or_empty(latest_row.get("timestamp"))
    if row_timestamp_utc:
        return_value = row_timestamp_utc
        return return_value
    for field_name in ("updated_at", "last_seen_at", "first_seen_at"):
        timestamp_utc = text_or_empty(unit.get(field_name))
        if timestamp_utc:
            return timestamp_utc
    return_value = ""
    return return_value


def _visible_context(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Project recent conversation rows into runner-visible case context."""

    visible_rows: list[dict[str, str]] = []
    for row in rows:
        body_text = text_or_empty(row.get("body_text"))
        if not body_text:
            continue
        visible_row = {
            "role": text_or_empty(row.get("role")),
            "body_text": body_text,
            "display_name": text_or_empty(row.get("display_name")),
            "timestamp": text_or_empty(row.get("timestamp")),
        }
        visible_rows.append(visible_row)
    return visible_rows


def _project_character_profile(
    character_profile: dict[str, Any],
) -> dict[str, Any]:
    """Project only cognition-relevant character fields for worker cases."""

    projected = {
        field_name: character_profile[field_name]
        for field_name in _CHARACTER_PROFILE_FIELDS
        if field_name in character_profile
    }
    if "name" not in projected:
        projected["name"] = "active character"
    return projected
