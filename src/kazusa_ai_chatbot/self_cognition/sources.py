"""Visible and actionable source collectors for the self-cognition worker."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from kazusa_ai_chatbot.calendar_scheduler import handlers as calendar_handlers
from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
from kazusa_ai_chatbot.calendar_scheduler import repository as calendar_repository
from kazusa_ai_chatbot.config import (
    CALENDAR_SCHEDULER_MAX_ATTEMPTS,
    CHARACTER_GLOBAL_USER_ID,
    SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED,
    SELF_COGNITION_TRIGGER_GROUP_CHAT_REVIEW_ENABLED,
)
from kazusa_ai_chatbot.consolidation.target import SYNTHETIC_USER_IDS
from kazusa_ai_chatbot.db import (
    get_conversation_history,
    get_user_memory_unit_by_unit_id,
    get_user_profile,
    query_active_commitment_memory_units,
)
from kazusa_ai_chatbot.reflection_cycle.activity_windows import (
    GroupActivityWindow,
    build_group_activity_windows,
)
from kazusa_ai_chatbot.reflection_cycle.group_scene_digest import (
    build_group_scene_digest,
    normalize_group_scene_digest_output,
)
from kazusa_ai_chatbot.reflection_cycle.selector import collect_reflection_inputs
from kazusa_ai_chatbot.rag.conversation_evidence import ConversationEvidenceAgent
from kazusa_ai_chatbot.self_cognition import models
from kazusa_ai_chatbot.self_cognition.group_review_participant_context import (
    THREAD_REFERENCE_GUIDANCE,
    THREAD_REFERENCE_ROW_LIMIT,
    VISIBLE_SAMPLE_CHAR_LIMIT,
    build_group_review_participant_context,
    build_group_review_thread_reference_context,
)
from kazusa_ai_chatbot.self_cognition.sleep_period import (
    is_self_cognition_sleep_period,
)
from kazusa_ai_chatbot.time_boundary import (
    format_storage_utc_for_llm,
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
_SCHEDULED_FUTURE_COGNITION_SOURCE_KIND = "scheduled_future_cognition_slot"
GROUP_REVIEW_CONVERSATION_EVIDENCE_LOOKBACK_HOURS = 72
GROUP_REVIEW_CONVERSATION_EVIDENCE_LIMIT = 5
GROUP_REVIEW_CONVERSATION_EVIDENCE_CHAR_LIMIT = 360
GROUP_REVIEW_CONVERSATION_EVIDENCE_ATTEMPTS = 1


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
    active_commitment_enabled = (
        SELF_COGNITION_TRIGGER_ACTIVE_COMMITMENT_ENABLED
        and remaining_cases > 0
        and not is_self_cognition_sleep_period(now)
    )
    if active_commitment_enabled:
        commitment_cases = await collect_commitment_due_cognition_cases(
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
    participant_context_builder: Callable[..., Any] | None = None,
    thread_reference_context_builder: Callable[..., Any] | None = None,
    scene_digest_builder: Callable[..., Any] | None = None,
    conversation_evidence_builder: Callable[..., Any] | None = None,
) -> list[models.SelfCognitionCase]:
    """Build group-review cases from precomputed activity windows."""

    cases: list[models.SelfCognitionCase] = []
    context_builder = (
        participant_context_builder or build_group_review_participant_context
    )
    thread_context_builder = (
        thread_reference_context_builder
        or build_group_review_thread_reference_context
    )
    digest_builder = scene_digest_builder or build_group_scene_digest
    evidence_builder = (
        conversation_evidence_builder
        or build_group_review_summary_conversation_evidence
    )
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
        participant_context = await _call_maybe_async(
            context_builder,
            participant_rows=[dict(row) for row in window.participant_rows],
            target_scope=case["target_scope"],
            character_profile=character_profile,
            window_start_utc=normalize_storage_utc_iso(
                window.window_start.isoformat(),
            ),
            current_timestamp_utc=normalize_storage_utc_iso(now.isoformat()),
        )
        if participant_context is not None:
            case["conversation_progress"]["participant_context"] = (
                participant_context
            )
        thread_reference_context = await _call_maybe_async(
            thread_context_builder,
            participant_rows=[dict(row) for row in window.participant_rows],
            character_profile=character_profile,
        )
        if _is_group_review_thread_reference_context(
            thread_reference_context,
        ):
            case["conversation_progress"]["thread_reference_context"] = (
                thread_reference_context
            )
        scene_digest = await _call_maybe_async(
            digest_builder,
            window=window,
        )
        normalized_scene_digest = _normalize_group_scene_digest(scene_digest)
        if normalized_scene_digest is not None:
            case["conversation_progress"]["group_scene_digest"] = {
                "digest": normalized_scene_digest["digest"].strip(),
            }
            summary = _group_scene_summary(normalized_scene_digest)
            if summary:
                case["conversation_progress"]["summary"] = summary
                conversation_evidence = await _call_maybe_async(
                    evidence_builder,
                    summary=summary,
                    window=window,
                    target_scope=case["target_scope"],
                    character_profile=character_profile,
                    current_timestamp_utc=normalize_storage_utc_iso(
                        now.isoformat(),
                    ),
                )
                evidence_items = _conversation_evidence_items(
                    conversation_evidence,
                )
                if evidence_items:
                    case["conversation_progress"]["conversation_evidence"] = (
                        evidence_items
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


async def build_group_review_summary_conversation_evidence(
    *,
    summary: str,
    window: GroupActivityWindow,
    target_scope: Mapping[str, Any],
    character_profile: Mapping[str, Any],
    current_timestamp_utc: str,
    conversation_agent: ConversationEvidenceAgent | None = None,
) -> list[str]:
    """Retrieve bounded earlier same-channel evidence for a group topic."""

    clean_summary = text_or_empty(summary).strip()
    if not clean_summary:
        return_value: list[str] = []
        return return_value

    window_start_utc = normalize_storage_utc_iso(window.window_start.isoformat())
    task = _group_review_summary_conversation_task(
        summary=clean_summary,
        window_start_utc=window_start_utc,
    )
    context = _group_review_summary_conversation_context(
        summary=clean_summary,
        window=window,
        target_scope=target_scope,
        character_profile=character_profile,
        window_start_utc=window_start_utc,
        current_timestamp_utc=current_timestamp_utc,
    )
    agent = conversation_agent
    if agent is None:
        agent = ConversationEvidenceAgent()
    raw_result = await agent.run(
        task,
        context,
        max_attempts=GROUP_REVIEW_CONVERSATION_EVIDENCE_ATTEMPTS,
    )
    return_value = _conversation_evidence_items(raw_result)
    return return_value


async def collect_scheduled_future_cognition_cases(
    now: datetime,
    character_profile: dict[str, Any],
    max_cases: int,
    list_due_calendar_runs_func: Callable[..., Any] | None = None,
    get_latest_private_channel_func: Callable[..., Any] | None = None,
    get_user_profile_func: Callable[..., Any] | None = None,
) -> list[models.SelfCognitionCase]:
    """Build worker cases from due calendar future-cognition slots.

    Args:
        now: Current worker tick time.
        character_profile: Current character state snapshot.
        max_cases: Maximum due slots to project.
        list_due_calendar_runs_func: Optional test seam for calendar-run reads.
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
    due_runs_reader = (
        list_due_calendar_runs_func or calendar_repository.list_due_calendar_runs
    )
    profile_reader = get_user_profile_func or get_user_profile
    raw_runs = due_runs_reader(
        current_timestamp_utc=current_now_utc.isoformat(),
        trigger_kinds=[calendar_models.TRIGGER_FUTURE_COGNITION],
        max_attempts=CALENDAR_SCHEDULER_MAX_ATTEMPTS,
        limit=max_cases,
    )
    if inspect.isawaitable(raw_runs):
        raw_runs = await raw_runs

    if raw_runs is None:
        return_value = []
        return return_value
    runs = raw_runs if isinstance(raw_runs, list) else list(raw_runs)

    cases: list[models.SelfCognitionCase] = []
    for run in runs:
        if len(cases) >= max_cases:
            break
        if not isinstance(run, dict):
            continue
        if not _is_due_future_cognition_run(run, current_now_utc):
            continue
        source_scope = _calendar_source_scope(run)
        source_user_id = _non_synthetic_user_id(
            source_scope.get("source_user_id")
        )
        if source_user_id:
            user_profile = await _call_maybe_async(
                profile_reader,
                source_user_id,
            )
        else:
            user_profile = None
        case = _build_scheduled_future_cognition_case(
            run,
            character_profile=character_profile,
            now=current_now_utc,
            user_profile=user_profile,
        )
        if case is None:
            continue
        await _attach_scheduled_delivery_binding(
            case,
            run,
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


async def collect_commitment_due_cognition_cases(
    *,
    now: datetime,
    character_profile: dict[str, Any],
    max_cases: int,
    list_due_calendar_runs_func: Callable[..., Any] | None = None,
    memory_unit_reader_func: Callable[..., Any] | None = None,
    get_conversation_history_func: Callable[..., Any] | None = None,
    get_user_profile_func: Callable[..., Any] | None = None,
    get_latest_private_channel_func: Callable[..., Any] | None = None,
) -> list[models.SelfCognitionCase]:
    """Build active-commitment cases from due commitment calendar runs."""

    if max_cases <= 0:
        return_value: list[models.SelfCognitionCase] = []
        return return_value

    current_now_utc = parse_storage_utc_datetime(now.isoformat())
    due_run_reader = list_due_calendar_runs_func or (
        calendar_repository.list_due_calendar_runs
    )
    memory_unit_reader_func = (
        memory_unit_reader_func or get_user_memory_unit_by_unit_id
    )
    history_reader = get_conversation_history_func or get_conversation_history
    profile_reader = get_user_profile_func or get_user_profile

    raw_runs = await _call_maybe_async(
        due_run_reader,
        current_timestamp_utc=current_now_utc.isoformat(),
        trigger_kinds=[calendar_models.TRIGGER_COMMITMENT_DUE_COGNITION],
        max_attempts=CALENDAR_SCHEDULER_MAX_ATTEMPTS,
        limit=max_cases,
    )
    if raw_runs is None:
        return_value = []
        return return_value

    runs = raw_runs if isinstance(raw_runs, list) else list(raw_runs)
    cases: list[models.SelfCognitionCase] = []

    async def read_memory_unit(unit_id: str) -> dict[str, Any] | None:
        stored_unit = await _call_maybe_async(memory_unit_reader_func, unit_id)
        if isinstance(stored_unit, dict):
            return_value = stored_unit
        else:
            return_value = None
        return return_value

    async def build_case(unit: dict[str, Any]) -> dict[str, Any]:
        case = await _build_commitment_due_case_from_unit(
            unit,
            now=current_now_utc,
            character_profile=character_profile,
            history_reader=history_reader,
            profile_reader=profile_reader,
            get_latest_private_channel_func=get_latest_private_channel_func,
        )
        if case is None:
            return_value: dict[str, Any] = {}
        else:
            return_value = case
        return return_value

    for run in runs:
        if len(cases) >= max_cases:
            break
        if not isinstance(run, dict):
            continue
        if not _is_due_commitment_cognition_run(run, current_now_utc):
            continue
        run_id = text_or_empty(run.get("run_id"))
        handler_result = await calendar_handlers.handle_commitment_due_cognition_run(
            run,
            memory_unit_reader=read_memory_unit,
            active_commitment_case_builder=build_case,
        )
        if handler_result.get("status") == "skipped":
            cases.append({
                "case_name": models.CASE_COMMITMENT_DUPLICATE_TICK,
                "case_id": f"commitment_due_skip:{run_id}",
                "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
                "source_calendar_run_id": run_id,
                "source_calendar_skip_reason": handler_result["reason"],
            })
            continue
        if handler_result.get("status") != "case_created":
            continue
        case = dict(handler_result)
        case.pop("status", None)
        if not text_or_empty(case.get("case_name")):
            continue
        case["source_calendar_run_id"] = run_id
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
            "affinity": models.DEFAULT_SELF_COGNITION_AFFINITY,
            "display_name": "group audience",
            "last_relationship_insight": "",
        },
        "current_mood": text_or_empty(character_profile.get("mood")),
        "global_vibe": text_or_empty(character_profile.get("global_vibe")),
    }
    return case


def _is_due_future_cognition_run(
    run: dict[str, Any],
    now: datetime,
) -> bool:
    """Return whether a calendar run is an eligible future-cognition slot."""

    if run.get("status") not in (
        calendar_models.RUN_STATUS_PENDING,
        calendar_models.RUN_STATUS_RUNNING,
    ):
        return_value = False
        return return_value
    if run.get("trigger_kind") != calendar_models.TRIGGER_FUTURE_COGNITION:
        return_value = False
        return return_value

    due_at = text_or_empty(run.get("due_at"))
    if not due_at:
        return_value = False
        return return_value
    try:
        due_time = parse_storage_utc_datetime(due_at)
    except ValueError:
        return_value = False
        return return_value

    return_value = due_time <= now
    return return_value


def _is_due_commitment_cognition_run(
    run: dict[str, Any],
    now: datetime,
) -> bool:
    """Return whether a calendar run is an eligible commitment due slot."""

    if run.get("status") not in (
        calendar_models.RUN_STATUS_PENDING,
        calendar_models.RUN_STATUS_RUNNING,
    ):
        return_value = False
        return return_value
    if (
        run.get("trigger_kind")
        != calendar_models.TRIGGER_COMMITMENT_DUE_COGNITION
    ):
        return_value = False
        return return_value

    run_id = text_or_empty(run.get("run_id"))
    payload = run.get("payload")
    if not run_id or not isinstance(payload, dict):
        return_value = False
        return return_value
    for field_name in ("unit_id", "global_user_id", "due_at"):
        if not text_or_empty(payload.get(field_name)):
            return_value = False
            return return_value

    due_at = text_or_empty(run.get("due_at"))
    if not due_at:
        return_value = False
        return return_value
    try:
        due_time = parse_storage_utc_datetime(due_at)
    except ValueError:
        return_value = False
        return return_value

    return_value = due_time <= now
    return return_value


async def _attach_scheduled_delivery_binding(
    case: models.SelfCognitionCase,
    run: dict[str, Any],
    *,
    get_latest_private_channel_func: Callable[..., Any] | None,
) -> None:
    """Attach target binding metadata to a scheduled source case."""

    source_scope = _calendar_source_scope(run)
    source_user_id = _non_synthetic_user_id(source_scope.get("source_user_id"))
    binding = await resolve_self_cognition_delivery_target(
        platform=text_or_empty(source_scope.get("source_platform")),
        source_platform_channel_id=text_or_empty(
            source_scope.get("source_channel_id")
        ),
        source_channel_type=text_or_empty(
            source_scope.get("source_channel_type")
        ),
        source_message_id=text_or_empty(source_scope.get("source_message_id")),
        source_ref=text_or_empty(case.get("case_id")),
        source_global_user_id=source_user_id,
        source_platform_bot_id=text_or_empty(
            source_scope.get("source_platform_bot_id")
        ),
        source_character_name=text_or_empty(
            source_scope.get("source_character_name")
        ),
        guild_id=_optional_text(source_scope.get("guild_id")),
        bot_permission_role=text_or_empty(source_scope.get("bot_role")),
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


async def _build_commitment_due_case_from_unit(
    unit: dict[str, Any],
    *,
    now: datetime,
    character_profile: dict[str, Any],
    history_reader: Callable[..., Any],
    profile_reader: Callable[..., Any],
    get_latest_private_channel_func: Callable[..., Any] | None,
) -> models.SelfCognitionCase | None:
    """Project a validated commitment due unit into a worker case."""

    global_user_id = text_or_empty(unit.get("global_user_id"))
    due_at = text_or_empty(unit.get("due_at"))
    if not global_user_id or not due_at:
        return_value = None
        return return_value

    due_state = _due_state(due_at, now)
    if not due_state:
        return_value = None
        return return_value

    rows = await _call_maybe_async(
        history_reader,
        global_user_id=global_user_id,
        limit=models.SOURCE_VISIBLE_CONTEXT_LIMIT,
    )
    if not isinstance(rows, list) or not rows:
        return_value = None
        return return_value

    user_profile = await _call_maybe_async(profile_reader, global_user_id)
    if not isinstance(user_profile, dict):
        user_profile = {}
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
    return case


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


def _normalize_group_scene_digest(value: object) -> dict[str, str] | None:
    """Return a prompt-safe scene digest or ``None`` for invalid output."""

    if not isinstance(value, dict):
        return_value = None
        return return_value
    normalized = normalize_group_scene_digest_output(value)
    return_value = normalized
    return return_value


def _is_group_review_thread_reference_context(value: object) -> bool:
    """Return whether a value matches the bounded thread-warning contract."""

    if not isinstance(value, dict):
        return_value = False
        return return_value
    expected_keys = {
        "source",
        "context_shape",
        "guidance",
        "ambiguous_second_person_rows",
    }
    if set(value.keys()) != expected_keys:
        return_value = False
        return return_value
    if value["source"] != "group_review_thread_reference":
        return_value = False
        return return_value
    if value["context_shape"] != "bounded_second_person_reference_warnings":
        return_value = False
        return return_value
    if value["guidance"] != THREAD_REFERENCE_GUIDANCE:
        return_value = False
        return return_value

    ambiguous_rows = value["ambiguous_second_person_rows"]
    if not isinstance(ambiguous_rows, list) or not ambiguous_rows:
        return_value = False
        return return_value
    if len(ambiguous_rows) > THREAD_REFERENCE_ROW_LIMIT:
        return_value = False
        return return_value

    expected_row_keys = {
        "speaker",
        "sample",
        "referent_status",
        "basis",
    }
    for row in ambiguous_rows:
        if not isinstance(row, dict):
            return_value = False
            return return_value
        if set(row.keys()) != expected_row_keys:
            return_value = False
            return return_value
        for field_name in expected_row_keys:
            if not isinstance(row[field_name], str) or not row[field_name]:
                return_value = False
                return return_value
        if len(row["sample"]) > VISIBLE_SAMPLE_CHAR_LIMIT:
            return_value = False
            return return_value
        if row["referent_status"] != "ambiguous_or_side_thread":
            return_value = False
            return return_value

    return_value = True
    return return_value


def _group_review_summary_conversation_task(
    *,
    summary: str,
    window_start_utc: str,
) -> str:
    """Build the topic-search task string for earlier group conversation."""

    window_start_local = format_storage_utc_for_llm(window_start_utc)
    task = (
        "Conversation-evidence: Find earlier same-channel conversation before "
        f"{window_start_local} that helps explain this group topic: {summary}"
    )
    return task


def _group_review_summary_conversation_context(
    *,
    summary: str,
    window: GroupActivityWindow,
    target_scope: Mapping[str, Any],
    character_profile: Mapping[str, Any],
    window_start_utc: str,
    current_timestamp_utc: str,
) -> dict[str, Any]:
    """Build trusted runtime constraints for topic conversation evidence."""

    lookback_start_utc = _group_review_summary_lookback_start(window_start_utc)
    search_end_utc = _group_review_summary_search_end(window_start_utc)
    context = {
        "platform": text_or_empty(target_scope.get("platform")),
        "platform_channel_id": text_or_empty(
            target_scope.get("platform_channel_id"),
        ),
        "channel_type": text_or_empty(target_scope.get("channel_type")),
        "current_timestamp_utc": current_timestamp_utc,
        "from_timestamp": lookback_start_utc,
        "to_timestamp": search_end_utc,
        "conversation_search_top_k": GROUP_REVIEW_CONVERSATION_EVIDENCE_LIMIT,
        "original_query": text_or_empty(summary),
        "current_slot": "group review topic evidence",
        "known_facts": [text_or_empty(window.labels.get("window_summary"))],
        "active_turn_platform_message_ids": (
            _window_platform_message_ids(window.participant_rows)
        ),
        "active_turn_conversation_row_ids": [],
        "character_profile": {
            "global_user_id": text_or_empty(
                character_profile.get("global_user_id"),
            ),
            "name": text_or_empty(character_profile.get("name")),
        },
    }
    return context


def _group_review_summary_search_end(window_start_utc: str) -> str:
    """Return the inclusive search end strictly before the reviewed window."""

    window_start = parse_storage_utc_datetime(window_start_utc)
    search_end = window_start - timedelta(microseconds=1)
    search_end_utc = normalize_storage_utc_iso(search_end.isoformat())
    return search_end_utc


def _group_review_summary_lookback_start(window_start_utc: str) -> str:
    """Return the earliest timestamp searched for topic evidence."""

    window_start = parse_storage_utc_datetime(window_start_utc)
    lookback_start = window_start - timedelta(
        hours=GROUP_REVIEW_CONVERSATION_EVIDENCE_LOOKBACK_HOURS,
    )
    lookback_start_utc = normalize_storage_utc_iso(lookback_start.isoformat())
    return lookback_start_utc


def _window_platform_message_ids(rows: list[dict[str, Any]]) -> list[str]:
    """Return platform message ids from the reviewed activity window."""

    message_ids = [
        message_id
        for row in rows
        if (message_id := text_or_empty(row.get("platform_message_id")))
    ]
    return message_ids


def _conversation_evidence_items(value: object) -> list[str]:
    """Return bounded evidence strings from a conversation helper result."""

    if isinstance(value, list):
        evidence = _string_items(
            value,
            limit=GROUP_REVIEW_CONVERSATION_EVIDENCE_LIMIT,
            char_limit=GROUP_REVIEW_CONVERSATION_EVIDENCE_CHAR_LIMIT,
        )
        return evidence

    if not isinstance(value, Mapping):
        evidence = []
        return evidence

    result_payload = value.get("result")
    if not isinstance(result_payload, Mapping):
        evidence = []
        return evidence

    raw_evidence = result_payload.get("evidence")
    evidence = _string_items(
        raw_evidence,
        limit=GROUP_REVIEW_CONVERSATION_EVIDENCE_LIMIT,
        char_limit=GROUP_REVIEW_CONVERSATION_EVIDENCE_CHAR_LIMIT,
    )
    if evidence:
        return evidence

    selected_summary = text_or_empty(result_payload.get("selected_summary"))
    evidence = _string_items(
        selected_summary.splitlines(),
        limit=GROUP_REVIEW_CONVERSATION_EVIDENCE_LIMIT,
        char_limit=GROUP_REVIEW_CONVERSATION_EVIDENCE_CHAR_LIMIT,
    )
    return evidence


def _string_items(
    value: object,
    *,
    limit: int,
    char_limit: int,
) -> list[str]:
    """Return capped non-empty text items from a list-like value."""

    if not isinstance(value, list):
        items: list[str] = []
        return items

    items = []
    for item in value:
        text = text_or_empty(item)
        if not text:
            continue
        if len(text) > char_limit:
            text = f"{text[:char_limit - 3].rstrip()}..."
        items.append(text)
        if len(items) >= limit:
            break
    return items


def _group_scene_summary(value: dict[str, Any]) -> str:
    """Return the optional short digest summary string."""

    summary = value.get("summary")
    if not isinstance(summary, str):
        return_value = ""
        return return_value
    return_value = summary.strip()
    return return_value


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
    run: dict[str, Any],
    *,
    character_profile: dict[str, Any],
    now: datetime,
    user_profile: object = None,
) -> models.SelfCognitionCase | None:
    """Project one due calendar future-cognition slot into a worker case."""

    payload = run.get("payload")
    if not isinstance(payload, dict):
        return_value = None
        return return_value
    if payload.get("episode_type") != "self_cognition":
        return_value = None
        return return_value

    continuation_objective = text_or_empty(
        payload.get("continuation_objective")
    )
    if not continuation_objective:
        return_value = None
        return return_value
    run_id = text_or_empty(run.get("run_id"))
    if not run_id:
        return_value = None
        return return_value

    due_at = text_or_empty(run.get("due_at"))
    slot_source_id = _scheduled_future_cognition_slot_source_id(run_id)
    case_id = slot_source_id
    source_action_attempt_id = text_or_empty(
        payload.get("source_action_attempt_id")
    )
    source_scope = _calendar_source_scope(run)
    source_platform = text_or_empty(source_scope.get("source_platform"))
    source_channel_id = text_or_empty(source_scope.get("source_channel_id"))
    source_channel_type = text_or_empty(
        source_scope.get("source_channel_type")
    )
    target_user_id = _non_synthetic_user_id(source_scope.get("source_user_id"))
    source_platform_bot_id = text_or_empty(
        source_scope.get("source_platform_bot_id")
    )
    if not target_user_id and source_channel_type == "group":
        display_name = "group audience"
    else:
        display_name = target_user_id or "scheduled follow-up"
    source_refs = _scheduled_future_cognition_source_refs(
        payload,
        slot_source_id=slot_source_id,
        continuation_objective=continuation_objective,
        due_at=due_at,
    )
    safe_continuation = _safe_continuation(payload.get("continuation"))
    if target_user_id:
        if isinstance(user_profile, dict):
            case_user_profile = dict(user_profile)
        else:
            case_user_profile = {}
    else:
        case_user_profile = {}

    case: models.SelfCognitionCase = {
        "case_name": models.CASE_SCHEDULED_FUTURE_COGNITION,
        "case_id": case_id,
        "idle_timestamp_utc": now.isoformat(),
        "last_evidence_timestamp_utc": (
            _scheduled_run_evidence_timestamp_utc(run)
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
        "platform_bot_id": source_platform_bot_id,
        "source_calendar_run_id": run_id,
        "source_action_attempt_id": source_action_attempt_id,
    }
    return case


def _scheduled_future_cognition_source_refs(
    payload: dict[str, Any],
    *,
    slot_source_id: str,
    continuation_objective: str,
    due_at: str,
) -> list[models.SelfCognitionSourceRef]:
    """Build prompt-safe source references for a calendar cognition slot."""

    source_refs: list[models.SelfCognitionSourceRef] = [
        {
            "source_kind": _SCHEDULED_FUTURE_COGNITION_SOURCE_KIND,
            "source_id": slot_source_id,
            "due_at": due_at or None,
            "summary": continuation_objective,
        }
    ]

    raw_source_refs = payload.get("source_refs")
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


def _scheduled_future_cognition_slot_source_id(run_id: str) -> str:
    """Build a prompt-safe identity for one scheduled cognition slot."""

    digest = calendar_models.stable_json_hash({"source_slot": run_id})
    digest_prefix = digest[:models.STABLE_ID_DIGEST_PREFIX_LENGTH]
    source_id = (
        f"{_SCHEDULED_FUTURE_COGNITION_SOURCE_KIND}:{digest_prefix}"
    )
    return source_id


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


def _calendar_source_scope(run: dict[str, Any]) -> dict[str, Any]:
    """Return calendar source scope or an empty mapping."""

    value = run.get("source_scope")
    if isinstance(value, dict):
        source_scope = value
    else:
        source_scope = {}
    return source_scope


def _scheduled_run_evidence_timestamp_utc(run: dict[str, Any]) -> str:
    """Choose the visible timestamp for a scheduled slot case."""

    due_at = text_or_empty(run.get("due_at"))
    if due_at:
        return due_at
    created_at = text_or_empty(run.get("created_at"))
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
    projected["global_user_id"] = (
        text_or_empty(character_profile.get("global_user_id"))
        or CHARACTER_GLOBAL_USER_ID
    )
    return projected
