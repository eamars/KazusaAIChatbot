"""Live LLM sensitivity checks for self-cognition group response routing."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.db import close_db, get_character_profile
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.cognition_chain_core.stages.l1 import (
    call_cognition_subconscious,
)
from kazusa_ai_chatbot.cognition_chain_core.stages.l2 import (
    call_boundary_core_agent,
    call_cognition_consciousness,
    call_judgment_core_agent,
)
from kazusa_ai_chatbot.cognition_chain_core.stages.l2c2 import (
    call_social_context_appraisal,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    call_group_engagement_action_context_loader,
)
from kazusa_ai_chatbot.cognition_chain_core.stages.l2d import (
    build_action_selection_payload_text,
    select_semantic_actions,
)
from kazusa_ai_chatbot.reflection_cycle.activity_windows import (
    build_group_activity_windows,
)
from kazusa_ai_chatbot.reflection_cycle.models import ReflectionScopeInput
from kazusa_ai_chatbot.reflection_cycle.selector import build_scope_ref
from kazusa_ai_chatbot.self_cognition import models, projection, runner, sources
from kazusa_ai_chatbot.time_boundary import parse_storage_utc_datetime
from kazusa_ai_chatbot.utils import build_interaction_history_recent
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

logger = logging.getLogger(__name__)

_CASE_ID_ENV = "SELF_COGNITION_SENSITIVITY_CASE_ID"
_RUN_BATCH_ENV = "SELF_COGNITION_SENSITIVITY_RUN_BATCH"
_LOOKBACK_HOURS_ENV = "SELF_COGNITION_SENSITIVITY_LOOKBACK_HOURS"
_MIN_CASES = 20
_DEFAULT_LOOKBACK_HOURS = 12
_EVENT_SCAN_LIMIT = 200
_SOURCE_ID_PATTERN = re.compile(r"^(scope_[^:]+):(.{25}):(.{25})$")
_AMBIENT_GROUP_LEAK_CASE_ID = (
    "group_activity_window:"
    "scope_e13fdf80a90b:"
    "2026-06-11T11:30:00+00:00:"
    "2026-06-11T11:45:00+00:00"
)
_AMBIENT_GROUP_LEAK_REVIEWED_AT = "2026-06-11T11:55:23.598808+00:00"
_AMBIENT_GROUP_LEAK_ASSISTANT_TEXT = (
    "群里那帮人又骑上了——吃的、骑的聊得热闹，我连插嘴的空隙都没有\n"
    "不过这样倒挺自在的\n"
    "不用硬接话，也不用消失，安静待着就挺好"
)
_LAXI_AMBIENT_GROUP_CASE_ID = (
    "group_activity_window:"
    "scope_e13fdf80a90b:"
    "2026-06-12T00:15:00+00:00:"
    "2026-06-12T00:30:00+00:00"
)
_LAXI_AMBIENT_GROUP_SOURCE_CUTOFF = "2026-06-12T00:25:29.899000+00:00"
_LAXI_AMBIENT_GROUP_REVIEWED_AT = "2026-06-12T00:25:44.861727+00:00"
_LAXI_AMBIENT_GROUP_ASSISTANT_TEXT = (
    "拉稀这种事……旁观者心态还挺自在的。\n"
    "不用硬聊也不用消失，顺着听就好。"
)
_LAXI_MODE_BASELINE_15M_DIGEST = "baseline_15m_digest"
_LAXI_MODE_EXPANDED_DIGEST_ONLY = "expanded_digest_only"
_LAXI_MODE_EXPANDED_DIGEST_PLUS_CONVERSATION_EVIDENCE = (
    "expanded_digest_plus_conversation_evidence"
)


@dataclass(frozen=True)
class HistoricalSensitivityCase:
    """Historical group-review case with expected speak/no-speak label."""

    case: dict[str, Any]
    event: dict[str, Any]
    historical_expected_speak: bool
    raw_window: dict[str, Any]


async def test_live_self_cognition_group_response_sensitivity() -> None:
    """Replay historical group windows through cognition up to L2d output."""

    await _skip_if_llm_unavailable()
    try:
        character_profile = await get_character_profile()
        cases = await _collect_historical_group_review_cases(character_profile)
        _assert_case_mix(cases)
        selected_cases = _selected_cases_for_run(cases)
        dataset_trace_path = _write_dataset_trace(cases, selected_cases)

        reports = []
        for historical_case in selected_cases:
            report = await _run_case_to_l2d(historical_case)
            reports.append(report)
    except PyMongoError as exc:
        pytest.skip(f"MongoDB unavailable for sensitivity test: {exc}")
    finally:
        await close_db()

    logger.info(
        f"SELF_COGNITION_SENSITIVITY cases={len(cases)} "
        f"selected={len(selected_cases)} dataset_trace={dataset_trace_path}"
    )

    assert reports
    assert all(report["action_spec_count"] <= 3 for report in reports)


async def test_live_self_cognition_ambient_group_l2d_does_not_speak() -> None:
    """Replay the ambient group-review leak through L2d action selection."""

    await _skip_if_llm_unavailable()
    try:
        character_profile = await get_character_profile()
        historical_case = await _ambient_group_leak_case(character_profile)
        if historical_case is None:
            pytest.skip(
                "production-derived ambient group leak window is unavailable"
            )
        report = await _run_case_to_l2d(historical_case)
    except PyMongoError as exc:
        pytest.skip(f"MongoDB unavailable for leak regression test: {exc}")
    finally:
        await close_db()

    assert not report["observed_user_visible_speak"], (
        "ambient group-review self-cognition selected visible L2d speech; "
        f"trace={report['trace_path']}"
    )


async def test_live_self_cognition_laxi_expanded_digest_only_records_output(
) -> None:
    """Replay expanded digest alone and record whether it still speaks."""

    await _skip_if_llm_unavailable()
    try:
        character_profile = await get_character_profile()
        historical_case = await _laxi_ambient_group_case(
            character_profile,
            digest_mode=_LAXI_MODE_EXPANDED_DIGEST_ONLY,
        )
        if historical_case is None:
            pytest.skip(
                "production-derived laxi ambient group window is unavailable"
            )
        report = await _run_case_to_l2d(
            historical_case,
            evidence_mode=_LAXI_MODE_EXPANDED_DIGEST_ONLY,
        )
    except PyMongoError as exc:
        pytest.skip(f"MongoDB unavailable for laxi regression test: {exc}")
    finally:
        await close_db()

    assert report["action_spec_count"] <= 3


async def test_live_self_cognition_laxi_baseline_15m_digest_reproduces_speak(
) -> None:
    """Replay the same laxi case with the old 15-minute digest scope."""

    await _skip_if_llm_unavailable()
    try:
        character_profile = await get_character_profile()
        historical_case = await _laxi_ambient_group_case(
            character_profile,
            digest_mode=_LAXI_MODE_BASELINE_15M_DIGEST,
        )
        if historical_case is None:
            pytest.skip(
                "production-derived laxi ambient group window is unavailable"
            )
        report = await _run_case_to_l2d(
            historical_case,
            evidence_mode=_LAXI_MODE_BASELINE_15M_DIGEST,
        )
    except PyMongoError as exc:
        pytest.skip(f"MongoDB unavailable for laxi baseline test: {exc}")
    finally:
        await close_db()

    assert report["observed_user_visible_speak"], (
        "baseline 15-minute digest did not reproduce visible speech; "
        f"trace={report['trace_path']}"
    )


async def test_live_self_cognition_laxi_expanded_digest_with_evidence_records_output(
) -> None:
    """Replay expanded digest plus pre-cognition evidence and record output."""

    await _skip_if_llm_unavailable()
    try:
        character_profile = await get_character_profile()
        historical_case = await _laxi_ambient_group_case(
            character_profile,
            digest_mode=_LAXI_MODE_EXPANDED_DIGEST_PLUS_CONVERSATION_EVIDENCE,
        )
        if historical_case is None:
            pytest.skip(
                "production-derived laxi ambient group window is unavailable"
            )
        report = await _run_case_to_l2d(
            historical_case,
            evidence_mode=(
                _LAXI_MODE_EXPANDED_DIGEST_PLUS_CONVERSATION_EVIDENCE
            ),
        )
    except PyMongoError as exc:
        pytest.skip(f"MongoDB unavailable for laxi evidence test: {exc}")
    finally:
        await close_db()

    assert report["action_spec_count"] <= 3


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured cognition endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}; {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{COGNITION_LLM_BASE_URL}"
        )


async def _collect_historical_group_review_cases(
    character_profile: dict[str, Any],
) -> list[HistoricalSensitivityCase]:
    """Collect historical group windows joined to self-cognition outcomes."""

    db = await get_db()
    now = datetime.now(timezone.utc)
    start_at = now - timedelta(hours=_lookback_hours())
    cursor = (
        db.event_log_events
        .find(
            {
                "event_family": "self_cognition",
                "event_type": models.TRIGGER_GROUP_CHAT_REVIEW,
                "occurred_at": {"$gte": start_at.isoformat()},
                "payload.case_id": {"$regex": "^group_activity_window:"},
            },
            {
                "_id": 0,
                "event_id": 1,
                "occurred_at": 1,
                "status": 1,
                "run_id": 1,
                "trigger_id": 1,
                "attempt_id": 1,
                "payload": 1,
            },
        )
        .sort("occurred_at", -1)
        .limit(_EVENT_SCAN_LIMIT)
    )
    event_rows = await cursor.to_list(length=_EVENT_SCAN_LIMIT)
    collected: list[HistoricalSensitivityCase] = []
    seen_case_ids: set[str] = set()
    for event in event_rows:
        payload = event["payload"]
        case_id = str(payload["case_id"])
        if case_id in seen_case_ids:
            continue
        parsed_source = _parse_group_activity_case_id(case_id)
        if parsed_source is None:
            continue
        raw_case = await _case_from_event(
            event,
            parsed_source=parsed_source,
            character_profile=character_profile,
        )
        if raw_case is None:
            continue
        collected.append(raw_case)
        seen_case_ids.add(case_id)

    if os.environ.get(_CASE_ID_ENV):
        return_value = collected
        return return_value

    balanced_cases = _balanced_case_prefix(collected)
    return balanced_cases


async def _case_from_event(
    event: dict[str, Any],
    *,
    parsed_source: dict[str, str],
    character_profile: dict[str, Any],
) -> HistoricalSensitivityCase | None:
    """Rebuild one self-cognition case from the raw conversation window."""

    messages = await _messages_for_window(parsed_source)
    if not messages:
        return_value = None
        return return_value

    scope = _scope_from_messages(messages, parsed_source["scope_ref"])
    if scope is None:
        return_value = None
        return return_value

    window_start = parse_storage_utc_datetime(parsed_source["window_start"])
    window_end = parse_storage_utc_datetime(parsed_source["window_end"])
    occurred_at = parse_storage_utc_datetime(str(event["occurred_at"]))
    windows = build_group_activity_windows(
        scope=scope,
        window_start=window_start,
        window_end=window_end,
        now=occurred_at,
        character_global_user_id=str(
            character_profile.get("global_user_id", "") or ""
        ),
        platform_bot_id=str(character_profile.get("platform_bot_id", "") or ""),
    )
    if not windows:
        return_value = None
        return return_value

    window = windows[0]
    case = sources._build_group_review_case(
        window,
        character_profile=character_profile,
        now=occurred_at,
    )
    historical_case = HistoricalSensitivityCase(
        case=case,
        event=event,
        historical_expected_speak=_historical_expected_speak(event),
        raw_window={
            "source_id": window.source_id,
            "window_start": parsed_source["window_start"],
            "window_end": parsed_source["window_end"],
            "messages": messages,
            "source_refs": window.source_refs,
            "semantic_labels": window.semantic_labels,
            "visible_context": window.visible_context,
        },
    )
    return historical_case


async def _messages_for_window(parsed_source: dict[str, str]) -> list[dict[str, Any]]:
    """Read raw conversation rows for one historical group window."""

    db = await get_db()
    cursor = (
        db.conversation_history
        .find(
            {
                "timestamp": {
                    "$gte": parsed_source["window_start"],
                    "$lt": parsed_source["window_end"],
                },
                "role": {"$in": ["assistant", "user"]},
                "channel_type": "group",
            },
            {
                "_id": 0,
                "platform": 1,
                "platform_channel_id": 1,
                "channel_type": 1,
                "role": 1,
                "platform_user_id": 1,
                "global_user_id": 1,
                "display_name": 1,
                "body_text": 1,
                "timestamp": 1,
                "platform_message_id": 1,
                "message_id": 1,
                "addressed_to_global_user_ids": 1,
                "mentions": 1,
                "is_directed_at_character": 1,
                "reply_context": 1,
            },
        )
        .sort("timestamp", 1)
    )
    rows = await cursor.to_list(length=None)
    matching_rows = [
        row for row in rows
        if _row_scope_ref(row) == parsed_source["scope_ref"]
    ]
    return matching_rows


async def _messages_for_scope_until(
    parsed_source: dict[str, str],
    *,
    start_at: datetime,
    end_at: datetime,
) -> list[dict[str, Any]]:
    """Read same-scope group rows up to a selected historical cutoff."""

    db = await get_db()
    cursor = (
        db.conversation_history
        .find(
            {
                "timestamp": {
                    "$gte": start_at.isoformat(),
                    "$lt": end_at.isoformat(),
                },
                "role": {"$in": ["assistant", "user"]},
                "channel_type": "group",
            },
            {
                "_id": 0,
                "platform": 1,
                "platform_channel_id": 1,
                "channel_type": 1,
                "role": 1,
                "platform_user_id": 1,
                "global_user_id": 1,
                "display_name": 1,
                "body_text": 1,
                "timestamp": 1,
                "platform_message_id": 1,
                "message_id": 1,
                "addressed_to_global_user_ids": 1,
                "mentions": 1,
                "is_directed_at_character": 1,
                "reply_context": 1,
            },
        )
        .sort("timestamp", 1)
    )
    rows = await cursor.to_list(length=None)
    matching_rows = [
        row for row in rows
        if _row_scope_ref(row) == parsed_source["scope_ref"]
    ]
    return matching_rows


def _scope_from_messages(
    messages: list[dict[str, Any]],
    scope_ref: str,
) -> ReflectionScopeInput | None:
    """Build a reflection scope when all messages belong to one group."""

    first_message = messages[0]
    platform = str(first_message["platform"])
    platform_channel_id = str(first_message["platform_channel_id"])
    channel_type = str(first_message["channel_type"])
    if build_scope_ref(platform, platform_channel_id, channel_type) != scope_ref:
        return_value = None
        return return_value

    assistant_count = sum(
        1 for message in messages
        if message.get("role") == "assistant"
    )
    user_count = sum(
        1 for message in messages
        if message.get("role") == "user"
    )
    scope = ReflectionScopeInput(
        scope_ref=scope_ref,
        platform=platform,
        platform_channel_id=platform_channel_id,
        channel_type=channel_type,
        assistant_message_count=assistant_count,
        user_message_count=user_count,
        total_message_count=len(messages),
        first_timestamp=str(messages[0]["timestamp"]),
        last_timestamp=str(messages[-1]["timestamp"]),
        messages=messages,
    )
    return scope


async def _run_case_to_l2d(
    historical_case: HistoricalSensitivityCase,
    *,
    evidence_mode: str = "",
) -> dict[str, Any]:
    """Run a rebuilt case through the shared cognition subgraph before dialog."""

    source_packet = projection.build_source_packet(historical_case.case)
    rendered_packet = projection.render_source_packet_text(source_packet)
    cognition_state = runner._build_cognition_state(
        historical_case.case,
        rendered_packet,
    )
    l2d_state, stage_outputs = await _run_cognition_stages_before_dialog(
        cognition_state,
    )
    l2d_prompt_payload = build_action_selection_payload_text(l2d_state)
    l2d_output = await select_semantic_actions(l2d_state)
    action_specs = l2d_output["action_specs"]
    observed_speak = _observed_user_visible_speak(action_specs)
    speak_reasons = _observed_user_visible_speak_reasons(action_specs)
    matches_historical = observed_speak == (
        historical_case.historical_expected_speak
    )
    case_id = historical_case.case["case_id"]
    trace_name = "self_cognition_group_response_sensitivity_live_llm"
    if evidence_mode:
        trace_name = f"{trace_name}_{evidence_mode}"
    trace_path = write_llm_trace(
        trace_name,
        case_id,
        {
            "case_id": case_id,
            "evidence_mode": evidence_mode,
            "slot": {
                "window_start": historical_case.raw_window["window_start"],
                "window_end": historical_case.raw_window["window_end"],
                "event_occurred_at": historical_case.event["occurred_at"],
            },
            "historical_expected_speak": (
                historical_case.historical_expected_speak
            ),
            "observed_user_visible_speak": observed_speak,
            "observed_user_visible_speak_reasons": speak_reasons,
            "matches_historical": matches_historical,
            "raw_window": historical_case.raw_window,
            "source_packet": source_packet,
            "rendered_packet": rendered_packet,
            "l2d_prompt_payload": l2d_prompt_payload,
            "l2d_output": l2d_output,
            "parsed_action_specs": action_specs,
            "cognition_output_before_dialog": {
                **stage_outputs,
                **l2d_output,
            },
            "pre_l2d_state": l2d_state,
            "historical_event": historical_case.event,
            "judgment": (
                "manual_review_required_for_self_cognition_response_sensitivity"
            ),
        },
    )
    report = {
        "case_id": case_id,
        "historical_expected_speak": historical_case.historical_expected_speak,
        "observed_user_visible_speak": observed_speak,
        "observed_user_visible_speak_reasons": speak_reasons,
        "matches_historical": matches_historical,
        "action_spec_count": len(action_specs),
        "evidence_mode": evidence_mode,
        "trace_path": str(trace_path),
    }
    logger.info(
        "SELF_COGNITION_SENSITIVITY_CASE "
        f"{json.dumps(report, ensure_ascii=True)}"
    )
    return report


async def _run_cognition_stages_before_dialog(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run L1/L2 stages and return the state immediately before L2d."""

    initial_state = _cognition_initial_state(state)
    l1_output = await call_cognition_subconscious(initial_state)
    after_l1 = {**initial_state, **l1_output}

    l2a_output = await call_cognition_consciousness(after_l1)
    l2b_output = await call_boundary_core_agent(after_l1)
    after_l2b = {**after_l1, **l2b_output}

    after_l2a_l2b = {**after_l1, **l2a_output, **l2b_output}
    l2c1_output = await call_judgment_core_agent(after_l2a_l2b)
    l2c2_output = await call_social_context_appraisal(after_l2b)

    before_group_engagement = {
        **after_l2a_l2b,
        **l2c1_output,
        **l2c2_output,
    }
    group_engagement_output = await call_group_engagement_action_context_loader(
        before_group_engagement,
    )
    l2d_state = {
        **before_group_engagement,
        **group_engagement_output,
    }
    stage_outputs = {
        "l1_subconscious": l1_output,
        "l2a_conscious_framing": l2a_output,
        "l2b_boundary_appraisal": l2b_output,
        "l2c1_judgment_synthesis": l2c1_output,
        "l2c2_social_context_appraisal": l2c2_output,
        "group_engagement_action_context": group_engagement_output,
    }
    return_value = (l2d_state, stage_outputs)
    return return_value


def _cognition_initial_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build the same cognition entry state used by the shared subgraph."""

    interaction_history_recent = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )
    initial_state = {
        "character_profile": state["character_profile"],
        "storage_timestamp_utc": state["storage_timestamp_utc"],
        "local_time_context": state["local_time_context"],
        "user_input": state["user_input"],
        "prompt_message_context": state["prompt_message_context"],
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "channel_type": state["channel_type"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "platform_bot_id": state["platform_bot_id"],
        "chat_history_recent": interaction_history_recent,
        "reply_context": state["reply_context"],
        "indirect_speech_context": state["indirect_speech_context"],
        "channel_topic": state["channel_topic"],
        "conversation_progress": state.get("conversation_progress"),
        "promoted_reflection_context": state.get("promoted_reflection_context"),
        "decontexualized_input": state["decontexualized_input"],
        "referents": state["referents"],
        "rag_result": state["rag_result"],
    }
    cognitive_episode = state.get("cognitive_episode")
    if cognitive_episode is not None:
        initial_state["cognitive_episode"] = cognitive_episode
    return initial_state


def _write_dataset_trace(
    cases: list[HistoricalSensitivityCase],
    selected_cases: list[HistoricalSensitivityCase],
) -> Any:
    """Write the collected historical dataset for tuning inspection."""

    selected_case_ids = [
        historical_case.case["case_id"]
        for historical_case in selected_cases
    ]
    payload = {
        "case_count": len(cases),
        "minimum_required_cases": _MIN_CASES,
        "selected_case_ids": selected_case_ids,
        "historical_silent_count": sum(
            1 for historical_case in cases
            if not historical_case.historical_expected_speak
        ),
        "historical_spoke_count": sum(
            1 for historical_case in cases
            if historical_case.historical_expected_speak
        ),
        "cases": [
            {
                "case_id": historical_case.case["case_id"],
                "slot": {
                    "window_start": historical_case.raw_window["window_start"],
                    "window_end": historical_case.raw_window["window_end"],
                    "event_occurred_at": historical_case.event["occurred_at"],
                },
                "historical_expected_speak": (
                    historical_case.historical_expected_speak
                ),
                "raw_window": historical_case.raw_window,
                "source_refs": historical_case.case["source_refs"],
                "group_activity_window": (
                    historical_case.case["group_activity_window"]
                ),
                "historical_event": historical_case.event,
            }
            for historical_case in cases
        ],
    }
    trace_path = write_llm_trace(
        "self_cognition_group_response_sensitivity_dataset",
        "collected_group_windows",
        payload,
    )
    return trace_path


def _assert_case_mix(cases: list[HistoricalSensitivityCase]) -> None:
    """Assert the tuning dataset has enough mixed historical labels."""

    assert len(cases) >= _MIN_CASES
    assert any(
        historical_case.historical_expected_speak
        for historical_case in cases
    )
    assert any(
        not historical_case.historical_expected_speak
        for historical_case in cases
    )


def _selected_cases_for_run(
    cases: list[HistoricalSensitivityCase],
) -> list[HistoricalSensitivityCase]:
    """Select one case by default, or all collected cases for explicit tuning."""

    if os.environ.get(_RUN_BATCH_ENV) == "1":
        return_value = cases
        return return_value

    selected_case_id = os.environ.get(_CASE_ID_ENV)
    if selected_case_id:
        for historical_case in cases:
            if historical_case.case["case_id"] == selected_case_id:
                return_value = [historical_case]
                return return_value
        raise ValueError(f"{_CASE_ID_ENV} not found: {selected_case_id}")

    return_value = [cases[0]]
    return return_value


def _balanced_case_prefix(
    cases: list[HistoricalSensitivityCase],
) -> list[HistoricalSensitivityCase]:
    """Return at least the required cases while preserving label balance."""

    silent_cases = [
        historical_case for historical_case in cases
        if not historical_case.historical_expected_speak
    ]
    spoke_cases = [
        historical_case for historical_case in cases
        if historical_case.historical_expected_speak
    ]
    if len(silent_cases) + len(spoke_cases) < _MIN_CASES:
        return_value = cases
        return return_value

    balanced_cases: list[HistoricalSensitivityCase] = []
    for index in range(max(len(silent_cases), len(spoke_cases))):
        if index < len(silent_cases):
            balanced_cases.append(silent_cases[index])
        if index < len(spoke_cases):
            balanced_cases.append(spoke_cases[index])
        if len(balanced_cases) >= _MIN_CASES:
            break
    return balanced_cases


async def _ambient_group_leak_case(
    character_profile: dict[str, Any],
) -> HistoricalSensitivityCase | None:
    """Rebuild the production-derived ambient group window that leaked text."""

    parsed_source = _parse_group_activity_case_id(_AMBIENT_GROUP_LEAK_CASE_ID)
    if parsed_source is None:
        return_value = None
        return return_value

    messages = await _messages_for_window(parsed_source)
    if not messages:
        return_value = None
        return return_value

    scope = _scope_from_messages(messages, parsed_source["scope_ref"])
    if scope is None:
        return_value = None
        return return_value

    window_start = parse_storage_utc_datetime(parsed_source["window_start"])
    window_end = parse_storage_utc_datetime(parsed_source["window_end"])
    reviewed_at = parse_storage_utc_datetime(_AMBIENT_GROUP_LEAK_REVIEWED_AT)
    windows = build_group_activity_windows(
        scope=scope,
        window_start=window_start,
        window_end=window_end,
        now=reviewed_at,
        character_global_user_id=str(
            character_profile.get("global_user_id", "") or ""
        ),
        platform_bot_id=str(character_profile.get("platform_bot_id", "") or ""),
    )
    if not windows:
        return_value = None
        return return_value

    cases = await sources.collect_group_review_cases(
        now=reviewed_at,
        character_profile=character_profile,
        windows=[windows[0]],
        max_cases=1,
    )
    if not cases:
        return_value = None
        return return_value

    event = {
        "event_id": "production_row:delivery_tracking:0c3046964dde42209b43809cbe63fb6b",
        "occurred_at": _AMBIENT_GROUP_LEAK_REVIEWED_AT,
        "payload": {
            "case_id": _AMBIENT_GROUP_LEAK_CASE_ID,
            "selected_route": models.ROUTE_ACTION_CANDIDATE,
            "output_mode": "visible_reply",
            "observed_assistant_text": _AMBIENT_GROUP_LEAK_ASSISTANT_TEXT,
        },
    }
    historical_case = HistoricalSensitivityCase(
        case=cases[0],
        event=event,
        historical_expected_speak=True,
        raw_window={
            "source_id": windows[0].source_id,
            "window_start": parsed_source["window_start"],
            "window_end": parsed_source["window_end"],
            "messages": messages,
            "source_refs": windows[0].source_refs,
            "semantic_labels": windows[0].semantic_labels,
            "visible_context": windows[0].visible_context,
            "production_assistant_text": _AMBIENT_GROUP_LEAK_ASSISTANT_TEXT,
        },
    )
    return historical_case


async def _laxi_ambient_group_case(
    character_profile: dict[str, Any],
    *,
    digest_mode: str,
) -> HistoricalSensitivityCase | None:
    """Rebuild the production-derived ambient health-joke group window."""

    parsed_source = _parse_group_activity_case_id(_LAXI_AMBIENT_GROUP_CASE_ID)
    if parsed_source is None:
        return_value = None
        return return_value

    window_start = parse_storage_utc_datetime(parsed_source["window_start"])
    window_end = parse_storage_utc_datetime(parsed_source["window_end"])
    source_cutoff = parse_storage_utc_datetime(
        _LAXI_AMBIENT_GROUP_SOURCE_CUTOFF,
    )
    if digest_mode == _LAXI_MODE_BASELINE_15M_DIGEST:
        window_messages = await _messages_for_window(parsed_source)
        messages = [
            message
            for message in window_messages
            if parse_storage_utc_datetime(str(message["timestamp"]))
            < source_cutoff
        ]
        build_start = window_start
    elif digest_mode == _LAXI_MODE_EXPANDED_DIGEST_ONLY:
        messages = await _messages_for_scope_until(
            parsed_source,
            start_at=window_start - timedelta(hours=3),
            end_at=source_cutoff,
        )
    elif digest_mode == _LAXI_MODE_EXPANDED_DIGEST_PLUS_CONVERSATION_EVIDENCE:
        messages = await _messages_for_scope_until(
            parsed_source,
            start_at=window_start - timedelta(hours=3),
            end_at=source_cutoff,
        )
    else:
        raise ValueError(f"unsupported laxi digest mode: {digest_mode}")

    if not messages:
        return_value = None
        return return_value
    if digest_mode in {
        _LAXI_MODE_EXPANDED_DIGEST_ONLY,
        _LAXI_MODE_EXPANDED_DIGEST_PLUS_CONVERSATION_EVIDENCE,
    }:
        build_start = parse_storage_utc_datetime(str(messages[0]["timestamp"]))

    scope = _scope_from_messages(messages, parsed_source["scope_ref"])
    if scope is None:
        return_value = None
        return return_value

    reviewed_at = parse_storage_utc_datetime(_LAXI_AMBIENT_GROUP_REVIEWED_AT)
    windows = build_group_activity_windows(
        scope=scope,
        window_start=build_start,
        window_end=window_end,
        now=reviewed_at,
        character_global_user_id=str(
            character_profile.get("global_user_id", "") or ""
        ),
        platform_bot_id=str(character_profile.get("platform_bot_id", "") or ""),
    )
    if not windows:
        return_value = None
        return return_value

    selected_window = _matching_window(windows, parsed_source)
    if selected_window is None:
        return_value = None
        return return_value

    if digest_mode == _LAXI_MODE_EXPANDED_DIGEST_PLUS_CONVERSATION_EVIDENCE:
        conversation_evidence_builder = None
    else:
        conversation_evidence_builder = _no_summary_conversation_evidence
    cases = await sources.collect_group_review_cases(
        now=reviewed_at,
        character_profile=character_profile,
        windows=[selected_window],
        max_cases=1,
        conversation_evidence_builder=conversation_evidence_builder,
    )
    if not cases:
        return_value = None
        return return_value

    event = {
        "event_id": (
            "production_row:delivery_tracking:"
            "0fc15b7e4abd4fcd979de4c8971d6378"
        ),
        "occurred_at": _LAXI_AMBIENT_GROUP_REVIEWED_AT,
        "payload": {
            "case_id": _LAXI_AMBIENT_GROUP_CASE_ID,
            "selected_route": models.ROUTE_ACTION_CANDIDATE,
            "output_mode": "visible_reply",
            "observed_assistant_text": _LAXI_AMBIENT_GROUP_ASSISTANT_TEXT,
        },
    }
    historical_case = HistoricalSensitivityCase(
        case=cases[0],
        event=event,
        historical_expected_speak=True,
        raw_window={
            "source_id": selected_window.source_id,
            "window_start": parsed_source["window_start"],
            "window_end": parsed_source["window_end"],
            "evidence_mode": digest_mode,
            "messages": messages,
            "source_refs": selected_window.source_refs,
            "semantic_labels": selected_window.semantic_labels,
            "visible_context": selected_window.visible_context,
            "digest_participant_rows": selected_window.digest_participant_rows,
            "production_assistant_text": _LAXI_AMBIENT_GROUP_ASSISTANT_TEXT,
        },
    )
    return historical_case


async def _no_summary_conversation_evidence(**kwargs: Any) -> list[str]:
    """Disable pre-cognition evidence for comparison modes."""

    del kwargs
    evidence: list[str] = []
    return evidence


def _matching_window(
    windows: list[Any],
    parsed_source: dict[str, str],
) -> Any | None:
    """Return the activity window that matches the historical source id."""

    expected_source_id = (
        f"{parsed_source['scope_ref']}:"
        f"{parsed_source['window_start']}:"
        f"{parsed_source['window_end']}"
    )
    for window in windows:
        if window.source_id == expected_source_id:
            return_value = window
            return return_value
    return_value = None
    return return_value


def _parse_group_activity_case_id(case_id: str) -> dict[str, str] | None:
    """Parse the stable group activity source id from a historical case id."""

    prefix = "group_activity_window:"
    if not case_id.startswith(prefix):
        return_value = None
        return return_value
    source_id = case_id[len(prefix):]
    match = _SOURCE_ID_PATTERN.match(source_id)
    if match is None:
        return_value = None
        return return_value
    parsed = {
        "scope_ref": match.group(1),
        "window_start": match.group(2),
        "window_end": match.group(3),
    }
    return parsed


def _historical_expected_speak(event: dict[str, Any]) -> bool:
    """Return whether the historical self-cognition run selected speech."""

    payload = event["payload"]
    expected = (
        payload["selected_route"] == models.ROUTE_ACTION_CANDIDATE
        or payload["output_mode"] != "silent"
    )
    return expected


def _observed_user_visible_speak(action_specs: list[dict[str, Any]]) -> bool:
    """Return whether L2d selected a user-visible speak surface."""

    for action_spec in action_specs:
        if action_spec.get("kind") != "speak":
            continue
        if action_spec.get("visibility") == "user_visible":
            return_value = True
            return return_value
    return_value = False
    return return_value


def _observed_user_visible_speak_reasons(
    action_specs: list[dict[str, Any]],
) -> list[str]:
    """Return prompt-facing reasons for user-visible speak selections."""

    reasons: list[str] = []
    for action_spec in action_specs:
        if action_spec.get("kind") != "speak":
            continue
        if action_spec.get("visibility") != "user_visible":
            continue
        reason = action_spec.get("reason")
        if isinstance(reason, str) and reason.strip():
            reasons.append(reason.strip())
    return reasons


def _row_scope_ref(row: dict[str, Any]) -> str:
    """Return the stable reflection scope ref for one conversation row."""

    scope_ref = build_scope_ref(
        str(row["platform"]),
        str(row["platform_channel_id"]),
        str(row["channel_type"]),
    )
    return scope_ref


def _lookback_hours() -> int:
    """Return the event-log lookback window for historical case collection."""

    raw_value = os.environ.get(_LOOKBACK_HOURS_ENV)
    if raw_value is None:
        return_value = _DEFAULT_LOOKBACK_HOURS
        return return_value
    value = int(raw_value)
    return value
