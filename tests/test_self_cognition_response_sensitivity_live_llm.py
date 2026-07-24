"""Live LLM sensitivity checks for self-cognition group response routing."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.db import close_db, get_character_profile
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.interaction_style_images import (
    build_group_engagement_action_context,
)
from kazusa_ai_chatbot.reflection_cycle.activity_windows import (
    build_group_activity_windows,
)
from kazusa_ai_chatbot.reflection_cycle.models import ReflectionScopeInput
from kazusa_ai_chatbot.reflection_cycle.selector import build_scope_ref
from kazusa_ai_chatbot.self_cognition import models, projection, runner, sources
from kazusa_ai_chatbot.self_cognition.group_review_participant_context import (
    build_group_review_thread_reference_context,
)
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
_ROOT = Path(__file__).resolve().parents[1]
_CAT_FAILURE_TRACE_PATH = (
    _ROOT
    / "test_artifacts"
    / "llm_traces"
    / "kazusa_cat_failure_self_cognition_repro.json"
)
_CAT_SUBJECT_INVERSION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r'把我.{0,12}比作',
        r'我.{0,12}被.{0,8}比作',
        r'我的头发',
        r'我.{0,16}暖气片旁边的猫',
        r'灯.{0,20}把我.{0,12}比作',
        r'把(?:千纱|杏山千纱).{0,12}比作',
        r'(?:千纱|杏山千纱).{0,12}被.{0,8}比作',
        r'(?:千纱|杏山千纱)的头发',
        r'(?:千纱|杏山千纱).{0,6}(?:像|如同).{0,16}猫',
    )
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


async def test_live_self_cognition_cat_side_thread_subject_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay the cat side-thread failure through cognition, L3, and dialog."""

    await _skip_if_llm_unavailable()
    source_trace = json.loads(
        _CAT_FAILURE_TRACE_PATH.read_text(encoding="utf-8"),
    )
    case = _cat_case_with_current_subject_boundary_contract(
        source_trace["case"],
    )
    prior_residue = source_trace["artifacts"][
        "self_cognition_cognition_output.json"
    ].get("internal_monologue_residue_context", "")
    captured_dialog_inputs: list[dict[str, Any]] = []

    async def load_repro_residue(_case: dict[str, Any]) -> str:
        del _case
        return str(prior_residue)

    async def tracing_dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_dialog_inputs.append(_trace_safe_dialog_input(state))
        dialog_output = await runner._default_dialog_client(state)
        return dialog_output

    monkeypatch.setattr(
        runner,
        "_load_residue_context_for_case",
        load_repro_residue,
    )

    artifacts = await runner.build_self_cognition_case_artifacts_async(
        case,
        dialog_client=tracing_dialog_client,
    )
    run_record = artifacts[models.ARTIFACT_RUN_RECORD]
    cognition_output = artifacts[models.ARTIFACT_COGNITION_OUTPUT]
    action_candidate = artifacts.get(models.ARTIFACT_ACTION_CANDIDATE, {})
    diagnostics = _cat_subject_boundary_diagnostics(
        cognition_output=cognition_output,
        captured_dialog_inputs=captured_dialog_inputs,
        action_candidate=action_candidate,
    )
    trace_path = write_llm_trace(
        "self_cognition_cat_side_thread_subject_boundary_live_llm",
        case["case_id"],
        {
            "case_id": case["case_id"],
            "source_fixture": str(_CAT_FAILURE_TRACE_PATH),
            "source_contract_adjustment": (
                "Replayed with current neutral digest and "
                "conversation_progress.thread_reference_context instead of "
                "the stale first-person digest stored in the reproduction "
                "trace."
            ),
            "manual_review_required": True,
            "manual_review_guidance": (
                "Inspect rendered source packet, L2 internal monologue, "
                "boundary assessment, L2d action requests, L3 content plan, "
                "and final dialog. Pass only if no inspected stage claims "
                "Lamp compared Kazusa to a cat, Kazusa's hair was described, "
                "or the side-thread `你` definitely refers to Kazusa."
            ),
            "selected_route": run_record["selected_route"],
            "source_packet": artifacts[models.ARTIFACT_COGNITION_INPUT][
                "source_packet"
            ],
            "rendered_packet": artifacts[models.ARTIFACT_COGNITION_INPUT][
                "rendered_text"
            ],
            "cognition_output": cognition_output,
            "captured_dialog_inputs": captured_dialog_inputs,
            "action_candidate": action_candidate,
            "run_record": run_record,
            "route_effect": artifacts[models.ARTIFACT_ROUTE_EFFECT],
            "subject_boundary_diagnostics": diagnostics,
        },
    )

    assert trace_path.exists()
    assert diagnostics["action_spec_count"] <= 3
    assert diagnostics["high_confidence_subject_inversion_hits"] == [], (
        "cat side-thread subject inversion still appears in generated fields; "
        f"trace={trace_path}"
    )


def _cat_case_with_current_subject_boundary_contract(
    raw_case: dict[str, Any],
) -> dict[str, Any]:
    """Replay the historical case through the current source contract.

    The reproduction trace intentionally stores the old bad source packet.  For
    after-fix live validation, keep the same surrounding visible messages and
    bad prior residue, but replace stale digest/evidence fields with the
    current neutral thread-boundary source shape.
    """

    case = deepcopy(raw_case)
    visible_rows = [
        dict(row)
        for row in case.get("visible_context", [])
        if isinstance(row, dict)
    ]
    character_profile = case.get("character_profile")
    if not isinstance(character_profile, dict):
        character_profile = {}
    thread_reference_context = build_group_review_thread_reference_context(
        visible_rows,
        character_profile,
    )
    assert thread_reference_context is not None

    conversation_progress = case.get("conversation_progress")
    if not isinstance(conversation_progress, dict):
        conversation_progress = {}
    else:
        conversation_progress = dict(conversation_progress)
    conversation_progress["thread_reference_context"] = (
        thread_reference_context
    )
    conversation_progress["group_scene_digest"] = {
        "digest": (
            "雪凪先 @杏山千纱 发猪头表情，随后 @灯（23岁）说“摸摸大姐姐”。"
            "灯（23岁）回应“摸到了”，又说“你的头发软软的，"
            "像rana家那只靠在暖气片旁边的猫”；这条二人称头发描述"
            "没有在同一行指向杏山千纱，按雪凪、灯（23岁）和被摸头对象"
            "之间的侧线处理。杏山千纱随后质问前面的暗示和挑衅；"
            "当前角色可见发言后没有新的文字线索。"
        ),
    }
    conversation_progress["summary"] = (
        "雪凪先点到杏山千纱，随后把摸头话题转给灯（23岁）；"
        "灯（23岁）的二人称头发描述属于侧线，杏山千纱随后质问暗示。"
    )
    conversation_progress.pop("conversation_evidence", None)
    case["conversation_progress"] = conversation_progress
    return case


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


def _trace_safe_dialog_input(state: dict[str, Any]) -> dict[str, Any]:
    """Keep the L3/dialog input fields needed for manual live review."""

    action_directives = state.get("action_directives")
    if not isinstance(action_directives, dict):
        action_directives = {}
    return {
        "internal_monologue": state.get("internal_monologue", ""),
        "boundary_core_assessment": state.get("boundary_core_assessment", {}),
        "judgment_note": state.get("judgment_note", ""),
        "action_specs": state.get("action_specs", []),
        "action_directives": action_directives,
        "logical_stance": state.get("logical_stance", ""),
        "character_intent": state.get("character_intent", ""),
    }


def _cat_subject_boundary_diagnostics(
    *,
    cognition_output: dict[str, Any],
    captured_dialog_inputs: list[dict[str, Any]],
    action_candidate: object,
) -> dict[str, Any]:
    """Return bounded diagnostics for the cat side-thread live regression."""

    generated_texts = _cat_generated_review_texts(
        cognition_output=cognition_output,
        captured_dialog_inputs=captured_dialog_inputs,
        action_candidate=action_candidate,
    )
    hits: list[dict[str, str]] = []
    for source_name, text in generated_texts:
        for pattern in _CAT_SUBJECT_INVERSION_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            hits.append({
                "source": source_name,
                "pattern": pattern.pattern,
                "match": match.group(0),
            })
    action_specs = cognition_output.get("action_specs")
    if not isinstance(action_specs, list):
        action_specs = []
    diagnostics = {
        "high_confidence_subject_inversion_hits": hits,
        "action_spec_count": len(action_specs),
        "reviewed_generated_field_count": len(generated_texts),
    }
    return diagnostics


def _cat_generated_review_texts(
    *,
    cognition_output: dict[str, Any],
    captured_dialog_inputs: list[dict[str, Any]],
    action_candidate: object,
) -> list[tuple[str, str]]:
    """Collect generated fields only; source transcript text is excluded."""

    texts: list[tuple[str, str]] = []
    for field_name in (
        "internal_monologue",
        "judgment_note",
        "social_distance",
        "relational_dynamic",
    ):
        value = cognition_output.get(field_name)
        if isinstance(value, str) and value:
            texts.append((field_name, value))

    boundary = cognition_output.get("boundary_core_assessment")
    if isinstance(boundary, dict):
        for field_name in ("boundary_summary", "trajectory"):
            value = boundary.get(field_name)
            if isinstance(value, str) and value:
                texts.append((f"boundary_core_assessment.{field_name}", value))

    action_specs = cognition_output.get("action_specs")
    if isinstance(action_specs, list):
        for index, action_spec in enumerate(action_specs):
            if not isinstance(action_spec, dict):
                continue
            reason = action_spec.get("reason")
            if isinstance(reason, str) and reason:
                texts.append((f"action_specs[{index}].reason", reason))
            params = action_spec.get("params")
            if isinstance(params, dict):
                requirements = params.get("surface_requirements")
                if isinstance(requirements, dict):
                    detail = requirements.get("detail")
                    if isinstance(detail, str) and detail:
                        texts.append((
                            f"action_specs[{index}].surface_detail",
                            detail,
                        ))

    for index, dialog_input in enumerate(captured_dialog_inputs):
        action_directives = dialog_input.get("action_directives")
        if not isinstance(action_directives, dict):
            continue
        serialized_directives = json.dumps(
            action_directives,
            ensure_ascii=False,
            sort_keys=True,
        )
        texts.append((
            f"dialog_input[{index}].action_directives",
            serialized_directives,
        ))

    if isinstance(action_candidate, dict):
        text = action_candidate.get("text")
        if isinstance(text, str) and text:
            texts.append(("action_candidate.text", text))

    return texts


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
    group_engagement_context = await build_group_engagement_action_context(
        channel_type=str(state["channel_type"]),
        platform=str(state["platform"]),
        platform_channel_id=str(state["platform_channel_id"]),
    )
    group_engagement_output = {
        "group_engagement_action_context": group_engagement_context,
    }
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
        "decontextualized_input": state["decontextualized_input"],
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
