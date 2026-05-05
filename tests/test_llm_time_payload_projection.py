"""Payload-level projection tests for character-local time context.

Verifies that representative LLM payloads built by the production modules
contain no raw UTC timestamps, timezone offsets, timezone names, or UTC
labels. These complement the unit-level tests in ``test_time_context.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from kazusa_ai_chatbot.rag.prompt_projection import (
    project_runtime_context_for_llm,
    project_tool_result_for_llm,
)
from kazusa_ai_chatbot.time_context import (
    build_character_time_context,
    format_history_for_llm,
)


_UTC_LEAK_RE = re.compile(
    r"[+-]\d{2}:\d{2}|"
    r"\bUTC\b|"
    r"\d{4}-\d{2}-\d{2}T|"
    r"Z(?=$|[\"\\s,}])|"
    r"Pacific/Auckland|America/|Europe/|Asia/",
)


def _assert_no_utc_leak(payload: Any, path: str = "$") -> None:
    """Recursively check that no string value in *payload* leaks UTC markers."""
    if isinstance(payload, str):
        assert not _UTC_LEAK_RE.search(payload), (
            f"UTC leak at {path}: {payload!r}"
        )
    elif isinstance(payload, dict):
        for key, value in payload.items():
            _assert_no_utc_leak(value, f"{path}.{key}")
    elif isinstance(payload, (list, tuple)):
        for idx, item in enumerate(payload):
            _assert_no_utc_leak(item, f"{path}[{idx}]")


# ---------------------------------------------------------------------------
# Representative payload builders
# ---------------------------------------------------------------------------

_TURN_TIMESTAMP = "2026-05-03T00:00:03+00:00"
_TIME_CONTEXT = build_character_time_context(_TURN_TIMESTAMP)


class _DummyResponse:
    """Small LangChain-like response wrapper for payload capture tests."""

    def __init__(self, content: str) -> None:
        self.content = content


class _CapturingAsyncLLM:
    """Capture one LLM invocation and return a fixed JSON payload."""

    def __init__(self, response_payload: dict) -> None:
        self.messages = None
        self._response_payload = response_payload

    async def ainvoke(self, messages):
        self.messages = messages
        content = json.dumps(self._response_payload, ensure_ascii=False)
        response = _DummyResponse(content)
        return response


def _fake_consolidator_state(*, time_context: dict | None = None) -> dict:
    """Minimal ConsolidatorState-like dict for payload builder tests."""
    return {
        "timestamp": _TURN_TIMESTAMP,
        "time_context": time_context or _TIME_CONTEXT,
        "global_user_id": "user-1",
        "user_name": "Tester",
        "decontexualized_input": "Test input.",
        "final_dialog": ["Test reply."],
        "internal_monologue": "",
        "emotional_appraisal": "",
        "interaction_subtext": "",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "chat_history_recent": [],
        "rag_result": {
            "user_image": {"user_memory_context": ""},
            "user_memory_unit_candidates": [],
        },
        "new_facts": [],
        "future_promises": [],
        "subjective_appraisals": [],
        "character_profile": {"name": "TestChar"},
        "action_directives": {
            "linguistic_directives": {"content_anchors": []},
        },
    }


# ---------------------------------------------------------------------------
# Facts harvester payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_facts_harvester_payload_uses_local_time(monkeypatch) -> None:
    """facts_harvester payload should contain only local time strings."""
    from kazusa_ai_chatbot.nodes import (
        persona_supervisor2_consolidator_facts as facts_module,
    )

    llm = _CapturingAsyncLLM({"new_facts": [], "future_promises": []})
    monkeypatch.setattr(facts_module, "_facts_harvester_llm", llm)
    state = _fake_consolidator_state()
    state["rag_result"] = {
        "user_image": {
            "user_memory_context": "",
            "updated_at": "2026-05-02T20:00:00+00:00",
        },
        "user_memory_unit_candidates": [
            {"timestamp": "2026-05-02T20:01:00+00:00"},
        ],
        "supervisor_trace": {
            "timestamp": "2026-05-02T20:02:00+00:00",
        },
    }

    await facts_module.facts_harvester(state)

    payload = json.loads(llm.messages[1].content)
    assert "12:00" in payload["timestamp"]
    _assert_no_utc_leak(payload, "$.facts_harvester")


# ---------------------------------------------------------------------------
# Memory unit stability payload
# ---------------------------------------------------------------------------


def test_stability_payload_timestamps_are_local() -> None:
    """Stability judge payload should contain only local timestamps."""
    from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_memory_units import (
        _stability_payload,
    )

    state = _fake_consolidator_state()
    candidate = {
        "candidate_id": "c1",
        "evidence_refs": [
            {"timestamp": "2026-05-02T20:00:00+00:00", "message_id": "m1"},
        ],
    }
    cluster = {
        "unit_id": "u1",
        "count": 2,
        "updated_at": "2026-04-30T10:00:00+00:00",
        "last_seen_at": "2026-05-01T08:00:00+00:00",
        "source_refs": [
            {"timestamp": "2026-04-30T10:00:00+00:00", "message_id": "m0"},
        ],
        "fact": "Tester likes trains.",
    }
    merge_result = {
        "decision": "merge",
        "cluster_id": "u1",
        "reason": "same topic",
    }

    payload = _stability_payload(
        state,
        unit_id="u1",
        candidate=candidate,
        merge_result=merge_result,
        candidate_clusters=[cluster],
    )

    recency = payload["stability_evidence"]["recency"]
    _assert_no_utc_leak(recency, "$.stability_evidence.recency")

    spread = payload["stability_evidence"]["session_spread"]
    for ts in spread.get("timestamps", []):
        _assert_no_utc_leak(ts, "$.session_spread.timestamps[]")

    recent_examples = payload["stability_evidence"]["recent_examples"]
    for ex in recent_examples:
        _assert_no_utc_leak(
            ex.get("updated_at", ""),
            "$.recent_examples[].updated_at",
        )


# ---------------------------------------------------------------------------
# History projection round-trip
# ---------------------------------------------------------------------------


def test_formatted_history_has_no_utc_leak() -> None:
    """chat_history_recent after projection should contain only local times."""
    raw_history = [
        {
            "role": "user",
            "content": "hello",
            "timestamp": "2026-05-02T20:30:00+00:00",
        },
        {
            "role": "assistant",
            "content": "hi",
            "timestamp": "2026-05-02T20:31:00+00:00",
        },
    ]
    formatted = format_history_for_llm(raw_history)
    for entry in formatted:
        _assert_no_utc_leak(entry, "$.chat_history_recent[]")


# ---------------------------------------------------------------------------
# Runtime context projection
# ---------------------------------------------------------------------------


def test_runtime_context_projects_episode_state_times() -> None:
    """RAG runtime context should not raw-copy stored episode timestamps."""
    context = {
        "current_timestamp": "2026-05-02T20:00:00+00:00",
        "time_context": _TIME_CONTEXT,
        "conversation_episode_state": {
            "status": "active",
            "created_at": "2026-05-02T20:00:00+00:00",
            "updated_at": "2026-05-02T20:05:00+00:00",
            "expires_at": "2026-05-03T20:00:00+00:00",
            "user_state_updates": [
                {
                    "text": "Tester is planning dinner.",
                    "first_seen_at": "2026-05-02T20:01:00+00:00",
                },
            ],
        },
        "conversation_progress": {
            "status": "active",
            "open_loops": [{"text": "Pick dinner.", "age_hint": "just now"}],
        },
    }

    projected = project_runtime_context_for_llm(context)

    assert "current_timestamp" not in projected
    episode_state = projected["conversation_episode_state"]
    assert episode_state["created_at"] == "2026-05-03 08:00"
    assert episode_state["updated_at"] == "2026-05-03 08:05"
    assert episode_state["expires_at"] == "2026-05-04 08:00"
    assert (
        episode_state["user_state_updates"][0]["first_seen_at"]
        == "2026-05-03 08:01"
    )
    _assert_no_utc_leak(projected, "$.runtime_context")


def test_tool_result_projects_character_recent_window_times() -> None:
    """Character-image recent-window rows should use local prompt time."""
    result = {
        "character_image": {
            "self_image": {
                "recent_window": [
                    {
                        "timestamp": "2026-05-02T20:00:00+00:00",
                        "summary": "Character remained calm.",
                    },
                ],
            },
        },
    }

    projected = project_tool_result_for_llm(result)

    recent_window = projected["character_image"]["self_image"]["recent_window"]
    assert recent_window[0]["timestamp"] == "2026-05-03 08:00"
    _assert_no_utc_leak(projected, "$.character_image")


# ---------------------------------------------------------------------------
# Runtime-backed live-context output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_context_runtime_result_has_no_utc_leak() -> None:
    """Runtime-backed live-context output should expose only sanitized local strings."""
    from kazusa_ai_chatbot.rag.live_context_agent import LiveContextAgent

    agent = LiveContextAgent()
    result = await agent.run(
        "Live-context: answer active character current local time",
        {"time_context": _TIME_CONTEXT},
    )

    assert result["resolved"] is True
    assert result["result"]["primary_worker"] == "runtime_context_provider"
    assert result["result"]["projection_payload"]["url"] == ""
    _assert_no_utc_leak(result, "$.live_context_runtime_result")


# ---------------------------------------------------------------------------
# Normalize future promises converts local due_time to UTC
# ---------------------------------------------------------------------------


def test_normalize_future_promises_converts_local_due_time() -> None:
    """_normalize_future_promises should convert local YYYY-MM-DD HH:MM to UTC ISO."""
    from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_persistence import (
        _normalize_future_promises,
    )

    promises = [
        {
            "target": "Tester",
            "action": "send reminder",
            "due_time": "2026-05-04 14:00",
            "commitment_type": "future_promise",
        },
    ]
    normalized = _normalize_future_promises(
        promises,
        timestamp="2026-05-03T00:00:00+00:00",
    )
    due_time = normalized[0]["due_time"]
    assert "+00:00" in due_time or due_time.endswith("Z"), (
        f"due_time should be UTC after normalization: {due_time!r}"
    )
    assert "2026-05-04T02:00:00" in due_time


def test_normalize_future_promises_normalizes_legacy_iso_offset() -> None:
    """Legacy ISO+offset due_time values should normalize to UTC."""
    from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_persistence import (
        _normalize_future_promises,
    )

    legacy_due = "2026-05-04T14:00:00+12:00"
    promises = [
        {
            "target": "Tester",
            "action": "send reminder",
            "due_time": legacy_due,
            "commitment_type": "future_promise",
        },
    ]
    normalized = _normalize_future_promises(
        promises,
        timestamp="2026-05-03T00:00:00+00:00",
    )
    assert normalized[0]["due_time"] == "2026-05-04T02:00:00+00:00"


def test_normalize_future_promises_drops_invalid_due_time() -> None:
    """Malformed explicit due_time values should not receive fallbacks."""
    from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator_persistence import (
        _normalize_future_promises,
    )

    promises = [
        {
            "target": "Tester",
            "action": "send reminder",
            "due_time": "tomorrow-ish",
            "commitment_type": "future_promise",
        },
    ]
    normalized = _normalize_future_promises(
        promises,
        timestamp="2026-05-03T00:00:00+00:00",
    )
    assert normalized == []


# ---------------------------------------------------------------------------
# RAG compact memory unit row projection
# ---------------------------------------------------------------------------


def test_compact_memory_unit_rows_timestamps_are_local() -> None:
    """_compact_memory_unit_rows should format updated_at and timestamp."""
    from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor2 import (
        _compact_memory_unit_rows,
    )

    rows = [
        {
            "unit_type": "stable_pattern",
            "fact": "Tester likes trains.",
            "updated_at": "2026-05-01T10:00:00+00:00",
            "timestamp": "2026-04-30T08:00:00+00:00",
        },
    ]
    compacted = _compact_memory_unit_rows(rows)
    for row in compacted:
        if isinstance(row, dict):
            _assert_no_utc_leak(row, "$.compact_memory_unit_rows[]")


# ---------------------------------------------------------------------------
# Evidence agent summaries are local
# ---------------------------------------------------------------------------


def test_message_row_text_uses_local_timestamp() -> None:
    """_message_row_text should render timestamps in local format."""
    from kazusa_ai_chatbot.rag.conversation_evidence_agent import _message_row_text

    row = {
        "display_name": "Tester",
        "body_text": "hello",
        "timestamp": "2026-05-02T22:00:00+00:00",
    }
    text = _message_row_text(row)
    _assert_no_utc_leak(text, "$._message_row_text")
    assert "2026-05-03 10:00" in text


# ---------------------------------------------------------------------------
# Cognition prompt helper projection
# ---------------------------------------------------------------------------


def test_cognition_helpers_project_rag_and_history_times() -> None:
    """Cognition L2/L3 helper payloads should not expose stored UTC values."""
    from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l2 as l2
    from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition_l3 as l3

    rag_result = {
        "user_image": {
            "user_memory_context": {
                "active_commitments": [
                    {
                        "fact": "Tester asked for a reminder.",
                        "updated_at": "2026-05-02T20:00:00+00:00",
                    },
                ],
                "stable_patterns": [],
                "recent_shifts": [],
                "objective_facts": [],
                "milestones": [],
            },
            "updated_at": "2026-05-02T20:01:00+00:00",
        },
        "known_facts": [
            {
                "raw_result": {
                    "timestamp": "2026-05-02T20:02:00+00:00",
                },
            },
        ],
        "user_memory_unit_candidates": [
            {"timestamp": "2026-05-02T20:03:00+00:00"},
        ],
    }
    state = {"rag_result": rag_result}
    chat_history = [
        {
            "role": "user",
            "content": "hello",
            "timestamp": "2026-05-02T20:04:00+00:00",
        },
    ]

    _assert_no_utc_leak(l2._current_user_rag_bundle(state), "$.l2.user_bundle")
    _assert_no_utc_leak(l2._cognition_rag_result(rag_result), "$.l2.rag")
    _assert_no_utc_leak(l3._current_user_rag_bundle(state), "$.l3.user_bundle")
    _assert_no_utc_leak(l3._cognition_rag_result(rag_result), "$.l3.rag")
    _assert_no_utc_leak(
        l3._surface_history_for_contextual(chat_history),
        "$.l3.contextual_history",
    )


# ---------------------------------------------------------------------------
# Top-level RAG selector payloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module_name", "response_payload"),
    [
        (
            "kazusa_ai_chatbot.rag.live_context_agent",
            {
                "fact_type": "other",
                "target_source": "unknown",
                "target": "",
                "missing_context": [],
                "selected_summary": "No summary.",
                "projection_payload": {},
                "worker_payloads": {},
            },
        ),
        (
            "kazusa_ai_chatbot.rag.conversation_evidence_agent",
            {
                "worker": "incompatible",
                "reason": "test",
                "requires_person_ref": False,
            },
        ),
        (
            "kazusa_ai_chatbot.rag.memory_evidence_agent",
            {
                "worker": "incompatible",
                "reason": "test",
            },
        ),
        (
            "kazusa_ai_chatbot.rag.person_context_agent",
            {
                "mode": "lookup",
                "target": "display_name",
                "reason": "test",
            },
        ),
    ],
)
async def test_top_level_rag_selector_payload_projects_known_facts(
    monkeypatch,
    module_name: str,
    response_payload: dict,
) -> None:
    """Selector LLM payloads should project known_facts before serialization."""
    import importlib

    module = importlib.import_module(module_name)
    llm = _CapturingAsyncLLM(response_payload)
    selector_function_name = "_select_plan"
    payload_path = f"$.{module_name}._select_plan"
    if module_name == "kazusa_ai_chatbot.rag.live_context_agent":
        monkeypatch.setattr(module, "_external_live_selector_llm", llm)
        selector_function_name = "_select_external_live_plan"
        payload_path = f"$.{module_name}._select_external_live_plan"
    else:
        monkeypatch.setattr(module, "_selector_llm", llm)
        monkeypatch.setattr(
            module,
            "_deterministic_plan",
            lambda task, context=None: None,
        )
    context = {
        "original_query": "selector test",
        "current_slot": "selector test",
        "known_facts": [
            {
                "slot": "prior slot",
                "summary": "Prior evidence.",
                "raw_result": {
                    "rows": [
                        {
                            "timestamp": "2026-05-02T20:00:00+00:00",
                            "body_text": "evidence text",
                        },
                    ],
                    "updated_at": "2026-05-02T20:01:00+00:00",
                },
            },
        ],
    }

    selector_function = getattr(module, selector_function_name)
    await selector_function("selector-only neutral slot", context)

    payload = json.loads(llm.messages[1].content)
    _assert_no_utc_leak(payload, payload_path)
    payload_json = json.dumps(payload, ensure_ascii=False)
    assert "2026-05-03 08:00" in payload_json
