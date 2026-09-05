"""Behavior checks for terminal dialog generation."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    bind_protected_chain_records,
    reset_protected_chain_records,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_episode import (
    build_goal_continuation_ref,
    build_tool_result_episode,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_generator
from tests.unit.nodes.dialog_fixtures import build_dialog_state


class _SequencedLLM:
    """Return deterministic dialog products or provider failures."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[object] = []

    async def ainvoke(self, messages: object, *, config: object) -> object:
        del config
        self.calls.append(messages)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, str):
            content = outcome
        else:
            content = json.dumps(outcome, ensure_ascii=False)
        return SimpleNamespace(content=content)


def _patch_dialog_recorders(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Capture dialog trace, contract, and quality events without persistence."""

    trace_events: list[dict[str, object]] = []
    contract_events: list[dict[str, object]] = []
    quality_events: list[dict[str, object]] = []

    async def record_trace(**kwargs: object) -> dict[str, object]:
        trace_events.append(kwargs)
        return {}

    async def record_contract(**kwargs: object) -> dict[str, object]:
        contract_events.append(kwargs)
        return {}

    async def record_quality(**kwargs: object) -> dict[str, object]:
        quality_events.append(kwargs)
        return {}

    monkeypatch.setattr(
        dialog_module.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_llm_stage_event",
        record_trace,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_model_contract_event",
        record_contract,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_dialog_quality_event",
        record_quality,
    )
    return trace_events, contract_events, quality_events


def _source_dialog_state(
    source_url: str,
    *,
    artifact_text: str | None = None,
) -> dict[str, object]:
    """Build dialog state whose completed tool evidence carries one URL."""

    state = build_dialog_state()
    created_at = "2026-07-14T00:00:00Z"
    continuation_ref = build_goal_continuation_ref(
        source_episode_id="dialog-source-evidence",
        source_message_id="dialog-source-message",
        branch_id="ordinary_response",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "v2-test-user",
        },
    )
    episode = build_tool_result_episode(
        result={
            "schema_version": "tool_result_ready.v1",
            "task_id": "dialog-source-task",
            "task_kind": "background_work",
            "semantic_summary": "Completed source evidence.",
            "artifact_text": artifact_text or source_url,
            "failure_text": "",
            "completed_at": created_at,
            "target_scope": {
                "platform": "debug",
                "platform_channel_id": "channel-test",
                "channel_type": "private",
                "current_platform_user_id": "platform-user-test",
                "current_global_user_id": "v2-test-user",
                "current_display_name": "Test User",
                "target_addressed_user_ids": ["v2-test-user"],
                "target_broadcast": False,
            },
            "evidence_refs": [],
            "result_ref": "dialog-source-result",
            "goal_continuation_ref": continuation_ref,
        },
        evidence_refs=[],
        local_time_context={
            "current_local_datetime": "2026-07-14 12:00",
            "current_local_weekday": "Tuesday",
        },
        created_at=created_at,
    )
    state["cognitive_episode"] = episode
    return state




def test_validated_dialog_messages_collapses_blank_line_runs() -> None:
    """Collapse internal blank lines while preserving message boundaries."""

    value = {
        "final_dialog": [
            "first\n\nsecond\n\nthird\n\nfourth\n\nfifth",
            "single\nline",
        ],
    }

    validated_messages = dialog_module._validated_dialog_messages(value)

    assert validated_messages == [
        "first\nsecond\nthird\nfourth\nfifth",
        "single\nline",
    ]
















