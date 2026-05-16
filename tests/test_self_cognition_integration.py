"""Deterministic integration tests for the self-cognition runtime boundary."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.dispatcher.task import DispatchResult, Task
from kazusa_ai_chatbot.db import user_memory_units as memory_units_module
from kazusa_ai_chatbot.self_cognition import models, projection, sources
from kazusa_ai_chatbot.self_cognition import tracking, worker
from kazusa_ai_chatbot.self_cognition.handoff import dispatch_action_candidate


@pytest.fixture(autouse=True)
def _disable_event_log_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep deterministic self-cognition integration tests off MongoDB."""

    monkeypatch.setattr(
        worker.event_logging,
        "record_self_cognition_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        worker.event_logging,
        "record_worker_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        worker.event_logging,
        "record_dispatcher_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        worker.event_logging,
        "record_runtime_error_event",
        AsyncMock(),
    )


def _target_scope() -> dict[str, str | None]:
    scope = {
        "platform": "qq",
        "platform_channel_id": "673225019",
        "channel_type": "private",
        "user_id": "673225019",
    }
    return scope


def _commitment_case() -> dict[str, Any]:
    case = {
        "case_name": models.CASE_COMMITMENT_PAST_DUE,
        "case_id": "commitment:promise-001",
        "idle_timestamp": "2026-05-13T00:30:00+00:00",
        "last_evidence_timestamp": "2026-05-13T00:00:00+00:00",
        "trigger_kind": models.TRIGGER_ACTIVE_COMMITMENT_DUE_CHECK,
        "semantic_due_state": models.DUE_STATE_PAST_DUE,
        "actionability": "past_due_commitment_contact_socially_available",
        "target_scope": _target_scope(),
        "source_refs": [
            {
                "source_kind": "user_memory_unit",
                "source_id": "promise-001",
                "due_at": "2026-05-13T00:00:00+00:00",
                "summary": "A promised follow-up is due.",
            }
        ],
        "visible_context": [
            {
                "role": "user",
                "body_text": "Please check back after the appointment.",
                "timestamp": "2026-05-12T23:50:00+00:00",
            }
        ],
    }
    return case


def _future_cognition_event() -> dict[str, Any]:
    event = {
        "event_id": "future_cognition:action_attempt:future-123",
        "tool": "trigger_future_cognition",
        "execute_at": "2026-05-16T10:00:00+00:00",
        "created_at": "2026-05-16T09:00:00+00:00",
        "status": "pending",
        "args": {
            "episode_type": "self_cognition",
            "trigger_at": "2026-05-16T10:00:00+00:00",
            "continuation_objective": "Re-check whether a natural pause appeared.",
            "source_action_attempt_id": "action_attempt:future-123",
            "source_refs": [
                {
                    "ref_kind": "cognitive_episode",
                    "ref_id": "episode-123",
                    "owner": "cognition",
                    "relationship": "basis",
                    "evidence_refs": [],
                }
            ],
            "continuation": {
                "mode": "scheduled_followup",
                "episode_type": "self_cognition",
                "max_depth": 1,
                "include_result_as": "scheduled_event",
            },
        },
        "source_platform": "orchestrator",
        "source_channel_id": "",
        "source_channel_type": "internal",
        "source_user_id": "self_cognition",
        "source_message_id": "action_attempt:future-123",
        "source_platform_bot_id": "",
        "source_character_name": "",
        "guild_id": None,
        "bot_role": "system",
    }
    return event


def _future_cognition_case() -> dict[str, Any]:
    case = {
        "case_name": models.CASE_SCHEDULED_FUTURE_COGNITION,
        "case_id": "future_cognition:action_attempt:future-123",
        "idle_timestamp": "2026-05-16T10:00:00+00:00",
        "last_evidence_timestamp": "2026-05-16T10:00:00+00:00",
        "trigger_kind": models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
        "semantic_due_state": models.DUE_STATE_DUE_NOW,
        "actionability": "scheduled_future_cognition_due",
        "target_scope": {
            "platform": "internal",
            "platform_channel_id": "",
            "channel_type": "internal",
            "user_id": None,
        },
        "source_refs": [
            {
                "source_kind": "scheduled_event",
                "source_id": "future_cognition:action_attempt:future-123",
                "due_at": "2026-05-16T10:00:00+00:00",
                "summary": "Re-check whether a natural pause appeared.",
            }
        ],
        "visible_context": [],
        "source_scheduled_event_id": "future_cognition:action_attempt:future-123",
    }
    return case


def _action_attempt(case: dict[str, Any], *, status: str) -> dict[str, Any]:
    source_ref = case["source_refs"][0]
    idempotency_key = tracking.build_idempotency_key(
        source_ref["source_kind"],
        source_ref["source_id"],
        source_ref["due_at"],
        case["target_scope"],
        models.ACTION_KIND_SEND_MESSAGE,
    )
    attempt = {
        "attempt_id": "self_cognition_attempt:promise-001",
        "run_id": "self_cognition_run:promise-001",
        "trigger_id": "self_cognition_trigger:promise-001",
        "source_kind": source_ref["source_kind"],
        "source_id": source_ref["source_id"],
        "target_scope": case["target_scope"],
        "action_kind": models.ACTION_KIND_SEND_MESSAGE,
        "due_at": source_ref["due_at"],
        "idempotency_key": idempotency_key,
        "status": status,
    }
    return attempt


def _action_candidate(attempt: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "attempt_id": attempt["attempt_id"],
        "target_platform": "qq",
        "target_channel": "673225019",
        "target_channel_type": "private",
        "text": "Checking in now.",
        "execute_at": None,
        "dispatch_shape": models.ACTION_KIND_SEND_MESSAGE,
        "production_handoff": False,
    }
    return candidate


def _progress_cognition_output() -> dict[str, Any]:
    """Build a cognition output that stays internal but affects state."""

    output = {
        "logical_stance": "OBSERVE",
        "character_intent": "WAIT",
        "mood": "subdued",
        "affinity_delta": -1,
        "action_directives": {
            "linguistic_directives": {
                "content_anchors": [
                    "[AUDIT_ONLY] The missed promise should be remembered.",
                ],
            },
        },
    }
    return output


def _action_cognition_output() -> dict[str, Any]:
    """Build a cognition output that requests dispatcher handoff."""

    output = {
        "logical_stance": "FOLLOW_UP",
        "character_intent": "FOLLOW_UP",
        "mood": "hurt",
        "affinity_delta": -1,
        "action_directives": {
            "linguistic_directives": {
                "content_anchors": [
                    "[ACTION_CANDIDATE] Checking in now.",
                ],
            },
        },
    }
    return output


def _consolidation_result() -> dict[str, Any]:
    """Build the shared consolidator metadata shape used by worker tests."""

    result = {
        "consolidation_metadata": {
            "write_success": {
                "character_state": True,
                "relationship_insight": True,
                "user_memory_units": False,
                "task_dispatch": False,
                "affinity": True,
                "character_image": False,
                "cache_invalidation": True,
            },
            "scheduled_event_ids": [],
            "cache_evicted_count": 1,
        },
    }
    return result


def _case_runner_with_candidate(
    case: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    del output_dir
    attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_CANDIDATE,
    )
    candidate = _action_candidate(attempt)
    payloads = {
        models.ARTIFACT_ACTION_ATTEMPT: attempt,
        models.ARTIFACT_ACTION_CANDIDATE: candidate,
    }
    return payloads


def _case_runner_with_tracking(
    case: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Build action artifacts using real tracking duplicate logic."""

    del output_dir
    trigger_record = tracking.build_trigger_record(case)
    existing_attempts = case.get("existing_attempts")
    if not isinstance(existing_attempts, list):
        existing_attempts = []
    action_attempt = tracking.build_action_attempt(
        case,
        trigger_record,
        [
            attempt
            for attempt in existing_attempts
            if isinstance(attempt, dict)
        ],
    )
    action_candidate = tracking.build_action_candidate(
        case,
        action_attempt,
        "Checking in now.",
    )
    payloads = {models.ARTIFACT_ACTION_ATTEMPT: action_attempt}
    if action_candidate is not None:
        payloads[models.ARTIFACT_ACTION_CANDIDATE] = action_candidate
    return payloads


@pytest.mark.asyncio
async def test_collect_scheduled_future_cognition_cases_projects_due_slots() -> None:
    """Due future-cognition slots become normal prompt-safe trigger cases."""

    now = datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc)
    calls: list[dict[str, Any]] = []

    async def list_due_events(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(dict(kwargs))
        return [_future_cognition_event()]

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=now,
        character_profile={"name": "TestCharacter"},
        max_cases=3,
        list_due_events_func=list_due_events,
    )

    assert calls == [{"current_timestamp": now.isoformat(), "limit": 3}]
    assert len(cases) == 1
    case = cases[0]
    assert case["case_name"] == models.CASE_SCHEDULED_FUTURE_COGNITION
    assert case["trigger_kind"] == models.TRIGGER_SCHEDULED_FUTURE_COGNITION
    assert case["case_id"] == "future_cognition:action_attempt:future-123"
    assert (
        case["source_scheduled_event_id"]
        == "future_cognition:action_attempt:future-123"
    )
    assert case["source_refs"][0]["source_kind"] == "scheduled_event"
    assert case["source_refs"][0]["source_id"] == (
        "scheduled_future_cognition_slot"
    )
    assert case["source_refs"][0]["summary"] == (
        "Re-check whether a natural pause appeared."
    )
    assert case["rag_query"] == "Re-check whether a natural pause appeared."
    assert case["conversation_progress"]["continuation_objective"] == (
        "Re-check whether a natural pause appeared."
    )
    assert "context_summary" not in case["conversation_progress"]
    source_packet = projection.build_source_packet(case)
    rendered_packet = projection.render_source_packet_text(source_packet)
    serialized = json.dumps(source_packet, ensure_ascii=False).lower()
    serialized = f"{serialized}\n{rendered_packet.lower()}"
    for forbidden in (
        "action_attempt:future-123",
        "episode-123",
        "future-123",
        "handler_id",
        "credential",
        "mongodb",
        "collection",
        "episode_type",
        "include_result_as",
        "max_depth",
        "raw_channel",
        "schema_version",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_collect_scheduled_future_cognition_cases_preserves_source_scope() -> None:
    """Scheduled future cognition should keep trusted scope for RAG/context."""

    now = datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc)
    event = _future_cognition_event()
    event.update(
        {
            "source_platform": "qq",
            "source_channel_id": "54369546",
            "source_channel_type": "group",
            "source_user_id": "673225019",
            "source_platform_bot_id": "bot-001",
            "source_character_name": "TestCharacter",
        }
    )

    async def list_due_events(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [event]

    cases = await sources.collect_scheduled_future_cognition_cases(
        now=now,
        character_profile={"name": "TestCharacter"},
        max_cases=1,
        list_due_events_func=list_due_events,
    )

    assert cases[0]["target_scope"] == {
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "user_id": "673225019",
        "display_name": "673225019",
    }
    assert cases[0]["platform_bot_id"] == "bot-001"


@pytest.mark.asyncio
async def test_collect_self_cognition_cases_includes_future_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared collector should include due scheduled cognition slots."""

    async def no_commitments(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return []

    async def future_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_future_cognition_case()]

    monkeypatch.setattr(sources, "collect_active_commitment_cases", no_commitments)
    monkeypatch.setattr(
        sources,
        "collect_scheduled_future_cognition_cases",
        future_cases,
    )

    cases = await sources.collect_self_cognition_cases(
        now=datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc),
        character_profile={"name": "TestCharacter"},
        max_cases=3,
    )

    assert [case["trigger_kind"] for case in cases] == [
        models.TRIGGER_SCHEDULED_FUTURE_COGNITION,
    ]


@pytest.mark.asyncio
async def test_worker_tick_marks_future_cognition_slot_completed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A processed scheduled cognition slot should not stay pending forever."""

    completed_event_ids: list[str] = []

    async def collect_cases(**kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return [_future_cognition_case()]

    async def run_case(case: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        assert case["trigger_kind"] == models.TRIGGER_SCHEDULED_FUTURE_COGNITION
        assert output_dir.name
        return {}

    async def mark_completed(event_id: str) -> bool:
        completed_event_ids.append(event_id)
        return True

    monkeypatch.setattr(worker.db, "mark_scheduled_event_completed", mark_completed)

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=None,
        now=datetime(2026, 5, 16, 10, 0, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=lambda **kwargs: [],
        record_attempt_func=lambda attempt: None,
        max_cases=3,
    )

    assert result.processed_count == 1
    assert completed_event_ids == ["future_cognition:action_attempt:future-123"]


class _FakeDispatcher:
    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject
        self.calls: list[dict[str, Any]] = []

    async def dispatch(self, raw_calls, ctx, *, instruction: str = ""):
        self.calls.append(
            {
                "raw_calls": raw_calls,
                "ctx": ctx,
                "instruction": instruction,
            }
        )
        raw_call = raw_calls[0]
        if self.reject:
            result = DispatchResult(
                scheduled=[],
                rejected=[(raw_call, "no adapters registered")],
            )
            return result
        task = Task(
            tool=raw_call.tool,
            args=dict(raw_call.args),
            execute_at=ctx.now,
        )
        result = DispatchResult(
            scheduled=[(task, "event-001")],
            rejected=[],
        )
        return result


class _AsyncCursor:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = iter(docs)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            row = next(self._docs)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        return row


class _FakeUserMemoryUnitsCollection:
    def __init__(self) -> None:
        self.pipeline: list[dict[str, Any]] = []

    def aggregate(self, pipeline: list[dict[str, Any]]):
        self.pipeline = pipeline
        cursor = _AsyncCursor([{"unit_id": "promise-001"}])
        return cursor


@pytest.mark.asyncio
async def test_worker_default_path_requests_production_consolidation_without_files(
    monkeypatch,
    tmp_path,
) -> None:
    """Default worker runs should request consolidation and stay in memory."""

    case = _commitment_case()
    captured_kwargs: dict[str, Any] = {}

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return []

    async def build_artifacts(
        next_case: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured_kwargs.update(kwargs)
        trigger_record = tracking.build_trigger_record(next_case)
        run_record = tracking.build_run_record(
            next_case,
            trigger_record,
            models.ROUTE_AUDIT_ONLY,
            {
                "rag_calls": 0,
                "cognition_calls": 1,
                "dialog_calls": 1,
                "topic_limit": models.TOPIC_LIMIT,
            },
        )
        payloads = {
            models.ARTIFACT_TRIGGER_RECORD: trigger_record,
            models.ARTIFACT_RUN_RECORD: run_record,
            models.ARTIFACT_CONSOLIDATION_OUTCOME: {
                "consolidation_called": True,
                "write_success": {"character_state": True},
                "scheduled_event_count": 0,
                "cache_evicted_count": 0,
                "origin_trigger_source": "internal_thought",
                "origin_episode_id": "self_cognition:dry_run:test",
            },
        }
        return payloads

    monkeypatch.setattr(
        worker.runner,
        "build_self_cognition_case_artifacts_async",
        build_artifacts,
    )

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=None,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        read_attempts_func=read_attempts,
        max_cases=3,
    )

    assert result.processed_count == 1
    assert captured_kwargs["apply_consolidation"] is True
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_default_path_applies_consolidation_without_dispatch_or_files(
    monkeypatch,
    tmp_path,
) -> None:
    """Internal-only cognition should consolidate without outward delivery."""

    case = _commitment_case()
    captured_consolidation_state: dict[str, Any] = {}
    dispatcher = _FakeDispatcher()

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return []

    async def cognition_client(state: dict[str, Any]) -> dict[str, Any]:
        assert state["cognitive_episode"]["trigger_source"] == "internal_thought"
        return _progress_cognition_output()

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        assert state["should_respond"] is False
        assert state["dialog_usage_mode"] == "self_cognition_private_finalization"
        return {"final_dialog": ["Private finalization for consolidation only."]}

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_consolidation_state.update(state)
        return _consolidation_result()

    monkeypatch.setattr(worker.runner, "_default_cognition_client", cognition_client)
    monkeypatch.setattr(worker.runner, "_default_dialog_client", dialog_client)
    monkeypatch.setattr(
        worker.runner,
        "_default_consolidation_client",
        consolidation_client,
    )

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        read_attempts_func=read_attempts,
        max_cases=3,
    )

    assert result.processed_count == 1
    assert result.dispatched_count == 0
    assert dispatcher.calls == []
    assert captured_consolidation_state["cognitive_episode"][
        "trigger_source"
    ] == "internal_thought"
    assert captured_consolidation_state["final_dialog"] == [
        "Private finalization for consolidation only.",
    ]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_default_path_preserves_action_handoff_with_consolidation(
    monkeypatch,
    tmp_path,
) -> None:
    """Consolidation should not bypass the existing dispatcher handoff."""

    case = _commitment_case()
    dispatcher = _FakeDispatcher()
    recorded_attempts: list[dict[str, Any]] = []
    captured_consolidation_state: dict[str, Any] = {}

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return list(recorded_attempts)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    async def cognition_client(state: dict[str, Any]) -> dict[str, Any]:
        assert state["cognitive_episode"]["trigger_source"] == "internal_thought"
        return _action_cognition_output()

    async def dialog_client(state: dict[str, Any]) -> dict[str, Any]:
        assert state["should_respond"] is False
        assert state["dialog_usage_mode"] == "self_cognition_private_finalization"
        return {"final_dialog": ["Private finalization for consolidation only."]}

    async def consolidation_client(state: dict[str, Any]) -> dict[str, Any]:
        captured_consolidation_state.update(state)
        return _consolidation_result()

    monkeypatch.setattr(worker.runner, "_default_cognition_client", cognition_client)
    monkeypatch.setattr(worker.runner, "_default_dialog_client", dialog_client)
    monkeypatch.setattr(
        worker.runner,
        "_default_consolidation_client",
        consolidation_client,
    )

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        read_attempts_func=read_attempts,
        record_attempt_func=record_attempt,
        max_cases=3,
    )

    assert result.dispatched_count == 1
    assert dispatcher.calls[0]["raw_calls"][0].tool == "send_message"
    assert dispatcher.calls[0]["raw_calls"][0].args["text"] == "Checking in now."
    assert recorded_attempts[0]["status"] == models.ACTION_ATTEMPT_STATUS_SCHEDULED
    assert captured_consolidation_state["final_dialog"] == [
        "Private finalization for consolidation only.",
    ]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_tick_loads_prior_attempts_before_running_case(
    tmp_path,
) -> None:
    """Prior persisted attempts should enter the next case run."""

    case = _commitment_case()
    prior_attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_SCHEDULED,
    )
    captured_case: dict[str, Any] = {}

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        assert max_cases == 3
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return [prior_attempt]

    async def run_case(next_case: dict[str, Any], output_dir: Path) -> dict[str, Any]:
        del output_dir
        captured_case.update(next_case)
        return {}

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=None,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_case,
        read_attempts_func=read_attempts,
        max_cases=3,
    )

    assert result.processed_count == 1
    assert captured_case["existing_attempts"][0]["idempotency_key"] == (
        prior_attempt["idempotency_key"]
    )


@pytest.mark.asyncio
async def test_worker_tick_dispatches_candidate_through_task_dispatcher(
    tmp_path,
) -> None:
    """A live candidate should use TaskDispatcher instead of direct adapter send."""

    case = _commitment_case()
    dispatcher = _FakeDispatcher()
    recorded_attempts: list[dict[str, Any]] = []

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return list(recorded_attempts)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=_case_runner_with_candidate,
        read_attempts_func=read_attempts,
        record_attempt_func=record_attempt,
        max_cases=3,
    )

    assert result.dispatched_count == 1
    assert result.artifact_paths == []
    assert dispatcher.calls[0]["raw_calls"][0].tool == "send_message"
    assert dispatcher.calls[0]["raw_calls"][0].args["text"] == "Checking in now."
    assert recorded_attempts[0]["status"] == models.ACTION_ATTEMPT_STATUS_SCHEDULED
    assert recorded_attempts[0]["dispatch_status"] == "accepted"
    assert recorded_attempts[0]["scheduled_event_ids"] == ["event-001"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_tick_records_dispatch_rejection_without_adapter_send(
    tmp_path,
) -> None:
    """Dispatcher rejection should be tracked locally without calling adapters."""

    case = _commitment_case()
    dispatcher = _FakeDispatcher(reject=True)
    recorded_attempts: list[dict[str, Any]] = []

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return list(recorded_attempts)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=_case_runner_with_candidate,
        read_attempts_func=read_attempts,
        record_attempt_func=record_attempt,
        max_cases=3,
    )

    assert result.rejected_count == 1
    assert recorded_attempts[0]["status"] == models.ACTION_ATTEMPT_STATUS_HELD
    assert recorded_attempts[0]["dispatch_status"] == "rejected"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_tick_suppresses_duplicate_due_occurrence_from_prior_attempts(
    tmp_path,
) -> None:
    """A prior persisted attempt should prevent a repeated dispatcher handoff."""

    case = _commitment_case()
    prior_attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_SCHEDULED,
    )
    recorded_attempts = [prior_attempt]
    dispatcher = _FakeDispatcher()

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        del now, max_cases
        return [case]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return list(recorded_attempts)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    def run_duplicate_case(
        next_case: dict[str, Any],
        output_dir: Path,
    ) -> dict[str, Any]:
        del output_dir
        assert next_case["existing_attempts"][0]["idempotency_key"] == (
            prior_attempt["idempotency_key"]
        )
        duplicate = _action_attempt(
            next_case,
            status=models.ACTION_ATTEMPT_STATUS_DUPLICATE,
        )
        payloads = {models.ARTIFACT_ACTION_ATTEMPT: duplicate}
        return payloads

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=run_duplicate_case,
        read_attempts_func=read_attempts,
        record_attempt_func=record_attempt,
        max_cases=3,
    )

    assert result.dispatched_count == 0
    assert dispatcher.calls == []
    assert recorded_attempts[-1]["status"] == (
        models.ACTION_ATTEMPT_STATUS_DUPLICATE
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_tick_uses_attempt_updates_between_cases(tmp_path) -> None:
    """Same-tick duplicate cases should see persisted attempts recorded earlier."""

    case = _commitment_case()
    dispatcher = _FakeDispatcher()
    recorded_attempts: list[dict[str, Any]] = []

    async def collect_cases(
        *,
        now: datetime,
        max_cases: int,
    ) -> list[dict[str, Any]]:
        del now, max_cases
        return [case, dict(case)]

    async def read_attempts(*, limit: int) -> list[dict[str, Any]]:
        assert limit > 0
        return list(recorded_attempts)

    async def record_attempt(attempt: dict[str, Any]) -> None:
        recorded_attempts.append(dict(attempt))

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: False,
        collect_cases_func=collect_cases,
        run_case_func=_case_runner_with_tracking,
        read_attempts_func=read_attempts,
        record_attempt_func=record_attempt,
        max_cases=3,
    )

    assert result.processed_count == 2
    assert result.dispatched_count == 1
    assert len(dispatcher.calls) == 1
    assert recorded_attempts[0]["status"] == models.ACTION_ATTEMPT_STATUS_SCHEDULED
    assert recorded_attempts[1]["status"] == (
        models.ACTION_ATTEMPT_STATUS_DUPLICATE
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_worker_tick_defers_when_primary_interaction_is_busy(tmp_path) -> None:
    """The idle worker should not compete with active chat work."""

    async def collect_cases(*, now: datetime, max_cases: int) -> list[dict[str, Any]]:
        raise AssertionError("busy tick should not collect cases")

    result = await worker.run_self_cognition_worker_tick(
        output_root=tmp_path,
        dispatcher=None,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
        is_primary_interaction_busy=lambda: True,
        collect_cases_func=collect_cases,
        max_cases=3,
    )

    assert result.deferred is True
    assert result.defer_reason == "primary interaction busy"
    assert result.processed_count == 0


@pytest.mark.asyncio
async def test_dispatch_action_candidate_builds_existing_dispatch_context() -> None:
    """The handoff layer should preserve candidate target and source context."""

    case = _commitment_case()
    attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_CANDIDATE,
    )
    candidate = _action_candidate(attempt)
    dispatcher = _FakeDispatcher()

    result = await dispatch_action_candidate(
        case,
        attempt,
        candidate,
        dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )

    raw_call = dispatcher.calls[0]["raw_calls"][0]
    ctx = dispatcher.calls[0]["ctx"]

    assert raw_call.tool == "send_message"
    assert raw_call.args["target_channel"] == "673225019"
    assert ctx.source_platform == "qq"
    assert ctx.source_channel_type == "private"
    assert result["production_handoff"] is True


@pytest.mark.asyncio
async def test_dispatch_action_candidate_preserves_delivery_mentions() -> None:
    """Self-cognition handoff should pass mention metadata as task args."""

    case = _commitment_case()
    case["target_scope"] = {
        "platform": "qq",
        "platform_channel_id": "54369546",
        "channel_type": "group",
        "user_id": "user-1",
    }
    attempt = _action_attempt(
        case,
        status=models.ACTION_ATTEMPT_STATUS_CANDIDATE,
    )
    mention = {
        "entity_kind": "user",
        "placement": "prefix",
        "platform_user_id": "platform-user-1",
        "global_user_id": "user-1",
        "display_name": "Target User",
        "requested_by": "dialog.mention_target_user",
    }
    candidate = _action_candidate(attempt)
    candidate["target_channel"] = "54369546"
    candidate["target_channel_type"] = "group"
    candidate["delivery_mentions"] = [mention]
    dispatcher = _FakeDispatcher()

    await dispatch_action_candidate(
        case,
        attempt,
        candidate,
        dispatcher,
        now=datetime(2026, 5, 13, tzinfo=timezone.utc),
    )

    raw_call = dispatcher.calls[0]["raw_calls"][0]
    assert raw_call.args["text"] == "Checking in now."
    assert raw_call.args["delivery_mentions"] == [mention]


@pytest.mark.asyncio
async def test_active_commitment_source_builds_due_case_from_memory_unit() -> None:
    """Active commitment collection should build visible/actionable case input."""

    unit = {
        "unit_id": "promise-001",
        "global_user_id": "673225019",
        "unit_type": "active_commitment",
        "status": "active",
        "fact": "A promised follow-up is due.",
        "subjective_appraisal": "The user may expect a check-in.",
        "relationship_signal": "Following through matters.",
        "due_at": "2026-05-13T00:00:00+00:00",
        "last_seen_at": "2026-05-12T23:55:00+00:00",
        "updated_at": "2026-05-12T23:55:00+00:00",
    }
    rows = [
        {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "role": "user",
            "global_user_id": "673225019",
            "display_name": "User",
            "body_text": "Please check back after the appointment.",
            "timestamp": "2026-05-12T23:50:00+00:00",
        }
    ]

    async def list_commitments(*, current_timestamp: str, limit: int):
        assert current_timestamp == "2026-05-13T00:30:00+00:00"
        assert limit == 3
        return [unit]

    async def get_history(**kwargs):
        assert kwargs["global_user_id"] == "673225019"
        return rows

    async def get_profile(global_user_id: str):
        assert global_user_id == "673225019"
        return {"affinity": 600, "display_name": "User"}

    cases = await sources.collect_active_commitment_cases(
        now=datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
        character_profile={"name": "Character", "mood": "focused"},
        max_cases=3,
        list_active_commitments_func=list_commitments,
        get_conversation_history_func=get_history,
        get_user_profile_func=get_profile,
    )

    assert len(cases) == 1
    assert cases[0]["case_name"] == models.CASE_COMMITMENT_PAST_DUE
    assert cases[0]["target_scope"] == {
        **_target_scope(),
        "platform_user_id": "",
        "display_name": "User",
    }
    assert cases[0]["source_refs"][0]["source_id"] == "promise-001"
    assert cases[0]["visible_context"][0]["body_text"].startswith("Please")


@pytest.mark.asyncio
async def test_active_commitment_query_prioritizes_due_work(
    monkeypatch,
) -> None:
    """Active commitment reads should prioritize due items inside the tick cap."""

    collection = _FakeUserMemoryUnitsCollection()

    class FakeDatabase:
        user_memory_units = collection

    async def fake_get_db():
        database = FakeDatabase()
        return database

    monkeypatch.setattr(memory_units_module, "get_db", fake_get_db)

    rows = await memory_units_module.query_active_commitment_memory_units(
        current_timestamp="2026-05-13T00:30:00+00:00",
        limit=3,
    )
    pipeline = collection.pipeline

    assert rows == [{"unit_id": "promise-001"}]
    assert pipeline[0]["$match"]["due_at"] == {"$type": "string", "$ne": ""}
    assert pipeline[1]["$addFields"]["_self_cognition_due_at"] == {
        "$dateFromString": {
            "dateString": {
                "$replaceOne": {
                    "input": "$due_at",
                    "find": " ",
                    "replacement": "T",
                }
            },
            "onError": None,
            "onNull": None,
        }
    }
    assert pipeline[2] == {"$match": {"_self_cognition_due_at": {"$ne": None}}}
    assert pipeline[3]["$addFields"]["_self_cognition_due_bucket"] == {
        "$cond": [
            {
                "$lte": [
                    "$_self_cognition_due_at",
                    datetime(2026, 5, 13, 0, 30, tzinfo=timezone.utc),
                ]
            },
            memory_units_module.ACTIVE_COMMITMENT_DUE_BUCKET_READY,
            memory_units_module.ACTIVE_COMMITMENT_DUE_BUCKET_FUTURE,
        ]
    }
    assert pipeline[4]["$sort"] == {
        "_self_cognition_due_bucket": 1,
        "_self_cognition_due_at": 1,
        "last_seen_at": -1,
        "updated_at": -1,
    }
    assert pipeline[5] == {"$limit": 3}
    assert pipeline[6]["$project"]["_self_cognition_due_at"] == 0
    assert pipeline[6]["$project"]["_self_cognition_due_bucket"] == 0
