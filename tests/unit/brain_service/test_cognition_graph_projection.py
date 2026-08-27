"""Deterministic ownership checks for Brain cognition observations."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import service
from kazusa_ai_chatbot.brain_service import (
    CognitionRunObservationV1,
    cognition_observation_projection,
)
from kazusa_ai_chatbot.brain_service.cognition_observation_projection import (
    build_live_cognition_observation,
)
from kazusa_ai_chatbot.chat_input_queue import QueuedChatItem
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    record_v2_shared_memory_prewarm_checkpoint,
    reset_v2_attempt_ledger,
)
from kazusa_ai_chatbot.state import (
    MAX_EPISODE_ATTEMPT_DIAGNOSTICS,
    append_attempt_diagnostics,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.test_service_background_consolidation import (
    _chat_http_request,
    _chat_request,
    _patch_chat_dependencies,
    _reset_queue_state,
)
from tests.test_shared_memory_prewarm import _ready_outcome

_NOW = datetime(2026, 8, 26, 0, 0, tzinfo=timezone.utc)


def _attempt_diagnostic(error_code: str) -> dict[str, object]:
    """Build one valid bounded episode-attempt diagnostic row."""

    return {
        "schema_version": "episode_attempt_diagnostic.v1",
        "stage": "relevance",
        "error_code": error_code,
        "attempt_count": 2,
        "safe_checkpoint": "pre_state_commit",
        "retryable": False,
        "final_status": "accepted_degraded",
    }


def _observation(*, status: str = "completed_private") -> CognitionRunObservationV1:
    """Build a small canonical live observation for service boundary tests."""

    state = {
        "user_input": "hello",
        "cognition_core_output": {
            "schema_version": "cognition_output.v3",
            "appraisals": [],
            "active_character_goal": {},
            "response_plan": {
                "action_requests": [],
                "resolver_requests": [],
            },
            "affect_projection": [],
            "private_monologue": "thinking",
        },
    }
    graph_result = {
        "should_respond": True,
        "reason_to_respond": "grounded reason",
        "final_dialog": [],
    }
    observation = build_live_cognition_observation(
        graph_result=graph_result,
        persona_state=state,
        run_id="run-1",
        cognition_invocation_id="invocation-1",
        terminal_status=status,
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert observation is not None
    return observation


def test_service_publishes_canonical_observation_without_legacy_graph_helpers() -> None:
    """Typed recorders accept only the Brain-owned observation DTO."""

    observation = _observation()
    service._clear_latest_cognition_graph()
    service._record_latest_cognition_graph(observation)
    assert isinstance(service._latest_cognition_graph, CognitionRunObservationV1)
    assert service._latest_cognition_graph is not observation


def test_live_terminal_status_mapping_and_cancellation_are_exact(monkeypatch) -> None:
    """Terminal mapping keeps cancellation out of the observation stream."""

    monkeypatch.setattr(
        cognition_observation_projection,
        "COGNITION_VISUAL_DIRECTIVES_ENABLED",
        True,
    )
    for terminal_status in (
        "completed_visible",
        "completed_private",
        "completed_action",
        "scheduled",
    ):
        completed = _observation(status=terminal_status)
        assert completed.status == "completed"

    failed = _observation(status="failed")
    assert failed.status == "failed"

    visual_failure = build_live_cognition_observation(
        graph_result={
            "should_respond": True,
            "reason_to_respond": "grounded reason",
            "final_dialog": [],
        },
        persona_state={
            "cognition_core_output": {
                "schema_version": "cognition_output.v3",
                "appraisals": [],
                "active_character_goal": {},
                "response_plan": {
                    "action_requests": [],
                    "resolver_requests": [],
                },
                "affect_projection": [],
                "private_monologue": "thinking",
            },
        },
        run_id="run-1",
        cognition_invocation_id="invocation-1",
        terminal_status="completed_visible",
        visual_stage_failed=True,
        visual_stage_reached=True,
        failure_code="visual_projection_failed",
        generated_at=_NOW,
    )
    assert visual_failure is not None
    assert visual_failure.status == "partial"

    failed_with_visual_failure = build_live_cognition_observation(
        graph_result={
            "should_respond": True,
            "reason_to_respond": "grounded reason",
            "final_dialog": [],
        },
        persona_state={},
        run_id="run-1",
        cognition_invocation_id="invocation-1",
        terminal_status="failed",
        visual_stage_failed=True,
        visual_stage_reached=True,
        failure_code="graph_failed",
        generated_at=_NOW,
    )
    assert failed_with_visual_failure is not None
    assert failed_with_visual_failure.status == "failed"

    cancelled = build_live_cognition_observation(
        graph_result={},
        persona_state={},
        run_id="run-1",
        cognition_invocation_id="invocation-1",
        terminal_status="cancelled",
        visual_stage_failed=False,
        visual_stage_reached=None,
        failure_code="",
        generated_at=_NOW,
    )
    assert cancelled is None


class _CheckpointFailingGraph:
    """Record a typed prewarm checkpoint before raising through the graph."""

    def __init__(self, outcome: dict[str, object]) -> None:
        """Store the checkpoint that the failed graph attempt publishes."""

        self.outcome = outcome

    async def ainvoke(self, _state: dict[str, object]) -> dict[str, object]:
        """Publish one checkpoint and raise a deterministic graph failure."""

        record_v2_shared_memory_prewarm_checkpoint(self.outcome)
        raise RuntimeError("graph failed after prewarm")


@pytest.mark.asyncio
async def test_failed_run_uses_current_attempt_prewarm_checkpoint(monkeypatch) -> None:
    """Failed publication keeps only the current graph-attempt outcome."""

    await _reset_queue_state()
    outcome = _ready_outcome(
        reason_code="shared_memory_ready",
        merged_shared_count=0,
    )
    ledgers = []

    def _create_ledger(correlation_id: str):
        ledger = create_v2_attempt_ledger(correlation_id)
        ledgers.append(ledger)
        return ledger

    monkeypatch.setattr(service, "create_v2_attempt_ledger", _create_ledger)
    _patch_chat_dependencies(monkeypatch, _CheckpointFailingGraph(outcome))

    async def _frontline(_state):
        return {
            "decision": {
                "intake_action": "start",
                "append_target": "none",
                "prelude_targets": [],
                "reason": "deterministic service fixture",
            },
            "attempt_diagnostics": [],
        }

    async def _settled(_state):
        return {
            "decision": {
                "response_action": "proceed",
                "reason_to_respond": "deterministic service fixture",
                "use_reply_feature": False,
                "channel_topic": "",
                "indirect_speech_context": "",
            },
            "attempt_diagnostics": [],
        }

    monkeypatch.setattr(service, "frontline_relevance_agent", _frontline)
    monkeypatch.setattr(service, "relevance_agent", _settled)
    service._clear_latest_cognition_graph()

    response = await service.chat(
        _chat_request(message_id="checkpoint-failure"),
        BackgroundTasks(),
        _chat_http_request(),
    )

    observation = response.cognition_graph
    assert observation is not None
    prewarm_section = next(
        section
        for section in observation.sections
        if section.section_id == "evidence.shared_memory_prewarm"
    )
    assert prewarm_section.status == "completed"
    assert prewarm_section.fields[1].value == "shared_memory_ready"
    assert observation.status == "failed"
    assert service._latest_cognition_graph == observation
    assert len(ledgers) == 1
    assert observation.correlation.run_id == (
        ledgers[0].cognition_invocation_id
    )
    assert observation.correlation.cognition_invocation_id == (
        ledgers[0].cognition_invocation_id
    )

    ledger = create_v2_attempt_ledger("retry-isolation")
    first_token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
    try:
        record_v2_shared_memory_prewarm_checkpoint(outcome)
        assert service.snapshot_v2_shared_memory_prewarm_checkpoint() == outcome
    finally:
        reset_v2_attempt_ledger(first_token)
    second_token = bind_v2_attempt_ledger(ledger, graph_attempt=2)
    try:
        assert service.snapshot_v2_shared_memory_prewarm_checkpoint() is None
    finally:
        reset_v2_attempt_ledger(second_token)
    await _reset_queue_state()


def test_legacy_cognition_graph_projection_symbols_are_absent_from_production() -> None:
    """Service no longer owns the removed graph projection vocabulary."""

    source = Path(service.__file__).read_text(encoding="utf-8")
    frozen_symbols = (
        "_build_response_cognition_graph",
        "_build_self_cognition_cognition_graph",
        "_graph_cognition_nodes",
        "_graph_memory_detail",
        "_graph_visual_node",
        "_GRAPH_EVIDENCE_FIELDS",
    )
    for symbol in frozen_symbols:
        assert symbol not in source


def test_cognition_contract_exhaustion_maps_to_model_contract_error_code() -> None:
    """Typed cognition model exhaustion keeps the stable service category."""

    error = service.CognitionExecutionError(
        "cognition stage contract exhausted",
        error_code="cognition_p_contract_exhausted",
        stage="cognition_core_v3.P",
        attempt_count=3,
        safe_checkpoint="pre_state_commit",
        retryable=True,
    )

    metadata = service._operational_failure_metadata(error)

    assert metadata[0] == "model_contract"
    assert metadata[1] == "cognition_core_v3.P"
    assert metadata[2] == 3
    assert metadata[3] is True


@pytest.mark.asyncio
async def test_pre_commit_contract_exhaustion_triggers_one_checkpoint_replay(
    monkeypatch,
) -> None:
    """The service replays one typed pre-commit contract exhaustion."""

    await _reset_queue_state()
    error = service.CognitionExecutionError(
        "cognition stage contract exhausted",
        error_code="cognition_p_contract_exhausted",
        stage="cognition_core_v3.P",
        attempt_count=3,
        safe_checkpoint="pre_state_commit",
        retryable=True,
    )

    class _ReplayGraph:
        """Raise once at the typed checkpoint, then complete normally."""

        def __init__(self) -> None:
            self.attempts = 0

        async def ainvoke(self, _state: dict[str, object]) -> dict[str, object]:
            self.attempts += 1
            if self.attempts == 1:
                raise error
            return {
                "should_respond": True,
                "use_reply_feature": False,
                "final_dialog": ["replayed response"],
                "future_promises": [],
                "consolidation_state": {},
            }

    graph = _ReplayGraph()
    _patch_chat_dependencies(monkeypatch, graph)

    async def _frontline(_state):
        return {
            "decision": {
                "intake_action": "start",
                "append_target": "none",
                "prelude_targets": [],
                "reason": "deterministic service fixture",
            },
            "attempt_diagnostics": [],
        }

    async def _settled(_state):
        return {
            "decision": {
                "response_action": "proceed",
                "reason_to_respond": "deterministic service fixture",
                "use_reply_feature": False,
                "channel_topic": "",
                "indirect_speech_context": "",
            },
            "attempt_diagnostics": [],
        }

    monkeypatch.setattr(service, "frontline_relevance_agent", _frontline)
    monkeypatch.setattr(service, "relevance_agent", _settled)

    response = await service.chat(
        _chat_request(message_id="contract-replay"),
        BackgroundTasks(),
        _chat_http_request(),
    )

    assert graph.attempts == 2
    assert response.operational_error is None
    assert service._can_retry_cognition_failure(error, 2) is False
    source = Path(service.__file__).read_text(encoding="utf-8")
    assert "Retrying cognition after pre-commit state conflict" not in source
    assert "Retrying cognition after retryable pre-commit model " in source
    assert "contract failure: {exc}" in source
    await _reset_queue_state()


def test_post_commit_degradations_do_not_trigger_replay() -> None:
    """A typed failure after state commit cannot repeat the graph."""

    error = service.CognitionExecutionError(
        "post-commit cognition degradation",
        error_code="cognition_p_contract_exhausted",
        stage="cognition.persistence",
        attempt_count=1,
        safe_checkpoint="post_state_commit",
        retryable=True,
    )

    assert service._can_retry_cognition_failure(error, 1) is False


def test_attempt_diagnostics_reducer_concatenates_within_bound() -> None:
    """The sole reducer retains the newest rows in chronological order."""

    current = [
        _attempt_diagnostic(f"row-{index:02d}")
        for index in range(1, 16)
    ]
    update = [_attempt_diagnostic("row-16"), _attempt_diagnostic("row-17")]

    reduced = append_attempt_diagnostics(current, update)

    assert len(reduced) == MAX_EPISODE_ATTEMPT_DIAGNOSTICS
    assert [row["error_code"] for row in reduced] == [
        f"row-{index:02d}" for index in range(2, 18)
    ]


def test_initial_process_state_seeds_relevance_diagnostics_before_persona() -> None:
    """Service places the settlement carrier in graph input before invocation."""

    source = Path(service.__file__).read_text(encoding="utf-8")
    initial_state_start = source.index(
        "initial_state: IMProcessState = {",
        source.index("async def _process_queued_chat_item"),
    )
    graph_invocation = source.index(
        "result = await _graph.ainvoke(",
        initial_state_start,
    )
    diagnostics_marker = source.index(
        '"attempt_diagnostics": [',
        initial_state_start,
    )

    assert initial_state_start < diagnostics_marker < graph_invocation
    assert "settled_attempt_diagnostics" in source[diagnostics_marker:graph_invocation]


@pytest.mark.asyncio
async def test_service_consumes_only_settlement_outcome_diagnostics(
    monkeypatch,
) -> None:
    """Graph input receives the separately handed-off settlement rows."""

    request = _chat_request(message_id="diagnostic-seed")
    turn_clock = build_turn_clock(request.local_timestamp or None)
    future = asyncio.get_running_loop().create_future()

    class _CapturingGraph:
        """Capture the initial graph state while returning a terminal result."""

        def __init__(self) -> None:
            self.states: list[dict[str, object]] = []

        async def ainvoke(self, state: dict[str, object]) -> dict[str, object]:
            self.states.append(state)
            return {
                "should_respond": True,
                "use_reply_feature": False,
                "final_dialog": [],
                "future_promises": [],
                "consolidation_state": {},
            }

    graph = _CapturingGraph()
    row = _attempt_diagnostic("settled_relevance_deterministic_degraded")
    item = QueuedChatItem(
        sequence=1,
        request=request,
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_timestamp=turn_clock["local_timestamp"],
        local_time_context=turn_clock["local_time_context"],
        future=future,
        conversation_row_id="row-diagnostic-seed",
        resolved_message_envelope=request.message_envelope.model_dump(),
        llm_trace_id="trace-diagnostic-seed",
    )

    _patch_chat_dependencies(monkeypatch, graph)
    monkeypatch.setattr(
        service,
        "build_interaction_style_context",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        service,
        "_settle_runtime_episode_trace",
        AsyncMock(return_value={"terminal_status": "completed_private"}),
    )
    monkeypatch.setattr(
        service,
        "_persist_post_turn_lifecycle_record",
        AsyncMock(),
    )

    await service._process_queued_chat_item(
        item,
        settled_decision={
            "response_action": "proceed",
            "reason_to_respond": "grounded request",
            "use_reply_feature": False,
            "channel_topic": "",
            "indirect_speech_context": "",
        },
        settled_attempt_diagnostics=[row],
    )

    assert len(graph.states) == 1
    assert graph.states[0]["attempt_diagnostics"] == [row]
    assert future.result().messages == []


def test_brain_response_contract_uses_canonical_cognition_observation() -> None:
    """Chat and latest contracts expose the typed observation directly."""

    from kazusa_ai_chatbot.brain_service.contracts import (
        ChatResponse,
        OpsLatestCognitionGraphResponse,
    )

    observation = _observation()
    response = ChatResponse(cognition_graph=observation)
    latest = OpsLatestCognitionGraphResponse(
        cognition_graph=observation,
        self_cognition_graph=None,
    )
    assert response.cognition_graph is observation
    assert latest.cognition_graph is observation
