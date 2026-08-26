"""Deterministic ownership checks for Brain cognition observations."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

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
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    record_v2_shared_memory_prewarm_checkpoint,
    reset_v2_attempt_ledger,
)
from tests.test_service_background_consolidation import (
    _chat_http_request,
    _chat_request,
    _patch_chat_dependencies,
    _reset_queue_state,
)
from tests.test_shared_memory_prewarm import _ready_outcome

_NOW = datetime(2026, 8, 26, 0, 0, tzinfo=timezone.utc)


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
