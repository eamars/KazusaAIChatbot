"""Protected outer parent-recovery capsule contracts."""

from __future__ import annotations

import asyncio

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
)
from kazusa_ai_chatbot.llm_tracing import guardrail_capsule


def _error() -> CognitionExecutionError:
    """Build one bounded trigger error without model content."""

    return CognitionExecutionError(
        "goal bid exhausted",
        error_code="goal_bid_structure_exhausted",
        branch_id="ordinary_response",
        stage="goal_cognition",
        attempt_count=3,
        safe_checkpoint="pre_state_commit",
        retryable=False,
    )


def test_guardrail_capsule_exposes_owned_contract() -> None:
    """The outer capsule module exposes its protected writer contract."""

    assert hasattr(guardrail_capsule, "begin_guardrail_capsule")
    assert hasattr(guardrail_capsule, "finish_guardrail_capsule")
    assert (
        guardrail_capsule.GUARDRAIL_CAPSULE_WRITE_TIMEOUT_SECONDS
        == 0.25
    )


@pytest.mark.asyncio
async def test_guardrail_capsule_contains_only_bounded_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outer persistence contains coordinates but no raw cognition payload."""

    written: list[dict[str, object]] = []
    persisted = asyncio.Event()

    async def insert_step(document: dict[str, object]) -> str:
        written.append(document)
        persisted.set()
        return str(document["step_id"])

    monkeypatch.setattr(
        guardrail_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(
        guardrail_capsule.db_llm_tracing,
        "insert_trace_step",
        insert_step,
    )
    digest = "a" * 64
    session = guardrail_capsule.begin_guardrail_capsule(
        trace_id="trace-guardrail",
        scope="persona_stage_1",
        cycle_index=0,
        checkpoint_sha256=digest,
    )
    guardrail_capsule.record_guardrail_trigger(session, error=_error())
    invocation_id = guardrail_capsule.finish_guardrail_capsule(
        session,
        coordinator_snapshot={
            "checkpoint_sha256": digest,
            "parent_recovery_disposition": "recovered",
        },
        attempt_ledger={
            "schema_version": "cognition_attempt_ledger.v2",
            "epochs": [{
                "epoch": 0,
                "attempts": [{
                    "stage": "goal_bid_structure",
                    "branch_id": "ordinary_response",
                    "attempt_disposition": "exhausted",
                }],
                "branch_dispositions": [],
            }],
            "parent_recovery": {
                "disposition": "recovered",
                "claimed_by": "parent_checkpoint",
                "epoch": 1,
            },
        },
        disposition="recovered",
    )
    await asyncio.wait_for(persisted.wait(), timeout=1)

    assert written
    capsule = written[0]["capsule"]
    assert capsule["schema_version"] == (
        "cognition_parent_guardrail_capsule.v1"
    )
    assert capsule["guardrail_invocation_id"] == invocation_id
    assert capsule["checkpoint_sha256"] == digest
    assert capsule["trigger"] == {
        "error_code": "goal_bid_structure_exhausted",
        "stage": "goal_cognition",
        "branch_id": "ordinary_response",
        "attempt_count": 3,
    }
    assert capsule["parent_recovery"] == {
        "disposition": "recovered",
        "claimed_by": "parent_checkpoint",
        "epoch": 1,
        "max_replays": 1,
    }
    assert "input_payload" not in capsule
    assert "raw_response_text" not in repr(capsule)
    assert "prompt" not in repr(capsule).lower()
    assert "credentials" not in repr(capsule).lower()


@pytest.mark.asyncio
async def test_guardrail_capsule_projects_adversarial_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchecked caller fields cannot enter the protected outer schema."""

    written: list[dict[str, object]] = []
    persisted = asyncio.Event()

    async def insert_step(document: dict[str, object]) -> str:
        written.append(document)
        persisted.set()
        return str(document["step_id"])

    monkeypatch.setattr(
        guardrail_capsule,
        "LLM_TRACE_CAPTURE_MODE",
        "metadata",
    )
    monkeypatch.setattr(
        guardrail_capsule.db_llm_tracing,
        "insert_trace_step",
        insert_step,
    )
    session_digest = "a" * 64
    session = guardrail_capsule.begin_guardrail_capsule(
        trace_id="trace-adversarial",
        scope="persona_stage_1",
        cycle_index=0,
        checkpoint_sha256=session_digest,
    )
    malicious_error = CognitionExecutionError(
        "internal detail",
        error_code="raw response payload",
        branch_id="raw response payload",
        stage="raw response payload",
        attempt_count=1000,
        safe_checkpoint="pre_state_commit",
        retryable=False,
    )
    guardrail_capsule.record_guardrail_trigger(
        session,
        error=malicious_error,
    )
    guardrail_capsule.finish_guardrail_capsule(
        session,
        coordinator_snapshot={
            "checkpoint_sha256": "b" * 64,
            "raw_response_text": "RAW_PAYLOAD",
        },
        attempt_ledger={
            "schema_version": "cognition_attempt_ledger.v2",
            "epochs": [{
                "epoch": 0,
                "attempts": [{
                    "raw_response_text": "RAW_PAYLOAD",
                }, {
                    "cognition_invocation_id": "invocation",
                    "graph_attempt": 99,
                    "branch_id": "ordinary_response",
                    "producing_stage": "goal_bid_structure",
                    "local_attempt": 1,
                    "cumulative_producer_attempt": 1,
                    "configured_limit": 3,
                    "epoch": 0,
                    "attempt_disposition": "started",
                }, {
                    "cognition_invocation_id": "invocation",
                    "graph_attempt": 1,
                    "branch_id": "ordinary_response",
                    "producing_stage": "goal_bid_structure",
                    "local_attempt": 1,
                    "cumulative_producer_attempt": 1,
                    "configured_limit": 3,
                    "epoch": 1,
                    "attempt_disposition": "started",
                }],
                "branch_dispositions": [],
            }],
            "parent_recovery": {
                "disposition": "recovered",
                "claimed_by": "parent_checkpoint",
                "epoch": 1,
                "checkpoint_sha256": session_digest,
                "max_replays": 1,
                "raw_response_text": "RAW_PAYLOAD",
            },
        },
        disposition="recovered",
    )
    await asyncio.wait_for(persisted.wait(), timeout=1)

    capsule = written[0]["capsule"]
    assert set(capsule) == {
        "schema_version",
        "trace_id",
        "guardrail_invocation_id",
        "scope",
        "cycle_index",
        "checkpoint_sha256",
        "trigger",
        "parent_recovery",
        "attempt_ledger",
    }
    assert capsule["checkpoint_sha256"] == session_digest
    assert capsule["trigger"] == {
        "error_code": "",
        "stage": "",
        "branch_id": "",
        "attempt_count": 32,
    }
    assert capsule["attempt_ledger"] == {
        "schema_version": "cognition_attempt_ledger.v2",
        "epochs": [{
            "epoch": 0,
            "attempts": [],
            "branch_dispositions": [],
        }],
        "parent_recovery": {},
    }
    assert "RAW_PAYLOAD" not in repr(capsule)
    assert "raw_response_text" not in repr(capsule)
