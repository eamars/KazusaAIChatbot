"""Deterministic tests for the V3 bounded parallel chain executor."""

from __future__ import annotations

import asyncio

import pytest

from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
    EXHAUSTION_FAILURE_CLASS,
)
from kazusa_ai_chatbot.cognition_core_v3.execution import (
    AttemptLedger,
    ChainTaskSpec,
    ExecutorContractError,
    StageAttemptOutcome,
    start_wave,
    validate_chain_spec,
)


def _accepting_producer(log: list[str]):
    async def producer(ctx):
        log.append(f"{ctx.chain_name}:{ctx.stage_name}:start:{ctx.attempt_number}")
        await asyncio.sleep(0)
        return StageAttemptOutcome(True, {"stage": ctx.stage_name}, f"summary {ctx.stage_name}", None)

    return producer


def test_executor_runs_parallel_chains_and_serial_registered_stages():
    async def scenario() -> list[str]:
        log: list[str] = []
        specs = [
            ChainTaskSpec(
                "causal_normative",
                ("event_agency", "moral_identity"),
                {
                    "event_agency": _accepting_producer(log),
                    "moral_identity": _accepting_producer(log),
                },
            ),
            ChainTaskSpec(
                "relationship",
                ("relationship_social",),
                {"relationship_social": _accepting_producer(log)},
            ),
        ]
        handle = start_wave(specs, ledger=AttemptLedger({
            "event_agency": 3,
            "moral_identity": 3,
            "relationship_social": 3,
        }))
        result = await handle.complete()

        assert set(result.outcomes) == {"causal_normative", "relationship"}
        assert not result.cancelled_chains and not result.failed_chains

        causal_order = [entry for entry in log if entry.startswith("causal_normative:")]
        assert causal_order == [
            "causal_normative:event_agency:start:1",
            "causal_normative:moral_identity:start:1",
        ]

        relationship_start_index = log.index("relationship:relationship_social:start:1")
        moral_start_index = log.index("causal_normative:moral_identity:start:1")
        assert relationship_start_index < moral_start_index, (
            "Sister chains must overlap in the wave; a serial schedule would start "
            "the relationship chain only after both causal stages completed"
        )

        return list(result.outcomes["causal_normative"].results)

    results = asyncio.run(scenario())
    assert [stage.stage_name for stage in results] == ["event_agency", "moral_identity"]
    assert all(stage.accepted for stage in results)

    with pytest.raises(ExecutorContractError, match="exact registry order"):
        validate_chain_spec(ChainTaskSpec("causal_normative", ("moral_identity", "event_agency"), {}))

    with pytest.raises(ExecutorContractError, match="Unknown registered chain"):
        validate_chain_spec(ChainTaskSpec("invented_chain", ("stage_a",), {}))

    async def producer(ctx):  # noqa: ANN001 - deterministic patched producer
        return StageAttemptOutcome(True, {}, "summary", None)

    with pytest.raises(ExecutorContractError, match="no producer bound"):
        validate_chain_spec(
            ChainTaskSpec("causal_normative", ("event_agency", "moral_identity"), {"event_agency": producer})
        )


def test_executor_preserves_global_attempt_caps():
    async def scenario():
        ledger = AttemptLedger({"event_agency": 2, "moral_identity": 3})
        event_agency_calls = 0

        async def always_structurally_invalid(ctx):
            nonlocal event_agency_calls
            event_agency_calls += 1
            await asyncio.sleep(0)
            return StageAttemptOutcome(False, None, None, "structural_contract")

        async def accepting(ctx):
            await asyncio.sleep(0)
            return StageAttemptOutcome(True, {"ok": ctx.stage_name}, f"summary {ctx.stage_name}", None)

        spec = ChainTaskSpec(
            "causal_normative",
            ("event_agency", "moral_identity"),
            {"event_agency": always_structurally_invalid, "moral_identity": accepting},
        )
        handle = start_wave([spec], ledger=ledger)
        result = await handle.complete()
        return result, ledger, event_agency_calls

    result, ledger, event_agency_calls = asyncio.run(scenario())

    assert event_agency_calls == 2

    outcome = result.outcomes["causal_normative"]
    exhausted_stage, continued_stage = outcome.results

    assert not exhausted_stage.accepted
    assert exhausted_stage.failure is not None
    assert exhausted_stage.failure.failure_class == EXHAUSTION_FAILURE_CLASS
    assert exhausted_stage.failure.error_code == APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
    assert exhausted_stage.failure.repair_attempted is True

    assert ledger.attempts_used("event_agency") == 2
    assert continued_stage.accepted
    assert continued_stage.stage_name == "moral_identity"
    assert ledger.attempts_used("moral_identity") == 1


def test_executor_cancels_owned_tasks_without_partial_effects():
    async def scenario():
        release = asyncio.Event()
        ledger = AttemptLedger({"event_agency": 2, "moral_identity": 2})

        async def blocking_on_first_stage(ctx):
            if ctx.stage_name == "event_agency":
                await release.wait()
            return StageAttemptOutcome(True, {}, "", None)

        spec = ChainTaskSpec(
            "causal_normative",
            ("event_agency", "moral_identity"),
            {"event_agency": blocking_on_first_stage, "moral_identity": blocking_on_first_stage},
        )
        handle = start_wave([spec], ledger=ledger)

        for _ in range(3):
            await asyncio.sleep(0)
        assert ledger.attempts_used("event_agency") == 1
        assert ledger.attempts_used("moral_identity") == 0

        handle.cancel()
        return await handle.complete(), ledger

    result, ledger = asyncio.run(scenario())

    assert "causal_normative" in result.cancelled_chains
    assert "causal_normative" not in result.outcomes
    assert not result.failed_chains

    # No partial effects: the second stage never ran and no chain result was
    # materialized; only the completed first-stage reservation remains.
    assert ledger.attempts_used("event_agency") == 1
    assert ledger.attempts_used("moral_identity") == 0
