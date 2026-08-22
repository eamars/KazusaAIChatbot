"""Deterministic tests for Cognition V3 lane coordination."""

from __future__ import annotations

import asyncio
import time

import pytest

from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    V2AttemptBudgetExhausted,
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    record_v2_attempt_disposition,
    reserve_v2_model_attempt,
    reset_v2_attempt_ledger,
)
from kazusa_ai_chatbot.cognition_core_v3 import lane
from kazusa_ai_chatbot.llm_interface import LLMCallConfig


def _config(*, route_name: str, model: str) -> LLMCallConfig:
    """Build one non-thinking lane config with a distinct route identity."""

    config = LLMCallConfig(
        stage_name="cognition_core_v3.test_lane",
        route_name=route_name,
        base_url="http://lane.test/v1",
        api_key="test-key",
        model=model,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_completion_tokens=8192,
        presence_penalty=None,
    )
    return config


@pytest.mark.asyncio
async def test_primary_lane_is_fifo_single_owner_and_sidecar_cannot_interleave() -> None:
    """Primary claims serialize in ticket order while sidecar stays separate."""

    llm = object()
    chain_config = _config(route_name="CHAIN", model="chain-model")
    sidecar_config = _config(route_name="SIDECAR", model="sidecar-model")
    primary = lane.primary_lane_coordinator(llm, chain_config)
    assert lane.primary_lane_coordinator(llm, chain_config) is primary
    sidecar = lane.sidecar_lane_coordinator(llm, sidecar_config)
    sidecar_state = lane.SidecarInvocationState()
    assert primary.identity != sidecar.identity

    entered_first = asyncio.Event()
    release_first = asyncio.Event()
    start_order: list[int] = []
    active_count = 0
    maximum_active = 0

    async def run_primary(index: int) -> None:
        nonlocal active_count, maximum_active
        async with primary.claim():
            start_order.append(index)
            active_count += 1
            maximum_active = max(maximum_active, active_count)
            if index == 0:
                entered_first.set()
                await release_first.wait()
            await asyncio.sleep(0)
            active_count -= 1

    first = asyncio.create_task(run_primary(0))
    await asyncio.wait_for(entered_first.wait(), timeout=1.0)
    second = asyncio.create_task(run_primary(1))
    await asyncio.sleep(0)
    third = asyncio.create_task(run_primary(2))
    await asyncio.sleep(0)

    try:
        async with sidecar.claim(
            stream_kind="json_repair",
            invocation_state=sidecar_state,
        ) as sidecar_claim:
            assert sidecar_claim.in_flight_at_start == 1
            assert start_order == [0]
    finally:
        release_first.set()
        await asyncio.gather(first, second, third)

    assert start_order == [0, 1, 2]
    assert maximum_active == 1
    assert primary.in_flight == 0

    async with primary.claim():
        with pytest.raises(lane.LaneContractError, match="reacquire"):
            async with primary.claim():
                pass

    deadline_owner_entered = asyncio.Event()
    release_deadline_owner = asyncio.Event()

    async def hold_primary_for_deadline() -> None:
        async with primary.claim():
            deadline_owner_entered.set()
            await release_deadline_owner.wait()

    deadline_owner = asyncio.create_task(hold_primary_for_deadline())
    await asyncio.wait_for(deadline_owner_entered.wait(), timeout=1.0)

    async def claim_before_expired_deadline() -> None:
        async with primary.claim(
            deadline_monotonic=time.monotonic() + 0.05,
        ):
            raise AssertionError("An expired queued ticket entered the lane")

    expired_waiter = asyncio.create_task(claim_before_expired_deadline())
    with pytest.raises(lane.LaneDeadlineError, match="expired while queued"):
        await expired_waiter

    deadline_follower_entered = asyncio.Event()

    async def run_deadline_follower() -> None:
        async with primary.claim():
            deadline_follower_entered.set()

    deadline_follower = asyncio.create_task(run_deadline_follower())
    await asyncio.sleep(0)
    release_deadline_owner.set()
    await asyncio.gather(deadline_owner, deadline_follower)
    assert deadline_follower_entered.is_set()


@pytest.mark.asyncio
async def test_sidecar_stream_serializes_l1_repair_and_authorization_with_fixed_caps() -> None:
    """Sidecar requests share one stream and logical producers keep fixed caps."""

    coordinator = lane.sidecar_lane_coordinator(
        object(),
        _config(route_name="SIDECAR", model="sidecar-model"),
    )
    invocation = lane.SidecarInvocationState()
    release_first = asyncio.Event()
    entered_first = asyncio.Event()
    start_order: list[str] = []
    active_count = 0
    maximum_active = 0

    async def run_stream(stream_kind: lane.SidecarStreamKind) -> None:
        nonlocal active_count, maximum_active
        async with coordinator.claim(
            stream_kind=stream_kind,
            invocation_state=invocation,
        ):
            start_order.append(stream_kind)
            active_count += 1
            maximum_active = max(maximum_active, active_count)
            if stream_kind == "l1":
                entered_first.set()
                await release_first.wait()
            await asyncio.sleep(0)
            active_count -= 1

    l1_task = asyncio.create_task(run_stream("l1"))
    await asyncio.wait_for(entered_first.wait(), timeout=1.0)
    repair_task = asyncio.create_task(run_stream("json_repair"))
    await asyncio.sleep(0)
    action_task = asyncio.create_task(run_stream("action_authorization"))
    await asyncio.sleep(0)
    release_first.set()
    await asyncio.gather(l1_task, repair_task, action_task)

    assert start_order == ["l1", "json_repair", "action_authorization"]
    assert maximum_active == 1
    assert coordinator.maximum_in_flight == 1
    stream_diagnostics = invocation.diagnostics()
    assert stream_diagnostics["l1_stream_count"] == 1
    assert stream_diagnostics["json_repair_call_count"] == 1
    assert stream_diagnostics["action_auth_attempt_count"] == 1
    assert stream_diagnostics["resolver_auth_attempt_count"] == 0
    assert stream_diagnostics["sidecar_queue_wait_ms_total"] >= 0
    assert stream_diagnostics["sidecar_max_in_flight"] == 1
    assert stream_diagnostics["l1_preempted_by_repair"] is False
    assert stream_diagnostics["sidecar_cancellation_count"] == 0

    admissions = lane.SidecarAdmissionLedger()
    admissions.reserve_l1()
    with pytest.raises(lane.SidecarAdmissionError, match="one L1"):
        admissions.reserve_l1()

    shared_ledger = create_v2_attempt_ledger("lane-test-invocation")
    ledger_token = bind_v2_attempt_ledger(shared_ledger, graph_attempt=1)
    try:
        planning_attempt = reserve_v2_model_attempt(
            stage="action_planning",
            branch_id="cycle:0",
            local_attempt=1,
        )
        admissions.reserve_json_repair(
            candidate_id="candidate-1",
            attempt_coordinates=planning_attempt,
        )
        with pytest.raises(lane.SidecarAdmissionError, match="raw candidate"):
            admissions.reserve_json_repair(
                candidate_id="candidate-1",
                attempt_coordinates=planning_attempt,
            )
        with pytest.raises(lane.SidecarAdmissionError, match="one JSON repair"):
            admissions.reserve_json_repair(
                candidate_id="candidate-2",
                attempt_coordinates=planning_attempt,
            )
        record_v2_attempt_disposition(
            planning_attempt,
            disposition="accepted",
        )
        with pytest.raises(lane.SidecarAdmissionError, match="live"):
            admissions.reserve_json_repair(
                candidate_id="candidate-3",
                attempt_coordinates=planning_attempt,
            )

        action_attempt_numbers: list[int] = []
        for local_attempt in range(1, 4):
            action_attempt = reserve_v2_model_attempt(
                stage="action_authorization",
                branch_id="cycle:0",
                local_attempt=local_attempt,
            )
            action_attempt_numbers.append(
                admissions.reserve_action_authorization(
                    cycle_index=0,
                    attempt_coordinates=action_attempt,
                )
            )
            action_disposition = (
                "regenerate" if local_attempt < 3 else "exhausted"
            )
            record_v2_attempt_disposition(
                action_attempt,
                disposition=action_disposition,
            )
        assert action_attempt_numbers == [1, 2, 3]
        with pytest.raises(
            V2AttemptBudgetExhausted,
            match="action_authorization",
        ):
            reserve_v2_model_attempt(
                stage="action_authorization",
                branch_id="cycle:0",
                local_attempt=4,
            )
        with pytest.raises(lane.SidecarAdmissionError, match="X1 must finish"):
            admissions.reserve_resolver_authorization(
                cycle_index=0,
                attempt_coordinates={},
            )

        admissions.finish_action_authorization(cycle_index=0)
        resolver_attempt_numbers: list[int] = []
        resolver_attempt = None
        for local_attempt in range(1, 4):
            resolver_attempt = reserve_v2_model_attempt(
                stage="resolver_authorization",
                branch_id="cycle:0",
                local_attempt=local_attempt,
            )
            resolver_attempt_numbers.append(
                admissions.reserve_resolver_authorization(
                    cycle_index=0,
                    attempt_coordinates=resolver_attempt,
                )
            )
            resolver_disposition = (
                "regenerate" if local_attempt < 3 else "exhausted"
            )
            record_v2_attempt_disposition(
                resolver_attempt,
                disposition=resolver_disposition,
            )
        assert resolver_attempt_numbers == [1, 2, 3]
        with pytest.raises(
            V2AttemptBudgetExhausted,
            match="resolver_authorization",
        ):
            reserve_v2_model_attempt(
                stage="resolver_authorization",
                branch_id="cycle:0",
                local_attempt=4,
            )
        admissions.finish_resolver_authorization(cycle_index=0)
        with pytest.raises(lane.SidecarAdmissionError, match="finished"):
            admissions.reserve_resolver_authorization(
                cycle_index=0,
                attempt_coordinates=resolver_attempt,
            )
    finally:
        reset_v2_attempt_ledger(ledger_token)


@pytest.mark.asyncio
async def test_l1_repair_preemption_and_cancellation_release_sidecar_fifo() -> None:
    """Repair drains a cancelled L1 claim and active cancellation releases FIFO."""

    coordinator = lane.sidecar_lane_coordinator(
        object(),
        _config(route_name="SIDECAR", model="sidecar-model"),
    )
    invocation = lane.SidecarInvocationState()
    l1_entered = asyncio.Event()
    l1_finished = asyncio.Event()

    async def run_l1() -> None:
        try:
            async with coordinator.claim(
                stream_kind="l1",
                invocation_state=invocation,
            ):
                l1_entered.set()
                await asyncio.Event().wait()
        finally:
            l1_finished.set()

    l1_task = asyncio.create_task(run_l1())
    invocation.register_l1_task(l1_task)
    await asyncio.wait_for(l1_entered.wait(), timeout=1.0)

    assert await invocation.preempt_l1_for_repair()
    assert l1_task.cancelled()
    assert l1_finished.is_set()
    assert invocation.l1_preempted_by_repair
    assert invocation.cancellation_count == 1
    assert coordinator.in_flight == 0

    repair_entered = asyncio.Event()

    async def run_repair() -> None:
        async with coordinator.claim(
            stream_kind="json_repair",
            invocation_state=invocation,
        ):
            repair_entered.set()
            await asyncio.Event().wait()

    repair_task = asyncio.create_task(run_repair())
    await asyncio.wait_for(repair_entered.wait(), timeout=1.0)

    async def queue_then_cancel() -> None:
        async with coordinator.claim(
            stream_kind="resolver_authorization",
            invocation_state=invocation,
        ):
            raise AssertionError("A cancelled queued ticket entered the lane")

    cancelled_waiter = asyncio.create_task(queue_then_cancel())
    await asyncio.sleep(0)
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter

    follower_entered = asyncio.Event()

    async def run_follower() -> None:
        async with coordinator.claim(
            stream_kind="action_authorization",
            invocation_state=invocation,
        ):
            follower_entered.set()

    follower_task = asyncio.create_task(run_follower())
    await asyncio.sleep(0)
    assert not follower_entered.is_set()

    repair_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await repair_task
    await asyncio.wait_for(follower_entered.wait(), timeout=1.0)
    await follower_task

    assert coordinator.in_flight == 0
    diagnostics = invocation.diagnostics()
    assert diagnostics["l1_stream_count"] == 1
    assert diagnostics["json_repair_call_count"] == 1
    assert diagnostics["action_auth_attempt_count"] == 1
    assert diagnostics["resolver_auth_attempt_count"] == 0
    assert diagnostics["sidecar_max_in_flight"] == 1
    assert diagnostics["l1_preempted_by_repair"] is True
    assert diagnostics["sidecar_cancellation_count"] == 3


@pytest.mark.asyncio
async def test_lane_owner_cleanup_survives_second_cancellation() -> None:
    """A second cancellation cannot strand the FIFO owner or its waiters."""

    fifo = lane._FifoLane((id(object()), "http://lane.test/v1", "model"))
    owner_entered = asyncio.Event()
    follower_entered = asyncio.Event()

    async def run_owner() -> None:
        async with fifo.claim(
            stream_kind="primary",
            deadline_monotonic=None,
        ):
            owner_entered.set()
            await asyncio.Event().wait()

    async def run_follower() -> None:
        async with fifo.claim(
            stream_kind="primary",
            deadline_monotonic=None,
        ):
            follower_entered.set()

    owner_task = asyncio.create_task(run_owner())
    await asyncio.wait_for(owner_entered.wait(), timeout=1.0)
    follower_task = asyncio.create_task(run_follower())
    await asyncio.sleep(0)

    await fifo._condition.acquire()
    owner_task.cancel()
    await asyncio.sleep(0)
    owner_task.cancel()
    fifo._condition.release()

    with pytest.raises(asyncio.CancelledError):
        await owner_task
    await asyncio.wait_for(follower_entered.wait(), timeout=1.0)
    await follower_task

    assert fifo.in_flight == 0
    assert fifo._owner is None


@pytest.mark.asyncio
async def test_failing_l1_task_is_advisory_during_repair_preemption() -> None:
    """A completed L1 exception becomes a bounded warning instead of raising."""

    invocation = lane.SidecarInvocationState()

    async def fail_l1() -> None:
        raise ValueError("provider detail must stay out of diagnostics")

    l1_task = asyncio.create_task(fail_l1())
    invocation.register_l1_task(l1_task)
    await asyncio.sleep(0)

    assert await invocation.preempt_l1_for_repair() is False
    assert invocation.l1_preempted_by_repair is False
    assert invocation.consume_l1_warning() == "sidecar_l1_unavailable"
    assert invocation.consume_l1_warning() is None


@pytest.mark.asyncio
async def test_repair_preemption_propagates_outer_cancellation_after_l1_drain() -> None:
    """Repair cleanup drains L1 before returning an outer cancellation."""

    invocation = lane.SidecarInvocationState()
    cancellation_seen = asyncio.Event()
    release_l1_cleanup = asyncio.Event()

    async def run_l1() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_l1_cleanup.wait()
            raise

    l1_task = asyncio.create_task(run_l1())
    invocation.register_l1_task(l1_task)
    preempt_task = asyncio.create_task(invocation.preempt_l1_for_repair())
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1.0)

    preempt_task.cancel()
    await asyncio.sleep(0)
    assert not preempt_task.done()
    release_l1_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await preempt_task
    assert l1_task.cancelled()
    assert invocation.l1_preempted_by_repair is True
    assert invocation.cancellation_count == 1


@pytest.mark.asyncio
async def test_repair_cancellation_failure_keeps_preemption_and_warning() -> None:
    """A repair-caused L1 failure remains advisory and records preemption."""

    invocation = lane.SidecarInvocationState()
    started = asyncio.Event()

    async def fail_after_repair_cancellation() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as exc:
            raise ValueError("provider detail stays outside diagnostics") from exc

    l1_task = asyncio.create_task(fail_after_repair_cancellation())
    invocation.register_l1_task(l1_task)
    await asyncio.wait_for(started.wait(), timeout=1.0)

    assert await invocation.preempt_l1_for_repair() is True
    assert invocation.l1_preempted_by_repair is True
    assert invocation.consume_l1_warning() == "sidecar_l1_unavailable"
