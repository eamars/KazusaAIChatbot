"""Deterministic proofs for the bounded Cognition V3 stage runner."""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CANONICAL_A1_FAMILIES,
)
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    CanonicalContractError,
    _run_cognition_stage,
    run_cognition,
)
from kazusa_ai_chatbot.cognition_shared.contracts import CognitionExecutionError
from tests.unit.cognition_core_v3.test_handleless_contract import (
    _FourStageInvoker,
    _input,
    _services,
)


class _SequencedInvoker:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0
        self.messages: list[object] = []

    async def ainvoke(self, messages: object, *, config: object) -> object:
        self.calls += 1
        self.messages.append(messages)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(content=json.dumps(outcome, ensure_ascii=False))


def _trace_capture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    async def record_trace_step(**kwargs: object) -> dict[str, object]:
        rows.append(kwargs)
        return {
            "accepted": True,
            "trace_id": "deterministic-trace",
            "status": "recorded",
            "reason": "",
        }

    monkeypatch.setattr(
        facade_module.llm_tracing,
        "record_llm_trace_step",
        record_trace_step,
    )
    return rows


def _valid_stage_value() -> dict[str, str]:
    return {"value": "accepted"}


def _stage_validator(value: object) -> object:
    if not isinstance(value, dict) or value.get("value") != "accepted":
        raise CanonicalContractError("value must be accepted")
    return value


async def _run_simple_stage(
    invoker: _SequencedInvoker,
    *,
    deadline_monotonic: float | None = None,
) -> object:
    return await _run_cognition_stage(
        services=_services(invoker),
        stage="P",
        packet={"stage": "P", "output_contract": {"value": "text"}},
        validator=_stage_validator,
        deadline_monotonic=(
            time.monotonic() + 60
            if deadline_monotonic is None
            else deadline_monotonic
        ),
    )


@pytest.mark.asyncio
async def test_stage_regenerates_with_exact_contract_error_and_rejected_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_rows = _trace_capture(monkeypatch)
    parser_kwargs: list[dict[str, object]] = []
    original_parser = facade_module.parse_llm_json_output

    def parse_stage_output(
        raw_output: str,
        **kwargs: object,
    ) -> dict:
        parser_kwargs.append(kwargs)
        return original_parser(raw_output, **kwargs)

    monkeypatch.setattr(
        facade_module,
        "parse_llm_json_output",
        parse_stage_output,
    )
    invoker = _SequencedInvoker([
        {"value": "rejected"},
        _valid_stage_value(),
    ])

    result = await _run_simple_stage(invoker)

    assert result == _valid_stage_value()
    assert invoker.calls == 2
    first_packet = json.loads(invoker.messages[0][1].content)
    repair_packet = json.loads(invoker.messages[1][1].content)
    assert set(repair_packet) == set(first_packet) | {"contract_repair"}
    assert set(repair_packet["contract_repair"]) == {
        "reason",
        "contract_error",
        "invalid_candidate",
    }
    assert repair_packet["contract_repair"]["reason"] == "contract_error"
    assert repair_packet["contract_repair"]["contract_error"] == (
        "value must be accepted"
    )
    assert '"value": "rejected"' in repair_packet["contract_repair"][
        "invalid_candidate"
    ]
    assert invoker.messages[0][0].content == invoker.messages[1][0].content
    assert "contract_repair" in invoker.messages[0][0].content
    assert "repair_instruction" not in invoker.messages[1][1].content
    assert [row["attempt_index"] for row in trace_rows] == [1, 2]
    assert [row["parse_status"] for row in trace_rows] == [
        "contract_error",
        "succeeded",
    ]
    assert [row["status"] for row in trace_rows] == ["failed", "succeeded"]
    assert parser_kwargs == [{}, {}]


@pytest.mark.asyncio
async def test_object_valued_response_goal_converges_after_one_feedback_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_rows = _trace_capture(monkeypatch)
    normalization_events: list[dict[str, object]] = []

    async def record_contract_event(**kwargs: object) -> dict[str, object]:
        normalization_events.append(kwargs)
        return {
            "accepted": True,
            "trace_id": "deterministic-trace",
            "status": "recorded",
            "reason": "",
        }

    monkeypatch.setattr(
        facade_module.event_logging,
        "record_model_contract_event",
        record_contract_event,
    )

    class _ObjectResponseGoalInvoker(_FourStageInvoker):
        def __init__(self) -> None:
            super().__init__()
            self.returned_object_goal = False
            self.returned_long_monologue = False

        async def ainvoke(self, messages: object, *, config: object) -> object:
            response = await super().ainvoke(messages, config=config)
            if (
                config.stage_name.endswith(".G")
                and not self.returned_long_monologue
            ):
                self.returned_long_monologue = True
                value = json.loads(response.content)
                value["private_monologue"] = "a" * 601
                response = SimpleNamespace(
                    content=json.dumps(value, ensure_ascii=False)
                )
            if (
                config.stage_name.endswith(".P")
                and not self.returned_object_goal
            ):
                self.returned_object_goal = True
                value = json.loads(response.content)
                value["response_goal"] = {"not": "text"}
                response = SimpleNamespace(
                    content=json.dumps(value, ensure_ascii=False)
                )
            return response

    invoker = _ObjectResponseGoalInvoker()
    output = await run_cognition(_input(), _services(invoker))

    assert output["response_plan"]["response_goal"] == "ask for clarification"
    assert output["private_monologue"] == "a" * 600
    assert invoker.calls == ["A1", "A2", "G", "P", "P"]
    g_rows = [
        row for row in trace_rows
        if row["stage_name"] == "cognition_core_v3.G"
    ]
    assert [row["parse_status"] for row in g_rows] == ["normalized"]
    assert [event["violation_kind"] for event in normalization_events] == [
        "private_monologue_clamped",
    ]
    p_rows = [
        row for row in trace_rows
        if row["stage_name"] == "cognition_core_v3.P"
    ]
    assert [row["attempt_index"] for row in p_rows] == [1, 2]
    assert [row["parse_status"] for row in p_rows] == [
        "contract_error",
        "succeeded",
    ]
    assert [row["status"] for row in p_rows] == ["failed", "succeeded"]


@pytest.mark.asyncio
async def test_url_only_ordinary_response_goal_is_recovered_as_p_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P rejects a URL-only ordinary response goal inside the stage ladder."""

    trace_rows = _trace_capture(monkeypatch)
    invalid_plan = {
        "goal_resolution": "answerable_now",
        "response_goal": "https://unusable.example/goal",
        "action_requests": [],
        "resolver_requests": [],
        "epistemic_boundary": "Keep unsupported details uncertain.",
    }
    valid_plan = {
        **invalid_plan,
        "response_goal": "ask for clarification",
    }
    invoker = _SequencedInvoker([invalid_plan, valid_plan])

    result = await _run_cognition_stage(
        services=_services(invoker),
        stage="P",
        packet={"stage": "P", "output_contract": {}},
        validator=lambda raw: facade_module._validate_plan_stage(
            raw,
            self_cognition=False,
            capabilities={"actions": [], "resolvers": []},
            response_plan_contract_variant="fresh_ordinary",
        ),
        deadline_monotonic=time.monotonic() + 60,
    )

    assert result.response_goal == "ask for clarification"
    assert invoker.calls == 2
    repair = json.loads(invoker.messages[1][1].content)["contract_repair"]
    assert "URL" in repair["contract_error"]
    assert "repair_instruction" not in repair
    assert [row["parse_status"] for row in trace_rows] == [
        "contract_error",
        "succeeded",
    ]








@pytest.mark.asyncio
async def test_post_pending_hil_echo_regenerates_without_reopening_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-answer P retries instead of reopening human clarification."""

    trace_rows = _trace_capture(monkeypatch)
    invalid_plan = {
        "response_goal": "deliver the completed task result",
        "action_requests": [],
        "epistemic_boundary": "The evidence remains source-owned.",
        "goal_resolution": "requires_user_input",
        "resolver_requests": [{
            "capability": "human_clarification",
            "goal": "ask for an already-resolved fact",
            "reason": "the stale clarification was echoed",
        }],
    }
    canonical_plan = {
        "goal_resolution": "answerable_now",
        "response_goal": invalid_plan["response_goal"],
        "action_requests": [],
        "resolver_requests": [],
        "epistemic_boundary": invalid_plan["epistemic_boundary"],
    }
    invoker = _SequencedInvoker([invalid_plan, canonical_plan])
    token = facade_module.bind_protected_chain_records(
        run_id="post-pending-variant-recovery",
    )
    try:
        result = await _run_cognition_stage(
            services=_services(invoker),
            stage="P",
            packet={"stage": "P", "output_contract": {}},
            validator=lambda raw: facade_module._validate_plan_stage(
                raw,
                self_cognition=False,
                capabilities={"actions": [], "resolvers": []},
                response_plan_contract_variant="post_pending_resolution",
            ),
            deadline_monotonic=time.monotonic() + 60,
        )
        records = facade_module.snapshot_protected_chain_records()
    finally:
        facade_module.reset_protected_chain_records(token)

    assert result.pending_resolution is None
    assert result.pending_task_continuation is None
    assert invoker.calls == 2
    repair = json.loads(invoker.messages[1][1].content)["contract_repair"]
    assert repair["reason"] == "contract_error"
    assert repair["contract_error"] == "resolver capability is not available"
    assert json.loads(repair["invalid_candidate"]) == invalid_plan
    assert json.loads(records[0]["raw_output"]) == invalid_plan
    assert records[0]["validation_error"] == repair["contract_error"]
    assert invoker.messages[0][0].content == invoker.messages[1][0].content
    assert [row["parse_status"] for row in trace_rows] == [
        "contract_error",
        "succeeded",
    ]




@pytest.mark.asyncio
async def test_tool_result_delivery_carrier_echo_regenerates_to_result_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Result delivery retains the raw carrier echo and retries fail-closed."""

    trace_rows = _trace_capture(monkeypatch)
    invalid_plan = {
        "goal_resolution": "answerable_now",
        "response_goal": "report the bounded tool result",
        "action_requests": [],
        "resolver_requests": [],
        "epistemic_boundary": "The result remains source-owned.",
        "pending_task_continuation": None,
    }
    canonical_plan = {
        key: value
        for key, value in invalid_plan.items()
        if key != "pending_task_continuation"
    }
    invoker = _SequencedInvoker([invalid_plan, canonical_plan])
    token = facade_module.bind_protected_chain_records(
        run_id="tool-result-delivery-carrier-recovery",
    )
    try:
        result = await _run_cognition_stage(
            services=_services(invoker),
            stage="P",
            packet={"stage": "P", "output_contract": {}},
            validator=lambda raw: facade_module._validate_plan_stage(
                raw,
                self_cognition=False,
                capabilities={
                    "actions": [],
                    "resolvers": [{"capability": "approval_preparation"}],
                },
                response_plan_contract_variant="tool_result_delivery",
            ),
            deadline_monotonic=time.monotonic() + 60,
        )
        records = facade_module.snapshot_protected_chain_records()
    finally:
        facade_module.reset_protected_chain_records(token)

    assert result.resolver_requests == ()
    assert invoker.calls == 2
    repair = json.loads(invoker.messages[1][1].content)["contract_repair"]
    assert repair["reason"] == "contract_error"
    assert repair["contract_error"] == (
        "response plan: unexpected fields ['pending_task_continuation']"
    )
    assert json.loads(repair["invalid_candidate"]) == invalid_plan
    assert json.loads(records[0]["raw_output"]) == invalid_plan
    assert records[0]["validation_error"] == repair["contract_error"]
    assert invoker.messages[0][0].content == invoker.messages[1][0].content
    assert [row["parse_status"] for row in trace_rows] == [
        "contract_error",
        "succeeded",
    ]


@pytest.mark.asyncio
async def test_url_only_self_cognition_response_goal_is_recovered_as_p_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P applies the same terminal-text invariant to self-cognition plans."""

    trace_rows = _trace_capture(monkeypatch)
    invalid_response = {
        "decision": "propose_visible_reply",
        "response_goal": "https://unusable.example/self-goal",
        "reason": "current evidence supports a response",
        "cause_summary": "current evidence",
    }
    valid_response = {
        **invalid_response,
        "response_goal": "offer a grounded reply",
    }
    invoker = _SequencedInvoker([
        {
            "self_cognition_response": invalid_response,
            "epistemic_boundary": "Keep unsupported details uncertain.",
        },
        {
            "self_cognition_response": valid_response,
            "epistemic_boundary": "Keep unsupported details uncertain.",
        },
    ])

    result = await _run_cognition_stage(
        services=_services(invoker),
        stage="P",
        packet={"stage": "P", "output_contract": {}},
        validator=lambda raw: facade_module._validate_plan_stage(
            raw,
            self_cognition=True,
            capabilities={"actions": [], "resolvers": []},
            response_plan_contract_variant="fresh_ordinary",
        ),
        deadline_monotonic=time.monotonic() + 60,
    )

    assert result.response_goal == "offer a grounded reply"
    assert invoker.calls == 2
    assert [row["parse_status"] for row in trace_rows] == [
        "contract_error",
        "succeeded",
    ]


@pytest.mark.asyncio
async def test_provider_failure_consumes_one_attempt_and_regenerates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_rows = _trace_capture(monkeypatch)
    invoker = _SequencedInvoker([
        RuntimeError("provider unavailable"),
        _valid_stage_value(),
    ])

    result = await _run_simple_stage(invoker)

    assert result == _valid_stage_value()
    assert invoker.calls == 2
    repair_packet = json.loads(invoker.messages[1][1].content)
    assert repair_packet["contract_repair"]["reason"] == "provider_error"
    assert repair_packet["contract_repair"]["contract_error"] == ""
    assert repair_packet["contract_repair"]["invalid_candidate"] == ""
    assert [row["parse_status"] for row in trace_rows] == [
        "provider_error",
        "succeeded",
    ]
    assert [row["status"] for row in trace_rows] == ["failed", "succeeded"]


@pytest.mark.asyncio
async def test_stage_exhaustion_raises_retryable_pre_commit_execution_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_rows = _trace_capture(monkeypatch)
    invoker = _SequencedInvoker([
        {"value": "rejected"},
        {"value": "rejected"},
        {"value": "rejected"},
    ])

    with pytest.raises(CognitionExecutionError) as error:
        await _run_simple_stage(invoker)

    assert invoker.calls == 3
    assert error.value.error_code == "cognition_p_contract_exhausted"
    assert error.value.stage == "cognition_core_v3.P"
    assert error.value.attempt_count == 3
    assert error.value.safe_checkpoint == "pre_state_commit"
    assert error.value.retryable is True
    assert [row["attempt_index"] for row in trace_rows] == [1, 2, 3]
    assert all(row["status"] == "failed" for row in trace_rows)


@pytest.mark.asyncio
async def test_rejected_attempt_records_contract_fault_before_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_rows = _trace_capture(monkeypatch)
    invoker = _SequencedInvoker([
        {"value": "rejected"},
        _valid_stage_value(),
    ])

    token = facade_module.bind_protected_chain_records(run_id="recovery-test")
    try:
        await _run_simple_stage(invoker)
        records = facade_module.snapshot_protected_chain_records()
    finally:
        facade_module.reset_protected_chain_records(token)

    assert [row["status"] for row in trace_rows] == ["failed", "succeeded"]
    assert [row["status"] for row in records] == [
        "contract_fault",
        "parsed",
    ]
    assert [row["attempt_index"] for row in records] == [1, 2]
    assert records[0]["parse_status"] == "contract_error"
    assert records[1]["parse_status"] == "succeeded"


@pytest.mark.asyncio
async def test_regeneration_is_skipped_below_the_remaining_deadline_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_rows = _trace_capture(monkeypatch)
    invoker = _SequencedInvoker([{"value": "rejected"}])

    with pytest.raises(CognitionExecutionError) as error:
        await _run_simple_stage(
            invoker,
            deadline_monotonic=(
                time.monotonic()
                + facade_module._COGNITION_ATTEMPT_TIME_FLOOR_SECONDS
                - 1
            ),
        )

    assert invoker.calls == 1
    assert error.value.error_code == "cognition_p_contract_exhausted"
    assert error.value.safe_checkpoint == "pre_state_commit"
    assert [row["attempt_index"] for row in trace_rows] == [1]


@pytest.mark.asyncio
async def test_appraisal_family_key_order_is_normalized_without_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_rows = _trace_capture(monkeypatch)
    normalization_events: list[dict[str, object]] = []

    async def record_contract_event(**kwargs: object) -> dict[str, object]:
        normalization_events.append(kwargs)
        return {
            "accepted": True,
            "trace_id": "deterministic-trace",
            "status": "recorded",
            "reason": "",
        }

    rows = {
        family: {
            "applicable": True,
            "semantic_summary": "meaning",
            "cause_summary": "cause",
            "axis_changes": [],
        }
        for family in CANONICAL_A1_FAMILIES
    }
    reordered = OrderedDict(
        (family, rows[family])
        for family in reversed(CANONICAL_A1_FAMILIES)
    )

    monkeypatch.setattr(
        facade_module.event_logging,
        "record_model_contract_event",
        record_contract_event,
    )
    invoker = _SequencedInvoker([reordered])
    token = facade_module.bind_protected_chain_records(run_id="appraisal-test")
    try:
        result = await _run_cognition_stage(
            services=_services(invoker),
            stage="A1",
            packet={"stage": "A1", "output_contract": {}},
            validator=lambda raw: facade_module._validate_appraisal_stage(
                raw,
                families=CANONICAL_A1_FAMILIES,
            ),
            deadline_monotonic=time.monotonic() + 60,
        )
        records = facade_module.snapshot_protected_chain_records()
    finally:
        facade_module.reset_protected_chain_records(token)

    assert invoker.calls == 1
    assert tuple(item.family for item in result) == CANONICAL_A1_FAMILIES
    assert [row["parse_status"] for row in trace_rows] == ["normalized"]
    assert [row["status"] for row in trace_rows] == ["succeeded"]
    assert [row["status"] for row in records] == ["parsed"]
    assert [row["parse_status"] for row in records] == ["normalized"]
    assert [event["violation_kind"] for event in normalization_events] == [
        "appraisal_family_key_order",
    ]
    assert normalization_events[0]["repair_used"] is False
    assert normalization_events[0]["status"] == "normalized"
