"""Deterministic tests for serial primary-chain execution."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.model_attempt_policy import (
    bind_v2_attempt_ledger,
    create_v2_attempt_ledger,
    current_v2_attempt_ledger,
    reset_v2_attempt_ledger,
    snapshot_v2_attempt_ledger,
)
from kazusa_ai_chatbot.cognition_core_v3 import execution
from kazusa_ai_chatbot.cognition_core_v3 import prompt as v3_prompt
from kazusa_ai_chatbot.cognition_core_v3.budget import (
    CognitionContextLimitError,
    ContextBudgetLedger,
    ContextBudgetPlan,
)
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    bind_protected_chain_records,
    current_chain_scope,
    reset_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_core_v3.execution import (
    AttemptLedger,
    ExecutorContractError,
    SerialChainHarness,
    SerialChainStep,
    SerialQuestionResult,
    StageAttemptOutcome,
    TurnDeadlineExceeded,
    config_for_turn_deadline,
    invoke_serial_model_step,
    invoke_serial_question_with_repair,
    run_serial_chain,
)
from kazusa_ai_chatbot.cognition_core_v3.transcript import (
    ChainMessageV1,
    ChainTranscriptV1,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig, LLMThinkingConfig


def _deadline_config() -> LLMCallConfig:
    """Build one route config for deterministic deadline assertions."""

    return LLMCallConfig(
        stage_name="test.stage",
        route_name="test-route",
        base_url="http://test.example/v1",
        api_key="test-key",
        model="test-model",
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_completion_tokens=128,
        presence_penalty=None,
        timeout_seconds=90.0,
        thinking=LLMThinkingConfig(enabled=False),
        context_window_tokens=50_000,
    )


def _budget(serving_window_tokens: int = 50_000) -> ContextBudgetLedger:
    """Build the real invocation budget used by execution tests."""

    return ContextBudgetLedger(
        ContextBudgetPlan(serving_window_tokens=serving_window_tokens),
    )


def test_registered_step_id_preserves_recurrence_base_and_repair_suffix() -> None:
    """Repair routing keeps the registered semantic step identity."""

    assert execution._registered_step_id("A1") == "A1"
    assert execution._registered_step_id("A1.repair1") == "A1"
    assert execution._registered_step_id("R.A1") == "A1"
    assert execution._registered_step_id("R.A1.repair2") == "A1"
    assert execution._registered_step_id("A1.event_agency") == "A1"
    assert execution._registered_step_id("R.A1.event_agency.repair1") == "A1"
    assert execution._registered_step_id("R.G1b.repair1") == "G1b"


def test_primary_attempt_records_the_admitted_lane_claim_facts() -> None:
    """Primary step telemetry carries the owning lane claim values."""

    async def scenario() -> None:
        invoker = _RecordingLLM(['{"accepted":true}'])
        harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({}),
            budget=_budget(),
            primary_queue_wait_ms=17,
            primary_in_flight_at_start=1,
        )
        question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal_kind": "ordinary_response"},
        )
        await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=question,
            validator=lambda parsed: parsed,
            attempt_limit=1,
            deterministic_only=True,
        )

    token = bind_protected_chain_records(run_id="lane-facts")
    try:
        asyncio.run(scenario())
        scope = current_chain_scope()
        assert scope is not None
        primary_steps = [
            step for step in scope.steps if step["lane_kind"] == "primary"
        ]
        assert primary_steps[0]["queue_wait_ms"] == 17
        assert primary_steps[0]["in_flight_at_start"] == 1
    finally:
        reset_protected_chain_records(token)


class _NeverInvokedLLM:
    """Raise if an expired turn reaches the provider boundary."""

    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages, *, config):
        del messages, config
        self.calls += 1
        raise AssertionError("expired turn started model work")


class _RecordingLLM:
    """Return fixed model results while retaining exact request messages."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = iter(responses)
        self.message_batches: list[tuple[object, ...]] = []
        self.configs: list[LLMCallConfig] = []

    async def ainvoke(self, messages, *, config):
        """Capture one request and return the next scripted response."""

        self.message_batches.append(tuple(messages))
        self.configs.append(config)
        response = SimpleNamespace(content=next(self._responses))
        return response


def _accepting_producer(log: list[str]):
    async def producer(ctx):
        log.append(f"{ctx.chain_name}:{ctx.stage_name}:start:{ctx.attempt_number}")
        await asyncio.sleep(0)
        return StageAttemptOutcome(
            True,
            {"stage": ctx.stage_name},
            f"summary {ctx.stage_name}",
            None,
        )

    return producer


def test_executor_runs_one_serial_primary_chain_and_preserves_attempt_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> list[str]:
        monkeypatch.setattr(execution.time, "monotonic", lambda: 100.0)
        bounded = config_for_turn_deadline(_deadline_config(), 110.0)
        assert bounded.timeout_seconds == 10.0
        invoker = _NeverInvokedLLM()
        expired_harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({}),
            budget=_budget(),
        )
        with pytest.raises(TurnDeadlineExceeded):
            await invoke_serial_model_step(
                harness=expired_harness,
                system_content="system",
                llm=invoker,
                config=_deadline_config(),
                deadline_monotonic=99.0,
            )
        assert invoker.calls == 0

        log: list[str] = []
        steps = [
            SerialChainStep("A1", "appraisal", _accepting_producer(log)),
            SerialChainStep("G1a", "goal_ordinary", _accepting_producer(log)),
        ]
        ledger = AttemptLedger({"appraisal": 2, "goal_ordinary": 2})
        result = await run_serial_chain(steps, ledger=ledger)

        assert [step.stage_name for step in result.step_results] == [
            "appraisal",
            "goal_ordinary",
        ]
        assert all(step.accepted for step in result.step_results)
        assert ledger.attempts_used("appraisal") == 1
        assert ledger.attempts_used("goal_ordinary") == 1
        return log

    log = asyncio.run(scenario())
    assert log == [
        "serial:appraisal:start:1",
        "serial:goal_ordinary:start:1",
    ]


def test_serial_chain_rejects_unknown_step() -> None:
    async def scenario() -> None:
        steps = [SerialChainStep("unknown", "appraisal", _accepting_producer([]))]
        ledger = AttemptLedger({"appraisal": 1})
        await run_serial_chain(steps, ledger=ledger)

    try:
        asyncio.run(scenario())
    except ExecutorContractError as exc:
        assert "Unknown serial chain step" in str(exc)
    else:
        raise AssertionError("expected ExecutorContractError")


def test_first_question_repairs_preserve_packet_and_later_rows_stay_compact() -> None:
    """The cold carrier survives repair and is committed only when accepted."""

    async def scenario() -> tuple[
        _RecordingLLM,
        SerialChainHarness,
        SerialQuestionResult,
        SerialQuestionResult,
    ]:
        """Run one repaired first question followed by one later question."""

        first_product = {"parsed": {"value": "unchanged"}}
        later_product = {"parsed": {"value": "later"}}
        invoker = _RecordingLLM(
            [
                "{invalid",
                json.dumps(first_product),
                json.dumps(later_product),
            ]
        )
        harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({}),
            budget=_budget(),
        )
        first_observation_context = {
            "conversation_frame": {
                "channel_scope": "private",
                "character_role": "current character",
                "conversation_continuity": "",
                "current_user_role": "current user",
                "dialogue_role_bindings": [],
                "participant_bindings": [],
                "public_group_scene": "",
                "semantic_temporal_context": "current turn",
            },
            "direct_facts": [],
            "entity_index": [],
            "evidence": [{
                "handle": "e1",
                "source_kind": "episode",
                "semantic_text": "evidence-marker",
                "authority": "current_episode",
                "provenance_role": "current_episode",
            }],
            "supplemental_context": {
                "dialogue_observation": [],
                "local_time_context": [],
                "non_dialog_percepts": [],
                "trigger_source": "user_message",
            },
        }
        first_question = v3_prompt.ChainQuestion(
            contract_name="semantic_appraisal_group.v1",
            payload={"questions": [], "l1_residue": {}},
        )

        def validate_first(parsed: dict[str, object]) -> dict[str, object]:
            """Accept only the exact scripted first-question product."""

            if parsed != first_product:
                raise ValueError("first product is not the scripted object")
            return parsed

        first_result = await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=first_question,
            validator=validate_first,
            attempt_limit=2,
            observation_context=first_observation_context,
            deterministic_only=True,
        )
        later_question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal": "later"},
        )
        later_result = await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=later_question,
            validator=lambda parsed: parsed,
            attempt_limit=1,
            observation_context=first_observation_context,
            deterministic_only=True,
        )
        return invoker, harness, first_result, later_result

    invoker, harness, first_result, later_result = asyncio.run(
        scenario()
    )

    assert [config.stage_name for config in invoker.configs] == [
        "test.stage",
        "test.stage.repair1",
        "test.stage",
    ]
    first_payload = invoker.message_batches[0][-1].content
    repair_payload = invoker.message_batches[1][-1].content
    first_packet = json.loads(first_payload)
    assert [next(iter(section)) for section in first_packet] == [
        "observation_context",
        "question",
    ]
    assert "evidence-marker" in first_payload
    assert repair_payload.startswith(
        f"{first_payload}\n[contract_repair]\n"
    )
    assert len(invoker.message_batches[0]) == 2
    assert len(invoker.message_batches[1]) == 2

    transcript_messages = harness.transcript.to_messages()
    assert transcript_messages[0] == ("human", repair_payload)
    assert harness.transcript.accepted_products[0] == {
        "question": first_packet[-1]["question"]["contract_name"],
        "typed_product": first_result.validated,
    }
    assert first_result.validated == {"parsed": {"value": "unchanged"}}
    assert later_result.validated == {"parsed": {"value": "later"}}
    assert first_result.raw_output == json.dumps(
        {"parsed": {"value": "unchanged"}}
    )
    assert first_result.disposition.kind == "accepted"
    assert later_result.disposition.kind == "accepted"

    later_payload = invoker.message_batches[2][-1].content
    later_packet = json.loads(later_payload)
    assert [next(iter(section)) for section in later_packet] == ["question"]
    assert invoker.message_batches[2][1].content == repair_payload


def test_reattached_transcript_uses_later_question_format() -> None:
    """A recurrence-style accepted prefix receives no new cold carrier."""

    async def scenario() -> tuple[_RecordingLLM, SerialChainHarness]:
        """Invoke one question from an already accepted session prefix."""

        transcript = ChainTranscriptV1().append_question(
            "accepted-cold-packet"
        ).accept_answer(
            "accepted-cold-answer",
            {"question": "semantic_appraisal_group.v1"},
        )
        harness = SerialChainHarness(
            transcript=transcript,
            ledger=AttemptLedger({}),
            budget=_budget(),
        )
        invoker = _RecordingLLM([json.dumps({"parsed": "recurrence"})])
        question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal": "recurrence"},
        )
        result = await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=question,
            validator=lambda parsed: parsed,
            attempt_limit=1,
            deterministic_only=True,
        )
        assert result.disposition.kind == "accepted"
        assert result.validated == {"parsed": "recurrence"}
        return invoker, harness

    invoker, harness = asyncio.run(scenario())

    messages = invoker.message_batches[0]
    assert [message.content for message in messages[:-1]] == [
        "system",
        "accepted-cold-packet",
        "accepted-cold-answer",
    ]
    recurrence_packet = json.loads(messages[-1].content)
    assert [next(iter(section)) for section in recurrence_packet] == [
        "question"
    ]
    assert harness.transcript.to_messages()[-2][1] == messages[-1].content


def test_repair_appendix_is_monotonic_typed_and_prompt_safe() -> None:
    """Repair retries retain bounded facts without echoing rejected output."""

    async def scenario() -> tuple[_RecordingLLM, SerialChainHarness]:
        invoker = _RecordingLLM([
            '{"candidate":"raw-one"}',
            '{"candidate":"raw-two"}',
            '{"accepted":true}',
        ])
        harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({"owner": 3}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="semantic_appraisal_group.v1",
            payload={
                "questions": [{
                    "evidence_handles": ["e2", "e1"],
                    "permitted_role_handles": ["self", "p1"],
                    "permitted_role_assignment_handles": ["self"],
                }],
                "bid_handles": {"b2": {}, "b1": {}},
            },
        )
        validation_count = 0

        def validator(parsed: dict[str, object]) -> object:
            nonlocal validation_count
            validation_count += 1
            if validation_count < 3:
                error = ValueError(
                    f"private validator prose {validation_count}"
                )
                if validation_count == 1:
                    error.field_path = "result.value"
                raise error
            return parsed

        result = await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=question,
            validator=validator,
            attempt_limit=3,
            attempt_owner="owner",
            deterministic_only=True,
        )
        assert result.disposition.kind == "accepted"
        assert result.validated == {
            "accepted": True,
        }
        return invoker, harness

    invoker, harness = asyncio.run(scenario())

    repair_payloads = [
        str(invoker.message_batches[index][-1].content)
        for index in (1, 2)
    ]
    appendices = [
        json.loads(payload.split("\n[contract_repair]\n", 1)[1])
        for payload in repair_payloads
    ]
    assert appendices[0]["attempt_index"] == 2
    assert appendices[0]["expected_contract"] == (
        "semantic_appraisal_group.v1"
    )
    assert appendices[0]["permitted_handles"] == [
        "b1",
        "b2",
        "e1",
        "e2",
        "p1",
        "self",
    ]
    assert appendices[0]["error_facts"] == [{
        "attempt_index": 1,
        "error_class": "structural_contract",
        "error_code": "contract_error",
        "field_path": "result.value",
    }]
    assert appendices[1]["attempt_index"] == 3
    assert appendices[1]["error_facts"] == [
        appendices[0]["error_facts"][0],
        {
            "attempt_index": 2,
            "error_class": "structural_contract",
            "error_code": "contract_error",
            "field_path": "$",
        },
    ]
    repair_text = "\n".join(repair_payloads)
    assert "raw-one" not in repair_text
    assert "raw-two" not in repair_text
    assert "private validator prose" not in repair_text
    transcript_text = "\n".join(
        content for _, content in harness.transcript.to_messages()
    )
    assert "raw-one" not in transcript_text
    assert "raw-two" not in transcript_text


def test_identical_repair_short_circuits_and_consumes_v2_attempts() -> None:
    """Repeated invalid bytes exhaust the owner without another provider call."""

    async def scenario() -> tuple[
        _RecordingLLM,
        SerialChainHarness,
        dict[str, object] | None,
    ]:
        invoker = _RecordingLLM(["same-invalid", "same-invalid", "unused"])
        harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({"owner": 3}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="action_plan.v1",
            payload={"bid_handles": ["b1", "b2"]},
        )
        ledger = create_v2_attempt_ledger("execution-identical-repair")
        token = bind_v2_attempt_ledger(ledger, graph_attempt=1)

        def reject_candidate(parsed: dict[str, object]) -> object:
            del parsed
            raise ValueError("invalid candidate")

        try:
            result = await invoke_serial_question_with_repair(
                harness=harness,
                system_content="system",
                llm=invoker,
                config=_deadline_config(),
                question=question,
                validator=reject_candidate,
                attempt_limit=3,
                attempt_owner="owner",
                v2_stage="action_planning",
                v2_branch_ids=("action_plan",),
                deterministic_only=True,
            )
            assert result.validated is None
            assert result.raw_output == "same-invalid"
            assert result.disposition.kind == "structural_exhausted"
            snapshot = snapshot_v2_attempt_ledger()
        finally:
            reset_v2_attempt_ledger(token)
        return invoker, harness, snapshot

    invoker, harness, snapshot = asyncio.run(scenario())

    assert len(invoker.message_batches) == 2
    assert harness.ledger.attempts_used("owner") == 3
    assert harness.transcript.to_messages() == ()
    assert snapshot is not None
    assert [row["local_attempt"] for row in snapshot["attempts"]] == [1, 2, 3]
    assert [row["attempt_disposition"] for row in snapshot["attempts"]] == [
        "regenerate",
        "exhausted",
        "exhausted",
    ]


def test_group_question_reserves_each_listed_branch_per_physical_attempt() -> None:
    """G1b reservations use the actual stable roster, never a group key."""

    async def scenario() -> dict[str, object] | None:
        invoker = _RecordingLLM(["{}", "{}"])
        harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({"owner": 2}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="active_goal_bid_group.v1",
            payload={"roster": [{"branch_id": "safety"}, {"branch_id": "trust_verification"}]},
        )
        ledger = create_v2_attempt_ledger("execution-group-roster")
        token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
        try:
            result = await invoke_serial_question_with_repair(
                harness=harness,
                system_content="system",
                llm=invoker,
                config=_deadline_config(),
                question=question,
                validator=lambda parsed: (_ for _ in ()).throw(
                    ValueError("invalid group")
                ),
                attempt_limit=2,
                attempt_owner="owner",
                v2_stage="goal_bid_structure",
                v2_branch_ids=("safety", "trust_verification"),
                deterministic_only=True,
            )
            assert result.disposition.kind == "structural_exhausted"
            return snapshot_v2_attempt_ledger()
        finally:
            reset_v2_attempt_ledger(token)

    snapshot = asyncio.run(scenario())
    assert snapshot is not None
    assert [row["branch_id"] for row in snapshot["attempts"]] == [
        "safety",
        "trust_verification",
        "safety",
        "trust_verification",
    ]
    assert all(
        row["branch_id"] != "active_goal_group"
        for row in snapshot["attempts"]
    )


def test_direct_family_recovery_reserves_the_second_v2_attempt() -> None:
    """Singleton recovery keeps the shared branch coordinate at attempt two."""

    async def scenario() -> dict[str, object] | None:
        invoker = _RecordingLLM(["{}", "{}"])
        harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({"owner": 2}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="semantic_appraisal_group.v1",
            payload={"families": [{"family": "event_agency"}]},
        )
        ledger = create_v2_attempt_ledger("execution-family-recovery")
        token = bind_v2_attempt_ledger(ledger, graph_attempt=1)
        try:
            for local_attempt_start in (1, 2):
                result = await invoke_serial_question_with_repair(
                    harness=harness,
                    system_content="system",
                    llm=invoker,
                    config=_deadline_config(),
                    question=question,
                    validator=lambda parsed: parsed,
                    attempt_limit=1,
                    attempt_owner="owner",
                    v2_stage="semantic_appraisal",
                    v2_branch_ids=("A1_event_agency",),
                    v2_local_attempt_start=local_attempt_start,
                    deterministic_only=True,
                )
                assert result.disposition.kind == "accepted"
            return snapshot_v2_attempt_ledger()
        finally:
            reset_v2_attempt_ledger(token)

    snapshot = asyncio.run(scenario())
    assert snapshot is not None
    assert [row["local_attempt"] for row in snapshot["attempts"]] == [1, 2]


def test_two_empty_repair_responses_short_circuit_local_budget() -> None:
    """Two empty failures stop repair calls while preserving local arithmetic."""

    async def scenario() -> tuple[_RecordingLLM, SerialChainHarness]:
        invoker = _RecordingLLM(["", "", "unused"])
        harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({"owner": 3}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"evidence_handles": ["e1"]},
        )

        def reject_empty(parsed: dict[str, object]) -> object:
            del parsed
            raise ValueError("empty result is invalid")

        await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=question,
            validator=reject_empty,
            attempt_limit=3,
            attempt_owner="owner",
            deterministic_only=True,
        )
        return invoker, harness

    invoker, harness = asyncio.run(scenario())

    assert len(invoker.message_batches) == 2
    assert harness.ledger.attempts_used("owner") == 3


def test_v2_coordinates_require_an_ambient_attempt_ledger() -> None:
    """A V2-coordinate request fails before provider work without its ledger."""

    assert current_v2_attempt_ledger() is None
    invoker = _RecordingLLM(['{"accepted":true}'])
    harness = SerialChainHarness(
        transcript=ChainTranscriptV1(),
        ledger=AttemptLedger({}),
        budget=_budget(),
    )
    question = v3_prompt.ChainQuestion(
        contract_name="action_plan.v1",
        payload={"bid_handles": ["b1"]},
    )

    async def scenario() -> None:
        await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=question,
            validator=lambda parsed: parsed,
            attempt_limit=1,
            v2_stage="action_planning",
            v2_branch_ids=("action_plan",),
            deterministic_only=True,
        )

    with pytest.raises(
        ExecutorContractError,
        match="ambient invocation ledger",
    ):
        asyncio.run(scenario())
    assert invoker.message_batches == []


def test_repaired_acceptance_uses_compact_canonical_json() -> None:
    """A sidecar-repaired object is stored compactly without rejected bytes."""

    async def scenario() -> tuple[_RecordingLLM, SerialChainHarness]:
        invoker = _RecordingLLM(["{invalid"])
        harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="action_plan.v1",
            payload={"bid_handles": ["b1"]},
        )
        ledger = create_v2_attempt_ledger("execution-repaired-row")
        token = bind_v2_attempt_ledger(ledger, graph_attempt=1)

        async def repair_callback(
            raw_output: str,
            stage_name: str,
            coordinates: dict[str, object],
        ) -> dict[str, object]:
            del raw_output, stage_name, coordinates
            return {"z": "é", "a": 1}

        try:
            await invoke_serial_question_with_repair(
                harness=harness,
                system_content="system",
                llm=invoker,
                config=_deadline_config(),
                question=question,
                validator=lambda parsed: parsed,
                attempt_limit=1,
                v2_stage="action_planning",
                v2_branch_ids=("action_plan",),
                deterministic_only=True,
                json_repair_callback=repair_callback,
            )
        finally:
            reset_v2_attempt_ledger(token)
        return invoker, harness

    invoker, harness = asyncio.run(scenario())

    assert len(invoker.message_batches) == 1
    assert harness.transcript.to_messages()[-1] == (
        "assistant",
        '{"a":1,"z":"é"}',
    )
    transcript_text = "\n".join(
        content for _, content in harness.transcript.to_messages()
    )
    assert "{invalid" not in transcript_text


def test_provider_repair_appendix_excludes_exception_prose() -> None:
    """Provider failures become closed repair facts without their messages."""

    class _ProviderThenSuccess:
        def __init__(self) -> None:
            self.calls = 0
            self.configs: list[LLMCallConfig] = []
            self.message_batches: list[tuple[object, ...]] = []

        async def ainvoke(self, messages, *, config):
            self.calls += 1
            self.message_batches.append(tuple(messages))
            self.configs.append(config)
            if self.calls == 1:
                raise OSError("provider secret prose")
            return SimpleNamespace(content='{"accepted":true}')

    async def scenario() -> tuple[_ProviderThenSuccess, SerialChainHarness]:
        invoker = _ProviderThenSuccess()
        harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({"owner": 2}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="action_plan.v1",
            payload={"bid_handles": ["b1"]},
        )
        result = await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=question,
            validator=lambda parsed: parsed,
            attempt_limit=2,
            attempt_owner="owner",
            deterministic_only=True,
        )
        assert result.disposition.kind == "accepted"
        assert result.validated == {"accepted": True}
        return invoker, harness

    invoker, harness = asyncio.run(scenario())

    assert invoker.calls == 2
    assert [config.stage_name for config in invoker.configs] == [
        "test.stage",
        "test.stage.repair1",
    ]
    repair_payload = str(invoker.message_batches[1][-1].content)
    appendix = json.loads(
        repair_payload.split("\n[contract_repair]\n", 1)[1]
    )
    assert appendix["error_facts"] == [{
        "attempt_index": 1,
        "error_class": "provider_error",
        "error_code": "provider_error",
        "field_path": "$",
    }]
    assert appendix["permitted_handles"] == ["b1"]
    assert "provider secret prose" not in repair_payload
    assert harness.transcript.to_messages()[0][1] == repair_payload


def test_primary_admission_records_the_configured_completion_cap() -> None:
    """Every admitted primary request records its exact derived reservation."""

    async def scenario() -> tuple[_RecordingLLM, SerialChainHarness]:
        invoker = _RecordingLLM(['{"accepted":true}'])
        harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal_kind": "ordinary_response", "bid_handles": ["b1"]},
        )
        await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=question,
            validator=lambda parsed: parsed,
            attempt_limit=1,
            deterministic_only=True,
        )
        return invoker, harness

    invoker, harness = asyncio.run(scenario())

    assert len(invoker.message_batches) == 1
    assert invoker.configs[0].max_completion_tokens == 128
    token_ledger = harness.transcript.token_ledger
    assert token_ledger is not None
    assert token_ledger["active_total_ceiling_tokens"] == 50_000
    assert token_ledger["max_estimated_prompt_tokens"] > 0
    assert token_ledger["max_reserved_completion_tokens"] == 128
    assert token_ledger["max_estimated_total_context_tokens"] == (
        token_ledger["max_estimated_prompt_tokens"] + 128
    )
    assert token_ledger["extension_available"] == 0
    assert token_ledger["extension_used"] == 0


def test_context_overflow_rejects_before_provider_work() -> None:
    """A request that remains over its serving window never reaches the LLM."""

    async def scenario() -> tuple[_RecordingLLM, SerialChainHarness]:
        invoker = _RecordingLLM(['{"accepted":true}'])
        harness = SerialChainHarness(
            transcript=ChainTranscriptV1(),
            ledger=AttemptLedger({"owner": 1}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal_kind": "ordinary_response"},
        )
        with pytest.raises(CognitionContextLimitError):
            await invoke_serial_question_with_repair(
                harness=harness,
                system_content="x" * 300_000,
                llm=invoker,
                config=_deadline_config(),
                question=question,
                validator=lambda parsed: parsed,
                attempt_limit=1,
                attempt_owner="owner",
                deterministic_only=True,
            )
        return invoker, harness

    invoker, harness = asyncio.run(scenario())

    assert invoker.message_batches == []
    assert harness.ledger.attempts_used("owner") == 0


def test_context_pressure_uses_one_prompt_safe_reanchor() -> None:
    """One re-anchor preserves typed facts and drops accepted model prose."""

    async def scenario() -> tuple[_RecordingLLM, SerialChainHarness]:
        accepted_prefix = (
            ChainTranscriptV1()
            .append_question("accepted question " + "q" * 220_000)
            .accept_answer(
                "rejected reasoning that must not be replayed " + "r" * 220_000,
                {
                    "question": "ordinary_goal_bid.v1",
                    "typed_product": {
                        "branch_id": "ordinary_response",
                        "private_monologue": "drop this prose",
                        "evidence_handles": ["e1"],
                    },
                },
            )
        )
        invoker = _RecordingLLM(['{"accepted":true}'])
        harness = SerialChainHarness(
            transcript=accepted_prefix,
            ledger=AttemptLedger({}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal_kind": "ordinary_response", "bid_handles": ["b1"]},
        )
        await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=question,
            validator=lambda parsed: parsed,
            attempt_limit=1,
            deterministic_only=True,
        )
        return invoker, harness

    invoker, harness = asyncio.run(scenario())

    assert len(invoker.message_batches) == 1
    request_contents = "\n".join(
        str(message.content) for message in invoker.message_batches[0]
    )
    assert "reanchor" in request_contents
    assert "rejected reasoning" not in request_contents
    assert "drop this prose" not in request_contents
    assert harness.transcript.reanchor_used is True
    assert harness.transcript.token_ledger is not None
    assert harness.transcript.token_ledger["reanchor_used"] == 1
    assert [role for role, _content in harness.transcript.to_messages()] == [
        "human",
        "assistant",
    ]


def test_second_context_pressure_fails_without_a_provider_call() -> None:
    """A prior re-anchor consumes the shared token for the whole invocation."""

    async def scenario() -> _RecordingLLM:
        transcript = ChainTranscriptV1(
            messages=(
                ChainMessageV1(
                    role="human",
                    content="old question " + "q" * 220_000,
                ),
            ),
            reanchor_used=True,
        )
        invoker = _RecordingLLM(['{"accepted":true}'])
        harness = SerialChainHarness(
            transcript=transcript,
            ledger=AttemptLedger({}),
            budget=_budget(),
        )
        harness.budget.reanchor_used = True
        question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal_kind": "ordinary_response"},
        )
        with pytest.raises(CognitionContextLimitError):
            await invoke_serial_question_with_repair(
                harness=harness,
                system_content="system",
                llm=invoker,
                config=_deadline_config(),
                question=question,
                validator=lambda parsed: parsed,
                attempt_limit=1,
                deterministic_only=True,
            )
        return invoker

    invoker = asyncio.run(scenario())
    assert invoker.message_batches == []


def test_reanchored_repair_persists_provider_accepted_human_tail() -> None:
    """A repaired re-anchor stores the exact human bytes that succeeded."""

    async def scenario() -> tuple[
        _RecordingLLM,
        SerialChainHarness,
        str,
    ]:
        accepted_prefix = (
            ChainTranscriptV1()
            .append_question("accepted question " + "q" * 220_000)
            .accept_answer(
                "accepted answer " + "a" * 220_000,
                {
                    "question": "ordinary_goal_bid.v1",
                    "typed_product": {"branch_id": "ordinary_response"},
                },
            )
        )
        invoker = _RecordingLLM([
            "not a valid accepted candidate",
            '{"accepted":true}',
            '{"next":true}',
        ])
        harness = SerialChainHarness(
            transcript=accepted_prefix,
            ledger=AttemptLedger({}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal_kind": "ordinary_response"},
        )

        def require_accepted(parsed: dict[str, object]) -> object:
            if parsed.get("accepted") is not True:
                raise ValueError("candidate is not accepted")
            return parsed

        await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=question,
            validator=require_accepted,
            attempt_limit=2,
            deterministic_only=True,
        )
        successful_repair_payload = str(
            invoker.message_batches[1][-1].content
        )
        next_question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal_kind": "ordinary_response", "next": True},
        )
        await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=next_question,
            validator=lambda parsed: parsed,
            attempt_limit=1,
            deterministic_only=True,
        )
        return invoker, harness, successful_repair_payload

    invoker, harness, successful_repair_payload = asyncio.run(scenario())

    assert harness.transcript.to_messages()[:2] == (
        ("human", successful_repair_payload),
        ("assistant", '{"accepted":true}'),
    )
    next_request = invoker.message_batches[2]
    assert next_request[1].content == successful_repair_payload
    assert next_request[2].content == '{"accepted":true}'


def test_reanchor_retains_bounded_semantics_without_raw_prose() -> None:
    """Re-anchor receipts retain validated planning facts and bounded handles."""

    async def scenario() -> tuple[_RecordingLLM, SerialChainHarness]:
        accepted_prefix = (
            ChainTranscriptV1()
            .append_question("accepted question " + "q" * 220_000)
            .accept_answer(
                "accepted answer " + "a" * 220_000,
                {
                    "question": "ordinary_goal_bid.v1",
                    "typed_product": {
                        "branch_id": "ordinary_response",
                        "goal_ref": {
                            "scope": "user",
                            "kind": "goal",
                            "entity_id": "private-goal-id",
                        },
                        "intention": "preserve the grounded reply",
                        "desired_outcome": "the reply remains useful",
                        "concrete_detail": "answer from accepted context",
                        "description": "maintain the active goal",
                        "expected_consequences": [
                            "the user receives a grounded answer",
                        ],
                        "evidence_handles": ["e1"],
                        "selected_response_operation": {
                            "direction": "reply",
                            "selection_required": False,
                        },
                        "resolver_goal_progress": {
                            "current_focus": "resolve the active question",
                            "deliverables": [{
                                "description": "keep the answer grounded",
                                "status": "partial",
                                "note": "continue from accepted facts",
                            }],
                        },
                        "resolver_observation": {
                            "evidence_handle": "e2",
                            "source_kind": "resolver",
                            "authority": "accepted",
                            "semantic_summary": "raw semantic summary",
                            "semantic_text": "raw evidence text",
                        },
                        "reason": "unbounded private reasoning",
                        "private_monologue": "private model prose",
                    },
                },
            )
        )
        invoker = _RecordingLLM(['{"accepted":true}'])
        harness = SerialChainHarness(
            transcript=accepted_prefix,
            ledger=AttemptLedger({}),
            budget=_budget(),
        )
        question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal_kind": "ordinary_response"},
        )
        await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=question,
            validator=lambda parsed: parsed,
            attempt_limit=1,
            deterministic_only=True,
        )
        return invoker, harness

    invoker, _harness = asyncio.run(scenario())
    anchor_text = str(invoker.message_batches[0][-1].content)

    for retained_fact in (
        "preserve the grounded reply",
        "the reply remains useful",
        "answer from accepted context",
        "maintain the active goal",
        "the user receives a grounded answer",
        "resolve the active question",
        "keep the answer grounded",
        "continue from accepted facts",
        '"evidence_handle":"e2"',
    ):
        assert retained_fact in anchor_text
    for excluded_text in (
        "private-goal-id",
        "unbounded private reasoning",
        "private model prose",
        "raw semantic summary",
        "raw evidence text",
    ):
        assert excluded_text not in anchor_text
