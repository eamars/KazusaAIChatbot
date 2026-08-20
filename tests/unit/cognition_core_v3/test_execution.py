"""Deterministic tests for serial primary-chain execution."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import execution
from kazusa_ai_chatbot.cognition_core_v3 import prompt as v3_prompt
from kazusa_ai_chatbot.cognition_core_v3.execution import (
    AttemptLedger,
    ExecutorContractError,
    SerialChainHarness,
    SerialChainStep,
    StageAttemptOutcome,
    TurnDeadlineExceeded,
    config_for_turn_deadline,
    invoke_serial_model_step,
    invoke_serial_question_with_repair,
    run_serial_chain,
)
from kazusa_ai_chatbot.cognition_core_v3.transcript import ChainTranscriptV1
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

    async def ainvoke(self, messages, *, config):
        """Capture one request and return the next scripted response."""

        del config
        self.message_batches.append(tuple(messages))
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
            budget=object(),
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
        dict[str, object],
        dict[str, object],
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
            budget=object(),
        )
        first_sections = (
            {
                "character_constraints": {"rule": "constraint-marker"},
                "character_operational_context": {},
            },
            {
                "relationship": {"state": "relationship-marker"},
                "mutable_state": {
                    "goals": [],
                    "threats": [],
                    "events": [],
                    "knowledge_gaps": [],
                    "affect": [],
                    "causal_candidates": [],
                },
            },
            {
                "episode": {
                    "episode_ref": "current_cognitive_episode",
                    "trigger_source": "user_message",
                    "visible_percepts": [
                        {
                            "input_source": "dialog",
                            "content": {"semantic_text": "episode-marker"},
                        }
                    ],
                },
                "scene_context": {
                    "channel_scope": "private",
                    "character_role": "current character",
                    "current_user_role": "current user",
                    "semantic_scene": "episode-marker",
                    "public_group_scene": "",
                    "conversation_continuity": "",
                    "semantic_temporal_context": "current turn",
                    "participant_bindings": [],
                },
            },
            {
                "evidence": [
                    {
                        "handle": "e1",
                        "source_kind": "episode",
                        "semantic_summary": "evidence-marker",
                    }
                ],
                "direct_facts": [],
                "available_actions": [],
                "available_resolver_capabilities": [],
                "resolver_context": "",
            },
        )
        first_question = v3_prompt.ChainQuestion(
            contract_name="semantic_appraisal_group.v1",
            payload={"questions": [], "l1_residue": {}},
        )

        def validate_first(parsed: dict[str, object]) -> dict[str, object]:
            """Accept only the exact scripted first-question product."""

            if parsed != first_product:
                raise ValueError("first product is not the scripted object")
            return parsed

        validated_first, _ = await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=first_question,
            validator=validate_first,
            attempt_limit=2,
            first_packet_sections=first_sections,
            deterministic_only=True,
        )
        later_question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal": "later"},
        )
        validated_later, _ = await invoke_serial_question_with_repair(
            harness=harness,
            system_content="system",
            llm=invoker,
            config=_deadline_config(),
            question=later_question,
            validator=lambda parsed: parsed,
            attempt_limit=1,
            first_packet_sections=first_sections,
            deterministic_only=True,
        )
        return invoker, harness, validated_first, validated_later

    invoker, harness, validated_first, validated_later = asyncio.run(
        scenario()
    )

    first_payload = invoker.message_batches[0][-1].content
    repair_payload = invoker.message_batches[1][-1].content
    first_packet = json.loads(first_payload)
    assert [next(iter(section)) for section in first_packet] == [
        "constraints_and_operational_state",
        "relationship_and_mutable_state",
        "episode_and_scene",
        "evidence_and_affordances",
        "question",
    ]
    assert "episode-marker" in first_payload
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
        "typed_product": validated_first,
    }
    assert validated_first == {"parsed": {"value": "unchanged"}}
    assert validated_later == {"parsed": {"value": "later"}}

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
            budget=object(),
        )
        invoker = _RecordingLLM([json.dumps({"parsed": "recurrence"})])
        question = v3_prompt.ChainQuestion(
            contract_name="ordinary_goal_bid.v1",
            payload={"goal": "recurrence"},
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
