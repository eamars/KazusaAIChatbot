"""Tests for canonical V2 dialog rendering."""

from __future__ import annotations

import json
import logging
import math
import typing
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage
import pytest

from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import (
    DialogAgentState,
    StateContractError,
    dialog_agent,
    dialog_generator,
)
from tests.cognition_core_v2_test_helpers import canonical_episode


@pytest.fixture(autouse=True)
def _stub_dialog_event_logging(monkeypatch):
    """Keep deterministic dialog tests away from event-log persistence."""

    monkeypatch.setattr(
        dialog_module.llm_tracing,
        "record_llm_trace_step",
        AsyncMock(),
    )
    for recorder_name in (
        "record_llm_stage_event",
        "record_model_contract_event",
        "record_dialog_quality_event",
    ):
        monkeypatch.setattr(
            dialog_module.event_logging,
            recorder_name,
            AsyncMock(),
        )
    verifier_outputs = {
        "_dialog_semantic_fidelity_llm": (
            '{"score": 1.0, "hard_errors": []}'
        ),
        "_dialog_surface_integrity_llm": (
            '{"score": 1.0, "issues": []}'
        ),
    }
    for llm_name, verifier_output in verifier_outputs.items():
        compliance_llm = MagicMock()
        compliance_llm.ainvoke = AsyncMock(
            return_value=AIMessage(
                content=verifier_output,
            )
        )
        monkeypatch.setattr(dialog_module, llm_name, compliance_llm)


def _text_surface_output() -> dict[str, object]:
    """Build a valid native text-surface output."""

    return {
        "schema_version": "text_surface_output.v2",
        "content_plan": "Greet the user warmly.",
        "content_requirements": ["Address the current user."],
        "visible_boundaries": [],
        "addressee_plan": [{
            "handle": "current_user",
            "display_name": "current user",
            "semantic_role": "direct_recipient",
            "wording_policy": "second_person_allowed",
        }],
        "delivery_profile": {
            "lexical_register": "warm",
            "sentence_shape": "concise",
            "rhythm": "steady",
            "hesitation": "minimal",
            "punctuation": "restrained",
        },
        "selected_surface_intent": "acknowledge",
        "permitted_action_results": [],
    }


def _text_surface_input() -> dict[str, object]:
    """Build the canonical surface input used by bounded repair."""

    return {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            episode_id="dialog-agent-test",
            content="Greet the current user.",
        ),
        "intention": {
            "route": "speech",
            "intention": "acknowledge the greeting",
            "target_roles": [],
            "reason": "the greeting is directly observed",
            "goal_continuation_ref": None,
        },
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "warm",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "brief and natural",
        "character_expression_context": {
            "tempo": "measured",
            "linguistic_texture": "reserved, analytical, and warm",
        },
        "visual_character_context": "complete visual character context",
    }


def _character_profile() -> dict[str, object]:
    """Build the dialog renderer's character-only wording context."""

    return {
        "name": "Kazusa",
        "personality_brief": {
            "logic": "analytical",
            "tempo": "moderate",
            "defense": "reserved",
            "quirks": "occasional hesitation",
            "taboos": "stay in character",
        },
        "linguistic_texture_profile": {
            "hesitation_density": 0.4,
            "fragmentation": 0.4,
            "emotional_leakage": 0.4,
            "rhythmic_bounce": 0.4,
            "direct_assertion": 0.4,
            "softener_density": 0.4,
            "counter_questioning": 0.4,
            "formalism_avoidance": 0.4,
            "abstraction_reframing": 0.4,
            "self_deprecation": 0.4,
        },
    }


def _base_global_state() -> dict[str, object]:
    """Build the canonical state consumed by dialog_agent."""

    return {
        "internal_monologue": "thinking about greeting",
        "text_surface_output_v2": _text_surface_output(),
        "cognitive_episode": canonical_episode(
            episode_id="dialog-agent-test",
            content="Greet the current user.",
        ),
        "chat_history_wide": [],
        "chat_history_recent": [],
        "debug_modes": {},
        "should_respond": True,
        "platform_user_id": "user-123",
        "platform_bot_id": "bot-456",
        "global_user_id": "global-user-123",
        "user_name": "TestUser",
        "user_profile": {
            "global_user_id": "global-user-123",
            "cognition_state": {"owner_user_id": "global-user-123"},
        },
        "character_profile": _character_profile(),
    }


def _dialog_state() -> dict[str, object]:
    """Build the direct dialog-generator state."""

    state = _base_global_state()
    state.update(
        {
            "target_addressed_user_ids": [],
            "target_broadcast": False,
            "dialog_usage_mode": "unit_test",
            "llm_trace_id": "trace-1",
            "text_surface_input_v2": _text_surface_input(),
        }
    )
    return state


@pytest.mark.parametrize(
    "invalid_score",
    [True, -0.01, 1.01, 10**400, float("nan"), float("inf")],
)
def test_dialog_score_validation_rejects_non_finite_or_out_of_range(
    invalid_score: object,
) -> None:
    """Evaluator scores are numeric, finite, and bounded to the contract."""

    with pytest.raises(StateContractError, match="score"):
        dialog_module._validate_numeric_score(
            invalid_score,
            label="dialog test",
        )


def test_dialog_score_validation_rejects_passing_score_with_issues() -> None:
    """Every focused owner must lower its score when it reports an issue."""

    with pytest.raises(StateContractError, match="inconsistent"):
        dialog_module._validate_semantic_fidelity_verdict(
            {"score": 0.9, "hard_errors": ["semantic conflict"]},
            max_issues=4,
        )
    with pytest.raises(StateContractError, match="inconsistent"):
        dialog_module._validate_compliance_verdict(
            {"score": 0.9, "issues": ["quality issue"]},
            max_issues=4,
        )
    with pytest.raises(StateContractError, match="inconsistent"):
        dialog_module._validate_role_direction_verdict(
            {
                "score": 0.9,
                "violations": [{
                    "kind": "selection_owner_transfer",
                    "evidence": "candidate text",
                    "explanation": "selection owner changed",
                }],
            },
            generated_dialog=["candidate text"],
        )
    with pytest.raises(StateContractError, match="inconsistent"):
        dialog_module._validate_surface_compliance_verdict(
            {
                "score": 0.9,
                "issues": [{
                    "kind": "false_execution",
                    "evidence": "candidate text",
                    "explanation": "execution has no result",
                }],
            },
            generated_dialog=["candidate text"],
        )


def test_dialog_score_aggregate_uses_available_dimensions() -> None:
    """Unavailable owners do not erase comparable numeric bids."""

    aggregate = {
        "semantic_fidelity": {
            "status": "scored",
            "score": 0.81,
            "issues": [],
        },
        "role_direction": {
            "status": "unavailable",
            "violations": [],
        },
        "surface_integrity": {
            "status": "scored",
            "score": 0.49,
            "issues": [],
        },
        "lexical_avoidance": {
            "status": "scored",
            "score": 1.0,
            "issues": [],
        },
    }

    assert math.isclose(
        dialog_module._dialog_verifier_aggregate_score(aggregate),
        (0.81 * 0.49 * 1.0) ** (1 / 3),
    )


def test_dialog_score_all_focused_unavailable_ranks_as_zero() -> None:
    """A clean lexical result cannot make an outage pass immediately."""

    aggregate = {
        "semantic_fidelity": {
            "status": "unavailable",
            "issues": [],
        },
        "role_direction": {
            "status": "unavailable",
            "violations": [],
        },
        "surface_integrity": {
            "status": "unavailable",
            "issues": [],
        },
        "lexical_avoidance": {
            "status": "scored",
            "score": 1.0,
            "issues": [],
        },
    }

    assert dialog_module._dialog_verifier_aggregate_score(aggregate) == 0.0
    assert not dialog_module._dialog_verifier_aggregate_is_passed(aggregate)


@pytest.mark.asyncio
async def test_dialog_exhaustion_selects_highest_score_not_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three below-threshold bids return the highest eligible candidate."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content='{"final_dialog": ["low bid"]}'),
        AIMessage(content='{"final_dialog": ["highest bid"]}'),
        AIMessage(content='{"final_dialog": ["latest bid"]}'),
    ])

    def compliance(score: float) -> dict[str, object]:
        return {
            "semantic_fidelity": {
                "status": "scored",
                "score": score,
                "issues": [],
            },
            "role_direction": {
                "status": "scored",
                "score": 1.0,
                "violations": [],
            },
            "surface_integrity": {
                "status": "scored",
                "score": 1.0,
                "issues": [],
            },
            "lexical_avoidance": {
                "status": "scored",
                "score": 1.0,
                "issues": [],
            },
        }

    compliance_llm = AsyncMock(side_effect=[
        compliance(0.01),
        compliance(0.02),
        compliance(0.015),
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_verify_dialog_compliance",
        compliance_llm,
    )

    result = await dialog_generator(_dialog_state())

    assert result["final_dialog"] == ["highest bid"]
    assert generator_llm.ainvoke.await_count == 3
    assert compliance_llm.await_count == 3


@pytest.mark.asyncio
async def test_dialog_exhaustion_all_unavailable_selects_latest_valid_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All focused outages consume attempts and tie-break to the latest bid."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content='{"final_dialog": ["first outage bid"]}'),
        AIMessage(content='{"final_dialog": ["second outage bid"]}'),
        AIMessage(content='{"final_dialog": ["latest outage bid"]}'),
    ])

    def unavailable() -> dict[str, object]:
        return {
            "semantic_fidelity": {
                "status": "unavailable",
                "issues": [],
            },
            "role_direction": {
                "status": "unavailable",
                "violations": [],
            },
            "surface_integrity": {
                "status": "unavailable",
                "issues": [],
            },
            "lexical_avoidance": {
                "status": "scored",
                "score": 1.0,
                "issues": [],
            },
        }

    compliance_llm = AsyncMock(side_effect=[
        unavailable(),
        unavailable(),
        unavailable(),
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_verify_dialog_compliance",
        compliance_llm,
    )

    result = await dialog_generator(_dialog_state())

    assert result["final_dialog"] == ["latest outage bid"]
    assert generator_llm.ainvoke.await_count == 3
    assert compliance_llm.await_count == 3


@pytest.mark.asyncio
async def test_dialog_exhaustion_ties_select_latest_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal aggregate bids resolve deterministically to the latest attempt."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content='{"final_dialog": ["first tied bid"]}'),
        AIMessage(content='{"final_dialog": ["second tied bid"]}'),
        AIMessage(content='{"final_dialog": ["latest tied bid"]}'),
    ])

    def compliance() -> dict[str, object]:
        return {
            "semantic_fidelity": {
                "status": "scored",
                "score": 0.02,
                "issues": [],
            },
            "role_direction": {
                "status": "scored",
                "score": 1.0,
                "violations": [],
            },
            "surface_integrity": {
                "status": "scored",
                "score": 1.0,
                "issues": [],
            },
            "lexical_avoidance": {
                "status": "scored",
                "score": 1.0,
                "issues": [],
            },
        }

    compliance_llm = AsyncMock(side_effect=[
        compliance(),
        compliance(),
        compliance(),
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_verify_dialog_compliance",
        compliance_llm,
    )

    result = await dialog_generator(_dialog_state())

    assert result["final_dialog"] == ["latest tied bid"]
    assert generator_llm.ainvoke.await_count == 3
    assert compliance_llm.await_count == 3


class TestDialogAgentState:
    """Verify the dialog graph state exposes only the canonical surface input."""

    def test_is_typed_dict(self):
        assert issubclass(DialogAgentState, dict)

    def test_has_native_surface_contract(self):
        hints = typing.get_type_hints(DialogAgentState)
        assert "text_surface_output_v2" in hints
        assert "action_directives" not in hints


def test_v2_prompt_describes_surface_renderer_boundary() -> None:
    """The active prompt must keep semantic decisions upstream."""

    prompt = dialog_module._V2_DIALOG_GENERATOR_PROMPT

    assert "text_surface_output_v2" in prompt
    assert "自然" in prompt
    assert "角色辨识度" in prompt
    assert "实际会说出或发送" in prompt
    assert "第一人称" in prompt
    assert "action description" not in prompt.casefold()
    assert "动作描写" not in prompt
    assert "final_dialog" in prompt
    assert "relational_willingness" in prompt
    assert "action_directives" not in prompt
    assert "新生成的对话使用简体中文" not in prompt


def test_dialog_generator_repairs_unresolved_context_once() -> None:
    """The second renderer follows the L3 replacement and capability truth."""

    repair_prompt = dialog_module._V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT

    assert "text_surface_output_v2" in repair_prompt
    assert "repair_context" in repair_prompt
    assert "current_visible_percepts" not in repair_prompt
    assert "executed 表达其有界的已完成效果" in repair_prompt
    assert "failed 与 unavailable 表达当前限制和可行下一步" in (
        repair_prompt
    )
    assert "runtime_capability_limits" in repair_prompt
    assert "可信的能力边界、等待状态和下一步条件" in repair_prompt
    assert "新生成的对话使用简体中文" not in repair_prompt


def test_dialog_prompts_map_pending_results_to_queue_only_truth() -> None:
    """Every dialog pass must keep pending work at the queue boundary."""

    prompts = (
        dialog_module._V2_DIALOG_GENERATOR_PROMPT,
        dialog_module._V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT,
        dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT,
    )

    for prompt in prompts:
        assert "pending" in prompt
        assert "已记录" in prompt
        assert "已排队" in prompt
        assert "等待" in prompt

    assert "立即执行" in dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT
    assert "立即反馈" in dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT
    for prompt in (
        dialog_module._V2_DIALOG_GENERATOR_PROMPT,
        dialog_module._V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT,
    ):
        assert "executed 表达其有界的已完成效果" in prompt
        assert "failed 与 unavailable 表达当前限制和可行下一步" in (
            prompt
        )


@pytest.mark.asyncio
async def test_surface_integrity_prompt_receives_runtime_limits(
    monkeypatch,
) -> None:
    """The verifier receives trusted unavailable-owner facts with the candidate."""

    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content='{"score": 1.0, "issues": []}')
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )
    surface_output = _text_surface_output()
    surface_output["runtime_capability_limits"] = [
        "当前调度能力不可用，不能把提醒说成已经安排。",
    ]

    await dialog_module._verify_dialog_surface_integrity(
        surface_output=surface_output,
        generated_dialog=["我现在还不能确认提醒已经安排。"],
        current_visible_percepts=[],
        llm_trace_id="runtime-limit-test",
    )

    payload = json.loads(
        surface_llm.ainvoke.await_args.args[0][1].content,
    )
    assert payload["runtime_capability_limits"] == (
        surface_output["runtime_capability_limits"]
    )


@pytest.mark.asyncio
async def test_dialog_generator_forwards_native_surface_without_legacy_fields(
    monkeypatch,
) -> None:
    """The model input contains the V2 surface and no legacy directive envelope."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content='{"final_dialog": ["Hello."]}')
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)

    result = await dialog_generator(_dialog_state())

    assert result["final_dialog"] == ["Hello."]
    human_payload = json.loads(generator_llm.ainvoke.await_args.args[0][1].content)
    assert set(human_payload) == {
        "text_surface_output_v2",
        "candidate_role_frame",
        "user_name",
    }
    assert set(human_payload["text_surface_output_v2"]) == {
        "schema_version",
        "content_plan",
        "content_requirements",
        "visible_boundaries",
        "addressee_plan",
        "delivery_profile",
        "selected_surface_intent",
        "permitted_action_results",
    }
    assert human_payload["text_surface_output_v2"]["schema_version"] == (
        "text_surface_output.v2"
    )
    rendered_payload = json.dumps(human_payload)
    for forbidden_field in (
        "action_directives",
        "internal_monologue",
        "chat_history_wide",
        "chat_history_recent",
        "user_profile",
        "character_profile",
        "cognition_core_output",
        "cognition_state",
    ):
        assert forbidden_field not in rendered_payload


@pytest.mark.asyncio
async def test_dialog_generator_preserves_valid_fragment_text(
    monkeypatch,
) -> None:
    """Final wording remains model-owned after exact contract validation."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=json.dumps({
                "final_dialog": ["  First line.  ", "Second line."],
            })
        )
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)

    result = await dialog_generator(_dialog_state())

    assert result["final_dialog"] == ["  First line.  ", "Second line."]


@pytest.mark.asyncio
async def test_dialog_generator_requires_native_surface_before_llm_call(
    monkeypatch,
) -> None:
    """Missing V2 surface state fails before any renderer call."""

    state = _dialog_state()
    del state["text_surface_output_v2"]
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock()
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)

    with pytest.raises(StateContractError, match="text_surface_output_v2"):
        await dialog_generator(state)

    generator_llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_dialog_agent_returns_final_dialog_and_target(monkeypatch) -> None:
    """The public dialog result remains a visible text surface."""

    state = _base_global_state()
    generator_response = AIMessage(
        content='{"final_dialog": ["Hello there!", "How are you?"]}'
    )

    async def mock_ainvoke(messages, *, config=None):
        del messages, config
        return generator_response

    generator_llm = MagicMock()
    generator_llm.ainvoke = mock_ainvoke
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)

    result = await dialog_agent(state)

    assert result["final_dialog"] == ["Hello there!", "How are you?"]
    assert result["target_addressed_user_ids"] == ["global-user-123"]
    assert result["target_broadcast"] is False
    assert result["text_surface_output_v2"] == _text_surface_output()


@pytest.mark.asyncio
async def test_dialog_agent_rejects_missing_surface_before_llm_call(monkeypatch):
    """The public dialog boundary cannot fall back to V1 directives."""

    state = _base_global_state()
    del state["text_surface_output_v2"]
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock()
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)

    with pytest.raises(StateContractError, match="text_surface_output_v2"):
        await dialog_agent(state)

    generator_llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_dialog_agent_rejects_total_empty_candidate_exhaustion(
    monkeypatch,
):
    """Three empty candidates leave no normal dialog output to deliver."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content='{"final_dialog": []}')
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)

    with pytest.raises(
        StateContractError,
        match="exhausted candidates",
    ):
        await dialog_agent(_base_global_state())

    assert generator_llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_dialog_verifies_terminal_candidate_before_delivery(monkeypatch):
    """A terminal candidate must pass every focused verifier before delivery."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content='{"final_dialog": ["Candidate."]}')
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    verifier = AsyncMock(return_value={
        "semantic_fidelity": {
            "status": "scored",
            "score": 0.01,
            "issues": [],
        },
        "role_direction": {
            "status": "scored",
            "score": 1.0,
            "violations": [],
        },
        "surface_integrity": {
            "status": "scored",
            "score": 1.0,
            "issues": [],
        },
        "lexical_avoidance": {
            "status": "scored",
            "score": 1.0,
            "issues": [],
        },
    })
    monkeypatch.setattr(dialog_module, "_verify_dialog_compliance", verifier)
    monkeypatch.setattr(
        dialog_module,
        "repair_text_surface_for_dialog",
        AsyncMock(return_value=_text_surface_output()),
    )

    result = await dialog_generator(_dialog_state())

    assert verifier.await_count == 3
    assert result["final_dialog"] == ["Candidate."]


@pytest.mark.asyncio
async def test_dialog_agent_logs_explicit_usage_mode(caplog, monkeypatch):
    """Dialog output logs identify the caller-supplied usage mode."""

    state = _base_global_state()
    state["dialog_usage_mode"] = "background_contract_render"
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(
        return_value=AIMessage(content='{"final_dialog": ["Internal only."]}')
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)

    with caplog.at_level(logging.INFO, logger=dialog_module.__name__):
        await dialog_agent(state)

    assert "usage_mode=background_contract_render" in caplog.text


def test_dialog_usage_mode_requires_internal_state_fields():
    """Missing routing fields remain typed contract failures."""

    state_without_debug_modes = _base_global_state()
    del state_without_debug_modes["debug_modes"]
    with pytest.raises(KeyError):
        dialog_module._dialog_usage_mode(state_without_debug_modes)

    state_without_should_respond = _base_global_state()
    del state_without_should_respond["should_respond"]
    with pytest.raises(KeyError):
        dialog_module._dialog_usage_mode(state_without_should_respond)
