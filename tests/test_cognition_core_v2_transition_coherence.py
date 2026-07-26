"""Focused contracts for V2 intra-turn transition coherence."""

from __future__ import annotations

import inspect
import json
import logging
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import kazusa_ai_chatbot.cognition_core_v2 as cognition_core_v2
from kazusa_ai_chatbot.cognition_core_v2 import contracts
from kazusa_ai_chatbot.cognition_core_v2 import surface
from kazusa_ai_chatbot.cognition_core_v2 import surface_stages
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
)
from kazusa_ai_chatbot.brain_service import post_turn
from kazusa_ai_chatbot.nodes import dialog_agent
from kazusa_ai_chatbot.nodes import persona_supervisor2
from tests.cognition_core_v2_test_helpers import canonical_episode


NOW = "2026-07-25T00:00:00Z"


class _UnifiedSurfaceLLM:
    """Capture the two planned surface calls and return exact stage outputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        """Return the exact object owned by the requested surface stage."""

        del config
        system_prompt = str(getattr(messages[0], "content", ""))
        human_payload = json.loads(
            str(getattr(messages[1], "content", "{}"))
        )
        surface_payload = human_payload["surface"]
        self.calls.append((system_prompt, surface_payload))
        if (
            "visible_boundaries" in system_prompt
            and "addressee_plan" in system_prompt
        ):
            result = {
                "visible_boundaries": [],
                "addressee_plan": ["Address the current user."],
            }
        elif "delivery_profile" in system_prompt:
            result = {
                "content_plan": "Accept the shared-participation framing.",
                "content_requirements": [
                    "Keep one accepting stance throughout the response.",
                ],
                "delivery_profile": _delivery_profile(),
            }
        else:
            raise AssertionError("unexpected V2 surface stage")
        response = SimpleNamespace(content=json.dumps(result))
        return response


def _delivery_profile() -> dict[str, str]:
    """Return one exact delivery-only profile."""

    return {
        "lexical_register": "casual and intimate",
        "sentence_shape": "short clauses",
        "rhythm": "measured with one brief pause",
        "hesitation": "light initial hesitation",
        "punctuation": "soft sentence endings",
    }


def _surface_input() -> dict[str, object]:
    """Build one valid target-shape text-surface input."""

    return {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            episode_id="transition-coherence-contract",
            content="We are accomplices in this together.",
        ),
        "intention": {
            "route": "speech",
            "intention": "confirm shared participation",
            "target_roles": [],
            "reason": "the character accepts the user's playful framing",
        },
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "embarrassed and playful",
            "intensity": "moderate",
            "directness": "indirect",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "Keep the exchange conversational.",
        "character_expression_context": {
            "tempo": "measured",
            "linguistic_texture": "light hesitation and rhythmic bounce",
        },
        "visual_character_context": "Full profile reserved for visual planning.",
    }


def _surface_output() -> dict[str, object]:
    """Build one valid target-shape text-surface output."""

    return {
        "schema_version": "text_surface_output.v2",
        "content_plan": "Confirm shared participation clearly and consistently.",
        "content_requirements": [
            "Keep one accepting stance throughout the response.",
        ],
        "visible_boundaries": [],
        "addressee_plan": ["Address the current user."],
        "delivery_profile": _delivery_profile(),
        "selected_surface_intent": "confirm shared participation",
        "permitted_action_results": [],
    }


def _dialog_state() -> dict[str, object]:
    """Build the direct renderer state for accepted-surface tests."""

    return {
        "text_surface_input_v2": _surface_input(),
        "text_surface_output_v2": _surface_output(),
        "cognitive_episode": canonical_episode(
            episode_id="transition-coherence-dialog",
            content="We are accomplices in this together.",
        ),
        "user_name": "Current User",
        "dialog_usage_mode": "unit_test",
        "llm_trace_id": "transition-coherence-dialog",
    }


def _dialog_verifier_aggregate(
    *,
    aligned: bool,
    semantic_issues: list[str],
) -> dict[str, object]:
    """Build the exact owner-preserving dialog verifier aggregate.

    Args:
        aligned: Whether semantic fidelity accepts the candidate.
        semantic_issues: Bounded semantic errors for a rejected candidate.

    Returns:
        Three-owner verifier state with role and surface checks aligned.
    """

    semantic_status = "aligned" if aligned else "misaligned"
    return {
        "semantic_fidelity": {
            "status": semantic_status,
            "issues": list(semantic_issues),
        },
        "role_direction": {
            "status": "aligned",
            "violations": [],
        },
        "surface_integrity": {
            "status": "aligned",
            "issues": [],
        },
    }


def test_v2_surface_services_have_no_independent_style_owner() -> None:
    """Normal V2 text planning exposes only content and preference configs."""

    assert [field.name for field in fields(contracts.TextSurfaceServicesV2)] == [
        "llm",
        "content_plan_config",
        "preference_config",
    ]


def test_v2_public_contract_exports_exact_delivery_profile() -> None:
    """The public package exports the named five-field delivery contract."""

    delivery_type = getattr(cognition_core_v2, "DeliveryProfileV2")

    assert delivery_type is contracts.DeliveryProfileV2
    assert set(delivery_type.__annotations__) == {
        "lexical_register",
        "sentence_shape",
        "rhythm",
        "hesitation",
        "punctuation",
    }


def test_character_constraints_require_personality_judgment() -> None:
    """Cognition receives semantic personality fields before stance selection."""

    character_state = build_character_production_state(updated_at=NOW)
    snapshot = {
        "drives": character_state["drives"],
        "standards": character_state["standards"],
        "meaning_state": character_state["meaning_state"],
        "personality_judgment": {
            "logic": "analytical",
            "defense": "reserved under embarrassment",
            "quirks": "occasionally self-conscious",
            "taboos": "preserve character agency",
        },
    }

    contracts._validate_character_constraints(snapshot)


def test_text_surface_contract_uses_split_context_and_delivery_profile() -> None:
    """The target V2 input and output shapes validate without legacy fields."""

    validated_input = contracts.validate_text_surface_input(_surface_input())
    validated_output = contracts.validate_text_surface_output(
        _surface_output()
    )

    assert set(validated_input["character_expression_context"]) == {
        "tempo",
        "linguistic_texture",
    }
    assert validated_output["delivery_profile"] == _delivery_profile()


@pytest.mark.asyncio
async def test_text_surface_planning_uses_two_atomic_owner_calls() -> None:
    """Content owns semantics and delivery while preference stays independent."""

    llm = _UnifiedSurfaceLLM()
    services = contracts.TextSurfaceServicesV2(
        llm=llm,
        content_plan_config=object(),
        preference_config=object(),
    )

    output = await surface.run_text_surface_planning(
        _surface_input(),
        services,
    )

    assert len(llm.calls) == 2
    assert output["delivery_profile"] == _delivery_profile()
    content_calls = [
        payload
        for system_prompt, payload in llm.calls
        if "delivery_profile" in system_prompt
    ]
    preference_calls = [
        payload
        for system_prompt, payload in llm.calls
        if "visible_boundaries" in system_prompt
    ]
    assert content_calls[0]["character_expression_context"] == {
        "tempo": "measured",
        "linguistic_texture": "light hesitation and rhythmic bounce",
    }
    assert "character_expression_context" not in preference_calls[0]
    assert all(
        "visual_character_context" not in payload
        for _, payload in llm.calls
    )


def test_surface_prompts_use_contextual_stance_and_boundary_scope() -> None:
    """Judge the whole stance while preserving characterful delivery."""

    for prompt in (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT,
        surface_stages.DIALOG_COMPLIANCE_REPAIR_SYSTEM_PROMPT,
    ):
        compact_prompt = "".join(prompt.split())
        assert "selectedintention及intention.reason为语义锚点" in (
            compact_prompt
        )
        assert "分清角色是在回应请求本身，还是在回应提问的时机、突然程度或直接程度" in (
            compact_prompt
        )
        assert "可自由组合惊讶、害羞、防御、调侃、嘴硬、迟疑、温柔、热烈或其他符合角色的情绪与特征" in (
            compact_prompt
        )
        assert "表达同一已选决定的角色化弧线" in compact_prompt
        assert "content_requirements使用正向目标句式" in compact_prompt

    preference_prompt = "".join(
        surface_stages.PREFERENCE_SYSTEM_PROMPT.split()
    )
    assert "每一条visible_boundaries都对应权威语境中明确生效的表达限制或细节范围" in (
        preference_prompt
    )
    assert "每一条addressee_plan都对应真实存在的称呼安排" in (
        preference_prompt
    )
    for prompt in (
        surface_stages.PREFERENCE_SYSTEM_PROMPT,
        surface_stages.DIALOG_COMPLIANCE_REPAIR_SYSTEM_PROMPT,
    ):
        compact_prompt = "".join(prompt.split())
        assert "普通场景事实、时间、情绪、关系状态和已选回应立场分别归入" in (
            compact_prompt
        )
        assert "拒绝、接受、指责、协商、条件和立场变化归入content_plan或content_requirements" in (
            compact_prompt
        )

    surface_repair_prompt = "".join(
        surface_stages.DIALOG_COMPLIANCE_REPAIR_SYSTEM_PROMPT.split()
    )
    assert "verified_hard_issues中的内容冲突对应content_plan和content_requirements中的正向修复目标" in (
        surface_repair_prompt
    )
    assert "visible_boundaries和addressee_plan仍各自取自权威语境中的具体来源" in (
        surface_repair_prompt
    )
    assert "没有具体来源时，这两个字段分别返回空列表" in (
        surface_repair_prompt
    )
    assert "addressee_plan的条目格式为“现有参与者+本轮实际使用的称呼形式”" in (
        surface_repair_prompt
    )
    assert "亲密感、语气词、词汇、句式和节奏由delivery_profile表达" in (
        surface_repair_prompt
    )
    assert "visible_boundaries的具体来源类型是权威语境明示的隐私、保密、同意、安全、内容审查或可见披露限制" in (
        surface_repair_prompt
    )
    assert "主题、比喻和已选立场进入content_plan或content_requirements" in (
        surface_repair_prompt
    )

    for prompt in (
        dialog_agent._V2_DIALOG_GENERATOR_PROMPT,
        dialog_agent._V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT,
    ):
        compact_prompt = "".join(prompt.split())
        assert "selected_surface_intent是本轮语义锚点" in (
            compact_prompt
        )
        assert "可自由组合惊讶、羞赧、防御、调侃、嘴硬、表面勉强、间接表达、温柔、热烈以及其他符合角色的情绪和特征" in (
            compact_prompt
        )
        assert "与后文共同传达同一已选决定" in compact_prompt
        assert "让相同语义呈现鲜明而多样的角色声音" in (
            compact_prompt
        )

    dialog_prompt = "".join(
        dialog_agent._V2_DIALOG_GENERATOR_PROMPT.split()
    )
    assert (
        "先整体阅读selected_surface_intent、content_plan、"
        "content_requirements、visible_boundaries和delivery_profile"
        in dialog_prompt
    )
    assert "判断规划中的开场反应指向行动或关系本身" in dialog_prompt
    repair_prompt = "".join(
        dialog_agent._V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT.split()
    )
    assert "先阅读text_surface_output_v2中的selected_surface_intent" in (
        repair_prompt
    )

    semantic_prompt = "".join(
        dialog_agent._V2_DIALOG_SEMANTIC_FIDELITY_PROMPT.split()
    )
    assert "依次阅读当前输入、权威语义和候选中的全部消息" in (
        semantic_prompt
    )
    assert "判断每句话回应的对象以及前后句如何承接" in semantic_prompt
    assert "先判断候选是否构成一条与selected_surface_intent一致的完整语义弧线" in (
        semantic_prompt
    )
    assert "分别提取开场与收尾的主体、行动或关系对象、肯定或否定极性" in (
        semantic_prompt
    )
    assert "针对提问时机、直接程度、标签或情绪的反应，按其真实对象判断" in (
        semantic_prompt
    )
    assert "惊讶、羞赧、防御、调侃、嘴硬、表面勉强、间接表达以及其他角色化情绪" in (
        semantic_prompt
    )
    assert "当这些表达的对象是时机、直接程度、标签或情绪" in (
        semantic_prompt
    )
    assert "且行动或关系极性与收尾一致时，整段属于aligned" in semantic_prompt
    assert "不论位于同一消息或多条消息" in semantic_prompt
    assert "对同一主体、同一行动或关系先明确拒绝或不愿，后明确接受或愿意" in (
        semantic_prompt
    )

    generation_prompts = (
        surface_stages.CONTENT_PLAN_SYSTEM_PROMPT,
        surface_stages.DIALOG_COMPLIANCE_REPAIR_SYSTEM_PROMPT,
        dialog_agent._V2_DIALOG_GENERATOR_PROMPT,
        dialog_agent._V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT,
    )
    for prompt in generation_prompts:
        compact_prompt = "".join(prompt.split())
        for overbroad_constraint in (
            "只能表现这个决定",
            "不能添加或改变拒绝",
            "不得新增权威语义中不存在的拒绝",
            "用户提出、追问、重复或坚持请求本身不是改变已选决定的新理由",
        ):
            assert overbroad_constraint not in compact_prompt


def test_surface_repair_uses_canonical_input_without_rejected_surface() -> None:
    """The repair API cannot echo rejected surface semantics or delivery."""

    signature = inspect.signature(surface.repair_text_surface_planning)

    assert list(signature.parameters) == [
        "input_payload",
        "verified_hard_issues",
        "services",
    ]


def test_semantic_fidelity_receives_surface_authority_and_exact_caps() -> None:
    """The focused semantic verifier owns bounded authoritative semantics."""

    signature = inspect.signature(
        dialog_agent._verify_dialog_semantic_fidelity
    )

    assert "surface_output" in signature.parameters
    assert dialog_agent.DIALOG_SEMANTIC_AUTHORITY_MAX_CHARS == 11000
    assert dialog_agent.DIALOG_CANDIDATE_MAX_CHARS == 12000
    assert dialog_agent.DIALOG_SEMANTIC_PAYLOAD_MAX_CHARS == 50000


@pytest.mark.asyncio
async def test_semantic_fidelity_payload_uses_only_surface_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delivery and action truth stay outside semantic verifier authority."""

    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"aligned": true, "hard_errors": []}',
    ))
    monkeypatch.setattr(
        dialog_agent,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_agent.llm_tracing,
        "record_llm_trace_step",
        AsyncMock(),
    )
    monkeypatch.setattr(
        dialog_agent.event_logging,
        "record_llm_stage_event",
        AsyncMock(),
    )

    await dialog_agent._verify_dialog_semantic_fidelity(
        surface_output=_surface_output(),
        generated_dialog=["We are in this together."],
        current_visible_percepts=[],
        llm_trace_id="semantic-authority",
    )

    payload = json.loads(
        semantic_llm.ainvoke.await_args.args[0][1].content
    )
    assert payload["authoritative_surface_semantics"] == {
        "selected_surface_intent": "confirm shared participation",
        "content_plan": "Confirm shared participation clearly and consistently.",
        "content_requirements": [
            "Keep one accepting stance throughout the response.",
        ],
        "visible_boundaries": [],
    }
    serialized_authority = json.dumps(
        payload["authoritative_surface_semantics"],
        ensure_ascii=False,
    )
    assert "delivery_profile" not in serialized_authority
    assert "permitted_action_results" not in serialized_authority


@pytest.mark.asyncio
async def test_semantic_fidelity_candidate_limit_degrades_before_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized candidate records typed verifier unavailability."""

    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock()
    trace_recorder = AsyncMock()
    contract_recorder = AsyncMock()
    monkeypatch.setattr(
        dialog_agent,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_agent.llm_tracing,
        "record_llm_trace_step",
        trace_recorder,
    )
    monkeypatch.setattr(
        dialog_agent.event_logging,
        "record_model_contract_event",
        contract_recorder,
    )

    verdict = await dialog_agent._verify_dialog_semantic_fidelity(
        surface_output=_surface_output(),
        generated_dialog=[
            "x" * (dialog_agent.DIALOG_CANDIDATE_MAX_CHARS + 1)
        ],
        current_visible_percepts=[],
        llm_trace_id="semantic-context-limit",
    )

    assert verdict == {"status": "unavailable", "issues": []}
    semantic_llm.ainvoke.assert_not_awaited()
    assert trace_recorder.await_args.kwargs["parse_status"] == (
        "not_called_context_limit"
    )
    assert trace_recorder.await_args.kwargs["status"] == "failed"
    assert contract_recorder.await_args.kwargs["violation_kind"] == (
        "semantic_verifier_context_limit"
    )


@pytest.mark.asyncio
async def test_dialog_repair_payload_excludes_rejected_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second renderer receives canonical replacement authority only."""

    replacement = _surface_output()
    replacement["content_plan"] = "State one coherent accepting position."
    surface_repair = AsyncMock(return_value=replacement)
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"final_dialog": ["We are in this together."]}',
    ))
    monkeypatch.setattr(
        dialog_agent,
        "repair_text_surface_for_dialog",
        surface_repair,
    )
    monkeypatch.setattr(
        dialog_agent,
        "_dialog_generator_llm",
        generator_llm,
    )
    monkeypatch.setattr(
        dialog_agent.llm_tracing,
        "record_llm_trace_step",
        AsyncMock(),
    )

    repaired_dialog, repaired_surface = (
        await dialog_agent._repair_dialog_hard_failure(
            repair_issues=["The response reverses stance without a cause."],
            surface_input=_surface_input(),
            user_name="Current User",
            llm_trace_id="candidate-exclusion",
        )
    )

    assert repaired_dialog == ["We are in this together."]
    assert repaired_surface == replacement
    surface_repair.assert_awaited_once_with(
        surface_input=_surface_input(),
        verified_hard_issues=[
            "The response reverses stance without a cause.",
        ],
    )
    repair_payload = json.loads(
        generator_llm.ainvoke.await_args.args[0][1].content
    )
    assert repair_payload == {
        "text_surface_output_v2": replacement,
        "user_name": "Current User",
        "repair_context": {
            "verified_hard_issues": [
                "The response reverses stance without a cause.",
            ],
        },
    }


@pytest.mark.asyncio
async def test_dialog_generator_returns_first_pass_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A first-pass dialog returns the surface that produced it."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content='{"final_dialog": ["We are in this together."]}',
    ))
    monkeypatch.setattr(
        dialog_agent,
        "_dialog_generator_llm",
        generator_llm,
    )
    monkeypatch.setattr(
        dialog_agent,
        "_verify_dialog_compliance",
        AsyncMock(return_value=_dialog_verifier_aggregate(
            aligned=True,
            semantic_issues=[],
        )),
    )
    monkeypatch.setattr(
        dialog_agent.llm_tracing,
        "record_llm_trace_step",
        AsyncMock(),
    )
    monkeypatch.setattr(
        dialog_agent.event_logging,
        "record_llm_stage_event",
        AsyncMock(),
    )

    result = await dialog_agent.dialog_generator(_dialog_state())

    assert result["text_surface_output_v2"] == _surface_output()


@pytest.mark.asyncio
async def test_dialog_generator_returns_repaired_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repaired dialog returns its replacement semantic surface."""

    replacement = _surface_output()
    replacement["content_plan"] = "State one coherent accepting position."
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(side_effect=[
        SimpleNamespace(
            content='{"final_dialog": ["Initial candidate."]}',
        ),
        SimpleNamespace(
            content='{"final_dialog": ["We are in this together."]}',
        ),
    ])
    surface_repair = AsyncMock(return_value=replacement)
    monkeypatch.setattr(
        dialog_agent,
        "_dialog_generator_llm",
        generator_llm,
    )
    monkeypatch.setattr(
        dialog_agent,
        "_verify_dialog_compliance",
        AsyncMock(side_effect=[
            _dialog_verifier_aggregate(
                aligned=False,
                semantic_issues=["Unsupported stance reversal."],
            ),
            _dialog_verifier_aggregate(
                aligned=True,
                semantic_issues=[],
            ),
        ]),
    )
    monkeypatch.setattr(
        dialog_agent,
        "repair_text_surface_for_dialog",
        surface_repair,
    )
    monkeypatch.setattr(
        dialog_agent.llm_tracing,
        "record_llm_trace_step",
        AsyncMock(),
    )
    monkeypatch.setattr(
        dialog_agent.event_logging,
        "record_llm_stage_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        dialog_agent.event_logging,
        "record_model_contract_event",
        AsyncMock(),
    )

    result = await dialog_agent.dialog_generator(_dialog_state())

    assert result["final_dialog"] == ["We are in this together."]
    assert result["text_surface_output_v2"] == replacement
    surface_repair.assert_awaited_once()


@pytest.mark.asyncio
async def test_persona_graph_stores_dialog_accepted_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The persona update replaces the initial surface after dialog repair."""

    initial_surface = _surface_output()
    replacement = _surface_output()
    replacement["content_plan"] = "State one coherent accepting position."
    monkeypatch.setattr(
        persona_supervisor2,
        "_execute_pre_surface_action_results",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        persona_supervisor2,
        "call_l3_text_surface_handler",
        AsyncMock(return_value={
            "text_surface_input_v2": _surface_input(),
            "text_surface_output_v2": initial_surface,
        }),
    )
    monkeypatch.setattr(
        persona_supervisor2,
        "_first_valid_action_attempt_id",
        MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        persona_supervisor2,
        "dialog_agent",
        AsyncMock(return_value={
            "final_dialog": ["We are in this together."],
            "target_addressed_user_ids": ["current-user"],
            "target_broadcast": False,
            "text_surface_output_v2": replacement,
        }),
    )
    monkeypatch.setattr(
        persona_supervisor2,
        "_action_results_for_state",
        AsyncMock(return_value=[]),
    )

    update = await persona_supervisor2.call_action_subgraph({
        "storage_timestamp_utc": NOW,
    })

    assert update["text_surface_output_v2"] == replacement
    assert update["final_dialog"] == ["We are in this together."]


@pytest.mark.asyncio
async def test_progress_receives_only_accepted_surface_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Progress receives accepted content and intent without delivery data."""

    accepted_surface = _surface_output()
    accepted_surface["content_plan"] = "Accepted repaired semantic plan."
    monkeypatch.setattr(
        post_turn,
        "_validated_episode_trace",
        lambda state, *, logger: {},
    )
    monkeypatch.setattr(
        post_turn,
        "_visible_trace_dialog",
        lambda trace: ["Accepted repaired dialog."],
    )
    recorder = AsyncMock(return_value={
        "written": True,
        "turn_count": 1,
        "continuity": "continued",
        "status": "recorded",
        "cache_updated": True,
    })
    state = {
        "character_profile": {
            "name": "Kazusa",
            "boundary_profile": {},
        },
        "text_surface_output_v2": accepted_surface,
        "platform": "debug",
        "platform_channel_id": "channel",
        "global_user_id": "current-user",
        "storage_timestamp_utc": NOW,
        "conversation_episode_state": None,
        "decontextualized_input": "We are accomplices.",
        "chat_history_recent": [],
        "logical_stance": "accept",
        "character_intent": "confirm shared participation",
    }

    await post_turn.run_conversation_progress_record_background(
        state,
        record_turn_progress_func=recorder,
        logger=logging.getLogger("transition-progress-test"),
    )

    record_input = recorder.await_args.kwargs["record_input"]
    assert record_input["content_plan"] == {
        "semantic_content": "Accepted repaired semantic plan.",
        "surface_intent": "confirm shared participation",
    }
    assert record_input["final_dialog"] == ["Accepted repaired dialog."]
    serialized_record = json.dumps(record_input, default=str)
    assert "delivery_profile" not in serialized_record
    assert "lexical_register" not in serialized_record
