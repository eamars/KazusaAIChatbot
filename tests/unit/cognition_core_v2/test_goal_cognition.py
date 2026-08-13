"""Deterministic ownership test for src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py."""

from __future__ import annotations

from importlib import import_module

import pytest

from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    CONTINUITY_AUTHORITY_INSTRUCTIONS,
    GOAL_COGNITION_PROMPT,
    NON_ORDINARY_GOAL_COGNITION_PROMPT,
    ORDINARY_RECURRENCE_GOAL_COGNITION_PROMPT,
    ORDINARY_RECURRENCE_SELECTION_GOAL_COGNITION_PROMPT,
    _build_goal_output_contract,
    _conversation_progress_evidence,
    validate_selection_goal_draft,
)
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    NO_ROLE,
)

MODULE_PATH = "kazusa_ai_chatbot.cognition_core_v2.goal_cognition"
EXPECTED_SYMBOLS = ["run_goal_cognition"]


def _input_operation() -> dict[str, object]:
    """Build the required-selection operation before character judgment."""

    return {
        "operation": "the character chooses a reward",
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": NO_ROLE,
        "embedded_target_role": NO_ROLE,
    }


def _selection_draft(
    selected_operation: dict[str, object],
) -> dict[str, object]:
    """Build one complete selection draft for the validator."""

    return {
        "selection": "choose a reward",
        "selected_response_operation": selected_operation,
        "reason": "the current episode requires a concrete choice",
        "private_monologue": "choose from the current request",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the selected reward is stated"],
        "confidence": "high",
    }


def test_goal_cognition_exposes_owned_contract() -> None:
    """Keep the module's named owner contract discoverable."""

    module = import_module(MODULE_PATH)
    missing_symbols = [
        symbol
        for symbol in EXPECTED_SYMBOLS
        if not hasattr(module, symbol)
    ]

    assert not missing_symbols, (
        f"{MODULE_PATH} is missing owner symbols: {missing_symbols}"
    )


def test_required_selection_emits_selected_response_operation() -> None:
    """Selection validation returns the operation chosen by cognition."""

    selected_operation = {
        **_input_operation(),
        "operation": "the user gives the selected reward to the character",
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }
    validated = validate_selection_goal_draft(
        _selection_draft(selected_operation),
        evidence_handles={"e1"},
        role_handles=set(),
        required_evidence_handles={"e1"},
        required_operations=[{
            "evidence_handle": "e1",
            "response_operation": _input_operation(),
        }],
        maximum_evidence_handles=4,
    )

    assert validated["selected_response_operation"] == selected_operation


def test_required_selection_rejects_fixed_role_conflict() -> None:
    """Selection validation rejects reversal of a known input endpoint."""

    selected_operation = {
        **_input_operation(),
        "operation": "the character gives the selected reward to the user",
        "embedded_actor_role": CURRENT_CHARACTER_ROLE,
        "embedded_target_role": CURRENT_USER_ROLE,
    }

    with pytest.raises(ValueError, match="known input role"):
        validate_selection_goal_draft(
            _selection_draft(selected_operation),
            evidence_handles={"e1"},
            role_handles=set(),
            required_evidence_handles={"e1"},
            required_operations=[{
                "evidence_handle": "e1",
                "response_operation": {
                    **_input_operation(),
                    "embedded_actor_role": CURRENT_USER_ROLE,
                },
            }],
            maximum_evidence_handles=4,
        )


def test_goal_prompt_labels_confidence_as_descriptor() -> None:
    """Goal prompts distinguish semantic confidence from evaluator scores."""

    prompt = GOAL_COGNITION_PROMPT

    assert "confidence 是有界的置信度描述" in prompt
    assert "不是 score" in prompt


def test_conversation_progress_evidence_preserves_temporal_provenance() -> None:
    """Goal projection carries source time and deterministic age metadata."""

    projected = _conversation_progress_evidence([{
        'evidence_handle': 'e-progress',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-progress-event:1',
            'occurred_at': '2026-07-30T00:00:00Z',
            'semantic_summary': 'same concrete matter',
        },
        'semantic_text': 'the same concrete matter remains open',
        'visible_to': ['q:event_agency'],
        'authority': 'participant_continuity',
        'temporal_provenance': {
            'occurred_at': '2026-07-30T00:00:00Z',
            'age_descriptor': 'recent',
        },
    }])

    assert projected == [{
        'evidence_handle': 'e-progress',
        'semantic_text': 'the same concrete matter remains open',
        'authority': 'participant_continuity',
        'temporal_provenance': {
            'occurred_at': '2026-07-30T00:00:00Z',
            'age_descriptor': 'recent',
        },
    }]


def test_goal_prompt_declares_one_primary_current_scene_objective() -> None:
    """Every goal mode receives the same generic evidence-authority contract."""

    prompts = (
        GOAL_COGNITION_PROMPT,
        ORDINARY_RECURRENCE_GOAL_COGNITION_PROMPT,
        ORDINARY_RECURRENCE_SELECTION_GOAL_COGNITION_PROMPT,
        NON_ORDINARY_GOAL_COGNITION_PROMPT,
    )
    for prompt in prompts:
        assert '当前事件' in prompt
        assert '目标' in prompt
        assert '每次只选择一个当前目标' in (
            prompt + CONTINUITY_AUTHORITY_INSTRUCTIONS
        )
        rendered_contract = prompt + CONTINUITY_AUTHORITY_INSTRUCTIONS
        for authority in (
            'current_event',
            'public_scene',
            'participant_continuity',
            'private_motive_only',
            'conditional_character_guidance',
        ):
            assert f'`{authority}`' in rendered_contract
        assert '所有返回的目标说明和预期后果都必须共同服务这个主要目标' in (
            rendered_contract
        )
        assert '连贯的从属行动' in rendered_contract
        assert '归属于某位说话者的公开发言' in rendered_contract
        assert '不单独证明其声称的外部命题' in rendered_contract
        assert '最新的第一人称更正或状态更新具有更高的判断权重' in (
            rendered_contract
        )
        assert '共享公开事项仍未解决' in rendered_contract
        assert '单项核查、子问题、局部信号或从属步骤' in rendered_contract
        assert '不足以单独证明整个事项已经解决' in rendered_contract
        assert '把竞争事项提升为主要目标前' in rendered_contract
        assert '当前可观察证据' in rendered_contract
        assert '对整个事项具有判断权的参与者作出明确更正' in (
            rendered_contract
        )
        assert '只有满足这一条件，才可开始新的主要话题' in (
            rendered_contract
        )
        assert '否则保留原事项为唯一主要目标' in rendered_contract
        assert '竞争内容只能作为对原目标的直接支持、拒绝或推后处理' in (
            rendered_contract
        )
        assert '可以质疑或核实冲突主张' in rendered_contract
        assert '角色仍自主决定立场以及服务主要目标的从属行动方式和先后' in (
            rendered_contract
        )
        assert '只允许原样出现在各自的类型化 handle 字段' in (
            rendered_contract
        )
        assert '必须使用语义角色描述或自然指代' in rendered_contract
        assert '不得复写任何 handle token 或内部标识' in rendered_contract
        assert '输入请求或模型设想的现实世界实体交互、感知或观察' in (
            rendered_contract
        )
        assert '提议或取得证据后再回应的目标' in rendered_contract
        assert '不能描述该行为已经发生' in rendered_contract
        assert '观察已经完成或相应结果已经确定' in rendered_contract
    assert '每次只选择一个当前目标' in CONTINUITY_AUTHORITY_INSTRUCTIONS
    assert '所有从属行动必须服务于同一个目标' in (
        CONTINUITY_AUTHORITY_INSTRUCTIONS
    )


def test_goal_output_contract_keeps_existing_schema() -> None:
    """Keep the generic goal schema and consequence bounds stable."""

    generic_contract = _build_goal_output_contract(
        evidence_handles={'e1'},
        episode_evidence_handles={'e1'},
        required_evidence_handles=set(),
        role_bindings={},
        selection_required=False,
        require_relational_willingness=False,
        maximum_evidence_handles=9,
    )
    selection_contract = _build_goal_output_contract(
        evidence_handles={'e1'},
        episode_evidence_handles={'e1'},
        required_evidence_handles={'e1'},
        role_bindings={},
        selection_required=True,
        require_relational_willingness=False,
        maximum_evidence_handles=9,
    )

    expected_fields = (
        {
            'intention',
            'desired_outcome',
            'concrete_detail',
            'reason',
            'private_monologue',
            'target_role_handles',
            'evidence_handles',
            'expected_consequences',
            'confidence',
        },
        {
            'selection',
            'selected_response_operation',
            'reason',
            'private_monologue',
            'target_role_handles',
            'evidence_handles',
            'expected_consequences',
            'confidence',
        },
    )
    for contract, fields in zip(
        (generic_contract, selection_contract),
        expected_fields,
        strict=True,
    ):
        assert set(contract['top_level_fields']) == fields
        assert contract['bounds']['expected_consequences'] == {
            'minimum_items': 1,
            'maximum_items': 8,
            'item_maximum_chars': 240,
        }
