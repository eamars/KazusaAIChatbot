"""Direct ownership tests for terminal dialog verification."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import CognitionContractError
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    NO_ROLE,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import (
    DialogGenerationContractError,
    StateContractError,
    dialog_generator,
)
from tests.unit.nodes.dialog_fixtures import build_dialog_state
from tests.cognition_core_v2_test_helpers import canonical_episode


def _input_operation() -> dict[str, object]:
    """Build an input operation whose action endpoints are unresolved."""

    return {
        "operation": "the character chooses a reward",
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": NO_ROLE,
        "embedded_target_role": NO_ROLE,
    }


def _selected_operation() -> dict[str, object]:
    """Build the resolved post-selection operation."""

    return {
        **_input_operation(),
        "operation": "the user gives the selected reward to the character",
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }


@pytest.mark.asyncio
async def test_terminal_candidate_opposite_polarity_is_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opposite-polarity terminal candidate is withheld after rejection."""

    invalid_dialog = "Ask me what to do next; I will follow your choice."
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=type(
        "Response",
        (),
        {"content": json.dumps({"final_dialog": [invalid_dialog]})},
    )())
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=type(
        "Response",
        (),
        {
            "content": json.dumps({
                "score": 0.1,
                "hard_errors": ["Subject reversal remains."],
            })
        },
    )())
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(return_value=type(
        "Response",
        (),
        {"content": '{"score": 1.0, "issues": []}'},
    )())
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )

    with pytest.raises(DialogGenerationContractError):
        await dialog_generator(build_dialog_state())

    assert generator_llm.ainvoke.await_count == 3
    assert semantic_llm.ainvoke.await_count == 3
    assert surface_llm.ainvoke.await_count == 3


def test_dialog_role_direction_uses_selected_response_operation() -> None:
    """Dialog role payloads use the post-selection operation, not stale input."""

    selected_operation = _selected_operation()
    role_operations = dialog_module._required_selection_role_operations(
        [{
            "input_source": "dialog_text",
            "content": {"response_operation": _input_operation()},
        }],
        selected_response_operation=selected_operation,
    )

    assert role_operations == [{
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }]


@pytest.mark.asyncio
async def test_dialog_role_direction_rejects_selected_actor_target_reversal() -> None:
    """The surface boundary rejects a selected actor/target reversal."""

    input_operation = {
        **_input_operation(),
        "embedded_actor_role": CURRENT_CHARACTER_ROLE,
        "embedded_target_role": CURRENT_USER_ROLE,
    }
    selected_operation = _selected_operation()
    state = build_dialog_state()
    state["cognitive_episode"] = canonical_episode(
        episode_id="dialog-selected-operation-reversal",
        content="current reward request",
        metadata={"response_operation": input_operation},
    )
    surface_input = state["text_surface_input_v2"]
    surface_input["episode"] = state["cognitive_episode"]
    surface_input["intention"][
        "selected_response_operation"
    ] = selected_operation
    surface_input["selected_response_operation"] = selected_operation

    with pytest.raises(CognitionContractError, match="known input role"):
        await dialog_module._verify_dialog_role_direction(
            surface_output=state["text_surface_output_v2"],
            generated_dialog=["The user gives the selected reward."],
            current_visible_percepts=[{
                "input_source": "dialog_text",
                "content": {"response_operation": input_operation},
            }],
            surface_input=surface_input,
            llm_trace_id="dialog-selected-operation-reversal",
        )


def test_role_direction_prompt_allows_compatible_nested_request_actions() -> None:
    """Role direction compares the role tuple to the corresponding action."""

    prompt = dialog_module._V2_DIALOG_ROLE_DIRECTION_PROMPT

    assert "required_role_operations" in prompt
    assert "权威角色字段" in prompt
    assert "不含 operation 文本" in prompt
    assert "用于在候选措辞中识别与角色元组对应的所选动作" in prompt
    assert "authoritative_surface_semantics" in prompt
    assert "不逐句核对每个语法动词" in prompt
    assert "同一个选定嵌入动作" in prompt
    assert "包装层主语可以与嵌入动作的行动者不同" in prompt
    assert "不得因此报告为 role reversal" in prompt
    assert "多种合理读法按高分处理" in prompt
    assert "唯一明确" in prompt
    assert "selection_owner_transfer" in prompt
    assert "typed_operation_role_reversal" in prompt
    assert "只有要求用户决定选择哪项动作才是转移选择权" in prompt
    assert "每个元组的 operation" not in prompt


def test_dialog_agent_exposes_owned_contract() -> None:
    """Keep terminal dialog generation attached to this source owner."""

    assert callable(dialog_generator)


def test_dialog_score_is_numeric_quality_signal() -> None:
    """Dialog quality uses evaluator-owned numeric scores, not confidence."""

    score = dialog_module._validate_numeric_score(
        0.75,
        label="dialog",
    )

    assert score == 0.75
    assert isinstance(score, float)


def test_numeric_score_rejects_boolean_and_out_of_range_values() -> None:
    """Boolean, non-finite, and out-of-range scores fail closed."""

    for value in (True, False, -0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(StateContractError, match="score"):
            dialog_module._validate_numeric_score(
                value,
                label="dialog",
            )


def test_validated_dialog_messages_collapses_blank_line_runs() -> None:
    """Collapse internal blank lines while preserving message boundaries."""

    value = {
        "final_dialog": [
            "first\n\nsecond\n\nthird\n\nfourth\n\nfifth",
            "single\nline",
        ],
    }

    validated_messages = dialog_module._validated_dialog_messages(value)

    assert validated_messages == [
        "first\nsecond\nthird\nfourth\nfifth",
        "single\nline",
    ]
