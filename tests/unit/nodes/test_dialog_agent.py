"""Direct ownership tests for terminal dialog generation."""

from __future__ import annotations

from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_generator
from tests.unit.nodes.dialog_fixtures import build_dialog_state


def test_dialog_agent_exposes_owned_contract() -> None:
    """Keep terminal dialog generation attached to this source owner."""

    assert callable(dialog_generator)


def test_dialog_prompt_prioritizes_epistemic_boundary() -> None:
    """Keep P-owned assertion authority above lower surface plan fields."""

    prompt = dialog_module._V2_DIALOG_GENERATOR_PROMPT

    assert "epistemic_boundary" in prompt
    assert "它的权威高于" in prompt
    assert "未观察到的特征不能用来排除" in prompt
    assert "从句、前提句、原因连接和反问" in prompt
    assert "输出前逐句检查可见断言" in prompt
    assert "不用动作舞台提示、拟声" in prompt
    assert "低于 permitted_action_results 的事实权威" in prompt
    assert "action_kind=speak 只授权说出或发送 final_dialog 的文字" in prompt
    assert "同一类型、同一效果的 executed 行精确支持" in prompt
    assert "输出前不可跳过的合同检查" in prompt
    assert "对未来外部效果的具体承诺也属于行动主张" in prompt
    assert "pending、scheduled 或 executed 行" in prompt


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
