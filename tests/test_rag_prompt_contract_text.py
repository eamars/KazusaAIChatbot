"""Deterministic prompt contract checks for RAG synthesis stages."""

from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_evaluator as rag_module


def test_rag_summarizer_prompt_keeps_cross_source_consistency_conservative() -> None:
    """Slot summaries must not turn adjacent links into source agreement."""

    prompt = rag_module._EVALUATOR_SUMMARIZER_PROMPT

    assert "当前 raw_result 没有确认的类别" in prompt
    assert "只能写成邻近线索" in prompt
    assert "不得把“未发现冲突”或“信息一致”写进摘要" in prompt


def test_rag_finalizer_prompt_keeps_cross_source_consistency_conservative() -> None:
    """RAG final answers must separate direct evidence from adjacent tracks."""

    prompt = rag_module._FINALIZER_PROMPT

    assert "跨来源一致性必须保守" in prompt
    assert "该来源类别未确认目标事实" in prompt
    assert "不要把相邻对象、派生轨道、集成说明" in prompt
