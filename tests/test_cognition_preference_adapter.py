"""V2 preference surface ownership tests."""

import inspect

from kazusa_ai_chatbot.cognition_core_v2 import surface_stages
from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    PREFERENCE_SYSTEM_PROMPT,
)


def test_preference_stage_owns_visible_boundaries_only() -> None:
    """Preferences shape rendering without rewriting cognition."""

    prompt = PREFERENCE_SYSTEM_PROMPT.casefold()

    assert "真实存在的可见表达边界" in prompt
    assert "相应约束为空时返回空列表，让角色按当前判断自然表达" in prompt
    assert "最终对话由 dialog 渲染器生成" in prompt
    assert "本阶段返回规划字段" in prompt


def test_preference_stage_has_no_keyword_based_user_input_adapter() -> None:
    """The LLM owns preference meaning; code only bounds its typed result."""

    source = "\n".join((
        inspect.getsource(surface_stages.run_preference_stage),
        inspect.getsource(surface_stages._run_surface_stage),
    ))

    assert "services.llm" in source
    assert ".ainvoke" in source
    assert "user_input" not in source
    assert "keyword" not in source
