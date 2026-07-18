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
    assert "没有相应约束时返回空列表" in prompt
    assert "不写最终对话" in prompt


def test_preference_stage_has_no_keyword_based_user_input_adapter() -> None:
    """The LLM owns preference meaning; code only bounds its typed result."""

    source = inspect.getsource(surface_stages.run_preference_stage)

    assert "services.llm" in source
    assert ".ainvoke" in source
    assert "user_input" not in source
    assert "keyword" not in source
