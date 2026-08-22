"""Preference surface ownership tests."""

import inspect

from kazusa_ai_chatbot.cognition_shared.surface_stages import (
    PREFERENCE_SYSTEM_PROMPT,
)
from kazusa_ai_chatbot.cognition_shared import surface_stages


def test_preference_stage_owns_visible_boundaries_only() -> None:
    """Preferences shape rendering without rewriting cognition."""

    prompt = PREFERENCE_SYSTEM_PROMPT.casefold()

    assert "typed source-bound visible-boundary contract" in prompt
    assert "visible_boundaries 始终返回空列表" in prompt
    assert "相应约束为空时返回空列表" in prompt
    assert "dialog 生成" in prompt
    assert "只返回一个 json 对象" in prompt


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
