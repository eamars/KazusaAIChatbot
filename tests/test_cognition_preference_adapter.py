"""V2 preference surface ownership tests."""

import inspect

from kazusa_ai_chatbot.cognition_core_v2 import surface_stages
from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    SURFACE_STAGE_PROMPTS,
)


def test_preference_stage_owns_visible_boundaries_only() -> None:
    """Preferences shape rendering without rewriting cognition."""

    prompt = SURFACE_STAGE_PROMPTS["preference"].casefold()

    assert "preference-sensitive boundaries" in prompt
    assert "without writing dialogue" in prompt


def test_preference_stage_has_no_keyword_based_user_input_adapter() -> None:
    """The LLM owns preference meaning; code only bounds its typed result."""

    source = inspect.getsource(surface_stages.run_surface_stage)

    assert "services.llm.ainvoke" in source
    assert "user_input" not in source
    assert "keyword" not in source
