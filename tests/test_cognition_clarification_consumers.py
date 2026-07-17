"""V2 clarification ownership tests."""

import inspect

from kazusa_ai_chatbot.cognition_core_v2 import action_selection
from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
    EMOTION_DEFINITIONS,
)


def test_clarification_is_a_semantic_tendency_and_resolver_route() -> None:
    """Branches can request clarification through the typed evidence route."""

    tendencies = {
        tendency
        for definition in EMOTION_DEFINITIONS.values()
        for tendency in definition.action_tendencies
    }

    assert "clarify" in tendencies
    assert "evidence" in action_selection.ACTION_PLANNING_PROMPT
    assert "resolver_handle" in action_selection.ACTION_PLANNING_PROMPT


def test_clarification_validation_does_not_classify_user_text() -> None:
    """Code validates handles while the model owns semantic interpretation."""

    source = inspect.getsource(
        action_selection._validate_action_plan_decision
    )

    assert "bid_handle" in source
    assert "user_input" not in source
    assert "keyword" not in source
