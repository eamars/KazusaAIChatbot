"""Deterministic text-surface preference projection tests."""

import inspect

from kazusa_ai_chatbot.cognition_shared import surface


def test_surface_projection_owns_visible_boundaries_deterministically() -> None:
    """Deterministic code owns empty boundaries and exact addressee rows."""

    source = inspect.getsource(surface._run_text_surface_planning)

    assert '"visible_boundaries": []' in source
    assert "addressee_plan" in source
    assert "run_preference_stage" not in source


def test_surface_projection_has_no_keyword_based_user_input_adapter() -> None:
    """Structural projection remains independent of user-text classifiers."""

    source = inspect.getsource(surface._run_text_surface_planning)

    assert "user_input" not in source
    assert "keyword" not in source
