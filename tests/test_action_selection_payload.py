"""V2 semantic action-planning payload ownership tests."""

from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    ACTION_PLANNING_PROMPT,
)


def test_action_prompt_owns_semantic_selection_only() -> None:
    """The planner selects grounded objectives without rewriting motives."""

    prompt = ACTION_PLANNING_PROMPT.casefold()

    assert "bid_handle" in prompt
    assert "rewrite bid content" in prompt
    assert "protocol code derives route" in prompt
    assert "semantic_goal" in prompt
    assert "decision" in prompt
    assert "task_willingness" not in prompt
    assert "raw media" not in prompt
