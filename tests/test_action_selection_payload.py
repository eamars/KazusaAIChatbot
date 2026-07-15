"""V2 route-selection payload ownership tests."""

from kazusa_ai_chatbot.cognition_core_v2.action_selection import ROUTE_PROMPT


def test_route_prompt_owns_selection_only() -> None:
    """The V2 router selects handles and cannot author semantic content."""

    prompt = ROUTE_PROMPT.casefold()

    assert "selected_bid_handle" in prompt
    assert "do not author intention" in prompt
    assert "task_willingness" not in prompt
    assert "raw media" not in prompt
