"""V2 action-selection prompt contract tests."""

from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    ACTION_PLANNING_PROMPT,
)


def test_action_prompt_exposes_fixed_compositional_shape() -> None:
    """The planner receives handles and one closed output vocabulary."""

    prompt = ACTION_PLANNING_PROMPT.casefold()

    for field in (
        "route",
        "action_requests",
        "resolver_requests",
        "resolver_pending_resolution",
        "resolver_goal_progress",
        "bid_handle",
        "decision",
    ):
        assert field in prompt
    for route in ("speech", "evidence", "action", "deferral", "silence"):
        assert route in prompt


def test_action_prompt_excludes_retired_action_router_authority() -> None:
    """V2 action routing has no V1 willingness or executor vocabulary."""

    prompt = ACTION_PLANNING_PROMPT.casefold()

    for retired_term in (
        "task_willingness",
        "background_work_allowed",
        "worker_metadata",
        "queue_state",
    ):
        assert retired_term not in prompt
    assert "up to three" in prompt
    assert "speech" in prompt
    assert "non-speech" in prompt


def test_action_prompt_requires_grounded_out_of_turn_effect() -> None:
    """Planner reasoning cannot be converted into a durable action request."""

    prompt = ACTION_PLANNING_PROMPT.casefold()

    assert "durable or out-of-turn effect" in prompt
    assert "planner's own reasoning" in prompt
    assert "reply preparation" in prompt
    assert "no supplied action capability actuates" in prompt
    assert "physical-action description" in prompt
