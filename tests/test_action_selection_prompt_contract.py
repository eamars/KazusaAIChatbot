"""V2 action-selection prompt contract tests."""

from kazusa_ai_chatbot.cognition_core_v2.action_selection import ROUTE_PROMPT


def test_route_prompt_exposes_only_frozen_route_fields() -> None:
    """The route model receives handles and a closed route vocabulary."""

    prompt = ROUTE_PROMPT.casefold()

    for field in (
        "selected_bid_handle",
        "route",
        "action_handle",
        "resolver_handle",
    ):
        assert field in prompt
    for route in ("speech", "evidence", "action", "deferral", "silence"):
        assert route in prompt


def test_route_prompt_excludes_retired_action_router_authority() -> None:
    """V2 action routing has no V1 willingness or executor vocabulary."""

    prompt = ROUTE_PROMPT.casefold()

    for retired_term in (
        "task_willingness",
        "background_work_allowed",
        "worker_metadata",
        "queue_state",
    ):
        assert retired_term not in prompt
