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
    assert "协议代码会在语义授权完成后派生 route" in prompt
    assert '"route"' not in prompt


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
    assert "最多包含三项" in prompt
    assert "发言" in prompt
    assert "语义能力请求" in prompt


def test_action_prompt_requires_grounded_out_of_turn_effect() -> None:
    """Planner reasoning cannot be converted into a durable action request."""

    prompt = ACTION_PLANNING_PROMPT.casefold()

    assert "持久化或跨轮效果" in prompt
    assert "规划者本轮的推理" in prompt
    assert "回复准备" in prompt
    assert "能力不会驱动角色身体" in prompt
    assert "身体动作表演描述" in prompt


def test_action_prompt_assigns_goal_ledger_shape_to_protocol_code() -> None:
    """The model emits semantic progress deltas, not duplicate state ledgers."""

    prompt = ACTION_PLANNING_PROMPT.casefold()

    assert "局部语义更新" in prompt
    assert "确定性代码" in prompt
    assert "保留省略" in prompt
