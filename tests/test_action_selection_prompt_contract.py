"""V2 action-selection prompt contract tests."""

from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    ACTION_PLANNING_PROMPT,
    ACTION_PLANNING_PROMPT_CAP,
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


def test_action_prompt_exposes_one_generic_task_resolution_capability() -> None:
    """The planner judges evidence need without selecting execution mechanics."""

    prompt = ACTION_PLANNING_PROMPT.casefold()

    assert "task_resolution_request" in prompt
    for forbidden in (
        "text_artifact",
        "inline_budget_seconds",
        "requested_worker",
        "lease",
    ):
        assert forbidden not in prompt


def test_action_prompt_requires_exact_task_resolution_routing_boolean() -> None:
    """The generic resolver row carries exactly one JSON boolean."""

    prompt = "".join(ACTION_PLANNING_PROMPT.split())

    assert "start_in_background" in prompt
    assert "JSON布尔值" in prompt
    assert "直接进入持久后台路径执行" in prompt
    assert "近似前台预算尝试内联执行" in prompt
    assert "task_resolution_request行必须恰好包含bid_handle" in prompt
    assert "其他resolver行必须恰好包含bid_handle" in prompt
    assert "本阶段不选择worker、队列、时限或执行参数" in prompt


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


def test_action_prompt_keeps_character_self_report_out_of_optional_retrieval() -> None:
    """A character's own current report is not a missing external fact."""

    prompt = ACTION_PLANNING_PROMPT.casefold()

    assert "角色自己的当前感受、经历、偏好或判断" in prompt
    assert "不能证明角色自己的私密状态" in prompt
    assert "直接自我报告" in prompt


def test_action_prompt_fits_the_system_inclusive_budget() -> None:
    """The static planner contract leaves room for its dynamic packet."""

    assert len(ACTION_PLANNING_PROMPT) < ACTION_PLANNING_PROMPT_CAP


def test_action_prompt_states_request_fidelity_generation_procedure() -> None:
    """The planner identifies the requested effect before deciding resolver need."""

    prompt = "".join(ACTION_PLANNING_PROMPT.split()).casefold()

    assert "生成步骤" in prompt
    assert "先识别当前用户请求要达成的效果" in prompt
    assert "目标对象、范围和明确的时间约束" in prompt
    assert "semantic_goal必须忠实保留用户要求的检索或工作效果" in prompt
    assert "reason只解释该请求如何推进已接纳目标，不是第二个目标" in prompt


def test_action_prompt_labels_evidence_authority_deterministically() -> None:
    """Current-request rows are authoritative; context rows are supporting."""

    prompt = "".join(ACTION_PLANNING_PROMPT.split())

    assert "provenance_role" in prompt
    assert "current_episode是当前用户请求与当前场景的权威来源" in prompt
    assert "current_user_history_only" in prompt
    assert "character_or_world_context_only" in prompt
    assert "contextual_fact_only" in prompt
    assert "只提供支持性上下文" in prompt


def test_action_prompt_keeps_capability_audit_for_explicit_questions() -> None:
    """Runtime constraints become objectives only when the user asks for them."""

    prompt = "".join(ACTION_PLANNING_PROMPT.split())

    assert "能力、权限、可行性和API支持是运行时约束" in prompt
    assert "除非当前用户明确要求审核是否能做、是否被允许或是否可行" in prompt
    assert "只有当当前用户明确询问能力、权限或可行性本身时" in prompt
    assert "semantic_goal才可以是该审计目标" in prompt


def test_action_prompt_requires_evidence_for_unanswered_explicit_audits() -> None:
    """An unanswered capability question remains a resolver-owned audit."""

    prompt = "".join(ACTION_PLANNING_PROMPT.split())

    assert "可信的运行时限制和证据不足以回答时" in prompt
    assert "goal_resolution必须为requires_required_evidence" in prompt
    assert "task_resolution_request保留该审计目标" in prompt
    assert "不得仅凭bid或角色自述将其判为answerable_now" in prompt


def test_action_prompt_does_not_create_empty_goal_progress_checklists() -> None:
    """A new resolver request does not invent a capability checklist."""

    prompt = "".join(ACTION_PLANNING_PROMPT.split())

    assert "current_resolver_goal_progress是空壳" in prompt
    assert "resolver_goal_progress必须为null" in prompt
    assert "不要仅因为选择了resolver就新建目标清单" in prompt
    assert "普通检索请求不能把能力、权限或可行性核验写成deliverable" in prompt


def test_action_prompt_forbids_deterministic_semantic_rewriting() -> None:
    """Semantic meaning stays model-owned with no keyword routing or filters."""

    prompt = ACTION_PLANNING_PROMPT.casefold()

    assert "不要添加关键词路由、确定性" in prompt
    assert "后处理" in prompt
    assert "不要依据用户原文关键词进行这个分类" in prompt
