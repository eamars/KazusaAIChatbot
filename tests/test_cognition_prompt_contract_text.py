"""Deterministic prompt-text contracts for the canonical V2 stages."""

from __future__ import annotations

import importlib
import json

from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    ACTION_AUTHORIZATION_PROMPT,
    _authorization_repair_message,
)
from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    ACTION_PLANNING_PROMPT,
    _action_planning_repair_message,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    GOAL_COGNITION_PROMPT,
    GOAL_COGNITION_REPAIR_PROMPT,
    MAX_GOAL_BID_EVIDENCE_HANDLES,
    MAX_GOAL_BID_ROLE_HANDLES,
    REQUIRED_SELECTION_GOAL_PROMPT,
)
from kazusa_ai_chatbot.cognition_core_v2.resolver_authorization import (
    RESOLVER_AUTHORIZATION_PROMPT,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    SEMANTIC_APPRAISAL_PROMPT,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_source_planner import (
    _QUESTION_DESCRIPTIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    CONTENT_PLAN_SYSTEM_PROMPT,
    PREFERENCE_SYSTEM_PROMPT,
    VISUAL_SYSTEM_PROMPT,
)
from kazusa_ai_chatbot.cognition_core_v2.workspace import COLLAPSE_PROMPT
from kazusa_ai_chatbot.consolidation import lane_router
from kazusa_ai_chatbot.consolidation import reflection
from kazusa_ai_chatbot.nodes import dialog_agent


frontline_relevance_agent = importlib.import_module(
    "kazusa_ai_chatbot.relevance.frontline_relevance_agent"
)
persona_relevance_agent = importlib.import_module(
    "kazusa_ai_chatbot.relevance.persona_relevance_agent"
)


def _branch_modified_prompt_fragments() -> tuple[str, ...]:
    """Return every branch-modified ordinary-instruction prompt fragment."""

    frontline_fragments = (
        frontline_relevance_agent._FRONTLINE_SYSTEM_PROMPT_COMMON,
        frontline_relevance_agent._FRONTLINE_GROUP_ACTION_CONTRACT,
        frontline_relevance_agent._FRONTLINE_PRIVATE_ACTION_CONTRACT,
        frontline_relevance_agent._FRONTLINE_AUTHORITATIVE_GROUP_ACTION_CONTRACT,
        frontline_relevance_agent._FRONTLINE_OPEN_OUTPUT_CONTRACT,
        frontline_relevance_agent._FRONTLINE_NO_OPEN_OUTPUT_CONTRACT,
        frontline_relevance_agent._FRONTLINE_WITH_PRELUDES_CONTRACT,
        frontline_relevance_agent._FRONTLINE_NO_PRELUDES_CONTRACT,
        frontline_relevance_agent._FRONTLINE_AUTHORITATIVE_OPEN_OUTPUT_CONTRACT,
        frontline_relevance_agent._FRONTLINE_AUTHORITATIVE_START_OUTPUT_CONTRACT,
    )
    settled_fragments = (
        persona_relevance_agent._SETTLED_SYSTEM_PROMPT_COMMON,
        persona_relevance_agent._SETTLED_WAIT_ACTION_CONTRACT,
        persona_relevance_agent._SETTLED_FINAL_ACTION_CONTRACT,
        persona_relevance_agent._SETTLED_AUTHORITATIVE_ACTION_CONTRACT,
        persona_relevance_agent._SETTLED_AUTHORITATIVE_REPAIR_PROMPT,
        *persona_relevance_agent._AUTHORITATIVE_DISPOSITION_GUIDANCE.values(),
    )
    return (
        ACTION_AUTHORIZATION_PROMPT,
        ACTION_PLANNING_PROMPT,
        GOAL_COGNITION_PROMPT,
        GOAL_COGNITION_REPAIR_PROMPT,
        REQUIRED_SELECTION_GOAL_PROMPT,
        RESOLVER_AUTHORIZATION_PROMPT,
        SEMANTIC_APPRAISAL_PROMPT,
        CONTENT_PLAN_SYSTEM_PROMPT,
        PREFERENCE_SYSTEM_PROMPT,
        VISUAL_SYSTEM_PROMPT,
        COLLAPSE_PROMPT,
        lane_router._ROUTER_PROMPT,
        reflection._CHARACTER_STATE_REVIEW_PROMPT,
        reflection._RELATIONSHIP_PROFILE_REVIEW_PROMPT,
        dialog_agent._V2_DIALOG_GENERATOR_PROMPT,
        dialog_agent._V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT,
        dialog_agent._V2_DIALOG_SEMANTIC_FIDELITY_PROMPT,
        dialog_agent._V2_DIALOG_ROLE_DIRECTION_PROMPT,
        dialog_agent._V2_DIALOG_SURFACE_INTEGRITY_PROMPT,
        *frontline_fragments,
        *settled_fragments,
    )


def test_branch_modified_runtime_prompts_use_chinese_instructions() -> None:
    """Keep ordinary runtime instructions in the cognition carrier language."""

    retired_english_instruction_markers = (
        "You are ",
        "Return exactly ",
        "Output Format",
        "Decision Procedure",
        "Rendering Procedure",
        "Do not ",
        "Never ",
    )
    for prompt in _branch_modified_prompt_fragments():
        assert any("\u4e00" <= character <= "\u9fff" for character in prompt)
        for marker in retired_english_instruction_markers:
            assert marker not in prompt

    repair_messages = (
        _authorization_repair_message(
            response_text="invalid",
            contract_error="invalid",
            candidate_handles=["c1"],
        ),
        _action_planning_repair_message(
            response_text="invalid",
            contract_error="invalid",
        ),
    )
    for message in repair_messages:
        repair_instruction = json.loads(str(message.content))["repair_instruction"]
        assert any(
            "\u4e00" <= character <= "\u9fff"
            for character in repair_instruction
        )

    for question in _QUESTION_DESCRIPTIONS.values():
        assert any("\u4e00" <= character <= "\u9fff" for character in question)
        assert "Assess " not in question

    bounded_authorization = json.loads(str(_authorization_repair_message(
        response_text="x" * 4000,
        contract_error="invalid",
        candidate_handles=["c1"],
    ).content))
    assert "已截断" in bounded_authorization["invalid_response"]


def test_settled_evidence_prompts_state_the_total_reference_cap() -> None:
    """Expose the deterministic three-ref list cap to both model attempts."""

    expected_contract = "每个列表合计最多 3 个 ref"

    assert expected_contract in (
        persona_relevance_agent._SETTLED_SYSTEM_PROMPT_COMMON
    )
    assert expected_contract in (
        persona_relevance_agent._SETTLED_AUTHORITATIVE_REPAIR_PROMPT
    )
    assert expected_contract in (
        frontline_relevance_agent._FRONTLINE_SYSTEM_PROMPT_COMMON
    )


def test_text_prompts_organically_favor_spoken_or_typed_dialog() -> None:
    """Keep staging out of upstream guidance without adding a rejection gate."""

    planning_and_rendering = "\n".join((
        CONTENT_PLAN_SYSTEM_PROMPT,
        dialog_agent._V2_DIALOG_GENERATOR_PROMPT,
        dialog_agent._V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT,
    ))
    verifier_prompts = "\n".join((
        dialog_agent._V2_DIALOG_SEMANTIC_FIDELITY_PROMPT,
        dialog_agent._V2_DIALOG_ROLE_DIRECTION_PROMPT,
        dialog_agent._V2_DIALOG_SURFACE_INTEGRITY_PROMPT,
    ))

    assert "实际会说出或发送" in planning_and_rendering
    assert "用词、句式与节奏" in planning_and_rendering
    for retired_guidance in (
        "action description",
        "动作描写",
        "bracketed",
        "方括号",
    ):
        assert retired_guidance not in planning_and_rendering.casefold()
        assert retired_guidance not in verifier_prompts.casefold()
    assert "false_execution" in dialog_agent._V2_DIALOG_SURFACE_INTEGRITY_PROMPT


def test_semantic_appraisal_prompt_limits_model_authority() -> None:
    """Keep state, lifecycle, and persistence outside appraisal authority."""

    for required_text in (
        "有界证据",
        "允许的 handle",
        "propositions",
        "deltas",
    ):
        assert required_text in SEMANTIC_APPRAISAL_PROMPT
    for forbidden_text in (
        "emotion_id",
        "activation_id",
        "replacement_state",
        "persistence route",
    ):
        assert forbidden_text not in SEMANTIC_APPRAISAL_PROMPT
    for required_contract in (
        'proposition 和 delta 各自只能是一个对象或 null',
        '不能使用数组，也不能列举多个候选',
        '不要输出 explanation、selected_evidence_handles、selected_role_handles、propositions',
        'delta 必须是 -40 到 40（含边界）的 JSON 整数',
        'question.candidate_origin_evidence',
        '来源 evidence handle',
    ):
        assert required_contract in SEMANTIC_APPRAISAL_PROMPT


def test_goal_prompt_requires_complete_grounded_bid() -> None:
    """Keep goal cognition handle-grounded and final-wording free."""

    prompt = " ".join(GOAL_COGNITION_PROMPT.split())
    for required_text in (
        "独立的目标认知分支",
        "evidence handle",
        "完整、有证据支持",
        "不写最终对话",
        "每个元素必须逐个等于一个已提供的 handle",
        "不得使用范围、通配符、组合写法或 source ID",
        "`evidence_handles` 最多九项",
        "`target_role_handles` 最多八项",
    ):
        assert required_text in prompt

    repair_prompt = " ".join(GOAL_COGNITION_REPAIR_PROMPT.split())
    assert MAX_GOAL_BID_EVIDENCE_HANDLES == 9
    assert MAX_GOAL_BID_ROLE_HANDLES == 8
    assert "每个元素必须逐个等于一个允许的 handle" in repair_prompt
    assert "不得使用范围、通配符、 组合写法或 source ID" in repair_prompt
    assert "`evidence_handles` 最多九项" in repair_prompt
    assert "`target_role_handles` 最多八项" in repair_prompt


def test_goal_prompt_owns_current_judgment_and_roles() -> None:
    """A bid chooses the present motive without reversing scene roles."""

    prompt = " ".join(GOAL_COGNITION_PROMPT.split())
    compact_prompt = "".join(GOAL_COGNITION_PROMPT.split())

    assert "此刻真实动机" in prompt
    assert "当前事件" in prompt
    assert "关系" in prompt
    assert "结构化用户对话角色具有权威性" in compact_prompt
    assert "行动者" in prompt
    assert "对象" in prompt


def test_goal_prompt_uses_latest_identity_as_current_authority() -> None:
    """Learned identity must shape goals instead of remaining appraisal data."""

    prompt = " ".join(GOAL_COGNITION_PROMPT.split())

    assert "semantic_context.character_identity" in prompt
    assert "最新且权威的角色身份" in prompt
    assert "覆盖初始种子身份" in prompt
    assert "不得用旧习惯" in prompt


def test_goal_prompt_prioritizes_current_episode_over_stale_progress() -> None:
    """Current direct evidence advances semantically matching old events."""

    prompt = " ".join(GOAL_COGNITION_PROMPT.split())

    for required_text in (
        "当前 episode 比进度更新",
        "部件与整体",
        "不要要求逐字相同",
        "优先于旧事件状态",
        "引用相关约束并推进",
    ):
        assert required_text in prompt


def test_goal_prompt_treats_physical_requests_as_verbal_stance() -> None:
    """The character brain does not invent a physical actuator."""

    prompt = GOAL_COGNITION_PROMPT

    assert "言语立场" in prompt
    assert "status 为 executed" in prompt
    assert "证明角色大脑完成" in prompt


def test_collapse_and_action_prompts_preserve_bid_ownership() -> None:
    """Prevent collapse and planning from rewriting admitted motives."""

    assert "本次 prompt 内" in COLLAPSE_PROMPT
    assert "保持候选内容原样" in COLLAPSE_PROMPT
    assert "语义能力请求" in ACTION_PLANNING_PROMPT
    assert "不改写目标候选" in ACTION_PLANNING_PROMPT
    assert "不能扩大能力适用范围" in ACTION_PLANNING_PROMPT
    assert "泛化词" in ACTION_PLANNING_PROMPT
    normalized_action_prompt = "".join(ACTION_PLANNING_PROMPT.split())
    assert "task、action" in normalized_action_prompt


def test_surface_prompts_leave_final_dialogue_to_dialog() -> None:
    """Keep all active surface stages semantic and non-rendering."""

    surface_prompts = (
        CONTENT_PLAN_SYSTEM_PROMPT,
        PREFERENCE_SYSTEM_PROMPT,
        VISUAL_SYSTEM_PROMPT,
    )
    assert all("最终对话" in prompt for prompt in surface_prompts)


def test_generated_semantic_prompts_preserve_language_policy() -> None:
    """Model-authored semantic prose retains the project language contract."""

    for prompt in (
        SEMANTIC_APPRAISAL_PROMPT,
        GOAL_COGNITION_PROMPT,
        ACTION_PLANNING_PROMPT,
        CONTENT_PLAN_SYSTEM_PROMPT,
        PREFERENCE_SYSTEM_PROMPT,
        VISUAL_SYSTEM_PROMPT,
    ):
        normalized = " ".join(prompt.split())
        assert "简体中文" in normalized
        assert "用户" in normalized
        assert "专有名词" in normalized
        assert "schema 或 enum token" in normalized


def test_semantic_prompts_preserve_typed_source_ownership() -> None:
    """Internal evidence cannot be reclassified as current user speech."""

    for prompt in (
        SEMANTIC_APPRAISAL_PROMPT,
        GOAL_COGNITION_PROMPT,
        ACTION_PLANNING_PROMPT,
        CONTENT_PLAN_SYSTEM_PROMPT,
        PREFERENCE_SYSTEM_PROMPT,
        VISUAL_SYSTEM_PROMPT,
    ):
        normalized = " ".join(prompt.split())
        compact = "".join(prompt.split())
        assert "角色自己" in compact
        assert "当前用户" in compact
        assert "即时" in compact
        assert "发言" in compact
        assert "运行元数据" in normalized


def test_v2_prompts_do_not_restore_operational_or_scalar_gates() -> None:
    """Character judgment stays semantic and free of retired gate vocabulary."""

    prompts = "\n".join((
        SEMANTIC_APPRAISAL_PROMPT,
        GOAL_COGNITION_PROMPT,
        COLLAPSE_PROMPT,
        ACTION_PLANNING_PROMPT,
        CONTENT_PLAN_SYSTEM_PROMPT,
        PREFERENCE_SYSTEM_PROMPT,
        VISUAL_SYSTEM_PROMPT,
    )).casefold()
    for forbidden in (
        "relationship score threshold",
        "mood gate",
        "vibe gate",
        "tool cost",
        "willingness score",
        "kazusa",
    ):
        assert forbidden not in prompts
