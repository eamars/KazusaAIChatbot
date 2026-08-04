"""Focused deterministic checks for the Core V2 prompt-owner boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage, SystemMessage
import pytest

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    GENERIC_GOAL_REPAIR_INSTRUCTIONS,
    GOAL_COGNITION_PROMPT,
    REQUIRED_SELECTION_GOAL_PROMPT,
    SELECTION_GOAL_REPAIR_INSTRUCTIONS,
    _ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT,
    _build_goal_repair_feedback,
    _fit_goal_prompt_payload,
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    SEMANTIC_APPRAISAL_PROMPT,
    _appraisal_repair_messages,
    _fit_appraisal_payload,
)


def test_core_v2_prompts_fit_local_model_guidance_targets() -> None:
    """Keep the stable instruction blocks short enough for weaker models."""

    assert len(SEMANTIC_APPRAISAL_PROMPT) <= 3_000
    assert len(GOAL_COGNITION_PROMPT) <= 3_000
    assert len(REQUIRED_SELECTION_GOAL_PROMPT) <= 2_600


def test_core_v2_prompts_keep_one_authoritative_handle_domain() -> None:
    """Expose the existing evidence, role, origin, and path owners clearly."""

    semantic_prompt = SEMANTIC_APPRAISAL_PROMPT
    for required_text in (
        "question.handle_field_domains",
        "question.candidate_origin_evidence",
        "question.permitted_delta_path_domains",
        "来源 evidence handle",
        "state_field.handle.axis",
        "不要输出 explanation、selected_evidence_handles、selected_role_handles、propositions",
    ):
        assert required_text in semantic_prompt

    goal_prompt = GOAL_COGNITION_PROMPT
    for required_text in (
        "独立的目标认知分支",
        "当前 episode 比进度更新",
        "角色拒绝、排斥或边界条件优先于旧关系",
        "当前 episode 的 semantic_text 明确写出当前角色排斥、拒绝或不愿意",
        "每个元素必须逐个等于一个已提供的 handle",
        "不得使用范围、通配符、组合写法或 source ID",
        "不写最终对话",
    ):
        assert required_text in goal_prompt
    assert "semantic_context.branch.goal_kind" not in goal_prompt


def test_active_selection_prompt_keeps_nonordinary_output_contract() -> None:
    """Keep active required-selection branches on their own exact schema."""

    prompt = _ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT
    for required_text in (
        "required_selection_operations",
        "conversation_progress_evidence",
        "supporting_evidence",
        "semantic_context.character_identity",
        "最新且权威的角色身份",
        "当前 episode 比进度更新",
        "言语立场",
        "status=executed",
        "证明相应能力已完成",
        "selection_kind",
        "expected_consequences",
    ):
        assert required_text in prompt
    assert "relational_willingness" not in prompt


def test_prompt_payloads_preserve_contract_order() -> None:
    """Keep production-shaped prompt declarations in stable source order."""

    semantic_text, _ = _fit_appraisal_payload(
        {
            "question": {
                "question_id": "q:event_agency",
                "question_kind": "event_agency",
                "semantic_question": "Identify the current event agency.",
                "permitted_role_handles": ["ce1"],
                "candidate_origin_evidence": {"ce1": "e1"},
                "permitted_delta_path_domains": [{
                    "state_field": "events",
                    "handles": ["ce1"],
                    "axes": ["salience"],
                }],
                "permitted_proposition_kinds": ["event"],
                "proposition_kind_semantics": {
                    "event": "one event proposition",
                },
                "handle_field_domains": {
                    "subject_handle": ["ce1"],
                    "object_handle": ["ce1"],
                    "entity_handle": ["ce1"],
                    "evidence_handles": ["e1"],
                },
                "role_handle_semantics": {
                    "self": {"structured_handle": "self"},
                    "current_user": {
                        "structured_handle": "current_user"
                    },
                },
                "micro_appraisal": {
                    "item_index": 1,
                    "maximum_items": 4,
                },
            },
            "evidence": [],
            "state": {},
        },
        system_prompt_chars=0,
    )
    semantic_question = json.loads(semantic_text)["question"]
    assert list(semantic_question) == [
        "question_id",
        "question_kind",
        "semantic_question",
        "permitted_role_handles",
        "candidate_origin_evidence",
        "permitted_delta_path_domains",
        "permitted_proposition_kinds",
        "proposition_kind_semantics",
        "handle_field_domains",
        "role_handle_semantics",
        "micro_appraisal",
    ]
    assert list(semantic_question["handle_field_domains"]) == [
        "subject_handle",
        "object_handle",
        "entity_handle",
        "evidence_handles",
    ]

    goal_text = _fit_goal_prompt_payload(
        {
            "branch": {
                "goal_kind": "active_branch",
                "action_tendencies": ["choose"],
            },
            "goal": {"goal_kind": "active_branch", "lifecycle": "active"},
            "semantic_context": {"current_event": "Choose the next step."},
            "role_handles": ["r1"],
            "role_summaries": {"r1": "The current relationship."},
            "required_selection_operations": [{
                "operation": "Choose the next step.",
                "evidence_handle": "e1",
            }],
            "conversation_progress_evidence": [{
                "handle": "e2",
                "source_kind": "conversation_progress",
                "semantic_text": "The prior step is complete.",
            }],
            "supporting_evidence": [{
                "handle": "e3",
                "source_kind": "episode",
                "semantic_text": "The user asks for a choice.",
            }],
        },
        system_prompt="prompt",
    )
    assert list(json.loads(goal_text)) == [
        "branch",
        "goal",
        "semantic_context",
        "role_handles",
        "role_summaries",
        "required_selection_operations",
        "conversation_progress_evidence",
        "supporting_evidence",
    ]
    goal_payload = json.loads(goal_text)
    assert goal_payload["required_selection_operations"][0][
        "evidence_handle"
    ] == "e1"
    assert goal_payload["conversation_progress_evidence"][0]["handle"] == (
        "e2"
    )
    assert goal_payload["supporting_evidence"][0]["handle"] == "e3"


def test_semantic_repair_projects_existing_contract_values() -> None:
    """Repair feedback exposes canonical domains beside the failed rule."""

    human_payload = {
        "question": {
            "handle_field_domains": {
                "subject_handle": ["ce1"],
                "evidence_handles": ["e1"],
            },
            "candidate_origin_evidence": {"ce1": "e1"},
            "permitted_delta_path_domains": [{
                "state_field": "events",
                "handles": ["ce1"],
                "axes": ["salience"],
            }],
        },
        "evidence": [],
        "state": {},
    }
    messages = _appraisal_repair_messages(
        system_message=SystemMessage(content="semantic"),
        human_message=HumanMessage(
            content=json.dumps(human_payload, ensure_ascii=False)
        ),
        invalid_candidate='{"delta": {}}',
        contract_error="semantic delta path is invalid",
        allowed_values={
            "handle_field_domains": human_payload["question"][
                "handle_field_domains"
            ],
            "candidate_origin_evidence": human_payload["question"][
                "candidate_origin_evidence"
            ],
            "permitted_delta_path_domains": human_payload["question"][
                "permitted_delta_path_domains"
            ],
        },
    )

    repair_payload = json.loads(str(messages[-1].content))
    assert list(repair_payload) == [
        "repair_instruction",
        "contract_error",
        "allowed_values",
    ]
    assert repair_payload["contract_error"] == (
        "semantic delta path is invalid"
    )
    assert repair_payload["allowed_values"]["candidate_origin_evidence"] == {
        "ce1": "e1"
    }
    assert repair_payload["allowed_values"][
        "permitted_delta_path_domains"
    ][0]["axes"] == ["salience"]
    assert "唯一失败规则" in repair_payload["repair_instruction"]


def test_goal_repair_feedback_preserves_cross_namespace_authority() -> None:
    """Repair payloads keep required evidence separate from role handles."""

    feedback = _build_goal_repair_feedback(
        validation_error="invalid_draft",
        response_text='{"evidence_handles": ["r1"]}',
        evidence_handles={"e1", "e2"},
        episode_evidence_handles={"e1"},
        role_bindings={"r1": {"semantic_text": "当前角色"}},
        required_evidence_handles={"e1"},
        selection_required=True,
        require_relational_willingness=True,
        maximum_evidence_handles=4,
    )

    assert feedback["allowed_evidence_handles"] == ["e1", "e2"]
    assert feedback["required_evidence_handles"] == ["e1"]
    assert feedback["current_episode_evidence_handles"] == ["e1"]
    assert feedback["allowed_role_handles"] == ["r1"]
    assert feedback["role_handles_forbidden_in_evidence_handles"] == ["r1"]
    assert feedback["invalid_draft"] == '{"evidence_handles": ["r1"]}'
    assert feedback["relational_willingness_contract"]["schema_version"] == (
        "relational_willingness.v2"
    )
    assert "required_evidence_handles" in " ".join(
        SELECTION_GOAL_REPAIR_INSTRUCTIONS
    )
    assert "current_user_relationship_state" in " ".join(
        GENERIC_GOAL_REPAIR_INSTRUCTIONS
    )


@pytest.mark.asyncio
async def test_active_selection_route_uses_rewritten_prompt_and_repair() -> None:
    """Exercise the runtime-selected active-selection prompt boundary."""

    semantic_text = json.dumps({
        "role_explicit_content": "The character must choose the next step.",
        "response_operation": {
            "operation": "The character chooses the next step.",
            "response_owner_role": "current character",
            "selection_owner_role": "current character",
            "selection_required": True,
            "embedded_actor_role": "current user",
            "embedded_target_role": "current character",
        },
    })
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode-1",
            "occurred_at": "2026-07-15T00:00:00Z",
            "semantic_summary": semantic_text,
        },
        "semantic_text": semantic_text,
        "visible_to": ["q:event_agency"],
    }]
    valid = {
        "selection_kind": "choice",
        "selection": "The character chooses the grounded next step.",
        "reason": "The current operation requires a concrete choice.",
        "private_monologue": "I should choose from the current evidence.",
        "target_role_handles": ["r1"],
        "evidence_handles": ["e1"],
        "expected_consequences": ["The active goal receives a clear choice."],
        "confidence": "high",
    }

    class _LLM:
        def __init__(self) -> None:
            self.messages: list[list[object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.messages.append(messages)
            candidate = dict(valid)
            if len(self.messages) == 1:
                candidate["evidence_handles"] = ["r1"]
            return SimpleNamespace(content=json.dumps(candidate))

    llm = _LLM()
    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["autonomy_boundary"],
        {"scope": "user", "kind": "goal", "entity_id": "goal:route"},
        {
            "_role_bindings": {
                "r1": {
                    "role": "target",
                    "entity_kind": "relationship",
                    "entity_id": "relationship:u1",
                },
            },
            "role_summaries": {"r1": "The current relationship."},
        },
        evidence,
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=object(),
        ),
    )

    assert len(llm.messages) == 2
    assert all(
        message_list[0].content == _ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT
        for message_list in llm.messages
    )
    repair_payload = json.loads(str(llm.messages[1][1].content))
    assert list(repair_payload) == [
        "branch",
        "goal",
        "semantic_context",
        "role_handles",
        "role_summaries",
        "required_selection_operations",
        "conversation_progress_evidence",
        "supporting_evidence",
        "repair_feedback",
    ]
    assert repair_payload["repair_feedback"][
        "role_handles_forbidden_in_evidence_handles"
    ] == ["r1"]
    assert "relational_willingness" not in repair_payload["repair_feedback"]
    assert bid["branch_id"] == "autonomy_boundary"
