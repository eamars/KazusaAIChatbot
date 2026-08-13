"""Focused deterministic checks for the Core V2 prompt-owner boundary."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
)
from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    ACTION_PLANNING_PROMPT,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    _ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT,
    CONTINUITY_AUTHORITY_INSTRUCTIONS,
    GENERIC_GOAL_REPAIR_INSTRUCTIONS,
    GOAL_COGNITION_PROMPT,
    NON_ORDINARY_GENERIC_GOAL_REPAIR_INSTRUCTIONS,
    NON_ORDINARY_GOAL_COGNITION_PROMPT,
    ORDINARY_RECURRENCE_SELECTION_GOAL_COGNITION_PROMPT,
    REQUIRED_SELECTION_GOAL_PROMPT,
    SELECTION_GOAL_REPAIR_INSTRUCTIONS,
    _build_goal_repair_feedback,
    _fit_goal_prompt_payload,
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.semantic_appraisal import (
    SEMANTIC_APPRAISAL_PROMPT,
    _appraisal_repair_messages,
    _compact_permitted_delta_path_domains,
    _compact_semantic_contract_error,
    _fit_appraisal_payload,
)
from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    CONTENT_PLAN_SYSTEM_PROMPT,
    DIALOG_COMPLIANCE_REPAIR_SYSTEM_PROMPT,
    PREFERENCE_SYSTEM_PROMPT,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from tests.cognition_core_v2_test_helpers import canonical_identity_context


def test_core_v2_prompts_fit_local_model_guidance_targets() -> None:
    """Keep the stable instruction blocks short enough for weaker models."""

    assert len(SEMANTIC_APPRAISAL_PROMPT) <= 3_000
    assert len(GOAL_COGNITION_PROMPT) <= 3_000
    assert len(NON_ORDINARY_GOAL_COGNITION_PROMPT) <= 3_000
    assert len(REQUIRED_SELECTION_GOAL_PROMPT) <= 2_600


def test_selected_response_operation_contract_is_documented() -> None:
    """Documentation distinguishes input provenance from selected authority."""

    repository_root = Path(__file__).resolve().parents[1]
    cognition_readme = (
        repository_root
        / "src"
        / "kazusa_ai_chatbot"
        / "cognition_core_v2"
        / "README.md"
    ).read_text(encoding="utf-8")
    nodes_readme = (
        repository_root
        / "src"
        / "kazusa_ai_chatbot"
        / "nodes"
        / "README.md"
    ).read_text(encoding="utf-8")

    assert "episode-level `response_operation` is input provenance" in (
        cognition_readme
    )
    assert "`selected_response_operation` after the character chooses" in (
        cognition_readme
    )
    assert (
        "nested role and response ownership are resolved once before goal cognition"
        not in cognition_readme
    )
    assert "selected operation" in nodes_readme
    assert "input-level `response_operation`" in nodes_readme
    assert "does not rewrite it" in nodes_readme
    assert "percept.content.response_operation" not in nodes_readme


def test_goal_prompt_documents_one_objective_evidence_authority() -> None:
    """The cognition ICD documents one objective and each lane's authority."""

    repository_root = Path(__file__).resolve().parents[1]
    cognition_readme = (
        repository_root
        / "src"
        / "kazusa_ai_chatbot"
        / "cognition_core_v2"
        / "README.md"
    ).read_text(encoding="utf-8")

    assert "one primary objective" in cognition_readme
    assert "same concrete matter" in cognition_readme
    assert "private residue" in cognition_readme
    assert "Conditional self-guidance" in cognition_readme
    assert "ordered sub-actions" in cognition_readme


def test_static_prompt_policy_audit_removes_application_owned_policy() -> None:
    """Active prompts and rendered context contain no removed policy defaults."""

    prompt_sources = (
        SEMANTIC_APPRAISAL_PROMPT,
        GOAL_COGNITION_PROMPT,
        NON_ORDINARY_GOAL_COGNITION_PROMPT,
        REQUIRED_SELECTION_GOAL_PROMPT,
        " ".join(GENERIC_GOAL_REPAIR_INSTRUCTIONS),
        " ".join(SELECTION_GOAL_REPAIR_INSTRUCTIONS),
        ACTION_PLANNING_PROMPT,
        CONTENT_PLAN_SYSTEM_PROMPT,
        PREFERENCE_SYSTEM_PROMPT,
        DIALOG_COMPLIANCE_REPAIR_SYSTEM_PROMPT,
        dialog_module._V2_DIALOG_GENERATOR_PROMPT,
        dialog_module._V2_DIALOG_HARD_FAILURE_REPAIR_PROMPT,
        dialog_module._V2_DIALOG_SEMANTIC_FIDELITY_PROMPT,
        dialog_module._V2_DIALOG_ROLE_DIRECTION_PROMPT,
        dialog_module._V2_DIALOG_SURFACE_INTEGRITY_PROMPT,
    )
    removed_policy_fragments = (
        "当前 episode 明确写出的角色拒绝、排斥或边界条件优先于",
        "当前 episode 的角色自我边界、明确拒绝、威胁或强迫条件优先于",
        "不把 compliance 当作意愿或同意",
        "不代表意愿或同意",
        "unestablished` 只能配 `reject`",
        "developing_or_uncertain` 不能 accept",
        "established` 才可按边界选择",
        "安全、内容审查、亲密程度或通用礼貌边界",
        "隐私、保密、同意、安全、内容审查或可见披露限制",
    )
    for prompt in prompt_sources:
        for fragment in removed_policy_fragments:
            assert fragment not in prompt

    projection = project_state_for_prompt(
        {
            "state_scope": "user",
            "updated_at": "2026-07-14T00:00:00Z",
            "owner_user_id": "user-prompt-audit",
            "goals": [],
            "threats": [],
            "active_events": [],
            "knowledge_gaps": [],
            "affect_activations": [],
            "drives": {},
        },
        character_constraints={
            "drives": {},
            "standards": [{
                "standard_id": "standard-1",
                "description": "保持诚实",
                "importance": 0.8,
            }],
            "meaning_state": {
                "purpose_coherence": 50,
                "agency": 50,
                "identity_continuity": 50,
                "salience": 50,
            },
            "personality_judgment": {
                "logic": "analytical",
                "defense": "reserved",
                "quirks": "precise",
                "taboos": "stay in character",
            },
        },
        character_identity_context=canonical_identity_context(),
    )
    rendered_context = json.dumps(
        projection.payload,
        ensure_ascii=False,
        sort_keys=True,
    )
    assert projection.payload["character_constraints"]["standards"] == []
    assert all(
        not (handle.startswith("s") and handle[1:].isdigit())
        for handle in projection.handle_to_ref
    )
    assert "保持诚实" not in rendered_context
    for fragment in removed_policy_fragments:
        assert fragment not in rendered_context


def test_core_v2_prompts_keep_one_authoritative_handle_domain() -> None:
    """Expose the existing evidence, role, origin, and path owners clearly."""

    semantic_prompt = SEMANTIC_APPRAISAL_PROMPT
    for required_text in (
        "question.handle_field_domains",
        "question.candidate_origin_evidence",
        "question.permitted_delta_path_domains",
        "来源 evidence handle",
        "state_field.handle.axis",
        "delta_limit",
        "role_assignments 是必填字段，证据不支持任何角色时写 []",
        "不要输出 explanation、selected_evidence_handles、selected_role_handles、propositions",
    ):
        assert required_text in semantic_prompt

    goal_prompt = GOAL_COGNITION_PROMPT
    for required_text in (
        "独立的目标认知分支",
        "当前 episode 是当前场景事实，进度和旧关系是补充语境",
        "不要把任何单一来源自动升级为最终立场",
        "对三个真实关系状态，`reject`、`deflect`、`negotiate`、`conditional_accept` 和 `accept` 都是可选的角色立场",
        "每个元素必须逐个等于一个已提供的 handle",
        "不得使用范围、通配符、组合写法或 source ID",
        "不写最终对话",
    ):
        assert required_text in goal_prompt
    assert "semantic_context.branch.goal_kind" not in goal_prompt


def test_nonordinary_generic_goal_prompt_excludes_relational_contract() -> None:
    """Keep active generic branches on the exact nine-field contract."""

    prompt = NON_ORDINARY_GOAL_COGNITION_PROMPT
    required_fields = (
        "intention",
        "desired_outcome",
        "concrete_detail",
        "reason",
        "private_monologue",
        "target_role_handles",
        "evidence_handles",
        "expected_consequences",
        "confidence",
    )

    for field_name in required_fields:
        assert field_name in prompt
    for required_text in (
        "response_operation",
        "role_summaries",
        "`pN`",
        "conversation_evidence",
        "status=executed",
        "branch.branch_intent_guidance",
        "ordinary_response",
    ):
        assert required_text in prompt
    for forbidden_text in (
        "relational_willingness",
        "relationship_sensitive",
        "current_user_relationship_state",
        "unestablished",
    ):
        assert forbidden_text not in prompt

    repair_instructions = " ".join(
        NON_ORDINARY_GENERIC_GOAL_REPAIR_INSTRUCTIONS
    )
    assert "invalid_draft" in repair_instructions
    assert "relational_willingness" not in repair_instructions


def test_active_selection_prompt_keeps_nonordinary_output_contract() -> None:
    """Keep active required-selection branches on their own exact schema."""

    prompt = _ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT
    for required_text in (
        "required_selection_operations",
        "selected_response_operation",
        "conversation_progress_evidence",
        "supporting_evidence",
        "semantic_context.character_identity",
        "最新且权威的角色身份",
        "当前 episode 比进度更新",
        "言语立场",
        "status=executed",
        "证明相应能力已完成",
        "expected_consequences",
    ):
        assert required_text in prompt
    assert "relational_willingness" not in prompt

    role_contract_prompts = (
        REQUIRED_SELECTION_GOAL_PROMPT,
        _ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT,
        ORDINARY_RECURRENCE_SELECTION_GOAL_COGNITION_PROMPT,
        " ".join(SELECTION_GOAL_REPAIR_INSTRUCTIONS),
    )
    for role_prompt in role_contract_prompts:
        for required_text in (
            "response_owner_role",
            "selection_owner_role",
            "embedded_actor_role",
            "embedded_target_role",
            "selection_required",
            "current_user",
            "self",
            "pN",
            "operation",
        ):
            assert required_text in role_prompt
    for role_prompt in (
        _ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT,
        ORDINARY_RECURRENCE_SELECTION_GOAL_COGNITION_PROMPT,
        " ".join(SELECTION_GOAL_REPAIR_INSTRUCTIONS),
    ):
        for role_value in ("当前角色", "当前用户", "其他参与者", "无"):
            assert role_value in role_prompt


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
                "branch_intent_guidance": '保持当前分支的语义关注。',
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
    assert list(goal_payload["branch"]) == [
        "goal_kind",
        "action_tendencies",
        "branch_intent_guidance",
    ]
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
                "delta_limit": 40,
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
    assert repair_payload["allowed_values"][
        "permitted_delta_path_domains"
    ][0]["delta_limit"] == 40
    assert "唯一失败规则" in repair_payload["repair_instruction"]
    assert "role_assignments 是必填字段，证据不支持任何角色时写 []" in (
        repair_payload["repair_instruction"]
    )
    assert "证据标签（如 beneficiary）不能写入 role" in (
        repair_payload["repair_instruction"]
    )


def test_permitted_delta_path_domains_expose_path_delta_limits() -> None:
    """Projected domains carry the reducer's per-field delta bound."""

    domains = _compact_permitted_delta_path_domains([
        "relationship.r1.attachment",
        "relationship.r1.perceived_closeness",
        "meaning_state.m1.purpose_coherence",
        "goals.g1.importance",
        "active_events.ce1.harm",
        "threats.ct1.likelihood",
        "knowledge_gaps.ck1.uncertainty",
        "drives.d1.pressure",
        "events.ev1.salience",
    ])
    limits_by_field = {
        domain["state_field"]: domain["delta_limit"]
        for domain in domains
    }
    assert limits_by_field["relationship"] == 10
    assert limits_by_field["meaning_state"] == 10
    for state_field in (
        "goals",
        "active_events",
        "threats",
        "knowledge_gaps",
        "drives",
        "events",
    ):
        assert limits_by_field[state_field] == 40


def test_semantic_repair_compacts_only_the_owned_path_suffix() -> None:
    """Keep the failed rule/path while projecting the path domain separately."""

    full_error = (
        "semantic delta path 'knowledge_gaps.k7.uncertainty' is not owned "
        "by question; permitted paths: [\"goals.g1.importance\"]"
    )

    assert _compact_semantic_contract_error(full_error) == (
        "semantic delta path 'knowledge_gaps.k7.uncertainty' is not owned "
        "by question"
    )
    assert _compact_semantic_contract_error(
        "semantic delta fields are not exact; permitted paths: [\"x\"]"
    ) == "semantic delta fields are not exact; permitted paths: [\"x\"]"
    assert _compact_semantic_contract_error(
        "semantic delta path 'x' is not owned by question"
    ) == "semantic delta path 'x' is not owned by question"


def test_goal_repair_feedback_preserves_cross_namespace_authority() -> None:
    """Repair payloads keep required evidence separate from role handles."""

    feedback = _build_goal_repair_feedback(
        validation_error="invalid_draft",
        parsed={"evidence_handles": ["r1"]},
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
    assert "schema_version" not in feedback[
        "relational_willingness_contract"
    ]
    assert "schema_version" not in feedback[
        "relational_willingness_contract"
    ]["required_fields"]
    assert "required_evidence_handles" in " ".join(
        SELECTION_GOAL_REPAIR_INSTRUCTIONS
    )
    assert "三个真实状态都可配合五种敏感立场" in " ".join(
        GENERIC_GOAL_REPAIR_INSTRUCTIONS
    )


def test_nonordinary_exact_field_repair_uses_key_facts_without_draft() -> None:
    """Exact-field repair exposes parsed keys without echoing candidate text."""

    parsed = {
        "intention": "continue the active goal",
        "desired_outcome": "preserve the active goal",
        "concrete_detail": "use the current evidence",
        "reason": "current evidence supports the goal",
        "private_monologue": "I should preserve the goal.",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the goal remains grounded"],
        "confidence": "high",
        "relational_willingness": {"unexpected": "candidate value"},
    }
    feedback = _build_goal_repair_feedback(
        validation_error="goal bid draft fields are not exact",
        parsed=parsed,
        response_text=json.dumps(parsed),
        evidence_handles={"e1"},
        episode_evidence_handles={"e1"},
        role_bindings={},
        required_evidence_handles=set(),
        selection_required=False,
        require_relational_willingness=False,
        maximum_evidence_handles=9,
    )

    assert feedback["observed_top_level_fields"] == sorted(parsed)
    assert feedback["missing_top_level_fields"] == []
    assert feedback["unexpected_top_level_fields"] == [
        "relational_willingness"
    ]
    assert "invalid_draft" not in feedback
    assert "candidate value" not in json.dumps(feedback)
    assert feedback["repair_instruction"] == [
        instruction
        for instruction in NON_ORDINARY_GENERIC_GOAL_REPAIR_INSTRUCTIONS
        if "invalid_draft" not in instruction
    ]
    repair_instructions = " ".join(feedback["repair_instruction"])
    assert "invalid_draft" not in repair_instructions
    assert "relational_willingness" not in repair_instructions


@pytest.mark.asyncio
async def test_active_selection_route_uses_rewritten_prompt_and_repair() -> None:
    """Exercise the runtime-selected active-selection prompt boundary."""

    semantic_text = json.dumps({
        "role_explicit_content": "The character must choose the next step.",
        "response_operation": {
            "operation": "The character chooses the next step.",
            "response_owner_role": CURRENT_CHARACTER_ROLE,
            "selection_owner_role": CURRENT_CHARACTER_ROLE,
            "selection_required": True,
            "embedded_actor_role": CURRENT_USER_ROLE,
            "embedded_target_role": CURRENT_CHARACTER_ROLE,
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
        "authority": "current_event",
    }]
    valid = {
        "selection": "The character chooses the grounded next step.",
        "selected_response_operation": {
            "operation": "the user gives the selected next step to the character",
        },
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
        message_list[0].content
        == (
            _ACTIVE_REQUIRED_SELECTION_GOAL_PROMPT
            + CONTINUITY_AUTHORITY_INSTRUCTIONS
        )
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
