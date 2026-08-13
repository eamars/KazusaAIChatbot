"""Deterministic checks for branch-owned generic intent guidance."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    BRANCH_REGISTRY_ORDER,
    DEFAULT_BRANCH_DEFINITIONS,
    DEFAULT_BRANCH_INTENT_GUIDANCE,
    select_final_branches,
    select_preliminary_branches,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    MAX_BRANCH_INTENT_GUIDANCE_CHARS,
    BranchDefinition,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    GOAL_COGNITION_PROMPT_CAP,
    MIN_PROMPT_EVIDENCE_TEXT_CHARS,
    NON_ORDINARY_GOAL_COGNITION_PROMPT,
    _fit_goal_prompt_payload,
    run_goal_cognition,
)

_EXPECTED_GUIDANCE = {
    "ordinary_response": (
        '为当前事件提供中性的上下文基线；在适用时保留现有 '
        'relational_willingness 的归属，不引入其他分支的专门焦点。'
    ),
    "relationship_connection": (
        '评估是否以及如何通过自愿且符合当前情境的互惠参与来建立、维持、'
        '调整或修复人际连接。'
    ),
    "bond_protection": (
        '评估当前事件是否对重要关系纽带造成有证据支持的威胁或损害，并考虑'
        '相称的保护或修复。'
    ),
    "trust_verification": (
        '评估当前证据是否支持信任、保留信任或需要核实；不把不确定性直接解释'
        '为背叛。'
    ),
    "autonomy_boundary": (
        '评估当前事件是否对角色自身的自主权、意愿或明确边界造成有证据支持的'
        '压力或代价；在有依据时保护边界，不假定恶意。'
    ),
    "safety_coping": (
        '评估当前事件是否存在有证据支持的威胁或压力，并考虑相称的保护或应对；'
        '不凭空升级恐惧。'
    ),
    "obstruction_strategy": (
        '评估当前事件是否阻碍当前目标的进展，并考虑相称的解决、对抗或修复。'
    ),
    "loss_recovery": (
        '评估当前事件是否构成有依据的损失，并考虑恢复、适应或适当的哀悼；'
        '不强迫悲伤。'
    ),
    "moral_repair": (
        '评估当前角色是否对伤害负有有证据支持的责任；如有，考虑相称的修复或'
        '道歉。'
    ),
    "social_care": (
        '评估受当前事件影响的人是否有有依据的需要，并考虑相称的支持或照护；'
        '不强迫温柔。'
    ),
    "reciprocal_response": (
        '确定当前角色对另一方行为的有证据支持且相称的回应；互惠不等于服从，'
        '也不要求匹配情绪价性。'
    ),
    "epistemic_exploration": (
        '通过探索、提问或比较，减少当前有依据的不确定性并增进理解；区分求知'
        '与无依据的断言。'
    ),
    "meaning_reconstruction": (
        '在当前事件造成有依据的叙事或存在性中断后，评估如何重建连贯意义；'
        '不强迫乐观。'
    ),
    "self_improvement": (
        '评估当前角色是否有有证据支持的学习、纠错或能力发展机会；不预设缺陷、'
        '乐观或成功。'
    ),
}
_NONORDINARY_BRANCH_IDS = tuple(
    branch_id
    for branch_id in BRANCH_REGISTRY_ORDER
    if branch_id != "ordinary_response"
)
_GENERIC_BID_FIELDS = {
    "branch_id",
    "goal_ref",
    "intention",
    "desired_outcome",
    "concrete_detail",
    "reason",
    "private_monologue",
    "target_roles",
    "evidence_handles",
    "expected_consequences",
    "confidence",
}


class _CapturingLLM:
    """Return deterministic candidates while preserving every prompt payload."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.messages: list[list[object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        """Capture one model boundary and return the next fixture response."""

        del args, config, kwargs
        self.messages.append(messages)
        response_index = min(
            len(self.messages) - 1,
            len(self.responses) - 1,
        )
        return SimpleNamespace(
            content=json.dumps(
                self.responses[response_index],
                ensure_ascii=False,
            )
        )


def _evidence(
    *,
    semantic_text: str = "The current episode provides one grounded fact.",
) -> list[dict[str, Any]]:
    """Build one current-episode evidence row for direct goal tests."""

    return [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:branch-intent",
            "occurred_at": "2026-08-09T00:00:00Z",
            "semantic_summary": semantic_text,
        },
        "semantic_text": semantic_text,
        "visible_to": [],
        "authority": "current_event",
    }]


def _generic_bid() -> dict[str, Any]:
    """Build one complete nonordinary bid fixture."""

    return {
        "intention": "keep the specialized responsibility grounded",
        "desired_outcome": "avoid unsupported specialized progress",
        "concrete_detail": "use only the current evidence",
        "reason": "the current evidence does not support a specialized advance",
        "private_monologue": "I should stay with what the evidence supports.",
        "target_role_handles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": [
            "the existing collapse can suppress unsupported progress",
        ],
        "confidence": "low",
    }


def _ordinary_bid() -> dict[str, Any]:
    """Build one complete ordinary-response bid fixture."""

    bid = _generic_bid()
    bid["relational_willingness"] = {
        "schema_version": "relational_willingness.v2",
        "applicability": "not_relationship_sensitive",
        "stance": "not_applicable",
        "current_user_relationship_state": "not_applicable",
        "reason": '当前事件只需要处理中性的当前回应。',
        "evidence_handles": ["e1"],
    }
    return bid


def _selection_bid() -> dict[str, Any]:
    """Build one complete typed-selection bid fixture."""

    return {
        "selection": '当前角色选择先确认事实',
        "selected_response_operation": {
            "operation": '当前用户执行当前角色选定的动作',
            "response_owner_role": '当前角色',
            "selection_owner_role": '当前角色',
            "selection_required": True,
            "embedded_actor_role": '当前用户',
            "embedded_target_role": '当前角色',
        },
        "reason": '当前选择操作要求角色直接决定下一步',
        "private_monologue": '我应该先确认当前事实。',
        "target_role_handles": ["r1"],
        "evidence_handles": ["e1"],
        "expected_consequences": ['当前角色保留对下一步的直接选择权'],
        "confidence": "high",
    }


def _services(llm: _CapturingLLM) -> SimpleNamespace:
    """Build the smallest service object accepted by direct goal cognition."""

    config = SimpleNamespace(route_name="test_goal_branch", model="test")
    return SimpleNamespace(
        llm=llm,
        goal_active_branch_config=config,
        goal_ordinary_response_config=config,
    )


async def _run_generic_branch(
    branch_id: str,
) -> tuple[dict[str, Any], _CapturingLLM]:
    """Run one nonordinary branch through initial and repair projection."""

    llm = _CapturingLLM([
        {"unexpected": "invalid candidate"},
        _generic_bid(),
    ])
    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS[branch_id],
        {
            "scope": "user",
            "kind": "goal",
            "entity_id": f"goal:{branch_id}",
        },
        {"_role_bindings": {}, "role_summaries": {}},
        _evidence(),
        _services(llm),
    )
    return bid, llm


def _payload(message_list: list[object]) -> dict[str, Any]:
    """Decode one captured human prompt payload."""

    return json.loads(str(getattr(message_list[1], "content", "{}")))


def test_default_branch_intent_guidance_matches_exact_registry_contract() -> None:
    """Keep the complete default map exact and bounded."""

    assert DEFAULT_BRANCH_INTENT_GUIDANCE == _EXPECTED_GUIDANCE
    assert tuple(DEFAULT_BRANCH_INTENT_GUIDANCE) == BRANCH_REGISTRY_ORDER
    assert {
        branch_id: definition.branch_intent_guidance
        for branch_id, definition in DEFAULT_BRANCH_DEFINITIONS.items()
    } == _EXPECTED_GUIDANCE
    assert all(
        0 < len(guidance) <= MAX_BRANCH_INTENT_GUIDANCE_CHARS
        for guidance in DEFAULT_BRANCH_INTENT_GUIDANCE.values()
    )


@pytest.mark.parametrize(
    ("invalid_guidance", "error_type"),
    [
        (object(), TypeError),
        (" \t", ValueError),
        ("x" * (MAX_BRANCH_INTENT_GUIDANCE_CHARS + 1), ValueError),
    ],
)
def test_branch_intent_guidance_rejects_invalid_values(
    invalid_guidance: object,
    error_type: type[Exception],
) -> None:
    """Reject malformed values before a branch can reach execution."""

    with pytest.raises(error_type):
        BranchDefinition(
            "custom",
            (),
            (),
            branch_intent_guidance=invalid_guidance,  # type: ignore[arg-type]
        )


def test_custom_branch_definition_keeps_empty_neutral_default() -> None:
    """Allow an omitted custom descriptor to retain generic neutral behavior."""

    definition = BranchDefinition("custom", (), ())

    assert definition.branch_intent_guidance == ""


def test_selected_cardinalities_follow_branch_id_under_reversed_completion() -> None:
    """Keep one-to-fourteen branch mapping independent of task order."""

    selected_sets = (
        ("ordinary_response",),
        ("ordinary_response", "relationship_connection"),
        (
            "ordinary_response",
            "relationship_connection",
            "bond_protection",
        ),
        (
            "ordinary_response",
            "relationship_connection",
            "bond_protection",
            "trust_verification",
        ),
        BRANCH_REGISTRY_ORDER[:12],
        BRANCH_REGISTRY_ORDER,
    )

    assert selected_sets[4][-1] == "epistemic_exploration"
    for selected_ids in selected_sets:
        selected = [
            DEFAULT_BRANCH_DEFINITIONS[branch_id]
            for branch_id in selected_ids
        ]
        completed_in_reverse = list(reversed(selected))
        guidance_by_branch = {
            definition.branch_id: definition.branch_intent_guidance
            for definition in completed_in_reverse
        }
        assert guidance_by_branch == {
            branch_id: _EXPECTED_GUIDANCE[branch_id]
            for branch_id in selected_ids
        }


def test_dependency_replacements_preserve_branch_guidance() -> None:
    """Preserve guidance through preliminary and dependency-selected copies."""

    goal = {
        "goal_kind": "autonomy_boundary",
        "status": "pursuing",
    }
    preliminary = select_preliminary_branches([goal])
    preliminary_autonomy = next(
        definition
        for definition in preliminary
        if definition.branch_id == "autonomy_boundary"
    )
    preliminary_copy = replace(
        preliminary_autonomy,
        dependencies=(),
        dependency_options=(),
    )
    assert preliminary_copy.branch_intent_guidance == (
        DEFAULT_BRANCH_DEFINITIONS[
            "autonomy_boundary"
        ].branch_intent_guidance
    )

    final = select_final_branches(
        [DEFAULT_BRANCH_DEFINITIONS["ordinary_response"]],
        [goal],
        question_ids=["q:moral_identity"],
    )
    final_autonomy = next(
        definition
        for definition in final
        if definition.branch_id == "autonomy_boundary"
    )
    assert final_autonomy.dependencies == ("q:moral_identity",)
    assert final_autonomy.branch_intent_guidance == (
        DEFAULT_BRANCH_DEFINITIONS[
            "autonomy_boundary"
        ].branch_intent_guidance
    )


def test_all_guidance_rows_fit_production_prompt_budget_and_evidence_floor() -> None:
    """Keep every projected guidance row inside the existing prompt envelope."""

    evidence_rows = [{
        "evidence_handle": f"e{i}",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": f"episode:budget:{i}",
            "occurred_at": "2026-08-09T00:00:00Z",
            "semantic_summary": "E" * 120,
        },
        "semantic_text": "E" * 12000,
        "visible_to": [],
        "authority": "current_event",
    } for i in range(3)]
    semantic_context = {
        "character_identity": {
            "current_character": "current character",
            "identity_basis": "I" * 1400,
        },
        "events": ["event context " + ("V" * 1200)] * 4,
        "past_dialog_cognition_context": "P" * 1800,
    }

    for branch_id in _NONORDINARY_BRANCH_IDS:
        definition = DEFAULT_BRANCH_DEFINITIONS[branch_id]
        payload = {
            "branch": {
                "goal_kind": definition.goal_kind,
                "action_tendencies": list(definition.action_tendencies),
                "branch_intent_guidance": (
                    definition.branch_intent_guidance
                ),
            },
            "goal": {
                "scope": "user",
                "kind": "goal",
                "entity_id": f"goal:{branch_id}",
            },
            "semantic_context": semantic_context,
            "evidence": evidence_rows,
            "role_handles": [],
            "role_summaries": {},
        }

        fitted = _fit_goal_prompt_payload(
            json.loads(json.dumps(payload, ensure_ascii=False)),
            system_prompt=NON_ORDINARY_GOAL_COGNITION_PROMPT,
        )
        fitted_payload = json.loads(fitted)

        assert (
            len(NON_ORDINARY_GOAL_COGNITION_PROMPT) + len(fitted)
            <= GOAL_COGNITION_PROMPT_CAP
        )
        assert fitted_payload["branch"]["branch_intent_guidance"] == (
            _EXPECTED_GUIDANCE[branch_id]
        )
        assert all(
            len(row["semantic_text"]) >= MIN_PROMPT_EVIDENCE_TEXT_CHARS
            for row in fitted_payload["evidence"]
        )


@pytest.mark.asyncio
async def test_nonordinary_initial_and_repair_payloads_project_guidance() -> None:
    """Project each nonordinary literal without changing the bid schema."""

    for branch_id in _NONORDINARY_BRANCH_IDS:
        bid, llm = await _run_generic_branch(branch_id)

        assert set(bid) == _GENERIC_BID_FIELDS
        assert len(llm.messages) == 2
        for message_list in llm.messages:
            payload = _payload(message_list)
            assert list(payload["branch"]) == [
                "goal_kind",
                "action_tendencies",
                "branch_intent_guidance",
            ]
            assert payload["branch"]["goal_kind"] == (
                DEFAULT_BRANCH_DEFINITIONS[branch_id].goal_kind
            )
            assert payload["branch"]["branch_intent_guidance"] == (
                _EXPECTED_GUIDANCE[branch_id]
            )
            assert len(str(getattr(message_list[1], "content", ""))) <= (
                GOAL_COGNITION_PROMPT_CAP
                - len(str(getattr(message_list[0], "content", "")))
            )
        assert "branch_intent_guidance" not in bid


@pytest.mark.asyncio
async def test_ordinary_prompt_omits_branch_guidance() -> None:
    """Keep ordinary prompt and payload keys unchanged."""

    llm = _CapturingLLM([_ordinary_bid()])
    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
        {"scope": "user", "kind": "goal", "entity_id": "goal:ordinary"},
        {"_role_bindings": {}, "role_summaries": {}},
        _evidence(),
        _services(llm),
    )

    payload = _payload(llm.messages[0])
    assert list(payload["branch"]) == [
        "goal_kind",
        "action_tendencies",
    ]
    assert "branch_intent_guidance" not in payload["branch"]
    assert "branch.branch_intent_guidance" not in str(
        getattr(llm.messages[0][0], "content", "")
    )
    assert "branch_intent_guidance" not in bid
    assert bid["branch_id"] == "ordinary_response"


@pytest.mark.asyncio
async def test_required_selection_prompt_omits_branch_guidance() -> None:
    """Keep typed required-selection payloads guidance-free."""

    semantic_text = json.dumps({
        "role_explicit_content": "The current character must choose.",
        "response_operation": {
            "operation": "The current character chooses the next step.",
            "response_owner_role": "当前角色",
            "selection_owner_role": "当前角色",
            "selection_required": True,
            "embedded_actor_role": "当前用户",
            "embedded_target_role": "当前角色",
        },
    })
    llm = _CapturingLLM([_selection_bid()])
    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["autonomy_boundary"],
        {"scope": "user", "kind": "goal", "entity_id": "goal:selection"},
        {
            "_role_bindings": {
                "r1": {
                    "role": "target",
                    "entity_kind": "relationship",
                    "entity_id": "relationship:test",
                },
            },
            "role_summaries": {"r1": "The current relationship."},
        },
        _evidence(semantic_text=semantic_text),
        _services(llm),
    )

    payload = _payload(llm.messages[0])
    assert "branch_intent_guidance" not in payload["branch"]
    assert "branch_intent_guidance" not in bid
    assert bid["branch_id"] == "autonomy_boundary"
