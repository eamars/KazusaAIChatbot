"""Deterministic tests for serialized goal-cognition semantics."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import goal_cognition as gc
from kazusa_ai_chatbot.cognition_core_v3.registry import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_shared.contracts import CognitionContractError


def test_ordinary_goal_remains_relational_willingness_owner() -> None:
    owners = {kind: gc.goal_kind_owns_relational_willingness(kind) for kind in ("ordinary_response", "safety")}
    assert owners["ordinary_response"] is True
    assert owners["safety"] is False


def test_active_goal_group_output_enforces_frozen_roster_order() -> None:
    bid = {
        "branch_id": "safety_coping",
        "intention": "核实边界",
        "desired_outcome": "保持安全",
        "concrete_detail": "先确认证据",
        "reason": "角色需要确认当前事件边界。",
        "private_monologue": "我先看证据。",
        "target_role_handles": ["self"],
        "evidence_handles": ["ev_1"],
        "expected_consequences": ["回应保持有依据"],
        "confidence": "medium",
        "relational_willingness": {
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "current_user_relationship_state": "not_applicable",
            "reason": "当前分支不拥有关系立场。",
            "evidence_handles": ["ev_1"],
        },
    }
    normalized = gc.validate_active_goal_group_output(
        {"bids": [dict(bid)]},
        branch_roster=[
            {"branch_id": "safety_coping", "goal_kind": "safety"}
        ],
        evidence_handles={"ev_1"},
        role_handles={"self"},
    )
    assert normalized[0]["branch_id"] == "safety_coping"
    assert "relational_willingness" not in normalized[0]


def test_salvage_active_group_keeps_only_unique_independently_valid_rows() -> None:
    """Final structural output is filtered without repairing sibling fields."""

    valid_bid = {
        "branch_id": "safety_coping",
        "intention": "核实边界",
        "desired_outcome": "保持安全",
        "concrete_detail": "先确认证据",
        "reason": "角色需要确认当前事件边界。",
        "private_monologue": "我先看证据。",
        "target_role_handles": ["self"],
        "evidence_handles": ["ev_1"],
        "expected_consequences": ["回应保持有依据"],
        "confidence": "medium",
    }
    rows, failures = gc.salvage_active_goal_group_output(
        {
            "bids": [
                dict(valid_bid),
                dict(valid_bid),
                {
                    "branch_id": "trust_verification",
                    "intention": "缺少其余字段",
                },
            ]
        },
        branch_roster=[
            {"branch_id": "safety_coping", "goal_kind": "safety"},
            {
                "branch_id": "trust_verification",
                "goal_kind": "trust_verification",
            },
        ],
        evidence_handles={"ev_1"},
        role_handles={"self"},
    )
    assert [row["branch_id"] for row in rows] == []
    assert failures == {
        "safety_coping": "active_goal_group_duplicate_branch",
        "trust_verification": "active_goal_group_invalid_bid",
    }

    rows, failures = gc.salvage_active_goal_group_output(
        {"bids": [dict(valid_bid)]},
        branch_roster=[
            {"branch_id": "safety_coping", "goal_kind": "safety"},
            {
                "branch_id": "trust_verification",
                "goal_kind": "trust_verification",
            },
        ],
        evidence_handles={"ev_1"},
        role_handles={"self"},
    )
    assert [row["branch_id"] for row in rows] == ["safety_coping"]
    assert failures == {
        "trust_verification": "active_goal_group_missing_branch",
    }


def test_active_roster_accepts_canonical_branch_labels_for_distinct_goal_kinds() -> None:
    """Every registry branch keeps its label separate from its V2 kind."""

    for branch_id, definition in DEFAULT_BRANCH_DEFINITIONS.items():
        if branch_id == definition.goal_kind:
            continue
        bid = {
            "branch_id": branch_id,
            "intention": "evaluate",
            "desired_outcome": "continue",
            "concrete_detail": "stay grounded",
            "reason": "grounded",
            "private_monologue": "bounded",
            "target_role_handles": [],
            "evidence_handles": ["ev_1"],
            "expected_consequences": ["the exchange continues"],
            "confidence": "medium",
        }
        roster = [{"branch_id": branch_id, "goal_kind": definition.goal_kind}]

        normalized = gc.validate_active_goal_group_output(
            {"bids": [bid]},
            branch_roster=roster,
            evidence_handles={"ev_1"},
            role_handles={"self"},
        )
        assert normalized[0]["branch_id"] == branch_id

        rows, failures = gc.salvage_active_goal_group_output(
            {"bids": [bid]},
            branch_roster=roster,
            evidence_handles={"ev_1"},
            role_handles={"self"},
        )
        assert [row["branch_id"] for row in rows] == [branch_id]
        assert failures == {}


def test_active_roster_rejects_malformed_or_duplicate_identity_rows() -> None:
    """Roster identity errors fail before any candidate can be accepted."""

    bid = {
        "branch_id": "safety_coping",
        "intention": "evaluate",
        "desired_outcome": "continue",
        "concrete_detail": "stay grounded",
        "reason": "grounded",
        "private_monologue": "bounded",
        "target_role_handles": [],
        "evidence_handles": ["ev_1"],
        "expected_consequences": ["the exchange continues"],
        "confidence": "medium",
    }
    with pytest.raises(ValueError, match="goal_kind"):
        gc.validate_active_goal_group_output(
            {"bids": [bid]},
            branch_roster=[
                {"branch_id": "safety_coping", "goal_kind": "safety_coping"}
            ],
            evidence_handles={"ev_1"},
            role_handles={"self"},
        )

    with pytest.raises(ValueError, match="duplicated"):
        gc.salvage_active_goal_group_output(
            {"bids": [bid, bid]},
            branch_roster=[
                {"branch_id": "safety_coping", "goal_kind": "safety"},
                {"branch_id": "safety_coping", "goal_kind": "safety"},
            ],
            evidence_handles={"ev_1"},
            role_handles={"self"},
        )


def test_required_selection_preserves_code_owned_fields() -> None:
    authoritative_operation = {
        "operation": "",
        "embedded_actor_role": gc.NO_ROLE,
        "embedded_target_role": "current_user",
        "response_owner_role": "self",
        "selection_owner_role": "self",
        "selection_required": True,
    }
    bound = gc.bind_selected_response_operation(
        {"operation": "回答当前问题"},
        authoritative_operation,
    )
    assert bound["selection_required"] is True

    with pytest.raises(CognitionContractError, match="stance is invalid"):
        bad_stance = {
            "intention": "核实边界",
            "desired_outcome": "保持安全",
            "concrete_detail": "先确认证据",
            "reason": "角色需要确认当前事件边界。",
            "private_monologue": "我先看证据。",
            "target_role_handles": ["self"],
            "evidence_handles": ["ev_1"],
            "expected_consequences": ["回应保持有依据"],
            "confidence": "medium",
            "relational_willingness": {
                "applicability": "relationship_sensitive",
                "stance": "invented_stance",
                "current_user_relationship_state": "developing_or_uncertain",
                "reason": "关系仍在发展中，先协商边界。",
                "evidence_handles": ["ev_1"],
            },
        }
        gc.validate_goal_bid_draft(
            bad_stance,
            evidence_handles={"ev_1"},
            role_handles={"self"},
            require_relational_willingness=True,
            episode_handles={"ev_1"},
        )


def _ordinary_bid_candidate(evidence_handle: str) -> dict[str, object]:
    """Build one complete ordinary draft for canonical-owner tests."""

    return {
        "intention": "reply",
        "desired_outcome": "continue",
        "concrete_detail": "answer",
        "reason": "grounded",
        "private_monologue": "bounded",
        "target_role_handles": [],
        "evidence_handles": [evidence_handle],
        "expected_consequences": ["the exchange continues"],
        "confidence": "medium",
        "relational_willingness": {
            "applicability": "not_relationship_sensitive",
            "stance": "not_applicable",
            "current_user_relationship_state": "not_applicable",
            "reason": "当前关系立场来自已验证的前一轮。",
            "evidence_handles": [evidence_handle],
        },
    }


def test_cold_ordinary_willingness_requires_current_episode_evidence() -> None:
    """Cold ordinary validation rejects a non-episode-only stance."""

    candidate = _ordinary_bid_candidate("prior_1")
    with pytest.raises(
        CognitionContractError,
        match="must cite current episode evidence",
    ):
        gc.validate_goal_bid_draft(
            candidate,
            evidence_handles={"prior_1", "episode_1"},
            role_handles=set(),
            require_relational_willingness=True,
            episode_handles={"episode_1"},
        )


def test_recurrence_carrier_revalidates_through_v2_owner() -> None:
    """Recurrence carries the accepted stance without current-episode rebinding."""

    candidate = _ordinary_bid_candidate("new_1")
    carried = dict(candidate["relational_willingness"])
    assert isinstance(carried, dict)
    carried["schema_version"] = "relational_willingness.v2"
    validated = gc.validate_recurrence_ordinary_goal_bid_draft(
        {
            key: value
            for key, value in candidate.items()
            if key != "relational_willingness"
        },
        carried_relational_willingness=carried,
        evidence_handles={"prior_1", "new_1"},
        role_handles=set(),
    )
    assert validated["relational_willingness"] == carried
