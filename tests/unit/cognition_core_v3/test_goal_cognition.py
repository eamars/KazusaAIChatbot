"""Deterministic tests for serialized goal-cognition semantics."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import CognitionContractError
from kazusa_ai_chatbot.cognition_core_v3 import goal_cognition as gc


def test_ordinary_goal_remains_relational_willingness_owner() -> None:
    owners = {kind: gc.goal_kind_owns_relational_willingness(kind) for kind in ("ordinary_response", "safety")}
    assert owners["ordinary_response"] is True
    assert owners["safety"] is False


def test_active_goal_group_output_enforces_frozen_roster_order() -> None:
    bid = {
        "branch_id": "safety",
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
    normalized = gc.validate_active_goal_group_output(
        {"bids": [dict(bid)]},
        branch_roster=[{"branch_id": "safety"}],
        evidence_handles={"ev_1"},
        role_handles={"self"},
    )
    assert normalized[0]["branch_id"] == "safety"


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
            goal_kind="ordinary_response",
            evidence_handles={"ev_1"},
            role_handles={"self"},
        )
