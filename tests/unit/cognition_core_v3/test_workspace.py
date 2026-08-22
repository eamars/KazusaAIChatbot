"""Deterministic tests for V3 complete-bid workspace collapse."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import workspace as v2_workspace
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import branch_order_key
from kazusa_ai_chatbot.cognition_core_v3 import workspace as ws


def _complete_bid(
    branch_id: str,
    *,
    entity_id: str = "g1",
    confidence: str = "medium",
) -> dict:
    bid = {
        "branch_id": branch_id,
        "goal_ref": {"entity_id": entity_id},
        "intention": f"{branch_id} intention",
        "desired_outcome": f"{branch_id} outcome",
        "concrete_detail": f"{branch_id} detail",
        "reason": f"{branch_id} reason",
        "private_monologue": f"{branch_id} monologue",
        "target_roles": [
            {"role": "actor", "entity_kind": "character", "entity_id": "character:global"},
        ],
        "evidence_handles": ["ev_1"],
        "expected_consequences": [f"{branch_id} consequence"],
        "confidence": confidence,
    }
    if branch_id == "ordinary_response":
        bid["relational_willingness"] = {
            "schema_version": "relational_willingness.v2",
            "applicability": "relationship_sensitive",
            "stance": "negotiate",
            "current_user_relationship_state": "developing_or_uncertain",
            "reason": "关系仍在发展中，先协商边界。",
            "evidence_handles": ["ev_1"],
        }
    return bid


def test_workspace_preserves_ordinary_relational_authority():
    decision = dict(_complete_bid("ordinary_response")["relational_willingness"])
    bids = [
        _complete_bid("bond_protection", entity_id="g2"),
        _complete_bid("ordinary_response"),
        _complete_bid("loss_recovery", entity_id="g3"),
    ]

    envelope = v2_workspace.collapse_authoritative_relational_bid(
        bids,
        decision,
    )
    assert envelope["primary_branch_id"] == "ordinary_response"
    assert envelope["supporting_branch_ids"] == []
    assert envelope["suppressed_branch_ids"] == ["bond_protection", "loss_recovery"]
    assert [bid["branch_id"] for bid in envelope["competing_bids"]] == [
        "bond_protection",
        "loss_recovery",
    ]

    # Confidence descriptors never rank: swapping them leaves the partition
    # structure identical even though the carried bid copies differ.
    inverted = [dict(bid) for bid in bids]
    inverted[0]["confidence"] = "high"
    inverted[2]["confidence"] = "low"
    re_collapsed = v2_workspace.collapse_authoritative_relational_bid(
        inverted,
        decision,
    )
    assert re_collapsed["primary_branch_id"] == envelope["primary_branch_id"]
    assert re_collapsed["supporting_branch_ids"] == envelope["supporting_branch_ids"]
    assert re_collapsed["suppressed_branch_ids"] == envelope["suppressed_branch_ids"]
    assert [bid["branch_id"] for bid in re_collapsed["competing_bids"]] == [
        "bond_protection",
        "loss_recovery",
    ]

    non_sensitive = dict(decision)
    non_sensitive["applicability"] = "not_relationship_sensitive"
    with pytest.raises(ValueError, match="requires a sensitive decision"):
        v2_workspace.collapse_authoritative_relational_bid(
            bids,
            non_sensitive,
        )

    without_ordinary = [bid for bid in bids if bid["branch_id"] != "ordinary_response"]
    with pytest.raises(ValueError, match="exactly one ordinary bid"):
        v2_workspace.collapse_authoritative_relational_bid(
            without_ordinary,
            decision,
        )

    duplicate_ordinary = [
        _complete_bid("bond_protection", entity_id="g2"),
        _complete_bid("ordinary_response"),
        _complete_bid("ordinary_response"),
    ]
    with pytest.raises(ValueError, match="exactly one ordinary bid"):
        v2_workspace.collapse_authoritative_relational_bid(
            duplicate_ordinary,
            decision,
        )

    mismatched = [dict(bid) for bid in bids]
    mismatched[1]["relational_willingness"] = dict(decision)
    mismatched[1]["relational_willingness"]["stance"] = "reject"
    with pytest.raises(ValueError, match="requires the equal decision"):
        v2_workspace.collapse_authoritative_relational_bid(
            mismatched,
            decision,
        )


def test_workspace_admits_only_complete_current_matter_bids():
    admitted = ws.validate_complete_bids(
        [
            _complete_bid("ordinary_response"),
            _complete_bid("bond_protection", entity_id="g2"),
        ]
    )
    assert [bid["branch_id"] for bid in admitted] == [
        "ordinary_response",
        "bond_protection",
    ]

    mutated = ws.validate_complete_bids([_complete_bid("safety")])
    mutated[0]["intention"] = "mutated"
    revalidated = ws.validate_complete_bids([_complete_bid("safety")])
    assert revalidated[0]["intention"] == "safety intention"

    incomplete = _complete_bid("ordinary_response")
    del incomplete["reason"]
    with pytest.raises(ValueError, match="not complete"):
        ws.validate_complete_bids([incomplete])

    unknown_field = _complete_bid("safety")
    unknown_field["invented_field"] = True
    with pytest.raises(ValueError, match="not complete"):
        ws.validate_complete_bids([unknown_field])

    ordinary_without_willingness = _complete_bid("ordinary_response")
    del ordinary_without_willingness["relational_willingness"]
    with pytest.raises(ValueError, match="lacks the relational willingness decision"):
        ws.validate_complete_bids([ordinary_without_willingness])

    bad_goal_ref = _complete_bid("safety")
    bad_goal_ref["goal_ref"] = {"entity_id": ""}
    with pytest.raises(ValueError, match="invalid goal ref"):
        ws.validate_complete_bids([bad_goal_ref])

    # Admission is all-or-nothing: one field-missing candidate rejects the batch.
    incomplete_member = _complete_bid("safety")
    del incomplete_member["concrete_detail"]
    with pytest.raises(ValueError, match="not complete"):
        ws.validate_complete_bids(
            [
                _complete_bid("ordinary_response"),
                incomplete_member,
            ]
        )

    # A content-violating candidate rejects the batch just as decisively.
    with pytest.raises(ValueError, match="invalid intention"):
        ws.validate_complete_bids(
            [
                _complete_bid("ordinary_response"),
                {**_complete_bid("safety"), "intention": ""},
            ]
        )

    single_envelope = ws.collapse_single_bid(_complete_bid("safety"))
    assert single_envelope["primary_branch_id"] == "safety"
    assert single_envelope["supporting_bids"] == []
    assert single_envelope["competing_bids"] == []

    ordinary = _complete_bid("ordinary_response")
    persistent = _complete_bid("bond_protection", entity_id="g2")
    request = ws.prepare_partition(
        [persistent, ordinary],
        [{"text": "current event row"}],
        {"g2": {"goal_kind": "bond_protection", "status": "pursuing"}},
    )

    expected_order = sorted(
        [ordinary, persistent], key=lambda bid: branch_order_key(bid["branch_id"])
    )
    assert list(request.handles) == ["b1", "b2"]
    assert request.handles["b1"]["branch_id"] == expected_order[0]["branch_id"]
    assert request.handles["b2"]["branch_id"] == expected_order[1]["branch_id"]

    ordinary_handle = next(
        handle for handle, bid in request.handles.items() if bid["branch_id"] == "ordinary_response"
    )
    persistent_handle = next(
        handle for handle, bid in request.handles.items() if bid["branch_id"] == "bond_protection"
    )
    assert request.prompt_payload["bid_index"][ordinary_handle][
        "persistent_goal"
    ] is None
    assert request.prompt_payload["bid_index"][persistent_handle][
        "persistent_goal"
    ] == {
        "goal_kind": "bond_protection",
        "lifecycle": "pursuing",
    }

    # A persistent bid whose goal ref has no matter context fails fail-fast.
    with pytest.raises(KeyError, match="g1"):
        ws.prepare_partition(
            [_complete_bid("safety")],
            [],
            {},
        )

    handles = request.handles
    partition = {
        "primary_bid_handle": persistent_handle,
        "supporting_bid_handles": [ordinary_handle],
        "suppressed_bid_handles": [],
    }
    assert v2_workspace.validate_workspace_partition(
        partition,
        set(handles),
    ) == partition

    envelope = ws.materialize_partition(handles, partition)
    assert envelope["primary_branch_id"] == "bond_protection"
    assert envelope["supporting_branch_ids"] == ["ordinary_response"]
    assert envelope["suppressed_branch_ids"] == []
    assert [bid["branch_id"] for bid in envelope["competing_bids"]] == []

    extra_field = {**partition, "invented_field": True}
    with pytest.raises(ValueError, match="fields are not exact"):
        v2_workspace.validate_workspace_partition(extra_field, set(handles))

    bad_primary = dict(partition)
    bad_primary["primary_bid_handle"] = "b9"
    with pytest.raises(ValueError, match="primary handle is unavailable"):
        v2_workspace.validate_workspace_partition(bad_primary, set(handles))

    unknown_supporting = {**partition, "supporting_bid_handles": ["b9"]}
    with pytest.raises(ValueError, match="handle is unavailable"):
        v2_workspace.validate_workspace_partition(
            unknown_supporting,
            set(handles),
        )

    incomplete_without_suppressed = {
        "primary_bid_handle": persistent_handle,
        "supporting_bid_handles": [],
        "suppressed_bid_handles": [],
    }
    with pytest.raises(ValueError, match="partition is incomplete"):
        v2_workspace.validate_workspace_partition(
            incomplete_without_suppressed,
            set(handles),
        )

    duplicated_ordinary = {
        "primary_bid_handle": persistent_handle,
        "supporting_bid_handles": [ordinary_handle],
        "suppressed_bid_handles": [ordinary_handle],
    }
    with pytest.raises(ValueError, match="partition is incomplete"):
        v2_workspace.validate_workspace_partition(
            duplicated_ordinary,
            set(handles),
        )

    fallback = ws.fallback_partition_envelope([persistent, ordinary])
    assert fallback["primary_branch_id"] == "bond_protection"
    assert fallback["supporting_branch_ids"] == []
    assert fallback["suppressed_branch_ids"] == ["ordinary_response"]
    assert [bid["branch_id"] for bid in fallback["competing_bids"]] == [
        "ordinary_response",
    ]

    with pytest.raises(ValueError, match="at least one bid"):
        ws.fallback_partition_envelope([])


def test_target_roles_require_structured_role_refs():
    """Admission accepts materialized RoleRefV2 entries and rejects legacy handles.

    The goal-bid materializer emits the V2 ``RoleRefV2`` shape, so a complete
    bid carrying structured refs must be admitted unchanged while the retired
    string-handle shape fails closed with an entry-level error.
    """
    admitted = ws.validate_complete_bids([_complete_bid("bond_protection")])
    assert admitted[0]["target_roles"] == [
        {"role": "actor", "entity_kind": "character", "entity_id": "character:global"},
    ]

    string_handle = _complete_bid("loss_recovery")
    string_handle["target_roles"] = ["self"]
    with pytest.raises(ValueError, match="has an invalid target_role entry at 0"):
        ws.validate_complete_bids([string_handle])
