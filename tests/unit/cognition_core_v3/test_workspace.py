"""Deterministic tests for V3 complete-bid workspace collapse."""

from __future__ import annotations

import pytest

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
        "target_roles": ["self"],
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

    envelope = ws.collapse_authoritative_relational_bid(bids, decision)
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
    re_collapsed = ws.collapse_authoritative_relational_bid(inverted, decision)
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
        ws.collapse_authoritative_relational_bid(bids, non_sensitive)

    without_ordinary = [bid for bid in bids if bid["branch_id"] != "ordinary_response"]
    with pytest.raises(ValueError, match="exactly one ordinary bid"):
        ws.collapse_authoritative_relational_bid(without_ordinary, decision)

    duplicate_ordinary = [
        _complete_bid("bond_protection", entity_id="g2"),
        _complete_bid("ordinary_response"),
        _complete_bid("ordinary_response"),
    ]
    with pytest.raises(ValueError, match="exactly one ordinary bid"):
        ws.collapse_authoritative_relational_bid(duplicate_ordinary, decision)

    mismatched = [dict(bid) for bid in bids]
    mismatched[1]["relational_willingness"] = dict(decision)
    mismatched[1]["relational_willingness"]["stance"] = "reject"
    with pytest.raises(ValueError, match="requires the equal decision"):
        ws.collapse_authoritative_relational_bid(mismatched, decision)


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
        {"g2": {"kind": "bond_protection", "matter": "bond matter context"}},
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
    assert request.prompt_payload["bids"][ordinary_handle]["persistent_goal"] is None
    assert request.prompt_payload["bids"][persistent_handle]["persistent_goal"] == {
        "kind": "bond_protection",
        "matter": "bond matter context",
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
    assert ws.validate_partition(partition, handles) == partition

    envelope = ws.materialize_partition(handles, partition)
    assert envelope["primary_branch_id"] == "bond_protection"
    assert envelope["supporting_branch_ids"] == ["ordinary_response"]
    assert envelope["suppressed_branch_ids"] == []
    assert [bid["branch_id"] for bid in envelope["competing_bids"]] == []

    extra_field = {**partition, "invented_field": True}
    with pytest.raises(ValueError, match="fields are not exact"):
        ws.validate_partition(extra_field, handles)

    bad_primary = dict(partition)
    bad_primary["primary_bid_handle"] = "b9"
    with pytest.raises(ValueError, match="primary handle is unavailable"):
        ws.validate_partition(bad_primary, handles)

    unknown_supporting = {**partition, "supporting_bid_handles": ["b9"]}
    with pytest.raises(ValueError, match="handle is unavailable"):
        ws.validate_partition(unknown_supporting, handles)

    incomplete_without_suppressed = {
        "primary_bid_handle": persistent_handle,
        "supporting_bid_handles": [],
        "suppressed_bid_handles": [],
    }
    with pytest.raises(ValueError, match="partition is incomplete"):
        ws.validate_partition(incomplete_without_suppressed, handles)

    duplicated_ordinary = {
        "primary_bid_handle": persistent_handle,
        "supporting_bid_handles": [ordinary_handle],
        "suppressed_bid_handles": [ordinary_handle],
    }
    with pytest.raises(ValueError, match="partition is incomplete"):
        ws.validate_partition(duplicated_ordinary, handles)

    fallback = ws.fallback_partition_envelope([persistent, ordinary])
    assert fallback["primary_branch_id"] == "bond_protection"
    assert fallback["supporting_branch_ids"] == []
    assert fallback["suppressed_branch_ids"] == ["ordinary_response"]
    assert [bid["branch_id"] for bid in fallback["competing_bids"]] == [
        "ordinary_response",
    ]

    with pytest.raises(ValueError, match="at least one bid"):
        ws.fallback_partition_envelope([])
