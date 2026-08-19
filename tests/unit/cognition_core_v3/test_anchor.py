"""Deterministic tests for the Cognition V3 system anchor."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import anchor, prompt


def test_system_head_excludes_dynamic_turn_data_and_is_byte_stable() -> None:
    """The system head contains only the static manual and ordered identity."""

    character_identity = {
        "self_image": {
            "self_concept": "quietly self-directed",
            "current_growth_edges": ["state needs directly"],
        },
        "boundaries": {
            "self_integrity": "strong",
            "control_sensitivity": "high",
        },
        "personality": {
            "logic": "keep source and role ownership clear",
            "tempo": "measured",
        },
        "core": {
            "name": "character-name-marker",
            "description": "identity-description-marker",
        },
    }
    reordered_identity = {
        "core": dict(character_identity["core"]),
        "personality": dict(character_identity["personality"]),
        "boundaries": dict(character_identity["boundaries"]),
        "self_image": dict(character_identity["self_image"]),
    }

    first_head = anchor.build_system_head(character_identity)
    second_head = anchor.build_system_head(reordered_identity)

    assert first_head == second_head
    assert "episode-dynamic-marker" not in first_head
    assert "relationship-dynamic-marker" not in first_head
    assert "evidence-dynamic-marker" not in first_head
    assert "route-dynamic-marker" not in first_head

    decoded = json.loads(first_head)
    assert [next(iter(section)) for section in decoded] == [
        "engine_manual",
        "character_identity",
    ]
    assert decoded[0] == {"engine_manual": anchor.ENGINE_MANUAL}
    for contract_name in prompt.CHAIN_CONTRACT_NAMES:
        assert contract_name in anchor.ENGINE_MANUAL
    for stable_schema_field in (
        "question_id",
        "proposition",
        "delta",
        "relational_willingness",
        "selected_response_operation",
        "primary_bid_handle",
        "supporting_bid_handles",
        "suppressed_bid_handles",
        "action_requests",
        "resolver_requests",
        "goal_resolution",
        "resolver_pending_resolution",
        "resolver_goal_progress",
    ):
        assert stable_schema_field in anchor.ENGINE_MANUAL
    for closed_value in (
        "relationship_sensitive",
        "conditional_accept",
        "answerable_now",
        "requires_required_evidence",
        "stay_silent",
        "propose_visible_reply",
    ):
        assert closed_value in anchor.ENGINE_MANUAL
    identity_rows = decoded[1]["character_identity"]
    assert [next(iter(row)) for row in identity_rows] == list(
        anchor.IDENTITY_PARTITION_ORDER
    )
    assert identity_rows == [
        {partition: reordered_identity[partition]}
        for partition in anchor.IDENTITY_PARTITION_ORDER
    ]

    changed_identity = dict(reordered_identity)
    changed_identity["core"] = {
        **reordered_identity["core"],
        "description": "changed-identity-description",
    }
    assert anchor.build_system_head(changed_identity) != first_head

    with pytest.raises(anchor.AnchorContractError, match="exact partitions"):
        anchor.build_system_head(
            {
                **reordered_identity,
                "episode_and_scene": {"episode_id": "forbidden"},
            }
        )
    with pytest.raises(anchor.AnchorContractError, match="dynamic field"):
        anchor.build_system_head(
            {
                **reordered_identity,
                "core": {
                    **reordered_identity["core"],
                    "trace_id": "forbidden",
                },
            }
        )
