"""Deterministic ownership test for src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py."""

from __future__ import annotations

from importlib import import_module

import pytest

import kazusa_ai_chatbot.nodes.persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.action_spec.registry import (
    FUTURE_SPEAK_CAPABILITY,
    SPEAK_CAPABILITY,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)

MODULE_PATH = "kazusa_ai_chatbot.nodes.persona_supervisor2_cognition"
EXPECTED_SYMBOLS = [
    "build_cognition_input_from_global_state",
    "build_scene_context_from_global_state",
]


def test_persona_supervisor2_cognition_exposes_owned_contract() -> None:
    """Keep the module's named owner contract discoverable."""

    module = import_module(MODULE_PATH)
    missing_symbols = [
        symbol
        for symbol in EXPECTED_SYMBOLS
        if not hasattr(module, symbol)
    ]

    assert not missing_symbols, (
        f"{MODULE_PATH} is missing owner symbols: {missing_symbols}"
    )


@pytest.mark.asyncio
async def test_user_cognition_commit_uses_compare_and_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Commit user cognition through the complete-state CAS boundary."""

    previous = build_acquaintance_user_state(
        global_user_id="node-cas-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    replacement = build_acquaintance_user_state(
        global_user_id="node-cas-user",
        updated_at="2026-08-18T00:01:00Z",
    )
    captured: dict[str, object] = {}

    async def compare_and_replace(
        owner_key: str,
        expected_state: dict[str, object],
        replacement_state: dict[str, object],
    ) -> bool:
        """Capture the state boundary and acknowledge the commit."""

        captured.update({
            "owner_key": owner_key,
            "expected_state": expected_state,
            "replacement_state": replacement_state,
        })
        return True

    async def record_commit(*args: object, **kwargs: object) -> None:
        """Keep the unit test focused on the persistence call."""

        del args, kwargs

    monkeypatch.setattr(
        cognition_module,
        "compare_and_replace_user_cognition_state",
        compare_and_replace,
    )
    monkeypatch.setattr(
        cognition_module,
        "_record_state_commit_event",
        record_commit,
    )

    await cognition_module._commit_cognition_state({
        "intention": {"selected_branch_id": "ordinary_response"},
        "state_update": {
            "state_scope": "user",
            "owner_key": "node-cas-user",
            "expected_previous_state": previous,
            "replacement_state": replacement,
        },
    })

    assert captured == {
        "owner_key": "node-cas-user",
        "expected_state": previous,
        "replacement_state": replacement,
    }


def test_group_self_cognition_proposal_materializes_existing_speak_surface(
    monkeypatch,
) -> None:
    """A speech intention uses the existing canonical speak materializer."""

    captured: list[list[dict[str, object]]] = []

    def materialize(
        requests: list[dict[str, object]],
        state: dict[str, object],
    ) -> list[dict[str, object]]:
        del state
        captured.append(requests)
        return [{"kind": request["capability"]} for request in requests]

    monkeypatch.setattr(
        cognition_module,
        "materialize_semantic_action_requests",
        materialize,
    )
    output = {
        "action_requests": [],
        "intention": {
            "route": "speech",
            "intention": "intervene in the current group scene",
            "target_roles": [],
            "reason": "The current scene supports a bounded intervention.",
        },
        "admitted_bid": {"evidence_handles": ["e1"]},
    }

    action_specs = cognition_module._materialize_v2_action_requests(
        output,
        {},
    )

    assert len(captured) == 2
    assert action_specs == [{"kind": SPEAK_CAPABILITY}]
    assert captured[1][0]["capability"] == SPEAK_CAPABILITY
    assert captured[1][0]["evidence_handles"] == ["e1"]


def test_promoted_reflection_preserves_source_updated_at() -> None:
    """Promoted rows retain their valid source time and omit invalid rows."""

    evidence = cognition_module._promoted_reflection_evidence(
        {
            'promoted_lore': [{
                'memory_name': 'world context',
                'content': 'the setting remains stable',
                'updated_at': '2026-07-29T23:00:00Z',
            }],
            'promoted_self_guidance': [{
                'memory_name': 'tactic hint',
                'content': 'verify the current scene first',
                'updated_at': 'not-a-timestamp',
            }],
        },
        '2026-07-30T00:00:00Z',
    )

    assert len(evidence) == 1
    assert evidence[0]['evidence_ref']['occurred_at'] == (
        '2026-07-29T23:00:00Z'
    )


def test_promoted_self_guidance_is_goal_only_conditional_context() -> None:
    """Self-guidance carries conditional authority and no current fact role."""

    evidence = cognition_module._promoted_reflection_evidence(
        {
            'promoted_self_guidance': [{
                'memory_name': 'tactic hint',
                'content': 'verify the current scene first',
                'updated_at': '2026-07-29T23:00:00Z',
            }],
        },
        '2026-07-30T00:00:00Z',
    )

    assert evidence[0]['authority'] == 'conditional_character_guidance'
    assert evidence[0]['evidence_ref']['source_id'] == (
        'promoted-reflection:self_guidance:1'
    )


def test_future_speak_v2_bridge_preserves_validated_authority_proposal(
    monkeypatch,
) -> None:
    """The runtime V2 bridge keeps the validated proposal on future_speak."""

    captured: list[list[dict[str, object]]] = []
    proposal = {
        "schema_version": "scheduled_authority_proposal.v1",
        "temporal_alignment": "aligned",
        "authorized_content_summary": "在约定时间开始补偿考核。",
        "authorized_detail_refs": [
            {
                "evidence_handle": "e1",
                "semantic_summary": (
                    "当前对话明确约定在该时间开始补偿考核。"
                ),
                "provenance_role": "current_event",
            }
        ],
    }

    def materialize(
        requests: list[dict[str, object]],
        state: dict[str, object],
    ) -> list[dict[str, object]]:
        del state
        captured.append(requests)
        return [
            {"kind": request["capability"]}
            for request in requests
            if isinstance(request, dict)
        ]

    monkeypatch.setattr(
        cognition_module,
        "materialize_semantic_action_requests",
        materialize,
    )
    output = {
        "action_requests": [
            {
                "action_kind": FUTURE_SPEAK_CAPABILITY,
                "decision": "2026-05-10 13:00",
                "context_ref": "current episode",
                "semantic_goal": "在约定时间开始补偿考核。",
                "reason": "用户要求在未来时间开始补偿考核。",
                "target_roles": [],
                "evidence_handles": ["e1"],
                "scheduled_authority_proposal": proposal,
            }
        ],
        "intention": {
            "route": "silence",
            "intention": "stay silent",
            "target_roles": [],
            "reason": "No visible surface selected.",
        },
    }

    action_specs = cognition_module._materialize_v2_action_requests(
        output,
        {},
    )

    assert len(captured) == 1
    materialized_request = captured[0][0]
    assert materialized_request["capability"] == FUTURE_SPEAK_CAPABILITY
    assert materialized_request["scheduled_authority_proposal"] == proposal
    assert action_specs == [{"kind": FUTURE_SPEAK_CAPABILITY}]
