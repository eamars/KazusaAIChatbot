"""Direct ownership tests for stance-to-effect action propagation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    ACTION_PLANNING_PROMPT,
    plan_actions,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
)
from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
)
from tests.cognition_core_v2_test_helpers import canonical_episode


def _decision() -> dict[str, object]:
    """Build a non-accepting selected stance without policy semantics."""

    return {
        "schema_version": "relational_willingness.v2",
        "applicability": "relationship_sensitive",
        "stance": "reject",
        "current_user_relationship_state": "unestablished",
        "reason": "the selected character stance is non-accepting",
        "evidence_handles": ["e1"],
    }


def _primary_bid() -> dict[str, object]:
    """Build the complete ordinary bid required by action planning."""

    return {
        "branch_id": "ordinary_response",
        "goal_ref": {
            "scope": "user",
            "kind": "goal",
            "entity_id": "goal:ordinary-response",
        },
        "intention": "maintain the current response",
        "desired_outcome": "preserve the selected direction",
        "concrete_detail": "use current evidence",
        "reason": "the selected character stance owns this turn",
        "private_monologue": "keep the selected stance intact",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["preserve response continuity"],
        "confidence": "high",
        "selected_response_operation": {
            "operation": "the user gives the selected reward to the character",
            "response_owner_role": CURRENT_CHARACTER_ROLE,
            "selection_owner_role": CURRENT_CHARACTER_ROLE,
            "selection_required": True,
            "embedded_actor_role": CURRENT_USER_ROLE,
            "embedded_target_role": CURRENT_CHARACTER_ROLE,
        },
        "relational_willingness": _decision(),
    }


def _evidence() -> list[dict[str, object]]:
    """Build one current-episode evidence row for action planning."""

    return [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode-action",
            "occurred_at": "2026-07-14T00:00:00Z",
            "semantic_summary": "current evidence",
        },
        "semantic_text": "The current turn supplies the selected direction.",
        "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS["episode"]),
    }]


@pytest.mark.asyncio
async def test_non_accepting_stance_suppresses_downstream_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-accepting stances suppress effects before authorization."""

    async def fake_planner(**_: object) -> dict[str, object]:
        return {
            "action_requests": [{
                "bid_handle": "b1",
                "action_handle": "a1",
                "decision": "",
                "semantic_goal": "perform the proposed effect",
                "reason": "the planner proposed the effect",
            }],
            "resolver_requests": [],
            "goal_resolution": "answerable_now",
            "resolver_pending_resolution": None,
            "resolver_goal_progress": None,
        }

    async def unexpected_authorization(**_: object) -> list[dict[str, str]]:
        raise AssertionError("selected stance must suppress authorization")

    import kazusa_ai_chatbot.cognition_core_v2.action_selection as action_module

    monkeypatch.setattr(action_module, "_invoke_action_planner", fake_planner)
    monkeypatch.setattr(
        action_module,
        "authorize_action_requests",
        unexpected_authorization,
    )
    monkeypatch.setattr(
        action_module,
        "authorize_resolver_requests",
        unexpected_authorization,
    )

    action_affordance = {
        "action_kind": "background_work_request",
        "capability": "bounded background work",
        "permission": "allowed",
        "decision_mode": "optional",
        "allowed_decisions": [],
        "default_decision": "",
        "decision_pattern": "",
        "context_ref": "current episode",
        "target_roles": [],
    }
    result = await plan_actions(
        primary_bid=_primary_bid(),
        supporting_bids=[],
        episode=canonical_episode(
            episode_id="action-episode",
            content="current request",
        ),
        evidence=_evidence(),
        available_actions=[action_affordance],
        available_resolvers=[],
        resolver_context="",
        services=SimpleNamespace(),
    )

    assert result["action_requests"] == []
    assert result["resolver_requests"] == []
    assert result["intention"]["route"] == "speech"


@pytest.mark.asyncio
async def test_selected_intention_preserves_selected_response_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Action selection carries the selected operation into the intention."""

    async def fake_planner(**_: object) -> dict[str, object]:
        return {
            "action_requests": [],
            "resolver_requests": [],
            "goal_resolution": "answerable_now",
            "resolver_pending_resolution": None,
            "resolver_goal_progress": None,
        }

    import kazusa_ai_chatbot.cognition_core_v2.action_selection as action_module

    monkeypatch.setattr(action_module, "_invoke_action_planner", fake_planner)
    result = await plan_actions(
        primary_bid=_primary_bid(),
        supporting_bids=[],
        episode=canonical_episode(
            episode_id="selected-operation-action-episode",
            content="current request",
        ),
        evidence=_evidence(),
        available_actions=[],
        available_resolvers=[],
        resolver_context="",
        services=SimpleNamespace(),
    )

    assert result["intention"]["selected_response_operation"] == (
        _primary_bid()["selected_response_operation"]
    )


def test_action_selection_exposes_owned_contract() -> None:
    """Keep the action planner entrypoint attached to this owner."""

    assert callable(plan_actions)


def test_action_planning_keeps_confidence_descriptor_advisory() -> None:
    """Action planning cannot treat confidence context as a quality score."""

    prompt = ACTION_PLANNING_PROMPT

    assert "confidence 是有界的置信度描述" in prompt
    assert "不是 score" in prompt
