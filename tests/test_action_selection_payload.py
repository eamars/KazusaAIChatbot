"""V2 semantic action-planning payload ownership tests."""

import json
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import (
    ACTION_PLANNING_PROMPT,
    plan_actions,
)
from tests.test_task_resolution_orchestrator import _scene_context


def _bid() -> dict[str, object]:
    """Build one admitted motive with complete deterministic provenance."""

    return {
        "branch_id": "ordinary_response",
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        "intention": "advance the admitted motive",
        "desired_outcome": "preserve a grounded interaction",
        "concrete_detail": "use only current evidence",
        "reason": "the admitted evidence supports this motive",
        "private_monologue": "I should respond deliberately.",
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "user-1",
        }],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the interaction remains coherent"],
        "confidence": "high",
    }


def test_action_prompt_owns_semantic_selection_only() -> None:
    """The planner selects grounded objectives without rewriting motives."""

    prompt = ACTION_PLANNING_PROMPT.casefold()

    assert "bid_handle" in prompt
    assert "不改写目标候选" in prompt
    assert "协议代码会在语义授权完成后派生 route" in prompt
    assert "semantic_goal" in prompt
    assert "decision" in prompt
    assert "task_willingness" not in prompt
    assert "raw media" not in prompt
    assert "task_resolution_request" in prompt


def test_action_prompt_keeps_runtime_constraints_out_of_objectives() -> None:
    """Capability, permission, and feasibility stay downstream runtime facts."""

    prompt = "".join(ACTION_PLANNING_PROMPT.split())

    assert "能力、权限、可行性和API支持是运行时约束" in prompt
    assert "task_resolution_request的semantic_goal" in prompt
    assert "审计目标" in prompt
    assert "provenance_role" in prompt


def test_action_prompt_exposes_routing_boolean_without_queue_mechanics() -> None:
    """The planner owns the narrow boolean, never durable execution details."""

    prompt = "".join(ACTION_PLANNING_PROMPT.split()).casefold()

    assert "start_in_background" in prompt
    for forbidden in (
        "accepted_task_id",
        "idempotency_key",
        "inline_budget_seconds",
        "checkpoint",
        "queue_request",
    ):
        assert forbidden not in prompt


@pytest.mark.asyncio
async def test_action_planning_payload_projects_provenance_roles_only() -> None:
    """The model sees semantic authority labels, never raw source metadata."""

    captured: dict[str, object] = {}

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            captured.update(json.loads(str(messages[-1].content)))
            return SimpleNamespace(content=json.dumps({
                "action_requests": [],
                "resolver_requests": [],
                "goal_resolution": "answerable_now",
                "resolver_pending_resolution": None,
                "resolver_goal_progress": None,
            }))

    await plan_actions(
        primary_bid=_bid(),
        supporting_bids=[],
        episode={
            "episode_id": "episode-provenance",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[
            {
                "evidence_handle": "e1",
                "evidence_ref": {
                    "source_kind": "episode",
                    "source_id": "raw-episode-1",
                    "occurred_at": "2026-08-07T00:00:00Z",
                    "semantic_summary": "the user made a retrieval request",
                },
                "semantic_text": "the user asked to retrieve chat history",
                "visible_to": ["q:event_agency"],
                "authority": "current_event",
            },
            {
                "evidence_handle": "e2",
                "evidence_ref": {
                    "source_kind": "conversation_evidence",
                    "source_id": "raw-conversation-2",
                    "occurred_at": "2026-08-06T00:00:00Z",
                    "semantic_summary": "a recent conversation row",
                },
                "semantic_text": "a recent conversation row",
                "visible_to": ["q:event_agency"],
                "authority": "participant_continuity",
            },
        ],
        available_actions=[],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    projected_evidence = captured["evidence"]
    assert projected_evidence == [
        {
            "handle": "e1",
            "source_kind": "episode",
            "provenance_role": "current_episode",
            "semantic_text": "the user asked to retrieve chat history",
        },
        {
            "handle": "e2",
            "source_kind": "conversation_evidence",
            "provenance_role": "contextual_fact_only",
            "semantic_text": "a recent conversation row",
        },
    ]
    serialized = json.dumps(projected_evidence, ensure_ascii=False)
    assert "raw-episode-1" not in serialized
    assert "raw-conversation-2" not in serialized
    assert "occurred_at" not in serialized


@pytest.mark.asyncio
async def test_action_planning_payload_projects_authoritative_scene_context() -> None:
    """The planner receives only the bounded canonical scene projection."""

    captured: dict[str, object] = {}

    class _LLM:
        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            captured.update(json.loads(str(messages[-1].content)))
            return SimpleNamespace(content=json.dumps({
                "action_requests": [],
                "resolver_requests": [],
                "goal_resolution": "answerable_now",
                "resolver_pending_resolution": None,
                "resolver_goal_progress": None,
            }))

    scene = _scene_context()
    scene["public_group_scene"] = "The public group scene contains one bounded fact."
    scene["participant_bindings"] = [{
        "handle": "participant_1",
        "role": "current_user",
        "display_name": "current user",
    }]
    await plan_actions(
        primary_bid=_bid(),
        supporting_bids=[],
        episode={
            "episode_id": "episode-scene",
            "trigger_source": "user_message",
            "output_mode": "visible_reply",
        },
        evidence=[],
        available_actions=[],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        scene_context=scene,
        services=SimpleNamespace(
            llm=_LLM(),
            action_planning_config=object(),
            action_authorization_config=object(),
            resolver_authorization_config=object(),
        ),
    )

    assert captured["scene_context"] == scene
    serialized = json.dumps(captured["scene_context"], ensure_ascii=False)
    assert "platform_channel_id" not in serialized
    assert "raw" not in serialized
