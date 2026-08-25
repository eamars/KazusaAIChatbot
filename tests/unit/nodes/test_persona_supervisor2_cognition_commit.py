"""Focused commit and continuation-lineage checks for canonical cognition."""

from __future__ import annotations

from copy import deepcopy

import pytest

from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionContractError,
    _validate_evidence_rows,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_node
from tests.unit.cognition_core_v3.test_handleless_contract import _input


def test_dialog_semantic_projection_excludes_procedural_provider_metadata() -> None:
    """Cognition receives explicit dialog meaning without response mechanics."""

    episode = deepcopy(_input()["episode"])
    dialog_content = episode["percepts"][0]["content"]
    dialog_content["role_explicit_content"] = "当前用户请求当前角色回应。"
    dialog_content["response_operation"] = {
        "operation": "当前角色提供回复内容。",
        "response_owner_role": "当前角色",
        "response_content_provider_role": "当前角色",
        "selection_required": True,
        "embedded_actor_role": "无",
        "embedded_target_role": "无",
    }

    projection = cognition_node._dialog_semantic_projection_text(episode)

    assert projection == dialog_content["role_explicit_content"]
    assert "response_content_provider_role" not in projection


@pytest.mark.asyncio
async def test_resolver_recurrence_commits_against_original_user_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _input()
    original = payload["mutable_state"]
    replacement = dict(original)
    replacement["relationship"] = dict(original["relationship"])
    replacement["relationship"]["trust"] = 10
    captured: dict[str, object] = {}

    async def replace_user(owner: str, expected: dict, next_state: dict) -> bool:
        captured["owner"] = owner
        captured["expected"] = expected
        captured["replacement"] = next_state
        return True

    monkeypatch.setattr(
        cognition_node,
        "compare_and_replace_user_cognition_state",
        replace_user,
    )
    output = {
        "schema_version": "cognition_output.v3",
        "state_projection": {
            "state_scope": "user",
            "owner_key": "user-1",
            "expected_previous_state": replacement,
            "original_persisted_state": original,
            "replacement_state": replacement,
        },
    }
    await cognition_node.commit_cognition_output(output)
    assert captured["owner"] == "user-1"
    assert captured["expected"] == original
    assert captured["replacement"] == replacement


def test_current_continuation_uses_exact_private_goal_ref() -> None:
    payload = _input()
    replacement = build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=payload["mutable_state"]["updated_at"],
    )
    output = {
        "state_projection": {
            "continuation_goal_ref": {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary_response:user:current",
            },
        },
    }
    continuation = cognition_node._canonical_goal_continuation_ref(
        output,
        {"cognitive_episode": payload["episode"]},
        replacement,
    )
    assert continuation["goal_ref"]["entity_id"] == (
        "goal:ordinary_response:user:current"
    )


def test_global_projection_preserves_exact_private_monologue() -> None:
    """Global residue state receives G subjectivity rather than goal analysis."""

    payload = _input()
    caller_state = {
        **payload,
        "global_user_id": "user-1",
        "cognitive_episode": payload["episode"],
    }
    output = {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {
            "goal_kind": "clarify",
            "intent": "understand the unfamiliar object",
            "reason": "the observation does not identify the object",
            "cause_summary": "an unfamiliar object appeared",
        },
        "private_monologue": (
            "I am curious, but I do not want to pretend I recognize it."
        ),
        "response_plan": {
            "goal_resolution": "answerable_now",
            "response_goal": "ask what the unfamiliar object is",
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": (
                "Describe its visible form and leave its identity unknown."
            ),
        },
        "state_projection": {
            "replacement_state": payload["mutable_state"],
        },
        "affect_projection": [],
        "relationship_projection": {},
        "relational_willingness": {},
        "cause_provenance": [],
    }

    projected = cognition_node._project_output_to_global_state(
        output,
        caller_state,
        available_actions=payload["available_actions"],
        available_resolver_capabilities=(
            payload["available_resolver_capabilities"]
        ),
    )

    assert projected["internal_monologue"] == output["private_monologue"]
    assert projected["internal_monologue"] != (
        output["active_character_goal"]["reason"]
    )


def test_global_projection_supplies_consolidation_interaction_subtext() -> None:
    """Project the goal reason and private monologue into separate fields."""

    payload = _input()
    caller_state = {
        **payload,
        "global_user_id": "user-1",
        "cognitive_episode": payload["episode"],
    }
    reason = "the compass reacts to a direction I cannot see"
    private_monologue = "I should ask what makes the needle move before guessing."
    output = {
        "schema_version": "cognition_output.v3",
        "active_character_goal": {
            "goal_kind": "clarify",
            "intent": "understand the unfamiliar compass",
            "reason": reason,
            "cause_summary": "the compass needle moved without an obvious cause",
        },
        "private_monologue": private_monologue,
        "response_plan": {
            "goal_resolution": "answerable_now",
            "response_goal": "ask what makes the compass needle move",
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": (
                "Describe the movement and leave its cause unknown."
            ),
        },
        "state_projection": {
            "replacement_state": payload["mutable_state"],
        },
        "affect_projection": [],
        "relationship_projection": {},
        "relational_willingness": {},
        "cause_provenance": [],
    }

    projected = cognition_node._project_output_to_global_state(
        output,
        caller_state,
        available_actions=payload["available_actions"],
        available_resolver_capabilities=(
            payload["available_resolver_capabilities"]
        ),
    )

    assert projected["interaction_subtext"] == reason
    assert projected["internal_monologue"] == private_monologue
    assert projected["interaction_subtext"] != projected["internal_monologue"]


def test_rag_memory_authority_maps_self_guidance_to_conditional_context() -> None:
    """Canonical projected self-guidance gets a typed cognition row."""

    evidence = cognition_node._rag_evidence(
        {
            "memory_evidence": [{
                "memory_unit_id": "guidance-unit-1",
                "memory_type": "defense_rule",
                "content": "A certified character guidance rule.",
                "source_kind": "reflection_inferred",
                "scope_type": "global",
                "authority": "reflection_promoted",
                "status": "active",
                "privacy_review": {
                    "global_applicability": "global",
                    "target_specific_meaning_removed": True,
                    "affects_identity_or_boundaries": False,
                    "private_detail_risk": "low",
                    "user_details_removed": True,
                    "boundary_assessment": "deidentified global meaning",
                    "reviewer": "automated_llm",
                },
            }],
        },
        "2026-06-08T00:00:00Z",
        current_user_id="user-1",
    )

    assert len(evidence) == 1
    row = evidence[0]
    assert row["authority"] == "conditional_character_guidance"
    assert row["memory_scope"] == "shared_character_or_world"
    assert row["evidence_ref"]["source_id"] == (
        "promoted-memory:self_guidance:guidance-unit-1"
    )
    assert row["memory_metadata"]["stable_id"] == "guidance-unit-1"
    assert row["memory_metadata"]["memory_type"] == "defense_rule"
    assert row["memory_metadata"]["status"] == "active"
    assert row["memory_metadata"]["privacy_review"]["reviewer"] == (
        "automated_llm"
    )


def test_rag_memory_rejects_unmarked_promoted_memory_source() -> None:
    """Promoted-memory evidence requires its canonical typed source id."""

    row = {
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "promoted_memory",
            "source_id": "memory:unmarked",
            "occurred_at": "2026-06-08T00:00:00Z",
            "semantic_summary": "unmarked memory",
        },
        "semantic_text": "unmarked memory",
        "visible_to": ["q:event_agency"],
        "authority": "character_world_context",
        "memory_scope": "shared_character_or_world",
        "memory_metadata": {},
    }
    with pytest.raises(CognitionContractError, match="canonical"):
        _validate_evidence_rows([row])


def test_promoted_reflection_context_rejects_incomplete_legacy_certificate() -> None:
    """Legacy reflection rows cannot enter cognition without certification."""

    evidence = cognition_node._promoted_reflection_evidence(
        {
            "promoted_lore": [{
                "memory_name": "Legacy memory",
                "content": "A bounded legacy memory without the current scope certificate.",
                "memory_unit_id": "legacy-fact-unit",
                "memory_type": "fact",
                "source_kind": "reflection_inferred",
                "source_global_user_id": "",
                "authority": "reflection_promoted",
                "status": "active",
                "scope_type": "global",
                "privacy_review": {
                    "private_detail_risk": "low",
                    "user_details_removed": True,
                    "boundary_assessment": "Generic deidentified background meaning.",
                    "reviewer": "automated_llm",
                },
                "updated_at": "2026-06-08T00:00:00Z",
            }],
        },
        "2026-06-08T00:00:00Z",
    )

    assert evidence == []


def test_promoted_reflection_context_maps_certified_rows_through_typed_memory_contract() -> None:
    """Certified reflection context uses the canonical promoted-memory shape."""

    privacy_review = {
        "global_applicability": "global",
        "target_specific_meaning_removed": True,
        "affects_identity_or_boundaries": False,
        "private_detail_risk": "low",
        "user_details_removed": True,
        "boundary_assessment": "The meaning is deidentified and global.",
        "reviewer": "automated_llm",
    }
    evidence = cognition_node._promoted_reflection_evidence(
        {
            "promoted_self_guidance": [{
                "memory_name": "Certified guidance",
                "content": "A certified character guidance rule.",
                "memory_unit_id": "reflection-guidance-1",
                "memory_type": "defense_rule",
                "source_kind": "reflection_inferred",
                "source_global_user_id": "",
                "authority": "reflection_promoted",
                "status": "active",
                "scope_type": "global",
                "privacy_review": privacy_review,
                "updated_at": "2026-06-08T00:00:00Z",
            }],
        },
        "2026-06-08T00:00:00Z",
    )

    assert len(evidence) == 1
    row = evidence[0]
    assert row["evidence_ref"]["source_kind"] == "promoted_memory"
    assert row["evidence_ref"]["source_id"] == (
        "promoted-memory:self_guidance:reflection-guidance-1"
    )
    assert row["authority"] == "conditional_character_guidance"
    assert row["memory_scope"] == "shared_character_or_world"
    assert row["memory_metadata"] == {
        "stable_id": "reflection-guidance-1",
        "memory_type": "defense_rule",
        "source_kind": "reflection_inferred",
        "authority": "reflection_promoted",
        "status": "active",
        "scope_type": "global",
        "privacy_review": privacy_review,
    }
    row["evidence_handle"] = "e1"
    _validate_evidence_rows(evidence)


@pytest.mark.asyncio
async def test_persona_character_commit_reads_canonical_state_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = build_character_production_state(updated_at="2026-01-01T00:00:00Z")
    replacement = build_character_production_state(updated_at="2026-01-01T00:00:00.000001Z")
    captured: dict[str, object] = {}

    async def replace_character(*, expected_updated_at: str, replacement: dict) -> bool:
        captured["expected"] = expected_updated_at
        captured["replacement"] = replacement
        return True

    monkeypatch.setattr(
        cognition_node,
        "compare_and_replace_character_cognition_state",
        replace_character,
    )
    output = {
        "schema_version": "cognition_output.v3",
        "state_projection": {
            "state_scope": "character",
            "owner_key": "character",
            "expected_previous_state": replacement,
            "original_persisted_state": original,
            "replacement_state": replacement,
        },
    }
    await cognition_node.commit_cognition_output(
        output,
        expected_character_updated_at=original["updated_at"],
    )
    assert captured["expected"] == original["updated_at"]
    assert captured["replacement"] == replacement
