"""Focused commit and continuation-lineage checks for canonical cognition."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition_node
from tests.unit.cognition_core_v3.test_handleless_contract import _input


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
