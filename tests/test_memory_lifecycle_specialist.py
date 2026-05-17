"""Tests for the dedicated memory lifecycle specialist handoff."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.action_spec.registry import (
    APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    SPEAK_CAPABILITY,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_memory_lifecycle as module


def _commitment(index: int) -> dict[str, object]:
    return {
        "unit_id": f"unit-{index:03d}",
        "fact": f"User promised item {index}.",
        "summary": f"Active commitment {index}.",
        "status": "active",
        "due_at": "2026-05-17T18:00:00+12:00",
        "due_state": "past_due",
    }


def _state_with_commitments(commitments: list[dict[str, object]]) -> dict[str, Any]:
    return {
        "storage_timestamp_utc": "2026-05-17T06:00:49+00:00",
        "user_input": "Here is the promised item.",
        "decontexualized_input": "Here is the promised item.",
        "logical_stance": "CONFIRM",
        "character_intent": "ACKNOWLEDGE_FULFILLMENT",
        "judgment_note": "The user may have fulfilled an active commitment.",
        "internal_monologue": "Check the active promise without guessing.",
        "rag_result": {
            "answer": "The user has active commitments.",
            "memory_evidence": [
                {
                    "summary": "The current turn may fulfill one commitment.",
                    "fact": "The user is handing over the promised item.",
                }
            ],
            "user_image": {
                "user_memory_context": {
                    "active_commitments": commitments,
                }
            },
        },
        "conversation_progress": {
            "current_thread": "Promise fulfillment check.",
        },
        "cognitive_episode": {
            "episode_id": "episode-001",
        },
    }


def _memory_lifecycle_route_spec() -> dict[str, object]:
    return {
        "schema_version": "action_spec.v1",
        "kind": MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
        "cognition_mode": "deliberative",
        "source_refs": [],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "cognitive_episode",
            "target_id": None,
            "owner": "memory_lifecycle_specialist",
            "scope": {"unit_type": "active_commitment"},
        },
        "params": {
            "review_kind": "active_commitment_lifecycle",
            "detail": "Review active commitment lifecycle.",
        },
        "urgency": "background",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "Lifecycle review may be needed.",
    }


def _speak_spec() -> dict[str, object]:
    return {
        "schema_version": "action_spec.v1",
        "kind": SPEAK_CAPABILITY,
        "visibility": "user_visible",
    }


class _FakeAsyncLLM:
    """Capture specialist messages and return one fixed JSON response."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.messages: list[object] = []

    async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
        self.messages = messages
        content = json.dumps(self.payload, ensure_ascii=False)
        return SimpleNamespace(content=content)


def test_prepare_review_aliases_first_12_commitments_without_prompt_ids() -> None:
    """Prompt payload should expose aliases only and keep overflow explicit."""

    state = _state_with_commitments([
        _commitment(index)
        for index in range(1, 14)
    ])

    prepared = module.prepare_memory_lifecycle_review(state)
    payload = prepared["prompt_payload"]
    aliases = [
        row["target_alias"]
        for row in payload["active_commitments"]
    ]
    serialized_payload = json.dumps(payload, ensure_ascii=False)

    assert prepared["visible_alias_count"] == 12
    assert prepared["omitted_alias_count"] == 1
    assert aliases == [f"commitment_{index}" for index in range(1, 13)]
    assert len(prepared["alias_bindings"]) == 12
    assert prepared["alias_bindings"][0]["unit_id"] == "unit-001"
    assert prepared["alias_bindings"][-1]["unit_id"] == "unit-012"
    assert "unit_id" not in serialized_payload
    assert "unit-001" not in serialized_payload
    assert "unit-013" not in serialized_payload


def test_prepare_review_falls_back_when_raw_candidates_are_unrelated() -> None:
    """Unrelated raw candidates should not hide active commitment bindings."""

    state = _state_with_commitments([_commitment(1)])
    state["rag_result"]["user_memory_unit_candidates"] = [
        {
            "unit_id": "unit-preference-001",
            "unit_type": "preference",
            "fact": "The user likes tea.",
            "status": "active",
        }
    ]

    prepared = module.prepare_memory_lifecycle_review(state)

    assert len(prepared["alias_bindings"]) == 1
    assert prepared["alias_bindings"][0]["unit_id"] == "unit-001"


def test_no_change_output_materializes_no_actions_and_prompt_safe_context() -> None:
    """Explicit no-change output should become empty lifecycle work."""

    state = _state_with_commitments([_commitment(1)])
    prepared = module.prepare_memory_lifecycle_review(state)
    normalized = module.normalize_memory_lifecycle_output(
        {
            "decision": "no_lifecycle_change",
            "content_anchor_roles": [
                {
                    "role": "keep_waiting",
                    "anchor": "No promise has been fulfilled yet.",
                }
            ],
        },
        prepared["alias_bindings"],
    )

    materialized = module.materialize_memory_lifecycle_actions(
        normalized,
        prepared["alias_bindings"],
        visible_alias_count=prepared["visible_alias_count"],
        omitted_alias_count=prepared["omitted_alias_count"],
    )
    serialized_context = json.dumps(
        materialized["memory_lifecycle_context"],
        ensure_ascii=False,
    )

    assert normalized["decision"] == "no_lifecycle_change"
    assert normalized["lifecycle_decisions"] == []
    assert normalized["errors"] == []
    assert materialized["action_specs"] == []
    assert materialized["memory_lifecycle_context"]["decision"] == (
        "no_lifecycle_change"
    )
    assert "unit_id" not in serialized_context
    assert "unit-001" not in serialized_context


def test_valid_alias_materializes_apply_action_and_prompt_safe_context() -> None:
    """A specialist alias decision should resolve to one trusted DB action."""

    state = _state_with_commitments([_commitment(1), _commitment(2)])
    prepared = module.prepare_memory_lifecycle_review(state)
    normalized = module.normalize_memory_lifecycle_output(
        {
            "decision": "lifecycle_change",
            "lifecycle_decisions": [
                {
                    "target_alias": "commitment_2",
                    "decision": "fulfilled",
                    "role": "acknowledge_fulfillment",
                    "evidence_anchor": "The user delivered item 2.",
                }
            ],
            "content_anchor_roles": [
                {
                    "role": "avoid_reopening",
                    "anchor": "Do not reopen item 2 as unfulfilled.",
                }
            ],
        },
        prepared["alias_bindings"],
    )

    materialized = module.materialize_memory_lifecycle_actions(
        normalized,
        prepared["alias_bindings"],
        visible_alias_count=prepared["visible_alias_count"],
        omitted_alias_count=prepared["omitted_alias_count"],
    )
    action_spec = materialized["action_specs"][0]
    serialized_context = json.dumps(
        materialized["memory_lifecycle_context"],
        ensure_ascii=False,
    )

    assert normalized["errors"] == []
    assert action_spec["kind"] == APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY
    assert action_spec["target"]["target_id"] == "unit-002"
    assert action_spec["params"]["unit_id"] == "unit-002"
    assert action_spec["params"]["lifecycle_decision"] == "fulfilled"
    assert materialized["memory_lifecycle_context"]["decision"] == (
        "lifecycle_change"
    )
    assert "unit_id" not in serialized_context
    assert "unit-002" not in serialized_context


def test_materialization_caps_valid_lifecycle_updates_to_three() -> None:
    """A single turn should not batch-close more than three commitments."""

    state = _state_with_commitments([
        _commitment(index)
        for index in range(1, 5)
    ])
    prepared = module.prepare_memory_lifecycle_review(state)
    normalized = module.normalize_memory_lifecycle_output(
        {
            "decision": "lifecycle_change",
            "lifecycle_decisions": [
                {
                    "target_alias": f"commitment_{index}",
                    "decision": "fulfilled",
                    "role": "acknowledge_fulfillment",
                    "evidence_anchor": f"Item {index} was delivered.",
                }
                for index in range(1, 5)
            ],
            "content_anchor_roles": [],
        },
        prepared["alias_bindings"],
    )

    materialized = module.materialize_memory_lifecycle_actions(
        normalized,
        prepared["alias_bindings"],
        visible_alias_count=prepared["visible_alias_count"],
        omitted_alias_count=prepared["omitted_alias_count"],
    )

    assert len(materialized["action_specs"]) == 3
    assert [
        action_spec["params"]["unit_id"]
        for action_spec in materialized["action_specs"]
    ] == ["unit-001", "unit-002", "unit-003"]
    assert materialized["memory_lifecycle_context"]["warnings"]


@pytest.mark.asyncio
async def test_handler_consumes_route_and_materializes_apply_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The specialist handler should replace route intents with apply actions."""

    state = _state_with_commitments([_commitment(1)])
    state["action_specs"] = [
        _speak_spec(),
        _memory_lifecycle_route_spec(),
    ]
    fake_llm = _FakeAsyncLLM({
        "decision": "lifecycle_change",
        "lifecycle_decisions": [
            {
                "target_alias": "commitment_1",
                "decision": "fulfilled",
                "role": "acknowledge_fulfillment",
                "evidence_anchor": "The user delivered the promised item.",
            }
        ],
        "content_anchor_roles": [
            {
                "role": "avoid_reopening",
                "anchor": "Do not reopen the fulfilled commitment.",
            }
        ],
    })
    monkeypatch.setattr(module, "_memory_lifecycle_specialist_llm", fake_llm)

    result = await module.call_memory_lifecycle_update_handler(state)
    action_specs = result["action_specs"]
    prompt_payload = json.loads(fake_llm.messages[1].content)
    serialized_prompt = json.dumps(prompt_payload, ensure_ascii=False)

    assert [
        action_spec["kind"]
        for action_spec in action_specs
    ] == [
        SPEAK_CAPABILITY,
        APPLY_MEMORY_LIFECYCLE_UPDATE_CAPABILITY,
    ]
    assert action_specs[1]["params"]["unit_id"] == "unit-001"
    assert result["memory_lifecycle_context"]["content_anchor_roles"] == [
        {
            "role": "avoid_reopening",
            "anchor": "Do not reopen the fulfilled commitment.",
        }
    ]
    assert "unit_id" not in serialized_prompt
    assert "unit-001" not in serialized_prompt
