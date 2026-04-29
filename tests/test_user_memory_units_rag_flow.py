"""Tests for RAG-owned user memory retrieval and consolidator reuse."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.db.schemas import UserMemoryUnitType
from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator_memory_units as memory_units_module
from kazusa_ai_chatbot.rag import user_memory_unit_retrieval as retrieval_module


def _unit(unit_id: str, fact: str, *, unit_type: str = UserMemoryUnitType.OBJECTIVE_FACT) -> dict:
    return {
        "unit_id": unit_id,
        "unit_type": unit_type,
        "fact": fact,
        "subjective_appraisal": f"Kazusa appraisal for {fact}",
        "relationship_signal": f"Kazusa signal for {fact}",
        "updated_at": "2026-04-29T00:00:00+12:00",
    }


@pytest.mark.asyncio
async def test_build_user_memory_context_bundle_merges_semantic_and_recent(monkeypatch) -> None:
    calls: dict[str, object] = {}

    async def _search_user_memory_units_by_vector(global_user_id, embedding, **kwargs):
        calls["semantic"] = {
            "global_user_id": global_user_id,
            "embedding": embedding,
            "kwargs": kwargs,
        }
        return [_unit("semantic-1", "Semantic fact"), _unit("shared-1", "Shared fact")]

    async def _query_user_memory_units(global_user_id, **kwargs):
        calls["recent"] = {
            "global_user_id": global_user_id,
            "kwargs": kwargs,
        }
        return [_unit("shared-1", "Shared fact"), _unit("recent-1", "Recent fact")]

    monkeypatch.setattr(
        retrieval_module,
        "search_user_memory_units_by_vector",
        _search_user_memory_units_by_vector,
    )
    monkeypatch.setattr(retrieval_module, "query_user_memory_units", _query_user_memory_units)

    context, source_units = await retrieval_module.build_user_memory_context_bundle(
        "user-1",
        query_embedding=[0.1, 0.2, 0.3],
        include_semantic=True,
    )

    assert [unit["unit_id"] for unit in source_units] == ["semantic-1", "shared-1", "recent-1"]
    assert context["objective_facts"][0]["fact"] == "Semantic fact"
    assert calls["semantic"]["global_user_id"] == "user-1"
    assert calls["recent"]["global_user_id"] == "user-1"


@pytest.mark.asyncio
async def test_process_memory_unit_candidate_reuses_rag_surfaced_units(monkeypatch) -> None:
    captured: dict[str, object] = {}
    surfaced_unit = _unit("existing-1", "Existing fact")
    candidate = {
        "candidate_id": "candidate-1",
        "unit_type": UserMemoryUnitType.OBJECTIVE_FACT,
        "fact": "New fact",
        "subjective_appraisal": "Kazusa appraisal",
        "relationship_signal": "Kazusa signal",
        "evidence_refs": [],
    }

    async def _retrieve_memory_unit_merge_candidates(global_user_id, **kwargs):
        captured["global_user_id"] = global_user_id
        captured["surfaced_units"] = kwargs["surfaced_units"]
        return []

    async def _invoke_json(system_prompt, payload):
        captured["merge_payload"] = payload
        return {
            "candidate_id": "candidate-1",
            "decision": "create",
            "cluster_id": "",
            "reason": "no candidate matches",
        }

    async def _insert_user_memory_units(global_user_id, units, **kwargs):
        captured["inserted_units"] = units
        return [{"unit_id": "created-1"}]

    monkeypatch.setattr(
        memory_units_module,
        "retrieve_memory_unit_merge_candidates",
        _retrieve_memory_unit_merge_candidates,
    )
    monkeypatch.setattr(memory_units_module, "_invoke_json", _invoke_json)
    monkeypatch.setattr(memory_units_module, "insert_user_memory_units", _insert_user_memory_units)

    result = await memory_units_module.process_memory_unit_candidate(
        {
            "timestamp": "2026-04-29T00:00:00+12:00",
            "global_user_id": "user-1",
            "rag_result": {"user_memory_unit_candidates": [surfaced_unit]},
        },
        candidate,
    )

    assert captured["global_user_id"] == "user-1"
    assert captured["surfaced_units"] == [surfaced_unit]
    assert captured["merge_payload"]["candidate_clusters"] == []
    assert captured["inserted_units"] == [candidate]
    assert result["decision"] == "create"


def test_extractor_payload_includes_recent_history() -> None:
    state = {
        "timestamp": "2026-04-29T00:00:00+12:00",
        "global_user_id": "user-1",
        "user_name": "User",
        "decontexualized_input": "The memory lacks factual basis.",
        "final_dialog": ["That makes sense."],
        "internal_monologue": "Kazusa follows the architecture discussion.",
        "emotional_appraisal": "Kazusa feels focused.",
        "interaction_subtext": "The user is defining memory boundaries.",
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "chat_history_recent": [
            {
                "role": "user",
                "content": "historical summary and recent window overlap",
            }
        ],
        "rag_result": {
            "user_image": {"user_memory_context": _unit_context()},
        },
        "new_facts": [],
        "future_promises": [],
        "subjective_appraisals": [],
    }

    payload = memory_units_module._json_payload(state)

    assert payload["chat_history_recent"] == state["chat_history_recent"]


def _unit_context() -> dict:
    return {
        "stable_patterns": [],
        "recent_shifts": [],
        "objective_facts": [],
        "milestones": [],
        "active_commitments": [],
    }
