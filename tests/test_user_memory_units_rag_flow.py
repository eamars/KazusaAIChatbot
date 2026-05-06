"""Tests for RAG-owned user memory retrieval and consolidator reuse."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.db.schemas import UserMemoryUnitType
from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator_memory_units as memory_units_module
from kazusa_ai_chatbot.rag import user_memory_unit_retrieval as retrieval_module
from kazusa_ai_chatbot.time_context import build_character_time_context


class _DummyResponse:
    """Minimal async LLM response wrapper for memory-unit tests."""

    def __init__(self, content: str) -> None:
        self.content = content


class _StaticAsyncLLM:
    """Capture one LLM call and return a fixed JSON payload."""

    def __init__(self, response_payload: dict) -> None:
        self.messages = []
        self._response_payload = response_payload

    async def ainvoke(self, messages):
        self.messages = messages
        return_value = _DummyResponse(
            json.dumps(self._response_payload, ensure_ascii=False),
        )
        return return_value


def _unit(unit_id: str, fact: str, *, unit_type: str = UserMemoryUnitType.OBJECTIVE_FACT) -> dict:
    return {
        "unit_id": unit_id,
        "unit_type": unit_type,
        "fact": fact,
        "subjective_appraisal": f"Kazusa appraisal for {fact}",
        "relationship_signal": f"Kazusa signal for {fact}",
        "updated_at": "2026-04-29T00:00:00+12:00",
    }


def test_valid_candidates_drops_incomplete_memory_unit_rows(caplog) -> None:
    caplog.set_level("WARNING", logger=memory_units_module.__name__)
    result = {
        "memory_units": [
            {
                "candidate_id": "missing-appraisal",
                "unit_type": UserMemoryUnitType.OBJECTIVE_FACT,
                "fact": "User is testing malformed memory output.",
                "relationship_signal": "Use this only if structurally valid.",
                "evidence_refs": [],
            },
            {
                "candidate_id": "valid-candidate",
                "unit_type": UserMemoryUnitType.OBJECTIVE_FACT,
                "fact": "User asked the extractor to keep valid rows.",
                "subjective_appraisal": "The character reads this as schema discipline.",
                "relationship_signal": "Future memory extraction should keep all required fields.",
                "evidence_refs": [],
            },
        ],
    }

    candidates = memory_units_module._valid_candidates(result)

    assert [candidate["candidate_id"] for candidate in candidates] == ["valid-candidate"]
    assert "missing-appraisal" in caplog.text
    assert "subjective_appraisal" in caplog.text


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

    async def _insert_user_memory_units(global_user_id, units, **kwargs):
        captured["inserted_units"] = units
        return [{"unit_id": "created-1"}]

    merge_judge_llm = _StaticAsyncLLM({
        "candidate_id": "candidate-1",
        "decision": "create",
        "cluster_id": "",
        "reason": "no candidate matches",
    })

    monkeypatch.setattr(
        memory_units_module,
        "retrieve_memory_unit_merge_candidates",
        _retrieve_memory_unit_merge_candidates,
    )
    monkeypatch.setattr(memory_units_module, "_merge_judge_llm", merge_judge_llm)
    monkeypatch.setattr(memory_units_module, "insert_user_memory_units", _insert_user_memory_units)

    result = await memory_units_module.process_memory_unit_candidate(
        {
            "timestamp": "2026-04-29T00:00:00+12:00",
            "global_user_id": "user-1",
            "character_profile": {"name": "杏山千纱 (Kyōyama Kazusa)"},
            "rag_result": {"user_memory_unit_candidates": [surfaced_unit]},
        },
        candidate,
    )

    assert captured["global_user_id"] == "user-1"
    assert captured["surfaced_units"] == [surfaced_unit]
    merge_payload = json.loads(merge_judge_llm.messages[1].content)
    assert merge_payload["candidate_clusters"] == []
    assert captured["inserted_units"] == [candidate]
    assert result["decision"] == "create"


@pytest.mark.asyncio
async def test_process_memory_unit_candidate_merges_memory_evidence_surfaced_scoped_unit(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    surfaced_unit = {
        **_unit("existing-scoped-1", "冰淇淋摊老板是千纱的初中学姐。"),
        "source_system": "user_memory_units",
        "scope_type": "user_continuity",
        "scope_global_user_id": "user-1",
        "authority": "scoped_continuity",
        "truth_status": "character_lore_or_interaction_continuity",
        "origin": "consolidated_interaction",
    }
    candidate = {
        "candidate_id": "candidate-scoped-1",
        "unit_type": UserMemoryUnitType.OBJECTIVE_FACT,
        "fact": "冰淇淋摊老板是千纱的初中学姐，还会给双倍抹茶酱。",
        "subjective_appraisal": "Kazusa now treats this as reinforced continuity.",
        "relationship_signal": "Keep this scoped lore available for the same user.",
        "evidence_refs": [],
    }

    async def _retrieve_memory_unit_merge_candidates(global_user_id, **kwargs):
        captured["global_user_id"] = global_user_id
        captured["surfaced_units"] = kwargs["surfaced_units"]
        return [surfaced_unit]

    async def _update_user_memory_unit_semantics(unit_id, semantics, **kwargs):
        captured["updated_unit_id"] = unit_id
        captured["updated_semantics"] = semantics
        captured["merge_history"] = kwargs["merge_history_entry"]

    async def _insert_user_memory_units(*args, **kwargs):
        raise AssertionError("duplicate insert should not run when a surfaced scoped unit is merged")

    merge_judge_llm = _StaticAsyncLLM({
        "candidate_id": "candidate-scoped-1",
        "decision": "merge",
        "cluster_id": "existing-scoped-1",
        "reason": "same scoped continuity with one new detail",
    })
    rewrite_llm = _StaticAsyncLLM({
        "candidate_id": "candidate-scoped-1",
        "cluster_id": "existing-scoped-1",
        "fact": "冰淇淋摊老板是千纱的初中学姐，还会给双倍抹茶酱。",
        "subjective_appraisal": "Kazusa now treats this as reinforced continuity.",
        "relationship_signal": "Keep this scoped lore available for the same user.",
    })

    monkeypatch.setattr(
        memory_units_module,
        "retrieve_memory_unit_merge_candidates",
        _retrieve_memory_unit_merge_candidates,
    )
    monkeypatch.setattr(memory_units_module, "_merge_judge_llm", merge_judge_llm)
    monkeypatch.setattr(memory_units_module, "_rewrite_llm", rewrite_llm)
    monkeypatch.setattr(
        memory_units_module,
        "update_user_memory_unit_semantics",
        _update_user_memory_unit_semantics,
    )
    monkeypatch.setattr(memory_units_module, "insert_user_memory_units", _insert_user_memory_units)

    result = await memory_units_module.process_memory_unit_candidate(
        {
            "timestamp": "2026-04-29T00:00:00+12:00",
            "global_user_id": "user-1",
            "character_profile": {"name": "杏山千纱 (Kyōyama Kazusa)"},
            "rag_result": {"user_memory_unit_candidates": [surfaced_unit]},
        },
        candidate,
    )

    assert captured["global_user_id"] == "user-1"
    assert captured["surfaced_units"] == [surfaced_unit]
    assert captured["updated_unit_id"] == "existing-scoped-1"
    assert captured["updated_semantics"] == {
        "fact": "冰淇淋摊老板是千纱的初中学姐，还会给双倍抹茶酱。",
        "subjective_appraisal": "Kazusa now treats this as reinforced continuity.",
        "relationship_signal": "Keep this scoped lore available for the same user.",
    }
    assert captured["merge_history"]["candidate_id"] == "candidate-scoped-1"
    assert result["decision"] == "merge"
    assert result["unit_id"] == "existing-scoped-1"


@pytest.mark.asyncio
async def test_process_memory_unit_candidate_normalizes_merge_candidate_id(monkeypatch) -> None:
    captured: dict[str, object] = {}
    existing_unit = _unit("existing-1", "Existing fact")
    candidate = {
        "candidate_id": "expected-candidate",
        "unit_type": UserMemoryUnitType.OBJECTIVE_FACT,
        "fact": "New fact",
        "subjective_appraisal": "Kazusa appraisal",
        "relationship_signal": "Kazusa signal",
        "evidence_refs": [],
    }

    async def _retrieve_memory_unit_merge_candidates(global_user_id, **kwargs):
        return [existing_unit]

    async def _update_user_memory_unit_semantics(unit_id, semantics, **kwargs):
        captured["updated_unit_id"] = unit_id
        captured["merge_history"] = kwargs["merge_history_entry"]
        captured["semantics"] = semantics

    merge_judge_llm = _StaticAsyncLLM({
        "candidate_id": "unknown-candidate-from-llm",
        "decision": "merge",
        "cluster_id": "existing-1",
        "reason": "same memory with extra detail",
    })
    rewrite_llm = _StaticAsyncLLM({
        "candidate_id": "expected-candidate",
        "cluster_id": "existing-1",
        "fact": "Updated fact",
        "subjective_appraisal": "Updated appraisal",
        "relationship_signal": "Updated signal",
    })

    monkeypatch.setattr(
        memory_units_module,
        "retrieve_memory_unit_merge_candidates",
        _retrieve_memory_unit_merge_candidates,
    )
    monkeypatch.setattr(memory_units_module, "_merge_judge_llm", merge_judge_llm)
    monkeypatch.setattr(memory_units_module, "_rewrite_llm", rewrite_llm)
    monkeypatch.setattr(
        memory_units_module,
        "update_user_memory_unit_semantics",
        _update_user_memory_unit_semantics,
    )

    result = await memory_units_module.process_memory_unit_candidate(
        {
            "timestamp": "2026-04-29T00:00:00+12:00",
            "global_user_id": "user-1",
            "character_profile": {"name": "杏山千纱 (Kyōyama Kazusa)"},
            "rag_result": {"user_memory_unit_candidates": [existing_unit]},
        },
        candidate,
    )

    assert result["candidate_id"] == "expected-candidate"
    assert result["unit_id"] == "existing-1"
    assert result["decision"] == "merge"
    assert captured["updated_unit_id"] == "existing-1"
    assert captured["merge_history"]["candidate_id"] == "expected-candidate"


@pytest.mark.asyncio
async def test_process_memory_unit_candidate_uses_validated_merge_cluster_when_rewrite_ids_drift(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    existing_unit = _unit("existing-1", "Existing fact")
    candidate = {
        "candidate_id": "expected-candidate",
        "unit_type": UserMemoryUnitType.OBJECTIVE_FACT,
        "fact": "New fact",
        "subjective_appraisal": "Kazusa appraisal",
        "relationship_signal": "Kazusa signal",
        "evidence_refs": [],
    }

    async def _retrieve_memory_unit_merge_candidates(global_user_id, **kwargs):
        return [existing_unit]

    async def _update_user_memory_unit_semantics(unit_id, semantics, **kwargs):
        captured["updated_unit_id"] = unit_id
        captured["merge_history"] = kwargs["merge_history_entry"]
        captured["semantics"] = semantics

    merge_judge_llm = _StaticAsyncLLM({
        "candidate_id": "expected-candidate",
        "decision": "merge",
        "cluster_id": "existing-1",
        "reason": "same memory with extra detail",
    })
    rewrite_llm = _StaticAsyncLLM({
        "candidate_id": "expected-candidate",
        "cluster_id": "wrong-cluster-from-rewrite",
        "fact": "Updated fact",
        "subjective_appraisal": "Updated appraisal",
        "relationship_signal": "Updated signal",
    })

    monkeypatch.setattr(
        memory_units_module,
        "retrieve_memory_unit_merge_candidates",
        _retrieve_memory_unit_merge_candidates,
    )
    monkeypatch.setattr(memory_units_module, "_merge_judge_llm", merge_judge_llm)
    monkeypatch.setattr(memory_units_module, "_rewrite_llm", rewrite_llm)
    monkeypatch.setattr(
        memory_units_module,
        "update_user_memory_unit_semantics",
        _update_user_memory_unit_semantics,
    )

    result = await memory_units_module.process_memory_unit_candidate(
        {
            "timestamp": "2026-04-29T00:00:00+12:00",
            "global_user_id": "user-1",
            "character_profile": {"name": "杏山千纱 (Kyōyama Kazusa)"},
            "rag_result": {"user_memory_unit_candidates": [existing_unit]},
        },
        candidate,
    )

    rewrite_payload = json.loads(rewrite_llm.messages[1].content)

    assert rewrite_payload["existing_unit_id"] == "existing-1"
    assert rewrite_payload["decision"]["cluster_id"] == "existing-1"
    assert result["candidate_id"] == "expected-candidate"
    assert result["unit_id"] == "existing-1"
    assert result["decision"] == "merge"
    assert captured["updated_unit_id"] == "existing-1"
    assert captured["merge_history"]["candidate_id"] == "expected-candidate"
    assert captured["semantics"] == {
        "fact": "Updated fact",
        "subjective_appraisal": "Updated appraisal",
        "relationship_signal": "Updated signal",
    }


@pytest.mark.asyncio
async def test_stability_judge_receives_count_session_and_recent_examples(monkeypatch) -> None:
    captured: dict[str, object] = {"llm_payloads": []}
    existing_unit = {
        "unit_id": "existing-pattern-1",
        "unit_type": UserMemoryUnitType.RECENT_SHIFT,
        "fact": "The user has repeatedly corrected memory architecture boundaries.",
        "subjective_appraisal": "Kazusa reads this as careful system stewardship.",
        "relationship_signal": "Kazusa should keep memory records factual and inspectable.",
        "count": 3,
        "source_refs": [
            {
                "source": "chat",
                "timestamp": "2026-04-28T11:00:00+12:00",
                "message_id": "m-1",
            },
            {
                "source": "chat",
                "timestamp": "2026-04-29T12:00:00+12:00",
                "message_id": "m-2",
            },
        ],
        "updated_at": "2026-04-29T12:00:00+12:00",
        "last_seen_at": "2026-04-29T12:00:00+12:00",
    }
    candidate = {
        "candidate_id": "candidate-pattern-1",
        "unit_type": UserMemoryUnitType.RECENT_SHIFT,
        "fact": "The user again asked for factual memory boundaries.",
        "subjective_appraisal": "Kazusa sees this as repeated architecture guidance.",
        "relationship_signal": "Future consolidation should preserve the boundary.",
        "evidence_refs": [
            {
                "source": "chat",
                "timestamp": "2026-04-30T12:00:00+12:00",
                "message_id": "m-3",
            }
        ],
    }

    async def _retrieve_memory_unit_merge_candidates(global_user_id, **kwargs):
        captured["retrieval_user_id"] = global_user_id
        captured["surfaced_units"] = kwargs["surfaced_units"]
        return [existing_unit]

    async def _update_user_memory_unit_semantics(unit_id, semantics, **kwargs):
        captured["updated_unit_id"] = unit_id
        captured["updated_semantics"] = semantics
        captured["merge_history"] = kwargs["merge_history_entry"]

    async def _update_user_memory_unit_window(unit_id, **kwargs):
        captured["window_unit_id"] = unit_id
        captured["window"] = kwargs["window"]

    merge_judge_llm = _StaticAsyncLLM({
        "candidate_id": "candidate-pattern-1",
        "decision": "evolve",
        "cluster_id": "existing-pattern-1",
        "reason": "same boundary pattern with updated evidence",
    })
    rewrite_llm = _StaticAsyncLLM({
        "candidate_id": "candidate-pattern-1",
        "cluster_id": "existing-pattern-1",
        "fact": "The user repeatedly corrects memory architecture boundaries.",
        "subjective_appraisal": "Kazusa treats this as durable system guidance.",
        "relationship_signal": "Keep future memory records factual and inspectable.",
    })
    stability_llm = _StaticAsyncLLM({
        "unit_id": "existing-pattern-1",
        "window": "stable",
        "reason": "multiple sessions indicate a durable pattern",
    })

    monkeypatch.setattr(
        memory_units_module,
        "retrieve_memory_unit_merge_candidates",
        _retrieve_memory_unit_merge_candidates,
    )
    monkeypatch.setattr(memory_units_module, "_merge_judge_llm", merge_judge_llm)
    monkeypatch.setattr(memory_units_module, "_rewrite_llm", rewrite_llm)
    monkeypatch.setattr(memory_units_module, "_stability_llm", stability_llm)
    monkeypatch.setattr(
        memory_units_module,
        "update_user_memory_unit_semantics",
        _update_user_memory_unit_semantics,
    )
    monkeypatch.setattr(
        memory_units_module,
        "update_user_memory_unit_window",
        _update_user_memory_unit_window,
    )

    result = await memory_units_module.process_memory_unit_candidate(
        {
            "timestamp": "2026-04-30T12:00:00+12:00",
            "time_context": build_character_time_context("2026-04-30T12:00:00+12:00"),
            "global_user_id": "user-1",
            "character_profile": {"name": "杏山千纱 (Kyōyama Kazusa)"},
            "rag_result": {"user_memory_unit_candidates": [existing_unit]},
        },
        candidate,
    )

    stability_payload = json.loads(stability_llm.messages[1].content)
    stability_evidence = stability_payload["stability_evidence"]
    session_spread = stability_evidence["session_spread"]
    stability_payload_json = json.dumps(stability_payload, ensure_ascii=False)

    assert result["decision"] == "evolve"
    assert result["stability"]["window"] == "stable"
    assert stability_evidence["occurrence_count"] == 4
    assert stability_evidence["occurrence_count_label"] == "several_observations"
    assert stability_evidence["existing_unit_count"] == 3
    assert stability_evidence["new_evidence_ref_count"] == 1
    assert session_spread["spread_label"] == "multiple_days_or_sessions"
    assert session_spread["distinct_day_count"] == 3
    assert len(stability_evidence["recent_examples"]) == 2
    assert "+12:00" not in stability_payload_json
    assert "2026-04-30 12:00" in stability_payload_json
    assert captured["window"] == "stable"


@pytest.mark.asyncio
async def test_update_user_memory_units_skips_bad_candidate_and_continues(
    monkeypatch,
    caplog,
) -> None:
    first_candidate = {
        "candidate_id": "bad-candidate",
        "unit_type": UserMemoryUnitType.OBJECTIVE_FACT,
        "fact": "Bad candidate",
        "subjective_appraisal": "Bad appraisal",
        "relationship_signal": "Bad signal",
        "evidence_refs": [],
    }
    second_candidate = {
        "candidate_id": "good-candidate",
        "unit_type": UserMemoryUnitType.OBJECTIVE_FACT,
        "fact": "Good candidate",
        "subjective_appraisal": "Good appraisal",
        "relationship_signal": "Good signal",
        "evidence_refs": [],
    }

    async def _extract_memory_unit_candidates(state):
        return [first_candidate, second_candidate]

    async def _process_memory_unit_candidate(state, candidate):
        if candidate["candidate_id"] == "bad-candidate":
            raise ValueError("invalid LLM structure")
        return {
            "candidate_id": candidate["candidate_id"],
            "unit_id": "created-good",
            "decision": "create",
            "stability": {},
        }

    monkeypatch.setattr(
        memory_units_module,
        "extract_memory_unit_candidates",
        _extract_memory_unit_candidates,
    )
    monkeypatch.setattr(
        memory_units_module,
        "process_memory_unit_candidate",
        _process_memory_unit_candidate,
    )
    caplog.set_level("ERROR", logger=memory_units_module.__name__)

    results = await memory_units_module.update_user_memory_units_from_state({
        "global_user_id": "user-1",
    })

    assert results == [{
        "candidate_id": "good-candidate",
        "unit_id": "created-good",
        "decision": "create",
        "stability": {},
    }]
    assert "bad-candidate" in caplog.text
    assert "invalid LLM structure" in caplog.text


@pytest.mark.asyncio
async def test_update_user_memory_units_drops_bad_extractor_output(
    monkeypatch,
    caplog,
) -> None:
    async def _extract_memory_unit_candidates(state):
        raise ValueError("extractor returned malformed JSON")

    monkeypatch.setattr(
        memory_units_module,
        "extract_memory_unit_candidates",
        _extract_memory_unit_candidates,
    )
    caplog.set_level("ERROR", logger=memory_units_module.__name__)

    results = await memory_units_module.update_user_memory_units_from_state({
        "global_user_id": "user-1",
    })

    assert results == []
    assert "extractor returned malformed JSON" in caplog.text


def test_extractor_payload_includes_recent_history() -> None:
    state = {
        "timestamp": "2026-04-29T00:00:00+12:00",
        "time_context": build_character_time_context("2026-04-29T00:00:00+12:00"),
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
                "timestamp": "2026-04-28T12:01:00+00:00",
            }
        ],
        "rag_result": {
            "user_image": {
                "user_memory_context": {
                    **_unit_context(),
                    "active_commitments": [
                        {
                            "fact": "User asked for a reminder.",
                            "updated_at": "2026-04-28T12:02:00+00:00",
                        },
                    ],
                },
            },
        },
        "new_facts": [],
        "future_promises": [],
        "subjective_appraisals": [],
    }

    payload = memory_units_module._json_payload(state)

    assert payload["chat_history_recent"][0]["timestamp"] == "2026-04-29 00:01"
    commitments = payload["rag_user_memory_context"]["active_commitments"]
    assert commitments[0]["updated_at"] == "2026-04-29 00:02"


def _unit_context() -> dict:
    return {
        "stable_patterns": [],
        "recent_shifts": [],
        "objective_facts": [],
        "milestones": [],
        "active_commitments": [],
    }
