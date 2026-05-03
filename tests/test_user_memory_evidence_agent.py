from __future__ import annotations

import pytest

from kazusa_ai_chatbot.rag import user_memory_evidence_agent as module
from kazusa_ai_chatbot.rag.user_memory_evidence_agent import UserMemoryEvidenceAgent


def _context(**overrides: object) -> dict[str, object]:
    context: dict[str, object] = {
        "global_user_id": "user-1",
        "user_name": "Tester",
        "known_facts": [],
    }
    context.update(overrides)
    return context


def _memory_row(unit_id: str, fact: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "unit_id": unit_id,
        "unit_type": "objective_fact",
        "fact": fact,
        "subjective_appraisal": f"Kazusa appraisal for {fact}",
        "relationship_signal": f"Kazusa signal for {fact}",
        "updated_at": "2026-05-03T00:00:00+00:00",
    }
    row.update(overrides)
    return row


@pytest.mark.asyncio
async def test_user_memory_evidence_requires_global_user_id(monkeypatch) -> None:
    agent = UserMemoryEvidenceAgent()

    async def _fail_embedding(text: str) -> list[float]:
        raise AssertionError(f"embedding should not run: {text}")

    async def _fail_vector(*args, **kwargs):
        raise AssertionError("vector search should not run")

    async def _fail_keyword(*args, **kwargs):
        raise AssertionError("keyword search should not run")

    async def _fail_recent(*args, **kwargs):
        raise AssertionError("recent query should not run")

    monkeypatch.setattr(module, "get_text_embedding", _fail_embedding)
    monkeypatch.setattr(module, "search_user_memory_units_by_vector", _fail_vector)
    monkeypatch.setattr(module, "search_user_memory_units_by_keyword", _fail_keyword)
    monkeypatch.setattr(module, "query_user_memory_units", _fail_recent)

    result = await agent.run(
        'Memory-evidence: retrieve durable evidence about the exact term "学姐" in the current user\'s continuity',
        {"known_facts": []},
    )

    assert result["resolved"] is False
    assert result["attempts"] == 1
    assert result["cache"] == {
        "enabled": False,
        "hit": False,
        "reason": "scoped_user_memory_uncached",
    }
    assert result["result"] == {
        "selected_summary": "",
        "memory_rows": [],
        "source_system": "user_memory_units",
        "scope_type": "user_continuity",
        "scope_global_user_id": "",
        "missing_context": ["global_user_id"],
    }


@pytest.mark.asyncio
async def test_user_memory_evidence_exact_cjk_term_uses_scoped_keyword(monkeypatch) -> None:
    agent = UserMemoryEvidenceAgent()
    calls: dict[str, object] = {}

    async def _embedding(text: str) -> list[float]:
        raise AssertionError(f"embedding should not run for literal lookup: {text}")

    async def _vector(*args, **kwargs):
        raise AssertionError("vector search should not run for literal lookup")

    async def _keyword(global_user_id: str, keyword: str, **kwargs):
        keyword_calls = calls.setdefault("keyword_calls", [])
        keyword_calls.append({
            "global_user_id": global_user_id,
            "keyword": keyword,
            "kwargs": kwargs,
        })
        if keyword != "学姐":
            return []
        return [
            _memory_row(
                "unit-x",
                "冰淇淋摊老板是千纱的初中学姐，千纱每次去都能蹭到双倍抹茶酱。",
            )
        ]

    async def _recent(*args, **kwargs):
        raise AssertionError("recent fallback should not run when keyword retrieval resolves")

    monkeypatch.setattr(module, "get_text_embedding", _embedding)
    monkeypatch.setattr(module, "search_user_memory_units_by_vector", _vector)
    monkeypatch.setattr(module, "search_user_memory_units_by_keyword", _keyword)
    monkeypatch.setattr(module, "query_user_memory_units", _recent)

    result = await agent.run(
        'Memory-evidence: retrieve durable evidence about “学姐抹茶冰淇淋店” setting',
        _context(),
    )

    assert result["resolved"] is True
    assert calls["keyword_calls"][0]["global_user_id"] == "user-1"
    assert calls["keyword_calls"][0]["keyword"] == "学姐抹茶冰淇淋店"
    assert calls["keyword_calls"][1]["global_user_id"] == "user-1"
    assert calls["keyword_calls"][1]["keyword"] == "学姐"
    payload = result["result"]
    assert payload["scope_global_user_id"] == "user-1"
    assert payload["source_system"] == "user_memory_units"
    assert payload["scope_type"] == "user_continuity"
    assert payload["missing_context"] == []
    assert payload["selected_summary"] == "冰淇淋摊老板是千纱的初中学姐，千纱每次去都能蹭到双倍抹茶酱。"
    assert payload["memory_rows"] == [
        {
            "unit_id": "unit-x",
            "unit_type": "objective_fact",
            "fact": "冰淇淋摊老板是千纱的初中学姐，千纱每次去都能蹭到双倍抹茶酱。",
            "subjective_appraisal": "Kazusa appraisal for 冰淇淋摊老板是千纱的初中学姐，千纱每次去都能蹭到双倍抹茶酱。",
            "relationship_signal": "Kazusa signal for 冰淇淋摊老板是千纱的初中学姐，千纱每次去都能蹭到双倍抹茶酱。",
            "content": "冰淇淋摊老板是千纱的初中学姐，千纱每次去都能蹭到双倍抹茶酱。",
            "updated_at": "2026-05-03T00:00:00+00:00",
            "source_system": "user_memory_units",
            "scope_type": "user_continuity",
            "scope_global_user_id": "user-1",
            "authority": "scoped_continuity",
            "truth_status": "character_lore_or_interaction_continuity",
            "origin": "consolidated_interaction",
        }
    ]


@pytest.mark.asyncio
async def test_user_memory_evidence_literal_miss_is_unresolved(monkeypatch) -> None:
    agent = UserMemoryEvidenceAgent()
    calls: dict[str, object] = {"keyword_count": 0}

    async def _embedding(text: str) -> list[float]:
        raise AssertionError(f"embedding should not run for literal miss: {text}")

    async def _vector(*args, **kwargs):
        raise AssertionError("vector search should not run for literal miss")

    async def _keyword(global_user_id: str, keyword: str, **kwargs):
        calls["keyword_count"] += 1
        calls["global_user_id"] = global_user_id
        return []

    async def _recent(*args, **kwargs):
        raise AssertionError("literal misses should not use recency fallback")

    monkeypatch.setattr(module, "get_text_embedding", _embedding)
    monkeypatch.setattr(module, "search_user_memory_units_by_vector", _vector)
    monkeypatch.setattr(module, "search_user_memory_units_by_keyword", _keyword)
    monkeypatch.setattr(module, "query_user_memory_units", _recent)

    result = await agent.run(
        'Memory-evidence: retrieve durable evidence about the exact term "不存在设定" in the current user\'s continuity',
        _context(),
    )

    assert result["resolved"] is False
    assert calls["global_user_id"] == "user-1"
    assert calls["keyword_count"] >= 1
    assert result["result"]["memory_rows"] == []
    assert result["result"]["missing_context"] == ["user_memory_evidence"]


@pytest.mark.asyncio
async def test_user_memory_evidence_scopes_results_to_current_user(monkeypatch) -> None:
    agent = UserMemoryEvidenceAgent()
    calls: dict[str, object] = {}

    async def _embedding(text: str) -> list[float]:
        raise AssertionError(f"embedding should not run for literal lookup: {text}")

    async def _vector(*args, **kwargs):
        raise AssertionError("vector search should not run for literal lookup")

    async def _keyword(global_user_id: str, keyword: str, **kwargs):
        calls["keyword_user_id"] = global_user_id
        if global_user_id == "user-2":
            return [_memory_row("unit-user-2", "Only user 2 should see this continuity.")]
        return [_memory_row("unit-user-1", "Wrong user continuity.")]

    async def _recent(*args, **kwargs):
        raise AssertionError("recent fallback should not run when keyword retrieval resolves")

    monkeypatch.setattr(module, "get_text_embedding", _embedding)
    monkeypatch.setattr(module, "search_user_memory_units_by_vector", _vector)
    monkeypatch.setattr(module, "search_user_memory_units_by_keyword", _keyword)
    monkeypatch.setattr(module, "query_user_memory_units", _recent)

    result = await agent.run(
        'Memory-evidence: retrieve durable evidence about the exact term "arcade" in the current user\'s private continuity',
        _context(global_user_id="user-2"),
    )

    assert result["resolved"] is True
    assert calls["keyword_user_id"] == "user-2"
    assert result["result"]["scope_global_user_id"] == "user-2"
    assert result["result"]["memory_rows"][0]["unit_id"] == "unit-user-2"
    assert result["result"]["memory_rows"][0]["scope_global_user_id"] == "user-2"


@pytest.mark.asyncio
async def test_user_memory_evidence_specific_semantic_miss_is_unresolved(
    monkeypatch,
) -> None:
    agent = UserMemoryEvidenceAgent()
    calls: dict[str, object] = {}

    async def _embedding(text: str) -> list[float]:
        return [0.5, 0.4, 0.3]

    async def _vector(global_user_id: str, embedding: list[float], **kwargs):
        calls["vector"] = True
        return []

    async def _keyword(global_user_id: str, keyword: str, **kwargs):
        calls["keyword"] = keyword
        return []

    async def _recent(*args, **kwargs):
        raise AssertionError("specific semantic misses should not use recency fallback")

    monkeypatch.setattr(module, "get_text_embedding", _embedding)
    monkeypatch.setattr(module, "search_user_memory_units_by_vector", _vector)
    monkeypatch.setattr(module, "search_user_memory_units_by_keyword", _keyword)
    monkeypatch.setattr(module, "query_user_memory_units", _recent)

    result = await agent.run(
        "Memory-evidence: retrieve durable evidence about the current user's accepted preference for tea",
        _context(),
    )

    assert result["resolved"] is False
    assert calls["vector"] is True
    assert "keyword" not in calls
    assert result["result"]["memory_rows"] == []
    assert result["result"]["missing_context"] == ["user_memory_evidence"]


@pytest.mark.asyncio
async def test_user_memory_evidence_falls_back_when_embedding_call_is_unavailable(
    monkeypatch,
) -> None:
    agent = UserMemoryEvidenceAgent()
    calls: dict[str, object] = {}

    async def _embedding(text: str) -> list[float]:
        raise module.OpenAIError("embedding endpoint unavailable")

    async def _vector(*args, **kwargs):
        raise AssertionError("vector search should not run when embedding generation fails")

    async def _keyword(*args, **kwargs):
        raise AssertionError("keyword search should not run without literal anchors")

    async def _recent(global_user_id: str, **kwargs):
        calls["recent"] = {
            "global_user_id": global_user_id,
            "kwargs": kwargs,
        }
        return [_memory_row("recent-embed-down", "The user still expects the 学姐 continuity.")]

    monkeypatch.setattr(module, "get_text_embedding", _embedding)
    monkeypatch.setattr(module, "search_user_memory_units_by_vector", _vector)
    monkeypatch.setattr(module, "search_user_memory_units_by_keyword", _keyword)
    monkeypatch.setattr(module, "query_user_memory_units", _recent)

    result = await agent.run(
        "Memory-evidence: retrieve current user's private continuity",
        _context(),
    )

    assert result["resolved"] is True
    assert calls["recent"]["global_user_id"] == "user-1"
    assert result["result"]["memory_rows"][0]["unit_id"] == "recent-embed-down"


@pytest.mark.asyncio
async def test_user_memory_evidence_preserves_stronger_provenance_fields(monkeypatch) -> None:
    agent = UserMemoryEvidenceAgent()

    async def _embedding(text: str) -> list[float]:
        return [0.9, 0.1]

    async def _vector(global_user_id: str, embedding: list[float], **kwargs):
        return [
            _memory_row(
                "unit-stronger",
                "The user and Kazusa already adopted the 学姐 ice-cream-shop lore.",
                truth_status="explicitly_reinforced_continuity",
                origin="user_reinforced_interaction",
                authority="explicitly_scoped_continuity",
            )
        ]

    async def _keyword(*args, **kwargs):
        raise AssertionError("keyword fallback should not run when semantic retrieval resolves")

    async def _recent(*args, **kwargs):
        raise AssertionError("recent fallback should not run when semantic retrieval resolves")

    monkeypatch.setattr(module, "get_text_embedding", _embedding)
    monkeypatch.setattr(module, "search_user_memory_units_by_vector", _vector)
    monkeypatch.setattr(module, "search_user_memory_units_by_keyword", _keyword)
    monkeypatch.setattr(module, "query_user_memory_units", _recent)

    result = await agent.run(
        "Memory-evidence: retrieve durable evidence about the current user's story continuity around the ice-cream shop",
        _context(),
    )

    assert result["resolved"] is True
    assert result["attempts"] == 1
    assert result["cache"] == {
        "enabled": False,
        "hit": False,
        "reason": "scoped_user_memory_uncached",
    }
    row = result["result"]["memory_rows"][0]
    assert row["source_system"] == "user_memory_units"
    assert row["scope_type"] == "user_continuity"
    assert row["scope_global_user_id"] == "user-1"
    assert row["authority"] == "explicitly_scoped_continuity"
    assert row["truth_status"] == "explicitly_reinforced_continuity"
    assert row["origin"] == "user_reinforced_interaction"
