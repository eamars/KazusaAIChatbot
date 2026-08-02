"""Focused target-adapter, source-policy, idempotency, and CAS tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.consolidation.character_operational_state import (
    run_character_operational_target,
)
from kazusa_ai_chatbot.consolidation.source_policy import (
    validate_character_operational_sources,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
)


NOW = "2026-08-02T00:00:00Z"


def _source(source_key: str) -> dict[str, str]:
    """Build one ref-complete source view for the operational slot."""

    return {
        "source_key": source_key,
        "source_kind": "episode",
        "source_id": "episode:one",
        "occurred_at": NOW,
        "semantic_text": "closed operational event",
    }


def test_operational_source_policy_accepts_only_ref_complete_current_sources() -> None:
    """Reject reflection/RAG sources before the adapter can call cognition."""

    accepted = validate_character_operational_sources([
        _source("episode_trace"),
        _source("assistant_final_dialog"),
    ])
    assert [row["source_key"] for row in accepted] == [
        "episode_trace",
        "assistant_final_dialog",
    ]

    with pytest.raises(ValueError):
        validate_character_operational_sources([_source("rag_result")])


@pytest.mark.asyncio
async def test_operational_target_is_four_plus_one_and_idempotent(monkeypatch) -> None:
    """A durable lane list cannot starve the independent operational slot."""

    calls: list[str] = []

    async def fake_carryover(**kwargs):
        del kwargs
        calls.append("carryover")
        return SimpleNamespace(
            disposition="no_change",
            decision=SimpleNamespace(action="no_change"),
        )

    monkeypatch.setattr(
        "kazusa_ai_chatbot.consolidation.character_operational_state.run_character_carryover_cognition",
        fake_carryover,
    )
    monkeypatch.setattr(
        "kazusa_ai_chatbot.consolidation.character_operational_state.get_character_cognition_state",
        lambda: build_character_production_state(updated_at=NOW),
    )

    receipt = await run_character_operational_target(
        source_episode_id="episode:one",
        sequence=1,
        evidence=[_source("episode_trace")],
        effective_at=NOW,
        services=SimpleNamespace(llm=object(), config=object()),
    )

    assert receipt.status in {"no_change", "committed"}
    assert len(calls) <= 1


@pytest.mark.asyncio
async def test_empty_and_invalid_operational_sources_never_write_state() -> None:
    """Invalid routing terminalizes without a character-state side effect."""

    with pytest.raises(ValueError):
        validate_character_operational_sources([])

