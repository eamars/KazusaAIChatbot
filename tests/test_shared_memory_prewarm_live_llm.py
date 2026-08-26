"""Guarded live LLM/DB proof for the seeded ``#napcat`` prewarm path."""

from __future__ import annotations

import os

import pytest

from kazusa_ai_chatbot.cognition_resolver import capabilities
from tests.test_shared_memory_prewarm import (
    _empty_result,
    _minimal_persona_state,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_LIVE_GUARD_ENV = "NAPCAT_PREWARM_LIVE_GUARD"
_LIVE_DATABASE_ENV = "NAPCAT_PREWARM_LIVE_DATABASE"
_MEMORY_UNIT_ID = "seed_7ac6348ccd9bf7a80fbc74584c6b3ce3"


def _require_napcat_live_runtime() -> None:
    """Require an explicit database and execution guard for this live proof."""

    if os.environ.get(_LIVE_GUARD_ENV, "").strip() != "1":
        pytest.skip(
            f"set {_LIVE_GUARD_ENV}=1 to run the isolated napcat live proof"
        )
    expected_database = os.environ.get(_LIVE_DATABASE_ENV, "").strip()
    if not expected_database:
        pytest.skip(
            f"{_LIVE_DATABASE_ENV} must name the explicitly authorized database"
        )
    if os.environ.get("MONGODB_DB_NAME", "").strip() != expected_database:
        pytest.skip(
            "napcat live proof requires MONGODB_DB_NAME to match "
            f"{_LIVE_DATABASE_ENV}"
        )


def _safe_outcome_diagnostic(outcome: object) -> str:
    """Return only disposition/count/id fields from a live outcome."""

    if not isinstance(outcome, dict):
        return f"outcome_type={type(outcome).__name__}"
    rag_result = outcome.get("rag_result")
    evidence_ids: list[str] = []
    if isinstance(rag_result, dict):
        memory_evidence = rag_result.get("memory_evidence")
        if isinstance(memory_evidence, list):
            evidence_ids = [
                str(row.get("memory_unit_id"))
                for row in memory_evidence
                if isinstance(row, dict) and row.get("memory_unit_id")
            ]
    return (
        f"status={outcome.get('status')!r} "
        f"reason_code={outcome.get('reason_code')!r} "
        f"retrieved_shared_count={outcome.get('retrieved_shared_count')!r} "
        f"merged_shared_count={outcome.get('merged_shared_count')!r} "
        f"evidence_memory_unit_ids={evidence_ids!r}"
    )


async def test_production_napcat_command_recalls_seeded_shared_memory() -> None:
    """Verify the real ``@character #napcat`` row survives prewarm merge."""

    _require_napcat_live_runtime()
    state = _minimal_persona_state()
    state["decontextualized_input"] = '@一之濑明日奈 #napcat'
    state["prompt_message_context"]["body_text"] = '@一之濑明日奈 #napcat'
    state["prompt_message_context"]["mentions"] = [{
        "global_user_id": "character-1",
        "display_name": "一之濑明日奈",
        "entity_kind": "bot",
    }]

    outcome = await capabilities.run_first_cycle_shared_memory_prewarm(state)
    print(f"napcat prewarm outcome: {_safe_outcome_diagnostic(outcome)}")

    assert outcome["status"] == "completed", _safe_outcome_diagnostic(outcome)
    assert outcome["reason_code"] == "shared_memory_ready"
    evidence = outcome["rag_result"]["memory_evidence"]
    assert any(
        row.get("memory_unit_id") == _MEMORY_UNIT_ID
        for row in evidence
        if isinstance(row, dict)
    ), _safe_outcome_diagnostic(outcome)

    merged_rag_result, merged_outcome = (
        capabilities.merge_shared_memory_prewarm_outcome(
            _empty_result(),
            outcome,
        )
    )
    print(
        "napcat merged outcome: "
        f"{_safe_outcome_diagnostic(merged_outcome)}"
    )
    assert merged_outcome["status"] == "completed"
    assert merged_outcome["reason_code"] == "shared_memory_merged"
    assert any(
        row.get("memory_unit_id") == _MEMORY_UNIT_ID
        for row in merged_rag_result["memory_evidence"]
        if isinstance(row, dict)
    )
