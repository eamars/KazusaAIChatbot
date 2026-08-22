"""Focused target-adapter, source-policy, idempotency, and CAS tests."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import kazusa_ai_chatbot.consolidation.character_operational_state as operational_state
from kazusa_ai_chatbot.consolidation.character_operational_state import (
    CharacterOperationalExecutionContext,
    run_character_operational_target,
)
from kazusa_ai_chatbot.consolidation.source_policy import (
    validate_character_operational_sources,
)
from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_character_production_state,
)


NOW = "2026-08-02T00:00:00Z"


def _source(source_key: str) -> dict[str, str]:
    """Build one ref-complete source view for the operational slot."""

    return {
        "source_key": source_key,
        "source_kind": "episode",
        "source_id": "episode:operational-target-fixture",
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
async def test_operational_target_is_four_plus_one_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable lane list cannot starve the independent operational slot."""

    calls: list[str] = []
    claim_count = 0
    completion_count = 0
    terminal_receipt = {
        "schema_version": "character_operational_receipt.v1",
        "source_episode_id": "episode:operational-target-fixture",
        "status": "no_change",
        "sequence": 1,
        "durable": True,
        "base_updated_at": NOW,
        "committed_updated_at": "",
        "registered_at": NOW,
        "completed_at": NOW,
        "lease_owner": "operational-test-lease",
        "lease_expires_at": "",
        "attempt_count": 0,
        "error_code": None,
    }

    async def fake_carryover(**kwargs):
        del kwargs
        calls.append("carryover")
        return SimpleNamespace(
            disposition="no_change",
            decision=SimpleNamespace(action="no_change"),
        )

    async def fake_claim(**kwargs):
        nonlocal claim_count
        claim_count += 1
        if claim_count == 1:
            return {
                "claim_status": "claimed",
                "receipt": {
                    "schema_version": "character_operational_receipt.v1",
                    "source_episode_id": kwargs["lifecycle_record"][
                        "source_episode_id"
                    ],
                    "status": "pending",
                    "sequence": kwargs["sequence"],
                    "durable": True,
                    "base_updated_at": kwargs["base_updated_at"],
                    "committed_updated_at": "",
                    "registered_at": kwargs["registered_at"],
                    "completed_at": "",
                    "lease_owner": kwargs["lease_owner"],
                    "lease_expires_at": kwargs["lease_expires_at"],
                    "attempt_count": 0,
                    "error_code": None,
                },
            }
        return {"claim_status": "terminal", "receipt": terminal_receipt}

    async def fake_completion(**kwargs):
        nonlocal completion_count
        completion_count += 1
        del kwargs
        return terminal_receipt

    monkeypatch.setattr(
        operational_state,
        "run_character_carryover_cognition",
        fake_carryover,
    )
    monkeypatch.setattr(
        operational_state,
        "get_character_cognition_state",
        lambda: build_character_production_state(updated_at=NOW),
    )
    monkeypatch.setattr(
        operational_state,
        "_remaining_lease_seconds",
        lambda context: 30.0,
    )
    monkeypatch.setattr(
        operational_state,
        "claim_character_operational_receipt",
        fake_claim,
    )
    monkeypatch.setattr(
        operational_state,
        "complete_character_operational_receipt",
        fake_completion,
    )

    receipt = await run_character_operational_target(
        source_episode_id="episode:operational-target-fixture",
        sequence=1,
        evidence=[_source("episode_trace")],
        effective_at=NOW,
        services=SimpleNamespace(llm=object(), config=object()),
    )

    assert receipt.status == "no_change"
    assert len(calls) == 1
    assert claim_count == 1
    assert completion_count == 1

    repeated_receipt = await run_character_operational_target(
        source_episode_id="episode:operational-target-fixture",
        sequence=1,
        evidence=[_source("episode_trace")],
        effective_at=NOW,
        services=SimpleNamespace(llm=object(), config=object()),
    )

    assert len(calls) == 1
    assert claim_count == 2
    assert completion_count == 1
    assert repeated_receipt["status"] == receipt["status"] == "no_change"
    assert dict(repeated_receipt) == terminal_receipt


@pytest.mark.asyncio
async def test_commit_persistence_failure_preserves_typed_disposition(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Commit persistence failures stay typed and never leak raw messages."""

    captured: dict[str, object] = {}
    protected_failure: dict[str, object] = {}

    async def raising_commit(**kwargs):
        del kwargs
        raise RuntimeError("character operational transaction failed")

    async def applying_carryover(**kwargs):
        del kwargs
        return SimpleNamespace(
            disposition="apply",
            decision=SimpleNamespace(semantic_appraisal={"bonded": "high"}),
            state_update={
                "replacement_state": build_character_production_state(
                    updated_at=NOW,
                ),
            },
            attempts=0,
        )

    async def fake_completion(**kwargs):
        captured.update(kwargs)
        return {
            "status": kwargs["status"],
            "error_code": kwargs["error_code"],
        }

    monkeypatch.setattr(
        operational_state,
        "run_character_carryover_cognition",
        applying_carryover,
    )
    monkeypatch.setattr(
        operational_state,
        "_remaining_lease_seconds",
        lambda context: 30.0,
    )
    monkeypatch.setattr(
        operational_state,
        "commit_character_operational_update",
        raising_commit,
    )
    monkeypatch.setattr(
        operational_state,
        "_complete_or_in_memory_failure",
        fake_completion,
    )
    monkeypatch.setattr(
        operational_state,
        "mark_current_failure",
        lambda **kwargs: protected_failure.update(kwargs),
    )

    context = CharacterOperationalExecutionContext(
        claim={
            "claim_status": "claimed",
            "receipt": {"lease_expires_at": "2026-08-02T00:00:30Z"},
        },
        base_state=build_character_production_state(updated_at=NOW),
        lease_owner="operational-test-lease",
        registered_at=NOW,
    )
    with caplog.at_level(
        logging.ERROR,
        logger="kazusa_ai_chatbot.consolidation.character_operational_state",
    ):
        receipt = await run_character_operational_target(
            source_episode_id="episode:commit-failure",
            sequence=1,
            evidence=[_source("episode_trace")],
            effective_at=NOW,
            services=SimpleNamespace(llm=object(), config=object()),
            execution_context=context,
        )

    assert captured["status"] == "failed"
    assert captured["error_code"] == "transaction_failed"
    assert receipt["status"] == "failed"
    assert protected_failure["failure_kind"] == "operational_commit_failed"
    assert isinstance(protected_failure["exception"], RuntimeError)
    assert "transaction_failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "character operational transaction failed" not in caplog.text


@pytest.mark.asyncio
async def test_unexpected_carryover_exception_preserves_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unexpected carry-over errors do not masquerade as native rejection."""

    async def raising_carryover(**kwargs):
        del kwargs
        raise RuntimeError("provider boundary failure")

    captured: dict[str, object] = {}
    protected_failure: dict[str, object] = {}

    async def fake_completion(**kwargs):
        captured.update(kwargs)
        return {
            "status": kwargs["status"],
            "error_code": kwargs["error_code"],
        }

    monkeypatch.setattr(
        operational_state,
        "run_character_carryover_cognition",
        raising_carryover,
    )
    monkeypatch.setattr(
        operational_state,
        "_remaining_lease_seconds",
        lambda context: 30.0,
    )
    monkeypatch.setattr(
        operational_state,
        "_complete_or_in_memory_failure",
        fake_completion,
    )
    monkeypatch.setattr(
        operational_state,
        "mark_current_failure",
        lambda **kwargs: protected_failure.update(kwargs),
    )

    context = CharacterOperationalExecutionContext(
        claim={
            "claim_status": "claimed",
            "receipt": {"lease_expires_at": "2026-08-02T00:00:30Z"},
        },
        base_state=build_character_production_state(updated_at=NOW),
        lease_owner="operational-test-lease",
        registered_at=NOW,
    )
    with caplog.at_level(
        logging.ERROR,
        logger="kazusa_ai_chatbot.consolidation.character_operational_state",
    ):
        receipt = await run_character_operational_target(
            source_episode_id="episode:unexpected-carryover",
            sequence=1,
            evidence=[_source("episode_trace")],
            effective_at=NOW,
            services=SimpleNamespace(llm=object(), config=object()),
            execution_context=context,
        )

    assert captured["status"] == "failed"
    assert captured["error_code"] == "transaction_failed"
    assert captured["error_code"] != "state_rejected"
    assert receipt["status"] == "failed"
    assert protected_failure["failure_kind"] == "carryover_execution_error"
    assert isinstance(protected_failure["exception"], RuntimeError)
    assert "transaction_failed" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "provider boundary failure" not in caplog.text


@pytest.mark.asyncio
async def test_empty_and_invalid_operational_sources_never_write_state() -> None:
    """Invalid routing terminalizes without a character-state side effect."""

    with pytest.raises(ValueError):
        validate_character_operational_sources([])
