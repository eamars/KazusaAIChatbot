"""Deterministic contract tests for the RAG2 Recall helper."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.db.schemas import UserMemoryUnitStatus, UserMemoryUnitType
from kazusa_ai_chatbot.rag.recall import RecallAgent
from kazusa_ai_chatbot.rag.recall import agent as recall_agent_module
from kazusa_ai_chatbot.rag.recall.collectors import commitments as commitments_module
from kazusa_ai_chatbot.rag.recall.collectors import (
    calendar_runs as calendar_runs_module,
)
from kazusa_ai_chatbot.rag.recall.collectors import history as history_module


def _base_context(**overrides: object) -> dict:
    """Build the minimum scoped Recall context used by deterministic tests.

    Args:
        overrides: Context fields that should replace the default scoped values.

    Returns:
        Context dict passed to ``RecallAgent.run``.
    """

    context = {
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "global_user_id": "user-1",
        "current_timestamp_utc": "2026-05-02T00:00:00+00:00",
        "conversation_progress": {
            "status": "active",
            "continuity": "same_episode",
            "current_thread": "User will pick up the character at 9:30 for the roller coaster plan.",
            "open_loops": [{"text": "Go to the amusement park today."}],
            "resolved_threads": [{"text": "Pickup time is 9:30."}],
            "next_affordances": ["Confirm pickup details."],
            "progression_guidance": "Continue the appointment setup.",
        },
        "conversation_episode_state": {
            "updated_at": "2026-05-01T23:00:00+00:00",
            "expires_at": "2026-05-03T00:00:00+00:00",
            "turn_count": 12,
        },
    }
    context.update(overrides)
    return context


async def _empty_active_commitments(
    global_user_id: str,
    *,
    unit_types: list[str] | None = None,
    statuses: list[str] | None = None,
    limit: int = 100,
) -> list[dict]:
    """Return no active commitments while preserving query arguments."""
    del global_user_id, unit_types, statuses, limit
    return_value: list[dict] = []
    return return_value


async def _empty_calendar_runs(
    *,
    platform: str,
    platform_channel_id: str,
    global_user_id: str,
    current_timestamp_utc: str,
    limit: int = 10,
) -> list[dict]:
    """Return no calendar runs while preserving query arguments."""
    del platform, platform_channel_id, global_user_id
    del current_timestamp_utc, limit
    return_value: list[dict] = []
    return return_value


async def _empty_history(
    platform: str | None = None,
    platform_channel_id: str | None = None,
    limit: int = 20,
    global_user_id: str | None = None,
    display_name: str | None = None,
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
) -> list[dict]:
    """Return no conversation rows while preserving query arguments."""
    del platform, platform_channel_id, limit, global_user_id
    del display_name, from_timestamp, to_timestamp
    return_value: list[dict] = []
    return return_value


def _patch_empty_sources(monkeypatch) -> None:
    """Patch every external Recall source to an empty deterministic result."""
    monkeypatch.setattr(commitments_module, "query_user_memory_units", _empty_active_commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _empty_calendar_runs,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _empty_history)


@pytest.mark.asyncio
async def test_recall_missing_mandatory_scope_does_not_query_db(monkeypatch) -> None:
    """Missing scope returns the explicit unresolved result before DB access."""
    queried = False

    async def _fail_query(*args, **kwargs):
        nonlocal queried
        queried = True
        raise AssertionError("Recall must not query DB without mandatory scope")

    monkeypatch.setattr(commitments_module, "query_user_memory_units", _fail_query)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _fail_query,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _fail_query)

    result = await RecallAgent().run(
        "Recall: retrieve active_episode_agreement relevant to today's appointment",
        {
            "platform": "qq",
            "platform_channel_id": "",
            "global_user_id": "user-1",
            "current_timestamp_utc": "2026-05-02T00:00:00+00:00",
        },
    )

    assert queried is False
    assert result["resolved"] is False
    assert result["attempts"] == 1
    assert result["cache"] == {
        "enabled": False,
        "hit": False,
        "reason": "volatile_recall",
    }
    assert result["result"]["error"] == "missing_mandatory_context"
    assert "platform_channel_id" in result["result"]["freshness_basis"]


@pytest.mark.asyncio
async def test_recall_active_progress_answers_current_agreement(monkeypatch) -> None:
    """Active episode recall should prefer current progress evidence."""
    history_called = False

    async def _history_probe(**kwargs):
        nonlocal history_called
        history_called = True
        del kwargs
        return_value: list[dict] = []
        return return_value

    monkeypatch.setattr(commitments_module, "query_user_memory_units", _empty_active_commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _empty_calendar_runs,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _history_probe)

    result = await RecallAgent().run(
        "Recall: retrieve active_episode_agreement relevant to today's appointment",
        _base_context(),
    )

    assert history_called is False
    assert result["resolved"] is True
    assert result["result"]["recall_type"] == "active_episode_agreement"
    assert result["result"]["primary_source"] == "conversation_progress"
    assert "9:30" in result["result"]["selected_summary"]
    assert result["result"]["freshness_basis"]


@pytest.mark.asyncio
async def test_recall_durable_commitment_prefers_active_memory_units(monkeypatch) -> None:
    """Durable commitment mode should select active commitment memory first."""

    async def _commitments(
        global_user_id: str,
        *,
        unit_types: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        assert global_user_id == "user-1"
        assert unit_types == [UserMemoryUnitType.ACTIVE_COMMITMENT]
        assert statuses == [UserMemoryUnitStatus.ACTIVE]
        assert limit == 6
        return_value = [
            {
                "unit_id": "commitment-1",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "The user promised to bring tickets for the amusement park.",
                "updated_at": "2026-05-01T22:00:00+00:00",
            }
        ]
        return return_value

    monkeypatch.setattr(commitments_module, "query_user_memory_units", _commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _empty_calendar_runs,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _empty_history)

    result = await RecallAgent().run(
        "Recall: retrieve durable_commitment relevant to ongoing promises",
        _base_context(),
    )

    assert result["resolved"] is True
    assert result["result"]["recall_type"] == "durable_commitment"
    assert result["result"]["primary_source"] == "user_memory_units"
    assert "tickets" in result["result"]["selected_summary"]


@pytest.mark.asyncio
async def test_recall_includes_pending_calendar_runs(monkeypatch) -> None:
    """Pending calendar runs should appear as executable future evidence."""

    async def _events(
        *,
        platform: str,
        platform_channel_id: str,
        global_user_id: str,
        current_timestamp_utc: str,
        limit: int = 10,
    ) -> list[dict]:
        assert platform == "qq"
        assert platform_channel_id == "chan-1"
        assert global_user_id == "user-1"
        assert current_timestamp_utc == "2026-05-02T00:00:00+00:00"
        assert limit == 10
        return_value = [
            {
                "run_id": "run-1",
                "trigger_kind": "future_cognition",
                "payload": {
                    "continuation_objective": (
                        "Re-evaluate leaving for the amusement park."
                    )
                },
                "due_at": "2026-05-02T01:00:00+00:00",
                "status": "pending",
            }
        ]
        return return_value

    monkeypatch.setattr(commitments_module, "query_user_memory_units", _empty_active_commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _events,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _empty_history)

    result = await RecallAgent().run(
        "Recall: retrieve active_episode_agreement relevant to today's appointment",
        _base_context(conversation_progress=None),
    )

    sources = {candidate["source"] for candidate in result["result"]["candidates"]}
    calendar_claims = [
        candidate["claim"]
        for candidate in result["result"]["candidates"]
        if candidate["source"] == "calendar_runs"
    ]

    assert "calendar_runs" in sources
    assert calendar_claims == [
        "Pending calendar future cognition at 2026-05-02T01:00:00+00:00: "
        "Re-evaluate leaving for the amusement park."
    ]


@pytest.mark.asyncio
async def test_recall_active_mode_uses_commitment_before_calendar_run(
    monkeypatch,
) -> None:
    """Active recall without progress should select commitments before runs."""

    async def _commitments(
        global_user_id: str,
        *,
        unit_types: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        del global_user_id, unit_types, statuses, limit
        return_value = [
            {
                "unit_id": "commitment-1",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "The user promised to meet at the station.",
                "updated_at": "2026-05-01T22:00:00+00:00",
            }
        ]
        return return_value

    async def _events(
        *,
        platform: str,
        platform_channel_id: str,
        global_user_id: str,
        current_timestamp_utc: str,
        limit: int = 10,
    ) -> list[dict]:
        del platform, platform_channel_id, global_user_id
        del current_timestamp_utc, limit
        return_value = [
            {
                "run_id": "run-1",
                "trigger_kind": "future_cognition",
                "payload": {
                    "continuation_objective": (
                        "Calendar run should remain supporting."
                    )
                },
                "due_at": "2026-05-02T01:00:00+00:00",
                "status": "pending",
            }
        ]
        return return_value

    monkeypatch.setattr(commitments_module, "query_user_memory_units", _commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _events,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _empty_history)

    result = await RecallAgent().run(
        "Recall: retrieve active_episode_agreement relevant to today's appointment",
        _base_context(conversation_progress=None),
    )

    assert result["resolved"] is True
    assert result["result"]["primary_source"] == "user_memory_units"
    assert result["result"]["supporting_sources"] == ["calendar_runs"]
    assert "station" in result["result"]["selected_summary"]
    assert (
        "Active-episode state was unavailable"
        in result["result"]["freshness_basis"]
    )


@pytest.mark.asyncio
async def test_recall_review_rejects_unrelated_fallback_commitments(monkeypatch) -> None:
    """Multiple fallback commitments should not resolve a concrete slot by category."""

    async def _commitments(
        global_user_id: str,
        *,
        unit_types: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        del global_user_id, unit_types, statuses, limit
        return_value = [
            {
                "unit_id": "commitment-dessert",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "The user owes the character a dessert challenge reward.",
                "updated_at": "2026-05-01T22:00:00+00:00",
            },
            {
                "unit_id": "commitment-noodle",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "The user promised to cook noodles as compensation.",
                "updated_at": "2026-05-01T23:00:00+00:00",
            },
        ]
        return return_value

    async def _review(
        *,
        task: str,
        mode: str,
        candidates: list[dict[str, str]],
    ) -> dict[str, object]:
        del task, mode, candidates
        return_value = {
            "confirmed_candidate_indexes": [],
            "nearby_candidate_indexes": [0],
            "summary": "",
            "uncertainty": "Only nearby fallback commitments were found.",
            "source_hints": ["Search conversation evidence for exact history."],
        }
        return return_value

    monkeypatch.setattr(commitments_module, "query_user_memory_units", _commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _empty_calendar_runs,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _empty_history)
    monkeypatch.setattr(recall_agent_module, "_review_recall_candidates", _review)

    result = await RecallAgent().run(
        "Recall: retrieve active_episode_agreement relevant to Beers by Bacon Bros and tiramisu promises",
        _base_context(conversation_progress=None),
    )

    assert result["resolved"] is False
    assert result["result"]["selected_summary"] == ""
    assert result["result"]["missing_context"] == ["recall_evidence"]
    assert result["result"]["observation_candidates"][0]["source"] == (
        "recall:user_memory_units"
    )
    assert result["result"]["source_hints"] == [
        {
            "kind": "recall",
            "source": "Search conversation evidence for exact history.",
        }
    ]


@pytest.mark.asyncio
async def test_recall_review_accepts_matching_fallback_commitment(monkeypatch) -> None:
    """A reviewer-confirmed durable fallback commitment remains accepted."""

    async def _commitments(
        global_user_id: str,
        *,
        unit_types: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        del global_user_id, unit_types, statuses, limit
        return_value = [
            {
                "unit_id": "commitment-dessert",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "The user owes the character a dessert challenge reward.",
                "updated_at": "2026-05-01T22:00:00+00:00",
            },
            {
                "unit_id": "commitment-noodle",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "The user promised to cook noodles as compensation.",
                "updated_at": "2026-05-01T23:00:00+00:00",
            },
        ]
        return return_value

    async def _review(
        *,
        task: str,
        mode: str,
        candidates: list[dict[str, str]],
    ) -> dict[str, object]:
        del task, mode, candidates
        return_value = {
            "confirmed_candidate_indexes": [1],
            "nearby_candidate_indexes": [],
            "summary": "The user promised to cook noodles as compensation.",
            "uncertainty": "",
            "source_hints": [],
        }
        return return_value

    monkeypatch.setattr(commitments_module, "query_user_memory_units", _commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _empty_calendar_runs,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _empty_history)
    monkeypatch.setattr(recall_agent_module, "_review_recall_candidates", _review)

    result = await RecallAgent().run(
        "Recall: retrieve durable_commitment relevant to noodle compensation",
        _base_context(conversation_progress=None),
    )

    assert result["resolved"] is True
    assert result["result"]["primary_source"] == "user_memory_units"
    assert result["result"]["selected_summary"] == (
        "The user promised to cook noodles as compensation."
    )


@pytest.mark.asyncio
async def test_recall_active_review_accepts_exact_durable_memory_fallback(
    monkeypatch,
) -> None:
    """A reviewer-confirmed active memory can answer when progress is absent."""

    async def _commitments(
        global_user_id: str,
        *,
        unit_types: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        del global_user_id, unit_types, statuses, limit
        return_value = [
            {
                "unit_id": "commitment-dessert",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "The user owes the character a dessert challenge reward.",
                "updated_at": "2026-05-01T22:00:00+00:00",
            },
            {
                "unit_id": "commitment-noodle",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "The user promised to cook noodles as compensation.",
                "updated_at": "2026-05-01T23:00:00+00:00",
            },
        ]
        return return_value

    async def _review(
        *,
        task: str,
        mode: str,
        candidates: list[dict[str, str]],
    ) -> dict[str, object]:
        del task, mode, candidates
        return_value = {
            "confirmed_candidate_indexes": [1],
            "nearby_candidate_indexes": [0],
            "summary": "The user promised to cook noodles as compensation.",
            "uncertainty": "",
            "source_hints": [],
        }
        return return_value

    monkeypatch.setattr(commitments_module, "query_user_memory_units", _commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _empty_calendar_runs,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _empty_history)
    monkeypatch.setattr(recall_agent_module, "_review_recall_candidates", _review)

    result = await RecallAgent().run(
        "Recall: retrieve active_episode_agreement relevant to noodle compensation",
        _base_context(conversation_progress=None),
    )

    assert result["resolved"] is True
    assert result["result"]["recall_type"] == "durable_commitment"
    assert result["result"]["primary_source"] == "user_memory_units"
    assert result["result"]["selected_summary"] == (
        "The user promised to cook noodles as compensation."
    )
    assert (
        "Active-episode state was unavailable"
        in result["result"]["freshness_basis"]
    )


@pytest.mark.asyncio
async def test_recall_active_review_accepts_broad_memory_inventory(
    monkeypatch,
) -> None:
    """A broad reviewed inventory can return multiple durable commitments."""

    async def _commitments(
        global_user_id: str,
        *,
        unit_types: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        del global_user_id, unit_types, statuses, limit
        return_value = [
            {
                "unit_id": "commitment-dessert",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "The user owes the character a dessert challenge reward.",
                "updated_at": "2026-05-01T22:00:00+00:00",
            },
            {
                "unit_id": "commitment-noodle",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "The user promised to cook noodles as compensation.",
                "updated_at": "2026-05-01T23:00:00+00:00",
            },
        ]
        return return_value

    async def _review(
        *,
        task: str,
        mode: str,
        candidates: list[dict[str, str]],
    ) -> dict[str, object]:
        del task, mode, candidates
        return_value = {
            "confirmed_candidate_indexes": [0, 1],
            "nearby_candidate_indexes": [],
            "summary": "Two active commitments remain relevant.",
            "uncertainty": "",
            "source_hints": [],
        }
        return return_value

    monkeypatch.setattr(commitments_module, "query_user_memory_units", _commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _empty_calendar_runs,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _empty_history)
    monkeypatch.setattr(recall_agent_module, "_review_recall_candidates", _review)

    result = await RecallAgent().run(
        "Recall: retrieve active_episode_agreement relevant to open commitments",
        _base_context(conversation_progress=None),
    )

    assert result["resolved"] is True
    assert result["result"]["selected_summary"] == (
        "Two active commitments remain relevant."
    )


@pytest.mark.asyncio
async def test_recall_exact_history_runs_history_gate(monkeypatch) -> None:
    """Exact agreement history mode should retrieve bounded transcript proof."""
    history_calls: list[dict] = []

    async def _history_probe(**kwargs):
        history_calls.append(kwargs)
        return_value = [
            {
                "body_text": "We agreed on pickup at 9:30.",
                "timestamp": "2026-05-01T23:00:00+00:00",
                "role": "user",
                "display_name": "User",
            }
        ]
        return return_value

    monkeypatch.setattr(commitments_module, "query_user_memory_units", _empty_active_commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _empty_calendar_runs,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _history_probe)

    result = await RecallAgent().run(
        "Recall: retrieve exact_agreement_history relevant to when the agreement was made",
        _base_context(),
    )

    assert history_calls == [
        {
            "platform": "qq",
            "platform_channel_id": "chan-1",
            "global_user_id": "user-1",
            "limit": 20,
        }
    ]
    assert result["resolved"] is True
    assert result["result"]["recall_type"] == "exact_history"
    assert result["result"]["primary_source"] == "conversation_history"


@pytest.mark.asyncio
async def test_recall_ignores_stale_progress_and_uses_active_commitment(monkeypatch) -> None:
    """Sharp-transition progress must not answer as active episode evidence."""

    async def _commitments(
        global_user_id: str,
        *,
        unit_types: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        del global_user_id, unit_types, statuses, limit
        return_value = [
            {
                "unit_id": "commitment-1",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "The user promised to meet at the station.",
                "updated_at": "2026-05-01T22:00:00+00:00",
            }
        ]
        return return_value

    sharp_progress = {
        "status": "active",
        "continuity": "sharp_transition",
        "current_thread": "Old plan that should not be active.",
    }
    monkeypatch.setattr(commitments_module, "query_user_memory_units", _commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _empty_calendar_runs,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _empty_history)

    result = await RecallAgent().run(
        "Recall: retrieve active_episode_agreement relevant to today's appointment",
        _base_context(conversation_progress=sharp_progress),
    )

    assert result["resolved"] is True
    assert result["result"]["primary_source"] == "user_memory_units"
    assert "station" in result["result"]["selected_summary"]
    assert "Old plan" not in result["result"]["selected_summary"]


@pytest.mark.asyncio
async def test_recall_excludes_inactive_memory_units(monkeypatch) -> None:
    """Recall must discard inactive commitment rows even if a DB mock returns them."""

    async def _commitments(
        global_user_id: str,
        *,
        unit_types: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        del global_user_id, unit_types, statuses, limit
        return_value = [
            {
                "unit_id": "completed-1",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.COMPLETED,
                "fact": "Completed plan must not appear.",
                "updated_at": "2026-05-01T20:00:00+00:00",
            },
            {
                "unit_id": "cancelled-1",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.CANCELLED,
                "fact": "Cancelled plan must not appear.",
                "updated_at": "2026-05-01T21:00:00+00:00",
            },
            {
                "unit_id": "active-1",
                "unit_type": UserMemoryUnitType.ACTIVE_COMMITMENT,
                "status": UserMemoryUnitStatus.ACTIVE,
                "fact": "Active plan should appear.",
                "updated_at": "2026-05-01T22:00:00+00:00",
            },
        ]
        return return_value

    monkeypatch.setattr(commitments_module, "query_user_memory_units", _commitments)
    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _empty_calendar_runs,
    )
    monkeypatch.setattr(history_module, "get_conversation_history", _empty_history)

    result = await RecallAgent().run(
        "Recall: retrieve durable_commitment relevant to ongoing promises",
        _base_context(conversation_progress=None),
    )

    rendered = repr(result)

    assert "Active plan should appear." in result["result"]["selected_summary"]
    assert "Completed plan" not in rendered
    assert "Cancelled plan" not in rendered


@pytest.mark.asyncio
async def test_recall_output_caps(monkeypatch) -> None:
    """Recall result text should be compact before entering downstream prompts."""
    long_text = "A" * 1000
    _patch_empty_sources(monkeypatch)

    result = await RecallAgent().run(
        "Recall: retrieve active_episode_agreement relevant to today's appointment",
        _base_context(
            conversation_progress={
                "status": "active",
                "continuity": "same_episode",
                "current_thread": long_text,
                "open_loops": [{"text": long_text}],
                "resolved_threads": [{"text": long_text}],
                "next_affordances": [long_text],
                "progression_guidance": long_text,
            }
        ),
    )

    assert len(result["result"]["selected_summary"]) <= 600
    assert len(result["result"]["freshness_basis"]) <= 400
    assert len(result["result"]["conflicts"]) <= 5
    assert len(result["result"]["candidates"]) <= 29
    for candidate in result["result"]["candidates"]:
        assert len(candidate["claim"]) <= 240


@pytest.mark.asyncio
async def test_calendar_run_collector_scopes_pending_runs(monkeypatch) -> None:
    """Calendar-run recall should issue one scoped pending-run read."""
    recorded: dict = {}

    async def _list_pending_calendar_runs_for_source(**kwargs):
        recorded.update(kwargs)
        return_value = [
            {
                "run_id": "run-1",
                "trigger_kind": "future_cognition",
                "payload": {
                    "continuation_objective": (
                        "Review the appointment follow-up."
                    )
                },
                "due_at": "2026-05-02T01:00:00+00:00",
                "status": "pending",
            }
        ]
        return return_value

    monkeypatch.setattr(
        calendar_runs_module,
        "list_pending_calendar_runs_for_source",
        _list_pending_calendar_runs_for_source,
    )

    candidates = await calendar_runs_module.CalendarRunCollector().collect(
        _base_context(),
    )

    assert recorded == {
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "global_user_id": "user-1",
        "current_timestamp_utc": "2026-05-02T00:00:00+00:00",
        "limit": 10,
    }
    assert candidates == [
        {
            "source": "calendar_runs",
            "claim": (
                "Pending calendar future cognition at "
                "2026-05-02T01:00:00+00:00: Review the appointment follow-up."
            ),
            "temporal_scope": "pending_future_action",
            "lifecycle_status": "pending",
            "evidence_time": "2026-05-02T01:00:00+00:00",
            "authority": "supporting",
        }
    ]
