"""Read-only repository adapter tests for unavailable dependencies."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_repository_returns_safe_unavailable_summaries_without_db() -> None:
    """Lookup adapters should degrade to bounded unavailable data."""

    from control_console.repository import ControlConsoleRepository
    from kazusa_ai_chatbot.db.errors import DatabaseOperationError

    async def unavailable_helper(**kwargs):
        _ = kwargs
        raise DatabaseOperationError("db unavailable")

    async def unavailable_character_helper():
        raise DatabaseOperationError("db unavailable")

    repository = ControlConsoleRepository(
        get_character_runtime_state=unavailable_character_helper,
        list_growth_traits=unavailable_helper,
    )

    character = await repository.latest_character_status()
    growth = await repository.global_growth_summary()
    memory = await repository.lookup_memory(
        global_user_id="",
        query="",
        limit=5,
    )

    assert character["status"] in {"unavailable", "empty"}
    assert growth["status"] in {"unavailable", "empty"}
    assert memory["items"] == []
    assert memory["redaction"]["embeddings"] == "excluded"
    assert "prompt" not in repr(memory).lower()


@pytest.mark.asyncio
async def test_repository_projects_character_and_growth_helpers_safely() -> None:
    """DB-owned helper outputs should be projected into bounded console data."""

    from control_console.repository import ControlConsoleRepository

    async def get_character_runtime_state():
        return {
            "mood": "focused",
            "global_vibe": "steady",
            "reflection_summary": "short safe summary",
            "prompt_text": "must redact",
            "updated_at": "2026-06-17T00:00:00+00:00",
        }

    async def list_growth_traits(*, limit: int):
        assert limit == 12
        return [
            {
                "trait_id": "trait-1",
                "growth_axis": "communication",
                "status": "active",
                "maturity_band": "promoted",
                "prompt_text": "must redact",
                "updated_at": "2026-06-17T00:00:00+00:00",
            }
        ]

    repository = ControlConsoleRepository(
        get_character_runtime_state=get_character_runtime_state,
        list_growth_traits=list_growth_traits,
    )

    character = await repository.latest_character_status()
    growth = await repository.global_growth_summary()

    assert character["status"] == "available"
    assert character["summary"]["mood"] == "focused"
    assert "prompt_text" not in character["summary"]
    assert growth["status"] == "available"
    assert growth["items"][0]["trait_id"] == "trait-1"
    assert "prompt_text" not in growth["items"][0]


@pytest.mark.asyncio
async def test_repository_projects_application_identity_from_character_profile() -> None:
    """The browser brand should come from the active character profile."""

    from control_console.repository import ControlConsoleRepository
    from kazusa_ai_chatbot.db.errors import DatabaseOperationError

    async def get_character_profile():
        return {
            "name": "杏山千纱 (Kyōyama Kazusa)",
            "prompt_text": "must redact",
        }

    repository = ControlConsoleRepository(
        get_character_profile=get_character_profile,
    )

    identity = await repository.application_identity()

    assert identity["status"] == "available"
    assert identity["character_name"] == "杏山千纱 (Kyōyama Kazusa)"
    assert identity["source"] == "character_state"
    assert "generated_at" in identity
    assert "prompt" not in repr(identity).lower()

    async def unavailable_character_profile():
        raise DatabaseOperationError("db unavailable")

    unavailable_repository = ControlConsoleRepository(
        get_character_profile=unavailable_character_profile,
    )

    unavailable_identity = await unavailable_repository.application_identity()

    assert unavailable_identity["status"] == "unavailable"
    assert unavailable_identity["character_name"] == "not connected"
    assert unavailable_identity["reason"] == "db unavailable"


@pytest.mark.asyncio
async def test_repository_projects_user_memory_units_with_redaction() -> None:
    """Memory lookup should use DB-owned helpers and expose safe fields only."""

    from control_console.repository import ControlConsoleRepository

    calls: list[dict[str, object]] = []

    async def query_user_memory_units(global_user_id: str, *, limit: int):
        calls.append({
            "helper": "recent",
            "global_user_id": global_user_id,
            "limit": limit,
        })
        return [
            {
                "unit_id": "unit-1",
                "unit_type": "stable_pattern",
                "status": "active",
                "fact": "User likes direct technical reviews.",
                "relationship_signal": "prefers honesty",
                "subjective_appraisal": "operator trust context",
                "last_seen_at": "2026-06-17T00:00:00+00:00",
                "updated_at": "2026-06-17T00:00:00+00:00",
                "embedding": [0.1, 0.2],
                "prompt_text": "must redact",
                "raw_message": "must redact",
            }
        ]

    async def search_user_memory_units_by_keyword(
        global_user_id: str,
        keyword: str,
        *,
        limit: int,
    ):
        calls.append({
            "helper": "keyword",
            "global_user_id": global_user_id,
            "keyword": keyword,
            "limit": limit,
        })
        return [
            {
                "unit_id": "unit-2",
                "unit_type": "objective_fact",
                "status": "active",
                "fact": "User reviews every console workflow.",
                "relationship_signal": "",
                "last_seen_at": "2026-06-16T00:00:00+00:00",
            }
        ]

    repository = ControlConsoleRepository(
        query_user_memory_units=query_user_memory_units,
        search_user_memory_units_by_keyword=search_user_memory_units_by_keyword,
    )

    recent = await repository.lookup_memory(
        global_user_id="user-1",
        query="",
        limit=5,
    )
    keyword = await repository.lookup_memory(
        global_user_id="user-1",
        query="reviews",
        limit=3,
    )

    assert calls == [
        {"helper": "recent", "global_user_id": "user-1", "limit": 5},
        {
            "helper": "keyword",
            "global_user_id": "user-1",
            "keyword": "reviews",
            "limit": 3,
        },
    ]
    assert recent["status"] == "available"
    assert recent["items"][0] == {
        "unit_id": "unit-1",
        "unit_type": "stable_pattern",
        "status": "active",
        "fact": "User likes direct technical reviews.",
        "relationship_signal": "prefers honesty",
        "subjective_appraisal": "operator trust context",
        "last_seen_at": "2026-06-17T00:00:00+00:00",
        "updated_at": "2026-06-17T00:00:00+00:00",
    }
    assert keyword["items"][0]["unit_id"] == "unit-2"
    assert "embedding" not in repr(recent["items"]).lower()
    assert "prompt" not in repr(recent["items"]).lower()
    assert "raw_message" not in repr(recent["items"]).lower()


@pytest.mark.asyncio
async def test_repository_memory_lookup_requires_global_user_id() -> None:
    """Memory lookup should not query all users from a blank scope."""

    from control_console.repository import ControlConsoleRepository

    repository = ControlConsoleRepository()

    page = await repository.lookup_memory(
        global_user_id="",
        query="anything",
        limit=5,
    )

    assert page["status"] == "needs_input"
    assert page["items"] == []
    assert "global_user_id" in page["reason"]


@pytest.mark.asyncio
async def test_repository_memory_lookup_reports_invalid_configuration() -> None:
    """Configuration failures should render unavailable instead of HTTP 500."""

    from control_console.repository import ControlConsoleRepository

    async def query_user_memory_units(global_user_id: str, *, limit: int):
        _ = global_user_id
        _ = limit
        raise ValueError("EMBEDDING_MODEL must be configured")

    async def search_user_memory_units_by_keyword(
        global_user_id: str,
        keyword: str,
        *,
        limit: int,
    ):
        _ = global_user_id
        _ = keyword
        _ = limit
        raise AssertionError("keyword helper should not run")

    repository = ControlConsoleRepository(
        query_user_memory_units=query_user_memory_units,
        search_user_memory_units_by_keyword=search_user_memory_units_by_keyword,
    )

    page = await repository.lookup_memory(
        global_user_id="user-1",
        query="",
        limit=5,
    )

    assert page["status"] == "unavailable"
    assert "EMBEDDING_MODEL" in page["reason"]


@pytest.mark.asyncio
async def test_repository_projects_interaction_style_context_safely() -> None:
    """Interaction-style lookup should expose scoped guideline summaries only."""

    from control_console.repository import ControlConsoleRepository

    calls: list[dict[str, str]] = []

    async def build_interaction_style_context(
        *,
        global_user_id: str,
        channel_type: str,
        platform: str,
        platform_channel_id: str,
    ):
        calls.append({
            "global_user_id": global_user_id,
            "channel_type": channel_type,
            "platform": platform,
            "platform_channel_id": platform_channel_id,
        })
        return {
            "application_order": ["user_style", "group_channel_style"],
            "user_style": {
                "speech_guidelines": ["be direct"],
                "social_guidelines": ["avoid pretending certainty"],
                "pacing_guidelines": [],
                "engagement_guidelines": ["ask for evidence"],
                "confidence": "medium",
                "source_reflection_run_ids": ["must-redact"],
            },
            "group_channel_style": {
                "speech_guidelines": [],
                "social_guidelines": ["keep operator context visible"],
                "pacing_guidelines": ["short updates"],
                "engagement_guidelines": [],
                "confidence": "low",
            },
        }

    repository = ControlConsoleRepository(
        build_interaction_style_context=build_interaction_style_context,
    )

    page = await repository.lookup_interaction_style(
        global_user_id="user-1",
        platform="debug",
        platform_channel_id="group-1",
    )

    assert calls == [
        {
            "global_user_id": "user-1",
            "channel_type": "group",
            "platform": "debug",
            "platform_channel_id": "group-1",
        }
    ]
    assert page["status"] == "available"
    assert page["items"] == [
        {
            "scope": "user_style",
            "field": "speech_guidelines",
            "guidelines": ["be direct"],
            "confidence": "medium",
        },
        {
            "scope": "user_style",
            "field": "social_guidelines",
            "guidelines": ["avoid pretending certainty"],
            "confidence": "medium",
        },
        {
            "scope": "user_style",
            "field": "engagement_guidelines",
            "guidelines": ["ask for evidence"],
            "confidence": "medium",
        },
        {
            "scope": "group_channel_style",
            "field": "social_guidelines",
            "guidelines": ["keep operator context visible"],
            "confidence": "low",
        },
        {
            "scope": "group_channel_style",
            "field": "pacing_guidelines",
            "guidelines": ["short updates"],
            "confidence": "low",
        },
    ]
    assert "source_reflection_run_ids" not in repr(page)


@pytest.mark.asyncio
async def test_repository_interaction_style_lookup_requires_scope() -> None:
    """Interaction-style lookup should not query without a scoped identity."""

    from control_console.repository import ControlConsoleRepository

    repository = ControlConsoleRepository()

    page = await repository.lookup_interaction_style(
        global_user_id="",
        platform="",
        platform_channel_id="",
    )

    assert page["status"] == "needs_input"
    assert page["items"] == []
    assert "global_user_id" in page["reason"]


@pytest.mark.asyncio
async def test_repository_projects_due_calendar_runs_safely() -> None:
    """Calendar lookup should expose due-run state without raw payloads."""

    from control_console.repository import ControlConsoleRepository
    from kazusa_ai_chatbot.calendar_scheduler import models

    calls: list[dict[str, object]] = []

    async def list_due_calendar_runs(
        *,
        current_timestamp_utc: str,
        trigger_kinds: list[str],
        max_attempts: int,
        limit: int,
    ):
        calls.append({
            "current_timestamp_utc": current_timestamp_utc,
            "trigger_kinds": trigger_kinds,
            "max_attempts": max_attempts,
            "limit": limit,
        })
        return [
            {
                "run_id": "run-1",
                "schedule_id": "schedule-1",
                "trigger_kind": models.TRIGGER_FUTURE_COGNITION,
                "status": models.RUN_STATUS_PENDING,
                "due_at": "2026-06-17T00:00:00+00:00",
                "attempt_count": 1,
                "max_attempts": 3,
                "lease_owner": "worker-1",
                "lease_expires_at": "2026-06-17T00:05:00+00:00",
                "result_summary": {"case_id": "case-1"},
                "failure_summary": {"reason": "safe bounded reason"},
                "payload": {"global_user_id": "must-not-leak"},
                "source_scope": {"source_channel_id": "must-not-leak"},
                "idempotency_key": "must-not-leak",
            }
        ]

    repository = ControlConsoleRepository(
        list_due_calendar_runs=list_due_calendar_runs,
    )

    page = await repository.lookup_due_calendar_runs(
        current_timestamp_utc="2026-06-17T00:10:00+00:00",
        limit=5,
    )

    assert calls == [{
        "current_timestamp_utc": "2026-06-17T00:10:00+00:00",
        "trigger_kinds": sorted(models.CALENDAR_TRIGGER_KINDS),
        "max_attempts": models.DEFAULT_RUN_MAX_ATTEMPTS,
        "limit": 5,
    }]
    assert page["status"] == "available"
    assert page["items"] == [{
        "run_id": "run-1",
        "schedule_id": "schedule-1",
        "trigger_kind": models.TRIGGER_FUTURE_COGNITION,
        "status": models.RUN_STATUS_PENDING,
        "due_at": "2026-06-17T00:00:00+00:00",
        "attempt_count": 1,
        "max_attempts": 3,
        "lease_owner": "worker-1",
        "lease_expires_at": "2026-06-17T00:05:00+00:00",
        "result_summary": {"case_id": "case-1"},
        "failure_summary": {"reason": "safe bounded reason"},
    }]
    rendered = repr(page)
    assert "global_user_id" not in rendered
    assert "source_channel_id" not in rendered
    assert "must-not-leak" not in rendered
