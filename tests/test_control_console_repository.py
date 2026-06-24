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
        find_user_profile_by_identifier=unavailable_helper,
    )

    character = await repository.latest_character_status()
    growth = await repository.global_growth_summary()
    memory = await repository.lookup_memory(
        platform="qq",
        platform_user_id="platform-user-1",
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
                "trait_name": "concrete follow-up",
                "guidance": "ask one concrete follow-up when intent is vague",
                "strength": 0.5,
                "status": "active",
                "maturity_band": "emerging",
                "evidence_count": 2,
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
    assert growth["items"][0]["trait_name"] == "concrete follow-up"
    assert growth["items"][0]["guidance"] == (
        "ask one concrete follow-up when intent is vague"
    )
    assert growth["items"][0]["maturity_band"] == "emerging"
    assert growth["items"][0]["evidence_count"] == 2
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

    async def find_user_profile_by_identifier(
        *,
        identifier: str,
        platform: str | None = None,
    ) -> dict[str, object] | None:
        assert identifier == "platform-user-1"
        assert platform == "qq"
        return {
            "global_user_id": "global-user-1",
            "platform_accounts": [
                {
                    "platform": "qq",
                    "platform_user_id": "platform-user-1",
                    "display_name": "Tester",
                }
            ],
        }

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
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        query_user_memory_units=query_user_memory_units,
        search_user_memory_units_by_keyword=search_user_memory_units_by_keyword,
    )

    recent = await repository.lookup_memory(
        platform="qq",
        platform_user_id="platform-user-1",
        query="",
        limit=5,
    )
    keyword = await repository.lookup_memory(
        platform="qq",
        platform_user_id="platform-user-1",
        query="reviews",
        limit=3,
    )

    assert calls == [
        {"helper": "recent", "global_user_id": "global-user-1", "limit": 5},
        {
            "helper": "keyword",
            "global_user_id": "global-user-1",
            "keyword": "reviews",
            "limit": 3,
        },
    ]
    assert recent["status"] == "available"
    assert recent["identity"] == {
        "platform": "qq",
        "platform_user_id": "platform-user-1",
        "display_name": "Tester",
        "resolution_status": "resolved",
    }
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
async def test_repository_composes_owner_entity_envelopes() -> None:
    """Owner pages should group profile, memory, and style by semantic owner."""

    from control_console.repository import ControlConsoleRepository

    async def get_character_profile():
        return {
            "name": "Test Character",
            "self_image": {
                "historical_summary": "quiet precision under pressure",
                "recent_window": [
                    {
                        "timestamp": "2026-06-19T00:00:00+00:00",
                        "summary": "accepted direct operator review",
                    },
                ],
                "meta": {
                    "last_updated": "2026-06-19T00:00:00+00:00",
                },
            },
            "prompt_text": "must redact",
        }

    async def get_character_runtime_state():
        return {
            "mood": "focused",
            "global_vibe": "steady",
            "reflection_summary": "short safe summary",
        }

    async def list_growth_traits(*, limit: int):
        assert limit == 12
        return [{
            "trait_id": "trait-1",
            "growth_axis": "operator trust",
            "trait_name": "direct-review calibration",
            "guidance": "treat direct review as a high-signal correction",
            "strength": 0.75,
            "status": "active",
            "maturity_band": "observed",
            "evidence_count": 3,
        }]

    async def find_user_profile_by_identifier(
        *,
        identifier: str,
        platform: str | None = None,
    ) -> dict[str, object] | None:
        assert identifier == "platform-user-1"
        assert platform == "qq"
        return {
            "global_user_id": "global-user-1",
            "affinity": 742,
            "relationship_summary": "trusts direct review",
            "platform_accounts": [
                {
                    "platform": "qq",
                    "platform_user_id": "platform-user-1",
                    "display_name": "Tester",
                }
            ],
            "raw_prompt": "must redact",
        }

    async def query_user_memory_units(global_user_id: str, *, limit: int):
        assert global_user_id == "global-user-1"
        assert limit == 5
        return [{
            "unit_id": "unit-1",
            "unit_type": "stable_pattern",
            "status": "active",
            "fact": "User wants product-grade UI checks.",
            "embedding": [0.1],
        }]

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

    async def build_interaction_style_context(
        *,
        global_user_id: str,
        channel_type: str,
        platform: str,
        platform_channel_id: str,
    ):
        assert platform == "qq"
        if channel_type == "private":
            assert global_user_id == "global-user-1"
            assert platform_channel_id == ""
            return {
                "application_order": ["user_style"],
                "user_style": {
                    "speech_guidelines": ["be concrete"],
                    "confidence": "high",
                },
            }

        assert global_user_id == ""
        assert channel_type == "group"
        assert platform_channel_id == "group-1"
        return {
            "application_order": ["group_channel_style"],
            "group_channel_style": {
                "social_guidelines": ["keep shared context concise"],
                "confidence": "medium",
            },
        }

    repository = ControlConsoleRepository(
        get_character_profile=get_character_profile,
        get_character_runtime_state=get_character_runtime_state,
        list_growth_traits=list_growth_traits,
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        query_user_memory_units=query_user_memory_units,
        search_user_memory_units_by_keyword=search_user_memory_units_by_keyword,
        build_interaction_style_context=build_interaction_style_context,
    )

    character = await repository.character_entity(limit=5)
    user = await repository.lookup_user_entity(
        platform="qq",
        platform_user_id="platform-user-1",
        query="",
        limit=5,
    )
    group = await repository.lookup_group_entity(
        platform="qq",
        group_id="group-1",
        limit=5,
    )

    assert character["owner"] == "character"
    assert character["status"] == "available"
    assert character["panels"]["profile"]["items"][0]["name"] == "Test Character"
    self_image = character["panels"]["self_image"]["items"][0]
    assert self_image["historical_summary"] == "quiet precision under pressure"
    assert self_image["recent_window"][0]["summary"] == (
        "accepted direct operator review"
    )
    assert self_image["last_updated"] == "2026-06-19T00:00:00+00:00"
    assert character["panels"]["state"]["items"][0]["key"] == "mood"
    state_keys = {
        item["key"]
        for item in character["panels"]["state"]["items"]
    }
    assert "reflection_summary" not in state_keys
    assert character["panels"]["growth"]["items"][0]["trait_id"] == "trait-1"
    assert character["panels"]["growth"]["items"][0]["trait_name"] == (
        "direct-review calibration"
    )
    assert character["panels"]["growth"]["items"][0]["guidance"] == (
        "treat direct review as a high-signal correction"
    )
    assert "learning" in character["panels"]
    assert character["panels"]["learning"]["items"][0]["summary"] == (
        "short safe summary"
    )

    assert user["owner"] == "user"
    assert user["identity"]["display_name"] == "Tester"
    assert "global_user_id" not in user["identity"]
    user_profile = user["panels"]["profile"]["items"][0]
    assert user_profile == {
        "platform": "qq",
        "platform_user_id": "platform-user-1",
        "display_name": "Tester",
    }
    relationship_values = {
        item["key"]: item["value"]
        for item in user["panels"]["relationship"]["items"]
    }
    assert relationship_values["affinity"] == 742
    assert relationship_values["relationship_summary"] == "trusts direct review"
    assert user["panels"]["memory"]["items"][0]["unit_id"] == "unit-1"
    assert user["panels"]["style"]["items"][0]["scope"] == "user_style"

    assert group["owner"] == "group"
    assert group["identity"] == {"platform": "qq", "group_id": "group-1"}
    assert group["panels"]["style"]["items"][0]["scope"] == "group_channel_style"
    assert "progress" in group["panels"]
    assert "guidance" in group["panels"]

    rendered = repr({"character": character, "user": user, "group": group})
    assert "global-user-1" not in rendered
    assert "embedding" not in rendered
    assert "prompt_text" not in rendered
    assert "raw_prompt" not in rendered
    assert "must redact" not in rendered


@pytest.mark.asyncio
async def test_repository_memory_lookup_requires_platform_user_id() -> None:
    """Memory lookup should not query all users from a blank platform account."""

    from control_console.repository import ControlConsoleRepository

    repository = ControlConsoleRepository()

    page = await repository.lookup_memory(
        platform="qq",
        platform_user_id="",
        query="anything",
        limit=5,
    )

    assert page["status"] == "needs_input"
    assert page["items"] == []
    assert "platform user id" in page["reason"]


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

    async def find_user_profile_by_identifier(
        *,
        identifier: str,
        platform: str | None = None,
    ) -> dict[str, str] | None:
        assert identifier == "platform-user-1"
        assert platform == "qq"
        return {"global_user_id": "global-user-1"}

    repository = ControlConsoleRepository(
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        query_user_memory_units=query_user_memory_units,
        search_user_memory_units_by_keyword=search_user_memory_units_by_keyword,
    )

    page = await repository.lookup_memory(
        platform="qq",
        platform_user_id="platform-user-1",
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

    async def find_user_profile_by_identifier(
        *,
        identifier: str,
        platform: str | None = None,
    ) -> dict[str, object] | None:
        assert identifier == "platform-user-1"
        assert platform == "debug"
        return {
            "global_user_id": "global-user-1",
            "platform_accounts": [
                {
                    "platform": "debug",
                    "platform_user_id": "platform-user-1",
                    "display_name": "Debug User",
                }
            ],
        }

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
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        build_interaction_style_context=build_interaction_style_context,
    )

    page = await repository.lookup_interaction_style(
        platform="debug",
        platform_user_id="platform-user-1",
        platform_channel_id="group-1",
    )

    assert calls == [
        {
            "global_user_id": "global-user-1",
            "channel_type": "group",
            "platform": "debug",
            "platform_channel_id": "group-1",
        }
    ]
    assert page["status"] == "available"
    assert page["identity"] == {
        "platform": "debug",
        "platform_user_id": "platform-user-1",
        "display_name": "Debug User",
        "resolution_status": "resolved",
    }
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
        platform="",
        platform_user_id="",
        platform_channel_id="",
    )

    assert page["status"] == "needs_input"
    assert page["items"] == []
    assert "platform user id" in page["reason"]


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


@pytest.mark.asyncio
async def test_repository_fallback_and_empty_branches_are_explicit() -> None:
    """Repository fallback branches should return explicit product states."""

    from control_console.repository import ControlConsoleRepository

    async def invalid_character_profile():
        return ["invalid"]

    async def blank_character_profile():
        return {"name": "   "}

    async def empty_character_runtime_state():
        return {}

    async def empty_growth_traits(*, limit: int):
        assert limit == 12
        return []

    async def memory_import_failure(*args, **kwargs):
        del args, kwargs
        raise ImportError("memory helper missing")

    async def calendar_failure(**kwargs):
        del kwargs
        raise ValueError("calendar helper invalid")

    async def style_failure(**kwargs):
        del kwargs
        raise KeyError("style helper unavailable")

    async def find_user_profile_by_identifier(
        *,
        identifier: str,
        platform: str | None = None,
    ) -> dict[str, str] | None:
        assert identifier == "platform-user-1"
        assert platform in {"qq", "debug"}
        return {"global_user_id": "global-user-1"}

    invalid_identity = await ControlConsoleRepository(
        get_character_profile=invalid_character_profile,
    ).application_identity()
    blank_identity = await ControlConsoleRepository(
        get_character_profile=blank_character_profile,
    ).application_identity()
    empty_character = await ControlConsoleRepository(
        get_character_runtime_state=empty_character_runtime_state,
    ).latest_character_status()
    empty_growth = await ControlConsoleRepository(
        list_growth_traits=empty_growth_traits,
    ).global_growth_summary()
    memory_page = await ControlConsoleRepository(
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        query_user_memory_units=memory_import_failure,
        search_user_memory_units_by_keyword=memory_import_failure,
    ).lookup_memory(
        platform="qq",
        platform_user_id="platform-user-1",
        query="keyword",
        limit=5,
    )
    calendar_page = await ControlConsoleRepository(
        list_due_calendar_runs=calendar_failure,
    ).lookup_due_calendar_runs(
        current_timestamp_utc="2026-06-17T00:00:00+00:00",
        limit=5,
    )
    style_missing_platform = await ControlConsoleRepository().lookup_interaction_style(
        platform="",
        platform_user_id="",
        platform_channel_id="group-1",
    )
    style_unavailable = await ControlConsoleRepository(
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        build_interaction_style_context=style_failure,
    ).lookup_interaction_style(
        platform="debug",
        platform_user_id="platform-user-1",
        platform_channel_id="",
    )

    assert invalid_identity["status"] == "unavailable"
    assert "invalid data" in invalid_identity["reason"]
    assert blank_identity["status"] == "empty"
    assert empty_character["status"] == "empty"
    assert empty_growth["status"] == "empty"
    assert memory_page["status"] == "unavailable"
    assert "memory helper missing" in memory_page["reason"]
    assert calendar_page["status"] == "unavailable"
    assert "calendar helper invalid" in calendar_page["reason"]
    assert style_missing_platform["status"] == "needs_input"
    assert "platform is required" in style_missing_platform["reason"]
    assert style_unavailable["status"] == "unavailable"
    assert "style helper unavailable" in style_unavailable["reason"]


def test_repository_style_projection_skips_invalid_entries_and_limits_rows() -> None:
    """Style projection should skip malformed overlays and stop at the limit."""

    from control_console.repository import _project_interaction_style_context

    rows = _project_interaction_style_context(
        {
            "application_order": [
                123,
                "missing",
                "bad_overlay",
                "valid",
            ],
            "bad_overlay": ["not a dict"],
            "valid": {
                "speech_guidelines": ["one", "two"],
                "social_guidelines": ["three"],
                "confidence": "high",
            },
        },
        limit=1,
    )

    assert rows == [{
        "scope": "valid",
        "field": "speech_guidelines",
        "guidelines": ["one"],
        "confidence": "high",
    }]
