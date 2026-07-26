"""Read-only repository adapter tests for unavailable dependencies."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_repository_returns_safe_unavailable_summaries_without_db() -> None:
    """Owner envelopes should degrade to bounded unavailable data."""

    from control_console.repository import ControlConsoleRepository
    from kazusa_ai_chatbot.db.errors import DatabaseOperationError

    async def unavailable_helper(**kwargs):
        _ = kwargs
        raise DatabaseOperationError("db unavailable")

    async def unavailable_character_helper():
        raise DatabaseOperationError("db unavailable")

    repository = ControlConsoleRepository(
        get_character_profile=unavailable_character_helper,
        get_character_runtime_state=unavailable_character_helper,
        list_growth_traits=unavailable_helper,
        list_recent_global_character_growth_runs=unavailable_helper,
        find_user_profile_by_identifier=unavailable_helper,
        load_residue_context=unavailable_helper,
    )

    character = await repository.character_entity(limit=5)
    memory = await repository.lookup_memory(
        platform="qq",
        platform_user_id="platform-user-1",
        query="",
        limit=5,
    )

    assert character["status"] == "unavailable"
    assert all(
        panel["status"] == "unavailable"
        for panel in character["panels"].values()
    )
    assert memory["items"] == []
    assert memory["redaction"]["embeddings"] == "excluded"
    assert "prompt" not in repr(memory).lower()


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

    async def invalid_character_profile():
        return ["invalid"]

    async def blank_character_profile():
        return {"name": "   "}

    invalid_identity = await ControlConsoleRepository(
        get_character_profile=invalid_character_profile,
    ).application_identity()
    blank_identity = await ControlConsoleRepository(
        get_character_profile=blank_character_profile,
    ).application_identity()

    assert invalid_identity["status"] == "unavailable"
    assert "invalid data" in invalid_identity["reason"]
    assert blank_identity["status"] == "empty"


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
        "unit_type": "stable_pattern",
        "status": "active",
        "fact": "User likes direct technical reviews.",
        "relationship_signal": "prefers honesty",
        "subjective_appraisal": "operator trust context",
        "last_seen_at": "2026-06-17T00:00:00+00:00",
        "updated_at": "2026-06-17T00:00:00+00:00",
    }
    assert keyword["items"][0]["fact"] == (
        "User reviews every console workflow."
    )
    assert "unit_id" not in repr(recent["items"])
    assert "unit_id" not in repr(keyword["items"])
    assert "embedding" not in repr(recent["items"]).lower()
    assert "prompt" not in repr(recent["items"]).lower()
    assert "raw_message" not in repr(recent["items"]).lower()


@pytest.mark.asyncio
async def test_repository_projects_native_v2_character_and_user_state() -> None:
    """Owner pages should expose native V2 state without legacy identifiers."""

    from control_console.repository import ControlConsoleRepository
    from kazusa_ai_chatbot.cognition_core_v2.state_models import (
        build_acquaintance_user_state,
        build_character_production_state,
    )
    from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
        project_numeric_band,
    )

    updated_at = "2026-07-27T00:00:00Z"
    character_state = build_character_production_state(updated_at=updated_at)
    character_state["goals"] = [{
        "entity_id": "goal-character-secret",
        "description": "Preserve direct operator trust.",
        "status": "active",
    }]
    user_state = build_acquaintance_user_state(
        global_user_id="global-user-secret",
        updated_at=updated_at,
    )
    relationship = user_state["relationship"]
    relationship.update({
        "familiarity": 55,
        "positive_regard": 42,
        "trust": 37,
        "attachment": 31,
        "desired_closeness": 48,
        "perceived_closeness": 44,
        "care": 62,
        "boundary_safety": 25,
        "exclusivity": 8,
        "unresolved_injury": 12,
        "salience": 66,
        "evidence_refs": [{"ref_id": "relationship-evidence-secret"}],
    })
    user_state["goals"] = [{
        "entity_id": "goal-user-secret",
        "description": "Understand the operator's review standard.",
        "status": "active",
    }]
    user_state["knowledge_gaps"] = [{
        "entity_id": "gap-user-secret",
        "description": "Which console facts are highest priority?",
        "status": "active",
    }]

    async def get_character_profile():
        return {
            "name": "Test Character",
            "description": "A precise character.",
            "tone": "restrained",
            "speech_patterns": "short direct clauses",
            "backstory": "Safe operator-visible background.",
            "personality_brief": {"mbti": "ISTP"},
            "boundary_profile": {
                "self_integrity": 0.9,
                "control_sensitivity": 0.8,
            },
            "linguistic_texture_profile": {
                "fragmentation": 0.4,
                "direct_assertion": 0.7,
            },
            "self_image": {"historical_summary": "quiet precision"},
            "cognition_state": character_state,
        }

    async def get_character_runtime_state():
        return {
            "self_image": {"historical_summary": "quiet precision"},
            "cognition_state": character_state,
            "updated_at": updated_at,
        }

    async def list_growth_traits(*, limit: int):
        assert limit == 12
        return [{
            "trait_id": "growth-trait-secret",
            "growth_axis": "operator trust",
            "trait_name": "direct-review calibration",
            "guidance": "Treat concrete corrections as high-signal evidence.",
            "status": "active",
            "maturity_band": "observed",
            "evidence_count": 3,
        }]

    async def list_recent_global_character_growth_runs(*, limit: int):
        assert limit == 1
        return [
            {
                "run_id": "growth-run-secret",
                "status": "completed",
                "summary": "Accepted one direct-review calibration change.",
                "accepted_candidates": [{
                    "growth_axis": "operator trust",
                    "summary": "Direct correction improved calibration.",
                }],
                "trait_updates": [{
                    "trait_name": "direct-review calibration",
                    "change": "strengthened",
                }],
                "shadow_projection": {
                    "growth_axis": "operator trust",
                    "guidance": "Use the correction on the next review.",
                    "maturity": "observed",
                    "prompt_visible_now": True,
                    "review_note": "Semantically useful and bounded.",
                },
                "source_reflection_run_ids": ["growth-source-secret"],
            },
            {
                "run_id": "older-growth-run-secret",
                "status": "completed",
                "summary": "Older growth outcome that must stay hidden.",
            },
        ]

    async def find_user_profile_by_identifier(
        *,
        identifier: str,
        platform: str | None = None,
    ):
        assert identifier == "platform-user-1"
        assert platform == "qq"
        return {
            "global_user_id": "global-user-secret",
            "platform_accounts": [{
                "platform": "qq",
                "platform_user_id": "platform-user-1",
                "display_name": "Operator",
            }],
            "suspected_aliases": [
                "suspected-alias-secret-1",
                "suspected-alias-secret-2",
            ],
            "cognition_state": user_state,
        }

    async def query_user_memory_units(global_user_id: str, *, limit: int):
        assert global_user_id == "global-user-secret"
        assert limit == 5
        return []

    async def search_user_memory_units_by_keyword(
        global_user_id: str,
        keyword: str,
        *,
        limit: int,
    ):
        _ = global_user_id
        _ = keyword
        _ = limit
        raise AssertionError("keyword lookup should not run")

    async def build_interaction_style_context(**kwargs):
        _ = kwargs
        return {"application_order": []}

    async def load_progress_context(**kwargs):
        _ = kwargs
        return {"source": "empty", "conversation_progress": {}}

    async def load_residue_context(**kwargs):
        _ = kwargs
        return {
            "status": "empty",
            "internal_monologue_residue_context": "",
            "selected_count": 0,
            "candidate_count": 0,
            "scope_order": [],
        }

    repository = ControlConsoleRepository(
        get_character_profile=get_character_profile,
        get_character_runtime_state=get_character_runtime_state,
        list_growth_traits=list_growth_traits,
        list_recent_global_character_growth_runs=(
            list_recent_global_character_growth_runs
        ),
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        query_user_memory_units=query_user_memory_units,
        search_user_memory_units_by_keyword=search_user_memory_units_by_keyword,
        build_interaction_style_context=build_interaction_style_context,
        load_progress_context=load_progress_context,
        load_residue_context=load_residue_context,
    )

    character = await repository.character_entity(limit=5)
    user = await repository.lookup_user_entity(
        platform="qq",
        platform_user_id="platform-user-1",
        platform_channel_id="private-1",
        channel_type="private",
        query="",
        current_timestamp_utc=updated_at,
        limit=5,
    )

    assert set(character["panels"]) == {
        "profile",
        "cognition_state",
        "self_image",
        "growth",
        "carry_over",
    }
    character_profile = character["panels"]["profile"]["items"][0]
    assert character_profile["tone"] == "restrained"
    assert character_profile["speech_patterns"] == "short direct clauses"
    assert character_profile["boundary_profile"]["self_integrity"] == 0.9
    assert (
        character_profile["linguistic_texture_profile"]["direct_assertion"]
        == 0.7
    )
    character_keys = {
        item["key"]
        for item in character["panels"]["cognition_state"]["items"]
    }
    assert {
        "drives",
        "standards",
        "meaning_state",
        "goals",
        "threats",
        "active_events",
        "knowledge_gaps",
        "affect_activations",
        "updated_at",
    } <= character_keys
    growth_rendered = repr(character["panels"]["growth"])
    assert "Accepted one direct-review calibration change." in growth_rendered
    assert "prompt_visible_now" in growth_rendered
    assert "growth-run-secret" not in growth_rendered
    assert "growth-source-secret" not in growth_rendered
    assert "Older growth outcome that must stay hidden." not in growth_rendered

    assert set(user["panels"]) == {
        "profile",
        "relationship",
        "cognition_state",
        "memory",
        "style",
        "conversation_progress",
        "carry_over",
    }
    user_profile = user["panels"]["profile"]["items"][0]
    assert user_profile["alias_count"] == 2
    relationship_rows = {
        item["axis"]: item
        for item in user["panels"]["relationship"]["items"]
    }
    assert len(relationship_rows) == 11
    assert relationship_rows["trust"] == {
        "axis": "trust",
        "value": 37,
        "band": project_numeric_band(37, signed=True),
    }
    assert user["panels"]["relationship"]["evidence_count"] == 1
    assert user["panels"]["relationship"]["updated_at"] == updated_at
    user_cognition_keys = {
        item["key"]
        for item in user["panels"]["cognition_state"]["items"]
    }
    assert user_cognition_keys == {
        "goals",
        "threats",
        "active_events",
        "knowledge_gaps",
        "affect_activations",
        "updated_at",
    }
    rendered = repr({"character": character, "user": user})
    for forbidden in (
        "global-user-secret",
        "suspected-alias-secret",
        "relationship-evidence-secret",
        "goal-character-secret",
        "goal-user-secret",
        "gap-user-secret",
        "relationship_id",
        "other_user_id",
        "evidence_refs",
        "affinity",
        "relationship_summary",
    ):
        assert forbidden not in rendered


@pytest.mark.asyncio
async def test_repository_lists_safe_user_and_group_directories() -> None:
    """Owner pages should discover bounded users and groups from real owners."""

    from control_console.repository import ControlConsoleRepository
    from kazusa_ai_chatbot.cognition_core_v2.state_models import (
        build_acquaintance_user_state,
    )

    updated_at = "2026-07-27T00:00:00Z"
    user_state = build_acquaintance_user_state(
        global_user_id="global-user-secret",
        updated_at=updated_at,
    )

    async def list_recent_user_profiles(*, limit: int):
        assert limit == 5
        return [{
            "global_user_id": "global-user-secret",
            "platform_accounts": [{
                "platform": "qq",
                "platform_user_id": "platform-user-1",
                "display_name": "Operator",
            }],
            "suspected_aliases": ["suspected-alias-secret"],
            "cognition_state": user_state,
        }]

    async def list_recent_group_summaries(
        *,
        limit: int,
        platform: str | None = None,
        platform_channel_id: str | None = None,
    ):
        assert limit in {1, 5}
        if platform is not None:
            assert platform == "qq"
            assert platform_channel_id == "group-1"
        return [{
            "platform": "qq",
            "platform_channel_id": "group-1",
            "channel_name": "Review group",
            "last_activity_at": updated_at,
            "message_count": 12,
            "participant_count": 3,
        }]

    async def list_group_review_windows(
        *,
        platform: str,
        platform_channel_id: str,
        limit: int,
    ):
        assert platform == "qq"
        assert platform_channel_id == "group-1"
        assert limit == 1
        return [
            {
                "source_id": "group-review-secret",
                "case_id": "case-secret",
                "platform": "qq",
                "platform_channel_id": "group-1",
                "window_start": "2026-07-26T23:00:00Z",
                "window_end": updated_at,
                "status": "reviewed",
                "reviewed_at": updated_at,
                "selected_route": "proceed",
                "dispatch_status": "not_requested",
                "skip_reason": "coalesced_into_newer_window",
            },
            {
                "status": "reviewed",
                "reviewed_at": "2026-07-26T22:00:00Z",
            },
        ]

    async def build_interaction_style_context(**kwargs):
        _ = kwargs
        return {"application_order": []}

    async def load_residue_context(**kwargs):
        _ = kwargs
        return {
            "status": "empty",
            "internal_monologue_residue_context": "",
            "selected_count": 0,
            "candidate_count": 0,
            "scope_order": [],
        }

    async def load_progress_context(**kwargs):
        _ = kwargs
        return {"source": "empty", "conversation_progress": {}}

    async def get_character_profile():
        return {"global_user_id": "character-global"}

    repository = ControlConsoleRepository(
        get_character_profile=get_character_profile,
        list_recent_user_profiles=list_recent_user_profiles,
        list_recent_group_summaries=list_recent_group_summaries,
        list_group_review_windows=list_group_review_windows,
        build_interaction_style_context=build_interaction_style_context,
        load_residue_context=load_residue_context,
        load_progress_context=load_progress_context,
    )

    users = await repository.list_user_entities(limit=5)
    groups = await repository.list_group_entities(limit=5)
    group = await repository.lookup_group_entity(
        platform="qq",
        group_id="group-1",
        limit=5,
        current_timestamp_utc=updated_at,
    )

    assert users["status"] == "available"
    assert users["items"] == [{
        "display_name": "Operator",
        "accounts": [{
            "platform": "qq",
            "platform_user_id": "platform-user-1",
            "display_name": "Operator",
        }],
        "account_count": 1,
        "alias_count": 1,
        "updated_at": updated_at,
    }]
    assert groups["status"] == "available"
    assert groups["items"][0] == {
        "platform": "qq",
        "group_id": "group-1",
        "channel_name": "Review group",
        "last_activity_at": updated_at,
        "message_count": 12,
        "participant_count": 3,
    }
    assert set(group["panels"]) == {
        "activity",
        "review",
        "style",
        "carry_over",
        "participant_progress",
    }
    assert group["panels"]["activity"]["items"][0]["message_count"] == 12
    review_items = group["panels"]["review"]["items"]
    assert review_items == [{
        "window_start": "2026-07-26T23:00:00Z",
        "window_end": updated_at,
        "status": "reviewed",
        "reviewed_at": updated_at,
        "skip_reason": "coalesced_into_newer_window",
    }]
    rendered = repr({"users": users, "groups": groups, "group": group})
    assert "global-user-secret" not in rendered
    assert "suspected-alias-secret" not in rendered
    assert "group-review-secret" not in rendered
    assert "case-secret" not in rendered


def test_group_summary_keeps_missing_channel_name_empty() -> None:
    """An unnamed group should fall back to its source id, not string None."""

    from control_console.repository import _project_group_summary

    projected = _project_group_summary({
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_name": None,
        "last_activity_at": "2026-07-27T00:00:00Z",
        "message_count": 12,
        "participant_count": 3,
    })

    assert projected["group_id"] == "group-1"
    assert projected["channel_name"] == ""
    assert "None" not in repr(projected)


@pytest.mark.asyncio
async def test_repository_calendar_includes_recent_terminal_runs() -> None:
    """Calendar inspection should show history independently of due work."""

    from control_console.repository import ControlConsoleRepository

    async def list_calendar_schedules(*, limit: int):
        assert limit == 5
        return []

    async def list_recent_calendar_runs(*, limit: int):
        assert limit == 5
        return [{
            "run_id": "calendar-run-secret",
            "schedule_id": "calendar-schedule-secret",
            "trigger_kind": "reflection_phase_slot",
            "status": "completed",
            "due_at": "2026-07-26T23:00:00Z",
            "completed_at": "2026-07-26T23:00:05Z",
            "result_summary": {
                "status": "completed",
                "run_kind": "reflection_phase_slot",
                "processed_count": 1,
                "succeeded_count": 1,
                "failed_count": 0,
                "skipped_count": 0,
                "run_ids": ["nested-run-secret"],
            },
            "lease_owner": "worker-secret",
            "max_attempts": 3,
        }]

    repository = ControlConsoleRepository(
        list_calendar_schedules=list_calendar_schedules,
        list_recent_calendar_runs=list_recent_calendar_runs,
    )

    page = await repository.lookup_calendar(
        platform="",
        platform_channel_id="",
        platform_user_id="",
        channel_type="",
        current_timestamp_utc="2026-07-27T00:00:00Z",
        limit=5,
    )

    assert page["status"] == "available"
    assert set(page["panels"]) == {
        "summary",
        "schedules",
        "runs",
        "cognition_visibility",
    }
    assert page["panels"]["summary"]["items"][0] == {
        "active_schedules": 0,
        "upcoming": 0,
        "overdue": 0,
        "running": 0,
        "completed": 1,
        "failed": 0,
        "skipped": 0,
    }
    assert page["panels"]["runs"]["items"] == [{
        "trigger_kind": "reflection_phase_slot",
        "status": "completed",
        "due_at": "2026-07-26T23:00:00Z",
        "completed_at": "2026-07-26T23:00:05Z",
        "result_summary": {
            "processed_count": 1,
            "succeeded_count": 1,
        },
    }]
    assert page["panels"]["cognition_visibility"]["status"] == "needs_input"
    assert "prompt view" not in page["panels"]["cognition_visibility"][
        "reason"
    ].lower()
    rendered = repr(page)
    assert "calendar-run-secret" not in rendered
    assert "calendar-schedule-secret" not in rendered
    assert "nested-run-secret" not in rendered
    assert "worker-secret" not in rendered
    assert "max_attempts" not in rendered


@pytest.mark.asyncio
async def test_repository_background_work_aggregates_worker_outcomes() -> None:
    """An empty queue should retain useful bounded worker activity."""

    from control_console.repository import ControlConsoleRepository

    async def find_deliverable_background_work_jobs(*, limit: int):
        assert limit == 5
        return []

    async def list_recent_background_work_jobs(*, limit: int):
        assert limit == 5
        return []

    repository = ControlConsoleRepository(
        find_deliverable_background_work_jobs=(
            find_deliverable_background_work_jobs
        ),
        list_recent_background_work_jobs=list_recent_background_work_jobs,
    )
    worker_events = [
        {
            "source": "kazusa",
            "event_type": "tick",
            "component": "background_work.worker",
            "status": "skipped",
            "created_at": "2026-07-27T00:01:00Z",
            "processed_count": 0,
            "succeeded_count": 0,
            "failed_count": 0,
            "skipped_count": 1,
            "deferred": True,
            "defer_reason": "worker capacity reached",
            "worker_name": "background_work",
        },
        {
            "source": "kazusa",
            "event_type": "tick",
            "component": "background_work.worker",
            "status": "succeeded",
            "created_at": "2026-07-27T00:00:00Z",
            "processed_count": 2,
            "succeeded_count": 2,
            "failed_count": 0,
            "skipped_count": 0,
            "deferred": False,
            "worker_name": "background_work",
        },
    ]

    page = await repository.lookup_background_work(
        worker_event_rows=worker_events,
        limit=5,
    )

    assert page["status"] == "available"
    assert set(page["panels"]) == {
        "summary",
        "jobs",
        "worker_activity",
        "errors",
        "delivery_detail",
    }
    assert page["panels"]["summary"]["items"][0] == {
        "queued": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "delivery_ready": 0,
        "deferred": 0,
    }
    assert page["panels"]["worker_activity"]["items"] == [{
        "worker_name": "background_work",
        "event_count": 2,
        "last_status": "skipped",
        "last_created_at": "2026-07-27T00:01:00Z",
        "processed_count": 2,
        "succeeded_count": 2,
        "failed_count": 0,
        "skipped_count": 1,
        "deferred_count": 1,
        "defer_reason": "worker capacity reached",
    }]
    assert page["panels"]["errors"]["status"] == "empty"
    assert page["panels"]["delivery_detail"]["status"] == "empty"
    rendered = repr(page)
    assert "panel_contract" not in rendered
    assert "projection_owner" not in rendered


@pytest.mark.asyncio
async def test_repository_background_job_deduplicates_equal_summaries() -> None:
    """One job should not render the same semantic outcome twice."""

    from control_console.repository import ControlConsoleRepository

    repeated_summary = "No enabled worker owns this task."

    async def find_deliverable_background_work_jobs(*, limit: int):
        assert limit == 5
        return []

    async def list_recent_background_work_jobs(*, limit: int):
        assert limit == 5
        return [{
            "status": "delivered",
            "delivery_state": "delivered",
            "worker": "none",
            "result_summary": repeated_summary,
            "failure_summary": repeated_summary,
            "created_at": "2026-07-27T00:00:00Z",
        }]

    repository = ControlConsoleRepository(
        find_deliverable_background_work_jobs=(
            find_deliverable_background_work_jobs
        ),
        list_recent_background_work_jobs=list_recent_background_work_jobs,
    )
    page = await repository.lookup_background_work(
        worker_event_rows=[],
        limit=5,
    )

    job = page["panels"]["jobs"]["items"][0]
    assert job["result_summary"] == repeated_summary
    assert "failure_summary" not in job
    assert repr(page).count(repeated_summary) == 1


def test_repository_status_uses_only_declared_required_panels() -> None:
    """A successful panel must not hide another required source failure."""

    from control_console.repository import (
        _combined_panel_status,
        _owner_entity_envelope,
        _panel_lookup_page,
    )

    panels = {
        "profile": {"status": "available", "items": [{"name": "Character"}]},
        "cognition_state": {
            "status": "unavailable",
            "items": [],
            "reason": "database unavailable",
        },
    }

    assert _combined_panel_status(panels) == "partial"
    assert _combined_panel_status({
        "profile": {"status": "available"},
        "memory": {"status": "empty"},
    }) == "available"
    assert _combined_panel_status({
        "profile": {"status": "empty"},
        "memory": {"status": "empty"},
    }) == "empty"

    owner = _owner_entity_envelope(
        owner="user",
        identity={"platform": "qq", "platform_user_id": "user-1"},
        panels={
            "profile": {"status": "available", "items": [{}]},
            "relationship": {"status": "empty", "items": []},
            "cognition_state": {"status": "empty", "items": []},
            "carry_over": {"status": "unavailable", "items": []},
        },
        required_panel_names=(
            "profile",
            "relationship",
            "cognition_state",
        ),
    )
    assert owner["status"] == "available"

    owner["panels"]["cognition_state"]["status"] = "unavailable"
    required_failure = _owner_entity_envelope(
        owner="user",
        identity={"platform": "qq", "platform_user_id": "user-1"},
        panels=owner["panels"],
        required_panel_names=(
            "profile",
            "relationship",
            "cognition_state",
        ),
    )
    assert required_failure["status"] == "partial"

    lookup = _panel_lookup_page(
        namespace="calendar",
        panels={
            "summary": {"status": "available", "items": [{}]},
            "schedules": {"status": "empty", "items": []},
            "runs": {"status": "available", "items": [{}]},
            "cognition_visibility": {"status": "unavailable", "items": []},
        },
        required_panel_names=("summary", "schedules", "runs"),
    )
    assert lookup["status"] == "available"


@pytest.mark.asyncio
async def test_user_entity_requires_thread_scope_for_thread_continuity() -> None:
    """A selected account must not report unscoped thread data as empty."""

    from control_console.repository import ControlConsoleRepository

    async def find_user_profile_by_identifier(**kwargs):
        _ = kwargs
        return {
            "global_user_id": "global-user-1",
            "platform_accounts": [{
                "platform": "qq",
                "platform_user_id": "user-1",
                "display_name": "User",
            }],
            "cognition_state": {},
        }

    async def query_user_memory_units(*args, **kwargs):
        _ = args
        _ = kwargs
        return []

    async def build_interaction_style_context(**kwargs):
        assert kwargs == {
            "global_user_id": "global-user-1",
            "channel_type": "private",
            "platform": "qq",
            "platform_channel_id": "",
        }
        return {"application_order": []}

    async def must_not_load_thread_context(**kwargs):
        raise AssertionError(f"thread loader received incomplete scope: {kwargs}")

    repository = ControlConsoleRepository(
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        query_user_memory_units=query_user_memory_units,
        build_interaction_style_context=build_interaction_style_context,
        load_progress_context=must_not_load_thread_context,
        load_residue_context=must_not_load_thread_context,
    )

    page = await repository.lookup_user_entity(
        platform="qq",
        platform_user_id="user-1",
        platform_channel_id="",
        channel_type="",
        query="",
        limit=5,
    )

    progress = page["panels"]["conversation_progress"]
    carry_over = page["panels"]["carry_over"]
    assert progress["status"] == "needs_input"
    assert progress["reason"] == (
        "channel id and channel type are required for conversation progress"
    )
    assert carry_over["status"] == "needs_input"
    assert carry_over["reason"] == (
        "channel id and channel type are required for user-thread carry-over"
    )
    assert page["status"] == "available"


@pytest.mark.asyncio
async def test_console_residue_read_disables_loader_telemetry(monkeypatch) -> None:
    """Browser inspection must not create residue-load event rows."""

    from control_console import repository as repository_module

    calls: list[dict] = []

    async def load_residue_context(**kwargs):
        calls.append(kwargs)
        return {
            "status": "empty",
            "internal_monologue_residue_context": "",
        }

    monkeypatch.setattr(
        repository_module,
        "default_load_residue_context",
        load_residue_context,
    )
    repository = repository_module.ControlConsoleRepository()

    panel = await repository._residue_panel(
        trigger_scope={
            "character_id": "character-1",
            "platform": "",
            "platform_channel_id": "",
            "channel_type": "",
            "global_user_id": "",
        },
        current_timestamp_utc="2026-07-27T00:00:00+00:00",
        empty_reason="no carry-over",
    )

    assert panel["status"] == "empty"
    assert calls[0]["record_telemetry"] is False


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
