"""Focused tests for cognition-debug control-console panels."""

from __future__ import annotations

import re
from typing import Any

import pytest


def _authenticated_client(tmp_path):
    """Create an authenticated control-console client for route tests."""

    from fastapi.testclient import TestClient

    from control_console.app import create_app
    from control_console.auth import hash_operator_token
    from control_console.settings import ControlConsoleSettings

    settings = ControlConsoleSettings(
        state_dir=tmp_path,
        operator_token_hash=hash_operator_token("secret"),
    )
    client = TestClient(create_app(settings=settings))
    login = client.post("/api/auth/login", json={"token": "secret"})
    assert login.status_code == 200
    return client


def test_cognition_debug_routes_pass_exact_scope_to_repository(
    monkeypatch,
    tmp_path,
) -> None:
    """Routes should pass operator-selected runtime scope to repository helpers."""

    from control_console import app as app_module
    from control_console import repository as repository_module

    captured: dict[str, Any] = {}

    async def read_kazusa_events(query):
        assert query.service_id == "background_work.worker"
        return [{"event_id": "worker-event-1"}]

    async def lookup_calendar(self, **kwargs):
        _ = self
        captured["calendar"] = kwargs
        return {
            "status": "available",
            "panels": {
                "cognition_pending_runs": {"items": [], "prompt_view": True},
            },
        }

    async def lookup_background_work(self, **kwargs):
        _ = self
        captured["background"] = kwargs
        return {
            "status": "available",
            "panels": {
                "result_ready_cognition_deliveries": {
                    "items": [],
                    "prompt_view": True,
                },
            },
        }

    async def lookup_user_entity(self, **kwargs):
        _ = self
        captured["user"] = kwargs
        return {
            "status": "available",
            "owner": "user",
            "panels": {
                "conversation_progress_prompt": {
                    "content": {},
                    "prompt_view": True,
                },
            },
        }

    async def lookup_group_entity(self, **kwargs):
        _ = self
        captured["group"] = kwargs
        return {
            "status": "available",
            "owner": "group",
            "panels": {
                "group_carry_over": {"content": "", "prompt_view": True},
            },
        }

    async def character_entity(self, **kwargs):
        _ = self
        captured["character"] = kwargs
        return {
            "status": "available",
            "owner": "character",
            "panels": {
                "promoted_global_growth_prompt": {
                    "content": {},
                    "prompt_view": True,
                },
            },
        }

    monkeypatch.setattr(app_module, "_read_kazusa_events", read_kazusa_events)
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "lookup_calendar",
        lookup_calendar,
        raising=False,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "lookup_background_work",
        lookup_background_work,
        raising=False,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "lookup_user_entity",
        lookup_user_entity,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "lookup_group_entity",
        lookup_group_entity,
    )
    monkeypatch.setattr(
        repository_module.ControlConsoleRepository,
        "character_entity",
        character_entity,
    )

    client = _authenticated_client(tmp_path)

    calendar = client.get(
        "/api/lookups/calendar"
        "?platform=qq&platform_channel_id=group-1"
        "&platform_user_id=platform-user-1&channel_type=group&limit=4",
    )
    background = client.get("/api/lookups/background-work?limit=6")
    user = client.get(
        "/api/entities/users/qq/platform-user-1"
        "?platform_channel_id=group-1&channel_type=group&query=debug&limit=7",
    )
    group = client.get(
        "/api/entities/groups/qq/group-1"
        "?participant_platform_user_id=platform-user-1&limit=8",
    )
    character = client.get("/api/entities/character?limit=9")

    assert calendar.status_code == 200
    assert background.status_code == 200
    assert user.status_code == 200
    assert group.status_code == 200
    assert character.status_code == 200
    assert captured["calendar"]["platform"] == "qq"
    assert captured["calendar"]["platform_channel_id"] == "group-1"
    assert captured["calendar"]["platform_user_id"] == "platform-user-1"
    assert captured["calendar"]["channel_type"] == "group"
    assert captured["calendar"]["limit"] == 4
    assert captured["background"]["worker_event_rows"] == [
        {"event_id": "worker-event-1"},
    ]
    assert captured["background"]["limit"] == 6
    assert captured["user"]["platform_channel_id"] == "group-1"
    assert captured["user"]["channel_type"] == "group"
    assert captured["user"]["query"] == "debug"
    assert captured["user"]["limit"] == 7
    assert captured["group"]["participant_platform_user_id"] == "platform-user-1"
    assert captured["group"]["limit"] == 8
    assert captured["character"]["limit"] == 9
    assert captured["character"]["current_timestamp_utc"]


def test_static_surface_exposes_semantic_v2_owner_panels(tmp_path) -> None:
    """Static UI should expose owner information without prompt internals."""

    client = _authenticated_client(tmp_path)

    index = client.get("/")
    assert index.status_code == 200
    html = index.text
    assert 'id="user-platform-channel-id"' in html
    assert 'id="user-channel-type"' in html
    assert 'id="user-conversation-progress-table"' in html
    assert 'id="user-carry-over-table"' in html
    assert 'id="group-participant-platform-user-id"' in html
    assert 'id="group-carry-over-table"' in html
    assert 'id="group-participant-progress-table"' in html
    assert 'id="calendar-platform"' in html
    assert 'id="calendar-platform-channel-id"' in html
    assert 'id="calendar-platform-user-id"' in html
    assert 'id="calendar-channel-type"' in html
    assert 'id="calendar-summary-table"' in html
    assert 'id="calendar-schedules-table"' in html
    assert 'id="calendar-runs-table"' in html
    assert 'id="calendar-cognition-visibility-table"' in html
    assert 'id="background-summary-table"' in html
    assert 'id="background-jobs-table"' in html
    assert 'id="background-worker-table"' in html
    assert 'id="background-errors-table"' in html
    assert 'id="background-delivery-table"' in html
    assert 'id="character-cognition-state-table"' in html
    assert 'id="character-growth-table"' in html
    assert 'id="character-carry-over-table"' in html
    assert 'class="card-content record-list"' in html
    high_volume_targets = [
        "user-memory-table",
        "calendar-schedules-table",
        "calendar-runs-table",
        "background-jobs-table",
        "background-worker-table",
    ]
    for target in high_volume_targets:
        assert re.search(
            rf'<article class="card span-two"[^>]*>(?:(?!</article>).)*id="{target}"',
            html,
            re.DOTALL,
        )
    assert "<thead>" in html
    assert "Brain stopped" not in html
    assert 'data-component="Card"' in html
    assert 'class="input"' in html
    for obsolete_copy in (
        "Event stream",
        "Growth Runs Audit",
        "Prompt View",
        "Operational Backing",
        "Background work state",
    ):
        assert obsolete_copy not in html

    script = client.get("/static/console.js")
    assert script.status_code == 200
    script_text = script.text
    assert "record-card" in script_text
    assert "setSummaryMetric" in script_text
    assert "platform_channel_id" in script_text
    assert "participant_platform_user_id" in script_text
    assert "calendar-cognition-visibility-table" in script_text
    assert "background-delivery-table" in script_text
    assert "character-cognition-state-table" in script_text
    assert "/api/entities/users" in script_text
    assert "/api/entities/groups" in script_text
    assert "/api/lookups/background-work" in script_text
    for obsolete_internal in (
        "renderPromptPanel",
        "renderOperationalPanel",
        "panel_contract",
        "projection_owner",
        "scope_order",
        "scope_summary",
        "Model inputs",
        "synthesis count",
        "No current entries.",
    ):
        assert obsolete_internal not in script_text

    stylesheet = client.get("/static/console.css")
    assert stylesheet.status_code == 200
    css = stylesheet.text
    assert ".record-card" in css
    assert "background: var(--panel)" in css
    assert "color: var(--ink)" in css


def test_static_renderers_tolerate_missing_optional_panel_targets(tmp_path) -> None:
    """Panel renderers should not crash on stale or partial static shells."""

    client = _authenticated_client(tmp_path)

    script = client.get("/static/console.js")
    assert script.status_code == 200
    script_text = script.text
    guarded_renderers = [
        "renderPanelState",
        "renderLookupTable",
        "renderReadableLookupTable",
        "renderPanelEmptyContent",
        "renderCharacterProfilePanel",
        "renderCharacterSelfImagePanel",
        "renderIdentityGrowthPanel",
        "renderIdentityLineagePanel",
        "renderMemoryUnitRows",
        "renderStyleOverlayRows",
    ]

    for function_name in guarded_renderers:
        marker = f"function {function_name}"
        function_start = script_text.index(marker)
        next_function = script_text.find("\nfunction ", function_start + 1)
        function_body = script_text[function_start:next_function]
        assert "if (!element) return;" in function_body


def test_static_renderers_do_not_write_inner_html_through_raw_selectors(
    tmp_path,
) -> None:
    """Render output writes should use the null-safe DOM helper."""

    client = _authenticated_client(tmp_path)

    script = client.get("/static/console.js")
    assert script.status_code == 200
    script_text = script.text

    assert "function setHtml" in script_text
    assert not re.search(r"qs\([^)]*\)\.innerHTML\s*=", script_text)
    assert not re.search(r"qs\([^)]*\)\.insertAdjacentHTML\(", script_text)


def test_static_shell_dom_access_uses_guarded_helpers(tmp_path) -> None:
    """Direct DOM property access through qs should stay behind helpers."""

    client = _authenticated_client(tmp_path)

    script = client.get("/static/console.js")
    assert script.status_code == 200
    script_text = script.text

    for helper_name in [
        "setText",
        "setClassName",
        "setHidden",
        "setDisabled",
        "setValue",
        "getValue",
        "isChecked",
        "bind",
    ]:
        assert f"function {helper_name}" in script_text

    assert not re.search(
        r"qs\([^)]*\)\."
        r"(textContent|className|hidden|value|checked|disabled|"
        r"addEventListener|scrollTop|scrollHeight|placeholder)\b",
        script_text,
    )
    assert 'getValue("#event-source", "all") || "all"' in script_text


@pytest.mark.asyncio
async def test_calendar_lookup_uses_semantic_schedule_and_visibility_panels() -> None:
    """Calendar should separate schedule/run state from scoped visibility."""

    from control_console.repository import ControlConsoleRepository

    calls: list[dict[str, Any]] = []

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
                },
            ],
        }

    async def collect_calendar_pending_runs(context: dict[str, Any]):
        calls.append({"collector_context": dict(context)})
        return [
            {
                "source": "calendar_runs",
                "claim": "Pending calendar future cognition at 2026-06-25T00:00:00+00:00: follow up",
                "temporal_scope": "pending_future_action",
                "lifecycle_status": "pending",
                "evidence_time": "2026-06-25T00:00:00+00:00",
                "authority": "supporting",
            },
        ]

    async def list_calendar_schedules(*, limit: int):
        assert limit == 5
        return [
            {
                "schedule_id": "schedule-1",
                "trigger_kind": "future_cognition",
                "status": "active",
                "next_run_at": "2026-06-25T00:00:00+00:00",
                "source_scope": {
                    "source_platform": "qq",
                    "source_channel_type": "group",
                    "source_channel_id": "must-not-leak",
                },
                "payload": {"global_user_id": "must-not-leak"},
                "idempotency_key": "must-not-leak",
                "recurrence": {"kind": "once"},
            },
        ]

    async def list_recent_calendar_runs(*, limit: int):
        assert limit == 5
        return []

    repository = ControlConsoleRepository(
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        collect_calendar_pending_runs=collect_calendar_pending_runs,
        list_calendar_schedules=list_calendar_schedules,
        list_recent_calendar_runs=list_recent_calendar_runs,
    )

    page = await repository.lookup_calendar(
        platform="qq",
        platform_channel_id="group-1",
        platform_user_id="platform-user-1",
        channel_type="group",
        current_timestamp_utc="2026-06-24T00:00:00+00:00",
        limit=5,
    )

    assert calls == [
        {
            "collector_context": {
                "platform": "qq",
                "platform_channel_id": "group-1",
                "global_user_id": "global-user-1",
                "current_timestamp_utc": "2026-06-24T00:00:00+00:00",
            },
        },
    ]
    panels = page["panels"]
    assert set(panels) == {
        "summary",
        "schedules",
        "runs",
        "cognition_visibility",
    }
    visibility_panel = panels["cognition_visibility"]
    assert visibility_panel["status"] == "available"
    assert visibility_panel["items"][0]["source"] == "calendar_runs"
    assert visibility_panel["items"][0]["claim"].startswith("Pending calendar")
    schedule_panel = panels["schedules"]
    assert schedule_panel["items"] == [
        {
            "trigger_kind": "future_cognition",
            "status": "active",
            "next_run_at": "2026-06-25T00:00:00+00:00",
            "source_platform": "qq",
            "source_channel_type": "group",
            "recurrence": {"kind": "once"},
        },
    ]
    assert panels["runs"]["status"] == "empty"
    assert panels["runs"]["items"] == []
    rendered = repr(page)
    assert "must-not-leak" not in rendered
    assert "global-user-1" not in rendered
    assert "panel_contract" not in rendered
    assert "projection_owner" not in rendered
    assert "scope_summary" not in rendered


@pytest.mark.asyncio
async def test_background_lookup_separates_jobs_and_delivery_detail() -> None:
    """Background lookup should avoid duplicating jobs as prompt episodes."""

    from control_console.repository import ControlConsoleRepository

    async def find_deliverable_background_work_jobs(*, limit: int):
        assert limit == 3
        return [
            {
                "job_id": "job-1",
                "status": "completed",
                "delivery_state": "ready",
                "task_brief": "summarize the benchmark notes",
                "source_context": "must-not-leak",
                "artifact_text": "model-visible artifact",
                "result_summary": "summary ready",
                "source_platform": "qq",
                "source_channel_type": "private",
                "updated_at": "2026-06-24T00:00:00+00:00",
                "idempotency_key": "must-not-leak",
            },
        ]

    async def list_recent_background_work_jobs(*, limit: int):
        assert limit == 3
        return [
            {
                "job_id": "job-2",
                "status": "queued",
                "delivery_state": "queued",
                "task_brief": "must-not-leak",
                "result_summary": "waiting",
                "artifact_text": "must-not-leak",
                "artifact_char_count": 0,
                "updated_at": "2026-06-24T00:00:00+00:00",
            },
        ]

    repository = ControlConsoleRepository(
        find_deliverable_background_work_jobs=find_deliverable_background_work_jobs,
        list_recent_background_work_jobs=list_recent_background_work_jobs,
    )

    page = await repository.lookup_background_work(
        worker_event_rows=[],
        limit=3,
    )

    panels = page["panels"]
    assert set(panels) == {
        "summary",
        "jobs",
        "worker_activity",
        "errors",
        "delivery_detail",
    }
    assert panels["jobs"]["items"][0]["status"] == "queued"
    assert panels["delivery_detail"]["items"][0] == {
        "status": "completed",
        "delivery_state": "ready",
        "updated_at": "2026-06-24T00:00:00+00:00",
        "result_summary": "summary ready",
        "source_platform": "qq",
        "source_channel_type": "private",
    }
    rendered = repr(page)
    assert "source_context" not in rendered
    assert "idempotency_key" not in rendered
    assert "artifact_text" not in rendered
    assert "task_brief" not in rendered
    assert "job_id" not in rendered
    assert "prompt_view" not in repr(panels)
    assert "must-not-leak" not in rendered


@pytest.mark.asyncio
async def test_background_lookup_reports_delivery_source_failure() -> None:
    """A failed delivery read should remain distinct from an empty queue."""

    from control_console.repository import ControlConsoleRepository

    async def find_deliverable_background_work_jobs(*, limit: int):
        assert limit == 2
        raise KeyError("delivery source unavailable")

    async def list_recent_background_work_jobs(*, limit: int):
        assert limit == 2
        return []

    repository = ControlConsoleRepository(
        find_deliverable_background_work_jobs=find_deliverable_background_work_jobs,
        list_recent_background_work_jobs=list_recent_background_work_jobs,
    )

    page = await repository.lookup_background_work(
        worker_event_rows=[],
        limit=2,
    )

    panel = page["panels"]["delivery_detail"]
    assert panel["status"] == "unavailable"
    assert panel["items"] == []
    assert "delivery source unavailable" in panel["reason"]


@pytest.mark.asyncio
async def test_user_entity_shows_scoped_progress_and_carry_over() -> None:
    """User continuity panels should preserve meaning without scope internals."""

    from control_console.repository import ControlConsoleRepository

    progress_calls: list[dict[str, str]] = []
    residue_calls: list[dict[str, Any]] = []
    style_calls: list[dict[str, str]] = []
    prompt_doc = {
        "status": "active",
        "episode_label": "current",
        "continuity": "same_thread",
        "turn_count": 4,
        "conversation_mode": "debug",
        "episode_phase": "working",
        "topic_momentum": "steady",
        "current_thread": "Console debugging.",
        "user_goal": "Verify cognition chain.",
        "current_blocker": "",
        "user_state_updates": [],
        "assistant_moves": [],
        "overused_moves": [],
        "open_loops": [],
        "resolved_threads": [],
        "avoid_reopening": [],
        "emotional_trajectory": "steady",
        "next_affordances": ["show the next grounded step"],
        "progression_guidance": "Stay concrete.",
    }

    async def get_character_profile():
        return {"name": "Test Character", "global_user_id": "character-1"}

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
                },
            ],
        }

    async def load_progress_context(
        *,
        scope,
        current_timestamp_utc: str,
        platform_bot_id: str,
        active_turn_conversation_row_ids: list[str],
    ):
        assert platform_bot_id == "character-platform-user"
        assert active_turn_conversation_row_ids == []
        progress_calls.append({
            "platform": scope.platform,
            "platform_channel_id": scope.platform_channel_id,
            "global_user_id": scope.global_user_id,
            "current_timestamp_utc": current_timestamp_utc,
            "platform_bot_id": platform_bot_id,
            "active_turn_conversation_row_ids": (
                active_turn_conversation_row_ids
            ),
        })
        return {
            "episode_state": {"last_user_input": "must-not-leak"},
            "conversation_progress": prompt_doc,
            "source": "db",
        }

    async def load_residue_context(*, trigger_scope, current_timestamp_utc: str):
        residue_calls.append({
            "trigger_scope": dict(trigger_scope),
            "current_timestamp_utc": current_timestamp_utc,
        })
        return {
            "internal_monologue_residue_context": "约1分钟前: still thinking about debug state.",
            "selected_count": 1,
            "candidate_count": 2,
            "scope_order": ["user_thread"],
            "status": "loaded",
        }

    async def get_character_user_profile(global_user_id: str):
        assert global_user_id == "character-1"
        return {
            "platform_accounts": [{
                "platform": "qq",
                "platform_user_id": "character-platform-user",
            }],
        }

    async def query_user_memory_units(global_user_id: str, *, limit: int):
        assert global_user_id == "global-user-1"
        assert limit == 5
        return []

    async def search_user_memory_units_by_keyword(*args, **kwargs):
        _ = args
        _ = kwargs
        return []

    async def build_interaction_style_context(**kwargs):
        style_calls.append(dict(kwargs))
        return {"application_order": []}

    repository = ControlConsoleRepository(
        get_character_profile=get_character_profile,
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        query_user_memory_units=query_user_memory_units,
        search_user_memory_units_by_keyword=(
            search_user_memory_units_by_keyword
        ),
        build_interaction_style_context=build_interaction_style_context,
        get_character_user_profile=get_character_user_profile,
        load_progress_context=load_progress_context,
        load_residue_context=load_residue_context,
    )

    page = await repository.lookup_user_entity(
        platform="qq",
        platform_user_id="platform-user-1",
        platform_channel_id="group-1",
        channel_type="group",
        query="",
        current_timestamp_utc="2026-06-24T00:00:00+00:00",
        limit=5,
    )

    assert progress_calls == [
        {
            "platform": "qq",
            "platform_channel_id": "group-1",
            "global_user_id": "global-user-1",
            "current_timestamp_utc": "2026-06-24T00:00:00+00:00",
            "platform_bot_id": "character-platform-user",
            "active_turn_conversation_row_ids": [],
        },
    ]
    assert residue_calls[0]["trigger_scope"] == {
        "character_id": "character-1",
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "global_user_id": "global-user-1",
    }
    assert style_calls == [{
        "global_user_id": "global-user-1",
        "channel_type": "group",
        "platform": "qq",
        "platform_channel_id": "group-1",
    }]
    progress_panel = page["panels"]["conversation_progress"]
    assert progress_panel["status"] == "available"
    assert progress_panel["items"][0]["source"] == "db"
    assert progress_panel["items"][0]["state"]["current_thread"] == (
        "Console debugging."
    )
    assert progress_panel["items"][0]["state"]["turn_count"] == 4
    residue_panel = page["panels"]["carry_over"]
    assert residue_panel["status"] == "available"
    assert residue_panel["items"][0] == {
        "context": "约1分钟前: still thinking about debug state.",
    }
    rendered = repr(page)
    assert "last_user_input" not in rendered
    assert "must-not-leak" not in rendered
    assert "global-user-1" not in rendered
    assert "scope_order" not in rendered
    assert "prompt_view" not in rendered


@pytest.mark.asyncio
async def test_group_entity_splits_group_residue_and_participant_progress() -> None:
    """Group entity should not fake participant progress without a user."""

    from control_console.repository import ControlConsoleRepository

    async def get_character_profile():
        return {"name": "Test Character", "global_user_id": "character-1"}

    async def find_user_profile_by_identifier(
        *,
        identifier: str,
        platform: str | None = None,
    ) -> dict[str, object] | None:
        assert identifier == "platform-user-1"
        assert platform == "qq"
        return {"global_user_id": "global-user-1"}

    async def load_progress_context(
        *,
        scope,
        current_timestamp_utc: str,
        platform_bot_id: str,
        active_turn_conversation_row_ids: list[str],
    ):
        assert scope.platform == "qq"
        assert scope.platform_channel_id == "group-1"
        assert scope.global_user_id == "global-user-1"
        assert current_timestamp_utc == "2026-06-24T00:00:00+00:00"
        assert platform_bot_id == "character-platform-user"
        assert active_turn_conversation_row_ids == []
        return {
            "episode_state": None,
            "conversation_progress": {"status": "active", "turn_count": 2},
            "source": "db",
        }

    async def get_character_user_profile(global_user_id: str):
        assert global_user_id == "character-1"
        return {
            "platform_accounts": [{
                "platform": "qq",
                "platform_user_id": "character-platform-user",
            }],
        }

    async def load_residue_context(*, trigger_scope, current_timestamp_utc: str):
        assert trigger_scope == {
            "character_id": "character-1",
            "platform": "qq",
            "platform_channel_id": "group-1",
            "channel_type": "group",
            "global_user_id": "",
        }
        assert current_timestamp_utc == "2026-06-24T00:00:00+00:00"
        return {
            "internal_monologue_residue_context": "group-scene carry-over",
            "selected_count": 1,
            "candidate_count": 1,
            "scope_order": ["group_scene"],
            "status": "loaded",
        }

    async def list_recent_group_summaries(
        *,
        limit: int,
        platform: str | None = None,
        platform_channel_id: str | None = None,
    ):
        assert limit == 1
        assert platform == "qq"
        assert platform_channel_id == "group-1"
        return [{
            "platform": "qq",
            "platform_channel_id": "group-1",
            "channel_name": "Review group",
            "last_activity_at": "2026-06-24T00:00:00+00:00",
            "message_count": 4,
            "participant_count": 2,
        }]

    async def list_group_review_windows(**kwargs):
        _ = kwargs
        return []

    async def build_interaction_style_context(**kwargs):
        _ = kwargs
        return {"application_order": []}

    repository = ControlConsoleRepository(
        get_character_profile=get_character_profile,
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        list_recent_group_summaries=list_recent_group_summaries,
        list_group_review_windows=list_group_review_windows,
        build_interaction_style_context=build_interaction_style_context,
        get_character_user_profile=get_character_user_profile,
        load_progress_context=load_progress_context,
        load_residue_context=load_residue_context,
    )

    missing_participant = await repository.lookup_group_entity(
        platform="qq",
        group_id="group-1",
        participant_platform_user_id="",
        current_timestamp_utc="2026-06-24T00:00:00+00:00",
        limit=5,
    )
    with_participant = await repository.lookup_group_entity(
        platform="qq",
        group_id="group-1",
        participant_platform_user_id="platform-user-1",
        current_timestamp_utc="2026-06-24T00:00:00+00:00",
        limit=5,
    )

    carry_over = missing_participant["panels"]["carry_over"]
    assert carry_over["items"][0]["context"] == "group-scene carry-over"
    participant_panel = missing_participant["panels"]["participant_progress"]
    assert participant_panel["status"] == "needs_input"
    assert participant_panel["items"] == []
    participant_with_data = with_participant["panels"]["participant_progress"]
    assert participant_with_data["status"] == "available"
    assert participant_with_data["items"][0]["state"] == {
        "status": "active",
        "turn_count": 2,
    }
    rendered = repr(with_participant)
    assert "scope_order" not in rendered
    assert "prompt_view" not in rendered


@pytest.mark.asyncio
async def test_character_entity_shows_growth_and_revision_carry_over() -> None:
    """Character should show redacted identity process and continuity."""

    from control_console.repository import ControlConsoleRepository
    from kazusa_ai_chatbot.cognition_core_v2.state_models import (
        build_character_production_state,
    )

    async def get_character_profile():
        return {
            "name": "Test Character",
            "global_user_id": "character-1",
            "self_image": {
                "self_concept": "steady under review",
                "current_growth_edges": ["repair after tension"],
            },
        }

    async def get_character_runtime_state():
        return {
            "cognition_state": build_character_production_state(
                updated_at="2026-06-24T00:00:00+00:00",
            ),
        }

    async def list_identity_growth_candidates(**kwargs):
        assert kwargs == {"character_id": "character-1", "limit": 5}
        return [{
            "status": "ready",
            "base_revision_number": 1,
            "change_kind": "inferred_growth",
            "proposed_changes": [{
                "path": "personality_brief.defense",
                "value_kind": "text",
                "replacement_text": "must-not-leak",
            }],
            "distinct_episode_count": 3,
            "distinct_local_dates": ["2026-06-23", "2026-06-24"],
            "source_scope_kinds": ["group", "private"],
            "fresh_post_revision_root_count": 3,
            "reversal_of_paths": [],
            "character_authorship": "inferred",
            "proposal_confidence": "high",
            "review_confidence": "high",
            "privacy_review": "low",
            "promoted_revision_number": None,
            "rejection_reason": None,
            "created_at": "2026-06-23T00:00:00+00:00",
            "updated_at": "2026-06-24T00:00:00+00:00",
            "candidate_id": "candidate-secret",
            "claimed_root_episode_ids": ["root-secret"],
        }]

    async def list_recent_identity_growth_runs(**kwargs):
        assert kwargs == {"character_id": "character-1", "limit": 5}
        return [{
            "run_kind": "episode",
            "base_revision_number": 1,
            "root_episode_ids": ["root-secret"],
            "source_evidence_count": 1,
            "attempt_count_by_stage": {"proposal": 1, "review": 1},
            "lifecycle_state": "complete",
            "disposition": "candidate_updated",
            "proposal_reason_code": "candidate_ready",
            "review_reason_code": "candidate_ready",
            "policy_reason_code": "candidate_ready",
            "persistence_reason_code": "candidate_ready",
            "promoted_revision_number": None,
            "validation_error_codes": [],
            "first_consumption": None,
            "post_commit_attempt_count": 0,
            "started_at": "2026-06-24T00:00:00+00:00",
            "completed_at": "2026-06-24T00:00:00+00:00",
            "run_id": "run-secret",
            "correlation_id": "correlation-secret",
        }]

    async def list_identity_revisions(**kwargs):
        assert kwargs == {"character_id": "character-1", "limit": 5}
        return [{
            "revision_number": 1,
            "revision_kind": "explicit_turning_point",
            "base_revision_number": 0,
            "changed_paths": ["self_image.self_concept"],
            "change_diff": [{
                "path": "self_image.self_concept",
                "value_kind": "text",
                "before": "must-not-leak",
                "after": "must-not-leak",
            }],
            "evidence_summary": "Reviewed turning point.",
            "source_scope_kinds": ["private"],
            "evidence_refs": [{
                "root_episode_id": "revision-root-secret",
                "character_local_date": "2026-06-24",
            }],
            "proposal_confidence": "high",
            "review_confidence": "high",
            "created_at": "2026-06-24T00:00:00+00:00",
            "revision_id": "revision-secret",
        }]

    async def build_identity_growth_health(**kwargs):
        assert kwargs == {"character_id": "character-1"}
        return {
            "state": "promotion_ready",
            "routed_count": 4,
            "no_change_count": 1,
            "emerging_candidate_count": 0,
            "ready_candidate_count": 1,
            "rejected_count": 0,
            "failed_count": 0,
            "promoted_count": 1,
            "consumed_count": 1,
            "latest_revision_number": 1,
            "latest_consumed_revision_number": 1,
            "latest_reason_code": "candidate_ready",
            "root_count": 3,
            "local_date_count": 2,
        }

    async def residue_must_not_load(**kwargs):
        _ = kwargs
        raise AssertionError("Character carry-over is identity lineage")

    repository = ControlConsoleRepository(
        get_character_profile=get_character_profile,
        get_character_runtime_state=get_character_runtime_state,
        list_identity_revisions=list_identity_revisions,
        list_identity_growth_candidates=list_identity_growth_candidates,
        list_recent_identity_growth_runs=list_recent_identity_growth_runs,
        build_identity_growth_health=build_identity_growth_health,
        load_residue_context=residue_must_not_load,
    )

    page = await repository.character_entity(
        current_timestamp_utc="2026-06-24T00:00:00+00:00",
        limit=5,
    )

    growth_panel = page["panels"]["growth"]
    assert growth_panel["status"] == "available"
    assert growth_panel["items"][0]["status"] == "ready"
    assert growth_panel["items"][0]["root_count"] == 3
    assert growth_panel["items"][1]["latest_reason_code"] == "candidate_ready"
    carry_over = page["panels"]["carry_over"]
    assert carry_over["items"][0]["state"] == "promotion_ready"
    assert carry_over["items"][1]["revision_number"] == 1
    assert carry_over["items"][1]["is_current"] is True
    rendered = repr(page)
    assert "run-secret" not in rendered
    assert "candidate-secret" not in rendered
    assert "revision-secret" not in rendered
    assert "root-secret" not in rendered
    assert "correlation-secret" not in rendered
    assert "prompt_view" not in rendered
    assert "must-not-leak" not in rendered
