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
    background = client.get("/api/lookups/background?limit=6")
    user = client.get(
        "/api/entities/user?platform=qq&platform_user_id=platform-user-1"
        "&platform_channel_id=group-1&channel_type=group&query=debug&limit=7",
    )
    group = client.get(
        "/api/entities/group?platform=qq&group_id=group-1"
        "&participant_platform_user_id=platform-user-1&limit=8",
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


def test_static_surface_adds_existing_widget_prompt_panels(tmp_path) -> None:
    """Static UI should expose new panels through existing console widgets."""

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
    assert 'id="calendar-prompt-runs-table"' in html
    assert 'id="calendar-schedules-table"' in html
    assert 'id="calendar-due-runs-table"' in html
    assert 'id="background-result-ready-table"' in html
    assert 'id="background-job-queue-table"' in html
    assert 'id="background-worker-events-table"' in html
    assert 'id="character-growth-prompt-table"' in html
    assert 'id="character-carry-over-table"' in html
    assert 'id="character-growth-runs-table"' in html
    assert "Prompt View" in html
    assert "Operational Backing" in html
    assert 'data-component="Card"' in html
    assert 'class="input"' in html

    script = client.get("/static/console.js")
    assert script.status_code == 200
    script_text = script.text
    assert "renderPromptPanel" in script_text
    assert "renderOperationalPanel" in script_text
    assert "panel_contract" in script_text
    assert "scope_summary" in script_text
    assert "platform_channel_id" in script_text
    assert "participant_platform_user_id" in script_text
    assert "calendar-prompt-runs-table" in script_text
    assert "background-result-ready-table" in script_text
    assert "character-growth-prompt-table" in script_text

    stylesheet = client.get("/static/console.css")
    assert stylesheet.status_code == 200
    css = stylesheet.text
    assert ".prompt-content" in css
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
        "renderPromptPanel",
        "renderOperationalPanel",
        "renderReadableLookupTable",
        "renderPanelEmptyContent",
        "renderCharacterProfilePanel",
        "renderCharacterSelfImagePanel",
        "renderCharacterGrowthPanel",
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
    assert 'getValue("#event-source", "console") || "console"' in script_text


@pytest.mark.asyncio
async def test_calendar_lookup_uses_prompt_collector_and_backing_panels() -> None:
    """Calendar lookup should separate prompt candidates from backing rows."""

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

    async def list_due_calendar_runs(**kwargs):
        assert kwargs["limit"] == 5
        return []

    repository = ControlConsoleRepository(
        find_user_profile_by_identifier=find_user_profile_by_identifier,
        collect_calendar_pending_runs=collect_calendar_pending_runs,
        list_calendar_schedules=list_calendar_schedules,
        list_due_calendar_runs=list_due_calendar_runs,
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
    prompt_panel = panels["cognition_pending_runs"]
    assert prompt_panel["prompt_view"] is True
    assert prompt_panel["projection_owner"] == "CalendarRunCollector.collect"
    assert prompt_panel["panel_contract"] == "production prompt input"
    assert prompt_panel["scope_summary"] == {
        "platform": "qq",
        "channel_type": "group",
        "has_platform_channel_id": True,
        "has_user_identifier": True,
        "user_identifier_kind": "platform_user_id",
    }
    assert prompt_panel["items"][0]["source"] == "calendar_runs"
    assert prompt_panel["items"][0]["claim"].startswith("Pending calendar")
    schedule_panel = panels["schedule_definitions"]
    assert schedule_panel["prompt_view"] is False
    assert schedule_panel["projection_owner"] == (
        "calendar_scheduler.repository.list_calendar_schedules_for_inspection"
    )
    assert schedule_panel["items"] == [
        {
            "schedule_id": "schedule-1",
            "trigger_kind": "future_cognition",
            "status": "active",
            "next_run_at": "2026-06-25T00:00:00+00:00",
            "source_platform": "qq",
            "source_channel_type": "group",
            "recurrence": {"kind": "once"},
        },
    ]
    assert panels["due_runs"]["prompt_view"] is False
    assert panels["due_runs"]["panel_contract"] == (
        "operational backing; not prompt input"
    )
    rendered = repr(page)
    assert "must-not-leak" not in rendered
    assert "global-user-1" not in rendered


@pytest.mark.asyncio
async def test_background_lookup_uses_result_ready_episode_projection() -> None:
    """Background lookup should display result-ready cognition inputs."""

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
                "source_platform": "qq",
                "source_channel_type": "private",
                "updated_at": "2026-06-24T00:00:00+00:00",
                "idempotency_key": "must-not-leak",
            },
        ]

    def build_result_ready_episode_from_job(job: dict[str, Any]):
        assert job["job_id"] == "job-1"
        return {
            "episode_id": "background_work_result_ready:job-1",
            "trigger_source": "background_work_result_ready",
            "output_mode": "visible_reply",
            "target_scope": {
                "platform": "qq",
                "platform_channel_id": "private-1",
                "channel_type": "private",
                "current_global_user_id": "must-not-leak",
            },
            "percepts": [
                {
                    "input_source": "background_work_result",
                    "content": "model-visible artifact",
                    "metadata": {
                        "task_brief": "summarize the benchmark notes",
                        "failure_summary": "",
                        "result_summary": "summary ready",
                        "worker": "text_artifact",
                        "worker_metadata": {"task_type": "summary"},
                    },
                },
            ],
        }

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
        build_result_ready_episode_from_job=build_result_ready_episode_from_job,
        list_recent_background_work_jobs=list_recent_background_work_jobs,
    )

    page = await repository.lookup_background_work(
        worker_event_rows=[],
        limit=3,
    )

    panels = page["panels"]
    prompt_panel = panels["result_ready_cognition_deliveries"]
    assert prompt_panel["prompt_view"] is True
    assert prompt_panel["projection_owner"] == (
        "background_work.result_source.build_result_ready_episode_from_job"
    )
    assert prompt_panel["items"][0]["episode_id"] == (
        "background_work_result_ready:job-1"
    )
    assert prompt_panel["items"][0]["content"] == "model-visible artifact"
    assert prompt_panel["items"][0]["metadata"]["task_brief"] == (
        "summarize the benchmark notes"
    )
    assert panels["job_queue"]["prompt_view"] is False
    assert panels["job_queue"]["items"][0]["job_id"] == "job-2"
    rendered = repr(page)
    assert "source_context" not in rendered
    assert "idempotency_key" not in rendered
    assert "must-not-leak" not in rendered


@pytest.mark.asyncio
async def test_background_lookup_handles_malformed_result_ready_jobs() -> None:
    """Malformed deliverable rows should not break the console route."""

    from control_console.repository import ControlConsoleRepository

    async def find_deliverable_background_work_jobs(*, limit: int):
        assert limit == 2
        return [{"job_id": "legacy-job-without-required-fields"}]

    def build_result_ready_episode_from_job(job: dict[str, Any]):
        assert job["job_id"] == "legacy-job-without-required-fields"
        raise KeyError("created_at")

    async def list_recent_background_work_jobs(*, limit: int):
        assert limit == 2
        return []

    repository = ControlConsoleRepository(
        find_deliverable_background_work_jobs=find_deliverable_background_work_jobs,
        build_result_ready_episode_from_job=build_result_ready_episode_from_job,
        list_recent_background_work_jobs=list_recent_background_work_jobs,
    )

    page = await repository.lookup_background_work(
        worker_event_rows=[],
        limit=2,
    )

    panel = page["panels"]["result_ready_cognition_deliveries"]
    assert panel["status"] == "unavailable"
    assert panel["items"] == []
    assert "could not be projected" in panel["reason"]


@pytest.mark.asyncio
async def test_user_entity_shows_exact_progress_and_residue_prompt_views() -> None:
    """User entity should load exact scoped cognition prompt windows."""

    from control_console.repository import ControlConsoleRepository

    progress_calls: list[dict[str, str]] = []
    residue_calls: list[dict[str, Any]] = []
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
        "next_affordances": ["show exact prompt window"],
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

    async def load_progress_context(*, scope, current_timestamp_utc: str):
        progress_calls.append({
            "platform": scope.platform,
            "platform_channel_id": scope.platform_channel_id,
            "global_user_id": scope.global_user_id,
            "current_timestamp_utc": current_timestamp_utc,
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
            "scope_order": ["user_thread", "character_global"],
            "status": "loaded",
        }

    repository = ControlConsoleRepository(
        get_character_profile=get_character_profile,
        find_user_profile_by_identifier=find_user_profile_by_identifier,
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
        },
    ]
    assert residue_calls[0]["trigger_scope"] == {
        "character_id": "character-1",
        "platform": "qq",
        "platform_channel_id": "group-1",
        "channel_type": "group",
        "global_user_id": "global-user-1",
    }
    progress_panel = page["panels"]["conversation_progress_prompt"]
    assert progress_panel["prompt_view"] is True
    assert progress_panel["content"] == prompt_doc
    assert progress_panel["source"] == "db"
    assert progress_panel["scope_summary"] == {
        "platform": "qq",
        "channel_type": "group",
        "has_platform_channel_id": True,
        "has_user_identifier": True,
        "user_identifier_kind": "resolved_global_user",
    }
    residue_panel = page["panels"]["current_carry_over"]
    assert residue_panel["prompt_view"] is True
    assert residue_panel["content"] == "约1分钟前: still thinking about debug state."
    rendered = repr(page)
    assert "last_user_input" not in rendered
    assert "must-not-leak" not in rendered
    assert "global-user-1" not in rendered


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

    async def load_progress_context(*, scope, current_timestamp_utc: str):
        assert scope.platform == "qq"
        assert scope.platform_channel_id == "group-1"
        assert scope.global_user_id == "global-user-1"
        assert current_timestamp_utc == "2026-06-24T00:00:00+00:00"
        return {
            "episode_state": None,
            "conversation_progress": {"status": "active", "turn_count": 2},
            "source": "db",
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
            "scope_order": ["group_scene", "character_global"],
            "status": "loaded",
        }

    repository = ControlConsoleRepository(
        get_character_profile=get_character_profile,
        find_user_profile_by_identifier=find_user_profile_by_identifier,
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

    assert missing_participant["panels"]["group_carry_over"]["content"] == (
        "group-scene carry-over"
    )
    participant_panel = missing_participant["panels"][
        "participant_conversation_progress_prompt"
    ]
    assert participant_panel["status"] == "needs_input"
    assert participant_panel["prompt_view"] is True
    assert with_participant["panels"][
        "participant_conversation_progress_prompt"
    ]["content"] == {"status": "active", "turn_count": 2}


@pytest.mark.asyncio
async def test_character_entity_shows_growth_context_runs_and_global_residue() -> None:
    """Character entity should expose prompt growth and backing run audit."""

    from control_console.repository import ControlConsoleRepository

    async def get_character_profile():
        return {
            "name": "Test Character",
            "global_user_id": "character-1",
        }

    async def build_global_character_growth_context():
        return {
            "global_character_growth": [
                {
                    "growth_axis": "repair",
                    "guidance": "repair quickly after tension",
                    "maturity": "promoted",
                    "updated_at": "2026-06-24",
                },
            ],
        }

    async def list_recent_global_character_growth_runs(*, limit: int):
        assert limit == 5
        return [
            {
                "run_id": "run-1",
                "status": "applied",
                "mode": "apply",
                "accepted_count": 1,
                "promoted_count": 1,
                "raw_llm_output": "must-not-leak",
                "source_memory_ids": ["must-not-leak"],
                "updated_at": "2026-06-24T00:00:00+00:00",
            },
        ]

    async def load_residue_context(*, trigger_scope, current_timestamp_utc: str):
        assert trigger_scope == {
            "character_id": "character-1",
            "platform": "",
            "platform_channel_id": "",
            "channel_type": "",
            "global_user_id": "",
        }
        assert current_timestamp_utc == "2026-06-24T00:00:00+00:00"
        return {
            "internal_monologue_residue_context": "character-global carry-over",
            "selected_count": 1,
            "candidate_count": 1,
            "scope_order": ["character_global"],
            "status": "loaded",
        }

    repository = ControlConsoleRepository(
        get_character_profile=get_character_profile,
        build_global_character_growth_context=build_global_character_growth_context,
        list_recent_global_character_growth_runs=list_recent_global_character_growth_runs,
        load_residue_context=load_residue_context,
    )

    page = await repository.character_entity(
        current_timestamp_utc="2026-06-24T00:00:00+00:00",
        limit=5,
    )

    growth_prompt = page["panels"]["promoted_global_growth_prompt"]
    assert growth_prompt["prompt_view"] is True
    assert growth_prompt["content"] == {
        "global_character_growth": [
            {
                "growth_axis": "repair",
                "guidance": "repair quickly after tension",
                "maturity": "promoted",
                "updated_at": "2026-06-24",
            },
        ],
    }
    assert page["panels"]["current_carry_over"]["content"] == (
        "character-global carry-over"
    )
    runs_panel = page["panels"]["growth_runs_audit"]
    assert runs_panel["prompt_view"] is False
    assert runs_panel["items"] == [
        {
            "run_id": "run-1",
            "status": "applied",
            "mode": "apply",
            "accepted_count": 1,
            "promoted_count": 1,
            "updated_at": "2026-06-24T00:00:00+00:00",
        },
    ]
    rendered = repr(page)
    assert "raw_llm_output" not in rendered
    assert "source_memory_ids" not in rendered
    assert "must-not-leak" not in rendered
