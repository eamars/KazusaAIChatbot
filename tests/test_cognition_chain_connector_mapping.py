"""Checkpoint F connector mapping tests for the canonical V2 caller."""

import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.character_identity_growth.projection import (
    identity_projection_digest,
    project_identity_for_cognition,
    project_identity_for_surface,
    projected_identity_consumer_kinds,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as connector

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    new_empty_goal_progress,
)
from tests.cognition_core_v2_test_helpers import (
    canonical_character_identity,
    canonical_episode,
)


NOW = "2026-07-14T00:00:00Z"


def _core_output() -> dict[str, object]:
    """Build the bounded output fields exercised by the commit connector."""

    replacement = build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=NOW,
    )
    return {
        "intention": {
            "selected_branch_id": "ordinary_response",
            "route": "speech",
            "intention": "acknowledge the episode",
            "target_roles": [],
            "reason": "the current episode is grounded",
        },
        "supporting_bids": [],
        "state_update": {
            "state_scope": "user",
            "owner_key": "user-1",
            "replacement_state": replacement,
            "comparison_results": [],
            "changed_paths": [],
        },
        "affect_projection": [],
        "action_requests": [],
        "resolver_requests": [],
        "goal_resolution": "answerable_now",
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
        "resolver_progress": {
            "status": "not_requested",
            "semantic_summary": "no resolver request",
        },
        "selected_bid_reason": "the current episode is grounded",
        "private_monologue": "I want to answer this clearly.",
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "平静",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "diagnostics": {},
    }


def _global_state() -> dict[str, object]:
    """Build the adapter-owned fields needed by the V2 mapper."""

    character_profile = canonical_character_identity()
    character_profile.update({
        "global_user_id": "character-1",
        "name": "千纱",
    })
    personality = character_profile["personality_brief"]
    assert isinstance(personality, dict)
    personality.update({
        "logic": "Keep present evidence and role ownership clear.",
        "defense": "Preserve boundaries and the selected intent.",
        "quirks": "Use occasional dry, characterful phrasing.",
        "taboos": "Ground scene facts in available evidence.",
    })
    episode = canonical_episode(
        episode_id="episode-1",
        current_global_user_id="global-user-1",
        content="hello",
    )
    revision = {
        "effective_identity": {
            key: value
            for key, value in character_profile.items()
            if key != "global_user_id"
        }
    }
    cognition_context = project_identity_for_cognition(revision)
    surface_context = project_identity_for_surface(revision)
    return {
        "character_profile": character_profile,
        "character_identity_revision_number": 0,
        "character_identity_context": cognition_context,
        "character_identity_surface_context": surface_context,
        "character_identity_projection_digest": identity_projection_digest(
            revision_number=0,
            cognition_context=cognition_context,
            surface_context=surface_context,
        ),
        "character_identity_consumer_kinds": (
            projected_identity_consumer_kinds(cognition_context)
        ),
        "character_identity_episode_id": "episode-1",
        "character_identity_epistemic_core_included": False,
        "character_cognition_state": build_character_production_state(
            updated_at=NOW,
        ),
        "storage_timestamp_utc": NOW,
        "user_input": "hello",
        "decontextualized_input": "hello",
        "prompt_message_context": {},
        "cognitive_episode": episode,
        "user_multimedia_input": [],
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "channel_type": "dm",
        "channel_name": "",
        "platform_message_id": "message-1",
        "platform_user_id": "platform-user-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {},
        "platform_bot_id": "bot-1",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "",
        "rag_result": {"memory_evidence": []},
        "resolver_context": "resolver_state: status=idle",
    }


def test_persona_connector_maps_one_native_user_scope() -> None:
    """The caller sends native V2 state and typed evidence to the core."""

    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        build_cognition_input_from_global_state,
    )

    payload = build_cognition_input_from_global_state(
        _global_state(),
        mutable_state=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        ),
    )

    assert payload["schema_version"] == "cognition_core_input.v2"
    assert payload["state_scope"] == "user"
    assert payload["mutable_state"]["state_scope"] == "user"
    assert payload["evidence"][0]["evidence_ref"]["source_kind"] == "episode"
    assert payload["episode"]["target_scope"]["platform_channel_id"] == (
        "channel-test"
    )
    assert payload["resolver_context"].startswith("resolver_state:")
    assert "dialog_text 的发言者" in payload["scene_context"][
        "current_user_role"
    ]
    assert "千纱" in payload["scene_context"]["character_role"]
    assert "User" in payload["scene_context"]["current_user_role"]
    assert "隐含主语" in payload["scene_context"]["character_role"]
    assert payload["character_constraints"]["personality_judgment"] == {
        "logic": "Keep present evidence and role ownership clear.",
        "defense": "Preserve boundaries and the selected intent.",
        "quirks": "Use occasional dry, characterful phrasing.",
        "taboos": "Ground scene facts in available evidence.",
    }


def test_connector_rejects_overlong_personality_judgment() -> None:
    """The connector preserves the exact bounded personality contract."""

    state = _global_state()
    character_profile = state["character_profile"]
    assert isinstance(character_profile, dict)
    personality_brief = character_profile["personality_brief"]
    assert isinstance(personality_brief, dict)
    personality_brief["logic"] = "x" * (
        connector.PERSONALITY_JUDGMENT_MAX_CHARS + 1
    )

    with pytest.raises(
        connector.CognitionExecutionError,
        match="personality logic",
    ):
        connector.build_cognition_input_from_global_state(
            state,
            mutable_state=build_acquaintance_user_state(
                global_user_id="user-1",
                updated_at=NOW,
            ),
        )


def test_connector_projects_protocol_owned_resolver_goal_progress() -> None:
    """The planner receives canonical goal state beside its text projection."""

    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        build_cognition_input_from_global_state,
    )

    state = _global_state()
    goal_progress = new_empty_goal_progress(original_goal="answer the user")
    state["resolver_state"] = {"goal_progress": goal_progress}
    payload = build_cognition_input_from_global_state(
        state,
        mutable_state=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        ),
    )

    assert payload["resolver_goal_progress"] == goal_progress


def test_connector_projects_runtime_owner_limits_into_cognition() -> None:
    """Cognition receives the same trusted owner limits as the surface."""

    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        build_cognition_input_from_global_state,
    )

    state = _global_state()
    state["action_availability_runtime"] = {
        "scheduler_status": "unavailable",
        "worker_status": {
            "accepted_task": "unavailable",
            "background_work": "unavailable",
        },
    }
    payload = build_cognition_input_from_global_state(
        state,
        mutable_state=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        ),
    )

    assert payload["runtime_capability_limits"]
    assert any(
        "future_speak" in item and "不可用" in item
        for item in payload["runtime_capability_limits"]
    )
    assert any(
        "绑定既有 coding_run_ref" in item and "待执行" in item
        for item in payload["runtime_capability_limits"]
    )


def test_connector_keeps_queue_intake_available_without_automatic_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queue-only runtime state must not block a generic accepted handover."""

    monkeypatch.setattr(
        connector,
        "BACKGROUND_WORK_WORKER_ENABLED",
        False,
    )
    state = _global_state()

    snapshot = connector.build_action_availability_snapshot(state)
    limits = connector.build_runtime_capability_limits(state)

    assert snapshot["worker_status"]["accepted_task"] == "healthy"
    assert snapshot["worker_status"]["background_work"] == "degraded"
    assert all("仓库代码读取 owner 不可用" not in item for item in limits)
    assert any(
        row["action_kind"] == "accepted_task_request"
        for row in connector._available_action_affordances(state)
    )
    assert all(
        row["action_kind"] != "accepted_coding_task_request"
        for row in connector._available_action_affordances(state)
    )


def test_connector_projects_new_work_through_one_accepted_task_owner() -> None:
    """New coding work must not compete with an unbound coding-run start."""

    resolvers = connector._available_resolver_affordances(_global_state())
    actions = connector._available_action_affordances(_global_state())

    assert {row["capability"] for row in resolvers} == {
        "local_context_recall",
        "public_answer_research",
        "human_clarification",
        "approval_preparation",
        "self_goal_resolution",
    }
    action_kinds = {row["action_kind"] for row in actions}
    assert "speak" not in action_kinds
    assert "apply_memory_lifecycle_update" not in action_kinds
    assert "background_work_request" not in action_kinds
    assert {
        "accepted_task_request",
        "future_speak",
    } <= action_kinds
    assert "accepted_coding_task_request" not in action_kinds
    assert "trigger_future_cognition" not in action_kinds
    assert "memory_lifecycle_update" not in action_kinds
    assert all(row["context_ref"] == "" for row in actions)


def test_connector_omits_capabilities_with_unavailable_runtime_routes() -> None:
    """Cognition receives only capabilities admitted by deterministic probes."""

    state = _global_state()
    state["action_availability_runtime"] = {
        "route_health": {"background_work": "down"},
    }

    actions = connector._available_action_affordances(state)

    assert all(
        row["action_kind"] != "background_work_request"
        for row in actions
    )
    assert any(
        row["action_kind"] == "accepted_task_status_check"
        for row in actions
    )


@pytest.mark.parametrize("trigger_source", ["internal_thought", "scheduled_tick"])
def test_connector_projects_future_cognition_for_private_sources(
    trigger_source: str,
) -> None:
    """Private cognition keeps the production scheduling capability."""

    state = _global_state()
    state["cognitive_episode"] = canonical_episode(
        episode_id=f"private-{trigger_source}",
        trigger_source=trigger_source,
        content="Continue one grounded private cognition objective.",
    )

    actions = connector._available_action_affordances(state)
    future_cognition = next(
        row
        for row in actions
        if row["action_kind"] == "trigger_future_cognition"
    )

    assert future_cognition["decision_mode"] == "closed"
    assert future_cognition["allowed_decisions"] == ["schedule"]
    assert future_cognition["default_decision"] == "schedule"


def test_connector_projects_memory_lifecycle_only_for_active_commitments() -> None:
    """Lifecycle review remains available exactly when it can execute."""

    state = _global_state()
    state["rag_result"] = {
        "memory_evidence": [],
        "user_memory_unit_candidates": [{
            "unit_id": "commitment-1",
            "unit_type": "active_commitment",
            "status": "active",
            "fact": "Kazusa agreed to answer after checking the result.",
        }],
    }

    actions = connector._available_action_affordances(state)
    lifecycle = next(
        row
        for row in actions
        if row["action_kind"] == "memory_lifecycle_update"
    )

    assert lifecycle["decision_mode"] == "closed"
    assert lifecycle["allowed_decisions"] == [
        "active_commitment_lifecycle",
    ]
    assert lifecycle["default_decision"] == "active_commitment_lifecycle"


def test_connector_projects_only_bound_open_coding_run_affordances() -> None:
    """Each trusted open run remains selectable without an unbound start."""

    state = _global_state()
    state["action_selection_context"] = {
        "coding_runs": [
            {
                "coding_run_ref": "coding_run:run-1",
                "status": "awaiting_approval",
                "objective_summary": "update the parser",
                "allowed_next_actions": ["approve_and_verify", "cancel"],
                "active_blocker": None,
            },
            {
                "coding_run_ref": "coding_run:run-2",
                "status": "blocked",
                "objective_summary": "repair the scheduler",
                "allowed_next_actions": ["respond_to_blocker", "status"],
                "active_blocker": {
                    "blocker_kind": "user_choice",
                    "question": "Which execution boundary should apply?",
                    "options": ["focused", "full"],
                },
            },
        ],
    }

    actions = connector._available_action_affordances(state)
    coding_actions = [
        row
        for row in actions
        if row["action_kind"] == "accepted_coding_task_request"
    ]

    assert [row["context_ref"] for row in coding_actions] == [
        "coding_run:run-1",
        "coding_run:run-2",
    ]
    assert coding_actions[0]["allowed_decisions"] == [
        "approve_and_verify",
        "cancel",
    ]
    assert coding_actions[1]["allowed_decisions"] == [
        "respond_to_blocker",
        "status",
    ]
    assert "update the parser" in coding_actions[0]["capability"]
    assert "Which execution boundary" in coding_actions[1]["capability"]


def test_connector_routes_unavailable_coding_status_to_persisted_lookup() -> None:
    """An unavailable coding worker leaves direct task status available."""

    state = _global_state()
    state["action_availability_runtime"] = {
        "worker_status": {
            "background_work": "unavailable",
        },
        "coding_workspace_status": "healthy",
    }
    state["action_selection_context"] = {
        "coding_runs": [{
            "coding_run_ref": "coding_run:run-1",
            "status": "proposal_ready",
            "objective_summary": "update the parser",
            "allowed_next_actions": ["status", "cancel"],
            "active_blocker": None,
        }],
    }

    actions = connector._available_action_affordances(state)
    coding_actions = [
        row
        for row in actions
        if row["action_kind"] == "accepted_coding_task_request"
    ]

    assert any(
        row["action_kind"] == "accepted_task_status_check"
        for row in actions
    )
    assert [row["allowed_decisions"] for row in coding_actions] == [
        ["cancel"],
    ]
    assert "当前作用域的既有 coding run 只提供以下实际可用决定：cancel" in (
        coding_actions[0]["capability"]
    )
    assert "status" not in coding_actions[0]["capability"]


def test_connector_projects_persisted_coding_status_into_semantic_scene() -> None:
    """Goal cognition receives the current scoped coding status as context."""

    state = _global_state()
    state["action_selection_context"] = {
        "coding_runs": [{
            "coding_run_ref": "coding_run:run-1",
            "status": "proposal_ready",
            "objective_summary": "update the parser",
            "allowed_next_actions": ["status", "cancel"],
            "active_blocker": None,
        }],
    }

    scene = connector._semantic_episode_text(state)

    assert "当前作用域已有持久化代码任务状态" in scene
    assert "proposal_ready" in scene
    assert "update the parser" in scene


def test_connector_keeps_media_as_typed_evidence_without_wire_payloads() -> None:
    """Media descriptions remain semantic evidence while raw bytes and URLs stay out."""

    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        build_cognition_input_from_global_state,
    )

    state = _global_state()
    state["user_multimedia_input"] = [{
        "content_type": "image/png",
        "base64_data": "raw-bytes",
        "url": "https://example.invalid/image.png",
        "description": "whiteboard observation",
    }]
    state["user_input"] = ""
    state["decontextualized_input"] = ""
    payload = build_cognition_input_from_global_state(
        state,
        mutable_state=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        ),
    )
    rendered = json.dumps(payload, ensure_ascii=False)

    assert "whiteboard observation" in rendered
    assert "raw-bytes" not in rendered
    assert "example.invalid" not in rendered


def test_connector_preserves_accepted_task_source_ownership() -> None:
    """Accepted-task outcomes use their source-owned evidence visibility."""

    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        build_cognition_input_from_global_state,
    )

    state = _global_state()
    state["cognitive_episode"] = canonical_episode(
        episode_id="accepted-task-episode",
        trigger_source="tool_result",
        current_global_user_id="user-1",
        content="The requested report is ready.",
        metadata={
            "accepted_task_id": "task-1",
            "accepted_task_summary": "Prepare the report.",
            "result_summary": "Report completed.",
            "failure_summary": "",
        },
    )
    state["cognitive_episode"]["percepts"][0]["source_id"] = "task-1"
    payload = build_cognition_input_from_global_state(
        state,
        mutable_state=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        ),
    )

    evidence = payload["evidence"][0]
    assert evidence["evidence_ref"]["source_kind"] == "tool_result"
    assert evidence["evidence_ref"]["source_id"] == "task-1"
    assert evidence["visible_to"] == [
        "q:event_agency",
        "q:relationship_social",
        "q:moral_identity",
        "q:goal_threat_outcome",
        "q:epistemic_comparison_memory",
    ]


def test_connector_selects_self_cognition_scope_from_typed_metadata() -> None:
    """Self-cognition with a target user selects that one mutable user scope."""

    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
        _scope_caller,
    )

    episode = {
        "trigger_source": "internal_thought",
        "percepts": [{
            "metadata": {"source": "self_cognition_source_packet"},
        }],
    }

    assert _scope_caller(episode) == "self_cognition"


@pytest.mark.asyncio
async def test_final_commit_emits_bounded_success_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful terminal commit records its bounded branch and scope."""

    replace_state = AsyncMock()
    record_event = AsyncMock(return_value={"accepted": True})
    monkeypatch.setattr(connector, "replace_user_cognition_state", replace_state)
    monkeypatch.setattr(connector, "record_cognition_v2_event", record_event)

    await connector._commit_cognition_state(_core_output())

    replace_state.assert_awaited_once()
    record_event.assert_awaited_once_with(
        component="nodes.persona_supervisor2_cognition",
        cognition_component="state_commit",
        status="completed",
        stage_status="completed",
        selected_branch_id="ordinary_response",
        state_scope="user",
        state_commit_status="committed",
        severity="info",
    )


@pytest.mark.asyncio
async def test_failed_final_commit_emits_failure_event_and_remains_authoritative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistence failure is re-raised after best-effort failure telemetry."""

    replace_state = AsyncMock(side_effect=RuntimeError("database unavailable"))
    record_event = AsyncMock(return_value={"accepted": True})
    monkeypatch.setattr(connector, "replace_user_cognition_state", replace_state)
    monkeypatch.setattr(connector, "record_cognition_v2_event", record_event)

    with pytest.raises(RuntimeError, match="database unavailable"):
        await connector._commit_cognition_state(_core_output())

    assert record_event.await_args.kwargs["state_commit_status"] == "failed"
    assert record_event.await_args.kwargs["stage_status"] == "failed"


@pytest.mark.asyncio
async def test_event_write_failure_does_not_override_successful_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The event sink remains non-authoritative after durable replacement."""

    replace_state = AsyncMock()
    record_event = AsyncMock(side_effect=RuntimeError("event sink unavailable"))
    monkeypatch.setattr(connector, "replace_user_cognition_state", replace_state)
    monkeypatch.setattr(connector, "record_cognition_v2_event", record_event)

    await connector._commit_cognition_state(_core_output())

    replace_state.assert_awaited_once()


@pytest.mark.asyncio
async def test_intermediate_commit_false_cycle_emits_no_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolver recurrence defers persistence and terminal telemetry together."""

    user_state = build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=NOW,
    )
    character_state = build_character_production_state(updated_at=NOW)
    replace_state = AsyncMock()
    record_event = AsyncMock(return_value={"accepted": True})
    monkeypatch.setattr(
        connector,
        "get_user_cognition_state",
        AsyncMock(return_value=user_state),
    )
    monkeypatch.setattr(
        connector,
        "get_character_cognition_state",
        AsyncMock(return_value=character_state),
    )
    monkeypatch.setattr(
        connector,
        "run_cognition",
        AsyncMock(return_value=_core_output()),
    )
    monkeypatch.setattr(connector, "replace_user_cognition_state", replace_state)
    monkeypatch.setattr(connector, "record_cognition_v2_event", record_event)

    await connector.call_cognition_subgraph(_global_state(), commit=False)

    replace_state.assert_not_awaited()
    record_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_recurrent_cycle_uses_uncommitted_replacement_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later resolver cycle advances from the prior in-memory V2 state."""

    stored_state = build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=NOW,
    )
    replacement_state = build_acquaintance_user_state(
        global_user_id="user-1",
        updated_at=NOW,
    )
    replacement_state["relationship"]["care"] = 77
    character_state = build_character_production_state(updated_at=NOW)
    get_user_state = AsyncMock(return_value=stored_state)
    run_cognition = AsyncMock(return_value=_core_output())
    monkeypatch.setattr(connector, "get_user_cognition_state", get_user_state)
    monkeypatch.setattr(
        connector,
        "get_character_cognition_state",
        AsyncMock(return_value=character_state),
    )
    monkeypatch.setattr(connector, "run_cognition", run_cognition)

    state = _global_state()
    state["cognition_state_update"] = {
        "state_scope": "user",
        "owner_key": "user-1",
        "replacement_state": replacement_state,
        "comparison_results": [],
        "changed_paths": ["relationship.care"],
    }
    await connector.call_cognition_subgraph(state, commit=False)

    get_user_state.assert_not_awaited()
    cognition_input = run_cognition.await_args.args[0]
    assert cognition_input["mutable_state"]["relationship"]["care"] == 77


@pytest.mark.asyncio
async def test_cognition_entry_replaces_stale_profile_with_episode_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared chat/self-cognition boundary should load identity once."""

    state = _global_state()
    for field_name in (
        "character_identity_revision_number",
        "character_identity_context",
        "character_identity_surface_context",
        "character_identity_projection_digest",
        "character_identity_consumer_kinds",
        "character_identity_episode_id",
        "character_identity_epistemic_core_included",
    ):
        state.pop(field_name)
    latest_identity = canonical_character_identity(marker="revision-n")
    latest_profile = {
        **latest_identity,
        "global_user_id": "character-1",
    }
    latest_revision = {"effective_identity": latest_identity}
    cognition_context = project_identity_for_cognition(latest_revision)
    surface_context = project_identity_for_surface(latest_revision)
    snapshot = {
        "revision_number": 5,
        "character_profile": latest_profile,
        "cognition_context": cognition_context,
        "surface_context": surface_context,
        "projection_digest": identity_projection_digest(
            revision_number=5,
            cognition_context=cognition_context,
            surface_context=surface_context,
        ),
        "consumer_kinds": (
            projected_identity_consumer_kinds(cognition_context)
        ),
    }
    load_latest = AsyncMock(return_value=snapshot)
    run_cognition = AsyncMock(return_value=_core_output())
    monkeypatch.setattr(
        connector,
        "load_latest_identity_for_episode",
        load_latest,
    )
    monkeypatch.setattr(
        connector,
        "get_user_cognition_state",
        AsyncMock(return_value=build_acquaintance_user_state(
            global_user_id="user-1",
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(
        connector,
        "get_character_cognition_state",
        AsyncMock(return_value=build_character_production_state(
            updated_at=NOW,
        )),
    )
    monkeypatch.setattr(connector, "run_cognition", run_cognition)

    update = await connector.call_cognition_subgraph(state, commit=False)

    load_latest.assert_awaited_once_with(
        episode_id="episode-1",
        correlation_id="episode-1",
        include_epistemic_core=False,
    )
    cognition_input = run_cognition.await_args.args[0]
    assert (
        cognition_input["character_identity_context"]
        == cognition_context
    )
    assert "revision-n" in str(cognition_input["character_identity_context"])
    assert "canonical" not in str(
        cognition_input["character_identity_context"]
    )
    assert update["character_profile"]["name"] == latest_profile["name"]
    assert update["character_identity_revision_number"] == 5
