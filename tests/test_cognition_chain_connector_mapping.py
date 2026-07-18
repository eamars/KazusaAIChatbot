"""Checkpoint F connector mapping tests for the canonical V2 caller."""

import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as connector

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    new_empty_goal_progress,
)
from tests.cognition_core_v2_test_helpers import canonical_episode


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
            "emotional_tone": "composed",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "diagnostics": {},
    }


def _global_state() -> dict[str, object]:
    """Build the adapter-owned fields needed by the V2 mapper."""

    return {
        "character_profile": {"global_user_id": "character-1"},
        "character_cognition_state": build_character_production_state(
            updated_at=NOW,
        ),
        "storage_timestamp_utc": NOW,
        "user_input": "hello",
        "decontexualized_input": "hello",
        "prompt_message_context": {},
        "cognitive_episode": canonical_episode(
            episode_id="episode-1",
            current_global_user_id="global-user-1",
            content="hello",
        ),
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
    assert "speaker of dialog_text" in payload["scene_context"][
        "current_user_role"
    ]
    assert "implicit subject" in payload["scene_context"]["character_role"]


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


def test_connector_projects_full_registry_capacity() -> None:
    """Visible cognition exposes every runtime-eligible public action."""

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
    assert {
        "background_work_request",
        "future_speak",
    } <= action_kinds
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


def test_connector_projects_distinct_open_coding_run_affordances() -> None:
    """Start and each trusted open run remain separately selectable."""

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
        "",
        "coding_run:run-1",
        "coding_run:run-2",
    ]
    assert coding_actions[0]["allowed_decisions"] == ["start"]
    assert coding_actions[1]["allowed_decisions"] == [
        "approve_and_verify",
        "cancel",
    ]
    assert coding_actions[2]["allowed_decisions"] == [
        "respond_to_blocker",
        "status",
    ]
    assert "update the parser" in coding_actions[1]["capability"]
    assert "Which execution boundary" in coding_actions[2]["capability"]


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
    state["decontexualized_input"] = ""
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
