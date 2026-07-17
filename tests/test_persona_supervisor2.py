"""Persona V2 graph routing and resolver-boundary tests."""

import inspect
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_msg_decontexualizer as decontextualizer_module,
)

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2 as persona_module
from kazusa_ai_chatbot.cognition_resolver.loop import (
    _terminal_blocker_speak_action_spec,
)
from tests.cognition_core_v2_test_helpers import canonical_episode


NOW = "2026-07-14T00:00:00Z"


def test_decontextualizer_preserves_direct_imperative_roles() -> None:
    """A clear command keeps its implicit character subject and user pronouns."""

    prompt = decontextualizer_module._MSG_DECONTEXUALIZER_PROMPT

    assert "祈使句或命令句" in prompt
    assert "隐含动作主体是{character_name}" in prompt
    assert "不得添加{character_name}作为语法主语" in prompt


def _cognition_output(route: str) -> dict[str, object]:
    """Build one exact native V2 result for persona routing."""

    return {
        "schema_version": "cognition_core_output.v2",
        "intention": {
            "route": route,
            "intention": "respond to the grounded episode",
            "target_roles": [],
            "reason": "the current episode establishes the selected route",
        },
        "supporting_bids": [],
        "state_update": {
            "state_scope": "user",
            "owner_key": "user-1",
            "replacement_state": build_acquaintance_user_state(
                global_user_id="user-1",
                updated_at=NOW,
            ),
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
        "selected_bid_reason": "the current episode establishes the route",
        "private_monologue": "I want to respond to this clearly.",
        "expression_policy": {
            "visibility": "visible" if route == "speech" else "none",
            "emotional_tone": "composed",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "diagnostics": {
            "run_id": "persona-routing-test",
            "stage_status": {},
            "selected_question_count": 0,
            "dispatched_question_count": 0,
            "selected_branch_count": 0,
            "dispatched_branch_count": 0,
            "completed_branch_count": 0,
            "failed_branch_count": 0,
            "overlap_ms": 0,
            "dependency_wait_ms": 0,
            "total_ms": 0,
            "warnings": [],
        },
    }


def _persona_state() -> dict[str, object]:
    """Build the adapter-owned state required by the persona graph."""

    return {
        "storage_timestamp_utc": NOW,
        "local_time_context": {
            "current_local_datetime": "2026-07-14 12:00",
            "current_local_weekday": "Tuesday",
        },
        "user_name": "Test User",
        "platform": "debug",
        "platform_message_id": "message-1",
        "platform_user_id": "platform-user-1",
        "global_user_id": "user-1",
        "user_input": "hello",
        "prompt_message_context": {
            "body_text": "hello",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": False,
        },
        "user_profile": {},
        "platform_bot_id": "debug-bot",
        "character_profile": {
            "name": "Test Character",
            "global_user_id": "character-1",
        },
        "platform_channel_id": "channel-1",
        "channel_type": "private",
        "channel_name": "",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "should_respond": True,
        "indirect_speech_context": "",
        "channel_topic": "",
        "debug_modes": {},
        "cognitive_episode": canonical_episode(
            episode_id="persona-episode-1",
            current_global_user_id="user-1",
            content="hello",
        ),
    }


def _resolver_update(route: str) -> dict[str, object]:
    """Build one committed V2 resolver update for graph plumbing tests."""

    return {
        "cognition_core_output": _cognition_output(route),
        "cognition_state_committed": True,
        "action_specs": [],
        "rag_result": {},
    }


def test_route_after_cognition_uses_validated_v2_speech_intention() -> None:
    """A validated V2 speech intention enters the visible surface path."""

    assert persona_module._route_after_cognition({
        "cognition_core_output": _cognition_output("speech"),
    }) == "respond"


def test_route_after_cognition_uses_validated_v2_silence_intention() -> None:
    """A validated V2 silence intention remains private."""

    assert persona_module._route_after_cognition({
        "cognition_core_output": _cognition_output("silence"),
    }) == "silent"


def test_route_after_cognition_rejects_missing_v2_output() -> None:
    """Legacy action specs cannot become routing authority."""

    with pytest.raises(
        CognitionExecutionError,
        match="validated V2 cognition output is required",
    ):
        persona_module._route_after_cognition({
            "action_specs": [{"kind": "speak"}],
        })


def test_route_after_cognition_validates_v2_output_before_routing() -> None:
    """A partial V2-shaped mapping cannot select a surface."""

    with pytest.raises(CognitionContractError):
        persona_module._route_after_cognition({
            "cognition_core_output": {
                "intention": {"route": "speech"},
            },
        })


@pytest.mark.asyncio
async def test_live_persona_loads_open_coding_run_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user-message turn receives the existing trusted run projections."""

    captured: dict[str, object] = {}
    contexts = [{
        "schema_version": "coding_run_context.v1",
        "coding_run_ref": "coding_run:run-1",
        "status": "blocked",
        "objective_summary": "repair the scheduler",
        "allowed_next_actions": ["respond_to_blocker", "status"],
        "active_blocker": None,
        "followup_open": True,
        "updated_at": NOW,
    }]

    async def load_contexts(**kwargs: object) -> list[dict[str, object]]:
        captured.update(kwargs)
        return contexts

    monkeypatch.setattr(
        persona_module,
        "load_open_coding_run_contexts_for_scope",
        load_contexts,
    )

    result = await persona_module._load_live_action_selection_context(
        _persona_state(),
    )

    assert captured == {
        "source_platform": "debug",
        "source_channel_id": "channel-1",
        "requester_global_user_id": "user-1",
        "limit": 3,
    }
    assert result["action_selection_context"] == {"coding_runs": contexts}


def test_resolver_owned_speak_spec_overrides_stale_evidence_route() -> None:
    """A terminal resolver surface remains visible after recurrence."""

    request = {
        "schema_version": "resolver_capability_request.v1",
        "capability_kind": "local_context_recall",
        "objective": "retrieve grounded breakfast evidence",
        "reason": "the direct question requested one grounded answer",
        "priority": "now",
    }
    blocker = {
        "schema_version": "resolver_observation.v1",
        "observation_id": "resolver_obs_duplicate_request",
        "capability_kind": "local_context_recall",
        "request_objective": "retrieve grounded breakfast evidence",
        "request_reason": "the direct question requested one grounded answer",
        "status": "failed",
        "prompt_safe_summary": "No additional grounded evidence was found.",
        "evidence_refs": [],
        "created_at_utc": NOW,
    }
    state = {
        "cognition_core_output": _cognition_output("evidence"),
        "action_specs": [
            _terminal_blocker_speak_action_spec(request, blocker)
        ],
    }

    assert persona_module._cognition_selects_text_surface(state) is True


@pytest.mark.asyncio
async def test_persona_stage_uses_canonical_resolver_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live persona stage enters the shared recurrence and commits once."""

    captured: dict[str, object] = {}

    async def load_action_context(state: dict) -> dict:
        return {**state, "action_selection_context": {"coding_runs": []}}

    async def load_pending(state: dict) -> dict:
        return state

    async def run_resolver_loop(state: dict, **kwargs: object) -> dict:
        captured["state"] = state
        captured["kwargs"] = kwargs
        return {
            **state,
            "cognition_core_output": _cognition_output("silence"),
        }

    monkeypatch.setattr(
        persona_module,
        "_load_live_action_selection_context",
        load_action_context,
    )
    monkeypatch.setattr(
        persona_module,
        "load_matching_pending_resume_into_state",
        load_pending,
    )
    monkeypatch.setattr(
        persona_module,
        "ensure_initial_resolver_inputs",
        lambda state, *, max_cycles: state,
    )
    monkeypatch.setattr(
        persona_module,
        "call_cognition_resolver_loop",
        run_resolver_loop,
    )
    commit = AsyncMock()
    monkeypatch.setattr(persona_module, "commit_cognition_output", commit)

    await persona_module.stage_1_goal_resolver({
        "storage_timestamp_utc": NOW,
    })

    kwargs = captured["kwargs"]
    assert callable(kwargs["call_cognition_subgraph_func"])
    assert callable(kwargs["execute_capability_func"])
    assert kwargs["max_cycles"] == persona_module.COGNITION_RESOLVER_MAX_CYCLES
    assert kwargs["capability_timeout_seconds"] == (
        persona_module.COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS
    )
    commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_action_subgraph_preserves_dialog_and_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The V2 surface path preserves dialog fragments in its episode trace."""

    state = _persona_state()
    state.update(_resolver_update("speech"))
    surface = AsyncMock(return_value={
        "text_surface_output_v2": {
            "content_plan": "acknowledge the current episode",
        },
        "visual_surface_output_v2": {
            "schema_version": "visual_surface_output.v2",
            "visual_directives": "terminal visual scene",
            "selected_surface_intent": "illustrate the terminal scene",
        },
    })
    dialog = AsyncMock(return_value={
        "final_dialog": ["Hello.", "How are you?"],
        "target_addressed_user_ids": ["user-1"],
        "target_broadcast": False,
    })
    monkeypatch.setattr(
        persona_module,
        "call_l3_text_surface_handler",
        surface,
    )
    monkeypatch.setattr(persona_module, "dialog_agent", dialog)

    result = await persona_module.call_action_subgraph(state)

    assert result["final_dialog"] == ["Hello.", "How are you?"]
    assert result["target_addressed_user_ids"] == ["user-1"]
    assert result["target_broadcast"] is False
    assert result["episode_trace"]["surface_outputs"][0]["fragments"] == [
        "Hello.",
        "How are you?",
    ]
    assert result["visual_surface_output_v2"]["visual_directives"] == (
        "terminal visual scene"
    )
    dialog_state = dialog.await_args.args[0]
    assert "text_surface_output_v2" in dialog_state
    assert "visual_surface_output_v2" not in dialog_state
    assert result["episode_trace"]["surface_outputs"][1]["fragments"] == [
        "terminal visual scene"
    ]
    surface.assert_awaited_once()
    dialog.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_action_subgraph_preserves_empty_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty renderer result remains an empty visible surface."""

    state = _persona_state()
    state.update(_resolver_update("speech"))
    monkeypatch.setattr(
        persona_module,
        "call_l3_text_surface_handler",
        AsyncMock(return_value={"text_surface_output_v2": {}}),
    )
    monkeypatch.setattr(
        persona_module,
        "dialog_agent",
        AsyncMock(return_value={
            "final_dialog": [],
            "target_addressed_user_ids": [],
            "target_broadcast": False,
        }),
    )

    result = await persona_module.call_action_subgraph(state)

    assert result["final_dialog"] == []
    assert result["target_addressed_user_ids"] == []
    assert result["target_broadcast"] is False


@pytest.mark.asyncio
async def test_stage_3_no_response_records_private_v2_trace() -> None:
    """A V2 silence result still leaves bounded consolidation evidence."""

    state = _persona_state()
    state.update(_resolver_update("silence"))

    result = await persona_module.stage_3_no_response(state)

    assert result["final_dialog"] == []
    assert result["surface_outputs"][0]["surface_kind"] == "private"
    assert result["episode_trace"]["trigger_source"] == "user_message"


def _patch_persona_graph_stages(
    monkeypatch: pytest.MonkeyPatch,
    *,
    route: str,
) -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    """Patch external persona stages with deterministic V2 updates."""

    decontextualizer = AsyncMock(
        return_value={"decontexualized_input": "hello"},
    )
    resolver = AsyncMock(return_value=_resolver_update(route))
    dialog = AsyncMock(return_value={
        "final_dialog": ["Hello."],
        "target_addressed_user_ids": ["user-1"],
        "target_broadcast": False,
    })
    monkeypatch.setattr(
        persona_module,
        "call_msg_decontexualizer",
        decontextualizer,
    )
    monkeypatch.setattr(persona_module, "stage_1_goal_resolver", resolver)
    monkeypatch.setattr(
        persona_module,
        "call_memory_lifecycle_update_handler",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        persona_module,
        "call_l3_text_surface_handler",
        AsyncMock(return_value={"text_surface_output_v2": {}}),
    )
    monkeypatch.setattr(persona_module, "dialog_agent", dialog)
    return decontextualizer, resolver, dialog


@pytest.mark.asyncio
async def test_persona_supervisor_returns_dialog_and_consolidation_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The V2 graph returns dialog plus its post-cognition snapshot."""

    decontextualizer, resolver, _ = _patch_persona_graph_stages(
        monkeypatch,
        route="speech",
    )

    result = await persona_module.persona_supervisor2(_persona_state())

    assert result["final_dialog"] == ["Hello."]
    assert result["target_addressed_user_ids"] == ["user-1"]
    assert result["target_broadcast"] is False
    assert result["cognition_state_committed"] is True
    assert result["consolidation_state"]["decontexualized_input"] == "hello"
    assert result["consolidation_state"]["final_dialog"] == ["Hello."]
    decontextualizer.assert_awaited_once()
    resolver.assert_awaited_once()


@pytest.mark.asyncio
async def test_persona_supervisor_v2_silence_skips_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validated V2 silence intention cannot enter the dialog renderer."""

    _, _, dialog = _patch_persona_graph_stages(
        monkeypatch,
        route="silence",
    )

    result = await persona_module.persona_supervisor2(_persona_state())

    assert result["should_respond"] is False
    assert result["final_dialog"] == []
    assert result["target_addressed_user_ids"] == []
    assert result["target_broadcast"] is False
    assert result["consolidation_state"]["should_respond"] is False
    dialog.assert_not_awaited()


@pytest.mark.asyncio
async def test_persona_supervisor_scopes_history_before_cognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cognition sees only the current user's interaction history."""

    state = _persona_state()
    state["chat_history_wide"] = [
        {
            "role": "user",
            "platform_user_id": "platform-user-1",
            "global_user_id": "user-1",
            "body_text": "current user context",
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": NOW,
        },
        {
            "role": "assistant",
            "platform_user_id": "debug-bot",
            "global_user_id": "character-1",
            "body_text": "current user reply",
            "addressed_to_global_user_ids": ["user-1"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": NOW,
        },
        {
            "role": "user",
            "platform_user_id": "platform-user-2",
            "global_user_id": "user-2",
            "body_text": "other user private context",
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
            "mentions": [],
            "reply_context": {},
            "timestamp": NOW,
        },
    ]
    decontextualizer, resolver, _ = _patch_persona_graph_stages(
        monkeypatch,
        route="silence",
    )

    await persona_module.persona_supervisor2(state)

    decontextualizer_text = [
        row["body_text"]
        for row in decontextualizer.await_args.args[0]["chat_history_recent"]
    ]
    resolver_text = [
        row["body_text"]
        for row in resolver.await_args.args[0]["chat_history_recent"]
    ]
    assert "other user private context" in decontextualizer_text
    assert resolver_text == ["current user context", "current user reply"]


@pytest.mark.asyncio
async def test_persona_supervisor_preserves_no_remember_for_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supervisor returns no-remember metadata for the service owner."""

    state = _persona_state()
    state["debug_modes"] = {"no_remember": True}
    _patch_persona_graph_stages(monkeypatch, route="speech")

    result = await persona_module.persona_supervisor2(state)

    assert result["final_dialog"] == ["Hello."]
    assert result["consolidation_state"]["debug_modes"] == {
        "no_remember": True,
    }


def test_persona_graph_commits_v2_state_before_terminal_surface() -> None:
    """The graph has one resolver stage and one explicit commit marker path."""

    resolver_source = inspect.getsource(persona_module.stage_1_goal_resolver)
    terminal_source = inspect.getsource(persona_module.persona_supervisor2)

    assert "call_cognition_resolver_loop" in resolver_source
    assert "cognition_state_committed" in resolver_source
    assert "cognition_core_output" in terminal_source
