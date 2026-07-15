"""Persona V2 graph routing and resolver-boundary tests."""

import inspect
import json
import logging
from collections.abc import Mapping
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    CognitionExecutionError,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2 as persona_module
from tests.cognition_core_v2_test_helpers import canonical_episode


NOW = "2026-07-14T00:00:00Z"


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
        "resolver_progress": {
            "status": "not_requested",
            "semantic_summary": "no resolver request",
        },
        "residue": "bounded private residue",
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
async def test_resolver_failure_observation_excludes_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Backend details stay outside recurrence observations and protected logs."""

    sensitive_text = "mongodb://secret-user:secret-password@private-host"
    captured_observation: dict[str, object] = {}

    async def fail_recall(*args: object, **kwargs: object) -> dict[str, object]:
        del args
        del kwargs
        raise TimeoutError(sensitive_text)

    async def run_one_capability_cycle(
        state: Mapping[str, object],
        *,
        cognition_func: object,
        capability_func: object,
        max_cycles: int,
    ) -> dict[str, object]:
        del cognition_func
        del max_cycles
        observation = await capability_func(  # type: ignore[operator]
            {
                "capability": "local_context_recall",
                "semantic_goal": "retrieve bounded context",
            },
            state,
        )
        captured_observation.update(observation)
        return {
            "cognition_output": {
                "cognition_core_output": _cognition_output("silence"),
            },
            "observations": [observation],
            "telemetry": {},
        }

    monkeypatch.setattr(
        persona_module,
        "run_rag_evidence_for_persona_state",
        fail_recall,
    )
    monkeypatch.setattr(
        persona_module,
        "call_v2_resolver_loop",
        run_one_capability_cycle,
    )
    commit = AsyncMock()
    monkeypatch.setattr(persona_module, "commit_cognition_output", commit)
    caplog.set_level(logging.WARNING, logger=persona_module.__name__)

    result = await persona_module.stage_1_goal_resolver({
        "storage_timestamp_utc": NOW,
    })

    assert captured_observation["semantic_summary"] == (
        "local context recall failed"
    )
    assert sensitive_text not in json.dumps(result["resolver_observations"])
    assert sensitive_text not in caplog.text
    assert "TimeoutError" in caplog.text
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

    assert "call_v2_resolver_loop" in resolver_source
    assert "cognition_state_committed" in resolver_source
    assert "cognition_core_output" in terminal_source
