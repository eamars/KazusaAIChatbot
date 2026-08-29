"""Brain semantic decision tests."""

from __future__ import annotations

import pytest

from tests.test_dsh_brain_interaction_contracts import _request_mapping


@pytest.mark.asyncio
async def test_brain_semantic_decision_is_enacted_without_keyword_or_post_llm_reclassification() -> None:
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV1
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine

    request = DshBrainInteractionRequestV1.from_mapping(_request_mapping())

    async def judge(request, context):
        return {"schema_version": "dsh_brain_interaction.v1", "interaction_id": request.interaction_id, "request_digest": request.request_digest, "kind": request.kind, "decision": "answer", "answer": "A semantic answer", "reason": "judged", "response_goal": None, "relay_mode": None}

    decision = await BrainDecisionEngine(judge=judge).decide(request, context={"user_text": "allow this"})
    assert decision.decision == "answer"
    assert decision.answer == "A semantic answer"


def test_allow_once_grant_binds_request_activation_and_lease() -> None:
    """Immediate approval grants retain the exact activation lease lineage."""

    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionDecisionV1,
        DshBrainInteractionRequestV1,
    )
    from kazusa_ai_chatbot.dsh_interaction.decision import enact_decision

    request_mapping = _request_mapping()
    request_mapping.update({"kind": "approval", "tool_name": "pwsh"})
    request = DshBrainInteractionRequestV1.from_mapping(request_mapping)
    decision = DshBrainInteractionDecisionV1.from_mapping({
        "schema_version": request.schema_version,
        "interaction_id": request.interaction_id,
        "request_digest": request.request_digest,
        "kind": request.kind,
        "decision": "allow_once",
        "answer": None,
        "response_goal": None,
        "relay_mode": None,
        "reason": "the exact native operation is allowed once",
    })

    result = enact_decision(request, decision, now=request.issued_at)

    grant = result["grant"]
    assert grant["activation_id"] == request.activation_id
    assert grant["lease_epoch"] == request.lease_epoch


def test_global_state_carries_typed_pending_interaction_and_semantic_decision() -> None:
    from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
        DshPendingInteractionContext,
        DshSemanticInteractionDecision,
    )

    assert DshPendingInteractionContext
    assert DshSemanticInteractionDecision


@pytest.mark.asyncio
async def test_persona_projects_pending_dsh_context_and_returns_canonical_p_stage_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persona graph preserves pending DSH input and typed output fields."""

    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV1,
    )
    from kazusa_ai_chatbot.nodes import persona_supervisor2 as persona_module

    request = DshBrainInteractionRequestV1.from_mapping(_request_mapping())
    pending = request.unsigned_dict()
    decision = {
        "schema_version": "dsh_brain_interaction.v1",
        "interaction_id": request.interaction_id,
        "request_digest": request.request_digest,
        "kind": request.kind,
        "decision": "reject",
        "answer": None,
        "response_goal": None,
        "relay_mode": None,
        "reason": "insufficient context",
    }
    captured: dict[str, object] = {}

    class FakeStateGraph:
        """Capture the persona graph boundary without running semantic nodes."""

        def __init__(self, _schema: object) -> None:
            pass

        def add_node(self, *_args: object, **_kwargs: object) -> None:
            return None

        def add_edge(self, *_args: object, **_kwargs: object) -> None:
            return None

        def add_conditional_edges(
            self,
            *_args: object,
            **_kwargs: object,
        ) -> None:
            return None

        def compile(self) -> FakeStateGraph:
            return self

        async def ainvoke(
            self,
            initial_state: dict[str, object],
        ) -> dict[str, object]:
            captured["initial_state"] = initial_state
            return {
                **initial_state,
                "should_respond": False,
                "final_dialog": [],
                "target_addressed_user_ids": [],
                "target_broadcast": False,
                "public_group_scene_projection_status": "skipped",
                "public_group_scene_projection_reason": "not_group",
                "dsh_interaction_decision": decision,
            }

    monkeypatch.setattr(persona_module, "StateGraph", FakeStateGraph)

    state = {
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-1",
        },
        "storage_timestamp_utc": request.issued_at,
        "local_time_context": {},
        "llm_trace_id": "dsh-trace-id",
        "user_input": request.transient_detail,
        "prompt_message_context": {
            "body_text": request.transient_detail,
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "platform": request.platform,
        "platform_channel_id": request.platform_channel_id,
        "channel_type": "private",
        "channel_name": "",
        "platform_message_id": "message-1",
        "active_turn_platform_message_ids": [],
        "active_turn_conversation_row_ids": [],
        "active_turn_conversation_source_refs": [],
        "platform_user_id": "platform-user-1",
        "global_user_id": request.global_user_id,
        "user_name": "User",
        "user_profile": {},
        "platform_bot_id": "character-1",
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "",
        "debug_modes": {},
        "should_respond": False,
        "attempt_diagnostics": [],
        "ambient_logical_turns": [],
        "interaction_logical_turns": [],
        "conversation_progress": None,
        "conversation_episode_state": None,
        "pending_dsh_interaction": pending,
        "pending_dsh_reply": True,
    }

    result = await persona_module.persona_supervisor2(state)

    initial_state = captured["initial_state"]
    assert isinstance(initial_state, dict)
    assert initial_state["pending_dsh_interaction"] == pending
    assert initial_state["pending_dsh_interaction"] is not pending
    assert initial_state["pending_dsh_reply"] is True
    projected_decision = result["dsh_interaction_decision"]
    assert projected_decision == decision
    assert projected_decision is not decision


@pytest.mark.asyncio
async def test_canonical_dsh_cognition_helper_runs_the_existing_chain_with_exact_pending_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV1
    from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition

    request = DshBrainInteractionRequestV1.from_mapping(_request_mapping())
    pending = request.unsigned_dict()
    captured: dict[str, object] = {}
    state = {
        "llm_trace_id": "dsh-trace-id",
        "cognitive_episode": {
            "episode_id": "dsh-episode-id",
            "target_scope": {"platform": "debug"},
        },
    }

    def build_input(state):
        captured["state"] = state
        return {"canonical": True, "episode": state["cognitive_episode"]}

    async def run_chain(payload, services):
        captured["payload"] = payload
        captured["services"] = services
        captured["trace_id"] = cognition.llm_tracing.current_trace_id()
        captured["diagnostics"] = cognition.current_chain_scope()
        return {
            "response_plan": {
                "dsh_interaction_decision": {
                    "interaction_id": request.interaction_id,
                    "kind": request.kind,
                    "decision": "reject",
                    "answer": None,
                    "response_goal": None,
                    "relay_mode": None,
                    "reason": "the canonical chain declined the request",
                },
            },
        }

    monkeypatch.setattr(cognition, "build_cognition_input_from_global_state", build_input)
    monkeypatch.setattr(cognition, "run_cognition", run_chain)
    services = object()
    result = await cognition.run_dsh_interaction_cognition(
        state,
        pending_interaction=pending,
        services=services,
    )
    assert result["decision"] == "reject"
    staged = captured["state"]
    assert isinstance(staged, dict)
    assert staged["pending_dsh_interaction"] == pending
    assert captured["payload"] == {
        "canonical": True,
        "episode": state["cognitive_episode"],
    }
    assert captured["services"] is services
    assert captured["trace_id"] == "dsh-trace-id"
    diagnostics = captured["diagnostics"]
    assert diagnostics is not None
    assert diagnostics.run_id == "dsh-episode-id"
    assert diagnostics.source_kind == "debug"
    assert diagnostics.llm_trace_id == "dsh-trace-id"
    assert diagnostics.cognition_invocation_id == "dsh-episode-id"
    assert cognition.llm_tracing.current_trace_id() == ""
    assert cognition.current_chain_scope() is None


@pytest.mark.asyncio
async def test_canonical_dsh_cognition_helper_restores_scopes_when_chain_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV1
    from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition as cognition

    request = DshBrainInteractionRequestV1.from_mapping(_request_mapping())
    pending = request.unsigned_dict()
    state = {
        "llm_trace_id": "dsh-failure-trace-id",
        "cognitive_episode": {
            "episode_id": "dsh-failure-episode-id",
            "target_scope": {"platform": "debug"},
        },
    }
    captured: dict[str, object] = {}

    def build_input(staged_state):
        return {"canonical": staged_state["pending_dsh_interaction"]}

    async def run_chain(payload, services):
        del payload, services
        captured["trace_id"] = cognition.llm_tracing.current_trace_id()
        captured["diagnostics"] = cognition.current_chain_scope()
        raise RuntimeError("synthetic DSH cognition failure")

    monkeypatch.setattr(cognition, "build_cognition_input_from_global_state", build_input)
    monkeypatch.setattr(cognition, "run_cognition", run_chain)

    with pytest.raises(RuntimeError, match="synthetic DSH cognition failure"):
        await cognition.run_dsh_interaction_cognition(
            state,
            pending_interaction=pending,
            services=object(),
        )

    assert captured["trace_id"] == "dsh-failure-trace-id"
    assert captured["diagnostics"] is not None
    assert cognition.llm_tracing.current_trace_id() == ""
    assert cognition.current_chain_scope() is None
