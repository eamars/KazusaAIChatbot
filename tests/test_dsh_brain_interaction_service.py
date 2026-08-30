"""Deterministic Brain interaction V2 service and composition tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
from kazusa_ai_chatbot.dsh_interaction.contracts import (
    DshBrainInteractionRequestV2,
)
from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService
from tests.test_dsh_brain_interaction_contracts import _request_mapping


def _signed_request(**overrides: object) -> DshBrainInteractionRequestV2:
    now = datetime.now(UTC)
    raw = _request_mapping(
        issued_at=now.isoformat().replace("+00:00", "Z"),
        expires_at=(now + timedelta(minutes=5)).isoformat().replace(
            "+00:00",
            "Z",
        ),
        **overrides,
    )
    request = DshBrainInteractionRequestV2.from_mapping(raw)
    return sign_request(request, secret=b"brain-secret")


async def _empty_dsh_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    from kazusa_ai_chatbot import service as service_module

    async def character_profile():
        return {"name": "Kazusa", "global_user_id": "character-1"}

    async def user_profile(_global_user_id):
        return {"display_name": "User"}

    async def cognition_state(*_args, **_kwargs):
        return {}

    async def conversation_history(**_kwargs):
        return []

    monkeypatch.setattr(
        service_module,
        "_load_latest_character_profile_snapshot",
        character_profile,
    )
    monkeypatch.setattr(service_module, "get_user_profile", user_profile)
    monkeypatch.setattr(
        service_module,
        "get_user_cognition_state",
        cognition_state,
    )
    monkeypatch.setattr(
        service_module,
        "get_character_cognition_state",
        cognition_state,
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        conversation_history,
    )
    monkeypatch.setattr(
        service_module,
        "_action_availability_runtime_for_target",
        lambda **_kwargs: {},
    )


@pytest.mark.asyncio
async def test_dsh_cognition_state_contains_canonical_empty_rag_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Internal DSH cognition receives the ordinary bounded context shape."""

    await _empty_dsh_dependencies(monkeypatch)
    from kazusa_ai_chatbot import service as service_module

    request = _signed_request()
    state = await service_module._build_dsh_cognition_state(request)
    rag_result = state["rag_result"]
    assert rag_result["answer"] == ""
    assert rag_result["conversation_evidence"] == []
    assert rag_result["memory_evidence"] == []
    assert rag_result["user_image"]["user_memory_context"]
    assert state["debug_modes"]["no_remember"] is True
    episode = state["cognitive_episode"]
    assert episode["trigger_source"] == "self_cognition"
    assert episode["privacy_scope"] == "private"
    percept = next(
        row
        for row in episode["percepts"]
        if row["percept_kind"] == "dsh_interaction_context"
    )
    assert "runtime-authored system observation" in percept["content"][
        "semantic_text"
    ]
    assert request.transient_detail not in percept["content"]["semantic_text"]


@pytest.mark.asyncio
async def test_production_dsh_judge_passes_complete_internal_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _empty_dsh_dependencies(monkeypatch)
    from kazusa_ai_chatbot import service as service_module

    request = _signed_request()
    captured: list[dict[str, object]] = []

    async def fake_run_dsh_cognition(state, *, pending_interaction, services):
        captured.append(dict(pending_interaction))
        assert state["cognitive_episode"]["trigger_source"] == "self_cognition"
        assert services is service_module._dsh_cognition_services
        return {
            "decision": "reject",
            "answer": None,
            "reason": "the character declined this internal interaction",
        }

    monkeypatch.setattr(
        service_module,
        "run_dsh_interaction_cognition",
        fake_run_dsh_cognition,
    )
    result = await service_module._production_dsh_judge(request, {})
    assert captured == [request.unsigned_dict()]
    assert result == {
        "schema_version": "dsh_brain_interaction.v2",
        "interaction_id": request.interaction_id,
        "request_digest": request.request_digest,
        "kind": request.kind,
        "decision": "reject",
        "answer": None,
        "reason": "the character declined this internal interaction",
    }


@pytest.mark.asyncio
async def test_service_returns_internal_decision_without_checkpoint_or_delivery() -> None:
    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore

    request = _signed_request()

    async def judge(request, context):
        assert context["workspace_fingerprint"] == request.workspace_fingerprint
        return {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "reject",
            "answer": None,
            "reason": "the internal question has insufficient grounds",
        }

    service = BrainInteractionService(
        secret=b"brain-secret",
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=InMemoryInteractionStore(),
        context_provider=lambda value: {
            "workspace_fingerprint": value.workspace_fingerprint,
            "policy_epoch": value.policy_epoch,
        },
    )
    result = await service.handle_signed(request)
    assert result["decision"] == "reject"
    assert set(result) == {
        "schema_version", "interaction_id", "request_digest", "kind",
        "decision", "answer", "reason",
    }
    row = await service._interaction_store.get(request.interaction_id)
    assert row is not None
    assert row["schema_version"] == "dsh_brain_interaction.v2"
    assert row["status"] == "decided"
    assert "pending" not in row
    assert "delivery" not in row


@pytest.mark.asyncio
async def test_immediate_allow_once_consumes_exact_durable_grant() -> None:
    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore

    request = _signed_request(kind="approval", tool_name="pwsh")

    async def judge(request, _context):
        return {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "allow_once",
            "answer": None,
            "reason": "the exact operation is grounded and permitted",
        }

    store = InMemoryInteractionStore()
    service = BrainInteractionService(
        secret=b"brain-secret",
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=store,
    )
    result = await service.handle_signed(request)
    assert result["decision"] == "allow_once"
    assert result["grant"]["grant_status"] == "consumed"
    row = await store.get(request.interaction_id)
    assert row is not None
    assert row["grant_status"] == "consumed"
    assert row["status"] == "decided"


def test_brain_service_exposes_only_the_v2_interaction_route() -> None:
    import kazusa_ai_chatbot.service as service_module

    paths = {route.path for route in service_module.app.routes}
    assert "/runtime/dsh/interactions" in paths
    assert "/runtime/dsh/interactions/checkpoint" not in paths
    middleware = next(
        layer
        for layer in service_module.app.user_middleware
        if layer.cls.__name__ == "_DshInteractionBodyLimitMiddleware"
    )
    assert "/runtime/dsh/interactions/checkpoint" not in middleware.cls._PATHS
