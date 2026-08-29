"""Individual real-local-model coverage for Brain interaction judgment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from tests.test_dsh_brain_interaction_contracts import _request_mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.live_llm


def _require_live_backend() -> None:
    """Require explicit local-model execution for Brain judgment cases."""

    if os.environ.get("KAZUSA_RUN_LIVE_LLM") != "1":
        pytest.skip("set KAZUSA_RUN_LIVE_LLM=1 for real-local-model coverage")
    for name in (
        "AGENTIC_RESOLVER_LLM_API_KEY",
        "AGENTIC_RESOLVER_LLM_BASE_URL",
        "AGENTIC_RESOLVER_LLM_MODEL",
        "KAZUSA_DSH_BRAIN_SHARED_SECRET",
    ):
        if not os.environ.get(name):
            pytest.fail(f"live Brain configuration is missing: {name}")


def _request(kind: str, interaction_id: str) -> dict[str, object]:
    """Return one identity-complete interaction request for the live model."""

    value = _request_mapping()
    value["kind"] = kind
    value["interaction_id"] = interaction_id
    value["nonce"] = f"nonce-{interaction_id}"
    if kind == "approval":
        value["tool_name"] = "pwsh"
        value["transient_detail"] = (
            "A one-time native workspace write was requested, but the existing "
            "conversation contains no user permission for that exact operation."
        )
    else:
        value["transient_detail"] = (
            "The bounded context states that the verification marker is three. "
            "Answer what the verification marker is directly from that context."
        )
    return value


def _live_service():
    """Compose the real cognition owners with bounded test delivery seams."""

    from kazusa_ai_chatbot import service as service_module
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
    from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService

    service_module._dsh_cognition_services = (
        service_module.build_cognition_core_services()
    )

    async def deliver(_surface, request):
        return {
            "platform_message_id": f"delivered-{request.interaction_id}",
            "delivered_at": request.issued_at,
        }

    async def continue_resolution(**fields):
        return {
            "status": "continued",
            "resolution_thread_id": fields["resolution_thread_id"],
            "segment_id": fields["segment_id"],
        }

    async def issue_authority(_request, _row, _grant):
        return f"live-continuation-{uuid4().hex}"

    return BrainInteractionService.from_environment(
        judge=BrainDecisionEngine(judge=service_module._production_dsh_judge),
        reply_judge=service_module._production_dsh_reply_judge,
        deliver=deliver,
        context_provider=service_module._production_dsh_context,
        continue_resolution=continue_resolution,
        issue_continuation_authority=issue_authority,
    )


async def _delete_interaction(interaction_id: str) -> None:
    """Remove one uniquely scoped live interaction fixture."""

    from kazusa_ai_chatbot.db import dsh_interactions
    from kazusa_ai_chatbot.db._client import get_db

    collection = (await get_db())[dsh_interactions.DSH_INTERACTIONS_COLLECTION]
    await collection.delete_one({"interaction_id": interaction_id})


@pytest.mark.asyncio
async def test_brain_cognition_answers_or_rejects_dsh_request_from_context() -> None:
    """Brain cognition decides a question without direct user delivery."""

    _require_live_backend()
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV1
    interaction_id = f"live-question-{uuid4().hex}"
    secret = os.environ["KAZUSA_DSH_BRAIN_SHARED_SECRET"].encode("utf-8")
    service = _live_service()
    request = DshBrainInteractionRequestV1.from_mapping(
        _request("question", interaction_id)
    )
    try:
        result = await service.handle_signed(
            sign_request(request, secret=secret)
        )
        assert result["decision"] in {"answer", "reject"}
        assert result.get("checkpoint_required") is not True
        assert "delivered-" not in json.dumps(result, ensure_ascii=False)
    finally:
        await _delete_interaction(interaction_id)


@pytest.mark.asyncio
async def test_brain_cognition_relays_ambiguous_permission_then_interprets_user_reply() -> None:
    """Brain cognition relays approval and resumes only after typed reply judgment."""

    _require_live_backend()
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV1
    interaction_id = f"live-relay-{uuid4().hex}"
    secret = os.environ["KAZUSA_DSH_BRAIN_SHARED_SECRET"].encode("utf-8")
    service = _live_service()
    request = DshBrainInteractionRequestV1.from_mapping(
        _request("approval", interaction_id)
    )
    try:
        first = await service.handle_signed(
            sign_request(request, secret=secret)
        )
        assert first["decision"] == "relay_to_user"
        assert first["checkpoint_required"] is True
        resumed = await service.handle_user_reply(
            platform="debug",
            platform_channel_id="channel-1",
            global_user_id="user-1",
            reply_to_platform_message_id=first["delivered_platform_message_id"],
            reply_platform_message_id=f"reply-{interaction_id}",
            reply_text="I approve this exact operation once.",
        )
        assert resumed["status"] in {"continued", "rejected", "waiting"}
        assert "reply_text" not in json.dumps(resumed, ensure_ascii=False)
    finally:
        await _delete_interaction(interaction_id)
