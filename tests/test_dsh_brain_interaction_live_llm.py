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


def _question_request(interaction_id: str) -> dict[str, object]:
    """Return one identity-complete question request for the live model."""

    value = _request_mapping()
    value["kind"] = "question"
    value["interaction_id"] = interaction_id
    value["nonce"] = f"nonce-{interaction_id}"
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

    return BrainInteractionService.from_environment(
        judge=BrainDecisionEngine(judge=service_module._production_dsh_judge),
        context_provider=service_module._production_dsh_context,
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
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV2
    interaction_id = f"live-question-{uuid4().hex}"
    secret = os.environ["KAZUSA_DSH_BRAIN_SHARED_SECRET"].encode("utf-8")
    service = _live_service()
    request = DshBrainInteractionRequestV2.from_mapping(
        _question_request(interaction_id)
    )
    try:
        result = await service.handle_signed(
            sign_request(request, secret=secret)
        )
        assert result["decision"] in {"answer", "reject"}
        assert result["kind"] == "question"
        assert set(result) == {
            "schema_version",
            "interaction_id",
            "request_digest",
            "kind",
            "decision",
            "answer",
            "reason",
        }
        assert "checkpoint" not in json.dumps(result, ensure_ascii=False)
    finally:
        await _delete_interaction(interaction_id)
