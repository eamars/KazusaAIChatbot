"""RAG3 media-subagent projection coverage."""

import pytest

from kazusa_ai_chatbot.local_context_resolver.subagent.media import MediaSubagent
from kazusa_ai_chatbot.media_inspection.session_cache import (
    clear_session_media,
    put_session_media,
)


@pytest.mark.asyncio
async def test_current_media_cache_miss_returns_bounded_evidence() -> None:
    """Report missing media without invoking an inspector or leaking a ref."""

    scope = ("debug", "channel-1", "user-1")
    clear_session_media(scope)
    subagent = MediaSubagent()
    result = await subagent.run({
        "schema_version": "local_context_subagent_request.v1",
        "node_id": "task_1",
        "subagent": "media",
        "action": "inspect_media",
        "objective": "Describe the current image.",
        "payload": {
            "selector": {
                "selector_kind": "current",
                "ordinal": 1,
                "question": "Describe the current image.",
            },
        },
        "constraints": {},
    }, {
        "platform": scope[0],
        "platform_channel_id": scope[1],
        "global_user_id": scope[2],
    })

    assert result["status"] == "unavailable"
    assert result["result"]["artifacts"] == []
    assert "cache_miss" in result["result"]["node_update"]["evidence_boundary_notes"]


@pytest.mark.asyncio
async def test_current_media_projection_strips_cache_ref() -> None:
    """Expose aliases and observations without trusted payload lookup fields."""

    scope = ("debug", "channel-1", "user-1")
    clear_session_media(scope)
    put_session_media(scope, [{
        "media_kind": "image",
        "content_type": "image/png",
        "base64_data": "aGVsbG8=",
        "source_summary": "current upload",
    }])
    subagent = MediaSubagent(inspect_func=_answered_inspection)
    result = await subagent.run({
        "schema_version": "local_context_subagent_request.v1",
        "node_id": "task_1",
        "subagent": "media",
        "action": "inspect_media",
        "objective": "What color is visible?",
        "payload": {
            "selector": {
                "selector_kind": "current",
                "ordinal": 1,
                "question": "What color is visible?",
            },
        },
        "constraints": {},
    }, {
        "platform": scope[0],
        "platform_channel_id": scope[1],
        "global_user_id": scope[2],
    })

    artifact = result["result"]["artifacts"][0]
    assert artifact["projection_payload"]["answer"] == "Blue is visible."
    media_evidence = artifact["projection_payload"]["media_evidence"][0]
    assert media_evidence["alias"] == "current_media_1"
    assert "cache_ref" not in media_evidence


async def _answered_inspection(request: dict[str, object]) -> dict[str, object]:
    """Return deterministic inspector output for RAG3 projection coverage."""

    return {
        "schema_version": "media_inspection_result.v1",
        "status": "answered",
        "answer": "Blue is visible.",
        "evidence_boundary_notes": [],
    }
