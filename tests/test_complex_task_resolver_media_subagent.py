"""Complex-task external image subagent safety coverage."""

import pytest

from kazusa_ai_chatbot.complex_task_resolver.subagent.media import MediaSubagent


@pytest.mark.asyncio
async def test_external_media_subagent_rejects_private_network_target() -> None:
    """Reject a loopback URL before any media request is made."""

    subagent = MediaSubagent()
    result = await subagent.run({
        "schema_version": "complex_task_subagent_request.v1",
        "node_id": "task_1",
        "subagent": "media",
        "action": "inspect_media",
        "objective": "Inspect the image.",
        "payload": {"url": "http://127.0.0.1/private.png", "question": "What?"},
        "constraints": {},
    }, {})

    assert result["status"] == "failed"
    assert "private" in result["result"]["evidence_boundary_notes"][0]
