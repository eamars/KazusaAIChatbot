"""Conversation semantic service tests."""

from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_conversation_services_use_semantic_queries_opaque_refs_pagination_and_provenance() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.conversation import ConversationSemanticService
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec

    rows = [
        (-1.0, {"message_id": "m1", "body_text": "first", "timestamp": "2026-08-28T00:00:00Z", "display_name": "A", "role": "user"}),
        (-1.0, {"message_id": "m2", "body_text": "second", "timestamp": "2026-08-28T00:01:00Z", "display_name": "B", "role": "user"}),
    ]

    async def search(query, **kwargs):
        assert query == "topic"
        assert kwargs["limit"] == 2
        return rows

    service = ConversationSemanticService(
        codec=OpaqueReferenceCodec(b"conversation-test-secret"),
        search=search,
    )
    result = await service.search_conversation_history(query="topic", max_results=1)
    assert result.page.has_more is True
    assert result.page.next_page_ref
    assert result.entities[0]["entry_ref"].startswith("kr2.")
    assert result.evidence[0].semantic_ref == result.entities[0]["entry_ref"]
    second = await service.search_conversation_history(
        query="topic",
        max_results=1,
        next_page_ref=result.page.next_page_ref,
    )
    assert second.entities[0]["text"] == "second"


@pytest.mark.asyncio
async def test_authority_bound_ten_participant_page_stays_under_worker_frame_and_pages() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.conversation import ConversationSemanticService
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec
    from tests.test_dsh_tool_gateway_authority import _authority

    participants = [
        {
            "_id": f"participant-{index}",
            "display_name": f"Participant {index}",
            "message_count": 100 - index,
        }
        for index in range(11)
    ]
    requested_limits = []

    async def summarize(**kwargs):
        requested_limits.append(kwargs["limit"])
        return {"participants": participants}

    authority = _authority()
    secret = b"participant-page-secret"
    codec = OpaqueReferenceCodec(secret).with_authority(authority)
    service = ConversationSemanticService(codec=codec, summarize=summarize)

    first = await service.summarize_conversation_participants(max_people=10)
    serialized = json.dumps(
        first.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(serialized) < 32 * 1024
    assert len(first.entities) == 10
    assert len(first.evidence) == 10
    assert {
        receipt.semantic_ref
        for receipt in first.evidence
    } == {entity["person_ref"] for entity in first.entities}
    assert first.page.has_more is True
    assert first.page.next_page_ref

    restarted = OpaqueReferenceCodec(secret).with_authority(authority)
    assert [
        restarted.resolve(entity["person_ref"], "person")["source_id"]
        for entity in first.entities
    ] == [f"participant-{index}" for index in range(10)]

    second = await service.summarize_conversation_participants(
        max_people=10,
        next_page_ref=first.page.next_page_ref,
    )
    assert requested_limits == [11, 21]
    assert len(second.entities) == 1
    assert second.entities[0]["name"] == "Participant 10"
    assert second.page.has_more is False
    assert second.page.next_page_ref is None
