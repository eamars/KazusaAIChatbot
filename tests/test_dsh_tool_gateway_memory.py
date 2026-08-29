"""Memory semantic service tests."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_memory_services_search_read_remember_revise_and_change_lifecycle_without_storage_vocabulary() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.memory import MemorySemanticService
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec

    rows = {
        "memory-1": {
            "memory_unit_id": "memory-1",
            "lineage_id": "lineage-1",
            "version": 1,
            "memory_name": "tea",
            "content": "Alice likes tea",
            "memory_type": "fact",
            "status": "active",
        }
    }

    async def search(query, **kwargs):
        return [(0.9, rows["memory-1"])]

    async def read(memory_id):
        return rows.get(memory_id)

    async def insert(*, document):
        rows[document["memory_unit_id"]] = document
        return document

    async def revise(*, active_unit_id, replacement):
        rows[replacement["memory_unit_id"]] = replacement
        return replacement

    statuses = {}

    async def lifecycle(memory_id, fields):
        statuses[memory_id] = fields["status"]

    service = MemorySemanticService(
        codec=OpaqueReferenceCodec(b"memory-test-secret"),
        source_global_user_id="user-1",
        search=search,
        read=read,
        insert=insert,
        revise=revise,
        update_lifecycle=lifecycle,
    )
    found = await service.search_memories(query="tea")
    reference = found.entities[0]["memory_ref"]
    assert "memory_unit_id" not in str(found.to_dict())
    read_result = await service.read_memories(memory_refs=[reference])
    assert read_result.entities[0]["information"] == "Alice likes tea"
    remembered = await service.remember_information(
        subject="current_user",
        information="Bob likes coffee",
        memory_kind="profile_fact",
        reason="explicit",
        provenance={"current_task": "test"},
        idempotency_key="i1",
    )
    assert remembered.mutation.outcome == "committed"
    again = await service.remember_information(
        subject="current_user",
        information="Bob likes coffee",
        memory_kind="profile_fact",
        reason="explicit",
        provenance={"current_task": "test"},
        idempotency_key="i1",
    )
    assert again.mutation.semantic_ref == remembered.mutation.semantic_ref
    assert again.mutation.outcome == "already_committed"
    revised = await service.revise_memory(memory_ref=reference, revised_information="Alice likes green tea", reason="updated", idempotency_key="i2")
    assert revised.mutation.outcome == "committed"
    changed = await service.change_memory_lifecycle(memory_ref=reference, transition="complete", reason="done", idempotency_key="i3")
    assert changed.mutation.semantic_ref == reference
