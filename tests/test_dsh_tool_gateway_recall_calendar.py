"""Active-context and calendar semantic service tests."""

from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_recall_and_calendar_services_return_semantic_entries_and_opaque_pagination() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.recall_calendar import RecallCalendarSemanticService
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec

    async def recall(**kwargs):
        return [{"id": "ctx-1", "claim": "finish the plan", "status": "active"}]

    async def schedules(**kwargs):
        return [{"schedule_id": "s1", "title": "review", "status": "active"}, {"schedule_id": "s2", "title": "ship", "status": "active"}]

    service = RecallCalendarSemanticService(
        codec=OpaqueReferenceCodec(b"calendar-test-secret"),
        recall=recall,
        schedules=schedules,
    )
    context = await service.recall_active_context(kinds=["commitments"])
    assert context.entities[0]["context_ref"].startswith("kr2.")
    page = await service.read_calendar_context(view="schedules", max_results=1)
    assert page.page.has_more is True
    assert page.page.next_page_ref


@pytest.mark.asyncio
async def test_authority_bound_ten_item_calendar_page_stays_under_worker_frame_and_pages() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec
    from kazusa_ai_chatbot.dsh_tool_gateway.recall_calendar import RecallCalendarSemanticService
    from tests.test_dsh_tool_gateway_authority import _authority

    schedules = [
        {
            "schedule_id": f"schedule-{index}",
            "title": f"Review calendar item {index}",
            "status": "active",
            "due_at": f"2026-08-{index + 1:02d}T09:00:00Z",
            "next_run_at": f"2026-08-{index + 1:02d}T09:00:00Z",
            "updated_at": f"2026-08-{index + 1:02d}T08:00:00Z",
        }
        for index in range(11)
    ]
    requested_limits = []

    async def schedule_rows(*, limit):
        requested_limits.append(limit)
        return schedules

    authority = _authority()
    secret = b"calendar-page-secret"
    codec = OpaqueReferenceCodec(secret).with_authority(authority)
    service = RecallCalendarSemanticService(
        codec=codec,
        schedules=schedule_rows,
    )

    first = await service.read_calendar_context(
        view="schedules",
        max_results=10,
    )
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
    } == {entity["context_ref"] for entity in first.entities}
    assert first.page.has_more is True
    assert first.page.next_page_ref

    restarted = OpaqueReferenceCodec(secret).with_authority(authority)
    assert [
        restarted.resolve(entity["context_ref"], "context")
        for entity in first.entities
    ] == [
        {"kind": "calendar-schedules", "source_id": f"schedule-{index}"}
        for index in range(10)
    ]

    second = await service.read_calendar_context(
        view="schedules",
        max_results=10,
        next_page_ref=first.page.next_page_ref,
    )
    assert requested_limits == [11, 21]
    assert len(second.entities) == 1
    assert second.entities[0]["title"] == "Review calendar item 10"
    assert second.page.has_more is False
    assert second.page.next_page_ref is None
