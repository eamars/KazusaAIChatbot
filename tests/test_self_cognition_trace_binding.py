from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.self_cognition import worker


@pytest.mark.asyncio
async def test_self_cognition_trace_binds_parent_and_calendar_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_trace_run = AsyncMock()
    monkeypatch.setattr(
        worker.llm_tracing,
        "ensure_llm_trace_run",
        ensure_trace_run,
    )
    case = {
        "case_id": "case-future-1",
        "target_scope": {
            "platform": "debug",
            "platform_channel_id": "channel-1",
            "channel_type": "private",
            "user_id": "user-1",
        },
        "source_llm_trace_id": "llmtrace-parent-1",
        "source_calendar_run_id": "calendar-run-1",
    }

    await worker._ensure_self_cognition_trace(case, "llmtrace-child-1")

    ensure_trace_run.assert_awaited_once_with(
        trace_id="llmtrace-child-1",
        platform="debug",
        platform_channel_id="channel-1",
        channel_type="private",
        platform_message_id="self_cognition:case-future-1",
        global_user_id="user-1",
        parent_llm_trace_id="llmtrace-parent-1",
        source_calendar_run_id="calendar-run-1",
    )
