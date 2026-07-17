"""Regression coverage for frozen real-conversation replay clocks."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from experiments.cognition_core_v2_real_conversation_replay import (
    _align_historical_assistant_timeline,
)


@pytest.mark.asyncio
async def test_historical_replay_aligns_only_generated_response_rows() -> None:
    """A frozen reply remains before the next captured source turn."""

    update_many = AsyncMock(
        return_value=SimpleNamespace(modified_count=2),
    )
    database = SimpleNamespace(
        conversation_history=SimpleNamespace(update_many=update_many),
    )

    aligned_count = await _align_historical_assistant_timeline(
        database,
        delivery_tracking_id="delivery-11",
        source_timestamp_utc="2026-07-15T23:31:15.228696Z",
    )

    assert aligned_count == 2
    update_many.assert_awaited_once_with(
        {"delivery_tracking_id": "delivery-11"},
        {
            "$set": {
                "timestamp": "2026-07-15T23:31:15.228696Z",
            }
        },
    )


@pytest.mark.asyncio
async def test_historical_replay_skips_silent_response_alignment() -> None:
    """A silent response has no persisted assistant timeline to adjust."""

    update_many = AsyncMock()
    database = SimpleNamespace(
        conversation_history=SimpleNamespace(update_many=update_many),
    )

    aligned_count = await _align_historical_assistant_timeline(
        database,
        delivery_tracking_id="",
        source_timestamp_utc="2026-07-15T23:31:15.228696Z",
    )

    assert aligned_count == 0
    update_many.assert_not_awaited()
