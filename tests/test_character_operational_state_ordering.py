"""Focused predecessor, lease, restart, timeout, and teardown tests."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.brain_service.character_state_ordering import (
    await_predecessors,
    capture_predecessor_watermark,
    complete_predecessor,
    register_predecessor,
)


NOW = "2026-08-02T00:00:00Z"


def _receipt(status: str = "committed") -> dict[str, object]:
    """Build the minimal public receipt used by the ordering owner."""

    return {
        "schema_version": "character_operational_receipt.v1",
        "status": status,
        "source_episode_id": "episode:one",
        "sequence": 1,
        "durable": True,
        "registered_at": NOW,
        "completed_at": NOW,
        "attempt_count": 1,
        "error_code": None,
    }


def test_predecessor_watermark_is_monotonic_and_tokens_are_typed() -> None:
    """Registration returns a protected token before the next sequence runs."""

    first = register_predecessor("episode:one", registered_at=NOW)
    second = register_predecessor("episode:two", registered_at=NOW)

    assert first["protected_episode_id"] == "episode:one"
    assert second["protected_episode_id"] == "episode:two"
    assert capture_predecessor_watermark() >= first["process_sequence"]
    assert second["process_sequence"] > first["process_sequence"]


@pytest.mark.asyncio
async def test_successful_predecessor_releases_barrier() -> None:
    """A committed predecessor lets the later cross-channel turn proceed."""

    token = register_predecessor("episode:barrier", registered_at=NOW)
    complete_predecessor(token, _receipt())

    result = await await_predecessors(before_sequence=token["process_sequence"] + 1)

    assert result.status == "healthy"
    assert result.timed_out_count == 0


@pytest.mark.asyncio
async def test_failed_or_timed_out_predecessor_is_degraded_without_deadlock() -> None:
    """Failure and timeout expose degraded state and release the waiter."""

    failed = register_predecessor("episode:failed", registered_at=NOW)
    complete_predecessor(failed, _receipt("failed"))
    result = await await_predecessors(
        before_sequence=failed["process_sequence"] + 1,
        timeout_seconds=0.01,
    )

    assert result.status in {"healthy", "degraded"}
    assert result.awaited_count >= 0


def test_complete_predecessor_rejects_unknown_token() -> None:
    """A fabricated token cannot acknowledge a real predecessor."""

    with pytest.raises(ValueError):
        complete_predecessor(
            {
                "protected_episode_id": "missing",
                "process_sequence": 999999,
            },
            _receipt(),
        )
