"""Latest-only runtime integration for character identity consumers."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.character_identity_growth import runtime
from kazusa_ai_chatbot.db.character_identity_growth import (
    IdentityPostCommitPendingError,
    IdentityRevisionStaleError,
)
from tests.cognition_core_v2_test_helpers import (
    canonical_character_identity,
)


def _revision(
    *,
    marker: str,
    revision_number: int,
) -> dict[str, object]:
    """Build one complete generic revision with a unique latest marker."""

    identity = canonical_character_identity(marker=marker)
    return {
        "revision_number": revision_number,
        "effective_identity": identity,
        "promotion_run_id": f"run-{revision_number}",
    }


def _consumed_receipt(kwargs: dict[str, object]) -> dict[str, object]:
    """Build the receipt returned by a patched transactional owner."""

    return {
        "episode_id": kwargs["episode_id"],
        "correlation_id": kwargs["correlation_id"],
        "claimed_at": "2026-07-28T00:00:00Z",
        "loaded_revision_number": kwargs["loaded_revision_number"],
        "consumer_kinds": list(kwargs["consumer_kinds"]),
        "projection_digest": kwargs["projection_digest"],
        "status": "consumed",
    }


@pytest.mark.asyncio
async def test_stale_load_retries_once_and_returns_only_revision_n(
    monkeypatch,
) -> None:
    """A promotion race should replace N-1 before any cognition consumer runs."""

    old_revision = _revision(marker="old", revision_number=1)
    latest_revision = _revision(marker="new", revision_number=2)
    get_current = AsyncMock(side_effect=[old_revision, latest_revision])
    claim_calls = 0

    async def claim(**kwargs):
        nonlocal claim_calls
        claim_calls += 1
        if claim_calls == 1:
            raise IdentityRevisionStaleError(
                loaded_revision_number=1,
                latest_revision_number=2,
            )
        return _consumed_receipt(kwargs)

    monkeypatch.setattr(runtime, "get_current_identity", get_current)
    monkeypatch.setattr(
        runtime,
        "get_character_runtime_state",
        AsyncMock(return_value={
            "cognition_state": {"state_scope": "character"},
            "updated_at": "2026-07-28T00:00:00Z",
        }),
    )
    monkeypatch.setattr(
        runtime,
        "claim_identity_revision_consumption",
        claim,
    )
    record_event = AsyncMock(return_value={"accepted": True})
    monkeypatch.setattr(
        runtime,
        "record_character_identity_growth_event",
        record_event,
    )

    snapshot = await runtime.load_latest_identity_for_episode(
        episode_id="episode-1",
        correlation_id="correlation-1",
        include_epistemic_core=False,
        character_id="character-1",
    )

    assert snapshot["revision_number"] == 2
    assert "new" in str(snapshot["cognition_context"])
    assert "old" not in str(snapshot["cognition_context"])
    assert "new" in str(snapshot["surface_context"])
    assert "old" not in str(snapshot["surface_context"])
    assert get_current.await_count == 2
    assert claim_calls == 2
    record_event.assert_awaited_once()
    assert (
        record_event.await_args.kwargs["projection_digest"]
        == snapshot["projection_digest"]
    )


@pytest.mark.asyncio
async def test_pending_post_commit_is_reconciled_before_consumption(
    monkeypatch,
) -> None:
    """A committed revision becomes visible only after its side effects finish."""

    revision = _revision(marker="latest", revision_number=3)
    claim_calls = 0

    async def claim(**kwargs):
        nonlocal claim_calls
        claim_calls += 1
        if claim_calls == 1:
            raise IdentityPostCommitPendingError(
                run_id="run-3",
                revision_number=3,
            )
        return _consumed_receipt(kwargs)

    monkeypatch.setattr(
        runtime,
        "get_current_identity",
        AsyncMock(return_value=revision),
    )
    monkeypatch.setattr(
        runtime,
        "get_character_runtime_state",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        runtime,
        "claim_identity_revision_consumption",
        claim,
    )
    reconcile = AsyncMock(return_value={
        "completed_count": 1,
        "failed_count": 0,
    })
    monkeypatch.setattr(
        runtime,
        "reconcile_identity_growth_post_commit",
        reconcile,
    )
    monkeypatch.setattr(
        runtime,
        "record_character_identity_growth_event",
        AsyncMock(return_value={"accepted": True}),
    )

    snapshot = await runtime.load_latest_identity_for_episode(
        episode_id="episode-3",
        correlation_id="correlation-3",
        include_epistemic_core=True,
        character_id="character-1",
    )

    assert snapshot["revision_number"] == 3
    assert snapshot["cognition_context"][
        "epistemic_comparison_memory"
    ]
    reconcile.assert_awaited_once_with(run_id="run-3")
    assert claim_calls == 2


def test_snapshot_state_update_is_detached_and_episode_scoped() -> None:
    """Resolver recurrence should reuse one detached episode snapshot."""

    revision = _revision(marker="latest", revision_number=4)
    cognition = runtime.project_identity_for_cognition(revision)
    surface = runtime.project_identity_for_surface(revision)
    snapshot = {
        "revision_number": 4,
        "character_profile": deepcopy(revision["effective_identity"]),
        "cognition_context": cognition,
        "surface_context": surface,
        "projection_digest": runtime.identity_projection_digest(
            revision_number=4,
            cognition_context=cognition,
            surface_context=surface,
        ),
        "consumer_kinds": (
            runtime.projected_identity_consumer_kinds(cognition)
        ),
    }

    update = runtime.snapshot_state_update(
        snapshot,
        episode_id="episode-4",
        include_epistemic_core=False,
    )
    update["character_profile"]["name"] = "mutated"

    assert snapshot["character_profile"]["name"] != "mutated"
    assert update["character_identity_episode_id"] == "episode-4"
    assert (
        update["character_identity_revision_number"]
        == snapshot["revision_number"]
    )
