"""Live disposable-DB proof for the Stage 3 internal-thought latch."""

from __future__ import annotations

import os
from datetime import timedelta
from uuid import uuid4

import pytest

from tests.stage3_fresh_database import validate_stage3_environment


pytestmark = [pytest.mark.asyncio, pytest.mark.live_db]


async def test_stage3_claimed_latch_recovers_without_duplicate_consumption() -> None:
    """A restart-style lease recovery must leave one consumable latch."""

    required_guard_names = (
        "STAGE3_TEST_MONGODB_URI",
        "STAGE3_TEST_MONGODB_DB_NAME",
        "STAGE3_TEST_MONGODB_ENDPOINT_FINGERPRINT",
        "PRODUCTION_MONGODB_ENDPOINT_FINGERPRINT",
        "CHARACTER_PROFILE_PATH",
    )
    missing_guard_names = [
        name for name in required_guard_names if not os.environ.get(name)
    ]
    if missing_guard_names:
        pytest.skip(
            "Stage 3 disposable database guard is not configured: "
            + ", ".join(missing_guard_names)
        )
    guarded_environment = validate_stage3_environment()
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["MONGODB_URI"] = guarded_environment["mongodb_uri"]
    os.environ["MONGODB_DB_NAME"] = guarded_environment["database_name"]
    os.environ["CHARACTER_PROFILE_PATH"] = (
        guarded_environment["character_profile_path"]
    )

    from kazusa_ai_chatbot.db import close_db, db_bootstrap
    from kazusa_ai_chatbot.db.internal_action_latches import (
        claim_due_internal_action_latch,
        consume_internal_action_latch,
        issue_internal_action_latch,
    )
    from kazusa_ai_chatbot.time_boundary import (
        parse_storage_utc_datetime,
        storage_utc_now_iso,
    )

    await db_bootstrap()
    now = storage_utc_now_iso()
    source_action_attempt_id = f"stage3-live-{uuid4().hex}"
    try:
        latch = await issue_internal_action_latch(
            source_episode_id=f"episode-{uuid4().hex}",
            source_action_attempt_id=source_action_attempt_id,
            continuation_objective="continue one grounded technical check",
            evidence_refs=[],
            target_scope={
                "platform": "debug",
                "platform_channel_id": "stage3-live",
                "channel_type": "private",
                "current_global_user_id": "stage3-user",
            },
            privacy_scope="private",
            continuation_depth=0,
            now=now,
        )
        first_claim = await claim_due_internal_action_latch(
            worker_id="stage3-worker-a",
            now=now,
        )
        assert first_claim is not None
        assert first_claim["latch"]["latch_id"] == latch["latch_id"]

        claim_time = parse_storage_utc_datetime(now) + timedelta(seconds=301)
        second_claim = await claim_due_internal_action_latch(
            worker_id="stage3-worker-b",
            now=claim_time.isoformat(),
        )
        assert second_claim is not None
        assert second_claim["latch"]["latch_id"] == latch["latch_id"]
        assert second_claim["claim_token"] != first_claim["claim_token"]

        with pytest.raises(ValueError, match="stale"):
            await consume_internal_action_latch(
                latch_id=latch["latch_id"],
                claim_token=first_claim["claim_token"],
                consumed_episode_id=f"episode-stale-{uuid4().hex}",
                now=claim_time.isoformat(),
            )

        consumed = await consume_internal_action_latch(
            latch_id=latch["latch_id"],
            claim_token=second_claim["claim_token"],
            consumed_episode_id=f"episode-consumed-{uuid4().hex}",
            now=claim_time.isoformat(),
        )
        assert consumed["status"] == "consumed"
        assert consumed["attempt_count"] == 2
    finally:
        await close_db()
