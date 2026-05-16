"""Live end-to-end checks for private self-cognition lifecycle actions."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from uuid import uuid4

import httpx
import pytest
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.db import close_db, get_character_profile
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.user_memory_units import insert_user_memory_units
from kazusa_ai_chatbot.self_cognition import models, runner
from kazusa_ai_chatbot.self_cognition.sources import collect_active_commitment_cases
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

RETIRED_STATUSES = frozenset(("archived", "completed", "cancelled"))


async def test_self_cognition_e2e_retires_controlled_past_due_commitment() -> None:
    """Run real cognition and execute the chosen lifecycle action against DB."""

    await _skip_if_llm_unavailable()
    now = datetime.now(timezone.utc)
    unit_id = f"live_e2e_lifecycle_{uuid4().hex}"
    global_user_id = f"live-e2e-user-{uuid4().hex}"
    created_at = "2026-04-20T00:00:00+00:00"
    due_at = "2026-05-01T00:00:00+00:00"
    before_doc: dict | None = None
    after_doc: dict | None = None
    character_state_before: dict | None = None
    artifact_payloads: dict = {}
    trace_path = None

    try:
        character_state_before = await _snapshot_character_state()
        before_doc = await _insert_controlled_commitment(
            global_user_id=global_user_id,
            unit_id=unit_id,
            created_at=created_at,
            due_at=due_at,
        )
        character_profile = await get_character_profile()
        cases = await collect_active_commitment_cases(
            now=now,
            character_profile=character_profile,
            max_cases=1,
            list_active_commitments_func=lambda **kwargs: _commitment_rows(
                before_doc,
            ),
            get_conversation_history_func=lambda **kwargs: _history_rows(
                global_user_id,
            ),
            get_user_profile_func=lambda global_user_id: _user_profile(),
        )
        assert len(cases) == 1

        artifact_payloads = await runner.build_self_cognition_case_artifacts_async(
            cases[0],
            apply_consolidation=True,
            execute_private_actions=True,
        )
        after_doc = await _read_memory_unit(unit_id)
        cognition_output = artifact_payloads[models.ARTIFACT_COGNITION_OUTPUT]
        consolidation_outcome = artifact_payloads.get(
            models.ARTIFACT_CONSOLIDATION_OUTCOME
        )
        trace_path = write_llm_trace(
            "self_cognition_memory_lifecycle_e2e_live_llm",
            unit_id,
            {
                "case": cases[0],
                "before_doc": before_doc,
                "cognition_output": cognition_output,
                "consolidation_outcome": consolidation_outcome,
                "run_record": artifact_payloads[models.ARTIFACT_RUN_RECORD],
                "route_effect": artifact_payloads[models.ARTIFACT_ROUTE_EFFECT],
                "after_doc": after_doc,
                "judgment": (
                    "manual_review_required_for_end_to_end_lifecycle_goal"
                ),
            },
        )

        action_specs = cognition_output.get("action_specs")
        action_results = cognition_output.get("action_results")
        lifecycle_specs = _rows_for_kind(
            action_specs,
            "memory_lifecycle_update",
            field_name="kind",
        )
        lifecycle_results = _rows_for_kind(
            action_results,
            "memory_lifecycle_update",
            field_name="action_kind",
        )

        assert lifecycle_specs, f"trace={trace_path}"
        assert lifecycle_specs[0]["visibility"] == "private"
        assert lifecycle_results, f"trace={trace_path}"
        assert lifecycle_results[0]["status"] == "executed"
        assert consolidation_outcome is not None, f"trace={trace_path}"
        assert after_doc is not None
        assert after_doc["status"] in RETIRED_STATUSES
        assert after_doc["status"] != "active"
        assert models.ARTIFACT_ACTION_CANDIDATE not in artifact_payloads
    finally:
        await _delete_synthetic_user_state(global_user_id)
        await _restore_character_state(character_state_before)
        await close_db()


async def _skip_if_llm_unavailable() -> None:
    """Skip the live test when the configured local LLM is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}; {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{COGNITION_LLM_BASE_URL}"
        )


async def _insert_controlled_commitment(
    *,
    global_user_id: str,
    unit_id: str,
    created_at: str,
    due_at: str,
) -> dict:
    """Insert one synthetic active commitment for live lifecycle validation."""

    unit = {
        "unit_id": unit_id,
        "unit_type": "active_commitment",
        "fact": (
            '角色曾承诺继续提醒一个临时测试窗口；测试窗口已经结束，'
            '用户后来明确说这件事不用再提醒，可以关闭。'
        ),
        "subjective_appraisal": (
            '这条承诺继续保持 active 只会让待办列表失真。'
        ),
        "relationship_signal": (
            '关闭这条过期承诺比继续打扰用户更符合当前关系。'
        ),
        "status": "active",
        "due_at": due_at,
        "source_refs": [
            {
                "source": "live_e2e_lifecycle_test",
                "summary": "Synthetic active commitment for controlled test.",
            }
        ],
    }
    docs = await insert_user_memory_units(
        global_user_id,
        [unit],
        timestamp=created_at,
        include_embeddings=False,
    )
    inserted_doc = docs[0]
    return inserted_doc


def _controlled_history(global_user_id: str) -> list[dict]:
    """Return recent visible context that makes abandonment semantically valid."""

    history = [
        {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "platform_user_id": "live-e2e-user",
            "global_user_id": global_user_id,
            "display_name": "live lifecycle test user",
            "role": "user",
            "text": '这个临时测试窗口已经结束了，之前那个提醒不用继续留着。',
            "body_text": '这个临时测试窗口已经结束了，之前那个提醒不用继续留着。',
            "timestamp": "2026-05-02T00:00:00+00:00",
        }
    ]
    return history


async def _commitment_rows(before_doc: dict | None) -> list[dict]:
    """Return the inserted test commitment through the source reader seam."""

    if before_doc is None:
        return_value: list[dict] = []
        return return_value
    return_value = [before_doc]
    return return_value


async def _history_rows(global_user_id: str) -> list[dict]:
    """Return controlled recent context through the source reader seam."""

    history = _controlled_history(global_user_id)
    return history


async def _user_profile() -> dict:
    """Return a bounded profile for the synthetic test user."""

    profile = {
        "affinity": 900,
        "display_name": "live lifecycle test user",
        "last_relationship_insight": "",
    }
    return profile


async def _read_memory_unit(unit_id: str) -> dict | None:
    """Read one memory unit without Mongo internals or embeddings."""

    db = await get_db()
    doc = await db.user_memory_units.find_one(
        {"unit_id": unit_id},
        {"_id": 0, "embedding": 0},
    )
    return doc


async def _delete_synthetic_user_state(global_user_id: str) -> None:
    """Remove user-scoped rows created for the synthetic live test."""

    if not global_user_id.startswith("live-e2e-user-"):
        raise ValueError("refusing to delete non-test user state")
    db = await get_db()
    await db.user_memory_units.delete_many({"global_user_id": global_user_id})
    await db.user_profiles.delete_many({"global_user_id": global_user_id})


async def _snapshot_character_state() -> dict | None:
    """Capture the singleton character state before live consolidation."""

    db = await get_db()
    doc = await db.character_state.find_one({"_id": "global"})
    if doc is None:
        return_value = None
        return return_value
    return_value = dict(doc)
    return return_value


async def _restore_character_state(snapshot: dict | None) -> None:
    """Restore the singleton character state after live consolidation."""

    db = await get_db()
    if snapshot is None:
        await db.character_state.delete_one({"_id": "global"})
        return
    await db.character_state.replace_one(
        {"_id": "global"},
        snapshot,
        upsert=True,
    )


def _rows_for_kind(
    rows: object,
    action_kind: str,
    *,
    field_name: str,
) -> list[dict]:
    """Return action rows whose kind field matches the requested action."""

    if not isinstance(rows, list):
        return_value: list[dict] = []
        return return_value
    matches = [
        row for row in rows
        if isinstance(row, dict) and row.get(field_name) == action_kind
    ]
    return matches
