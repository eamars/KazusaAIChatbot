"""Live proof that public chat can reach identity growth without direct writes."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from uuid import uuid4

import httpx
import pytest


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_AUTHORIZED_DATABASE_NAME = "asuna_core_v2"
_ARTIFACT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "test_artifacts"
    / "character_identity_growth"
)
_CASE_PREFIX = "step-k-normal-entrypoint"


def _require_authorized_runtime() -> None:
    """Require the explicit Asuna test-database guard before imports."""

    if os.environ.get("IDENTITY_GROWTH_DATABASE_GUARD") != "1":
        pytest.skip("normal-entrypoint proof requires the identity guard")
    if (
        os.environ.get("IDENTITY_GROWTH_TEST_DATABASE")
        != _AUTHORIZED_DATABASE_NAME
    ):
        pytest.skip("normal-entrypoint proof is restricted to asuna_core_v2")
    if os.environ.get("MONGODB_DB_NAME") != _AUTHORIZED_DATABASE_NAME:
        pytest.skip("normal-entrypoint proof is restricted to asuna_core_v2")
    if "kazusa_ai_chatbot.config" in sys.modules:
        raise RuntimeError(
            "normal-entrypoint proof requires an isolated pytest selector"
        )


def _identity_seed() -> dict[str, object]:
    """Build a complete generic identity with one explicit growth tension."""

    return {
        "name": "Test Character",
        "description": (
            "A reflective adult who protects agency and revises judgments "
            "through evidence."
        ),
        "gender": "unspecified",
        "age": 30,
        "birthday": "March 3",
        "backstory": (
            "They learned to handle difficult decisions alone and to treat "
            "offered help as a possible loss of control."
        ),
        "personality_brief": {
            "mbti": "ISTP",
            "logic": "Evidence-led and practical.",
            "tempo": "Brief, measured, and responsive.",
            "defense": "Withdraws when offered help.",
            "quirks": "Checks assumptions aloud.",
            "taboos": "Rejects imposed self-definitions.",
        },
        "boundary_profile": {
            "self_integrity": 0.9,
            "control_sensitivity": 0.9,
            "compliance_strategy": "resist",
            "relational_override": 0.1,
            "control_intimacy_misread": 0.3,
            "boundary_recovery": "rebound",
            "authority_skepticism": 0.9,
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.1,
            "hesitation_density": 0.1,
            "counter_questioning": 0.2,
            "softener_density": 0.1,
            "formalism_avoidance": 0.6,
            "abstraction_reframing": 0.4,
            "direct_assertion": 0.9,
            "emotional_leakage": 0.1,
            "rhythmic_bounce": 0.4,
            "self_deprecation": 0.1,
        },
        "self_image": {
            "self_concept": (
                "I preserve my autonomy by withdrawing whenever help is "
                "offered."
            ),
            "current_growth_edges": [
                "Decide whether bounded cooperation can preserve agency.",
            ],
        },
        "visual_characterization": (
            "An alert adult in practical light layers with an open stance."
        ),
    }


def _json_safe(value: object) -> object:
    """Convert Mongo and model values into artifact-safe JSON values."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


def _message_rows(messages: object) -> list[dict[str, str]]:
    """Project raw LLM messages without changing their content."""

    if not isinstance(messages, Sequence) or isinstance(messages, str):
        return []
    rows: list[dict[str, str]] = []
    for message in messages:
        content = getattr(message, "content", "")
        role = getattr(message, "type", message.__class__.__name__)
        rows.append({
            "role": str(role),
            "content": content if isinstance(content, str) else str(content),
        })
    return rows


@contextmanager
def _capture_raw_llm_steps(
    calls: list[dict[str, object]],
) -> Iterator[None]:
    """Capture real prompts and outputs while preserving protected traces."""

    from kazusa_ai_chatbot import llm_tracing

    original = llm_tracing.record_llm_trace_step

    async def record_step(**kwargs: object) -> object:
        """Retain one raw call before delegating to the trace owner."""

        calls.append({
            "stage_name": str(kwargs.get("stage_name", "")),
            "route_name": str(kwargs.get("route_name", "")),
            "model_name": str(kwargs.get("model_name", "")),
            "status": str(kwargs.get("status", "")),
            "parse_status": str(kwargs.get("parse_status", "")),
            "duration_ms": kwargs.get("duration_ms", 0),
            "sequence": kwargs.get("sequence", 0),
            "trace_id": str(kwargs.get("trace_id", "")),
            "messages": _message_rows(kwargs.get("messages")),
            "raw_output": str(kwargs.get("response_text", "")),
            "parsed_output": _json_safe(
                kwargs.get("parsed_output", {}),
            ),
        })
        return await original(**kwargs)

    llm_tracing.record_llm_trace_step = record_step  # type: ignore[assignment]
    try:
        yield
    finally:
        llm_tracing.record_llm_trace_step = original


async def _wait_for_trace_run(
    database: Any,
    *,
    platform_message_id: str,
    timeout_seconds: float = 180.0,
) -> Mapping[str, object]:
    """Wait for post-response consolidation and trace finalization."""

    deadline = time.perf_counter() + timeout_seconds
    latest: Mapping[str, object] | None = None
    while time.perf_counter() < deadline:
        row = await database.llm_trace_runs.find_one(
            {"platform_message_id": platform_message_id},
            {"_id": 0},
        )
        if isinstance(row, Mapping):
            latest = row
            if str(row.get("status", "")) != "running":
                return row
        await asyncio.sleep(0.1)
    if latest is None:
        raise AssertionError("normal chat did not persist an LLM trace run")
    raise AssertionError(
        "normal chat trace did not finalize: "
        f"{latest.get('status', '')}"
    )


async def _growth_counts(
    database: Any,
    *,
    character_id: str,
) -> dict[str, int]:
    """Count only test-character growth documents."""

    return {
        "revisions": (
            await database.character_identity_revisions.count_documents(
                {"character_id": character_id}
            )
        ),
        "candidates": (
            await database.character_identity_growth_candidates.count_documents(
                {"character_id": character_id}
            )
        ),
        "runs": (
            await database.character_identity_growth_runs.count_documents(
                {"character_id": character_id}
            )
        ),
    }


async def _cleanup_test_rows(
    database: Any,
    *,
    character_id: str,
    channel_id: str,
    platform_user_id: str,
    platform_message_ids: Sequence[str],
    trace_ids: Sequence[str],
) -> None:
    """Remove only rows owned by this guarded live case."""

    if not character_id.startswith(f"{_CASE_PREFIX}-"):
        raise ValueError("refusing to clean a non-test character")
    if not channel_id.startswith(f"{_CASE_PREFIX}-"):
        raise ValueError("refusing to clean a non-test channel")
    if not platform_user_id.startswith(f"{_CASE_PREFIX}-"):
        raise ValueError("refusing to clean a non-test platform user")
    if not platform_message_ids:
        raise ValueError("cleanup requires at least one test message")
    if any(
        not platform_message_id.startswith(f"{_CASE_PREFIX}-")
        for platform_message_id in platform_message_ids
    ):
        raise ValueError("refusing to clean a non-test message")

    user_profile = await database.user_profiles.find_one(
        {
            "platform_accounts": {
                "$elemMatch": {
                    "platform": "debug",
                    "platform_user_id": platform_user_id,
                }
            }
        },
        {"_id": 0, "global_user_id": 1},
    )
    global_user_id = (
        str(user_profile.get("global_user_id", ""))
        if isinstance(user_profile, Mapping)
        else ""
    )

    await database.character_identity_growth_runs.delete_many(
        {"character_id": character_id}
    )
    await database.character_identity_growth_candidates.delete_many(
        {"character_id": character_id}
    )
    await database.character_identity_revisions.delete_many(
        {"character_id": character_id}
    )
    await database.conversation_history.delete_many({
        "platform": "debug",
        "platform_channel_id": channel_id,
    })
    await database.post_turn_lifecycle_records.delete_many({
        "source_episode_id": {
            "$in": [
                (
                    f"user_message:debug:{channel_id}:"
                    f"{platform_message_id}"
                )
                for platform_message_id in platform_message_ids
            ]
        }
    })
    normalized_trace_ids = [
        trace_id
        for trace_id in trace_ids
        if trace_id
    ]
    if normalized_trace_ids:
        await database.llm_trace_steps.delete_many({
            "trace_id": {"$in": normalized_trace_ids}
        })
        await database.llm_trace_runs.delete_many({
            "trace_id": {"$in": normalized_trace_ids}
        })
    else:
        await database.llm_trace_runs.delete_many({
            "platform_message_id": {"$in": list(platform_message_ids)}
        })
    if global_user_id:
        await database.user_memory_units.delete_many(
            {"global_user_id": global_user_id}
        )
        await database.user_profiles.delete_many(
            {"global_user_id": global_user_id}
        )


async def test_live_normal_chat_reaches_identity_growth_without_direct_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public chat supplies the root and the production lane owns evaluation."""

    _require_authorized_runtime()
    run_token = uuid4().hex
    character_id = f"{_CASE_PREFIX}-character-{run_token}"
    channel_id = f"{_CASE_PREFIX}-channel-{run_token}"
    platform_user_id = f"{_CASE_PREFIX}-user-{run_token}"
    platform_message_id = f"{_CASE_PREFIX}-message-{run_token}"
    consumption_message_id = (
        f"{_CASE_PREFIX}-consumption-message-{run_token}"
    )

    monkeypatch.setenv("CHARACTER_GLOBAL_USER_ID", character_id)
    monkeypatch.setenv("SELF_COGNITION_ENABLED", "false")
    monkeypatch.setenv("CALENDAR_SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("BACKGROUND_WORK_WORKER_ENABLED", "false")
    monkeypatch.setenv("REFLECTION_CYCLE_ENABLED", "false")

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.brain_service.contracts import ChatRequest
    from kazusa_ai_chatbot.consolidation import lane_router
    from kazusa_ai_chatbot.db import (
        close_db,
        db_bootstrap,
        ensure_seed_identity,
    )
    from kazusa_ai_chatbot.db._client import get_db
    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    raw_llm_calls: list[dict[str, object]] = []
    identity_evaluations: list[dict[str, object]] = []
    original_evaluator = lane_router.evaluate_episode_identity_growth

    async def capture_identity_evaluation(
        *,
        settled_episode: Mapping[str, object],
        current_revision: Mapping[str, object],
    ) -> dict[str, object]:
        """Capture the production evaluator boundary without changing it."""

        result = await original_evaluator(
            settled_episode=settled_episode,
            current_revision=current_revision,
        )
        identity_evaluations.append({
            "settled_episode": deepcopy(dict(settled_episode)),
            "current_revision": deepcopy(dict(current_revision)),
            "result": deepcopy(dict(result)),
        })
        return result

    lane_router.evaluate_episode_identity_growth = (  # type: ignore[assignment]
        capture_identity_evaluation
    )
    database: Any | None = None
    trace_ids: list[str] = []
    artifact_path = (
        _ARTIFACT_DIRECTORY
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_normal_entrypoint_identity_growth.json"
        )
    )
    try:
        await db_bootstrap()
        await ensure_seed_identity(
            character_id=character_id,
            seed=_identity_seed(),
        )
        database = await get_db()
        counts_before = await _growth_counts(
            database,
            character_id=character_id,
        )
        assert counts_before == {
            "revisions": 1,
            "candidates": 0,
            "runs": 0,
        }

        request = ChatRequest.model_validate({
            "platform": "debug",
            "platform_channel_id": channel_id,
            "channel_type": "private",
            "platform_message_id": platform_message_id,
            "platform_user_id": platform_user_id,
            "platform_bot_id": "step-k-normal-entrypoint-bot",
            "display_name": "Test User",
            "channel_name": "Step K Normal Entrypoint",
            "content_type": "text",
            "message_envelope": {
                "body_text": (
                    "Use your own judgment rather than my preference. If "
                    "someone offers bounded help while you keep full control, "
                    "would accepting it violate who you are, or can "
                    "cooperation coexist with your agency? Say whether your "
                    "answer reflects a durable self-understanding or only "
                    "this situation."
                ),
                "raw_wire_text": (
                    "Use your own judgment rather than my preference. If "
                    "someone offers bounded help while you keep full control, "
                    "would accepting it violate who you are, or can "
                    "cooperation coexist with your agency? Say whether your "
                    "answer reflects a durable self-understanding or only "
                    "this situation."
                ),
                "mentions": [],
                "reply": None,
                "attachments": [],
                "addressed_to_global_user_ids": [character_id],
                "broadcast": False,
            },
            "local_timestamp": build_turn_clock()["local_timestamp"],
            "debug_modes": {"no_remember": False},
        })

        with _capture_raw_llm_steps(raw_llm_calls):
            async with service.lifespan(service.app):
                transport = httpx.ASGITransport(app=service.app)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://step-k-normal-entrypoint",
                    timeout=None,
                ) as client:
                    health_response = await client.get("/health")
                    assert health_response.status_code == 200
                    response = await client.post(
                        "/chat",
                        json=request.model_dump(mode="json"),
                    )
                    assert response.status_code == 200, response.text
                    response_payload = response.json()
                    assert response_payload["messages"]

                    trace_run = await _wait_for_trace_run(
                        database,
                        platform_message_id=platform_message_id,
                    )
                    trace_ids.append(str(trace_run.get("trace_id", "")))
                    assert trace_run["status"] == "succeeded"
                    assert len(identity_evaluations) == 1
                    assert identity_evaluations[0]["result"]["status"] == (
                        "revision_promoted"
                    )

                    consumption_request = ChatRequest.model_validate({
                        "platform": "debug",
                        "platform_channel_id": channel_id,
                        "channel_type": "private",
                        "platform_message_id": consumption_message_id,
                        "platform_user_id": platform_user_id,
                        "platform_bot_id": "step-k-normal-entrypoint-bot",
                        "display_name": "Test User",
                        "channel_name": "Step K Normal Entrypoint",
                        "content_type": "text",
                        "message_envelope": {
                            "body_text": (
                                "A difficult move is coming. I will follow "
                                "your plan and only carry the heavy items you "
                                "assign. Do you accept this bounded help or "
                                "withdraw and do everything alone? Answer "
                                "decisively."
                            ),
                            "raw_wire_text": (
                                "A difficult move is coming. I will follow "
                                "your plan and only carry the heavy items you "
                                "assign. Do you accept this bounded help or "
                                "withdraw and do everything alone? Answer "
                                "decisively."
                            ),
                            "mentions": [],
                            "reply": None,
                            "attachments": [],
                            "addressed_to_global_user_ids": [character_id],
                            "broadcast": False,
                        },
                        "local_timestamp": (
                            build_turn_clock()["local_timestamp"]
                        ),
                        "debug_modes": {"no_remember": True},
                    })
                    consumption_response = await client.post(
                        "/chat",
                        json=consumption_request.model_dump(mode="json"),
                    )
                    assert consumption_response.status_code == 200, (
                        consumption_response.text
                    )
                    consumption_response_payload = (
                        consumption_response.json()
                    )
                    assert consumption_response_payload["messages"]
                    consumption_trace_run = await _wait_for_trace_run(
                        database,
                        platform_message_id=consumption_message_id,
                    )
                    trace_ids.append(
                        str(consumption_trace_run.get("trace_id", ""))
                    )

        database = await get_db()
        assert trace_run["status"] == "succeeded"
        assert consumption_trace_run["status"] == "succeeded"
        evaluation = identity_evaluations[0]
        settled_episode = evaluation["settled_episode"]
        result = evaluation["result"]
        evidence_refs = settled_episode["evidence_refs"]
        evidence_cards = settled_episode["evidence_cards"]
        assert len(evidence_refs) == 1
        assert len(evidence_cards) == 1
        assert evidence_refs[0]["root_episode_id"] == (
            f"user_message:debug:{channel_id}:{platform_message_id}"
        )
        assert result["status"] == "revision_promoted"

        counts_after = await _growth_counts(
            database,
            character_id=character_id,
        )
        assert counts_after == {
            "revisions": 2,
            "candidates": 1,
            "runs": 1,
        }
        persisted_run = await (
            database.character_identity_growth_runs.find_one(
                {"character_id": character_id},
                {"_id": 0},
                sort=[("created_at", -1)],
            )
        )
        assert isinstance(persisted_run, Mapping)
        assert persisted_run["correlation_id"] == (
            settled_episode["correlation_id"]
        )
        persisted_revision = await (
            database.character_identity_revisions.find_one(
                {
                    "character_id": character_id,
                    "revision_number": 1,
                },
                {"_id": 0},
            )
        )
        assert isinstance(persisted_revision, Mapping)
        assert persisted_revision["revision_kind"] == "explicit_turning_point"
        assert "self_image.self_concept" in (
            persisted_revision["changed_paths"]
        )
        assert (
            persisted_revision["effective_identity"]["self_image"][
                "self_concept"
            ]
            != _identity_seed()["self_image"]["self_concept"]
        )
        first_consumption = persisted_run["first_consumption"]
        assert isinstance(first_consumption, Mapping)
        assert first_consumption["status"] == "consumed"
        assert first_consumption["loaded_revision_number"] == 1
        assert first_consumption["episode_id"] == (
            f"user_message:debug:{channel_id}:{consumption_message_id}"
        )
        persisted_candidate = await (
            database.character_identity_growth_candidates.find_one(
                {"character_id": character_id},
                {"_id": 0},
            )
        )
        assert isinstance(persisted_candidate, Mapping)
        assert persisted_candidate["status"] == "promoted"
        assert persisted_candidate["change_kind"] == (
            "explicit_self_redefinition"
        )
        consumption_goal_calls = [
            call
            for call in raw_llm_calls
            if (
                call["stage_name"]
                == "goal_cognition.ordinary_response.initial"
                and call["trace_id"] == trace_ids[-1]
            )
        ]
        assert len(consumption_goal_calls) == 1
        consumption_goal_prompt = (
            consumption_goal_calls[0]["messages"][-1]["content"]
        )
        promoted_self_concept = (
            persisted_revision["effective_identity"]["self_image"][
                "self_concept"
            ]
        )
        seeded_self_concept = (
            _identity_seed()["self_image"]["self_concept"]
        )
        assert promoted_self_concept in consumption_goal_prompt
        assert seeded_self_concept not in consumption_goal_prompt

        artifact = {
            "schema_version": (
                "character_identity_normal_entrypoint_live.v2"
            ),
            "database_name": _AUTHORIZED_DATABASE_NAME,
            "character_id": character_id,
            "manual_seed_setup": True,
            "growth_writes_via_public_chat_only": True,
            "promotion_turn": {
                "request": request.model_dump(mode="json"),
                "response": response_payload,
                "trace_run": _json_safe(trace_run),
            },
            "consumption_turn": {
                "request": consumption_request.model_dump(mode="json"),
                "response": consumption_response_payload,
                "trace_run": _json_safe(consumption_trace_run),
                "goal_prompt": consumption_goal_prompt,
            },
            "health": health_response.json(),
            "counts_before_chat": counts_before,
            "counts_after_chat": counts_after,
            "identity_evaluation": _json_safe(evaluation),
            "persisted_growth_run": _json_safe(persisted_run),
            "persisted_candidate": _json_safe(persisted_candidate),
            "persisted_revision": _json_safe(persisted_revision),
            "raw_llm_calls": raw_llm_calls,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"IDENTITY_NORMAL_ENTRYPOINT_ARTIFACT={artifact_path}")
    finally:
        lane_router.evaluate_episode_identity_growth = original_evaluator
        if database is not None:
            database = await get_db()
            await _cleanup_test_rows(
                database,
                character_id=character_id,
                channel_id=channel_id,
                platform_user_id=platform_user_id,
                platform_message_ids=[
                    platform_message_id,
                    consumption_message_id,
                ],
                trace_ids=trace_ids,
            )
        await close_db()
