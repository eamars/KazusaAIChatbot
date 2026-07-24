"""Live proof that L2d can queue and run future_speak background work."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi import BackgroundTasks
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot import service as brain_service
from kazusa_ai_chatbot.action_spec.execution import execute_action_specs_for_trace
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
    FUTURE_SPEAK_CAPABILITY,
    SPEAK_CAPABILITY,
    build_initial_action_capabilities,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.background_work.worker import run_background_work_worker_tick
from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.db import (
    close_db,
    db_bootstrap,
    get_character_profile,
    resolve_global_user_id,
    split_character_profile_runtime_state,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.background_work_jobs import (
    ensure_background_work_job_indexes,
)
from kazusa_ai_chatbot.db.accepted_tasks import ensure_accepted_task_indexes
from kazusa_ai_chatbot.dispatcher import AdapterRegistry, SendResult
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_cognition_actions as action_connector,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from kazusa_ai_chatbot.self_cognition import worker as self_cognition_worker
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

logger = logging.getLogger(__name__)

_REMINDER_OBJECTIVE = "Remind the user to drink water"
_E2E_REMINDER_YEAR = 2030
_E2E_REMINDER_TEXT = (
    "Can you remind me on July 3, 2030 at 09:00 to drink water?"
)


async def test_live_l2d_future_speak_runs_real_background_worker() -> None:
    """Run real L2d, queue a real job, and schedule a real calendar run."""

    await _skip_if_llm_unavailable()
    await _skip_if_db_unavailable()
    await ensure_accepted_task_indexes()
    await ensure_background_work_job_indexes()

    run_id = uuid4().hex
    frozen_state = _future_speak_l2d_state(run_id)
    prompt_payload = build_action_selection_payload_text(frozen_state)
    services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(services.llm)
    token = l2d.set_action_selection_llm(
        LLMStageBinding(capturing_llm, services.action_selection_config)
    )

    job_id = ""
    accepted_task_id = ""
    calendar_schedule_id = ""
    calendar_run_id = ""
    db = await get_db()
    try:
        result = await select_semantic_actions(frozen_state)
        raw_output = capturing_llm.raw_output
        raw_parsed_output = parse_llm_json_output(raw_output)
        action_specs = action_connector.materialize_semantic_action_requests(
            result.get("semantic_action_requests", []),
            frozen_state,
        )
        observed_kinds = [spec["kind"] for spec in action_specs]
        future_specs = [
            spec for spec in action_specs
            if spec.get("kind") == FUTURE_SPEAK_CAPABILITY
        ]
        assert len(future_specs) == 1

        queue_results = await execute_action_specs_for_trace(
            future_specs,
            storage_timestamp_utc=frozen_state["storage_timestamp_utc"],
        )
        queue_result = queue_results[0]
        accepted_task = await db.accepted_tasks.find_one(
            {
                "action_kind": FUTURE_SPEAK_CAPABILITY,
                "source_channel_id": frozen_state["platform_channel_id"],
                "requester_global_user_id": frozen_state["global_user_id"],
                "requester_platform_user_id": frozen_state["platform_user_id"],
            },
            {"_id": 0},
        )
        assert accepted_task is not None
        accepted_task_id = _text(accepted_task.get("accepted_task_id"))
        job_id = _text(accepted_task.get("executor_ref"))
        duplicate_results = await execute_action_specs_for_trace(
            future_specs,
            storage_timestamp_utc=frozen_state["storage_timestamp_utc"],
        )
        duplicate_result = duplicate_results[0]
        job_count_after_duplicate = await db.background_work_jobs.count_documents(
            {"accepted_task_id": accepted_task_id}
        )
        status_specs = action_connector.materialize_semantic_action_requests(
            [
                {
                    "capability": ACCEPTED_TASK_STATUS_CHECK_CAPABILITY,
                    "decision": "check_active_task",
                    "detail": "Check the active reminder task.",
                    "reason": "The user asks whether the reminder is active.",
                }
            ],
            frozen_state,
        )
        status_results = await execute_action_specs_for_trace(
            status_specs,
            storage_timestamp_utc=frozen_state["storage_timestamp_utc"],
        )
        status_result = status_results[0]
        job_count_after_status = await db.background_work_jobs.count_documents(
            {"accepted_task_id": accepted_task_id}
        )
        await db.background_work_jobs.update_one(
            {"job_id": job_id},
            {"$set": {"created_at": "1970-01-01T00:00:00+00:00"}},
        )

        worker_tick_result = await run_background_work_worker_tick(
            claim_limit=1,
            lease_seconds=60,
            max_attempts=3,
            worker_id=f"future-speak-live-{run_id}",
        )
        job = await db.background_work_jobs.find_one(
            {"job_id": job_id},
            {"_id": 0},
        )
        worker_metadata = _mapping(job, "worker_metadata")
        calendar_schedule_id = _text(
            worker_metadata.get("calendar_schedule_id")
        )
        calendar_run_id = _text(worker_metadata.get("calendar_run_id"))
        calendar_schedule = await db.calendar_schedules.find_one(
            {"schedule_id": calendar_schedule_id},
            {"_id": 0},
        )
        calendar_run = await db.calendar_runs.find_one(
            {"run_id": calendar_run_id},
            {"_id": 0},
        )
        accepted_task_after_worker = await db.accepted_tasks.find_one(
            {"accepted_task_id": accepted_task_id},
            {"_id": 0},
        )
        trace_path = write_llm_trace(
            "background_work_future_speak_live_llm",
            "l2d_to_background_worker_to_calendar",
            {
                "run_id": run_id,
                "prompt_payload": prompt_payload,
                "raw_model_output": raw_output,
                "raw_parsed_output": raw_parsed_output,
                "parsed_result": result,
                "materialized_action_specs": action_specs,
                "observed_kinds": observed_kinds,
                "queue_results": queue_results,
                "duplicate_results": duplicate_results,
                "status_results": status_results,
                "job_count_after_duplicate": job_count_after_duplicate,
                "job_count_after_status": job_count_after_status,
                "accepted_task": accepted_task,
                "accepted_task_after_worker": accepted_task_after_worker,
                "worker_tick_result": worker_tick_result,
                "background_job": job,
                "calendar_schedule": calendar_schedule,
                "calendar_run": calendar_run,
                "judgment": (
                    "real_llm_and_real_db_proof_for_future_speak_background_work"
                ),
            },
        )
        logger.info(
            f"BACKGROUND_WORK_FUTURE_SPEAK_LIVE trace={trace_path} "
            f"kinds={json.dumps(observed_kinds, ensure_ascii=True)} "
            f"job_id={job_id} calendar_run_id={calendar_run_id}"
        )

        assert SPEAK_CAPABILITY in observed_kinds
        assert FUTURE_SPEAK_CAPABILITY in observed_kinds
        assert queue_result["status"] == "pending"
        assert queue_result["accepted_task_state"] == "scheduled"
        assert _objective_text(queue_result["accepted_task_summary"]) == (
            _REMINDER_OBJECTIVE
        )
        assert queue_result["acknowledgement_constraint"] == "promise_allowed"
        assert accepted_task["state"] == "pending"
        assert accepted_task["executor_kind"] == "background_work"
        assert accepted_task["executor_ref"] == job_id
        assert duplicate_result["status"] == "pending"
        assert duplicate_result["accepted_task_state"] == "already_active"
        assert duplicate_result["acknowledgement_constraint"] == (
            "progress_report_allowed"
        )
        assert status_result["status"] == "executed"
        assert status_result["accepted_task_state"] == "scheduled"
        assert status_result["acknowledgement_constraint"] == (
            "progress_report_allowed"
        )
        assert job_count_after_duplicate == 1
        assert job_count_after_status == 1
        assert worker_tick_result["processed_count"] == 1
        assert worker_tick_result["succeeded_count"] == 1
        assert job is not None
        assert job["accepted_task_id"] == accepted_task_id
        assert job["worker"] == "future_speak"
        assert job["status"] == "completed"
        assert job["delivery_state"] == "delivered"
        assert accepted_task_after_worker is not None
        assert accepted_task_after_worker["state"] == "delivered"
        assert "active_identity_key" not in accepted_task_after_worker
        assert calendar_schedule is not None
        assert calendar_run is not None
        assert calendar_run["trigger_kind"] == "future_cognition"
        assert calendar_run["status"] == "pending"
        assert _objective_text(
            calendar_run["payload"]["continuation_objective"]
        ) == _REMINDER_OBJECTIVE
    finally:
        l2d.reset_action_selection_llm(token)
        await _cleanup_live_rows(
            job_id=job_id,
            calendar_schedule_id=calendar_schedule_id,
            calendar_run_id=calendar_run_id,
            accepted_task_id=accepted_task_id,
        )
        await close_db()


async def test_live_user_message_future_speak_e2e_feedback_handover() -> None:
    """Run user input through future_speak, due cognition, and adapter handover."""

    await _skip_if_llm_unavailable()
    await _skip_if_db_unavailable()
    await db_bootstrap()
    await ensure_accepted_task_indexes()
    await ensure_background_work_job_indexes()

    run_id = uuid4().hex
    platform = "debug"
    platform_channel_id = f"debug:user:future-speak-e2e-{run_id}"
    platform_user_id = f"debug-user-{run_id}"
    platform_bot_id = "debug-bot-future-speak-e2e"
    display_name = "Live Future Speak E2E User"
    global_user_id = ""
    job_id = ""
    accepted_task_id = ""
    calendar_schedule_id = ""
    calendar_run_id = ""
    trace_path = None
    adapter = _FakeDebugMessagingAdapter(platform_bot_id=platform_bot_id)
    adapter_registry = AdapterRegistry()
    adapter_registry.register(adapter)

    db = await get_db()
    try:
        global_user_id = await resolve_global_user_id(
            platform=platform,
            platform_user_id=platform_user_id,
            display_name=display_name,
        )
        character_profile = await get_character_profile()
        if not character_profile.get("name"):
            pytest.fail("Character profile is missing from MongoDB.")
        (
            brain_service._static_character_profile,
            brain_service._runtime_character_state,
        ) = split_character_profile_runtime_state(character_profile)
        brain_service._graph = brain_service._build_graph()

        request = _future_speak_chat_request(
            platform=platform,
            platform_channel_id=platform_channel_id,
            platform_user_id=platform_user_id,
            platform_bot_id=platform_bot_id,
            platform_message_id=f"message-future-speak-e2e-{run_id}",
            display_name=display_name,
        )
        background_tasks = BackgroundTasks()
        response = await brain_service.chat(request, background_tasks)
        for task in background_tasks.tasks:
            await task()

        accepted_task = await db.accepted_tasks.find_one(
            {
                "action_kind": FUTURE_SPEAK_CAPABILITY,
                "source_platform": platform,
                "source_channel_id": platform_channel_id,
                "requester_global_user_id": global_user_id,
                "requester_platform_user_id": platform_user_id,
            },
            {"_id": 0},
        )
        assert accepted_task is not None
        accepted_task_id = _text(accepted_task.get("accepted_task_id"))
        job_id = _text(accepted_task.get("executor_ref"))

        job_count_after_chat = await db.background_work_jobs.count_documents(
            {"accepted_task_id": accepted_task_id}
        )
        await db.background_work_jobs.update_one(
            {"job_id": job_id},
            {"$set": {"created_at": "1970-01-01T00:00:00+00:00"}},
        )
        background_worker_result = await run_background_work_worker_tick(
            claim_limit=1,
            lease_seconds=60,
            max_attempts=3,
            worker_id=f"future-speak-e2e-{run_id}",
        )
        background_job = await db.background_work_jobs.find_one(
            {"job_id": job_id},
            {"_id": 0},
        )
        worker_metadata = _mapping(background_job, "worker_metadata")
        calendar_schedule_id = _text(
            worker_metadata.get("calendar_schedule_id")
        )
        calendar_run_id = _text(worker_metadata.get("calendar_run_id"))
        calendar_run_after_schedule = await db.calendar_runs.find_one(
            {"run_id": calendar_run_id},
            {"_id": 0},
        )
        if calendar_run_after_schedule is None:
            cleanup_preview = await _count_live_scope_rows(
                platform=platform,
                platform_channel_id=platform_channel_id,
                global_user_id=global_user_id,
                accepted_task_id=accepted_task_id,
                job_id=job_id,
                calendar_schedule_id=calendar_schedule_id,
                calendar_run_id=calendar_run_id,
            )
            trace_path = write_llm_trace(
                "background_work_future_speak_e2e_live_llm",
                "user_message_to_due_feedback_handover_schedule_failure",
                {
                    "run_id": run_id,
                    "input_message": _E2E_REMINDER_TEXT,
                    "chat_request": request.model_dump(),
                    "chat_response": response.model_dump(),
                    "accepted_task": accepted_task,
                    "job_count_after_chat": job_count_after_chat,
                    "background_worker_result": background_worker_result,
                    "background_job": background_job,
                    "calendar_run_after_schedule": calendar_run_after_schedule,
                    "cleanup_preview": cleanup_preview,
                    "judgment": (
                        "schedule_handoff_failed_before_due_self_cognition"
                    ),
                },
            )
            logger.info(
                f"BACKGROUND_WORK_FUTURE_SPEAK_E2E_LIVE_FAILED "
                f"trace={trace_path} job_id={job_id} "
                f"calendar_run_id={calendar_run_id}"
            )
        assert calendar_run_after_schedule is not None
        scheduled_due_at_raw = calendar_run_after_schedule["due_at"]
        assert isinstance(scheduled_due_at_raw, str)
        scheduled_due_at = datetime.fromisoformat(scheduled_due_at_raw)
        assert scheduled_due_at.year == _E2E_REMINDER_YEAR
        due_time = (
            scheduled_due_at + timedelta(minutes=1)
        )

        self_cognition_result = await self_cognition_worker.run_self_cognition_worker_tick(
            now=due_time,
            is_primary_interaction_busy=lambda: False,
            character_profile=dict(character_profile),
            adapter_registry_provider=lambda: adapter_registry,
            max_cases=1,
        )
        calendar_run_after_due = await db.calendar_runs.find_one(
            {"run_id": calendar_run_id},
            {"_id": 0},
        )
        final_attempt = await db.self_cognition_action_attempts.find_one(
            {
                "source_kind": "scheduled_future_cognition_slot",
                "target_scope.platform_channel_id": platform_channel_id,
            },
            {"_id": 0},
            sort=[("updated_at", -1)],
        )
        final_conversation_rows = await db.conversation_history.find(
            {
                "platform": platform,
                "platform_channel_id": platform_channel_id,
            },
            {"_id": 0},
        ).sort("timestamp", 1).to_list(length=20)
        cleanup_preview = await _count_live_scope_rows(
            platform=platform,
            platform_channel_id=platform_channel_id,
            global_user_id=global_user_id,
            accepted_task_id=accepted_task_id,
            job_id=job_id,
            calendar_schedule_id=calendar_schedule_id,
            calendar_run_id=calendar_run_id,
        )
        trace_path = write_llm_trace(
            "background_work_future_speak_e2e_live_llm",
            "user_message_to_due_feedback_handover",
            {
                "run_id": run_id,
                "input_message": _E2E_REMINDER_TEXT,
                "chat_request": request.model_dump(),
                "chat_response": response.model_dump(),
                "accepted_task": accepted_task,
                "job_count_after_chat": job_count_after_chat,
                "background_worker_result": background_worker_result,
                "background_job": background_job,
                "calendar_run_after_schedule": calendar_run_after_schedule,
                "self_cognition_worker_result": (
                    self_cognition_result.__dict__
                ),
                "calendar_run_after_due": calendar_run_after_due,
                "final_action_attempt": final_attempt,
                "adapter_calls": adapter.calls,
                "conversation_rows": final_conversation_rows,
                "cleanup_preview": cleanup_preview,
                "judgment": (
                    "real_user_message_to_future_speak_to_due_cognition_"
                    "to_adapter_handover"
                ),
            },
        )
        logger.info(
            f"BACKGROUND_WORK_FUTURE_SPEAK_E2E_LIVE trace={trace_path} "
            f"job_id={job_id} calendar_run_id={calendar_run_id} "
            f"adapter_calls={len(adapter.calls)}"
        )

        assert response.messages
        assert accepted_task["state"] == "pending"
        assert accepted_task["executor_kind"] == "background_work"
        assert accepted_task["executor_ref"] == job_id
        assert job_count_after_chat == 1
        assert background_worker_result["processed_count"] == 1
        assert background_worker_result["succeeded_count"] == 1
        assert background_job is not None
        assert background_job["worker"] == "future_speak"
        assert background_job["status"] == "completed"
        assert calendar_run_after_schedule["trigger_kind"] == "future_cognition"
        assert self_cognition_result.processed_count == 1
        assert calendar_run_after_due is not None
        assert calendar_run_after_due["status"] == "completed"
        assert adapter.calls
        assert adapter.calls[-1]["channel_id"] == platform_channel_id
        assert adapter.calls[-1]["channel_type"] == "private"
        assert _text(adapter.calls[-1]["text"])
        assert final_attempt is not None
        assert final_attempt["status"] == "sent"
    finally:
        await brain_service._stop_chat_input_worker()
        await _cleanup_live_rows(
            job_id=job_id,
            calendar_schedule_id=calendar_schedule_id,
            calendar_run_id=calendar_run_id,
            accepted_task_id=accepted_task_id,
            platform=platform,
            platform_channel_id=platform_channel_id,
            platform_user_id=platform_user_id,
            global_user_id=global_user_id,
        )
        await close_db()


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured cognition endpoint is unavailable."""

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


async def _skip_if_db_unavailable() -> None:
    """Skip when MongoDB is unavailable for the live background-worker proof."""

    try:
        db = await get_db()
        await db.command("ping")
    except PyMongoError as exc:
        pytest.skip(f"MongoDB unavailable for live future_speak test: {exc}")


async def _cleanup_live_rows(
    *,
    job_id: str,
    calendar_schedule_id: str,
    calendar_run_id: str,
    accepted_task_id: str,
    platform: str = "",
    platform_channel_id: str = "",
    platform_user_id: str = "",
    global_user_id: str = "",
) -> None:
    """Remove live proof rows so the scheduled reminder is not delivered later."""

    db = await get_db()
    if accepted_task_id:
        await db.accepted_tasks.delete_one({"accepted_task_id": accepted_task_id})
    if job_id:
        await db.background_work_jobs.delete_one({"job_id": job_id})
    if calendar_schedule_id:
        await db.calendar_schedules.delete_one(
            {"schedule_id": calendar_schedule_id}
        )
    if calendar_run_id:
        await db.calendar_runs.delete_one({"run_id": calendar_run_id})
    if platform_channel_id:
        await db.accepted_tasks.delete_many(
            {"source_channel_id": platform_channel_id}
        )
        await db.background_work_jobs.delete_many(
            {"source_channel_id": platform_channel_id}
        )
        await db.calendar_schedules.delete_many(
            {"source_scope.source_channel_id": platform_channel_id}
        )
        await db.calendar_runs.delete_many(
            {"source_scope.source_channel_id": platform_channel_id}
        )
        await db.conversation_history.delete_many(
            {
                "platform": platform,
                "platform_channel_id": platform_channel_id,
            }
        )
        await db.self_cognition_action_attempts.delete_many(
            {"target_scope.platform_channel_id": platform_channel_id}
        )
    if global_user_id:
        await db.user_profiles.delete_one({"global_user_id": global_user_id})
    if platform and platform_user_id:
        await db.user_profiles.update_many(
            {},
            {
                "$pull": {
                    "platform_accounts": {
                        "platform": platform,
                        "platform_user_id": platform_user_id,
                    }
                }
            },
        )


async def _count_live_scope_rows(
    *,
    platform: str,
    platform_channel_id: str,
    global_user_id: str,
    accepted_task_id: str,
    job_id: str,
    calendar_schedule_id: str,
    calendar_run_id: str,
) -> dict[str, int]:
    """Count exact live E2E rows before cleanup for trace review."""

    db = await get_db()
    accepted_task_filter = {
        "$or": [
            {"accepted_task_id": accepted_task_id},
            {"source_channel_id": platform_channel_id},
        ]
    }
    background_job_filter = {
        "$or": [
            {"job_id": job_id},
            {"source_channel_id": platform_channel_id},
        ]
    }
    calendar_schedule_filter = {
        "$or": [
            {"schedule_id": calendar_schedule_id},
            {"source_scope.source_channel_id": platform_channel_id},
        ]
    }
    calendar_run_filter = {
        "$or": [
            {"run_id": calendar_run_id},
            {"source_scope.source_channel_id": platform_channel_id},
        ]
    }
    counts = {
        "accepted_task_rows": await db.accepted_tasks.count_documents(
            accepted_task_filter
        ),
        "background_job_rows": await db.background_work_jobs.count_documents(
            background_job_filter
        ),
        "calendar_schedule_rows": await db.calendar_schedules.count_documents(
            calendar_schedule_filter
        ),
        "calendar_run_rows": await db.calendar_runs.count_documents(
            calendar_run_filter
        ),
        "conversation_rows": await db.conversation_history.count_documents(
            {
                "platform": platform,
                "platform_channel_id": platform_channel_id,
            }
        ),
        "self_cognition_action_attempt_rows": (
            await db.self_cognition_action_attempts.count_documents(
                {"target_scope.platform_channel_id": platform_channel_id}
            )
        ),
        "user_profile_rows": await db.user_profiles.count_documents(
            {"global_user_id": global_user_id}
        ),
    }
    return counts


def _future_speak_chat_request(
    *,
    platform: str,
    platform_channel_id: str,
    platform_user_id: str,
    platform_bot_id: str,
    platform_message_id: str,
    display_name: str,
) -> brain_service.ChatRequest:
    """Build the service-level user input for the E2E reminder proof."""

    message_envelope = {
        "body_text": _E2E_REMINDER_TEXT,
        "raw_wire_text": _E2E_REMINDER_TEXT,
        "mentions": [],
        "attachments": [],
        "addressed_to_global_user_ids": [
            brain_service.CHARACTER_GLOBAL_USER_ID,
        ],
        "broadcast": False,
    }
    request = brain_service.ChatRequest(
        platform=platform,
        platform_channel_id=platform_channel_id,
        channel_type="private",
        platform_message_id=platform_message_id,
        platform_user_id=platform_user_id,
        platform_bot_id=platform_bot_id,
        display_name=display_name,
        channel_name="private reminder E2E",
        message_envelope=message_envelope,
        debug_modes={
            "listen_only": False,
            "think_only": False,
            "no_remember": True,
        },
    )
    return request


def _future_speak_l2d_state(run_id: str) -> dict[str, Any]:
    """Build one frozen upstream L2d state for an accepted reminder request."""

    state: dict[str, Any] = {
        "character_profile": {
            "name": "杏山千纱",
            "mood": "calm",
            "vibe_check": "focused and helpful",
        },
        "storage_timestamp_utc": "2026-07-02T00:00:00+00:00",
        "local_time_context": {
            "current_local_datetime": "2026-07-02 12:00",
            "current_local_weekday": "Thursday",
        },
        "prompt_message_context": {
            "body_text": "Can you remind me tomorrow at 09:00 to drink water?",
            "broadcast": False,
        },
        "cognitive_episode": {
            "episode_id": f"future-speak-live-{run_id}",
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
        },
        "platform": "debug",
        "platform_channel_id": f"debug:user:future-speak-live-{run_id}",
        "channel_type": "private",
        "platform_message_id": f"message-future-speak-live-{run_id}",
        "platform_user_id": f"debug-user-{run_id}",
        "global_user_id": f"global-user-{run_id}",
        "user_name": "Live Future Speak User",
        "user_profile": {
            "display_name": "Live Future Speak User",
            "relationship_state": 100,
            "semantic_relationship_projection": "Test user for future-speak proof.",
        },
        "platform_bot_id": "debug-bot-future-speak-live",
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "private reminder request",
        "referents": [],
        "decontextualized_input": (
            "The user asks Kazusa to remind them tomorrow at 09:00 to drink "
            "water."
        ),
        "media_summary": "",
        "logical_stance": "ACCEPT_FUTURE_REMINDER",
        "character_intent": "SCHEDULE_FUTURE_SPEAK",
        "judgment_note": (
            "The request is an accepted user-facing future reminder. The "
            "action layer should select one visible speak acknowledgement and "
            "one private future_speak action. The future_speak decision must "
            "be exactly 2026-07-03 09:00 and the detail must be exactly "
            "Remind the user to drink water. Do not use "
            "trigger_future_cognition for this user-facing future message."
        ),
        "internal_monologue": (
            "They want a reminder tomorrow morning. I can acknowledge it now "
            "and set a future speak slot for that exact time."
        ),
        "emotional_appraisal": "A straightforward reminder request.",
        "interaction_subtext": "The user expects acknowledgement now and a later reminder.",
        "boundary_core_assessment": {
            "boundary_issue": "none",
            "acceptance": "allow",
            "behavior_primary": "comply",
        },
        "social_distance": "private direct request",
        "emotional_intensity": "low",
        "vibe_check": "practical and calm",
        "relational_dynamic": "ordinary reminder handoff",
        "conversation_progress": {
            "source": "future_speak_live_llm_fixture",
            "status": "accepted_future_reminder",
            "next_affordances": [
                "acknowledge the reminder",
                "schedule future_speak",
            ],
        },
        "rag_result": {
            "answer": "",
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "recall_evidence": [],
            "user_image": {
                "user_memory_context": {
                    "active_commitments": [],
                },
            },
        },
        "resolver_context": "",
        "available_action_affordances": project_prompt_affordances(
            build_initial_action_capabilities()
        ),
        "max_action_requests": 3,
        "max_resolver_requests": 3,
        "background_work_output_char_limit": 4000,
    }
    return state


def _mapping(value: object, field_name: str) -> dict[str, object]:
    """Return one nested mapping from an optional DB row."""

    if not isinstance(value, dict):
        return_value: dict[str, object] = {}
        return return_value
    nested = value.get(field_name)
    if not isinstance(nested, dict):
        return_value = {}
        return return_value
    return_value = dict(nested)
    return return_value


def _text(value: object) -> str:
    """Return one stripped text value."""

    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _objective_text(value: object) -> str:
    """Normalize one reminder objective without weakening its semantics."""

    return_value = _text(value).rstrip(".").strip()
    return return_value


class _CapturingLLM:
    """Capture raw LLM output while preserving the production call path."""

    def __init__(self, inner_llm: object) -> None:
        self._inner_llm = inner_llm
        self.raw_output = ""

    async def ainvoke(self, messages: object, *, config=None) -> object:
        """Call the wrapped LLM and store the raw message content."""

        response = await self._inner_llm.ainvoke(messages, config=config)
        self.raw_output = str(getattr(response, "content", ""))
        return response


class _FakeDebugMessagingAdapter:
    """Capture self-cognition dispatcher sends for the live E2E proof."""

    platform = "debug"
    display_name = "杏山千纱"

    def __init__(self, *, platform_bot_id: str) -> None:
        self.platform_bot_id = platform_bot_id
        self.calls: list[dict[str, Any]] = []

    async def can_send_message(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        """Accept any generated debug channel used by this test."""

        del channel_id, channel_type
        return_value = True
        return return_value

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: list[dict[str, Any]] | None = None,
    ) -> SendResult:
        """Record the final handover text and return adapter metadata."""

        self.calls.append({
            "channel_id": channel_id,
            "text": text,
            "channel_type": channel_type,
            "reply_to_msg_id": reply_to_msg_id,
            "delivery_mentions": delivery_mentions or [],
        })
        result = SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id=f"adapter-message-{uuid4().hex}",
            sent_at=datetime.now(timezone.utc),
        )
        return result
