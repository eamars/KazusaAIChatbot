"""Live E2E proof for deferred public research and visible delivery."""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi import BackgroundTasks
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot import service as brain_service
from kazusa_ai_chatbot.background_work.delivery import (
    run_background_work_delivery_tick,
)
from kazusa_ai_chatbot.background_work.worker import (
    run_background_work_worker_tick,
)
from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL, SEARXNG_URL
from kazusa_ai_chatbot.db import (
    close_db,
    db_bootstrap,
    get_character_profile,
    resolve_global_user_id,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.accepted_tasks import ensure_accepted_task_indexes
from kazusa_ai_chatbot.db.background_work_jobs import (
    ensure_background_work_job_indexes,
)
from kazusa_ai_chatbot.dispatcher import AdapterRegistry, SendResult
from kazusa_ai_chatbot.nodes import dialog_agent
from kazusa_ai_chatbot.task_resolution.specialists import (
    public_research as public_research_specialist,
)
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_USER_MESSAGE = (
    "Search the live retail price of an NVIDIA GeForce RTX 5090 right now. "
    "Report concrete prices with currencies, retailer names, and source links."
)


async def test_live_rtx5090_research_survives_background_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run user input through real research, durable resume, and delivery."""

    await _skip_if_live_dependencies_unavailable()
    await db_bootstrap()
    await ensure_accepted_task_indexes()
    await ensure_background_work_job_indexes()

    run_id = uuid4().hex
    platform = "debug"
    platform_channel_id = f"debug:rtx5090-research-e2e:{run_id}"
    platform_user_id = f"debug-user-rtx5090-{run_id}"
    platform_bot_id = "debug-bot-rtx5090-research-e2e"
    display_name = "RTX 5090 Research E2E User"
    accepted_task_id = ""
    global_user_id = ""
    job_id = ""
    evidence: dict[str, object] = {
        "run_id": run_id,
        "input_message": _USER_MESSAGE,
        "platform_channel_id": platform_channel_id,
        "quality_evaluation": "agent_review_required",
    }
    public_research_calls: list[dict[str, object]] = []
    dialog_candidate_attempts: list[dict[str, object]] = []
    original_resolve_complex_task = (
        public_research_specialist.resolve_complex_task
    )
    original_render_dialog_candidate = dialog_agent._render_dialog_candidate

    async def capture_resolve_complex_task(
        request: dict[str, object],
        context: dict[str, object],
        options: dict[str, object],
    ) -> dict[str, object]:
        call: dict[str, object] = {
            "request": request,
            "context": context,
            "options": options,
        }
        public_research_calls.append(call)
        try:
            packet = await original_resolve_complex_task(
                request,
                context,
                options,
            )
        except asyncio.CancelledError as exc:
            call["exception"] = f"{type(exc).__name__}: {exc}"
            raise
        except Exception as exc:
            call["exception"] = f"{type(exc).__name__}: {exc}"
            raise
        call["packet"] = packet
        return packet

    monkeypatch.setattr(
        public_research_specialist,
        "resolve_complex_task",
        capture_resolve_complex_task,
    )
    evidence["public_research_calls"] = public_research_calls

    async def capture_render_dialog_candidate(
        *,
        surface_output: dialog_agent.TextSurfaceOutputV2,
        user_name: str,
        repair_issues: list[str],
        attempt_number: int,
        llm_trace_id: str,
        required_source_urls: list[str] | None = None,
    ) -> tuple[list[str], str | None]:
        attempt: dict[str, object] = {
            "attempt_number": attempt_number,
            "surface_output": surface_output,
            "user_name": user_name,
            "repair_issues": repair_issues,
            "llm_trace_id": llm_trace_id,
            "required_source_urls": required_source_urls or [],
        }
        dialog_candidate_attempts.append(attempt)
        try:
            generated_dialog, failure_kind = (
                await original_render_dialog_candidate(
                    surface_output=surface_output,
                    user_name=user_name,
                    repair_issues=repair_issues,
                    attempt_number=attempt_number,
                    llm_trace_id=llm_trace_id,
                    required_source_urls=required_source_urls,
                )
            )
        except Exception as exc:
            attempt["exception"] = f"{type(exc).__name__}: {exc}"
            raise
        attempt["generated_dialog"] = generated_dialog
        attempt["failure_kind"] = failure_kind
        return generated_dialog, failure_kind

    monkeypatch.setattr(
        dialog_agent,
        "_render_dialog_candidate",
        capture_render_dialog_candidate,
    )
    evidence["dialog_candidate_attempts"] = dialog_candidate_attempts
    adapter = _DebugAdapter(platform_bot_id=platform_bot_id)
    adapter_registry = AdapterRegistry()
    adapter_registry.register(adapter)
    original_registry = brain_service._adapter_registry
    original_graph = brain_service._graph

    monkeypatch.setattr(
        capabilities,
        "TASK_RESOLUTION_INLINE_BUDGET_SECONDS",
        1.0,
    )
    db = await get_db()
    try:
        global_user_id = await resolve_global_user_id(
            platform=platform,
            platform_user_id=platform_user_id,
            display_name=display_name,
        )
        character_profile = await get_character_profile()
        assert character_profile.get("name")
        brain_service._adopt_character_profile_snapshot(character_profile)
        brain_service._graph = brain_service._build_graph()

        request = _chat_request(
            platform=platform,
            platform_channel_id=platform_channel_id,
            platform_user_id=platform_user_id,
            platform_bot_id=platform_bot_id,
            platform_message_id=f"message-rtx5090-{run_id}",
            display_name=display_name,
        )
        background_tasks = BackgroundTasks()
        chat_response = await brain_service.chat(request, background_tasks)
        for task in background_tasks.tasks:
            await task()
        evidence["chat_request"] = request.model_dump()
        evidence["chat_response"] = chat_response.model_dump()

        accepted_task = await db.accepted_tasks.find_one(
            {
                "task_kind": "task_resolution",
                "source_platform": platform,
                "source_channel_id": platform_channel_id,
                "requester_global_user_id": global_user_id,
            },
            {"_id": 0},
        )
        evidence["accepted_task_after_chat"] = accepted_task
        assert accepted_task is not None
        accepted_task_id = str(accepted_task["accepted_task_id"])
        job_id = str(accepted_task["executor_ref"])

        await db.background_work_jobs.update_one(
            {"job_id": job_id},
            {"$set": {"created_at": "1970-01-01T00:00:00+00:00"}},
        )
        worker_result = await run_background_work_worker_tick(
            claim_limit=1,
            lease_seconds=120,
            max_attempts=3,
            worker_id=f"rtx5090-research-e2e-{run_id}",
        )
        background_job = await db.background_work_jobs.find_one(
            {"job_id": job_id},
            {"_id": 0},
        )
        accepted_task_after_worker = await db.accepted_tasks.find_one(
            {"accepted_task_id": accepted_task_id},
            {"_id": 0},
        )
        evidence["worker_result"] = worker_result
        evidence["background_job_after_worker"] = background_job
        evidence["accepted_task_after_worker"] = accepted_task_after_worker

        brain_service._adapter_registry = adapter_registry
        delivery_result = await run_background_work_delivery_tick(
            deliver_result_episode_func=(
                brain_service._deliver_accepted_task_result_episode
            ),
            limit=1,
        )
        accepted_task_after_delivery = await db.accepted_tasks.find_one(
            {"accepted_task_id": accepted_task_id},
            {"_id": 0},
        )
        background_job_after_delivery = await db.background_work_jobs.find_one(
            {"job_id": job_id},
            {"_id": 0},
        )
        conversation_rows = await db.conversation_history.find(
            {
                "platform": platform,
                "platform_channel_id": platform_channel_id,
            },
            {"_id": 0},
        ).sort("timestamp", 1).to_list(length=30)
        evidence["delivery_result"] = delivery_result
        evidence["adapter_calls"] = adapter.calls
        evidence["accepted_task_after_delivery"] = accepted_task_after_delivery
        evidence["background_job_after_delivery"] = (
            background_job_after_delivery
        )
        evidence["conversation_rows"] = conversation_rows

        assert worker_result["processed_count"] == 1
        assert worker_result["succeeded_count"] == 1
        assert background_job is not None
        task_result = background_job["task_resolution_result"]
        assert task_result["status"] in {"resolved", "partial"}
        task_evidence = task_result["evidence"]
        assert isinstance(task_evidence, list)

        assert delivery_result["processed_count"] == 1
        assert delivery_result["delivered_count"] == 1
        assert accepted_task_after_delivery is not None
        assert accepted_task_after_delivery["state"] == "delivered"
        assert adapter.calls
        delivered_text = str(adapter.calls[-1]["text"])
        assert delivered_text.strip()
    finally:
        trace_path = write_llm_trace(
            "task_resolution_background_research_e2e_live_llm",
            "rtx5090_live_price_full_workflow",
            evidence,
        )
        print(f"TASK_RESOLUTION_BACKGROUND_RESEARCH_E2E={trace_path}")
        brain_service._adapter_registry = original_registry
        brain_service._graph = original_graph
        await brain_service._stop_chat_input_worker()
        await _cleanup_test_rows(
            db=db,
            accepted_task_id=accepted_task_id,
            job_id=job_id,
            platform=platform,
            platform_channel_id=platform_channel_id,
            platform_user_id=platform_user_id,
            global_user_id=global_user_id,
        )
        await close_db()


async def _skip_if_live_dependencies_unavailable() -> None:
    """Skip only when a required external live dependency is unavailable."""

    if not SEARXNG_URL:
        pytest.skip("SEARXNG_URL is not configured for live public research.")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {exc}")
    if response.status_code >= 500:
        pytest.skip(f"LLM endpoint returned {response.status_code}")
    try:
        db = await get_db()
        await db.command("ping")
    except PyMongoError as exc:
        pytest.skip(f"MongoDB is unavailable: {exc}")


def _chat_request(
    *,
    platform: str,
    platform_channel_id: str,
    platform_user_id: str,
    platform_bot_id: str,
    platform_message_id: str,
    display_name: str,
) -> brain_service.ChatRequest:
    """Build one service-level request for the live research workflow."""

    request = brain_service.ChatRequest(
        platform=platform,
        platform_channel_id=platform_channel_id,
        channel_type="private",
        platform_message_id=platform_message_id,
        platform_user_id=platform_user_id,
        platform_bot_id=platform_bot_id,
        display_name=display_name,
        channel_name="RTX 5090 live research E2E",
        message_envelope={
            "body_text": _USER_MESSAGE,
            "raw_wire_text": _USER_MESSAGE,
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [
                brain_service.CHARACTER_GLOBAL_USER_ID,
            ],
            "broadcast": False,
        },
        debug_modes={
            "listen_only": False,
            "think_only": False,
            "no_remember": True,
        },
    )
    return request


async def _cleanup_test_rows(
    *,
    db: Any,
    accepted_task_id: str,
    job_id: str,
    platform: str,
    platform_channel_id: str,
    platform_user_id: str,
    global_user_id: str,
) -> None:
    """Remove rows created under the unique debug identity for this test."""

    if accepted_task_id:
        await db.accepted_tasks.delete_one(
            {"accepted_task_id": accepted_task_id}
        )
    if job_id:
        await db.background_work_jobs.delete_one({"job_id": job_id})
    await db.conversation_history.delete_many(
        {
            "platform": platform,
            "platform_channel_id": platform_channel_id,
        }
    )
    if global_user_id:
        await db.user_profiles.delete_one({"global_user_id": global_user_id})
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


class _DebugAdapter:
    """Capture the final dispatcher handoff without contacting a real user."""

    platform = "debug"
    display_name = "RTX 5090 E2E Adapter"

    def __init__(self, *, platform_bot_id: str) -> None:
        self.platform_bot_id = platform_bot_id
        self.calls: list[dict[str, object]] = []

    async def can_send_message(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        """Accept the unique private debug channel owned by the test."""

        del channel_id, channel_type
        return True

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: list[dict[str, Any]] | None = None,
    ) -> SendResult:
        """Record visible text and return deterministic adapter metadata."""

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
