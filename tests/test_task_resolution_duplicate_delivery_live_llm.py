"""Live reproduction for one turn producing two visible task surfaces."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest
from fastapi import BackgroundTasks
from pymongo.errors import PyMongoError
from starlette.requests import Request

from kazusa_ai_chatbot import service as brain_service
from kazusa_ai_chatbot.background_work.delivery import (
    run_background_work_delivery_tick,
)
from kazusa_ai_chatbot.background_work.worker import (
    run_background_work_worker_tick,
)
from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.cognition_core_v2 import action_selection
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
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
from kazusa_ai_chatbot.task_resolution.specialists import (
    local_context as local_context_specialist,
)
from tests.llm_trace import write_llm_trace
from tests.test_cognition_core_v2_action_planning_live_llm import (
    _action,
    _bid,
    _resolver,
    _run_case as run_live_action_planner_case,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_USER_MESSAGE = '@一之濑明日奈 对了对了，刚刚除了我们俩，群友都在聊什么呢？'
_SOURCE_MESSAGES = (
    ('千早爱音', '群里最帅的当然是我——千早爱音本音啦'),
    ('千早爱音', '吉他弹得帅，人长得也帅，嘿嘿'),
    ('阿影', '嘎嘎嘎'),
)
_HISTORICAL_PARENT_TRACE_PATH = (
    Path(__file__).resolve().parents[1]
    / "test_artifacts"
    / "diagnostics"
    / "job-d9087f00ede5413f917930b07ad14846_parent_trace.json"
)
_OBSERVED_GROUP_HISTORY_PATH = (
    Path(__file__).resolve().parents[1]
    / "test_artifacts"
    / "diagnostics"
    / "job-d9087f00ede5413f917930b07ad14846_adjacent_qq_group_history.json"
)
_HISTORICAL_REQUEST_TIMESTAMP = "2026-08-13T05:22:54.705341+00:00"
_LLM_TRACE_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "test_artifacts"
    / "llm_traces"
)


async def test_live_replays_historical_pending_task_action_plan() -> None:
    """Record whether the real planner repeats the historical route."""

    case = _load_historical_pending_task_case()
    result = await run_live_action_planner_case(
        case_id="d908_historical_pending_task_action_replay",
        user_input=case["user_input"],
        bid=case["bid"],
        actions=case["actions"],
        resolvers=case["resolvers"],
        evidence_rows=case["evidence_rows"],
        resolver_context=case["resolver_context"],
        current_goal_progress=case["current_goal_progress"],
    )
    resolver_requests = result["resolver_requests"]
    assert isinstance(resolver_requests, list)
    repeats_historical_route = bool(
        resolver_requests
        and resolver_requests[0].get("start_in_background") is True
    )
    planner_trace_path = _latest_llm_trace_path(
        "cognition_core_v2_action_planning_live_llm__"
        "d908_historical_pending_task_action_replay"
    )
    assert planner_trace_path is not None
    evidence = {
        "case_id": "d908_historical_pending_task_action_replay",
        "input": case["user_input"],
        "historical_route_repeated": repeats_historical_route,
        "model_context": _live_model_context(),
        "planner_trace_path": str(planner_trace_path),
        "planner_result": result,
    }
    trace_path = write_llm_trace(
        "task_resolution_duplicate_delivery_live_llm",
        "historical_planner_diagnostic",
        evidence,
    )
    print(f"TASK_RESOLUTION_DUPLICATE_DELIVERY_DIAGNOSTIC={trace_path}")


def _load_historical_pending_task_case() -> dict[str, object]:
    """Project the historical final action-planning prompt into live inputs."""

    if not _HISTORICAL_PARENT_TRACE_PATH.exists():
        raise AssertionError(
            f"historical parent trace is missing: {_HISTORICAL_PARENT_TRACE_PATH}"
        )
    trace = json.loads(
        _HISTORICAL_PARENT_TRACE_PATH.read_text(encoding="utf-8")
    )
    candidate_payload: dict[str, object] | None = None
    for capsule in trace["cognition_failure_capsules"]:
        for attempt in capsule.get("attempts", []):
            if attempt["stage_name"] != "action_planning.repair":
                continue
            parsed_output = attempt.get("parsed_output", {})
            resolver_requests = parsed_output.get("resolver_requests", [])
            if not resolver_requests:
                continue
            if resolver_requests[0].get("start_in_background") is not True:
                continue
            messages = attempt.get("messages", [])
            if len(messages) < 2:
                continue
            candidate_payload = json.loads(messages[1]["content"])
    if candidate_payload is None:
        raise AssertionError(
            "historical parent trace has no pending background action payload"
        )

    raw_bid = dict(candidate_payload["bids"]["b1"])
    bid = _bid(
        branch_id="ordinary_response",
        intention=str(raw_bid["intention"]),
        desired_outcome=str(raw_bid["desired_outcome"]),
        reason=str(raw_bid["reason"]),
    )
    for field_name in (
        "concrete_detail",
        "private_monologue",
        "relational_willingness",
        "expected_consequences",
        "evidence_handles",
        "confidence",
    ):
        if field_name in raw_bid:
            bid[field_name] = raw_bid[field_name]

    evidence_rows: list[dict[str, object]] = []
    for row in candidate_payload["evidence"]:
        evidence_handle = str(row["handle"])
        semantic_text = str(row["semantic_text"])
        evidence_rows.append({
            "evidence_handle": evidence_handle,
            "evidence_ref": {
                "source_kind": str(row["source_kind"]),
                "source_id": f"job-d908:{evidence_handle}",
                "occurred_at": "2026-08-13T10:00:00Z",
                "semantic_summary": semantic_text[:200],
            },
            "semantic_text": semantic_text,
            "visible_to": ["q:event_agency"],
        })
    resolver_rows = [
        _resolver(
            str(row["capability"]),
            str(row["semantic_capability"]),
        )
        for row in candidate_payload["resolver_handles"].values()
    ]
    return {
        "user_input": str(
            candidate_payload["current_resolver_goal_progress"]["original_goal"]
        ),
        "bid": bid,
        "actions": [
            _action("accepted_task_status_check"),
            _action("future_speak"),
        ],
        "resolvers": resolver_rows,
        "evidence_rows": evidence_rows,
        "resolver_context": str(candidate_payload["resolver_context"]),
        "current_goal_progress": candidate_payload[
            "current_resolver_goal_progress"
        ],
    }


async def test_live_task_resolution_duplicate_delivery_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept direct or valid background resolution with one factual result."""

    await _skip_if_live_dependencies_unavailable()
    await db_bootstrap()
    await ensure_accepted_task_indexes()
    await ensure_background_work_job_indexes()

    run_id = uuid4().hex
    platform = "debug"
    platform_channel_id = f"debug:duplicate-task-delivery:{run_id}"
    platform_user_id = f"debug-user-duplicate-task-{run_id}"
    platform_bot_id = f"debug-bot-duplicate-task-{run_id}"
    display_name = "蚝爹油"
    accepted_task_id = ""
    job_id = ""
    global_user_id = ""
    settled_traces: list[dict[str, object]] = []
    test_outcome = "failed"
    evidence: dict[str, object] = {
        "run_id": run_id,
        "input_message": _USER_MESSAGE,
        "source_messages": [
            {"display_name": name, "body_text": text}
            for name, text in _SOURCE_MESSAGES
        ],
        "platform_channel_id": platform_channel_id,
        "quality_evaluation": "agent_review_required",
        "model_context": _live_model_context(),
    }
    local_context_calls: list[dict[str, object]] = []
    original_local_context = local_context_specialist.resolve_with_local_context

    async def capture_local_context(
        request: dict[str, object],
        context: dict[str, object],
    ) -> dict[str, object]:
        """Capture the worker's real specialist input and result."""

        call: dict[str, object] = {
            "request": request,
            "context": context,
        }
        local_context_calls.append(call)
        try:
            result = await original_local_context(request, context)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            call["exception"] = f"{type(exc).__name__}: {exc}"
            raise
        call["result"] = result
        return result

    monkeypatch.setattr(
        local_context_specialist,
        "resolve_with_local_context",
        capture_local_context,
    )
    evidence["local_context_calls"] = local_context_calls

    adapter = _DebugAdapter(platform_bot_id=platform_bot_id)
    adapter_registry = AdapterRegistry()
    adapter_registry.register(adapter)
    original_registry = brain_service._adapter_registry
    original_graph = brain_service._graph
    original_settle = brain_service._settle_runtime_episode_trace

    async def capture_settled_trace(**kwargs: object) -> dict[str, object]:
        """Capture the typed trace at the production settlement boundary."""

        trace = await original_settle(**kwargs)
        settled_traces.append(trace)
        return trace

    monkeypatch.setattr(
        brain_service,
        "_settle_runtime_episode_trace",
        capture_settled_trace,
    )
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
        await _seed_source_messages(
            db=db,
            platform=platform,
            platform_channel_id=platform_channel_id,
            platform_user_id=platform_user_id,
            platform_bot_id=platform_bot_id,
            global_user_id=global_user_id,
            run_id=run_id,
        )
        character_profile = await get_character_profile()
        assert character_profile.get("name")
        brain_service._adopt_character_profile_snapshot(character_profile)
        brain_service._graph = brain_service._build_graph()
        brain_service._adapter_registry = adapter_registry

        request = _chat_request(
            platform=platform,
            platform_channel_id=platform_channel_id,
            platform_user_id=platform_user_id,
            platform_bot_id=platform_bot_id,
            platform_message_id=f"message-duplicate-task-{run_id}",
            display_name=display_name,
        )
        background_tasks = BackgroundTasks()
        http_request = Request({
            "type": "http",
            "method": "POST",
            "path": "/chat",
            "headers": [],
        })
        chat_response = await brain_service.chat(
            request,
            background_tasks,
            http_request,
        )
        for task in background_tasks.tasks:
            await task()
        evidence["chat_request"] = request.model_dump()
        evidence["chat_response"] = chat_response.model_dump()
        parent_trace_count = len(settled_traces)
        evidence["parent_settled_traces"] = settled_traces[:parent_trace_count]

        task_scope_query = {
            "task_kind": "task_resolution",
            "source_platform": platform,
            "source_channel_id": platform_channel_id,
            "requester_global_user_id": global_user_id,
        }
        accepted_tasks = await db.accepted_tasks.find(
            task_scope_query,
            {"_id": 0},
        ).to_list(length=20)
        evidence["accepted_tasks_after_chat"] = accepted_tasks
        assert len(accepted_tasks) <= 1

        parent_surfaces = _surface_rows(
            settled_traces[:parent_trace_count]
        )
        if not accepted_tasks:
            factual_surfaces = [
                row for row in parent_surfaces
                if row["surface_role"] == "factual_answer"
            ]
            assert chat_response.messages
            assert len(factual_surfaces) == 1
            assert not [
                row for row in parent_surfaces
                if row["surface_role"] in {
                    "task_acknowledgement",
                    "task_result",
                    "task_status",
                }
            ]
            test_outcome = "passed"
            evidence["acceptance"] = {
                "route": "direct",
                "factual_surface_count": len(factual_surfaces),
                "parent_surfaces": parent_surfaces,
            }
            return

        accepted_task = accepted_tasks[0]
        accepted_task_id = str(accepted_task["accepted_task_id"])
        job_id = str(accepted_task["executor_ref"])
        continuation_ref = accepted_task.get("goal_continuation_ref")
        assert isinstance(continuation_ref, Mapping)
        parent_continuation_surfaces = [
            row for row in parent_surfaces
            if row["goal_continuation_ref"] == continuation_ref
        ]
        assert not [
            row for row in parent_continuation_surfaces
            if row["surface_role"] in {"factual_answer", "task_result"}
        ]
        assert len([
            row for row in parent_continuation_surfaces
            if row["surface_role"] == "task_acknowledgement"
        ]) <= 1

        await db.background_work_jobs.update_one(
            {"job_id": job_id},
            {"$set": {"created_at": "1970-01-01T00:00:00+00:00"}},
        )
        worker_result = await run_background_work_worker_tick(
            claim_limit=1,
            lease_seconds=120,
            max_attempts=3,
            worker_id=f"duplicate-task-delivery-{run_id}",
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
        ).sort("timestamp", 1).to_list(length=40)
        evidence["delivery_result"] = delivery_result
        evidence["adapter_calls"] = adapter.calls
        evidence["accepted_task_after_delivery"] = accepted_task_after_delivery
        evidence["background_job_after_delivery"] = (
            background_job_after_delivery
        )
        evidence["conversation_rows"] = conversation_rows
        evidence["settled_traces"] = settled_traces

        assert worker_result["processed_count"] == 1
        assert delivery_result["processed_count"] == 1
        assert delivery_result["delivered_count"] == 1
        assert accepted_task_after_delivery is not None
        assert accepted_task_after_delivery["state"] == "delivered"
        assert adapter.calls, "the accepted-task result was not visibly delivered"
        result_surfaces = [
            row for row in _surface_rows(settled_traces)
            if row["goal_continuation_ref"] == continuation_ref
            and row["surface_role"] in {"task_result", "task_status"}
        ]
        assert len(result_surfaces) == 1
        assert len([
            row for row in _surface_rows(settled_traces)
            if row["goal_continuation_ref"] == continuation_ref
            and row["surface_role"] in {
                "factual_answer",
                "task_result",
                "task_status",
            }
        ]) == 1
        test_outcome = "passed"
        evidence["acceptance"] = {
            "route": "background",
            "continuation_ref": continuation_ref,
            "acknowledgement_surface_count": len([
                row for row in parent_continuation_surfaces
                if row["surface_role"] == "task_acknowledgement"
            ]),
            "result_surface_count": len(result_surfaces),
            "result_surfaces": result_surfaces,
        }
    finally:
        evidence["settled_traces"] = settled_traces
        evidence["test_outcome"] = test_outcome
        cleanup_errors: list[str] = []
        brain_service._adapter_registry = original_registry
        brain_service._graph = original_graph
        await _run_bounded_cleanup(
            label="stop_chat_input_worker",
            cleanup=brain_service._stop_chat_input_worker(),
            errors=cleanup_errors,
        )
        await _run_bounded_cleanup(
            label="cleanup_test_rows",
            cleanup=_cleanup_test_rows(
                db=db,
                accepted_task_id=accepted_task_id,
                job_id=job_id,
                platform=platform,
                platform_channel_id=platform_channel_id,
                platform_user_id=platform_user_id,
                global_user_id=global_user_id,
            ),
            errors=cleanup_errors,
        )
        await _run_bounded_cleanup(
            label="close_db",
            cleanup=close_db(),
            errors=cleanup_errors,
        )
        if cleanup_errors:
            evidence["cleanup_errors"] = cleanup_errors
        trace_path = write_llm_trace(
            "task_resolution_duplicate_delivery_live_llm",
            "post_fix_group_history_acceptance",
            evidence,
        )
        evidence["raw_trace_path"] = str(trace_path)
        print(f"TASK_RESOLUTION_DUPLICATE_DELIVERY={trace_path}")


async def _run_bounded_cleanup(
    *,
    label: str,
    cleanup: Any,
    errors: list[str],
) -> None:
    """Run one live-test cleanup step without blocking test completion."""

    try:
        await asyncio.wait_for(cleanup, timeout=30.0)
    except Exception as exc:
        errors.append(f"{label}: {type(exc).__name__}: {exc}")


def _live_model_context() -> dict[str, object]:
    """Capture non-secret model, code, and prompt identity for raw evidence."""

    services = build_cognition_core_services()
    configs = {
        "goal_ordinary_response": services.goal_ordinary_response_config,
        "goal_active_branch": services.goal_active_branch_config,
        "action_planning": services.action_planning_config,
        "action_authorization": services.action_authorization_config,
        "resolver_authorization": services.resolver_authorization_config,
    }
    source_path = Path(action_selection.__file__).resolve()
    return {
        "model_routes": {
            stage_name: {
                "route_name": config.route_name,
                "model": config.model,
            }
            for stage_name, config in configs.items()
        },
        "code_sha256": {
            str(source_path): hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest(),
        },
        "prompt_sha256": {
            "cognition_core_v2.action_planning": hashlib.sha256(
                action_selection.ACTION_PLANNING_PROMPT.encode("utf-8")
            ).hexdigest(),
        },
    }


def _latest_llm_trace_path(prefix: str) -> Path | None:
    """Return the newest raw trace written for one live case."""

    candidates = list(_LLM_TRACE_DIRECTORY.glob(f"{prefix}*.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _surface_rows(
    traces: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Flatten settled typed surfaces while retaining their episode owner."""

    rows: list[dict[str, object]] = []
    for trace in traces:
        raw_surfaces = trace.get("surface_outputs", [])
        if not isinstance(raw_surfaces, list):
            continue
        for raw_surface in raw_surfaces:
            if not isinstance(raw_surface, Mapping):
                continue
            row = dict(raw_surface)
            row["episode_id"] = trace.get("episode_id", "")
            rows.append(row)
    return rows


async def _skip_if_live_dependencies_unavailable() -> None:
    """Skip when the real LLM endpoint or MongoDB is unavailable."""

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


async def _seed_source_messages(
    *,
    db: Any,
    platform: str,
    platform_channel_id: str,
    platform_user_id: str,
    platform_bot_id: str,
    global_user_id: str,
    run_id: str,
) -> None:
    """Seed the exported pre-turn group window before the live turn starts."""

    if not _OBSERVED_GROUP_HISTORY_PATH.exists():
        raise AssertionError(
            f"observed group history is missing: {_OBSERVED_GROUP_HISTORY_PATH}"
        )
    history_export = json.loads(
        _OBSERVED_GROUP_HISTORY_PATH.read_text(encoding="utf-8")
    )
    historical_messages = history_export.get("messages")
    if not isinstance(historical_messages, list):
        raise AssertionError("observed group history has no messages list")
    historical_request_time = datetime.fromisoformat(
        _HISTORICAL_REQUEST_TIMESTAMP
    )
    time_shift = (
        datetime.now(timezone.utc)
        - historical_request_time
        + timedelta(seconds=5)
    )

    original_bot_platform_id = "3768713357"
    original_requester_platform_id = "673225019"
    original_bot_global_user_id = (
        "00000000-0000-4000-8000-000000000001"
    )
    original_requester_global_user_id = (
        "4759394b-a4d2-4634-9d12-b6423a92a248"
    )
    platform_ids: dict[str, str] = {}
    global_ids: dict[str, str] = {
        original_bot_global_user_id: brain_service.CHARACTER_GLOBAL_USER_ID,
        original_requester_global_user_id: global_user_id,
    }

    def mapped_platform_id(original_id: str) -> str:
        """Map one source platform identity into this isolated channel."""

        if original_id == original_bot_platform_id:
            return platform_bot_id
        if original_id == original_requester_platform_id:
            return platform_user_id
        if original_id not in platform_ids:
            platform_ids[original_id] = (
                f"debug-history-{len(platform_ids)}-{run_id}"
            )
        return platform_ids[original_id]

    def mapped_global_id(original_id: str, *, row_index: int) -> str:
        """Map one source global identity into this isolated channel."""

        if original_id in global_ids:
            return global_ids[original_id]
        mapped_id = f"debug-history-global-{row_index}-{run_id}"
        global_ids[original_id] = mapped_id
        return mapped_id

    rows: list[dict[str, object]] = []
    for row_index, source_row in enumerate(historical_messages):
        timestamp = str(source_row.get("timestamp", ""))
        if not timestamp or timestamp >= _HISTORICAL_REQUEST_TIMESTAMP:
            continue
        original_platform_id = str(source_row.get("platform_user_id", ""))
        original_global_id = str(source_row.get("global_user_id", ""))
        current_platform_id = mapped_platform_id(original_platform_id)
        current_global_id = mapped_global_id(
            original_global_id or original_platform_id,
            row_index=row_index,
        )
        mentions: list[dict[str, object]] = []
        for source_mention in source_row.get("mentions", []):
            mention = dict(source_mention)
            mention_platform_id = str(
                mention.get("platform_user_id", "")
            )
            mention_global_id = str(mention.get("global_user_id", ""))
            if mention_platform_id:
                mention["platform_user_id"] = mapped_platform_id(
                    mention_platform_id
                )
            if mention_global_id:
                mention["global_user_id"] = mapped_global_id(
                    mention_global_id,
                    row_index=row_index,
                )
            mentions.append(mention)
        addressed_to = [
            mapped_global_id(str(addressed_id), row_index=row_index)
            for addressed_id in source_row.get(
                "addressed_to_global_user_ids",
                [],
            )
        ]
        body_text = str(source_row.get("body_text", ""))
        shifted_timestamp = _shift_historical_timestamp(
            timestamp,
            time_shift,
        )
        delivered_at = str(source_row.get("delivered_at", timestamp))
        shifted_delivered_at = _shift_historical_timestamp(
            delivered_at,
            time_shift,
        )
        received_at = str(source_row.get("received_at", timestamp))
        shifted_received_at = _shift_historical_timestamp(
            received_at,
            time_shift,
        )
        rows.append({
            "platform": platform,
            "platform_channel_id": platform_channel_id,
            "channel_type": "group",
            "channel_name": "Duplicate task delivery live group",
            "role": str(source_row.get("role", "user")),
            "platform_message_id": (
                f"history-{row_index}-{run_id}"
            ),
            "platform_user_id": current_platform_id,
            "global_user_id": current_global_id,
            "display_name": str(source_row.get("display_name", "")),
            "body_text": body_text,
            "raw_wire_text": str(source_row.get("raw_wire_text", body_text)),
            "content_type": str(source_row.get("content_type", "text")),
            "addressed_to_global_user_ids": list(dict.fromkeys(addressed_to)),
            "mentions": mentions,
            "broadcast": bool(source_row.get("broadcast", False)),
            "attachments": list(source_row.get("attachments", [])),
            "reply_context": dict(source_row.get("reply_context", {})),
            "delivery_tracking_id": str(
                source_row.get("delivery_tracking_id", "")
            ),
            "logical_message_index": int(
                source_row.get("logical_message_index", 0)
            ),
            "delivery_status": str(
                source_row.get("delivery_status", "delivered")
            ),
            "delivered_at": shifted_delivered_at,
            "delivery_adapter": str(
                source_row.get("delivery_adapter", "debug")
            ),
            "llm_trace_id": str(source_row.get("llm_trace_id", "")),
            "source_episode_id": str(
                source_row.get("source_episode_id", "")
            ),
            "timestamp": shifted_timestamp,
            "received_at": shifted_received_at,
            "embedding": [],
        })
    if not rows:
        raise AssertionError("observed group history produced no pre-turn rows")
    await db.conversation_history.insert_many(rows)


def _shift_historical_timestamp(
    value: str,
    offset: timedelta,
) -> str:
    """Move exported timestamps near the live test clock, preserving order."""

    timestamp = datetime.fromisoformat(value)
    shifted_timestamp = timestamp + offset
    return shifted_timestamp.isoformat()


def _chat_request(
    *,
    platform: str,
    platform_channel_id: str,
    platform_user_id: str,
    platform_bot_id: str,
    platform_message_id: str,
    display_name: str,
) -> brain_service.ChatRequest:
    """Build the addressed group request from the observed failure."""

    request = brain_service.ChatRequest(
        platform=platform,
        platform_channel_id=platform_channel_id,
        channel_type="group",
        platform_message_id=platform_message_id,
        platform_user_id=platform_user_id,
        platform_bot_id=platform_bot_id,
        display_name=display_name,
        channel_name="Duplicate task delivery live group",
        message_envelope={
            "body_text": _USER_MESSAGE,
            "raw_wire_text": _USER_MESSAGE,
            "mentions": [{
                "platform_user_id": platform_bot_id,
                "display_name": "一之濑明日奈",
                "entity_kind": "bot",
                "raw_text": "@一之濑明日奈",
            }],
            "attachments": [],
            "addressed_to_global_user_ids": [],
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
    """Remove rows and identities created by the isolated live run."""

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
        await db.user_profiles.delete_one(
            {"global_user_id": global_user_id}
        )
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
    """Capture the final dispatcher handoff without contacting a platform."""

    platform = "debug"
    display_name = "Duplicate Task Delivery E2E Adapter"

    def __init__(self, *, platform_bot_id: str) -> None:
        self.platform_bot_id = platform_bot_id
        self.calls: list[dict[str, object]] = []

    async def can_send_message(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        """Accept the unique debug channel owned by the test."""

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
        """Record visible text and return adapter-shaped delivery metadata."""

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
