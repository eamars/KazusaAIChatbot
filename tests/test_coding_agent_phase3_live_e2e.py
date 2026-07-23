"""Real LLM E2E checks for coding-agent accepted-task delivery."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.utils import load_personality
from tests.test_coding_agent_phase3_handoff_e2e import (
    CODE_TASK,
    PROJECT_SUMMARY_TASK,
    _InMemoryAcceptedCodeWorkStore,
    _persona_state_for_input,
    _public_background_job_trace,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

TRACE_ROOT = Path("test_artifacts/llm_traces/coding_agent_phase3_live_e2e")
PERSONALITY_PATH = Path(__file__).resolve().parents[1] / "personalities" / "asuna.json"


async def test_live_gate01_writing_runs_from_user_input_to_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run Gate 01 through the real LLM workflow."""

    await _run_live_case(
        monkeypatch,
        tmp_path,
        case_id="live_gate_01_writing",
        user_request=CODE_TASK,
    )


async def test_live_project_summary_runs_from_user_input_to_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run the Chinese project-summary request through the real LLM workflow."""

    await _run_live_case(
        monkeypatch,
        tmp_path,
        case_id="live_kazusa_project_summary",
        user_request=PROJECT_SUMMARY_TASK,
    )


async def _run_live_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    case_id: str,
    user_request: str,
) -> dict[str, Any]:
    """Run one live coding task from user input to final result delivery."""

    from kazusa_ai_chatbot.accepted_task import lifecycle as accepted_lifecycle
    from kazusa_ai_chatbot.background_work import delivery as background_delivery
    from kazusa_ai_chatbot.background_work import jobs as background_jobs
    from kazusa_ai_chatbot.background_work import worker as background_worker
    from kazusa_ai_chatbot.background_work.subagent import (
        coding_agent as coding_worker,
    )
    from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition
    from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2
    from kazusa_ai_chatbot import service as service_module

    store = _InMemoryAcceptedCodeWorkStore()
    _install_in_memory_persistence(
        monkeypatch,
        store=store,
    )
    monkeypatch.setattr(
        coding_worker,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "coding-workspace"),
    )
    monkeypatch.setattr(
        persona_supervisor2_cognition,
        "COGNITION_TASK_WILLINGNESS_BOUNDARY_ENABLED",
        False,
    )

    trace: dict[str, Any] = {
        "case_id": case_id,
        "run_id": _new_trace_run_id(),
        "user_request": user_request,
        "persistence_mode": "in_memory",
        "llm_mode": "real",
        "visible_dialog_events": [],
        "trace_snapshots": [],
        "test_bypasses": {
            "task_willingness_boundary_enabled": False,
        },
    }
    final_delivery: dict[str, object] = {}

    try:
        acknowledgement = await persona_supervisor2(
            _live_persona_state_for_input(
                user_input=user_request,
                platform_message_id=f"{case_id}-input-message",
            ),
        )
    except Exception as exc:
        _record_stage_exception(
            case_id,
            trace,
            stage="acknowledgement",
            exc=exc,
        )
        raise
    acknowledgement_result = _public_persona_result(acknowledgement)
    trace["acknowledgement"] = acknowledgement_result
    _record_visible_dialog_event(
        trace,
        stage="acknowledgement",
        persona_result=acknowledgement_result,
    )
    trace["accepted_task_after_ack"] = _safe_copy(store.accepted_task)
    trace["background_job_after_ack"] = _safe_background_job(store.background_job)
    _write_trace(case_id, trace, snapshot_label="acknowledgement")

    _assert_ack_scheduled_background_work(trace)

    try:
        worker_result = await background_worker.run_background_work_worker_tick(
            claim_limit=1,
            lease_seconds=60,
            max_attempts=3,
            worker_id=f"{case_id}-live-worker",
        )
    except Exception as exc:
        _record_stage_exception(case_id, trace, stage="worker", exc=exc)
        raise
    trace["worker_tick"] = worker_result
    trace["accepted_task_after_worker"] = _safe_copy(store.accepted_task)
    trace["background_job_after_worker"] = _safe_background_job(store.background_job)
    _write_trace(case_id, trace, snapshot_label="worker")

    _assert_worker_completed(trace)

    async def deliver_result_episode(episode: dict[str, Any]) -> dict[str, str]:
        result_state = _live_persona_state_for_input(
            user_input=service_module._accepted_task_result_text(episode),
            platform_message_id=f"{case_id}-result-message",
            cognitive_episode=episode,
        )
        result_state["prompt_message_context"] = (
            service_module._accepted_task_prompt_message_context(episode)
        )
        final_response = await persona_supervisor2(result_state)
        final_persona_result = _public_persona_result(final_response)
        final_delivery["episode"] = episode
        final_delivery["dialog"] = final_response["final_dialog"]
        final_delivery["persona_result"] = final_persona_result
        _record_visible_dialog_event(
            trace,
            stage="result_delivery",
            persona_result=final_persona_result,
        )
        final_dialog_value = final_response.get("final_dialog")
        final_dialog = _dialog_fragments(final_dialog_value)
        if not final_dialog:
            result = {
                "status": "failed",
                "reason": "result-ready cognition selected no visible text",
            }
            return result
        result = {
            "status": "delivered",
            "conversation_message_id": f"{case_id}-conversation-message",
        }
        return result

    try:
        delivery_result = await background_delivery.run_background_work_delivery_tick(
            deliver_result_episode_func=deliver_result_episode,
            limit=1,
        )
    except Exception as exc:
        _record_stage_exception(case_id, trace, stage="delivery", exc=exc)
        raise
    trace["delivery_tick"] = delivery_result
    trace["accepted_task_after_delivery"] = _safe_copy(store.accepted_task)
    trace["background_job_after_delivery"] = _safe_background_job(
        store.background_job,
    )
    trace["result_episode"] = final_delivery.get("episode")
    trace["final_dialog"] = final_delivery.get("dialog")
    trace["final_persona_result"] = final_delivery.get("persona_result")
    trace_path = _write_trace(case_id, trace, snapshot_label="delivery")

    _assert_delivery_completed(trace, trace_path=trace_path)
    return trace


def _live_persona_state_for_input(
    *,
    user_input: str,
    platform_message_id: str,
    cognitive_episode: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a persona state with the real profile fields needed by live LLMs."""

    state = _persona_state_for_input(
        user_input=user_input,
        platform_message_id=platform_message_id,
        cognitive_episode=cognitive_episode,
    )
    character_profile = load_personality(PERSONALITY_PATH)
    character_profile.setdefault("mood", "neutral")
    character_profile.setdefault("vibe_check", "calm")
    character_profile.setdefault("character_reflection", "")
    character_profile.setdefault("global_user_id", CHARACTER_GLOBAL_USER_ID)
    state["character_profile"] = character_profile
    state["character_name"] = character_profile.get("name", "Character")
    state["channel_topic"] = "coding-agent live E2E"
    return state


def _install_in_memory_persistence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    store: _InMemoryAcceptedCodeWorkStore,
) -> None:
    """Patch only durable storage seams for isolated live LLM execution."""

    from kazusa_ai_chatbot.accepted_task import lifecycle as accepted_lifecycle
    from kazusa_ai_chatbot.background_work import delivery as background_delivery
    from kazusa_ai_chatbot.background_work import jobs as background_jobs
    from kazusa_ai_chatbot.background_work import worker as background_worker

    monkeypatch.setattr(
        accepted_lifecycle,
        "insert_or_get_active_accepted_task",
        store.insert_or_get_active_accepted_task,
    )
    monkeypatch.setattr(
        accepted_lifecycle,
        "repository_mark_pending",
        store.mark_accepted_task_pending,
    )
    monkeypatch.setattr(
        accepted_lifecycle,
        "repository_mark_running",
        store.mark_accepted_task_running,
    )
    monkeypatch.setattr(
        accepted_lifecycle,
        "repository_mark_result_ready",
        store.mark_tool_result_ready,
    )
    monkeypatch.setattr(
        accepted_lifecycle,
        "repository_mark_failure_ready",
        store.mark_accepted_task_failure_ready,
    )
    monkeypatch.setattr(
        accepted_lifecycle,
        "repository_mark_delivery_in_progress",
        store.mark_accepted_task_delivery_in_progress,
    )
    monkeypatch.setattr(
        accepted_lifecycle,
        "repository_mark_delivered",
        store.mark_accepted_task_delivered,
    )
    monkeypatch.setattr(
        accepted_lifecycle,
        "repository_mark_delivery_failed",
        store.mark_accepted_task_delivery_failed,
    )
    monkeypatch.setattr(
        accepted_lifecycle,
        "repository_recover_delivery",
        store.recover_accepted_task_delivery,
    )
    monkeypatch.setattr(
        background_jobs,
        "insert_background_work_job",
        store.insert_background_work_job,
    )
    monkeypatch.setattr(
        background_worker,
        "claim_background_work_job",
        store.claim_background_work_job,
    )
    monkeypatch.setattr(
        background_worker,
        "complete_background_work_job",
        store.complete_background_work_job,
    )
    monkeypatch.setattr(
        background_worker,
        "fail_background_work_job",
        store.fail_background_work_job,
    )
    monkeypatch.setattr(
        background_delivery,
        "recover_stale_background_work_delivery_in_progress",
        store.recover_background_work_delivery,
    )
    monkeypatch.setattr(
        background_delivery,
        "find_deliverable_background_work_jobs",
        store.find_deliverable_background_work_jobs,
    )
    monkeypatch.setattr(
        background_delivery,
        "mark_background_work_delivery_in_progress",
        store.mark_background_work_delivery_in_progress,
    )
    monkeypatch.setattr(
        background_delivery,
        "mark_background_work_delivered",
        store.mark_background_work_delivered,
    )
    monkeypatch.setattr(
        background_delivery,
        "mark_background_work_delivery_failed",
        store.mark_background_work_delivery_failed,
    )


def _assert_ack_scheduled_background_work(trace: dict[str, Any]) -> None:
    action_results = trace["acknowledgement"]["action_results"]
    background_results = [
        row
        for row in action_results
        if row.get("action_kind") == "background_work_request"
    ]
    assert background_results, (
        "Live L2d did not select accepted background coding work; "
        f"trace={_trace_path_text(trace)}"
    )
    assert background_results[0]["status"] == "pending", (
        "Accepted background coding work was not scheduled; "
        f"trace={_trace_path_text(trace)}"
    )
    assert trace["accepted_task_after_ack"]["state"] == "pending"
    assert trace["background_job_after_ack"]["task_brief"] == trace["user_request"]


def _assert_worker_completed(trace: dict[str, Any]) -> None:
    assert trace["worker_tick"] == {
        "processed_count": 1,
        "succeeded_count": 1,
        "failed_count": 0,
    }, f"Live worker did not complete successfully; trace={_trace_path_text(trace)}"
    job = trace["background_job_after_worker"]
    assert job["worker"] == "coding_agent", (
        "Live background router did not select coding_agent; "
        f"trace={_trace_path_text(trace)}"
    )
    assert job["status"] == "completed"
    assert job["artifact_text"].strip()
    assert job["worker_metadata"].get("coding_operation") in (
        "code_reading",
        "code_writing",
    )


def _assert_delivery_completed(
    trace: dict[str, Any],
    *,
    trace_path: Path,
) -> None:
    assert trace["delivery_tick"] == {
        "processed_count": 1,
        "delivered_count": 1,
        "failed_count": 0,
        "recovered_count": 0,
    }, f"Live delivery did not complete successfully; trace={trace_path}"
    assert trace["accepted_task_after_delivery"]["state"] == "delivered"
    assert trace["background_job_after_delivery"]["status"] == "delivered"
    assert trace["result_episode"]["trigger_source"] == "tool_result"
    assert trace["final_dialog"], f"Live final dialog was empty; trace={trace_path}"


def _public_persona_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the live trace focused on workflow-visible persona output."""

    keys = (
        "final_dialog",
        "action_results",
        "action_specs",
        "logical_stance",
        "character_intent",
        "judgment_note",
        "decontexualized_input",
    )
    projected = {
        key: result.get(key)
        for key in keys
        if key in result
    }
    return projected


def _record_visible_dialog_event(
    trace: dict[str, Any],
    *,
    stage: str,
    persona_result: dict[str, Any],
) -> None:
    """Append user-visible persona dialog for one workflow stage."""

    final_dialog_value = persona_result.get("final_dialog")
    final_dialog = _dialog_fragments(final_dialog_value)
    event: dict[str, Any] = {
        "stage": stage,
        "source": "persona_supervisor2",
        "dialog_count": len(final_dialog),
        "final_dialog": final_dialog,
        "visible_text": "\n".join(final_dialog),
    }
    for key in (
        "logical_stance",
        "character_intent",
        "judgment_note",
        "decontexualized_input",
    ):
        if key in persona_result:
            event[key] = persona_result[key]

    events_value = trace.get("visible_dialog_events")
    if isinstance(events_value, list):
        events = events_value
    else:
        events = []
        trace["visible_dialog_events"] = events
    events.append(event)


def _dialog_fragments(value: object) -> list[str]:
    """Extract non-empty visible dialog fragments from a persona result."""

    if not isinstance(value, list):
        fragments: list[str] = []
        return fragments
    fragments = [
        fragment
        for fragment in value
        if isinstance(fragment, str) and fragment.strip()
    ]
    return fragments


def _safe_copy(value: dict[str, Any] | None) -> dict[str, Any]:
    """Copy nullable trace dictionaries."""

    if value is None:
        return {}
    return dict(value)


def _safe_background_job(value: dict[str, Any] | None) -> dict[str, Any]:
    """Project background job data when present."""

    if value is None:
        return {}
    return _public_background_job_trace(value)


def _write_trace(
    case_id: str,
    trace: dict[str, Any],
    *,
    snapshot_label: str,
) -> Path:
    """Write an inspectable live E2E trace snapshot without overwriting runs."""

    trace_run_id = trace["run_id"]
    if not isinstance(trace_run_id, str) or not trace_run_id:
        raise AssertionError("Live trace is missing a run_id")
    trace_dir = TRACE_ROOT / case_id / trace_run_id
    trace_dir.mkdir(parents=True, exist_ok=True)
    snapshot_paths_value = trace.get("trace_snapshots")
    if isinstance(snapshot_paths_value, list):
        snapshot_paths = snapshot_paths_value
    else:
        snapshot_paths = []
        trace["trace_snapshots"] = snapshot_paths
    snapshot_number = len(snapshot_paths) + 1
    safe_label = _safe_snapshot_label(snapshot_label)
    trace_path = trace_dir / f"{snapshot_number:02d}_{safe_label}.json"
    if trace_path.exists():
        trace_path = trace_dir / (
            f"{snapshot_number:02d}_{safe_label}_{uuid4().hex[:8]}.json"
        )
    trace["trace_path"] = str(trace_path)
    trace["trace_snapshot_label"] = snapshot_label
    snapshot_paths.append(str(trace_path))
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"trace_path={trace_path}")
    return trace_path


def _new_trace_run_id() -> str:
    """Create a unique run id for one live LLM workflow execution."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{timestamp}-{uuid4().hex[:8]}"
    return run_id


def _safe_snapshot_label(snapshot_label: str) -> str:
    """Return a filesystem-safe snapshot label."""

    safe_chars = [
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in snapshot_label
    ]
    safe_label = "".join(safe_chars).strip("_")
    if not safe_label:
        safe_label = "snapshot"
    return safe_label


def _record_stage_exception(
    case_id: str,
    trace: dict[str, Any],
    *,
    stage: str,
    exc: Exception,
) -> None:
    """Persist the live workflow state before re-raising a stage failure."""

    trace["exception"] = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
    }
    _write_trace(case_id, trace, snapshot_label=f"{stage}_exception")


def _trace_path_text(trace: dict[str, Any]) -> str:
    """Return the trace path if it has already been written."""

    trace_path = trace.get("trace_path")
    if isinstance(trace_path, str) and trace_path:
        return trace_path
    return str(TRACE_ROOT)
