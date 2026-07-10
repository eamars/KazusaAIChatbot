"""E2E contract tests for coding-agent accepted-task handoff."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


CODE_TASK = (
    "Create a single Python command-line script that reads a plain text "
    "application log file and counts entries by severity. Each valid line "
    "starts with one of DEBUG, INFO, WARNING, ERROR, or CRITICAL followed by "
    "a space and the message. The script should print a terminal summary "
    "with one count per severity, report how many malformed lines were "
    "skipped, handle a missing input file clearly, and use only the Python "
    "standard library."
)
CODE_TASK_RESULT = (
    "Proposed a standard-library log parser script with severity counts, "
    "malformed-line reporting, and missing-file handling."
)
PROJECT_SUMMARY_TASK = (
    '请帮我总结一下 [eamars/KazusaAIChatbot]'
    '(https://github.com/eamars/KazusaAIChatbot) 的项目设计和项目亮点'
)
PROJECT_SUMMARY_RESULT = (
    'KazusaAIChatbot 的设计核心是平台适配层、brain service、队列/RAG、'
    'cognition、dialog、持久化和后台反思的分层协作；亮点是角色判断、'
    '可审计的延迟任务和代码代理工作流。'
)


def test_l2d_capability_projection_exposes_accepted_code_task_without_internals() -> None:
    """Capability projection should expose accepted delayed code work safely."""

    from kazusa_ai_chatbot.action_spec.registry import (
        build_initial_action_capabilities,
    )
    from kazusa_ai_chatbot.cognition_chain_core.action_selection import (
        build_action_selection_payload,
    )

    payload = build_action_selection_payload(
        _minimal_action_selection_state(),
        build_initial_action_capabilities(),
    )

    action_affordances = payload["capabilities"]["action_affordances"]
    accepted_rows = [
        row
        for row in action_affordances
        if row["capability"] == "accepted_task_request"
    ]
    assert len(accepted_rows) == 1
    summary_text = " ".join(accepted_rows[0]["semantic_input_summary"])

    assert "code" in summary_text.lower()
    assert "accepted_task_request" in {
        row["capability"] for row in action_affordances
    }
    assert "background_work_request" not in {
        row["capability"] for row in action_affordances
    }
    for hidden_term in (
        "worker",
        "queue",
        "job",
        "lease",
        "workspace_root",
        "tool_args",
    ):
        assert hidden_term not in summary_text


@pytest.mark.asyncio
async def test_accepted_code_task_materializes_queue_and_l3_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted code work should persist before L3 can promise completion."""

    from kazusa_ai_chatbot.action_spec.handlers import background_work
    from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_surface

    captured_queue_requests: list[dict[str, Any]] = []

    async def fake_create_or_return_active_accepted_task(request: dict[str, Any]):
        assert request["accepted_task_seed"] == CODE_TASK
        assert request["accepted_task_detail"] == CODE_TASK
        assert request["accepted_task_summary"] == CODE_TASK
        return {
            "status": "created",
            "task": {
                "accepted_task_id": "accepted-task-001",
                "task_identity_key": "accepted-task-key-001",
                "accepted_task_summary": CODE_TASK,
            },
        }

    async def fake_mark_accepted_task_pending(
        *,
        accepted_task_id: str,
        executor_ref: str,
        updated_at: str,
    ):
        assert accepted_task_id == "accepted-task-001"
        assert executor_ref == "background-work-job-001"
        assert updated_at == "2026-07-03T01:00:00+00:00"
        return {
            "accepted_task_id": accepted_task_id,
            "task_identity_key": "accepted-task-key-001",
            "accepted_task_summary": CODE_TASK,
        }

    async def fake_enqueue_background_work(request: dict[str, Any]):
        captured_queue_requests.append(dict(request))
        return {
            "status": "pending",
            "job_id": "background-work-job-001",
            "task_brief": request["task_brief"],
        }

    monkeypatch.setattr(
        background_work,
        "create_or_return_active_accepted_task",
        fake_create_or_return_active_accepted_task,
    )
    monkeypatch.setattr(
        background_work,
        "mark_accepted_task_pending",
        fake_mark_accepted_task_pending,
    )

    queue_result = await background_work.enqueue_background_work_action(
        _background_work_action(CODE_TASK),
        storage_timestamp_utc="2026-07-03T01:00:00+00:00",
        action_attempt_id="action-attempt-001",
        enqueue_background_work_func=fake_enqueue_background_work,
    )

    assert queue_result["status"] == "pending"
    assert queue_result["accepted_task_state"] == "scheduled"
    assert queue_result["accepted_task_summary"] == CODE_TASK
    assert queue_result["acknowledgement_constraint"] == "promise_allowed"
    assert captured_queue_requests[0]["task_brief"] == CODE_TASK
    assert captured_queue_requests[0]["accepted_task_id"] == "accepted-task-001"

    l3_intent = l3_surface._selected_text_surface_intent(
        _l3_state_with_action_result(queue_result),
    )

    assert "accepted_task_request" in l3_intent
    assert CODE_TASK in l3_intent
    assert "promise_allowed" in l3_intent
    assert "background_work_request" not in l3_intent
    assert "background-work-job-001" not in l3_intent


def test_completed_coding_task_delivers_as_accepted_task_result_ready() -> None:
    """Completed accepted-task-backed coding work should use final delivery path."""

    from kazusa_ai_chatbot.background_work.result_source import (
        build_result_ready_episode_from_job,
    )

    episode = build_result_ready_episode_from_job(_completed_coding_job())
    percept = episode["percepts"][0]

    assert episode["trigger_source"] == "accepted_task_result_ready"
    assert episode["input_sources"] == ["accepted_task_result"]
    assert percept["input_source"] == "accepted_task_result"
    assert percept["content"] == CODE_TASK_RESULT
    assert percept["metadata"]["accepted_task_id"] == "accepted-task-001"
    assert percept["metadata"]["accepted_task_summary"] == CODE_TASK
    assert "coding_agent" not in repr(percept["metadata"])
    assert "workspace_root" not in repr(episode)
    assert "local_root" not in repr(episode)
    assert "cache_key" not in repr(episode)


@pytest.mark.asyncio
async def test_gate01_writing_task_runs_from_user_input_to_final_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Run the gate-01 coding request through the full handoff path."""

    trace = await _run_full_coding_handoff_case(
        monkeypatch,
        tmp_path,
        case_id="gate_01_writing",
        user_request=CODE_TASK,
        coding_result_text=CODE_TASK_RESULT,
        coding_operation="code_writing",
    )

    assert trace["acknowledgement_dialog"] == [
        "I accepted the coding task and will report back after it finishes."
    ]
    assert trace["worker_tick"] == {
        "processed_count": 1,
        "succeeded_count": 1,
        "failed_count": 0,
    }
    assert trace["delivery_tick"] == {
        "processed_count": 1,
        "delivered_count": 1,
        "failed_count": 0,
        "recovered_count": 0,
    }
    assert trace["coding_agent_requests"] == [
        {
            "question": CODE_TASK,
            "source_summary": "The user asked for delayed coding work.",
            "workspace_root": str(tmp_path / "coding-workspace"),
            "max_answer_chars": 3000,
            "max_artifact_chars": 24000,
        }
    ]
    assert trace["background_job"]["worker"] == "coding_agent"
    assert trace["background_job"]["worker_metadata"]["coding_operation"] == (
        "code_writing"
    )
    assert trace["accepted_task"]["state"] == "delivered"
    assert trace["result_episode"]["trigger_source"] == (
        "accepted_task_result_ready"
    )
    assert trace["result_episode"]["percepts"][0]["content"] == CODE_TASK_RESULT
    assert trace["final_dialog"] == [CODE_TASK_RESULT]
    serialized_delivery = repr({
        "result_episode": trace["result_episode"],
        "final_dialog": trace["final_dialog"],
        "background_job": trace["background_job"],
    })
    assert "workspace_root" not in serialized_delivery
    assert "local_root" not in serialized_delivery
    assert "cache_key" not in serialized_delivery


@pytest.mark.asyncio
async def test_project_summary_question_runs_from_user_input_to_final_delivery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Run the Chinese repository-summary request through the full path."""

    trace = await _run_full_coding_handoff_case(
        monkeypatch,
        tmp_path,
        case_id="kazusa_project_summary",
        user_request=PROJECT_SUMMARY_TASK,
        coding_result_text=PROJECT_SUMMARY_RESULT,
        coding_operation="code_reading",
    )

    assert trace["acknowledgement_dialog"] == [
        "I accepted the coding task and will report back after it finishes."
    ]
    assert trace["worker_tick"]["succeeded_count"] == 1
    assert trace["delivery_tick"]["delivered_count"] == 1
    assert trace["coding_agent_requests"] == [
        {
            "question": PROJECT_SUMMARY_TASK,
            "source_summary": "The user asked for delayed coding work.",
            "workspace_root": str(tmp_path / "coding-workspace"),
            "max_answer_chars": 3000,
            "max_artifact_chars": 24000,
        }
    ]
    assert trace["background_job"]["worker_metadata"]["coding_operation"] == (
        "code_reading"
    )
    assert trace["accepted_task"]["state"] == "delivered"
    assert trace["result_episode"]["percepts"][0]["content"] == (
        PROJECT_SUMMARY_RESULT
    )
    assert trace["final_dialog"] == [PROJECT_SUMMARY_RESULT]
    serialized_delivery = repr({
        "result_episode": trace["result_episode"],
        "final_dialog": trace["final_dialog"],
        "background_job": trace["background_job"],
    })
    assert "workspace_root" not in serialized_delivery
    assert "local_root" not in serialized_delivery
    assert "cache_key" not in serialized_delivery


async def _run_full_coding_handoff_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    case_id: str,
    user_request: str,
    coding_result_text: str,
    coding_operation: str,
) -> dict[str, Any]:
    """Run user input through acceptance, worker execution, and delivery."""

    from kazusa_ai_chatbot.accepted_task import lifecycle as accepted_lifecycle
    from kazusa_ai_chatbot.background_work import delivery as background_delivery
    from kazusa_ai_chatbot.background_work import jobs as background_jobs
    from kazusa_ai_chatbot.background_work import router as background_router
    from kazusa_ai_chatbot.background_work import worker as background_worker
    from kazusa_ai_chatbot.background_work.subagent import (
        coding_agent as coding_worker,
    )
    from kazusa_ai_chatbot.nodes import persona_supervisor2 as persona_module
    from kazusa_ai_chatbot.nodes.persona_supervisor2 import persona_supervisor2

    store = _InMemoryAcceptedCodeWorkStore(
        coding_result_text=coding_result_text,
        coding_operation=coding_operation,
    )
    final_delivery: dict[str, object] = {}

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
        store.mark_accepted_task_result_ready,
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
    monkeypatch.setattr(
        background_router,
        "_background_work_router_llm",
        _StaticJsonLLM({
            "action": "execute",
            "worker": "coding_agent",
            "reason": "The task asks for delayed coding work.",
        }),
    )
    monkeypatch.setattr(
        coding_worker,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "coding-workspace"),
    )
    monkeypatch.setattr(
        coding_worker,
        "handle_background_coding_task",
        store.handle_background_coding_task,
    )
    monkeypatch.setattr(
        persona_module,
        "call_msg_decontexualizer",
        _fake_decontextualizer,
    )
    monkeypatch.setattr(
        persona_module,
        "load_matching_pending_resume_into_state",
        _fake_pending_resume_loader,
    )
    monkeypatch.setattr(
        persona_module,
        "call_cognition_resolver_loop",
        _fake_cognition_resolver_loop,
    )
    monkeypatch.setattr(
        persona_module,
        "call_l3_text_surface_handler",
        _fake_l3_text_surface_handler,
    )
    monkeypatch.setattr(
        persona_module,
        "dialog_agent",
        _fake_dialog_agent,
    )

    acknowledgement = await persona_supervisor2(
        _persona_state_for_input(
            user_input=user_request,
            platform_message_id=f"{case_id}-input-message",
        ),
    )

    assert acknowledgement["final_dialog"] == [
        "I accepted the coding task and will report back after it finishes."
    ]
    assert store.accepted_task is not None
    assert store.accepted_task["state"] == "pending"
    assert store.background_job is not None
    assert store.background_job["task_brief"] == user_request
    action_results = {
        result["action_kind"]: result
        for result in acknowledgement["action_results"]
    }
    assert action_results["speak"]["status"] == "executed"
    assert action_results["background_work_request"]["accepted_task_state"] == (
        "scheduled"
    )

    worker_result = await background_worker.run_background_work_worker_tick(
        claim_limit=1,
        lease_seconds=60,
        max_attempts=3,
        worker_id=f"{case_id}-phase3-e2e-worker",
    )

    assert worker_result == {
        "processed_count": 1,
        "succeeded_count": 1,
        "failed_count": 0,
    }
    assert store.background_job is not None
    assert store.background_job["worker"] == "coding_agent"
    assert store.background_job["status"] == "completed"
    assert store.background_job["delivery_state"] == "ready"
    assert store.accepted_task is not None
    assert store.accepted_task["state"] == "result_ready"

    async def deliver_result_episode(episode: dict[str, Any]) -> dict[str, str]:
        final_response = await persona_supervisor2(
            _persona_state_for_input(
                user_input=episode["percepts"][0]["content"],
                platform_message_id=f"{case_id}-result-message",
                cognitive_episode=episode,
            ),
        )
        final_delivery["episode"] = episode
        final_delivery["dialog"] = final_response["final_dialog"]
        result = {
            "status": "delivered",
            "conversation_message_id": "conversation-message-001",
        }
        return result

    delivery_result = await background_delivery.run_background_work_delivery_tick(
        deliver_result_episode_func=deliver_result_episode,
        limit=1,
    )

    assert delivery_result == {
        "processed_count": 1,
        "delivered_count": 1,
        "failed_count": 0,
        "recovered_count": 0,
    }
    result_episode = final_delivery["episode"]
    assert result_episode["trigger_source"] == "accepted_task_result_ready"
    assert result_episode["percepts"][0]["content"] == coding_result_text
    assert final_delivery["dialog"] == [coding_result_text]
    assert store.background_job is not None
    assert store.background_job["status"] == "delivered"
    assert store.accepted_task is not None
    assert store.accepted_task["state"] == "delivered"
    serialized_delivery = repr(final_delivery)
    assert "workspace_root" not in serialized_delivery
    assert "local_root" not in serialized_delivery
    assert "cache_key" not in serialized_delivery
    trace = {
        "case_id": case_id,
        "user_request": user_request,
        "acknowledgement_dialog": acknowledgement["final_dialog"],
        "action_results": acknowledgement["action_results"],
        "worker_tick": worker_result,
        "delivery_tick": delivery_result,
        "coding_agent_requests": store.coding_agent_requests,
        "accepted_task": dict(store.accepted_task),
        "background_job": _public_background_job_trace(store.background_job),
        "result_episode": result_episode,
        "final_dialog": final_delivery["dialog"],
    }
    _write_e2e_trace_artifact(case_id, trace)
    return trace


def _public_background_job_trace(job: dict[str, Any]) -> dict[str, Any]:
    """Project the durable job state used by the E2E artifact."""

    public_keys = (
        "job_id",
        "accepted_task_id",
        "task_brief",
        "source_context",
        "status",
        "worker",
        "router_action",
        "router_reason",
        "delivery_state",
        "artifact_text",
        "result_summary",
        "worker_metadata",
    )
    trace = {
        key: job[key]
        for key in public_keys
        if key in job
    }
    return trace


def _write_e2e_trace_artifact(case_id: str, trace: dict[str, Any]) -> None:
    """Write a human-readable trace for the accepted-task E2E handoff case."""

    artifact_dir = Path("test_artifacts/coding_agent_phase3_handoff_e2e")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{case_id}.json"
    artifact_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _minimal_action_selection_state() -> dict[str, object]:
    """Build the smallest state needed for action-selection payload projection."""

    state: dict[str, object] = {
        "cognitive_episode": {
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "live_response",
        },
        "channel_type": "private",
        "decontexualized_input": CODE_TASK,
        "media_summary": "",
        "logical_stance": "ACCEPT",
        "character_intent": "HELP",
        "judgment_note": "The user asked for bounded delayed coding work.",
        "internal_monologue": "I should accept this as later work.",
        "emotional_appraisal": "calm",
        "interaction_subtext": "direct coding request",
        "boundary_core_assessment": {},
        "social_distance": "friendly",
        "emotional_intensity": "low",
        "vibe_check": "focused",
        "relational_dynamic": "direct request",
        "rag_result": {},
        "conversation_progress": {},
        "resolver_context": "",
        "background_work_output_char_limit": 4000,
    }
    return state


def _background_work_action(task_brief: str) -> dict[str, Any]:
    """Build one validated internal delayed-work action spec."""

    action = {
        "schema_version": "action_spec.v1",
        "kind": "background_work_request",
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "cognitive_episode",
                "ref_id": "episode-001",
                "owner": "cognition",
                "relationship": "basis",
                "evidence_refs": [],
            }
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "current_user",
            "target_id": None,
            "owner": "background_work",
            "scope": {
                "source_platform": "debug",
                "source_channel_id": "debug-channel-001",
                "source_channel_type": "private",
                "source_message_id": "debug-message-001",
                "source_platform_bot_id": "debug-bot-001",
                "source_character_name": "Kazusa",
                "source_trigger_source": "user_message",
                "requester_global_user_id": "global-user-001",
                "requester_platform_user_id": "platform-user-001",
                "requester_display_name": "Test User",
            },
        },
        "params": {
            "task_brief": task_brief,
            "requested_delivery": "send_result_when_done",
            "max_output_chars": 2000,
        },
        "urgency": "background",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "The user asked for bounded delayed coding work.",
    }
    return action


def _l3_state_with_action_result(queue_result: dict[str, Any]) -> dict[str, Any]:
    """Build the L3 surface state for accepted-task acknowledgement."""

    state = {
        "action_specs": [
            {
                "schema_version": "action_spec.v1",
                "kind": "speak",
                "cognition_mode": "deliberative",
                "source_refs": [],
                "target": {
                    "schema_version": "action_target.v1",
                    "target_kind": "current_channel",
                    "target_id": None,
                    "owner": "l3_text",
                    "scope": {"surface": "text"},
                },
                "params": {
                    "delivery_mode": "visible_reply",
                    "execute_at": None,
                    "surface_requirements": {
                        "intent": "acknowledge the accepted delayed code work",
                    },
                },
                "urgency": "now",
                "visibility": "user_visible",
                "deadline": None,
                "continuation": {
                    "schema_version": "action_continuation.v1",
                    "mode": "none",
                    "episode_type": None,
                    "max_depth": 0,
                    "include_result_as": None,
                },
                "reason": "The durable accepted task was scheduled.",
            }
        ],
        "pre_surface_action_results": [
            {
                "action_attempt_id": "action-attempt-001",
                "action_kind": "background_work_request",
                "status": queue_result["status"],
                "accepted_task_state": queue_result["accepted_task_state"],
                "accepted_task_summary": queue_result["accepted_task_summary"],
                "wait_guidance": queue_result["wait_guidance"],
                "acknowledgement_constraint": queue_result[
                    "acknowledgement_constraint"
                ],
            }
        ],
    }
    return state


def _completed_coding_job() -> dict[str, Any]:
    """Build one completed accepted-task-backed coding-agent job."""

    job = {
        "schema_version": "background_work_job.v1",
        "job_id": "background-work-job-001",
        "accepted_task_id": "accepted-task-001",
        "task_brief": CODE_TASK,
        "source_context": "The user asked for delayed coding work.",
        "source_platform": "debug",
        "source_channel_id": "debug-channel-001",
        "source_channel_type": "private",
        "source_message_id": "debug-message-001",
        "source_platform_bot_id": "debug-bot-001",
        "source_character_name": "Kazusa",
        "requester_global_user_id": "global-user-001",
        "requester_platform_user_id": "platform-user-001",
        "requester_display_name": "Test User",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 2000,
        "status": "completed",
        "worker": "coding_agent",
        "artifact_text": CODE_TASK_RESULT,
        "failure_summary": "",
        "result_summary": "Completed accepted coding work.",
        "worker_metadata": {
            "schema_version": "coding_agent_worker_metadata.v1",
            "coding_operation": "code_writing",
        },
        "created_at": "2026-07-03T01:00:00+00:00",
        "updated_at": "2026-07-03T01:02:00+00:00",
        "completed_at": "2026-07-03T01:02:00+00:00",
    }
    return job


class _StaticJsonLLM:
    """Return one fixed JSON payload from an LLM-shaped test double."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    async def ainvoke(
        self,
        _messages: list[object],
        *,
        config,
    ) -> SimpleNamespace:
        del config
        content = json.dumps(self._payload, ensure_ascii=False)
        response = SimpleNamespace(content=content)
        return response


class _InMemoryAcceptedCodeWorkStore:
    """In-memory persistence seams for one accepted coding workflow."""

    def __init__(
        self,
        *,
        coding_result_text: str = CODE_TASK_RESULT,
        coding_operation: str = "code_writing",
    ) -> None:
        self.accepted_task: dict[str, Any] | None = None
        self.background_job: dict[str, Any] | None = None
        self.coding_agent_requests: list[dict[str, Any]] = []
        self._coding_result_text = coding_result_text
        self._coding_operation = coding_operation

    async def insert_or_get_active_accepted_task(
        self,
        task: dict[str, Any],
        *,
        source_message_id: str,
        observed_at: str,
    ) -> dict[str, object]:
        del source_message_id, observed_at
        self.accepted_task = dict(task)
        result = {
            "status": "created",
            "task": dict(task),
        }
        return result

    async def mark_accepted_task_pending(
        self,
        *,
        accepted_task_id: str,
        executor_ref: str,
        updated_at: str,
    ) -> dict[str, Any] | None:
        task = self._accepted_task(accepted_task_id)
        task.update({
            "state": "pending",
            "executor_kind": "background_work",
            "executor_ref": executor_ref,
            "updated_at": updated_at,
        })
        return dict(task)

    async def mark_accepted_task_running(
        self,
        *,
        accepted_task_id: str,
        started_at: str,
    ) -> dict[str, Any] | None:
        task = self._accepted_task(accepted_task_id)
        task.update({
            "state": "running",
            "started_at": started_at,
            "updated_at": started_at,
        })
        return dict(task)

    async def mark_accepted_task_result_ready(
        self,
        *,
        accepted_task_id: str,
        artifact_text: str,
        result_summary: str,
        completed_at: str,
        coding_run_context: dict[str, object] | None = None,
    ) -> dict[str, Any] | None:
        task = self._accepted_task(accepted_task_id)
        task.update({
            "state": "result_ready",
            "result_kind": "artifact",
            "artifact_text": artifact_text,
            "result_summary": result_summary,
            "completed_at": completed_at,
            "updated_at": completed_at,
        })
        if coding_run_context is not None:
            task["coding_run_context"] = dict(coding_run_context)
        return dict(task)

    async def mark_accepted_task_failure_ready(
        self,
        *,
        accepted_task_id: str,
        failure_summary: str,
        completed_at: str,
    ) -> dict[str, Any] | None:
        task = self._accepted_task(accepted_task_id)
        task.update({
            "state": "failure_ready",
            "result_kind": "failure",
            "failure_summary": failure_summary,
            "completed_at": completed_at,
            "updated_at": completed_at,
        })
        return dict(task)

    async def mark_accepted_task_delivery_in_progress(
        self,
        *,
        accepted_task_id: str,
        delivery_tracking_id: str,
        updated_at: str,
    ) -> dict[str, Any] | None:
        task = self._accepted_task(accepted_task_id)
        task.update({
            "state": "delivery_in_progress",
            "delivery_tracking_id": delivery_tracking_id,
            "updated_at": updated_at,
        })
        return dict(task)

    async def mark_accepted_task_delivered(
        self,
        *,
        accepted_task_id: str,
        delivered_conversation_message_id: str,
        delivered_at: str,
    ) -> dict[str, Any] | None:
        task = self._accepted_task(accepted_task_id)
        task.update({
            "state": "delivered",
            "delivered_conversation_message_id": (
                delivered_conversation_message_id
            ),
            "delivered_at": delivered_at,
            "updated_at": delivered_at,
        })
        task.pop("active_identity_key", None)
        return dict(task)

    async def mark_accepted_task_delivery_failed(
        self,
        *,
        accepted_task_id: str,
        failure_summary: str,
        failed_at: str,
    ) -> dict[str, Any] | None:
        task = self._accepted_task(accepted_task_id)
        task.update({
            "state": "delivery_retryable",
            "delivery_failure_summary": failure_summary,
            "updated_at": failed_at,
        })
        return dict(task)

    async def recover_accepted_task_delivery(
        self,
        *,
        stale_before_utc: str,
        recovered_at: str,
    ) -> int:
        del stale_before_utc, recovered_at
        return 0

    async def insert_background_work_job(
        self,
        job: dict[str, Any],
    ) -> dict[str, Any]:
        self.background_job = dict(job)
        return dict(job)

    async def claim_background_work_job(
        self,
        *,
        lease_owner: str,
        lease_seconds: int,
        now_utc: str,
        max_attempts: int,
    ) -> dict[str, Any] | None:
        del lease_seconds, max_attempts
        job = self._background_job_or_none()
        if job is None or job["status"] != "queued":
            return None
        job["status"] = "in_progress"
        job["lease_owner"] = lease_owner
        job["lease_expires_at"] = now_utc
        job["updated_at"] = now_utc
        job["attempt_count"] = int(job["attempt_count"]) + 1
        return dict(job)

    async def complete_background_work_job(
        self,
        *,
        job_id: str,
        lease_owner: str,
        router_action: str,
        worker: str,
        routed_task: str,
        router_reason: str,
        artifact_text: str,
        result_summary: str,
        worker_metadata: dict[str, object],
        completed_at: str,
        skip_result_delivery: bool = False,
    ) -> dict[str, Any] | None:
        job = self._leased_background_job(job_id, lease_owner)
        delivery_state = "delivered" if skip_result_delivery else "ready"
        delivered_at = completed_at if skip_result_delivery else ""
        job.update({
            "status": "completed",
            "delivery_state": delivery_state,
            "router_action": router_action,
            "worker": worker,
            "routed_task": routed_task,
            "router_reason": router_reason,
            "artifact_text": artifact_text,
            "artifact_char_count": len(artifact_text),
            "failure_summary": "",
            "result_summary": result_summary,
            "worker_metadata": dict(worker_metadata),
            "completed_at": completed_at,
            "delivered_at": delivered_at,
            "updated_at": completed_at,
            "lease_owner": None,
            "lease_expires_at": None,
        })
        return dict(job)

    async def fail_background_work_job(
        self,
        *,
        job_id: str,
        lease_owner: str,
        failure_summary: str,
        failed_at: str,
        router_action: str = "",
        worker: str = "",
        routed_task: str = "",
        router_reason: str = "",
        result_summary: str = "",
        worker_metadata: dict[str, object] | None = None,
    ) -> dict[str, Any] | None:
        job = self._leased_background_job(job_id, lease_owner)
        job.update({
            "status": "failed",
            "delivery_state": "ready",
            "router_action": router_action,
            "worker": worker,
            "routed_task": routed_task,
            "router_reason": router_reason,
            "failure_summary": failure_summary,
            "result_summary": result_summary,
            "worker_metadata": dict(worker_metadata or {}),
            "updated_at": failed_at,
            "lease_owner": None,
            "lease_expires_at": None,
        })
        return dict(job)

    async def recover_background_work_delivery(
        self,
        *,
        stale_before_utc: str,
        recovered_at: str,
    ) -> int:
        del stale_before_utc, recovered_at
        return 0

    async def find_deliverable_background_work_jobs(
        self,
        *,
        limit: int,
    ) -> list[dict[str, Any]]:
        job = self._background_job_or_none()
        if job is None:
            return []
        is_deliverable = (
            job["status"] in ("completed", "failed", "delivery_failed")
            and job["delivery_state"] in ("ready", "failed")
        )
        if not is_deliverable:
            return []
        rows = [dict(job)][:limit]
        return rows

    async def mark_background_work_delivery_in_progress(
        self,
        *,
        job_id: str,
        delivery_tracking_id: str,
        started_at: str,
    ) -> dict[str, Any] | None:
        job = self._background_job(job_id)
        job.update({
            "status": "delivery_in_progress",
            "delivery_state": "in_progress",
            "delivery_tracking_id": delivery_tracking_id,
            "delivery_attempt_count": int(job["delivery_attempt_count"]) + 1,
            "updated_at": started_at,
        })
        return dict(job)

    async def mark_background_work_delivered(
        self,
        *,
        job_id: str,
        delivered_conversation_message_id: str,
        delivered_at: str,
    ) -> dict[str, Any] | None:
        job = self._background_job(job_id)
        job.update({
            "status": "delivered",
            "delivery_state": "delivered",
            "delivered_conversation_message_id": (
                delivered_conversation_message_id
            ),
            "delivered_at": delivered_at,
            "updated_at": delivered_at,
        })
        return dict(job)

    async def mark_background_work_delivery_failed(
        self,
        *,
        job_id: str,
        failure_summary: str,
        failed_at: str,
    ) -> dict[str, Any] | None:
        job = self._background_job(job_id)
        job.update({
            "status": "delivery_failed",
            "delivery_state": "failed",
            "delivery_failure_summary": failure_summary,
            "updated_at": failed_at,
        })
        return dict(job)

    async def handle_background_coding_task(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.coding_agent_requests.append(dict(request))
        response = {
            "status": "succeeded",
            "operation": self._coding_operation,
            "answer_text": self._coding_result_text,
            "repository": {
                "provider": "github",
                "owner": "eamars",
                "repo": "KazusaAIChatbot",
                "source_url": "https://github.com/eamars/KazusaAIChatbot",
                "requested_ref": "main",
                "resolved_ref": "main",
                "current_commit": "b" * 40,
                "default_branch": "main",
                "storage_kind": "managed_clone",
                "managed_checkout": True,
                "dirty_state": "clean",
            },
            "source_scope": {
                "kind": "repository",
                "repo_relative_path": None,
                "source_url": "https://github.com/eamars/KazusaAIChatbot",
                "requested_ref": "main",
                "interpretation": "entire repository",
            },
            "evidence": [
                {
                    "path": "src/kazusa_ai_chatbot/coding_agent/README.md",
                    "line_start": 10,
                    "line_end": 20,
                    "symbol_or_topic": "coding agent handoff",
                    "excerpt": "raw source excerpt",
                    "reason": "Shows coding-agent background integration.",
                }
            ],
            "patch_artifacts": [
                {
                    "artifact_id": "coding_result",
                    "files": ["src/log_parser.py"],
                    "summary": "Proposed coding artifact.",
                }
            ] if self._coding_operation == "code_writing" else [],
            "created_files": [
                {
                    "path": "src/log_parser.py",
                    "role": "source",
                }
            ] if self._coding_operation == "code_writing" else [],
            "changed_files": [],
            "validation": None,
            "limitations": [],
            "trace_summary": [f"{self._coding_operation}:succeeded"],
        }
        return response

    def _accepted_task(self, accepted_task_id: str) -> dict[str, Any]:
        if self.accepted_task is None:
            raise AssertionError("accepted task was not created")
        if self.accepted_task["accepted_task_id"] != accepted_task_id:
            raise AssertionError("unexpected accepted task id")
        return self.accepted_task

    def _background_job_or_none(self) -> dict[str, Any] | None:
        return self.background_job

    def _background_job(self, job_id: str) -> dict[str, Any]:
        if self.background_job is None:
            raise AssertionError("background job was not created")
        if self.background_job["job_id"] != job_id:
            raise AssertionError("unexpected background job id")
        return self.background_job

    def _leased_background_job(
        self,
        job_id: str,
        lease_owner: str,
    ) -> dict[str, Any]:
        job = self._background_job(job_id)
        if job["lease_owner"] != lease_owner:
            raise AssertionError("background job lease owner mismatch")
        return job


async def _fake_decontextualizer(state: dict[str, Any]) -> dict[str, str]:
    """Return the prompt-safe task text for the current episode."""

    episode = state.get("cognitive_episode")
    if isinstance(episode, dict):
        trigger_source = episode.get("trigger_source")
        if trigger_source == "accepted_task_result_ready":
            result = {
                "decontexualized_input": "Accepted task result is ready."
            }
            return result
    result = {"decontexualized_input": str(state.get("user_input", ""))}
    return result


async def _fake_pending_resume_loader(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Keep the E2E workflow independent from resolver persistence."""

    result = dict(state)
    return result


async def _fake_cognition_resolver_loop(
    state: dict[str, Any],
    *_,
    **__,
) -> dict[str, Any]:
    """Return materialized actions for acknowledgement or final delivery."""

    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_actions import (
        materialize_semantic_action_requests,
    )

    episode = state.get("cognitive_episode")
    trigger_source = ""
    if isinstance(episode, dict):
        trigger_source = str(episode.get("trigger_source", ""))
    if trigger_source == "accepted_task_result_ready":
        semantic_requests = [
            {
                "capability": "speak",
                "decision": "visible_reply",
                "detail": "deliver the completed coding result",
                "reason": "The accepted coding task is complete.",
            }
        ]
    else:
        semantic_requests = [
            {
                "capability": "speak",
                "decision": "visible_reply",
                "detail": "acknowledge accepted delayed coding work",
                "reason": "The task needs delayed coding work.",
            },
            {
                "capability": "accepted_task_request",
                "decision": "background_task",
                "detail": "",
                "reason": "The user asked for delayed coding work.",
            },
        ]
    action_specs = materialize_semantic_action_requests(
        semantic_requests,
        state,
    )
    result = _cognition_update_for_actions(state, action_specs)
    return result


async def _fake_l3_text_surface_handler(_state: dict[str, Any]) -> dict[str, Any]:
    """Return a minimal selected-text-surface directive payload."""

    result = {
        "action_directives": {
            "contextual_directives": {},
            "linguistic_directives": {},
            "visual_directives": {},
        }
    }
    return result


async def _fake_dialog_agent(state: dict[str, Any]) -> dict[str, Any]:
    """Return visible text for either acknowledgement or final delivery."""

    episode = state.get("cognitive_episode")
    trigger_source = ""
    if isinstance(episode, dict):
        trigger_source = str(episode.get("trigger_source", ""))
    if trigger_source == "accepted_task_result_ready":
        final_dialog = [str(state.get("user_input", ""))]
    else:
        final_dialog = [
            (
                "I accepted the coding task and will report back after "
                "it finishes."
            )
        ]
    result = {
        "final_dialog": final_dialog,
        "target_addressed_user_ids": ["global-user-001"],
        "target_broadcast": False,
    }
    return result


def _persona_state_for_input(
    *,
    user_input: str,
    platform_message_id: str,
    cognitive_episode: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one persona graph input state from a user-visible event."""

    storage_timestamp_utc = "2026-07-03T01:00:00+00:00"
    local_time_context = {
        "current_local_datetime": "2026-07-03 13:00",
        "current_local_weekday": "Friday",
    }
    if cognitive_episode is None:
        cognitive_episode = _user_message_episode(
            user_input=user_input,
            platform_message_id=platform_message_id,
            storage_timestamp_utc=storage_timestamp_utc,
            local_time_context=local_time_context,
        )
    prompt_message_context = {
        "body_text": user_input,
        "mentions": [],
        "attachments": [],
        "addressed_to_global_user_ids": ["character-001"],
        "broadcast": False,
    }
    state = {
        "storage_timestamp_utc": storage_timestamp_utc,
        "local_time_context": local_time_context,
        "llm_trace_id": "phase3-e2e-trace",
        "user_input": user_input,
        "prompt_message_context": prompt_message_context,
        "platform": "debug",
        "platform_channel_id": "debug-channel-001",
        "channel_type": "private",
        "channel_name": "debug",
        "platform_message_id": platform_message_id,
        "platform_user_id": "platform-user-001",
        "global_user_id": "global-user-001",
        "user_name": "Test User",
        "user_profile": {
            "affinity": 500,
            "last_relationship_insight": "stable",
        },
        "platform_bot_id": "platform-bot-001",
        "character_name": "Kazusa",
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-001",
            "mood": "neutral",
            "global_vibe": "calm",
            "reflection_summary": "",
            "personality_brief": {
                "logic": "direct",
                "tempo": "brief",
                "defense": "none",
                "quirks": "none",
                "taboos": "none",
                "mbti": "INTJ",
            },
            "linguistic_texture_profile": {},
            "boundary_profile": {},
        },
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": "debug",
        "debug_modes": {},
        "should_respond": True,
        "cognitive_episode": cognitive_episode,
    }
    return state


def _user_message_episode(
    *,
    user_input: str,
    platform_message_id: str,
    storage_timestamp_utc: str,
    local_time_context: dict[str, str],
) -> dict[str, Any]:
    """Build a source episode for the initial user code question."""

    episode = {
        "episode_id": "phase3-e2e-user-message",
        "trigger_source": "user_message",
        "input_sources": ["dialog_text"],
        "output_mode": "visible_reply",
        "percepts": [
            {
                "percept_id": "phase3-e2e-user-message:dialog:0",
                "input_source": "dialog_text",
                "content": user_input,
                "visibility": "model_visible",
                "metadata": {},
            }
        ],
        "target_scope": {
            "platform": "debug",
            "platform_channel_id": "debug-channel-001",
            "channel_type": "private",
            "current_platform_user_id": "platform-user-001",
            "current_global_user_id": "global-user-001",
            "current_display_name": "Test User",
            "target_addressed_user_ids": ["character-001"],
            "target_broadcast": False,
        },
        "origin_metadata": {
            "platform": "debug",
            "platform_message_id": platform_message_id,
            "active_turn_platform_message_ids": [platform_message_id],
            "active_turn_conversation_row_ids": [],
            "debug_modes": {},
        },
        "storage_timestamp_utc": storage_timestamp_utc,
        "local_time_context": local_time_context,
    }
    return episode


def _cognition_update_for_actions(
    state: dict[str, Any],
    action_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the persona cognition update around selected action specs."""

    result = {
        "decontexualized_input": state["decontexualized_input"],
        "referents": [],
        "rag_result": {},
        "emotional_appraisal": "calm",
        "interaction_subtext": "direct coding request",
        "internal_monologue": "I can handle this as delayed code work.",
        "character_intent": "HELP",
        "logical_stance": "ACCEPT",
        "judgment_note": "The requested work is bounded and delayed.",
        "social_distance": "friendly",
        "emotional_intensity": "low",
        "vibe_check": "focused",
        "relational_dynamic": "direct request",
        "resolver_capability_requests": [],
        "resolver_pending_resolution": None,
        "resolver_goal_progress": None,
        "action_specs": action_specs,
    }
    return result
