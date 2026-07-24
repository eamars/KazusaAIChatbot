"""Contracts for accepted coding tasks backed by durable coding runs."""

from __future__ import annotations

import pytest

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_actions import (
    materialize_semantic_action_requests,
)


@pytest.mark.asyncio
async def test_accepted_coding_task_execution_enqueues_requested_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coding accepted-task action should bind the coding worker."""

    from kazusa_ai_chatbot.action_spec import execution as execution_module
    from kazusa_ai_chatbot.action_spec.handlers import (
        background_work as background_work_handler,
    )

    action_spec = _materialized_coding_action("start")
    queued_requests: list[dict[str, object]] = []
    accepted_task = {
        "accepted_task_id": "task-coding-001",
        "task_identity_key": "accepted-task-identity-coding-001",
        "accepted_task_summary": "Coding task: Add slugify tests.",
    }

    async def create_accepted_task(request: dict[str, object]) -> dict:
        assert request["action_kind"] == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY
        assert request["accepted_task_seed"] == "Add slugify tests."
        return {
            "status": "created",
            "task": accepted_task,
        }

    async def mark_pending(
        *,
        accepted_task_id: str,
        executor_ref: str,
        updated_at: str,
    ) -> dict:
        assert accepted_task_id == "task-coding-001"
        assert executor_ref == "job-coding-001"
        assert updated_at == "2026-07-09T01:00:00+00:00"
        return {
            **accepted_task,
            "state": "pending",
        }

    async def enqueue_background_work(request: dict[str, object]) -> dict:
        queued_requests.append(dict(request))
        return {
            "status": "pending",
            "queue_state": "queued",
            "job_id": "job-coding-001",
            "job_ref": "background_work_job:job-coding-001",
            "task_summary": request["task_brief"],
            "result_summary": "Background work job queued.",
            "operational_owner": "background_work_job",
            "acknowledgement_constraint": "promise_allowed",
        }

    monkeypatch.setattr(
        background_work_handler,
        "create_or_return_active_accepted_task",
        create_accepted_task,
    )
    monkeypatch.setattr(
        background_work_handler,
        "mark_accepted_task_pending",
        mark_pending,
    )

    results = await execution_module.execute_action_specs_for_trace(
        [action_spec],
        storage_timestamp_utc="2026-07-09T01:00:00+00:00",
        enqueue_background_work_func=enqueue_background_work,
    )

    assert results[0]["status"] == "pending"
    assert results[0]["action_kind"] == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY
    assert results[0]["accepted_task_state"] == "scheduled"
    assert len(queued_requests) == 1
    queued = queued_requests[0]
    assert queued["task_brief"] == "Add slugify tests."
    assert queued["requested_worker"] == "coding_agent"
    assert queued["worker_payload"] == {
        "schema_version": "coding_agent_worker_payload.v2",
        "operation": "start",
        "task_brief": "Add slugify tests.",
        "coding_run_ref": "",
        "execution_request": "",
    }


def test_l2d_materializes_accepted_coding_task_run_action() -> None:
    """Model-facing coding requests should become executable action specs."""

    action_spec = _materialized_coding_action("start")

    assert action_spec["kind"] == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY
    assert action_spec["target"]["owner"] == "background_work"
    assert action_spec["params"]["coding_action"] == "start"
    assert action_spec["params"]["task_brief"] == "Add slugify tests."

    result = ActionSpecEvaluator().evaluate(action_spec)

    assert result["ok"] is True
    assert result["handler_owner"] == "background_work"


def test_l2d_materializes_coding_revision_action() -> None:
    """Proposal revisions should be first-class durable coding actions."""

    action_spec = _materialized_coding_action(
        "revise_proposal",
        coding_run_ref="coding_run:run-001",
    )

    assert action_spec["params"]["coding_action"] == "revise_proposal"
    assert action_spec["params"]["coding_run_ref"] == "coding_run:run-001"
    result = ActionSpecEvaluator().evaluate(action_spec)
    assert result["ok"] is True


def test_l2d_materializes_coding_summary_action() -> None:
    """Run summaries should be first-class durable coding actions."""

    action_spec = _materialized_coding_action(
        "summarize",
        coding_run_ref="coding_run:run-001",
    )

    assert action_spec["params"]["coding_action"] == "summarize"
    assert action_spec["params"]["coding_run_ref"] == "coding_run:run-001"
    result = ActionSpecEvaluator().evaluate(action_spec)
    assert result["ok"] is True


def test_l2d_materializes_coding_blocker_response_action() -> None:
    """A resumable blocker response is an executable durable action."""

    action_spec = _materialized_coding_action(
        "respond_to_blocker",
        coding_run_ref="coding_run:run-001",
    )

    assert action_spec["params"]["coding_action"] == "respond_to_blocker"
    assert action_spec["params"]["coding_run_ref"] == "coding_run:run-001"
    result = ActionSpecEvaluator().evaluate(action_spec)
    assert result["ok"] is True


def test_accepted_coding_task_rejects_worker_local_params() -> None:
    """Model-facing coding actions must not carry execution internals."""

    action_spec = _materialized_coding_action("start")
    action_spec["params"]["workspace_root"] = "C:\\workspace\\kazusa_ai_chatbot"
    action_spec["params"]["execution_specs"] = [{
        "kind": "pytest",
        "args": ["tests/test_slug_tools.py"],
    }]

    result = ActionSpecEvaluator().evaluate(action_spec)

    assert result["ok"] is False
    assert any("worker-local fields" in error for error in result["errors"])


def test_coding_followup_uniquely_binds_current_prompt_safe_run_ref() -> None:
    """A ref-less continuation binds the only eligible current run."""

    action_spec = _materialized_coding_action("approve_and_verify")

    result = ActionSpecEvaluator().evaluate(action_spec)

    assert action_spec["params"]["coding_run_ref"] == "coding_run:run-001"
    assert result["ok"] is True


def test_coding_revision_uniquely_binds_current_prompt_safe_run_ref() -> None:
    """Revision and summary bind the only eligible current run."""

    for decision in ("revise_proposal", "summarize"):
        action_spec = _materialized_coding_action(decision)

        result = ActionSpecEvaluator().evaluate(action_spec)

        assert action_spec["params"]["coding_run_ref"] == "coding_run:run-001"
        assert result["ok"] is True


def test_coding_approval_rejects_missing_message_quote() -> None:
    """Approval requires a non-empty current user-message quote."""

    action_spec = _materialized_coding_action("approve_and_verify")
    action_spec["params"]["approval_evidence"]["quote"] = ""

    result = ActionSpecEvaluator().evaluate(action_spec)

    assert result["ok"] is False
    assert any("approval_evidence.quote" in error for error in result["errors"])


def test_coding_approval_rejects_non_user_or_mismatched_evidence() -> None:
    """Approval provenance must match the current user-message scope exactly."""

    action_spec = _materialized_coding_action("approve_and_verify")
    action_spec["params"]["approval_evidence"]["source_trigger_source"] = (
        "tool_result"
    )
    non_user_result = ActionSpecEvaluator().evaluate(action_spec)

    action_spec = _materialized_coding_action("approve_and_verify")
    action_spec["params"]["approval_evidence"]["requester_global_user_id"] = (
        "other-user"
    )
    mismatched_result = ActionSpecEvaluator().evaluate(action_spec)

    assert non_user_result["ok"] is False
    assert mismatched_result["ok"] is False
    assert any(
        "source_trigger_source" in error
        for error in non_user_result["errors"]
    )
    assert any(
        "scope mismatch" in error
        for error in mismatched_result["errors"]
    )


def test_coding_followup_does_not_scan_user_text_for_run_ref() -> None:
    """A raw user-text ref cannot choose between multiple offered runs."""

    requests = [
        {
            "capability": ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
            "decision": "status",
            "detail": "Check status for coding_run:run-001.",
            "reason": "The user asked for the current coding run status.",
        }
    ]

    state = _cognition_state()
    state["action_selection_context"] = {
        "coding_runs": [
            *state["action_selection_context"]["coding_runs"],
            {
                "coding_run_ref": "coding_run:run-002",
                "allowed_next_actions": ["status"],
            },
        ],
    }
    specs = materialize_semantic_action_requests(requests, state)

    assert specs == []


def test_coding_revision_does_not_scan_user_text_for_run_ref() -> None:
    """A raw user-text ref cannot authorize an offered continuation."""

    requests = [
        {
            "capability": ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
            "decision": "revise_proposal",
            "detail": (
                "For coding_run:run-001, keep tests unchanged and revise "
                "only runtime files."
            ),
            "reason": "The user asked to revise an existing proposal.",
        }
    ]

    state = _cognition_state()
    state["action_selection_context"] = {"coding_runs": []}
    specs = materialize_semantic_action_requests(requests, state)

    assert specs == []


@pytest.mark.asyncio
async def test_queue_rejects_malformed_coding_requested_worker_payload() -> None:
    """The queue should reject unvalidated coding-agent payloads."""

    from kazusa_ai_chatbot.background_work import jobs

    request = _queue_request()
    request["requested_worker"] = "coding_agent"

    with pytest.raises(ValueError, match="worker_payload"):
        await jobs.enqueue_background_work_request(request)

    request["worker_payload"] = {
        "schema_version": "coding_agent_worker_payload.v2",
        "operation": "shell",
        "task_brief": "Run arbitrary shell.",
    }

    with pytest.raises(ValueError, match="operation"):
        await jobs.enqueue_background_work_request(request)


@pytest.mark.asyncio
async def test_queue_accepts_revision_and_summary_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The queue should accept the new validated coding operations."""

    from kazusa_ai_chatbot.background_work import jobs

    async def insert_job(job: dict[str, object]) -> dict[str, object]:
        return dict(job)

    monkeypatch.setattr(jobs, "insert_background_work_job", insert_job)

    for operation in ("revise_proposal", "summarize"):
        request = _queue_request()
        request["requested_worker"] = "coding_agent"
        request["worker_payload"] = {
            "schema_version": "coding_agent_worker_payload.v2",
            "operation": operation,
            "task_brief": "Revise or summarize a coding run.",
            "coding_run_ref": "coding_run:run-001",
            "execution_request": "",
        }

        result = await jobs.enqueue_background_work_request(request)

        assert result["status"] == "pending"


@pytest.mark.asyncio
async def test_coding_worker_start_payload_starts_durable_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A start payload should route to the durable coding-run API."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    decide = AsyncMock(return_value=("code_modifying", "source-backed patch"))
    start = AsyncMock(return_value=_coding_run_response(
        status="awaiting_approval",
        run_id="run-001",
        objective_type="propose_patch",
        answer_text="Patch proposal is ready.",
    ))
    monkeypatch.setattr(
        coding_agent,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(coding_agent, "decide_background_coding_operation", decide)
    monkeypatch.setattr(coding_agent, "start_coding_run", start)

    result = await coding_agent.execute(
        _worker_decision({
            "schema_version": "coding_agent_worker_payload.v2",
            "operation": "start",
            "task_brief": "Modify slugify behavior.",
            "coding_run_ref": "",
            "execution_request": "",
        }),
        max_output_chars=1000,
    )

    assert result["status"] == "succeeded"
    assert "coding_run:run-001" in result["artifact_text"]
    assert result["worker_metadata"]["schema_version"] == (
        "coding_agent_worker_metadata.v3"
    )
    assert result["worker_metadata"]["coding_run_status"] == "awaiting_approval"
    start_request = start.await_args.args[0]
    assert start_request["objective_type"] == "propose_patch"
    assert start_request["question"] == "Modify slugify behavior."


@pytest.mark.asyncio
async def test_coding_worker_start_metadata_projects_created_files_and_alignment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Worker metadata v2 should expose durable proposal alignment evidence."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    run_response = _coding_run_response(
        status="awaiting_approval",
        run_id="run-001",
        objective_type="propose_patch",
        answer_text="Patch proposal is ready.",
    )
    run_response["created_files"] = [{
        "path": "counter_cli/formatters.py",
        "role": "Create formatter helper.",
    }]
    run_response["alignment"] = {
        "status": "pass",
        "request_satisfied": True,
        "matched_criteria": ["criterion-format"],
        "missing_criteria": [],
        "blockers": [],
    }
    run_response["trace_summary"] = [
        *[f"step:{index}" for index in range(20)],
        "writing_alignment:status=pass",
    ]
    decide = AsyncMock(return_value=("code_modifying", "source-backed patch"))
    start = AsyncMock(return_value=run_response)
    monkeypatch.setattr(
        coding_agent,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(coding_agent, "decide_background_coding_operation", decide)
    monkeypatch.setattr(coding_agent, "start_coding_run", start)

    result = await coding_agent.execute(
        _worker_decision({
            "schema_version": "coding_agent_worker_payload.v2",
            "operation": "start",
            "task_brief": "Modify slugify behavior.",
            "coding_run_ref": "",
            "execution_request": "",
        }),
        max_output_chars=1000,
    )

    assert result["status"] == "succeeded"
    assert result["worker_metadata"]["created_files"] == [{
        "path": "counter_cli/formatters.py",
        "role": "Create formatter helper.",
    }]
    assert result["worker_metadata"]["alignment"]["status"] == "pass"
    assert "writing_alignment:status=pass" in (
        result["worker_metadata"]["trace_summary"]
    )


@pytest.mark.asyncio
async def test_background_operation_routes_source_backed_mixed_work_to_modifying(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Explicit source plus mixed create/edit must choose modifying."""

    from kazusa_ai_chatbot.coding_agent import supervisor

    payloads: list[dict[str, object]] = []
    source_root = tmp_path / "source"
    source_root.mkdir()

    async def fake_ainvoke(
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        payloads.append(json.loads(messages[-1].content))
        return SimpleNamespace(
            content=json.dumps({
                "operation": "code_writing",
                "reason": "The task asks for a new helper file.",
            }),
        )

    monkeypatch.setattr(
        supervisor._background_coding_router_llm,
        "ainvoke",
        fake_ainvoke,
    )

    operation, reason = await supervisor.decide_background_coding_operation({
        "question": (
            "Use the local source checkout, create a formatter module, "
            "and wire the existing CLI to call it."
        ),
        "local_root_hint": str(source_root),
        "source_scope_hint": "repository",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert payloads[0]["source_context"] == {
        "kind": "explicit_source",
        "fields": ["local_root_hint", "source_scope_hint"],
    }
    assert operation == "code_modifying"
    assert "source" in reason.lower()


@pytest.mark.asyncio
async def test_background_operation_keeps_source_free_artifact_as_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Truly source-free artifact requests remain in code_writing."""

    from kazusa_ai_chatbot.coding_agent import supervisor

    payloads: list[dict[str, object]] = []

    async def fake_ainvoke(
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        payloads.append(json.loads(messages[-1].content))
        return SimpleNamespace(
            content=json.dumps({
                "operation": "code_writing",
                "reason": "The task asks for a new standalone artifact.",
            }),
        )

    monkeypatch.setattr(
        supervisor._background_coding_router_llm,
        "ainvoke",
        fake_ainvoke,
    )

    operation, reason = await supervisor.decide_background_coding_operation({
        "question": "Create a standalone Python script that formats CSV rows.",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert payloads[0]["source_context"] == {
        "kind": "source_free",
        "fields": [],
    }
    assert operation == "code_writing"
    assert "standalone" in reason


@pytest.mark.asyncio
async def test_background_operation_routes_managed_preflight_patch_to_modifying(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Managed-copy preflight keeps its semantic existing-source patch route."""

    from kazusa_ai_chatbot.coding_agent import supervisor

    source_root = tmp_path / "source"
    source_root.mkdir()

    async def fake_ainvoke(
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        prompt = messages[0].content
        assert "managed-copy\n  preflight" in prompt.casefold()
        payload = json.loads(messages[-1].content)
        assert "preflight" in payload["task"].casefold()
        assert "enabled" not in json.dumps(payload["operation_limits"]).casefold()
        return SimpleNamespace(
            content=json.dumps({
                "operation": "code_modifying",
                "reason": "The task is an existing-source patch proposal.",
            }),
        )

    monkeypatch.setattr(
        supervisor._background_coding_router_llm,
        "ainvoke",
        fake_ainvoke,
    )

    operation, _ = await supervisor.decide_background_coding_operation({
        "question": "Fix the parser and preflight the patch in a managed copy.",
        "local_root_hint": str(source_root),
        "source_scope_hint": "repository",
        "workspace_root": str(tmp_path / "workspace"),
    })

    assert operation == "code_modifying"


@pytest.mark.asyncio
async def test_coding_worker_start_payload_preserves_visible_local_path_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """A visible local source path should reach the durable run request."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    source_root = tmp_path / "source checkout with spaces"
    source_root.mkdir()
    task_brief = (
        f"Use the local source checkout at {source_root} and explain app.py."
    )
    decide = AsyncMock(return_value=("code_reading", "source-backed read"))
    start = AsyncMock(return_value=_coding_run_response(
        status="completed",
        run_id="run-local-001",
        objective_type="read_only",
        answer_text="app.py was explained.",
    ))
    monkeypatch.setattr(
        coding_agent,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(coding_agent, "decide_background_coding_operation", decide)
    monkeypatch.setattr(coding_agent, "start_coding_run", start)

    result = await coding_agent.execute(
        _worker_decision({
            "schema_version": "coding_agent_worker_payload.v2",
            "operation": "start",
            "task_brief": task_brief,
            "coding_run_ref": "",
            "execution_request": "",
        }),
        max_output_chars=1000,
    )

    assert result["status"] == "succeeded"
    assert decide.await_args.args[0]["local_path_hint"] == str(source_root)
    assert start.await_args.args[0]["local_path_hint"] == str(source_root)


def test_local_source_hint_ignores_repo_relative_paths_before_checkout(
    tmp_path,
) -> None:
    """Repo-relative path mentions should not swallow a later local checkout."""

    from kazusa_ai_chatbot.background_work.subagent.coding_agent import (
        _local_source_hints_from_task_brief,
    )

    source_root = tmp_path / "source checkout"
    source_root.mkdir()
    task_brief = (
        "Add counter_cli/formatters.py, wire counter_cli/cli.py, and use "
        f"the local source checkout at {source_root}."
    )

    assert _local_source_hints_from_task_brief(task_brief) == {
        "local_path_hint": str(source_root),
    }


@pytest.mark.asyncio
async def test_coding_worker_revision_payload_continues_without_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Revision payloads should continue the run without approval fields."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    continue_run = AsyncMock(return_value=_coding_run_response(
        status="awaiting_approval",
        run_id="run-001",
        objective_type="propose_patch",
        answer_text="Proposal was revised.",
    ))
    monkeypatch.setattr(
        coding_agent,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(coding_agent, "continue_coding_run", continue_run)

    result = await coding_agent.execute(
        _worker_decision({
            "schema_version": "coding_agent_worker_payload.v2",
            "operation": "revise_proposal",
            "task_brief": "Keep tests unchanged; revise runtime files only.",
            "coding_run_ref": "coding_run:run-001",
            "execution_request": "",
        }),
        max_output_chars=1000,
    )

    assert result["status"] == "succeeded"
    assert result["worker_metadata"]["worker_operation"] == "revise_proposal"
    continue_request = continue_run.await_args.args[0]
    assert continue_request["run_id"] == "run-001"
    assert continue_request["action"] == "revise_proposal"
    assert continue_request["revision_instruction"] == (
        "Keep tests unchanged; revise runtime files only."
    )
    assert "approval" not in continue_request
    assert "execution_specs" not in continue_request


@pytest.mark.asyncio
async def test_coding_worker_summary_payload_continues_without_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Summary payloads should project run state without approval fields."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    continue_run = AsyncMock(return_value={
        **_coding_run_response(
            status="awaiting_approval",
            run_id="run-001",
            objective_type="propose_patch",
            answer_text="Files changed: app.py.",
        ),
        "allowed_next_actions": [
            "revise_proposal",
            "summarize",
            "approve_and_verify",
            "cancel",
        ],
    })
    monkeypatch.setattr(
        coding_agent,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(coding_agent, "continue_coding_run", continue_run)

    result = await coding_agent.execute(
        _worker_decision({
            "schema_version": "coding_agent_worker_payload.v2",
            "operation": "summarize",
            "task_brief": "Summarize changed files.",
            "coding_run_ref": "coding_run:run-001",
            "execution_request": "",
        }),
        max_output_chars=1000,
    )

    assert result["status"] == "succeeded"
    assert result["worker_metadata"]["worker_operation"] == "summarize"
    assert result["worker_metadata"]["allowed_next_actions"] == [
        "revise_proposal",
        "summarize",
        "approve_and_verify",
        "cancel",
    ]
    continue_request = continue_run.await_args.args[0]
    assert continue_request["action"] == "summarize"
    assert "approval" not in continue_request
    assert "execution_specs" not in continue_request


@pytest.mark.asyncio
async def test_coding_worker_approval_payload_forwards_semantic_extra_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Approval payloads leave primary verification planning to coding_run."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    continue_run = AsyncMock(return_value=_coding_run_response(
        status="completed",
        run_id="run-001",
        objective_type="propose_patch",
        answer_text="Verification passed.",
    ))
    monkeypatch.setattr(
        coding_agent,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(coding_agent, "continue_coding_run", continue_run)

    result = await coding_agent.execute(
        _worker_decision({
            "schema_version": "coding_agent_worker_payload.v2",
            "operation": "approve_and_verify",
            "task_brief": "Approved; run focused pytest.",
            "coding_run_ref": "coding_run:run-001",
            "execution_request": "Run focused pytest.",
            "execution_specs": [
                {
                    "tool": "pytest",
                    "pytest_selectors": ["tests/test_slug_tools.py"],
                    "timeout_seconds": 20,
                }
            ],
            "source_scope": {"source_user_id": "global-user-001"},
            "storage_timestamp_utc": "2026-07-09T01:00:00+00:00",
            "approval_evidence": {
                "source_message_id": "message-001",
                "source_trigger_source": "user_message",
                "requester_global_user_id": "global-user-001",
                "quote": "Approved; run focused pytest.",
                "storage_timestamp_utc": "2026-07-09T01:00:00+00:00",
            },
        }),
        max_output_chars=1000,
    )

    assert result["status"] == "succeeded"
    assert result["worker_metadata"]["coding_run_status"] == "completed"
    continue_request = continue_run.await_args.args[0]
    assert continue_request["run_id"] == "run-001"
    assert continue_request["action"] == "approve_and_verify"
    assert continue_request["approval"]["approved"] is True
    assert continue_request["approval"]["approved_by"] == "global-user-001"
    assert continue_request["approval"]["approval_evidence"] == {
        "source_message_id": "message-001",
        "source_trigger_source": "user_message",
        "requester_global_user_id": "global-user-001",
        "quote": "Approved; run focused pytest.",
        "storage_timestamp_utc": "2026-07-09T01:00:00+00:00",
    }
    assert continue_request["execution_request"] == "Run focused pytest."
    assert "execution_specs" not in continue_request


@pytest.mark.asyncio
async def test_coding_worker_rejects_approval_without_evidence_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Worker approval never fabricates requester or timestamp provenance."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    continue_run = AsyncMock()
    monkeypatch.setattr(
        coding_agent,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(tmp_path / "workspace"),
    )
    monkeypatch.setattr(coding_agent, "continue_coding_run", continue_run)

    result = await coding_agent.execute(
        _worker_decision({
            "schema_version": "coding_agent_worker_payload.v2",
            "operation": "approve_and_verify",
            "task_brief": "Approved; run focused pytest.",
            "coding_run_ref": "coding_run:run-001",
            "execution_request": "Run focused pytest.",
            "source_scope": {"source_user_id": "global-user-001"},
            "storage_timestamp_utc": "2026-07-09T01:00:00+00:00",
        }),
        max_output_chars=1000,
    )

    assert result["status"] == "rejected"
    assert continue_run.await_count == 0


@pytest.mark.asyncio
async def test_worker_tick_dispatches_requested_coding_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requested coding jobs should bypass the generic router."""

    from kazusa_ai_chatbot.background_work import worker as worker_module

    fake_job = {
        "job_id": "job-coding-001",
        "accepted_task_id": "task-coding-001",
        "task_brief": "Modify slugify behavior.",
        "max_output_chars": 3000,
        "source_context": "User accepted durable coding work.",
        "requested_worker": "coding_agent",
        "worker_payload": {
            "schema_version": "coding_agent_worker_payload.v2",
            "operation": "start",
            "task_brief": "Modify slugify behavior.",
            "coding_run_ref": "",
            "execution_request": "",
        },
        "source_action_attempt_id": "action_attempt:coding-001",
        "source_platform": "debug",
        "source_channel_id": "debug:user:test-user",
        "source_channel_type": "private",
        "source_platform_bot_id": "debug-bot-001",
        "source_character_name": "Test Character",
        "source_message_id": "message-001",
        "requester_global_user_id": "global-user-001",
    }
    dispatch = AsyncMock(return_value={
        "status": "succeeded",
        "worker": "coding_agent",
        "artifact_text": "Coding run ref: coding_run:run-001",
        "failure_summary": "",
        "result_summary": "coding_agent run; start; awaiting_approval",
        "worker_metadata": {
            "schema_version": "coding_agent_worker_metadata.v3",
            "coding_run_ref": "coding_run:run-001",
            "coding_run_context": {
                "schema_version": "coding_run_context.v1",
                "coding_run_ref": "coding_run:run-001",
                "status": "awaiting_approval",
                "objective_summary": "Modify slugify behavior.",
                "allowed_next_actions": ["approve_and_verify", "cancel"],
                "active_blocker": None,
                "followup_open": True,
                "updated_at": "2026-07-10T00:00:00Z",
                "execution_specs": [{"tool": "pytest"}],
                "approval_evidence": {"quote": "private"},
            },
        },
    })
    monkeypatch.setattr(
        worker_module,
        "claim_background_work_job",
        AsyncMock(side_effect=[fake_job, None]),
    )
    monkeypatch.setattr(
        worker_module,
        "route_background_work",
        AsyncMock(side_effect=AssertionError("router should not run")),
    )
    monkeypatch.setattr(worker_module, "dispatch_background_work", dispatch)
    monkeypatch.setattr(
        worker_module,
        "mark_accepted_task_running",
        AsyncMock(),
    )
    mark_result_ready = AsyncMock()
    monkeypatch.setattr(
        worker_module,
        "mark_tool_result_ready",
        mark_result_ready,
    )
    monkeypatch.setattr(
        worker_module,
        "complete_background_work_job",
        AsyncMock(),
    )

    result = await worker_module.run_background_work_worker_tick(
        claim_limit=1,
        lease_seconds=60,
        max_attempts=3,
        worker_id="worker-test",
    )

    assert result["processed_count"] == 1
    assert result["succeeded_count"] == 1
    dispatch_decision = dispatch.await_args.args[0]
    assert dispatch_decision["worker"] == "coding_agent"
    assert dispatch_decision["worker_payload"]["operation"] == "start"
    assert dispatch_decision["worker_payload"]["source_action_attempt_id"] == (
        "action_attempt:coding-001"
    )
    persisted_context = mark_result_ready.await_args.kwargs["coding_run_context"]
    assert persisted_context is not None
    assert "execution_specs" not in persisted_context
    assert "approval_evidence" not in persisted_context


def _materialized_coding_action(
    decision: str,
    *,
    coding_run_ref: str = "",
) -> dict[str, object]:
    requests = [
        {
            "capability": ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
            "decision": decision,
            "detail": "Run focused pytest if approval is being given.",
            "reason": "The user asked for durable coding work.",
            "coding_run_ref": coding_run_ref,
        }
    ]
    specs = materialize_semantic_action_requests(requests, _cognition_state())
    assert len(specs) == 1
    return specs[0]


def _cognition_state() -> dict[str, object]:
    return {
        "storage_timestamp_utc": "2026-07-09T01:00:00+00:00",
        "decontextualized_input": "Add slugify tests.",
        "platform": "debug",
        "platform_channel_id": "debug:user:test-user",
        "channel_type": "private",
        "platform_message_id": "message-001",
        "platform_bot_id": "debug-bot-001",
        "global_user_id": "global-user-001",
        "platform_user_id": "debug-user-001",
        "user_name": "Test User",
        "character_profile": {
            "name": "Test Character",
            "global_user_id": "character-global-001",
        },
        "cognitive_episode": {
            "trigger_source": "user_message",
            "storage_timestamp_utc": "2026-07-09T01:00:00+00:00",
            "created_at": "2026-07-09T01:00:00+00:00",
            "target_scope": {
                "current_global_user_id": "global-user-001",
            },
            "origin_metadata": {
                "platform_message_id": "message-001",
            },
            "percepts": [{
                "schema_version": "percept.v1",
                "percept_kind": "dialog",
                "source_kind": "dialog",
                "source_id": "percept:message-001",
                "content": {
                    "semantic_text": "Add slugify tests.",
                    "text": "Add slugify tests.",
                },
                "observed_at": "2026-07-09T01:00:00+00:00",
            }],
        },
        "action_selection_context": {
            "coding_runs": [{
                "coding_run_ref": "coding_run:run-001",
                "allowed_next_actions": [
                    "revise_proposal",
                    "summarize",
                    "status",
                    "approve_and_verify",
                    "respond_to_blocker",
                    "cancel",
                ],
            }],
        },
        "conversation_progress": {},
    }


def _queue_request() -> dict[str, object]:
    return {
        "action_attempt_id": "action_attempt:coding-001",
        "idempotency_key": "background_work:coding-001",
        "task_brief": "Modify slugify behavior.",
        "source_platform": "debug",
        "source_channel_id": "debug:user:test-user",
        "source_channel_type": "private",
        "source_message_id": "message-001",
        "source_platform_bot_id": "debug-bot-001",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-001",
        "requester_platform_user_id": "debug-user-001",
        "requester_display_name": "Test User",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "storage_timestamp_utc": "2026-07-09T01:00:00+00:00",
    }


def _worker_decision(worker_payload: dict[str, object]) -> dict[str, object]:
    return {
        "action": "execute",
        "worker": "coding_agent",
        "reason": "Validated background action requested this worker.",
        "task_brief": "Modify slugify behavior.",
        "source_summary": "User asked for durable coding work.",
        "worker_payload": worker_payload,
    }


def _coding_run_response(
    *,
    status: str,
    run_id: str,
    objective_type: str,
    answer_text: str,
) -> dict[str, object]:
    return {
        "status": status,
        "run_id": run_id,
        "goal": "Modify slugify behavior.",
        "objective_type": objective_type,
        "answer_text": answer_text,
        "repository": None,
        "source_scope": None,
        "evidence": [],
        "patch_artifacts": [],
        "changed_files": [],
        "apply_attempts": [],
        "execution_attempts": [],
        "repair_attempts": [],
        "attempts": [],
        "blockers": [],
        "events": [],
        "limitations": [],
        "trace_summary": ["coding_run:test"],
    }
