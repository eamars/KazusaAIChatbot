"""Tests for the coding-agent background-work adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


CODE_TASK = "Explain how image reading is implemented in the repository."


@pytest.mark.asyncio
async def test_coding_agent_worker_maps_success_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Generic coding work should classify and start one durable read run."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    workspace_root = tmp_path / "workspace"
    decide = AsyncMock(return_value=(
        "code_reading",
        "The task asks for bounded source-code reading.",
    ))
    start = AsyncMock(return_value={
        "status": "completed",
        "run_id": "run-reading",
        "goal": CODE_TASK,
        "objective_type": "read_only",
        "answer_text": "Image reading uses media attachments.",
        "repository": {
            "provider": "github",
            "owner": "fixture",
            "repo": "reader",
            "source_url": "https://github.com/fixture/reader",
            "requested_ref": "main",
            "resolved_ref": "main",
            "current_commit": "a" * 40,
            "default_branch": "main",
            "storage_kind": "managed_download",
            "managed_checkout": True,
            "dirty_state": "clean",
        },
        "source_scope": {
            "kind": "repository",
            "repo_relative_path": None,
            "source_url": "https://github.com/fixture/reader",
            "requested_ref": "main",
            "interpretation": "entire repository",
        },
        "evidence": [{
            "path": "src/app/image_pipeline.py",
            "line_start": 10,
            "line_end": 20,
            "symbol_or_topic": "image reading",
            "excerpt": "raw source excerpt should not enter metadata",
            "reason": "Shows image handling.",
        }],
        "patch_artifacts": [],
        "created_files": [],
        "changed_files": [],
        "apply_attempts": [],
        "execution_attempts": [],
        "repair_attempts": [],
        "blockers": [],
        "limitations": [],
        "allowed_next_actions": [],
        "trace_summary": ["fetch:succeeded", "reading:succeeded"],
    })

    monkeypatch.setattr(
        coding_agent,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(workspace_root),
    )
    monkeypatch.setattr(
        coding_agent,
        "decide_background_coding_operation",
        decide,
    )
    monkeypatch.setattr(coding_agent, "start_coding_run", start)

    result = await coding_agent.execute(
        {
            "action": "execute",
            "worker": "coding_agent",
            "reason": "The task asks for bounded source-code reading.",
            "task_brief": CODE_TASK,
            "source_summary": "User asked about image reading.",
        },
        max_output_chars=120,
    )

    route_request = decide.await_args.args[0]
    assert route_request["question"] == CODE_TASK
    assert route_request["source_summary"] == "User asked about image reading."
    start_request = start.await_args.args[0]
    assert start_request["question"] == CODE_TASK
    assert start_request["objective_type"] == "read_only"
    assert result["status"] == "succeeded"
    assert result["worker"] == "coding_agent"
    assert "Image reading uses media attachments." in result["artifact_text"]
    assert "coding_run:run-reading" in result["artifact_text"]
    assert result["failure_summary"] == ""
    assert result["worker_metadata"]["coding_operation"] == "code_reading"
    assert result["worker_metadata"]["worker_operation"] == "start"
    assert result["worker_metadata"]["repository"]["owner"] == "fixture"
    evidence_refs = result["worker_metadata"]["evidence_refs"]
    assert evidence_refs == [
        {
            "path": "src/app/image_pipeline.py",
            "line_start": 10,
            "line_end": 20,
            "symbol_or_topic": "image reading",
            "reason": "Shows image handling.",
        }
    ]
    assert "raw source excerpt" not in repr(result)
    assert str(workspace_root) not in repr(result)
    assert "local_root" not in repr(result)
    assert "cache_key" not in repr(result)


@pytest.mark.asyncio
async def test_coding_agent_worker_maps_writing_proposal_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Generic coding work should classify and start one durable proposal run."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    workspace_root = tmp_path / "workspace"
    decide = AsyncMock(return_value=(
        "code_writing",
        "The task asks for a new code artifact.",
    ))
    start = AsyncMock(return_value={
        "status": "awaiting_approval",
        "run_id": "run-writing",
        "goal": "Create a Python command-line script that summarizes logs.",
        "objective_type": "propose_patch",
        "answer_text": "Proposed a standard-library log parser script.",
        "repository": None,
        "source_scope": None,
        "evidence": [],
        "patch_artifacts": [{
            "artifact_id": "log_parser",
            "base": "new file",
            "diff_text": "--- raw diff should not be stored here",
            "files": ["src/log_parser.py"],
            "summary": "Creates the parser script.",
        }],
        "created_files": [{
            "path": "src/log_parser.py",
            "role": "source",
        }],
        "changed_files": [],
        "apply_attempts": [],
        "execution_attempts": [],
        "repair_attempts": [],
        "blockers": [],
        "limitations": [],
        "allowed_next_actions": ["approve_and_verify", "cancel"],
        "trace_summary": ["background_coding:code_writing"],
    })

    monkeypatch.setattr(
        coding_agent,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(workspace_root),
    )
    monkeypatch.setattr(
        coding_agent,
        "decide_background_coding_operation",
        decide,
    )
    monkeypatch.setattr(coding_agent, "start_coding_run", start)

    result = await coding_agent.execute(
        {
            "action": "execute",
            "worker": "coding_agent",
            "reason": "The task asks for a code artifact proposal.",
            "task_brief": (
                "Create a Python command-line script that summarizes logs."
            ),
            "source_summary": "User asked for new code.",
        },
        max_output_chars=200,
    )

    assert decide.await_args.args[0]["question"] == (
        "Create a Python command-line script that summarizes logs."
    )
    assert start.await_args.args[0]["objective_type"] == "propose_patch"
    assert result["status"] == "succeeded"
    assert "Proposed a standard-library log parser script." in (
        result["artifact_text"]
    )
    assert "coding_run:run-writing" in result["artifact_text"]
    assert result["worker_metadata"]["coding_operation"] == "code_writing"
    assert result["worker_metadata"]["patch_artifacts"] == [
        {
            "artifact_id": "log_parser",
            "files": ["src/log_parser.py"],
            "summary": "Creates the parser script.",
        }
    ]
    assert result["worker_metadata"]["created_files"] == [
        {
            "path": "src/log_parser.py",
            "role": "source",
        }
    ]
    assert "raw diff" not in repr(result)
    assert str(workspace_root) not in repr(result)


@pytest.mark.asyncio
async def test_coding_agent_worker_fails_closed_without_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing workspace config should not call the coding agent."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    monkeypatch.setattr(coding_agent, "CODING_AGENT_WORKSPACE_ROOT", "")
    decide = AsyncMock()
    start = AsyncMock()
    monkeypatch.setattr(coding_agent, "decide_background_coding_operation", decide)
    monkeypatch.setattr(coding_agent, "start_coding_run", start)

    result = await coding_agent.execute(
        {
            "action": "execute",
            "worker": "coding_agent",
            "reason": "The task asks for bounded source-code reading.",
            "task_brief": CODE_TASK,
        },
        max_output_chars=120,
    )

    assert result["status"] == "failed"
    assert result["worker"] == "coding_agent"
    assert result["artifact_text"] == ""
    assert "workspace" in result["failure_summary"].lower()
    assert "workspace_root" not in repr(result)
    decide.assert_not_awaited()
    start.assert_not_awaited()
