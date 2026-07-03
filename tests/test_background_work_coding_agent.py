"""Tests for the coding-agent background-work adapter."""

from __future__ import annotations

from typing import Any

import pytest


CODE_TASK = "Explain how image reading is implemented in the repository."


@pytest.mark.asyncio
async def test_coding_agent_worker_maps_success_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The worker should call the public coding-agent interface and map result."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    calls: list[dict[str, Any]] = []
    workspace_root = tmp_path / "workspace"

    async def fake_handle_background_coding_task(
        request: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(dict(request))
        return {
            "status": "succeeded",
            "operation": "code_reading",
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
            "evidence": [
                {
                    "path": "src/app/image_pipeline.py",
                    "line_start": 10,
                    "line_end": 20,
                    "symbol_or_topic": "image reading",
                    "excerpt": "raw source excerpt should not enter metadata",
                    "reason": "Shows image handling.",
                }
            ],
            "patch_artifacts": [],
            "created_files": [],
            "changed_files": [],
            "validation": None,
            "limitations": [],
            "trace_summary": ["fetch:succeeded", "reading:succeeded"],
        }

    monkeypatch.setattr(
        coding_agent,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(workspace_root),
    )
    monkeypatch.setattr(
        coding_agent,
        "handle_background_coding_task",
        fake_handle_background_coding_task,
    )

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

    assert calls == [
        {
            "question": CODE_TASK,
            "source_summary": "User asked about image reading.",
            "workspace_root": str(workspace_root),
            "max_answer_chars": 120,
            "max_artifact_chars": 960,
        }
    ]
    assert result["status"] == "succeeded"
    assert result["worker"] == "coding_agent"
    assert result["artifact_text"] == "Image reading uses media attachments."
    assert result["failure_summary"] == ""
    assert "fixture/reader" in result["result_summary"]
    assert result["worker_metadata"]["coding_operation"] == "code_reading"
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
    """The worker should expose writing proposals without raw diff metadata."""

    from kazusa_ai_chatbot.background_work.subagent import coding_agent

    calls: list[dict[str, Any]] = []
    workspace_root = tmp_path / "workspace"

    async def fake_handle_background_coding_task(
        request: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(dict(request))
        return {
            "status": "succeeded",
            "operation": "code_writing",
            "answer_text": "Proposed a standard-library log parser script.",
            "repository": None,
            "source_scope": None,
            "evidence": [],
            "patch_artifacts": [
                {
                    "artifact_id": "log_parser",
                    "base": "new file",
                    "diff_text": "--- raw diff should not be stored here",
                    "files": ["src/log_parser.py"],
                    "summary": "Creates the parser script.",
                }
            ],
            "created_files": [
                {
                    "path": "src/log_parser.py",
                    "role": "source",
                }
            ],
            "changed_files": [],
            "validation": {
                "status": "succeeded",
                "parsed": True,
                "sandbox_applied": False,
                "errors": [],
                "warnings": [],
                "files": ["src/log_parser.py"],
            },
            "limitations": [],
            "trace_summary": ["background_coding:code_writing"],
        }

    monkeypatch.setattr(
        coding_agent,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(workspace_root),
    )
    monkeypatch.setattr(
        coding_agent,
        "handle_background_coding_task",
        fake_handle_background_coding_task,
    )

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

    assert calls == [
        {
            "question": "Create a Python command-line script that summarizes logs.",
            "source_summary": "User asked for new code.",
            "workspace_root": str(workspace_root),
            "max_answer_chars": 200,
            "max_artifact_chars": 1600,
        }
    ]
    assert result["status"] == "succeeded"
    assert result["artifact_text"] == "Proposed a standard-library log parser script."
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
            "change_type": "",
            "summary": "",
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

    async def fake_handle_background_coding_task(
        _request: dict[str, Any],
    ) -> dict[str, Any]:
        raise AssertionError("handle_background_coding_task should not be called")

    monkeypatch.setattr(coding_agent, "CODING_AGENT_WORKSPACE_ROOT", "")
    monkeypatch.setattr(
        coding_agent,
        "handle_background_coding_task",
        fake_handle_background_coding_task,
    )

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
