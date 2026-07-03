"""Live LLM role diagnostics for code-writing downstream roles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
    run_writing_programmer_contract,
)
from kazusa_ai_chatbot.coding_agent.code_writing.synthesizer import (
    synthesize_patch_proposal,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

TRACE_DIR = Path("test_artifacts/llm_traces/coding_agent_phase2_roles")


def _write_trace(name: str, payload: dict[str, object]) -> Path:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{name}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


async def _run_programmer_case(
    *,
    case_id: str,
    contract: dict[str, Any],
) -> None:
    trace: dict[str, object] = {}
    result = await run_writing_programmer_contract(
        artifact_contract=contract,
        trace=trace,
    )
    trace_path = _write_trace(
        f"{case_id}_programmer",
        {
            "case_id": case_id,
            "role": "writing_programmer",
            "contract": contract,
            "result": result,
            "trace": trace,
        },
    )
    print(f"Writing programmer live trace={trace_path}")

    assert result["status"] == "succeeded", f"trace={trace_path}"
    assert result["artifact_id"] == contract["artifact_id"], f"trace={trace_path}"
    assert result["code_artifact"].strip(), f"trace={trace_path}"


async def test_live_writing_programmer_source_artifact() -> None:
    await _run_programmer_case(
        case_id="source_artifact",
        contract={
            "artifact_id": "log_counter",
            "file_label": "log counter source",
            "file_kind": "source",
            "content_format": "python",
            "purpose": (
                "Provide a command-line utility that counts log lines by "
                "severity."
            ),
            "imports": ["import argparse", "from pathlib import Path"],
            "provided_interfaces": [
                "count_log_severities(path) returns counts and malformed count",
            ],
            "consumed_interfaces": [],
            "required_behavior": [
                "Recognize DEBUG, INFO, WARNING, ERROR, and CRITICAL prefixes.",
                "Skip malformed lines and count them.",
                "Handle missing files with a clear error path.",
                "Use only the Python standard library.",
            ],
        },
    )


async def test_live_writing_programmer_markdown_artifact() -> None:
    await _run_programmer_case(
        case_id="markdown_artifact",
        contract={
            "artifact_id": "usage_readme",
            "file_label": "usage README",
            "file_kind": "docs",
            "content_format": "markdown",
            "purpose": "Document command-line usage for a generated utility.",
            "imports": [],
            "provided_interfaces": [],
            "consumed_interfaces": [
                "A command-line script accepts input and output file paths.",
            ],
            "required_behavior": [
                "Explain the input file shape.",
                "Explain command invocation.",
                "Describe output columns.",
                "Do not claim commands were executed.",
            ],
        },
    )


async def test_live_synthesizer_reports_patch_proposal_without_execution() -> None:
    trace: dict[str, object] = {}
    answer_text, limitations = await synthesize_patch_proposal(
        question=(
            "Create a small standard-library Python script that counts log "
            "entries by severity."
        ),
        pm_decision={
            "status": "complete",
            "reason": "One source artifact was generated.",
            "information_request": None,
            "child_pm_task": None,
            "programmer_task": None,
            "repair_instruction": None,
            "completion_report": {
                "pm_id": "writing_pm_root",
                "status": "complete",
                "provided_facts": ["log counter artifact generated"],
                "created_artifacts": [
                    {
                        "artifact_id": "log_counter",
                        "purpose": "Count log entries by severity.",
                    }
                ],
                "consumed_facts": [],
                "open_risks": [],
                "next_dependency_needs": [],
            },
            "blocker": None,
        },
        generated_artifacts=[
            {
                "artifact_id": "log_counter",
                "file_label": "log counter source",
                "file_kind": "source",
                "content_format": "python",
                "path": "src/log_counter.py",
                "content": "VALUE = 1\n",
                "purpose": "Count log entries by severity.",
            }
        ],
        patch_artifacts=[
            {
                "artifact_id": "log_counter_patch",
                "base": "new_file",
                "diff_text": "diff --git a/src/log_counter.py b/src/log_counter.py\n",
                "files": ["src/log_counter.py"],
                "summary": "Create log counter source file.",
            }
        ],
        validation={
            "status": "succeeded",
            "parsed": True,
            "sandbox_applied": True,
            "errors": [],
            "warnings": [],
            "files": ["src/log_counter.py"],
        },
        external_evidence=[],
        limitations=[],
        preferred_language=None,
        max_answer_chars=1200,
        trace=trace,
    )
    trace_path = _write_trace(
        "synthesizer_patch_proposal",
        {
            "case_id": "synthesizer_patch_proposal",
            "role": "synthesizer",
            "answer_text": answer_text,
            "limitations": limitations,
            "trace": trace,
        },
    )
    print(f"Writing synthesizer live trace={trace_path}")

    assert answer_text.strip(), f"trace={trace_path}"
    assert limitations, f"trace={trace_path}"
