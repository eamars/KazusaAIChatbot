"""Live E2E gates for coding-agent new-artifact proposals."""

import json
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.coding_agent import propose_code_change


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

TRACE_ROOT = Path("test_artifacts/llm_traces/coding_agent_phase2_e2e")
WORKSPACE_ROOT = Path("test_artifacts/coding_agent_phase2_e2e_workspace")
MAX_ANSWER_CHARS = 4000
MAX_ARTIFACT_CHARS = 48000

GATE_01_REQUEST = """Create a single Python command-line script that reads a plain text application
log file and counts entries by severity. Each valid line starts with one of
DEBUG, INFO, WARNING, ERROR, or CRITICAL followed by a space and the message.
The script should print a terminal summary with one count per severity, report
how many malformed lines were skipped, handle a missing input file clearly, and
use only the Python standard library."""

GATE_02_REQUEST = """Create a small Python utility that converts JSONL records into CSV. It should
include a command-line script and focused tests. The CLI must accept input and
output paths, an optional list of fields, preserve stable column order, write
blank cells for missing fields, report malformed JSON lines without aborting
the whole conversion, and use only the Python standard library."""

GATE_03_REQUEST = """Create a small Python package for checking local Markdown links. It should
provide a reusable function and a CLI. The checker must scan markdown files
under a directory, collect headings as anchors, report duplicate anchors inside
one file, report broken relative links to local markdown files or anchors, and
include focused tests for anchor generation, duplicate anchors, and broken
relative links."""

GATE_04_REQUEST = """Create a small Python CLI project that summarizes task notes. It should read a
directory of dated text notes, group entries by project name, write a summary
Markdown file, support a simple JSON config file for input directory, output
path, and included projects, include a README explaining the workflow, and
include focused tests for parsing notes, applying config filters, and rendering
the summary."""

GATE_05_REQUEST = """Create a small Python project that reads a CSV inventory of pages, fetches each
listed URL, extracts the HTML title and first h1 heading, merges those values
with the inventory rows, and writes a consolidated CSV report. It should include
a CLI, source modules, mocked HTTP tests, and a README that explains the input
CSV columns and command workflow. The project may use only the Python standard
library."""


async def test_live_gate_01_single_file_log_counter() -> None:
    """Run Gate 01 through the public coding-agent patch proposal interface."""

    response = await _run_e2e_gate(
        gate_id="gate_01",
        request_text=GATE_01_REQUEST,
        session_id="phase2_gate_01_log_counter",
    )

    _assert_reviewable_evidence(
        gate_id="gate_01",
        response=response,
    )


async def test_live_gate_02_jsonl_to_csv_cli_and_tests() -> None:
    """Run Gate 02 through the public coding-agent patch proposal interface."""

    response = await _run_e2e_gate(
        gate_id="gate_02",
        request_text=GATE_02_REQUEST,
        session_id="phase2_gate_02_jsonl_to_csv",
    )

    _assert_reviewable_evidence(
        gate_id="gate_02",
        response=response,
    )


async def test_live_gate_03_markdown_link_checker_package() -> None:
    """Run Gate 03 through the public coding-agent patch proposal interface."""

    response = await _run_e2e_gate(
        gate_id="gate_03",
        request_text=GATE_03_REQUEST,
        session_id="phase2_gate_03_markdown_link_checker",
    )

    _assert_reviewable_evidence(
        gate_id="gate_03",
        response=response,
    )


async def test_live_gate_04_task_notes_cli_project() -> None:
    """Run Gate 04 through the public coding-agent patch proposal interface."""

    response = await _run_e2e_gate(
        gate_id="gate_04",
        request_text=GATE_04_REQUEST,
        session_id="phase2_gate_04_task_notes_cli_project",
    )

    _assert_reviewable_evidence(
        gate_id="gate_04",
        response=response,
    )


async def test_live_gate_05_multi_source_data_tool() -> None:
    """Run Gate 05 through the public coding-agent patch proposal interface."""

    response = await _run_e2e_gate(
        gate_id="gate_05",
        request_text=GATE_05_REQUEST,
        session_id="phase2_gate_05_multi_source_data_tool",
    )

    _assert_reviewable_evidence(
        gate_id="gate_05",
        response=response,
    )


async def _run_e2e_gate(
    *,
    gate_id: str,
    request_text: str,
    session_id: str,
) -> dict[str, Any]:
    """Call the coding-agent E2E entrypoint and persist raw review evidence."""

    response = await propose_code_change({
        "question": request_text,
        "workspace_root": str(WORKSPACE_ROOT / gate_id),
        "session_id": session_id,
        "preferred_language": "English",
        "max_answer_chars": MAX_ANSWER_CHARS,
        "max_artifact_chars": MAX_ARTIFACT_CHARS,
    })

    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    response_path = TRACE_ROOT / f"{gate_id}_response.json"
    response_path.write_text(
        json.dumps(response, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_paths = _materialized_artifact_paths(
        gate_id=gate_id,
        response=response,
    )
    artifact_path = TRACE_ROOT / f"{gate_id}_materialized_artifacts.json"
    artifact_path.write_text(
        json.dumps(artifact_paths, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"response_path={response_path}")
    print(f"materialized_artifacts_path={artifact_path}")
    for path_text in artifact_paths:
        print(f"materialized_artifact={path_text}")

    return response


def _assert_reviewable_evidence(
    *,
    gate_id: str,
    response: dict[str, Any],
) -> None:
    """Assert only that enough evidence exists for AI and human review."""

    assert response
    assert response["trace"]
    assert response["answer_text"]
    assert response["patch_artifacts"]
    assert response["validation"]["files"]
    assert response["validation"]["sandbox_applied"] is True
    assert not response["repository"]
    assert not response["source_scope"]

    materialized_paths = _materialized_artifact_paths(
        gate_id=gate_id,
        response=response,
    )
    assert materialized_paths
    for path_text in materialized_paths:
        assert Path(path_text).is_file()


def _materialized_artifact_paths(
    *,
    gate_id: str,
    response: dict[str, Any],
) -> list[str]:
    """Return direct review-package paths for generated artifact files."""

    validation_root = WORKSPACE_ROOT / gate_id / "writing_validation"
    candidate_roots = [
        path
        for path in validation_root.iterdir()
        if path.is_dir()
    ]
    candidate_roots.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidate_roots:
        return []

    sandbox_root = candidate_roots[0]
    materialized_paths: list[str] = []
    for relative_path in response["validation"]["files"]:
        file_path = sandbox_root / str(relative_path)
        materialized_paths.append(str(file_path))
    return materialized_paths
