"""Primary real-LLM gates for code writing."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


_TEST_NAME = "coding_agent_phase2_live_llm"
_MANIFEST_PATH = (
    Path("test_artifacts")
    / "live_gate"
    / "coding_agent_phase2_challenges.json"
)
_WORKSPACE_ROOT = (
    Path("test_artifacts")
    / "coding_agent_live_workspaces"
    / "phase2_hard_gates"
)


def _load_case(case_id: str) -> dict[str, Any]:
    manifest = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    case = manifest[case_id]
    return case


def _question_with_artifacts(case: dict[str, Any]) -> str:
    question = case["question"]
    artifact_paths = case.get("input_artifacts", [])
    if not artifact_paths:
        return question

    sections = [question, "\nSupplied input artifacts:"]
    for artifact_path in artifact_paths:
        path = Path(artifact_path)
        artifact_text = path.read_text(encoding="utf-8")
        sections.extend([
            f"\n--- {path.as_posix()} ---",
            artifact_text,
        ])
    combined_question = "\n".join(sections)
    return combined_question


def _public_text(result: dict[str, Any]) -> str:
    text = json.dumps(result, ensure_ascii=False, default=str)
    return text


def _assert_sanitized(result: dict[str, Any]) -> None:
    public_text = _public_text(result)
    assert "local_root" not in public_text
    assert "cache_key" not in public_text
    assert re.search(r"(?<![A-Za-z0-9_])\.git(?![A-Za-z0-9_])", public_text) is None
    assert re.search(r"(?<![A-Za-z0-9_])\.env(?![A-Za-z0-9_])", public_text) is None


async def _run_gate(case_id: str) -> None:
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    case = _load_case(case_id)
    request = {
        "question": _question_with_artifacts(case),
        "workspace_root": str(_WORKSPACE_ROOT / case_id),
        "max_artifact_chars": 24000,
        "session_id": case_id,
    }
    source_url = case.get("source_url")
    if source_url:
        request["source_url"] = source_url
    repo_url = case.get("repo_url")
    if repo_url:
        request["repo_url"] = repo_url
    requested_ref = case.get("requested_ref")
    if requested_ref:
        request["requested_ref"] = requested_ref

    response = await propose_code_change(request)
    trace_path = write_llm_trace(
        _TEST_NAME,
        case_id,
        {
            "request": request,
            "response": response,
        },
    )

    print(f"trace_path={trace_path}")
    print(json.dumps(response, ensure_ascii=False, indent=2, default=str))

    assert response["status"] == "succeeded"
    assert response["patch_artifacts"]
    assert response["validation"]["status"] == "succeeded"
    assert response["validation"]["sandbox_applied"] is True
    assert response["trace_summary"]
    _assert_sanitized(response)


async def test_phase2_gate_01() -> None:
    await _run_gate("gate_01")


async def test_phase2_gate_02() -> None:
    await _run_gate("gate_02")


async def test_phase2_gate_03() -> None:
    await _run_gate("gate_03")


async def test_phase2_gate_04() -> None:
    await _run_gate("gate_04")


async def test_phase2_gate_05() -> None:
    await _run_gate("gate_05")
