"""Individually inspected live-controller cases for the coding action loop."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _live_source_root(root: Path) -> Path:
    """Create a minimal committed source checkout for one live graph."""

    source_root = root / "source"
    source_root.mkdir()
    (source_root / "runtime.py").write_text(
        "VALUE = 1\n\n\ndef add_one(value: int) -> int:\n    return value + 1\n",
        encoding="utf-8",
    )
    (source_root / "tests").mkdir()
    (source_root / "tests" / "test_runtime.py").write_text(
        "from runtime import add_one\n\n\ndef test_add_one():\n    assert add_one(1) == 2\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    subprocess.run(["git", "add", "."], cwd=source_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Stage5 Live",
            "-c",
            "user.email=stage5-live@example.invalid",
            "commit",
            "-qm",
            "live fixture",
        ],
        cwd=source_root,
        check=True,
    )
    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            "https://github.com/fixture/stage5-live.git",
        ],
        cwd=source_root,
        check=True,
    )
    return source_root


def _record_live_graph(
    *,
    case_id: str,
    workspace_root: Path,
    response: dict[str, object],
) -> None:
    """Persist raw response, parsed state, terminal projection, and judgment path."""

    run_id = str(response.get("run_id", ""))
    run_root = workspace_root / "coding_runs" / run_id
    action_root = run_root / "action_loop"
    state_path = action_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    artifact_root = Path("test_artifacts/llm_traces")
    artifact_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = artifact_root / f"{case_id}_{timestamp}.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "coding_action_loop_live_graph.v1",
                "case_id": case_id,
                "recorded_at": timestamp,
                "route": "action_loop_v1",
                "raw_response": response,
                "parsed_state_path": str(state_path),
                "parsed_state": state,
                "terminal_projection": {
                    "status": response.get("status"),
                    "answer_text": response.get("answer_text"),
                    "blocker": response.get("blocker"),
                },
                "judgment_path": "manual review required",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    assert state["observations"]
    assert state["status"] in {"completed", "blocked", "awaiting_approval"}


async def _run_live_graph(
    *,
    tmp_path: Path,
    case_id: str,
    question: str,
    objective_type: str,
) -> tuple[Path, dict[str, object]]:
    """Run one complete live action-loop start and retain its durable state."""

    from kazusa_ai_chatbot.coding_agent.coding_run.evaluation import (
        run_evaluation_coding_run,
    )

    source_root = _live_source_root(tmp_path)
    workspace_root = tmp_path / "workspace"
    response = await run_evaluation_coding_run(
        {
            "workspace_root": str(workspace_root),
            "question": question,
            "objective_type": objective_type,
            "local_root_hint": str(source_root),
            "source_scope_hint": "repository",
        },
        engine_id="action_loop_v1",
    )
    _record_live_graph(
        case_id=case_id,
        workspace_root=workspace_root,
        response=response,
    )
    return workspace_root, response


@pytest.mark.asyncio
@pytest.mark.live_llm
async def test_live_read_only_graph_persists_grounded_finish(tmp_path: Path) -> None:
    """Exercise live read/search evidence through the durable terminal graph."""

    _workspace_root, response = await _run_live_graph(
        tmp_path=tmp_path,
        case_id="test_live_read_only_graph_persists_grounded_finish",
        question="Find add_one in the repository and explain its current behavior.",
        objective_type="read_only",
    )
    assert response["status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.live_llm
async def test_live_propose_patch_graph_materializes_review(tmp_path: Path) -> None:
    """Exercise live edit evidence through proposal materialization."""

    _workspace_root, response = await _run_live_graph(
        tmp_path=tmp_path,
        case_id="test_live_propose_patch_graph_materializes_review",
        question="Propose a minimal change that makes add_one add two instead of one.",
        objective_type="propose_patch",
    )
    assert response["status"] in {"awaiting_approval", "completed"}
    assert response.get("patch_artifacts") is not None


@pytest.mark.asyncio
@pytest.mark.live_llm
async def test_live_verify_repair_graph_uses_current_effect(tmp_path: Path) -> None:
    """Exercise live verification evidence and current-effect projection."""

    _workspace_root, response = await _run_live_graph(
        tmp_path=tmp_path,
        case_id="test_live_verify_repair_graph_uses_current_effect",
        question="Repair add_one so the focused test passes, then verify the current candidate.",
        objective_type="verify_repair",
    )
    assert response["status"] in {"awaiting_approval", "completed", "blocked"}


@pytest.mark.asyncio
@pytest.mark.live_llm
async def test_controller_live_chooses_targeted_repository_search() -> None:
    """Ask the configured controller to choose the first grounded action."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.context import (
        render_controller_context,
    )
    from kazusa_ai_chatbot.coding_agent.code_action_loop.supervisor import (
        invoke_controller,
    )

    context = render_controller_context(
        goal="Find where the release feed cache timeout is implemented.",
        acceptance_criteria=["Identify the relevant source owner."],
        capabilities=["read", "search", "note", "finish", "block"],
        source_identity_digest="live-fixture-source",
        candidate_revision=0,
        changed_paths=[],
        current_failure=None,
        working_notes="",
        observations=[],
    )
    result = await invoke_controller(
        context=context,
        allowed_actions={"read", "search", "note", "finish", "block"},
    )
    artifact_root = Path("test_artifacts/llm_traces")
    artifact_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = artifact_root / (
        f"coding_agent_phase_d_controller_targeted_search_{timestamp}.json"
    )
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "coding_action_loop_live_trace.v1",
                "case_id": "controller_targeted_search",
                "recorded_at": timestamp,
                "input_context": json.loads(context),
                "controller_result": result,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert result["status"] == "ok"
    assert result["action"]["action"] == "search"
    assert result["action"]["args"]["mode"] in {
        "literal",
        "path",
        "regex",
        "symbol",
    }


@pytest.mark.asyncio
@pytest.mark.live_llm
async def test_controller_live_changes_request_after_empty_path_search() -> None:
    """Inspect a generic no-progress recovery choice without benchmark fixtures."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.context import (
        render_controller_context,
    )
    from kazusa_ai_chatbot.coding_agent.code_action_loop.supervisor import (
        invoke_controller,
    )

    prior_request = {
        "action": "search",
        "args": {"mode": "path", "query": "orchard*"},
    }
    context = render_controller_context(
        goal="Locate the lighthouse telemetry parser.",
        acceptance_criteria=["Identify the source owner."],
        capabilities=["read", "search", "note", "finish", "block"],
        source_identity_digest="live-no-progress-fixture",
        candidate_revision=0,
        changed_paths=[],
        current_failure=None,
        working_notes="The previous path search returned no evidence.",
        observations=[{
            "outcome": "ok",
            "kind": "search_result",
            "evidence": [],
            "request": prior_request,
        }],
    )
    result = await invoke_controller(
        context=context,
        allowed_actions={"read", "search", "note", "finish", "block"},
    )
    artifact_root = Path("test_artifacts/llm_traces")
    artifact_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = artifact_root / (
        f"coding_agent_phase_d_controller_no_progress_{timestamp}.json"
    )
    artifact_path.write_text(
        json.dumps(
            {
                "schema_version": "coding_action_loop_live_trace.v1",
                "case_id": "controller_no_progress_recovery",
                "recorded_at": timestamp,
                "input_context": json.loads(context),
                "controller_result": result,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    assert result["status"] == "ok"
    action = result["action"]
    if action["action"] == "search":
        assert action["args"] != prior_request["args"]
    else:
        assert action["action"] in {"finish", "block", "note"}
