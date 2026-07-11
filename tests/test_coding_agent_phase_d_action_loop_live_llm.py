"""Individually inspected live-controller cases for the coding action loop."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


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
