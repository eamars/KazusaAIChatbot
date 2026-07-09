"""Live LLM full-workflow gates for the coding-agent background entrypoint."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from kazusa_ai_chatbot.action_spec.execution import execute_action_specs_for_trace
from kazusa_ai_chatbot.action_spec.registry import (
    ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
    SPEAK_CAPABILITY,
    build_initial_action_capabilities,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.cognition_chain_core.contracts import LLMStageBinding
from kazusa_ai_chatbot.cognition_chain_core.stages import l2d
from kazusa_ai_chatbot.cognition_chain_core.stages.l2d import (
    select_semantic_actions,
)
from kazusa_ai_chatbot.config import (
    CODING_AGENT_PM_LLM_BASE_URL,
    COGNITION_LLM_BASE_URL,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_cognition_actions as action_connector,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_chain_services,
)
from tests.llm_trace import write_llm_trace
from tests.test_coding_agent_phase3_handoff_e2e import (
    _InMemoryAcceptedCodeWorkStore,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "coding_agent_full_workflow"


async def test_live_gate_01_read_only_question_from_l2d_to_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Read-only source question through L2d, queue, and coding worker."""

    await _skip_if_llm_unavailable()
    source_root = _prepare_fixture_checkout(
        tmp_path,
        "gate_01_cli_command_discovery",
    )
    gate_trace = await _run_live_background_sequence(
        monkeypatch,
        tmp_path,
        case_id="gate_01_read_only_question",
        turns=[
            (
                "In this fixture repo, explain where the CLI discovers "
                f"commands. Use the local source checkout at {source_root}."
            ),
        ],
    )

    turn = gate_trace["turns"][0]
    _assert_worker_succeeded(turn)
    metadata = turn["worker_metadata"]
    assert metadata["worker_operation"] == "start"
    assert metadata["coding_run_ref"]
    assert metadata["evidence_refs"], gate_trace["trace_path"]
    assert metadata["patch_artifacts"] == [], gate_trace["trace_path"]
    assert metadata["apply_attempts"] == [], gate_trace["trace_path"]
    assert metadata["execution_attempts"] == [], gate_trace["trace_path"]
    assert metadata["repair_attempts"] == [], gate_trace["trace_path"]
    _assert_no_private_leaks(gate_trace)


async def test_live_gate_02_source_free_proposal_with_revision_followups(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Source-free proposal should revise and summarize one durable run."""

    await _skip_if_llm_unavailable()
    gate_trace = await _run_live_background_sequence(
        monkeypatch,
        tmp_path,
        case_id="gate_02_source_free_revision",
        turns=[
            (
                "Create a small Python CSV normalizer CLI with tests. It "
                "should trim whitespace from headers and values, sort rows "
                "deterministically, and stay review-only until I approve it."
            ),
            (
                "For {coding_run_ref}, revise the proposal to add a dry-run "
                "mode and make the output deterministic."
            ),
            (
                "For {coding_run_ref}, summarize the files I should review."
            ),
        ],
    )

    start_turn, revision_turn, summary_turn = gate_trace["turns"]
    _assert_worker_succeeded(start_turn)
    _assert_worker_succeeded(revision_turn)
    _assert_worker_succeeded(summary_turn)
    coding_run_ref = _coding_run_ref_from_trace(start_turn)
    assert _coding_run_ref_from_trace(revision_turn) == coding_run_ref
    assert _coding_run_ref_from_trace(summary_turn) == coding_run_ref
    assert revision_turn["worker_metadata"]["worker_operation"] == (
        "revise_proposal"
    )
    assert summary_turn["worker_metadata"]["worker_operation"] == "summarize"
    assert revision_turn["worker_metadata"]["coding_run_status"] == (
        "awaiting_approval"
    )
    assert summary_turn["worker_metadata"]["changed_files"], gate_trace["trace_path"]
    for turn in gate_trace["turns"]:
        metadata = turn["worker_metadata"]
        assert metadata["apply_attempts"] == [], gate_trace["trace_path"]
        assert metadata["execution_attempts"] == [], gate_trace["trace_path"]
        assert metadata["repair_attempts"] == [], gate_trace["trace_path"]
    _assert_no_private_leaks(gate_trace)


async def test_live_gate_03_existing_source_proposal_with_runtime_only_followups(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing-source proposal should revise to keep tests unchanged."""

    await _skip_if_llm_unavailable()
    source_root = _prepare_fixture_checkout(tmp_path, "gate_03_counter_cli_json")
    gate_trace = await _run_live_background_sequence(
        monkeypatch,
        tmp_path,
        case_id="gate_03_existing_source_runtime_only",
        turns=[
            (
                "In this fixture repo, add JSON output to the counter CLI. "
                f"Use the local source checkout at {source_root}."
            ),
            (
                "For {coding_run_ref}, keep tests unchanged; revise the "
                "proposal so it changes only runtime source files."
            ),
            (
                "For {coding_run_ref}, summarize the exact files changed and "
                "why."
            ),
        ],
    )

    start_turn, revision_turn, summary_turn = gate_trace["turns"]
    _assert_worker_succeeded(start_turn)
    _assert_worker_succeeded(revision_turn)
    _assert_worker_succeeded(summary_turn)
    coding_run_ref = _coding_run_ref_from_trace(start_turn)
    assert _coding_run_ref_from_trace(revision_turn) == coding_run_ref
    assert _coding_run_ref_from_trace(summary_turn) == coding_run_ref
    assert revision_turn["worker_metadata"]["worker_operation"] == (
        "revise_proposal"
    )
    changed_files = _changed_file_paths(summary_turn)
    assert changed_files, gate_trace["trace_path"]
    assert all(
        not path.startswith("tests/")
        for path in changed_files
    ), gate_trace["trace_path"]
    _assert_no_private_leaks(gate_trace)


async def test_live_gate_04_approval_verify_and_repair_followups(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Approval should run focused verification and expose attempt history."""

    await _skip_if_llm_unavailable()
    source_root = _prepare_fixture_checkout(
        tmp_path,
        "gate_04_slug_normalization",
    )
    gate_trace = await _run_live_background_sequence(
        monkeypatch,
        tmp_path,
        case_id="gate_04_approval_verify_repair",
        turns=[
            (
                "Fix the slug normalization bug in this fixture repo. "
                f"Use the local source checkout at {source_root}."
            ),
            (
                "I approve {coding_run_ref}. Run the focused slug pytest "
                "tests/test_slug.py. If it fails, repair source without "
                "editing tests."
            ),
            (
                "For {coding_run_ref}, summarize the verification attempts "
                "and final changed files."
            ),
        ],
    )

    start_turn, approval_turn, summary_turn = gate_trace["turns"]
    _assert_worker_succeeded(start_turn)
    _assert_worker_succeeded(approval_turn)
    _assert_worker_succeeded(summary_turn)
    coding_run_ref = _coding_run_ref_from_trace(start_turn)
    assert _coding_run_ref_from_trace(approval_turn) == coding_run_ref
    assert _coding_run_ref_from_trace(summary_turn) == coding_run_ref
    assert approval_turn["worker_metadata"]["worker_operation"] == (
        "approve_and_verify"
    )
    assert approval_turn["worker_metadata"]["apply_attempts"], (
        gate_trace["trace_path"]
    )
    assert approval_turn["worker_metadata"]["execution_attempts"], (
        gate_trace["trace_path"]
    )
    changed_files = _changed_file_paths(summary_turn)
    assert changed_files, gate_trace["trace_path"]
    assert all(
        not path.startswith("tests/")
        for path in changed_files
    ), gate_trace["trace_path"]
    _assert_no_private_leaks(gate_trace)


async def test_live_gate_05_hard_multifile_approval_history_cancel_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Hard multi-file workflow should preserve durable run state."""

    await _skip_if_llm_unavailable()
    source_root = _prepare_fixture_checkout(
        tmp_path,
        "gate_05_release_feed_cache_cli",
    )
    gate_trace = await _run_live_background_sequence(
        monkeypatch,
        tmp_path,
        case_id="gate_05_hard_multifile_status",
        turns=[
            (
                "Fix the release feed cache timeout and CLI flag behavior in "
                f"this repo. Use the local source checkout at {source_root}."
            ),
            (
                "I approve {coding_run_ref}. Run the focused release feed "
                "tests tests/test_cache.py and tests/test_cli.py. Repair "
                "source if those checks fail, but keep protected tests "
                "unchanged."
            ),
            (
                "For {coding_run_ref}, show the attempt history and final "
                "changed source files."
            ),
            (
                "For {coding_run_ref}, cancel any remaining work if protected "
                "tests would need edits."
            ),
            "For {coding_run_ref}, give me the final status.",
        ],
    )

    start_turn, approval_turn, summary_turn, cancel_turn, final_turn = (
        gate_trace["turns"]
    )
    _assert_worker_succeeded(start_turn)
    _assert_worker_succeeded(approval_turn)
    _assert_worker_succeeded(summary_turn)
    _assert_worker_finished(cancel_turn)
    _assert_worker_succeeded(final_turn)
    coding_run_ref = _coding_run_ref_from_trace(start_turn)
    for turn in gate_trace["turns"][1:]:
        assert _coding_run_ref_from_trace(turn) == coding_run_ref
    assert approval_turn["worker_metadata"]["apply_attempts"], (
        gate_trace["trace_path"]
    )
    assert approval_turn["worker_metadata"]["execution_attempts"], (
        gate_trace["trace_path"]
    )
    changed_files = _changed_file_paths(summary_turn)
    assert changed_files, gate_trace["trace_path"]
    assert all(
        not path.startswith("tests/")
        for path in changed_files
    ), gate_trace["trace_path"]
    final_status = final_turn["worker_metadata"]["coding_run_status"]
    assert final_status in (
        "completed",
        "failed",
        "blocked",
        "cancelled",
    ), gate_trace["trace_path"]
    if cancel_turn["background_job"]["status"] == "completed":
        cancel_status = cancel_turn["worker_metadata"]["coding_run_status"]
        assert cancel_status in ("cancelled", final_status), gate_trace["trace_path"]
    _assert_no_private_leaks(gate_trace)


async def test_live_gate_06_mixed_create_and_existing_edit_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real feature often needs a new helper module wired into old code."""

    await _skip_if_llm_unavailable()
    source_root = _prepare_fixture_checkout(tmp_path, "gate_03_counter_cli_json")
    gate_trace = await _run_live_background_sequence(
        monkeypatch,
        tmp_path,
        case_id="gate_06_mixed_create_existing_edit",
        turns=[
            (
                "In this fixture repo, add a new module "
                "counter_cli/formatters.py for output formatting, then wire "
                "counter_cli/cli.py to use it for the existing text output. "
                f"Use the local source checkout at {source_root}."
            ),
            (
                "For {coding_run_ref}, summarize the exact created and "
                "modified source files."
            ),
        ],
    )

    start_turn, summary_turn = gate_trace["turns"]
    _assert_worker_succeeded(start_turn)
    _assert_worker_succeeded(summary_turn)
    changed_files = _changed_file_paths(summary_turn)
    assert "counter_cli/formatters.py" in changed_files, gate_trace["trace_path"]
    assert "counter_cli/cli.py" in changed_files, gate_trace["trace_path"]
    _assert_no_private_leaks(gate_trace)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known assessment gap: patch proposals are delivered before managed "
        "preflight apply and execution evidence exists."
    ),
)
async def test_live_gate_07_proposal_has_preapproval_preflight_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A proposal should be mechanically exercised before user approval."""

    await _skip_if_llm_unavailable()
    source_root = _prepare_fixture_checkout(
        tmp_path,
        "gate_04_slug_normalization",
    )
    gate_trace = await _run_live_background_sequence(
        monkeypatch,
        tmp_path,
        case_id="gate_07_preapproval_preflight",
        turns=[
            (
                "Fix the slug normalization bug in this fixture repo. "
                "Before asking for approval, preflight the proposed patch in "
                "a managed copy with the focused slug tests. Use the local "
                f"source checkout at {source_root}."
            ),
        ],
    )

    start_turn = gate_trace["turns"][0]
    _assert_worker_succeeded(start_turn)
    metadata = start_turn["worker_metadata"]
    assert metadata["worker_operation"] == "start"
    assert metadata["coding_run_status"] == "awaiting_approval"
    assert metadata["apply_attempts"], gate_trace["trace_path"]
    assert metadata["execution_attempts"], gate_trace["trace_path"]
    assert "pytest" in _execution_tools(start_turn), gate_trace["trace_path"]
    _assert_no_private_leaks(gate_trace)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known assessment gap: approval checks are planned from approval "
        "prose instead of deterministically from changed files and repo tests."
    ),
)
async def test_live_gate_08_vague_approval_runs_changed_file_tests(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Vague approval should still run the focused tests for touched code."""

    await _skip_if_llm_unavailable()
    source_root = _prepare_fixture_checkout(
        tmp_path,
        "gate_04_slug_normalization",
    )
    gate_trace = await _run_live_background_sequence(
        monkeypatch,
        tmp_path,
        case_id="gate_08_changed_file_execution_derivation",
        turns=[
            (
                "Fix the slug normalization bug in this fixture repo. "
                f"Use the local source checkout at {source_root}."
            ),
            (
                "I approve {coding_run_ref}. Run the right focused "
                "verification for the changed behavior, then repair source "
                "if it fails."
            ),
        ],
    )

    start_turn, approval_turn = gate_trace["turns"]
    _assert_worker_succeeded(start_turn)
    _assert_worker_succeeded(approval_turn)
    assert "pytest" in _execution_tools(approval_turn), gate_trace["trace_path"]
    assert "tests/test_slug.py" in _executed_paths(approval_turn), (
        gate_trace["trace_path"]
    )
    _assert_no_private_leaks(gate_trace)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known assessment gap: verifier failures do not produce typed "
        "environment dependency blockers."
    ),
)
async def test_live_gate_09_missing_dependency_becomes_typed_blocker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing project dependencies should block instead of looping repairs."""

    await _skip_if_llm_unavailable()
    source_root = _prepare_fixture_checkout(tmp_path, "gate_09_missing_dependency")
    gate_trace = await _run_live_background_sequence(
        monkeypatch,
        tmp_path,
        case_id="gate_09_dependency_blocker",
        turns=[
            (
                "Fix YAML config loading in this fixture repo. The protected "
                "tests express the required external package. Use the local "
                f"source checkout at {source_root}."
            ),
            (
                "I approve {coding_run_ref}. Run pytest "
                "tests/test_yaml_dependency.py. If the external dependency "
                "is missing, return a typed environment blocker and do not "
                "edit protected tests."
            ),
        ],
    )

    start_turn, approval_turn = gate_trace["turns"]
    _assert_worker_succeeded(start_turn)
    _assert_worker_finished(approval_turn)
    metadata = approval_turn["worker_metadata"]
    assert metadata["coding_run_status"] == "blocked", gate_trace["trace_path"]
    assert _has_blocker_code(
        approval_turn,
        "environment_dependency_missing",
    ), gate_trace["trace_path"]
    assert metadata["repair_attempts"] == [], gate_trace["trace_path"]
    _assert_no_private_leaks(gate_trace)


async def test_live_gate_10_source_free_proposal_records_alignment_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Source-free proposals should pass a semantic alignment gate."""

    await _skip_if_llm_unavailable()
    gate_trace = await _run_live_background_sequence(
        monkeypatch,
        tmp_path,
        case_id="gate_10_source_free_alignment",
        turns=[
            (
                "Create a Python TOML settings linter CLI with tests. It "
                "must read TOML, report duplicate logical keys, exit nonzero "
                "on invalid settings, and stay review-only until approval."
            ),
        ],
    )

    start_turn = gate_trace["turns"][0]
    _assert_worker_succeeded(start_turn)
    trace_summary = _metadata_trace_summary(start_turn)
    assert any(
        row.startswith("writing_alignment:status=pass")
        for row in trace_summary
    ), gate_trace["trace_path"]
    _assert_no_private_leaks(gate_trace)


async def _run_live_background_sequence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    case_id: str,
    turns: list[str],
) -> dict[str, Any]:
    """Run a multi-turn coding workflow through background work."""

    rendered_turns: list[dict[str, Any]] = []
    coding_run_ref = ""
    coding_workspace = tmp_path / case_id / "coding-workspace"
    for turn_index, turn_template in enumerate(turns, start=1):
        user_request = turn_template.format(coding_run_ref=coding_run_ref)
        turn_trace = await _run_live_background_turn(
            monkeypatch,
            tmp_path,
            case_id=f"{case_id}_turn_{turn_index}",
            user_request=user_request,
            coding_workspace=coding_workspace,
        )
        rendered_turns.append(turn_trace)
        next_ref = _optional_coding_run_ref_from_trace(turn_trace)
        if next_ref:
            coding_run_ref = next_ref

    gate_trace = {
        "case_id": case_id,
        "fixture_root": str(FIXTURE_ROOT),
        "turns": rendered_turns,
    }
    trace_path = write_llm_trace(
        "coding_agent_full_workflow_integration_live_llm",
        case_id,
        gate_trace,
    )
    gate_trace["trace_path"] = str(trace_path)
    return gate_trace


async def _run_live_background_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    case_id: str,
    user_request: str,
    coding_workspace: Path,
) -> dict[str, Any]:
    """Run one live L2d coding action through queue and worker tick."""

    del tmp_path
    from kazusa_ai_chatbot.background_work import worker as background_worker
    from kazusa_ai_chatbot.background_work.subagent import (
        coding_agent as coding_worker,
    )

    store = _InMemoryAcceptedCodeWorkStore()
    _install_in_memory_persistence(
        monkeypatch,
        store=store,
    )
    monkeypatch.setattr(
        coding_worker,
        "CODING_AGENT_WORKSPACE_ROOT",
        str(coding_workspace),
    )

    state = _l2d_state(user_request)
    services = build_cognition_chain_services()
    capturing_llm = _CapturingLLM(services.llm)
    token = l2d.set_action_selection_llm(
        LLMStageBinding(capturing_llm, services.action_selection_config)
    )
    try:
        l2d_result = await select_semantic_actions(state)
    finally:
        l2d.reset_action_selection_llm(token)

    action_specs = action_connector.materialize_semantic_action_requests(
        l2d_result.get("semantic_action_requests", []),
        state,
    )
    coding_specs = [
        spec for spec in action_specs
        if spec.get("kind") == ACCEPTED_CODING_TASK_REQUEST_CAPABILITY
    ]
    assert coding_specs, (
        "Live L2d did not produce accepted_coding_task_request. "
        f"raw={capturing_llm.raw_output}"
    )

    queue_results = await execute_action_specs_for_trace(
        coding_specs,
        storage_timestamp_utc=state["storage_timestamp_utc"],
    )
    worker_tick = await background_worker.run_background_work_worker_tick(
        claim_limit=1,
        lease_seconds=60,
        max_attempts=3,
        worker_id=f"{case_id}-{uuid4().hex}",
    )
    background_job = dict(store.background_job or {})
    trace = {
        "case_id": case_id,
        "user_request": user_request,
        "raw_l2d_output": capturing_llm.raw_output,
        "semantic_action_requests": l2d_result.get(
            "semantic_action_requests",
            [],
        ),
        "materialized_action_specs": action_specs,
        "queue_results": queue_results,
        "worker_tick": worker_tick,
        "accepted_task": dict(store.accepted_task or {}),
        "background_job": background_job,
        "worker_metadata": dict(background_job.get("worker_metadata", {})),
    }
    trace_path = write_llm_trace(
        "coding_agent_full_workflow_integration_live_llm",
        case_id,
        trace,
    )
    trace["trace_path"] = str(trace_path)
    assert store.background_job is not None, trace["trace_path"]
    return trace


def _install_in_memory_persistence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    store: _InMemoryAcceptedCodeWorkStore,
) -> None:
    """Patch durable stores while preserving L2d and coding-agent LLM calls."""

    from kazusa_ai_chatbot.accepted_task import lifecycle as accepted_lifecycle
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
        store.mark_accepted_task_result_ready,
    )
    monkeypatch.setattr(
        accepted_lifecycle,
        "repository_mark_failure_ready",
        store.mark_accepted_task_failure_ready,
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


def _l2d_state(user_request: str) -> dict[str, object]:
    """Build a live L2d state focused on coding-agent actions."""

    action_affordances = [
        row for row in project_prompt_affordances(
            build_initial_action_capabilities(),
        )
        if row["capability"] in (
            ACCEPTED_CODING_TASK_REQUEST_CAPABILITY,
            SPEAK_CAPABILITY,
        )
    ]
    return {
        "storage_timestamp_utc": "2026-07-09T01:00:00+00:00",
        "cognitive_episode": {
            "trigger_source": "user_message",
            "input_sources": ["dialog_text"],
            "output_mode": "visible_reply",
            "target_scope": {
                "platform": "debug",
                "platform_channel_id": "debug:user:coding-live",
                "channel_type": "private",
                "current_global_user_id": "global-user-001",
                "current_platform_user_id": "debug-user-001",
                "current_display_name": "Live Coding User",
            },
            "origin_metadata": {
                "platform_message_id": f"message-{uuid4().hex}",
                "platform_bot_id": "debug-bot-001",
            },
        },
        "platform": "debug",
        "platform_channel_id": "debug:user:coding-live",
        "channel_type": "private",
        "platform_message_id": f"message-{uuid4().hex}",
        "platform_bot_id": "debug-bot-001",
        "global_user_id": "global-user-001",
        "platform_user_id": "debug-user-001",
        "user_name": "Live Coding User",
        "character_profile": {"name": "Kazusa"},
        "character_name": "Kazusa",
        "decontexualized_input": user_request,
        "media_summary": "",
        "logical_stance": "ACCEPT",
        "character_intent": "HELP_WITH_CODING_TASK",
        "judgment_note": (
            "The user is asking for bounded accepted coding-agent work."
        ),
        "internal_monologue": (
            "Use the accepted coding task action for durable background work."
        ),
        "emotional_appraisal": "focused",
        "interaction_subtext": "direct coding request",
        "boundary_core_assessment": {},
        "social_distance": "friendly",
        "emotional_intensity": "low",
        "vibe_check": "work-focused",
        "relational_dynamic": "cooperative",
        "rag_result": {},
        "conversation_progress": {},
        "resolver_context": "",
        "available_action_affordances": action_affordances,
        "background_work_output_char_limit": 6000,
        "max_action_requests": 2,
        "max_resolver_requests": 1,
    }


def _prepare_fixture_checkout(tmp_path: Path, fixture_name: str) -> Path:
    """Copy one committed fixture into a git-backed temporary checkout."""

    fixture_source = FIXTURE_ROOT / fixture_name
    assert fixture_source.exists(), str(fixture_source)
    checkout_root = tmp_path / "fixture-checkouts" / fixture_name
    if checkout_root.exists():
        shutil.rmtree(checkout_root)
    shutil.copytree(fixture_source, checkout_root)
    _run_git(["init"], checkout_root)
    _run_git(["config", "user.email", "test@example.com"], checkout_root)
    _run_git(["config", "user.name", "Test User"], checkout_root)
    _run_git(["add", "."], checkout_root)
    _run_git(["commit", "-m", "initial fixture"], checkout_root)
    _run_git(
        [
            "remote",
            "add",
            "origin",
            f"https://github.com/fixture/{fixture_name}.git",
        ],
        checkout_root,
    )
    return checkout_root


def _run_git(args: list[str], cwd: Path) -> str:
    """Run git for local live-test fixture setup."""

    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _assert_worker_succeeded(trace: dict[str, Any]) -> None:
    """Assert successful worker completion and expose trace path on failure."""

    _assert_worker_finished(trace)
    assert trace["worker_tick"]["succeeded_count"] == 1, trace["trace_path"]
    assert trace["background_job"]["status"] == "completed", trace["trace_path"]


def _assert_worker_finished(trace: dict[str, Any]) -> None:
    """Assert one worker processed the job."""

    assert trace["worker_tick"]["processed_count"] == 1, trace["trace_path"]
    job = trace["background_job"]
    assert job["worker"] == "coding_agent", trace["trace_path"]
    assert job["status"] in ("completed", "failed"), trace["trace_path"]
    assert trace["worker_metadata"]["schema_version"] == (
        "coding_agent_worker_metadata.v2"
    )


def _optional_coding_run_ref_from_trace(trace: dict[str, Any]) -> str:
    """Return the prompt-safe coding run ref when one is present."""

    metadata = trace.get("worker_metadata", {})
    if not isinstance(metadata, dict):
        return ""
    coding_run_ref = str(metadata.get("coding_run_ref", ""))
    if coding_run_ref.startswith("coding_run:"):
        return coding_run_ref
    return ""


def _coding_run_ref_from_trace(trace: dict[str, Any]) -> str:
    """Return the prompt-safe coding run ref from a worker trace."""

    coding_run_ref = _optional_coding_run_ref_from_trace(trace)
    assert coding_run_ref.startswith("coding_run:"), trace["trace_path"]
    return coding_run_ref


def _changed_file_paths(trace: dict[str, Any]) -> list[str]:
    """Return changed file paths from worker metadata."""

    changed_files = trace["worker_metadata"].get("changed_files", [])
    if not isinstance(changed_files, list):
        return []
    paths = [
        str(row.get("path", "")).replace("\\", "/")
        for row in changed_files
        if isinstance(row, dict) and row.get("path")
    ]
    return paths


def _execution_tools(trace: dict[str, Any]) -> list[str]:
    """Return execution tool names from worker metadata."""

    attempts = trace["worker_metadata"].get("execution_attempts", [])
    if not isinstance(attempts, list):
        return []
    return [
        str(row.get("tool", ""))
        for row in attempts
        if isinstance(row, dict) and row.get("tool")
    ]


def _executed_paths(trace: dict[str, Any]) -> list[str]:
    """Return executed paths from worker metadata execution attempts."""

    attempts = trace["worker_metadata"].get("execution_attempts", [])
    if not isinstance(attempts, list):
        return []
    paths: list[str] = []
    for row in attempts:
        if not isinstance(row, dict):
            continue
        executed = row.get("executed_paths")
        if not isinstance(executed, list):
            continue
        paths.extend(
            str(path).replace("\\", "/")
            for path in executed
            if isinstance(path, str)
        )
    return paths


def _has_blocker_code(trace: dict[str, Any], code: str) -> bool:
    """Return whether worker metadata exposes a specific blocker code."""

    blockers = trace["worker_metadata"].get("blockers", [])
    if not isinstance(blockers, list):
        return False
    return any(
        isinstance(row, dict) and row.get("code") == code
        for row in blockers
    )


def _metadata_trace_summary(trace: dict[str, Any]) -> list[str]:
    """Return worker metadata trace rows."""

    rows = trace["worker_metadata"].get("trace_summary", [])
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, str)
    ]


def _assert_no_private_leaks(gate_trace: dict[str, Any]) -> None:
    """Assert public trace projections omit private roots and raw command text."""

    public_rows: list[dict[str, object]] = []
    for turn in gate_trace["turns"]:
        job = turn["background_job"]
        public_rows.append({
            "artifact_text": job.get("artifact_text"),
            "result_summary": job.get("result_summary"),
            "worker_metadata": turn.get("worker_metadata"),
        })
    serialized = json.dumps(public_rows, ensure_ascii=False)
    for forbidden in (
        "workspace_root",
        "local_root",
        "source_root",
        "stdout_excerpt",
        "stderr_excerpt",
        ".env",
        ".git/",
        ".git\\",
    ):
        assert forbidden not in serialized, gate_trace["trace_path"]


class _CapturingLLM:
    """Capture raw L2d output while using the configured real LLM."""

    def __init__(self, llm: object) -> None:
        self._llm = llm
        self.raw_output = ""

    async def ainvoke(self, messages: list[object], *, config) -> object:
        response = await self._llm.ainvoke(messages, config=config)
        raw_content = getattr(response, "content", "")
        if isinstance(raw_content, str):
            self.raw_output = raw_content
        return response


async def _skip_if_llm_unavailable() -> None:
    """Skip when configured cognition or coding LLM endpoints are unavailable."""

    for label, base_url in (
        ("cognition", COGNITION_LLM_BASE_URL),
        ("coding_agent_pm", CODING_AGENT_PM_LLM_BASE_URL),
    ):
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{base_url.rstrip('/')}/models")
        except httpx.HTTPError as exc:
            pytest.skip(f"{label} LLM endpoint unavailable: {base_url}; {exc}")
        if response.status_code >= 500:
            pytest.skip(
                f"{label} LLM endpoint returned {response.status_code}: "
                f"{base_url}"
            )
