"""Live LLM full-workflow gates for the coding-agent background entrypoint."""

from __future__ import annotations

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


async def test_live_gate_01_read_only_question_from_l2d_to_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Simple read-only task through L2d, accepted task, and worker tick."""

    await _skip_if_llm_unavailable()
    source_root = _simple_source_tree(tmp_path, "gate01")
    user_request = (
        "Use the local source checkout at "
        f"{source_root} and explain what normalize_name does. "
        "This is a coding-agent task; start a durable coding run."
    )

    trace = await _run_live_background_turn(
        monkeypatch,
        tmp_path,
        case_id="gate_01_read_only",
        user_request=user_request,
    )

    _assert_worker_succeeded(trace)
    assert trace["worker_metadata"]["coding_run_status"] in (
        "completed",
        "blocked",
        "awaiting_approval",
    )
    assert trace["worker_metadata"]["coding_run_ref"]


async def test_live_gate_02_source_free_proposal_from_l2d_to_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Source-free artifact request should create a durable proposal run."""

    await _skip_if_llm_unavailable()
    user_request = (
        "Create a small Python CLI script that reads newline-delimited names "
        "from a file and prints a sorted unique list. Use only the standard "
        "library. Start this as a durable coding run."
    )

    trace = await _run_live_background_turn(
        monkeypatch,
        tmp_path,
        case_id="gate_02_source_free_proposal",
        user_request=user_request,
    )

    _assert_worker_succeeded(trace)
    assert trace["worker_metadata"]["coding_run_status"] in (
        "awaiting_approval",
        "completed",
        "blocked",
    )
    assert trace["worker_metadata"]["coding_run_ref"]


async def test_live_gate_03_existing_source_proposal_then_status_followup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Follow-up status should bind to the prompt-safe coding run ref."""

    await _skip_if_llm_unavailable()
    source_root = _simple_source_tree(tmp_path, "gate03")
    start_trace = await _run_live_background_turn(
        monkeypatch,
        tmp_path,
        case_id="gate_03_start",
        user_request=(
            "Use the local source checkout at "
            f"{source_root}. Modify normalize_name so it collapses internal "
            "whitespace to a single space and update focused tests. Start a "
            "durable coding run."
        ),
    )
    coding_run_ref = _coding_run_ref_from_trace(start_trace)

    status_trace = await _run_live_background_turn(
        monkeypatch,
        tmp_path,
        case_id="gate_03_status",
        user_request=(
            f"Check status for coding_run_ref={coding_run_ref}. "
            "Use accepted_coding_task_request with decision=status."
        ),
    )

    _assert_worker_succeeded(status_trace)
    assert status_trace["worker_metadata"]["coding_run_ref"] == coding_run_ref


async def test_live_gate_04_approval_verify_followup_from_l2d_to_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Approval follow-up should continue the referenced run with safe checks."""

    await _skip_if_llm_unavailable()
    source_root = _simple_source_tree(tmp_path, "gate04")
    start_trace = await _run_live_background_turn(
        monkeypatch,
        tmp_path,
        case_id="gate_04_start",
        user_request=(
            "Use the local source checkout at "
            f"{source_root}. Modify normalize_name so it collapses internal "
            "whitespace to a single space and update focused tests. Start a "
            "durable coding run."
        ),
    )
    coding_run_ref = _coding_run_ref_from_trace(start_trace)

    approval_trace = await _run_live_background_turn(
        monkeypatch,
        tmp_path,
        case_id="gate_04_approve_verify",
        user_request=(
            f"I approve coding_run_ref={coding_run_ref}. Run focused pytest "
            "for tests/test_names.py. Use accepted_coding_task_request with "
            "decision=approve_and_verify."
        ),
    )

    _assert_worker_succeeded(approval_trace)
    assert approval_trace["worker_metadata"]["coding_run_ref"] == coding_run_ref
    assert approval_trace["worker_metadata"]["worker_operation"] == (
        "approve_and_verify"
    )


async def test_live_gate_05_cancel_followup_from_l2d_to_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancellation follow-up should persist through the background entrypoint."""

    await _skip_if_llm_unavailable()
    source_root = _simple_source_tree(tmp_path, "gate05")
    start_trace = await _run_live_background_turn(
        monkeypatch,
        tmp_path,
        case_id="gate_05_start",
        user_request=(
            "Use the local source checkout at "
            f"{source_root}. Propose a small refactor to normalize_name. "
            "Start a durable coding run."
        ),
    )
    coding_run_ref = _coding_run_ref_from_trace(start_trace)

    cancel_trace = await _run_live_background_turn(
        monkeypatch,
        tmp_path,
        case_id="gate_05_cancel",
        user_request=(
            f"Cancel coding_run_ref={coding_run_ref}. Use "
            "accepted_coding_task_request with decision=cancel."
        ),
    )

    _assert_worker_succeeded(cancel_trace)
    assert cancel_trace["worker_metadata"]["coding_run_ref"] == coding_run_ref
    assert cancel_trace["worker_metadata"]["coding_run_status"] == "cancelled"


async def _run_live_background_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    case_id: str,
    user_request: str,
) -> dict[str, Any]:
    """Run one live L2d coding action through queue and worker tick."""

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
        str(tmp_path / "coding-workspace"),
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


def _simple_source_tree(tmp_path: Path, name: str) -> Path:
    """Create a small local source tree for live coding-agent gates."""

    source_root = tmp_path / name / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    (source_root / "names.py").write_text(
        "def normalize_name(value):\n"
        "    return value.strip()\n",
        encoding="utf-8",
    )
    tests_root = source_root / "tests"
    tests_root.mkdir(exist_ok=True)
    (tests_root / "test_names.py").write_text(
        "from names import normalize_name\n\n\n"
        "def test_normalize_name_strips_outer_whitespace():\n"
        "    assert normalize_name('  Ada Lovelace  ') == 'Ada Lovelace'\n",
        encoding="utf-8",
    )
    _run_git(["init"], source_root)
    _run_git(["config", "user.email", "test@example.com"], source_root)
    _run_git(["config", "user.name", "Test User"], source_root)
    _run_git(["add", "names.py", "tests/test_names.py"], source_root)
    _run_git(["commit", "-m", "initial fixture"], source_root)
    _run_git(
        [
            "remote",
            "add",
            "origin",
            "https://github.com/fixture/coding-agent-live-fixture.git",
        ],
        source_root,
    )
    return source_root


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
    """Assert worker completion and expose trace path on failure."""

    assert trace["worker_tick"]["processed_count"] == 1, trace["trace_path"]
    assert trace["worker_tick"]["succeeded_count"] == 1, trace["trace_path"]
    job = trace["background_job"]
    assert job["worker"] == "coding_agent", trace["trace_path"]
    assert job["status"] == "completed", trace["trace_path"]
    assert trace["worker_metadata"]["schema_version"] == (
        "coding_agent_worker_metadata.v2"
    )


def _coding_run_ref_from_trace(trace: dict[str, Any]) -> str:
    """Return the prompt-safe coding run ref from a worker trace."""

    coding_run_ref = str(trace["worker_metadata"].get("coding_run_ref", ""))
    assert coding_run_ref.startswith("coding_run:"), trace["trace_path"]
    return coding_run_ref


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
