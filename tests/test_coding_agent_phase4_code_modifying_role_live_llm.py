"""Live role diagnostics for existing-source code modification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.coding_agent.code_modifying.models import (
    ALLOWED_OPERATION_KINDS,
    RAW_DIFF_MARKERS,
    normalize_modifying_pm_decision,
)
from kazusa_ai_chatbot.coding_agent.code_modifying.programmer import (
    run_modifying_programmer,
)

pytestmark = pytest.mark.live_llm

TRACE_DIR = Path("test_artifacts/llm_traces/coding_agent_phase4_roles")
FIXTURE_ROOT = Path("tests/fixtures/coding_agent_existing_source_gates")


def _write_trace(name: str, payload: dict[str, object]) -> Path:
    """Persist raw role evidence for human review."""

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    path = TRACE_DIR / f"{name}.json"
    encoded_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(encoded_payload, encoding="utf-8")
    return path


def _file_context(fixture_dir: str, relative_path: str) -> dict[str, object]:
    """Load one fixture file as bounded modifying programmer context."""

    file_path = FIXTURE_ROOT / fixture_dir / relative_path
    content = file_path.read_text(encoding="utf-8")
    context = {
        "path": relative_path,
        "content": content,
        "truncated": False,
    }
    return context


def _evidence(path: str, evidence_id: str) -> dict[str, object]:
    """Create one source evidence row with the id expected by role prompts."""

    row = {
        "evidence_id": evidence_id,
        "path": path,
        "line_start": 1,
        "line_end": 999,
        "symbol_or_topic": "role test source evidence",
        "excerpt": "Fixture source file included in file_contexts.",
        "reason": "Role-level source modification contract evidence.",
    }
    return row


def _payload(
    *,
    question: str,
    evidence: list[dict[str, object]],
    file_contexts: list[dict[str, object]],
) -> dict[str, object]:
    """Build the direct modifying programmer payload used by live role cases."""

    payload = {
        "question": question,
        "source_scope": {
            "kind": "repository",
            "repo_relative_path": None,
            "source_url": "local://role-fixture",
            "requested_ref": "main",
            "interpretation": "role-level fixture",
        },
        "reading_answer": "The provided files own the requested behavior.",
        "evidence": evidence,
        "file_contexts": file_contexts,
        "output_contract": {
            "operation_kinds": sorted(ALLOWED_OPERATION_KINDS),
            "raw_diffs_allowed": False,
            "command_execution_allowed": False,
        },
    }
    return payload


def _successful_artifacts(result: dict[str, object]) -> list[dict[str, object]]:
    """Return normalized successful artifacts from one programmer result."""

    raw_artifacts = result.get("artifacts")
    assert isinstance(raw_artifacts, list)
    artifacts: list[dict[str, object]] = []
    for artifact in raw_artifacts:
        assert isinstance(artifact, dict)
        if artifact.get("status") != "succeeded":
            continue
        artifacts.append(artifact)
    return artifacts


def _assert_structured_artifacts(
    *,
    result: dict[str, object],
    payload: dict[str, object],
    expected_paths: set[str],
    trace_path: Path,
) -> None:
    """Validate structural live programmer output without judging semantics."""

    artifacts = _successful_artifacts(result)
    assert artifacts, f"trace={trace_path}"

    contexts = payload["file_contexts"]
    assert isinstance(contexts, list)
    context_by_path = {
        context["path"]: context["content"]
        for context in contexts
        if isinstance(context, dict)
    }
    seen_expected_paths: set[str] = set()
    for artifact in artifacts:
        target_path = artifact["target_path"]
        assert target_path in expected_paths, f"trace={trace_path}"
        seen_expected_paths.add(str(target_path))

        operation_kind = artifact["operation_kind"]
        assert operation_kind in ALLOWED_OPERATION_KINDS, f"trace={trace_path}"

        evidence_ids = artifact["evidence_ids"]
        assert evidence_ids, f"trace={trace_path}"

        content = str(artifact["replacement_or_insert_content"])
        assert content.strip(), f"trace={trace_path}"
        assert not any(marker in content for marker in RAW_DIFF_MARKERS)

        if operation_kind == "replace_file_small":
            continue

        anchor = str(artifact["exact_anchor"])
        source_text = str(context_by_path[target_path])
        assert anchor in source_text, f"trace={trace_path}"

    assert seen_expected_paths, f"trace={trace_path}"


async def _run_programmer_case(
    *,
    case_id: str,
    payload: dict[str, object],
    expected_paths: set[str],
) -> None:
    """Run one live modifying programmer case and persist raw evidence."""

    result = await run_modifying_programmer(payload)
    trace_path = _write_trace(
        f"{case_id}_programmer",
        {
            "case_id": case_id,
            "role": "modifying_programmer",
            "payload": payload,
            "result": result,
        },
    )
    print(f"Modifying programmer live trace={trace_path}")

    _assert_structured_artifacts(
        result=result,
        payload=payload,
        expected_paths=expected_paths,
        trace_path=trace_path,
    )


@pytest.mark.asyncio
async def test_live_modifying_programmer_single_file_edit() -> None:
    """Prove the programmer can propose a bounded single-file edit."""

    payload = _payload(
        question=(
            "Add a --json flag to log_counter.py. Keep text output unchanged "
            "when the flag is absent."
        ),
        evidence=[_evidence("log_counter.py", "evidence-1")],
        file_contexts=[
            _file_context("gate_01_log_counter", "log_counter.py"),
        ],
    )
    await _run_programmer_case(
        case_id="single_file_edit",
        payload=payload,
        expected_paths={"log_counter.py"},
    )


@pytest.mark.asyncio
async def test_live_modifying_programmer_parser_edit() -> None:
    """Prove the programmer can propose parser/scanner operations."""

    payload = _payload(
        question=(
            "Modify the Markdown link checker so scanner logic ignores links "
            "inside fenced code blocks and HTML comments."
        ),
        evidence=[
            _evidence("mdlinkcheck/scanner.py", "evidence-1"),
            _evidence("tests/test_scanner.py", "evidence-2"),
        ],
        file_contexts=[
            _file_context(
                "gate_03_markdown_link_checker",
                "mdlinkcheck/scanner.py",
            ),
            _file_context(
                "gate_03_markdown_link_checker",
                "tests/test_scanner.py",
            ),
        ],
    )
    await _run_programmer_case(
        case_id="parser_edit",
        payload=payload,
        expected_paths={"mdlinkcheck/scanner.py", "tests/test_scanner.py"},
    )


@pytest.mark.asyncio
async def test_live_modifying_programmer_cross_layer_edit() -> None:
    """Prove the programmer can span model, store, API, and test files."""

    payload = _payload(
        question=(
            "Add soft-delete support to the issue tracker. Archive issues "
            "instead of deleting them permanently, hide archived issues from "
            "normal lists, and update API and tests."
        ),
        evidence=[
            _evidence("issue_tracker/models.py", "evidence-1"),
            _evidence("issue_tracker/store.py", "evidence-2"),
            _evidence("issue_tracker/api.py", "evidence-3"),
            _evidence("tests/test_store.py", "evidence-4"),
        ],
        file_contexts=[
            _file_context(
                "gate_04_issue_tracker_soft_delete",
                "issue_tracker/models.py",
            ),
            _file_context(
                "gate_04_issue_tracker_soft_delete",
                "issue_tracker/store.py",
            ),
            _file_context(
                "gate_04_issue_tracker_soft_delete",
                "issue_tracker/api.py",
            ),
            _file_context(
                "gate_04_issue_tracker_soft_delete",
                "tests/test_store.py",
            ),
        ],
    )
    await _run_programmer_case(
        case_id="cross_layer_edit",
        payload=payload,
        expected_paths={
            "issue_tracker/models.py",
            "issue_tracker/store.py",
            "issue_tracker/api.py",
            "tests/test_store.py",
        },
    )


@pytest.mark.asyncio
async def test_live_modifying_programmer_mixed_existing_files() -> None:
    """Prove the programmer can coordinate fetch, CLI, tests, and docs."""

    payload = _payload(
        question=(
            "Add inventory vendor fetch caching with a configurable TTL, wire "
            "the CLI flag, update mocked HTTP tests, and document the behavior."
        ),
        evidence=[
            _evidence("inventory_sync/fetch.py", "evidence-1"),
            _evidence("inventory_sync/cli.py", "evidence-2"),
            _evidence("tests/test_fetch.py", "evidence-3"),
            _evidence("README.md", "evidence-4"),
        ],
        file_contexts=[
            _file_context(
                "gate_05_inventory_sync_fetch_cache",
                "inventory_sync/fetch.py",
            ),
            _file_context(
                "gate_05_inventory_sync_fetch_cache",
                "inventory_sync/cli.py",
            ),
            _file_context(
                "gate_05_inventory_sync_fetch_cache",
                "tests/test_fetch.py",
            ),
            _file_context(
                "gate_05_inventory_sync_fetch_cache",
                "README.md",
            ),
        ],
    )
    await _run_programmer_case(
        case_id="mixed_existing_files",
        payload=payload,
        expected_paths={
            "inventory_sync/fetch.py",
            "inventory_sync/cli.py",
            "tests/test_fetch.py",
            "README.md",
        },
    )


def test_modifying_pm_contract_lifecycle_decisions_recorded() -> None:
    """Record deterministic PM contract coverage beside live role traces."""

    raw_decisions: list[dict[str, Any]] = [
        {
            "status": "request_information",
            "reason": "Need source evidence for target path.",
            "owned_paths": [],
            "read_only_paths": ["README.md"],
            "required_evidence_ids": [],
        },
        {
            "status": "create_child_pm",
            "reason": "Split store and API analysis.",
            "owned_paths": ["issue_tracker/store.py"],
            "read_only_paths": ["issue_tracker/api.py"],
            "required_evidence_ids": ["evidence-1"],
        },
        {
            "status": "create_programmer_task",
            "reason": "Evidence is sufficient for a bounded edit.",
            "owned_paths": ["log_counter.py"],
            "read_only_paths": ["tests/test_log_counter.py"],
            "required_evidence_ids": ["evidence-1"],
            "programmer_task": {
                "task_id": "task-1",
                "target_paths": ["log_counter.py"],
                "change_goal": "Add JSON output flag.",
                "required_behavior": ["Keep text output unchanged."],
                "forbidden_changes": ["Do not replace the script."],
                "consumed_interfaces": ["main(argv)"],
                "expected_operations": ["replace"],
                "acceptance_checks": ["JSON output parses."],
                "local_risks": ["CLI argument order."],
            },
        },
        {
            "status": "repair_child",
            "reason": "Patch validation reported an anchor issue.",
            "owned_paths": ["log_counter.py"],
            "read_only_paths": [],
            "required_evidence_ids": ["evidence-1"],
            "repair_instruction": {
                "child_id": "programmer-1",
                "feedback_source": "patch_validation",
                "feedback": "Anchor was not found.",
                "expected_correction": "Return an exact anchor from source.",
            },
        },
        {
            "status": "complete",
            "reason": "Selected artifacts satisfy the bounded request.",
            "owned_paths": ["log_counter.py"],
            "read_only_paths": [],
            "required_evidence_ids": ["evidence-1"],
        },
        {
            "status": "blocked",
            "reason": "Source evidence is insufficient.",
            "owned_paths": [],
            "read_only_paths": [],
            "required_evidence_ids": [],
        },
    ]
    normalized = [
        normalize_modifying_pm_decision(raw_decision)
        for raw_decision in raw_decisions
    ]
    blocked_repair = normalize_modifying_pm_decision({
        "status": "repair_child",
        "reason": "Executed test output is not structural feedback.",
        "repair_instruction": {
            "child_id": "programmer-1",
            "feedback_source": "executed_test_output",
            "feedback": "pytest failed",
            "expected_correction": "Patch runtime behavior.",
        },
    })
    trace_path = _write_trace(
        "pm_contract_lifecycle",
        {
            "case_id": "pm_contract_lifecycle",
            "role": "modifying_pm_contract",
            "raw_decisions": raw_decisions,
            "normalized": normalized,
            "blocked_repair": blocked_repair,
        },
    )
    print(f"Modifying PM contract trace={trace_path}")

    statuses = [decision["status"] for decision in normalized]
    assert statuses == [
        "request_information",
        "create_child_pm",
        "create_programmer_task",
        "repair_child",
        "complete",
        "blocked",
    ], f"trace={trace_path}"
    assert blocked_repair["status"] == "blocked", f"trace={trace_path}"
