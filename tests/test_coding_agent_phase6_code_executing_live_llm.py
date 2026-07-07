"""Live LLM gates for bounded coding-agent verification execution."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, TypedDict

import pytest

from kazusa_ai_chatbot.coding_agent import propose_code_change


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

TRACE_ROOT = Path("test_artifacts/llm_traces/coding_agent_phase6_code_executing")
WORKSPACE_ROOT = Path("test_artifacts/coding_agent_phase6_code_executing_workspace")
FIXTURE_ROOT = Path("tests/fixtures/coding_agent_existing_source_gates")
MANAGED_APPLY_DIR_NAME = "patch_apply"
APPLIED_SOURCE_DIR_NAME = "source"
MAX_ANSWER_CHARS = 5000
MAX_ARTIFACT_CHARS = 64000
MAX_APPLY_FILES = 64
MAX_OUTPUT_CHARS = 6000


class CodeExecutionGate(TypedDict):
    """One real LLM code-execution signoff gate."""

    gate_id: str
    title: str
    objective: str
    fixture_dir: str
    existing_code_base: str
    modification_instruction: str
    execution_spec: dict[str, object]
    expected_state: list[str]
    pass_criteria: list[str]
    behavior_rubric: list[str]
    forbidden_failure_modes: list[str]
    trace_required: list[str]
    expected_evidence_paths: list[str]
    expected_executed_paths: list[str]
    acceptable_execution_statuses: list[str]


GATE_01: CodeExecutionGate = {
    "gate_id": "gate_01_cli_json_compile",
    "title": "Compile applied CLI JSON patch",
    "objective": (
        "Prove that a live LLM patch for a small CLI change can be applied "
        "and then verified through the bounded python_compileall execution "
        "tool inside the managed apply workspace."
    ),
    "fixture_dir": "gate_01_log_counter",
    "existing_code_base": (
        "A standard-library log_counter.py CLI with severity counting, text "
        "summary output, missing-file handling, README usage notes, and "
        "focused pytest coverage."
    ),
    "modification_instruction": (
        "Modify the existing log counter CLI to add a --json flag. When "
        "--json is provided, stdout must be valid JSON containing the same "
        "severity counts and the skipped-line count. The default text output "
        "must remain unchanged. Use only the Python standard library and "
        "update focused tests for the new JSON mode."
    ),
    "execution_spec": {
        "tool": "python_compileall",
        "paths": ["log_counter.py"],
        "pytest_selectors": [],
        "timeout_seconds": 15,
    },
    "expected_state": [
        "Patch proposal succeeds and approved artifacts apply first.",
        "Execution runs only under patch_apply/<package>/source.",
        "compileall receives argv, not a free-form shell command.",
        "Execution status is succeeded for syntactically valid applied code.",
        "Public result contains bounded stdout/stderr excerpts only.",
    ],
    "pass_criteria": [
        "Proposal and apply statuses are succeeded.",
        "Execution status is succeeded.",
        "tool is python_compileall.",
        "executed_paths contains log_counter.py as a relative path.",
        "No absolute source or workspace paths appear in public responses.",
        "Original source hashes remain unchanged after execution.",
    ],
    "behavior_rubric": [
        "The executor validates the managed apply workspace identity before "
        "running compileall.",
        "The executor reports timing, exit code, stdout/stderr excerpts, and "
        "truncation status.",
        "No package installation, shell syntax, or project mutation command "
        "is involved.",
    ],
    "forbidden_failure_modes": [
        "Execution runs in the original source checkout.",
        "The request accepts a free-form command string.",
        "stdout or stderr exposes absolute workspace paths.",
        "Execution silently succeeds without an exit code.",
    ],
    "trace_required": [
        "Gate input and source identity.",
        "Raw proposal, apply, and execution responses.",
        "Source hashes before and after execution.",
        "Managed apply package id and relative executed paths.",
        "Manual-review rubric and forbidden failure modes.",
    ],
    "expected_evidence_paths": [
        "log_counter.py",
        "tests/test_log_counter.py",
    ],
    "expected_executed_paths": ["log_counter.py"],
    "acceptable_execution_statuses": ["succeeded"],
}

GATE_02: CodeExecutionGate = {
    "gate_id": "gate_02_jsonl_pytest",
    "title": "Run JSONL utility focused pytest after apply",
    "objective": (
        "Prove bounded pytest execution for a live LLM multi-file utility "
        "patch, including structured reporting when generated target tests "
        "pass or fail."
    ),
    "fixture_dir": "gate_02_contacts_jsonl_to_csv",
    "existing_code_base": (
        "A contacts_jsonl_to_csv package with converter.py, cli.py, README, "
        "converter tests, and CLI tests. Baseline behavior has unstable "
        "requested field ordering and weak malformed-line handling."
    ),
    "modification_instruction": (
        "Modify the existing JSONL-to-CSV utility so --fields defines the "
        "exact CSV column order, missing fields are written as blank cells, "
        "malformed JSON is reported with 1-based line numbers, default "
        "behavior continues past malformed lines, and --strict fails fast on "
        "the first malformed line. Use only the Python standard library. "
        "Update focused converter, CLI, and README coverage."
    ),
    "execution_spec": {
        "tool": "pytest",
        "paths": [],
        "pytest_selectors": [
            "tests/test_converter.py",
            "tests/test_cli.py",
        ],
        "timeout_seconds": 30,
    },
    "expected_state": [
        "Proposal and approved apply complete before execution.",
        "pytest runs inside the managed apply workspace only.",
        "Executor returns succeeded when focused tests pass or failed when "
        "the target tests expose an LLM patch defect.",
        "Nonzero exit remains a structured execution result, not a harness "
        "crash.",
        "Output excerpts are capped and sanitized.",
    ],
    "pass_criteria": [
        "Proposal and apply statuses are succeeded.",
        "Execution status is one of succeeded or failed.",
        "tool is pytest.",
        "exit_code is present for non-timeout results.",
        "executed_paths contains both focused pytest selectors.",
        "Raw public responses omit absolute source and workspace paths.",
    ],
    "behavior_rubric": [
        "The executor reports pytest success or target-test failure honestly.",
        "Failed target tests remain available as bounded evidence for future "
        "manual review or repair planning.",
        "The executor does not retry, repair, or feed raw output into an LLM.",
    ],
    "forbidden_failure_modes": [
        "A failing target pytest run is hidden as a succeeded execution.",
        "Executor failure is represented by an unhandled exception.",
        "pytest is invoked through shell text or command chaining.",
        "Output includes absolute local paths or environment values.",
    ],
    "trace_required": [
        "Gate input and source identity.",
        "Raw proposal, apply, and execution responses.",
        "Focused pytest selectors.",
        "Source hash comparison after execution.",
        "Manual-review rubric and forbidden failure modes.",
    ],
    "expected_evidence_paths": [
        "contacts_jsonl_to_csv/converter.py",
        "contacts_jsonl_to_csv/cli.py",
        "tests/test_converter.py",
        "README.md",
    ],
    "expected_executed_paths": [
        "tests/test_converter.py",
        "tests/test_cli.py",
    ],
    "acceptable_execution_statuses": ["succeeded", "failed"],
}

GATE_03: CodeExecutionGate = {
    "gate_id": "gate_03_markdown_pytest",
    "title": "Run Markdown parser focused pytest after apply",
    "objective": (
        "Prove bounded pytest execution for a parser-owned live LLM patch "
        "where scanner and anchor tests exercise edge cases."
    ),
    "fixture_dir": "gate_03_markdown_link_checker",
    "existing_code_base": (
        "A mdlinkcheck package with anchors.py, scanner.py, cli.py, README, "
        "and anchor/scanner tests. Baseline scanning treats links inside "
        "fences and comments as real links and lacks duplicate heading "
        "suffix resolution."
    ),
    "modification_instruction": (
        "Modify the existing Markdown link checker to ignore links inside "
        "fenced code blocks and HTML comments. Support duplicate heading "
        "anchors with GitHub-style suffixes when resolving links: the first "
        "duplicate heading keeps the base anchor, the second becomes base-1, "
        "the third becomes base-2, and so on. Update focused parser/scanner "
        "tests and keep normal links outside comments and fences working."
    ),
    "execution_spec": {
        "tool": "pytest",
        "paths": [],
        "pytest_selectors": [
            "tests/test_anchors.py",
            "tests/test_scanner.py",
        ],
        "timeout_seconds": 30,
    },
    "expected_state": [
        "Patch proposal and apply complete before execution.",
        "pytest selectors are validated as relative managed-workspace targets.",
        "Executor reports parser test success or failure structurally.",
        "The original source checkout is unchanged.",
        "Execution output is capped and sanitized.",
    ],
    "pass_criteria": [
        "Proposal and apply statuses are succeeded.",
        "Execution status is one of succeeded or failed.",
        "tool is pytest.",
        "executed_paths contains anchor and scanner test selectors.",
        "No shell string, package manager, or network command is used.",
        "Original source hashes match after execution.",
    ],
    "behavior_rubric": [
        "Execution result makes target parser-test failures inspectable.",
        "The executor does not attempt automatic repair.",
        "stdout/stderr are short enough for manual review and contain no "
        "absolute workspace paths.",
    ],
    "forbidden_failure_modes": [
        "pytest selectors escape the managed apply workspace.",
        "Executor accepts hidden shell syntax.",
        "Output cap is ignored on noisy pytest output.",
        "Original checkout is mutated by execution.",
    ],
    "trace_required": [
        "Gate input and source identity.",
        "Raw proposal, apply, and execution responses.",
        "Focused pytest selectors.",
        "Source hash comparison.",
        "Manual-review rubric and forbidden failure modes.",
    ],
    "expected_evidence_paths": [
        "mdlinkcheck/anchors.py",
        "mdlinkcheck/scanner.py",
        "tests/test_anchors.py",
        "tests/test_scanner.py",
    ],
    "expected_executed_paths": [
        "tests/test_anchors.py",
        "tests/test_scanner.py",
    ],
    "acceptable_execution_statuses": ["succeeded", "failed"],
}

GATE_04: CodeExecutionGate = {
    "gate_id": "gate_04_issue_tracker_pytest",
    "title": "Run issue tracker focused pytest after apply",
    "objective": (
        "Prove bounded pytest execution for a cross-layer live LLM patch "
        "spanning model, store, API, tests, and docs."
    ),
    "fixture_dir": "gate_04_issue_tracker_soft_delete",
    "existing_code_base": (
        "An issue_tracker package with models.py, store.py, api.py, README, "
        "and store/API tests. The baseline delete path hard-removes issues "
        "from the in-memory store."
    ),
    "modification_instruction": (
        "Modify the existing issue tracker to implement soft delete. Deleting "
        "an issue should mark it archived instead of removing it from "
        "storage. Normal list results should hide archived issues. "
        "Single-item lookup should return not found for archived issues. "
        "List should support an include_archived option. Update store/API "
        "tests and README. Keep behavior coherent with the existing project "
        "style and avoid compatibility wrappers around the old hard-delete "
        "semantics."
    ),
    "execution_spec": {
        "tool": "pytest",
        "paths": [],
        "pytest_selectors": [
            "tests/test_store.py",
            "tests/test_api.py",
        ],
        "timeout_seconds": 30,
    },
    "expected_state": [
        "Patch proposal and approved apply complete first.",
        "pytest runs only against the managed applied issue tracker copy.",
        "Executor reports target test result without repair loops.",
        "Public response sanitizes paths and output.",
        "Original source remains unchanged.",
    ],
    "pass_criteria": [
        "Proposal and apply statuses are succeeded.",
        "Execution status is one of succeeded or failed.",
        "tool is pytest.",
        "executed_paths contains store and API test selectors.",
        "exit_code is represented for non-timeout execution.",
        "Public responses omit absolute source and workspace paths.",
    ],
    "behavior_rubric": [
        "A passing pytest result indicates the live LLM patch likely kept "
        "model, store, and API semantics aligned.",
        "A failing pytest result remains a valid executor signal and must be "
        "reviewed as patch-quality evidence, not executor failure.",
        "Execution never reintroduces hard-delete compatibility behavior.",
    ],
    "forbidden_failure_modes": [
        "Executor reports succeeded while pytest failed.",
        "Execution runs against the original checkout.",
        "Output leaks local paths, environment data, or raw traces.",
        "Executor starts a repair loop or invokes a coding LLM.",
    ],
    "trace_required": [
        "Gate input and source identity.",
        "Raw proposal, apply, and execution responses.",
        "Focused pytest selectors.",
        "Source hash comparison.",
        "Manual-review rubric and forbidden failure modes.",
    ],
    "expected_evidence_paths": [
        "issue_tracker/models.py",
        "issue_tracker/store.py",
        "issue_tracker/api.py",
        "tests/test_store.py",
        "README.md",
    ],
    "expected_executed_paths": [
        "tests/test_store.py",
        "tests/test_api.py",
    ],
    "acceptable_execution_statuses": ["succeeded", "failed"],
}

GATE_05: CodeExecutionGate = {
    "gate_id": "gate_05_inventory_compile",
    "title": "Compile applied inventory cache patch",
    "objective": (
        "Prove bounded compile execution for a hard live LLM patch that may "
        "introduce a helper file while preserving the existing inventory_sync "
        "package layout."
    ),
    "fixture_dir": "gate_05_inventory_sync_fetch_cache",
    "existing_code_base": (
        "An inventory_sync package with CSV reading, urllib fetch, HTML "
        "metadata extraction, report writing, CLI, README, and mocked HTTP "
        "tests. Baseline fetch has no explicit timeout, retry, or cache "
        "support."
    ),
    "modification_instruction": (
        "Modify the existing inventory sync project to add timeout and retry "
        "handling for vendor page fetches, add a file-backed response cache, "
        "expose CLI flags --cache-dir, --refresh-cache, and --timeout, update "
        "mocked HTTP tests, and document the workflow. Use only the Python "
        "standard library. Do not run real network calls in tests."
    ),
    "execution_spec": {
        "tool": "python_compileall",
        "paths": ["inventory_sync"],
        "pytest_selectors": [],
        "timeout_seconds": 20,
    },
    "expected_state": [
        "Patch proposal and approved apply complete first.",
        "compileall runs against the managed applied inventory_sync package.",
        "Execution status is succeeded for syntactically valid applied code.",
        "Original source remains unchanged.",
        "Public output remains bounded and sanitized.",
    ],
    "pass_criteria": [
        "Proposal and apply statuses are succeeded.",
        "Execution status is succeeded.",
        "tool is python_compileall.",
        "executed_paths contains inventory_sync.",
        "No package manager, network command, or shell string appears.",
        "Original source hashes match after execution.",
    ],
    "behavior_rubric": [
        "The executor validates the package directory as a relative path.",
        "compileall does not execute project tests or network calls.",
        "Execution metadata is sufficient to tell whether syntax passed.",
    ],
    "forbidden_failure_modes": [
        "Execution uses a free-form shell command.",
        "Executor runs package installation or network checks.",
        "Original checkout is mutated.",
        "stdout/stderr leak absolute workspace paths.",
    ],
    "trace_required": [
        "Gate input and source identity.",
        "Raw proposal, apply, and execution responses.",
        "Relative executed paths.",
        "Source hash comparison.",
        "Manual-review rubric and forbidden failure modes.",
    ],
    "expected_evidence_paths": [
        "inventory_sync/fetch.py",
        "inventory_sync/cli.py",
        "tests/test_fetch.py",
        "tests/test_cli.py",
        "README.md",
    ],
    "expected_executed_paths": ["inventory_sync"],
    "acceptable_execution_statuses": ["succeeded"],
}


async def test_phase6_live_gate_01_cli_json_compile() -> None:
    """Run the simple compile execution gate."""

    result = await _run_code_execution_gate(GATE_01)
    _assert_code_execution_gate(gate=GATE_01, result=result)


async def test_phase6_live_gate_02_jsonl_pytest() -> None:
    """Run the JSONL focused pytest execution gate."""

    result = await _run_code_execution_gate(GATE_02)
    _assert_code_execution_gate(gate=GATE_02, result=result)


async def test_phase6_live_gate_03_markdown_pytest() -> None:
    """Run the Markdown parser focused pytest execution gate."""

    result = await _run_code_execution_gate(GATE_03)
    _assert_code_execution_gate(gate=GATE_03, result=result)


async def test_phase6_live_gate_04_issue_tracker_pytest() -> None:
    """Run the issue tracker focused pytest execution gate."""

    result = await _run_code_execution_gate(GATE_04)
    _assert_code_execution_gate(gate=GATE_04, result=result)


async def test_phase6_live_gate_05_inventory_compile() -> None:
    """Run the hard mixed compile execution gate."""

    result = await _run_code_execution_gate(GATE_05)
    _assert_code_execution_gate(gate=GATE_05, result=result)


async def _run_code_execution_gate(
    gate: CodeExecutionGate,
) -> dict[str, dict[str, Any]]:
    """Produce, apply, execute, and persist evidence for one live gate."""

    gate_id = gate["gate_id"]
    workspace_root = _reset_gate_workspace(gate_id)
    fixture_root = FIXTURE_ROOT / gate["fixture_dir"]
    source_root = workspace_root / "source"
    shutil.copytree(fixture_root, source_root)
    _initialize_fixture_git_checkout(source_root, gate_id)

    before_hashes = _hash_tree(source_root)
    proposal_response = await propose_code_change({
        "question": gate["modification_instruction"],
        "local_root_hint": str(source_root),
        "workspace_root": str(workspace_root),
        "session_id": gate_id,
        "preferred_language": "English",
        "max_answer_chars": MAX_ANSWER_CHARS,
        "max_artifact_chars": MAX_ARTIFACT_CHARS,
    })
    source_identity = _source_identity_from_proposal(proposal_response)

    from kazusa_ai_chatbot.coding_agent import apply_approved_patch
    from kazusa_ai_chatbot.coding_agent import execute_code_check

    apply_response = apply_approved_patch({
        "workspace_root": str(workspace_root),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "expected_source_identity": source_identity,
        "patch_artifacts": proposal_response["patch_artifacts"],
        "approval": {
            "approved": True,
            "approved_by": "phase6_live_llm_gate",
            "approved_at": "2026-07-08T00:00:00Z",
            "approval_reason": gate["objective"],
        },
        "max_files": MAX_APPLY_FILES,
        "max_diff_chars": MAX_ARTIFACT_CHARS,
    })
    execution_response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_response["apply_package_id"],
        "apply_workspace_ref": apply_response["apply_workspace_ref"],
        "execution": gate["execution_spec"],
        "max_stdout_chars": MAX_OUTPUT_CHARS,
        "max_stderr_chars": MAX_OUTPUT_CHARS,
    })
    after_hashes = _hash_tree(source_root)

    trace_payload = {
        "gate": gate,
        "source_root": str(source_root),
        "workspace_root": str(workspace_root),
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "source_tree_unchanged": before_hashes == after_hashes,
        "proposal_response": proposal_response,
        "apply_response": apply_response,
        "execution_response": execution_response,
        "managed_execution_root": str(_managed_apply_source_root(
            workspace_root=workspace_root,
            apply_response=apply_response,
        )),
        "judgment": "manual_review_required_for_phase6_execution_quality",
    }
    trace_path = TRACE_ROOT / f"{gate_id}_raw_evidence.json"
    _write_json(trace_path, trace_payload)

    print(f"gate_id={gate_id}")
    print(f"phase6_raw_evidence_path={trace_path}")
    print(f"execution_status={execution_response['status']}")
    print(f"execution_tool={execution_response['tool']}")

    result = {
        "proposal": proposal_response,
        "apply": apply_response,
        "execution": execution_response,
    }
    return result


def _source_identity_from_proposal(
    proposal_response: dict[str, Any],
) -> dict[str, object]:
    """Build the source identity used by the apply contract."""

    repository = proposal_response["repository"]
    assert repository is not None
    source_identity = {
        "provider": repository["provider"],
        "owner": repository["owner"],
        "repo": repository["repo"],
        "current_commit": repository["current_commit"],
        "dirty_state": repository["dirty_state"],
    }
    return source_identity


def _initialize_fixture_git_checkout(source_root: Path, gate_id: str) -> None:
    """Make the copied fixture satisfy the local checkout source contract."""

    commands = [
        ["git", "init", "-b", "main"],
        ["git", "config", "user.email", "coding-agent-gate@example.invalid"],
        ["git", "config", "user.name", "Coding Agent Gate"],
        ["git", "config", "core.autocrlf", "false"],
        ["git", "config", "core.eol", "lf"],
        [
            "git",
            "remote",
            "add",
            "origin",
            f"https://github.com/kazusa-fixtures/{gate_id}",
        ],
        ["git", "add", "."],
        ["git", "commit", "-m", "fixture baseline"],
    ]
    for command in commands:
        subprocess.run(
            command,
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )


def _reset_gate_workspace(gate_id: str) -> Path:
    """Create a clean managed workspace for one live gate run."""

    workspace_root = WORKSPACE_ROOT.resolve()
    gate_workspace = (WORKSPACE_ROOT / gate_id).resolve()
    if not gate_workspace.is_relative_to(workspace_root):
        raise AssertionError("Gate workspace escaped the managed workspace root.")

    if gate_workspace.exists():
        shutil.rmtree(gate_workspace, onerror=_retry_remove_readonly)
    gate_workspace.mkdir(parents=True, exist_ok=True)

    return gate_workspace


def _retry_remove_readonly(
    function: object,
    path: str,
    _exc_info: object,
) -> None:
    """Clear Windows read-only bits left by git object files and retry."""

    if not callable(function):
        raise AssertionError(f"Cleanup callback was not callable for {path}.")
    os.chmod(path, 0o700)
    function(path)


def _hash_tree(root: Path) -> dict[str, str]:
    """Hash every file in a source tree for non-mutation checks."""

    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative_path] = digest
    return hashes


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write raw structured evidence for later human-authored review."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(encoded_payload, encoding="utf-8")


def _managed_apply_source_root(
    *,
    workspace_root: Path,
    apply_response: dict[str, Any],
) -> Path:
    """Resolve the managed apply source root from public package metadata."""

    apply_package_id = str(apply_response["apply_package_id"])
    source_root = (
        workspace_root
        / MANAGED_APPLY_DIR_NAME
        / apply_package_id
        / APPLIED_SOURCE_DIR_NAME
    )
    return source_root


def _assert_code_execution_gate(
    *,
    gate: CodeExecutionGate,
    result: dict[str, dict[str, Any]],
) -> None:
    """Assert structural evidence for one live execution gate."""

    proposal_response = result["proposal"]
    apply_response = result["apply"]
    execution_response = result["execution"]

    assert proposal_response["status"] == "succeeded"
    assert proposal_response["patch_artifacts"]
    assert proposal_response["evidence"]
    assert apply_response["status"] == "succeeded"
    assert apply_response["apply_workspace_ref"]["kind"] == (
        "managed_apply_workspace"
    )
    assert execution_response["status"] in gate["acceptable_execution_statuses"]
    assert execution_response["tool"] == gate["execution_spec"]["tool"]
    assert execution_response["duration_ms"] >= 0
    assert isinstance(execution_response["timed_out"], bool)
    assert isinstance(execution_response["output_truncated"], bool)
    assert "stdout_excerpt" in execution_response
    assert "stderr_excerpt" in execution_response
    if execution_response["status"] != "timed_out":
        assert "exit_code" in execution_response

    evidence_paths = [
        str(evidence_item["path"])
        for evidence_item in proposal_response["evidence"]
    ]
    for expected_path in gate["expected_evidence_paths"]:
        assert any(expected_path in path for path in evidence_paths)

    executed_paths = [
        str(executed_path)
        for executed_path in execution_response["executed_paths"]
    ]
    for expected_path in gate["expected_executed_paths"]:
        assert any(expected_path in path for path in executed_paths)

    gate_workspace = WORKSPACE_ROOT / gate["gate_id"]
    raw_response = json.dumps(result, ensure_ascii=False)
    copied_source_root = gate_workspace / "source"
    assert str(copied_source_root.resolve()) not in raw_response
    assert str(gate_workspace.resolve()) not in raw_response

    raw_evidence_path = TRACE_ROOT / f"{gate['gate_id']}_raw_evidence.json"
    raw_evidence = json.loads(raw_evidence_path.read_text(encoding="utf-8"))
    assert raw_evidence["source_tree_unchanged"] is True
    assert gate["objective"]
    assert gate["existing_code_base"]
    assert gate["modification_instruction"]
    assert gate["execution_spec"]
    assert gate["expected_state"]
    assert gate["pass_criteria"]
