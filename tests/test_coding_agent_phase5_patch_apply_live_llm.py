"""Live LLM gates for approved coding-agent patch application."""

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

TRACE_ROOT = Path("test_artifacts/llm_traces/coding_agent_phase5_patch_apply")
WORKSPACE_ROOT = Path("test_artifacts/coding_agent_phase5_patch_apply_workspace")
FIXTURE_ROOT = Path("tests/fixtures/coding_agent_existing_source_gates")
MANAGED_APPLY_DIR_NAME = "patch_apply"
APPLIED_SOURCE_DIR_NAME = "source"
MAX_ANSWER_CHARS = 5000
MAX_ARTIFACT_CHARS = 64000
MAX_APPLY_FILES = 64


class PatchApplyGate(TypedDict):
    """One real LLM patch-apply signoff gate."""

    gate_id: str
    title: str
    objective: str
    fixture_dir: str
    existing_code_base: str
    modification_instruction: str
    expected_state: list[str]
    pass_criteria: list[str]
    behavior_rubric: list[str]
    forbidden_failure_modes: list[str]
    trace_required: list[str]
    expected_evidence_paths: list[str]
    expected_applied_paths: list[str]


GATE_01: PatchApplyGate = {
    "gate_id": "gate_01_cli_json_apply",
    "title": "Apply a single-file CLI JSON output patch",
    "objective": (
        "Prove that an explicitly approved live LLM patch proposal for a "
        "small single-file CLI change is applied into a managed copy while "
        "the original checkout remains unchanged."
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
    "expected_state": [
        "Patch proposal succeeds through the existing source modification path.",
        "Approved patch artifacts are applied under the managed apply workspace.",
        "The original copied checkout remains byte-identical after apply.",
        "Applied files include log_counter.py and its focused test file.",
        "The apply response exposes only package id, relative paths, and "
        "public source identity.",
    ],
    "pass_criteria": [
        "Proposal status is succeeded and contains patch artifacts.",
        "Apply status is succeeded.",
        "Apply response includes apply_package_id and apply_workspace_ref.",
        "Managed applied files exist under patch_apply/<package>/source.",
        "Original source hashes before and after proposal/apply are identical.",
        "Serialized public responses do not contain absolute source or "
        "workspace paths.",
    ],
    "behavior_rubric": [
        "The applied workspace contains a targeted CLI modification, not a "
        "replacement project.",
        "The default text-output path remains visibly present.",
        "The focused test update checks JSON parsing rather than only a "
        "substring.",
    ],
    "forbidden_failure_modes": [
        "Patch application mutates the original source checkout.",
        "Apply succeeds without a structured approval object.",
        "Apply response exposes absolute local roots or raw diffs as public "
        "metadata.",
        "Patch artifacts are ignored while the apply stage reports success.",
    ],
    "trace_required": [
        "Gate input and copied source identity.",
        "Raw proposal response and raw apply response.",
        "Source tree hashes before and after apply.",
        "Managed applied file paths for local inspection.",
        "Manual-review rubric and forbidden failure modes.",
    ],
    "expected_evidence_paths": [
        "log_counter.py",
        "tests/test_log_counter.py",
    ],
    "expected_applied_paths": [
        "log_counter.py",
        "tests/test_log_counter.py",
    ],
}

GATE_02: PatchApplyGate = {
    "gate_id": "gate_02_jsonl_errors_apply",
    "title": "Apply a multi-file JSONL utility patch",
    "objective": (
        "Prove managed apply for a realistic parser plus CLI patch where "
        "source, tests, and documentation move together."
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
    "expected_state": [
        "Patch proposal includes converter, CLI, tests, and README changes.",
        "Approved patch artifacts apply under a managed apply package.",
        "The original checkout remains byte-identical.",
        "Applied files are inspectable by repo-relative path.",
        "No command execution or pytest output appears in apply metadata.",
    ],
    "pass_criteria": [
        "Proposal status is succeeded and evidence cites converter.py and cli.py.",
        "Apply status is succeeded and changed_files is non-empty.",
        "Managed applied files include converter, CLI, tests, and README paths.",
        "Original source hashes before and after apply are identical.",
        "Apply metadata contains no stdout, stderr, shell command, or exit code.",
    ],
    "behavior_rubric": [
        "The applied patch preserves user-requested field order.",
        "Malformed JSON handling belongs in converter and CLI behavior rather "
        "than docs only.",
        "Strict and non-strict paths are both represented in tests.",
    ],
    "forbidden_failure_modes": [
        "Requested field order is sorted or normalized against the instruction.",
        "Malformed-line reporting is implemented only as CLI text while "
        "converter behavior remains weak.",
        "Apply stage runs tests or commands in Phase 5.",
        "Public metadata leaks local workspace paths.",
    ],
    "trace_required": [
        "Gate input and source identity.",
        "Proposal response and apply response.",
        "Applied path list.",
        "Source hash comparison.",
        "Manual-review rubric and forbidden failure modes.",
    ],
    "expected_evidence_paths": [
        "contacts_jsonl_to_csv/converter.py",
        "contacts_jsonl_to_csv/cli.py",
        "tests/test_converter.py",
        "README.md",
    ],
    "expected_applied_paths": [
        "contacts_jsonl_to_csv/converter.py",
        "contacts_jsonl_to_csv/cli.py",
        "tests/test_converter.py",
        "tests/test_cli.py",
        "README.md",
    ],
}

GATE_03: PatchApplyGate = {
    "gate_id": "gate_03_markdown_parser_apply",
    "title": "Apply a Markdown parser edge-case patch",
    "objective": (
        "Prove managed apply for a parser-owned change where Markdown link "
        "scanner behavior and anchor behavior must remain coherent."
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
    "expected_state": [
        "Patch proposal includes scanner, anchor, and focused test changes.",
        "Approved artifacts apply atomically into one managed apply package.",
        "Original fixture checkout remains unchanged.",
        "Applied workspace retains existing package layout.",
        "Apply response contains no raw command execution surface.",
    ],
    "pass_criteria": [
        "Proposal status is succeeded and evidence cites scanner and anchors.",
        "Apply status is succeeded.",
        "Managed applied files include scanner, anchors, and tests.",
        "Original source hash comparison passes.",
        "Public apply response omits absolute paths.",
    ],
    "behavior_rubric": [
        "Ignored regions are handled in parser or scanner logic, not by CLI "
        "suppression.",
        "Duplicate anchors are resolved using base, base-1, base-2 semantics.",
        "Normal links outside ignored regions remain visible in the applied "
        "source.",
    ],
    "forbidden_failure_modes": [
        "The patch hardcodes fixture headings, filenames, or expected strings.",
        "Only one fence style is handled without acknowledging the other.",
        "Apply response reports success while applied files are absent.",
        "Original source checkout is modified.",
    ],
    "trace_required": [
        "Gate input and source identity.",
        "Raw proposal and apply responses.",
        "Managed applied paths.",
        "Source hash comparison.",
        "Manual-review rubric and forbidden failure modes.",
    ],
    "expected_evidence_paths": [
        "mdlinkcheck/anchors.py",
        "mdlinkcheck/scanner.py",
        "tests/test_anchors.py",
        "tests/test_scanner.py",
    ],
    "expected_applied_paths": [
        "mdlinkcheck/anchors.py",
        "mdlinkcheck/scanner.py",
        "tests/test_anchors.py",
        "tests/test_scanner.py",
    ],
}

GATE_04: PatchApplyGate = {
    "gate_id": "gate_04_issue_tracker_apply",
    "title": "Apply a cross-layer soft-delete patch",
    "objective": (
        "Prove managed apply for a cross-layer semantic change spanning "
        "model, store, API, tests, and README without old-behavior shims."
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
    "expected_state": [
        "Patch proposal spans model, store, API, tests, and README.",
        "Approved patch applies into a managed apply package.",
        "Original fixture source remains unchanged.",
        "Applied workspace contains one coherent issue_tracker package.",
        "Public response contains only sanitized metadata.",
    ],
    "pass_criteria": [
        "Proposal status is succeeded and evidence cites model, store, and API.",
        "Apply status is succeeded.",
        "Managed applied files include source, tests, and README changes.",
        "Original source hashes match before and after apply.",
        "No compatibility shim or alternate hard-delete vocabulary is visible "
        "in the trace rubric as acceptable behavior.",
    ],
    "behavior_rubric": [
        "Archived issues remain stored but hidden from normal lookup and list.",
        "include_archived exposes archived issues without changing defaults.",
        "API and store semantics stay aligned.",
        "The patch avoids parallel hard-delete compatibility paths.",
    ],
    "forbidden_failure_modes": [
        "The old hard-delete path remains as an alias or fallback.",
        "Archived issues appear in default list results.",
        "API and store behavior diverge.",
        "Apply stage mutates the original source checkout.",
    ],
    "trace_required": [
        "Gate input and source identity.",
        "Raw proposal and apply responses.",
        "Managed applied paths.",
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
    "expected_applied_paths": [
        "issue_tracker/models.py",
        "issue_tracker/store.py",
        "issue_tracker/api.py",
        "tests/test_store.py",
        "tests/test_api.py",
        "README.md",
    ],
}

GATE_05: PatchApplyGate = {
    "gate_id": "gate_05_inventory_cache_apply",
    "title": "Apply a mixed existing-file and helper patch",
    "objective": (
        "Prove managed apply for a hard realistic change that may include one "
        "new helper module while preserving existing fetch, CLI, parser, "
        "report, test, and README responsibilities."
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
    "expected_state": [
        "Patch proposal includes fetch, CLI, tests, README, and any justified "
        "helper file.",
        "Approved patch applies atomically under a managed apply package.",
        "Original source fixture remains byte-identical.",
        "Applied workspace preserves the inventory_sync package layout.",
        "Apply response remains sanitized and contains no execution output.",
    ],
    "pass_criteria": [
        "Proposal status is succeeded and evidence cites fetch.py and cli.py.",
        "Apply status is succeeded.",
        "Managed applied files include fetch, CLI, tests, and README paths.",
        "Original source hashes match before and after apply.",
        "Apply metadata contains no stdout, stderr, shell command, or exit code.",
    ],
    "behavior_rubric": [
        "Fetch behavior owns timeout propagation, retry, and cache semantics.",
        "CLI flags wire into the existing workflow instead of bypassing it.",
        "Tests mock network behavior and do not perform real network calls.",
        "Any new helper module is imported by existing package code.",
    ],
    "forbidden_failure_modes": [
        "A replacement end-to-end script bypasses existing modules.",
        "Tests perform real network calls.",
        "Cache behavior is only documented but not wired into fetch or CLI.",
        "Apply response leaks local paths or command output.",
    ],
    "trace_required": [
        "Gate input and source identity.",
        "Raw proposal and apply responses.",
        "Managed applied paths.",
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
    "expected_applied_paths": [
        "inventory_sync/fetch.py",
        "inventory_sync/cli.py",
        "tests/test_fetch.py",
        "tests/test_cli.py",
        "README.md",
    ],
}


async def test_phase5_live_gate_01_cli_json_apply() -> None:
    """Run the simple single-file apply gate."""

    proposal_response, apply_response = await _run_patch_apply_gate(GATE_01)
    _assert_patch_apply_gate(
        gate=GATE_01,
        proposal_response=proposal_response,
        apply_response=apply_response,
    )


async def test_phase5_live_gate_02_jsonl_errors_apply() -> None:
    """Run the small multi-file utility apply gate."""

    proposal_response, apply_response = await _run_patch_apply_gate(GATE_02)
    _assert_patch_apply_gate(
        gate=GATE_02,
        proposal_response=proposal_response,
        apply_response=apply_response,
    )


async def test_phase5_live_gate_03_markdown_parser_apply() -> None:
    """Run the parser edge-case apply gate."""

    proposal_response, apply_response = await _run_patch_apply_gate(GATE_03)
    _assert_patch_apply_gate(
        gate=GATE_03,
        proposal_response=proposal_response,
        apply_response=apply_response,
    )


async def test_phase5_live_gate_04_issue_tracker_apply() -> None:
    """Run the cross-layer soft-delete apply gate."""

    proposal_response, apply_response = await _run_patch_apply_gate(GATE_04)
    _assert_patch_apply_gate(
        gate=GATE_04,
        proposal_response=proposal_response,
        apply_response=apply_response,
    )


async def test_phase5_live_gate_05_inventory_cache_apply() -> None:
    """Run the hard mixed apply gate."""

    proposal_response, apply_response = await _run_patch_apply_gate(GATE_05)
    _assert_patch_apply_gate(
        gate=GATE_05,
        proposal_response=proposal_response,
        apply_response=apply_response,
    )


async def _run_patch_apply_gate(
    gate: PatchApplyGate,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Produce a live LLM patch proposal, apply it, and persist evidence."""

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

    apply_response: dict[str, Any]
    if _proposal_is_apply_eligible(proposal_response):
        apply_response = apply_approved_patch({
            "workspace_root": str(workspace_root),
            "source_root": str(source_root),
            "source_identity": source_identity,
            "expected_source_identity": source_identity,
            "patch_artifacts": proposal_response["patch_artifacts"],
            "approval": {
                "approved": True,
                "approved_by": "phase5_live_llm_gate",
                "approved_at": "2026-07-08T00:00:00Z",
                "approval_reason": gate["objective"],
            },
            "max_files": MAX_APPLY_FILES,
            "max_diff_chars": MAX_ARTIFACT_CHARS,
        })
    else:
        apply_response = {
            "status": "not_run",
            "apply_package_id": "",
            "source_identity": source_identity,
            "apply_workspace_ref": {
                "kind": "",
                "apply_package_id": "",
                "source_identity": source_identity,
                "applied_files": [],
            },
            "applied_files": [],
            "changed_files": [],
            "validation": {
                "status": "not_run",
                "errors": ["Proposal review validation did not pass."],
                "warnings": [],
            },
            "limitations": ["Proposal review validation did not pass."],
            "trace_summary": ["patch_apply:not_run:proposal_failed"],
        }
    after_hashes = _hash_tree(source_root)
    applied_paths = _managed_applied_paths(
        workspace_root=workspace_root,
        apply_response=apply_response,
    )

    trace_payload = {
        "gate": gate,
        "source_root": str(source_root),
        "workspace_root": str(workspace_root),
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "source_tree_unchanged": before_hashes == after_hashes,
        "proposal_response": proposal_response,
        "apply_response": apply_response,
        "managed_applied_paths": [str(path) for path in applied_paths],
        "judgment": "manual_review_required_for_phase5_patch_apply_quality",
    }
    trace_path = TRACE_ROOT / f"{gate_id}_raw_evidence.json"
    _write_json(trace_path, trace_payload)

    print(f"gate_id={gate_id}")
    print(f"phase5_raw_evidence_path={trace_path}")
    for path in applied_paths:
        print(f"managed_applied_file={path}")

    return proposal_response, apply_response


def _proposal_is_apply_eligible(proposal_response: dict[str, Any]) -> bool:
    """Return whether a proposal can be approved for live apply."""

    if proposal_response.get("status") != "succeeded":
        return False
    validation = proposal_response.get("validation")
    if isinstance(validation, dict) and validation.get("status") != "succeeded":
        return False
    patch_artifacts = proposal_response.get("patch_artifacts")
    return isinstance(patch_artifacts, list) and bool(patch_artifacts)


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


def _managed_applied_paths(
    *,
    workspace_root: Path,
    apply_response: dict[str, Any],
) -> list[Path]:
    """Resolve managed applied files from public package metadata."""

    apply_package_id = str(apply_response["apply_package_id"])
    apply_source_root = (
        workspace_root
        / MANAGED_APPLY_DIR_NAME
        / apply_package_id
        / APPLIED_SOURCE_DIR_NAME
    )
    paths: list[Path] = []
    for relative_path in apply_response["applied_files"]:
        paths.append(apply_source_root / str(relative_path))
    return paths


def _assert_patch_apply_gate(
    *,
    gate: PatchApplyGate,
    proposal_response: dict[str, Any],
    apply_response: dict[str, Any],
) -> None:
    """Assert structural evidence for one managed apply live gate."""

    assert proposal_response
    assert proposal_response["status"] == "succeeded"
    assert proposal_response["patch_artifacts"]
    assert proposal_response["evidence"]
    assert apply_response
    assert apply_response["status"] == "succeeded"
    assert apply_response["apply_package_id"]
    assert apply_response["applied_files"]
    assert apply_response["changed_files"]
    assert apply_response["validation"]["status"] == "succeeded"
    assert apply_response["apply_workspace_ref"]["kind"] == (
        "managed_apply_workspace"
    )

    evidence_paths = [
        str(evidence_item["path"])
        for evidence_item in proposal_response["evidence"]
    ]
    for expected_path in gate["expected_evidence_paths"]:
        assert any(expected_path in path for path in evidence_paths)

    applied_paths = [str(path) for path in apply_response["applied_files"]]
    for expected_path in gate["expected_applied_paths"]:
        assert any(expected_path in path for path in applied_paths)

    gate_workspace = WORKSPACE_ROOT / gate["gate_id"]
    managed_paths = _managed_applied_paths(
        workspace_root=gate_workspace,
        apply_response=apply_response,
    )
    assert managed_paths
    for path in managed_paths:
        assert path.is_file()

    raw_response = json.dumps(
        {
            "proposal": proposal_response,
            "apply": apply_response,
        },
        ensure_ascii=False,
    )
    copied_source_root = gate_workspace / "source"
    assert str(copied_source_root.resolve()) not in raw_response
    assert str(gate_workspace.resolve()) not in raw_response
    assert "stdout" not in apply_response
    assert "stderr" not in apply_response
    assert "exit_code" not in apply_response

    raw_evidence_path = TRACE_ROOT / f"{gate['gate_id']}_raw_evidence.json"
    raw_evidence = json.loads(raw_evidence_path.read_text(encoding="utf-8"))
    assert raw_evidence["source_tree_unchanged"] is True
    assert gate["objective"]
    assert gate["existing_code_base"]
    assert gate["modification_instruction"]
    assert gate["expected_state"]
    assert gate["pass_criteria"]
