"""Live E2E gates for coding-agent existing-source patch proposals."""

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

TRACE_ROOT = Path("test_artifacts/llm_traces/coding_agent_existing_source_e2e")
WORKSPACE_ROOT = Path("test_artifacts/coding_agent_existing_source_e2e_workspace")
FIXTURE_ROOT = Path("tests/fixtures/coding_agent_existing_source_gates")
MAX_ANSWER_CHARS = 5000
MAX_ARTIFACT_CHARS = 64000


class ExistingSourceGate(TypedDict):
    """One source-backed live gate contract for patch proposal review."""

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
    decomposition_requirements: list[str]
    expected_evidence_paths: list[str]


GATE_01: ExistingSourceGate = {
    "gate_id": "gate_01_log_counter_json",
    "title": "CLI JSON output toggle",
    "objective": (
        "Prove a bounded single-file CLI behavior extension with a focused "
        "existing test update and no replacement project."
    ),
    "fixture_dir": "gate_01_log_counter",
    "existing_code_base": (
        "A standard-library log_counter.py CLI with count_severities, "
        "format_summary, main, README text-output usage, and focused pytest "
        "coverage in tests/test_log_counter.py."
    ),
    "modification_instruction": (
        "Modify the existing log counter CLI to add a --json flag. When "
        "--json is provided, stdout must be valid JSON containing the same "
        "severity counts and the skipped-line count. The default text output "
        "must remain unchanged. Use only the Python standard library and "
        "update focused tests for the new JSON mode."
    ),
    "expected_state": [
        "log_counter.py still owns the CLI; no replacement script is added.",
        "Default invocation still prints the existing text summary format.",
        "The --json output contains DEBUG, INFO, WARNING, ERROR, CRITICAL, "
        "and skipped integer keys.",
        "Missing-file behavior and exit code remain unchanged.",
        "Focused tests prove JSON output parses and preserves the same counts.",
    ],
    "pass_criteria": [
        "The public response status is succeeded.",
        "Evidence cites log_counter.py and tests/test_log_counter.py.",
        "Patch artifacts modify existing files rather than only creating files.",
        "Review materialization contains proposed changed source and tests.",
        "The copied source tree remains byte-identical after proposal creation.",
        "The public response does not expose local roots, cache keys, raw "
        "traces, or worker internals.",
    ],
    "behavior_rubric": [
        "The proposal modifies existing CLI parsing instead of replacing the "
        "fixture project.",
        "JSON mode reuses the same count results as text mode.",
        "Default text output and missing-file behavior remain stable.",
        "The test update would fail if JSON is malformed or skips the "
        "skipped-line count.",
    ],
    "forbidden_failure_modes": [
        "A separate replacement script is created while log_counter.py is left "
        "unchanged.",
        "Default text output or missing-file exit behavior changes.",
        "The JSON output uses Python repr text or string counts instead of "
        "JSON integers.",
        "The proposed tests only check for a substring rather than parsing "
        "the JSON object.",
    ],
    "trace_required": [
        "Gate input and copied source identity.",
        "Raw public response and parsed patch proposal.",
        "Evidence paths proving source and test inspection.",
        "Materialized review paths and source tree hash comparison.",
    ],
    "decomposition_requirements": [
        "A single bounded programmer task may own the CLI/source behavior and "
        "focused test update.",
    ],
    "expected_evidence_paths": [
        "log_counter.py",
        "tests/test_log_counter.py",
    ],
}

GATE_02: ExistingSourceGate = {
    "gate_id": "gate_02_contacts_jsonl_errors",
    "title": "JSONL import error handling",
    "objective": (
        "Prove a small multi-file utility change where parser behavior, CLI "
        "behavior, tests, and documentation stay coherent."
    ),
    "fixture_dir": "gate_02_contacts_jsonl_to_csv",
    "existing_code_base": (
        "A contacts_jsonl_to_csv package with converter.py, cli.py, README, "
        "and basic converter/CLI tests. The baseline conversion has unstable "
        "requested field ordering and weak malformed-line reporting."
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
        "The existing package and CLI entrypoint remain in place.",
        "--fields id,name,email writes columns exactly in id,name,email order.",
        "Missing requested fields produce empty CSV cells.",
        "Malformed JSON reports a 1-based line number.",
        "Non-strict mode skips malformed lines and exits successfully.",
        "--strict aborts on the first malformed line and exits non-zero.",
        "Tests cover field order, missing fields, malformed lines, and strict "
        "failure.",
    ],
    "pass_criteria": [
        "The public response status is succeeded.",
        "Evidence cites converter.py, cli.py, tests, and README.",
        "Patch artifacts include existing source and test edits.",
        "New helpers remain inside the existing package boundary.",
        "Review materialization contains coherent converter, CLI, tests, and "
        "docs changes.",
        "The copied source tree remains byte-identical after proposal creation.",
    ],
    "behavior_rubric": [
        "Converter behavior preserves exact requested field order.",
        "Missing requested fields become blank CSV cells.",
        "Malformed JSON reporting includes a 1-based input line number.",
        "Non-strict mode continues with valid rows while strict mode fails "
        "fast.",
        "CLI tests and README match the converter behavior.",
    ],
    "forbidden_failure_modes": [
        "The solution sorts or normalizes user-supplied field order.",
        "Missing values are written as None or omitted inconsistently.",
        "Strict mode reports failure after writing a successful-looking "
        "partial conversion.",
        "Malformed-line handling is implemented only in CLI output while the "
        "converter remains weak.",
        "Non-standard-library dependencies are introduced.",
    ],
    "trace_required": [
        "Gate input and copied source identity.",
        "Raw public response and parsed patch proposal.",
        "Evidence paths for converter, CLI, tests, and README.",
        "Materialized review paths and source tree hash comparison.",
    ],
    "decomposition_requirements": [
        "Separate converter semantics from CLI wiring and README updates.",
        "Cite existing converter tests before proposing test edits.",
    ],
    "expected_evidence_paths": [
        "contacts_jsonl_to_csv/converter.py",
        "contacts_jsonl_to_csv/cli.py",
        "tests/test_converter.py",
        "README.md",
    ],
}

GATE_03: ExistingSourceGate = {
    "gate_id": "gate_03_markdown_parser_edges",
    "title": "Markdown link checker parser upgrade",
    "objective": (
        "Prove a parser-owned existing-source change that avoids CLI-only "
        "suppression and updates edge-case tests."
    ),
    "fixture_dir": "gate_03_markdown_link_checker",
    "existing_code_base": (
        "A mdlinkcheck package with anchors.py, scanner.py, cli.py, README, "
        "and focused anchor/scanner tests. The baseline scanner treats links "
        "inside fences and comments as real links and cannot resolve "
        "GitHub-style duplicate heading suffixes."
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
        "Parser/scanner logic owns fenced-code and HTML-comment exclusions.",
        "Triple-backtick and triple-tilde fences work across multiple lines.",
        "Multi-line HTML comments are ignored by link scanning.",
        "Normal inline links outside comments and fences are still checked.",
        "Duplicate headings produce base, base-1, base-2 anchors.",
        "Links to duplicate suffixed anchors resolve successfully.",
    ],
    "pass_criteria": [
        "The public response status is succeeded.",
        "Evidence cites anchors.py, scanner.py, and existing tests.",
        "Patch artifacts modify parser/scanner logic and tests.",
        "Review materialization shows parser-level handling.",
        "The copied source tree remains byte-identical after proposal creation.",
        "Production code does not hardcode fixture filenames or headings.",
    ],
    "behavior_rubric": [
        "Parser or scanner logic filters fenced-code and HTML-comment content "
        "before link checks.",
        "Duplicate heading anchors resolve as base, base-1, base-2, and later "
        "suffixes.",
        "Normal links outside ignored regions still work.",
        "Tests cover fences, comments, outside links, and duplicate anchors.",
    ],
    "forbidden_failure_modes": [
        "The CLI suppresses output instead of fixing parser/scanner behavior.",
        "Only triple-backtick fences are handled while triple-tilde fences are "
        "ignored.",
        "HTML comments spanning multiple lines still produce link reports.",
        "The code hardcodes fixture headings, filenames, or expected test "
        "strings.",
        "Normal valid relative links regress.",
    ],
    "trace_required": [
        "Gate input and copied source identity.",
        "Raw public response and parsed patch proposal.",
        "Evidence paths for anchors, scanner, and existing tests.",
        "Materialized review paths and source tree hash comparison.",
    ],
    "decomposition_requirements": [
        "Identify parser/scanner ownership before proposing CLI changes.",
        "Coordinate anchor suffix behavior with scanner link resolution.",
    ],
    "expected_evidence_paths": [
        "mdlinkcheck/anchors.py",
        "mdlinkcheck/scanner.py",
        "tests/test_anchors.py",
        "tests/test_scanner.py",
    ],
}

GATE_04: ExistingSourceGate = {
    "gate_id": "gate_04_issue_tracker_soft_delete",
    "title": "Issue tracker soft delete",
    "objective": (
        "Prove a cross-layer semantic change where model, store, API, tests, "
        "and docs move together without compatibility shims."
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
        "Issue has an archived state without breaking existing creation.",
        "delete_issue marks active issues archived and returns True.",
        "Deleting a missing issue returns False.",
        "Normal get_issue returns None for archived issues.",
        "Normal list_issues excludes archived issues.",
        "list_issues(include_archived=True) includes archived issues.",
        "API functions expose the same soft-delete semantics as the store.",
        "Tests and README describe the new behavior.",
    ],
    "pass_criteria": [
        "The public response status is succeeded.",
        "Evidence cites models.py, store.py, api.py, tests, and README.",
        "Patch artifacts modify model, store, API, tests, and docs coherently.",
        "No compatibility shim, alias method, fallback mapper, or parallel "
        "hard-delete vocabulary is introduced.",
        "The copied source tree remains byte-identical after proposal creation.",
    ],
    "behavior_rubric": [
        "Model state, store behavior, API behavior, tests, and README move "
        "together.",
        "Archived issues remain stored but are hidden from normal lookup and "
        "listing.",
        "include_archived exposes archived issues without changing default "
        "list behavior.",
        "Missing-delete behavior remains consistent with the baseline style.",
    ],
    "forbidden_failure_modes": [
        "The old hard-delete behavior is preserved behind a compatibility "
        "wrapper or alias.",
        "Archived issues appear in default list results.",
        "Single-item lookup returns archived issues by default.",
        "API and store semantics diverge.",
        "Tests are updated to match implementation quirks instead of the "
        "soft-delete contract.",
    ],
    "trace_required": [
        "Gate input and copied source identity.",
        "Raw public response and parsed patch proposal.",
        "Evidence paths for model, store, API, tests, and README.",
        "Materialized review paths and source tree hash comparison.",
    ],
    "decomposition_requirements": [
        "Plan model, store, API, tests, and README as one coherent domain "
        "change.",
        "Avoid introducing old hard-delete vocabulary as a parallel path.",
    ],
    "expected_evidence_paths": [
        "issue_tracker/models.py",
        "issue_tracker/store.py",
        "issue_tracker/api.py",
        "tests/test_store.py",
        "tests/test_api.py",
        "README.md",
    ],
}

GATE_05: ExistingSourceGate = {
    "gate_id": "gate_05_inventory_fetch_cache",
    "title": "Inventory vendor fetch cache",
    "objective": (
        "Prove a hard mixed proposal with existing-file modifications plus a "
        "justified new helper file, cache behavior, CLI flags, mocked HTTP "
        "tests, and documentation."
    ),
    "fixture_dir": "gate_05_inventory_sync_fetch_cache",
    "existing_code_base": (
        "An inventory_sync package with CSV reading, urllib fetch, HTML "
        "metadata extraction, report writing, CLI, README, and mocked HTTP "
        "tests. The baseline fetch path has no explicit timeout, retry, or "
        "cache support."
    ),
    "modification_instruction": (
        "Modify the existing inventory sync project to add timeout and retry "
        "handling for vendor page fetches, add a file-backed response cache, "
        "expose CLI flags --cache-dir, --refresh-cache, and --timeout, update "
        "mocked HTTP tests, and document the workflow. Use only the Python "
        "standard library. Do not run real network calls in tests."
    ),
    "expected_state": [
        "The fetch layer accepts a timeout and passes it to urlopen.",
        "Transient fetch failures are retried with a deterministic small count.",
        "Cache keys are deterministic and filesystem-safe.",
        "--cache-dir enables file-backed raw HTML caching.",
        "Existing cache entries avoid network calls unless --refresh-cache is "
        "provided.",
        "--timeout wires through CLI parsing into the fetch layer.",
        "CSV parsing, HTML extraction, and report writing keep their current "
        "module responsibilities.",
        "Mocked HTTP tests cover cache miss, cache hit, refresh, timeout, and "
        "retry.",
        "README documents cache, refresh, timeout, and workflow behavior.",
    ],
    "pass_criteria": [
        "The public response status is succeeded.",
        "Evidence cites fetch.py, cli.py, existing tests, and README.",
        "Patch artifacts include existing-source edits and may include one "
        "focused new helper file.",
        "Tests remain mocked and do not add real network calls.",
        "Existing parser/report responsibilities are not bypassed by a "
        "replacement script.",
        "The copied source tree remains byte-identical after proposal creation.",
    ],
    "behavior_rubric": [
        "Fetch behavior owns timeout propagation, retry, and cache read/write "
        "semantics.",
        "CLI flags wire into the existing fetch/report workflow.",
        "Cache keys are deterministic and safe for filesystem paths.",
        "Tests mock network behavior for miss, hit, refresh, timeout, and "
        "retry cases.",
        "README describes the cache, refresh, timeout, and workflow behavior.",
    ],
    "forbidden_failure_modes": [
        "A replacement end-to-end script bypasses csv_io, html_extract, or "
        "report responsibilities.",
        "Tests perform real network calls.",
        "Cache behavior is only documented but not wired into fetch or CLI.",
        "Timeout is parsed by the CLI but not passed to urlopen.",
        "Retry count becomes a new user-facing flag without a requirement.",
        "A new helper file is created without explicit imports from existing "
        "modules.",
    ],
    "trace_required": [
        "Gate input and copied source identity.",
        "Raw public response and parsed patch proposal.",
        "Evidence paths for fetch, CLI, tests, README, and adjacent report "
        "or CSV ownership.",
        "Materialized review paths and source tree hash comparison.",
    ],
    "decomposition_requirements": [
        "Decompose fetch timeout, retry, and cache integration separately "
        "from CLI wiring.",
        "Plan mocked HTTP tests for cache miss, cache hit, refresh-cache, "
        "timeout propagation, and transient retry.",
        "Update README after source and test behavior are defined.",
        "Justify any new cache helper module by naming its interface and "
        "consuming imports.",
    ],
    "expected_evidence_paths": [
        "inventory_sync/fetch.py",
        "inventory_sync/cli.py",
        "tests/test_fetch.py",
        "tests/test_cli.py",
        "README.md",
    ],
}


async def test_existing_source_gate_01_cli_json_output_toggle() -> None:
    """Run the single-file CLI modification gate."""

    response = await _run_existing_source_gate(GATE_01)
    _assert_reviewable_existing_source_gate(gate=GATE_01, response=response)


async def test_existing_source_gate_02_jsonl_error_handling() -> None:
    """Run the small multi-file JSONL utility modification gate."""

    response = await _run_existing_source_gate(GATE_02)
    _assert_reviewable_existing_source_gate(gate=GATE_02, response=response)


async def test_existing_source_gate_03_markdown_parser_upgrade() -> None:
    """Run the Markdown parser edge-case modification gate."""

    response = await _run_existing_source_gate(GATE_03)
    _assert_reviewable_existing_source_gate(gate=GATE_03, response=response)


async def test_existing_source_gate_04_issue_tracker_soft_delete() -> None:
    """Run the cross-layer issue tracker modification gate."""

    response = await _run_existing_source_gate(GATE_04)
    _assert_reviewable_existing_source_gate(gate=GATE_04, response=response)


async def test_existing_source_gate_05_inventory_fetch_cache() -> None:
    """Run the mixed existing-file and new-file modification gate."""

    response = await _run_existing_source_gate(GATE_05)
    _assert_reviewable_existing_source_gate(gate=GATE_05, response=response)


async def _run_existing_source_gate(
    gate: ExistingSourceGate,
) -> dict[str, Any]:
    """Run one source-backed gate and persist raw evidence for review."""

    gate_id = gate["gate_id"]
    workspace_root = _reset_gate_workspace(gate_id)
    fixture_root = FIXTURE_ROOT / gate["fixture_dir"]
    source_root = workspace_root / "source"
    shutil.copytree(fixture_root, source_root)
    _initialize_fixture_git_checkout(source_root, gate_id)

    before_hashes = _hash_tree(source_root)
    response = await propose_code_change({
        "question": gate["modification_instruction"],
        "local_root_hint": str(source_root),
        "workspace_root": str(workspace_root),
        "session_id": gate_id,
        "preferred_language": "English",
        "max_answer_chars": MAX_ANSWER_CHARS,
        "max_artifact_chars": MAX_ARTIFACT_CHARS,
    })
    after_hashes = _hash_tree(source_root)

    materialized_paths = _materialized_artifact_paths(
        gate_id=gate_id,
        response=response,
    )
    review_contract = {
        "behavior_rubric": gate["behavior_rubric"],
        "forbidden_failure_modes": gate["forbidden_failure_modes"],
        "trace_required": gate["trace_required"],
        "decomposition_requirements": gate["decomposition_requirements"],
    }
    trace_payload = {
        "gate": gate,
        "review_contract": review_contract,
        "source_root": str(source_root),
        "workspace_root": str(workspace_root),
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "source_tree_unchanged": before_hashes == after_hashes,
        "response": response,
        "materialized_artifacts": materialized_paths,
    }
    trace_path = TRACE_ROOT / f"{gate_id}_raw_evidence.json"
    _write_json(trace_path, trace_payload)

    print(f"gate_id={gate_id}")
    print(f"raw_evidence_path={trace_path}")
    for path_text in materialized_paths:
        print(f"materialized_artifact={path_text}")

    return response


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
        shutil.rmtree(gate_workspace, onexc=_retry_remove_readonly)
    gate_workspace.mkdir(parents=True, exist_ok=True)

    return gate_workspace


def _retry_remove_readonly(function: object, path: str, exc: BaseException) -> None:
    """Clear Windows read-only bits left by git object files and retry."""

    del exc
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


def _assert_reviewable_existing_source_gate(
    *,
    gate: ExistingSourceGate,
    response: dict[str, Any],
) -> None:
    """Assert only structural evidence required for source-backed review."""

    assert response
    assert response["status"] == "succeeded"
    assert response["answer_text"]
    assert response["patch_artifacts"]
    assert response["validation"]["files"]
    assert response["validation"]["sandbox_applied"] is True
    assert response["changed_files"]
    assert response["source_scope"]
    assert response["evidence"]
    assert gate["behavior_rubric"]
    assert gate["forbidden_failure_modes"]
    assert gate["trace_required"]
    assert gate["decomposition_requirements"]

    evidence_paths = [
        str(evidence_item["path"])
        for evidence_item in response["evidence"]
    ]
    for expected_path in gate["expected_evidence_paths"]:
        assert any(expected_path in path for path in evidence_paths)

    materialized_paths = _materialized_artifact_paths(
        gate_id=gate["gate_id"],
        response=response,
    )
    assert materialized_paths
    for path_text in materialized_paths:
        assert Path(path_text).is_file()

    copied_source_root = WORKSPACE_ROOT / gate["gate_id"] / "source"
    raw_response = json.dumps(response, ensure_ascii=False)
    assert str(copied_source_root.resolve()) not in raw_response
    assert str((WORKSPACE_ROOT / gate["gate_id"]).resolve()) not in raw_response

    raw_evidence_path = TRACE_ROOT / f"{gate['gate_id']}_raw_evidence.json"
    raw_evidence = json.loads(raw_evidence_path.read_text(encoding="utf-8"))
    assert raw_evidence["source_tree_unchanged"] is True


def _materialized_artifact_paths(
    *,
    gate_id: str,
    response: dict[str, Any],
) -> list[str]:
    """Return direct review-package paths for proposed changed files."""

    validation = response["validation"]
    validation_files = validation.get("files", [])
    if not validation_files:
        return []

    gate_workspace = WORKSPACE_ROOT / gate_id
    candidate_roots: list[Path] = []
    for validation_dir_name in ("patch_validation", "writing_validation"):
        validation_root = gate_workspace / validation_dir_name
        if not validation_root.exists():
            continue
        roots = [
            path
            for path in validation_root.iterdir()
            if path.is_dir()
        ]
        candidate_roots.extend(roots)

    candidate_roots.sort(
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidate_roots:
        return []

    sandbox_root = candidate_roots[0]
    materialized_paths: list[str] = []
    for relative_path in validation_files:
        file_path = sandbox_root / str(relative_path)
        materialized_paths.append(str(file_path))

    return materialized_paths
