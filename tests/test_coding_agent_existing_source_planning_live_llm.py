"""Live LLM gates for existing-source planning quality."""

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

TRACE_ROOT = Path("test_artifacts/llm_traces/coding_agent_existing_source_planning")
WORKSPACE_ROOT = Path("test_artifacts/coding_agent_existing_source_planning_workspace")
FIXTURE_ROOT = Path("tests/fixtures/coding_agent_existing_source_gates")
MAX_ANSWER_CHARS = 5000
MAX_ARTIFACT_CHARS = 64000

REQUIRED_PLANNING_TRACE_MARKERS = [
    "modifying:file_plan_ready",
    "modifying_pm:decision=",
    "modifying_pm:programmer_task=",
    "modifying_pm:sufficiency=",
]


class ExistingSourcePlanningGate(TypedDict):
    """One live source-planning gate contract."""

    gate_id: str
    title: str
    fixture_dir: str
    instruction: str
    expected_evidence_paths: list[str]
    expected_owner_paths: list[str]
    expected_companion_paths: list[str]
    behavior_rubric: list[str]
    forbidden_failure_modes: list[str]


GATE_01: ExistingSourcePlanningGate = {
    "gate_id": "planning_gate_01_cli_json_owner",
    "title": "CLI JSON output owner selection",
    "fixture_dir": "gate_01_log_counter",
    "instruction": (
        "Modify the existing log counter CLI to add a --json flag. When "
        "--json is provided, stdout must be valid JSON containing the same "
        "severity counts and the skipped-line count. The default text output "
        "must remain unchanged. Use only the Python standard library and "
        "update focused tests for the new JSON mode."
    ),
    "expected_evidence_paths": [
        "log_counter.py",
        "tests/test_log_counter.py",
    ],
    "expected_owner_paths": ["log_counter.py"],
    "expected_companion_paths": ["tests/test_log_counter.py"],
    "behavior_rubric": [
        "The planning flow identifies log_counter.py as the source owner.",
        "Tests are treated as focused companion updates, not the only change.",
        "The patch proposal remains reviewable and source-backed.",
    ],
    "forbidden_failure_modes": [
        "A replacement CLI script is created instead of editing log_counter.py.",
        "Only tests or README are changed while runtime behavior is untouched.",
        "Planning trace lacks File Agent or modifying PM decisions.",
    ],
}

GATE_02: ExistingSourcePlanningGate = {
    "gate_id": "planning_gate_02_jsonl_utility",
    "title": "JSONL utility multi-file planning",
    "fixture_dir": "gate_02_contacts_jsonl_to_csv",
    "instruction": (
        "Modify the existing JSONL-to-CSV utility so --fields defines the "
        "exact CSV column order, missing fields are written as blank cells, "
        "malformed JSON is reported with 1-based line numbers, default "
        "behavior continues past malformed lines, and --strict fails fast on "
        "the first malformed line. Use only the Python standard library. "
        "Update focused converter, CLI, and README coverage."
    ),
    "expected_evidence_paths": [
        "contacts_jsonl_to_csv/converter.py",
        "contacts_jsonl_to_csv/cli.py",
        "tests/test_converter.py",
        "README.md",
    ],
    "expected_owner_paths": [
        "contacts_jsonl_to_csv/converter.py",
        "contacts_jsonl_to_csv/cli.py",
    ],
    "expected_companion_paths": ["tests/test_converter.py", "README.md"],
    "behavior_rubric": [
        "The planning flow separates converter semantics from CLI wiring.",
        "Tests and README are companion updates to runtime behavior.",
        "The proposed owner paths stay inside the existing package.",
    ],
    "forbidden_failure_modes": [
        "Field ordering is handled only in CLI formatting.",
        "Malformed-line behavior is documented but not planned for source.",
        "Planning trace lacks owned and read-only path separation.",
    ],
}

GATE_03: ExistingSourcePlanningGate = {
    "gate_id": "planning_gate_03_markdown_parser",
    "title": "Markdown parser edge-case planning",
    "fixture_dir": "gate_03_markdown_link_checker",
    "instruction": (
        "Modify the existing Markdown link checker to ignore links inside "
        "fenced code blocks and HTML comments. Support duplicate heading "
        "anchors with GitHub-style suffixes when resolving links: the first "
        "duplicate heading keeps the base anchor, the second becomes base-1, "
        "the third becomes base-2, and so on. Update focused parser/scanner "
        "tests and keep normal links outside comments and fences working."
    ),
    "expected_evidence_paths": [
        "mdlinkcheck/anchors.py",
        "mdlinkcheck/scanner.py",
        "tests/test_anchors.py",
        "tests/test_scanner.py",
    ],
    "expected_owner_paths": [
        "mdlinkcheck/anchors.py",
        "mdlinkcheck/scanner.py",
    ],
    "expected_companion_paths": [
        "tests/test_anchors.py",
        "tests/test_scanner.py",
    ],
    "behavior_rubric": [
        "The planning flow assigns parser/scanner ownership before CLI work.",
        "Anchor suffix behavior and scanner filtering are planned together.",
        "Existing edge-case tests are used as companion context.",
    ],
    "forbidden_failure_modes": [
        "CLI suppression is planned instead of parser/scanner behavior.",
        "Only one of fences, comments, or duplicate anchors is considered.",
        "Planning trace lacks source-owner PM rationale.",
    ],
}

GATE_04: ExistingSourcePlanningGate = {
    "gate_id": "planning_gate_04_issue_tracker",
    "title": "Issue tracker cross-layer planning",
    "fixture_dir": "gate_04_issue_tracker_soft_delete",
    "instruction": (
        "Modify the existing issue tracker to implement soft delete. Deleting "
        "an issue should mark it archived instead of removing it from "
        "storage. Normal list results should hide archived issues. "
        "Single-item lookup should return not found for archived issues. "
        "List should support an include_archived option. Update store/API "
        "tests and README. Keep behavior coherent with the existing project "
        "style and avoid compatibility wrappers around the old hard-delete "
        "semantics."
    ),
    "expected_evidence_paths": [
        "issue_tracker/models.py",
        "issue_tracker/store.py",
        "issue_tracker/api.py",
        "tests/test_store.py",
        "tests/test_api.py",
        "README.md",
    ],
    "expected_owner_paths": [
        "issue_tracker/models.py",
        "issue_tracker/store.py",
        "issue_tracker/api.py",
    ],
    "expected_companion_paths": [
        "tests/test_store.py",
        "tests/test_api.py",
        "README.md",
    ],
    "behavior_rubric": [
        "The planning flow coordinates model, store, API, tests, and docs.",
        "Archived-state ownership is assigned to runtime source files.",
        "Tests and README follow the source behavior instead of driving it.",
    ],
    "forbidden_failure_modes": [
        "A compatibility hard-delete shim or alias is planned.",
        "API and store behavior are planned independently with divergent rules.",
        "Planning trace lacks multi-file PM decomposition.",
    ],
}

GATE_05: ExistingSourcePlanningGate = {
    "gate_id": "planning_gate_05_inventory_cache",
    "title": "Inventory fetch cache planning",
    "fixture_dir": "gate_05_inventory_sync_fetch_cache",
    "instruction": (
        "Modify the existing inventory sync project to add timeout and retry "
        "handling for vendor page fetches, add a file-backed response cache, "
        "expose CLI flags --cache-dir, --refresh-cache, and --timeout, update "
        "mocked HTTP tests, and document the workflow. Use only the Python "
        "standard library. Do not run real network calls in tests."
    ),
    "expected_evidence_paths": [
        "inventory_sync/fetch.py",
        "inventory_sync/cli.py",
        "tests/test_fetch.py",
        "tests/test_cli.py",
        "README.md",
    ],
    "expected_owner_paths": [
        "inventory_sync/fetch.py",
        "inventory_sync/cli.py",
    ],
    "expected_companion_paths": [
        "tests/test_fetch.py",
        "tests/test_cli.py",
        "README.md",
    ],
    "behavior_rubric": [
        "The planning flow separates fetch behavior from CLI flag wiring.",
        "Any new helper file is justified by existing source imports.",
        "Tests remain mocked and are planned as companion verification.",
    ],
    "forbidden_failure_modes": [
        "A replacement end-to-end script bypasses existing modules.",
        "Real network calls are planned for tests.",
        "Cache behavior is documented but not planned in fetch or CLI source.",
    ],
}


async def test_existing_source_planning_cli_json_owner_gate() -> None:
    """Run the simple CLI owner-planning gate."""

    response = await _run_planning_gate(GATE_01)
    _assert_planning_gate_response(gate=GATE_01, response=response)


async def test_existing_source_planning_jsonl_utility_gate() -> None:
    """Run the small multi-file utility planning gate."""

    response = await _run_planning_gate(GATE_02)
    _assert_planning_gate_response(gate=GATE_02, response=response)


async def test_existing_source_planning_markdown_parser_gate() -> None:
    """Run the parser edge-case planning gate."""

    response = await _run_planning_gate(GATE_03)
    _assert_planning_gate_response(gate=GATE_03, response=response)


async def test_existing_source_planning_issue_tracker_gate() -> None:
    """Run the cross-layer behavior planning gate."""

    response = await _run_planning_gate(GATE_04)
    _assert_planning_gate_response(gate=GATE_04, response=response)


@pytest.mark.xfail(
    reason=(
        "Accepted diagnostic failure: local model omits required fetch/CLI "
        "test artifacts after bounded contract repair."
    ),
    strict=False,
)
async def test_existing_source_planning_inventory_cache_gate() -> None:
    """Run the retained diagnostic hard mixed-source planning gate."""

    response = await _run_planning_gate(GATE_05)
    _assert_planning_gate_response(gate=GATE_05, response=response)


async def _run_planning_gate(
    gate: ExistingSourcePlanningGate,
) -> dict[str, Any]:
    """Run one live planning gate and persist raw evidence."""

    gate_id = gate["gate_id"]
    workspace_root = _reset_gate_workspace(gate_id)
    fixture_root = FIXTURE_ROOT / gate["fixture_dir"]
    source_root = workspace_root / "source"
    shutil.copytree(fixture_root, source_root)
    _initialize_fixture_git_checkout(source_root, gate_id)

    before_hashes = _hash_source_files(source_root)
    response = await propose_code_change({
        "question": gate["instruction"],
        "local_root_hint": str(source_root),
        "workspace_root": str(workspace_root),
        "session_id": gate_id,
        "preferred_language": "English",
        "max_answer_chars": MAX_ANSWER_CHARS,
        "max_artifact_chars": MAX_ARTIFACT_CHARS,
    })
    after_hashes = _hash_source_files(source_root)

    materialized_paths = _materialized_artifact_paths(
        gate_id=gate_id,
        response=response,
    )
    trace_payload = {
        "gate": gate,
        "model_routes": [
            "CODING_AGENT_PM_LLM",
            "CODING_AGENT_PROGRAMMER_LLM",
        ],
        "planning_trace_requirements": REQUIRED_PLANNING_TRACE_MARKERS,
        "source_root": str(source_root),
        "workspace_root": str(workspace_root),
        "source_hashes_before": before_hashes,
        "source_hashes_after": after_hashes,
        "source_tree_unchanged": before_hashes == after_hashes,
        "response": response,
        "materialized_artifacts": materialized_paths,
        "human_review_required": True,
    }
    trace_path = TRACE_ROOT / f"{gate_id}_raw_evidence.json"
    _write_json(trace_path, trace_payload)

    print(f"gate_id={gate_id}")
    print(f"raw_evidence_path={trace_path}")
    for path_text in materialized_paths:
        print(f"materialized_artifact={path_text}")

    return response


def _initialize_fixture_git_checkout(source_root: Path, gate_id: str) -> None:
    """Make a copied fixture satisfy the local checkout source contract."""

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
    exc_info: object,
) -> None:
    """Clear Windows read-only bits left by git object files and retry."""

    del exc_info
    if not callable(function):
        raise AssertionError(f"Cleanup callback was not callable for {path}.")
    os.chmod(path, 0o700)
    function(path)


def _hash_source_files(root: Path) -> dict[str, str]:
    """Hash non-git files in a source tree for non-mutation checks."""

    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        if relative_path.startswith(".git/"):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[relative_path] = digest
    return hashes


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write raw structured evidence for later human-authored review."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(encoded_payload, encoding="utf-8")


def _assert_planning_gate_response(
    *,
    gate: ExistingSourcePlanningGate,
    response: dict[str, Any],
) -> None:
    """Assert structural evidence required for planning gate review."""

    assert response
    assert response["status"] == "succeeded"
    assert response["answer_text"]
    assert response["patch_artifacts"]
    assert response["validation"]["files"]
    assert response["validation"]["sandbox_applied"] is True
    assert response["changed_files"]
    assert response["source_scope"]
    assert response["evidence"]

    _assert_expected_evidence_paths(gate=gate, response=response)
    _assert_expected_owner_paths(gate=gate, response=response)
    _assert_expected_companion_paths(gate=gate, response=response)
    _assert_required_trace_markers(response=response)
    _assert_materialized_paths_exist(gate=gate, response=response)
    _assert_public_response_is_sanitized(gate=gate, response=response)
    _assert_source_tree_unchanged(gate=gate)


def _assert_expected_evidence_paths(
    *,
    gate: ExistingSourcePlanningGate,
    response: dict[str, Any],
) -> None:
    """Require evidence for the paths that make the case realistic."""

    evidence_paths = [
        str(evidence_item["path"])
        for evidence_item in response["evidence"]
    ]
    for expected_path in gate["expected_evidence_paths"]:
        assert any(expected_path in path for path in evidence_paths)


def _assert_expected_owner_paths(
    *,
    gate: ExistingSourcePlanningGate,
    response: dict[str, Any],
) -> None:
    """Require at least one planned change to a source owner path."""

    changed_paths = _validated_changed_paths(response)
    for expected_path in gate["expected_owner_paths"]:
        assert any(expected_path in changed_path for changed_path in changed_paths)


def _assert_expected_companion_paths(
    *,
    gate: ExistingSourcePlanningGate,
    response: dict[str, Any],
) -> None:
    """Require at least one focused requested companion update."""

    changed_paths = _validated_changed_paths(response)
    for expected_path in gate["expected_companion_paths"]:
        assert any(expected_path in changed_path for changed_path in changed_paths)


def _validated_changed_paths(response: dict[str, Any]) -> list[str]:
    """Return paths that review materialization validated as changed."""

    validation = response["validation"]
    files = validation.get("files", [])
    return [str(path) for path in files]


def _assert_required_trace_markers(response: dict[str, Any]) -> None:
    """Require planning trace markers emitted by the PM-mediated path."""

    trace_summary = [
        str(trace_item)
        for trace_item in response["trace_summary"]
    ]
    for required_marker in REQUIRED_PLANNING_TRACE_MARKERS:
        assert any(
            trace_item.startswith(required_marker)
            for trace_item in trace_summary
        )


def _assert_materialized_paths_exist(
    *,
    gate: ExistingSourcePlanningGate,
    response: dict[str, Any],
) -> None:
    """Require materialized review files for human inspection."""

    materialized_paths = _materialized_artifact_paths(
        gate_id=gate["gate_id"],
        response=response,
    )
    assert materialized_paths
    for path_text in materialized_paths:
        assert Path(path_text).is_file()


def _assert_public_response_is_sanitized(
    *,
    gate: ExistingSourcePlanningGate,
    response: dict[str, Any],
) -> None:
    """Require public metadata to hide managed local roots."""

    copied_source_root = WORKSPACE_ROOT / gate["gate_id"] / "source"
    gate_workspace = WORKSPACE_ROOT / gate["gate_id"]
    raw_response = json.dumps(response, ensure_ascii=False)
    assert str(copied_source_root.resolve()) not in raw_response
    assert str(gate_workspace.resolve()) not in raw_response


def _assert_source_tree_unchanged(gate: ExistingSourcePlanningGate) -> None:
    """Require proposal generation to leave the source checkout unchanged."""

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
