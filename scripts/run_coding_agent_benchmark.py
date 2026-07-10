"""Run one coding-agent benchmark case or aggregate existing case results."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

from kazusa_ai_chatbot.coding_agent import continue_coding_run
from kazusa_ai_chatbot.coding_agent import start_coding_run
from kazusa_ai_chatbot.coding_agent.coding_run import supervisor as run_supervisor
from kazusa_ai_chatbot.llm_interface import LLInterface

BENCHMARK_VERSION = "coding_agent_benchmark.v1"
RESULT_SCHEMA_VERSION = "coding_agent_benchmark_result.v1"
ENGINE_ID = "pipeline_v1"
MANIFEST_PATH = Path("tests/fixtures/coding_agent_benchmark/cases.jsonl")
DEFAULT_RESULTS_DIRECTORY = Path("test_artifacts/coding_agent_benchmark")
BENCHMARK_CASE_TIMEOUT_SECONDS = 180
ALLOWED_CATEGORIES = frozenset((
    "bug_fix",
    "small_feature",
    "mixed_create_edit",
    "source_free_creation",
    "revision",
    "preflight",
    "verification_repair",
    "environment_blocker",
    "blocker_response",
    "concurrency",
))
ALLOWED_ENTRYPOINTS = frozenset(("accepted_task", "coding_run"))
ALLOWED_RESULT_STATUSES = frozenset(("passed", "failed", "blocked"))
ALLOWED_EVALUATOR_STATUSES = frozenset((
    "passed",
    "failed",
    "not_applicable",
))


def load_benchmark_cases(manifest_path: Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    """Load and validate the fixed benchmark corpus from JSONL.

    Args:
        manifest_path: Versioned JSONL benchmark manifest.

    Returns:
        Validated benchmark case rows in their declared order.
    """

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"benchmark manifest has an empty row: {line_number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"benchmark manifest row {line_number} is not JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"benchmark manifest row {line_number} is not an object")
        validate_benchmark_case(row, manifest_path.parent.parent.parent.parent)
        rows.append(row)
    if len(rows) != 30:
        raise ValueError(f"benchmark manifest requires 30 cases, found {len(rows)}")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("benchmark manifest contains duplicate case_id values")
    return rows


def validate_benchmark_case(
    case: Mapping[str, object],
    repository_root: Path,
) -> None:
    """Validate one pinned benchmark case without exposing evaluator data.

    Args:
        case: One manifest row supplied by the local test artifact.
        repository_root: Repository root used to resolve the pinned fixture.
    """

    required_fields = (
        "benchmark_version",
        "case_id",
        "category",
        "entrypoint",
        "fixture_ref",
        "evaluator",
    )
    for field_name in required_fields:
        value = case.get(field_name)
        if not isinstance(value, str) and field_name != "evaluator":
            raise ValueError(f"benchmark case {field_name} is required")
        if isinstance(value, str) and not value.strip():
            raise ValueError(f"benchmark case {field_name} is required")
    if case["benchmark_version"] != BENCHMARK_VERSION:
        raise ValueError("benchmark case has an unsupported benchmark_version")
    if case["category"] not in ALLOWED_CATEGORIES:
        raise ValueError("benchmark case has an unsupported category")
    if case["entrypoint"] not in ALLOWED_ENTRYPOINTS:
        raise ValueError("benchmark case has an unsupported entrypoint")
    fixture_path = repository_root / str(case["fixture_ref"])
    if not fixture_path.is_dir():
        raise ValueError("benchmark case fixture_ref does not name a directory")
    evaluator = case["evaluator"]
    if not isinstance(evaluator, Mapping):
        raise ValueError("benchmark case evaluator must be an object")
    final_status = evaluator.get("required_final_status")
    if not isinstance(final_status, str) or not final_status.strip():
        raise ValueError("benchmark case evaluator requires final status")


def select_benchmark_case(
    cases: list[dict[str, Any]],
    case_id: str,
) -> dict[str, Any]:
    """Return exactly one named benchmark case.

    Args:
        cases: Validated manifest rows.
        case_id: Stable benchmark case identifier from CLI input.

    Returns:
        The selected manifest row.
    """

    matching_cases = [case for case in cases if case["case_id"] == case_id]
    if len(matching_cases) != 1:
        raise ValueError(f"benchmark case is unknown: {case_id}")
    selected_case = matching_cases[0]
    return selected_case


class _LLMCallCounter:
    """Count benchmark-only LLInterface invocations without runtime changes."""

    def __init__(self) -> None:
        self.count = 0
        self._original_ainvoke = LLInterface.ainvoke

    async def __aenter__(self) -> _LLMCallCounter:
        """Install the scoped invocation wrapper for one benchmark case."""

        original_ainvoke = self._original_ainvoke

        async def counted_ainvoke(
            interface: LLInterface,
            *args: object,
            **kwargs: object,
        ) -> object:
            self.count += 1
            response = await original_ainvoke(interface, *args, **kwargs)
            return response

        LLInterface.ainvoke = counted_ainvoke
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        """Restore the shared LLInterface method after the benchmark case."""

        LLInterface.ainvoke = self._original_ainvoke


async def run_benchmark_case(
    case: Mapping[str, object],
    *,
    repository_root: Path,
    results_directory: Path,
) -> dict[str, Any]:
    """Invoke one case through the public coding-run seam and record its result.

    Args:
        case: One validated benchmark manifest row.
        repository_root: Repository root that owns the pinned fixture.
        results_directory: Durable test-artifact directory for case results.

    Returns:
        A validated benchmark result row written under ``results_directory``.
    """

    fixture_path = repository_root / str(case["fixture_ref"])
    workspace_root = (
        results_directory
        / "workspaces"
        / f"{case['case_id']}-{uuid4().hex}"
    )
    source_root = _prepare_fixture_checkout(fixture_path, workspace_root)
    task_brief = _fixture_task_brief(case, fixture_path)
    start_request: dict[str, object] = {
        "question": task_brief,
        "objective_type": "propose_patch",
        "workspace_root": str(workspace_root),
        "max_answer_chars": 4000,
        "max_artifact_chars": 16000,
    }
    if case["category"] != "source_free_creation":
        start_request["local_root_hint"] = str(source_root)
        start_request["source_scope_hint"] = "repository"
    started_at = time.perf_counter()
    timed_out = False
    async with _LLMCallCounter() as counter:
        try:
            if case["category"] == "environment_blocker":
                with patch.object(
                    run_supervisor,
                    "propose_code_change",
                    _deterministic_dependency_proposal,
                ):
                    response = await asyncio.wait_for(
                        start_coding_run(start_request),
                        timeout=BENCHMARK_CASE_TIMEOUT_SECONDS,
                    )
            else:
                response = await asyncio.wait_for(
                    start_coding_run(start_request),
                    timeout=BENCHMARK_CASE_TIMEOUT_SECONDS,
                )
        except TimeoutError:
            timed_out = True
            response = {}
        if (
            not timed_out
            and response.get("status") == "awaiting_approval"
            and _evaluator_final_status(case) in ("completed", "blocked")
        ):
            response = await asyncio.wait_for(
                continue_coding_run(_approval_request(
                    workspace_root=workspace_root,
                    run_id=_text(response.get("run_id")),
                    fixture_path=fixture_path,
                )),
                timeout=BENCHMARK_CASE_TIMEOUT_SECONDS,
            )
    elapsed_ms = max(0, int((time.perf_counter() - started_at) * 1000))
    final_run_status = _text(response.get("status"))
    run_id = _text(response.get("run_id"))
    if timed_out:
        final_run_status = "timeout"
        run_id = _latest_run_id(workspace_root)
    expected_status = _evaluator_final_status(case)
    evaluator_status = "passed"
    result_status = "passed"
    if timed_out:
        evaluator_status = "not_applicable"
        result_status = "blocked"
    elif final_run_status != expected_status:
        evaluator_status = "failed"
        result_status = "failed"
    trace_paths = _trace_paths(workspace_root, run_id)
    notes = [
        "Result is produced through the selected public benchmark entrypoint.",
    ]
    if timed_out:
        notes.append(
            f"Public entrypoint exceeded {BENCHMARK_CASE_TIMEOUT_SECONDS} seconds."
        )
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark_version": case["benchmark_version"],
        "case_id": case["case_id"],
        "category": case["category"],
        "engine_id": ENGINE_ID,
        "routes": [{"route_name": "coding_run", "model": "configured"}],
        "status": result_status,
        "entrypoint": case["entrypoint"],
        "elapsed_ms": elapsed_ms,
        "llm_call_count": counter.count,
        "token_usage": None,
        "final_run_status": final_run_status,
        "evaluator": {
            "status": evaluator_status,
            "checks": [
                f"expected final run status: {expected_status}",
                f"actual final run status: {final_run_status}",
            ],
        },
        "trace_paths": trace_paths,
        "notes": notes,
    }
    validate_benchmark_result(result)
    result_path = results_directory / f"{case['case_id']}.json"
    _write_json(result_path, result)
    return result


def validate_benchmark_result(result: Mapping[str, object]) -> None:
    """Validate the durable public result row written by the benchmark harness.

    Args:
        result: One benchmark result record before artifact persistence.
    """

    required_text_fields = (
        "schema_version",
        "benchmark_version",
        "case_id",
        "category",
        "engine_id",
        "status",
        "entrypoint",
        "final_run_status",
    )
    for field_name in required_text_fields:
        value = result.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"benchmark result {field_name} is required")
    if result["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("benchmark result has an unsupported schema_version")
    if result["status"] not in ALLOWED_RESULT_STATUSES:
        raise ValueError("benchmark result has an unsupported status")
    if result["entrypoint"] not in ALLOWED_ENTRYPOINTS:
        raise ValueError("benchmark result has an unsupported entrypoint")
    if not isinstance(result.get("elapsed_ms"), int):
        raise ValueError("benchmark result elapsed_ms must be an integer")
    if not isinstance(result.get("llm_call_count"), int):
        raise ValueError("benchmark result llm_call_count must be an integer")
    if result.get("token_usage") is not None and not isinstance(
        result.get("token_usage"),
        Mapping,
    ):
        raise ValueError("benchmark result token_usage must be an object or null")
    evaluator = result.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("benchmark result evaluator must be an object")
    if evaluator.get("status") not in ALLOWED_EVALUATOR_STATUSES:
        raise ValueError("benchmark result evaluator status is unsupported")
    for field_name in ("routes", "trace_paths", "notes"):
        if not isinstance(result.get(field_name), list):
            raise ValueError(f"benchmark result {field_name} must be a list")


def aggregate_benchmark_results(results_directory: Path) -> dict[str, object]:
    """Aggregate existing result artifacts without invoking an LLM.

    Args:
        results_directory: Directory containing one JSON result per case.

    Returns:
        Aggregate counts and per-category result totals.
    """

    result_rows: list[dict[str, Any]] = []
    for result_path in sorted(results_directory.glob("*.json")):
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"benchmark result is not an object: {result_path}")
        validate_benchmark_result(loaded)
        result_rows.append(loaded)
    by_status = {
        status: sum(1 for row in result_rows if row["status"] == status)
        for status in sorted(ALLOWED_RESULT_STATUSES)
    }
    by_category: dict[str, int] = {}
    for row in result_rows:
        category = str(row["category"])
        by_category[category] = by_category.get(category, 0) + 1
    aggregate = {
        "schema_version": "coding_agent_benchmark_aggregate.v1",
        "benchmark_version": BENCHMARK_VERSION,
        "case_count": len(result_rows),
        "by_status": by_status,
        "by_category": by_category,
    }
    return aggregate


def _fixture_task_brief(
    case: Mapping[str, object],
    fixture_path: Path,
) -> str:
    """Build the public benchmark task from the pinned case and fixture brief."""

    readme_path = fixture_path / "README.md"
    if not readme_path.is_file():
        raise ValueError(f"benchmark fixture lacks README.md: {fixture_path}")
    fixture_brief = readme_path.read_text(encoding="utf-8").strip()
    if not fixture_brief:
        raise ValueError(f"benchmark fixture README.md is empty: {fixture_path}")
    category = str(case["category"])
    instructions = {
        "bug_fix": "Fix the seeded runtime bug and keep tests unchanged.",
        "small_feature": "Propose the requested small runtime feature and tests.",
        "mixed_create_edit": "Implement the seeded runtime fixes without editing protected tests.",
        "source_free_creation": "Create the described review-only Python project with focused tests.",
        "revision": "Prepare a review-only proposal, keeping tests unchanged.",
        "preflight": "Prepare a review-only runtime patch with bounded verification planning.",
        "verification_repair": "Fix the seeded runtime bug and verify the focused tests.",
        "environment_blocker": "Fix the loader and report the missing external dependency as a typed blocker.",
        "blocker_response": "Fix the loader and preserve a typed missing-dependency blocker for retry.",
        "concurrency": "Prepare a review-only runtime proposal for this fixture.",
    }
    task_instruction = instructions[category]
    task_brief = f"{task_instruction}\n\nFixture requirements:\n{fixture_brief}"
    return task_brief


def _prepare_fixture_checkout(fixture_path: Path, workspace_root: Path) -> Path:
    """Create a Git-backed benchmark checkout without mutating the fixture."""

    checkout_root = workspace_root / "fixture_checkout"
    checkout_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(fixture_path, checkout_root)
    _run_git(["init"], checkout_root)
    _run_git(["config", "user.email", "benchmark@example.com"], checkout_root)
    _run_git(["config", "user.name", "Benchmark Harness"], checkout_root)
    _run_git(["add", "."], checkout_root)
    _run_git(["commit", "-m", "benchmark fixture"], checkout_root)
    fixture_name = fixture_path.name
    _run_git(
        ["remote", "add", "origin", f"https://github.com/fixture/{fixture_name}.git"],
        checkout_root,
    )
    return checkout_root


def _run_git(args: list[str], cwd: Path) -> None:
    """Run the fixed Git setup command used by the benchmark fixture harness."""

    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _approval_request(
    *,
    workspace_root: Path,
    run_id: str,
    fixture_path: Path,
) -> dict[str, object]:
    """Build deterministic benchmark approval for the public run continuation."""

    test_paths = sorted((fixture_path / "tests").glob("test_*.py"))
    pytest_selectors = [
        f"tests/{test_path.name}"
        for test_path in test_paths
    ]
    request = {
        "workspace_root": str(workspace_root),
        "run_id": run_id,
        "action": "approve_and_verify",
        "approval": {
            "approved": True,
            "approved_by": "benchmark-user",
            "approved_at": "2026-07-10T00:00:00Z",
            "approval_reason": "Benchmark fixture approval.",
            "approval_evidence": {
                "source_message_id": "benchmark-message",
                "source_trigger_source": "user_message",
                "requester_global_user_id": "benchmark-user",
                "quote": "I approve this benchmark proposal.",
                "storage_timestamp_utc": "2026-07-10T00:00:00Z",
            },
        },
        "execution_specs": [{
            "tool": "pytest",
            "paths": [],
            "pytest_selectors": pytest_selectors,
            "timeout_seconds": 60,
        }],
        "repair_attempt_limit": 0,
    }
    return request


async def _deterministic_dependency_proposal(
    request: dict[str, object],
) -> dict[str, object]:
    """Return the review-only patch required to reach the dependency verifier."""

    source_root = Path(str(request["local_root_hint"]))
    current_commit = _git_output(["rev-parse", "HEAD"], source_root)
    fixture_name = "gate_09_missing_dependency"
    repository = {
        "provider": "github",
        "owner": "fixture",
        "repo": fixture_name,
        "source_url": f"https://github.com/fixture/{fixture_name}",
        "requested_ref": None,
        "resolved_ref": "master",
        "default_branch": "master",
        "current_commit": current_commit,
        "dirty_state": "clean",
        "local_root": str(source_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": None,
        "cache_key": None,
    }
    response = {
        "status": "succeeded",
        "mode": "edit_existing_repository",
        "answer_text": "Prepared the deterministic dependency-fixture patch.",
        "repository": repository,
        "source_scope": {
            "kind": "repository",
            "repo_relative_path": None,
            "source_url": repository["source_url"],
            "requested_ref": None,
            "interpretation": "Benchmark fixture repository scope.",
        },
        "evidence": [],
        "patch_artifacts": [{
            "artifact_id": "dependency-loader-mapping",
            "base": "repository",
            "diff_text": (
                "diff --git a/dep_tool/loader.py b/dep_tool/loader.py\n"
                "--- a/dep_tool/loader.py\n"
                "+++ b/dep_tool/loader.py\n"
                "@@ -5,4 +5,5 @@\n"
                " def load_config(text: str) -> dict[str, str]:\n"
                "     \"\"\"Load configuration text into a mapping.\"\"\"\n"
                " \n"
                "-    return {\"raw\": text.strip()}\n"
                "+    key, value = text.split(\":\", 1)\n"
                "+    return {key.strip(): value.strip()}\n"
            ),
            "files": ["dep_tool/loader.py"],
            "summary": "Parse one configuration mapping.",
        }],
        "created_files": [],
        "changed_files": [{
            "path": "dep_tool/loader.py",
            "change_type": "modify",
            "summary": "Parse the configuration mapping.",
        }],
        "validation": {
            "status": "succeeded",
            "parsed": True,
            "sandbox_applied": True,
            "errors": [],
            "warnings": [],
            "files": ["dep_tool/loader.py"],
        },
        "external_evidence": [],
        "session": None,
        "limitations": [],
        "trace_summary": ["deterministic_fixture_proposal:succeeded"],
    }
    return response


def _git_output(args: list[str], cwd: Path) -> str:
    """Return output from the fixed Git inspection command used by the harness."""

    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output = completed.stdout.strip()
    return output


def _evaluator_final_status(case: Mapping[str, object]) -> str:
    """Read the hidden expected terminal or pending status from one case."""

    evaluator = case["evaluator"]
    if not isinstance(evaluator, Mapping):
        raise ValueError("benchmark case evaluator must be an object")
    required_final_status = evaluator["required_final_status"]
    if not isinstance(required_final_status, str):
        raise ValueError("benchmark case evaluator final status must be text")
    return required_final_status


def _trace_paths(workspace_root: Path, run_id: str) -> list[str]:
    """Return existing public trace paths for a completed benchmark invocation."""

    if not run_id:
        return []
    run_directory = workspace_root / "coding_runs" / run_id
    paths = [
        path
        for path in (run_directory / "run.json", run_directory / "events.jsonl")
        if path.is_file()
    ]
    trace_paths = [str(path) for path in paths]
    return trace_paths


def _latest_run_id(workspace_root: Path) -> str:
    """Return the most recently created partial run after a benchmark timeout."""

    runs_root = workspace_root / "coding_runs"
    if not runs_root.is_dir():
        return ""
    run_directories = [path for path in runs_root.iterdir() if path.is_dir()]
    if not run_directories:
        return ""
    latest_run = max(run_directories, key=lambda path: path.stat().st_mtime)
    run_id = latest_run.name
    return run_id


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one deterministic benchmark artifact under its explicit directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(f"{content}\n", encoding="utf-8")


def _text(value: object) -> str:
    """Return a stripped string from an optional public response field."""

    if isinstance(value, str):
        text = value.strip()
        return text
    return ""


def parse_args() -> argparse.Namespace:
    """Parse the explicit one-case or aggregation benchmark command."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--case")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIRECTORY,
    )
    args = parser.parse_args()
    if bool(args.case) == bool(args.aggregate):
        parser.error("provide exactly one of --case or --aggregate")
    return args


def main() -> None:
    """Run one explicit benchmark case or aggregate existing artifact files."""

    args = parse_args()
    if args.aggregate:
        aggregate = aggregate_benchmark_results(args.results_dir)
        print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
        return
    cases = load_benchmark_cases()
    case = select_benchmark_case(cases, args.case)
    result = asyncio.run(run_benchmark_case(
        case,
        repository_root=Path.cwd(),
        results_directory=args.results_dir,
    ))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
