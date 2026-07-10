"""Deterministic contracts for the coding-agent benchmark seam."""

import asyncio
import json
from pathlib import Path

import pytest

from scripts.run_coding_agent_benchmark import (
    BENCHMARK_CASE_TIMEOUT_SECONDS,
    BENCHMARK_VERSION,
    RESULT_SCHEMA_VERSION,
    aggregate_benchmark_results,
    load_benchmark_cases,
    select_benchmark_case,
    run_benchmark_case,
    validate_benchmark_result,
)

BENCHMARK_MANIFEST_PATH = Path(
    "tests/fixtures/coding_agent_benchmark/cases.jsonl"
)


def test_benchmark_manifest_exists_with_thirty_cases() -> None:
    """The benchmark corpus is a versioned, fixed-size test artifact."""

    rows = load_benchmark_cases(BENCHMARK_MANIFEST_PATH)

    assert len(rows) == 30
    assert {row["benchmark_version"] for row in rows} == {BENCHMARK_VERSION}
    assert len({row["category"] for row in rows}) == 10


def test_benchmark_case_selection_returns_one_pinned_fixture() -> None:
    """The CLI selector cannot run multiple or unpinned benchmark cases."""

    cases = load_benchmark_cases(BENCHMARK_MANIFEST_PATH)
    selected = select_benchmark_case(cases, "source_backed_preflight_bugfix")

    assert selected["entrypoint"] == "coding_run"
    assert Path(selected["fixture_ref"]).is_relative_to("tests/fixtures")
    with pytest.raises(ValueError, match="unknown"):
        select_benchmark_case(cases, "missing-case")


def test_benchmark_result_schema_preserves_hidden_evaluator_boundary() -> None:
    """Result rows record outcomes without exposing manifest evaluator internals."""

    result = _benchmark_result(case_id="source_backed_preflight_bugfix")

    validate_benchmark_result(result)

    serialized = json.dumps(result, ensure_ascii=False)
    assert "required_final_status" not in serialized
    assert result["token_usage"] is None


def test_benchmark_aggregate_reads_existing_results_without_running_cases(
    tmp_path: Path,
) -> None:
    """Aggregation is offline and counts only validated stored result rows."""

    first = _benchmark_result(case_id="case-one", category="bug_fix")
    second = _benchmark_result(
        case_id="case-two",
        category="environment_blocker",
        status="blocked",
    )
    (tmp_path / "case-one.json").write_text(
        json.dumps(first),
        encoding="utf-8",
    )
    (tmp_path / "case-two.json").write_text(
        json.dumps(second),
        encoding="utf-8",
    )

    aggregate = aggregate_benchmark_results(tmp_path)

    assert aggregate == {
        "schema_version": "coding_agent_benchmark_aggregate.v1",
        "benchmark_version": BENCHMARK_VERSION,
        "case_count": 2,
        "by_status": {"blocked": 1, "failed": 0, "passed": 1},
        "by_category": {"bug_fix": 1, "environment_blocker": 1},
    }


@pytest.mark.asyncio
async def test_benchmark_timeout_writes_blocked_result_and_partial_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stalled public run records bounded benchmark evidence."""

    from scripts import run_coding_agent_benchmark as benchmark

    async def stalled_start(request: dict[str, object]) -> dict[str, object]:
        workspace_root = Path(str(request["workspace_root"]))
        run_path = workspace_root / "coding_runs" / "partial-run"
        run_path.mkdir(parents=True)
        (run_path / "run.json").write_text("{}", encoding="utf-8")
        await asyncio.sleep(0.1)
        return {"status": "awaiting_approval", "run_id": "partial-run"}

    cases = load_benchmark_cases(BENCHMARK_MANIFEST_PATH)
    case = select_benchmark_case(cases, "source_backed_preflight_bugfix")
    monkeypatch.setattr(benchmark, "start_coding_run", stalled_start)
    monkeypatch.setattr(benchmark, "BENCHMARK_CASE_TIMEOUT_SECONDS", 0.01)

    result = await run_benchmark_case(
        case,
        repository_root=Path.cwd(),
        results_directory=tmp_path,
    )

    assert BENCHMARK_CASE_TIMEOUT_SECONDS > 0
    assert result["status"] == "blocked"
    assert result["final_run_status"] == "timeout"
    assert result["evaluator"]["status"] == "not_applicable"
    assert result["trace_paths"]


def _benchmark_result(
    *,
    case_id: str,
    category: str = "bug_fix",
    status: str = "passed",
) -> dict[str, object]:
    """Build a complete public result row for deterministic schema checks."""

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "case_id": case_id,
        "category": category,
        "engine_id": "pipeline_v1",
        "routes": [{"route_name": "coding_run", "model": "configured"}],
        "status": status,
        "entrypoint": "coding_run",
        "elapsed_ms": 12,
        "llm_call_count": 2,
        "token_usage": None,
        "final_run_status": "completed",
        "evaluator": {"status": "passed", "checks": ["status matched"]},
        "trace_paths": ["test_artifacts/trace.json"],
        "notes": ["provider token usage was unavailable"],
    }
    return result
