"""Deterministic contracts for the coding-agent benchmark seam."""

import asyncio
import json
from pathlib import Path

import pytest

from scripts.run_coding_agent_benchmark import (
    BENCHMARK_CASE_TIMEOUT_SECONDS,
    BENCHMARK_VERSION,
    RESULT_SCHEMA_VERSION,
    _digest_engine_contract_records,
    _engine_contract_digest,
    _engine_contract_manifest,
    aggregate_benchmark_results,
    load_benchmark_cases,
    select_benchmark_case,
    run_benchmark_case,
    validate_benchmark_result,
)

BENCHMARK_MANIFEST_PATH = Path(
    "tests/fixtures/coding_agent_benchmark/cases.jsonl"
)


def test_benchmark_manifest_exists_with_twenty_two_cases() -> None:
    """The cleaned benchmark corpus is a versioned fixed-size artifact."""

    rows = load_benchmark_cases(BENCHMARK_MANIFEST_PATH)

    assert len(rows) == 22
    assert {row["benchmark_version"] for row in rows} == {BENCHMARK_VERSION}
    assert len({row["category"] for row in rows}) == 9


def test_benchmark_case_selection_returns_one_pinned_fixture() -> None:
    """The CLI selector cannot run multiple or unpinned benchmark cases."""

    cases = load_benchmark_cases(BENCHMARK_MANIFEST_PATH)
    selected = select_benchmark_case(cases, "source_backed_preflight_bugfix")

    assert selected["entrypoint"] == "coding_run"
    assert Path(selected["fixture_ref"]).is_relative_to("tests/fixtures")
    with pytest.raises(ValueError, match="unknown"):
        select_benchmark_case(cases, "missing-case")


def test_action_loop_engine_contract_records_runtime_closure() -> None:
    """Freeze shared runtime owners used by the evaluated action loop."""

    manifest = _engine_contract_manifest("action_loop_v1")
    files = manifest["files"]
    assert isinstance(files, list)
    paths = {row["path"] for row in files}

    required_paths = {
        "src/kazusa_ai_chatbot/coding_agent/code_action_loop/supervisor.py",
        "src/kazusa_ai_chatbot/coding_agent/code_patching/apply.py",
        "src/kazusa_ai_chatbot/coding_agent/code_patching/patch_operations.py",
        "src/kazusa_ai_chatbot/coding_agent/code_patching/patch_validation.py",
        "src/kazusa_ai_chatbot/coding_agent/code_fetching/source_resolver.py",
        "src/kazusa_ai_chatbot/coding_agent/code_executing/runner.py",
        "src/kazusa_ai_chatbot/coding_agent/code_verifying/supervisor.py",
        "src/kazusa_ai_chatbot/coding_agent/coding_run/evaluation.py",
        "src/kazusa_ai_chatbot/coding_agent/repository_index/builder.py",
        "src/kazusa_ai_chatbot/coding_agent/safety.py",
        "src/kazusa_ai_chatbot/llm_interface/interface.py",
        "src/kazusa_ai_chatbot/config.py",
    }
    assert required_paths <= paths
    assert manifest["closure_digest"] == _engine_contract_digest(
        "action_loop_v1",
    )


def test_action_loop_engine_contract_changes_with_included_content() -> None:
    """Bind the closure digest to every included file content identity."""

    manifest = _engine_contract_manifest("action_loop_v1")
    files = manifest["files"]
    assert isinstance(files, list)
    changed_files = [dict(row) for row in files]
    changed_files[0]["content_sha256"] = "f" * 64

    assert _digest_engine_contract_records(
        changed_files,
    ) != manifest["closure_digest"]


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
    first_path = tmp_path / "case-one" / "result.json"
    first_path.parent.mkdir(parents=True)
    first_path.write_text(
        json.dumps(first),
        encoding="utf-8",
    )
    second_path = tmp_path / "case-two" / "result.json"
    second_path.parent.mkdir(parents=True)
    second_path.write_text(
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
    monkeypatch.setattr(
        benchmark,
        "run_evaluation_coding_run",
        lambda request, engine_id: stalled_start(dict(request)),
    )
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


@pytest.mark.asyncio
async def test_verify_repair_benchmark_starts_with_approval_and_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Provide the required verification authorization on the initial request."""

    from scripts import run_coding_agent_benchmark as benchmark

    captured_request: dict[str, object] = {}

    async def rejected_start(request: dict[str, object]) -> dict[str, object]:
        captured_request.update(request)
        return {"status": "rejected", "run_id": "verification-run"}

    case = select_benchmark_case(
        load_benchmark_cases(BENCHMARK_MANIFEST_PATH),
        "repair_cli",
    )
    monkeypatch.setattr(
        benchmark,
        "run_evaluation_coding_run",
        lambda request, engine_id: rejected_start(dict(request)),
    )

    await run_benchmark_case(
        case,
        repository_root=Path.cwd(),
        results_directory=tmp_path,
    )

    approval = captured_request["approval"]
    assert isinstance(approval, dict)
    assert approval["approved"] is True
    assert captured_request["execution_specs"]


@pytest.mark.asyncio
async def test_benchmark_reapproves_each_repaired_proposal_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Continue a repaired proposal with a distinct explicit approval record."""

    from scripts import run_coding_agent_benchmark as benchmark

    approval_message_ids: list[str] = []

    async def awaiting_start(
        _request: dict[str, object],
        engine_id: str,
    ) -> dict[str, object]:
        assert engine_id == "action_loop_v1"
        return {"status": "awaiting_approval", "run_id": "repair-run"}

    async def continue_repair(
        request: dict[str, object],
        engine_id: str,
    ) -> dict[str, object]:
        assert engine_id == "action_loop_v1"
        approval = request["approval"]
        assert isinstance(approval, dict)
        evidence = approval["approval_evidence"]
        assert isinstance(evidence, dict)
        approval_message_ids.append(str(evidence["source_message_id"]))
        if len(approval_message_ids) == 1:
            return {"status": "awaiting_approval", "run_id": "repair-run"}
        return {"status": "completed", "run_id": "repair-run"}

    case = select_benchmark_case(
        load_benchmark_cases(BENCHMARK_MANIFEST_PATH),
        "source_backed_slug_bugfix",
    )
    monkeypatch.setattr(benchmark, "run_evaluation_coding_run", awaiting_start)
    monkeypatch.setattr(
        benchmark,
        "continue_evaluation_coding_run",
        continue_repair,
    )

    result = await run_benchmark_case(
        case,
        repository_root=Path.cwd(),
        results_directory=tmp_path,
        engine_id="action_loop_v1",
    )

    assert result["final_run_status"] == "completed"
    assert approval_message_ids == ["benchmark-message-1", "benchmark-message-2"]


@pytest.mark.asyncio
async def test_benchmark_case_rejects_replacing_durable_evidence(
    tmp_path: Path,
) -> None:
    """Require an explicit retirement step before a case can be rerun."""

    case = select_benchmark_case(
        load_benchmark_cases(BENCHMARK_MANIFEST_PATH),
        "repair_cli",
    )
    existing_result = tmp_path / case["case_id"] / "result.json"
    existing_result.parent.mkdir(parents=True)
    existing_result.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="durable evidence"):
        await run_benchmark_case(
            case,
            repository_root=Path.cwd(),
            results_directory=tmp_path,
        )


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
        "scenario_driver": "source_backed_bug_fix",
        "scenario_contract_digest": "e" * 64,
        "engine_id": "pipeline_v1",
        "engine_contract_digest": _engine_contract_digest("pipeline_v1"),
        "manifest_sha256": "a" * 64,
        "objective_type": "propose_patch",
        "fixture_manifest_sha256": "b" * 64,
        "route_policy_digest": "c" * 64,
        "routes": [{"route_name": "coding_run", "model": "configured"}],
        "status": status,
        "entrypoint": "coding_run",
        "elapsed_ms": 12,
        "llm_call_count": 2,
        "token_usage": None,
        "final_run_status": "completed",
        "terminal_status_match": True,
        "evaluator": {"status": "passed", "checks": ["status matched"]},
        "acceptance_outcomes": {"terminal": True},
        "hard_safety_gate_outcomes": {"source_immutable": True},
        "judgment_status": "reviewed",
        "rubric": {
            "task_progress": 2,
            "repository_grounding": 2,
            "repair_quality": 2,
            "safety_privacy": 2,
            "authorization": 2,
        },
        "judgment_note": "Status matched the locked target.",
        "review_reference": "test_artifacts/review.md",
        "judgment_artifact_path": "test_artifacts/judgment.json",
        "judgment_trace_citations": {
            "task_progress": ["test_artifacts/trace.json"],
            "repository_grounding": ["test_artifacts/trace.json"],
            "repair_quality": ["test_artifacts/trace.json"],
            "safety_privacy": ["test_artifacts/trace.json"],
            "authorization": ["test_artifacts/trace.json"],
        },
        "trace_paths": ["test_artifacts/trace.json"],
        "parsed_observation_trace_paths": ["test_artifacts/trace.json"],
        "context_manifest_paths": ["test_artifacts/context_manifest.json"],
        "raw_trace_sha256": "d" * 64,
        "notes": ["provider token usage was unavailable"],
    }
    return result
