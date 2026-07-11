"""Contracts for the Phase D two-engine benchmark artifacts."""

import hashlib
import inspect
import json
from pathlib import Path
import subprocess

from scripts import run_coding_agent_benchmark as benchmark

import pytest


def test_phase_d_benchmark_declares_both_engine_ids() -> None:
    """Require the benchmark harness to select either evaluated engine."""

    assert benchmark.BENCHMARK_VERSION == "coding_agent_benchmark.v2"
    assert benchmark.ENGINE_IDS == frozenset(("pipeline_v1", "action_loop_v1"))


def test_benchmark_continuation_stays_inside_private_engine_boundary() -> None:
    """Keep approval resume paired with the engine selected for case start."""

    source = inspect.getsource(benchmark._execute_case_lifecycle)

    assert "continue_evaluation_coding_run" in source
    assert "continue_coding_run(" not in source


def test_manifest_rows_carry_paired_evaluation_contracts() -> None:
    """Require every locked case to carry complete comparable evaluation data."""

    for case in benchmark.load_benchmark_cases():
        assert case["objective_type"] in {
            "read_only",
            "propose_patch",
            "verify_repair",
        }
        assert isinstance(case["task_text"], str) and case["task_text"]
        assert isinstance(case["fixture_manifest_sha256"], str)
        assert isinstance(case["route_policy_digest"], str)
        assert isinstance(case["acceptance_checks"], list)
        assert case["acceptance_checks"] == benchmark._expected_acceptance_checks(
            case
        )
        assert isinstance(case["hard_safety_gates"], list)
        assert set(case["rubric"]) == {
            "task_progress",
            "repository_grounding",
            "repair_quality",
            "safety_privacy",
            "authorization",
        }


@pytest.mark.asyncio
async def test_action_loop_evaluation_isolated_from_public_run_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject a source-less read without calling the public legacy runtime."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    async def public_start_called(_request: dict[str, object]) -> dict[str, object]:
        raise AssertionError("action-loop evaluation called public pipeline dispatch")

    monkeypatch.setattr(evaluation, "start_coding_run", public_start_called)

    response = await evaluation.run_evaluation_coding_run(
        {
            "workspace_root": str(tmp_path),
            "question": "Read the source.",
        },
        engine_id="action_loop_v1",
    )

    assert response["status"] == "needs_user_input"


@pytest.mark.asyncio
async def test_environment_precondition_is_identical_across_native_ledgers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Materialize one neutral reviewed proposal without an LLM proposal stub."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    case = next(
        row
        for row in benchmark.load_benchmark_cases()
        if row["case_id"] == "dependency_blocker_response"
    )
    fixture_path = Path(str(case["fixture_ref"]))
    approval_request = benchmark._approval_request(
        workspace_root=tmp_path,
        run_id="",
        fixture_path=fixture_path,
    )

    async def unexpected_controller(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("scenario proposal invoked the model controller")

    monkeypatch.setattr(evaluation, "invoke_controller", unexpected_controller)
    materialized: dict[str, tuple[dict[str, object], Path]] = {}
    preconditions: dict[str, dict[str, object]] = {}
    for engine_id in ("pipeline_v1", "action_loop_v1"):
        engine_root = tmp_path / engine_id
        source_root = benchmark._prepare_fixture_checkout(
            fixture_path,
            engine_root,
        )
        request = {
            "question": case["task_text"],
            "objective_type": case["objective_type"],
            "workspace_root": str(engine_root),
            "local_root_hint": str(source_root),
            "source_scope_hint": "repository",
        }
        precondition = evaluation.build_environment_scenario_precondition(
            request,
            fixture_manifest_sha256=str(case["fixture_manifest_sha256"]),
            approval=approval_request["approval"],
            execution_specs=approval_request["execution_specs"],
        )
        response = await evaluation.materialize_evaluation_scenario(
            request,
            engine_id=engine_id,
            precondition=precondition,
        )
        assert response["status"] == "awaiting_approval"
        materialized[engine_id] = (response, engine_root)
        preconditions[engine_id] = precondition

    pipeline_precondition = preconditions["pipeline_v1"]
    action_precondition = preconditions["action_loop_v1"]
    assert pipeline_precondition["scenario_precondition_digest"] == (
        action_precondition["scenario_precondition_digest"]
    )
    assert pipeline_precondition["canonical_operation_records"] == (
        action_precondition["canonical_operation_records"]
    )
    assert pipeline_precondition["review_sha256"] == action_precondition[
        "review_sha256"
    ]
    review_contents = []
    for engine_id, (response, engine_root) in materialized.items():
        run_root = engine_root / "coding_runs" / str(response["run_id"])
        review_path = run_root / "evaluation_precondition" / "review.json"
        review_contents.append(review_path.read_bytes())
        state_path = run_root / "run.json"
        if engine_id == "action_loop_v1":
            state_path = run_root / "action_loop" / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        assert state["scenario_precondition_digest"] == pipeline_precondition[
            "scenario_precondition_digest"
        ]
        assert state["scenario_canonical_operation_records"] == (
            pipeline_precondition["canonical_operation_records"]
        )
    assert review_contents[0] == review_contents[1]
    action_started, action_root = materialized["action_loop_v1"]
    mismatched_approval = dict(approval_request["approval"])
    mismatched_approval["approval_reason"] = "Changed approval identity."
    rejected = await evaluation.continue_evaluation_coding_run(
        {
            "workspace_root": str(action_root),
            "run_id": action_started["run_id"],
            "action": "approve_and_verify",
            "approval": mismatched_approval,
            "execution_specs": approval_request["execution_specs"],
            "repair_attempt_limit": 0,
        },
        engine_id="action_loop_v1",
    )
    assert rejected["status"] == "rejected"
    assert rejected["limitations"] == [
        "Evaluation scenario approval identity mismatch."
    ]


@pytest.mark.asyncio
async def test_environment_precondition_runs_native_verifier_and_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Require both native continuations to retain real dependency evidence."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.parser import parse_action
    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    case = next(
        row
        for row in benchmark.load_benchmark_cases()
        if row["case_id"] == "dependency_blocker_response"
    )
    fixture_path = Path(str(case["fixture_ref"]))

    async def dependency_blocker_controller(
        *,
        context: str,
        allowed_actions: set[str],
    ) -> dict[str, object]:
        assert "definitely_missing_kazusa_yaml_20260709" in context
        payload = {
            "schema_version": "coding_action.v1",
            "action_id": "dependency-blocker",
            "action": "block",
            "reason": "Verification found the unavailable external dependency.",
            "args": {
                "blocker_type": "environment",
                "question": "Provide the required YAML dependency.",
                "options": ["Install the dependency and retry verification."],
                "blocking_evidence_refs": ["execution_verification"],
            },
        }
        return parse_action(payload, allowed_actions=allowed_actions)

    monkeypatch.setattr(
        evaluation,
        "invoke_controller",
        dependency_blocker_controller,
    )
    for engine_id in ("pipeline_v1", "action_loop_v1"):
        engine_root = tmp_path / engine_id
        source_root = benchmark._prepare_fixture_checkout(
            fixture_path,
            engine_root,
        )
        approval_request = benchmark._approval_request(
            workspace_root=engine_root,
            run_id="",
            fixture_path=fixture_path,
        )
        request = {
            "question": case["task_text"],
            "objective_type": case["objective_type"],
            "workspace_root": str(engine_root),
            "local_root_hint": str(source_root),
            "source_scope_hint": "repository",
        }
        precondition = evaluation.build_environment_scenario_precondition(
            request,
            fixture_manifest_sha256=str(case["fixture_manifest_sha256"]),
            approval=approval_request["approval"],
            execution_specs=approval_request["execution_specs"],
        )
        started = await evaluation.materialize_evaluation_scenario(
            request,
            engine_id=engine_id,
            precondition=precondition,
        )
        approval_request["run_id"] = started["run_id"]
        continued = await evaluation.continue_evaluation_coding_run(
            approval_request,
            engine_id=engine_id,
        )
        assert continued["status"] == "blocked"
        run_root = engine_root / "coding_runs" / str(started["run_id"])
        state_path = run_root / "run.json"
        if engine_id == "action_loop_v1":
            state_path = run_root / "action_loop" / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        attempts = state["execution_attempts"]
        assert attempts and attempts[0]["status"] == "failed"
        if engine_id == "pipeline_v1":
            blockers = state["blockers"]
            assert blockers[0]["blocker_kind"] == "environment"
            assert blockers[0]["details"]["missing_module"] == (
                "definitely_missing_kazusa_yaml_20260709"
            )
        else:
            combined_output = (
                attempts[0]["stdout_excerpt"] + attempts[0]["stderr_excerpt"]
            )
            assert "definitely_missing_kazusa_yaml_20260709" in combined_output
            assert state["blocker"]["blocker_type"] == "environment"
        assert not subprocess.run(
            ["git", "status", "--short"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout


def test_comparator_rejects_missing_paired_engine_results(tmp_path) -> None:
    """Require a locked comparator to reject incomplete paired evidence."""

    with pytest.raises(ValueError, match="missing result"):
        benchmark.compare_benchmark_results(
            manifest_path=benchmark.MANIFEST_PATH,
            pipeline_results_directory=tmp_path / "pipeline_v1",
            action_loop_results_directory=tmp_path / "action_loop_v1",
        )


def test_comparator_ignores_retained_attempt_results(tmp_path: Path) -> None:
    """Keep retired evidence outside the canonical paired result cohort."""

    first_case = benchmark.load_benchmark_cases()[0]
    retained_result = (
        tmp_path
        / "action_loop_v1"
        / first_case["case_id"]
        / "attempts"
        / "pre-freeze-incomplete"
        / "result.json"
    )
    retained_result.parent.mkdir(parents=True)
    retained_result.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="missing result"):
        benchmark.compare_benchmark_results(
            manifest_path=benchmark.MANIFEST_PATH,
            pipeline_results_directory=tmp_path / "pipeline_v1",
            action_loop_results_directory=tmp_path / "action_loop_v1",
        )


def test_comparator_rejects_unlocked_or_unsafe_paired_results(
    tmp_path: Path,
) -> None:
    """Reject pair drift and any failed hard gate before scoring a cutover."""

    pipeline_root = tmp_path / "pipeline_v1"
    action_loop_root = tmp_path / "action_loop_v1"
    _write_paired_results(pipeline_root, engine_id="pipeline_v1")
    _write_paired_results(action_loop_root, engine_id="action_loop_v1")
    first_case = benchmark.load_benchmark_cases()[0]
    unsafe_path = action_loop_root / first_case["case_id"] / "result.json"
    unsafe = json.loads(unsafe_path.read_text(encoding="utf-8"))
    unsafe["hard_safety_gate_outcomes"]["original_source_immutable"] = False
    unsafe_path.write_text(json.dumps(unsafe), encoding="utf-8")

    with pytest.raises(ValueError, match="hard safety gate"):
        benchmark.compare_benchmark_results(
            manifest_path=benchmark.MANIFEST_PATH,
            pipeline_results_directory=pipeline_root,
            action_loop_results_directory=action_loop_root,
        )

    unsafe["hard_safety_gate_outcomes"]["original_source_immutable"] = True
    unsafe["fixture_manifest_sha256"] = "f" * 64
    unsafe_path.write_text(json.dumps(unsafe), encoding="utf-8")

    with pytest.raises(ValueError, match="fixture manifest mismatch"):
        benchmark.compare_benchmark_results(
            manifest_path=benchmark.MANIFEST_PATH,
            pipeline_results_directory=pipeline_root,
            action_loop_results_directory=action_loop_root,
        )


def test_comparator_rejects_environment_precondition_drift(tmp_path: Path) -> None:
    """Require paired blocker cases to share one neutral proposal identity."""

    pipeline_root = tmp_path / "pipeline_v1"
    action_loop_root = tmp_path / "action_loop_v1"
    _write_paired_results(pipeline_root, engine_id="pipeline_v1")
    _write_paired_results(action_loop_root, engine_id="action_loop_v1")
    environment_case = next(
        row
        for row in benchmark.load_benchmark_cases()
        if row["category"] == "environment_blocker"
    )
    result_path = action_loop_root / environment_case["case_id"] / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["scenario_precondition_digest"] = "f" * 64
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="scenario precondition mismatch"):
        benchmark.compare_benchmark_results(
            manifest_path=benchmark.MANIFEST_PATH,
            pipeline_results_directory=pipeline_root,
            action_loop_results_directory=action_loop_root,
        )

def test_comparator_rejects_provisional_or_unreviewed_judgment(
    tmp_path: Path,
) -> None:
    """Keep unreviewed behavior scores out of paired cutover evidence."""

    pipeline_root = tmp_path / "pipeline_v1"
    action_loop_root = tmp_path / "action_loop_v1"
    _write_paired_results(pipeline_root, engine_id="pipeline_v1")
    _write_paired_results(action_loop_root, engine_id="action_loop_v1")
    first_case = benchmark.load_benchmark_cases()[0]
    result_path = action_loop_root / first_case["case_id"] / "result.json"
    provisional = json.loads(result_path.read_text(encoding="utf-8"))
    provisional.update({
        "judgment_status": "provisional",
        "rubric": None,
        "judgment_note": None,
        "review_reference": None,
        "judgment_artifact_path": None,
        "judgment_trace_citations": None,
    })
    result_path.write_text(json.dumps(provisional), encoding="utf-8")

    with pytest.raises(ValueError, match="unreviewed"):
        benchmark.compare_benchmark_results(
            manifest_path=benchmark.MANIFEST_PATH,
            pipeline_results_directory=pipeline_root,
            action_loop_results_directory=action_loop_root,
        )


def test_reviewed_result_rejects_missing_non_target_trace_citation() -> None:
    """Require trace citations whenever an agent gives a non-target score."""

    case = benchmark.load_benchmark_cases()[0]
    result = {
        "schema_version": benchmark.RESULT_SCHEMA_VERSION,
        "benchmark_version": benchmark.BENCHMARK_VERSION,
        "case_id": case["case_id"],
        "category": case["category"],
        "engine_id": "action_loop_v1",
        "engine_contract_digest": benchmark._engine_contract_digest(
            "action_loop_v1"
        ),
        "manifest_sha256": benchmark._manifest_sha256(benchmark.MANIFEST_PATH),
        "objective_type": case["objective_type"],
        "fixture_manifest_sha256": case["fixture_manifest_sha256"],
        "route_policy_digest": case["route_policy_digest"],
        "routes": [{"route_name": "action_loop_v1", "model": "configured"}],
        "status": "passed",
        "entrypoint": case["entrypoint"],
        "elapsed_ms": 1,
        "llm_call_count": 1,
        "token_usage": None,
        "final_run_status": case["evaluator"]["required_final_status"],
        "evaluator": {"status": "passed", "checks": ["matched"]},
        "acceptance_outcomes": {check: True for check in case["acceptance_checks"]},
        "hard_safety_gate_outcomes": {
            gate: True for gate in case["hard_safety_gates"]
        },
        "judgment_status": "reviewed",
        "rubric": {
            "task_progress": 1,
            "repository_grounding": 2,
            "repair_quality": 2,
            "safety_privacy": 2,
            "authorization": 2,
        },
        "judgment_note": "Trace showed incomplete progress.",
        "review_reference": "test_artifacts/review.md",
        "judgment_artifact_path": "test_artifacts/judgment.json",
        "judgment_trace_citations": {
            "task_progress": [],
            "repository_grounding": ["test_artifacts/trace.json"],
            "repair_quality": ["test_artifacts/trace.json"],
            "safety_privacy": ["test_artifacts/trace.json"],
            "authorization": ["test_artifacts/trace.json"],
        },
        "trace_paths": ["test_artifacts/trace.json"],
        "parsed_observation_trace_paths": ["test_artifacts/trace.json"],
        "context_manifest_paths": ["test_artifacts/context_manifest.json"],
        "raw_trace_sha256": "d" * 64,
        "notes": ["test fixture"],
    }

    with pytest.raises(ValueError, match="non-target score"):
        benchmark.validate_benchmark_result(result)


def test_finalize_binds_reviewed_judgment_to_provisional_trace(
    tmp_path: Path,
) -> None:
    """Finalize only an agent review bound to the exact retained run facts."""

    case = benchmark.load_benchmark_cases()[0]
    provisional_path, judgment_path = _write_provisional_and_judgment(
        tmp_path,
        case,
    )

    finalized = benchmark.finalize_existing_benchmark_result(
        case,
        provisional_result_path=provisional_path,
        judgment_path=judgment_path,
        engine_id="action_loop_v1",
    )

    assert finalized["judgment_status"] == "reviewed"
    assert finalized["rubric"]["task_progress"] == 2
    assert (provisional_path.parent / "result.json").is_file()


def test_retirement_preserves_complete_reviewed_attempt_before_reuse(
    tmp_path: Path,
) -> None:
    """Move reviewed evidence atomically into an auditable noncanonical attempt."""

    case = benchmark.load_benchmark_cases()[0]
    provisional_path, judgment_path = _write_provisional_and_judgment(
        tmp_path,
        case,
    )
    benchmark.finalize_existing_benchmark_result(
        case,
        provisional_result_path=provisional_path,
        judgment_path=judgment_path,
        engine_id="action_loop_v1",
    )
    results_directory = tmp_path.parent / "results"
    case_root = results_directory / case["case_id"]
    case_root.mkdir(parents=True)
    for name in (
        "result.json",
        "result.provisional.json",
        "judgment.json",
        "review.md",
        "context_manifest.json",
    ):
        (provisional_path.parent / name).replace(case_root / name)

    retirement = benchmark.retire_canonical_benchmark_attempt(
        results_directory=results_directory,
        case_id=case["case_id"],
        attempt_id="post-guard-failure",
        reason="Preserve failed reviewed evidence before prompt remediation.",
    )

    attempt_root = case_root / "attempts" / "post-guard-failure"
    assert not (case_root / "result.json").exists()
    assert (attempt_root / "result.json").is_file()
    assert (attempt_root / "result.provisional.json").is_file()
    assert (attempt_root / "judgment.json").is_file()
    assert (attempt_root / "review.md").is_file()
    assert (attempt_root / "retirement.json").is_file()
    assert retirement["prior_result_sha256"] == hashlib.sha256(
        (attempt_root / "result.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("provisional_updates", "judgment_updates", "error"),
    [
        ({"case_id": "wrong-case"}, {}, "case identity"),
        ({"engine_id": "pipeline_v1"}, {}, "engine identity"),
        ({"manifest_sha256": "f" * 64}, {}, "manifest identity"),
        ({"fixture_manifest_sha256": "f" * 64}, {}, "source identity"),
        ({"route_policy_digest": "f" * 64}, {}, "route identity"),
        ({"raw_trace_sha256": "f" * 64}, {}, "raw trace identity"),
        ({}, {"case_id": "wrong-case"}, "case_id does not bind"),
        ({}, {"engine_id": "pipeline_v1"}, "engine_id does not bind"),
        ({}, {"manifest_sha256": "f" * 64}, "manifest_sha256 does not bind"),
        ({}, {"raw_trace_sha256": "f" * 64}, "raw_trace_sha256 does not bind"),
        ({}, {"review_reference": "missing-review.md"}, "review reference is missing"),
    ],
)
def test_finalize_rejects_identity_or_review_reference_drift(
    tmp_path: Path,
    provisional_updates: dict[str, object],
    judgment_updates: dict[str, object],
    error: str,
) -> None:
    """Reject changed locked facts and a review reference that cannot be read."""

    case = benchmark.load_benchmark_cases()[0]
    provisional_path, judgment_path = _write_provisional_and_judgment(
        tmp_path,
        case,
        provisional_updates=provisional_updates,
        judgment_updates=judgment_updates,
    )

    with pytest.raises(ValueError, match=error):
        benchmark.finalize_existing_benchmark_result(
            case,
            provisional_result_path=provisional_path,
            judgment_path=judgment_path,
            engine_id="action_loop_v1",
        )


def _write_paired_results(root: Path, *, engine_id: str) -> None:
    """Materialize one complete locked engine result set for comparator tests."""

    manifest_digest = benchmark._manifest_sha256(benchmark.MANIFEST_PATH)
    for case in benchmark.load_benchmark_cases():
        result = {
            "schema_version": benchmark.RESULT_SCHEMA_VERSION,
            "benchmark_version": benchmark.BENCHMARK_VERSION,
            "case_id": case["case_id"],
            "category": case["category"],
            "engine_id": engine_id,
            "engine_contract_digest": benchmark._engine_contract_digest(engine_id),
            "manifest_sha256": manifest_digest,
            "objective_type": case["objective_type"],
            "fixture_manifest_sha256": case["fixture_manifest_sha256"],
            "route_policy_digest": case["route_policy_digest"],
            "routes": [{"route_name": engine_id, "model": "configured"}],
            "status": "passed",
            "entrypoint": case["entrypoint"],
            "elapsed_ms": 1,
            "llm_call_count": 1,
            "token_usage": None,
            "final_run_status": case["evaluator"]["required_final_status"],
            "evaluator": {"status": "passed", "checks": ["matched"]},
            "acceptance_outcomes": {
                check: True for check in case["acceptance_checks"]
            },
            "hard_safety_gate_outcomes": {
                gate: True for gate in case["hard_safety_gates"]
            },
            "judgment_status": "reviewed",
            "rubric": dict(case["rubric"]),
            "judgment_note": "Locked test result.",
            "review_reference": "test_artifacts/review.md",
            "judgment_artifact_path": "test_artifacts/judgment.json",
            "judgment_trace_citations": {
                key: ["test_artifacts/trace.json"] for key in case["rubric"]
            },
            "trace_paths": ["test_artifacts/trace.json"],
            "parsed_observation_trace_paths": ["test_artifacts/trace.json"],
            "context_manifest_paths": ["test_artifacts/context_manifest.json"],
            "raw_trace_sha256": "d" * 64,
            "notes": ["test fixture"],
        }
        if case["category"] in {"environment_blocker", "blocker_response"}:
            result["scenario_precondition_digest"] = "e" * 64
        path = root / case["case_id"] / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result), encoding="utf-8")


def _write_provisional_and_judgment(
    root: Path,
    case: dict[str, object],
    *,
    provisional_updates: dict[str, object] | None = None,
    judgment_updates: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    """Build one trace-backed provisional result and its reviewed judgment."""

    source_root = root / "source"
    source_root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=source_root, check=True)
    run_path = root / "workspace" / "coding_runs" / "run-one" / "run.json"
    run_path.parent.mkdir(parents=True)
    run_path.write_text(
        json.dumps({
            "source_request": {"local_root_hint": str(source_root)},
            "approvals": [],
            "apply_attempts": [],
        }),
        encoding="utf-8",
    )
    trace_paths = [str(run_path)]
    provisional = {
        "schema_version": benchmark.RESULT_SCHEMA_VERSION,
        "benchmark_version": benchmark.BENCHMARK_VERSION,
        "case_id": case["case_id"],
        "category": case["category"],
        "engine_id": "action_loop_v1",
        "engine_contract_digest": benchmark._engine_contract_digest(
            "action_loop_v1"
        ),
        "manifest_sha256": benchmark._manifest_sha256(benchmark.MANIFEST_PATH),
        "objective_type": case["objective_type"],
        "fixture_manifest_sha256": case["fixture_manifest_sha256"],
        "route_policy_digest": case["route_policy_digest"],
        "routes": [{"route_name": "action_loop_v1", "model": "configured"}],
        "status": "passed",
        "entrypoint": case["entrypoint"],
        "elapsed_ms": 1,
        "llm_call_count": 1,
        "token_usage": None,
        "final_run_status": case["evaluator"]["required_final_status"],
        "evaluator": {"status": "passed", "checks": ["matched"]},
        "acceptance_outcomes": {check: True for check in case["acceptance_checks"]},
        "hard_safety_gate_outcomes": {
            gate: True for gate in case["hard_safety_gates"]
        },
        "judgment_status": "provisional",
        "rubric": None,
        "judgment_note": None,
        "review_reference": None,
        "judgment_artifact_path": None,
        "judgment_trace_citations": None,
        "trace_paths": trace_paths,
        "parsed_observation_trace_paths": trace_paths,
        "context_manifest_paths": ["test_artifacts/context_manifest.json"],
        "raw_trace_sha256": benchmark._trace_sha256(trace_paths),
        "notes": ["test fixture"],
    }
    if provisional_updates:
        provisional.update(provisional_updates)
    case_root = root / "case"
    provisional_path = case_root / "result.provisional.json"
    provisional_path.parent.mkdir(parents=True, exist_ok=True)
    provisional_path.write_text(json.dumps(provisional), encoding="utf-8")
    review_path = case_root / "review.md"
    review_path.write_text("Agent reviewed the retained trace.\n", encoding="utf-8")
    judgment = {
        "schema_version": benchmark.JUDGMENT_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "engine_id": "action_loop_v1",
        "engine_contract_digest": benchmark._engine_contract_digest(
            "action_loop_v1"
        ),
        "manifest_sha256": benchmark._manifest_sha256(benchmark.MANIFEST_PATH),
        "raw_trace_sha256": benchmark._trace_sha256(trace_paths),
        "review_reference": "review.md",
        "rubric": {
            "task_progress": 2,
            "repository_grounding": 2,
            "repair_quality": 2,
            "safety_privacy": 2,
            "authorization": 2,
        },
        "judgment_note": "The retained trace satisfied the case rubric.",
        "trace_citations": {
            "task_progress": [str(run_path)],
            "repository_grounding": [str(run_path)],
            "repair_quality": [str(run_path)],
            "safety_privacy": [str(run_path)],
            "authorization": [str(run_path)],
        },
    }
    if judgment_updates:
        judgment.update(judgment_updates)
    judgment_path = case_root / "judgment.json"
    judgment_path.write_text(json.dumps(judgment), encoding="utf-8")
    return provisional_path, judgment_path
