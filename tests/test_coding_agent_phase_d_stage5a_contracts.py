"""Target-state contracts for the Stage 5A evidence and benchmark closure."""

import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_coding_agent_benchmark as benchmark
from kazusa_ai_chatbot.coding_agent.code_action_loop.context import (
    render_controller_context,
)
from kazusa_ai_chatbot.coding_agent.path_classification import is_test_path


def test_shared_test_path_classifier_covers_root_nested_and_non_test_paths() -> None:
    """All callers use one normalized test-artifact classification contract."""

    assert is_test_path("test_root.py")
    assert is_test_path("pkg/test_nested.py")
    assert is_test_path("pkg/nested/example_test.py")
    assert is_test_path("pkg/tests/support.py")
    assert not is_test_path("src/runtime.py")
    assert not is_test_path("src/testimony.py")


def test_context_preserves_safe_nested_edit_result_hash() -> None:
    """Model evidence retains the resulting hash needed for the next edit."""

    context = render_controller_context(
        goal="edit the file twice",
        capabilities=["edit"],
        candidate_revision=2,
        working_notes="",
        observations=[
            {
                "kind": "edit_result",
                "candidate_revision": 2,
                "evidence": [
                    {
                        "patch_operation": {
                            "kind": "replace",
                            "path": "src/example.py",
                            "content_sha256": "a" * 64,
                            "operation_id": "private-operation-id",
                        },
                    },
                ],
            },
        ],
    )

    payload = json.loads(context)
    operation = payload["observations"][0]["evidence"][0]["patch_operation"]
    assert operation == {
        "kind": "replace",
        "path": "src/example.py",
        "content_sha256": "a" * 64,
    }


def test_prompt_preserves_unresolved_external_failure_semantics() -> None:
    """The controller prompt explains continuation after unchanged unavailability."""

    from kazusa_ai_chatbot.coding_agent.code_action_loop.prompts import (
        CONTROLLER_PROMPT,
    )

    lowered = CONTROLLER_PROMPT.casefold()
    assert "remains unresolved" in lowered
    assert "another grounded action" in lowered


def test_current_verification_requires_exactly_one_current_effect() -> None:
    """Historical failures cannot make a stale or ambiguous repair pass."""

    current = {
        "candidate_revision": 3,
        "apply_effect_id": "effect-3",
        "execution_attempts": [
            {
                "candidate_revision": 2,
                "apply_effect_id": "effect-2",
                "status": "succeeded",
            },
            {
                "candidate_revision": 3,
                "apply_effect_id": "effect-3",
                "status": "succeeded",
            },
        ],
    }

    assert benchmark._current_verification_succeeded(current) is True
    current["execution_attempts"].append(
        {
            "candidate_revision": 3,
            "apply_effect_id": "effect-3",
            "status": "succeeded",
        },
    )
    assert benchmark._current_verification_succeeded(current) is False


def test_evaluator_status_requires_terminal_match_and_all_acceptance() -> None:
    """A terminal match cannot override a failed locked acceptance check."""

    assert benchmark._derive_evaluator_status(
        terminal_status_match=True,
        acceptance_outcomes={"final_run_status:completed": True},
        timed_out=False,
    ) == "passed"
    assert benchmark._derive_evaluator_status(
        terminal_status_match=True,
        acceptance_outcomes={"final_run_status:completed": False},
        timed_out=False,
    ) == "failed"
    assert benchmark._derive_evaluator_status(
        terminal_status_match=True,
        acceptance_outcomes={"final_run_status:completed": True},
        timed_out=True,
    ) == "not_applicable"


def test_v3_manifest_declares_closed_scenario_driver_and_allocation() -> None:
    """The v3 manifest must expose the frozen category allocation."""

    assert benchmark.BENCHMARK_VERSION == "coding_agent_benchmark.v3"
    cases = benchmark.load_benchmark_cases()
    assert all(isinstance(case.get("scenario_driver"), str) for case in cases)
    counts = {}
    for case in cases:
        counts[case["category"]] = counts.get(case["category"], 0) + 1
    assert counts == {
        "source_backed_bug_fix": 3,
        "source_free_creation": 3,
        "small_feature": 2,
        "revision": 2,
        "preflight": 2,
        "verification_repair": 3,
        "blocker_response": 3,
        "same_source_concurrency": 2,
        "mixed_create_edit": 2,
    }


def test_v3_scenario_contract_digest_changes_with_locked_driver() -> None:
    """A scenario-driver change retires the prior comparator cohort."""

    case = dict(benchmark.load_benchmark_cases()[0])
    original = benchmark._scenario_contract_digest(case)
    case["scenario_driver"] = "rename"
    changed = benchmark._scenario_contract_digest(case)
    assert original != changed


def _stage5a_source_fixture(root: Path) -> Path:
    """Create one committed source fixture covering the driver contracts."""

    source = root / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "obsolete.py").write_text(
        "VALUE = 'obsolete'\n",
        encoding="utf-8",
    )
    (source / "old.py").write_text(
        "VALUE = 'rename'\n",
        encoding="utf-8",
    )
    (source / "target_module.py").write_text(
        "VALUE = 'target'\n",
        encoding="utf-8",
    )
    for index in range(130):
        (source / f"module_{index:03d}.py").write_text(
            f"VALUE = {index}\n",
            encoding="utf-8",
        )
    tests_root = source / "tests"
    tests_root.mkdir()
    (tests_root / "test_module.py").write_text(
        "from module import VALUE\n\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/fixture/stage5a.git"],
        cwd=source,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Stage5A",
            "-c",
            "user.email=stage5a@example.invalid",
            "commit",
            "-qm",
            "scenario fixture",
        ],
        cwd=source,
        check=True,
    )
    return source


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario_driver",
    sorted(benchmark.SCENARIO_DRIVERS - {"same_source_concurrency"}),
)
async def test_private_stage5a_driver_persists_category_lifecycle(
    tmp_path: Path,
    scenario_driver: str,
) -> None:
    """Exercise every non-concurrent v3 driver through private durable state."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    source = _stage5a_source_fixture(tmp_path)
    request: dict[str, object] = {
        "workspace_root": str(tmp_path / "workspace"),
        "question": f"Run the {scenario_driver} Stage 5A scenario.",
        "objective_type": "propose_patch",
        "local_root_hint": str(source),
        "source_scope_hint": "repository",
    }
    if scenario_driver == "source_free_creation":
        request.pop("local_root_hint")
        request.pop("source_scope_hint")
    result = await evaluation.run_stage5a_scenario_driver(
        request,
        scenario_driver=scenario_driver,
    )

    assert result["status"] == "completed"
    evidence = result["scenario_evidence"]
    assert evidence["approval_required"] is True
    assert evidence["execution_before_approval"] is False
    assert Path(result["scenario_path"]).is_file()
    operations = evidence["operations"]
    assert operations
    kinds = [operation["kind"] for operation in operations]
    if scenario_driver == "revision":
        assert len(evidence["proposal_revisions"]) == 2
        assert evidence["proposal_revisions"][0] != evidence["proposal_revisions"][1]
    elif scenario_driver == "preflight":
        assert evidence["preflight_plan"]["plan_id"]
    elif scenario_driver == "verification_repair":
        attempts = evidence["execution_attempts"]
        assert [attempt["status"] for attempt in attempts] == [
            "failed",
            "succeeded",
        ]
    elif scenario_driver == "mixed_create_edit":
        assert kinds == ["create_file", "replace_file_small"]
        assert operations[1]["expected_source_sha256"]
    elif scenario_driver == "repository_scale":
        assert evidence["search"]["target_path_supplied"] is False
        assert evidence["search"]["discovered_file_count"] > 120
    elif scenario_driver == "stale_index_cursor":
        assert evidence["search"]["stale_rejected"] is True
    elif scenario_driver == "delete":
        assert kinds == ["delete_file"]
    elif scenario_driver == "rename":
        assert kinds == ["rename_file"]
    elif scenario_driver == "blocker_response":
        assert evidence["blocker"]["status"] == "open"
        assert evidence["blocker"]["unresolved"] is True
        assert evidence["verbatim_response"]


@pytest.mark.asyncio
async def test_private_stage5a_concurrency_driver_proves_overlap_and_isolation(
    tmp_path: Path,
) -> None:
    """Run two same-source tasks and retain both lock and ledger identities."""

    from kazusa_ai_chatbot.coding_agent.coding_run import evaluation

    source = _stage5a_source_fixture(tmp_path)
    result = await evaluation.run_stage5a_scenario_driver(
        {
            "workspace_root": str(tmp_path / "workspace"),
            "question": "Run concurrent Stage 5A source tasks.",
            "objective_type": "propose_patch",
            "local_root_hint": str(source),
            "source_scope_hint": "repository",
        },
        scenario_driver="same_source_concurrency",
    )

    evidence = result["scenario_evidence"]
    assert evidence["overlap_observed"] is True
    assert evidence["isolated_ledgers"] is True
    assert len(evidence["worker_ledgers"]) == 2
    assert all(worker["status"] == "completed" for worker in result["workers"])


def test_historical_audit_and_cutover_inventory_are_diagnostic_artifacts() -> None:
    """Stage 5A retains review evidence without claiming public cutover."""

    audit = Path("test_artifacts/stage5a_v2_trace_audit.md").read_text(
        encoding="utf-8",
    )
    inventory = Path("test_artifacts/stage5a_cutover_inventory.md").read_text(
        encoding="utf-8",
    )
    assert "3 surfaced behavioral failures" in audit
    assert "5 false-negative" in audit
    assert "6 or more terminal-matching false assurances" in audit
    assert "no public cutover" in inventory


def test_full_graph_live_nodes_are_prepared_under_live_marker() -> None:
    """The named Stage 5 graph nodes exist before any live execution."""

    source = Path(
        "tests/test_coding_agent_phase_d_action_loop_live_llm.py",
    ).read_text(encoding="utf-8")
    for name in (
        "test_live_read_only_graph_persists_grounded_finish",
        "test_live_propose_patch_graph_materializes_review",
        "test_live_verify_repair_graph_uses_current_effect",
    ):
        assert f"async def {name}" in source
    assert "@pytest.mark.live_llm" in source


@pytest.mark.parametrize("path", ["test_a.py", "pkg/tests/test_a.py"])
def test_shared_classifier_accepts_backslash_normalization(path: str) -> None:
    """Windows fixture paths are normalized before role classification."""

    assert is_test_path(path.replace("/", "\\"))
