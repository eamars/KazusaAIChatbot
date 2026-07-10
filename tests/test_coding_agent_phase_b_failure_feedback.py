"""Contracts for bounded verification failure evidence."""

from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_verifying.supervisor import (
    classify_execution_failure,
    build_execution_failure_bundle,
)


def test_missing_external_module_is_environment_blocker(tmp_path: Path) -> None:
    """A missing external dependency blocks repair before model work begins."""

    bundle = build_execution_failure_bundle(
        spec_id="spec-1",
        execution={
            "tool": "pytest",
            "status": "failed",
            "exit_code": 1,
            "stdout_excerpt": "",
            "stderr_excerpt": "ModuleNotFoundError: No module named 'yaml'",
            "executed_paths": ["tests/test_loader.py"],
        },
        candidate_root=tmp_path,
    )


def test_pytest_collection_header_preserves_missing_module_root_cause(
    tmp_path: Path,
) -> None:
    """A pytest collection header cannot hide a later missing-module cause."""

    bundle = build_execution_failure_bundle(
        spec_id="spec-1",
        execution={
            "tool": "pytest",
            "status": "failed",
            "exit_code": 2,
            "stdout_excerpt": "",
            "stderr_excerpt": (
                "ImportError while importing test module tests/test_loader.py\n"
                "ModuleNotFoundError: No module named 'yaml'"
            ),
            "executed_paths": ["tests/test_loader.py"],
        },
        candidate_root=tmp_path,
    )

    assert bundle["exception_message"] == (
        "ModuleNotFoundError: No module named 'yaml'"
    )
    assert classify_execution_failure(bundle, candidate_root=tmp_path) == (
        "environment_dependency_missing"
    )

    assert classify_execution_failure(bundle, candidate_root=tmp_path) == (
        "environment_dependency_missing"
    )


def test_missing_local_module_remains_a_source_failure(tmp_path: Path) -> None:
    """A candidate-local missing import remains repairable source evidence."""

    (tmp_path / "dep_tool").mkdir()
    (tmp_path / "missing_local.py").write_text("VALUE = 1\n", encoding="utf-8")
    bundle = build_execution_failure_bundle(
        spec_id="spec-1",
        execution={
            "tool": "pytest",
            "status": "failed",
            "exit_code": 1,
            "stdout_excerpt": "",
            "stderr_excerpt": "ModuleNotFoundError: No module named 'missing_local'",
            "executed_paths": ["tests/test_loader.py"],
        },
        candidate_root=tmp_path,
    )

    assert classify_execution_failure(bundle, candidate_root=tmp_path) == "exception"
