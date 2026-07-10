"""Contracts for proposal-bound deterministic execution planning."""

from pathlib import Path

from kazusa_ai_chatbot.coding_agent.code_verifying.execution_planning import (
    derive_base_execution_plan,
    patch_artifact_digest,
    validate_execution_plan_binding,
)


def _artifact(path: str, content: str) -> dict[str, object]:
    """Build one minimal create-file patch artifact for planning tests."""

    return {
        "artifact_id": f"artifact-{path}",
        "base": "empty_source_free",
        "files": [path],
        "summary": "Create test artifact.",
        "diff_text": (
            f"diff --git a/{path} b/{path}\n"
            "new file mode 100644\n"
            "index 0000000..1111111\n"
            "--- /dev/null\n"
            f"+++ b/{path}\n"
            "@@ -0,0 +1 @@\n"
            f"+{content}\n"
        ),
    }


def test_source_free_plan_uses_exact_python_and_generated_test_paths(
    tmp_path: Path,
) -> None:
    """Generated source and test files produce compile and focused pytest specs."""

    source_file = tmp_path / "slug_tools" / "slug.py"
    source_file.parent.mkdir()
    source_file.write_text("def slugify(value):\n    return value\n", encoding="utf-8")
    test_file = tmp_path / "tests" / "test_slug.py"
    test_file.parent.mkdir()
    test_file.write_text("def test_slug():\n    assert True\n", encoding="utf-8")
    artifacts = [
        _artifact("slug_tools/slug.py", "def slugify(value): return value"),
        _artifact("tests/test_slug.py", "def test_slug(): assert True"),
    ]

    plan = derive_base_execution_plan(
        candidate_root=tmp_path,
        patch_artifacts=artifacts,
        run_id="run-1",
        source_identity={"current_commit": "inline-sha256:abc"},
        proposal_revision=1,
    )

    assert plan["base_specs"] == [
        {
            "tool": "python_compileall",
            "paths": ["slug_tools/slug.py", "tests/test_slug.py"],
        },
        {"tool": "pytest", "pytest_selectors": ["tests/test_slug.py"]},
    ]
    assert plan["limitations"] == []


def test_compile_only_plan_records_no_focused_test_discovered(tmp_path: Path) -> None:
    """A Python-only change keeps compile verification when no safe test exists."""

    source_file = tmp_path / "tool.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    artifacts = [_artifact("tool.py", "VALUE = 1")]

    plan = derive_base_execution_plan(
        candidate_root=tmp_path,
        patch_artifacts=artifacts,
        run_id="run-1",
        source_identity={"current_commit": "inline-sha256:abc"},
        proposal_revision=1,
    )

    assert plan["base_specs"] == [
        {"tool": "python_compileall", "paths": ["tool.py"]},
    ]
    assert plan["limitations"] == ["no_focused_test_discovered"]


def test_plan_rejects_stale_artifact_digest(tmp_path: Path) -> None:
    """Execution plans cannot be reused after proposal artifacts change."""

    artifact = _artifact("tool.py", "VALUE = 1")
    (tmp_path / "tool.py").write_text("VALUE = 1\n", encoding="utf-8")
    plan = derive_base_execution_plan(
        candidate_root=tmp_path,
        patch_artifacts=[artifact],
        run_id="run-1",
        source_identity={"current_commit": "inline-sha256:abc"},
        proposal_revision=1,
    )
    changed_artifact = _artifact("tool.py", "VALUE = 2")

    error = validate_execution_plan_binding(
        plan=plan,
        run_id="run-1",
        source_identity={"current_commit": "inline-sha256:abc"},
        proposal_revision=1,
        patch_artifact_digest=patch_artifact_digest([changed_artifact]),
    )

    assert error == "Execution plan patch artifact digest is stale."
