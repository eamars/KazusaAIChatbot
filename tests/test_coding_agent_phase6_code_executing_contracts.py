"""Deterministic contracts for bounded code execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def test_execution_rejects_unsupported_tool(tmp_path: Path) -> None:
    """Only closed structured verification tools are executable."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root, apply_package_id, apply_ref = _managed_apply_workspace(
        tmp_path,
    )
    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_package_id,
        "apply_workspace_ref": apply_ref,
        "execution": {
            "tool": "pip_install",
            "paths": ["app.py"],
            "pytest_selectors": [],
            "timeout_seconds": 5,
        },
    })

    assert response["status"] == "rejected"
    assert response["tool"] == "pip_install"
    assert response["exit_code"] is None
    assert any("unsupported" in item.casefold() for item in response["limitations"])


def test_execution_rejects_free_form_command(tmp_path: Path) -> None:
    """Command strings are not part of the execution request contract."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root, apply_package_id, apply_ref = _managed_apply_workspace(
        tmp_path,
    )
    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_package_id,
        "apply_workspace_ref": apply_ref,
        "execution": {
            "tool": "pytest",
            "command": f"{sys.executable} -m pytest tests",
            "paths": [],
            "pytest_selectors": ["tests/test_app.py"],
            "timeout_seconds": 5,
        },
    })

    assert response["status"] == "rejected"
    assert any("command" in item.casefold() for item in response["limitations"])


def test_execution_rejects_path_traversal(tmp_path: Path) -> None:
    """Execution targets must stay inside the managed apply source root."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root, apply_package_id, apply_ref = _managed_apply_workspace(
        tmp_path,
    )
    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_package_id,
        "apply_workspace_ref": apply_ref,
        "execution": {
            "tool": "python_compileall",
            "paths": ["../outside.py"],
            "pytest_selectors": [],
            "timeout_seconds": 5,
        },
    })

    assert response["status"] == "rejected"
    assert response["executed_paths"] == []
    assert any("path" in item.casefold() for item in response["limitations"])


def test_execution_rejects_secret_like_path(tmp_path: Path) -> None:
    """Secret-like files are not valid execution targets."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root, apply_package_id, apply_ref = _managed_apply_workspace(
        tmp_path,
    )
    source_root = _apply_source_root(workspace_root, apply_package_id)
    (source_root / ".env").write_text("TOKEN=secret\n", encoding="utf-8")

    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_package_id,
        "apply_workspace_ref": apply_ref,
        "execution": {
            "tool": "python_compileall",
            "paths": [".env"],
            "pytest_selectors": [],
            "timeout_seconds": 5,
        },
    })

    assert response["status"] == "rejected"
    assert any("unsafe" in item.casefold() for item in response["limitations"])


def test_execution_rejects_missing_apply_workspace(tmp_path: Path) -> None:
    """Execution requires an existing managed apply source directory."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root = tmp_path / "workspace"
    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": "missing",
        "apply_workspace_ref": _apply_ref("missing"),
        "execution": {
            "tool": "python_compileall",
            "paths": ["app.py"],
            "pytest_selectors": [],
            "timeout_seconds": 5,
        },
    })

    assert response["status"] == "rejected"
    assert any("workspace" in item.casefold() for item in response["limitations"])


def test_execution_reports_timeout(tmp_path: Path) -> None:
    """Long-running target tests are stopped and represented structurally."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root, apply_package_id, apply_ref = _managed_apply_workspace(
        tmp_path,
    )
    source_root = _apply_source_root(workspace_root, apply_package_id)
    tests_root = source_root / "tests"
    tests_root.mkdir()
    (tests_root / "test_slow.py").write_text(
        "import time\n\n"
        "def test_slow():\n"
        "    time.sleep(5)\n",
        encoding="utf-8",
    )

    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_package_id,
        "apply_workspace_ref": apply_ref,
        "execution": {
            "tool": "pytest",
            "paths": [],
            "pytest_selectors": ["tests/test_slow.py"],
            "timeout_seconds": 1,
        },
    })

    assert response["status"] == "timed_out"
    assert response["timed_out"] is True
    assert response["exit_code"] is None


def test_execution_caps_and_sanitizes_output(tmp_path: Path) -> None:
    """Noisy command output is capped and stripped of managed root paths."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root, apply_package_id, apply_ref = _managed_apply_workspace(
        tmp_path,
    )
    source_root = _apply_source_root(workspace_root, apply_package_id)
    tests_root = source_root / "tests"
    tests_root.mkdir()
    (tests_root / "test_noisy.py").write_text(
        "from pathlib import Path\n\n"
        "def test_noisy():\n"
        "    root = str(Path.cwd())\n"
        "    raise AssertionError(root + ' ' + ('x' * 1000))\n",
        encoding="utf-8",
    )

    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_package_id,
        "apply_workspace_ref": apply_ref,
        "execution": {
            "tool": "pytest",
            "paths": [],
            "pytest_selectors": ["tests/test_noisy.py"],
            "timeout_seconds": 10,
        },
        "max_stdout_chars": 120,
        "max_stderr_chars": 120,
    })
    serialized = json.dumps(response, ensure_ascii=False)

    assert response["status"] == "failed"
    assert response["output_truncated"] is True
    assert len(response["stdout_excerpt"]) <= 120
    assert str(source_root.resolve()) not in serialized
    assert str(workspace_root.resolve()) not in serialized


def test_execution_sanitizes_environment_values(tmp_path: Path) -> None:
    """Captured output redacts environment values supplied to the process."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root, apply_package_id, apply_ref = _managed_apply_workspace(
        tmp_path,
    )
    source_root = _apply_source_root(workspace_root, apply_package_id)
    tests_root = source_root / "tests"
    tests_root.mkdir()
    (tests_root / "test_env.py").write_text(
        "import os\n\n"
        "def test_env_value_leak():\n"
        "    raise AssertionError(os.environ['PYTHONPATH'])\n",
        encoding="utf-8",
    )

    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_package_id,
        "apply_workspace_ref": apply_ref,
        "execution": {
            "tool": "pytest",
            "paths": [],
            "pytest_selectors": ["tests/test_env.py"],
            "timeout_seconds": 10,
        },
    })
    serialized = json.dumps(response, ensure_ascii=False)

    assert response["status"] == "failed"
    assert "[execution-env]" in serialized or "[managed-workspace]" in serialized
    assert str(source_root.resolve()) not in serialized


def test_execution_rejects_directory_above_file_count_cap(
    tmp_path: Path,
) -> None:
    """Directory targets are scanned before recursive tool execution."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root, apply_package_id, apply_ref = _managed_apply_workspace(
        tmp_path,
    )
    source_root = _apply_source_root(workspace_root, apply_package_id)
    package_root = source_root / "large_package"
    package_root.mkdir()
    for index in range(513):
        (package_root / f"module_{index}.py").write_text(
            "VALUE = 1\n",
            encoding="utf-8",
        )

    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_package_id,
        "apply_workspace_ref": apply_ref,
        "execution": {
            "tool": "python_compileall",
            "paths": ["large_package"],
            "pytest_selectors": [],
            "timeout_seconds": 10,
        },
    })

    assert response["status"] == "rejected"
    assert any("file count" in item.casefold() for item in response["limitations"])


def test_execution_reports_nonzero_exit(tmp_path: Path) -> None:
    """Target-project pytest failures are failed execution results."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root, apply_package_id, apply_ref = _managed_apply_workspace(
        tmp_path,
    )
    source_root = _apply_source_root(workspace_root, apply_package_id)
    tests_root = source_root / "tests"
    tests_root.mkdir()
    (tests_root / "test_app.py").write_text(
        "def test_app():\n"
        "    assert False\n",
        encoding="utf-8",
    )

    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_package_id,
        "apply_workspace_ref": apply_ref,
        "execution": {
            "tool": "pytest",
            "paths": [],
            "pytest_selectors": ["tests/test_app.py"],
            "timeout_seconds": 10,
        },
    })

    assert response["status"] == "failed"
    assert response["exit_code"] not in (None, 0)
    assert response["executed_paths"] == ["tests/test_app.py"]


def test_execution_succeeds_for_compileall(tmp_path: Path) -> None:
    """Compile execution succeeds for syntactically valid Python targets."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root, apply_package_id, apply_ref = _managed_apply_workspace(
        tmp_path,
    )
    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_package_id,
        "apply_workspace_ref": apply_ref,
        "execution": {
            "tool": "python_compileall",
            "paths": ["app.py"],
            "pytest_selectors": [],
            "timeout_seconds": 10,
        },
    })

    assert response["status"] == "succeeded"
    assert response["tool"] == "python_compileall"
    assert response["exit_code"] == 0
    assert response["executed_paths"] == ["app.py"]


def test_execution_succeeds_for_pytest(tmp_path: Path) -> None:
    """Pytest execution succeeds when focused target tests pass."""

    from kazusa_ai_chatbot.coding_agent import execute_code_check

    workspace_root, apply_package_id, apply_ref = _managed_apply_workspace(
        tmp_path,
    )
    source_root = _apply_source_root(workspace_root, apply_package_id)
    tests_root = source_root / "tests"
    tests_root.mkdir()
    (tests_root / "test_app.py").write_text(
        "def test_app():\n"
        "    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )

    response = execute_code_check({
        "workspace_root": str(workspace_root),
        "apply_package_id": apply_package_id,
        "apply_workspace_ref": apply_ref,
        "execution": {
            "tool": "pytest",
            "paths": [],
            "pytest_selectors": ["tests/test_app.py"],
            "timeout_seconds": 10,
        },
    })

    assert response["status"] == "succeeded"
    assert response["tool"] == "pytest"
    assert response["exit_code"] == 0
    assert response["executed_paths"] == ["tests/test_app.py"]


def _managed_apply_workspace(
    tmp_path: Path,
) -> tuple[Path, str, dict[str, object]]:
    """Create a minimal managed apply workspace fixture."""

    workspace_root = tmp_path / "workspace"
    apply_package_id = "apply123"
    source_root = _apply_source_root(workspace_root, apply_package_id)
    source_root.mkdir(parents=True)
    (source_root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    apply_ref = _apply_ref(apply_package_id)
    return workspace_root, apply_package_id, apply_ref


def _apply_source_root(workspace_root: Path, apply_package_id: str) -> Path:
    """Resolve the managed apply source fixture path."""

    source_root = workspace_root / "patch_apply" / apply_package_id / "source"
    return source_root


def _apply_ref(apply_package_id: str) -> dict[str, object]:
    """Build the public managed apply workspace reference fixture."""

    apply_ref = {
        "kind": "managed_apply_workspace",
        "apply_package_id": apply_package_id,
        "source_identity": {
            "provider": "github",
            "owner": "fixture",
            "repo": "demo",
            "current_commit": "abc123",
            "dirty_state": "clean",
        },
        "applied_files": ["app.py"],
    }
    return apply_ref
