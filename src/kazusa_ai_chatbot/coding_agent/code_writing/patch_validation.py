"""Unified-diff safety and sandbox validation for patch proposals."""

from __future__ import annotations

import ast
import builtins
import io
import os
import re
import shutil
import subprocess
import sys
import tokenize
import uuid
from pathlib import Path, PurePosixPath

from kazusa_ai_chatbot.coding_agent.code_reading.planner import (
    is_binary_like_path,
    is_secret_like_path,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    PatchArtifact,
    PatchValidationSummary,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)

VALIDATION_ROOT_NAME = "writing_validation"
MAX_VALIDATION_SECONDS = 20
MAX_VALIDATION_TEST_OUTPUT_CHARS = 1600
MAX_SYMBOL_ERROR_ITEMS = 8
MAX_IMPORT_ERROR_ITEMS = 8
_DIFF_HEADER_RE = re.compile(r"^diff --git (?P<old>\S+) (?P<new>\S+)$")
_FILE_MARKER_RE = re.compile(r"^(?:---|\+\+\+) (?P<path>\S+)")
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,}=.*$")
_RAISE_OR_EXCEPT_NAME_RE = re.compile(
    r"\b(?:raise|except)\s+([A-Z][A-Za-z0-9_]*)\b"
)
_PYTEST_RAISES_NAME_RE = re.compile(
    r"\bpytest\.raises\(\s*([A-Z][A-Za-z0-9_]*)\b"
)
_CALL_NAME_RE = re.compile(r"(?<![.\w])([A-Z][A-Za-z0-9_]*)\s*\(")
_RESPONSE_TEXT_ASSERT_RE = re.compile(
    r"assert\s+(?P<quote>[\"'])(?P<value>.*?)(?P=quote)\s+in\s+"
    r"(?:str\()?response\.text"
)


def validate_patch_artifacts(
    *,
    repo_root: Path | None,
    workspace_root: Path,
    patch_artifacts: list[PatchArtifact],
    max_files: int,
    max_diff_chars: int,
) -> PatchValidationSummary:
    """Validate patch artifacts without mutating the target repository.

    Args:
        repo_root: Optional existing repository root used as the validation
            base. New-project proposals may pass `None`.
        workspace_root: Caller-configured storage root for isolated validation.
        patch_artifacts: Unified-diff artifacts to parse and check.
        max_files: Maximum distinct repo-relative paths allowed.
        max_diff_chars: Maximum combined diff text accepted.

    Returns:
        Public-safe validation status with no filesystem paths.
    """

    parse_result = _parse_patch_artifacts(
        patch_artifacts=patch_artifacts,
        max_files=max_files,
        max_diff_chars=max_diff_chars,
    )
    if parse_result["errors"]:
        summary = _summary(
            status="rejected",
            parsed=False,
            sandbox_applied=False,
            errors=parse_result["errors"],
            warnings=parse_result["warnings"],
            files=parse_result["files"],
        )
        return summary

    sandbox_result = _sandbox_check(
        repo_root=repo_root,
        workspace_root=workspace_root,
        diff_text=parse_result["diff_text"],
        files=parse_result["files"],
    )
    if sandbox_result is not None:
        summary = _summary(
            status="failed",
            parsed=True,
            sandbox_applied=False,
            errors=[sandbox_result],
            warnings=parse_result["warnings"],
            files=parse_result["files"],
        )
        return summary

    summary = _summary(
        status="succeeded",
        parsed=True,
        sandbox_applied=True,
        errors=[],
        warnings=parse_result["warnings"],
        files=parse_result["files"],
    )
    return summary


def _parse_patch_artifacts(
    *,
    patch_artifacts: list[PatchArtifact],
    max_files: int,
    max_diff_chars: int,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    files: list[str] = []
    diff_parts: list[str] = []
    total_chars = 0

    if not patch_artifacts:
        errors.append("No patch artifacts were provided.")

    for artifact in patch_artifacts:
        diff_text = artifact.get("diff_text", "")
        if not isinstance(diff_text, str) or not diff_text.strip():
            errors.append("Patch artifact omitted unified diff text.")
            continue
        total_chars += len(diff_text)
        if total_chars > max_diff_chars:
            errors.append("Patch artifact diff text exceeds the configured limit.")
            continue
        artifact_files = _paths_from_diff(diff_text)
        declared_files = artifact.get("files", [])
        if isinstance(declared_files, list):
            for declared_file in declared_files:
                if isinstance(declared_file, str):
                    artifact_files.append(declared_file)

        for path_text in artifact_files:
            safe_path = _safe_repo_relative_path(path_text)
            if safe_path is None:
                errors.append("Patch artifact includes an unsafe path.")
                continue
            if safe_path not in files:
                files.append(safe_path)
        if not artifact_files:
            errors.append("Patch artifact did not include file paths.")
        diff_parts.append(_diff_text_with_trailing_newline(diff_text))

    if len(files) > max_files:
        errors.append("Patch artifact touches too many files.")
    if not files and patch_artifacts:
        errors.append("No safe file paths were parsed from patch artifacts.")
    parsed = {
        "errors": errors,
        "warnings": warnings,
        "files": files[:max_files],
        "diff_text": "".join(diff_parts),
    }
    return parsed


def _diff_text_with_trailing_newline(diff_text: str) -> str:
    if diff_text.endswith("\n"):
        return diff_text
    return diff_text + "\n"


def _paths_from_diff(diff_text: str) -> list[str]:
    paths: list[str] = []
    for line in diff_text.splitlines():
        header_match = _DIFF_HEADER_RE.match(line)
        if header_match is not None:
            _append_diff_path(paths, header_match.group("old"))
            _append_diff_path(paths, header_match.group("new"))
            continue

        marker_match = _FILE_MARKER_RE.match(line)
        if marker_match is None:
            continue
        marker_path = marker_match.group("path")
        if marker_path == "/dev/null":
            continue
        _append_diff_path(paths, marker_path)
    return paths


def _append_diff_path(paths: list[str], raw_path: str) -> None:
    if raw_path == "/dev/null":
        return
    path_text = raw_path
    if path_text.startswith("a/") or path_text.startswith("b/"):
        path_text = path_text[2:]
    paths.append(path_text)


def _safe_repo_relative_path(path_text: str) -> str | None:
    stripped = path_text.strip().replace("\\", "/")
    if not stripped:
        return None
    path = PurePosixPath(stripped)
    if path.is_absolute() or ".." in path.parts:
        return None
    if any(part in ("", ".git") for part in path.parts):
        return None
    lowered_name = path.name.casefold()
    if lowered_name == ".env" or lowered_name.startswith(".env."):
        return None
    if is_secret_like_path(path.as_posix()):
        return None
    if is_binary_like_path(path.as_posix()):
        return None
    safe_path = path.as_posix()
    return safe_path


def _sandbox_check(
    *,
    repo_root: Path | None,
    workspace_root: Path,
    diff_text: str,
    files: list[str],
) -> str | None:
    try:
        sandbox_root = _prepare_sandbox(repo_root, workspace_root)
    except PathSafetyError:
        return "Patch validation storage could not be prepared."
    except OSError:
        return "Patch validation base could not be copied."

    completed = _run_git_apply(sandbox_root=sandbox_root, diff_text=diff_text)
    if completed == "git-unavailable":
        return "git executable is unavailable for patch validation."
    if completed == "timed-out":
        return "Patch validation timed out."
    if completed.returncode != 0:
        return _git_apply_error(completed.stderr)
    if _git_apply_has_malformed_warning(completed.stderr):
        return _git_apply_error(completed.stderr)

    applied = _run_git_apply(
        sandbox_root=sandbox_root,
        diff_text=diff_text,
        check_only=False,
    )
    if applied == "git-unavailable":
        return "git executable is unavailable for patch validation."
    if applied == "timed-out":
        return "Patch validation timed out."
    if applied.returncode != 0:
        return _git_apply_error(applied.stderr)
    if _git_apply_has_malformed_warning(applied.stderr):
        return _git_apply_error(applied.stderr)

    syntax_error = _python_syntax_error(sandbox_root=sandbox_root, files=files)
    if syntax_error is not None:
        return syntax_error

    broad_exception_error = _broad_exception_error(
        diff_text=diff_text,
        files=files,
    )
    if broad_exception_error is not None:
        return broad_exception_error

    test_exception_error = _broad_test_exception_error(
        diff_text=diff_text,
        files=files,
    )
    if test_exception_error is not None:
        return test_exception_error

    test_exception_type_error = _test_exception_type_change_error(
        diff_text=diff_text,
        files=files,
    )
    if test_exception_type_error is not None:
        return test_exception_type_error

    test_exception_match_error = _test_exception_match_change_error(
        diff_text=diff_text,
        files=files,
    )
    if test_exception_match_error is not None:
        return test_exception_match_error

    response_text_error = _response_text_assertion_error(
        diff_text=diff_text,
        files=files,
    )
    if response_text_error is not None:
        return response_text_error

    test_assertion_error = _test_assertion_error(
        diff_text=diff_text,
        files=files,
    )
    if test_assertion_error is not None:
        return test_assertion_error

    behavior_error = _runtime_behavior_error(diff_text=diff_text, files=files)
    if behavior_error is not None:
        return behavior_error

    symbol_error = _python_symbol_error(
        sandbox_root=sandbox_root,
        diff_text=diff_text,
        files=files,
    )
    if symbol_error is not None:
        return symbol_error

    import_error = _python_import_error(
        sandbox_root=sandbox_root,
        files=files,
        require_dependency_metadata=repo_root is None,
    )
    if import_error is not None:
        return import_error

    module_reference_error = _python_module_reference_error(
        sandbox_root=sandbox_root,
        files=files,
    )
    if module_reference_error is not None:
        return module_reference_error

    test_execution_error = _python_test_execution_error(
        sandbox_root=sandbox_root,
        files=files,
    )
    if test_execution_error is not None:
        return test_execution_error

    markdown_error = _markdown_content_error(
        sandbox_root=sandbox_root,
        files=files,
    )
    if markdown_error is not None:
        return markdown_error
    return None


def _run_git_apply(
    *,
    sandbox_root: Path,
    diff_text: str,
    check_only: bool = True,
) -> subprocess.CompletedProcess[str] | str:
    args = ["git", "apply"]
    if check_only:
        args.append("--check")
    args.extend(["--recount", "--"])
    env = _sandbox_git_env(sandbox_root)
    try:
        completed = subprocess.run(
            args,
            cwd=sandbox_root,
            env=env,
            input=diff_text,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=MAX_VALIDATION_SECONDS,
        )
    except FileNotFoundError:
        return "git-unavailable"
    except subprocess.TimeoutExpired:
        return "timed-out"
    return completed


def _python_syntax_error(*, sandbox_root: Path, files: list[str]) -> str | None:
    for safe_path in files:
        if Path(safe_path).suffix.casefold() != ".py":
            continue
        file_path = ensure_path_inside(sandbox_root / safe_path, sandbox_root)
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "Patched Python content could not be inspected."
        try:
            ast.parse(text)
        except SyntaxError as exc:
            return _python_syntax_error_message(safe_path=safe_path, error=exc)
    return None


def _python_syntax_error_message(*, safe_path: str, error: SyntaxError) -> str:
    line_number = error.lineno if error.lineno is not None else "unknown"
    detail = error.msg or "invalid syntax"
    return (
        "Patched Python content is not syntactically valid: "
        f"{safe_path} line {line_number}: {detail}."
    )


def _python_test_execution_error(
    *,
    sandbox_root: Path,
    files: list[str],
) -> str | None:
    test_files = _existing_test_python_files(
        sandbox_root=sandbox_root,
        files=files,
    )
    if not test_files:
        return None

    config_path = ensure_path_inside(
        sandbox_root / ".writing_validation_pytest.ini",
        sandbox_root,
    )
    try:
        config_path.write_text("[pytest]\naddopts =\n", encoding="utf-8")
    except OSError:
        return "Patched Python tests could not be prepared."

    args = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "-c",
        config_path.name,
        *test_files,
    ]
    try:
        completed = subprocess.run(
            args,
            cwd=sandbox_root,
            env=_sandbox_test_env(sandbox_root=sandbox_root),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=MAX_VALIDATION_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "Patched Python tests timed out in isolated validation."

    if completed.returncode == 0:
        return None
    return _python_test_failure_error(
        completed=completed,
        sandbox_root=sandbox_root,
    )


def _existing_test_python_files(
    *,
    sandbox_root: Path,
    files: list[str],
) -> list[str]:
    test_files: list[str] = []
    for safe_path in files:
        if Path(safe_path).suffix.casefold() != ".py":
            continue
        if not _is_test_path(safe_path):
            continue
        file_path = ensure_path_inside(sandbox_root / safe_path, sandbox_root)
        if not file_path.exists() or not file_path.is_file():
            continue
        test_files.append(safe_path)
    return sorted(test_files)


def _sandbox_test_env(*, sandbox_root: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in (
        "COMSPEC",
        "PATH",
        "PATHEXT",
        "SystemRoot",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
    ):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    src_root = ensure_path_inside(sandbox_root / "src", sandbox_root)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(src_root), str(sandbox_root)]
    )
    return env


def _python_test_failure_error(
    *,
    completed: subprocess.CompletedProcess[str],
    sandbox_root: Path,
) -> str:
    output = "\n".join(
        part
        for part in (completed.stdout, completed.stderr)
        if part
    )
    compact_output = _compact_validation_output(
        output,
        sandbox_root=sandbox_root,
    )
    if not compact_output:
        compact_output = f"pytest exited with code {completed.returncode}"
    return (
        "Patched Python tests fail in isolated validation: "
        f"{compact_output}"
    )


def _compact_validation_output(text: str, *, sandbox_root: Path) -> str:
    root_text = str(sandbox_root)
    normalized_root = root_text.replace("\\", "/")
    compact = text.replace(root_text, "[validation-root]")
    compact = compact.replace(normalized_root, "[validation-root]")
    compact = " ".join(compact.split())
    if len(compact) > MAX_VALIDATION_TEST_OUTPUT_CHARS:
        compact = compact[:MAX_VALIDATION_TEST_OUTPUT_CHARS].rstrip()
    return compact


def _sandbox_git_env(sandbox_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    ceiling = str(sandbox_root.parent.resolve(strict=False))
    existing = env.get("GIT_CEILING_DIRECTORIES")
    if existing:
        env["GIT_CEILING_DIRECTORIES"] = (
            ceiling + os.pathsep + existing
        )
    else:
        env["GIT_CEILING_DIRECTORIES"] = ceiling
    return env


def _broad_exception_error(*, diff_text: str, files: list[str]) -> str | None:
    if not any(_is_runtime_python_path(safe_path) for safe_path in files):
        return None
    broad_catches = _added_broad_runtime_exception_lines(diff_text)
    if not broad_catches:
        return None
    return (
        "Patch adds broad runtime exception wrapping: "
        f"{_broad_exception_detail(broad_catches)}. Remove the new try/except "
        "and modify the smallest existing error branch, or use a specific "
        "observed exception type while preserving original exception "
        "propagation."
    )


def _added_broad_runtime_exception_lines(diff_text: str) -> list[tuple[str, str]]:
    current_path: str | None = None
    broad_catches: list[tuple[str, str]] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            current_path = _safe_path_from_diff_marker(line[4:])
            continue
        if current_path is None or not _is_runtime_python_path(current_path):
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        stripped = line[1:].strip()
        if stripped.startswith(("except Exception", "except BaseException")):
            broad_catches.append((current_path, stripped))
    return broad_catches


def _broad_exception_detail(broad_catches: list[tuple[str, str]]) -> str:
    items: list[str] = []
    for safe_path, stripped in broad_catches[:MAX_SYMBOL_ERROR_ITEMS]:
        items.append(f"{safe_path} adds `{stripped}`")
    detail = "; ".join(items)
    if len(broad_catches) > len(items):
        detail = f"{detail}; additional lines omitted"
    return detail


def _broad_test_exception_error(*, diff_text: str, files: list[str]) -> str | None:
    if not any(_is_test_path(safe_path) for safe_path in files):
        return None
    if not _diff_adds_broad_test_exception(diff_text):
        return None
    return (
        "Patch adds broad exception handling in tests; use a specific "
        "exception type or a test-framework exception assertion so the test "
        "fails when the expected exception is not raised."
    )


def _diff_adds_broad_test_exception(diff_text: str) -> bool:
    current_path: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            current_path = _safe_path_from_diff_marker(line[4:])
            continue
        if current_path is None or not _is_test_path(current_path):
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        stripped = line[1:].strip()
        if stripped.startswith(("except Exception", "except BaseException")):
            return True
    return False


def _test_exception_type_change_error(
    *,
    diff_text: str,
    files: list[str],
) -> str | None:
    if not any(_is_test_path(safe_path) for safe_path in files):
        return None
    changed_names = _changed_test_exception_names(diff_text)
    if not changed_names:
        return None
    if _diff_adds_runtime_raise_for_any(diff_text, changed_names):
        return None
    return (
        "Patch changes test expected exception types without matching runtime "
        "raise changes."
    )


def _changed_test_exception_names(diff_text: str) -> set[str]:
    removed_names = {
        block["exception"]
        for block in _pytest_raises_blocks_from_diff(diff_text, "-")
    }
    if not removed_names:
        return set()
    added_names = {
        block["exception"]
        for block in _pytest_raises_blocks_from_diff(diff_text, "+")
    }
    return added_names - removed_names


def _test_exception_match_change_error(
    *,
    diff_text: str,
    files: list[str],
) -> str | None:
    if not any(_is_test_path(safe_path) for safe_path in files):
        return None
    changed_names = _changed_test_exception_match_names(diff_text)
    if not changed_names:
        return None
    if _diff_adds_runtime_raise_for_any(diff_text, changed_names):
        return None
    return (
        "Patch changes test expected exception messages without matching "
        "runtime raise or message changes for the same exception type."
    )


def _changed_test_exception_match_names(diff_text: str) -> set[str]:
    removed_by_exception = _pytest_raises_matches_by_exception(
        _pytest_raises_blocks_from_diff(diff_text, "-")
    )
    added_by_exception = _pytest_raises_matches_by_exception(
        _pytest_raises_blocks_from_diff(diff_text, "+")
    )
    changed_names: set[str] = set()
    for exception_name, added_matches in added_by_exception.items():
        removed_matches = removed_by_exception.get(exception_name)
        if not removed_matches:
            continue
        if added_matches == removed_matches:
            continue
        changed_names.add(exception_name)
    return changed_names


def _pytest_raises_matches_by_exception(
    blocks: list[dict[str, str]],
) -> dict[str, set[str]]:
    matches_by_exception: dict[str, set[str]] = {}
    for block in blocks:
        match_text = block.get("match", "")
        if not match_text:
            continue
        matches_by_exception.setdefault(block["exception"], set()).add(
            match_text
        )
    return matches_by_exception


def _pytest_raises_blocks_from_diff(
    diff_text: str,
    prefix: str,
) -> list[dict[str, str]]:
    current_path: str | None = None
    collecting = False
    block_lines: list[str] = []
    blocks: list[dict[str, str]] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            current_path = _safe_path_from_diff_marker(line[4:])
            collecting = False
            block_lines = []
            continue
        if current_path is None or not _is_test_path(current_path):
            continue
        if line.startswith(("---", "+++", "@@")):
            continue
        if not line.startswith(prefix):
            continue
        stripped = line[1:].strip()
        if collecting:
            block_lines.append(stripped)
            if _pytest_raises_block_complete(stripped):
                _append_pytest_raises_block(blocks, block_lines)
                collecting = False
                block_lines = []
            continue
        if "pytest.raises" not in stripped:
            continue
        block_lines = [stripped]
        if _pytest_raises_block_complete(stripped):
            _append_pytest_raises_block(blocks, block_lines)
            block_lines = []
            continue
        collecting = True
    if collecting:
        _append_pytest_raises_block(blocks, block_lines)
    return blocks


def _pytest_raises_block_complete(stripped: str) -> bool:
    return stripped.endswith(":") or "):" in stripped


def _response_text_assertion_error(
    *,
    diff_text: str,
    files: list[str],
) -> str | None:
    if not any(_is_test_path(safe_path) for safe_path in files):
        return None

    current_path: str | None = None
    hunk_lines: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            error = _response_text_hunk_error(
                current_path=current_path,
                hunk_lines=hunk_lines,
            )
            if error is not None:
                return error
            current_path = _safe_path_from_diff_marker(line[4:])
            hunk_lines = []
            continue
        if line.startswith("@@"):
            error = _response_text_hunk_error(
                current_path=current_path,
                hunk_lines=hunk_lines,
            )
            if error is not None:
                return error
            hunk_lines = []
            continue
        if current_path is None or not _is_test_path(current_path):
            continue
        if line.startswith((" ", "+", "-")) and not line.startswith(
            ("+++", "---")
        ):
            hunk_lines.append(line)

    error = _response_text_hunk_error(
        current_path=current_path,
        hunk_lines=hunk_lines,
    )
    if error is not None:
        return error
    return None


def _response_text_hunk_error(
    *,
    current_path: str | None,
    hunk_lines: list[str],
) -> str | None:
    if current_path is None or not _is_test_path(current_path):
        return None
    expected_values = _added_response_text_expected_values(hunk_lines)
    if not expected_values:
        return None
    visible_body = _visible_response_body_assertions(hunk_lines)
    if not visible_body:
        return None
    for expected_value in expected_values:
        if expected_value not in visible_body:
            return (
                "Patch adds response text assertions that are not supported "
                "by the visible response body assertions."
            )
    return None


def _added_response_text_expected_values(hunk_lines: list[str]) -> list[str]:
    expected_values: list[str] = []
    for line in hunk_lines:
        if not line.startswith("+") or line.startswith("+++"):
            continue
        match = _RESPONSE_TEXT_ASSERT_RE.search(line[1:].strip())
        if match is None:
            continue
        expected_values.append(match.group("value"))
    return expected_values


def _visible_response_body_assertions(hunk_lines: list[str]) -> str:
    body_lines: list[str] = []
    for line in hunk_lines:
        if not line.startswith((" ", "+")) or line.startswith("+++"):
            continue
        stripped = line[1:].strip()
        if "response.json()" in stripped and "==" in stripped:
            body_lines.append(stripped)
            continue
        if "response.text" in stripped and "==" in stripped:
            body_lines.append(stripped)
    return "\n".join(body_lines)


def _pytest_raises_names(line: str) -> set[str]:
    names: set[str] = set()
    for match in _PYTEST_RAISES_NAME_RE.finditer(line.strip()):
        names.add(match.group(1))
    return names


def _append_pytest_raises_block(
    blocks: list[dict[str, str]],
    block_lines: list[str],
) -> None:
    block_text = " ".join(block_lines)
    name_match = re.search(
        r"\bpytest\.raises\(\s*([A-Z][A-Za-z0-9_]*)\b",
        block_text,
    )
    if name_match is None:
        return
    block = {"exception": name_match.group(1)}
    match_match = re.search(
        r"\bmatch\s*=\s*(?:[rRuUbBfF]+)?([\"'])(?P<value>.*?)\1",
        block_text,
    )
    if match_match is not None:
        block["match"] = match_match.group("value")
    blocks.append(block)


def _diff_adds_runtime_raise_for_any(
    diff_text: str,
    names: set[str],
) -> bool:
    current_path: str | None = None
    hunk_raise_names: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            current_path = _safe_path_from_diff_marker(line[4:])
            hunk_raise_names = set()
            continue
        if line.startswith("@@"):
            hunk_raise_names = set()
            continue
        if current_path is None or not _is_runtime_python_path(current_path):
            continue
        if line.startswith((" ", "+", "-")) and not line.startswith(
            ("+++", "---")
        ):
            hunk_raise_names.update(_line_raise_names(line[1:], names))
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if _line_raises_any_name(line[1:], names):
            return True
        if hunk_raise_names.intersection(names) and _line_can_change_message(
            line[1:]
        ):
            return True
    return False


def _line_raises_any_name(line: str, names: set[str]) -> bool:
    return bool(_line_raise_names(line, names))


def _line_raise_names(line: str, names: set[str]) -> set[str]:
    stripped = line.strip()
    raised_names: set[str] = set()
    for name in names:
        pattern = rf"\braise\s+(?:[A-Za-z_][A-Za-z0-9_]*\.)?{re.escape(name)}\b"
        if re.search(pattern, stripped) is not None:
            raised_names.add(name)
    return raised_names


def _line_can_change_message(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    return stripped.startswith(("'", '"', "f'", 'f"', "r'", 'r"'))


def _test_assertion_error(*, diff_text: str, files: list[str]) -> str | None:
    if not any(_is_test_path(safe_path) for safe_path in files):
        return None
    if _diff_adds_executable_test_assertion(diff_text):
        return None
    return (
        "Patch changes test files without adding or modifying executable "
        "test assertions."
    )


def _diff_adds_executable_test_assertion(diff_text: str) -> bool:
    current_path: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            current_path = _safe_path_from_diff_marker(line[4:])
            continue
        if current_path is None or not _is_test_path(current_path):
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        stripped = line[1:].strip()
        if _is_executable_test_assertion_line(stripped):
            return True
    return False


def _is_executable_test_assertion_line(stripped: str) -> bool:
    if not stripped or stripped.startswith("#"):
        return False
    assertion_markers = (
        "assert ",
        "with pytest.raises",
        "pytest.raises(",
        "self.assert",
        "assertRaises",
    )
    return any(marker in stripped for marker in assertion_markers)


def _python_symbol_error(
    *,
    sandbox_root: Path,
    diff_text: str,
    files: list[str],
) -> str | None:
    symbol_uses = _added_python_symbol_uses_by_file(diff_text)
    if not symbol_uses:
        return None

    missing_by_file: dict[str, list[str]] = {}
    safe_files = {
        safe_path
        for safe_path in files
        if Path(safe_path).suffix.casefold() == ".py"
    }
    for safe_path, used_names in symbol_uses.items():
        if safe_path not in safe_files:
            continue
        file_path = ensure_path_inside(sandbox_root / safe_path, sandbox_root)
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "Patched Python content could not be inspected."
        names_result = _defined_or_imported_python_names(text)
        if names_result is None:
            return "Patched Python content is not syntactically valid."
        available_names, has_wildcard_import = names_result
        if has_wildcard_import:
            continue
        missing_names = [
            name
            for name in sorted(used_names)
            if name not in available_names
        ]
        if missing_names:
            missing_by_file[safe_path] = missing_names
    if missing_by_file:
        return _missing_python_symbol_error(missing_by_file)
    return None


def _missing_python_symbol_error(
    missing_by_file: dict[str, list[str]],
) -> str:
    items: list[str] = []
    for safe_path in sorted(missing_by_file):
        names = ", ".join(sorted(missing_by_file[safe_path]))
        items.append(f"{safe_path}: {names}")
        if len(items) >= MAX_SYMBOL_ERROR_ITEMS:
            break

    suffix = "."
    if len(items) < len(missing_by_file):
        suffix = "; additional files omitted."
    detail = "; ".join(items)
    return (
        "Patch uses Python symbols without imports or local definitions: "
        f"{detail}{suffix}"
    )


def _python_import_error(
    *,
    sandbox_root: Path,
    files: list[str],
    require_dependency_metadata: bool,
) -> str | None:
    python_files = _existing_python_files(sandbox_root=sandbox_root, files=files)
    if not python_files:
        return None

    module_index = _local_python_module_index(python_files)
    exported_names = _exported_names_by_module(
        sandbox_root=sandbox_root,
        module_index=module_index,
    )
    if exported_names is None:
        return "Patched Python content is not syntactically valid."

    has_dependency_metadata = _has_dependency_metadata(files)
    unresolved_imports: list[str] = []
    for safe_path in sorted(python_files):
        tree = _python_ast_for_file(sandbox_root=sandbox_root, safe_path=safe_path)
        if tree is None:
            return "Patched Python content is not syntactically valid."
        package_parts = _package_parts_for_python_file(safe_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                _append_import_errors(
                    unresolved_imports=unresolved_imports,
                    safe_path=safe_path,
                    node=node,
                    module_index=module_index,
                    require_dependency_metadata=require_dependency_metadata,
                    has_dependency_metadata=has_dependency_metadata,
                )
                continue
            if isinstance(node, ast.ImportFrom):
                _append_from_import_errors(
                    unresolved_imports=unresolved_imports,
                    safe_path=safe_path,
                    node=node,
                    package_parts=package_parts,
                    module_index=module_index,
                    exported_names=exported_names,
                    require_dependency_metadata=require_dependency_metadata,
                    has_dependency_metadata=has_dependency_metadata,
                )

    if not unresolved_imports:
        return None

    error = _python_import_error_message(
        unresolved_imports=unresolved_imports,
        module_index=module_index,
    )
    return error


def _existing_python_files(*, sandbox_root: Path, files: list[str]) -> list[str]:
    python_files: list[str] = []
    for safe_path in files:
        if Path(safe_path).suffix.casefold() != ".py":
            continue
        file_path = ensure_path_inside(sandbox_root / safe_path, sandbox_root)
        if not file_path.exists() or not file_path.is_file():
            continue
        python_files.append(safe_path)
    return python_files


def _local_python_module_index(python_files: list[str]) -> dict[str, str]:
    module_index: dict[str, str] = {}
    for safe_path in python_files:
        for module_name in _module_names_for_python_path(safe_path):
            module_index[module_name] = safe_path
    return module_index


def _module_names_for_python_path(safe_path: str) -> list[str]:
    path = PurePosixPath(safe_path)
    path_without_suffix = path.with_suffix("")
    parts = list(path_without_suffix.parts)
    if path.name == "__init__.py":
        parts = list(path.parent.parts)
    if not parts:
        return []
    module_name = ".".join(parts)
    return [module_name]


def _exported_names_by_module(
    *,
    sandbox_root: Path,
    module_index: dict[str, str],
) -> dict[str, set[str]] | None:
    exported_names: dict[str, set[str]] = {}
    parsed_by_path: dict[str, ast.Module] = {}
    for module_name, safe_path in module_index.items():
        tree = parsed_by_path.get(safe_path)
        if tree is None:
            tree = _python_ast_for_file(
                sandbox_root=sandbox_root,
                safe_path=safe_path,
            )
            if tree is None:
                return None
            parsed_by_path[safe_path] = tree
        exported_names[module_name] = _module_exported_names(tree)
    return exported_names


def _python_ast_for_file(
    *,
    sandbox_root: Path,
    safe_path: str,
) -> ast.Module | None:
    file_path = ensure_path_inside(sandbox_root / safe_path, sandbox_root)
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    return tree


def _module_exported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
            continue
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name)
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_assigned_names(target))
            continue
        if isinstance(node, ast.AnnAssign):
            names.update(_assigned_names(node.target))
    return names


def _append_import_errors(
    *,
    unresolved_imports: list[str],
    safe_path: str,
    node: ast.Import,
    module_index: dict[str, str],
    require_dependency_metadata: bool,
    has_dependency_metadata: bool,
) -> None:
    if not require_dependency_metadata or has_dependency_metadata:
        return
    for alias in node.names:
        imported_name = alias.name
        if _import_is_known(imported_name, module_index):
            continue
        unresolved_imports.append(f"{safe_path} imports {imported_name}")


def _append_from_import_errors(
    *,
    unresolved_imports: list[str],
    safe_path: str,
    node: ast.ImportFrom,
    package_parts: list[str],
    module_index: dict[str, str],
    exported_names: dict[str, set[str]],
    require_dependency_metadata: bool,
    has_dependency_metadata: bool,
) -> None:
    if node.module == "__future__":
        return
    module_name = _resolved_from_import_module(
        node=node,
        package_parts=package_parts,
    )
    if not module_name:
        return
    if module_name in module_index:
        _append_missing_imported_names(
            unresolved_imports=unresolved_imports,
            safe_path=safe_path,
            module_name=module_name,
            node=node,
            module_index=module_index,
            exported_names=exported_names,
        )
        return

    if _from_import_targets_local_submodule(node, module_name, module_index):
        return

    if not require_dependency_metadata or has_dependency_metadata:
        return
    if _import_is_known(module_name, module_index):
        return
    unresolved_imports.append(f"{safe_path} imports {module_name}")


def _append_missing_imported_names(
    *,
    unresolved_imports: list[str],
    safe_path: str,
    module_name: str,
    node: ast.ImportFrom,
    module_index: dict[str, str],
    exported_names: dict[str, set[str]],
) -> None:
    available_names = exported_names[module_name]
    for alias in node.names:
        if alias.name == "*":
            continue
        submodule_name = f"{module_name}.{alias.name}"
        if submodule_name in module_index:
            continue
        if alias.name in available_names:
            continue
        unresolved_imports.append(
            f"{safe_path} imports {alias.name} from {module_name}"
        )


def _resolved_from_import_module(
    *,
    node: ast.ImportFrom,
    package_parts: list[str],
) -> str:
    if node.level <= 0:
        module_name = node.module or ""
        return module_name

    keep_count = max(0, len(package_parts) - node.level + 1)
    parts = package_parts[:keep_count]
    if node.module:
        parts.extend(node.module.split("."))
    module_name = ".".join(parts)
    return module_name


def _package_parts_for_python_file(safe_path: str) -> list[str]:
    path = PurePosixPath(safe_path)
    if path.name == "__init__.py":
        parts = list(path.parent.parts)
        return parts
    parts = list(path.parent.parts)
    return parts


def _from_import_targets_local_submodule(
    node: ast.ImportFrom,
    module_name: str,
    module_index: dict[str, str],
) -> bool:
    if node.module is not None:
        return False
    for alias in node.names:
        submodule_name = f"{module_name}.{alias.name}"
        if submodule_name in module_index:
            return True
    return False


def _import_is_known(
    module_name: str,
    module_index: dict[str, str],
) -> bool:
    top_level = module_name.split(".", 1)[0]
    if top_level in sys.builtin_module_names:
        return True
    if top_level in getattr(sys, "stdlib_module_names", set()):
        return True
    if module_name in module_index:
        return True
    if top_level in module_index:
        return True
    for local_name in module_index:
        if local_name.startswith(f"{top_level}."):
            return True
    return False


def _has_dependency_metadata(files: list[str]) -> bool:
    metadata_names = {
        "Pipfile",
        "pyproject.toml",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
        "uv.lock",
    }
    for safe_path in files:
        name = PurePosixPath(safe_path).name
        if name in metadata_names:
            return True
    return False


def _python_import_error_message(
    *,
    unresolved_imports: list[str],
    module_index: dict[str, str],
) -> str:
    unique_imports: list[str] = []
    for item in unresolved_imports:
        if item in unique_imports:
            continue
        unique_imports.append(item)

    items = unique_imports[:MAX_IMPORT_ERROR_ITEMS]
    detail = "; ".join(items)
    suffix = "."
    if len(unique_imports) > len(items):
        suffix = "; additional imports omitted."
    message = f"Patch has unresolved local Python imports: {detail}{suffix}"
    available_modules = _available_runtime_module_names(module_index)
    if available_modules:
        joined_modules = ", ".join(available_modules)
        message = (
            f"{message} Available local Python modules: {joined_modules}."
        )
    return message


def _available_runtime_module_names(module_index: dict[str, str]) -> list[str]:
    module_names: list[str] = []
    for module_name, safe_path in sorted(module_index.items()):
        if _is_test_path(safe_path):
            continue
        if module_name in module_names:
            continue
        module_names.append(module_name)
        if len(module_names) >= MAX_IMPORT_ERROR_ITEMS:
            break
    return module_names


def _python_module_reference_error(
    *,
    sandbox_root: Path,
    files: list[str],
) -> str | None:
    missing_by_file: dict[str, list[str]] = {}
    python_files = _existing_python_files(sandbox_root=sandbox_root, files=files)
    for safe_path in python_files:
        tree = _python_ast_for_file(sandbox_root=sandbox_root, safe_path=safe_path)
        if tree is None:
            return "Patched Python content is not syntactically valid."
        visitor = _ModuleReferenceVisitor()
        visitor.visit(tree)
        if visitor.missing_references:
            missing_by_file[safe_path] = sorted(visitor.missing_references)

    if not missing_by_file:
        return None

    error = _missing_module_reference_error(missing_by_file)
    return error


class _ModuleReferenceVisitor(ast.NodeVisitor):
    """Find dotted references whose root name is not visible in scope."""

    def __init__(self) -> None:
        self._scopes: list[set[str]] = [set(dir(builtins))]
        self.missing_references: set[str] = set()

    def visit_Module(self, node: ast.Module) -> None:
        self._define_module_names(node)
        for item in node.body:
            self.visit(item)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._define(node.name)
        self._visit_function_body(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._define(node.name)
        self._visit_function_body(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._define(node.name)
        self._scopes.append(set())
        for item in node.body:
            self.visit(item)
        self._scopes.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._define(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            self._define(alias.asname or alias.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._define_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
        self._define_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._define_target(node.target)

    def visit_For(self, node: ast.For) -> None:
        self.visit(node.iter)
        self._define_target(node.target)
        for item in node.body:
            self.visit(item)
        for item in node.orelse:
            self.visit(item)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._define_target(node.target)
        for item in node.body:
            self.visit(item)
        for item in node.orelse:
            self.visit(item)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._define_target(item.optional_vars)
        for body_item in node.body:
            self.visit(body_item)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._define_target(item.optional_vars)
        for body_item in node.body:
            self.visit(body_item)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self._define(node.name)
        for item in node.body:
            self.visit(item)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.visit(node.iter)
        self._define_target(node.target)
        for item in node.ifs:
            self.visit(item)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension_scope(
            generators=node.generators,
            result_nodes=[node.elt],
        )

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension_scope(
            generators=node.generators,
            result_nodes=[node.elt],
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension_scope(
            generators=node.generators,
            result_nodes=[node.elt],
        )

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension_scope(
            generators=node.generators,
            result_nodes=[node.key, node.value],
        )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            root_name = node.value.id
            if not self._is_defined(root_name):
                self.missing_references.add(f"{root_name}.{node.attr}")
        self.generic_visit(node)

    def _visit_function_body(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        function_scope = _argument_names(node.args)
        self._scopes.append(function_scope)
        for item in node.body:
            self.visit(item)
        self._scopes.pop()

    def _visit_comprehension_scope(
        self,
        *,
        generators: list[ast.comprehension],
        result_nodes: list[ast.AST],
    ) -> None:
        self._scopes.append(set())
        for generator in generators:
            self.visit(generator.iter)
            self._define_target(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        for result_node in result_nodes:
            self.visit(result_node)
        self._scopes.pop()

    def _define(self, name: str) -> None:
        self._scopes[-1].add(name)

    def _define_target(self, target: ast.AST) -> None:
        for name in _assigned_names(target):
            self._define(name)

    def _is_defined(self, name: str) -> bool:
        for scope in reversed(self._scopes):
            if name in scope:
                return True
        return False

    def _define_module_names(self, node: ast.Module) -> None:
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self._define(item.name)
                continue
            if isinstance(item, ast.Import):
                for alias in item.names:
                    self._define(alias.asname or alias.name.split(".", 1)[0])
                continue
            if isinstance(item, ast.ImportFrom):
                for alias in item.names:
                    if alias.name == "*":
                        continue
                    self._define(alias.asname or alias.name)
                continue
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    self._define_target(target)
                continue
            if isinstance(item, ast.AnnAssign):
                self._define_target(item.target)


def _argument_names(args: ast.arguments) -> set[str]:
    names: set[str] = set()
    for arg in [
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
    ]:
        names.add(arg.arg)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def _missing_module_reference_error(
    missing_by_file: dict[str, list[str]],
) -> str:
    items: list[str] = []
    for safe_path in sorted(missing_by_file):
        references = ", ".join(missing_by_file[safe_path])
        items.append(f"{safe_path}: {references}")
        if len(items) >= MAX_SYMBOL_ERROR_ITEMS:
            break

    suffix = "."
    if len(items) < len(missing_by_file):
        suffix = "; additional files omitted."
    detail = "; ".join(items)
    return (
        "Patch uses Python module references without imports or local "
        f"definitions: {detail}{suffix}"
    )


def _added_python_symbol_uses_by_file(diff_text: str) -> dict[str, set[str]]:
    current_path: str | None = None
    uses_by_file: dict[str, set[str]] = {}
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            current_path = _safe_path_from_diff_marker(line[4:])
            continue
        if current_path is None:
            continue
        if Path(current_path).suffix.casefold() != ".py":
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        names = _python_symbol_uses_from_added_line(line[1:])
        if not names:
            continue
        uses_by_file.setdefault(current_path, set()).update(names)
    return uses_by_file


def _safe_path_from_diff_marker(raw_path: str) -> str | None:
    if raw_path == "/dev/null":
        return None
    path_text = raw_path
    if path_text.startswith("a/") or path_text.startswith("b/"):
        path_text = path_text[2:]
    return _safe_repo_relative_path(path_text)


def _python_symbol_uses_from_added_line(line: str) -> set[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith(("class ", "def ", "async def ")):
        return set()
    if stripped.startswith(("#", "'''", '"""', "r'''", 'r"""')):
        return set()

    scan_text = _python_line_without_strings_or_comments(stripped)
    names: set[str] = set()
    for pattern in (
        _RAISE_OR_EXCEPT_NAME_RE,
        _PYTEST_RAISES_NAME_RE,
        _CALL_NAME_RE,
    ):
        for match in pattern.finditer(scan_text):
            name = match.group(1)
            if name in {"Exception", "BaseException"}:
                continue
            if hasattr(builtins, name):
                continue
            names.add(name)
    return names


def _python_line_without_strings_or_comments(line: str) -> str:
    """Remove string/comment token spans before regex-based symbol scanning.

    Args:
        line: One added Python diff line without the leading `+`.

    Returns:
        The line with string and comment token spans replaced by spaces, keeping
        code token positions and punctuation intact for existing regex checks.
    """

    chars = list(line)
    try:
        tokens = tokenize.generate_tokens(io.StringIO(line).readline)
        token_list = list(tokens)
    except (IndentationError, tokenize.TokenError):
        return line

    for token in token_list:
        if token.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        start_column = token.start[1]
        if token.end[0] == token.start[0]:
            end_column = token.end[1]
        else:
            end_column = len(chars)
        for index in range(start_column, min(end_column, len(chars))):
            chars[index] = " "
    sanitized_line = "".join(chars)
    return sanitized_line


def _defined_or_imported_python_names(
    text: str,
) -> tuple[set[str], bool] | None:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    names = set(dir(builtins))
    has_wildcard_import = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
            continue
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    has_wildcard_import = True
                    continue
                names.add(alias.asname or alias.name)
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_assigned_names(target))
            continue
        if isinstance(node, ast.AnnAssign):
            names.update(_assigned_names(node.target))
    return names, has_wildcard_import


def _assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_assigned_names(item))
        return names
    return set()


def _runtime_behavior_error(*, diff_text: str, files: list[str]) -> str | None:
    if not any(_is_test_path(safe_path) for safe_path in files):
        return None
    if not any(_is_runtime_python_path(safe_path) for safe_path in files):
        return None
    if _diff_has_runtime_python_behavior(diff_text):
        return None
    return (
        "Patch updates tests but does not include executable runtime source "
        "changes."
    )


def _diff_has_runtime_python_behavior(diff_text: str) -> bool:
    current_path: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            current_path = _safe_path_from_diff_marker(line[4:])
            continue
        if current_path is None or not _is_runtime_python_path(current_path):
            continue
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if _added_python_line_has_runtime_behavior(line[1:]):
            return True
    return False


def _is_runtime_python_path(safe_path: str) -> bool:
    suffix = Path(safe_path).suffix.casefold()
    return suffix == ".py" and not _is_test_path(safe_path)


def _is_test_path(safe_path: str) -> bool:
    path = PurePosixPath(safe_path)
    name = path.name.casefold()
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part.casefold() in ("test", "tests") for part in path.parts)


def _added_python_line_has_runtime_behavior(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(("#", "class ", "import ", "from ")):
        return False
    if stripped.startswith(("'''", '"""')) or stripped.endswith(("'''", '"""')):
        return False
    if stripped.startswith(("'", '"')):
        return True

    behavior_prefixes = (
        "if ",
        "elif ",
        "else:",
        "for ",
        "while ",
        "try:",
        "except ",
        "with ",
        "async with ",
        "async for ",
        "return ",
        "raise ",
        "await ",
        "def ",
        "async def ",
    )
    if stripped.startswith(behavior_prefixes):
        return True
    if "=" in stripped and "==" not in stripped and "!=" not in stripped:
        return True
    return False


def _markdown_content_error(
    *,
    sandbox_root: Path,
    files: list[str],
) -> str | None:
    for safe_path in files:
        if not _is_markdown_path(safe_path):
            continue
        file_path = ensure_path_inside(sandbox_root / safe_path, sandbox_root)
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "Patched Markdown content could not be inspected."
        if _has_raw_env_assignment_outside_fence(text):
            return (
                "Markdown environment-style assignments must be inside "
                "fenced code blocks."
            )
    return None


def _is_markdown_path(safe_path: str) -> bool:
    suffix = Path(safe_path).suffix.casefold()
    return suffix in (".md", ".markdown")


def _has_raw_env_assignment_outside_fence(text: str) -> bool:
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if _ENV_ASSIGNMENT_RE.match(stripped) is not None:
            return True
    return False


def _git_apply_error(stderr: str) -> str:
    compact_error = " ".join(stderr.strip().split())
    if len(compact_error) > 500:
        compact_error = compact_error[:500].rstrip()
    if compact_error:
        return f"Patch did not apply cleanly in isolated validation: {compact_error}"
    return "Patch did not apply cleanly in isolated validation."


def _git_apply_has_malformed_warning(stderr: str) -> bool:
    lowered = stderr.casefold()
    return "recount:" in lowered or "unexpected line" in lowered


def _prepare_sandbox(repo_root: Path | None, workspace_root: Path) -> Path:
    root = workspace_root.expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    validation_root = ensure_path_inside(root / VALIDATION_ROOT_NAME, root)
    validation_root.mkdir(parents=True, exist_ok=True)
    sandbox_root = ensure_path_inside(validation_root / uuid.uuid4().hex, root)

    if repo_root is None:
        sandbox_root.mkdir(parents=True, exist_ok=False)
        return sandbox_root

    resolved_repo = repo_root.expanduser().resolve(strict=True)
    shutil.copytree(
        resolved_repo,
        sandbox_root,
        ignore=shutil.ignore_patterns(".git", ".tmp_pytest"),
    )
    return sandbox_root


def _summary(
    *,
    status: str,
    parsed: bool,
    sandbox_applied: bool,
    errors: list[str],
    warnings: list[str],
    files: list[str],
) -> PatchValidationSummary:
    summary: PatchValidationSummary = {
        "status": status,
        "parsed": parsed,
        "sandbox_applied": sandbox_applied,
        "errors": errors,
        "warnings": warnings,
        "files": files,
    }
    return summary
