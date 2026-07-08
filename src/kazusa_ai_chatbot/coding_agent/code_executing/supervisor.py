"""Validator and runner orchestration for bounded code execution."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import re
import sys
from collections.abc import Mapping

from kazusa_ai_chatbot.coding_agent.code_executing.models import (
    CodeExecutionRequest,
    CodeExecutionResponse,
    CodeExecutionStatus,
)
from kazusa_ai_chatbot.coding_agent.code_executing.runner import run_argv
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)

APPLY_ROOT_NAME = "patch_apply"
APPLIED_SOURCE_DIR_NAME = "source"
DEFAULT_OUTPUT_CHARS = 6000
MAX_OUTPUT_CHARS = 20_000
DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 60
MAX_EXECUTION_TARGETS = 32
MAX_TARGET_FILE_COUNT = 512
SAFE_PACKAGE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
UNSAFE_PATH_PARTS = {".git", ".env", "__pycache__"}
UNSAFE_PATH_FRAGMENTS = ("secret", "token", "credential", "password")
BINARY_EXTENSIONS = {
    ".bin",
    ".dll",
    ".exe",
    ".gif",
    ".ico",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".png",
    ".pyc",
    ".zip",
}
SHELL_TOKENS = ("|", "&", ";", "<", ">", "`", "$", "\n", "\r")


def run(request: CodeExecutionRequest) -> CodeExecutionResponse:
    """Validate and run one structured command in a managed apply workspace.

    Args:
        request: Trusted direct execution request from a caller that already
            owns approval, source identity, and command selection.

    Returns:
        Public-safe execution response with bounded output.
    """

    execution_value = request.get("execution")
    execution = _mapping_or_empty(execution_value)
    tool = _tool_name(execution)
    root_result = _managed_apply_source_root(request)
    if isinstance(root_result, str):
        response = _response(
            status="rejected",
            tool=tool,
            limitations=[root_result],
            trace_summary=["code_execution:rejected:workspace"],
        )
        return response
    workspace_root, apply_source_root = root_result

    max_stdout_result = _output_cap(request.get("max_stdout_chars"))
    max_stderr_result = _output_cap(request.get("max_stderr_chars"))
    timeout_result = _timeout_seconds(execution.get("timeout_seconds"))
    if isinstance(max_stdout_result, str):
        response = _response(
            status="rejected",
            tool=tool,
            limitations=[max_stdout_result],
            trace_summary=["code_execution:rejected:stdout_cap"],
        )
        return response
    if isinstance(max_stderr_result, str):
        response = _response(
            status="rejected",
            tool=tool,
            limitations=[max_stderr_result],
            trace_summary=["code_execution:rejected:stderr_cap"],
        )
        return response
    if isinstance(timeout_result, str):
        response = _response(
            status="rejected",
            tool=tool,
            limitations=[timeout_result],
            trace_summary=["code_execution:rejected:timeout"],
        )
        return response

    command_result = _command_for_execution(
        execution=execution,
        tool=tool,
        apply_source_root=apply_source_root,
    )
    if isinstance(command_result, str):
        response = _response(
            status="rejected",
            tool=tool,
            limitations=[command_result],
            trace_summary=["code_execution:rejected:command"],
        )
        return response
    command, executed_paths = command_result

    run_result = run_argv(
        command,
        cwd=apply_source_root,
        timeout_seconds=timeout_result,
        max_stdout_chars=max_stdout_result,
        max_stderr_chars=max_stderr_result,
        scrub_roots=[workspace_root, apply_source_root],
    )
    status = _status_from_run(run_result.timed_out, run_result.returncode)
    response = _response(
        status=status,
        tool=tool,
        exit_code=run_result.returncode,
        timed_out=run_result.timed_out,
        duration_ms=run_result.duration_ms,
        stdout_excerpt=run_result.stdout,
        stderr_excerpt=run_result.stderr,
        output_truncated=run_result.output_truncated,
        executed_paths=executed_paths,
        trace_summary=[
            f"code_execution:tool={tool}",
            f"code_execution:status={status}",
        ],
    )
    return response


def _managed_apply_source_root(
    request: Mapping[str, object],
) -> tuple[Path, Path] | str:
    workspace_root_text = request.get("workspace_root")
    apply_package_id = request.get("apply_package_id")
    apply_workspace_ref = request.get("apply_workspace_ref")
    if not isinstance(workspace_root_text, str) or not workspace_root_text.strip():
        return "Code execution requires a workspace root."
    if not isinstance(apply_package_id, str) or not apply_package_id.strip():
        return "Code execution requires an apply package id."
    if not SAFE_PACKAGE_ID_RE.fullmatch(apply_package_id):
        return "Code execution apply package id is unsafe."
    if not isinstance(apply_workspace_ref, Mapping):
        return "Code execution requires a managed apply workspace reference."
    if apply_workspace_ref.get("kind") != "managed_apply_workspace":
        return "Code execution requires a managed apply workspace reference."
    if apply_workspace_ref.get("apply_package_id") != apply_package_id:
        return "Code execution apply workspace reference does not match."
    if not isinstance(apply_workspace_ref.get("source_identity"), Mapping):
        return "Code execution requires source identity in the workspace reference."
    if not isinstance(apply_workspace_ref.get("applied_files"), list):
        return "Code execution requires applied files in the workspace reference."

    workspace_root = Path(workspace_root_text).expanduser().resolve(strict=False)
    try:
        apply_source_root = ensure_path_inside(
            workspace_root
            / APPLY_ROOT_NAME
            / apply_package_id
            / APPLIED_SOURCE_DIR_NAME,
            workspace_root,
        )
    except PathSafetyError:
        return "Code execution managed workspace path is unsafe."
    if not apply_source_root.is_dir():
        return "Code execution managed apply workspace is missing."

    return workspace_root, apply_source_root


def _tool_name(execution: Mapping[str, object]) -> str:
    tool_value = execution.get("tool")
    if isinstance(tool_value, str):
        return tool_value
    return ""


def _command_for_execution(
    *,
    execution: Mapping[str, object],
    tool: str,
    apply_source_root: Path,
) -> tuple[list[str], list[str]] | str:
    if "command" in execution:
        return "Code execution does not accept command strings."
    if tool == "python_compileall":
        paths_result = _safe_targets(
            execution.get("paths"),
            apply_source_root=apply_source_root,
            allow_pytest_selector=False,
        )
        if isinstance(paths_result, str):
            return paths_result
        command = [sys.executable, "-m", "compileall", *paths_result]
        return command, paths_result
    if tool == "pytest":
        selectors_result = _safe_targets(
            execution.get("pytest_selectors"),
            apply_source_root=apply_source_root,
            allow_pytest_selector=True,
        )
        if isinstance(selectors_result, str):
            return selectors_result
        command = [sys.executable, "-m", "pytest", *selectors_result, "-q"]
        return command, selectors_result
    return "Code execution tool is unsupported."


def _safe_targets(
    value: object,
    *,
    apply_source_root: Path,
    allow_pytest_selector: bool,
) -> list[str] | str:
    if not isinstance(value, list) or not value:
        return "Code execution requires at least one execution target."
    if len(value) > MAX_EXECUTION_TARGETS:
        return "Code execution target count exceeds the configured cap."

    targets: list[str] = []
    for target_value in value:
        if not isinstance(target_value, str) or not target_value.strip():
            return "Code execution target paths must be non-empty strings."
        target = target_value.strip().replace("\\", "/")
        path_part = target
        selector_suffix = ""
        if allow_pytest_selector and "::" in target:
            path_part, selector_suffix = target.split("::", 1)
            if not selector_suffix:
                return "Code execution pytest selector is incomplete."
        target_error = _target_error(path_part, selector_suffix)
        if target_error:
            return target_error
        try:
            resolved_target = ensure_path_inside(
                apply_source_root / path_part,
                apply_source_root,
            )
        except PathSafetyError:
            return "Code execution target path is unsafe."
        if not resolved_target.exists():
            return "Code execution target path does not exist."
        tree_error = _target_tree_error(
            resolved_target,
            apply_source_root=apply_source_root,
        )
        if tree_error:
            return tree_error
        targets.append(target)
    return targets


def _target_error(path_part: str, selector_suffix: str) -> str:
    if not path_part or path_part.startswith("/"):
        return "Code execution target path is unsafe."
    if re.match(r"^[A-Za-z]:", path_part):
        return "Code execution target path is unsafe."
    for token in SHELL_TOKENS:
        if token in path_part or token in selector_suffix:
            return "Code execution target path contains unsafe shell syntax."

    pure_path = PurePosixPath(path_part)
    parts = pure_path.parts
    if any(part in ("", ".", "..") for part in parts):
        return "Code execution target path is unsafe."
    for part in parts:
        lowered_part = part.casefold()
        if lowered_part in UNSAFE_PATH_PARTS:
            return "Code execution target path is unsafe."
        for fragment in UNSAFE_PATH_FRAGMENTS:
            if fragment in lowered_part:
                return "Code execution target path is unsafe."
    if pure_path.suffix.casefold() in BINARY_EXTENSIONS:
        return "Code execution target path is binary-only."
    return ""


def _target_tree_error(target: Path, *, apply_source_root: Path) -> str:
    if target.is_file():
        return ""
    if not target.is_dir():
        return "Code execution target path is unsupported."

    file_count = 0
    for child in target.rglob("*"):
        if not child.is_file():
            continue
        file_count += 1
        if file_count > MAX_TARGET_FILE_COUNT:
            return "Code execution target file count exceeds the configured cap."
        try:
            ensure_path_inside(child, apply_source_root)
        except PathSafetyError:
            return "Code execution target tree escapes the managed workspace."
    return ""


def _timeout_seconds(value: object) -> int | str:
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    if not isinstance(value, int):
        return "Code execution timeout must be an integer."
    if value < 1 or value > MAX_TIMEOUT_SECONDS:
        return "Code execution timeout exceeds the configured cap."
    return value


def _output_cap(value: object) -> int | str:
    if value is None:
        return DEFAULT_OUTPUT_CHARS
    if not isinstance(value, int):
        return "Code execution output cap must be an integer."
    if value < 1 or value > MAX_OUTPUT_CHARS:
        return "Code execution output cap exceeds the configured cap."
    return value


def _status_from_run(timed_out: bool, returncode: int | None) -> CodeExecutionStatus:
    if timed_out:
        return "timed_out"
    if returncode == 0:
        return "succeeded"
    return "failed"


def _response(
    *,
    status: CodeExecutionStatus,
    tool: str,
    exit_code: int | None = None,
    timed_out: bool = False,
    duration_ms: int = 0,
    stdout_excerpt: str = "",
    stderr_excerpt: str = "",
    output_truncated: bool = False,
    executed_paths: list[str] | None = None,
    limitations: list[str] | None = None,
    trace_summary: list[str] | None = None,
) -> CodeExecutionResponse:
    response: CodeExecutionResponse = {
        "status": status,
        "tool": tool,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
        "stdout_excerpt": stdout_excerpt,
        "stderr_excerpt": stderr_excerpt,
        "output_truncated": output_truncated,
        "executed_paths": executed_paths or [],
        "limitations": limitations or [],
        "trace_summary": trace_summary or [],
    }
    return response


def _mapping_or_empty(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        mapped = dict(value)
        return mapped
    return {}
