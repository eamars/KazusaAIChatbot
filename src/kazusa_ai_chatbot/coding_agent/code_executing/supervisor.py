"""Validator and runner orchestration for bounded code execution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from collections.abc import Mapping
from uuid import uuid4

from kazusa_ai_chatbot.coding_agent.code_executing.models import (
    CodeExecutionRequest,
    CodeExecutionResponse,
    CodeExecutionStatus,
)
from kazusa_ai_chatbot.coding_agent.code_executing.runner import run_argv
from kazusa_ai_chatbot.coding_agent.safety import (
    copy_managed_source_tree,
    managed_source_tree_digest,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)

APPLY_ROOT_NAME = "patch_apply"
APPLIED_SOURCE_DIR_NAME = "source"
CANDIDATE_EXECUTION_ROOT_NAME = "candidate_exec"
EXECUTION_EPHEMERAL_PARTS = {".pytest_cache", "__pycache__"}
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
    root_result = _managed_execution_source_root(request)
    if isinstance(root_result, str):
        response = _response(
            status="rejected",
            tool=tool,
            limitations=[root_result],
            trace_summary=["code_execution:rejected:workspace"],
        )
        return response
    workspace_root, execution_source_root, candidate_execution_root = root_result

    if candidate_execution_root is not None:
        try:
            cached = _load_candidate_execution_result(
                candidate_execution_root,
                request=request,
            )
        except ValueError as exc:
            return _response(
                status="rejected",
                tool=tool,
                limitations=[str(exc)],
                trace_summary=["code_execution:rejected:cached_identity"],
            )
        if cached is not None:
            return cached

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
        apply_source_root=execution_source_root,
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
        cwd=execution_source_root,
        timeout_seconds=timeout_result,
        max_stdout_chars=max_stdout_result,
        max_stderr_chars=max_stderr_result,
        scrub_roots=[workspace_root, execution_source_root],
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
    if candidate_execution_root is not None:
        _persist_candidate_execution_result(
            candidate_execution_root,
            request=request,
            response=response,
        )
    return response


def _managed_execution_source_root(
    request: Mapping[str, object],
) -> tuple[Path, Path, Path | None] | str:
    """Resolve one exact managed apply or candidate execution workspace."""

    candidate_identity = request.get("candidate_execution_identity")
    has_apply_identity = any(
        request.get(field_name) is not None
        for field_name in ("apply_package_id", "apply_workspace_ref")
    )
    if candidate_identity is not None:
        if has_apply_identity:
            return "Code execution request mixes apply and candidate identities."
        return _managed_candidate_execution_source_root(request)
    apply_result = _managed_apply_source_root(request)
    if isinstance(apply_result, str):
        return apply_result
    workspace_root, apply_source_root = apply_result
    return workspace_root, apply_source_root, None


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
    unresolved_apply_root = workspace_root / APPLY_ROOT_NAME
    unresolved_package_root = unresolved_apply_root / apply_package_id
    unresolved_source_root = unresolved_package_root / APPLIED_SOURCE_DIR_NAME
    if any(
        path.is_symlink()
        for path in (
            unresolved_apply_root,
            unresolved_package_root,
            unresolved_source_root,
        )
    ):
        return "Code execution managed workspace path is unsafe."
    try:
        apply_source_root = ensure_path_inside(
            unresolved_source_root,
            workspace_root,
        )
    except PathSafetyError:
        return "Code execution managed workspace path is unsafe."
    if not apply_source_root.is_dir():
        return "Code execution managed apply workspace is missing."

    return workspace_root, apply_source_root


def _managed_candidate_execution_source_root(
    request: Mapping[str, object],
) -> tuple[Path, Path, Path] | str:
    """Materialize and validate one immutable current-candidate workspace."""

    workspace_root_text = request.get("workspace_root")
    identity = request.get("candidate_execution_identity")
    if not isinstance(workspace_root_text, str) or not workspace_root_text.strip():
        return "Code execution requires a workspace root."
    identity_error = _candidate_execution_identity_error(identity, request)
    if identity_error:
        return identity_error
    if not isinstance(identity, Mapping):
        raise ValueError("validated candidate execution identity is invalid")
    run_id = str(identity["run_id"])
    workspace_root = Path(workspace_root_text).expanduser().resolve(strict=False)
    unresolved_run_root = workspace_root / "coding_runs" / run_id
    unresolved_candidate_root = unresolved_run_root / "candidate"
    unresolved_candidate_source = unresolved_candidate_root / "source"
    if any(
        path.is_symlink()
        for path in (
            unresolved_run_root,
            unresolved_candidate_root,
            unresolved_candidate_source,
        )
    ):
        return "Code execution candidate workspace path is unsafe."
    try:
        run_root = ensure_path_inside(
            unresolved_run_root,
            workspace_root,
        )
        candidate_source_root = ensure_path_inside(
            unresolved_candidate_source,
            run_root,
        )
    except PathSafetyError:
        return "Code execution candidate workspace path is unsafe."
    if not candidate_source_root.is_dir():
        return "Code execution current candidate source is missing."
    candidate_state = _read_json_object(run_root / "candidate" / "state.json")
    if (
        candidate_state is None
        or candidate_state.get("revision") != identity["candidate_revision"]
    ):
        return "Code execution candidate revision is stale."
    expected_manifest_digest = str(identity["candidate_manifest_digest"])
    if _tree_digest(candidate_source_root) != expected_manifest_digest:
        return "Code execution candidate identity is stale."
    identity_digest = _canonical_digest(identity)
    unresolved_execution_parent = (
        workspace_root / CANDIDATE_EXECUTION_ROOT_NAME
    )
    unresolved_execution_root = (
        unresolved_execution_parent / identity_digest
    )
    if (
        unresolved_execution_parent.is_symlink()
        or unresolved_execution_root.is_symlink()
    ):
        return "Code execution candidate materialization path is unsafe."
    try:
        execution_root = ensure_path_inside(
            unresolved_execution_root,
            workspace_root,
        )
    except PathSafetyError:
        return "Code execution candidate materialization path is unsafe."
    execution_source_root = execution_root / "source"
    if execution_root.exists():
        if (
            execution_source_root.is_symlink()
            or (execution_root / "identity.json").is_symlink()
        ):
            return "Code execution candidate workspace identity does not match."
        persisted_identity = _read_json_object(execution_root / "identity.json")
        if persisted_identity != dict(identity):
            return "Code execution candidate workspace identity does not match."
        if (
            not execution_source_root.is_dir()
            or _tree_digest(execution_source_root) != expected_manifest_digest
        ):
            return "Code execution candidate workspace content does not match."
        return workspace_root, execution_source_root, execution_root

    execution_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = execution_root.with_name(
        f"{execution_root.name}.tmp-{uuid4().hex}",
    )
    try:
        copy_managed_source_tree(
            candidate_source_root,
            temporary_root / "source",
            extra_excluded_names=EXECUTION_EPHEMERAL_PARTS,
        )
        if _tree_digest(temporary_root / "source") != expected_manifest_digest:
            return "Code execution candidate changed during materialization."
        _write_json_atomic(temporary_root / "identity.json", dict(identity))
        try:
            os.replace(temporary_root, execution_root)
        except FileExistsError:
            shutil.rmtree(temporary_root, ignore_errors=True)
    except (OSError, ValueError):
        return "Code execution candidate workspace materialization failed."
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root, ignore_errors=True)
    persisted_identity = _read_json_object(execution_root / "identity.json")
    if (
        persisted_identity != dict(identity)
        or not execution_source_root.is_dir()
        or _tree_digest(execution_source_root) != expected_manifest_digest
    ):
        return "Code execution candidate workspace materialization failed."
    return workspace_root, execution_source_root, execution_root


def _candidate_execution_identity_error(
    value: object,
    request: Mapping[str, object],
) -> str:
    """Validate candidate, snapshot, and execution-spec identity binding."""

    if not isinstance(value, Mapping):
        return "Code execution requires a candidate execution identity."
    required_keys = {
        "run_id",
        "candidate_id",
        "candidate_revision",
        "candidate_manifest_digest",
        "base_snapshot_id",
        "execution_policy_digest",
        "execution_spec_digest",
    }
    if set(value) != required_keys:
        return "Code execution candidate identity keys are invalid."
    for key in ("run_id", "candidate_id"):
        field_value = value.get(key)
        if not isinstance(field_value, str) or not SAFE_PACKAGE_ID_RE.fullmatch(
            field_value,
        ):
            return "Code execution candidate identity is unsafe."
    run_id = str(value["run_id"])
    expected_candidate_id = hashlib.sha256(
        f"{run_id}:candidate".encode("utf-8"),
    ).hexdigest()
    if value.get("candidate_id") != expected_candidate_id:
        return "Code execution candidate id does not match the run."
    revision = value.get("candidate_revision")
    if not isinstance(revision, int) or revision < 0:
        return "Code execution candidate revision is invalid."
    for key in (
        "candidate_manifest_digest",
        "base_snapshot_id",
        "execution_policy_digest",
        "execution_spec_digest",
    ):
        digest = value.get(key)
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            return "Code execution candidate digest is invalid."
    execution = request.get("execution")
    if not isinstance(execution, Mapping):
        return "Code execution candidate spec is invalid."
    if _canonical_digest(execution) != value["execution_spec_digest"]:
        return "Code execution candidate spec identity does not match."
    return ""


def _load_candidate_execution_result(
    execution_root: Path,
    *,
    request: Mapping[str, object],
) -> CodeExecutionResponse | None:
    """Reuse one exact durable result without executing the command again."""

    result_path = execution_root / "result.json"
    if not result_path.is_file():
        return None
    payload = _read_json_object(result_path)
    identity = request.get("candidate_execution_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("candidate execution result identity is missing")
    if (
        payload is None
        or payload.get("identity_digest") != _canonical_digest(identity)
        or payload.get("execution_spec_digest")
        != identity.get("execution_spec_digest")
    ):
        raise ValueError("candidate execution result identity mismatch")
    response = payload.get("response")
    if not isinstance(response, dict):
        raise ValueError("candidate execution cached response is invalid")
    cached = dict(response)
    trace_summary = cached.get("trace_summary")
    if not isinstance(trace_summary, list):
        raise ValueError("candidate execution cached trace is invalid")
    cached["trace_summary"] = [*trace_summary, "code_execution:cache_hit"]
    return cached


def _persist_candidate_execution_result(
    execution_root: Path,
    *,
    request: Mapping[str, object],
    response: CodeExecutionResponse,
) -> None:
    """Persist a bounded result under its complete candidate identity."""

    identity = request.get("candidate_execution_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("candidate execution identity is missing")
    _write_json_atomic(
        execution_root / "result.json",
        {
            "schema_version": "candidate_execution_result.v1",
            "identity_digest": _canonical_digest(identity),
            "execution_spec_digest": identity["execution_spec_digest"],
            "response": response,
        },
    )


def _tree_digest(root: Path) -> str:
    """Return a deterministic digest for one managed candidate tree."""

    tree_digest = managed_source_tree_digest(
        root,
        extra_excluded_names=EXECUTION_EPHEMERAL_PARTS,
    )
    return tree_digest


def _canonical_digest(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_json_object(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp-{uuid4().hex}")
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)


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
