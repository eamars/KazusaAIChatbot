"""Structural evaluation for resolved file/module writing plans."""

from __future__ import annotations

from pathlib import PurePosixPath

from kazusa_ai_chatbot.coding_agent.code_fetching.models import CodeSourceScope
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    WritingFileModuleContract,
    WritingFilePlanEvaluation,
    WritingMode,
)
from kazusa_ai_chatbot.coding_agent.code_writing.patch_validation import (
    _safe_repo_relative_path,
)

MAX_FILE_PLAN_ERRORS = 12


def evaluate_file_plan(
    *,
    file_contracts: list[WritingFileModuleContract],
    source_scope: CodeSourceScope | None,
    mode: WritingMode,
) -> WritingFilePlanEvaluation:
    """Validate resolved file/module contracts before File PM dispatch."""

    errors: list[str] = []
    if not file_contracts:
        errors.append("File agent did not return any file contracts.")

    owner_paths: list[tuple[str, str]] = []
    for file_contract in file_contracts:
        errors.extend(_single_file_contract_errors(
            file_contract=file_contract,
            source_scope=source_scope,
            mode=mode,
        ))
        for owned_path in _safe_paths(file_contract.get("owned_paths", [])):
            owner_paths.append((file_contract["file_contract_id"], owned_path))
    errors.extend(_owned_path_overlap_errors(owner_paths))

    limited_errors = errors[:MAX_FILE_PLAN_ERRORS]
    status = "accepted"
    if limited_errors:
        status = "repair_required"
    evaluation: WritingFilePlanEvaluation = {
        "status": status,
        "errors": limited_errors,
        "repair_feedback": _repair_feedback(limited_errors),
    }
    return evaluation


def _single_file_contract_errors(
    *,
    file_contract: WritingFileModuleContract,
    source_scope: CodeSourceScope | None,
    mode: WritingMode,
) -> list[str]:
    errors: list[str] = []
    contract_id = file_contract.get("file_contract_id", "")
    if not contract_id:
        errors.append("File contract has no file_contract_id.")
        contract_id = "unknown"

    owned_paths = _safe_paths(file_contract.get("owned_paths", []))
    if not owned_paths:
        errors.append(f"File contract {contract_id!r} has no owned path.")
    for owned_path in owned_paths:
        if not _path_inside_source_scope(owned_path, source_scope):
            errors.append(
                f"File contract {contract_id!r} owns path {owned_path!r} "
                "outside source scope.",
            )

    if mode == "create_new_project" and any(path == "." for path in owned_paths):
        errors.append(f"File contract {contract_id!r} owns an invalid path.")

    for field_name in ("purpose", "change_goal"):
        if not file_contract.get(field_name):
            errors.append(f"File contract {contract_id!r} has no {field_name}.")
    if not file_contract.get("validation_expectations"):
        errors.append(
            f"File contract {contract_id!r} has no validation expectations.",
        )

    errors.extend(_unsafe_path_errors(
        contract_id=contract_id,
        field_name="read_only_paths",
        values=file_contract.get("read_only_paths", []),
    ))
    errors.extend(_unsafe_path_errors(
        contract_id=contract_id,
        field_name="forbidden_paths",
        values=file_contract.get("forbidden_paths", []),
    ))
    return errors


def _owned_path_overlap_errors(
    owner_paths: list[tuple[str, str]],
) -> list[str]:
    errors: list[str] = []
    seen_paths: list[tuple[str, str]] = []
    for contract_id, owned_path in owner_paths:
        for previous_id, previous_path in seen_paths:
            if _paths_overlap(owned_path, previous_path):
                errors.append(
                    f"File contracts {previous_id!r} and {contract_id!r} "
                    f"own overlapping paths {previous_path!r} and "
                    f"{owned_path!r}.",
                )
        seen_paths.append((contract_id, owned_path))
    return errors


def _unsafe_path_errors(
    *,
    contract_id: str,
    field_name: str,
    values: object,
) -> list[str]:
    errors: list[str] = []
    for value in _string_values(values):
        safe_path = _safe_repo_relative_path(value)
        if safe_path is None:
            errors.append(
                f"File contract {contract_id!r} {field_name} contains unsafe "
                f"path {value!r}.",
            )
    return errors


def _path_inside_source_scope(
    path_text: str,
    source_scope: CodeSourceScope | None,
) -> bool:
    if source_scope is None:
        return True

    scoped_path = source_scope.get("repo_relative_path")
    if scoped_path is None:
        return True

    safe_scope = _safe_repo_relative_path(scoped_path)
    if safe_scope is None:
        return False

    path = PurePosixPath(path_text)
    scope_path = PurePosixPath(safe_scope)
    if source_scope["kind"] == "file":
        return path == scope_path
    return path == scope_path or scope_path in path.parents


def _safe_paths(values: object) -> list[str]:
    paths: list[str] = []
    for value in _string_values(values):
        safe_path = _safe_repo_relative_path(value)
        if safe_path is None or safe_path in paths:
            continue
        paths.append(safe_path)
    return paths


def _string_values(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _paths_overlap(first_path: str, second_path: str) -> bool:
    first = PurePosixPath(first_path)
    second = PurePosixPath(second_path)
    return first == second or first in second.parents or second in first.parents


def _repair_feedback(errors: list[str]) -> list[str]:
    if not errors:
        return []
    return [
        "Return corrected file demands for the same writing goal.",
        *errors,
    ]
