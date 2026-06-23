"""Shared file planning for coding-agent workflows."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Any

from kazusa_ai_chatbot.coding_agent.code_fetching.github import (
    is_safe_repo_relative_path,
)
from kazusa_ai_chatbot.coding_agent.code_fetching.models import (
    CodeRepositoryRef,
    CodeSourceScope,
)
from kazusa_ai_chatbot.coding_agent.code_reading.planner import (
    is_binary_like_path,
    is_secret_like_path,
)
from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    SourceOwnerCandidate,
    WritingFileDemand,
    WritingFileModuleContract,
    WritingFileResolution,
    WritingMode,
)

MAX_FILE_AGENT_ERRORS = 12
MAX_FILE_NAME_CHARS = 80
DEFAULT_SOURCE_DIR = "src"
DEFAULT_TEST_DIR = "tests"
DEFAULT_DOCS_DIR = "docs"
FILE_NAME_CLEANUP_RE = re.compile(r"[^A-Za-z0-9_.-]+")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
PY_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
COMMON_MATCH_TOKENS = {
    "and",
    "behavior",
    "change",
    "component",
    "contract",
    "current",
    "existing",
    "file",
    "for",
    "from",
    "implementation",
    "input",
    "interface",
    "module",
    "new",
    "output",
    "purpose",
    "required",
    "source",
    "support",
    "that",
    "the",
    "this",
    "to",
    "validation",
    "worker",
}


def resolve_writing_file_demands(
    *,
    mode: WritingMode,
    repository: CodeRepositoryRef | dict[str, object] | None,
    source_scope: CodeSourceScope | None,
    owner_candidates: list[SourceOwnerCandidate],
    file_demands: list[WritingFileDemand],
) -> WritingFileResolution:
    """Resolve PM semantic file demands into file/module contracts.

    Args:
        mode: Writing mode selected for the current request.
        repository: Existing repository contract when available.
        source_scope: Source scope returned by fetching for existing work.
        owner_candidates: Evidence-derived source file hints.
        file_demands: PM-authored semantic file needs.

    Returns:
        Accepted resolved file plan or compact repair feedback.
    """

    repo_root = _repo_root(repository)
    errors: list[str] = []
    file_contracts: list[WritingFileModuleContract] = []

    if not file_demands:
        errors.append("PM decision did not provide file demands.")

    for index, demand in enumerate(file_demands, start=1):
        file_contract, demand_errors = _resolve_one_demand(
            demand=demand,
            index=index,
            mode=mode,
            repo_root=repo_root,
            source_scope=source_scope,
            owner_candidates=owner_candidates,
        )
        errors.extend(demand_errors)
        if file_contract is not None:
            file_contracts.append(file_contract)

    _attach_cross_file_imports(file_contracts)
    errors.extend(_file_contract_overlap_errors(file_contracts))
    limited_errors = errors[:MAX_FILE_AGENT_ERRORS]
    if limited_errors:
        resolution: WritingFileResolution = {
            "status": "repair_required",
            "file_contracts": [],
            "owned_path_map": {},
            "read_only_path_map": {},
            "errors": limited_errors,
            "repair_feedback": _repair_feedback(limited_errors),
        }
        return resolution

    resolution = {
        "status": "accepted",
        "file_contracts": file_contracts,
        "owned_path_map": _owned_path_map(file_contracts),
        "read_only_path_map": _read_only_path_map(file_contracts),
        "errors": [],
        "repair_feedback": [],
    }
    return resolution


def _resolve_one_demand(
    *,
    demand: WritingFileDemand,
    index: int,
    mode: WritingMode,
    repo_root: Path | None,
    source_scope: CodeSourceScope | None,
    owner_candidates: list[SourceOwnerCandidate],
) -> tuple[WritingFileModuleContract | None, list[str]]:
    demand_id = _bounded_text(demand.get("demand_id")) or f"file-{index}"
    errors: list[str] = []
    preferred_path = demand.get("preferred_path")
    if (
        isinstance(preferred_path, str)
        and preferred_path.strip()
        and _safe_path(preferred_path) is None
    ):
        errors.append(
            f"File demand {demand_id!r} preferred_path is not a safe "
            "repo-relative path.",
        )
    owned_path = _resolve_owned_path(
        demand=demand,
        demand_id=demand_id,
        mode=mode,
        owner_candidates=owner_candidates,
    )
    if owned_path is None:
        errors.append(
            f"File demand {demand_id!r} does not provide enough placement "
            "information for a concrete file.",
        )
        return None, errors
    if errors:
        return None, errors

    path_errors = _path_mechanics_errors(
        demand_id=demand_id,
        demand=demand,
        owned_path=owned_path,
        repo_root=repo_root,
        source_scope=source_scope,
    )
    errors.extend(path_errors)

    read_only_paths = _read_only_paths(
        demand,
        source_scope,
        owned_path=owned_path,
    )
    for read_only_path in read_only_paths:
        if not _path_inside_source_scope(read_only_path, source_scope):
            errors.append(
                f"File demand {demand_id!r} read-only path "
                f"{read_only_path!r} is outside source scope.",
            )

    if errors:
        return None, errors

    file_contract = _file_contract_from_demand(
        demand=demand,
        demand_id=demand_id,
        owned_path=owned_path,
        read_only_paths=read_only_paths,
    )
    result = file_contract, []
    return result


def _resolve_owned_path(
    *,
    demand: WritingFileDemand,
    demand_id: str,
    mode: WritingMode,
    owner_candidates: list[SourceOwnerCandidate],
) -> str | None:
    preferred_path = _safe_path(demand.get("preferred_path"))
    if preferred_path is not None:
        return preferred_path

    file_kind = _bounded_text(demand.get("file_kind"))
    if (
        mode == "edit_existing_repository"
        and file_kind in {"existing", "docs", "config"}
    ):
        return None
    if file_kind == "existing":
        return None

    preferred_name = _safe_file_name(demand.get("preferred_name"))
    if preferred_name is None:
        preferred_name = _default_file_name(demand=demand, demand_id=demand_id)
    if preferred_name is None:
        return None

    directory = _placement_directory(
        demand=demand,
        mode=mode,
        owner_candidates=owner_candidates,
    )
    if directory is None:
        return None

    path = PurePosixPath(directory) / preferred_name
    safe_path = _safe_path(path.as_posix())
    return safe_path


def _path_mechanics_errors(
    *,
    demand_id: str,
    demand: WritingFileDemand,
    owned_path: str,
    repo_root: Path | None,
    source_scope: CodeSourceScope | None,
) -> list[str]:
    errors: list[str] = []
    if not _path_inside_source_scope(owned_path, source_scope):
        errors.append(
            f"File demand {demand_id!r} resolved path {owned_path!r} is "
            "outside source scope.",
        )

    if repo_root is None:
        return errors

    file_path = repo_root / owned_path
    path_exists = file_path.exists()
    file_kind = _bounded_text(demand.get("file_kind")) or "support"
    if file_kind == "existing" and not path_exists:
        errors.append(
            f"File demand {demand_id!r} requires existing path "
            f"{owned_path!r}, but it does not exist.",
        )
    if file_kind == "new" and path_exists:
        errors.append(
            f"File demand {demand_id!r} requests new path {owned_path!r}, "
            "but it already exists.",
        )
    if path_exists and file_path.is_dir():
        errors.append(
            f"File demand {demand_id!r} resolved path {owned_path!r} is a "
            "directory, not a file.",
        )
    return errors


def _file_contract_from_demand(
    *,
    demand: WritingFileDemand,
    demand_id: str,
    owned_path: str,
    read_only_paths: list[str],
) -> WritingFileModuleContract:
    role = _bounded_text(demand.get("role")) or "file owner"
    purpose = _bounded_text(demand.get("purpose")) or "Implement assigned file."
    work_instructions = _string_list(demand.get("work_instructions"))
    if not work_instructions:
        work_instructions = [purpose]
    required_slots = _string_list(demand.get("required_slots"))
    if not required_slots:
        required_slots = [purpose]
    file_kind = _bounded_text(demand.get("file_kind"))
    if file_kind not in ("existing", "new", "test", "docs", "config", "support"):
        file_kind = "support"

    contract: WritingFileModuleContract = {
        "file_contract_id": demand_id,
        "role": role,
        "purpose": purpose,
        "file_kind": file_kind,
        "owned_path": owned_path,
        "owned_paths": [owned_path],
        "read_only_paths": read_only_paths,
        "interface_contract": _object_dict(demand.get("interface_contract")),
        "integration_contract": _object_dict(demand.get("integration_contract")),
        "change_goal": _bounded_text(demand.get("change_goal")) or purpose,
        "cross_file_imports": [],
        "work_instructions": work_instructions,
        "required_slots": required_slots,
        "validation_expectations": _string_list(
            demand.get("validation_expectations"),
        ),
        "forbidden_paths": _safe_path_list(demand.get("forbidden_paths")),
    }
    return contract


def _placement_directory(
    *,
    demand: WritingFileDemand,
    mode: WritingMode,
    owner_candidates: list[SourceOwnerCandidate],
) -> str | None:
    placement_hint = _safe_path(demand.get("placement_hint"))
    if placement_hint is not None:
        path = PurePosixPath(placement_hint)
        if path.suffix:
            directory = path.parent.as_posix()
        else:
            directory = path.as_posix()
        if directory == ".":
            directory = ""
        return directory

    related_paths = _safe_path_list(demand.get("related_paths"))
    if related_paths:
        related_directory = PurePosixPath(related_paths[0]).parent.as_posix()
        return "" if related_directory == "." else related_directory

    file_kind = _bounded_text(demand.get("file_kind"))
    if file_kind == "test":
        return DEFAULT_TEST_DIR
    if file_kind == "docs":
        return DEFAULT_DOCS_DIR
    if mode == "create_new_project":
        return DEFAULT_SOURCE_DIR
    runtime_directory = _common_owner_directory(
        owner_candidates,
        preferred_role="runtime",
    )
    if runtime_directory is not None:
        return runtime_directory
    return None


def _common_owner_directory(
    owner_candidates: list[SourceOwnerCandidate],
    *,
    preferred_role: str,
) -> str | None:
    directory_counts: dict[str, int] = {}
    for candidate in owner_candidates:
        if candidate["role"] != preferred_role:
            continue
        safe_path = _safe_path(candidate["path"])
        if safe_path is None:
            continue
        directory = PurePosixPath(safe_path).parent.as_posix()
        if directory == ".":
            directory = ""
        directory_counts[directory] = directory_counts.get(directory, 0) + 1
    if not directory_counts:
        return None
    ranked_directories = sorted(
        directory_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    directory = ranked_directories[0][0]
    return directory


def _read_only_paths(
    demand: WritingFileDemand,
    source_scope: CodeSourceScope | None,
    *,
    owned_path: str,
) -> list[str]:
    paths: list[str] = []
    for raw_path in _string_list(demand.get("read_only_paths")):
        safe_path = _safe_path(raw_path)
        if safe_path is None or safe_path == owned_path or safe_path in paths:
            continue
        paths.append(safe_path)
    for raw_path in _string_list(demand.get("related_paths")):
        safe_path = _safe_path(raw_path)
        if safe_path is None or safe_path == owned_path or safe_path in paths:
            continue
        if not _path_inside_source_scope(safe_path, source_scope):
            continue
        paths.append(safe_path)
    return paths


def _file_contract_overlap_errors(
    file_contracts: list[WritingFileModuleContract],
) -> list[str]:
    errors: list[str] = []
    seen_paths: list[tuple[str, str]] = []
    for file_contract in file_contracts:
        file_contract_id = file_contract["file_contract_id"]
        for owned_path in file_contract["owned_paths"]:
            for previous_id, previous_path in seen_paths:
                if _paths_overlap(owned_path, previous_path):
                    errors.append(
                        f"File demands {previous_id!r} and {file_contract_id!r} "
                        f"resolve overlapping owned paths {previous_path!r} "
                        f"and {owned_path!r}.",
                    )
            seen_paths.append((file_contract_id, owned_path))
    return errors


def _owned_path_map(
    file_contracts: list[WritingFileModuleContract],
) -> dict[str, str]:
    owned_path_map: dict[str, str] = {}
    for file_contract in file_contracts:
        owner = file_contract["file_contract_id"]
        for owned_path in file_contract["owned_paths"]:
            owned_path_map[owned_path] = owner
    return owned_path_map


def _read_only_path_map(
    file_contracts: list[WritingFileModuleContract],
) -> dict[str, list[str]]:
    read_only_path_map: dict[str, list[str]] = {}
    for file_contract in file_contracts:
        read_only_path_map[file_contract["file_contract_id"]] = (
            file_contract["read_only_paths"]
        )
    return read_only_path_map


def _path_inside_source_scope(
    path_text: str,
    source_scope: CodeSourceScope | None,
) -> bool:
    if source_scope is None:
        return True

    scoped_path = source_scope.get("repo_relative_path")
    if scoped_path is None:
        return True

    safe_scope = _safe_path(scoped_path)
    if safe_scope is None:
        return False

    path = PurePosixPath(path_text)
    scope_path = PurePosixPath(safe_scope)
    if source_scope["kind"] == "file":
        return path == scope_path

    inside_scope = path == scope_path or scope_path in path.parents
    return inside_scope


def _repo_root(
    repository: CodeRepositoryRef | dict[str, object] | None,
) -> Path | None:
    if repository is None:
        return None
    local_root = repository.get("local_root")
    if not isinstance(local_root, str) or not local_root.strip():
        return None
    root = Path(local_root).expanduser().resolve(strict=False)
    return root


def _safe_path(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None
    if not is_safe_repo_relative_path(normalized):
        return None
    path = PurePosixPath(normalized)
    safe_path = path.as_posix().rstrip("/")
    if not safe_path:
        return None
    if is_secret_like_path(safe_path) or is_binary_like_path(safe_path):
        return None
    return safe_path


def _safe_file_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.replace("\\", "/").strip().split("/")[-1]
    if not name or name in (".", ".."):
        return None
    safe_name = FILE_NAME_CLEANUP_RE.sub("_", name)[:MAX_FILE_NAME_CHARS]
    if "." not in safe_name:
        safe_name = f"{safe_name}.py"
    if _safe_path(safe_name) is None:
        return None
    return safe_name


def _default_file_name(
    *,
    demand: WritingFileDemand,
    demand_id: str,
) -> str | None:
    file_kind = _bounded_text(demand.get("file_kind"))
    for source in _file_name_sources(demand, demand_id):
        generated_name = _file_name_from_text(source, file_kind=file_kind)
        if generated_name is not None:
            return generated_name
        safe_name = _safe_file_name(source)
        if safe_name is not None:
            return safe_name
    return None


def _file_name_sources(
    demand: WritingFileDemand,
    demand_id: str,
) -> list[object]:
    interface_contract = _object_dict(demand.get("interface_contract"))
    integration_contract = _object_dict(demand.get("integration_contract"))
    sources: list[object] = [
        demand.get("preferred_name"),
        interface_contract.get("component"),
        interface_contract.get("exports"),
        interface_contract.get("outputs"),
        integration_contract.get("provides_to_pm"),
        demand.get("purpose"),
        demand_id,
        demand.get("role"),
    ]
    return sources


def _file_name_from_text(value: object, *, file_kind: str) -> str | None:
    text = _first_text(value)
    if not text:
        return None

    words = _word_parts(text)
    if not words:
        return None
    stem = "_".join(words[:6])
    if not stem:
        return None

    if file_kind == "test" and not stem.startswith("test_"):
        stem = f"test_{stem}"
    suffix = ".md" if file_kind == "docs" else ".py"
    safe_name = _safe_file_name(f"{stem}{suffix}")
    return safe_name


def _first_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            text = _first_text(item)
            if text:
                return text
    return ""


def _word_parts(text: str) -> list[str]:
    parts: list[str] = []
    for token in TOKEN_RE.findall(text.replace("-", "_")):
        for piece in CAMEL_BOUNDARY_RE.sub("_", token).split("_"):
            clean_piece = piece.strip().casefold()
            if not clean_piece or clean_piece in COMMON_MATCH_TOKENS:
                continue
            if clean_piece in parts:
                continue
            parts.append(clean_piece)
    return parts


def _attach_cross_file_imports(
    file_contracts: list[WritingFileModuleContract],
) -> None:
    provider_rows: list[tuple[WritingFileModuleContract, str, list[str]]] = []
    for provider in file_contracts:
        module_name = _python_module_name(provider.get("owned_path", ""))
        provided_symbols = _provided_python_symbols(provider)
        if module_name is None or not provided_symbols:
            continue
        provider_rows.append((provider, module_name, provided_symbols))

    for consumer in file_contracts:
        imports = _string_list(consumer.get("cross_file_imports"))
        for provider, module_name, provided_symbols in provider_rows:
            if provider is consumer:
                continue
            if not _contract_consumes_provider(consumer, provider):
                continue
            import_line = (
                f"from {module_name} import "
                + ", ".join(provided_symbols[:8])
            )
            if import_line not in imports:
                imports.append(import_line)
        consumer["cross_file_imports"] = imports


def _provided_python_symbols(
    provider: WritingFileModuleContract,
) -> list[str]:
    interface_contract = _object_dict(provider.get("interface_contract"))
    values = [
        interface_contract.get("component"),
        interface_contract.get("exports"),
        interface_contract.get("outputs"),
        interface_contract.get("invariants"),
    ]
    symbols: list[str] = []
    for value in values:
        for symbol in _python_identifiers_from_object(value):
            if symbol in symbols:
                continue
            symbols.append(symbol)
    return symbols


def _python_identifiers_from_object(value: object) -> list[str]:
    identifiers: list[str] = []
    if isinstance(value, str):
        for token in TOKEN_RE.findall(value):
            if not PY_IDENTIFIER_RE.fullmatch(token):
                continue
            if not _looks_like_symbol_token(token):
                continue
            if token.casefold() in COMMON_MATCH_TOKENS:
                continue
            if token in identifiers:
                continue
            identifiers.append(token)
        return identifiers
    if isinstance(value, list):
        for item in value:
            for identifier in _python_identifiers_from_object(item):
                if identifier in identifiers:
                    continue
                identifiers.append(identifier)
        return identifiers
    if isinstance(value, dict):
        for item in value.values():
            for identifier in _python_identifiers_from_object(item):
                if identifier in identifiers:
                    continue
                identifiers.append(identifier)
    return identifiers


def _looks_like_symbol_token(token: str) -> bool:
    if "_" in token:
        return True
    if token.isupper() and len(token) > 1:
        return True
    if token[:1].isupper() and any(char.islower() for char in token[1:]):
        return True
    return False


def _python_module_name(owned_path: str) -> str | None:
    safe_path = _safe_path(owned_path)
    if safe_path is None:
        return None
    path = PurePosixPath(safe_path)
    if path.suffix != ".py":
        return None
    parts = list(path.with_suffix("").parts)
    if parts and parts[0] == DEFAULT_SOURCE_DIR:
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    if any(PY_IDENTIFIER_RE.fullmatch(part) is None for part in parts):
        return None
    module_name = ".".join(parts)
    return module_name


def _contract_consumes_provider(
    consumer: WritingFileModuleContract,
    provider: WritingFileModuleContract,
) -> bool:
    integration_contract = _object_dict(consumer.get("integration_contract"))
    consumer_refs = _normalized_refs([
        integration_contract.get("consumes_from"),
        integration_contract.get("calls"),
    ])
    if not consumer_refs:
        consumer_refs = _normalized_refs(consumer.get("required_slots", []))
    provider_refs = _normalized_refs([
        provider.get("file_contract_id", ""),
        provider.get("role", ""),
        _object_dict(provider.get("interface_contract")).get("component"),
        _object_dict(provider.get("interface_contract")).get("exports"),
    ])
    if not consumer_refs or not provider_refs:
        return False
    has_shared_ref = bool(consumer_refs & provider_refs)
    return has_shared_ref


def _normalized_refs(values: object) -> set[str]:
    refs: set[str] = set()
    if isinstance(values, str):
        normalized = _normalize_ref(values)
        if normalized:
            refs.add(normalized)
        return refs
    if isinstance(values, list):
        for item in values:
            refs.update(_normalized_refs(item))
        return refs
    if isinstance(values, dict):
        for item in values.values():
            refs.update(_normalized_refs(item))
    return refs


def _normalize_ref(value: str) -> str:
    parts = _word_parts(value)
    if not parts:
        return ""
    normalized = "_".join(parts)
    return normalized


def _safe_path_list(value: object) -> list[str]:
    paths: list[str] = []
    for raw_path in _string_list(value):
        safe_path = _safe_path(raw_path)
        if safe_path is None or safe_path in paths:
            continue
        paths.append(safe_path)
    return paths


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    strings = [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]
    return strings


def _bounded_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text


def _object_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        result = dict(value)
        return result
    return {}


def _paths_overlap(first_path: str, second_path: str) -> bool:
    first = PurePosixPath(first_path)
    second = PurePosixPath(second_path)
    paths_overlap = (
        first == second
        or first in second.parents
        or second in first.parents
    )
    return paths_overlap


def _repair_feedback(errors: list[str]) -> list[str]:
    feedback = [
        "Return corrected semantic file demands for the same writing goal.",
        *errors,
    ]
    return feedback


__all__ = ["resolve_writing_file_demands"]
