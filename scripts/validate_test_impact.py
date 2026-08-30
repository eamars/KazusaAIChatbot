"""Validate and run exact tests for changed semantic source owners."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

DEFAULT_MANIFEST_PATH = Path(
    "tests/ownership/source_test_impact_manifest.json"
)
STRICT_SOURCE_PREFIXES = ("src/", "scripts/")


class ImpactValidationError(RuntimeError):
    """Identify a source-to-test ownership or execution failure."""


def _repository_root() -> Path:
    """Return the repository root containing this maintenance script."""

    root = Path(__file__).resolve().parents[1]
    return root


def normalize_repository_path(value: str) -> str:
    """Normalize a repository-relative path to the manifest slash format."""

    normalized_value = value.replace("\\", "/")
    normalized_path = PurePosixPath(normalized_value)
    return normalized_path.as_posix()


def _path_is_under(path: str, root: str) -> bool:
    """Return whether a normalized path is equal to or below a root."""

    return path == root or path.startswith(f"{root.rstrip('/')}/")


def _is_strict_source_path(path: str) -> bool:
    """Return whether a changed path is a governed Python source module."""

    return (
        path.endswith(".py")
        and any(path.startswith(prefix) for prefix in STRICT_SOURCE_PREFIXES)
    )


def _source_paths_for_root(
    repository_root: Path,
    source_root: str,
) -> list[str]:
    """List non-package Python modules represented by one manifest root."""

    root_path = repository_root / Path(*source_root.split("/"))
    explicit_file_root = root_path.is_file()
    if explicit_file_root:
        candidates = [root_path]
    elif root_path.is_dir():
        candidates = sorted(root_path.rglob("*.py"))
    else:
        raise ImpactValidationError(
            f"manifest source root does not exist: {source_root}"
        )
    source_paths = [
        normalize_repository_path(
            str(candidate.relative_to(repository_root))
        )
        for candidate in candidates
        if explicit_file_root or candidate.name != "__init__.py"
    ]
    return source_paths


def load_manifest(
    repository_root: Path | None = None,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Load the JSON ownership manifest from the repository."""

    root = repository_root or _repository_root()
    absolute_path = manifest_path
    if not absolute_path.is_absolute():
        absolute_path = root / manifest_path
    try:
        with absolute_path.open("r", encoding="utf-8") as file_handle:
            value = json.load(file_handle)
    except OSError as exc:
        raise ImpactValidationError(
            f"cannot read ownership manifest {absolute_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ImpactValidationError(
            f"ownership manifest is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ImpactValidationError("ownership manifest must be a JSON object")
    return value


def _string_list(value: object, label: str, errors: list[str]) -> list[str]:
    """Validate and return a JSON list of strings."""

    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        errors.append(f"{label} must be a list of strings")
        return []
    return list(value)


def _removed_source_entries(
    manifest: Mapping[str, Any],
    errors: list[str],
    repository_root: Path,
) -> dict[str, list[str]]:
    """Validate and normalize the explicit removed-source vocabulary."""

    if "removed_sources" not in manifest:
        errors.append("removed_sources must be a mapping or list")
        return {}
    raw_entries = manifest.get("removed_sources")
    candidates: list[tuple[object, object]] = []
    if isinstance(raw_entries, Mapping):
        for source_value, nodes_value in raw_entries.items():
            if isinstance(nodes_value, Mapping):
                candidates.append(
                    (source_value, nodes_value.get("required_unit_tests"))
                )
            else:
                candidates.append((source_value, nodes_value))
    elif isinstance(raw_entries, list):
        for index, entry in enumerate(raw_entries):
            if not isinstance(entry, Mapping):
                errors.append(f"removed_sources[{index}] must be an object")
                continue
            candidates.append((entry.get("source"), entry.get("required_unit_tests")))
    else:
        errors.append("removed_sources must be a mapping or list")
        return {}

    normalized_entries: dict[str, list[str]] = {}
    for source_value, nodes_value in candidates:
        if not isinstance(source_value, str) or not source_value.strip():
            errors.append("removed_sources source must be a path string")
            continue
        source = normalize_repository_path(source_value)
        if "*" in source:
            errors.append(
                f"removed_sources source may not contain a wildcard: {source}"
            )
        if not _is_strict_source_path(source):
            errors.append(
                f"removed_sources source must be a strict Python source: {source}"
            )
        if source in normalized_entries:
            errors.append(f"duplicate removed source: {source}")
        normalized_entries[source] = _string_list(
            nodes_value,
            f"removed_sources.{source}.required_unit_tests",
            errors,
        )
        if not normalized_entries[source]:
            errors.append(
                f"removed_sources.{source}.required_unit_tests must not be empty"
            )
        for node_id in normalized_entries[source]:
            if "::" not in node_id:
                errors.append(
                    f"removed_sources.{source} has a non-exact unit node: {node_id}"
                )
        if (repository_root / Path(*source.split("/"))).exists():
            errors.append(f"removed source path must be absent: {source}")
    return normalized_entries


def validate_manifest(
    manifest: dict[str, Any],
    repository_root: Path | None = None,
) -> list[str]:
    """Return every structural and completeness error in the manifest."""

    root = repository_root or _repository_root()
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    source_roots = _string_list(
        manifest.get("source_roots"),
        "source_roots",
        errors,
    )
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        errors.append("entries must be a list")
        entries = []

    removed_sources = _removed_source_entries(manifest, errors, root)

    expected_sources: set[str] = set()
    for source_root in source_roots:
        if "*" in source_root:
            errors.append(f"source root may not contain a wildcard: {source_root}")
            continue
        try:
            expected_sources.update(
                _source_paths_for_root(root, source_root)
            )
        except ImpactValidationError as exc:
            errors.append(str(exc))

    entry_by_source: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            errors.append(f"entries[{index}] must be an object")
            continue
        source = entry.get("source")
        if not isinstance(source, str) or not source:
            errors.append(f"entries[{index}].source must be a path string")
            continue
        normalized_source = normalize_repository_path(source)
        if "*" in normalized_source:
            errors.append(
                f"entries[{index}].source may not contain a wildcard: {source}"
            )
        if normalized_source in entry_by_source:
            errors.append(f"duplicate manifest source: {normalized_source}")
        entry_by_source[normalized_source] = entry
        for required_field in ("owner", "contract"):
            if not isinstance(entry.get(required_field), str) or not entry[
                required_field
            ].strip():
                errors.append(
                    f"{normalized_source}.{required_field} must be non-empty"
                )
        required_unit_tests = _string_list(
            entry.get("required_unit_tests"),
            f"{normalized_source}.required_unit_tests",
            errors,
        )
        if not required_unit_tests:
            errors.append(
                f"{normalized_source}.required_unit_tests must not be empty"
            )
        for node_id in required_unit_tests:
            if "::" not in node_id:
                errors.append(
                    f"{normalized_source} has a non-exact unit node: {node_id}"
                )
        supplemental_tests = _string_list(
            entry.get("supplemental_tests", []),
            f"{normalized_source}.supplemental_tests",
            errors,
        )
        for node_id in supplemental_tests:
            if "::" not in node_id:
                errors.append(
                    f"{normalized_source} has a non-exact supplemental node: "
                    f"{node_id}"
                )

    for source in sorted(expected_sources - set(entry_by_source)):
        errors.append(f"source module has no manifest entry: {source}")
    for source in sorted(set(entry_by_source) - expected_sources):
        errors.append(f"manifest source is outside source_roots: {source}")
    for source in sorted(set(removed_sources) & set(entry_by_source)):
        errors.append(f"source is both active and removed: {source}")
    return errors


def manifest_test_nodes(
    manifest: dict[str, Any],
    *,
    unit_only: bool = True,
) -> list[str]:
    """Return stable, de-duplicated node IDs from a valid manifest."""

    nodes: list[str] = []
    for entry in manifest["entries"]:
        nodes.extend(entry["required_unit_tests"])
        if not unit_only:
            nodes.extend(entry.get("supplemental_tests", []))
    removed_sources = manifest.get("removed_sources", {})
    if isinstance(removed_sources, Mapping):
        for node_ids in removed_sources.values():
            if isinstance(node_ids, list):
                nodes.extend(
                    node_id for node_id in node_ids if isinstance(node_id, str)
                )
    elif isinstance(removed_sources, list):
        for entry in removed_sources:
            if not isinstance(entry, Mapping):
                continue
            node_ids = entry.get("required_unit_tests", [])
            if isinstance(node_ids, list):
                nodes.extend(
                    node_id for node_id in node_ids if isinstance(node_id, str)
                )
    unique_nodes = sorted(set(nodes))
    return unique_nodes


def resolve_impacted_test_nodes(
    manifest: dict[str, Any],
    changed_paths: Sequence[str],
    repository_root: Path | None = None,
) -> list[str]:
    """Resolve changed strict-boundary source paths to exact unit nodes."""

    entries = {
        normalize_repository_path(entry["source"]): entry
        for entry in manifest["entries"]
    }
    errors: list[str] = []
    removed_sources = _removed_source_entries(
        manifest,
        errors,
        repository_root or _repository_root(),
    )
    if errors:
        raise ImpactValidationError("\n".join(errors))
    impacted_nodes: list[str] = []
    for changed_path in changed_paths:
        normalized_path = normalize_repository_path(changed_path)
        if not _is_strict_source_path(normalized_path):
            continue
        if normalized_path in entries:
            impacted_nodes.extend(entries[normalized_path]["required_unit_tests"])
        elif normalized_path in removed_sources:
            impacted_nodes.extend(removed_sources[normalized_path])
        else:
            raise ImpactValidationError(
                f"changed production source has no manifest entry: "
                f"{normalized_path}"
            )
    return sorted(set(impacted_nodes))


def _git_output(repository_root: Path, arguments: Sequence[str]) -> list[str]:
    """Run one fixed-argument Git query and return non-empty path lines."""

    command = ["git", *arguments]
    try:
        result = subprocess.run(
            command,
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise ImpactValidationError(
            f"cannot inspect changed paths with Git: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ImpactValidationError(
            f"Git path query failed with exit code {result.returncode}: {detail}"
        )
    paths = [
        normalize_repository_path(line.strip())
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return paths


def changed_repository_paths(
    repository_root: Path,
    base_ref: str,
) -> list[str]:
    """Return tracked and untracked paths changed from a baseline ref."""

    diff_paths = _git_output(
        repository_root,
        ["diff", "--name-only", base_ref, "--"],
    )
    untracked_paths = _git_output(
        repository_root,
        ["ls-files", "--others", "--exclude-standard"],
    )
    changed_paths = sorted(set(diff_paths) | set(untracked_paths))
    return changed_paths


def missing_collected_nodes(
    required_nodes: Sequence[str],
    collected_nodes: Sequence[str],
) -> list[str]:
    """Return required node IDs absent from the current pytest collection."""

    normalized_collected = {
        normalize_repository_path(node_id) for node_id in collected_nodes
    }
    missing_nodes = sorted(
        {
            node_id
            for node_id in required_nodes
            if normalize_repository_path(node_id) not in normalized_collected
        }
    )
    return missing_nodes


def _run_pytest(
    repository_root: Path,
    node_ids: Sequence[str],
    *,
    collect_only: bool,
) -> None:
    """Run exact nodes under the project interpreter and enforcement hook."""

    if not node_ids:
        return
    environment = os.environ.copy()
    environment["KAZUSA_TEST_IMPACT_REQUIRED"] = "1"
    environment["KAZUSA_TEST_IMPACT_NODES"] = json.dumps(list(node_ids))
    command = [sys.executable, "-m", "pytest"]
    if collect_only:
        command.extend(["--collect-only", "-q"])
    else:
        command.append("-q")
    command.extend(node_ids)
    try:
        result = subprocess.run(
            command,
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    except OSError as exc:
        raise ImpactValidationError(
            f"cannot execute pytest for impacted nodes: {exc}"
        ) from exc
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ImpactValidationError(
            f"pytest impact command failed with exit code "
            f"{result.returncode}: {detail}"
        )


def _argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the impact verifier."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate source ownership and run exact deterministic pytest nodes."
        )
    )
    parser.add_argument(
        "--base-ref",
        help="Git ref used to resolve changed paths.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Repository-relative ownership manifest path.",
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Collect every manifest unit node instead of only changed nodes.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the exact nodes after collection succeeds.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the manifest and optionally collect and run its exact nodes."""

    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    repository_root = _repository_root()
    try:
        manifest = load_manifest(
            repository_root,
            Path(arguments.manifest),
        )
        manifest_errors = validate_manifest(manifest, repository_root)
        if manifest_errors:
            joined_errors = "\n".join(f"- {error}" for error in manifest_errors)
            raise ImpactValidationError(joined_errors)
        if arguments.check_all:
            node_ids = manifest_test_nodes(manifest)
        elif arguments.base_ref:
            changed_paths = changed_repository_paths(
                repository_root,
                arguments.base_ref,
            )
            node_ids = resolve_impacted_test_nodes(
                manifest,
                changed_paths,
                repository_root,
            )
            print(
                "Changed source paths: "
                + ", ".join(
                    path
                    for path in changed_paths
                    if _is_strict_source_path(path)
                )
            )
        else:
            node_ids = []
        if arguments.run and not arguments.base_ref and not arguments.check_all:
            raise ImpactValidationError(
                "--run requires --base-ref or --check-all"
            )
        if node_ids:
            _run_pytest(repository_root, node_ids, collect_only=True)
            if arguments.run:
                _run_pytest(repository_root, node_ids, collect_only=False)
            print(f"Validated {len(node_ids)} exact impact-test node(s).")
        else:
            print("No impacted strict-boundary source nodes were selected.")
    except ImpactValidationError as exc:
        print(f"test-impact validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
