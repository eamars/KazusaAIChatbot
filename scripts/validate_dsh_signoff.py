"""Validate one complete, current DSH live sign-off artifact set."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tests.dsh_trigger_source_e2e_support import (
    CONFIGURED_WEATHER_CASE_SPEC,
    SIDECAR_LOSS_CASE_SPEC,
    signoff_case_ids,
    signoff_code_fingerprint,
)

COMMON_ARTIFACTS = frozenset({
    "behavior_review_input.json",
    "callbacks.json",
    "case_result.json",
    "case_spec.json",
    "cleanup.json",
    "dsh_lineage.json",
    "mongo_state.json",
    "source_execution.json",
    "trace.json",
})


def _read_object(path: Path) -> dict[str, Any]:
    """Read one required JSON object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read sign-off artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"sign-off artifact is not an object: {path}")
    return value


def _latest_case_directories(artifact_root: Path) -> dict[str, Path]:
    """Select the newest result for each case, including newest failures."""

    selected: dict[str, tuple[float, Path]] = {}
    for result_path in artifact_root.glob("*/case_result.json"):
        result = _read_object(result_path)
        case_id = result.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            continue
        modified = result_path.stat().st_mtime
        current = selected.get(case_id)
        if current is None or modified > current[0]:
            selected[case_id] = (modified, result_path.parent)
    return {case_id: value[1] for case_id, value in selected.items()}


def validate_case_artifact(
    artifact_dir: Path,
    *,
    expected_fingerprint: str,
) -> dict[str, Any]:
    """Validate one case dossier and return its manifest row."""

    result = _read_object(artifact_dir / "case_result.json")
    case_id = result.get("case_id")
    if result.get("technical_status") != "passed":
        raise ValueError(
            f"latest {case_id} result failed: {result.get('failures', [])}"
        )
    if result.get("signoff_code_fingerprint") != expected_fingerprint:
        raise ValueError(f"latest {case_id} result certifies stale code")
    checks = result.get("checks")
    if not isinstance(checks, dict) or not checks:
        raise ValueError(f"latest {case_id} result has no sign-off checks")
    failed_checks = [name for name, passed in checks.items() if passed is not True]
    if failed_checks:
        raise ValueError(f"latest {case_id} has failed checks: {failed_checks}")

    missing_artifacts = sorted(
        name for name in COMMON_ARTIFACTS if not (artifact_dir / name).is_file()
    )
    if missing_artifacts:
        raise ValueError(
            f"latest {case_id} is missing artifacts: {missing_artifacts}"
        )
    cleanup = _read_object(artifact_dir / "cleanup.json")
    if cleanup.get("database_dropped") is not True or cleanup.get("errors"):
        raise ValueError(f"latest {case_id} cleanup is incomplete: {cleanup}")

    if case_id == CONFIGURED_WEATHER_CASE_SPEC.case_id:
        if cleanup.get("services_stopped") is not True:
            raise ValueError("configured service canary left owned services running")
        configured_artifacts = (
            "configured_readiness.json",
            "configured_service_states.json",
        )
        if any(not (artifact_dir / name).is_file() for name in configured_artifacts):
            raise ValueError("configured service canary lacks lifecycle evidence")
    else:
        required_cleanup = ("adapter_stopped", "server_stopped", "sidecar_stopped")
        if any(cleanup.get(name) is not True for name in required_cleanup):
            raise ValueError(f"latest {case_id} left runtime resources active")
        readiness_file = (
            "readiness_after_sidecar_loss.json"
            if case_id == SIDECAR_LOSS_CASE_SPEC.case_id
            else "readiness_after_source.json"
        )
        for name in ("readiness_before_source.json", readiness_file):
            if not (artifact_dir / name).is_file():
                raise ValueError(f"latest {case_id} lacks {name}")

    return {
        "case_id": case_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "duration_ms": result.get("duration_ms"),
        "check_count": len(checks),
        "technical_status": "passed",
    }


def validate_artifact_root(artifact_root: Path) -> dict[str, Any]:
    """Validate the newest complete twelve-case sign-off campaign."""

    current_fingerprint = signoff_code_fingerprint()
    selected = _latest_case_directories(artifact_root)
    required_case_ids = signoff_case_ids()
    missing = [case_id for case_id in required_case_ids if case_id not in selected]
    if missing:
        raise ValueError(f"missing DSH sign-off cases: {missing}")
    unexpected = sorted(set(selected).difference(required_case_ids))
    cases = [
        validate_case_artifact(
            selected[case_id],
            expected_fingerprint=current_fingerprint,
        )
        for case_id in required_case_ids
    ]
    return {
        "schema_version": "dsh_signoff_manifest.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "signoff_code_fingerprint": current_fingerprint,
        "required_case_count": len(required_case_ids),
        "cases": cases,
        "ignored_unrecognized_case_ids": unexpected,
        "status": "passed",
    }


def _parse_args() -> argparse.Namespace:
    """Parse validator arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--write-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    """Validate and optionally persist the current sign-off manifest."""

    args = _parse_args()
    try:
        manifest = validate_artifact_root(args.artifact_root.resolve())
    except ValueError as exc:
        print(f"DSH sign-off failed: {exc}")
        return 1
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    if args.write_manifest is not None:
        destination = args.write_manifest.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
