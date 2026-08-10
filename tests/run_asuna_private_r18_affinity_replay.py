"""Run and review the complete Asuna private R18 E2E sequences."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4


_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_PATH = _ROOT / "venv" / "Scripts" / "python.exe"
_TEST_DATABASE_NAME = "_test_kazusa_live_llm"
_ARTIFACT_ROOT = (
    _ROOT
    / "test_artifacts"
    / "cognition_core_v2"
    / "asuna_private_r18_affinity_replay"
)
_MANIFEST_PATH = (
    _ROOT
    / "test_artifacts"
    / "cognition_core_v2"
    / "private_r18_replay"
    / "replay_manifest.json"
)
_LIVE_TEST_NODE = (
    "tests/test_asuna_private_r18_affinity_live_llm.py::"
    "test_live_asuna_private_r18_affinity_sequence"
)
_RESTORE_TEST_NODE = (
    "tests/test_asuna_private_r18_affinity_live_llm.py::"
    "test_restore_asuna_private_r18_baseline"
)
_CONDITIONS = ("high_affinity", "default_affinity")
_SEQUENCE_TIMEOUT_SECONDS = 7200


def _configure_utf8_streams() -> None:
    """Keep exact CJK inputs and live output printable on Windows."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_streams()


def _json_safe(value: object) -> object:
    """Convert controller values into JSON-safe evidence."""

    if isinstance(value, dict):
        json_value = {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    elif isinstance(value, list):
        json_value = [_json_safe(item) for item in value]
    elif isinstance(value, (str, int, float, bool)) or value is None:
        json_value = value
    else:
        json_value = str(value)
    return json_value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one controller JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one controller JSON artifact."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _guarded_path(path: Path) -> Path:
    """Require a controller artifact path below the replay root."""

    candidate = path.resolve()
    guard_root = _ARTIFACT_ROOT.resolve()
    if candidate == guard_root or guard_root not in candidate.parents:
        raise ValueError(f"replay path escaped artifact root: {candidate}")
    return candidate


def _condition_root(*, run_id: str, condition: str) -> Path:
    """Return one guarded condition directory."""

    if condition not in _CONDITIONS:
        raise ValueError(f"unsupported condition: {condition}")
    condition_root = _guarded_path(_ARTIFACT_ROOT / run_id / condition)
    return condition_root


def _new_run_id() -> str:
    """Build a readable collision-resistant run identifier."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"e2e_{timestamp}_{uuid4().hex[:10]}"
    return run_id


def _load_source_manifest() -> dict[str, Any]:
    """Validate the exact twenty-input source at the controller boundary."""

    manifest = _load_json_object(_MANIFEST_PATH)
    if manifest.get("schema_version") != "real_conversation_replay.v2":
        raise ValueError("frozen private R18 manifest schema is invalid")
    if manifest.get("scenario") != "private_r18":
        raise ValueError("frozen private R18 manifest scenario is invalid")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError("frozen private R18 manifest is not exactly 20 cases")
    indexes = [
        case.get("case_index")
        for case in cases
        if isinstance(case, dict)
    ]
    if indexes != list(range(1, 21)):
        raise ValueError("frozen private R18 input order is invalid")
    return manifest


def _child_environment(
    *,
    run_id: str,
    condition: str,
    output_root: Path,
) -> dict[str, str]:
    """Build the guarded environment for one full-sequence child."""

    environment = dict(os.environ)
    environment.pop("PYTHON_DOTENV_DISABLED", None)
    environment.update({
        "ASUNA_R18_RUN_ID": run_id,
        "ASUNA_R18_CONDITION": condition,
        "ASUNA_R18_OUTPUT_ROOT": str(output_root),
        "MONGODB_DB_NAME": _TEST_DATABASE_NAME,
        "KAZUSA_TEST_DB_GUARD": "1",
        "CHARACTER_TIME_ZONE": "Pacific/Auckland",
        "SELF_COGNITION_ENABLED": "false",
        "CALENDAR_SCHEDULER_ENABLED": "false",
        "BACKGROUND_WORK_WORKER_ENABLED": "false",
        "REFLECTION_CYCLE_ENABLED": "false",
        "LLM_TRACE_CAPTURE_MODE": "full",
        "PYTEST_ADDOPTS": "",
    })
    return environment


def _run_condition(
    *,
    run_id: str,
    condition: str,
) -> dict[str, Any]:
    """Run all twenty turns in one child process for one condition."""

    output_root = _condition_root(run_id=run_id, condition=condition)
    run_manifest_path = _guarded_path(output_root / "run_manifest.json")
    if run_manifest_path.exists():
        raise ValueError(f"condition run already exists: {run_manifest_path}")
    environment = _child_environment(
        run_id=run_id,
        condition=condition,
        output_root=output_root,
    )
    completed = subprocess.run(
        [
            str(_PYTHON_PATH),
            "-m",
            "pytest",
            _LIVE_TEST_NODE,
            "-q",
            "-s",
            "-o",
            "addopts=",
        ],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_SEQUENCE_TIMEOUT_SECONDS,
        check=False,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "sequence.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (output_root / "sequence.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if not run_manifest_path.is_file():
        raise RuntimeError(
            f"full E2E child produced no run manifest: {condition}; "
            f"exit={completed.returncode}"
        )
    run_manifest = _load_json_object(run_manifest_path)
    if completed.returncode != 0:
        raise RuntimeError(
            f"full E2E child failed: {condition}; "
            f"status={run_manifest.get('technical_status')}; "
            f"exit={completed.returncode}"
        )
    if run_manifest.get("completed_case_indexes") != list(range(1, 21)):
        raise RuntimeError(
            f"full E2E child did not complete all twenty turns: {condition}"
        )
    return run_manifest


def _run_restore(*, run_id: str) -> dict[str, Any]:
    """Run the final guarded default-baseline restoration child."""

    output_root = _condition_root(run_id=run_id, condition="default_affinity")
    restore_path = _guarded_path(output_root / "final_restore.json")
    environment = dict(os.environ)
    environment.pop("PYTHON_DOTENV_DISABLED", None)
    environment.update({
        "MONGODB_DB_NAME": _TEST_DATABASE_NAME,
        "KAZUSA_TEST_DB_GUARD": "1",
        "ASUNA_R18_RESTORE_OUTPUT_PATH": str(restore_path),
        "CHARACTER_TIME_ZONE": "Pacific/Auckland",
        "SELF_COGNITION_ENABLED": "false",
        "CALENDAR_SCHEDULER_ENABLED": "false",
        "BACKGROUND_WORK_WORKER_ENABLED": "false",
        "REFLECTION_CYCLE_ENABLED": "false",
        "PYTEST_ADDOPTS": "",
    })
    completed = subprocess.run(
        [
            str(_PYTHON_PATH),
            "-m",
            "pytest",
            _RESTORE_TEST_NODE,
            "-q",
            "-s",
            "-o",
            "addopts=",
        ],
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_SEQUENCE_TIMEOUT_SECONDS,
        check=False,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "restore.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (output_root / "restore.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if not restore_path.is_file():
        raise RuntimeError("final restore produced no evidence artifact")
    artifact = _load_json_object(restore_path)
    if completed.returncode != 0 or artifact.get("technical_status") != "passed":
        raise RuntimeError("final guarded baseline restoration failed")
    return artifact


def _run_all(args: argparse.Namespace) -> int:
    """Run both conditions and restoration, retaining data-only evidence."""

    _load_source_manifest()
    run_id = str(args.run_id or "").strip() or _new_run_id()
    high_run = _run_condition(run_id=run_id, condition="high_affinity")
    default_run = _run_condition(run_id=run_id, condition="default_affinity")
    restore = _run_restore(run_id=run_id)
    print(json.dumps({
        "run_id": run_id,
        "high_turns": len(high_run["completed_case_indexes"]),
        "default_turns": len(default_run["completed_case_indexes"]),
        "condition_manifests": [
            str(_condition_root(
                run_id=run_id,
                condition="high_affinity",
            ) / "run_manifest.json"),
            str(_condition_root(
                run_id=run_id,
                condition="default_affinity",
            ) / "run_manifest.json"),
        ],
        "review_status": "agent_authored_review_required",
        "restore": restore,
    }, ensure_ascii=False, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the full-sequence controller CLI."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_all = subparsers.add_parser("run-all")
    run_all.add_argument("--run-id", default="")
    run_one = subparsers.add_parser("run-condition")
    run_one.add_argument("--condition", choices=_CONDITIONS, required=True)
    run_one.add_argument("--run-id", required=True)
    restore = subparsers.add_parser("restore-final")
    restore.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one controller command."""

    args = _build_parser().parse_args(argv)
    if args.command == "run-all":
        exit_code = _run_all(args)
    elif args.command == "run-condition":
        _load_source_manifest()
        result = _run_condition(
            run_id=args.run_id,
            condition=args.condition,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        exit_code = 0
    elif args.command == "restore-final":
        print(json.dumps(
            _run_restore(run_id=args.run_id),
            ensure_ascii=False,
            indent=2,
        ))
        exit_code = 0
    else:
        raise ValueError(f"unsupported controller command: {args.command}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
