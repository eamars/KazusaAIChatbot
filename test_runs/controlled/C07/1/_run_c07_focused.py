"""Run the current C07 fixture once without advancing the corpus ledger."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests"))

import cognition_baseline_comparison as controller  # noqa: E402


def main() -> int:
    """Generate and execute one current-checkout C07::r1 worker manifest."""

    corpus = "post_fix_v2"
    repetition = 1
    case = next(
        dict(row)
        for row in controller._load_case_rows()
        if row["case_id"] == "C07"
    )
    case["repetition_ordinal"] = repetition
    case["execution_id"] = "C07::r1"

    revision, target_root, revision_sha = controller._corpus_config(corpus)
    workspace_root = controller._case_workspace_root(case)
    database_name = controller._database_name(
        corpus,
        str(case["case_id"]),
        repetition,
    )
    case_root = controller._EVIDENCE_ROOT / corpus / "C07"
    manifest_path = case_root / "r1.input.json"
    output_path = case_root / "r1.json"
    worker_manifest = {
        "schema_version": "cognition_baseline_worker_input.v1",
        "execution_id": "C07::r1",
        "corpus": corpus,
        "revision": revision,
        "revision_sha": revision_sha,
        "target_root": str(target_root.resolve()),
        "profile_path": str(controller._PROFILE_PATH),
        "profile_sha256": controller._PROFILE_SHA256,
        "history_path": str(controller._HISTORY_PATH),
        "history_sha256": controller._HISTORY_SHA256,
        "database_name": database_name,
        "fixed_local_timestamp": controller._FIXED_LOCAL_TIMESTAMP,
        "fixed_scheduled_local_timestamp": (
            controller._FIXED_SCHEDULED_LOCAL_TIMESTAMP
        ),
        "case_sha256": controller._case_digest(case),
        "workspace_root": str(workspace_root),
        "output_path": str(output_path.resolve()),
        "case": case,
    }
    controller._write_json(manifest_path, worker_manifest)
    output_path.unlink(missing_ok=True)

    worker_path = ROOT / "tests" / "cognition_baseline_worker.py"
    python_path = ROOT / "venv" / "Scripts" / "python.exe"
    completed = subprocess.run(
        [
            str(python_path),
            str(worker_path),
            "--input",
            str(manifest_path.resolve()),
        ],
        cwd=ROOT,
        env=controller._worker_environment(
            database_name=database_name,
            target_root=target_root,
            workspace_root=workspace_root,
        ),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (case_root / "r1.worker.stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (case_root / "r1.worker.stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if not output_path.is_file():
        raise RuntimeError(
            "C07::r1 worker produced no artifact; "
            f"exit={completed.returncode}"
        )
    artifact = controller._load_json(output_path)
    print(json.dumps({
        "execution_id": artifact.get("execution_id"),
        "technical_status": artifact.get("technical_status"),
        "hard_gate_failures": artifact.get("hard_gate_failures", []),
        "failure_type": artifact.get("failure_type"),
        "failure_message": artifact.get("failure_message"),
        "artifact_path": str(output_path),
        "worker_exit_code": completed.returncode,
    }, ensure_ascii=False, indent=2))
    if artifact.get("technical_status") != "passed":
        return completed.returncode or 1
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
