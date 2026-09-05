"""Subprocess integration tests for the permanent DSH runtime probe."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBE_SCRIPT = PROJECT_ROOT / "experiments" / "dsh_runtime_probe.py"


def _run_probe(tmp_path: Path, probe_name: str) -> dict[str, object]:
    """Run one public CLI probe and return its persisted result."""

    artifact_dir = tmp_path / probe_name
    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE_SCRIPT),
            probe_name,
            "--artifact-dir",
            str(artifact_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        check=False,
    )
    result_path = artifact_dir / "result.json"
    if not result_path.is_file():
        pytest.fail(
            f"probe returned {completed.returncode} without result.json: "
            f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert completed.returncode == 0, result
    assert result["schema_version"] == "dsh_runtime_probe_result.v1"
    assert result["probe_name"] == probe_name
    assert result["status"] == "passed"
    assert result["tested_revision"]["commit"]
    assert result["tested_revision"]["dirty_state_digest"].startswith("sha256:")
    assert result["processes"]
    assert all(row["exit_code"] is not None for row in result["processes"])
    return result


def test_sidecar_lifecycle_probe(tmp_path: Path) -> None:
    """The public probe covers semantic, restart, and replay process behavior."""

    result = _run_probe(tmp_path, "sidecar-lifecycle")

    kinds = {row["kind"] for row in result["observations"]}
    assert kinds == {
        "authenticated_boot_and_semantic_worker",
        "sqlite_checkpoint_restart",
        "terminal_commit_response_loss_replay",
    }


@pytest.mark.live_db
def test_brain_task_lifecycle_probe(tmp_path: Path) -> None:
    """The public task service binds the real controller and guarded Mongo."""

    result = _run_probe(tmp_path, "brain-task-lifecycle")

    observation = result["observations"][0]
    assert observation["task_status"] == "resolved"
    assert observation["binding_state"] == "consumed_inline"
    assert observation["database_name"].startswith("_test_kazusa_dsh_probe_")
    assert any(row["status"] == "dropped" for row in result["cleanup"])


@pytest.mark.live_db
def test_transport_loss_probe(tmp_path: Path) -> None:
    """Owned sidecar loss becomes blocked evidence and a faulted binding."""

    result = _run_probe(tmp_path, "transport-loss")

    observation = result["observations"][0]
    assert observation["readiness_before_loss"] == "ready"
    assert observation["observation_status"] == "failed"
    assert observation["evidence_state"] == "blocked"
    assert observation["binding_state"] == "faulted"
