"""Deterministic tests for DSH sign-off artifact validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_dsh_signoff import COMMON_ARTIFACTS, validate_case_artifact
from tests import dsh_trigger_source_e2e_support as signoff_support


def _write_json(path: Path, value: object) -> None:
    """Write one small validator fixture."""

    path.write_text(json.dumps(value), encoding="utf-8")


def _complete_case_artifact(tmp_path: Path) -> Path:
    """Create one internally complete normal-case dossier."""

    artifact_dir = tmp_path / "user_message_local_fact"
    artifact_dir.mkdir()
    for name in COMMON_ARTIFACTS:
        _write_json(artifact_dir / name, {})
    _write_json(
        artifact_dir / "case_result.json",
        {
            "case_id": "user_message_local_fact",
            "technical_status": "passed",
            "signoff_code_fingerprint": "sha256:current",
            "checks": {"source_trace_succeeded": True},
            "duration_ms": 10,
        },
    )
    _write_json(
        artifact_dir / "cleanup.json",
        {
            "database_dropped": True,
            "adapter_stopped": True,
            "server_stopped": True,
            "sidecar_stopped": True,
            "errors": [],
        },
    )
    _write_json(artifact_dir / "readiness_before_source.json", {})
    _write_json(artifact_dir / "readiness_after_source.json", {})
    return artifact_dir


def test_validator_rejects_stale_live_artifacts(tmp_path: Path) -> None:
    """Changed code cannot be certified by an earlier passing case."""

    artifact_dir = _complete_case_artifact(tmp_path)

    with pytest.raises(ValueError, match="stale code"):
        validate_case_artifact(
            artifact_dir,
            expected_fingerprint="sha256:new-code",
        )


def test_validator_accepts_complete_current_case(tmp_path: Path) -> None:
    """A complete current dossier should produce one manifest row."""

    artifact_dir = _complete_case_artifact(tmp_path)

    result = validate_case_artifact(
        artifact_dir,
        expected_fingerprint="sha256:current",
    )

    assert result["case_id"] == "user_message_local_fact"
    assert result["technical_status"] == "passed"


def test_signoff_fingerprint_covers_entire_production_source_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any production source edit should invalidate earlier live artifacts."""

    owner = (
        tmp_path
        / "src"
        / "kazusa_ai_chatbot"
        / "reflection_cycle"
        / "worker.py"
    )
    owner.parent.mkdir(parents=True)
    owner.write_text("first production version\n", encoding="utf-8")
    assert "src" in signoff_support.SIGNOFF_FINGERPRINT_PATHS
    monkeypatch.setattr(signoff_support, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(signoff_support, "SIGNOFF_FINGERPRINT_PATHS", ("src",))

    first = signoff_support.signoff_code_fingerprint()
    owner.write_text("second production version\n", encoding="utf-8")
    second = signoff_support.signoff_code_fingerprint()

    assert first != second
