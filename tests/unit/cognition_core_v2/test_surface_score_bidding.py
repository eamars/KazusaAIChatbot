"""Deterministic checks for owner-specific surface score artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path


ARTIFACT_PATH = Path(
    "experiments/cognition_surface_score_bidding/thresholds.json"
)
EXPECTED_OWNERS = {
    "surface_content_plan",
}


def _load_threshold_artifact() -> dict[str, object]:
    """Load the checked-in owner-specific threshold artifact."""

    with ARTIFACT_PATH.open(encoding="utf-8") as file_handle:
        artifact = json.load(file_handle)
    assert isinstance(artifact, dict)
    return artifact


def test_calibration_artifact_requires_owner_specific_threshold() -> None:
    """Each included owner has an independently recorded bounded threshold."""

    artifact = _load_threshold_artifact()
    assert artifact["schema_version"] == (
        "cognition_surface_score_thresholds.v1"
    )
    assert artifact["status"] == "blocked_pending_calibration"
    assert artifact["accepted_for_production"] is False
    owners = artifact["owners"]
    assert isinstance(owners, dict)
    assert set(owners) == EXPECTED_OWNERS

    for owner in EXPECTED_OWNERS:
        owner_artifact = owners[owner]
        assert isinstance(owner_artifact, dict)
        threshold = owner_artifact["threshold"]
        assert isinstance(threshold, (int, float))
        assert not isinstance(threshold, bool)
        assert math.isfinite(float(threshold))
        assert 0.0 <= float(threshold) <= 1.0
        assert owner_artifact["threshold_status"] == (
            "placeholder_not_calibrated"
        )
        assert owner_artifact["calibration_contexts"] == 0
        assert owner_artifact["held_out_contexts"] == 0
        assert owner_artifact["score_samples"] == []


def test_calibration_artifact_rejects_boolean_or_nonfinite_score() -> None:
    """The artifact contract treats invalid score values as unusable."""

    artifact = _load_threshold_artifact()
    owners = artifact["owners"]
    assert isinstance(owners, dict)

    for owner_artifact in owners.values():
        assert isinstance(owner_artifact, dict)
        score_samples = owner_artifact["score_samples"]
        assert isinstance(score_samples, list)
        for score in score_samples:
            assert isinstance(score, (int, float))
            assert not isinstance(score, bool)
            assert math.isfinite(float(score))
            assert 0.0 <= float(score) <= 1.0
