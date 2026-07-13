"""Deterministic lifecycle coverage for every required V2 emotion family."""

import json
from pathlib import Path

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import LocalMotivationalState
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import run_lifecycle_case
from kazusa_ai_chatbot.cognition_core_v2.emotion_derivation import (
    derive_emotion_activations,
)
from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
    EMOTION_DEFINITIONS,
)


_FIXTURE_PATH = Path(
    "tests/fixtures/cognition_core_v2_emotion_lifecycle_cases.json",
)


def _cases() -> list[dict[str, object]]:
    """Load the explicit lifecycle cases used by the deterministic harness."""

    fixture_text = _FIXTURE_PATH.read_text(encoding="utf-8")
    cases = json.loads(fixture_text)
    return cases


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["case_id"])
def test_each_required_emotion_has_a_causal_lifecycle_case(
    case: dict[str, object],
) -> None:
    """Require baseline, begin, sustain, fade, and missing-root evidence."""

    emotion_id = case["emotion_id"]
    result = run_lifecycle_case(case)
    phase_results = result["phases"]

    assert emotion_id in EMOTION_DEFINITIONS
    assert phase_results["baseline"]["activation"] == 0.0
    assert phase_results["begin"]["activation"] > 0.0
    assert phase_results["begin"]["trend"] == "beginning"
    assert phase_results["sustain"]["activation"] > 0.0
    assert phase_results["sustain"]["trend"] == "sustained"
    assert phase_results["fade"]["activation"] < phase_results["sustain"]["activation"]
    assert phase_results["fade"]["trend"] in {"fading", "inactive"}
    assert phase_results["negative_control"]["activation"] == 0.0
    assert phase_results["negative_control"]["guard_passed"] is False


def test_required_fixture_covers_exactly_the_twenty_one_emotion_ids() -> None:
    """Prevent accidental omissions or additions to the approved test matrix."""

    fixture_ids = {case["emotion_id"] for case in _cases()}

    assert fixture_ids == set(EMOTION_DEFINITIONS)
    assert len(fixture_ids) == 21


def test_compatible_emotions_activate_concurrently_from_distinct_roots() -> None:
    """Preserve distinct compatible causes instead of collapsing them early."""

    activations = derive_emotion_activations(
        LocalMotivationalState(),
        {
            "goal_reward": 0.8,
            "attributed_benefit": 0.6,
        },
    )

    assert activations["joy"].trend == "beginning"
    assert activations["joy"].activation == 0.8
    assert activations["gratitude"].trend == "beginning"
    assert activations["gratitude"].activation == 0.6


def test_lifecycle_fixture_rejects_a_causally_mismatched_emotion_root() -> None:
    """Fail closed when a case claims an emotion's wrong causal source."""

    invalid_case = {
        "case_id": "invalid-joy-root",
        "emotion_id": "joy",
        "root": "credible_threat",
    }

    with pytest.raises(ValueError, match="lifecycle root does not match"):
        run_lifecycle_case(invalid_case)
