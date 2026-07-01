"""Fixture coverage tests for complex-task resolver review cases."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "complex_task_resolver_review_cases.json"
)


def test_complex_task_resolver_review_fixture_has_required_coverage(
    capsys,
) -> None:
    """Validate fixture metadata without treating expected answers as prompts."""

    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    category_counts = Counter()
    review_outcome_counts = Counter()
    required_stages = set()
    for case in cases:
        category_counts[case["category"]] += 1
        review_outcome_counts[case["expected_review_outcome"]] += 1
        required_stages.update(case["required_stages"])
        assert case["minimum_viable_answer"]
        assert case["expected_final_answer"]
        assert case["forbidden_failure_modes"]

    coverage_summary = {
        "case_count": len(cases),
        "categories": dict(sorted(category_counts.items())),
        "expected_review_outcomes": dict(
            sorted(review_outcome_counts.items())
        ),
        "required_stages": sorted(required_stages),
        "review_metadata_only": fixture["anti_cheat_contract"][
            "fixture_is_review_metadata_only"
        ],
    }
    print(json.dumps(coverage_summary, ensure_ascii=True, sort_keys=True))

    assert len(cases) == 32
    assert coverage_summary["review_metadata_only"] is True
    assert "web_retrieval" in required_stages
    assert "summarization" in required_stages
    assert "arithmetic" in required_stages
