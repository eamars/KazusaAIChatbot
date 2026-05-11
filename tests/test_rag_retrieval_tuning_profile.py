import json
from pathlib import Path

import pytest

from scripts.profile_rag_retrieval import (
    evaluate_profile_case,
    load_profile_cases,
    project_profile_row,
)


def _case(case_id: str, kind: str = "positive") -> dict[str, object]:
    return {
        "case_id": case_id,
        "kind": kind,
        "platform": "qq",
        "platform_channel_id": "905393941",
        "query": "query",
        "expected_any": ["expected"] if kind == "positive" else [],
        "forbidden_any": ["forbidden"],
    }


def test_profile_case_loader_requires_positive_and_negative_cases(
    tmp_path: Path,
) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([_case("positive")]), encoding="utf-8")

    with pytest.raises(ValueError, match="negative"):
        load_profile_cases(cases_path)

    cases_path.write_text(
        json.dumps([_case("positive"), _case("negative", kind="negative")]),
        encoding="utf-8",
    )

    loaded_cases = load_profile_cases(cases_path)

    assert [case["case_id"] for case in loaded_cases] == ["positive", "negative"]


def test_profile_metrics_marks_missing_expected_terms_as_false_negative() -> None:
    result = evaluate_profile_case(
        _case("positive"),
        [
            {
                "rank": 1,
                "score": 0.91,
                "body_text": "unrelated result",
                "speaker_display_name": "speaker",
            }
        ],
    )

    assert result["resolved"] is False
    assert result["false_negative"] is True
    assert result["matched_expected_terms"] == []


def test_profile_metrics_marks_forbidden_terms_as_false_positive() -> None:
    result = evaluate_profile_case(
        _case("negative", kind="negative"),
        [
            {
                "rank": 1,
                "score": 0.88,
                "body_text": "this row contains forbidden evidence",
                "speaker_display_name": "speaker",
            }
        ],
    )

    assert result["resolved"] is False
    assert result["false_positive"] is True
    assert result["matched_forbidden_terms"] == ["forbidden"]


def test_profile_metrics_records_max_score() -> None:
    result = evaluate_profile_case(
        _case("positive"),
        [
            {"score": 0.41, "body_text": "unrelated"},
            {"score": 0.82, "body_text": "expected evidence"},
        ],
    )

    assert result["max_score"] == 0.82


def test_profile_metrics_bounds_row_text_for_artifacts() -> None:
    projected = project_profile_row(
        rank=1,
        row={
            "score": 0.75,
            "body_text": "x" * 300,
            "speaker_display_name": "speaker",
        },
        text_limit=24,
    )

    assert projected["body_text"] == ("x" * 24)
    assert projected["body_text_truncated"] is True
    assert "embedding" not in projected
