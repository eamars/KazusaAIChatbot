"""Deterministic tests for cognition-stage connection case helpers."""

from __future__ import annotations

import json
from pathlib import Path

from tests.cognition_stage_connection_cases import (
    build_cognition_connection_comparison_report,
    load_cognition_stage_connection_case_set,
    select_cognition_stage_connection_case,
)


def test_case_loader_selects_one_captured_case(tmp_path: Path) -> None:
    """Captured case sets should load without running live LLMs."""

    case_file = tmp_path / "cases.json"
    case_file.write_text(
        json.dumps(
            {
                "schema_version": "cognition_stage_connection_case_set.v1",
                "source_platform": "qq",
                "source_platform_user_id": "673225019",
                "source_kind": "qq_private",
                "created_at": "2026-05-16T00:00:00+00:00",
                "cases": [
                    {
                        "schema_version": "cognition_stage_connection_case.v1",
                        "case_id": "qq_private_001",
                        "source_kind": "qq_private",
                        "source_channel_type": "private",
                        "source_channel_id": "private_scope_001",
                        "seed_state": {"user_input": "hello"},
                        "historical_user_message": "hello",
                        "historical_assistant_reply": ["hi"],
                        "historical_comparison": {
                            "expected_visible_surface": True,
                            "comparison_basis": "assistant replied",
                        },
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    case_set = load_cognition_stage_connection_case_set(case_file)
    selected_case = select_cognition_stage_connection_case(
        case_set,
        "qq_private_001",
    )

    assert selected_case["historical_user_message"] == "hello"


def test_comparison_report_requires_speak_for_historical_reply() -> None:
    """A historical assistant reply should require a visible speak route."""

    case = {
        "historical_comparison": {
            "expected_visible_surface": True,
        }
    }

    report = build_cognition_connection_comparison_report(
        case,
        action_specs=[],
        final_dialog=[],
        l3_ran=False,
        l4_ran=False,
    )

    assert report["ok"] is False
    assert report["errors"] == [
        "historical assistant reply expected a visible speak action",
    ]
