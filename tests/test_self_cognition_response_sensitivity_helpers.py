"""Deterministic helper tests for self-cognition sensitivity replay."""

from __future__ import annotations

from tests.test_self_cognition_response_sensitivity_live_llm import (
    HistoricalSensitivityCase,
    _balanced_case_prefix,
    _historical_expected_speak,
    _observed_user_visible_speak,
    _parse_group_activity_case_id,
)


def test_parse_group_activity_case_id_extracts_window_bounds() -> None:
    """A historical group-review case id should expose its source window."""

    parsed = _parse_group_activity_case_id(
        "group_activity_window:"
        "scope_abcdef123456:"
        "2026-05-18T04:00:00+00:00:"
        "2026-05-18T04:15:00+00:00"
    )

    assert parsed == {
        "scope_ref": "scope_abcdef123456",
        "window_start": "2026-05-18T04:00:00+00:00",
        "window_end": "2026-05-18T04:15:00+00:00",
    }


def test_observed_user_visible_speak_reads_l2d_action_specs() -> None:
    """Only user-visible speak specs count as observed speech."""

    action_specs = [
        {"kind": "memory_lifecycle_update", "visibility": "private"},
        {"kind": "speak", "visibility": "user_visible"},
    ]

    observed = _observed_user_visible_speak(action_specs)

    assert observed is True


def test_historical_expected_speak_uses_route_and_output_mode() -> None:
    """Historical labels should come from self-cognition route metadata."""

    spoke_event = {
        "payload": {
            "selected_route": "action_candidate",
            "output_mode": "scheduled_action_request",
        }
    }
    silent_event = {
        "payload": {
            "selected_route": "audit_only",
            "output_mode": "silent",
        }
    }

    assert _historical_expected_speak(spoke_event) is True
    assert _historical_expected_speak(silent_event) is False


def test_balanced_case_prefix_keeps_silent_and_spoke_cases() -> None:
    """The collected tuning prefix should include both historical labels."""

    cases = [
        _historical_case(case_id=f"silent-{index}", expected_speak=False)
        for index in range(14)
    ]
    cases.extend(
        _historical_case(case_id=f"spoke-{index}", expected_speak=True)
        for index in range(14)
    )

    balanced_cases = _balanced_case_prefix(cases)

    assert len(balanced_cases) == 20
    assert sum(
        1 for historical_case in balanced_cases
        if historical_case.historical_expected_speak
    ) == 10
    assert sum(
        1 for historical_case in balanced_cases
        if not historical_case.historical_expected_speak
    ) == 10


def _historical_case(
    *,
    case_id: str,
    expected_speak: bool,
) -> HistoricalSensitivityCase:
    """Build one lightweight historical case for helper tests."""

    historical_case = HistoricalSensitivityCase(
        case={"case_id": case_id},
        event={},
        historical_expected_speak=expected_speak,
        raw_window={},
    )
    return historical_case
