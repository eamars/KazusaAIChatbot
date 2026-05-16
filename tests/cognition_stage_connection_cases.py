"""Local case-set helpers for cognition-stage connection live tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.action_spec.evaluator import ActionSpecEvaluator
from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY

CASE_SET_SCHEMA_VERSION = "cognition_stage_connection_case_set.v1"
CASE_SCHEMA_VERSION = "cognition_stage_connection_case.v1"
SUPPORTED_SOURCE_KINDS = frozenset(("qq_private", "qq_group"))


def load_cognition_stage_connection_case_set(path: Path) -> dict[str, Any]:
    """Load and validate one local cognition-stage case set.

    Args:
        path: JSON case-set artifact created by the capture script.

    Returns:
        Validated case-set document. Message text remains in local artifacts and
        is not echoed by this helper.
    """

    text = path.read_text(encoding="utf-8")
    document = json.loads(text)
    if not isinstance(document, dict):
        raise ValueError("case set must be a JSON object")
    schema_version = document.get("schema_version")
    if schema_version != CASE_SET_SCHEMA_VERSION:
        raise ValueError("case set schema_version is not supported")
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    for case in cases:
        _validate_case(case)
    return document


def select_cognition_stage_connection_case(
    case_set: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    """Return one case by id so live LLM tests run one case at a time."""

    cases = case_set["cases"]
    for case in cases:
        if case["case_id"] == case_id:
            return case
    raise ValueError(f"case_id not found: {case_id}")


def speak_action_selected(action_specs: list[dict[str, Any]]) -> bool:
    """Return whether the materialized action set contains a valid speak spec."""

    evaluator = ActionSpecEvaluator()
    for action_spec in action_specs:
        if action_spec.get("kind") != SPEAK_CAPABILITY:
            continue
        eval_result = evaluator.evaluate(action_spec)
        if eval_result["ok"]:
            return True
    return False


def build_cognition_connection_comparison_report(
    case: dict[str, Any],
    *,
    action_specs: list[dict[str, Any]],
    final_dialog: list[str],
    l3_ran: bool,
    l4_ran: bool,
) -> dict[str, Any]:
    """Compare live route shape against the historical assistant reply."""

    selected_speak = speak_action_selected(action_specs)
    expected_visible = bool(
        case["historical_comparison"]["expected_visible_surface"]
    )
    errors: list[str] = []
    if expected_visible and not selected_speak:
        errors.append("historical assistant reply expected a visible speak action")
    if selected_speak and not l3_ran:
        errors.append("speak was selected but selected L3 did not run")
    if selected_speak and not l4_ran:
        errors.append("speak was selected but L4 collector did not run")
    if selected_speak and not any(segment.strip() for segment in final_dialog):
        errors.append("speak was selected but dialog output was empty")
    if not selected_speak and final_dialog:
        errors.append("dialog produced text without a selected speak action")

    report = {
        "ok": not errors,
        "errors": errors,
        "expected_visible_surface": expected_visible,
        "selected_speak": selected_speak,
        "l3_ran": l3_ran,
        "l4_ran": l4_ran,
        "final_dialog_count": len(final_dialog),
    }
    return report


def _validate_case(value: object) -> None:
    """Validate the minimum case shape needed by live graph tests."""

    if not isinstance(value, dict):
        raise ValueError("case must be a JSON object")
    schema_version = value.get("schema_version")
    if schema_version != CASE_SCHEMA_VERSION:
        raise ValueError("case schema_version is not supported")
    case_id = value.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("case_id must be a non-empty string")
    source_kind = value.get("source_kind")
    if source_kind not in SUPPORTED_SOURCE_KINDS:
        raise ValueError("source_kind is not supported")
    seed_state = value.get("seed_state")
    if not isinstance(seed_state, dict):
        raise ValueError("seed_state must be a JSON object")
    historical_user_message = value.get("historical_user_message")
    if not isinstance(historical_user_message, str):
        raise ValueError("historical_user_message must be a string")
    historical_assistant_reply = value.get("historical_assistant_reply")
    if not isinstance(historical_assistant_reply, list):
        raise ValueError("historical_assistant_reply must be a list")
    historical_comparison = value.get("historical_comparison")
    if not isinstance(historical_comparison, dict):
        raise ValueError("historical_comparison must be a JSON object")
    expected_visible = historical_comparison.get("expected_visible_surface")
    if not isinstance(expected_visible, bool):
        raise ValueError("expected_visible_surface must be boolean")
