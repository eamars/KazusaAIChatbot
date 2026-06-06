"""Deterministic checks for frozen L2d routing case fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.l2d_action_selection_cases import (
    L2D_ROUTING_CASE_SET_SCHEMA_VERSION,
    L2D_ROUTING_CASE_SCHEMA_VERSION,
    compare_action_specs_to_expectations,
    load_l2d_routing_case_set,
    select_l2d_routing_case,
)


def test_compare_accepts_required_visible_speak_route() -> None:
    """A historical visible reply should accept a visible speak action."""

    case = _case(
        expectations={
            "required_action_kinds": ["speak"],
            "required_visibility_by_kind": {"speak": "user_visible"},
            "forbidden_action_kinds": ["send_message"],
        },
    )

    report = compare_action_specs_to_expectations(
        case,
        [_action_spec("speak", "user_visible")],
    )

    assert report["ok"] is True
    assert report["errors"] == []
    assert report["observed_kinds"] == ["speak"]
    assert report["observed_visibility_by_kind"] == {"speak": ["user_visible"]}


def test_compare_rejects_missing_required_route() -> None:
    """A case requiring a visible reply should fail when L2d emits no action."""

    case = _case(
        expectations={
            "required_action_kinds": ["speak"],
            "required_visibility_by_kind": {"speak": "user_visible"},
        },
    )

    report = compare_action_specs_to_expectations(case, [])

    assert report["ok"] is False
    assert report["errors"] == [
        "missing required action kind: speak",
        "missing required visibility for speak: user_visible",
    ]


def test_compare_rejects_forbidden_dispatch_route() -> None:
    """Ordinary chat fixtures must not route through external send_message."""

    case = _case(
        expectations={
            "required_action_kinds": ["speak"],
            "forbidden_action_kinds": ["send_message"],
        },
    )

    report = compare_action_specs_to_expectations(
        case,
        [
            _action_spec("speak", "user_visible"),
            _action_spec("send_message", "user_visible"),
        ],
    )

    assert report["ok"] is False
    assert report["errors"] == [
        "invalid action spec at index 1: kind: unsupported capability send_message",
        "forbidden action kind emitted: send_message",
    ]


def test_compare_requires_specific_action_params() -> None:
    """Lifecycle fixtures can require the router review kind."""

    case = _case(
        source_kind="self_cognition",
        expectations={
            "required_action_kinds": ["memory_lifecycle_update"],
            "required_visibility_by_kind": {
                "memory_lifecycle_update": "private",
            },
            "required_params_by_kind": {
                "memory_lifecycle_update": {
                    "review_kind": "active_commitment_lifecycle",
                }
            },
        },
    )

    report = compare_action_specs_to_expectations(
        case,
        [_memory_lifecycle_action("active_commitment_lifecycle")],
    )

    assert report["ok"] is True
    assert report["errors"] == []


def test_compare_accepts_background_artifact_request_params() -> None:
    """Background artifact fixtures can require bounded work-kind params."""

    case = _case(
        source_kind="background_artifact_poc",
        expectations={
            "required_action_kinds": ["background_artifact_request"],
            "required_visibility_by_kind": {
                "background_artifact_request": "private",
            },
            "required_params_by_kind": {
                "background_artifact_request": {
                    "work_kind": "coding_snippet",
                    "requested_delivery": "send_result_when_done",
                }
            },
            "forbidden_action_kinds": ["send_message", "trigger_future_cognition"],
        },
    )

    report = compare_action_specs_to_expectations(
        case,
        [_background_artifact_action("coding_snippet")],
    )

    assert report["ok"] is True
    assert report["errors"] == []


def test_compare_rejects_wrong_action_params() -> None:
    """Lifecycle fixtures should fail when the review kind differs."""

    case = _case(
        source_kind="self_cognition",
        expectations={
            "required_action_kinds": ["memory_lifecycle_update"],
            "required_params_by_kind": {
                "memory_lifecycle_update": {
                    "detail": "Review the active commitment lifecycle.",
                }
            },
        },
    )

    report = compare_action_specs_to_expectations(
        case,
        [_memory_lifecycle_action(
            "active_commitment_lifecycle",
            detail="Review a different lifecycle concern.",
        )],
    )

    assert report["ok"] is False
    assert report["errors"] == [
        (
            "missing required param for memory_lifecycle_update: "
            "detail=Review the active commitment lifecycle."
        )
    ]


def test_load_case_set_rejects_missing_frozen_state(tmp_path) -> None:
    """Fixture files should fail fast if they cannot drive L2d directly."""

    fixture_path = tmp_path / "cases.json"
    document = {
        "schema_version": L2D_ROUTING_CASE_SET_SCHEMA_VERSION,
        "cases": [
            {
                "schema_version": L2D_ROUTING_CASE_SCHEMA_VERSION,
                "case_id": "qq_001",
                "source_kind": "qq_history",
                "expectations": {},
                "historical_comparison": {},
            }
        ],
    }
    fixture_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen_l2d_state"):
        load_l2d_routing_case_set(fixture_path)


def test_select_case_returns_requested_fixture(tmp_path) -> None:
    """Live tests should be able to run one selected case at a time."""

    fixture_path = tmp_path / "cases.json"
    document = {
        "schema_version": L2D_ROUTING_CASE_SET_SCHEMA_VERSION,
        "cases": [
            _case(case_id="qq_001"),
            _case(case_id="self_001", source_kind="self_cognition"),
        ],
    }
    fixture_path.write_text(json.dumps(document), encoding="utf-8")

    case_set = load_l2d_routing_case_set(fixture_path)
    selected = select_l2d_routing_case(case_set, "self_001")

    assert selected["case_id"] == "self_001"
    assert selected["source_kind"] == "self_cognition"


def test_background_artifact_fixture_file_loads() -> None:
    """Production-style background artifact cases should drive live L2d runs."""

    case_set = load_l2d_routing_case_set(
        Path("tests/fixtures/l2d_background_artifact_cases.json"),
    )

    selected = select_l2d_routing_case(
        case_set,
        "coding_snippet_accept_fibonacci",
    )

    assert selected["source_kind"] == "background_artifact_poc"
    assert selected["expectations"]["required_action_kinds"] == [
        "speak",
        "background_artifact_request",
    ]


def _case(
    *,
    case_id: str = "qq_001",
    source_kind: str = "qq_history",
    expectations: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build one minimal frozen L2d case fixture."""

    if expectations is None:
        expectations = {}
    case = {
        "schema_version": L2D_ROUTING_CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "source_kind": source_kind,
        "frozen_l2d_state": {
            "final_l2": {
                "logical_stance": "CONFIRM",
                "character_intent": "PROVIDE",
            }
        },
        "historical_comparison": {
            "comparison_kind": "assistant_reply",
            "past_route": "visible_reply",
        },
        "expectations": expectations,
    }
    return case


def _action_spec(kind: str, visibility: str) -> dict[str, object]:
    """Build a minimal action spec for routing comparison tests."""

    target_owner = "l3_text"
    target_kind = "current_channel"
    params: dict[str, object] = {
        "delivery_mode": "visible_reply",
        "execute_at": None,
        "surface_requirements": {},
    }
    if kind == "send_message":
        target_owner = "dispatcher"
        params = {
            "target_channel": "current",
            "text": "message",
            "execute_at": None,
            "delivery_mentions": [],
        }

    action = {
        "schema_version": "action_spec.v1",
        "kind": kind,
        "cognition_mode": "deliberative",
        "source_refs": [
            {
                "schema_version": "action_source_ref.v1",
                "ref_kind": "cognitive_episode",
                "ref_id": "case-episode",
                "owner": "orchestrator",
                "relationship": "basis",
                "evidence_refs": [],
            }
        ],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": target_kind,
            "target_id": None,
            "owner": target_owner,
            "scope": {},
        },
        "params": params,
        "urgency": "now",
        "visibility": visibility,
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "none",
            "episode_type": None,
            "max_depth": 0,
            "include_result_as": None,
        },
        "reason": "routing test",
    }
    return action


def _background_artifact_action(work_kind: str) -> dict[str, object]:
    """Build a background artifact route action for comparison tests."""

    action = _action_spec("background_artifact_request", "private")
    action["target"] = {
        "schema_version": "action_target.v1",
        "target_kind": "current_user",
        "target_id": None,
        "owner": "background_artifact",
        "scope": {
            "source_platform": "debug",
            "source_channel_id": "debug:user:test-user",
            "source_channel_type": "private",
            "source_message_id": "message-001",
            "source_platform_bot_id": "debug-bot-001",
            "source_character_name": "Test Character",
            "requester_global_user_id": (
                "00000000-0000-4000-8000-000000000002"
            ),
            "requester_platform_user_id": "debug-user-001",
            "requester_display_name": "Test User",
        },
    }
    action["params"] = {
        "work_kind": work_kind,
        "objective": "Generate a Fibonacci function snippet.",
        "input_summary": "The user asked for a simple Fibonacci generator.",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
    }
    action["urgency"] = "background"
    action["reason"] = "The character accepted bounded async snippet work."
    return action


def _memory_lifecycle_action(
    review_kind: str,
    *,
    detail: str = "Review whether an active commitment lifecycle changed.",
) -> dict[str, object]:
    """Build a memory lifecycle route action for comparison tests."""

    action = _action_spec("speak", "private")
    action["kind"] = "memory_lifecycle_update"
    action["source_refs"] = [
        {
            "schema_version": "action_source_ref.v1",
            "ref_kind": "cognitive_episode",
            "ref_id": "case-episode",
            "owner": "orchestrator",
            "relationship": "basis",
            "evidence_refs": [],
        },
    ]
    action["target"] = {
        "schema_version": "action_target.v1",
        "target_kind": "cognitive_episode",
        "target_id": None,
        "owner": "memory_lifecycle_specialist",
        "scope": {"unit_type": "active_commitment"},
    }
    action["params"] = {
        "review_kind": review_kind,
        "detail": detail,
    }
    action["visibility"] = "private"
    action["urgency"] = "background"
    return action
