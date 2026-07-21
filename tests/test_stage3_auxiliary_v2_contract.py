"""Focused tests for Stage 3 action availability and trace ownership."""

from __future__ import annotations

import importlib
import inspect

import pytest

from tests.cognition_core_v2_test_helpers import canonical_episode


EXPECTED_CAPABILITIES = {
    "memory_lifecycle_update",
    "apply_memory_lifecycle_update",
    "speak",
    "trigger_future_cognition",
    "future_speak",
    "accepted_task_request",
    "accepted_coding_task_request",
    "accepted_task_status_check",
    "background_work_request",
}


def test_action_registry_contains_the_complete_native_roster() -> None:
    """The declarative registry must own all nine supported capabilities."""

    registry_module = importlib.import_module(
        "kazusa_ai_chatbot.action_spec.registry",
    )

    capabilities = registry_module.build_initial_action_capabilities()

    assert set(capabilities) == EXPECTED_CAPABILITIES


def test_tool_result_affordances_exclude_new_delayed_task_creation() -> None:
    """Completed results must not expose delayed-task creation affordances."""

    registry_module = importlib.import_module(
        "kazusa_ai_chatbot.action_spec.registry",
    )
    capabilities = registry_module.build_initial_action_capabilities()
    snapshot = registry_module.build_runtime_capability_snapshot()
    affordances = registry_module.build_episode_affordances(
        capabilities,
        {"source_kind": "tool_result"},
        snapshot,
    )

    projected_capabilities = {
        row["capability_kind"]
        for row in affordances
    }
    assert projected_capabilities == {
        "accepted_task_status_check",
        "memory_lifecycle_update",
        "speak",
        "trigger_future_cognition",
    }


def test_trace_settlement_has_one_public_owner_signature() -> None:
    """Trace construction must move behind the post-turn public boundary."""

    post_turn_module = importlib.import_module(
        "kazusa_ai_chatbot.brain_service.post_turn",
    )
    settlement = post_turn_module.settle_episode_trace
    parameter_names = list(inspect.signature(settlement).parameters)

    assert parameter_names == [
        "episode",
        "cognition_output",
        "action_specs",
        "action_results",
        "surface_outputs",
        "terminal_status",
        "attempt_diagnostics",
        "delivery_correlation",
        "settled_at",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_status", "surface_outputs", "response_dialog"),
    [
        ("completed_visible", [], ["Visible result."]),
        ("completed_private", [], []),
        ("completed_action", [], []),
        ("scheduled", [], []),
        ("failed", [], []),
        ("cancelled", [], []),
    ],
)
async def test_runtime_settlement_has_one_trace_per_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
    surface_outputs: list[dict[str, object]],
    response_dialog: list[str],
) -> None:
    """Every attempted terminal outcome settles exactly one immutable trace."""

    post_turn_module = importlib.import_module(
        "kazusa_ai_chatbot.brain_service.post_turn",
    )
    original_settlement = post_turn_module.settle_episode_trace
    settlement_calls: list[str] = []

    def count_settlement(**kwargs: object) -> object:
        settlement_calls.append(str(kwargs["episode"]["episode_id"]))
        return original_settlement(**kwargs)

    monkeypatch.setattr(
        post_turn_module,
        "settle_episode_trace",
        count_settlement,
    )
    trace = await post_turn_module.settle_runtime_episode_trace(
        episode=canonical_episode(
            episode_id=f"stage3-cardinality:{terminal_status}",
        ),
        graph_result={
            "terminal_status": terminal_status,
            "surface_outputs": surface_outputs,
            "action_specs": [],
            "action_results": [],
        },
        response_dialog=response_dialog,
        delivery_tracking_id=(
            "delivery-stage3-cardinality"
            if response_dialog
            else ""
        ),
        settled_at="2026-07-19T00:00:01Z",
    )

    assert trace["schema_version"] == "episode_trace.v2"
    assert trace["terminal_status"] == terminal_status
    assert settlement_calls == [
        f"stage3-cardinality:{terminal_status}",
    ]


def test_post_turn_lifecycle_identity_is_stable_per_episode() -> None:
    """Retries reuse one idempotent lifecycle identity for an episode."""

    post_turn_module = importlib.import_module(
        "kazusa_ai_chatbot.brain_service.post_turn",
    )
    build_record = post_turn_module.build_post_turn_lifecycle_record
    first = build_record(
        source_episode_id="episode-cardinality-001",
        delivery_tracking_id="delivery-cardinality-001",
        action_specs=[],
        action_results=[],
        error_codes=[],
        created_at="2026-07-19T00:00:00Z",
    )
    second = build_record(
        source_episode_id="episode-cardinality-001",
        delivery_tracking_id="delivery-cardinality-001",
        action_specs=[],
        action_results=[],
        error_codes=[],
        created_at="2026-07-19T00:00:00Z",
    )

    assert first["lifecycle_record_id"] == second["lifecycle_record_id"]
    assert first["source_episode_id"] == second["source_episode_id"]
