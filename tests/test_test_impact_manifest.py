"""Deterministic tests for source-to-test ownership enforcement."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.validate_test_impact import (
    ImpactValidationError,
    load_manifest,
    missing_collected_nodes,
    resolve_impacted_test_nodes,
    validate_manifest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_covers_strict_cognition_source_boundary() -> None:
    """Every strict source module has an explicit ownership entry."""

    manifest = load_manifest(REPOSITORY_ROOT)

    assert validate_manifest(manifest, REPOSITORY_ROOT) == []


def test_manifest_covers_relevance_diagnostic_source_boundary() -> None:
    """Relevance producers and their canonical contracts have exact gates."""

    manifest = load_manifest(REPOSITORY_ROOT)
    entries = {entry["source"]: entry for entry in manifest["entries"]}
    required_rows = {
        "src/kazusa_ai_chatbot/relevance/contracts.py": (
            "tests/test_relevance_turn_settlement.py::"
            "test_relevance_exports_canonical_decision_types_without_producer_duplicates"
        ),
        "src/kazusa_ai_chatbot/relevance/__init__.py": (
            "tests/test_relevance_turn_settlement.py::"
            "test_relevance_evaluation_envelope_has_nested_decision_and_diagnostics_only"
        ),
        "src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py": (
            "tests/test_frontline_relevance_agent.py::"
            "test_frontline_provider_exhaustion_starts_authoritative_turn"
        ),
        "src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py": (
            "tests/test_persona_relevance_agent.py::"
            "test_non_authoritative_provider_exhaustion_returns_ignore"
        ),
    }

    for source, node in required_rows.items():
        assert source in entries
        assert node in entries[source]["required_unit_tests"]

    assert validate_manifest(manifest, REPOSITORY_ROOT) == []


def test_manifest_covers_turn_settlement_diagnostic_source_boundary() -> None:
    """Coordinator and reducer metadata carriers have exact gates."""

    manifest = load_manifest(REPOSITORY_ROOT)
    entries = {entry["source"]: entry for entry in manifest["entries"]}
    required_rows = {
        "src/kazusa_ai_chatbot/brain_service/turn_settlement.py": (
            "tests/test_relevance_turn_settlement.py::"
            "test_relevance_diagnostics_are_bounded_to_sixteen_in_occurrence_order"
        ),
        "src/kazusa_ai_chatbot/state.py": (
            "tests/unit/brain_service/test_cognition_graph_projection.py::"
            "test_attempt_diagnostics_reducer_concatenates_within_bound"
        ),
    }

    for source, node in required_rows.items():
        assert source in entries
        assert node in entries[source]["required_unit_tests"]

    assert validate_manifest(manifest, REPOSITORY_ROOT) == []


def test_manifest_accepts_an_explicit_package_init_source_root() -> None:
    """Release metadata may be owned by a package initializer explicitly."""

    manifest = load_manifest(REPOSITORY_ROOT)

    assert "src/control_console/__init__.py" in {
        entry["source"] for entry in manifest["entries"]
    }
    assert validate_manifest(manifest, REPOSITORY_ROOT) == []




def test_manifest_contains_group_topic_continuity_owner_rows() -> None:
    """Every changed continuity owner has an exact required gate."""

    manifest = load_manifest(REPOSITORY_ROOT)
    entries = {
        entry["source"]: entry
        for entry in manifest["entries"]
    }

    required_rows = {
        "src/kazusa_ai_chatbot/conversation_progress/history.py":
            "tests/test_conversation_progress_history_policy.py::test_group_scene_selection_preserves_current_user_anchors_before_recent_cap",
        "src/kazusa_ai_chatbot/conversation_progress/projection.py":
            "tests/test_conversation_progress_group_scene.py::test_group_scene_final_fit_keeps_protected_anchors_within_render_cap",
        "src/kazusa_ai_chatbot/internal_monologue_residue/loader.py":
            "tests/test_internal_monologue_residue_loader.py::test_noncanonical_rows_are_excluded_from_the_residue_window",
        "src/kazusa_ai_chatbot/db/schemas.py":
            "tests/test_internal_monologue_residue_database.py::test_v2_residue_schema_requires_disposition_operation_and_retention",
        "src/kazusa_ai_chatbot/event_logging/recording.py":
            "tests/test_event_logging_interface.py::test_continuity_boundary_payload_is_bounded_and_text_free",
        "src/kazusa_ai_chatbot/brain_service/post_turn.py":
            "tests/test_service_event_logging.py::test_progress_disposition_telemetry_is_trace_linked_and_sanitized",
    }

    for source, node in required_rows.items():
        assert source in entries
        assert node in entries[source]["required_unit_tests"]

    assert validate_manifest(manifest, REPOSITORY_ROOT) == []


def test_manifest_rejects_empty_unit_mapping() -> None:
    """A semantic source cannot be mapped without a deterministic unit node."""

    manifest = load_manifest(REPOSITORY_ROOT)
    invalid_manifest = deepcopy(manifest)
    invalid_manifest["entries"][0]["required_unit_tests"] = []

    errors = validate_manifest(invalid_manifest, REPOSITORY_ROOT)

    assert any("required_unit_tests must not be empty" in error for error in errors)


def test_unmapped_changed_source_fails_closed() -> None:
    """A new strict source module must be registered before execution."""

    manifest = load_manifest(REPOSITORY_ROOT)

    for changed_path in (
        "src/kazusa_ai_chatbot/cognition_core_v3/new_owner.py",
        "scripts/new_owner.py",
    ):
        with pytest.raises(ImpactValidationError, match="no manifest entry"):
            resolve_impacted_test_nodes(manifest, [changed_path])


def test_stale_required_node_fails_closed() -> None:
    """A required node absent from collection is a verification failure."""

    missing_nodes = missing_collected_nodes(
        ["tests/unit/cognition_core_v3/test_contracts.py::test_contract"],
        [],
    )

    assert missing_nodes == [
        "tests/unit/cognition_core_v3/test_contracts.py::test_contract"
    ]


def test_required_node_collection_failure_is_reported() -> None:
    """The collection comparison reports every omitted exact node."""

    missing_nodes = missing_collected_nodes(
        [
            "tests/unit/cognition_core_v3/test_contracts.py::test_contract",
            "tests/unit/cognition_resolver/test_state.py::test_state",
        ],
        ["tests/unit/cognition_core_v3/test_contracts.py::test_contract"],
    )

    assert missing_nodes == [
        "tests/unit/cognition_resolver/test_state.py::test_state"
    ]
