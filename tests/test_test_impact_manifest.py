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

    with pytest.raises(ImpactValidationError, match="no manifest entry"):
        resolve_impacted_test_nodes(
            manifest,
            ["src/kazusa_ai_chatbot/cognition_core_v3/new_owner.py"],
        )


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


def test_documented_impact_command_is_registered() -> None:
    """The project interpreter exposes the verifier command."""

    pyproject_text = (REPOSITORY_ROOT / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert 'validate-test-impact = "scripts.validate_test_impact:main"' in (
        pyproject_text
    )


def test_root_documentation_describes_impact_command() -> None:
    """Root testing guidance names the changed-source command."""

    readme_text = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert "scripts.validate_test_impact" in readme_text


def test_howto_documents_impact_command() -> None:
    """Operator testing guidance names exact impact enforcement."""

    howto_text = (REPOSITORY_ROOT / "docs" / "HOWTO.md").read_text(
        encoding="utf-8"
    )

    assert "scripts.validate_test_impact" in howto_text
    assert "--base-ref" in howto_text


def test_cognition_readme_documents_mirrored_unit_tree() -> None:
    """Cognition ownership guidance points to the canonical unit tree."""

    readme_text = (
        REPOSITORY_ROOT
        / "src"
        / "kazusa_ai_chatbot"
        / "cognition_core_v3"
        / "README.md"
    ).read_text(encoding="utf-8")

    assert "tests/unit/cognition_core_v3" in readme_text
    assert "source_test_impact_manifest.json" in readme_text
