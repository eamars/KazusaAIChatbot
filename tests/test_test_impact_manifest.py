"""Deterministic tests for source-to-test ownership enforcement."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.validate_test_impact import (
    ImpactValidationError,
    load_manifest,
    manifest_test_nodes,
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


def test_manifest_contains_agentic_resolver_owner_rows() -> None:
    """Every standalone resolver and shared stream owner has an exact gate."""

    manifest = load_manifest(REPOSITORY_ROOT)
    entries = {
        entry["source"]: entry
        for entry in manifest["entries"]
    }
    required_rows = {
        "src/agentic_resolver/__init__.py":
            "tests/test_agentic_resolver_decommission.py::test_old_resolver_contracts_facades_and_aliases_are_absent",
        "src/agentic_resolver/contracts.py":
            "tests/test_agentic_resolver_contracts.py::test_public_contracts_expose_no_dsh_event_or_receipt_types",
        "src/agentic_resolver/runtime.py":
            "tests/test_agentic_resolver_runtime.py::test_resolve_preserves_standalone_entrypoint_and_returns_typed_exhaust",
        "src/kazusa_ai_chatbot/llm_interface/__init__.py":
            "tests/test_agentic_resolver_decommission.py::test_old_native_tool_stream_surface_is_absent",
        "src/kazusa_ai_chatbot/llm_interface/contracts.py":
            "tests/test_agentic_resolver_decommission.py::test_old_native_tool_stream_surface_is_absent",
        "src/kazusa_ai_chatbot/llm_interface/interface.py":
            "tests/test_agentic_resolver_decommission.py::test_old_native_tool_stream_surface_is_absent",
        "src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py":
            "tests/test_llm_interface_openai_provider.py::test_provider_maps_config_to_chat_model_constructor",
        "src/kazusa_ai_chatbot/llm_interface/reload.py":
            "tests/test_llm_interface_reload.py::test_async_unload_error_retries_same_call_once",
    }

    for source, node in required_rows.items():
        assert source in entries
        assert node in entries[source]["required_unit_tests"]

    assert validate_manifest(manifest, REPOSITORY_ROOT) == []


def test_manifest_contains_dsh_plan3_owner_rows() -> None:
    """Every Plan 3 owner node is represented by an exact manifest row."""

    plan_path = (
        REPOSITORY_ROOT
        / "development_plans"
        / "active"
        / "short_term"
        / "dsh_brain_bigbang_cutover_and_legacy_resolution_decommission_plan_2026-08-26.md"
    )
    plan_text = plan_path.read_text(encoding="utf-8")
    owner_paths = {
        "tests/test_accepted_task_lifecycle.py",
        "tests/test_agentic_resolver_sidecar_process.py",
        "tests/test_background_work_delivery.py",
        "tests/test_background_work_future_speak.py",
        "tests/test_cognition_llm_producer_matrix.py",
        "tests/test_cognition_resolver_loop.py",
        "tests/test_dsh_brain_interaction_resume.py",
        "tests/test_dsh_brain_interaction_service.py",
        "tests/test_dsh_plan3_documentation.py",
        "tests/test_dsh_plan3_e2e_live_llm.py",
        "tests/test_dsh_plan3_task_resolution.py",
        "tests/test_dsh_plan3_task_resolution_live_db.py",
        "tests/test_dsh_tool_gateway_contracts.py",
        "tests/test_dsh_tool_gateway_media.py",
        "tests/test_stage3_fresh_database_bootstrap.py",
        "tests/test_test_impact_manifest.py",
        "tests/unit/accepted_task/test_dsh_task_lifecycle.py",
        "tests/unit/action_spec/test_accepted_task_control.py",
        "tests/unit/agentic_resolver/test_runtime_task_lifecycle.py",
        "tests/unit/background_work/test_dsh_jobs.py",
        "tests/unit/background_work/test_dsh_worker.py",
        "tests/unit/background_work/test_result_source.py",
        "tests/unit/brain_service/test_dsh_task_readiness.py",
        "tests/unit/cognition_core_v3/test_dsh_task_handoff.py",
        "tests/unit/cognition_episode/test_task_result_source.py",
        "tests/unit/cognition_resolver/test_capabilities.py",
        "tests/unit/db/test_accepted_tasks.py",
        "tests/unit/db/test_task_resolution_sessions.py",
        "tests/unit/llm_interface/test_route_report.py",
        "tests/unit/nodes/test_persona_supervisor2_dsh_task_actions.py",
        "tests/unit/scripts/test_check_dsh_plan3_drain.py",
        "tests/unit/service/test_dsh_task_composition.py",
        "tests/unit/task_resolution/test_contracts.py",
        "tests/unit/task_resolution/test_decommission.py",
        "tests/unit/task_resolution/test_projection.py",
        "tests/unit/task_resolution/test_service.py",
        "tests/unit/test_config_dsh_cutover.py",
    }
    planned_nodes = {
        match.group(1)
        for match in re.finditer(r"`(tests/[^`]+::[^`]+)`", plan_text)
        if match.group(1).split("::", 1)[0] in owner_paths
    }
    superseded_nodes = {
        (
            "tests/test_dsh_plan3_e2e_live_llm.py::"
            "test_e2e_real_debug_user_replies_to_dsh_relay_and_resumes_same_session"
        ),
        (
            "tests/unit/background_work/test_dsh_jobs.py::"
            "test_job_claim_excludes_waiting_and_v1_payloads"
        ),
    }
    manifest = load_manifest(REPOSITORY_ROOT)

    missing = sorted(
        (set(planned_nodes) - superseded_nodes)
        - set(manifest_test_nodes(manifest, unit_only=False))
    )

    assert not missing, f"Plan 3 nodes lack manifest ownership rows: {missing}"


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


def test_removed_source_manifest_maps_deleted_sources_to_surviving_nodes() -> None:
    """Every planned deleted source resolves through explicit decommission rows."""

    manifest = load_manifest(REPOSITORY_ROOT)
    removed_sources = manifest["removed_sources"]
    assert isinstance(removed_sources, dict)
    assert len(removed_sources) == 119

    common_nodes = {
        (
            "tests/unit/task_resolution/test_decommission.py::"
            "test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent"
        ),
        (
            "tests/unit/task_resolution/test_decommission.py::"
            "test_runtime_import_graph_contains_no_legacy_executor_imports"
        ),
    }
    media_path = "src/kazusa_ai_chatbot/complex_task_resolver/subagent/media.py"
    media_nodes = set(resolve_impacted_test_nodes(manifest, [media_path]))
    assert media_nodes == common_nodes | {
        (
            "tests/test_dsh_tool_gateway_media.py::"
            "test_public_media_inspection_preserves_bounded_safe_fetch_and_visual_result"
        ),
        (
            "tests/test_dsh_tool_gateway_media.py::"
            "test_public_media_rejects_private_redirect_oversize_or_invalid_image_before_inspection"
        ),
    }
    assert set(resolve_impacted_test_nodes(
        manifest,
        ["scripts/run_coding_agent_benchmark.py"],
    )) == common_nodes


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
