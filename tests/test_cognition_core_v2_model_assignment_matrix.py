"""Deterministic contracts for the Cognition Core V2 model matrix harness."""

from __future__ import annotations

import ast
import json
import sys
from copy import deepcopy
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.character_profile import (
    load_character_profile_seed,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.llm_interface import (
    LLMCallConfig,
    LLMThinkingConfig,
)
from tests.cognition_core_v2_test_helpers import (
    canonical_cognition_output,
    canonical_identity_context,
)
from tests.cognition_core_v2_model_assignment_matrix import (
    FACTOR_FIELDS,
    FACTOR_SERVICE_FIELDS,
    PROFILE_REPRESENTATIVE_FIELDS,
    aggregate_matrix_evidence,
    build_case_snapshots,
    build_ledger_contract,
    build_services_for_cell,
    build_source_manifest,
    enumerate_assignment_matrix,
    execute_sample,
    load_source_manifest,
    load_snapshot_index,
    main as matrix_main,
    route_profile_digest,
    verify_configured_model_profiles,
    verify_ledger_contract,
    verify_unblinding_contract,
)


CASE_COUNT = 8
REPETITION_COUNT = 3
MATRIX_CELL_COUNT = 16
SAMPLE_COUNT = CASE_COUNT * REPETITION_COUNT * MATRIX_CELL_COUNT
SOURCE_TIME = "2026-07-20T00:00:00Z"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_V2_SOURCE_ROOT = (
    PROJECT_ROOT
    / "src"
    / "kazusa_ai_chatbot"
    / "cognition_core_v2"
)
DATABASE_MODULE_PREFIXES = (
    "kazusa_ai_chatbot.db",
    "motor",
    "pymongo",
)


class _UnusedLLM:
    """Fail if a deterministic contract unexpectedly invokes an LLM."""

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: LLMCallConfig,
    ) -> object:
        """Reject unexpected model work."""

        del messages, config
        raise AssertionError("deterministic test invoked an LLM")


def _config(
    *,
    stage_name: str,
    route_name: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
) -> LLMCallConfig:
    """Build one explicit model config for substitution tests."""

    config = LLMCallConfig(
        stage_name=stage_name,
        route_name=route_name,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        top_p=0.7,
        top_k=20,
        max_completion_tokens=4096,
        presence_penalty=0.1,
        timeout_seconds=120.0,
        thinking=LLMThinkingConfig(enabled=False),
    )
    return config


def _services() -> CognitionCoreServicesV2:
    """Build thirteen stage configs grouped into four historical factors."""

    profile_d = {
        "base_url": "http://profile-d.invalid/v1",
        "api_key": "profile-d-secret",
        "model": "profile-d-model",
    }
    profile_m = {
        "base_url": "http://profile-m.invalid/v1",
        "api_key": "profile-m-secret",
        "model": "profile-m-model",
    }
    services = CognitionCoreServicesV2(
        llm=_UnusedLLM(),
        appraisal_event_agency_config=_config(
            stage_name="appraisal_event_agency",
            route_name="COGNITION_LLM_APPRAISAL_EVENT_AGENCY",
            temperature=0.11,
            **profile_d,
        ),
        appraisal_relationship_social_config=_config(
            stage_name="appraisal_relationship_social",
            route_name="COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL",
            temperature=0.12,
            **profile_m,
        ),
        appraisal_moral_identity_config=_config(
            stage_name="appraisal_moral_identity",
            route_name="COGNITION_LLM_APPRAISAL_MORAL_IDENTITY",
            temperature=0.13,
            **profile_m,
        ),
        appraisal_goal_threat_outcome_config=_config(
            stage_name="appraisal_goal_threat_outcome",
            route_name="COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME",
            temperature=0.14,
            **profile_m,
        ),
        appraisal_epistemic_comparison_memory_config=_config(
            stage_name="appraisal_epistemic_comparison_memory",
            route_name=(
                "COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY"
            ),
            temperature=0.15,
            **profile_m,
        ),
        appraisal_existential_drive_config=_config(
            stage_name="appraisal_existential_drive",
            route_name="COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE",
            temperature=0.16,
            **profile_m,
        ),
        goal_ordinary_response_config=_config(
            stage_name="goal_ordinary_response",
            route_name="COGNITION_LLM_GOAL_ORDINARY_RESPONSE",
            temperature=0.17,
            **profile_d,
        ),
        goal_active_branch_config=_config(
            stage_name="goal_active_branch",
            route_name="COGNITION_LLM_GOAL_ACTIVE_BRANCH",
            temperature=0.18,
            **profile_m,
        ),
        required_selection_verifier_config=_config(
            stage_name="required_selection_verifier",
            route_name="COGNITION_LLM_REQUIRED_SELECTION_VERIFIER",
            temperature=0.19,
            **profile_d,
        ),
        workspace_collapse_config=_config(
            stage_name="workspace_collapse",
            route_name="COGNITION_LLM_WORKSPACE_COLLAPSE",
            temperature=0.20,
            **profile_d,
        ),
        action_planning_config=_config(
            stage_name="action_planning",
            route_name="COGNITION_LLM_ACTION_PLANNING",
            temperature=0.21,
            **profile_d,
        ),
        action_authorization_config=_config(
            stage_name="action_authorization",
            route_name="COGNITION_LLM_ACTION_AUTHORIZATION",
            temperature=0.22,
            **profile_d,
        ),
        resolver_authorization_config=_config(
            stage_name="resolver_authorization",
            route_name="COGNITION_LLM_RESOLVER_AUTHORIZATION",
            temperature=0.23,
            **profile_d,
        ),
    )
    return services


def _manifest_rows() -> list[dict[str, str]]:
    """Build eight neutral source identities."""

    rows = []
    for index in range(CASE_COUNT):
        rows.append({
            "case_id": f"scenario_{index + 1:02d}",
            "platform": "debug",
            "platform_channel_id": "matrix-channel",
            "platform_message_id": f"message-{index + 1}",
            "scenario_dimension": f"dimension_{index + 1:02d}",
        })
    return rows


def _source_row(index: int) -> dict[str, Any]:
    """Build one persisted-message-shaped source row."""

    source_row = {
        "_id": f"row-{index}",
        "platform": "debug",
        "platform_channel_id": "matrix-channel",
        "channel_type": "private",
        "role": "user",
        "platform_message_id": f"message-{index}",
        "platform_user_id": f"platform-user-{index}",
        "global_user_id": f"global-user-{index}",
        "display_name": f"Participant {index}",
        "body_text": f"Neutral scenario input {index}.",
        "content_type": "text",
        "addressed_to_global_user_ids": ["active-character"],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
        "reply_context": {},
        "timestamp": SOURCE_TIME,
    }
    return source_row


def test_assignment_matrix_enumerates_all_sixteen_cells() -> None:
    """Enumerate every D/M assignment in fixed factor order."""

    matrix = enumerate_assignment_matrix()

    assert len(matrix) == MATRIX_CELL_COUNT
    assert [row["cell_id"] for row in matrix] == [
        f"Q{index:02d}"
        for index in range(MATRIX_CELL_COUNT)
    ]
    assert tuple(matrix[0]["assignment"]) == FACTOR_FIELDS
    assert list(matrix[0]["assignment"].values()) == ["D", "D", "D", "D"]
    assert list(matrix[-1]["assignment"].values()) == ["M", "M", "M", "M"]
    assert {
        tuple(row["assignment"].values())
        for row in matrix
    } == {
        tuple("M" if index & bit else "D" for bit in (8, 4, 2, 1))
        for index in range(MATRIX_CELL_COUNT)
    }


def test_ledger_contract_has_balanced_complete_sample_order() -> None:
    """Build 384 unique rows with all cells in every case/repetition block."""

    snapshots = [
        {
            "case_id": row["case_id"],
            "input_digest": f"digest-{row['case_id']}",
            "snapshot_path": f"snapshots/{row['case_id']}.json",
        }
        for row in _manifest_rows()
    ]

    ledger, unblinding_key = build_ledger_contract(
        snapshots,
        snapshot_set_digest="snapshot-set",
        route_profile_digest="route-profile",
        blind_seed="fixed-test-seed",
    )
    verification = verify_ledger_contract(
        ledger,
        snapshot_index={
            "snapshot_set_digest": "snapshot-set",
            "cases": snapshots,
        },
        route_profile_digest="route-profile",
        require_artifacts=False,
    )

    assert verification["sample_count"] == SAMPLE_COUNT
    assert verification["pending_count"] == SAMPLE_COUNT
    assert len(ledger["rows"]) == SAMPLE_COUNT
    assert len({
        row["sample_id"]
        for row in ledger["rows"]
    }) == SAMPLE_COUNT
    assert len({
        row["blind_label"]
        for row in ledger["rows"]
    }) == SAMPLE_COUNT
    assert len(unblinding_key["assignments"]) == SAMPLE_COUNT
    unblinding_verification = verify_unblinding_contract(
        unblinding_key,
        ledger=ledger,
    )
    assert unblinding_verification["assignment_count"] == SAMPLE_COUNT

    block_positions: dict[str, list[int]] = {
        cell["cell_id"]: []
        for cell in enumerate_assignment_matrix()
    }
    for block_index in range(CASE_COUNT * REPETITION_COUNT):
        start = block_index * MATRIX_CELL_COUNT
        block = ledger["rows"][start:start + MATRIX_CELL_COUNT]
        assert len({row["cell_id"] for row in block}) == MATRIX_CELL_COUNT
        for position, row in enumerate(block):
            block_positions[row["cell_id"]].append(position)
    assert all(
        min(positions) == 0 or max(positions) == MATRIX_CELL_COUNT - 1
        for positions in block_positions.values()
    )

    tampered_key = deepcopy(unblinding_key)
    tampered_key["assignments"][0]["assignment"][
        FACTOR_FIELDS[0]
    ] = "M"
    with pytest.raises(ValueError, match="assignment"):
        verify_unblinding_contract(tampered_key, ledger=ledger)

    tampered_ledger = deepcopy(ledger)
    tampered_ledger["rows"][0]["artifact_path"] = "../outside.json"
    with pytest.raises(ValueError, match="artifact path"):
        verify_ledger_contract(
            tampered_ledger,
            snapshot_index={
                "snapshot_set_digest": "snapshot-set",
                "cases": snapshots,
            },
            route_profile_digest="route-profile",
            require_artifacts=False,
        )


def test_model_substitution_changes_only_endpoint_identity() -> None:
    """Retain each factor's generation contract while switching its profile."""

    base_services = _services()
    matrix = enumerate_assignment_matrix()
    mixed_cell = matrix[10]
    substituted = build_services_for_cell(base_services, mixed_cell)
    config_fields = {
        field.name
        for field in fields(LLMCallConfig)
    }
    endpoint_fields = {"base_url", "api_key", "model"}

    for factor_field in FACTOR_FIELDS:
        selected_level = mixed_cell["assignment"][factor_field]
        selected_profile = getattr(
            base_services,
            PROFILE_REPRESENTATIVE_FIELDS[selected_level],
        )
        for service_field in FACTOR_SERVICE_FIELDS[factor_field]:
            original = getattr(base_services, service_field)
            updated = getattr(substituted, service_field)
            assert updated is not original
            assert updated.base_url == selected_profile.base_url
            assert updated.api_key == selected_profile.api_key
            assert updated.model == selected_profile.model
            for field_name in config_fields - endpoint_fields:
                assert (
                    getattr(updated, field_name)
                    == getattr(original, field_name)
                )

    modified_services = replace(
        base_services,
        goal_ordinary_response_config=replace(
            base_services.goal_ordinary_response_config,
            temperature=0.99,
        ),
    )
    assert (
        route_profile_digest(modified_services)
        != route_profile_digest(base_services)
    )


@pytest.mark.asyncio
async def test_model_preflight_rejects_a_divergent_stage_binding() -> None:
    """Require all stages assigned to one profile to share its identity."""

    base_services = _services()
    divergent_services = replace(
        base_services,
        workspace_collapse_config=replace(
            base_services.workspace_collapse_config,
            model="unexpected-third-model",
        ),
    )
    fetch_called = False

    async def fetch_models(
        base_url: str,
        api_key: str,
    ) -> dict[str, Any]:
        nonlocal fetch_called
        del base_url, api_key
        fetch_called = True
        return {"status": "available", "models": []}

    with pytest.raises(ValueError, match="D stage bindings"):
        await verify_configured_model_profiles(
            divergent_services,
            model_fetcher=fetch_models,
        )
    assert fetch_called is False


@pytest.mark.asyncio
async def test_snapshot_uses_only_read_helpers_and_freezes_inputs(
    tmp_path: Path,
) -> None:
    """Create eight snapshots through injected public reads and close the DB."""

    manifest_path = tmp_path / "source_cases.json"
    manifest = build_source_manifest(manifest_path, _manifest_rows())
    manifest["selection_date"] = "2026-07-20"
    manifest["source"] = "read-only discovery"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False),
        encoding="utf-8",
    )
    assert load_source_manifest(manifest_path)["case_count"] == CASE_COUNT
    _repository_root = Path(__file__).resolve().parents[1]
    profile = dict(load_character_profile_seed(
        _repository_root / "personalities" / "example.json",
    ))
    profile["cognition_state"] = build_character_production_state(
        updated_at=SOURCE_TIME,
    )
    calls: list[str] = []

    async def get_character_profile() -> dict[str, Any]:
        calls.append("get_character_profile")
        return deepcopy(profile)

    async def get_source_row(
        *,
        platform: str,
        platform_channel_id: str,
        platform_message_id: str,
    ) -> dict[str, Any]:
        calls.append("get_conversation_by_platform_message_id")
        assert platform == "debug"
        assert platform_channel_id == "matrix-channel"
        index = int(platform_message_id.rsplit("-", 1)[1])
        return _source_row(index)

    async def get_user_profile(global_user_id: str) -> dict[str, Any]:
        calls.append("get_user_profile")
        index = int(global_user_id.rsplit("-", 1)[1])
        return {
            "global_user_id": global_user_id,
            "cognition_state": build_acquaintance_user_state(
                global_user_id=global_user_id,
                updated_at=SOURCE_TIME,
            ),
            "platform_accounts": [{
                "platform": "debug",
                "platform_user_id": f"platform-user-{index}",
                "display_name": f"Participant {index}",
            }],
            "suspected_aliases": [],
        }

    async def get_history(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append("get_conversation_history")
        assert kwargs["sort_direction"] == -1
        return [{
            **_source_row(99),
            "platform_message_id": "earlier-message",
            "timestamp": "2026-07-19T23:59:00+00:00",
        }]

    async def close_db() -> None:
        calls.append("close_db")

    result = await build_case_snapshots(
        manifest_path,
        artifact_root=tmp_path / "artifacts",
        get_character_profile_func=get_character_profile,
        get_user_profile_func=get_user_profile,
        get_source_message_func=get_source_row,
        get_conversation_history_func=get_history,
        close_db_func=close_db,
    )

    assert result["case_count"] == CASE_COUNT
    assert len(result["cases"]) == CASE_COUNT
    assert calls.count("get_character_profile") == 1
    assert calls.count("get_conversation_by_platform_message_id") == CASE_COUNT
    assert calls.count("get_user_profile") == CASE_COUNT
    assert calls.count("get_conversation_history") == CASE_COUNT
    assert calls[-1] == "close_db"
    assert (tmp_path / "artifacts" / "snapshots" / "index.json").exists()
    assert (
        tmp_path
        / "artifacts"
        / "tracked_forbidden_tokens.txt"
    ).exists()
    loaded_index = load_snapshot_index(
        artifact_root=tmp_path / "artifacts",
    )
    assert loaded_index == result
    input_digests = {
        row["input_digest"]
        for row in result["cases"]
    }
    assert len(input_digests) == CASE_COUNT

    snapshot_path = (
        tmp_path
        / "artifacts"
        / result["cases"][0]["snapshot_path"]
    )
    tampered_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    tampered_snapshot["review_projection"]["current_input"] = "tampered"
    snapshot_path.write_text(
        json.dumps(tampered_snapshot, ensure_ascii=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="snapshot digest"):
        load_snapshot_index(artifact_root=tmp_path / "artifacts")


@pytest.mark.asyncio
async def test_sample_execution_is_db_free_blinded_and_input_immutable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Execute from one frozen snapshot without DB access or identity leakage."""

    owner_id = "global-user-1"
    mutable_state = build_acquaintance_user_state(
        global_user_id=owner_id,
        updated_at=SOURCE_TIME,
    )
    character_state = build_character_production_state(
        updated_at=SOURCE_TIME,
    )
    snapshot_input = {
        "schema_version": "cognition_core_input.v2",
        "episode": {
            "schema_version": "cognitive_episode.v1",
            "episode_id": "sample-episode",
            "trigger_source": "user_message",
            "origin_metadata": {
                "schema_version": "user_message_origin.v1",
                "owner": "test",
                "privacy_scope": "private",
                "delivery_permission_ref": "",
                "created_at": SOURCE_TIME,
            },
            "target_scope": {
                "platform": "debug",
                "platform_channel_id": "channel",
                "channel_type": "private",
                "current_platform_user_id": "platform-user-1",
                "current_global_user_id": owner_id,
                "current_display_name": "Participant",
                "target_addressed_user_ids": [],
                "target_broadcast": False,
            },
            "percepts": [{
                "schema_version": "percept.v1",
                "percept_kind": "dialog",
                "source_kind": "dialog",
                "source_id": "message-1",
                "content": {"semantic_text": "Test input."},
                "observed_at": SOURCE_TIME,
            }],
            "evidence_refs": [],
            "created_at": SOURCE_TIME,
            "privacy_scope": "private",
            "continuation_depth": 0,
        },
        "state_scope": "user",
        "mutable_state": mutable_state,
        "character_constraints": {
            "drives": character_state["drives"],
            "standards": character_state["standards"],
            "meaning_state": character_state["meaning_state"],
            "personality_judgment": {
                "logic": "grounded",
                "defense": "bounded",
                "quirks": "concise",
                "taboos": "preserve roles",
            },
        },
        "character_identity_context": canonical_identity_context(),
        "evidence": [],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "resolver_context": "",
        "scene_context": {
            "channel_scope": "private",
            "character_role": "active character",
            "current_user_role": "current user",
            "semantic_scene": "Test input.",
            "conversation_continuity": "",
            "semantic_temporal_context": "immediate",
        },
        "private_continuity_context": "",
    }
    snapshot = {
        "case_id": "scenario_01",
        "input_digest": "",
        "input": snapshot_input,
        "review_projection": {"current_input": "Test input."},
    }
    original_snapshot = deepcopy(snapshot)
    db_helpers = (
        "get_character_profile",
        "get_user_profile",
        "get_conversation_by_platform_message_id",
        "get_conversation_history",
    )

    async def forbidden_db_access(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("matrix execution accessed MongoDB")

    for helper_name in db_helpers:
        monkeypatch.setattr(
            "tests.cognition_core_v2_model_assignment_matrix."
            f"{helper_name}",
            forbidden_db_access,
        )

    async def run_cognition(
        payload: dict[str, Any],
        services: CognitionCoreServicesV2,
    ) -> dict[str, Any]:
        del services
        payload["scene_context"]["semantic_scene"] = "mutated working copy"
        output = canonical_cognition_output(owner_user_id=owner_id)
        return output

    sample_row = {
        "sample_id": "sample_0001",
        "blind_label": "B000000000001",
        "case_id": "scenario_01",
        "repetition": 1,
        "cell_id": "Q15",
        "input_digest": "",
        "artifact_path": "runs/B000000000001.json",
    }
    artifact = await execute_sample(
        sample_row,
        snapshot,
        _services(),
        assignment=enumerate_assignment_matrix()[-1],
        assignment_digest="opaque-assignment-digest",
        artifact_root=tmp_path,
        run_cognition_func=run_cognition,
    )

    assert snapshot == original_snapshot
    assert artifact["terminal_status"] == "completed"
    artifact_text = json.dumps(artifact, ensure_ascii=False)
    for forbidden_text in (
        "profile-d-model",
        "profile-m-model",
        "profile-d.invalid",
        "profile-m.invalid",
        "profile-d-secret",
        "profile-m-secret",
        "Q15",
    ):
        assert forbidden_text not in artifact_text


def test_cognition_core_v2_runtime_has_no_database_dependencies() -> None:
    """Keep the evaluated runtime package outside every database boundary."""

    violations: list[str] = []
    for path in sorted(CORE_V2_SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(
            path.read_text(encoding="utf-8"),
            filename=str(path),
        )
        for node in ast.walk(tree):
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported_modules = [node.module or ""]
            for module_name in imported_modules:
                if module_name.startswith(DATABASE_MODULE_PREFIXES):
                    relative_path = path.relative_to(PROJECT_ROOT)
                    violations.append(
                        f"{relative_path}:{node.lineno} imports "
                        f"{module_name}"
                    )

    assert violations == []


def test_cli_failure_output_withholds_untrusted_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep command failures useful without exposing exception-held secrets."""

    secret_value = "credential-that-must-stay-private"

    async def fail_dispatch(args: object) -> dict[str, Any]:
        del args
        raise RuntimeError(f"provider rejected {secret_value}")

    monkeypatch.setattr(
        "tests.cognition_core_v2_model_assignment_matrix._dispatch_command",
        fail_dispatch,
    )
    monkeypatch.setattr(sys, "argv", ["matrix-harness", "status"])

    assert matrix_main() == 1
    captured = capsys.readouterr()
    failure = json.loads(captured.err)
    assert failure == {
        "error": (
            "command failed; exception details withheld by the "
            "credential-safe CLI boundary"
        ),
        "error_class": "RuntimeError",
        "status": "failed",
    }
    assert secret_value not in captured.err


def test_aggregate_selects_highest_qualified_level_m_call_share() -> None:
    """Choose the full-M cell when every assignment has baseline quality."""

    snapshots = [
        {
            "case_id": row["case_id"],
            "input_digest": f"digest-{row['case_id']}",
            "snapshot_path": f"snapshots/{row['case_id']}.json",
        }
        for row in _manifest_rows()
    ]
    ledger, _ = build_ledger_contract(
        snapshots,
        snapshot_set_digest="snapshot-set",
        route_profile_digest="route-profile",
        blind_seed="aggregate-seed",
    )
    artifacts: dict[str, dict[str, Any]] = {}
    reviews: dict[str, dict[str, Any]] = {}
    for row in ledger["rows"]:
        artifacts[row["blind_label"]] = {
            "terminal_status": "completed",
            "factor_call_counts": {
                factor_field: 1
                for factor_field in FACTOR_FIELDS
            },
        }
        reviews[row["blind_label"]] = {
            "scores": {
                "input_responsiveness": 4,
                "character_state_consistency": 4,
                "situational_suitability": 4,
                "role_evidence_grounding": 4,
                "cross_stage_coherence": 4,
            },
            "baseline_relative_verdict": "equivalent",
            "rationale": "The reaction remains equally suitable.",
        }

    result = aggregate_matrix_evidence(
        ledger,
        artifacts=artifacts,
        reviews=reviews,
    )

    assert len(result["cells"]) == MATRIX_CELL_COUNT
    assert all(cell["quality_qualified"] for cell in result["cells"])
    assert result["selected_cell_id"] == "Q15"
    assert result["selected_assignment"] == {
        factor_field: "M"
        for factor_field in FACTOR_FIELDS
    }
