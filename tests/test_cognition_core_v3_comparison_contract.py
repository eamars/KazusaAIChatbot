"""Static and patched contracts for the cognition comparison harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.state_reducers import (
    _matching_event,
    canonical_event_entity_id,
)
from kazusa_ai_chatbot.llm_interface.contracts import LLMCallConfig
from tests import cognition_core_v3_comparison_harness as harness


def _incoming_event_for_source(
    state: dict[str, Any],
    evidence_ref: dict[str, Any],
) -> dict[str, Any]:
    """Return the minimum event identity surface consumed by matching."""

    return {
        "entity_id": canonical_event_entity_id(state, evidence_ref),
        "role_refs": [
            {
                "role": "actor",
                "entity_kind": "relationship",
                "entity_id": "relationship:user:user-1",
            }
        ],
        "status": "active",
    }


def test_custom_event_id_and_empty_roles_do_not_match_canonical_candidate(
) -> None:
    """Freeze the duplicate-event identity condition found in Gate 1."""

    payload = harness.render_case_input(
        harness.find_case_row("verbal_abuse_boundary")
    )
    state = payload["mutable_state"]
    evidence_ref = payload["evidence"][0]["evidence_ref"]
    incoming = _incoming_event_for_source(state, evidence_ref)

    assert state["active_events"][0]["entity_id"] != incoming["entity_id"]
    assert state["active_events"][0]["role_refs"] == []
    assert _matching_event(state, incoming) is None


def test_canonical_event_identity_matches_same_source_not_distinct_source(
) -> None:
    """Prove canonical identity distinguishes same and different evidence."""

    payload = harness.render_case_input(
        harness.find_case_row("verbal_abuse_boundary")
    )
    state = payload["mutable_state"]
    evidence_ref = payload["evidence"][0]["evidence_ref"]
    incoming = _incoming_event_for_source(state, evidence_ref)
    stored = state["active_events"][0]
    stored["entity_id"] = incoming["entity_id"]

    assert _matching_event(state, incoming) is stored

    distinct_ref = {
        **evidence_ref,
        "source_id": "diagnostic:distinct-expired-transit-pass",
    }
    distinct = _incoming_event_for_source(state, distinct_ref)
    assert distinct["entity_id"] != incoming["entity_id"]
    assert _matching_event(state, distinct) is None


class _UnusedLLM:
    """Fail if a patched runner unexpectedly invokes a model."""

    async def ainvoke(self, messages: object, *, config: object) -> object:
        del messages, config
        raise AssertionError("patched runner must not invoke the LLM")

    def invoke(self, messages: object, *, config: object) -> object:
        del messages, config
        raise AssertionError("patched runner must not invoke the LLM")


def _services() -> CognitionCoreServicesV2:
    """Return a complete inert V2 service bundle."""

    config = LLMCallConfig(
        stage_name="patched_stage",
        route_name="PATCHED_ROUTE",
        base_url="http://127.0.0.1:9/v1",
        api_key="test-only-key",
        model="patched-model",
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        max_completion_tokens=64,
        presence_penalty=None,
    )
    return CognitionCoreServicesV2(
        llm=_UnusedLLM(),
        appraisal_event_agency_config=config,
        appraisal_relationship_social_config=config,
        appraisal_moral_identity_config=config,
        appraisal_goal_threat_outcome_config=config,
        appraisal_epistemic_comparison_memory_config=config,
        appraisal_existential_drive_config=config,
        goal_ordinary_response_config=config,
        goal_active_branch_config=config,
        workspace_collapse_config=config,
        action_planning_config=config,
        action_authorization_config=config,
        resolver_authorization_config=config,
    )


def _add_baseline_artifact(
    baseline_root: Path,
    records: list[dict[str, Any]],
    *,
    relative_path: str,
    kind: str,
    content: bytes,
) -> Path:
    """Write one protected artifact and append its immutable index row.

    Args:
        baseline_root: Temporary root representing one sealed baseline.
        records: Mutable artifact-index rows owned by the test fixture.
        relative_path: POSIX path below ``baseline_root``.
        kind: Closed artifact classification consumed by the validator.
        content: Exact bytes to seal and hash.

    Returns:
        The path containing the written artifact bytes.
    """

    artifact_path = baseline_root / Path(relative_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(content)
    records.append({
        "path": relative_path,
        "kind": kind,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    })
    return artifact_path


@pytest.mark.asyncio
async def test_comparison_harness_is_effect_free_and_hashes_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A direct-facade run hashes an isolated input and declares zero effects."""

    row = harness.find_case_row("ordinary_neutral_response")
    original_row = json.loads(json.dumps(row))
    seen_payloads: list[dict[str, Any]] = []

    async def patched_runner(
        input_payload: dict[str, Any],
        services: CognitionCoreServicesV2,
    ) -> dict[str, Any]:
        assert services.llm.__class__.__name__ == "CapturingLLMInvoker"
        seen_payloads.append(input_payload)
        return {"schema_version": "patched_output.v1", "accepted": True}

    monkeypatch.setattr(
        harness,
        "validate_cognition_core_output",
        lambda output: output,
    )
    identity = harness.TrialIdentity(
        baseline_id="baseline-contract",
        case_id=row["case_id"],
        engine="v2",
        trial_index=1,
    )
    artifact = await harness.run_effect_free_trial(
        row,
        identity=identity,
        services=_services(),
        runner=patched_runner,
        artifact_root=tmp_path,
    )

    expected_input = row["canonical_input"]
    assert artifact["input_sha256"] == harness.canonical_sha256(expected_input)
    assert artifact["output_sha256"] == harness.canonical_sha256(
        {"schema_version": "patched_output.v1", "accepted": True}
    )
    assert artifact["canonical_input"] == expected_input
    assert seen_payloads[0] is not expected_input
    assert row == original_row
    assert artifact["validator_result"] == {
        "input": "passed",
        "output": "passed",
        "input_unchanged": True,
    }
    assert artifact["effect_free_contract"] == {
        "direct_engine_facade_only": True,
        "state_commit": False,
        "action_execution": False,
        "resolver_execution": False,
        "surface_delivery": False,
        "database_semantic_write": False,
        "adapter_delivery": False,
        "scheduler_effect": False,
    }
    sealed_path = Path(artifact["artifact_path"])
    assert sealed_path.is_file()
    assert json.loads(sealed_path.read_text(encoding="utf-8"))[
        "input_sha256"
    ] == artifact["input_sha256"]


@pytest.mark.asyncio
async def test_comparison_harness_forbids_outcome_conditioned_reruns_and_seals_all_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible results remain immutable and invalid reruns stay closed."""

    row = harness.find_case_row("ordinary_neutral_response")

    async def patched_runner(
        input_payload: dict[str, Any],
        services: CognitionCoreServicesV2,
    ) -> dict[str, Any]:
        del input_payload, services
        return {"schema_version": "patched_output.v1", "accepted": False}

    monkeypatch.setattr(
        harness,
        "validate_cognition_core_output",
        lambda output: output,
    )
    identity = harness.TrialIdentity(
        baseline_id="baseline-retention",
        case_id=row["case_id"],
        engine="v2",
        trial_index=2,
    )
    artifact = await harness.run_effect_free_trial(
        row,
        identity=identity,
        services=_services(),
        runner=patched_runner,
        artifact_root=tmp_path,
    )
    assert artifact["disposition"] == harness.ELIGIBLE_RESULT
    assert artifact["semantic_result_available"] is True

    with pytest.raises(harness.TrialAlreadySealedError):
        await harness.run_effect_free_trial(
            row,
            identity=identity,
            services=_services(),
            runner=patched_runner,
            artifact_root=tmp_path,
        )
    with pytest.raises(harness.TrialAlreadySealedError):
        harness.assert_rerun_allowed(
            [artifact],
            reason="provider_transport_no_result",
        )
    with pytest.raises(ValueError):
        harness.assert_rerun_allowed(
            [{
                "disposition": harness.INVALID_NO_RESULT,
                "semantic_result_available": False,
            }],
            reason="semantic_quality_failure",
        )

    harness.assert_rerun_allowed(
        [{
            "disposition": harness.INVALID_NO_RESULT,
            "semantic_result_available": False,
        }],
        reason="harness_invalid_no_result",
    )
    paths = {
        harness.trial_artifact_path(
            tmp_path,
            harness.TrialIdentity(
                baseline_id="baseline-retention",
                case_id=row["case_id"],
                engine="v2",
                trial_index=trial_index,
            ),
        )
        for trial_index in (1, 2, 3)
    }
    assert len(paths) == 3


@pytest.mark.asyncio
async def test_harness_allows_only_attested_matched_pair_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A harness-invalid no-result attempt needs a hashed pair invalidation."""

    original_row = harness.find_case_row("ordinary_neutral_response")
    corrected_row = json.loads(json.dumps(original_row))
    corrected_row["canonical_input"]["resolver_cycle_index"] = 0
    call_count = 0

    async def patched_runner(
        input_payload: dict[str, Any],
        services: CognitionCoreServicesV2,
    ) -> dict[str, Any]:
        nonlocal call_count
        del input_payload, services
        call_count += 1
        if call_count < 3:
            raise RuntimeError("synthetic harness fixture mismatch")
        return {"schema_version": "patched_output.v1", "accepted": True}

    monkeypatch.setattr(
        harness,
        "validate_cognition_core_output",
        lambda output: output,
    )
    first_identity = harness.TrialIdentity(
        baseline_id="baseline-pair-invalidation",
        case_id=original_row["case_id"],
        engine="v2",
        trial_index=1,
    )
    with pytest.raises(harness.TrialExecutionError):
        await harness.run_effect_free_trial(
            original_row,
            identity=first_identity,
            services=_services(),
            runner=patched_runner,
            artifact_root=tmp_path,
        )

    prior_path = harness.trial_artifact_path(tmp_path, first_identity)
    prior_artifact = json.loads(prior_path.read_text(encoding="utf-8"))
    corrected_sha256 = harness.canonical_sha256(
        harness.render_case_input(corrected_row)
    )
    invalidation = {
        "schema_version": harness.PAIR_INVALIDATION_SCHEMA,
        "baseline_id": first_identity.baseline_id,
        "case_id": first_identity.case_id,
        "trial_index": first_identity.trial_index,
        "reason": "harness_invalid_no_result",
        "invalidated_input_sha256": prior_artifact["input_sha256"],
        "corrected_input_sha256": corrected_sha256,
        "replacement_attempt_index": 2,
        "pair_members": [
            {
                "engine": "v2",
                "attempt_index": 1,
                "status": "retained_invalid_artifact",
                "artifact_path": prior_path.relative_to(
                    tmp_path / first_identity.baseline_id
                ).as_posix(),
                "artifact_sha256": hashlib.sha256(
                    prior_path.read_bytes()
                ).hexdigest(),
                "original_disposition": harness.HARD_BOUNDARY_FAILURE,
            },
            {
                "engine": "v3",
                "attempt_index": 1,
                "status": "invalidated_before_execution",
                "artifact_path": None,
                "artifact_sha256": None,
                "original_disposition": None,
            },
        ],
    }
    replacement_identity = harness.TrialIdentity(
        baseline_id=first_identity.baseline_id,
        case_id=first_identity.case_id,
        engine="v2",
        trial_index=first_identity.trial_index,
        attempt_index=2,
    )
    with pytest.raises(harness.TrialExecutionError):
        await harness.run_effect_free_trial(
            corrected_row,
            identity=replacement_identity,
            services=_services(),
            runner=patched_runner,
            artifact_root=tmp_path,
            rerun_invalidation=invalidation,
        )

    replacement_path = harness.trial_artifact_path(
        tmp_path,
        replacement_identity,
    )
    replacement_artifact = json.loads(
        replacement_path.read_text(encoding="utf-8")
    )
    final_row = json.loads(json.dumps(corrected_row))
    final_row["canonical_input"]["private_continuity_context"] = (
        "corrected fixture"
    )
    final_sha256 = harness.canonical_sha256(
        harness.render_case_input(final_row)
    )
    final_invalidation = {
        **invalidation,
        "invalidated_input_sha256": replacement_artifact["input_sha256"],
        "corrected_input_sha256": final_sha256,
        "replacement_attempt_index": 3,
        "pair_members": [
            {
                "engine": "v2",
                "attempt_index": 2,
                "status": "retained_invalid_artifact",
                "artifact_path": replacement_path.relative_to(
                    tmp_path / first_identity.baseline_id
                ).as_posix(),
                "artifact_sha256": hashlib.sha256(
                    replacement_path.read_bytes()
                ).hexdigest(),
                "original_disposition": harness.HARD_BOUNDARY_FAILURE,
            },
            {
                "engine": "v3",
                "attempt_index": 2,
                "status": "invalidated_before_execution",
                "artifact_path": None,
                "artifact_sha256": None,
                "original_disposition": None,
            },
        ],
    }
    final_identity = harness.TrialIdentity(
        baseline_id=first_identity.baseline_id,
        case_id=first_identity.case_id,
        engine="v2",
        trial_index=first_identity.trial_index,
        attempt_index=3,
    )
    artifact = await harness.run_effect_free_trial(
        final_row,
        identity=final_identity,
        services=_services(),
        runner=patched_runner,
        artifact_root=tmp_path,
        rerun_invalidation=final_invalidation,
    )

    assert artifact["attempt_index"] == 3
    assert artifact["disposition"] == harness.ELIGIBLE_RESULT
    assert call_count == 3


def test_baseline_index_validator_rejects_missing_or_changed_artifacts(
    tmp_path: Path,
) -> None:
    """The sealed ledger covers every governed and protected artifact byte."""

    baseline_id = "cogv3-g1-test-baseline"
    repository_root = tmp_path / "repository"
    baseline_root = tmp_path / baseline_id
    repository_root.mkdir()
    baseline_root.mkdir()

    canonical_inputs = {
        f"case-{case_index:02d}": {"case_index": case_index}
        for case_index in range(1, 25)
    }
    live_manifest = {
        "cases": [
            {
                "case_id": case_id,
                "canonical_input": canonical_input,
            }
            for case_id, canonical_input in canonical_inputs.items()
        ]
    }
    owned_paths = {
        "delete": [],
        "create": [
            f"create/path-{path_index:02d}.py"
            for path_index in range(1, 33)
        ],
        "modify": [
            f"modify/path-{path_index:02d}.py"
            for path_index in range(1, 85)
        ],
    }
    architecture_manifest = {"owned_paths": owned_paths}
    governed_sources = [
        (
            "approved_execution_contract_at_gate_1_entry",
            "governance/plan.md",
            b"approved entry contract",
            b"current plan with execution evidence",
            False,
        ),
        (
            "governing_architecture",
            "governance/architecture.md",
            b"governing architecture",
            b"governing architecture",
            True,
        ),
        (
            "architecture_manifest",
            "tests/architecture_manifest.json",
            harness.canonical_json_bytes(architecture_manifest),
            harness.canonical_json_bytes(architecture_manifest),
            True,
        ),
        (
            "live_case_manifest",
            "tests/live_case_manifest.json",
            harness.canonical_json_bytes(live_manifest),
            harness.canonical_json_bytes(live_manifest),
            True,
        ),
        (
            "token_calibration_corpus",
            "tests/token_corpus.json",
            b"{}",
            b"{}",
            True,
        ),
        (
            "comparison_harness",
            "tests/comparison_harness.py",
            b"comparison harness",
            b"comparison harness",
            True,
        ),
        (
            "comparison_contract_test",
            "tests/comparison_contract.py",
            b"comparison contract",
            b"comparison contract",
            True,
        ),
        (
            "live_case_test",
            "tests/live_case.py",
            b"live case contract",
            b"live case contract",
            True,
        ),
        (
            "manifest_contract_test",
            "tests/manifest_contract.py",
            b"manifest contract",
            b"manifest contract",
            True,
        ),
    ]
    governed_files: list[dict[str, Any]] = []
    for role, relative_path, sealed_bytes, current_bytes, verify_current in (
        governed_sources
    ):
        governed_path = repository_root / Path(relative_path)
        governed_path.parent.mkdir(parents=True, exist_ok=True)
        governed_path.write_bytes(current_bytes)
        governed_files.append({
            "path": relative_path,
            "sha256": hashlib.sha256(sealed_bytes).hexdigest(),
            "role": role,
            "verify_current": verify_current,
        })

    artifact_records: list[dict[str, Any]] = []
    fingerprint_payload = {
        "schema_version": "architecture_path_fingerprints.v1",
        "path_count": 116,
        "records": [
            *[
                {"path": path, "disposition": "create"}
                for path in owned_paths["create"]
            ],
            *[
                {"path": path, "disposition": "modify"}
                for path in owned_paths["modify"]
            ],
        ],
    }
    fingerprint_path = "architecture_path_fingerprints.json"
    _add_baseline_artifact(
        baseline_root,
        artifact_records,
        relative_path=fingerprint_path,
        kind="baseline_governance",
        content=harness.canonical_json_bytes(fingerprint_payload),
    )

    eligible_paths: list[Path] = []
    for trial_number in range(1, 73):
        trial_payload = {
            "trial_id": f"{baseline_id}:v2:trial-{trial_number}",
            "disposition": harness.ELIGIBLE_RESULT,
            "semantic_result_available": True,
        }
        eligible_path = _add_baseline_artifact(
            baseline_root,
            artifact_records,
            relative_path=(
                f"raw_trials/v2/eligible-trial-{trial_number:02d}.json"
            ),
            kind="eligible_raw_trial",
            content=harness.canonical_json_bytes(trial_payload),
        )
        eligible_paths.append(eligible_path)

    for attempt_number in range(1, 3):
        _add_baseline_artifact(
            baseline_root,
            artifact_records,
            relative_path=(
                f"raw_trials/v2/invalid-attempt-{attempt_number}.json"
            ),
            kind="invalidated_raw_attempt",
            content=harness.canonical_json_bytes({
                "semantic_result_available": False,
            }),
        )

    for review_number in range(1, 73):
        _add_baseline_artifact(
            baseline_root,
            artifact_records,
            relative_path=f"reviews/v2/review-{review_number:02d}.md",
            kind="v2_review",
            content=f"review {review_number}\n".encode(),
        )

    for invalidation_number in range(1, 3):
        _add_baseline_artifact(
            baseline_root,
            artifact_records,
            relative_path=(
                f"invalidations/pair-{invalidation_number}.json"
            ),
            kind="pair_invalidation",
            content=harness.canonical_json_bytes({
                "schema_version": harness.PAIR_INVALIDATION_SCHEMA,
                "reason": "harness_invalid_no_result",
            }),
        )

    for reset_number in range(1, 7):
        _add_baseline_artifact(
            baseline_root,
            artifact_records,
            relative_path=(
                "local_semantic_resets/"
                f"reset-{reset_number}/local_semantic_reset.json"
            ),
            kind="local_semantic_reset",
            content=harness.canonical_json_bytes({
                "schema_version": "local_semantic_reset.v1",
            }),
        )

    _add_baseline_artifact(
        baseline_root,
        artifact_records,
        relative_path="v2_semantic_baseline_defects.json",
        kind="defect_registry",
        content=harness.canonical_json_bytes({
            "schema_version": "v2_semantic_baseline_defects.v1",
            "baseline_id": baseline_id,
            "hard_boundary_failures": [],
            "defects": [
                {"defect_id": f"defect-{defect_number}"}
                for defect_number in range(1, 8)
            ],
        }),
    )
    artifact_records.sort(key=lambda record: record["path"])
    fingerprint_record = next(
        record
        for record in artifact_records
        if record["path"] == fingerprint_path
    )
    architecture_record = next(
        record
        for record in governed_files
        if record["role"] == "architecture_manifest"
    )
    index = {
        "schema_version": harness.BASELINE_INDEX_SCHEMA,
        "baseline_id": baseline_id,
        "phase": harness.BASELINE_INDEX_PHASE,
        "repository": {
            "branch": "test-branch",
            "head": "0" * 40,
        },
        "governed_files": governed_files,
        "canonical_input_sha256": {
            case_id: harness.canonical_sha256(canonical_input)
            for case_id, canonical_input in canonical_inputs.items()
        },
        "architecture_path_closure": {
            "architecture_manifest_path": architecture_record["path"],
            "architecture_manifest_sha256": architecture_record["sha256"],
            "owned_paths_sha256": harness.canonical_sha256(owned_paths),
            "path_count": 116,
            "create_count": 32,
            "modify_count": 84,
            "fingerprint_artifact_path": fingerprint_path,
            "fingerprint_artifact_sha256": fingerprint_record["sha256"],
        },
        "artifact_records": artifact_records,
        "summary": dict(harness.BASELINE_INDEX_SUMMARY),
    }

    harness.validate_baseline_index(
        index,
        baseline_root=baseline_root,
        repository_root=repository_root,
    )

    changed_path = eligible_paths[0]
    original_bytes = changed_path.read_bytes()
    changed_path.write_bytes(original_bytes + b"changed")
    with pytest.raises(ValueError, match="artifact (size|hash) differs"):
        harness.validate_baseline_index(
            index,
            baseline_root=baseline_root,
            repository_root=repository_root,
        )

    changed_path.write_bytes(original_bytes)
    changed_path.unlink()
    with pytest.raises(ValueError, match="baseline artifact is missing"):
        harness.validate_baseline_index(
            index,
            baseline_root=baseline_root,
            repository_root=repository_root,
        )

    changed_path.write_bytes(original_bytes)
    unexpected_path = baseline_root / "unindexed-artifact.json"
    unexpected_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact path closure differs"):
        harness.validate_baseline_index(
            index,
            baseline_root=baseline_root,
            repository_root=repository_root,
        )
