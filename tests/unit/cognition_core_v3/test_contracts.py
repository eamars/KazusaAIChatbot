"""Deterministic tests for V3 internal chain and cache-domain contracts."""

from __future__ import annotations

import dataclasses

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import contracts


def _accepted_result() -> contracts.StageResult:
    return contracts.StageResult(
        chain_name="causal_normative",
        stage_name="event_agency",
        accepted=True,
        local_state={"propositions": []},
        semantic_summary="bounded summary",
    )


def _rejected_result(failure: contracts.StageFailure) -> contracts.StageResult:
    return contracts.StageResult(
        chain_name=failure.chain_name,
        stage_name=failure.stage_name,
        accepted=False,
        local_state=None,
        semantic_summary=None,
        failure=failure,
    )


def _boundary_failure(**overrides: object) -> contracts.StageFailure:
    base: dict[str, object] = {
        "chain_name": "relationship",
        "stage_name": "relationship_social",
        "failure_class": contracts.CANDIDATE_ORIGIN_MISSING,
        "error_code": contracts.BOUNDARY_REJECTED_ERROR_CODE,
    }
    base.update(overrides)
    return contracts.StageFailure(**base)


def test_chain_contracts_reject_unknown_fields_and_values():
    accepted = _accepted_result()
    assert contracts.validate_stage_result(accepted) is accepted

    with pytest.raises(ValueError, match="Unknown registered chain"):
        contracts.validate_stage_result(dataclasses.replace(accepted, chain_name="invented_chain"))

    with pytest.raises(ValueError, match="not registered under chain"):
        cross = dataclasses.replace(
            _rejected_result(_boundary_failure()),
            stage_name="event_agency",
        )
        # Stage event_agency is not ordered under the relationship chain.
        contracts.validate_stage_result(cross)

    paired_accepted = dataclasses.replace(accepted, failure=_boundary_failure())
    with pytest.raises(ValueError, match="failure record"):
        contracts.validate_stage_result(paired_accepted)

    unpaired_rejected = dataclasses.replace(
        _rejected_result(_boundary_failure()),
        failure=None,
    )
    with pytest.raises(ValueError, match="failure record"):
        contracts.validate_stage_result(unpaired_rejected)

    missing_state = dataclasses.replace(accepted, local_state=None)
    with pytest.raises(ValueError, match="typed local state and a bounded summary"):
        contracts.validate_stage_result(missing_state)

    unknown_class = _rejected_result(_boundary_failure(failure_class="invented_failure"))
    with pytest.raises(ValueError, match="Unknown stage failure class"):
        contracts.validate_stage_result(unknown_class)

    mismatched_owner = contracts.StageResult(
        chain_name="relationship",
        stage_name="relationship_social",
        accepted=False,
        local_state=None,
        semantic_summary=None,
        failure=contracts.StageFailure(
            chain_name="causal_normative",
            stage_name="event_agency",
            failure_class=contracts.CANDIDATE_ORIGIN_MISSING,
            error_code=contracts.BOUNDARY_REJECTED_ERROR_CODE,
        ),
    )
    with pytest.raises(ValueError, match="exact failing owner"):
        contracts.validate_stage_result(mismatched_owner)

    checkpoint = contracts.ChainCheckpoint(
        chain_name="epistemic_meaning",
        accepted_local_state={"propositions": []},
        semantic_summaries=("bounded summary",),
        next_owner_projection={"state_slice": {}},
    )
    assert contracts.validate_chain_checkpoint(checkpoint) is checkpoint

    with pytest.raises(ValueError, match="Unknown registered chain"):
        contracts.validate_chain_checkpoint(dataclasses.replace(checkpoint, chain_name="invented_chain"))

    with pytest.raises(TypeError, match="mapping of accepted typed values"):
        bad_state = dataclasses.replace(checkpoint, accepted_local_state=("not", "a", "mapping"))  # type: ignore[arg-type]
        contracts.validate_chain_checkpoint(bad_state)

    with pytest.raises(ValueError, match="non-empty strings"):
        empty_summary = dataclasses.replace(checkpoint, semantic_summaries=("",))
        contracts.validate_chain_checkpoint(empty_summary)


def test_boundary_class_failures_are_terminal_without_repair_calls():
    valid_result = _rejected_result(_boundary_failure())
    assert contracts.validate_stage_result(valid_result).failure is valid_result.failure

    repaired = _rejected_result(_boundary_failure(repair_attempted=True))
    with pytest.raises(ValueError, match="zero repair calls"):
        contracts.validate_stage_result(repaired)

    for boundary_class in sorted(contracts.TERMINAL_BOUNDARY_CLASSES):
        wrong_code = _rejected_result(
            _boundary_failure(failure_class=boundary_class, error_code="something_else")
        )
        with pytest.raises(ValueError, match=contracts.BOUNDARY_REJECTED_ERROR_CODE):
            contracts.validate_stage_result(wrong_code)

    exhausted_wrong_code = _rejected_result(
        contracts.StageFailure(
            chain_name="epistemic_meaning",
            stage_name="existential_drive",
            failure_class=contracts.EXHAUSTION_FAILURE_CLASS,
            error_code="wrong_error_code",
            repair_attempted=True,
        )
    )
    with pytest.raises(ValueError, match="contract-exhausted"):
        contracts.validate_stage_result(exhausted_wrong_code)

    exhausted_replacement_issued = _rejected_result(
        contracts.StageFailure(
            chain_name="epistemic_meaning",
            stage_name="existential_drive",
            failure_class=contracts.EXHAUSTION_FAILURE_CLASS,
            error_code=contracts.APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE,
            repair_attempted=True,
        )
    )
    assert contracts.validate_stage_result(exhausted_replacement_issued).failure is not None


def test_cache_domain_identity_is_deterministic_and_credential_free():
    prompt_hash = contracts.hash_static_prompt("static appraisal contract")
    credential_a = contracts.hash_credential_identity("raw-credential-a")

    identity = contracts.CacheDomainIdentity(
        normalized_backend_url="https://backend.test:8080/v1",
        credential_identity_hash=credential_a,
        backend_kind="openai_compatible",
        model="local-model",
        template_strategy="chat",
        static_system_prompt_hash=prompt_hash,
    )
    assert contracts.validate_cache_domain_identity(identity) is identity

    rebuilt = contracts.CacheDomainIdentity(
        normalized_backend_url=identity.normalized_backend_url,
        credential_identity_hash=credential_a,
        backend_kind=identity.backend_kind,
        model=identity.model,
        template_strategy=identity.template_strategy,
        static_system_prompt_hash=prompt_hash,
    )
    assert rebuilt.domain_key() == identity.domain_key()

    different_credential = dataclasses.replace(
        identity,
        credential_identity_hash=contracts.hash_credential_identity("raw-credential-b"),
    )
    assert different_credential.domain_key() != identity.domain_key()

    raw_credential_text = "raw-credential-a"
    assert raw_credential_text not in identity.domain_key()
    assert raw_credential_text not in repr(identity)
    for field_value in dataclasses.asdict(identity).values():
        assert isinstance(field_value, str) and raw_credential_text not in field_value

    with pytest.raises(ValueError, match="must be non-empty"):
        contracts.validate_cache_domain_identity(dataclasses.replace(identity, model=""))

    with pytest.raises(ValueError, match="SHA-256 hex digest"):
        short_hash = dataclasses.replace(identity, credential_identity_hash="nothex")
        contracts.validate_cache_domain_identity(short_hash)
