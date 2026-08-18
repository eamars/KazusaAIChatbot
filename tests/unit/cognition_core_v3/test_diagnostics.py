"""Deterministic tests for V3 public diagnostics and protected chain metadata."""

from __future__ import annotations

import json

from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    EXHAUSTION_FAILURE_CLASS,
    StageFailure,
    StageResult,
)
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    CONFIG_IDENTITY_FIELDS,
    PROTECTED_CHAIN_FIELDS,
    PROTECTED_FAILURE_FIELDS,
    STAGE_TRACE_PUBLIC_FIELDS,
    build_chain_trace_record,
    project_config_identity,
    project_protected_chain_failure,
    project_protected_chain_result,
)


class _Thinking:
    enabled = False


class _Config:
    stage_name = "appraisal_stage"
    route_name = "cognition_appraisal"
    base_url = "https://llm.example/v1"
    model = "test-model"
    temperature = 0.2
    top_p = None
    top_k = None
    max_completion_tokens = 512
    presence_penalty = 0.0
    timeout_seconds = 60.0
    api_key = "sk-test-secret-credential-value"

    def __init__(self) -> None:
        self.thinking = _Thinking()


def test_v3_public_diagnostics_match_v2_contract():
    config = project_config_identity(_Config())
    assert set(config) == set(CONFIG_IDENTITY_FIELDS)
    assert config["stage_name"] == "appraisal_stage"
    assert config["route_name"] == "cognition_appraisal"
    assert config["model"] == "test-model"
    assert config["thinking_enabled"] is False

    record = build_chain_trace_record(
        chain_name="causal_normative",
        stage_id="event_agency",
        config=_Config(),
        system_prompt="static appraisal prompt",
        human_payload="dynamic tail payload",
        raw_output='{"propositions": []}',
        parsed_output={"propositions": []},
        parse_status="succeeded",
        started_at=10.0,
        ended_at=10.75,
        branch_id=None,
        attempt_number=2,
        error=None,
    )

    # The V3 record carries exactly the V2 public field set plus the two
    # protected chain-scope fields and nothing else.
    assert set(record) == (set(STAGE_TRACE_PUBLIC_FIELDS) | set(PROTECTED_CHAIN_FIELDS))
    assert record["chain_name"] == "causal_normative"
    assert record["stage_id"] == "event_agency"
    assert record["branch_id"] is None
    assert record["system_prompt"] == "static appraisal prompt"
    assert record["human_payload"] == "dynamic tail payload"
    assert record["raw_output"] == '{"propositions": []}'
    assert record["parsed_output"] == {"propositions": []}
    assert record["parse_status"] == "succeeded"
    assert record["started_at_monotonic"] == 10.0
    assert record["ended_at_monotonic"] == 10.75
    assert record["duration_ms"] == 750
    assert record["attempt_number"] == 2
    assert record["error"] is None
    assert record["config"] == config

    # A failed attempt records its concrete failure text in the public slot,
    # matching the V2 validation-capture behavior exactly.
    failed = build_chain_trace_record(
        chain_name="causal_normative",
        stage_id="event_agency",
        config=_Config(),
        system_prompt="static appraisal prompt",
        human_payload="dynamic tail payload",
        raw_output=None,
        parsed_output=None,
        parse_status="contract_error",
        started_at=10.0,
        ended_at=10.25,
        attempt_number=2,
        error="appraisal candidate fields are not exact",
    )
    assert failed["error"] == "appraisal candidate fields are not exact"
    assert failed["parse_status"] == "contract_error"
    assert failed["duration_ms"] == 250


def test_protected_chain_metadata_excludes_secrets_and_rejected_content():
    config_identity = project_config_identity(_Config())
    projected_text = json.dumps(config_identity, ensure_ascii=False)
    assert "api_key" not in projected_text
    assert _Config().api_key not in projected_text

    record = build_chain_trace_record(
        chain_name="relationship",
        stage_id="relationship_social",
        config=_Config(),
        system_prompt="static goal prompt",
        human_payload="dynamic tail payload",
        raw_output=None,
        parsed_output=None,
        parse_status="provider_error",
        started_at=5.0,
        ended_at=5.2,
        attempt_number=1,
    )
    record_text = json.dumps(record, ensure_ascii=False)
    assert "api_key" not in record_text
    assert _Config().api_key not in record_text

    boundary_failure = StageFailure(
        chain_name="epistemic_meaning",
        stage_name="existential_drive",
        failure_class="semantic_boundary_terminal",
        error_code="cognition_boundary_rejected",
        repair_attempted=False,
    )
    protected_failure = project_protected_chain_failure(boundary_failure)
    assert set(protected_failure) == set(PROTECTED_FAILURE_FIELDS)
    assert protected_failure["failure_class"] == "semantic_boundary_terminal"
    assert protected_failure["error_code"] == "cognition_boundary_rejected"
    assert protected_failure["repair_attempted"] is False

    exhausted_result = StageResult(
        chain_name="epistemic_meaning",
        stage_name="existential_drive",
        accepted=False,
        local_state=None,
        semantic_summary=None,
        failure=StageFailure(
            chain_name="epistemic_meaning",
            stage_name="existential_drive",
            failure_class=EXHAUSTION_FAILURE_CLASS,
            error_code="semantic_appraisal_contract_exhausted",
            repair_attempted=True,
        ),
    )
    protected_result = project_protected_chain_result(exhausted_result)
    assert set(protected_result) == {"chain_name", "stage_name", "accepted", "failure"}
    assert protected_result["accepted"] is False
    assert protected_result["failure"]["repair_attempted"] is True

    # Raw rejected candidate text and provider metadata never cross the
    # protection boundary: they are absent from every protected projection.
    rejected_candidate_body = "REJECTED_CANDIDATE_BODY_7f3a"
    raw_evidence_only = {
        "rejected_candidate": rejected_candidate_body,
        "provider_request_id": "req-protected-metadata",
    }
    for projection in (
        protected_failure,
        protected_result,
        record,
        config_identity,
    ):
        projected_text = json.dumps(projection, ensure_ascii=False)
        assert rejected_candidate_body not in projected_text
        assert raw_evidence_only["provider_request_id"] not in projected_text
