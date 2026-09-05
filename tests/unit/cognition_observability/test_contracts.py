"""Contract-level tests for canonical cognition observations."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from kazusa_ai_chatbot.brain_service.cognition_observation_contracts import (
    CognitionRunObservationV1,
)
from kazusa_ai_chatbot.brain_service.cognition_observation_projection import (
    build_live_cognition_observation,
    build_self_cognition_observation,
)

_NOW = datetime(2026, 8, 26, 1, 2, 3, tzinfo=timezone.utc)




def _observation_payload() -> dict[str, object]:
    """Return a validated payload suitable for negative contract cases."""

    observation = build_live_cognition_observation(
        graph_result={
            "should_respond": True,
            "reason_to_respond": "a grounded reason",
            "final_dialog": ["A bounded response."],
        },
        persona_state={
            "user_input": "hello",
            "cognition_core_output": {
                "schema_version": "cognition_output.v3",
                "appraisals": [],
                "active_character_goal": {},
                "response_plan": {
                    "action_requests": [],
                    "resolver_requests": [],
                },
                "affect_projection": [],
                "private_monologue": "thinking",
            },
        },
        run_id="run-contract",
        cognition_invocation_id="invocation-contract",
        terminal_status="completed_visible",
        visual_stage_failed=False,
        visual_stage_reached=False,
        failure_code="",
        generated_at=_NOW,
    )
    assert observation is not None
    return observation.model_dump(mode="json")


def _records_section(payload: dict[str, object]) -> dict[str, object]:
    """Return the generated multi-record section used by boundary cases."""

    return next(
        section
        for section in payload["sections"]
        if len(section["records"]) >= 2
    )


def _fields_section(payload: dict[str, object]) -> dict[str, object]:
    """Return the first generated fields section used by boundary cases."""

    return next(
        section
        for section in payload["sections"]
        if section["fields"]
    )


def _additive_section() -> dict[str, object]:
    """Build one valid producer-approved additive section."""

    return {
        "section_id": "producer.extra",
        "label": "Producer extra",
        "category": "producer",
        "presentation": "fields",
        "status": "completed",
        "summary": "Additional producer section.",
        "fields": [{
            "key": "extra_value",
            "label": "Extra value",
            "value": "retained",
        }],
        "records": [],
        "reported_record_count": 0,
        "displayed_record_count": 0,
        "truncated": False,
    }


def test_observation_contract_rejects_unknown_fields_invalid_references_and_over_budget_payloads() -> None:
    """Strict DTOs reject extra keys, dangling refs, and oversized values."""

    base = _observation_payload()
    additive = deepcopy(base)
    additive["sections"].append(_additive_section())
    validated_additive = CognitionRunObservationV1.model_validate(additive)
    assert validated_additive.sections[-1].section_id == "producer.extra"

    unknown = deepcopy(base)
    unknown["unexpected"] = True

    nested_extra = deepcopy(base)
    nested_extra["sections"][0]["fields"][0]["unexpected"] = True

    strict_bool = deepcopy(base)
    strict_bool["nodes"][0]["column"] = True

    naive_datetime = deepcopy(base)
    naive_datetime["generated_at"] = datetime.fromisoformat(
        "2026-08-26T01:02:03"
    )

    invalid_node_id = deepcopy(base)
    invalid_node_id["nodes"][0]["node_id"] = "Invalid.Node"

    invalid_section_id = deepcopy(base)
    invalid_section_id["sections"][0]["section_id"] = "input"

    invalid_field_key = deepcopy(base)
    invalid_field_key["sections"][0]["fields"][0]["key"] = "bad-key"

    duplicate_node_id = deepcopy(base)
    duplicate_node_id["nodes"].append(deepcopy(duplicate_node_id["nodes"][0]))

    duplicate_section_id = deepcopy(base)
    duplicate_section_id["sections"].append(
        deepcopy(duplicate_section_id["sections"][0])
    )

    duplicate_field_key = deepcopy(base)
    field_section = _fields_section(duplicate_field_key)
    field_section["fields"].append(deepcopy(field_section["fields"][0]))

    duplicate_record_key = deepcopy(base)
    record_section = _records_section(duplicate_record_key)
    record_section["records"].append(deepcopy(record_section["records"][0]))

    duplicate_reference = deepcopy(base)
    reference_node = next(
        node
        for node in duplicate_reference["nodes"]
        if len(node["section_refs"]) >= 2
    )
    reference_node["section_refs"][1] = reference_node["section_refs"][0]

    non_item_record_key = deepcopy(base)
    non_item_section = _records_section(non_item_record_key)
    non_item_section["records"][0]["key"] = "record_01"

    out_of_order_record_keys = deepcopy(base)
    out_of_order_section = _records_section(out_of_order_record_keys)
    out_of_order_section["records"][0]["key"] = "item_02"
    out_of_order_section["records"][1]["key"] = "item_01"

    nested_mapping_value = deepcopy(base)
    nested_mapping_value["sections"][0]["fields"][0]["value"] = {
        "nested": "mapping",
    }

    nested_list_value = deepcopy(base)
    nested_list_value["sections"][0]["fields"][0]["value"] = [{
        "nested": "mapping",
    }]

    dangling_edge = deepcopy(base)
    dangling_edge["edges"][0]["target"] = "missing.node"

    invalid_edge_kind = deepcopy(base)
    invalid_edge_kind["edges"][0]["kind"] = "branch"

    missing_required_section = deepcopy(base)
    missing_required_section["sections"] = [
        section
        for section in missing_required_section["sections"]
        if section["section_id"] != "input.turn"
    ]

    self_observation = build_self_cognition_observation(
        artifact_payloads={
            "self_cognition_run_record.json": {
                "run_id": "self-contract-run",
                "status": "completed",
            },
        },
        visual_stage_failed=False,
        visual_stage_reached=False,
        generated_at=_NOW,
    )
    assert self_observation is not None
    opposite_run_kind = deepcopy(base)
    opposite_run_kind["sections"].append(next(
        section.model_dump(mode="json")
        for section in self_observation.sections
        if section.section_id == "self.source"
    ))

    over_budget = deepcopy(base)
    for section_index in range(2):
        over_budget["sections"].append({
            "section_id": f"producer.extra_{section_index}",
            "label": "Producer extra",
            "category": "producer",
            "presentation": "fields",
            "status": "completed",
            "summary": "Large producer section.",
            "fields": [
                {
                    "key": f"field_{field_index:02d}",
                    "label": "Large field",
                    "value": "x" * 4000,
                }
                for field_index in range(24)
            ],
            "records": [],
            "reported_record_count": 0,
            "displayed_record_count": 0,
            "truncated": False,
        })

    invalid_candidates = [
        ("unknown", unknown),
        ("nested_extra", nested_extra),
        ("strict_bool", strict_bool),
        ("naive_datetime", naive_datetime),
        ("invalid_node_id", invalid_node_id),
        ("invalid_section_id", invalid_section_id),
        ("invalid_field_key", invalid_field_key),
        ("duplicate_node_id", duplicate_node_id),
        ("duplicate_section_id", duplicate_section_id),
        ("duplicate_field_key", duplicate_field_key),
        ("duplicate_record_key", duplicate_record_key),
        ("duplicate_reference", duplicate_reference),
        ("non_item_record_key", non_item_record_key),
        ("out_of_order_record_keys", out_of_order_record_keys),
        ("nested_mapping_value", nested_mapping_value),
        ("nested_list_value", nested_list_value),
        ("dangling_edge", dangling_edge),
        ("invalid_edge_kind", invalid_edge_kind),
        ("missing_required_section", missing_required_section),
        ("opposite_run_kind", opposite_run_kind),
        ("over_budget", over_budget),
    ]
    accepted_candidates: list[str] = []
    for candidate_name, candidate in invalid_candidates:
        try:
            CognitionRunObservationV1.model_validate(candidate)
        except ValidationError:
            continue
        accepted_candidates.append(candidate_name)
    assert accepted_candidates == [], (
        "contract accepted invalid candidates: "
        f"{accepted_candidates}"
    )


def test_observation_contract_enforces_truthful_record_counts_statuses_and_utc_serialization() -> None:
    """Counts, aggregation, and timestamps remain canonical at the wire edge."""

    payload = _observation_payload()
    payload["generated_at"] = "2026-08-26T13:02:03+12:00"
    validated = CognitionRunObservationV1.model_validate(payload)
    assert validated.generated_at.tzinfo == timezone.utc
    assert validated.model_dump(mode="json")["generated_at"].endswith("Z")

    invalid_counts = _observation_payload()
    invalid_counts["sections"][0]["displayed_record_count"] = 1
    with pytest.raises(ValidationError):
        CognitionRunObservationV1.model_validate(invalid_counts)

    records_below_displayed = _observation_payload()
    records_section = _records_section(records_below_displayed)
    records_section["reported_record_count"] = 1
    with pytest.raises(ValidationError):
        CognitionRunObservationV1.model_validate(records_below_displayed)

    false_truncated = _observation_payload()
    false_truncated_section = _records_section(false_truncated)
    false_truncated_section["reported_record_count"] = (
        false_truncated_section["displayed_record_count"] + 1
    )
    false_truncated_section["truncated"] = False
    with pytest.raises(ValidationError):
        CognitionRunObservationV1.model_validate(false_truncated)

    true_without_omission = _observation_payload()
    true_without_omission_section = _records_section(true_without_omission)
    true_without_omission_section["truncated"] = True
    with pytest.raises(ValidationError):
        CognitionRunObservationV1.model_validate(true_without_omission)

    invalid_status = _observation_payload()
    invalid_status["nodes"][0]["status"] = "failed"
    with pytest.raises(ValidationError):
        CognitionRunObservationV1.model_validate(invalid_status)
