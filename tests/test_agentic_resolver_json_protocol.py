"""Deterministic JSON-only model protocol tests."""

from __future__ import annotations

from agentic_resolver.contracts import (
    AgenticResolverContextV1,
    AgenticResolverRequestV1,
    AgenticResolverSubagentEvidenceV1,
    AgenticResolverSubagentResultV1,
    AgenticResolverSubagentTaskV1,
)
from agentic_resolver.json_protocol import (
    compacted_observation_message,
    contract_error_message,
    parse_json_object,
    skill_catalog_message,
    skill_content_message,
    subagent_result_message,
    subagent_task_message,
    system_policy_message,
    task_message,
    tool_observation_message,
)


def _messages() -> list[str]:
    """Build one instance of every resolver-authored message family."""

    context = AgenticResolverContextV1(
        facts=("fact",),
        constraints=("constraint",),
        desired_output="result",
    )
    request = AgenticResolverRequestV1(objective="objective", context=context)
    child_task = AgenticResolverSubagentTaskV1(
        description="child",
        objective="child objective",
        context=context,
    )
    child_result = AgenticResolverSubagentResultV1(
        subagent_id="child-1",
        observation_id="root:observation:1",
        description="child",
        status="resolved",
        summary="child result",
        evidence=(AgenticResolverSubagentEvidenceV1(
            summary="child evidence",
            provenance_refs=("child-source",),
            limitations=(),
        ),),
        remaining_needs=(),
    )
    messages = [
        system_policy_message(),
        skill_catalog_message(
            catalog_digest="digest",
            skills=({"name": "sample", "description": "Sample skill."},),
        ),
        task_message(request),
        tool_observation_message(
            tool_call_id="call-1",
            observation_id="observation-1",
            tool_name="lookup",
            status="success",
            output={"summary": "result"},
            error=None,
        ),
        skill_content_message(
            name="sample",
            description="Sample skill.",
            catalog_digest="digest",
            content="# Instructions",
        ),
        subagent_task_message(child_task),
        subagent_result_message(child_result),
        contract_error_message(
            code="no_tool_call",
            message="Return one tool call.",
            remaining_replacements=1,
        ),
        compacted_observation_message(
            observation_id="observation-1",
            tool_name="lookup",
            status="success",
            summary="result",
            evidence_refs=("source-1",),
        ),
    ]
    return messages


def test_every_model_message_serializes_to_one_json_object() -> None:
    """Every non-empty resolver-authored payload has one object root."""

    parsed_messages = [parse_json_object(message) for message in _messages()]

    assert all(isinstance(message, dict) for message in parsed_messages)
    assert all("schema_version" in message for message in parsed_messages)


def test_model_protocol_contains_no_xml_catalog_or_freeform_envelopes() -> None:
    """Catalog and instructions remain JSON strings without pseudo-XML frames."""

    combined = "\n".join(_messages())

    assert "<skills" not in combined
    assert "<skill" not in combined
    assert combined.count("{") >= len(_messages())


def test_contract_error_and_compaction_messages_are_json() -> None:
    """Replacement and compaction feedback retain explicit typed families."""

    error = parse_json_object(contract_error_message(
        code="multiple_tool_calls",
        message="Return exactly one tool call.",
        remaining_replacements=0,
    ))
    compacted = parse_json_object(compacted_observation_message(
        observation_id="observation-1",
        tool_name="lookup",
        status="success",
        summary="bounded summary",
        evidence_refs=("source-1",),
    ))

    assert error["message_type"] == "contract_error"
    assert compacted["message_type"] == "compacted_observation"


def test_subagent_result_protocol_exposes_parent_observation_scope() -> None:
    """Parent evidence uses one top-level ID while child IDs stay private."""

    child_result = AgenticResolverSubagentResultV1(
        subagent_id="child-1",
        observation_id="root:observation:1",
        description="child",
        status="resolved",
        summary="child result",
        evidence=(AgenticResolverSubagentEvidenceV1(
            summary="child evidence",
            provenance_refs=("child-source",),
            limitations=(),
        ),),
        remaining_needs=(),
    )

    parsed = parse_json_object(subagent_result_message(child_result))

    assert parsed["observation_id"] == "root:observation:1"
    assert parsed["evidence"] == [{
        "summary": "child evidence",
        "provenance_refs": ["child-source"],
        "limitations": [],
    }]
    assert "observation_id" not in parsed["evidence"][0]
    policy = parse_json_object(system_policy_message())
    protocol = policy["protocol"]
    assert protocol["subagent_result"] == {
        "parent_evidence_observation_id": "top_level_observation_id",
        "nested_child_evidence": "provenance_context_only",
        "nested_child_observation_id": "omitted",
    }
    assert protocol["observation_handle_placement"] == {
        "allowed_field": "submit_result.evidence[].observation_id",
        "semantic_text": "must_not_repeat_current_session_observation_ids",
        "provenance_refs": "separate_validated_channel",
    }
