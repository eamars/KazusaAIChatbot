"""Deterministic public-contract tests for agentic_resolver."""

from __future__ import annotations

import inspect

import pytest

import agentic_resolver
from agentic_resolver.contracts import (
    AgenticResolverContractError,
    AgenticResolverEvidenceV1,
    AgenticResolverRequestV1,
    AgenticResolverSubagentEvidenceV1,
    AgenticResolverSubagentResultV1,
    SubmitResultV1,
)
from agentic_resolver.model import (
    AgenticModelCapabilitiesV1,
    AgenticModelClient,
)


def _request_value() -> dict[str, object]:
    """Return one exact valid public request JSON object."""

    value = {
        "schema_version": "agentic_resolver_request.v1",
        "objective": "Resolve one bounded task.",
        "context": {
            "facts": ["One prompt-safe fact."],
            "constraints": ["Use only registered tools."],
            "desired_output": "A concise supported result.",
        },
    }
    return value


def _submit_value() -> dict[str, object]:
    """Return one structurally valid terminal argument object."""

    value = {
        "status": "resolved",
        "summary": "The task is complete.",
        "evidence": [],
        "completed_tasks": ["Resolve one bounded task."],
        "remaining_needs": [],
    }
    return value


def test_public_api_exposes_standalone_runtime_only() -> None:
    """The package facade exports direct-call contracts without workflow state."""

    assert set(agentic_resolver.__all__) == {
        "AgenticModelCapabilitiesV1",
        "AgenticModelClient",
        "AgenticResolverContractError",
        "AgenticResolverLimitsV1",
        "AgenticResolverRequestV1",
        "AgenticResolverResultV1",
        "AgenticResolverRuntime",
        "ModelStreamAssembler",
        "ModelStreamChunk",
        "SkillCatalog",
        "ToolDefinition",
        "ToolRegistry",
        "discover_skills",
    }
    assert "brain_service" not in agentic_resolver.__dict__
    assert "cognition" not in agentic_resolver.__dict__


def test_request_and_result_contracts_are_strict_json_objects() -> None:
    """Unknown keys and non-object roots fail before runtime use."""

    request = AgenticResolverRequestV1.from_mapping(_request_value())
    invalid_request = _request_value()
    invalid_request["unknown"] = True

    assert request.to_dict() == _request_value()
    with pytest.raises(AgenticResolverContractError, match="unknown"):
        AgenticResolverRequestV1.from_mapping(invalid_request)
    with pytest.raises(AgenticResolverContractError, match="expected object"):
        AgenticResolverRequestV1.from_mapping([])


def test_submit_result_rejects_unknown_status_and_missing_fields() -> None:
    """Terminal semantic fields retain a closed strict schema."""

    unknown_status = _submit_value()
    unknown_status["status"] = "done"
    missing_summary = _submit_value()
    del missing_summary["summary"]

    with pytest.raises(AgenticResolverContractError, match="unsupported"):
        SubmitResultV1.from_mapping(unknown_status)
    with pytest.raises(AgenticResolverContractError, match="missing"):
        SubmitResultV1.from_mapping(missing_summary)


def test_subagent_result_requires_parent_observation_and_private_child_evidence() -> None:
    """The parent projection separates its citeable ID from child details."""

    child_evidence = AgenticResolverEvidenceV1(
        observation_id="child-session:observation:1",
        summary="Child fact.",
        provenance_refs=("fixture:child",),
        limitations=(),
    )
    projected_evidence = (
        AgenticResolverSubagentEvidenceV1.from_terminal_evidence(
            child_evidence
        ),
    )
    result = AgenticResolverSubagentResultV1(
        subagent_id="child-session",
        observation_id="root-session:observation:2",
        description="Inspect child evidence.",
        status="resolved",
        summary="Child completed.",
        evidence=projected_evidence,
        remaining_needs=(),
    )

    value = result.to_dict()

    assert value["observation_id"] == "root-session:observation:2"
    assert value["evidence"] == [{
        "summary": "Child fact.",
        "provenance_refs": ["fixture:child"],
        "limitations": [],
    }]
    assert "child-session:observation:1" not in str(value)


def test_agentic_model_client_requires_native_tool_chunk_stream() -> None:
    """The model seam declares one async native-tool chunk stream."""

    signature = inspect.signature(AgenticModelClient.astream)

    assert list(signature.parameters) == ["self", "messages", "tools"]
    assert signature.parameters["tools"].kind is inspect.Parameter.KEYWORD_ONLY


def test_agentic_model_client_requires_enabled_thinking_capabilities() -> None:
    """Capability construction fails for disabled thinking or streaming."""

    capabilities = AgenticModelCapabilitiesV1(
        thinking_strategy="qwen3_enabled",
        reasoning_replay_policy="adapter_owned",
    )

    assert capabilities.streaming is True
    assert capabilities.thinking_enabled is True
    with pytest.raises(AgenticResolverContractError, match="streaming"):
        AgenticModelCapabilitiesV1(
            thinking_strategy="qwen3_enabled",
            reasoning_replay_policy="adapter_owned",
            streaming=False,
        )
    with pytest.raises(AgenticResolverContractError, match="thinking"):
        AgenticModelCapabilitiesV1(
            thinking_strategy="disabled",
            reasoning_replay_policy="adapter_owned",
            thinking_enabled=False,
        )
