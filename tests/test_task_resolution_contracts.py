"""Target-state contracts for the unified task-resolution boundary."""

from __future__ import annotations

import importlib
import inspect
from importlib.util import find_spec
from typing import get_type_hints

import pytest


PUBLIC_EXPORTS = (
    "TaskResolutionCheckpointV1",
    "TaskResolutionContractError",
    "TaskResolutionExecutionContextV1",
    "TaskResolutionResultV1",
    "TaskSpecialistRequestV1",
    "TaskSpecialistResultV1",
    "resolve_task_inline",
    "resume_task_resolution",
    "validate_task_resolution_checkpoint",
    "validate_task_resolution_result",
    "validate_task_specialist_request",
    "validate_task_specialist_result",
)


def _evidence() -> dict[str, object]:
    return {
        "schema_version": "task_resolution_evidence.v1",
        "evidence_id": "evidence-1",
        "task_node_id": "node-1",
        "specialist": "local_context",
        "summary": "A relevant prior commitment was found.",
        "provenance_refs": ["memory:item-1"],
        "limitations": [],
    }


def _result(*, status: str, evidence: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "task_resolution_result.v1",
        "status": status,
        "prompt_safe_summary": "One relevant fact was recovered.",
        "evidence": evidence,
        "completed_subgoals": [],
        "remaining_needs": ["Confirm the current constraint."],
        "checkpoint": {},
        "coding_run_context": {},
    }


def test_task_resolution_public_boundary_exists() -> None:
    """The package exposes one canonical inline and resume boundary."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")

    for name in PUBLIC_EXPORTS:
        assert hasattr(module, name), name


def test_task_resolution_entrypoint_signatures_are_frozen() -> None:
    """Callers pass only typed task state and an explicit inline budget."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")

    inline = inspect.signature(module.resolve_task_inline)
    resume = inspect.signature(module.resume_task_resolution)

    assert tuple(inline.parameters) == (
        "request",
        "execution_context",
        "inline_budget_seconds",
    )
    assert inline.parameters["inline_budget_seconds"].kind is (
        inspect.Parameter.KEYWORD_ONLY
    )
    assert tuple(resume.parameters) == ("checkpoint", "execution_context")
    hints = get_type_hints(module.resolve_task_inline)
    cognition_contracts = importlib.import_module(
        "kazusa_ai_chatbot.cognition_shared.contracts"
    )
    assert hints["request"] is cognition_contracts.ResolverCapabilityRequestV2


def test_removed_background_provider_module_is_absent() -> None:
    """The v1 provider dispatcher has no importable compatibility path."""

    assert find_spec("kazusa_ai_chatbot.background_work.providers") is None


def test_evidence_bearing_partial_is_valid() -> None:
    """Partial is a terminal success only when grounded by evidence."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")

    validated = module.validate_task_resolution_result(
        _result(status="partial", evidence=[_evidence()]),
    )

    assert validated["status"] == "partial"
    assert validated["evidence"][0]["evidence_id"] == "evidence-1"


def test_zero_evidence_partial_is_rejected() -> None:
    """Completed-subgoal claims cannot turn an ungrounded result into partial."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")
    result = _result(status="partial", evidence=[])
    result["completed_subgoals"] = ["Claimed completion without evidence."]

    with pytest.raises(module.TaskResolutionContractError, match="partial"):
        module.validate_task_resolution_result(result)


def test_non_coding_specialist_cannot_emit_coding_context() -> None:
    """Only the frozen coding adapter may project coding-run context."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")
    result = {
        "schema_version": "task_specialist_result.v1",
        "specialist": "local_context",
        "status": "resolved",
        "evidence": [_evidence()],
        "completed_subgoals": ["Found public evidence."],
        "remaining_needs": [],
        "reason": "Public sources resolved the node.",
        "retryable": False,
        "coding_run_context": {
            "schema_version": "coding_run_context.v1",
            "coding_run_ref": "coding-run:1",
        },
    }

    with pytest.raises(module.TaskResolutionContractError, match="coding"):
        module.validate_task_specialist_result(result)


def test_specialist_result_rejects_foreign_evidence() -> None:
    """A specialist cannot claim evidence attributed to another owner."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")
    result = {
        "schema_version": "task_specialist_result.v1",
        "specialist": "public_research",
        "status": "resolved",
        "evidence": [_evidence()],
        "completed_subgoals": [],
        "remaining_needs": [],
        "reason": "The result must own all returned evidence.",
        "retryable": False,
    }

    with pytest.raises(module.TaskResolutionContractError, match="specialist"):
        module.validate_task_specialist_result(result)


def test_temporary_unavailable_requires_retryable_result() -> None:
    """A typed temporary failure must explicitly authorize one bounded retry."""

    module = importlib.import_module("kazusa_ai_chatbot.task_resolution")
    result = {
        "schema_version": "task_specialist_result.v1",
        "specialist": "public_research",
        "status": "temporarily_unavailable",
        "evidence": [],
        "completed_subgoals": [],
        "remaining_needs": ["Retry the public source once."],
        "reason": "The provider is temporarily unavailable.",
        "retryable": False,
    }

    with pytest.raises(module.TaskResolutionContractError, match="retryable"):
        module.validate_task_specialist_result(result)
