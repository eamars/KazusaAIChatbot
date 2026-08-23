"""Canonical JSON-only semantic messages for resolver model history."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from agentic_resolver.contracts import (
    AgenticResolverContractError,
    AgenticResolverRequestV1,
    AgenticResolverSubagentResultV1,
    AgenticResolverSubagentTaskV1,
)


def canonical_json_object(value: Mapping[str, object]) -> str:
    """Serialize exactly one object with stable cache-friendly ordering."""

    try:
        serialized = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AgenticResolverContractError(
            f"JSON protocol value is not serializable: {exc}",
            code="invalid_json_message",
        ) from exc
    parse_json_object(serialized)
    return serialized


def parse_json_object(value: str) -> dict[str, object]:
    """Parse one complete object without transport or semantic repair."""

    if not isinstance(value, str):
        raise AgenticResolverContractError(
            "JSON protocol content must be a string",
            code="invalid_json_message",
        )
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AgenticResolverContractError(
            f"JSON protocol content is malformed: {exc}",
            code="invalid_json_message",
        ) from exc
    if not isinstance(parsed, dict):
        raise AgenticResolverContractError(
            "JSON protocol content must have an object root",
            code="invalid_json_message",
        )
    return parsed


def system_policy_message() -> str:
    """Build the stable process policy supplied to every root and child."""

    payload = {
        "schema_version": "agentic_resolver_system.v1",
        "message_type": "system_policy",
        "role": (
            "Resolve the supplied task by selecting registered tools and "
            "returning a typed result."
        ),
        "decision_process": [
            "Inspect the task and current observations.",
            "Load every clearly applicable skill before following it.",
            "Use a registered tool only when it advances the task.",
            "Use run_subagent for a focused independent branch when available.",
            "Use submit_result when the task or a terminal limitation is known.",
        ],
        "protocol": {
            "response_transport": "native_tool_call",
            "tool_calls_per_step": 1,
            "terminal_tool": "submit_result",
            "assistant_text_format": "empty_or_json_object",
            "subagent_result": {
                "parent_evidence_observation_id": "top_level_observation_id",
                "nested_child_evidence": "provenance_context_only",
                "nested_child_observation_id": "omitted",
            },
            "observation_handle_placement": {
                "allowed_field": "submit_result.evidence[].observation_id",
                "semantic_text": (
                    "must_not_repeat_current_session_observation_ids"
                ),
                "provenance_refs": "separate_validated_channel",
            },
        },
    }
    message = canonical_json_object(payload)
    return message


def skill_catalog_message(
    *,
    catalog_digest: str,
    skills: Sequence[Mapping[str, str]],
) -> str:
    """Build the summary-only immutable startup skill catalog."""

    payload = {
        "schema_version": "agentic_resolver_skill_catalog.v1",
        "message_type": "skill_catalog",
        "catalog_digest": catalog_digest,
        "skills": [dict(skill) for skill in skills],
        "selection": {
            "tool": "skill",
            "instruction": (
                "Load every clearly applicable skill before taking task actions."
            ),
        },
    }
    message = canonical_json_object(payload)
    return message


def task_message(request: AgenticResolverRequestV1) -> str:
    """Build the per-session task message from validated caller input."""

    payload = {
        "schema_version": "agentic_resolver_task.v1",
        "message_type": "task",
        "objective": request.objective,
        "context": request.context.to_dict(),
    }
    message = canonical_json_object(payload)
    return message


def tool_observation_message(
    *,
    tool_call_id: str,
    observation_id: str,
    tool_name: str,
    status: str,
    output: Mapping[str, object],
    error: str | None,
) -> str:
    """Build one bounded ordinary-tool observation."""

    payload = {
        "schema_version": "agentic_resolver_tool_observation.v1",
        "message_type": "tool_observation",
        "tool_call_id": tool_call_id,
        "observation_id": observation_id,
        "tool_name": tool_name,
        "status": status,
        "output": dict(output),
        "error": error,
    }
    message = canonical_json_object(payload)
    return message


def skill_content_message(
    *,
    name: str,
    description: str,
    catalog_digest: str,
    content: str,
) -> str:
    """Build the lazy-loaded Markdown instruction envelope."""

    payload = {
        "schema_version": "agentic_resolver_skill_content.v1",
        "message_type": "skill_content",
        "name": name,
        "description": description,
        "catalog_digest": catalog_digest,
        "content_format": "markdown",
        "content": content,
    }
    message = canonical_json_object(payload)
    return message


def subagent_task_message(task: AgenticResolverSubagentTaskV1) -> str:
    """Build one explicit child task without parent transcript material."""

    payload = {
        "schema_version": "agentic_resolver_subagent_task.v1",
        "message_type": "subagent_task",
        "description": task.description,
        "objective": task.objective,
        "context": task.context.to_dict(),
    }
    message = canonical_json_object(payload)
    return message


def subagent_result_message(result: AgenticResolverSubagentResultV1) -> str:
    """Serialize the bounded typed child projection for its parent."""

    message = canonical_json_object(result.to_dict())
    return message


def contract_error_message(
    *,
    code: str,
    message: str,
    remaining_replacements: int,
) -> str:
    """Build structural replacement feedback owned by the loop."""

    payload = {
        "schema_version": "agentic_resolver_contract_error.v1",
        "message_type": "contract_error",
        "code": code,
        "message": message,
        "remaining_replacements": remaining_replacements,
    }
    serialized = canonical_json_object(payload)
    return serialized


def compacted_observation_message(
    *,
    observation_id: str,
    tool_name: str,
    status: str,
    summary: str,
    evidence_refs: Sequence[str],
) -> str:
    """Build the atomic replacement for one old tool exchange."""

    payload = {
        "schema_version": "agentic_resolver_compacted_observation.v1",
        "message_type": "compacted_observation",
        "observation_id": observation_id,
        "tool_name": tool_name,
        "status": status,
        "summary": summary,
        "evidence_refs": list(evidence_refs),
    }
    serialized = canonical_json_object(payload)
    return serialized
