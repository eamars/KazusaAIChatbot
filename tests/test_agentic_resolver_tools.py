"""Deterministic frozen tool-registry tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import pytest

from agentic_resolver.contracts import AgenticResolverContractError
from agentic_resolver.tools import (
    CORE_TOOL_GRAMMAR_SAFE_MAX_LENGTH,
    ToolDefinition,
    ToolRegistry,
)


def _schema() -> dict[str, object]:
    """Return one strict object-rooted argument schema."""

    schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
                "maxLength": 100,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    return schema


async def _successful_tool(arguments: Mapping[str, object]) -> object:
    """Return one bounded mapping for registry execution tests."""

    result = {
        "summary": f"Found {arguments['query']}",
        "provenance_refs": ["source-1"],
    }
    return result


def _definition(name: str) -> ToolDefinition:
    """Build one ordinary read-only tool definition."""

    definition = ToolDefinition(
        name=name,
        description="Look up one bounded value.",
        input_schema=_schema(),
        execute=_successful_tool,
    )
    return definition


def _walk_schema_nodes(value: object) -> Iterator[Mapping[str, object]]:
    """Yield each mapping nested in one JSON-schema value.

    Args:
        value: A JSON-schema mapping or a nested list/value to inspect.

    Yields:
        Every mapping node reachable from the supplied schema value.
    """

    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_schema_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_schema_nodes(child)


def test_registry_freezes_sorted_unique_json_schemas() -> None:
    """Registry order is stable and duplicate or reserved names fail startup."""

    registry = ToolRegistry((_definition("zeta"), _definition("alpha")))

    assert registry.names == ("alpha", "zeta")
    assert registry.definitions[0].input_schema["type"] == "object"
    with pytest.raises(AgenticResolverContractError, match="duplicate"):
        ToolRegistry((_definition("alpha"), _definition("alpha")))
    with pytest.raises(AgenticResolverContractError, match="core name"):
        ToolRegistry((_definition("skill"),))


def test_core_model_definitions_use_grammar_safe_caps_and_strict_objects() -> None:
    """Root and child core schemas retain strict structure and safe caps."""

    root_registry = ToolRegistry().with_core_tools(include_subagent=True)
    child_registry = ToolRegistry().with_core_tools(include_subagent=False)
    assert root_registry.names == ("run_subagent", "skill", "submit_result")
    assert child_registry.names == ("skill", "submit_result")

    expected_safe_cap_counts = {
        "run_subagent": 3,
        "skill": 0,
        "submit_result": 5,
    }
    for registry in (root_registry, child_registry):
        for definition in registry.model_definitions():
            schema_nodes = tuple(_walk_schema_nodes(definition.parameters))
            object_nodes = tuple(
                node
                for node in schema_nodes
                if node.get("type") == "object"
            )
            assert object_nodes
            for node in object_nodes:
                assert isinstance(node.get("properties"), Mapping)
                assert isinstance(node.get("required"), list)
                assert node.get("additionalProperties") is False

            max_lengths = tuple(
                node["maxLength"]
                for node in schema_nodes
                if "maxLength" in node
            )
            assert 2000 not in max_lengths
            assert set(max_lengths) <= {
                64,
                200,
                4000,
                CORE_TOOL_GRAMMAR_SAFE_MAX_LENGTH,
            }
            assert (
                max_lengths.count(CORE_TOOL_GRAMMAR_SAFE_MAX_LENGTH)
                == expected_safe_cap_counts[definition.name]
            )


def test_submit_result_schema_places_observation_handles_only_in_evidence() -> None:
    """Terminal schema documents the structured handle-only observation slot."""

    submit_definition = ToolRegistry().with_core_tools(
        include_subagent=False,
    ).get("submit_result")
    properties = submit_definition.input_schema["properties"]
    evidence = properties["evidence"]
    evidence_item = evidence["items"]
    evidence_properties = evidence_item["properties"]

    assert "only field where an observation_id may appear" in (
        evidence_properties["observation_id"]["description"]
    )
    assert "do not repeat" in evidence_properties["summary"]["description"]
    assert "do not repeat" in evidence_properties["limitations"]["description"]
    assert "do not repeat" in properties["summary"]["description"]
    assert "do not repeat" in (
        properties["completed_tasks"]["items"]["description"]
    )
    assert "do not repeat" in (
        properties["remaining_needs"]["items"]["description"]
    )
    assert "separate" in evidence_properties["provenance_refs"]["description"]


@pytest.mark.asyncio
async def test_registry_validates_arguments_before_execution() -> None:
    """Missing or unknown argument keys cannot reach an ordinary executor."""

    calls = 0

    async def _counted_tool(
        arguments: Mapping[str, object],
    ) -> object:
        nonlocal calls
        calls += 1
        output = {"summary": str(arguments["query"])}
        return output

    registry = ToolRegistry((ToolDefinition(
        name="lookup",
        description="Look up one bounded value.",
        input_schema=_schema(),
        execute=_counted_tool,
    ),))

    with pytest.raises(AgenticResolverContractError, match="missing"):
        await registry.execute_tool(
            "lookup",
            {},
            permission_scope={},
            timeout_seconds=1,
            maximum_result_characters=8000,
        )

    assert calls == 0


@pytest.mark.asyncio
async def test_tool_exception_becomes_bounded_json_error() -> None:
    """Tool-boundary failures remove private paths and credential values."""

    async def _failing_tool(arguments: Mapping[str, object]) -> object:
        del arguments
        raise RuntimeError(
            "failed at C:\\private\\workspace\\secret.py api_key=visible-secret"
        )

    registry = ToolRegistry((ToolDefinition(
        name="lookup",
        description="Look up one bounded value.",
        input_schema=_schema(),
        execute=_failing_tool,
    ),))

    result = await registry.execute_tool(
        "lookup",
        {"query": "value"},
        permission_scope={},
        timeout_seconds=1,
        maximum_result_characters=8000,
    )

    assert result.status == "error"
    assert result.error is not None
    assert "visible-secret" not in result.error
    assert "C:\\private" not in result.error
    assert len(result.error) <= 600
