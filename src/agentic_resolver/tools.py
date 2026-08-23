"""Frozen native tool definitions, validation, permissions, and execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from agentic_resolver.contracts import (
    MODEL_VISIBLE_RESULT_CHAR_CAP,
    AgenticResolverContractError,
)
from agentic_resolver.model import AgenticModelToolDefinition

ToolArgumentValidator = Callable[
    [Mapping[str, object]],
    Mapping[str, object],
]
ToolExecutor = Callable[[Mapping[str, object]], Awaitable[object]]
ToolPermissionCheck = Callable[[Mapping[str, object]], bool]
ToolResultProjector = Callable[[object], Mapping[str, object]]

RESERVED_CORE_TOOL_NAMES = frozenset({
    "skill",
    "run_subagent",
    "submit_result",
})
# Keep model-authored core strings below llama.cpp's MAX_REPETITION_THRESHOLD
# of 2000 while preserving larger controller-owned bounds elsewhere.
CORE_TOOL_GRAMMAR_SAFE_MAX_LENGTH = 1_999
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"[A-Za-z]:\\[^\s]+")
_UNIX_ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:[^\s/]+/)+[^\s]+")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+"
)


@dataclass(frozen=True)
class ToolDefinition:
    """One immutable native tool contract and its trusted implementation."""

    name: str
    description: str
    input_schema: Mapping[str, object]
    execute: ToolExecutor
    validate_arguments: ToolArgumentValidator | None = None
    permission_check: ToolPermissionCheck | None = None
    project_result: ToolResultProjector | None = None
    maximum_result_characters: int = MODEL_VISIBLE_RESULT_CHAR_CAP
    side_effect_class: Literal["read", "compute", "approval_gated"] = "read"

    def __post_init__(self) -> None:
        if _TOOL_NAME_PATTERN.fullmatch(self.name) is None:
            raise AgenticResolverContractError(
                f"invalid tool name: {self.name!r}"
            )
        if not self.description.strip() or len(self.description) > 500:
            raise AgenticResolverContractError(
                f"tool {self.name}: description must be 1..500 characters"
            )
        _validate_object_schema(self.input_schema, f"tool {self.name}")
        if (
            isinstance(self.maximum_result_characters, bool)
            or not isinstance(self.maximum_result_characters, int)
            or not 1 <= self.maximum_result_characters <= MODEL_VISIBLE_RESULT_CHAR_CAP
        ):
            raise AgenticResolverContractError(
                f"tool {self.name}: invalid result character bound"
            )
        object.__setattr__(
            self,
            "input_schema",
            MappingProxyType(dict(self.input_schema)),
        )

    def model_definition(self) -> AgenticModelToolDefinition:
        """Project this trusted tool into its prompt-visible native schema."""

        definition = AgenticModelToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_schema,
        )
        return definition


@dataclass(frozen=True)
class ToolExecutionResult:
    """Bounded deterministic outcome from one ordinary tool dispatch."""

    status: Literal["success", "error", "denied"]
    output: Mapping[str, object]
    error: str | None


class ToolRegistry:
    """Sorted frozen tool roster with reserved core-name protection."""

    def __init__(
        self,
        definitions: Sequence[ToolDefinition] = (),
        *,
        _allow_core_names: bool = False,
    ) -> None:
        by_name: dict[str, ToolDefinition] = {}
        for definition in definitions:
            if not isinstance(definition, ToolDefinition):
                raise AgenticResolverContractError(
                    "tool registry accepts ToolDefinition values only"
                )
            if (
                definition.name in RESERVED_CORE_TOOL_NAMES
                and not _allow_core_names
            ):
                raise AgenticResolverContractError(
                    f"ordinary tool cannot shadow core name: {definition.name}"
                )
            if definition.name in by_name:
                raise AgenticResolverContractError(
                    f"duplicate tool name: {definition.name}"
                )
            by_name[definition.name] = definition
        self._definitions = tuple(
            by_name[name] for name in sorted(by_name)
        )
        self._by_name = MappingProxyType(by_name)
        self._schema_digest = _registry_schema_digest(self._definitions)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return the immutable sorted trusted tool definitions."""

        return self._definitions

    @property
    def names(self) -> tuple[str, ...]:
        """Return the immutable prompt-visible tool roster."""

        names = tuple(definition.name for definition in self._definitions)
        return names

    @property
    def schema_digest(self) -> str:
        """Return the canonical digest of the visible tool schemas."""

        return self._schema_digest

    def get(self, name: str) -> ToolDefinition:
        """Return one registered definition or fail the structural boundary."""

        definition = self._by_name.get(name)
        if definition is None:
            raise AgenticResolverContractError(
                f"unknown tool name: {name}",
                code="unknown_tool",
            )
        return definition

    def model_definitions(self) -> tuple[AgenticModelToolDefinition, ...]:
        """Return the sorted native schemas supplied to the model."""

        definitions = tuple(
            definition.model_definition()
            for definition in self._definitions
        )
        return definitions

    def with_core_tools(self, *, include_subagent: bool) -> ToolRegistry:
        """Return a frozen root or child view with controller-owned tools."""

        core_definitions = _core_tool_definitions(
            include_subagent=include_subagent
        )
        registry = ToolRegistry(
            (*self._definitions, *core_definitions),
            _allow_core_names=True,
        )
        return registry

    async def execute_tool(
        self,
        name: str,
        arguments: Mapping[str, object],
        *,
        permission_scope: Mapping[str, object],
        timeout_seconds: float,
        maximum_result_characters: int,
    ) -> ToolExecutionResult:
        """Validate and execute one ordinary tool through all hard gates.

        Args:
            name: Registered ordinary tool selected by the model.
            arguments: Native JSON object arguments from the assembled call.
            permission_scope: Trusted code-owned permissions for this runtime.
            timeout_seconds: Caller-lowered deadline for one execution.
            maximum_result_characters: Runtime-wide model-visible output cap.

        Returns:
            A bounded projected result without provider or stack internals.
        """

        definition = self.get(name)
        if definition.name in RESERVED_CORE_TOOL_NAMES:
            raise AgenticResolverContractError(
                f"core tool {name} must be dispatched by AgentLoop"
            )
        validator = definition.validate_arguments
        if validator is None:
            normalized_arguments = validate_json_schema_arguments(
                arguments,
                definition.input_schema,
                label=f"tool {name} arguments",
            )
        else:
            normalized_arguments = validator(arguments)
        permission_check = definition.permission_check
        permitted = True
        if permission_check is not None:
            permitted = permission_check(permission_scope)
        if not permitted:
            result = ToolExecutionResult(
                status="denied",
                output={},
                error="trusted execution scope denied this tool",
            )
            return result

        try:
            raw_result = await asyncio.wait_for(
                definition.execute(normalized_arguments),
                timeout=timeout_seconds,
            )
        except TimeoutError as exc:
            result = ToolExecutionResult(
                status="error",
                output={},
                error=f"tool timed out: {sanitized_exception_text(exc)}",
            )
            return result
        # Injected tool implementations may raise domain-specific exceptions.
        except Exception as exc:  # noqa: BLE001
            result = ToolExecutionResult(
                status="error",
                output={},
                error=f"tool failed: {sanitized_exception_text(exc)}",
            )
            return result

        projector = definition.project_result
        if projector is None:
            if not isinstance(raw_result, Mapping):
                result = ToolExecutionResult(
                    status="error",
                    output={},
                    error="tool returned a non-object result",
                )
                return result
            projected = dict(raw_result)
        else:
            try:
                projected = dict(projector(raw_result))
            # Injected projectors may raise domain-specific exceptions.
            except Exception as exc:  # noqa: BLE001
                result = ToolExecutionResult(
                    status="error",
                    output={},
                    error=(
                        "tool result projection failed: "
                        f"{sanitized_exception_text(exc)}"
                    ),
                )
                return result
        serialized = _canonical_json(projected)
        effective_maximum = min(
            definition.maximum_result_characters,
            maximum_result_characters,
        )
        if len(serialized) > effective_maximum:
            result = ToolExecutionResult(
                status="error",
                output={},
                error="tool result exceeded the model-visible size bound",
            )
            return result
        result = ToolExecutionResult(
            status="success",
            output=MappingProxyType(projected),
            error=None,
        )
        return result


def validate_json_schema_arguments(
    arguments: Mapping[str, object],
    schema: Mapping[str, object],
    *,
    label: str,
) -> Mapping[str, object]:
    """Validate the strict JSON-Schema subset used by resolver tools."""

    if not isinstance(arguments, Mapping):
        raise AgenticResolverContractError(f"{label}: expected object")
    _validate_schema_value(arguments, schema, label)
    normalized = MappingProxyType(dict(arguments))
    return normalized


def _validate_schema_value(
    value: object,
    schema: Mapping[str, object],
    label: str,
) -> None:
    """Validate one value against the bounded tool-schema subset."""

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, Mapping):
            raise AgenticResolverContractError(f"{label}: expected object")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise AgenticResolverContractError(f"{label}: invalid schema")
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise AgenticResolverContractError(f"{label}: invalid schema")
        missing = [key for key in required if key not in value]
        if missing:
            raise AgenticResolverContractError(
                f"{label}: missing required keys {sorted(missing)}"
            )
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise AgenticResolverContractError(
                    f"{label}: unknown keys {unknown}"
                )
        for key, child_value in value.items():
            child_schema = properties.get(key)
            if isinstance(child_schema, Mapping):
                _validate_schema_value(
                    child_value,
                    child_schema,
                    f"{label}.{key}",
                )
        return
    if schema_type == "array":
        if not isinstance(value, list):
            raise AgenticResolverContractError(f"{label}: expected array")
        maximum_items = schema.get("maxItems")
        if isinstance(maximum_items, int) and len(value) > maximum_items:
            raise AgenticResolverContractError(f"{label}: too many items")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, f"{label}[{index}]")
        return
    if schema_type == "string":
        if not isinstance(value, str):
            raise AgenticResolverContractError(f"{label}: expected string")
        minimum_length = schema.get("minLength")
        maximum_length = schema.get("maxLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            raise AgenticResolverContractError(f"{label}: string is too short")
        if isinstance(maximum_length, int) and len(value) > maximum_length:
            raise AgenticResolverContractError(f"{label}: string is too long")
    elif schema_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise AgenticResolverContractError(f"{label}: expected integer")
    elif schema_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AgenticResolverContractError(f"{label}: expected number")
    elif schema_type == "boolean" and not isinstance(value, bool):
        raise AgenticResolverContractError(f"{label}: expected boolean")
    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        raise AgenticResolverContractError(f"{label}: unsupported value")


def _validate_object_schema(schema: Mapping[str, object], label: str) -> None:
    """Validate one prompt-visible root schema before registry freeze."""

    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise AgenticResolverContractError(
            f"{label}: input schema must have an object root"
        )
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise AgenticResolverContractError(
            f"{label}: input schema requires properties and required"
        )


def _registry_schema_digest(definitions: Sequence[ToolDefinition]) -> str:
    """Return the immutable canonical identity of one visible tool roster."""

    payload = [
        {
            "name": definition.name,
            "description": definition.description,
            "input_schema": dict(definition.input_schema),
        }
        for definition in definitions
    ]
    serialized = _canonical_json(payload)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest


def _canonical_json(value: object) -> str:
    """Serialize a JSON-compatible value for size and digest enforcement."""

    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AgenticResolverContractError(
            f"tool value is not JSON serializable: {exc}"
        ) from exc
    return serialized


def sanitized_exception_text(exc: BaseException) -> str:
    """Return bounded exception text without paths, secrets, or trace data."""

    text = str(exc).replace("\r", " ").replace("\n", " ")
    text = _WINDOWS_ABSOLUTE_PATH.sub("<private-path>", text)
    text = _UNIX_ABSOLUTE_PATH.sub("<private-path>", text)
    text = _SECRET_ASSIGNMENT.sub(r"\1=<redacted>", text)
    normalized = " ".join(text.split())[:500]
    if not normalized:
        normalized = exc.__class__.__name__
    return normalized


async def _core_tool_executor(arguments: Mapping[str, object]) -> object:
    """Fail if controller-owned core tools reach ordinary dispatch."""

    del arguments
    raise RuntimeError("controller-owned core tool reached ordinary dispatch")


def _core_tool_definitions(*, include_subagent: bool) -> tuple[ToolDefinition, ...]:
    """Build the fixed controller-owned schemas for one registry view."""

    string_list_schema = {
        "type": "array",
        "items": {
            "type": "string",
            "minLength": 1,
            "maxLength": CORE_TOOL_GRAMMAR_SAFE_MAX_LENGTH,
        },
        "maxItems": 32,
    }
    context_schema = {
        "type": "object",
        "properties": {
            "facts": string_list_schema,
            "constraints": string_list_schema,
            "desired_output": {
                "type": "string",
                "maxLength": CORE_TOOL_GRAMMAR_SAFE_MAX_LENGTH,
            },
        },
        "required": ["facts", "constraints", "desired_output"],
        "additionalProperties": False,
    }
    evidence_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "observation_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 200,
                    "description": (
                        "Accepted current-session observation handle. This is "
                        "the only field where an observation_id may appear."
                    ),
                },
                "summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": CORE_TOOL_GRAMMAR_SAFE_MAX_LENGTH,
                    "description": (
                        "Semantic evidence summary; do not repeat any "
                        "observation_id here."
                    ),
                },
                "provenance_refs": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": CORE_TOOL_GRAMMAR_SAFE_MAX_LENGTH,
                    },
                    "maxItems": 16,
                    "description": (
                        "Validated provenance references kept separate from "
                        "observation_id handles."
                    ),
                },
                "limitations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": CORE_TOOL_GRAMMAR_SAFE_MAX_LENGTH,
                    },
                    "maxItems": 16,
                    "description": (
                        "Semantic limitations; do not repeat any "
                        "observation_id here."
                    ),
                },
            },
            "required": [
                "observation_id",
                "summary",
                "provenance_refs",
                "limitations",
            ],
            "additionalProperties": False,
        },
        "maxItems": 16,
    }
    definitions = [
        ToolDefinition(
            name="skill",
            description="Load the full instructions for one catalog skill.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            execute=_core_tool_executor,
        ),
        ToolDefinition(
            name="submit_result",
            description=(
                "Submit the typed terminal result when the task or a terminal "
                "limitation is known. Observation handles may appear only in "
                "evidence[].observation_id; do not repeat them in semantic "
                "text."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": [
                            "resolved",
                            "partial",
                            "needs_user_input",
                            "approval_required",
                            "unavailable",
                            "budget_exhausted",
                            "failed",
                        ],
                    },
                    "summary": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 4000,
                        "description": (
                            "Semantic terminal summary; do not repeat any "
                            "observation_id."
                        ),
                    },
                    "evidence": evidence_schema,
                    "completed_tasks": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": CORE_TOOL_GRAMMAR_SAFE_MAX_LENGTH,
                            "description": (
                                "Completed task text; do not repeat any "
                                "observation_id."
                            ),
                        },
                        "maxItems": 16,
                    },
                    "remaining_needs": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": CORE_TOOL_GRAMMAR_SAFE_MAX_LENGTH,
                            "description": (
                                "Remaining need text; do not repeat any "
                                "observation_id."
                            ),
                        },
                        "maxItems": 16,
                    },
                },
                "required": [
                    "status",
                    "summary",
                    "evidence",
                    "completed_tasks",
                    "remaining_needs",
                ],
                "additionalProperties": False,
            },
            execute=_core_tool_executor,
            side_effect_class="compute",
        ),
    ]
    if include_subagent:
        definitions.append(ToolDefinition(
            name="run_subagent",
            description=(
                "Run one focused independent child with an isolated session "
                "and a self-contained task."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 200,
                    },
                    "objective": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 4000,
                    },
                    "context": context_schema,
                },
                "required": ["description", "objective", "context"],
                "additionalProperties": False,
            },
            execute=_core_tool_executor,
            side_effect_class="compute",
        ))
    core_definitions = tuple(definitions)
    return core_definitions
