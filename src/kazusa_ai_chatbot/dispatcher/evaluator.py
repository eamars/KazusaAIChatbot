"""Pure validation layer between raw LLM tool calls and scheduled tasks."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from kazusa_ai_chatbot.dispatcher.adapter_iface import AdapterRegistry
from kazusa_ai_chatbot.dispatcher.task import DispatchContext, RawToolCall, Task, parse_iso_datetime
from kazusa_ai_chatbot.dispatcher.tool_spec import ToolRegistry

logger = logging.getLogger(__name__)

_TYPE_MAP = {
    "array": list,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "object": dict,
    "string": str,
}


def _normalize_schema_types(schema_type: object) -> tuple[object, ...]:
    """Return the JSON-schema type field as a tuple for validation.

    Args:
        schema_type: Raw ``type`` field from one schema node.

    Returns:
        Tuple of declared type names.
    """

    if isinstance(schema_type, list):
        return tuple(schema_type)
    return (schema_type,)


def _validate_value(field_name: str, value: object, schema: dict) -> list[str]:
    """Validate one field value against a simple JSON-schema subset.

    Args:
        field_name: Name of the field being validated.
        value: Candidate value.
        schema: Schema fragment for that field.

    Returns:
        List of validation error strings.
    """

    errors: list[str] = []
    allowed_types = _normalize_schema_types(schema.get("type"))
    if "null" in allowed_types and value is None:
        return []

    if allowed_types and allowed_types != (None,):
        python_types = tuple(
            _TYPE_MAP[type_name]
            for type_name in allowed_types
            if type_name != "null" and type_name in _TYPE_MAP
        )
        if python_types and not isinstance(value, python_types):
            expected = ", ".join(str(type_name) for type_name in allowed_types)
            errors.append(f"{field_name}: expected {expected}")
            return errors

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        errors.append(f"{field_name}: expected one of {enum_values}")
    return errors


def _validate_args(args: dict, schema: dict) -> list[str]:
    """Validate a raw argument mapping against a JSON-schema subset.

    Args:
        args: Raw argument mapping emitted by the LLM.
        schema: Tool schema describing required and typed fields.

    Returns:
        List of validation error strings.
    """

    errors: list[str] = []
    if not isinstance(args, dict):
        return ["args: expected object"]

    for required in schema.get("required", []):
        if required not in args:
            errors.append(f"{required}: missing required field")

    properties = schema.get("properties", {})
    for field_name, value in args.items():
        field_schema = properties.get(field_name)
        if field_schema is None:
            continue
        errors.extend(_validate_value(field_name, value, field_schema))
    return errors


@dataclass(frozen=True)
class EvalResult:
    """Result of validating one raw tool call.

    Args:
        ok: Whether validation succeeded.
        task: Validated task when ``ok`` is true.
        errors: Human-readable rejection reasons.
    """

    ok: bool
    task: Task | None
    errors: list[str]


class ToolCallEvaluator:
    """Validate LLM-emitted tool calls against the registry and context."""

    def __init__(self, registry: ToolRegistry, adapters: AdapterRegistry) -> None:
        self._registry = registry
        self._adapters = adapters

    def evaluate(self, raw: RawToolCall, ctx: DispatchContext) -> EvalResult:
        """Validate one raw tool call and produce a concrete scheduled task.

        Args:
            raw: LLM-emitted tool name and raw argument mapping.
            ctx: Source-side dispatch context used for defaulting and permissions.

        Returns:
            ``EvalResult`` containing either a validated task or rejection errors.
        """

        visible_names = self._registry.visible_names(ctx)
        if raw.tool not in visible_names:
            return EvalResult(ok=False, task=None, errors=[f"unknown or unavailable tool: {raw.tool}"])

        spec = self._registry.get(raw.tool)
        errors = _validate_args(raw.args, spec.args_schema)
        if errors:
            return EvalResult(ok=False, task=None, errors=errors)

        args = dict(raw.args)
        target_platform = str(args.get("target_platform") or ctx.source_platform).strip()
        args["target_platform"] = target_platform
        if args.get("target_channel") == "same":
            args["target_channel"] = ctx.source_channel_id

        registered_platforms = self._adapters.platforms()
        if not registered_platforms:
            return EvalResult(ok=False, task=None, errors=["no adapters registered"])
        if target_platform not in registered_platforms:
            return EvalResult(ok=False, task=None, errors=[f"unknown_platform: {target_platform}"])

        execute_at_raw = args.pop("execute_at", None)
        if execute_at_raw in (None, ""):
            execute_at = ctx.now
        else:
            try:
                execute_at = parse_iso_datetime(str(execute_at_raw))
            except ValueError:
                logger.warning("Rejecting tool call with unparseable execute_at: %r", execute_at_raw)
                return EvalResult(
                    ok=False,
                    task=None,
                    errors=["unparseable execute_at"],
                )

        task = Task(tool=raw.tool, args=args, execute_at=execute_at)
        return EvalResult(ok=True, task=task, errors=[])
