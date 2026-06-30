"""Deterministic expression calculator for complex-task resolver nodes."""

from __future__ import annotations

import ast
import math
import statistics
from decimal import Decimal
from fractions import Fraction

from .contracts import (
    COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
    ComplexTaskSubagentRequestV1,
    ComplexTaskSubagentResultV1,
    validate_complex_task_subagent_result,
)

_SAFE_GLOBALS: dict[str, object] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "pow": pow,
    "Decimal": Decimal,
    "Fraction": Fraction,
    "math": math,
    "statistics": statistics,
}

for _math_name in dir(math):
    if not _math_name.startswith("_"):
        _SAFE_GLOBALS[_math_name] = getattr(math, _math_name)

_ATTRIBUTE_MODULE_NAMES = frozenset(("math", "statistics"))
_ALLOWED_NODE_TYPES = frozenset((
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,
    ast.keyword,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Invert,
))
_DISALLOWED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.NamedExpr,
    ast.Lambda,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.comprehension,
    ast.For,
    ast.While,
    ast.If,
    ast.With,
    ast.Try,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
)
_NUMERIC_RESULT_TYPES = (int, float, Decimal, Fraction, bool)


class AlgorithmicSubagent:
    """Evaluate caller-prepared numeric expressions without model calls."""

    async def run(
        self,
        request: ComplexTaskSubagentRequestV1,
        context: dict[str, object],
        max_attempts: int = 1,
    ) -> ComplexTaskSubagentResultV1:
        """Dispatch one typed calculation and return an IO envelope."""

        del context
        if max_attempts < 0:
            raise ValueError("max_attempts: expected non-negative integer")
        action = request["action"]
        if action != "evaluate_expression":
            result = self._invalid_result(
                request=request,
                max_attempts=max_attempts,
                reason=f"unsupported algorithmic action: {action}",
            )
            return result
        result = self._run_expression(request, max_attempts)
        return result

    def _run_expression(
        self,
        request: ComplexTaskSubagentRequestV1,
        max_attempts: int,
    ) -> ComplexTaskSubagentResultV1:
        """Evaluate a safe expression already prepared by the caller."""

        payload = request["payload"]
        expression = _require_non_empty_string(payload, "expression")
        label = _optional_label(payload)
        try:
            evaluated_result = _evaluate_expression(expression)
        except (
            ArithmeticError,
            AttributeError,
            KeyError,
            NameError,
            SyntaxError,
            TypeError,
            ValueError,
        ) as exc:
            invalid_result = self._invalid_result(
                request=request,
                max_attempts=max_attempts,
                reason=str(exc),
            )
            return invalid_result
        if not isinstance(evaluated_result, _NUMERIC_RESULT_TYPES):
            invalid_result = self._invalid_result(
                request=request,
                max_attempts=max_attempts,
                reason="expression result: expected numeric or boolean value",
            )
            return invalid_result
        result_text = str(evaluated_result)
        display = f"{label}: {expression} = {result_text}"
        envelope: ComplexTaskSubagentResultV1 = {
            "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
            "resolved": True,
            "status": "resolved",
            "result": {
                "label": label,
                "expression": expression,
                "result_repr": repr(evaluated_result),
                "result_str": result_text,
                "result_type": type(evaluated_result).__name__,
                "display": display,
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "cache_name": "algorithmic",
                "reason": "deterministic calculation",
            },
            "trace": {
                "node_id": request["node_id"],
                "action": request["action"],
                "attempt_limit": max_attempts,
            },
            "unresolved_items": [],
        }
        validated_result = validate_complex_task_subagent_result(envelope)
        return validated_result

    def _invalid_result(
        self,
        *,
        request: ComplexTaskSubagentRequestV1,
        max_attempts: int,
        reason: str,
    ) -> ComplexTaskSubagentResultV1:
        """Build a fail-closed result for unsupported calculations."""

        envelope: ComplexTaskSubagentResultV1 = {
            "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
            "resolved": False,
            "status": "invalid",
            "result": {},
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "cache_name": "algorithmic",
                "reason": "unsupported calculation",
            },
            "trace": {
                "node_id": request["node_id"],
                "action": request["action"],
                "attempt_limit": max_attempts,
            },
            "unresolved_items": [reason],
        }
        validated_result = validate_complex_task_subagent_result(envelope)
        return validated_result


def _evaluate_expression(expression: str) -> object:
    """Evaluate one AST-validated numeric expression."""

    parsed = ast.parse(expression, mode="eval")
    _validate_ast(parsed, set(_SAFE_GLOBALS))
    compiled = compile(parsed, "<complex_task_calc>", "eval")
    result = eval(compiled, {"__builtins__": {}}, dict(_SAFE_GLOBALS))
    return result


def _validate_ast(node: ast.AST, allowed_names: set[str]) -> None:
    """Reject syntax outside the calculator expression subset."""

    for child in ast.walk(node):
        if isinstance(child, _DISALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(child).__name__}")
        if type(child) not in _ALLOWED_NODE_TYPES:
            raise ValueError(f"Unsupported syntax: {type(child).__name__}")
        if isinstance(child, ast.Name) and child.id not in allowed_names:
            raise ValueError(f"Unknown name: {child.id}")
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            _validate_attribute(child)
        if isinstance(child, ast.Call):
            _validate_call(child)


def _validate_attribute(node: ast.Attribute) -> None:
    """Allow only public attributes from approved helper modules."""

    if node.attr.startswith("_"):
        raise ValueError(
            f"Private attribute access is not allowed: {node.attr}"
        )
    if not isinstance(node.value, ast.Name):
        raise ValueError("Attribute access is only allowed on safe modules")
    if node.value.id not in _ATTRIBUTE_MODULE_NAMES:
        raise ValueError("Attribute access is only allowed on safe modules")


def _validate_call(node: ast.Call) -> None:
    """Allow calls only to safe globals or safe module attributes."""

    if isinstance(node.func, ast.Name):
        if node.func.id not in _SAFE_GLOBALS:
            raise ValueError(f"Unknown callable: {node.func.id}")
        return
    if isinstance(node.func, ast.Attribute):
        _validate_attribute(node.func)
        return
    raise ValueError("Unsupported callable syntax")


def _require_non_empty_string(data: dict[str, object], field_name: str) -> str:
    """Require one non-empty string from operation payload data."""

    value = data.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name}: expected non-empty string")
    return_value = value.strip()
    return return_value


def _optional_label(data: dict[str, object]) -> str:
    """Return a display label for the expression result."""

    value = data.get("label", "result")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("label: expected non-empty string")
    label = value.strip()
    return label
