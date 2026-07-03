"""Real-LLM tests for function-only programmer output quality."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
    FUNCTION_PROGRAMMER_PROMPT,
    run_function_programmer_task,
)
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


_TEST_NAME = "coding_agent_writing_function_programmer_live_llm"
_CASES_PATH = (
    Path("test_artifacts")
    / "live_gate"
    / "coding_agent_function_programmer_cases.json"
)


def _load_cases() -> dict[str, dict[str, Any]]:
    """Load function programmer cases from the live-gate artifact."""

    text = _CASES_PATH.read_text(encoding="utf-8")
    cases = json.loads(text)
    return cases


def _load_case(case_id: str) -> dict[str, Any]:
    """Load one function programmer case."""

    cases = _load_cases()
    case = cases[case_id]
    return case


async def _run_case(case_id: str) -> tuple[dict[str, object], Path]:
    """Run one live function-programmer case and write a trace."""

    case = _load_case(case_id)
    function_contract = case["function_contract"]
    trace: dict[str, object] = {}
    result = await run_function_programmer_task(
        function_task=function_contract,
        trace=trace,
    )
    evaluation = _evaluate_function_result(
        function_contract=function_contract,
        result=result,
        trace=trace,
    )
    trace_payload = {
        "case_id": case_id,
        "category": case["category"],
        "system_prompt": FUNCTION_PROGRAMMER_PROMPT,
        "function_contract": function_contract,
        "programmer_trace": trace,
        "programmer_result": result,
        "evaluation": evaluation,
    }
    trace_path = write_llm_trace(_TEST_NAME, case_id, trace_payload)
    return evaluation, trace_path


def _evaluate_function_result(
    *,
    function_contract: dict[str, Any],
    result: dict[str, object],
    trace: dict[str, object],
) -> dict[str, object]:
    """Evaluate one assembled function against the function-only contract."""

    errors: list[str] = []
    warnings: list[str] = []
    raw_output = trace.get("raw_output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        errors.append("Programmer returned empty raw output.")
    if trace.get("markdown_code_block_found") is not True:
        errors.append("Programmer did not return exactly one markdown Python block.")

    assembled_function = result.get("assembled_function")
    if not isinstance(assembled_function, str) or not assembled_function.strip():
        errors.append("Programmer produced no assembled_function source.")
        evaluation = _evaluation(errors=errors, warnings=warnings)
        return evaluation

    try:
        module = ast.parse(assembled_function)
    except SyntaxError as exc:
        errors.append(f"assembled_function is not valid Python: {exc}")
        evaluation = _evaluation(errors=errors, warnings=warnings)
        return evaluation

    function_nodes = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(function_nodes) != 1:
        errors.append(
            "assembled_function must contain exactly one function; found "
            f"{len(function_nodes)}."
        )
        evaluation = _evaluation(errors=errors, warnings=warnings)
        return evaluation

    extra_nodes = [
        type(node).__name__ for node in module.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if extra_nodes:
        errors.append(
            "assembled_function contains top-level nodes outside one function: "
            + ", ".join(extra_nodes)
        )

    function_node = function_nodes[0]
    expected_name = function_contract["function_name"]
    if function_node.name != expected_name:
        errors.append(
            f"Function name mismatch: expected {expected_name!r}, got "
            f"{function_node.name!r}."
        )

    expected_args = _expected_arg_names(function_contract["function_args"])
    actual_args = [arg.arg for arg in function_node.args.args]
    if actual_args != expected_args:
        errors.append(
            f"Argument mismatch: expected {expected_args!r}, got "
            f"{actual_args!r}."
        )

    expected_return = _normalize_annotation(
        function_contract["function_return_type"],
    )
    actual_return = ""
    if function_node.returns is not None:
        actual_return = _normalize_annotation(ast.unparse(function_node.returns))
    if actual_return != expected_return:
        errors.append(
            f"Return annotation mismatch: expected {expected_return!r}, got "
            f"{actual_return!r}."
        )

    should_be_async = _expects_async(function_contract["function_intention"])
    is_async = isinstance(function_node, ast.AsyncFunctionDef)
    if should_be_async and not is_async:
        errors.append("Function intention requires async def, but output used def.")
    if not should_be_async and is_async:
        warnings.append("Function output used async def without async intention.")

    if _body_is_placeholder(function_node):
        errors.append("Function body is empty or placeholder-only.")

    evaluation = _evaluation(
        errors=errors,
        warnings=warnings,
        function_name=function_node.name,
        actual_args=actual_args,
        actual_return=actual_return,
        is_async=is_async,
    )
    return evaluation


def _evaluation(
    *,
    errors: list[str],
    warnings: list[str],
    function_name: str = "",
    actual_args: list[str] | None = None,
    actual_return: str = "",
    is_async: bool = False,
) -> dict[str, object]:
    status = "passed"
    if errors:
        status = "failed"
    evaluation = {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "function_name": function_name,
        "actual_args": actual_args or [],
        "actual_return": actual_return,
        "is_async": is_async,
    }
    return evaluation


def _expected_arg_names(function_args: str) -> list[str]:
    """Extract parameter names from the compact contract string."""

    if function_args.strip() == "()":
        return []
    names: list[str] = []
    for raw_part in function_args.split(","):
        part = raw_part.strip()
        if not part:
            continue
        name = part.split(":", maxsplit=1)[0].split("=", maxsplit=1)[0].strip()
        if name:
            names.append(name)
    return names


def _normalize_annotation(value: str) -> str:
    """Normalize an annotation string for structural comparison."""

    normalized = "".join(value.strip().split())
    return normalized


def _expects_async(function_intention: str) -> bool:
    """Return whether the contract explicitly asks for async source."""

    lowered = function_intention.lower()
    expected = (
        "async method" in lowered
        or "async function" in lowered
        or "async service function" in lowered
        or "async worker" in lowered
        or "async helper" in lowered
        or "async test" in lowered
    )
    return expected


def _body_is_placeholder(
    function_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """Return whether a generated function body has no real implementation."""

    body = function_node.body
    if not body:
        return True
    if len(body) != 1:
        return False
    only_statement = body[0]
    if isinstance(only_statement, ast.Pass):
        return True
    if isinstance(only_statement, ast.Expr):
        value = only_statement.value
        if isinstance(value, ast.Constant) and value.value is Ellipsis:
            return True
    return False


async def _assert_case(case_id: str) -> None:
    """Run and assert one live function-programmer case."""

    evaluation, trace_path = await _run_case(case_id)
    print(f"TRACE_PATH={trace_path.as_posix()}")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, default=str))
    assert evaluation["status"] == "passed", "\n".join(evaluation["errors"])


async def test_function_programmer_case_01() -> None:
    await _assert_case("case_01")


async def test_function_programmer_case_02() -> None:
    await _assert_case("case_02")


async def test_function_programmer_case_03() -> None:
    await _assert_case("case_03")


async def test_function_programmer_case_04() -> None:
    await _assert_case("case_04")


async def test_function_programmer_case_05() -> None:
    await _assert_case("case_05")


async def test_function_programmer_case_06() -> None:
    await _assert_case("case_06")


async def test_function_programmer_case_07() -> None:
    await _assert_case("case_07")


async def test_function_programmer_case_08() -> None:
    await _assert_case("case_08")


async def test_function_programmer_case_09() -> None:
    await _assert_case("case_09")


async def test_function_programmer_case_10() -> None:
    await _assert_case("case_10")


async def test_function_programmer_case_11() -> None:
    await _assert_case("case_11")


async def test_function_programmer_case_12() -> None:
    await _assert_case("case_12")


async def test_function_programmer_case_13() -> None:
    await _assert_case("case_13")


async def test_function_programmer_case_14() -> None:
    await _assert_case("case_14")


async def test_function_programmer_case_15() -> None:
    await _assert_case("case_15")


async def test_function_programmer_case_16() -> None:
    await _assert_case("case_16")
