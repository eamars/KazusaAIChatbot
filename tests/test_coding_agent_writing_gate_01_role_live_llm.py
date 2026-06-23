"""Focused real-LLM role tests for one hard writing gate."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.coding_agent.code_writing.module_product_manager import (
    MODULE_PM_CONTRACT_PROMPT,
    decide_module_programmer_contract,
)
from kazusa_ai_chatbot.coding_agent.code_writing.product_manager import (
    WRITING_PM_PROMPT,
    _pm_payload,
    decide_writing_work,
)
from kazusa_ai_chatbot.coding_agent.code_writing.programmer import (
    MODULE_PROGRAMMER_PROMPT,
    run_module_programmer_contract,
)
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


_TEST_NAME = "coding_agent_writing_hard_gate_role_live_llm"
_CASES_PATH = (
    Path("test_artifacts")
    / "live_gate"
    / "coding_agent_gate_01_role_inputs.json"
)
_FORBIDDEN_PM_KEYS = {
    "base_revision",
    "current_file_context",
    "diff",
    "imports",
    "owned_path",
    "owned_paths",
    "patch",
    "preferred_path",
    "read_only_paths",
    "repo_relative_path",
    "symbols_to_define",
}
_FORBIDDEN_MODULE_PM_KEYS = {
    "base_revision",
    "diff",
    "owned_path",
    "owned_paths",
    "patch",
    "read_only_paths",
    "repo_relative_path",
    "validation_trace",
}


def _cases() -> dict[str, Any]:
    text = _CASES_PATH.read_text(encoding="utf-8")
    loaded = json.loads(text)
    return loaded


async def test_role_top_pm_case_01() -> None:
    cases = _cases()
    case = cases["top_level"]["case_01"]
    pm_input = case["pm_input"]
    trace: dict[str, object] = {}

    decision = await decide_writing_work(pm_input, trace=trace)
    evaluation = _evaluate_top_pm(
        model_payload=_pm_payload(pm_input),
        decision=decision,
        trace=trace,
    )
    trace_path = write_llm_trace(
        _TEST_NAME,
        f"top_pm_{case['case_id']}",
        {
            "category": case["category"],
            "system_prompt": WRITING_PM_PROMPT,
            "pm_input": pm_input,
            "pm_trace": trace,
            "normalized_decision": decision,
            "evaluation": evaluation,
        },
    )

    print(f"TRACE_PATH={trace_path.as_posix()}")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, default=str))
    assert evaluation["status"] == "passed", "\n".join(evaluation["errors"])


async def test_role_module_pm_case_01() -> None:
    await _assert_module_pm_case("case_01")


async def test_role_module_pm_case_02() -> None:
    await _assert_module_pm_case("case_02")


async def test_role_module_pm_case_03() -> None:
    await _assert_module_pm_case("case_03")


async def test_role_programmer_case_01() -> None:
    await _assert_programmer_case("case_01")


async def test_role_programmer_case_02() -> None:
    await _assert_programmer_case("case_02")


async def _assert_module_pm_case(case_id: str) -> None:
    cases = _cases()
    case = cases["module_pm"][case_id]
    module_pm_input = case["module_pm_input"]
    trace: dict[str, object] = {}

    contract = await decide_module_programmer_contract(
        module_pm_input,
        trace=trace,
    )
    evaluation = _evaluate_module_pm(
        module_pm_input=module_pm_input,
        contract=contract,
        trace=trace,
    )
    trace_path = write_llm_trace(
        _TEST_NAME,
        f"module_pm_{case['case_id']}",
        {
            "category": case["category"],
            "system_prompt": MODULE_PM_CONTRACT_PROMPT,
            "module_pm_input": module_pm_input,
            "module_pm_trace": trace,
            "module_programmer_contract": contract,
            "evaluation": evaluation,
        },
    )

    print(f"TRACE_PATH={trace_path.as_posix()}")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, default=str))
    assert evaluation["status"] == "passed", "\n".join(evaluation["errors"])


async def _assert_programmer_case(case_id: str) -> None:
    cases = _cases()
    case = cases["programmer"][case_id]
    module_contract = case["module_contract"]
    trace: dict[str, object] = {}

    result = await run_module_programmer_contract(
        module_contract=module_contract,
        trace=trace,
    )
    evaluation = _evaluate_programmer(
        module_contract=module_contract,
        result=result,
        trace=trace,
    )
    trace_path = write_llm_trace(
        _TEST_NAME,
        f"programmer_{case['case_id']}",
        {
            "category": case["category"],
            "system_prompt": MODULE_PROGRAMMER_PROMPT,
            "module_contract": module_contract,
            "programmer_trace": trace,
            "programmer_result": result,
            "evaluation": evaluation,
        },
    )

    print(f"TRACE_PATH={trace_path.as_posix()}")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, default=str))
    assert evaluation["status"] == "passed", "\n".join(evaluation["errors"])


def _evaluate_top_pm(
    *,
    model_payload: dict[str, Any],
    decision: dict[str, Any],
    trace: dict[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    if not isinstance(trace.get("raw_output"), str) or not trace["raw_output"]:
        errors.append("Top PM returned empty raw output.")
    if not isinstance(trace.get("parsed_output"), dict):
        errors.append("Top PM output did not parse as JSON.")
    if decision.get("status") != "need_module_pms":
        errors.append("Top PM did not request Module PM work.")
    file_demands = decision.get("file_demands")
    if not isinstance(file_demands, list) or not file_demands:
        errors.append("Top PM returned no file demands.")
    leaked_input = sorted(_forbidden_keys(model_payload, _FORBIDDEN_PM_KEYS))
    if leaked_input:
        errors.append("Top PM input leaked lower-role fields: " + ", ".join(leaked_input))
    leaked_output = sorted(
        _forbidden_keys(trace.get("parsed_output"), _FORBIDDEN_PM_KEYS)
    )
    if leaked_output:
        errors.append(
            "Top PM output leaked lower-role fields: " + ", ".join(leaked_output)
        )
    return _evaluation(errors=errors)


def _evaluate_module_pm(
    *,
    module_pm_input: dict[str, Any],
    contract: dict[str, Any],
    trace: dict[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    if not isinstance(trace.get("raw_output"), str) or not trace["raw_output"]:
        errors.append("Module PM returned empty raw output.")
    if not isinstance(trace.get("parsed_output"), dict):
        errors.append("Module PM output did not parse as JSON.")
    for field_name in ("file_label", "edit_mode", "content_format"):
        if contract.get(field_name) != module_pm_input[field_name]:
            errors.append(f"Module PM changed {field_name}.")
    if not isinstance(contract.get("imports"), list):
        errors.append("Module PM imports is not a list.")
    if not isinstance(contract.get("symbols_to_define"), list):
        errors.append("Module PM symbols_to_define is not a list.")
    elif not contract["symbols_to_define"]:
        errors.append("Module PM returned no symbols_to_define.")
    if not isinstance(contract.get("required_behavior"), list):
        errors.append("Module PM required_behavior is not a list.")
    leaked = sorted(_forbidden_keys(contract, _FORBIDDEN_MODULE_PM_KEYS))
    if leaked:
        errors.append("Module PM leaked non-programmer fields: " + ", ".join(leaked))
    return _evaluation(errors=errors)


def _evaluate_programmer(
    *,
    module_contract: dict[str, Any],
    result: dict[str, object],
    trace: dict[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    if trace.get("markdown_code_block_found") is not True:
        errors.append("Programmer did not return exactly one accepted fenced block.")
    content = result.get("code_artifact")
    if not isinstance(content, str) or not content.strip():
        errors.append("Programmer returned no source content.")
        return _evaluation(errors=errors)
    if module_contract.get("content_format") == "python":
        errors.extend(_python_errors(content, module_contract))
    else:
        if "```" in content:
            errors.append("Text output retained markdown fence markers.")
    return _evaluation(errors=errors)


def _python_errors(content: str, module_contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        module = ast.parse(content)
    except SyntaxError as exc:
        return [f"Python source is not valid: {exc}"]
    defined = _defined_top_level_symbols(module)
    expected = _expected_symbols(module_contract)
    missing = sorted(expected - defined)
    if missing:
        errors.append("Python output missed expected symbols: " + ", ".join(missing))
    if "TODO" in content or "NotImplementedError" in content:
        errors.append("Python output contains placeholder implementation text.")
    return errors


def _expected_symbols(module_contract: dict[str, Any]) -> set[str]:
    symbols = module_contract.get("symbols_to_define")
    if not isinstance(symbols, list):
        return set()
    names: set[str] = set()
    for symbol in symbols:
        if not isinstance(symbol, dict):
            continue
        name = symbol.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _defined_top_level_symbols(module: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            names.update(_target_names(node.target))
    return names


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_target_names(element))
        return names
    return set()


def _forbidden_keys(value: object, forbidden_keys: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden_keys:
                found.add(key)
            found.update(_forbidden_keys(child, forbidden_keys))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_keys(item, forbidden_keys))
    return found


def _evaluation(*, errors: list[str]) -> dict[str, object]:
    status = "failed" if errors else "passed"
    return {
        "status": status,
        "errors": errors,
    }
