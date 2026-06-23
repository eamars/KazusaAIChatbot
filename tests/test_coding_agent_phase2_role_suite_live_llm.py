"""Phase 2 replacement plan role-suite live LLM tests.

Loads role inputs from the canonical gate-derived fixture:
    test_artifacts/live_gate/coding_agent_phase2_role_suite.json

Each test function runs one role case from one gate.
Run one at a time with ``pytest -q -s``.
"""

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


_SUITE_PATH = (
    Path("test_artifacts")
    / "live_gate"
    / "coding_agent_phase2_role_suite.json"
)
_TEST_NAME_PREFIX = "phase2_role_suite"


# ---------------------------------------------------------------------------
# Fixture loader
# ---------------------------------------------------------------------------

def _load_suite() -> dict[str, Any]:
    text = _SUITE_PATH.read_text(encoding="utf-8")
    return json.loads(text)


def _role_case(gate_id: str, role: str, case_id: str) -> dict[str, Any]:
    suite = _load_suite()
    return suite[gate_id]["roles"][role][case_id]


# ---------------------------------------------------------------------------
# Top-level writing PM evaluator
# ---------------------------------------------------------------------------

_ALLOWED_PM_STATUSES = {
    "need_reading",
    "need_module_pms",
    "ready_to_write",
    "needs_user_input",
    "overloaded",
    "rejected",
}
_FORBIDDEN_TOP_PM_OUTPUT_KEYS = {
    "base_revision", "code_artifact", "diff", "file_content", "mutex_id",
    "owned_path", "owned_paths", "patch", "patch_hunks", "repo_relative_path",
    "symbols_to_define", "unified_diff", "current_file_context", "imports",
    "placement_hint", "preferred_path", "questions",
    "read_only_paths", "related_paths",
}


def _evaluate_top_pm(
    *,
    decision: dict[str, Any],
    trace: dict[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    raw_output = trace.get("raw_output")
    parsed_output = trace.get("parsed_output")

    if not isinstance(raw_output, str) or not raw_output.strip():
        errors.append("Top PM returned empty raw output.")
    if not isinstance(parsed_output, dict):
        errors.append("Top PM raw output did not parse as JSON object.")

    status = decision.get("status")
    if status not in _ALLOWED_PM_STATUSES:
        errors.append(f"Top PM returned unsupported status {status!r}.")
    if status != "need_module_pms":
        warnings.append(f"Top PM status was {status!r}, not need_module_pms.")

    file_demands = decision.get("file_demands")
    if not isinstance(file_demands, list) or not file_demands:
        errors.append("Top PM returned no file_demands.")
    else:
        for idx, demand in enumerate(file_demands, start=1):
            if not isinstance(demand, dict):
                errors.append(f"File demand {idx} is not a dict.")
                continue
            if not _clean(demand.get("purpose")):
                errors.append(f"File demand {idx} has no purpose.")
            if not isinstance(demand.get("interface_contract"), dict):
                errors.append(f"File demand {idx} has no interface_contract.")
            if not isinstance(demand.get("integration_contract"), dict):
                errors.append(f"File demand {idx} has no integration_contract.")

    leaked = sorted(_forbidden_keys(parsed_output, _FORBIDDEN_TOP_PM_OUTPUT_KEYS))
    if leaked:
        errors.append("Top PM emitted lower-role keys: " + ", ".join(leaked))

    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "file_demand_count": len(file_demands) if isinstance(file_demands, list) else 0,
        "model": str(trace.get("model", "")),
        "thinking_enabled": bool(trace.get("thinking_enabled")),
    }


async def _run_top_pm_case(gate_id: str, case_key: str) -> None:
    case = _role_case(gate_id, "top_level_writing_pm", case_key)
    pm_input = case["pm_input"]
    trace: dict[str, object] = {}
    decision = await decide_writing_work(pm_input, trace=trace)
    evaluation = _evaluate_top_pm(decision=decision, trace=trace)
    trace_path = write_llm_trace(
        f"{_TEST_NAME_PREFIX}_top_pm",
        case["case_id"],
        {
            "gate_id": gate_id,
            "category": case["category"],
            "system_prompt": WRITING_PM_PROMPT,
            "pm_input": pm_input,
            "model_payload": _pm_payload(pm_input),
            "pm_trace": trace,
            "decision": decision,
            "evaluation": evaluation,
        },
    )
    print(f"TRACE_PATH={trace_path.as_posix()}")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, default=str))
    assert evaluation["status"] == "passed", "\n".join(evaluation["errors"])


# ---------------------------------------------------------------------------
# Module PM evaluator
# ---------------------------------------------------------------------------

_FORBIDDEN_MODULE_PM_KEYS = {
    "base_revision", "dependencies", "dependency", "diff", "file_contract_id",
    "insertion_point", "mutex_id", "owned_path", "owned_paths", "patch",
    "patch_hunks", "patch_location", "peer_output", "read_path",
    "repo_relative_path", "repo_url", "test_path", "validation_trace",
    "write_path",
}
_EDIT_MODES = {"complete_file", "symbol_bundle"}


def _evaluate_module_pm(
    *,
    module_pm_input: dict[str, Any],
    contract: dict[str, Any],
    trace: dict[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    raw_output = trace.get("raw_output")
    parsed_output = trace.get("parsed_output")

    if not isinstance(raw_output, str) or not raw_output.strip():
        errors.append("Module PM returned empty raw output.")
    if not isinstance(parsed_output, dict):
        errors.append("Module PM raw output did not parse as JSON.")

    if contract.get("file_label") != module_pm_input["file_label"]:
        errors.append("Module PM did not preserve file_label.")
    if contract.get("edit_mode") not in _EDIT_MODES:
        errors.append("Module PM returned unsupported edit_mode.")
    if contract.get("edit_mode") != module_pm_input["edit_mode"]:
        errors.append("Module PM changed the requested edit_mode.")
    if contract.get("content_format") != module_pm_input["content_format"]:
        errors.append("Module PM changed the requested content_format.")
    if not _clean(contract.get("module_purpose")):
        errors.append("Module PM returned no module_purpose.")
    if not _clean(contract.get("lifecycle_owner")):
        errors.append("Module PM returned no lifecycle_owner.")

    contract_imports = _string_list(contract.get("imports"))
    missing_imports = [
        v for v in _string_list(module_pm_input.get("imports"))
        if v not in contract_imports
    ]
    if missing_imports:
        errors.append("Module PM omitted imports: " + "; ".join(missing_imports))

    symbols_define = contract.get("symbols_to_define")
    symbols_modify = contract.get("symbols_to_modify")
    has_define = isinstance(symbols_define, list) and len(symbols_define) > 0
    has_modify = isinstance(symbols_modify, list) and len(symbols_modify) > 0
    if not has_define and not has_modify:
        errors.append("Module PM returned no symbols_to_define or symbols_to_modify.")
    else:
        if has_define:
            errors.extend(_symbol_shape_errors(symbols_define))
        if has_modify:
            errors.extend(_symbol_shape_errors(symbols_modify))

    if not isinstance(contract.get("required_behavior"), list):
        errors.append("Module PM required_behavior is not a list.")
    elif not contract["required_behavior"]:
        errors.append("Module PM returned no required_behavior.")

    leaked = sorted(_forbidden_keys(contract, _FORBIDDEN_MODULE_PM_KEYS))
    if leaked:
        errors.append("Module PM emitted forbidden keys: " + ", ".join(leaked))

    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "model": str(trace.get("model", "")),
        "thinking_enabled": bool(trace.get("thinking_enabled")),
    }


async def _run_module_pm_case(gate_id: str, case_key: str) -> None:
    case = _role_case(gate_id, "module_pm", case_key)
    module_pm_input = case["module_pm_input"]
    trace: dict[str, object] = {}
    contract = await decide_module_programmer_contract(module_pm_input, trace=trace)
    evaluation = _evaluate_module_pm(
        module_pm_input=module_pm_input, contract=contract, trace=trace,
    )
    trace_path = write_llm_trace(
        f"{_TEST_NAME_PREFIX}_module_pm",
        case["case_id"],
        {
            "gate_id": gate_id,
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


# ---------------------------------------------------------------------------
# Module programmer evaluator
# ---------------------------------------------------------------------------

def _evaluate_programmer(
    *,
    module_contract: dict[str, Any],
    result: dict[str, object],
    trace: dict[str, object],
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    raw_output = trace.get("raw_output")

    if not isinstance(raw_output, str) or not raw_output.strip():
        errors.append("Programmer returned empty raw output.")
    if trace.get("markdown_code_block_found") is not True:
        errors.append("Programmer did not return exactly one code block.")

    code_artifact = result.get("code_artifact")
    if not isinstance(code_artifact, str) or not code_artifact.strip():
        errors.append("Programmer produced no code_artifact.")
        return _programmer_eval(errors=errors, warnings=warnings)

    content_format = module_contract.get("content_format", "python")
    if content_format == "python":
        try:
            module = ast.parse(code_artifact)
        except SyntaxError as exc:
            errors.append(f"code_artifact is not valid Python: {exc}")
            return _programmer_eval(
                errors=errors, warnings=warnings,
                code_chars=len(code_artifact),
            )
        expected = _expected_symbols(module_contract)
        defined = _defined_top_level(module)
        missing = sorted(expected - defined)
        if missing:
            errors.append("Missing symbols: " + ", ".join(missing))
        errors.extend(_placeholder_errors(module, code_artifact))

    return _programmer_eval(
        errors=errors, warnings=warnings,
        code_chars=len(code_artifact) if isinstance(code_artifact, str) else 0,
        model=str(trace.get("model", "")),
        thinking_enabled=bool(trace.get("thinking_enabled")),
    )


def _programmer_eval(
    *,
    errors: list[str],
    warnings: list[str],
    code_chars: int = 0,
    model: str = "",
    thinking_enabled: bool = False,
) -> dict[str, object]:
    return {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "code_chars": code_chars,
        "model": model,
        "thinking_enabled": thinking_enabled,
    }


async def _run_programmer_case(gate_id: str, case_key: str) -> None:
    case = _role_case(gate_id, "module_programmer", case_key)
    module_contract = case["module_contract"]
    trace: dict[str, object] = {}
    result = await run_module_programmer_contract(
        module_contract=module_contract, trace=trace,
    )
    evaluation = _evaluate_programmer(
        module_contract=module_contract, result=result, trace=trace,
    )
    trace_path = write_llm_trace(
        f"{_TEST_NAME_PREFIX}_programmer",
        case["case_id"],
        {
            "gate_id": gate_id,
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _clean(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_clean(item) for item in value if _clean(item)]


def _forbidden_keys(value: object, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                found.add(key)
            found.update(_forbidden_keys(child, forbidden))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_keys(item, forbidden))
    return found


def _symbol_shape_errors(symbols: list[object]) -> list[str]:
    errors: list[str] = []
    for idx, sym in enumerate(symbols, start=1):
        if not isinstance(sym, dict):
            errors.append(f"Symbol {idx} is not a dict.")
            continue
        if not _clean(sym.get("name")):
            errors.append(f"Symbol {idx} has no name.")
        if not _clean(sym.get("kind")):
            errors.append(f"Symbol {idx} has no kind.")
        if not _clean(sym.get("body_contract")):
            errors.append(f"Symbol {idx} has no body_contract.")
    return errors


def _expected_symbols(module_contract: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("symbols_to_define", "symbols_to_modify"):
        symbols = module_contract.get(key)
        if not isinstance(symbols, list):
            continue
        for sym in symbols:
            if isinstance(sym, dict):
                name = _clean(sym.get("name"))
                if name:
                    names.add(name)
    return names


def _defined_top_level(module: ast.Module) -> set[str]:
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
        for elt in target.elts:
            names.update(_target_names(elt))
        return names
    return set()


def _placeholder_errors(module: ast.Module, code: str) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Pass):
            errors.append("code_artifact contains pass.")
            break
    for node in ast.walk(module):
        if isinstance(node, ast.Expr) and isinstance(
            node.value, ast.Constant
        ) and node.value.value is Ellipsis:
            errors.append("code_artifact contains ellipsis.")
            break
    for node in ast.walk(module):
        if isinstance(node, ast.Raise):
            exc = node.exc
            if isinstance(exc, ast.Call):
                exc = exc.func
            if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
                errors.append("code_artifact raises NotImplementedError.")
                break
    if "TODO" in code:
        errors.append("code_artifact contains TODO.")
    return errors


# ===========================================================================
# Gate 01 test cases
# ===========================================================================

async def test_gate01_top_pm_01() -> None:
    await _run_top_pm_case("gate_01", "case_01")

async def test_gate01_module_pm_01() -> None:
    await _run_module_pm_case("gate_01", "case_01")

async def test_gate01_module_pm_02() -> None:
    await _run_module_pm_case("gate_01", "case_02")

async def test_gate01_module_pm_03() -> None:
    await _run_module_pm_case("gate_01", "case_03")

async def test_gate01_programmer_01() -> None:
    await _run_programmer_case("gate_01", "case_01")

async def test_gate01_programmer_02() -> None:
    await _run_programmer_case("gate_01", "case_02")


# ===========================================================================
# Gate 02 test cases
# ===========================================================================

async def test_gate02_top_pm_01() -> None:
    await _run_top_pm_case("gate_02", "case_01")

async def test_gate02_module_pm_01() -> None:
    await _run_module_pm_case("gate_02", "case_01")

async def test_gate02_module_pm_02() -> None:
    await _run_module_pm_case("gate_02", "case_02")

async def test_gate02_programmer_01() -> None:
    await _run_programmer_case("gate_02", "case_01")


# ===========================================================================
# Gate 03 test cases
# ===========================================================================

async def test_gate03_top_pm_01() -> None:
    await _run_top_pm_case("gate_03", "case_01")

async def test_gate03_module_pm_01() -> None:
    await _run_module_pm_case("gate_03", "case_01")

async def test_gate03_programmer_01() -> None:
    await _run_programmer_case("gate_03", "case_01")


# ===========================================================================
# Gate 04 test cases
# ===========================================================================

async def test_gate04_top_pm_01() -> None:
    await _run_top_pm_case("gate_04", "case_01")

async def test_gate04_module_pm_01() -> None:
    await _run_module_pm_case("gate_04", "case_01")

async def test_gate04_programmer_01() -> None:
    await _run_programmer_case("gate_04", "case_01")


# ===========================================================================
# Gate 05 test cases
# ===========================================================================

async def test_gate05_top_pm_01() -> None:
    await _run_top_pm_case("gate_05", "case_01")

async def test_gate05_module_pm_01() -> None:
    await _run_module_pm_case("gate_05", "case_01")

async def test_gate05_module_pm_02() -> None:
    await _run_module_pm_case("gate_05", "case_02")

async def test_gate05_module_pm_03() -> None:
    await _run_module_pm_case("gate_05", "case_03")

async def test_gate05_module_pm_04() -> None:
    await _run_module_pm_case("gate_05", "case_04")

async def test_gate05_module_pm_05() -> None:
    await _run_module_pm_case("gate_05", "case_05")

async def test_gate05_module_pm_06() -> None:
    await _run_module_pm_case("gate_05", "case_06")

async def test_gate05_programmer_01() -> None:
    await _run_programmer_case("gate_05", "case_01")

async def test_gate05_programmer_02() -> None:
    await _run_programmer_case("gate_05", "case_02")
