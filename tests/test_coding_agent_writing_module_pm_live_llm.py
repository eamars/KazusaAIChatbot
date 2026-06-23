"""Real-LLM role tests for Module PM ideal inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.coding_agent.code_writing.module_product_manager import (
    MODULE_PM_CONTRACT_PROMPT,
    decide_module_programmer_contract,
)
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


_TEST_NAME = "coding_agent_writing_module_pm_ideal_input_live_llm"
_CASES_PATH = (
    Path("test_artifacts")
    / "live_gate"
    / "coding_agent_pm_ideal_inputs.json"
)
_EDIT_MODES = {"complete_file", "symbol_bundle"}
_FORBIDDEN_MODULE_PM_KEYS = {
    "base_revision",
    "dependencies",
    "dependency",
    "diff",
    "file_contract_id",
    "insertion_point",
    "mutex_id",
    "owned_path",
    "owned_paths",
    "patch",
    "patch_hunks",
    "patch_location",
    "peer_output",
    "read_path",
    "repo_relative_path",
    "repo_url",
    "test_path",
    "validation_trace",
    "write_path",
}


def _load_cases() -> dict[str, Any]:
    """Load the Module PM ideal-input fixture."""

    text = _CASES_PATH.read_text(encoding="utf-8")
    loaded = json.loads(text)
    return loaded


async def _run_case(case_id: str) -> tuple[dict[str, object], Path]:
    """Run one Module PM live case and write a durable trace."""

    cases = _load_cases()
    case = cases["module_pm"][case_id]
    module_pm_input = case["module_pm_input"]
    trace: dict[str, object] = {}

    contract = await decide_module_programmer_contract(
        module_pm_input,
        trace=trace,
    )
    evaluation = _evaluate_module_pm_contract(
        module_pm_input=module_pm_input,
        contract=contract,
        trace=trace,
    )
    trace_path = write_llm_trace(
        _TEST_NAME,
        case["case_id"],
        {
            "category": case["category"],
            "system_prompt": MODULE_PM_CONTRACT_PROMPT,
            "module_pm_input": module_pm_input,
            "module_pm_trace": trace,
            "module_programmer_contract": contract,
            "evaluation": evaluation,
        },
    )
    return evaluation, trace_path


def _evaluate_module_pm_contract(
    *,
    module_pm_input: dict[str, Any],
    contract: dict[str, Any],
    trace: dict[str, object],
) -> dict[str, object]:
    """Evaluate one Module PM result against the module contract boundary."""

    errors: list[str] = []
    warnings: list[str] = []
    raw_output = trace.get("raw_output")
    parsed_output = trace.get("parsed_output")

    if not isinstance(raw_output, str) or not raw_output.strip():
        errors.append("Module PM returned empty raw output.")
    if not isinstance(parsed_output, dict):
        errors.append("Module PM raw output did not parse as a JSON object.")

    if contract.get("file_label") != module_pm_input["file_label"]:
        errors.append("Module PM did not preserve the input file_label.")
    if contract.get("edit_mode") not in _EDIT_MODES:
        errors.append("Module PM returned an unsupported edit_mode.")
    if contract.get("edit_mode") != module_pm_input["edit_mode"]:
        errors.append("Module PM changed the requested edit_mode.")
    if contract.get("content_format") != module_pm_input["content_format"]:
        errors.append("Module PM changed the requested content_format.")
    if not _clean_string(contract.get("module_purpose")):
        errors.append("Module PM returned no module_purpose.")
    if not _clean_string(contract.get("lifecycle_owner")):
        errors.append("Module PM returned no lifecycle_owner.")

    contract_imports = _string_list(contract.get("imports"))
    missing_imports = [
        value
        for value in _string_list(module_pm_input.get("imports"))
        if value not in contract_imports
    ]
    if missing_imports:
        errors.append(
            "Module PM omitted required imports: " + "; ".join(missing_imports)
        )

    current_context = contract.get("current_file_context")
    if not isinstance(current_context, str):
        errors.append("Module PM current_file_context is not a string.")

    symbols_define = contract.get("symbols_to_define")
    symbols_modify = contract.get("symbols_to_modify")
    has_define = isinstance(symbols_define, list) and len(symbols_define) > 0
    has_modify = isinstance(symbols_modify, list) and len(symbols_modify) > 0
    if not has_define and not has_modify:
        errors.append(
            "Module PM returned no symbols_to_define or symbols_to_modify."
        )
        symbol_names: set[str] = set()
    else:
        symbol_names = set()
        if has_define:
            symbol_names.update(_symbol_names(symbols_define))
            errors.extend(_symbol_shape_errors(symbols_define))
        if has_modify:
            symbol_names.update(_symbol_names(symbols_modify))
            errors.extend(_symbol_shape_errors(symbols_modify))

    provided_names = [
        ifc.get("name", "") if isinstance(ifc, dict) else ""
        for ifc in (module_pm_input.get("provided_interfaces") or [])
    ]
    missing_outputs = [
        name for name in provided_names
        if name and name not in symbol_names
    ]
    if missing_outputs:
        warnings.append(
            "Module PM did not cover all provided_interfaces in symbols: "
            + ", ".join(missing_outputs)
        )

    if not isinstance(contract.get("required_behavior"), list):
        errors.append("Module PM required_behavior is not a list.")
    elif not contract["required_behavior"]:
        errors.append("Module PM returned no required_behavior.")

    leaked_keys = sorted(_forbidden_keys(contract, _FORBIDDEN_MODULE_PM_KEYS))
    if leaked_keys:
        errors.append(
            "Module PM emitted fields owned by another role: "
            + ", ".join(leaked_keys)
        )

    if len(symbol_names) > len(provided_names) + 4:
        warnings.append(
            "Module PM produced many extra symbols; review whether the contract "
            "is too broad for one programmer."
        )

    status_text = "failed" if errors else "passed"
    evaluation = {
        "status": status_text,
        "errors": errors,
        "warnings": warnings,
        "symbol_names": sorted(symbol_names),
        "model": str(trace.get("model", "")),
        "thinking_enabled": bool(trace.get("thinking_enabled")),
    }
    return evaluation


def _symbol_shape_errors(symbols: list[object]) -> list[str]:
    """Return structural errors for symbols_to_define rows."""

    errors: list[str] = []
    for index, symbol in enumerate(symbols, start=1):
        if not isinstance(symbol, dict):
            errors.append(f"Symbol {index} is not a JSON object.")
            continue
        if not _clean_string(symbol.get("name")):
            errors.append(f"Symbol {index} has no name.")
        if not _clean_string(symbol.get("kind")):
            errors.append(f"Symbol {index} has no kind.")
        if not _clean_string(symbol.get("signature")):
            errors.append(f"Symbol {index} has no signature.")
        if not _clean_string(symbol.get("body_contract")):
            errors.append(f"Symbol {index} has no body_contract.")
        children = symbol.get("children")
        if children is not None and not isinstance(children, list):
            errors.append(f"Symbol {index} children is not a list.")
    return errors


def _symbol_names(symbols: list[object]) -> set[str]:
    """Return top-level symbol names from a Module PM contract."""

    names: set[str] = set()
    for symbol in symbols:
        if not isinstance(symbol, dict):
            continue
        name = _clean_string(symbol.get("name"))
        if name:
            names.add(name)
    return names


def _forbidden_keys(value: object, forbidden_keys: set[str]) -> set[str]:
    """Return forbidden keys found anywhere inside a role output."""

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


def _string_list(value: object) -> list[str]:
    """Return clean string values from a list-like field."""

    if not isinstance(value, list):
        return []
    strings: list[str] = []
    for item in value:
        text = _clean_string(item)
        if text:
            strings.append(text)
    return strings


def _clean_string(value: object) -> str:
    """Return a stripped string or an empty value."""

    if not isinstance(value, str):
        return ""
    cleaned = value.strip()
    return cleaned


async def _assert_case(case_id: str) -> None:
    """Run one Module PM live case and assert structural role gates."""

    evaluation, trace_path = await _run_case(case_id)
    print(f"TRACE_PATH={trace_path.as_posix()}")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, default=str))
    assert evaluation["status"] == "passed", "\n".join(evaluation["errors"])


async def test_phase2_role_04_case_01() -> None:
    await _assert_case("case_01")


async def test_phase2_role_04_case_02() -> None:
    await _assert_case("case_02")


async def test_phase2_role_04_case_03() -> None:
    await _assert_case("case_03")


async def test_phase2_role_04_case_04() -> None:
    await _assert_case("case_04")
