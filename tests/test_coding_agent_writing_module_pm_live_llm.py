"""Real-LLM role tests for File PM ideal inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.coding_agent.code_writing.module_product_manager import (
    FILE_PM_MODULE_CONTRACT_PROMPT,
    decide_module_programmer_contract,
)
from tests.llm_trace import write_llm_trace


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


_TEST_NAME = "coding_agent_writing_file_pm_ideal_input_live_llm"
_CASES_PATH = (
    Path("test_artifacts")
    / "live_gate"
    / "coding_agent_pm_ideal_inputs.json"
)
_EDIT_MODES = {"complete_file", "symbol_bundle"}
_FORBIDDEN_FILE_PM_KEYS = {
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
    """Load the File PM ideal-input fixture."""

    text = _CASES_PATH.read_text(encoding="utf-8")
    loaded = json.loads(text)
    return loaded


async def _run_case(case_id: str) -> tuple[dict[str, object], Path]:
    """Run one File PM live case and write a durable trace."""

    cases = _load_cases()
    case = cases["file_pm"][case_id]
    file_pm_input = case["file_pm_input"]
    trace: dict[str, object] = {}

    contract = await decide_module_programmer_contract(
        file_pm_input,
        trace=trace,
    )
    evaluation = _evaluate_file_pm_contract(
        file_pm_input=file_pm_input,
        contract=contract,
        trace=trace,
    )
    trace_path = write_llm_trace(
        _TEST_NAME,
        case["case_id"],
        {
            "category": case["category"],
            "system_prompt": FILE_PM_MODULE_CONTRACT_PROMPT,
            "file_pm_input": file_pm_input,
            "file_pm_trace": trace,
            "module_programmer_contract": contract,
            "evaluation": evaluation,
        },
    )
    return evaluation, trace_path


def _evaluate_file_pm_contract(
    *,
    file_pm_input: dict[str, Any],
    contract: dict[str, Any],
    trace: dict[str, object],
) -> dict[str, object]:
    """Evaluate one File PM result against the module contract boundary."""

    errors: list[str] = []
    warnings: list[str] = []
    raw_output = trace.get("raw_output")
    parsed_output = trace.get("parsed_output")

    if not isinstance(raw_output, str) or not raw_output.strip():
        errors.append("File PM returned empty raw output.")
    if not isinstance(parsed_output, dict):
        errors.append("File PM raw output did not parse as a JSON object.")

    if contract.get("file_label") != file_pm_input["file_label"]:
        errors.append("File PM did not preserve the input file_label.")
    if contract.get("edit_mode") not in _EDIT_MODES:
        errors.append("File PM returned an unsupported edit_mode.")
    if contract.get("edit_mode") != file_pm_input["edit_mode"]:
        errors.append("File PM changed the requested edit_mode.")
    if contract.get("content_format") != file_pm_input["content_format"]:
        errors.append("File PM changed the requested content_format.")
    if not _clean_string(contract.get("file_purpose")):
        errors.append("File PM returned no file_purpose.")

    contract_imports = _string_list(contract.get("imports"))
    missing_imports = [
        value
        for value in _string_list(file_pm_input.get("imports"))
        if value not in contract_imports
    ]
    if missing_imports:
        errors.append(
            "File PM omitted required imports: " + "; ".join(missing_imports)
        )

    current_context = contract.get("current_file_context")
    if not isinstance(current_context, str):
        errors.append("File PM current_file_context is not a string.")

    symbols = contract.get("symbols_to_define")
    if not isinstance(symbols, list) or not symbols:
        errors.append("File PM returned no symbols_to_define.")
        symbol_names: set[str] = set()
    else:
        symbol_names = _symbol_names(symbols)
        errors.extend(_symbol_shape_errors(symbols))

    missing_outputs = [
        output_name
        for output_name in _string_list(file_pm_input.get("module_outputs"))
        if output_name not in symbol_names
    ]
    if missing_outputs:
        errors.append(
            "File PM did not pass required module outputs to the programmer: "
            + ", ".join(missing_outputs)
        )

    if not isinstance(contract.get("required_behavior"), list):
        errors.append("File PM required_behavior is not a list.")
    elif not contract["required_behavior"]:
        errors.append("File PM returned no required_behavior.")

    leaked_keys = sorted(_forbidden_keys(contract, _FORBIDDEN_FILE_PM_KEYS))
    if leaked_keys:
        errors.append(
            "File PM emitted fields owned by another role: "
            + ", ".join(leaked_keys)
        )

    if len(symbol_names) > len(_string_list(file_pm_input.get("module_outputs"))) + 4:
        warnings.append(
            "File PM produced many extra symbols; review whether the contract "
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
    """Return top-level symbol names from a File PM contract."""

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
    """Run one File PM live case and assert structural role gates."""

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
