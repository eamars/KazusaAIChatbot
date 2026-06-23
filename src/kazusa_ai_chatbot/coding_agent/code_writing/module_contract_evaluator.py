"""Structural evaluation for File-PM module programmer contracts."""

from __future__ import annotations

import ast
import json

from kazusa_ai_chatbot.coding_agent.code_writing.models import (
    ModuleContractEvaluation,
    ModuleProgrammerContract,
    WritingFileModuleContract,
)

MAX_MODULE_CONTRACT_ERRORS = 12
ALLOWED_EDIT_MODES = {"complete_file", "symbol_bundle"}
ALLOWED_CONTENT_FORMATS = {"python", "text"}
FORBIDDEN_PROGRAMMER_FIELDS = {
    "base_revision",
    "diff",
    "file_contract_id",
    "mutex_id",
    "owned_path",
    "owned_paths",
    "patch",
    "patch_hunks",
    "patch_location",
    "peer_output",
    "read_only_paths",
    "repo_relative_path",
    "validation_trace",
    "write_path",
}


def evaluate_module_contract(
    *,
    file_contract: WritingFileModuleContract,
    module_contract: ModuleProgrammerContract,
    file_pm_input: dict[str, object] | None = None,
) -> ModuleContractEvaluation:
    """Validate a File PM contract before programmer dispatch."""

    errors: list[str] = []
    file_contract_id = file_contract.get("file_contract_id", "")
    file_label = module_contract.get("file_label", "")

    if not file_contract_id:
        errors.append("File contract has no file_contract_id.")
    if not _clean_string(file_label):
        errors.append("Module programmer contract has no file_label.")
    if module_contract.get("edit_mode") not in ALLOWED_EDIT_MODES:
        errors.append("Module programmer contract has unsupported edit_mode.")
    if module_contract.get("content_format") not in ALLOWED_CONTENT_FORMATS:
        errors.append(
            "Module programmer contract has unsupported content_format."
        )
    if not _clean_string(module_contract.get("file_purpose")):
        errors.append("Module programmer contract has no file_purpose.")
    if not isinstance(module_contract.get("imports"), list):
        errors.append("Module programmer contract imports is not a list.")
    if not isinstance(module_contract.get("current_file_context"), str):
        errors.append(
            "Module programmer contract current_file_context is not a string."
        )
    errors.extend(_symbol_errors(module_contract.get("symbols_to_define")))
    if module_contract.get("content_format") == "python":
        errors.extend(_python_contract_errors(module_contract))
        errors.extend(_project_import_grounding_errors(
            module_contract,
            file_pm_input=file_pm_input,
        ))
    if not _string_values(module_contract.get("required_behavior")):
        errors.append("Module programmer contract has no required_behavior.")

    leaked_fields = sorted(_forbidden_keys(module_contract))
    if leaked_fields:
        errors.append(
            "Module programmer contract leaks non-programmer fields: "
            + ", ".join(leaked_fields)
        )

    limited_errors = errors[:MAX_MODULE_CONTRACT_ERRORS]
    status = "accepted"
    if limited_errors:
        status = "repair_required"
    evaluation: ModuleContractEvaluation = {
        "status": status,
        "file_contract_id": file_contract_id,
        "file_label": file_label,
        "errors": limited_errors,
        "repair_feedback": _repair_feedback(limited_errors),
    }
    return evaluation


def _python_contract_errors(
    module_contract: ModuleProgrammerContract,
) -> list[str]:
    errors: list[str] = []
    for import_line in _string_values(module_contract.get("imports")):
        try:
            parsed_import = ast.parse(import_line)
        except SyntaxError:
            errors.append(f"Import line is not valid Python: {import_line}")
            continue
        if not parsed_import.body or not isinstance(
            parsed_import.body[0],
            (ast.Import, ast.ImportFrom),
        ):
            errors.append(f"Import line is not an import statement: {import_line}")

    symbols = module_contract.get("symbols_to_define")
    if not isinstance(symbols, list):
        return errors
    for symbol in symbols:
        if not isinstance(symbol, dict):
            continue
        signature = _clean_string(symbol.get("signature"))
        if not signature:
            continue
        parse_text = _parseable_signature_text(signature)
        if not parse_text:
            continue
        try:
            ast.parse(parse_text)
        except SyntaxError:
            name = _clean_string(symbol.get("name")) or signature
            errors.append(f"Symbol {name!r} signature is not valid Python.")
    return errors


def _parseable_signature_text(signature: str) -> str:
    if signature.startswith(("def ", "async def ", "class ")):
        return signature.rstrip(":") + ":\n    pass"
    if "=" in signature:
        return signature
    return ""


def _project_import_grounding_errors(
    module_contract: ModuleProgrammerContract,
    *,
    file_pm_input: dict[str, object] | None,
) -> list[str]:
    if file_pm_input is None:
        return []

    allowed_import_lines = set(_string_values(file_pm_input.get("imports")))
    grounding_text = json.dumps(file_pm_input, ensure_ascii=False)
    errors: list[str] = []
    for import_line in _string_values(module_contract.get("imports")):
        if not _is_project_import(import_line):
            continue
        if import_line in allowed_import_lines:
            continue
        imported_names = _imported_names(import_line)
        missing_names = [
            name for name in imported_names
            if name and name not in grounding_text
        ]
        if missing_names:
            errors.append(
                "Project import is not grounded in the File PM input context: "
                + import_line
            )
    return errors


def _is_project_import(import_line: str) -> bool:
    try:
        parsed_import = ast.parse(import_line)
    except SyntaxError:
        return False
    if not parsed_import.body or not isinstance(parsed_import.body[0], ast.ImportFrom):
        return False
    module = parsed_import.body[0].module or ""
    return module.startswith("kazusa_ai_chatbot")


def _imported_names(import_line: str) -> list[str]:
    try:
        parsed_import = ast.parse(import_line)
    except SyntaxError:
        return []
    if not parsed_import.body or not isinstance(parsed_import.body[0], ast.ImportFrom):
        return []
    return [alias.name for alias in parsed_import.body[0].names]


def _symbol_errors(value: object) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, list) or not value:
        return ["Module programmer contract has no symbols_to_define."]

    for index, symbol in enumerate(value, start=1):
        if not isinstance(symbol, dict):
            errors.append(f"Symbol {index} is not an object.")
            continue
        for field_name in ("name", "kind", "signature", "body_contract"):
            if not _clean_string(symbol.get(field_name)):
                errors.append(f"Symbol {index} has no {field_name}.")
        children = symbol.get("children")
        if children is not None and not isinstance(children, list):
            errors.append(f"Symbol {index} children is not a list.")
    return errors


def _forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PROGRAMMER_FIELDS:
                found.add(key)
            found.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_keys(item))
    return found


def _string_values(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def _clean_string(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _repair_feedback(errors: list[str]) -> list[str]:
    if not errors:
        return []
    return [
        "Return one corrected module programmer contract for the same file.",
        *errors,
    ]
