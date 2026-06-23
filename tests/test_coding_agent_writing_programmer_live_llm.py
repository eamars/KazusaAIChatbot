"""Real-LLM tests for module-level programmer ideal inputs."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pytest

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


_TEST_NAME = "coding_agent_writing_module_programmer_ideal_input_live_llm"
_CASES_PATH = (
    Path("test_artifacts")
    / "live_gate"
    / "coding_agent_module_programmer_ideal_inputs.json"
)


def _load_cases() -> dict[str, dict[str, Any]]:
    """Load module programmer ideal-input cases."""

    text = _CASES_PATH.read_text(encoding="utf-8")
    cases = json.loads(text)
    return cases


def _load_case(case_id: str) -> dict[str, Any]:
    """Load one module programmer ideal-input case."""

    cases = _load_cases()
    case = cases[case_id]
    return case


async def _run_case(case_id: str) -> tuple[dict[str, object], Path]:
    """Run one live module-programmer case and write a trace."""

    case = _load_case(case_id)
    module_contract = case["module_contract"]
    trace: dict[str, object] = {}
    result = await run_module_programmer_contract(
        module_contract=module_contract,
        trace=trace,
    )
    evaluation = _evaluate_module_result(
        module_contract=module_contract,
        result=result,
        trace=trace,
    )
    trace_payload = {
        "case_id": case_id,
        "category": case["category"],
        "system_prompt": MODULE_PROGRAMMER_PROMPT,
        "module_contract": module_contract,
        "programmer_trace": trace,
        "programmer_result": result,
        "evaluation": evaluation,
    }
    trace_path = write_llm_trace(_TEST_NAME, case_id, trace_payload)
    return evaluation, trace_path


def _evaluate_module_result(
    *,
    module_contract: dict[str, Any],
    result: dict[str, object],
    trace: dict[str, object],
) -> dict[str, object]:
    """Evaluate one programmer result against the module contract shape."""

    errors: list[str] = []
    warnings: list[str] = []
    raw_output = trace.get("raw_output")
    if not isinstance(raw_output, str) or not raw_output.strip():
        errors.append("Programmer returned empty raw output.")
    if trace.get("markdown_code_block_found") is not True:
        errors.append("Programmer did not return exactly one markdown Python block.")

    code_artifact = result.get("code_artifact")
    if not isinstance(code_artifact, str) or not code_artifact.strip():
        errors.append("Programmer produced no code_artifact source.")
        return _evaluation(errors=errors, warnings=warnings)

    try:
        module = ast.parse(code_artifact)
    except SyntaxError as exc:
        errors.append(f"code_artifact is not valid Python: {exc}")
        return _evaluation(
            errors=errors,
            warnings=warnings,
            code_chars=len(code_artifact),
        )

    expected_symbols = _expected_symbol_names(module_contract)
    defined_symbols = _defined_top_level_symbols(module)
    missing_symbols = sorted(expected_symbols - defined_symbols)
    if missing_symbols:
        errors.append(
            "Programmer did not define expected top-level symbols: "
            + ", ".join(missing_symbols)
        )

    errors.extend(_import_errors(module, module_contract))
    errors.extend(_signature_errors(module, module_contract))
    errors.extend(_placeholder_errors(module, code_artifact))

    return _evaluation(
        errors=errors,
        warnings=warnings,
        code_chars=len(code_artifact),
        expected_symbols=sorted(expected_symbols),
        defined_symbols=sorted(defined_symbols),
        missing_symbols=missing_symbols,
        model=str(trace.get("model", "")),
        thinking_enabled=bool(trace.get("thinking_enabled")),
    )


def _evaluation(
    *,
    errors: list[str],
    warnings: list[str],
    code_chars: int = 0,
    expected_symbols: list[str] | None = None,
    defined_symbols: list[str] | None = None,
    missing_symbols: list[str] | None = None,
    model: str = "",
    thinking_enabled: bool = False,
) -> dict[str, object]:
    status = "passed"
    if errors:
        status = "failed"
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "code_chars": code_chars,
        "expected_symbols": expected_symbols or [],
        "defined_symbols": defined_symbols or [],
        "missing_symbols": missing_symbols or [],
        "model": model,
        "thinking_enabled": thinking_enabled,
    }


def _expected_symbol_names(module_contract: dict[str, Any]) -> set[str]:
    """Return expected top-level symbol names from the module contract."""

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


def _import_errors(
    module: ast.Module,
    module_contract: dict[str, Any],
) -> list[str]:
    """Return import errors for required and unsupported imports."""

    allowed_imports = {
        _normalize_source_line(value)
        for value in module_contract.get("imports", [])
        if isinstance(value, str) and value.strip()
    }
    actual_import_nodes = [
        node
        for node in ast.walk(module)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    actual_imports = [
        _normalize_source_line(ast.unparse(node))
        for node in actual_import_nodes
    ]
    errors: list[str] = []
    unexpected_imports = [
        import_line
        for import_line, import_node in zip(actual_imports, actual_import_nodes)
        if (
            import_line not in allowed_imports
            and not _is_standard_library_import(import_node)
        )
    ]
    if unexpected_imports:
        errors.append(
            "Programmer emitted non-standard imports not present in contract "
            "imports: "
            + "; ".join(unexpected_imports)
        )

    missing_imports = sorted(set(allowed_imports) - set(actual_imports))
    if missing_imports:
        errors.append(
            "Programmer omitted contract imports: " + "; ".join(missing_imports)
        )
    return errors


def _is_standard_library_import(node: ast.Import | ast.ImportFrom) -> bool:
    """Return whether an import targets the Python standard library."""

    roots: list[str] = []
    if isinstance(node, ast.Import):
        roots = [
            alias.name.split(".", maxsplit=1)[0]
            for alias in node.names
        ]
    elif node.level == 0 and node.module:
        roots = [node.module.split(".", maxsplit=1)[0]]
    if not roots:
        return False

    stdlib_names = getattr(sys, "stdlib_module_names", set())
    is_standard_library = all(root in stdlib_names for root in roots)
    return is_standard_library


def _signature_errors(
    module: ast.Module,
    module_contract: dict[str, Any],
) -> list[str]:
    """Return function, class, and class-child signature mismatches."""

    symbols = module_contract.get("symbols_to_define")
    if not isinstance(symbols, list):
        return []

    top_level = _top_level_nodes(module)
    errors: list[str] = []
    for symbol in symbols:
        if not isinstance(symbol, dict):
            continue
        name = _clean_string(symbol.get("name"))
        kind = _clean_string(symbol.get("kind"))
        signature = _clean_string(symbol.get("signature"))
        node = top_level.get(name)
        if not name or node is None:
            continue
        if kind in {"function", "test"}:
            errors.extend(_callable_signature_errors(
                node,
                signature=signature,
                symbol_name=name,
            ))
        elif kind in {"class", "dataclass"}:
            errors.extend(_class_signature_errors(
                node,
                signature=signature,
                symbol_name=name,
            ))
            errors.extend(_class_child_errors(
                node,
                children=symbol.get("children"),
                symbol_name=name,
            ))
        elif kind == "module_variable" and not isinstance(
            node,
            (ast.Assign, ast.AnnAssign),
        ):
            errors.append(f"{name} is not a top-level assignment.")
    return errors


def _top_level_nodes(module: ast.Module) -> dict[str, ast.AST]:
    """Return named top-level definition and assignment nodes."""

    nodes: dict[str, ast.AST] = {}
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for name in _target_names(target):
                    nodes[name] = node
        elif isinstance(node, ast.AnnAssign):
            for name in _target_names(node.target):
                nodes[name] = node
    return nodes


def _callable_signature_errors(
    node: ast.AST,
    *,
    signature: str,
    symbol_name: str,
) -> list[str]:
    """Return signature mismatches for a function-like symbol."""

    expected = _parse_callable_signature(signature)
    errors: list[str] = []
    if expected is None:
        return errors
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return [f"{symbol_name} is not a function."]

    actual_async = isinstance(node, ast.AsyncFunctionDef)
    if actual_async != expected["is_async"]:
        errors.append(f"{symbol_name} async/def shape does not match signature.")

    actual_args = [arg.arg for arg in node.args.args]
    if actual_args != expected["args"]:
        errors.append(
            f"{symbol_name} args mismatch: expected {expected['args']!r}, "
            f"got {actual_args!r}."
        )

    actual_return = ""
    if node.returns is not None:
        actual_return = _normalize_source_line(ast.unparse(node.returns))
    if actual_return != expected["return"]:
        errors.append(
            f"{symbol_name} return annotation mismatch: expected "
            f"{expected['return']!r}, got {actual_return!r}."
        )
    return errors


def _class_signature_errors(
    node: ast.AST,
    *,
    signature: str,
    symbol_name: str,
) -> list[str]:
    """Return signature mismatches for a class symbol."""

    expected = _parse_class_signature(signature)
    if expected is None:
        return []
    if not isinstance(node, ast.ClassDef):
        return [f"{symbol_name} is not a class."]

    errors: list[str] = []
    actual_bases = [_normalize_source_line(ast.unparse(base)) for base in node.bases]
    if actual_bases != expected["bases"]:
        errors.append(
            f"{symbol_name} bases mismatch: expected {expected['bases']!r}, "
            f"got {actual_bases!r}."
        )
    return errors


def _class_child_errors(
    node: ast.AST,
    *,
    children: object,
    symbol_name: str,
) -> list[str]:
    """Return missing class field or method errors from child contracts."""

    if not isinstance(node, ast.ClassDef) or not isinstance(children, list):
        return []

    field_names = _class_field_names(node)
    method_nodes = {
        child.name: child
        for child in node.body
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    errors: list[str] = []
    for child in children:
        if isinstance(child, str):
            field_name = child.split(":", maxsplit=1)[0].strip()
            if field_name and field_name not in field_names:
                errors.append(f"{symbol_name} is missing field {field_name!r}.")
            continue
        if not isinstance(child, dict):
            continue
        child_name = _clean_string(child.get("name"))
        signature = _clean_string(child.get("signature"))
        child_node = method_nodes.get(child_name)
        if child_node is None:
            errors.append(f"{symbol_name} is missing method {child_name!r}.")
            continue
        errors.extend(_callable_signature_errors(
            child_node,
            signature=signature,
            symbol_name=f"{symbol_name}.{child_name}",
        ))
    return errors


def _class_field_names(node: ast.ClassDef) -> set[str]:
    """Return field names assigned directly in a class body."""

    names: set[str] = set()
    for child in node.body:
        if isinstance(child, ast.AnnAssign):
            names.update(_target_names(child.target))
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                names.update(_target_names(target))
    return names


def _parse_callable_signature(signature: str) -> dict[str, object] | None:
    """Parse a contract function signature into comparable parts."""

    if not signature:
        return None
    source = f"{signature}\n    pass"
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    if not module.body:
        return None
    node = module.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return_annotation = ""
    if node.returns is not None:
        return_annotation = _normalize_source_line(ast.unparse(node.returns))
    return {
        "is_async": isinstance(node, ast.AsyncFunctionDef),
        "args": [arg.arg for arg in node.args.args],
        "return": return_annotation,
    }


def _parse_class_signature(signature: str) -> dict[str, object] | None:
    """Parse a contract class signature into comparable parts."""

    if not signature:
        return None
    source = f"{signature}\n    pass"
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    if not module.body or not isinstance(module.body[0], ast.ClassDef):
        return None
    node = module.body[0]
    return {
        "bases": [_normalize_source_line(ast.unparse(base)) for base in node.bases],
    }


def _normalize_source_line(value: str) -> str:
    """Normalize one source line for structural comparison."""

    normalized = " ".join(value.strip().split())
    return normalized


def _clean_string(value: object) -> str:
    """Return a stripped string or empty value."""

    if not isinstance(value, str):
        return ""
    return value.strip()


def _defined_top_level_symbols(module: ast.Module) -> set[str]:
    """Return top-level symbols defined by Python source."""

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
    """Return plain names assigned by one target node."""

    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_target_names(element))
        return names
    return set()


def _placeholder_errors(module: ast.Module, code_artifact: str) -> list[str]:
    """Return obvious placeholder-code failures."""

    errors: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Pass):
            errors.append("code_artifact contains pass.")
            break
    for node in ast.walk(module):
        if isinstance(node, ast.Expr) and _is_ellipsis(node.value):
            errors.append("code_artifact contains ellipsis.")
            break
    for node in ast.walk(module):
        if isinstance(node, ast.Raise) and _raises_not_implemented(node):
            errors.append("code_artifact raises NotImplementedError.")
            break
    if "TODO" in code_artifact:
        errors.append("code_artifact contains TODO.")
    return errors


def _is_ellipsis(node: ast.AST) -> bool:
    """Return whether a node is the ellipsis literal."""

    is_ellipsis = isinstance(node, ast.Constant) and node.value is Ellipsis
    return is_ellipsis


def _raises_not_implemented(node: ast.Raise) -> bool:
    """Return whether a raise statement raises NotImplementedError."""

    exception = node.exc
    if isinstance(exception, ast.Call):
        exception = exception.func
    if isinstance(exception, ast.Name):
        return exception.id == "NotImplementedError"
    return False


async def _assert_case(case_id: str) -> None:
    """Run and assert one live module-programmer case."""

    evaluation, trace_path = await _run_case(case_id)
    print(f"TRACE_PATH={trace_path.as_posix()}")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, default=str))
    assert evaluation["status"] == "passed", "\n".join(evaluation["errors"])


async def test_module_programmer_case_01() -> None:
    await _assert_case("case_01")


async def test_module_programmer_case_02() -> None:
    await _assert_case("case_02")


async def test_module_programmer_case_03() -> None:
    await _assert_case("case_03")


async def test_module_programmer_case_04() -> None:
    await _assert_case("case_04")
