"""Static package-level coherence checks for generated writing artifacts."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, TypedDict

from kazusa_ai_chatbot.coding_agent.code_writing.models import GeneratedArtifact
from kazusa_ai_chatbot.coding_agent.code_writing.models import WritingFileKind


PackageCoherenceStatus = Literal["succeeded", "failed"]


class PackageCoherenceReport(TypedDict):
    """Deterministic source-free package coherence report."""

    status: PackageCoherenceStatus
    errors: list[str]
    warnings: list[str]
    files: list[str]


@dataclass(frozen=True)
class _FunctionSignature:
    """Callable arity facts needed for deterministic local call checks."""

    module_name: str
    function_name: str
    required_positional_names: tuple[str, ...]
    accepts_varargs: bool


@dataclass(frozen=True)
class _PythonModule:
    """One generated Python module indexed by import name."""

    path: str
    file_kind: WritingFileKind
    module_names: tuple[str, ...]
    tree: ast.Module
    symbols: frozenset[str]
    functions: dict[str, _FunctionSignature]


def evaluate_package_coherence(
    generated_artifacts: list[GeneratedArtifact],
) -> PackageCoherenceReport:
    """Evaluate generated Python artifacts as one coherent source-free package."""

    modules: list[_PythonModule] = []
    errors: list[str] = []
    warnings: list[str] = []
    files: list[str] = []

    for artifact in generated_artifacts:
        if not _is_python_artifact(artifact):
            continue
        path = _normalized_path(artifact["path"])
        files.append(path)
        try:
            tree = ast.parse(artifact["content"], filename=path)
        except SyntaxError as exc:
            errors.append(
                "Package coherence could not parse "
                f"'{path}' as Python: {exc.msg}."
            )
            continue
        module_names = _module_names_for_path(path)
        modules.append(
            _PythonModule(
                path=path,
                file_kind=artifact["file_kind"],
                module_names=module_names,
                tree=tree,
                symbols=frozenset(_exported_symbols(tree)),
                functions=_function_signatures(tree, module_names),
            )
        )

    module_index = _module_index(modules)
    errors.extend(_import_errors(modules=modules, module_index=module_index))
    errors.extend(_call_signature_errors(modules=modules, module_index=module_index))
    errors.extend(_duplicate_cli_entrypoint_errors(modules))

    status: PackageCoherenceStatus = "succeeded"
    if errors:
        status = "failed"
    report: PackageCoherenceReport = {
        "status": status,
        "errors": _dedupe_strings(errors),
        "warnings": warnings,
        "files": files,
    }
    return report


def _is_python_artifact(artifact: GeneratedArtifact) -> bool:
    if artifact["content_format"] == "python":
        return True
    path = _normalized_path(artifact["path"])
    return path.endswith(".py")


def _normalized_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip("/")
    return normalized


def _module_names_for_path(path: str) -> tuple[str, ...]:
    if not path.endswith(".py"):
        return ()

    without_suffix = path[:-3]
    if without_suffix.endswith("/__init__"):
        without_suffix = without_suffix[: -len("/__init__")]
    dotted = without_suffix.replace("/", ".")
    names = [dotted]
    if dotted.startswith("src."):
        names.append(dotted[len("src.") :])
    return tuple(_dedupe_strings(names))


def _exported_symbols(tree: ast.Module) -> set[str]:
    symbols: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(statement.name)
            continue
        if isinstance(statement, ast.ClassDef):
            symbols.add(statement.name)
            continue
        if isinstance(statement, ast.Assign):
            symbols.update(_assigned_names(statement.targets))
            continue
        if isinstance(statement, ast.AnnAssign):
            symbols.update(_assigned_names([statement.target]))
            continue
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            for alias in statement.names:
                exported = alias.asname or alias.name.split(".", maxsplit=1)[0]
                symbols.add(exported)
    return symbols


def _assigned_names(targets: list[ast.expr]) -> set[str]:
    names: set[str] = set()
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
            continue
        if isinstance(target, (ast.Tuple, ast.List)):
            names.update(_assigned_names(list(target.elts)))
    return names


def _function_signatures(
    tree: ast.Module,
    module_names: tuple[str, ...],
) -> dict[str, _FunctionSignature]:
    primary_module_name = module_names[0] if module_names else ""
    signatures: dict[str, _FunctionSignature] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional_args = [
            *statement.args.posonlyargs,
            *statement.args.args,
        ]
        default_count = len(statement.args.defaults)
        required_count = len(positional_args) - default_count
        required_names = tuple(arg.arg for arg in positional_args[:required_count])
        signatures[statement.name] = _FunctionSignature(
            module_name=primary_module_name,
            function_name=statement.name,
            required_positional_names=required_names,
            accepts_varargs=statement.args.vararg is not None,
        )
    return signatures


def _module_index(modules: list[_PythonModule]) -> dict[str, _PythonModule]:
    index: dict[str, _PythonModule] = {}
    for module in modules:
        for module_name in module.module_names:
            index[module_name] = module
    return index


def _import_errors(
    *,
    modules: list[_PythonModule],
    module_index: dict[str, _PythonModule],
) -> list[str]:
    errors: list[str] = []
    for module in modules:
        for statement in ast.walk(module.tree):
            if not isinstance(statement, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(statement, ast.Import):
                errors.extend(_plain_import_errors(statement, module, module_index))
            else:
                errors.extend(_from_import_errors(statement, module, module_index))
    return errors


def _plain_import_errors(
    statement: ast.Import,
    module: _PythonModule,
    module_index: dict[str, _PythonModule],
) -> list[str]:
    errors: list[str] = []
    for alias in statement.names:
        imported_module = alias.name
        if imported_module in module_index:
            continue
        if _is_required_local_module(imported_module, module_index):
            errors.append(
                "Package coherence failed: "
                f"'{module.path}' imports missing generated local module "
                f"'{imported_module}'."
            )
    return errors


def _from_import_errors(
    statement: ast.ImportFrom,
    module: _PythonModule,
    module_index: dict[str, _PythonModule],
) -> list[str]:
    imported_module = _resolved_from_module(statement, module)
    if not imported_module:
        return []

    target_module = module_index.get(imported_module)
    if target_module is None:
        if not _is_required_local_module(imported_module, module_index):
            return []
        return [
            "Package coherence failed: "
            f"'{module.path}' imports missing generated local module "
            f"'{imported_module}'."
        ]

    errors: list[str] = []
    for alias in statement.names:
        if alias.name == "*":
            continue
        submodule_name = f"{imported_module}.{alias.name}"
        if alias.name in target_module.symbols or submodule_name in module_index:
            continue
        errors.append(
            "Package coherence failed: "
            f"'{module.path}' imports missing generated symbol "
            f"'{alias.name}' from module '{imported_module}'."
        )
    return errors


def _resolved_from_module(
    statement: ast.ImportFrom,
    module: _PythonModule,
) -> str:
    imported_module = statement.module or ""
    if statement.level == 0:
        return imported_module
    current = module.module_names[0] if module.module_names else ""
    current_parts = current.split(".")
    if module.path.endswith("/__init__.py"):
        package_parts = current_parts
    else:
        package_parts = current_parts[:-1]
    keep_count = max(0, len(package_parts) - statement.level + 1)
    base_parts = package_parts[:keep_count]
    if imported_module:
        base_parts.extend(imported_module.split("."))
    return ".".join(part for part in base_parts if part)


def _is_required_local_module(
    imported_module: str,
    module_index: dict[str, _PythonModule],
) -> bool:
    if imported_module == "src" or imported_module.startswith("src."):
        return True
    if imported_module == "tests" or imported_module.startswith("tests."):
        return True
    imported_root = imported_module.split(".", maxsplit=1)[0]
    for module_name in module_index:
        if module_name == imported_root or module_name.startswith(f"{imported_root}."):
            return True
    return False


def _call_signature_errors(
    *,
    modules: list[_PythonModule],
    module_index: dict[str, _PythonModule],
) -> list[str]:
    errors: list[str] = []
    for module in modules:
        imported_symbols = _imported_symbol_index(module, module_index)
        imported_modules = _imported_module_index(module, module_index)
        for call in ast.walk(module.tree):
            if not isinstance(call, ast.Call):
                continue
            signature = _signature_for_call(
                call=call,
                imported_symbols=imported_symbols,
                imported_modules=imported_modules,
                module_index=module_index,
            )
            if signature is None or signature.accepts_varargs:
                continue
            required_names = signature.required_positional_names
            if not required_names:
                continue
            positional_count = _known_positional_count(call)
            if positional_count is None:
                continue
            provided_required_names = _provided_required_keyword_names(
                call=call,
                required_names=required_names,
            )
            provided_count = positional_count + len(provided_required_names)
            required_count = len(required_names)
            if provided_count >= required_count:
                continue
            errors.append(
                "Package coherence failed: "
                f"'{module.path}' calls "
                f"'{signature.module_name}.{signature.function_name}' with "
                f"{positional_count} positional arguments, but generated "
                f"signature requires at least {required_count}."
            )
    return errors


def _imported_symbol_index(
    module: _PythonModule,
    module_index: dict[str, _PythonModule],
) -> dict[str, tuple[str, str]]:
    index: dict[str, tuple[str, str]] = {}
    for statement in ast.walk(module.tree):
        if not isinstance(statement, ast.ImportFrom):
            continue
        imported_module = _resolved_from_module(statement, module)
        if imported_module not in module_index:
            continue
        for alias in statement.names:
            if alias.name == "*":
                continue
            local_name = alias.asname or alias.name
            index[local_name] = (imported_module, alias.name)
    return index


def _imported_module_index(
    module: _PythonModule,
    module_index: dict[str, _PythonModule],
) -> dict[str, str]:
    index: dict[str, str] = {}
    for statement in ast.walk(module.tree):
        if not isinstance(statement, ast.Import):
            continue
        for alias in statement.names:
            imported_module = alias.name
            if imported_module not in module_index:
                continue
            local_name = alias.asname or imported_module.split(".", maxsplit=1)[0]
            index[local_name] = imported_module
    return index


def _signature_for_call(
    *,
    call: ast.Call,
    imported_symbols: dict[str, tuple[str, str]],
    imported_modules: dict[str, str],
    module_index: dict[str, _PythonModule],
) -> _FunctionSignature | None:
    func = call.func
    if isinstance(func, ast.Name):
        target = imported_symbols.get(func.id)
        if target is None:
            return None
        module_name, function_name = target
        return module_index[module_name].functions.get(function_name)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        module_name = imported_modules.get(func.value.id)
        if module_name is None:
            return None
        return module_index[module_name].functions.get(func.attr)
    return None


def _known_positional_count(call: ast.Call) -> int | None:
    for argument in call.args:
        if isinstance(argument, ast.Starred):
            return None
    return len(call.args)


def _provided_required_keyword_names(
    *,
    call: ast.Call,
    required_names: tuple[str, ...],
) -> set[str]:
    required_name_set = set(required_names)
    provided: set[str] = set()
    for keyword in call.keywords:
        if keyword.arg is None:
            return set(required_names)
        if keyword.arg in required_name_set:
            provided.add(keyword.arg)
    return provided


def _duplicate_cli_entrypoint_errors(
    modules: list[_PythonModule],
) -> list[str]:
    entrypoints = [
        module
        for module in modules
        if _is_cli_entrypoint_module(module)
    ]
    if len(entrypoints) <= 1:
        return []
    paths = ", ".join(module.path for module in entrypoints)
    return [
        "Package coherence failed: multiple generated CLI entrypoint "
        f"wrappers were produced ({paths}). Keep one coherent CLI entrypoint "
        "and make tests import that entrypoint."
    ]


def _is_cli_entrypoint_module(module: _PythonModule) -> bool:
    if module.file_kind != "source":
        return False
    path = PurePosixPath(module.path)
    basename = path.name.casefold()
    if not (
        "cli" in basename
        or "wrapper" in basename
        or "entrypoint" in basename
    ):
        return False
    has_main = "main" in module.symbols
    has_main_guard = _has_main_guard(module.tree)
    return has_main or has_main_guard


def _has_main_guard(tree: ast.Module) -> bool:
    for statement in ast.walk(tree):
        if not isinstance(statement, ast.If):
            continue
        if _is_main_guard_test(statement.test):
            return True
    return False


def _is_main_guard_test(expr: ast.expr) -> bool:
    if not isinstance(expr, ast.Compare):
        return False
    if not isinstance(expr.left, ast.Name) or expr.left.id != "__name__":
        return False
    if len(expr.ops) != 1 or not isinstance(expr.ops[0], ast.Eq):
        return False
    if len(expr.comparators) != 1:
        return False
    comparator = expr.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value in deduped:
            continue
        deduped.append(value)
    return deduped
