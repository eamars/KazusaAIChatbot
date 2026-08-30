"""Exact static decommission gates for the retired task surfaces."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

PRODUCTION_DELETION_PATHS = (
    "scripts/run_rag2_e2e_case.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/__init__.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/algorithmic.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/constants.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/contracts.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/graph.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/service.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/stages.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/subagent/__init__.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/subagent/algorithmic.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/subagent/evidence.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/subagent/media.py",
    "src/kazusa_ai_chatbot/complex_task_resolver/subagents.py",
    "src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py",
    "src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py",
    "src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py",
    "src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_prompt_views.py",
    "src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py",
    "src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_types.py",
    "src/kazusa_ai_chatbot/rag/quote_aware_sequence.py",
    "src/kazusa_ai_chatbot/task_resolution/orchestrator.py",
    "src/kazusa_ai_chatbot/task_resolution/specialists/__init__.py",
    "src/kazusa_ai_chatbot/task_resolution/specialists/local_context.py",
    "src/kazusa_ai_chatbot/task_resolution/specialists/public_research.py",
    "src/kazusa_ai_chatbot/task_resolution/specialists/text_computation.py",
    "src/kazusa_ai_chatbot/task_resolution/state.py",
)

TEST_DELETION_PATHS = (
    "tests/test_complex_task_resolver_algorithmic.py",
    "tests/test_complex_task_resolver_contracts.py",
    "tests/test_complex_task_resolver_evidence.py",
    "tests/test_complex_task_resolver_fixture.py",
    "tests/test_complex_task_resolver_graph.py",
    "tests/test_complex_task_resolver_live_llm.py",
    "tests/test_complex_task_resolver_media_subagent.py",
    "tests/test_complex_task_resolver_prompt_contract.py",
    "tests/test_complex_task_resolver_service.py",
    "tests/test_local_context_resolver_rag2_vs_rag3_live_llm.py",
    "tests/test_persona_supervisor2_rag2_integration.py",
    "tests/test_persona_supervisor2_rag_supervisor2_live.py",
    "tests/test_quote_aware_rag_sequence.py",
    "tests/test_quote_aware_rag_sequence_live.py",
    "tests/test_rag_finalizer_time_context.py",
    "tests/test_rag_initializer_cache2.py",
    "tests/test_rag_phase3_initializer_live_llm.py",
    "tests/test_rag_phase3_supervisor_integration.py",
    "tests/test_rag_phase4_continuation_live_llm.py",
    "tests/test_rag_prompt_contract_text.py",
    "tests/test_rag_recall_live_llm.py",
    "tests/test_task_resolution_background_research_e2e_live_llm.py",
    "tests/test_task_resolution_live_llm.py",
    "tests/test_task_resolution_orchestrator.py",
    "tests/test_task_resolution_specialists.py",
    "tests/test_task_resolution_state.py",
    "tests/fixtures/complex_task_resolver_review_cases.json",
)


def _path_list(paths: tuple[str, ...]) -> list[Path]:
    """Resolve a governed relative path without broad filesystem matching."""

    return [REPOSITORY_ROOT / path for path in paths]


def test_legacy_task_complex_and_rag2_executor_sources_are_absent() -> None:
    """All 26 retained production/script artifacts disappear atomically."""

    assert len(PRODUCTION_DELETION_PATHS) == 26
    assert len(set(PRODUCTION_DELETION_PATHS)) == 26
    present = [
        str(path)
        for path in _path_list(PRODUCTION_DELETION_PATHS)
        if path.exists()
    ]
    assert not present, f"legacy production artifacts remain: {present}"


def _legacy_imports(path: Path) -> set[str]:
    """Extract actual import targets from one surviving production module."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        pytest.fail(f"cannot audit production module {path}: {exc}")
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            if name.startswith((
                "kazusa_ai_chatbot.complex_task_resolver",
                "kazusa_ai_chatbot.rag.supervisor2",
            )):
                targets.add(name)
    return targets


def test_runtime_import_graph_contains_no_legacy_executor_imports() -> None:
    """The full surviving production Python surface has no retired imports."""

    deleted = {
        (REPOSITORY_ROOT / path).resolve()
        for path in PRODUCTION_DELETION_PATHS
        if path.startswith("src/")
    }
    production_roots = (
        REPOSITORY_ROOT / "src" / "agentic_resolver",
        REPOSITORY_ROOT / "src" / "kazusa_ai_chatbot",
    )
    findings: dict[str, list[str]] = {}
    for root in production_roots:
        for path in root.rglob("*.py"):
            if path.resolve() in deleted:
                continue
            imports = _legacy_imports(path)
            if imports:
                findings[str(path.relative_to(REPOSITORY_ROOT))] = sorted(imports)
    assert not findings, f"legacy imports remain: {findings}"


def _test_module_name(path: Path) -> str:
    """Convert a repository test path into its importable module name."""

    relative = path.relative_to(REPOSITORY_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _deleted_test_modules() -> set[str]:
    """Return importable module names covered by the deletion inventory."""

    return {
        _test_module_name(REPOSITORY_ROOT / path)
        for path in TEST_DELETION_PATHS
        if path.endswith(".py")
    }


def _relative_test_import(source: Path, node: ast.ImportFrom) -> str:
    """Resolve a relative import against its surviving test module."""

    source_parts = _test_module_name(source).split(".")[:-1]
    parent_parts = source_parts[: len(source_parts) - node.level + 1]
    if node.module:
        parent_parts.extend(node.module.split("."))
    return ".".join(parent_parts)


def _test_import_targets(source: Path, node: ast.AST) -> set[str]:
    """Extract import targets, including dynamic import calls, from an AST node."""

    if isinstance(node, ast.Import):
        return {alias.name for alias in node.names}
    if isinstance(node, ast.ImportFrom):
        base = (
            _relative_test_import(source, node)
            if node.level
            else node.module or ""
        )
        targets = {base} if base else set()
        targets.update(
            f"{base}.{alias.name}" if base else alias.name
            for alias in node.names
        )
        return targets
    if not isinstance(node, ast.Call) or not node.args:
        return set()
    function = node.func
    dynamic_import = (
        isinstance(function, ast.Name) and function.id in {"__import__", "find_spec"}
    ) or (
        isinstance(function, ast.Attribute)
        and function.attr in {"import_module", "find_spec"}
    )
    if not dynamic_import:
        return set()
    try:
        value = ast.literal_eval(node.args[0])
    except (TypeError, ValueError, SyntaxError):
        return set()
    return {value} if isinstance(value, str) else set()


def _surviving_deleted_test_imports() -> dict[str, list[str]]:
    """Find retired test/fixture imports from every surviving test module."""

    deleted_modules = _deleted_test_modules()
    findings: dict[str, list[str]] = {}
    for path in (REPOSITORY_ROOT / "tests").rglob("*.py"):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        if relative in TEST_DELETION_PATHS:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            pytest.fail(f"cannot audit surviving test module {path}: {exc}")
        hits = {
            target
            for node in ast.walk(tree)
            for target in _test_import_targets(path, node)
            if any(
                target == deleted
                or target.startswith(f"{deleted}.")
                for deleted in deleted_modules
            )
        }
        if hits:
            findings[relative] = sorted(hits)
    return findings


def test_surviving_test_tree_imports_no_deleted_modules_or_fixtures() -> None:
    """Surviving tests remain independent of every retired test/fixture path."""

    assert not _surviving_deleted_test_imports()


def test_retained_rag_package_has_no_rag2_runtime_claim() -> None:
    """The retained RAG package identifies only the live RAG3 boundary."""

    package = REPOSITORY_ROOT / "src" / "kazusa_ai_chatbot" / "rag" / "__init__.py"
    assert package.is_file()
    assert "rag2" not in package.read_text(encoding="utf-8").lower()


def test_legacy_test_and_fixture_artifacts_are_absent() -> None:
    """All 27 retained test and fixture artifacts disappear atomically."""

    assert len(TEST_DELETION_PATHS) == 27
    assert len(set(TEST_DELETION_PATHS)) == 27
    present = [
        str(path)
        for path in _path_list(TEST_DELETION_PATHS)
        if path.exists()
    ]
    assert not present, f"retired test/fixture artifacts remain: {present}"
