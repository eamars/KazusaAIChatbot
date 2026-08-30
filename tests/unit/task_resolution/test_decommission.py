"""Exact static decommission gates for the retired task surfaces."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

PRODUCTION_DELETION_PATHS = (
    "scripts/run_coding_agent_benchmark.py",
    "scripts/run_rag2_e2e_case.py",
    "src/kazusa_ai_chatbot/coding_agent/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/code_action_loop/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/code_action_loop/actions.py",
    "src/kazusa_ai_chatbot/coding_agent/code_action_loop/context.py",
    "src/kazusa_ai_chatbot/coding_agent/code_action_loop/models.py",
    "src/kazusa_ai_chatbot/coding_agent/code_action_loop/parser.py",
    "src/kazusa_ai_chatbot/coding_agent/code_action_loop/prompts.py",
    "src/kazusa_ai_chatbot/coding_agent/code_action_loop/state.py",
    "src/kazusa_ai_chatbot/coding_agent/code_action_loop/supervisor.py",
    "src/kazusa_ai_chatbot/coding_agent/code_executing/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/code_executing/models.py",
    "src/kazusa_ai_chatbot/coding_agent/code_executing/runner.py",
    "src/kazusa_ai_chatbot/coding_agent/code_executing/supervisor.py",
    "src/kazusa_ai_chatbot/coding_agent/code_fetching/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/code_fetching/agent.py",
    "src/kazusa_ai_chatbot/coding_agent/code_fetching/github.py",
    "src/kazusa_ai_chatbot/coding_agent/code_fetching/local_checkout.py",
    "src/kazusa_ai_chatbot/coding_agent/code_fetching/managed_clone.py",
    "src/kazusa_ai_chatbot/coding_agent/code_fetching/managed_download.py",
    "src/kazusa_ai_chatbot/coding_agent/code_fetching/managed_inline.py",
    "src/kazusa_ai_chatbot/coding_agent/code_fetching/models.py",
    "src/kazusa_ai_chatbot/coding_agent/code_fetching/source_intake.py",
    "src/kazusa_ai_chatbot/coding_agent/code_fetching/source_resolver.py",
    "src/kazusa_ai_chatbot/coding_agent/code_fetching/source_scope.py",
    "src/kazusa_ai_chatbot/coding_agent/code_modifying/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/code_modifying/models.py",
    "src/kazusa_ai_chatbot/coding_agent/code_modifying/product_manager.py",
    "src/kazusa_ai_chatbot/coding_agent/code_modifying/programmer.py",
    "src/kazusa_ai_chatbot/coding_agent/code_modifying/supervisor.py",
    "src/kazusa_ai_chatbot/coding_agent/code_patching/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/code_patching/apply.py",
    "src/kazusa_ai_chatbot/coding_agent/code_patching/models.py",
    "src/kazusa_ai_chatbot/coding_agent/code_patching/patch_operations.py",
    "src/kazusa_ai_chatbot/coding_agent/code_patching/patch_validation.py",
    "src/kazusa_ai_chatbot/coding_agent/code_patching/patcher.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/agent.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/evidence.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/llm_config.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/master_pm.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/models.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/planner.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/product_manager.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/programmer.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/prompts.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/repository_map.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/supervisor.py",
    "src/kazusa_ai_chatbot/coding_agent/code_reading/synthesizer.py",
    "src/kazusa_ai_chatbot/coding_agent/code_verifying/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/code_verifying/execution_planning.py",
    "src/kazusa_ai_chatbot/coding_agent/code_verifying/models.py",
    "src/kazusa_ai_chatbot/coding_agent/code_verifying/supervisor.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/acceptance.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/agent.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/diagnostic_trace.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/llm_config.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/models.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/package_coherence.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/product_manager.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/programmer.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/supervisor.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/synthesizer.py",
    "src/kazusa_ai_chatbot/coding_agent/code_writing/workspace.py",
    "src/kazusa_ai_chatbot/coding_agent/coding_run/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/coding_run/evaluation.py",
    "src/kazusa_ai_chatbot/coding_agent/coding_run/ledger.py",
    "src/kazusa_ai_chatbot/coding_agent/coding_run/locking.py",
    "src/kazusa_ai_chatbot/coding_agent/coding_run/models.py",
    "src/kazusa_ai_chatbot/coding_agent/coding_run/supervisor.py",
    "src/kazusa_ai_chatbot/coding_agent/context_budget.py",
    "src/kazusa_ai_chatbot/coding_agent/external_evidence.py",
    "src/kazusa_ai_chatbot/coding_agent/file_agent.py",
    "src/kazusa_ai_chatbot/coding_agent/models.py",
    "src/kazusa_ai_chatbot/coding_agent/path_classification.py",
    "src/kazusa_ai_chatbot/coding_agent/repository_index/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/repository_index/builder.py",
    "src/kazusa_ai_chatbot/coding_agent/repository_index/identity.py",
    "src/kazusa_ai_chatbot/coding_agent/repository_index/models.py",
    "src/kazusa_ai_chatbot/coding_agent/repository_index/overlay.py",
    "src/kazusa_ai_chatbot/coding_agent/repository_index/regex_worker.py",
    "src/kazusa_ai_chatbot/coding_agent/repository_index/search.py",
    "src/kazusa_ai_chatbot/coding_agent/repository_index/storage.py",
    "src/kazusa_ai_chatbot/coding_agent/safety.py",
    "src/kazusa_ai_chatbot/coding_agent/supervisor.py",
    "src/kazusa_ai_chatbot/coding_agent/tools/__init__.py",
    "src/kazusa_ai_chatbot/coding_agent/tools/git.py",
    "src/kazusa_ai_chatbot/coding_agent/tools/paths.py",
    "src/kazusa_ai_chatbot/coding_agent/work_ledger.py",
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
    "src/kazusa_ai_chatbot/task_resolution/specialists/coding.py",
    "src/kazusa_ai_chatbot/task_resolution/specialists/local_context.py",
    "src/kazusa_ai_chatbot/task_resolution/specialists/public_research.py",
    "src/kazusa_ai_chatbot/task_resolution/specialists/text_computation.py",
    "src/kazusa_ai_chatbot/task_resolution/state.py",
)

TEST_DELETION_PATHS = (
    "tests/test_coding_agent_async_boundaries.py",
    "tests/test_coding_agent_benchmark_contracts.py",
    "tests/test_coding_agent_fetching.py",
    "tests/test_coding_agent_fetching_internet.py",
    "tests/test_coding_agent_image_reading_acceptance.py",
    "tests/test_coding_agent_interface.py",
    "tests/test_coding_agent_phase2_new_artifact_contracts.py",
    "tests/test_coding_agent_phase4_code_modifying_contracts.py",
    "tests/test_coding_agent_phase4_code_patching_contracts.py",
    "tests/test_coding_agent_phase4_interface.py",
    "tests/test_coding_agent_phase5_interface.py",
    "tests/test_coding_agent_phase5_patch_apply_contracts.py",
    "tests/test_coding_agent_phase6_code_executing_contracts.py",
    "tests/test_coding_agent_phase6_interface.py",
    "tests/test_coding_agent_phase8_interface.py",
    "tests/test_coding_agent_phase8_verify_repair_contracts.py",
    "tests/test_coding_agent_phase9_e2e_workflows.py",
    "tests/test_coding_agent_phase9_interface.py",
    "tests/test_coding_agent_phase9_run_supervisor_contracts.py",
    "tests/test_coding_agent_phase_b_execution_planning.py",
    "tests/test_coding_agent_phase_b_failure_feedback.py",
    "tests/test_coding_agent_phase_c_accepted_task_live_db.py",
    "tests/test_coding_agent_phase_c_locking.py",
    "tests/test_coding_agent_phase_c_run_context_contracts.py",
    "tests/test_coding_agent_phase_d_action_loop_contracts.py",
    "tests/test_coding_agent_phase_d_benchmark_contracts.py",
    "tests/test_coding_agent_phase_d_candidate_recovery.py",
    "tests/test_coding_agent_phase_d_coding_run_integration.py",
    "tests/test_coding_agent_phase_d_patch_operations.py",
    "tests/test_coding_agent_phase_d_repository_index.py",
    "tests/test_coding_agent_reading.py",
    "tests/test_coding_agent_reading_acceptance.py",
    "tests/test_coding_agent_reading_pm_programmer.py",
    "tests/test_coding_agent_source_intake.py",
    "tests/test_coding_agent_source_resolution.py",
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
    "tests/fixtures/coding_agent_benchmark/cases.jsonl",
    "tests/fixtures/coding_agent_existing_source_gates/conftest.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_01_log_counter/README.md",
    "tests/fixtures/coding_agent_existing_source_gates/gate_01_log_counter/log_counter.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_01_log_counter/tests/test_log_counter.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/README.md",
    "tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/contacts_jsonl_to_csv/__init__.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/contacts_jsonl_to_csv/cli.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/contacts_jsonl_to_csv/converter.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/tests/test_cli.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_02_contacts_jsonl_to_csv/tests/test_converter.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/README.md",
    "tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/mdlinkcheck/__init__.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/mdlinkcheck/anchors.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/mdlinkcheck/cli.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/mdlinkcheck/scanner.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/tests/test_anchors.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_03_markdown_link_checker/tests/test_scanner.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/README.md",
    "tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/issue_tracker/__init__.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/issue_tracker/api.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/issue_tracker/models.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/issue_tracker/store.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/tests/test_api.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_04_issue_tracker_soft_delete/tests/test_store.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/README.md",
    "tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/__init__.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/cli.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/csv_io.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/fetch.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/html_extract.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/inventory_sync/report.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/tests/test_cli.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/tests/test_fetch.py",
    "tests/fixtures/coding_agent_existing_source_gates/gate_05_inventory_sync_fetch_cache/tests/test_report.py",
    "tests/fixtures/coding_agent_full_workflow/gate_01_cli_command_discovery/README.md",
    "tests/fixtures/coding_agent_full_workflow/gate_01_cli_command_discovery/src/tooling/__init__.py",
    "tests/fixtures/coding_agent_full_workflow/gate_01_cli_command_discovery/src/tooling/cli.py",
    "tests/fixtures/coding_agent_full_workflow/gate_01_cli_command_discovery/src/tooling/commands.py",
    "tests/fixtures/coding_agent_full_workflow/gate_01_cli_command_discovery/tests/test_cli.py",
    "tests/fixtures/coding_agent_full_workflow/gate_02_csv_normalizer/README.md",
    "tests/fixtures/coding_agent_full_workflow/gate_03_counter_cli_json/README.md",
    "tests/fixtures/coding_agent_full_workflow/gate_03_counter_cli_json/counter_cli/__init__.py",
    "tests/fixtures/coding_agent_full_workflow/gate_03_counter_cli_json/counter_cli/cli.py",
    "tests/fixtures/coding_agent_full_workflow/gate_03_counter_cli_json/tests/test_cli.py",
    "tests/fixtures/coding_agent_full_workflow/gate_04_slug_normalization/README.md",
    "tests/fixtures/coding_agent_full_workflow/gate_04_slug_normalization/slug_tools/__init__.py",
    "tests/fixtures/coding_agent_full_workflow/gate_04_slug_normalization/slug_tools/slug.py",
    "tests/fixtures/coding_agent_full_workflow/gate_04_slug_normalization/tests/test_slug.py",
    "tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/README.md",
    "tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/release_feed/__init__.py",
    "tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/release_feed/cache.py",
    "tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/release_feed/cli.py",
    "tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/release_feed/feed.py",
    "tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/tests/test_cache.py",
    "tests/fixtures/coding_agent_full_workflow/gate_05_release_feed_cache_cli/tests/test_cli.py",
    "tests/fixtures/coding_agent_full_workflow/gate_09_missing_dependency/README.md",
    "tests/fixtures/coding_agent_full_workflow/gate_09_missing_dependency/dep_tool/__init__.py",
    "tests/fixtures/coding_agent_full_workflow/gate_09_missing_dependency/dep_tool/loader.py",
    "tests/fixtures/coding_agent_full_workflow/gate_09_missing_dependency/tests/test_yaml_dependency.py",
    "tests/fixtures/coding_agent_full_workflow/manifest.md",
    "tests/fixtures/coding_agent_source_intake_signoff_cases.json",
    "tests/fixtures/complex_task_resolver_review_cases.json",
)


def _path_list(paths: tuple[str, ...]) -> list[Path]:
    """Resolve a governed relative path without broad filesystem matching."""

    return [REPOSITORY_ROOT / path for path in paths]


def test_legacy_task_complex_coding_and_rag2_executor_sources_are_absent() -> None:
    """All 117 planned production/script artifacts disappear atomically."""

    assert len(PRODUCTION_DELETION_PATHS) == 117
    assert len(set(PRODUCTION_DELETION_PATHS)) == 117
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
                "kazusa_ai_chatbot.coding_agent",
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
    """All 124 planned test and fixture artifacts disappear atomically."""

    assert len(TEST_DELETION_PATHS) == 124
    assert len(set(TEST_DELETION_PATHS)) == 124
    present = [
        str(path)
        for path in _path_list(TEST_DELETION_PATHS)
        if path.exists()
    ]
    assert not present, f"retired test/fixture artifacts remain: {present}"
