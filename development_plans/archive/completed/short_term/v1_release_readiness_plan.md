# v1_release_readiness_plan

## Summary

- Goal: prepare the current `main` tip for the local `v1.0.0` release.
- Plan class: release readiness and bounded contract cleanup.
- Status: completed
- Source baseline: `main` at `fa5410df`, clean against `origin/main` before
  this plan.
- Release direction: make product versioning authoritative, describe the
  shipped runtime accurately in both top-level READMEs, align deterministic
  tests with the current typed contracts, verify packaging, and create the
  annotated local `v1.0.0` tag after all gates pass.
- Scope boundary: no remote push, no environment inspection, no live database
  or live LLM execution, and no change to internal schema or prompt versions
  solely because the product version changes.

## Baseline evidence

- Default command: `venv\\Scripts\\python -m pytest -q` completed with
  deterministic failures after the broad suite ran.
- Collection: `4356/5334` tests selected by the default marker policy and `978`
  deselected.
- Bounded cached-failure review identified stale direct callers for the
  current `service.chat` and console-client contracts, a missing resolver
  evidence-state fixture, a script importing a private DB dependency, stale
  action-executor fakes, a stale preference assertion, three unavailable
  ignored diagnostic captures, and a clean-tree owner-matrix precondition.

## Fixed scope and ownership

| Change surface | Owner | Required deterministic evidence |
|---|---|---|
| `pyproject.toml`, `src/kazusa_ai_chatbot/version.py`, `src/control_console/__init__.py` | product release identity and package metadata | `tests/test_release_metadata.py::test_product_release_identity_is_consistent`; `tests/test_test_impact_manifest.py::test_manifest_accepts_an_explicit_package_init_source_root`; package wheel metadata inspection |
| `README.md`, `README_CN.md`, `CHANGELOG.md` | release documentation | `tests/test_documentation_harmonization.py::test_top_level_readmes_include_current_route_families`; `tests/test_documentation_harmonization.py::test_top_level_readmes_link_current_runtime_subsystems` |
| `src/scripts/export_trace_correlation_manifest.py` | public DB maintenance boundary | `tests/test_script_db_boundary.py::test_scripts_do_not_import_raw_or_private_db_boundary` |
| `tests/test_control_console_bootstrap.py` | current console-client constructor contract | `tests/test_control_console_bootstrap.py::test_bootstrap_projects_live_health_without_overview_duplication`; `tests/test_control_console_bootstrap.py::test_bootstrap_projects_live_health_when_brain_is_unmanaged`; `tests/test_control_console_bootstrap.py::test_bootstrap_does_not_query_brain_for_stale_unowned_conflict` |
| `tests/test_past_dialog_cognition_rag_integration.py` | current resolver observation contract | `tests/test_past_dialog_cognition_rag_integration.py::test_resolver_loop_attaches_rag_residual_to_private_cognition_state` |
| `tests/test_cognition_preference_adapter.py` | current preference prompt wording | `tests/test_cognition_preference_adapter.py::test_preference_stage_owns_visible_boundaries_only` |
| `tests/test_self_cognition_tracking.py` | current private action executor seam | `tests/test_self_cognition_tracking.py::test_runner_executes_private_lifecycle_action_for_consolidation`; `tests/test_self_cognition_tracking.py::test_runner_routes_lifecycle_intent_through_specialist_before_execution` |
| `tests/test_service_background_consolidation.py`, `tests/test_service_input_queue.py` | current `/chat` direct-call contract | all 35 nodes in `tests/test_service_background_consolidation.py` and the three direct-call nodes `tests/test_service_input_queue.py::test_chat_enqueue_commits_receipt_before_queue_admission`, `tests/test_service_input_queue.py::test_worker_consumes_precommitted_receipt_without_duplicate`, `tests/test_service_input_queue.py::test_listen_only_drop_keeps_precommitted_receipt_without_duplicate` |
| `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py` | canonical task-resolution model vocabulary | `tests/test_task_resolution_contracts.py::test_removed_model_facing_handles_are_absent` |
| `scripts/validate_test_impact.py` | explicit-file source ownership inventory | `tests/test_test_impact_manifest.py::test_manifest_accepts_an_explicit_package_init_source_root` |
| `tests/test_cognition_core_v2_prompt_budget_continuity.py` | protected diagnostic artifact policy | `tests/test_cognition_core_v2_prompt_budget_continuity.py::test_captured_near_cap_repairs_fit_the_existing_budget[a1a573_near_cap_semantic_repair]`; same node for `caad1a_near_cap_semantic_repair` and `df6eb4_near_cap_semantic_repair` |

## Acceptance gates

1. Product version is `1.0.0` in one source of truth, exposed by the runtime
   and console package, and emitted by a built wheel; the package classifier is
   `Development Status :: 5 - Production/Stable`.
2. English and Chinese top-level release status identifies v1.0.0 and retains
   the explicit permissioned-preview boundary for autonomous contact.
3. No script imports raw MongoDB or a private DB module; the trace-manifest
   script uses the public DB facade.
4. The scoped deterministic tests pass, including service, console, resolver,
   self-cognition, documentation, and packaging metadata checks.
5. The full default deterministic suite passes, with any unavailable protected
   diagnostic captures reported as explicit skips or restored committed
   fixtures rather than collection-time failures.
6. `python -m compileall -q src`, `git diff --check`, `python -m pip check`,
   and a no-dependency wheel build pass.
7. The final worktree contains only the release changes, the release commit is
   created, and annotated tag `v1.0.0` points at that commit. Remote publication
   remains for the user to perform separately.

## Execution record

- Plan created after clean-tree baseline capture.
- Implementation: completed in the release commit recorded in Git history.
- Verification: focused release gates passed with `145 passed, 3 skipped`;
  the repository-wide deterministic gate passed with `4350 passed, 8 skipped,
  979 deselected`; the post-commit owner-matrix gate passed with `1 passed`;
  the impact gate passed four exact nodes; the no-dependency wheel emitted
  `kazusa_ai_chatbot-1.0.0`; `pip check`, compilation, and diff checks passed.
- Protected near-cap diagnostic captures remain explicit skips because the
  private files are unavailable in this checkout and are intentionally ignored
  by Git.
- Release commit/tag: annotated local `v1.0.0` tag created after plan closeout.
