# Legacy LLM Configuration Cleanup Plan

## Summary

- Goal: Remove the ten audited legacy `.env` keys and the three stale
  Required Selection Verifier Compose bindings.
- Status: completed
- Scope boundary: Local root `.env`, `docker-compose.yml`, and this plan's
  lifecycle registry entries.
- Change direction: Big-bang deletion of configuration that has no current
  runtime owner.
- Acceptance state: Accepted on 2026-08-08 after the exact legacy names were
  removed and focused deployment checks passed.

## Confirmed Decisions

- The user's 2026-08-08 instruction to perform the regression map's
  recommended action authorizes this exact cleanup.
- The current Cognition V2 producer contracts remain authoritative. Required
  selections use the ordinary-goal route and same-stage regeneration.
- Current dialog generation and specialist RAG limits remain authoritative.
- Secret values remain private; inspection and evidence expose key names and
  counts only.

## Scope And Change Direction

Configuration ownership moves fully to the current runtime routes and their
bounded producer contracts. Remove only configuration that the accepted
baseline feature regression map classifies as `legacy_config_unmapped`.

The cleanup changes deployment inputs, not cognition, dialog, RAG, adapter,
persistence, scheduler, or delivery behavior.

## Mandatory Skills

- Apply `.agents/skills/development-plan` for execution and lifecycle
  closeout.
- Apply `.agents/skills/test-style-and-execution` before running focused
  pytest verification.

## Mandatory Rules

- Preserve all unrelated worktree changes, including the current character
  carryover Compose bindings.
- Keep `.env` values out of command output, plan evidence, and the final
  report.
- Keep historical plan and diagnostic references intact as execution history.
- Add no compatibility aliases, fallback routes, or replacement settings.

## Must Do

Remove these keys from the local root `.env`:

- `COGNITION_LLM_REQUIRED_SELECTION_VERIFIER_BASE_URL`
- `COGNITION_LLM_REQUIRED_SELECTION_VERIFIER_API_KEY`
- `COGNITION_LLM_REQUIRED_SELECTION_VERIFIER_MODEL`
- `COGNITION_LLM_REQUIRED_SELECTION_VERIFIER_MAX_COMPLETION_TOKENS`
- `COGNITION_LLM_REQUIRED_SELECTION_VERIFIER_THINKING_ENABLED`
- `DIALOG_EVALUATOR_LLM_BASE_URL`
- `DIALOG_EVALUATOR_LLM_API_KEY`
- `DIALOG_EVALUATOR_LLM_MODEL`
- `MAX_DIALOG_AGENT_RETRY`
- `MAX_FACT_HARVESTER_RETRY`

Remove these bindings from `docker-compose.yml`:

- `COGNITION_LLM_REQUIRED_SELECTION_VERIFIER_BASE_URL`
- `COGNITION_LLM_REQUIRED_SELECTION_VERIFIER_API_KEY`
- `COGNITION_LLM_REQUIRED_SELECTION_VERIFIER_MODEL`

Record key-name-only verification and close this plan through the normal
registry lifecycle.

## Deferred

- Changes to Python source, tests, prompts, ICDs, or runtime behavior.
- Removal of any other `.env` key, including unrelated unreferenced keys.
- Changes to the retained generic `COGNITION_LLM` route or any current
  Cognition V2 stage route.
- Cleanup of historical plan and diagnostic references.

## Target State

- The ten named keys occur zero times as assignments in the local `.env`.
- The three named verifier bindings occur zero times in `docker-compose.yml`.
- Compose still forwards every fail-fast environment variable required by
  `src/kazusa_ai_chatbot/config.py`.
- Required-selection behavior continues to use
  `COGNITION_LLM_GOAL_ORDINARY_RESPONSE` with no independent verifier route.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Local `.env` | bigbang | Delete the ten named legacy assignments in one cleanup. |
| Compose | bigbang | Delete the three stale verifier bindings without replacement. |
| Runtime behavior | keep | Preserve the current V2, dialog, and RAG contracts unchanged. |
| Historical evidence | keep | Retain archived references that explain the removed configuration. |

## Change Surface

### Modify

- `.env`: delete the ten exact legacy assignments without exposing values.
- `docker-compose.yml`: delete the three exact stale verifier bindings while
  preserving current route bindings and unrelated edits.
- `development_plans/README.md`: register active execution and completed
  lifecycle history.

### Create Then Archive

- `development_plans/active/short_term/legacy_llm_configuration_cleanup_plan.md`:
  approved execution contract and evidence record; move to
  `development_plans/archive/completed/short_term/` after acceptance.

### Keep

- Production source, tests, runtime contracts, prompts, and documentation.
- Existing historical plans and diagnostics.

## Agent Autonomy Boundaries

The implementation owner may choose safe line-removal mechanics and the
smallest focused verification set that proves this contract. Any additional
key removal, runtime change, compatibility behavior, or documentation rewrite
requires a plan amendment and new user authorization.

## Verification

- Run a key-name-only `.env` assignment count for all ten keys and require
  zero for each.
- Scan `docker-compose.yml` for the three exact bindings and require no hits.
- Run the deployment consistency test that compares Compose bindings with
  fail-fast runtime configuration.
- Run the required-selection routing test that proves there is no independent
  model route.
- Run Compose configuration validation without rendering secret values.
- Inspect the final scoped diff and `git diff --check` output.

## Acceptance Criteria

- All ten exact `.env` assignments are absent.
- All three exact verifier Compose bindings are absent.
- Current route bindings, including character carryover, remain present.
- Focused deterministic tests and Compose validation pass.
- No secret value appears in execution evidence.
- The completed plan is archived and the registry reflects its final status.

## Progress Checklist

- [x] User authorization and exact scope confirmed.
- [x] Current runtime ownership and existing worktree overlap inspected.
- [x] Ten legacy `.env` assignments removed.
- [x] Three stale Compose bindings removed.
- [x] Focused verification passed.
- [x] Execution evidence recorded and lifecycle closed.

## Execution Evidence

- Pre-change key-name-only inventory: each of the ten target `.env` keys
  occurred exactly once.
- Pre-change tracked scan: the three verifier endpoint/model bindings occurred
  only in `docker-compose.yml`; current production config exposes no matching
  verifier route.
- Pre-change worktree review: `docker-compose.yml` also contains an unrelated
  current character-carryover binding addition, which this cleanup preserves.
- `.env` cleanup: a guarded key-name filter required one occurrence of every
  target assignment, removed ten assignments, and confirmed a final aggregate
  count of zero without emitting values.
- Compose cleanup: the three stale verifier bindings were removed; a scoped
  live-surface scan across Compose, README, docs, and source was clean.
- Deterministic tests:
  `tests/test_deployment_configuration.py::test_compose_passes_every_required_runtime_environment_variable`
  and
  `tests/test_cognition_core_v2_stage_model_routing.py::test_required_selection_has_no_independent_model_route`
  passed (`2 passed in 0.72s`).
- Compose artifact validation: the installed YAML parser loaded the document,
  confirmed the brain environment is a non-empty list, and found no duplicate
  entries.
- Native `docker compose config --quiet` was unavailable because the Docker
  CLI is not installed in this workspace. The YAML parse and deployment
  consistency test cover the changed static configuration surface.
- Final review: character-carryover bindings remain present, the scoped diff
  contains only the authorized verifier deletion alongside preserved prior
  work, and `git diff --check` passed.
- Lifecycle closeout: the completed plan is present only under
  `development_plans/archive/completed/short_term/`, and the registry points
  to that archived path.
- Result: accepted and archived as completed.
