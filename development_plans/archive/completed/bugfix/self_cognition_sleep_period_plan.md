# self cognition sleep period plan

## Summary

- Goal: add one character-local sleep period that suppresses selected
  self-cognition triggers while leaving self-reflection, consolidation, and
  scheduled future cognition unchanged.
- Plan class: medium.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`; apply `cjk-safety` before editing
  Python files that contain CJK string content.
- Overall cutover strategy: bigbang for the default local sleep period;
  compatible for operator opt-out with an empty period.
- Highest-risk areas: stopping reflection instead of only the
  reflection-attached group self-cognition sidecar, suppressing scheduled
  future cognition, and expanding the config surface.
- Acceptance criteria: default config sleeps from 02:00 to 12:00 local time;
  empty config preserves current behavior; sleep suppresses active-commitment
  self-cognition and reflection-attached group self-cognition; scheduled
  future cognition, reflection, and consolidation continue.

## Context

Read-only diagnostic artifacts under
`test_artifacts/self_cognition_night_review/` show heavy nighttime
self-cognition activity: 361 of the latest 833 self-cognition events occurred
between 21:00 and 05:59 local time. Most wrote character state, and recent
residue rows showed repeated overdue-promise rumination.

The confirmed product decision is narrow:

- Humans sleep; Kazusa should have one sleep period where selected
  self-cognition triggers do not run.
- Memory consolidation and self-reflection continue during sleep.
- Scheduled future cognition is unchanged.
- Config is one sleep-period value, not a set of per-source controls.

Current ownership boundary: the self-cognition worker keeps scheduled future
cognition first and suppresses only active commitments; the reflection worker
keeps hourly/daily/global reflection and suppresses only its group
self-cognition sidecar; consolidation, scheduler, dispatcher, and adapters are
unchanged.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, or recording
  evidence for this plan.
- `local-llm-architecture`: load before changing cognition, reflection,
  scheduler, RAG, prompt, or background LLM behavior.
- `py-style` and `test-style-and-execution`: load before editing Python or
  test files.
- `cjk-safety`: load before editing Python files containing CJK strings.

## Mandatory Rules

- Do not execute this plan while `Status` is `draft`.
- Before execution, check `git status --short`, read `README.md`,
  `docs/HOWTO.md`, this plan, and relevant subsystem READMEs.
- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual edits.
- Do not read `.env`.
- Add exactly one config value: `CHARACTER_SLEEP_LOCAL_PERIOD`.
- Do not add enable flags, per-trigger sleep flags, weekday schedules, mood
  rules, per-channel rules, or user-specific rules.
- Do not stop or gate hourly reflection, daily reflection, style update,
  global reflection promotion, global growth, consolidation, scheduler
  claiming, dispatcher validation, or adapter delivery.
- Do not change scheduled future cognition collection, claiming, processing,
  completion marking, or delivery binding.
- Do not add, remove, or rewrite any LLM prompt. Do not add new LLM calls.
- Do not change MongoDB schemas, indexes, collection names, or historical
  data.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before completion, lifecycle changes, merge, or sign-off, run the
  `Independent Code Review` gate and record the result in `Execution Evidence`.
- Use parent-led native subagent execution unless the user explicitly approves
  fallback execution.

## Must Do

- Add `CHARACTER_SLEEP_LOCAL_PERIOD`, defaulting to `02:00-12:00`.
- Validate non-empty values as exact `HH:MM-HH:MM` local wall-clock ranges.
- Support overnight ranges such as `23:30-07:30`.
- Treat an empty period as disabled and preserve current behavior.
- Add one deterministic self-cognition sleep predicate shared by source
  selection and the reflection group-review sidecar.
- During sleep, prevent production source selection from calling
  `collect_active_commitment_cases(...)`.
- During sleep, prevent `_run_group_self_cognition_review(...)` from
  collecting group cases or running the self-cognition worker tick.
- Keep `collect_scheduled_future_cognition_cases(...)` first and unchanged.
- Cover config parsing, period matching, active-commitment suppression,
  scheduled future preservation, and group-review suppression with focused
  deterministic tests.
- Update `docs/HOWTO.md`,
  `src/kazusa_ai_chatbot/self_cognition/README.md`,
  `src/kazusa_ai_chatbot/reflection_cycle/README.md`, and the registry.

## Deferred

- Do not redesign self-cognition memory semantics.
- Do not change self-cognition prompts, dialog wording, route selection, or
  action-candidate generation.
- Do not change scheduled future cognition behavior at night.
- Do not pause worker loops, the service, reflection/self-reflection, global
  growth, consolidation, or reflection promotion.
- Do not add sleep state storage, sleep events, UI controls, command handlers,
  adapter overrides, or scheduler rewrites.
- Do not change action attempt ledgers, event logs, MongoDB indexes,
  historical records, response ratio, group relevance, or engagement policy.

## Cutover Policy

Overall strategy: bigbang for the default local sleep period; compatible for
operator opt-out with an empty period.

| Area                         | Policy     | Instruction                                                                 |
| ---------------------------- | ---------- | --------------------------------------------------------------------------- |
| Config default               | bigbang    | Default `CHARACTER_SLEEP_LOCAL_PERIOD=02:00-12:00`.                        |
| Period parsing               | bigbang    | Use only `HH:MM-HH:MM`; invalid non-empty values fail fast.                 |
| Active commitment source     | bigbang    | During sleep, skip this source in production source selection.              |
| Group self-cognition review  | bigbang    | During sleep, skip before profile fetch, case collection, and worker tick.  |
| Scheduled future cognition   | compatible | Preserve due-slot collection, delivery binding, processing, and completion. |
| Reflection and consolidation | compatible | Preserve existing reflection cycle and consolidation behavior.              |
| Database                     | compatible | No migration, backfill, schema change, or historical rewrite.               |

Enforcement: follow the table exactly; bigbang areas keep no unsuppressed path,
compatible areas preserve only listed surfaces, and policy changes need user
approval.

## Target State

Operators can set:

```text
CHARACTER_SLEEP_LOCAL_PERIOD=02:00-12:00
```

When the current instant projected into `CHARACTER_TIME_ZONE` falls inside the
period:

- the standalone self-cognition collector still checks due scheduled future
  cognition;
- the standalone self-cognition collector does not call active commitment
  collection;
- the reflection worker still runs hourly, daily, style, promotion, and global
  reflection work;
- the reflection worker's group self-cognition review sidecar returns a
  skipped result with `defer_reason="self-cognition sleep period"` before
  profile fetch, group case collection, or worker tick.

When the period is empty or the current local time is outside the period,
behavior remains the same as current code.

## Design Decisions

| Topic                  | Decision                                         | Rationale                                                           |
| ---------------------- | ------------------------------------------------ | ------------------------------------------------------------------- |
| Config shape           | Use one `CHARACTER_SLEEP_LOCAL_PERIOD` string.   | The accepted design is one sleep period.                            |
| Time basis             | Use `CHARACTER_TIME_ZONE`.                       | Existing character-local scheduling already uses this boundary.     |
| Empty value            | Empty string disables sleep suppression.         | Operators can intentionally opt out.                                 |
| Format                 | Exact `HH:MM-HH:MM`; start and end must differ.  | This is deterministic and avoids ambiguous full-day sleep.          |
| Overnight support      | Start later than end wraps across midnight.      | Human sleep commonly crosses midnight.                              |
| Active commitment gate | Gate in `collect_self_cognition_cases(...)`.     | Production source selection owns whether a source triggers.         |
| Group review gate      | Gate in `_run_group_self_cognition_review(...)`. | Reflection owns the sidecar cadence; reflection itself stays awake. |
| Scheduled future       | Leave first and unchanged.                       | The user explicitly kept it unchanged.                              |

## Contracts And Data Shapes

### Config Contract

Add this constant in `src/kazusa_ai_chatbot/config.py`:

```python
CHARACTER_SLEEP_LOCAL_PERIOD = _optional_local_period_from_env(
    "CHARACTER_SLEEP_LOCAL_PERIOD",
    "02:00-12:00",
)
```

Parser contract: `""` and whitespace-only input normalize to `""`; non-empty
values must match exact `HH:MM-HH:MM`; hours must be `00` through `23`; minutes
must be `00` through `59`; start and end must differ; invalid non-empty values
raise `ValueError` during config import.

### Sleep Predicate Contract

Create `src/kazusa_ai_chatbot/self_cognition/sleep_period.py` with this public
entrypoint:

```python
def is_self_cognition_sleep_period(
    now: datetime,
    *,
    sleep_local_period: str = CHARACTER_SLEEP_LOCAL_PERIOD,
    character_time_zone: str = CHARACTER_TIME_ZONE,
) -> bool:
    ...
```

Predicate contract: `now` must be timezone-aware; empty
`sleep_local_period` returns `False`; the function projects `now` into
`character_time_zone` and compares only local time of day; same-day periods
include the start minute and exclude the end minute; overnight periods include
local times greater than or equal to the start minute or less than the end
minute; the function performs no source selection and no side effects.

### Trigger Contract

| Source                           | Outside sleep    | During sleep                                                    |
| -------------------------------- | ---------------- | --------------------------------------------------------------- |
| Scheduled future cognition       | Current behavior | Current behavior                                                |
| Active commitment due check      | Current behavior | Not collected by production `collect_self_cognition_cases(...)` |
| Reflection-attached group review | Current behavior | Skipped before case collection and worker tick                  |
| Hourly/daily/global reflection   | Current behavior | Current behavior                                                |

## LLM Call And Context Budget

No prompt, model, or context payload changes are approved. During sleep,
active-commitment and group-review self-cognition produce zero cases and zero
LLM calls. Scheduled future, live chat, RAG, dialog, reflection,
consolidation, and scheduler LLM budgets are unchanged.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/self_cognition/sleep_period.py`: deterministic
  `is_self_cognition_sleep_period(...)`; module docstring states it gates
  self-cognition triggers only.
- `tests/test_self_cognition_sleep_period.py`: empty, same-day, overnight,
  boundary, naive-datetime, and timezone projection cases.

### Modify

- `src/kazusa_ai_chatbot/config.py`: optional local-period parser and
  `CHARACTER_SLEEP_LOCAL_PERIOD`.
- `src/kazusa_ai_chatbot/self_cognition/sources.py`: skip active commitments
  during sleep while preserving scheduled future cognition.
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`: skip only group review
  during sleep.
- `tests/test_config.py`: default, valid, normalized, overnight, and invalid
  config cases.
- `tests/test_self_cognition_integration.py`: active-commitment sleep skip and
  scheduled future preservation cases.
- `tests/test_reflection_cycle_stage1c_worker.py`: group-review sleep skip and
  hourly-reflection preservation cases.
- `docs/HOWTO.md`, `src/kazusa_ai_chatbot/self_cognition/README.md`, and
  `src/kazusa_ai_chatbot/reflection_cycle/README.md`: document the operator
  contract, predicate ownership, and unchanged paths.
- `development_plans/README.md`: keep lifecycle status current.

### Keep

- `src/kazusa_ai_chatbot/self_cognition/runner.py` and
  `src/kazusa_ai_chatbot/self_cognition/worker.py`: no runner or loop cadence
  change.
- Scheduled tasks, action dispatch, adapter delivery, reflection non-sidecar
  work, MongoDB schemas, indexes, and historical data: unchanged.

## Overdesign Guardrail

- Actual problem: nighttime active-commitment and group-review self-cognition
  can amplify rumination when the character should be asleep.
- Minimal change: one local sleep-period config and one deterministic predicate
  suppress only the approved self-cognition trigger sources.
- Ownership boundaries: deterministic code owns config, time matching, and
  trigger selection; reflection owns reflection cadence; self-cognition owns
  source collection; LLM stages judge only created cases.
- Rejected complexity: no dynamic sleep model, mood detector, weekday rules,
  per-source flags, adapter command, persistence state, scheduler rewrite,
  prompt instruction, retry path, or compatibility alias.
- Evidence threshold: add complexity later only after approved product scope
  requires multiple windows, runtime operator controls, or persisted sleep
  events and tests or production evidence show the single-period gate is
  insufficient.

## Agent Autonomy Boundaries

- The responsible agent may choose local mechanics only when they preserve the
  contracts in this plan.
- Do not introduce new architecture, migrations, compatibility layers,
  fallback paths, helper aliases, or extra features.
- Treat changes outside `Change Surface` as out of scope unless the plan is
  updated first.
- Search for existing time-boundary helpers before adding code; reuse or
  extract equivalent behavior into `self_cognition/sleep_period.py`.
- Do not perform unrelated cleanup, formatting churn, dependency upgrades,
  prompt rewrites, or broad refactors.
- If the plan and code disagree, preserve the plan's intent and report the
  discrepancy.
- If a required instruction is impossible, stop and report the blocker.

## Implementation Order

1. Add failing config tests in `tests/test_config.py`.
2. Add failing predicate tests in `tests/test_self_cognition_sleep_period.py`.
3. Implement config parsing and `self_cognition/sleep_period.py`; rerun config and
   predicate tests.
4. Add self-cognition integration tests proving sleep skips active commitment
   collection and preserves scheduled future cognition.
5. Wire `self_cognition/sources.py`; rerun the self-cognition tests.
6. Add reflection worker tests proving group review sleeps and hourly
   reflection still runs.
7. Wire `reflection_cycle/worker.py`; rerun the reflection tests.
8. Update docs and registry lifecycle text; run static greps and focused
   regression.
9. Run independent code review and remediate only in-scope findings.

## Execution Model

- Parent agent owns orchestration, tests, verification, evidence, review
  remediation, lifecycle updates, and final sign-off.
- Parent establishes the focused test contract and records expected failures
  before production implementation starts.
- Production-code subagent: exactly one native subagent after the focused test
  contract exists; owns production code changes only; closes after planned
  production changes are complete.
- Parent may continue integration tests, static checks, and evidence while the
  production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent after planned
  verification passes; reviews the plan, diff, and evidence; does not
  implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - config and predicate contract established. Covers steps 1-3; verify `venv\Scripts\python -m pytest tests\test_config.py tests\test_self_cognition_sleep_period.py -q`.
- [x] Stage 2 - self-cognition source selection wired. Covers steps 4-5.
  Verify `venv\Scripts\python -m pytest tests\test_self_cognition_integration.py::test_collect_self_cognition_cases_skips_active_commitments_during_sleep tests\test_self_cognition_integration.py::test_collect_self_cognition_cases_keeps_scheduled_future_slots_during_sleep -q`; record baseline failure, changed files, passing output, and `<agent/date>`.
- [x] Stage 3 - reflection group sidecar wired. Covers steps 6-7.
  Verify `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_worker.py::test_group_self_cognition_review_skips_cases_during_sleep tests\test_reflection_cycle_stage1c_worker.py::test_worker_tick_keeps_hourly_reflection_during_sleep -q`; record baseline failure, changed files, passing output, and `<agent/date>`.
- [x] Stage 4 - docs, regression, and review complete. Covers steps 8-9; run all `Verification` commands and record review output, risks, and sign-off.

## Verification

### Static Greps

- `rg -n "CHARACTER_SLEEP_LOCAL_PERIOD|is_self_cognition_sleep_period" src tests docs development_plans`
  - Expected: matches only approved config, helper, source-selection,
    reflection worker, docs, tests, and this plan.
- `rg -n "self-cognition sleep period" src tests docs development_plans`
  - Expected: matches only the group-review skipped reason, tests, docs, and
    this plan.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_config.py tests\test_self_cognition_sleep_period.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_integration.py::test_collect_self_cognition_cases_skips_active_commitments_during_sleep tests\test_self_cognition_integration.py::test_collect_self_cognition_cases_keeps_scheduled_future_slots_during_sleep -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_worker.py::test_group_self_cognition_review_skips_cases_during_sleep tests\test_reflection_cycle_stage1c_worker.py::test_worker_tick_keeps_hourly_reflection_during_sleep -q`

### Focused Regression

- `venv\Scripts\python -m pytest tests\test_self_cognition_integration.py tests\test_self_cognition_group_review_source.py tests\test_reflection_cycle_stage1c_worker.py -q`

No live LLM or DB test is required because prompts, models, schemas, indexes,
migrations, and real database write behavior are unchanged.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Use one native independent code-review subagent when available.

Review scope: project rules, style, ownership boundaries, hidden fallbacks,
prompt/RAG leaks, persistence risk, fixture brittleness, blast radius, and
alignment with this plan. Fix findings only inside the approved change surface
or review-only fixture/docs corrections. Record findings, fixes, reruns,
residual risks, and approval status in `Execution Evidence`.

## Execution Evidence

- Parent verification on 2026-05-22: `45 passed`, `7 passed`, and
  `51 passed`; `py_compile` and `git diff --check` passed; static greps
  matched approved surfaces.
- Independent review subagent Russell found no blocking or important issues.
  Minor tick-level test weakness fixed; duplicate parser risk accepted because
  config import and the predicate validate their own inputs.
- Parent sign-off: Codex, 2026-05-22.

## Acceptance Criteria

This plan is complete when:

- `CHARACTER_SLEEP_LOCAL_PERIOD` exists, defaults to `02:00-12:00`, validates
  non-empty values, supports overnight local periods, and preserves current
  behavior when explicitly empty.
- During sleep, production `collect_self_cognition_cases(...)` still returns
  due scheduled future cognition cases and does not call active commitment
  collection.
- During sleep, `_run_group_self_cognition_review(...)` returns a skipped
  result before profile fetch, group case collection, or worker tick.
- Reflection, consolidation, scheduler, dispatcher, and adapter behavior are
  unchanged outside the approved sidecar gate.
- Docs state what sleeps and what remains awake; verification commands pass;
  independent review has no unresolved blocking findings.
