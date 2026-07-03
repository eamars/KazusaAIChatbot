# daily affect settling plan

## Summary

- Goal: Add a persistent daily affect-settling pass that lets the active
  character's global mood and global vibe cool gradually after sleep without
  erasing user-specific memory, relationship evidence, or current conflict.
- Plan class: medium.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `cjk-safety`,
  `test-style-and-execution`, and `debug-llm`.
- Overall cutover strategy: additive background maintenance. Existing live
  chat, consolidation, user memory, daily reflection, promotion, calendar
  scheduling, and self-cognition behavior remain intact except for a narrow
  post-sleep worker pause that prevents self-cognition source collection from
  starting before an already due affect-settling pass reaches a terminal
  state.
- Highest-risk areas: deterministic semantic interpretation of free-form mood
  text, sharp mood jumps, stale background writes overwriting night messages,
  process-local runtime state becoming stale after a background write, and
  weak local LLM output being treated as safe without review.
- Acceptance criteria: the reflection worker can persist exactly one
  `daily_affect_settling` audit run per sleep cycle, ask LLM stages to propose
  and review a gradual free-form affect transition, write only
  `character_state.mood`, `character_state.global_vibe`,
  `character_state.reflection_summary`, and `character_state.updated_at` after
  a stale-write check, refresh runtime/cache state after success, and prove the
  timing, idempotency, stale-write, prompt-contract, and no-lookup-table
  behavior with deterministic and live-LLM verification.

## Planning Notes

The `development-plan` skill names
`development_plans/references/plan_contract.md`,
`development_plans/references/execution_gates.md`, and
`development_plans/references/cutover_policy.md`. Those files are not present
in this checkout. This draft follows `development_plans/README.md`, existing
active-plan style, and the relevant subsystem ICDs instead.

Implementation was explicitly requested on 2026-07-02. Execute this plan
without subagents, using parent-only fallback execution.

## Plan Review Record

Review performed on 2026-07-02 surfaced four issues, all addressed in this
draft:

- Stale skipped runs must be terminal and non-retryable for that sleep cycle.
- Raw `character_state.updated_at` and other operational control identifiers
  must not appear in proposal or reviewer prompts.
- The self-cognition coordination point must be a service/worker pause
  boundary, not source-case semantic filtering.
- Sleep timing must reject non-empty sleep-period configurations where affect
  settling would be due after the wake-defer grace window.

## Context

Kazusa currently stores global runtime affect in the singleton
`character_state` document:

- `mood`
- `global_vibe`
- `reflection_summary`
- `updated_at`

Live turns and consolidation can write these fields. Reflection runs outside
live chat and stores hourly, daily-channel, and daily-global-promotion audit
records in `character_reflection_runs`. Self-cognition sleep currently
suppresses self-cognition triggers during `CHARACTER_SLEEP_LOCAL_PERIOD`, but
reflection, consolidation, scheduler execution, dispatcher validation, adapter
delivery, and normal chat are not paused.

The target behavior is human-like affect settling: sleep can reduce emotional
sharpness and make the next active period less reactive, but it must not
pretend the conflict never happened. A person may remember the argument and
keep distance while no longer carrying the same immediate heat.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before editing prompt contracts, LLM payloads,
  worker integration, or semantic responsibility boundaries.
- `no-prepost-user-input`: load before editing affect-settling interpretation
  code or prompt-review behavior. Free-form `mood`, `global_vibe`, and
  `reflection_summary` must remain LLM-owned semantic text.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files containing CJK prompt text.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before running live LLM affect-settling checks or writing
  LLM evaluation artifacts.

## Mandatory Rules

- Do not add lookup tables, ladders, keyword maps, score maps, severity maps,
  polarity classifiers, sentiment classifiers, or deterministic clamps for
  `mood`, `global_vibe`, or `reflection_summary`.
- Do not deterministically decide whether one free-form mood phrase is
  stronger, weaker, angrier, calmer, closer, colder, safer, warmer, or more
  neutral than another.
- Do not rewrite, normalize, downgrade, upgrade, translate, classify, or
  otherwise semantically alter LLM-emitted affect fields in Python.
- Deterministic code may only validate JSON shape, required fields, string
  presence, string length bounds, idempotency, timing, stale-write compare
  tokens, cache invalidation, runtime-state refresh, and event/audit writes.
- Do not render `character_state.updated_at`, run ids, scheduler ids, compare
  tokens, leases, or other operational control identifiers into proposal or
  reviewer prompts. Keep those values in deterministic code only.
- The self-cognition wake pause is an operational worker-pause boundary only.
  It must not filter, rewrite, rank, fabricate, or suppress individual
  self-cognition source cases by semantic content.
- The LLM owns the semantic affect transition. A separate LLM reviewer owns
  whether the proposed transition is gradual and continuous enough to persist.
- If the affect-settling proposal or reviewer output is structurally invalid,
  overlong, missing required fields, or reviewer-rejected, persist a skipped or
  failed audit run and do not write `character_state`.
- The settling pass must never write `user_profiles`, `user_memory_units`,
  `conversation_episode_state`, `interaction_style_images`, `memory`,
  `global_character_growth_traits`, dispatcher rows, adapter output, or
  conversation history.
- The settling pass must never call the live cognition resolver or dialog
  graph.
- The settling pass must not add a live response-path LLM call.
- The prompt must describe gradual sleep settling in plain language, not by
  showing a mood hierarchy or example ladder.
- Prompt examples must not hard-code a concrete character name unless the
  handler injects the runtime `character_profile["name"]`; prefer role-neutral
  wording.
- Use the existing consolidation LLM route for affect-settling LLM calls unless
  a later approved plan introduces a dedicated route.
- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or final
  reporting.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  an independent code review gate and record the result in `Execution
  Evidence`.

## Scope

This plan owns persistent global affect settling for the active character after
a sleep period.

In scope:

- A new reflection-cycle affect-settling module.
- A new `daily_affect_settling` reflection run kind and deterministic run id.
- A new public reflection facade for manual and worker affect settling.
- Worker integration that schedules the pass near the end of the configured
  sleep period and after daily global promotion has had a chance to run.
- A structural compare-and-set character-state writer based on
  `character_state.updated_at`.
- Cache invalidation and process-local runtime-state refresh after successful
  writes.
- A short self-cognition wake pause for due-but-not-terminal affect settling.
- Config, docs, deterministic tests, and live LLM prompt review artifacts.

Out of scope:

- User-specific mood decay.
- User image, affinity, relationship insight, user memory, or conversation
  progress mutation.
- Any deterministic semantic mood/vibe interpretation.
- Any response-path change to L1, L2, L2d, L3, dialog, relevance, or RAG.
- Any adapter delivery behavior.
- Any delayed visible text.
- Any migration of existing `character_state` documents.

## Architecture

Add `src/kazusa_ai_chatbot/reflection_cycle/affect_settling.py`.

This module owns:

- `AFFECT_SETTLING_PROMPT_VERSION`
- `AFFECT_SETTLING_SYSTEM_PROMPT`
- `AFFECT_SETTLING_REVIEW_SYSTEM_PROMPT`
- `build_affect_settling_payload(...)`
- `build_affect_settling_prompt(...)`
- `run_affect_settling_llm(...)`
- `run_affect_settling_review_llm(...)`
- `run_daily_affect_settling(...)`
- `_run_daily_affect_settling(...)`

The reflection package facade exports:

```python
async def run_daily_affect_settling(
    *,
    settling_local_date: str,
    dry_run: bool,
    enable_character_state_write: bool,
) -> ReflectionWorkerResult: ...
```

`settling_local_date` is the character-local date containing the sleep end.
The pass may read previous-day daily reflection/promotion documents and
sleep-window hourly reflection documents, but it writes only character-state
runtime fields.

## Timing Contract

Use `CHARACTER_SLEEP_LOCAL_PERIOD` as the source of the sleep window. Empty
sleep period disables affect settling.

Keep only this env-backed affect-settling config:

- `AFFECT_SETTLING_WAKE_PREP_MINUTES=30`

Keep the remaining affect-settling policy values as named module constants in
`kazusa_ai_chatbot.reflection_cycle.affect_settling`, not environment
settings:

- `AFFECT_SETTLING_PROMPT_MAX_CHARS=12000`
- `AFFECT_SETTLING_REVIEW_PROMPT_MAX_CHARS=8000`
- `AFFECT_SETTLING_AFTER_PROMOTION_GRACE_MINUTES=15`
- `AFFECT_SETTLING_WAKE_DEFER_GRACE_MINUTES=15`

For a sleep cycle ending on `settling_local_date`, compute the due local time
as:

```text
max(
  REFLECTION_PROMOTION_RUN_AFTER_LOCAL_TIME
    + AFFECT_SETTLING_AFTER_PROMOTION_GRACE_MINUTES,
  sleep_end_local_time - AFFECT_SETTLING_WAKE_PREP_MINUTES
)
```

With the current defaults:

```text
sleep period: 02:00-12:00
promotion gate: 05:00
promotion grace: 15 minutes
wake prep: 30 minutes
affect settling due: 11:30
```

This timing lets night and morning sleep-window messages be observed by normal
message persistence and hourly reflection while still settling global affect
before self-cognition wakes.

If the service is down at the due time, the reflection worker catches up once
for only the current or immediately preceding sleep cycle, subject to the
idempotency and stale-write rules below.

Timing validation:

- Validate both same-day and overnight sleep windows from
  `CHARACTER_SLEEP_LOCAL_PERIOD`.
- Reject a non-empty sleep-period affect-settling configuration when the
  computed due time is later than
  `sleep_end + AFFECT_SETTLING_WAKE_DEFER_GRACE_MINUTES`; otherwise
  self-cognition can wake before affect settling is ever allowed to run.
- Treat an empty `CHARACTER_SLEEP_LOCAL_PERIOD` as disabling both affect
  settling and the wake pause.
- Do not backfill old sleep cycles. Catch-up is limited to the current or
  immediately preceding sleep cycle.

## LLM Contract

The affect proposal LLM receives compact prompt-safe context:

- Current `character_state.mood` as quoted free-form text.
- Current `character_state.global_vibe` as quoted free-form text.
- Current `character_state.reflection_summary` as quoted free-form text.
- A natural-language statement that the supplied affect fields are current as
  of prompt build time. The raw `character_state.updated_at` compare token is
  not rendered into the prompt.
- `settling_local_date`.
- A natural-language sleep context such as: the character has been inside the
  configured sleep period for most of the sleep window, and self-cognition has
  been suppressed.
- Previous-day daily reflection cards when available.
- Sleep-window hourly reflection cards when available.
- Validation notes such as missing daily reflection or partial sleep-window
  reflection input.

The proposal prompt instructs:

- Treat the current affect fields as free-form source text.
- Preserve continuity with the previous state.
- Reduce emotional sharpness only when psychologically plausible after sleep.
- Keep residual tension if the sleep-window evidence supports it.
- Do not invent forgiveness, affection, apology, repair, or trust.
- Do not erase user-specific conflict; user-specific evidence remains outside
  this pass.
- Output natural short descriptors, not enum labels.

The proposal output shape is:

```json
{
  "mood": "free-form short text",
  "global_vibe": "free-form short text",
  "reflection_summary": "free-form summary text"
}
```

The reviewer LLM receives the original affect fields, the same compact
evidence notes, and the proposed output. It decides whether the proposal is a
gradual, psychologically plausible sleep-settling transition.

The reviewer output shape is:

```json
{
  "write_decision": "accept|reject",
  "review_reason": "free-form text"
}
```

Python code may honor `write_decision == "accept"` as an LLM-authored control
field. It must not independently interpret the mood/vibe text.

The stale-write compare token remains in Python. It is not an instruction to
the LLM and is not evidence of the character's emotional state.

## Persistence Contract

Add a reflection run kind:

```python
REFLECTION_RUN_KIND_DAILY_AFFECT_SETTLING = "daily_affect_settling"
```

Add a deterministic run id helper:

```text
daily_affect_settling:{settling_local_date}:{prompt_version}
```

The run document uses the existing `character_reflection_runs` collection with
system scope:

```text
scope_ref="daily_affect_settling"
platform="system"
platform_channel_id="global"
channel_type="system"
```

Persist the proposal parsed output, reviewer parsed output, source reflection
run ids, validation warnings, attempt count, status, and error text. Do not
persist raw source messages, prompt text, user ids, raw channel ids, or full
character state documents in event logs. The run document may store the
bounded parsed LLM outputs for audit, following the existing reflection-run
audit model.

Add a structural DB helper in `db.character`:

```python
async def compare_and_upsert_character_state(
    *,
    expected_updated_at: str,
    mood: str,
    global_vibe: str,
    reflection_summary: str,
    updated_at_utc: str,
) -> bool: ...
```

The helper matches `_id="global"` and `updated_at=expected_updated_at`. It
updates only the three affect fields and `updated_at`. It returns `False` when
the compare token is stale. It performs no semantic interpretation.

After a successful character-state write:

- invalidate Cache2 dependencies for `character_state` using the existing cache
  invalidation path used by character-state writers;
- call an optional reflection-worker callback supplied by service startup to
  refresh process-local runtime character state;
- record a sanitized database-operation event for the character-state write.

## Run Status And Retry Policy

Terminal status must be explicit because stale state and reviewer rejection are
valid outcomes, not invitations to keep re-running until a calmer answer wins.

- `succeeded`: accepted proposal was written to `character_state`. Never rerun
  for the same `settling_local_date` and prompt version.
- `skipped`: no character-state write occurred and the sleep cycle is terminal
  when the reason is `stale_character_state`, `reviewer_rejected`,
  `proposal_structurally_invalid`, `missing_character_state`,
  `settling_window_expired`, or `write_disabled`.
- `failed`: infrastructure or transient execution failure before a semantic
  terminal decision. This may be retried under the existing bounded worker
  attempt policy.
- Dry-run output never blocks a later apply run and never marks a sleep cycle
  terminal.

The worker must treat non-retryable skipped runs as complete for wake-pause
purposes. This prevents a night or morning message from repeatedly causing new
affect-settling attempts against newer live mood.

## Worker Integration

Integrate affect settling into `reflection_cycle.worker` after daily
promotion/growth maintenance and before the configured sleep period ends.

The worker-level maintenance key must include:

```text
period_start
settling_local_date
daily_affect_settling
```

The durable run id remains the source of truth across process restarts.

Manual CLI support belongs in `src/scripts/run_reflection_cycle.py` as an
`affect-settle` command with dry-run default and explicit apply/write flag.

Service startup passes an optional callback into the reflection worker:

```python
character_state_refresh_callback=_refresh_runtime_character_state
```

The reflection worker must not import service state directly.

## Self-Cognition Wake Pause

Add a narrow operational pause so the self-cognition worker does not start
post-sleep source collection before a due affect-settling pass reaches a
terminal state.

Implement this as a service-wired pause probe or callback at the worker
boundary. `self_cognition` must not import `reflection_cycle` internals or
inspect affect-settling run documents directly. The service layer that owns
worker startup and scheduling may provide a callable such as:

```text
should_pause_self_cognition_for_affect_settling(now) -> bool
```

The pause:

- applies only when `CHARACTER_SLEEP_LOCAL_PERIOD` is non-empty;
- applies only from affect-settling due time through
  `sleep_end + AFFECT_SETTLING_WAKE_DEFER_GRACE_MINUTES`;
- checks only whether the current sleep cycle has a terminal
  `daily_affect_settling` run document;
- is evaluated before source collection, prompt building, and LLM calls;
- records a skipped/deferred worker tick when active and returns without
  mutating pending calendar rows or self-cognition source cases;
- stops blocking after the grace window, even if affect settling failed or was
  skipped.

The pause must not inspect or interpret mood, vibe, reflection summaries, user
messages, or LLM output.

The pause is not semantic cognition logic. It is only a short scheduling pause
to let the persisted global affect settling complete before the first wake
self-cognition pass observes the morning state.

## Failure Modes And Required Handling

| Failure mode | Required handling |
|---|---|
| User sends a message after prompt build but before affect write | Re-read `character_state.updated_at`; if changed, persist a non-retryable skipped run with `stale_character_state` and do not write. Treat that sleep cycle as terminal for wake-pause purposes. |
| User sends a night/morning message before due time | Include current character state and available sleep-window reflection evidence; let the LLM decide whether tension remains. |
| Service down at due time | Catch up once after startup/tick for only the current or immediately preceding sleep cycle using durable run-id idempotency; do not run repeatedly or backfill old cycles. |
| Daily reflection missing | Pass missing-input notes to the LLM; if proposal/reviewer accepts, allow write. Persist validation warning. |
| Sleep-window hourly reflection missing or partial | Pass partial-input notes to the LLM; do not block by default. Persist validation warning. |
| Proposal JSON malformed | Persist failed/skipped audit run; do not write character state. Use the existing JSON parser/repair helper only for malformed JSON string recovery. |
| Reviewer rejects proposal | Persist a non-retryable skipped audit run with reviewer reason; do not write character state. |
| Proposal strings missing or over length | Persist a non-retryable skipped audit run; do not truncate semantically meaningful text into state fields. |
| Character state missing | Persist a non-retryable skipped run with validation warning; do not seed a new profile/state document from affect settling. |
| Cache invalidation or runtime refresh fails after DB write | Record runtime error event; DB write remains committed. Next live chat refresh still reads DB state. |
| Event logging fails | Continue normal production behavior; event logging is best-effort. |
| Self-cognition worker reaches wake boundary while affect settling is due but non-terminal | Service pause probe defers the worker tick before source collection until terminal run or grace expiry; no source cases are filtered or mutated. |

## Must Do

- Keep only `AFFECT_SETTLING_WAKE_PREP_MINUTES` in `config.py`; keep the
  remaining affect-settling policy values as named constants in
  `reflection_cycle.affect_settling` with fail-fast timing validation there.
- Add `daily_affect_settling` constants and result typing in
  `reflection_cycle.models`.
- Add reflection repository helpers for deterministic run ids and run lookup
  support where needed.
- Add `reflection_cycle.affect_settling` with proposal and reviewer LLM blocks.
- Add compact payload builders that pass free-form mood/vibe/reflection text
  quoted as source text and do not classify it.
- Add structural validation for JSON shape, required string fields, and length
  bounds only.
- Add `db.character.compare_and_upsert_character_state(...)`.
- Wire worker timing into `reflection_cycle.worker`.
- Add a reflection facade export and manual CLI command.
- Add explicit run status and retry policy handling for `succeeded`,
  non-retryable `skipped`, and retryable `failed` outcomes.
- Add the optional service callback to refresh runtime character state after a
  successful affect-settling write.
- Add the short self-cognition wake pause as a service-provided worker pause,
  without a `self_cognition` -> `reflection_cycle` import.
- Update subsystem READMEs and HOWTO config documentation.
- Add deterministic tests and live LLM evaluation artifacts.

## Deferred

- Do not implement user-scoped affect decay.
- Do not change RAG retrieval weighting for user memories.
- Do not modify cognition prompts to reinterpret mood/vibe.
- Do not add a new LLM route.
- Do not add a new MongoDB collection unless a later approved plan requires
  affect-specific audit outside `character_reflection_runs`.
- Do not add operator UI changes.
- Do not add automatic repair prompts beyond the existing malformed-JSON parse
  helper.
- Do not backfill old sleep cycles.

## Cutover Policy

Overall strategy: compatible additive background pass.

| Area | Policy | Instruction |
|---|---|---|
| Reflection cycle | additive | Add `daily_affect_settling` beside existing daily maintenance; do not replace hourly, daily-channel, promotion, or growth passes. |
| Character state | compatible structural write | Add compare-and-set helper; existing `upsert_character_state` behavior remains unchanged. |
| Live chat | no response-path change | Do not add LLM calls or logic to the live response path. |
| Self-cognition | narrow worker pause | Add only the short due-but-not-terminal wake pause; do not change self-cognition source semantics. |
| Database | no migration | Use existing `character_state` and `character_reflection_runs`; bootstrap index changes only if required by new run-kind query patterns. |
| Config | narrow env surface | Keep only `AFFECT_SETTLING_WAKE_PREP_MINUTES` as deploy-time config. There is no `AFFECT_SETTLING_ENABLED` feature flag. Empty `CHARACTER_SLEEP_LOCAL_PERIOD` disables affect settling through the shared sleep-period contract. |

Operational disable:

```text
CHARACTER_SLEEP_LOCAL_PERIOD=
```

This stops future affect-settling runs and self-cognition wake pausing, while
also disabling the normal sleep-period self-cognition suppression. Existing
audit rows and character-state writes remain historical data.

## Implementation Checklist

1. Tests first: config and sleep-timing contract.
   - Add config validation tests for new affect settings.
   - Add deterministic tests for same-day and overnight sleep windows.
   - Prove non-empty sleep-period config is rejected when due time would fall after
     `sleep_end + AFFECT_SETTLING_WAKE_DEFER_GRACE_MINUTES`.

2. Tests first: persistence and stale-write contract.
   - Add DB-unit tests for `compare_and_upsert_character_state`.
   - Prove exact `updated_at` match writes and stale token returns `False`.
   - Prove `stale_character_state` becomes a non-retryable skipped run.

3. Tests first: prompt and no-lookup-table contract.
   - Add patched LLM tests proving free-form proposed fields are persisted
     exactly when reviewer accepts.
   - Prove `character_state.updated_at`, run ids, scheduler ids, compare
     tokens, and leases are absent from proposal and reviewer rendered prompts.
   - Add static review tests or grep checks proving no mood/vibe lookup table,
     severity map, sentiment classifier, or deterministic transition clamp is
     introduced in affect-settling code.

4. Implement affect-settling module.
   - Add prompt constants, LLM configs, payload builder, parser, validation,
     proposal call, reviewer call, run document persistence, and write flow.

5. Implement worker, facade, and CLI integration.
   - Add daily maintenance key, due-time calculation, idempotent run behavior,
     CLI dry-run/apply command, and event logging.

6. Implement runtime refresh and cache invalidation.
   - Add optional reflection-worker callback from service.
   - Invalidate character-state cache dependencies after successful writes.

7. Implement self-cognition wake pause.
   - Add the narrow due-window terminal-run check behind a service-provided
     pause probe/callback.
   - Prove self-cognition does not import reflection-cycle internals and does
     not inspect affect-settling run documents directly.
   - Prove active pause returns before source collection and does not mutate
     pending calendar rows or source cases.
   - Prove the pause expires after the configured grace window.

8. Update docs.
   - Update `reflection_cycle/README.md`, `self_cognition/README.md`,
     `db/README.md`, and `docs/HOWTO.md`.

9. Live LLM evaluation.
   - Run one case at a time with `debug-llm`.
   - Save artifacts under `test_artifacts/llm_traces/` or
     `test_artifacts/diagnostics/`.
   - Include at least:
     - angry global state with no new sleep-window conflict;
     - angry global state with a fresh sleep-window conflict.

10. Final verification and review.
    - Run deterministic test batches.
    - Run prompt rendering checks.
    - Run `py_compile` for touched Python files.
    - Run `git diff --check`.
    - Run independent code review and record findings.

## Required Verification

Use `venv\Scripts\python.exe` for Python commands.

Required deterministic checks:

```powershell
venv\Scripts\python.exe -m pytest tests\test_config.py -q
venv\Scripts\python.exe -m pytest tests\test_self_cognition_sleep_period.py -q
venv\Scripts\python.exe -m pytest tests\test_db.py -q
venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_stage1c_worker.py -q
venv\Scripts\python.exe -m pytest tests\test_reflection_affect_settling.py -q
venv\Scripts\python.exe -m pytest tests\test_self_cognition_integration.py -q
```

The affect-settling tests must cover:

- same-day and overnight sleep windows;
- due-time validation against wake-defer grace;
- succeeded, non-retryable skipped, and retryable failed run handling;
- stale `character_state.updated_at` write prevention;
- absence of operational compare tokens and run identifiers from rendered LLM
  prompts;
- self-cognition pause-probe behavior before source collection.

Required static checks:

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\reflection_cycle\affect_settling.py src\kazusa_ai_chatbot\reflection_cycle\worker.py src\kazusa_ai_chatbot\db\character.py src\kazusa_ai_chatbot\self_cognition\worker.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\config.py
rg -n "mood.*(ladder|lookup|score|severity|sentiment|level|clamp)|global_vibe.*(ladder|lookup|score|severity|sentiment|level|clamp)" src\kazusa_ai_chatbot tests
git diff --check
```

The `rg` check is a review aid, not the sole proof. The implementation review
must inspect affect-settling code for deterministic semantic mood/vibe
interpretation.

The static review must also confirm `self_cognition` does not import
`reflection_cycle` solely for the wake pause; the dependency should be injected
by the service or worker startup boundary.

Live LLM checks:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm tests\test_reflection_affect_settling_live_llm.py::test_live_affect_settling_angry_state_without_fresh_conflict -q -s
venv\Scripts\python.exe -m pytest -m live_llm tests\test_reflection_affect_settling_live_llm.py::test_live_affect_settling_preserves_fresh_sleep_conflict -q -s
```

Run live LLM checks one at a time and inspect the output before continuing.

## Acceptance Criteria

- Exactly one successful affect-settling run can be persisted per
  `settling_local_date` and prompt version.
- A successful accepted run writes only `mood`, `global_vibe`,
  `reflection_summary`, and `updated_at` in `character_state`.
- Free-form LLM affect output is persisted exactly after structural trimming
  of surrounding whitespace only.
- No deterministic code semantically classifies, maps, scores, clamps, or
  rewrites mood/vibe/reflection text.
- Stale `character_state.updated_at` prevents character-state writes.
- Stale `character_state.updated_at` persists a non-retryable skipped run, so
  the same sleep cycle is not retried against newer live mood.
- Missing daily or sleep-window reflection evidence becomes prompt-visible
  limitation text and validation warning, not deterministic semantic behavior.
- Proposal and reviewer prompts never contain raw operational metadata such as
  `character_state.updated_at`, run ids, scheduler ids, compare tokens, or
  lease fields.
- Self-cognition is paused only during the narrow due wake window and resumes
  after the configured grace window even if settling failed.
- Self-cognition wake pause is an operational worker pause; it does not filter
  or mutate source cases.
- `self_cognition` does not import `reflection_cycle` internals for the wake
  pause.
- Cache invalidation and runtime-state refresh happen after successful writes.
- Operator/event logs remain sanitized and contain no raw prompt, raw message,
  raw output, raw channel id, user id, or full character-state document.
- There is no `AFFECT_SETTLING_ENABLED` public config surface; empty
  `CHARACTER_SLEEP_LOCAL_PERIOD` disables future affect settling through the
  shared sleep-period contract.

## Execution Evidence

Record implementation evidence here during execution. This draft does not
authorize implementation.

| Stage | Evidence |
|---|---|
| Status change approved | User explicitly requested execution without subagents on 2026-07-02; plan moved to `in_progress`. Missing `development_plans/references/*` noted. |
| Config/timing tests | `venv\Scripts\python.exe -m pytest tests\test_config.py ... -q` passed in the full deterministic batch; final follow-up keeps only `AFFECT_SETTLING_WAKE_PREP_MINUTES` in config, proves removed affect env names do not create config attributes, and validates invalid due-after-wake-defer timing at affect-module import. |
| DB stale-write tests | `tests\test_db.py` passed in the full deterministic batch; includes `compare_and_upsert_character_state` matched and stale-token cases. |
| Prompt contract tests | `tests\test_reflection_affect_settling.py` passed; prompt hygiene asserts no `updated_at`, state token, run id, or source run id in rendered proposal prompt. |
| Worker/facade/CLI tests | `tests\test_reflection_cycle_stage1c_worker.py` and `tests\test_reflection_cycle_stage1c_service.py` passed in targeted broader runs; CLI command added to `src\scripts\run_reflection_cycle.py`. |
| Self-cognition wake-pause tests | `tests\test_self_cognition_integration.py` passed in the full deterministic batch; pause returns before source collection and is injected through service. |
| Live LLM checks | Passed one at a time: `test_live_affect_settling_angry_state_without_fresh_conflict` and `test_live_affect_settling_preserves_fresh_sleep_conflict`. Trace artifacts written under `test_artifacts\llm_traces\reflection_affect_settling_live_llm__*.json`; inspected parsed proposals and reviews. |
| Static checks | `py_compile` passed for touched Python modules and live test; `git diff --check` passed with line-ending warnings only; no affect-settling lookup/score/clamp hits in the plan `rg` review aid. |
| Independent code review | Parent-only review completed. Findings: worker initially lacked post-grace catch-up, and affect result initially left `processed_count=0`; both were fixed and retested. No remaining blocking findings. |
| Final close review | Parent-only final review before archiving found stale plan text for the removed `AFFECT_SETTLING_ENABLED` surface and one refresh-callback failure mode. Plan text was corrected; `run_daily_affect_settling` now records a recovered runtime-error event and still returns the successful result after a committed character-state write if the runtime refresh callback fails. Regression coverage added in `tests/test_reflection_affect_settling.py`. |
| Final sign-off | Focused deterministic verification for this plan passed: 219 passed, 13 deselected. Live LLM checks passed one at a time: 2 passed, with traces under `test_artifacts\llm_traces\reflection_affect_settling_live_llm__*.json`. Broad default `pytest -q` was also run and exposed two unrelated baseline failures outside this plan's change surface: `ACTION_ROUTER_PROMPT` fingerprint drift and stale reflection readonly projection allowlist for existing `channel_name`. |

## Progress Checklist

- [x] Plan approved for implementation.
- [x] Config and timing contract implemented.
- [x] Structural character-state compare-and-set implemented.
- [x] Affect-settling proposal and reviewer prompts implemented.
- [x] Worker, facade, and CLI integration implemented.
- [x] Cache invalidation and runtime refresh implemented.
- [x] Self-cognition wake pause implemented.
- [x] Documentation updated.
- [x] Deterministic verification complete.
- [x] Live LLM verification complete.
- [x] Independent code review complete.
- [x] Plan status updated with execution evidence.
