# reflection phase scheduled group review plan

## Summary

- Goal: Replace the reflection-attached group self-cognition global burst batch
  with a deterministic phase scheduler over current monitor-eligible channels,
  while enforcing one group per phase slot and once-only group-window review.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `database-data-pull` for optional live
  diagnostics, and `cjk-safety` if Python prompt/source-packet text with CJK is
  edited.
- Overall cutover strategy: bigbang for reflection worker scheduling and
  group-review source selection; compatible for historical reflection run
  documents and historical self-cognition action attempts; migration for the
  new reviewed-window ledger indexes only.
- Highest-risk areas: silently partial daily reflection input, reintroducing
  bursts through overflow batching, repeated silent reviews, hidden magic
  thresholds, starving channels when monitor eligibility changes, and leaking
  raw reflection output into self-cognition.
- Acceptance criteria: implementation is complete only when phase scheduling is
  deterministic and test-covered, current monitor-eligible channels are phased
  without random delay, each slot runs at most one group, each group phase
  reviews at most one group activity window, reviewed or coalesced windows are
  durably suppressed, daily reflection refuses silently partial input caused by
  missing expected hourly docs, docs and ops config reflect the new contract,
  deterministic verification passes, and independent code review approves the
  result.

## Context

The current reflection worker runs on a 15-minute cadence controlled by
`REFLECTION_WORKER_INTERVAL_SECONDS`. Each tick runs up to
`REFLECTION_HOURLY_SLOTS_PER_TICK` hourly reflection slots and then runs the
group self-cognition sidecar with up to `SELF_COGNITION_MAX_CASES_PER_TICK`
cases. Group review collection currently gathers all eligible recent group
activity windows, sorts them newest-first globally, and processes the first
batch. This produces both log bursts and possible visible-message bursts.

The current monitor rule is intentionally preserved: a channel is
monitor-eligible when the character has an assistant message in that channel
inside the existing monitor window. The scheduler must use the current
monitor-eligible channel set, not a new adapter watched-channel roster.

The current repeat-suppression collection,
`self_cognition_action_attempts`, suppresses repeated visible send attempts
for the same action identity. It does not suppress repeated silent/audit-only
review of the same group activity window because silent routes do not create
action-attempt rows. This plan adds a small reviewed-window ledger for group
review source consumption. Event logging remains observability only and must
not become production control state.

The current daily reflection path summarizes existing terminal hourly docs for
the previous character-local day. If a phased scheduler delays or skips
expected hourly docs, daily synthesis can become silently partial unless the
worker checks expected hourly coverage before running daily-channel synthesis.

## Mandatory Skills

- `development-plan`: load before editing, reviewing, approving, executing, or
  signing off this plan.
- `local-llm-architecture`: load before changing background cognition,
  self-cognition source packets, prompt-facing fields, LLM budgets, RAG,
  cognition, dialog, or consolidation behavior.
- `py-style`: load before editing Python source.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `database-data-pull`: load before optional live MongoDB diagnostics; do not
  read `.env`.
- `cjk-safety`: load before editing Python files that contain CJK prompt or
  source-packet strings.

## Mandatory Rules

- Do not implement this draft plan until the user explicitly approves it.
- Do not read `.env`.
- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Keep the change self-contained around `reflection_cycle`,
  `self_cognition` source execution, DB helper/index support, config, ops
  status, docs, and focused tests. Do not refactor unrelated scheduler,
  adapter, RAG, cognition, dialog, consolidation, or memory modules.
- Preserve the live chat response path. Do not serialize normal `/chat` behind
  reflection or self-cognition.
- Preserve the current monitor-eligible channel rule. Do not introduce an
  adapter watched-channel registry in this plan.
- Recompute the phase plan from the current monitor-eligible channel snapshot
  at phase-period boundaries. Channels added or removed during a period take
  effect in a later period; immediate reshuffle is not required.
- Do not use random delays. Phase offsets must be deterministic and stable for
  a given period snapshot.
- Do not hard-code scheduler thresholds or control constants inside worker
  logic. Every threshold must be a named config value or named module constant
  with tests covering the default.
- A phase slot is a load boundary. The worker must not pack multiple groups
  into a slot to catch up with overflow.
- Each phase slot may run at most one monitor-eligible group for group
  self-cognition. A private channel slot runs no group self-cognition.
- Each selected group phase may review at most one group activity window.
  Older unreviewed windows for that group are coalesced or skipped rather than
  caught up visibly.
- Raw hourly and daily reflection output must not feed self-cognition. The
  group sidecar may share only the monitored activity projection already
  approved for group review.
- Deterministic code owns channel selection, phase math, scope fetching,
  closed-hour targeting, reviewed-window suppression, coalescing, stale-window
  policy, delivery target binding, limits, and persistence.
- LLM stages own only the semantic decision for an already-selected source
  case: silence, audit, progress maintenance, or visible speech.
- Event logs are append-only observability. They must not be used as the
  reviewed-window ledger or as any production control state.
- The new reviewed-window ledger must be tolerant of future schema additions
  and must not require changes to historical self-cognition action-attempt
  rows.
- If implementation touches prompt-facing CJK source text, run prompt-render
  checks and apply `cjk-safety`.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.

## Must Do

- Add a pure deterministic phase-scheduler helper under `reflection_cycle`.
- Use `REFLECTION_WORKER_INTERVAL_SECONDS` as the phase period.
- Add `REFLECTION_PHASE_MIN_SLOT_SPACING_SECONDS` with default `60` and fail
  fast if it cannot produce at least one slot inside the phase period.
- Preserve one group per phase slot as the named module invariant
  `REFLECTION_PHASE_GROUPS_PER_SLOT = 1`. Do not add overflow group batching
  or an environment override for this invariant in this plan.
- Build phase plans from current monitor-eligible channel snapshots.
- Rotate overflow channels fairly across future periods when the number of
  eligible channels exceeds the maximum phase slots.
- Refactor reflection hourly execution so a phase slot can process due work for
  the selected channel scope instead of the current global newest-first batch.
- Refactor the group sidecar so it builds group windows only for the selected
  group scope and passes at most one case to the normal self-cognition worker.
- Add a durable reviewed-window ledger for group activity windows. Terminal
  ledger rows must suppress future review of the same `source_id`, including
  silent/audit-only outcomes.
- Coalesce old unreviewed windows for a selected group so backlog cannot create
  a later burst.
- Add a daily-readiness guard so daily-channel synthesis does not run when
  expected previous-day hourly docs for current monitor-eligible channels are
  missing.
- Keep global reflection promotion, interaction style update, and global
  character growth behavior semantically unchanged.
- Update docs and tests that currently describe or assert global newest-first
  group-review batching.
- Add operator-visible config/status coverage for the effective phase period,
  minimum slot spacing, and one-group-per-slot invariant.

## Deferred

- Do not implement a watched-channel adapter registry.
- Do not implement multi-brain or multi-process scheduler coordination.
- Do not use `scheduled_events` as the phase scheduler control plane.
- Do not use event logs as production control state.
- Do not add random jitter.
- Do not add multi-group slot packing.
- Do not add delivery retries or private-channel fallback.
- Do not rewrite self-cognition prompts, cognition routing, dialog rendering,
  RAG, consolidation, or memory promotion.
- Do not migrate historical reflection run documents.
- Do not migrate historical self-cognition action-attempt rows.
- Do not broaden group review eligibility beyond current monitor-eligible
  channels.
- Do not make daily reflection summarize silently partial input caused by
  missing expected hourly docs.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Reflection worker cadence | bigbang | Replace the single global batch tick with the phase-slot loop. Do not preserve the old global newest-first group-review batch path in production. |
| Monitor eligibility | compatible | Preserve the existing assistant-message-in-monitor-window rule and existing cap semantics unless this plan explicitly changes a consumer's ordering. |
| Hourly reflection run documents | compatible | Keep existing `character_reflection_runs` run ids, kinds, statuses, and hourly/daily document shapes. |
| Group self-cognition source collection | bigbang | Production group review must run from the selected phase group only, one window per group phase. |
| Reviewed-window ledger | migration | Create the new collection/indexes through DB bootstrap. No historical backfill is required. |
| Self-cognition action attempts | compatible | Keep existing idempotency behavior and historical rows readable. The new ledger supplements silent-window suppression and does not replace action attempts. |
| Event logging | compatible | Keep existing event families. Use existing worker event count/status fields only. Do not add optional worker metrics in this plan. |
| Tests | bigbang | Update tests that assert global newest-first group-review batching to assert phase-scoped behavior. Preserve helper-level tests only where they remain non-production utilities. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- Bigbang areas must delete or rewrite legacy production references instead of
  preserving old behavior behind fallbacks.
- Migration areas must follow the exact collection/index creation path in this
  plan.
- Compatible areas must preserve only the compatibility surfaces listed in
  this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

At the start of each phase period, the reflection worker builds a deterministic
snapshot of current monitor-eligible channel scopes using the existing
reflection selector. The scheduler sorts scopes by stable scope identity and
selects up to the maximum number of phase slots for that period. If there are
more eligible scopes than slots, the scheduler rotates which scopes are served
across later periods.

For each selected phase slot, the worker runs this bounded sequence:

1. Fetch fresh bounded messages for the selected channel scope.
2. Run at most one due closed-hour reflection slot for that selected scope.
3. If the selected scope is a group and self-cognition is not in its sleep
   period, build group activity windows for that selected group.
4. Exclude terminally reviewed ledger windows.
5. Select the newest remaining non-empty group window only.
6. Mark older remaining windows for that group as coalesced/skipped through the
   reviewed-window ledger.
7. Pass the selected case into the normal self-cognition worker with
   `max_cases` equal to the named one-window-per-group-phase invariant.
8. Record a terminal reviewed-window ledger row after the worker returns for
   that case.

Daily-channel synthesis, interaction-style update, global reflection promotion,
and global character growth remain period-level maintenance, not per-slot
group catch-up. They must not rerun once per phase slot. The worker should run
them through a period-level gate equivalent to the current local-time gates,
plus the new daily-readiness guard.

## Design Decisions

1. Keep phase scheduling in `reflection_cycle`.

   The reflection worker already owns the 15-minute sidecar and hourly/daily
   reflection cadence. The self-cognition worker remains a bounded case
   executor.

2. Use current monitor-eligible channels.

   The scheduler uses the existing assistant-message monitor rule. It does not
   introduce a new adapter-side watched-channel contract.

3. Rebalance only at period boundaries.

   This avoids worker churn and satisfies the requirement that channel
   additions/removals do not need immediate reshuffle.

4. Rotate overflow, never batch overflow.

   If eligible channels exceed the maximum phase-slot count, the scheduler
   carries the skipped channels into later periods by deterministic rotation.
   It does not place multiple groups in one slot.

5. Use a reviewed-window ledger for once-only semantics.

   Action-attempt idempotency is still required for visible send suppression,
   but it is not enough for silent/audit-only review suppression.

6. Coalesce old windows for load control.

   A selected group phase reviews the latest eligible window only. Older
   windows in the same group backlog are terminally recorded as coalesced
   skips, preventing future catch-up bursts.

7. Preserve reflection output boundary.

   Self-cognition receives activity-window evidence only. Hourly/daily
   reflection outputs remain audit/promotion inputs and normal cognition can
   see them only through existing promoted, gated context.

## Contracts And Data Shapes

Add a pure scheduler contract in `reflection_cycle`, using named dataclasses or
typed dictionaries:

```python
ReflectionPhaseSlot = {
    "period_start_utc": str,
    "slot_index": int,
    "offset_seconds": int,
    "scope_ref": str,
    "platform": str,
    "platform_channel_id": str,
    "channel_type": str,
}
```

The scheduler contract must guarantee:

- every returned slot has one channel scope;
- at most one group scope can appear in a slot;
- slot spacing is never below the configured minimum;
- overflow channels are omitted for the current period and become eligible for
  earlier selection in later periods;
- output is deterministic for the same period start, config, and eligible
  scope list.

Add a reviewed-window ledger collection owned through DB helpers, with a
document shape equivalent to:

```python
SelfCognitionGroupReviewWindowDoc = {
    "source_id": str,
    "case_id": str,
    "scope_ref": str,
    "platform": str,
    "platform_channel_id": str,
    "channel_type": "group",
    "window_start": str,
    "window_end": str,
    "status": (
        "reviewed"
        | "coalesced_skipped"
        | "stale_skipped"
        | "target_binding_failed"
        | "review_failed"
    ),
    "reviewed_at": str,
    "selected_route": str,
    "dispatch_status": str,
    "skip_reason": str,
}
```

Required indexes:

- unique `source_id`;
- `scope_ref`, `status`, `window_start`;
- `reviewed_at`.

Terminal statuses suppress future review of the same `source_id`. A process
crash before the terminal ledger write may allow one retry after restart. Strict
multi-process exactly-once is out of scope for this plan.

## LLM Call And Context Budget

- Do not add new LLM calls.
- Do not change self-cognition prompt-facing source-packet text unless a test
  proves the existing packet cannot represent the phased source case.
- Group self-cognition remains at most one cognition case per selected group
  phase.
- Coalesced and stale skipped windows must not call LLMs.
- Hourly reflection still uses the existing hourly reflection LLM path, but the
  scheduler limits which channel scope can run in a phase slot.
- Daily, style, promotion, and global growth LLM paths remain governed by their
  existing gates and must not run per group phase slot.

## Change Surface

Expected production files:

- Modify `src/kazusa_ai_chatbot/config.py` for named phase scheduling config
  and fail-fast validation.
- Modify `src/kazusa_ai_chatbot/reflection_cycle/models.py` for phase and
  reviewed-window constants/types only if shared across modules.
- Create `src/kazusa_ai_chatbot/reflection_cycle/phase_scheduler.py` for pure
  phase planning.
- Modify `src/kazusa_ai_chatbot/reflection_cycle/selector.py` to expose scoped
  channel-scope fetching without changing the monitor eligibility rule.
- Modify `src/kazusa_ai_chatbot/reflection_cycle/worker.py` to run phase slots,
  scoped hourly reflection, phase-scoped group review, and period-level
  maintenance gates.
- Modify `src/kazusa_ai_chatbot/reflection_cycle/repository.py` only for
  expected hourly run-id helpers needed by daily readiness.
- Modify `src/kazusa_ai_chatbot/db/self_cognition.py` for reviewed-window
  ledger helpers.
- Modify `src/kazusa_ai_chatbot/db/bootstrap.py` for reviewed-window ledger
  indexes.
- Modify `src/kazusa_ai_chatbot/db/schemas.py` for reviewed-window ledger
  typed document shape if this repo's DB schema module remains the canonical
  typed doc location.
- Modify `src/kazusa_ai_chatbot/service.py` to expose phase config in
  `/ops/runtime-status`.
- Update `src/kazusa_ai_chatbot/reflection_cycle/README.md`,
  `src/kazusa_ai_chatbot/self_cognition/README.md`,
  `src/kazusa_ai_chatbot/db/README.md`, and `docs/HOWTO.md`.

Expected tests:

- Create `tests/test_reflection_phase_scheduler.py`.
- Modify `tests/test_reflection_cycle_stage1c_worker.py`.
- Modify `tests/test_self_cognition_group_review_source.py` only if the
  production collector contract changes; otherwise preserve helper-level tests
  and add worker-level phase tests.
- Add or modify DB helper/index coverage in `tests/test_db.py`.
- Modify `tests/test_config.py` for named phase config defaults and invalid
  values.
- Modify `tests/test_service_ops_status.py` if ops config payload changes.
- Modify docs architecture tests if they assert the old global batch contract.

## Overdesign Guardrail

- Do not create a general-purpose scheduling framework.
- Do not add a queue broker, external scheduler, cron layer, or
  `scheduled_events` integration.
- Do not introduce multi-process leases or distributed locks.
- Do not build a watched-channel registry.
- Do not create a second self-cognition worker.
- Do not add prompt-level load-balancing instructions.
- Do not make overflow batching configurable in this plan.
- Do not add compatibility switches for old global group-review batching.

## Agent Autonomy Boundaries

- Execution agents may choose internal helper names only when the plan does not
  specify a public symbol name and the helper stays within the approved change
  surface.
- Execution agents may split pure helper functions for readability inside
  `reflection_cycle` if tests prove the same contract.
- Execution agents must request user approval before adding a feature flag,
  changing monitor eligibility, changing one-group-per-slot behavior, adding
  multi-group batching, introducing multi-process coordination, changing LLM
  prompts, changing delivery behavior, or broadening the change surface.
- Execution agents must not silently relax daily-readiness checks to make tests
  pass.

## Implementation Order

1. Parent adds focused phase-scheduler tests in
   `tests/test_reflection_phase_scheduler.py`.
   - Prove deterministic offsets when eligible scopes fit inside the period.
   - Prove overflow rotation when eligible scopes exceed the maximum slots.
   - Prove no slot contains more than one group.
   - Prove channel additions/removals affect later period snapshots without
     random reshuffle.
   - Run the test before implementation and record the missing-module failure.

2. Parent adds config tests in `tests/test_config.py`.
   - Prove phase minimum spacing has a named default.
   - Prove invalid spacing fails fast when it cannot produce one slot inside
     the phase period.
   - Run the tests before implementation and record the expected failure.

3. Parent starts the production-code subagent with ownership limited to
   `config.py`, `reflection_cycle/phase_scheduler.py`, and any minimal shared
   reflection-cycle model types needed for the focused tests.

4. Production-code subagent implements the pure phase scheduler and config.
   - Do not touch worker integration in this step.
   - Report changed files, commands run, blockers, and residual risks.

5. Parent runs the focused scheduler and config tests and records the result.

6. Parent adds DB ledger tests in `tests/test_db.py`.
   - Prove ledger upsert/read helpers use `source_id` as the unique identity.
   - Prove DB bootstrap creates the reviewed-window indexes.
   - Run the tests before implementation and record the expected failure.

7. Production-code subagent implements DB ledger helpers, typed schema, and
   bootstrap indexes.

8. Parent runs DB ledger tests and records the result.

9. Parent adds worker integration tests in
   `tests/test_reflection_cycle_stage1c_worker.py`.
   - Prove one phase slot calls hourly reflection for the selected scope only.
   - Prove group review sees only the selected group scope.
   - Prove a private selected scope does not run group self-cognition.
   - Prove reviewed ledger rows suppress repeated silent/audit-only windows.
   - Prove older windows are coalesced/skipped instead of caught up.
   - Prove daily maintenance does not run once per phase slot.
   - Prove daily-channel synthesis is deferred when expected previous-day
     hourly docs are missing.
   - Run the tests before implementation and record the expected failure or
     old-behavior baseline.

10. Production-code subagent implements scoped selector helpers, worker phase
    integration, group reviewed-window suppression, coalescing, and daily
    readiness.

11. Parent runs worker integration tests and records the result.

12. Parent updates ops/config/docs tests.
    - Update `/ops/runtime-status` config expectations if phase config is
      exposed there.
    - Update architecture docs tests that describe group review cadence.
    - Run the affected tests and record the result.

13. Parent updates docs and HOWTO.
    - Document phase scheduler defaults through named settings.
    - Document one group per slot, once-only window review, overflow rotation,
      and daily-readiness behavior.

14. Parent runs full focused verification.
    - Run py-compile for changed source and tests.
    - Run all focused pytest commands listed in `Verification`.
    - Record output in `Execution Evidence`.

15. Parent starts the independent code-review subagent.
    - Provide the approved plan, full diff, verification evidence, and review
      scope.
    - The review subagent reports findings only.

16. Parent remediates approved review findings, reruns affected verification,
    records final evidence, and updates the plan lifecycle only after user
    approval or completion criteria are satisfied.

## Execution Model

Execution is parent-led and requires native subagent capability unless the user
explicitly approves fallback execution.

The parent owns:

- test-contract-first setup;
- integration tests;
- verification commands;
- execution evidence;
- review feedback remediation;
- plan status updates;
- final sign-off.

The production-code subagent owns planned production code changes only. It must
not edit plan lifecycle status, run unrelated refactors, change prompt
contracts, or expand the change surface.

The independent code-review subagent owns review only. It must not implement
fixes.

If native subagent capability is unavailable at execution time, stop before
execution and report the blocker unless the user explicitly asks for fallback
execution.

## Progress Checklist

- [ ] Stage 1 - phase scheduler and config contract established
  - Covers: implementation steps 1-5.
  - Files: `tests/test_reflection_phase_scheduler.py`,
    `tests/test_config.py`, `src/kazusa_ai_chatbot/config.py`,
    `src/kazusa_ai_chatbot/reflection_cycle/phase_scheduler.py`.
  - Verify: focused scheduler and config tests pass.
  - Evidence: record failing-test baseline, changed files, and passing output
    in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 2 - reviewed-window ledger contract complete
  - Covers: implementation steps 6-8.
  - Files: `tests/test_db.py`, `src/kazusa_ai_chatbot/db/self_cognition.py`,
    `src/kazusa_ai_chatbot/db/bootstrap.py`,
    `src/kazusa_ai_chatbot/db/schemas.py`.
  - Verify: DB helper/index tests pass.
  - Evidence: record failing-test baseline, changed files, and passing output
    in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 3 - reflection worker phase integration complete
  - Covers: implementation steps 9-11.
  - Files: `tests/test_reflection_cycle_stage1c_worker.py`,
    `src/kazusa_ai_chatbot/reflection_cycle/selector.py`,
    `src/kazusa_ai_chatbot/reflection_cycle/worker.py`,
    `src/kazusa_ai_chatbot/reflection_cycle/repository.py`.
  - Verify: reflection worker focused tests pass.
  - Evidence: record old-behavior baseline, changed files, and passing output
    in `Execution Evidence`.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 4 - docs and ops surfaces updated
  - Covers: implementation steps 12-13.
  - Files: `tests/test_service_ops_status.py`, docs architecture tests if
    affected, `src/kazusa_ai_chatbot/service.py`,
    `src/kazusa_ai_chatbot/reflection_cycle/README.md`,
    `src/kazusa_ai_chatbot/self_cognition/README.md`,
    `src/kazusa_ai_chatbot/db/README.md`, `docs/HOWTO.md`.
  - Verify: ops/docs focused tests pass.
  - Evidence: record changed files and passing output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 5 - full focused verification complete
  - Covers: implementation step 14.
  - Verify: all commands in `Verification` pass or have recorded blocker
    evidence.
  - Evidence: record command output and residual risk.
  - Handoff: next agent starts independent code review.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

- [ ] Stage 6 - independent code review complete
  - Covers: implementation steps 15-16.
  - Verify: review findings are addressed or explicitly accepted as residual
    risk, and affected verification commands are rerun.
  - Evidence: record reviewer findings, fixes, rerun commands, and approval
    status.
  - Handoff: plan can move toward completion only after user acceptance.
  - Sign-off: `<agent/date>` after review approval and evidence are recorded.

## Verification

Syntax checks:

```powershell
venv\Scripts\python -m py_compile `
  src\kazusa_ai_chatbot\config.py `
  src\kazusa_ai_chatbot\reflection_cycle\phase_scheduler.py `
  src\kazusa_ai_chatbot\reflection_cycle\selector.py `
  src\kazusa_ai_chatbot\reflection_cycle\worker.py `
  src\kazusa_ai_chatbot\reflection_cycle\repository.py `
  src\kazusa_ai_chatbot\db\self_cognition.py `
  src\kazusa_ai_chatbot\db\bootstrap.py `
  src\kazusa_ai_chatbot\db\schemas.py `
  src\kazusa_ai_chatbot\service.py
```

Focused deterministic tests:

```powershell
venv\Scripts\python -m pytest `
  tests\test_reflection_phase_scheduler.py `
  tests\test_reflection_cycle_stage1c_worker.py `
  tests\test_self_cognition_group_review_source.py `
  tests\test_reflection_cycle_activity_windows.py `
  tests\test_db.py `
  tests\test_config.py `
  tests\test_service_ops_status.py `
  -q
```

Broader regression tests:

```powershell
venv\Scripts\python -m pytest `
  tests\test_reflection_cycle_stage1c_repository.py `
  tests\test_reflection_cycle_stage1c_service.py `
  tests\test_reflection_event_logging.py `
  tests\test_self_cognition_tracking.py `
  tests\test_self_cognition_event_logging.py `
  tests\test_self_cognition_delivery_target.py `
  -q
```

Optional live diagnostic, only when the user asks for live DB evidence:

```powershell
venv\Scripts\python -m scripts.export_event_log --help
```

The optional live diagnostic command is a safety probe for script availability
only. Any real DB export must use the `database-data-pull` skill, must avoid
`.env` inspection, and must write sanitized artifacts under
`test_artifacts/diagnostics/`.

## Independent Plan Review

Before changing `Status` from `draft` to `approved`, run an independent plan
review focused on:

- whether the plan preserves current monitor eligibility;
- whether one group per slot is enforced without overflow batching;
- whether once-only reviewed-window suppression covers silent/audit-only
  routes;
- whether daily-readiness prevents silently partial daily input;
- whether the change surface is self-contained;
- whether verification covers scheduling, persistence, worker integration,
  docs, and ops status.

Record the review result in `Execution Evidence` before approval.

## Independent Code Review

Before final completion, run an independent code review against the full diff
and this approved plan.

Review scope:

- phase scheduler correctness and deterministic rotation;
- no random delay and no hard-coded scheduler thresholds;
- one group per slot and one group window per group phase;
- reviewed-window ledger correctness and idempotency;
- daily-readiness behavior;
- preservation of reflection output boundary;
- self-cognition delivery and action-attempt behavior unchanged except for
  once-only window source suppression;
- tests proving old burst behavior is gone;
- docs and ops status accuracy.

The review subagent must report findings only. The parent agent owns fixes and
must rerun affected verification before sign-off.

## Acceptance Criteria

- Phase scheduling is deterministic, named-config-driven, and unit-tested.
- The phase scheduler uses current monitor-eligible channels and does not
  introduce a watched-channel adapter registry.
- Channels added or removed during a phase period affect later period
  snapshots without immediate random reshuffle.
- A phase slot runs at most one channel scope and at most one group for group
  self-cognition.
- Overflow channels rotate across later periods and are not packed into
  existing slots.
- A selected group phase reviews at most one group activity window.
- Older unreviewed windows for the selected group are coalesced or skipped and
  cannot create later catch-up bursts.
- Reviewed, coalesced, stale, target-binding-failed, and failed terminal
  window rows suppress future review of the same `source_id`.
- Existing self-cognition action-attempt idempotency remains intact for visible
  send suppression.
- Daily-channel synthesis does not run with silently missing expected hourly
  docs for current monitor-eligible channels.
- Raw reflection output does not enter self-cognition.
- No new LLM calls, prompt changes, adapter fallback, delivery retry, or
  scheduled-events control path are introduced.
- Focused and regression verification commands pass.
- Independent code review approves the result or all findings are resolved and
  reverified.

## Risks

- Low-traffic channels may be skipped for longer than one phase period when
  eligible channel count exceeds available slots. This is accepted to protect
  load and avoid group batching.
- A process crash before the reviewed-window ledger terminal write can allow
  one retry after restart. Strict multi-process exactly-once is out of scope.
- Current monitor eligibility is recency-biased and capped by the existing
  selector. This plan preserves that rule by user decision, so fairness is
  fairness within the current monitor-eligible set, not across all adapter
  channels.
- Daily-readiness checks can defer daily synthesis if hourly backlog exists.
  This is preferable to silently partial daily reflection and must be visible
  through worker status/evidence.

## Execution Evidence

No execution evidence yet. This plan is a draft and does not authorize
production-code changes.
