# conversation episode state lane lifecycle plan

## Summary

- Goal: repair malformed active `conversation_episode_state` rows and harden
  the episode-state lane so short-horizon flow state expires, closes, records,
  and reads according to one active-memory contract.
- Plan class: high_risk_migration.
- Status: completed.
- Mandatory skills: `development-plan`, `database-data-pull`, `py-style`,
  `test-style-and-execution`.
- Overall cutover strategy: migration for existing data; bigbang for active
  read and active recall filtering; compatible for historical row retention.
- Highest-risk areas: stale progress reaching cognition, cleanup locking future
  writes behind a high `turn_count`, accidental TTL deletion of diagnostic
  state, and group-channel progress being treated as broader durable memory.
- Acceptance criteria: expired or malformed active rows are not used by active
  readers; approved cleanup is idempotent and non-deleting; new writes can
  start a fresh active episode after a closed or expired row; verification
  proves active-read, cache, RAG recall, migration, and bootstrap robustness.

New-write hardening supersession: all lane-specific new-write hardening,
source-generation proof, and fixing-strategy instructions in this plan are
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.
Execute this lane plan for cleanup, audit, dry-run/apply, data migration, and
post-cleanup verification only. If another section still names new-write
validators, prompts, or tests, treat that text as historical context rather
than execution scope.

## Context

The 2026-07-02 audit found 174 `conversation_episode_state` rows:

- 169 active;
- 5 closed;
- 159 active rows expired as of 2026-07-02.

This lane is short-horizon conversation-flow state. It is not durable identity
memory, not user memory, and not RAG evidence by itself. The project basis for
this bugfix is explicit: an episode-state document can be valid history but bad
active memory when it is expired, missing expiry, malformed, or otherwise still
marked active in the active lane.

Relevant current implementation observations:

- `src/kazusa_ai_chatbot/db/conversation_progress.py` currently loads one row
  by `platform`, `platform_channel_id`, and `global_user_id` without filtering
  `status` or `expires_at`.
- `src/kazusa_ai_chatbot/conversation_progress/runtime.py` passes the selected
  loaded row both to prompt projection and later to the post-turn recorder as
  `prior_episode_state`.
- `src/kazusa_ai_chatbot/conversation_progress/cache.py` selects a process-local
  cached row by cache age and higher `turn_count`; it does not validate the
  cached row's lane `status` or storage `expires_at`.
- `src/kazusa_ai_chatbot/conversation_progress/policy.py` writes canonical UTC
  string expiry via `expires_at_for(...)`, using a 48-hour lane TTL.
- `src/kazusa_ai_chatbot/db/bootstrap.py` creates a Mongo TTL index on
  `conversation_episode_state.expires_at`, but the stored field is an ISO string.
  Mongo TTL deletion is not the lane correctness boundary and must not be made
  effective by converting this field to a BSON date in this plan.
- `src/kazusa_ai_chatbot/rag/recall/collectors/progress.py` can treat missing
  or unparseable progress expiry as active recall evidence. That behavior can
  reactivate malformed state even if the primary loader is corrected.
- The collection has a unique scope index on `(platform, platform_channel_id,
  global_user_id)`. This means an expired row is the one operational row for
  that scope. If cleanup closes a row with `turn_count=20` and a later new
  episode builds `turn_count=1`, the current guarded upsert would fail unless
  the write guard is changed to allow replacement of non-active, expired, or
  missing-expiry rows.

Known consumers and boundaries:

- Brain service owns calling `load_progress_context(...)` only after relevance
  approves a response.
- Conversation progress owns active prompt projection and post-turn recording.
- Database helpers own query filters, indexes, and lifecycle mutation.
- RAG recall may use already-loaded current progress as active-episode evidence
  only when the same active-memory contract is satisfied.
- Cognition and dialog consume prompt-facing current progress; they do not own
  expiry, cleanup, storage repair, or lifecycle validation.

Adjacent scope intentionally left out:

- No conversation-progress redesign.
- No new group-thread identifier.
- No durable archive collection for all old episode-state content.
- No periodic scheduler or background sweeper in this bugfix.

## Lane Analysis Requirements

### Issue Description Based On Deep Analysis

`conversation_episode_state` is operational working memory for the current
local episode. Active rows that are expired, missing expiry, or invalidly
expired can still be selected by the current DB load path and then projected
into cognition as if they were current. Because the lane guides L3 content
planning, open-loop handling, repeat avoidance, and active RAG recall, stale
active rows can nudge Kazusa to reopen old topics, repeat stale obligations,
misread group-channel state as current, or answer from old progress rather than
from the live turn.

The defect is not that every expired document is false. The defect is that
expired or malformed rows remain eligible active memory. The fix must separate
historical validity from active-lane eligibility.

### Plan To Remove Malformed Data

1. Export the current `conversation_episode_state` collection before mutation.
2. Run a read-only lane audit that classifies every active row by scope,
   status, expiry shape, expiry time, and recommended action.
3. Generate a reviewed dry-run repair report that plans only non-deleting
   actions: close expired active rows, set a derived expiry only when
   `updated_at` is valid and still within the 48-hour window, and classify
   invalid or under-evidenced rows for manual review.
4. Apply only from a reviewed dry-run report. Apply mode must reread each row
   and match the dry-run row identity, scope, `status`, `expires_at`,
   `updated_at`, and `turn_count` before writing.
5. Preserve auditability by default. Do not delete episode-state rows in this
   plan. Because the existing collection is one operational row per scope, the
   pre-apply export and apply report are the durable audit evidence for row
   content that future normal writes may overwrite.

### RCA Of The Failure Mode

The root cause is a lane-contract gap across four places:

- Writes generate `expires_at`, but active reads do not require
  `status == "active"` and a valid non-expired storage-UTC expiry.
- Cleanup was not implemented as required state correctness; expired active
  rows were left to a TTL expectation or manual hygiene.
- The TTL index is not a safe correctness mechanism. The lane stores ISO
  strings, TTL deletion is asynchronous even with BSON dates, and making TTL
  effective would delete diagnostic state instead of closing it.
- The write guard only compares `turn_count`. Closing or ignoring an old row
  can prevent a fresh episode from being written unless replacement of
  non-active, expired, and missing-expiry rows is explicitly allowed.

### Plan To Harden The Corrupted Data Source

New-write hardening for this lane is superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

This lane plan keeps its cleanup, audit, dry-run/apply, and post-cleanup
verification scope. Do not implement lane-specific new-write prompts,
validators, routing rules, or semantic filters from the superseded draft text;
implement new-write memory-pollution prevention through `consolidator_lane_router_memory_pollution_bigbang_plan.md` only.

### Plan To Prove Robustness Of The Data Source Generation

Robustness of new data-source generation for this lane is superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

This lane plan proves cleanup robustness only: baseline export, dry-run report,
manual review, approved apply, post-apply audit, retrieval/cache smoke, and
rollback evidence where the cleanup section requires it. The cross-lane
new-write robustness gate is the router plan's deterministic tests plus its 26
one-at-a-time live-LLM memory-write use cases.

## Failure Modes And Root Cause Analysis

| Failure mode | Current or plausible cause | Live impact | Required fix |
| --- | --- | --- | --- |
| Expired active rows | DB load ignores `expires_at`; cleanup absent; TTL expectation does not enforce active semantics | Old progress becomes active prompt state | Active-read filter, cache filter, cleanup close action |
| Missing expiry on active rows | Legacy or partial writes, manual rows, failed migrations, schema drift | Row can remain active forever or block new writes | Audit classification; fail closed for reads; derive expiry only from valid `updated_at`; otherwise manual review |
| Active-read status gap | Load by scope only | Closed or suspended rows can be projected as current | Require `status == "active"` before projection or recall |
| Cleanup/sweeper absence | No operator script and no required lifecycle maintenance gate | Expired rows accumulate and correctness relies on readers | Add read-only audit and report-driven apply cleanup; no periodic sweeper in this plan |
| TTL/delete risk | Bootstrap defines a TTL index on a string field; converting to BSON date would make deletion possible | Useful diagnostic rows could disappear without reviewed close evidence | Do not make TTL effective; replace correctness with explicit non-deleting lifecycle repair |
| Timezone and clock skew | Active checks could compare local time, unnormalized offsets, `Z`, `+00:00`, or malformed strings inconsistently | Future or expired rows classified incorrectly | Parse with `parse_storage_utc_datetime`; require storage UTC; use the turn's `storage_timestamp_utc` consistently |
| Cache stale path | Cache TTL is separate from document expiry and does not inspect document status | A cached higher-turn document can bypass fixed DB filtering | Pass current storage time into cache selection and reject non-active or expired cached docs |
| Group-thread scope | Scope is per platform, channel, and user, not per group subthread | In busy groups, stale state from the same user/channel can feel like a wrong thread | Keep scope unchanged, but enforce expiry and sharp-transition handling; no new thread id in this bugfix |
| Stale cognition impact | Conversation progress reaches L3 content planning and RAG recall as current-episode evidence | Reopened old tasks, over-avoidance, wrong current answer basis, or stale user emotional state | Exclude stale progress before cognition and before active recall |
| Write lock after close | Unique scope index plus turn-count-only guarded upsert rejects a fresh `turn_count=1` row after a closed high-turn row | Cleanup fixes reads but prevents new episode state from being recorded | Allow guarded replacement of non-active, expired, and missing-expiry rows |

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `database-data-pull`: load before exporting or inspecting live episode-state
  rows.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before changing or running tests.

## Mandatory Rules

- Do not execute this plan while status is `draft`.
- Do not apply live cleanup without reviewed dry-run output and an explicit user
  command for apply mode.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual edits.
- Check `git status --short` before editing.
- Do not read `.env` manually.
- Do not delete historical or operational episode-state rows in this plan.
- Active readers must filter expired, missing-expiry, invalid-expiry, and
  non-active rows even before cleanup runs.
- Active recall must not treat missing or invalid progress expiry as active.
- Cleanup must be idempotent and report-driven. Apply mode must not mutate a
  row that changed since the reviewed dry-run report.
- Deterministic code owns lifecycle validation, cleanup, indexes, and write
  guards. LLM stages own only semantic episode judgment inside the recorder.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.

## Must Do

- Add an episode-state lane audit for expired active rows, missing expiry,
  invalid expiry, non-UTC expiry, active row counts, closed row counts, scope
  shape, and index state.
- Add idempotent dry-run/apply cleanup that closes expired active rows and
  preserves data by default.
- Harden active DB reads to require current active lifecycle state.
- Harden runtime cache selection with the same lifecycle rule.
- Harden RAG current-episode progress recall so missing or invalid expiry fails
  closed.
- Harden guarded writes so a fresh active episode can replace closed, expired,
  or missing-expiry rows without weakening protection for newer active rows.
- Add focused deterministic tests for active filtering, cache filtering,
  cleanup idempotency, missing-expiry reporting, guarded replacement, group
  scope isolation, and RAG recall fail-closed behavior.
- Add operational verification that post-cleanup audit reports zero expired
  active rows and zero unexpected mutation skips.

## Deferred

- Do not redesign conversation progress.
- Do not add a new group-thread scope field.
- Do not add a durable archive collection for episode-state history.
- Do not delete `conversation_episode_state` documents.
- Do not convert `expires_at` to BSON dates in this plan.
- Do not introduce TTL deletion as the correctness mechanism.
- Do not add a background sweeper, scheduler worker, or startup repair job.
- Do not change cognition or dialog semantics beyond excluding invalid active
  progress state.
- Do not migrate unrelated conversation history, user memory, shared memory,
  character state, interaction-style images, or reflection data.

## Cutover Policy

Overall strategy: migration for existing rows, bigbang for active-read and
active-recall filtering, compatible for historical row retention.

| Area | Policy | Instruction |
| --- | --- | --- |
| Active DB reads | bigbang | Replace scope-only load with active lifecycle validation. No fallback to expired, missing-expiry, invalid-expiry, closed, or suspended rows. |
| Runtime cache | bigbang | Cache may win only when the cached document also satisfies the active lifecycle contract. |
| RAG current-episode progress recall | bigbang | Missing or invalid expiry fails closed; do not preserve the current permissive behavior. |
| Guarded writes | bigbang | Preserve newer-active protection, but allow replacement of non-active, expired, and missing-expiry rows in the same scope. |
| Existing expired active rows | migration | Close through reviewed dry-run/apply repair. Do not delete. |
| Missing-expiry active rows | migration | Derive expiry only from valid `updated_at`; close if derived expiry is expired; otherwise set derived expiry. Manual-review rows remain inactive for reads until repaired. |
| Historical/diagnostic evidence | compatible | Preserve current collection rows where possible and require pre-apply export plus apply evidence because future normal writes may overwrite the one row for a scope. |
| TTL index | migration | Do not make TTL deletion effective. Audit existing index state and remove or neutralize the misleading TTL index only through reviewed apply behavior. |
| Scheduler/sweeper | compatible | Do not add a periodic sweeper. Reader correctness and operator cleanup are sufficient for this bugfix. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The responsible execution agent must not choose TTL deletion, a compatibility
  fallback, a new sweeper, or a new scope model as a substitute.
- Any change to the cutover policy requires explicit user approval before
  implementation.

## Target State

`conversation_episode_state` remains short-horizon flow state keyed by
`platform`, `platform_channel_id`, and `global_user_id`.

Completed behavior:

- Active prompt reads return a row only when it is active and non-expired at the
  current turn's storage UTC time.
- Active prompt reads return empty progress when the stored row is expired,
  closed, suspended, missing expiry, invalidly expired, or non-UTC.
- Runtime cache cannot bypass the active lifecycle contract.
- RAG active progress recall cannot use progress when expiry is missing,
  invalid, expired, or unavailable.
- Cleanup closes expired active rows with a timestamped reason and preserves
  row content.
- Missing-expiry rows are either repaired from valid `updated_at`, closed from
  valid derived expiry, or left in manual-review status in the report; they are
  not active for readers.
- A new active write can replace non-active, expired, or missing-expiry rows
  despite a higher old `turn_count`.
- Bootstrap/index behavior no longer implies that TTL deletion is required for
  lane correctness.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Active definition | `status == "active"` and parsed UTC `expires_at > current_timestamp_utc` | Matches short-horizon working-memory semantics and rejects malformed active state. |
| Expired active repair | Close, not delete | Preserves diagnostic content and prevents silent data loss. |
| Missing expiry | Fail closed for active reads; derive only from valid `updated_at` during repair | Avoids immortal active state while preserving safely reconstructable current rows. |
| Invalid expiry | Manual review; fail closed | Invalid time text cannot be compared safely. |
| Cache | Validate document lifecycle before cache selection | Prevents process-local state from bypassing DB hardening. |
| Guarded writes | Allow replacement of non-active, expired, and missing-expiry rows; preserve turn-count guard for valid active rows | Prevents cleanup from blocking future episode-state recording. |
| TTL | Do not rely on TTL; do not make TTL deletion effective | Deletion is unsafe for audit and asynchronous even when configured correctly. |
| Group scope | Keep existing platform/channel/user scope | Scope redesign is not needed for the observed expired-active defect. |
| Sweeper | Operator cleanup only | Active-read filtering already protects live cognition; a background worker would add operational blast radius. |

## Contracts And Data Shapes

### Active Read Contract

Public facade remains:

```python
async def load_progress_context(
    *,
    scope: ConversationProgressScope,
    current_timestamp_utc: str,
) -> ConversationProgressLoadResult:
    ...
```

Repository and DB helpers may accept `current_timestamp_utc` so the storage
layer can enforce the same clock used by the turn. The active contract is:

```python
status == "active"
parse_storage_utc_datetime(expires_at) > parse_storage_utc_datetime(current_timestamp_utc)
```

Fail-closed cases:

- `status` missing or not `"active"`;
- `expires_at` missing, null, empty, non-string, invalid, non-UTC, or expired;
- `current_timestamp_utc` invalid.

`current_timestamp_utc` invalid is a service bug and should raise rather than
silently projecting stale progress.

### Guarded Write Contract

`upsert_episode_state_guarded(...)` must still reject stale writes to a valid
active row, but it must allow replacement when the existing row is not eligible
active memory.

Allowed replacement cases for an existing same-scope row:

- existing `status != "active"`;
- existing active row has `expires_at <= current_timestamp_utc`;
- existing active row has missing expiry;
- existing active row has no `turn_count`;
- existing active row has `turn_count < new_document["turn_count"]`.

Invalid non-empty expiry rows are not replaced by default because Mongo cannot
parse their meaning inside a narrow guarded update. The cleanup report must
classify them for manual review. Active readers still fail closed.

### Audit Finding Shape

```python
class ConversationEpisodeStateFinding(TypedDict):
    episode_state_id: str
    platform: str
    platform_channel_id: str
    global_user_id: str
    status: str
    turn_count: int | None
    updated_at: str
    expires_at: str | None
    issue_code: Literal[
        "ok_active_current",
        "expired_active",
        "missing_expiry_active",
        "invalid_expiry_active",
        "non_utc_expiry_active",
        "non_active_history",
        "missing_scope_field",
    ]
    audit_classification: Literal[
        "keep",
        "close_expired",
        "derive_expiry",
        "manual_review",
    ]
    recommended_action: Literal[
        "keep",
        "close_expired",
        "set_derived_expiry",
        "manual_review",
    ]
```

### Repair Action Shape

```python
class ConversationEpisodeStateRepairAction(TypedDict):
    action_id: str
    row_match: dict[str, object]
    action: Literal["close_expired", "set_derived_expiry", "drop_ttl_index"]
    set_fields: dict[str, object]
    reason: str
```

`row_match` must include enough fields to detect drift before mutation:
`episode_state_id`, `platform`, `platform_channel_id`, `global_user_id`,
`status`, `expires_at`, `updated_at`, and `turn_count`.

### Cleanup Mutation Fields

Closing an expired active row sets:

```python
{
    "status": "closed",
    "closed_at": storage_timestamp_utc,
    "closed_reason": "expired_active_lane_repair",
    "lifecycle_repair": {
        "plan": "conversation_episode_state_lane_lifecycle_plan",
        "action": "close_expired",
        "source_expires_at": original_expires_at,
        "applied_at": storage_timestamp_utc,
    },
    "updated_at": storage_timestamp_utc,
}
```

Setting a derived expiry on a still-current missing-expiry row sets:

```python
{
    "expires_at": derived_expires_at,
    "lifecycle_repair": {
        "plan": "conversation_episode_state_lane_lifecycle_plan",
        "action": "set_derived_expiry",
        "source_updated_at": original_updated_at,
        "applied_at": storage_timestamp_utc,
    },
    "updated_at": storage_timestamp_utc,
}
```

Do not modify semantic fields such as `current_thread`, `open_loops`,
`user_state_updates`, `assistant_moves`, or `progression_guidance`.

## LLM Call And Context Budget

No LLM call changes. This plan is deterministic lifecycle, persistence, cache,
and recall hardening only. It does not add response-path LLM calls, background
LLM calls, prompt fields, prompt budget, or model routing changes.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/db/conversation_progress.py`: add active lifecycle
  filtering, guarded replacement semantics, and narrow DB update mechanics.
- `src/kazusa_ai_chatbot/conversation_progress/repository.py`: pass current
  storage time into DB load/write helpers and keep active validation at the
  conversation-progress boundary.
- `src/kazusa_ai_chatbot/conversation_progress/runtime.py`: pass
  `current_timestamp_utc` into cache selection and use active-only
  `episode_state` for prompt and recorder input.
- `src/kazusa_ai_chatbot/conversation_progress/cache.py`: reject cached rows
  that are non-active, expired, missing expiry, or invalidly expired at the
  current storage time.
- `src/kazusa_ai_chatbot/rag/recall/collectors/progress.py`: fail closed when
  active progress lacks a valid unexpired expiry basis.
- `src/kazusa_ai_chatbot/db/script_operations.py`: add read-only audit,
  report-driven repair planning, report-driven apply, index inspection, and
  non-deleting lifecycle mutation helpers for this lane.
- `src/kazusa_ai_chatbot/db/bootstrap.py`: stop treating TTL deletion as
  episode-state correctness. Add or keep non-TTL lookup/index support required
  by the new active read contract and prevent a live TTL-delete path from being
  introduced for this lane.

### Create

- `src/scripts/audit_conversation_episode_state_lane.py`: read-only operator
  report for counts, findings, and index state.
- `src/scripts/repair_conversation_episode_state_lane.py`: dry-run by default;
  apply only from a reviewed input report and explicit `--apply`.
- `tests/test_conversation_episode_state_lane_lifecycle.py`: focused tests for
  audit classifications, dry-run/apply idempotency, active-read filtering,
  guarded replacement, and index policy.

### Test

- `tests/test_conversation_episode_state.py`: update existing DB helper tests
  for active-read filters and guarded replacement.
- `tests/test_conversation_episode_cache.py`: add cache lifecycle validation
  tests.
- `tests/test_conversation_progress_runtime.py`: add runtime tests proving
  expired or malformed rows produce empty progress and do not reach recorder
  prior state.
- `tests/test_rag_recall_agent.py` or a focused RAG collector test file: add
  fail-closed tests for expired, missing, and invalid progress expiry.
- `tests/test_db.py`: update bootstrap/index assertions for the lane's non-TTL
  correctness policy.

### Keep

- Existing public facade names:
  `load_progress_context(...)` and `record_turn_progress(...)`.
- Existing storage scope: `platform`, `platform_channel_id`, `global_user_id`.
- Existing prompt-facing progress shape.
- Existing semantic recorder behavior.
- Existing relevance boundary: conversation progress loads only after relevance
  allows a response.

## Overdesign Guardrail

- Actual problem: expired and malformed active `conversation_episode_state`
  rows can remain eligible active memory and affect live cognition.
- Minimal change: enforce active lifecycle at DB load, cache selection, RAG
  progress recall, cleanup, and guarded write replacement; repair existing rows
  through reviewed non-deleting maintenance actions.
- Ownership boundaries: deterministic database and conversation-progress code
  owns status, expiry, cleanup, indexes, and write guards; RAG recall owns
  active-evidence admission from already-loaded state; cognition and dialog own
  semantic response decisions after stale state has been excluded.
- Rejected complexity: no TTL deletion, no new background sweeper, no new
  archive collection, no group-thread scope redesign, no compatibility fallback
  that lets expired progress through, no prompt rewrite, no LLM-based cleanup.
- Evidence threshold: a future plan may add archival or scheduled cleanup only
  after this repair produces stable dry-run/apply evidence and a separate user
  approves the added operational behavior.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, TTL deletion, or
  extra features.
- The responsible agent must not delete episode-state rows.
- The responsible agent must treat changes outside conversation progress, DB
  helpers, RAG progress recall, and the two operator scripts as out of scope.
- The responsible agent must search for existing equivalent maintenance helper
  patterns before adding helpers in `db.script_operations`.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, the responsible agent must preserve the plan's
  stated active-memory intent and record the discrepancy in `Execution
  Evidence`.
- If a required instruction is impossible, the responsible agent must stop and
  report the blocker instead of inventing a substitute.

## Implementation Order

1. Parent establishes the focused test contract.
   - Add tests for active DB load filtering, cache lifecycle filtering, guarded
     replacement after closed/expired rows, RAG recall fail-closed behavior,
     audit classification, and cleanup idempotency.
   - Run the focused tests before production changes and record expected
     failures or current permissive behavior.
2. Parent starts one production-code subagent after the focused tests exist.
   - Scope: production code under `conversation_progress`, `db`,
     `rag/recall/collectors/progress.py`, and the two operator scripts only.
3. Production-code subagent implements active validation helpers.
   - Add a single shared deterministic helper inside the
     conversation-progress boundary for "is active at storage UTC" if it
     removes duplication between repository/cache tests and implementation.
   - Do not add a generic cross-subsystem time abstraction.
4. Production-code subagent updates DB load and guarded write semantics.
   - DB load must be active-only.
   - Guarded write must allow replacement of non-active, expired, and
     missing-expiry rows while preserving newer-active protection.
5. Production-code subagent updates runtime cache selection.
   - Runtime passes current storage time to cache selection.
   - Cache rejects expired, malformed, and non-active rows.
6. Production-code subagent updates RAG progress recall admission.
   - Missing or invalid expiry fails closed.
7. Production-code subagent implements audit and repair helpers in
   `db.script_operations`.
   - Audit is read-only.
   - Dry-run produces planned actions.
   - Apply requires reviewed input and drift checks.
8. Production-code subagent adds the operator CLI wrappers.
   - Scripts parse args, write JSON reports, close DB handles, and call only
     `db.script_operations` for DB mechanics.
9. Parent runs focused tests and fixes test-contract issues inside scope.
10. Parent runs static greps and broader regression tests listed in
    `Verification`.
11. Parent runs read-only audit and repair dry-run only after explicit user
    command for live DB inspection.
12. Parent starts one independent code-review subagent after planned
    verification passes.
13. Parent remediates review findings inside the approved change surface and
    reruns affected verification.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it; closes after planned
  production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - focused lifecycle test contract established.
  - Covers: active DB load, cache filtering, guarded replacement, RAG fail
    closed, audit classification, cleanup idempotency, index policy, and group
    scope isolation tests.
  - Verify: run focused tests and record expected failures or baseline
    permissive behavior in `Execution Evidence`.
  - Handoff: production-code subagent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - active read, cache, recall, and guarded write hardening complete.
  - Covers: `db.conversation_progress`, `conversation_progress.repository`,
    `conversation_progress.runtime`, `conversation_progress.cache`, and
    `rag.recall.collectors.progress`.
  - Verify: focused lifecycle tests pass.
  - Handoff: next stage implements maintenance helpers.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - audit and repair maintenance path complete.
  - Covers: `db.script_operations`, audit script, repair script, dry-run/apply
    report shapes, and drift checks.
  - Verify: focused audit/repair tests pass; dry-run mode performs no writes.
  - Handoff: next stage runs regression tests and static checks.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - verification and operator dry-run evidence recorded.
  - Covers: all commands in `Verification`, with live DB commands only when
    explicitly approved by the user.
  - Verify: deterministic tests pass; read-only audit and repair dry-run reports
    are recorded when live DB inspection is approved.
  - Handoff: independent code review starts.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - independent code review completed and remediated.
  - Covers: review findings, fixes, rerun commands, residual risks, and final
    approval status.
  - Verify: affected tests and static checks rerun after review fixes.
  - Handoff: parent may request approval for live apply or final lifecycle
    update.
  - Sign-off: `<agent/date>` after review evidence is recorded.

## Verification

### Deterministic Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_conversation_episode_state.py tests/test_conversation_episode_cache.py tests/test_conversation_progress_runtime.py tests/test_conversation_episode_state_lane_lifecycle.py -q
venv\Scripts\python.exe -m pytest tests/test_conversation_progress_cognition.py tests/test_rag_recall_agent.py -q
venv\Scripts\python.exe -m pytest tests/test_db.py -q
```

Expected result: all selected deterministic tests pass.

### Static Boundary Checks

```powershell
rg -n "expireAfterSeconds=0|conversation_episode_expires_at_ttl" src\kazusa_ai_chatbot\db tests
```

Expected result: no production code creates or depends on TTL deletion for
`conversation_episode_state`. Test references are allowed only when they assert
the TTL-delete path is absent or neutralized.

```powershell
rg -n "conversation_episode_state|conversation_progress" src\kazusa_ai_chatbot\nodes\relevance_agent.py
```

Expected result: no matches. Relevance must remain independent from
conversation progress.

```powershell
rg -n "load_episode_state\(" src\kazusa_ai_chatbot tests
```

Expected result: all production call sites pass current storage time or route
through a facade that does.

### Operator Dry-Run Commands

Run only after explicit user approval for live DB inspection:

```powershell
venv\Scripts\python.exe -m scripts.audit_conversation_episode_state_lane --output test_artifacts\conversation_episode_state_lane_audit.json
venv\Scripts\python.exe -m scripts.repair_conversation_episode_state_lane --dry-run --output test_artifacts\conversation_episode_state_lane_repair_dry_run.json
```

Expected result:

- audit reports counts for total, active, expired active, missing expiry,
  invalid expiry, non-UTC expiry, non-active history, and index state;
- repair dry-run reports planned non-deleting actions and performs zero writes;
- rows requiring manual review are listed separately and are not counted as
  apply-ready actions.

### Apply Verification

Run only after dry-run review and explicit user approval for live apply:

```powershell
venv\Scripts\python.exe -m scripts.repair_conversation_episode_state_lane --apply --input test_artifacts\conversation_episode_state_lane_repair_dry_run.json --output test_artifacts\conversation_episode_state_lane_repair_apply.json
venv\Scripts\python.exe -m scripts.audit_conversation_episode_state_lane --output test_artifacts\conversation_episode_state_lane_audit_after_apply.json
```

Expected result:

- apply report includes before counts, after counts, applied count,
  drift-skipped count, blocked/manual-review count, and zero deletes;
- post-apply audit reports zero expired active rows excluding manual-review
  rows that are already fail-closed for active reads.

## Independent Plan Review

Run this gate before approval, execution, or handoff. If no separate reviewer is
available, the drafting agent must reread this plan, `development_plans/README.md`,
the development-plan references, and the relevant source/test context from a
fresh-review posture.

Review scope:

- Plan status remains `draft` until the user approves execution.
- Every user-requested item is explicit: issue description, malformed-data
  removal plan, RCA, corrupted-source hardening plan, and robustness proof.
- The plan names exact ownership boundaries: conversation progress, DB helpers,
  RAG progress recall, operator scripts, and no cognition/dialog ownership for
  lifecycle mechanics.
- The plan prevents cleanup from blocking future writes after closed or expired
  high-turn rows.
- The plan rejects TTL deletion and row deletion by default.
- The plan includes dry-run/apply behavior, drift checks, data migration safety,
  active-read contract, cleanup idempotency, verification, and residual risks.

Record blockers, non-blocking findings, required edits, residual risks, and
approval status in `Execution Evidence` or the plan-review section before
approval.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, script,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, persistence risk, stale cache behavior,
  unsafe TTL/index changes, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including focused and regression tests,
  execution evidence, dry-run/apply artifacts, and path-safe commands.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Active readers ignore expired, closed, suspended, missing-expiry,
  invalid-expiry, and non-UTC episode-state rows.
- Runtime cache cannot return a stale or malformed active document.
- RAG active progress recall fails closed without a valid unexpired expiry.
- Cleanup dry-run reports expired active rows, missing-expiry rows, invalid
  expiry rows, non-UTC expiry rows, and index state.
- Apply mode closes approved expired active rows idempotently and deletes zero
  documents.
- Missing-expiry rows are repaired only when `updated_at` proves a derived
  expiry; otherwise they remain manual-review and inactive for reads.
- New active writes can replace closed, expired, or missing-expiry rows without
  weakening newer-active write protection.
- Post-apply audit reports zero expired active rows, excluding manual-review
  rows that active readers already fail closed.
- Verification commands pass and independent code review is recorded.

## Data Migration

1. Export `conversation_episode_state` before mutation.
2. Run read-only audit.
3. Generate repair dry-run.
4. Review planned actions and manual-review rows.
5. Apply only from reviewed dry-run report after explicit user approval.
6. Re-run audit.
7. Record before/after counts, applied actions, drift skips, manual-review
   rows, and zero-delete confirmation in `Execution Evidence`.

Rollback:

- Use the pre-apply export and apply report to restore prior `status`,
  `expires_at`, `updated_at`, and lifecycle repair fields for rows touched by
  apply mode.
- Do not roll back by deleting rows.
- If a normal post-repair write has already overwritten the one row for a
  scope, restore only after explicit user approval because it can replace newer
  active state.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Stale progress still reaches cognition | Active DB read, cache, and RAG recall all fail closed | Focused tests and RAG tests |
| Cleanup blocks new writes | Guarded replacement covers closed, expired, and missing-expiry rows | Guarded replacement test with higher old `turn_count` |
| Useful diagnostic history deleted | No deletes; pre-apply export; close by default | Apply report zero-delete assertion |
| TTL unexpectedly deletes rows | Do not make TTL effective; audit/index checks; no BSON expiry conversion | Static grep and bootstrap/index tests |
| Missing-expiry rows remain confusing | Fail closed for reads; derive only from valid `updated_at`; manual-review report | Audit and repair dry-run tests |
| Clock skew or timezone bugs misclassify rows | Use storage UTC parser and current turn storage timestamp | Unit tests for `+00:00`, `Z`, non-UTC, invalid, past, and future cases |
| Group-channel state leaks across unrelated subthreads | Keep scope unchanged and enforce short expiry; no thread-scope redesign | Scope isolation tests |
| Apply mutates rows that changed after dry-run | Apply matches row identity and lifecycle fields from the reviewed report | Drift-skip tests |

## Self Plan Review And Remediation

Review performed during draft refinement on 2026-07-02. The review checked the
user-requested scope, `development_plans/README.md`, the development-plan
contract references, required top matter, required sections, status, line
budget, placeholder scan, and source-backed ownership boundaries.

Findings and remediation:

- Finding: the repo plan contract treats live data mutation and production
  behavior changes as high-risk migration scope, while the original draft was
  classed as `large`. Remediation: changed the plan class to
  `high_risk_migration` and kept the expanded draft within the 1200-line maximum
  for that class.
- Finding: the plan needed explicit coverage for every user-requested item.
  Remediation: added named sections for issue description, malformed-data
  removal, RCA, corrupted-source hardening, and robustness proof, and verified
  those headings remained present after editing.
- Finding: the plan contract requires concrete execution gates rather than
  broad implementation instructions. Remediation: added active-read, guarded
  write, audit, repair, verification, progress-checklist, independent plan
  review, and independent code-review gates with exact file paths and commands.
- Finding: placeholder and unresolved-choice language would make the plan unsafe
  for later approval. Remediation: scanned for placeholder markers,
  unresolved-choice phrasing, delayed-implementation wording, copy-forward
  wording, broad edge-case wording, and vague test instructions; no actionable
  matches remained.
- Finding: the original draft treated expired active reads and cleanup as the
  main issue but did not account for the unique scope index plus turn-count-only
  write guard. Remediation: added the write-lock failure mode, guarded write
  contract, implementation steps, tests, and acceptance criteria for replacing
  closed, expired, and missing-expiry rows.
- Finding: the original draft mentioned TTL risk but did not connect it to the
  current string `expires_at` storage shape or deletion hazard. Remediation:
  added bootstrap/index context, TTL cutover policy, static verification, and
  the explicit rule to avoid BSON expiry conversion and TTL deletion.
- Finding: cache and RAG recall were not covered even though both can carry
  current progress into cognition. Remediation: added cache and RAG recall
  failure modes, change surface, tests, and bigbang fail-closed policy.
- Finding: missing expiry was reported but not operationally classified.
  Remediation: added audit classifications, derived-expiry rules, manual-review
  handling, and fail-closed active-read behavior.
- Finding: group-thread scope was not explicit. Remediation: added group-scope
  failure analysis, decision to keep current platform/channel/user scope, and
  scope-isolation verification.
- Finding: dry-run/apply behavior lacked drift safety. Remediation: added
  report-driven apply, row-match requirements, before/after counts, drift-skip
  reporting, and rollback instructions.
- Finding: the requested proof of robustness was too narrow. Remediation:
  expanded verification across deterministic tests, static greps, operator
  dry-run/apply gates, independent plan review, and independent code review.

Residual risks:

- Existing production data has one operational row per scope, so preserving all
  historical content after future normal writes would require a separate archive
  plan. This plan preserves evidence through pre-apply export and apply reports.
- Invalid non-empty expiry rows cannot be repaired automatically without
  interpreting malformed time text. They are fail-closed and require manual
  review.
- This plan does not add a sweeper. Active-read filtering protects live
  cognition; recurring maintenance can be planned after operator evidence shows
  remaining accumulation pressure.

## Execution Evidence

Cleanup-only execution completed on 2026-07-03. New-write hardening remains
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

Commands and artifacts:

- `venv\Scripts\python.exe -m py_compile src\scripts\_lane_cleanup.py src\scripts\repair_conversation_episode_state_lane.py src\kazusa_ai_chatbot\db\script_operations.py`
- `venv\Scripts\python.exe -m scripts.repair_conversation_episode_state_lane --dry-run --output test_artifacts\conversation_episode_state_lane_repair_dry_run.json`
- `venv\Scripts\python.exe -m scripts.repair_conversation_episode_state_lane --apply --input test_artifacts\conversation_episode_state_lane_repair_dry_run.json --output test_artifacts\conversation_episode_state_lane_repair_apply.json`
- `venv\Scripts\python.exe -m scripts.repair_conversation_episode_state_lane --dry-run --output test_artifacts\conversation_episode_state_lane_post_repair_dry_run.json`

Results:

- Baseline: 180 total rows, 174 active rows, 149 expired-active findings, 149
  deterministic planned actions.
- Apply: 149 expired active rows closed, 0 deletes, 0 blocked actions.
- Post-audit: 0 findings and 0 deterministic planned actions remain.
