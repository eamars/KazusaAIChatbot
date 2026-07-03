# user memory units lane data integrity plan

## Summary

- Goal: repair malformed `user_memory_units` rows and harden user-memory
  generation so plausible user continuity persists only in the correct lane,
  with validated user target, durable semantics, source provenance, and
  explicit commitment lifecycle.
- Plan class: high_risk_migration.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `database-data-pull`, `debug-llm`, `py-style`,
  `cjk-safety`, and `test-style-and-execution`.
- Overall cutover strategy: bigbang for new user-memory write validation;
  migration for existing rows through export, dry-run classification, reviewed
  apply, and post-apply audit.
- Highest-risk areas: deleting plausible source-less legacy continuity,
  continuing to drop new provenance through the `evidence_refs` /
  `source_refs` mismatch, treating volatile public facts as user memory,
  archiving real ongoing commitments as stale, and allowing scoped RAG to
  reinforce wrong-subject memory.
- Acceptance criteria: new non-diagnostic user-memory writes require a
  validated real user target, non-empty deterministic `source_refs`, accepted
  subject scope, sane timestamps, and explicit commitment schedule semantics;
  malformed legacy rows are archived, repaired, or marked legacy-unverified
  through reviewed migration evidence; focused deterministic, live DB, and
  one-at-a-time live LLM checks prove robustness.

New-write hardening supersession: all lane-specific new-write hardening,
source-generation proof, and fixing-strategy instructions in this plan are
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.
Execute this lane plan for cleanup, audit, dry-run/apply, data migration, and
post-cleanup verification only. If another section still names new-write
validators, prompts, or tests, treat that text as historical context rather
than execution scope.

## Context

`user_memory_units` is the durable user-scoped continuity lane. It stores
facts, preferences, milestones, stable/recent interaction patterns, and active
commitments for one real `global_user_id`. The product basis is explicit:
invented or learned user continuity can be good data when it is grounded in
the interaction, useful for the character, and scoped to the correct user.
Good data in the wrong memory lane, without provenance, or with an invalid
lifecycle is bad data.

The 2026-07-02 audit found 821 `user_memory_units` rows across 79 users:

- 655 active, 82 completed, 76 archived, and 8 cancelled rows;
- unit types: 380 objective facts, 209 active commitments, 136 recent shifts,
  74 milestones, and 22 stable patterns;
- all 821 rows had empty `source_refs`;
- 43 active commitments existed, and 40 active commitments had no `due_at`;
- 2 rows had future timestamp anomalies;
- case review found plausible but unverifiable rows, time-limited
  unverifiable rows, volatile public facts, episodic artifacts, likely stale
  commitments, wrong-subject rows, and invalid timestamps.

Current code inspection adds concrete RCA evidence:

- `src/kazusa_ai_chatbot/consolidation/memory_units.py` prompts the extractor
  to emit `evidence_refs`, but `_candidate_validation_errors(...)` only checks
  that the field is a list and allows an empty list.
- `src/kazusa_ai_chatbot/db/user_memory_units.py` persists
  `unit.get("source_refs")`, not `evidence_refs`, so even a valid extractor
  evidence list is dropped on create unless another caller maps it first.
- `update_user_memory_unit_semantics(...)` updates semantic text and
  `merge_history`, but does not append current-turn source refs on merge or
  evolve, so reinforced memories still do not gain provenance.
- `build_consolidation_target_plan(...)` validates that a real user profile
  exists and matches `global_user_id`, but no user-memory-specific validator
  proves the candidate subject is the current user or the
  user-character relationship.
- `origin_policy.py` allows `user_memory_units` writes for supported
  `internal_thought` origins; that path still needs real-user target proof and
  source refs because internal thought text is not itself user speech.
- RAG scoped retrieval reads active rows by the current `global_user_id` and
  projects them as `source_system="user_memory_units"` with
  `scope_type="user_continuity"`. Bad rows therefore affect live evidence and
  later merge/evolve decisions, not only offline storage.
- `query_active_commitment_memory_units(...)` only returns active commitments
  with parseable `due_at` for due checks, while
  `query_active_commitment_memory_units_for_user(...)` intentionally includes
  no-due rows for lifecycle review. The plan must distinguish unscheduled
  ongoing rules from malformed time-bound commitments.

## Lane Analysis Requirements

### Issue Description Based On Deep Analysis

The lane mixes plausible user continuity with structurally unsafe records. The
common storage defect is empty `source_refs`, but the user-visible risk comes
from several failure modes acting together:

- source evidence is requested from the extractor as `evidence_refs` but
  persistence stores only `source_refs`;
- source-less candidates pass validation and tests;
- merge/evolve operations do not append current source refs;
- target planning proves a real user row, but not candidate subject scope;
- active commitments can be written without a due date or an explicit
  non-scheduled class;
- volatile public/current facts can be remembered as durable user facts;
- timestamp anomalies can make rows appear future-authored or distort recency;
- scoped RAG reuses active rows as current-user continuity and can reinforce
  the wrong memory during consolidation;
- legacy source-less data may contain real relationship continuity and must not
  be mass-deleted simply because it lacks provenance.

### Plan To Remove Malformed Data

Remove malformed data from active retrieval, not by blanket deletion. The
maintenance path must export the lane, classify every row, generate a dry-run
repair report, and apply only reviewed deterministic actions:

- mark plausible source-less rows as `legacy_unverified` provenance while
  keeping them active when they are user-scoped and durable;
- archive wrong-subject, character-global, group/channel, episode-local, and
  volatile public/current rows so scoped RAG no longer retrieves them;
- repair parseable timestamp anomalies by restoring safe UTC timestamps from
  row evidence or migration review metadata;
- archive irreparable timestamp rows whose recency cannot be trusted;
- archive stale or malformed time-bound active commitments;
- preserve no-due active commitments only when they are explicit ongoing rules
  or manually reviewed open commitments;
- never delete plausible source-less continuity unless a reviewed apply report
  marks the row as deterministic corruption and the user explicitly approves
  deletion in a later cleanup command.

### RCA Of The Failure Mode

The root cause is a contract split across extraction, validation, persistence,
and retrieval:

1. The LLM extractor was asked for `evidence_refs`, while the DB document
   builder accepted only `source_refs`. The expected lineage field is lost at
   the create boundary.
2. Candidate validation treated evidence refs as optional structure, not as a
   mandatory new-write provenance contract.
3. DB document construction defaulted missing `source_refs` to `[]`, which is
   necessary for legacy-tolerant reads but unsafe for new writes.
4. Merge/evolve persistence tracked a merge reason but not the current
   supporting source refs, so old source-less rows could stay source-less
   forever.
5. Target validation is lane-level. It prevents synthetic user ids and missing
   profile rows, but it does not validate that the candidate memory itself is
   about the target user, not the character, another user, a group, or a public
   entity.
6. Commitment lifecycle was encoded mostly in free text plus optional `due_at`;
   the system had no deterministic way to separate a dated promise from an
   accepted ongoing rule when `due_at` was absent.
7. Prompt instructions discouraged unresolved relative commitments and volatile
   facts, but deterministic code had no structural contract for subject scope,
   durability scope, source refs, or non-scheduled commitment class.
8. Tests covered structural triples, due-date normalization, merge safety, and
   target routing, but fixtures allowed empty `evidence_refs` and did not prove
   source refs persisted or were appended.

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

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing user-memory extraction
  prompts, source-ref handoff, RAG projection, or write-intent contracts.
- `no-prepost-user-input`: load before changing code that decides whether
  user instructions, preferences, permissions, accepted commands, or
  commitments should be persisted.
- `database-data-pull`: load before exporting, inspecting, or sampling live
  `user_memory_units` data.
- `debug-llm`: load before live consolidation, prompt, or generation quality
  checks.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files with CJK prompt or memory
  strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute this plan while status is `draft`.
- Do not apply live data cleanup without a reviewed dry-run report and
  explicit user command.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Use public database facade helpers or `db.script_operations`; no raw MongoDB
  access outside the database package.
- Do not mass-delete source-less plausible legacy memories.
- New non-diagnostic user-memory writes must have non-empty canonical
  `source_refs`; `evidence_refs` is not a runtime compatibility alias for new
  writes.
- LLM stages extract semantic memory candidates, subject scope, durability
  scope, and commitment schedule meaning. Deterministic code owns target
  validation, source refs, timestamps, lifecycle shape, write permission,
  persistence, cache invalidation, and migration mechanics.
- Do not add keyword-only semantic blockers over user text. If the LLM chooses
  the wrong lane, fix prompt/schema/test coverage rather than adding
  deterministic user-text classifiers.
- Prompt changes must keep stable rules in `SystemMessage`, per-run facts in
  `HumanMessage`, and no added LLM calls.
- Prompt-facing RAG evidence must not expose raw source refs, raw platform ids,
  DB ids, embeddings, or migration metadata as prose.
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

- Add a user-memory lane audit that classifies source refs, source-ref shape,
  user target validity, subject alignment, durability scope, commitment
  lifecycle, timestamp sanity, volatile public facts, RAG retrieval impact, and
  legacy source-less risk.
- Add a dry-run/apply repair path for deterministic malformed classes.
- Replace new-write `evidence_refs` handoff with canonical `source_refs`.
- Attach trusted source refs from consolidation origin metadata before
  persistence.
- Enforce real user target validation before new user-memory persistence.
- Enforce non-empty canonical source refs for new non-diagnostic user-memory
  creates and merge/evolve updates.
- Add subject-scope and durability-scope validation without keyword-only
  user-text filtering.
- Add commitment lifecycle validation for active commitments, including a
  required `due_at` or explicit non-scheduled class.
- Add tests proving valid invented or learned user-scoped data persists with
  source refs and wrong-lane, wrong-subject, source-less, volatile, and
  malformed commitment data is rejected or migrated.
- Prove cleaned data does not break scoped user-memory RAG retrieval or active
  commitment recall.

## Deferred

- Do not reconstruct source refs for every legacy row from full chat history.
- Do not delete plausible source-less rows solely because they are
  unverifiable.
- Do not redesign affinity, relationship scoring, or user-profile identity.
- Do not redesign RAG helper routing or scoped-memory retrieval strategy.
- Do not move volatile public facts into a new durable collection in this
  plan; only stop and repair user-memory persistence.
- Do not change dialog behavior to hide bad memory.
- Do not add compatibility shims, dual-write fields, fallback aliases, or
  migration bridges for new runtime writes.
- Do not retire commitments by deterministic age alone.
- Do not add extra LLM calls, retry loops, or generic truth adjudication.

## Cutover Policy

Overall strategy: bigbang for new writes; migration for existing data.

| Area | Policy | Instruction |
| --- | --- | --- |
| New user-memory writes | bigbang | Reject writes without validated user target, canonical `source_refs`, accepted subject scope, and accepted durability scope. |
| `evidence_refs` legacy handoff | bigbang | Stop using `evidence_refs` in the user-memory runtime contract; new extraction and tests use `source_refs`. Do not add a runtime alias mapper. |
| Source-ref generation | bigbang | Build trusted source refs from `consolidation_origin` and active turn identifiers before persistence. |
| Commitments | bigbang | Active commitments require `due_at` for scheduled/time-bound commitments or an explicit non-scheduled class for accepted ongoing rules. |
| Existing source-less rows | migration | Preserve plausible rows as legacy-unverified unless deterministic corruption is found. |
| Existing malformed rows | migration | Repair timestamps and archive stale, wrong-subject, group/channel, character-global, episode-local, and volatile rows after dry-run review. |
| RAG scoped memory | compatible | Keep the current result shape and scope contract; cleaned row status/provenance changes alter available evidence but not public API shape. |
| Cache invalidation | bigbang | Emit existing user-profile/user-memory invalidation only for successful durable writes and approved migration mutations. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- For bigbang areas, delete or rewrite legacy new-write references instead of
  preserving them.
- For migration areas, follow the export, dry-run, review, apply, post-audit,
  and rollback gates in this plan.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed here.
- Any change to a cutover policy requires user approval before implementation.

## Target State

`user_memory_units` stores durable facts about a specific real user or that
user's relationship with the active character:

- preferences, habits, milestones, stable patterns, recent shifts, and durable
  user-specific lore learned through interaction;
- active commitments with explicit schedule semantics and lifecycle state;
- source refs identifying the episode, conversation row, message, or trusted
  internal source that caused the write;
- provenance labels for migrated source-less rows when direct source evidence
  is unavailable.

It does not store:

- character-global mood, self-image, or state;
- another user's facts;
- group/channel atmosphere;
- episode-local facts without durable value;
- volatile public/current facts such as prices, weather, schedules, current
  model specs, current public status, or live web answers;
- source-less new writes;
- active scheduled commitments without due state.

Scoped RAG continues to retrieve only current-user active rows. It may surface
`legacy_unverified` rows as scoped continuity with uncertainty/provenance
metadata, but prompt-facing evidence text must remain semantic and must not
render raw source refs or migration internals.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Legacy source-less rows | Preserve as `legacy_unverified` unless deterministically malformed | They may be valid continuity even without auditable source refs. |
| New source refs | Required canonical `source_refs` | Future correctness must be auditable and queryable. |
| Source-ref owner | Deterministic consolidation code builds refs from trusted origin metadata | Local LLMs should not invent operational provenance. |
| `evidence_refs` | Remove from new runtime user-memory contract | The name mismatch is the observed provenance-loss bug. |
| Subject scope | LLM emits semantic scope; deterministic code validates closed accepted values | Avoid keyword-only user-text filters while still failing closed structurally. |
| Volatile facts | Reject from durable user memory by `durability_scope` contract | Live/current evidence owns changing public facts. |
| Commitments | Keep `active_commitment`, add explicit schedule class | Ongoing rules and dated promises need different mechanics. |
| Migration mutation style | Archive/mark/repair, not blanket delete | Keeps rollback and manual review possible. |
| RAG public shape | Preserve current capability result shape | The bug is data integrity, not RAG API redesign. |

## Contracts And Data Shapes

### Canonical Source Ref

Use the existing `source_refs` field and extend its documented shape through
`src/kazusa_ai_chatbot/db/schemas.py`:

```python
class UserMemoryUnitSourceRef(TypedDict, total=False):
    source: str
    timestamp: str
    message_id: str
    conversation_row_id: str
    episode_id: str
    platform: str
    platform_channel_id: str
    trigger_source: str
```

Minimum valid new-write source ref:

- `source` is one of `conversation_history`, `cognitive_episode`, or
  `internal_source_case`;
- `timestamp` is a parseable storage UTC timestamp;
- at least one of `conversation_row_id`, `message_id`, or `episode_id` is
  non-empty.

For normal `user_message` consolidation, source refs come from
`consolidation_origin.active_turn_conversation_row_ids`,
`platform_message_id`, `episode_id`, and `storage_timestamp_utc`.

For `internal_thought` consolidation, source refs must identify the internal
source case or cognitive episode plus any source rows that led to the
user-memory update. If no trusted source identifier exists, the write is
rejected.

### User Memory Candidate Contract

The extractor output changes to canonical fields:

```python
class UserMemoryUnitCandidate(TypedDict):
    unit_type: Literal[
        "stable_pattern",
        "recent_shift",
        "objective_fact",
        "milestone",
        "active_commitment",
    ]
    subject_scope: Literal[
        "current_user",
        "current_user_relationship",
        "character",
        "other_user",
        "group_channel",
        "public_external",
        "episode_local",
    ]
    durability_scope: Literal[
        "durable_user_continuity",
        "volatile_current_fact",
        "episode_only",
        "wrong_lane",
    ]
    commitment_schedule_kind: Literal[
        "scheduled_due",
        "ongoing_rule",
        "not_commitment",
    ]
    fact: str
    subjective_appraisal: str
    relationship_signal: str
    due_at: str | None
```

Validation accepts only:

- `subject_scope in {"current_user", "current_user_relationship"}`;
- `durability_scope == "durable_user_continuity"`;
- `commitment_schedule_kind == "not_commitment"` for non-commitment unit
  types;
- `commitment_schedule_kind == "scheduled_due"` with parseable `due_at`, or
  `commitment_schedule_kind == "ongoing_rule"` with empty `due_at`, for
  `active_commitment`.

The LLM decides these semantic fields. Deterministic code validates enum
membership and permitted combinations; it does not keyword-match the user's
text to change the decision.

### New Write Validator Contract

Add a local validator in the consolidation/user-memory write path with this
effective contract:

```python
def validate_user_memory_unit_write_candidate(
    candidate: Mapping[str, object],
    *,
    global_user_id: str,
    target_plan: ConsolidationTargetPlan,
    source_refs: list[UserMemoryUnitSourceRef],
    storage_timestamp_utc: str,
) -> dict[str, object]:
    ...
```

The returned dict is the persistence-ready candidate with canonical
`source_refs`. The function raises `ValueError` for structural failures.

Required checks:

- `validate_write_intent(...)` accepts `target_alias="current_user"` and
  `write_lane="user_memory_units"`;
- `global_user_id` matches the validated target id;
- `source_refs` is non-empty and each ref satisfies the shape above;
- the semantic triple fields are non-empty;
- `subject_scope`, `durability_scope`, `unit_type`, and
  `commitment_schedule_kind` are valid;
- active-commitment schedule combination is valid;
- `due_at`, when present, parses as storage UTC after local-time normalization;
- created/updated/seen timestamps are deterministic storage timestamps, not
  LLM-authored values;
- new writes cannot set `completed_at`, `cancelled_at`, or `archived_at`.

### Audit Finding Contract

Add a maintenance audit finding:

```python
class UserMemoryUnitFinding(TypedDict):
    memory_unit_id: str
    global_user_id: str
    unit_type: str
    status: str
    issue_code: str
    issue_description: str
    severity: Literal["info", "warning", "error", "blocker"]
    recommended_action: Literal[
        "keep",
        "mark_legacy_unverified",
        "archive",
        "repair_timestamp",
        "downgrade_commitment",
        "mark_ongoing_rule",
        "manual_review",
        "block_apply",
    ]
    evidence_fields: dict[str, object]
```

Audit classifications must include:

- `missing_source_refs`;
- `invalid_source_ref_shape`;
- `legacy_source_less_plausible`;
- `target_profile_missing`;
- `target_profile_mismatch`;
- `synthetic_or_invalid_global_user_id`;
- `subject_current_user`;
- `subject_current_user_relationship`;
- `wrong_subject_other_user`;
- `wrong_subject_character`;
- `wrong_subject_group_channel`;
- `volatile_public_fact`;
- `episode_local_artifact`;
- `active_commitment_missing_due_timebound`;
- `active_commitment_no_due_ongoing_rule`;
- `active_commitment_stale_or_obsolete`;
- `inactive_commitment_missing_terminal_timestamp`;
- `future_non_due_timestamp`;
- `invalid_timestamp_format`;
- `rag_active_retrieval_risk`;
- `manual_review_required`.

### Migration Metadata

Approved apply operations must write bounded metadata, either in row-level
fields or `merge_history`:

```python
{
    "operation": "lane_integrity_migration",
    "audit_run_id": "user_memory_units_lane_audit_<UTC>",
    "action": "mark_legacy_unverified | archive | repair_timestamp | downgrade_commitment | mark_ongoing_rule",
    "reason_code": "missing_source_refs | volatile_public_fact | ...",
    "previous_status": "active",
    "previous_unit_type": "active_commitment",
    "previous_fields": {"updated_at": "..."},
    "timestamp": "storage UTC timestamp"
}
```

Rows marked as plausible source-less legacy continuity should preserve active
retrieval and set provenance labels such as:

```python
{
    "authority": "legacy_unverified_continuity",
    "truth_status": "legacy_unverified",
    "origin": "source_missing_migration"
}
```

## LLM Call And Context Budget

Affected path: background consolidation user-memory extraction.

- Added LLM calls: none.
- Removed LLM calls: none.
- Response path: unchanged; user-memory generation remains post-turn
  consolidation.
- Prompt/schema change: update the extractor to emit `subject_scope`,
  `durability_scope`, `commitment_schedule_kind`, and canonical field names.
- Prompt/schema removal: remove `evidence_refs` from user-memory extractor,
  merge, rewrite, and stability prompt contracts; deterministic code attaches
  trusted `source_refs`.
- Context input change: no new broad context. The extractor already receives
  current turn origin, recent chat, final dialog, RAG user-memory context, new
  facts, promises, and appraisal evidence.
- Context budget: no increase beyond the existing consolidation prompt budget.
  The added enum fields are output-only and replace the old optional evidence
  field in model output.
- Latency impact: no normal-path latency increase.
- Deterministic owner: validation and persistence. The LLM must not decide
  whether missing source refs are acceptable.
- Verification: prompt-render tests plus one-at-a-time live LLM extractor
  cases for valid user preference, valid milestone, valid stable pattern,
  wrong-subject rejection, volatile fact rejection, and no-due ongoing
  commitment classification.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/consolidation/memory_units.py`: change extractor,
  merge, rewrite, and stability contracts from `evidence_refs` to canonical
  source-ref handling; build trusted source refs from consolidation origin;
  add candidate validator; append source refs on merge/evolve; reject invalid
  subject, durability, lifecycle, and timestamp shapes before DB writes.
- `src/kazusa_ai_chatbot/consolidation/persistence.py`: pass target-plan and
  origin metadata needed by the validator; keep lane gating through
  `validate_write_intent(...)`.
- `src/kazusa_ai_chatbot/consolidation/target.py`: add no broad new target
  kinds; only extend tests or helper access if the validator needs the
  validated user target id in a stable way.
- `src/kazusa_ai_chatbot/db/user_memory_units.py`: enforce canonical
  `source_refs` for new runtime writes; support source-ref appends on semantic
  updates; keep legacy-tolerant reads; validate optional provenance and
  commitment schedule fields structurally.
- `src/kazusa_ai_chatbot/db/schemas.py`: document the extended source-ref
  shape, provenance labels, and commitment schedule field.
- `src/kazusa_ai_chatbot/db/script_operations.py`: add read-only audit helpers
  and reviewed repair/apply helpers behind named maintenance functions.
- `src/kazusa_ai_chatbot/rag/user_memory_unit_retrieval.py`: preserve current
  public context shape; ensure archived/cancelled/completed rows stay excluded
  from active scoped context and no raw refs enter prompt-facing entries.
- `src/kazusa_ai_chatbot/rag/memory_evidence/workers/user_memory.py`: preserve
  result shape while keeping source refs out of `selected_summary`; preserve
  provenance labels for reviewer/debug payloads.
- `src/kazusa_ai_chatbot/rag/memory_evidence/projection.py`: ensure user-memory
  source labels remain compact and do not expose raw source refs as public
  prose.
- `src/kazusa_ai_chatbot/consolidation/README.md`: update the consolidation
  ICD only if the new validator contract changes the documented target/write
  boundary.
- `src/kazusa_ai_chatbot/db/README.md`: update the `user_memory_units`
  collection contract for source refs, provenance labels, commitment schedule
  semantics, and migration safety.
- `src/kazusa_ai_chatbot/rag/README.md` and
  `src/kazusa_ai_chatbot/rag/memory_evidence/README.md`: update only if
  provenance labels or prompt-facing source-ref redaction affect documented
  scoped evidence behavior.

### Create

- `src/scripts/audit_user_memory_units_lane.py`: read-only lane audit CLI.
- `src/scripts/repair_user_memory_units_lane.py`: dry-run/apply repair CLI,
  defaulting to dry-run and requiring an approved report for apply.
- `tests/test_user_memory_units_lane_integrity.py`: deterministic validator,
  audit, migration, and source-ref persistence tests.

### Keep

- Existing `user_memory_units` collection.
- Existing memory-evidence scoped retrieval result shape.
- Existing active/inactive status enum.
- Existing RAG and recall capability ownership boundaries.

## Overdesign Guardrail

- Actual problem: `user_memory_units` lacks provenance and contains
  wrong-lane, wrong-subject, lifecycle, timestamp, and volatile-fact defects
  that can be reused as scoped user continuity.
- Minimal change: canonicalize source refs, validate new writes, add focused
  audit/repair tooling, and migrate deterministic malformed legacy classes.
- Ownership boundaries: LLM extraction owns semantic memory meaning and scope
  labels; deterministic consolidation owns target proof, source refs,
  timestamp/lifecycle structure, and write permission; DB owns persistence
  mechanics; RAG owns retrieval/projection; migration scripts own data repair.
- Rejected complexity: no global truth adjudicator, no full history backfill,
  no new memory collection, no keyword-only user-text blocker, no alias bridge
  for new writes, no dual writes, no RAG router redesign, no added LLM call,
  and no automatic deletion of plausible legacy continuity.
- Evidence threshold: add richer provenance reconstruction or a separate
  adjudication stage only after dry-run samples prove that source refs from
  current origin metadata are insufficient for new writes or that a specific
  legacy class cannot be safely classified without source reconstruction.

## Agent Autonomy Boundaries

- The execution agent may implement validators, scripts, tests, and docs only
  inside the change surface after approval.
- The execution agent must not apply cleanup without an approved dry-run
  report and explicit user command.
- The execution agent must not delete plausible source-less legacy memory
  without explicit manual-review action.
- The execution agent must stop if target identity cannot be resolved.
- The execution agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, or extra features.
- The execution agent must treat changes outside the target modules as
  high-scrutiny changes and justify them against this plan before editing.
- If an equivalent helper already exists, reuse or move it into the correct
  owner instead of duplicating it.
- If implementation reveals that a contract in this plan is impossible, stop
  and update the plan or ask for approval before changing scope.

## Implementation Order

1. Parent establishes focused deterministic tests in
   `tests/test_user_memory_units_lane_integrity.py` for source-ref
   canonicalization, target proof, subject/durability scope rejection,
   commitment schedule validation, timestamp sanity, and no raw source refs in
   prompt-facing RAG summaries. Record expected failures.
2. Parent updates existing tests that currently allow empty `evidence_refs` so
   the expected failure proves the real bug.
3. Production-code subagent updates `memory_units.py`, `db/user_memory_units.py`,
   and schema docs for canonical `source_refs`, validator contract, and
   merge/evolve source-ref append.
4. Parent runs focused validator and persistence tests, then loops on failures
   before integration work.
5. Parent or production-code subagent updates consolidation persistence wiring
   and the minimal README/ICD surfaces needed by the changed contract.
6. Parent adds audit helpers in `db.script_operations` and the read-only audit
   CLI. Run unit tests proving the audit classifies fixture rows without
   writes.
7. Parent adds repair helpers and CLI dry-run/apply behavior. Run tests proving
   dry-run writes nothing, apply uses exact filters, and apply records previous
   values.
8. Parent runs RAG scoped retrieval tests proving archived malformed rows are
   excluded, legacy-unverified active rows remain scoped, and source refs are
   not rendered as public prose.
9. Parent runs prompt-render checks and one-at-a-time live LLM extractor cases;
   inspect each output and record judgment.
10. Parent runs a live DB audit dry-run only when MongoDB is available and the
    user has approved live DB inspection for execution.
11. Parent reviews dry-run samples by issue code and obtains explicit user
    approval before any apply command.
12. Parent applies approved cleanup only after approval, then reruns audit,
    focused tests, RAG tests, and migration consistency checks.
13. Parent starts independent code review after planned verification passes,
    remediates findings within scope, reruns affected checks, and records
    evidence before sign-off.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent establishes the focused test contract first and records the expected
  failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only; does
  not edit tests unless the parent explicitly directs it; closes after planned
  production code changes are complete, excluding review fixes.
- Parent may continue integration tests, migration tests, static checks, and
  validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - focused test contract established.
  - Covers: implementation steps 1-2.
  - Verify: focused tests fail for missing canonical source refs and current
    empty-ref fixtures.
  - Evidence: record commands and expected failures in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - new-write validator and source-ref persistence implemented.
  - Covers: implementation steps 3-5.
  - Verify: focused validator/persistence tests pass; prompt render check
    passes; `rg` confirms no new user-memory runtime `evidence_refs` contract.
  - Evidence: record changed files, static checks, and test output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - audit command implemented.
  - Covers: implementation step 6.
  - Verify: audit unit tests pass and dry-run fixture report has expected issue
    counts.
  - Evidence: record audit report path and no-write proof.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - repair dry-run/apply command implemented.
  - Covers: implementation step 7.
  - Verify: dry-run no-write tests, apply exact-filter tests, and rollback
    metadata tests pass.
  - Evidence: record dry-run/apply fixture outputs.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - RAG and recall impact verified.
  - Covers: implementation step 8.
  - Verify: scoped user-memory RAG tests and active-commitment reader/recall
    tests pass.
  - Evidence: record test output and any projection changes.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - live LLM and live DB gates completed when approved.
  - Covers: implementation steps 9-12.
  - Verify: live LLM cases run one at a time and inspected; live DB audit
    dry-run and apply run only with explicit approval.
  - Evidence: record trace files, judgment notes, audit report, apply report,
    and post-apply report.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 7 - independent code review completed.
  - Covers: implementation step 13.
  - Verify: review findings are resolved or explicitly accepted as residual
    risk; affected commands are rerun.
  - Evidence: record review findings, fixes, reruns, and approval status.
  - Handoff: plan can be signed off only after this stage.
  - Sign-off: `<agent/date>` after review evidence is recorded.

## Verification

### Static Checks

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\consolidation\memory_units.py src\kazusa_ai_chatbot\consolidation\persistence.py src\kazusa_ai_chatbot\db\user_memory_units.py src\kazusa_ai_chatbot\db\script_operations.py src\kazusa_ai_chatbot\rag\user_memory_unit_retrieval.py src\kazusa_ai_chatbot\rag\memory_evidence\workers\user_memory.py src\scripts\audit_user_memory_units_lane.py src\scripts\repair_user_memory_units_lane.py
rg -n "evidence_refs" src\kazusa_ai_chatbot\consolidation src\kazusa_ai_chatbot\db\user_memory_units.py tests\test_user_memory_units_rag_flow.py tests\test_user_memory_units_lane_integrity.py
git diff --check
```

Expected:

- `py_compile` exits 0.
- The `rg "evidence_refs"` check returns no matches in the listed
  user-memory runtime contract files. A nonzero `rg` exit because there are no
  matches is acceptable. Matches outside the listed paths, such as action-spec
  contracts, are not part of this gate.
- `git diff --check` exits 0.

### Deterministic Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_user_memory_units_lane_integrity.py -q
venv\Scripts\python.exe -m pytest tests/test_user_memory_units_rag_flow.py tests/test_user_memory_unit_lifecycle.py tests/test_user_memory_units_active_commitment_reader.py tests/test_user_memory_evidence_agent.py -q
venv\Scripts\python.exe -m pytest tests/test_consolidation_target_routing.py tests/test_memory_writer_information_flow_contracts.py tests/test_service_background_consolidation.py -q
```

Expected: all selected deterministic tests pass. If unrelated failures appear,
record them and stop for user direction unless they block this plan.

### Maintenance Script Tests

```powershell
venv\Scripts\python.exe -m scripts.audit_user_memory_units_lane --dry-run --fixture tests\fixtures\user_memory_units_lane_audit_fixture.json --output test_artifacts\user_memory_units_lane_audit_fixture.json
venv\Scripts\python.exe -m scripts.repair_user_memory_units_lane --dry-run --input test_artifacts\user_memory_units_lane_audit_fixture.json --output test_artifacts\user_memory_units_lane_repair_fixture_dry_run.json
```

Expected:

- audit fixture report contains every planned issue code;
- repair dry-run writes no DB data;
- output includes row ids, issue codes, recommended actions, previous values,
  and blocked/manual-review counts.

### Live DB Gates

Run only after explicit live DB inspection approval:

```powershell
venv\Scripts\python.exe -m scripts.audit_user_memory_units_lane --dry-run --output test_artifacts\user_memory_units_lane_audit.json
venv\Scripts\python.exe -m scripts.repair_user_memory_units_lane --dry-run --input test_artifacts\user_memory_units_lane_audit.json --output test_artifacts\user_memory_units_lane_repair_dry_run.json
```

Apply only after reviewed dry-run and explicit cleanup approval:

```powershell
venv\Scripts\python.exe -m scripts.repair_user_memory_units_lane --apply --input test_artifacts\user_memory_units_lane_repair_dry_run.json --output test_artifacts\user_memory_units_lane_repair_apply.json
venv\Scripts\python.exe -m scripts.audit_user_memory_units_lane --dry-run --output test_artifacts\user_memory_units_lane_post_apply_audit.json
```

Expected:

- dry-run produces no writes;
- apply refuses if the report has blockers, unknown issue codes, stale row
  filters, or missing previous values;
- post-apply audit shows source-less plausible rows marked legacy-unverified,
  deterministic malformed active rows removed from active retrieval, and no
  new timestamp anomalies.

### Live LLM Gates

Run one case at a time with `-q -s -m live_llm` and inspect each trace before
running the next:

```powershell
venv\Scripts\python.exe -m pytest tests/test_user_memory_units_live_llm.py::test_live_extractor_persists_valid_user_preference_with_source_refs -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests/test_user_memory_units_live_llm.py::test_live_extractor_persists_valid_user_milestone_with_source_refs -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests/test_user_memory_units_live_llm.py::test_live_extractor_persists_stable_pattern_with_source_refs -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests/test_user_memory_units_live_llm.py::test_live_extractor_rejects_wrong_subject_memory -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests/test_user_memory_units_live_llm.py::test_live_extractor_rejects_volatile_public_fact -q -s -m live_llm
venv\Scripts\python.exe -m pytest tests/test_user_memory_units_live_llm.py::test_live_extractor_classifies_no_due_ongoing_commitment -q -s -m live_llm
```

Expected: each trace shows parseable structured output, correct subject scope,
correct durability scope, correct commitment schedule semantics, and no raw
source refs in model-authored free text.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact;
- source refs are required for new non-diagnostic writes;
- `evidence_refs` is not preserved as a new-write compatibility path;
- legacy source-less rows are not mass-deleted;
- target validation fails closed and does not fabricate user ids;
- subject/durability validation does not keyword-filter user text;
- repair defaults to dry-run and apply uses exact stale-write filters;
- prompt-facing RAG evidence does not expose raw source refs or migration
  metadata;
- tests cover valid invented user data, wrong-lane rejection, source-less
  rejection, volatile facts, commitment lifecycle, and migration rollback.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Self Plan Review And Remediation

Review performed during draft refinement on 2026-07-02 by the assigned
planning agent without spawning additional subagents.

Requested-scope coverage review:

- Requested item: issue description. Result: covered by
  `Issue Description Based On Deep Analysis`; no remediation needed after
  review.
- Requested item: plan to remove malformed data. Result: covered by
  `Plan To Remove Malformed Data`, `Data Migration`, and apply verification;
  no remediation needed after review.
- Requested item: RCA. Result: expanded to include `evidence_refs` /
  `source_refs` mismatch, empty-ref validation gap, merge/evolve provenance
  loss, target-subject split, commitment ambiguity, timestamp gaps, RAG scoped
  impact, and test blind spots.
- Requested item: plan to harden corrupted data source. Result: covered by
  validator contract, canonical source refs, target proof, subject/durability
  scopes, commitment schedule class, no keyword-only filtering rule, and
  source-ref append on merge/evolve.
- Requested item: plan to prove robustness. Result: covered by static checks,
  deterministic tests, maintenance fixture tests, live DB gates, one-at-a-time
  live LLM gates, and independent code review.
- Requested failure-mode list. Result: missing source refs, target validation,
  wrong-subject writes, commitment lifecycle, timestamp anomalies, volatile
  public facts, RAG scoped retrieval impact, and legacy source-less data risk
  are all named in Context, RCA, contracts, verification, and risks.

Repo plan-contract review:

- Status remains `draft`, so this document is not executable authorization.
- Required top matter is present and uses plan class `high_risk_migration`.
- Mandatory sections required by the plan contract are present.
- Optional sections relevant to data migration, contracts, LLM budget, risks,
  execution evidence, and self review are present.
- High-risk migration length is within the documented maximum line budget.
- Plan-contract red-flag placeholder scan returned no matches in executable
  instruction text.
- The plan preserves project ownership boundaries: RAG retrieves evidence,
  LLM stages judge semantic memory scope, deterministic code validates and
  persists, DB owns storage mechanics, and migration scripts own reviewed data
  repair.

Findings and remediations:

- Finding: The original draft named missing `source_refs` but did not identify
  the concrete `evidence_refs` output versus `source_refs` persistence
  mismatch. Remediation: RCA, Must Do, Cutover Policy, Contracts, Change
  Surface, Verification, and Risks now make the canonical field mismatch a
  first-class bug.
- Finding: The original draft said target validation was needed but did not
  separate real-user target proof from semantic subject validation.
  Remediation: Contracts now add `subject_scope` and `durability_scope`, with
  deterministic validation of LLM-owned semantic labels and explicit
  no-keyword-filter rules.
- Finding: The original draft treated no-due active commitments too broadly.
  Remediation: Contracts now require `commitment_schedule_kind` so dated
  commitments need `due_at`, while ongoing accepted rules remain representable
  without being scheduler due items.
- Finding: The original draft described dry-run/apply behavior but did not
  define audit classifications, stale-write safety, previous-value recording,
  or rollback evidence. Remediation: Audit Finding Contract, Migration
  Metadata, Data Migration, and Verification now define issue codes, actions,
  exact-filter apply, report requirements, and rollback source.
- Finding: The original draft did not spell out RAG scoped retrieval impact.
  Remediation: Context, Target State, Change Surface, Verification, and Risks
  now state how active malformed rows affect scoped evidence and later
  merge/evolve decisions.
- Finding: Plan-contract review required explicit cutover enforcement, progress
  sign-off rules, and independent review gates. Remediation: those sections
  were expanded while keeping status `draft`.

Residual risks:

- Legacy source-less plausible rows remain unverifiable without a separate
  source-reconstruction plan.
- The LLM can still misclassify subject or durability scope; live LLM tests and
  deterministic structural rejection reduce but do not eliminate this risk.
- Parallel lane-integrity plans may touch shared maintenance helpers or docs;
  execution must reconcile diffs before approval.
- Applying migration to live data remains high risk and requires a separate
  explicit approval after dry-run sample review.

## Acceptance Criteria

This plan is complete when:

- New user-memory writes without canonical source refs are rejected.
- New user-memory writes without a real validated user target are rejected.
- New user-memory writes about the character, another user, a group/channel,
  an episode-local artifact, or volatile public/current facts are rejected
  before persistence.
- The extractor/runtime contract no longer drops evidence because of the
  `evidence_refs` / `source_refs` mismatch.
- Merge/evolve updates append current source refs and preserve merge history.
- Active scheduled commitments without due dates are rejected, repaired,
  downgraded, archived, or reported for manual review; ongoing accepted rules
  have an explicit non-scheduled class.
- Future timestamp anomalies are repaired or archived.
- Plausible source-less legacy rows remain as legacy-unverified unless
  deterministic corruption is proven.
- Scoped RAG retrieval excludes archived malformed rows and does not expose raw
  refs or migration metadata as prompt-facing prose.
- Verification commands pass, live LLM cases are inspected one by one, and any
  live DB apply evidence is recorded.

## Data Migration

1. Export all `user_memory_units` rows through `db.script_operations` before
   audit or repair.
2. Run read-only audit and write a report with audit run id, row count,
   collection count, issue-code counts, per-row findings, and sample rows for
   every issue code.
3. Block apply if any row has unknown issue code, malformed row id, missing
   previous values, stale audit metadata, or manual-review-required action.
4. Generate dry-run repair report from the audit report. Dry-run must perform
   no writes and must include exact planned update filters for each row.
5. Review samples for every issue code and obtain explicit user approval for
   apply.
6. Apply approved deterministic repairs with exact filters containing
   `unit_id`, current `status`, current `unit_type`, and current `updated_at`
   when present. Stale rows are skipped and reported, not overwritten.
7. Record previous values and migration metadata in each changed row.
8. Re-run audit after apply and compare before/after counts.
9. Rollback uses the pre-apply export plus apply report. Status-only changes
   must record previous status, previous unit type, and previous timestamp
   fields.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Valid old memory removed from live continuity | Mark plausible source-less rows `legacy_unverified` instead of deleting | Dry-run samples and post-apply audit |
| New useful memory rejected | Keep LLM semantic scope decision, validate only structure, and run live LLM cases | Live LLM traces and deterministic validator tests |
| Source refs leak into prompts | Keep refs out of prompt-facing prose; preserve only trace/debug metadata | RAG projection tests and static review |
| Ongoing rules misclassified as stale commitments | Add explicit non-scheduled commitment class | Commitment validator and live LLM no-due case |
| Merge/evolve keeps source-less rows source-less | Append current source refs on semantic updates | Merge/evolve tests |
| Migration overwrites changed live rows | Exact filters and stale-row skip reporting | Apply fixture tests and live post-audit |
| Parallel active plans touch shared helpers | Reconcile git status and diffs before execution | Pre-edit `git status --short` and review gate |

## Execution Evidence

Cleanup-only execution completed on 2026-07-03. New-write hardening remains
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

Commands and artifacts:

- `venv\Scripts\python.exe -m py_compile src\scripts\_lane_cleanup.py src\scripts\audit_user_memory_units_lane.py src\scripts\repair_user_memory_units_lane.py src\kazusa_ai_chatbot\db\script_operations.py`
- `venv\Scripts\python.exe -m scripts.audit_user_memory_units_lane --dry-run --output test_artifacts\user_memory_units_lane_audit.json`
- `venv\Scripts\python.exe -m scripts.repair_user_memory_units_lane --dry-run --input test_artifacts\user_memory_units_lane_audit.json --output test_artifacts\user_memory_units_lane_repair_dry_run.json`
- `venv\Scripts\python.exe -m scripts.repair_user_memory_units_lane --apply --input test_artifacts\user_memory_units_lane_repair_dry_run.json --output test_artifacts\user_memory_units_lane_repair_apply.json`
- `venv\Scripts\python.exe -m scripts.audit_user_memory_units_lane --dry-run --output test_artifacts\user_memory_units_lane_post_audit.json`

Results:

- Baseline: 831 total rows, 664 active rows, 980 findings, 831 deterministic
  planned actions.
- Apply: 830 rows marked `legacy_unverified_continuity`, 1 orphan active row
  archived, 0 blocked actions.
- Post-audit: 0 deterministic planned actions remain. Residual findings are
  830 `legacy_source_refs_acknowledged`, 149
  `active_commitment_missing_due_manual_review`, and 1
  `archived_target_profile_missing`.
- Closure note: the 149 no-due active-commitment findings require semantic
  review or the consolidator router hardening plan; they were not mutated by
  deterministic cleanup because valid ongoing rules can be source-less legacy
  continuity.
