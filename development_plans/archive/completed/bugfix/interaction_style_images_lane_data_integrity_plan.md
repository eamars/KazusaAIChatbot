# interaction style images lane data integrity plan

## Summary

- Goal: repair and harden `interaction_style_images` so learned user and
  group/channel style overlays remain reusable interaction guidance, not event
  facts, commitments, volatile public facts, reflection noise, or no-op
  revision churn.
- Plan class: high_risk_migration.
- Status: completed.
- Mandatory skills: `development-plan`, `database-data-pull`,
  `local-llm-architecture`, `debug-llm`, `py-style`, `cjk-safety`,
  `test-style-and-execution`.
- Overall cutover strategy: compatible repair for existing rows; bigbang
  validation and churn control for new writes.
- Highest-risk areas: erasing useful learned style, preserving fact-like
  pollution as style, cross-scope user/group leakage, silent source-lineage
  gaps, and continuing revision churn after cleanup.
- Acceptance criteria: malformed current overlays are audited and repaired only
  through reviewed dry-run/apply gates; new active overlays require valid
  scope, source evidence, style-only content, and meaningful diff; event facts
  and commitments are rejected from this lane.

New-write hardening supersession: all lane-specific new-write hardening,
source-generation proof, and fixing-strategy instructions in this plan are
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.
Execute this lane plan for cleanup, audit, dry-run/apply, data migration, and
post-cleanup verification only. If another section still names new-write
validators, prompts, or tests, treat that text as historical context rather
than execution scope.

## Context

The 2026-07-02 audit found 37 active `interaction_style_images` rows:

- 20 user-scoped rows;
- 17 group/channel-scoped rows;
- maximum revision 724.

The lane is structurally present and mostly plausible, but a maximum revision
of 724 is inconsistent with a compact current-state style overlay. The direct
source inspection for this plan found the active ownership boundary:

```text
reflection_cycle.interaction_style
  -> db.interaction_style_images.upsert_user_style_image(...)
  -> db.interaction_style_images.upsert_group_channel_style_image(...)

consolidation.persistence
  -> consolidation.group_channel.persist_group_channel_style_image(...)
  -> db.interaction_style_images.upsert_group_channel_style_image(...)

normal chat consumers
  -> db.interaction_style_images.build_interaction_style_context(...)
  -> L3 style/preference prompts
  -> db.interaction_style_images.build_group_engagement_action_context(...)
  -> L2d group engagement context
  -> db.interaction_style_images.build_user_engagement_relevance_context(...)
  -> relevance engagement context
```

`src/kazusa_ai_chatbot/consolidation/images.py` is the character self-image
lane, not this lane. The hardening target is the shared interaction-style DB
helper plus the reflection/consolidation sources that call it.

Project basis: learned interaction style is good data when it remains scoped
as style. A row saying to answer a specific user with concise warmth, let them
lead pacing, or avoid over-explaining in a specific group is useful. A row that
stores what happened, what someone promised, what topic was discussed, what is
currently true in the world, or what task should be done is bad data in this
lane even when it was learned from real conversation.

## Lane Analysis Requirements

### Issue Description Based On Deep Analysis

`interaction_style_images` is a durable current-state style overlay keyed by
either one `global_user_id` or one `(platform, platform_channel_id)` group
scope. It is not `user_memory_units`, not RAG evidence, not reflection
promotion, not calendar state, and not dialog text. It should store only
abstract interaction handling guidance:

- speech style, directness, teasing/warmth, clarification shape;
- social handling, conversational distance, group atmosphere;
- pacing and engagement guidance that can be reused across future turns.

The issue is data integrity, not absence of the feature. Current validators
reject obvious dates, IDs, and quote-heavy source examples, but the lane still
has plausible failure paths:

- active writes can carry empty or weak source evidence;
- current documents replace `source_reflection_run_ids`, so duplicate
  suppression can forget older applied daily runs;
- all current upserts increment `revision` even when the normalized overlay is
  semantically unchanged;
- daily reflection signal fields and group evidence rows can contain noisy
  event detail that the extractor may copy into "style";
- malformed group/user scope fields can let a valid-looking overlay influence
  the wrong consumer.

### Plan To Remove Malformed Data

1. Export the full `interaction_style_images` collection to a timestamped
   backup artifact before any apply operation.
2. Run a read-only lane audit that classifies each row by target scope,
   source evidence, overlay validity, content lane, revision churn, and
   downstream risk.
3. Generate a dry-run repair manifest. The manifest may recommend `keep`,
   `rewrite_style_only`, `disable_runtime_use`, `repair_scope_metadata`, or
   `manual_review`; it must not mutate data.
4. Review the dry-run manifest before apply. Useful learned style stays unless
   the row is malformed by scope or content lane.
5. Apply only an approved manifest. Apply defaults must be non-destructive:
   disabling a row sets `status="disabled"` so runtime projection returns an
   empty overlay while the original overlay remains recoverable from the
   backup and apply report.
6. Rewrite only rows with an explicitly approved style-only replacement
   overlay in the repair manifest.
7. Re-run the audit after apply and compare counts. No unresolved malformed
   active rows may remain.

High-revision rows are not reset by default. Revision 724 is evidence for RCA
and future hardening; lowering revision without a historical ledger would hide
diagnostic history. Repair acts on current effective content and stops future
churn through write gates.

### RCA Of The Failure Mode

The likely root cause is a combination of weak lane validation and current
document replacement semantics:

- `upsert_user_style_image(...)` and `upsert_group_channel_style_image(...)`
  always increment `revision` when a document exists, even if the normalized
  overlay is identical to the current overlay.
- The same upserts replace `source_reflection_run_ids` with the latest source
  list. `_style_doc_contains_source_run(...)` checks only the current list, so
  an older daily run can be forgotten after later writes and become eligible
  for repeated processing.
- Active writes accept candidate overlays without requiring non-empty source
  evidence at the DB boundary.
- The style extractor prompt says to produce abstract guidance, but the
  deterministic validator mostly catches structural leakage markers. It does
  not fully distinguish reusable style from event facts, active commitments,
  or volatile current/public facts.
- Daily reflection quality signals can carry source-noise. Group participant
  evidence rows intentionally include compact source text as style examples;
  this is useful but creates a copy-through risk when the LLM is weak or the
  source row is event-heavy.
- User and group scope integrity is split across callers and storage helpers.
  If a row's `style_image_id`, `scope_type`, `global_user_id`, `platform`, and
  `platform_channel_id` disagree, the wrong runtime consumer can receive the
  row or a malformed row can stay silently active.

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

## Failure Modes And RCA Detail

| Failure mode | Observable symptom | Likely root cause | Downstream impact |
| --- | --- | --- | --- |
| Revision churn | High revisions such as 724 on a current-state row | Upserts always increment revision; equivalent overlays are not diff-gated; old source refs are replaced | Operator trust drops; repeated background work and LLM cost; hard to tell real style evolution from noise |
| Event-fact contamination | Guideline says what happened, who did what, or what topic occurred | Reflection output or evidence rows copied into overlay; validator catches markers but not all fact semantics | L3 may treat stale event memory as current style or reopen old topics |
| Commitment contamination | Guideline says to remind, follow up, deliver, wait for, or complete a promise | Accepted tasks/calendar/user commitments leak through reflection-style extraction | Style prompt can pressure dialog or relevance as if a durable promise exists outside `user_memory_units` and scheduler |
| User/group scope confusion | User row has group fields, group row has user id, or `style_image_id` disagrees with scope fields | Scope validation split across callers; stored document replacement trusts caller arguments | One user's style can affect another user, or group atmosphere can affect private chat |
| Reflection noise | Daily quality patterns or synthesis limitations contain event recaps, low-confidence speculation, or raw reflection conclusions | Style extractor consumes abstract fields that are not strictly style-only after noisy reflection | Overlays become mini reflection summaries rather than interaction guidance |
| Semantic no-op updates | Same overlay text appears across revisions with only source ids or timestamps changed | No normalized comparison before persistence | Churn continues even after content cleanup; duplicate suppression remains weak |
| Source evidence gaps | Active row has no source refs or only malformed source refs | Upsert helpers normalize lists but do not require evidence for active writes | Row cannot be audited back to a daily/reflection source; repair confidence is lower |
| Volatile public/current fact contamination | Guideline stores external current facts, dates, schedules, model status, prices, news, or one-time public events | Extractor treats "what was discussed" as style signal | Future dialog can sound anchored to outdated public facts |
| Quote/example leakage | Guideline contains direct quoted source text or a prompt-like example | Weak abstraction or insufficient quote filtering | Private content can reach L3/relevance consumers as reusable style |
| Downstream behavior distortion | Relevance, L2d engagement, or L3 style changes for the wrong reason | Style image consumers receive compact overlay as soft guidance and may not know it is polluted | Character may speak too often, avoid needed topics, reopen stale facts, or overfit to a group mood |

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `database-data-pull`: load before exporting or inspecting live
  `interaction_style_images` rows.
- `local-llm-architecture`: load before changing extractor prompts, LLM
  payloads, style-source contracts, or downstream prompt consumers.
- `debug-llm`: load before live or debug style-generation checks.
- `py-style`: load before editing Python production or test files.
- `cjk-safety`: load before editing Python files that contain CJK prompt or
  style strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute this plan while status is `draft`.
- Do not apply live cleanup without a reviewed dry-run report and explicit
  user command.
- Before production-code edits, run `git status --short`, read `README.md`,
  `docs/HOWTO.md`, `development_plans/README.md`, this plan, and relevant
  subsystem READMEs.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual edits.
- Do not read `.env`.
- Learned interaction style is valid data when scoped as reusable style.
- Event facts, commitments, volatile public/current facts, user-memory facts,
  calendar state, and reflection-run conclusions are invalid in this lane.
- Do not delete, disable, or rewrite a useful style overlay merely because it
  was invented or learned through conversation.
- New active revisions must have valid user or group/channel target scope,
  source evidence, style-only content, and meaningful diff.
- DB helpers own storage mechanics, scope validation, source evidence
  validation, revision updates, and duplicate-source merging.
- Reflection/consolidation code owns source eligibility and prompt-safe style
  payload construction; it must not access raw MongoDB directly.
- LLM stages may propose style guidance, but deterministic code validates
  structure, lane eligibility, scope, evidence, and diff before persistence.
- Do not add RAG retrieval, vector search, lexical search, or user-memory
  writes for style images.
- Do not add a live response-path LLM call.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's Independent Code Review gate and record the
  result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution
  unless the user explicitly approves fallback execution.

## Must Do

- Add a style-lane audit for target scope, event facts, commitments, volatile
  facts, missing source evidence, reflection noise, semantic no-op risk, and
  high revision churn.
- Add dry-run/apply repair for malformed active overlays, unresolved targets,
  source gaps, and scope mismatches.
- Add a source-bound backup and repair manifest contract before any apply.
- Add new-write validation for style-only content, target kind, source
  evidence, and target-field consistency.
- Add semantic diff/churn control so equivalent updates do not increment
  revision.
- Merge bounded source refs on no-op/equivalent updates so duplicate
  suppression does not forget already-applied daily runs.
- Harden reflection style signal projection so obvious source-detail noise is
  not treated as style signal.
- Keep group-channel consolidation writes on the same DB validation path.
- Add focused validation tests proving valid learned style guidance is accepted
  and wrong-lane data is rejected.
- Add verification proving RAG, `user_memory_units`, dialog state, and
  promoted reflection context remain uninvolved.

## Deferred

- Do not redesign the `interaction_style_images` schema.
- Do not add an interaction-style history collection.
- Do not rebuild all overlays from historical reflection data.
- Do not lower high revision numbers by default.
- Do not change affinity scoring.
- Do not move user facts, promises, or commitments into this lane.
- Do not change dialog wording style directly.
- Do not change adapters, message-envelope parsing, or platform delivery.
- Do not add a compatibility shim, alternate storage helper, or fallback
  mapper for old call shapes.

## Cutover Policy

Overall strategy: compatible cleanup for stored rows; bigbang validation for
new writes.

| Area | Policy | Instruction |
| --- | --- | --- |
| Existing overlays | compatible | Keep effective overlays unless audit proves malformed target, wrong content lane, missing evidence, or runtime risk. |
| Existing malformed rows | migration | Export, classify, dry-run, review, then apply only approved row-level repairs. |
| New active writes | bigbang | Require valid target, source evidence, style-only content, and meaningful diff through the shared DB helpers. |
| Revision churn | bigbang | Equivalent updates must not increment revision; source refs may merge without prompt-facing style change. |
| Reflection extraction | compatible | Preserve current background cadence and LLM count shape except for skipping or rejecting invalid style inputs. |
| Consolidation group-channel writes | compatible | Preserve the existing persistence route but rely on the shared hardened DB boundary. |
| RAG/cognition memory | bigbang no-exposure | Do not add style images to RAG, `user_memory_units`, promoted reflection context, or dialog state. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, rewrite the existing write path directly instead of
  preserving a fallback to the old permissive behavior.
- If an area is `migration`, follow the exact dry-run/apply phases and cleanup
  gates in this plan.
- If an area is `compatible`, preserve only the compatibility surfaces listed
  in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

`interaction_style_images` stores:

- user-specific interaction style for one validated `global_user_id`;
- group/channel atmosphere and interaction guidance for one platform channel;
- stable learned social texture;
- source-bound current overlays with meaningful revisions.

It does not store:

- event logs;
- direct conversation quotes;
- commitments, tasks, or future promises;
- volatile public/current facts;
- character-global mood or self-image;
- ordinary user facts that belong in `user_memory_units`;
- raw reflection output or promoted reflection memory.

Target write path:

```text
style source candidate
  -> prompt-safe style signal projection
  -> optional background style extractor
  -> validate_interaction_style_write(...)
       target scope
       source evidence
       style-only content
       normalized diff
       bounded source-ref merge
  -> upsert current style document
```

Runtime read path remains deterministic and prompt-safe:

```text
db.interaction_style_images
  -> user_style / group_channel_style compact overlays
  -> relevance engagement context, L2d group engagement context, and L3 style
```

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Valid learned style | Preserve | Social adaptation is a project goal. |
| Event facts | Reject from style lane | Conversation history, reflection audit, and user memory own facts. |
| Commitments | Reject from style lane | `user_memory_units`, accepted-task lifecycle, and calendar scheduler own commitments. |
| Source evidence | Require for active writes | Style overlays must be auditable to reflection/consolidation source evidence. |
| No-op churn | Gate by normalized diff | Revision count should mean prompt-facing style changed. |
| Source refs | Merge bounded refs on equivalent writes | Duplicate suppression needs lineage without creating fake revisions. |
| Existing high-revision rows | Audit first; do not reset revision by default | Avoid hiding history and losing effective state. |
| Scope confusion | Validate `style_image_id` against scope fields | Prevent user/group cross-lane contamination. |
| Repair default | Disable rather than delete | Runtime safety improves while original content remains recoverable from backup/report. |

## Contracts And Data Shapes

### Audit Finding

```python
class InteractionStyleImageAuditFinding(TypedDict):
    style_image_id: str
    scope_kind: Literal["user", "group_channel", "invalid"]
    scope_ref: str
    revision: int
    status: str
    issue_code: Literal[
        "valid_style",
        "scope_shape_invalid",
        "scope_id_mismatch",
        "missing_source_evidence",
        "event_fact_contamination",
        "commitment_contamination",
        "volatile_fact_contamination",
        "reflection_noise_leak",
        "quote_or_example_leak",
        "semantic_noop_churn_risk",
        "high_revision_churn",
        "status_overlay_mismatch",
        "manual_review_required",
    ]
    severity: Literal["info", "warning", "error", "blocked"]
    recommended_action: Literal[
        "keep",
        "rewrite_style_only",
        "disable_runtime_use",
        "repair_scope_metadata",
        "manual_review",
    ]
    evidence_fields: dict[str, object]
```

`evidence_fields` may include source ids, counts, status, scope fields,
normalized overlay hashes, and short overlay snippets for operator review. It
must not include raw conversation history rows.

### Repair Manifest

```python
class InteractionStyleImageRepairAction(TypedDict):
    style_image_id: str
    expected_current_hash: str
    action: Literal[
        "keep",
        "rewrite_style_only",
        "disable_runtime_use",
        "repair_scope_metadata",
    ]
    approved_overlay: NotRequired[InteractionStyleOverlayDoc]
    approved_scope_fields: NotRequired[dict[str, str]]
    reason: str
```

Apply must skip a row if `expected_current_hash` does not match the stored row
at apply time.

### Active Write Eligibility

New active write eligibility:

- caller uses `upsert_user_style_image(...)` or
  `upsert_group_channel_style_image(...)`;
- target kind is exactly `user` or `group_channel`;
- target fields match the deterministic `style_image_id`;
- active overlay contains interaction guidance;
- source refs are non-empty, normalized, and bounded;
- proposed overlay passes the style-only validator;
- normalized proposed overlay differs meaningfully from current overlay before
  revision increments.

Equivalent writes:

- must not increment `revision`;
- must not alter prompt-facing `overlay` or `status`;
- may merge bounded source refs so the same daily run is not repeatedly
  reprocessed.

### Style-Only Validator Contract

`validate_interaction_style_overlay(...)` remains the structural overlay
normalizer. Add a write-level validator such as
`validate_interaction_style_write(...)` in
`src/kazusa_ai_chatbot/db/interaction_style_images.py` to enforce source,
scope, lane, and diff rules.

The validator must reject:

- exact dates, times, run ids, message ids, platform ids, URLs, and long opaque
  identifiers;
- direct quote examples or source excerpts;
- statements that summarize a concrete episode;
- future promises, reminders, accepted tasks, due dates, or obligation wording;
- volatile public facts such as current prices, news, model availability,
  schedules, releases, or "today" facts;
- ordinary user facts, preferences, milestones, and relationship facts that
  belong in `user_memory_units`;
- group/channel facts stored in user scope or user facts stored in group scope.

The validator must preserve style guidance such as:

- concise/warm/direct speech style;
- preference for fewer clarifying questions;
- group atmosphere and timing guidance;
- soft teasing, reassurance, challenge level, pacing, and engagement shape.

## LLM Call And Context Budget

Affected path: background reflection style extraction and existing
group-channel consolidation style writes.

- Live chat path: no new LLM calls.
- Background reflection path: no new mandatory LLM calls beyond the existing
  style extractor calls. Invalid source inputs should skip before extraction
  when possible.
- New prompt changes, if needed, must use short semantic lane descriptions and
  must not add operational routing, DB schema, source ids, or repair policy to
  model-facing text.
- Deterministic code owns target validation, source evidence, revision
  decisions, and migration apply behavior.
- LLM output should propose style guidance only. It must not propose revision
  operations, target ids, repair actions, or persistence status.
- Context budget for any extractor call remains below the existing
  consolidation route budget and below 8,000 dynamic payload characters for
  style source payloads.

Smallest semantic question for the LLM:

```text
Given approved style signals for one user or group, what reusable interaction
guidance should the active character carry forward?
```

Rejected LLM responsibilities:

- deciding whether a target is valid;
- deciding whether source refs are sufficient;
- deciding whether a proposed overlay should increment revision;
- repairing live DB rows;
- classifying user commitments or event facts into other storage lanes.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/db/interaction_style_images.py`
  - Add write-level validation for scope, source evidence, style-only content,
    normalized diff, and bounded source-ref merge.
  - Update user and group upsert helpers so equivalent overlays do not
    increment revision.
  - Keep runtime read projections shape-compatible.
- `src/kazusa_ai_chatbot/reflection_cycle/interaction_style.py`
  - Use hardened DB upsert outcomes.
  - Skip or log noisy/invalid style source inputs before persistence.
  - Preserve dry-run behavior and existing background-only execution.
- `src/kazusa_ai_chatbot/reflection_cycle/interaction_style_sources.py`
  - Keep deterministic group participant source construction; any update must
    be limited to preventing source-noise copy-through and must be covered by
    focused tests.
- `src/kazusa_ai_chatbot/consolidation/group_channel.py`
  - Continue routing group-channel style writes through the hardened DB helper.
  - Do not create a separate validator.
- `src/kazusa_ai_chatbot/consolidation/persistence.py`
  - Tighten `group_channel_style_image` payload acceptance only if needed to
    require source refs before DB persistence.
- `src/kazusa_ai_chatbot/db/script_operations.py`
  - Add maintenance-only audit, export, dry-run, and apply helpers for this
    lane.

### Create

- `src/scripts/audit_interaction_style_images_lane.py`
  - Read-only by default; writes only report artifacts.
- `src/scripts/repair_interaction_style_images_lane.py`
  - Requires `--dry-run` or `--apply`; apply requires a reviewed manifest and
    backup path.
- `tests/test_interaction_style_images_lane_integrity.py`
  - Focused validator, diff, source-evidence, and repair-contract tests.
- `tests/test_interaction_style_images_lane_scripts.py`
  - Script helper tests using fake DB/script-operation boundaries if needed.

### Keep

- Existing `interaction_style_images` collection name and public runtime read
  shapes.
- Existing L3/relevance/L2d prompt-facing style context shape.
- RAG and user-memory-unit contracts.
- `src/kazusa_ai_chatbot/consolidation/images.py`; it is character
  self-image and not part of this lane.
- `src/kazusa_ai_chatbot/consolidation/target.py` unless focused tests prove
  the current target lane contract itself is wrong.

## Overdesign Guardrail

- Actual problem: the interaction-style lane can accumulate wrong-lane content
  and revision churn because active writes are not fully source-, scope-, and
  diff-gated.
- Minimal change: harden the shared DB write boundary, add a read-only audit,
  add reviewed dry-run/apply repair, and keep runtime read shapes unchanged.
- Ownership boundaries: LLM stages propose style text; deterministic
  reflection/consolidation code selects source eligibility; DB helpers
  validate scope, evidence, style-only content, source-ref merge, and revision
  mechanics; maintenance scripts orchestrate approved repair.
- Rejected complexity: no new style schema, no history collection, no full
  overlay regeneration, no RAG route, no response-path style model, no
  compatibility shim, no keyword-only truth filter as the sole defense, and no
  adapter/platform behavior change.
- Evidence threshold: schema redesign or a history ledger requires proof that
  current single-document overlays cannot be safely repaired with source,
  scope, content, and diff validation.

## Agent Autonomy Boundaries

- The execution agent may implement validators, audit scripts, repair scripts,
  and tests after approval.
- The execution agent must not rewrite existing overlays without reviewed
  dry-run output and explicit apply approval.
- The execution agent must preserve learned/invented style when the lane is
  correct.
- The execution agent must stop if target scope cannot be resolved safely.
- The execution agent must not add new architecture, alternate migration
  strategies, compatibility layers, fallback paths, extra features, prompt
  retries, or helper agents.
- The execution agent must search for existing helpers before adding new ones.
- The execution agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, broad prompt rewrites, or registry edits unless
  separately instructed.
- If this plan and source code disagree, preserve the plan's intent and report
  the discrepancy before expanding scope.

## Implementation Order

1. Parent adds focused DB helper tests in
   `tests/test_interaction_style_images_lane_integrity.py`:
   `test_active_user_write_requires_source_evidence`,
   `test_active_group_write_requires_scope_consistency`,
   `test_style_only_validator_rejects_event_fact`,
   `test_style_only_validator_rejects_commitment`,
   `test_style_only_validator_rejects_volatile_current_fact`,
   `test_equivalent_overlay_does_not_increment_revision`, and
   `test_equivalent_overlay_merges_source_refs_without_overlay_change`.
2. Parent runs those focused tests and records expected failures or current
   behavior in `Execution Evidence`.
3. Parent adds audit/repair helper tests for finding classifications and
   dry-run/apply manifest safety.
4. Production-code subagent implements the DB write-level validator, diff
   gate, and bounded source-ref merge in `db/interaction_style_images.py`.
5. Parent reruns focused DB helper tests and records results.
6. Production-code subagent updates reflection/consolidation callers only as
   needed to handle the hardened upsert result and source-evidence failures.
7. Parent implements or completes maintenance script wrappers on top of
   `db.script_operations` helpers.
8. Parent runs script helper tests and focused reflection tests.
9. Parent runs static boundary checks for RAG/user-memory non-exposure and raw
   DB script boundaries.
10. Parent generates a dry-run audit report if live DB access is explicitly
    approved for this plan execution. Do not apply cleanup by default.
11. Parent runs full verification commands.
12. Parent runs Independent Code Review, remediates approved-scope findings,
    reruns affected verification, and records evidence.

## Execution Model

- Parent owns tests, orchestration, verification, evidence, lifecycle updates,
  review feedback remediation, and final sign-off.
- Parent establishes the focused test contract first and records expected
  failures or baseline behavior before production implementation starts.
- Use exactly one production-code subagent for production edits and one
  independent code-review subagent after verification.
- Production-code subagent owns approved production changes only and must not
  edit tests unless parent explicitly directs it.
- Parent may continue integration tests, script tests, static checks, and
  evidence collection while the production-code subagent edits production
  code.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - focused integrity test contract established.
  - Covers: implementation steps 1-3.
  - Verify: focused tests are added and expected failures or baseline behavior
    are recorded.
  - Evidence: record command output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - DB write boundary hardened.
  - Covers: implementation steps 4-5.
  - Verify: focused DB helper tests pass.
  - Evidence: changed files and test output recorded.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - source callers and maintenance scripts complete.
  - Covers: implementation steps 6-8.
  - Verify: focused reflection tests and script helper tests pass.
  - Evidence: command output and any dry-run limitations recorded.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - audit dry-run and full verification complete.
  - Covers: implementation steps 9-11.
  - Verify: all `Verification` commands pass or are recorded as explicitly
    blocked.
  - Evidence: audit report path, static checks, and test outputs recorded.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - Independent Code Review complete.
  - Covers: implementation step 12.
  - Verify: review findings are recorded, approved-scope fixes are applied,
    affected checks are rerun, and unresolved risks are documented.
  - Evidence: review identity/harness role, findings, fixes, reruns, and final
    approval status recorded.
  - Handoff: plan can be signed off only after this stage.
  - Sign-off: `<agent/date>` after final evidence is recorded.

## Verification

Run from the repository root with the project venv.

### Focused Tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_interaction_style_images_lane_integrity.py -q
venv\Scripts\python.exe -m pytest tests\test_interaction_style_images_lane_scripts.py -q
venv\Scripts\python.exe -m pytest tests\test_interaction_style_images.py tests\test_reflection_interaction_style.py -q
```

### Regression Tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_memory_writer_information_flow_contracts.py tests\test_rag_projection.py tests\test_cognition_interaction_style_context.py tests\test_persona_relevance_agent.py -q
```

### Static Boundary Checks

```powershell
rg -n "interaction_style|style_image" src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag* tests
rg -n "user_memory_units|active_commitment|future_promises|scheduled_events|calendar" src\kazusa_ai_chatbot\db\interaction_style_images.py src\kazusa_ai_chatbot\reflection_cycle\interaction_style*.py
rg -n "get_db|db\\.[a-zA-Z_]+\\.(find|find_one|aggregate|insert|insert_one|insert_many|update|update_one|update_many|replace_one|delete|delete_one|delete_many)" src\scripts --glob "!src\scripts\audit_interaction_style_images_lane.py" --glob "!src\scripts\repair_interaction_style_images_lane.py"
```

Expected results:

- RAG check: no production RAG module imports, retrieves, or projects
  interaction style images. Test matches are allowed only for non-exposure
  assertions.
- Commitment/fact-lane check: any matches in style modules must be validator
  rejection rules or tests, not persistence or projection of those lanes into
  style images.
- Raw DB check: maintenance scripts must not use raw DB operations directly;
  the two new scripts must call `kazusa_ai_chatbot.db.script_operations`.

### Audit And Repair Dry Run

```powershell
venv\Scripts\python.exe -m scripts.audit_interaction_style_images_lane --output test_artifacts\interaction_style_images_lane_audit.json
venv\Scripts\python.exe -m scripts.repair_interaction_style_images_lane --dry-run --manifest test_artifacts\interaction_style_images_lane_repair_manifest.json --output test_artifacts\interaction_style_images_lane_repair_dry_run.json
```

Run these only when live DB inspection is explicitly allowed for the execution
session. Do not run `--apply` unless the user explicitly approves the reviewed
dry-run manifest.

### Robustness Cases

- valid invented user interaction style is accepted;
- valid invented group-channel atmosphere is accepted;
- event fact is rejected;
- commitment is rejected;
- volatile public/current fact is rejected;
- source evidence is required for active writes;
- repeated equivalent style input does not increment revision;
- repeated equivalent style input can merge source refs;
- malformed user/group scope fields cannot become active;
- dry-run apply reports exact planned mutations before any write.

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not draft the plan. If no separate reviewer is available, the drafting
agent must reread this plan, `development_plans/README.md`, the development
plan contract, and the direct source/test context from a fresh-review posture.

Review scope:

- The proposed scope aligns with the project boundary: reflection and
  consolidation produce style candidates, DB helpers validate and persist,
  runtime consumers only read prompt-safe style overlays.
- The plan gives concrete execution instructions: contracts, change surface,
  exact files, verification gates, progress checklist, and evidence
  requirements.
- Agent creativity is bounded: no unresolved migration choices, compatibility
  shims, alternate storage, unowned helper freedom, broad prompt rewrites, or
  fallback paths remain.
- Data migration safety is explicit: backup, dry-run, manifest hash, apply
  skip on stale rows, and post-apply audit.
- Project basis remains explicit: style is good data only when scoped as
  reusable style; event facts and commitments are bad data in this lane.

Record blockers, non-blocking findings, required edits, open questions, and
approval status before changing this plan to `approved`.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- DB write boundary correctly validates target scope, source evidence,
  style-only content, normalized diff, and source-ref merge.
- Reflection and consolidation callers do not bypass the DB validator.
- Audit/repair scripts are dry-run by default, use `script_operations`, require
  approved manifests for apply, and skip stale rows.
- Tests cover valid learned style, event facts, commitments, volatile facts,
  source gaps, scope confusion, semantic no-op updates, and downstream
  non-exposure.
- No RAG, `user_memory_units`, dialog-state, adapter, or live response-path LLM
  behavior was added.
- Project style, CJK prompt safety, and test style are followed.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- New active style overlay writes reject event facts, commitments, volatile
  facts, malformed scopes, and missing source evidence.
- Equivalent updates do not increment revision.
- Bounded source-ref merge prevents repeated daily-run reprocessing without
  changing prompt-facing style.
- Existing high-churn rows are audited and repaired only after review.
- User and group/channel style scopes remain separate.
- Dry-run/apply repair is backed by export, manifest, stale-row protection,
  and post-apply audit.
- Verification commands pass, or live DB dry-run gates are explicitly recorded
  as not run because live DB inspection was not approved.
- Independent Code Review approves or all findings are remediated inside the
  approved scope.

## Data Migration

1. Export `interaction_style_images` to a timestamped backup artifact.
2. Run read-only audit.
3. Generate dry-run repair report and manifest.
4. Review rows flagged for rewrite, disable, scope repair, or manual review.
5. Apply only an approved manifest with stale-row hash checks.
6. Re-run audit and compare before/after counts.
7. Record backup path, manifest path, apply report, and post-apply audit in
   `Execution Evidence`.

Apply safety:

- default mode is dry-run;
- `--apply` requires an explicit manifest path and backup path;
- each row selector must include `style_image_id` and `expected_current_hash`;
- stale rows are skipped, not force-updated;
- disabled rows retain original stored overlay in backup/report;
- rewrites require an approved replacement overlay in the manifest;
- no row is deleted by this plan.

Rollback:

- restore from the pre-apply backup artifact through a separately reviewed
  restore command;
- do not reconstruct original rows from memory or from the dry-run report
  alone.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Useful style rewritten away | Dry-run review, keep-by-default classification, approved rewrite manifest only | Audit samples and repair tests |
| Validator false negatives allow subtle facts | Prompt payload hardening plus wrong-lane tests and post-apply audit | Event/commitment/volatile-fact tests |
| Validator false positives block real style | Preserve concise style examples in tests and allow manual review | Valid style acceptance tests |
| Churn gate blocks real evolution | Normalize only prompt-facing overlay equality; meaningful text changes still revise | Diff-gate tests |
| Source-ref merge grows without bound | Bounded, deduplicated source ref list | Source-ref cap tests |
| Current high revision hides history | Do not reset revision by default; record audit findings | Audit report |
| Live DB unavailable during execution | Keep deterministic tests and record dry-run gate as blocked | Execution Evidence |

## Self Plan Review And Remediation

Review performed during this refinement against the user-requested scope and
the repo development-plan contract.

Findings and fixes applied:

- Finding: the previous draft named `consolidation/images.py` as a modify
  target, but source inspection shows that file owns character self-image, not
  interaction style images.
  Fix: corrected the change surface to `db/interaction_style_images.py`,
  `reflection_cycle/interaction_style*.py`, `consolidation/group_channel.py`,
  `consolidation/persistence.py`, and maintenance script boundaries.
- Finding: the previous RCA did not explain how revision churn could happen.
  Fix: added the current upsert behavior, source-ref replacement, duplicate
  suppression weakness, and semantic no-op update path to RCA.
- Finding: failure modes were too general for the lane.
  Fix: added detailed lane-specific failure modes for revision churn,
  event-fact contamination, commitment contamination, user/group scope
  confusion, reflection noise, semantic no-op updates, source evidence gaps,
  volatile public facts, quote leakage, and downstream behavior impact.
- Finding: cleanup strategy did not specify data migration safety.
  Fix: added backup, dry-run, reviewed manifest, stale-row hash checks,
  disable-not-delete default, approved rewrite behavior, post-apply audit, and
  rollback constraints.
- Finding: the source hardening plan did not clearly separate LLM judgment from
  deterministic mechanics.
  Fix: added explicit ownership boundaries: LLM proposes style; reflection and
  consolidation own source eligibility; DB helpers own validation, evidence,
  diff, revision, and persistence.
- Finding: requested explicit items could be lost across sections.
  Fix: retained named subsections for issue description, plan to remove
  malformed data, RCA, plan to harden the corrupted data source, and plan to
  prove robustness.
- Finding: plan-review content was missing.
  Fix: added this `Self Plan Review And Remediation` section plus an
  `Independent Plan Review` gate.
- Finding: the plan must remain a draft and must not authorize code/data
  changes.
  Fix: kept `Status: draft`, left all progress boxes unchecked, and repeated
  the no-execution/no-apply rules.
- Finding: the refined plan exceeded the `large` line-count maximum from the
  repo plan contract.
  Fix: reclassified the plan as `high_risk_migration` because it includes
  live-data audit, repair, apply safety, and rollback constraints. The active
  registry row was not edited because this refinement is constrained to this
  single plan file.
- Finding: placeholder scan surfaced broad test-addition wording and a loose
  source-module update phrase.
  Fix: changed the wording to focused validation tests and bounded any
  source-module update to source-noise copy-through prevention with focused
  coverage.

Residual risks:

- No live DB rows were inspected during this refinement; the 37-row and
  revision-724 facts come from the supplied audit context in the existing
  draft.
- Semantic contamination detection cannot be perfect with deterministic rules
  alone. The plan mitigates this through prompt-source hardening, explicit
  rejection tests, manual review classification, and post-apply audit.
- Historical revision causes cannot be proven from the current single-document
  schema alone. The plan treats source-ref replacement and missing diff gate as
  code-supported likely causes and verifies the fix by preventing future churn.

## Execution Evidence

Cleanup-only execution completed on 2026-07-03. New-write hardening remains
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

Commands and artifacts:

- `venv\Scripts\python.exe -m py_compile src\scripts\_lane_cleanup.py src\scripts\repair_interaction_style_images_lane.py src\kazusa_ai_chatbot\db\script_operations.py`
- `venv\Scripts\python.exe -m scripts.repair_interaction_style_images_lane --dry-run --manifest test_artifacts\interaction_style_images_lane_repair_manifest.json --output test_artifacts\interaction_style_images_lane_repair_dry_run.json`
- `venv\Scripts\python.exe -m scripts.repair_interaction_style_images_lane --apply --manifest test_artifacts\interaction_style_images_lane_repair_dry_run.json --output test_artifacts\interaction_style_images_lane_repair_apply.json`
- `venv\Scripts\python.exe -m scripts.repair_interaction_style_images_lane --dry-run --manifest test_artifacts\interaction_style_images_lane_repair_manifest.json --output test_artifacts\interaction_style_images_lane_post_repair_dry_run.json`

Results:

- Baseline: 38 total rows, 38 active rows, 0 findings, 0 deterministic planned
  actions.
- Apply: 0 actions, 0 blocked actions.
- Post-audit: 0 findings and 0 deterministic planned actions remain.
