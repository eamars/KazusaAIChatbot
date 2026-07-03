# shared memory lane data integrity plan

## Summary

- Goal: repair malformed shared `memory` rows and harden every shared-memory
  write path so project-global durable memory stays separate from user,
  group-channel, episode-local, volatile public-fact, behavior-control, and
  audit-only data.
- Plan class: high_risk_migration.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `database-data-pull`, `memory-knowledge-maintenance`, `debug-llm`,
  `py-style`, `cjk-safety`, `test-style-and-execution`.
- Overall cutover strategy: bigbang for new non-seed shared-memory write
  eligibility; migration for existing malformed rows through export,
  dry-run classification, reviewed apply, and post-apply audit.
- Highest-risk areas: archiving valid project-continuity data because it
  differs from external canon, preserving reflection-promoted global leakage,
  moving rows without resolvable target ownership, stale Cache2 evidence after
  repair, and weakening `memory_evolution` provenance requirements.
- Acceptance criteria: active shared memory contains only curated/project-global
  rows and approved reflection/global-growth source rows; non-seed shared
  writes require evidence refs, privacy review, shared-lane eligibility, and
  truthful authority; wrong-lane reflection-promoted rows are archived, moved,
  relabeled, or left in explicit manual review with zero silent active leakage.

New-write hardening supersession: all lane-specific new-write hardening,
source-generation proof, and fixing-strategy instructions in this plan are
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.
Execute this lane plan for cleanup, audit, dry-run/apply, data migration, and
post-cleanup verification only. If another section still names new-write
validators, prompts, or tests, treat that text as historical context rather
than execution scope.

## Context

The 2026-07-02 memory audit found 304 shared `memory` rows: 202 active and 102
superseded. Seed sync dry-run reported the 104 seed-managed rows unchanged, and
seed validation passed. The primary corruption is in active
`reflection_promoted` rows, not in the seed-maintenance mechanism.

Case review classified the shared lane as:

- 103 correct or acceptable rows;
- 102 inactive superseded rows;
- 53 active rows with invalid global scope;
- 20 wrong-lane or unverified rows;
- 14 wrong-lane rows;
- 7 volatile public facts in the wrong lane;
- 2 behaviorally harmful rules;
- 2 overbroad behavior rules;
- 1 mislabeled project/canon authority row.

The current source contracts explain the failure boundary:

- `src/kazusa_ai_chatbot/memory_evolution/repository.py` validates evolving
  memory shape, ids, status, authority enum, embedding ownership, write lock,
  and Cache2 invalidation, but it does not yet validate that non-seed active
  rows are genuinely shared/global.
- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py` builds
  reflection-promoted `EvolvingMemoryDoc` rows with `source_global_user_id=""`,
  `source_kind="reflection_inferred"`, and
  `authority="reflection_promoted"`. It requires evidence refs and privacy
  review, but the validator still treats `lore` and `self_guidance` as enough
  lane intent and lacks deterministic shared-scope classification.
- `src/kazusa_ai_chatbot/rag/memory_retrieval_tools.py` and
  `src/kazusa_ai_chatbot/rag/memory_evidence/workers/persistent_search.py`
  retrieve active shared `memory` rows as durable evidence. A wrong active row
  can therefore reach normal cognition through full RAG and the first-cycle
  shared-memory prewarm.
- `src/kazusa_ai_chatbot/reflection_cycle/context.py` projects active
  reflection-promoted `fact` and `defense_rule` rows into promoted reflection
  context. Wrong-lane rows in those memory types can become prompt-visible
  outside the originating conversation.
- `src/kazusa_ai_chatbot/global_character_growth/runner.py` consumes active
  reflection-promoted memory as input cards for character-growth trait
  generation. It is not the primary corrupted writer, but noisy promoted memory
  can contaminate global growth inputs.
- `personalities/knowledge/memory_seed.jsonl` contains one seed row whose
  content claims an official address while the confidence note says it is
  artificial project world knowledge. That row is a truthful-authority defect,
  not proof that project-continuity data is invalid.

Kazusa does not aim to mimic original Kazusa canon. Invented or learned
project-continuity data can be good data when it is global, durable,
source-labeled, prompt-safe, and stored in the correct lane. Good data in the
wrong memory lane is bad data because it changes retrieval scope and
authority.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing reflection promotion,
  generation prompts, LLM write-intent contracts, RAG prompt contracts, or
  global-character-growth source-card selection.
- `database-data-pull`: load before exporting or inspecting live shared-memory
  rows.
- `memory-knowledge-maintenance`: load before changing seed JSONL, validation,
  or shared-memory sync.
- `debug-llm`: load before live reflection-promotion checks or prompt-quality
  comparisons.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files with CJK prompt or memory
  strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute this plan while status is `draft`.
- Do not apply live data cleanup without a reviewed dry-run report and explicit
  user command.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Use public memory-evolution APIs and `db.script_operations`; do not add raw
  MongoDB access outside `kazusa_ai_chatbot.db`.
- Do not treat external canon mismatch as invalidity by itself. Invented or
  learned project-continuity data can be valid shared data when lane, scope,
  provenance, privacy, and authority are correct.
- Do not persist active non-seed shared memory without `evidence_refs`,
  `privacy_review`, empty `source_global_user_id`, approved shared-lane audit
  classification, and truthful authority.
- Do not use keyword filters as semantic truth adjudication. LLM extraction may
  propose content; deterministic validation owns lane, target, provenance,
  privacy, lifecycle, authority, and persistence checks.
- Do not change RAG ranking, dispatcher routing, cognition prompts, dialog
  prompts, or retrieval result shape to compensate for bad stored data.
- Repair scripts must default to dry-run, emit planned filters/actions, and
  require an explicit `--apply` plus reviewed action classes before mutation.
- Repair apply must record before/after counts and invalidate Cache2
  `source="memory"` after successful memory lifecycle changes, moves, relabels,
  or archives.
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

- Add a shared-memory lane audit that classifies active rows by source,
  authority, scope, lifecycle, provenance, privacy, retrieval impact, and
  recommended repair action.
- Add dry-run/apply repair for malformed shared-memory rows.
- Preserve seed/project-continuity rows when their lane is correct.
- Relabel the seed row that claims unsupported official/canon address authority
  so it clearly states project-continuity authority, or fix the fixture wording
  that generates it.
- Archive or move wrong-lane active reflection-promoted rows.
- Harden reflection promotion and memory-evolution insertion so non-seed
  shared writes require evidence refs, privacy review, shared-lane eligibility,
  truthful authority, and lifecycle validity.
- Prove RAG retrieval, shared-memory prewarm, promoted reflection context, and
  global-character-growth inputs no longer surface archived wrong-lane rows.
- Add tests proving valid invented project-global facts are accepted and
  wrong-lane content is rejected or routed to manual review.

## Deferred

- Do not redesign RAG retrieval or memory ranking.
- Do not fact-check all shared memory against external Blue Archive sources.
- Do not delete plausible project-continuity memory only because it differs
  from external canon.
- Do not add a new shared-memory collection.
- Do not change dialog, cognition, RAG finalizer, or retrieval prompts to
  compensate for bad storage.
- Do not implement a general ontology, canon verifier, or semantic migration
  LLM agent.
- Do not introduce compatibility shims, dual-write paths, fallback mappers, or
  alternate shared-memory vocabularies.

## Cutover Policy

Overall strategy: bigbang for new writes; migration for existing data.

| Area | Policy | Instruction |
| --- | --- | --- |
| New non-seed shared writes | bigbang | Reject rows that fail evidence, privacy, scope, authority, lifecycle, or audit-class checks. |
| Reflection promotion | bigbang | Split proposed durable output by explicit lane before persistence; no global default and no LLM-only lane authority. |
| `memory_evolution` validation | bigbang | Enforce the non-seed shared validator in insert, supersede, and merge replacement documents. |
| Existing malformed rows | migration | Export, audit dry-run, review, then archive/move/relabel through approved repair actions only. |
| Seed memory | compatible | Keep JSONL source and seed reset shape; correct authority wording without changing seed sync mechanics. |
| RAG memory evidence | compatible | Keep retrieval shape stable; cleaned rows change available evidence but no consumer contract changes. |
| Cache2 invalidation | bigbang | Every applied repair batch emits memory invalidation before success is reported. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative compatibility strategy by
  default.
- For bigbang areas, reject or rewrite legacy behavior instead of preserving
  old write shapes.
- For migration areas, follow the exact audit, dry-run, review, apply, and
  post-apply gates in this plan.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

Shared `memory` stores only:

- curated seed/project-global facts;
- intentionally project-owned global continuity facts;
- approved reflection-promoted global lore with evidence and privacy review;
- approved global character-growth source memory that remains suitable for
  growth input;
- curated behavior rules only after explicit seed or reviewed approval.

It does not store:

- user preferences, commitments, impressions, private relationship claims, or
  current-user continuity;
- group/channel style, social atmosphere, local norms, or channel-specific
  interaction images;
- episode-local observations or one-off chat events;
- volatile public/current facts without expiry and live-context ownership;
- unapproved reflection-derived behavior-control rules;
- raw reflection output, raw transcripts, adapter ids, private user details, or
  LLM-generated source refs.

Observable completed behavior:

- A new active reflection-promoted row cannot be written unless it passes the
  shared validator.
- The known project-continuity address row no longer claims official/canon
  authority.
- Post-cleanup RAG persistent-memory retrieval and first-cycle prewarm can only
  surface rows that remain active after audit repair.
- Promoted reflection context and global-character-growth input cards consume
  only approved active reflection-promoted shared rows.
- Manual review rows are inactive for normal retrieval.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Correctness basis | Lane, scope, provenance, lifecycle, privacy, and authority; not original canon | Project continuity can invent new facts. |
| Primary source hardening | Enforce at both reflection promotion and `memory_evolution` repository | Promotion should give precise warnings; repository must be the final persistence guard. |
| Existing wrong-lane rows | Archive or move only with deterministic target and evidence | Avoid deleting useful continuity or guessing ownership. |
| Project address row | Relabel as project-continuity and remove official/canon claim | The problem is overclaimed authority, not the existence of artificial continuity. |
| Reflection promotion | Require explicit shared-lane audit classification before memory write | Importance to character is not global truth. |
| Behavior rules | Active shared behavior rules require seed or reviewed approval | Reflection should not create global policy by itself. |
| RAG consumers | Keep retrieval contract stable | Bad storage is fixed at write/repair boundaries, not by changing evidence consumers. |
| Cache invalidation | Use existing `source="memory"` Cache2 invalidation | Current cache dependency model already owns memory freshness. |

## Lane Analysis Requirements

### Issue Description Based On Deep Analysis

Shared `memory` is the project-global durable lane. The seed-managed subset is
mostly stable, but active reflection-promoted rows include user-specific,
group-specific, episode-local, volatile, and behavior-rule data that should not
be globally retrievable as shared truth. The defect is not that reflection
learned invented continuity; it is that the promotion path accepted content
whose scope, authority, lifecycle, or privacy review did not justify active
shared-memory storage.

The bad state has direct retrieval impact. Full RAG persistent-memory search,
shared-memory prewarm, promoted reflection context, consolidation dedup checks,
and global-character-growth source-card selection can all consume active
shared `memory` rows. Once a wrong-lane row is active, it can influence
cognition and future background learning even when the row originally came
from one user, one group, one episode, or one volatile public fact.

### Plan To Remove Malformed Data

Add a read-only lane audit, export the full `memory` collection before cleanup,
classify active rows by source, authority, scope, lifecycle, privacy, evidence,
retrieval impact, and repair action, then run repair in dry-run mode by
default. Apply only reviewed action classes.

Repair actions are:

- Keep valid seed, manual, and reflection-promoted rows whose scope is global,
  durable, prompt-safe, and truthful.
- Relabel seed/project-continuity authority wording when the row is valid
  project-global data but overclaims external canon or official authority.
- Archive wrong-lane reflection-promoted rows when no target lane, target id,
  and source evidence can be resolved without guessing.
- Move rows only when the target lane and target identity are deterministic:
  current-user continuity to `user_memory_units`, group/channel style to
  `interaction_style_images`, character posture to character-state or global
  character-growth review, and audit-only rows to archived/rejected memory.
- Route unapproved or harmful behavior rules to manual review instead of
  keeping them as active shared `defense_rule` rows.
- Leave ambiguous rows in manual review with `status="rejected"` or archived
  active-off lifecycle metadata rather than deleting them.

### RCA Of The Failure Mode

Root cause is a contract gap between reflection promotion and
`memory_evolution` persistence:

- Reflection promotion treated "important to the character" or "useful for
  future behavior" as sufficient to write shared memory. It did not require a
  deterministic proof that the row was global rather than user, group,
  episode, volatile, or local policy.
- The promotion prompt has good instructions, but deterministic validation
  trusts the LLM's `lane` plus structural privacy fields too much. Local LLM
  output is allowed to propose semantic conclusions; it must not be the final
  owner of storage lane, target identity, lifecycle, privacy clearance, or
  behavior-rule approval.
- `memory_evolution` stores non-seed provenance fields but does not fail closed
  when an active non-seed shared row has empty `evidence_refs`, missing or weak
  `privacy_review`, user-scoped `source_global_user_id`, unsupported authority
  labeling, volatile content without expiry, or behavior-rule content without
  approval.
- Seed maintenance correctly owns the local JSONL source and reset mechanics,
  but seed text can still overclaim authority. The known address row's issue is
  wording and authority labeling; it should be project-continuity knowledge, not
  "official" external canon.
- Retrieval consumers correctly retrieve active memory; they are not the source
  of corruption. They reveal the impact because shared-memory search and
  promoted reflection context assume active rows have already passed storage
  governance.

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

## Failure Modes And Root Cause Detail

| Failure mode | Likely root cause | Impact | Planned fix |
| --- | --- | --- | --- |
| User preference or commitment stored in shared `memory` | Reflection promotion lacks target-lane proof and uses empty `source_global_user_id` for all promoted rows | Any user can retrieve another user's continuity as global truth | Audit as `wrong_lane_user_scoped`; move only with resolvable user id and source evidence, otherwise archive/manual review |
| Group/channel atmosphere stored as global lore | Daily/group reflection compresses channel patterns into global promotion cards | Private group style can affect unrelated chats | Audit as `wrong_lane_group_channel_scoped`; route to `interaction_style_images` only with platform/channel id |
| Episode-local observation stored as durable shared fact | "Important today" collapsed into "long-term fact" | Stale single-event details become permanent evidence | Audit as `episode_local`; archive unless repeated durable evidence exists |
| Volatile public fact stored without expiry | Memory lane used as current-fact cache | RAG returns stale prices, laws, versions, or schedules | Audit as `volatile_public`; archive or require expiry plus live-context policy |
| Harmful or overbroad behavior rule stored as active `defense_rule` | Reflection self-guidance promoted without manual/global approval | Character behavior is globally constrained by one noisy reflection | Audit as `behavior_rule_unapproved` or `behavior_rule_harmful`; manual review only |
| Seed row overclaims official/canon authority | Seed content text and confidence note disagree | RAG answers artificial continuity as official external fact | Relabel content/note to project-continuity wording and add validator coverage |
| Active non-seed row has empty evidence refs | Repository normalizes provenance but does not require it | Cannot audit source or safely migrate | Reject new writes; existing rows become `missing_provenance` manual review |
| Privacy review is missing or weak | Prompt review result trusted without repository guard | Private detail may become shared evidence | Repository rejects non-seed rows without low/medium privacy review and user-detail removal |
| Cache2 serves stale memory evidence after cleanup | Repair updates lifecycle without invalidation | RAG can continue using deleted/archived rows in process cache | Repair apply must invalidate `source="memory"` after every mutation batch |
| Migration moves useful data to wrong target | Repair guesses user/channel/character target from text | Data remains corrupt under a different collection | Move only when target id and evidence refs are deterministic; otherwise archive/manual review |

## Contracts And Data Shapes

### Audit Finding Contract

Add a maintenance audit finding shape in the approved maintenance boundary:

```python
class SharedMemoryLaneFinding(TypedDict):
    memory_unit_id: str
    lineage_id: str
    status: str
    source_kind: str
    authority: str
    source_global_user_id: str
    memory_type: str
    audit_classification: Literal[
        "valid_shared_seed",
        "valid_project_global",
        "valid_reflection_global",
        "valid_behavior_rule",
        "wrong_lane_user_scoped",
        "wrong_lane_group_channel_scoped",
        "episode_local",
        "volatile_public",
        "behavior_rule_unapproved",
        "behavior_rule_harmful",
        "missing_provenance",
        "privacy_blocked",
        "authority_mislabeled",
        "manual_review_required",
    ]
    issue_code: str
    issue_description: str
    retrieval_impact: Literal[
        "rag_persistent_memory",
        "shared_memory_prewarm",
        "promoted_reflection_context",
        "global_character_growth_input",
        "none_inactive",
    ]
    recommended_action: Literal[
        "keep",
        "archive",
        "move_to_user_memory_units",
        "move_to_interaction_style_images",
        "move_to_character_state_review",
        "relabel_project_continuity",
        "manual_review",
    ]
    target_ref: dict[str, object]
    evidence_fields: dict[str, object]
    apply_allowed: bool
```

Audit classifications are deterministic labels over stored fields and approved
source metadata. They are not canon judgments. Content semantics that require
human interpretation remain `manual_review_required`.

### Repair Report Contract

Dry-run and apply reports must include:

```python
class SharedMemoryLaneRepairReport(TypedDict):
    mode: Literal["dry_run", "apply"]
    generated_at: str
    source_export_path: str
    requested_action_classes: list[str]
    counts_before: dict[str, int]
    counts_after: dict[str, int]
    findings: list[SharedMemoryLaneFinding]
    planned_actions: list[dict[str, object]]
    applied_actions: list[dict[str, object]]
    blocked_actions: list[dict[str, object]]
    cache_invalidated: bool
    warnings: list[str]
```

Dry-run must populate `planned_actions` and never mutate. Apply must require
the caller to pass action classes from the reviewed dry-run and must record
`applied_actions`, `blocked_actions`, before/after counts, and cache
invalidation status.

### Shared-Memory Insert Validator Contract

Add a local validator used by `insert_memory_unit`, `supersede_memory_unit`,
and `merge_memory_units` replacement documents:

```python
def validate_shared_memory_write_eligibility(
    document: EvolvingMemoryDoc,
    *,
    operation: Literal["insert", "supersede", "merge"],
) -> SharedMemoryWriteEligibility
```

Validation rules:

- Required base fields remain `memory_unit_id`, `lineage_id`, `memory_name`,
  `content`, `memory_type`, `source_kind`, `authority`, and `status`.
- Active shared rows must have `source_global_user_id == ""`.
- Seed rows must use `authority="seed"` and `source_kind` in
  `{"seeded_manual", "external_imported"}`.
- Non-seed active rows must use `authority in {"reflection_promoted",
  "manual"}` and must not use seed-managed authority.
- Non-seed active rows require non-empty `evidence_refs` with source ids
  derived from repository-owned records, not from untrusted LLM output.
- Non-seed active rows require `privacy_review.user_details_removed is True`,
  `privacy_review.private_detail_risk in {"low", "medium"}`, and a non-empty
  `boundary_assessment`.
- `memory_type="defense_rule"` is allowed only for seed rows or rows carrying
  approved behavior-rule classification.
- Volatile rows require explicit `expiry_timestamp`; durable rows must not use
  expiry to hide uncertain target ownership.
- `conversation_extracted` and `relationship_inferred` source kinds are not
  active shared-global writes unless a caller supplies approved shared-global
  classification; user or relationship data belongs outside shared memory.

The validator returns structured reasons for audit logs and tests, but write
APIs raise `ValueError` before embedding, DB insert/update, or cache
invalidation when eligibility fails.

### Reflection Promotion Decision Contract

Reflection promotion keeps a single background LLM call. It must not add a new
repair LLM or ask the LLM to generate database operations.

The promotion path adds deterministic post-parse fields before write:

```python
{
    "shared_audit_classification": "shared_global_fact | project_continuity_global | approved_global_behavior_rule | reject",
    "shared_audit_reason": "bounded reason for logs and run document",
    "write_eligibility": "accepted | rejected | manual_review",
}
```

The LLM may still output `lane`, `sanitized_memory_name`,
`sanitized_content`, `boundary_assessment`, and `privacy_review`.
Deterministic code derives repository evidence refs, validates scope/authority,
rejects target-lane ambiguity, and decides whether to call `memory_evolution`.

### Migration Safety Contract

- Export before cleanup with embeddings excluded by default.
- No repair action may delete rows. Wrong-lane shared rows become inactive by
  lifecycle update or are copied to a target lane and then archived/superseded.
- A move requires deterministic target identity and source evidence. Without
  both, the row is archived or marked manual review.
- Apply is idempotent by `memory_unit_id`, `lineage_id`, target refs, and
  lifecycle status.
- Apply blocks when the dry-run report is missing, stale, or mismatched against
  current row counts.

## LLM Call And Context Budget

Affected LLM path: background daily global reflection promotion. Global
character growth is an affected downstream consumer but does not receive new
prompt input under this plan.

- Before: daily global promotion uses one consolidation-route LLM call and can
  produce plausible durable conclusions that reach shared-memory insertion
  without enough deterministic shared-scope enforcement.
- After: daily global promotion still uses one consolidation-route LLM call.
  The prompt can be tightened to ask for explicit global/shared reasoning, but
  deterministic validation owns storage lane, target, evidence, privacy,
  lifecycle, behavior-rule approval, and authority.
- Added LLM calls: none.
- Response path: unchanged. This remains background reflection and migration
  work outside the live chat response path.
- Context budget: no increase above `GLOBAL_PROMOTION_PROMPT_MAX_CHARS=25000`.
  If prompt wording changes, use short semantic lane descriptions and keep raw
  database schema, migration language, plan names, and Mongo fields out of the
  prompt except for already-visible output keys.
- Local LLM constraint: do not ask the model to infer collection ownership,
  target ids, privacy policy, cache invalidation, or migration actions. Those
  remain deterministic.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/memory_evolution/models.py`: add audit/eligibility
  typed shapes or constants only where they are part of the public memory
  contract.
- `src/kazusa_ai_chatbot/memory_evolution/repository.py`: enforce non-seed
  shared-memory eligibility before embedding and before insert/supersede/merge
  replacement writes.
- `src/kazusa_ai_chatbot/memory_evolution/reset.py`: preserve seed sync while
  validating seed authority wording and keeping project-continuity rows in the
  seed-managed global lane.
- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`: require explicit
  shared-scope audit classification, preserve repository-derived evidence
  refs, and reject/manual-review wrong-lane decisions before memory writes.
- `src/kazusa_ai_chatbot/reflection_cycle/context.py`: keep the public
  projection shape, but add tests proving inactive/manual-review rows are not
  returned through existing active-only repository calls.
- `src/kazusa_ai_chatbot/global_character_growth/projection.py` or
  `validation.py`: only add validation needed to reject corrupted
  reflection-promoted memory cards after cleanup; do not change trait storage
  contracts.
- `src/kazusa_ai_chatbot/db/script_operations.py`: add shared-memory audit,
  dry-run repair planning, reviewed apply, and cache-invalidation helper calls
  for maintenance scripts.
- `personalities/knowledge/memory_seed.jsonl`: relabel the project-continuity
  address row so content and confidence note do not claim official/canon
  authority.

### Create

- `src/scripts/audit_shared_memory_lane.py`: read-only audit/report CLI.
- `src/scripts/repair_shared_memory_lane.py`: dry-run-by-default repair CLI
  with explicit apply gate.
- `tests/test_shared_memory_lane_integrity.py`: validator, audit, repair, and
  retrieval-impact tests.

### Keep

- RAG memory-evidence result shape.
- Persistent-memory worker public tool signatures.
- Memory-evolution public repository entrypoints.
- Seed sync dry-run/apply workflow.
- Global-character-growth collection contracts.

## Overdesign Guardrail

- Actual problem: active shared `memory` contains wrong-lane
  reflection-promoted rows and at least one overclaimed authority label.
- Minimal change: enforce shared-lane eligibility, add a lane-specific audit,
  correct authority wording, and repair only deterministic malformed classes.
- Ownership boundaries: LLM stages propose semantic durable content;
  deterministic reflection promotion validates lane intent; `memory_evolution`
  validates persistence eligibility; `db.script_operations` owns maintenance
  backend access; RAG retrieves evidence but does not repair storage.
- Rejected complexity: no new collection, no canon fact-checker, no live
  response-path LLM call, no semantic migration LLM, no keyword truth filter,
  no compatibility path, no RAG rank changes, no prompt-wide architecture
  rewrite.
- Evidence threshold: add more architecture only when a valid project-global
  fact cannot be represented by current `memory` fields after eligibility
  hardening and that limitation is proven by a failing deterministic test or
  approved live/debug LLM case.

## Agent Autonomy Boundaries

- The execution agent may implement audit/repair scripts and validators inside
  the listed change surface after approval.
- The execution agent must not apply data cleanup without an approved dry-run.
- The execution agent must not delete invented project-continuity data solely
  because it is non-canon.
- The execution agent must stop if moving a row requires guessing a user,
  group/channel, character, or source target.
- The execution agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, or extra features.
- The execution agent must search for existing export, cache invalidation,
  reset, and repository helper behavior before adding new helper code.
- The execution agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, broad prompt rewrites, or refactors outside the listed
  files.
- If this plan and source code disagree, preserve the plan's stated lane
  integrity intent and report the discrepancy.

## Implementation Order

1. Establish focused deterministic test contract:
   - repository rejects active non-seed shared rows without evidence/privacy;
   - repository accepts valid invented project-global rows with provenance;
   - reflection promotion rejects user, group, episode, volatile, and
     unapproved behavior-rule decisions before write;
   - seed authority validation catches the project-continuity address wording;
   - retrieval-impact tests prove inactive wrong-lane rows do not reach RAG
     memory evidence, shared-memory prewarm, promoted reflection context, or
     global-character-growth input cards.
2. Implement shared-memory audit classification in
   `db.script_operations` and expose `audit_shared_memory_lane.py`.
3. Implement repository-level shared write eligibility for insert,
   supersede, and merge replacement documents.
4. Harden reflection promotion to attach deterministic shared audit
   classification and to write only accepted shared-global rows.
5. Correct seed authority wording and validation for project-continuity rows.
6. Implement repair dry-run/apply planning in `db.script_operations` and expose
   `repair_shared_memory_lane.py`.
7. Generate dry-run report against the current database, review action counts,
   and record the reviewed action classes in execution evidence.
8. Apply only approved action classes after explicit user command.
9. Re-run audit, seed validation, retrieval-impact tests, repository tests,
   reflection promotion tests, and global-character-growth focused tests.

## Detailed Fixing Strategy

The lane-specific fixing strategy for preventing future writes is superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

Retain only the cleanup, audit, migration, and post-cleanup verification actions
already defined in this lane plan. If implementation finds a new-write hardening
need outside the consolidator router plan, stop and update the plan instead of
executing the superseded fixing strategy.


## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only;
  does not edit tests unless the parent explicitly directs it; closes after
  planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Focused shared-memory tests added and baseline recorded.
- [ ] Read-only shared-memory audit command implemented.
- [ ] Shared-memory eligibility validation implemented.
- [ ] Reflection promotion hardened for shared-lane writes.
- [ ] Seed authority wording corrected and validated.
- [ ] Repair dry-run/apply command implemented.
- [ ] Dry-run report reviewed and action classes approved.
- [ ] Approved cleanup applied.
- [ ] Post-apply audit and seed validation recorded.
- [ ] Retrieval-impact verification completed.
- [ ] Independent code review completed.

## Verification

Focused deterministic tests:

```powershell
venv\Scripts\python.exe -m pytest tests/test_shared_memory_lane_integrity.py -q
venv\Scripts\python.exe -m pytest tests/test_memory_evolution_repository.py tests/test_memory_evolution_reset.py tests/test_memory_evolution_retrieval.py -q
venv\Scripts\python.exe -m pytest tests/test_memory_knowledge_sync_runtime_lore.py tests/test_reflection_cycle_stage1c_promotion.py tests/test_reflection_cycle_stage1c_reflection_context.py -q
venv\Scripts\python.exe -m pytest tests/test_shared_memory_prewarm.py tests/test_global_character_growth_runner.py tests/test_global_character_growth_validation.py -q
```

Static and maintenance verification:

```powershell
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/memory_evolution/models.py src/kazusa_ai_chatbot/memory_evolution/repository.py src/kazusa_ai_chatbot/memory_evolution/reset.py src/kazusa_ai_chatbot/reflection_cycle/promotion.py src/kazusa_ai_chatbot/db/script_operations.py src/scripts/audit_shared_memory_lane.py src/scripts/repair_shared_memory_lane.py
venv\Scripts\python.exe -m scripts.manage_memory_knowledge validate
venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync
venv\Scripts\python.exe -m scripts.audit_shared_memory_lane --output test_artifacts\shared_memory_lane_audit.json
venv\Scripts\python.exe -m scripts.repair_shared_memory_lane --dry-run --output test_artifacts\shared_memory_lane_repair_dry_run.json
```

Apply verification after explicit user command:

```powershell
venv\Scripts\python.exe -m scripts.repair_shared_memory_lane --apply --reviewed-report test_artifacts\shared_memory_lane_repair_dry_run.json --action-class archive --action-class relabel_project_continuity --output test_artifacts\shared_memory_lane_repair_apply.json
venv\Scripts\python.exe -m scripts.audit_shared_memory_lane --output test_artifacts\shared_memory_lane_post_apply_audit.json
```

Live/debug LLM robustness cases, run one at a time with `debug-llm`:

- valid invented project-global fact persists to shared memory;
- project-continuity address/lore avoids official-canon wording;
- user preference is rejected from shared memory;
- group-channel atmosphere is rejected from shared memory;
- episode-local observation is rejected from shared memory;
- volatile public fact is rejected or requires expiry and live-context
  ownership;
- unapproved behavior rule enters manual review, not active shared memory.

## Self Plan Review And Remediation

Review performed during draft refinement on 2026-07-02.

| Finding | Remediation applied to this plan | Residual risk |
| --- | --- | --- |
| The prior draft named the issue but did not trace exact corrupted write and retrieval boundaries. | Added context for `memory_evolution.repository`, reflection promotion, RAG persistent memory, shared-memory prewarm, promoted reflection context, and global-character-growth consumers. | Implementation still must confirm no additional writer exists through grep before coding. |
| Failure modes were too broad and did not separate good project-continuity data from wrong-lane data. | Added explicit failure-mode table and repeated the project basis: invented/learned project continuity can be valid; good data in the wrong lane is bad data. | Ambiguous rows still require manual review instead of automatic migration. |
| RCA did not explain why prompt instructions alone were insufficient. | Added LLM/local architecture split: LLM proposes content; deterministic code owns lane, target, lifecycle, privacy, authority, and persistence. | Prompt wording can still be improved, but repository validation is the required final guard. |
| Repair strategy lacked action taxonomy and apply safety. | Added audit finding contract, repair report contract, dry-run/apply gates, action classes, baseline export, stale-report blocking, and rollback inputs. | Live DB cleanup remains high risk until dry-run counts are reviewed. |
| Seed authority labeling was present but not concrete. | Added explicit requirement to relabel the project-continuity address row away from official/canon wording while preserving valid project continuity. | Correcting `memory_name` could churn deterministic seed id; plan instructs preferring content/note correction. |
| Retrieval impact was under-specified. | Added verification coverage for RAG persistent-memory retrieval, shared-memory prewarm, promoted reflection context, and global-character-growth input. | Search result ranking can still expose low-quality valid rows; ranking redesign is deferred. |
| Cache invalidation was mentioned but not enforced. | Added mandatory repair apply invalidation and post-apply cache verification requirements. | Existing process-local cache must be tested under the real Cache2 dependency path. |
| Plan-review content was missing. | Added this self-review table with findings, remediations, and residual risks. | Independent code review still required after implementation. |
| Detailed analysis sections initially appeared before mandatory plan-control sections. | Moved lane analysis and failure modes after settled design decisions so top matter follows the repo plan contract more closely. | This remains a draft, but execution agents should keep final-plan section order stable before approval. |
| Required user items could drift during edits. | Preserved explicit sections for issue description, plan to remove malformed data, RCA, plan to harden the corrupted source, and plan to prove robustness. | Execution agents must reread this plan after compaction and major stages. |

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- No raw MongoDB access outside `kazusa_ai_chatbot.db`.
- No external-canon cleanup rule was introduced.
- Non-seed shared writes require evidence refs, privacy review, shared-lane
  classification, lifecycle validity, and truthful authority.
- Repair dry-run is default; apply requires reviewed report and explicit action
  classes.
- Cache2 invalidation is emitted after applied memory repairs.
- Tests cover valid invented project-global data and wrong-lane rejection.
- RAG retrieval shape, shared-memory prewarm shape, promoted reflection context
  shape, and global-character-growth collection contracts remain stable.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

- Active shared memory has zero automatic wrong-lane reflection-promoted rows
  after approved cleanup, excluding inactive manual-review rows.
- Project-continuity data is preserved with truthful authority labeling.
- The project-continuity address row no longer claims official/canon authority.
- Non-seed shared-memory writes without evidence refs, privacy review,
  shared-lane classification, or empty global scope are rejected before
  embedding and persistence.
- Reflection promotion cannot write user, group, episode, volatile, privacy
  blocked, or unapproved behavior-rule content into active shared memory.
- RAG persistent-memory retrieval, shared-memory prewarm, promoted reflection
  context, and global-character-growth input tests prove archived/manual-review
  wrong-lane rows are not prompt-visible.
- Verification commands pass and reports are stored under `test_artifacts/`.

## Data Migration

1. Export `memory` before cleanup.
2. Run shared-memory audit.
3. Generate repair dry-run report.
4. Review actions by issue code and action class.
5. Apply only approved action classes after explicit user command.
6. Re-run audit and compare before/after counts.
7. Keep pre-apply export and apply report as rollback inputs.

Rollback uses the pre-apply export and apply report. Archived source rows remain
available until moved-target rows are verified. No row is hard-deleted by this
plan.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Valid project lore archived | Preserve non-canon data when lane and authority are correct | Audit samples and valid project-global tests |
| Wrong-lane rows remain active | Deterministic audit and post-apply report | Zero active wrong-lane findings |
| Reflection stops learning useful global facts | Accept valid invented project-global robustness case | Deterministic and live/debug LLM cases |
| Private data remains prompt-visible | Privacy review validator and inactive manual-review rows | Retrieval-impact tests |
| Stale Cache2 evidence survives cleanup | Invalidate `source="memory"` after repair apply | Cache invalidation assertions and post-apply retrieval smoke |
| Migration moves data to wrong target | Move only with deterministic target id and source evidence | Blocked-action report and manual-review counts |
| Seed id churn from authority relabel | Prefer content/note correction over `memory_name` change | Seed sync dry-run count review |

## Execution Evidence

Cleanup-only execution completed on 2026-07-03. New-write hardening remains
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

Commands and artifacts:

- `venv\Scripts\python.exe -m py_compile src\scripts\_lane_cleanup.py src\scripts\repair_shared_memory_lane.py src\kazusa_ai_chatbot\db\script_operations.py`
- `venv\Scripts\python.exe -m scripts.repair_shared_memory_lane --dry-run --output test_artifacts\shared_memory_lane_repair_dry_run.json`
- `venv\Scripts\python.exe -m scripts.repair_shared_memory_lane --apply --reviewed-report test_artifacts\shared_memory_lane_repair_dry_run.json --action-class archive --action-class relabel_project_continuity --output test_artifacts\shared_memory_lane_repair_apply.json`
- `venv\Scripts\python.exe -m scripts.repair_shared_memory_lane --dry-run --output test_artifacts\shared_memory_lane_post_repair_dry_run.json`

Results:

- Baseline: 305 total rows, 203 active rows, 0 findings, 0 deterministic
  planned actions.
- Apply: 0 actions, 0 blocked actions.
- Post-audit: 0 findings and 0 deterministic planned actions remain.
