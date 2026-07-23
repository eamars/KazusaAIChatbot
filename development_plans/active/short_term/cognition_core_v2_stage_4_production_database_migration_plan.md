# Cognition Core V2 Stage 4 Production Database Migration Plan

## Summary

- Goal: migrate the production database and deploy the completed Stage 3
  native Cognition Core V2 release candidate without losing valid character,
  relationship, conversation, memory, commitment, reflection, task, calendar,
  audit, or continuity data.
- Plan class: high_risk_migration.
- Status: draft.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `database-data-pull`, `py-style`, `cjk-safety`,
  `test-style-and-execution`, `debug-llm`, `character-test`,
  `control-console-web-development`, and `python-venv`.
- Overall cutover strategy: discover read-only, freeze exact transforms,
  rehearse backup/restore and migration on an isolated copy, then perform a
  coordinated big-bang application/database cutover with one rollback pair.
- Highest-risk areas: production identity/relationship semantics, mutable
  character state, memory lineage, pending commitments/tasks, legacy
  scheduler rows, malformed variants, migration duration, and revision skew.
- Acceptance criteria: every discovered row is preserved, transformed,
  quarantined, or explicitly rejected by an approved rule; backup/restore,
  rehearsal, production apply, native validation, behavioral smoke,
  observation, and rollback gates have inspectable user sign-off.
- Record role: non-executable Stage 4 placeholder.
- Detail level: lifecycle boundary and discovery checklist only. This document
  is not executable until Stage 3 completes, its target contracts are frozen,
  production discovery is explicitly authorized by the user, exact source
  shapes are inventoried read-only, and this plan is expanded and independently
  reviewed.
- Production authority: withheld. Creating this placeholder grants no
  permission to read, export, back up, mutate, migrate, deploy, restart, or
  validate production.
- Cutover intent: big-bang native V2 database/application cutover with verified
  backup and rollback. No compatibility, dual-read, dual-write, or runtime
  legacy translator.

## Context

Stage 3 proves the entire runtime on an absent dedicated test database and
freezes the production target schema. Stage 4 then addresses the different
problem: preserving and transforming existing production continuity.

Stage 4 now tracks the completed Stage 3 handoff inputs:

```text
development_plans/archive/completed/short_term/
  cognition_core_v2_stage_3_system_adoption_plan.md
  cognition_core_v2_stage_3_execution_manifest.md
  cognition_core_v2_stage_3_change_radius.md
```

After Stage 3 completion, these inputs move to
`development_plans/archive/completed/short_term/`; Stage 4 must update the
links before activation. It also consumes the Stage 3 handoff packet containing
the release-candidate revision, native collection/index inventory, target
schemas, migration-only file boundary, and preliminary source-code legacy
inventory.

## Activation Gates

All gates are mandatory:

1. Stage 3 status is `completed` and its plan is archived.
2. The user has signed the Stage 3 quality, performance, fresh-database,
   restart, and auxiliary evidence.
3. The exact Stage 3 release-candidate commit is frozen.
4. The native V2 target schemas and indexes are frozen.
5. The user explicitly authorizes read-only production database discovery.
6. Read-only discovery records database identity, topology, collection counts,
   index definitions, schema variants, malformed rows, and legacy-field usage
   without exposing secrets or unbounded user content.
7. This placeholder is expanded into a complete executable plan with exact
   transforms, scripts, commands, rollback, verification, owners, maintenance
   window, and acceptance criteria.
8. Independent plan review and user approval complete.
9. The user separately authorizes production migration execution.

## Mandatory Skills

The future planning/execution owner must apply:

- `development-plan`;
- `local-llm-architecture`;
- `database-data-pull` for authorized read-only diagnostic exports;
- `py-style`, `cjk-safety`, and `test-style-and-execution`;
- `debug-llm` and `character-test` for post-migration behavioral evidence;
- `control-console-web-development` for production operator validation;
- `python-venv` before environment/dependency work.

## Mandatory Rules

1. Production discovery and migration require separate explicit user commands.
2. Secrets, credentials, full raw conversations, protected prompts, and
   unrestricted traces stay out of plan artifacts.
3. Discovery is read-only and uses approved repository scripts/facades.
4. A complete verified backup exists before the first production write.
5. Migration defaults to dry-run and produces a deterministic bounded plan.
6. Apply is idempotent, resumable, auditable, and fails before partial writes
   when an unknown blocking source shape is found.
7. Native V2 rows already valid are preserved rather than regenerated.
8. Semantic relationship, memory, and character meaning is not fabricated by
   deterministic conversion. Unmappable semantics are listed for user review
   or a separately approved LLM-assisted migration gate.
9. LLM-assisted migration, if approved, runs offline on bounded records with
   source/target evidence, deterministic validation, retry limits, audit, and
   human review. It never runs implicitly during service startup.
10. The production service is quiesced or placed behind a verified write
    barrier before the final snapshot and apply phase.
11. Runtime code contains no migration fallback after cutover.
12. Rollback restores the verified pre-migration database and prior application
    revision together.
13. After automatic context compaction and after each signed checkpoint, the
    active agent rereads the complete expanded Stage 4 plan.
14. Before approval or execution, the expanded plan must use parent-led native
    subagent execution and include independent plan/code review gates.
15. Apply begins with a complete read-only preflight that identifies every
    blocking source variant before writes. Only then may idempotent, resumable,
    checkpointed batches begin.

## Known Production Change Radius

This inventory comes from source contracts only and must be reconciled against
authorized read-only production discovery.

### Character singleton

Potential source fields:

```text
character_state._id = "global"
static profile fields
self_image
cognition_state
mood
global_vibe
reflection_summary
```

Target:

- one complete validated static character profile;
- one valid `cognition_state.v2`;
- preserved valid self-image/growth continuity;
- no runtime authority from legacy prose affect.

### User profiles and relationship continuity

Potential source fields:

```text
user_profiles.global_user_id
identity/profile fields
affinity
last_relationship_insight
relationship_state
embedded cognition state
user image/style fields
```

Target:

- one canonical user identity row;
- native V2 relationship axes/state;
- preserved valid relationship insight and style evidence through current
  owners;
- no runtime affinity authority;
- no invented maximum-affinity equivalent.

### Conversation and short-term progress

```text
conversation_history
conversation_episode_state
internal_monologue_residue_state
```

Default intent: preserve valid history and delivery/role metadata, validate
indexes, expire only through existing TTL/lifecycle rules, and rebuild only
derived caches. Exact malformed/legacy row handling waits for discovery.

### Memory and growth

```text
user_memory_units
memory
interaction_style_images
global_character_growth_traits
global_character_growth_runs
```

Default intent: preserve lineage, status, authority, evidence, revisions,
active commitments, and user scope. Stage 4 must detect duplicate identities,
invalid lifecycle states, orphan lineage, and stale legacy projections before
apply.

### Reflection, scheduler, tasks, and background work

```text
character_reflection_runs
calendar_schedules
calendar_runs
scheduled_events
accepted_tasks
background_work_jobs
self_cognition_action_attempts
self_cognition_group_review_windows
internal_action_latches
```

Known scheduler rule: legacy `scheduled_events` is migration/audit input only.
Pending future-cognition rows map through the existing calendar migration
contract; pending delayed visible `send_message` rows are cancelled rather
than copied as prewritten future text. Exact production counts and statuses
wait for discovery.

### Cache, trace, and operator data

```text
rag_cache2_persistent_entries
rag_cache_index
rag_metadata_index
llm_trace_runs
llm_trace_steps
event_log_events
event_log_snapshots
internal_monologue_residue_state
post_turn_lifecycle_records
background_artifact_jobs
```

`rag_cache_index`, `rag_metadata_index`, and `background_artifact_jobs` are
legacy collection names dropped by the pre-Stage-3 bootstrap. Stage 3 removes
that destructive startup behavior, so Stage 4 discovery must count them
explicitly before disposition. Derived caches may be
invalidated/rebuilt. Trace/event/residue/post-turn-lifecycle retention,
archival, and schema changes require explicit rules in the expanded plan.
Control-console audit views are
derived from their current owners and are not treated as a standalone Mongo
collection unless authorized discovery proves one exists.

### Known Stage-4-only source files

```text
src/kazusa_ai_chatbot/db/script_operations.py
src/scripts/_lane_cleanup.py
src/scripts/migrate_scheduled_events_to_calendar_scheduler.py
```

The expanded plan must classify these as reuse, modify, replace, or retire and
name every additional migration file exactly.

## Required Discovery Artifact

After user authorization, produce a bounded
`Stage4ProductionDiscoveryV1` artifact containing:

```python
class Stage4ProductionDiscoveryV1(TypedDict):
    schema_version: Literal["stage4_production_discovery.v1"]
    database_identity_hash: str
    discovered_at: str
    application_revision: str
    collection_counts: dict[str, int]
    index_summaries: dict[str, list[str]]
    schema_variant_counts: dict[str, dict[str, int]]
    legacy_field_counts: dict[str, int]
    malformed_row_counts: dict[str, int]
    duplicate_identity_counts: dict[str, int]
    pending_work_counts: dict[str, int]
    cache_rebuild_candidates: dict[str, int]
    blockers: list[str]
    secrets_in_artifact: Literal[False]
```

Hashes, counts, field paths, and statuses are allowed. Row content and samples,
credentials, and protected prompts/traces are excluded from this discovery
artifact. If a later transform cannot be designed from structural counts, a
separate user-authorized bounded diagnostic export must define its own typed,
redacted artifact.

## Expanded Plan Must Define

Before approval, replace this placeholder with exact:

- source schemas and every observed variant;
- target schemas and index definitions from Stage 3;
- per-collection transform functions;
- preservation, default, rejection, quarantine, and manual-review rules;
- affinity/prose-affect disposition based on observed production data;
- character-profile completion strategy;
- relationship-state conversion strategy;
- memory/lineage and active-commitment integrity checks;
- scheduler/task/background-work transition rules;
- cache invalidation/rebuild rules;
- dry-run and apply script paths and CLI arguments;
- batch size, checkpoint, retry, idempotency, and resume contracts;
- backup and restore commands;
- maintenance/write-barrier procedure;
- application/database deployment order;
- health, smoke, real-LLM, adapter, console, worker, and restart verification;
- rollback triggers, maximum rollback time, and responsible owner;
- post-cutover observation window and incident response;
- exact evidence and sign-off records.

## Preliminary Implementation Order

This order is directional and becomes executable only after discovery:

1. Complete read-only production preflight and blocker classification before
   any write-capable command exists.
2. Exact target/source contract freeze and independent plan review.
3. Migration tool failing-first tests against synthetic copies of every
   observed schema variant.
4. Offline sanitized production-shape dry run.
5. Backup/restore rehearsal on an isolated database.
6. Full migration rehearsal with the Stage 3 application candidate.
7. User review of transform counts, quarantines, quality, latency, and rollback
   evidence.
8. Production maintenance window, write barrier, final backup, full read-only
   preflight, dry run, idempotent/resumable batch apply, application deployment,
   index verification, and startup.
9. Foreground/background/adapter/console validation and controlled observation.
10. User cutover sign-off or coordinated application+database rollback.

## Progress Checklist

- [x] Stage 4 purpose separated from Stage 3 fresh-database adoption.
- [x] Production authority and activation gates recorded.
- [x] Preliminary collection/field change radius recorded from source code.
- [x] Required discovery artifact and expanded-plan obligations drafted.
- [ ] Stage 3 completed and release candidate frozen.
- [ ] User authorizes read-only production discovery.
- [ ] Production discovery artifact completed.
- [ ] Exact transforms, commands, rollback, verification, and ownership
  expanded.
- [ ] Independent plan review completed.
- [ ] User approves the executable Stage 4 plan.
- [ ] User authorizes production migration execution.
- [ ] Migration, deployment, observation, and sign-off complete.

## Acceptance Boundary

This placeholder is complete as a Stage 4 reservation when it:

- prevents Stage 3 from touching production;
- identifies the known production data lanes;
- requires read-only discovery before transform design;
- requires backup, rehearsal, idempotent apply, coordinated rollback, and user
  authorization;
- contains no executable production command or unverified transform rule.

It is not an accepted implementation plan and cannot authorize Stage 4
execution.

## Risks To Resolve After Discovery

| Risk | Required future evidence |
|---|---|
| Production profile lacks fields required by Stage 3 startup | Exact missing-field counts and completion decision |
| Affinity cannot be faithfully mapped to multidimensional V2 relationship state | Distribution, correlated evidence, transform/non-transform decision |
| Existing native V2 state conflicts with legacy fields | Per-variant precedence and preservation rules |
| Duplicate or malformed user identities | Counts, quarantine/merge policy, rollback |
| Pending scheduler/task rows change meaning at cutover | Status inventory and exact transition matrix |
| Derived cache contains legacy projections | Cache ownership inventory and rebuild proof |
| Migration duration exceeds maintenance window | Rehearsal throughput and batch/resume evidence |
| Application/database revisions become mismatched | Coordinated deployment and rollback rehearsal |
| Production memories materially change character quality or performance | Controlled post-migration real-LLM corpus and user sign-off |

## Execution Evidence

Placeholder drafted on 2026-07-18 from repository contracts only. No
production connection, discovery, export, backup, migration, deployment, or
restart was performed.
