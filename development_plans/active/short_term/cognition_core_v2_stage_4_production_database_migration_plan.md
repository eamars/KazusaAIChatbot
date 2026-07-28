# Cognition Core V2 Stage 4 Production Database Construction Plan

## Summary

- Goal: construct and verify `kazusa_core_v2` from the critical, compatible
  continuity data in `kazusa_bot_core`.
- Plan class: `high_risk_migration`.
- Status: `draft`.
- Executability: the implementation boundary, conversion rules, commands,
  safeguards, tests, artifacts, and acceptance gates are complete. Status
  remains `draft` until independent plan review and explicit user approval.
- Source database: `kazusa_bot_core`.
- Target database: `kazusa_core_v2`.
- Source disposition: preserve the complete source unchanged as the user's
  pre-migration backup.
- Target disposition: build a native V2 production candidate that the user may
  select later.
- Runtime boundary: Stage 4 performs offline database construction and
  verification only; the user owns configuration, database selection,
  deployment, restart, final delta handling, and cutover.
- Workspace boundary: repository changes, synthetic tests, and test-database
  rehearsals occur in the DEV workspace. The running production application
  and its configuration remain unchanged.
- Data principle: migrate durable identity, relationship, conversation,
  memory, style, self-image, and active growth; rebuild derived data; leave
  transient, executor, scheduler, raw-reflection, trace, log, and audit state
  in `kazusa_bot_core`.
- Relationship principle: a valid owner-matching V2 state wins. Affinity
  contributes only to `positive_regard`; every other axis requires separate
  bounded evidence.
- Snapshot principle: the target is a high-water-marked migration candidate
  built while the source may remain live. Stage 4 records source drift and
  never claims that the candidate includes later writes.
- Execution authority: this update authorizes documentation only. Discovery,
  implementation, target writes, and verification require the authority
  recorded under Agent Autonomy Boundaries.

## Context

Stage 3 established native Cognition Core V2 runtime contracts. Stage 4 has a
different ownership boundary: it prepares the data store that can support that
runtime while preserving the old database intact.

The governing direction is a bounded offline conversion from read-only,
retained `kazusa_bot_core` into the new native V2 candidate
`kazusa_core_v2`.

Stage 4 does not redirect any caller to the new database. It does not update
`.env`, `MONGODB_DB_NAME`, application settings, service definitions, adapter
settings, deployment manifests, or running processes. The word "production"
in this plan describes the future role of `kazusa_core_v2`, not an application
state transition performed by Stage 4.

The source may continue receiving writes while a candidate is built. The
migration therefore records a bounded source image using stable row
identifiers, row fingerprints, per-collection scan start/end times, and
high-water markers. This is not a globally atomic cross-collection snapshot.
The verification report states the observed source interval and all detected
drift. The user decides how to handle later source writes before selecting the
new database.

The completed Stage 3 handoff inputs are:

```text
development_plans/archive/completed/short_term/
  cognition_core_v2_stage_3_system_adoption_plan.md
  cognition_core_v2_stage_3_execution_manifest.md
  cognition_core_v2_stage_3_change_radius.md
```

Stage 4 uses the repository's current validators and schema builders as the
canonical target contract. It does not duplicate those contracts inside the
migration tool.

## Mandatory Skills

The execution owner applies these skills at the point where they become
relevant:

- `development-plan` before reviewing, approving, executing, or signing off
  this plan;
- `local-llm-architecture` before implementing or reviewing the bounded
  relationship reconstruction step;
- `database-data-pull` before any authorized read-only source discovery or
  bounded diagnostic export;
- `py-style` before creating or modifying Python;
- `cjk-safety` if a changed Python file contains CJK text;
- `test-style-and-execution` before creating, modifying, or running tests;
- `debug-llm` before running or evaluating offline relationship
  reconstruction calls;
- `python-venv` before environment verification or dependency work.

`character-test` and `control-console-web-development` are outside this plan:
Stage 4 does not exercise the running application or a production control
surface.

## Mandatory Rules

1. `kazusa_bot_core` is read-only for every Stage 4 command.
2. `kazusa_core_v2` is the only database that Stage 4 may create or mutate.
3. Database handles are passed explicitly to migration and schema functions.
   The configured runtime database remains untouched.
4. Source and target names must equal the two literal names in this plan.
   The CLI rejects aliases, omissions, matching names, and swapped names.
5. A fresh apply requires every target runtime collection to contain zero
   documents. Existing indexes and empty collections are allowed.
6. Stage 4 never drops either database or any source or target collection.
7. Stage 4 never edits environment files, runtime configuration, deployment
   files, service definitions, or application database selection.
8. Discovery and dry-run are read-only. Apply requires an explicit
   `--apply` flag and the acknowledgement value `kazusa_core_v2`.
9. Preflight classifies every source row before the first target write.
   An unknown blocking variant prevents apply.
10. Apply is idempotent for the frozen source manifest, uses stable target
    keys, writes in batches of 250, and records a protected local checkpoint
    after every committed batch.
11. Resume requires matching discovery, source-manifest, conversion-plan,
    relationship-decision, code-revision, source-database, and target-database
    digests.
12. A resume mismatch stops before a target write. Recovery is review and an
    explicit user decision; automatic target cleanup is prohibited.
13. Existing native V2 data is copied only after canonical validation.
14. Deterministic conversion may normalize shape, bounds, and timestamps. It
    may not invent relationship, memory, identity, or character meaning.
15. Offline LLM reconstruction may use bounded corroborating evidence only
    under the contract in Design Decisions. It has no runtime role.
16. Raw conversations, credentials, protected prompts, and unrestricted
    traces stay out of committed artifacts and shareable reports.
17. Malformed critical data is excluded from the target, identified by stable
    hash and reason, and blocks completion until the user approves its
    disposition.
18. Noncritical incompatible data is omitted with a counted reason. Its full
    original remains available in `kazusa_bot_core`.
19. Target writes use native V2 shapes only. Runtime compatibility readers,
    dual writes, aliases, and legacy translation layers are outside scope.
20. After context compaction and after each signed execution stage, the active
    agent rereads this complete plan before continuing.
21. Plan status is an execution gate, not production authorization. Every
    authority gate below must also be satisfied.

## Must Do

Stage 4 must:

1. implement handle-explicit target schema/index creation and guarded
   `discover`, `dry-run`, `apply`, and `verify` commands while preserving
   configured bootstrap behavior;
2. inventory every source collection and every observed document variant;
3. freeze a protected source manifest with stable identifiers and
   fingerprints;
4. apply the exact per-lane rules and validate bounded relationship decisions
   before target writes;
5. create `kazusa_core_v2` while preserving `kazusa_bot_core`;
6. regenerate embeddings and create current native ordinary/vector indexes;
7. reconcile counts/dispositions, active-work handoff, and source drift;
8. pass deterministic, CLI, isolated MongoDB, independent plan, and
   independent code review gates;
9. stop at a verified candidate and hand the evidence to the user.

## Deferred

The following work belongs to a later user-owned production transition:

- changing `MONGODB_DB_NAME` or any equivalent runtime database selector;
- changing `.env`, secrets, service definitions, containers, deployment
  manifests, or adapter configuration;
- stopping, restarting, quiescing, or deploying the production application;
- redirecting foreground or background workers;
- choosing the final source high-water mark;
- migrating source writes that occur after the Stage 4 manifest;
- reconciling a later delta into the candidate;
- production health, adapter, browser, control-console, or live-character
  smoke tests;
- declaring `kazusa_core_v2` active;
- retiring, deleting, renaming, or mutating `kazusa_bot_core`;
- defining rollback for a later application database selection.

The deferred items are not acceptance gates for this database-construction
plan.

## Cutover Policy

Stage 4 contains no application cutover.

The Stage 4 completion boundary is source preserved, target constructed,
target verified, and evidence handed off. The user later controls the final
delta decision, configuration change, deployment/restart, and validation.

Consequently:

- the source database remains the active application database throughout
  Stage 4 unless the user independently changes it;
- the target database remains disconnected from the application throughout
  Stage 4;
- Stage 4 records target readiness without asserting production activation;
- later cutover and rollback procedures require their own approved plan or
  explicit user-run procedure based on the state that exists at that time.

## Target State

### Database identity

`kazusa_core_v2` contains native current-schema collections and indexes. It
contains only the durable data lanes selected below plus empty runtime
collections created by native bootstrap.

`kazusa_bot_core` retains every original collection and document. Stage 4
performs no cleanup or archival writes against it.

### Character continuity

The target contains exactly one `character_state` singleton with `_id:
"global"`.

It contains:

- the source static character profile after validation against the current
  character profile contract;
- a valid `self_image` if the source value passes the current self-image
  contract;
- the source `cognition_state` when it is a valid
  `cognition_state.v2` character state;
- otherwise, the exact current V2 character default produced by the canonical
  builder.

It excludes:

- `mood`;
- `global_vibe`;
- `reflection_summary`;
- any legacy affect prose or unknown character-state field.

A missing or invalid required static character profile is critical malformed
data and blocks completion. An invalid optional `self_image` is omitted and
counted. An invalid or absent character `cognition_state` takes the explicit
V2-default path and is counted.

### User identity and relationship continuity

Each migrated `user_profiles` document contains:

- one non-empty canonical `global_user_id`;
- valid platform account rows;
- normalized current display names;
- valid `suspected_aliases` that point to another migrated canonical user;
- one valid owner-matching `cognition_state.v2` user state.

The configured character identity may have an empty platform-account list and
is preserved as the reserved character identity. Every other migrated user
must own at least one valid platform account.

A platform account requires non-empty `platform` and `platform_user_id`;
`display_name` must be a string, and `linked_at` must be a valid timestamp when
present. Values receive whitespace/timestamp normalization, while an empty
display name remains empty.

The migration rejects a user identity as synthetic when it is not the reserved
character identity and has no valid platform account. It does not infer
synthetic status from a display-name keyword or naming style.

If one `(platform, platform_user_id)` pair belongs to multiple
`global_user_id` values, all affected identities are excluded and completion
is blocked pending an explicit user disposition. The migration does not merge
those identities automatically.

Legacy `facts`, `affinity`, `last_relationship_insight`, and source embeddings
are not copied into the target profile document. Active facts already
represented by valid `user_memory_units` move through that owner.

### Durable user memory

`user_memory_units` contains all structurally valid active memory units and
active commitments for migrated canonical users.

Preserved values include:

- `unit_id`, owner, `unit_type`, fact, appraisal, and relationship signal;
- lifecycle status, count, first/last/update timestamps;
- valid source references and merge history;
- commitment due date;
- existing completion, cancellation, and archive fields only when the active
  status contract permits them.

Terminal archived, completed, and cancelled units remain in
`kazusa_bot_core`. Source embeddings are discarded and regenerated.

### Shared memory

The `memory` collection contains each structurally valid active lineage head
and every valid ancestor required to preserve that head's provenance.

The target excludes:

- rejected or expired lineages;
- unrelated superseded branches;
- orphan rows that cannot reach their required ancestor;
- source embeddings.

A broken required ancestor chain is critical malformed data and blocks
completion. Embeddings are regenerated.

### Conversation history

The target preserves every usable user or assistant message for a migrated
identity.

For a current native row, `body_text` remains the text source. For a recognized
legacy row, `content` becomes `body_text`. The transform preserves:

- platform and channel identifiers;
- channel type and role;
- canonical global user identity;
- display name;
- content type;
- normalized message text;
- timestamp;
- valid platform message identifier;
- valid typed address, mention, and reply-context metadata when present;
- attachment `media_type` and normalized `description` only.

Legacy rows without typed address, mention, or reply metadata receive empty
typed lists, absent reply context, and `broadcast=False`. The transform does
not parse raw Discord, QQ, or debug-wire syntax to synthesize these fields.

The target excludes:

- binary attachment payloads and base64 data;
- attachment URLs, byte counts, and storage-shape metadata;
- raw wire text;
- delivery bookkeeping;
- LLM trace references;
- source embeddings;
- unknown legacy fields.

A row without a valid platform, channel, channel type, user/assistant role,
canonical user mapping, timestamp, or text field is critical malformed data.
It is excluded and blocks completion. Embeddings are regenerated.

### Interaction style and growth

`interaction_style_images` contains valid active overlays and the active
revision history required by the current owner contract. Invalid, inactive,
or superseded unrelated revisions remain in the source.

`global_character_growth_traits` contains valid active traits. Growth run
history and audit rows remain in the source.

### Empty operational lanes

The following target owners start empty:

- `conversation_episode_state`;
- `internal_monologue_residue_state`;
- internal action latches;
- post-turn lifecycle records;
- accepted task executor state;
- background-work job state;
- calendar schedules and runs;
- legacy scheduled events;
- raw character-reflection runs;
- self-cognition attempt and review ledgers;
- global-character-growth runs;
- RAG persistent caches;
- event logs and snapshots;
- LLM trace runs and steps.

Promoted reflection outcomes survive only through their current durable owners:
memory, interaction style, self-image, or active growth traits.

### Index and cache state

All current native ordinary indexes and vector indexes are created from the
current bootstrap contract using the explicit `kazusa_core_v2` handle.

All migrated embedding-bearing rows receive fresh current-dimension
embeddings. No source embedding or derived RAG cache entry is copied.

## Design Decisions

### Source image and high-water contract

Discovery scans each in-scope source collection in ascending stable `_id`
order. For every row the protected `Stage4SourceRowV1` manifest records
`collection`, `source_id`, `source_revision`, `source_fingerprint`, and
`disposition`.

`source_revision` is the normalized native update timestamp when present,
otherwise the empty string. `source_fingerprint` is SHA-256 over canonical
Extended JSON after excluding source embedding arrays only. The fingerprint
still covers all semantic and lifecycle fields.

For each collection, `Stage4CollectionHighWaterV1` records `collection`,
`scan_started_at`, `scan_finished_at`, `highest_stable_id`, `row_count`, and
`manifest_digest`.

Apply migrates exactly the manifest rows. Before transforming each row, it
recomputes the fingerprint. A changed or missing row stops the current batch
before its writes and records source drift. Rows created after the collection
scan are outside this candidate and appear in the verification drift report.

### Character-state precedence

The precedence order is:

1. validate and copy the source V2 character cognition state;
2. when step 1 fails or the value is absent, call the canonical current
   character-state builder once;
3. validate the resulting state;
4. block apply if the canonical builder cannot produce a valid state.

Legacy mood, vibe, and reflection prose never influences the default.

### User relationship precedence

For each non-character user:

1. If the source contains a complete canonical V2 state that validates and
   `owner_user_id == global_user_id`, copy it unchanged.
2. Otherwise create the canonical acquaintance user state.
3. If source affinity is a numeric finite value, calculate:

   ```text
   positive_regard = clamp(round((affinity - 500) / 5), -100, 100)
   ```

4. Set only `relationship.positive_regard` from that result.
5. Keep every other relationship axis at the acquaintance default unless the
   bounded relationship reconstruction contract returns separately
   corroborated evidence for that exact axis.
6. Validate the completed user state and owner match before it enters the
   conversion plan.

The acquaintance relationship defaults are:

| Axis | Default |
|---|---:|
| `familiarity` | 10 |
| `positive_regard` | 0 |
| `trust` | 0 |
| `attachment` | 0 |
| `desired_closeness` | 10 |
| `perceived_closeness` | 10 |
| `care` | 0 |
| `boundary_safety` | 0 |
| `exclusivity` | 0 |
| `unresolved_injury` | 0 |
| `salience` | 0 |

Affinity never maps to `trust`, `attachment`, `desired_closeness`,
`perceived_closeness`, `care`, `boundary_safety`, `exclusivity`,
`unresolved_injury`, `familiarity`, or `salience`.

Non-numeric, Boolean, NaN, or infinite affinity is omitted and counted as
invalid optional relationship evidence. Out-of-range finite affinity is
clamped by the formula.

### Bounded relationship reconstruction

Relationship reconstruction is eligible only when:

- a full valid owner-matching V2 state is absent;
- `last_relationship_insight` contains bounded non-empty text; and
- at least one active memory unit or usable conversation row independently
  corroborates the insight.

It uses the existing
`COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL` route. It introduces no
configuration key and performs no call during application startup or normal
chat.

The semantic input contains:

- a qualitative positive-regard band derived after the affinity formula;
- at most 2,000 characters of normalized relationship insight;
- at most eight active memory evidence entries;
- at most twelve recent usable conversation excerpts;
- the active interaction-style overlay when present;
- opaque evidence handles for every supplied item.

The serialized semantic input is capped at 20,000 characters and 50,000 model
input tokens. Selection is deterministic: newest valid evidence first, with a
stable identifier tie-breaker.

Positive-regard bands are:

| Value | Band |
|---|---|
| `-100..-61` | strongly negative |
| `-60..-21` | negative |
| `-20..20` | neutral or mixed |
| `21..60` | positive |
| `61..100` | strongly positive |

The prompt states that this band supports only `positive_regard` and cannot
support another axis.

The producing LLM must return exactly:

```python
class Stage4RelationshipAxisUpdateV1(TypedDict):
    axis: Literal[
        "familiarity",
        "trust",
        "attachment",
        "desired_closeness",
        "perceived_closeness",
        "care",
        "boundary_safety",
        "exclusivity",
        "unresolved_injury",
        "salience",
    ]
    value: int
    evidence_handles: list[str]
    semantic_summary: str


class Stage4RelationshipDecisionV1(TypedDict):
    schema_version: Literal["stage4_relationship_decision.v1"]
    supported: bool
    axis_updates: list[Stage4RelationshipAxisUpdateV1]
```

Contract evaluation rules are:

- pass raw output through
  `kazusa_ai_chatbot.utils.parse_llm_json_output(...)`;
- permit at most one semantic generation call per eligible user and at most
  one canonical JSON-repair call for residual syntax or wrapper repair;
- reject unknown keys, duplicate axes, unknown axes, wrong types, empty
  summaries, or evidence handles absent from the input;
- apply declared integer clamp normalization to the axis's native interval;
- require at least one non-insight memory or conversation handle for every
  changed axis;
- keep every omitted axis at its acquaintance default;
- preserve the deterministic affinity-derived `positive_regard` regardless of
  model output;
- use at most eight newest unique validated handles to build canonical
  `evidence_refs`;
- fall back to the affinity-derived `positive_regard` plus acquaintance
  defaults when output is absent, unsupported, or invalid.

There is no semantic retry. Every fallback is counted and does not block the
migration because the default state is an approved conversion.

The protected relationship decision file contains the user identifier, final
axis updates, evidence handles, model/route identifier, prompt digest, output
digest, parser disposition, and timestamp. It contains no raw conversation,
prompt, or model output. Full raw LLM evidence follows the protected
`debug-llm` artifact rules and stays outside version control.

Memory handles become `promoted_memory` evidence refs; conversation handles
become `conversation_evidence` refs. `source_id` is the stable handle,
`occurred_at` is the source timestamp, and `semantic_summary` is the validated
bounded axis summary.

### Active and terminal lifecycle classification

Lifecycle classification uses each current owner module's exact status
constants. The migration implementation imports those constants or canonical
validators; it does not duplicate status strings.

- Active user memories and commitments migrate.
- Terminal user memories and commitments remain in the source.
- Active shared-memory lineage heads and required ancestors migrate.
- Task, background-work, calendar, reflection-run, growth-run, and
  self-cognition executor rows never migrate, regardless of status.

### Volatile work handoff

Discovery creates a bounded handoff report for active or undelivered work left
in the source:

```python
class Stage4VolatileHandoffItemV1(TypedDict):
    lane: str
    source_id_hash: str
    owner_id_hash: str
    status: str
    due_at: str
    updated_at: str
    disposition: Literal["left_in_kazusa_bot_core"]
```

The report covers accepted tasks, background work, calendar schedules and
runs, legacy scheduled events, pending deliveries, and active executor
leases. It contains no task prompt, generated message, calendar content, or
user text.

Commitments represented by active `user_memory_units` migrate through the
memory lane even when related executor state remains in the source.

### Critical and noncritical disposition

Critical lanes are:

- required character profile;
- canonical user identity and account ownership;
- selected conversation messages;
- selected active user memory and commitments;
- selected shared-memory heads and required ancestors;
- every target cognition state;
- target schema and index creation.

Malformed selected data in a critical lane has disposition
`blocked_malformed_critical`. It is not written to the target. The discovery
artifact includes collection, hashed source identifier, variant, reason, and
count. Completion waits for an explicit user-approved disposition and a
regenerated plan digest.

Invalid optional self-image, relationship insight, style overlay, growth
trait, source reference, attachment description, or alias has disposition
`omitted_incompatible_optional` when its omission does not invalidate its
owning critical row. These omissions are counted and do not block apply.

Unknown collections have disposition `left_in_source_unknown_collection`.
They are never copied. A document in a known in-scope collection with an
unknown structural variant blocks preflight.

### Embedding reconstruction

Embeddings are generated only after semantic target rows validate. The
migration calls the current embedding facade in deterministic stable-key order
and writes current-dimension vectors in batches of 250.

An embedding failure records the stable target key and stops the batch.
Resume regenerates or validates only the failed and later keys. A source
embedding is never used as a fallback.

### Migration artifacts

Execution writes protected local artifacts under:

```text
test_artifacts/cognition_core_v2/stage4/<run_id>/
```

Required artifacts are:

- `production_discovery_v2.json`;
- `source_manifest_v1.jsonl`;
- `conversion_plan_v1.json`;
- `relationship_decisions_v1.jsonl`;
- `migration_checkpoint_v1.json`;
- `volatile_handoff_v1.json`;
- `verification_v1.json`;
- `independent_plan_review.md`;
- `independent_code_review.md`.

The shareable JSON reports contain counts, hashes, timestamps, code revision,
database-name literals, and dispositions only. The manifest and relationship
decision files are protected execution inputs and remain outside version
control.

### Required artifact contracts

All artifact objects reject unknown top-level keys and use UTC ISO-8601
timestamps. Their exact required fields are:

| Contract and schema version | Required fields |
|---|---|
| `Stage4ProductionDiscoveryV2`, `stage4_production_discovery.v2` | `run_id`, literal source/target database names, `database_identity_hash`, `application_revision`, `discovered_at`, `collection_high_waters`, `collection_counts`, `index_summaries`, `schema_variant_counts`, `disposition_counts`, `duplicate_account_ownership_count`, `active_handoff_counts`, `blockers`, `source_manifest_digest`, and `secrets_in_artifact=False` |
| `Stage4MigrationPlanV1`, `stage4_migration_plan.v1` | `run_id`, literal source/target database names, `code_revision`, `discovery_digest`, `source_manifest_digest`, `relationship_decisions_digest`, `ordered_lane_counts`, `disposition_counts`, `expected_target_counts`, `blockers`, and `apply_ready` |
| `Stage4MigrationCheckpointV1`, `stage4_migration_checkpoint.v1` | `run_id`, `migration_plan_digest`, `source_manifest_digest`, `relationship_decisions_digest`, `code_revision`, literal source/target database names, `completed_lanes`, `current_lane`, `last_stable_key`, `committed_count`, and `updated_at` |
| `Stage4VerificationV1`, `stage4_verification.v1` | `run_id`, `migration_plan_digest`, `source_manifest_digest`, `source_drift_counts`, expected/actual target counts, `disposition_counts`, schema/referential/forbidden-field/empty-lane failure counts, `index_failures`, `embedding_failures`, `blockers`, and `candidate_ready` |

## Change Surface

Future execution changes exactly these production files:

| File | Change |
|---|---|
| `src/kazusa_ai_chatbot/db/stage4_migration.py` | New owner for source discovery, exact transforms, manifest/checkpoint validation, target writes, reconciliation, and verification |
| `src/kazusa_ai_chatbot/db/script_operations.py` | Re-export semantic Stage 4 maintenance entry points for scripts; keep database access in the DB package |
| `src/kazusa_ai_chatbot/db/bootstrap.py` | Extract `ensure_database_schema(database)` and have existing `db_bootstrap()` call it with the configured handle |
| `src/kazusa_ai_chatbot/db/_client.py` | Add handle-explicit vector-index creation used by `ensure_database_schema(database)` |
| `src/scripts/migrate_cognition_core_v2_database.py` | New guarded CLI with `discover`, `dry-run`, `apply`, and `verify` subcommands |
| `src/kazusa_ai_chatbot/db/README.md` | Document the cross-database maintenance boundary and safety invariants |
| `docs/HOWTO.md` | Document DEV rehearsal and separately authorized production commands |

Future execution changes exactly these test files:

| File | Change |
|---|---|
| `tests/test_stage4_database_migration.py` | New deterministic conversion, validation, precedence, drift, checkpoint, and LLM-contract tests |
| `tests/test_stage4_database_migration_cli.py` | New database-name, acknowledgement, empty-target, and command-mode guard tests |
| `tests/test_stage4_database_migration_integration.py` | New isolated MongoDB source/target schema, idempotency, resume, index, and reconciliation tests |

Plan execution updates this plan and `development_plans/README.md` only for
progress and evidence.

The following remain unchanged:

- `src/kazusa_ai_chatbot/config.py`;
- all `.env` files;
- service, adapter, dispatcher, scheduler, worker, cognition, dialog, and
  control-console runtime call paths;
- `src/scripts/_lane_cleanup.py`;
- `src/scripts/migrate_scheduled_events_to_calendar_scheduler.py`;
- production deployment and service definitions.

`_lane_cleanup.py` and the scheduled-events migration script remain unchanged
because their lanes are report-only in Stage 4.

## Overdesign Guardrail

The implementation is one offline maintenance boundary, one CLI, one
handle-explicit bootstrap extraction, and three focused test modules.

It must not introduce:

- a generic migration framework;
- a runtime database router;
- schema aliases or legacy model classes used by the application;
- dual-read or dual-write support;
- a background migration service;
- a control-console migration UI;
- a generalized workflow engine;
- target-side quarantine collections;
- automatic source cleanup;
- automatic production activation.

Variant recognition and transformation remain private to
`stage4_migration.py`. The application continues to know only current native
contracts.

## Agent Autonomy Boundaries

### Authorized by the current request

- read repository documentation and source contracts;
- update this plan and its registry metadata;
- run documentation-only checks such as `git diff --check` and placeholder
  scans.

### Requires a separate explicit user command

- implement or modify Python and tests;
- run source database discovery;
- create bounded diagnostic exports;
- invoke an LLM for relationship reconstruction;
- create or write `kazusa_core_v2`;
- run production-connected verification.

### Reserved to the user

- change any application database setting;
- change production configuration;
- stop, deploy, restart, or redirect the production service;
- choose how to reconcile writes after the source manifest;
- activate `kazusa_core_v2`;
- delete, rename, archive, or mutate `kazusa_bot_core`.

### Halt conditions

The execution owner stops and asks the user when:

- the literal database identity checks fail;
- source and target resolve to the same database;
- an observed in-scope schema variant is unknown;
- duplicate platform-account ownership exists;
- critical malformed data exists;
- the target contains user data before a fresh apply;
- a resume digest does not match;
- source drift affects a batch selected for apply;
- current target validators reject a planned row;
- embedding generation or index creation cannot complete;
- completing the task would require a deferred production action.

## Implementation Order

### Stage 0 — Reconfirm scope and authority

1. Reread this plan and `development_plans/README.md`.
2. Run `git status --short`.
3. Read `README.md`, `docs/HOWTO.md`, relevant DB/cognition/memory/task/
   scheduler READMEs, and all directly affected source/tests.
4. Record the exact authorized phase: implementation, source discovery, or
   target apply.
5. Confirm source `kazusa_bot_core`, target `kazusa_core_v2`, and DEV workspace.
6. Stop if the requested phase would cross an authority boundary.

Sign-off: scope and authority recorded in Progress Checklist.

### Stage 1 — Freeze tests and artifact contracts

1. Apply `py-style`, `test-style-and-execution`, and
   `local-llm-architecture`.
2. Create failing deterministic tests for every conversion and guard in this
   plan.
3. Create CLI tests for literal-name, source-read-only, target-empty,
   acknowledgement, dry-run, and resume guards.
4. Create isolated Mongo integration tests using names that contain a
   generated test run identifier and can never equal either production
   database literal.
5. Freeze the TypedDict artifact contracts and canonical digest rules in tests.

Sign-off: parent agent records failing-test evidence before production-code
delegation.

### Stage 2 — Implement the offline maintenance boundary

1. Add `stage4_migration.py`.
2. Extract handle-explicit schema and vector-index creation.
3. Add semantic exports to `script_operations.py`.
4. Add the guarded CLI.
5. Implement canonical Extended JSON fingerprinting, artifact redaction,
   source manifest, target plan, batch checkpoints, and verification.
6. Implement exact lane transforms and canonical validator calls.
7. Keep all runtime call sites on their existing configured behavior.

Sign-off: all deterministic and CLI tests pass.

### Stage 3 — Isolated DEV rehearsal

1. Create generated test source and target database names.
2. Load synthetic rows covering every recognized current and legacy variant.
3. Run the corresponding core discover, dry-run, apply, resume, and verify
   functions with injected generated database handles.
4. Prove source bytes/documents are unchanged.
5. Prove second apply is idempotent for the same frozen plan.
6. Prove blockers stop before writes.
7. Prove target rows pass current validators and indexes match bootstrap.

Sign-off: integration evidence is reviewed and target test databases remain
explicitly scoped to the generated run.

### Stage 4 — Independent plan review

1. Assign a fresh reviewer the plan, repository instructions, target
   contracts, and completed test contract.
2. Require review of data loss, affinity semantics, LLM ownership, source
   safety, target isolation, idempotency, resume, drift, and user-owned
   production boundaries.
3. Resolve every blocking finding in the plan.
4. Ask the user to approve the revised plan and, on approval, set its status
   and registry status to `approved`.

Sign-off: independent plan review artifact and user plan approval recorded.

### Stage 5 — Authorized source discovery

Run only after explicit read-only production discovery authority:

```powershell
venv\Scripts\python -m scripts.migrate_cognition_core_v2_database discover --source-database kazusa_bot_core --target-database kazusa_core_v2 --output-root test_artifacts/cognition_core_v2/stage4
```

The command:

1. verifies both literal database names and distinct handles;
2. reads source metadata, indexes, rows, variants, and high-water markers;
3. verifies the target without creating it;
4. emits discovery, source-manifest, and volatile-handoff artifacts;
5. exits nonzero when any preflight blocker exists.

Sign-off: user reviews counts, blocked hashes, variants, source interval, and
handoff counts.

### Stage 6 — Authorized dry-run and relationship freeze

Run only after the user approves discovery evidence and authorizes bounded
offline relationship reconstruction:

```powershell
venv\Scripts\python -m scripts.migrate_cognition_core_v2_database dry-run --source-database kazusa_bot_core --target-database kazusa_core_v2 --discovery test_artifacts/cognition_core_v2/stage4/<run_id>/production_discovery_v2.json --source-manifest test_artifacts/cognition_core_v2/stage4/<run_id>/source_manifest_v1.jsonl --output-root test_artifacts/cognition_core_v2/stage4
```

The command:

1. validates every source fingerprint;
2. performs deterministic transforms in memory;
3. runs eligible relationship reconstruction once per user;
4. validates all target rows without writing them;
5. emits frozen relationship decisions and conversion plan;
6. sets `apply_ready=True` only when blockers are empty.

Sign-off: user reviews transform and omission counts, LLM fallbacks, expected
target counts, and blockers.

### Stage 7 — Authorized target construction

Run only after explicit user authority to create and write the target:

```powershell
venv\Scripts\python -m scripts.migrate_cognition_core_v2_database apply --source-database kazusa_bot_core --target-database kazusa_core_v2 --discovery test_artifacts/cognition_core_v2/stage4/<run_id>/production_discovery_v2.json --source-manifest test_artifacts/cognition_core_v2/stage4/<run_id>/source_manifest_v1.jsonl --conversion-plan test_artifacts/cognition_core_v2/stage4/<run_id>/conversion_plan_v1.json --relationship-decisions test_artifacts/cognition_core_v2/stage4/<run_id>/relationship_decisions_v1.jsonl --checkpoint test_artifacts/cognition_core_v2/stage4/<run_id>/migration_checkpoint_v1.json --apply --target-write-ack kazusa_core_v2
```

Apply order is:

1. empty-target and digest preflight;
2. native collections and ordinary indexes;
3. `character_state`;
4. reserved character identity and canonical user profiles;
5. user relationship cognition states;
6. active `user_memory_units`;
7. shared-memory lineage;
8. `conversation_history`;
9. active `interaction_style_images`;
10. active `global_character_growth_traits`;
11. regenerated embeddings;
12. vector indexes;
13. count and schema verification.

Each lane commits stable-key batches of 250 and updates the protected
checkpoint after the database acknowledges the batch. Target writes are
upserts keyed by native logical identifier. A resumed batch validates an
already-present row against its planned fingerprint before treating it as
complete.

Sign-off: apply checkpoint shows every lane completed.

### Stage 8 — Candidate verification

Run:

```powershell
venv\Scripts\python -m scripts.migrate_cognition_core_v2_database verify --source-database kazusa_bot_core --target-database kazusa_core_v2 --discovery test_artifacts/cognition_core_v2/stage4/<run_id>/production_discovery_v2.json --source-manifest test_artifacts/cognition_core_v2/stage4/<run_id>/source_manifest_v1.jsonl --conversion-plan test_artifacts/cognition_core_v2/stage4/<run_id>/conversion_plan_v1.json --relationship-decisions test_artifacts/cognition_core_v2/stage4/<run_id>/relationship_decisions_v1.jsonl --output-root test_artifacts/cognition_core_v2/stage4
```

Verification reads both databases, writes neither, and emits
`verification_v1.json`.

Sign-off: `candidate_ready=True`, or the user receives the exact blockers and
the candidate remains inactive.

### Stage 9 — Independent code review and handoff

1. Run the independent code review gate below.
2. Resolve blocking findings and rerun affected verification.
3. Update progress/evidence in this plan.
4. Hand the user the discovery interval, source drift, target verification,
   handoff report, artifact paths, commit revision, and remaining user-owned
   actions.
5. Stop without changing application state.

Sign-off: user accepts the database-construction evidence.

## Execution Model

Execution uses parent-led native subagent work:

1. The parent owns scope, repository inspection, test contract, commands,
   evidence, and user communication.
2. After the parent creates the failing tests in Stage 1, exactly one
   production-code subagent implements the bounded Change Surface.
3. The parent reviews the subagent diff, runs all required tests, and performs
   database rehearsals.
4. After verification, one fresh independent reviewer subagent performs the
   Independent Code Review.
5. Subagents receive the complete plan path, relevant repository instructions,
   exact file boundary, and explicit database safety rules.
6. Neither subagent receives authority to access production data, change
   configuration, or perform a target apply.
7. If native subagent execution is unavailable, the parent stops and requests
   user approval for a parent-only fallback before production-code edits.

## Progress Checklist

- [x] User confirmed `kazusa_core_v2` as the new production candidate.
- [x] User confirmed `kazusa_bot_core` remains unchanged as the backup/source.
- [x] User confirmed Stage 4 contains no application cutover.
- [x] User retained ownership of configuration and production activation.
- [x] Critical and excluded data lanes are classified.
- [x] Affinity conversion and V2 relationship precedence are frozen.
- [x] Bounded incompatible relationship-insight handling is frozen.
- [x] Source high-water, drift, idempotency, and resume rules are frozen.
- [x] Exact implementation files, commands, tests, and evidence are defined.
- [ ] Current plan passes independent plan review.
- [ ] User approves the independently reviewed plan for implementation.
- [ ] User authorizes migration implementation.
- [ ] Failing deterministic and CLI test contract is recorded.
- [ ] Offline maintenance implementation is complete.
- [ ] Isolated DEV MongoDB rehearsal passes.
- [ ] User authorizes read-only `kazusa_bot_core` discovery.
- [ ] Production discovery and source manifest complete without blockers.
- [ ] User approves discovery and authorizes bounded relationship dry-run.
- [ ] Conversion plan is frozen with `apply_ready=True`.
- [ ] User authorizes creation and writes to `kazusa_core_v2`.
- [ ] Target construction completes.
- [ ] Verification reports `candidate_ready=True`.
- [ ] Independent code review passes.
- [ ] User accepts Stage 4 database-construction evidence.
- [ ] Plan is marked completed and archived.

The execution owner updates this checklist after each stage, records evidence
paths, and rereads the full plan before the next stage.

## Verification

### Documentation checks

Run after every plan edit:

```powershell
git diff --check
$terms = @('T' + 'BD', 'T' + 'ODO', 'choose ' + 'one', 'option ' + 'A', 'option ' + 'B')
Select-String -LiteralPath development_plans/active/short_term/cognition_core_v2_stage_4_production_database_migration_plan.md -Pattern $terms
```

The placeholder scan must return no matches.

### Deterministic and CLI tests

Run with the project virtual environment:

```powershell
venv\Scripts\python -m pytest tests/test_stage4_database_migration.py tests/test_stage4_database_migration_cli.py -q
```

Required test cases include:

- literal source/target names and distinct database handles;
- source collection handles expose no write path;
- empty-target preflight and explicit write acknowledgement;
- native character/user V2 state precedence;
- exact affinity boundary values and rounding;
- every acquaintance default;
- LLM axis allowlist, evidence-handle ownership, bound normalization, parser
  failure, unsupported output, and one-call budget;
- duplicate account ownership blocker;
- synthetic identity rule;
- active/terminal user-memory classification;
- shared-memory head/ancestor selection and orphan blocker;
- current and recognized legacy conversation transformation;
- binary, trace, delivery, and source-embedding exclusion;
- task/calendar/background/reflection/log/trace report-only disposition;
- source manifest fingerprint and drift detection;
- batch checkpoint and digest-locked resume;
- forbidden legacy target fields;
- redacted shareable artifacts.

### Isolated MongoDB integration

The integration suite receives a test MongoDB URI through the explicit process
environment defined by the test harness. It generates source and target names
that cannot equal `kazusa_bot_core` or `kazusa_core_v2`.

Run:

```powershell
venv\Scripts\python -m pytest tests/test_stage4_database_migration_integration.py -q
```

It verifies:

- discover and dry-run make zero writes;
- apply changes only the generated test target;
- source document and index digests remain unchanged;
- a complete apply and interrupted/resumed apply converge;
- a second apply is idempotent;
- canonical validators accept every target row;
- ordinary and vector indexes match the handle-explicit bootstrap contract;
- expected and actual target counts match;
- operational target lanes contain zero documents.

### Production-connected verification

Production-connected verification occurs only after its separate authority
gate. It must prove:

1. `kazusa_bot_core` still exists and all Stage 4 operations against it were
   reads.
2. Source-manifest rows used by apply match their frozen fingerprints.
3. Post-manifest source creations and changes appear in drift counts.
4. `kazusa_core_v2` has exactly one valid character singleton.
5. Every migrated user state is valid and owner-matching.
6. Every platform account has exactly one target owner.
7. Every active user memory belongs to a migrated canonical user.
8. Every migrated shared-memory head has its required ancestor chain.
9. Every conversation row has canonical identity, typed shape, timestamp, and
   usable text.
10. Every embedding has the current configured dimension.
11. All current ordinary and vector indexes exist.
12. Empty operational lanes contain zero documents.
13. Target forbidden-field counts are zero for:
    `affinity`, `last_relationship_insight`, `mood`, `global_vibe`,
    `reflection_summary`, raw wire fields, delivery bookkeeping, trace
    references, and source embeddings.
14. Expected target counts equal actual target counts.
15. Discovery and verification blockers are empty.
16. `candidate_ready=True`.

Application startup, adapter delivery, browser validation, production LLM
behavior, deployment, restart, and database-selection checks are deliberately
absent because they belong to the user-owned later transition.

## Independent Code Review

After implementation and verification, a fresh reviewer receives:

- this complete plan;
- `AGENTS.md`;
- the full scoped diff;
- deterministic, CLI, and integration test output;
- redacted discovery, conversion, checkpoint, and verification summaries;
- target index and count reconciliation;
- source drift and volatile-work handoff summaries.

The reviewer must answer:

1. Can any code path write to `kazusa_bot_core`?
2. Can configured runtime database selection affect the explicit target
   handle?
3. Can a name, acknowledgement, target-empty, digest, or drift guard be
   bypassed?
4. Does any transform invent relationship semantics?
5. Can affinity influence an axis other than `positive_regard`?
6. Can invalid LLM output or an unverified evidence handle enter target state?
7. Are all omitted data lanes represented in dispositions or handoff counts?
8. Can resume duplicate, overwrite inconsistently, or skip a row?
9. Are source embeddings, caches, logs, traces, and transient executor state
   excluded?
10. Are artifact contents bounded and protected?
11. Do changes stay inside the declared Change Surface?
12. Does the implementation stop before every deferred production action?

Every critical or high finding blocks completion. Medium findings require
resolution or explicit user acceptance. Low findings are recorded.

## Acceptance Criteria

Stage 4 is complete only when:

1. the plan has independent review and explicit user approval;
2. implementation and all required tests pass;
3. isolated DEV rehearsal proves source isolation, idempotency, and resume;
4. the user separately authorizes each production-connected phase;
5. discovery classifies every source row and has no unresolved blocker;
6. the frozen conversion plan has `apply_ready=True`;
7. `kazusa_bot_core` remains unchanged by Stage 4;
8. `kazusa_core_v2` contains the exact selected durable data in native V2
   shapes;
9. existing valid owner-matching V2 states win;
10. legacy affinity affects only `positive_regard` by the approved formula;
11. every other reconstructed relationship axis has separate validated
    evidence or remains at the acquaintance default;
12. malformed critical rows have an explicit approved disposition;
13. incompatible optional and excluded lanes have reconciled counts;
14. embeddings and indexes are rebuilt successfully;
15. verification reports zero schema, integrity, forbidden-field, index, and
    embedding failures;
16. the source interval and later drift are explicit;
17. the volatile-work handoff report is delivered;
18. independent code review has no unresolved blocking finding;
19. `candidate_ready=True`;
20. the user receives the evidence packet with a clear statement that
    application configuration and cutover remain pending under user control.

Completion of this plan means the database candidate is ready for the user's
later production transition. It does not mean the running application uses
`kazusa_core_v2`.

## Risks

| Risk | Control | Completion evidence |
|---|---|---|
| Source is accidentally mutated | Read-only source facade, literal-name guard, integration proof | Source operation audit and unchanged source digest |
| Target is mistaken for an active cutover | Explicit runtime boundary and unchanged configuration surface | Scoped diff and handoff statement |
| Live source changes during construction | Frozen row manifest, per-collection high water, fingerprint checks, drift report | Verification drift counts |
| Legacy affinity overstates relationship | One-axis formula and canonical defaults | Boundary tests and target-field review |
| Prose insight fabricates relationship state | Bounded evidence handles, exact output contract, deterministic validation, fallback | LLM decision audit counts |
| Duplicate identities corrupt ownership | Duplicate account preflight blocker | Zero duplicate target ownership |
| Critical data is silently lost | Blocking critical disposition and count reconciliation | Zero unresolved critical blockers |
| Terminal or operational state creates stale work | Explicit report-only disposition and handoff report | Empty target lane counts |
| Source vectors are incompatible | Full regeneration and dimension validation | Zero embedding failures |
| Resume mixes revisions or source images | Digest-locked checkpoint | Resume rejection and convergence tests |
| Migration artifacts expose private content | Protected local inputs and redacted shareable reports | Artifact schema tests |
| Target activation exceeds Stage 4 authority | Halt before configuration or service actions | Scoped diff and final handoff |

## Execution Evidence

- 2026-07-28: user confirmed the new database construction boundary,
  preservation of `kazusa_bot_core`, absence of Stage 4 application cutover,
  DEV-only workspace boundary, user ownership of production activation, and
  the detailed conversion rules captured above.
- 2026-07-28: plan expanded from a lifecycle reservation into an executable
  draft.
- No production database connection, source discovery, diagnostic export,
  target creation, migration implementation, database write, configuration
  change, deployment, restart, or cutover was performed by this plan update.
