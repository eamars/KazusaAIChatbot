# Cognition Core V2 Stage 4 One-Off Database Migration Plan

## Summary

- Goal: perform one one-off data migration from `kazusa_bot_core` into the new
  native V2 database `kazusa_core_v2`.
- Plan class: `high_risk_migration`.
- Status: `superseded`.
- Approval: the user approved this database-migration-only boundary on
  2026-07-28 and waived independent plan and code review.
- Source: `kazusa_bot_core`, retained unchanged as the complete pre-migration
  backup.
- Target: `kazusa_core_v2`, created and populated as the user's future
  production database.
- Application boundary: the user owns database selection, configuration,
  deployment, restart, final delta handling, activation, and cutover.
- Repository boundary: Stage 4 changes the database only. Repository source,
  workflow, bootstrap, configuration, tests, and runtime behavior stay
  unchanged.
- Implementation principle: use temporary, uncommitted migration mechanics
  only. Stage 4 creates no reusable migration framework, CLI, runtime router,
  compatibility layer, or persistent maintenance feature.

## Confirmed Ownership Boundary

Stage 4 owns:

1. read-only discovery of `kazusa_bot_core`;
2. classification and conversion of the selected durable data;
3. creation and population of `kazusa_core_v2`;
4. regeneration of target embeddings and indexes;
5. direct database verification and a bounded migration report.

The user owns:

1. application configuration and database selection;
2. production deployment, stop, restart, and health validation;
3. reconciliation of writes made after the migration source image;
4. activation of `kazusa_core_v2`;
5. retention, archival, or later removal of `kazusa_bot_core`.

## Change Surface

Tracked repository changes are limited to:

- this plan;
- its status and execution-evidence row in `development_plans/README.md`.

Database activity is limited to:

- reads from `kazusa_bot_core`;
- collection, index, and document creation or writes in `kazusa_core_v2`.

Temporary execution files may be created under
`test_artifacts/cognition_core_v2/stage4/<run_id>/`. They remain uncommitted and
contain only the evidence needed to execute or verify this one migration.

## Mandatory Safety Rules

1. Source database name must equal `kazusa_bot_core`.
2. Target database name must equal `kazusa_core_v2`.
3. Source and target handles must be distinct.
4. Every source operation is read-only.
5. Every database write targets `kazusa_core_v2`.
6. Fresh migration starts only when the target is absent or every existing
   target collection contains zero documents.
7. Stage 4 drops no database or collection.
8. Stage 4 changes no application setting, environment file, service,
   deployment definition, process, adapter, scheduler, worker, or workflow.
9. Critical malformed data blocks target completion.
10. Optional incompatible data is omitted with a counted reason.
11. Source embeddings and derived caches are discarded and rebuilt.
12. The migration records its source interval and later detected source drift.
13. A repeated write verifies the existing target row against the planned row
    before treating it as complete.
14. A digest, source-row, database-name, or target-state mismatch stops the
    migration before the next write.
15. Concurrent workspace changes remain outside Stage 4 and are preserved.

## Data To Migrate

### Character continuity

Create exactly one target `character_state` singleton with `_id: "global"`.

Migrate:

- the validated static character profile;
- a structurally valid `self_image`;
- the source character `cognition_state` when it is a valid
  `cognition_state.v2` character state;
- otherwise the exact current V2 character default from the canonical builder.

Exclude:

- `mood`;
- `global_vibe`;
- `reflection_summary`;
- legacy affect prose;
- unknown top-level runtime fields.

A missing or invalid required static profile blocks completion. An invalid
optional self-image is omitted and counted. The current self-image storage
contract is an open dictionary, so the migration accepts a non-empty,
JSON-compatible mapping with bounded nesting and size and copies it without
semantic rewriting.

### Canonical user identity

Migrate one target `user_profiles` row per canonical `global_user_id` with:

- valid platform accounts;
- current display names stored on those accounts;
- valid aliases that reference another migrated canonical user;
- one valid owner-matching `cognition_state.v2` user state.

The reserved character identity may have no platform account. Every other
migrated identity requires at least one valid account with non-empty
`platform` and `platform_user_id`.

If one `(platform, platform_user_id)` pair belongs to multiple canonical user
IDs, all affected identities are excluded and completion is blocked for user
disposition. The migration performs no automatic identity merge.

Legacy `facts`, `affinity`, `last_relationship_insight`, and profile embeddings
do not enter target user-profile rows. Durable facts migrate through valid
active `user_memory_units`.

### User relationship conversion

For each non-character user:

1. A complete valid V2 user state wins when
   `owner_user_id == global_user_id`.
2. Otherwise build the canonical acquaintance state.
3. A finite numeric legacy affinity maps only to `positive_regard`:

   ```text
   positive_regard = clamp(round((affinity - 500) / 5), -100, 100)
   ```

4. Boolean, non-numeric, NaN, and infinite affinity values are omitted and
   counted.
5. Every other relationship axis stays at the acquaintance default unless
   separate memory or conversation evidence supports that exact axis.

The acquaintance defaults are:

| Axis | Value |
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

Affinity never changes `familiarity`, `trust`, `attachment`,
`desired_closeness`, `perceived_closeness`, `care`, `boundary_safety`,
`exclusivity`, `unresolved_injury`, or `salience`.

### Incompatible relationship insight

`last_relationship_insight` may support a one-off bounded reconstruction only
when:

- a valid owner-matching V2 state is absent;
- the insight contains bounded non-empty text;
- at least one valid active memory unit or usable conversation row is
  available as separate evidence.

The existing `COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL` route may receive:

- the qualitative affinity-derived positive-regard band;
- at most 2,000 characters of relationship insight;
- at most eight active memory entries;
- at most twelve recent conversation excerpts;
- the active style overlay when available;
- opaque handles for every evidence item.

The one semantic call may propose only:

- `familiarity`;
- `trust`;
- `attachment`;
- `desired_closeness`;
- `perceived_closeness`;
- `care`;
- `boundary_safety`;
- `exclusivity`;
- `unresolved_injury`;
- `salience`.

Every accepted axis update must cite at least one supplied memory or
conversation handle. Unknown axes, keys, types, handles, duplicate axes, or
empty summaries invalidate the result. Values are clamped to their native V2
intervals. `positive_regard` always remains the deterministic affinity result.
Invalid, unsupported, or absent output falls back to affinity-derived
`positive_regard` plus acquaintance defaults. The fallback is counted and
does not block migration.

### User memory

Migrate structurally valid active `user_memory_units`, including active
commitments, for migrated canonical users.

Preserve:

- `unit_id`, owner, unit type, fact, appraisal, relationship signal;
- active lifecycle status and count;
- valid first-seen, last-seen, update, and due timestamps;
- valid source references and merge history.

Terminal archived, completed, and cancelled units remain in
`kazusa_bot_core`. Source embeddings are regenerated in the target.

### Shared memory

Migrate each structurally valid active shared-memory lineage head and every
valid ancestor required for that head's provenance.

Leave rejected, expired, unrelated superseded branches, and unusable orphan
rows in the source. A selected active head with a broken required ancestor
chain blocks completion. Regenerate embeddings.

### Conversation history

Migrate every usable user or assistant conversation row belonging to a
migrated canonical identity.

For a native row, retain `body_text`. For a recognized legacy row, convert
`content` to `body_text`.

Preserve:

- platform, channel identifier, and channel type;
- user or assistant role;
- canonical global identity and display name;
- content type and normalized message text;
- timestamp and valid platform message ID;
- valid typed addressee, mention, and reply metadata;
- attachment `media_type` and normalized `description`.

Legacy rows without typed addressing metadata receive empty typed lists,
absent reply context, and `broadcast=False`. The migration does not parse raw
platform syntax to invent typed metadata.

Exclude:

- base64 or binary attachment payloads;
- attachment URLs, sizes, and storage metadata;
- raw wire text;
- delivery bookkeeping;
- LLM trace references;
- source embeddings;
- unknown legacy fields.

A selected row missing valid platform, channel, channel type, role, canonical
identity, timestamp, or usable text is critical malformed data and blocks
completion.

### Interaction style and character growth

Migrate valid active `interaction_style_images` using the current overlay
validator and one current row per stable `style_image_id`.

Migrate valid active `global_character_growth_traits`. Preserve stable trait
identity, axis, guidance, strength, maturity, evidence identifiers, version,
and timestamps. Growth run history stays in the source.

Invalid style and growth rows are optional omissions and are counted.

## Data To Leave In `kazusa_bot_core`

The following target collections start empty:

- conversation episode state;
- internal monologue residue state;
- internal action latches;
- post-turn lifecycle records;
- accepted-task executor state;
- background-work jobs;
- calendar schedules and runs;
- legacy scheduled events;
- raw reflection runs;
- self-cognition attempt/review ledgers;
- global-character-growth runs;
- persistent RAG caches;
- event logs and snapshots;
- LLM traces.

Active or undelivered tasks, background work, schedules, runs, deliveries, and
leases are summarized by hashed identifier, status, due time, and update time
for the user's later handling. Their prompts, generated content, and user text
stay out of the report.

## One-Off Execution Procedure

### Step 1 — Preflight

1. Verify exact source and target names and distinct handles.
2. Verify source access is read-only.
3. Verify the target is absent or contains zero documents in every collection.
4. Record source collection names, counts, indexes, and observed variants.
5. Stop on duplicate account ownership or unknown critical variants.

### Step 2 — Freeze the source image

Scan selected collections in stable `_id` order. Record each selected row's
collection, stable identifier, update timestamp when present, and canonical
SHA-256 fingerprint after excluding embedding arrays.

Record per-collection scan start/end times, row count, highest observed stable
identifier, and manifest digest. Raw conversation and memory text remain in
protected temporary inputs and stay out of shareable summaries.

### Step 3 — Build the conversion set

Apply the conversion rules above in memory. Validate every target document
against the current native contract or the narrow structural rule defined in
this plan. Freeze relationship decisions and expected target counts.

Any critical blocker stops before target creation. Optional omissions are
counted by collection and reason.

### Step 4 — Create and populate `kazusa_core_v2`

Create current target collections and indexes directly as a one-off database
operation. Write stable-key batches in this order:

1. character singleton;
2. reserved character identity and canonical users;
3. user cognition states;
4. active user memory;
5. shared-memory lineages;
6. conversation history;
7. active interaction styles;
8. active growth traits;
9. regenerated embeddings;
10. vector indexes.

Before each batch, reread the selected source rows and compare their frozen
fingerprints. A changed or missing row stops that batch. Target writes use
stable native keys and verify an existing equal row for safe continuation.

### Step 5 — Verify the candidate

Verify:

- source documents and indexes remain unchanged by Stage 4;
- expected and actual target counts match;
- exactly one valid character singleton exists;
- every user state is valid and owner-matching;
- every platform account has one owner;
- every memory owner and shared-memory ancestor is valid;
- every conversation row has canonical identity and usable typed content;
- source-only fields and embeddings are absent;
- regenerated embeddings have the current dimension;
- current ordinary and vector indexes exist;
- operational lanes contain zero documents;
- source drift after the frozen image is counted;
- all blockers are empty.

Candidate readiness means the database migration is verified. It does not mean
the application uses `kazusa_core_v2`.

## Halt Conditions

Stop and report when:

- database names or handle identity fail;
- the target contains documents before a fresh migration;
- duplicate account ownership exists;
- critical character, identity, conversation, memory, or state data is
  malformed;
- a required shared-memory ancestor is missing;
- a selected source row drifts before its batch;
- target validation rejects a converted row;
- embedding or index creation fails;
- completion would require application configuration, deployment, restart,
  cutover, source mutation, or target cleanup.

## Evidence

Keep a minimal uncommitted migration packet under:

```text
test_artifacts/cognition_core_v2/stage4/<run_id>/
```

The packet contains:

- source interval and collection counts;
- selected, omitted, blocked, and left-in-source counts;
- hashed critical blocker identifiers;
- expected and actual target counts;
- relationship reconstruction and fallback counts;
- volatile-work handoff counts;
- source drift counts;
- embedding and index verification;
- final candidate readiness.

Shareable evidence excludes credentials, prompts, raw LLM output, conversation
text, memory text, and unrestricted traces.

## Acceptance Criteria

Stage 4 is complete when:

1. `kazusa_bot_core` remains unchanged;
2. `kazusa_core_v2` contains the selected durable continuity data in native V2
   shapes;
3. valid owner-matching V2 states win;
4. legacy affinity affects only `positive_regard` by the approved formula;
5. every other reconstructed axis has separate validated evidence or remains
   at its acquaintance default;
6. critical malformed data has an explicit user disposition;
7. optional omissions and source-only lanes have reconciled counts;
8. embeddings and indexes are rebuilt;
9. target schema, integrity, forbidden-field, embedding, and index checks pass;
10. the source interval and later drift are explicit;
11. candidate readiness is reported;
12. application configuration and cutover remain with the user.

## Progress

- [x] Database-migration-only scope confirmed.
- [x] Source and target identities confirmed.
- [x] Source preservation and user-owned cutover confirmed.
- [x] Critical data and conversion rules confirmed.
- [x] Reusable repository tooling and workflow changes removed from scope.
- [x] Independent review waived by the user.
- [ ] One-off source preflight complete.
- [ ] Frozen source image and conversion set complete.
- [ ] `kazusa_core_v2` constructed.
- [ ] Candidate verification complete.
- [ ] Evidence handed to the user.

## Execution Evidence

- 2026-07-28: user confirmed `kazusa_core_v2` as the new production database
  candidate and `kazusa_bot_core` as the unchanged pre-migration backup.
- 2026-07-28: user confirmed Stage 4 contains no application cutover and
  retained ownership of production database selection and activation.
- 2026-07-28: user confirmed the detailed affinity, relationship,
  incompatible-data, identity, memory, conversation, style, and growth rules.
- 2026-07-28: user rejected reusable migration tooling and defined Stage 4 as
  a one-off database migration only.
- 2026-07-28: plan reduced to the one-off database operation; repository code,
  workflow, bootstrap, configuration, and tests remain outside scope.
