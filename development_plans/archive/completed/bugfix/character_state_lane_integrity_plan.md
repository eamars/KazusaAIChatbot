# character state lane integrity plan

## Summary

- Goal: harden `character_state` so global active-character mood, vibe,
  reflection summary, and self-state can evolve without absorbing user-scoped
  facts, user commitments, group/channel style, shared lore, raw reflection
  output, or stale affect.
- Plan class: high_risk_migration.
- Status: completed.
- Mandatory skills: `development-plan`, `database-data-pull`,
  `local-llm-architecture`, `no-prepost-user-input`, `debug-llm`, `py-style`,
  `cjk-safety`, `test-style-and-execution`.
- Overall cutover strategy: bigbang validation for new character-state writes;
  compatible audit and separately gated repair for the existing singleton row.
- Highest-risk areas: blocking valid invented global character evolution,
  treating user commitments as character traits, allowing group style to become
  global mood, and letting stale mood/vibe steer relevance or L1 cognition.
- Acceptance criteria: every new `character_state` write is targeted at the
  active character, carries state-only payload, preserves explicit freshness,
  rejects wrong-lane data, and is proven by deterministic tests, audit dry-run,
  and live/debug robustness checks.

New-write hardening supersession: all lane-specific new-write hardening,
source-generation proof, and fixing-strategy instructions in this plan are
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.
Execute this lane plan for cleanup, audit, dry-run/apply, data migration, and
post-cleanup verification only. If another section still names new-write
validators, prompts, or tests, treat that text as historical context rather
than execution scope.

## Context

`character_state` is the singleton global active-character lane. It is read into
runtime character profile snapshots, relevance, L1 cognition, L2/L3 cognition
chain inputs, RAG character-profile cache dependencies, and operator surfaces.
It currently stores runtime fields such as `mood`, `global_vibe`,
`reflection_summary`, `self_image`, and `updated_at`.

The 2026-07-02 audit found one structurally valid `character_state` row with
current mood/vibe data. No stored corruption was confirmed in that audit. This
plan exists because wrong-lane writes would be subtle: a document can remain
valid MongoDB and valid Python structure while carrying the wrong semantic
owner.

Current source inspection for this draft found:

- `src/kazusa_ai_chatbot/consolidation/target.py` builds deterministic targets
  and already separates `user`, `group_channel`, `character`, and `internal`
  lanes.
- `validate_write_intent(...)` currently validates target alias and allowed
  lane membership, but it does not validate that `character_state` payload text
  is state-only, fresh, or free of wrong-lane semantics.
- `src/kazusa_ai_chatbot/consolidation/persistence.py` writes
  `mood`, `global_vibe`, and `reflection_summary` through
  `upsert_character_state(...)` after origin policy and target-plan checks.
- `src/kazusa_ai_chatbot/consolidation/reflection.py` already prompts the
  global-state updater not to write user images, relationship memory, or
  targeted facts into mood/vibe, but deterministic code accepts whatever
  payload is returned.
- `src/kazusa_ai_chatbot/reflection_cycle/affect_settling.py` separately
  writes daily affect settling through
  `compare_and_upsert_character_state(...)`, with stale-write protection but no
  shared character-state lane validator.
- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l1.py` and
  `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py` consume
  `mood`/`global_vibe`; bad data here can alter response eligibility and the
  first emotional frame before later cognition sees the turn.

Invented or evolved character state is good data when it is global to the
active character. User-specific or group-specific data in this lane is bad
data even if the text is plausible. The plan must preserve that distinction.

## Lane Analysis Requirements

### Issue Description Based On Deep Analysis

`character_state` is the global active-character affect and self-state lane.
The confirmed current state is not known to be corrupted, but the write source
is under-hardened:

- target routing proves that a character lane is eligible for the episode;
- origin policy proves that the source kind may perform durable writes;
- prompt instructions ask the LLM to produce state-only text;
- persistence writes the returned strings without a lane-specific payload
  classifier, freshness contract, or audit trail for rejected candidates.

The failure is therefore a lane-integrity risk, not a schema-only bug. The
system can write a valid document that later makes relevance, L1, L2, L3, RAG
profile reads, and operator views behave as if a user fact, commitment,
channel behavior rule, shared lore row, or raw reflection output were part of
the active character's global state.

### Plan To Remove Malformed Data

Default data action is read-only audit. The existing singleton row is exported
and classified before any live change. Do not delete or rewrite it merely
because it contains evolved character mood or invented self-state.

If the audit classifies fields as concrete malformed lane data, the repair path
is:

1. Generate an immutable export of the full current `character_state` singleton
   before repair.
2. Generate a dry-run repair report that lists field-level classifications,
   evidence snippets, proposed field values, unchanged fields, and rollback
   source path.
3. Require explicit user command before apply mode.
4. In apply mode, update only malformed `mood`, `global_vibe`, or
   `reflection_summary` fields; preserve valid global character evolution and
   preserve `self_image` unless a separate approved plan owns self-image
   cleanup.
5. Record the repair report under `test_artifacts/` and leave this plan in
   draft or execution status until the user approves lifecycle changes.

Malformed data is not moved automatically into user memory, shared memory,
group style, or commitments by this plan. Relocation creates new durable
meaning and requires a separate approved lane-specific plan unless the
malformed content is already present in the correct lane.

### RCA Of The Failure Mode

The root cause is a gap between broad write eligibility and lane-specific
semantic eligibility.

Existing deterministic controls answer these questions:

- Is this origin allowed to write durable state?
- Does the current consolidation state have a character target?
- Is `character_state` listed as an allowed lane for that target?
- Does daily affect settling still match the state freshness token?

They do not answer these character-state-specific questions:

- Is the payload about the active character globally, rather than about the
  current user, one relationship, or one channel?
- Is the payload mood/vibe/self-state rather than an accepted ongoing user
  instruction, reminder, commitment, fact, or preference?
- Did an internal/self-cognition source summarize a group scene, reflection
  artifact, participant context, or raw cognition window into global mood?
- Is a transient state fresh enough to steer the next response path?
- Did a shared-lore or memory-evidence item become `reflection_summary`
  instead of remaining evidence?

LLM prompts already carry much of this policy, but the final persistence
boundary lacks a reusable validator and audit classification. The fix belongs
at the character-state lane boundary, not in ad hoc downstream consumers.

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

## Failure Modes And RCA Details

### User-Specific Contamination

Failure mode: a field such as `reflection_summary` says that one specific user
likes, believes, owns, did, requested, or feels something. It may also encode a
relationship claim such as "she trusts Ran more now" when that belongs in
`user_profiles.last_relationship_insight` or `user_memory_units`.

Root cause: the global-state updater reads user-message episode material and
relationship-rich cognition artifacts. Its prompt tells it not to write user
memory, but persistence lacks a final lane-specific classifier.

Impact: L1 and relevance may treat a one-user relationship event as the
background for every future user and group.

### Commitment Contamination

Failure mode: accepted ongoing rules, promises, reminders, due work, address
preferences, suffix/style instructions, or future follow-through are written as
character mood/vibe/reflection summary. Examples include "will remind the user
tomorrow", "must call the user X", or "should keep replies in a certain style".

Root cause: user commitments are LLM-owned semantic decisions emitted through
`future_promises` and later `user_memory_units.active_commitment`. The
character-state writer does not currently prove that state text is not a
commitment-like operational rule.

Impact: commitments can become globally applied personality or mood, bypassing
the accepted-task, calendar, memory-lifecycle, and user-scoped retrieval
contracts.

Architectural constraint: do not add deterministic keyword rules that override
the LLM's commitment channel decision. Use validator rejection and prompt/schema
hardening so the LLM emits the correct channel; persistence may reject a
character-state candidate that is structurally an operational obligation.

### Group-Style Confusion

Failure mode: group-channel pacing, reply density, local jokes, participant
dynamics, channel customs, or group engagement rules are written into global
character state.

Root cause: group review and reflection source packets can carry group scene
digests, participant context, and group-channel style overlays. Target planning
already avoids creating a character target for group-review episodes with a
group target, but user-message group turns can still have both user and
character targets, and the payload has no group-style classifier.

Impact: a style appropriate for one channel can bias private chats and unrelated
groups through relevance and cognition.

### Shared-Lore Confusion

Failure mode: durable world/common-sense memory, official lore, character-world
facts, or reflection-promoted shared memory become `reflection_summary` or
vibe rather than staying in `memory` or promoted reflection context.

Root cause: RAG and promoted reflection evidence can appear in episode state as
evidence for cognition. The global-state updater can summarize the episode
aftermath, but the persistence boundary does not enforce that factual lore
belongs outside `character_state`.

Impact: RAG evidence stops being evidence and starts acting like persona state,
making future cognition over-trust it and making memory provenance harder to
audit.

### Stale Mood And Freshness Drift

Failure mode: old mood/vibe survives as if current, empty strings preserve old
values while still advancing `updated_at`, or concurrent writes overwrite a
newer state.

Root cause: daily affect settling uses
`compare_and_upsert_character_state(...)`, but consolidation still calls
`upsert_character_state(...)` without compare-and-set freshness. Empty strings
are treated as "leave unchanged" in the DB helper, which is useful for
partial updates but can mask an extractor that failed to produce fresh state.

Impact: relevance, L1, and L2 receive stale affect and may interpret ordinary
inputs as hostile, intimate, exhausted, or otherwise miscolored.

### Self-Cognition And Reflection Source Confusion

Failure mode: internal-thought transport summaries, raw reflection artifacts,
group activity window metadata, source ids, scheduler state, or a private
self-cognition window are summarized as active-character state.

Root cause: source packets intentionally let self-cognition and reflection run
through the shared cognition/consolidation path. Prompts distinguish
`internal_thought`, `reflection_artifact`, and user messages, but validator and
audit logic do not classify source-derived metadata or raw internal evidence
when it appears in persisted state.

Impact: a character may carry an internal maintenance artifact as emotional
truth, or turn an observed group scene into global personal mood.

### Downstream Cognition Impact

Failure mode: corrupted `mood` or `global_vibe` changes future relevance
decisions, L1 emotional appraisal, L2 interpretation, L3 tone packaging, RAG
character-profile cache invalidation behavior, or operator diagnosis.

Root cause: `character_state` is intentionally loaded early and widely because
it is supposed to be compact, global, and trusted. This trust multiplies the
blast radius of wrong-lane writes.

Impact: one bad write can change when the character speaks, how she frames a
message, and what future agents think the active character currently is.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `database-data-pull`: load before exporting, inspecting, or comparing live
  `character_state` data.
- `local-llm-architecture`: load before changing character-state extraction,
  prompt payloads, validation contracts, cognition inputs, RAG projection, or
  background LLM behavior.
- `no-prepost-user-input`: load before changing facts/commitments/persistence
  channel selection for user instructions, preferences, accepted rules, or
  future promises.
- `debug-llm`: load before live character-state generation checks or LLM trace
  review.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files with CJK prompt or state
  strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute this plan while status is `draft`.
- Do not alter live `character_state` without reviewed dry-run evidence and
  explicit user command.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Preserve valid invented/evolved global character mood, vibe, and self-state.
- Reject user-specific, group-specific, commitment-like, shared-lore, or raw
  source-metadata content from `character_state`.
- LLM stages own semantic judgment. Deterministic code owns target validation,
  structural shape, freshness checks, persistence mechanics, audit
  classification, and apply safety.
- Do not add compatibility shims, fallback write paths, dual writes, alias
  modules, or hidden repair layers.
- Do not add deterministic keyword gates that override the LLM's accepted
  user-preference or commitment channel decision. Improve prompt/schema output
  and reject invalid character-state payloads structurally.
- Do not feed raw reflection output into normal cognition. Only promoted,
  gated reflection context may enter normal cognition.
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

- Add a canonical character-state validator contract covering target,
  payload shape, lane semantics, field freshness, and audit classifications.
- Wire that validator into consolidation character-state writes, daily
  affect-settling writes, and approved maintenance repair apply.
- Add a read-only character-state lane audit with field-level classifications
  for user-specific contamination, commitment contamination,
  group-style confusion, shared-lore confusion, stale/missing freshness,
  self-cognition/reflection source confusion, malformed structure, and valid
  evolved global state.
- Add dry-run/apply behavior for any repair script, with apply blocked unless
  the user explicitly commands live mutation after reviewing dry-run output.
- Add tests proving valid global invented/evolved state persists while
  wrong-lane candidates are rejected or skipped before DB writes.
- Add affect-settling tests proving stale-write protection remains and the
  shared validator is applied before compare-and-upsert.
- Add live/debug robustness cases that inspect LLM behavior one case at a time.
- Record verification, audit output, review findings, remediation, and
  residual risks in `Execution Evidence`.

## Deferred

- Do not redesign character self-image.
- Do not move `character_state` into shared memory.
- Do not rewrite cognition, dialog, relevance, RAG, or reflection architecture
  except for the validator call sites explicitly listed in this plan.
- Do not enforce original-character canon on character state.
- Do not delete or rewrite the current singleton row unless audit finds
  concrete malformed content and the user explicitly commands apply mode.
- Do not relocate malformed data into user memory, group style, shared memory,
  or commitments under this plan.
- Do not add feature flags for the validator. New write validation cuts over
  directly.
- Do not add a new response-path LLM call.
- Do not preserve legacy unvalidated writes as a fallback.

## Cutover Policy

Overall strategy: bigbang validation for new writes; compatible audit for the
existing row.

| Area | Policy | Instruction |
| --- | --- | --- |
| New consolidation `character_state` writes | bigbang | Require canonical validator acceptance before `upsert_character_state(...)`. No fallback write. |
| New affect-settling writes | bigbang | Require canonical validator acceptance before `compare_and_upsert_character_state(...)`, while preserving stale-token compare-and-set. |
| Existing singleton row | compatible | Audit first. Keep valid evolved state. Repair only concrete malformed fields after explicit user command. |
| Maintenance repair apply | migration | Apply only after export, dry-run report, user command, compare/freshness check, and rollback record. |
| User commitments and preferences | bigbang | Keep channel ownership in LLM prompt/schema and user memory/commitment lanes. Character-state persistence may reject invalid payloads but must not rewrite commitments. |
| Group style | bigbang | Keep group pacing/style in `interaction_style_images` group-channel lane. Do not copy into global mood/vibe. |
| Shared lore | bigbang | Keep shared/world facts in `memory` and promoted reflection context. Do not copy factual lore into `reflection_summary`. |
| Valid personality evolution | compatible | Continue accepting invented global active-character self-state when not scoped to a user, group, commitment, or factual lore. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative compatibility strategy by
  default.
- If an area is `bigbang`, delete or rewrite legacy unvalidated call behavior
  instead of preserving fallback writes.
- If an area is `migration`, follow the exact export, dry-run, apply, evidence,
  and rollback gates in this plan.
- If an area is `compatible`, preserve only the compatibility surfaces listed
  here.
- Any change to a cutover policy requires user approval before implementation.

## Target State

`character_state` stores only global active-character runtime state:

- `mood`: compact current/near-term character affect;
- `global_vibe`: compact background atmosphere that may color future turns;
- `reflection_summary`: brief third-person summary of the active character's
  current psychological residue;
- `self_image`: existing active-character self-image document, unchanged by
  this plan except that lane validation must not weaken its current target
  boundary;
- `updated_at`: storage UTC freshness timestamp for runtime state fields.

`character_state` does not store:

- user preferences, facts, schedules, identity, ownership, relationship claims,
  or per-user continuity;
- user commitments, accepted rules, address preferences, suffix/style
  instructions, future reminders, or delayed work;
- group/channel style, group pacing, channel customs, participant dynamics, or
  response-ratio tuning;
- shared/world lore, public facts, common-sense knowledge, official setting
  rows, or memory-evolution facts;
- raw reflection output, raw self-cognition source packets, source ids,
  scheduler ids, prompt metadata, or transport summaries;
- volatile public/current facts such as prices, schedules, weather, current
  events, or external state.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Current singleton row | Audit first, repair only after explicit approval | No stored corruption is confirmed and valid evolved mood is expected product behavior. |
| Validator ownership | Add a canonical deterministic validator in the character-state/database boundary and call it from all writers | Prevents divergent checks in consolidation, affect settling, and maintenance scripts. |
| Semantic source of truth | LLM prompts/schemas decide facts vs commitments vs state; validator rejects invalid character-state payloads without rewriting channels | Preserves LLM-first user-input interpretation and avoids brittle keyword post-processing. |
| Freshness | Preserve `updated_at` and require explicit freshness semantics for every runtime-state write | `mood`/`global_vibe` are transient enough to affect response path behavior. |
| Existing `upsert_character_state` empty-string behavior | Keep only when the validator marks the write as an intentional partial state update | Prevents extractor failure from silently advancing freshness while preserving all old fields. |
| Daily affect settling | Keep compare-and-set stale protection and add shared validator before the compare write | Affect settling already protects concurrency; it still needs lane semantics. |
| Repair relocation | Do not auto-move malformed text into other lanes | Relocation creates new durable meaning and belongs to lane-specific plans. |
| Runtime prompts | Use role-neutral "active character" wording unless runtime `{character_name}` is required | Avoids hardcoding character names into reusable prompts. |

## Contracts And Data Shapes

### Character-State And Character-Target Validator Contract

Create a canonical validator owned by the character-state persistence boundary.
The final file location is chosen during implementation by existing ownership:
prefer `src/kazusa_ai_chatbot/db/character.py` for DB-helper-adjacent
validation or a narrowly named module under `src/kazusa_ai_chatbot/db/` if the
helper would otherwise become too large. Re-export only if runtime callers need
the public facade.

Required public contract:

```python
class CharacterStateValidationFinding(TypedDict):
    issue_code: str
    severity: Literal["keep", "warn", "reject", "repair"]
    field_name: str
    issue_description: str
    evidence_preview: str


class CharacterStateValidationResult(TypedDict):
    accepted: bool
    sanitized_payload: dict[str, str]
    findings: list[CharacterStateValidationFinding]
```

Required validator inputs:

```python
def validate_character_state_payload(
    *,
    payload: Mapping[str, object],
    source: Literal["consolidation", "affect_settling", "maintenance_repair"],
    target_kind: str,
    target_id: Mapping[str, str],
    expected_updated_at: str | None,
    storage_timestamp_utc: str,
    allow_partial: bool,
) -> CharacterStateValidationResult: ...
```

Required accepted payload shape:

```python
{
    "mood": str,
    "global_vibe": str,
    "reflection_summary": str,
    "updated_at": str,
}
```

Validator rules:

- `target_kind` must be `character`.
- `target_id.character_id` must be non-empty and must not be a synthetic user
  source label such as `self_cognition`, `internal_thought`,
  `group_chat_review`, `system`, `group`, or `group_channel`.
- `mood`, `global_vibe`, and `reflection_summary` must be strings after
  structural normalization. `None`, dictionaries, lists, booleans, and numbers
  are rejected.
- `mood` and `global_vibe` must stay compact. Use the existing prompt contract
  as the guide: short free text, not long explanations.
- `reflection_summary` may be longer than `mood`/`global_vibe`, but it must
  remain one bounded summary of the active character's psychological residue.
- `updated_at` must be a non-empty storage UTC timestamp supplied by the
  runtime caller.
- When `source="affect_settling"`, `expected_updated_at` must be present and
  the caller must still use compare-and-set persistence.
- When `allow_partial=False`, an empty output for all three state fields is
  rejected. When `allow_partial=True`, empty fields may mean "preserve existing
  value" only if at least one state field is non-empty and freshness evidence is
  valid.
- The validator may return `reject` findings for payloads whose structure
  clearly encodes commitments, per-user facts, group/channel style, shared
  factual lore, raw source metadata, or volatile current facts. It must not
  rewrite those fields into another lane.

### Audit Finding Contract

Create a read-only audit helper under `db.script_operations` and a script under
`src/scripts`.

```python
class CharacterStateLaneAuditFinding(TypedDict):
    character_state_id: str
    field_name: str
    issue_code: Literal[
        "valid_global_character_state",
        "user_specific_contamination",
        "commitment_contamination",
        "group_style_confusion",
        "shared_lore_confusion",
        "volatile_public_fact",
        "stale_or_missing_freshness",
        "self_cognition_source_confusion",
        "reflection_source_confusion",
        "raw_metadata_leak",
        "malformed_structure",
    ]
    severity: Literal["keep", "warn", "manual_review", "repair"]
    issue_description: str
    evidence_fields: dict[str, object]
    recommended_action: Literal["keep", "manual_review", "repair"]
```

Audit classifications:

- `keep`: valid global active-character state, including invented/evolved mood
  or self-state.
- `warn`: structurally valid but stale, overly long, ambiguous, or weakly
  source-grounded; no automatic repair.
- `manual_review`: likely wrong-lane but not safe for automatic field rewrite.
- `repair`: concrete malformed structure or wrong-lane text with a safe field
  replacement plan approved by the user.

### Dry-Run And Apply Behavior

Script contract:

```powershell
venv\Scripts\python.exe -m scripts.audit_character_state_lane --output test_artifacts\character_state_lane_audit.json
venv\Scripts\python.exe -m scripts.audit_character_state_lane --dry-run-repair --output test_artifacts\character_state_lane_repair_dry_run.json
venv\Scripts\python.exe -m scripts.audit_character_state_lane --apply-repair --repair-file test_artifacts\character_state_lane_repair_dry_run.json
```

Required behavior:

- Default mode is read-only audit.
- `--dry-run-repair` writes a proposed repair report and does not mutate DB.
- `--apply-repair` requires an existing dry-run repair file, verifies it still
  targets `_id="global"`, verifies current `updated_at` still matches the
  dry-run source row, writes only approved fields, records before/after
  previews, and refuses if the source row changed.
- Apply mode must not run from live service code.

## LLM Call And Context Budget

Affected background LLM paths:

| Path | Before | After | Response path? | Context impact |
| --- | --- | --- | --- | --- |
| consolidation `global_state_updater` | 1 `CONSOLIDATION_LLM` call | same call count; prompt/schema may be tightened | No | No cap increase; same state payload with clearer state-only contract. |
| consolidation `relationship_recorder` and facts harvester | unchanged | unchanged except tests may cover channel separation | No | No new prompt inputs. |
| daily affect-settling proposal | 1 `CONSOLIDATION_LLM` call | same call count; prompt may mention lane ownership/freshness more explicitly | No | No cap increase; prompt remains under existing `AFFECT_SETTLING_PROMPT_MAX_CHARS`. |
| daily affect-settling review | 1 `CONSOLIDATION_LLM` call | same call count; review may reject wrong-lane affect proposals | No | No cap increase; prompt remains under existing review cap. |
| live persona response | no new call | no new call | Yes, but unchanged | No latency increase. |

No new response-path LLM call is allowed. New deterministic validation must not
send raw database rows or raw metadata to an LLM. Prompt text changes must use
role-neutral "active character" wording unless runtime `{character_name}` is
required.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/consolidation/target.py`
  - Extend write-intent validation or target metadata so character target ids
    are validated as active-character targets before character lanes are used.
  - Add tests for character target acceptance and rejection.
- `src/kazusa_ai_chatbot/consolidation/persistence.py`
  - Call the canonical character-state validator before
    `upsert_character_state(...)`.
  - Record validation findings in `metadata` without exposing raw user content
    beyond bounded previews.
  - Skip DB writes and Cache2 invalidation when validation rejects the payload.
- `src/kazusa_ai_chatbot/consolidation/reflection.py`
  - Tighten the global-state updater prompt/schema only as needed to reinforce
    state-only output, source distinction, and no commitments/user facts/group
    style/shared lore in `mood`, `global_vibe`, or `reflection_summary`.
- `src/kazusa_ai_chatbot/reflection_cycle/affect_settling.py`
  - Call the canonical validator before compare-and-upsert.
  - Preserve existing dry-run, disabled-write, stale-state, and refresh
    behavior.
- `src/kazusa_ai_chatbot/db/character.py`
  - Add or call validator logic near the existing character-state helpers.
  - Preserve the existing singleton shape and existing public DB helper names.
- `src/kazusa_ai_chatbot/db/script_operations.py`
  - Add read-only audit and approved repair helper functions for maintenance
    scripts.
- `src/scripts/README.md`
  - Document the new audit script only if the script is created.

### Create

- `src/scripts/audit_character_state_lane.py`
  - Operator script for read-only audit, dry-run repair, and apply repair.
- `tests/test_character_state_lane_integrity.py`
  - Focused validator, audit, db-writer, and affect-settling coverage.

### Keep

- Existing `character_state` collection and singleton `_id="global"` shape.
- Existing `mood`, `global_vibe`, `reflection_summary`, `self_image`, and
  `updated_at` fields.
- Existing cognition/dialog surface behavior.
- Existing daily affect-settling compare-and-set freshness behavior.
- Existing RAG Cache2 dependency source name `character_state`.

## Detailed Fixing Strategy

The lane-specific fixing strategy for preventing future writes is superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

Retain only the cleanup, audit, migration, and post-cleanup verification actions
already defined in this lane plan. If implementation finds a new-write hardening
need outside the consolidator router plan, stop and update the plan instead of
executing the superseded fixing strategy.


## Overdesign Guardrail

- Actual problem: no stored corruption is confirmed, but new writes can still
  persist wrong-lane character-state text because the character lane lacks a
  payload validator and audit/repair contract.
- Minimal change: add one canonical validator, wire it into existing write
  paths, add a read-only audit plus gated repair script, and add focused tests.
- Ownership boundaries: LLM stages produce semantic state proposals;
  deterministic target planning selects eligible durable targets;
  deterministic validator classifies payload structure/lane/freshness;
  DB helpers persist only validated state; cognition consumes only the trusted
  singleton; maintenance scripts own dry-run/apply repair.
- Rejected complexity: no new character-state schema, no replacement memory
  model, no response-path LLM call, no compatibility fallback, no automatic
  relocation to other lanes, no canon enforcement, no per-user character-state
  documents, no group-specific global mood, no feature flag to keep
  unvalidated writes.
- Evidence threshold: schema redesign or a second state lane requires repeated
  valid global state that cannot fit `mood`, `global_vibe`, or
  `reflection_summary`, plus an approved follow-up plan.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside the target modules as
  high-scrutiny changes. Updating an existing module outside `consolidation`,
  `db`, `reflection_cycle`, `src/scripts`, or focused tests requires strong
  justification in `Execution Evidence` before implementation continues.
- The responsible agent may implement helper functions only after searching for
  equivalent behavior. If an equivalent helper exists, reuse or move it instead
  of duplicating it.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, preserve the plan's stated lane-integrity
  intent and record the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.
- If validation would block clearly valid global character evolution, stop and
  adjust the validator contract in this plan before changing code.

## Implementation Order

1. Parent adds focused failing tests in
   `tests/test_character_state_lane_integrity.py`.
2. Parent runs the focused test file and records expected failures.
3. Parent starts exactly one production-code subagent with this approved plan,
   the test contract, and the production change surface.
4. Production-code subagent implements validator contract in the DB/character
   boundary.
5. Production-code subagent wires validator into consolidation `db_writer`.
6. Production-code subagent wires validator into daily affect settling.
7. Production-code subagent adds audit and gated repair helpers/script.
8. Parent runs focused tests and loops until validator/write behavior passes.
9. Parent updates integration or regression tests only inside the approved
   change surface.
10. Parent runs full verification commands listed below.
11. Parent runs audit dry-run and records report path when live DB inspection
    is explicitly approved for execution.
12. Parent runs live/debug LLM cases one at a time and records inspected output.
13. Parent starts exactly one independent code-review subagent after planned
    verification passes.
14. Parent remediates review findings inside approved scope and reruns affected
    verification.

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

- [ ] Stage 1 - focused character-state contract tests added.
  - Covers: implementation order steps 1-2.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_character_state_lane_integrity.py -q`
  - Evidence: record expected failures or baseline in `Execution Evidence`.
  - Handoff: next agent starts validator implementation.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - canonical validator implemented.
  - Covers: implementation order step 4.
  - Verify: focused validator tests pass.
  - Evidence: record changed files and validator test output.
  - Handoff: next agent wires consolidation.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - consolidation write hardening complete.
  - Covers: implementation order step 5.
  - Verify: db-writer focused tests and Cache2 invalidation tests pass.
  - Evidence: record write-skip and valid-write test output.
  - Handoff: next agent wires affect settling.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - affect-settling write hardening complete.
  - Covers: implementation order step 6.
  - Verify: affect-settling focused tests pass.
  - Evidence: record stale-state and validator-reject test output.
  - Handoff: next agent adds audit script.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - audit and dry-run/apply repair tooling complete.
  - Covers: implementation order step 7.
  - Verify: script unit tests and dry-run command shape pass.
  - Evidence: record audit output path or dry-run fixture output.
  - Handoff: next agent runs regression gates.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - deterministic regression verification complete.
  - Covers: implementation order steps 8-10.
  - Verify: all deterministic commands in `Verification`.
  - Evidence: record command outputs and any allowed static-grep matches.
  - Handoff: next agent runs live/debug cases if approved.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 7 - audit dry-run and live/debug robustness recorded.
  - Covers: implementation order steps 11-12.
  - Verify: audit report reviewed; live/debug LLM outputs inspected one case at
    a time.
  - Evidence: record report paths, trace ids, and inspected conclusions.
  - Handoff: next agent starts independent code review.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 8 - independent code review completed and remediated.
  - Covers: implementation order steps 13-14.
  - Verify: rerun affected tests and static checks after fixes.
  - Evidence: record findings, fixes, rerun commands, residual risks, and
    approval status.
  - Handoff: plan can be signed off only when all findings are resolved.
  - Sign-off: `<agent/date>` after review approval and evidence are recorded.

## Verification

### Static Greps

- `rg -n "upsert_character_state|compare_and_upsert_character_state" src tests`
  - Expected: all production write call sites either run through the canonical
    validator or are snapshot/restore/export utilities explicitly documented
    as maintenance-only.
- `rg -n "character_state_validation|validate_character_state_payload" src tests`
  - Expected: validator implementation, consolidation call site,
    affect-settling call site, maintenance repair call site, and tests.
- `rg -n "character_state.*fallback|unvalidated.*character_state|legacy.*character_state" src`
  - Expected: no production fallback that preserves unvalidated writes. Exit
    code 1 from `rg` is acceptable when there are zero matches.

### Deterministic Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_character_state_lane_integrity.py -q
venv\Scripts\python.exe -m pytest tests/test_consolidation_target_routing.py -q
venv\Scripts\python.exe -m pytest tests/test_consolidation_origin_policy.py -q
venv\Scripts\python.exe -m pytest tests/test_consolidator_origin_policy_db_writer.py -q
venv\Scripts\python.exe -m pytest tests/test_db_writer_cache2_invalidation.py -q
venv\Scripts\python.exe -m pytest tests/test_reflection_affect_settling.py -q
venv\Scripts\python.exe -m pytest tests/test_db.py::test_upsert_character_state tests/test_db.py::test_compare_and_upsert_character_state_matches_updated_at tests/test_db.py::test_compare_and_upsert_character_state_returns_false_when_stale tests/test_db.py::test_upsert_character_state_preserves_on_empty_string -q
venv\Scripts\python.exe -m pytest tests/test_user_profile_agent.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_prompt_contract_text.py::test_l1_subconscious_payload_passes_character_state_in_human_json -q
```

### Audit Dry-Run

Run only during approved execution and after loading `database-data-pull`:

```powershell
venv\Scripts\python.exe -m scripts.audit_character_state_lane --output test_artifacts\character_state_lane_audit.json
venv\Scripts\python.exe -m scripts.audit_character_state_lane --dry-run-repair --output test_artifacts\character_state_lane_repair_dry_run.json
```

Expected:

- default audit performs no writes;
- dry-run repair performs no writes;
- reports classify each field as `keep`, `warn`, `manual_review`, or `repair`;
- valid invented global character state is not classified as malformed.

### Apply Repair Gate

Apply mode is not part of normal verification. It may run only after explicit
user command:

```powershell
venv\Scripts\python.exe -m scripts.audit_character_state_lane --apply-repair --repair-file test_artifacts\character_state_lane_repair_dry_run.json
```

Expected:

- refuses when the repair file is missing, stale, or not for `_id="global"`;
- refuses when current `updated_at` differs from the dry-run source token;
- writes only approved fields;
- records before/after previews and modified count.

### Live/Debug Robustness Cases

Run live LLM cases one at a time with output inspected:

- valid invented global mood persists;
- valid invented global self-state/reflection summary persists;
- user preference is not written to `character_state`;
- accepted address/style rule becomes commitment/user lane evidence, not
  character mood;
- user future reminder/commitment is rejected from `character_state`;
- one-user relationship claim is rejected from `character_state`;
- group-channel style/pacing is rejected from `character_state`;
- shared lore/common-sense memory is rejected from `character_state`;
- volatile public/current fact is rejected from `character_state`;
- self-cognition group-review source packet metadata is rejected from
  `character_state`;
- raw reflection output/source ids do not enter normal `character_state`;
- stale affect-settling write is skipped without overwriting newer state.

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not draft the plan. If no separate reviewer is available, the drafting
agent must reread the plan registry, this plan, relevant READMEs, and source
context from a fresh-review posture.

Review scope:

- The plan keeps `Status: draft` until explicitly approved.
- The issue description, plan to remove malformed data, RCA, source hardening,
  and robustness proof remain explicit.
- The plan preserves valid invented/evolved global character state.
- The plan rejects user-specific, group-specific, commitment, shared-lore,
  stale/freshness, self-cognition, reflection-source, and downstream cognition
  risks.
- The change surface is limited to consolidation, db, reflection-cycle,
  scripts, and focused tests.
- The validator contract does not add deterministic semantic rewrites over user
  commitments or accepted preferences.
- Dry-run/apply behavior and data migration safety are concrete.
- Verification gates map to every failure mode and acceptance criterion.

Record blockers, non-blocking findings, fixes, residual risks, and approval
status in `Execution Evidence`.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Validator design quality: deterministic only, no hidden LLM calls, no
  keyword-based user-commitment rewriting, no compatibility fallback.
- Persistence safety: rejected character-state payloads do not call DB helpers
  or emit Cache2 invalidation; valid state still writes.
- Freshness safety: affect settling keeps compare-and-set; maintenance apply
  verifies source `updated_at`.
- Audit safety: default and dry-run modes do not mutate live DB; apply mode is
  explicitly gated.
- Plan alignment with `Must Do`, `Deferred`, `Change Surface`,
  `Implementation Order`, `Verification`, and `Acceptance Criteria`.
- Regression coverage: tests cover every failure mode listed in this plan.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Character-state writes require an active-character target.
- New consolidation and affect-settling writes pass the canonical validator
  before persistence.
- User facts, commitments, accepted user rules, one-user relationship claims,
  group style, shared lore, volatile public facts, raw reflection output, and
  self-cognition source metadata are rejected from `character_state`.
- Valid invented/evolved global mood, vibe, and self-state remain accepted.
- Stale affect-settling writes cannot overwrite newer state.
- The current singleton row is kept unless audit proves concrete malformed
  content and the user explicitly commands repair apply.
- Dry-run/apply repair behavior is documented, tested, and evidence-backed.
- All deterministic verification commands pass.
- Live/debug robustness cases are inspected and recorded.
- Independent code review findings are resolved or explicitly accepted as
  residual risks.

## Data Migration

Default migration is audit-only.

1. Export `character_state/_id="global"` before any repair decision.
2. Run read-only audit.
3. If every field is `keep` or acceptable `warn`, make no data changes.
4. If any field is `manual_review`, stop and record the finding; do not repair.
5. If any field is `repair`, generate dry-run repair report.
6. Apply repair only after explicit user command and stale-token verification.
7. Preserve `self_image` and all fields not named in the repair report.
8. Store rollback source and apply result under `test_artifacts/`.

Rollback uses the pre-repair export and the apply report. A rollback is itself
a live DB mutation and requires explicit user command.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Valid global evolution blocked | Validator accepts invented/evolved active-character state unless scoped to user/group/commitment/lore | Focused valid-state tests and live/debug valid evolution case |
| User commitments become mood | Prompt/schema hardening plus validator rejection; no rewrite into another lane | Commitment contamination tests and live/debug commitment case |
| Group style becomes global state | Target-plan and validator coverage for group-style text | Group-style rejection tests |
| Shared lore becomes persona state | Validator rejects factual lore/current facts from `reflection_summary` | Shared-lore and volatile-fact tests |
| Stale mood steers cognition | Preserve affect-settling compare-and-set and add freshness checks | Affect-settling stale tests |
| Repair damages good data | Audit-only default, dry-run report, explicit apply, updated_at compare | Audit/repair script tests |
| Hidden fallback preserves old behavior | Bigbang cutover, static greps for unvalidated writes | Static grep and code review |
| Validator becomes brittle semantic filter | Keep structural and lane-focused; move semantic channel decisions to prompts/schemas | Code review with `no-prepost-user-input` checklist |

## Self Plan Review And Remediation

Review performed during draft refinement on 2026-07-02.

Findings and fixes applied:

- Finding: the previous draft named the issue but did not enumerate
  character-state-specific failure modes deeply enough.
  Fix: added `Failure Modes And RCA Details` covering user-specific
  contamination, commitment contamination, group-style confusion,
  shared-lore confusion, stale freshness, self-cognition/reflection source
  confusion, and downstream cognition impact.
- Finding: the previous draft mentioned validation but did not define a
  concrete validator contract.
  Fix: added `Character-State Validator Contract` with inputs, output shape,
  accepted payload shape, target requirements, freshness rules, and rejection
  behavior.
- Finding: dry-run/apply and data repair safety were too high level.
  Fix: added script commands, default read-only behavior, dry-run repair,
  explicit apply gate, `updated_at` stale-token check, field-only writes, and
  rollback source requirements.
- Finding: the plan risked deterministic post-processing of user commitments.
  Fix: added `no-prepost-user-input` as mandatory, and stated that persistence
  may reject invalid character-state payloads but must not rewrite commitments
  into another lane.
- Finding: daily affect settling was not separated from consolidation writes.
  Fix: added affect-settling call-site strategy that preserves existing
  compare-and-set behavior while adding shared lane validation.
- Finding: downstream impact was not explicit.
  Fix: documented relevance and L1 consumption of `mood/global_vibe`, plus
  broader cognition/RAG/operator impact.
- Finding: requested explicit items could become scattered.
  Fix: kept named subsections for issue description, malformed-data removal,
  RCA, corrupted-source hardening, and robustness proof under
  `Lane Analysis Requirements`.
- Finding: plan-review content was missing.
  Fix: added this `Self Plan Review And Remediation` section and an
  `Independent Plan Review` gate.
- Finding: the refined draft exceeded the `large` plan-class maximum line
  budget after adding required failure modes, migration gates, validator
  contract, and review content.
  Fix: reclassified the plan as `high_risk_migration` because it includes live
  data audit, gated repair apply behavior, stale-token repair safety, and
  production persistence hardening. The plan remains `Status: draft`.
- Finding: placeholder/open-ended wording scan matched a prohibited substring
  inside a normal phrase.
  Fix: replaced that phrase with "repair decision" so the scan no longer
  reports prohibited planning language.

Residual risks:

- `development_plans/README.md` still describes this plan as large. This
  assigned-plan-only refinement cannot edit the registry; the parent should
  reconcile that row before approval if the high-risk plan class is accepted.
- Without running live DB audit during drafting, the current singleton row's
  semantic cleanliness remains assumed from the prior audit summary. Execution
  must record a fresh audit report before any repair decision.
- Validator wording must be implemented carefully to avoid blocking valid
  evolved global character state. Focused valid-state tests and live/debug
  review are mandatory.
- Prompt hardening may still be needed after validator implementation if live
  LLM checks show repeated wrong-lane proposals.

## Execution Evidence

Cleanup-only execution completed on 2026-07-03. New-write hardening remains
superseded by
`C:/workspace/kazusa_ai_chatbot/development_plans/archive/completed/bugfix/consolidator_lane_router_memory_pollution_bigbang_plan.md`.

Commands and artifacts:

- `venv\Scripts\python.exe -m py_compile src\scripts\_lane_cleanup.py src\scripts\audit_character_state_lane.py src\kazusa_ai_chatbot\db\script_operations.py`
- `venv\Scripts\python.exe -m scripts.audit_character_state_lane --dry-run-repair --output test_artifacts\character_state_lane_repair_dry_run.json`
- `venv\Scripts\python.exe -m scripts.audit_character_state_lane --apply-repair --repair-file test_artifacts\character_state_lane_repair_dry_run.json --output test_artifacts\character_state_lane_repair_apply.json`
- `venv\Scripts\python.exe -m scripts.audit_character_state_lane --dry-run-repair --output test_artifacts\character_state_lane_post_repair_dry_run.json`

Results:

- Baseline: 1 total row, 3 valid global character-state field findings, 0
  deterministic planned actions.
- Apply: 0 actions, 0 blocked actions.
- Post-audit: 3 `valid_global_character_state` findings and 0 deterministic
  planned actions remain.
