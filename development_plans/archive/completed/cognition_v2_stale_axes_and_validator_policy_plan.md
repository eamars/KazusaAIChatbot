# Cognition V2 stale relationship axes and validator admission policy

## Summary

- **Status:** completed
- **Independent review status:** the validator/relationship content passed the earlier GPT-5.6 SOL alignment reviewer, Terra returned PASS on the completed implementation, and SOL returned PASS on final work sign-off after parent remediation. The user explicitly approved this amended plan and commanded implementation; that approval was the execution gate, and the pre-execution plan review was skipped as directed.
- **Owner:** Cognition V2 contract owner
- **Goal:** restore reducer-owned relationship familiarity and salience maintenance, remove semantic validation/retry from producer admission, and keep structural and boundary safety explicit.
- **Scope boundary:** Cognition V2 producer parsing/structural admission, the existing boundary validators, relationship state/reducer maintenance, persisted relationship-maintenance metadata, tests, source-to-test ownership, migration tooling, and current Cognition V2 documentation.
- **Change direction:** one big-bang runtime-contract cutover, preceded by an explicit persisted-state backfill. Boundary, provenance, permission, state/FSM, persistence, operation, action, and delivery safety remain deterministic.
- **Execution allocation:** the parent owns production fixes, test changes, all test execution, migration execution, evidence, and integration; GPT-5.6 Terra performs the independent read-only implementation review at the normal speed tier; GPT-5.6 SOL performs the independent read-only final work sign-off at the normal speed tier; parent owns all remaining work.
- **Acceptance state:** the user approved this plan and explicitly commanded implementation. Production implementation, migration, verification, Terra review, and SOL final work sign-off are complete; the plan is closed as `completed`.

The phrase **“let semantic error pass”** means that no semantic validator is used to reject, classify, omit, or retry a structurally usable candidate. Structural parsing/validation and the existing deterministic boundary, state, permission, operation, action, and delivery checks remain active. A boundary or state check may still reject an unsafe result; that is not a semantic retry.

## Scope and change direction

### Current contract problem

Relationship familiarity and salience exist in the native state and are read by downstream projections, goals, and emotion derivation, but they have no complete relationship-axis maintenance owner. The model-owned relationship delta allowlist intentionally excludes those derived axes.

Cognition producer handling currently mixes:

- provider/transport failures;
- JSON and producer-object structure;
- semantic ownership;
- evidence and role provenance;
- target scope and state-transition safety.

The requested change removes the semantic-validation branch from this mixture. Structural handling remains bounded, and deterministic effectful boundaries remain authoritative.

### In scope

1. Add a reducer-owned relationship-maintenance metadata shape and deterministic familiarity/salience updates.
2. Add an explicit persisted-state backfill for the new maintenance metadata before runtime activation.
3. Remove semantic ownership/mismatch validation and semantic-specific producer retries.
4. Validate producer structure; let recoverable structural errors continue through the existing recovery path, and retry only unrecoverable structure.
5. Keep provider/transport retry behavior and prompt-cap dispositions explicit and bounded.
6. Keep existing boundary, provenance, permission, target, state/FSM, operation, action, and delivery validators authoritative.
7. Preserve the existing JSON parsing/repair behavior; JSON repair is structural recovery and is not changed by this plan.
8. Keep the planner ownership contract unchanged: goal_threat_outcome retains the knowledge_answered proposition only; it does not gain knowledge-gap uncertainty delta ownership.
9. Update deterministic tests, the source-to-test impact manifest, migration tests, protected-trace replay coverage, and current Cognition V2 documentation.

### Explicitly out of scope

- Changing sexual-content policy, consent policy, or relational-willingness semantics.
- Treating the semantic relationship axis boundary_safety as authorization.
- Letting invalid evidence, role, target, permission, state, operation, action, or delivery data reach an effectful boundary.
- Adding an LLM call for familiarity or salience.
- Changing existing JSON_REPAIR_LLM behavior, repair eligibility, or repair-call policy.
- Adding a generic compatibility validator, alternate vocabulary, parallel runtime path, or silent legacy fallback.
- Expanding goal_threat_outcome to own knowledge_gaps.<handle>.uncertainty.
- Changing dialog wording, surface generation, adapter delivery, model routing, scheduler behavior, or unrelated persistence.
- Using sexual explicitness, message count, or model-authored intimacy language as a direct familiarity increment.

## Cutover Policy

Overall strategy: bigbang, with a required pre-activation state backfill.

| Area | Policy | Instruction |
|---|---|---|
| Semantic and goal producer admission | bigbang | Remove semantic validation/retry while retaining structural handling and existing deterministic boundary checks. |
| Boundary safety | bigbang | Keep deterministic boundary rejection and fail-closed effectful behavior. |
| Relationship state metadata | migration | Dry-run, backup, backfill, validate, and activate required relationship-maintenance metadata before strict runtime validation. |
| Tests and source ownership | bigbang | Replace stale retry expectations, add exact owner tests, and update the impact manifest in the same change. |
| Current documentation | bigbang | Update the active Cognition V2 README; archived historical contracts remain immutable. |

## Must Do

1. Define and validate RelationshipMaintenanceV1 as non-model-facing persisted metadata.
2. Backfill every existing user cognition relationship with the maintenance defaults before enabling strict new-state admission.
3. Preserve existing familiarity, salience, and all other relationship-axis values during backfill.
4. Apply familiarity reinforcement once per UTC interaction date, with a bounded same-day relationship/fact bonus and source-id idempotency.
5. Apply elapsed relationship-salience decay before downstream derivation, then reinforce from the strongest accepted unique relationship delta.
6. Make source replay protection durable with a bounded processed-source ledger, explicit episode date, monotonic admission, and expected-previous-state compare-and-set commits.
7. Admit relationship maintenance exactly once after the final accepted appraisal prefix, using the explicit accepted-delta carrier and trusted-fact allowlist.
8. Make the canonical transition guard return the exact state-plus-receipt result and update every caller in one contract cutover.
9. Remove semantic ownership validation and semantic-specific retry/omission behavior from producer admission.
10. Let recoverable producer structure continue through the existing structural recovery path without a producer retry; retry unrecoverable structure within the existing owner cap.
11. Keep existing boundary and relational-willingness enforcement; an invalid boundary carrier cannot reach downstream effects.
12. Preserve provider retry caps, prompt-cap dispositions, and existing terminal branch dispositions.
13. Preserve existing JSON repair behavior and measure it as existing structural recovery rather than changing it here.
14. Update the exact source-to-test manifest entries, public DB facade, DB ICD, and script registry before accepting any production source change.
15. Replay the captured QQ cases one at a time and inspect the resulting human-readable evidence, with synthetic dispositions attributed separately.

## Deferred

- Population-wide retry-frequency measurement beyond the single protected QQ trace.
- Adaptive familiarity-rate tuning based on large-scale behavioral data.
- Any new LLM-based evaluator, critic, repair agent, or semantic fallback.
- Changes to action authorization, resolver authorization, selected-operation schemas, or relational-willingness field semantics.
- Automatic migration of unrelated cognition-state fields.
- Changes to archived development plans or historical trace artifacts.

## Confirmed admission policy

The runtime sequence is fixed:

1. Determine whether the provider returned a usable candidate.
2. Parse and clean it through the canonical parser and the existing structural-recovery path.
3. Validate required producer-object structure.
4. Pass the complete structurally usable candidate to the existing deterministic boundary/state checks.
5. Do not produce a semantic verdict, semantic diagnostic, semantic omission, or semantic retry.

| Structural/operational condition | Result | Model retry |
|---|---|---:|
| Provider/transport failure | Preserve the existing bounded provider retry loop and owner terminal disposition. No candidate enters admission. | Existing provider cap |
| Prompt-cap disposition | Preserve the existing prompt-cap disposition and issue no further call. | 0 |
| Recoverable producer structure | Complete the existing structural recovery and continue without a producer retry. | Existing recovery behavior |
| Unrecoverable producer structure | Use the existing owner-specific replacement loop; exhaustion preserves the current branch disposition. | Existing structural cap |

Boundary validation is an independent mandatory gate after structural parsing:

```text
structural parse/recovery
    -> existing boundary/provenance/state validation
    -> reducer, persistence, scheduling, execution, or delivery
```

The boundary gate keeps its existing rejection and fail-closed behavior for
unauthorized evidence, roles, target paths, enums, provenance, state targets,
operations, permission carriers, and relational-willingness carriers. A
structurally usable candidate is passed as a complete candidate; this plan does
not identify or omit semantic components. A semantic relationship axis is never
converted into authorization. Transition guards, state validation, and
effectful boundary validators remain authoritative.

### Deterministic parsing and call budget

- Continue using the canonical parse_llm_json_output(...) entry point.
- Use the existing deterministic transport cleanup and object-shape normalization.
- Preserve the existing JSON_REPAIR_LLM eligibility and behavior. It remains a structural recovery mechanism and is not part of the semantic-validator removal.
- If the existing structural recovery cannot produce the required producer object, use the existing unrecoverable-structure retry path.
- Provider errors remain the existing bounded operational exception because they produce no candidate.
- The existing attempt/trace record remains authoritative. This plan does not add a semantic verdict, semantic diagnostic, or semantic-omission disposition.
- Existing provider, prompt-cap, structural-recovery, structural-retry, boundary, and terminal dispositions remain distinguishable according to their current owner contracts.

### Structural-only producer envelope

The current combined semantic-appraisal validation path is split at the
producer boundary in one big-bang contract change:

- The structural normalizer validates the JSON/object envelope, exact required
  structural fields, types, bounded lengths/counts, and routing identity. It
  preserves semantic prose and structured values opaquely.
- The complete structurally usable candidate is then passed to the existing
  deterministic boundary/state guards. Those guards remain responsible for
  allowed handles, evidence provenance, target paths, proposition/operation
  carriers, duplicate conflicts, permission, and FSM/state safety.
- Unknown or unauthorized structured handles, targets, proposition kinds, or
  boundary carriers fail at that existing boundary before reducer mutation. No
  semantic component detector or per-component omission carrier is introduced.
- If the structural envelope cannot be formed, the recoverable/unrecoverable
  structural policy above applies. Free-text meaning is not independently
  judged by deterministic validation.

### Authorized structural normalization

The complete candidate means the candidate after the single canonical
structural normalization pass. Structural normalization may remove a field
that is outside the exact schema of the selected producer branch when the
branch schema is the deterministic owner of that field. This is an
object-shape operation, not a semantic judgment.

The one authorized case in this plan is `relational_willingness` on a
non-owning goal branch. The goal normalizer records a deterministic
`structural_normalization` entry containing the branch, field name, and reason
`non_owning_branch_field`, removes the field before exact ordinary-goal
construction, and forwards the normalized candidate without a semantic
verdict, semantic diagnostic, or producer retry. Missing ordinary required
fields, invalid required types, and owned relational-willingness carriers do
not use this normalization; they follow their existing structural or boundary
dispositions.

This keeps the producer contract complete after normalization while preserving
the existing relational boundary owner. No component-level meaning detector,
semantic omission carrier, or compatibility vocabulary is introduced.

## Relationship-axis target state

### RelationshipMaintenanceV1

Add a required, non-model-facing relationship field:

- relationship_maintenance.schema_version: relationship_maintenance.v1
- relationship_maintenance.last_interaction_date_utc: canonical UTC date string or null
- relationship_maintenance.last_bonus_date_utc: canonical UTC date string or null
- relationship_maintenance.last_source_id: canonical current-episode source identity or null
- relationship_maintenance.processed_source_ids: bounded canonical source-id list for the current interaction date

The top-level cognition state schema remains cognition_state.v2. The nested maintenance schema versions the new persisted subshape. New states receive the default object. Existing states are backfilled before strict validation is activated. The maintenance object is excluded from state projection, operational relationship context, prompts, and model output.

`processed_source_ids` has a hard maximum of 256 entries. Entries are never
evicted within the active interaction date; an attempted overflow fails
closed. `last_source_id` remains the most recently accepted source identity
for diagnostics and the bounded ledger is the durable replay authority.

The live carrier is `payload["episode"]["episode_id"]`, passed explicitly as
`source_episode_id`. The facade also derives the canonical UTC interaction
date from `payload["episode"]["created_at"]` and passes it as
`interaction_date_utc`. Both carriers are deterministic and model-independent.
An absent or malformed carrier fails the state-update contract when a
committed episode is expected.

Maintenance admission is monotonic: a source older than
`last_interaction_date_utc` is a maintenance no-op; a source already present
in `processed_source_ids` is a maintenance no-op; a new source on the current
date is processed once and appended; a newer date resets the bounded ledger to
that source. A newer source can advance the date but no replay can move the
date backward. The persistence envelope also carries the complete canonical
`expected_previous_state`, so an out-of-order full-state commit fails
compare-and-set
instead of overwriting an intervening episode.

### Familiarity

Use the effective UTC date from the accepted episode/state update:

- If the source is a new source on a newer effective date, add 1, set last_interaction_date_utc, and reset processed_source_ids to that source.
- If the source is a new source on the current effective date, append it without adding the daily interaction signal.
- If the same accepted episode contains at least one accepted relationship-axis delta or trusted user-specific fact, and last_bonus_date_utc differs from the effective date, add 1 and set last_bonus_date_utc.
- The total daily increment is capped at 2.
- A repeated source_id, including a replay after another episode has advanced the state, is idempotently ignored by the durable source ledger.
- A source dated before the current interaction date does not reinforce familiarity or salience; the complete-state compare-and-set gate separately rejects a stale full-state commit, including same-timestamp collisions.
- The axis never decays and remains clamped to 0..100.
- The +1/+1 schedule is a fixed conservative initial policy: one daily interaction signal prevents message-count inflation, and one accepted relationship/fact signal distinguishes substantive interaction. Future tuning is a separate plan.

### Relationship salience

In the user-state update order:

1. Apply `floor(elapsed_seconds * USER_SALIENCE_DECAY_RATE_PER_HOUR / 3600)`
   to relationship salience, where the named authoritative constant is
   `USER_SALIENCE_DECAY_RATE_PER_HOUR = 4` points per hour. This freezes the
   current user-path rate; no configuration lookup or implicit rate is added.
2. Collect unique accepted relationship-axis deltas from the current episode.
3. Add the maximum absolute delta to relationship salience; positive and negative changes both increase attention.
4. Clamp salience to 0..100.
5. Run lifecycle, emotion, goal gating, projection-cache, and other downstream derivation.

No model-owned relationship delta may target familiarity or salience. The existing relationship projection continues to expose semantic bands, not maintenance metadata.

### Single maintenance transaction phase

Relationship maintenance is a final-accepted-prefix operation, not a generic
side effect of `apply_state_update`:

1. The preliminary `apply_state_update` call applies direct facts, elapsed
   entity evolution, and existing deterministic transitions without any
   source-ledger admission, familiarity increment, relationship-salience
   decay, or relationship-delta reinforcement.
2. Each cumulative appraisal trial inside
   `_reduce_appraisals_with_isolation` applies only the candidate semantic
   state and existing trial derivation. Trial reductions pass no maintenance
   carrier and cannot mutate the durable source ledger or relationship
   maintenance axes.
3. The canonical `transition_guards.apply_semantic_deltas` result contract
   returns the updated state plus `accepted_delta_receipts` and
   `rejected_delta_receipts`. `apply_semantic_appraisals` only propagates that
   canonical result; it does not independently decide acceptance. Each
   accepted receipt has exactly these keys:
   `target_path`, `relationship_axis`, `requested_delta`, `applied_delta`,
   `previous_value`, `next_value`, `evidence_refs`, and
   `duplicate_disposition`. `relationship_axis` is one of
   `positive_regard`, `trust`, `attachment`, `desired_closeness`,
   `perceived_closeness`, `care`, `boundary_safety`, `exclusivity`, or
   `unresolved_injury` for a relationship path and is null otherwise;
   `duplicate_disposition` is `unique` for every accepted receipt.
   `rejected_delta_receipts` contains the exact `target_path` and disposition
   `duplicate_target` for duplicate targets. Invalid or unauthorized targets
   raise the existing typed state error and produce no receipt. Duplicate
   targets never enter `accepted_delta_receipts`.
4. The canonical receipt records clamping exactly: `requested_delta` is the
   validated proposal, `previous_value` is the pre-application integer,
   `next_value` is the bounded post-application integer, and
   `applied_delta == next_value - previous_value`. Relationship proposals are
   validated at the existing +/-10 per-event bound and the native axis bounds;
   salience reinforcement uses `abs(applied_delta)`, not the requested value.
   The final `accepted_relationship_deltas` carrier is the filtered subset of
   canonical accepted receipts with a non-null relationship_axis. It is
   internal reduction data and is never model-facing or persisted.

The exact `SemanticDeltaApplicationResultV2` shape is:

```text
{
  "updated_state": dict[str, Any],
  "accepted_delta_receipts": list[SemanticDeltaReceiptV2],
  "rejected_delta_receipts": list[SemanticDeltaRejectionReceiptV2],
}
```

`SemanticDeltaReceiptV2` has exactly:
`target_path: str`, `relationship_axis: RelationshipAxisV2 | None`,
`requested_delta: int`, `applied_delta: int`, `previous_value: int`,
`next_value: int`, `evidence_refs: list[CognitionEvidenceV2]`, and
`duplicate_disposition: Literal["unique"]`. The evidence refs are complete
structured `CognitionEvidenceV2` rows in source order, capped at eight, not
source-ID strings. `SemanticDeltaRejectionReceiptV2` has exactly
`target_path: str` and `disposition: Literal["duplicate_target"]`. There is
one rejected receipt per unique duplicate target path, not one per duplicate
proposal. Both receipt lists are lexicographically sorted by target_path.
Invalid, unauthorized, or malformed proposals raise the existing typed state
error and return no application result. The result's `updated_state` key is
the only state carrier; no alternate `state` key is introduced.
5. After the accepted prefix is finalized, `_run_cognition` calls
   `apply_relationship_maintenance` exactly once with that carrier, the
   explicit source episode ID/date, elapsed seconds, and the trusted fact
   subset. It applies source admission, familiarity, relationship-salience
   decay, and strongest-delta reinforcement in one ordered transaction.
6. The final state then runs `create_deterministic_goals` and final state
   validation once more so goal gating and downstream projections see the
   maintained relationship state. Only this post-maintenance state reaches
   output projection and persistence.

The trusted user-specific fact subset is deterministic and exact: facts whose
producer is one of `action_result`, `resolver_observation`, or `tool_result`,
whose fact kind is one of `goal_progress_observed`, `goal_completed`,
`goal_terminal_failure`, `goal_obstruction_removed`, `threat_resolved`,
`event_repaired`, or `knowledge_answered`, and which have already passed the
existing `apply_direct_fact` producer, evidence, target, and FSM guards. The
scheduler-only `deadline_reached` and `source_occurred` facts, promoted source
metadata, and any rejected fact are excluded.

### Receipt caller disposition

Every current caller is updated in the same big-bang contract change:

- `state_reducers.py::apply_semantic_appraisals` unwraps
  `updated_state`, retains the canonical receipt lists while constructing the
  cumulative result, and returns the exact application-result shape.
- `state_reducers.py::apply_state_update` unwraps `updated_state` for its
  generic preliminary and trial transitions and intentionally discards both
  receipt lists because it is not the final maintenance phase.
- `facade.py::_reduce_appraisals_with_isolation` discards receipts from every
  rejected or intermediate trial, retains only the final accepted-prefix
  receipt list, and passes its filtered relationship subset once to
  `apply_relationship_maintenance`.
- `semantic_appraisal.py` uses the result only as a state-compatibility probe,
  unwraps `updated_state`, and intentionally discards receipts; its trial
  cannot maintain relationship axes.
- `character_carryover.py` unwraps `updated_state` and intentionally discards
  receipts because the character-scope `CharacterCarryoverResultV2` has no
  relationship-maintenance output carrier. Its public state-update contract
  remains unchanged; the caller must not independently classify receipts.

No caller reconstructs accepted/rejected status from proposals. Only
`transition_guards.py` emits the receipt lists.

### End-to-end create/consume/update/persist/re-read loop

The two stale axes are complete only when this loop is connected:

1. `db/users.py::create_user_profile` creates a relationship through
   `build_acquaintance_user_state`, including the default
   `RelationshipMaintenanceV1` object.
2. `db/users.py::get_user_cognition_state` reads and validates that persisted
   state. The facade consumes familiarity and relationship salience through
   the existing state projection, operational context, emotion, goal, and
   output paths; maintenance metadata remains excluded from model context.
3. `cognition_core_v2/facade.py::_run_cognition` passes the canonical
   `payload["episode"]["episode_id"]` as an explicit `source_episode_id` and
   derives the canonical UTC date from `payload["episode"]["created_at"]` as
   `interaction_date_utc`. The reducer stores the canonical
   `episode:<episode_id>` source identity in the bounded replay ledger.
4. `state_reducers.py::apply_relationship_maintenance` applies the familiarity
   and salience maintenance exactly once after final accepted-prefix
   reduction, and `output_projection.py::build_state_update` carries the
   complete replacement state plus `expected_previous_state` in the
   persistence envelope.
5. `nodes/persona_supervisor2_cognition.py::commit_cognition_output` commits
   the user replacement through the DB-owned
   `compare_and_replace_user_cognition_state` boundary using that expected
   state. A compare-and-set conflict fails closed and produces no downstream
   effect.
6. A fresh `get_user_cognition_state` returns the updated axes and maintenance
   metadata, and the next projection consumes the updated semantic bands.

The source identity is an input/episode carrier, never model-authored. A
missing source episode ID is a deterministic state-update contract failure and
cannot apply familiarity reinforcement or claim replay idempotency.

## Producer behavior

### Semantic appraisal

Modify semantic_appraisal.py around _appraise_semantic_item and its current
validation/repair branch:

1. Call the canonical parser and preserve its existing deterministic/repair
   behavior.
2. Apply the existing structural validation and recovery path.
3. Remove the semantic ownership/mismatch validator and its retry trigger.
4. Use the existing appraisal retry loop for provider failure or unrecoverable
   producer structure according to the current owner cap.
5. Keep existing boundary, provenance, target, state, and FSM checks unchanged.

The semantic stage does not infer whether free-text meaning is correct, owned,
or coherent. It also does not create component-level omission records. Existing
boundary and state guards remain responsible for rejecting unauthorized handles,
targets, provenance, or transitions.

### Goal cognition

Modify goal_cognition.py around run_goal_cognition and
validate_goal_bid_draft:

- Normalize model-owned optional fields before exact downstream construction.
- A foreign relational_willingness field on a non-owning branch is recorded as
  the authorized structural normalization and removed without propagation or
  retry; it is not semantically classified.
- A relational_willingness field on an owning branch is validated by its
  existing boundary contract. Invalid enum pairings, evidence, role scope, or
  current-episode binding fail closed and cannot reach downstream action or
  surface stages.
- Missing required ordinary goal fields or unusable required types remain
  unrecoverable producer structure and use the existing three-attempt goal
  producer cap.
- The downstream ordinary goal schema remains unchanged.

### Facade and state boundary

Keep the facade's existing result and failure contracts. Do not add a
per-component semantic carrier or semantic diagnostic propagation. Final state
validation and existing state/FSM failure handling remain at the current
boundary and do not become semantic producer retries.

Keep transition_guards.py as the deterministic owner of bounded deltas, allowed
paths, provenance, duplicate conflict handling, and FSM transitions. Keep
contracts.py as the owner of public input/output, selected operation, relational
willingness, and persistence boundary validation.

### Boundary safety answer

The state axis relationship.boundary_safety is a semantic contextual axis. Its
accepted delta may influence relationship context, but it is not permission,
authorization, or consent. It must not authorize an action, override
relational_willingness, bypass selected-operation validation, or bypass action or
resolver authorization.

Effectful boundary safety is therefore not subject to a semantic trust decision.
It is enforced independently by deterministic validation in contracts.py,
action/resolver authorization, state validation, and delivery checks. Removing
semantic validation does not weaken any of those checks.

## Persisted-state migration

Create a dedicated migration command under the DB ICD-owned script package:

- src/scripts/migrate_cognition_relationship_maintenance.py

The command is an orchestration surface only. Extend the existing named DB
maintenance boundary in `src/kazusa_ai_chatbot/db/script_operations.py` with
`list_user_cognition_states_for_relationship_maintenance_migration` and
`compare_and_replace_user_cognition_state_for_migration`. Those helpers own
the user-profile query, deterministic ordering, state replacement selector,
expected-state digest compare, and Mongo write. The command owns CLI parsing,
dry-run classification, bounded report/backup files, digest verification, and
exit status; it does not access Mongo collections directly.

Because the runtime cognition node imports the user CAS operation through the
public runtime facade, add the runtime helper to
`src/kazusa_ai_chatbot/db/__init__.py` and document the new runtime and
maintenance ownership in `src/kazusa_ai_chatbot/db/README.md`. Register the
new operator entry and its dry-run/apply contract in `src/scripts/README.md`.
The durable `relationship_maintenance.v1` shape, bounded ledger, and
compare-and-set write semantics are part of that ICD update.

The command has two explicit modes:

1. dry-run: read user cognition states, classify missing/invalid maintenance
   metadata, export a bounded report and backup digest, and perform no writes;
2. apply: require a matching dry-run report and backup digest, add the default
   RelationshipMaintenanceV1 object, preserve all existing relationship axes,
   validate each replacement state, and record per-document success or
   fail-closed disposition.

Activation gate:

- dry-run report is complete;
- backup artifact is durable;
- every changed state validates;
- the runtime validator and builders require the new object;
- the migration apply report is accepted;
- only then may the new Cognition V2 runtime contract be enabled.

No legacy relationship state without relationship_maintenance is accepted after
the cutover. Migration drift, unknown state shape, or concurrent document
change fails closed for that document and blocks activation until resolved.
Migration replacement uses the same expected-state compare contract as the
live user commit and records `updated`, `drift`, and `missing` dispositions.

## Evidence and review record

The accepted relationship-maintenance migration and post-apply validation are
summarized in
`test_artifacts/diagnostics/cognition_v2_relationship_maintenance_closure_migration.md`.

### Protected QQ trace

The protected trace evidence remains:

- [JSON trace export](../../../test_artifacts/diagnostics/llmtrace_93482_validator_evidence.json)
- [Human-readable validator review](../../../test_artifacts/diagnostics/llmtrace_93482_validator_evidence_review.md)

It contains one completed trace run, 13 recorded LLM steps, 18 attempt records,
and partial_failure disposition. The self_improvement goal branch spent three
calls on a parseable candidate containing an extra model-owned field. The
semantic appraisal branch rejected a parseable knowledge_answered proposition
because its unauthorized numeric delta was present in the candidate. The
evidence supports investigating semantic-validator retries and recoverable
structural retries. It does not justify changing JSON repair behavior or
adding component-level omission/admission machinery.

### Independent plan review

The requested independent reviewer was:

- executor: GPT-5.6 SOL
- reasoning: high
- service tier: priority
- role: independent read-only plan reviewer
- verdict: REJECT for the initial expanded draft; corrected in this plan.

The initial review surfaced six issues. This revision keeps the durable-state,
planner-ownership, provider-budget, and traceability corrections that remain
within scope, and removes the previously proposed per-component omission and
JSON-repair changes.

| Finding | Remediation in this draft |
|---|---|
| Knowledge-gap planner expansion duplicated ownership | Removed the planner change. Keep knowledge_answered proposition-only ownership. |
| Semantic and boundary classifications overlapped | Removed semantic mismatch classification from the plan; retain existing boundary/state checks as the separate safety owner. |
| A safe component carrier was absent | Removed per-component omission from scope because semantic components are no longer omitted by this change. |
| Provider and prompt-cap behavior was missing | Preserve the existing provider and prompt-cap behavior; only the semantic retry path is removed. |
| Familiarity daily/idempotency state was not durable | Added RelationshipMaintenanceV1, UTC date/source fields, migration command, activation gate, and date/idempotency tests. |
| Test traceability was incomplete | Expanded every row with symbols, semantic owners, exact nodes, test modes, regressions, and the source-test impact manifest update. |

The initial reviewer’s complete finding and recommendation record is preserved
in `test_artifacts/diagnostics/cognition_v2_stale_axes_plan_review_gpt56sol.md`.
The alignment review of this narrowed draft is preserved in
`test_artifacts/diagnostics/cognition_v2_stale_axes_plan_alignment_review_gpt56sol.md`.
The same reviewer subsequently returned PASS after the corrections. A later
independent execution-readiness review returned REJECT because the plan was
still draft, the structural-normalization wording contradicted the non-owning
goal-field requirement, the migration command bypassed the DB maintenance
boundary, `last_source_id` did not provide durable out-of-order replay
protection, and the test matrix lacked exact protected replay, prompt-cap,
fixture, and cross-boundary propagation nodes. That finding is preserved in
`test_artifacts/diagnostics/cognition_v2_stale_axes_plan_execution_readiness_review_gpt56sol.md`.

### Required amendment after execution-readiness review

| Finding | Required plan decision | Acceptance evidence |
|---|---|---|
| Lifecycle and allocation were not executable | Keep this file `draft` through review; after a fresh PASS, set this file and the registry row to `in_progress`; parent owns implementation and Terra/SOL perform normal-speed read-only reviews. | User approval, registry/status diff, and final review artifacts. |
| Admission wording contradicted non-owning goal-field handling | Authorize only the named structural normalization case and record it in the existing attempt record; preserve complete normalized-candidate forwarding and existing owned-boundary validation. | Exact goal normalization tests and protected replay artifact. |
| Migration ownership bypassed the DB ICD | Extend `db/script_operations.py` with named query and expected-digest CAS helpers; the root command orchestrates only. | Exact helper unit tests and dry-run/apply report. |
| `last_source_id` did not survive out-of-order replay safely | Require bounded `processed_source_ids`, explicit episode date, monotonic source admission, and expected-previous-state compare-and-set for every user cognition commit. | Same-date, duplicate-after-intervening-episode, older-source, overflow, and stale-base tests. |
| Source-to-test matrix lacked exact protected/prompt-cap/fixture/cross-boundary nodes | Add the exact nodes listed in the amended matrix, including strict Mongo seed fixture validation and action/resolver/delivery propagation. | Manifest collection output plus one-at-a-time replay review. |

A fresh review of that amendment then returned REJECT with four additional
mandatory findings. Its response is preserved in
`test_artifacts/diagnostics/cognition_v2_stale_axes_plan_execution_readiness_review_gpt56sol_round2.md`.

| Second-review finding | Required decision in this revision | Acceptance evidence |
|---|---|---|
| Preliminary and cumulative trial reductions could multiply maintenance | Make `apply_relationship_maintenance` the sole final accepted-prefix transaction; define the accepted-delta carrier, exact trusted-fact allowlist, and post-maintenance final goal derivation. | Exact reducer/facade cumulative-trial tests and one final maintenance call assertion. |
| Migration command path and public DB ownership were incomplete | Move the command to `src/scripts`, extend `db.script_operations`, re-export the runtime CAS helper from `db/__init__.py`, and update both DB/script ICD documents. | Manifest rows, public-facade test, ICD documentation tests, and named-helper migration test. |
| Captured QQ evidence was over-attributed | Attribute only its semantic-appraisal and non-owning-goal cases to the captured artifact; keep provider, prompt-cap, structural-cap, and ledger cases as separate synthetic/deterministic tests. | Exact captured replay nodes and separate deterministic node collection. |
| Execution roles lacked complete contracts | Add applicable skills, capability floor, independence, entry/exit gates, acceptance output, and fixed-constraint authority for Parent, Terra, and SOL. | Role-contract section and handoff/review records. |

The amendment is contract scope, not implementation. No production source is
changed until the fresh sign-off and lifecycle promotion gates pass.

## Mandatory skills and rules

Execution must use:

- development-plan for lifecycle, contract, migration, traceability, and review gates;
- local-llm-architecture for producer/reducer ownership, latency, and call-budget review;
- debug-llm for live or replayed LLM evaluation and its human-readable artifact;
- llm-trace-debug for protected trace retrieval and correlation;
- py-style before every Python production or test edit;
- test-style-and-execution before adding or running tests;
- database-data-pull for any requested read-only database export or inspection;
- memory-knowledge-maintenance only for a separately authorized curated-memory change, which this plan does not include.

Rules:

- Read development_plans/README.md before plan execution or promotion.
- Implement only after this plan is approved or in_progress and the user explicitly commands implementation.
- Do not read .env.
- Capture the pre-implementation worktree and owned-file baseline.
- Preserve unrelated worktree changes.
- Keep semantic ownership in the producing LLM stage, deterministic safety ownership in reducers/boundaries, and derived-axis ownership in reducers.
- Run live or captured LLM cases one at a time and inspect each artifact.
- Keep raw private or explicit dialog out of source-controlled code and documentation.

## Test impact and traceability

Every changed production source path must have an entry in
tests/ownership/source_test_impact_manifest.json. The manifest entry is updated
in the same change as the source and its required unit nodes are collected and
run before acceptance.

| Path | Changed symbol or contract | Semantic owner | Exact deterministic unit node(s) | Supplemental node(s) | Mode | Regression prevented |
|---|---|---|---|---|---|---|
| src/kazusa_ai_chatbot/cognition_core_v2/state_models.py | RelationshipStateV2, RelationshipMaintenanceV1, build_acquaintance_user_state, validate_relationship_state, _validate_relationship | state-model owner | tests/unit/cognition_core_v2/test_state_models.py::test_state_models_exposes_owned_contract; tests/unit/cognition_core_v2/test_state_models.py::test_relationship_maintenance_metadata_is_required_and_validated; tests/unit/cognition_core_v2/test_state_models.py::test_relationship_maintenance_rejects_unbounded_source_ledger | tests/test_cognition_relationship_maintenance_migration.py::test_dry_run_backfill_preserves_existing_relationship_axes; tests/test_cognition_v2_fixture_contract.py::test_mongo_seed_user_states_include_relationship_maintenance | deterministic unit | Missing, malformed, model-visible, or unbounded maintenance metadata cannot enter native state. |
| src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py | apply_state_update, apply_semantic_appraisals accepted_relationship_deltas carrier, apply_relationship_maintenance | relationship reducer owner | tests/unit/cognition_core_v2/test_state_reducers.py::test_state_reducers_exposes_owned_contract; tests/unit/cognition_core_v2/test_state_reducers.py::test_relationship_familiarity_reinforces_once_per_utc_interaction_date; tests/unit/cognition_core_v2/test_state_reducers.py::test_relationship_familiarity_applies_same_day_bonus_once; tests/unit/cognition_core_v2/test_state_reducers.py::test_relationship_familiarity_crosses_utc_date_boundary; tests/unit/cognition_core_v2/test_state_reducers.py::test_relationship_familiarity_is_idempotent_for_replayed_source; tests/unit/cognition_core_v2/test_state_reducers.py::test_relationship_duplicate_source_after_intervening_episode_is_ignored; tests/unit/cognition_core_v2/test_state_reducers.py::test_relationship_maintenance_ignores_older_source_after_newer_episode; tests/unit/cognition_core_v2/test_state_reducers.py::test_relationship_maintenance_rejects_source_ledger_overflow; tests/unit/cognition_core_v2/test_state_reducers.py::test_relationship_salience_decays_before_downstream_derivation; tests/unit/cognition_core_v2/test_state_reducers.py::test_relationship_salience_uses_four_points_per_hour; tests/unit/cognition_core_v2/test_state_reducers.py::test_relationship_salience_reinforces_from_strongest_unique_delta; tests/unit/cognition_core_v2/test_state_reducers.py::test_cumulative_trial_reductions_do_not_multiply_relationship_maintenance | tests/test_cognition_core_v2_semantic_terminalization.py::test_terminal_postconditions_survive_same_batch_deltas[knowledge_answered-knowledge_gap-knowledge_gaps-axis_deltas3-expected3] | deterministic unit and integration | Stale axes remain frozen, cumulative appraisal trials cannot multiply maintenance, familiarity double-counts, crosses UTC dates, or replays out of order incorrectly, or salience reinforces after downstream derivation. |
| src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py | UNCHANGED GOVERNED CONTRACT — relationship prompt projection excludes maintenance metadata | projection owner | tests/unit/cognition_core_v2/test_state_projection.py::test_state_projection_exposes_owned_contract; tests/unit/cognition_core_v2/test_state_projection.py::test_relationship_maintenance_metadata_is_not_projected | none | deterministic unit | Internal dates and source IDs leak into model context; the projection owner remains unchanged. |
| src/kazusa_ai_chatbot/cognition_core_v2/facade.py | _run_cognition final accepted-prefix maintenance wiring and source/date carriers | cognition orchestration owner | tests/unit/cognition_core_v2/test_facade.py::test_cognition_passes_episode_source_id_and_interaction_date_to_relationship_maintenance; tests/unit/cognition_core_v2/test_facade.py::test_cognition_applies_relationship_maintenance_once_after_cumulative_trials | tests/test_cognition_relationship_maintenance_loop.py::test_relationship_maintenance_round_trips_create_consume_update_persist_reread | deterministic unit and integration | The reducer cannot enforce episode-source idempotency because the live episode identity or effective UTC date is dropped, or trial reductions multiply maintenance before final commit. |
| src/kazusa_ai_chatbot/cognition_core_v2/output_projection.py | build_state_update replacement-state envelope and complete expected previous state | persistence-envelope owner; existing contract explicitly covered | tests/unit/cognition_core_v2/test_output_projection.py::test_state_update_carries_expected_previous_state; tests/test_cognition_relationship_maintenance_loop.py::test_relationship_maintenance_round_trips_create_consume_update_persist_reread | none | deterministic unit and integration | The updated axes, non-model-facing maintenance metadata, or complete expected previous state are lost between reduction and the persisted replacement envelope. |
| src/kazusa_ai_chatbot/db/users.py | create_user_profile, get_user_cognition_state, compare_and_replace_user_cognition_state | user-state persistence owner | tests/unit/db/test_users_cognition_state.py::test_compare_and_replace_user_cognition_state_rejects_stale_base; tests/unit/db/test_users_cognition_state.py::test_compare_and_replace_user_cognition_state_rejects_same_timestamp_stale_state; tests/test_cognition_relationship_maintenance_loop.py::test_relationship_maintenance_round_trips_create_consume_update_persist_reread | tests/test_cognition_relationship_maintenance_migration.py::test_dry_run_backfill_preserves_existing_relationship_axes | deterministic unit and integration and live-DB rehearsal | Defaults, updated axes, maintenance metadata, or complete-state conflicts fail to survive the create/read/commit/read cycle. |
| src/kazusa_ai_chatbot/db/__init__.py | public runtime export of compare_and_replace_user_cognition_state | DB runtime-facade owner | tests/unit/db/test_public_facade.py::test_user_cognition_compare_and_replace_is_public | tests/test_cognition_relationship_maintenance_loop.py::test_relationship_maintenance_round_trips_create_consume_update_persist_reread | deterministic unit and integration | The cognition node reaches a private or different persistence path instead of the named public DB operation. |
| src/kazusa_ai_chatbot/db/script_operations.py | list_user_cognition_states_for_relationship_maintenance_migration, compare_and_replace_user_cognition_state_for_migration | DB maintenance-boundary owner | tests/unit/db/test_script_operations.py::test_relationship_maintenance_migration_helpers_use_expected_digest; tests/unit/db/test_script_operations.py::test_relationship_maintenance_migration_helpers_report_drift | tests/test_cognition_relationship_maintenance_migration.py::test_migration_command_uses_named_db_maintenance_boundary | deterministic unit and migration integration | The migration command bypasses the DB ICD, scans in nondeterministic order, or overwrites concurrent state. |
| src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py | commit_cognition_output, _commit_cognition_state | cognition commit owner | tests/unit/nodes/test_persona_supervisor2_cognition.py::test_user_cognition_commit_uses_compare_and_replace; tests/test_cognition_relationship_maintenance_loop.py::test_relationship_maintenance_round_trips_create_consume_update_persist_reread | tests/test_cognition_v2_boundary_propagation.py::test_boundary_rejection_stops_before_action_resolver_delivery | deterministic unit and integration | The replacement state is computed but does not reach the user cognition document, stale commits overwrite newer state, or unsafe output reaches effectful propagation. |
| src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py | _appraise_semantic_item and existing structural validation/recovery path | semantic producer owner | tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_semantic_appraisal_exposes_owned_contract; tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_structurally_usable_semantic_content_does_not_trigger_semantic_retry; tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_recoverable_structure_does_not_trigger_producer_retry; tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_unrecoverable_structure_retries_within_appraisal_budget; tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_provider_failure_preserves_appraisal_attempt_cap; tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_prompt_cap_preserves_zero_call_disposition; tests/unit/cognition_core_v2/test_semantic_appraisal.py::test_boundary_validation_preserves_existing_rejection_behavior | tests/test_cognition_core_v2_semantic_terminalization.py::test_terminal_postconditions_survive_same_batch_deltas[knowledge_answered-knowledge_gap-knowledge_gaps-axis_deltas3-expected3]; tests/test_cognition_v2_protected_qq_replay.py::test_protected_qq_replay_semantic_appraisal_avoids_semantic_retry | deterministic unit, integration, and captured replay | The captured QQ semantic-appraisal rejection is not conflated with synthetic provider, prompt-cap, or structural-cap tests; semantic validation remains out of the retry trigger, recoverable structure spends no producer retry, prompt caps issue no call, and boundary behavior remains unchanged. |
| src/kazusa_ai_chatbot/cognition_core_v2/semantic_source_planner.py | UNCHANGED GOVERNED CONTRACT — _permitted_delta_paths and knowledge_answered proposition-only ownership | semantic source planner owner | tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_semantic_source_planner_exposes_owned_contract; tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_goal_outcome_filters_terminal_handles_and_delta_paths; tests/unit/cognition_core_v2/test_semantic_source_planner.py::test_goal_outcome_keeps_eligible_handles_and_candidates | tests/test_cognition_core_v2_alignment_gates.py::test_each_question_receives_only_family_local_handles_and_state | deterministic unit and integration | Knowledge-gap numeric uncertainty ownership is not accidentally expanded into goal_threat_outcome; the planner owner remains unchanged. |
| src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py | run_goal_cognition, validate_goal_bid_draft, goal-bid structural normalization | goal producer owner and relational boundary owner | tests/unit/cognition_core_v2/test_goal_cognition.py::test_goal_cognition_exposes_owned_contract; tests/unit/cognition_core_v2/test_goal_cognition.py::test_nonowning_relational_willingness_is_stripped_without_propagation; tests/unit/cognition_core_v2/test_goal_cognition.py::test_owned_relational_willingness_boundary_failure_fails_closed; tests/unit/cognition_core_v2/test_goal_cognition.py::test_unrecoverable_goal_structure_retries_within_budget; tests/unit/cognition_core_v2/test_goal_cognition.py::test_provider_failure_preserves_goal_attempt_cap; tests/unit/cognition_core_v2/test_goal_cognition.py::test_prompt_cap_preserves_zero_call_disposition; tests/unit/cognition_core_v2/test_goal_cognition.py::test_goal_output_contract_keeps_existing_schema | tests/test_cognition_core_v2_attempt_ledger.py::test_goal_attempt_budget_is_monotonic_across_graph_attempts; tests/test_cognition_core_v2_relational_carrier_failure_live_llm.py::test_live_f408_carrier_failure_without_episode_binding; tests/test_cognition_v2_protected_qq_replay.py::test_protected_qq_replay_nonowning_goal_field_uses_structural_normalization | deterministic unit, integration, and captured replay | The captured QQ goal retry is separated from synthetic provider/prompt/structure tests; foreign fields are normalized once without semantic retry, ordinary structure retains its cap, provider/prompt dispositions remain bounded, and an invalid owned relational carrier cannot reach downstream behavior. |
| src/kazusa_ai_chatbot/cognition_core_v2/transition_guards.py | apply_semantic_deltas authoritative state-plus-receipt result, _apply_delta_path, transition guards | deterministic state-boundary owner | tests/unit/cognition_core_v2/test_transition_guards.py::test_transition_guards_exposes_owned_contract; tests/unit/cognition_core_v2/test_transition_guards.py::test_apply_semantic_deltas_returns_authoritative_receipts; tests/unit/cognition_core_v2/test_transition_guards.py::test_duplicate_delta_is_rejected_from_receipts; tests/unit/cognition_core_v2/test_transition_guards.py::test_relationship_receipt_records_clamped_applied_delta; tests/test_cognition_core_v2_failures.py::test_candidate_delta_rejects_mismatched_evidence; tests/test_cognition_core_v2_failures.py::test_semantic_deltas_reject_reducer_owned_goal_axes; tests/test_cognition_core_v2_failures.py::test_conflicting_duplicate_semantic_targets_are_rejected | tests/test_cognition_core_v2_failures.py::test_semantic_uncertainty_decrease_drives_the_frozen_gap_fsm | deterministic unit and integration | Semantic-validator removal is mistaken for permission to weaken state/provenance/FSM guards, or accepted maintenance input is derived outside the canonical transition owner. |
| src/kazusa_ai_chatbot/cognition_core_v2/character_carryover.py | source-free carryover caller unwraps and intentionally discards the canonical delta-receipt result | character carryover owner | tests/unit/cognition_core_v2/test_character_carryover.py::test_character_carryover_consumes_canonical_delta_receipts | tests/test_cognition_core_v2_character_carryover.py::test_character_carryover_discards_internal_delta_receipts_after_state_application | deterministic unit and integration | A secondary reducer caller ignores the canonical result contract or independently reclassifies accepted/rejected deltas. |
| src/kazusa_ai_chatbot/cognition_core_v2/contracts.py | StateUpdateV2 expected_previous_state, RelationshipAxisV2, SemanticDeltaReceiptV2, SemanticDeltaRejectionReceiptV2, SemanticDeltaApplicationResultV2, public input/output, selected operation, relational willingness, effectful boundary | effectful boundary, receipt-contract, and persistence-envelope owner | tests/unit/cognition_core_v2/test_contracts.py::test_state_update_requires_expected_previous_state; tests/unit/cognition_core_v2/test_contracts.py::test_state_update_rejects_missing_expected_previous_state; tests/unit/cognition_core_v2/test_contracts.py::test_semantic_delta_receipt_has_exact_contract; tests/unit/cognition_core_v2/test_contracts.py::test_semantic_delta_rejection_receipt_has_exact_contract; tests/unit/cognition_core_v2/test_contracts.py::test_semantic_delta_application_result_has_exact_contract; tests/unit/cognition_core_v2/test_contracts.py::test_selected_response_operation_has_exact_contract; tests/unit/cognition_core_v2/test_contracts.py::test_selected_response_operation_rejects_missing_required_fields; tests/unit/cognition_core_v2/test_contracts.py::test_sensitive_relational_willingness_accepts_all_real_states_and_stances; tests/unit/cognition_core_v2/test_contracts.py::test_semantic_boundary_safety_cannot_override_relational_willingness | tests/test_cognition_v2_boundary_propagation.py::test_boundary_rejection_stops_before_action_resolver_delivery | deterministic unit and integration | relationship.boundary_safety is incorrectly used as authorization, semantic state bypass, or a way around the complete-state commit and canonical receipt boundaries. |
| tests/ownership/source_test_impact_manifest.json | exact source-to-test ownership entries for every changed src path | verification contract owner | tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary; tests/test_test_impact_manifest.py::test_manifest_rejects_empty_unit_mapping | scripts.validate_test_impact --base-ref HEAD --run | static manifest validation | A changed source path lacks an exact collected test owner. |
| src/scripts/migrate_cognition_relationship_maintenance.py | dry-run/apply migration and backup-digest gate | persisted-state migration owner | tests/test_cognition_relationship_maintenance_migration.py::test_dry_run_backfill_preserves_existing_relationship_axes; tests/test_cognition_relationship_maintenance_migration.py::test_apply_requires_matching_dry_run_digest; tests/test_cognition_relationship_maintenance_migration.py::test_migration_fails_closed_on_concurrent_drift; tests/test_cognition_relationship_maintenance_migration.py::test_migration_uses_named_db_maintenance_boundary | explicit live-DB migration rehearsal after approval | deterministic unit and live-DB rehearsal | Existing persisted states lose axes, bypass the DB ICD, overwrite concurrent changes, or activate runtime before backfill. |
| tests/fixtures/cognition_core_v2_mongo_seed.json | strict user cognition-state seed shape | fixture contract owner | tests/test_cognition_v2_fixture_contract.py::test_mongo_seed_user_states_include_relationship_maintenance | tests/test_cognition_relationship_maintenance_migration.py::test_dry_run_backfill_preserves_existing_relationship_axes | deterministic fixture validation | The strict seed path silently omits the required maintenance object or changes existing relationship axes during the cutover. |
| src/kazusa_ai_chatbot/cognition_core_v2/README.md | active admission, boundary, axis, migration, and retry contract | documentation owner | tests/test_cognition_v2_parent_guardrail_docs.py::test_core_readme_documents_two_epoch_owner_budget; tests/test_cognition_v2_parent_guardrail_docs.py::test_core_readme_documents_semantic_admission_and_boundary_contract | none | static documentation test | Runtime behavior and the active documented contract diverge. |
| src/kazusa_ai_chatbot/db/README.md | DB ICD runtime-CAS, maintenance-helper, durable-shape, and migration ownership contract | documentation owner | tests/test_cognition_v2_parent_guardrail_docs.py::test_db_icd_documents_relationship_maintenance_ownership | none | static documentation test | Runtime code or scripts bypass the public DB facade, named maintenance helpers, or durable migration boundary. |
| src/scripts/README.md | migration command registry and dry-run/apply contract | script registry owner | tests/test_cognition_v2_parent_guardrail_docs.py::test_script_registry_documents_relationship_maintenance_migration | none | static documentation test | The operator entrypoint is absent, mislocated, or undocumented. |

### Affected exact test carriers

The complete `expected_previous_state` field is a big-bang contract update for
these existing parent-owned carriers and their exact consumers:

- `tests/cognition_core_v2_test_helpers.py::canonical_cognition_output`,
  collected through `tests/test_cognition_chain_connector_mapping.py::test_persona_connector_maps_one_native_user_scope` and
  `tests/test_persona_supervisor2.py::test_route_after_cognition_uses_validated_v2_speech_intention`.
- `tests/test_persona_supervisor2.py::test_route_after_cognition_uses_validated_v2_silence_intention`.
- `tests/test_cognition_chain_connector_mapping.py::test_final_commit_emits_bounded_success_event` and
  `tests/test_cognition_chain_connector_mapping.py::test_cognition_entry_replaces_stale_profile_with_episode_snapshot`.

The parent updates every exact fixture occurrence in those files and runs the
listed consumers before production handoff; no compatibility default is added
to the state-update contract.

## Change surface

### Delete

- Delete semantic ownership mismatch as a producer repair trigger.
- Delete exact-key retry behavior for recoverable model-owned goal fields.
- Delete the proposed goal_threat_outcome ownership of knowledge_gaps.<handle>.uncertainty.
- Keep the existing structural, transition, and boundary validator modules; remove only the semantic ownership/mismatch retry path.

### Modify

- src/kazusa_ai_chatbot/cognition_core_v2/state_models.py: add and validate RelationshipMaintenanceV1 and require it after migration.
- src/kazusa_ai_chatbot/cognition_core_v2/state_reducers.py: add the final-only ordered relationship-maintenance transaction and consume the canonical accepted-delta receipts without changing the reducer's existing validated-result boundary.
- src/kazusa_ai_chatbot/cognition_core_v2/transition_guards.py: return the canonical state-plus-delta-receipt result, including duplicate and clamping dispositions.
- src/kazusa_ai_chatbot/cognition_core_v2/character_carryover.py: update the source-free caller to consume the canonical delta-receipt result.
- src/kazusa_ai_chatbot/cognition_core_v2/semantic_appraisal.py: remove semantic ownership/mismatch validation and its retry trigger; preserve existing structural recovery and retry behavior.
- src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py: normalize foreign model-owned fields and preserve relational boundary rejection.
- src/kazusa_ai_chatbot/cognition_core_v2/facade.py: pass the canonical episode identity/date into the final-only relationship-maintenance owner after cumulative appraisal reduction.
- src/kazusa_ai_chatbot/cognition_core_v2/contracts.py: carry and validate the complete expected previous cognition state in StateUpdateV2 and the canonical delta-receipt result types.
- src/kazusa_ai_chatbot/cognition_core_v2/output_projection.py: include the complete expected previous cognition state in the persistence envelope.
- src/kazusa_ai_chatbot/db/users.py: preserve the maintenance object and updated relationship axes through user-state create/read/compare-and-replace paths.
- src/kazusa_ai_chatbot/db/script_operations.py: add named migration query and expected-digest compare-and-set helpers inside the DB maintenance boundary.
- src/kazusa_ai_chatbot/db/__init__.py: re-export the runtime user-state compare-and-replace helper used by the cognition node.
- src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py: preserve the final replacement-state commit path for the end-to-end loop.
- tests/ownership/source_test_impact_manifest.json: add exact required nodes for every changed strict source path.
- tests/cognition_core_v2_test_helpers.py: add `expected_previous_state` to the canonical cognition-output fixture.
- tests/test_persona_supervisor2.py: update the exact V2 output fixture to carry the complete expected previous state.
- tests/test_cognition_chain_connector_mapping.py: update user- and character-scope exact output fixtures to carry the complete expected previous state.
- src/kazusa_ai_chatbot/cognition_core_v2/README.md: document the revised admission, retry, boundary, axis, and migration contracts.
- src/kazusa_ai_chatbot/db/README.md: document the runtime CAS helper, DB maintenance ownership, and durable relationship-maintenance shape.
- src/scripts/README.md: register the DB ICD-owned migration command and dry-run/apply contract.
- tests/fixtures/cognition_core_v2_mongo_seed.json: add the required relationship-maintenance object to strict user-state seeds while preserving existing axes.
- The exact test files and nodes listed in the traceability table.

### Create

- src/scripts/migrate_cognition_relationship_maintenance.py: dry-run/apply backfill with backup and drift gates.
- tests/test_cognition_relationship_maintenance_migration.py: deterministic migration contract tests.
- tests/test_cognition_relationship_maintenance_loop.py: create/consume/update/persist/re-read integration contract tests.
- tests/test_cognition_v2_protected_qq_replay.py: one-at-a-time protected replay dispositions and human-readable evidence.
- tests/test_cognition_v2_boundary_propagation.py: action/resolver/delivery fail-closed propagation contract.
- tests/test_cognition_v2_fixture_contract.py: strict Mongo seed fixture contract tests.
- The new unit nodes listed in the traceability table.
- test_artifacts/diagnostics/cognition_v2_stale_axes_plan_execution_readiness_review_gpt56sol.md: preserved independent review and remediation record.
- test_artifacts/diagnostics/cognition_v2_stale_axes_plan_execution_readiness_review_gpt56sol_round2.md: preserved second independent review and remediation record.
- test_artifacts/diagnostics/cognition_v2_stale_axes_plan_execution_readiness_review_gpt56sol_round3.md: preserved third independent review and remediation record.
- test_artifacts/diagnostics/cognition_v2_stale_axes_plan_execution_readiness_review_gpt56sol_round4.md: preserved fourth independent review and remediation record.

### Keep

- contracts.py effectful boundary validators and selected-operation/relational-willingness ownership.
- transition_guards.py provenance, bounded-delta, target-scope, duplicate-conflict, and FSM ownership.
- semantic_source_planner.py proposition-only knowledge_answered ownership and family-local numeric paths.
- state_projection.py's public semantic relationship projection shape.
- output_projection.py's complete replacement-state envelope; explicitly covered by the round-trip test.
- existing provider attempt caps and terminal branch dispositions.
- existing JSON parsing and JSON_REPAIR_LLM structural-recovery behavior.
- existing cognition_state.v2 top-level schema; the nested RelationshipMaintenanceV1 shape is activated through the explicit migration.
- archived development plans and historical evidence artifacts.

## Execution roles and autonomy boundaries

### Parent: implementation, coordinator, test, and execution owner

- **Responsibility:** own the plan, source/test baseline, all production and test implementation, all test execution, integration, manifest, documentation, diagnostic artifacts, protected replay, migration dry-run/apply gates, and final sign-off.
- **Owned surface:** every approved production and `tests/**` path, `tests/ownership/source_test_impact_manifest.json`, `development_plans/**`, `test_artifacts/**`, current README documentation, and all coordination/integration work.
- **Test-first gate:** implement and collect the required tests before production fixes, record the expected-failure baseline, and preserve the test-owned evidence while production code is changed.
- **Isolation rule:** the parent keeps production, test, plan, documentation, and evidence changes integrated in the approved worktree and reviews every attributable diff.
- **Authority:** owns all decisions not explicitly assigned below, including migration operation. Database apply still requires explicit user authorization and the plan's backup/digest/drift gates.
- **Execution:** the parent runs all deterministic, live-LLM, live-DB, migration, and replay tests and inspects every result.

Delegation is limited to independent review: Terra reviews the completed
implementation and SOL performs final work sign-off. Both reviewers use the
normal speed tier and read-only access.

### GPT-5.6 Terra: independent implementation code reviewer

- **Configuration:** GPT-5.6 Terra at the normal speed tier with high reasoning.
- **Responsibility:** read-only review of the completed production diff against the approved plan and the parent-owned test results.
- **Owned surface:** read-only production diff, test artifacts, migration reports, trace evidence, and plan.
- **Authority:** issue a pass/reject verdict and findings; it cannot edit files, run tests, or approve its own work.
- **Gate:** review starts only after the parent has completed the test-first implementation cycle and supplied reproducible test evidence.

### GPT-5.6 SOL: final work sign-off reviewer

- **Configuration:** GPT-5.6 SOL at the normal speed tier with high reasoning.
- **Responsibility:** perform the final independent review of the completed work, closure evidence, lifecycle status, and registry state against the user's requirements.
- **Owned surface:** read-only implementation diff, plan, source/test ownership map, migration and replay evidence, and closure record.
- **Authority:** pass/reject with exact findings; no edits, implementation, test execution, or migration execution.
- **Current status:** final work sign-off follows Terra's implementation review; the user's instruction skips pre-execution plan review and does not waive final work sign-off.

### Role contract fields

| Role | Applicable skills | Capability floor | Independence requirement | Entry gate | Exit and acceptance output | Authority over fixed constraints |
|---|---|---|---|---|---|---|
| Parent | development-plan; local-llm-architecture; debug-llm; llm-trace-debug; py-style; test-style-and-execution; database-data-pull for any authorized read-only DB inspection | Read/write repository access, `venv\\Scripts\\python`, deterministic and live-test execution, artifact inspection, manifest validation, and migration report review | Coordinator and integrator; does not serve as the independent Terra/SOL verdict | Fresh amended-plan SOL PASS, this file and registry `in_progress`, clean owned-file baseline, exact source/test matrix, and tests-first scope frozen | Produces test patches, expected-failure and passing reports, manifest output, replay review, migration dry-run/apply evidence, diff review, and final acceptance record | May coordinate and implement parent-owned tests/docs/artifacts and stop the rollout; may not alter fixed contracts or execution limits without a plan amendment and fresh SOL review. |
| GPT-5.6 Terra | development-plan; local-llm-architecture; py-style; test-style-and-execution for evidence interpretation | Normal-speed, read-only diff and artifact review against the plan, source contracts, and parent evidence | Independent of the parent implementation; no edits, tests, or migration commands | Parent has completed implementation, mapped tests, deterministic verification, and supplied reproducible artifacts | Returns a pass/reject verdict with exact file/symbol findings and acceptance blockers; parent closes every reject before sign-off | May reject the implementation and require parent remediation; cannot edit code, waive a fixed constraint, or approve its own work. |
| GPT-5.6 SOL | development-plan; local-llm-architecture; debug-llm for protected-evidence scope | Normal-speed, read-only lifecycle, contract, ownership, and final work sign-off review of the plan and registry | Independent of the parent and Terra; no edits, tests, or migration commands | Amended plan, registry, current worktree status, ownership map, prior findings, and final evidence are available; user approval is recorded separately | Returns a fresh PASS/REJECT with exact closure blockers; PASS is the gate for lifecycle promotion and plan closure | May approve or reject final work sign-off and require parent remediation; cannot modify the plan, change implementation scope, or waive user/database authorization gates. |

Delegation consists of Terra’s independent implementation review and SOL’s
independent final work sign-off. The parent owns migration execution and every
remaining activity.

## Agent autonomy boundaries

The parent may choose function decomposition and local helper placement that
preserve the fixed contracts. The parent owns production fixes, test fixture
mechanics, test execution, migration execution, and integration. Parent
implementation must:

- reintroduce planner ownership for knowledge-gap uncertainty;
- treat a boundary rejection as a semantic pass;
- pass structurally unusable or boundary-unsafe output to an effectful boundary;
- add a new LLM repair or semantic-evaluator call;
- increase retry caps;
- classify the authorized non-owning relational field normalization as a semantic verdict;
- use relationship.boundary_safety as authorization;
- change the +1/+1 familiarity schedule, salience order, bounded source ledger, compare-and-set commit, or migration shape;
- bypass the named DB maintenance helpers from the migration command;
- add compatibility/fallback behavior;
- record migration apply evidence before activation;
- modify unrelated files.

If code and plan disagree, record the conflict and request a plan amendment.
Do not silently reinterpret the plan.

## Verification and acceptance criteria

### Test-first and isolation gate

1. The parent implements or updates every required test and exact manifest
   mapping before production-code changes.
2. The parent collects the test nodes and records the expected-failure or
   baseline result without changing production code.
3. The parent records the test-owned file hash/status snapshot and applies the
   approved production fixes in the same owned worktree.
4. The parent reviews the integrated diff for scope and ownership, then runs
   the complete mapped test set and all authorized replay/migration checks.
6. GPT-5.6 Terra receives the production diff and parent-owned evidence only
   after the test-first cycle completes.

### Pre-implementation verification

- Confirm this plan is approved or in_progress and the user has explicitly commanded implementation.
- Capture git status --short and the explicit owned-file set.
- Run manifest validation and exact-node collection before source edits.
- Complete migration dry-run and backup/digest review before runtime cutover.

### Deterministic verification

- Run every exact required unit node in the source-to-test matrix.
- Run scripts.validate_test_impact --base-ref HEAD --run after source edits.
- Run git diff --check.
- Confirm existing JSON_REPAIR_LLM structural-recovery behavior is unchanged.
- Confirm all new test nodes are collected; a broader passing suite cannot replace a missing mapped node.
- Confirm the user commit path uses the expected-previous-state compare-and-set and that migration helpers, rather than the root command, own Mongo writes.

### Migration verification

- Run dry-run with no writes.
- Confirm existing relationship axes are byte/value preserved except for the new maintenance object.
- Confirm backup digest matches apply input.
- Confirm concurrent drift fails closed.
- Validate every migrated state with validate_cognition_state.
- Record the apply report before activation.

### Protected replay verification

Replay the captured QQ trace cases one at a time and attribute only the
dispositions present in that artifact:

- the extra foreign relational_willingness case records the authorized
  structural normalization without semantic validation/retry;
- the parseable knowledge_answered case does not spend a
  semantic-validation retry;
- unauthorized knowledge-gap numeric data remains subject to the existing
  boundary/state guards.

Run provider failure, prompt-cap, unrecoverable-structure, and durable
duplicate-after-intervening-episode scenarios as separate deterministic or
synthetic replay tests. Their results must not be reported as captured QQ
evidence. The final propagation test verifies that dialog/surface boundaries
receive only safe state and accepted operations.

### Acceptance criteria

1. Familiarity increases once per UTC interaction date, applies at most one same-day accepted relationship/fact bonus, is idempotent for the same source, never decays, and remains bounded.
2. Relationship salience decays before downstream derivation and increases from the strongest accepted unique relationship delta regardless of sign.
3. Existing persisted states are backfilled and validated before new runtime admission activates.
4. No LLM call is added for either stale axis, semantic validation, or JSON repair; existing JSON repair behavior remains unchanged.
5. Structurally usable semantic content consumes zero semantic-validation retries because that validator is removed; structural and boundary checks remain active.
6. Recoverable structure consumes zero producer retries beyond the existing structural recovery behavior; unrecoverable structure retains the existing retry cap.
7. Boundary rejection never reaches the reducer's effectful boundary, persistence, scheduling, execution, or delivery.
8. Invalid owned relational_willingness fails closed; foreign non-owning relational_willingness cannot propagate.
9. Provider failures, prompt-cap failures, and unrecoverable structure retain explicit bounded dispositions and existing caps.
10. The QQ trace no longer spends semantic-validation retries on structurally usable semantic content, while the existing boundary/state handling remains intact.
11. The source-to-test manifest covers every changed strict source path, and every exact node is collected and run.
12. No unrelated production code, external data, or archived record changes.
13. The end-to-end relationship loop proves create -> consume -> update -> persist -> re-read: a new state receives maintenance metadata, the final accepted-prefix reducer transaction updates familiarity/salience using the explicit episode source ID exactly once, the replacement is committed, a fresh read returns the updated axes/metadata, and projection excludes only the maintenance metadata.
14. A newer episode cannot be overwritten by an older source replay: the reducer's bounded ledger suppresses stale maintenance and the persistence envelope's compare-and-set rejects a stale base version.
15. Migration dry-run/apply performs all Mongo access through named `db.script_operations` helpers and records drift without overwriting concurrent state.
16. Two user commits with the same canonical `updated_at` still have collision-safe compare-and-set behavior because the DB selector compares the complete canonical `expected_previous_state`, not a timestamp alone.

## Independent review and progress evidence

- [x] Repository lifecycle, architecture, source, tests, and protected trace inspected.
- [x] Initial draft plan created.
- [x] GPT-5.6 SOL high-reasoning normal-speed independent review completed with REJECT verdict.
- [x] Planner ownership expansion removed.
- [x] Semantic-validator removal and structural/boundary separation narrowed to user scope.
- [x] Existing provider/prompt-cap behavior preserved.
- [x] JSON repair behavior explicitly preserved and excluded from this change.
- [x] Durable familiarity metadata and migration gate added.
- [x] Exact traceability and manifest update requirements added.
- [x] Alignment review completed; its surfaced issues were corrected in this draft.
- [x] Post-remediation review of the validator/relationship content returned PASS.
- [x] Execution allocation fixed with parent-owned implementation and normal-speed Terra/SOL review roles.
- [x] Independent execution-readiness review returned REJECT; its findings are preserved and amended above.
- [x] Structural normalization, durable source ledger, compare-and-set persistence, DB migration ownership, and exact matrix nodes amended.
- [x] Second independent execution-readiness review returned REJECT; reducer-phase, ICD-path, replay-attribution, and role-contract findings are preserved and amended above.
- [x] Final-only maintenance transaction, exact accepted-delta/fact boundaries, `src/scripts` ownership, public DB export, corrected replay attribution, and complete role contracts amended.
- [x] Third independent execution-readiness review returned REJECT; canonical receipt ownership, collision-safe CAS, fixed decay rate, and explicit Luna ownership findings are preserved and amended above.
- [x] Canonical state-plus-receipt result, complete-state CAS, 4-point/hour salience constant, same-timestamp coverage, and plan-scoped executor constraints amended.
- [x] Fourth independent execution-readiness review returned REJECT; exact receipt shape, caller disposition, unchanged governed rows, and fixture-carrier findings are preserved and amended above.
- [x] Exact application-result/rejection types, caller unwrap/discard rules, unchanged row labels, and all affected expected-state fixture carriers amended.
- [x] User approval and explicit implementation command received.
- [x] User approval accepted as the execution gate; no additional plan review invoked.
- [x] Promote this plan and its registry row to `in_progress` after the explicit user approval.
- [x] Production implementation completed across the approved cognition, persistence, migration, documentation, fixture, and test ownership surfaces.
- [x] Deterministic verification, migration contract evidence, and protected replay execution evidence completed; exact impact validation covered 113 nodes, the focused remediation suite covered 54 tests, and the scoped carrier regression covered 267 passing tests with two unrelated baseline failures recorded.
- [x] Terra post-remediation implementation review returned PASS after accepting the migration activation, stale-goal reconciliation, and propagation-path evidence.
- [x] SOL final work sign-off initially returned REJECT; the parent remediated the stale salience-gated goal, vacuous propagation test, and stale closure-record findings.
- [x] SOL final work sign-off returned PASS after the parent remediation; the plan is archived and closed.
