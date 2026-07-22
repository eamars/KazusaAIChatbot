# Cognition Core V2 Stage 3 System Adoption Plan

## Summary

- Goal: make every Kazusa runtime reasoning source use the native Cognition
  Core V2 episode, action, surface, trace, and persistence contracts when the
  service starts against an empty database.
- Plan class: high_risk_migration.
- Status: in_progress.
- Mandatory skills: `development-plan`, `py-style`,
  `test-style-and-execution`, `local-llm-architecture`,
  `no-prepost-user-input`, `cjk-safety`, `debug-llm`, `character-test`, and
  `control-console-web-development`.
- Overall cutover strategy: contract-first big-bang adoption on an isolated
  fresh database; all normal runtime callers move together and obsolete
  scaffolds are deleted. Production data remains untouched.
- Highest-risk areas: fresh-profile bootstrap, grounded trigger ownership,
  capability availability, single trace settlement, local-LLM call/latency
  bounds, background continuation, and proving database isolation.
- Acceptance criteria: a clean service can bootstrap a configured character,
  reason from every canonical source, select and execute currently supported
  actions, render or stay silent by character judgment, persist continuity,
  expose operable traces, and restart without manual seed or dry-run helpers.
- Mandatory companions:
  [cognition_core_v2_stage_3_execution_manifest.md](cognition_core_v2_stage_3_execution_manifest.md)
  and
  [cognition_core_v2_stage_3_change_radius.md](cognition_core_v2_stage_3_change_radius.md).
  Their contracts, exact inventory, budgets, and commands are part of this
  plan and carry the same approval boundary.
- Execution authority: the user explicitly authorized Stage 3 implementation
  on 2026-07-19.
- Database-isolation amendment: the user explicitly authorized database-level
  isolation on the configured MongoDB URI, using the exact reserved database
  `_test_kazusa_core_v2`, because a separate endpoint is unavailable.

## User Quality Sign-off Record — 2026-07-22

The user approved the Phase 3 artifact set, including the consolidated raw
Chinese real-LLM dialog/monologue report and the emotion, abuse-boundary, role,
mechanical-path, and bounded-error evidence overlay. This records approval of
the artifact evidence and its retained test cases. The plan remains
`in_progress` until the separately tracked external Browser acceptance and
remaining lifecycle gates are closed; this record does not infer those gates.

## Context

Stage 2 established the live V2 cognition chain and closed its integration and
bug-fix plans. Its final evidence is:

- [Stage 2 integration plan](../../archive/completed/short_term/cognition_core_v2_stage_2_integration_plan.md)
- [Stage 2 contract](../../archive/completed/short_term/cognition_core_v2_stage_2_contract_spec.md)
- [Stage 2 execution manifest](../../archive/completed/short_term/cognition_core_v2_stage_2_execution_manifest.md)
- [40-turn monologue/dialog review](../../../test_artifacts/cognition_core_v2/fresh_40_turn_signoff/cognition_v2_fresh_40_turn_monologue_dialog_review.md)
- [Private 05 / Group 15 RCA](../../../test_artifacts/cognition_core_v2/fresh_40_turn_signoff/private_turn_05_group_turn_15_system_rca.md)
- [RCA-fix review](../../../test_artifacts/cognition_core_v2/fresh_40_turn_signoff/private_turn_05_group_turn_15_fix_review.md)

Stage 2 still leaves system-wide adoption work. Chat, idle cognition,
scheduled commitments, accepted/background results, reflection context,
action availability, trace finalization, startup profile loading, operations,
and fresh-database behavior do not yet share one complete production contract.
Several earlier multi-source modules are dry-run or scaffold surfaces rather
than runtime owners. A repository checkout also depends on a manual character
profile load before a fresh database can reason as Kazusa.

Stage 3 closes that radius without migrating production data. It establishes
the native target that Stage 4 will populate. The user will run progressively
more production-like quality and performance tests during Stage 3; those tests
may refine implementation inside the contracts here, while any public-contract
change requires a plan amendment and approval.

## Mandatory Skills

- `development-plan`: govern lifecycle, checkpoints, evidence, independent
  review, and the Stage 4 handoff.
- `py-style`: load before reviewing or editing any Python file.
- `test-style-and-execution`: load before creating, changing, or running any
  test; run live LLM cases one at a time and inspect each result.
- `local-llm-architecture`: govern LLM-stage ownership, finite context,
  bounded loops, routing, prompts, and latency.
- `no-prepost-user-input`: govern user-instruction interpretation and prevent
  keyword or deterministic semantic gates around user input.
- `cjk-safety`: load before writing Python source containing Chinese text.
- `debug-llm`: govern real-LLM artifacts, prompt comparison, anti-cheat
  evidence, and quality review.
- `character-test`: govern simulated live behavior, trace inspection, and
  fresh-database multi-turn testing.
- `control-console-web-development`: govern console contract/UI changes and
  browser verification.

## Mandatory Rules

- After automatic context compaction, reread this plan and both mandatory
  companions before implementation, verification, handoff, or reporting.
- After signing off each major checklist checkpoint, reread all three documents
  before starting the next checkpoint.
- Before completion, lifecycle change, merge, or sign-off, the parent agent
  must run the Independent Code Review gate and record it in Execution
  Evidence.
- This drafting agent stops after the planning handoff. A future implementation
  turn begins only after explicit user authorization and then uses the
  parent-led native subagent model in Execution Model. If native subagents are
  unavailable, stop until the user authorizes a fallback.
- Use `venv\Scripts\python`; install no package without first applying
  `python-venv` and receiving scope consistent with the active task.
- Read `git status --short`, root `README.md`, `docs/HOWTO.md`, relevant
  subsystem READMEs, source, and tests before production edits. Preserve all
  unrelated worktree changes.
- Use `apply_patch` for manual edits. Use `rg` for inventory. Keep Python at
  PEP 8 and project `py-style`; required internal values fail fast rather than
  receiving invented defaults.
- Never read `.env` without explicit user instruction.
- Stage 3 uses the configured `MONGODB_URI` and exact database name
  `_test_kazusa_core_v2` as its database-level isolation boundary. It
  never inspects production rows or writes/drops any database other than that
  exact reserved name.
- The URI database path must be empty or exactly `_test_kazusa_core_v2`.
  The harness injects the exact name into each child process, and the DB
  package guard rejects every other name before opening a client.
- LLMs own semantic appraisal, character judgment, action preference, and
  final wording. Deterministic code owns schemas, availability probes,
  permissions, limits, persistence, retry safety, scheduling, and delivery.
- RAG returns evidence. It does not author persona, final stance, or dialog.
- Internal monologue residue and promoted reflection are evidence only. Their
  existence is never sufficient to initiate cognition or visible speech. A
  native `internal_thought` episode additionally requires a durable active
  action latch/continuation from a previously settled episode.
- Source selection is deterministic from a real runtime event; prompt text may
  not select or rewrite the trigger source.
- Adapters remain thin and platform-neutral. Brain code consumes typed
  envelopes and must not parse QQ, Discord, or debug-wire syntax as its main
  contract.
- Prompts changed by Stage 3 are written in Chinese, use configured character
  and user names instead of mixed-language `self`, and read organically.
  Prompt changes express ownership and output contracts rather than repeated
  strengthening words or case-specific suppression.
- Deterministic pre/post-processing must not classify, rewrite, accept, or
  reject user meaning. Mechanical schema validation remains required.
- Action descriptions are organically discouraged at the dialog surface and
  remain valid model output; deterministic stripping or rejection is
  prohibited.
- The visual surface agent remains disabled by default and terminal: it may
  produce future local text-to-image directives and has no downstream agent.
- No new response-path LLM stage, larger model-output cap, resolver-cycle cap,
  or increase to an existing retry cap is permitted. The new internal-action
  latch uses the fixed three-attempt lifecycle in the execution companion. Its
  budget is a hard boundary.
- Failures use typed categories. Every raw LLM response first uses the
  canonical JSON parser. Non-recoverable structural or semantic-contract
  errors receive bounded regeneration or complete replacement from the
  producing stage; after its explicit cap the boundary fails closed with a
  typed terminal result. Recoverable bound violations use only the contract's
  deterministic normalization followed by revalidation. Invalid candidates
  stay out of affect, action, persistence, scheduling, dialog, and delivery
  paths. Transient failures retry only at the existing safe checkpoint; content
  drift remains evidence and is never deterministically rewritten.
- Compatibility aliases, dual trigger vocabularies, legacy fallbacks, and
  scaffold-to-native translation layers are prohibited.
- Generated review artifacts contain sanitized prompts/outputs and identifiers
  only. Protected traces remain under existing access and redaction rules.

## Must Do

1. Establish automatic, idempotent character profile bootstrap before any
   worker, queue consumer, scheduler, or request can reason.
2. Replace the six historical episode labels with five grounded canonical
   runtime sources: `user_message`, `internal_thought`, `self_cognition`,
   `scheduled_tick`, and `tool_result`.
3. Preserve `internal_thought` as a native action-latch-driven source, retire
   `reflection_signal` as a trigger source, and preserve reflection/residue as
   bounded evidence lanes.
4. Give each canonical source one public episode builder, one runtime owner,
   typed origin metadata, privacy/delivery rules, and continuation limits.
5. Keep the action registry as declarative schema/permission/prompt/handler
   authority; make deterministic probes own availability, the evaluator own
   authorization, handlers own effects, and action results plus the single
   settlement owner record outcomes and trace identity.
6. Register every currently usable action, including accepted-task and
   background-work actions; remove separate allowlists as authorities.
7. Probe deterministic runtime affordances before cognition and recheck them
   immediately before execution, preserving current action-selection capacity.
8. Route every canonical source through the same V2 facade and resolver loop,
   with source-specific evidence and output policy rather than separate
   reasoning scaffolds.
9. Settle exactly one `EpisodeTraceV2` per attempted episode through one
   post-turn owner, including visible, silent, action-only, retried, failed,
   and non-delivered outcomes.
10. Feed only the settled trace to consolidation, state updates, progress,
    scheduler follow-through, and operational surfaces.
11. Retire dry-run reflection cognition, standalone internal-thought cognition,
    and proactive-output scaffold modules after their real owners are covered.
12. Update health, status, trace export, control console, operator scripts, and
    documentation to expose the canonical source/action/trace contracts.
13. Prove cold bootstrap, multi-source reasoning, persistence, restart, and
    operational inspection against an absent dedicated database.
14. Freeze the native Stage 4 target schema and produce a preliminary legacy
    source-code inventory without reading or changing production data.
15. Run all verification and independent-review gates and record inspectable
    evidence before proposing Stage 3 completion.
16. Retain the Chinese emotion, abuse-boundary, mechanical sadness, and
    bounded-error real-LLM evidence overlay as a mandatory input to final Stage
    3 user-quality sign-off.

## Deferred

- Production database connection, collection discovery, sampling, backup,
  transformation, validation, cutover, cleanup, and rollback are Stage 4.
- Exact production source-to-target row counts and transforms are Stage 4
  discovery outputs; Stage 3 freezes only the native target.
- Production adapter rollout is deferred until Stage 4 establishes a validated
  candidate database.
- New action kinds, new resolver capabilities, new autonomous-contact policy,
  and new LLM stages are outside Stage 3.
- Reopening accepted Stage 2 emotion formulas, dialog policy, or personality
  tuning requires a demonstrated unacceptable conflict/fatal failure and a
  plan amendment. Ordinary quality drift is review evidence, not automatic
  scope expansion.
- A standalone wheel containing bundled character profiles is outside Stage 3;
  deployed processes receive an explicit absolute profile path.

## Cutover Policy

Stage 3 uses a big-bang contract cutover on the fresh-database environment:

1. Focused tests define the new contracts and initially fail.
2. Canonical episode, availability, and trace contracts land.
3. All runtime producers and consumers move to those contracts.
4. Fresh bootstrap and service wiring land.
5. Legacy scaffolds and duplicate vocabularies are deleted.
6. Deterministic, database, live-LLM, console, and restart gates pass.
7. The native schema manifest is frozen for Stage 4.

No feature flag selects old versus new cognition. Rollback during Stage 3 is a
branch/revision rollback against the disposable database, followed by database
recreation. Production runtime and production data remain unchanged.

## Target State

```text
adapter/debug event
  -> brain service intake or grounded background runtime owner
  -> CognitiveEpisodeV1
  -> source-specific evidence projection
  -> available resolver/action affordances
  -> Cognition Core V2 bounded resolver
  -> authorized action execution and/or L3 surface
  -> one settled EpisodeTraceV2
  -> consolidation/state/progress/scheduler/audit
  -> adapter delivery receipt when a visible surface exists
```

On an absent database, service startup loads and validates the configured
static character profile, inserts it idempotently, bootstraps native indexes
and singleton state, then starts runtime workers. Runtime-owned relationship,
emotion, memory, progress, reflection, scheduler, task, and trace data begin
empty and are built only from subsequent episodes.

The five canonical sources are:

| Source | Grounded event | Runtime owner |
|---|---|---|
| `user_message` | accepted typed inbound message/settled turn | brain service intake/persona path |
| `internal_thought` | active action latch/continuation emitted by a previously settled episode | self-cognition runner through the public internal-thought builder |
| `self_cognition` | eligible non-scheduled idle source case with observed context | self-cognition runner |
| `scheduled_tick` | claimed due calendar commitment | calendar scheduler/self-cognition bridge |
| `tool_result` | durable accepted/background task result ready for re-entry | accepted-task/background delivery owner |

No source exists solely because an internal cognition window or monologue
residue exists. Reflection remains an offline producer of promoted evidence.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Profile deployment | `CHARACTER_PROFILE_PATH` is required and absolute; Docker sets `/app/personalities/kazusa.json` | avoids repository-root inference and duplicate packaged profile authorities |
| Seed behavior | validate one static `CharacterProfileSeedV1`; reject runtime-owned fields; insert only when absent; verify identity when present | deterministic fresh start without overwriting learned state |
| Trigger roster | five grounded sources; `internal_thought` requires a durable prior action latch, while reflection/residue alone remain evidence | preserves the authoritative source capacity without self-referential firing |
| Source migration | one big-bang vocabulary update | avoids aliases and mixed consolidation policy |
| V2 entrypoint | all sources call the existing public V2 facade | preserves one reasoning implementation |
| Availability | registry entries expose deterministic health/permission probes | cognition sees real capabilities without semantic availability guesses |
| Execution recheck | evaluator rechecks the selected capability before side effects | closes race between prompt projection and execution |
| Trace settlement | `brain_service.post_turn` owns one settled trace builder; `action_spec.results` remains a pure result/trace schema helper | removes competing trace owners |
| Consolidation | consumes settled trace only | aligns persistence with what actually happened |
| Dialog | Stage 2 Chinese organic prompt remains; upstream passes semantic intent/content, not action-description guidance | protects vivid character output without mechanical tuning |
| Visual directives | disabled by default and terminal | preserves future local text-to-image boundary |
| Database safety | same configured URI with exact reserved database name, URI-path consistency, child-process guard, and exact-name cleanup | user-authorized database-level isolation where a second endpoint is unavailable |
| Goal answerability | Cognition Core emits a validated `goal_resolution` for the accepted user goal: `answerable_now`, `requires_required_evidence`, `requires_user_input`, or `blocked` | separates user-goal sufficiency from source-specific evidence coverage |
| Evidence status | RAG `resolved` fields retain source-coverage meaning; resolver observations expose capability/evidence outcomes without converting missing optional evidence into goal failure | preserves RAG ownership and prevents unnecessary resolver cycles |
| Resolver termination | `answerable_now` settles the current episode without optional retrieval; required-evidence and user-input states retain resolver/clarification paths; deterministic code enforces the typed decision and cycle/no-progress limits | addresses the private-18 loop and reduces P95 without semantic keyword routing |
| Latency acceptance | User-approved fixed-sequence ordinary-response foreground p95 ceiling of 120 seconds; individual blocking LLM calls remain bounded at 120 seconds; the maximum case remains separately reported | accepts justified complex-tail latency without raising LLM or resolver budgets |
| Remediation boundary | modify the existing Cognition Core/action/resolver contracts and prompts within the approved Stage 3 change radius; add no LLM stage, cap increase, compatibility vocabulary, or deterministic answerability classifier | keeps the fix contract-first and bounded |
| Stage 4 boundary | Stage 3 freezes native targets; Stage 4 discovers real production sources and performs migration | prevents unverified production assumptions |

### Approved Decision Record — 2026-07-19

The user approved implementation of the answerability-separation mitigation after
reviewing the Stage 3 traces and failure-mode analysis. The accepted decision is:

1. A source-specific RAG result such as `conversation_evidence.resolved=false`
   records evidence coverage only. It does not decide whether the accepted user
   goal can be answered.
2. Cognition Core remains the semantic owner of the accepted user goal and must
   emit the typed `goal_resolution` decision. The resolver remains the execution
   owner for explicitly requested capabilities and typed observations.
3. A general answer that is sufficient with the current user input, character
   judgment, monologue, and available context may settle as `answerable_now`
   even when optional retrieval is empty or unrelated.
4. A goal that genuinely depends on a missing required fact keeps the
   `requires_required_evidence` path. A goal that requires the user to supply
   information keeps the `requires_user_input` path. Technical inability remains
   `blocked`.
5. Deterministic code validates and enforces this LLM-owned decision, prevents
   repeated no-progress capability requests, and preserves the existing maximum
   of three resolver cycles. It does not infer answerability from keywords or
   rewrite the LLM decision after the fact.
6. Verification must include focused failing-then-passing contract tests, the
   affected deterministic suites, one-at-a-time real-LLM coverage of the
   previously divergent private-18 shape and a required-evidence shape, and a
   refreshed sequential Stage 3 latency/call/quality/trace ledger. The existing
   40-case p95 gate remains binding; this decision does not approve Stage 3
   signoff by itself.

### Follow-up Decision Record — 2026-07-19

The first post-remediation live checks add one bounded prompt-contract
refinement:

1. `private_18` reached a complete direct answer with eight LLM calls and no
   resolver stage, demonstrating the `answerable_now` separation behavior.
2. `private_08` correctly asked for the missing referent, but repeated the
   failed resolver request before surfacing that clarification. When resolver
   context reports that a required referent or user-provided detail is
   missing, the action-planning prompt must direct the LLM-owned decision to
   `requires_user_input` and omit another resolver request.
3. This refinement remains inside the existing Cognition Core prompt and
   contract boundary. Deterministic code continues to validate and bound the
   decision; it does not classify the user text or rewrite the LLM decision.
4. The two live cases must be rerun one at a time, followed by the binding
   sequential latency/call/quality/trace ledger. The existing 40-case p95 gate
   remains binding.

### Follow-up Decision Record — capability observation refinement — 2026-07-19

The required-evidence trace identified a projection defect: local context
recall logs an unresolved referent and skips retrieval, but the typed
observation is currently shaped like an empty successful recall. The local
context capability will therefore emit a bounded `blocked` observation with a
prompt-safe user-input-required summary when its structured referent precondition
is missing. This improves the semantic context presented to Cognition Core;
deterministic code still does not assign or rewrite `goal_resolution`, and the
LLM remains the owner of choosing `requires_user_input`.

### Follow-up Decision Record — direct-answer triage wording — 2026-07-19

The updated live probes preserve semantic safety but still show resolver
over-selection: a general relationship question used one resolver pass in a
13-call run, and the unresolved-referent clarification used 19 calls. The
action-planning prompt will state the triage boundary more directly: a general
question, opinion, analysis, or advice request that can be answered from the
accepted bid, current input, monologue, and available context defaults to
`answerable_now`; resolver availability, an empty/failed optional source, or
missing unrelated evidence is not sufficient reason to request retrieval.
Only a fact explicitly necessary to complete the answer, or information that
the user must provide, selects `requires_required_evidence` or
`requires_user_input`. The decision remains LLM-owned and deterministic code
continues to validate/enforce only the typed result and recurrence bounds.

### Follow-up Decision Record — independent review remediation — 2026-07-19

The independent implementation review identified four remaining remediation
items and one accepted configuration disposition:

1. The unresolved-referent local-context capability will emit a typed
   user-input blocker boundary in its observation. After that observation,
   the resolver may run one final Cognition Core pass so the LLM can produce
   the clarification surface. If that pass repeats the blocked capability or
   emits neither a clarification action nor another resolver request,
   deterministic loop control will clear the request and settle a prompt-safe
   clarification surface for a user-message episode. Non-user sources remain
   private. This bounds the required-user-input path without assigning or
   rewriting `goal_resolution` from user text.
2. A deterministic contract test will exercise an answerable goal alongside
   a `conversation_evidence` result whose source-specific `resolved` value is
   false, proving that source coverage does not force an optional resolver
   request.
3. MongoDB connection diagnostics will log only a sanitized endpoint
   description; credentials, query options, and the raw URI remain out of
   logs.
4. The change-radius companion will explicitly include the direct contract
   fixture files touched by this remediation.
5. The Stage 3 harness intentionally reads the user's configured `.env`
   through `load_dotenv` when no explicit mapping is supplied. The exact
   reserved database name, URI-path consistency check, child-process guard,
   and endpoint fingerprint remain the isolation controls; the assistant does
   not inspect or copy `.env` contents.

The 40-case p95 gate, broader non-live regression failures, Browser session
acceptance, and final user sign-off remain open gates. This record authorizes
only the bounded remediation above and does not approve Stage 3 completion.

### Follow-up Decision Record — live trace finalization polling — 2026-07-19

The sequential live refresh exposed a harness timing false negative: `group_17`
and `group_18` both produced a visible response and a clean skipped lifecycle,
but their trace records were still `running` when the fixed 15-second poll
expired. The service completed post-turn work immediately afterward. The live
harness will therefore use a 60-second terminal-trace poll window. This changes
only test observation tolerance; it adds no runtime stage, LLM call, resolver
cycle, or latency-budget allowance. The 40-case p95 gate and all other signoff
gates remain binding.

### User Decision Record — latency ceiling — 2026-07-20

The user approved expanding the fixed Stage 3 ordinary-response foreground p95
acceptance ceiling to 120 seconds. This changes the acceptance threshold only;
it adds no LLM stage and raises no LLM, prompt, output, repair, retry, or
resolver-cycle cap.

The refreshed 40-case sequence recorded a 103,807 ms (103.8-second) nearest-rank
p95, which passes the revised 120,000 ms ceiling. Its 125,418 ms (125.4-second)
maximum remains a separately reported tail metric and is not treated as a p95
criterion. The individual blocking LLM-call limit remains 120 seconds. Broader
regression reconciliation, Browser acceptance, character-quality review, and
final user sign-off remain open.

### User Decision Record — emotion and bounded-error sign-off overlay — 2026-07-21

The recently run Chinese emotion and abuse probes are retained as a mandatory
Stage 3 user-quality sign-off overlay. They supplement the frozen 40-case
source sequence and do not change the production call, output, retry, or
resolver-cycle budgets.

The overlay preserves the following evidence boundaries:

- `sadness` can reach visible crying without a `crying` emotion id; fear and
  shame also rendered crying after explicit emotional permission.
- The tested loneliness and anger arms rendered vulnerability or exhaustion,
  not crying. High-attachment and sustained-abuse natural arms remained on
  anger, with relational cutoff changing visible stance toward cold separation.
- A mechanically established negative abuse outcome reached very-high sadness
  and visible Chinese grief/relationship-loss dialog through real `/chat`.
- Outcome-neutral abuse inference emitted a signed `+5` despite loss language;
  this is retained as a model/prompt quality failure and is never rewritten by
  deterministic code into sadness.
- Structural, unsupported, conflicting, or unsafe bound candidates are excluded
  from affect evidence. The producing semantic stage owns bounded regeneration
  or complete replacement; after its cap the boundary fails closed with a typed
  terminal trace. Only contract-defined safe normalization may repair a bound
  violation.

The complete test inventory, one-at-a-time commands, raw evidence, and the
positive/negative dispositions are recorded in
`test_artifacts/cognition_core_v2/stage_3/emotion_boundary_bounded_error_signoff_review.md`.
This decision keeps Stage 3 in progress pending the existing regression,
independent-review, Browser, and final user-signoff gates.

## Data Migration

Stage 3 performs no data migration. Its database operation is a cold bootstrap
against the exact reserved database selected from the configured MongoDB URI.
The test harness inspects only that database, asserts it has no persistent
collections before the first service start, records native collections/indexes
after exercised behavior, restarts against the same URI/database session, and
drops only that exact database during explicit cleanup.

The resulting `Stage3NativeSchemaManifestV1` is a Stage 4 input. It records
collection/index/schema ownership and representative sanitized shapes generated
by Stage 3. It does not claim that production contains matching legacy rows.

## Contracts And Data Shapes

The mandatory companions freeze:

- `CharacterProfileSeedV1` and startup order;
- authoritative `TriggerSourceSpecV1` registry and `CognitiveEpisodeV1`
  envelope with five source-specific origin records;
- capability availability and registry/evaluator contracts;
- settled trace ownership and outcome matrix;
- `Stage3FreshDatabaseEvidenceV1` and `Stage3NativeSchemaManifestV1`;
- forbidden legacy shapes and source labels.

The implementation must use those exact names and invariants. Public callers
depend on public builders/registry functions, never private profile, prompt,
storage, or scaffold internals.

## LLM Call And Context Budget

Stage 3 adds no LLM stage and increases no call, repair, resolver-cycle, output,
or retry cap. The execution companion contains the route-by-route formula,
context inputs, 50k-token cap, 50,000-character per-call projection ceiling,
truncation policy, blocking behavior, and test selectors.

Completion thresholds on the fixed fresh-database proof are:

- zero fatal technical failures;
- zero unacceptable within-message/user-input/subject conflicts in the final
  user sign-off set;
- ordinary response-path p95 at or below the approved 120-second ceiling;
- no episode exceeds the existing configured resolver maximum of three cycles;
- no `KazusaLiveBot is busy right now, please try again later.`-class collapse;
  safe-checkpoint retry/fallback must produce a typed terminal outcome;
- background/source episodes complete or reach a typed terminal outcome within
  300 seconds, without blocking chat intake;
- each projected LLM call stays within 50,000 characters and the 50k-token cap.

The emotion/boundary sign-off overlay is test-only evidence. It adds no
response-path production stage, output cap, retry cap, resolver-cycle cap, or
runtime behavior. Its exact one-at-a-time selectors and disposition rules are
in the execution companion's
`emotion_boundary_bounded_error_signoff_review.md`.

Quality drift and make-up content are recorded for user judgment and do not
stop the run. Live cases run sequentially and are evaluated only after the
planned sequence completes, except fatal technical evidence is captured for
immediate repair.

## Change Surface

Target ownership boundary: the brain service's episode-to-settled-trace path,
including the grounded background producers and the fresh-profile bootstrap it
requires. The change-radius companion provides the exact Create/Modify/Delete/Keep inventory
and symbol purpose for every affected runtime, test, console, script, document,
and Stage-4-only file.

The radius includes these ownership groups:

- startup/config/profile/database bootstrap and Docker deployment;
- canonical episode builders, V2 facade/resolver, action registry/evaluator,
  L3/dialog, and trace/result settlement;
- RAG projection, relevance, consolidation, conversation progress, and state;
- self-cognition, calendar, accepted tasks, background work, and reflection
  evidence promotion;
- brain health/status, event logging, protected tracing, control console, and
  diagnostic scripts;
- all tests/documents containing retired source vocabulary or manual bootstrap
  assumptions.

The only Stage-4-only legacy files allowed to retain migration vocabulary are:

```text
src/kazusa_ai_chatbot/db/script_operations.py
src/scripts/_lane_cleanup.py
src/scripts/migrate_scheduled_events_to_calendar_scheduler.py
```

Normal service imports, Stage 3 scripts, and Stage 3 tests must not import or
execute those files.

## Overdesign Guardrail

- Actual problem: Stage 2's V2 chat chain is not yet the single deployable
  reasoning contract for all runtime sources on an empty database.
- Minimal change: add deterministic profile bootstrap, normalize episodes to
  five grounded sources, centralize existing capability availability and trace
  settlement, move current callers to the public V2 path, and delete superseded
  scaffolds.
- Ownership boundaries: LLM stages own semantic judgment and wording;
  deterministic runtime owners select grounded sources, validate contracts,
  probe capabilities, authorize/execute actions, settle traces, persist state,
  schedule work, and deliver adapter output.
- Rejected complexity: trigger aliases, compatibility mappers, second action
  registries, generic plugin frameworks, prompt-selected source types,
  monologue-driven autonomous triggers, new LLM stages, new retries, automatic
  production discovery, packaged-profile duplication, and adapter cognition.
- Evidence threshold: any rejected dimension requires an observed Stage 3
  blocker or an approved near-term integration that cannot use the frozen
  contract, followed by an explicit plan amendment and user approval.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve this plan and both companion contracts.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, extra features,
  or extra agents.
- Changes outside the change-radius companion's exact inventory require a documented plan
  amendment and approval before editing.
- Explicitly listed deletions require reference greps and affected tests; they
  do not require a compatibility shim.
- Before implementing any helper, search for existing equivalent behavior and
  move/abstract it under the named owner rather than duplicate it.
- Avoid unrelated cleanup, formatting churn, dependency changes, prompt
  rewrites, and broad refactors.
- If plan and code disagree, preserve the stated contract, record the mismatch,
  and amend the plan before changing a public boundary.
- If a mandatory instruction cannot be satisfied, stop at the current safe
  checkpoint and report the blocker.

## Implementation Order

Detailed contracts, file boundaries, and test steps are in the companions. The
required order is:

1. The future authorized parent records the clean-diff boundary, verifies
   isolated DB inputs, writes
   focused failing tests for profile seed, source contracts, capability
   availability, and one-trace settlement, and records the failures.
2. That parent starts exactly one production-code subagent with the approved plan,
   companions, mandatory skills, test contract, and production-only boundary.
3. The parent prepares integration/fresh-DB tests while the subagent implements the
   profile/bootstrap and canonical episode contracts.
4. Production subagent moves action availability and trace settlement under
   their single owners, then migrates all runtime producers/consumers.
5. Production subagent updates operations/console/scripts/docs and deletes
   scaffolds only after zero-reference greps pass.
6. The parent runs focused tests, affected deterministic suites, isolated live DB
   cold-start/restart proof, and sequential real-LLM source/quality cases.
7. The parent records native schema, traces, latency/call counts, and anti-cheat
   evidence; remediates in-scope test/integration findings.
8. The parent starts exactly one independent code-review subagent after planned
   verification passes.
9. The parent remediates review findings inside scope, reruns affected gates, seeks
   user quality sign-off, and updates lifecycle/evidence.

## Execution Model

- The future explicitly authorized parent agent owns orchestration, test code,
  verification, evidence, review
  remediation, lifecycle updates, and final sign-off.
- Parent establishes focused tests and records expected failures/baselines
  before production implementation begins.
- Production-code subagent: exactly one native subagent, started after the test
  contract; owns production code only and closes after planned production edits
  are complete, excluding review fixes.
- Parent may continue integration tests, static checks, and evidence while the
  production subagent works.
- Independent code-review subagent: exactly one different native subagent,
  started after verification; reviews plan, companions, full diff, and evidence;
  reports findings and makes no edits.
- Parent routes in-scope production review fixes back to the same production
  subagent when it remains available; parent owns test/evidence/doc remediation.
- User performs the final character-quality review and authorizes Stage 3
  completion. Plan status alone grants no production-code authority.

## Progress Checklist

- [x] A — baseline and focused contracts established.
  - Files/steps: execution companion A1-A6; focused Stage 3 tests only.
  - Verify: exact A commands in the execution companion fail for the expected missing
    contract, and DB guards pass without starting the service.
  - Evidence: changed-file baseline, occurrence inventory, call/latency
    baseline, DB fingerprints, expected test failures.
  - Handoff: start Checkpoint B and the production-code subagent.
  - Sign-off: `parent/2026-07-19`.
- [x] B — profile bootstrap and native database base complete.
  - Files/steps: execution companion B1-B8.
  - Verify: profile/config/bootstrap focused tests and cold-start bootstrap.
  - Evidence: seed validation, idempotence, mismatch rejection, native indexes.
  - Handoff: reread plan; start C.
  - Sign-off: `parent/2026-07-19`.
- [x] C — canonical source contracts complete.
  - Files/steps: execution companion C1-C8.
  - Verify: episode/RAG/origin/source-policy tests and retired-label greps.
  - Evidence: five source packets and zero unapproved runtime source labels.
  - Handoff: reread plan; start D.
  - Sign-off: `parent/2026-07-19`.
- [x] D — capability availability/action path complete.
  - Files/steps: execution companion D1-D8.
  - Verify: registry/evaluator/handler/resolver focused tests.
  - Evidence: availability matrix, race recheck, typed unavailable outcomes.
  - Handoff: reread plan; start E.
  - Sign-off: `parent/2026-07-19`.
- [x] E — trace settlement and persistence complete.
  - Files/steps: execution companion E1-E9.
  - Verify: trace/consolidation/progress/delivery tests.
  - Evidence: one settled trace for every outcome row.
  - Handoff: reread plan; start F.
  - Sign-off: `parent/2026-07-19`.
- [x] F — all background/runtime producers adopted and scaffolds deleted.
  - Files/steps: execution companion F1-F9.
  - Verify: self-cognition/calendar/task/background/reflection tests plus
    zero-reference/delete greps.
  - Evidence: source-to-owner matrix exercised; deletion inventory clean.
  - Handoff: reread plan; start G.
  - Sign-off: `parent/2026-07-19`.
- [ ] G — operations, console, scripts, and documentation aligned.
  - Files/steps: execution companion G1-G8.
  - Verify: ops/trace/script tests and control-console browser checks.
  - Evidence: screenshots/contract captures, redaction proof, doc-link check.
  - Handoff: reread plan; start H.
  - Sign-off: `<parent/date>`.
- [x] H — fresh-database E2E and real-LLM proof complete.
  - Files/steps: execution companion H1-H11.
  - Verify: cold start, five sources, restart, final sequential quality set.
  - Evidence: `Stage3FreshDatabaseEvidenceV1`, monologue/dialog/action/trace
    review, call/latency ledger, native schema manifest.
  - Handoff: reread plan; start I.
  - Sign-off: `parent/2026-07-19`.
- [ ] I — full deterministic regression and Stage 4 handoff complete.
  - Files/steps: execution companion I1-I6.
  - Verify: all exact I commands, `git diff --check`, link and placeholder
    scans.
  - Evidence: regression totals, allowed exceptions, Stage 4 input paths.
  - Handoff: reread plan; start J.
  - Sign-off: `<parent/date>`.
- [ ] J — independent code review and user sign-off complete.
  - Scope: plan/manifest alignment, full diff, architecture/style, DB safety,
    budgets, tests, evidence, deletions, Stage 4 separation, and the
    emotion/boundary/bounded-error sign-off overlay.
  - Verify: rerun every gate affected by review fixes.
  - Evidence: reviewer identity/findings, remediation, reruns, residual risks,
    user quality decision, lifecycle update.
  - Handoff: archive Stage 3 only after all findings and user gates close.
  - Sign-off: `<parent/reviewer/user/date>`.

Current execution evidence and external gate status are recorded in
`test_artifacts/cognition_core_v2/stage_3/checkpoint_i_verification_summary.md`.
Checkpoints B-F and the technical H gates are complete. The revised 120-second
p95 latency gate passes for the frozen sequence; repository-wide contract
reconciliation is still open, in-app Browser acceptance is environment-pending,
and final character-quality/user sign-off remains open.

## Verification

The execution companion lists exact commands, selectors, expected matches, and allowed
exceptions. Required gate groups are:

1. static inventory and forbidden-reference greps;
2. focused contract and bootstrap tests;
3. affected deterministic runtime suites;
4. isolated live Mongo cold-start/restart tests;
5. one-at-a-time real-LLM source, action, dialog, and continuity tests;
6. control-console API/browser and redaction tests;
7. 40-turn final sequence only after the earlier gates pass;
8. schema/evidence validation, diff check, and Markdown link check.

No pytest file-level live-LLM batch is permitted. Each test-process invocation
targets one case. The final PowerShell orchestration loops over 40 one-case
processes, and the run ledger flushes command, trace id, result, latency, and
technical status before the next process starts. Content evaluation begins
after the full planned sequence finishes.

## Independent Plan Review

Before approval, a reviewer independent from the drafting pass must inspect:

- this plan, both mandatory companions, Stage 2 closure evidence, the Stage 4
  placeholder, relevant source/tests, and current git state;
- architecture ownership, fresh-DB isolation, source completeness, action
  capacity, trace completeness, LLM budgets, exact change radius, commands,
  checklist granularity, deletion safety, and Stage 4 non-overlap;
- placeholder/creativity scans for unresolved choices, broad verbs,
  compatibility/fallback freedom, missing files, or stale names.

All blockers must be fixed before status changes to `approved`. Review evidence
records reviewer identity, findings, changes, and final disposition.

## Independent Code Review

After verification and before completion, exactly one independent review
subagent receives the approved plan/companions, complete implementation diff,
all test/static/DB/live-LLM evidence, generated schema manifest, Stage 4
placeholder, and lifecycle records.

The reviewer checks:

- project/skill rules, Python/CJK/test style, fail-fast contracts, and no
  unrelated changes;
- five-source ownership, no prompt-selected triggers, one action authority,
  one trace settlement owner, and correct LLM/deterministic boundaries;
- fresh-DB database-name isolation, seed idempotence, runtime state ownership,
  restart continuity, and absence of production-row access;
- call/context/latency budgets, bounded safe-checkpoint recovery, anti-cheat
  traces, visual terminal behavior, and dialog upstream contract;
- exact deletion/reference evidence and Stage 4 handoff completeness.

The reviewer reports only. The parent remediates findings inside the approved
surface and reruns affected gates. Contract/scope expansion requires plan
amendment and approval.

## Acceptance Criteria

1. Service startup on the guarded absent database automatically creates the
   validated configured character and native indexes before reasoning begins.
2. Repeated startup is idempotent; a conflicting existing identity fails fast;
   runtime-owned state is never overwritten from the profile file.
3. Every cognition episode uses exactly one of the five canonical sources and
   one source-specific origin record.
4. Internal monologue/reflection can inform a grounded episode but cannot
   initiate one; `internal_thought` additionally proves a durable prior action
   latch/continuation.
5. Chat, internal thought, self-cognition, scheduled commitments, and tool
   results all enter the same public V2 facade and source-aware resolver path.
6. The action registry exposes every supported production capability as the
   declarative authority; probes, evaluator, handlers, results, and settlement
   retain their exact availability/authorization/effect/outcome ownership.
7. Availability is projected before cognition and rechecked before side
   effects without reducing baseline action-selection capacity.
8. Every attempted episode produces one settled trace; consolidation and
   continuity consume that trace and preserve private/visible boundaries; the
   later memory-lifecycle task writes one linked idempotent lifecycle record
   without rebuilding the trace.
9. Dry-run/internal-thought/proactive-output scaffolds and all normal-runtime
   references to them are removed.
10. Health, status, traces, console, scripts, and documentation expose the
    canonical contracts without protected-data leakage.
11. Fresh-DB cold start, five-source exercise, persistence, restart, final
    sequential real-LLM review, and the retained emotion/boundary/bounded-error
    overlay pass the technical and budget gates or have an explicit user-
    reviewed residual disposition.
12. The final user review finds no fatal failures or unacceptable content
    conflicts; acceptable/encouraged drift remains available to character
    judgment.
13. `Stage3NativeSchemaManifestV1` and the preliminary legacy code inventory
    are complete Stage 4 inputs.
14. Independent code review has no unresolved blocker, and the user explicitly
    signs off Stage 3 completion.

## Risks

- Canonical label cutover touches many persisted/test expectations. Big-bang
  greps and source-specific tests prevent mixed vocabularies.
- Static profile data can overwrite learned state if ownership is blurred.
  Strict seed schema and insert-only bootstrap prevent it.
- A second MongoDB endpoint is unavailable in the current environment. The
  exact reserved database name, URI/database consistency, child-process guard,
  session fingerprint continuity, and exact-name cleanup are the approved
  database-level isolation controls.
- Capability health can change between planning and execution. Deterministic
  pre-projection and execution-time recheck provide truthful selection.
- Parallel/background paths can double-settle or lose traces. One settlement
  owner and outcome-matrix tests make cardinality explicit.
- Weak local models can regress under prompt accumulation. Stage 3 adds no LLM
  stage, keeps Chinese organic prompts, caps context, and verifies broad
  sequential behavior before evaluation.
- Real-LLM latency is environment-sensitive. The ledger separates technical
  timeout/retry evidence from content judgment and blocks completion on the
  stated ceilings.

## Execution Evidence

- Planning evidence: Stage 2 artifacts and source inventory linked above.
- Independent plan review: one independent review completed on 2026-07-18;
  its bootstrap, latch, episode-input, affordance, trace, budget, command,
  change-radius, and Stage 4 inventory blockers were incorporated. Its final
  re-review then identified boundary-enum, post-turn-record, action-roster,
  source-owner, and latch-API gaps; those are corrected in this revision. Per
  the user's instruction, further agent review has stopped. The user authorized
  execution on 2026-07-19; implementation is now in progress.
- Focused test baseline: Checkpoint A recorded 15 expected contract failures and
  5 fresh-database guard passes in
  `test_artifacts/cognition_core_v2/stage_3/checkpoint_a_focused_tests_baseline.txt`.
- Checkpoint A occurrence inventory and pre-service environment guard evidence
  are recorded under `test_artifacts/cognition_core_v2/stage_3/`.
- Checkpoint A call/latency baseline is recorded in
  `test_artifacts/cognition_core_v2/stage_3/checkpoint_a_call_latency_baseline.md`.
- Database-isolation amendment: user authorization is recorded above; the
  harness and DB guard now use the ordinary URI plus the exact reserved name.
- Fresh-database bootstrap contracts and safe report/native-manifest generation:
  verified; the configured `_test_kazusa_core_v2` database is now the
  database-level isolation target. The guarded output directory completed
  cold start and all 20 sequential group cases against that exact database;
  the previous populated database was not touched.
- Deterministic tests and static gates: current evidence is recorded in
  `test_artifacts/cognition_core_v2/stage_3/checkpoint_i_verification_summary.md`.
- Answerability remediation evidence: the focused contract/capability probe
  passed, including typed blocker, one-final-cognition convergence, source
  separation, and URI-redaction coverage. The clean follow-up sequence under
  `test_artifacts/cognition_core_v2/stage_3/answerability_remediation_followup2/`
  recorded `private_18` at 8 calls and 65.7 seconds with no resolver stage and
  a complete direct answer; `private_08` at 11 calls and 91.6 seconds with one
  blocked local-context attempt and a semantically correct raw-language
  clarification. The first post-blocker retry exposed a silent final pass and
  is preserved in the prior follow-up directory; the typed fallback then
  corrected it. The direct-answer and required-user-input convergence
  improvements are demonstrated; the full sequential p95 gate remained open
  under the then-current baseline-relative threshold.
  The deterministic source-separation regression supplies a
  `conversation_evidence.resolved=false` row, preserves an LLM-owned
  `answerable_now` decision, and passes with one cognition call and zero
  capability executions. The latest affected deterministic rerun recorded
  135 passes, one preexisting route-expectation failure, and seven deselected
  tests.
- Rejected-tail remediation evidence (2026-07-20): pending HIL/approval
  resumes now require an exact typed reply to the pending source message, and
  the action-planning prompt keeps direct character self-reports out of
  optional local-context retrieval. Focused tests recorded 45 passes. Clean
  guarded live evidence recorded `group_16` at 7 calls and 72.4 seconds with
  no resolver stage, and `private_03` at 8 calls and 70.8 seconds with no
  local-context resolver stage; each produced one settled trace and one
  lifecycle record. The detailed evidence is under
  `test_artifacts/cognition_core_v2/stage_3/answerability_remediation_rejected_clean/`.
- Live database and restart: exact guarded cold start passed, followed by
  all 40 frozen group/private sequence cases. Each clean result produced one
  terminal visible trace, one lifecycle record, and a succeeded persisted
  trace. The refreshed answerability sequence captured 313 LLM calls, a
  75.6-second average foreground duration, a 103.8-second nearest-rank p95,
  and a 125.4-second maximum. The user-approved 120-second p95 ceiling is
  satisfied; the 125.4-second maximum remains a separately reported tail
  metric. `group_17` and `group_18` required exact
  failed-row cleanup after the harness's 15-second trace poll produced false
  negatives; both passed after the poll window was widened to 60 seconds.
- Frozen-sequence real-LLM review: all 40 cases executed individually and have
  technical artifacts. Content is usable; recovered relevance/action warnings,
  visible-surface drift, generated narrative details, and latency outliers
  remain in the final technical/performance and user-quality gates.
- Call/context/latency ledger: the full machine report, native schema
  manifest, and 40-case review are recorded under
  `test_artifacts/cognition_core_v2/stage_3/`, including
  `fresh_40_turn_answerability_remediation_report.md` and
  `answerability_remediation_40_case_review.md`. All eight separate source/edge focused
  real-LLM commands completed one case at a time; the source comparison and
  latency disposition are recorded in `focused_source_comparison.md`.
  Technical completion carries the quality and latency residuals listed there.
- Broader non-live collection note: the manifest-defined affected suites are
  green. The final repository-wide non-live collection reported 3,229 passed,
  2 skipped, 21 failures, and 744 deselected. The 21 failures are outside the
  manifest's targeted affected-regression command set but within the broader
  change-radius inventory; the exact list is recorded in
  `test_artifacts/cognition_core_v2/stage_3/broad_non_live_regression_summary.md`.
  They remain open for separate contract reconciliation. This result keeps
  Checkpoint I open and is not converted into an unreviewed Stage 3
  production change.
- Control-console/browser review: API and external Playwright E2E are green;
  in-app Browser acceptance remains pending because no session is available.
- Native schema manifest and Stage 4 handoff: schema-manifest generation is
  verified; final handoff remains pending until H/I/J close.
- Independent code review: completed by `Hilbert`; remediation and affected
  reruns are recorded in the Stage 3 verification summary.
- Emotion/boundary/bounded-error sign-off overlay: the readable consolidation at
  `test_artifacts/cognition_core_v2/stage_3/emotion_boundary_bounded_error_signoff_review.md`
  retains the individually run sadness-to-crying, fear/shame/loneliness/anger,
  high-attachment abuse, verbal-abuse boundary, natural abuse-to-sadness,
  deterministic mechanical, and full-dialog real-LLM evidence. It records
  `sadness` visible-dialog proof, the natural `+5` semantic-sign failure, the
  `model_contract_invalid` bound failures, the fear repair rows, and the anger
  high-risk phrase as raw visible model output. No content keyword filter,
  censorship rule, or safety rewrite is part of the harness; invalid
  structural candidates are excluded only from affect conclusions.
- User quality sign-off for the Phase 3 artifact set: approved by the user on
  2026-07-22. External Browser acceptance remains pending because no Browser
  session is available; the plan lifecycle status remains `in_progress` until
  that gate and the remaining completion checks are closed.

## Latest Verification Addendum — 2026-07-21 Chinese-Only Semantic Contract

- The model-facing semantic contract is Chinese-only. The temporary
  dual-language branch was removed, and interaction-style, conversation-
  progress, cognition, goal, action, and surface projections now use Chinese
  role labels and Chinese explanatory text in Chinese contexts.
- Exact internal handles and fixed schema/enum tokens remain solely for the
  typed machine contract where required. They are not alternate natural-
  language labels in the semantic output.
- The final representative real-LLM abuse-to-sadness dialog passed at
  `test_artifacts/cognition_core_v2/crying_sadness_e2e/abuse_to_sadness_visible_dialog_82f0a133f1/`.
  The visible output is Chinese, the final semantic projection contains
  sadness at `极高` intensity, and the inspected model-facing fields contain
  no forbidden English role terms.
- Fifteen retained real-LLM/mechanical sign-off selectors passed individually:
  sadness-to-crying, fear/shame/loneliness/anger secondary-crying probes,
  high-attachment abuse, verbal-abuse boundaries, natural and seeded
  abuse-to-sadness, visible dialog, and mechanical reachability.
- Latest gates: `3,264 passed`, `2 skipped`, `758 deselected` for the non-live
  collection; `25 passed` for control-console gates; `1 passed` for fresh
  database console E2E; compile, diff, and targeted Chinese-only contract
  scans passed.
- The user approved the Phase 3 artifact set on 2026-07-22. Historical
  broader-suite failures recorded earlier in this document remain historical
  evidence and are superseded for the latest verification count by the run
  above; they are not silently deleted. External Browser acceptance and the
  remaining lifecycle completion checks are still pending.
