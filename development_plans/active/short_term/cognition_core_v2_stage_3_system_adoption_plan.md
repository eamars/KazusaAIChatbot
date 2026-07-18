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
- Stage 3 may connect only to the dedicated disposable endpoint supplied as
  `STAGE3_TEST_MONGODB_URI`. It must never fall back to `MONGODB_URI`, inspect
  production rows, or write/drop a production database.
- The test database name is exactly `_test_kazusa_stage3_fresh`; the guarded
  endpoint fingerprint must differ from the configured production endpoint
  fingerprint before the service process starts.
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
- Failures use typed categories. Fatal technical failures receive bounded
  retry/fallback according to the existing safe checkpoint; content drift is
  recorded and the remaining planned cases continue.
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
| Database safety | dedicated URI and endpoint fingerprint with exact disposable database name | database-name checks alone do not isolate a production server |
| Stage 4 boundary | Stage 3 freezes native targets; Stage 4 discovers real production sources and performs migration | prevents unverified production assumptions |

## Data Migration

Stage 3 performs no data migration. Its database operation is a cold bootstrap
against an endpoint/database that passes the isolation guard. The test harness
asserts the database is absent before the first service start, records native
collections/indexes after exercised behavior, restarts against the same data,
and drops only that guarded database during cleanup.

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
- ordinary response-path p95 at or below 60 seconds;
- no episode exceeds the existing configured resolver maximum of three cycles;
- no `KazusaLiveBot is busy right now, please try again later.`-class collapse;
  safe-checkpoint retry/fallback must produce a typed terminal outcome;
- background/source episodes complete or reach a typed terminal outcome within
  300 seconds, without blocking chat intake;
- each projected LLM call stays within 50,000 characters and the 50k-token cap.

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
- [ ] B — profile bootstrap and native database base complete.
  - Files/steps: execution companion B1-B8.
  - Verify: profile/config/bootstrap focused tests and cold-start bootstrap.
  - Evidence: seed validation, idempotence, mismatch rejection, native indexes.
  - Handoff: reread plan; start C.
  - Sign-off: `<parent/date>`.
- [ ] C — canonical source contracts complete.
  - Files/steps: execution companion C1-C8.
  - Verify: episode/RAG/origin/source-policy tests and retired-label greps.
  - Evidence: five source packets and zero unapproved runtime source labels.
  - Handoff: reread plan; start D.
  - Sign-off: `<parent/date>`.
- [ ] D — capability availability/action path complete.
  - Files/steps: execution companion D1-D8.
  - Verify: registry/evaluator/handler/resolver focused tests.
  - Evidence: availability matrix, race recheck, typed unavailable outcomes.
  - Handoff: reread plan; start E.
  - Sign-off: `<parent/date>`.
- [ ] E — trace settlement and persistence complete.
  - Files/steps: execution companion E1-E9.
  - Verify: trace/consolidation/progress/delivery tests.
  - Evidence: one settled trace for every outcome row.
  - Handoff: reread plan; start F.
  - Sign-off: `<parent/date>`.
- [ ] F — all background/runtime producers adopted and scaffolds deleted.
  - Files/steps: execution companion F1-F9.
  - Verify: self-cognition/calendar/task/background/reflection tests plus
    zero-reference/delete greps.
  - Evidence: source-to-owner matrix exercised; deletion inventory clean.
  - Handoff: reread plan; start G.
  - Sign-off: `<parent/date>`.
- [ ] G — operations, console, scripts, and documentation aligned.
  - Files/steps: execution companion G1-G8.
  - Verify: ops/trace/script tests and control-console browser checks.
  - Evidence: screenshots/contract captures, redaction proof, doc-link check.
  - Handoff: reread plan; start H.
  - Sign-off: `<parent/date>`.
- [ ] H — fresh-database E2E and real-LLM proof complete.
  - Files/steps: execution companion H1-H11.
  - Verify: cold start, five sources, restart, final sequential quality set.
  - Evidence: `Stage3FreshDatabaseEvidenceV1`, monologue/dialog/action/trace
    review, call/latency ledger, native schema manifest.
  - Handoff: reread plan; start I.
  - Sign-off: `<parent/date>`.
- [ ] I — full deterministic regression and Stage 4 handoff complete.
  - Files/steps: execution companion I1-I6.
  - Verify: all exact I commands, `git diff --check`, link and placeholder
    scans.
  - Evidence: regression totals, allowed exceptions, Stage 4 input paths.
  - Handoff: reread plan; start J.
  - Sign-off: `<parent/date>`.
- [ ] J — independent code review and user sign-off complete.
  - Scope: plan/manifest alignment, full diff, architecture/style, DB safety,
    budgets, tests, evidence, deletions, and Stage 4 separation.
  - Verify: rerun every gate affected by review fixes.
  - Evidence: reviewer identity/findings, remediation, reruns, residual risks,
    user quality decision, lifecycle update.
  - Handoff: archive Stage 3 only after all findings and user gates close.
  - Sign-off: `<parent/reviewer/user/date>`.

Current execution evidence and external gate status are recorded in
`test_artifacts/cognition_core_v2/stage_3/checkpoint_i_verification_summary.md`.
Deterministic implementation gates are green. Fresh-database, real-LLM,
40-case quality, in-app Browser, and final user sign-off remain open.

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
- fresh-DB endpoint isolation, seed idempotence, runtime state ownership,
  restart continuity, and absence of production access;
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
11. Fresh-DB cold start, five-source exercise, persistence, restart, and final
    sequential real-LLM review pass the technical and budget gates.
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
- Database-name guards can still target a production server. Required isolated
  URI plus endpoint fingerprint separation prevents service startup.
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
- Fresh-database bootstrap contracts and safe report/native-manifest generation:
  verified; guarded live cold-start/restart remains pending.
- Deterministic tests and static gates: current evidence is recorded in
  `test_artifacts/cognition_core_v2/stage_3/checkpoint_i_verification_summary.md`.
- Live database and restart: pending because the dedicated Stage 3 endpoint
  and fingerprint guards are not configured.
- Sequential real-LLM review: eight required cases were invoked individually
  and guarded-skipped; real quality review remains pending.
- Call/context/latency ledger: harness implementation is verified; the
  40-case ledger remains pending until the guarded run executes.
- Control-console/browser review: API and external Playwright E2E are green;
  in-app Browser acceptance remains pending because no session is available.
- Native schema manifest and Stage 4 handoff: schema-manifest generation is
  verified; final handoff remains pending until H/I/J close.
- Independent code review: completed by `Hilbert`; remediation and affected
  reruns are recorded in the Stage 3 verification summary.
- User sign-off: pending.
