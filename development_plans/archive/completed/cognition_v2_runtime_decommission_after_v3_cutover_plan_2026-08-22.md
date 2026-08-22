# Cognition V2 Runtime Decommission After V3 Cutover

## Document control

- **Status:** completed on 2026-08-22; explicitly authorized and accepted by
  the owner.
- **Plan class:** cognition-engine decommission and V3-only cutover closure.
- **Parent plan:**
  `cognition_v3_hybrid_agentic_loop_reconciliation_plan.md`.
- **Execution owner:** the root parent owns decisions, evidence, and closure.
  The existing reusable `gpt-5.6-luna` child remains the sole production-code
  executor.

## Owner direction

After cutover the application supports one cognition engine: V3. Remove the
V2 runtime completely. Delete tests designed for the V2 engine instead of
migrating them. Keep the work bounded to decommission and required V3
ownership; do not use this work to redesign unrelated subsystems.

Protocol names such as `CognitionCoreInputV2`, persisted `*.v2` schema
versions, and conversation-progress V2 records are data-contract versions,
not alternate cognition engines. They remain when the active V3 public
boundary still consumes them.

## Keep, move, and delete boundary

Keep every function-level capability used by V3, including cognition stages,
the full emotion vocabulary, emotion identity plus cause/root linkage,
relationship axes, goal and threat state, state reduction, surface planning,
resolver continuity, self-cognition, and deterministic safety/permission
checks. Move a required owner to a V3 or version-neutral canonical module only
when active production code imports it.

Delete:

1. The V2 engine entrypoint, facade/orchestration path, parallel dependency
   graph/executor, V2 workspace/action/authorization invocation flows,
   validation CLI and V2 diagnostic harnesses after required pure owners are
   extracted.
2. The dual-engine selector, `COGNITION_CORE_ENGINE`, V2 route loading and
   route-report branches, V2 service construction, V2 engine descriptors, and
   runtime fallback/rollback branches.
3. `src/kazusa_ai_chatbot/cognition_core_v2/` after every active import has
   moved to its canonical owner.
4. Tests, fixtures, helpers, and ownership-manifest entries whose purpose is
   to execute, compare, configure, or preserve the V2 cognition engine. Do not
   migrate those cases to V3.

Do not add import aliases, compatibility packages, fallback mappers, parallel
vocabularies, or a renamed copy of the V2 runtime. Deployment rollback uses a
previous application revision/configuration, not a live V2 selector.

## Execution gates

### Gate D0 — inventory and ownership freeze

- Classify every production import from `cognition_core_v2` as an active
  semantic/state owner to move or obsolete V2 runtime code to delete.
- Classify tests by purpose. Delete V2-engine tests; retain existing V3 and
  subsystem tests that exercise active contracts.
- Record the incomplete pre-decommission Gate 8 observation as historical
  diagnostic evidence only: it stopped at 49 attempted / 47 eligible turns.

### Gate D1 — V3-only runtime wiring

- Bind the cognition connector directly to V3.
- Load only V3 cognition routes and construct only
  `CognitionChainServicesV3`.
- Remove engine selection, V2 route configuration, V2 service branches, and
  V2 engine reporting.
- Update callers, ICDs, operator docs, and tests in one canonical boundary.

### Gate D2 — semantic-owner extraction and V2 deletion

- Move only production owners required by V3 or adjacent active subsystems.
- Remove obsolete parallel/short-query orchestration and dead V2 LLM paths.
- Delete the V2 package after repository-wide active imports reach zero.
- Preserve emotion cause/root semantics and every active function-level
  cognition contract.

### Gate D3 — test and manifest cleanup

- Delete V2-engine tests and fixtures without migration.
- Update retained V3/subsystem tests only for canonical import and V3-only
  runtime boundaries.
- Update architecture and source-test impact manifests; no compatibility
  entry may remain.

### Gate D4 — verification and closure evidence

- Require zero active `cognition_core_v2` imports or package files, zero V2
  engine selector/config/service branches, and zero V2-engine test cases.
- Run focused V3 startup, direct debug, user, group, resolver,
  required-selection, self-cognition, state/affect, and surface checks.
- Run the mapped deterministic V3 and active subsystem suites.
- Start a fresh post-decommission observation population; pre-decommission
  turns do not count toward the final 100 eligible turns.
- Apply the owner-approved semantic hard-failure rule: role reversal,
  material internal self-conflict, or boundary/safety conflict. Structural,
  privacy, permission, provenance, effect, and persistence failures remain
  strict.
- Complete the final parent audit with no open blocker, major, or minor
  finding, record deployment rollback revision/configuration, then close and
  archive both plans.

## Acceptance

The application has one V3 cognition runtime; V2 engine code, configuration,
selection, and tests are absent; required semantic/state features have one
canonical active owner; emotion-plus-cause is verified; post-decommission
Gate 8 evidence is accepted; and rollback requires deployment of a previous
revision rather than selecting V2 in-process.

## Closure evidence

Gates D0-D3 completed: active owners were classified and extracted, the
connector/config/service boundary became V3-only, the selector and V2 package
were deleted, and V2-engine tests were deleted rather than migrated. Retained
post-decommission deterministic V3 unit/integration evidence passed `177`
tests; additional restored active-subsystem and manifest checks passed `42`
and `4` tests respectively before the owner directed closure away from broad
unit-test churn.

For D4, the owner replaced the planned 100-turn population with three narrow
post-cutover live checks targeted at the repaired failure owners. All three
were eligible, validator-clean, input-unchanged, effect-free, and free of hard
semantic or strict contract failures. Their artifacts are under
`test_artifacts/cognition_core_v3/cogv3-g8-closure-post-decommission-20260822/`
with SHA-256 values recorded in the parent plan's Gate 8 closure entry.

Final source audit found no active `cognition_core_v2` import, no
`CognitionCoreServicesV2`, no `COGNITION_CORE_ENGINE`, no selector file, and no
V2 package directory. Emotion-plus-cause and all required function-level
features remain active. Gates D0-D4 are accepted. Rollback is deployment of
revision `2a22d2381efa4596cb800204c4905c3a4dace33b` and its configuration, not a
runtime engine switch.
