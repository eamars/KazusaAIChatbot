# Cognition Core V2 Stage 2 Integration Plan

## Summary

- Goal: build and fully test one `cognition_core_v2` release candidate that
  persists structured causes, derives emotion and personality constraints
  deterministically above the LLM, runs independent goal cognition in parallel,
  integrates the V2 surface contract, and removes V1 and affinity from the
  candidate runtime.
- Plan class: high_risk_migration.
- Status: in_progress.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `cjk-safety`,
  `test-style-and-execution`, and `debug-llm`.
- Overall cutover strategy: bigbang inside the source release candidate;
  production data migration and deployment are deferred.
- Highest-risk areas: Stage 1 architecture drift, causal-state correctness,
  deterministic/LLM ownership leakage, parallel critical-path latency,
  isolated Mongo safety, cross-module contract replacement, and legacy deletion.
- Acceptance criteria: Checkpoints A-I signed; all completion outcomes
  `S2-O1` through `S2-O10` demonstrated; exact value/cost/quality evidence
  delivered to the user.
- Decision authority: agents report exact evidence and limitations; the user
  decides whether to begin Stage 3 or later cutover planning.

This plan has two mandatory frozen companions:

- `cognition_core_v2_stage_2_contract_spec.md`: exact architecture, state
  models, formulas, state machines, public APIs, failure behavior, branch
  registry, semantic projection, and LLM budget.
- `cognition_core_v2_stage_2_execution_manifest.md`: exact Create/Modify/Delete/
  Keep surface, test ownership, Mongo guard, commands, calibration artifacts,
  checkpoints, and Stage 3 residual allowlist.

The parent and both subagents must read this plan and both companions in full.
The three documents form one executable contract. A conflicting instruction in
the older reference architecture has no Stage 2 execution authority.

## Context

Stage 2 exists for two product reasons:

1. Emotion must retain its structured cause. Every accepted event is compared
   with the goals, threats, causal events, relationships, standards, knowledge
   gaps, and meaning state that can produce emotion. An emotion word or prose
   reason is never the matching key.
2. Emotion and personality authority must remain above the model. Deterministic
   code owns validated state, event matching, goal status, emotion derivation,
   lifecycle, elapsed-time evolution, persistence, branch activation,
   permission, feasibility, and delivery. LLM calls interpret ambiguous
   meaning, reason within activated goals, reconcile competing bids, choose a
   semantic route, and support downstream wording.

Stage 1 validated useful execution mechanics but its state was shallow, its
appraisal was monolithic, its default goal graph was incomplete, and its
branch/action handoff was lossy. Stage 2 reuses reviewed mechanics only after
they pass the frozen RCA tripwires.

The current workspace remains a development release candidate. The deployed
service and production MongoDB are external and remain outside every Stage 2
command. Stage 2 uses synthetic native-V2 data in `_test_kazusa_live_llm`.

Stage 3 remains a separate auxiliary-adoption plan for the web console, export,
audit, snapshot, operator, and auxiliary test surfaces listed in the execution
manifest. A future separately approved cutover plan owns production migration,
deployment, restart, and production verification.

## Mandatory Skills

Load skills in this order:

1. `development-plan` before plan lifecycle, checkpoint, evidence, or handoff work.
2. `local-llm-architecture` before cognition graph, prompt, model-call,
   projection, latency, or L3 work.
3. `no-prepost-user-input` before user-input meaning, preference, permission,
   commitment, or direct-fact work.
4. `py-style` before editing Python.
5. `cjk-safety` before editing Python containing CJK text.
6. `test-style-and-execution` before adding, changing, or running tests.
7. `debug-llm` before live/model-facing tests or semantic review artifacts.
8. Any subsystem skill triggered by an exact execution-manifest path.

## Mandatory Rules

- After automatic context compaction and after each signed checkpoint, reread
  this plan and both mandatory companions before continuing.
- Before final completion, lifecycle change, merge, or sign-off, run the
  independent code-review gate and record its result.
- Use the parent-led native-subagent model in `Execution Model`.
- Treat every contract-spec type, scalar range, transition, formula, branch,
  signature, failure result, and call cap as immutable implementation input.
  A required change reopens Alignment Gate A and the plan.
- Run real-LLM cases one at a time and inspect one retained trace before the
  next case.
- Use pure deterministic tests for schemas, reducers, formulas, projections,
  permissions, state machines, branch scheduling, and persistence order; use
  patched LLM tests for handoffs; use real LLM only for actual model behavior.
- Project every value supplied to an LLM into approved semantic text. Raw
  numbers, timestamps, ids, enum codes, state documents, and unrestricted
  objects stay outside model messages.
- Accept LLM numeric output only as contract-listed integer deltas. Validate
  path ownership, evidence, type, range, per-event limit, duplicates, and
  transition guards before application.
- Keep free-form model descriptions free-form when deterministic code does not
  require a control value. Never hard-fail an explanatory label for differing
  from an expected semantic word.
- Keep emotion identity/lifecycle, persistent-goal state, branch activation,
  state mutation, permission, feasibility, persistence, and delivery outside
  LLM authority.
- Interpret user-language intent, preference, permission, commitment,
  responsibility, and social meaning through the LLM semantic lane. Apply no
  keyword classifier or post-LLM semantic override.
- Bind the exact `_test_kazusa_live_llm` database and fail-closed guard before
  any project import in a guarded DB test. Use only synthetic seed material.
- Mark every Stage 2 real-LLM test with both `live_llm` and `live_db`; load its
  V2 input state from the guarded synthetic seed/owner scope and record the
  selected database in evidence.
- Run affected database tests without xdist, with unique owner ids and exact
  owner/shared-seed queries. Restore the singleton character document in
  `finally` after any test write.
- Connect to no production database and run no migration, deployment, restart,
  production snapshot, or production verification command.
- Preserve existing user changes and remain inside the execution manifest.
- Read root `README.md` and `docs/HOWTO.md` as required current-system baseline;
  where they describe V1/affinity behavior, the three Stage 2 contract
  documents govern implementation and the Stage 3 plan owns auxiliary
  documentation adoption.

## Must Do

- Implement the contract-spec two-phase flow so immediate deterministic state,
  emotion, and ready branches proceed without unrelated semantic LLM work.
- Persist one validated cause-first state scope per episode and preserve a
  derived activation cache for restart-safe begin/sustain/fade continuity.
- Implement the exact relationship, goal, threat, event, knowledge-gap, drive,
  standard, meaning, and activation shapes and defaults.
- Implement the five-state goal FSM and all twenty-one frozen emotion formulas,
  lifecycle guards, adjacent distinctions, mixed-emotion behavior, elapsed
  decay, and character sleep recovery.
- Use the exact fourteen-branch goal registry and `MAX_GOAL_BRANCHES = 14`.
  Remove the V2 code semaphore; the configured LLM service owns concurrent-call
  capacity.
- Preserve complete admitted bids and whole-bid provenance through collapse,
  route selection, action materialization, V2 L3, and dialog.
- Integrate every exact runtime caller and one-scope persistence path before
  deleting V1.
- Remove V1, affinity, relationship-insight authority, and top-level prose
  affect authority from the Stage 2 runtime and core tests.
- Run all exact deterministic, integration, isolated-DB, real-LLM, benchmark,
  static, regression, and independent-review gates.
- Calibrate every checkpoint against the named Stage 2 outcomes and correct
  drift at the checkpoint where it first appears.

## Deferred

- Ordinary topic boredom as a persistent emotion; current low relevance remains
  absence of a grounded reason to speak.
- Dynamic coefficients, calibration tools, per-user schedulers, concurrent
  writer/version machinery, transaction ledgers, and new entity collections.
- Per-emotion stores, per-emotion LLM calls, emotion-label branches, retry or
  repair calls, fallback runtimes, compatibility shims, and alternate V1/V2
  vocabularies.
- Code-owned LLM concurrency throttling; the logical semantic-question and
  branch registries remain bounded by the contract spec.
- Stage 3 auxiliary adoption outside the exact residual allowlist.
- Production data conversion, migration, deployment, restart, and verification.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Core package/API | bigbang | Switch all candidate callers once at Checkpoint F; retain no V1 shim, flag, adapter, or fallback. |
| Candidate source | bigbang | Delete V1 and legacy runtime/tests after isolated live sign-off at Checkpoint H. |
| State vocabulary | bigbang | Use native embedded `cognition_state.v2`; preserve no affinity, incident alias, or prose-affect authority. |
| Test Mongo data | bigbang native V2 | Seed and validate synthetic V2 documents; perform no legacy migration. |
| Auxiliary Stage 3 paths | deferred | Permit only exact execution-manifest residual paths until Stage 3. |
| Production DB/deployment | deferred | A later separately approved cutover plan owns migration and deployment. |

Changing a policy requires a plan update and explicit user approval.

## Target State

Completion means one tested source release candidate has this observable state:

| ID | Observable outcome |
|---|---|
| `S2-O1` | Every runtime cognition source calls the two canonical V2 APIs; V1 runtime/contracts/tests are absent from the candidate. |
| `S2-O2` | User and singleton character documents contain validated native V2 state with one-scope causal ownership, exact defaults, caps, evidence, and restart-safe activation continuity; runtime affinity and prose affect are absent. |
| `S2-O3` | Every accepted event is matched by structured refs/axes/evidence and deterministically produces traceable concurrent begin/sustain/fade behavior for the twenty-one registry emotions. |
| `S2-O4` | Relationship, drive, standard, boundary, goal, elapsed-user decay, and scheduled character sleep recovery constrain behavior independently of model prose. |
| `S2-O5` | Model messages contain only scoped semantic projections; model outputs can propose only contract-listed propositions, free-form explanations/bids, handles, and bounded deltas. |
| `S2-O6` | Immediate branches start without unrelated LLM waits; all fourteen goal branches are available; unique result slots, dependencies, complete bids, and provenance-safe collapse prevent overwrite or invention. |
| `S2-O7` | Persona, resolver, self-cognition, task/scheduler results, persistence, action, L3, dialog, consolidation, reflection, and tracing use the frozen V2 ownership boundary. |
| `S2-O8` | Every database-processing test uses guarded `_test_kazusa_live_llm`, synthetic data, owner isolation, scoped singleton restore, and no production access. |
| `S2-O9` | Exact deterministic, integration, real-LLM, cross-model, latency, call-count, overlap, prompt, state-I/O, failure, and independent-review evidence is complete. |
| `S2-O10` | Stage 2 ends with a tested V2 release candidate and stable contracts for Stage 3; production migration and deployment remain future work. |

Passing dialog examples cannot substitute for any row.

## Design Decisions

| Topic | Frozen decision | Reason |
|---|---|---|
| Package | `src/kazusa_ai_chatbot/cognition_core_v2/` | Approved canonical package beside V1 during construction. |
| Runtime switch | One source caller switch at Checkpoint F | Avoid dual execution and duplicate cutover instructions. |
| V1 deletion | Checkpoint H after isolated live sign-off | Preserve a measurable baseline until V2 is integrated and tested. |
| Mutable scope | One user or character state per cognition result | Prevent dual-document partial commits without adding transaction machinery. |
| User default | Exact acquaintance state in contract spec | `null` and implicit legacy conversion are invalid. |
| Character default | Exact fixed production default in contract spec | One bootstrap, seed, and test authority; no calibration subsystem. |
| Emotion continuity | Persist a generic derived activation cache with causal roots | Preserve lifecycle across restart while causes remain authoritative. |
| Goal lifecycle | Five explicit states and guarded transitions | Meet the approved simple model without semantic-only status changes. |
| Event matching | Structured refs/axes/evidence plus fixed outcome guards | Description text and emotion labels cannot own identity. |
| Direct facts | Exact trusted producer allowlist | Preserve LLM ownership of user-language meaning. |
| Numeric input | Central deterministic semantic projection | The LLM never interprets raw numeric state. |
| Numeric output | Integer delta proposals on exact paths | Preserve the useful affinity-style output pattern without affinity authority. |
| Parallelism | Two-phase readiness plus isolated branch slots | Deterministic work proceeds and parallel calls cannot overwrite. |
| Branch count | Fourteen and `MAX_GOAL_BRANCHES = 14` | Full frozen registry remains available as requested. |
| Concurrency | No code semaphore | LLM service owns simultaneous-call capacity. |
| Persistence order | Valid cognition, one state commit, then action/L3/dialog | Preserve causal continuity before visible or executable output. |
| Sleep | Existing daily trigger plus deterministic recovery | Preserve schedule while removing affect-rewrite LLM calls. |
| Failure | Exact fail/continue table in contract spec | Prevent execution agents from inventing retries, fallback, or partial behavior. |
| Stage 3 | Exact residual path allowlist only | Keep auxiliary adoption separate and statically enforceable. |

## Contracts And Data Shapes

The mandatory contract spec freezes:

- the upstream/core/downstream diagram and preliminary/final dependency flow;
- scalar ranges, delta bounds, stable reducer ordering, and duplicate-target rejection;
- every common ref/evidence type and embedded user/character entity;
- exact acquaintance and character production defaults;
- one mutable persistence scope per episode;
- complete goal and emotion state machines;
- the twenty-one formula/guard/decay registry and adjacent distinctions;
- elapsed user evolution and idempotent deterministic character sleep recovery;
- every numeric/time/control semantic projection band;
- trusted direct-fact producers and the LLM-first user-input boundary;
- exact `run_cognition(...)` and `run_text_surface_planning(...)` signatures;
- complete bid, collapse, state-update, surface, failure, and commit contracts;
- the fourteen-branch registry and call/context ceilings.

Implementation may select private function decomposition only inside the exact
files and while preserving these public and behavioral contracts.

## Drift Closure Matrix

| Surfaced issue | Frozen correction | First proving checkpoint |
|---|---|---|
| Stage 1 binary/shallow roots | Full relationship, goal, threat, event, gap, drive, standard, meaning, role, and evidence shapes | B-C |
| Monolithic all-emotion appraisal | Six source-selected, path-owned, concurrent semantic question families | D |
| Incomplete default goal graph | Exact causal goal-creation guards and fourteen branch registry | C-E |
| Lossy bid/action handoff | Complete internal bids, handle-only collapse, whole-bid copies, constrained route output, and bounded primary/supporting projections through L3 | E-F |
| Fade projected as currently active | Cause status, phase, trend, score, and semantic lifecycle projection | C-D |
| Universal synthetic fade evidence | One natural begin/sustain/fade/negative real-LLM case per emotion | G |
| Self/other and user/character ambiguity | Role assignments plus exact caller-to-one-scope matrix | B-D |
| Non-goal statuses lacked guards | Threat, event, knowledge-gap, goal, and activation FSMs | C |
| Arbitrary caller-authored direct mutation | Exact producer/fact transition table with no path-bearing direct facts | C |
| New semantic causes could not reach onset | Transient semantic delta ceiling aligned to the fixed begin threshold; stable axes remain limited | C-D |
| Collapse/route could invent details | Model outputs prompt-local handles/canonical choices only; code copies complete admitted bids | E |
| L3 could run across an uncommitted state | Frozen `run_cognition -> commit -> action/resolver -> surface -> dialog` order | F |
| Real-LLM cases could bypass test Mongo | Every real-LLM node also carries `live_db`, seeded owner state, guard evidence, and singleton restore | G |
| Legacy reference contradicted the plan | Stage-2-aligned v1.0 reference; exact authority remains in this plan and companions | A |
| Broad test/delete ownership | Exact rewrite, delete, and Stage 3 path dispositions | A, H |
| Stage 2/Stage 3 overlap | Exact residual allowlist and six diagnostic exclusions | H-I |

## LLM Call And Context Budget

| Family | Per episode | Per-call input cap | Blocking behavior |
|---|---:|---:|---|
| Scoped appraisal | `0..6` | 8,000 chars | Only declared semantic dependents wait. |
| Goal cognition | `1..14` | 24,000 chars | Dependency-ready branches overlap. |
| Collapse | `0..1` | 24,000 chars | Required only for multiple admitted bids. |
| Route selection | `0..1` | 12,000 chars | Zero when no valid bid remains; otherwise waits for admitted intention. |
| V2 L3 | `0..4` | Existing bounded surface projections | Zero without a permitted text surface; otherwise runs after state persistence. |

Worst case is 22 calls before L3 and 26 including L3. Every call remains under
the 50k-token context cap. The parent reports exact observed counts, critical
path, overlap, waits, failures, and context use; agents apply no proceed threshold.

## Change Surface

The exact file/symbol ownership is in the execution manifest and is mandatory.

### Delete

- `src/kazusa_ai_chatbot/cognition_chain_core/` at Checkpoint H after
  Checkpoint G sign-off.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_store.py` after DB-backed state.
- The four exact obsolete V1/affinity tests named in the manifest.
- The exact legacy symbols and test paths classified by the execution
  manifest's static searches, rewrite inventory, deletion table, and Stage 3
  residual allowlist.

### Modify

- Existing V2 core implementation and validation CLI.
- Exact DB, bootstrap, persona, resolver, source-builder, RAG/context, action,
  service, consolidation, reflection, dialog, and event-log files in the manifest.
- Exact deterministic, integration, live, benchmark, and regression owners in
  the manifest.
- `pytest.ini` to stop collecting the fixture project as repository tests.

### Create

- Six exact V2 core modules for state, guards, source planning, projection, and L3.
- Exact Mongo helper/seed, alignment, projection, failure, integration, and
  isolation test files.

### Keep

- Adapter normalization/delivery, Mongo connection behavior outside the test
  guard, memory collections/evolution, existing scheduler timing/repository,
  LLM interface/provider code, and every production cutover operation.
- Exact Stage 3 auxiliary residual paths, unchanged until Stage 3.

Changes outside the manifest require a plan update before editing.

## Overdesign Guardrail

- Actual problem: emotion lacks persistent structured causes and deterministic
  model-independent lifecycle/personality authority, and Stage 1 showed broad
  wording allows internal architecture drift.
- Minimal change: one shared embedded causal model, one generic activation
  cache, one two-phase reducer/derivation path, one fixed goal branch registry,
  one state commit, and the existing downstream service boundaries.
- Ownership boundaries: LLM handles scoped semantic judgment and bid language;
  deterministic code handles validation, state, transitions, projections,
  dependencies, admission, persistence, scheduling, permissions, feasibility,
  and delivery.
- Rejected complexity: per-emotion schemas/calls/branches, new collections,
  dynamic calibration, compatibility, dual writes, locks/version ledgers,
  retries/repair calls, code throttles, multiple test databases, production
  cloning, routine reseed/reset, and Stage 2 console redesign.
- Evidence threshold: only a failed frozen requirement or approved near-term
  integration, followed by a plan update and user approval, can add a rejected
  dimension.

## Agent Autonomy Boundaries

- The parent owns contract interpretation, tests, commands, evidence,
  architect calibration/sign-off, lifecycle, and user reporting.
- One production-code subagent owns all manifest production paths through
  sequential bounded checkpoint packets, its implementation self-calibration,
  and in-scope review remediation. It edits no tests and makes no architecture,
  contract, scope, failure, or persistence decision.
- One independent review subagent reviews only after verification and
  implements no fixes.
- The implementation agent may choose local mechanics only when the contract
  and exact surface determine one observable outcome.
- Contract, formula, schema, branch, call-budget, persistence, cutover, or path
  changes stop execution and reopen the plan.
- Search for equivalent code before adding helpers. Reuse or move established
  behavior when it satisfies the contract.
- Scope excludes unrelated cleanup, formatting churn, dependency upgrades,
  generic prompt rewrites, and feature additions.

## Implementation Order

Execution follows the exact Checkpoints A-I in the execution manifest:

1. Checkpoint A locks the frozen contract, path/test inventory, synthetic V1
   baseline, known regression baseline, ledger, and Gate A review.
2. Checkpoint B creates isolated Mongo safety and exact V2 state/schema/defaults.
3. Checkpoint C implements deterministic reducers, goal FSM, all emotion
   lifecycles, elapsed user evolution, and sleep recovery.
4. Checkpoint D implements the two-phase source planner/appraisal/projection
   boundary and proves deterministic work starts before unrelated LLM completion.
5. Checkpoint E implements the fourteen-branch DAG, parallel executor, complete
   bids, collapse, action route, facade, and V2 L3.
6. Checkpoint F performs the single caller switch and integrates upstream,
   persistence, action, dialog, consolidation, reflection, and diagnostics.
7. Checkpoint G runs isolated one-at-a-time real-LLM, live-DB, cross-model,
   smoke, and performance evidence while V1 remains unreferenced but available
   for comparison.
8. Checkpoint H deletes V1/affinity/prose-affect authority and runs static,
   compile, collection, and full non-live regression gates.
9. Checkpoint I performs independent review, remediation, final calibration,
   value/cost/quality reporting, documentation, Stage 3 handoff, and closure.

Each production checkpoint uses the test-first sequence: parent writes/runs the
named failing contract; the same production subagent implements the bounded
packet; parent reruns the same tests; implementation agent completes its
self-calibration record; parent completes the architect calibration and signs
before the next packet.

## Execution Model

- Parent agent owns orchestration, test code, verification, static checks,
  evidence, architect calibration, plan lifecycle, and final sign-off.
- Parent establishes Checkpoint B's focused test contract after Checkpoint A.
- Production-code subagent: exactly one native subagent. Record its canonical
  id at Checkpoint A; deliver later checkpoint packets and Checkpoint I
  remediation through follow-up tasks; it owns production changes and
  implementation self-calibration, then closes after Checkpoint I sign-off.
- Parent may prepare broader tests/evidence while the production agent works,
  but dependent integration begins only after the preceding checkpoint signs.
- Independent code-review subagent: exactly one different native subagent after
  Checkpoint H verification; it reviews and reports only.
- Native subagent unavailability stops execution unless the user explicitly
  approves fallback execution.

### Execution-model deviation audit — 2026-07-15

The prior execution records identify Checkpoint A-F implementation as
`parent-owned single-agent execution` and do not record the canonical native
production-subagent id required by this execution model. The commit metadata
also contains no resumable native-agent identity. Checkpoints B and C remain
reopened and unsigned, and the production implementation agent is therefore
not resumable from the repository evidence currently available.

Production implementation and Checkpoint I remediation remain paused pending
explicit user direction. The user subsequently approved the takeover recorded
below, which resolves this execution pause under a documented plan-level
execution-model change.

### User-authorized production-owner takeover — 2026-07-15

The user confirmed that the original production implementation agent no longer
exists and explicitly authorized the current Codex parent agent to take over
its role. This is a plan-level execution-model deviation; the frozen Stage 2
contracts, change surface, test ownership, checkpoint order, and evidence
requirements remain unchanged.

The current Codex agent is the sole production implementation owner for the B/C
remediation packets, all later production packets, and any Checkpoint I
production remediation. It remains available through Checkpoint I and closes
only after Checkpoint I sign-off. The parent continues to own tests, commands,
evidence, architect calibration, lifecycle, and final sign-off. The takeover
identity is recorded in every implementation calibration as the execution
owner and deviation; B/C must reach `Status: aligned` before handoff to D.

## Progress Checklist

Every checkpoint's `Calibration` entry requires both
`calibration/<gate>_implementation.md` from the production implementation agent
and `calibration/<gate>_architect.md` from the parent/architect. Checkpoint I
also requires the independent review record.

- [x] Checkpoint A — contract and baseline lock signed.
  - Verify/evidence/handoff: execution-manifest Checkpoint A.
  - Calibration: the Gate A implementation/architect pair maps frozen contracts to all outcomes.
  - Sign-off: `Codex/2026-07-14`; evidence is under `test_artifacts/cognition_core_v2/stage_2/`; then reread all three plan documents.
- [x] Checkpoint B — isolated DB harness and exact V2 schema aligned after takeover remediation.
  - Verify/evidence/handoff: execution-manifest Checkpoint B.
  - Calibration: `S2-O2`, `S2-O4`, `S2-O8`.
  - Sign-off: `Codex takeover implementation / Codex parent architect / 2026-07-15`; exact B command 20 passed with 3 deselected and guarded live command 3 passed with 2 deselected; both calibration records are `Status: aligned`; evidence is under test_artifacts/cognition_core_v2/stage_2/persistence/ and calibration/; then reread all three plan documents.
- [x] Checkpoint C — deterministic state and twenty-one lifecycles aligned after takeover remediation.
  - Verify/evidence/handoff: execution-manifest Checkpoint C.
  - Calibration: `S2-O2`, `S2-O3`, `S2-O4`.
  - Sign-off: `Codex takeover implementation / Codex parent architect / 2026-07-15`; exact C command 64 passed; both calibration records are `Status: aligned`; evidence is under test_artifacts/cognition_core_v2/stage_2/lifecycle/ and calibration/; then reread all three plan documents.
- [x] Checkpoint D — two-phase appraisal/projection boundary aligned after drift remediation.
  - Verify/evidence/handoff: execution-manifest Checkpoint D.
  - Calibration: `S2-O3`, `S2-O5` and appraisal/projection RCA rows.
  - Sign-off: `Codex/2026-07-14` (parent-owned single-agent execution); exact D packet 28 passed; evidence is under test_artifacts/cognition_core_v2/stage_2/contracts/ and calibration/; then reread all three plan documents.
- [x] Checkpoint E — goal DAG, collapse, facade, and V2 L3 aligned after drift remediation.
  - Verify/evidence/handoff: execution-manifest Checkpoint E.
  - Calibration: `S2-O5`, `S2-O6`, `S2-O7` and branch/collapse RCA rows.
  - Sign-off: `Codex/2026-07-14` (parent-owned single-agent execution); exact E packet 25 passed; evidence is under test_artifacts/cognition_core_v2/stage_2/parallelism/ and calibration/; then reread all three plan documents.
- [x] Checkpoint F — one caller switch and runtime integration aligned after drift remediation.
  - Verify/evidence/handoff: execution-manifest Checkpoint F.
  - Calibration: `S2-O1`, `S2-O2`, `S2-O5`, `S2-O7`.
  - Sign-off: `Codex/2026-07-14` (parent-owned single-agent execution); exact F packet 200 passed and extended deterministic packet 228 passed; evidence is under test_artifacts/cognition_core_v2/stage_2/integration/ and calibration/; then reread all three plan documents.
- [ ] Checkpoint G — isolated live/model/DB/performance evidence signed.
  - Verify/evidence/handoff: execution-manifest Checkpoint G and fixed live manifest.
  - Calibration: `S2-O3` through `S2-O9`.
  - Sign-off: `<architect/date>`; then reread all three plan documents.
- [ ] Checkpoint H — legacy deletion and full candidate regression signed.
  - Verify/evidence/handoff: execution-manifest Checkpoint H.
  - Calibration: `S2-O1`, `S2-O2`, `S2-O8`, `S2-O9`, `S2-O10`.
  - Sign-off: `<architect/date>`; keep the production subagent idle for review remediation and reread all three documents.
- [ ] Checkpoint I — independent review, remediation, final calibration, and report signed.
  - Verify/evidence/handoff: execution-manifest Checkpoint I.
  - Calibration: all `S2-O1` through `S2-O10`.
  - Sign-off: `<reviewer/date>` and `<parent/date>`; close the production subagent.
- [ ] User decision about Stage 3 and future cutover planning recorded.
- [ ] Documentation, registry, Execution Evidence, and completed-plan archive updated.

## Verification

### Focused and integration commands

Run the exact commands at each execution-manifest checkpoint. Test filenames and
node ids are frozen and cannot be substituted without reopening the plan.

### Real LLM

Run every fixed node in the execution-manifest real-LLM section separately.
Inspect and sign the retained trace before the next node. A schema-valid trace
without a grounded semantic review does not pass.

### Database

- Guard accepts only `_test_kazusa_live_llm` and rejects every alternate name
  before client creation.
- Every DB test records the selected name and owner/shared-seed identifiers.
- Shared seed validation is idempotent and performs no overwrite/reset.
- Every singleton write has before/after/restore evidence.

### Static and full regression

Checkpoint H runs exactly:

```powershell
rg -n "from kazusa_ai_chatbot\.cognition_chain_core|import kazusa_ai_chatbot\.cognition_chain_core|CognitionChain(Input|Output|Services)V1|run_cognition_chain" src tests
rg -n "active_incidents|IncidentState" src tests
rg -n "affinity|Affinity|last_relationship_insight|global_vibe|reflection_summary" src tests
venv\Scripts\python -m compileall -q src\kazusa_ai_chatbot tests
venv\Scripts\python -m pytest --collect-only -q
venv\Scripts\python -m pytest -q
venv\Scripts\python -m pytest -q --ignore=tests\control_console_e2e\test_page_navigation_e2e.py --ignore=tests\test_control_console_repository.py --ignore=tests\test_control_console_review_edges.py --ignore=tests\test_script_db_boundary.py --ignore=tests\test_user_state_snapshot.py --ignore=tests\test_character_state_snapshot.py
git diff --check
```

The first two greps return zero matches with `rg` exit code `1`. The third
returns only exact Stage 3 allowlist paths. Compile and collection pass. The
diagnostic full run may fail only at the six exact Stage 3 test paths excluded
by the following command; the Stage 2-owned non-live command must pass.
`pytest.ini` excludes live markers and the fixture project.

### Production boundary

Evidence proves zero production connection attempts and absence of migration,
deployment, or restart commands. Any statement about external production state
comes only from later operator/user confirmation.

## Independent Plan Review

Before approval, review this plan, both companions, the Stage 1 plan/RCA, the
reference architecture, Stage 3 inventory, current source/tests, and registry
from a fresh-review posture. Approve only when:

- every surfaced architecture, data, lifecycle, API, failure, branch, call,
  DB, cutover, path, test, gate, ownership, and terminology issue maps to one
  frozen contract or executable checkpoint;
- no placeholder token, optional file, mutable test name, broad wildcard ownership, hidden
  helper freedom, alternate call path, compatibility, fallback, or unresolved
  decision remains;
- every outcome maps to implementation, verification, calibration, evidence,
  handoff, and sign-off;
- exact Stage 2 and Stage 3 ownership is non-overlapping.

Record pre-approval findings and fixes in the plan review response. Plan review
does not authorize production implementation.

## Independent Code Review

Run Checkpoint I after Checkpoint H passes and before completion, merge,
lifecycle sign-off, or archive. The independent reviewer receives the approved
three-document contract, Stage 1 evidence, full diff, every command/result,
calibration packet, live review, DB evidence, and Stage 3 residual list.

Review scope:

- project/style/skill compliance and exact change-surface adherence;
- two-phase deterministic readiness and model authority boundaries;
- state ownership, formulas, state machines, projections, failure/commit order,
  branch isolation, collapse provenance, and no compatibility/fallback path;
- DB guard, seed/owner/singleton isolation, zero production access, and legacy deletion;
- test taxonomy, real-LLM human review, performance accuracy, regression, and handoff.

The reviewer implements no fixes. The parent routes in-scope production fixes
to the same implementation subagent, owns test-only fixes, reruns the affected
checkpoint, and sends remediation evidence to the same reviewer for closure. A
contract/scope finding reopens the plan.

## Acceptance Criteria

Stage 2 is technically complete only when:

- all Checkpoints A-I and calibration records are signed with no unexplained drift;
- every `S2-O1` through `S2-O10` row has linked reproducible evidence;
- the candidate has one V2 runtime/API/vocabulary and no V1 runtime or core test;
- native V2 state, defaults, one-scope persistence, causal roots, goal FSM,
  activation continuity, and all twenty-one lifecycles pass deterministic tests;
- no model-facing payload contains raw internal numeric/operational state;
- model numeric output changes only allowlisted paths through guarded deltas;
- all fourteen branches are available, deterministic work does not await
  unrelated semantic work, parallel results cannot overwrite, and collapse
  cannot invent content;
- every affected DB test uses guarded synthetic `_test_kazusa_live_llm` data;
- focused, integration, full non-live, one-at-a-time real-LLM, cross-model,
  performance, static, and independent-review evidence is complete;
- only exact Stage 3 auxiliary residual paths retain legacy display/export text;
- production connection, migration, deployment, restart, and production
  verification are absent;
- the user receives exact value, cost, quality, latency, call, failure,
  limitation, and model-variance evidence without an agent-selected threshold.

The user's decision about Stage 3 or future cutover does not retroactively alter
whether Stage 2 met these technical criteria.

## Risks

| Risk | Control | Verification |
|---|---|---|
| Stage 1 architecture drift repeats | Frozen spec plus test-first Checkpoints B-E | RCA tripwires and per-gate calibration |
| LLM blocks deterministic work | Mandatory preliminary/final flow | readiness timing tests |
| Lifecycle cannot survive restart | Derived activation cache with causal validation | reload lifecycle tests |
| Semantic labels become authority | Structured refs/axes/evidence and free-form explanation boundary | negative matching tests |
| Parallel calls overwrite | Unique target owners and `branch_results[branch_id]` | completion-order tests |
| Collapse invents details | Handle-only collapse output and whole-bid deterministic copies | invalid-collapse tests |
| Weak model sees raw state | Central projection bands and sentinel inspection | prompt projection tests |
| Test reaches production DB | Pre-import exact-name guard | isolation and selected-name evidence |
| Persistent test rows interfere | Unique owners and exact shared-seed ids | repeated-run isolation tests |
| Candidate mistaken for deployment | Explicit source/external-system boundary | command/diff audit |
| Legacy coverage is deleted blindly | Exact rewrite/delete/Stage 3 disposition | replacement test and grep matrix |

## Approval Boundary

This document remains a planning artifact while `Status` is `draft`. Production
implementation begins only after the user explicitly approves this plan and
commands execution.

## Execution Evidence

During approved execution, record:

- approval/status, pre-existing changes, V1 synthetic baseline, and collection baseline;
- Checkpoint A is signed under the user-authorized parent-owned fallback; its
  contract hashes, baseline, inventory, ledger, and calibration pair are under
  `test_artifacts/cognition_core_v2/stage_2/`;
- frozen contract/spec/manifest hashes, the unavailable native-agent deviation,
  and the user-authorized takeover owner identity;
- Checkpoint A-I packets, commands, signers, calibration, deviations, and remediation;
- selected test DB, guard, seed, owner, singleton, and zero-production-access evidence;
- exact changed/deleted files and commits;
- deterministic, patched, integration, full-regression, real-LLM, cross-model,
  smoke, and benchmark results;
- prompt/projection samples, state before/after, causal roots, lifecycle, branch,
  collapse, route, state-commit, and failure evidence;
- independent review findings/fixes/reruns;
- value/cost/quality report, limitations, user decision, Stage 3 residual
  handoff, documentation, registry, and archive completion.
