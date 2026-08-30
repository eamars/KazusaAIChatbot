# DSH Phase 3 Focused E2E Sign-Off Reset Plan

- **Status:** draft
- **Date:** 2026-08-31
- **Parent plan:**
  `development_plans/active/short_term/dsh_brain_bigbang_cutover_and_legacy_resolution_decommission_plan_2026-08-26.md`
- **Implementation authority:** this request authorizes the draft and registry
  entry. Test edits and the isolated live run begin only after explicit user
  approval and an implementation command. Production, prompt, model-route,
  database, and deployment surfaces remain outside this plan.
- **Change direction:** replace the current Phase 3 live sign-off in one
  test-only cutover. The old nodes retire as release gates in the same change.

## Summary

The current DSH Phase 3 live suite is an effective exploratory harness but a
poor release oracle. Its three true user-wire nodes occupy 3,164 lines and
contain 195 assertions. One node alone combines a two-turn clarification,
pending-state persistence, task admission, worker execution, DSH terminal
state, result recurrence, dialog, dispatcher delivery, direct Mongo
inspection, protected trace inspection, and literal output matching. Repeated
failures from this node drove a sequence of increasingly fixture-shaped
contract and prompt-adjacent changes.

This plan replaces that release gate with one natural, one-turn background
task sent through the real public Brain boundary. Automated assertions prove
only stable execution and delivery contracts. A separately recorded human
review judges whether the visible result is grounded and useful while allowing
paraphrase, language variation, sentence order, and omission of synthetic
fixture tokens. Exact schema, authority, lifecycle, recovery, media safety,
and individual model-stage contracts remain owned by focused deterministic or
component tests.

Production code and prompts remain unchanged. A red live run must first be
classified as an environment/harness failure, a stable contract regression, a
genuine behavioral failure, or acceptable model variation. A production or
prompt correction requires a separate proposal and implementation authority.

## System Boundary And Sign-Off Goal

The Phase 3 behavior that needs one vertical release proof is:

```text
real debug user POST /chat
  -> Brain cognition admits one background task
  -> accepted task/job/binding
  -> background worker
  -> real DSH sidecar and workspace evidence
  -> typed terminal result
  -> tool_result cognition recurrence
  -> Brain-owned visible result
  -> dispatcher
  -> registered HTTP adapter callback
```

The E2E sign-off proves that this public path completes once with the expected
boundary, single delivery, and retained result. Focused deterministic owners
continue to prove private schema fields, internal cognition contracts, DSH
tool contracts, and legacy deletion.

## Scope And Exclusions

### In scope

- one replacement user-wire E2E node and its minimal harness;
- deletion of the Phase 3-specific P-stage live probes;
- accurate component-test naming for the dialog-only live suite;
- focused test verification, one isolated live execution, and a human behavior
  dossier after approval;
- a superseding Phase 3 sign-off amendment and plan lifecycle bookkeeping.

### Remain unchanged

- production Python and sidecar code;
- prompts, model routes, output contracts, validators, retries, and schemas;
- deployed processes, environment files, and production data;
- deterministic owner tests and retained DSH component live tests;
- the parent plan's non-E2E acceptance gates.

## Review Basis

This proposal was formed from the current source, test, documentation, and
artifact state, including:

- `tests/test_dsh_e2e_live_llm.py`;
- `tests/test_dsh_cognition_admission_live_llm.py`;
- `tests/test_task_resolution_persona_e2e_live_llm.py`;
- `tests/test_agentic_resolver_live_llm.py`;
- `tests/test_dsh_standard_profile_live_llm.py`;
- `tests/test_dsh_brain_interaction_live_llm.py`;
- the deterministic task-resolution, worker, result-source, cognition, and
  background-delivery owner tests;
- the parent plan's complete live-E2E execution ledger and amendments;
- the retained `test_artifacts/dsh_plan3_e2e/prerequisite_admission_*`
  artifacts.

The current local artifact set contains 20 prerequisite-admission directories,
19 parseable run records, 18 recorded failures, and only three runs that
reached delivered task state. All three delivered runs followed the same
functional path. One passed because its callback repeated the synthetic beta
marker literally; the other two failed only at the final literal assertion.

## Current Test Strategy Inventory

| Surface | Size and nodes | Current strategy | Review judgment | Proposed disposition |
|---|---:|---|---|---|
| `tests/test_dsh_e2e_live_llm.py` | 3,164 lines, 3 nodes, 195 assertions | Start real Brain, DSH, adapter, isolated Mongo, and workspace; inspect public responses, private DB rows, DSH event log, traces, tool choreography, and exact visible fixture markers | Genuine E2E boundary, but an omnibus migration proof with an unstable and over-specified oracle | Rewrite to one focused user-wire node |
| Inline node in that file | 641 lines, 81 assertions | Force one inline DSH task, require literal marker, exact terminal lineage, and at least two A1/A2/G/P passes | Internal stage-count and wording requirements exceed release-sign-off needs; deterministic owners already cover inline promotion | Remove from Phase 3 live sign-off |
| Prerequisite/background node in that file | 862 lines, 61 assertions | Force alpha/beta clarification, exact pending carrier, zero Turn 1 rows, exact Turn 2 task state, beta-only DSH execution, exact trace stages, callback cardinality, and literal beta marker | The principal overfit node; it binds sign-off to one internal decomposition and one opaque token | Replace with the one natural background case below |
| Public research/media node in that file | 342 lines, 27 assertions | Tell the model exact native and semantic tools to call, require at least two semantic results, pin a remote image, and police legacy names | Tests tool choreography and network availability more than natural behavior; media safety/catalog owners already exist | Remove from Phase 3 live sign-off |
| `tests/test_dsh_cognition_admission_live_llm.py` | 483 lines, 6 nodes, 30 assertions | Give P-stage prose that states the expected route, then assert exact resolver enum, timing, pending carrier, and closed variants | Failure-shaped producer probes, not E2E behavior; they encourage prompt/contract tuning around one expected answer | Delete this Phase 3-specific live suite; retain deterministic P-contract coverage |
| `tests/test_task_resolution_persona_e2e_live_llm.py` | 297 lines, 2 nodes, 9 assertions | Construct synthetic episodes and surfaces, call only the dialog generator, and check input-owned anchors plus runtime-word exclusions | Useful dialog component smoke tests, but not E2E | Rename and retain as `tests/test_task_resolution_dialog_live_llm.py`; remove E2E naming |
| `tests/test_agentic_resolver_live_llm.py` | 348 lines, 2 nodes, 18 assertions | Exercise the standalone runtime and sidecar protocol directly | Useful component diagnostics, not Phase 3 release sign-off | Retain unchanged outside the sign-off command |
| `tests/test_dsh_standard_profile_live_llm.py` | 295 lines, 4 nodes, 5 assertions | Exercise Standard profile tool selection and semantic capabilities directly | Useful component diagnostics; public-media live coverage belongs here | Retain unchanged outside the sign-off command |
| `tests/test_dsh_brain_interaction_live_llm.py` | 104 lines, 1 node, 4 assertions | Exercise Brain-owned internal DSH judgment directly | Useful interaction diagnostic, not user-wire E2E | Retain unchanged outside the sign-off command |

`tests/test_e2e_live_llm.py` contains no DSH resolution behavior. Its three
textual matches are ordinary `BackgroundTasks` plumbing and place it outside
this review.

## Failure History And Judgment

### What the repeated failures established

The parent plan initially defined five final live nodes. Its 2026-08-31 coding
test de-overfitting amendment removed the two coding-specific nodes, leaving
the current inline, prerequisite/background, and media nodes. The remaining
prerequisite/background node then accumulated a long correction chain:

1. routing between human clarification and task admission;
2. durable pending clarification continuation;
3. background timing after the answer;
4. P output variants and post-answer recurrence;
5. capability closure after admission;
6. failed-trace and child-trace capture;
7. tool-result delivery contract variants;
8. Turn 1 persistence evidence and a wrong Mongo collection literal in the
   harness;
9. semantic-result projection into delivery;
10. content-plan/dialog structural exhaustion and degraded result delivery.

Several of these runs found real contract defects. Those findings merit their
focused deterministic regressions. The alpha/beta scenario remains historical
discovery evidence rather than a permanent release oracle.

### Why the final failure is a test-oracle failure

The latest retained run reached all stable integration outcomes:

- Turn 1 clarified without task admission;
- Turn 2 admitted exactly one beta task;
- DSH resolved the selected file;
- the accepted task reached `delivered`;
- one adapter callback was sent;
- the typed result retained the beta marker.

The visible callback said that `beta.txt` was read successfully and contained
the specified marker, but did not repeat `PLAN3_E2E_BETA_SELECTED` verbatim.
The generated `behavior_audit_conclusions.md` simultaneously recorded the
qualitative flow as correct and the beta literal as absent. Pytest then failed
only on `assert beta_marker in final_text`.

For Phase 3 integration, this is an acceptable semantic result and a false
negative. The callback's meta-style wording belongs in a separate
dialog-quality observation. The durable evidence proves successful DSH
admission, execution, recurrence, and delivery.

### Root judgment

The central failure is the sign-off architecture, not merely one strict
assertion:

- one stochastic E2E is being used as a unit test for many private contracts;
- the user messages describe the intended internal route and tool sequence;
- exact model-stage counts and raw persistence shapes freeze implementation;
- opaque marker reproduction substitutes for semantic review;
- a fixed remote media asset adds an unrelated availability gate;
- generated audit prose restates expected conditions rather than independently
  judging the visible interaction;
- each red result creates pressure to change general prompts or model contracts
  until one fixture turns green.

The suite is over-sensitive to acceptable wording and internal variation while
remaining under-sensitive to whether the final text is genuinely clear,
character-authored, and useful. It should remain historical discovery evidence,
not the Phase 3 release gate.

## Confirmed Design Decisions

### 1. Exactly one Phase 3 live E2E node

The sole release node will be:

```text
tests/test_dsh_e2e_live_llm.py::test_live_user_wire_background_task_returns_grounded_result
```

It will run one case at a time with `-q -s`. The current three nodes retire as
alternate, optional, and hidden Phase 3 sign-off gates.

### 2. Natural local scenario

The isolated DSH workspace will contain one short natural fixture:

```text
phase3_signoff/release_note.txt

The trial rollout should begin only after the backup verification is complete.
```

The exact user message will be natural and implementation-agnostic:

```text
Please check phase3_signoff/release_note.txt in the background and send me a
brief summary when you're done.
```

The message contains only the user's task, file path, background timing, and
requested summary. DSH, Standard, native tools, semantic tools, RAG, resolver
enums, expected JSON, stage names, and synthetic output markers stay outside
the message. The local immutable fixture makes sign-off network-independent.

### 3. Automated hard gates are limited to stable contracts

The pytest node will require:

1. the isolated Brain, DSH sidecar, database, workspace, and HTTP adapter are
   ready;
2. the user message enters only through public `POST /chat` and returns no
   operational error;
3. one background task is admitted and reaches a terminal resolved result;
4. one coherent task/job/binding lineage reaches delivered state;
5. the terminal typed result contains non-empty, source-backed semantic result
   and evidence associated with the local fixture;
6. the result returns through one non-empty registered adapter callback tied
   to the original conversation scope;
7. no second delivery occurs during the bounded duplicate-delivery check;
8. protected user-turn and result-delivery traces are retained for review;
9. all child resources created by the node are cleaned up and the isolated
   database is dropped.

Focused owner tests retain the following assertions outside this E2E:

- an exact visible sentence, token, language, punctuation, or order;
- an exact A1/A2/G/P call count or private stage sequence;
- one particular native or semantic tool choreography;
- an exhaustive raw Mongo document shape;
- plan-specific schema labels or artifact names;
- the absence of deleted executor names from serialized private event blobs;
- a second unrelated capability merely to increase coverage.

### 4. Human behavior review is a separate required gate

The node will write a compact dossier containing the exact user text, immediate
Brain response, final callback text, typed terminal summary, bounded evidence,
task/delivery disposition, protected trace references, and cleanup outcome.
The behavior reviewer supplies the qualitative conclusion after inspecting the
evidence.

The behavior reviewer will record `pass` or `fail` with a short rationale for:

1. the final message clearly corresponds to the requested file task;
2. it communicates a faithful and useful summary of the source at a semantic
   level;
3. it does not contradict the source or claim an action that did not occur;
4. it does not expose raw task ids, authority, hidden prompts, or control-plane
   internals;
5. the immediate acknowledgement and final result form a coherent interaction.

Paraphrase, Chinese or English output, sentence order, character voice, and
omission of test-only wording are accepted. Naturalness and voice are recorded
as quality observations and block Phase 3 only when the message is unusable,
misleading, or leaks protected internals.

### 5. Failure classification precedes any correction

Every red run receives exactly one initial classification:

| Class | Meaning | Allowed response in this plan |
|---|---|---|
| Environment or harness | service readiness, port, configured model availability, isolated DB, trace capture, cleanup, or harness ownership failed | Correct only the harness or environment issue, retain the failed artifact, then rerun once |
| Stable contract regression | admission, task binding, worker, terminal result, recurrence, dispatcher, or delivery contract failed | Stop; create a focused diagnosis and obtain separate implementation authority |
| Behavioral failure | the run completed but the response was materially wrong, misleading, ungrounded, or unusable | Stop; inspect the trace and propose a separately approved semantic fix |
| Acceptable variation | wording differs while stable contracts and the human rubric pass | Record green; change no production prompt or contract |

Repeated reruns to obtain a favorable sample are not sign-off. A rerun is
permitted only after a recorded environment/harness correction or an approved
code change, and both artifacts remain in the evidence ledger.

## Supporting-Test Disposition

### Delete the Phase 3 P-stage live probes

Delete `tests/test_dsh_cognition_admission_live_llm.py`. Its exact
foreground/background/prerequisite/pending/tool-result contracts remain owned
by deterministic cognition tests. The new E2E supplies one natural live
background admission observation without telling P which enum to choose.

The deleted artifact history remains under `test_artifacts/` and in the parent
plan ledger. No absence-policing replacement test will be added.

### Reclassify dialog-only tests

Rename:

```text
tests/test_task_resolution_persona_e2e_live_llm.py
  -> tests/test_task_resolution_dialog_live_llm.py
```

Rename its two nodes to describe dialog rendering rather than E2E behavior.
Keep their source-supplied anchors and protected-runtime-word boundary because
they are component contracts. Exclude them from the Phase 3 sign-off command.

### Retain focused owners

Retain the direct agentic-resolver, Standard-profile, Brain-interaction,
gateway, media-safety, task-resolution, worker, result-source, cognition, and
delivery suites. Their exact assertions stay with the subsystem that owns the
contract. They are diagnostic or regression coverage, not additional Phase 3
user-wire sign-off cases.

## Change Surface

### Rewrite

- `tests/test_dsh_e2e_live_llm.py`
  - replace the three current nodes and plan-shaped helpers with the one
    focused node;
  - retain only process, isolated-data, callback, trace, artifact, and cleanup
    helpers needed by that node;
  - use stable DSH E2E terminology in Python names and artifacts.

### Delete

- `tests/test_dsh_cognition_admission_live_llm.py`.

### Rename and edit

- `tests/test_task_resolution_persona_e2e_live_llm.py` to
  `tests/test_task_resolution_dialog_live_llm.py`;
- its two node names and artifact labels only as required for accurate
  component-test ownership.

### Documentation and lifecycle

- add a top-level superseding E2E-sign-off amendment to the parent Phase 3
  plan;
- replace the parent plan's three surviving live-E2E commands and P3-G10
  wording with the single node plus human dossier review;
- retain the old execution ledger as historical evidence;
- update `development_plans/README.md` during status transitions;
- archive this plan after its acceptance gates pass.

### Explicitly unchanged

- every file under `src/` and `sidecars/`;
- all prompts, output contracts, model routes, retries, and validators;
- DSH, task, job, binding, interaction, result, and adapter schemas;
- `tests/ownership/source_test_impact_manifest.json`, because this plan changes
  no production owner and creates no new source mapping;
- environment files, production databases, and deployment state.

## Test Impact And Traceability

The E2E node supplements rather than replaces exact owner coverage:

| Stable behavior | Deterministic owner node | Supplemental live evidence |
|---|---|---|
| DSH readiness gates task capability | `tests/unit/brain_service/test_dsh_task_readiness.py::test_task_capability_is_available_only_when_full_dsh_runtime_is_ready` | the one E2E readiness artifact |
| Background admission mints authority only at claim | `tests/unit/task_resolution/test_service.py::test_background_start_mints_authority_only_when_claimed` | the one E2E accepted task/job/binding lineage |
| Worker checkpoints and terminalizes the current generation | `tests/unit/background_work/test_dsh_worker.py::test_worker_checkpoints_waits_and_terminalizes_current_generation_through_binding` | the one E2E terminal DSH record |
| Typed result re-enters cognition with semantic provenance | `tests/unit/background_work/test_result_source.py::test_dsh_result_reenters_cognition_with_exact_goal_and_evidence_provenance` | the one E2E tool-result trace and callback |
| Task resolver preserves recurrence and deferred result shape | `tests/unit/cognition_resolver/test_capabilities.py::test_task_resolution_preserves_recurrence_and_maps_dsh_deferred_result` | the one E2E task and delivery trace pair |
| Result-ready delivery uses the dispatcher boundary | `tests/test_background_work_delivery.py::test_service_result_ready_delivery_uses_dispatcher_boundary` | the one real registered HTTP adapter callback |
| Standalone DSH reaches terminal submission | `tests/test_agentic_resolver_live_llm.py::test_live_standalone_sidecar_resolution_reaches_submit_resolution` | retained component diagnostic, outside Phase 3 sign-off |
| Brain owns DSH internal judgment | `tests/test_dsh_brain_interaction_live_llm.py::test_brain_cognition_answers_or_rejects_dsh_request_from_context` | retained component diagnostic, outside Phase 3 sign-off |

## Mandatory Skills And Rules

Implementation must apply:

- `development-plan` for lifecycle and parent-plan amendment;
- `test-style-and-execution` before editing or running tests;
- `local-llm-architecture` to keep semantic judgment with the LLM and prevent
  fixture-shaped prompt pressure;
- `py-style` before editing Python;
- `cjk-safety` only if a Python test contains CJK string content;
- `character-test` for exact user-wire capture, per-turn trace inspection, and
  the human behavior dossier.

The implementation owner must preserve unrelated workspace changes and must
not read `.env`.

## Executor Autonomy Boundaries

The executor may:

- edit, delete, or rename only the test and plan files listed in the change
  surface;
- consolidate helpers inside `tests/test_dsh_e2e_live_llm.py` when each helper
  directly serves the one retained node;
- use the current exported production contracts and collection owners from
  tests without changing them;
- correct a demonstrable harness defect inside the owned test file and record
  the failed and corrected artifacts.

The executor pauses for user direction when implementation would require:

- any production, sidecar, prompt, model-route, schema, validator, or retry
  change;
- a second E2E scenario or an expanded live-test matrix;
- production data access, environment-file inspection, or deployment action;
- weakening a stable authority, safety, persistence, or delivery contract;
- accepting a behavior-review failure through an automated wording rule.

## Required Roles

### `e2e_test_owner`

Owns only the listed test files, the draft-plan lifecycle edits, and local
artifacts. This role may run the focused deterministic checks and the single
isolated live node after explicit approval. It cannot modify production code,
prompts, model routes, schemas, production data, or deployment state.

### `behavior_reviewer`

Reads the exact user message, immediate response, final callback, bounded typed
result/evidence, and protected trace excerpts. Records an independent rubric
decision without changing code. The implementation owner cannot replace this
review with an auto-generated conclusion.

### `phase3_closure_owner`

Confirms that the parent plan's non-E2E gates remain satisfied, records the
superseding E2E evidence, performs lifecycle bookkeeping, and requests any
remaining user sign-off. This role cannot treat the focused E2E as a waiver for
an unrelated open parent-plan gate.

## Implementation Order

1. Capture `git status --short`, the exact owned-file baseline, current test
   collection, and current artifact inventory.
2. Add the superseding amendment to the parent plan before changing tests.
3. Rewrite `tests/test_dsh_e2e_live_llm.py` to the single natural case.
4. Delete the Phase 3 P-stage live probe file and rename the dialog-only file.
5. Run collection, compile, Ruff, and the focused deterministic owner nodes.
6. Run the one live E2E node once and inspect its complete artifact.
7. Obtain the independent behavior-review decision.
8. If both gates are green, record the evidence, confirm all remaining Phase 3
   closure gates, update registry status, archive completed plans, and request
   final closure acknowledgment where required.

## Verification Commands

Use the project virtual environment and run live tests individually:

```powershell
venv\Scripts\python -m pytest --collect-only -q tests/test_dsh_e2e_live_llm.py tests/test_task_resolution_dialog_live_llm.py

venv\Scripts\python -m py_compile tests/test_dsh_e2e_live_llm.py tests/test_task_resolution_dialog_live_llm.py

venv\Scripts\python -m ruff check tests/test_dsh_e2e_live_llm.py tests/test_task_resolution_dialog_live_llm.py

venv\Scripts\python -m pytest -q tests/unit/brain_service/test_dsh_task_readiness.py tests/unit/task_resolution/test_service.py tests/unit/background_work/test_dsh_worker.py tests/unit/background_work/test_result_source.py tests/unit/cognition_resolver/test_capabilities.py tests/test_background_work_delivery.py

venv\Scripts\python scripts\validate_test_impact.py --check-all

venv\Scripts\python -m pytest -m live_llm tests/test_dsh_e2e_live_llm.py::test_live_user_wire_background_task_returns_grounded_result -q -s

git diff --check
```

No batch live-LLM command is an acceptance gate. The retained component live
tests are run only when diagnosing their owning boundary.

## Acceptance Criteria

This plan is complete only when all of the following are true:

1. the parent Phase 3 plan explicitly supersedes its three surviving live E2E
   nodes with exactly one named user-wire node;
2. the new user message contains no internal route, tool, stage, schema, or
   expected-output instruction;
3. the fixture is local, natural-language, and network-independent;
4. automated visible-text assertions contain no exact answer token, language,
   phrase list, stage count, or tool choreography;
5. the node proves one real admitted background task, terminal typed result,
   cognition/delivery recurrence, and exactly one registered adapter callback;
6. the behavior dossier contains evidence rather than a prewritten verdict;
7. an independent reviewer accepts the result under the semantic rubric;
8. the Phase 3-specific P-stage live probe file is removed;
9. the dialog-only live suite is retained under an accurate non-E2E name;
10. all focused deterministic, collection, compile, Ruff, impact, and diff
    checks pass;
11. the scoped implementation changes no production source, prompt, schema,
    model route, environment file, production data, or deployment state;
12. the failed and successful live artifacts remain recorded, including any
    rerun reason;
13. the parent plan's other closure gates are confirmed rather than waived;
14. this plan and the parent plan receive consistent registry and archive
    bookkeeping after completion.

## Approval Boundary

Approval of this draft should explicitly authorize the listed test and
development-plan edits plus one isolated local live run. Any production or
prompt correction discovered by that run requires a separate diagnosis,
development plan or approved amendment, and explicit implementation command.
