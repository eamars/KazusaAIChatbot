# DSH Phase 3 Trigger-Source E2E Sign-Off Reset Plan

- **Status:** completed on 2026-08-31; the final-code matrix passed 10/10
  technical and independent behavior review, and independent closure review
  passed.
- **Date:** 2026-08-31
- **Parent plan:**
  `development_plans/archive/completed/short_term/dsh_brain_bigbang_cutover_and_legacy_resolution_decommission_plan_2026-08-26.md`
- **Implementation authority:** the user approved this plan on 2026-08-31 and
  explicitly commanded implementation, a batch-first initial live pass,
  systematic failure analysis, and remediation. The listed test and plan edits,
  isolated local services, guarded test databases/workspaces, and live runs are
  authorized. Any evidence-driven production remediation must first be added as
  a bounded amendment naming its exact owner and acceptance checks. Deployment,
  production data, and environment-file changes remain outside this plan.
- **Change direction:** replace the current Phase 3 live release oracle with
  exactly two independent E2E cases for each canonical cognition trigger
  source. The resulting matrix contains ten live nodes: six positive DSH-entry
  proofs and four explicit non-entry proofs for sources whose current
  contracts close DSH admission.

## Summary

The current DSH Phase 3 live suite is an effective exploratory harness and a
poor release oracle. Its three true user-wire nodes occupy 3,164 lines and
contain 195 assertions. One node alone combines a two-turn clarification,
pending-state persistence, task admission, worker execution, DSH terminal
state, result recurrence, dialog, dispatcher delivery, direct Mongo
inspection, protected trace inspection, internal stage counts, and literal
output matching. Repeated failures from that node drove increasingly
fixture-shaped corrections even after the functional DSH result had reached
the user.

This revised proposal uses the production `TriggerSource` registry as the
coverage boundary:

1. `user_message`;
2. `internal_thought`;
3. `self_cognition`;
4. `scheduled_tick`;
5. `tool_result`.

Each source receives exactly two independently runnable live cases. The tests
enter through the source's production owner rather than injecting a fabricated
episode after that owner. Automated assertions prove stable source lineage,
DSH admission or deliberate non-admission, typed lifecycle state, result
grounding, delivery, idempotency, and cleanup. A separately recorded behavior
review judges semantic usefulness and character behavior without requiring an
exact phrase, language, sentence order, opaque marker, tool sequence, or model
stage count.

## Approved Batch-First Execution Amendment — 2026-08-31

The user approved the plan with one execution-order change. The first live pass
runs all ten source-matrix nodes in one pytest invocation, regardless of the
status of earlier nodes. The harness must retain a complete case dossier and
cleanup record for every node that reaches case setup. No behavioral or prompt
change is made during that batch.

After the batch finishes, the executor groups failures by shared boundary and
looks for systematic causes across source setup, readiness, cognition,
admission, DSH lifecycle, recurrence, delivery, evidence capture, and cleanup.
A case-specific expected phrase, language choice, stage count, or tool sequence
is not a fix target. Remediation addresses the smallest demonstrated owning
boundary and preserves every first-pass artifact. Subsequent verification runs
are one node at a time with output and evidence inspected before the next run.

This amendment intentionally overrides the repository's usual one-at-a-time
live-LLM execution order for the initial ten-node diagnostic pass only. It does
not waive stable contracts, behavior review, cleanup, or the requirement to
inspect each retained case dossier.

## First-Pass Systematic Failure Analysis And Remediation Amendment — 2026-08-31

The required first pass ran all ten nodes in one unchanged pytest invocation:

```powershell
venv\Scripts\python -m pytest -m live_llm -q -s tests\test_dsh_user_message_e2e_live_llm.py tests\test_dsh_internal_thought_e2e_live_llm.py tests\test_dsh_self_cognition_e2e_live_llm.py tests\test_dsh_scheduled_tick_e2e_live_llm.py tests\test_dsh_tool_result_e2e_live_llm.py
```

The batch completed in 468.42 seconds with 2 passed and 8 failed. No test,
prompt, model route, or production file changed while it ran. Every artifact
directory listed below is preserved under
`test_artifacts/dsh_trigger_source_e2e/`.

| Matrix node | First-pass result | Preserved artifact suffix | Initial classification |
|---|---|---|---|
| `user_message_local_fact` | failed | `user_message_local_fact_20260831T001134Z_68d84294` | harness settlement and cleanup reporting |
| `user_message_background_summary` | failed before readiness | `user_message_background_summary_20260831T001337Z_60fcba30` | transient Mongo startup plus harness exception-boundary defect |
| `internal_thought_file_check` | failed after a grounded DSH result | `internal_thought_file_check_20260831T001345Z_0e5f8504` | production internal-latch profile hydration defect |
| `internal_thought_comparison` | failed before readiness | `internal_thought_comparison_20260831T001500Z_d4f8a05f` | transient Mongo startup plus harness exception-boundary defect |
| `self_cognition_targetless_group` | failed | `self_cognition_targetless_group_20260831T001506Z_251e38c5` | production null-carrier normalization plus stale trace selector |
| `self_cognition_promoted_group` | failed | `self_cognition_promoted_group_20260831T001554Z_0adc187f` | stale trace selector only; source worker and non-entry gate passed |
| `scheduled_tick_commitment_due` | passed | `scheduled_tick_commitment_due_20260831T001653Z_629de028` | green control for positive scheduled entry |
| `scheduled_tick_future` | failed before readiness | `scheduled_tick_future_20260831T001815Z_4643ae07` | transient Mongo startup plus harness exception-boundary defect |
| `tool_result_resolved` | failed before cognition | `tool_result_resolved_20260831T001824Z_0ca6bbf1` | invalid test-owned evidence-handle fixture |
| `tool_result_failed` | passed | `tool_result_failed_20260831T001831Z_0858dacf` | green control for recurrence-closed failed result |

### Shared evidence and root judgment

The eight red nodes reduce to five shared causes. They do not establish eight
independent DSH behavior failures:

1. Three cases lost the same remote Mongo connection during Brain
   `db_bootstrap` and Uvicorn exited with `SystemExit(3)`. The case runner caught
   `Exception`, while `SystemExit` escaped that boundary; consequently those
   cases retained their spec and process logs but missed `case_result.json` and
   `cleanup.json`. Both recorded process ids for every affected case are now
   stopped. The three exact guarded test databases remained; bounded recovery
   cleanup then verified all three present before the drop and absent after it.
   This is one external readiness incident plus one harness dossier/cleanup
   defect, not evidence about the three trigger paths.
2. The local user-message case satisfied every DSH binding, source, terminal
   result, evidence, and visible-answer check. The harness captured its source
   trace while asynchronous post-turn settlement still marked it `running`,
   then treated a slow embedded-Uvicorn shutdown as an incomplete functional
   result even though the owned child process exited and the guarded database
   was dropped. Evidence capture needs a bounded source-trace settlement probe
   and cleanup needs to distinguish graceful shutdown from verified forced
   closure.
3. Both group-review traces used the production source identity
   `self_cognition:group_activity_window:<scope-ref>:...`; the harness searched
   for the retired string `self_cognition:group_chat_review:`. The promoted
   case's worker succeeded, its review ledger settled, and it created zero DSH
   bindings. Its two red trace checks were false negatives.
4. The resolved tool-result fixture put `local-result:<case-id>` in
   `evidence_handles`, while the production result-source contract requires the
   evidence row's semantic `summary` to be present in that list. The failed
   result fixture contained no evidence and therefore passed. The resolved
   case failed before cognition because its test seed was internally
   inconsistent.
5. Two product-owned boundaries were exposed. First,
   `self_cognition.worker._case_from_internal_action_latch` resolves a bound
   target user id and then unconditionally replaces the case's `user_profile`
   with `{}`. DSH itself resolved the internal file correctly, after which
   consolidation failed on the missing user `cognition_state` and the latch
   returned to pending. Second, the targetless group model returned the same
   semantically empty `"pending_task_continuation": null` carrier on all three
   bounded P attempts. The `fresh_ordinary` contract already treats that field
   as optional, but the evaluator rejected explicit null even though canonical
   output represents absence as `None`. Removing an explicit null in this one
   optional fresh-response position changes no semantic decision, authority,
   permission, or task admission.

No first-pass failure showed a DSH sidecar protocol error, wrong positive-source
binding, recursive `tool_result` admission, fabricated group user, ungrounded
terminal result, or exact-wording mismatch. The first pass therefore supports
bounded owner fixes and does not support prompt tuning or a broader DSH
architecture change.

### Authorized bounded remediation

The user's approved instruction to perform the evidence-backed fix now applies
to the following exact surface:

- `tests/dsh_trigger_source_e2e_support.py`:
  - convert embedded Uvicorn `SystemExit` into a typed case failure, observe a
    completed Brain task during readiness, and always reach dossier/cleanup;
  - poll the source-bound trace to a terminal state before capture;
  - bind group traces by the case's real `scope_ref` and canonical
    `group_activity_window` identity;
  - report graceful versus forced embedded-server closure while requiring the
    owned task to be stopped;
  - retry only exact guarded-database cleanup after transient driver errors;
  - make the resolved tool-result evidence handle equal its semantic evidence
    reference.
- `src/kazusa_ai_chatbot/self_cognition/worker.py`: hydrate the real bound user
  profile through the existing DB owner before projecting an internal-latch
  case; preserve `{}` for genuinely targetless cases and retain downstream
  validation.
- `tests/test_self_cognition_integration.py`: add a deterministic regression
  proving the internal-latch case receives the hydrated bound profile.
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`: for
  `fresh_ordinary` only, normalize an explicit null
  `pending_task_continuation` to canonical absence when no human-clarification
  request exists. A non-null carrier without human clarification remains an
  error, and every non-fresh variant retains its exact field allowlist.
- `tests/unit/cognition_core_v3/test_handleless_contract.py`: prove null-only
  fresh-response normalization and retain the existing rejection tests for a
  non-null unauthorized carrier and non-fresh carrier echoes.

This amendment changes no prompt, model route, DSH schema, source registry,
task-readiness rule, targetless-self identity policy, sidecar, environment
file, production data, or deployment state. Acceptance requires the two new
deterministic regressions, the focused deterministic suite, and individual
reruns of affected live nodes with complete dossiers. A new semantic or
architectural failure discovered during rerun receives another recorded
diagnosis before any further change.

### Individual-rerun infrastructure diagnosis — 2026-08-31

The first isolated rerun of `internal_thought_file_check` failed before
readiness and preserved
`internal_thought_file_check_20260831T004550Z_5ed9b39a`. Its Brain log records
the same remote-Mongo `AutoReconnect: connection closed` during
`db_bootstrap` seen in the first-pass batch. The source entrypoint, cognition,
and DSH runtime never ran. The amended exception boundary produced a typed
failed dossier, stopped the Brain and sidecar, stopped the adapter, and dropped
the exact guarded database successfully. This confirms that dossier/cleanup
repair while also showing that a DSH sign-off case remains vulnerable to an
unrelated transient during repeated collection creation.

The test owner will prepare the exact guarded case database before starting
Brain by running the production `db_bootstrap` and canonical character seed
under a maximum of three attempts. Only `PyMongoError` is retryable; every
other exception and an exhausted Mongo retry fail the case. The harness closes
the Mongo client between attempts, records each attempt in
`database_preparation.json`, and still requires the real Brain lifespan to
start and report ready afterward. This is test-environment conditioning for
the remote database dependency; it does not retry a cognition/DSH outcome or
turn a failed source execution green.

### Individual-rerun causal-state diagnosis — 2026-08-31

The isolated `tool_result_failed` rerun preserved
`tool_result_failed_20260831T010234Z_50214975`. It retained the typed failed
outcome, `blocked` evidence state, `tool_result` source, settled failed trace,
and zero DSH bindings or sessions. Delivery failed before dialog with
`CognitionStateError: duplicate knowledge_gaps entity id`, leaving the accepted
task retryable and producing no callback.

The protected trace contains three ordinary cognition cycles. The first two
requested the offered `self_goal_resolution` capability, which correctly
returned a private-only/user-source blocker; the final P result requested no
resolver and selected a visible blocked response. During recurrence, a prior
cycle terminalized the source-scoped knowledge gap. The next A1 call revisited
the same primary evidence. `materialize_causal_root` excluded the terminal row
from eligible reuse, then generated the same deterministic entity id and
appended it, causing the duplicate-id validator failure. This is a shared
deterministic state-lifecycle defect exposed by a valid failed-result path, not
a DSH recurrence error or wording failure.

The bounded production correction owns only
`src/kazusa_ai_chatbot/cognition_shared/state_reducers.py::materialize_causal_root`.
When the exact deterministic causal id already belongs to a terminal row with
the same evidence identity, the reducer will reactivate that identity in
place: preserve creation time and retained evidence, reset it to the canonical
active shape for its kind, and apply the current appraisal normally. An
eligible row continues to be reused; an id/evidence conflict still fails
closed; no parallel id, alias, prompt rule, or semantic post-classifier is
introduced. The mapped regression is
`tests/unit/cognition_core_v3/test_state_transaction.py::test_terminal_same_source_causal_root_reactivates_without_duplicate`.
The state-reducer row in
`tests/ownership/source_test_impact_manifest.json` will name that exact node.

The source census exposes one important architectural fact. The currently
reachable plain `self_cognition` producer is targetless group review. It cannot
advertise `task_resolution_request` because task readiness requires a real
user identity. `tool_result` is also deliberately recurrence-closed by the
`tool_result_delivery` response-plan contract. Creating positive DSH-entry
tests for either source would fabricate an unreachable runtime case or change
production architecture. This plan therefore gives both sources two negative
E2E proofs. If positive, identity-bound plain self-cognition DSH entry is a
product requirement, Phase 3 cannot be signed off under this test-only plan;
that requirement needs a separate production design and explicit authority.

### Post-rerun oracle review — independent sidecar lineage

The implementation-owner code review found that `dsh_lineage.sessions` was
constructed by iterating persisted Brain bindings. Consequently, a negative
case with zero bindings necessarily reported zero sessions without
independently reading the sidecar store. The zero-binding check still proves
that the Brain did not persist DSH admission, but the advertised zero-session
proof was redundant and insufficient for final sign-off.

The shared harness will enumerate the isolated case's SQLite `sessions` table
directly. A store whose persistence schema has not been materialized represents
zero sessions; a materialized store is read by exact session id. Positive
cases require exactly one sidecar session matched to the sole Brain binding
and its typed terminal event. Negative cases require both zero Brain bindings
and zero independently enumerated sidecar sessions. Because this changes a
hard release oracle, all ten cases rerun individually and produce new dossiers
before technical completion can be restored. This correction changes only
test evidence collection and assertions; it does not alter production or
prompt behavior.

### Independent-lineage rerun status-coherence diagnosis — 2026-08-31

The stronger `scheduled_tick_future` rerun preserved
`scheduled_tick_future_20260831T013656Z_7e2b720f`. Its new lineage gate passed:
one Brain binding matched exactly one independently enumerated sidecar session,
and the DSH terminal event/result contained grounded file evidence. The source
worker and calendar run failed afterward. DSH had submitted the conflicting
combination `status=resolved`, `evidence_state=complete`, and a non-empty
`remaining_needs` list describing optional independent cross-check data. The
canonical task-result validator accepted it; the resolver-observation
validator then correctly rejected complete evidence with remaining needs.

This is a cross-boundary terminal-status contract gap, not a scheduled-source,
sidecar-lineage, prompt, or test-wording failure. The resolver cannot map
`resolved` to `partial` because that would deterministically reinterpret the
producer's semantic disposition. The producing DSH tool contract must reject
the incoherent combination so its bounded agent loop can replace the terminal
submission. Brain's canonical result boundary must independently fail closed
if the sidecar ever emits it.

The bounded correction owns exactly:

- `sidecars/dsh_resolution/src/contracts.ts::validateSubmitResolution`, where
  `resolved` requires zero remaining needs and `partial` requires at least one;
- `src/kazusa_ai_chatbot/task_resolution/contracts.py::validate_task_resolution_result`,
  where `resolved` requires evidence and no remaining needs, while `partial`
  requires evidence and remaining needs;
- `src/kazusa_ai_chatbot/task_resolution/projection.py::project_dsh_exhaust`,
  which must return the validated canonical result rather than an unchecked
  shaped dictionary.

Acceptance adds status-coherence cases to the existing sidecar contract test,
the exact Python task-result contract test, and the exact exhaust-projection
test. Sidecar typecheck, unit tests, and build must pass before the affected
live node reruns. No prompt, model route, status mapper, evidence text, or
visible-output assertion changes. A different subsequent live failure receives
another diagnosis before any further edit.

### Independent behavior review and L3 evidence-authority remediation — 2026-08-31

The first independent review accepted nine authoritative dossiers and failed
`tool_result_resolved_20260831T014454Z_19b55a40`. Its typed result establishes
only that the bounded source check completed and the requested handover fact is
available. P preserves that the fact's text remains unknown. The L3 content
planner broadens this into reduced risk and readiness to proceed, then dialog
strengthens it into eliminated risk and permission to proceed. This is a
material false-completion failure rather than acceptable wording variation.

The repeated mode is supported by the retained earlier technical-pass artifact
`tool_result_resolved_20260831T010042Z_85aa45e0`, whose callback claims an
immediate next vigilance-task execution stage from the same bounded availability
evidence. That artifact is historical rather than part of the authoritative
independent-lineage ledger; it is cited here to support the systematic diagnosis.

Upstream DSH projection, recurrence, P response selection, and P's epistemic
boundary remain correct owners and are unchanged. The bounded semantic
remediation owns exactly:

- `src/kazusa_ai_chatbot/cognition_shared/surface_stages.py::CONTENT_PLAN_SYSTEM_PROMPT`,
  which must distinguish scoped resolver/evidence completion from authority to
  assert downstream consequences, safety, permission, follow-through, or
  readiness;
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py::_V2_DIALOG_GENERATOR_PROMPT`,
  which must treat the epistemic boundary, typed resolver evidence, and
  permitted action results as the upper assertion bound even when a content
  plan overstates them.

Both semantic stages may preserve character stance, propose or condition a next
step, and render natural voice. They may not convert source availability,
empty `remaining_needs`, or `evidence_state=complete` into proof of a broader
scene consequence. Private motive, relationship posture, and character idiom
shape expression only; they do not supply factual or action authority.

The correction changes no schema, model route, temperature, retry cap,
deterministic validator, result projection, prompt input, harness pass
condition, or production data. It adds no phrase list, forbidden-answer token,
exact visible-output assertion, or post-generation semantic rewrite.

Acceptance requires:

1. the existing L3 surface and dialog owner suites pass unchanged;
2. both changed Python files compile and are Ruff-clean;
3. the two retained dialog-component live nodes rerun individually for prompt
   regressions;
4. the six matrix cases with visible L3 output rerun individually:
   `user_message_local_fact`, `user_message_background_summary`,
   `scheduled_tick_commitment_due`, `scheduled_tick_future`,
   `tool_result_resolved`, and `tool_result_failed`;
5. the four silent/non-speech dossiers remain valid because their production
   path does not invoke either changed L3 owner;
6. a fresh independent reviewer applies the same semantic rubric to all six new
   dossiers, reconfirms the four unaffected decisions, and accepts the final
   implementation scope before closure.

A new red run receives a fresh systematic diagnosis before any additional
production or prompt edit.

### Deferred dialog-component fixture diagnosis — 2026-08-31

The first post-remediation run of
`test_live_dialog_renders_deferred_grounded_result` failed before any model
call. Its handcrafted `background_work_job.v2` fixture carried only legacy
job-level `result_summary` and omitted the required persisted
`TaskResolutionResultV1`. The production result-source boundary correctly
failed closed because job summaries are not result authority.

This is a stale reclassified component fixture, not an L3 prompt or production
failure. The test-only correction replaces the legacy job literal with the
existing canonical completed-job fixture, then sets a coherent typed `partial`
result containing Pydantic evidence and the optional plugin check as its
remaining need. The visible semantic assertions, production result-source
contract, prompt remediation, and ten-node E2E harness remain unchanged. The
node reruns once after this fixture correction, and both failed and passing
outputs remain in the execution ledger.

That first corrected rerun reached evidence-reference projection and failed a
second canonical fixture invariant: each result evidence `summary` must appear
in `evidence_handles`, while the fixture had placed its evidence id there. The
same test-only correction uses the evidence summary as the semantic handle.
This changes no result meaning, visible oracle, or production owner.

The next rerun exposed the remaining legacy oracle: it required the projected
semantic summary to equal the job-level summary exactly. Canonical partial
projection deliberately appends the typed remaining limitation. The systematic
fixture migration therefore removes job-summary authority from this component
node and asserts the stable typed `partial` status, evidence state/excerpts,
and remaining need. Natural visible wording remains free to paraphrase those
facts.

### Scheduled recurrence multi-excerpt provenance diagnosis — 2026-08-31

The first post-L3-remediation run of
`scheduled_tick_commitment_due` passed every technical gate but its visible
result reported only owner `Priya` and omitted the deployment-window
prerequisite. The DSH terminal result contains three ordered evidence excerpts,
including both requested findings. The post-resolver P input contains only
excerpt index zero. P and L3 therefore cannot express the missing prerequisite.

The loss occurs in the shared resolver recurrence projection. One DSH evidence
receipt may carry several prompt-safe findings. `_task_resolution_evidence_refs`
associates the receipt with the first excerpt, while
`project_resolver_observation_for_cognition` reads only that ref and ignores the
validated `knowledge_projection.knowledge_we_know_so_far` list that already
preserves all typed excerpts. This is deterministic provenance loss, not model
variation or a pass-condition issue.

The bounded correction owns exactly:

- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py::project_resolver_observation_for_cognition`,
  which merges the validated task knowledge rows into the cognition evidence
  excerpt projection in stored order, deduplicates exact repeats, and retains
  the existing four-item/500-character and aggregate prompt bounds;
- `tests/unit/cognition_resolver/test_capabilities.py`, which proves that two
  typed findings under one evidence receipt both reach cognition exactly once;
- the existing `cognition_resolver/capabilities.py` impact-manifest row, which
  names that regression.

The correction preserves the observation schema, evidence ids, owners,
authority classification, status, prompt-safe summary, P prompt, L3 prompts,
and test pass conditions. It performs no semantic synthesis. After owner,
compile, Ruff, impact, and diff gates pass, only the affected scheduled case
reruns before the remaining planned live cases continue. Any further red or
semantically incomplete result receives a new diagnosis before another edit.

The correction was implemented after the first post-L3 reruns of both
`user_message` cases. It also affects positive `internal_thought` task-result
recurrence, even though those cases are normally silent. Final-code evidence
therefore reruns both user-message and both internal-thought nodes after the
scheduled correction. The scheduled and tool-result nodes already ran after
the correction. The two targetless self-cognition cases invoke neither L3 nor
task-result recurrence and retain their authoritative dossiers.

### Final non-live closure fixture diagnosis — 2026-08-31

The first final-code full non-live suite completed 3,367 passes before ending
with three deterministic failures:

- `tests/test_task_resolution_background_resume.py::test_worker_projects_terminal_runtime_result`;
- `tests/test_task_resolution_contracts.py::test_nonempty_coding_context_is_rejected`;
- `tests/unit/db/test_task_resolution_sessions.py::test_binding_generation_attach_checkpoint_terminal_and_followup_reconcile_is_revision_guarded`.

All three construct `status=resolved` with zero DSH evidence. The strengthened
terminal-coherence contract correctly rejects those fixtures before the tests
reach their intended worker-projection, coding-context, or binding-CAS owner.
The failure is shared test-fixture drift, not a production result-contract
defect. The bounded correction adds a valid typed DSH evidence receipt to each
resolved fixture, aligns the shared contract-test helper's resolved
`remaining_needs` with the canonical empty value, and updates the worker test
to assert the projected evidence. It changes no production source, pass
condition, prompt, schema, model route, data, or deployment state.

Acceptance requires the three failed nodes to pass together, their three test
files to compile and pass Ruff, impact ownership to remain valid, and the full
non-live suite to pass before closure bookkeeping resumes.

## Execution Evidence And Current Gate State — 2026-08-31

The required unchanged all-ten diagnostic batch completed first with 2 passed
and 8 failed in 468.42 seconds. Every reachable dossier was retained before
remediation. The grouped analysis above separated database and harness faults,
fixture invalidity, two narrow source/cognition contract defects, and the later
shared causal-state recurrence defect. No failure supported prompt tuning,
literal-output matching, a source-registry change, or a DSH architecture
rewrite.

The first isolated reruns passed a binding-derived session oracle. Post-rerun
review found that this was not an independent proof of sidecar materialization,
so it was retained only as historical evidence. The harness now enumerates the
isolated sidecar SQLite `sessions` table directly. All ten nodes then reran one
at a time under exact one-binding/one-matched-session positive gates or exact
zero-binding/zero-session negative gates.

The first independent review then exposed one material L3 false-completion,
and the first post-L3 scheduled recurrence run exposed deterministic
multi-excerpt provenance loss. The two prompt owners and recurrence projection
were corrected under the bounded amendments above. Every affected node reran
individually under the final code. The two targetless self-cognition cases
invoke neither changed owner and retain their previously reviewed artifacts.
These are the final-code authoritative artifacts:

| Trigger source | Case | Expected disposition | Authoritative passing artifact |
|---|---|---|---|
| `user_message` | `user_message_local_fact` | one DSH entry | `user_message_local_fact_20260831T023137Z_bd34ab0c` |
| `user_message` | `user_message_background_summary` | one DSH entry | `user_message_background_summary_20260831T023332Z_393501cc` |
| `internal_thought` | `internal_thought_file_check` | one DSH entry | `internal_thought_file_check_20260831T023641Z_175f97f9` |
| `internal_thought` | `internal_thought_comparison` | one DSH entry | `internal_thought_comparison_20260831T024123Z_a2b5bd8a` |
| `self_cognition` | `self_cognition_targetless_group` | zero DSH entry | `self_cognition_targetless_group_20260831T013227Z_36a89d38` |
| `self_cognition` | `self_cognition_promoted_group` | zero DSH entry | `self_cognition_promoted_group_20260831T013337Z_b4ce2afc` |
| `scheduled_tick` | `scheduled_tick_commitment_due` | one DSH entry | `scheduled_tick_commitment_due_20260831T022500Z_9e0717ed` |
| `scheduled_tick` | `scheduled_tick_future` | one DSH entry | `scheduled_tick_future_20260831T022648Z_f0c202ea` |
| `tool_result` | `tool_result_resolved` | zero recursive DSH entry | `tool_result_resolved_20260831T022839Z_8fc8d347` |
| `tool_result` | `tool_result_failed` | zero recursive DSH entry | `tool_result_failed_20260831T022951Z_841cd835` |

Every authoritative artifact has `technical_status=passed`, a finalized source
trace, and successful guarded database/service cleanup. Each of the six
positive cases has exactly one Brain binding and exactly one independently
enumerated sidecar session matched to that binding. Each of the four non-entry
cases has zero bindings and zero independently enumerated sidecar sessions.
The failed `tool_result` case preserves `failed`, `blocked`, and its remaining
need through one eligible delivery without recursive DSH lineage.

The stronger `scheduled_tick_future` rerun first failed in retained artifact
`scheduled_tick_future_20260831T013656Z_7e2b720f`: the DSH terminal submission
combined `resolved`, complete evidence, and non-empty remaining needs. The
recorded cross-contract correction makes both the sidecar producer and Brain
result boundary reject that incoherent disposition. A subsequent passing
sample returned coherent `partial` evidence. The final-code artifact listed
above instead returned coherent `resolved/complete` because the model treated
the note's stated canary-health and error-budget conditions as satisfying the
requested note check; its warning explicitly says no independent telemetry or
cross-check was performed. The fresh independent review accepted this as
truthful for the requested note-inspection objective rather than a claim of
independently verified rollout health. No fixed terminal-status expectation is
added.

The first post-L3 `scheduled_tick_commitment_due` artifact passed the stable
oracle but omitted the deployment-window prerequisite from the visible result.
The typed DSH result contained it; shared recurrence projected only the first
receipt-linked excerpt. The final projection now merges receipt excerpts with
the already validated knowledge projection, preserves source order, removes
duplicates, and retains the existing item and character bounds without
semantic synthesis. Its final artifact reports both owner `Priya` and the
deployment-window prerequisite.

Both dialog-component nodes passed individually after the stale deferred
fixture was replaced by a coherent typed `partial` result:
`test_artifacts/task_resolution/raw/dialog_inline_grounded_result.json` and
`test_artifacts/task_resolution/raw/dialog_deferred_grounded_result.json`.
The latter preserves the optional plugin compatibility check as a remaining
need instead of asserting an exact legacy job summary.

Deterministic and mechanical evidence after the final correction is:

- 153 focused DSH owner tests passed;
- 33 direct L3 surface/dialog owner tests passed;
- six exact remediation variants passed: latch profile hydration, fresh null
  carrier normalization, all three event/threat/knowledge-gap same-source
  recurrence variants, and multi-excerpt cognition projection;
- all five task-result contract/projection tests passed;
- 486 exact source-impact nodes validated;
- the full sidecar suite passed 101 tests across 14 files;
- sidecar typecheck and production build passed;
- all changed Python files compiled;
- live collection is exactly ten trigger-source E2Es plus the two retained
  dialog-component nodes;
- the first final-code full non-live run exposed the three stale resolved/
  zero-evidence fixtures recorded above; their exact three-node rerun passed
  after test-only correction, and the authoritative full rerun passed 3,370
  tests with four opt-in skips and 508 live tests deselected;
- scoped Ruff and `git diff --check` pass.

Full-file Ruff reports ten existing production findings outside the edited
hunks in `self_cognition/worker.py`, `cognition_core_v3/facade.py`, and
`cognition_shared/surface_stages.py`; sixteen paired legacy `return None`
findings elsewhere in `test_self_cognition_integration.py`; and one existing
generator-style finding before the new recurrence test in
`test_state_transaction.py`. The new and modified hunks introduce none of
those findings. A scoped run excluding those documented legacy codes passes;
broad unrelated style cleanup is outside this focused sign-off plan.

The first independent reviewer accepted nine authoritative dossiers and failed
the resolved `tool_result` dossier for the repeated false-completion pattern
recorded in the amendment above. The input dossiers remain decision-free with
`behavior_decision=null`; the independent decisions are recorded in this plan
ledger rather than written back into evidence by the read-only reviewer. The
stable technical oracle remains green because it owns typed-result
preservation, delivery, and recurrence closure rather than exact wording.

The fresh read-only independent review accepted all ten final-code dossiers.
It found no blocker, high, or medium behavior issue. It explicitly accepted:

- the corrected resolved-tool callback as future character intention rather
  than risk clearance, permission, readiness, or completed follow-through;
- the scheduled-future result as a bounded interpretation of its fixture note
  with the lack of independent telemetry clearly disclosed;
- the failed-tool blind-spot/vulnerability language as grounded character
  framing for an unavailable required source;
- the background-summary timestamp limitation and both silent internal paths.

The reviewer recorded one low, non-material language observation: the
internal-comparison result calls English `owns` present-progressive even though
it is simple present. The quoted evidence, actor, responsibility change, and
behavioral conclusion remain exact, so this is acceptable natural-language
variation rather than evidence distortion or a reason to tune the prompt.

The strengthened technical matrix remains complete at 10/10 under the final
code, and the bounded L3 and recurrence remediations plus all required live
reruns are complete. Independent behavior/scope review is green. This focused
E2E layer is signed off; parent release-candidate closure reconciliation and
archive bookkeeping remain open at this checkpoint.

## Sign-Off Answer

Yes. The technically complete ten-node matrix is sufficient as the final E2E
layer for Phase 3 DSH integration, and an independent reviewer has accepted all
ten final behavior dossiers. It covers every currently reachable positive
DSH-entry source twice and proves that both canonical recurrence/readiness-
closed sources remain closed. It supplements, rather than replaces, the
deterministic authority, schema, recovery, safety, decommission, and component-
live gates in the parent plan.

It does not prove that targetless plain self-cognition can enter DSH; current
production contracts intentionally prevent that. A requirement for two
positive plain-self-cognition entries changes the answer to **not sufficient**
until an identity-bearing production source is designed, implemented, and
verified under a separate approved plan.

## System Boundary And Definition Of A Path

For this sign-off, a path is one canonical cognition `TriggerSource` and its
production source owner. Foreground versus delayed execution, one versus
several specialist steps, native versus semantic tools, and exact model-stage
choreography are implementation variants rather than additional trigger
paths. Focused owner tests retain those private contracts.

The source-to-DSH boundary is:

```text
production trigger owner
  -> canonical CognitiveEpisodeV1.trigger_source
  -> shared cognition and resolver affordances
  -> deterministic task-resolution readiness
  -> DSH task edge when semantically selected and ready
  -> inline result or accepted task/job/binding
  -> DSH terminal result
  -> tool_result recurrence when delayed
  -> dialog/dispatcher/adapter when outward delivery is selected
```

### Production source census

| Canonical source | Production owner and real entry | Current DSH eligibility | Sign-off treatment |
|---|---|---|---|
| `user_message` | `brain_service.intake` through `service._process_queued_chat_item` and public `POST /chat` | Entry-capable when the normal user envelope and DSH runtime are ready | Two positive entry E2Es |
| `internal_thought` | durable `db.internal_action_latches`, claimed by `self_cognition.worker`, then passed to the shared runner | Entry-capable when the latch carries a bound real-user target | Two positive entry E2Es |
| `self_cognition` | reflection-owned group activity window through `self_cognition.sources.collect_group_review_cases` and the self-cognition worker | The reachable producer is targetless; readiness omits task resolution | Two negative non-entry E2Es |
| `scheduled_tick` | due calendar runs collected by the self-cognition worker; concrete producers are active-commitment due and scheduled-future cognition | Entry-capable when the due source carries a bound real-user target | Two positive entry E2Es, one per concrete producer |
| `tool_result` | `background_work.delivery` through `background_work.result_source.build_result_ready_episode_from_job` | Deliberately closed to recursive task admission by the `tool_result_delivery` contract | Two negative recurrence E2Es |

Internal DSH question, approval, and plan-review cognition is an internal
sub-loop marked `dsh_interaction_episode=True`, not a sixth trigger source. Its
non-recursion contract remains in focused deterministic and component-live
tests and is not inflated into two more release E2Es.

## Scope And Exclusions

### Must do

- five source-focused live-E2E files with exactly two nodes each;
- one shared test-support module for isolated services, data, workspace,
  callbacks, evidence capture, and cleanup;
- retirement of the current Phase 3 omnibus DSH E2E nodes as release gates;
- deletion of the Phase 3-specific P-stage live probes;
- accurate component-test naming for the dialog-only live suite;
- focused deterministic verification, one initial all-ten live diagnostic
  batch, and isolated one-case remediation verification runs;
- one evidence dossier and one independent behavior decision per live node;
- a superseding Phase 3 sign-off amendment and plan lifecycle bookkeeping;
- the bounded production corrections named in the remediation amendments and
  their deterministic regressions.

### Deferred and unchanged

- every production function except the exact owners named in the remediation
  amendments;
- prompts except the exact two L3 prompt owners named in the independent-review
  amendment, model routes, retry policies, and contract field sets, plus all
  validator behavior except the optional-carrier normalization and terminal
  status-coherence rules named in earlier amendments;
- deployed processes, environment files, and production data;
- deterministic owner tests and retained DSH component-live tests;
- the general Stage 3 fresh-database source-smoke suite;
- the parent plan's non-E2E acceptance gates.

## Review Basis

This proposal was formed from the current source, tests, documentation, and
retained artifacts, including:

- `src/kazusa_ai_chatbot/cognition_episode.py` and its five-source registry;
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py` readiness rules;
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` affordance
  projection and internal-DSH recursion closure;
- `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` response-plan variant
  closure;
- the self-cognition source, worker, runner, reflection, internal-latch,
  calendar, result-source, and delivery owners;
- `tests/test_dsh_e2e_live_llm.py`;
- `tests/test_dsh_cognition_admission_live_llm.py`;
- `tests/test_task_resolution_persona_e2e_live_llm.py`;
- `tests/test_stage3_fresh_database_e2e_live_llm.py`;
- `tests/test_agentic_resolver_live_llm.py`;
- `tests/test_dsh_standard_profile_live_llm.py`;
- `tests/test_dsh_brain_interaction_live_llm.py`;
- the deterministic task-resolution, worker, result-source, cognition,
  self-cognition, calendar, and background-delivery owner tests;
- the parent plan's complete live-E2E execution ledger and amendments;
- the retained `test_artifacts/dsh_plan3_e2e/prerequisite_admission_*`
  artifacts.

The local prerequisite-admission artifact set contains 20 directories, 19
parseable run records, 18 recorded failures, and only three runs that reached
delivered task state. All three delivered runs followed the same functional
path. One passed because its callback repeated the synthetic beta marker
literally; the other two failed only at the final literal assertion.

## Current Test Strategy Inventory

| Surface | Size and nodes | Current strategy | Review judgment | Proposed disposition |
|---|---:|---|---|---|
| `tests/test_dsh_e2e_live_llm.py` | 3,164 lines, 3 nodes, 195 assertions | Start real Brain, DSH, adapter, isolated Mongo, and workspace; inspect public responses, private DB rows, DSH event log, traces, tool choreography, and exact visible fixture markers | Genuine user-wire E2E boundary, but an omnibus migration proof with an unstable and over-specified oracle; covers only `user_message` | Replace with the five-source matrix |
| Inline node in that file | 641 lines, 81 assertions | Force one inline DSH task, require a literal marker, exact terminal lineage, and at least two A1/A2/G/P passes | Internal stage-count and wording requirements exceed release-sign-off needs | Replace with one natural user-message entry case |
| Prerequisite/background node in that file | 862 lines, 61 assertions | Force alpha/beta clarification, exact pending carrier, zero Turn 1 rows, exact Turn 2 state, beta-only execution, exact trace stages, callback cardinality, and a literal beta marker | Principal overfit node; binds sign-off to one internal decomposition and one opaque token | Replace with one natural user-message background case |
| Public research/media node in that file | 342 lines, 27 assertions | Tell the model exact native and semantic tools to call, require at least two semantic results, pin a remote image, and police legacy names | Tests choreography and network availability more than natural task behavior | Remove from Phase 3 sign-off; retain media/tool owners |
| `tests/test_dsh_cognition_admission_live_llm.py` | 483 lines, 6 nodes, 30 assertions | Give P-stage prose that states the expected route, then assert exact resolver enum, timing, pending carrier, and closed variants | Failure-shaped producer probes, not E2E behavior; encourage prompt/contract tuning around one expected answer | Delete; retain deterministic P-contract coverage |
| `tests/test_task_resolution_persona_e2e_live_llm.py` | 297 lines, 2 nodes, 9 assertions | Construct synthetic episodes and surfaces, call only the dialog generator, and check input-owned anchors plus runtime-word exclusions | Useful dialog component smoke tests, but not E2E | Rename and retain as `tests/test_task_resolution_dialog_live_llm.py` |
| `tests/test_stage3_fresh_database_e2e_live_llm.py` | 1,911 lines, 10 nodes, 46 explicit checks | Exercise source labels, lifecycle settlement, selected Stage 3 paths, and fresh-DB evidence | Valuable general source smoke. Its internal/self/scheduled helpers construct cases directly at the runner, set `execute_private_actions=False`, and never require DSH selection, admission, execution, or result; its private ordinary self-cognition case has no current production producer | Retain unchanged and exclude from DSH sign-off evidence |
| `tests/test_agentic_resolver_live_llm.py` | 348 lines, 2 nodes, 18 assertions | Exercise the standalone runtime and sidecar protocol directly | Useful component diagnostics, not Brain/source integration | Retain outside the sign-off command |
| `tests/test_dsh_standard_profile_live_llm.py` | 295 lines, 4 nodes, 5 assertions | Exercise Standard-profile tool selection and semantic capabilities directly | Useful component diagnostics | Retain outside the sign-off command |
| `tests/test_dsh_brain_interaction_live_llm.py` | 104 lines, 1 node, 4 assertions | Exercise Brain-owned internal DSH judgment directly | Useful internal-interaction diagnostic, not source-entry E2E | Retain outside the sign-off command |

`tests/test_e2e_live_llm.py` contains no DSH resolution behavior. Its textual
matches are ordinary `BackgroundTasks` plumbing and place it outside this
review.

## Failure History And Judgment

### What the repeated failures established

The parent plan initially defined five final live nodes. Its 2026-08-31 coding
test de-overfitting amendment removed two coding-specific nodes, leaving the
current inline, prerequisite/background, and media nodes. The remaining
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

Several runs found real contract defects. Those findings merit their focused
deterministic regressions. The alpha/beta scenario remains useful historical
discovery evidence rather than a permanent release oracle.

### Why the latest recorded failure is a test-oracle failure

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
negative. The callback's meta-style wording belongs in a separate behavior
observation. Durable evidence proves successful DSH admission, execution,
recurrence, and delivery.

### Root judgment

The central failure is the sign-off architecture:

- one stochastic E2E is used as a unit test for many private contracts;
- the user messages describe the intended internal route and tool sequence;
- exact model-stage counts and raw persistence shapes freeze implementation;
- opaque marker reproduction substitutes for semantic review;
- a fixed remote media asset adds an unrelated availability gate;
- generated audit prose restates expected conditions rather than independently
  judging the visible interaction;
- source coverage is almost entirely `user_message`, while self-cognition,
  internal-latch, scheduled, and result-recursion behavior is only component or
  source-label coverage;
- each red result creates pressure to change general prompts or model contracts
  until one fixture turns green.

The suite is over-sensitive to acceptable wording and internal variation while
remaining under-sensitive to source reachability and whether the final behavior
is grounded, useful, and appropriately scoped.

## Proposed Ten-Node Sign-Off Matrix

All fixture content is natural language in an isolated local workspace. Test
inputs state the source-owned objective and timing only. They do not mention
DSH, Standard profile, resolver enums, native or semantic tool names, RAG,
expected JSON, prompt stages, expected output wording, or synthetic markers.

### `user_message`: two positive entry cases

File: `tests/test_dsh_user_message_e2e_live_llm.py`

1. `test_live_user_message_local_fact_reaches_dsh`
   - create one short rollout note containing a precondition and owner;
   - submit one natural private `POST /chat` request asking what the note says;
   - accept inline completion or bounded durable promotion;
   - require one source-bound DSH resolution result with evidence referring to
     that file and one coherent visible answer path.
2. `test_live_user_message_background_summary_reaches_dsh`
   - create a different short handover note;
   - submit one natural request that explicitly asks for background work and a
     brief later summary;
   - require one accepted task/job/binding lineage, terminal DSH result,
     `tool_result` recurrence, and one registered adapter callback.

These two cases cover direct and explicitly delayed user intent without
requiring a particular private tool or model-stage sequence.

### `internal_thought`: two positive entry cases

File: `tests/test_dsh_internal_thought_e2e_live_llm.py`

1. `test_live_internal_thought_file_check_reaches_dsh`
   - issue a real identity-bound internal-action latch whose natural
     continuation objective requires checking one local status note;
   - let the production self-cognition worker claim and consume the latch;
   - require canonical `internal_thought` lineage and one grounded DSH result.
2. `test_live_internal_thought_comparison_reaches_dsh`
   - issue a second real latch whose continuation objective requires comparing
     two short local shift notes before a planned update;
   - run the same production claim/worker boundary;
   - require one DSH result grounded in both source files.

Outward speech is not a hard requirement for an internal source. Silence,
private continuation, or source-bound delivery remains character-owned. If an
adapter send occurs, it must use the latch's bound target, occur once, and pass
the behavior review.

### `self_cognition`: two negative non-entry cases

File: `tests/test_dsh_self_cognition_e2e_live_llm.py`

1. `test_live_targetless_group_review_omits_dsh_task_resolution`
   - seed a real group activity window containing an ambient, task-like file
     question;
   - run the production reflection group-review phase and self-cognition
     worker without injecting a case at the runner;
   - require canonical `self_cognition`, a targetless group scope, no fabricated
     user identity, and zero source-linked task/DSH records.
2. `test_live_promoted_group_review_omits_dsh_task_resolution`
   - seed a distinct group window with promoted reflection context and a
     different task-like comparison question;
   - run the same production source boundary;
   - require the same readiness closure and zero recursive or fabricated DSH
     admission.

The model may stay silent or produce a grounded group reply through other
valid affordances. The hard gate checks the executable-identity boundary, not
one exact response decision. These are the honest E2Es for the currently
reachable plain-self-cognition source.

### `scheduled_tick`: two positive entry cases

File: `tests/test_dsh_scheduled_tick_e2e_live_llm.py`

1. `test_live_commitment_due_tick_reaches_dsh`
   - seed a real user profile, visible source history, active-commitment memory
     unit, and its due calendar run;
   - let the production calendar handler/source collector and self-cognition
     worker build the case;
   - require canonical `scheduled_tick`, preserved user identity, and one DSH
     result grounded in the requested local note.
2. `test_live_scheduled_future_tick_reaches_dsh`
   - seed a distinct due future-cognition calendar run with real user-bound
     source scope and a natural local evidence objective;
   - let the production due-run collector and worker build the case;
   - require canonical `scheduled_tick` and one grounded DSH result.

These are the two concrete production producers currently collapsed into the
canonical scheduled source. Contact timing and visible speech remain
character-owned unless the source contract itself requires delivery.

### `tool_result`: two negative recurrence cases

File: `tests/test_dsh_tool_result_e2e_live_llm.py`

1. `test_live_resolved_tool_result_delivers_without_recursive_dsh`
   - persist one completed background job with a validated resolved
     `TaskResolutionResultV1` and source-backed evidence;
   - run the production result-ready cognition and delivery boundary;
   - require canonical `tool_result`, one settled recurrence, one eligible
     delivery attempt, and zero new source-linked task/DSH admission.
2. `test_live_failed_tool_result_settles_without_recursive_dsh`
   - persist a distinct completed job with a validated failed result and its
     declared remaining need;
   - run the same production recurrence boundary;
   - require a truthful settled failure surface or character-owned silence and
     zero recursive task/DSH admission.

These tests protect the exact recursion closure that repeated Node 2 failures
eventually established. They do not require the model to repeat the stored
summary verbatim.

## Test Construction Rules

### Real source-owner entry

Each node starts before its trigger owner:

- user messages enter only through public `POST /chat`;
- internal thoughts enter through durable latch issue and worker claim;
- self-cognition enters through a real reflection activity window and group
  review phase;
- scheduled ticks enter through stored due calendar runs and production source
  collection;
- tool results enter through a stored typed terminal job and production
  result-ready delivery.

The live nodes may replace external transport with the registered test HTTP
adapter and may isolate Mongo, ports, and workspace. They may not inject
`CognitiveEpisodeV1`, call the cognition resolver directly, patch the model
answer, pass a custom `collect_cases_func`, or call the self-cognition runner
with a case that no production source emits.

### Automated positive-entry hard gates

Every positive node requires:

1. the isolated Brain, DSH sidecar, database, workspace, and relevant worker
   owner are ready;
2. the source enters through its production boundary and settles under the
   expected canonical trigger source;
3. one source-linked task-resolution execution reaches DSH;
4. the final typed result validates and has a terminal status allowed by the
   task contract;
5. a factual `resolved` or `partial` result carries non-empty DSH evidence and
   local fixture provenance;
6. source trigger, source episode, task/binding, job when present, DSH session,
   result, and delivery when present form one coherent lineage;
7. no duplicate task, DSH terminalization, or adapter delivery occurs during a
   bounded duplicate check;
8. protected source and child traces plus a compact review dossier are retained;
9. all test-owned resources are cleaned and the isolated database is dropped.

An inline-sized case may promote to durable work if the foreground budget is
exhausted. That is an allowed implementation outcome, provided the same source
lineage reaches a terminal grounded result. The explicit-background user case
must honor its requested delayed timing.

### Automated non-entry hard gates

Every negative node requires:

1. the source is produced and settled through its real owner;
2. the expected canonical source and response-plan/readiness boundary are
   recorded;
3. zero task, accepted-task, background-job, DSH-session, and task-binding rows
   are attributable to that source episode;
4. self-cognition does not fabricate an executable user identity;
5. tool-result recurrence retains its original task/result lineage and does
   not create a new task lineage;
6. any eligible outward delivery occurs at most once;
7. protected traces, review evidence, and cleanup outcomes are retained.

The absence check is source-correlated. It does not assert that an isolated
database is globally empty or inspect unrelated concurrent records.

### Assertions excluded from live sign-off

The ten nodes do not hard-code:

- an exact visible sentence, token, language, punctuation, or sentence order;
- a phrase list used as a proxy semantic judge;
- an exact A1/A2/G/P call count or private stage sequence;
- one particular native or semantic tool choreography;
- a fixed remote network asset;
- an exhaustive raw Mongo document shape;
- plan-specific schema labels or artifact names in model output;
- deleted executor names inside arbitrary private serialized blobs;
- repeated reruns until a preferred model sample appears.

## Independent Behavior Review

Each node writes a compact dossier containing its natural source input, source
kind, target scope, visible context, immediate response if any, final callback
if any, typed result summary, bounded evidence and limitations, task/delivery
disposition, protected trace references, and cleanup outcome. The dossier
contains evidence and no prewritten verdict.

The reviewer records `pass` or `fail` with a short rationale for:

1. whether DSH entry or non-entry was appropriate for the source and available
   identity/evidence;
2. whether a factual result is faithful to its local source and useful for the
   source objective;
3. whether the response avoids contradictions, invented completion, and
   fabricated user attribution;
4. whether silence or outward speech is appropriate to the character, scene,
   privacy, and trigger source;
5. whether visible text avoids raw task ids, hidden prompts, authority tokens,
   and control-plane internals;
6. whether the overall source-to-result interaction is coherent.

Paraphrase, Chinese or English output, sentence order, character voice, and
omission of test-only wording are accepted. Naturalness and voice block Phase
3 only when the behavior is materially misleading, unusable, privacy-breaking,
or inconsistent with grounded character judgment.

## Failure Classification And Rerun Policy

Every red run receives exactly one initial classification:

| Class | Meaning | Response allowed in this plan |
|---|---|---|
| Environment or harness | service readiness, port, configured model availability, isolated DB, trace capture, source setup, cleanup, or harness ownership failed | Complete the initial batch, correct the owned harness/environment issue, retain the failed artifact, then rerun affected nodes individually |
| Stable contract regression | source binding, readiness, admission, task binding, worker, terminal result, recurrence, dispatcher, or delivery contract failed | Complete the initial batch, group related failures, record a bounded production-remediation amendment naming the exact owner and checks, then implement the user-authorized systematic fix |
| Behavioral failure | the path completed but entry judgment or visible behavior was materially wrong, misleading, ungrounded, privacy-breaking, or unusable | Complete the initial batch, inspect the related traces together, record the systematic semantic diagnosis and exact remediation boundary, then implement the user-authorized fix without fixture-shaped prompt tuning |
| Acceptable variation | wording or private decomposition differs while stable gates and the human rubric pass | Record green; make no prompt or contract change |
| Topology mismatch | a proposed positive path has no reachable production source or executable identity | Stop; amend the architecture under a separate approved production plan rather than fabricating a test case |

Repeated reruns to obtain a favorable sample are not sign-off. A rerun is
permitted only after a recorded environment/harness correction or an approved
code change, and both artifacts remain in the evidence ledger.

## Supporting-Test Disposition

### Retire the omnibus Phase 3 E2E file

Delete `tests/test_dsh_e2e_live_llm.py` after its reusable infrastructure has
been reduced and moved into `tests/dsh_trigger_source_e2e_support.py`. Its
three nodes and alpha/beta/media scenarios remain historical artifact evidence
and cease to be release gates.

### Delete the Phase 3 P-stage live probes

Delete `tests/test_dsh_cognition_admission_live_llm.py`. Its exact foreground,
background, prerequisite, pending, post-pending, and tool-result contracts
remain owned by deterministic cognition tests. The new positive E2Es observe
natural live admission without telling P which enum to choose.

### Reclassify dialog-only tests

Rename:

```text
tests/test_task_resolution_persona_e2e_live_llm.py
  -> tests/test_task_resolution_dialog_live_llm.py
```

Rename its two nodes to describe dialog rendering rather than E2E behavior.
Keep their source-supplied anchors and protected-runtime-word boundary because
they are component contracts. Exclude them from the Phase 3 sign-off command.

### Retain general source and component suites

Retain `tests/test_stage3_fresh_database_e2e_live_llm.py` as general Stage 3
source/lifecycle evidence. Its source-label cases do not count toward this DSH
matrix because they do not prove DSH admission and some begin after the real
source owner.

Retain the direct agentic-resolver, Standard-profile, Brain-interaction,
gateway, media-safety, task-resolution, self-cognition, calendar, worker,
result-source, cognition, and delivery suites. They remain diagnostic or exact
owner coverage rather than extra Phase 3 release E2Es.

## Change Surface

### Add

- `tests/dsh_trigger_source_e2e_support.py`: own only shared isolated service,
  guarded DB/workspace, callback, source-lineage, dossier, and cleanup support;
- `tests/test_dsh_user_message_e2e_live_llm.py`: own the two public `/chat`
  positive-entry cases;
- `tests/test_dsh_internal_thought_e2e_live_llm.py`: own the two durable-latch
  positive-entry cases;
- `tests/test_dsh_self_cognition_e2e_live_llm.py`: own the two targetless group
  review non-entry cases;
- `tests/test_dsh_scheduled_tick_e2e_live_llm.py`: own the two concrete due-run
  positive-entry producers;
- `tests/test_dsh_tool_result_e2e_live_llm.py`: own the two recursion-closed
  terminal-result cases.

### Delete

- `tests/test_dsh_e2e_live_llm.py`: remove the omnibus alpha/beta/media release
  oracle after bounded shared infrastructure is extracted;
- `tests/test_dsh_cognition_admission_live_llm.py`: remove outcome-shaped P
  probes after confirming their exact contracts remain deterministically owned.

### Rename and edit

- `tests/test_task_resolution_persona_e2e_live_llm.py` to
  `tests/test_task_resolution_dialog_live_llm.py`: preserve the two dialog-only
  component cases under accurate ownership; change only node names and artifact
  labels needed to remove the false E2E claim.

### Evidence-driven production remediation

- `src/kazusa_ai_chatbot/self_cognition/worker.py`: hydrate the identity-bound
  internal-latch user profile before shared cognition and consolidation;
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`: normalize only an
  explicit null optional fresh-response continuation carrier to absence;
- `src/kazusa_ai_chatbot/cognition_shared/state_reducers.py`: reactivate an
  exact terminal same-source causal identity in place before current-cycle
  appraisal reduction;
- `tests/test_self_cognition_integration.py`: add the bound-profile regression;
- `tests/unit/cognition_core_v3/test_handleless_contract.py`: add the null-only
  normalization regression while preserving non-null and non-fresh rejection;
- `tests/unit/cognition_core_v3/test_state_transaction.py`: add the exact
  terminal same-source reactivation regression;
- `tests/ownership/source_test_impact_manifest.json`: add that exact node only
  to the existing state-reducer owner row.

### Documentation and lifecycle

- add a top-level superseding trigger-source E2E amendment to the parent Phase
  3 plan;
- replace the parent plan's surviving live-E2E commands and P3-G10 wording with
  the ten-node matrix plus independent dossiers;
- retain the old execution ledger as historical evidence;
- update `development_plans/README.md` during status transitions;
- archive this plan after every acceptance gate passes.

### Explicitly unchanged

- every file under `src/` except the exact production owners above, and
  every file under `sidecars/`;
- `tests/test_stage3_fresh_database_e2e_live_llm.py`;
- all prompts, output schemas, model routes, retries, and validator semantics
  other than the null-only normalization above;
- DSH, task, job, binding, interaction, result, and adapter schemas;
- every test-impact manifest row except the existing state-reducer row;
- environment files, production databases, and deployment state.

## Test Impact And Traceability

| Exact governed path or contract | Semantic owner | Exact deterministic pytest node IDs | Supplemental live node IDs | Mode | Regression prevented |
|---|---|---|---|---|---|
| `src/kazusa_ai_chatbot/cognition_episode.py::TriggerSource`; `build_trigger_source_registry` | Cognitive episode source registry | `tests/test_stage3_trigger_source_cutover.py::test_trigger_source_literal_contains_only_five_grounded_sources`; `tests/test_stage3_trigger_source_cutover.py::test_trigger_source_registry_is_complete` | none | deterministic unit | A source disappears, an alias path returns, or the sign-off census diverges from the canonical five-source contract. |
| `src/kazusa_ai_chatbot/service.py::_process_queued_chat_item`; `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py::_execute_task_resolution_request` | User-message intake and generic task-resolution edge | `tests/unit/brain_service/test_dsh_task_readiness.py::test_task_capability_is_available_only_when_full_dsh_runtime_is_ready`; `tests/unit/cognition_resolver/test_capabilities.py::test_task_resolution_preserves_recurrence_and_maps_dsh_deferred_result`; `tests/unit/task_resolution/test_service.py::test_inline_checkpoint_promotes_same_bound_dsh_session_without_canceling_reasoning`; `tests/unit/task_resolution/test_service.py::test_background_start_mints_authority_only_when_claimed` | `tests/test_dsh_user_message_e2e_live_llm.py::test_live_user_message_local_fact_reaches_dsh`; `tests/test_dsh_user_message_e2e_live_llm.py::test_live_user_message_background_summary_reaches_dsh` | deterministic unit plus isolated live LLM/DB/service | Public user input fails to reach DSH, delayed authority is minted at the wrong boundary, or result recurrence/delivery loses source lineage. |
| `src/kazusa_ai_chatbot/db/internal_action_latches.py`; `src/kazusa_ai_chatbot/self_cognition/worker.py::_case_from_internal_action_latch`; `runner._build_cognitive_episode` internal-latch branch | Durable internal-thought source and shared cognition runner | `tests/test_internal_action_latches.py::test_internal_action_latch_schema_contains_fixed_lifecycle_fields`; `tests/test_internal_action_latches.py::test_internal_action_latch_repository_exposes_atomic_lifecycle_api`; `tests/test_self_cognition_integration.py::test_internal_latch_case_hydrates_bound_user_profile`; `tests/test_cognition_resolver_loop.py::test_internal_thought_uses_unified_task_resolution_path` | `tests/test_dsh_internal_thought_e2e_live_llm.py::test_live_internal_thought_file_check_reaches_dsh`; `tests/test_dsh_internal_thought_e2e_live_llm.py::test_live_internal_thought_comparison_reaches_dsh` | deterministic unit plus isolated live LLM/DB/worker | A latch bypasses the shared DSH edge, drops its bound user's cognition state, loses its bound target, duplicates consumption, or becomes a fabricated user-message path. |
| `src/kazusa_ai_chatbot/reflection_cycle/worker.py::_run_group_self_cognition_review_for_scope`; `src/kazusa_ai_chatbot/self_cognition/sources.py::collect_group_review_cases`; task-resolution readiness | Reflection-owned targetless plain self-cognition | `tests/test_self_cognition_group_review_source.py::test_collect_group_chat_review_cases_builds_same_group_cases`; `tests/test_cognition_resolver_contracts.py::test_targetless_group_self_cognition_bootstraps_without_user_owner`; `tests/unit/cognition_resolver/test_capabilities.py::test_task_capability_uses_runtime_readiness_without_legacy_fallback` | `tests/test_dsh_self_cognition_e2e_live_llm.py::test_live_targetless_group_review_omits_dsh_task_resolution`; `tests/test_dsh_self_cognition_e2e_live_llm.py::test_live_promoted_group_review_omits_dsh_task_resolution` | deterministic unit plus isolated live LLM/DB/reflection worker | Group review fabricates an executable user, admits unauthorized DSH work, or a synthetic private case is mistaken for production reachability. |
| `src/kazusa_ai_chatbot/self_cognition/sources.py::collect_commitment_due_cognition_cases`; `collect_scheduled_future_cognition_cases`; `runner._build_cognitive_episode` scheduled branch | Calendar scheduler and self-cognition scheduled-source adapter | `tests/test_self_cognition_integration.py::test_collect_commitment_due_cognition_cases_projects_calendar_runs`; `tests/test_self_cognition_integration.py::test_worker_tick_marks_commitment_due_run_completed`; `tests/test_self_cognition_integration.py::test_collect_scheduled_future_cognition_cases_preserves_source_scope`; `tests/test_self_cognition_integration.py::test_worker_tick_marks_future_cognition_run_completed` | `tests/test_dsh_scheduled_tick_e2e_live_llm.py::test_live_commitment_due_tick_reaches_dsh`; `tests/test_dsh_scheduled_tick_e2e_live_llm.py::test_live_scheduled_future_tick_reaches_dsh` | deterministic unit plus isolated live LLM/DB/calendar/self-cognition worker | A due producer is skipped, loses user/source provenance, maps to the wrong trigger, or cannot use the shared DSH task edge. |
| `src/kazusa_ai_chatbot/background_work/result_source.py::build_result_ready_episode_from_job`; `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` `tool_result_delivery` capability closure; result-ready delivery | Background-work result source, cognition recurrence, and dispatcher | `tests/unit/background_work/test_result_source.py::test_dsh_result_reenters_cognition_with_exact_goal_and_evidence_provenance`; `tests/unit/cognition_core_v3/test_handleless_contract.py::test_tool_result_delivery_variant_closes_recursive_admission`; `tests/test_background_work_delivery.py::test_service_result_ready_delivery_uses_dispatcher_boundary` | `tests/test_dsh_tool_result_e2e_live_llm.py::test_live_resolved_tool_result_delivers_without_recursive_dsh`; `tests/test_dsh_tool_result_e2e_live_llm.py::test_live_failed_tool_result_settles_without_recursive_dsh` | deterministic unit plus isolated live LLM/DB/service/adapter | Result recurrence recursively opens a task, loses typed evidence, misstates a failure, or bypasses dispatcher delivery. |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` internal DSH interaction affordance closure | Brain-owned internal DSH judgment | `tests/test_dsh_brain_interaction_decision.py::test_dsh_interaction_full_loop_advertises_only_internal_resolvers`; `tests/test_dsh_brain_interaction_decision.py::test_dsh_interaction_runs_full_reusable_cognition_loop_and_returns_internal_decision` | `tests/test_dsh_brain_interaction_live_llm.py::test_brain_cognition_answers_or_rejects_dsh_request_from_context` | deterministic unit plus retained component live LLM | The authenticated internal DSH sub-loop recursively admits a new user task or leaks into the trigger-source sign-off matrix. |
| `src/kazusa_ai_chatbot/cognition_core_v3/facade.py::_validate_plan` optional fresh continuation carrier | Canonical P-stage structural normalization | `tests/unit/cognition_core_v3/test_handleless_contract.py::test_fresh_ordinary_null_pending_task_continuation_normalizes_to_absence`; existing non-null and non-fresh rejection nodes in the same file | `tests/test_dsh_self_cognition_e2e_live_llm.py::test_live_targetless_group_review_omits_dsh_task_resolution` | deterministic unit plus isolated live LLM/DB/reflection worker | A semantically empty null carrier exhausts bounded cognition, or normalization accidentally permits unauthorized task continuation. |
| `src/kazusa_ai_chatbot/cognition_shared/state_reducers.py::materialize_causal_root` terminal same-source recurrence | Shared deterministic causal-state lifecycle | `tests/unit/cognition_core_v3/test_state_transaction.py::test_terminal_same_source_causal_root_reactivates_without_duplicate` | `tests/test_dsh_tool_result_e2e_live_llm.py::test_live_failed_tool_result_settles_without_recursive_dsh` | deterministic unit plus isolated live LLM/DB/service/adapter | Resolver recurrence revisits one terminal same-source root, appends its stable id twice, and aborts a grounded failure delivery. |

## Mandatory Skills And Rules

Implementation must apply:

- `development-plan` for lifecycle and the parent-plan amendment;
- `test-style-and-execution` before editing or running tests;
- `local-llm-architecture` to preserve semantic judgment and prevent
  fixture-shaped prompt pressure;
- `py-style` before editing Python;
- `cjk-safety` only if a Python test contains CJK string content;
- `character-test` for source-to-result evidence capture, per-turn/episode
  trace inspection, and independent behavior dossiers.

The implementation owner preserves unrelated workspace changes and does not
read `.env`.

## Executor Autonomy Boundaries

The executor may:

- edit, delete, rename, or add only the test, plan, and bounded production
  owners listed in the amended change surface;
- extract shared test-only process, data, callback, artifact, and cleanup
  helpers into the named support module;
- use current exported production contracts and collection owners from tests;
- correct a demonstrable harness defect inside the owned test files and record
  both failed and corrected artifacts;
- implement the exact evidence-driven production corrections recorded in the
  remediation amendments and their mapped deterministic tests.

The executor pauses for user direction when implementation would require:

- any production, sidecar, prompt, model-route, schema, validator, or retry
  change beyond the exact amended production corrections;
- making targetless self-cognition identity-bound or DSH-entry-capable;
- adding a sixth trigger source or more than two live cases for one source;
- production data access, environment-file inspection, or deployment action;
- weakening a stable authority, safety, persistence, or delivery contract;
- accepting a behavior-review failure through an automated wording rule.

## Required Roles

### `e2e_test_owner`

- **Responsibility:** implement the five-source test matrix and produce complete
  hard-gate evidence for each node.
- **Owned surface:** the listed add/delete/rename test paths in `Change Surface`,
  this plan, the parent-plan amendment, the registry row, and test-owned local
  artifacts, plus the exact production owners and mapped deterministic
  regressions in the remediation amendments.
- **Authority:** edit the owned test/plan files, start isolated local services,
  create and drop only the guarded test database, create and remove only the
  test workspace, implement the bounded production corrections, and run
  the listed checks after approval. It has no authority over other production
  code, prompts, model routes, schemas, production data, environment files, or
  deployment.
- **Applicable skills:** `development-plan`, `test-style-and-execution`,
  `local-llm-architecture`, `py-style`, `character-test`, and `cjk-safety` when
  its trigger condition applies.
- **Capability floor:** senior Python test engineering, async service/worker
  orchestration, Mongo isolation, DSH/task lifecycle knowledge, protected trace
  handling, and the ability to distinguish stable contracts from stochastic
  semantics.
- **Independence requirement:** none for implementation; this role cannot issue
  its own behavior or final closure sign-off.
- **Acceptance output:** scoped diff, exact collection output, deterministic
  verification, ten live artifacts, ten cleanup records, and a completed
  execution-evidence ledger.
- **Gate:** starts only after draft approval and an implementation command;
  exits only when the owned diff passes mechanical/deterministic checks and
  every live node has one classified artifact ready for independent review.

### `behavior_reviewer`

- **Responsibility:** decide whether each source's DSH entry/non-entry judgment
  and visible or silent behavior is semantically acceptable.
- **Owned surface:** read-only access to the ten dossiers, bounded protected
  trace evidence, typed results, callbacks, and source inputs; one review
  decision artifact per node.
- **Authority:** inspect and pass or fail behavior under the stated rubric. It
  cannot edit tests or production code, reinterpret hard failures as wording
  variation, or authorize remediation.
- **Applicable skills:** `character-test`, `local-llm-architecture`, and
  `development-plan` for the acceptance boundary.
- **Capability floor:** character-brain judgment, source/provenance reasoning,
  privacy awareness, and enough DSH architecture knowledge to distinguish
  grounded paraphrase from a false completion.
- **Independence requirement:** must be separate from the executor that wrote or
  remediated the reviewed test/result. If remediation occurs, a fresh
  independent review is required.
- **Acceptance output:** ten explicit `pass`/`fail` decisions with short
  evidence-based rationales and no code changes.
- **Gate:** starts after a node's hard-gate artifact is complete; exits only
  after every node has a decision and every failure is preserved and routed to
  the correct remediation boundary.

### `phase3_closure_owner`

- **Responsibility:** perform the independent scope/evidence audit and decide
  whether the E2E layer is ready to close Phase 3 alongside the parent gates.
- **Owned surface:** read-only implementation diff and evidence; plan/registry
  lifecycle fields and archival moves after all gates pass.
- **Authority:** fail closure for scope drift, stale or missing nodes, weak
  evidence, unresolved behavior findings, or unmet parent gates; update plan
  lifecycle records after acceptance. It cannot remediate code or waive a gate.
- **Applicable skills:** `development-plan`, `test-style-and-execution`, and
  `local-llm-architecture`.
- **Capability floor:** independent system-level review of source ownership,
  test architecture, git scope, evidence completeness, and Phase 3 lifecycle.
- **Independence requirement:** must be separate from the executor responsible
  for the final implementation/remediation diff.
- **Acceptance output:** independent scope review, ten-node evidence index,
  parent-gate reconciliation, residual-risk statement, and consistent
  plan/registry/archive status.
- **Gate:** starts after implementation verification and behavior review;
  exits only when all findings are resolved or explicitly returned to the user
  and no parent-plan gate is represented as waived.

## Implementation Order

1. Capture `git status --short`, the exact owned-file baseline, current test
   collection, source registry, production call-site census, and artifact
   inventory.
2. Add the superseding trigger-source and batch-first amendments to the parent
   plan before changing tests.
3. Extract the minimal shared test support and add the five two-node source
   files.
4. Delete the old omnibus and P-stage live files; rename the dialog-only file.
5. Run collection, compile, Ruff, focused deterministic owners, impact
   validation, and diff hygiene.
6. Run all ten live nodes in one pytest invocation. Preserve every outcome and
   perform no mid-batch remediation.
7. Inspect all dossiers together, classify generic/shared failure modes, and
   record the systematic diagnosis before changing behavior.
8. Apply the smallest evidence-backed fix at the owning boundary. Amend this
   plan first if the demonstrated owner is production code.
9. Rerun affected live nodes individually and inspect each complete artifact
   before starting the next node.
10. Obtain an independent behavior decision for each final dossier.
11. If all hard and behavior gates are green, record evidence, confirm the
   remaining Phase 3 closure gates, update registry status, archive completed
   plans, and request final closure acknowledgment where required.

## Verification Commands

Use the project virtual environment. Deterministic tests run in a batch. The
first live diagnostic pass uses the single all-ten command below. After its
failure-mode review, affected live-LLM nodes run one at a time with output and
artifacts inspected after each command.

```powershell
venv\Scripts\python -m pytest -m live_llm --collect-only -q tests/test_dsh_user_message_e2e_live_llm.py tests/test_dsh_internal_thought_e2e_live_llm.py tests/test_dsh_self_cognition_e2e_live_llm.py tests/test_dsh_scheduled_tick_e2e_live_llm.py tests/test_dsh_tool_result_e2e_live_llm.py tests/test_task_resolution_dialog_live_llm.py

venv\Scripts\python -m py_compile tests/dsh_trigger_source_e2e_support.py tests/test_dsh_user_message_e2e_live_llm.py tests/test_dsh_internal_thought_e2e_live_llm.py tests/test_dsh_self_cognition_e2e_live_llm.py tests/test_dsh_scheduled_tick_e2e_live_llm.py tests/test_dsh_tool_result_e2e_live_llm.py tests/test_task_resolution_dialog_live_llm.py

venv\Scripts\python -m ruff check tests/dsh_trigger_source_e2e_support.py tests/test_dsh_user_message_e2e_live_llm.py tests/test_dsh_internal_thought_e2e_live_llm.py tests/test_dsh_self_cognition_e2e_live_llm.py tests/test_dsh_scheduled_tick_e2e_live_llm.py tests/test_dsh_tool_result_e2e_live_llm.py tests/test_task_resolution_dialog_live_llm.py

venv\Scripts\python -m pytest -q tests/unit/brain_service/test_dsh_task_readiness.py tests/unit/task_resolution/test_service.py tests/unit/background_work/test_dsh_worker.py tests/unit/background_work/test_result_source.py tests/unit/cognition_resolver/test_capabilities.py tests/test_cognition_resolver_loop.py tests/test_internal_action_latches.py tests/test_calendar_scheduler_worker.py tests/test_calendar_scheduler_active_commitments.py tests/test_self_cognition_group_review_source.py tests/test_background_work_delivery.py

venv\Scripts\python -m pytest -q tests/test_self_cognition_integration.py::test_internal_latch_case_hydrates_bound_user_profile tests/unit/cognition_core_v3/test_handleless_contract.py::test_fresh_ordinary_null_pending_task_continuation_normalizes_to_absence

venv\Scripts\python -m pytest -q tests/unit/cognition_core_v3/test_state_transaction.py::test_terminal_same_source_causal_root_reactivates_without_duplicate

venv\Scripts\python scripts\validate_test_impact.py --check-all

venv\Scripts\python -m pytest -m live_llm -q -s tests/test_dsh_user_message_e2e_live_llm.py tests/test_dsh_internal_thought_e2e_live_llm.py tests/test_dsh_self_cognition_e2e_live_llm.py tests/test_dsh_scheduled_tick_e2e_live_llm.py tests/test_dsh_tool_result_e2e_live_llm.py

# The following node commands are post-analysis remediation verification.
venv\Scripts\python -m pytest -m live_llm tests/test_dsh_user_message_e2e_live_llm.py::test_live_user_message_local_fact_reaches_dsh -q -s

venv\Scripts\python -m pytest -m live_llm tests/test_dsh_user_message_e2e_live_llm.py::test_live_user_message_background_summary_reaches_dsh -q -s

venv\Scripts\python -m pytest -m live_llm tests/test_dsh_internal_thought_e2e_live_llm.py::test_live_internal_thought_file_check_reaches_dsh -q -s

venv\Scripts\python -m pytest -m live_llm tests/test_dsh_internal_thought_e2e_live_llm.py::test_live_internal_thought_comparison_reaches_dsh -q -s

venv\Scripts\python -m pytest -m live_llm tests/test_dsh_self_cognition_e2e_live_llm.py::test_live_targetless_group_review_omits_dsh_task_resolution -q -s

venv\Scripts\python -m pytest -m live_llm tests/test_dsh_self_cognition_e2e_live_llm.py::test_live_promoted_group_review_omits_dsh_task_resolution -q -s

venv\Scripts\python -m pytest -m live_llm tests/test_dsh_scheduled_tick_e2e_live_llm.py::test_live_commitment_due_tick_reaches_dsh -q -s

venv\Scripts\python -m pytest -m live_llm tests/test_dsh_scheduled_tick_e2e_live_llm.py::test_live_scheduled_future_tick_reaches_dsh -q -s

venv\Scripts\python -m pytest -m live_llm tests/test_dsh_tool_result_e2e_live_llm.py::test_live_resolved_tool_result_delivers_without_recursive_dsh -q -s

venv\Scripts\python -m pytest -m live_llm tests/test_dsh_tool_result_e2e_live_llm.py::test_live_failed_tool_result_settles_without_recursive_dsh -q -s

git diff --check
```

## Acceptance Criteria

This plan is complete only when all of the following are true:

1. the parent Phase 3 plan explicitly supersedes its surviving live E2E nodes
   with this five-source, ten-node matrix;
2. collection shows exactly two named live E2E nodes for each canonical source;
3. the source census still matches the five production registry rows and real
   call sites;
4. both `user_message` nodes enter DSH through public `/chat`;
5. both `internal_thought` nodes enter DSH through durable latch issue, worker
   claim, and worker consumption;
6. both `scheduled_tick` producers enter DSH through their real due-run source
   boundaries;
7. both targetless `self_cognition` nodes settle without fabricated identity or
   DSH admission;
8. both `tool_result` nodes settle and deliver when eligible without recursive
   DSH admission;
9. all positive factual results carry validated local-source evidence and
   coherent source-to-result lineage;
10. automated visible-text assertions contain no exact answer token, language,
    phrase list, stage count, or tool choreography;
11. every dossier contains evidence rather than a prewritten verdict;
12. an independent reviewer accepts all ten results under the semantic rubric;
13. the Phase 3-specific P-stage live probe file is removed;
14. the dialog-only suite is retained under an accurate non-E2E name;
15. the general Stage 3 source-smoke suite remains intact and is not counted as
    DSH sign-off evidence;
16. all focused deterministic, collection, compile, Ruff, impact, and diff
    checks pass;
17. the first pass runs all ten nodes together and preserves every reachable
    outcome; shared failure modes are recorded before any remediation; affected
    nodes are then rerun individually, and every failed and successful artifact
    plus rerun reason remains recorded;
18. the implementation changes only the production functions and two L3 prompt
    owners named in the remediation amendments, changes no other prompt,
    schema, model route, environment file, production data, or deployment
    state, and passes their exact owner gates;
19. the parent plan's other closure gates are confirmed rather than waived;
20. this plan and the parent plan receive consistent registry and archive
    bookkeeping after completion.

## Completion Record

The final independent closure review found no material discrepancy and
approved completion/archive bookkeeping. All twenty acceptance criteria are
closed. This plan signs off the focused Phase 3 E2E layer and is archived at
`development_plans/archive/completed/bugfix/`. The parent Plan 3 record owns
the release-candidate-versus-deployment disposition.

## Approval Boundary

The user approved this plan on 2026-08-31, including the listed test and
development-plan edits, the initial all-ten local live batch, systematic
failure analysis, evidence-backed remediation, and individual verification
runs. Approval also confirms the current architectural interpretation that
reachable plain `self_cognition` and `tool_result` are covered by two non-entry
E2Es each.

Any production or prompt correction discovered by a run requires a recorded
systematic diagnosis and a bounded amendment naming the exact owner and
verification. The user's implementation command authorizes that demonstrated
fix; any material architecture expansion returns for explicit direction. A
request for positive plain-self-cognition DSH entry likewise requires a new
production design rather than a synthetic E2E case.
