# background artifact handoff poc and review-gated implementation plan

## Summary

- Goal: Add a reliable async background artifact handoff path where Kazusa can
  accept bounded artifact work, release the live chat response, complete the
  work outside the `/chat` wait path, and later deliver the result to the
  original requester through cognition/dialog/dispatcher.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`, `py-style`, `cjk-safety`,
  `test-style-and-execution`
- Overall cutover strategy: compatible. Stage 0 POC evidence is accepted.
  Stage 1+ is approved by the user on 2026-06-06, subject to this reviewed
  plan revision, the LLM rewrite/input-audit gate, test-contract-first
  execution, and the parent-led native subagent execution model below.
- Highest-risk areas: L2d route fragility, false acknowledgement before durable
  enqueue, duplicate operational ownership with consolidator commitments,
  worker blocking the live chat path, wrong requester/channel delivery,
  duplicate result sends, and first-scope coding scope creep.
- Acceptance criteria: Stage 0 sign-off and Stage 1+ approval are recorded in
  this active plan. Production implementation may start only from this reviewed
  revision and only inside the `Change Surface` and execution gates below.

## Context

The motivating workflow is asynchronous follow-through without blocking normal
chat:

```text
UserA asks Kazusa for a bounded artifact.
Kazusa acknowledges after durable enqueue.
Other users continue normal chat.
The background worker completes the artifact later.
Kazusa delivers the result back to UserA through normal delivery boundaries.
```

The user does not want the `/chat` interface blocked while long-running work is
in progress. The handoff must therefore be operationally owned by durable job
state, not by the consolidator, and not by a future-cognition polling loop.

Stage 0 validated the architecture direction with a production-prompt-parity
L2d POC and a dispatcher dry run. The user accepted the Stage 0 result on
2026-06-06 as best-attempt feasibility evidence. That acceptance does not mean
the production prompt should be overfit to the Stage 0 fixture set.

Stage 0 artifacts:

```text
test_artifacts/background_artifact_handoff/20260606T0338_stage0/
```

Key report:

```text
test_artifacts/background_artifact_handoff/20260606T0338_stage0/report.md
```

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, updating, or
  signing off this plan.
- `local-llm-architecture`: load before changing L2d prompts, action routing,
  background worker LLM behavior, source-case contracts, or delivery cognition.
- `no-prepost-user-input`: load before changing code or prompts that decide
  whether user requests are accepted, queued, remembered, or converted into
  commitments.
- `debug-llm`: load before live/local LLM POC runs, prompt comparison, output
  inspection, or LLM evidence reports.
- `py-style`: load before editing Python production, test, or experiment code.
- `cjk-safety`: load before editing Python files containing CJK strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- The `Execution Model` must use parent-led native subagent execution for
  production implementation unless the user explicitly approves fallback
  execution.
- Stage 1+ execution is approved only after this reviewed revision is the
  active working contract. The parent agent must not begin production edits
  until the LLM rewrite/input-audit checklist item has recorded the affected
  prompt/input contracts and focused test baselines.
- Plan status is not production-code authorization.
- Any production LLM modification must pass an LLM rewrite and input-audit
  gate before production-code subagent execution. The gate must rewrite the
  affected LLM contract, audit every model-facing input type, and record the
  semantic question, required inputs, output fields, deterministic owners, and
  rejected inputs.
- Stage 0 failures are best-attempt risk evidence. Do not tune production
  prompts, examples, deterministic gates, or output parsing to memorize the
  fixed Stage 0 cases.
- Do not add deterministic keyword routing or post-processing over raw user
  text to rescue L2d acceptance decisions.
- L2d owns semantic acceptance and `work_kind`; deterministic code owns
  validation, target binding, enqueue, idempotency, leases, retries, adapter
  feasibility, persistence, and delivery status.
- L2d remains the top-level semantic router only. It must not receive a
  detailed worker/subagent registry or choose concrete worker entrypoints.
- User-facing result delivery must go through source-bound cognition/dialog and
  dispatcher. A worker must not send adapter text directly.
- Durable enqueue must complete before L3/dialog may promise later delivery.
- Do not use consolidator-generated promises as the operational handoff owner.
- Do not use `trigger_future_cognition` as a polling loop for job status.
- Do not use the calendar scheduler as a generic artifact job queue.
- Do not use proactive output as the production delivery path for this feature.
- First-scope coding work is snippet-only: no filesystem writes, shell
  commands, package installs, repository mutation, test execution, downloads,
  network execution, or external research.
- Live/local LLM checks must run one case at a time and be inspected one case
  at a time.

## Must Do

- Preserve Stage 0 sign-off and learnings inside this active plan.
- Keep Stage 1+ production edits blocked until this reviewed plan revision,
  user approval, and the LLM rewrite/input-audit evidence are all recorded.
- Preserve the root capability as `background_artifact_request`.
- Preserve first-scope `work_kind` values:
  `coding_snippet`, `text_rewrite`, and `summary`.
- Implement future production work through durable job ownership, not through
  consolidator promises.
- Bind requester and delivery target from trusted episode/message-envelope
  state, not from L2d-authored ids.
- Add pre-surface durable enqueue before acknowledgement wording in production.
- Add job-layer duplicate enqueue and duplicate delivery suppression.
- Keep the first worker monolithic and snippet/text/summary only.
- Add an explicit `background_artifact_result_ready` cognition source contract
  for completed-job delivery; do not reuse `internal_thought` as the
  model-facing result-delivery source.
- Create `src/kazusa_ai_chatbot/background_artifact/README.md` as a mandatory
  ICD-style module README before production sign-off. It must define document
  control, purpose, scope, parties, boundary summary, public interface, job
  lifecycle, LLM input contract, persistence/delivery ownership, and forbidden
  paths.

## Deferred

- Do not execute Stage 1+ production-code changes outside this reviewed plan,
  the listed file manifest, the Stage 1A test-first gate, or the LLM
  rewrite/input-audit contract.
- Do not implement multi-agent coding orchestration in first scope.
- Do not implement filesystem-editing coding work in first scope.
- Do not implement package install, shell execution, test execution, downloads,
  external network research, image generation, file attachments, or chunked
  multi-message delivery in first scope.
- Do not globally remove existing `future_promises` or active commitments in
  this plan.
- Do not redesign calendar scheduler, proactive output, reflection, RAG,
  adapter protocols, or generic self-cognition loop in this plan.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Stage 0 POC evidence | compatible | Keep as accepted evidence; do not rerun or tune to fixtures unless a later review explicitly requests a new POC. |
| Production action vocabulary | compatible | Add only `background_artifact_request`; keep existing `speak`, `memory_lifecycle_update`, and `trigger_future_cognition` behavior intact. |
| Production job store | migration | Introduce `background_artifact_jobs` through explicit schema/index setup in the approved production stage. |
| Consolidator promises | compatible | Preserve existing commitment memory generally; add only a background-job overlap guard once production work is approved. |
| Calendar scheduler | compatible | Keep calendar for typed time triggers. Do not make it a background artifact queue. |
| Proactive output | compatible | Keep out of this feature's production delivery path. |
| Coding worker | compatible | First worker is snippet-only and monolithic. No repo mutation or shell execution. |

Cutover enforcement:

- Follow the selected policy for each area exactly.
- Preserve only the compatibility surfaces listed in this plan.
- For the migration area, follow the reviewed migration stages and cleanup
  gates recorded by the Stage 1+ plan review.
- Any change to a cutover policy requires user approval before implementation.

## Target State

The intended production flow after approved implementation is:

```text
live user message
  -> shared cognition through current production path
  -> L2d selects speak + background_artifact_request
  -> deterministic pre-surface validation and durable enqueue
  -> L3/dialog acknowledges only if enqueue succeeded
  -> /chat returns
  -> standalone/background worker completes artifact
  -> result-ready source case enters cognition/dialog
  -> dispatcher persists outbound row and delivers through adapter
  -> job delivery status records terminal result
  -> consolidation records memory/audit without duplicate operational promise
```

First-scope artifacts are bounded text artifacts. Large artifacts, attachments,
filesystem edits, and external research remain out of scope.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Root capability | Use one `background_artifact_request` capability with bounded `work_kind`. | Keeps L2d routing pressure low and avoids capability sprawl. |
| Async vs sync | L2d decides semantic acceptance; deterministic policy decides enqueue and validation. | Prevents L2d from owning persistence, worker, retry, or adapter mechanics. |
| Subagent routing | L2d selects `background_artifact_request` plus coarse `work_kind`; the background job owner dispatches internally to the first monolithic worker. | Keeps L2d as semantic router while allowing worker specialization later without prompt sprawl. |
| Dispatcher decision | L2d provides the semantic request; deterministic action handling dispatches sync acknowledgement versus async enqueue based on validated action materialization. | Avoids a keyword router while keeping persistence and adapter safety deterministic. |
| Pre-surface commit | Enqueue before acknowledgement wording. | Prevents false promises when durable job creation fails. |
| Worker boundary | Worker stores artifacts and status only; it does not send adapter text. | Preserves source-bound cognition/dialog and dispatcher delivery. |
| Dispatcher boundary | Dispatcher delivers prepared result messages and receipts; job layer owns duplicate delivery suppression. | Stage 0 proved dispatcher delivery shape but not built-in job dedupe. |
| Consolidator boundary | Consolidator is memory/audit only, not operational handoff owner. | Avoids duplicate schedulable commitments for a job already owned by durable state. |
| Stage 0 interpretation | Accept as best-attempt evidence, not fixture optimization. | Prevents overfitting prompt changes to nine POC cases. |

## Change Surface

This plan-update turn may modify only `development_plans/README.md` and this
plan. The production files below are the approved Stage 1+ execution manifest;
production implementation must still begin with Stage 1A focused tests and must
stay inside this manifest.

If Stage 1+ execution needs any file outside this manifest, stop before the
edit, update this plan, rerun the independent plan review, and obtain user
approval for the revised surface.

### Delete

- No files are planned to be removed.

### Modify

Plan and documentation files:

- `development_plans/README.md`: keep the lifecycle registry row pointing to
  this active plan.
- `development_plans/active/short_term/background_artifact_handoff_poc_plan.md`:
  record Stage 0 evidence, Stage 1+ gates, and the exact file manifest.
- `README.md`: add the `BACKGROUND_ARTIFACT_LLM` route to the model-route
  table after production approval.
- `docs/HOWTO.md`: document `BACKGROUND_ARTIFACT_LLM_*` and
  `BACKGROUND_ARTIFACT_*` worker settings after production approval.
- `src/kazusa_ai_chatbot/brain_service/README.md`: update the brain-service
  ICD for `/ops/runtime-status` background artifact config, worker liveness,
  and callback-delivery boundary text.
- `src/kazusa_ai_chatbot/event_logging/README.md`: update the event-logging
  ICD for background artifact worker event families, safe event fields, and
  aggregate descriptor semantics.
- `src/kazusa_ai_chatbot/action_spec/README.md`: document
  `background_artifact_request` as an L2d semantic action and its forbidden
  paths.
- `src/kazusa_ai_chatbot/nodes/README.md`: document the pre-surface enqueue
  and result-ready source handoff in the persona graph boundary.

Configuration, service, and observability files:

- `src/kazusa_ai_chatbot/config.py`: add the background artifact LLM route and
  worker enable, interval, lease, attempt, input, and output caps.
- `src/kazusa_ai_chatbot/llm_route_report.py`: include
  `BACKGROUND_ARTIFACT_LLM` in startup route reporting without API keys.
- `src/kazusa_ai_chatbot/service.py`: start and stop the background artifact
  runtime, pass the adapter registry provider to delivery code, and expose
  worker liveness in trusted ops status.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`: extend trusted ops
  status config shape with background artifact worker fields.
- `src/kazusa_ai_chatbot/event_logging/status.py`: count background artifact
  worker events and runtime errors in trusted aggregate descriptors.

Action-spec and persona graph files:

- `src/kazusa_ai_chatbot/action_spec/models.py`: add the background artifact
  owner needed by `ActionSpecV1` validation; keep the existing `current_user`
  target kind and bind trusted delivery metadata in deterministic target
  scope.
- `src/kazusa_ai_chatbot/action_spec/registry.py`: register
  `background_artifact_request` as an L2d-visible semantic capability.
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`: validate
  `background_artifact_request` params without exposing adapter ids or job
  internals to L2d.
- `src/kazusa_ai_chatbot/action_spec/execution.py`: execute the validated
  request as a durable enqueue and return a prompt-safe pending result.
- `src/kazusa_ai_chatbot/action_spec/results.py`: allow action results to carry
  the prompt-safe background job reference needed by L3 and consolidation.
- `src/kazusa_ai_chatbot/cognition_episode.py`: add the
  `background_artifact_result_ready` trigger source, `background_artifact_result`
  input source, and builder/validation contract for completed-job delivery
  episodes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`:
  add the `background_artifact_result_ready_background_artifact_result`
  prompt variant and source-payload projection.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`: rewrite
  the L2d prompt/schema/materialization for the new semantic request and
  `work_kind` field under the LLM rewrite/input-audit gate.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: insert the pre-surface
  commit before L3/dialog acknowledgement and carry enqueue outcomes into the
  episode trace.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: add state fields
  for pre-surface background artifact enqueue results.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: expose the
  enqueue result to L3 so acknowledgement wording is allowed only after a
  durable queue row exists.

Persistence and consolidation files:

- `src/kazusa_ai_chatbot/db/__init__.py`: export named background artifact job
  helpers through the DB facade.
- `src/kazusa_ai_chatbot/db/bootstrap.py`: create
  `background_artifact_jobs` and its indexes during startup bootstrap.
- `src/kazusa_ai_chatbot/consolidation/facts.py`: rewrite the facts harvester
  and evaluator prompts so an operationally owned background job may be
  recorded as memory/audit evidence without creating a duplicate
  `future_promises` owner.

Existing test files:

- `tests/test_config.py`: cover new required route variables and worker config.
- `tests/test_llm_route_report.py`: cover the route inventory and API-key
  omission for `BACKGROUND_ARTIFACT_LLM`.
- `tests/test_db.py`: cover bootstrap/index creation for
  `background_artifact_jobs`.
- `tests/test_service_ops_status.py`: cover trusted ops status config and
  liveness projection.
- `tests/test_event_logging_status.py`: cover background artifact worker event
  families in runtime status and semantic descriptors.
- `tests/test_action_spec_models.py`: cover the new owner and target shape.
- `tests/test_action_spec_evaluator.py`: cover valid, rejected, and
  out-of-scope background artifact action params.
- `tests/test_action_spec_results.py`: cover prompt-safe job refs in trace
  projection.
- `tests/test_cognitive_episode_contract.py`: cover the
  `background_artifact_result_ready` episode builder, allowed fields, and
  rejection of adapter/job internals outside the prompt-safe payload.
- `tests/test_multi_source_cognition_stage_03_prompt_selection.py`: cover the
  result-ready prompt variant for every cognition stage and payload projection.
- `tests/test_cognition_prompt_contract_text.py`: cover L2d prompt contract
  wording for background artifact request boundaries.
- `tests/l2d_action_selection_cases.py`: extend fixture validation and
  comparison support for `background_artifact_request` expectations without
  adding deterministic keyword routing.
- `tests/test_l2d_action_selection_cases.py`: cover deterministic L2d
  materialization and normalization cases.
- `tests/test_l2d_action_selection_live_llm.py`: run one-case-at-a-time local
  LLM checks for the accepted, unsupported, and malformed-output boundaries.
- `tests/test_cognition_resolver_l2d_contract.py`: cover resolver-to-L2d
  action-spec handoff for the new capability.
- `tests/test_l2d_l3_surface_handoff.py`: cover acknowledgement gating from
  pre-surface enqueue results.
- `tests/test_persona_supervisor2_schema.py`: cover the new graph state fields.
- `tests/test_persona_supervisor2.py`: cover graph routing and trace assembly
  with a queued background artifact action.
- `tests/test_service_background_consolidation.py`: cover consolidator overlap
  so operationally owned background jobs do not become duplicate active
  commitments.

### Create

New background artifact module contract and public entrypoints:

- `src/kazusa_ai_chatbot/background_artifact/README.md`: mandatory ICD-style
  module contract for ownership boundaries, public imports, job lifecycle,
  LLM payload limits, persistence, delivery, observability, and forbidden
  paths.
- `src/kazusa_ai_chatbot/background_artifact/__init__.py`: export only the
  public runtime, queue, and delivery functions used by service and action
  execution.
- `src/kazusa_ai_chatbot/background_artifact/runtime.py`: own process-local
  start, stop, and tick orchestration for generation and delivery loops.

New background artifact internals:

- `src/kazusa_ai_chatbot/background_artifact/models.py`: define job status,
  work-kind, result, and delivery TypedDict contracts.
- `src/kazusa_ai_chatbot/background_artifact/jobs.py`: provide semantic queue,
  claim, completion, failure, and delivery-status helpers over the DB facade.
- `src/kazusa_ai_chatbot/background_artifact/prompts.py`: hold the monolithic
  background artifact worker prompt and output contract.
- `src/kazusa_ai_chatbot/background_artifact/worker.py`: run the bounded
  monolithic artifact LLM worker. This file must not import dispatcher or
  adapter delivery modules.
- `src/kazusa_ai_chatbot/background_artifact/delivery.py`: turn completed jobs
  into source-bound cognition/dialog delivery attempts and record terminal
  delivery status.
- `src/kazusa_ai_chatbot/background_artifact/result_source.py`: build the
  prompt-safe `background_artifact_result_ready` cognitive episode from a
  completed or failed job.

New action-spec and DB files:

- `src/kazusa_ai_chatbot/action_spec/handlers/background_artifact.py`: own
  deterministic enqueue handling for the new action capability.
- `src/kazusa_ai_chatbot/db/background_artifact_jobs.py`: own raw MongoDB
  collection access, indexes, leases, idempotency, and status transitions.

New test files:

- `tests/fixtures/l2d_background_artifact_cases.json`: production-style frozen
  L2d route fixtures derived from Stage 0 categories without fixture-specific
  prompt overfitting.
- `tests/test_background_artifact_jobs.py`: focused deterministic job helper
  and state-transition tests.
- `tests/test_background_artifact_worker.py`: patched-LLM worker tests for
  success, unsupported work, failure, and output caps.
- `tests/test_background_artifact_delivery.py`: patched cognition/dialog and
  adapter tests for result-ready source delivery, duplicate suppression, and
  delivery failure.
- `tests/test_background_artifact_runtime.py`: service-runtime start/stop and
  tick orchestration tests.
- `tests/test_background_artifact_worker_live_llm.py`: one-case-at-a-time real
  LLM checks for the monolithic worker prompt.

### Keep

- `src/kazusa_ai_chatbot/dispatcher/handlers.py` remains the dispatcher send
  implementation; do not turn dispatcher into a job queue.
- `src/kazusa_ai_chatbot/dispatcher/task.py` and
  `src/kazusa_ai_chatbot/dispatcher/adapter_iface.py` remain unchanged unless
  the independent plan review finds a concrete adapter-contract blocker.
- `src/kazusa_ai_chatbot/calendar_scheduler/**` remains the typed future
  trigger scheduler; do not use it as the background artifact queue.
- `src/kazusa_ai_chatbot/proactive_output/**` remains outside this feature.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py` remains the dialog renderer;
  pass source-bound anchors through existing L3/dialog contracts instead of
  adding a background-job dialog path.
- `src/kazusa_ai_chatbot/self_cognition/**` remains unchanged; background
  artifact delivery may reuse the same dispatcher delivery boundary but must
  not change self-cognition ownership in this plan.
- Existing adapter protocols remain intact.

## Overdesign Guardrail

- Actual problem: Kazusa needs one reliable way to accept bounded artifact
  work, stop blocking live chat, and deliver the result later without relying
  on consolidator-generated promises.
- Minimal change: add one semantic capability, one durable job owner, one
  worker boundary, one result-ready source path, one pre-surface commit, and
  one consolidation overlap guard.
- Ownership boundaries: L2d owns semantic acceptance; deterministic action
  code owns validation and enqueue; job code owns async work and idempotency;
  dialog owns wording; dispatcher owns adapter delivery; consolidation owns
  memory/audit only.
- Rejected complexity: multiple coding agents, shell execution, repo editing,
  web research, image generation, attachments, chunking, job dashboard,
  calendar-as-job-queue, proactive-output delivery, dynamic worker registry
  exposed to L2d, and Stage 0 fixture overfitting.
- Add rejected complexity only after a separate approved plan with observed
  production need, explicit contracts, and focused tests.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve this plan's contracts.
- The responsible agent must not introduce new architecture, alternate cutover
  strategies, compatibility layers, fallback paths, or extra features.
- The responsible production-code subagent may modify, create, or delete only
  files listed in `Change Surface`. Any unlisted file requires a plan update,
  independent plan review, and explicit user approval before editing.
- The responsible agent must treat changes outside the approved change surface
  as high-scrutiny changes that require plan review before implementation.
- The responsible agent must search for existing helper behavior before adding
  new Python helpers.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors unless listed in
  `Must Do`.
- If the plan and code disagree, preserve the plan intent and report the
  discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

Current executable order:

1. Preserve Stage 0 sign-off and Stage 1+ approval in this active plan.
2. Complete the LLM rewrite/input-audit evidence from the contracts above.
3. Parent adds or updates the focused failing tests first and records expected
   failures or baselines in `Execution Evidence`.
4. Parent starts exactly one production-code subagent with this plan, the
   mandatory skills, the file manifest, the symbol contract, and the focused
   tests.
5. Production-code subagent implements only the approved change surface.
6. Parent owns integration tests, command execution, trace inspection,
   evidence recording, and any plan amendments.
7. Parent runs the focused commands first, then broader regression commands.
8. Parent starts exactly one independent code-review subagent after
   verification.
9. Parent remediates approved review findings inside the approved change
   surface and reruns affected checks.
10. Parent records final execution evidence and asks the user before lifecycle
   completion or archive.

Stage 1+ production sequence:

1. Stage 1A - LLM and test contract:
   add failing/baseline tests for L2d, action-spec validation, pre-surface
   enqueue, result-ready source selection, worker prompt payload, consolidation
   overlap, config, docs, and ops status.
2. Stage 1B - durable job owner:
   create `background_artifact` and `db.background_artifact_jobs`, bootstrap
   indexes, queue helpers, idempotency, leases, worker tick, and runtime tick.
3. Stage 1C - live graph handoff:
   add the action capability, L2d materialization, action execution enqueue,
   pre-surface graph node, L3 acknowledgement gating, and prompt-safe action
   results.
4. Stage 1D - result delivery:
   add result-ready cognitive episode construction, prompt selection,
   source-bound delivery tick, dispatcher handoff, and duplicate delivery
   suppression.
5. Stage 1E - service, docs, and observability:
   wire config, service startup/shutdown, route reporting, ops status,
   event-log aggregate descriptors, README/HOWTO, and subsystem ICD updates.
6. Stage 1F - live/local validation and review:
   run one-case-at-a-time L2d and worker LLM checks, record debug-channel smoke
   status, then independent code review.

## Execution Model

- Current plan cleanup and Stage 1+ review preparation are parent-owned and do
  not authorize production-code edits.
- Stage 0 POC implementation was parent-only because the user explicitly
  prohibited subagents for that stage.
- Future production implementation must use parent-led native subagent
  execution unless the user explicitly approves fallback execution.
- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent owns the LLM rewrite and input-audit gate. The production-code
  subagent receives that reviewed contract and must not invent additional LLM
  prompt, payload, route, or worker-selection changes.
- Production-code subagent owns production code changes only and must receive
  the reviewed plan, mandatory skills, change surface, focused test contract,
  LLM rewrite/audit contract, and production ownership boundary.
- Independent code-review subagent owns review only and must not implement
  fixes.
- If native subagent capability is unavailable during production execution,
  stop before production edits unless the user explicitly requests fallback
  execution.

## Progress Checklist

- [x] Stage 0 POC completed and accepted.
  - Evidence: Stage 0 artifacts and `Execution Evidence`.
  - Sign-off: User accepted the POC result on 2026-06-06.
- [x] Stage 0 conclusion recorded in the active plan.
  - Evidence: this plan records accepted POC evidence, partial failures, and
    non-overfit rule.
- [x] Registry points to the active plan instead of a completed archive record.
  - Evidence: `development_plans/README.md` active short-term row.
- [x] Stage 1+ independent plan review completed.
  - Scope: inspect this plan, Stage 0 artifacts, relevant source, tests, and
    production ownership boundaries.
  - Verify: findings and required edits are recorded in `Execution Evidence`.
- [x] Stage 1+ plan revision completed after review.
  - Scope: validate or revise the `Change Surface` file manifest, then add
    exact symbols, test functions, commands, expected failures or baselines,
    and evidence gates.
  - Verify: no open questions, no production placeholders, and mandatory
    section order remains intact.
- [x] LLM rewrite and input-audit contract recorded.
  - Scope: rewrite every affected LLM contract and audit model-facing input
    types before production-code subagent execution.
  - Verify: `LLM Call And Context Budget` records semantic question, required
    inputs, output fields, deterministic owners, rejected inputs, and
    prompt/load impact.
- [x] Live/local LLM checks completed.
  - Scope: run one case at a time after production prompt/payload edits.
  - Verify: `Execution Evidence` records trace paths and manual route-quality
    inspection for L2d and worker checks.
  - Sign-off: parent verification, 2026-06-06; evidence recorded below with
    residual text-rewrite input-projection risk.
- [x] User approval for production-code execution recorded.
  - Scope: user explicitly authorizes Stage 1+ production implementation.
  - Verify: approval is recorded in `Execution Evidence`.
- [x] Stage 1A focused tests established.
  - Scope: add or update tests listed in `Verification` before production
    implementation.
  - Verify: expected failures or accepted baselines are recorded.
- [x] Stage 1B durable job owner implemented.
  - Scope: background artifact module, DB helpers, bootstrap, leases, runtime,
    and worker tick.
  - Verify: focused background artifact job/runtime/worker tests pass.
  - Sign-off: parent verification, 2026-06-06; evidence recorded below.
- [x] Stage 1C live graph handoff implemented.
  - Scope: action spec, L2d materialization, pre-surface enqueue, schema, L3
    acknowledgement gating, and action results.
  - Verify: focused L2d/action-spec/persona graph tests pass.
  - Sign-off: parent verification, 2026-06-06; evidence recorded below.
- [x] Stage 1D result delivery implemented.
  - Scope: result-ready cognitive episode, prompt selection, delivery tick,
    dispatcher handoff, and duplicate suppression.
  - Verify: focused delivery and prompt-selection tests pass.
  - Sign-off: parent verification, 2026-06-06; evidence recorded below.
- [x] Stage 1E service, docs, and observability implemented.
  - Scope: config, route report, service runtime startup/shutdown, ops status,
    event-log status, README/HOWTO, and ICD docs.
  - Verify: focused config/service/docs/status tests pass.
  - Sign-off: parent verification, 2026-06-06; evidence recorded below.
- [x] Production implementation executed under the reviewed checklist.
  - Scope: run only the production stages added by the Stage 1+ plan revision.
  - Verify: all revised verification commands pass or accepted blockers are
    recorded.
  - Sign-off: parent verification, 2026-06-06; evidence recorded below.
- [x] Independent code review completed after production verification.
  - Scope: full implementation diff, plan alignment, tests, artifacts, and
    residual risks.
  - Verify: findings, fixes, rerun commands, and approval status are recorded.
  - Sign-off: Cicero final re-review, 2026-06-06; no blocking findings remain.

## Verification

Plan-format verification for this update:

- `git status --short` shows only development-plan file changes for this
  plan update.
- `Test-Path -LiteralPath 'development_plans\active\short_term\background_artifact_handoff_poc_plan.md'`
  returns `True`.
- `Test-Path -LiteralPath 'development_plans\archive\completed\short_term\background_artifact_handoff_stage0_poc_record.md'`
  returns `False`; Stage 0 is recorded in this active plan, not as a completed
  archive record.
- `Select-String -LiteralPath 'development_plans\active\short_term\background_artifact_handoff_poc_plan.md' -Pattern '^## '`
  shows the required top-level sections.
- `Select-String -LiteralPath 'development_plans\active\short_term\background_artifact_handoff_poc_plan.md' -Pattern '^### (Delete|Modify|Create|Keep)$'`
  shows all four change-surface manifest groups.
- The prohibited open-question wording scan against the active plan returns no
  matches.
- `Select-String -LiteralPath 'development_plans\README.md' -Pattern 'background_artifact_handoff'`
  shows the active registry row.

Stage 0 completed verification:

- `venv\Scripts\python -m py_compile experiments\__init__.py experiments\background_artifact_handoff\__init__.py experiments\background_artifact_handoff\cases.py experiments\background_artifact_handoff\l2d_prompt_parity_poc.py experiments\background_artifact_handoff\dispatcher_dry_run_poc.py experiments\background_artifact_handoff\report.py`
- `git status --short --ignored -- experiments test_artifacts/background_artifact_handoff/20260606T0338_stage0`

Stage 1A expected failing/baseline tests after the test contract is written:

```powershell
venv\Scripts\python -m pytest tests/test_action_spec_models.py::test_capability_spec_accepts_background_artifact_owner tests/test_action_spec_evaluator.py::test_background_artifact_request_validates_bounded_params tests/test_action_spec_results.py::test_background_artifact_result_ref_projects_prompt_safe_job_ref tests/test_cognitive_episode_contract.py::test_background_artifact_result_ready_builder_creates_valid_episode tests/test_multi_source_cognition_stage_03_prompt_selection.py::test_selector_returns_background_artifact_result_variant_for_every_stage tests/test_l2d_l3_surface_handoff.py::test_background_artifact_acknowledgement_requires_pending_queue_result -q
```

Focused production verification after implementation:

```powershell
venv\Scripts\python -m pytest tests/test_background_artifact_jobs.py tests/test_background_artifact_worker.py tests/test_background_artifact_delivery.py tests/test_background_artifact_runtime.py -q
venv\Scripts\python -m pytest tests/test_config.py tests/test_llm_route_report.py tests/test_db.py::test_db_bootstrap_creates_background_artifact_collection_and_indexes tests/test_service_ops_status.py tests/test_event_logging_status.py tests/test_action_spec_models.py tests/test_action_spec_evaluator.py tests/test_action_spec_results.py tests/test_cognitive_episode_contract.py tests/test_multi_source_cognition_stage_03_prompt_selection.py tests/test_cognition_prompt_contract_text.py tests/test_l2d_action_selection_cases.py tests/test_cognition_resolver_l2d_contract.py tests/test_l2d_l3_surface_handoff.py tests/test_persona_supervisor2_schema.py tests/test_persona_supervisor2.py tests/test_service_background_consolidation.py -q
```

Static boundary verification after implementation:

```powershell
rg "AdapterRegistry|dispatcher|send_message|RemoteHttpAdapter" src/kazusa_ai_chatbot/background_artifact/worker.py
rg "background_artifact" src/kazusa_ai_chatbot/calendar_scheduler src/kazusa_ai_chatbot/proactive_output
rg "BACKGROUND_ARTIFACT_LLM" README.md docs/HOWTO.md src/kazusa_ai_chatbot/config.py src/kazusa_ai_chatbot/llm_route_report.py tests/test_config.py tests/test_llm_route_report.py
Test-Path -LiteralPath 'src\kazusa_ai_chatbot\background_artifact\README.md'
Select-String -LiteralPath 'src\kazusa_ai_chatbot\background_artifact\README.md' -Pattern 'Document Control|Public Interface|Forbidden Paths|LLM Input Contract'
```

Expected static-boundary evidence:

- The first `rg` returns no matches from `background_artifact/worker.py`.
- The second `rg` returns no matches from calendar scheduler or proactive
  output.
- The third `rg` returns matches in route docs, config, route report, and tests.
- The README path exists and contains the mandatory ICD sections.

One-case-at-a-time live/local LLM verification after production prompt edits:

```powershell
$env:L2D_LIVE_CASE_FILE='tests\fixtures\l2d_background_artifact_cases.json'; $env:L2D_LIVE_CASE_ID='coding_snippet_accept_fibonacci'; venv\Scripts\python -m pytest tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -m live_llm
$env:L2D_LIVE_CASE_FILE='tests\fixtures\l2d_background_artifact_cases.json'; $env:L2D_LIVE_CASE_ID='text_rewrite_accept_polish'; venv\Scripts\python -m pytest tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -m live_llm
$env:L2D_LIVE_CASE_FILE='tests\fixtures\l2d_background_artifact_cases.json'; $env:L2D_LIVE_CASE_ID='summary_accept_chat_recap'; venv\Scripts\python -m pytest tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -m live_llm
$env:L2D_LIVE_CASE_FILE='tests\fixtures\l2d_background_artifact_cases.json'; $env:L2D_LIVE_CASE_ID='risky_coding_shell_install_rejected'; venv\Scripts\python -m pytest tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -m live_llm
venv\Scripts\python -m pytest tests/test_background_artifact_worker_live_llm.py::test_background_artifact_worker_live_case -q -m live_llm
```

Debug-channel smoke after focused tests pass:

- UserA asks for a bounded Fibonacci snippet.
- Kazusa acknowledges only after durable enqueue.
- UserB sends unrelated chat while the job is running.
- Kazusa responds normally to UserB.
- Background worker completes the job.
- Kazusa delivers the result to UserA through source-bound cognition/dialog and
  dispatcher.

## Independent Code Review

Run this gate after all approved production verification commands pass and
before final sign-off. The parent agent must create one independent code-review
subagent through the current harness's native subagent capability. If native
subagents are unavailable, stop unless the user explicitly approves fallback
execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and architecture, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including Stage 0 artifacts, focused and
  regression tests, execution evidence, next-stage handoff notes, and path-safe
  commands.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

Current acceptance:

- User accepted Stage 0 POC result as best-attempt feasibility evidence.
- User approved Stage 1+ production implementation on 2026-06-06, bounded by
  this reviewed plan, the LLM audit, and the execution gates above.
- Stage 0 artifacts and report are present.
- Stage 0 sign-off and learnings are recorded in this active plan.
- The active plan registry points to this plan.
- Stage 1+ production implementation has been executed by the designated
  production-code subagent and independently re-reviewed with no blocking
  findings remaining.

Production acceptance after reviewed implementation:

- `/chat` returns after acknowledgement and durable enqueue.
- Other users can continue normal chat while a job runs.
- Completed artifacts are delivered to the original requester/source channel.
- Duplicate enqueue and duplicate delivery are prevented.
- Worker output remains bounded and text-only in first scope.
- First-scope coding remains snippet-only and no-shell.
- `src/kazusa_ai_chatbot/background_artifact/README.md` exists and follows the
  repo's ICD-style README pattern for a new subsystem boundary.
- Consolidation does not create duplicate schedulable active commitments for
  work already owned by a background artifact job.
- L2d behavior is evaluated as best-attempt route quality under production
  prompt pressure, not fixture memorization.

## Data Migration

Production implementation introduces one new collection after approval:

```text
background_artifact_jobs
```

No existing data is migrated into this collection. Existing `future_promises`,
active commitments, calendar runs, self-cognition attempts, conversation
history, and dispatcher outbound rows remain in their existing collections.

Index/bootstrap requirements:

- unique `job_id`;
- unique `idempotency_key`;
- `(status, created_at)`;
- `(lease_expires_at, status)`;
- `(delivery_state, updated_at)`.

## Contracts And Data Shapes

### Background Artifact Public Interface

Existing service and action-spec code may import only the public module
entrypoint:

```python
from kazusa_ai_chatbot.background_artifact import (
    BackgroundArtifactQueueRequest,
    BackgroundArtifactQueueResult,
    BackgroundArtifactRuntimeHandle,
    enqueue_background_artifact_request,
    run_background_artifact_runtime_tick,
    start_background_artifact_runtime,
    stop_background_artifact_runtime,
)
```

Public signatures required by the first production scope:

```python
async def enqueue_background_artifact_request(
    request: BackgroundArtifactQueueRequest,
) -> BackgroundArtifactQueueResult: ...

def start_background_artifact_runtime(
    *,
    is_primary_interaction_busy: Callable[[], bool],
    deliver_result_episode_func: Callable[..., Awaitable[dict[str, object]]] | None = None,
) -> BackgroundArtifactRuntimeHandle: ...

async def stop_background_artifact_runtime(
    handle: BackgroundArtifactRuntimeHandle,
) -> None: ...
```

`background_artifact.worker`, `background_artifact.delivery`,
`background_artifact.prompts`, and `background_artifact.result_source` are
internals. Existing callers must not import them directly outside focused
tests. The queue helper is exposed through the public package entrypoint; DB
row mutation stays behind the `kazusa_ai_chatbot.db` facade.

### Stage 1+ Symbol Contract

Required production symbols and integration points:

- `src/kazusa_ai_chatbot/config.py`: add
  `BACKGROUND_ARTIFACT_LLM_BASE_URL`, `BACKGROUND_ARTIFACT_LLM_API_KEY`,
  `BACKGROUND_ARTIFACT_LLM_MODEL`, `BACKGROUND_ARTIFACT_WORKER_ENABLED`,
  `BACKGROUND_ARTIFACT_WORKER_INTERVAL_SECONDS`,
  `BACKGROUND_ARTIFACT_WORKER_CLAIM_LIMIT`,
  `BACKGROUND_ARTIFACT_WORKER_LEASE_SECONDS`,
  `BACKGROUND_ARTIFACT_WORKER_MAX_ATTEMPTS`,
  `BACKGROUND_ARTIFACT_INPUT_CHAR_LIMIT`, and
  `BACKGROUND_ARTIFACT_OUTPUT_CHAR_LIMIT`.
- `src/kazusa_ai_chatbot/action_spec/registry.py`: add
  `BACKGROUND_ARTIFACT_REQUEST_CAPABILITY` and project it from
  `build_initial_action_capabilities()` and `project_prompt_affordances()`.
- `src/kazusa_ai_chatbot/action_spec/handlers/background_artifact.py`: define
  `validate_background_artifact_action()` and
  `enqueue_background_artifact_action()`.
- `src/kazusa_ai_chatbot/action_spec/execution.py`: extend
  `execute_action_specs_for_trace()` with an optional
  `enqueue_background_artifact_func` test seam and execute only validated
  `background_artifact_request` specs through it.
- `src/kazusa_ai_chatbot/action_spec/results.py`: extend
  `build_action_result()` with `result_refs: list[EvidenceRefV1] | None = None`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`: extend
  `ActionRequestV1`, `ALLOWED_ACTION_CAPABILITIES`,
  `_normalize_action_requests()`, and `_materialize_action_request()` for
  `background_artifact_request` and `work_kind`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: add
  `_pre_surface_action_results_for_state()` and
  `stage_2a_background_artifact_enqueue`; graph order must become
  `stage_2_memory_lifecycle -> stage_2a_background_artifact_enqueue -> route`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: add
  `pre_surface_action_results: NotRequired[list[ActionResultV1]]`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: extend the
  selected text-surface intent builder so L3 can acknowledge only
  `status=pending` and `queue_state=queued` pre-surface results.
- `src/kazusa_ai_chatbot/cognition_episode.py`: add
  `build_background_artifact_result_ready_cognitive_episode()`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`:
  add prompt variant
  `background_artifact_result_ready_background_artifact_result`.
- `src/kazusa_ai_chatbot/db/background_artifact_jobs.py`: define
  `ensure_background_artifact_job_indexes()`,
  `insert_background_artifact_job()`, `claim_background_artifact_job()`,
  `complete_background_artifact_job()`, `fail_background_artifact_job()`,
  `find_deliverable_background_artifact_jobs()`,
  `mark_background_artifact_delivery_in_progress()`,
  `mark_background_artifact_delivered()`, and
  `mark_background_artifact_delivery_failed()`.
- `src/kazusa_ai_chatbot/background_artifact/runtime.py`: define
  `BackgroundArtifactRuntimeHandle`, `start_background_artifact_runtime()`,
  `stop_background_artifact_runtime()`, and
  `run_background_artifact_runtime_tick()`.
- `src/kazusa_ai_chatbot/background_artifact/worker.py`: define
  `run_background_artifact_worker_tick()` and keep it free of dispatcher or
  adapter imports.
- `src/kazusa_ai_chatbot/background_artifact/delivery.py`: define
  `run_background_artifact_delivery_tick()` and keep adapter sending behind the
  existing dispatcher boundary.

### L2d Semantic Request

```json
{
  "capability": "background_artifact_request",
  "decision": "bounded semantic decision",
  "detail": "bounded objective summary",
  "reason": "semantic acceptance reason",
  "work_kind": "coding_snippet | text_rewrite | summary"
}
```

### Trusted Target Scope

Production materialization must build trusted target scope from episode state:

```text
source_platform
source_channel_id
source_channel_type
source_message_id
source_platform_bot_id
source_character_name
requester_global_user_id
requester_platform_user_id
requester_display_name
```

L2d must not author these fields.

### Action Spec Params

```text
work_kind: coding_snippet | text_rewrite | summary
objective: bounded semantic objective string
input_summary: bounded prompt-safe input summary
requested_delivery: send_result_when_done
max_output_chars: 3000
```

### Pre-Surface Result

```text
pre_surface_action_results: [
  {
    action_attempt_id,
    action_kind: background_artifact_request,
    status: pending | rejected | failed,
    queue_state: queued | none,
    work_kind,
    objective_summary,
    operational_owner: background_artifact_job | none,
    job_ref: prompt_safe_job_ref_or_empty,
    acknowledgement_constraint:
      promise_allowed | promise_forbidden_explain_failure
  }
]
```

### Job Row

The production job collection must include at least:

```text
job_id
idempotency_key
source_action_attempt_id
status
work_kind
objective
requester_global_user_id
requester_platform_user_id
requester_display_name
source_platform
source_channel_id
source_channel_type
source_message_id
source_platform_bot_id
source_character_name
lease_owner
lease_expires_at
attempt_count
max_attempts
created_at
updated_at
completed_at
artifact_text
failure_summary
delivery_tracking_id
delivered_conversation_message_id
artifact_char_count
delivery_attempt_count
```

### Result-Ready Source

```text
source_kind: background_artifact_result_ready
job_id
work_kind
request objective
artifact text or bounded artifact summary
failure summary when failed
target requester and source channel binding
delivery_expectation: speak_required
delivery_failure_policy
```

## LLM Call And Context Budget

Stage 0 used the production L2d route and a 362-character prompt delta.

Affected model contracts:

| Contract | Path | Route | Blocking | Before | After |
|---|---|---|---|---|---|
| L2d action selection | live response | `COGNITION_LLM` | yes, existing path | 1 current L2d call | 1 current L2d call, prompt/schema delta only |
| Background artifact worker | background | `BACKGROUND_ARTIFACT_LLM` | no | 0 calls | up to 1 call per claimed job |
| Result-ready delivery cognition/dialog | background delivery | existing cognition/dialog routes | no `/chat` blocking | 0 artifact result source calls | 1 source-bound pass per completed or failed job |
| Consolidation facts overlap guard | background consolidation | existing consolidation route | no `/chat` blocking | current consolidation calls | no new call; prompt/input audit adds job-owned-action evidence |

L2d action-selection contract:

- Semantic question: should Kazusa speak now, persist memory lifecycle work,
  trigger a normal future cognition, and/or accept a bounded async artifact
  request?
- Required inputs: the existing compact action-context string, existing L2d
  upstream cognition residues, and the projected action affordance list.
- Output fields added: `capability="background_artifact_request"` and
  `work_kind` in `coding_snippet | text_rewrite | summary`.
- Deterministic owners: target binding, adapter feasibility, duplicate enqueue,
  validation, enqueue, queue result status, retries, and delivery.
- Rejected inputs: target ids, adapter ids, queue state, worker registry,
  lease/retry state, job ids, filesystem paths, shell affordances, credentials,
  and Stage 0 fixture examples.

Background worker contract:

- Semantic question: generate one bounded text artifact for the queued
  objective, or return an unsupported/failure result.
- Required inputs: `work_kind`, bounded objective, bounded source text or
  summary, `max_output_chars`, and refusal boundaries.
- Output fields: `status`, `artifact_text`, and `failure_summary`.
- Caps: input 8000 chars, artifact output 3000 chars in first scope.
- Deterministic owners: claiming, leases, attempts, status transitions,
  idempotency, output caps, artifact character count, persistence, delivery
  scheduling, and terminal failure.
- Rejected inputs: adapter targets, raw conversation rows, credentials, DB
  internals, job lease internals, shell/filesystem/network affordances,
  repository paths, package-install requests, and dispatch targets.

Result-ready delivery contract:

- Semantic question: how should Kazusa present this completed or failed artifact
  to the original requester in-character while preserving the artifact text?
- Required inputs: prompt-safe result-ready source packet with `job_id`,
  `work_kind`, objective summary, artifact text or bounded artifact summary,
  failure summary when applicable, requester display name, and source-channel
  binding description.
- Output fields: normal cognition/dialog outputs only; no new adapter command
  or job mutation fields.
- Deterministic owners: source-channel binding, dispatcher send, duplicate
  delivery suppression, receipt persistence, and terminal delivery status.
- Rejected inputs: raw platform credentials, adapter ids not needed for model
  wording, queue internals, lease state, retry counters, MongoDB document
  bodies, and alternate delivery targets authored by the model.

Consolidation overlap-guard contract:

- Semantic question: what memory/audit facts are worth keeping from this
  interaction without creating a second operational promise?
- Required inputs: existing consolidation context plus prompt-safe action result
  evidence that the operational owner is `background_artifact_job`.
- Output constraint: no `future_promises` or active commitment owner for the
  same job-owned work.
- Deterministic owners: dedupe against job-owned action evidence and all
  durable job lifecycle state.

Implementation guard:

- No polling cognition loops are allowed.
- The production-code subagent may implement only the LLM prompt and payload
  changes recorded in this section and in later execution evidence.
- Live/local LLM checks must run one case at a time and produce trace artifacts
  for manual route-quality inspection.

## Independent Plan Review

Run this gate before Stage 1 production-code execution.

Review scope:

- Stage 0 artifacts are named, accepted, and carried forward without fixture
  overfitting.
- The LLM rewrite and input-audit gate is complete for every affected LLM
  prompt, payload, and output contract.
- Exact production files, symbols, tests, commands, expected failures or
  baselines, and evidence gates are added before production execution.
- The `Change Surface` manifest has explicit `Delete`, `Modify`, `Create`, and
  `Keep` groups, and production execution edits no file outside those groups.
- Architecture boundaries match adapter/brain/RAG/cognition/dialog/persistence
  ownership.
- LLM-owned semantic judgment and deterministic-code responsibilities stay
  separate.
- L2d remains a top-level semantic router. Detailed worker/subagent selection
  is deterministic background-job dispatch, not an L2d prompt concern.
- No unresolved choices, broad verbs, optional fallbacks, compatibility shims,
  or unowned side paths remain.
- Stage 1+ execution is unblocked only after findings are resolved and user
  approval is recorded.

Record blockers, non-blocking findings, required edits, approval status, and
user approval in `Execution Evidence`.

## Execution Evidence

### 2026-06-06 Stage 0 POC Run

- Executor: Codex parent agent only; no subagents used for Stage 0
  implementation after user instruction.
- Artifact directory:
  `test_artifacts/background_artifact_handoff/20260606T0338_stage0/`.
- Production code changes: none.
- Prompt delta: `background_artifact_request` plus `work_kind`, +362 chars per
  case.
- L2d result: partial but accepted by user as best-attempt POC evidence.
- Dispatcher result: passed dry run.
- User accepted Stage 0 result on 2026-06-06.

Stage 0 route evidence:

- Positive routes passed for `coding_snippet_accept_fibonacci` and
  `text_rewrite_accept_polish`.
- `summary_accept_chat_recap` missed the expected top-level action request
  because the raw output stopped early.
- Missing rewrite input produced `human_clarification`.
- Shell/install risk produced a refusal/alternative path.
- Web research routed to `web_evidence` instead of background work.
- Normal chat and immediate-answer cases did not create a background false
  positive, but showed malformed-output risk under prompt pressure.
- Filesystem-write risk showed safe direction but omitted required `reason`, so
  production normalization dropped the visible action.

Dispatcher dry-run evidence:

- `dispatcher_dry_run.json` recorded `status=succeeded`.
- The dry run recorded one write-ahead row, one fake adapter send, and one
  delivery receipt.
- The second delivery attempt was suppressed as a duplicate by job-layer state.

Stage 0 command evidence:

```powershell
venv\Scripts\python -m experiments.background_artifact_handoff.l2d_prompt_parity_poc --case-id coding_snippet_accept_fibonacci --output-dir test_artifacts/background_artifact_handoff/20260606T0338_stage0
venv\Scripts\python -m experiments.background_artifact_handoff.l2d_prompt_parity_poc --case-id text_rewrite_accept_polish --output-dir test_artifacts/background_artifact_handoff/20260606T0338_stage0
venv\Scripts\python -m experiments.background_artifact_handoff.l2d_prompt_parity_poc --case-id summary_accept_chat_recap --output-dir test_artifacts/background_artifact_handoff/20260606T0338_stage0
venv\Scripts\python -m experiments.background_artifact_handoff.l2d_prompt_parity_poc --case-id normal_chat_joke_no_background --output-dir test_artifacts/background_artifact_handoff/20260606T0338_stage0
venv\Scripts\python -m experiments.background_artifact_handoff.l2d_prompt_parity_poc --case-id immediate_answer_concept_no_background --output-dir test_artifacts/background_artifact_handoff/20260606T0338_stage0
venv\Scripts\python -m experiments.background_artifact_handoff.l2d_prompt_parity_poc --case-id missing_info_rewrite_without_text --output-dir test_artifacts/background_artifact_handoff/20260606T0338_stage0
venv\Scripts\python -m experiments.background_artifact_handoff.l2d_prompt_parity_poc --case-id risky_coding_filesystem_write_rejected --output-dir test_artifacts/background_artifact_handoff/20260606T0338_stage0
venv\Scripts\python -m experiments.background_artifact_handoff.l2d_prompt_parity_poc --case-id risky_coding_shell_install_rejected --output-dir test_artifacts/background_artifact_handoff/20260606T0338_stage0
venv\Scripts\python -m experiments.background_artifact_handoff.l2d_prompt_parity_poc --case-id unsupported_web_research_clarify_or_reject --output-dir test_artifacts/background_artifact_handoff/20260606T0338_stage0
venv\Scripts\python -m experiments.background_artifact_handoff.dispatcher_dry_run_poc --output-dir test_artifacts/background_artifact_handoff/20260606T0338_stage0
venv\Scripts\python -m experiments.background_artifact_handoff.report --output-dir test_artifacts/background_artifact_handoff/20260606T0338_stage0
```

### 2026-06-06 Plan Cleanup

- Restored this plan as the active ongoing document after an incorrect archive
  move.
- Reformatted the active plan to the development-plan mandatory section order.
- Recorded Stage 0 sign-off and learnings inside the active plan.
- Added explicit Stage 1+ independent plan review gate before production-code
  execution.
- Added non-overfit rule so Stage 0 fixture outcomes do not dominate later
  implementation decisions.
- Added an explicit `Delete`/`Modify`/`Create`/`Keep` file manifest and the
  rule that production execution must stop before touching any unlisted file.
- Added mandatory ICD-style README creation for the new
  `background_artifact` subsystem.

### 2026-06-06 Independent Plan Review

- Reviewer: Codex parent agent using the self-review fallback from the
  development-plan procedure; native subagent review was not used because the
  user requested plan review, not delegated subagent work.
- Inputs inspected: `development_plans/README.md`, this active plan,
  `README.md`, `docs/HOWTO.md`, development-plan references, relevant ICD
  READMEs, and source contracts for action results, action specs,
  brain-service ops status, event-log status, service worker startup, and
  existing worker public entrypoints.
- Blocker found and remediated: the production manifest modified
  `brain_service/contracts.py` and `event_logging/status.py` without listing
  `src/kazusa_ai_chatbot/brain_service/README.md` and
  `src/kazusa_ai_chatbot/event_logging/README.md`; both ICD updates are now in
  `Change Surface`.
- Blocker found and remediated: the pre-surface result shape used
  `status=queued`, but existing `ActionResultStatus` supports `pending`, not
  `queued`; the plan now uses `status=pending` plus `queue_state=queued`.
- Blocker found and remediated: the new module public interface was implicit;
  the plan now lists the approved public imports and first-scope public
  signatures.
- Non-blocking finding: reusing the existing internal-thought source variant is
  still feasible, but the `background_artifact_result_ready` source packet is a
  new model-facing use and must be audited in the LLM rewrite/input-audit gate.
- Approval status at that review moment: Stage 1+ production-code execution was
  not yet approved. The next required step was the Stage 1+ plan revision with
  exact symbols, test functions, commands, expected failures or baselines, and
  LLM audit evidence, followed by explicit user approval.

### 2026-06-06 Stage 1+ Approval And Review Remediation

- User approval: the user stated, "I will approve the stage 1+" on 2026-06-06.
- Production code changes in this plan-update step: none.
- Plan status: Stage 1+ production execution is approved only inside this
  reviewed plan, the listed file manifest, the symbol contract, and the
  execution gates.
- Review issue remediated: stale wording still said Stage 1+ needed later
  approval; the summary, rules, checklist, and acceptance criteria now record
  approval and preserve the remaining LLM/test/review gates.
- Review issue remediated: result-ready delivery previously reused
  `internal_thought_internal_monologue`; the plan now requires an explicit
  `background_artifact_result_ready` cognitive episode and prompt variant.
- Review issue remediated: the L2d/action-spec target manifest previously
  implied a new trusted target kind; the plan now keeps `current_user` target
  kind and binds trusted requester/channel data deterministically.
- Review issue remediated: the L2d live-fixture surface previously pointed at
  the Python contract file for cases; the plan now creates
  `tests/fixtures/l2d_background_artifact_cases.json` and keeps
  `tests/l2d_action_selection_cases.py` as validation/comparison support.
- LLM rewrite/input-audit evidence recorded in plan: the affected L2d,
  background worker, result-ready delivery, and consolidation contracts now
  list semantic questions, required inputs, outputs, deterministic owners, and
  rejected inputs in `LLM Call And Context Budget`.
- Remaining incomplete execution gates: focused tests, production
  implementation, live/local LLM traces, debug-channel smoke, independent code
  review, remediation, and lifecycle completion.

### 2026-06-06 Stage 1A Focused Test Contract

- Branch: `feature/background-artifact-handoff-stage1`.
- Production code changes before this gate: none.
- Test files changed by parent:
  `tests/test_action_spec_models.py`,
  `tests/test_action_spec_evaluator.py`,
  `tests/test_action_spec_results.py`,
  `tests/test_cognitive_episode_contract.py`,
  `tests/test_multi_source_cognition_stage_03_prompt_selection.py`, and
  `tests/test_l2d_l3_surface_handoff.py`.
- Command run:

```powershell
venv\Scripts\python -m pytest tests/test_action_spec_models.py::test_capability_spec_accepts_background_artifact_owner tests/test_action_spec_evaluator.py::test_background_artifact_request_validates_bounded_params tests/test_action_spec_results.py::test_background_artifact_result_ref_projects_prompt_safe_job_ref tests/test_cognitive_episode_contract.py::test_background_artifact_result_ready_builder_creates_valid_episode tests/test_multi_source_cognition_stage_03_prompt_selection.py::test_selector_returns_background_artifact_result_variant_for_every_stage tests/test_l2d_l3_surface_handoff.py::test_background_artifact_acknowledgement_requires_pending_queue_result -q
```

- Expected result recorded: 6 failures.
- Failure summary:
  `background_artifact` is not an allowed action owner;
  `background_artifact_request` is not registered/evaluable;
  `build_action_result()` lacks `result_refs`;
  `build_background_artifact_result_ready_cognitive_episode()` does not exist;
  `background_artifact_result_ready` is not an allowed cognition source; and
  L3 text-surface intent does not include queued background artifact
  acknowledgement constraints.
- Stage 1A sign-off: parent agent, 2026-06-06. Next gate is one production-code
  subagent for Stage 1B-1E production implementation inside the approved change
  surface.

Additional Stage 1A test-contract evidence:

- Additional test files changed by parent:
  `tests/test_config.py`, `tests/test_llm_route_report.py`,
  `tests/test_service_ops_status.py`, `tests/test_event_logging_status.py`,
  `tests/test_background_artifact_runtime.py`,
  `tests/test_background_artifact_jobs.py`,
  `tests/test_background_artifact_worker.py`, and
  `tests/test_background_artifact_delivery.py`.
- Command run:

```powershell
venv\Scripts\python -m pytest tests/test_background_artifact_runtime.py tests/test_background_artifact_jobs.py tests/test_background_artifact_worker.py tests/test_background_artifact_delivery.py tests/test_config.py::TestRouteLlmConfig::test_background_artifact_worker_config_values_are_present tests/test_llm_route_report.py::test_llm_route_inventory_contains_all_routes_once tests/test_service_ops_status.py::test_ops_runtime_status_merges_config_and_worker_liveness tests/test_event_logging_status.py::test_build_runtime_status_uses_bounded_latest_events -q
```

- Expected result recorded: 9 failures.
- Failure summary:
  `kazusa_ai_chatbot.background_artifact` and
  `kazusa_ai_chatbot.db.background_artifact_jobs` do not exist;
  the mandatory background artifact ICD README does not exist;
  background artifact worker config constants do not exist;
  `BACKGROUND_ARTIFACT_LLM` is absent from route reporting;
  service ops status has no background artifact worker config/liveness fields;
  and event-log runtime status does not report the background artifact worker
  family.

Additional L2d fixture evidence:

- Additional test files changed by parent:
  `tests/l2d_action_selection_cases.py`,
  `tests/test_l2d_action_selection_cases.py`, and
  `tests/fixtures/l2d_background_artifact_cases.json`.
- Command run:

```powershell
venv\Scripts\python -m pytest tests/test_l2d_action_selection_cases.py::test_compare_accepts_background_artifact_request_params tests/test_l2d_action_selection_cases.py::test_background_artifact_fixture_file_loads -q
```

- Expected result recorded: 1 failure and 1 pass.
- Failure summary:
  the new background artifact fixture file loads and can be selected, but
  action comparison rejects `background_artifact_request` until production
  action registration and evaluator support exist.

Additional DB bootstrap evidence:

- Additional test file changed by parent: `tests/test_db.py`.
- Command run:

```powershell
venv\Scripts\python -m pytest tests/test_db.py::test_db_bootstrap_creates_background_artifact_collection_and_indexes -q
```

- Expected result recorded: 1 failure.
- Failure summary:
  `kazusa_ai_chatbot.db.bootstrap` has no
  `ensure_background_artifact_job_indexes` helper yet, so bootstrap cannot
  delegate durable background artifact job index setup.

Additional graph handoff evidence:

- Additional test file changed by parent: `tests/test_persona_supervisor2.py`.
- Command run:

```powershell
venv\Scripts\python -m pytest tests/test_persona_supervisor2.py::test_background_artifact_executes_before_l3_acknowledgement -q
```

- Current in-progress production result recorded: collection fails before the
  graph assertion because `src\kazusa_ai_chatbot\config.py` reads
  `BACKGROUND_ARTIFACT_LLM_BASE_URL` through direct `os.environ[...]` at import
  time. The evidence was sent back to the production-code subagent because
  production code remains under that single subagent's ownership.
- Intended graph contract:
  background artifact enqueue results must be present as
  `pre_surface_action_results` before `call_l3_text_surface_handler` runs, and
  the final episode trace must include the pending background artifact result.
- Rerun after production subagent updates and parent test isolation:
  the same command passes with 1 test passed, and the test now patches
  pending-resume and memory-lifecycle stages to avoid live DB access.

### 2026-06-06 Stage 1B-1E Production Verification

- Production-code subagent: Faraday
  (`019e9b14-cbe7-7d93-8cfd-855bb468e777`).
- Subagent handoff: implementation stopped on parent request after production
  files, focused test commands, static checks, known unrun live LLM/full-suite
  checks, and residual delivery/RAG risks were reported.
- Parent manifest check:
  changed production/runtime documentation files are inside the approved
  `Change Surface`; parent-owned test and plan files are outside the
  production-code subagent ownership boundary.

Focused background artifact module command:

```powershell
venv\Scripts\python -m pytest tests/test_background_artifact_jobs.py tests/test_background_artifact_worker.py tests/test_background_artifact_delivery.py tests/test_background_artifact_runtime.py -q
```

- Result: 5 passed.

Focused production integration command:

```powershell
venv\Scripts\python -m pytest tests/test_config.py tests/test_llm_route_report.py tests/test_db.py::test_db_bootstrap_creates_background_artifact_collection_and_indexes tests/test_service_ops_status.py tests/test_event_logging_status.py tests/test_action_spec_models.py tests/test_action_spec_evaluator.py tests/test_action_spec_results.py tests/test_cognitive_episode_contract.py tests/test_multi_source_cognition_stage_03_prompt_selection.py tests/test_cognition_prompt_contract_text.py tests/test_l2d_action_selection_cases.py tests/test_cognition_resolver_l2d_contract.py tests/test_l2d_l3_surface_handoff.py tests/test_persona_supervisor2_schema.py tests/test_persona_supervisor2.py tests/test_service_background_consolidation.py -q
```

- Result: 247 passed.

Original Stage 1A focused slice rerun:

```powershell
venv\Scripts\python -m pytest tests/test_action_spec_models.py::test_capability_spec_accepts_background_artifact_owner tests/test_action_spec_evaluator.py::test_background_artifact_request_validates_bounded_params tests/test_action_spec_results.py::test_background_artifact_result_ref_projects_prompt_safe_job_ref tests/test_cognitive_episode_contract.py::test_background_artifact_result_ready_builder_creates_valid_episode tests/test_multi_source_cognition_stage_03_prompt_selection.py::test_selector_returns_background_artifact_result_variant_for_every_stage tests/test_l2d_l3_surface_handoff.py::test_background_artifact_acknowledgement_requires_pending_queue_result -q
```

- Result: 6 passed.

Static and compile checks:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\action_spec\models.py src\kazusa_ai_chatbot\action_spec\registry.py src\kazusa_ai_chatbot\action_spec\evaluator.py src\kazusa_ai_chatbot\action_spec\execution.py src\kazusa_ai_chatbot\action_spec\results.py src\kazusa_ai_chatbot\action_spec\handlers\background_artifact.py src\kazusa_ai_chatbot\background_artifact\__init__.py src\kazusa_ai_chatbot\background_artifact\models.py src\kazusa_ai_chatbot\background_artifact\jobs.py src\kazusa_ai_chatbot\background_artifact\prompts.py src\kazusa_ai_chatbot\background_artifact\worker.py src\kazusa_ai_chatbot\background_artifact\delivery.py src\kazusa_ai_chatbot\background_artifact\runtime.py src\kazusa_ai_chatbot\background_artifact\result_source.py src\kazusa_ai_chatbot\db\background_artifact_jobs.py src\kazusa_ai_chatbot\cognition_episode.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_l3_surface.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\config.py src\kazusa_ai_chatbot\llm_route_report.py src\kazusa_ai_chatbot\event_logging\status.py
Select-String -LiteralPath 'src\kazusa_ai_chatbot\background_artifact\worker.py' -Pattern 'AdapterRegistry|dispatcher|send_message|RemoteHttpAdapter'
Get-ChildItem -LiteralPath 'src\kazusa_ai_chatbot\calendar_scheduler','src\kazusa_ai_chatbot\proactive_output' -Recurse | Select-String -Pattern 'background_artifact'
Select-String -LiteralPath 'README.md','docs\HOWTO.md','src\kazusa_ai_chatbot\config.py','src\kazusa_ai_chatbot\llm_route_report.py','tests\test_config.py','tests\test_llm_route_report.py' -Pattern 'BACKGROUND_ARTIFACT_LLM'
Select-String -LiteralPath 'src\kazusa_ai_chatbot\background_artifact\README.md' -Pattern '^## Document Control|^## Purpose|^## Scope|^## Parties|^## Boundary Summary|^## Public Interface|^## Job Lifecycle|^## LLM Input Contract|^## Forbidden Paths'
Test-Path -LiteralPath 'src\kazusa_ai_chatbot\background_artifact\README.md'
git diff --check
```

- Result:
  `py_compile` passed; forbidden worker import scan returned no matches;
  calendar/proactive scan returned no matches; route/ICD scans found expected
  rows; README path exists; `git diff --check` returned no whitespace errors.

Live/local L2d one-case checks:

```powershell
$env:L2D_LIVE_CASE_FILE='tests\fixtures\l2d_background_artifact_cases.json'; $env:L2D_LIVE_CASE_ID='coding_snippet_accept_fibonacci'; venv\Scripts\python -m pytest tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -m live_llm
$env:L2D_LIVE_CASE_FILE='tests\fixtures\l2d_background_artifact_cases.json'; $env:L2D_LIVE_CASE_ID='text_rewrite_accept_polish'; venv\Scripts\python -m pytest tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -m live_llm
$env:L2D_LIVE_CASE_FILE='tests\fixtures\l2d_background_artifact_cases.json'; $env:L2D_LIVE_CASE_ID='summary_accept_chat_recap'; venv\Scripts\python -m pytest tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -m live_llm
$env:L2D_LIVE_CASE_FILE='tests\fixtures\l2d_background_artifact_cases.json'; $env:L2D_LIVE_CASE_ID='risky_coding_shell_install_rejected'; venv\Scripts\python -m pytest tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -m live_llm
```

- Result:
  `coding_snippet_accept_fibonacci`, `summary_accept_chat_recap`, and
  `risky_coding_shell_install_rejected` passed. `text_rewrite_accept_polish`
  failed route comparison because the model emitted a
  `human_clarification` request and no action specs.
- Trace paths:
  `test_artifacts/llm_traces/l2d_action_selection_live_llm__coding_snippet_accept_fibonacci.json`,
  `test_artifacts/llm_traces/l2d_action_selection_live_llm__text_rewrite_accept_polish.json`,
  `test_artifacts/llm_traces/l2d_action_selection_live_llm__summary_accept_chat_recap.json`,
  and
  `test_artifacts/llm_traces/l2d_action_selection_live_llm__risky_coding_shell_install_rejected.json`.
- Human-readable review:
  `test_artifacts/llm_reviews/background_artifact_l2d_live_review_20260606.md`.
- Live LLM quality note:
  the failed rewrite case is attributed to input projection, not a direct
  prompt-boundary failure. The prompt payload says the user asked to polish a
  supplied short message, but it does not include the concrete source text, so
  the model's clarification request is defensible. Do not over-tune the prompt
  to this fixture; track it as an input-projection risk.
- Live/local LLM gate status:
  not fully signed off. L2d was run one case at a time with 3 passes and 1
  route-quality failure. A background artifact worker live LLM test file from
  the manifest is not present and was not run.

Additional delivery duplicate-suppression test evidence:

- Additional parent-owned tests added:
  `tests/test_background_artifact_jobs.py::test_delivery_in_progress_claim_requires_ready_delivery_state`
  and
  `tests/test_background_artifact_delivery.py::test_service_result_ready_delivery_uses_dispatcher_boundary`.
- Command run:

```powershell
venv\Scripts\python -m pytest tests/test_background_artifact_jobs.py tests/test_background_artifact_delivery.py -q
```

- Result before production fix:
  3 passed and 1 failed.
- Failure summary:
  `mark_background_artifact_delivery_in_progress()` updates by `job_id` only,
  so it can reopen already delivered, active delivery, or otherwise non-ready
  rows. The expected filter is restricted to
  `status in [completed, failed, delivery_failed]` and
  `delivery_state in [ready, failed]`. The failure was sent back to the same
  production-code subagent for a minimal production fix.
- Production-code subagent fix:
  `src\kazusa_ai_chatbot\db\background_artifact_jobs.py` now atomically claims
  delivery only for ready deliverable rows. The production-code subagent
  reported `py_compile` passed and the focused delivery/job command passed.
- Parent rerun after fix:

```powershell
venv\Scripts\python -m pytest tests/test_background_artifact_jobs.py tests/test_background_artifact_delivery.py -q
```

- Result after fix: 4 passed.
- Parent broad rerun after fix:

```powershell
venv\Scripts\python -m pytest tests/test_background_artifact_jobs.py tests/test_background_artifact_worker.py tests/test_background_artifact_delivery.py tests/test_background_artifact_runtime.py tests/test_config.py tests/test_llm_route_report.py tests/test_db.py::test_db_bootstrap_creates_background_artifact_collection_and_indexes tests/test_service_ops_status.py tests/test_event_logging_status.py tests/test_action_spec_models.py tests/test_action_spec_evaluator.py tests/test_action_spec_results.py tests/test_cognitive_episode_contract.py tests/test_multi_source_cognition_stage_03_prompt_selection.py tests/test_cognition_prompt_contract_text.py tests/test_l2d_action_selection_cases.py tests/test_cognition_resolver_l2d_contract.py tests/test_l2d_l3_surface_handoff.py tests/test_persona_supervisor2_schema.py tests/test_persona_supervisor2.py tests/test_service_background_consolidation.py -q
```

- Result after fix: 254 passed.

### 2026-06-06 Independent Code Review Findings

- Independent review subagent: Cicero
  (`019e9b40-5a5b-7f73-b623-4654ea398629`).
- Review scope:
  full implementation diff, approved plan, focused test evidence, LLM traces,
  persistence/delivery boundaries, and changed-file manifest.
- Review outcome:
  findings require remediation before final sign-off; the independent code
  review checklist remains unchecked.
- Blocker:
  result-ready prompt selection can return
  `background_artifact_result_ready_background_artifact_result`, but stage
  prompt-template maps did not register that variant, so completed-job
  delivery could raise `KeyError` before cognition/dialog.
- High-risk finding:
  worker completion and failure transitions updated by `job_id` only, without
  proving the current worker lease owner.
- High-risk finding:
  background artifact enqueue could acknowledge jobs whose trusted target
  scope lacked delivery-critical fields such as `source_channel_id` or
  `requester_global_user_id`.
- Medium-risk finding:
  the failed live L2d `text_rewrite_accept_polish` case is an input-projection
  gap because the model-facing payload did not contain the concrete source
  text to rewrite.
- Medium-risk finding:
  plan/interface drift must be reconciled for the result-delivery runtime
  callback shape, DB indexes, and any modified README files outside the
  approved `Change Surface`.

Review-derived red-test command:

```powershell
venv\Scripts\python -m pytest tests/test_multi_source_cognition_stage_03_prompt_selection.py::test_result_ready_variant_is_registered_in_stage_prompt_maps tests/test_background_artifact_jobs.py::test_worker_completion_requires_current_lease_owner tests/test_background_artifact_jobs.py::test_worker_failure_requires_current_lease_owner tests/test_background_artifact_worker.py::test_worker_threads_lease_owner_when_completing_job tests/test_background_artifact_worker.py::test_worker_threads_lease_owner_when_failing_job tests/test_action_spec_evaluator.py::test_background_artifact_request_rejects_missing_delivery_target_scope -q
```

- Result before production remediation: 7 failed.
- Failure summary:
  stage prompt handlers do not register the result-ready variant; DB
  completion/failure helpers do not accept `lease_owner`; worker completion and
  failure calls do not pass `lease_owner`; and the action evaluator still
  accepts missing delivery target scope for `source_channel_id` and
  `requester_global_user_id`.
- Remediation ownership:
  the same production-code subagent, Faraday
  (`019e9b14-cbe7-7d93-8cfd-855bb468e777`), owns production fixes for these
  findings. Parent owns tests, reruns, evidence, and final review status.

### 2026-06-06 Review Remediation Verification

- Production-code subagent: Faraday
  (`019e9b14-cbe7-7d93-8cfd-855bb468e777`).
- Parent-owned tests added or updated:
  `tests/test_multi_source_cognition_stage_03_prompt_selection.py`,
  `tests/test_background_artifact_jobs.py`,
  `tests/test_background_artifact_worker.py`,
  `tests/test_action_spec_evaluator.py`,
  `tests/test_l2d_action_selection_cases.py`, and
  `tests/test_background_artifact_worker_live_llm.py`.
- Production remediation:
  prompt maps register the result-ready variant; worker completion/failure
  require the current lease owner; the worker threads `worker_id` into leased
  DB updates; and background artifact action validation rejects missing trusted
  delivery target scope before enqueue.
- Manifest remediation:
  `src/kazusa_ai_chatbot/db/README.md`,
  `src/kazusa_ai_chatbot/dispatcher/README.md`, and
  `src/kazusa_ai_chatbot/consolidation/README.md` now have no diff.
- Plan/interface reconciliation:
  this plan now records the implemented `deliver_result_episode_func` callback
  boundary and the actual background artifact DB indexes used by bootstrap.
  It also records the narrower first-scope worker output and job-row contract:
  worker output is `status`, `artifact_text`, and `failure_summary`; job
  `artifact_char_count` is computed deterministically during completion.

Review-derived focused command:

```powershell
venv\Scripts\python -m pytest tests/test_multi_source_cognition_stage_03_prompt_selection.py::test_result_ready_variant_is_registered_in_stage_prompt_maps tests/test_background_artifact_jobs.py::test_worker_completion_requires_current_lease_owner tests/test_background_artifact_jobs.py::test_worker_failure_requires_current_lease_owner tests/test_background_artifact_worker.py::test_worker_threads_lease_owner_when_completing_job tests/test_background_artifact_worker.py::test_worker_threads_lease_owner_when_failing_job tests/test_action_spec_evaluator.py::test_background_artifact_request_rejects_missing_delivery_target_scope -q
```

- Result after remediation: 7 passed.

Focused production verification command:

```powershell
venv\Scripts\python -m pytest tests/test_background_artifact_jobs.py tests/test_background_artifact_worker.py tests/test_background_artifact_delivery.py tests/test_background_artifact_runtime.py tests/test_config.py tests/test_llm_route_report.py tests/test_db.py::test_db_bootstrap_creates_background_artifact_collection_and_indexes tests/test_service_ops_status.py tests/test_event_logging_status.py tests/test_action_spec_models.py tests/test_action_spec_evaluator.py tests/test_action_spec_results.py tests/test_cognitive_episode_contract.py tests/test_multi_source_cognition_stage_03_prompt_selection.py tests/test_cognition_prompt_contract_text.py tests/test_l2d_action_selection_cases.py tests/test_cognition_resolver_l2d_contract.py tests/test_l2d_l3_surface_handoff.py tests/test_persona_supervisor2_schema.py tests/test_persona_supervisor2.py tests/test_service_background_consolidation.py -q
```

- Result after remediation: 261 passed.

Worker live/local LLM command:

```powershell
venv\Scripts\python -m pytest tests/test_background_artifact_worker_live_llm.py::test_background_artifact_worker_live_case -q -m live_llm -s
```

- Result: 1 passed.
- Trace:
  `test_artifacts/llm_traces/background_artifact_worker_live_llm__coding_snippet_fibonacci.json`.
- Human-readable review:
  `test_artifacts/llm_reviews/background_artifact_worker_live_review_20260606.md`.
- Quality note:
  the worker produced a bounded Python Fibonacci snippet from semantic input
  only. The payload excluded adapter ids, channel ids, leases, retries, DB
  fields, filesystem paths, credentials, and delivery mechanics.
- Residual delivery risk:
  if the process crashes after `delivery_in_progress` is claimed and before a
  terminal delivery update, the row may require manual repair or a follow-up
  delivery-lease recovery plan. This plan covers duplicate suppression and
  normal failure retry, not crash recovery for in-progress delivery claims.

Static checks after remediation:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\action_spec\handlers\background_artifact.py src\kazusa_ai_chatbot\background_artifact\worker.py src\kazusa_ai_chatbot\db\background_artifact_jobs.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2c2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py
git diff --check
rg "AdapterRegistry|dispatcher|send_message|RemoteHttpAdapter" src\kazusa_ai_chatbot\background_artifact\worker.py
rg "background_artifact" src\kazusa_ai_chatbot\calendar_scheduler src\kazusa_ai_chatbot\proactive_output
Select-String -LiteralPath 'src\kazusa_ai_chatbot\background_artifact\README.md' -Pattern '^## Document Control|^## Purpose|^## Scope|^## Parties|^## Boundary Summary|^## Public Interface|^## Job Lifecycle|^## LLM Input Contract|^## Forbidden Paths'
```

- Result:
  `py_compile` passed; `git diff --check` reported no whitespace errors and
  only line-ending warnings; worker forbidden import scan returned no matches;
  calendar/proactive scan returned no matches; ICD heading scan found expected
  sections.

Public facade import remediation:

- Review-derived test added:
  `tests/test_background_artifact_runtime.py::test_action_handler_uses_public_background_artifact_entrypoint`.
- Result before production remediation:
  failed because `src/kazusa_ai_chatbot/action_spec/handlers/background_artifact.py`
  imported `kazusa_ai_chatbot.background_artifact.jobs` directly.
- Production-code subagent fix:
  Faraday changed the handler to import `enqueue_background_artifact_request`
  through the public `kazusa_ai_chatbot.background_artifact` facade.
- Parent rerun:

```powershell
venv\Scripts\python -m pytest tests/test_background_artifact_runtime.py::test_action_handler_uses_public_background_artifact_entrypoint tests/test_action_spec_evaluator.py::test_background_artifact_request_validates_bounded_params -q
```

- Result after fix: 2 passed.

Focused production verification after public-facade fix:

```powershell
venv\Scripts\python -m pytest tests/test_background_artifact_jobs.py tests/test_background_artifact_worker.py tests/test_background_artifact_delivery.py tests/test_background_artifact_runtime.py tests/test_config.py tests/test_llm_route_report.py tests/test_db.py::test_db_bootstrap_creates_background_artifact_collection_and_indexes tests/test_service_ops_status.py tests/test_event_logging_status.py tests/test_action_spec_models.py tests/test_action_spec_evaluator.py tests/test_action_spec_results.py tests/test_cognitive_episode_contract.py tests/test_multi_source_cognition_stage_03_prompt_selection.py tests/test_cognition_prompt_contract_text.py tests/test_l2d_action_selection_cases.py tests/test_cognition_resolver_l2d_contract.py tests/test_l2d_l3_surface_handoff.py tests/test_persona_supervisor2_schema.py tests/test_persona_supervisor2.py tests/test_service_background_consolidation.py -q
```

- Result after public-facade fix: 262 passed.

Static checks after public-facade fix:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\action_spec\handlers\background_artifact.py src\kazusa_ai_chatbot\background_artifact\worker.py src\kazusa_ai_chatbot\db\background_artifact_jobs.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2c2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py
git diff --check
rg "background_artifact\.jobs" src\kazusa_ai_chatbot\action_spec\handlers\background_artifact.py
rg "AdapterRegistry|dispatcher|send_message|RemoteHttpAdapter" src\kazusa_ai_chatbot\background_artifact\worker.py
rg "background_artifact" src\kazusa_ai_chatbot\calendar_scheduler src\kazusa_ai_chatbot\proactive_output
git diff --name-only -- src\kazusa_ai_chatbot\db\README.md src\kazusa_ai_chatbot\dispatcher\README.md src\kazusa_ai_chatbot\consolidation\README.md
```

- Result:
  `py_compile` passed; `git diff --check` reported no whitespace errors and
  only line-ending warnings; public-facade forbidden import scan returned no
  matches; worker forbidden import scan returned no matches; calendar/proactive
  scan returned no matches; out-of-manifest README diff check returned no
  files.

Final independent code-review outcome:

- Review subagent: Cicero
  (`019e9b40-5a5b-7f73-b623-4654ea398629`).
- Review result:
  no blocking findings remain under the updated active plan.
- Closed findings:
  result-ready prompt-map registration; worker completion/failure lease-owner
  guards; delivery claim state guard; undeliverable enqueue rejection; public
  facade import boundary; and out-of-manifest README drift.
- Accepted residual risks:
  the L2d `text_rewrite_accept_polish` live failure remains a recorded
  input-projection risk; `delivery_in_progress` crash recovery is out of this
  first scope; worker live LLM evidence covers one `coding_snippet` case only;
  live debug-channel smoke was not run in this turn.
- Final review sign-off:
  Cicero final re-review, 2026-06-06.

## Completion Closure

- Closed by parent agent on 2026-06-06 after user requested plan closure.
- Final implementation verification and independent review evidence are
  recorded in `Execution Evidence`; no blocking review findings remain.
- Accepted residual risks are intentionally not reopened in this completed
  plan. Follow-up work must use a separate active plan.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| L2d overroutes normal chat to background work | Kazusa delays simple/social responses | One root capability, best-attempt LLM checks, no keyword post-routing |
| L2d underroutes artifact work | Async handoff feels unreliable | General capability wording and broader one-case-at-a-time LLM checks |
| False acknowledgement before enqueue | User sees promised work that was not persisted | Pre-surface commit before dialog acknowledgement |
| Duplicate operational ownership | Job queue and active commitments both follow up | Consolidation overlap guard |
| Worker blocks chat service | Other chat waits behind artifact generation | Standalone/background worker and durable queue |
| Result sent to wrong target | Privacy leak or group confusion | Trusted requester/channel binding from episode state |
| Artifact corrupted by dialog | Code or rewrite differs from worker output | Preserve artifact text and bound dialog wrapping |
| Duplicate send after retry | User receives repeated result | Job-layer delivery terminal state before dispatcher |
| Scope creep into real coding execution | Safety and reliability risk | Snippet-only no-shell/no-filesystem first scope |
