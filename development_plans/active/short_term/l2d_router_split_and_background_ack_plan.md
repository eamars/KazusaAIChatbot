# l2d router split and background worker subagent plan

## Summary

- Goal: Reduce L2d overload by moving long-running background work into a
  Web Agent 3 style router and worker-subagent architecture, while preserving
  the visible-handoff invariant for promised background work.
- Plan class: high_risk_migration
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`, `py-style`, `cjk-safety`,
  `test-style-and-execution`
- Overall cutover strategy: migration. New live-turn background requests use a
  generic `background_work` queue; the current text artifact behavior is kept
  as the first worker subagent.
- Highest-risk areas: L2d route regression, accidental live-path latency,
  router/worker responsibility bleed, silent queued work without visible
  handoff, generic job persistence migration, result-ready cognition routing,
  and prompt overfitting to Stage 0 cases.
- Acceptance criteria: L2d routes only at the top level, no LLM both
  routes/classifies and generates semantic parameters or artifacts, background
  work is durably queued before acknowledgement, the first text worker
  preserves existing coding/rewrite/summary behavior, and focused tests plus
  live LLM evidence validate the new contract.

## Context

The completed background artifact handoff POC proved that Kazusa can accept
bounded background text work during a live turn and later re-enter cognition
with the result. The follow-up problem is architectural: the current
`work_kind` path narrows the system around three artifact labels and pressures
L2d to make both route and parameter decisions.

The Stage 0 quality review showed that L2d is already overloaded:

```text
test_artifacts/background_artifact_handoff/20260606T_stage0_input_quality/quality_review.md
```

Observed failures included malformed `speak` actions, invalid goal-progress
fields, resolver/action confusion, and implementation vocabulary leaking into
semantic objectives. The user rejected fixture-specific prompt tuning. The
accepted direction is to correct the ownership model.

The Web Agent 3 ICD provides the local pattern to follow:

```text
router/generator -> executor -> source subagent -> evaluator/finalizer
```

Its router emits only `action`, `source`, and `query`; the executor dispatches
by `source`; the selected source subagent owns source-specific parameter
generation and tools. Background work should follow the same shape with
`worker` replacing `source`.

This plan supersedes the earlier draft's claim that background artifact intake
owns top-level `work_kind`. `work_kind` becomes worker-local to the first
`text_artifact` worker. L2d and the background-work router must not emit it.

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, updating,
  or signing off this plan.
- `local-llm-architecture`: load before changing L2d, background-work routing,
  worker-subagent contracts, graph routing, prompt inputs, or prompt outputs.
- `no-prepost-user-input`: load before changing logic that accepts, rejects,
  persists, or acts on user requests.
- `debug-llm`: load before live/local LLM runs, prompt comparison, or quality
  review artifacts.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files that contain CJK strings.
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
- The `Execution Model` must use parent-led native subagent execution unless
  the user explicitly approves fallback execution.
- Plan status is not production-code authorization. Production edits require
  explicit user approval after this draft is reviewed.
- No LLM may both route/classify and generate semantic parameters, tool
  parameters, or artifacts. This applies at every level, including worker-local
  task classification inside `text_artifact`.
- L2d must not receive a worker registry, DB schema, adapter fields, job
  leases, retry state, delivery mechanics, or worker-local schemas.
- The background-work router may choose only `action`, `worker`, `task`, and
  `reason`. It must not output `work_kind`, worker payloads, tool parameters,
  target ids, platform ids, job ids, adapter names, delivery status, leases,
  retries, DB fields, filesystem paths, final user-visible wording, or worker
  internals.
- Worker subagents may generate semantic parameters and use approved tools only
  inside their own domain. They must not choose the top-level worker route,
  deliver adapter text, call cognition directly, or mutate persistence outside
  their public worker result contract.
- Deterministic code owns validation, persistence, idempotency, adapter
  delivery feasibility, delivery status, permissions, retries, queue leases,
  and invariant enforcement.
- A background-work job must not be durably queued unless the same turn has a
  valid visible handoff route or an explicitly approved no-visible-handoff
  state. This plan's first scope uses the visible handoff route only.
- Live `/chat` must not wait for background worker routing, worker parameter
  generation, artifact generation, web research, code execution, or tool loops.
- Do not tune prompt wording to memorize the nine Stage 0 cases.
- Do not add deterministic keyword routing or post-processing over raw user
  text to rescue LLM route decisions.
- Live/local LLM checks must run one case at a time and be inspected one case
  at a time.

## Must Do

- Split L2d into a top-level semantic route contract that chooses only route
  family and immediate visible-surface need.
- Replace top-level `background_artifact_request` selection with a generic
  background-work route that carries a prompt-safe task brief.
- Add a mandatory ICD-styled README for the new `background_work` module.
- Add a generic durable `background_work` queue so the live turn can enqueue a
  pending work item before worker routing starts.
- Add a Web Agent 3 style background-work router and worker-subagent registry.
- Add the first worker subagent, `text_artifact`, as the monolithic worker that
  preserves current coding-snippet, text-rewrite, and summary behavior.
- Implement `text_artifact` with two separate LLM stages inside the worker
  module: a worker-local task router that classifies the task and a generator
  that produces the artifact or failure summary.
- Keep `coding_snippet`, `text_rewrite`, and `summary` as worker-local
  `text_artifact` task types, not L2d or background-router fields.
- Stop requiring full `resolver_goal_progress` for ordinary social turns that
  do not have a concrete user goal, pending resolver row, resolver observation,
  or background-work job.
- Preserve existing resolver goal progress for real multi-turn goals and
  pending resolver flows.
- Preserve L3/dialog as the only owner of visible wording.
- Preserve completed background-work delivery through source-bound cognition;
  workers must never send adapter messages directly.

## Deferred

- Do not implement repository-editing coding work.
- Do not run shell commands, package installs, tests, downloads, filesystem
  writes, image generation, attachment processing, or chunked delivery.
- Do not add a multi-agent coding swarm in this plan; `text_artifact` is the
  only production worker subagent.
- Do not expose Web Agent 3 web research as an enabled background worker in
  this plan. A `web_research` worker may appear only as a documented future
  extension or disabled stub if tests require registry shape.
- Do not redesign the whole cognition resolver loop.
- Do not remove existing resolver pending HIL or approval behavior.
- Do not add retry prompts, repair prompts, fallback LLM calls, compatibility
  shims, keyword fallback routers, or alternate dispatch paths.
- Do not change adapter protocols, dispatcher delivery contracts, calendar
  scheduler, proactive output, reflection, RAG retrieval ownership, or
  consolidator write routing.
- Do not solve `delivery_in_progress` crash recovery in this plan; it requires
  a separate delivery-lease recovery plan if prioritized.

## Cutover Policy

Overall strategy: migration.

| Area | Policy | Instruction |
|---|---|---|
| L2d output contract | migration | Replace detailed background artifact params with a route-only background-work request. Remove prompt-visible `work_kind` from L2d. |
| Background request persistence | migration | Add `background_work_jobs` as the new durable queue for newly accepted work. Do not enqueue new jobs through L2d directly into `background_artifact_jobs`. |
| Existing text artifact behavior | compatible | Preserve coding-snippet, rewrite, and summary output through the first `text_artifact` worker. |
| Old `background_artifact` package | compatible | Keep existing code only as implementation support or historical compatibility during this plan. It must not remain the top-level routing abstraction. |
| Result-ready cognition | migration | Add background-work result-ready source handling while preserving prompt-safe result delivery boundaries. |
| Visible text surface | compatible | Keep L3/dialog as the only user-visible wording owner. |
| Existing live APIs | compatible | Do not change `/chat`, adapter, dispatcher, or delivery receipt API contracts. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative strategy by default.
- If an area is `migration`, follow the exact migration steps and cleanup gates
  listed in this plan.
- If an area is `compatible`, preserve only the compatibility surfaces
  explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

The live response path becomes:

```text
L1/L2 cognition
  -> L2d top-level router
       route_family:
         speak_now | resolver_needed | private_action_needed |
         background_work_candidate | no_action
       task_brief: short semantic task when route_family is background_work_candidate
       visible_handoff_required: bool
  -> deterministic graph/action layer
       validates visible handoff
       persists generic background_work_job.v1
       returns prompt-safe pending result to L3
  -> L3/dialog visible acknowledgement
```

The background worker path becomes:

```text
background_work_job.v1 queued
  -> background_work router LLM
       action + worker + task only
  -> deterministic provider dispatch
  -> selected worker subagent
       worker-local semantic parameter generation
       approved tool or artifact generation
  -> deterministic job completion/failure
  -> background_work_result_ready cognition source
  -> L3/dialog optional visible result
```

The first production worker is:

```text
text_artifact module
  -> text_artifact task-router LLM
       chooses worker-local task_type only:
       coding_snippet | text_rewrite | summary
  -> deterministic validation
  -> text_artifact generator LLM
       produces one bounded text artifact or failure summary
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| L2d role | L2d becomes a top-level semantic router. | It should judge whether Kazusa is accepting/deferring work, not classify worker internals. |
| Async boundary | Live `/chat` enqueues generic background work before worker routing. | The coding agent or future workers may take minutes and must not block Kazusa's chat interface. |
| Router shape | Background-work router mirrors Web Agent 3 and emits only `action`, `worker`, `task`, and `reason`. | Keeps routing separate from semantic parameter generation. |
| Worker registry | Use auto-discovered worker subagents with `WORKER`, `DESCRIPTION`, and `execute(...)`. | Matches the established Web Agent 3 source-subagent pattern. |
| First worker | Start with one monolithic `text_artifact` worker module containing separate task-router and generator LLM stages. | Preserves current behavior while enforcing the no combined routing/generation rule. |
| `work_kind` | Keep `work_kind` only as `text_artifact` worker-local `task_type` or legacy queue compatibility. | Prevents the narrow enum from controlling top-level architecture. |
| Acknowledgement invariant | Deterministic graph/action code blocks durable enqueue when no valid visible handoff route exists. | Prevents silent operational commitments caused by malformed LLM output. |
| Result delivery | Completed work re-enters cognition as a source-bound result episode. | Preserves character judgment and avoids direct worker-to-adapter sends. |
| Prompt fix style | Rewrite contracts around ownership, not fixture-specific failures. | Avoids overfitting to Stage 0 missing-field failures. |

## Contracts And Data Shapes

### L2d Router Output

L2d returns route intent, not detailed executable action params:

```python
{
    "resolver_capability_requests": list[ResolverCapabilityRequestV1],
    "route_requests": [
        {
            "route_family": (
                "speak_now"
                " | resolver_needed"
                " | private_action_needed"
                " | background_work_candidate"
                " | no_action"
            ),
            "task_brief": str,
            "decision": str,
            "reason": str,
            "visible_handoff_required": bool,
        }
    ],
    "resolver_pending_resolution": ResolverPendingResolutionV1 | None,
    "resolver_goal_progress": ResolverGoalProgressV1 | None,
}
```

`resolver_goal_progress` is present only when the turn has a concrete user
goal, pending resolver state, resolver observations, or an in-progress
deliverable. Ordinary social chat may omit it.

### Background Work Queue Request

The live path persists a generic prompt-safe work item:

```python
{
    "action_attempt_id": str,
    "idempotency_key": str,
    "task_brief": str,
    "source_platform": str,
    "source_channel_id": str,
    "source_channel_type": str,
    "source_message_id": str,
    "source_platform_bot_id": str,
    "source_character_name": str,
    "requester_global_user_id": str,
    "requester_platform_user_id": str,
    "requester_display_name": str,
    "requested_delivery": "send_result_when_done",
    "max_output_chars": int,
    "storage_timestamp_utc": str,
}
```

The queue request must not contain worker-local task type, tool parameters,
filesystem paths, web URLs extracted as tool inputs, adapter delivery handles,
lease state, retry state, or final user-visible wording.

### Background Work Router Output

The background-work router returns only:

```python
{
    "action": "execute | reject | needs_user_input | stop",
    "worker": "text_artifact | none",
    "task": str,
    "reason": str,
}
```

The router dispatches by worker. It must not emit `work_kind`,
`coding_snippet`, `summary`, repository paths, shell commands, web-search
queries, tool arguments, DB ids, adapter ids, or delivery decisions.

### Text Artifact Task Router Output

After the background-work router selects `worker = "text_artifact"`, the
worker-local task router returns only:

```python
{
    "task_type": (
        "coding_snippet"
        " | text_rewrite"
        " | summary"
        " | unsupported"
        " | needs_user_input"
    ),
    "task": str,
    "reason": str,
}
```

The text-artifact task router must not emit artifact text, code, rewritten
content, summaries, repository paths, shell commands, tool parameters, DB ids,
adapter ids, delivery decisions, or final user-visible wording.

### Text Artifact Generator Output

The text-artifact generator consumes a validated task-router result and returns
only:

```python
{
    "status": "succeeded | failed | needs_user_input | rejected",
    "artifact_text": str,
    "failure_summary": str,
    "result_summary": str,
}
```

The text-artifact generator must not choose `worker`, choose `task_type`,
dispatch providers, deliver adapter text, call cognition directly, or mutate
persistence.

### Worker Subagent Interface

Each worker module exposes:

```python
WORKER = "text_artifact"
DESCRIPTION = "..."

async def execute(decision: BackgroundWorkRouterDecision) -> BackgroundWorkResult:
    ...
```

The worker result is prompt-safe:

```python
{
    "status": "succeeded | failed | needs_user_input | rejected",
    "worker": str,
    "artifact_text": str,
    "failure_summary": str,
    "result_summary": str,
    "worker_metadata": dict[str, object],
}
```

For `text_artifact`, `worker_metadata` may include a worker-local
`task_type` value of `coding_snippet`, `text_rewrite`, or `summary`. That value
comes from the text-artifact task router only and is not projected to L2d or
the background-work router.

### Deterministic Background Handoff Invariant

Before enqueue, deterministic code verifies:

```text
L2d selected background_work_candidate
AND valid visible handoff route exists in the same cognition result
AND target scope is complete
AND task_brief is bounded and non-empty
AND requested delivery is supported
```

If the invariant fails, no durable job is created. The graph records a
prompt-safe rejection result for L3 or consolidation.

## LLM Call And Context Budget

Default cap: 50k tokens, estimated as characters divided by four.

| Call | Before | After | Response path | Context policy |
|---|---|---|---|---|
| L2d action initializer | 1 call; route, action params, resolver, goal progress | 1 call; route family, resolver request, pending resolution, limited goal progress | yes | Shorter output contract; no worker registry, no `work_kind`, no job mechanics. |
| Background-work router | none | 0 live calls; 1 background call after job claim | no | Receives task brief, prompt-safe source context, and worker descriptions. Emits route only. |
| Text-artifact task router | none | 0 live calls; 1 background call after `text_artifact` dispatch | no | Receives task, output cap, and semantic source summary. Emits worker-local task type only. |
| Text-artifact generator | background artifact worker call after job claim | 0 live calls; 1 background call after task-router validation | no | Receives validated task type, task, output cap, and semantic source summary. Generates artifact or failure summary only. |

No new live-response LLM call is allowed for ordinary chat, direct answers,
resolver-needed turns, or accepted background-work candidates. The only live
work added to `/chat` is deterministic validation and durable enqueue.

## Change Surface

### Delete

- No files are planned for deletion.

### Create

- `src/kazusa_ai_chatbot/background_work/README.md`: mandatory ICD for the
  generic background-work router, queue, worker registry, result source, and
  forbidden paths.
- `src/kazusa_ai_chatbot/background_work/__init__.py`: public exports for the
  queue facade, runtime tick, and worker dispatch entrypoints.
- `src/kazusa_ai_chatbot/background_work/models.py`: typed queue request,
  job document, router decision, worker result, and prompt-safe queue result
  contracts.
- `src/kazusa_ai_chatbot/background_work/jobs.py`: validation and durable
  enqueue facade for `background_work_job.v1`.
- `src/kazusa_ai_chatbot/background_work/worker.py`: background runtime tick
  that claims jobs, runs router, dispatches workers, and records completion or
  failure.
- `src/kazusa_ai_chatbot/background_work/router.py`: background-work router
  prompt, LLM call, parser, and normalization.
- `src/kazusa_ai_chatbot/background_work/providers.py`: deterministic
  executor that dispatches by `worker` and `action` only.
- `src/kazusa_ai_chatbot/background_work/subagent/__init__.py`: Web Agent 3
  style auto-discovery and validation for worker modules.
- `src/kazusa_ai_chatbot/background_work/subagent/text_artifact.py`: first
  monolithic worker subagent for bounded code snippets, rewrites, and
  summaries. This module contains separate worker-local task-router and
  generator prompt/LLM stages.
- `src/kazusa_ai_chatbot/background_work/delivery.py`: result-ready cognition
  source handoff for completed or failed background-work jobs.
- `src/kazusa_ai_chatbot/background_work/result_source.py`: prompt-safe
  cognitive-episode source projection for background-work results.
- `src/kazusa_ai_chatbot/db/background_work_jobs.py`: MongoDB facade for the
  generic background-work job collection and indexes.
- `tests/test_background_work_jobs.py`: deterministic queue facade, validation,
  idempotency, claim, completion, failure, and delivery-state tests.
- `tests/test_background_work_router.py`: router parser, normalizer, worker
  registry rendering, and no-parameter-output tests.
- `tests/test_background_work_providers.py`: provider dispatch tests proving
  query/task pass-through and worker-only execution.
- `tests/test_background_work_text_artifact.py`: patched LLM tests for the
  first worker's separate task-router and generator stages, including proof
  that no single stage both emits `task_type` and artifact text.
- `tests/test_background_work_delivery.py`: result-ready cognition source
  projection and prompt-safe delivery tests.
- `tests/test_background_work_router_live_llm.py`: one-case-at-a-time live
  LLM checks for worker routing quality.
- `tests/test_background_work_text_artifact_live_llm.py`: one-case-at-a-time
  live LLM checks for worker-local task routing and generator quality.
- `test_artifacts/llm_reviews/l2d_background_work_router_review_<date>.md`:
  human-readable LLM quality review artifact.

### Modify

- `README.md`: replace background artifact architecture wording with generic
  background work plus first text-artifact worker.
- `docs/HOWTO.md`: document background-work worker config and clarify that
  worker routing/generation is outside `/chat`.
- `src/kazusa_ai_chatbot/db/__init__.py`: export background-work DB facade
  helpers.
- `src/kazusa_ai_chatbot/db/bootstrap.py`: create background-work indexes.
- `src/kazusa_ai_chatbot/config.py`: add `BACKGROUND_WORK_*` worker runtime
  settings and background-work LLM route settings according to the existing
  config pattern. Keep existing `BACKGROUND_ARTIFACT_*` settings only as
  compatibility inputs for legacy text-artifact code during this migration.
- `src/kazusa_ai_chatbot/service.py`: replace background-artifact runtime
  startup, shutdown, runtime-status worker liveness, and result-ready delivery
  entrypoints with background-work equivalents. Preserve the old
  background-artifact delivery path only for compatibility rows explicitly
  listed in this plan.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`: update
  `OpsRuntimeConfigResponse` and runtime-status response typing for
  background-work worker config and compatibility fields.
- `src/kazusa_ai_chatbot/brain_service/README.md`: document background-work
  runtime lifecycle, `/ops/runtime-status` fields, and the compatibility
  boundary for old background-artifact rows.
- `src/kazusa_ai_chatbot/event_logging/status.py`: report
  `background_work.worker` status and error counts in runtime snapshots; keep
  old `background_artifact.worker` aggregation only where compatibility rows
  require it.
- `src/kazusa_ai_chatbot/event_logging/README.md`: document the
  `background_work.worker` event component and runtime-status source counts.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2d.py`: rewrite
  L2d prompt, parser normalization, and materialization boundary so L2d emits
  route requests instead of `background_artifact_request` params.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: wire L2d route output
  to generic background-work enqueue and enforce the no-silent-background-work
  invariant before selected L3 text surface.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: add the minimal
  graph state keys for route requests and background-work pending results.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`:
  add `background_work_result_ready` source projection. Preserve the existing
  `background_artifact_result_ready` projection only for old compatibility
  rows until a separate cleanup plan removes it.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: consume the
  prompt-safe background-work pending/rejection result without exposing worker
  internals.
- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`: adjust validation
  helpers only if full goal progress becomes optional for non-goal turns.
- `src/kazusa_ai_chatbot/action_spec/models.py`: add `background_work` to
  allowed owners and support the route-only `background_work_request` action
  contract used for live handoff materialization.
- `src/kazusa_ai_chatbot/action_spec/registry.py`: replace L2d-visible
  `background_artifact_request` projection with a route-only
  `background_work_request` projection. Keep `background_artifact_request`
  hidden from L2d if legacy execution compatibility still requires it.
- `src/kazusa_ai_chatbot/action_spec/evaluator.py`: align deterministic
  validation with the chosen route-only background-work action shape.
- `src/kazusa_ai_chatbot/action_spec/execution.py`: enqueue generic
  background-work jobs instead of directly enqueueing background artifact jobs
  from L2d-authored params.
- `src/kazusa_ai_chatbot/action_spec/results.py`: expose prompt-safe
  background-work pending and rejection fields.
- `src/kazusa_ai_chatbot/action_spec/handlers/background_artifact.py`: keep as
  hidden legacy/text-artifact support for old `background_artifact_request`
  rows only; it must not be L2d's top-level background route.
- `src/kazusa_ai_chatbot/background_artifact/README.md`: mark the module as
  legacy text-artifact support or historical Stage 1 compatibility, not the
  top-level architecture.
- `src/kazusa_ai_chatbot/background_artifact/prompts.py`: keep the current
  text artifact prompt as a legacy compatibility export. New worker-local
  prompt ownership lives in `background_work.subagent.text_artifact`.
- `src/kazusa_ai_chatbot/background_artifact/worker.py`: keep the old worker
  runtime as legacy compatibility only. New generic queue dispatch and first
  worker execution live under `background_work`.
- `src/kazusa_ai_chatbot/nodes/README.md`: document the L2d route-only
  boundary, background-work queue, router, worker registry, and visible
  handoff invariant.
- `src/kazusa_ai_chatbot/action_spec/README.md`: document the changed
  ownership split between router, background-work queue, worker subagents,
  deterministic validation, and L3.
- `tests/l2d_action_selection_cases.py`: update frozen-case comparison support
  for route-only L2d output.
- `tests/test_l2d_action_selection_cases.py`: add deterministic route
  normalization and materialization tests.
- `tests/test_l2d_action_selection_live_llm.py`: update one-case live LLM
  evidence to inspect route quality rather than detailed work-kind output.
- `tests/test_cognition_prompt_contract_text.py`: cover L2d router prompt
  contract and forbidden implementation vocabulary.
- `tests/test_persona_supervisor2.py`: cover graph wiring from L2d route to
  generic background-work pending result.
- `tests/test_l2d_l3_surface_handoff.py`: cover no visible handoff means no
  background-work enqueue and no later delivery promise.
- `tests/test_action_spec_evaluator.py`: replace old L2d-visible
  background-artifact prompt projection expectations with route-only
  background-work validation and hidden legacy-action coverage.
- `tests/test_background_artifact_runtime.py`: reduce to legacy compatibility
  coverage for the old facade and prove new L2d-routed jobs do not use it.
- `tests/test_background_artifact_worker_live_llm.py`: remove active live LLM
  expectations from the old worker path and migrate text-artifact quality
  coverage to `tests/test_background_work_text_artifact.py`.
- `tests/test_service_ops_status.py`: update runtime-status expectations for
  background-work worker config, liveness, and old background-artifact
  compatibility fields.
- `tests/test_event_logging_status.py`: update worker latest-status and error
  aggregation coverage for `background_work.worker`.
- `tests/test_event_logging_interface.py`: update public event-logging
  runtime-status coverage for the background-work worker component.
- `tests/test_fetch_ops_status_script.py`: update ops-status script fixture
  expectations for background-work runtime fields.
- `tests/test_config.py`: update config coverage for `BACKGROUND_WORK_*`
  settings and retained compatibility settings.
- `tests/test_db.py`: update bootstrap coverage for `background_work_jobs`
  collection and indexes.

### Keep

- Keep `/chat`, adapter, dispatcher, delivery receipt, calendar, proactive
  output, reflection, RAG, and consolidator public contracts unchanged.
- Keep current text artifact output behavior for coding snippets, rewrites,
  and summaries through the new `text_artifact` worker.
- Keep existing `background_artifact_jobs` data readable if existing tests or
  local development jobs require it, but do not create new L2d-routed work in
  that collection after the migration.

## Overdesign Guardrail

- Actual problem: L2d and the current background artifact path mix top-level
  routing with detailed artifact classification, creating a narrow and fragile
  architecture for long-running work.
- Minimal change: add generic durable background-work enqueue, route-only
  background worker selection, and one `text_artifact` worker subagent while
  preserving current text artifact behavior.
- Ownership boundaries: L2d owns top-level character acceptance and route
  family; deterministic code owns queue persistence and handoff invariants; the
  background-work router owns worker selection only; worker subagents own
  worker-local semantic parameters and approved tools; L3/dialog owns visible
  wording.
- Rejected complexity: no coding swarm, no repository mutation, no enabled web
  research worker, no shell/filesystem tools, no retry prompts, no keyword
  routers, no adapter changes, no generic tool marketplace, and no delivery
  crash-recovery mechanism.
- Evidence threshold: add additional workers only after repeated approved use
  cases or live LLM evidence show that `text_artifact` is the wrong owner for
  the requested background work class.

## Agent Autonomy Boundaries

- The responsible agent may choose local helper names only when they preserve
  the contracts in this plan.
- The responsible agent must not introduce new architecture, compatibility
  layers, fallback paths, feature flags, or extra agents beyond this plan.
- The responsible agent must not tune prompts to the Stage 0 fixture nouns.
- The responsible agent must keep changes outside the listed files out of
  scope unless the plan is updated and re-reviewed.
- If implementation discovers that a generic job shape requires a public
  contract not listed here, stop and update the plan before production edits.
- If live LLM output quality is mixed but the semantic architecture is correct,
  record the quality evidence and avoid overfitting.
- If native subagent capability is unavailable during execution, stop before
  production-code implementation unless the user explicitly approves fallback
  execution.

## Implementation Order

1. Parent records the focused test contract before production edits:
   - L2d route-only normalization accepts valid background-work route output;
   - silent background-work enqueue is rejected when no valid visible handoff
     route exists;
   - generic background-work queue accepts a bounded task brief and rejects
     unsupported delivery or empty tasks;
   - background-work router normalization never produces worker-local
     parameters;
   - `text_artifact` task router classifies coding/rewrite/summary without
     generating artifacts;
   - `text_artifact` generator produces artifacts without choosing task type.
2. Parent runs the focused tests and records the expected failures or baseline.
3. Parent starts exactly one production-code subagent for production edits.
4. Production-code subagent implements generic background-work contracts,
   queue, router, provider dispatch, worker registry, service runtime/status
   migration, result-ready delivery, first worker, graph wiring, and invariant
   enforcement inside the approved change surface.
5. Parent adds or updates integration, prompt-contract, and live LLM tests while
   the production subagent works.
6. Parent reruns focused deterministic tests, integration tests, static checks,
   and one-case live LLM checks.
7. Parent starts the independent code-review subagent.
8. Parent remediates review findings only inside the approved change surface,
   reruns affected checks, and records evidence.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only;
  does not edit tests unless the parent explicitly directs it; closes after
  planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 0 - plan review and approval
  - Covers: this draft, registry row, Web Agent 3 alignment, and user approval.
  - Verify: independent plan review records no blockers.
  - Evidence: record review result in `Execution Evidence`.
  - Handoff: next agent starts at Stage 1.
  - Sign-off: `<agent/date>` after user approval.
- [ ] Stage 1 - focused test contract established
  - Covers: L2d route-only parser, no-silent enqueue invariant, generic queue
    contract, router contract, service runtime/status contract, and first
    worker two-stage contract.
  - Verify:
    `venv\Scripts\python -m pytest tests/test_l2d_action_selection_cases.py tests/test_l2d_l3_surface_handoff.py tests/test_background_work_jobs.py tests/test_background_work_router.py tests/test_background_work_text_artifact.py tests/test_service_ops_status.py tests/test_event_logging_status.py -q`
  - Evidence: record expected failures or baseline.
  - Handoff: next agent starts production-code subagent for Stage 2.
  - Sign-off: `<agent/date>`.
- [ ] Stage 2 - generic background-work infrastructure implemented
  - Covers: `background_work` ICD, models, queue facade, DB facade, indexes,
    router contract, provider dispatch, worker registry, service
    startup/shutdown, runtime-status typing, and event-status aggregation.
  - Verify:
    `venv\Scripts\python -m pytest tests/test_background_work_jobs.py tests/test_background_work_router.py tests/test_background_work_providers.py tests/test_service_ops_status.py tests/test_event_logging_status.py tests/test_event_logging_interface.py tests/test_fetch_ops_status_script.py tests/test_config.py tests/test_db.py -q`
  - Evidence: changed production files and focused test result.
  - Handoff: next agent starts Stage 3.
  - Sign-off: `<agent/date>`.
- [ ] Stage 3 - first worker and result-ready delivery implemented
  - Covers: `text_artifact` worker, prompt-safe worker result, result-source
    projection, result-ready cognition handoff, service delivery entrypoint,
    and legacy artifact behavior.
  - Verify:
    `venv\Scripts\python -m pytest tests/test_background_work_text_artifact.py tests/test_background_work_delivery.py tests/test_cognitive_episode_contract.py -q`
  - Evidence: changed production files and focused result.
  - Handoff: next agent starts Stage 4.
  - Sign-off: `<agent/date>`.
- [ ] Stage 4 - L2d and graph wiring implemented
  - Covers: route-only L2d, generic background-work enqueue, no-silent handoff
    invariant, L3 pending/rejection consumption, and optional goal-progress
    narrowing.
  - Verify:
    `venv\Scripts\python -m pytest tests/test_l2d_action_selection_cases.py tests/test_l2d_l3_surface_handoff.py tests/test_persona_supervisor2.py tests/test_action_spec_evaluator.py -q`
  - Evidence: changed production files and focused result.
  - Handoff: next agent starts Stage 5.
  - Sign-off: `<agent/date>`.
- [ ] Stage 5 - prompt-quality and regression verification complete
  - Covers: prompt-contract greps, one-case live LLM checks, focused
    deterministic regression, and human-readable LLM review artifact.
  - Verify: all commands in `Verification`.
  - Evidence: trace paths, review artifact, command outputs, and residual
    risks.
  - Handoff: next agent starts independent code review.
  - Sign-off: `<agent/date>`.
- [ ] Stage 6 - independent code review complete
  - Covers: review of full diff, plan alignment, prompt boundaries, tests, and
    evidence.
  - Verify: review findings resolved and affected checks rerun.
  - Evidence: review subagent identity, findings, fixes, residual risks.
  - Handoff: ready for final lifecycle sign-off.
  - Sign-off: `<agent/date>`.

## Verification

### Deterministic Tests

```powershell
venv\Scripts\python -m pytest tests/test_l2d_action_selection_cases.py tests/test_l2d_l3_surface_handoff.py tests/test_persona_supervisor2.py tests/test_action_spec_evaluator.py tests/test_background_work_jobs.py tests/test_background_work_router.py tests/test_background_work_providers.py tests/test_background_work_text_artifact.py tests/test_background_work_delivery.py tests/test_cognitive_episode_contract.py tests/test_service_ops_status.py tests/test_event_logging_status.py tests/test_event_logging_interface.py tests/test_fetch_ops_status_script.py tests/test_config.py tests/test_db.py -q
```

Expected result after implementation: all selected deterministic tests pass.

### Live LLM Tests

Run one case at a time and inspect output after each run:

```powershell
$env:L2D_LIVE_CASE_FILE='tests\fixtures\l2d_background_artifact_cases.json'; $env:L2D_LIVE_CASE_ID='coding_snippet_accept_fibonacci'; venv\Scripts\python -m pytest tests/test_l2d_action_selection_live_llm.py::test_l2d_live_case_against_frozen_upstream -q -m live_llm -s
venv\Scripts\python -m pytest tests/test_background_work_router_live_llm.py::test_background_work_router_live_case -q -m live_llm -s
venv\Scripts\python -m pytest tests/test_background_work_text_artifact_live_llm.py::test_background_work_text_artifact_live_case -q -m live_llm -s
```

The parent must write a human-readable review artifact under
`test_artifacts/llm_reviews/` before claiming prompt quality.

### Static Checks

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\brain_service\contracts.py src\kazusa_ai_chatbot\event_logging\status.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\background_work\models.py src\kazusa_ai_chatbot\background_work\router.py src\kazusa_ai_chatbot\background_work\worker.py src\kazusa_ai_chatbot\background_work\providers.py src\kazusa_ai_chatbot\background_work\delivery.py src\kazusa_ai_chatbot\background_work\result_source.py src\kazusa_ai_chatbot\background_work\subagent\text_artifact.py
git diff --check
rg "AdapterRegistry|dispatcher|send_message|RemoteHttpAdapter" src\kazusa_ai_chatbot\background_work src\kazusa_ai_chatbot\background_artifact
rg "work_kind|coding_snippet|text_rewrite|summary" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py
rg "首轮 artifact|artifact 阶段|resolver_capability_requests.*background_artifact_request" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2d.py src\kazusa_ai_chatbot\background_work
rg "background_artifact_worker" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\brain_service\contracts.py src\kazusa_ai_chatbot\event_logging\status.py
```

Expected static results: compile succeeds; `git diff --check` has no
whitespace errors; forbidden delivery imports do not appear in background
worker files; the L2d prompt no longer mentions worker-local `work_kind` or
text-artifact task labels; runtime prompts do not contain user-facing
implementation vocabulary such as `首轮 artifact`.
The final `background_artifact_worker` grep may return retained compatibility
fields only when they are documented in the changed README files and tests.

## Independent Plan Review

Run this gate before approval or execution. Prefer a reviewer that did not
draft the plan. If no separate reviewer is available, the drafting agent must
reread the completed background artifact plan, Web Agent 3 ICD, this draft,
relevant source, and the Stage 0 quality review from a fresh-review posture.

Review scope:

- L2d is truly reduced to top-level routing.
- Background-work router chooses workers only and never worker-local params.
- Worker subagents own semantic parameters only inside their domain.
- Worker-local LLM stages do not combine classification with artifact or tool
  parameter generation.
- The generic durable queue lets `/chat` acknowledge without waiting for worker
  routing or long-running worker execution.
- Service startup, shutdown, `/ops/runtime-status`, event-status aggregation,
  and result-ready delivery surfaces are included in the change surface.
- The no-silent-enqueue bug is covered by deterministic tests.
- Prompt changes address general contract logic, not fixture-specific wording.
- File manifest is complete and does not authorize unrelated graph, adapter,
  dispatcher, calendar, reflection, RAG, or consolidator refactors.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, prompt,
  documentation, test, and command artifact.
- L2d prompt ownership, input audit, output schema, and local LLM context
  budget.
- Background-work router public boundary, worker registry, and no combined
  route/classification plus parameter/artifact-generation rule.
- Worker-subagent result contract and forbidden runtime fields.
- Service lifecycle, runtime-status typing, event-status aggregation, and
  result-ready delivery migration.
- Deterministic no-silent-enqueue invariant and durable queue idempotency.
- Regression coverage and live LLM evidence quality.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- L2d no longer owns background artifact `work_kind`, detailed objective
  shaping, or worker-local task classification.
- The live path durably queues generic background work before L3 acknowledges
  the promise.
- The background-work router emits only `action`, `worker`, `task`, and
  `reason`.
- Worker subagents own semantic parameter generation and approved tool/artifact
  work within their own domain.
- The first `text_artifact` worker preserves current coding-snippet, rewrite,
  and summary behavior without exposing those labels to L2d or the
  background-work router, and without using one LLM call to both classify task
  type and generate artifact text.
- Service startup/shutdown, runtime status, event-status aggregation, and
  result-ready delivery use the generic background-work abstraction for new
  work.
- Ordinary social chat is not forced into full resolver goal progress.
- No durable background-work job is enqueued when the current turn lacks a
  valid visible handoff route.
- Deterministic tests, static checks, and one-case-at-a-time live LLM reviews
  pass or have accepted documented residual risks.
- Independent code review reports no blocking findings.

## Execution Evidence

- 2026-06-06 plan review: identified three blocking plan issues before
  Stage 1 execution.
  - The plan omitted service startup, shutdown, `/ops/runtime-status`,
    event-status aggregation, and result-ready delivery files even though the
    current implementation owns background worker liveness and delivery there.
    Fixed by adding `service.py`, `brain_service/contracts.py`,
    `brain_service/README.md`, `event_logging/status.py`,
    `event_logging/README.md`, and affected tests to the change surface and
    verification gates.
  - The first `text_artifact` worker still allowed one LLM stage to classify
    task type and generate artifact content. Fixed by requiring separate
    worker-local task-router and generator LLM stages inside the same worker
    module.
  - The plan class was too small for a durable queue plus runtime behavior
    migration. Fixed by changing the class to `high_risk_migration`.
- 2026-06-06 post-review hygiene checks: placeholder/open-choice scan returned
  no matches; plan length is 894 lines, within the `high_risk_migration`
  contract; `git diff --check` returned no whitespace errors and only CRLF
  working-copy warnings for the two changed plan files.
- This plan remains draft and is not approved for production edits.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| L2d route regression | Route-only focused tests and live LLM review | `test_l2d_action_selection_live_llm.py` |
| Background router starts generating params | Strict output contract and parser tests | `test_background_work_router.py` |
| Text-artifact worker combines classification and generation | Separate task-router and generator contracts with patched and live LLM tests | `test_background_work_text_artifact.py` |
| Worker registry becomes a hidden tool marketplace | One enabled worker, deferred future workers, registry tests | `test_background_work_providers.py` |
| Extra response-path latency | Router and worker calls run only after background job claim | LLM budget review and graph tests |
| Silent background job | Deterministic invariant blocks enqueue | `test_l2d_l3_surface_handoff.py` |
| Worker runtime not actually migrated | Service lifecycle, ops status, event status, and config tests are in scope | `test_service_ops_status.py` |
| Goal continuity loss | Keep full progress for resolver goals only | Resolver and L3 handoff tests |
| Persistence migration drift | New queue facade and DB tests | `test_background_work_jobs.py` |
| Legacy artifact path confusion | Docs and greps prove L2d no longer routes by `work_kind` | static checks |
