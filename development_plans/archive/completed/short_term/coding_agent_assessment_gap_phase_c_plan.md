# coding_agent_assessment_gap_phase_c_plan

## Summary

- Goal: Close the outer coding-run loop by persisting prompt-safe run
  affordances across accepted-task delivery, giving blockers exact resume
  semantics, preserving user-message approval evidence, serializing run/source
  operations, and establishing a reproducible benchmark seam.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `development-plan`, `py-style`,
  `test-style-and-execution`, `local-llm-architecture`,
  `no-prepost-user-input`
- Overall cutover strategy: Big-bang the coding-run worker/action contract to a
  prompt-safe `coding_run_context.v1`, persist that semantic context with the
  accepted-task lifecycle, enforce the same state machine at L2d materialization
  and worker execution, then add kernel-backed workspace locks and a test-only
  benchmark harness.
- Highest-risk areas: L2d affordance drift, ambiguous blocker semantics,
  approval evidence leakage, and concurrency races.
- Acceptance criteria: Typed blockers resume through their stored target on the
  same `coding_run_ref`; L2d receives current run-specific actions on result and
  later user turns; stale/unavailable actions are rejected; approval events
  retain exact bounded message provenance; same-run/source mutations serialize;
  a 30-case manifest and representative benchmark smoke results are reproducible.

## Context

After Phase A and B, the coding agent has stronger proposal and execution
loops, but the outer cognition seam still treats many coding results as
narration input. The assessment correctly flags that `allowed_next_actions` is
informational, blockers are not a typed user-resolution channel,
`approve_and_verify` approval is fabricated from L2d's semantic decision
without durable quote/message evidence, and concurrent jobs against the same
source can race longer-running verification loops.

This phase makes the accepted-task/coding-run interface self-contained for
follow-up behavior without exposing coding internals to the persona layer.

This plan depends on Phase B's typed blocker contract. Phase C must not
reclassify execution failures or invent blockers; it must make already-typed
blockers answerable through the same accepted-task/coding-run loop.

The current implementation has a missing middle that the previous draft did
not name:

- the coding worker stores `allowed_next_actions` and blockers in
  `worker_metadata`, but `background_work/result_source.py` does not project
  worker metadata into the accepted-task cognition episode;
- `cognition_episode.py` and `prompt_selection.py` expose only task/result/
  failure summaries, and the L2d payload has no coding-run section;
- after result delivery, the accepted-task row does not retain a prompt-safe
  coding-run context that a later user turn can load;
- `continue_coding_run(...)` derives actions from status in one module while
  action-spec, jobs, worker, materialization, and prompts each repeat their own
  action vocabulary;
- `respond_to_blocker` has no defined way to distinguish a proposal answer
  from an environment retry or a non-resumable safety blocker;
- ledger writes and event sequence allocation are atomic per file replacement
  but are not protected across concurrent read-modify-write operations.

This phase closes those data and ownership seams. `accepted_task` owns the
prompt-safe durable follow-up binding; `coding_run` remains the authoritative
state machine; cognition receives only semantic run refs, statuses, allowed
actions, and blocker questions.

### Assessment Coverage And Handoff

| Assessment finding | Phase C disposition |
|---|---|
| Typed blockers are narration-only | Close with durable prompt-safe context and `respond_to_blocker`. |
| `allowed_next_actions` is informational | Close with L2d projection, materialization checks, and live-ledger revalidation. |
| Approval is fabricated without message evidence | Close with trusted current-message provenance and no fallback fabrication. |
| Same-run/source jobs can race | Close for the current multi-process workspace through ordered OS file locks. |
| Model/time tradeoff is not measurable | Close the measurement seam with a 30-case manifest and per-case artifacts. |
| Continuous interactive IO | Reserve through events and continuation actions; do not build a live session. |
| Generic JSON-action inner loop | Defer to a separate benchmark-gated Phase D plan behind the same run API. |
| Breadth features from assessment Phase 4 | Defer to a separate Phase E breadth plan. |

## Mandatory Skills

- `development-plan`: load before changing this plan or executing it.
- `py-style`: load before editing Python production code.
- `test-style-and-execution`: load before changing or running tests.
- `local-llm-architecture`: load before prompt, L2d, action-spec, worker, or
  coding-run contract changes.
- `no-prepost-user-input`: load before changing approval, blocker-answer, or
  user follow-up interpretation.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan
  before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the Independent Code Review gate and record the result in Execution Evidence.
- Use parent-led native subagent execution unless the user explicitly approves
  fallback execution.
- Keep L2d inputs semantic and prompt-safe. Worker-local fields, execution
  specs, paths, shell commands, and approval internals must not become model
  selectable action params.
- Deterministic code validates run refs, allowed actions, approval evidence,
  source locks, and benchmark result schemas.
- Use deterministic tests for schemas, validators, locking, repository queries,
  and materialization; use patched LLM tests for cross-layer handoff; use live
  LLM tests only for model-facing behavior. Run each live case separately,
  inspect its durable trace, and judge behavior rather than relying on pytest
  status alone.
- Let L2d decide whether the user's message means approve, cancel, revise, or
  answer a blocker. Deterministic code binds that semantic decision to current
  run state and validates provenance; it must not keyword-classify the message
  or rewrite the decision.
- Keep one canonical `allowed_next_actions` function in `coding_run`; L2d,
  accepted-task projection, action materialization, and the worker consume its
  public result and must not recreate status/action tables.
- Bind a continuation only to a `coding_run_ref` offered in the current trusted
  `CodingRunContextV1` set. Remove raw-user-text run-ref scanning. An omitted
  ref binds only when exactly one offered context permits the selected action;
  zero or multiple eligible contexts enqueue no coding job.
- Project at most three unique unresolved coding-run contexts for the trusted
  requester/channel. If more than one run can satisfy a ref-less continuation,
  L2d must ask which run; deterministic materialization must not guess. Carry
  a prompt-safe ambiguity summary to L3 so the visible reply identifies the
  distinct objective summaries and asks the user to select one.
- Acquire mutation locks in sorted key order and hold the per-run lock across
  ledger load, state validation, specialist work, event append, and ledger
  write. A lock timeout leaves the ledger unchanged and returns a retryable
  `operation_outcome=busy` projection.
- For `start`, allocate the run id before the first ledger write, derive the
  immutable normalized source key from the validated request, then acquire the
  sorted run/source key set before creating the ledger or starting specialist
  work. For a continuation, derive the immutable source key from a read-only
  ledger read, acquire the sorted run/source key set, then reload and validate
  the ledger inside the lock before any transition.
- Anti-cheat rule: tests must prove the integrated L2d-to-background-worker path
  honors run-specific affordances. Do not bypass L2d by directly calling
  `continue_coding_run` for the integration assertions.
- Anti-cheat rule: blocker-response tests must start from a real or
  deterministic typed blocker in a durable coding run. Do not fabricate a
  successful continuation by directly editing the ledger or bypassing worker
  payload validation.

## Must Do

- Add canonical typed blocker fields to `CodingRunBlocker` and public
  projections.
- Add `respond_to_blocker` as a coding-run continuation action and an
  `accepted_coding_task_request` decision.
- Enforce one active unresolved blocker per run. Store a private blocker id and
  public `resume_target`: `replan_proposal`, `retry_verification`, or `none`.
- Route `respond_to_blocker` to the current active blocker on the referenced
  run. `replan_proposal` passes the original goal, blocker question, and exact
  user answer into the existing writing/modifying LLM path;
  `retry_verification` reruns the stored Phase B execution plan without source
  repair; `none` never exposes `respond_to_blocker`.
- Preserve Phase B environment blockers as blockers until the user supplies an
  actionable answer through `respond_to_blocker`; the response path must not
  rerun repair or execution before the blocker answer has been validated.
- When an environment blocker is answered, treat L2d's
  `respond_to_blocker` selection as the semantic decision to retry the stored
  plan. If the dependency remains missing, replace the active blocker with a
  new environment-blocker event and spend zero repair calls.
- Add `CodingRunContextV1` as the only prompt-safe cross-layer run projection:
  run ref, public status, objective summary, allowed next actions, one active
  blocker question/options, and `followup_open`. Exclude paths, diffs,
  execution specs/output, approvals, locks, and ledger events.
- Persist `CodingRunContextV1` on the accepted-task row whenever a coding worker
  completes. On result-ready cognition use the current job's sanitized context;
  on later `user_message` turns load the newest context per run for the trusted
  requester/channel, collapse older rows by run ref, and project at most three
  rows whose latest `followup_open` is true.
- Add an indexed accepted-task repository query for that loader. Do not query
  raw background-work jobs from cognition and do not expose worker metadata.
- Add a dedicated `coding_runs` section to the L2d human payload. Keep stable
  decision semantics in the L2d system prompt; current run refs/status/actions/
  blockers remain dynamic human-message facts.
- Carry a separate prompt-safe `coding_run_followup` L3 source field for the
  current result or later user turn. It contains only `none`, `single`, or
  `ambiguous` mode, objective summaries, and active blocker question/options;
  it excludes run refs and every operational field. L3 uses it only to render
  the already-determined blocker question or ambiguity clarification.
- Make run-specific `allowed_next_actions` authoritative at three gates:
  L2d sees the list; deterministic materialization accepts only a listed
  decision and resolves an omitted ref only when exactly one eligible context
  exists; the coding-run worker reloads the ledger and revalidates before work.
- Add accepted-task integration tests proving L2d can choose
  `respond_to_blocker` only when the run projection exposes that allowed action,
  and cannot approve, revise, or cancel when those actions are absent.
- Carry approval evidence from the triggering user message into the worker
  payload, approval object, ledger event, and public-safe projection:
  bounded quote, source message id, source trigger, requester id, and timestamp.
- Add per-run and per-source-identity locking for coding-run worker execution.
- Implement locks under `<workspace_root>/.locks/coding_agent/` with hashed
  public-safe keys and standard-library nonblocking kernel locks (`msvcrt` on
  Windows, `fcntl` on POSIX). Acquire keys in sorted order, release in `finally`,
  and rely on kernel release after process exit rather than stale lease files.
- Use `run:<run_id>` for every mutation and a canonical source key derived from
  normalized explicit source fields for start, then from persisted repository
  identity for continuations. Source-free runs use only the run key.
- Add `operation_outcome="applied | busy | rejected"` to run responses. Lock
  timeout returns `busy`, preserves the stored run status and allowed actions,
  appends no event, and lets the user retry the same semantic action.
- Add `scripts/run_coding_agent_benchmark.py`, a 30-case versioned manifest,
  deterministic schema tests, and per-case invocation through the public
  coding-run or accepted-coding-task seam. Record route/model config, engine id,
  final state, hidden evaluator result, elapsed time, LLM-call count and token
  usage when available, trace paths, and limitations.
- Count benchmark LLM calls in the test harness by wrapping
  `LLInterface.ainvoke(...)` for the benchmark context only. Keep measurement
  out of production routing and do not make missing provider token usage fail a
  case.
- Add deterministic and live integration tests for blocker response,
  approval-evidence recording, allowed-action affordance binding, and locking.
- Add five committed, separately inspected `live_llm` gates with raw/reviewed
  evidence: same-run blocker answer, ambiguous-run selection, approval
  provenance, result-ready blocker delivery, and mixed create/edit binding.

## Deferred

- Do not add a live interactive CLI, web console surface, arbitrary shell,
  dependency installation, git branch/PR publishing, or JSON-action loop.
- Do not expose worker-local execution specs or raw ledger internals to L2d.
- Do not make benchmark results a production runtime dependency.
- Do not broaden Phase B's environment classifier or add installation behavior
  in this phase.
- Do not add deterministic keyword parsing for approval, blocker answers, or
  run selection.
- Do not add a second coding-run state store, query raw worker metadata from
  cognition, or preserve both v1 and v2 worker payloads after cutover.
- Do not implement cross-repository alias resolution for source locks. The lock
  identity covers equal normalized source requests and equal persisted source
  identities; broader checkout coordination requires separate evidence.

## Cutover Policy

Overall strategy: bigbang for coding-run actions and worker payloads; additive
schema evolution for accepted-task prompt-safe context.

| Area | Policy | Instruction |
|---|---|---|
| Coding actions | bigbang | Add `respond_to_blocker` to one canonical coding-run action/state table and rewrite all consumers in the same change. |
| Worker payload | bigbang | Replace `coding_agent_worker_payload.v1` with v2. Drain queued v1 coding jobs before deployment; do not dual-read or translate them. |
| Worker metadata | bigbang | Replace v2 with `coding_agent_worker_metadata.v3` containing `coding_run_context.v1`. Keep raw operational rows audit-only. |
| Accepted-task rows | compatible | Add optional `coding_run_context`. Existing rows require no backfill and project no run context when absent. |
| L2d input | bigbang | Add one dedicated coding-runs section and remove generic decision prose that conflicts with run-specific actions. |
| Approval evidence | bigbang | Reject new approval actions without trusted current-user-message evidence. Do not fall back to `current_user` or worker time. |
| Locks | additive | Add ordered kernel-backed locks around coding-run mutations; reads remain public projections. |
| Benchmark | additive test-only | Add manifest, runner, evaluator, and artifacts without a production scheduler or dependency. |

Existing `revise_proposal` remains for ordinary proposal change.
`respond_to_blocker` is the only canonical action for answering an active typed
blocker. Any cutover-policy change requires user approval.

## Cutover Policy Enforcement

- Follow each selected policy. For every bigbang area, rewrite or delete legacy
  runtime consumers; do not add dual reads, translators, aliases, or fallbacks.
- Drain queued v1 jobs and verify no runtime v1/v2 contract references before
  deployment. Preserve only the stated optional absent-field behavior for
  accepted tasks; do not backfill or create a parallel row shape.
- Any change to a cutover policy requires user approval before implementation.

## Data Migration

- Keep `accepted_task.v1`; add optional `coding_run_context` without backfill,
  so older rows project no context. Add the named index through bootstrap and
  prove idempotence with MongoDB enabled.
- Before v2 deploy, drain or expire v1 jobs, record queue count/time in Execution Evidence, and remove all v1 reads/translations.

## Target State

A coding run can ask a typed question or report an environment/scope blocker,
and the next user answer can continue the same durable run through
`respond_to_blocker` according to the blocker's stored resume target. The
prompt-safe context survives result delivery and is reloadable on later user
turns by trusted requester/channel scope. L2d receives up to three current run
contexts and cannot approve, answer, cancel, or revise outside each run's
allowed actions. Approval records show the exact bounded current user message,
message id, requester, trigger, and timestamp that caused approval.

Concurrent mutations for the same run or normalized source serialize across
Kazusa processes that share the coding workspace. Lock contention returns a
non-mutating busy outcome. A versioned 30-case benchmark manifest can be run one
case at a time at the public seam, with repeatable result artifacts suitable
for later pipeline-versus-action-loop and model-swap comparisons.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Blocker action | Add `respond_to_blocker` | It separates missing-information answers from proposal revision. |
| Blocker resume | Store a deterministic `resume_target` with each blocker | Coding-run orchestration must not infer how an answer resumes work from user prose. |
| Blocker source | Consume typed blockers from coding-run state | Classification belongs to Phase B verification or the owning PM, not L2d. |
| Semantic context owner | Persist `CodingRunContextV1` on accepted tasks | Accepted task is the persona-facing lifecycle; raw worker jobs remain operational. |
| Context cardinality | Project at most three newest unique open runs | Bounded local-model context with explicit ambiguity behavior. |
| L2d affordance source | Use coding-run public state, not generic capability prose | The run state owns what actions are legal. |
| Enforcement | Check at L2d materialization and live worker continuation | Prevent prompt drift and stale queued actions. |
| Approval evidence | Store bounded quote plus source ids | Keeps semantic approval auditable without adding UI approval. |
| Approval quote source | Current `dialog_text` percept, never decontextualized text | The ledger must preserve what the user actually said. |
| Lock owner | Coding-run workspace with OS file locks | Protects file-ledger read-modify-write across processes without a new dependency. |
| Lock contention | Return non-mutating `operation_outcome=busy` | Contention is retryable operational state, not a user blocker. |
| Benchmark seam | Drive public coding-run or accepted-task entrypoints | Measures the same behavior users exercise. |
| Benchmark corpus | Versioned 30-case manifest; invoke one live case at a time | Makes comparisons repeatable while preserving live-LLM inspection rules. |

## Contracts And Data Shapes

Typed blocker:

```python
{
    "blocker_id": str,
    "code": str,
    "blocker_kind": "needs_user_input | environment | scope | safety",
    "message": str,
    "question": str,
    "options": list[str],
    "resume_target": "replan_proposal | retry_verification | none",
    "status": "open | answered | superseded",
    "details": dict[str, object],
    "created_at": str,
    "answered_at": str | None,
}
```

Only the active open blocker is projected. `blocker_id`, details, timestamps,
and response text remain ledger/event fields; L2d receives kind, question,
options, and whether `respond_to_blocker` is allowed.

`respond_to_blocker` action payload:

```python
{
    "operation": "respond_to_blocker",
    "coding_run_ref": "coding_run:<run_id>",
    "task_brief": "user's answer",
}
```

Prompt-safe coding-run context:

```python
{
    "schema_version": "coding_run_context.v1",
    "coding_run_ref": "coding_run:<run_id>",
    "status": str,
    "objective_summary": str,
    "allowed_next_actions": list[str],
    "active_blocker": {
        "blocker_kind": str,
        "question": str,
        "options": list[str],
    } | None,
    "followup_open": bool,
    "updated_at": str,
}
```

`allowed_next_actions` is generated only by `coding_run`. A blocked run exposes
`respond_to_blocker` only when the active blocker's `resume_target` is not
`none`; it also exposes `summarize`, `status`, and `cancel`. A safety blocker
with `resume_target=none` excludes `respond_to_blocker`.

Canonical state/action table:

| Stored state | Allowed next actions |
|---|---|
| `created`, `source_resolved`, `evidence_collected`, `proposal_ready` | `summarize`, `status`, `cancel` |
| `awaiting_approval` | `revise_proposal`, `summarize`, `status`, `approve_and_verify`, `cancel` |
| `applying`, `verifying`, `repairing` | `summarize`, `status` |
| `blocked` with resumable active blocker | `respond_to_blocker`, `summarize`, `status`, `cancel` |
| `blocked` with `resume_target=none` | `summarize`, `status`, `cancel` |
| `completed`, `rejected`, `failed`, `cancelled` | `summarize`, `status` |

`start` creates a new run and is outside continuation-state validation. This
table also corrects the current `CodingRunAction` type omission of `status`.

L2d run-affordance input and follow-up handoff:

```python
{
    "coding_runs": list[CodingRunContextV1],
    "coding_run_followup": {
        "mode": "none | single | ambiguous",
        "runs": list[{"objective_summary": str, "active_blocker": {
            "question": str, "options": list[str],
        } | None}],
    },
}
```

`coding_runs` is L2d-only dynamic human input and includes the opaque run ref.
`coding_run_followup` is L3-only and excludes refs and every operational field.
L2d retains the existing `accepted_coding_task_request` fields. Materialization
accepts only an offered ref and its allowed action; it rejects stale/unknown
actions and omitted refs with zero or multiple eligible contexts. It does not
parse a ref from user prose.

Accepted-task context loader:

```python
async def load_open_coding_run_contexts_for_scope(
    *,
    source_platform: str,
    source_channel_id: str,
    requester_global_user_id: str,
    limit: int = 3,
) -> list[CodingRunContextV1]: ...
```

The accepted-task repository owns this public loader. The MongoDB query uses a
partial compound index named `accepted_task_open_coding_run_context_lookup` on
`source_platform`, `source_channel_id`, `requester_global_user_id`,
`action_kind`, and descending `updated_at`, filtered by
`coding_run_context.followup_open=true`. The repository reads newest rows,
collapses them by `coding_run_ref` before applying the three-row public cap, and
returns only validated `CodingRunContextV1` values. Nodes call this public
loader and never import the DB repository directly.

Approval evidence:

```python
{
    "approved": True,
    "approved_by": str,
    "approved_at": str,
    "approval_reason": str,
    "approval_evidence": {
        "source_message_id": str,
        "source_trigger_source": "user_message",
        "requester_global_user_id": str,
        "quote": str,
        "storage_timestamp_utc": str,
    },
}
```

The quote is the current user-message `dialog_text` content capped at 500
characters before public sanitization. `approved_by` equals the trusted
requester id and `approved_at` equals the trusted storage timestamp. Empty ids,
an empty quote, a non-user trigger, or mismatched requester ids reject the
action before queueing.

Run operation outcome:

```python
{
    "operation_outcome": "applied | busy | rejected",
    "status": "stored coding-run status",
    "allowed_next_actions": list[str],
    "retry_guidance": str,
}
```

`busy` is response-only: it does not write a ledger status or event.

Benchmark result:

```python
{
    "schema_version": "coding_agent_benchmark_result.v1",
    "benchmark_version": str,
    "case_id": str,
    "category": str,
    "engine_id": "pipeline_v1",
    "routes": list[dict[str, str]],
    "status": "passed | failed | blocked",
    "entrypoint": "accepted_task | coding_run",
    "elapsed_ms": int,
    "llm_call_count": int,
    "token_usage": dict[str, int] | None,
    "final_run_status": str,
    "evaluator": {
        "status": "passed | failed | not_applicable",
        "checks": list[str],
    },
    "trace_paths": list[str],
    "notes": list[str],
}
```

The manifest contains exactly 30 pinned local-fixture cases across bug fixes,
small features, mixed create/edit, source-free creation, revision, preflight,
verification repair, environment blocker, blocker response, and concurrency.
Hidden evaluator specs and expected outcomes are harness data and must not be
included in model prompts.

## LLM Call And Context Budget

This phase changes L2d prompt inputs and accepted-task result projections. It
must not add a new response-path LLM call. L2d receives bounded run affordance
context and typed blocker text only; raw worker metadata, diffs, execution
output, private paths, approval evidence, locks, and ledger internals remain
excluded. The dynamic projection is capped at three contexts, 500 characters
per objective/question, five actions per run, and five blocker options.

L2d decides which allowed continuation, if any, the current message expresses.
It receives only that message and bounded `coding_runs`, returning existing
action fields. `coding_run` owns legality; materialization owns scope/ref
binding; background work owns delivery; L3 owns visible wording. The prompt
excludes locks, approval evidence, worker/execution metadata, and storage
structure; this change adds no fallback, repair call, classifier, or capability.

`respond_to_blocker` adds no classifier call. L2d already makes the semantic
decision from the user turn. `retry_verification` is deterministic;
`replan_proposal` uses the existing writing/modifying background LLM path with
the user answer faithfully included as current-run context. Approval evidence
validation is structural and adds no LLM call.

The accepted-task context loader adds at most one indexed MongoDB read on a
`user_message` cognition turn. Result-ready turns use the current sanitized job
context and do not issue that lookup. Lock acquisition adds bounded waiting but
no model calls.

The benchmark harness may run live LLM calls, but only under explicit test or
benchmark invocation. It records calls and token usage outside production. Run
one live case at a time, inspect its trace, and aggregate existing result files
without batching LLM cases.

## Change Surface

### Delete

- Remove v1 coding-agent worker-payload validation and v2 metadata emission
  after queued v1 coding jobs are drained.
- Remove duplicated status-to-action tables outside `coding_run` and generic
  L2d decision prose that contradicts run-specific actions.

### Modify

- `src/kazusa_ai_chatbot/coding_agent/coding_run/models.py`,
  `coding_run/supervisor.py`, and `coding_run/ledger.py`: blocker lifecycle,
  resume targets, canonical actions, approval evidence, operation outcomes,
  schema increment, and lock-scoped ledger/event mutation.
- `src/kazusa_ai_chatbot/coding_agent/code_patching/models.py`: add approval
  evidence to the coding-run approval shape without weakening the trusted
  direct patch-apply boundary.
- `src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py`: payload and
  metadata v2/v3 cutover, blocker responses, strict approval evidence, and
  `CodingRunContextV1` projection.
- `src/kazusa_ai_chatbot/background_work/jobs.py` and `models.py`: worker
  payload v2 validation and job storage for trusted evidence fields.
- `src/kazusa_ai_chatbot/background_work/worker.py` and `result_source.py`:
  copy only sanitized coding-run context into accepted-task state and the
  result-ready cognition episode.
- `src/kazusa_ai_chatbot/accepted_task/models.py`, `lifecycle.py`,
  `__init__.py`, and `src/kazusa_ai_chatbot/db/accepted_tasks.py`: persist the
  optional semantic run context, add the indexed latest-context lookup, and
  collapse contexts by run ref.
- `src/kazusa_ai_chatbot/cognition_episode.py`: allow one prompt-safe
  `coding_run_context` in accepted-task result metadata.
- `src/kazusa_ai_chatbot/action_spec/registry.py`,
  `action_spec/handlers/background_work.py`,
  `cognition_chain_core/contracts.py`, `chain.py`, `prompt_selection.py`,
  `action_selection.py`, `action_selection_prompt.py`, and `stages/l2d.py`:
  action contract, coding-run context input, L2d decisions, and authoritative
  materialization checks.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`:
  materialize the new decision, uniquely bind current run context, and attach
  trusted current-message approval evidence.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` and
  `persona_supervisor2_schema.py`: load bounded unresolved run contexts for
  user-message turns and pass the distinct L2d and L3 projections through the
  core without replacing existing group-engagement action context.
- Accepted-task-specific L3 source instructions in
  `src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py`: render an active
  blocker question/options when present without exposing operational fields.
- Coding-agent, coding-run, accepted-task, background-work, action-spec,
  cognition-chain, and nodes README files: document the full semantic data
  path, cardinality, provenance, locks, and cutover.
- Existing tests covering accepted-task persistence, result-source projection,
  action-spec, background worker, L2d, coding-run, and full workflow integration.

### Create

- `src/kazusa_ai_chatbot/coding_agent/coding_run/locking.py`: ordered
  standard-library Windows/POSIX file-lock implementation.
- `scripts/run_coding_agent_benchmark.py`: one-case runner and result
  aggregator.
- `tests/fixtures/coding_agent_benchmark/cases.jsonl`: versioned 30-case
  manifest with pinned fixture refs and hidden evaluator contracts.
- `tests/test_coding_agent_phase_c_run_context_contracts.py`: blocker/context,
  allowed-action, approval-evidence, and result-source contracts.
- `tests/test_coding_agent_phase_c_locking.py`: same-run/source serialization,
  lock order, process contention, release, and busy outcome.
- `tests/test_coding_agent_benchmark_contracts.py`: manifest, result schema,
  evaluator isolation, call counting, and aggregation.
- `tests/test_coding_agent_phase_c_accepted_task_live_db.py`: opt-in `live_db`
  proof that index bootstrap succeeds and the scoped latest-context loader
  collapses rows by run ref without crossing requester/channel scope.
- `test_artifacts/llm_traces/coding_agent_phase_c/`: committed raw and reviewed
  evidence for the five Phase C `live_llm` gates; no production consumer reads
  this directory.

### Keep

- Existing accepted-task lifecycle and background-worker queue ownership.
- Existing async result-ready cognition delivery, run refs, managed-copy
  containment, Phase B execution planning/classification, and L3 visible-wording
  ownership.

## Overdesign Guardrail

- Actual problem: The outer loop cannot reliably answer typed blockers, bind
  legal next actions to L2d, audit approval, or serialize same-source work.
- Minimal change: Extend existing durable run/action-spec contracts and add a
  single accepted-task semantic context, ordered workspace locks, and a
  test-only benchmark seam; keep the async accepted-task model.
- Ownership boundaries: Coding run owns state, blockers, locks, and public
  projection; accepted task owns persona-facing durable context; action-spec
  owns deterministic validation; L2d owns semantic action selection from
  prompt-safe affordances; worker owns queue handoff; L3 owns visible blocker
  wording.
- Rejected complexity: interactive CLI, web UI, generic action loop, raw worker
  metadata in prompts, production benchmark scheduler, and new approval UI.
- Evidence threshold: Add the JSON-action engine only through a Phase D plan
  after the 30-case harness shows it beats `pipeline_v1` on end-state pass rate
  without violating safety budgets. Add breadth only after category failures
  show a specific missing operation/tool is the blocker.

## Agent Autonomy Boundaries

- The responsible agent may add helper functions only within the listed
  ownership modules.
- The responsible agent must not expose raw execution output, raw diffs,
  private roots, or worker-local execution specs to L2d.
- Locking must fail closed with a retryable public status rather than allowing
  concurrent writes to the same run or source identity.
- If approval evidence cannot be sourced from a user message trigger, the
  action must be rejected.
- The responsible agent must not let an accepted-task loader read arbitrary
  workspace ledgers or raw background jobs. It reads only accepted-task
  `CodingRunContextV1` projections under trusted requester/channel scope.
- The responsible agent must not let L2d emit `allowed_next_actions`, blocker
  ids, resume targets, approval objects, lock keys, or benchmark evaluator
  fields.
- The responsible agent must not implement lock cleanup by deleting an
  unverified computed path. Lock files stay inside the resolved workspace lock
  root and kernel locks release on handle close/process exit.
- The responsible agent must not batch live benchmark cases. Each case requires
  a separate command and trace judgment before the next case starts.

## Implementation Order

1. Parent establishes the focused run/context contract in
   `tests/test_coding_agent_phase_c_run_context_contracts.py`.
   - Cover one-active-blocker invariants, all resume targets, canonical allowed
     actions, context sanitization, accepted-task persistence, result-ready
     projection, later-user-turn lookup, unique auto-binding, multi-run
     ambiguity, stale/unknown action rejection, no raw-user-text ref binding,
     L3 follow-up handoff, and no raw worker metadata.
   - Expected baseline: worker metadata stops before cognition and later turns
     have no structured coding-run context.
2. Parent adds approval-evidence tests through current-message action
   materialization, v2 queue payload validation, worker approval creation,
   ledger/event persistence, and public redaction.
   - Include empty quote/id, result-ready trigger, mismatched requester, and
     worker-time fallback rejection.
   - Expected baseline: current worker fabricates `current_user` and current
     time and records no message quote/id.
3. Parent adds locking tests in `tests/test_coding_agent_phase_c_locking.py`.
   - Use two processes sharing one temporary workspace to prove same-run and
     same-source mutation serialization, deterministic key order, release after
     normal/exception exits, start-before-first-write locking, and non-mutating
     busy timeout.
   - Expected baseline: ledger/event read-modify-write has no cross-process
     lock.
4. Parent adds benchmark schema tests and the 30-case manifest before the
   runner. Validate category counts, pinned fixtures, hidden evaluator fields,
   one-case selection, result aggregation, and LLInterface call counting.
5. Parent starts exactly one production-code subagent with ownership limited to
   Phase C production files in `Change Surface`, this approved plan, and the
   focused contracts from steps 1-4.
6. Production subagent implements the canonical coding-run action/blocker state
   first, then v2/v3 worker contracts, accepted-task semantic persistence, L2d
   projection/materialization, approval evidence, and finally ordered locks.
   It deletes duplicated action tables during the same cutover.
7. Parent adds patched-LLM integration coverage while production work runs.
   - Prove `result_source -> cognitive_episode -> chain input -> L2d payload ->
     semantic request -> action spec -> v2 job` without calling a live model.
   - Prove the visible accepted-task source includes the active blocker
     question/options for L3 planning.
8. Parent runs focused deterministic suites. Contract failures return to steps
   1-7 before benchmark or live prompt work.
9. Parent implements the test-only benchmark runner and aggregator after the
   public run seam is stable. Run deterministic harness tests, then run three
   representative live smoke cases separately: source-backed bug fix with
   preflight, mixed create/edit with approval, and missing-dependency blocker
   plus response.
10. Parent runs all five committed Phase C live gates separately, including the
    blocker-response and ambiguous-multiple-run cases, inspects each durable
    trace and review artifact, then runs the independent code-review gate and
    remediates in-scope findings.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes focused tests and baselines before production
  implementation starts.
- Production-code subagent: exactly one native subagent after focused tests are
  established; owns production code changes only.
- Independent code-review subagent: exactly one native subagent after planned
  verification passes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - blocker, run-context, approval, lock, and benchmark contracts
  established.
  - Covers: steps 1-4.
  - Verify: all three new deterministic test files collect; record current
    failures/missing symbols and the 30-case manifest validation result.
  - Handoff: start the production-code subagent at Stage 2.
  - Sign-off: `Codex/2026-07-10` after baselines and focused contracts passed.
- [x] Stage 2 - canonical run state and semantic context persistence complete.
  - Covers: step 6 through coding-run, worker, accepted-task, and DB contracts.
  - Verify: Phase C run-context tests, the opt-in accepted-task `live_db` test,
    and accepted-task/background-run regressions pass.
  - Evidence: v1 drain/cutover, one-active-blocker behavior, resume targets,
    v2/v3 payloads, indexed lookup, and sanitized context.
  - Handoff: Stage 3 L2d/action integration.
  - Sign-off: `Codex/2026-07-10` after regression, live-db, and cutover checks.
- [x] Stage 3 - L2d affordance binding and visible blocker handoff complete.
  - Covers: steps 6-7 cognition/action work.
  - Verify: patched handoff, action-spec, cognition-core, and L3 source tests
    pass.
  - Evidence: result-ready and later-turn paths, unique/multiple run behavior,
    unavailable action rejection, and no operational-field leakage.
  - Handoff: Stage 4 approval and locks.
  - Sign-off: `Codex/2026-07-10` after 110 focused deterministic regressions
    passed, including L2d semantic action vocabulary, run-affordance binding,
    result-ready/later-turn context projection, ambiguity rejection, and L3
    prompt-safe blocker handoff.
- [x] Stage 4 - approval provenance and operation serialization complete.
  - Covers: steps 2-3 and remaining step 6 work.
  - Verify: approval negative/positive paths and two-process locking tests pass.
  - Evidence: exact quote/message/timestamp chain, no fallback fabrication,
    same-run/source serialization, busy outcome, and exception release.
  - Handoff: Stage 5 benchmark.
  - Sign-off: `Codex/2026-07-10` after approval provenance and two-process
    locking contracts passed; the inspected Gate 11 approval used the trusted
    current user message and produced a zero-repair environment blocker.
- [x] Stage 5 - benchmark seam and representative evidence complete.
  - Covers: step 9.
  - Verify: 30-case manifest and result-schema tests pass; three live cases run
    and are inspected one at a time.
  - Evidence: result/summary paths, model routes, call counts, elapsed time,
    evaluator results, and trace judgments.
  - Handoff: Stage 6 live outer-loop gates.
  - Sign-off: `Codex/2026-07-10` after three individually inspected smoke
    cases and an offline aggregate passed.
- [x] Stage 6 - live blocker/ambiguity behavior and independent review complete.
  - Covers: step 10.
  - Verify: all five committed live gates pass after individual inspection;
    reviewer approves the full diff after any remediation reruns.
  - Evidence: behavior judgments, reviewer identity/findings/fixes/reruns,
    residual risks, and approval status.
  - Handoff: update lifecycle only after this stage.
  - Sign-off: `Codex/2026-07-10` after final verification and no-subagent
    independent review remediation.

## Verification

Run deterministic suites:

```powershell
venv\Scripts\python -m pytest tests/test_coding_agent_phase_c_run_context_contracts.py -q
venv\Scripts\python -m pytest tests/test_coding_agent_phase_c_locking.py -q
venv\Scripts\python -m pytest tests/test_coding_agent_benchmark_contracts.py -q
venv\Scripts\python -m pytest tests/test_accepted_task_lifecycle.py -q
venv\Scripts\python -m pytest tests/test_accepted_task_prompt_contract.py -q
venv\Scripts\python -m pytest tests/test_coding_agent_background_run_contracts.py -q
venv\Scripts\python -m pytest tests/test_action_spec_evaluator.py -q
venv\Scripts\python -m pytest tests/test_action_spec_results.py -q
venv\Scripts\python -m pytest tests/test_cognition_chain_core_action_selection.py -q
venv\Scripts\python -m pytest tests/test_cognition_chain_connector_mapping.py -q
venv\Scripts\python -m pytest tests/test_coding_agent_phase9_run_supervisor_contracts.py -q
```

Run the accepted-task database proof separately with an available MongoDB:

```powershell
venv\Scripts\python -m pytest tests/test_coding_agent_phase_c_accepted_task_live_db.py -q -s -m live_db
```

This opt-in test must create the named index idempotently and prove newest-row
selection, per-run collapse, the three-context cap, and strict requester/channel
scope isolation. It must clean up only its dedicated test rows.

Run static cutover/leak checks:

```powershell
rg -n "coding_agent_worker_payload\.v1|coding_agent_worker_metadata\.v2" src tests
rg -n "current_user|storage_utc_now_iso" src/kazusa_ai_chatbot/background_work/subagent/coding_agent.py
rg -n "worker_metadata|execution_specs|approval_evidence|lock_key" src/kazusa_ai_chatbot/cognition_chain_core src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py
```

The first grep must return no runtime or current-test contract matches after
the v1/v2 cutover; historical documentation outside the active ICDs is allowed.
The second grep may retain unrelated text but must show no approval fallback.
The third grep may show structural sanitization tests or trusted materialization
code; no match may place raw worker metadata, execution specs, approval
evidence, or lock keys into an LLM payload.

Run benchmark smoke cases separately and inspect each result/trace before the
next command:

```powershell
venv\Scripts\python scripts/run_coding_agent_benchmark.py --case source_backed_preflight_bugfix
venv\Scripts\python scripts/run_coding_agent_benchmark.py --case mixed_create_edit_approval
venv\Scripts\python scripts/run_coding_agent_benchmark.py --case dependency_blocker_response
venv\Scripts\python scripts/run_coding_agent_benchmark.py --aggregate
```

The aggregator performs no LLM calls. Each live case records a result row,
route/model configuration, call count, elapsed time, hidden evaluator outcome,
and durable trace path.

Add and run these five full-workflow live cases one at a time after deterministic
tests pass:

```powershell
venv\Scripts\python -m pytest tests/test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_11_blocker_answer_resumes_same_run -q -s -m live_llm
venv\Scripts\python -m pytest tests/test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_12_ambiguous_open_runs_require_run_selection -q -s -m live_llm
venv\Scripts\python -m pytest tests/test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_13_approval_evidence_survives_queue_path -q -s -m live_llm
venv\Scripts\python -m pytest tests/test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_14_result_ready_blocker_question_survives_delivery -q -s -m live_llm
venv\Scripts\python -m pytest tests/test_coding_agent_full_workflow_integration_live_llm.py::test_live_gate_15_mixed_create_edit_approval_uses_run_affordance -q -s -m live_llm
```

Gate 11 must start from a real Phase B typed blocker and traverse L2d,
action-spec, accepted task, worker, and the same run without direct ledger edits.
Gate 12 must expose two run contexts and produce a visible clarification rather
than enqueueing a guessed continuation. Gate 13 must prove the exact current
message provenance reaches the ledger without worker fallback. Gate 14 must
prove result-ready L3 delivery renders the typed blocker question/options.
Gate 15 must cover the hard mixed workflow while honoring the selected run's
current actions. Each gate stores raw and reviewed evidence under
`test_artifacts/llm_traces/coding_agent_phase_c/`. Judge prompt behavior from
the stored trace, not pytest status alone.

## Independent Plan Review

Run before approval, execution, or handoff. With no separate reviewer, reread
the Phase B record, architecture, this plan, and current contracts; confirm
outer-loop-only scope, v2/v3 cutover, persistence/index, action/ref binding,
locks, benchmark contracts, prompt-safe L2d/L3 ownership, and five live gates.

Review record: 2026-07-10 no-subagent review resolved cutover/data migration, trusted ref binding, start-lock ordering, L3 ambiguity handoff, and five-gate live evidence. User approval moved the plan to `in_progress`; execution uses the explicitly authorized no-subagent fallback.

## Independent Code Review

Run this gate after all Verification commands pass and before final sign-off.
The reviewer must inspect action-spec safety, L2d prompt-safe boundaries,
accepted-task semantic ownership, later-turn context lookup/cardinality,
blocker resume correctness, approval evidence provenance, worker payload
cutover, cross-process lock correctness and ordering, benchmark evaluator
isolation/reproducibility, and metadata redaction. The parent agent may fix
findings only inside this plan's Change Surface, then rerun affected
verification commands and record the result.

## Acceptance Criteria

This plan is complete when:

- Typed blockers are durable, public-safe, and answerable through
  `respond_to_blocker` only when their stored resume target allows it.
- The accepted-task lifecycle persists `CodingRunContextV1`; result-ready and
  later user turns expose at most three newest unique open contexts without raw
  worker data.
- L2d sees and follows run-specific allowed next actions; deterministic
  materialization and the live worker reject absent, stale, unknown, or
  raw-user-text-derived refs.
- Multiple eligible ref-less runs produce clarification rather than guessed
  continuation.
- Approval events include the bounded exact user quote, source message id,
  user trigger, requester id, and original storage timestamp, with no fallback
  fabrication.
- Same-run and same-normalized-source mutations serialize across processes;
  timeout produces a non-mutating busy outcome.
- The versioned manifest contains 30 valid pinned cases. Three representative
  live smoke outputs and an aggregate are archived and reproducible.
- The accepted-task index bootstrap and scoped latest-context query pass the
  explicit `live_db` proof without backfilling existing rows.
- Deterministic regression tests, relevant live integration tests, and
  independent code review pass.

## Execution Evidence

- 2026-07-10 Stage 1 baseline: `venv\Scripts\python -m pytest tests/test_coding_agent_phase_c_run_context_contracts.py tests/test_coding_agent_phase_c_locking.py tests/test_coding_agent_benchmark_contracts.py -q` failed during collection as expected. `allowed_next_actions` and `project_coding_run_context` are absent from `coding_run.ledger`; `coding_run.locking` is absent. The benchmark assertion did not run because collection stopped first.
- 2026-07-10 Focused implementation evidence: the three new Stage 1 files pass 4/4; `tests/test_coding_agent_phase9_run_supervisor_contracts.py` passes 14/14 with the new public response fields; `tests/test_coding_agent_background_run_contracts.py` passes 23/23 after the v2 payload/v3 metadata cutover. An initial combined regression run timed out because v1 test payloads entered the legacy worker path; updating the planned test contract to v2 resolved that behavior.
- 2026-07-10 Stage 1 sign-off: expanded deterministic contracts now cover trusted affordance binding, raw-text ref rejection, L3 prompt-safe follow-up projection, approval-evidence negative paths, multi-process source-lock contention, exception release, and start-before-ledger locking. `venv\Scripts\python -m pytest tests\test_coding_agent_phase_c_run_context_contracts.py tests\test_coding_agent_phase_c_locking.py tests\test_coding_agent_benchmark_contracts.py tests\test_accepted_task_lifecycle.py tests\test_accepted_task_prompt_contract.py tests\test_coding_agent_background_run_contracts.py tests\test_action_spec_evaluator.py tests\test_action_spec_results.py tests\test_cognition_chain_core_action_selection.py tests\test_cognition_chain_connector_mapping.py tests\test_coding_agent_phase9_run_supervisor_contracts.py -q` passed 105/105. The 30-case manifest validation passed. The separate indexed scope-loader proof `venv\Scripts\python -m pytest tests\test_coding_agent_phase_c_accepted_task_live_db.py -q -s -m live_db` passed 1/1.
- 2026-07-10 Stage 2 sign-off: `CodingRunContextV1` now persists on accepted tasks, has a named partial scope index, is collapsed by the public accepted-task loader, and remains sanitized through result-ready episode projection. The focused run-context/background suites passed 33/33 after result-source sanitization. The live Mongo proof passed 1/1. Read-only deployment audit found `queued_v1_coding_agent_jobs=0`; static v1/v2 contract checks returned no matches in `src` or `tests`.
- 2026-07-10 Live Gate 11 attempt: `test_live_gate_11_blocker_answer_resumes_same_run` was run alone with `-s -m live_llm` and exceeded the 120-second command window before a durable trace was written. No live gate sign-off is recorded; the case requires a longer, separately inspected run.
- 2026-07-10 Live Gate 11 rerun: after updating the in-memory accepted-task store for `coding_run_context`, the case reached its second L2d turn. The model selected `accepted_coding_task_request` / `approve_and_verify`, but materialization rejected the continuation as outside the offered context. The raw L2d output and next rerun must inspect the propagated context list; Stage 6 remains open.
- 2026-07-10 Stage 3 sign-off: `venv\Scripts\python -m pytest tests\test_coding_agent_phase_c_run_context_contracts.py tests\test_coding_agent_phase_c_locking.py tests\test_coding_agent_benchmark_contracts.py tests\test_accepted_task_lifecycle.py tests\test_accepted_task_prompt_contract.py tests\test_coding_agent_background_run_contracts.py tests\test_action_spec_evaluator.py tests\test_action_spec_results.py tests\test_cognition_chain_core_action_selection.py tests\test_cognition_chain_connector_mapping.py tests\test_coding_agent_phase9_run_supervisor_contracts.py -q` passed 110/110. The L2d prompt registry and default action summary now name `respond_to_blocker` and require every continuation to be present in the offered run context.
- 2026-07-10 Gate 11 correction and inspection: deterministic review-only fixture proposal input established an `awaiting_approval` run through the public background-worker/coding-run path. Approval then produced a real Phase B `environment_dependency_missing` blocker with `resume_target=retry_verification`, an active prompt-safe question, and zero repair attempts. The third live L2d turn selected `respond_to_blocker`; the same run completed. Each turn persisted input, offered contexts, rendered L2d payload, raw and parsed output, materialized action specs, and any rejection reason before materialization assertions. The corrected gate passed individually in 52.42 seconds; Stage 6 remains open until Gates 12-15 and independent review pass.
- 2026-07-10 Stage 4 sign-off: focused verification passed 110/110. Approval tests reject empty quote/id, non-user triggers, mismatched requesters, and worker fallback; positive queue/worker/ledger coverage preserves the trusted quote, source message id, requester, trigger, and storage timestamp. Kernel lock coverage proves sorted keys, cross-process same-source contention, exception release, start-before-ledger locking, and non-mutating busy continuation results.
- 2026-07-10 Stage 5 first smoke: the benchmark runner now creates an isolated Git checkout, supplies an explicit fixture task, performs the public approval/verification continuation when required, and writes a bounded blocked result with partial trace evidence on timeout. The deterministic benchmark contracts passed 5/5. `source_backed_preflight_bugfix` passed individually in 117500 ms with two LLM calls: its managed-copy patch applied, focused pytest succeeded, approval provenance was recorded, and repair attempts remained empty. Mixed accepted-task and blocker-response benchmark smoke cases remain required.
- 2026-07-10 Stage 5 sign-off: the benchmark uses fresh per-invocation fixture workspaces so interrupted runs remain inspectable and never require retry-time deletion. The three one-at-a-time smoke cases passed and were inspected: `source_backed_preflight_bugfix` (117500 ms, 2 LLM calls, completed), `mixed_create_edit_approval` (162766 ms, 2 LLM calls, completed), and `dependency_blocker_response` (1239 ms, 0 LLM calls with deterministic review-only fixture proposal, blocked). The dependency trace contains one open `environment_dependency_missing` blocker with `resume_target=retry_verification`, failed pytest, and zero repair attempts. `venv\Scripts\python scripts\run_coding_agent_benchmark.py --aggregate` read existing artifacts only and reported three passed rows across bug-fix, mixed-create/edit, and environment-blocker categories.
- 2026-07-10 Stage 6 final verification: `venv\Scripts\python -m pytest tests\test_coding_agent_phase_c_run_context_contracts.py tests\test_coding_agent_phase_c_locking.py tests\test_coding_agent_benchmark_contracts.py tests\test_accepted_task_lifecycle.py tests\test_accepted_task_prompt_contract.py tests\test_coding_agent_background_run_contracts.py tests\test_action_spec_evaluator.py tests\test_action_spec_results.py tests\test_cognition_chain_core_action_selection.py tests\test_cognition_chain_connector_mapping.py tests\test_coding_agent_phase9_run_supervisor_contracts.py -q` passed 111/111. The indexed Mongo proof passed 1/1. Cutover and leak greps returned no matches, and `git diff --check` passed.
- 2026-07-10 Stage 6 live gates: all five `live_llm` cases ran individually and were inspected from their durable traces. Gate 11 passed in 43.46 seconds with the same run progressing from awaiting approval to a real `retry_verification` blocker to `respond_to_blocker`; Gate 12 passed in 10.14 seconds with two eligible runs and only a visible clarification; Gate 13 passed in 29.09 seconds and verified exact approval quote/id/trigger/requester/timestamp in the ledger; Gate 14 passed in 7.17 seconds with the blocker question/options rendered and no run ref in L3; Gate 15 passed in 32.32 seconds with the offered approval action bound to the same run and non-test runtime changes. Current evidence is under `test_artifacts/llm_traces/coding_agent_full_workflow_integration_live_llm__gate_11...` through `...__gate_15...`.
- 2026-07-10 No-subagent independent code review: reviewed the action-spec safety, L2d/L3 prompt boundaries, accepted-task persistence/query cardinality, blocker resume, approval provenance, v2/v3 cutover, ordered file locks, benchmark isolation, and metadata redaction. Two findings were remediated: the manifest now labels every runner-driven case as `coding_run`, and one centralized coding-run context sanitizer now strips unlisted metadata before accepted-task storage, result delivery, and repository lookup. The worker-tick contract asserts that execution specs and approval evidence cannot persist in the accepted-task context. Affected contracts reran before the final 111/111 regression and live gates. Residual observation: the live model occasionally emits invalid optional `missing_user_inputs` goal-progress values; L2d drops that optional field while all required action contracts remained valid. Review outcome: approved for completion.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| L2d receives private worker internals | Prompt-safe projection only | Action prompt/result tests |
| Run context disappears after result delivery | Persist semantic context on accepted-task rows and load by trusted scope | Result-ready and later-turn contract tests |
| Multiple open runs bind the wrong task | Cap at three, require explicit ref when ambiguous, reject guessed materialization | Ambiguity deterministic/live gates |
| User-supplied ref crosses trusted scope | Bind only refs supplied by the scoped context loader; remove raw-text ref scanning | Ref-binding deterministic contracts |
| Approval evidence leaks sensitive text | Bound quote and sanitize projections | Approval evidence tests |
| Approval is still fabricated by fallback | Require trusted dialog percept/id/requester/timestamp before enqueue | Negative provenance tests and static grep |
| Locks deadlock background work | Sorted acquisition, nonblocking kernel locks, bounded timeout, release in `finally` | Two-process lock contention/order/exception tests |
| A start writes before its source lock exists | Allocate run id and acquire run/source keys before the first ledger write | Start-lock ordering tests |
| Lock identity misses two aliases of one checkout | Contract covers equal normalized request/persisted identities and documents alias limit | Source-key tests and residual risk evidence |
| Benchmark hidden checks leak into prompts | Separate manifest evaluator fields from entrypoint payload construction | Benchmark contract tests |
| Benchmark becomes flaky | Pin fixtures, run one live case at a time, archive traces, and aggregate offline | Three smoke cases plus schema tests |
