# generic pipeline cancellation and channel guard bugfix plan

## Summary

- Goal: prevent same-channel background self-cognition and reflection-attached
  self-cognition from producing visible output from a stale context snapshot
  while foreground channel work is pending or running.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `systematic-debugging`, `py-style`, `test-style-and-execution`
- Overall cutover strategy: bigbang for runtime coordination contracts,
  self-cognition cancellation wiring, reflection phase wiring, and ICD updates.
- Highest-risk areas: leaking a background send after a foreground event,
  completing a durable calendar run after a cooperative deferral, consuming
  calendar retry budget for deferrals, overfitting the guard to `/chat`,
  and introducing a compatibility shim around the old global busy probe.
- Acceptance criteria: any runtime function can request cancellation for a
  scoped pipeline; same-channel foreground work cancels or defers background
  self-cognition before dispatcher send; reflection phase calendar runs are
  rescheduled rather than completed on pipeline deferral; different channels
  remain independent; the shared ICDs document the rule and the generic API.

## Context

The RCA for the 2026-07-02 23:42:55 NZT QQ group output found that the visible
assistant row was not a normal `/chat` response. It was a
`self_cognition.group_chat_trigger_review` dispatch from the reflection phase
group-review path:

- platform: `qq`
- channel: `638473184`
- platform message id: `764812011`
- delivery tracking id: `3838d2d2b46f449882d2263cb562e90a`
- self-cognition attempt:
  `self_cognition_attempt:5d7201a14acdf3f680ef177c`
- reflection group activity window:
  `scope_11afa3456af9:2026-07-02T11:30:00+00:00:2026-07-02T11:45:00+00:00`

Adjacent chat evidence showed a normal `/chat` response at 23:41:17 NZT, new
same-channel user messages at 23:41:51, 23:41:59, 23:42:10, and 23:42:22 NZT,
then the self-cognition visible send at 23:42:55 NZT. The user then challenged
the stale assumption at 23:43:45 NZT, and a normal `/chat` response at
23:44:35 NZT answered the current full-precision question.

The code-level cause is a coordination gap across modules:

- `service._handle_calendar_reflection_phase_run(...)` currently calls
  `handle_reflection_phase_calendar_run(...)` with
  `is_primary_interaction_busy=lambda: False`.
- The standalone self-cognition worker has pre-case busy checks, but the
  reflection phase calendar path bypasses the real service busy probe.
- `self_cognition.worker.run_self_cognition_worker_tick(...)` has no
  mid-flight cancellation token after a case starts building artifacts.
- `self_cognition.runner` projects the pre-collected
  `case["visible_context"]` into chat history and does not re-read the
  channel before dialog/send.
- `dispatcher.handlers.handle_send_message(...)` persists and sends once it is
  called; the caller is responsible for cancellation checkpoints before send.
- `calendar_scheduler.worker.run_calendar_worker_tick(...)` treats only
  handler `status="skipped"` specially. Other handler results are marked
  completed, so a reflection handler `deferred` outcome must become an
  explicit calendar contract instead of an opaque completion summary.

The product-level rule is not `/chat`-specific. Foreground work in a channel
can come from current or future applications. Background pipelines that may
produce visible output must cooperate with a generic scope cancellation
interface before they send.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  archiving, or signing off this plan.
- `local-llm-architecture`: load before changing pipeline ownership,
  prompt/context boundaries, LLM call sequencing, or background versus
  foreground responsibility.
- `systematic-debugging`: load before changing race-condition behavior or
  interpreting failed cancellation/concurrency tests.
- `py-style`: load before editing Python production files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`.
- Before implementation, read `README.md`, `docs/HOWTO.md`,
  `development_plans/README.md`, this full plan, and these subsystem ICDs:
  `src/kazusa_ai_chatbot/brain_service/README.md`,
  `src/kazusa_ai_chatbot/self_cognition/README.md`,
  `src/kazusa_ai_chatbot/reflection_cycle/README.md`,
  `src/kazusa_ai_chatbot/calendar_scheduler/README.md`, and
  `src/kazusa_ai_chatbot/dispatcher/README.md`.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, documentation, and plan edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Do not add a compatibility shim, alias module, fallback mapper, feature flag,
  or parallel vocabulary for the old global-only busy probe.
- Do not make the cancellation interface specific to `/chat`, service input
  queues, QQ, Discord, or self-cognition.
- Do not ask an LLM to decide runtime cancellation, lock ownership, scheduler
  retry semantics, adapter availability, or dispatcher permission.
- Do not add a new LLM call, retry loop, prompt branch, or model route for
  this fix.
- Do not force-cancel provider calls at the socket/task level in this plan.
  Cancellation is cooperative through deterministic checkpoints.
- Do not let raw reflection output enter normal cognition as a workaround.
- Keep adapters thin. Adapters do not own pipeline cancellation policy.
- Calendar deferral must not consume max-attempts budget.
- Calendar deferral must reverse the claim-time `attempt_count` increment
  before releasing the run back to `pending`.
- A background pipeline that is cancelled or deferred must not persist a
  successful action attempt, record a reviewed group window, or call
  dispatcher send for the cancelled visible output.
- Parent-led native subagent execution is required for production-code
  execution. If native subagent capability is unavailable, stop unless the
  user explicitly approves fallback execution.
- The parent must establish the focused failing test contract before
  production implementation starts.
- Any change to this plan's cutover policy requires user approval before
  implementation.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Add a generic runtime coordination interface that any in-process function can
  call to request cancellation for a scoped pipeline.
- Define a canonical channel pipeline scope using platform,
  platform-channel-id, and channel type.
- Track foreground and background pipeline runs per scope.
- Allow foreground work to request cooperative cancellation of lower-precedence
  background runs in the same scope.
- Allow background work to test admission before starting and to checkpoint
  cancellation during execution.
- Wire normal chat handling as one foreground caller of the generic
  cancellation API, not as a special-case gate.
- Wire reflection phase group self-cognition and standalone self-cognition as
  background callers of the generic cancellation API.
- Preserve different-channel independence. A foreground event in one channel
  must not cancel or defer background work in another channel.
- Add cancellation checkpoints before expensive source/LLM/dialog stages and
  immediately before dispatcher delivery.
- Convert reflection phase deferred outcomes into durable calendar requeue,
  not calendar completion.
- Preserve group review ledger semantics: deferred/cancelled cases leave the
  selected window unreviewed so a later run can rebuild fresh context.
- Update the relevant ICDs to state the shared same-channel rule and the
  generic cancellation interface.
- Add focused deterministic tests for coordinator behavior, service wiring,
  self-cognition cancellation, reflection phase deferral, calendar requeue, and
  dispatcher non-delivery on cancellation.
- Keep LLM prompt-facing payloads free of scheduler ids, leases, cancellation
  run ids, and runtime coordination metadata.

## Deferred

- Cross-process or distributed cancellation across multiple service processes.
- A durable database-backed active-pipeline registry.
- External HTTP or control-console APIs for manually cancelling pipelines.
- Force-cancelling in-flight model provider calls or adapter sends.
- Rewriting dispatcher semantics to own cancellation policy.
- Reworking reflection phase materialization or eligible-scope selection.
- New scheduler trigger kinds.
- New visible response gating, response-ratio tuning, or character judgment
  suppression unrelated to stale same-channel concurrency.
- Retrospective data repair for historical self-cognition sends.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Runtime coordination API | bigbang | Introduce one canonical package and import it through its public entrypoint. Do not add alias modules or alternate busy-probe interfaces. |
| Reflection phase calendar execution | bigbang | Replace the service-owned `lambda: False` bypass with scoped coordination. Do not retain a fallback calendar path that ignores channel foreground work. |
| Self-cognition visible delivery | bigbang | Route background visible delivery through scoped admission and checkpoints. Do not preserve an unchecked action-attempt/send path. |
| Calendar deferred handling | bigbang | Add one deferred-result branch that requeues running calendar runs as pending and restores the claim attempt. Do not model deferral as completed, skipped, or failed. |
| Dispatcher | compatible | Keep dispatcher delivery semantics unchanged. The only compatibility surface is that callers still invoke dispatcher after passing cancellation checkpoints. |
| Tests and ICDs | bigbang | Rewrite tests and docs to the new coordination contract in the same change. Do not leave tests asserting the old reflection phase bypass. |

Cutover policy enforcement:

- The responsible execution agent must follow the selected policy for each
  area.
- For `bigbang` areas, rewrite old references instead of preserving old call
  shapes.
- The `compatible` dispatcher policy preserves only dispatcher delivery
  semantics; it does not authorize dispatcher-owned cancellation policy.
- Any change to a local cutover policy requires user approval before
  implementation.

## Target State

The target runtime ownership is:

```text
adapter/debug client
-> service intake
-> runtime coordination foreground admission/cancellation
-> RAG/cognition/dialog/persistence for foreground work

reflection/calendar/self-cognition/background work
-> runtime coordination background admission
-> cooperative checkpoints
-> fresh source/context build
-> cooperative checkpoint before dispatcher
-> dispatcher send only if not cancelled
```

The shared rule is:

```text
For one canonical channel scope, foreground work has precedence over
background work that can produce visible output. When foreground work is
pending or running, same-scope background work must either not start, or must
cooperatively cancel and reschedule before visible delivery.
```

The rule is generic. `/chat` is only one foreground caller. Future
applications can call the same cancellation interface when they start
foreground work or when they know a scoped pipeline should be cancelled.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Scope identity | Use `(platform, platform_channel_id, channel_type)`. | The defect is same-channel stale output. User id would under-cancel group-visible work. |
| Private conversations | Do not add user id to the key. | Private channel id is the conversation scope. |
| Coordination storage | Keep runtime coordination process-local. | The observed race is in-process; distributed cancellation is deferred. |
| Precedence model | Use only `foreground` and `background`. | This covers current and future foreground applications without inventing priority weights. |
| Cancellation mechanics | Use cooperative cancellation tokens and deterministic checkpoints. | Force-cancelling model/provider calls is outside scope; delivery prevention is required. |
| Foreground lifetime | Start foreground scope ownership when channel work becomes pending and release it after post-turn persistence finishes. | Background must see queued foreground work, not only active model execution. |
| Dropped foreground work | Release the foreground handle when a queued item is dropped, collapsed, rejected, or cancelled before processing. | Queue pruning must not leak a permanent same-scope block. |
| Background admission | Deny same-scope background admission while foreground is pending or active. | This prevents stale source snapshots from starting. |
| Calendar deferral | Represent deferral as a handler result that requeues the run as pending and restores the claim attempt. | Calendar runs have no durable `deferred` status and should not burn retry budget. |
| Dispatcher | Keep dispatcher as a delivery executor. | Cancellation policy belongs to callers before dispatcher send. |
| LLM boundary | Keep runtime coordination metadata out of prompts. | LLM stages own semantic judgment, not scheduling or delivery policy. |

## Contracts And Data Shapes

Create this package:

```text
src/kazusa_ai_chatbot/runtime_coordination/
  README.md
  __init__.py
  models.py
  coordinator.py
```

Production callers must import public symbols from
`kazusa_ai_chatbot.runtime_coordination`, not from private module internals.
The package must expose this public contract:

```python
@dataclass(frozen=True)
class PipelineScope:
    platform: str
    platform_channel_id: str
    channel_type: str

@dataclass(frozen=True)
class PipelineCancellation:
    run_id: str
    scope: PipelineScope
    requested_by: str
    reason: str
    checkpoint: str

class PipelineCancelled(Exception):
    cancellation: PipelineCancellation

class PipelineRunHandle:
    run_id: str
    scope: PipelineScope
    owner: str
    precedence: Literal["foreground", "background"]

    async def __aenter__(self) -> PipelineRunHandle: ...
    async def __aexit__(self, exc_type, exc, tb) -> None: ...
    def cancelled(self) -> bool: ...
    def raise_if_cancelled(self, checkpoint: str) -> None: ...

@dataclass(frozen=True)
class PipelineRunAdmission:
    admitted: bool
    handle: PipelineRunHandle | None
    defer_reason: str | None

class PipelineCoordinator:
    def request_cancellation(
        self,
        *,
        scope: PipelineScope,
        requested_by: str,
        reason: str,
        target_precedence: Collection[str] = ("background",),
    ) -> list[str]: ...

    async def start_run(
        self,
        *,
        scope: PipelineScope,
        owner: str,
        precedence: Literal["foreground", "background"],
        run_kind: str,
    ) -> PipelineRunAdmission: ...
```

If admitted, `PipelineRunAdmission.handle` is non-null and must be used as an
async context manager. If deferred, `handle` is null and `defer_reason` is one
of these strings:

- `same_scope_foreground_active`
- `same_scope_foreground_pending`
- `pipeline_shutdown`

The coordinator must support:

- same-scope cancellation of active background handles;
- same-scope background admission denial while foreground work is pending or
  active;
- different-scope independence;
- idempotent handle release;
- no leaked active handles after exceptions.

The self-cognition worker/runner contract must accept an optional cancellation
handle or checkpoint callback. The runner must checkpoint at least:

- before source collection or case artifact build;
- before each LLM stage it owns;
- after each LLM stage before using the output;
- before action-attempt persistence;
- immediately before dispatcher delivery;
- after dispatcher delivery only for cleanup/accounting paths.

The reflection phase calendar summary contract must include:

```python
{
    "status": "deferred",
    "run_kind": "reflection_phase_slot",
    "defer_reason": "same_scope_foreground_active",
    "processed_count": 1,
    "succeeded_count": 1,
    "failed_count": 0,
    "skipped_count": 0,
    "run_ids": ["hourly-run-1"],
}
```

when any phase result is deferred. Counts and run ids must reflect any work
already persisted inside the phase execution, but the calendar phase run itself
must not be marked completed.

The calendar repository must expose a lease-aware deferral transition for a
claimed run. It must:

- match `run_id`, `status="running"`, and matching `lease_owner`;
- clear lease fields;
- set stored run status back to `pending`;
- store a bounded deferral summary in `failure_summary` with
  `{"deferred": true, "reason": <reason>, "retryable": true}`;
- decrement `attempt_count` by `1` in the same update, with a filter requiring
  `attempt_count >= 1`;
- not consume max-attempts budget for the deferred execution;
- update `updated_at`.

## LLM Call And Context Budget

This plan must not add any LLM calls.

Prompt-facing payloads must not include runtime coordination internals:

- pipeline run ids;
- cancellation reasons;
- calendar leases;
- scheduler attempt counts;
- dispatcher tracking ids;
- internal worker names.

The only context behavior change is that cancelled or deferred background work
does not continue to dialog/send from an old snapshot. When the background
pipeline runs later, it must rebuild its source case from current channel and
memory state.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/runtime_coordination/README.md`: ICD for the generic
  scoped pipeline cancellation interface.
- `src/kazusa_ai_chatbot/runtime_coordination/__init__.py`: public import
  entrypoint for the coordinator contract.
- `src/kazusa_ai_chatbot/runtime_coordination/models.py`: public dataclasses,
  literals, and exception types.
- `src/kazusa_ai_chatbot/runtime_coordination/coordinator.py`: process-local
  admission, cancellation, and handle-release logic.
- `tests/test_runtime_coordination.py`: focused module-contract tests.

### Modify

- `src/kazusa_ai_chatbot/service.py`: own the process coordinator, foreground
  scope handles, queue-drop release, and service-to-calendar/reflection wiring.
- `src/kazusa_ai_chatbot/self_cognition/worker.py`: accept cancellation
  handles, return deferred results for cancellation, and avoid successful
  action persistence on cancelled runs.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`: checkpoint before LLM,
  action-attempt, and dispatcher boundaries.
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`: start group review as a
  scoped background run and propagate deferred self-cognition outcomes.
- `src/kazusa_ai_chatbot/calendar_scheduler/reflection_phase.py`: summarize
  deferred phase results for the calendar worker.
- `src/kazusa_ai_chatbot/calendar_scheduler/repository.py`: add the
  lease-aware pending requeue transition that restores the claim attempt.
- `src/kazusa_ai_chatbot/calendar_scheduler/worker.py`: handle
  `status="deferred"` and return `deferred_count`.
- `src/kazusa_ai_chatbot/brain_service/README.md`: document the brain-service
  boundary and same-scope foreground/background rule.
- `src/kazusa_ai_chatbot/self_cognition/README.md`: document cooperative
  checkpoints for visible background output.
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`: document reflection
  phase same-channel deferral.
- `src/kazusa_ai_chatbot/calendar_scheduler/README.md`: document deferred
  handler results and pending requeue semantics.
- `src/kazusa_ai_chatbot/dispatcher/README.md`: document that callers own
  cancellation checkpoints before send.
- `docs/HOWTO.md`: update only if existing runtime operation guidance would
  otherwise contradict the new coordination contract.
- `tests/test_self_cognition_integration.py`: cancellation/deferred worker
  behavior and no-successful-attempt assertions.
- `tests/test_reflection_cycle_stage1c_worker.py`: scoped background admission
  and group-window non-review on deferral.
- `tests/test_reflection_cycle_stage1c_service.py`: service calendar/reflection
  wiring no longer uses the bypass.
- `tests/test_calendar_scheduler_reflection_phase.py`: deferred summary
  propagation, including partial phase work.
- `tests/test_calendar_scheduler_worker.py`: deferred calendar requeue branch
  and `deferred_count`.
- `tests/test_service_input_queue.py`: foreground handle creation,
  cancellation request on enqueue, and handle release for dropped/collapsed
  queued items.

### Keep

- Dispatcher production delivery code remains behavior-compatible.
- Adapter packages remain thin and do not receive cancellation policy.
- Reflection phase materialization and eligible-scope selection stay unchanged.
- LLM prompts, model routing, RAG, cognition prompts, and dialog prompts stay
  unchanged.

## Overdesign Guardrail

- Actual problem: same-channel background self-cognition can send visible
  output from a stale source snapshot while foreground channel work is pending
  or running.
- Minimal change: add one in-process scoped coordinator, wire foreground and
  background callers through it, checkpoint before visible delivery, and
  requeue deferred calendar phase runs.
- Ownership boundaries: deterministic runtime code owns admission,
  cancellation, scheduler requeue, and dispatcher call permission; LLM stages
  own semantic judgment only after the deterministic runtime admits the work;
  dispatcher owns delivery mechanics only.
- Rejected complexity: distributed locks, persistent active-pipeline storage,
  public cancellation APIs, adapter-owned policy, force-cancelling provider
  calls, new prompt instructions, priority weights, fairness algorithms,
  response-ratio tuning, and unrelated character gating.
- Evidence threshold: add distributed coordination, external cancellation APIs,
  or richer priority models only after an approved near-term multi-process or
  external-application integration proves the process-local foreground versus
  background contract is insufficient.

## Agent Autonomy Boundaries

An implementing agent may:

- choose private helper names that preserve the public contracts above;
- adjust tests to match existing helper style;
- add narrow deterministic helpers for scope extraction;
- update docs/ICDs in the listed modules;
- add bounded event logging for cancellation/deferred outcomes.

An implementing agent must not:

- modify production code while this plan is `draft`;
- change database schemas beyond the calendar run deferral fields already
  allowed by nullable result/failure summaries and existing status values;
- add persistent cancellation storage;
- add new LLM prompt instructions for cancellation;
- change adapter delivery semantics except by ensuring cancelled callers do not
  invoke dispatcher send;
- mark this plan complete without independent code review and verification
  evidence.

## Implementation Order

1. Parent records context baseline.
   - Load mandatory skills.
   - Read the files named in `Mandatory Rules`.
   - Read directly involved source and test files from `Change Surface`.
   - Run `git status --short` and record it in `Execution Evidence`.

2. Parent writes focused coordinator tests first.
   - File: `tests/test_runtime_coordination.py`.
   - Add tests named
     `test_background_admission_defers_for_same_scope_foreground`,
     `test_request_cancellation_marks_same_scope_background_only`,
     `test_different_scope_background_survives_cancellation`,
     `test_cancelled_checkpoint_raises_pipeline_cancelled`, and
     `test_handle_context_manager_releases_after_exception`.
   - Run `venv\Scripts\python.exe -m pytest tests\test_runtime_coordination.py -q`.
   - Expected before implementation: import or missing-symbol failure for
     `kazusa_ai_chatbot.runtime_coordination`.

3. Parent writes focused integration tests before production wiring.
   - `tests/test_service_input_queue.py`: foreground handle starts on enqueue,
     cancellation is requested for same-scope background, and queue prune or
     collapse releases dropped foreground handles.
   - `tests/test_self_cognition_integration.py`: cancelled background work
     returns deferred and does not persist successful action attempts.
   - `tests/test_reflection_cycle_stage1c_worker.py`: group review admission
     defers while same-scope foreground is active and does not mark the
     selected group window reviewed.
   - `tests/test_reflection_cycle_stage1c_service.py`: service reflection
     phase wiring uses the coordinator and does not inject the old bypass.
   - `tests/test_calendar_scheduler_reflection_phase.py`: deferred phase
     summaries propagate, including partial prior phase work.
   - `tests/test_calendar_scheduler_worker.py`: handler `status="deferred"`
     calls calendar requeue and increments `deferred_count`.
   - Run the changed tests and record expected failures.

4. Parent starts the production-code subagent.
   - Provide this approved plan, mandatory skills, failing test contract, and
     `Change Surface`.
   - Production-code subagent owns production code only.
   - Production-code subagent must not edit tests except by explicit parent
     direction.
   - If native subagent capability is unavailable, stop unless the user has
     explicitly approved fallback execution.

5. Production-code subagent implements `runtime_coordination`.
   - Create the exact package files in `Change Surface`.
   - Export the public contract from `__init__.py`.
   - Implement same-scope cancellation, background deferral, different-scope
     independence, context-manager release, and exception-safe cleanup.

6. Parent runs coordinator tests.
   - Command:
     `venv\Scripts\python.exe -m pytest tests\test_runtime_coordination.py -q`.
   - If the test fails, fix only the coordinator contract before continuing.

7. Production-code subagent wires service foreground lifecycle.
   - Create one service-owned coordinator.
   - Start foreground handles when scoped work enters the queue.
   - Request same-scope background cancellation on enqueue.
   - Release foreground handles for processed, dropped, collapsed, rejected,
     or cancelled queue items.
   - Hold processed foreground handles through post-turn persistence.

8. Production-code subagent wires background cancellation and calendar deferral.
   - Pass cancellation handles through self-cognition worker/runner.
   - Checkpoint before source/LLM/action-attempt/dispatcher boundaries.
   - Start reflection group review as scoped background work.
   - Propagate `deferred` through reflection phase summaries.
   - Add calendar repository requeue that restores the claim attempt.
   - Add calendar worker deferred-result handling.

9. Parent runs focused integration tests.
   - Run every focused command listed in `Verification`.
   - If an integration failure exposes a coordinator contract gap, return to
     step 2 before changing more integration code.

10. Parent updates shared ICDs and docs.
    - Update every documentation file listed in `Change Surface`.
    - Keep docs aligned with the public runtime coordination contract and
      same-channel foreground/background rule.

11. Parent runs full verification.
    - Run the broad deterministic suite and static searches in `Verification`.
    - Record outputs in `Execution Evidence`.

12. Parent runs independent code review and final sign-off.
    - Start one independent code-review subagent after verification passes.
    - Address findings inside approved scope and rerun affected verification.
    - Update lifecycle status only after verification and review pass.

## Execution Model

- Parent agent owns orchestration, test code, verification, static checks,
  execution evidence, review feedback remediation, lifecycle updates, and final
  sign-off.
- Parent agent establishes the focused failing test contract before production
  implementation starts.
- Production-code subagent: exactly one native subagent, started after focused
  tests are written and baseline failures are recorded. It owns production code
  changes only and must close after planned production code changes are
  complete, excluding review fixes.
- Parent agent may update integration tests, documentation, execution
  evidence, and verification while the production-code subagent edits
  production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes. It reviews the approved plan, full diff, and
  evidence, and does not implement fixes.
- If native subagent capability is unavailable, stop before production-code
  execution unless the user explicitly approves fallback execution.
- No two agents may edit the same file concurrently. The parent must inspect
  all diffs before final verification.

## Progress Checklist

- [x] Stage 0 - plan approved for execution.
  Covers: lifecycle gate before implementation.
  Verify: plan status is `approved` or `in_progress` in this file and registry.
  Evidence: record approval source and `git status --short`.
  Sign-off: `Codex / 2026-07-03` after evidence is recorded.

- [x] Stage 1 - focused test contract established.
  Covers: implementation steps 1-3.
  Verify: changed focused tests run and fail for missing symbols or old
  behavior exactly as expected.
  Evidence: record each command and failure summary.
  Sign-off: `Codex / 2026-07-03` after baseline failures are recorded.

- [x] Stage 2 - runtime coordination module implemented.
  Covers: implementation steps 4-6.
  Verify:
  `venv\Scripts\python.exe -m pytest tests\test_runtime_coordination.py -q`.
  Evidence: record changed production files and passing output.
  Sign-off: `Codex / 2026-07-03` after coordinator tests pass.

- [x] Stage 3 - service and background wiring complete.
  Covers: implementation steps 7-9.
  Verify: all focused integration commands in `Verification` pass.
  Evidence: record commands, outputs, and any fixed integration findings.
  Sign-off: `Codex / 2026-07-03` after focused integration tests pass.

- [x] Stage 4 - ICD and operator documentation updated.
  Covers: implementation step 10.
  Verify: static search confirms docs mention the generic coordination API and
  no doc contradicts same-channel foreground precedence.
  Evidence: record doc paths changed and grep output.
  Sign-off: `Codex / 2026-07-03` after documentation review.

- [x] Stage 5 - broad verification complete.
  Covers: implementation step 11.
  Verify: broad deterministic suite and static searches in `Verification`.
  Evidence: record pass/fail output and unrelated-failure rationale.
  Sign-off: `Codex / 2026-07-03` after verification evidence is recorded.

- [x] Stage 6 - independent code review complete.
  Covers: implementation step 12.
  Verify: one independent code-review subagent reviewed plan, diff, and
  evidence; all Critical and Important findings are fixed or escalated.
  Evidence: record findings, fixes, rerun commands, residual risks, and review
  approval status.
  Sign-off: `Codex / 2026-07-03` after review approval and reruns.

## Verification

Focused tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_runtime_coordination.py -q
venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py -q
venv\Scripts\python.exe -m pytest tests\test_self_cognition_integration.py -q
venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_stage1c_worker.py -q
venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_stage1c_service.py -q
venv\Scripts\python.exe -m pytest tests\test_calendar_scheduler_reflection_phase.py -q
venv\Scripts\python.exe -m pytest tests\test_calendar_scheduler_worker.py -q
```

Broad deterministic suite:

```powershell
venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q
```

Static searches:

```powershell
rg -n "is_primary_interaction_busy=lambda: False" src\kazusa_ai_chatbot tests
rg -n "PipelineCoordinator|PipelineCancelled|request_cancellation" src\kazusa_ai_chatbot tests
rg -n "\"status\": \"deferred\"|status=\"deferred\"" src\kazusa_ai_chatbot tests
```

Expected static-search outcomes:

- Production reflection phase calendar execution must not contain the old
  `is_primary_interaction_busy=lambda: False` bypass.
- Any remaining `lambda: False` occurrences must be public one-shot/manual
  entry points or tests that do not own service calendar execution.
- New runtime coordination symbols must appear in production code and focused
  tests.
- Deferred calendar result handling must be covered by tests.
- `tests/test_calendar_scheduler_worker.py` must assert `deferred_count`.
- `tests/test_service_input_queue.py` must assert foreground handle release
  for dropped or collapsed queued items.

No live LLM or live DB test is required for this plan unless implementation
changes introduce uncertainty that deterministic tests cannot cover.

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not draft the plan. If separate reviewer delegation is unavailable or not
authorized, the parent agent must reread the plan, development-plan contract,
execution gates, cutover policy, relevant source/test context, and subsystem
ICDs from a fresh-review posture.

Review scope:

- Architecture alignment, execution readiness, instruction completeness,
  creativity suppression, and current/deferred/future scope separation.

Review record, 2026-07-03:

- Reviewer: parent-agent self-review. A separate subagent was not spawned
  because the active delegation tool requires explicit subagent authorization.
- Blockers resolved in this revision: missing cutover matrix, open file-split
  choices, missing claim-attempt restoration, optional subagents, weak
  progress/evidence gates, flat change surface, and incomplete guardrail shape.
- Approval status: blockers are resolved; the `draft` plan is not executable
  until the owner approves or moves status to `approved` or `in_progress`.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must start one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Race windows between foreground enqueue, foreground start, background
  admission, cancellation checkpoints, and dispatcher send.
- Leaked coordinator handles after queue pruning, rejection, cancellation, or
  exceptions.
- Same-channel versus different-channel scope matching.
- Calendar deferral requeue restoring the claim attempt and not consuming retry
  budget.
- Old reflection phase calendar bypass removal.
- Self-cognition action attempts and group review windows not being marked
  successful on cancellation.
- Absence of `/chat`-specific cancellation policy in generic code.
- Dispatcher and adapter ownership boundaries.
- Documentation consistency across ICDs.
- Alignment with `Must Do`, `Deferred`, `Change Surface`, implementation
  order, verification gates, and acceptance criteria.

The parent may fix findings directly only when the fix is inside this plan's
approved change surface. If a finding requires a new contract or an out-of-scope
file, stop and update the plan or request approval before implementation.

Record findings, fixes, rerun commands, residual risks, and review approval
status in `Execution Evidence`.

## Acceptance Criteria

- A generic runtime coordination API exists and is documented as the shared
  interface for scoped pipeline cancellation.
- Any function can request cancellation for a `PipelineScope` without importing
  `/chat` implementation details.
- Same-channel foreground work cancels or defers background self-cognition
  before visible delivery.
- Background self-cognition checks cancellation before dispatcher send.
- Reflection phase calendar group review no longer bypasses service
  coordination with `lambda: False`.
- Reflection phase `deferred` outcomes requeue the calendar run instead of
  marking it completed.
- Group review selected windows are not recorded as `reviewed` when the
  self-cognition pipeline is cancelled or deferred.
- Different-channel background work is not cancelled by unrelated foreground
  work.
- Dispatcher remains a delivery executor and does not own cancellation policy.
- Shared ICDs document the rule and future-application interface.
- Focused tests and static searches pass.

## Risks

- Cooperative cancellation cannot prevent the cost of an already-started LLM
  provider call. It must still prevent visible delivery after the call returns.
- Requeueing calendar runs as pending can repeatedly defer while a channel is
  busy. The existing worker poll interval bounds production retry cadence.
- Holding foreground activity through post-turn persistence can defer more
  background work than the old global count did. This is acceptable because the
  stale-context visible-send risk is higher than the short deferral.
- A scope extraction bug could cause over-cancellation or under-cancellation.
  Scope unit tests and service integration tests are mandatory.
- Existing tests that assert `lambda: False` injection will need to be updated
  to assert generic coordinator injection or explicit public one-shot behavior.

## Execution Evidence

- 2026-07-03 Stage 0: user explicitly requested execution with one
  production-code subagent and one code-review subagent. Plan status moved to
  `in_progress`. Baseline `git status --short` showed modified
  `development_plans/README.md` plus untracked active bugfix plan files,
  including this plan and unrelated user-owned lane-integrity plans.
- 2026-07-03 Stage 1 red-test contract:
  `venv\Scripts\python.exe -m pytest tests\test_runtime_coordination.py tests\test_calendar_scheduler_worker.py tests\test_calendar_scheduler_repository.py tests\test_calendar_scheduler_reflection_phase.py tests\test_service_input_queue.py tests\test_self_cognition_integration.py tests\test_reflection_cycle_stage1c_worker.py tests\test_reflection_cycle_stage1c_service.py -q`
  failed as expected with 17 failures and 130 passes. Failures covered the
  missing runtime coordination API, calendar `deferred_count` and requeue
  branch, repository `mark_calendar_run_deferred`, deferred reflection phase
  summary, foreground enqueue cancellation, foreground handle release,
  self-cognition cancellation deferral, reflection group review scoped
  admission, and service calendar coordinator injection.
- 2026-07-03 Stage 2/3 production implementation: one production-code
  subagent (`Averroes`) edited production Python only, then closed. Changed
  production files:
  `src/kazusa_ai_chatbot/runtime_coordination/__init__.py`,
  `src/kazusa_ai_chatbot/runtime_coordination/models.py`,
  `src/kazusa_ai_chatbot/runtime_coordination/coordinator.py`,
  `src/kazusa_ai_chatbot/chat_input_queue.py`,
  `src/kazusa_ai_chatbot/service.py`,
  `src/kazusa_ai_chatbot/self_cognition/runner.py`,
  `src/kazusa_ai_chatbot/self_cognition/worker.py`,
  `src/kazusa_ai_chatbot/reflection_cycle/worker.py`,
  `src/kazusa_ai_chatbot/calendar_scheduler/reflection_phase.py`,
  `src/kazusa_ai_chatbot/calendar_scheduler/repository.py`, and
  `src/kazusa_ai_chatbot/calendar_scheduler/worker.py`.
- 2026-07-03 parent integration fix: while reviewing the subagent diff, parent
  found that a standalone self-cognition cancellation after claiming a
  `future_cognition` or `commitment_due_cognition` source run would leave that
  source run running until lease expiry. Added focused coverage and wired the
  self-cognition worker to call `mark_calendar_run_deferred` for claimed source
  calendar runs on `PipelineCancelled`.
- 2026-07-03 focused verification:
  `venv\Scripts\python.exe -m pytest tests\test_self_cognition_integration.py::test_worker_tick_defers_pipeline_cancelled_case tests\test_self_cognition_integration.py::test_worker_tick_defer_requeues_claimed_source_calendar_run -q`
  passed with 2 passed.
- 2026-07-03 focused plan suite:
  `venv\Scripts\python.exe -m pytest tests\test_runtime_coordination.py tests\test_calendar_scheduler_worker.py tests\test_calendar_scheduler_repository.py tests\test_calendar_scheduler_reflection_phase.py tests\test_service_input_queue.py tests\test_self_cognition_integration.py tests\test_reflection_cycle_stage1c_worker.py tests\test_reflection_cycle_stage1c_service.py -q`
  passed with 148 passed.
- 2026-07-03 py compile:
  `venv\Scripts\python.exe -m py_compile` over all changed production Python
  files passed.
- 2026-07-03 Stage 4 ICD updates: updated
  `src/kazusa_ai_chatbot/brain_service/README.md`,
  `src/kazusa_ai_chatbot/runtime_coordination/README.md`,
  `src/kazusa_ai_chatbot/self_cognition/README.md`,
  `src/kazusa_ai_chatbot/reflection_cycle/README.md`,
  `src/kazusa_ai_chatbot/calendar_scheduler/README.md`, and
  `src/kazusa_ai_chatbot/dispatcher/README.md`. Static search confirmed
  runtime-coordination terms in all intended ICDs. No remaining old service
  calendar `is_primary_interaction_busy=lambda: False` bypass was found; the
  remaining production hits are public reflection facade defaults in
  `reflection_cycle/worker.py`.
- 2026-07-03 static verification:
  `rg -n "PipelineCoordinator|PipelineCancelled|request_cancellation" src\kazusa_ai_chatbot tests`
  found the new production symbols and focused tests. `rg -n
  'status.*deferred|deferred_count|mark_calendar_run_deferred'
  src\kazusa_ai_chatbot tests` found the deferred calendar production branch
  and test coverage. `git diff --check` passed with line-ending warnings only.
- 2026-07-03 broad deterministic suite:
  `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q`
  reported 3 failed, 2661 passed, 2 skipped, 386 deselected. Rerun of the three
  failures showed they are in clean, untouched files and unrelated to this
  plan: `README_CN.md` residue wording guard, stale `ACTION_ROUTER_PROMPT`
  fingerprint fixture, and `tests/test_reflection_cycle_readonly.py` expecting
  no `channel_name` while clean production
  `db/conversation_reflection.py` already projects `channel_name`.
- 2026-07-03 Stage 6 independent code review: one code-review subagent
  (`Nietzsche`) reviewed the approved plan, diff, and evidence, then closed.
  The review found three Important issues and no Critical issues:
  foreground enqueue caller cancellation could release the foreground handle
  after queue append; self-cognition could leak a background handle if pre-run
  source-claim setup raised before the main `finally`; and reflection phase
  calendar execution still allowed a no-coordinator fallback. Parent fixed all
  three inside the approved change surface.
- 2026-07-03 review remediation: `ChatInputQueue.enqueue` gained an
  `on_enqueued` callback so `_enqueue_chat_request` transfers foreground-handle
  ownership only after append; self-cognition worker pre-run setup now releases
  an admitted background handle on setup failure; and
  `handle_reflection_phase_calendar_run` now requires explicit
  `pipeline_coordinator` and `is_primary_interaction_busy` dependencies.
  Added focused regression tests:
  `test_cancelled_enqueue_wait_keeps_foreground_handle`,
  `test_worker_tick_releases_pipeline_handle_when_claim_raises`, and
  `test_reflection_phase_calendar_handler_requires_coordinator`.
- 2026-07-03 review residual: the reviewer noted the coordinator only emits
  `same_scope_foreground_active` from background admission. Accepted with no
  code change because foreground pending work is represented by an acquired
  foreground handle before enqueue wait; callers may still use
  `same_scope_foreground_pending` as the cancellation request reason when
  starting foreground work.
- 2026-07-03 review-fix targeted verification:
  `venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py::test_cancelled_enqueue_wait_keeps_foreground_handle tests\test_self_cognition_integration.py::test_worker_tick_releases_pipeline_handle_when_claim_raises tests\test_calendar_scheduler_reflection_phase.py::test_reflection_phase_calendar_handler_uses_execution_seam tests\test_calendar_scheduler_reflection_phase.py::test_reflection_phase_calendar_handler_returns_deferred_summary tests\test_calendar_scheduler_reflection_phase.py::test_reflection_phase_calendar_handler_requires_coordinator tests\test_reflection_cycle_stage1c_service.py::test_calendar_reflection_handler_passes_runtime_coordinator -q`
  passed with 6 passed.
- 2026-07-03 final focused plan suite after review fixes:
  `venv\Scripts\python.exe -m pytest tests\test_runtime_coordination.py tests\test_calendar_scheduler_worker.py tests\test_calendar_scheduler_repository.py tests\test_calendar_scheduler_reflection_phase.py tests\test_service_input_queue.py tests\test_self_cognition_integration.py tests\test_reflection_cycle_stage1c_worker.py tests\test_reflection_cycle_stage1c_service.py -q`
  passed with 151 passed.
- 2026-07-03 final static and compile verification: `py_compile` over all
  changed production Python files passed; static searches found expected
  runtime-coordination and deferred-calendar production/test coverage; old
  `is_primary_interaction_busy=lambda: False` hits remain only in public
  reflection facade defaults; `git diff --check` passed with line-ending
  warnings only.
- 2026-07-03 final broad deterministic suite after review fixes:
  `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q`
  reported 3 failed, 2664 passed, 2 skipped, 386 deselected. The same three
  unrelated baseline failures remained in clean, untouched files:
  `README_CN.md` residue wording guard, stale `ACTION_ROUTER_PROMPT`
  fingerprint fixture, and the reflection read-only allowlist expectation for
  `channel_name`.
- 2026-07-03 post-completion baseline remediation: per owner direction, the
  prompt fingerprint regression guard was removed from
  `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`;
  `README_CN.md` residue terminology was aligned to `私念残留`; and
  `tests/test_reflection_cycle_readonly.py` now treats `channel_name` as
  allowed sanitized source-preparation metadata per the DB ICD. Targeted
  baseline verification passed with 31 passed:
  `venv\Scripts\python.exe -m pytest tests\test_internal_monologue_residue_prompt_boundaries.py::test_root_readmes_document_residue_architecture tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py tests\test_reflection_cycle_readonly.py::test_db_interface_uses_message_field_allowlist -q`.
- 2026-07-03 post-completion plan-proof verification: focused plan suite
  passed with 151 passed:
  `venv\Scripts\python.exe -m pytest tests\test_runtime_coordination.py tests\test_calendar_scheduler_worker.py tests\test_calendar_scheduler_repository.py tests\test_calendar_scheduler_reflection_phase.py tests\test_service_input_queue.py tests\test_self_cognition_integration.py tests\test_reflection_cycle_stage1c_worker.py tests\test_reflection_cycle_stage1c_service.py -q`.
  Broad deterministic suite passed with 2666 passed, 2 skipped, 386
  deselected:
  `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q`.
