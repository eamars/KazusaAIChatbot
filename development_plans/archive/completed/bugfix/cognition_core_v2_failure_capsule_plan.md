# Cognition Core V2 Failure Capsule Plan

## Summary

- Goal: preserve the exact Cognition Core V2 input and model-attempt evidence
  for terminal and partial failures without changing normal cognition behavior.
- Plan class: `medium`.
- Status: `completed`.
- Mandatory skills: `development-plan`, `llm-trace-debug`,
  `local-llm-architecture`, `py-style`, and `test-style-and-execution`.
- Cutover: compatible protected-trace extension.
- Highest risk: observability code affecting response latency or replacing the
  original cognition result or exception.
- Acceptance: clean runs produce no capsule write; capture and persistence
  failures never alter cognition calls, outputs, retries, or exceptions.

## Context

`LLM_TRACE_CAPTURE_MODE=metadata` currently stores hashes and status but
discards raw messages, responses, and parsed output. Cognition input is added to
graph state only after `run_cognition(...)` succeeds. A metadata-mode failure
therefore cannot be reconstructed exactly.

Cognition V2 also has model stages that do not consistently use protected trace
recording. The validation-only diagnostics module demonstrates a useful
context-local capture shape, but it is test-only, incomplete, and writes local
artifacts rather than protected production traces.

This capability is a prerequisite for resuming
`required_selection_partial_recovery_bugfix_plan.md`.

## Mandatory Skills

- `development-plan`: plan execution and lifecycle.
- `llm-trace-debug`: protected trace, export, privacy, and retention contract.
- `local-llm-architecture`: preserve cognition ownership and call budgets.
- `py-style`: all Python implementation and review.
- `test-style-and-execution`: deterministic and integration test execution.

## Mandatory Rules

- Capture must never change an LLM request, response, retry, parser,
  validator, branch decision, returned output, or raised exception.
- Clean cognition runs perform no new capsule database write.
- Capsule persistence is scheduled asynchronously and is never awaited by the
  response path.
- Capture setup, buffering, serialization, task scheduling, and persistence
  failures are contained and reported through sanitized warnings.
- Raw capsule data stays in protected trace storage and never enters
  `event_log_events`, public status APIs, adapter responses, or normal logs.
- API keys are never captured. Route endpoint and generation settings may be
  stored only in the protected capsule.
- Use `DEBUG_LOG_TTL_DAYS`; add no capsule-specific retention setting.
- `LLM_TRACE_CAPTURE_MODE=off` remains an explicit capture opt-out.
- Add no prompt, model, endpoint, token-cap, retry, cognition schema, queue,
  action, dialog, or persistence behavior change.
- After context compaction or each major stage signoff, reread this entire plan.
- Before completion, run the independent code-review gate and record evidence.
- Preserve unrelated worktree changes.

## Must Do

- Capture the exact public-entrypoint input before validation.
- Assign a unique `cognition_invocation_id` for each invocation under one
  `llm_trace_id`, including service safe retries.
- Capture every Cognition V2 LLM attempt with messages, non-secret config, raw
  response, parsed output, parse status, validation error, branch, and attempt.
- Persist a capsule on terminal failure, recovered contract/provider failure,
  failed appraisal, failed branch, or degraded surface disposition.
- Keep successful first-attempt runs metadata-only.
- Extend existing trace export so one invocation can be exported as a
  self-contained replay input.
- Prove capture failure isolation, concurrency isolation, and no clean-path
  capsule persistence.

## Deferred

- Deterministic model-output injection replay.
- Process-crash, forced-termination, and MongoDB-outage durability.
- Local write-ahead files or encrypted spooling.
- Capturing systems outside Cognition Core V2.
- Changing normal metadata/full/off trace meanings beyond failure promotion.
- Resuming the required-selection recovery implementation.

## Cutover Policy

Overall strategy: compatible.

- Existing `off`, `metadata`, and `full` modes remain valid.
- Existing trace-run and trace-step documents remain readable.
- Failure capsules use new protected step records under the same trace id.
- No historical data migration or backfill is attempted.
- Rollback removes the new capture session and capsule records without changing
  cognition contracts.

## Target State

```text
Cognition V2 public entrypoint
  -> begin context-local capture with exact input
  -> run unchanged cognition code
  -> record exact model attempts in memory
  -> clean success
       -> discard buffer
       -> return unchanged output
  -> partial or terminal failure
       -> schedule protected capsule write
       -> immediately return unchanged output or re-raise original exception
```

The capsule covers `run_cognition(...)`, `run_text_surface_planning(...)`,
`repair_text_surface_planning(...)`, and
`run_visual_surface_planning(...)`.

## Design Decisions

| Topic | Decision | Reason |
|---|---|---|
| Capture owner | `llm_tracing` owns context buffering and protected persistence | It already owns trace privacy, TTL, and export boundaries. |
| Failure owner | Cognition entrypoints and stage handlers mark partial or terminal failure | They know the real disposition without semantic inference. |
| Storage | Reuse `llm_trace_steps` with capsule-specific records | Avoids a new collection, index family, or retention policy. |
| Clean path | Keep exact evidence only in invocation-local memory and discard it on clean success | Prevents new clean-path database writes. |
| Persistence | Fire-and-observe background task | Persistence cannot delay or replace cognition behavior. |
| Safe retry | New invocation id under the same trace id | Keeps repeated cognition inputs and attempts distinguishable. |
| LLM behavior | No new model call or prompt change | This is deterministic observability only. |

## Contracts And Data Shapes

Add a context-local capture session with this protected payload:

```python
{
    "schema_version": "cognition_failure_capsule.v1",
    "trace_id": str,
    "cognition_invocation_id": str,
    "entrypoint": str,
    "input_payload": object,
    "input_sha256": str,
    "attempts": [
        {
            "stage_name": str,
            "branch_id": str,
            "attempt_index": int,
            "config": {
                "route_name": str,
                "base_url": str,
                "model": str,
                "temperature": float | None,
                "top_p": float | None,
                "top_k": int | None,
                "max_completion_tokens": int | None,
                "presence_penalty": float | None,
                "timeout_seconds": float | None,
                "thinking_enabled": bool,
            },
            "messages": list[dict[str, str]],
            "raw_response_text": str,
            "parsed_output": object,
            "parse_status": str,
            "validation_error": str,
            "status": str,
        }
    ],
    "failure_events": list[dict[str, object]],
    "outcome": "partial_failure" | "terminal_failure",
    "exception": dict[str, object] | None,
}
```

The API key is absent by contract. Values are copied into a JSON/BSON-safe
snapshot so later mutation cannot change the capsule.

`record_llm_trace_step(...)` continues writing its existing metadata/full
document and additionally appends exact evidence to an active capsule session.
When no capsule session is bound, its behavior is unchanged.

Capsule persistence creates protected `llm_trace_steps` rows with
`capture_reason="cognition_failure_capsule"` and the invocation id. The export
tool groups these rows into one replay artifact.

## LLM Call And Context Budget

- LLM call count: unchanged.
- Model routes and prompts: unchanged.
- Retry caps: unchanged.
- Model context sizes: unchanged.
- Clean path: no new database write and no awaited persistence.
- Failure path: one background protected-capsule persistence task.
- Memory: bounded by existing Cognition V2 input, prompt, output, branch, and
  attempt caps for one invocation.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/llm_tracing/failure_capsule.py`
  - context-local session, safe buffering, failure marking, and asynchronous
    persistence scheduling.

### Modify

- `src/kazusa_ai_chatbot/llm_tracing/__init__.py`
  - append exact attempts to an active capsule before metadata omission.
- `src/kazusa_ai_chatbot/db/llm_tracing.py`
  - insert capsule records into existing protected trace-step storage.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`
  - bind/finalize capture around `run_cognition(...)`.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface.py`
  - bind/finalize capture around the three public surface entrypoints.
- Cognition model owners under
  `src/kazusa_ai_chatbot/cognition_core_v2/`
  - ensure every attempt records protected trace evidence and concrete
    validation errors.
- `src/scripts/export_llm_trace.py`
  - group capsule rows by invocation id without exposing them elsewhere.
- `src/kazusa_ai_chatbot/llm_tracing/README.md`,
  `src/kazusa_ai_chatbot/cognition_core_v2/README.md`, and `docs/HOWTO.md`
  - document capture, privacy, export, and failure-isolation behavior.
- `tests/test_llm_tracing.py`, `tests/test_llm_trace_export.py`, and
  `tests/test_cognition_core_v2_failures.py`
  - focused contract and integration coverage.

### Keep

- Current cognition input/output schemas.
- Current trace modes, shared TTL, and protected/sanitized data separation.
- Current service safe-retry behavior.

## Overdesign Guardrail

- Actual problem: exact V2 failure input is lost in metadata mode.
- Minimal change: one context-local buffer plus protected failure-only
  persistence using existing trace storage.
- Ownership: cognition marks real failure; tracing stores evidence;
  deterministic code isolates failures; LLM stages retain semantic ownership.
- Rejected complexity: new database collection, new trace mode, write-ahead
  log, replay engine, prompt changes, extra model calls, and public debug API.
- Evidence threshold: process-crash or database-outage loss must be observed
  before adding durable local spooling.

## Agent Autonomy Boundaries

- Implement only the capsule contract and listed trace instrumentation.
- Do not catch and convert cognition exceptions; schedule capture and re-raise
  the original object.
- Do not infer failure from user text or model semantics.
- Do not add new persistence fallbacks when background capsule writing fails.
- Stop if exact capture requires an awaited response-path write, a public raw
  trace endpoint, or a cognition schema change.

## Implementation Order

1. Add failing tracing tests for exact input, partial/terminal promotion, clean
   discard, capture exceptions, blocked persistence, and concurrent isolation.
2. Implement context-local buffering and background protected persistence.
3. Instrument existing Cognition V2 trace calls and add missing stage calls.
4. Wrap all four public Cognition V2 entrypoints.
5. Add export grouping and export tests.
6. Run focused tests, then Cognition V2 and service non-live regression.
7. Update documentation and execution evidence.
8. Run independent code review and remediate in-scope findings.

## Execution Model

- The parent owns test contracts, integration tests, verification, evidence,
  lifecycle updates, and review remediation.
- One production-code subagent implements the listed production surface after
  the initial failing tests exist.
- One fresh review subagent reviews the final diff and evidence.
- If native subagents are unavailable, stop unless the user explicitly
  authorizes fallback execution.

## Progress Checklist

- [x] Failure-isolation and exact-input tests fail for the current code.
- [x] Capsule session and protected persistence implemented.
- [x] All Cognition V2 model attempts instrumented.
- [x] Four public entrypoints integrated.
- [x] Export grouping and privacy tests pass.
- [x] Focused and non-live regression pass.
- [x] Documentation and execution evidence complete.
- [x] Independent review passes with no unresolved findings.

## Verification

```powershell
venv\Scripts\python.exe -m pytest tests\test_llm_tracing.py tests\test_llm_trace_export.py tests\test_cognition_core_v2_failures.py -q
venv\Scripts\python.exe -m pytest tests -m "not live_llm and not live_db" -q
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\llm_tracing\failure_capsule.py src\kazusa_ai_chatbot\llm_tracing\__init__.py src\kazusa_ai_chatbot\db\llm_tracing.py src\kazusa_ai_chatbot\cognition_core_v2\facade.py src\kazusa_ai_chatbot\cognition_core_v2\surface.py
git diff --check
```

Required assertions:

- A clean cognition output is byte/structure-equivalent to the baseline and
  schedules no capsule write.
- A terminal exception is the same exception object after capture scheduling.
- A recovered or optional failure returns the same output and schedules one
  capsule.
- A blocked or raising persistence coroutine cannot delay, cancel, or replace
  cognition behavior.
- Concurrent invocations never mix inputs or attempts.
- API keys and raw capsule content are absent from sanitized event logs.
- `off` mode stores no capsule.
- Export by trace and invocation id reproduces the exact captured input and
  ordered attempts.

## Independent Code Review

The reviewer checks capture isolation, ContextVar concurrency, task lifecycle,
secret exclusion, protected storage, TTL use, complete V2 call-site coverage,
unchanged LLM budgets, unchanged exception identity, tests, and plan alignment.
The parent remediates only findings inside this change surface and reruns the
affected gates.

## Execution Evidence

- Baseline/failing tests: Focused suite fails during collection because
  `kazusa_ai_chatbot.llm_tracing.failure_capsule` does not exist (2026-07-31).
- Changed files: Added
  `src/kazusa_ai_chatbot/llm_tracing/failure_capsule.py`; updated protected
  tracing, the four V2 public entrypoints, all six direct V2 model owners,
  canonical JSON repair tracing, protected export, subsystem documentation,
  and focused regression tests. The existing database trace-step insertion
  boundary and shared TTL were sufficient, so `db/llm_tracing.py` remained
  unchanged. Pre-existing required-selection plan and live-test work were
  preserved outside this scope.
- Focused tests: `45 passed` for `test_llm_tracing.py`,
  `test_llm_trace_export.py`, and `test_cognition_core_v2_failures.py`.
  Adjacent V2/parser/routing/retry/dialog coverage: `263 passed, 7 deselected`.
- Full non-live regression: Final filtered repository run passed with
  `3809 passed, 3 skipped, 828 deselected`. The raw mandated command remains
  blocked during collection by unrelated missing replay, personality, and
  real-history fixture artifacts in four modules. The filtered run also
  excluded four previously identified unrelated artifact/baseline-debt
  modules; no failure from this change remained.
- Export/privacy checks: Exact pre-validation input, ordered attempts,
  concurrent invocation isolation, clean discard, `off` mode, same-exception
  identity, sanitized setup/scheduling/persistence failure, API-key exclusion,
  stalled-write timeout cleanup, compatible trace export, and one-invocation
  export selection all pass.
- Independent reviewer and findings: Fresh read-only reviewer found no
  Critical or High issues. Three Medium findings—surface non-string parser
  behavior, stalled persistence lifecycle, and invocation-specific export—
  plus one Low Python-style finding were remediated. The reviewer reran the
  three remediation tests (`3 passed`), confirmed `git diff --check`, and
  returned final gate `PASS` with no unresolved findings.
- Final diff: Production prompts, routes, model configuration, retry caps,
  completion budgets, cognition schemas, and call counts remain unchanged.
  Final `py_compile` and `git diff --check` pass. Direct in-memory attempt
  capture for previously untraced owners was retained because routing clean
  attempts through ordinary trace persistence would violate the failure-only,
  response-path isolation contract.

## Acceptance Criteria

- Every terminal or surfaced partial Cognition V2 failure produces a protected
  capsule when tracing is enabled.
- The capsule contains the exact entry input and every recorded model attempt,
  grouped by a unique invocation id.
- Clean runs create no capsule persistence task.
- Capture and persistence failures never change normal output, exception,
  retries, model calls, or response-path control flow.
- Existing trace exports remain compatible and can export a capsule.
- No raw capsule data or API key enters sanitized/public surfaces.
- Focused tests, non-live regression, static checks, and independent review
  pass.
