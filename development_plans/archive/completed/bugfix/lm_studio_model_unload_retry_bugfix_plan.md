# lm studio model unload retry bugfix plan

## Summary

- Goal: add one central, per-model retry-and-pause boundary for LM Studio model
  unload errors without changing prompt stages, graph behavior, or generic
  retry policy.
- Plan class: medium
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`
- Overall cutover strategy: compatible central wrapper at `get_llm()`; no LLM
  call-site rewrites.
- Highest-risk areas: retrying real prompt/config errors, serializing normal
  LLM traffic, broken sync JSON repair, and dead wait state after retry
  failure.
- Acceptance criteria: all `get_llm()` clients retry exactly once for confirmed
  LM Studio unload; same-model calls pause only during reload retry; other
  models keep running; non-unload errors raise without retry.

## Confirmed Signature

- Confirmed LM Studio unload signature:
  `The model has crashed without additional information.`
- The classifier must match only that confirmed signature. Do not include other
  generic crash text without a new approval.

## Context

Kazusa builds route-specific OpenAI-compatible chat clients through
`src/kazusa_ai_chatbot/utils.py::get_llm()`. Most runtime stages call
`.ainvoke(...)`; JSON repair uses `.invoke(...)`.

The observed failure was a local OpenAI-compatible 400 response. The OpenAI SDK
does not retry ordinary 400 responses. The accepted requirement is narrow:

- handle only LM Studio unload errors;
- retrying the same request is the LM Studio reload trigger;
- while one call is doing that reload-triggering retry, calls for the same
  `(base_url, model)` wait;
- calls for different model keys continue;
- no rate limiter, readiness probe, fallback model, broad retry wrapper, or
  stage-specific retry code.

This is deterministic runtime recovery. It must not change prompts, graph
topology, RAG, cognition, dialog, consolidation, scheduling, adapters,
persistence, or model routing.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  archiving, or signing off this plan.
- `local-llm-architecture`: load before changing the LLM call boundary or
  evaluating broader retry, fallback, prompt, graph, or routing proposals.
- `py-style`: load before editing Python production files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Execute production changes only while this plan status is `in_progress`.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, and plan edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Do not add or change prompts, graph edges, route config, env vars, worker
  retry loops, event logs, DB schema, adapters, or scheduler behavior.
- Do not use LangChain `.with_retry()` or OpenAI SDK `max_retries` for this
  fix.
- Do not add a rate limiter, semaphore, concurrency cap, readiness probe,
  `/models` polling, fallback model, feature flag, or recovery background task.
- Do not retry context overflow, malformed JSON, schema/prompt errors, generic
  400s, timeouts, network errors, rate limits, 5xx errors, or unknown errors.
- Retry only a confirmed LM Studio model-unload error signature.
- Retry the same call at most once after unload on the first attempt.
- If the retry fails, release same-model waiters and raise the retry exception
  through the existing caller or worker error path.
- Key monitor state by `(base_url.rstrip("/"), model)`.
- Cover both `.ainvoke(...)` and `.invoke(...)`.
- After context compaction, reread this plan before implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major checklist stage, reread this plan before the next
  stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the `Independent Code Review` gate and record the result in
  `Execution Evidence`.

## Must Do

- Create a central model-unload monitor used only by `get_llm()`.
- Preserve the current `get_llm()` construction pattern and existing call
  sites.
- Return a monitored wrapper exposing `.ainvoke(...)`, `.invoke(...)`, and
  delegated attributes.
- Update `get_llm()` return typing so it no longer claims to return a raw
  `ChatOpenAI` when it returns a wrapper.
- Add an exact unload-error classifier.
- Make the first call that observes unload become the reload owner when no
  owner exists for that model key.
- If another in-flight call observes unload while an owner exists, make it wait
  for the owner, then retry its same call once.
- Make new same-model calls wait before first send while a reload owner exists.
- Release waiters in `finally` for both successful and failed owner retries.
- Add deterministic fake-LLM tests for async retry, sync retry, non-unload
  propagation, same-model wait, different-model non-wait, and concurrent
  unload ownership.
- Add a short HOWTO operator note for central one-retry LM Studio unload
  recovery.

## Deferred

- Do not fix generic local-model crashes.
- Do not classify the observed `model has crashed` message unless validation
  proves it is LM Studio unload.
- Do not add normal-call concurrency management.
- Do not add global process-wide pause behavior.
- Do not add telemetry, counters, runtime status fields, operator endpoints, or
  live LLM tests.
- Do not change existing worker-level retries in reflection, dialog, facts
  harvesting, web search, or other stages.

## Cutover Policy

Overall strategy: compatible central wrapper.

| Area | Policy | Instruction |
|---|---|---|
| LLM factory | compatible | `get_llm()` remains the construction entrypoint and keeps the same parameters. |
| Runtime call sites | compatible | Existing `.ainvoke(...)` and `.invoke(...)` call sites must not change. |
| Error behavior | bigbang for unload only | Confirmed LM Studio unload errors get one same-request retry; all other errors keep existing behavior. |
| Model scoping | bigbang | Pause state is keyed by `(base_url.rstrip("/"), model)`, not route name or process-wide status. |
| Prompt/model config | unchanged | No prompt, environment, or route variable changes. |

## Target State

Normal path:

```text
caller -> monitored LLM -> inner ChatOpenAI -> response
```

Unload path:

```text
caller A for model M
  -> first attempt raises confirmed unload
  -> A becomes reload owner for M
  -> A retries the same request once
  -> A clears reload state in finally
  -> A returns retry response or raises retry error

caller B for model M during A's owner retry
  -> waits before first send or after its own first-attempt unload race
  -> sends or retries after A clears reload state

caller C for model N
  -> never waits on M
```

The monitor is not a scheduler or rate limiter. It blocks only calls for a model
key already in unload recovery.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Boundary | Put recovery behind `get_llm()` | It is the shared LLM construction seam and avoids stage-specific retries. |
| Model key | `(base_url.rstrip("/"), model)` | Same model names on different endpoints must not share pause state. |
| Reload trigger | Retry the same request once | LM Studio reloads on retry by backend policy. |
| Readiness | No probe | Retry is the recovery mechanism; polling is outside scope. |
| Race handling | Single owner per model key | Concurrent unload failures should not stampede reload retries. |
| Sync support | Cover `.invoke(...)` and `.ainvoke(...)` | JSON repair uses sync invoke through the same factory. |

## Contracts And Data Shapes

Create `src/kazusa_ai_chatbot/llm_reload_monitor.py` with:

```python
ModelKey = tuple[str, str]

def monitored_chat_model(
    inner_llm: object,
    *,
    base_url: str,
    model: str,
) -> MonitoredChatModel: ...

def is_lm_studio_model_unload_error(exc: BaseException) -> bool: ...
```

`MonitoredChatModel` contract:

- expose `async def ainvoke(self, *args: Any, **kwargs: Any) -> Any`;
- expose `def invoke(self, *args: Any, **kwargs: Any) -> Any`;
- delegate all other attributes to `inner_llm` through `__getattr__`;
- wait before first send when the model key is already reloading;
- on confirmed unload, either become the reload owner or wait for the current
  owner, then retry the same method once;
- release same-model waiters in all owner completion paths;
- raise non-unload first-attempt exceptions unchanged;
- raise retry exceptions unchanged.

Implementation mechanics:

- Use one shared, thread-safe in-process monitor state so sync and async calls
  observe the same model-key reload status.
- Async waits must not block the event loop.
- Do not expose monitor mutation hooks except where tests need a public
  wrapper factory and classifier.

Update `src/kazusa_ai_chatbot/utils.py::get_llm()`:

- keep constructing `ChatOpenAI` with the existing arguments;
- return `monitored_chat_model(_llm, base_url=base_url, model=model)`;
- update the return annotation to match the monitored wrapper contract.

## LLM Call And Context Budget

- Normal path before and after: one LLM call per stage invocation.
- Unload path before: one failed LLM call and caller-level failure.
- Unload path after: one failed LLM call plus one same-request retry.
- No prompt, context, token budget, or stage-count changes.
- Latency impact is limited to calls for a model key currently in unload
  recovery.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/llm_reload_monitor.py`: monitor, wrapper, classifier,
  sync/async retry behavior.
- `tests/test_llm_reload_monitor.py`: focused fake-LLM tests.

### Modify

- `src/kazusa_ai_chatbot/utils.py`: wrap `ChatOpenAI` in `get_llm()` and update
  return typing.
- `docs/HOWTO.md`: add a short operator note.
- `development_plans/README.md`: keep this plan listed under Active Bugfix
  Plans.

### Keep

- All LLM stage modules, prompts, worker retry helpers, model configuration,
  adapters, persistence, and graph definitions remain unchanged.

## Overdesign Guardrail

- Actual problem: LM Studio model unload can make a local model call fail even
  though retrying that same request reloads the model.
- Minimal change: wrap `get_llm()` outputs with one model-key-scoped unload
  retry and temporary same-model pause.
- Ownership boundaries: deterministic code owns transient unload recovery; LLM
  stages own semantic generation; existing callers/workers own terminal
  failures.
- Rejected complexity: generic retries, crash retries, SDK retry changes, rate
  limiting, concurrency caps, probes, polling, fallback models, telemetry,
  feature flags, prompt edits, graph changes, and per-stage retry code.
- Evidence threshold: add rejected complexity only after a production failure
  proves this unload-only recovery is insufficient and the user approves a new
  plan.

## Agent Autonomy Boundaries

- The responsible agent may choose implementation mechanics only when they
  preserve the contracts in this plan.
- The responsible agent must not broaden the classifier or add retry categories
  without user approval.
- The responsible agent must treat edits outside the approved change surface as
  blocked.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or refactors.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Confirm the exact LM Studio unload error signature and record it in this
   plan before implementation.
2. Parent adds `tests/test_llm_reload_monitor.py` with focused fake-LLM tests
   for async retry success, retry failure waiter release, non-unload
   propagation, same-model wait, different-model non-wait, concurrent unload
   ownership, and sync retry success.
3. Parent runs `venv\Scripts\python.exe -m pytest tests/test_llm_reload_monitor.py -q`.
   Expected before implementation: fail for missing module or behavior.
4. Production-code subagent creates `llm_reload_monitor.py` and wires
   `utils.get_llm()`.
5. Parent reruns `venv\Scripts\python.exe -m pytest tests/test_llm_reload_monitor.py -q`.
   Expected after implementation: pass.
6. Parent runs `venv\Scripts\python.exe -m pytest tests/test_utils.py -q`.
   Expected after implementation: pass.
7. Parent runs the static grep in `Verification`.
8. Parent updates `docs/HOWTO.md`.
9. Parent runs `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q`.
   Expected after implementation: pass.
10. Parent runs independent code review, fixes in-scope findings, and reruns
    affected verification.

## Execution Model

- Parent agent owns orchestration, tests, verification, evidence, review
  remediation, lifecycle updates, and sign-off.
- Parent establishes the focused test contract before production code changes.
- Production-code subagent: exactly one native subagent after the focused test
  contract; owns production code only; closes after planned production changes.
- Independent code-review subagent: exactly one native subagent after planned
  verification passes; reviews only and reports findings.
- If native subagents are unavailable, stop before execution unless the user
  explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 0 - unload signature confirmed and plan approved
  - Verify/evidence/sign-off: signature recorded, `Status` changed to
    `in_progress`, confirmation source and approval note recorded,
    `Codex/2026-05-25`.
- [x] Stage 1 - focused test contract established
  - Verify/evidence/sign-off: focused test command fails before implementation
    for missing module or behavior; failing output recorded;
    `Codex/2026-05-25`.
- [x] Stage 2 - central monitor implemented and wired
  - Verify/evidence/sign-off: focused test command passes; changed files and
    output recorded; `Codex/2026-05-25`.
- [x] Stage 3 - utility behavior preserved
  - Verify/evidence/sign-off: `tests/test_utils.py` passes; output recorded;
    `Codex/2026-05-25`.
- [x] Stage 4 - no overbroad retry machinery added
  - Verify/evidence/sign-off: static grep has no forbidden matches introduced
    by this plan; grep output recorded; `Codex/2026-05-25`.
- [x] Stage 5 - operator note and deterministic regression signoff exception
  - Verify/evidence/sign-off: HOWTO note present; deterministic regression
    blocker recorded and accepted for signoff; doc diff and test output
    recorded; `Codex/2026-05-25`.
- [x] Stage 6 - independent code review complete
  - Verify/evidence/sign-off: no unresolved blockers, in-scope fixes applied,
    affected tests rerun, findings/fixes/risks recorded,
    `Codex/2026-05-25`.

## Verification

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests/test_llm_reload_monitor.py -q`
  - Before implementation: fails for missing module or missing behavior.
  - After implementation: passes.

### Utility Regression

- `venv\Scripts\python.exe -m pytest tests/test_utils.py -q`
  - Expected after implementation: passes.

### Static Grep

- `rg "with_retry|max_retries|rate_limiter|semaphore|/models|fallback" src/kazusa_ai_chatbot/llm_reload_monitor.py src/kazusa_ai_chatbot/utils.py`
  - Expected: no matches introduced by this plan.
  - If a match appears in new code, remove it unless the user approves a plan
    change.

### Deterministic Regression

- `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q`
  - Expected after implementation: passes.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- exact unload-only classifier and one-retry cap;
- model-keyed pause behavior, including concurrent unload races;
- sync and async behavior sharing one monitor state;
- no hidden generic retry, rate limiter, probe, fallback, prompt change, graph
  change, route change, or unrelated cleanup;
- focused tests, utility regression, static grep, deterministic regression, and
  execution evidence.

The parent may fix review findings only inside the approved change surface. If
a finding requires broader classification or new operational machinery, stop
and request approval.

## Acceptance Criteria

This plan is complete when:

- `get_llm()` returns monitored clients for route-specific chat models.
- Confirmed LM Studio unload errors receive exactly one same-request retry.
- Same-model calls wait only while the model key is in unload recovery.
- Concurrent unload races produce one reload owner for that model key.
- Different model keys do not wait.
- Non-unload errors are not retried by this monitor.
- Sync JSON repair `.invoke(...)` is covered.
- No prompt, graph, route, scheduler, adapter, database, event-log, or
  worker-specific retry behavior is changed.
- Focused tests, utility tests, static grep, user-accepted deterministic
  regression exception, and independent code review pass.

## Execution Evidence

- Unload signature confirmation: user confirmed
  `The model has crashed without additional information.` on 2026-05-25; plan
  moved to `in_progress`.
- Focused pre-implementation result:
  `venv\Scripts\python.exe -m pytest tests/test_llm_reload_monitor.py -q`
  failed during collection with `ModuleNotFoundError: No module named
  'kazusa_ai_chatbot.llm_reload_monitor'`.
- Focused post-implementation result:
  `venv\Scripts\python.exe -m pytest tests/test_llm_reload_monitor.py -q`
  passed with 7 focused monitor tests, including async retry, retry failure
  waiter release, non-unload propagation, same-model wait, different-model
  non-wait, concurrent unload ownership, and sync retry.
- Utility regression result:
  `venv\Scripts\python.exe -m pytest tests/test_utils.py -q` passed with 21
  tests and 3 deselected.
- Static grep result:
  `rg "with_retry|max_retries|rate_limiter|semaphore|/models|fallback" src/kazusa_ai_chatbot/llm_reload_monitor.py src/kazusa_ai_chatbot/utils.py`
  found only the existing `utils.py` JSON-repair docstring mention of
  `fallback`. Added-line diff grep for the same terms returned no introduced
  matches.
- Deterministic regression result:
  `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q`
  is blocked by 16 collection-time import errors in existing tests importing
  `scripts.*`. Python resolves the root `scripts` package, while the requested
  modules live under `src/scripts`. No `scripts` files were changed by this
  plan.
- Independent code review result:
  Review subagent `019e5d4d-ab4e-7190-a28d-d987db0fcbf7` reported no blocking
  or important findings. Residual risks: no live LM Studio smoke was run,
  no explicit sync-owner/async-waiter cross-mode race test was added, and the
  broad deterministic regression remains blocked by the pre-existing
  `scripts` package shadowing issue.
- Final scoped verification:
  `venv\Scripts\python.exe -m pytest tests/test_llm_reload_monitor.py tests/test_utils.py -q`
  passed with 28 tests and 3 deselected. `git diff --check` passed. The static
  forbidden-term grep still found only the existing `utils.py` JSON-repair
  docstring mention of `fallback`.
- Signoff exception:
  User accepted signoff on 2026-05-25 with the broad deterministic regression
  blocked by the unrelated `scripts` import collection issue.
