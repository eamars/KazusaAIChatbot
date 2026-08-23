# Cognition Full Protected Trace Capture Bugfix Plan

## Summary

- **Goal:** Make successful live cognition and L3 model calls inspectable in
  the existing protected trace export, including raw prompt messages, raw
  responses, parsed outputs, contract status, and stage identity when
  `LLM_TRACE_CAPTURE_MODE=full`.
- **Status:** `completed` on 2026-08-23 after focused checks, production
  restart, and a full protected real-LLM export.
- **Scope:** Existing protected trace storage, Cognition V3 A1/A2/G/P call
  boundaries, text/visual surface call boundaries, the deployment capture-mode
  setting, and narrow trace verification.
- **Out of scope:** Cognition semantics, prompts, goal schema, dialog wording,
  validators, repair policy, public/control-console exposure, retention policy,
  and V2 compatibility.

## Observed Failure

`llmtrace_05655f590641449596c0d035cf8678d4` persisted only relevance,
decontextualization, dialog, and memory-lifecycle rows. All rows had empty raw
messages, raw output, and parsed output because deployment capture mode was
`metadata`. Successful A1/A2/G/P and L3 content/preference calls were not
persisted at all: they existed only in invocation-local diagnostics or a
failure-only capsule that is discarded on success.

The exporter is already capable of returning every persisted step. The gap is
at the producing call boundaries, not the export format.

## Design

1. Reuse `kazusa_ai_chatbot.llm_tracing.record_llm_trace_step(...)` for every
   A1/A2/G/P attempt after the provider returns, including contract-fault
   dispositions.
2. Keep the existing invocation-local protected records for direct test and
   failure diagnosis.
3. Route every content-plan, preference, and enabled visual-surface attempt
   through the same trace recorder. Replace the current failure-capsule-only
   call so one model attempt is recorded once.
4. Preserve best-effort behavior: trace storage failure cannot fail cognition,
   surface planning, or dialog.
5. Set the deployment `LLM_TRACE_CAPTURE_MODE` to `full`. Retain the existing
   protected TTL and private collection; expose none of this through the public
   console.
6. Leave existing relevance, decontextualization, dialog, RAG, and lifecycle
   trace call sites unchanged.

## Owned Files

- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
- `src/kazusa_ai_chatbot/cognition_shared/surface_stages.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
- focused existing trace/cognition/surface tests only
- `.env` exact `LLM_TRACE_CAPTURE_MODE` deployment line
- relevant trace/cognition README text if the old success-discard statement
  becomes false

## Acceptance

- A deterministic focused test observes one persisted trace call for each of
  A1, A2, G, and P, with exact stage identity and parsed result.
- A focused surface test observes successful content/preference attempts in
  the protected trace path without duplicate failure-capsule attempts.
- Existing cognition first-pass call count remains exactly four.
- Existing dialog trace behavior remains unchanged.
- One isolated real-LLM debug turn with full capture produces an export that
  contains raw A1/A2/G/P, L3 content/preference, and dialog prompt/output rows.
- Trace write failure remains nonfatal.
- No tests are added for development-plan or documentation prose.

## Verification Sequence

1. Run the smallest deterministic trace tests.
2. Run existing focused cognition and surface contract tests affected by the
   changed call sites.
3. Restart the production brain with full capture.
4. Submit one `no_remember=true` unknown-input case.
5. Export and inspect the trace manually; record the artifact under
   `test_artifacts/diagnostics/`.
6. Restart the production brain through the existing control path so the
   deployment setting takes effect, then verify health.

## Completion Evidence

- `LLM_TRACE_CAPTURE_MODE=full` is active in the deployment environment.
- The production brain restarted healthy on the project venv entrypoint.
- The two affected unit modules passed: 17 tests before the L3 context fix;
  the five focused L3 tests passed after it. Ruff, `py_compile`, and
  `git diff --check` passed.
- The first export,
  `test_artifacts/diagnostics/llm_trace_llmtrace_9474437979d94d95a18f3a121e2a8d22_full_20260823.json`,
  proved full raw A1/A2/G/P capture and exposed a missing L3 trace-context
  handoff.
- The corrected export,
  `test_artifacts/diagnostics/llm_trace_llmtrace_23531e49a2994b74b4fbf50f0475f3de_full_20260823.json`,
  contains raw prompts, responses, parsed products, and status for A1/A2/G/P,
  every content/preference attempt, dialog, and the protected partial-failure
  capsule.
- Public and control-console contracts were unchanged; the console remains
  agnostic to internal stage names and trace storage.

## Rollback

Revert the three trace call-site/context changes and restore the deployment
capture mode.
No cognition schema or persisted cognition state is changed by this plan.
