# Control Console V3 Cutover Startup And Configuration Cleanup

## Document control

- **Status:** completed on 2026-08-22.
- **Class:** production startup incident and incomplete V3 cutover cleanup.
- **Owner:** root parent; the reusable gpt-5.6-luna worker owns the surgical
  implementation and verification handoff.

## Observed failure

`kazusa-control-console --host 127.0.0.1 --port 8765` exits before binding its
port. `control_console.kazusa_client` imports `time_boundary`, which imports the
complete Brain config; eager V3 route validation then raises because the
console process has no `COGNITION_V3_CHAIN_LLM_CONTEXT_WINDOW_TOKENS`.

The deployed `.env` also retains the twelve removed V2 per-stage cognition
route bundles and has no canonical V3 chain/sidecar bundle.

## Change boundary

1. Make the shared time boundary independent of complete Brain route loading
   while preserving the configured character timezone as one canonical value.
2. Migrate `.env` to the V3 chain/sidecar bundle, retain active shared routes,
   and remove the selector plus all obsolete V2 per-stage route keys.
3. Remove current control-console and operator-document route entries that
   expose the deleted bundles. Delete V2-specific test cases instead of
   preserving their old configuration surface.
4. Verify the actual console entrypoint, authenticated API/bootstrap path,
   rendered browser shell, Brain startup/config import, and one real debug-chat
   end-to-end turn. Use targeted runtime/browser checks rather than broad unit
   churn.

## Acceptance

- The console binds `127.0.0.1:8765` without requiring Brain model settings.
- `.env` has a complete valid V3 chain bundle and no V2 selector/per-stage
  cognition route keys; secrets are never printed.
- The Brain imports/starts with the migrated environment.
- The console route catalog shows only active shared/V3 cognition routes.
- A browser can authenticate, load Overview and Services without console/page
  errors, and exercise the affected route surface.
- One debug-chat request completes through Brain cognition, dialog, response,
  trace, and persistence inspection, or an exact external dependency blocker
  is reported with all in-process boundaries proven.

## Closure evidence

- Brain `/health` and both console roots returned HTTP 200 after the final
  restart.
- The actual control-console entrypoint, authenticated twelve-page browser
  signoff, protected trace handoff, and one no-memory debug-chat path completed.
- Browser console errors, page errors, and HTTP failures were all zero.
- The live cognition chain reported `completed` with a nonzero step count and
  returned a coherent visible acknowledgement.
- `.env` contains the complete chain bundle and shared control-plane trace
  secret, with zero engine-selector or removed per-stage cognition route keys.
- The active Services route catalog contains sixteen routes and exposes only
  the chain and sidecar cognition routes.
- Current operator surfaces use canonical agentic-cognition labels; V2-only
  route, graph, replay, documentation, and plan tests were deleted instead of
  migrated.

The observed `partial` graph / `accepted-degraded` appraisal result is outside
this startup-and-cutover recovery scope. It is retained as the initiating
evidence for a separate cognition first-pass quality investigation.
