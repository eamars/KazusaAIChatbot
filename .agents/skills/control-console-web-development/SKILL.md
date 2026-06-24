---
name: control-console-web-development
description: Use when planning, modifying, debugging, testing, or validating the Kazusa web control console frontend, including static HTML/CSS/JS pages, FastAPI control-console routes, cognition/prompt debug views, UI placement, Playwright or browser validation, screenshot signoff, stale Chrome in-memory JavaScript/cache issues, and control-console framework rules.
---

# Control Console Web Development

Use this skill for Kazusa control-console web frontend work and adjacent
control-console API behavior. Treat the console as an operational debugging
surface for the character brain, not a separate product shell.

## Entry Contract

1. State the target workspace, served checkout, port, and URL before acting.
2. Never edit the production checkout directly. Apply implementation changes in
   the dev workspace unless the user explicitly gives a different target.
3. Before substantive edits, inspect `git status --short`, `README.md`,
   `docs/HOWTO.md`, relevant subsystem READMEs, source files, tests, and active
   development plans.
4. Do not read `.env` unless the user explicitly asks for environment
   inspection.
5. If work is plan-driven, use the `development-plan` skill and read
   `development_plans/README.md`.
6. If tests are added, changed, or run, use `test-style-and-execution`.
7. If Python production code is edited, use `py-style`.
8. Communicate before meaningful action, especially before browser automation,
   production-like service inspection, or file edits.

## Browser Availability

- Use the in-app Browser first when it is available, and read the
  `browser:control-in-app-browser` instructions before using it.
- For rendered frontend debugging, also use
  `build-web-apps:frontend-testing-debugging` when available.
- If the in-app Browser is unavailable or reports an error such as `iab`, record
  that exact reason and use Playwright with installed/system Chrome.
- Do not claim "Browser validated" when validation used Playwright fallback.
  Say which browser path was used.
- Browser validation must capture page identity, URL, logged-in/session state,
  visible page, console errors, page errors, screenshots when required, and at
  least one interaction proof for each affected page.
- Prefer loopback URLs such as `127.0.0.1` or `localhost`; document the exact
  URL and token/session assumptions used.

## Chrome Stale-JavaScript Pitfall

Chrome can keep an existing tab's in-memory JavaScript after the backend app or
dev server restarts. A fresh Playwright context may load current assets while
the user's visible Chrome tab still runs older functions.

When the user reports a frontend error that cannot be reproduced in a clean
context, inspect the active tab state before changing code:

```js
({
  hasSetHtml: typeof setHtml,
  refreshCalendarHasOldTable:
    typeof refreshCalendar === "function" &&
    String(refreshCalendar).includes("#calendar-table"),
  refreshBackgroundHasOldTable:
    typeof refreshBackground === "function" &&
    String(refreshBackground).includes("#background-table"),
  ids: [
    "calendar-table",
    "calendar-prompt-runs-table",
    "calendar-schedules-table",
    "calendar-due-runs-table",
    "background-table",
    "background-result-ready-table",
    "background-job-queue-table",
    "background-worker-events-table",
  ].map((id) => [id, Boolean(document.getElementById(id))]),
  scripts: [...document.scripts].map((s) => s.src),
  resources: performance.getEntriesByType("resource")
    .filter((entry) => entry.name.includes("console"))
    .map((entry) => entry.name),
})
```

Use this incident as the canonical pattern: the page HTML had new split-panel
tables, but `refreshCalendar()` still referenced `#calendar-table`; setting
`innerHTML` on a missing old element raised `Cannot set properties of null`.

Hard rules:

- Reproduce the user's browser-visible failure first when the user demands
  reproduction. Do not touch code until reproduction is proven or the mismatch
  is explained with evidence.
- A clean browser profile is not sufficient to clear a stale-tab report.
- Test the same active tab when possible; otherwise ask the user to run a small
  diagnostic snippet and compare the loaded function bodies and DOM IDs.
- Force reload with `Ctrl+Shift+R`, disable cache in DevTools, or add a
  development cache-busting/version query when validating changed static assets.
- For persistent local tooling, consider no-cache headers or asset versioning
  for console static files so server restarts cannot hide old JavaScript.
- Do not call a fix done until the stale-tab path and fresh-context path are
  both understood.

## Framework Rules

- The control console is a separate top-level Python/FastAPI service. Do not
  mount it into the brain service for UI convenience.
- The frontend is buildless static HTML/CSS/JavaScript served by FastAPI.
- Do not introduce Node.js, npm, pnpm, yarn, React, Vue, Vite, Webpack, or
  Tailwind as the fundamental console stack unless a new approved plan says so.
- Use shadcn component-family anatomy in static HTML/CSS/JS.
- Prefer the existing console widgets and helpers: Sidebar, Button, Card,
  Badge, Table, Input, Select, Textarea, Separator, ScrollArea, Field/Form
  grouping, dialog/sheet patterns, notice areas, and existing render helpers.
- Do not create custom widgets when an existing framework/widget pattern can
  carry the information.
- Use a snapshot plus SSE data model with bounded detail GETs. Do not add broad
  polling, full-page auto-refresh, WebSocket, or multiple independent live
  streams without an approved plan.
- Keep `/chat`, cognition, RAG, scheduler, memory promotion, adapter transport,
  and prompt wording unchanged unless they are explicitly in scope.
- Route handlers should use established repository/domain helpers, not raw
  MongoDB clients, unless the local subsystem already defines that boundary.

## UI Rules

- Align with the current console color scheme, spacing, typography, border
  radius, table density, badge treatment, card anatomy, and interaction style.
- Build operational screens, not landing pages or hero sections.
- Avoid new palettes, decorative gradients, custom icons, and parallel visual
  systems.
- Keep pages dense but readable. Use cards for individual bounded panels, not
  nested page sections.
- Tables and prompt/debug text must be bounded, wrapped, and responsive. Avoid
  accidental horizontal overflow, overlap, truncation, and clipped controls.
- Use in-page ARIA live notices for errors; do not use `alert()`.
- Async buttons must disable while pending and show loading/completion/error
  state through existing notice/status patterns.
- Disabled or unavailable controls must show a clear reason.
- Do not present dummy/static placeholders as working capability. Mark partial,
  unavailable, or needs-input states explicitly.
- Every visible control added or changed must be clicked or otherwise exercised
  in rendered validation.

## Cognition Debug Views

- The web console must display the same bounded window of data that is fed into
  the cognition chain when the page is meant to debug cognition behavior.
- Prompt View panels must call the production projection/window function when
  such a function exists. Do not duplicate production prompt-window projection
  logic in frontend-specific code.
- Supporting panels may use console-specific formatting, but their extraction
  must be bounded, redacted, and labeled as supporting context rather than
  prompt input.
- Missing scope must return a `needs_input` state. Do not infer or reuse a
  previous browser selection silently.
- `calendar_schedules`, `background_work_jobs`,
  `conversation_episode_state`, `global_character_growth_runs`, and the current
  carry-over `internal_monologue_residue_state` are useful debug surfaces when
  mapped through bounded projections.
- `event_log_snapshots` are debugging internals and stay out of product console
  scope unless a new approved plan explicitly includes them.

## Security And Redaction

- Require authentication and CSRF protection for console APIs except static
  files and minimal session/login endpoints.
- Never expose secrets, tokens, raw prompts, embeddings, raw messages, full
  conversation rows, full memory bodies, raw reflection output, raw LLM output,
  tool arguments, idempotency keys, unrestricted source scopes, or unbounded
  lookup tables.
- Use bounded summaries, stable counts, timestamps, statuses, and redacted
  snippets instead of raw database documents.
- Keep real-data screenshots and traces outside committed source unless the
  user explicitly requests an artifact and sensitive data is reviewed.

## Debugging Workflow

1. Reproduce the exact user-visible failure first, or explain the environment
   mismatch with evidence.
2. Capture the failing page, active URL, selected tab, page notice text,
   console/page errors, function identity, relevant DOM IDs, network status,
   and loaded static asset URLs.
3. Identify whether the failure is in stale browser state, DOM contract,
   async timing, API contract, auth/CSRF/session state, data shape, or styling.
4. Write or update a failing deterministic or browser test where practical
   before implementing the fix.
5. Fix the smallest owning boundary. Do not patch symptoms in unrelated layers.
6. Verify fresh context, stale/hard-reload context when relevant, all affected
   tabs, console/page error health, and screenshots when UI changed.
7. Clean up temporary servers, ports, browser profiles, screenshots, and traces
   that are not intended artifacts.

## Completion Gate

Do not claim the frontend work is fixed or complete until:

- The original issue was reproduced or the reproduction mismatch was proven.
- The fix was applied in the correct workspace.
- Static checks and relevant tests passed.
- Rendered browser validation covered the affected page or pages with real data
  when the feature is a debug/observability surface.
- Screenshots demonstrate the changed UI when the user asked for signoff or the
  plan requires visual evidence.
- No unexpected console errors, page errors, warning logs, alerts, horizontal
  overflow, stale button states, or sensitive field leaks remain in the accepted
  flow.
