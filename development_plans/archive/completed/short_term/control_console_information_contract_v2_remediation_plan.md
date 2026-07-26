# Control Console Information Contract V2 Remediation Plan

## Summary

- Goal: make every control-console page show useful, correctly sourced operator
  information, with one canonical page owner for each datum and native
  Cognition Core V2 character/user state.
- Plan class: large control-console information-contract remediation.
- Status: completed — 2026-07-27.
- Scope: all twelve pages served by the control console: Overview, Services,
  Live logs, Debug chat, Event monitor, Character, Users, Groups, Calendar,
  Background work, Health/cache, and Audit.
- Primary failure class: data-contract correctness and information ownership.
  This is not a visual restyling plan.
- Highest-severity defects:
  - the Users page reads removed V1 affinity and relationship fields while all
    five inspected user profiles store relationship state under
    `cognition_state.v2.relationship`;
  - the Character page omits profile and V2 cognition fields and still asks for
    legacy latest-state fields that cannot be returned by the current database
    projection;
  - the Calendar page queries only due runs, hiding the four inspected
    completed runs;
  - the Audit page renders object targets as `[object Object]` and asks for a
    `status` field that is absent from all one hundred inspected audit rows;
  - page-level availability can report `available` when only one of several
    required panels loaded successfully.
- Overall cutover strategy: one big-bang console contract update across
  repository projections, HTTP responses, page renderers, tests, and console
  documentation. No V1 aliases, dual reads, fallback mappers, or parallel page
  contracts remain after cutover.
- Data migration: none. The plan reads existing canonical data through its
  owning repository modules and changes only console projections.
- LLM budget: zero calls before and zero calls after. No model-generated
  summaries are introduced.
- Styling boundary: preserve the current stylesheet, page layout family,
  typography, and interaction model. Content elements may be removed, renamed,
  or populated only where the information contract requires it.
- Production authority: granted by the user's `Execute the plan` instruction
  for the scoped code changes; deployment, database writes, and migration
  remain outside scope.
- Plan review: parent-agent self-review only, as explicitly required by the
  user. No plan-review subagent is used.

## Context

The audit followed the current control-console ownership boundary:

```text
authoritative domain repository
  -> control-console safe projection
  -> HTTP response
  -> page-specific renderer
  -> authenticated operator review
```

The control console currently breaks that boundary in three ways:

1. Several projections still read removed V1 fields instead of native V2 state.
2. Generic panel rendering exposes internal projection metadata and storage
   machinery instead of the semantic result an operator needs.
3. The same state is repeated across Overview and owner pages without a defined
   aggregate/detail relationship.

The following evidence was gathered before writing this plan:

- the current source, tests, root documentation, console documentation, and
  completed predecessor plans were read;
- all twelve signed-out page layouts were opened from the running checkout at
  `http://localhost:8764/` through regular Playwright after the in-app Browser
  reported `No browser is available`;
- populated browser inspection could not be completed because the running
  console requires an operator token;
- approved read-only database export scripts produced bounded diagnostic
  artifacts under
  `test_artifacts/control_console_audit_20260726/`;
- live brain health and runtime-status endpoints were inspected read-only;
- the local bounded audit log was inspected through aggregate counts rather
  than copied into this plan.

Authenticated, populated browser evidence is therefore a mandatory acceptance
gate. Source inspection or signed-out screenshots cannot satisfy that gate.

### Current bounded data evidence

| Source | Inspected result | Contract implication |
|---|---:|---|
| Character state | 1 row | The page has a real V2 cognition source. |
| User profiles | 5 rows; 5 contain `cognition_state.v2` | V2 relationship display is mandatory. |
| Interaction style images | 0 rows | Empty style state must not make Users or Groups look unavailable. |
| Calendar schedules | 0 rows | A real empty schedule state needs an explicit empty message. |
| Calendar runs | 4 rows; all completed reflection runs | Recent completed runs must be queryable and visible. |
| Background jobs | 0 rows | Worker activity must remain useful when the queue is empty. |
| Group review windows | 2 rows | Group review state exists independently of style images. |
| Group conversation metadata | 100 bounded rows across 8 scopes | Groups need a discoverable directory sourced from activity metadata. |
| Application events | 25 rows | Structured events contain useful operational evidence. |
| Growth traits/runs | 0/0 rows | Growth needs a truthful empty state; internal run scaffolding adds no value. |
| Cache agents | 2 agents | Hit count, miss count, and hit rate should be visible once. |
| Local audit | 100 bounded rows | Object targets, paired lifecycle rows, and view-event noise require projection. |

No raw conversation content, protected prompt content, credentials, or
unbounded database rows are included in this plan.

## Mandatory Skills

Implementation and verification must apply:

- `development-plan` for lifecycle, cutover, and review control;
- `control-console-web-development` for console contracts, browser validation,
  current-checkout verification, and screenshot evidence;
- `database-data-pull` for any additional read-only database evidence;
- `py-style` before modifying Python;
- `test-style-and-execution` before changing or running tests;
- `python-venv` before dependency or environment work.

`local-llm-architecture` and `debug-llm` are outside this scope because the
target contains no prompt, model-routing, cognition, or LLM-output change.

## Mandatory Rules

1. Treat persisted domain state as authoritative; do not reconstruct semantic
   state from UI labels, event text, or browser-local data.
2. Read V2 character and user cognition through the canonical schema and
   repository owners.
3. Keep adapter, brain, database, and domain behavior unchanged.
4. Keep raw messages, protected prompts, evidence references, credentials,
   tokens, internal alias identifiers, and unrestricted traces out of console
   responses.
5. Bound every list, directory, aggregation, event query, and audit query.
6. Distinguish `empty`, `needs_input`, `partial`, and `unavailable`; do not use
   one state as a substitute for another.
7. Give each datum one canonical owner page. Overview may show only a compact
   aggregate or exception that links to that owner.
8. Remove visible `panel_contract`, `projection_owner`, raw scope summaries,
   lease internals, and internal run identifiers from default page content.
9. Preserve useful diagnostics on their proper detail surface instead of
   duplicating them on summary pages.
10. Do not infer severity from stdout/stderr stream selection.
11. Do not fabricate group cognition, relationship, progress, guidance, or
    growth semantics when no authoritative source exists.
12. Do not introduce a generic page-builder framework, new cache, new
    collection, or new background worker.
13. Execute the console response changes as a big-bang contract update. Delete
    replaced V1 projectors and update every caller and test in the same scope.
14. Verify populated pages in an authenticated browser against the exact
    checkout and port before sign-off.
15. Record current data counts at verification time; acceptance depends on the
    source result and not on the July 26 sample counts remaining unchanged.
16. Preserve the current visual styling. Image generation and visual redesign
    are outside this plan.
17. Reread this complete plan after context compaction and after each major
    checklist-stage sign-off before continuing.
18. Record verification and parent code-review results in Execution
    Evidence before lifecycle completion.

## Must Do

### P0 — Correctness blockers

1. Replace every V1 user affinity/relationship projection with
   `UserCognitionStateV2.relationship`.
2. Project all operator-relevant `CharacterProfileDoc` and
   `CharacterCognitionStateV2` fields.
3. Add bounded user and group directories so pages with existing data do not
   begin as blank exact-ID forms.
4. Query recent calendar run history as well as due/running state.
5. Repair audit target and outcome projection, and collapse request/completion
   pairs into one operator action.
6. Expose nested worker liveness, semantic worker error level, and complete
   Cache2 metrics on Health/cache.
7. Make combined page status accurately report partial source failure.

### P1 — Information ownership and usefulness

1. Remove duplicate brain/service/route state from Overview.
2. Remove the meaningless Overview `Event stream` capability label and internal
   stream/CSRF implementation details.
3. Make Event monitor a structured application-event surface rather than a
   second Logs and Audit page.
4. Replace generic Calendar, Background work, and Audit metadata cards with
   real counts, outcomes, and semantic state.
5. Replace Growth Runs Audit machinery with growth meaning, accepted changes,
   and prompt visibility when those records exist.
6. Remove false or dead capabilities, including the Character memory claim and
   hardcoded empty Group progress/guidance panels.

### P2 — Clarity and provenance

1. Label Debug chat history as current-browser-session history.
2. Move static request-contract and visible-field descriptions from primary
   data regions into existing help/documentation text.
3. Show a concise source label and last-updated time where operator
   interpretation depends on freshness.
4. Keep machine identifiers available only in a bounded detail disclosure when
   they are needed for correlation.

## Deferred

- visual redesign, CSS modernization, new layout systems, and image-generated
  mockups;
- database schema changes or migrations;
- new group-level cognition or affinity state;
- model-generated summaries of logs, events, memories, growth, or audit rows;
- global search, arbitrary database browsing, or raw-record viewers;
- changes to cognition, dialog, RAG, persistence, scheduler execution, or
  adapter delivery;
- production deployment and operator-token provisioning.

## Cutover Policy

The console projection/API/renderer contract changes use a big-bang cutover:

1. Add the bounded domain-owner query helpers required by the frozen target
   contracts.
2. Replace the repository projections and HTTP responses.
3. Replace all affected JavaScript renderers and remove obsolete elements.
4. Replace V1 fixtures and assertions with V2 fixtures and semantic assertions.
5. Update console documentation.
6. Run deterministic tests and authenticated page verification on the same
   revision.

The following are explicitly prohibited:

- returning old and new panel names together;
- falling back from missing V2 state to V1 affinity fields;
- keeping legacy projectors behind aliases;
- accepting either old or new shapes in tests;
- deploying the server and static client from different revisions.

Rollback is revision-level: restore the preceding console application revision.
No database rollback is required because this plan performs no writes.

## Target State

### Page ownership map

| Page | Canonical responsibility | Information owned elsewhere |
|---|---|---|
| Overview | Aggregate exceptions, degraded dependencies, recent failures/changes, cognition graph entry points | Full service state, logs, detailed events, cache metrics, and audit history |
| Services | Managed-process lifecycle and brain model-route configuration | Raw logs, runtime worker health, and audit history |
| Live logs | Raw bounded stdout/stderr per managed service and its stream connection state | Structured application events and operator actions |
| Debug chat | One debug request/reply and its cognition graph | Persisted user history, raw logs, and general runtime events |
| Event monitor | Structured application runtime events | Raw process streams and full operator audit |
| Character | Character profile, V2 cognition, self-image, growth, and carry-over | Service health and generic event history |
| Users | User identity projection, V2 relationship/cognition, memory, style, thread progress, and carry-over | Group aggregate state |
| Groups | Group activity directory, review state, style, carry-over, and participant progress | Fabricated group cognition or relationship state |
| Calendar | Schedule/run state and recent run outcomes | General worker liveness and raw event history |
| Background work | Job queue/outcomes, delivery readiness, and worker activity | Raw worker events and scheduler configuration |
| Health/cache | Database/scheduler readiness, worker liveness, semantic error level, and Cache2 performance | Process lifecycle and operator actions |
| Audit | Collapsed operator/security actions and outcomes | Application events and raw process logs |

### Page-by-page findings and required information

#### Overview

- **Critical defects:** brain status is repeated in summary, Brain service, and managed-service content; `Event stream` is a static capability/path rather than observed health; Runtime Summary repeats it; Visible workflows is a DOM count; CSRF, guardrail, workflow, and safety prose occupy the operational surface; service/route detail duplicates Services.
- **Target/source:** show one managed-service aggregate, one internal-readiness aggregate, bounded recent failures and state-changing actions, and cognition graph entry points; link to the owner pages for detail.
- **Remove/move:** remove Event stream, workflow counts/lists, CSRF/stream paths, guardrail/safety cards, per-service state, and model-route detail; keep stream connection state only on Live logs.

#### Services

- **Critical defects:** Brain state is repeated on Overview; each route tile repeats current model/source/family in its editor; PID, version, and dependency identifiers compete with lifecycle state and action-block reasons.
- **Target/source:** show one authoritative service state/action/block reason from the service manager and one current value per model route from canonical route endpoints; put PID/version/dependencies in bounded detail and resulting actions on Audit.
- **Remove/move:** remove service/route duplicates from Overview and remove repeated current values from edit controls.

#### Live logs

- **Critical defects:** the same process streams appear in Event monitor, and stderr is treated as severity `error` even though it is only a stream.
- **Target/source:** remain the sole owner of bounded service-manager stdout/stderr, showing service, stream, available timestamp, literal text, and connection state.
- **Remove/move:** remove process streams from Event monitor and preserve stderr without manufactured severity.

#### Debug chat

- **Critical defects:** Request Contract and Visible Fields are static documentation presented as changing data; History is browser-local but lacks current-session provenance.
- **Target/source:** retain request controls, current response/error, cognition graph, and browser-local history; label History `current browser session`.
- **Remove/move:** move contract/field prose to concise help text outside the result region.

#### Event monitor

- **Critical defects:** raw process logs and audit rows duplicate their owner pages; Local/Tail/Log cards are static words; the table drops projected severity, duration, errors, and correlation metadata; the UI omits supported filters.
- **Target/source:** show structured application events with dynamic severity/status/component counts, time/severity/component/event/outcome/duration/error columns, filters for component/event/severity/time/correlation, and identifiers in row detail.
- **Remove/move:** remove `process`; keep audit detail on Audit; define `all` as all structured application-event sources, with only optional collapsed action correlation.

#### Character

- **Critical defects:** profile projection omits tone, speech patterns, backstory, boundary profile, and linguistic texture; Latest State requests legacy mood/vibe/reflection fields; full V2 drives, standards, meaning, goals, threats, events, gaps, and affects are absent; memory is falsely claimed; growth is duplicated; Growth Runs Audit shows machinery while omitting summary, accepted changes, trait updates, and guidance.
- **Target/source:** `profile` contains all safe `CharacterProfileDoc` fields; `cognition_state` contains native `CharacterCognitionStateV2`; `self_image` uses its current safe source; `growth` combines traits, prompt visibility, latest semantic summary/changes/updates/guidance/maturity/review note; `carry_over` contains bounded promoted continuity.
- **Remove/move:** delete legacy Latest State and Background Learning projectors, the unsupported memory claim/panel, duplicate growth panels, and default run/source/execution identifiers.

#### Users

- **Critical defects:** the page starts with an undiscoverable exact-ID form; profile returns only `updated_at`; relationship reads removed V1 summary/status/affinity fields; all five inspected profiles store V2 relationship under `cognition_state.v2`; tests encode V1 and can pass when relationship is broken; remaining V2 cognition is absent.
- **Target/source:** add a bounded recent/known-user directory from the canonical profile repository; show safe account labels/display names and alias count; show every V2 relationship axis with exact value, canonical band, updated time, and evidence count; show V2 goals/threats/events/gaps/affects plus memory, style, thread progress, and carry-over with owner/thread provenance.
- **Contract:** add `GET /api/entities/users?limit=<bounded>` sorted by recent profile/cognition update with deterministic tie-breaker; use `GET /api/entities/users/{platform}/{platform_user_id}` for detail; exclude global IDs, alias IDs, and evidence references.

#### Groups

- **Critical defects:** the page starts with platform/channel inputs and cannot discover groups; the live test selects only from style rows, which are empty despite eight activity scopes and two review rows; progress/guidance are hardcoded empty; no group V2 cognition exists.
- **Target/source:** add a bounded platform/channel directory from conversation metadata with last activity, message count, and participant count; show selected activity, latest review status/window/time/skip reason, optional style, and sourced carry-over/participant progress without raw messages.
- **Contract:** add `GET /api/entities/groups?limit=<bounded>` and owner-module conversation/review helpers; retain the detail route; delete hardcoded panels and add no fabricated group cognition, affinity, or relationship.

#### Calendar

- **Critical defects:** Source/Actions/Excluded are static metadata; due-run queries hide completed/failed/skipped history; IDs, lease owner/expiry, and max attempts dominate cards; schedules, execution, and cognition visibility are mixed.
- **Target/source:** show active/upcoming/overdue/running/recent-completed/failed/skipped counts, bounded schedules and recent runs, one row per run with kind/times/outcome/safe result summary, and a separate scoped cognition-visibility diagnostic.
- **Contract:** add bounded `list_recent_calendar_runs`; retain due-run queries only for scheduler execution; keep lease/attempt internals in detail.

#### Background work

- **Critical defects:** Source/Actions/Excluded are static metadata; an empty queue looks valueless despite worker events; tick events become repetitive cards; job and result-ready prompt can duplicate work; safe worker counts/status are discarded.
- **Target/source:** show queue/running/completed/failed/delivery-ready/deferred counts, one canonical row per job, one latest/aggregate worker summary, and bounded recent errors; attach delivery detail to its job.
- **Contract:** whitelist existing safe processed/succeeded/failed/skipped/deferred/run-kind/worker-name event fields; exclude raw payloads/prompts/run IDs/lease machinery.

#### Health/cache

- **Critical defects:** primitive top-level rendering drops nested worker state and semantic descriptors; readiness is repeated; Cache2 hit rate is omitted; four current workers have hidden enabled/task-alive/last-status values.
- **Target/source:** show one DB/scheduler readiness summary, semantic `worker_error_level`, one row per worker with enabled/liveness/last status/time, and one row per cache agent with hits/misses/hit rate/total.
- **Remove/move:** remove duplicate status cards/rows and keep configuration detail only when it changes interpretation.

#### Audit

- **Critical defects:** object targets render `[object Object]`; Status reads an absent field (0 of 100 inspected rows contain it); request/completion pairs duplicate actions; 71 of 100 rows are view events; opening Audit adds `audit_view`; metric cards are static.
- **Target/source:** group exact `request_id` pairs into one action, derive outcome through an explicit terminal-event/new-state map, humanize targets from whitelisted keys, default to changes/failures, summarize views separately, and filter by category/event/service/operator/outcome/request/time.
- **Contract:** retain bounded raw detail for investigation but keep it out of the default action list; never infer outcome from nonexistent `status`.

## Contracts and Data Shapes

### Shared panel state

Every data-bearing panel uses:

```text
status: available | empty | needs_input | partial | unavailable
summary: bounded human-readable semantic summary
items: bounded safe projected items
updated_at: newest authoritative source timestamp or null
reason: bounded operator-safe reason for needs_input/partial/unavailable
```

`panel_contract`, `projection_owner`, internal `scope_order`, and raw
`scope_summary` may exist in server diagnostics but are not visible page data.

Page status is calculated as follows:

1. `needs_input` only when a detail selection is required and the unscoped
   directory/summary loaded successfully.
2. `empty` when every required source query succeeded and every required
   data-bearing panel is empty.
3. `available` when every required source query succeeded and at least one
   required data-bearing panel has data.
4. `partial` when at least one required source succeeded and at least one
   required source failed.
5. `unavailable` when all required sources failed.

Optional-panel failure is shown on that panel and does not make a truthful
required-data page unavailable.

### Frozen endpoint panel names

| Endpoint | Target panels/result |
|---|---|
| `GET /api/overview` | `service_summary`, `internal_readiness`, `recent_failures`, `recent_changes`, `cognition_graphs` |
| `GET /api/events` | structured `events`, dynamic `facets`, bounded query metadata |
| `GET /api/entities/character` | `profile`, `cognition_state`, `self_image`, `growth`, `carry_over` |
| `GET /api/entities/users` | bounded user directory |
| `GET /api/entities/users/{platform}/{platform_user_id}` | `profile`, `relationship`, `cognition_state`, `memory`, `style`, `conversation_progress`, `carry_over` |
| `GET /api/entities/groups` | bounded group directory |
| `GET /api/entities/groups/{platform}/{channel_id}` | `activity`, `review`, `style`, `carry_over`, `participant_progress` |
| `GET /api/lookups/calendar` | `summary`, `schedules`, `runs`, `cognition_visibility` |
| `GET /api/lookups/background-work` | `summary`, `jobs`, `worker_activity`, `errors`, `delivery_detail` |
| health/runtime endpoints | `readiness`, `workers`, `cache_agents` in the console projection |
| `GET /api/audit` | collapsed `actions`, `view_summary`, dynamic `facets` |

Replaced panel names and legacy field aliases are removed in the same change.

### Relationship representation

The Users page shows the stored numeric V2 values and the canonical semantic
bands produced by `project_relationship_context()`. It does not calculate a
replacement aggregate affinity score. Signed axes retain their
`-100..100` meaning; unsigned axes retain their `0..100` meaning.

### Empty-state contract

An empty source names what was queried and what the result means:

- `No schedules are configured` is an empty result.
- `No background jobs are queued or recently completed` is an empty result.
- `No style image has been learned for this user/group` is an empty optional
  result.
- `Select a user from the directory` is `needs_input`.
- `Relationship source failed: <safe reason>` is `partial` or `unavailable`,
  never empty.

## Design Decisions

1. Exact V2 relationship axes remain visible because operators are reviewing
   state, while semantic bands make those values readable.
2. Overview is an exception aggregate, not a second copy of every owner page.
3. Raw process output remains valuable and belongs only to Live logs.
4. Event monitor uses structured events so filtering and status have defined
   semantics.
5. Audit actions are grouped by request identity rather than timestamp
   proximity.
6. Group discovery comes from metadata and review ledgers; style-image
   existence is not a prerequisite.
7. Calendar history uses a dedicated recent-run read path; scheduler due-work
   queries retain their execution semantics.
8. Background worker ticks become an aggregate plus recent errors, while jobs
   remain individually inspectable.
9. Character growth shows what changed and why; execution metadata remains
   diagnostic.
10. Unsupported Character memory is removed instead of filled from an
    unrelated source.
11. Current styling remains intact so verification can isolate information
    correctness from visual change.

## Change Surface

### Control-console production files

- `src/control_console/repository.py`
  - replace V1 character/user projections;
  - define the frozen page panels;
  - collapse audit actions;
  - retain safe event payload semantics;
  - repair combined status.
- `src/control_console/app.py`
  - add bounded user/group directory routes;
  - update endpoint query parameters and response assembly.
- `src/control_console/contracts.py`
  - update typed public response contracts where new directory, facet, target,
    or outcome envelopes require them.
- `src/control_console/event_monitor.py`
  - remove raw process-log merging;
  - preserve structured severity, duration, error, and correlation fields.
- `src/control_console/static/index.html`
  - remove dead/static data regions;
  - add semantic containers required by the frozen panel contracts.
- `src/control_console/static/console.js`
  - render page-specific semantic panels;
  - remove generic internal-metadata cards;
  - implement directories, filters, detail disclosures, and truthful states.
- `src/control_console/README.md`
  - document page ownership, target endpoints, safe-field boundaries, and
    authenticated verification.

No stylesheet change is planned. A CSS edit requires a concrete content
overflow or accessibility defect found during implementation and must preserve
the current design language.

### Domain-owner read helpers

- `src/kazusa_ai_chatbot/calendar_scheduler/repository.py`
  - add bounded recent-run inspection without changing scheduler execution.
- `src/kazusa_ai_chatbot/db/conversation.py`
  - add bounded recent-group metadata aggregation without returning message
    text.
- `src/kazusa_ai_chatbot/db/users.py`
  - add bounded recent-user profile inspection without returning global ids.
- `src/kazusa_ai_chatbot/db/self_cognition.py`
  - add bounded recent/group-scope review lookup.
- `src/kazusa_ai_chatbot/db/__init__.py`
  - export the new owner helpers through the existing database facade.
- `src/kazusa_ai_chatbot/internal_monologue_residue/{loader.py,README.md}`
  - support documented non-mutating console inspection reads.

These helpers are read-only and contain no console rendering logic.

### Tests

- `tests/test_control_console_repository.py`
- `tests/test_control_console_bootstrap.py`
- `tests/test_control_console_event_monitor.py`
- `tests/test_control_console_web_surface.py`
- `tests/test_console_lookup_limits.py`
- `tests/test_control_console_cognition_debug_visibility.py`
- `tests/control_console_e2e/test_page_navigation_e2e.py`
- `tests/control_console_e2e/test_live_database_owner_pages_e2e.py`
- `tests/control_console_e2e/test_clickable_inventory_e2e.py`
- `tests/test_internal_monologue_residue_loader.py`
- focused existing tests for changed calendar, conversation, and
  self-cognition repository modules.

Existing test files are extended unless a distinct contract cannot be expressed
without mixing unrelated test ownership.

## Overdesign Guardrail

Implementation stays within these limits:

- no frontend framework migration;
- no generic schema-driven renderer;
- no duplicate data-access layer in `control_console`;
- no new persistence collection or materialized view;
- no LLM summarizer;
- no unbounded event, user, group, calendar, job, or audit query;
- no compatibility shim;
- no visual redesign;
- no attempt to expose every stored field.

Safe operator meaning, source freshness, and actionability determine inclusion.

## Agent Autonomy Boundaries

After explicit implementation approval, the implementation owner may:

- edit only the files listed in Change Surface and directly owned tests/docs;
- add the four bounded read helpers and two directory endpoints frozen here;
- remove obsolete V1 console projectors and renderers;
- run deterministic tests and local authenticated browser validation.

The implementation owner must stop and request direction before:

- changing a database schema or writing data;
- exposing a raw message, prompt, evidence reference, credential, token, or
  unrestricted trace;
- changing cognition, scheduler execution, model routes, service lifecycle
  semantics, or adapter delivery;
- adding a page, new visual design, LLM call, cache, worker, or dependency;
- expanding the endpoint contracts beyond the frozen panels in this plan;
- deploying or restarting a production service.

## Implementation Order

1. **Baseline and failing contract evidence**
   - record exact revision, URL, port, server PID/revision evidence, and current
     worktree state;
   - load required coding/test skills;
   - replace obsolete V1 fixtures with canonical V2 fixtures;
   - add failing assertions for partial status, audit object targets/outcomes,
     calendar recent history, and event payload fields.
2. **Domain-owner bounded reads**
   - implement recent calendar runs;
   - implement recent group metadata aggregation;
   - implement recent/group-scope review lookup;
   - verify limits, sort order, safe fields, and empty results.
3. **Character and Users native V2 projections**
   - replace character profile/latest-state projectors;
   - replace user affinity/relationship projectors;
   - add user directory;
   - validate exact axes, semantic bands, optional empty panels, and excluded
     identifiers.
4. **Groups discovery and detail**
   - add group directory and activity panel;
   - source review/style/carry-over/progress independently;
   - remove hardcoded panels and style-dependent E2E selection.
5. **Calendar and Background work semantics**
   - assemble frozen summary/run/job/worker panels;
   - keep canonical rows unique;
   - move lease/run machinery into bounded detail or omit it.
6. **Health/cache and Audit correctness**
   - project nested runtime workers and semantic error level;
   - show complete cache metrics once;
   - implement explicit audit event mapping, target labels, request grouping,
     view summary, and filters.
7. **Event, Logs, Services, Debug, and Overview ownership**
   - remove process streams from Event monitor;
   - retain raw streams in Logs without inferred severity;
   - remove duplicate Services and route content;
   - label Debug history provenance;
   - reduce Overview to aggregates/exceptions.
8. **Static renderer and documentation cutover**
   - replace all old panel names and element references;
   - remove generic internal-metadata rendering;
   - update console documentation and capability descriptions.
9. **Verification and code review**
   - run the scoped deterministic test batches;
   - start the exact checkout on port 8764;
   - authenticate with an operator token supplied through the approved local
     workflow;
   - verify every page against real source data;
   - capture screenshots and source/result notes;
   - complete parent-only code review and resolve findings;
   - obtain user authorization before lifecycle closure.

## Execution Model

- The parent owns test contracts, integration, verification, execution evidence,
  review remediation, lifecycle updates, and final sign-off.
- Initial scoped production implementation preceded the user's parent-only
  review direction; every subsequent review, fix, and closure action is
  parent-owned.
- Plan review remains a parent-only self-review.
- Browser verification uses the in-app Browser first. If it again reports no
  available browser, regular Playwright is the recorded fallback.
- A page is complete only when its backend contract, renderer, deterministic
  tests, and populated browser evidence agree.

## Progress Checklist

- [x] Read governing project and console documentation.
- [x] Inspect all twelve current page layouts.
- [x] Inspect relevant repository, route, renderer, schema, and test code.
- [x] Gather bounded read-only current-data evidence.
- [x] Freeze the page ownership and target panel contracts in this draft.
- [x] Complete parent-only plan self-review.
- [x] Receive explicit user approval for production-code implementation.
- [x] Record implementation baseline and failing contract tests.
- [x] Implement domain-owner bounded reads.
- [x] Implement Character and Users native V2 projections.
- [x] Implement Groups discovery and detail.
- [x] Implement Calendar and Background work semantic projections.
- [x] Implement Health/cache and Audit correctness fixes.
- [x] Implement Event/Logs/Services/Debug/Overview ownership cleanup.
- [x] Complete static renderer and documentation cutover.
- [x] Pass scoped deterministic tests.
- [x] Pass authenticated populated browser verification for all twelve pages.
- [x] Complete parent-only code review and resolve findings.
- [x] Receive user authorization to commit and close the plan.

## Verification

### Deterministic contract verification

After applying `test-style-and-execution`, run through the project virtual
environment:

```powershell
venv\Scripts\python -m pytest `
  tests/test_control_console_repository.py `
  tests/test_control_console_bootstrap.py `
  tests/test_control_console_event_monitor.py `
  tests/test_control_console_web_surface.py
```

Run the changed domain-owner repository tests in the same deterministic batch.
Then run:

```powershell
venv\Scripts\python -m pytest `
  tests/control_console_e2e/test_page_navigation_e2e.py `
  tests/control_console_e2e/test_live_database_owner_pages_e2e.py
```

The live-database owner-page test must:

- require V2 relationship fields when a selected profile contains V2
  relationship state;
- select groups from the new recent-group directory rather than style images;
- verify completed calendar runs independently of due runs;
- verify a zero-row optional source as `empty`, not unavailable;
- fail when any default page shows `[object Object]`, raw projection metadata,
  or an obsolete V1 affinity field.

### Source safety verification

Contract tests must prove that responses exclude:

- raw conversation text;
- prompt text and protected trace content;
- relationship evidence references;
- suspected-alias internal identifiers;
- credentials and operator tokens;
- unbounded raw event/audit payloads;
- lease owners and internal run IDs from default panels.

### Authenticated browser matrix

Verify the exact running checkout at `http://localhost:8764/`:

| Page | Required populated-browser proof |
|---|---|
| Overview | Brain/service aggregate appears once; no Event stream or internal-contract cards; links reach owner pages. |
| Services | One lifecycle state and one route current value; action-block reasons are readable. |
| Live logs | Raw process lines appear only here; stderr has no manufactured severity. |
| Debug chat | Request/reply and graph work; History is labeled current-session-only. |
| Event monitor | Structured facets/filters work; duration/error/correlation detail is available; process lines are absent. |
| Character | Full safe profile and V2 cognition fields render; growth is semantic and nonduplicated; truthful empty growth is readable. |
| Users | Directory selects a real user; every stored V2 relationship axis and cognition collection renders from V2; optional empty panels remain truthful. |
| Groups | Directory discovers active scopes without style rows; activity and available review state render; no invented cognition. |
| Calendar | Empty schedule state and recent completed run history can coexist; one run appears once. |
| Background work | Empty queue remains useful through worker outcome summary; job/prompt duplication is absent. |
| Health/cache | DB, scheduler, every worker, semantic error level, and full Cache2 metrics appear once. |
| Audit | Targets are human-readable; outcome is populated; paired lifecycle rows collapse; views are summarized separately. |

Capture one full-page screenshot per page plus focused evidence for expandable
detail where needed. Screenshots support, but do not replace, source/response
inspection.

### Duplication verification

Search server contracts and visible DOM text for the following:

- brain service detail outside Services;
- stream URL or `Event stream` outside Live logs diagnostics;
- raw process output outside Live logs;
- audit action rows outside Audit, except bounded Overview change/failure
  aggregates;
- cache-agent details outside Health/cache;
- old user `affinity`, `relationship_summary`,
  `last_relationship_insight`, and `relationship_status`;
- visible `panel_contract`, `projection_owner`, `scope_order`, or raw
  `scope_summary`.

Any duplicate must be either removed or documented as a bounded Overview
aggregate that links to its canonical owner.

## Plan Self-Review

The parent agent completed the plan review without a subagent.

Review checks:

- every one of the twelve pages has a defined owner, defect list, target
  information state, and verification row;
- each proposed datum has an authoritative source or is explicitly removed;
- V2 relationship and character contracts are named explicitly;
- current empty collections are distinguished from missing sources;
- endpoint changes, read helpers, files, tests, cutover, rollback, and
  acceptance gates are specified;
- no database mutation, LLM call, visual redesign, compatibility layer, or
  unresolved implementation choice is included;
- populated browser verification was retained as a mandatory execution gate.

Self-review disposition: approved for scoped implementation by the user on
2026-07-27.

## Execution Evidence

- Approval/baseline: user instructed `Execute the plan`; revision
  `87cd6df5a869dc78b6e1ed48dd1d580db9d5f6da`; root HTTP 200 on port 8764.
- Verification: 145 scoped tests, 22 regular browser E2E tests, one live-DB
  owner-page test, and one exact-running-console signoff test passed.
- Browser: in-app Browser had no available runtime; regular Playwright reviewed
  all 12 pages with zero console, request, or HTTP failures and zero LLM calls.
- Parent review resolved event-query starvation, residue read telemetry, status
  pollution, stale worker fixtures, duplicate content, and raw labels.

## Parent Code Review

The parent completed the user-directed code review and verified:

1. V1 relationship and character projectors are deleted rather than hidden
   behind fallback paths.
2. Each new query is bounded, deterministic, and owned by the correct domain
   repository.
3. safe-field projection prevents sensitive content and internal identifiers
   from crossing the console boundary.
4. status aggregation distinguishes empty, needs-input, partial, and
   unavailable exactly as frozen.
5. audit grouping uses request identity and an explicit event/outcome map.
6. Event monitor no longer imports raw process streams.
7. renderer code has no generic dumping of internal objects or projection
   metadata.
8. tests would fail on the current V1 implementation and exercise populated V2
   state.
9. no styling redesign, schema migration, LLM call, or unrelated behavior
   entered the diff.
10. authenticated browser evidence comes from the reviewed revision.

Disposition: all findings were resolved; the user directed commit and plan
closure on 2026-07-27.

## Risks and Controls

| Risk | Control |
|---|---|
| Current empty collections hide broken populated behavior | Canonical fixtures plus authenticated live-data verification cover both empty and populated states. |
| User/group directories expose identity data | Return only bounded safe identifiers/display labels and counts; exclude alias internals and raw messages. |
| Directory aggregation adds database load | Enforce hard limits, indexed sort/filter paths, deterministic ordering, and repository tests. |
| Removing duplicate diagnostics makes investigation harder | Preserve raw/detail information on Logs, Events, Health/cache, or Audit according to ownership. |
| Audit request grouping merges unrelated actions | Group only exact request IDs and use an explicit terminal-event map. |
| Static client and server contract revisions diverge | Big-bang revision cutover and exact-checkout browser verification. |
| Existing user worktree changes are overwritten | Keep edits scoped and inspect `git status --short` before every implementation checkpoint. |

## Acceptance Criteria

The plan is implemented only when all conditions are true:

1. All twelve pages pass their authenticated browser matrix on the exact
   reviewed revision.
2. Overview contains each aggregate once and contains no duplicate detailed
   brain/service/route/cache/event/audit information.
3. `Event stream`, internal stream paths, CSRF header names, visible workflow
   counts, and generic projection metadata are absent from primary content.
4. A V2 user profile renders all stored relationship axes, canonical bands,
   updated time, evidence count, and V2 cognition collections without reading
   V1 affinity fields.
5. Character renders all safe profile fields and native V2 cognition fields;
   unsupported memory and legacy latest-state projections are absent.
6. Groups are discoverable from actual conversation/review metadata even when
   interaction-style collections are empty.
7. Calendar simultaneously represents a truthful empty schedule source and
   bounded recent completed runs.
8. Background work remains informative with an empty job queue and shows safe
   worker outcomes without repeated tick-card junk.
9. Health/cache shows every returned worker and cache agent with complete
   operator-relevant metrics exactly once.
10. Audit never renders `[object Object]`, never depends on an absent `status`
    field, collapses paired lifecycle events, and separates view summaries from
    state-changing actions.
11. Event monitor contains structured runtime events and no raw process stream
    duplication.
12. Page status is `partial` whenever a required panel fails while another
    succeeds.
13. Default page content exposes no prohibited sensitive or internal fields.
14. Scoped deterministic tests, live-database E2E, parent code review, and
    user-authorized lifecycle closure all pass.
15. LLM call count remains zero and the database remains unmodified.
