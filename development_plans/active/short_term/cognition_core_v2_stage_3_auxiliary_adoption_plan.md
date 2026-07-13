# Cognition Core V2 Stage 3 Auxiliary Adoption Plan

## Summary

- Goal: adopt the stable Cognition Core V2 release-candidate contracts across
  auxiliary operator, debugging, export, audit, test-harness, web-console, and
  documentation surfaces after Stage 2 completes isolated validation.
- Plan class: high-level follow-on plan.
- Status: draft.
- Detail level: scope inventory only. Exact contracts, files, UI layouts,
  implementation order, test commands, and acceptance criteria are deferred
  until the Stage 2 output and diagnostics contracts are stable.
- Runtime boundary: Stage 3 consumes V2 state and diagnostics. It does not
  redesign emotion reducers, cognition contracts, persistence, L3, dialog,
  resolver behavior, or action selection.
- Database and deployment boundary: every Stage 3 database-backed test and
  browser/data validation uses `_test_kazusa_live_llm` with synthetic data.
  Production database access, migration, deployment, and restart belong to a
  separately approved future cutover plan.
- Exact incoming legacy-residual scope: the `Exact Stage 3 residual allowlist`
  in `cognition_core_v2_stage_2_execution_manifest.md` is the complete Stage 2
  legacy-text handoff. The detailed Stage 3 plan must classify every listed
  path as `change`, `retire`, or `verified unchanged`. The wider auxiliary
  inventory below is audit-only until a fresh repository inventory identifies
  exact additional V2 adoption paths and the user approves that scope.

## Activation Gate

Detailed Stage 3 planning begins after:

1. Stage 2 freezes `CognitionCoreOutputV2`, semantic state projections,
   diagnostics, cognition-graph data, and embedded persistence schemas.
2. Stage 2 completes Checkpoints A through I, demonstrates `S2-O1`
   through `S2-O10`, and produces the V2 release candidate with V1 and affinity
   removed from the candidate source tree.
3. Stage 2 verifies the `_test_kazusa_live_llm` fail-closed guard, synthetic
   seed, owner isolation, singleton-character restore, and zero production
   database access.
4. Stage 2 attaches the exact residual allowlist and the diagnostic failures
   from the six excluded Stage 3 test paths.
5. The user explicitly requests and approves a detailed Stage 3 plan.

## Stage 3 Scope Inventory

### 1. Web Control Console

| Surface | Required auxiliary adoption |
|---|---|
| Overview cognition graph | Display V2 appraisal, reducer, derived-emotion, goal-branch, dependency, collapse, route-selection, and L3 nodes from the bounded brain diagnostics contract |
| Debug chat cognition graph | Replace V1 stage expectations with the actual V2 branch DAG, status, dependency, timing, and collapse summary |
| Cognition graph inspector | Show bounded branch purpose, dependencies, lifecycle trend, selected/suppressed bid summary, and failure status without raw reasoning or prompt content |
| Character page | Replace mood/global-vibe/reflection-summary rows with read-only character drives, meaning state, active causal entities, and semantic affect summaries; database-backed validation uses the seeded test singleton and scoped restore after any write-capable setup |
| Users page | Replace affinity and relationship-insight rows with the V2 relationship axes, semantic relationship projection, and bounded user-scoped active entities; database-backed validation uses owner-scoped synthetic users in `_test_kazusa_live_llm` |
| Prompt View panels | Call production V2 projection functions and show the same bounded semantic windows supplied to scoped appraisal, goal cognition, collapse, and L3 where those views are useful |
| Event monitor | Recognize V2 component names, branch lifecycle events, collapse outcomes, state persistence outcomes, and sleep recovery summaries |
| Health/cache page | Show available V2 runtime/diagnostic health and state-cache freshness summaries from existing bounded ops APIs |
| Debug modes | Verify visible-reply, think-only, listen-only, and no-remember behavior against V2 state-persistence ownership |
| SSE invalidation | Retain the single status stream and update cognition-graph invalidation handling for V2 run identifiers |
| Page capabilities | Report explicit ready, partial, unavailable, or disabled state for every V2-dependent panel |
| Redaction | Exclude raw prompts, raw numbers where the panel promises semantic prompt input, private branch text, raw state documents, internal ids, source refs, and unrestricted traces |
| Browser presentation | Reuse existing static HTML/CSS/JavaScript widgets and cognition graph renderer; update only the panels and graph semantics required by V2 |

Expected console-owned change surface:

```text
src/control_console/contracts.py
src/control_console/kazusa_client.py
src/control_console/repository.py
src/control_console/app.py
src/control_console/redaction.py
src/control_console/stream.py
src/control_console/static/index.html
src/control_console/static/console.js
src/control_console/static/console.css
src/control_console/README.md
```

The console remains read-only for cognition state. Stage 3 adds no relationship
or emotion editing controls, reset buttons, alternate state authority, new LLM
calls, frontend build system, WebSocket, or second live-event stream. The
detailed plan records the served checkout, port, loopback URL,
authentication/session assumptions, Browser or Playwright path, fresh-context
validation, and hard-reload/static-asset validation.

### 2. Brain-Side Auxiliary Operations APIs

Inventory:

- latest cognition graph snapshot and `/ops/latest-cognition-graph`;
- `/chat` debug response `cognition_graph` projection;
- self-cognition latest-graph projection;
- runtime-status summaries related to V2 cognition and character state;
- bounded diagnostic status returned to the console;
- redaction and size limits for graph nodes, edges, detail, and run metadata.

Likely owners:

```text
src/kazusa_ai_chatbot/service.py
src/kazusa_ai_chatbot/brain_service/contracts.py
src/kazusa_ai_chatbot/brain_service/health.py
src/kazusa_ai_chatbot/event_logging/
```

Stage 3 consumes the stable diagnostics emitted by Stage 2 and avoids adding
new live-cognition work.

### 3. Seeded Test-Database Inspection, Export, and Audit Tools

Inventory:

- user profile export and user-state snapshot output;
- character-state export and snapshot output;
- user-profile and character-state audit commands;
- native-V2 seed and owner-scope inspection;
- auxiliary tools that currently check affinity or prose character affect;
- generic collection exports whose redaction rules need V2 state awareness;
- artifact inventory/count commands that classify cognition-state evidence;
- script README and operator examples.

Expected script surface includes:

```text
src/scripts/export_user_profile.py
src/scripts/export_character_state.py
src/scripts/user_state_snapshot.py
src/scripts/character_state_snapshot.py
src/scripts/audit_user_profiles_lane.py
src/scripts/audit_character_state_lane.py
src/scripts/fetch_ops_status.py
src/scripts/count_project_artifacts.py
src/scripts/README.md
src/kazusa_ai_chatbot/db/script_operations.py
```

These tools display or validate canonical embedded V2 documents in
`_test_kazusa_live_llm`. They use approved DB facades, synthetic data, owner
scope, bounded/redacted output, and the Stage 2 fail-closed guard. They create
no second schema, legacy-affinity translator, production migration operation,
database-wide reset, or routine reseed. Existing repair scripts are audited
and changed only when a test-database-only V2 use case is approved in the
detailed plan.

### 4. Protected Tracing and Diagnostic Export

Inventory:

- V2 stage and branch names in protected LLM traces;
- semantic appraisal, goal branch, collapse, action selection, and L3 trace
  grouping;
- branch count, dependency wait, overlap, latency, failure, and selected-bid
  diagnostic export;
- semantic lifecycle and state-update summaries;
- trace retrieval/export scripts and review-input generation;
- retention, redaction, bounded preview, and operator-access rules;
- Stage 2 artifact directory discovery and review helpers.

Expected surface:

```text
src/kazusa_ai_chatbot/llm_tracing/
src/kazusa_ai_chatbot/event_logging/
src/scripts/export_llm_trace.py
src/scripts/export_event_log.py
src/scripts/export_dialog_trace_review_input.py
tests/llm_trace.py
```

### 5. Debug, Benchmark, and Evaluation Utilities

Inventory:

- V2 validation CLI and benchmark harness usability after the production
  contract replaces the Stage 1 V1-compatible facade;
- supported-emotion lifecycle selection and explicit `N/A` reporting;
- production-default state seeding for diagnostic cases;
- branch/DAG timing output and failure summaries;
- debug-adapter and console fake-brain fixtures;
- human-readable value/cost/quality report generation;
- artifact naming and directory conventions.

Likely surface:

```text
src/kazusa_ai_chatbot/cognition_core_v2/validation_cli.py
tests/fixtures/cognition_core_v2_*.json
tests/control_console_e2e/fake_brain.py
tests/control_console_e2e/fake_services.py
test_artifacts/cognition_core_v2/
```

### 6. Auxiliary Tests and Browser Validation

Inventory:

- control-console contract, repository, client, route, redaction, stream, and
  static-web tests;
- cognition graph and debug-visibility tests;
- Character and Users page expectations for V2 state;
- fake-brain cognition graph fixtures;
- end-to-end debug chat, navigation, seeded-database owner pages, error paths,
  service lifecycle, and visual product acceptance;
- responsive layout, bounded tables, graph overflow, keyboard focus, notices,
  loading states, and console/page error checks;
- fresh-context and hard-reload browser validation for changed static assets;
- screenshots and review artifacts required by the eventual detailed plan.

Primary test areas:

```text
tests/test_control_console_*.py
tests/test_console_debug_chat.py
tests/control_console_e2e/
```

Every affected test that processes database data uses `_test_kazusa_live_llm`;
affected real-LLM cases carry `live_db`, execute one at a time, and retain the
selected database in evidence. Tests use unique owner identifiers. Shared seed
data persists between routine runs. A test that writes the singleton character
state snapshots and restores only that document.

Browser execution details remain deferred. The detailed Stage 3 plan will
state the served checkout, port, loopback URL, authentication/session
assumptions, in-app Browser or recorded Playwright-fallback path, hard-reload
path, and screenshot/error evidence before rendered validation.

### 7. Documentation and Architecture Presentation

Inventory:

- control-console ICD and page capability table;
- root README architecture graph and cognition terminology;
- `docs/HOWTO.md` operator and debugging instructions;
- scripts and event/trace documentation;
- Character and Users page field descriptions;
- cognition graph legend and status vocabulary;
- Stage 2 supported-emotion and `N/A` inventory presentation;
- removal of stale V1 stage, affinity, mood, global-vibe, and
  reflection-summary examples from auxiliary documentation;
- development-plan registry and final Stage 3 evidence links.

### 8. Auxiliary Surfaces Requiring Audit Only

These areas receive a Stage 3 compatibility audit and change only when the
stable V2 contract materially affects them:

- Brain model-route configuration UI;
- Calendar and background-work Prompt View panels;
- Groups page and interaction-style panels;
- health/cache statistics unrelated to cognition;
- adapter lifecycle controls and live process logs;
- audit JSONL and service configuration pages;
- RAG, memory, scheduler, reflection, and growth panels whose production
  source contract remains unchanged.

The audit records `unchanged` when no V2 adoption is required.

## Stage 3 Exclusions

Stage 3 excludes:

- emotion-state redesign or new emotion families;
- Stage 2 persistence schema changes;
- persona/resolver/L3/dialog/action behavior changes;
- new console write authority over cognition state;
- new LLM calls for display or explanation;
- a new frontend framework or build tool;
- raw prompt, raw state-document, raw branch-reasoning, or secret exposure;
- adapter transport changes;
- physiological or biological emotion simulation.
- production database connection, read, write, migration, reset, or seed;
- production deployment, restart, or release cutover;
- production migration/repair tooling;
- database-wide routine reset or reseed of `_test_kazusa_live_llm`;
- production-derived seed data.

## Future Detailed Plan Requirements

When activated, the detailed Stage 3 plan must:

1. Reinspect the completed Stage 2 contracts and actual repository diff.
2. Classify every inventory item as `change`, `audit only`, or `retire`.
3. Freeze exact console/API/export contracts and redaction rules.
4. Provide file-level implementation steps and test commands.
5. Define rendered browser validation for every changed page and control.
6. Preserve the control console as a separate buildless FastAPI/static service.
7. Report remaining stale V1/affinity/prose-affect references and remove them
   from the auxiliary scope.
8. Map every database-touching test and tool to the exact
   `_test_kazusa_live_llm` guard, synthetic seed, owner scope, and singleton
   restore behavior.
9. Define the served checkout, port, loopback URL, authentication/session
   assumptions, browser path, fresh-context path, and hard-reload validation.
10. Demonstrate zero production database access and keep migration/deployment
    in the separately approved future cutover plan.
11. Receive explicit user approval before production or console-code edits.

## Progress Checklist

- [x] Stage 3 purpose separated from Stage 2 runtime integration.
- [x] High-level auxiliary scope inventory recorded.
- [x] Web control-console surfaces included.
- [x] Operator APIs, scripts, exports, audits, traces, fixtures, tests, and
  documentation included.
- [x] Stage 3 database-backed validation constrained to synthetic data in
  `_test_kazusa_live_llm`; production migration and deployment deferred.
- [ ] Stage 2 V2 contracts and diagnostics stabilized.
- [ ] Stage 2 Checkpoints A through I and isolated-database evidence attached.
- [ ] Stage 2 deferred auxiliary references attached.
- [ ] Detailed Stage 3 plan requested and drafted.
- [ ] Detailed Stage 3 plan approved.
- [ ] Stage 3 implementation and verification executed.

## Approval Boundary

This document records future scope only. It authorizes planning edits and does
not authorize production, control-console, script, test, database, deployment,
or browser changes.

## Execution Evidence

This section remains empty until a detailed Stage 3 plan is activated.
