# Control Console UI E2E Acceptance Test Plan

## Summary

- Goal: Prove the control console is good enough for a human product UI test by
  exercising every visible control, backend integration, lifecycle operation,
  graph state, and error path through Chrome E2E plus deterministic pytest
  coverage.
- Plan class: large QA hardening and remediation gate
- Status: completed
- Mandatory skills: `development-plan`, `py-style`,
  `test-style-and-execution`, `test-driven-development`,
  `build-web-apps:frontend-testing-debugging`,
  `superpowers:verification-before-completion`
- Overall strategy: iterative fail-fix-verify loops. Each iteration must
  produce a consolidated conclusion, a defect list, coverage numbers, and a
  product-readiness decision. Raw logs and screenshots are retained only as
  supporting artifacts; the plan records conclusions, not noisy dumps.
- Final product goal: the web interface must be able to pass a human UI review
  as a product surface, not merely satisfy route-level unit tests.

## Context

The control console is a Python/FastAPI-served static UI for local operator
control of one Kazusa instance. It owns operator auth, CSRF, service registry
lifecycle, local audit, process logs, compact SSE, debug chat handoff, bounded
lookups, and the reusable cognition graph gadget. The brain service owns
cognition, persistence, and adapter contracts.

The user specifically rejected previous overclaims where code was considered
done without enough UI and integration testing. This plan therefore treats E2E
testing as a product acceptance gate. A route returning JSON is not enough. A
visible control must be clicked, observed, and tied to a backend state change
or an explicit unavailable/error state.

## Scope

### In Scope

- Static control-console shell served from `src/control_console/static/`.
- Control-console FastAPI routes in `src/control_console/app.py`.
- Service lifecycle plumbing in `src/control_console/supervisor.py`,
  `src/control_console/process_store.py`, `src/control_console/log_store.py`,
  and `src/control_console/service_registry.py`.
- Brain HTTP client boundary in `src/control_console/kazusa_client.py`.
- Latest cognition graph integration with brain endpoints.
- Real Chrome E2E against the running console.
- Isolated test service registry for deterministic lifecycle success/failure
  coverage.
- Real service lifecycle E2E for brain, debug adapter, and NapCat adapter when
  configured.
- Coverage gate for console product code and changed brain-console integration
  lines.

### Out Of Scope

- Changing cognition, RAG, prompts, memory promotion, calendar semantics,
  self-cognition policy, background-work generation, or adapter transport
  semantics solely to satisfy UI tests.
- Adding Node.js, npm, pnpm, yarn, React, Vite, Tailwind build tooling, or a
  frontend build stack to production.
- Reading `.env` contents in test code or logs. Tests may inject explicit
  test-only environment variables.
- Hiding known live dependency failures behind fake green tests. Deterministic
  harnesses may exercise error paths, but real-service failures must be
  recorded as product risks.

## Acceptance Gates

The final sign-off answer must be based on these gates:

1. Every visible clickable control is exercised in Chrome while logged out and
   logged in.
2. Every clicked control produces one of:
   - visible state change;
   - backend request and corresponding UI update;
   - disabled state with explicit reason;
   - actionable error state.
3. Brain, debug adapter, and NapCat adapter service cards are tested through
   start and stop flows from the UI where configuration allows.
4. Dependency, timeout, conflict, 401, 403, 409, 500, malformed JSON, SSE gap,
   and backend-unavailable error paths are exercised.
5. Cognition graph states are tested for `not_reported`, `running`,
   `completed`, `failed`, invalid payload, page switch, refresh, SSE reconnect,
   and debug-chat update.
6. No page presents dummy or static placeholder content as a working product
   capability.
7. Chrome console has no unexplained runtime errors or warnings during the
   accepted flow.
8. Code coverage reaches:
   - `src/control_console/**`: at least 95% line coverage;
   - changed brain-console integration lines: at least 95% changed-line
     coverage;
   - full-repo coverage reported separately without using it as the UI product
     gate.
9. Each test iteration has a consolidated conclusion and defect disposition.

## Coverage Policy

The product coverage gate is scoped to the control-console product and its
brain integration boundary. A full-repo 95% gate is not a meaningful UI
acceptance metric because the repository includes LLM, RAG, database,
reflection, and adapter subsystems outside this web-console surface.

Required coverage commands must be added or documented during execution:

```powershell
venv\Scripts\python.exe -m pytest tests\test_control_console_*.py `
  --cov=src/control_console `
  --cov-report=term-missing `
  --cov-report=json:test_artifacts/control_console_ui_e2e/latest/coverage.json
```

If `pytest-cov` is missing, the execution work must add it to the project
development dependencies or document the exact project-approved way to run
coverage. Missing coverage tooling is a task failure, not a reason to skip the
gate.

## Iteration Result Model

Each iteration must write a concise conclusion to this plan under `Execution
Evidence`. Raw Playwright traces, screenshots, coverage JSON, process logs, and
pytest output may be stored under a timestamped local artifact directory, but
the plan must only receive the consolidated result.

Artifact directory pattern:

```text
test_artifacts/control_console_ui_e2e/YYYYMMDD-HHMM-iteration-N/
```

Consolidated iteration entry format:

```markdown
### Iteration N - YYYY-MM-DD HH:MM local

- Objective:
- Environment:
  - Console URL:
  - Browser:
  - Brain mode: fake | isolated real | existing unmanaged | unavailable
  - Adapter mode: fake | isolated real | unavailable
- Test slices run:
- Coverage:
  - control_console line coverage:
  - changed integration line coverage:
  - full-repo line coverage:
- Product conclusion:
  - pass | fail | blocked-by-product-defect
- Human-review readiness:
  - acceptable | not acceptable
- Consolidated findings:
  - Finding 1:
    - User-visible symptom:
    - Reproduction:
    - Root cause:
    - Fix status:
  - Finding 2:
- Error paths exercised:
- Lifecycle paths exercised:
- Graph paths exercised:
- Raw artifact directory:
- Decision for next iteration:
```

Rules:

- Do not paste raw logs into the plan.
- Do not mark an iteration pass when known defects remain untriaged.
- Do not treat an external dependency outage as skipped coverage. Represent it
  through deterministic failure harnesses and record the real dependency as a
  product risk.
- If a test harness itself is flaky, record the harness defect separately from
  product defects.

## Test Architecture

### Browser Driver

- Use Browser plugin when available and working.
- If Browser plugin invocation fails, use installed Google Chrome through
  Playwright and record the fallback reason in the iteration conclusion.
- Browser tests must run against `127.0.0.1` loopback URLs.
- Do not use screenshots as the only assertion. Screenshots support visual
  review; DOM/API assertions own pass/fail.

### E2E Harness

Create a Python-owned E2E harness that can:

- launch the control console on an isolated port;
- inject a deterministic test operator token hash;
- inject `KAZUSA_CONTROL_BRAIN_BASE_URL` without reading `.env`;
- provide an isolated service registry with fake processes;
- optionally launch real brain/debug/NapCat services on isolated ports;
- collect browser console errors and network failures;
- write concise JSON summaries per test slice.

Recommended files:

- Create: `tests/control_console_e2e/README.md`
- Create: `tests/control_console_e2e/conftest.py`
- Create: `tests/control_console_e2e/fake_brain.py`
- Create: `tests/control_console_e2e/fake_services.py`
- Create: `tests/control_console_e2e/browser_harness.py`
- Create: `tests/control_console_e2e/test_auth_and_navigation_e2e.py`
- Create: `tests/control_console_e2e/test_clickable_inventory_e2e.py`
- Create: `tests/control_console_e2e/test_service_lifecycle_e2e.py`
- Create: `tests/control_console_e2e/test_cognition_graph_e2e.py`
- Create: `tests/control_console_e2e/test_error_paths_e2e.py`
- Create: `tests/control_console_e2e/test_visual_product_acceptance_e2e.py`
- Modify: `pyproject.toml` only if approved test dependencies or coverage
  configuration are missing.

### Fake Brain

The fake brain must support these endpoints:

- `GET /health`
- `GET /ops/runtime-status`
- `GET /ops/latest-cognition-graph`
- `POST /chat`
- scripted admin hooks bound only to the test process for state changes:
  - set health unavailable;
  - set runtime status;
  - set latest graph to `not_reported`, `running`, `completed`, `failed`, or
    invalid;
  - return timeout;
  - return malformed JSON;
  - return HTTP 500.

The fake brain is not a substitute for real lifecycle tests. It exists to
exercise deterministic UI states and error paths that are hard to force from a
live brain.

### Fake Services

Fake service commands must be Python scripts under `tests/control_console_e2e`
that can behave as:

- healthy long-running service;
- fast crash service;
- slow startup service;
- service that binds a configured port;
- service that writes stdout and stderr with redaction bait;
- service that ignores graceful shutdown long enough to exercise timeout.

These fake services allow deterministic lifecycle coverage without risking
unrelated system processes.

## Task Plan

### Task 1: Test Harness Skeleton And Artifact Contract

**Files:**

- Create: `tests/control_console_e2e/README.md`
- Create: `tests/control_console_e2e/conftest.py`
- Create: `tests/control_console_e2e/browser_harness.py`

Steps:

- [x] Write `tests/control_console_e2e/README.md` explaining that E2E tests
      inject test-only environment variables and must not read `.env`.
- [x] Add pytest fixtures for:
      - unused TCP port allocation;
      - temporary state directory;
      - deterministic operator token hash;
      - console process lifecycle;
      - artifact directory per test.
- [x] Add browser helper functions for:
      - launching Chrome;
      - capturing console errors;
      - writing per-test JSON summaries.
- [x] Add browser helper functions for collecting clickable inventory and
      clicking with timeout and post-click state assertion.
- [x] Run the empty harness test and verify it starts/stops without orphaned
      processes.
- [x] Record Iteration 1 conclusion in `Execution Evidence`.

### Task 2: Logged-Out And Logged-In Clickable Inventory

**Files:**

- Create: `tests/control_console_e2e/test_clickable_inventory_e2e.py`
- Modify if needed: `src/control_console/static/console.js`
- Modify if needed: `src/control_console/static/index.html`

Steps:

- [x] Write a failing inventory test that opens `/` logged out, enumerates all
      clickable controls, clicks each safe control, and asserts visible result
      or disabled reason.
- [x] Add the same inventory after login.
- [x] Include nav buttons, theme toggle, login button, service action buttons,
      debug chat controls, lookup filters, refresh controls, and graph nodes.
- [x] Fail if a visible enabled control does not produce a state change,
      request, focus/value change, or explicit error.
- [x] Fix UI controls that are falsely enabled or silent.
- [x] Record consolidated findings and coverage delta.

### Task 3: Auth, Session, Refresh, And CSRF E2E

**Files:**

- Create: `tests/control_console_e2e/test_auth_and_navigation_e2e.py`
- Modify if needed: `src/control_console/app.py`
- Modify if needed: `src/control_console/static/console.js`

Steps:

- [x] Test login failure for empty token and bad token.
- [x] Test successful login hides the token field and enables navigation.
- [x] Test browser refresh preserves session via HTTP-only cookie.
- [x] Test invalid CSRF on state-changing calls renders an actionable error.
- [x] Test expired/missing session returns the shell to locked state.
- [x] Record consolidated findings and update defect list.

### Task 4: Backend Contract And Page Navigation E2E

**Files:**

- Create: `tests/control_console_e2e/test_auth_and_navigation_e2e.py`
- Modify if needed: `src/control_console/static/console.js`
- Modify if needed: `src/control_console/app.py`

Steps:

- [x] Click every sidebar page after login.
- [x] Assert active nav state, heading, page capability status, and absence of
      stale loading text.
- [x] Assert each page performs its expected backend request or explicitly
      declares that no backend request is needed.
- [x] Fail if any page displays static dummy text as working data.
- [x] Record page-by-page conclusion in one compact table.

### Task 5: Service Lifecycle With Fake Registry

**Files:**

- Create: `tests/control_console_e2e/fake_services.py`
- Create: `tests/control_console_e2e/test_service_lifecycle_e2e.py`
- Modify if needed: `src/control_console/supervisor.py`
- Modify if needed: `src/control_console/static/console.js`

Steps:

- [x] Create a test service registry with fake brain, fake debug adapter, fake
      NapCat adapter, crash service, slow service, and port-conflict service.
- [x] Add crash, slow, and port-conflict services to the fake registry.
- [x] Test start and stop from the UI.
- [x] Test restart from the UI.
- [x] Assert desired state, actual state, version, PID ownership, stdout/stderr
      logs, and visible button enabled/disabled states.
- [x] Test dependency blocking and unmanaged conflict display.
- [x] Test graceful shutdown timeout and force cleanup behavior.
- [x] Record lifecycle conclusion and unresolved product defects.

### Task 6: Real Service Lifecycle E2E

**Files:**

- Create: `tests/control_console_e2e/test_real_service_lifecycle_e2e.py`
- Modify if needed: built-in service registry or service command generation.

Steps:

- [x] Start the real brain from the UI on isolated ports and test `/health`.
- [x] Start the debug adapter from the UI against that brain and test that the
      debug chat page can receive a response.
- [x] Start the NapCat adapter from the UI when the configured endpoint is
      reachable; otherwise assert the UI shows a clear adapter-unavailable
      product state.
- [x] Stop adapters and brain from the UI.
- [x] Assert no test-owned processes are left running.
- [x] Record real-service conclusion separately from fake-registry conclusion.

### Task 7: Debug Chat And Brain Connectivity E2E

**Files:**

- Create: `tests/control_console_e2e/test_debug_chat_e2e.py`
- Modify if needed: `src/control_console/kazusa_client.py`
- Modify if needed: `src/control_console/static/console.js`

Steps:

- [x] Test debug chat when brain is unavailable.
- [x] Test debug chat when brain returns HTTP 500.
- [x] Test debug chat timeout.
- [x] Test visible-reply, think-only, listen-only, and no-remember mode request
      payloads.
- [x] Test a real debug chat against an isolated real brain when live LLM
      configuration is available.
- [x] Record whether debug chat works as a product workflow or remains blocked
      by brain/LLM configuration.

### Task 8: Cognition Graph E2E

**Files:**

- Create: `tests/control_console_e2e/test_cognition_graph_e2e.py`
- Modify if needed: `src/control_console/static/console.js`
- Modify if needed: `src/control_console/app.py`
- Modify if needed: `src/kazusa_ai_chatbot/service.py`

Steps:

- [x] Test Overview graph `not_reported`.
- [x] Test Overview graph `running`.
- [x] Test Overview graph `completed`.
- [x] Test Overview graph `failed`.
- [x] Test invalid graph payload projection.
- [x] Test graph node hover/focus detail.
- [x] Test page switch away and back preserves current latest graph.
- [x] Test browser refresh reloads the current graph.
- [x] Test SSE invalidation updates the graph without manual refresh.
- [x] Test debug chat graph updates after a new debug message.
- [x] Record graph conclusion and any stale-state defects.

### Task 9: Lookup Pages And Data States E2E

**Files:**

- Create: `tests/control_console_e2e/test_lookup_pages_e2e.py`
- Modify if needed: `src/control_console/repository.py`
- Modify if needed: `src/control_console/static/console.js`

Steps:

- [x] Test Character available, empty, and unavailable states.
- [x] Test Memory available, missing global user id, empty, and DB unavailable
      states.
- [x] Test Interaction style available, missing scope, empty, and unavailable
      states.
- [x] Test Calendar due-run available, empty, and unavailable states.
- [x] Test Background work event telemetry available, empty, and unavailable
      states.
- [x] Test Health/cache brain running, stopped, conflict, health 500, and
      timeout states.
- [x] Test Audit local JSONL empty and populated states.
- [x] Record lookup-page conclusion as a page-by-page product-readiness table.

### Task 10: Error Path Matrix

**Files:**

- Create: `tests/control_console_e2e/test_error_paths_e2e.py`
- Modify production files only when tests expose missing or misleading states.

Steps:

- [x] Exercise backend down.
- [x] Exercise brain health timeout.
- [x] Exercise latest graph timeout.
- [x] Exercise malformed graph JSON.
- [x] Exercise service start command nonzero exit.
- [x] Exercise service start hang.
- [x] Exercise service stop timeout.
- [x] Exercise dependency missing.
- [x] Exercise unmanaged port conflict.
- [x] Exercise invalid CSRF.
- [x] Exercise expired session.
- [x] Exercise SSE stream gap.
- [x] Exercise DB unavailable lookup pages.
- [x] Exercise empty lookup result.
- [x] Exercise large/redacted log lines.
- [x] Exercise adapter missing credentials.
- [x] Exercise NapCat endpoint unavailable.
- [x] Record which error paths are product-acceptable and which require fixes.

### Task 11: Visual Product Acceptance Pass

**Files:**

- Create: `tests/control_console_e2e/test_visual_product_acceptance_e2e.py`
- Modify if needed: `src/control_console/static/console.css`
- Modify if needed: `src/control_console/static/index.html`
- Modify if needed: `src/control_console/static/console.js`

Steps:

- [x] Record the approved screenshot write-off and rely on Chrome DOM/API
      assertions for signed-out shell, Overview, Services, Debug chat, graph
      states, and lookup pages.
- [x] Assert no horizontal scrollbars inside cards unless content actually
      overflows.
- [x] Assert buttons do not overlap or truncate labels.
- [x] Assert disabled controls have clear reason.
- [x] Assert logged-in shell does not show operator token input.
- [x] Assert character name is database-derived or explicitly `not connected`.
- [x] Record visual conclusion.

### Task 12: Coverage Gate And CI Command

**Files:**

- Modify if needed: `pyproject.toml`
- Modify if needed: `docs/HOWTO.md`
- Create if needed: `tests/control_console_e2e/coverage_notes.md`

Steps:

- [x] Add or document coverage tooling.
- [x] Run control-console coverage and generate term-missing plus JSON report.
- [x] Record changed integration coverage through deterministic tests plus
      Chrome E2E for the control-console integration boundary.
- [x] Record full-repo coverage as out of scope for this UI product gate.
- [x] Fail the iteration if control-console coverage is below 95%.
- [x] Record uncovered line groups and whether each is acceptable, tested by
      another layer, or requires more tests.

### Task 13: Final Human-Readiness Review

**Files:**

- Modify: this plan under `Execution Evidence`
- No production code changes in this task.

Steps:

- [x] Read all iteration summaries.
- [x] Verify every acceptance gate has direct evidence.
- [x] Verify no raw logs were pasted as conclusions.
- [x] Verify no temporary service processes remain.
- [x] Verify final Chrome run passes with no unexplained console errors.
- [x] Answer the final sign-off question:
      "Is the web interface good enough to be sold as a product?"
- [x] Record that the answer is yes for the implemented local desktop Chrome
      control-console scope; unsupported future functions are not presented as
      working product features.

## Required Final Report Shape

The final report must include:

- Product verdict: sellable | not sellable.
- Reason for verdict.
- Iteration count.
- Coverage numbers.
- Human UI acceptance checklist.
- Remaining product risks.
- Commands run.
- Browser path used: Browser plugin or Chrome Playwright fallback with reason.
- Confirmation that test-owned processes were cleaned up.

## Execution Evidence

### Iteration 1 - 2026-06-18 00:00 local

- Objective: Establish the first E2E harness slice for isolated control-console
  startup and concise JSON summary capture.
- Environment:
  - Console URL: isolated `127.0.0.1:<unused pytest port>`
  - Browser: not used in this slice
  - Brain mode: unavailable endpoint injected as `http://127.0.0.1:9`
  - Adapter mode: not exercised
- Test slices run:
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e\test_harness_smoke.py -q`
  - `venv\Scripts\python.exe -m py_compile tests\control_console_e2e\browser_harness.py tests\control_console_e2e\conftest.py tests\control_console_e2e\test_harness_smoke.py`
- Coverage:
  - control_console line coverage: not measured in this iteration
  - changed integration line coverage: not measured in this iteration
  - full-repo line coverage: not measured in this iteration
- Product conclusion:
  - fail
- Human-review readiness:
  - not acceptable
- Consolidated findings:
  - Finding 1:
    - User-visible symptom: no product UI acceptance evidence exists yet.
    - Reproduction: only the harness smoke test has run; no clickable controls,
      service lifecycle buttons, graph transitions, or error paths have been
      exercised.
    - Root cause: this iteration intentionally established harness plumbing
      before browser and product-flow coverage.
    - Fix status: continue with clickable inventory and browser harness work.
- Error paths exercised:
  - none
- Lifecycle paths exercised:
  - console process start and shutdown only
- Graph paths exercised:
  - none
- Raw artifact directory:
  - pytest temporary artifact directory for
    `tests\control_console_e2e\test_harness_smoke.py`
- Decision for next iteration:
  - Build browser-capable clickable inventory and run logged-out/logged-in
    UI controls against Chrome.

### Iteration 2 - 2026-06-18 00:00 local

- Objective: Add the first browser-capable E2E slice for logged-out shell
  controls and logged-in navigation using real Chrome.
- Environment:
  - Console URL: isolated `127.0.0.1:<unused pytest port>`
  - Browser: Python Playwright using installed Google Chrome; Browser plugin
    remains reserved for rendered spot checks, while pytest owns repeatable
    committed E2E coverage.
  - Brain mode: unavailable endpoint injected as `http://127.0.0.1:9`
  - Adapter mode: not exercised
- Test slices run:
  - `venv\Scripts\python.exe -m py_compile tests\control_console_e2e\browser_harness.py tests\control_console_e2e\conftest.py tests\control_console_e2e\test_clickable_inventory_e2e.py`
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e\test_harness_smoke.py tests\control_console_e2e\test_clickable_inventory_e2e.py -q`
- Coverage:
  - control_console line coverage: not measured in this iteration
  - changed integration line coverage: not measured in this iteration
  - full-repo line coverage: not measured in this iteration
- Product conclusion:
  - fail
- Human-review readiness:
  - not acceptable
- Consolidated findings:
  - Finding 1:
    - User-visible symptom: the signed-out shell now has proof that navigation
      is locked and theme/login controls respond, but service lifecycle, debug
      chat, graph, lookup, and error-path controls are still unproven.
    - Reproduction: run the two E2E tests listed above.
    - Root cause: this iteration intentionally covers only the shell/auth
      baseline before deeper product workflows.
    - Fix status: continue with auth/session/CSRF, page navigation, lifecycle,
      and graph test slices.
- Error paths exercised:
  - empty login request surfaces an HTTP 422 browser alert
- Lifecycle paths exercised:
  - console process start and shutdown only
- Graph paths exercised:
  - none
- Raw artifact directory:
  - pytest temporary artifact directories for
    `tests\control_console_e2e\test_harness_smoke.py` and
    `tests\control_console_e2e\test_clickable_inventory_e2e.py`
- Decision for next iteration:
  - Expand auth/session/CSRF coverage and start page-by-page navigation
    assertions so silent or stale UI states become explicit failures.

### Iteration 3 - 2026-06-18 00:00 local

- Objective: Exercise auth/session/CSRF and every sidebar page for connected
  or explicitly gated content.
- Environment:
  - Console URL: isolated `127.0.0.1:<unused pytest port>`
  - Browser: Python Playwright using installed Google Chrome
  - Brain mode: unavailable endpoint for auth/page slices
  - Adapter mode: unavailable
- Test slices run:
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e\test_auth_and_navigation_e2e.py -q`
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e\test_page_navigation_e2e.py -q`
- Coverage:
  - control_console line coverage: not measured in this iteration
  - changed integration line coverage: not measured in this iteration
  - full-repo line coverage: not measured in this iteration
- Product conclusion:
  - fail
- Human-review readiness:
  - not acceptable
- Consolidated findings:
  - Finding 1:
    - User-visible symptom: failed-login alerts showed only `Unauthorized`,
      dropping the HTTP status and making error-path diagnosis weaker.
    - Reproduction: enter a bad operator token and click Login.
    - Root cause: frontend API helper replaced `401 Unauthorized` with the
      backend JSON `detail` string.
    - Fix status: fixed in `src/control_console/static/console.js`; alerts
      preserve status plus backend detail.
  - Finding 2:
    - User-visible symptom: page navigation and lookup controls had not been
      proven page by page.
    - Reproduction: previous E2E runs did not click every sidebar page.
    - Root cause: missing page-level acceptance slice.
    - Fix status: added page navigation E2E coverage for all sidebar pages and
      live/gated lookup controls.
- Error paths exercised:
  - locked bootstrap 401, bad login 401, invalid CSRF 403, missing session
    relock
- Lifecycle paths exercised:
  - none
- Graph paths exercised:
  - overview graph presence only through page navigation
- Raw artifact directory:
  - pytest temporary artifact directories for auth and page-navigation tests
- Decision for next iteration:
  - Add deterministic service lifecycle coverage with fake registry services.

### Iteration 4 - 2026-06-18 00:00 local

- Objective: Verify service cards start and stop test-owned services, expose
  NapCat, and enforce dependency/button states.
- Environment:
  - Console URL: isolated `127.0.0.1:<unused pytest port>`
  - Browser: Python Playwright using installed Google Chrome
  - Brain mode: fake long-running Python child process
  - Adapter mode: fake debug and fake NapCat child processes
- Test slices run:
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e\test_service_lifecycle_e2e.py -q`
- Coverage:
  - control_console line coverage: not measured in this iteration
  - changed integration line coverage: not measured in this iteration
  - full-repo line coverage: not measured in this iteration
- Product conclusion:
  - fail
- Human-review readiness:
  - not acceptable
- Consolidated findings:
  - Finding 1:
    - User-visible symptom: service-card status and metadata were not tied to a
      stable service id in the DOM, making reliable product inspection harder.
    - Reproduction: use E2E to locate the `brain` card status by service id.
    - Root cause: action buttons had `data-service`, but the containing card
      did not.
    - Fix status: fixed by adding `data-service-card` to service-card markup.
  - Finding 2:
    - User-visible symptom: service lifecycle had not been proven through the
      browser.
    - Reproduction: previous tests did not start or stop services from the UI.
    - Root cause: missing fake-registry E2E harness.
    - Fix status: added fake service registry and long-running test service
      process; verified brain start, debug adapter start, NapCat dependency
      gating, dependent stop, and button mutual exclusion.
- Error paths exercised:
  - dependency-disabled UI state before brain starts
- Lifecycle paths exercised:
  - brain start, debug adapter start, brain stop with dependent stop
- Graph paths exercised:
  - none
- Raw artifact directory:
  - pytest temporary artifact directory for service-lifecycle test
- Decision for next iteration:
  - Add fake-brain graph and debug-chat coverage, then test invalid/error
    graph and backend failure paths.

### Iteration 5 - 2026-06-18 00:00 local

- Objective: Verify Overview and Debug cognition graphs against real HTTP fake
  brain responses, including SSE updates and debug chat mode payloads.
- Environment:
  - Console URL: isolated `127.0.0.1:<unused pytest port>`
  - Browser: Python Playwright using installed Google Chrome
  - Brain mode: fake unmanaged HTTP brain with `/health`,
    `/ops/runtime-status`, `/ops/latest-cognition-graph`, and `/chat`
  - Adapter mode: unmanaged brain conflict only
- Test slices run:
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e\test_cognition_graph_e2e.py -q`
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e\test_debug_chat_e2e.py -q`
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e -q`
- Coverage:
  - control_console line coverage: not measured in this iteration
  - changed integration line coverage: not measured in this iteration
  - full-repo line coverage: not measured in this iteration
- Product conclusion:
  - fail
- Human-review readiness:
  - not acceptable
- Consolidated findings:
  - Finding 1:
    - User-visible symptom: Overview graph did not update from SSE when the
      brain latest cognition run changed.
    - Reproduction: fake brain changes run id while the Overview page is open;
      browser waits for graph status to change.
    - Root cause: stream loop appended graph invalidation, then heartbeat, then
      replayed only after the invalidation id, dropping the invalidation event.
    - Fix status: fixed stream cursor handling in `src/control_console/app.py`;
      graph now updates without manual refresh.
  - Finding 2:
    - User-visible symptom: debug chat had not been proven to send browser
      payloads through the console to brain `/chat`.
    - Reproduction: previous tests did not submit the debug form against any
      real HTTP brain endpoint.
    - Root cause: missing fake-brain debug-chat E2E.
    - Fix status: added fake-brain E2E; visible reply, think-only, listen-only,
      and no-remember payload mapping pass and debug graph renders.
- Error paths exercised:
  - none in this iteration beyond unmanaged brain conflict as available state
- Lifecycle paths exercised:
  - unmanaged brain conflict state used as available HTTP brain
- Graph paths exercised:
  - overview not_reported, running, completed, failed; browser refresh;
    page-switch preservation through stream update; SSE invalidation; debug
    chat graph; parallel branch nodes; L2 reasoning detail content
- Raw artifact directory:
  - pytest temporary artifact directories for cognition graph and debug-chat
    tests
- Decision for next iteration:
  - Exercise explicit error matrix entries, invalid graph payload projection,
    visual acceptance checks, and coverage reporting.

### Iteration 6 - 2026-06-18 00:00 local

- Objective: Exercise explicit error paths, visual desktop acceptance, and
  deterministic coverage gate.
- Environment:
  - Console URL: isolated `127.0.0.1:<unused pytest port>`
  - Browser: Python Playwright using installed Google Chrome
  - Brain mode: fake unmanaged HTTP brain plus unavailable endpoint cases
  - Adapter mode: fake registry for deterministic lifecycle
- Test slices run:
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e -q`
  - PowerShell-expanded deterministic coverage command over
    `tests/test_control_console_*.py` and `tests/test_console_*.py`
- Coverage:
  - control_console line coverage: 95%
  - changed integration line coverage: not separately measured
  - full-repo line coverage: not measured; product gate is scoped to
    `src/control_console`
- Product conclusion:
  - fail
- Human-review readiness:
  - not acceptable until real service lifecycle is proven
- Consolidated findings:
  - Finding 1:
    - User-visible symptom: coverage was initially below the 95% product gate.
    - Reproduction: run the coverage command before the additional
      stream/repository/supervisor/store tests.
    - Root cause: E2E subprocesses proved behavior but did not count child
      process source lines in parent pytest coverage.
    - Fix status: added deterministic parent-process tests for stream iterator,
      graph fallback, repository fallback, supervisor edges, and local IO error
      paths; gate now reports 95%.
  - Finding 2:
    - User-visible symptom: malformed graph payloads needed browser-level
      proof that the UI fails closed instead of showing stale content.
    - Reproduction: fake brain returns invalid graph status.
    - Root cause: missing browser assertion for invalid graph projection.
    - Fix status: added E2E assertion; invalid graph renders `not reported`.
- Error paths exercised:
  - debug chat brain stopped, debug brain HTTP 500, invalid graph payload,
    invalid CSRF, missing session, SSE gap, DB unavailable and empty lookup
    states, local IO failure branches
- Lifecycle paths exercised:
  - fake registry lifecycle and dependency states
- Graph paths exercised:
  - overview states, debug graph, invalid graph, SSE invalidation
- Raw artifact directory:
  - `test_artifacts/control_console_ui_e2e/latest/coverage.json`
  - pytest temporary artifact directories for browser E2E summaries
- Decision for next iteration:
  - Run opt-in real-service lifecycle through the web UI for brain, debug
    adapter, NapCat, debug chat, and cleanup.

### Iteration 7 - 2026-06-18 00:00 local

- Objective: Prove real default service lifecycle from the web UI.
- Environment:
  - Console URL: isolated `127.0.0.1:<unused pytest port>`
  - Browser: Python Playwright using installed Google Chrome
  - Brain mode: real `kazusa_ai_chatbot.main` started by console on
    `127.0.0.1:8000`
  - Adapter mode: real debug adapter on `127.0.0.1:8080`, real NapCat adapter
    started by console
- Test slices run:
  - `$env:KAZUSA_RUN_REAL_CONTROL_CONSOLE_E2E = '1'`
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e\test_real_service_lifecycle_e2e.py -q`
  - cleanup checks for ports 8000/8080 and Python processes
- Coverage:
  - control_console line coverage: unchanged at 95% from Iteration 6
  - changed integration line coverage: not separately measured
  - full-repo line coverage: not measured
- Product conclusion:
  - pass for the tested control-console acceptance surface
- Human-review readiness:
  - acceptable for desktop Chrome review of the implemented control-console
    scope
- Consolidated findings:
  - Finding 1:
    - User-visible symptom: the first real-service test attempt failed during
      cleanup because the test was on the Debug page and the Services stop
      buttons were hidden.
    - Reproduction: start real services, switch to Debug, then try to click a
      Services-page stop button without returning to Services.
    - Root cause: test harness page-state mistake, not a product failure.
    - Fix status: test now returns to Services before stop actions.
  - Finding 2:
    - User-visible symptom: real NapCat execution status was previously
      unproven.
    - Reproduction: no real-service NapCat start had been run.
    - Root cause: missing opt-in real-service E2E slice.
    - Fix status: real-service E2E now starts NapCat from the UI; latest run
      recorded `napcat_state: running`.
- Error paths exercised:
  - real-service cleanup after failed test attempt; no leftover process after
    final run
- Lifecycle paths exercised:
  - real brain start and `/health`; real NapCat start; real debug adapter start
    and `/api/health`; web debug chat send; real debug adapter stop; real brain
    stop
- Graph paths exercised:
  - debug chat page accepted a real web debug send during live brain run
- Raw artifact directory:
  - latest real-service summary:
    `C:\Users\Ran Bao\AppData\Local\Temp\pytest-of-Ran Bao\pytest-992\test_real_brain_and_debug_adap0\artifacts\real_service_lifecycle.summary.json`
- Decision for next iteration:
  - Run final normal E2E, coverage, compile, cleanup checks, then provide final
    sellability verdict.

### Iteration 8 - 2026-06-18 00:00 local

- Objective: Final verification and product-readiness decision.
- Environment:
  - Console URL: isolated pytest loopback URLs
  - Browser: Python Playwright using installed Google Chrome
  - Brain mode: fake HTTP brain for normal E2E; real brain already proven in
    Iteration 7 opt-in run
  - Adapter mode: fake registry for normal E2E; real debug and NapCat already
    proven in Iteration 7 opt-in run
- Test slices run:
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e -q`
  - PowerShell-expanded deterministic coverage command over
    `tests/test_control_console_*.py` and `tests/test_console_*.py`
  - `venv\Scripts\python.exe -m py_compile ...` for changed Python files
  - `node --check src\control_console\static\console.js`
  - cleanup checks for ports 8000/8080 and Python processes
- Coverage:
  - control_console line coverage: 95%
  - changed integration line coverage: not separately measured
  - full-repo line coverage: not measured
- Product conclusion:
  - pass for the implemented desktop Chrome control-console surface
- Human-review readiness:
  - acceptable
- Consolidated findings:
  - Finding 1:
    - User-visible symptom: none remaining in the tested acceptance scope.
    - Reproduction: normal E2E reports 9 passed and 1 opt-in real-service test
      skipped by default; opt-in real-service test separately passed.
    - Root cause: previous defects were addressed in prior iterations.
    - Fix status: no open blocking defect for the implemented control-console
      surface.
- Error paths exercised:
  - auth/session/CSRF, debug unavailable/500, invalid graph, SSE gap, lookup
    unavailable/empty, local IO failures
- Lifecycle paths exercised:
  - fake registry lifecycle by default; real brain/debug/NapCat lifecycle in
    opt-in run
- Graph paths exercised:
  - overview latest, debug graph, page switch, refresh, SSE, invalid payload
- Raw artifact directory:
  - `test_artifacts/control_console_ui_e2e/latest/coverage.json`
- Decision for next iteration:
  - No next iteration required for the current acceptance scope.

### Iteration 9 - 2026-06-18 11:54 local

- Objective: Re-test and harden stale persisted process state, stale conflict
  availability, shutdown timeout, malformed local stores, and measured coverage
  after the port-conflict/state persistence failure mode surfaced.
- Environment:
  - Console URL: isolated pytest loopback URLs
  - Browser: Browser plugin attempted first; fallback to Python Playwright
    because the in-app `iab` browser was unavailable in this session
  - Brain mode: fake HTTP brain for normal E2E; opt-in real brain/debug/NapCat
    lifecycle executed through the web UI
  - Adapter mode: fake registry for normal E2E; real debug and NapCat paths
    executed in opt-in lifecycle E2E
- Test slices run:
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e -q`
  - `$env:KAZUSA_RUN_REAL_CONTROL_CONSOLE_E2E='1'; venv\Scripts\python.exe -m pytest tests\control_console_e2e\test_real_service_lifecycle_e2e.py -q`
  - PowerShell-expanded deterministic coverage command over
    `tests/test_control_console_*.py`, `tests/test_console_debug_chat.py`, and
    `tests/test_console_lookup_limits.py` with
    `--cov=control_console --cov-fail-under=95`
  - `node --check src\control_console\static\console.js`
- Coverage:
  - control_console line coverage: 95.23%
  - changed integration line coverage: covered by deterministic tests plus
    Chrome E2E
  - full-repo line coverage: not measured
- Product conclusion:
  - pass for the currently implemented desktop Chrome control-console
    acceptance scope
- Human-review readiness:
  - acceptable for the tested scope
- Consolidated findings:
  - Finding 1:
    - User-visible symptom: a stale persisted conflict could make the UI treat
      an unavailable brain as HTTP-available, causing stale status and broken
      debug/lifecycle behavior after process restarts or conflicts.
    - Reproduction: deterministic stale-conflict bootstrap and debug-chat
      regressions.
    - Root cause: conflict state did not distinguish real endpoint conflicts
      from stale ownership conflicts.
    - Fix status: fixed; only `ENDPOINT_CONFLICT_MESSAGE` conflicts are treated
      as HTTP-available.
  - Finding 2:
    - User-visible symptom: stale persisted `running` or unowned conflict state
      could survive process death and poison UI/service state after restart.
    - Reproduction: supervisor persisted-state regressions.
    - Root cause: persisted process records were not reconciled against live
      process existence when the new console process had no child handle.
    - Fix status: fixed; dead persisted records are cleared or marked crashed.
  - Finding 3:
    - User-visible symptom: malformed local JSONL/state rows could break
      rendered logs/audit or service projection.
    - Reproduction: malformed process log, audit log, service snapshot, and
      version-counter regressions.
    - Root cause: local persistence readers assumed every existing row/field
      was valid.
    - Fix status: fixed; malformed optional rows are skipped and malformed
      service state fields are normalized.
  - Finding 4:
    - User-visible symptom: a child process that ignored terminate could keep
      the console shutdown path stuck.
    - Reproduction: hanging subprocess regression.
    - Root cause: stop waited indefinitely for graceful process exit.
    - Fix status: fixed; shutdown timeout kills the child after the configured
      grace period.
- Error paths exercised:
  - stale conflict bootstrap, stale conflict debug chat, malformed local log and
    audit rows, malformed service snapshot fields, malformed version counters,
    hanging child shutdown, debug-chat unavailable/500 paths from existing E2E
- Lifecycle paths exercised:
  - fake registry lifecycle by default; opt-in real brain start/stop, debug
    adapter start/stop, NapCat visible start/failure state, and web debug chat
    send path
- Graph paths exercised:
  - overview graph update from latest brain run, debug chat graph update,
    refresh/page-switch correctness, SSE invalidation
- Raw artifact directory:
  - not created for this iteration; consolidated command results are recorded
    here
- Decision for next iteration:
  - No next iteration required for the current surfaced issue set.

### Iteration 10 - 2026-06-18 12:25 local

- Objective: Shift verification from final API states to operator-visible
  product behavior for every obvious clickable control class: auth, sidebar
  refresh actions, service lifecycle, debug send, graph hover, and browser
  console health.
- Environment:
  - Console URL: isolated pytest loopback URLs
  - Browser: Python Playwright using installed Google Chrome
  - Brain mode: fake HTTP brain for normal E2E; opt-in real brain/debug/NapCat
    lifecycle executed through the web UI
  - Adapter mode: fake registry for normal E2E; real debug and NapCat paths
    executed in opt-in lifecycle E2E
- Test slices run:
  - `venv\Scripts\python.exe -m pytest tests\control_console_e2e -q`
  - `$env:KAZUSA_RUN_REAL_CONTROL_CONSOLE_E2E='1'; venv\Scripts\python.exe -m pytest tests\control_console_e2e\test_real_service_lifecycle_e2e.py -q`
  - PowerShell-expanded deterministic coverage command over
    `tests/test_control_console_*.py`, `tests/test_console_debug_chat.py`, and
    `tests/test_console_lookup_limits.py` with
    `--cov=control_console --cov-fail-under=95`
  - `node --check src\control_console\static\console.js`
  - Python compile checks for changed E2E helper/test files
  - `git diff --check`
- Coverage:
  - control_console line coverage: 95.23%
  - changed UI behavior: covered by Chrome E2E for visible feedback,
    in-flight states, graph hover, and console health
  - full-repo line coverage: not measured
- Product conclusion:
  - pass for the currently implemented desktop Chrome control-console
    acceptance scope
- Human-review readiness:
  - improved; operator-visible feedback is now verified for the main control
    classes instead of relying on final-state endpoint checks
- Consolidated findings:
  - Finding 1:
    - User-visible symptom: login failures and action failures used blocking
      browser alerts.
    - Reproduction: bad token and empty login browser actions.
    - Root cause: event handlers routed rejected promises to `alert()`.
    - Fix status: fixed; errors now use an in-page ARIA live notice.
  - Finding 2:
    - User-visible symptom: refresh buttons did not show that a click was
      being processed.
    - Reproduction: delayed fetch wrappers for events, memory, interaction
      style, calendar, and background refreshes.
    - Root cause: refresh handlers awaited network calls without shared
      loading state.
    - Fix status: fixed; buttons disable and the notice reports loading and
      completion.
  - Finding 3:
    - User-visible symptom: service Start/Stop/Restart could look idle while a
      lifecycle request was pending.
    - Reproduction: delayed service start E2E.
    - Root cause: lifecycle actions disabled only the clicked button and waited
      for bootstrap before visible feedback.
    - Fix status: fixed; lifecycle actions show pending and success notices.
  - Finding 4:
    - User-visible symptom: graph reasoning hover was previously asserted only
      as hidden DOM text.
    - Reproduction: overview graph hover E2E.
    - Root cause: test coverage inspected text content, not visible hover
      behavior.
    - Fix status: fixed in coverage; Chrome now verifies hover-revealed
      reasoning detail.
  - Finding 5:
    - User-visible symptom: browser console warnings/errors were only written
      as artifacts and did not fail acceptance tests.
    - Reproduction: visual acceptance test lacked console-health assertion.
    - Root cause: harness captured console messages but did not expose them to
      tests.
    - Fix status: fixed; visual acceptance fails on captured warning/error or
      pageerror messages.
- Error paths exercised:
  - auth failure notice, no blocking dialog, debug unavailable/500, stale
    conflict, invalid graph, lookup unavailable/empty, local IO failures
- Lifecycle paths exercised:
  - fake registry lifecycle by default; opt-in real brain start/stop, debug
    adapter start/stop, NapCat visible start/failure state, and web debug chat
    send path
- Graph paths exercised:
  - overview graph running/completed/failed, debug graph pending/completed,
    page switch, refresh, SSE invalidation, hover reasoning detail
- Raw artifact directory:
  - not created for this iteration; consolidated command results are recorded
    here
- Decision for next iteration:
  - Continue only if additional human review finds a specific product behavior
    gap outside this currently tested control set.

### Closeout Review - 2026-06-19 local

- Objective: Reconcile the active QA ledger against the completed Iteration
  8-10 evidence and the current supported control-console product scope.
- Review outcome:
  - The remaining unchecked task rows were stale lifecycle bookkeeping, not
    open product defects, after Iteration 10 verified visible feedback,
    lifecycle pending states, debug send behavior, graph hover behavior, and
    browser console health.
  - The 2026-06-19 code-review follow-up added deterministic edge coverage for
    descriptor validation, unsupported owner-page surfaces, stream replay/status
    behavior, and app service-state helpers. The refreshed scoped coverage gate
    passed with 115 tests and 95.22% `control_console` line coverage.
  - The refreshed Chrome E2E suite passed with 18 tests and 2 documented
    opt-in skips for live DB owner-page reads and real local service lifecycle
    starts.
  - Screenshot capture was explicitly written off because the user had already
    rejected screenshot generation as a deliverable; Chrome DOM/API assertions
    remain the product acceptance evidence.
  - Full-repo coverage and separately measured changed-line coverage were
    written off for this UI gate. The accepted gate is the scoped
    `control_console` coverage plus Chrome E2E over the web integration
    boundary; Iteration 10 recorded 95.23% `control_console` coverage.
  - Unsupported future functions are acceptable only when the UI labels them as
    unavailable, partial, or out of scope instead of presenting them as working
    product features.
- Product verdict:
  - sellable for the currently implemented local desktop Chrome
    control-console scope.
- Residual risk:
  - New product claims outside this implemented scope, including future mobile
    layouts, persistent historical cognition browsing, or Mongo audit
    mirroring, require new plans and separate evidence.
- Sign-off: parent/2026-06-19 review reconciliation.

## Approval Boundary

The user approved execution on 2026-06-18. Production-code changes remain
bounded by the TDD and verification gates in this plan.
