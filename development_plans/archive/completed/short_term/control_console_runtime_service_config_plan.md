# control console runtime service config plan

## Summary

- Goal: Add a generic control-console service configuration architecture and UI/API surface that lets operators apply ephemeral runtime overrides by automatically restarting the target service, with `.env` remaining the default source.
- Plan class: medium
- Status: completed
- Mandatory skills: `development-plan`, `py-style`, `test-style-and-execution`, `test-driven-development`, `build-web-apps:shadcn`, `build-web-apps:frontend-testing-debugging`
- Overall cutover strategy: compatible inside the control console. Existing service lifecycle behavior remains, and services without descriptors are unchanged.
- Highest-risk areas: accidentally touching adapter/brain code, misleading UI around restart-based apply, stale overrides after failed restart, unsafe config validation, and hiding config changes from audit.
- Acceptance criteria: the console exposes descriptor-driven generic per-service configuration controls, `adapter.napcat.active_groups` is only the first concrete descriptor, a fake non-NapCat descriptor proves extensibility in tests, UI overrides are not persisted across console restart, saving a running service config automatically restarts that service, and all implementation changes stay inside `src/control_console`, `tests/test_control_console*`, `tests/control_console_e2e`, and `src/control_console/README.md`.

## Context

The operator needs QQ active groups to be configurable from the web console.
The active group allowlist controls whether the NapCat adapter treats a QQ
group as active or listen-only. The user explicitly constrained the impact
radius to the control console and accepted restart-based apply. Therefore this
plan must not add runtime config endpoints to adapters or modify adapter
source code.

The control console already owns service registry loading, service lifecycle,
child-process environment overlays, local audit JSONL, authenticated/CSRF
state-changing API calls, static UI, and process state. The console can apply
service config by restarting a console-owned child process with adjusted argv
or environment. For NapCat, the console can preserve `.env` as the default
source by reading a canonical config variable and translating it into the
existing adapter CLI shape `--channels <group ids>` at service start. No
adapter code change is required.

## Mandatory Skills

- `development-plan`: load before editing this plan, executing stages, or reporting completion.
- `py-style`: load before editing Python production files or tests.
- `test-style-and-execution`: load before adding, changing, running, or interpreting pytest tests.
- `test-driven-development`: every production behavior change must have a focused failing test first.
- `build-web-apps:shadcn`: apply before editing static UI structure, component anatomy, density, and controls.
- `build-web-apps:frontend-testing-debugging`: apply before rendered-browser validation of the local UI.

## Mandatory Rules

- After automatic context compaction, reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run the independent code-review gate and record the result in `Execution Evidence`.
- Do not read `.env` through tools. Production code may use the existing `python-dotenv` pattern to load configured defaults.
- Do not modify files outside the approved change surface. In particular, do not modify `src/adapters/**`, `src/kazusa_ai_chatbot/**`, top-level `README.md`, `docs/HOWTO.md`, or non-control-console tests.
- Do not add runtime config endpoints to adapters, brain, or workers in this plan.
- Do not introduce Node.js, npm, pnpm, yarn, React, Vue, Vite, Webpack, Tailwind build tooling, or frontend package-manager workflows.
- UI controls must follow shadcn component family anatomy using the existing buildless HTML/CSS/JS stack: Button, Card, Badge, Input, Field/Form grouping, Dialog/Sheet-style panel, Separator, and Table where appropriate.
- The UI must clearly state when applying a config will restart a service. Do not label the action as zero-downtime or hot-apply.
- Config overrides are ephemeral. They may live in console process memory and may affect later service starts inside the same console process, but they must not be written to disk as durable operator settings.
- Audit config view, apply, reset, apply failure, and auto-restart attempts without exposing secrets or raw environment dumps.

## Must Do

- Add a generic service configuration descriptor system inside `control_console`; descriptors, snapshots, validation, API routes, and UI rendering must not be named around NapCat.
- Add a process-local runtime override store for service config values.
- Add a validated descriptor for `adapter.napcat.active_groups`.
- Add focused tests using a fake non-NapCat service descriptor, such as `adapter.fake.enabled`, to prove the same descriptor, snapshot, apply/reset, and UI-rendering path works without NapCat-specific logic.
- Read the default `adapter.napcat.active_groups` from `NAPCAT_ACTIVE_GROUPS` using the same `.env`-then-process-environment precedence style already used by console settings.
- Convert effective NapCat active groups into the existing adapter CLI arguments `--channels <group ids>` when the supervisor starts `adapter.napcat`.
- Add authenticated and CSRF-protected config APIs for get, apply, and reset.
- When applying config to a running console-owned service, automatically restart that service after storing the ephemeral override.
- When applying config to a stopped service, store the ephemeral override and show that it will be used on next start.
- Add a generic UI configure entrypoint on service cards, not a NapCat-only panel.
- Add a service configuration dialog/sheet that renders fields from descriptor metadata, can render at least string-list and boolean descriptor fields, shows default/effective/override state, and makes restart behavior obvious before apply.
- Ensure `src/control_console/static/console.js` contains no `if service.id === "adapter.napcat"` style special cases. Service-specific labels and values must arrive through API payloads.
- Add tests proving validation, default source precedence, command rendering, API auth/CSRF, audit, restart behavior, UI rendering, and failure behavior.
- Update `src/control_console/README.md` with the new config contract and UX semantics.

## Deferred

- Do not implement true hot mutation inside a running adapter process.
- Do not add adapter runtime config APIs.
- Do not add persistent config storage.
- Do not implement config descriptors for Discord, debug adapter, brain, workers, or support services beyond generic framework support.
- Do not hard-code NapCat-specific UI structure, API routes, request models, response models, state keys, or route names. The only NapCat-specific production data allowed is the descriptor entry and its command-overlay mapping.
- Do not build a command/env editor in the UI.
- Do not expose secrets, raw `.env` values, full environment dictionaries, tokens, prompts, message bodies, embeddings, or raw memory through config APIs or UI.
- Do not redesign the service registry format.
- Do not change cognition, RAG, message envelope, dispatcher, adapter transport, or brain service behavior.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Existing lifecycle APIs | compatible | Keep `start`, `stop`, and `restart` endpoints and UI behavior working. |
| Services without config descriptors | compatible | Show no configure button and keep their start command unchanged. |
| NapCat active groups | compatible | Add a console-owned config path that renders `--channels` on start; do not require adapter changes. |
| Config persistence | bigbang | Use ephemeral runtime overrides only. Do not create durable config files. |
| UI config surface | bigbang | Add the generic config dialog as the only operator-facing config editor. Do not add ad hoc per-service input rows inside cards. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- The agent must not preserve or add alternate adapter config paths beyond the descriptor system listed here.
- Any change to touch adapter, brain, or top-level docs requires user approval before implementation.

## Target State

The Services page remains the operator's primary lifecycle page. Each service
card shows service status and lifecycle actions. If a service has a descriptor,
the card also shows a `Configure` button and a small config-state badge such as
`default`, `override active`, or `apply failed`.

Opening `Configure` displays a shadcn-style dialog/sheet. For NapCat it shows
one field:

```text
Active QQ groups
Default source: NAPCAT_ACTIVE_GROUPS
Effective value: <group list or all groups listen-only>
Runtime override: <empty or edited group list>
Apply behavior: service restart required
```

Applying while `adapter.napcat` is running writes the override into the
process-local store, audits the change, automatically restarts
`adapter.napcat`, then refreshes bootstrap and the service card. Applying while
stopped stores the override and shows `will apply on next start`. Reset clears
the override and restarts the running service back to `.env` default.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Impact radius | Control console only | User explicitly requires no adapter/brain code changes. |
| Apply semantics | Restart-based apply | True hot apply requires changing adapter internals. Restart is accepted. |
| Persistence | Process-local only | User said dynamic changes are not persistent. |
| Default source | `.env`/environment variable | User wants `.env` to keep default groups. |
| NapCat concrete field | `active_groups` from `NAPCAT_ACTIVE_GROUPS` | Maps to the existing adapter CLI `--channels` without adapter changes. |
| UI shape | Generic configure dialog per service | Supports future descriptors without creating a NapCat-only page. |
| Generic framework | Descriptor-driven rendering and start overlays | Future services can add descriptors and overlay renderers inside control console. |

## Contracts And Data Shapes

### Descriptor Contract

Create `src/control_console/service_config.py` with public types and helpers:

```python
class ServiceConfigField(BaseModel):
    key: str
    label: str
    description: str
    value_type: Literal["string_list", "string", "boolean", "integer", "enum"]
    default_env: str | None = None
    sensitive: bool = False
    restart_required: bool = True
    max_items: int = 50
    max_item_length: int = 120
    pattern: str | None = None

class ServiceConfigDescriptor(BaseModel):
    service_id: str
    title: str
    description: str
    fields: list[ServiceConfigField]
```

The production descriptor registry must be generic. The initial production
descriptor set must include only:

```python
adapter.napcat:
  active_groups:
    type: string_list
    default_env: NAPCAT_ACTIVE_GROUPS
    pattern: ^[0-9]{1,32}$
    restart_required: true
```

Tests must also register a fake non-NapCat descriptor through the same public
descriptor registry or factory:

```python
adapter.fake:
  enabled:
    type: boolean
    default_env: FAKE_ADAPTER_ENABLED
    restart_required: true
```

The fake descriptor is test-only and must not be exposed in the production
built-in service registry.

### Snapshot API Shape

`GET /api/services/{service_id}/config` is generic and returns descriptor-driven
field snapshots for any service that has a descriptor:

```json
{
  "service_id": "adapter.napcat",
  "title": "NapCat QQ adapter",
  "apply_behavior": "restart",
  "state": "default|override_active|apply_failed|unavailable",
  "fields": [
    {
      "key": "active_groups",
      "label": "Active QQ groups",
      "description": "QQ groups where the adapter may visibly participate.",
      "value_type": "string_list",
      "default_source": "NAPCAT_ACTIVE_GROUPS",
      "default_value": ["54369546"],
      "override_value": null,
      "effective_value": ["54369546"],
      "restart_required": true,
      "sensitive": false,
      "validation": {"pattern": "^[0-9]{1,32}$", "max_items": 50}
    }
  ]
}
```

### Apply API Shape

`PUT /api/services/{service_id}/config` is generic and accepts values keyed by
descriptor field key:

```json
{
  "reason": "operator console action",
  "expected_version": 18,
  "values": {
    "active_groups": ["54369546", "905393941"]
  }
}
```

Response:

```json
{
  "service_id": "adapter.napcat",
  "config": {"...": "same snapshot shape"},
  "service": {"...": "ServiceRuntimeState"},
  "restart": {
    "attempted": true,
    "succeeded": true,
    "reason": "config apply requires restart"
  },
  "audit_event_id": "cc-audit-..."
}
```

`POST /api/services/{service_id}/config/reset` clears the override for the
service, restarts if running, and returns the same response shape.

### Supervisor Overlay Contract

The supervisor start path must receive a control-console-owned overlay
resolver. Overlay rendering must be service-specific behind a generic
interface. The only initial production renderer is for `adapter.napcat`; the
tests must include a fake non-NapCat renderer or no-op renderer to prove the
interface is not NapCat-shaped.

For `adapter.napcat`, the effective `active_groups` list becomes:

```text
python -m adapters.napcat_qq_adapter --channels <group1> <group2>
```

When the effective list is empty or unset, do not append `--channels`; the
adapter remains in its documented listen-only group mode.

## LLM Call And Context Budget

No LLM prompt, model call, graph, RAG, cognition, dialog, evaluator, or
background LLM behavior changes are allowed. Before and after LLM call count is
unchanged.

## Change Surface

### Create

- `src/control_console/service_config.py`: generic descriptor definitions, env/default resolution, validation, snapshot projection, ephemeral override store, descriptor registry, and command overlay rendering interface. NapCat-specific logic is limited to one descriptor registration and one renderer mapping `active_groups` to `--channels`.
- `tests/test_control_console_service_config.py`: focused deterministic tests for generic descriptors, a fake non-NapCat descriptor, validation, snapshots, overrides, reset, and command overlay rendering.

### Modify

- `src/control_console/contracts.py`: add strict request/response models for service config APIs.
- `src/control_console/app.py`: add config routes, audit events, bootstrap config summary wiring if needed, and restart orchestration.
- `src/control_console/supervisor.py`: accept and apply command/env overlays at service start without changing registry validation or shell safety.
- `src/control_console/static/index.html`: add generic configure dialog/sheet markup.
- `src/control_console/static/console.js`: render configure buttons, fetch config snapshots, edit list fields, apply/reset config, and refresh service state.
- `src/control_console/static/styles.css`: style the dialog/sheet, list editor, status badges, and restart warning using existing shadcn-like component anatomy.
- `src/control_console/README.md`: document service config contracts, ephemeral override behavior, restart semantics, and audit behavior.
- `tests/test_control_console_service_registry.py`: update or add focused expectations only if start command overlay changes observable registry assumptions.
- `tests/test_control_console_web_surface.py`: add static/UI contract checks for configure dialog elements and no misleading hot-apply wording.
- `tests/test_control_console_supervisor.py`: add supervisor start overlay tests.
- `tests/test_control_console_app.py` or the existing control-console route test file: add config API auth/CSRF/audit/restart tests.
- `tests/control_console_e2e/*`: add one Chrome/browser E2E test if the existing harness can run it without live credentials.

### Keep

- Keep `src/adapters/**` unchanged.
- Keep `src/kazusa_ai_chatbot/**` unchanged.
- Keep `docs/HOWTO.md` unchanged.
- Keep top-level `README.md` unchanged.
- Keep service registry JSON override format unchanged.

## Overdesign Guardrail

- Actual problem: operators need to change adapter/service runtime config from the control console without editing files or commands, while keeping changes non-persistent and control-console scoped.
- Minimal change: descriptor-driven config UI/API inside `control_console`, process-local overrides, and restart-based apply through a generic supervisor command-overlay interface.
- Ownership boundaries: the console owns operator configuration UX, validation, audit, and child-process restart; services continue owning runtime behavior once started; adapters and brain are not modified.
- Rejected complexity: no hot in-process adapter mutation, no adapter runtime config API, no persistent settings database, no command editor, no generic raw env editor, no plugin system, no schema registry outside control console, no production config descriptors for services not needed by this request, and no NapCat-only API/UI shortcut.
- Evidence threshold: add adapter runtime config APIs or persistent config only after a user-approved follow-up requires zero-restart apply or durable operator-managed settings.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside `src/control_console`, `tests/test_control_console*`, `tests/control_console_e2e`, and `src/control_console/README.md` as prohibited unless the user approves a plan update.
- The responsible agent must treat NapCat-specific branching in generic API or UI code as a plan violation. NapCat-specific behavior belongs only in descriptor registration and overlay rendering.
- The responsible agent must search for existing helper behavior before adding helpers and must not add thin wrappers around simple access.
- The responsible agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If a required instruction is impossible inside the approved change surface, stop and report the blocker instead of inventing an adapter-side substitute.

## Implementation Order

1. Parent writes focused tests for `service_config.py` contracts, including the fake non-NapCat descriptor, and verifies they fail because the module does not exist.
2. Parent writes route/API tests for config get/apply/reset and verifies the missing endpoints fail.
3. Production-code subagent implements `service_config.py`, contract models, app routes, and supervisor overlays inside `src/control_console`.
4. Parent reruns focused tests and fixes only test-contract gaps inside the approved surface.
5. Parent adds or updates UI static tests for configure controls and restart wording.
6. Production-code subagent or parent implements static UI changes inside `src/control_console/static`.
7. Parent runs rendered-browser validation through the available Browser plugin or records the plugin blocker and uses the existing Playwright harness if available.
8. Parent runs full focused verification and records evidence.
9. Parent starts one independent code-review subagent after verification passes.
10. Parent remediates review findings inside the approved surface and reruns affected verification.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the focused test contract is established; owns production code changes only; does not edit tests unless the parent explicitly directs it; closes after planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks, and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after planned verification passes; reviews the plan, diff, and evidence; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - focused service-config contract tests established.
  - Covers: descriptor, fake non-NapCat descriptor, default resolution, override store, validation, reset, and NapCat command overlay expected failures.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_control_console_service_config.py -q`
  - Evidence: record failing tests before production implementation.
  - Sign-off: `Codex/2026-06-18` after evidence is recorded.
- [x] Stage 2 - config backend implemented inside `src/control_console`.
  - Covers: `service_config.py`, contracts, app routes, audit, supervisor overlays.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_control_console_service_config.py tests\test_control_console_supervisor.py -q`
  - Evidence: record pass/fail and changed files.
  - Sign-off: `Codex/2026-06-18` after evidence is recorded.
- [x] Stage 3 - generic UI implemented and tested.
  - Covers: `static/index.html`, `static/console.js`, `static/styles.css`, static web-surface tests, one browser/E2E check.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_control_console_web_surface.py -q`
  - Evidence: record UI test output and browser validation result.
  - Sign-off: `Codex/2026-06-18` after evidence is recorded.
- [x] Stage 4 - integration and regression verification complete.
  - Covers: app route tests, auth/CSRF, audit, automatic restart, reset, stopped-service apply, failure path.
  - Verify: focused commands listed in `Verification`.
  - Evidence: record every command output summary.
  - Sign-off: `Codex/2026-06-18` after evidence is recorded.
- [x] Stage 5 - independent code review complete.
  - Covers: full diff, plan alignment, UI ergonomics, security, audit, restart semantics, no out-of-scope files.
  - Verify: independent review completed, findings resolved or documented, affected tests rerun.
  - Evidence: record review findings and rerun commands.
  - Sign-off: `Codex/2026-06-18` after evidence is recorded.

## Verification

### Static Scope Checks

- `git status --short` must list only files under `src/control_console/`,
  `tests/test_control_console*`, `tests/control_console_e2e/`,
  `development_plans/README.md`, and
  `development_plans/archive/completed/short_term/control_console_runtime_service_config_plan.md`.
- `git diff -- src/adapters src/kazusa_ai_chatbot docs README.md tests/test_runtime_adapter_registration.py`
  must return no diff. Existing adapter and HOWTO references to
  `NAPCAT_ACTIVE_GROUPS` are historical context; this plan must not modify or
  add out-of-scope references.
- `rg -n "adapter\\.napcat|active_groups|NAPCAT_ACTIVE_GROUPS" src/control_console/static/console.js src/control_console/static/index.html` must return no matches. Service-specific labels and values must be rendered from API payloads, not hard-coded in the static UI.
- `rg -n "/api/(napcat|adapters/napcat|services/adapter\\.napcat)" src/control_console tests` must return no matches. Config routes must use `/api/services/{service_id}/config`.

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests\test_control_console_service_config.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_control_console_supervisor.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_control_console_web_surface.py -q`
- Route test command for the config API file added or updated during implementation.

### Integration Tests

- Existing service lifecycle tests that cover start/restart must pass after overlay integration.
- Existing auth/CSRF control-console tests must pass after adding config routes.
- Browser/E2E validation must click `Configure`, edit active groups, apply, observe restart/apply feedback, reset to default, and observe updated state. If Browser plugin is unavailable, record the blocker and run the existing Playwright harness in Chrome.

### Static Checks

- `venv\Scripts\python.exe -m py_compile src\control_console\service_config.py src\control_console\contracts.py src\control_console\app.py src\control_console\supervisor.py`
- `git diff --check`

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, documentation, and static UI artifact.
- Alignment with the control-console-only change surface.
- Correctness of restart-based apply, reset, failure rollback messaging, audit events, and CSRF/auth protection.
- UI quality from a customer perspective: the dialog must not feel like a raw config editor, must explain restart behavior, must show default vs override vs effective values, and must avoid empty/compressed layouts.
- Regression and handoff quality, including focused tests, static scope checks, browser validation, and execution evidence.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface. If a finding requires adapter, brain, top-level
docs, persistent settings, or hot runtime apply, stop and request approval.

## Acceptance Criteria

This plan is complete when:

- The Services UI exposes a generic Configure action only for services with descriptors.
- `adapter.napcat.active_groups` shows default groups from `NAPCAT_ACTIVE_GROUPS`, supports UI override, and can reset to default.
- A fake non-NapCat descriptor test proves the same backend and snapshot machinery supports another service/field without adding NapCat-specific code paths.
- Static UI code has no NapCat-specific conditionals, route names, or field names.
- Applying while `adapter.napcat` is running automatically restarts only `adapter.napcat` and starts it with the effective active groups rendered as `--channels`.
- Applying while `adapter.napcat` is stopped stores an ephemeral override and the next start uses it.
- Runtime overrides are not persisted to disk and are gone after console process restart.
- Config apply/reset actions are authenticated, CSRF-protected, validated, and audited.
- UI clearly communicates restart behavior, default source, override state, apply success, apply failure, and reset.
- Verification confirms no adapter, brain, top-level docs, or non-control-console tests were changed.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| UI appears to hot-apply without restart | Label action `Apply and restart` for running services and show restart badge | UI static and browser tests inspect wording/state |
| Override accidentally persists | Keep overrides in process memory only | Unit test constructs a new store and verifies no previous override |
| Invalid group ids reach command args | Validate each id with `^[0-9]{1,32}$` and max item count | Validation tests reject invalid values |
| Supervisor command overlay weakens argv safety | Overlay returns argv list parts only and never shell strings | Supervisor tests inspect exact argv |
| Failed restart leaves customer confused | Return restart failure state and audit event; UI shows failure and current service state | Route and UI tests cover failure path |
| Scope creep into adapters | Static scope check and independent code review gate | `git diff --name-only` and `rg` checks |

## Execution Evidence

- 2026-06-18 Stage 1 RED:
  `venv\Scripts\python.exe -m pytest tests\test_control_console_service_config.py -q`
  failed with 5 expected failures, all caused by
  `ModuleNotFoundError: No module named 'control_console.service_config'`.
  The tests cover descriptor snapshots, ephemeral override reset, validation,
  fake non-NapCat descriptor behavior, and generic command-renderer behavior.
- 2026-06-18 API RED:
  `venv\Scripts\python.exe -m pytest tests\test_control_console_config_routes.py -q`
  failed with 4 expected failures because the generic
  `/api/services/{service_id}/config` and
  `/api/services/{service_id}/config/reset` routes do not exist yet. The tests
  cover auth/CSRF, validation, stopped-service apply without restart,
  running-service apply with restart, reset, and audit event expectations.
- 2026-06-18 UI RED:
  `venv\Scripts\python.exe -m pytest tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q`
  failed at the first new assertion because `id="service-config-dialog"` is
  not present yet. The new assertions cover generic dialog markup, restart
  wording, config JS handlers, and absence of hard-coded NapCat field/env
  strings in static UI assets.
- 2026-06-18 supervisor RED:
  `venv\Scripts\python.exe -m pytest tests\test_control_console_supervisor.py::test_start_service_uses_command_overlay_for_subprocess_argv -q`
  failed with `TypeError: ProcessSupervisor.__init__() got an unexpected
  keyword argument 'command_resolver'`, proving the argv overlay seam is not
  implemented yet.
- 2026-06-18 implementation verification:
  `venv\Scripts\python.exe -m pytest tests\test_control_console_service_config.py tests\test_control_console_config_routes.py tests\test_control_console_supervisor.py tests\test_control_console_web_surface.py -q`
  passed: 33 passed. This covered descriptor snapshots, validation, command
  overlay, config API auth/CSRF/apply/reset/audit, supervisor lifecycle
  regression, and static web-surface contracts.
- 2026-06-18 rendered UI verification:
  Browser plugin was installed but the in-app `iab` browser instance was
  unavailable. Fallback used installed Chrome through temporary Playwright
  automation without adding frontend tooling to the repository. Flow tested:
  login -> Services -> Configure -> edit active group list -> apply stopped
  service override -> reset. Result: 4 service cards, 1 Configure button,
  dialog title `NapCat QQ adapter`, initial field `54369546`, apply notice
  `configuration saved`, state `override active`, reset notice
  `configuration reset`, state `default`, and no browser console warnings or
  errors.
- 2026-06-18 regression verification:
  `$files = Get-ChildItem -LiteralPath 'tests' -Filter 'test_control_console*.py' | ForEach-Object { $_.FullName }; venv\Scripts\python.exe -m pytest $files -q`
  passed: 84 passed. This covered the deterministic control-console test
  suite including auth, bootstrap, lifecycle routes, stream, repository,
  redaction, service registry, config routes, supervisor, and web surface.
- 2026-06-18 static verification:
  `venv\Scripts\python.exe -m py_compile src\control_console\service_config.py src\control_console\contracts.py src\control_console\app.py src\control_console\supervisor.py`
  passed. `git diff --check` passed with only line-ending warnings. `rg -n
  "adapter\.napcat|active_groups|NAPCAT_ACTIVE_GROUPS"
  src\control_console\static\console.js src\control_console\static\index.html`
  returned no matches. `rg -n
  "/api/(napcat|adapters/napcat|services/adapter\.napcat)"
  src\control_console tests` returned no matches.
- 2026-06-18 changed-file scope:
  `git status --short` listed only `development_plans/README.md`,
  `development_plans/archive/completed/short_term/control_console_runtime_service_config_plan.md`,
  files under `src/control_console/`, and `tests/test_control_console*`.
- 2026-06-18 independent review:
  The single independent reviewer found one high-severity issue and one
  low-severity issue. High: restart failures were returned and audited like
  successful config applies, and UI notices still said success. Low: one PEP 8
  top-level spacing violation in `src/control_console/app.py`. Fix: added
  process-local `apply_failed` state tracking, route failure auditing for
  apply/reset restart failures, UI error notices when `restart.succeeded` is
  false, and the missing blank line. Added focused tests for apply restart
  failure and reset restart failure.
- 2026-06-18 review-fix verification:
  `venv\Scripts\python.exe -m pytest tests\test_control_console_config_routes.py::test_apply_config_restart_failure_returns_apply_failed_state -q`
  passed. `venv\Scripts\python.exe -m pytest tests\test_control_console_config_routes.py::test_reset_config_restart_failure_returns_apply_failed_state -q`
  passed. `venv\Scripts\python.exe -m pytest tests\test_control_console_service_config.py tests\test_control_console_config_routes.py tests\test_control_console_supervisor.py tests\test_control_console_web_surface.py -q`
  passed: 35 passed. Full deterministic control-console suite rerun passed:
  86 passed. `venv\Scripts\python.exe -m py_compile
  src\control_console\service_config.py src\control_console\contracts.py
  src\control_console\app.py src\control_console\supervisor.py` passed.
  `git diff --check` passed with line-ending warnings only. Static UI
  hard-code and concrete-route `rg` checks returned no matches. Out-of-scope
  diff check for adapters, brain, docs, top-level README, and runtime adapter
  registration returned no diff.
