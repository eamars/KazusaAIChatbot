# cognition_graph_multi_source_latest_run_plan

## Summary

- Goal: make `Latest cognition run` represent every production execution of
  the shared cognition system, regardless of its trigger source, through one
  canonical snapshot and one Overview widget.
- Plan class: medium cross-layer runtime-observability and control-console
  contract change.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `control-console-web-development`, `py-style`,
  `test-style-and-execution`.
- Dependency: the completed
  `cognition_graph_semantic_observability_plan.md`. This plan changes its
  latest-run ownership and source coverage; it does not reopen its semantic
  detail or visual-directive work.
- Overall cutover strategy: one coordinated brain/API/source-publication
  cutover. The completed frontend graph widget, semantic inspector, layout,
  and visual-node presentation are reused unchanged. No static HTML/JS
  widget change is required; the backend/API stops supplying a second latest
  graph and the existing hidden self-card path remains inert.
- Highest-risk areas: missing a non-`/chat` cognition caller, publishing a
  resolver cycle instead of one completed run, allowing concurrent runs to
  produce misleading latest state, and leaving a second self-specific widget
  or API field behind.
- Acceptance criteria: user-message, accepted-task-result, and self-cognition
  executions that reach the shared cognition runtime publish their actual
  graph under one latest snapshot with bounded trigger metadata; the existing
  widget renders each source graph without source-specific UI; pre-cognition
  skips do not fabricate runs; failure and malformed-source cases remain
  truthful; and the browser sign-off shows one widget with real data.

## Plan Review Record

- Frontend scope was over-stated. The shared graph widget, semantic inspector,
  layout, and visual-directive presentation are already complete. The plan now
  excludes static HTML/JS widget changes; the existing hidden self-card path
  remains inert when no self-specific payload is supplied.
- Source presentation was over-designed. The plan now requires source
  metadata at the graph/API boundary but adds no source badge, selector,
  source-specific card, renderer branch, or new empty state.
- Malformed-source behavior was previously ambiguous. An admitted run with
  missing/unsupported source metadata now publishes a minimal `partial`
  snapshot using `trigger_source=not_reported` and a safe reason; it never
  falls back to `user_message` and never silently leaves a prior run looking
  current.
- Failure behavior was incomplete when no graph state exists. The plan now
  requires a minimal empty-node failed/partial snapshot for an admitted
  failure, while keeping raw exception details in operational logs only.
- The source matrix now distinguishes live production owners from
  test-only dry-run seams and typed-but-unused source values, preventing the
  implementation from inventing unsupported runtime features.

## Context

The source boundary already exists in `CognitiveEpisode`:

```text
CognitiveEpisode.trigger_source
        + input_sources
        + existing source-specific state/artifacts
                    |
                    v
        shared cognition execution
                    |
                    v
        deterministic cognition graph projection
                    |
                    v
        one process-local latest cognition snapshot
                    |
                    v
        /ops/latest-cognition-graph -> Overview Latest widget
```

`stage_1_goal_resolver` accepts a source-bearing cognition state and invokes
the shared resolver/cognition path. The current publication boundary is
outside that source-neutral design: `service.py` records the latest graph only
from the normal `/chat` success and failure path. The accepted-task result
delivery path invokes the same persona cognition flow but does not publish its
graph through that latest path. The self-cognition worker invokes the shared
resolver through its own runner and publishes a second
`self_cognition_graph` snapshot.

The current typed trigger-source contract contains:

```text
user_message
reflection_signal
internal_thought
scheduled_recall
system_probe
accepted_task_result_ready
```

The production call-site inventory in this repository currently identifies
`user_message` for normal chat, `accepted_task_result_ready` for accepted-task
delivery, and `internal_thought` for the self-cognition worker as live paths
that execute the shared runtime. The reflection and internal-thought dry-run
helpers are currently test-only injected-call seams; `scheduled_recall` and
`system_probe` are typed source values without a production shared-cognition
caller in this checkout. The implementation must repeat this inventory before
editing and include every production caller found. It must not invent latest
runs for source values that did not execute cognition.

The `/ops/latest-cognition-graph` response currently contains parallel
`cognition_graph` and `self_cognition_graph` fields. The control console
fetches both and the Overview markup contains a primary latest card plus a
self-cognition card. That parallel shape is the source of duplicate or
source-specific latest views. The existing `renderCognitionGraph` renderer is
already reusable and remains the only graph widget owner.

## Mandatory Skills

- `development-plan`: keep this document as the execution boundary and keep
  it `in_progress` for this user-authorized implementation run.
- `local-llm-architecture`: preserve the shared cognition boundary, keep
  source identity in the typed episode contract, and make publication
  deterministic without adding a semantic model step.
- `control-console-web-development`: reuse the completed FastAPI console
  transport/widget ownership, remove only stale duplicate latest plumbing if
  required by the contract, and validate the unchanged renderer with browser
  checks and screenshots.
- `py-style`: load the project Python policy before changing Python runtime,
  contract, or test files.
- `test-style-and-execution`: use deterministic patched tests for publication,
  contracts, failure paths, and browser fixtures; inspect any live run one
  case at a time.

## Mandatory Rules

- This document is a draft discussion and implementation contract. Production
  edits require the user's explicit implementation command and promotion to
  `approved` or `in_progress`.
- The parent agent performs implementation, testing, browser validation, and
  review itself. No subagent is used, per the user's explicit instruction.
- Recheck `git status --short`, the repository README, `docs/HOWTO.md`, the
  relevant subsystem READMEs, and all direct source/test files before
  implementation. Do not read `.env`.
- Treat `/chat` as one transport/caller, not as the definition of a cognition
  source. Read `trigger_source` from the existing validated
  `CognitiveEpisode`; do not infer it from platform, URL, case name, or UI
  page.
- Publish one terminal graph snapshot per source execution. Resolver retries,
  capability observations, and repeated `call_cognition_subgraph` cycles are
  internal steps of that run, not separate latest runs.
- Use one canonical latest storage field, one latest API graph field, and one
  Overview latest widget. Do not retain a parallel self-cognition latest card,
  source-specific card, source filter, history view, or aggregate timeline.
- Preserve the completed semantic-detail contract, the existing visual
  directive node behavior, and the shared inspector. This plan changes source
  coverage and latest ownership only.
- Add no LLM call, prompt field, model route, cognition decision, RAG query,
  memory write, adapter field, database collection, or persistence schema.
- Keep raw prompts, raw model output, embeddings, message envelopes, internal
  identifiers, and exception traces outside the graph detail payload.
- Keep source metadata bounded and safely serialized. A future typed source
  must pass through the transport without requiring a new frontend branch.
- Treat the existing frontend graph widget as complete. Do not modify its
  HTML, renderer, selected-detail contract, CSS/layout, visual-directive node,
  or text rendering for this plan. The only console-side changes are backend
  transport/projection changes needed to feed the existing widget.
- Deploy/restart the brain and control console from the same revision for
  verification. The coordinated cutover does not add a compatibility mapper
  for mixed old/new latest payloads.

## Must Do

### 1. Make latest publication source-neutral

Introduce one deterministic publication seam around the existing graph
builders and process-local latest storage. The seam must accept the already
projected graph plus the existing episode identity needed to attach
`trigger_source` and `input_sources`, deep-copy the bounded snapshot, and
replace the single latest value.

The seam must be called exactly once after each current production source has
finished its cognition-owned work:

| Production source | Existing execution owner | Required publication point |
|---|---|---|
| `user_message` | normal persona `/chat` flow | existing success graph and existing cognition-failure graph |
| `accepted_task_result_ready` | accepted-task result delivery flow using `persona_supervisor2` | after success or source-owned failure graph is built |
| `internal_thought` | self-cognition runner/worker using the shared resolver | after the existing self graph artifacts are complete |

The publication seam must not be placed inside the LLM stage or at every
resolver-loop return. It belongs at the source runner's final graph handoff so
the snapshot contains every node the source actually executed and the actual
terminal statuses. Existing source-specific graph topology may remain
different where the source already has a different artifact shape; the widget
must render the supplied graph without a source-specific renderer.

The self-cognition publisher must write to the canonical latest field instead
of a second self-only field. Its existing artifacts, action handling,
consolidation, and runtime disablement remain unchanged.

### 2. Carry actual trigger identity in the graph snapshot

Add bounded top-level graph metadata separate from the console projection
`source` value:

```json
{
  "trigger_source": "internal_thought",
  "input_sources": ["internal_monologue"]
}
```

`source` continues to mean the console view/projection key such as
`overview_latest` or `debug_latest`; it must not be overloaded with a trigger
source. The existing graph widget does not need a new source badge, selector,
or source-specific branch. The selected node detail contract and graph layout
metadata remain as established by the completed observability plan.

The source metadata must come from the episode/state handed to the current
runner. The self path must receive that metadata through its existing runner
handoff or an updated internal publisher callback contract; it must not use a
case-name/platform heuristic. No new trigger-source values are introduced in
this plan.

### 3. Preserve the completed frontend widget and remove duplicate transport

Change the latest API and console transport as one contract update while
keeping the completed graph UI intact:

- `OpsLatestCognitionGraphResponse` exposes one canonical `cognition_graph`.
- The service removes the parallel `_latest_self_cognition_graph` storage and
  self-specific latest response branch.
- `KazusaClient` performs one latest fetch/projection; it does not fetch a
  second self latest graph.
- The console bootstrap and SSE invalidation track one latest run id.
- Overview continues to use the existing single `Latest cognition run` card
  and existing `renderCognitionGraph` widget.
- The existing self-cognition card remains hidden when its already-supported
  `not_reported` snapshot is absent. The canonical latest endpoint stops
  supplying a self-specific graph, so the hidden path cannot create a second
  visible latest run. Removing dead static markup is outside this plan.
- Debug cognition continues to use the same renderer for the debug response;
  it remains a page-local view of that debug request and is not a second
  Overview latest store.

Do not add a source badge, source selector, source-specific empty state, or
another graph component. A missing latest snapshot uses the existing honest
`not_reported` behavior.

### 4. Preserve actual graph content and status semantics

The latest snapshot must contain the existing graph nodes and edges produced
by the source's actual run. It must not combine multiple source runs into one
history-like graph and must not add synthetic nodes merely to make source
shapes identical.

- A source that reaches cognition and completes publishes its completed or
  source-appropriate partial graph.
- A source that reaches cognition and fails publishes the existing safe failed
  graph path with source metadata when enough state exists; it does not leave
  the last successful source looking current.
- A source skipped before cognition begins does not publish a fake graph.
- A source canceled after cognition begins publishes a truthful partial/failed
  snapshot if the existing state can be projected safely; pre-admission
  cancellation does not replace the latest run.
- A disabled visual stage keeps the existing grey/deactivated visual node
  behavior. This plan does not add or summarize visual-directive output.

### 5. Test source coverage and failure boundaries

Add deterministic tests for the publication seam and update existing graph,
API, console, and self-cognition tests. The tests must prove source coverage,
not only that a graph can be rendered:

- publish a `user_message` graph, then an `internal_thought` graph, and assert
  the latter is the one canonical latest snapshot with its source metadata;
- publish an `accepted_task_result_ready` graph and assert it follows the
  same contract;
- assert a repeated resolver cycle produces one publication;
- assert a pre-cognition skip does not change the prior latest snapshot;
- assert an admitted failure publishes a failed/partial source-bearing graph
  without raw exception text;
- assert missing, empty, unknown, overlong, and malformed source metadata are
  handled without a `/chat` fallback or render exception;
- assert the latest API has one canonical graph field and the existing console
  still has one Overview latest graph card;
- retain regression checks for the established semantic selected-detail rows,
  actual messages, and grey/deactivated visual node.

## Deferred

- Historical cognition-run storage, browsing, replay, or a latest-run history.
- A trigger-source filter, source comparison, aggregation, timeline, or
  per-source dashboard.
- New trigger sources or production wiring for typed-but-unused
  `reflection_signal`, `scheduled_recall`, or `system_probe`.
- Publishing test-only injected dry-run helper calls to the live process view
  without a production owner and explicit runtime contract.
- Redesigning graph topology or normalizing source-specific node ids.
- Changes to visual-directive prompts, enablement, logging, rendering, or
  selected-detail semantics already covered by the completed plan.
- Public adapter response changes, database persistence, or new telemetry
  storage.
- A new trigger-source badge, source selector, source comparison, or other
  frontend source-discovery feature.

## Cutover Policy

- Latest brain API payload: big-bang internal contract update. Replace the
  parallel `self_cognition_graph` response field with the canonical
  `cognition_graph` field while adding bounded trigger metadata.
- Process-local publication: big-bang ownership update. Replace dual latest
  storage with one canonical latest graph and update every current producer in
  the same revision.
- Console bootstrap/SSE transport: big-bang coordinated update. Remove the
  second latest identity and stale self fetch/state path with the backend
  contract. The rendered graph widget and static markup are unchanged.
- `/chat` response graph: compatible preservation. Keep the existing
  per-request `ChatResponse.cognition_graph` used by Debug; add only the
  source metadata needed by the shared graph projection.
- Rollback: revert the coordinated implementation revision and restart the
  brain and console together. No dual-write or old-field compatibility shim
  is introduced.

## Target State

```text
user_message --------------------+
accepted_task_result_ready ------+--> existing shared cognition
internal_thought ----------------+        |
future production source --------+        v
                                  source-owned final graph projection
                                             |
                                             v
                                  one canonical latest snapshot
                                             |
                       +---------------------+---------------------+
                       |                                           |
             Overview: existing Latest cognition run      Debug: same widget
             actual source graph rendered                 request-local graph
```

The latest snapshot means the most recently finalized cognition execution in
the process. It is one run at a time, not a merge of multiple runs. The
snapshot's trigger metadata identifies which source produced it, and the
existing graph nodes show all stages represented by that execution, including
actual skipped and failed stages.

## Design Decisions

1. **Episode metadata is authoritative.** `trigger_source` and
   `input_sources` are taken from the validated episode already entering
   cognition. Platform names, endpoint names, self-cognition case names, and
   UI page names are not source classifiers.
2. **One canonical latest store.** Self-cognition is a source of the shared
   latest run, not a separate latest product surface.
3. **One final publication per execution.** The resolver may run multiple
   cognition cycles internally, but publication occurs only after the source
   runner has its terminal graph state.
4. **Last finalized publication wins.** Concurrent source runs use the
   existing unique run identity and event-loop publication order. No source is
   given priority, and no history/ordering service is added.
5. **Skipped means no cognition.** Busy, empty, rejected, or otherwise
   pre-admission cases do not overwrite a real latest cognition run.
6. **Failure remains visible and bounded.** An admitted failure replaces the
   latest snapshot with a safe failed/partial graph. If no source graph state
   can be projected, publish a minimal empty-node failed/partial snapshot with
   source metadata and a safe reason, without exposing raw exceptions or
   prompts.
7. **The widget stays complete and generic.** The existing widget displays
   bounded graph data and has no branch for self-cognition, accepted tasks, or
   future sources. This plan supplies it a different source graph through the
   existing input contract instead of changing the widget.
8. **No semantic work is added.** Publication, source propagation, contract
   validation, and rendering are deterministic.

## Risks

| Risk | Mitigation and verification |
|---|---|
| A new or existing source still bypasses latest publication. | Re-run the production call-site inventory, maintain the source matrix in tests, and assert every current shared-cognition owner invokes the canonical seam exactly once. |
| Self-cognition remains visible through a second field/card. | Stop populating the self-specific backend/API field, remove its console transport fetch/invalidation path, preserve the existing hidden-state behavior, and assert only one latest card is visible in browser tests. |
| Resolver cycles overwrite the latest graph with intermediate state. | Publish only from the source runner's terminal handoff and assert one callback/publication per execution. |
| Concurrent chat, accepted-task, and self runs produce stale source state. | Publish immutable deep copies at terminal completion, use the existing unique run ids, and test interleaved completion order. |
| A future trigger source breaks the UI. | Treat graph trigger metadata as bounded transport data and keep the existing frontend free of source-specific conditionals. |
| Missing metadata is mislabeled as `/chat`. | Fail closed to `not_reported`/partial metadata and record a safe runtime diagnostic; never default to `user_message`. |
| Source-specific artifacts cannot build a complete graph. | Preserve the existing source graph builder, publish known nodes/statuses, and use honest partial/not-reported state instead of fabricated details. |
| Mixed brain/console revisions produce confusing payloads. | Use a coordinated restart and big-bang contract cutover; verify the deployed revision before browser sign-off. |
| Source metadata or existing semantic text injects markup or grows without bound. | Bound source metadata at the transport boundary, keep existing semantic redaction rules, escape through the existing renderer, and test multiline/special-character values. |
| The latest endpoint is unavailable or malformed. | Keep the existing console `not_reported` fallback, validate the single payload strictly, and assert no frontend render exception. |

## Failure-Mode Handling

| Failure mode | Required behavior | Verification |
|---|---|---|
| Normal `/chat` execution completes | Publish the existing response graph with `trigger_source=user_message` and its existing input sources. | Deterministic chat graph/publication test and real browser capture. |
| Accepted-task result cognition completes | Publish its graph through the same latest field with `trigger_source=accepted_task_result_ready`. | Accepted-task source fixture and latest endpoint assertion. |
| Self-cognition completes | Publish its existing self graph through the same latest field with `trigger_source=internal_thought`. | Self worker/runner test and Overview browser fixture. |
| Resolver requests multiple cycles/capability observations | Treat all cycles as one run and publish once after the terminal result. | Callback count and final run-id assertions. |
| Busy/empty/rejected case before cognition | Do not publish or alter the prior latest snapshot. | Before/after snapshot test for self and dry-run skip boundaries. |
| Exception after cognition admission | Publish a safe failed or partial graph. If no graph-safe state exists, publish a minimal empty-node failed/partial snapshot with the episode source metadata and safe reason; keep raw exception details in existing operational logging only. | Patched failure test asserts source/status/reason and absence of exception body in graph detail. |
| Cancellation before admission | Do not replace the current latest run. | Cancellation boundary test. |
| Cancellation after admission | Publish partial/failed state when safely projectable; never claim completed. | Pipeline cancellation fixture. |
| Missing or empty `trigger_source` | Publish a minimal `partial` snapshot when cognition was admitted, set the bounded metadata value to `not_reported`, attach a safe `trigger_source_missing` reason, and never map it to `user_message`. | Malformed publication inputs assert latest replacement, status/reason, and no UI exception. |
| Source value outside the current validated episode contract | Treat it as a publisher contract error: publish a minimal `partial` snapshot with `trigger_source=not_reported` and a safe diagnostic. Adding a new typed source remains a separate change; the existing widget requires no change for it. | Defensive unknown-source fixture and source-contract test. |
| Empty or malformed `input_sources` | Preserve the valid trigger metadata, omit invalid input-source values, and keep the graph renderable. | `null`, scalar, nested mapping, and empty-list fixtures. |
| Long or special-character source metadata | Bound only the source metadata field, preserve semantic graph text under the existing contract, escape HTML, and retain line breaks. | Unicode, quotes, `<script>`, emoji, CR/LF, and overlong fixtures. |
| Graph builder returns partial nodes or invalid detail | Keep valid nodes, mark the top-level state partial/not-reported as appropriate, and avoid stringifying arbitrary objects into misleading text. | Malformed graph payload projection tests and browser console inspection. |
| Old duplicate self payload appears during a mixed deployment | Deployment verification catches the revision mismatch before sign-off; the cutover does not add a second runtime mapper. | Revision check plus clean restart/browser test. |
| Latest run has no graph yet | Use the existing single-card `not_reported` empty state and do not render a source-specific placeholder card. | Fresh console bootstrap assertion. |

## Contracts/Data Shapes

The brain latest response becomes:

```json
{
  "cognition_graph": {
    "run_id": "existing-source-run-id",
    "status": "completed",
    "trigger_source": "internal_thought",
    "input_sources": ["internal_monologue"],
    "nodes": [],
    "edges": [],
    "redaction": {}
  }
}
```

`cognition_graph` remains optional/null when no production cognition run has
been published. `trigger_source` is a bounded string at the graph transport
boundary and uses `not_reported` only for a defensive malformed-publication
case. A future typed source would update the episode contract and reuse this
same graph field; it does not require a frontend schema branch.
`input_sources` is a bounded list of safe strings.
Neither field exposes `origin_metadata`, message envelopes, identifiers, or
model output.

The console projection continues to add its view-owned `source` value
(`overview_latest`, `debug_latest`, or `historical`) for renderer pinning and
layout state. It preserves the graph-owned `trigger_source` and
`input_sources`. Node ids, edges, statuses, selected semantic detail fields,
full approved text behavior, and visual-node disabled behavior remain the
existing completed contract.

No prompt or LLM budget changes are permitted: zero additional calls, zero
additional prompt tokens, and zero new model routes.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/service.py`
  - create the canonical source-bearing latest publication seam;
  - publish accepted-task result graphs on success/failure;
  - publish self-cognition graphs into the canonical field;
  - remove the parallel self latest storage/response branch;
  - attach episode source metadata without changing cognition behavior.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`
  - change the latest-graph response to one canonical graph field.
- `src/control_console/contracts.py`
  - validate bounded `trigger_source` and `input_sources` graph metadata.
- `src/control_console/kazusa_client.py`
  - project one latest graph and remove the separate self-latest fetch path if
    the coordinated one-field contract makes it obsolete;
  - preserve graph-owned trigger metadata.
- `src/control_console/app.py`
  - bootstrap and invalidate one latest graph/run id;
  - remove the self-latest API and SSE state path.
- `src/control_console/README.md`, relevant `docs/HOWTO.md`, and
  `development_plans/README.md`
  - document the source-neutral latest contract and register this plan.

### Do Not Modify

- `src/control_console/static/index.html` and
  `src/control_console/static/console.js`: the existing graph widget,
  semantic inspector, hidden self-card behavior, and shared rendering path
  are already complete for this change.

### Tests

- `tests/test_service_cognition_graph.py` or a focused companion module
  - cover graph source metadata, one publication per run, accepted-task and
    self publication, failure/skip boundaries, concurrent completion order,
    and malformed source fields.
- Existing accepted-task and self-cognition integration test modules
  - assert their real execution owners invoke the canonical publisher and
    preserve their current cognition/action/consolidation outcomes.
- Control-console contract/web-surface tests
  - assert the single latest response field, bounded source projection, and
    safe `not_reported` fallback.
- `tests/control_console_e2e/fake_brain.py` and
  `tests/control_console_e2e/test_cognition_graph_e2e.py`
  - use source-bearing graphs for user, accepted-task, and internal-thought
    states; assert one visible Overview widget/card, the unchanged shared
    renderer, existing semantic detail rows, and the existing hidden self
    panel. Assert source metadata at the API/graph snapshot boundary rather
    than adding source UI.
- Browser sign-off artifacts
  - capture real-data latest-run states through the requested Playwright
    workflow, including at least a normal user-message completion and a
    non-`/chat` source when that source can be triggered safely; capture
    failed/empty/disabled states only from actual runtime state and label any
    unavailable source honestly.

## Overdesign Guardrail

This change is limited to source propagation, final graph publication, the
single latest API payload, and removal of duplicate latest transport
ownership. It does not reimplement the frontend widget, add source discovery,
source configuration, filters, history, comparison, aggregation, a new
widget, a new graph topology, a new trigger type, a new agent, a prompt/model
change, a database schema, a public adapter field, or a second summary layer.
The existing shared widget receives the source-bearing snapshot it was
designed to render.

The plan also does not make test-only dry-run helpers appear in production
latest telemetry merely because they call an injected cognition function.
Only an actual production runner that owns the process-local latest view
publishes a run.

## Agent Autonomy Boundaries

The implementation agent may choose helper names, callback parameter names,
local function placement, and deterministic fixture values. The agent must
preserve the one-field latest API, existing Overview widget, episode-
authoritative source identity, one terminal publication per execution, no
pre-cognition publication, and all exclusions in this plan.

A request to add a trigger type, source filter/history, source-specific UI,
mixed-version compatibility mapper, prompt/model behavior, persistence, or
public adapter behavior requires a new user decision and a new/superseding
plan.

## Implementation Order

1. Obtain explicit implementation approval and promote this plan to
   `approved` or `in_progress`.
2. Recheck repository status and mandatory documentation/skills; repeat the
   production call-site inventory and record any newly discovered owner in
   the source matrix.
3. Add deterministic publication/source-contract tests for the canonical
   seam, source metadata, terminal publication, skip, failure, cancellation,
   and concurrent completion behavior.
4. Implement the source-neutral latest publication and wire every current
   production cognition owner, including accepted-task and self-cognition.
5. Update the brain latest response, console projection, and bootstrap/SSE
   invalidation in the same cutover. Keep static HTML/JS widget files
   unchanged.
6. Verify the existing shared renderer, semantic inspector, CSS, layout, and
   hidden self-card state without modifying them.
7. Update focused documentation and the plan checklist.
8. Run focused deterministic tests, regression tests, `git diff --check`, and
   Python compilation through `venv\\Scripts\\python`.
9. Restart clean dev brain/console processes and use the requested Playwright
   sign-off with real data across available source/status states. Inspect
   screenshots, the graph/API payload, and browser console output without
   adding source-specific UI assertions.
10. Perform an independent parent-only review of the final diff against every
    acceptance criterion, resolve findings, rerun affected checks, and record
    evidence before marking the plan complete.

## Execution Model

- Parent-led, sequential execution; no subagent.
- Production code changes happen only after user implementation approval and
  plan promotion.
- Deterministic tests use patched cognition seams and inspect every result.
- No live LLM test is needed unless an unrelated regression forces one; if a
  live check is required, run one case at a time and inspect its output.
- Browser validation uses the in-app Browser skill when available. If it is
  unavailable, use the permitted Playwright fallback and record that fact in
  the execution evidence.
- Real-data screenshots are sign-off evidence, not substitutes for contract
  and failure-mode tests.

## Progress Checklist

- [x] Repository and completed observability implementation audited.
- [x] Current production trigger-source and publication gaps recorded.
- [x] Single-widget and no-overdesign boundaries recorded.
- [x] User approves this plan and implementation scope.
- [x] Plan status is promoted to `in_progress`.
- [x] Mandatory skills and preflight documentation are reloaded before edits.
- [x] Focused source/publication contract tests are in place.
- [x] All current production cognition owners publish through one seam.
- [x] Latest API, console bootstrap/SSE, and Overview use one graph/card.
- [x] Source metadata and existing semantic graph detail are verified.
- [x] Skip, failure, cancellation, malformed, long/special-value cases pass.
- [x] Real-data Playwright screenshots and browser navigation/action checks pass.
- [x] Parent-only independent review is complete.
- [x] Final diff, worktree, and acceptance evidence are recorded.
- [x] User reviewed and approved the retained screenshots.

## Execution Evidence

Implementation was executed by the parent agent after this plan was promoted
to `in_progress`; no subagent was used.

- Source publication and failure coverage passed in the focused service,
  accepted-task, self-cognition, and console contract suites. The final
  self-cognition suite includes the admitted-failure and admitted-cancellation
  episode-propagation tests.
- The final console/client regression plus cognition E2E batch passed all 31
  tests with one existing Starlette/httpx deprecation warning. The final
  cognition, delivery, background, and self-cognition owner batch passed all
  143 tests. Python compilation and `git diff --check` passed.
- The parent review confirmed the only remaining `self_cognition_graph`
  occurrence is the safe internal projection reason/static hidden-card
  compatibility path; the latest brain/API/console transport has one
  canonical `cognition_graph` field and one Overview latest card.
- A clean development brain ran on `127.0.0.1:8877` and the matching
  development console on `127.0.0.1:8876`. The real `/ops/latest-cognition-graph`
  payload reached `status=completed`, `trigger_source=user_message`, and
  `input_sources=[dialog_text]` with run id
  `126b415a607248bd9c75dda73d244bfc`.
- The in-app Browser backend was unavailable (`agent.browsers.list()` returned
  no backends), so the permitted standalone Playwright fallback was used after
  installing the project Chromium binary. The real-data capture produced and
  visually inspected not-reported, running, completed, selected input,
  reasoning, memory, actions, visible-surface, and grey/deactivated visual
  directive states. No uncaught Playwright navigation/action error occurred;
  the unavailable in-app backend prevented a separate in-app console-listener
  check.
- Screenshot files are retained for user review under
  `.codex-runtime/cognition-plan-20260722/screenshots/`:
  `01-initial-not-reported.png`, `02-debug-real-running.png`,
  `02-debug-real-completed.png`, `03-overview-real-canonical.png`,
  `04-overview-real-input-detail.png`,
  `05-overview-real-reasoning-detail.png`,
  `06-overview-real-memory-detail.png`,
  `07-overview-real-actions-detail.png`,
  `08-overview-real-surface-detail.png`, and
  `09-overview-real-visual-detail.png`.
- The isolated development processes and temporary launcher/capture scripts
  were stopped/removed. The existing `8765` and `8000` processes were left
  untouched.

## Verification

Run from the repository root with `venv\\Scripts\\python` after approval:

1. Run focused service/publication/source tests and inspect the exact source,
   status, run id, callback count, and canonical latest field.
2. Run accepted-task and self-cognition deterministic regression slices and
   inspect that cognition/action/consolidation outcomes remain unchanged.
3. Run control-console contract and web-surface tests and inspect one latest
   graph payload, bounded source metadata, and honest empty fallback.
4. Run `tests\\control_console_e2e\\test_cognition_graph_e2e.py` through the
   project browser contract. Assert the Overview contains one latest graph
   card, the shared inspector remains functional for non-`/chat` graph data,
   and no self duplicate is visible. Assert source metadata at the API/graph
   snapshot boundary rather than adding source UI.
5. Restart clean development brain/console processes and perform the
   requested real-data Playwright screenshots under available states. Inspect
   the rendered widget, actual selected-node text, endpoint payload, browser
   console, and screenshot files.
6. Run the relevant broader deterministic regression batch, then
   `venv\\Scripts\\python -m py_compile` on modified Python modules and
   `git diff --check`.
7. Recheck `git status --short` and review the diff for `/chat`-only gates,
   dual latest fields, source heuristics, raw-data leakage, new LLM calls,
   and scope expansion.

## Independent Code Review

The parent agent performs a fresh review pass against the final diff and must
verify:

- every actual production shared-cognition caller publishes exactly once at
  its terminal graph handoff;
- `trigger_source` comes from the validated episode and is never inferred as
  `/chat`/`user_message`;
- accepted-task and self-cognition runs reach the same canonical latest field;
- the endpoint, bootstrap, SSE invalidation, and Overview contain one latest
  graph/card with no self duplicate;
- resolver cycles, concurrency, pre-cognition skips, failures, cancellation,
  malformed metadata, and unknown future source strings are truthful and
  safe;
- the existing semantic inspector, visual disabled state, redaction boundary,
  and Debug renderer remain intact;
- no prompt, model, RAG, memory, adapter, persistence, or unrelated UI
  feature changed.

## Acceptance Criteria

1. A normal `user_message` cognition execution publishes its actual graph to
   the canonical latest snapshot, with its bounded source metadata attached.
2. An `accepted_task_result_ready` execution that reaches shared cognition
   publishes its graph through the same latest field.
3. A self-cognition `internal_thought` execution publishes its graph through
   the same latest field; it does not require a separate Overview card or
   self-only API field.
4. The latest snapshot contains all nodes/edges represented by that actual
   source run, with existing completed/skipped/failed/partial statuses and no
   synthetic cross-run aggregation.
5. The graph metadata carries the actual bounded trigger source and input
   sources without exposing origin metadata or raw operational identifiers.
6. Resolver retries produce one terminal latest publication per execution.
7. Pre-cognition skips do not overwrite a prior real latest run; admitted
   failures/cancellations expose truthful bounded failure/partial state.
8. The control console has one Overview Latest cognition-run widget/card,
   uses the existing shared renderer unchanged, and retains the completed
   selected semantic-detail and visual disabled-state behavior.
9. Empty, malformed, long, multiline, Unicode, HTML-sensitive, unknown-source,
   unavailable-endpoint, and mixed-state fixtures render safely without
   uncaught browser errors or misleading `/chat` fallback.
10. No new LLM calls, prompts, model routes, trigger types, database writes,
    public adapter fields, source filters, history, or source-specific widgets
    are introduced.
11. Focused tests, relevant regressions, parent-only review, and the requested
    real-data Playwright screenshot sign-off pass with evidence recorded.

## User Sign-off

- Screenshot review approved by the user on 2026-07-23.
- Execution is closed without creating a commit; the implementation changes
  remain available for the user's separate commit decision.
