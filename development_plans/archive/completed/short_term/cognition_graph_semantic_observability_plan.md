# cognition_graph_semantic_observability_plan

## Summary

- Goal: make the visual-directive output and the cognition graph useful for
  operator inspection by exposing the semantic payload at each selected node,
  preserving the actual visible text, and applying one inspector-widget change
  to every cognition-run surface.
- Plan class: large cross-layer observability and control-console contract
  change.
- Status: completed.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `control-console-web-development`, `py-style`,
  `test-style-and-execution`, `cjk-safety`.
- Dependencies: the existing L3 visual agent, response cognition graph,
  self-cognition graph publication, control-console graph snapshot, and the
  shared `renderCognitionGraph` inspector. No database migration or adapter
  contract change is required.
- Overall cutover strategy: replace the selected-detail vocabulary and shared
  renderer in one coordinated change. Keep graph node layout metadata at the
  node level for drawing, while the selected-detail panel uses a semantic
  field contract. Add the visual node and its detail payload in both graph
  builders in the same cutover.
- Highest-risk areas: full current-turn text disclosure in the operator
  console, graph payload size, preserving meaningful evidence boundaries,
  distinguishing visual-agent enablement from an empty output, preserving the
  existing grey/deactivated node state, and keeping the self-cognition run's
  existing visual-directive disablement intact.
- Acceptance criteria: the brain log contains the full normalized visual
  directive when the existing visual-agent gate runs; the same graph inspector
  renders the semantic details in Overview Latest, Debug cognition, and
  self-cognition Latest; the visual node exposes all four directive fields; the
  selected panel shows full approved text through scrolling; and focused,
  deterministic, and browser checks pass.

## Context

The current ownership path is already separated and should remain separated:

```text
L3 visual agent -> normalized visual directive state -> cognition graph builder
                 -> control-console graph projection -> shared graph widget
L3 dialog       -> visible dialog log and response surface
```

The visual agent in
`src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py` already normalizes
four output lists and records a protected LLM trace. Its normal application
logger block is commented out, so the brain process log has no human-readable
visual-directive line. The existing dialog path logs the actual normalized
dialog value through the project's complete list preview helper; the visual
directive log will follow that same operator-facing convention.

The response graph in `src/kazusa_ai_chatbot/service.py` currently presents
generic summaries for memory, actions, and the final surface. It omits the
actual `action_directives.visual_directives` payload. The self-cognition graph
has a separate builder and therefore needs the same semantic projection rules
where its artifacts contain equivalent data. The self-cognition runner
currently sets `no_visual_directives` explicitly; this plan represents that
state as an existing grey/deactivated visual node and preserves the current
disabled runtime behavior.

The control-console backend currently allowlists a small generic detail map and
applies an 800-character/50-item redaction cap. The shared frontend renderer
then prioritizes generic keys, appends `status`, `stage`, `lane`, and `branch`,
and truncates the inspector to seven rows. This removes the information the
operator needs. The selected-detail contract will carry explicitly approved
semantic fields, while the generic redaction policy remains in force for data
outside that contract.

The three requested surfaces are the Overview Latest cognition run, the Debug
cognition-run view, and the self-cognition Latest view. They consume the same
graph inspector path; the implementation must keep that shared ownership rather
than creating page-specific renderers.

## Mandatory Skills

- `development-plan`: maintain this document as the lifecycle and execution
  boundary; keep it `draft` until the user approves implementation.
- `local-llm-architecture`: preserve the current bounded L3 pipeline and make
  the observability projection deterministic. This change adds no model call,
  prompt, routing branch, or semantic post-processor.
- `control-console-web-development`: use the control-console static frontend,
  shared snapshot/SSE contract, and browser validation rules. Validate the
  rendered widget in all three entry points.
- `py-style`: load the project Python style policy and its referenced rules
  before editing Python production or test files.
- `test-style-and-execution`: use deterministic tests for projection,
  formatting, and graph handoff; use the patched LLM path for the visual log
  test; run any live LLM test one case at a time only if a later implementation
  change unexpectedly affects prompt/model behavior.
- `cjk-safety`: apply the repository's CJK-string handling rules while editing
  the Python L3 module, which contains multilingual prompt content.

## Mandatory Rules

- Treat this document as a discussion artifact while `Status: draft`. The
  implementation gate requires both user approval and status `approved` or
  `in_progress`.
- Parent agent owns the implementation sequence, focused test creation, test
  output inspection, plan evidence, and final acceptance. Use the required
  native subagent sequence after approval: one production implementation
  subagent, followed by one independent review subagent.
- Inspect and preserve the user's worktree changes. Use `git status --short`
  before implementation, use `apply_patch` for manual edits, and use the
  project virtual environment at `venv\\Scripts\\python`.
- Keep the existing visual-agent enablement decision authoritative:
  `COGNITION_VISUAL_DIRECTIVES_ENABLED` plus the per-run
  `no_visual_directives` flag determine whether the visual agent runs. This
  plan does not turn visual directives on for self-cognition.
- Log the normalized four-field visual directive after successful enabled
  execution. A disabled/short-circuited visual stage produces no visual-agent
  output log line; its graph node remains present with the existing skipped
  state so the box stays grey/deactivated.
- Reuse the existing graph status implementation for the visual node:
  `status-skipped`, `is-terminal`, the terminated badge, and the existing grey
  CSS. Add no replacement inactive/deactivated component or styling system.
- Keep raw prompts, raw model responses, embeddings, message envelopes,
  internal trace identifiers, action target identifiers, and handler metadata
  outside the selected semantic detail contract.
- Preserve full text only for explicit semantic fields defined in this plan.
  Do not weaken the generic console redaction policy globally.
- Keep LLM stages responsible for semantic judgment and deterministic code
  responsible for projection, allowlisting, redaction boundaries, UI limits,
  and serialization.
- Keep graph rendering bounded by layout and panel scrolling, while giving the
  operator access to every approved semantic value without synthetic character
  or list-item truncation.

## Must Do

### 1. Brain process/live log

Update `call_visual_agent` so that the successful enabled path emits one
operator-facing `INFO` record containing the actual normalized mapping:

```text
Visual directive output: {"facial_expression": [...],
"body_language": [...], "gaze_direction": [...], "visual_vibe": [...]}
```

Use the same complete JSON/list rendering convention used by the visible
dialog log. Place the record after normalization and contract validation so the
log reflects the exact state handed to the downstream collector. Keep the
protected LLM trace unchanged; it remains the diagnostic record for model
metadata and raw-output capture mode.

Add a deterministic patched-LLM test proving that the complete four-field
mapping, including long directive text, reaches the brain log. Add the paired
disabled-path assertion that the short-circuit returns the existing empty
fields and emits no visual output log line.

### 2. Cognition graph L3 visual node

Extend both graph builders in `src/kazusa_ai_chatbot/service.py` with the same
node identity and semantic detail vocabulary:

- Node id: `l3.visual_directives`.
- Label: `Visual directive`.
- Stage: `L3`.
- Lane: `surface`.
- Column: the existing fourth/L3 column.
- Detail fields: the four actual normalized lists
  `facial_expression`, `body_language`, `gaze_direction`, and `visual_vibe`.

Both graph builders always include this node in the existing fourth/L3 column
whenever they emit a cognition graph. The builder uses the existing run-level
visual-agent gate and stage reachability to set the node status:

- enabled and completed: the existing completed/current status is used and the
  node detail contains all actual normalized directive lists;
- enabled and currently executing: the existing running status is used;
- enabled but failed: the existing failed status is used and the node receives
  the existing safe failure detail path;
- disabled by the existing global/per-run gate: `status: "skipped"` is used,
  the node remains grey/deactivated through the existing renderer, and no
  directive output is fabricated;
- not reached because an upstream stage stopped: the existing pending/skipped
  state is used according to the graph builder's current stage state.

An enabled stage with an empty result still has a completed visual node and an
explicit empty-state detail message; an empty result is never treated as
disablement. The implementation must use the run's existing enablement marker,
not the presence or absence of non-empty output, to make this decision.

The self-cognition graph uses the same node contract and keeps the node grey and
deactivated when its current explicit `no_visual_directives` setup is active.
When a self-cognition artifact represents an enabled visual stage, the node
uses the same status mapping and actual four-field detail as the response graph.
The current self-cognition disablement remains unchanged, so current self runs
retain their existing runtime cost and show no fabricated visual output.

Connect the visual node to the same L2 prerequisites used by the final surface
node. Keep it a sibling inspection node rather than merging visual directives
into the visible-message node. Preserve the existing top-level node metadata
needed to draw the graph; the selected inspector renders semantic detail only.
Use the existing `status-skipped`/terminated rendering path for disabled visual
nodes, including its grey border/background, dashed treatment, status badge,
and connector/group behavior.

### 3. Selected node semantic detail contract

Replace the current summary-first selected-detail content with this fixed
semantic order. The selected-detail contract contains no generic `summary`
field. Each field is populated from already-produced state or artifacts by
deterministic projection.

| Node | Selected-detail fields | Information shown |
|---|---|---|
| `intake` / L1 queued turn | `input`, `reply_context` | The actual queued input, plus reply-context information only when it contains useful context. |
| `l1.relevance` / L1 response decision | `decision`, `reasoning` | The existing response decision and the existing decision rationale. |
| `l2.reasoning` / L2 reasoning | `internal_monologue`, `logical_stance`, `character_intent`, `judgment_note` | The existing reasoning artifacts, with their actual content preserved. |
| `l2.memory` / L2 memory and evidence | `retrieval_answer`, `memory_evidence`, `conversation_evidence`, `external_evidence`, `recall_evidence`, `media_evidence`, `user_continuity`, `conversation_progress`, `active_commitments` | Retrieval conclusions and the important evidence/facts that informed cognition. Each evidence item keeps its useful semantic text, source/title where safe, relevance/recency, due state, and evidence-boundary notes. Retrieval process traces, candidate lists, raw IDs, and embeddings remain excluded. |
| `l2.actions` / L2 actions | `selected_actions`, `action_results`, `action_continuation` | Selected action kind/reason, urgency, visibility, deadline, continuation, and semantic result state/visibility/outcome. Handler ownership, internal attempt IDs, raw parameters, and target identifiers remain excluded. |
| `l3.visual_directives` / L3 visual directive | `facial_expression`, `body_language`, `gaze_direction`, `visual_vibe` | The actual visual directive lists returned by the visual agent. A disabled node uses the existing grey/deactivated status and a clear empty-state message instead of invented directive content. |
| `l3.surface` / L3 visible surface | `messages` | The actual final visible message fragments in their original order, preserving line breaks and all text. |

The graph card and tooltip need a short preview, but the graph builder will not
store a useless `summary` detail field for that purpose. The existing frontend
node-summary helper will derive its preview from the first useful semantic
value in the node's detail, and the latest-event helper will use the same
semantic fallback order. The selected inspector will show the actual semantic
fields above, not a generated summary row or a generic “no bounded detail” row.
The implementation will use existing evidence/action projection helpers where
they already provide the correct semantic boundary and will add narrowly
scoped helpers for the remaining fields. It will not ask an LLM to summarize
the data a second time.

For the self-cognition graph, apply the same vocabulary to equivalent
artifacts. Use the existing self run record for queued input, cognition output
for reasoning and selected actions, its RAG/evidence payload for memory, its
visual-directive payload when present, and its recorded visible-dialog surface
for `messages`. Keep self-cognition-specific route/action/consolidation nodes
and give them the same semantic detail quality where their artifacts support
it.

### 4. Shared widget and full-text behavior

Update the shared `renderCognitionGraph` path in
`src/control_console/static/console.js` so all three surfaces receive the
same inspector rows, labels, value rendering, and scroll behavior. Remove
`status`, `stage`, `lane`, and `branch` from selected-detail rows; retain the
same metadata in graph node cards and top-level snapshot data for navigation
and layout.

Remove the seven-row inspector cap and remove `summary` from the selected-row
field order. Render scalar strings with preserved line breaks, arrays as
individually readable entries in source order, and mappings as readable nested
key/value content. Add a scrollable selected-detail region with a stable
maximum height that fits the existing widget. The panel must support long
input, evidence excerpts, action explanations, visual directives, and final
messages without backend character/list truncation or frontend ellipsis. Keep
the diagram card preview compact by deriving it from actual semantic values;
full text belongs in the selected panel.

Update the control-console backend projection so approved semantic fields pass
through in full. Implement this as a graph-specific allowlist/projection rather
than changing `redact_mapping` or `MAX_SAFE_TEXT_CHARS` for unrelated console
payloads. The allowlist must accept the semantic field names in the table,
recursively preserve their approved text/list structure, and continue to
exclude prompt/raw-output/message-envelope/embedding and operational metadata
keys. Add documentation describing this intentional operator-console
disclosure boundary.

### 5. Documentation and contract evidence

Update the control-console README and the relevant cognition/debug HOWTO
section to describe:

- the shared inspector ownership across the three surfaces;
- the L1/L2/L3 semantic field order;
- the separate visual-directive node and its existing grey/deactivated skipped
  state;
- full approved text access through inspector scrolling; and
- the distinction between the human-readable brain log and protected LLM
  trace capture.

Keep the plan registry entry synchronized with this document's status.

## Deferred

- Changing the global visual-directive configuration or self-cognition's
  existing `no_visual_directives` default.
- Adding visual directives to the public adapter response or visible dialog
  wording.
- Changing visual-agent prompts, model selection, routing, retry behavior, or
  protected trace capture modes.
- Persisting historical full semantic graph payloads in a new database schema.
- Removing graph-level layout metadata used by the renderer.
- Building a second inspector component for any page or adding page-specific
  formatting overrides.

## Cutover Policy

- Brain logging: compatible additive change. Existing dialog and trace logging
  continue with one additional normalized visual-directive record on the
  enabled success path.
- Graph snapshot node layout: compatible additive change. Existing top-level
  node metadata and existing node ids remain available; the
  `l3.visual_directives` node is present with the existing skipped state when
  disabled and carries actual directive data when enabled.
- Selected-detail payload: big-bang internal control-console contract change.
  The old generic detail vocabulary is replaced by the semantic allowlist in
  the same patch as the shared renderer, with no parallel mapper or legacy
  widget.
- Generic console redaction: compatible unchanged policy. Full text is an
  explicit exception for approved cognition semantic fields only.
- Rollback: revert the single coordinated implementation commit/patch. The
  plan does not introduce a dual-schema runtime fallback.

## Target State

```text
existing visual enablement decision
        |
        +--> L3 visual agent --> full normalized directive --> brain INFO log
        |                                      |
        |                                      +--> l3.visual_directives node
        |
        +--> L3 surface --> actual final message fragments --> l3.surface node

L1/L2 produced state and evidence/action artifacts
        |
        +--> deterministic semantic graph projection
                        |
                        +--> one shared selected-detail widget
                              Overview Latest
                              Debug cognition run
                              Self-cognition Latest
```

The graph remains a bounded visual layout. Its selected inspector becomes the
operator's complete semantic readout for the current approved fields. The
visual directive and visible surface stay separate siblings, and each selected
node displays the actual useful payload rather than a generic status summary.

## Design Decisions

1. **One renderer owns all three surfaces.** The existing shared
   `renderCognitionGraph`/inspector path is the only frontend ownership point.
   Backend graph builders feed the same node-detail contract.
2. **Visual output is logged after validation.** This mirrors the visible
   dialog logging rule and avoids logging a pre-normalized model response.
3. **Enablement comes from the existing run decision.** An empty result is a
   valid enabled result and still gets a completed visual node; a disabled
   short-circuit keeps a grey/deactivated `status-skipped` node and produces no
   output log line.
4. **Diagram metadata and selected semantics have different jobs.** Status,
   stage, lane, and branch remain available to draw and navigate the graph;
   semantic rows replace them in the selected panel.
5. **Evidence is shown as evidence.** The memory panel exposes retrieval
   answer, useful evidence text, continuity, progress, and commitments. It
   does not turn retrieval process traces into facts or persona judgment.
6. **Actions are shown as decisions and outcomes.** The action panel exposes
   what was selected, why, urgency/visibility/timing, continuation, and
   semantic result. Internal routing and handler identifiers stay outside the
   operator readout.
7. **Full text is a deliberate local-console contract.** The approved user
   input, evidence excerpts, action explanations, visual lists, and visible
   messages remain complete. A scrollable inspector handles fit; the generic
   redaction cap remains active elsewhere.
8. **No additional semantic model step.** Existing LLM output is projected
   directly, avoiding latency, cost, and interpretation drift.

## Risks

| Risk | Mitigation and verification |
|---|---|
| Full user input, evidence, directive, or visible-message text increases operator-console disclosure and payload size. | Use an exact graph-semantic allowlist, retain generic redaction for every other console payload, exclude protected/operational keys, make the inspector scrollable, and test long values plus forbidden nested keys. |
| The graph cannot reliably distinguish an enabled stage with an empty result from a disabled short-circuit. | Carry the existing run-level enablement decision into the graph snapshot deterministically; test enabled-empty and disabled cases separately; assert disabled nodes use the existing `status-skipped` grey/deactivated class. |
| Self-cognition artifacts use different names or omit response-run fields. | Add a focused self-builder fixture, merge the existing cognition input/output artifacts for evidence, preserve the existing disabled default when the runtime gate is absent, and show only fields present in the artifact rather than inventing values. |
| A shared renderer change regresses one of the three entry points or the SSE update path. | Exercise Overview Latest, Debug, and self-cognition Latest with the same fake graph contract and browser assertions; inspect browser console output and screenshots. |
| Large semantic detail makes the graph card itself unreadable or causes frontend layout overflow. | Keep card summaries compact, move complete values into a fixed-height scrollable inspector, preserve line breaks, and verify the widget at its supported viewport sizes. |
| Observability work accidentally changes cognition behavior or adds an LLM call. | Keep the visual prompt, model binding, route, state judgment, and dialog path unchanged; inspect the diff and assert zero additional model calls in focused tests. |

## Failure-Mode Handling

The implementation must handle these cases deterministically and test each
relevant path without turning malformed data into a misleading semantic value:

| Failure mode | Required behavior | Verification |
|---|---|---|
| Very long input, evidence, directive, or visible message | Preserve the complete approved string in the graph payload and log. The inspector uses its existing widget width, wrapping, and vertical scroll; it does not insert ellipses or backend character caps. Diagram cards use a derived short preview only. | Fixture with multi-paragraph text beyond the old 800-character cap; assert the beginning, middle, and end are available in the selected panel and the browser has a scrollable detail region. |
| Very large list or many evidence/action items | Preserve source order and duplicate items. Render each item as a readable entry inside the scrollable panel. Keep graph node count/layout limits and allow only the defined semantic containers; apply no silent first-50 item cut to approved fields. | Fixture with more than 50 items and repeated values; assert all items are present in the backend projection and accessible in the browser. |
| Missing, `null`, empty string, or empty list | Omit optional empty rows instead of showing useless placeholders. For an enabled visual stage with no items, show one explicit empty-state message. For a disabled visual stage, keep the node with existing `status-skipped` grey/deactivated styling and show the disabled empty-state message. | Deterministic response/self graph cases for missing, null, empty-enabled, and disabled visual data; assert no generic summary/status/stage/lane/branch detail rows appear. |
| Erratic scalar/list/mapping type in an artifact | Validate expected shapes at the projection boundary. Preserve valid values, omit invalid fields, and expose a safe field-level projection warning or existing failure detail without stringifying data as `[object Object]` or inventing meaning. A visual-agent contract failure marks the existing node failure path and never logs raw model output as a directive. | Cases with numbers, booleans, nested mappings where lists are expected, malformed evidence items, and missing keys; assert graph construction remains safe and output is truthful. |
| Visual model returns malformed JSON or invalid directive fields | Keep the existing parse/contract-validation failure behavior and protected trace behavior. Emit the normal failure log/path, set the visual node to the existing failed state when the graph records the failure, and keep raw response content outside the selected detail. | Patched LLM responses for invalid JSON, wrong field types, and unexpected keys; inspect log, graph status, and absence of raw response leakage. |
| Disabled visual agent | Do not call the visual model or emit a visual output log line. Keep `l3.visual_directives` in the graph with `status: "skipped"`, allowing the existing `.graph-node.status-skipped` grey/dashed rendering, terminated badge, and group/connector behavior to communicate deactivation. | Browser and deterministic graph assertions for the disabled response and self-cognition paths; verify the node remains visible and grey/deactivated. |
| Special characters and multiline content | Serialize log values with the existing complete JSON/list helper. Preserve Unicode, CJK, emoji, quotes, backslashes, tabs, and newlines. Escape HTML-sensitive characters through the existing frontend escaping path and use CSS whitespace preservation so text is displayed as text, never interpreted as markup. | Fixtures containing `<script>`, `&`, quotes, backticks, CJK, emoji, CR/LF, tabs, and Unicode separators; assert safe DOM text and no script/markup execution. |
| Unexpected nested keys or operational data inside evidence/actions | Apply the graph-specific recursive allowlist at every nesting level. Keep prompt/raw-output/message-envelope/embedding/internal-id/handler/target fields out even when nested beside valid text. | Projection tests with forbidden keys at root and nested levels; assert valid semantic siblings survive and forbidden values are absent. |
| Missing or malformed graph snapshot/SSE update | Use the existing frontend fallback for absent node arrays/detail maps and unknown statuses. Render a safe empty/error state, preserve the pinned-node behavior when the id exists, and avoid throwing during re-render. | Fake-brain snapshots with missing detail, wrong detail type, unknown status, partial node arrays, and running-to-completed updates; inspect browser console for zero render exceptions. |
| Upstream failure or incomplete L3 reachability | Use the existing graph status vocabulary: failed for a recorded visual-stage failure, pending/skipped for work not reached, and skipped/grey for deliberate disablement. Keep the visual node separate so a visible-surface result cannot be mistaken for a visual-directive result. | Graph-builder fixtures for upstream failure, pending, enabled success, and disabled success-path bypass; assert status classes and detail contents. |
| Log serialization or UI rendering failure caused by an unexpected value | Keep logging based on the normalized validated mapping. Keep UI formatting total for JSON-compatible scalar/list/mapping values and fail closed to a readable field omission/error marker for unsupported values. The response path continues under the existing cognition failure policy. | Unit tests call formatter/projection helpers with every supported and malformed fixture type and assert no uncaught exception. |

## Contracts/Data Shapes

The existing `CognitionRunGraphNode.detail` mapping remains the transport
container. Its selected-detail keys become:

```text
queued turn:       input, reply_context
response decision: decision, reasoning
reasoning:         internal_monologue, logical_stance, character_intent,
                   judgment_note
memory/evidence:   retrieval_answer, memory_evidence,
                   conversation_evidence, external_evidence, recall_evidence,
                   media_evidence, user_continuity, conversation_progress,
                   active_commitments
actions:           selected_actions, action_results, action_continuation
visual directive:  facial_expression, body_language, gaze_direction,
                   visual_vibe
visible surface:   messages
```

The visual node reads the normalized fields returned by
`call_visual_agent`/`action_directives.visual_directives`. The visible-surface
node reads the final visible dialog fragments already produced by the dialog
stage. Memory and action fields use safe semantic projections of the existing
RAG, action-spec, and action-result contracts. The backend's graph-specific
projection validates these shapes before returning them to the console.

The selected-detail payload has no LLM budget change: zero additional LLM
calls, zero additional prompt tokens, and zero additional model routes. The
only new runtime work is deterministic projection, serialization, UI rendering,
and one INFO log serialization on successful visual-agent execution.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l3.py`
  - emit the complete normalized visual directive on the enabled success path;
  - retain existing validation, trace capture, and disabled return behavior.
- `src/kazusa_ai_chatbot/service.py`
  - project meaningful L1/L2/L3 detail fields without generic summaries;
  - add the response and self-cognition visual node with existing skipped/
    failed/completed status handling;
  - merge self-cognition cognition input/output evidence for memory detail
    and preserve the existing self-cognition visual-disable default when the
    artifact omits the runtime gate field;
  - propagate optional self-cognition visual reachability/failure markers so
    pending and failed visual nodes use the same status vocabulary;
  - carry the existing enablement decision into graph construction without
    changing the visual-agent gate.
- `src/control_console/app.py`, `src/control_console/contracts.py`,
  `src/kazusa_ai_chatbot/brain_service/contracts.py`, and
  `src/control_console/static/index.html`
  - transport the separate self-cognition Latest snapshot and expose its
    dedicated Overview panel through the same graph widget.
- `src/control_console/kazusa_client.py`
  - replace generic selected-detail allowlisting with the explicit semantic
    graph projection;
  - preserve full approved values and existing exclusions.
- `src/control_console/static/console.js`
  - update the shared inspector field order and value renderer;
  - remove summary and metadata rows plus the seven-row cap;
  - preserve full text and line breaks in the selected panel.
- `src/control_console/static/console.css`
  - make the selected-detail content scrollable and readable within the
    existing widget dimensions.
- `src/control_console/README.md` and the relevant `docs/HOWTO.md` section
  - document the graph semantic contract, visual node, full-text inspector,
    and log/trace distinction.
- `development_plans/README.md`
  - register this new draft plan.

### Tests

- `tests/test_l2d_l3_surface_handoff.py`
  - add enabled visual-agent logging and disabled-path assertions with a
    patched deterministic LLM response, including wrong-type output failure.
- `tests/test_service_cognition_graph.py` (new focused deterministic module)
  - cover response and self-cognition graph node/detail construction,
    existing skipped/deactivated visual-node behavior, actual visible
    messages, evidence, semantic action fields, empty fields, and malformed
    artifact shapes, including self visual pending/failed states.
- `tests/test_control_console_cognition_graph.py`
  - cover the graph-specific allowlist, full approved text, recursive semantic
    values, and exclusion of prompts, raw outputs, embeddings, and operational
    identifiers.
- `tests/control_console_e2e/fake_brain.py`
  - provide long semantic values and an enabled visual directive in the fake
    graph snapshot.
- `tests/control_console_e2e/test_cognition_graph_e2e.py`
  - verify the shared inspector in Overview Latest and Debug, the self graph
    fixture, the visual node/detail, existing grey/deactivated skipped state,
    metadata-row/summary replacement, special-character rendering, and
    scrollable complete text.
- `tests/control_console_e2e/test_debug_chat_e2e.py`
  - select the Debug visual-directive node and assert its actual semantic
    detail through the shared inspector.

## Overdesign Guardrail

The implementation stays inside the current L3 state, service graph snapshot,
control-console projection, and shared static widget. It adds one deterministic
semantic projection boundary and one visual graph node. It does not add a new
LLM summarizer, telemetry service, database collection, frontend framework,
adapter field, compatibility mapper, historical payload store, or page-specific
widget. Evidence remains bounded by field allowlists and graph node count; only
approved semantic values receive full-text treatment.

## Agent Autonomy Boundaries

The implementation agent may choose helper names, local function placement,
CSS class names, and test fixture values while preserving the contracts above.
The agent must keep the node id, detail field names, enablement semantics,
logging timing, shared-widget ownership, exclusions, and verification gates
exactly as specified. Any request to alter prompt/model behavior, enable visual
directives for self-cognition, expose protected trace data, change public
adapter surfaces, or add persistence requires a new user decision and plan
scope.

## Implementation Order

1. Confirm approval and promote this plan from `draft` to `approved` or
   `in_progress`; record the approval in the plan status before production
   edits.
2. Recheck `git status --short`; load the mandatory Python, test, console, and
   CJK skills; then create the focused deterministic test expectations and
   fixtures.
3. Implement the L3 visual log and the shared semantic graph-detail helpers
   in the response and self-cognition graph builders.
4. Implement the control-console graph-specific full-text projection and the
   shared inspector renderer/CSS.
5. Update the control-console and HOWTO documentation.
6. Run focused deterministic tests, then the control-console browser/E2E
   checks, inspect output artifacts and browser console logs, and run the
   broader relevant regression batch.
7. Run one independent code-review subagent against the final diff and all
   acceptance criteria. Parent agent resolves review findings, reruns affected
   checks, and records evidence before completion.

## Execution Model

- Parent-led, sequential execution after approval.
- The parent creates and reviews focused tests and owns integration.
- Production implementation uses one native production subagent after the
  focused test contract is established.
- Independent review uses a second native subagent with fresh context and
  read-only review scope.
- Real LLM execution is unnecessary for this observability-only change. The
  visual log test uses the existing patched LLM seam and asserts actual output
  content.
- Browser validation uses the in-app Browser skill when available, against a
  clean control-console process/session, and covers all three graph entry
  points.

## Progress Checklist

- [x] User approves this plan and implementation scope.
- [x] Plan status is promoted to `approved` or `in_progress`.
- [x] Mandatory skills are loaded for implementation and testing.
- [x] Focused tests and long-text fixtures are in place.
- [x] Enabled visual-directive brain logging is implemented and verified.
- [x] Response and self-cognition graph semantic projections are implemented.
- [x] `l3.visual_directives` uses the existing completed/running/failed/
  pending/skipped status rendering and actual four-field detail.
- [x] Shared inspector renders semantic fields with full approved text and no
  generic summary row.
- [x] Status/stage/lane/branch rows are replaced in selected detail while
  graph layout metadata and grey/deactivated skipped rendering remain
  functional.
- [x] Documentation and registry are updated.
- [x] Focused tests, regression tests, browser checks, and review are complete.
- [x] Final diff, worktree, and acceptance evidence are recorded.

## Verification

Run with `venv\\Scripts\\python` from the repository root:

1. `-m pytest tests\\test_l2d_l3_surface_handoff.py -q` and inspect the
   enabled/disabled visual log assertions.
2. `-m pytest tests\\test_service_cognition_graph.py tests\\test_control_console_cognition_graph.py -q`
   and inspect semantic fields, skipped/deactivated node shape, full text,
   malformed-value handling, and exclusion assertions.
3. `-m pytest tests\\control_console_e2e\\test_cognition_graph_e2e.py -q`
   with the control-console browser contract; inspect Overview Latest, Debug,
   and self-cognition Latest screenshots/DOM assertions and browser console
   output.
4. Run the relevant control-console and cognition deterministic regression
   batch defined by `test-style-and-execution`; inspect every failure artifact.
5. Run `python -m py_compile` through the project venv for the modified Python
   modules, then `git diff --check`.
6. Recheck `git status --short` and review the final diff for scope, prompt
   leakage, operational identifier leakage, synthetic truncation, and
   page-specific widget paths.

Final evidence recorded during execution:

- Real-data Playwright sign-off used the isolated brain/console processes.
  Enabled visual completion is captured in
  `test_artifacts/real-cognition-signoff/cognition-enabled-completed.png`,
  the selected visible surface in
  `test_artifacts/real-cognition-signoff/cognition-visible-surface-completed.png`,
  and Overview Latest in
  `test_artifacts/real-cognition-signoff/cognition-overview-latest-real.png`.
- The real disabled run is captured in
  `test_artifacts/real-cognition-signoff/cognition-disabled-skipped.png` and
  `test_artifacts/real-cognition-signoff/cognition-overview-disabled-real.png`.
  The visual node was `status-skipped` with the grey/deactivated class and the
  browser reported zero console messages in both state captures.
  A later post-fix disabled retry stalled in the external live-model path for
  180 seconds without an application or UI error; the completed capture above
  remains the accepted disabled-state evidence.
- The real self-cognition Latest state was `not reported`; the populated
  self-cognition inspector and shared-widget path are covered by the direct
  browser E2E fixture, while the real Overview capture records that honest
  runtime state rather than inventing a self run.
- The in-app Browser plugin reported no available browser, so the installed
  Playwright/system-Chrome fallback performed the real sign-off captures.

## Independent Code Review

A fresh-context review agent must inspect the final diff and verify:

- the visual log occurs only after enabled output normalization/validation and
  contains the actual four-field value;
- the response and self graph builders use the same node/detail vocabulary and
  preserve the self-cognition disabled default;
- the selected-detail projection exposes useful evidence/actions/messages while
  excluding prompt, raw-output, embedding, message-envelope, and operational
  identifier data;
- the frontend has one shared inspector path, no seven-row cap, no metadata rows
  in selected detail, and a working scroll/full-text layout;
- tests cover enabled, disabled, long-text, visual-node, and three-surface
  browser behavior;
- no prompt, model, adapter, persistence, or unrelated redaction behavior
  changed.

## Acceptance Criteria

1. With visual directives enabled for a response run, the brain process/live
   log prints the actual complete normalized `facial_expression`,
   `body_language`, `gaze_direction`, and `visual_vibe` values using the same
   operator-facing convention as visible dialog logging.
2. With visual directives disabled, the existing empty return remains intact,
   the brain log contains no visual output line, and the graph keeps a visible
   `l3.visual_directives` node with the existing grey/deactivated
   `status-skipped` rendering.
3. With visual directives enabled, the response graph shows a separate
   `l3.visual_directives` node in the existing fourth/L3 column, and selecting
   it shows all four actual directive lists.
4. The same inspector behavior is effective in Overview Latest, Debug
   cognition, and self-cognition Latest through the shared widget path.
5. Selected L1 queued-turn detail shows the actual `input` and useful
   `reply_context` when present; it contains no generic summary or duplicate
   decontextualized-input field. L1 response decision and L2 reasoning retain
   their actual existing semantic fields.
6. Selected L2 memory/evidence detail shows retrieval conclusions and useful
   evidence/facts, while selected L2 actions shows selected decisions,
   continuations, and semantic outcomes rather than count-only summaries.
7. Selected L3 visible-surface detail shows the actual final message fragments
   in order. The visual-directive detail shows the actual directive fields.
8. Selected detail contains no `summary`, `status`, `stage`, `lane`, or
   `branch` rows, and it has no seven-row limit. Graph drawing still uses its
   node metadata and existing status rendering.
9. Approved semantic strings, arrays, and nested values remain complete and
   readable through the scrollable inspector. Empty, malformed, oversized,
   special-character, partial-SSE, and failed-stage cases follow the explicit
   failure-mode rules. The generic console redaction cap remains unchanged for
   other payloads.
10. Focused tests, relevant deterministic regressions, browser/E2E checks,
    independent review, `py_compile`, and `git diff --check` pass with no
    unrelated worktree changes.
