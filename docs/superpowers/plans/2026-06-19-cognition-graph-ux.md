# Cognition Graph UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the low-information cognition graph with a reusable Run Story + Auto-Focused Current Node gadget that shows live status, current node, latest signal, non-overlapping branches, and a stable detail inspector.

**Architecture:** Keep the existing buildless FastAPI-served HTML/CSS/JS console. Implement all behavior in the existing reusable graph renderer in `src/control_console/static/console.js`, with CSS in `src/control_console/static/console.css` and deterministic surface tests in `tests/test_control_console_web_surface.py`.

**Tech Stack:** Python/FastAPI static asset tests, buildless HTML/CSS/JavaScript, existing shadcn-style Card/Badge/Button/ScrollArea anatomy, Playwright/Chrome for rendered validation.

---

## File Structure

- Modify `src/control_console/static/console.js`
  - Keep `renderOverviewCognitionGraph` and `renderDebugCognitionGraph` as the two public call sites.
  - Replace the old node-only graph renderer with a graph model, run summary, lane-grid stage, stable inspector, current-node pinning, and return-to-current behavior.
- Modify `src/control_console/static/console.css`
  - Replace overlapping node/tooltip styles with a responsive lane layout and inspector layout.
  - Use existing CSS variables and badge/button/card-like styling; no frontend build stack.
- Modify `tests/test_control_console_web_surface.py`
  - Add static asset assertions for the new reusable graph surface and helpers.
  - Keep tests deterministic; no live LLM or live database tests.
- Do not modify brain-service cognition semantics, graph payload schemas, adapters, or `.env`.

---

### Task 1: Add Static Regression Tests For The New Graph Surface

**Files:**
- Modify: `tests/test_control_console_web_surface.py`

- [ ] **Step 1: Write the failing test assertions**

Add these assertions inside `test_static_shell_favicon_and_generic_lookup_outputs`, near the existing cognition graph assertions:

```python
    assert "function cognitionGraphModel" in script.text
    assert "function cognitionGraphCurrentNode" in script.text
    assert "function cognitionGraphInspectorMarkup" in script.text
    assert "function setCognitionGraphPinnedNode" in script.text
    assert "GRAPH_STALE_AFTER_MS = 10000" in script.text
    assert "Return to current" in script.text
    assert "data-graph-current-node-id" in script.text
    assert "data-graph-selected-node-id" in script.text
    assert "graph-inspector" in script.text
    assert "graph-run-summary" in script.text
    assert "graph-lane-row" in script.text
    assert "is-current" in script.text
```

Update the stylesheet assertions in the same test:

```python
    assert ".cognition-graph-shell" in stylesheet.text
    assert ".graph-run-summary" in stylesheet.text
    assert ".graph-body" in stylesheet.text
    assert ".graph-inspector" in stylesheet.text
    assert ".graph-lane-row" in stylesheet.text
    assert ".graph-node.is-current" in stylesheet.text
    assert ".graph-node.is-selected" in stylesheet.text
    assert ".graph-node .node-detail" in stylesheet.text
    assert "grid-template-columns: minmax(0, 1fr) minmax(min(300px, 100%), 0.38fr)" in stylesheet.text
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q
```

Expected: FAIL because the existing static assets do not contain the new helper names or graph classes.

---

### Task 2: Implement The Graph Model And Markup

**Files:**
- Modify: `src/control_console/static/console.js`

- [ ] **Step 1: Add state and constants**

Near the existing top-level `state` declaration and constants, add:

```javascript
const GRAPH_STALE_AFTER_MS = 10000;
state.cognitionGraphPins = state.cognitionGraphPins || {};
```

If direct assignment inside the `state` literal is cleaner, add `cognitionGraphPins: {},` to the literal instead.

- [ ] **Step 2: Replace `renderCognitionGraph` with a summary/body/inspector renderer**

Replace the body of `renderCognitionGraph` so it:

- updates the existing status badge with `success`, `warn`, or `danger` classes;
- keeps the explicit empty state when there are no nodes;
- computes lanes and max column;
- creates a `model` using `cognitionGraphModel`;
- renders a `.cognition-graph-shell` with:
  - `.graph-run-summary`
  - `.graph-body`
  - `.cognition-graph-stage`
  - `.graph-inspector`
- installs click handlers on graph nodes and the return-to-current button;
- schedules edge drawing after layout.

The implementation should follow this shape:

```javascript
function renderCognitionGraph({containerSelector, statusSelector, snapshot, emptyMessage}) {
  const container = qs(containerSelector);
  const status = qs(statusSelector);
  if (!container || !status) return;

  const graph = snapshot || {};
  const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
  const edges = Array.isArray(graph.edges) ? graph.edges : [];
  const graphStatus = graph.status || "not_reported";
  status.textContent = graphStatus.replaceAll("_", " ");
  status.className = cognitionGraphStatusBadgeClass(graphStatus);

  if (!nodes.length) {
    container.innerHTML = `<p class="graph-empty">${escapeHtml(emptyMessage)}</p>`;
    return;
  }

  const lanes = cognitionGraphLanes(nodes);
  const maxColumn = nodes.reduce((maximum, node) => Math.max(maximum, Number(node.column) || 1), 1);
  const model = cognitionGraphModel({graph, nodes, edges, lanes, maxColumn});

  container.innerHTML = `
    <div class="cognition-graph-shell" data-graph-source="${escapeHtml(model.source)}" data-graph-run-id="${escapeHtml(model.runId)}" data-graph-current-node-id="${escapeHtml(model.currentNode?.id || "")}" data-graph-selected-node-id="${escapeHtml(model.selectedNode?.id || "")}">
      ${cognitionGraphSummaryMarkup(model)}
      <div class="graph-body">
        ${cognitionGraphStageMarkup(model)}
        ${cognitionGraphInspectorMarkup(model)}
      </div>
    </div>
  `;

  container.querySelectorAll("[data-graph-node]").forEach((button) => {
    button.addEventListener("click", () => {
      setCognitionGraphPinnedNode(model.source, model.runId, button.dataset.nodeId || "");
      renderCognitionGraph({containerSelector, statusSelector, snapshot, emptyMessage});
    });
  });
  const returnButton = container.querySelector("[data-graph-return-current]");
  if (returnButton) {
    returnButton.addEventListener("click", () => {
      setCognitionGraphPinnedNode(model.source, model.runId, "");
      renderCognitionGraph({containerSelector, statusSelector, snapshot, emptyMessage});
    });
  }

  window.requestAnimationFrame(() => drawCognitionGraphEdges(container, edges));
}
```

- [ ] **Step 3: Add graph model helpers**

Add helpers after `cognitionGraphLanes`:

```javascript
function cognitionGraphModel({graph, nodes, edges, lanes, maxColumn}) {
  const source = graph.source || "overview_latest";
  const runId = graph.run_id || "run id not reported";
  const currentNode = cognitionGraphCurrentNode(nodes, lanes);
  const highlightedIds = new Set(nodes.filter((node) => node.status === "running").map((node) => node.id));
  if (currentNode) highlightedIds.add(currentNode.id);
  const pinnedNodeId = cognitionGraphPinnedNodeId(source, runId, nodes);
  const selectedNode = nodes.find((node) => node.id === pinnedNodeId) || currentNode || nodes[0];
  const generatedAt = Date.parse(graph.generated_at || "");
  const ageMs = Number.isFinite(generatedAt) ? Math.max(0, Date.now() - generatedAt) : null;
  const stale = graph.status === "running" && ageMs !== null && ageMs > GRAPH_STALE_AFTER_MS;
  return {
    graph,
    nodes,
    edges,
    lanes,
    maxColumn,
    source,
    runId,
    currentNode,
    selectedNode,
    highlightedIds,
    pinned: Boolean(pinnedNodeId),
    generatedAt,
    ageMs,
    stale,
    freshness: cognitionGraphFreshnessLabel(ageMs, stale),
    latestEvent: cognitionGraphLatestEvent(selectedNode || currentNode, edges, nodes),
  };
}
```

Add helpers for current-node selection:

```javascript
function cognitionGraphCurrentNode(nodes, lanes) {
  const running = nodes.filter((node) => node.status === "running");
  if (running.length) return cognitionGraphFurthestNode(running, lanes);
  const failed = nodes.filter((node) => node.status === "failed");
  if (failed.length) return cognitionGraphFurthestNode(failed, lanes);
  const completed = nodes.filter((node) => node.status === "completed");
  if (completed.length) return cognitionGraphFurthestNode(completed, lanes);
  const pending = nodes.filter((node) => node.status === "pending" || node.status === "skipped");
  if (pending.length) return cognitionGraphEarliestNode(pending, lanes);
  return nodes[0] || null;
}
```

Add `cognitionGraphFurthestNode`, `cognitionGraphEarliestNode`, `cognitionGraphLaneIndex`, `cognitionGraphPinnedNodeId`, `setCognitionGraphPinnedNode`, `cognitionGraphFreshnessLabel`, `cognitionGraphAgeLabel`, `cognitionGraphStatusBadgeClass`, and `cognitionGraphLatestEvent` as small pure helpers.

- [ ] **Step 4: Replace node markup with lane-cell markup**

Replace `cognitionGraphNodeMarkup` with a version that accepts `model` and uses classes:

```javascript
function cognitionGraphNodeMarkup(node, model) {
  const status = node.status || "not_reported";
  const selected = model.selectedNode && model.selectedNode.id === node.id;
  const highlighted = model.highlightedIds.has(node.id);
  const current = model.currentNode && model.currentNode.id === node.id;
  const detail = cognitionGraphDetail(node.detail || {});
  return `
    <button class="graph-node status-${escapeHtml(status)}${current ? " is-current" : ""}${selected ? " is-selected" : ""}${highlighted ? " is-highlighted" : ""}" type="button" data-graph-node data-node-id="${escapeHtml(node.id)}" aria-pressed="${selected ? "true" : "false"}">
      <span class="node-stage">${escapeHtml(node.stage || "stage")}</span>
      <strong>${escapeHtml(node.label || node.id)}</strong>
      <span class="node-meta"><span>${escapeHtml(node.lane || "cognition")}</span>${node.branch ? `<span>${escapeHtml(node.branch)}</span>` : ""}</span>
      <span class="node-detail">${detail}</span>
    </button>
  `;
}
```

Implement `cognitionGraphStageMarkup(model)` as lane rows with stacked cells so nodes cannot overlap:

```javascript
function cognitionGraphStageMarkup(model) {
  const rows = model.lanes.map((lane) => {
    const cells = [];
    for (let column = 1; column <= model.maxColumn; column += 1) {
      const cellNodes = model.nodes.filter((node) => (node.lane || "cognition") === lane && Math.max(1, Number(node.column) || 1) === column);
      cells.push(`<div class="graph-cell">${cellNodes.map((node) => cognitionGraphNodeMarkup(node, model)).join("")}</div>`);
    }
    return `<div class="graph-lane-row"><div class="graph-lane-label">${escapeHtml(lane)}</div>${cells.join("")}</div>`;
  }).join("");
  return `<div class="cognition-graph-stage" data-component="ScrollArea" style="--graph-columns: ${model.maxColumn};"><svg class="graph-edge-layer" aria-hidden="true"></svg>${rows}</div>`;
}
```

- [ ] **Step 5: Add summary and inspector markup**

Add `cognitionGraphSummaryMarkup(model)` and `cognitionGraphInspectorMarkup(model)`.

The inspector must show:

- selected/current label and status;
- summary or fallback text;
- important detail rows ordered by summary/conclusion, important signal, decision/stance/blocker, timing, branch/lane;
- `Return to current` button only when pinned and selected differs from current.

- [ ] **Step 6: Run focused static test**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q
```

Expected: PASS after the JS and CSS task are both complete. If this task is run before CSS, CSS assertions may still fail; continue to Task 3.

---

### Task 3: Replace The Graph CSS With Non-Overlapping Layout And Inspector Styles

**Files:**
- Modify: `src/control_console/static/console.css`

- [ ] **Step 1: Replace existing graph CSS block**

Replace the old graph block from `.cognition-card .card-content` through `.graph-legend` with styles for:

- `.cognition-graph-shell`
- `.graph-run-summary`
- `.graph-body`
- `.cognition-graph-stage`
- `.graph-lane-row`
- `.graph-lane-label`
- `.graph-cell`
- `.graph-node`
- `.graph-node.is-current`
- `.graph-node.is-selected`
- `.graph-inspector`
- `.graph-inspector-actions`
- `.graph-empty`

Required layout characteristics:

```css
.graph-body {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(min(300px, 100%), 0.38fr);
}
```

The lane rows must use a fixed label column plus repeated stage columns:

```css
.graph-lane-row {
  display: grid;
  grid-template-columns: minmax(78px, 96px) repeat(var(--graph-columns), minmax(150px, 1fr));
}
```

The node detail must be inline, not an absolute overlay:

```css
.graph-node .node-detail { position: static; display: none; }
.graph-node:hover .node-detail,
.graph-node:focus .node-detail,
.graph-node.is-selected .node-detail { display: grid; }
```

- [ ] **Step 2: Add responsive stacking**

Inside the existing `@media (max-width: 900px)` block, add:

```css
  .graph-body { grid-template-columns: 1fr; }
  .graph-inspector { position: static; }
```

- [ ] **Step 3: Run focused static test**

Run:

```powershell
venv\Scripts\python -m pytest tests\test_control_console_web_surface.py::test_static_shell_favicon_and_generic_lookup_outputs -q
```

Expected: PASS.

---

### Task 4: Rendered Browser Validation With A Real Fixture

**Files:**
- No committed source files.
- Temporary script under `%TEMP%` only.

- [ ] **Step 1: Start a temporary control-console server with fake graph data**

Use a temporary Python script outside the repo that:

- creates `ControlConsoleSettings` with temp `state_dir`;
- hashes token `secret`;
- patches `ControlConsoleRepository.application_identity` to return `not connected`;
- patches `KazusaClient.get_latest_cognition_graph` to return a `CognitionRunGraphSnapshot` containing:
  - one completed input node;
  - one completed L1 node;
  - two running L2 nodes in parallel lanes;
  - one pending L3 node;
  - fork and join edges.

- [ ] **Step 2: Validate with Chrome/Playwright**

Browser flow:

```text
http://127.0.0.1:<temp-port>/ -> login with secret -> Overview -> inspect graph
```

Checks:

- page title and body render;
- `#overview-cognition-graph .cognition-graph-shell` exists;
- `data-graph-current-node-id` is the running node with highest column and lane tie-break;
- all running nodes have `.is-highlighted`;
- the selected/current node has `.is-current` and `.is-selected`;
- `.graph-inspector` contains the running node summary;
- clicking another node changes `data-graph-selected-node-id`;
- `Return to current` appears after pinning another node;
- clicking `Return to current` restores the current node selection;
- there are no browser console errors or warnings;
- no graph node bounding boxes overlap in the fixture viewport.

- [ ] **Step 3: Stop the temporary server**

Stop only the process started for this fixture. Verify the temp port is closed.

---

### Task 5: Full Verification And Commit

**Files:**
- Modified production/test files only.

- [ ] **Step 1: Run deterministic control-console tests**

Run:

```powershell
$files = Get-ChildItem -Path 'tests' -Filter 'test_control_console*.py' | ForEach-Object { $_.FullName }; venv\Scripts\python -m pytest $files -q
```

Expected: all control-console tests pass.

- [ ] **Step 2: Compile control-console Python**

Run:

```powershell
venv\Scripts\python -m compileall -q src\control_console
```

Expected: exit code 0.

- [ ] **Step 3: Check whitespace**

Run:

```powershell
git diff --check
```

Expected: exit code 0.

- [ ] **Step 4: Review diff and commit**

Run:

```powershell
git diff --stat
git diff -- src/control_console/static/console.js src/control_console/static/console.css tests/test_control_console_web_surface.py
git status --short
```

Commit:

```powershell
git add -- src/control_console/static/console.js src/control_console/static/console.css tests/test_control_console_web_surface.py docs/superpowers/plans/2026-06-19-cognition-graph-ux.md
git commit -m "feat: improve cognition graph operator view"
```

Expected: commit succeeds on `feature/control-console-web`.

