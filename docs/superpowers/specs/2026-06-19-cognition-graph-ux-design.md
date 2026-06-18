# Cognition Graph UX Design

Date: 2026-06-19

## Purpose

The cognition graph must become an operator-facing diagnostic widget, not a
decorative diagram. Its job is to answer useful questions quickly:

- Is a cognition run active, completed, failed, partial, or not reported?
- Which node or stage is current right now?
- What happened most recently?
- Where did the run branch in parallel?
- What did each stage conclude, using only bounded redacted detail?
- Is the displayed graph fresh or stale?

The approved direction is **Run Story + Auto-Focused Current Node**.

## Current Problems

The current graph is not useful enough because:

- The grid layout shows nodes but does not explain the run story.
- The user cannot reliably tell whether cognition is running.
- Hover detail is fragile and can be clipped or overlap nearby nodes.
- Parallel branches are visually unclear and can collide.
- The widget gives little indication of freshness or latest event.
- The user must manually inspect nodes to discover important information.

## Approved Design

The graph gadget renders three coordinated surfaces:

1. **Run Summary Header**
   Shows run id, source, generated/update age, status badge, current stage,
   and freshness state. This is the first thing an operator reads.

2. **Run Story Graph**
   Shows the ordered cognition path and parallel branches with clear lane
   labels. Nodes must not overlap. Running, failed, completed, skipped, and
   pending states use distinct but restrained shadcn-style badge/card styling.

3. **Stable Inspector**
   Shows selected node detail in a fixed panel beside or below the graph.
   Important text must not depend on hover-only overlays. Hover may show a
   compact tooltip, but the inspector is the primary detail surface.

## Current Node Behavior

The gadget automatically selects the most useful node:

- Prefer all `running` nodes for visual highlighting.
- If multiple nodes are running, highlight all and focus the furthest-progressed
  running node in the inspector. Furthest-progressed means highest `column`;
  ties use the rendered lane order.
- If no node is running, focus the first `failed` node.
- If no node failed, focus the terminal completed node.
- If only pending/skipped nodes exist, focus the lowest-column pending node.
- If telemetry is absent, render an explicit empty state and do not fabricate
  nodes.

When the operator clicks a node, the inspector is pinned to that node. If a
new current node arrives while another node is pinned, show a small
`Return to current` action. Refreshing or switching pages must not leave the
inspector showing stale detail for a different run.

## Detail Content

Each node detail should prioritize useful redacted information in this order:

1. Summary or conclusion.
2. Important signal, decision, stance, or blocker.
3. Latest timing/status information when available.
4. Next expected step when inferable from graph structure.
5. Compact branch, lane, and edge context.

The console must continue to exclude prompts, embeddings, raw messages, raw
message envelopes, secrets, and unbounded text. If a node has no detail, the
inspector should say that the node reported no bounded detail.

## Live Update Behavior

Overview uses the latest cognition graph from bootstrap and refreshes when the
authenticated SSE stream emits `control.cognition_graph_invalidated`. Debug
chat uses the same graph component for the current debug request and replaces
the optimistic running state with the reported result.

The graph should make delay visible:

- Show `updated <age> ago` using `generated_at` when present.
- Mark a snapshot stale if it has not changed for 10 seconds while status is
  `running`.
- Show the latest event/delay signal from current node and graph structure
  without requiring the user to read logs.

## Layout Rules

- Use the existing buildless static HTML/CSS/JS stack.
- Follow shadcn component anatomy already used in the console: Card, Badge,
  Button, ScrollArea, Table-like detail rows, and tooltip/inspector patterns.
- Do not introduce Node.js, React, Vite, Tailwind, Webpack, or a frontend build
  step.
- The graph must be responsive by construction, not hard-coded for one desktop
  width.
- On wide viewports, use graph plus side inspector.
- On narrow viewports, stack summary, graph, and inspector vertically.
- Hover cards must not be the only way to access detail.
- Node labels and detail text must wrap without overlapping other nodes.

## Data Contract Impact

No brain-service semantic change is required for the first implementation.
The existing `CognitionRunGraphSnapshot` contract already provides:

- `status`
- `run_id`
- `generated_at`
- `nodes`
- `edges`
- redacted node `detail`
- graph `redaction`

The console frontend will derive:

- current node id
- highlighted running node ids
- terminal node id
- branch groups
- latest event text
- freshness/staleness label

If future brain telemetry adds explicit timing or event fields, the same
gadget can display them without a second graph component.

## Testing Plan

Tests should cover:

- Static surface includes summary header, graph stage, stable inspector, and
  return-to-current behavior.
- Current-node selection prefers running, then failed, then terminal completed.
- Multiple running nodes are all highlighted while only one inspector node is
  focused.
- Empty/not-reported graphs show an explicit empty state.
- Hover/focus detail does not replace the stable inspector.
- Branch layout does not overlap in a representative parallel graph fixture.
- Debug chat pending graph shows a running current node before response.
- Overview graph refetch on `control.cognition_graph_invalidated` preserves
  fresh current-node state.
- Browser validation confirms no visible overlap, no clipping, and no console
  errors in Chrome.

## Non-Goals

- Do not implement historical graph browsing in this change.
- Do not expose raw cognition prompts, memories, embeddings, or message bodies.
- Do not replace the console UI stack.
- Do not trigger cognition from the Overview page.
- Do not create a second graph widget for Debug chat.
