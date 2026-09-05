# Cognition gadget implementation references

These artifacts accompany [the development plan](../cognition_live_observability_and_gadget_plan_2026-09-05.md).

## Preserved HTML

| Reference | Contents |
|---|---|
| [trace.html](trace.html) | Existing console shell with the proposed gadget, using selected evidence from `llmtrace_00cc86d6c36c4157ad40e59a01cd2572`. Opens during P's rejected first attempt: A1/A2/G are green and P is grey/retrying. |
| [multi-cycle.html](multi-cycle.html) | Clearly fictional three-cycle example. The first two cycles remain available; Cycle 3 begins partially complete. Evidence handoffs explain each recurrence. |
| [trace-preview.png](trace-preview.png) | Browser-checked trace view showing independently populated stages. |
| [multi-cycle-preview.png](multi-cycle-preview.png) | Browser-checked third-cycle view showing accepted, running and waiting stages together. |
| [manifest.json](manifest.json) | Provenance, byte counts and SHA-256 hashes of the preserved files. |
| [test-impact.md](test-impact.md) | Authoritative source-to-test impact and verification contract for the plan. |
| [deterministic-test-removal.md](deterministic-test-removal.md) | Completed user-authorized preparation: 327 affected deterministic definitions removed; real LLM tests preserved. |
| [deterministic-test-removal.json](deterministic-test-removal.json) | Exact removed node IDs, reasons, before/after hashes and preservation verification. |

Both HTML files are self-contained: the console CSS, avatar, selected scenario data and interaction code are embedded. Open either file in a browser; the example tabs use sibling-relative links. The temporary `127.0.0.1:8770` preview server and the author's local visualization folder are not dependencies. Copying this entire reference directory preserves the examples.

## What To Carry Into Production

- The existing console appearance and placement of Latest cognition run.
- A1, A2, G and P as separate stages with a clear dependency order.
- Grey waiting/running stages, and green validated results populated separately.
- Cycle groups, retained earlier results, evidence handoffs and the final state commit.
- Bounded stage cards backed by structured semantic details rather than a raw JSON dump.
- Stage selection, cycle expansion, expand/return diagram, following the active stage, and the narrow-panel detail dialog.

## What The Plan Changes

The reference files remain frozen, including their demonstration controls. The implementation follows these production requirements from the plan:

1. Remove Play/Replay/Pause, Next event, Show completed, speed selection, timeline scrubbing, scenario switching and simulated time/state. Render real observation lifecycle updates.
2. Implement the full failure/timeout/recovery/blocked/cancelled/availability matrix. The mockups demonstrate a P retry and an unresolved prewarm, not every failure state.
3. Draw the real text/visual join before dialog. The production caller currently awaits both enabled planning branches; the simple reference grouping is not an execution scheduler.
4. Distinguish pre-surface lifecycle operations from immediate post-turn review. The example's post-reply area is not permission to move runtime work.
5. Display actual producer timestamps. Recorded-trace playback uses stored completion times and illustrative placement for context/commit; the multi-cycle timing is fictional.
6. Render only the current approved Brain observation projection. The trace example contains selected real conversation and accepted cognition excerpts supplied for this diagnostic task; it is a visual reference, not a production fixture, raw-trace disclosure policy, or permission to expose arbitrary prompts/model output.
7. Provide separate connection health and exact run/cycle/attempt identities. A working observation stream and a successful cognition stage are different facts.

## Preservation And Authority

The user explicitly requested that the HTML mockups accompany the implementation plan. The files were copied unchanged from the browser-checked artifacts on 2026-09-05. Their hashes match the originals. They contain embedded visual assets and selected semantic examples; login credentials and the protected trace export were not included in the package.

Keep these originals immutable. If a later approved amendment needs new reference behavior, add a separately named revision and update its provenance. The development plan and canonical runtime ICD govern production contracts; screenshots and demonstration JavaScript do not override them. Build committed tests with synthetic data shaped like these cases rather than embedding the real conversation excerpts into test fixtures.
