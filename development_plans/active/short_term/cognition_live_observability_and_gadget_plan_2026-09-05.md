# Live cognition observability and cognition gadget

## Summary

- Goal: publish validated cognition information as each owner produces it and display the actual execution topology in the existing control console.
- Status: draft
- Created: 2026-09-05
- Scope boundary: observational instrumentation, the Brain-owned observation contract and transport, and the shared console cognition gadget. Preserve character judgment, model prompts, recovery policy, state-commit semantics, and adapter delivery behavior.
- Change direction: replace the terminal-only observation contract and renderer in one coordinated bigbang cutover.
- Acceptance state: live, independently populated stages; retained cognition cycles and graph attempts; truthful failure and availability states; preserved reference mockups; no production playback controls.
- Implementation authorization: this request authorizes the plan and reference package. Execution requires an approved plan and an explicit implementation instruction under `AGENTS.md`.
- Preparation authorization, 2026-09-05: the user separately instructed deletion of every existing deterministic test impacted by this development before implementation, while preserving real LLM tests. That preparation is complete: 327 test definitions removed across 48 files. The production plan remains draft.

## Confirmed Decisions And References

The owner requested two development parts: (1) real-time information emission from cognition and associated stages, and (2) a new cognition gadget based on the accepted HTML mockups. Failure behavior is required. Replay is excluded from the final product.

Use these accompanying references, which are preserved inside the repository and work without the temporary preview server:

- [Reference guide](cognition_live_observability_and_gadget_reference_2026-09-05/README.md)
- [Recorded-trace HTML](cognition_live_observability_and_gadget_reference_2026-09-05/trace.html)
- [Multi-cycle HTML](cognition_live_observability_and_gadget_reference_2026-09-05/multi-cycle.html)
- [Reference manifest](cognition_live_observability_and_gadget_reference_2026-09-05/manifest.json)
- [Exact source-to-test matrix](cognition_live_observability_and_gadget_reference_2026-09-05/test-impact.md)
- [Completed deterministic test removal](cognition_live_observability_and_gadget_reference_2026-09-05/deterministic-test-removal.md), with [exact node inventory and preservation hashes](cognition_live_observability_and_gadget_reference_2026-09-05/deterministic-test-removal.json)

The HTML files establish visual anatomy, semantic detail organization, progressive availability, cycle expansion, and stage selection. This plan governs production behavior where the demonstration differs: live transport, failures, real timestamps, actual branch dependencies, disclosure, and removal of playback.

The recorded example is `llmtrace_00cc86d6c36c4157ad40e59a01cd2572`, associated with invocation `chat:qq:ch_732699d6699040ae:805042988`. It contains one A1/A2/G/P cycle and two P attempts. A rejected P candidate is a stage attempt, not a second cognition cycle. The multi-cycle HTML is an explicitly fictional three-cycle scenario.

## Current System And Evidence

Inspection baseline: clean worktree at `f1c6f06f5b781391bf80d7df45b2c3b5fbda2b99`. Refresh this baseline when execution begins.

Preparation update: the working tree now includes the plan/reference package and the explicitly authorized deterministic test removals. The removal inventory records the baseline and every affected test definition. Production source and real LLM cases remain unchanged. Preserve these removals when establishing the implementation checkout.

| Current owner | Confirmed behavior | Required change |
|---|---|---|
| `service.py::_record_latest_cognition_graph` and `_process_chat` graph-attempt loop | The process-local observation is published after the graph succeeds or a terminal failure is handled. | Open a run before graph execution and publish progress throughout it. Finalization updates the same run. |
| `cognition_core_v3/facade.py::_run_cognition_stage` | A1, A2, G, and P validate separately; provider errors, rejected candidates, normalization, regeneration, and exhaustion already have owner decisions. | Emit those existing lifecycle decisions immediately; publish accepted semantics after validation, before awaiting unrelated trace persistence or the next stage. |
| `cognition_resolver/loop.py::_call_cognition_preserving_observations` | Full cognition can recur after capability results, duplicate requests, blockers, mixed lifecycle outcomes, and the maximum-cycle exit. | Identify every actual core invocation, including special final passes; retain each cycle and its incoming evidence. |
| `nodes/persona_supervisor2.py::stage_1_goal_resolver` | Intermediate cycles use `commit=False`; the caller commits the final replacement state once. | Show provisional cycle results separately from confirmed state commitment. |
| `brain_service/cognition_observation_projection.py::_core_source` | Current sections read the final `cognition_core_output`; previous A1/A2/G/P results cannot be recovered from that value. | Project at producer completion and retain instance-specific records. |
| `brain_service/cognition_observation_contracts.py` | V1 admits terminal states, a fixed bounded graph and detail sections. | Replace it with a strict V2 lifecycle and instance contract, with bounded detail retrieval. |
| `control_console/app.py::_stream_console_events` | Each browser stream periodically checks the latest run IDs. Same-run progress does not invalidate the view. | Subscribe once to Brain observation changes and distribute revision notifications through the existing console status SSE. |
| `control_console/static/console.js` | The renderer groups nodes by column and infers order/currentness; invalidation refreshes bootstrap. | Render producer topology and lifecycle; refresh only affected observation data. |
| `nodes/persona_supervisor2_l3_surface.py::_run_l3_text_surface_handler` | Text planning and enabled visual planning run together; the caller joins them before dialog. | Publish each branch when it resolves and draw the real join before dialog. |
| `nodes/persona_supervisor2_memory_lifecycle.py` and `brain_service/post_turn.py` | Pre-surface lifecycle routing and post-surface lifecycle review are distinct operations. | Give them distinct instances and placement. The mockup's post-reply lane does not relocate pre-surface work. |

The resolver's configured normal limit is 1–5 cycles, default 3; special final passes exist outside the normal loop. The service permits one existing safe graph retry before state commitment. Observability records actual invocations rather than deriving an assumed total from either limit.

## Scope And Change Direction

### Part 1: Live information emission

Instrument the existing semantic and deterministic owners. Cover admitted-turn preparation, perception, consumed context, first-cycle prewarm, A1, A2, G, P, resolver capability handoffs, final binding/commit, lifecycle routing, selected actions, L3 text planning, optional visual planning, dialog generation, and the immediate associated post-turn continuation. Apply the same contract to the live/debug entry, self-cognition entry, and internal DSH cognition entry.

The observation run starts when an admitted episode enters graph execution. Queue/frontline/settlement results already available then are supporting input records with their actual available timestamps or an explicit timing-unavailable marker. This plan does not turn every discarded incoming message into a cognition run or add new admission semantics.

### Part 2: Cognition gadget

Replace the shared renderer used by Overview Latest, Debug cognition, and Self Latest. Keep the existing console shell, design tokens, authentication, static frontend stack, and page placement. Add live lifecycle presentation, explicit dependency topology, cycle/attempt navigation, bounded semantic inspectors, and separate connection-health feedback.

### Must Do

1. Publish each validated stage result before the downstream stage finishes; waiting/running stages remain grey and contain no future semantic output.
2. Keep every cycle and stage attempt distinguishable within a retained run, including forced final passes and graph retries.
3. Explain each recurrence using the prior P decision, capability disposition, and returned evidence or blocker.
4. Display the confirmed final state commit independently from green provisional appraisals.
5. Preserve completed results through retries, downstream failures, connection loss, and a newer run starting.
6. Provide safe, bounded failure details and exact diagnostic correlation references.
7. Preserve the two HTML files byte-for-byte as implementation references. Keep demonstration controls solely in those reference files.

### Deferred

- Production replay, pause, speed, scrubber, next-event, show-completed, and rerun controls.
- Token streaming, partial JSON display, generated progress prose, additional LLM calls, prompt changes, new recovery strategies, or changed retry/cycle limits.
- Durable observation history, trace import/playback, database migrations, a historical-run search product, and full DSH/RAG internal execution explorers.
- Rebuilding adapters, delivery receipts, scheduler policy, memory semantics, reflection promotion, or the console frontend framework.
- Serving the reference HTML as a production page or embedding its simulation into the console.

Existing automatic graph retry remains an execution policy owned by the runtime. Removing the demonstration replay feature does not remove that recovery policy. Network cursor catch-up restores the current view and exposes no playback feature.

## Ownership And Minimal Contract

| Boundary | Owner and responsibility |
|---|---|
| Semantic results | Existing LLM stages decide appraisal, motivation, epistemic boundaries, response intent, and expression. |
| Execution truth | Stage/caller/resolver owners declare starts, accepted results, attempts, failures, skips, recurrence, and commit outcomes. |
| Internal observation interface | `cognition_shared/observation_events.py` carries scoped lifecycle events to an installed observational sink; it does not import Brain service or console packages. |
| Safe projection, storage, wire schema | Brain owns projection, strict DTO validation, revision ordering, retention, and read-only observation APIs. |
| Console transport | Console validates Brain DTOs, authenticates browser access, and multiplexes compact notifications. |
| Presentation | One console gadget renders producer-owned labels, order, dependency edges, statuses, and approved fields. It never reconstructs semantics from dialog or logs. |

The semantic question and output requirements of every existing model stage remain unchanged. Instrumentation uses accepted owner outputs and existing deterministic dispositions. It introduces no operational instructions into model-facing packets, no new context fields in prompts, and no semantic fallback.

## Part 1 Contract: Producer And Runtime

### Identity And Lifetime

Use this hierarchy:

```text
observation run (opaque run_id; independent of optional trace capture)
  graph attempt (existing service attempt number)
    cognition cycle (one actual A1 -> A2 -> G -> P invocation)
      stage instance
        model/validation attempt
```

- Allocate a unique observation `run_id` even when LLM trace capture is off. Preserve supplied `llm_trace_id` and `cognition_invocation_id` literally as optional correlation fields; neither is a cycle counter.
- Include a process `stream_epoch`, monotonically increasing run `revision`, and a start-order ordinal. A later completion of an older run cannot replace the latest-started run.
- A cycle has a monotonic display ordinal within its graph attempt and a typed entry reason: initial, capability result, duplicate request, user-input blocker, pending blocker, mixed lifecycle, or limit finalization. Retain the resolver's own index separately for diagnostics; it is not the observation identity.
- Instrument `_call_cognition_preserving_observations` around every invocation so helper-driven final passes are counted. A stage regeneration keeps the same cycle and stage instance. A service graph retry creates a new attempt with new stage/cycle instances and preserves the failed attempt.
- The first-cycle template may be displayed grey before invocation. Count started cycles separately from that template. Append later cycles only when the runtime actually enters them; never advertise a fixed future cycle total.
- Carry parent/child links only from trusted runtime lineage. A fresh scheduled, resumed, tool-result, or internal cognition episode is a separate run, not another cycle of an old invocation. Missing lineage is explicit; matching prose, channel, or timing cannot establish parentage.
- Live, self, and internal runs have distinct `run_kind` values. Internal runs are reachable as related activity or an explicitly selected retained run; they do not replace the Latest conversation or Latest self slots.

### Emission Interface

Create a small typed event/sink interface and scoped `ContextVar` binding in `cognition_shared/observation_events.py`. The service installs the sink and run context; the resolver binds the current cycle; each producing owner binds its stage instance. Propagate through awaited child tasks and reset tokens in `finally` blocks. Explicitly bind immediate post-turn activity and create a fresh scope for independently scheduled work.

The internal event vocabulary is `run_started`, `graph_attempt_started`, `cycle_started`, `stage_started`, `stage_attempt_finished`, `stage_result_ready`, `stage_terminal`, `cycle_finished`, `graph_attempt_finished`, `run_finished`, and `continuation_updated`. Each event carries its scoped identities, a UTC occurrence time, monotonic duration information where measured, and the relevant typed owner disposition. Stage-attempt completion carries attempt number and safe error classification; it never carries a rejected candidate or prompt.

For `stage_result_ready`, the producing call has already passed its own canonical parsing and validation. The Brain sink synchronously projects the explicitly allowed fields into immutable bounded data before returning. It does not retain raw input state, caller-owned mutable objects, or arbitrary model dictionaries. Publish before awaiting protected-trace writes or starting the next dependent operation. Deterministic stages publish after their real operation succeeds, including state commitment after its awaited commit returns.

Observation emission performs no network or database await. An absent sink is a supported observational mode. Projection/storage failure produces an observation-health diagnostic and keeps the previous valid snapshot; it cannot consume a model attempt, change a semantic result, delay a response behind a subscriber, or trigger a graph retry. Catch observation-owned errors at this boundary; preserve cancellation and existing execution exceptions. Do not add broad exception suppression around semantic execution.

### Canonical Wire Contract

Replace `cognition_run_observation.v1` with `cognition_run_observation.v2`; update all producers, consumers, public DTO exports, redaction recognition, and current ICDs together. Keep one canonical wire vocabulary and reject V1 at the updated boundary.

Use three strict, frozen, extra-forbid DTO families:

| DTO | Required content |
|---|---|
| Run observation | `schema_version`, `run_id`, `run_kind`, `stream_epoch`, `revision`, `started_ordinal`, UTC start/update/finish timestamps, run execution status, observation health, correlation, disclosure, graph-attempt records, ordered cycle records, stage-node summaries, typed edges, related activity, and truthful omission/count metadata. |
| Stage detail (`cognition_stage_detail.v2`) | Exact run/attempt/cycle/stage identity, detail revision, execution status, result quality, timestamps and actual duration, ordered attempt metadata, approved sections/fields/records, and count/omission metadata. |
| Change notification | Epoch, run ID/kind/revision, current latest-slot identities, changed stage IDs, and terminal/availability metadata. Keep semantic text in bounded GET responses. |

Stage nodes contain the stable producer `stage_key`, unique instance ID, label, purpose, lifecycle, result quality, bounded card summary, detail revision, dependency IDs, and whether the stage is required by the selected route. They identify a run-level lane or an exact graph attempt/cycle. Edges declare `sequence`, `reference`, `branch`, `join`, `recurrence`, or `continuation`; all endpoints refer to known instances. Layout uses these declarations and producer order, not label sorting or fabricated timestamps.

Retain the existing approved scalar/record grammar and field-level budgets: 180-character node summaries, 600-character section summaries, 4,000-character scalar values, 24 records/items, truthful truncation, strict finite numeric values, and UTC `Z` serialization. Bound each compact serialized run response and stage-detail response to 131,072 characters. Move full semantic sections to selected-stage detail GETs so earlier cycles do not disappear when the latest cycle is verbose.

The observation runtime has fixed operational bounds: 64 tracked active runs, 32 retained terminal runs with a 30-minute terminal TTL, 256 stage instances per run, 64 cycle records per run, and a 64 MiB total serialized-data budget. These are telemetry bounds, not cognition execution limits. Reserve space for compact lifecycle metadata and evict expired/old terminal detail first; active tracked runs retain identities, lifecycle, completed-stage summaries, and explicit detail-omission markers if detail capacity is reached. Never evict an active run merely because a newer run starts. If a new observation cannot fit the active-run or reserved-metadata bound, use a disabled observational session and publish a bounded latest-slot availability reason `observation_capacity_exhausted`; semantic admission/execution continues unchanged. The UI labels the previous diagram as the last observed run instead of presenting it as the unobserved new run. A compact index can reduce summary text with truthful truncation to preserve lifecycle identities within the response budget. Probe the maximum configured resolver path and its special final passes against these bounds.

### Lifecycle And Failure Presentation

Keep execution state separate from result quality and observation availability. `result_quality` is `full`, `empty`, `degraded`, or `not_reported`; it cannot make a failed operation successful.

| Stage state / condition | Tile and inspector | Dependency / run consequence |
|---|---|---|
| `pending` | Grey; “Waiting”; known upstream dependency. No semantic output. | Await the declared owner. |
| `running` | Grey with an activity indicator and elapsed time. No validated semantic result yet. | Model or deterministic operation is executing. |
| `retrying` | Grey; “Retrying · attempt N”; retain failed attempt metadata in the inspector. | Downstream remains waiting. This is not a new cycle. |
| `completed`, full | Green; validated summary and semantic sections. | Satisfies the owner's dependency. |
| `completed`, empty / inapplicable family | Green; explicit zero result or family applicability. | A successful empty lookup and an unresolved lookup remain distinguishable. An inapplicable appraisal family does not skip A1/A2. |
| `completed`, degraded | Green accepted result with an amber “Degraded” badge and disposition. | Display only the replacement/fallback actually accepted by its owner. |
| `failed` | Red; stage name, safe error code/category, attempts used, checkpoint, and existing recovery disposition. | Mark dependants `blocked` only when they cannot execute in this attempt. Preserve prior completed results. |
| `timed_out` | Red; measured/owner-reported timeout and affected operation. | Follow the owner's existing timeout route. A resolver timeout may become evidence for a new cognition cycle. |
| `blocked` | Grey with “Blocked by …” and a link to the actual failed dependency. | Distinct from ordinary waiting and deliberate skipping. |
| `skipped` | Grey dashed tile with an explicit owner reason. | Omitted optional work does not fail the run. |
| `cancelled` | Grey with a cancellation label and observed cause category. | Close affected in-flight work without inventing failed model output. Propagate cancellation normally. |
| Required semantic detail not published / projection invalid | Keep the known execution state; show “Details unavailable” and observation-health warning. | Observation failure does not become cognition failure. |

Run execution status is `running`, `completed`, `partial`, `failed`, or `cancelled`. A recovered stage followed by an ordinary accepted result can complete normally and retain a recovery badge. An accepted degraded terminal surface or an omitted failed optional branch yields “Completed with issues” (`partial`). A required exhausted stage with no allowed recovery yields `failed`. Final status comes from the authoritative caller, never the maximum severity of historical attempt records.

Settle each stage once, after the owner of its assigned recovery path resolves that path. An exhausted underlying model attempt can still lead to the existing accepted deterministic surface; record the attempt failure while the semantic stage remains active, then publish its actual accepted/degraded or failed terminal result. Do not publish a terminal stage failure and subsequently turn that same instance into ordinary success. The enclosing deadline owner distinguishes its own timeout from external cancellation before publishing the terminal stage disposition; an inner cancellation used by `asyncio.wait_for` must not first settle the stage as user-cancelled.

Additional required cases:

- **Graph retry:** keep the old failed graph attempt red and inspectable; show the new attempt as active. Successful stages in the failed attempt remain green and provisional. Publish one actual commit across the accepted final path; a post-commit error never starts a new retry.
- **Commit conflict/failure:** keep validated appraisal/plan results green, mark the commit operation failed, and block dependent surfaces. Show “State not committed”; never imply that an accepted P response persisted state.
- **Capability failure:** show the failed/blocked evidence request and its typed result. A subsequent cycle may interpret that result. Connect it to the next cycle without converting the failed request into a successful lookup.
- **Human input / accepted background continuation:** complete the current foreground execution according to its actual reply/private outcome and show “Awaiting input” or “Background work accepted” as a continuation disposition. Later re-entry creates a linked run. Do not leave a foreground stage running indefinitely.
- **Silent/private/internal outcome:** mark unselected text/visual branches skipped with the actual route reason; a successful private decision can complete with no dialog.
- **Text/visual siblings:** publish text when its own validated result is available even if visual is still running. Draw the current join before dialog. A visual failure preserves valid text and reports the existing omission/degradation policy.
- **Post-turn work:** separate the response completion from immediate lifecycle/consolidation progress. Post-turn failure cannot retroactively claim that dialog generation failed. Long-lived background work is a related activity, not an open cognition cycle.
- **Unobserved disappearance:** a dropped connection is “Updates disconnected”; a new Brain epoch with a missing old run is “Run no longer retained after restart.” Neither asserts stage failure or cancellation without an owner event. A graceful observed cancellation may be terminally recorded.
- **Long model call:** a fresh transport heartbeat keeps the run live even without a new stage result. Elapsed time is not a completion percentage or a client-invented timeout.

### Snapshot And Push Transport

Keep the console as a separate FastAPI service. Add one authenticated Brain observation-change stream and one console-lifespan subscriber for it; browser count must not multiply Brain polling or upstream subscriptions. Use the existing trusted-operator authorization boundary for `/ops/*` and the current console session/CSRF rules.

| Endpoint / surface | Contract |
|---|---|
| `GET /ops/latest-cognition-graph` | Existing response fields carry the V2 current live/self run indexes. A run becomes visible at start and changes revision throughout execution. |
| `GET /ops/cognition-runs/{run_id}` | Read the exact retained run, independent of the current latest slot. |
| `GET /ops/cognition-runs/{run_id}/stages/{stage_id}` | Return the bounded canonical stage-detail DTO, with its revision. |
| `GET /ops/cognition-observation-events` | One metadata-only change stream with epoch-aware cursors, heartbeat, and explicit gap/reset events. |
| `GET /api/cognition-runs/latest` with `view=conversation_latest` or `view=self_latest` | Dedicated console GET for the corresponding current run. |
| `GET /api/cognition-runs/{run_id}` and `/stages/{stage_id}` | Authenticated validation-only proxy views. Represent expired, missing, unavailable, and invalid data distinctly. |
| Existing `/api/stream` | Multiplex `control.cognition_changed` and observation-availability notifications alongside existing service events. Wake immediately on new events; heartbeat interval is not a delivery delay. |
| Bootstrap, Overview, Debug response | Carry the same V2 run/index DTO and availability wrapper. Avoid a second graph schema. |

Publish notification only after the new immutable run snapshot is readable. Allow one in-flight fetch per run, coalesce to the highest revision, and follow with another fetch if a newer revision arrived. A response for an older epoch/revision or different run/stage cannot replace newer selected data. Fetch full detail only for selected/expanded information; cache immutable completed-stage details by instance and detail revision.

Use bounded notification retention and subscriber queues: 256 metadata events and 64 queued notifications per subscriber. On overflow or a missing cursor, issue one gap/reset notification, discard obsolete queued notifications, and reconcile current snapshots. Retain no semantic payload in the event buffer. Reconnection resynchronizes both latest slots and the pinned run. It must not clear accepted semantic results or open another browser EventSource.

The browser continues to use one compact `/api/stream` connection. Existing process-log streaming remains independent. Remove per-browser latest-graph polling and the same-run-ID-only invalidation logic. A cognition change must not re-run full bootstrap, reset other pages, or reconnect unrelated streams.

For Debug's in-flight run, add a UUID `observation_request_id` to the console debug request. Generate it before sending the message. Forward it through a dedicated trusted-console observation header, record it as private request metadata in Brain, and expose the authorized correlation in run metadata. It is correlation only, excluded from model input, message envelopes, persistence decisions, and authentication authority. The Debug gadget accepts the matching run while the chat POST is pending; it never binds an unrelated global-latest run by channel or timing. A final response with an older revision cannot replace newer SSE-observed state.

The runtime publishes before awaiting downstream work. In a responsive localhost probe, an accepted stage must appear in the browser within one second while the next stage remains deliberately blocked. Slow subscribers never backpressure semantic execution. Preserve measured timing evidence; production displays actual owner timestamps, not timestamps reconstructed from trace rows.

## Part 2 Contract: Gadget And Semantic Content

### Layout And Interaction

Use the preserved console typography, borders, spacing, light/dark tokens, badges, and static component anatomy. Show the current input/source summary, run disposition, started-cycle count, current stage, elapsed time, connection health, and a collapsible correlation reference. Keep the main four-stage row `A1 -> A2 -> G -> P` visible together when the available width permits it.

Display graph attempts as explicit groups only when a retry exists. Within each attempt, show ordered collapsible cycle groups. Keep the active cycle open by default, compact completed prior cycles without losing their summaries, and let the operator inspect any retained cycle. Preserve an explicit stage selection and expansion state as new events arrive. Automatic following applies only while unpinned. Announce a newer run without replacing a pinned run; provide “View latest.”

Use the actual dependency structure:

```text
admitted episode / available preparation
  -> perception -> consumed context / prewarm
  -> Cycle 1: A1 -> A2 -> G -> P
       -> selected evidence capability -> returned observation
       -> Cycle 2+: A1 -> A2 -> G -> P
  -> final output binding -> confirmed state commit
  -> selected lifecycle / enqueue / action operations
  -> text content plan ----+
  -> optional visual -----+-> existing surface join -> dialog -> generated output
  -> or the selected private / no-speech terminal branch

generated output -> separately labelled immediate post-turn activity
accepted delayed work -> related future run when it actually starts
```

This is a dependency view. Runtime nodes and edges determine skipped paths and concurrency. Do not add a visual-to-dialog or dialog-to-visual sequence merely to fill a row, or hide a real join because the mockup simplified it.

Use a side inspector where width supports it, and a keyboard-accessible dialog/sheet in a narrow panel. Inspectors show ordered approved sections, useful applicability/absence information, source/provenance, timing, attempt history, and safe failure metadata. Preserve CJK, emoji, and multiline text and escape HTML. Selected details must remain visible and selectable during updates. Use concise ARIA live announcements for lifecycle changes rather than re-announcing all semantic content.

Production controls are stage selection, cycle/attempt expansion, expand/return diagram, follow current/pin, view latest, related retained run navigation, and the existing diagnostic-reference affordance. Remove all playback buttons, speed choices, demo selectors, simulated clocks, fixture-loaded runtime data, and timeline scrubbing. Keep a real elapsed-time readout; it never drives stage state.

### Required Information At Each Stage

| Stage / lane | Card when available | Inspector content and source boundary |
|---|---|---|
| Observation and admission | Safe current-event/source summary; proceed/private/no-response disposition. | Bounded producer summary, trigger/source kind, admission result/reason, current observation authority. Raw message envelopes and full conversation rows remain excluded. |
| Perception | Role-explicit current meaning and media availability. | Accepted decontextualization meaning, participant roles without private IDs, supported media observations and explicit omissions. |
| Context / prewarm | Available evidence, consumed context and unresolved/empty distinctions. | The production context/window projection actually consumed by this cycle; memory/prewarm disposition and counts, conversation progress, group-scene applicability, source provenance, and omissions. Earlier cycle input and newly returned evidence remain distinguishable. |
| A1 | Event meaning and knowledge change. | `event_agency`, `goal_threat_outcome`, `epistemic_comparison_memory`: applicability, semantic summary, cause, validated proposed axis shifts and reasons. |
| A2 | Personal/relationship meaning. | `relationship_social`, `moral_identity`, `existential_drive` with the same structure; keep proposed changes provisional. |
| G | Active goal and motivation. | Goal kind/intent/reason/cause, relational willingness, and the explicitly approved bounded `private_monologue` artifact. Provider reasoning text and raw model output remain excluded. |
| P | Response goal and answer/evidence/continuation disposition. | Goal resolution, response goal, epistemic boundary, semantic action/resolver requests, required-evidence relation, and ordinary/self/internal contract variant. Keep private action parameters out. |
| Evidence handoff | Requested capability, progress/disposition, returned finding or blocker. | Requesting P instance, semantic request, typed outcome, safe source summary, new facts, remaining unknowns, timing and exact consuming next-cycle relation. Preserve failed requests as failures. |
| Binding and final commit | Selected final cycle; committed, pending, or failed. | Actual binding/commit result, selected projection summary, provisional versus committed affect/axes, scope category, and conflict/error category. Exclude database owner keys and raw replacement state. |
| Lifecycle / actions | Selected operation and actual disposition. | Bounded intent/result, skipped reason, continuation status and permission outcome already provided by the owner. Requested, validated, executed, and delivered remain separate claims. |
| L3 text | Accepted content plan and delivery shape. | Content plan, content requirements, delivery profile, lexical avoidances, epistemic boundary, and real degraded disposition. |
| Visual | Accepted bounded directive or actual skip/failure reason. | Approved directive fields, status, duration, and recovery metadata. No raw model output or image bytes. |
| Dialog / generated output | Bounded accepted visible-message preview and generation status. | Approved text preview/count/omissions and relevant recovery result. A generated reply is not proof of an adapter receipt. |
| Immediate post-turn activity | Running/completed/failed/omitted follow-up work. | Safe lifecycle/consolidation counts and outcomes; distinct pre-surface/post-surface labels. Retain existing durable task references through approved correlation only. |

The Brain projection owns all field selection and semantic labels. Reuse its production context projection and safe field grammar for every consumer, including the Character context-consumption panel. The browser renders approved additive sections generically and does not implement a second whitelist of model fields.

### Connection And Request Errors

Display a connection banner independently from stage and run results: connecting, live, reconnecting, unavailable, invalid protocol, or expired session. Keep the last validated diagram visible with its timestamp and freshness label during interruption. A stage-detail GET error is confined to that inspector. Authentication expiry disables protected refresh and uses the existing sign-in flow.

Use heartbeat freshness to judge observation connectivity. A ten-second-old stage result alone is not a stale live run. A completed run remains a historical terminal observation rather than acquiring a failure badge as its age increases.

## Cutover Policy

Overall strategy: **bigbang**.

| Area | Instruction |
|---|---|
| Wire contract | Replace V1 with V2 in Brain publication, response DTOs, imports/exports, console validation, redaction recognition, fixtures and current ICDs in the same implementation. Keep existing route names where listed above; their observation payload becomes V2. |
| Runtime | Replace terminal reconstruction as the publication authority with the incremental store. Terminal summaries finalize that store. Retire obsolete latest globals and terminal graph builders where their responsibility is replaced. |
| Console | Replace run-ID-only invalidation, bootstrap-per-stage refresh, inferred alphabetical/column topology and old status derivation. Keep one generic detail renderer. |
| Historical records | Leave archived plans, protected LLM traces, existing episode persistence schemas, and the frozen reference mockups as historical records. The new UI performs no V1 backfill or trace reconstruction. |
| Recovery | Preserve the existing stage recovery ladder and safe graph-retry limit. Add observation hooks without alternative recovery paths. |

Restart the paired Brain/console services for the coordinated contract release when deployment is explicitly directed. During a mismatched-version connection, report invalid protocol through the existing availability boundary rather than interpreting V1 as V2. Validate both a fresh tab and an already-open tab after asset refresh.

## Change Surface

The exact path ownership and tests are enumerated in [Test Impact And Traceability](cognition_live_observability_and_gadget_reference_2026-09-05/test-impact.md). That matrix is part of this plan's fixed scope.

### Create

- `src/kazusa_ai_chatbot/cognition_shared/observation_events.py`: scoped producer event/sink protocol.
- `src/kazusa_ai_chatbot/brain_service/cognition_observation_runtime.py`: process-local run store, immutable revisions, subscriptions, retention, and observation-health isolation.
- `src/control_console/cognition_stream.py`: one lifespan-owned Brain notification subscriber and validated relay.
- `src/control_console/static/cognition-gadget.js`: the shared buildless gadget, pure view-state reduction, event/detail reconciliation and renderer. Keep existing generic section helpers in one location and import/reuse them explicitly.
- Deterministic, handoff and browser tests named `NEW` in the matrix. Use synthetic data for committed tests.

### Modify

Modify only the existing files individually named in the matrix for lifecycle hooks, typed transport, renderer integration, safe projection, fixture cutover, and current contract documentation. Within large owners such as `service.py`, `persona_supervisor2.py`, and `console.js`, changes remain restricted to the listed observation and presentation symbols.

### Delete

Delete replaced V1 observation classes/exports and their V1-only producer/renderer/invalidation logic after all governed callers move together. This is symbol removal inside the listed files, not a general module cleanup. Remove production playback code if any is introduced while porting the reference. Preserve the accompanying reference files intact.

### Keep

Keep existing model packets, output semantics, validation/recovery limits, cognition state transaction behavior, semantic resolver contracts, task execution policy, adapter transport, durable memory/episode schemas, historical plans, and protected trace storage. Observe associated work through its owning call boundary instead of adding observer fields to persisted semantic state.

## Mandatory Skills And Rules

- `development-plan`: scope, status, test impact, handoff and independent acceptance.
- `local-llm-architecture`: preserve semantic ownership, model inputs and latency.
- `control-console-web-development`: buildless stack, snapshot/SSE, safe projections and all shared gadget consumers.
- `probe-first-engineering`: run the critical live transport probe before extensive test expansion or UI polish during implementation.
- `py-style`, and `cjk-safety` when applicable: every Python edit and CJK literal handling.
- `test-style-and-execution` and `python-venv`: tests and the project interpreter `venv\Scripts\python`.
- `browser:control-in-app-browser` and frontend-testing-debugging: rendered verification; apply the browser skill before interaction.
- `llm-trace-debug` and `character-test`: only for the final live diagnostic conversation and trace comparison.

Preserve `AGENTS.md` ownership and change-control rules. Use the development workspace identified at execution start. Keep `.env` inspection outside this plan. Authentication references contain no credentials. Public observation APIs expose approved summaries and diagnostic IDs only, with existing operator authorization; raw prompts, rejected output, worker error text, action parameters, source records and private identifiers remain protected.

## Execution Roles

| Role | Contract |
|---|---|
| Implementation owner | **Responsibility:** deliver both parts as one coherent contract cutover and maintain plan/evidence. **Owned surface:** every path in the impact matrix, this plan, its registry row and test/evidence artifacts. **Authority:** implement the fixed scope after approval and explicit instruction; choose local decomposition and commands; remediate review findings. **Skills:** all applicable skills above. **Capability floor:** async Python/context isolation, strict DTOs, Brain/resolver ownership, FastAPI/SSE, buildless JS, deterministic handoff testing and browser verification. **Independence:** none for implementation; cannot independently sign off its own changes. **Output:** scoped diff, collected/passing mapped tests, live probe/browser evidence and updated checklist. **Gate:** approved plan plus implementation instruction on entry; all acceptance evidence on exit. |
| Independent reviewer | **Responsibility:** review the plan before execution and the completed implementation before acceptance. **Owned surface:** read access to the same code, contracts, tests, evidence and references. **Authority:** issue findings and pass/fail the review; remediation belongs to the implementation owner. **Skills:** development-plan, local-LLM architecture, console development, Python/test rules. **Capability floor:** independently trace lifecycle, concurrency, disclosure, rendering and source-to-test completeness across the full boundary. **Independence:** separate executor from author/remediator of the work being signed off. **Output:** review record with findings, resolved evidence and acceptance disposition. **Gate:** a concrete draft or completed implementation plus evidence; repeat independent review after remediation. |

Resolve eligible executors and configurations at runtime under the development-plan execution guidance. No fixed model or agent roster is prescribed. Every handoff records remaining scope, exact owned paths, skills, baseline, evidence, resolved executor/configuration, rationale and next gate.

## Implementation Sequence And Verification

1. **Contract gate:** review this draft, approve its fixed contracts, confirm explicit implementation authority, capture current git/served-workspace baseline, and verify the preserved reference hashes. Verify the completed pre-development removal record before production edits: all 327 retired deterministic definitions remain absent and real LLM cases/support remain preserved. Review the amended source-to-test matrix, whose nodes now describe future acceptance work.
2. **Early executable probe:** implement the smallest producer -> Brain store -> internal stream -> console SSE -> gadget path. Run real local Brain/console HTTP transport with a controlled stage invoker: release validated A1 while A2 remains blocked by an explicit test barrier. Observe A1 turn green with its semantics while A2 stays grey. Add a second cycle under the same run ID and prove its first update arrives. Interrupt/reconnect the browser stream and confirm snapshot reconciliation. This probe precedes broad test creation and styling work.
3. **Part 1 completion:** instrument all listed owners, forced final passes, graph retries, self/internal entry and immediate post-turn scopes. Finish the strict V2 projection/retention/transport and debug correlation. Exercise timeout, cancellation, projection failure, slow clients and interleaved runs with deterministic barriers.
4. **Part 2 completion:** port the reference anatomy into the shared live gadget. Implement topology, branches, attempts, cycle preservation, inspectors and availability UI. Remove demonstration controls and data. Update Overview, Debug, Self Latest and context-consumption consumer together.
5. **Mapped verification:** after the early probe, create and collect the NEW nodes in the impact matrix, run those deterministic nodes, then the specified handoff and browser cases. Derive checks from externally observable lifecycle, topology, transport and disclosure contracts. The deleted assertions are retired; they supply no baseline or requirement to restore old structure. Source-token assertions cannot prove live behavior. A broad passing suite cannot substitute for a missing NEW node.
6. **Rendered sign-off:** verify the actual changed checkout in the in-app browser. Cover progressive completion, retry/exhaustion, multi-cycle recurrence, optional failure, connection loss, exact Debug binding, empty self state, and selected self/internal outcomes. Exercise all controls, narrow and wide layout, keyboard/focus, CJK/emoji/multiline, escaping, light/dark, fresh-tab and stale-asset refresh paths. Capture URL, served checkout, session status, screenshots and zero unexpected page/console errors. Fault injections must produce the specified visible notices, not uncaught errors.
7. **Live evidence:** run one explicitly scoped real debug conversation through the current Brain, inspect its protected trace, and correlate actual stage acceptance and displayed observations. Use sequential deterministic fault probes for failures and forced recurrence rather than hoping a live model produces them. Run real LLM cases individually and inspect their evidence; no prompt-quality change is claimed by this plan.
8. **Independent acceptance:** reviewer checks the diff, required nodes, live timing evidence, failure matrix, disclosure, preserved references and absence of production playback. Remediate within scope and re-review. Mark completed/archive only after required gates pass; record deployment separately when directed.

## Acceptance Criteria

- AC1: In the blocked-downstream probe, each accepted upstream stage is readable through Brain and visible green within one second, with no downstream/final completion required.
- AC2: Grey pending/running/retrying stages have no future semantic output; accepted fields appear only under their exact instance and revision.
- AC3: A recorded-trace-shaped test retains one cycle and two P attempts, showing the rejected attempt before acceptance and no invalid candidate content.
- AC4: Three-cycle and configured-limit/special-final-pass tests retain every actual cycle, its A1/A2/G/P results, evidence relations and final commit selection.
- AC5: A graph retry retains the failed attempt, resets current-attempt context/prewarm correctly and commits exactly once on the final path. A commit failure blocks surfaces.
- AC6: Every failure/skip/degraded/cancelled/connection case in this plan produces its specified distinct state. Completed work remains inspectable.
- AC7: Text/visual completion is independently observable and the rendered join matches current execution. Immediate post-turn failure remains separate from generated-reply success.
- AC8: Same-run revisions update without bootstrap refresh; duplicate/out-of-order events, slow consumers, two clients, nested runs, restart and reconnect neither mix runs nor lose authoritative current state.
- AC9: Debug binds its exact pending request; Overview and Self Latest keep separate latest-started slots and a pinned selection survives new activity.
- AC10: Approved semantics match the producer-consumed/accepted projections across all consumers. Size limits and omitted-detail indicators are truthful, and protected fields are absent from responses, SSE and error notices.
- AC11: The production gadget uses the existing console design and works at narrow/wide widths without horizontal overflow, focus loss or uncaught errors. All visible controls work. Playback and simulation are absent from production assets.
- AC12: All exact mapped tests collect and pass; real browser and one-at-a-time live evidence are inspected; independent acceptance is recorded; both HTML references retain their manifest hashes and portable links.

## Agent Autonomy Boundaries

The implementation owner may select local helper placement within the declared files, internal names that preserve the defined contracts, CSS layout mechanics, and test command order after the early probe. Additional production paths, changed semantic/recovery policy, durable observation storage, alternative transport, disclosure expansion, or a framework change require a plan amendment. Preserve unrelated worktree changes. A newly discovered contract mismatch is recorded and resolved before that dependent portion proceeds.

## Progress And Evidence

- [x] Inspect current owners, contracts, representative tests and git baseline.
- [x] Specify both development parts, failure behavior and production playback exclusion.
- [x] Preserve both HTML mockups and screenshots with hashes and implementation notes.
- [x] Check 14 relative documentation links, all four reference hashes/byte counts, 15 existing test-function references, declared source paths and embedded HTML script syntax; review lifecycle settlement and telemetry-capacity boundaries.
- [x] Complete the subsequent user-authorized test reset: inspect 404 Python test/support files and 3,285 test definitions; remove all 327 identified affected deterministic definitions across 48 files; retire the 15 former EXISTING matrix references.
- [x] Verify the test reset: 383 remaining Python test/support files parse, all 2,958 remaining definitions match the expected inventory, preserved real LLM cases and helpers retain their content, and remaining tests have no imports to deleted test modules. Detailed preservation evidence accompanies the removal inventory.
- [ ] Independent plan review, owner approval and explicit implementation instruction.
- [ ] Early executable producer-to-browser probe.
- [ ] Part 1 instrumentation and contract/transport cutover.
- [ ] Part 2 gadget and shared-consumer cutover.
- [ ] Exact mapped test collection/execution and rendered/live evidence.
- [ ] Independent implementation acceptance and lifecycle closeout.

Planning evidence is source inspection and the previously browser-checked mockups. Production implementation tests have not been run for this documentation task. New test IDs in the accompanying matrix are specified acceptance work, not claims of existing coverage. Record execution assignments, results, artifacts, amendments and residual risks here as implementation proceeds.

The subsequent preparation performed test deletion and static preservation verification only. It changed 21 files by deletion and 27 by removing selected test functions. The real LLM suites were preserved rather than executed. The removal record supersedes the initial matrix's instructions to adapt and retain affected deterministic tests. Fresh implementation acceptance tests remain future work under this draft; production development requires its own approval and explicit instruction.
