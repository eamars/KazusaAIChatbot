# Cognition Observability ICD And Console Consistency Plan

## Summary

- Goal: replace the current cognition-graph dictionaries and duplicated
  console projections with one brain-owned, versioned cognition observation
  interface that every diagnostic consumer can validate and render directly.
- User-visible outcomes: the cognition graph reports shared-memory prewarm as
  an explicit attempted/skipped/empty/failed/completed activity; conversation
  progress and public group scene use the same detail shape and density; all
  graph views use the same labels, statuses, counts, and layout rules.
- Plan class: cross-stage runtime contract and control-console big-bang cutover.
- Status: completed.
- Decision authority: the user approves production execution and any later
  contract expansion. The architect owns the design and acceptance decisions.
- Final execution override: on 2026-08-27 the user directed Role 1 to stop the
  implementation handoff and complete the remaining frontend regression fixes,
  line audit, verification, and closure directly. This instruction supersedes
  the earlier fixed-executor binding for the final closure slice.
- Highest risks: losing evidence provenance, exposing protected cognition
  material, creating another schema owner, silently truncating detail, and
  allowing live/debug/self views to drift again.
- Acceptance summary: one strict `cognition_run_observation.v1` wire object
  leaves Brain; console code contains no cognition-field allowlist or
  reconstruction fallback; prewarm disposition survives into that object; and
  focused, regression, manifest, and browser gates pass.

## Scope And Change Direction

The current path has three independent schema owners:

```text
mutable cognition/consolidation state
    -> service.py graph dictionaries and allowlists
    -> control_console.kazusa_client allowlists and shape guesses
    -> console.js field order, labels, and generic object dumping
```

This cutover establishes this ownership path:

```text
cognition/resolver stage outcomes + surface outcomes
    -> Brain-service observation producer and strict v1 contract
    -> typed Brain response / latest-run endpoint
    -> console validation and one generic section renderer
    -> Overview Latest / Debug cognition / Self-cognition Latest
```

The Brain-service observation boundary owns the wire schema, approved source
projection, labels, ordering, detail budgets, and topology vocabulary. Its
pure contract module imports no cognition runtime or console code; its
projection module may read validated cognition artifacts. Brain service
publishes the resulting model. Console transport imports only the pure
Brain-service wire models, validates the HTTP payload, and renders its declared
sections. Consumers may add a new producer-approved section without adding a
console field name or label.

The cutover also repairs the runtime provenance loss that causes the current
prewarm symptom. The prewarm capability returns a typed outcome rather than
using the same empty RAG mapping for every disposition. Cognition retains the
outcome beside the merged RAG evidence, allowing the observation producer to
report whether prewarm ran and what safe evidence it contributed.

## Confirmed Current Findings

1. Shared-memory prewarm currently launches in parallel on eligible first
   cognition cycles and its shared rows are merged into `rag_result` before
   cognition input is built.
2. The merge retains ordinary `memory_evidence` only. Attempt, origin, result
   status, failure, empty-success, and merge counts are lost.
3. `ChatResponse` and the latest-run Brain response expose graph values as
   `dict[str, Any]`.
4. Brain `service.py`, console `kazusa_client.py`, and frontend `console.js`
   each maintain a separate detail vocabulary.
5. Brain omits `generated_at`; the console invents it at receipt time.
6. Conversation progress is projected as a mapping while public group scene is
   projected as one string even though both are consumed context sources.
7. Live and self-cognition graphs use different detail shapes. Several
   self-cognition values are silently dropped because the console expects a
   different scalar/list shape.
8. Console debug loading and request failure currently fabricate graph-shaped
   objects locally. Those objects are not Brain cognition observations.
9. The current snapshot is complete through the live cognition/surface
   boundary. Later background persistence and consolidation work occurs after
   the response snapshot and remains outside this interface version.
10. The cognition runtime remains web-free. Shared DTOs live at the
    Brain-service wire boundary; console imports that pure DTO module and no
    cognition engine/resolver/node module.
11. The current Brain-service README is authoritative for the operator graph:
    latest snapshots are process-local. The independent hybrid-loop draft's
    future persisted `cognition_chain_run.v2` is an engine-performance record,
    not this semantic graph contract.

## Mandatory Skills

- `development-plan`: preserve approval, lifecycle, traceability, review, and
  evidence gates in this document.
- `local-llm-architecture`: keep the observation path deterministic and keep
  semantic judgment with existing cognition stages.
- `control-console-web-development`: preserve shared console ownership and use
  browser validation for Overview, Debug, and Self-cognition surfaces.
- `py-style`: apply the project Python policy before every Python edit.
- `test-style-and-execution`: create and run exact deterministic tests through
  `venv\Scripts\python`; inspect each browser and test result.
- `cjk-safety`: apply its source-edit rules while modifying existing cognition
  Python modules that contain CJK text.
- `browser:control-in-app-browser`: use the in-app browser for final rendered
  validation when it is available; record the prescribed fallback when it is
  unavailable.

## Mandatory Rules

- Keep this plan at `draft` until the user explicitly approves production
  execution. Promote it to `approved` and then `in_progress` before the first
  production edit.
- Reuse the same persistent Luna executor for every production/test edit,
  remediation, test run, and browser run. A change to that fixed binding
  requires a user-approved plan amendment.
- Keep the architect as contract authority and final code-review/sign-off
  authority. The architect performs read-only review and directs the next
  implementation slice.
- Preserve the live response path, existing cognition decisions, prewarm
  eligibility, fail-soft behavior, direct-worker/call budget, authority and
  retry policy, and current response latency ownership. The sole retrieval
  input change is a prewarm-specific structural projection that excludes the
  typed active-character address mention from both the task and model-visible
  current-message context, and excludes exact typed active-turn rows from its
  history copies, without mutating the source envelope or older history.
- Keep model calls, prompts, model routes, RAG semantic judgment, dialog
  wording, persistence, adapters, and database schemas unchanged.
- Use one atomic internal contract cutover. Callers, producers, Brain response
  models, console transport, frontend renderer, tests, manifest, and docs move
  to v1 together.
- Keep prompts, raw model responses, embeddings, raw message envelopes,
  database identifiers, adapter identifiers, worker exception text, action
  parameters, handler metadata, and unapproved nested mappings outside the
  observation interface.
- Make every omission truthful. Bounded record sections report source count,
  displayed count, and the truncation flag instead of silently dropping rows.
- Keep source order for approved fields and records. Labels and ordering come
  from the Brain producer catalog rather than JavaScript.
- Treat malformed producer snapshots as an `invalid` console view envelope
  with a safe reason code and null observation. Console transport never mines
  other payload fields to reconstruct missing cognition semantics.
- Preserve existing user/concurrent work and use `apply_patch` for manual
  edits.

## Must Do

### 1. Publish The Cognition Observability ICD

Create `docs/architecture/cognition_observability_icd.md` as the authoritative
interface document. It defines:

- producer, publisher, transport, and consumer ownership;
- exact v1 models and status semantics;
- section identifiers and stable label ownership;
- value, record, node, edge, section, and whole-payload budgets;
- safe disclosure and forbidden-source rules;
- live-turn versus self-cognition semantics;
- availability and contract-failure behavior;
- versioning and big-bang cutover policy; and
- the rule that a new producer-approved section renders generically on all
  console graph surfaces.

Update `docs/architecture/cognition_contracts_design.md` to register cognition
run observability as the external inspection contract and link the detailed
ICD. Update the Brain-service, control-console, cognition-resolver, and nodes
READMEs plus `docs/HOWTO.md`.

### 2. Add One Strict Brain-Service-Owned Observation Contract

Create two Brain-service modules with a one-way dependency:

- `src/kazusa_ai_chatbot/brain_service/cognition_observation_contracts.py`:
  pure Pydantic models for the v1 wire schema, validators, limits, status
  vocabulary, canonical timestamp serialization, and payload-size checking;
- `src/kazusa_ai_chatbot/brain_service/cognition_observation_projection.py`:
  safe source projection, fixed label/order catalog, live-turn builder, and
  self-cognition builder.

Update `src/kazusa_ai_chatbot/brain_service/__init__.py` to export the wire
models only. Keep projection symbols private to Brain service. Update the Brain
service README with ownership, inputs, outputs, invariants, and test seam.

The contract contains no `Any` detail container. Its exact public shape is:

```text
CognitionRunObservationV1
  schema_version = "cognition_run_observation.v1"
  run_kind = "live_turn" | "self_cognition"
  status
  generated_at
  correlation
  sections[]
  nodes[]
  edges[]
  disclosure

CognitionObservationNodeV1
  node_id, label, stage, lane, column, category, status, summary
  section_refs[]

CognitionObservationSectionV1
  section_id, label, category, presentation, status, summary
  fields[], records[]
  reported_record_count, displayed_record_count, truncated

CognitionObservationFieldV1
  key, label, value

CognitionObservationRecordV1
  key, label, summary, fields[]

CognitionObservationEdgeV1
  source, target, kind = "sequence" | "reference", label
```

`CognitionObservationFieldV1.value` accepts strict JSON scalar values or a
bounded flat list of strict scalar values. Nested arbitrary mappings are not a
wire value. Structured material becomes ordered fields and records before it
leaves Brain.

The exact model rules are:

- every model uses `ConfigDict(extra="forbid", frozen=True)`;
- text values use constrained strict strings; Boolean values never satisfy an
  integer field;
- `generated_at` is a timezone-aware UTC `datetime` and serializes with a
  terminal `Z`;
- node ids, section ids, and edge endpoint ids match
  `^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$`;
- field keys, reason codes, and fixed record-key prefixes match
  `^[a-z][a-z0-9_]*$`; generated record keys are the unique, source-order
  values `item_01` through `item_24` within their section;
- `correlation` is `CognitionObservationCorrelationV1` with optional bounded
  strict-string fields `run_id`, `llm_trace_id`,
  `cognition_invocation_id`, and `source_calendar_run_id`;
- `disclosure` is `CognitionObservationDisclosureV1` with policy literal
  `approved_cognition_observation.v1` and this exact ordered exclusion list:
  `prompt`, `raw_model_output`, `embedding`, `raw_message`,
  `message_envelope`, `database_identifier`, `adapter_identifier`,
  `action_parameter`, `handler_metadata`, and `worker_error_text`;
- `presentation` is `fields | records`; a `fields` section has zero records,
  `reported_record_count=0`, `displayed_record_count=0`, and
  `truncated=false`, while a `records` section may also carry header fields;
- field keys are unique within their owner; record keys are unique within a
  section; section references are unique and preserve producer order;
- `node.summary` is the first non-empty referenced section summary in section
  order, bounded to 180 characters, with the lowercase status label as the
  fallback; and
- canonical payload-size validation serializes
  `model_dump(mode="json")` with UTF-8-preserving, sorted-key, compact JSON
  before comparing the 131,072-character limit.

The canonical terminal observation status vocabulary is:

```text
completed, empty, skipped, failed, partial, not_reported
```

Status meanings are exact:

- `completed`: the stage/activity completed and produced reportable content;
- `empty`: the stage/activity completed successfully with zero reportable
  content;
- `skipped`: policy or eligibility deliberately bypassed it;
- `failed`: an attempted stage/activity failed or returned an invalid owned
  contract;
- `not_reported`: the source did not provide an observation;
- `partial`: mixed subordinate dispositions or incomplete terminal output.

`CognitionRunObservationV1.status` is the narrower literal
`completed | failed | partial`. V1 builders publish terminal snapshots only.
The live producer maps the settled `EpisodeTerminalStatusV1` exactly:

| Episode terminal status | Observation result |
|---|---|
| `completed_visible` | `completed` |
| `completed_private` | `completed` |
| `completed_action` | `completed` |
| `scheduled` | `completed` |
| `failed` | `failed` |
| `cancelled` | no observation; preserve the prior latest snapshot |

A mapped `failed` remains `failed`. A mapped `completed` becomes `partial`
only when any emitted section or node is `failed` or `partial`; `empty`,
`skipped`, and `not_reported` remain truthful section coverage and do not
rewrite the settled run outcome. `visual_stage_failed=true` makes the visual
section/node failed and therefore downgrades an otherwise completed run to
`partial`; the ordinary service exception path supplies episode status
`failed`, so its top-level result remains failed. Before that failure
projection, the service checks `PipelineCancelled` and
`asyncio.CancelledError`; either clears the current observation checkpoint,
publishes no response/latest observation, and preserves the prior latest
snapshot. In-flight `pending` or
`running` cognition stages remain console loading UI outside the DTO.

Contract budgets are fixed in v1:

| Element | V1 limit |
|---|---:|
| Nodes per snapshot | 64 |
| Edges per snapshot | 96 |
| Sections per snapshot | 96 |
| Section references per node | 12 |
| Fields per section | 24 |
| Records per section | 24 |
| Fields per record | 16 |
| Labels/keys | 80 characters |
| Node summary | 180 characters |
| Section/record summary | 600 characters |
| Scalar semantic text | 4,000 characters |
| Scalar-list items | 24 items, 2,000 characters each |
| Serialized snapshot | 131,072 characters |

Validators enforce unique node/section identifiers, valid edge endpoints,
valid section references, `displayed_record_count == len(records)`,
`reported_record_count >= displayed_record_count`, and
`truncated == (reported_record_count > displayed_record_count)`. They also
require every run-kind-specific base section listed in section 5, reject a
base section owned only by the other run kind, and enforce the exact
terminal-status aggregation above. Additive producer-approved sections remain
allowed under the generic identifier/model/budget rules.

### 3. Preserve Explicit Shared-Memory Prewarm Outcome

Add `SharedMemoryPrewarmOutcomeV1` and its validator to the cognition-resolver
contract boundary. Replace the current RAG-only return and merge call shape in
one cutover.

The internal outcome contains:

```text
schema_version = "shared_memory_prewarm_outcome.v1"
status = completed | empty | skipped | failed
reason_code: one fixed lower-snake literal from the disposition table
attempted: strict bool
latency_ms: strict integer, 0..120000
retrieved_shared_count: strict integer, 0..24 projected evidence entries
merged_shared_count: strict integer, 0..24 appended evidence entries
rag_result: canonical prewarm-only projected RAG mapping
```

The validator accepts mappings only, rejects extra/missing keys and Boolean
integers, and returns a deep copy. `reason_code` is the exact literal union of
all codes in the disposition table. The RAG value must have exactly these
canonical keys: `answer`, `user_image`, `user_memory_unit_candidates`,
`character_image`, `third_party_profiles`, `memory_evidence`,
`recall_evidence`, `conversation_evidence`, `external_evidence`, and
`supervisor_trace`. It is normalized through the resolver contract's renamed
public `normalize_projected_rag_result` function, then must satisfy these
prewarm-only rules: `answer == ""`; the candidates, third-party, recall,
conversation, and external lists are empty; `memory_evidence` contains at
most 24 mapping entries; and compact UTF-8-preserving JSON is at most 65,536
characters. Nested prompt-safe RAG values retain the resolver normalizer's
600-character scalar bound and exclusion of `raw_id`, `raw_payload`, and
`raw_result`. Invalid projection shape produces `failed/projection_failed`
with the canonical empty RAG payload. Elapsed time is monotonically measured
and clamped to the declared latency bound.

`SharedMemoryPrewarmOutcomeV1.status` is exactly
`completed | empty | skipped | failed`. `call_cognition_subgraph` is the
eligibility owner and assigns
`GlobalPersonaState.shared_memory_prewarm_outcome` on every non-cancelled path
once eligibility evaluation is reached:

- cycle index other than zero: construct skipped `not_first_cycle`;
- unsupported episode: construct skipped `unsupported_episode`;
- eligible cycle zero: start `run_first_cycle_shared_memory_prewarm` and retain
  its validated result;
- successful evidence result: finalize it through
  `merge_shared_memory_prewarm_outcome` and store the finalized outcome beside
  the merged `rag_result`.

The exact signatures are:

```python
def build_skipped_shared_memory_prewarm_outcome(
    reason_code: Literal["not_first_cycle", "unsupported_episode"],
) -> SharedMemoryPrewarmOutcomeV1: ...

async def run_first_cycle_shared_memory_prewarm(
    state: GlobalPersonaState,
) -> SharedMemoryPrewarmOutcomeV1: ...

def merge_shared_memory_prewarm_outcome(
    base_rag_result: dict[str, Any],
    outcome: SharedMemoryPrewarmOutcomeV1,
) -> tuple[dict[str, Any], SharedMemoryPrewarmOutcomeV1]: ...
```

The old `merge_shared_memory_prewarm_result` symbol is removed in the same
cutover.

Reason mapping is fixed:

| Runtime disposition | Status | Reason code |
|---|---|---|
| Cycle is not first cycle | skipped | `not_first_cycle` |
| Episode cannot use prewarm | skipped | `unsupported_episode` |
| Query is empty after exact character-mention removal | skipped | `empty_query_after_character_mention` |
| Worker raises an owned operational exception | failed | `worker_error` |
| Worker violates its result shape | failed | `worker_contract_invalid` |
| Worker resolves no usable shared result | empty | `worker_unresolved` |
| Valid result contains zero safe shared rows | empty | `no_shared_memory` |
| Safe rows cannot produce the owned projection | failed | `projection_failed` |
| Safe rows are ready for caller merge | completed | `shared_memory_ready` |
| Safe rows merge into cognition evidence | completed | `shared_memory_merged` |

Outcome invariants are fixed:

- skipped: `attempted=false`, `latency_ms=0`, both counts are zero, and the
  projected empty RAG result is present;
- eligible mention-only skip: the same skipped invariant applies because the
  memory worker was never called;
- empty or failed after worker dispatch: `attempted=true`,
  `latency_ms >= 0`, and `merged_shared_count=0`;
- `shared_memory_ready`: `attempted=true`,
  `retrieved_shared_count > 0`, and `merged_shared_count=0`;
- `shared_memory_merged`: `attempted=true`, retrieved and merged counts are
  equal and greater than zero; and
- every other field/status/reason/count combination fails validation.

`outcome.rag_result` is always the validated, deep-copied, prewarm-only
projection produced by `_empty_projected_rag_result` or `project_known_facts`;
it never contains a raw worker object. The caller's merged `rag_result` remains
a separate state value. Merge preserves the base evidence order, then appends
each safe prewarm `memory_evidence` row in prewarm source order. V1 deliberately
does no semantic deduplication. The count unit is one post-projection
`memory_evidence` entry, not one raw worker row: source rows grouped into one
partition by `project_known_facts` count as one; invalid/non-shared raw rows
are omitted before projection and count as zero. `retrieved_shared_count` is
exactly `len(outcome.rag_result["memory_evidence"])` and
`merged_shared_count` is the number of those entries appended; pre-existing
base entries are excluded. Merge accepts only a validated
`completed/shared_memory_ready` outcome. The base must be a mapping and its
`memory_evidence`, when present, must be a list; otherwise merge raises
`ResolverValidationError("base_rag_result_invalid")` before mutation. Merge
deep-copies and otherwise preserves every base key/value, including
owner-approved fields such as `media_evidence`; it does not normalize or drop
the full resolver payload. A
second call with `shared_memory_merged` raises
`ResolverValidationError("prewarm_outcome_not_ready")`, preventing repeated
merge duplication. The finalized outcome retains the prewarm-only RAG value
and changes only reason/count. Failed, empty, and skipped outcomes bypass the
merge function and leave the base state value unchanged.

The owned recoverable worker exception set is exactly `OpenAIError`,
`DatabaseBackendError`, and built-in `TimeoutError`; each maps to
`failed/worker_error` without exception text. A non-mapping worker value, a
`resolved` value outside strict Boolean, or a non-list `result` maps to
`failed/worker_contract_invalid`; strict `resolved=false` maps to
`empty/worker_unresolved`. Unexpected exceptions propagate to
the normal cognition failure boundary. Projection catches only its declared
`KeyError`, `TypeError`, `ValueError`, and `ResolverValidationError` inputs and
maps them to `failed/projection_failed`.

Cancellation continues to propagate through the existing cognition
cancellation path and produces no terminal prewarm outcome or cognition
observation. The console view wrapper reports the request-level availability
or error state separately; it never invents a cancelled cognition stage.

`GlobalPersonaState.shared_memory_prewarm_outcome:
NotRequired[SharedMemoryPrewarmOutcomeV1]` retains the validated
outcome beside `rag_result` for every non-cancelled eligibility disposition.
The same finalized disposition survives a later graph failure through the
existing request-scoped V2 attempt-ledger boundary. Extend
`V2InvocationAttemptLedger` with private fields
`prewarm_checkpoint_graph_attempt: int` and
`shared_memory_prewarm_checkpoint: dict[str, object] | None`, plus exact
record/snapshot helpers. Binding a new graph attempt clears the prior
checkpoint. The recorder accepts only an already validated outcome, deep
copies it, stores the current graph-attempt number, and overwrites
`shared_memory_ready` with `shared_memory_merged` for the same attempt.

`call_cognition_subgraph` records skipped outcomes immediately, records an
eligible `shared_memory_ready` outcome immediately after awaiting the task,
and records the finalized merged outcome immediately after merge. It also
places the same finalized model in its returned state. If identity preparation
fails before the eligible task is consumed, a completed task result is
validated and checkpointed before cleanup; an unfinished task is cancelled
and has no terminal outcome. If cognition/surface later fails, the Brain
failure path deep-copies `original_initial_state`, injects the revalidated
current-attempt checkpoint under `shared_memory_prewarm_outcome`, and passes
that mapping to `build_live_cognition_observation`; it never reads a previous
graph attempt. Successful projection uses the returned state as authority.
Cancellation clears/discards the checkpoint and maps to no observation. This
carrier is diagnostic only and never enters prompts,
model inputs, persistence, or action ownership.
Evidence merging preserves the existing prompt-safe shared rows and does not
change their semantic authority. The observation producer emits a required
`evidence.shared_memory_prewarm` section on every live cognition snapshot that
reaches prewarm eligibility evaluation. It includes status, reason, attempted,
latency, retrieved count, merged count, and safe evidence records. Worker raw
results, errors, memory IDs, and query duplication remain excluded.

### 4. Normalize Conversation Progress And Group Scene

Add
`GlobalPersonaState.public_group_scene_context: NotRequired[GroupSceneContextV1]`
and the equivalent field to `CognitionState`, plus these exact discriminator
fields on both state contracts:

```text
public_group_scene_projection_status: NotRequired[
    completed | skipped | failed
]
public_group_scene_projection_reason: NotRequired[
    available | not_group | projection_unavailable
]
```

`persona_supervisor2` keeps the existing `public_group_scene` prompt string
separately and assigns the discriminator on every terminal path: successful
group projection stores the typed context plus `completed/available`;
non-group input stores no typed context plus `skipped/not_group`; caught group
projection failure stores no typed context plus
`failed/projection_unavailable`. The producer never infers disposition from
typed-field absence or parses the prompt string. It projects conversation
progress and public group scene as peer context-source sections:

```text
context.conversation_progress
context.public_group_scene
```

Both sections use:

- category `context`;
- presentation `records`;
- the same section header layout and status badge;
- one bounded summary;
- ordered fields for status/source/continuity/counts;
- ordered semantic records;
- the same record and text budgets; and
- explicit reported/displayed/truncated counts.

Conversation progress records come from its typed semantic events and prompt
fields. Public group scene records come from `GroupSceneContextV1.turns`, with
visible participant count and omitted-turn count as fields. Transport IDs,
conversation row IDs, platform IDs, and trace IDs remain excluded.

Group channel with a valid context reports `completed`; a valid zero-record
context reports `empty`; a non-group run reports `skipped`; and a group run
whose context projection failed reports `failed` with safe reason code
`projection_unavailable`.

The current self-cognition group-scene artifact boundary carries only the
rendered string and no trustworthy structured scene disposition. The v1 self
builder therefore always emits `context.public_group_scene` as `not_reported`
and never parses that string. Self prewarm reads only
`ARTIFACT_COGNITION_OUTPUT.shared_memory_prewarm_outcome`; self progress reads
only `ARTIFACT_COGNITION_INPUT.source_packet.conversation_progress`; either is
`not_reported` when its typed source is absent. A future typed self group-scene
artifact is an additive producer change requiring separate approved scope.

### 5. Normalize All Graph Detail And Labels

The producer catalog owns every domain label and section order. Use canonical
sections for queued input, response decision, appraisals, goal, response plan,
affect, reasoning, context consumption, memory evidence, prewarm, conversation
progress, group scene, actions, visual directives, visible messages, and
self-cognition route/consolidation information.

Similar data categories use the same presentation form:

| Category | Presentation | Required information density |
|---|---|---|
| Input/decision/reasoning prose | `fields` | status, summary, ordered named fields |
| Evidence/context | `records` | status, summary, source/display counts, bounded records |
| Action request/result/continuation | `records` | kind, decision/status, reason/outcome, timing/visibility where safe |
| Surface text/directives | `records` | ordered fragments/directives and explicit empty/skipped state |
| Diagnostic/availability | `fields` | safe reason code and source availability only |

Live-turn and self-cognition node sets may differ because their runtimes differ.
Equivalent categories use the same section models, labels, status meanings,
budgets, and renderer. Each reasoning node references a context-consumption
section; unavailable self artifacts report `not_reported` instead of silently
omitting the category.

The edge vocabulary remains `sequence | reference`. Update fake fixtures that
currently emit `fork | join` and add a positive rendered-edge assertion.

The v1 base section catalog is fixed as follows. Field and record order matches
the order shown; the producer omits an optional empty field while retaining
every required section and its terminal status. Producer-approved additive
sections use the same generic model and identifier grammar and require no
consumer vocabulary change.

| Section id / label | Presentation | Ordered fields | Ordered record fields |
|---|---|---|---|
| `input.turn` / Queued turn | fields | Input, Reply context, Channel scope | none |
| `decision.response` / Response decision | fields | Should respond, Reason | none |
| `cognition.appraisals` / Semantic appraisals | records | Applicable count, Reported count | Family, Applicable, Semantic summary, Cause summary, Axis changes |
| `cognition.goal` / Character goal | fields | Goal kind, Intent, Reason, Cause summary | none |
| `cognition.response_plan` / Response plan | fields | Goal resolution, Response goal, Epistemic boundary, Action request count, Resolver request count | none |
| `cognition.affect` / Affect projection | records | Reported count | Emotion, Phase, Intensity, Trend, Cause summary |
| `reasoning.subjective` / Subjective reasoning | fields | Private monologue, Logical stance, Character intent, Judgment note | none |
| `reasoning.context_consumption` / Context consumption | records | Overall status, Consumer count | Stage, Source kind, Status, Summary, Details |
| `evidence.memory` / Memory evidence | records | Retrieval answer, Reported count | Source kind, Summary, Content, Title, Relevance, Recency, Due state, Evidence boundary notes |
| `evidence.shared_memory_prewarm` / Shared-memory prewarm | records | Attempted, Reason code, Latency ms, Retrieved count, Merged count | Source kind, Summary, Content, Relevance, Evidence boundary notes |
| `context.conversation_progress` / Conversation progress | records | Status, Continuity, Turn count, Current thread, Character stance, User goal, Current blocker, Emotional trajectory, Episode narrative, Overused moves | Semantic summary, State, Actor, Action, Object, Beneficiary, Precondition |
| `context.public_group_scene` / Public group scene | records | Status, Visible participants, Visible participant count, Omitted turn count | Role, Speaker, Text, Addressed names, Reply-to name, Scene position, Anchor kind |
| `action.requests` / Action requests | records | Reported count | Action kind, Decision, Detail, Reason |
| `action.results` / Action results | records | Reported count | Action kind, Status, Visibility, Outcome, Reason, Due at |
| `action.continuation` / Action continuation | records | Reported count | Mode, Objective, Status, Reason, Due at |
| `surface.visual_directives` / Visual directives | records | Reported kind count | Directive kind, Values |
| `surface.visible_messages` / Visible messages | records | Message count | Position, Text |
| `self.source` / Self-cognition source | fields | Source kind, Summary, Reason, Due state | none |
| `self.route` / Self-cognition route | fields | Decision, Reason, Next topic | none |
| `self.consolidation` / Self-cognition consolidation | fields | Status, Summary, Changes | none |

Flat value projection is closed and deterministic. The producer implements
the rules below directly; it does not expose or call a generic recursive
flattener:

- `Axis changes` is a flat string list in source order. Each valid source row
  becomes `<axis>: <shift>` and appends ` — <reason>` only when a safe scalar
  reason exists. Invalid row shapes are omitted and make the section
  `partial`.
- `reasoning.context_consumption` creates one record per `(stage,
  source_kind)` pair in the closed source order defined below. `Details` is a
  flat string list of exact `key=value` projections; no unlisted key is read.
  Digests, source-update timestamps, database or transport identifiers, and
  arbitrary nested source values are omitted.
- `Values`, `Addressed names`, and `Overused moves` are bounded flat string
  lists in source order. Visual-directive mappings become one record per
  directive kind; directive values never remain mappings.
- Action `Detail`, `Reason`, `Outcome`, and `Objective` accept safe strict
  scalar text only. A mapping/list where scalar text is required is omitted
  and makes the affected action section `partial`; action parameters and
  handler metadata never receive a textual fallback.
- `self.consolidation.Changes` is a flat string list containing, in this order,
  `consolidation_called=<true|false>`, `scheduled_event_count=<integer>`,
  `cache_evicted_count=<integer>`, followed by each safe `write_success`
  snake-case key as `<key>=<true|false>` in source order. Protected keys, keys
  outside the lower-snake grammar, and non-Boolean
  write-success values are omitted and make the section `partial`.
- A wrong source type is never stringified. The affected field/record is
  omitted, its containing section becomes `partial`, and reported/displayed
  counts continue to describe source rows versus emitted rows truthfully.
  The renderer displays scalar lists as ordered bullet rows and performs no
  semantic conversion.

The source-to-wire projection table is executable and exhaustive. `state`
means the settled persona/consolidation mapping, `core` means the validated
`cognition_core_output` with schema `cognition_output.v3`, and `artifacts`
means the self-cognition artifact mapping. A slash lists scalar fallbacks in
left-to-right order; the first non-empty value wins. Header counts are derived
from the source container named in the row.

| Section | Exact source path(s) and projection |
|---|---|
| `input.turn` | `state.user_input -> Input`; `state.reply_context -> Reply context` as the ordered flat list `reply_to_display_name`, `reply_excerpt`, then each `reply_attachments[]` row's `media_kind`, `description`, `summary_status`; `state.cognitive_episode.target_scope.channel_type -> Channel scope` |
| `decision.response` | `graph_result.should_respond -> Should respond` as strict Boolean; `graph_result.reason_to_respond -> Reason` |
| `cognition.appraisals` | `core.appraisals[]`; one reported row per list item; exact fields `family`, `applicable`, `semantic_summary`, `cause_summary`, and flattened `axis_changes[]` using only `axis`, `shift`, `reason` |
| `cognition.goal` | `core.active_character_goal` exact scalar keys `goal_kind`, `intent`, `reason`, `cause_summary` |
| `cognition.response_plan` | `core.response_plan` keys `goal_resolution`, `response_goal`, `epistemic_boundary`; counts are the raw lengths of mapping rows in `action_requests` and `resolver_requests` |
| `cognition.affect` | `core.affect_projection[]`; exact fields `emotion`, `phase`, `intensity`, `trend`, `cause_summary`; strict finite numeric intensity remains numeric and every other value is strict text |
| `reasoning.subjective` | `core.private_monologue -> Private monologue`; `state.logical_stance`, `state.character_intent`, and `state.judgment_note` map directly; no fallback to prompt or raw model output |
| `evidence.memory` | concatenate, in order, `state.rag_result.memory_evidence`, `conversation_evidence`, `external_evidence`, `recall_evidence`, and `media_evidence`; `Source kind` is the fixed owning list name; mapping rows read only `summary`, `fact`, `excerpt`, `content`, `title`, `relevance`, `recency`, `due_state`, and `evidence_boundary_notes`; a string row maps only to `Content` |
| `evidence.shared_memory_prewarm` | live `state.shared_memory_prewarm_outcome`; self `ARTIFACT_COGNITION_OUTPUT.shared_memory_prewarm_outcome`; headers map outcome fields directly; records come only from `outcome.rag_result.memory_evidence[]`, use fixed source kind `shared_memory`, and the same evidence allowlist |
| `context.conversation_progress` | live `state.conversation_progress`; self `ARTIFACT_COGNITION_INPUT.source_packet.conversation_progress`; source must have schema `conversation_progress_prompt.v2`; headers map `status`, `continuity`, `turn_count`, `current_thread`, `character_stance`, `user_goal`, `current_blocker`, `emotional_trajectory`, `episode_narrative`, and `overused_moves`; records are `events[]` using only `semantic_summary`, `state`, `actor`, `action`, `object`, `beneficiary`, and `precondition` |
| `context.public_group_scene` | live exact discriminator fields plus `state.public_group_scene_context` schema `group_scene_context.v1`; headers map `visible_participants`, its length, and `omitted_turn_count`; records are `turns[]` using only `role`, `speaker_name`, `text`, `addressed_names`, `reply_to_name`, `scene_position`, and `anchor_kind`; self is always `not_reported` in v1 |
| `action.requests` | `core.response_plan.action_requests[]`; exact fields `action_kind`, `decision`, `detail`, `reason` |
| `action.results` | `state.action_results[]`, with self fallback `artifacts[ARTIFACT_ACTION_ATTEMPT]` as one row only when the list is empty; `Action kind = action_kind/kind`, `Status = status`, `Visibility = visibility`, `Outcome = result_summary/outcome/objective_summary`, `Reason = reason`, `Due at = due_at/deadline` |
| `action.continuation` | concatenate `state.action_continuation` mapping/list, each `state.action_specs[].continuation`, `graph_result.future_promises[]`, then self `ARTIFACT_ROUTE_EFFECT.next_topic`; `Mode = mode/episode_type`, `Objective = objective/objective_summary/summary/title/text/next_topic`, `Status = status/due_state`, `Reason = reason/condition`, `Due at = due_at` |
| `surface.visual_directives` | `state.action_directives.visual_directives/state.visual_directives`; directive kinds are exactly `facial_expression`, `body_language`, `gaze_direction`, `visual_vibe`; one record per kind in that order and `Values` is its strict string list |
| `surface.visible_messages` | live `graph_result.final_dialog`; self precedence is `ARTIFACT_ACTION_CANDIDATE.text`, then its `messages`, then cognition output `final_dialog`, then route effect `visible_dialog`; one record per strict string with one-based `Position` |
| `self.source` | `ARTIFACT_COGNITION_INPUT.source_packet`: `case_name -> Source kind`, `instruction -> Summary`, `actionability -> Reason`, `semantic_due_state -> Due state` |
| `self.route` | `ARTIFACT_RUN_RECORD.selected_route -> Decision`; `ARTIFACT_ROUTE_EFFECT.effect_summary -> Reason`; `Next topic` uses the first strict scalar from `next_topic.summary`, `title`, `text`, `objective` in that order |
| `self.consolidation` | `ARTIFACT_CONSOLIDATION_OUTCOME`; `Status` is completed when the artifact is a mapping and skipped when absent; `Summary` uses its strict scalar `summary`; `Changes` follows the exact flat rule above |

Context-consumption records have this exact stage/source order and source
mapping:

1. `settled_relevance`: `state.settled_relevance_context_consumption` sources
   `character_operational_context`, `relationship_context`, then `style`.
2. `cognition`: `state.cognition_input` sources
   `character_operational_context`, `relationship_context`, then
   `group_engagement_action_context`.
3. `surface`: when `state.text_surface_input` is a mapping, source `style` from
   `state.interaction_style_context.surface`.
4. `health`: settled-relevance `predecessor`, then
   `state/graph_result.episode_trace.attempt_diagnostics[]`, then
   `state.consolidation_metadata.character_operational_receipt`.

For `character_operational_context`, affect rows (maximum 3) use the exact key
order `emotion_id`, `intensity`, `phase`, `trend`, `root_kind`, `cause_class`,
`freshness`; pressure rows (maximum 4) use `kind`, `salience`, `lifecycle`,
`cause_class`, `freshness`. Relationship axes use, in order, `familiarity`,
`positive_regard`, `trust`, `attachment`, `desired_closeness`,
`perceived_closeness`, `care`, `boundary_safety`, `exclusivity`,
`unresolved_injury`, `salience`; causal rows (maximum 2) use `entity_kind`,
`semantic_summary`, `salience`, `lifecycle`, `freshness`; relationship affect
rows (maximum 2) use `emotion_id`, `intensity`, `phase`, `trend`, `freshness`;
then `relationship_freshness` and `evidence_freshness` are appended.

Style record order is consumer role, then source `user`, `group_channel`.
Relevance style uses `engagement_guidelines` (maximum 3); surface overlay uses
`speech_guidelines`, `social_guidelines`, `pacing_guidelines`, and
`engagement_guidelines` (maximum 8 each). Each includes `status`, non-negative
integer `revision`, and `confidence` when valid. Group engagement uses
`engagement_guidelines` (maximum 3) then `confidence`. Predecessor uses
`status`, `watermark`, `awaited_count`, `timed_out_count`, `wait_ms`; attempts
(maximum 8) use `stage`, `error_code`, `attempt_count`, `final_status`; receipt
uses only `status`, `error_code`, `durable`, and `attempt_count`.

Each context record's `Stage` and `Source kind` are those fixed catalog values.
`Status` uses an owned source status only when it is one of
`active | empty | missing | failed | healthy | degraded`; otherwise it is
`completed` for valid projected detail, `empty` for valid zero detail,
`partial` for mixed valid/invalid detail, or `failed` for an invalid owned
source. `Summary` reads only an explicit source `summary` or
`semantic_summary` scalar and is otherwise omitted. `Details` contains the
ordered `key=value` strings defined above. The section is `failed` when every
reported source is failed/invalid, `partial` when valid and failed/invalid
sources mix, `empty` when all reported sources are valid but empty,
`not_reported` when no source container is present, and `completed` otherwise.
Its `Overall status` field exactly repeats that section status and `Consumer
count` equals `displayed_record_count`.

For every records section, `reported_record_count` counts every item in the
named source list before type filtering; `displayed_record_count` counts
emitted valid records. A non-mapping item, wrong scalar type, invalid schema,
or invalid finite number is omitted and makes the section `partial`; invalid
rows never count toward prewarm retrieved/merged counts. A missing typed source
is `not_reported`, a valid empty source is `empty`, all-valid non-empty content
is `completed`, and a completely invalid owned source is `failed`. All source
ordering above is preserved before the 24-record display cap and truthful
truncation fields are applied.

Visual disposition is exact: feature/debug disabled or
`should_respond=false` is `skipped`; `visual_stage_failed=true` is `failed`;
`visual_stage_reached=false` is `not_reported`; a valid directive mapping with
zero values is `empty`; mixed valid/invalid kind values are `partial`; a wholly
invalid mapping is `failed`; and valid non-empty values are `completed`.

The stable machine field keys are the lowercase snake-case form of the labels
in this table. Section and record summaries are existing semantic text selected
by the producer; deterministic code performs no new semantic summarization.

The live node catalog and section-reference order are:

| Node id / label | Stage / lane / column / category | Section refs |
|---|---|---|
| `input.turn` / Queued turn | Input / input / 1 / input | `input.turn` |
| `decision.response` / Response decision | Decision / gate / 2 / decision | `decision.response` |
| `cognition.meaning` / Meaning appraisal | Cognition / cognition / 3 / appraisal | `cognition.appraisals` |
| `cognition.goal` / Character goal | Cognition / cognition / 3 / goal | `cognition.goal` |
| `cognition.response` / Response plan | Cognition / cognition / 3 / response | `cognition.response_plan` |
| `cognition.affect` / Affect projection | Cognition / cognition / 3 / affect | `cognition.affect` |
| `reasoning.context` / Reasoning and context | Reasoning / cognition / 3 / reasoning | `reasoning.subjective`, `reasoning.context_consumption` |
| `evidence.memory` / Memory and context | Evidence / memory / 3 / memory | `evidence.shared_memory_prewarm`, `evidence.memory`, `context.conversation_progress`, `context.public_group_scene` |
| `action.results` / Actions | Actions / action / 3 / action | `action.requests`, `action.results`, `action.continuation` |
| `surface.visual` / Visual directive | Surface / surface / 4 / visual | `surface.visual_directives` |
| `surface.visible` / Visible surface | Surface / surface / 4 / dialog | `surface.visible_messages` |

The self-cognition node catalog reuses equivalent ids and adds owned self
stages:

| Node id / label | Stage / lane / column / category | Section refs |
|---|---|---|
| `self.source` / Source case | Input / input / 1 / source | `self.source` |
| `cognition.meaning` / Meaning appraisal | Cognition / cognition / 2 / appraisal | `cognition.appraisals` |
| `cognition.goal` / Character goal | Cognition / cognition / 2 / goal | `cognition.goal` |
| `cognition.response` / Response plan | Cognition / cognition / 2 / response | `cognition.response_plan` |
| `cognition.affect` / Affect projection | Cognition / cognition / 2 / affect | `cognition.affect` |
| `reasoning.context` / Reasoning and context | Reasoning / cognition / 2 / reasoning | `reasoning.subjective`, `reasoning.context_consumption` |
| `evidence.memory` / Memory and context | Evidence / memory / 2 / memory | `evidence.shared_memory_prewarm`, `evidence.memory`, `context.conversation_progress`, `context.public_group_scene` |
| `self.route` / Route decision | Decision / decision / 3 / route | `self.route` |
| `action.results` / Actions | Actions / action / 4 / action | `action.requests`, `action.results`, `action.continuation` |
| `surface.visual` / Visual directive | Surface / surface / 4 / visual | `surface.visual_directives` |
| `surface.visible` / Visible surface | Surface / surface / 4 / dialog | `surface.visible_messages` |
| `self.consolidation` / Consolidation | Continuity / memory / 5 / continuity | `self.consolidation` |

Required section presence is exact. A builder emits each id in its run-kind
set once, even when its truthful status is `empty`, `skipped`, `failed`, or
`not_reported`:

```text
live_turn:
  input.turn
  decision.response
  cognition.appraisals
  cognition.goal
  cognition.response_plan
  cognition.affect
  reasoning.subjective
  reasoning.context_consumption
  evidence.memory
  evidence.shared_memory_prewarm
  context.conversation_progress
  context.public_group_scene
  action.requests
  action.results
  action.continuation
  surface.visual_directives
  surface.visible_messages

self_cognition:
  cognition.appraisals
  cognition.goal
  cognition.response_plan
  cognition.affect
  reasoning.subjective
  reasoning.context_consumption
  evidence.memory
  evidence.shared_memory_prewarm
  context.conversation_progress
  context.public_group_scene
  action.requests
  action.results
  action.continuation
  surface.visual_directives
  surface.visible_messages
  self.source
  self.route
  self.consolidation
```

`not_reported` means the current run-kind source boundary does not carry that
typed artifact; it is not a run failure. In v1, self prewarm and progress use
only the exact typed artifact paths above, while self group scene is always
`not_reported`. Their required presence therefore does not force a completed
self run to `partial`.

Every catalog section is present for its listed run kind. A missing source maps
to `not_reported`; deliberate policy/eligibility bypass maps to `skipped`;
successful zero content maps to `empty`; valid content maps to `completed`;
source-contract failure maps to `failed`; and mixed valid/failed subordinate
records map to `partial`. Node status is the deterministic aggregate of its
referenced sections in this priority order:
`failed`, `partial`, `completed`, `empty`, `skipped`, `not_reported`.

Edges use the exact catalog below:

- live sequence: `input.turn -> decision.response -> cognition.meaning ->
  cognition.goal -> cognition.response -> action.results`;
- live references: `evidence.memory -> cognition.meaning`,
  `cognition.response -> cognition.affect`,
  `cognition.response -> reasoning.context`, and each of
  `cognition.response`, `reasoning.context`, `evidence.memory`, and
  `action.results` to both surface nodes where the endpoint exists;
- self sequence: `self.source -> cognition.meaning -> cognition.goal ->
  cognition.response -> self.route -> action.results -> self.consolidation`;
- self references: `evidence.memory -> cognition.meaning`,
  `cognition.response -> cognition.affect`,
  `cognition.response -> reasoning.context`, `self.route` to both surface
  nodes, and each surface node to `self.consolidation`.

The projection entrypoints are exact:

```python
def build_live_cognition_observation(
    *,
    graph_result: Mapping[str, Any],
    persona_state: Mapping[str, Any],
    run_id: str,
    cognition_invocation_id: str,
    terminal_status: EpisodeTerminalStatusV1,
    visual_stage_failed: bool,
    visual_stage_reached: bool | None,
    failure_code: str,
    generated_at: datetime,
) -> CognitionRunObservationV1 | None: ...

def build_self_cognition_observation(
    *,
    artifact_payloads: Mapping[str, Any],
    visual_stage_failed: bool,
    visual_stage_reached: bool | None,
    generated_at: datetime,
) -> CognitionRunObservationV1 | None: ...
```

`failure_code` contains only the existing typed failure classification and is
empty on success. Builders accept an injected UTC `generated_at` so tests are
deterministic. They never accept exception text. The live caller maps the
current `cognition_attempt_ledger.cognition_invocation_id` to both
`correlation.run_id` and `correlation.cognition_invocation_id` on success and
failure, and maps the
validated `graph_result.llm_trace_id` to `correlation.llm_trace_id`. The self
builder reads `ARTIFACT_RUN_RECORD` as a mapping and performs this exact local
projection validation: `run_id` must be a non-empty strict string of at most
120 characters; `status` must be `completed | failed | cancelled`;
`llm_trace_id` and `source_calendar_run_id`, when present, must be strict
strings of at most 120 characters. `completed` maps to `completed`, `failed`
maps to `failed`, and `cancelled` returns no observation. Because the runner
binds both values to the same canonical
`self_cognition_run:<trigger_id>` value, `correlation.cognition_invocation_id`
is exactly the validated `run_id`; `correlation.llm_trace_id` is the validated
run-record value or that same `run_id` fallback; the calendar id is omitted
when absent. A missing/invalid record returns no self observation, leaves the
prior latest snapshot unchanged, and records the existing safe publisher
failure. No new self-cognition artifact or record validator is required.
Edge labels are required bounded strict strings and every v1 catalog edge
supplies the empty string.

### 6. Cut Brain Publication To The Typed Contract

Move cognition graph assembly and all graph-specific source allowlisting out
of `service.py` into
`brain_service.cognition_observation_projection`. `service.py` supplies
explicit live/self builder inputs, stores typed latest snapshots, and publishes
model dumps at the FastAPI boundary.

Change `ChatResponse.cognition_graph` and
`OpsLatestCognitionGraphResponse` fields to
`CognitionRunObservationV1 | None`. Brain owns `generated_at`; the console
retains that timestamp unchanged. Latest snapshot copies use Pydantic deep
copy semantics.

The v1 live snapshot ends after cognition, selected actions, and surface
generation. Existing later post-turn persistence/consolidation remains outside
v1 and receives no fabricated completed node. Self-cognition may report its
recorded consolidation artifact because that artifact belongs to the completed
self run.

### 7. Cut Console Transport To Validation-Only Consumption

Remove the console-local cognition graph models, semantic field allowlists,
nested shape guessing, inferred fallback projection, and local timestamp
replacement. Import the Brain-owned observation models into console response
contracts and validate the exact object returned by Brain.

Console availability uses one console-owned view envelope rather than a fake
Brain observation:

```text
ConsoleCognitionObservationView
  view_kind = overview_latest | debug_latest | self_latest
  availability = available | not_reported | unavailable | invalid
  reason_code
  generated_at
  observation = CognitionRunObservationV1 | null
```

`available` requires a non-null observation whose `run_kind` matches the view;
every other availability requires `observation = null`. `generated_at` is the
console view-envelope time in UTC and never replaces the Brain observation's
timestamp. `reason_code` is empty for `available` and a bounded lower-snake
code for every unavailable state. Console never reconstructs nodes from
`messages`, `cognition_output`, or other response fields.

`KazusaClient` validates a non-null graph with
`CognitionRunObservationV1.model_validate`. Invalid version/shape/run kind
raises a console-owned `CognitionObservationProtocolError` carrying only the
safe code `observation_contract_invalid`; it never includes raw payload text.
It does not perform the current secondary latest-telemetry fetch after debug
chat and never substitutes one request's observation with another endpoint's
snapshot. Its exact cognition boundaries are:

```python
async def get_latest_cognition_graph(
) -> CognitionRunObservationV1 | None: ...

async def get_latest_self_cognition_graph(
) -> CognitionRunObservationV1 | None: ...

async def send_debug_chat(
    request: ConsoleDebugChatRequest,
) -> KazusaDebugChatResult: ...

@dataclass(frozen=True)
class KazusaDebugChatResult:
    response_payload: dict[str, Any]
    cognition_observation: CognitionRunObservationV1 | None
```

`response_payload` retains the current non-cognition debug response metadata;
the client removes the raw `cognition_graph` key before storing that mapping.
The app creates all three envelopes through one exact factory:

```python
def _cognition_observation_view(
    *,
    view_kind: Literal["overview_latest", "debug_latest", "self_latest"],
    availability: Literal[
        "available", "not_reported", "unavailable", "invalid"
    ],
    reason_code: str,
    observation: CognitionRunObservationV1 | None,
    generated_at: datetime,
) -> ConsoleCognitionObservationView: ...
```

The console API performs one big-bang field rename so callers cannot confuse a
view envelope with a Brain observation:

```text
ConsoleDebugChatResponse.cognition_observation:
    ConsoleCognitionObservationView
ControlConsoleBootstrapResponse.latest_cognition_observation:
    ConsoleCognitionObservationView
ControlConsoleBootstrapResponse.latest_self_cognition_observation:
    ConsoleCognitionObservationView
overview.latest_cognition_observation:
    ConsoleCognitionObservationView JSON
overview.latest_self_cognition_observation:
    ConsoleCognitionObservationView JSON
overview.panels.cognition_observations.items[]:
    {observation_kind: conversation | self_cognition, view: <envelope JSON>}
```

All named view fields are required and never null; absence lives only in the
envelope's availability/null-observation invariant.

The old console API fields `cognition_graph`, `latest_cognition_graph`,
`latest_self_cognition_graph`, and panel `cognition_graphs` are deleted in the
same frontend/backend fixture cutover. Brain retains its own
`ChatResponse.cognition_graph` field because that field now carries the typed
Brain observation directly.

App view mapping is exact:

| Source result | Availability | Reason code |
|---|---|---|
| Valid matching observation | available | empty |
| Brain field is null | not_reported | `brain_not_reported` |
| Latest-view Brain preflight/HTTP unavailable | unavailable | `brain_unavailable` |
| DTO validation or run-kind mismatch | invalid | `observation_contract_invalid` |
| Debug preflight says Brain unavailable, request not attempted | unavailable | `brain_unavailable` |
| Attempted debug request raises `httpx.HTTPError` before a valid observation | unavailable | `debug_request_failed` |

`_load_operational_sources` catches `CognitionObservationProtocolError`
separately from `httpx.HTTPError`: protocol errors create `invalid`; HTTP and
preflight states create `unavailable`; null DTO fields create `not_reported`.
The debug route applies the same explicit catches after its preflight and
always returns a validated `ConsoleDebugChatResponse`. Browser request loading
and thrown browser/API errors remain banners outside the envelope. No error
path includes exception text in `reason_code` or fabricates nodes.

Correlation lookup after the top-level `run_id` deletion is exact:

```python
def _observation_run_id(
    observation: CognitionRunObservationV1 | None,
) -> str | None:
    if observation is None:
        return None
    value = observation.correlation.run_id
    return value if value else None
```

Overview/bootstrap state and both SSE invalidation helpers use this accessor.
Null/empty ids preserve the previous cached id and emit no invalidation;
available observations with a changed non-empty id emit the existing event.
The frontend reads only `view.observation.correlation.run_id` and passes
`view.observation` to the graph renderer when `availability == "available"`.

Update latest context consumption and the character operational posture panel
to consume canonical context sections by section identifier. The cutover uses
the new section shape directly and removes the former nested
`context_consumption` mapping projection.

The exact lookup boundary is:

```python
def _context_section(
    observation: CognitionRunObservationV1,
    section_id: str,
) -> CognitionObservationSectionV1 | None: ...

async def _latest_context_section(
    *,
    states: list[Any],
    kazusa_client: Any,
) -> CognitionObservationSectionV1 | None: ...
```

`_latest_context_section` catches HTTP/protocol errors and returns `None`; it
never unwraps a console view or indexes legacy graph detail.

The character posture repository parameter is renamed from
`latest_context_consumption` to `latest_context_section` and accepts only
`CognitionObservationSectionV1 | None`. It publishes that already-safe section
model dump under `latest_context`; it performs no second semantic allowlist.

### 8. Replace Field-Aware JavaScript With One Generic Section Renderer

Update the shared cognition graph renderer so Overview Latest, Debug
cognition, and Self-cognition Latest resolve `section_refs` from the snapshot
and render:

1. section label and status;
2. section summary;
3. ordered field rows;
4. ordered record cards;
5. displayed/reported counts and a visible omission marker; and
6. the same empty/skipped/failed treatment for every category.

Remove JavaScript cognition field-order lists, domain label lists, arbitrary
object dumping, and first-semantic-value inference. Node cards use the
producer-owned `node.summary`. JavaScript retains UI-owned status wording,
HTML escaping, graph selection, layout, and interaction behavior.

Replace locally fabricated pending/failed debug graph objects with a separate
debug loading/error banner. Only Brain observations enter the graph renderer.
Use common CSS classes for section headers, fields, records, counts, omission
markers, wrapping, scrolling, and every status.

Frontend state names cut over to
`latestCognitionObservationView`, `latestSelfCognitionObservationView`, and
`debugCognitionObservationView`. Bootstrap/overview/debug handlers assign only
the renamed console API fields. One `observationFromView(view, expectedKind)`
helper returns `view.observation` only for `availability="available"` and a
matching run kind; all other envelopes render their availability banner and
never enter graph layout. The Overview panel reads
`panelItems(panels.cognition_observations)[].view`. This is the only envelope-unwrapping
path in JavaScript.

### 8A. Freeze The Existing Portal Layout And Limit UI Work To Card Content

The renderer migration changes cognition-card content, not the surrounding
portal or the established graph-composition algorithm. The pre-plan `HEAD`
versions of `src/control_console/static/index.html`, `console.css`, and the
layout helpers in `console.js` are the visual baseline. The following remain
fixed:

- application shell, sidebar, navigation, top bar, page/content grids, card
  placement, and Debug/Overview/Self panel placement;
- graph DOM ordering: run summary, semantic summary, bounded relationship
  panel, graph body, producer-column stage rail, and inspector;
- producer-column grouping, producer-lane ordering, inter-stage connectors,
  current-node selection, independent per-view pins, and the fitted rail;
- the existing stage-wrapper and node-card visual hierarchy.

The approved UI change radius is inside cognition-card content:

- resolve the Brain-owned `section_refs` and render producer-owned labels,
  statuses, summaries, fields, records, counts, omission markers, and scalar
  lists in the existing inspector/card surfaces;
- use producer-owned node summaries in the existing node and semantic-summary
  cards;
- add only card-local wrapping required by the planned v1 catalog. Branch
  stacks retain natural vertical growth with the baseline `max-height: none`
  and `overflow-y: visible`; card-local vertical caps or internal sliders are
  outside the approved radius. The surrounding portal layout remains fixed.

The following are outside scope: making every node a top-level rail column,
using a `max-content` rail, changing global content-grid selectors, moving or
resizing page cards, replacing producer-column grouping with a new topology,
hiding planned contract nodes, moving the inspector, removing the existing
semantic-summary or bounded-relationship surfaces, or emitting unbounded edge
rows into the page flow.

Before closure, compare the pre-plan `HEAD` source and the candidate source
line by line, then run deterministic DOM assertions for Overview, Debug, and
Self. Screenshot capture is excluded by the user's final verification
instruction. Record the `index.html` diff, exact CSS selector delta, renderer
topology/selection diff, and DOM geometry for cognition cards, graph body,
stage rail, inspector, stage groups, relationship panel, and page overflow.
The candidate retains four live/debug producer-column groups, five self
groups, fitted horizontal geometry, reachable naturally growing branch
stacks, and zero page-level horizontal overflow. Any unexplained source or DOM
delta blocks signoff and archival.

### 9. Update Ownership Manifest And Documentation

Add the new production package paths to
`tests/ownership/source_test_impact_manifest.json`. Update every changed source
row to the exact canonical tests in this plan. Update the subsystem docs with:

- producer/publisher/consumer ownership;
- v1 versioning and cutover rules;
- prewarm disposition meanings;
- context-source parity rules;
- labels and information-density ownership;
- live/self timing boundaries;
- disclosure rules; and
- console browser validation procedure.

## Deferred

- Historical persistence or database storage for cognition observation
  snapshots.
- Streaming intermediate Brain stage updates during an in-flight chat request.
- Adding post-turn background persistence/consolidation nodes to the live v1
  snapshot.
- A new prewarm-specific operational event or event-monitor panel.
- Changes to prewarm eligibility, worker selection, authority mapping, prompt
  content, retry policy, or retrieval semantics beyond the exact structural
  active-character-address projection frozen in Handoff 3P.
- Changes to conversation-progress or group-scene semantic generation.
- Public adapter exposure of the operator observation snapshot.
- A frontend framework migration or page-specific cognition renderer.

## Target State

```text
first-cycle eligibility
    -> SharedMemoryPrewarmOutcomeV1 --------------------+
                                                        |
typed progress + typed group scene + RAG + cognition ---+
                                                        |
actions + surfaces -------------------------------------+
                                                        v
                         Brain-service cognition observation projection
                                                        |
                                   CognitionRunObservationV1
                                  /            |             \
                           ChatResponse    latest endpoint   other ops consumer
                                  \            |             /
                                   console validation only
                                             |
                             one generic graph/section renderer
                              /              |               \
                       Overview Latest      Debug       Self-cognition Latest
```

## Contracts And Data Shapes

### Correlation

The correlation object carries only operator-approved run-level references:

```text
run_id
llm_trace_id
cognition_invocation_id
source_calendar_run_id
```

Each field is optional and bounded to 120 characters. Correlation values stay
top-level and never appear inside semantic records.

### Console View Envelope

`ConsoleCognitionObservationView` is console-owned presentation metadata, not a
cognition schema. It uses `ConfigDict(extra="forbid", frozen=True)` and:

```text
view_kind: overview_latest | debug_latest | self_latest
availability: available | not_reported | unavailable | invalid
reason_code: lower_snake_case string, maximum 80 characters
generated_at: aware UTC datetime, terminal-Z serialization
observation: CognitionRunObservationV1 | null
```

Expected run-kind mapping is exact: Overview and Debug accept `live_turn`;
Self Latest accepts `self_cognition`. `available` requires an
empty reason code plus a matching non-null observation. Every non-available
state requires a non-empty reason code and null observation. Browser loading
and request-error banners remain transient UI state outside this envelope.

### Disclosure

The disclosure object is fixed to the v1 policy key
`approved_cognition_observation.v1` and lists the excluded categories as stable
codes. It contains no raw source values.

### Stable Required Section Identifiers

The v1 producer reserves these identifiers:

```text
input.turn
decision.response
cognition.appraisals
cognition.goal
cognition.response_plan
cognition.affect
reasoning.subjective
reasoning.context_consumption
evidence.memory
evidence.shared_memory_prewarm
context.conversation_progress
context.public_group_scene
action.requests
action.results
action.continuation
surface.visual_directives
surface.visible_messages
self.source
self.route
self.consolidation
```

Producer code may emit only safe projected fields. Future additive section
identifiers follow `^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$` and use the existing
generic section model. A breaking semantic reinterpretation requires a new
major observation schema and an approved cutover plan.

## Runtime Or Resource Constraints

- Additional LLM calls: zero.
- Prompt/model/token-budget changes: zero.
- Database reads/writes/migrations: zero.
- Network calls: zero beyond existing Brain-console and existing prewarm calls.
- Prewarm remains parallel with identity preparation on eligible cycle zero.
- Added runtime work is monotonic timing, strict validation, bounded safe
  projection, serialization, and rendering.
- Maximum serialized v1 observation: 131,072 characters.
- Current latest-run storage remains process-local and non-historical.
- The persisted `cognition_chain_run.v2` discussed in the independent draft
  hybrid-loop architecture is a separate future engine-performance artifact.
  This plan follows the current Brain-service README and keeps the semantic
  operator snapshot process-local. The architecture draft is updated to make
  that distinction explicit.
- Deterministic and patched tests provide the implementation gate; live LLM
  execution is outside this observability-only scope.

## Cutover Policy

- Strategy: atomic big-bang internal cutover.
- Producer: old service-owned `detail: dict` graph builders and helper
  allowlists are removed when the new producer lands.
- Brain wire: raw graph dictionaries are replaced by the strict v1 model.
- Console transport: projection and inference are replaced by direct v1
  validation.
- Frontend: hardcoded semantic field vocabulary is replaced by section-driven
  rendering in the same patch.
- Prewarm: RAG-only return and merge signatures are replaced by the typed
  outcome signatures in every caller and test.
- Context: old nested context-consumption graph detail is replaced by canonical
  sections in every consumer.
- Compatibility: the repository carries one active vocabulary after cutover;
  no aliases, dual schema, translation bridge, or legacy fallback remains.
- Rollback: revert the coordinated implementation commit/patch and its docs;
  no data migration needs reversal.

## Execution Roles

### Role 1: Architect And Change Owner

- Responsibility: own the ICD, settle contract decisions, sequence execution,
  inspect the diff/evidence, direct remediation, and sign off or reject the
  result.
- Owned surface: this plan, architecture decisions, approval state, execution
  handoffs, review findings, and lifecycle evidence.
- Authority: read-only source review plus plan/document lifecycle edits. The
  role may stop execution on scope or contract divergence.
- Applicable skills: `development-plan`, `local-llm-architecture`, and
  `control-console-web-development`.
- Capability floor: system architecture, strict API design, cognition ownership
  analysis, security review, and frontend contract review.
- Independence requirement: this role performs no production-code or test
  implementation and therefore remains independent of Role 2 output.
- Acceptance output: approved plan, bounded handoff slices, review findings,
  and final sign-off evidence.
- Gate: user approval precedes Role 2 production execution; all acceptance
  criteria and review findings are resolved before closure.

### Role 2: Fixed Implementation And Verification Executor

- Responsibility: implement the approved plan, edit production/tests/docs
  within assigned slices, run all code-level checks and browser checks, inspect
  results, and remediate architect findings.
- Owned surface: every path in Change Surface except plan approval/sign-off
  decisions.
- Authority: production, test, documentation, manifest, and artifact edits
  within the approved plan only.
- Applicable skills: every Mandatory Skill listed above.
- Capability floor: cross-layer Python/FastAPI/vanilla-JS implementation,
  strict contract migration, deterministic testing, Playwright/browser QA, and
  safe worktree handling.
- Fixed executor: existing persistent
  `/root/cognition_console_implementer`, `gpt-5.6-luna`, reasoning `max`,
  standard execution speed.
- Independence requirement: this agent did not author the architecture or
  approve the production scope. It may review the draft for implementability,
  while Role 1 retains architecture and sign-off authority.
- Acceptance output: scoped diff, exact test results, manifest result, browser
  screenshots/DOM/console evidence, documentation updates, and a concise
  deviation report.
- Gate: exact fixed binding, clean captured baseline, approved plan, mandatory
  skills loaded, and explicit owned-file slice before each handoff.

The user's 2026-08-27 instruction superseded the fixed binding for the final
frontend correction and closure slice. Role 1 therefore performed that slice
directly and records the loss of executor/reviewer independence as an explicit
execution deviation rather than claiming an independent final edit review.

## Test Impact And Traceability

Every listed production source or governed contract artifact has an exact
deterministic owner test. New node IDs below are required implementation output
and must collect successfully before the first passing execution checkpoint.

| Path | Changed symbol/interface | Semantic owner | Exact deterministic pytest node(s) | Supplemental node(s) | Mode | Observable regression prevented |
|---|---|---|---|---|---|---|
| `src/kazusa_ai_chatbot/brain_service/__init__.py` | public observation DTO exports | Brain-service wire owner | `tests/unit/cognition_observability/test_contracts.py::test_brain_service_exports_only_canonical_observation_wire_models` | none | deterministic unit | console imports projection/runtime internals |
| `src/kazusa_ai_chatbot/brain_service/cognition_observation_contracts.py` | all `*ObservationV1` DTOs, validators, canonical serialization | Brain-service wire owner | `tests/unit/cognition_observability/test_contracts.py::test_observation_contract_rejects_unknown_fields_invalid_references_and_over_budget_payloads`; `tests/unit/cognition_observability/test_contracts.py::test_observation_contract_enforces_truthful_record_counts_statuses_and_utc_serialization` | none | deterministic unit | loose detail, invalid refs, timestamp replacement, fake unavailable observation |
| `src/kazusa_ai_chatbot/brain_service/cognition_observation_projection.py` | catalog, closed source mapping, `build_live_cognition_observation`, `build_self_cognition_observation` | Brain-service projection owner | `tests/unit/cognition_observability/test_projection.py::test_live_projection_reports_all_shared_memory_prewarm_dispositions`; `tests/unit/cognition_observability/test_projection.py::test_context_sources_share_one_detail_shape_and_budget`; `tests/unit/cognition_observability/test_projection.py::test_live_and_self_projections_share_exact_section_catalog`; `tests/unit/cognition_observability/test_projection.py::test_projection_uses_closed_source_field_mapping_and_invalid_row_counts`; `tests/unit/cognition_observability/test_projection.py::test_projection_excludes_protected_and_operational_fields`; `tests/unit/cognition_observability/test_projection.py::test_projection_emits_only_canonical_sequence_and_reference_edges` | `tests/control_console_e2e/test_cognition_observability_e2e.py::test_live_debug_and_self_views_share_observation_section_layout` | deterministic unit + browser supplemental | producer drift, lost prewarm, source-shape inference, unequal detail, unsafe disclosure, edge loss |
| `src/kazusa_ai_chatbot/cognition_resolver/contracts.py` | `SharedMemoryPrewarmOutcomeV1`, `normalize_projected_rag_result`, validators | cognition-resolver contract owner | `tests/test_shared_memory_prewarm.py::test_shared_memory_prewarm_outcome_validator_rejects_invalid_disposition_and_counts`; `tests/test_shared_memory_prewarm.py::test_shared_memory_prewarm_outcome_enforces_exact_types_bounds_and_rag_shape`; `tests/unit/cognition_resolver/test_capabilities.py::test_capabilities_exposes_owned_contract`; `tests/unit/cognition_resolver/test_capabilities.py::test_resolver_observation_evidence_has_typed_authority` | none | deterministic unit | invalid type/status/reason/count/RAG combinations or changed evidence authority |
| `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py` | `run_first_cycle_shared_memory_prewarm`, `merge_shared_memory_prewarm_outcome` | cognition-resolver capability owner | `tests/test_shared_memory_prewarm.py::test_first_cycle_prewarm_returns_explicit_outcome_dispositions`; `tests/test_shared_memory_prewarm.py::test_merge_shared_memory_prewarm_outcome_counts_projected_entries_and_rejects_repeat_or_invalid_base`; `tests/test_shared_memory_prewarm.py::test_first_cycle_prewarm_propagates_cancellation_without_fabricating_terminal_outcome`; `tests/test_shared_memory_prewarm.py::test_napcat_character_mention_recalls_and_merges_corresponding_shared_memory` | `tests/test_local_context_resolver_integration.py::test_first_cycle_prewarm_uses_memory_worker_without_full_resolver`; `tests/test_shared_memory_prewarm_live_llm.py::test_production_napcat_command_recalls_seeded_shared_memory` | deterministic unit + patched integration + isolated live LLM/DB | empty/failure collapse, grouped-row miscount, repeated merge, unsafe base, swallowed cancellation, or loss of the exact `#napcat` shared-memory anchor |
| `src/kazusa_ai_chatbot/cognition_shared/model_attempt_policy.py` | request-scoped prewarm checkpoint record/snapshot/graph-attempt clearing | cognition diagnostic-carrier owner | `tests/unit/cognition_shared/test_observation_checkpoint.py::test_prewarm_checkpoint_is_deep_copied_scoped_to_graph_attempt_and_cleared` | `tests/unit/brain_service/test_cognition_graph_projection.py::test_failed_run_uses_current_attempt_prewarm_checkpoint` | deterministic unit | completed prewarm disappears when a later graph stage fails or leaks across retries |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` | `shared_memory_prewarm_outcome`, typed group scene, projection status/reason | persona-state contract owner | `tests/unit/nodes/test_persona_supervisor2_schema.py::test_persona_supervisor2_schema_exposes_owned_contract` | none | deterministic unit | untyped carrier or inability to distinguish non-group from failed projection |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` | structured group-scene retention and explicit projection discriminator | persona orchestration owner | `tests/test_conversation_progress_group_scene.py::test_persona_supervisor_reports_group_scene_success_non_group_and_failure` | `tests/test_conversation_progress_v2_service.py::test_service_load_passes_group_anchor_mode_and_keeps_user_scope` | deterministic unit/integration | prompt rendering destroys inspectable scene, failed projection looks non-group, or user scope changes |
| `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` | eligibility outcome creation and post-merge carrier | cognition execution owner | `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_cognition_state_preserves_shared_memory_prewarm_outcome_after_merge`; `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_cognition_records_noneligible_prewarm_without_starting_worker`; `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_cognition_cancellation_publishes_no_prewarm_outcome_or_observation` | none | deterministic unit | non-eligibility disappears, merge status is lost, or cancellation fabricates a terminal report |
| `src/kazusa_ai_chatbot/brain_service/contracts.py` | typed `ChatResponse` and `OpsLatestCognitionGraphResponse` fields | Brain-service response owner | `tests/unit/brain_service/test_cognition_graph_projection.py::test_brain_response_contract_uses_canonical_cognition_observation`; `tests/test_service_ops_status.py::test_ops_runtime_status_merges_config_and_worker_liveness` | none | deterministic unit | raw dict returns or unrelated ops status regression |
| `src/kazusa_ai_chatbot/service.py` | graph builders replaced by projection entrypoints; terminal-status mapping; current-attempt failure checkpoint; typed latest recorders | Brain runtime owner | `tests/unit/brain_service/test_cognition_graph_projection.py::test_service_publishes_canonical_observation_without_legacy_graph_helpers`; `tests/unit/brain_service/test_cognition_graph_projection.py::test_live_terminal_status_mapping_and_cancellation_are_exact`; `tests/unit/brain_service/test_cognition_graph_projection.py::test_failed_run_uses_current_attempt_prewarm_checkpoint`; `tests/unit/brain_service/test_cognition_graph_projection.py::test_legacy_cognition_graph_projection_symbols_are_absent_from_production`; `tests/test_service_background_consolidation.py::test_chat_response_tracks_deliverable_assistant_row`; `tests/test_conversation_progress_v2_service.py::test_service_load_passes_group_anchor_mode_and_keeps_user_scope` | `tests/test_self_cognition_integration.py::test_prepared_commitment_state_contains_public_group_scene`; `tests/test_real_history_personality_fixture_contract.py::test_private_monologue_uses_only_canonical_reasoning_node`; `tests/test_real_history_personality_fixture_contract.py::test_private_monologue_fails_closed_when_canonical_node_is_missing` | deterministic unit/integration | old builder remains, terminal state is guessed, failed run loses prewarm, response/latest diverge, or old node consumers survive |
| `src/control_console/contracts.py` | `ConsoleCognitionObservationView`; renamed bootstrap/debug view fields | console API contract owner | `tests/test_control_console_contracts.py::test_service_contracts_reject_extra_fields_and_unbounded_strings`; `tests/test_control_console_contracts.py::test_console_response_contract_uses_view_envelopes_for_bootstrap_and_debug` | none | deterministic unit | raw observation is confused with view metadata or availability/view pairing is invalid |
| `src/control_console/kazusa_client.py` | validation-only reads; typed debug result; legacy projectors/secondary fetch removed | console Brain-client owner | `tests/test_control_console_kazusa_client.py::test_client_validates_canonical_cognition_observation_without_reprojection`; `tests/test_control_console_kazusa_client.py::test_client_raises_protocol_error_for_invalid_observation_version`; `tests/test_control_console_kazusa_client.py::test_client_rejects_invalid_latest_observation_without_reconstruction`; `tests/test_control_console_kazusa_client.py::test_debug_client_returns_direct_response_observation_without_latest_fetch`; `tests/test_control_console_kazusa_client.py::test_kazusa_client_reads_health_and_posts_debug_chat`; `tests/unit/brain_service/test_cognition_graph_projection.py::test_legacy_cognition_graph_projection_symbols_are_absent_from_production` | none | deterministic unit | inference fallback, cross-request telemetry substitution, timestamp mutation, or old fixture acceptance |
| `src/control_console/app.py` | exact availability conversion, renamed envelope fields, nested correlation accessor, `_latest_context_section` | console web API owner | `tests/test_control_console_bootstrap.py::test_bootstrap_wraps_canonical_observations_with_view_metadata`; `tests/test_control_console_web_surface.py::test_debug_api_wraps_canonical_observation_without_reprojection`; `tests/test_control_console_web_surface.py::test_latest_and_debug_protocol_errors_map_to_exact_view_availability`; `tests/test_control_console_stream.py::test_stream_uses_nested_observation_run_id_and_ignores_empty_id`; `tests/test_control_console_web_surface.py::test_web_api_outputs_for_logs_events_audit_character_and_debug_error` | `tests/control_console_e2e/test_debug_chat_e2e.py::test_debug_chat_sends_to_brain_and_updates_history_and_graph` | deterministic API + browser supplemental | view/source ambiguity, protocol errors become availability errors, unavailable fake graph, stale SSE path, or context reconstruction |
| `src/control_console/redaction.py` | legacy cognition-context redactors removed | console disclosure owner | `tests/test_control_console_redaction.py::test_canonical_observation_sections_bypass_legacy_semantic_reprojection`; `tests/test_control_console_redaction.py::test_responses_exclude_secrets_prompts_embeddings_env_values_and_raw_messages` | none | deterministic unit | second semantic allowlist survives or generic redaction weakens |
| `src/control_console/repository.py` | `_project_character_operational_posture` accepts canonical context section | console entity-projection owner | `tests/test_control_console_repository.py::test_character_posture_consumes_canonical_context_observation_section`; `tests/test_control_console_repository.py::test_repository_operational_panels_accept_console_utc_offset` | none | deterministic unit | entity view expects deleted nested context mapping |
| `src/control_console/static/console.js` | section resolver/renderer; pending/error banners; legacy field lists removed | console graph renderer owner | `tests/test_control_console_web_surface.py::test_cognition_observation_renderer_uses_contract_labels_and_shared_detail_layout`; `tests/test_control_console_web_surface.py::test_renderer_accepts_unknown_producer_section_without_js_catalog`; `tests/test_control_console_web_surface.py::test_debug_loading_and_error_states_are_separate_from_cognition_graph`; `tests/unit/brain_service/test_cognition_graph_projection.py::test_legacy_cognition_graph_projection_symbols_are_absent_from_production` | `tests/control_console_e2e/test_cognition_observability_e2e.py::test_live_debug_and_self_views_share_observation_section_layout`; `tests/control_console_e2e/test_debug_chat_e2e.py::test_debug_chat_click_shows_live_running_state_before_response` | deterministic static + browser supplemental | hardcoded domain vocabulary, fake graphs, cross-view drift |
| `src/control_console/static/console.css` | shared section/record/count/status layout | console visual-system owner | `tests/test_control_console_web_surface.py::test_cognition_observation_renderer_uses_contract_labels_and_shared_detail_layout` | `tests/control_console_e2e/test_cognition_observability_e2e.py::test_prewarm_and_context_sources_render_status_counts_and_omissions` | deterministic static + browser supplemental | uneven density, overflow, hidden omission/status |
| `tests/ownership/source_test_impact_manifest.json` | new/changed owner rows and exact node ids | verification-governance owner | `tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary`; `tests/test_test_impact_manifest.py::test_stale_required_node_fails_closed` | none | deterministic governance | unmapped source or stale exact test node |
| `docs/architecture/cognition_observability_icd.md` | authoritative v1 ICD | architecture owner | `tests/test_cognition_observability_docs.py::test_icd_and_runtime_docs_name_one_brain_service_contract_owner` | none | deterministic docs | fourth schema owner or undocumented wire rules |
| `docs/architecture/cognition_contracts_design.md` | external observability contract registration | architecture owner | `tests/test_cognition_observability_docs.py::test_icd_and_runtime_docs_name_one_brain_service_contract_owner` | none | deterministic docs | cognition contract registry omits external interface |
| `docs/architecture/cognition_v3_hybrid_agentic_loop_architecture.md` | process-local semantic snapshot vs future persisted chain-run distinction | architecture owner | `tests/test_cognition_observability_docs.py::test_process_local_observation_and_future_persisted_chain_run_are_distinct` | none | deterministic docs | contradictory storage ownership |
| `docs/HOWTO.md` | operator verification procedure | operations-doc owner | `tests/test_cognition_observability_docs.py::test_howto_documents_canonical_observation_and_browser_checks` | none | deterministic docs | obsolete graph commands or missing validation steps |
| `src/kazusa_ai_chatbot/brain_service/README.md` | response/latest observation contract | Brain-service documentation owner | `tests/test_cognition_observability_docs.py::test_icd_and_runtime_docs_name_one_brain_service_contract_owner` | none | deterministic docs | README retains raw graph dictionary contract |
| `src/kazusa_ai_chatbot/cognition_resolver/README.md` | prewarm outcome/count/merge semantics | cognition-resolver documentation owner | `tests/test_cognition_observability_docs.py::test_runtime_readmes_document_prewarm_and_observation_carriers` | none | deterministic docs | resolver README retains RAG-only prewarm behavior |
| `src/kazusa_ai_chatbot/nodes/README.md` | persona-state outcome and group-scene discriminator | node-contract documentation owner | `tests/test_cognition_observability_docs.py::test_runtime_readmes_document_prewarm_and_observation_carriers` | none | deterministic docs | node README omits new state carriers |
| `src/control_console/README.md` | validation-only transport and shared renderer | console documentation owner | `tests/test_cognition_observability_docs.py::test_icd_and_runtime_docs_name_one_brain_service_contract_owner` | none | deterministic docs | console described as schema/projector owner |

Cross-boundary required tests:

- `tests/test_shared_memory_prewarm.py::test_first_cycle_prewarm_propagates_cancellation_without_fabricating_terminal_outcome`
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py::test_cognition_cancellation_publishes_no_prewarm_outcome_or_observation`
- `tests/unit/cognition_shared/test_observation_checkpoint.py::test_prewarm_checkpoint_is_deep_copied_scoped_to_graph_attempt_and_cleared`
- `tests/unit/brain_service/test_cognition_graph_projection.py::test_live_terminal_status_mapping_and_cancellation_are_exact`
- `tests/unit/brain_service/test_cognition_graph_projection.py::test_failed_run_uses_current_attempt_prewarm_checkpoint`
- `tests/test_local_context_resolver_integration.py::test_first_cycle_prewarm_uses_memory_worker_without_full_resolver`
- `tests/test_shared_memory_prewarm.py::test_napcat_character_mention_recalls_and_merges_corresponding_shared_memory`
- `tests/test_shared_memory_prewarm_live_llm.py::test_production_napcat_command_recalls_seeded_shared_memory`
- `tests/test_control_console_bootstrap.py::test_bootstrap_returns_initial_state_session_csrf_services_and_stream_url`
- `tests/test_control_console_stream.py::test_stream_poll_appends_graph_invalidation_for_new_latest_run`
- `tests/control_console_e2e/test_debug_chat_e2e.py::test_debug_chat_sends_to_brain_and_updates_history_and_graph`
- `tests/control_console_e2e/test_debug_chat_e2e.py::test_debug_chat_click_shows_live_running_state_before_response`
- `tests/control_console_e2e/test_cognition_observability_e2e.py::test_prewarm_and_context_sources_render_status_counts_and_omissions`
- `tests/control_console_e2e/test_cognition_observability_e2e.py::test_canonical_sequence_and_reference_edges_render`
- `tests/test_control_console_web_surface.py::test_latest_and_debug_protocol_errors_map_to_exact_view_availability`
- `tests/test_control_console_stream.py::test_stream_uses_nested_observation_run_id_and_ignores_empty_id`
- `tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary`

Legacy graph consumers cut over in the same patch; no test helper may retain
old node ids, nested `detail`, or raw-dictionary indexing:

| Consumer path | Exact consumer node/helper | Required v1 replacement | Verification mode |
|---|---|---|---|
| `tests/test_self_cognition_integration.py` | `test_prepared_commitment_state_contains_public_group_scene` | call `build_self_cognition_observation` through the publisher seam and assert the required group-scene section is honestly `not_reported` when the self artifact lacks the typed source | deterministic integration |
| `tests/control_console_e2e/test_stage3_fresh_database_e2e.py` | `test_stage3_fresh_database_graph_and_debug_handoff` | replace `l2.*`, `fork`, `join`, and nested detail fixture values with the exact v1 sections/nodes and `sequence | reference` edges | deterministic fake-service E2E |
| `tests/test_real_history_personality_fixture_contract.py` | `test_private_monologue_uses_only_canonical_reasoning_node`; `test_private_monologue_fails_closed_when_canonical_node_is_missing` | read `reasoning.subjective.private_monologue` through node `reasoning.context` and its section reference | deterministic fixture contract |
| `tests/test_real_history_personality_e2e_live_llm.py` | `_extract_private_monologue`; `test_live_real_history_personality_case`; `test_live_real_history_non_active_target_routing_guard` | validate the v1 observation and resolve the canonical reasoning section; preserve fail-closed behavior | live-LLM supplemental, collect/compile in this plan |
| `tests/test_short_horizon_state_composition_live_llm.py` | `_graph_node`; `_context_consumption`; `test_offence_emotion_specific_counterfactual`; `test_elapsed_global_affect_counterfactual`; `test_global_warmth_counterfactual`; `test_relationship_cause_counterfactual`; `test_style_scope_counterfactual` | resolve `reasoning.context_consumption` through section references and replace `l1.relevance`, `l2.reasoning`, `v2.appraisal`, and `v2.collapse` expectations with the exact live v1 node catalog | live-LLM supplemental, collect/compile in this plan |
| `tests/test_short_horizon_state_composition_e2e_live_llm.py` | `_graph_node`; `_context_consumption`; `test_offence_by_user_a_changes_next_user_b_turn`; `test_offence_global_affect_fades_before_sleep`; `test_apology_repairs_user_a_and_global_carryover`; `test_private_event_changes_next_group_turn`; `test_group_event_changes_next_private_turn`; `test_accepted_task_result_changes_next_turn` | use the same v1 section lookup and assert canonical correlation/node/section values at Brain and console boundaries | live-LLM supplemental, collect/compile in this plan |

Manifest replacements are exact: the `kazusa_client.py` row replaces
`test_graph_projection_preserves_semantic_cognition_rows` with
`test_client_validates_canonical_cognition_observation_without_reprojection`;
the `service.py` row replaces
`test_cognition_graph_projects_subjective_semantics_without_stage_topology`
with `test_service_publishes_canonical_observation_without_legacy_graph_helpers`;
the `console.js` and `console.css` rows replace
`test_static_shell_favicon_and_generic_lookup_outputs` as their cognition
owner with `test_cognition_observation_renderer_uses_contract_labels_and_shared_detail_layout`.
The cutover also creates previously missing manifest entries for
`src/control_console/redaction.py` owned by
`test_canonical_observation_sections_bypass_legacy_semantic_reprojection` and
`src/control_console/repository.py` owned by
`test_character_posture_consumes_canonical_context_observation_section`.
It adds entries for both new Brain-service observation modules and for
`src/kazusa_ai_chatbot/cognition_shared/model_attempt_policy.py` with the
checkpoint test named below. The manifest change is part of the atomic patch;
no legacy required node remains after it.

## Change Surface

### Create

- `docs/architecture/cognition_observability_icd.md`
- `src/kazusa_ai_chatbot/brain_service/cognition_observation_contracts.py`
- `src/kazusa_ai_chatbot/brain_service/cognition_observation_projection.py`
- `tests/unit/cognition_observability/__init__.py`
- `tests/unit/cognition_observability/test_contracts.py`
- `tests/unit/cognition_observability/test_projection.py`
- `tests/unit/cognition_shared/test_observation_checkpoint.py`
- `tests/control_console_e2e/test_cognition_observability_e2e.py`
- `tests/test_cognition_observability_docs.py`
- `tests/test_shared_memory_prewarm_live_llm.py`
- `development_plans/archive/completed/short_term/cognition_observability_icd_and_console_consistency_plan_2026-08-26.md`

### Modify

- `docs/architecture/cognition_contracts_design.md`
- `docs/architecture/cognition_v3_hybrid_agentic_loop_architecture.md`
- `docs/HOWTO.md`
- `src/kazusa_ai_chatbot/brain_service/__init__.py`
- `src/kazusa_ai_chatbot/brain_service/README.md`
- `src/kazusa_ai_chatbot/cognition_resolver/contracts.py`
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`
- `src/kazusa_ai_chatbot/cognition_resolver/README.md`
- `src/kazusa_ai_chatbot/cognition_shared/model_attempt_policy.py`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/brain_service/contracts.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/control_console/contracts.py`
- `src/control_console/kazusa_client.py`
- `src/control_console/app.py`
- `src/control_console/redaction.py`
- `src/control_console/repository.py`
- `src/control_console/static/console.js`
- `src/control_console/static/console.css`
- `src/control_console/README.md`
- `tests/test_shared_memory_prewarm.py`
- `tests/test_local_context_resolver_integration.py`
- `tests/unit/nodes/test_persona_supervisor2_schema.py`
- `tests/unit/nodes/test_persona_supervisor2_cognition_commit.py`
- `tests/test_conversation_progress_group_scene.py`
- `tests/unit/brain_service/test_cognition_graph_projection.py`
- `tests/test_service_background_consolidation.py`
- `tests/test_control_console_contracts.py`
- `tests/test_control_console_kazusa_client.py`
- `tests/test_control_console_bootstrap.py`
- `tests/test_control_console_stream.py`
- `tests/test_control_console_repository.py`
- `tests/test_control_console_redaction.py`
- `tests/test_control_console_web_surface.py`
- `tests/test_console_debug_chat.py`
- `tests/test_console_lookup_limits.py`
- `tests/control_console_e2e/fake_brain.py`
- `tests/control_console_e2e/test_debug_chat_e2e.py`
- `tests/control_console_e2e/test_stage3_fresh_database_e2e.py`
- `tests/test_self_cognition_integration.py`
- `tests/test_real_history_personality_fixture_contract.py`
- `tests/test_real_history_personality_e2e_live_llm.py`
- `tests/test_short_horizon_state_composition_live_llm.py`
- `tests/test_short_horizon_state_composition_e2e_live_llm.py`
- `tests/unit/cognition_resolver/test_capabilities.py`
- `tests/ownership/source_test_impact_manifest.json`
- `development_plans/README.md`
- `development_plans/active/short_term/cognition_observability_icd_and_console_consistency_plan_2026-08-26.md`

### Delete

- `control_console.contracts.CognitionRunGraphNode`,
  `CognitionRunGraphEdge`, `CognitionRunGraphSnapshot`, and
  `CognitionContextConsumption`.
- Console API fields `ConsoleDebugChatResponse.cognition_graph`,
  `ControlConsoleBootstrapResponse.latest_cognition_graph`, and
  `latest_self_cognition_graph`, plus overview keys/panel
  `latest_cognition_graph`, `latest_self_cognition_graph`, and
  `cognition_graphs`; replace them with the exact observation-envelope names
  in section 7.
- `control_console.kazusa_client.project_cognition_graph_snapshot`,
  `not_reported_cognition_graph`, `_project_known_cognition_fields`,
  `_project_graph_nodes`, `_project_graph_edges`, `_project_node_detail`, every
  `COGNITION_GRAPH_*` semantic allowlist, and their recursive detail helpers.
- `control_console.redaction.redact_context_consumption` and
  `redact_latest_context_consumption`; the generic non-cognition redaction
  boundary remains.
- `control_console.app._context_consumption_from_graph`; replace
  `_latest_context_consumption` with exact canonical-section lookup named
  `_latest_context_section`.
- `service._build_response_cognition_graph`,
  `_build_self_cognition_cognition_graph`, `_graph_memory_detail`,
  `_graph_context_consumption`, `_graph_cognition_nodes`, every
  graph-specific `_GRAPH_*` allowlist, and the frozen legacy-projection symbol
  set below after the new Brain-service projection owns the behavior.
- Raw-dictionary latest snapshot annotations in
  `service._latest_cognition_graph`, `_latest_self_cognition_graph`,
  `_record_latest_cognition_graph`, and
  `_record_latest_self_cognition_graph`; retain these symbols with typed model
  annotations and Pydantic deep-copy behavior.
- `cognition_resolver.capabilities.merge_shared_memory_prewarm_result`; replace
  it with `merge_shared_memory_prewarm_outcome` in the same cutover.
- `cognition_resolver.contracts._normalize_rag_result`; rename it to the one
  public internal boundary `normalize_projected_rag_result` and update its
  resolver/prewarm callers atomically.
- JavaScript `cognitionGraphInspectorRows`, its domain field/label arrays,
  `cognitionGraphFirstSemanticValue`, arbitrary nested-object dumping,
  `pendingDebugCognitionGraph`, and `failedDebugCognitionGraph`.

`test_legacy_cognition_graph_projection_symbols_are_absent_from_production`
is the repo-wide stale-symbol gate. It parses Python modules and reads the JS
source, failing when any frozen symbol below remains. The unrelated runtime
node `service._graph_relevance_node` is explicitly outside this legacy set and
remains.

```text
service.py functions:
  _build_response_cognition_graph
  _build_self_cognition_cognition_graph
  _graph_status
  _safe_graph_text
  _graph_full_text
  _graph_key_is_forbidden
  _graph_project_nested
  _graph_project_mapping
  _graph_project_semantic_rows
  _graph_project_text_list
  _graph_cognition_failure_node
  _graph_canonical_cognition_nodes
  _graph_cognition_nodes
  _graph_messages
  _graph_intake_detail
  _graph_reasoning_detail
  _graph_context_consumption
  _graph_settled_relevance_consumption
  _graph_cognition_context_consumption
  _graph_surface_context_consumption
  _graph_context_consumption_health
  _graph_context_consumption_status
  _graph_public_character_operational_context
  _graph_public_relationship_context
  _graph_public_style_projection
  _graph_public_group_engagement_context
  _graph_public_predecessor
  _graph_public_attempts
  _graph_public_operational_receipt
  _graph_public_rows
  _graph_public_text_fields
  _graph_public_text_list
  _graph_public_text
  _graph_public_code
  _graph_memory_detail
  _graph_action_specs
  _graph_action_results
  _graph_action_continuation
  _graph_visual_enabled
  _graph_visual_raw
  _graph_visual_node
  _graph_self_source_detail
  _graph_self_surface_messages
  _graph_consolidation_detail

service.py constants:
  _GRAPH_VISUAL_DIRECTIVE_FIELDS
  _GRAPH_EVIDENCE_FIELDS
  _GRAPH_PROGRESS_FIELDS
  _GRAPH_REPLY_FIELDS
  _GRAPH_CONTINUATION_FIELDS
  _GRAPH_ACTION_FIELDS
  _GRAPH_FUTURE_FIELDS
  _GRAPH_CONSOLIDATION_FIELDS
  _GRAPH_FORBIDDEN_KEY_PARTS
  _GRAPH_FORBIDDEN_KEYS

kazusa_client.py functions/constants:
  not_reported_cognition_graph
  project_cognition_graph_snapshot
  _first_graph_payload
  _project_graph_nodes
  _project_graph_edges
  _project_node_detail
  _cognition_graph_key_is_forbidden
  _project_cognition_graph_detail_value
  _project_context_consumption
  _project_cognition_graph_scalar
  _project_cognition_graph_text_list
  _project_cognition_graph_mapping
  _project_cognition_graph_rows
  _project_cognition_graph_nested
  _project_cognition_graph_message_fragments
  _project_known_cognition_fields
  _has_any
  COGNITION_GRAPH_DETAIL_KEYS
  COGNITION_GRAPH_SCALAR_DETAIL_KEYS
  COGNITION_GRAPH_TEXT_LIST_DETAIL_KEYS
  COGNITION_GRAPH_MAPPING_DETAIL_KEYS
  COGNITION_GRAPH_ROW_DETAIL_KEYS
  COGNITION_GRAPH_NESTED_DETAIL_KEYS
  COGNITION_GRAPH_FORBIDDEN_DETAIL_KEYS
  COGNITION_GRAPH_FORBIDDEN_DETAIL_PARTS
  COGNITION_GRAPH_RAW_KEYS

console.js functions:
  cognitionGraphInspectorRows
  cognitionGraphNodeSummary
  cognitionGraphValue
  cognitionGraphValuePresent
  cognitionGraphFirstSemanticValue
  pendingDebugCognitionGraph
  failedDebugCognitionGraph

other deleted Python symbols:
  control_console.contracts.CognitionRunGraphNode
  control_console.contracts.CognitionRunGraphEdge
  control_console.contracts.CognitionRunGraphSnapshot
  control_console.contracts.CognitionContextConsumption
  control_console.redaction.redact_context_consumption
  control_console.redaction.redact_latest_context_consumption
  control_console.app._context_consumption_from_graph
  control_console.app._latest_context_consumption
  cognition_resolver.capabilities.merge_shared_memory_prewarm_result
  cognition_resolver.contracts._normalize_rag_result

control-console legacy API accesses/fields:
  ConsoleDebugChatResponse.cognition_graph
  ControlConsoleBootstrapResponse.latest_cognition_graph
  ControlConsoleBootstrapResponse.latest_self_cognition_graph
  payload.latest_cognition_graph
  payload.latest_self_cognition_graph
  overview.latest_cognition_graph
  overview.latest_self_cognition_graph
  result.cognition_graph
  panels.cognition_graphs
```

### Keep

- Existing graph containers, selection/pinning behavior, SSE invalidation,
  status-driven node styling, responsive graph layout, HTML escaping, and
  operator authentication.
- Existing `src/control_console/static/index.html`; the current graph
  containers and `#ui-notice` loading/error region are sufficient for v1.
- Existing prewarm eligibility, direct persistent-memory worker, safe shared
  row filtering, cognition evidence authority, and one prewarm-only sanitized
  copy of the current-message context.
- Existing conversation-progress and group-scene semantic source contracts.
- Existing process-local latest-run storage behavior.
- `tests/test_rag3_media_debug_adapter_e2e_live_llm.py` and
  `tests/test_qq_group_public_scene_live_llm.py` are unchanged storage-only
  consumers: they capture the response graph value as an opaque artifact and
  do not index legacy nodes, edges, top-level correlation, or nested detail.
  The changed live-LLM collect gate confirms they still import after the typed
  Brain field cutover.

## Agent Autonomy Boundaries

Role 2 may choose private helper names, local function decomposition, fixture
text, CSS class names, and command grouping while preserving the contracts and
exact test nodes above.

The following decisions remain fixed:

- package and schema names;
- v1 status vocabulary and meanings;
- field/record-only detail model;
- budgets and truthful count rules;
- stable required section identifiers;
- prewarm outcome fields, statuses, and reason mapping;
- conversation-progress/group-scene peer section contract;
- producer-owned labels and ordering;
- validation-only console transport;
- one shared frontend renderer;
- live v1 timing boundary;
- protected-field exclusions;
- big-bang cutover; and
- fixed Luna executor binding.

Any discovered need for prompt/model changes, a second schema, persistence,
new event telemetry, changed retrieval semantics, increased budgets, or a
different executor pauses that slice for a user-approved amendment.

## Implementation Order

1. Obtain user approval, promote the plan to `approved`, capture worktree and
   owned-path baseline, and hand execution to the persistent Luna executor.
2. Complete the entire planned test surface before further production work:
   every deterministic unit/integration/static test, fake-service and browser
   E2E test, fixture, live-LLM compile/collection consumer, documentation test,
   and exact ownership-manifest row in Change Surface.
3. Collect every exact node in Test Impact And Traceability, then run the
   deterministic test surface once to record the comprehensive expected
   failing/passing baseline against the incomplete production cutover. Test
   failures at this checkpoint are implementation evidence, not invitations to
   edit production piecemeal.
4. Implement the remaining production cutover across resolver carriers,
   cognition-observability contracts/projection, Brain publication, console
   validation transport, repository/API consumption, and the shared frontend
   renderer while preserving the now-fixed test contract.
5. Update the ICD, runtime READMEs, HOWTO, registry, and execution evidence
   after production implementation matches the completed test surface.
6. Run all exact mapped tests, focused batches, broader regressions, manifest
   validation, static checks, live-LLM compile/collection gates, and browser
   checks.
7. Route the final diff and evidence to the architect for independent code
   review. Role 2 resolves findings and reruns affected gates; the architect
   re-reviews and signs off.

## Verification

Role 2 runs commands from the repository root with `venv\Scripts\python`.

1. Collect every exact node in Test Impact And Traceability and fail the slice
   on any missing or deselected node.
2. Run the focused contract/runtime batch:

   ```powershell
   venv\Scripts\python -m pytest tests\unit\cognition_observability tests\unit\cognition_shared\test_observation_checkpoint.py tests\unit\cognition_resolver\test_capabilities.py tests\test_shared_memory_prewarm.py tests\test_local_context_resolver_integration.py -q
   ```

3. After the entire test surface is implemented and its exact collection
   baseline is recorded, run the trace-specific live LLM/live DB case by
   itself:

   ```powershell
   $env:MONGODB_DB_NAME = "asuna_core_v2"
   $env:IDENTITY_GROWTH_DATABASE_GUARD = "1"
   $env:IDENTITY_GROWTH_TEST_DATABASE = "asuna_core_v2"
   $env:NAPCAT_PREWARM_LIVE_GUARD = "1"
   $env:NAPCAT_PREWARM_LIVE_DATABASE = "asuna_core_v2"
   venv\Scripts\python -m pytest -m "live_llm and live_db" tests\test_shared_memory_prewarm_live_llm.py::test_production_napcat_command_recalls_seeded_shared_memory -q -s
   ```

   This is the user-directed re-verification for
   `llmtrace_7177e22b64a84324a3f60481ae184353`. Inspect the single case and
   record the expected memory unit
   `seed_7ac6348ccd9bf7a80fbc74584c6b3ce3`, actual prewarm status/reason,
   retrieved/merged counts, and safe evidence references. Run it once in the
   tests-first failure baseline and again after production implementation. If
   passing it requires a prompt, model, route, worker-selection, or retrieval-
   semantic change, stop at the failing evidence and amend the approved scope
   before that production change.

4. Run Brain publication/state tests:

   ```powershell
   venv\Scripts\python -m pytest tests\unit\brain_service\test_cognition_graph_projection.py tests\test_service_background_consolidation.py tests\test_service_ops_status.py tests\test_conversation_progress_v2_service.py tests\unit\nodes\test_persona_supervisor2_schema.py tests\unit\nodes\test_persona_supervisor2_cognition_commit.py tests\test_conversation_progress_group_scene.py -q
   ```

5. Run console contract/client/repository/static/stream tests:

   ```powershell
   venv\Scripts\python -m pytest tests\test_control_console_contracts.py tests\test_control_console_kazusa_client.py tests\test_control_console_bootstrap.py tests\test_control_console_stream.py tests\test_control_console_redaction.py tests\test_control_console_repository.py tests\test_control_console_web_surface.py tests\test_cognition_observability_docs.py -q
   ```

6. Run cognition console E2E tests:

   ```powershell
   venv\Scripts\python -m pytest tests\control_console_e2e\test_debug_chat_e2e.py tests\control_console_e2e\test_cognition_observability_e2e.py tests\control_console_e2e\test_stage3_fresh_database_e2e.py -q
   ```

7. Run deterministic legacy-consumer replacements:

   ```powershell
   venv\Scripts\python -m pytest tests\test_self_cognition_integration.py::test_prepared_commitment_state_contains_public_group_scene tests\test_real_history_personality_fixture_contract.py::test_private_monologue_uses_only_canonical_reasoning_node tests\test_real_history_personality_fixture_contract.py::test_private_monologue_fails_closed_when_canonical_node_is_missing -q
   ```

8. Compile and collect the changed live-LLM supplemental files without
   executing their model-backed cases:

   ```powershell
   venv\Scripts\python -m py_compile tests\test_shared_memory_prewarm_live_llm.py tests\test_real_history_personality_e2e_live_llm.py tests\test_short_horizon_state_composition_live_llm.py tests\test_short_horizon_state_composition_e2e_live_llm.py tests\test_rag3_media_debug_adapter_e2e_live_llm.py tests\test_qq_group_public_scene_live_llm.py
   venv\Scripts\python -m pytest --collect-only tests\test_shared_memory_prewarm_live_llm.py tests\test_real_history_personality_e2e_live_llm.py tests\test_short_horizon_state_composition_live_llm.py tests\test_short_horizon_state_composition_e2e_live_llm.py tests\test_rag3_media_debug_adapter_e2e_live_llm.py tests\test_qq_group_public_scene_live_llm.py -q
   ```

   Any additional live-LLM execution remains one case at a time under the
   `test-style-and-execution` contract and requires a separately recorded
   execution decision. Step 3 records the user's explicit decision for the
   isolated `#napcat` case.
9. Run the ownership-manifest changed-path checker and
   `tests/test_test_impact_manifest.py` using the documented HOWTO command.
10. Run `py_compile` for every changed Python production module,
   `git diff --check`, and the relevant broader deterministic regression batch
   selected by `test-style-and-execution`.
11. Start clean isolated Brain/fake-Brain and console processes, then validate
   Overview Latest, Debug cognition, and Self-cognition Latest through
   deterministic DOM assertions, with screenshot capture excluded, for:

   - completed/empty/skipped/failed prewarm;
   - conversation progress and group scene peer layout;
   - record counts and omission marker;
   - generic producer-added section label;
   - canonical rendered edges;
   - debug loading banner followed by authoritative graph;
   - long/multiline/CJK/emoji/HTML-sensitive values; and
   - zero browser console errors.

12. Inspect the final worktree and line diff against the owned path baseline
   and record exact outputs, deviations, and residual risk.

## Progress Checklist

- [x] Three-pass implementability-review quota is complete; every final-pass
  finding is resolved in the architect's consolidated closure below.
- [x] User approves the architecture and production scope.
- [x] Plan status is promoted to `approved` and then `in_progress`.
- [x] Fixed Luna handoff, baseline, skills, and owned paths are recorded.
- [x] Complete tests-first surface and comprehensive pre-production baseline
  are recorded before remaining production implementation.
- [x] Strict v1 contract and ICD are implemented.
- [x] Explicit prewarm outcome survives retrieval, merge, cognition, Brain,
  console, and renderer boundaries.
- [x] Trace `llmtrace_7177e22b64a84324a3f60481ae184353` is re-verified
  through the isolated `#napcat` live LLM/DB case, including the exact outcome
  reason and seeded-memory evidence. The legacy trace lacks the worker task and
  outcome records needed to reconstruct its exact internal disposition; the
  current isolated reproduction is recorded below.
- [x] Conversation progress and group scene use peer context sections.
- [x] Live and self projection use the same section model and budgets.
- [x] Brain response/latest graph fields are typed.
- [x] Console semantic reconstruction and duplicate graph models are removed.
- [x] Shared renderer uses producer labels and generic sections.
- [x] Pending/error debug UI is separate from authoritative cognition graph.
- [x] Exact source-to-test manifest is current.
- [x] Focused and regression tests pass.
- [x] Deterministic UI/DOM evidence passes for all three views with zero console
  errors and no screenshot capture.
- [x] All architect findings and user-reported regressions are resolved and
  line-reviewed.
- [x] Plan evidence and registry are complete and the plan is archived.

## Execution Evidence

Production execution was approved by the user on 2026-08-26 through the
explicit instruction to execute this plan.

User execution-policy amendment on 2026-08-26: complete all remaining tests,
fixtures, and manifest expectations before resuming production implementation.
The accepted Handoff 1 work and partial Handoff 2 workspace state remain
preserved. Role 2 freezes further production edits, finishes the complete test
surface, collects every exact node, and records one comprehensive expected
baseline before production work resumes. This changes execution sequencing
only; the approved architecture, scope, fixed executor, contracts, exclusions,
and acceptance criteria remain unchanged.

User-added prewarm verification on 2026-08-26: re-verify the failure recorded
under `llmtrace_7177e22b64a84324a3f60481ae184353`, where direct input containing
`#napcat` should have recalled its corresponding shared memory. The protected
trace export resolved one successful run with nine LLM steps and no cognition
failure capsule. The active shared-memory export resolved canonical row
`seed_7ac6348ccd9bf7a80fbc74584c6b3ce3`; its ID and full content were absent
from every cognition-stage prompt and parsed output. These sanitized facts are
now a deterministic recall/merge regression plus one isolated live LLM/live DB
verification gate in the tests-first phase. Raw trace and memory payloads remain
only under `test_artifacts/diagnostics`.

Tests-first checkpoint result: PASS. Role 2 added the full remaining unit,
integration, browser-E2E, documentation, guarded live, fake-Brain, fixture,
legacy-consumer, and ownership-manifest surface without changing any frozen
production or documentation file. The plan-owned surface collected 309 tests;
the six live files collected 63 tests and compiled. Exact prewarm/resolver/
persona/group/local nodes passed 15/15, Brain/ops/progress/self/legacy nodes
passed 11/11, manifest checks passed 12/12 with 135 registered entries, 38
changed Python test paths compiled, targeted test Ruff passed, and
`git diff --check` passed. The frozen-production baseline intentionally reports
39 console deterministic passes with 22 failures, six browser-E2E failures,
and four documentation failures. One background-consolidation node timed out
after approximately 60 seconds following a cancelled event-log write and is
retained for post-implementation rerun. The unchanged resolver baseline remains
9 failures and 28 passes, all from the already-recorded required
`goal_continuation_ref` fixture drift.

The user-authorized isolated live LLM/live DB `#napcat` case passed against
`asuna_core_v2`: the current prewarm returned
`completed/shared_memory_ready`, retrieved one safe row with memory unit
`seed_7ac6348ccd9bf7a80fbc74584c6b3ce3`, and finalized as
`completed/shared_memory_merged` with merged count one. Therefore the semantic
omission in the historical trace is confirmed, while its old internal prewarm
reason is not recoverable from that pre-observability trace and the failure is
not deterministic under the current isolated path. The same exact case remains
mandatory after production implementation.

Draft reconnaissance record:

- Pre-draft worktree baseline: clean `git status --short` on 2026-08-26.
- Current plan-owned worktree state: modified `development_plans/README.md`
  plus this untracked draft plan. Concurrent untracked
  `docs/architecture/dsh_integration_architecture.md` is outside this plan's
  ownership and remains untouched; production and test paths remain unchanged.
- Architect: `/root`.
- Persistent implementation executor reserved:
  `/root/cognition_console_implementer`, `gpt-5.6-luna`, reasoning `max`,
  standard execution speed.
- The executor completed read-only reconnaissance, edited no files, and ran no
  tests. Its findings confirmed the architect's ownership and provenance map.
- First independent plan review: FAIL. It identified the Brain-service versus
  cognition-package boundary, view-envelope, cancellation, state-carrier,
  producer-catalog, exact-symbol, documentation, and traceability gaps. The
  architect revised this draft before requesting a second review.
- Second independent plan review: FAIL. It identified conflicting identifier
  grammars, undefined run-status aggregation, unclosed scalar flattening,
  incomplete prewarm merge/count semantics, an ambiguous group-scene failure
  discriminator, missing legacy graph consumers/manifest replacements, and a
  non-exact deletion gate. The architect closed those decisions and expanded
  the cutover/test surface before requesting a third review.
- Third and final independent plan review: FAIL before final revision. It
  identified live terminal mapping, run-kind section sets, self correlation,
  exact prewarm types/counts/failure retention, closed source paths, console
  envelope placement/error conversion, remaining stale symbols, exact live
  test ids, subsystem docs, nested SSE correlation, and storage-only consumer
  classification. At the user's explicit three-pass cap, the architect
  resolved all findings together in one consolidated final revision. No
  fourth review is permitted or requested.

## Independent Plan Review

The persistent Luna agent completed three read-only implementability passes;
all results are recorded under Execution Evidence. The user set three as the
hard review quota and directed the architect to close the final findings in
one pass. The final revision therefore uses the third review as a consolidated
finding inventory and records an explicit process deviation from the usual
independent-PASS gate. The architect traced every finding to a closed contract,
source path, change-surface entry, exact test, or verification rule in this
document. The next authority is the user's production-scope approval; another
plan-review pass is outside the approved workflow. Review activity grants no
production authorization.

Final closure map: section 2 fixes terminal/status/required-set rules; section
3 fixes prewarm types, bounds, count unit, repeat/invalid merge, exception set,
and request-scoped failure capture; sections 4-5 fix self/live typed sources
and every flat source path; section 7 fixes console field placement,
availability catches, direct debug carriage, and nested correlation; the
traceability/change-surface/delete/verification sections enumerate every
legacy consumer, manifest replacement, stale symbol, README, exact live test
node, and unchanged opaque consumer.

## Independent Code Review

The architect performs the final read-only code review because Role 1 remains
independent of Role 2 implementation. Review authority covers:

- one schema owner and absence of duplicate console vocabulary;
- exact prewarm disposition preservation and unchanged retrieval authority;
- strict bounds, truthful counts, reference integrity, and version checks;
- safe disclosure and absence of protected/raw/operational material;
- conversation-progress/group-scene parity;
- live/self category parity and honest status handling;
- direct Brain-to-console contract carriage;
- one generic frontend renderer and separated debug loading/error UI;
- test/manifest/browser evidence; and
- scope, style, compatibility, and unrelated-diff review.

Role 2 owns remediation. The architect re-reviews the corrected diff and is the
only final sign-off authority for this plan.

## Execution Handoff

After approval, each handoff to `/root/cognition_console_implementer` records:

- this approved plan identifier and current lifecycle status;
- the bounded implementation slice and exact owned files;
- the pre-handoff worktree/owned-file baseline;
- mandatory skills and acceptance nodes for that slice;
- the fixed model/reasoning/standard-speed configuration;
- completed evidence and outstanding findings;
- the next architect checkpoint; and
- the instruction to preserve concurrent work and return a scoped diff plus
  inspected verification output.

The same agent is reused for all later implementation, test, browser, and
remediation handoffs.

### Handoff 1: Resolver Prewarm And State Carriers

- Lifecycle state: `in_progress` after explicit user approval on 2026-08-26.
- Role: Role 2, fixed implementation and verification executor.
- Resolved executor: `/root/cognition_console_implementer` on
  `gpt-5.6-luna`, reasoning `max`, standard execution speed, with repository
  filesystem, terminal, patch, skill, and deterministic-test access.
- Resolution mode and rationale: plan-scoped fixed execution constraint. The
  exact binding is available and satisfies the cross-layer Python, contract,
  and deterministic verification capability floor; substitution is outside
  the approved contract.
- Baseline: `git status --short` was clean before lifecycle promotion. At
  handoff it contains only architect-owned modifications to this plan and
  `development_plans/README.md`. SHA-256 baselines were captured for every
  existing slice-owned path; the new checkpoint test path was absent.
- Owned production surface: resolver contracts/capabilities, the cognition
  attempt ledger, persona schema/orchestration/cognition carrier paths.
- Owned verification surface: the prewarm, resolver capability, checkpoint,
  persona schema/cognition, group-scene, local-context integration tests, and
  their ownership-manifest rows named in the plan.
- Mandatory constraints: all relevant mandatory skills loaded by Role 2;
  `venv\\Scripts\\python`, `apply_patch`, CJK-safe Python handling, no `.env`,
  no compatibility aliases, and no prompt/model/route/retrieval-semantic
  changes.
- Entry verification: production/test paths were unchanged from the captured
  baseline; the plan and registry alone carry lifecycle edits.
- Acceptance output: the exact mapped nodes for this owned slice collect and
  pass, changed production modules compile, `git diff --check` passes, and the
  executor returns a scoped diff plus inspected output and deviations.
- Next checkpoint: architect inspection of the resolver/state-carrier diff and
  evidence before assigning the observation-contract/Brain cutover slice.

### Handoff 1R: Resolver Checkpoint Remediation

- Prior assignment: Handoff 1 completed its 14 exact mapped nodes, 48 adjacent
  owned tests, 12 manifest checks, changed-module compilation, and
  `git diff --check`.
- Reviewer: Role 1 architect, read-only checkpoint review.
- Findings: required group-scene discriminator reads could silently produce
  `None`; one new exception log omitted the exception text; the group-scene
  owner test stopped at a private helper; fixed prewarm reason mappings lacked
  complete direct coverage; and touched manifest descriptions retained stale
  pre-change contract wording.
- Resolved executor/configuration: the same fixed
  `/root/cognition_console_implementer`, `gpt-5.6-luna`, reasoning `max`,
  standard speed. This is remediation within the original owned slice, not a
  reassignment.
- Acceptance output: findings corrected without semantic-scope expansion, all
  14 original mapped nodes recollected and passed, affected adjacent and
  manifest tests passed, production modules compiled, and diff checks passed.
- Additional evidence request: identify the reported nine baseline resolver
  fixture failures and the evidence that they pre-date this slice; preserve
  them outside the remediation diff.
- Next checkpoint: architect re-review and explicit Slice 1 acceptance.

Checkpoint result: PASS. Role 1 re-reviewed the corrected production, test,
and manifest diff. Role 2 collected all 14 exact nodes, passed all 14, passed
60 affected adjacent/manifest tests, compiled every changed production module,
passed `git diff --check`, and passed targeted Ruff checks. The complete fixed
prewarm reason matrix is directly exercised, group-scene propagation is tested
through the persona owner return boundary, and the checkpoint remains scoped
to the current graph attempt.

Recorded baseline deviation: the unchanged
`tests/test_cognition_resolver_contracts.py` suite reports 9 failures and 16
passes because nine existing fixtures omit the already-required
`goal_continuation_ref`. The test file is unchanged by this plan and the
requirement exists in `HEAD`; these stale fixtures pre-date Handoff 1 and remain
outside this plan's approved surface.

### Handoff 2: Observation Contract And Brain Publication

- Remaining scope: implement the complete strict v1 DTO and producer catalog,
  cut live/self Brain publication and latest storage to the typed model, remove
  the frozen service projection vocabulary, and migrate the exact legacy test
  consumers owned by this backend boundary.
- Owned surface: the two new Brain observation modules, Brain exports/contracts,
  `service.py`, cognition-observability and Brain projection tests, named
  service/legacy-consumer tests, three compile/collect-only live-LLM consumers,
  and corresponding ownership-manifest rows.
- Baseline: all new module/test paths are absent. SHA-256 baselines were
  captured for every existing owned path after accepted Handoff 1. The only
  worktree changes at entry are architect lifecycle evidence and accepted
  Handoff 1 files.
- Resolved executor/configuration: the same fixed
  `/root/cognition_console_implementer`, `gpt-5.6-luna`, reasoning `max`,
  standard speed, with Python, patch, collection, and deterministic-test access.
- Selection rationale: the plan-scoped binding remains eligible and retains
  the implementation context from Handoff 1; reassignment would violate the
  fixed constraint and increase cross-boundary rework.
- Mandatory constraints: plan/execution gates, local-LLM boundary audit,
  Python style, test execution, CJK safety, no `.env`, no compatibility layer,
  no prompts/models/routes/retrieval/persistence/adapter/database changes.
- Completed verification at entry: Handoff 1 exact and adjacent gates pass;
  the nine documented stale resolver fixtures remain outside this slice.
- Acceptance output: every exact mapped backend node collects and passes;
  Brain publication/state batches pass; changed Python compiles; changed
  live-LLM files compile and collect without execution; manifest/style/diff
  gates pass; and Role 2 returns inspected evidence and deviations.
- Next checkpoint: Role 1 independent DTO/projection/disclosure/publication
  review before any console transport handoff.

### Handoff 2T: Consolidated Tests-First Review Remediation

- Reviewer: Role 1 architect after the complete tests-first checkpoint; no
  remaining production or documentation implementation may resume until this
  test-only checkpoint passes.
- Findings: one console fixture used non-canonical terminal status `completed`;
  DTO negative coverage did not enforce several strict grammar, uniqueness,
  run-kind, edge, scalar, record-key, and payload invariants; live/self catalog
  assertions allowed missing or additive base entries; protected-field checks
  omitted some exclusion values; terminal mapping omitted accepted statuses;
  the failed-run prewarm test stopped at a checkpoint getter instead of the
  service failure-publication seam; fake-Brain categories, record keys, and
  settled prewarm reason were not canonical; and browser tests did not prove
  generic additive-section rendering, CJK/emoji/multiline/HTML-safe text, or
  zero console/page errors across Overview, Debug, and Self.
- Resolved executor/configuration: the same fixed
  `/root/cognition_console_implementer`, `gpt-5.6-luna`, reasoning `max`,
  standard speed. Ownership is limited to tests, fixtures, and their manifest
  rows; the existing production and documentation hashes remain frozen.
- Acceptance output: all findings become assertions under the existing exact
  nodes where possible, the whole plan surface recollects, affected
  deterministic tests and expected frozen-production failures are inspected,
  test Python compiles and passes targeted Ruff, manifest checks and
  `git diff --check` pass, and production hashes match the freeze baseline.
- Next checkpoint: Role 1 reviews the corrected test contract once, then the
  same executor resumes the consolidated production phase.

Checkpoint result: PASS. Role 1 re-reviewed the corrected tests and fixtures.
The exact observation contract cases now enforce nested-extra rejection,
strict integer and UTC handling, section/record/reference grammar and
uniqueness, canonical ordering, run-kind catalogs, truthful counts, bounded
payloads, and additive producer-owned sections. Projection tests assert the
exact live/self catalogs and node references, every protected value, all
terminal dispositions, partial-run downgrade, current-attempt failure
publication, and retry isolation. Browser fixtures now cover the exact shared
layout on Overview, Debug, and Self, including an unknown producer section,
CJK/emoji/multiline/HTML-sensitive text, escaped DOM output, and zero browser
errors.

Role 2 passed the 14 original mapped nodes, the deterministic `#napcat`
regression, five Brain projection tests, twelve ownership-manifest checks,
collection of 263 plan-owned tests, compilation of 42 changed Python test
files, targeted Ruff, and `git diff --check`. The manifest resolves 374 exact
required nodes across 94 files; its mapped surface collects 1,204 tests. Six
live-LLM files collect 63 tests without execution. The production hash freeze
remained intact. Expected red-state evidence against frozen production was
inspected: the missing console DTO/renderer and strict record-key enforcement
fail their new tests, while the previously recorded nine stale resolver
fixtures and background-consolidation timeout remain baseline deviations.

The tests-first implementation boundary is satisfied. Production and
documentation implementation may now proceed as one consolidated phase.

### Handoff 3: Consolidated Production And Documentation Implementation

- Resolved executor/configuration: the same fixed
  `/root/cognition_console_implementer`, `gpt-5.6-luna`, reasoning `max`,
  standard speed.
- Scope: the entire remaining plan-owned production, frontend, ICD, subsystem
  documentation, fixture, and ownership-manifest surface was implemented only
  after Handoff 2T accepted the complete tests-first surface.
- Verification result: the exact resolver/prewarm/node batch passed 14 tests;
  cognition/Brain batches passed 9 tests and the Brain projection batch passed
  5 tests; documentation/manifest checks passed 19 tests; the deterministic
  console batch passed 149 tests with two inspected stale-fixture failures;
  47 changed Python files compiled; targeted Ruff, JavaScript syntax, and
  `git diff --check` passed. The manifest manually resolved all 418 required
  exact nodes across 94 files. The official all-node impact checker could not
  launch its single Windows command because the command line exceeded the
  platform limit.
- Browser result: the available Playwright fallback produced and inspected the
  debug cognition and settlement-graph screenshots. The final debug-chat and
  canonical-edge smoke tests passed with no browser errors. Two supplemental
  cognition-observability browser cases exposed fixture/helper defects and are
  assigned to the tests-first remediation below.
- Live prewarm result: the one authorized post-production run of
  `tests/test_shared_memory_prewarm_live_llm.py::test_production_napcat_command_recalls_seeded_shared_memory`
  passed. The real prewarm returned `completed/shared_memory_ready`, retrieved
  memory unit `seed_7ac6348ccd9bf7a80fbc74584c6b3ce3`, and the production merge
  returned `completed/shared_memory_merged` with merged count one. This is an
  isolated retrieval-and-merge proof, not a replay of the whole historical
  cognition graph. It will not be repeated during deterministic remediation.
- Inspected deviations: two console tests retain pre-cutover fake signatures;
  two service descriptor tests reference symbols already absent from `HEAD`;
  nine resolver fixtures omit an already-required reference; and one required
  background-consolidation case waits on an unpatched event-logging seam. The
  plan-owned fixture defects are included below; unrelated baseline failures
  remain documented outside the production diff.

### Handoff 3T: Post-Production Independent Review Tests-First Remediation

- Reviewer: Role 1 architect after the complete consolidated implementation.
  Production and documentation hashes are frozen until this test-only
  checkpoint is accepted.
- Correlation and transport findings: the service failure publisher must map
  the current attempt-ledger cognition invocation identifier to both canonical
  correlation fields; an attempted debug HTTP failure must report
  `debug_request_failed`; and a missing observation response field must be a
  protocol error rather than a reported null value.
- Projection findings: context-consumption aggregate status must consider only
  reported source containers; character and relationship Details must follow
  their closed disclosure mappings; style relevance/surface records must use
  the exact consumer/source labels; public group scene must report its Status
  and classify invalid headers; conversation-progress invalid headers must
  affect status; self-source records must use the stable v1 labels; self action
  fallback applies only to a valid empty results list; invalid preferred
  self-message sources must not silently fall through; and visible-message
  reported counts must describe the source container rather than only valid
  displayed rows.
- Renderer findings: canonical `skipped` must remain visibly `skipped`; scalar
  lists must render as ordered list rows; and Overview/Debug/Self pins must be
  selected independently by view kind. Static and browser coverage must prove
  these behaviors through the generic renderer.
- Documentation findings: the authoritative ICD must state the full owner,
  producer/publisher/transport/consumer, model, status, catalog/mapping,
  prewarm, disclosure, budget, availability, versioning, live/self, and
  additive-rendering contracts. The Brain and architecture documentation must
  remove contradictory legacy snapshot/input-disclosure descriptions. Tests
  must verify the substantive contract rather than only isolated tokens.
- Fixture findings: migrate the two stale console fakes to canonical typed
  signatures; correct the two supplemental browser fixtures/helpers; and mock
  the background chat test's event-log intake seam so the required node
  terminates deterministically.
- Test-only ownership: cognition-observation projection/Brain tests, console
  client/web/static/browser tests and fixtures, the two stale console tests,
  the background-consolidation test, documentation tests, and manifest rows
  only when an exact node changes. All plan-owned production and documentation
  files remain frozen.
- Acceptance output: implement the complete remediation assertion surface
  first; run it against frozen production; inspect every expected contract
  failure and corrected fixture pass; compile and lint changed test Python;
  run JavaScript syntax and diff checks where applicable; and prove all frozen
  production/documentation hashes remain unchanged. The authorized live
  `#napcat` test is excluded from rerun.
- Next checkpoint: Role 1 reviews the consolidated corrected test surface once.
  Only then may the same executor receive one consolidated production and
  documentation remediation handoff.

Checkpoint result: PASS. Role 2 changed only the assigned tests and fixtures;
all 26 frozen production/documentation hashes remained exact and no prewarm
test or live test ran. The targeted batch produced 14 passing fixture,
contract, and Brain cases plus 12 inspected red assertions against frozen
production/docs: six projection mappings, missing-observation protocol
handling, debug HTTP failure reason, and four substantive documentation
contracts. Plan-owned deterministic collection found 254 tests; the six live
files collected 63 without execution; 12 manifest tests, exact-node impact
validation, affected Ruff, `py_compile`, JavaScript syntax, and diff checks
passed. Repository-wide collection retained two unrelated missing-module
errors. Browser fallback retained the two expected frozen-renderer red states
and passed seven existing debug/stage/error/lifecycle/visual smokes. Role 1
reviewed and accepted the assertion boundaries; Handoff 3P now extends this
same tests-first checkpoint before any consolidated production remediation.

### Handoff 3P: Historical Prewarm Regression Inventory And Tests-First Gate

The user required a complete review of prior prewarm repairs before any new
prewarm test or production change. Role 1 completed that read-only review and
froze the following chronology and non-regression rules:

1. April Stage 1 (`rag_reply_mention_and_vague_input_plan.md`) diagnosed the
   current-bot mention as addressing metadata that had been flattened into
   content and could retrieve unrelated bot-addressed history. Its RAG-slot
   sanitizer was explicitly a stopgap.
2. April Stage 2 (`typed_message_envelope_stage2_plan.md`) moved the canonical
   boundary to `body_text` plus typed mentions and prohibited brain-side
   platform-wire parsers. The readable `@display_name` representation can
   still occur in `body_text`, so downstream consumers must distinguish typed
   active-character addressing from authored retrieval content structurally.
3. Commit `4b1626a0` introduced the one-attempt first-cognition shared-memory
   lookup. It used the complete decontextualized input and collapsed worker
   failures or empty results into an empty RAG payload without a typed
   disposition.
4. Commit `01352ca7` replaced the bounded prewarm worker with the full RAG3
   resolver. Commit `42463e9b` restored the direct
   `PersistentMemorySearchAgent`, one attempt, shared memory only, and tests
   forbidding the full resolver and `user_memory_units` lane.
5. Commit `9210bede` disconnected the prewarm caller during the V2 cutover
   while helper-only tests remained green. Commit `27da3169` restored one
   cycle-zero caller, the join, and the cognition carrier. This establishes
   that helper-only verification is insufficient.
6. The July mention regression captured in
   `cognition_core_v2_prewarm_mention_content_query_bugfix_plan.md` showed
   `@一之濑明日奈 #napcat` entering the worker as one query and centering the
   generated search on the character name. The initial substring repair
   damaged longer literals such as `@一之濑明日奈-archive`; commit `3b4edbd7`
   retained a whitespace-bounded exact-token removal and tests for mention
   only, non-character mentions, plain character names, and longer literals.
7. Commit `779ab035` added a V2 in-connector parent-replay guard so that replay
   reused preparation and did not rerun cycle-zero prewarm. The later V3
   cutover removed that parent-replay architecture. Current outer service
   retries are fresh graph attempts with isolated prewarm checkpoints; this
   remediation preserves that current boundary rather than restoring V2.
8. Commit `ceecc252` deleted the 506-line caller-level prewarm suite during the
   cognition projection cutover while retaining the production caller and
   helper tests. The current plan must restore equivalent caller-to-cognition
   coverage and keep it mapped in the ownership manifest.
9. Commit `e5159722` made both normal RAG and prewarm use the same typed memory
   authority classifier: only active, global, curated/certified shared rows
   enter; private, scoped, target-bearing, unsupported, and
   `user_memory_units` rows remain excluded; fact and conditional
   self-guidance authority remain distinct.

Read-only inspection of the current path found a remaining secondary leak.
`run_first_cycle_shared_memory_prewarm` sanitizes the worker `task` to
`#napcat`, but passes the original `prompt_message_context` into
`build_text_chat_rag_request`. `project_runtime_context_for_llm` retains that
object, and `PersistentMemorySearchAgent._generator` serializes it into the
model request. The model therefore still sees
`body_text="@一之濑明日奈 #napcat"` and a typed bot-mention row containing the
same display name. Existing tests assert only the sanitized `task` and do not
inspect this model-visible context.

The same active address has a second model-visible route. Brain ingress commits
the current conversation row before the service loads `chat_history_recent`
and `chat_history_wide`. The RAG request adapter forwards those histories, and
the common worker projection renders them without applying the already-typed
active-turn row/message identifiers. A prewarm-specific projection must
therefore exclude current-turn history rows by those exact identifiers while
preserving every earlier history row; it must not remove character names or
mentions recursively from authored historical context.

The protected historical trace
`llmtrace_7177e22b64a84324a3f60481ae184353` has exactly that envelope shape:
readable body `@一之濑明日奈 #napcat`, raw QQ wire mention
`[CQ:at,qq=3768713357]`, and a typed bot mention whose global id is the active
character id. Its cognition prompts contain no shared-memory evidence and the
visible response asks what `#napcat` means. The old trace does not contain a
prewarm task, worker result, or disposition, so the exact old internal failure
mode cannot be reconstructed or relabeled after the fact.

Before any prewarm production edit, the consolidated tests-first checkpoint
must add and inspect all of these assertions against frozen production:

- parameterized Asuna and Kazusa active-character mentions produce the exact
  worker task `#napcat`, and the complete model-visible worker payload contains
  neither the active readable mention nor its display name from the current
  turn as retrieval content;
- the prewarm-specific prompt-context copy carries the sanitized body while
  the source state/envelope remains unchanged;
- prewarm-specific recent/wide history copies exclude rows matching the typed
  active-turn conversation-row or platform-message ids, while preserving
  earlier rows even when those rows legitimately mention the character;
- a typed mention is treated as structural active-character addressing only
  when its `global_user_id` matches `character_profile.global_user_id`;
  participant mentions and other-bot mentions remain authored retrieval
  content;
- mention-only, plain-name-subject, non-character/other-bot, and longer-literal
  cases remain explicit so the earlier substring and over-stripping mistakes
  cannot return;
- the prewarm still calls only the direct persistent-memory worker once,
  preserves literal `#napcat`, admits only certified shared rows, and never
  enters the full resolver or current-user memory lane;
- one deterministic caller-to-cognition test uses the historical typed-envelope
  shape and an exact seeded memory id, without patching away the prewarm
  boundary, and proves that the merged row reaches cognition input and the
  finalized outcome carrier;
- cancellation and later cycles retain their one-attempt and
  no-fabricated-outcome guarantees; fresh outer graph attempts retain isolated
  checkpoints and the existing service retry behavior;
- the historical protected trace export remains read-only evidence, and the
  already-authorized isolated live DB/LLM test is not rerun during this
  remediation.

This gate extends Handoff 3T. All observation, console, documentation, fixture,
and prewarm regression tests must be complete and reviewed together before one
consolidated production/documentation remediation handoff. Structural mention
sanitation may alter only the prewarm-specific request projection; it may not
add name/alias keyword classification, platform syntax parsing, a new LLM
call, prompt changes, retrieval-worker changes, or semantic post-processing.
The archived July residuals remain deferred: typed mentions still have no
occurrence offsets, so generalized punctuation cleanup and ambiguous repeated
visible-token interpretation require separate captured evidence and scope.

Checkpoint result: PASS. Role 2 changed only the assigned deterministic tests,
fixtures, and exact ownership-manifest mappings; all 26 frozen production and
documentation hashes remained exact. The consolidated prewarm batch produced
19 passes and seven inspected red assertions against frozen production: the
Asuna/Kazusa prompt-context copies still retained the active readable mention,
both active-turn history projections retained the current row, the other-bot
and unproven-bot identity cases exposed the existing over-stripping behavior,
and the real caller-to-cognition regression retained the unsanitized current
body. All direct-worker, one-attempt, certified-shared-authority, merge,
cancellation, later-cycle, source-immutability, older-history, plain-name,
longer-literal, and non-character-mention assertions passed. The three owned
files collected 36 deterministic tests; the isolated live prewarm file
collected one test without execution; 12 manifest tests, impact validation,
Ruff, `py_compile`, and diff checks passed. Role 1 reviewed and accepted the
assertion boundaries. Handoffs 3T and 3P now form the complete tests-first red
checkpoint for one consolidated production and documentation remediation.

## Closure Evidence

Closure review completed on 2026-08-27 under the user's final instruction to
use source-line and deterministic DOM evidence without screenshots.

### Historical Prewarm Re-verification

- The historical failure is correlated to
  `llmtrace_7177e22b64a84324a3f60481ae184353`: the `#napcat` turn should have
  recalled the corresponding shared memory, but the cognition run did not use
  it. That legacy trace predates worker task/outcome telemetry, so its exact
  internal prewarm disposition cannot be reconstructed truthfully.
- The authorized isolated production-path reproduction ran once. Memory unit
  `seed_7ac6348ccd9bf7a80fbc74584c6b3ce3` was retrieved and merged with
  `status=completed`, `reason=shared_memory_merged`, retrieved count `1`, and
  merged count `1`.
- The prior-fix inventory is covered by deterministic tests: exact typed
  active-character address removal for both Asuna and Kazusa, exact current-
  turn history exclusion, preservation of plain names/longer literals/other-
  bot and unproven mentions, direct persistent-memory worker ownership,
  one-attempt behavior, typed evidence authority, cancellation, and conditional
  guidance. No prompt, model, route, retrieval semantics, adapter, or database
  contract changed.

### Layout And Selection Line Audit

- `src/control_console/static/index.html` has zero diff from pre-plan `HEAD`.
- `console.css` has no removed or modified baseline rule. Its exact delta is 18
  added `.observation-*` declarations for approved inspector content. The
  baseline `.graph-branch-stack` declaration remains exact, with natural
  height and no `max-height` or `overflow-y` cap.
- Shell, sidebar, navigation, top bar, content grids, card placement, card
  widths, rail sizing, inspector placement, and page overflow selectors are
  unchanged.
- The JavaScript graph keeps the baseline run-summary -> semantic-summary ->
  relationship-panel -> graph-body order, fitted producer-column rail, sorted
  observed columns, lane then label ordering, cognition-first relationship
  ordering, connector algorithm, and independent per-view pins. The selection
  path again uses the one baseline `[data-graph-node]` handler and validated
  pinned-node lookup, adapted only from legacy `id` to canonical `node_id`.
- The one intentional wrapper-name delta is `graph-node-stack` to the already
  baseline-styled `graph-branch-stack`; this is the user-directed restoration
  of the existing 8 px gap between sibling cards. It adds no boundary, cap, or
  new topology.
- The three exercised E2E files contain zero screenshot calls or screenshot
  metadata. Deterministic DOM checks confirm four live/debug column groups,
  five self groups, equal group/card widths, 12 px rail gaps, 8 px sibling
  gaps, no nested stage wrappers, reachable nodes, no page-level horizontal
  overflow, and zero recorded browser console/page errors.

### Final Verification

- resolver/prewarm/observation focused batch: 37 passed;
- Brain publication, state, progress, and group-scene batch: 40 passed;
- console contract/client/repository/static/docs batch: 69 passed;
- deterministic control-console E2E batch: 7 passed;
- legacy-consumer replacements: 3 passed;
- exact source-impact validator: 64 of 64 mapped nodes passed;
- ownership-manifest suite: 12 passed;
- documentation contract suite: 4 passed;
- changed production Python compilation, JavaScript syntax, and
  `git diff --check`: passed;
- live-LLM supplemental compilation and collection: 63 nodes collected with
  pytest addopts cleared; no live case was executed during final closure.

### Final Change Radius

- The Change Surface declares every plan-owned source, test, documentation,
  manifest, registry, and lifecycle path. The previously omitted debug caller,
  lookup caller, isolated live prewarm, and plan-lifecycle paths are now
  explicit.
- Four pre-existing concurrent changes are outside this plan and remain
  untouched: the three active DSH plans and
  `tests/test_service_event_logging.py`.
- `tests/test_conversation_progress_v2_service.py` has staged/unstaged state in
  the shared worktree but zero net content diff from `HEAD`; it is absent from
  the final changed-path set.
- Final deviation: the user superseded the fixed Role 2 binding for the last
  frontend correction and closure slice. Role 1 performed that bounded work
  directly, and closure relies on the recorded line audit and deterministic
  gates rather than a claim of independent review for those final edits.

## Acceptance Criteria

1. Every live/self Brain cognition observation validates as
   `cognition_run_observation.v1`; Brain response contracts contain the typed
   model rather than a graph dictionary.
2. Console contracts import the Brain-owned model. Console transport performs
   version validation and availability handling only; it contains no cognition
   field allowlist, nested shape guesser, response-field inference, or local
   timestamp replacement.
3. An eligible prewarm run reports `completed` with reason
   `shared_memory_merged`, retrieved/merged counts, and safe evidence records.
   Skipped, empty, and failed outcomes remain distinguishable and truthful.
   Cancellation propagates and produces no fabricated terminal outcome or
   observation. The isolated `@character #napcat` verification retrieves and
   merges memory unit `seed_7ac6348ccd9bf7a80fbc74584c6b3ce3`; the test
   records the exact failure disposition before any remediation.
4. The prewarm outcome reaches the observer while preserving prompt-safe
   evidence authority, worker selection, retry behavior, and cognition
   semantics. The prewarm-only request projection excludes the typed active
   character address mention from task and model-visible current-message
   context and excludes exact typed active-turn rows from its history copies;
   it preserves other authored mentions, older history, and the source
   envelope unchanged.
5. Conversation progress and public group scene render as peer `context`
   sections with identical header, status, summary, fields/records, counts,
   budgets, wrapping, scrolling, and omission treatment.
6. Equivalent live and self categories use the same section model, labels,
   statuses, budgets, and renderer. Missing data appears as `empty`, `skipped`,
   `failed`, or `not_reported` according to the v1 definitions.
7. Domain field names, labels, order, summaries, and information density are
   producer-owned. A fixture section unknown to JavaScript renders correctly
   on Overview, Debug, and Self-cognition views without a JavaScript label or
   field-order change.
8. Approved record truncation is bounded and explicit: counts and the omission
   marker accurately report displayed versus source rows. Whole payload and
   per-field limits fail closed under the contract.
9. Graph edges use only `sequence | reference`; valid edges render, and invalid
   endpoint/kind payloads fail validation rather than disappearing silently.
10. Debug request loading and network failure use dedicated UI states. The
    graph renderer receives only canonical Brain/availability observations.
11. Raw prompts, raw model output, embeddings, raw messages/envelopes, IDs in
    semantic sections, worker error text, and unapproved nested values remain
    absent in backend payloads and escaped from DOM execution.
12. No LLM call, prompt, route, database, adapter, persistence, or live
    cognition behavior changes outside explicit observation carriers.
13. Every changed production source resolves to an existing exact manifest
    node and all focused, regression, manifest, static, and browser gates pass.
14. The final diff contains only plan-scoped paths, the architect records a
    passing independent review, and the completed plan is archived with exact
    evidence.
