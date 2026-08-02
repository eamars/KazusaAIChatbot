# Conversation Progress V2

`conversation_progress` is bounded short-term continuity memory for one
character, user, and conversation surface. It preserves established scene facts
and event lifecycles without asking every response-path LLM to reconstruct the
whole thread.

The module is factual memory, not a response planner. Cognition decides what the
next response should accomplish. Dialog owns final wording.

## Ownership Boundary

| Concern | Owner |
| --- | --- |
| Factual scene narrative, stance, relation, and episode movement | Scene-observer LLM |
| Exact prior-event reconciliation, concrete new events, lifecycle/relevance observations, obligation direction, and outcome | Event-reconciler LLM |
| Future response goals and choices | Cognition |
| Final visible wording | Dialog |
| Persisted lifecycle and retention state machines | Deterministic code |
| Persisted episode status and continuity mapping | Deterministic code |
| Event IDs, source IDs, and timestamps | Deterministic code |
| Prompt-safe character identity and semantic local-clock projection | Deterministic code |
| Temporary-handle mapping | Deterministic code |
| Validation, limits, persistence, and expiry | Deterministic code |
| Compaction selection and block construction | Deterministic code |

Persisted progress contains no `next_affordances` or
`progression_guidance`. The recorder describes what has happened; it does not
plan what cognition should do next.

## Public Facade

Callers use the package facade:

```python
load_result = await load_progress_context(
    scope=scope,
    current_timestamp_utc=current_timestamp_utc,
    platform_bot_id=platform_bot_id,
    active_turn_conversation_row_ids=active_row_ids,
)

record_result = await record_turn_progress(
    record_input=record_input,
)

validated_packet = validate_active_packet(packet)
```

`load_progress_context(...)` concurrently loads the active packet and the
bounded ambient and participant-history lanes. It returns a prompt projection,
logical turns, text-free diagnostics, and source telemetry.

`record_turn_progress(...)` runs after a visible response is accepted, or after
an eligible cognition-silence outcome settles. It invokes one scene observer
and one event reconciler concurrently, composes their independently validated
outputs, maps exact internal identities, applies deterministic compaction when
required, and performs one guarded replacement write.

`validate_active_packet(...)` is the canonical exact V2 shape validator used
by runtime and migration boundaries. A schema-version label alone never makes
a packet valid.

For group Cognition, the already-loaded ambient logical-turn lane is projected
transiently into one bounded public scene. This projection carries visible
speaker order, addresses, reply names, and the current trigger without adding
storage or a group packet. `ConversationProgressScope` and every persisted V2
packet remain keyed to the current platform, channel, and global user; that
participant continuity is a separate lane from the public group scene.

## Runtime Flow

```text
canonical conversation rows
  -> deterministic logical-turn assembly
  -> bounded history selection
  -> active packet load
  -> factual prompt projection
  -> cognition chooses the next response goal
  -> dialog produces and verifies the visible response
  -> scene observer + event reconciler (two concurrent post-turn calls)
  -> deterministic handle/identity/source mapping
  -> deterministic merge and optional compaction
  -> guarded packet/block persistence
```

The response-eligibility decision remains outside this module.

## Post-Turn Observer Contracts

Both observers receive a prompt-safe `semantic_context` containing the exact
runtime character name and the current configured-local semantic clock.
Logical turns expose `speaker_kind` and `speaker_name`; character turns always
use that exact runtime name. The clock supports absolute-or-omit grounding of
time-dependent operational facts and is distinct from protected source
timestamps.

The scene observer receives only the prior scene, bounded recent semantic
turns, semantic context, and the accepted turn. It returns
`conversation_progress_scene_observation.v2`: one scene relation, one episode
change, factual narrative/flow fields, and established overused moves. It sees
no prior-event ledger, source lineage, capacity operation, or future-response
instruction.

The event reconciler receives only bounded prior event definitions, bounded
source turns, semantic context, current input, and the actual accepted
response. Prior events use temporary handles such as `e1`; source evidence
uses `t1`, `current_input`, and `current_response`. It returns
`conversation_progress_event_observation_batch.v2`:

- exactly one `unchanged` or `changed` row for every supplied prior handle;
- no missing, duplicate, unknown, or extra prior handle;
- changed-event outcome, lifecycle change, relevance, and source handles;
- separately listed new events with concrete non-empty actor, action, and
  object identity.

Every active prior event is mandatory event-recognizer input, including
background-retention rows. Payload fitting may remove older source-turn text,
while the current accepted turn and complete prior ledger remain intact. If
those mandatory fields exceed the event payload cap, the lane returns a typed
pre-call context-limit result and retains the last valid packet.

For an existing changed event, deterministic merge preserves the prior
validated obligation flag, actor, action, object, beneficiary, and
precondition. Every changed or new summary identifies its concrete event
without relying on an ordinal or pronoun.

Neither producer receives or emits database IDs, source-reference objects,
source timestamps, persisted enums, compaction instructions, discard lists,
or persistence structure. Each producer has one attempt and no repair prompt.
An invalid event result fails closed and retains the prior packet. An invalid
scene result preserves the prior validated scene while a valid event batch
remains writable; for the first packet, code copies already accepted
input/stance/surface fields without semantic reinterpretation.

## Deterministic Mapping and Merge

Before the concurrent observer calls, code creates private maps from temporary
handles to:

- exact prior event IDs;
- canonical conversation-row and LLM-trace references;
- exact source timestamps.

Collapsed current inputs retain one protected conversation-row reference per
persisted row, with each row's own storage timestamp. The single
`current_input` semantic handle resolves to that complete ordered row set
before the existing per-event source cap is applied.

A grouped assistant logical turn has one canonical source identity: its
accepted LLM trace. Row-level timestamps are emitted only for a single-row
`row:<id>` turn, so grouping never fabricates per-fragment timestamps from the
turn timestamp. Incidental trace metadata on a user row does not replace that
row's canonical storage identity.

After validation, code resolves the handles and applies the delta:

- code maps `scene_relation=same|related|new` to stored continuity;
- code maps `episode_change=none|paused|finished|resumed` onto stored status;
- every supplied prior event must be explicitly reported; `unchanged` rows
  create no update;
- an emitted existing event preserves its validated obligation direction,
  actor, action, object, beneficiary, and precondition;
- an existing event keeps its prior lifecycle when `lifecycle_change=none`;
- `lifecycle_change=reopened` is accepted only for a prior terminal event;
- code maps the lifecycle classification to `open`, `in_progress`,
  `completed`, `rejected`, or `superseded`;
- code maps `relevance=decision|scene|history` to `decision_critical`,
  `active_scene`, or `background`;
- an updated event keeps its stable ID and original lineage;
- new events receive deterministic UUID5 identities;
- deliberate reopening preserves old and new source lineage;
- string, list, event, source, packet, and prompt caps are enforced;
- stale writes cannot replace newer packet state.

These operations preserve model-authored semantics. They do not classify user
meaning or invent event outcomes.

## State and Projection

The active `conversation_progress.v2` packet stores:

- scope and episode identity;
- status, continuity, and monotonic turn count;
- factual scene narrative, current thread, character stance, user goal,
  blocker, and emotional trajectory;
- bounded lifecycle events with exact provenance;
- established overused response patterns;
- recent turn references and active compacted-block references;
- creation, update, expiry, and purge metadata.

The prompt projection is capped independently from storage. It carries the
factual scene, the most useful active events, bounded logical turns, and block
references. Raw recent history supplies immediate wording and adjacency;
progress supplies semantic continuity.

The evidence projection selects at most eight rows first, then assigns each
selected row a deterministic share of the 1,800-character budget. Every
selected row preserves its summary, state, retention, actor, action, and
object before optional detail is admitted, so an early long event cannot erase
later selected event identities.

For a required-selection cognition call, the outer 24,000-character goal
fitter preserves the required operation and every active progress-event
evidence row intact and model-visible. It reduces optional evidence first and
fails before the model call when those complete facts cannot fit; retaining a
handle while truncating away its event identity is not an accepted projection.
Required operation citations remain mandatory. Cognition cites only progress
rows that materially constrain the current choice and leaves unrelated history
uncited.

V2 consumers, including Recall's active-progress collector, read only scene
fields and `events[].semantic_summary`; deleted V1 list fields have no runtime
reader.

## Deterministic Compaction

Compaction has no LLM call. Code uses its derived and validated event state and
retention labels to select only terminal, non-decision-critical events. It
prefers older `background` events before older `active_scene` events.

An immutable `conversation_progress_block.v1` block contains exact archived
event snapshots, bounded source-turn references, and optional child block IDs.
Its narrative and semantic keys reuse stored model-authored event or block text;
code does not write a new interpretation.

When the active block-reference cap is reached, the four lowest-level active
blocks become children of the new block, ordered oldest-first within the same
level. This keeps the graph balanced and the active packet bounded while
retaining transitive lineage to earlier blocks. The complete candidate graph,
including the new block, must satisfy both the depth and reachable-node caps
before persistence. At 128 reachable blocks, archival pauses and eligible facts
remain in the active packet until its own hard limit is reached; a 129th block
is never published. Conditional historical retrieval expands the active roots
through that exact same-scope graph before vector search, so a child remains
retrievable after it is superseded by its parent. Every reachable block also
receives the episode's sliding expiry refresh.

## Capacity Limits

The policy module is the canonical source of numerical limits. Principal caps
include:

- 24 active events;
- 16 recent turn references;
- 8 active compacted-block references;
- 16,000 characters in the active packet;
- 8 events and 24 turn references per compacted block;
- 4 child blocks per higher-level block;
- 8 block-graph levels and 128 reachable blocks per protected episode graph;
- 4,000 characters across the two continuation projections;
- 8,000 characters in the scene-observer payload;
- 24,000 characters in the event-reconciler payload.

Compaction starts at lower soft limits so the hard active-packet caps remain
available for current and decision-critical state. At full block capacity, the
exact storage frontier is 1,024 archived event snapshots plus up to 24 active
events, subject to the independent packet character cap.

## Failure and Diagnostics

Structural and contract failures are typed and fail closed. Invalid semantic
output, unknown handles, unsupported values, missing exact block references,
or unsafe compaction candidates cannot enter persistence or cognition.
History rows project authored `body_text` together with stored image
descriptions. A legacy row that has neither source is omitted from logical-turn
assembly and increments `incomplete_or_malformed_turn_count`, so one unusable
row cannot block the current chat turn.

Diagnostics contain counts, sizes, compaction level, exact recorder call and
per-owner attempt counts, scene/event dispositions, and write disposition.
They contain no conversation text.

## Storage Lifecycle

The active packet and immutable blocks are scoped by platform, channel, and
global user. Packet writes are monotonic and replacement-based. Blocks are
content-addressed and immutable apart from bounded expiry and supersession
metadata. The default short-term lifetime is 48 hours, with a bounded local
cache used only when it is fresher than durable state. Each successful write
refreshes every block reachable from the active roots, including superseded
children.
