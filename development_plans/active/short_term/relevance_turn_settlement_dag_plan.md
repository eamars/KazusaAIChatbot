# relevance_turn_settlement_dag_plan

## Summary

- Goal: Replace opportunistic group-message coalescing with a bounded
  relevance-owned turn-settlement workflow that lets short follow-up messages
  join the intended logical turn before cognition begins.
- Plan class: high_risk_migration.
- Status: approved.
- Mandatory skills: `development-plan`, `local-LLM-architecture`,
  `no-prepost-user-input`, `debug-llm`, `cjk-safety`, `py-style`, and
  `test-style-and-execution`.
- User decisions already locked:
  - Every active chat path retains a fast frontline relevance agent.
  - A more complex relevance DAG is authorized when the frontline alone cannot
    own filtering, continuation association, readiness, and final response
    judgment reliably.
  - Independent waiting turns remain independent.
  - The earliest eligible turn gets the next cognition opportunity when its
    settled relevance decision is `proceed`.
  - Delivery sign-off is gated by real-LLM tests for the identified user-input
    failure modes.
  - Semantic relevance ownership is the dedicated
    `kazusa_ai_chatbot.relevance` package. The `nodes` package no longer owns
    the frontline or settled relevance agent modules, and no compatibility
    import remains at the old paths.
- Overall cutover strategy: big-bang replacement of group pre-relevance
  coalescing and group burst pruning with one canonical frontline,
  turn-settlement, and settled-relevance path. The existing cognition and
  dialog chain remains downstream of the new cognition claim.
- Highest-risk areas: local-model saturation from per-message filtering,
  false continuation joins, stale relevance decisions at the cognition claim,
  and starvation between frontline intake and settled-turn assessment.
- Evidence-based timing: use a six-second group quiet window and a ten-second
  hard settlement deadline. A settled relevance decision may request the one
  bounded extension from the quiet deadline to the hard deadline.
- Acceptance criteria: the fast route stays within 8,000 input characters and
  256 completion tokens per message; the settled local route stays within
  16,000 input characters, 512 completion tokens, and two calls per logical
  turn; admitted media description is capped at four unique images per turn;
  first-ready ordering and all strict real-LLM gates pass.

## Context

### Database Evidence

The read-only conversation-history pull is retained at
`test_artifacts/diagnostics/kazusa_group_followup_evidence_summary.json`.
It covers QQ group traffic from 2026-05-11T20:00:00Z through
2026-07-15T09:00:00Z across 18 channels and 428,959 deduplicated rows.

| Observation | Evidence | Design consequence |
|---|---:|---|
| Structured user messages mentioning the bot | 1,760 | The denominator is large enough to characterize the pattern. |
| Same-sender follow-up before an assistant row within 3 seconds | 4 (0.23%) | A three-second fixed delay captures only a small slice. |
| Same-sender follow-up before an assistant row within 5 seconds | 13 (0.74%) | The pattern is uncommon but real and user-visible. |
| Same-sender follow-up before an assistant row within 10 seconds | 43 (2.44%) | A bounded semantic extension has measurable value. |
| Follow-ups within 5 seconds aimed at another target | 3 of 13 | Timing plus author identity is unsafe as a merge rule. |
| Follow-ups within 10 seconds containing an attachment | 12 | Attachment-only continuations are a first-class case. |

Representative evidence establishes four distinct facts:

1. `@杏山千纱 ？` followed by `吃了吗` after 1.587 seconds produced two
   Kazusa replies, strongly indicating one fragmented user turn entered
   cognition twice.
2. A bare mention followed by `告诉我这是怎么做的` after 8.414 seconds shows
   that a three-to-five-second hard cutoff is insufficient.
3. A complete-looking hardware question followed by an image after 4.779
   seconds, and a fleet question followed by an image after 5.08 seconds,
   show that apparent textual completeness cannot safely trigger immediate
   cognition.
4. `@杏山千纱 你是bot吗？` followed 4.673 seconds later by the same author
   asking another user the same question proves that same-author timing alone
   can create a false join.

The original observation is therefore directionally correct: Kazusa can enter
cognition before a user's follow-up arrives. Two corrections shape the target:

- A hard three-to-five-second cooldown leaves documented continuations outside
  the window. Six seconds covers the observed 5.08-second boundary with modest
  delivery slack, while a relevance-approved extension covers the 8.414-second
  incomplete summon.
- A cooldown cannot decide which message belongs to which turn. Semantic
  association must use typed addressee and reply evidence, author, channel,
  chronological fragments, and topic continuity.

### Current Runtime Findings

The current path is:

```text
adapter request
  -> ChatInputQueue.wait_for_next()
  -> adjacency-only private/group coalescing
  -> group burst pruning
  -> one global service worker
  -> multimedia descriptor
  -> persona relevance agent (boolean should_respond)
  -> cognition and dialog
```

The relevant source behavior is:

- `chat_input_queue.py` wakes and dequeues as soon as the queue becomes
  non-empty.
- Addressed group coalescing sees only queued, adjacent, same-author items.
  Another author's interleaved message breaks the run.
- The current 120-second group gap is an eligibility bound for items already
  waiting together; it creates no idle-time observation window.
- Group pruning can remove an unaddressed follow-up before any LLM sees whether
  it continues an addressed turn.
- `_chat_input_worker()` awaits the complete processing of one dequeued item
  before taking the next item. A sleep placed inside the current relevance
  node would therefore stall all chat intake and all cognition scopes.
- `persona_relevance_agent.py` can return only a final boolean response choice.
  It has no continuation target, wait action, deadline, or turn version.
- The current graph runs the media descriptor before relevance, adding avoidable
  work for messages a lightweight frontline could discard.

### Final System Shape

```text
incoming message
  -> persist typed message/envelope
  -> FIFO RELEVANCE EXECUTOR (one in-flight call)
  -> FRONTLINE RELEVANCE (small bounded question on the existing route)
       discard -------------------------------> silent completion
       start/append
          -> deterministic pending-turn store
          -> six-second quiet timer / ten-second hard deadline
          -> global ready heap ordered by (eligible_at, leader_sequence)
          -> media description for accepted fragments
          -> FIFO RELEVANCE EXECUTOR
          -> SETTLED RELEVANCE (persona-aware local route)
               ignore ------------------------> silent completion
               wait --------------------------> one bounded reschedule
               proceed
                  -> atomic version check + RUNNING claim
                  -> existing cognition -> dialog -> persistence
```

The workflow has two semantic agents because the roles require different
context and latency profiles:

- The frontline relevance agent answers one intake-routing question:
  `discard`, `start`, or `append` to a supplied open turn. It uses compact
  typed message evidence and open-turn candidates. It omits mood, affinity,
  wide history, user-engagement database reads, and media description.
- The settled relevance agent answers the character-level question after the
  turn has stabilized: `ignore`, `proceed`, or `wait`. It owns the current
  relevance fields such as reply anchoring, channel topic, and indirect speech
  context.
- Both semantic agents live in `kazusa_ai_chatbot.relevance` behind its public
  package facade. `brain_service.turn_settlement` remains the deterministic
  lifecycle owner and imports only the relevance package contract.
- Deterministic code owns clocks, state transitions, heap ordering, version
  checks, prompt bounds, persistence identity, and the single cognition claim.

The six-to-ten-second wait lives in the settlement scheduler. It never runs as
a sleep inside an LLM node or inside the cognition worker.

## Mandatory Skills

- `development-plan`: load before approval, execution, lifecycle updates, or
  archival of this plan.
- `local-LLM-architecture`: load before changing either relevance prompt,
  structured output, graph routing, call budget, or context projection.
- `no-prepost-user-input`: load before changing how text, addressees, replies,
  attachments, or corrections influence `discard`, `start`, `append`,
  `ignore`, `proceed`, or `wait`.
- `debug-llm`: load before creating, running, or judging the real-LLM gates and
  their review artifacts.
- `cjk-safety`: load before creating, moving, or editing Python prompts or
  tests containing CJK, and before printing live model text on Windows.
- `py-style`: load before editing or reviewing Python.
- `test-style-and-execution`: load before creating, changing, or running any
  deterministic or live-LLM test.

## Mandatory Rules

- Keep the frontline agent as the first semantic stage for every active chat
  candidate. `listen_only` remains the explicit debug bypass.
- Keep the two LLM questions narrow: frontline chooses semantic intake
  disposition; settled relevance chooses character response disposition.
- Use the existing `RELEVANCE_AGENT_LLM` route for both semantic stages. The
  frontline remains a separate bounded question with stage-local completion
  and input limits; no new deployment route variables are added.
- Keep thinking disabled for both stages. Render stable contract text in static
  system prompts and current-run evidence in human messages.
- Follow the CJK-safe source contract for prompt and test strings: preserve
  exact existing bytes when moving CJK text, use safe string delimiters for
  new text, specify UTF-8 for file I/O and live artifact output, and syntax-
  check each touched Python file immediately after editing it.
- Enforce total rendered-input caps of 8,000 characters for frontline and
  16,000 for settled relevance. Use one rendered character per token as the
  planning envelope for mixed CJK/English/JSON; the measured character caps
  remain authoritative when a route exposes no compatible tokenizer.
- Enforce completion caps of 256 tokens for frontline and 512 for settled
  relevance. Keep each schema's maximum allowed free-form content plus JSON
  structure within its completion cap under the same one-character-per-token
  planning envelope. Permit one frontline call per inbound logical message and
  at most two settled calls per assembled turn.
- Serialize frontline and settled calls through one FIFO relevance-work
  executor with one in-flight call. A timer enqueues work and holds no model,
  frontline, or cognition resource while waiting.
- Disable LLM-backed JSON repair for both relevance stages. Deterministic JSON
  repair and structural validation may run, while malformed output triggers no
  additional model call.
- Describe at most four unique image payloads per assembled turn, reusing the
  existing prompt-attachment bound. Preserve all attachments for audit, select
  the opening image plus the newest remaining images for description, expose
  `additional_media_present` when payloads overflow, cache selected results,
  and use deterministic-only parsing so descriptor repair cannot add calls.
- Give models semantic descriptors and bounded slot labels. Keep raw platform
  IDs, turn IDs, timestamps, deadline seconds, base64 media, queue counters,
  futures, handles, and numeric noise telemetry outside prompts.
- Treat typed addressee and reply fields as authoritative platform evidence.
  Preserve each fragment's target evidence separately through settlement.
- Express direct-address conservatism, continuation meaning, topic breaks, and
  response relevance in the prompts and schemas. Deterministic code validates
  shape/slots and faithfully applies a valid model action; it never keyword-
  classifies user text or rewrites one valid semantic action into another.
- Preserve message arrival chronology inside an assembled turn.
- Keep one global cognition lane. Open settlement timers and the frontline lane
  remain independent of that lane.
- Order ready turns by `(eligible_at, leader_sequence)`. A newly appended
  fragment recalculates only its target turn's eligibility.
- Use the ten-second hard deadline from the message that opens the turn. Later
  input starts or joins another eligible turn according to the frontline.
- Permit one semantic `wait` extension per logical turn. At the hard deadline,
  the settled agent chooses `ignore` or `proceed`.
- Re-run settled relevance against the latest assembled turn after a version
  invalidation. A first assessment invalidated by user input moves directly to
  the hard-deadline assessment, bounding settled relevance to two calls.
- Preserve the current downstream character judgment goal. `proceed` means the
  turn has earned cognition entry; cognition continues to own stance and
  dialog goals.
- Persist every inbound message once and carry every accepted fragment's
  platform message ID and conversation row ID into active-turn identity.
- Run deterministic tests in regular batches. Run each real-LLM case as its own
  pytest invocation and inspect its output before starting the next case.
- Require raw JSON trace evidence plus an agent-authored Markdown quality review
  for the live gates. Test-generated prose cannot satisfy the review gate.
- Treat an unavailable LLM endpoint, a skipped live gate, or an uninspected
  artifact as an incomplete delivery gate.
- Reread this complete plan after automatic context compaction and before each
  major execution checkpoint.
- After signing off a major checklist stage, reread this complete plan before
  starting the next stage.
- Run the Independent Code Review gate and record it in Execution Evidence
  before completion, lifecycle changes, merge, or final sign-off.
- Use parent-led native subagent execution for implementation and code review;
  this user-requested independent plan review is performed by the parent alone.

## Must Do

- Add the compact frontline relevance agent and strict `discard/start/append`
  contract.
- Refactor the existing persona relevance stage into the settled relevance
  decision with strict `ignore/proceed/wait` output.
- Reuse the existing `RELEVANCE_AGENT_LLM` route for frontline and settled
  relevance; apply the context, completion, thinking, and concurrency budgets
  in Mandatory Rules without adding a second route configuration.
- Add relevance-specific prompt projection that bounds candidate slots,
  preludes, history, assembled fragments, and free-form output lengths while
  preserving current-message and per-fragment typed-target evidence.
- Add a deterministic-only parsing mode to the existing JSON parser and use it
  for both relevance stages so their call counts stay hard-bounded.
- Add the process-local pending-turn lifecycle, deadline, version, global-ready
  heap, and atomic cognition-claim contracts.
- Split fast intake/frontline work from the single cognition lane; replace group
  adjacency coalescing and group threshold pruning with this path.
- Preserve current private-message timing and cognition behavior while routing
  its surviving logical input through the frontline and immediate-ready path.
- Run media after frontline admission, retain its per-fragment description, and
  load fresh channel history for settled relevance. Bound description to four
  selected unique images per turn and preserve an overflow descriptor.
- Complete request futures, pipeline handles, platform IDs, and row IDs under
  one documented assembled-turn rule.
- Update brain-service, message-envelope, runtime-coordination, nodes, and
  relevance READMEs to describe the final path and ownership boundaries.
- Replace superseded tests, add deterministic state/ordering coverage, run the
  strict real-LLM matrix, capture prompt/output sizes and durations, author its
  review, and complete independent review.

## Deferred

The following concerns remain outside this relevance-and-sequence plan:

- Cross-process coordination and crash recovery for settlement timers.
- Adapter transport/delivery retry and database/queue failure policy.
- Global GPU scheduling across cognition, dialog, RAG, vision, and relevance
  routes; this plan bounds only the two relevance stages.
- Platform message deletion or recall events absent from the typed chat input
  contract.
- Changes to RAG, cognition, dialog, memory, reflection, scheduler, or
  background-work semantics.
- A general-purpose conversation-thread model beyond the bounded live
  relevance window.
- Long-delay capture beyond ten seconds and per-channel parallel cognition.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Group queue/relevance | bigbang | Replace adjacency coalescing, threshold pruning, metadata semantic skips, and boolean chat relevance together. |
| LLM routes/prompts | bigbang | Use the existing `RELEVANCE_AGENT_LLM` route for both relevance roles and retain no old group prompt path. |
| Turn state/graph | bigbang | Move to pending versions, response action, and atomic claim in the same cutover. |
| Tests/docs | bigbang | Replace old group expectations and document only the final runtime path. |
| Private messages | retained outside cutover | Preserve current coalescing and immediate timing; its surviving logical input enters frontline once. |

Cutover enforcement:

- Update queue, service, graph, agents, state, tests, and docs cohesively.
- Remove superseded symbols after replacement deterministic tests pass and
  before live delivery gates.
- Keep one decision vocabulary and configuration per stage. Runtime aliases,
  fallback mappers, feature flags, and dual group paths remain forbidden.
- A policy change requires user approval before production implementation.

## Target State

### 1. Frontline Intake And Workload

One process-local relevance-work executor consumes frontline and settled work
in FIFO enqueue order with one in-flight relevance call. The intake path
persists and resolves the typed envelope, then enqueues a frontline call. A
frontline call receives only:

- the current message, clipped head-and-tail to 2,000 characters;
- semantic target labels such as `character`, `other_participant`, `multiple`,
  or `none`, plus semantic reply and attachment-kind labels;
- up to three same-platform/channel/author open turns, exposed as
  `open_1..open_3`, each with opening/latest excerpts and typed-target labels;
- up to two recent silent same-author preludes, exposed as
  `prelude_1..prelude_2`;
- a short latest-bot-continuity descriptor when it exists.

The frontline system prompt is at most 2,200 characters. Candidate projections
use at most 1,800 characters, preludes 400, and remaining JSON, continuity, and
semantic context 1,600, keeping both messages within the 8,000-character hard
cap.

The frontline returns:

```json
{
  "intake_action": "discard | start | append",
  "append_target": "none | open_1 | open_2 | open_3",
  "prelude_targets": ["prelude_1", "prelude_2"],
  "reason": "at most 80 characters"
}
```

`discard` completes the persisted input silently. `start` creates a turn and
may promote supplied prelude slots. `append` selects exactly one supplied open
slot. Deterministic code maps slots to internal IDs, validates cardinality, and
applies the valid state change without semantic rewriting. The prompt requires
direct typed addresses/replies to choose `start` or `append`; it also teaches
conservative admission for attachment-only and ambiguous short messages near
an open same-author turn.

The coordinator keeps at most three open turns per author/scope. When a valid
`start` would create a fourth, it freezes the oldest open turn for settled
assessment before creating the new one. The frontline uses the existing
`RELEVANCE_AGENT_LLM` route, temperature zero, thinking off, and a 256-token
completion cap.

### 2. Pending Turn Contract

Add one internal `PendingChatTurn` contract with these owned fields:

```text
turn_id
scope = (platform, platform_channel_id, channel_type)
author_platform_user_id
author_global_user_id
leader_sequence
response_owner_sequence
fragments[]
status = SETTLING | ASSESSMENT_READY | ASSESSING | RUNNING | DONE
created_at_monotonic
last_fragment_at_monotonic
eligible_at_monotonic
hard_deadline_monotonic
version
settled_assessment_count
wait_used
```

Each fragment retains:

- brain-local arrival sequence;
- enqueue-time monotonic timestamp;
- platform message ID;
- conversation row ID;
- storage UTC timestamp;
- resolved typed envelope;
- body text;
- attachments and any retained media description;
- its original request future and pipeline coordination handle.

`leader_sequence` is the message that opens settlement and controls ready-order
ties, including when an older silent prelude is promoted. The opening message
also owns the final response. An appended message completes empty after its row
is persisted and attached. `discard` and `ignore` complete their owners empty;
`proceed` returns the final `ChatResponse` only through the response owner.

The prompt-facing assembled turn preserves fragments as an ordered list. A
newline-joined convenience rendering may be provided, while typed targets stay
per-fragment so a later message aimed at another user cannot contaminate the
first fragment's addressee.

### 3. Timing And Lifecycle

For an admitted group turn:

```text
created_at = opening message time
hard_deadline = created_at + 10 seconds
eligible_at = min(last accepted fragment enqueue time + 6 seconds, hard_deadline)
```

- `start` creates `SETTLING` and schedules its eligibility token.
- `append` is available through the hard deadline while the turn has not
  entered `RUNNING`. It increments `version`, preserves chronology, and
  recalculates `eligible_at`.
- Heap entries carry turn ID and version. Superseded entries become inert when
  their version differs from the live turn.
- At `eligible_at`, the scheduler changes the earliest eligible turn to
  `ASSESSMENT_READY`.
- The cognition lane runs media description and settled relevance for that
  turn.
- `wait` sets `wait_used=true`, schedules the turn at `hard_deadline`, and lets
  the lane inspect the next ready turn.
- `ignore` closes the turn silently.
- `proceed` enters the deterministic claim node. A matching version changes the
  turn atomically to `RUNNING` and continues the existing cognition graph.
- A version mismatch returns the turn to settlement and schedules its final
  assessment at the hard deadline.
- The hard-deadline assessment freezes after the frontline has applied every
  input enqueued at or before that deadline. Later-enqueued input is routed as
  a new or different open turn.

Private input follows its current immediate-ready timing. The settled agent's
`wait` action is available only for group turns.

### 4. Ready Ordering And Interleaving

The ready heap is global because the cognition lane is global. Its stable key
is:

```text
(eligible_at_monotonic, leader_sequence, turn_id, version)
```

This gives the requested behavior:

- A waiting turn releases all scheduler and cognition ownership.
- B can become ready and enter cognition while A remains inside its settlement
  window.
- An A follow-up changes A's deadline only. It has no effect on B's turn.
- When multiple turns are ready, the earliest eligibility wins; arrival
  sequence resolves an exact tie.
- If the earliest turn returns `ignore` or `wait`, the lane immediately checks
  the next ready turn.
- Once a turn claims `RUNNING`, cognition is non-preemptive. A later message is
  a new relevance candidate with the completed/running dialog context visible
  at its own settled assessment.

Example:

```text
t=0.0  A1 starts turn A; eligible at 6.0
t=0.5  B1 starts turn B; eligible at 6.5
t=2.0  A2 appends to A; A moves to eligible at 8.0
t=6.5  B reaches settled relevance first and may claim cognition
t=8.0  A reaches settled relevance after B, subject to cognition availability
```

Interleaving is therefore an ordering input, rather than a reason to join or
split messages. The frontline uses semantic and typed evidence to associate
the fragments.

### 5. Settled Relevance Contract

The persona-aware settled agent receives a bounded semantic projection:

- static system contract capped at 4,000 characters;
- assembled turn text and per-fragment target/reply labels capped at 6,000;
- at most the existing ten fresh history rows, clipped to 4,000 total;
- mood, relationship, group-attention, bot-continuity, and at most two compact
  engagement guidelines capped with JSON overhead at 2,000.

Projection retains the opening fragment, newest fragments, every fragment's
target/reply labels, and newest correction text before older body excerpts. It
uses `earlier_context_present` as a semantic descriptor when older body text is
clipped. Raw timestamps, IDs, counters, deadlines, and queue state stay out of
the payload. Total rendered input is capped at 16,000 characters. It returns:

```json
{
  "response_action": "ignore | proceed | wait",
  "reason_to_respond": "at most 180 characters",
  "use_reply_feature": false,
  "channel_topic": "at most 60 characters",
  "indirect_speech_context": "at most 100 characters"
}
```

Contract rules:

- `ignore` means the assembled turn gives Kazusa no grounded reason to speak.
- `proceed` means the assembled turn is stable enough and relevant enough to
  enter cognition.
- `wait` means the turn still appears incomplete and the bounded extension is
  available.
- The runtime supplies the semantic observation status
  `more_time_available | observation_complete`. In the complete phase, valid
  decisions are `ignore` and `proceed`.
- The settled agent judges whether another user's intervening answer made a
  response redundant, whether corrections supersede earlier fragments, and
  whether the final addressed target remains Kazusa.
- The graph projects a claimed `proceed` into the existing downstream
  `should_respond=true` state. `ignore` and `wait` end before cognition.

The stage uses `RELEVANCE_AGENT_LLM`, temperature zero, thinking off, and a
512-token completion cap. Media descriptions are cached on accepted fragments
so a wait assessment repeats neither the vision call nor its payload growth.
At most four unique image payloads are described for one assembled turn: the
opening image is retained first, then the newest remaining images. All
attachment records remain persisted, and `additional_media_present` tells the
settled agent that further undescribed media exists. Descriptor parsing uses
the deterministic-only parser mode, so one selected cache miss creates exactly
one vision-model call.

## Contracts And Data Shapes

### Minimal LLM Contracts

| Stage | Smallest semantic question | Required input | Rejected input |
|---|---|---|---|
| Frontline | Does this message clearly deserve silence, start a candidate turn, or continue one supplied candidate? | Current message, semantic target/reply/media labels, <=3 open slots, <=2 prelude slots, latest bot continuity | Mood, affinity, engagement DB output, wide history, media body, raw IDs/times/counters |
| Settled | Given the stabilized user turn and current scene, should the character ignore, proceed, or use the available observation extension? | Bounded assembled turn, media descriptions, <=10 fresh history rows, semantic scene/relationship descriptors | Queue state, futures, handles, numeric deadlines/noise, base64, raw platform/turn IDs |

Deterministic code owns persistence, slot-to-ID mapping, limits, truncation,
work ordering, clocks, retries, response delivery, and graph claims. Neither
agent receives permission to change those mechanics.

### Coordinator Public Interface

`TurnSettlementCoordinator` is the sole service-facing entrypoint. Its public
constructor receives the two stage callables and monotonic clock; its public
operations are:

```python
async def evaluate_frontline(state: FrontlineState) -> FrontlineDecision: ...

async def apply_frontline_decision(
    fragment: PersistedChatFragment,
    decision: FrontlineDecision,
) -> FrontlineOutcome: ...

async def wait_for_assessment_ready() -> AssessmentLease: ...

async def evaluate_settled(
    lease: AssessmentLease,
    state: SettledRelevanceState,
) -> SettledRelevanceDecision: ...

async def apply_settled_decision(
    lease: AssessmentLease,
    decision: SettledRelevanceDecision,
) -> SettlementOutcome: ...

async def claim_for_cognition(turn_id: str, version: int) -> bool: ...
```

Service and graph code depend on these operations and typed results. Pending
maps, heap entries, timers, slot mappings, and relevance-work queue remain
module-private. `FrontlineDecision` uses only the bounded slot vocabulary;
internal IDs appear only after deterministic mapping.

### Relevance Package Public Interface

`kazusa_ai_chatbot.relevance` is the public semantic-agent facade. It exports
`frontline_relevance_agent`, `relevance_agent`, the two decision TypedDicts,
the two structural validators, and the bounded message builders used by
contract tests. Runtime callers import these public symbols from the facade.
Prompt constants, LLM instances, route configs, projections, and trace details
remain private to `relevance/frontline_relevance_agent.py` and
`relevance/persona_relevance_agent.py`.

The package owns semantic judgment only. It does not own persistence, queue
ordering, timers, deadlines, slot-to-ID mapping, media-cache mutation,
response delivery, or cognition claims. No forwarding module remains under
`kazusa_ai_chatbot.nodes`.

## LLM Call And Context Budget

The planning envelope uses one rendered character per token for mixed CJK,
English, and JSON. It intentionally avoids relying on an unavailable or
model-mismatched tokenizer. Every hard character cap covers the combined
system and human messages before invocation.

With every free-form field filled to its allowed maximum and compact JSON
separators, the frontline schema renders to 187 characters and the settled
schema to 466. Both remain inside their 256/512 completion envelopes before
normal model brevity is considered.

| Stage | Before | After | Response-path load and blocking |
|---|---|---|---|
| Frontline | No separate call | `RELEVANCE_AGENT_LLM`; <=8,000 chars (<=8,000-token planning envelope), <=256 completion tokens, thinking off | One call per inbound logical message; serialized by relevance executor |
| Settled relevance | `RELEVANCE_AGENT_LLM`; up to 10 history rows, no explicit rendered-char cap, shared default up to 8,192 completion tokens, possible JSON-repair LLM fallback | `RELEVANCE_AGENT_LLM`; <=16,000 chars (<=16,000-token planning envelope), <=512 completion tokens, thinking off, deterministic-only JSON repair | One normal call per admitted turn; two maximum after `wait` or stale-version invalidation |
| Vision description | Runs before current relevance with no per-turn descriptor-call cap and possible JSON-repair fallback | Existing `VISION_DESCRIPTOR_LLM`; after frontline admission, <=4 selected unique images, one call per cache miss, deterministic-only parsing, cached | Zero for text-only; <=4 descriptor calls per assembled turn; no repeat on settled re-assessment |
| Cognition/RAG/dialog | Existing downstream calls | Unchanged | Begins only after atomic `proceed`; outside this plan's call budget |

| Message case | Frontline | Settled | Vision | Maximum new relevance-family calls |
|---|---:|---:|---:|---:|
| Clearly irrelevant single message | 1 | 0 | 0 | 1 |
| Relevant text message | 1 | 1 | 0 | 2 |
| N-fragment text turn | N | 1 | 0 | N+1 |
| N-fragment turn with unique media | N | 1 | <=4 selected cache misses | N+1 relevance calls, <=4 vision calls |
| Semantic wait or stale first assessment | N | 2 | cached, <=4 total | N+2 relevance calls, <=4 vision calls |

The relevance executor permits one in-flight frontline or settled call. FIFO
enqueue order prevents later input from indefinitely overtaking an already
ready settled turn. The executor releases its model slot before cognition
begins. Settlement timers hold no slot. Both semantic stages use the existing
route, and the serialized executor prevents their calls from overlapping while
the stage-local frontline cap keeps its high-volume prompt bounded.

Both relevance stages and the admitted media descriptor use deterministic JSON
repair only. The JSON-repair LLM call count is therefore zero for this
workflow. Context overflow is handled before the call by the fixed projections
above; the runtime records `prompt_chars`,
`output_chars`, and duration in protected traces. A rendered prompt over its
cap is a deterministic test failure and cannot be sent. A structurally invalid
model result follows the current relevance fail-closed empty-response behavior,
is traced as invalid, and triggers no semantic rewrite or retry.

The user's authorization for the relevance DAG covers one compact frontline
call per inbound message and one optional settled re-assessment. This plan adds
no model-native tool loop and no cognition, RAG, or dialog call.

## Design Decisions

| Decision | Locked result | Reason |
|---|---|---|
| Semantic stages | Compact frontline plus persona-aware settled relevance | Clear irrelevant traffic avoids full context while final judgment remains character-grounded. |
| Model route | Existing `RELEVANCE_AGENT_LLM` route for both stages | The user requires one current route; stage-local caps keep frontline work bounded without adding deployment configuration. |
| Relevance concurrency | One FIFO executor, one in-flight call | Bounds local-model memory/load and prevents either relevance role from overtaking queued work indefinitely. |
| Prompt projection | Slot labels and semantic descriptors under 8k/16k character caps | Weak local models avoid raw IDs, telemetry, unbounded candidates, and oversized history. |
| Parse recovery | Deterministic repair only | Preserves the hard call budget and avoids a hidden JSON-repair model invocation. |
| Initial observation | Six seconds for every admitted group turn | Future images and clarifications cannot be predicted from complete-looking first text. |
| Extension | One `wait`, capped at ten seconds from the opening message | Covers the observed 8.414-second summon while bounding latency and calls. |
| Ownership | LLM associates meaning; code owns timing/state | Typed meaning stays semantic and sequence invariants stay deterministic. |
| Semantic module boundary | Move both relevance agents into `kazusa_ai_chatbot.relevance` with a public facade | The two agents form one semantic subsystem; `nodes` remains focused on persona perception/orchestration, while `brain_service.turn_settlement` retains deterministic lifecycle ownership. |
| Scheduling | Global first-ready heap feeding one cognition lane | An incomplete A turn cannot delay independent B solely because they share a channel. |
| Claim | Versioned, atomic, and non-preemptive after `RUNNING` | Boundary follow-ups invalidate stale relevance while active cognition stays coherent. |
| Stage order | Frontline before group discard and media | An unaddressed continuation reaches semantic association and irrelevant traffic stays cheap. |
| Final context | Fresh history at settled assessment | Intervening answers and topic movement affect relevance without joining another user's rows into the turn. |

## Failure Modes And Solutions

| ID | User-input or sequence failure | Required solution | Deterministic proof | Real-LLM gate |
|---|---|---|---|---|
| F01 | Bare mention or partial phrase enters cognition before the request arrives | Admit conservatively, use six-second quiet time, allow one relevance-approved extension to ten seconds | Deadline and extension state test | L02 |
| F02 | Complete-looking text is followed by an image or clarification | Apply the initial quiet window to every admitted group turn; append semantic supplement | Version/deadline/media retention test | L03 |
| F03 | Attachment-only follow-up has little text signal | Give frontline attachment presence/media kind and open same-author candidates | Ordered fragment/media aggregation test | L03 |
| F04 | A1, B1, A2 interleave in one channel | Scope open candidates by author and let A2 append to A while B stays independent | Interleaved heap/state test | L04 |
| F05 | Same author starts an unrelated topic during settlement | Frontline chooses `start` when topic and target evidence break continuity | Multiple-open-turn state test | L05 |
| F06 | Same author addresses another user within the timing window | Preserve per-fragment typed targets and select `discard` or `start`, never time-merge | Target-preservation test | L06 |
| F07 | One message addresses Kazusa and another participant | Preserve all typed addressees and let settled relevance judge Kazusa's listener role | Multi-addressee projection test | L07 |
| F08 | User sends content first and tags Kazusa afterward | Supply recent silent same-author messages; allow `start` to promote semantic prelude IDs | Prelude eligibility/identity test | L08 |
| F09 | Three or more fragments include correction, replacement, or retraction | Preserve chronological fragments; settled relevance treats later correction as current intent | Fragment-order/version test | L09 |
| F10 | Another user answers the pending question before Kazusa's turn | Keep the other user's row out of A's turn while loading fresh history for settled relevance | History/turn separation test | L10 |
| F11 | Follow-up arrives while media or settled relevance is running | Increment version; atomic claim rejects stale result; final assessment uses latest turn | Controlled concurrency/version test | L11 |
| F12 | Follow-up arrives after cognition has claimed RUNNING | Route as a new candidate and expose prior/running context through normal history | RUNNING append rejection/new-turn test | L12 |
| F13 | Two independent turns become ready together | Order by eligibility then leader sequence; earliest `proceed` receives the cognition claim | Tie-order test | L13 |
| F14 | Earliest ready turn returns `wait` or `ignore` | Release the lane immediately and assess the next ready turn | Ready-heap fall-through test | L10, L19 |
| F15 | Continuous fragments keep resetting the timer | Clamp every update to the ten-second hard deadline and freeze the fragment set for final assessment | Fake-clock hard-cap test | L14 |
| F16 | Settled relevance emits repeated `wait` at the deadline | Supply `observation_status=observation_complete`; validate the final contract as `ignore/proceed` | Output-contract test | L14 |
| F17 | Cross-channel or cross-author content looks semantically similar | Supply only matching scope/author open-turn candidates | Candidate-scope isolation test | L15 |
| F18 | Frontline false-negative discards an ambiguous continuation | Make discard conservative; directly addressed, bot-reply, attachment-near-pending, and short ambiguous pending continuations remain admitted | Prompt/payload contract test | L01, L02, L03 |
| F19 | Current group burst pruning removes A2 before relevance | Retire group pre-frontline threshold pruning; frontline owns active group discard | Queue path test | L04 |
| F20 | A wait inside the current worker blocks all other turns | Keep timers in the pending-turn coordinator and keep the frontline lane independent from cognition | Worker non-blocking event test | L04, L19 |
| F21 | A message arrives at the settlement boundary | Compare live version in the atomic claim; an applied boundary append invalidates the decision and moves to final assessment | Boundary race test | L11 |
| F22 | The hard deadline splits a genuinely late continuation | Treat the later input as a new relevance candidate with prior-turn history; record this as the bounded residual behavior | Late-arrival test | L12 |
| F23 | Busy input queues many frontline calls and starves a ready settled turn | Serialize both roles in FIFO relevance-work order with one in-flight call | Controlled-latency FIFO test | L18 |
| F24 | One author starts more simultaneous topics than the local model can compare reliably | Expose at most three open slots and freeze the oldest before a fourth start | Open-turn cardinality test | L16 |
| F25 | Long or numerous fragments overflow the local model context or hide the latest correction | Apply fixed head/tail/latest projections while preserving all target/reply labels | Rendered-prompt cap and retention test | L17 |
| F26 | One user sends enough images across fragments to fan out vision calls and crowd the settled model | Persist all media; describe the opening plus newest images up to four, expose overflow semantically, cache results, and disable repair-model calls | Descriptor selection/call-count/cache test | L20 |

## Change Surface

Target ownership boundary: `kazusa_ai_chatbot.relevance` owns the two semantic
relevance calls and their structural contracts. `brain_service.turn_settlement`
owns deterministic turn assembly, relevance-work serialization, deadlines,
ready order, and cognition claims. Node modules own persona perception,
orchestration, cognition, and dialog; they do not own relevance agents.

### Delete

- Delete the group coalescing constant/branch, group threshold-pruning path,
  and metadata-only semantic short-circuits from the active group route after
  their replacement tests pass. Delete no unrelated module file.
- `src/kazusa_ai_chatbot/nodes/frontline_relevance_agent.py` and
  `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py`: remove the old
  ownership paths after the byte-preserving move into the relevance package;
  add no compatibility aliases or forwarding modules.

### Modify

- `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py` and
  `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`: keep the
  existing semantic contracts and private prompt/model/validation ownership
  after the package move; runtime callers use the package facade.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`:
  apply four-image settled-turn selection and deterministic-only parsing while
  retaining the existing vision prompt and route.
- `src/kazusa_ai_chatbot/chat_input_queue.py`: send active group items to
  frontline before semantic discard and remove group adjacency coalescing.
- `src/kazusa_ai_chatbot/service.py`: persist fragments, call the coordinator,
  separate relevance intake from cognition, and resolve futures/handles.
- `src/kazusa_ai_chatbot/brain_service/graph.py`: route settled action through
  the deterministic versioned cognition-claim node.
- `src/kazusa_ai_chatbot/state.py`: add response action, semantic observation
  status, turn/version, and claim result.
- `src/kazusa_ai_chatbot/config.py`: reuse the existing
  `RELEVANCE_AGENT_LLM_*` route settings and add only the hard stage
  completion/context caps; lower settled default and hard completion cap to
  512.
- `src/kazusa_ai_chatbot/utils.py`: add an explicit deterministic-only mode to
  `parse_llm_json_output` while retaining current behavior for existing callers.
- `tests/test_frontline_relevance_agent.py`,
  `tests/test_persona_relevance_agent.py`, `tests/test_service_input_queue.py`,
  `tests/test_msg_decontexualizer.py`, and `tests/test_utils.py`: replace old
  contracts and prove bounded parsing, descriptor calls, queue, prompt, and
  private regressions.
- `README.md`, `docs/HOWTO.md`, and the brain-service, message-envelope,
  runtime-coordination, nodes, relevance, and LLM-interface READMEs: document
  route, budget, settlement, and ownership changes.

### Create

- `src/kazusa_ai_chatbot/relevance/__init__.py`: public semantic-agent facade
  exposing the approved relevance contracts and entrypoints.
- `src/kazusa_ai_chatbot/relevance/README.md`: relevance ICD documenting the
  interface and ownership boundary.
- `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py`: moved
  prompt, shared route config, bounded projector, parser call, and trace for
  `discard/start/append`.
- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`: moved settled
  prompt, shared route config, bounded projector, parser call, and trace for
  `ignore/proceed/wait`.
- `src/kazusa_ai_chatbot/brain_service/turn_settlement.py`: expose only the
  public `TurnSettlementCoordinator` interface named above; keep pending
  records, FIFO executor, timers, heap, versions, and claim mechanics private.
- `tests/test_frontline_relevance_agent.py`: prompt/schema/budget node tests.
- `tests/test_relevance_turn_settlement.py`: fake-clock lifecycle, workload
  serialization, cardinality, interleaving, and heap tests.
- `tests/test_relevance_turn_settlement_graph.py`: patched-LLM graph handoff and
  claim routing tests.
- `tests/test_relevance_turn_settlement_live_llm.py`: L01-L20 real-model gates.

### Keep

- Keep private coalescing/timing, typed adapter contracts, RAG, cognition,
  dialog, consolidation, reflection, background work, and delivery behavior.
- Keep existing `RELEVANCE_AGENT_LLM_*` route identity for both relevance
  stages.

## Overdesign Guardrail

- Actual problem: a group user's fragmented turn can enter cognition too early,
  while timing-only coalescing can join the wrong target and a high-volume LLM
  filter can overload the local final model.
- Minimal change: two bounded relevance prompts plus one process-local
  coordinator and FIFO relevance executor, retaining one cognition lane.
- Ownership boundaries: agents own semantic intake/response judgment;
  deterministic code owns candidate slots, prompt caps, parsing mode, clocks,
  work order, persistence, versions, and claims.
- Rejected complexity: micro-batched classifications, more relevance agents,
  prompt retries, model-native tools, per-channel cognition, distributed turn
  state, global GPU scheduling, compatibility routes, and alternate group paths.
- Evidence threshold: add any rejected mechanism only after retained traces or
  a measured workload gate proves the 8k/16k projections, one-call/two-call
  limits, or FIFO executor cannot satisfy the approved behavior.

## Agent Autonomy Boundaries

- Execution agents may choose local implementation mechanics only when they
  preserve every schema, route, cap, state transition, file boundary, and gate
  in this plan.
- Execution agents must keep one canonical group path and cannot add helper
  agents, retries, fallback routes, compatibility shims, alternate prompts,
  feature flags, or unrelated cleanup.
- Before adding a helper, search for equivalent behavior. Shared parser changes
  are limited to the named deterministic-only option; prompt projectors remain
  stage-local.
- A required change outside Change Surface, a different public interface, or a
  disagreement between plan and code stops execution for plan correction.
- The parent agent owns plan continuity, test contracts, all test-file edits,
  integration, test execution, live artifact inspection, and final reporting.
- After the parent establishes failing target-state tests, exactly one
  production-code subagent may implement the production and documentation
  change surface.
- The production subagent receives the locked contracts, failure-mode matrix,
  and file ownership list. It edits production/docs only and returns changed
  files plus static evidence.
- After parent verification, exactly one independent code-review subagent
  reviews the implementation and tests without inheriting the implementation
  agent's reasoning.
- The parent resolves review findings and repeats focused verification when a
  production correction is required.
- Lifecycle promotion, real-LLM execution authorization, and final sign-off
  remain user/parent decisions.

## Implementation Order

1. Promote this plan from `draft` to `approved` after user review.
2. Reread required skills, repository guidance, this plan, current git status,
   relevant READMEs, source, and tests.
3. Parent: add failing parser, schema, prompt-render, context-cap, and
   completion-config tests in `tests/test_utils.py`,
   `tests/test_frontline_relevance_agent.py`, and
   `tests/test_persona_relevance_agent.py`; record current missing contracts.
4. Parent: add fake-clock, FIFO one-in-flight, three-open-turn, version/claim,
   and graph handoff tests in `tests/test_relevance_turn_settlement.py` and
   `tests/test_relevance_turn_settlement_graph.py`; record expected failures.
5. Parent: add L01-L20 to the live test file with captured evidence fixtures,
   strict enum gates, prompt/output size capture, and qualitative rubrics;
   leave live execution pending.
6. Start exactly one production-code subagent with the approved plan and the
   parent-owned failing tests.
7. Production subagent: implement config/parser and both bounded node contracts,
   then the coordinator public interface and private FIFO executor.
8. Production subagent: wire queue/service/state/graph, retain private timing,
   remove superseded group paths, update docs, run static render checks, report
   files/commands/risks, and close.
9. Parent: inspect the complete diff and confirm only Change Surface paths
   changed before running focused deterministic tests.
10. Parent: run the focused parser, node, coordinator, graph, and queue suite;
    resolve only approved-contract failures and record evidence.
11. Parent: run affected envelope, media, service, runtime-coordination, and
    existing relevance live-contract tests as specified in Verification.
12. With explicit live-run authorization, parent: run L01-L20 one case at a
    time, inspect each artifact, and stop at the first quality or budget failure.
13. Parent: write
    `test_artifacts/debug/relevance_turn_settlement_live_llm_review.md` from
    inspected evidence, including per-gate expected behavior, observed action,
    prompt/output sizes, durations, association quality, and judgment.
14. Start exactly one independent code-review subagent for code review.
15. Parent: resolve findings, rerun affected deterministic and live gates, and
    record execution evidence in this plan.
16. After all acceptance criteria pass, mark the plan completed and archive it
    through the registry lifecycle.

## Execution Model

### Checkpoint A: Test Contract

Parent creates the strict agent schemas, fake monotonic clock, pending-turn
fixtures, failure-mode mapping, evidence-derived live cases, and current-state
expected failures. Exit gate: every F01-F26 row has deterministic coverage and
an L01-L20 mapping where semantic judgment is involved.

### Checkpoint B: Production Cutover

The production subagent delivers bounded stage contracts on the existing route,
deterministic-only parsing, FIFO one-in-flight relevance work, settlement
lifecycle, ready heap, settled action, atomic claim, unchanged downstream
cognition responsibility, and docs. Exit gate: parent review confirms exact
workload/timing contracts and old-path removal.

### Checkpoint C: Deterministic Verification

Run with the project virtual environment:

```powershell
venv\Scripts\python -m pytest tests/test_utils.py tests/test_frontline_relevance_agent.py tests/test_persona_relevance_agent.py tests/test_relevance_turn_settlement.py tests/test_relevance_turn_settlement_graph.py tests/test_service_input_queue.py -q
```

Then run affected message-envelope, graph, media, and runtime-coordination tests.
Exit gate: all pass, and fake-clock tests prove the 6/10-second bounds,
first-ready order, wait independence, stale rejection, one in-flight relevance
call, zero LLM JSON repair, and rendered prompt/completion caps.

### Checkpoint D: Strict Real-LLM Delivery Gates

Each case is one pytest test and one command. The implementation may share
fixtures, while each test contains exactly one semantic scenario.

| Gate | Test name | Required behavior |
|---|---|---|
| L01 | `test_live_frontline_discards_clear_third_party_message` | Clear third-party group talk is discarded by the compact frontline. |
| L02 | `test_live_mention_only_waits_and_accepts_request_followup` | Bare mention remains open and the 8.414-second request joins it. |
| L03 | `test_live_complete_question_accepts_delayed_image_followup` | Complete-looking question and 5.08-second image form one turn. |
| L04 | `test_live_interleaved_authors_keep_independent_turns` | A1/B1/A2 maps A2 to A while B remains independent. |
| L05 | `test_live_same_author_topic_change_starts_new_turn` | Unrelated same-author input starts a distinct turn. |
| L06 | `test_live_same_author_other_recipient_avoids_false_join` | The retained 4.673-second other-recipient case avoids an A-to-Kazusa merge. |
| L07 | `test_live_multi_recipient_message_preserves_kazusa_relevance` | A message genuinely addressing Kazusa and another user remains eligible for settled judgment. |
| L08 | `test_live_content_before_tag_promotes_recent_prelude` | A later direct tag can promote the supplied recent same-author prelude. |
| L09 | `test_live_multifragment_correction_uses_latest_intent` | Three chronological fragments and a correction produce one current intent. |
| L10 | `test_live_other_user_answer_makes_pending_reply_redundant` | Fresh history lets settled relevance ignore a now-answered turn. |
| L11 | `test_live_boundary_followup_invalidates_stale_relevance` | A follow-up during assessment prevents stale cognition entry and is present in the final judgment. |
| L12 | `test_live_followup_after_running_becomes_new_candidate` | A post-claim continuation is judged as a new turn with prior context. |
| L13 | `test_live_earliest_ready_relevant_turn_wins_cognition_claim` | Two relevant ready turns both choose `proceed`; the earlier eligibility receives the claim. |
| L14 | `test_live_hard_deadline_closes_continuous_fragments` | Continuous fragments reach a final `ignore` or `proceed` decision at ten seconds. |
| L15 | `test_live_cross_scope_candidates_never_attach` | Cross-author and cross-channel candidates remain unavailable for append. |
| L16 | `test_live_fourth_same_author_topic_bounds_open_turns` | A fourth independent topic starts cleanly while the oldest of three open turns freezes for assessment. |
| L17 | `test_live_long_multifragment_projection_preserves_latest_intent` | A capped prompt retains typed targets and the latest correction without exceeding 8k/16k limits. |
| L18 | `test_live_frontline_burst_and_ready_turn_respect_workload_contract` | A captured burst uses one in-flight relevance call, FIFO work order, and the configured stage budgets. |
| L19 | `test_live_waiting_ready_turn_releases_next_candidate` | The earlier turn chooses `wait`; the next ready relevant turn proceeds without waiting for its extension. |
| L20 | `test_live_attachment_burst_bounds_vision_and_preserves_overflow_signal` | More than four images produce at most four selected descriptions, preserve opening/latest evidence and `additional_media_present`, and still yield a grounded settled decision. |

Canonical invocation pattern:

```powershell
venv\Scripts\python -m pytest tests/test_relevance_turn_settlement_live_llm.py::TEST_NAME -q -s
```

For each case, run only that test, inspect terminal output and its new JSON
trace, then record the decision and qualitative judgment before continuing.

Live-gate assertions are strict by default. An environment flag cannot weaken
the expected action. A schema-valid but semantically wrong association,
waiting decision, or relevance reason fails the human review gate.

### Checkpoint E: Independent Review And Sign-Off

The reviewer checks frontline conservatism, typed-target priority, ownership,
wait independence, ready order, claim races, call/context bounds, cutover
completeness, private behavior, test realism, and artifact quality. Exit gate:
blocking findings are resolved, affected tests are rerun, and the diff remains
scoped.

## Progress Checklist

- [x] Pulled and characterized read-only conversation-history evidence.
- [x] Inspected current code/tests/plans and locked the two-agent scheduler.
- [x] Mapped failure modes to solutions and delivery gates.
- [x] Stage 0 - user approval recorded.
  - Covers: implementation step 1.
  - Verify/evidence: status and registry both read `approved`; record date and
    user instruction in Execution Evidence.
  - Handoff/sign-off: parent rereads the plan, signs Stage 0, starts Stage 1.
  - Sign-off: parent / 2026-07-16; explicit user approval and parent-only
    fallback recorded in Execution Evidence.
- [x] Stage 1 - parent-owned test contract established.
  - Covers: steps 2-5.
  - Verify: focused tests collect the expected missing-symbol/contract failures;
    live tests remain unexecuted.
  - Evidence: commands, failures, fixture provenance, prompt-size rubrics.
  - Handoff/sign-off: parent / 2026-07-16; parent-owned production fallback
    starts because the user explicitly required no subagent.
- [x] Stage 2 - production cutover complete.
  - Covers: steps 6-9.
  - Verify: changed-file inventory matches Change Surface; prompt render and
    static import checks succeed.
  - Evidence: parent-owned diff/status inventory, Python syntax/import checks,
    and removed-symbol greps; no subagent was used per the user's explicit
    instruction.
  - Handoff/sign-off: parent rereads and starts deterministic verification.
  - Sign-off: parent / 2026-07-16; production cutover and static checks passed.
- [x] Stage 3 - deterministic and affected regression verification passes.
  - Covers: steps 10-11.
  - Verify: all commands in deterministic/static Verification pass.
  - Evidence: command outputs, max rendered chars, call/concurrency assertions.
  - Handoff/sign-off: parent rereads and obtains live-run authorization.
  - Sign-off: parent / 2026-07-16; focused, contract, and affected suites passed.
- [ ] Stage 4 - L01-L20 and human review pass.
  - Covers: steps 12-13.
  - Verify: every case ran individually; the clean prompt rerun passed 18 of
    20 gates, while L05 and L16 remain semantic failures returning `discard`
    where a new character-relevant turn is expected. No trace remains
    uninspected.
  - Evidence: raw trace paths and agent-authored Markdown review.
  - Handoff/sign-off: parent rereads and starts independent code review.
  - Sign-off: pending remediation; the parent review records the two clean-run
    semantic failures and the remaining architect findings for Stage 5.
- [ ] Stage 5 - independent code review and remediation pass.
  - Covers: steps 14-15.
  - Verify: reviewer approves; affected deterministic/live gates rerun after
    fixes; no blocking finding remains.
  - Evidence: reviewer identity, findings, fixes, rerun commands, residual risk.
  - Handoff/sign-off: parent rereads and evaluates final acceptance.
- [ ] Stage 6 - acceptance, evidence, lifecycle, and archival complete.
  - Covers: step 16.
  - Verify/evidence: every acceptance row maps to recorded proof; plan and
    registry move to `completed` and archive paths consistently.
  - Handoff/sign-off: final parent sign-off and user report.

## Verification

### Static Verification

```powershell
rg "GROUP_COALESCE_MAX_GAP_SECONDS|_coalesce_addressed_group" src tests
rg "_should_ignore_third_party_reply|chaotic group noise without|group attention requires platform address" src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py
rg "kazusa_ai_chatbot\.nodes\.(frontline_relevance_agent|persona_relevance_agent)" src tests README.md docs
rg "RELEVANCE_AGENT_LLM" src/kazusa_ai_chatbot README.md docs/HOWTO.md tests
```

- The first three greps expect zero matches; exit code 1 is the accepted empty
  result. Any match blocks cutover.
- The route grep expects matches in config, frontline node, route/brain/node
  docs, HOWTO/root README, and tests; it must reveal no alternate route alias.
- Inspect the graph/service path and prove one group path, semantic slot labels
  rather than raw IDs, monotonic timers, UTC persistence, version-matched
  RUNNING, private immediate timing, and protected stage/action traces.

### Deterministic Verification

```powershell
venv\Scripts\python -m pytest tests/test_utils.py tests/test_frontline_relevance_agent.py tests/test_persona_relevance_agent.py tests/test_msg_decontexualizer.py tests/test_relevance_turn_settlement.py tests/test_relevance_turn_settlement_graph.py tests/test_service_input_queue.py -q
venv\Scripts\python -m pytest tests/test_message_envelope.py tests/test_media_inspection_contracts.py tests/test_runtime_coordination.py -q
```

The focused suite must prove all lifecycle actions, 6/10-second timing,
interleaving, three-open-turn policy, boundary invalidation, first-ready order,
per-fragment evidence, futures/handles/IDs, private behavior, four-image
descriptor selection/call bounds, media caching,
and fresh history. Workload assertions must prove:

- rendered messages are <=8,000/16,000 characters for worst-case fixtures;
- configs are <=256/512 completion tokens with thinking disabled;
- each maximally populated valid JSON schema fits its stage completion cap
  under the one-character-per-token planning envelope;
- frontline/settled active-call count never exceeds one combined;
- normal and extended logical-turn call counts match the budget table;
- an attachment burst selects no more than four unique descriptor cache misses,
  preserves opening/latest media order, and exposes the overflow descriptor;
- `parse_llm_json_output(... deterministic_only=True)` invokes no JSON-repair
  model from frontline, settled relevance, or admitted media description, and
  existing default parser tests remain unchanged.

### Live-LLM Verification

- Run L01-L20 individually and retain raw traces under
  `test_artifacts/llm_traces/` with exact route/stage evidence.
- Author the Markdown review from inspected action, association,
  incompleteness, response-reason, prompt/output characters, durations, and
  cross-case consistency evidence.
- After a prompt change, rerun affected gates and retain both artifact sets.

## Independent Plan Review

- Reviewer: primary agent in a fresh-review posture, without a subagent, per
  the user's explicit instruction.
- Review date: 2026-07-16.
- Inputs: this draft, plan/execute/cutover contracts, retained DB evidence,
  current queue/service/graph/relevance/config/parser code, subsystem docs, and
  existing deterministic/live relevance tests.
- Result: all surfaced blockers were corrected inline; the plan was `draft` at
  review time and was later promoted to `approved` after explicit user
  approval, as recorded in the execution evidence below.

| Finding | Severity | Resolution |
|---|---|---|
| Plan class used descriptive text and top matter lacked risk/acceptance fields | blocker | Set `high_risk_migration`; added highest-risk and acceptance summaries. |
| Change Surface left rename/add-test alternatives to executors | blocker | Locked the existing persona filename, exact created tests, and grouped Create/Modify/Delete/Keep paths. |
| Frontline reused the final route while every input adds a call | blocker | The initial review proposed a separate high-volume route; the user later clarified that the existing `RELEVANCE_AGENT_LLM` route must remain shared, so the implementation serializes both stages on that route. |
| Frontline 12k/512 and settled 50k/current 8192 defaults overloaded weak local models | blocker | Set total 8k/256 and 16k/512 hard budgets with conservative workload envelopes. |
| The two-characters-per-token estimate understated mixed-CJK load | blocker | Switched to a one-character-per-token planning envelope, tightened both input caps, and reduced free-form output lengths to fit completion caps. |
| Open candidates, preludes, and long fragment/history payloads were unbounded | blocker | Added three open slots, two preludes, 2k current text, bounded component projections, and fourth-topic policy. |
| Prompts exposed internal IDs/timing facts and expected local inference from operational data | blocker | Replaced them with bounded slot labels and semantic descriptors; deterministic code maps mechanics. |
| Frontline and settled calls could overlap or starve each other | blocker | Added one FIFO relevance executor with one combined in-flight call and explicit blocking semantics. |
| Generic parser could invoke a hidden JSON-repair LLM beyond stated call caps | blocker | Added deterministic-only parser mode and zero repair-model calls for these stages. |
| New coordinator lacked an executable public interface | blocker | Added exact service-facing operations and private-internal boundary. |
| Workload verification lacked prompt, completion, concurrency, and call-count gates | blocker | Added deterministic caps, L16-L20 workload/overflow live cases, trace measurements, and exact commands. |
| User-controlled attachment bursts could create unbounded descriptor and repair-model calls | blocker | Reused the four-attachment projection bound per settled turn, retained opening/latest media plus an overflow descriptor, cached results, and disabled descriptor repair-model calls. |
| CJK prompt and live-output handling lacked an execution safeguard | blocker | Added the mandatory CJK skill plus source-delimiter, exact-byte, UTF-8, console-output, and immediate syntax gates. |
| Overdesign/autonomy/continuity/checklist sections were below plan contract | blocker | Rewrote guardrail and autonomy rules; added reread/review rules and evidence-bearing stage checklists. |
| High-risk risk records lacked the contract's exact section name and verification column | blocker | Normalized `Risks` and mapped every mitigation to deterministic or live proof. |
| Change-surface ordering and the code-review gate boundary were underspecified | blocker | Reordered Delete/Modify/Create/Keep, made the coordinator public/private split explicit, and added the reviewer start/stop contract. |
| One live gate combined two ordering outcomes | non-blocking | Separated two-relevant first-ready behavior from wait fall-through in L13 and L19. |

Residual accepted boundary: this plan controls relevance-stage workload only.
Global scheduling when cognition, RAG, dialog, vision, and relevance share one
physical GPU remains Deferred and requires measured evidence before expansion.

## Independent Code Review

Run this gate after every Verification command and live review pass, and before
completion, lifecycle updates, merge, or sign-off. The parent creates exactly
one independent code-review subagent. If native subagents are unavailable, the
execution stops pending explicit user authorization for a fallback.

The reviewer receives this plan, final diff/status, deterministic output, all
L01-L20 commands/outcomes, raw trace paths, the agent-authored review, and
recorded residual limitations. The reviewer inspects only and implements no
fix.

The reviewer reports findings by severity with file and line references,
checks every F01-F26 row, and gives an explicit delivery recommendation.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Frontline false negative hides a real continuation | Conservative admission contract, typed evidence priority, and recent-prelude promotion | L01-L03 and prompt-contract tests |
| Frontline false positive raises LLM load | 8,000-character/256-token cap, one in-flight executor, and settled final silence | Budget assertions and L01/L18 |
| Semantic false join contaminates cognition | Per-fragment targets, supplied-candidate validation, and no timing-only join | L05-L07 and association tests |
| Added latency affects normal group replies | Six-second window only for admitted group turns; irrelevant input exits after frontline; private timing stays immediate | Fake-clock/private regressions and trace durations |
| Wait starves other users | Timer owns no cognition lane; global heap falls through | L19 and ready-heap tests |
| Frontline burst starves settled assessment | FIFO relevance work prevents later arrivals overtaking enqueued settled work | Controlled-latency test and L18 |
| Two relevance calls overload one local backend | Combined one-in-flight executor serializes both stages on the shared route | Concurrency assertion and L18 durations |
| Final local model loses judgment under excessive context | 16,000-character/512-token cap, stable system prompt, and semantic projection | Render-cap tests and L17 |
| Malformed JSON silently adds a repair-model call | Deterministic-only repair keeps the call budget exact | Parser call-count tests |
| Stale relevance enters cognition | Versioned claim node and hard-deadline final assessment | Boundary race test and L11 |
| Continuous fragments cause indefinite delay or calls | Ten-second hard cap and two settled assessments maximum | Fake-clock test and L14 |
| Re-assessment repeats media cost | Cache accepted fragment descriptions within the pending turn | Cache call-count test |
| Attachment bursts multiply vision and repair calls | Describe four selected unique images maximum, expose overflow, cache, and parse deterministically | Descriptor call-count test and L20 |
| Fresh history changes the apparent topic | Keep assembled-turn boundaries separate from channel history | Projection test and L10 |
| Hard deadline splits a late genuine follow-up | Route later input as a new candidate with prior-turn history | Late-arrival test and L12 |
| Local model struggles with a combined contract | Two narrow agents, closed enums, compact examples, and individual gates | L01-L20 human review |

## Acceptance Criteria

- Every active group message reaches the compact frontline before group
  discard/media/full relevance, and only supplied candidates can receive a
  valid `append`.
- Frontline uses `RELEVANCE_AGENT_LLM` within 8,000 rendered characters, 256
  completion tokens, three open slots, two prelude slots, and thinking off.
- Settled relevance uses `RELEVANCE_AGENT_LLM` within 16,000 rendered
  characters, 512 completion tokens, ten history rows, and thinking off.
- The FIFO relevance executor permits one combined in-flight relevance call;
  each inbound message gets one frontline call, each logical turn gets one
  settled call normally and two maximum, and JSON-repair LLM calls remain zero.
- Each assembled turn causes at most four unique vision cache-miss calls;
  opening/latest media and `additional_media_present` remain visible to the
  settled model, and reassessment repeats no descriptor call.
- Admitted group turns use monotonic six-second quiet and ten-second hard
  deadlines; settled relevance returns valid `ignore/proceed/wait` actions.
- Waiting holds neither lane; A1/B1/A2 stays independent; earliest eligibility
  and leader-sequence ties control the next claim with fall-through.
- Version changes prevent stale cognition, and post-`RUNNING` input becomes a
  new non-preempting candidate.
- Every accepted fragment contributes ordered IDs and retains its own typed
  target/reply evidence and media description.
- Old group coalescing, threshold pruning, and metadata semantic skips are
  retired while private timing remains regression-covered.
- Focused and affected deterministic suites pass; L01-L20 pass individually
  against the real relevance model and have inspected raw artifacts.
- The agent-authored review, independent review, and subsystem docs confirm the
  final contracts with no unresolved blocking finding.

## Execution Evidence

This section remains append-only during implementation.

- 2026-07-16: Draft created from repository inspection and the retained
  conversation-history evidence artifact. Production implementation and test
  execution remain pending user approval of the draft.
- 2026-07-16: Parent-only independent plan review completed at the user's
  request. Seventeen findings were resolved inline, including bounded local-model
  route usage, 8k/16k input caps, 256/512 completion caps, three bounded candidate
  slots, FIFO one-in-flight relevance work, deterministic-only JSON parsing,
  a four-image descriptor cap, exact interfaces/change paths, workload gates,
  CJK execution safeguards, and plan-contract corrections. No plan-review
  blocker remains; implementation status is now `approved`.
- 2026-07-16: User approved execution from Stage 0 through Stage 4 and
  explicitly required parent-only execution with no subagent. The parent owns
  implementation, tests, verification, live-gate inspection, and the Stage 4
  handoff; Stage 5 onward is handed to the architect agent for independent
  review and remediation.
- 2026-07-16: Stage 1 contract tests were added in
  `tests/test_utils.py`, `tests/test_frontline_relevance_agent.py`,
  `tests/test_persona_relevance_agent.py`,
  `tests/test_relevance_turn_settlement.py`,
  `tests/test_relevance_turn_settlement_graph.py`, and
  `tests/test_relevance_turn_settlement_live_llm.py`. The collection command
  was run with the project virtual environment. Expected failures identified
  the missing `frontline_relevance_agent` module, missing
  `brain_service.turn_settlement` module, missing settled-relevance exports,
  and the unextended graph claim boundary. Fixtures cover closed enums,
  prompt caps, completion budgets, deterministic parsing, fake clocks,
  versioned claims, FIFO one-in-flight work, graph handoff, and L01-L20
  evidence capture.
- 2026-07-16: Stage 2 parent-owned production cutover completed. The status
  inventory covers the locked Change Surface plus the two new production
  modules and four new relevance/settlement test modules. `git diff --check`
  completed with only Git's existing LF-to-CRLF normalization warnings.
  `py_compile` passed for every touched Python source/test file, static imports
  passed for config, graph, coordinator, frontline, settled relevance, and
  service, and the retired group coalescing/pruning/metadata-skip greps returned
  zero matches. No subagent report exists because the user explicitly required
  parent-only execution.
- 2026-07-16: Stage 3 verification completed. The focused deterministic command
  passed 92 tests with 3 live cases deselected; the envelope/media/runtime
  command passed 16 tests; and the affected cognition/media/service command
  passed 48 tests. The route inventory test was updated to verify the shared
  `RELEVANCE_AGENT_LLM` route. The unchanged L3 content-plan prompt's
  stale fingerprint fixture was corrected from 14,137 bytes to its repository
  baseline of 14,466 bytes and digest
  `a000f8bb9afd733a28094a1cfec99cfff1bca354140c77fe695e686ba719bbcf`.
- 2026-07-16: Stage 4 completed parent-only. Each L01-L20 live gate was run
  as an individual `pytest -m live_llm` invocation against the configured
  local OpenAI-compatible endpoint, with the raw response artifact inspected
  before the next case. All 20 gates passed. Latest evidence measured a
  2,903-character maximum settled prompt, a 2,618-character maximum frontline
  prompt, 367-character maximum settled output, 188-character maximum
  frontline output, and L18 frontline concurrency of one with FIFO order
  `burst-0`, `burst-1`, `burst-2`. The complete trace inventory and qualitative
  review are recorded in
  `test_artifacts/debug/relevance_turn_settlement_live_llm_review.md`.
- 2026-07-16: Final Stage 4 static verification passed: all touched Python
  files compiled, `git diff --check` passed with only LF-to-CRLF warnings, the
  retired group coalescing and semantic skip greps returned no matches, and
  the shared `RELEVANCE_AGENT_LLM` route appeared in config, docs, and tests.
  The parent review leaves two explicit Stage 5 architect findings:
  service-level prelude mapping is incomplete and L18 does not execute a real
  settled branch. Stage 5 onward is handed to the architect agent and remains
  unexecuted by the parent as directed.
- 2026-07-16: User corrected the route requirement: frontline and settled
  relevance must continue using the existing `RELEVANCE_AGENT_LLM` settings.
  The dedicated frontline deployment variables and route inventory entry were
  removed; the frontline keeps only its stage-local input/completion caps.
- 2026-07-16: Anti-cheat/local-LLM prompt correction completed before the
  clean live rerun. Fixture-shaped instructions naming the known topic-break
  case and its expected routing were removed from the runtime prompt. The
  remaining prompt is general semantic/schema guidance and contains no test
  labels or expected answers.
- 2026-07-16: Clean Stage 4 live rerun executed each L01-L20 case individually
  against the shared configured route and inspected each current raw trace.
  Eighteen gates passed. L05 and L16 returned `discard` instead of the
  contract's expected new-turn `start`; these remain honest semantic failures
  for Stage 5 review. Corrected results and trace paths are recorded in
  `test_artifacts/debug/relevance_turn_settlement_live_llm_review.md`.
- 2026-07-16: Post-correction deterministic verification was rerun
  sequentially: the focused suite passed 92 tests with 3 live cases
  deselected, the envelope/media/runtime suite passed 16 tests, and the
  affected cognition/multimodal/service group passed 61 tests. Touched Python
  files compiled, the retired route grep returned no matches, and the current
  prompt contains no fixture-shaped routing instruction.
- 2026-07-16: User requested that relevance-agent code have a dedicated module
  boundary similar to the other subsystem packages. The approved scope is now
  recorded as a big-bang move of both semantic agents into
  `kazusa_ai_chatbot.relevance`, with `relevance/__init__.py` as the public
  facade and `relevance/README.md` as its ICD. The old `nodes` paths are
  deleted without compatibility shims; `brain_service.turn_settlement` keeps
  deterministic lifecycle ownership. Existing contracts, route settings, and
  LLM call budgets remain unchanged.
- 2026-07-16: Relevance package extraction completed parent-only. The moved
  modules are `relevance/frontline_relevance_agent.py` and
  `relevance/persona_relevance_agent.py`; runtime callers use the package
  facade, while concrete-module imports remain limited to tests that inspect
  private model/config state. Python compilation passed, the old `nodes`
  import-path grep returned zero matches, and the `FRONTLINE_RELEVANCE_LLM`
  alias grep returned zero matches.
- 2026-07-16: Post-extraction verification passed the focused 60-test suite,
  the plan's 92-test deterministic/queue suite with 3 live cases deselected,
  the 16-test envelope/media/runtime suite, and the 61-test affected
  cognition/multimodal/service suite. Individual post-extraction live smoke
  gates L01 and L03 each passed against the shared configured route.
- 2026-07-16: Stage 4 remains open for the architect handoff. The corrected
  full clean live run remains 18/20, with L05 and L16 returning `discard`
  instead of the expected new-turn `start`; the package move preserved that
  semantic result and introduced no prompt, route, completion-budget, or
  additional-call change. Stage 5 onward is handed to the architect agent for
  independent review and remediation.
