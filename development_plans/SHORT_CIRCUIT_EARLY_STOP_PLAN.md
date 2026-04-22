# Short-Circuit Early Stop Plan

## Problem Statement

Some incoming user messages should not consume the full reasoning pipeline.

Examples:

- the message does not make semantic sense
- the message is too fragmentary to be actionable
- the message is low-value ambient chatter
- the message is repetitive and contributes nothing new
- the message is so boring or non-useful that the character would naturally choose not to engage

In these cases, the desired system behavior is:

- the user message still enters conversation history
- the pipeline stops early
- the character sends **no reply**

This is not a retrieval problem and not a dialog-generation problem.
It is a **front-of-pipeline gating problem**.

---

## Current Architecture Constraint

In the current live code path:

- the user message is saved immediately in `service.py` before graph invocation
- the graph enters through `relevance_agent`
- the graph stops early when `should_respond` is false
- the final API response is determined by:
  - `final_dialog`
  - `use_reply_feature`

This means the current system already has a natural short-circuit point.
The main missing piece is a richer and more explicit **early-stop policy**.

---

## Desired Behavior

## Functional Goal

When the message is:

- nonsensical
- too weak to interpret
- not socially worth engaging
- or judged by the character as unproductive / too boring

then the system should:

- set `should_respond = False`
- set `use_reply_feature = False`
- skip downstream expensive nodes
- return:

```python
ChatResponse(
    messages=[],
    should_reply=False,
)
```

## Important Non-Goal

This is **not** a clarification system.
If the message falls into the early-stop band, the desired output is silence, not a follow-up question.

Clarification should remain reserved for messages that are:

- meaningful
- directed
- and potentially valuable

but still missing a required slot.

---

## System-Level Principle

Short-circuit should happen as early as possible, but only after applying **value-aware gating**.

The system should distinguish between:

- **unworthy of response**
- **worthy but incomplete**
- **worthy and actionable**

This is the key difference between:

- early stop
- clarification
- full pipeline continuation

---

## Recommended Architecture

## Phase 1 Recommendation: Extend the Existing Front Gate

Because the current live pipeline already uses `relevance_agent` as the front gate, the fastest and lowest-risk implementation is:

- keep `relevance_agent` as the first semantic gate
- extend it with explicit short-circuit categories and policy
- continue using the existing `should_respond` branch in `service.py`

This preserves the current graph shape while adding richer decision quality.

## Phase 2 Recommendation: Split into Dedicated Gate Node

After the policy stabilizes, refactor into:

```text
START
  -> multimedia_descriptor_agent (if needed)
  -> short_circuit_gate
  -> relevance_agent
  -> persona_supervisor2
  -> END
```

### Why split later

A dedicated `short_circuit_gate` gives cleaner separation between:

- response-worthiness
- topic/context analysis

But for initial rollout, extending the current front gate is lower risk.

---

## Decision Model

The gate should classify each input into one of three outcomes:

## 1. STOP_SILENTLY

The message should not get a reply.

### Examples

- nonsense / corrupted text
- accidental fragment with no social weight
- low-value ambient filler
- boring repetition with no new content
- weak non-addressed noise
- stale continuation fragment with no recoverable task and no value in clarification

## 2. CONTINUE

The message should proceed through the normal pipeline.

### Examples

- directed question
- direct request
- emotional disclosure
- conflict / challenge
- meaningful continuation
- actionable task
- socially salient interaction

## 3. CONTINUE_WITH_CLARIFICATION

The message is worth engaging but cannot be safely acted on yet.

### Examples

- meaningful but underspecified task
- valid continuation with missing slot
- reply continuation with ambiguous target

This third class is important because it prevents the early-stop policy from swallowing meaningful but incomplete messages.

---

## Stop Categories

The gate should emit an explicit stop category so the system remains observable.

Recommended categories:

- `nonsense_unparseable`
- `platform_noise`
- `accidental_fragment`
- `not_addressed`
- `low_information`
- `boring_low_value`
- `repetitive_no_novelty`
- `stale_unanchored_continuation`

These are internal categories only.
They are not user-facing.

---

## Never-Stop Overrides

To avoid over-silencing, the gate must include strong continuation overrides.

The system should almost never early-stop when any of the following hold:

- direct question
- explicit request / command to the bot
- reply-to-bot with clear intent
- direct mention of bot identity
- emotionally salient message
- conflict / accusation / challenge
- user expresses hurt, distress, anger, urgency, or vulnerability
- the message contains a clearly meaningful ask even if awkwardly phrased

### Practical rule

"Boring" is never enough by itself to silence a **directly addressed meaningful message**.

---

## What Counts as “Too Boring”

This must be defined at the system level so the behavior is stable.

A message should only count as `boring_low_value` if most of the following are true:

- not directly addressed to the bot
- not a reply-bound task continuation
- not emotionally salient
- not meaningfully novel relative to recent history
- not a real request
- not a social bid that the current character would naturally catch
- contributes no clear content, decision need, or relationship movement

### Examples

Candidates for stop:

- empty filler like `哦`, `嗯`, `...` in a non-directed context
- repetitive nudges with no content and no task context
- non-addressed channel noise
- same low-value message repeated several times without novelty

Non-candidates for stop:

- `嗯` after an emotional confession
- `哦？` in a tense flirt/challenge context
- `一定要哦` when clearly continuing a known task

---

## Recommended Implementation Strategy

## Layer 0 — Deterministic Cheap Filters

Run first inside the front gate.

### Purpose

Catch cheap obvious no-reply cases without LLM cost.

### Checks

- empty or whitespace-only input
- unsupported artifact text
- exact duplicate user input within short cooldown window
- message containing only punctuation / emoji with no addressing signal
- obvious platform event residue

### Output

If matched strongly:

```python
{
    "should_respond": False,
    "stop_category": "platform_noise",
    "stop_confidence": 1.0,
}
```

These are safe hard stops.

---

## Layer 1 — Semantic Utility Assessment

If deterministic filters do not stop the message, run a semantic front-gate assessment.

### Recommended placement

Inside the current `relevance_agent` first, then split later if desired.

### Purpose

Judge whether the message is:

- socially meaningful
- actionable
- emotionally salient
- or too low-value to justify a reply

### Required Output Fields

```python
{
    "decision": "stop_silently | continue | continue_with_clarification",
    "should_respond": true,
    "reason_to_respond": "...",
    "stop_category": "",
    "stop_reason": "",
    "use_reply_feature": false,
    "channel_topic": "...",
    "indirect_speech_context": "..."
}
```

### Important behavior

`stop_silently` must be explicit.
Do not overload it into a vague `should_respond = false` without explanation.

---

## Layer 2 — Novelty / Repetition Check

Some low-value cases only become stoppable in context.

### Purpose

Detect repeated low-yield user behavior such as:

- same filler again and again
- same non-progressive poke
- repeated empty demand with no new content

### Signals

- overlap with last N user messages
- low semantic novelty
- no new slots / facts / emotional shift
- no escalation in urgency or tone

### Result

This should feed into:

- `repetitive_no_novelty`
- not directly into hard stop unless the novelty score is clearly exhausted and no override applies

---

## Proposed State Additions

Update `IMProcessState` with explicit short-circuit metadata.

### Recommended fields

```python
short_circuit_triggered: bool
short_circuit_decision: str
short_circuit_category: str
short_circuit_reason: str
short_circuit_confidence: float
```

### Purpose

These fields allow:

- logs
- test assertions
- future analytics
- debugging false positives

---

## Proposed Graph Behavior

## Minimal-Change Version

Use the existing graph branch:

```text
START -> relevance_agent -> (should_respond ? continue : END)
```

### Meaning

`relevance_agent` now owns:

- normal reply-worthiness
- short-circuit stop logic
- clarification-worthy continuation logic

### Benefit

- minimal graph churn
- easy rollback
- no service-level contract breakage

---

## Ideal Version After Stabilization

```text
START
  -> short_circuit_gate
  -> if stop: END
  -> relevance_agent
  -> persona_supervisor2
  -> END
```

### Benefit

Separates:

- “Is this worth any reply at all?”
from
- “What is this about?”

This is architecturally cleaner.

---

## API / Service Behavior

In the current `service.py` flow:

- user input is already saved before graph execution
- short-circuit therefore should only suppress downstream work and outgoing reply

### Desired final behavior after stop

- `final_dialog = []`
- `use_reply_feature = False`
- `should_reply = False`
- no bot message is saved in background
- no downstream RAG / supervisor / dialog generation runs

### This is already mostly supported

Because `service.py` already does:

- no bot save if `final_dialog` is empty
- `should_reply` comes from `use_reply_feature`

So implementation mainly needs to guarantee that short-circuit paths set:

```python
should_respond = False
use_reply_feature = False
```

---

## Prompt / Policy Guidance for the Gate

The front gate prompt should explicitly distinguish:

- message is meaningless
- message is meaningful but incomplete
- message is meaningful but low-value
- message is emotionally or socially salient

### Key instruction

A message may be stopped silently only when:

- replying would add no meaningful conversational value
- silence would be socially natural for this character in this context
- no override condition applies

### Another key instruction

Do not ask clarification for junk.
Do not continue full reasoning for junk.
Just stop.

---

## Implementation Phases

## Phase 1 — Add Short-Circuit Metadata to State

### Files

- `src/kazusa_ai_chatbot/state.py`

### Changes

Add:

- `short_circuit_triggered`
- `short_circuit_decision`
- `short_circuit_category`
- `short_circuit_reason`
- `short_circuit_confidence`

---

## Phase 2 — Add Deterministic Prefilter Helpers

### Files

Recommended options:

- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
- or a new helper module such as `src/kazusa_ai_chatbot/nodes/input_gate_utils.py`

### Changes

Add cheap filters for:

- empty input
- punctuation-only noise
- duplicate low-value input
- obvious platform artifacts

These should run before the LLM call.

---

## Phase 3 — Extend Front-Gate LLM Contract

### Files

- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`

### Changes

Expand the front-gate prompt/output schema to include:

- `decision`
- `stop_category`
- `stop_reason`
- short-circuit-aware `should_respond`

### Policy

The model must be told that some messages should produce **no reply at all**.

---

## Phase 4 — Add Branch-Aware Logging

### Files

- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
- optionally `src/kazusa_ai_chatbot/service.py`

### Changes

Log:

- stop / continue / clarify decision
- stop category
- stop reason
- whether deterministic or semantic path triggered the stop

Example:

```text
Short Circuit Analysis:
  decision: stop_silently
  category: boring_low_value
  reason: non-directed low-novelty filler with no task or emotional salience
  confidence: 0.91
```

---

## Phase 5 — Optional Dedicated Gate Refactor

### Files

- `src/kazusa_ai_chatbot/service.py`
- new node file, e.g. `src/kazusa_ai_chatbot/nodes/short_circuit_gate.py`

### Changes

Split short-circuit responsibility out of `relevance_agent` once the policy has stabilized.

This is optional and should be done only after behavior is validated.

---

## Test Plan

## Case 1 — Pure Noise

### Input

- whitespace
- punctuation-only
- obvious platform residue

### Expected

- `should_respond = False`
- stop category reflects noise
- no downstream pipeline

---

## Case 2 — Low-Value Boring Ambient Message

### Input

- low-information filler in group context
- not addressed to bot
- no emotional salience

### Expected

- `stop_silently`
- no reply

---

## Case 3 — Same Message Repeated With No Novelty

### Input

- repeated low-value poke
- no new information

### Expected

- `repetitive_no_novelty`
- no reply

---

## Case 4 — Direct Request Must Not Be Silenced

### Input

- direct question or explicit bot-directed request

### Expected
n
- continue
- never dropped as boring

---

## Case 5 — Emotionally Salient Minimal Message

### Input

- short but emotionally important message
- e.g. anger, sadness, accusation, disappointment

### Expected

- continue
- no short-circuit

---

## Case 6 — Meaningful but Incomplete Request

### Input

- valuable continuation but missing slot

### Expected

- `continue_with_clarification`
- not `stop_silently`

---

## Case 7 — Character-Specific Social Silence

### Input

- dull, low-value message in a context where silence is socially natural for this character

### Expected

- stop is allowed
- no reply is returned

---

## Observability Requirements

Track short-circuit rates over time by:

- stop category
- channel type
- direct vs indirect address
- false-positive review samples

This is necessary because over-silencing is the main system risk.

---

## Main Risks

## Risk 1 — Over-Silencing Real Messages

### Mitigation

- strong never-stop overrides
- explicit emotional salience override
- direct-address override
- staged rollout with logs

## Risk 2 — Character Feels Too Passive

### Mitigation

- define `boring_low_value` narrowly
- keep stop behavior character-aware but not overaggressive
- review stop-rate samples manually

## Risk 3 — Clarification and Silence Get Mixed Up

### Mitigation

- force explicit three-way decision:
  - stop
  - continue
  - continue_with_clarification

## Risk 4 — Hidden Debuggability Loss

### Mitigation

- add explicit stop metadata to state
- log stop reasons and categories
- add regression tests

---

## Acceptance Criteria

The feature is complete when:

- nonsense and low-value inputs can terminate the pipeline early
- the final API response contains no reply for stop cases
- directly addressed meaningful messages are not dropped as boring
- meaningful but incomplete inputs are clarified rather than silently ignored
- stop decisions are observable in logs and testable in code
- no downstream expensive nodes run on stop cases

---

## Recommended Decision

The recommended implementation path is:

1. **extend the current `relevance_agent` into an explicit short-circuit front gate**
2. **add deterministic prefilters before the LLM call**
3. **add semantic stop categories and explicit stop metadata**
4. **use the existing `should_respond == False` branch to end the graph early**
5. **optionally split into a dedicated `short_circuit_gate` node after the policy stabilizes**

This gives you early stop with minimal graph churn and leads exactly to the desired end state:

- the user message is stored
- the pipeline stops
- the character sends no reply
