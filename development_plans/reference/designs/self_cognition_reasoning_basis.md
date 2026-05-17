> Superseded Architecture Document
>
> Status: superseded
> Superseded by plan: development_plans/active/bugfix/self_cognition_speak_delivery_bugfix_plan.md
> Canonical current doc: src/kazusa_ai_chatbot/self_cognition/README.md
> Supersession rule: private-candidate-only and no-production-delivery claims
> in this document are no longer architecture authority. Current production
> self-cognition selected `speak` must resolve a target before cognition and
> attempt delivery after dialog rendering.

# self cognition reasoning basis

## Document Control

- Status: reference
- Related execution plan:
  `development_plans/archive/completed/short_term/self_cognition_agency_loop_plan.md`
- Related ICD:
  `development_plans/reference/designs/self_cognition_tracking_icd.md`
- Purpose: collect rationale, online research, code observations, and
  experiment results that justify the execution plan.
- Execution rule: this document is not an implementation contract. Use the
  archived plan for historical execution evidence and the production module
  README for current interfaces.

## Online Research Basis

Industry practice for production agents is not "let the model act directly."
The recurring pattern is:

```text
model reasoning
-> structured candidate action
-> structured action record
-> durable or resumable state
-> validated tool execution
-> trace / audit
```

Research sources used for this design:

- Anthropic, [Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents).
  Lesson: use simple, composable patterns; distinguish fixed workflows from
  agents that dynamically direct tool use; route complex tasks to specialized
  downstream processes when categories are clear.
- Anthropic, [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents).
  Lesson: agent tools need clear purpose, namespacing, meaningful context,
  token-efficient responses, evaluations, and strict input/output contracts.
- OpenAI Agents SDK, [Guardrails](https://openai.github.io/openai-agents-python/guardrails/).
  Lesson: input, output, and tool guardrails run at different boundaries;
  side-effect tools need tool-level checks.
- OpenAI Agents SDK, [Tracing](https://openai.github.io/openai-agents-python/tracing/).
  Lesson: production agent runs need traces of LLM generations, tool calls,
  handoffs, guardrails, and custom events.
- OpenAI, [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
  Lesson: guardrails and human intervention are first-class safeguards during
  rollout, even when the end state is autonomous execution.
- LangGraph, [Persistence](https://langchain-5e9cc07a.mintlify.app/oss/python/langgraph/persistence),
  [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop),
  and [Durable execution](https://docs.langchain.com/oss/python/langgraph/durable-execution).
  Lesson: long-running agents need persisted state, pause/resume,
  idempotency, and side effects wrapped so they are not repeated on replay.
- Microsoft AutoGen, [Human-in-the-loop](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/human-in-the-loop.html).
  Lesson: agent teams need explicit handoff and feedback points; blocking
  human approval is useful for sensitive rollout but should not become the
  permanent semantic decision policy.

Applied to Kazusa:

- L1/L2/L3 cognition is the reasoning layer.
- Self-cognition tracking is the durable/resumable state layer.
- Action attempts and outbox candidates are the structured action layer.
- Dispatcher and scheduler are the validated execution layer.
- Audit and tracking records are the trace layer.
- Cognition owns the social decision to send. Deterministic code owns
  execution mechanics, idempotency, cadence, and validation.

## Project Evidence

The completed multi-source cognition architecture established the reusable
source pattern:

```text
trigger source
-> source-specific episode builder
-> shared input-source/percept normalization
-> shared context stack planner
-> shared RAG supervisor
-> shared L1/L2/L3 cognition nodes
-> shared action/output router
-> origin-aware persistence, audit, delivery, or no-op
```

For self-cognition, the important implication is that a new idle source should
wake the same cognition path. A standalone prompt that pretends to be Kazusa is
not sufficient.

Current code observations:

- `CognitiveEpisode` already supports non-chat output modes such as `preview`
  and `scheduled_action_request`.
- `internal_thought_cognition.py` already accepts non-chat cognition modes,
  but `think_only` is not suitable for outward-capable production runs.
- Conversation progress is read by L3, but normal progress recording assumes a
  completed responsive turn.
- The current consolidator origin path is user-message-based.
- Scheduler pending-event deduplication only covers pending rows.
- User memory units have active/completed/cancelled lifecycle fields, but a
  proactive send does not automatically complete an active commitment.
- Current no-dialog cognition residue is ephemeral: if there is no
  `final_dialog`, normal live-chat progress recording, assistant history save,
  and consolidation are not run.

Current live-chat side-effect coupling:

```text
final_dialog exists
-> assistant conversation row may be saved
-> conversation progress may be recorded
-> consolidation may run
-> character state, relationship state, memory units, affinity,
   character image, and cache invalidation may happen
```

This coupling is suitable for normal `/chat` turns but too broad for idle
self-cognition. Idle cognition therefore needs route-specific tracking rather
than reuse of the normal `final_dialog` side-effect switch.

## Experiment Results

### Timely Unfinished Promises

Synthetic unfinished-promise evidence produced the clearest useful
self-cognition behavior.

Visible conversation pattern:

```text
User asks Kazusa to remind them before a real-world time.
Kazusa accepts the reminder in dialog.
No fulfillment or cancellation evidence exists.
```

Before due time:

```text
due_state: future_due
expected route: progress/scheduler-maintenance candidate
expected send: no
```

After due time:

```text
due_state: past_due
expected route: action_candidate
expected send: one candidate, subject to execution mechanics
```

Finding:

- The same visible evidence can correctly produce silence before due time and
  proactive contact after due time.
- The source packet must include semantic due state, not only raw timestamps.
- Timely commitments are the strongest observed trigger source.

### Repeat Risk

A past-due, unfinished commitment can keep qualifying as a trigger on every
idle tick.

Finding:

- Pending scheduler dedup is not enough because completed or failed events no
  longer appear as pending duplicates.
- Generated wording is not a stable duplicate identity.
- Self-cognition needs action-attempt tracking before any live send path.

Required idempotency identity:

```text
source_kind + source_id + due_at + target_scope + action_kind
```

### No-Action Residual

When Kazusa decides not to interact, the result can still be useful if it is
routed explicitly.

Observed routes:

- settled conversation -> audit or avoid-reopening progress candidate;
- before-due commitment -> progress/scheduler-maintenance candidate;
- cadence-held desire to contact -> held action-attempt record;
- low-quality noisy group input -> rejected or audit-only trigger.

Finding:

- Silence is not one outcome.
- Silent cognition must not silently mutate memory or personality.
- If silence should influence future behavior, the tracking system must store
  a compact route effect.

### Trigger Source Quality

Observed trigger quality from strongest to weakest:

| Rank | Trigger source | Quality | Reason |
|---|---|---|---|
| 1 | active commitment or due reminder | high | concrete time, target, and action expectation |
| 2 | active conversation-progress open loop | medium-high | useful continuity signal without needing a send |
| 3 | recent private/direct actionable dialog | medium | can produce natural silence or follow-up |
| 4 | pending outbox/scheduler state | medium | useful for retry, cooldown, and duplicate review |
| 5 | topic-triggered RAG follow-up | medium | valuable but more expensive and less proven |
| 6 | noisy group chat | low | weak unless direct actionable evidence exists |
| 7 | reflection artifact | rejected | produced self-observation, not useful agency trigger |

Reflection-derived fields such as mood, global vibe, and promoted growth may
modify interpretation after a valid trigger exists. They should not start a
self-cognition run by themselves.

### Prompt Framing Issue

Early examples over-weighted passive framing such as "no input" and "waiting."
That nudged cognition toward silence even when the source packet contained a
useful commitment.

Finding:

- Self-cognition input must not frame the character as passively waiting.
- The packet should describe visible evidence, due state, target scope,
  current mood/global context as modifiers, and the concrete decision needed.
- The cognition layers should still decide whether to send, hold, search, or
  stay silent.

## Consumer Priority Basis

This ranking is based on observed usefulness, risk reduction, and cost. The
boundary column reflects the implemented module contract after the
self-cognition plan completed.

| Rank | Consumer | Priority reason from experiments | First useful output | Implemented boundary |
|---|---|---|---|---|
| 1 | Self-cognition tracking | Every useful outcome needs a place to land. Current no-dialog cognition residue is ephemeral, so tracking is the foundation for both silence and action. | run, trigger, evidence ref, route effect | local tracking artifacts |
| 2 | Action-attempt lifecycle and runtime agency controls | The expired-promise case can re-trigger every idle tick unless attempts, idempotency, retry, and duplicate suppression exist first. | action_attempt with stable idempotency key | local attempt ledger |
| 3 | Proactive preview / outbox candidate | The clearest character-growth benefit was a past-due commitment becoming one natural outward message candidate. | preview/outbox candidate tied to an action_attempt | superseded preview-only boundary; see banner |
| 4 | Progress-maintenance candidate | Before-due commitments and settled conversations should still preserve short-term continuity without sending. | `progress_maintenance` route effect | local route effect only |
| 5 | Audit and evaluation | The system cannot be trusted without inspecting why the loop sent, held, or stayed silent. This is cheap and mandatory for quality. | readable loop trace and route explanation | local loop trace |
| 6 | RAG / research planning | Topic-triggered RAG can make the loop more autonomous, but it should come after commitment/action routing is stable. | bounded RAG request/result linked to the run | read-only production/RAG call |
| 7 | Future-cognition scheduler handoff | This creates a delayed thinking opportunity without prewriting user-visible text. | future cognition request | private scheduler slot; later shared cognition decides whether to speak |
| 8 | Reflection / growth projection | Growth is important, but hourly self-cognition should not directly mutate personality; only compact projections may feed slower systems. | growth_candidate projection | future approved projection only |
| 9 | Source-aware consolidation / production projections | This has the widest blast radius because it touches existing persistence semantics. It is useful later, not needed to prove the loop. | consolidation/progress/history projection candidate | future approved projection only |

## Example Outcomes

### No-Action Private Thought

```text
Recent state:
User and Kazusa ended a playful exchange cleanly.

Self-cognition:
The thread is settled. Keep the warm teasing tone available, but do not reopen
the topic unless the user brings it back.

Consumers:
self-cognition tracking: write run record
progress_maintenance candidate: avoid reopening unless the user brings it back
outbox/action attempt: none
production handoff: none
```

### Autonomous Reminder

```text
Recent state:
Kazusa privately recognizes that a planned departure reminder is due.

Self-cognition:
This is the promised reminder moment.

Preview:
喂，到点了。别磨蹭啦，钥匙和外套都带上。

Consumers:
self-cognition tracking: write run and trigger records
action_attempt: new idempotency key for this due occurrence
outbox candidate: ready if structural and cadence mechanics pass
production delivery: superseded preview-only boundary; current selected
  `speak` delivery is defined by the canonical self-cognition README
progress effect: candidate update after delivery, not a direct write here
```

### Topic-Triggered Research And Follow-Up

```text
Recent state:
The user previously mentioned wanting to go to a concert but never confirmed
the time. Kazusa is idle and the topic remains salient.

Loop iteration 1:
self-cognition topic: Should I follow up about the concert?
RAG2:
  Recall recent conversation evidence.
  Use live context or web search if the event time/status is unknown.

Loop iteration 2:
shared cognition decides whether to send, hold, or do nothing.
```
