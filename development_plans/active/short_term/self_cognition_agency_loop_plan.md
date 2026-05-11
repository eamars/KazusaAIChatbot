# self cognition agency loop plan

## Summary

- Goal: Define self-cognition as an idle cognition and agency loop that reuses
  the current multi-source cognition core and routes its output into existing
  consumers: conversation progress, proactive output, dispatcher/scheduler,
  reflection/growth, and audit.
- Plan class: high_risk_migration
- Status: draft
- Runtime behavior change: none until a future approved implementation stage.
- Database schema change: future stages likely need proactive outbox and action
  audit persistence; this draft does not authorize schema work.
- Prompt change: future stages likely need a self-cognition prompt variant and
  proactive-preview rendering contract; this draft does not authorize prompt
  edits.
- New LLM call: future stages may add idle background calls only; no live
  `/chat` response-path call increase is allowed.
- Mandatory skills for future executable stages: `development-plan-writing`,
  `local-llm-architecture`, `no-prepost-user-input`, `py-style`,
  `test-style-and-execution`, `database-data-pull` when using real data, and
  `cjk-safety` before editing Python files that contain CJK prompt text.
- Highest-risk areas: unbounded autonomous contact, bypassing the current
  L1/L2/L3 cognition stack, treating self-generated thought as user evidence,
  writing private thought into user memory, direct adapter sends, unbounded idle
  loops, duplicate scheduled messages, spammy contact cadence, and audit gaps.

## Context

The earlier self-cognition reference under
`development_plans/reference/designs/self_cognition_loop_architecture.md`
correctly identified the main boundary: self-cognition must enter through
`CognitiveEpisode` and the production `call_cognition_subgraph` path. That
reference was too narrow in one important way: it treated self-cognition mostly
as private growth or audit. The intended target is broader.

Self-cognition should let the character privately reprocess recent state and
decide whether anything should happen next. That "anything" may be:

- no visible action;
- short-term conversation progress cleanup;
- additional retrieval over memory, conversation history, pending commitments,
  live context, or the public web;
- a proactive message candidate;
- a scheduled action request;
- a follow-up self-cognition step triggered by a discovered topic;
- slower relationship or personality-growth evidence;
- an audit/evaluation artifact.

Self-cognition can end with the decision to send a message, the same way a
person's cognition can end with outward action. The runtime still needs
mechanical execution infrastructure so that "I will send this" becomes a real,
auditable adapter action instead of an untracked side effect:

- `src/kazusa_ai_chatbot/cognition_episode.py` already defines
  `output_mode="preview"` and `output_mode="scheduled_action_request"`.
- `src/kazusa_ai_chatbot/internal_thought_cognition.py` already accepts
  `think_only`, `preview`, and `silent` output modes for non-chat cognition.
- `src/kazusa_ai_chatbot/conversation_progress/` already owns short-term
  episode continuity used by L3 cognition.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` and
  `src/kazusa_ai_chatbot/rag/` already own full RAG2 retrieval, including
  memory, conversation, recall, live-context, continuation, and web-search
  helper paths.
- `src/kazusa_ai_chatbot/proactive_output/` already owns contract/test-only
  proactive preview records, deterministic record checks, idempotency fields,
  target matching, and outbox shapes. It does not yet own persistent outbox
  storage or live scheduler handoff.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py` already
  owns post-cognition consolidation, but current origin construction is
  user-message-only and must become source-aware before self-cognition can use
  it.
- `src/kazusa_ai_chatbot/dispatcher/` already owns validated `send_message`
  task dispatch.
- `src/kazusa_ai_chatbot/scheduler.py` already owns delayed execution through
  registered adapters.

The right direction is to improve these owners, not add a separate
self-cognition module that competes with them.

## Industrial Research Basis

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

Research sources inspected for this plan:

- Anthropic, [Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents).
  Key lesson: use simple, composable patterns; distinguish fixed workflows from
  agents that dynamically direct tool use; route complex tasks to specialized
  downstream processes when the categories are clear.
- Anthropic, [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents).
  Key lesson: tools for agents need clear purpose, namespacing, meaningful
  context, token-efficient responses, evaluations, and strict input/output
  contracts.
- OpenAI Agents SDK, [Guardrails](https://openai.github.io/openai-agents-python/guardrails/).
  Key lesson: input, output, and tool guardrails run at different workflow
  boundaries; side-effect tools need tool-level checks, not only final-output
  checks.
- OpenAI Agents SDK, [Tracing](https://openai.github.io/openai-agents-python/tracing/).
  Key lesson: production agent runs need traces of LLM generations, tool calls,
  handoffs, guardrails, and custom events.
- OpenAI, [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf).
  Key lesson: guardrails and human intervention are first-class safeguards,
  especially early in deployment.
- LangGraph, [Persistence](https://langchain-5e9cc07a.mintlify.app/oss/python/langgraph/persistence),
  [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop),
  and [Durable execution](https://docs.langchain.com/oss/python/langgraph/durable-execution).
  Key lesson: long-running or approval-based agents need persisted state, pause
  and resume, idempotency, and side effects wrapped so they are not repeated on
  replay.
- Microsoft AutoGen, [Human-in-the-loop](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/human-in-the-loop.html).
  Key lesson: agent teams need explicit handoff/feedback points; blocking
  human approval is useful for short, sensitive decisions but should be used
  carefully.

Applied to this project:

- Kazusa's L1/L2/L3 cognition is the reasoning layer.
- `proactive_output` is an action-preview and outbox boundary, not the source
  of Kazusa's social agency. It may provide structural checks, duplicate
  suppression, and audit records, but the decision to send must come from
  cognition.
- `dispatcher` and `scheduler` are the validated tool execution layer.
- `conversation_progress`, proactive outbox records, scheduler rows, and
  reflection run documents are the durable/resumable state surfaces.
- Logs and future outbox/audit rows are the trace layer.

## Explicit Goal

The self-cognition agency loop should answer:

> While idle, what does Kazusa privately decide needs to be preserved, updated,
> postponed, or acted on, and which existing consumer is allowed to receive that
> decision?

The loop must preserve this invariant:

```text
self-cognition may propose
deterministic code validates mechanics
existing owners persist or execute
adapters deliver only after dispatcher validation and scheduler execution
```

## Consumer Map

### Consumer 1: Audit And Evaluation

Purpose: prove what the cognition stack thought without changing runtime state.

Current owner:

- `src/kazusa_ai_chatbot/internal_thought_cognition.py`
- `src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py`

Valid input:

- idle source packet;
- selected target scope;
- current conversation progress;
- recent bounded evidence;
- current character/runtime state;
- current user relationship evidence when scoped to one user.

Valid output:

- cognition output keys and values;
- selected prompt variants;
- source packet digest;
- no persistence side effect except explicit audit artifact.

Example:

```text
Input:
Kazusa privately replays the last few QQ turns with user 673225019.
The conversation ended with a playful pickup/drop-off correction and a future
"next week" light obligation.

Self-cognition output:
"I should keep the teasing accountability alive, but not reopen it every idle
hour. If the user brings up next week, continue naturally."

Consumer:
Audit artifact only.
```

### Consumer 2: RAG / Research Planning

Purpose: let one self-cognition topic trigger real retrieval work before the
character decides whether to act.

Current owner:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`
- `src/kazusa_ai_chatbot/rag/live_context_agent.py`
- `src/kazusa_ai_chatbot/rag/web_search_agent.py`

Required improvement:

- Let idle self-cognition build a normal RAG request payload with a
  source-aware query such as "What does Kazusa need to know before deciding
  whether to contact this user about X?"
- Reuse RAG2's existing slot, dispatcher, helper-agent, continuation, and
  finalizer contracts.
- Keep RAG bounded. The existing RAG continuation cap and web-search retry caps
  are the model for any wider self-cognition loop budget.
- Treat retrieved facts as evidence for cognition, not as actions by
  themselves.

Example:

```text
Self-cognition topic:
"He mentioned a local concert might be tonight. Should I ask him about it?"

RAG request:
Live-context: answer current schedule/status for the named concert or venue.
Recall: find recent conversation evidence that the user wanted reminders or
follow-up about this event.

RAG result:
The event is tonight at 19:30; prior conversation suggests the user was
interested but did not make a clear plan.

Next consumer:
Run shared cognition again with the RAG result before deciding whether to send,
hold, or only update conversation progress.
```

### Consumer 3: Conversation Progress

Purpose: improve short-term flow for the next live chat without durable
personality or user-memory writes.

Current owner:

- `src/kazusa_ai_chatbot/conversation_progress/`

Required improvement:

- Make `conversation_progress` source-aware. It currently records completed
  responsive turns and assumes `final_dialog` exists. It should accept an
  `idle_self_cognition` maintenance source without pretending a new user turn
  occurred.

Allowed fields for hourly idle updates:

- `current_thread`
- `topic_momentum`
- `episode_phase`
- `open_loops`
- `resolved_threads`
- `avoid_reopening`
- `emotional_trajectory`
- `next_affordances`
- `progression_guidance`

Forbidden fields for hourly idle updates:

- `assistant_moves`, unless a real assistant message was sent;
- `last_user_input` as if a new user message occurred;
- durable facts, promises, user-memory units, or character profile fields.

Example:

```text
Input:
Kazusa sees the previous chat has an old "next week don't be late" tease.

Self-cognition conclusion:
Keep it as a soft open loop for the next relevant turn, but suppress repeated
proactive nagging.

Conversation progress update:
open_loops: ["下周接送/别迟到的玩笑约定仍可自然承接"]
avoid_reopening: ["如果用户没有提到下周，不要每次主动催迟到"]
progression_guidance: "下次用户接续时用轻微调侃承接，不要重启压力。"
```

### Consumer 4: Proactive Preview / Outbox

Purpose: let Kazusa form a candidate outward message from cognition and keep
that action inspectable until it is handed to dispatcher/scheduler for real
delivery.

Current owner:

- `src/kazusa_ai_chatbot/proactive_output/`

Required improvement:

- Move from contract/test-only records to production persistence and runtime
  outbox handling.
- Keep `ProactivePreviewRecord` as the candidate boundary.
- Remove reliance on the Stage 10 permission-check helper as the outbound
  decision point. Future implementation may reuse only structural checks that
  remain useful, such as target shape, adapter availability, blank text
  rejection, and idempotency.

Valid proactive preview:

```text
trigger_source: internal_thought or future self_cognition source
output_mode: preview
visibility: model_visible
target: exact platform/channel/user chosen by cognition and structurally
validated by runtime
preview_text: public message candidate
```

Example send candidate:

```text
Self-cognition:
"I should remind him before leaving."

Preview text:
"喂，到点了。别磨蹭啦，钥匙和外套都带上。"

Execution checks:
target is resolved, adapter is available, idempotency is new, preview text is
not blank, and cadence limits do not classify the send as spammy repetition.

Outbox:
ready for dispatcher/scheduler handoff.
```

Example held candidate:

```text
Self-cognition:
"I kind of want to ask why he has been quiet."

Preview text:
"你今天怎么这么安静？"

Execution checks:
This may still be held or skipped if the target is unresolved, adapter is
unavailable, the outbox already has an equivalent pending send, or cadence
limits say the contact would be repetitive.

Outbox:
held or dry_run only. No scheduler event. No adapter send.
```

### Consumer 5: Dispatcher And Scheduler

Purpose: execute cognition-selected real-world interaction through the existing
`send_message` tool path.

Current owners:

- `src/kazusa_ai_chatbot/dispatcher/`
- `src/kazusa_ai_chatbot/scheduler.py`
- runtime adapter registry in `src/kazusa_ai_chatbot/service.py`

Required improvement:

- Add a proactive-output handoff into the existing dispatcher path.
- Do not call adapters directly from self-cognition or `proactive_output`.
- Convert a ready outbox item into a validated `RawToolCall` for
  `send_message`, then let `TaskDispatcher` and `scheduler` do the normal
  validation, deduplication, persistence, and delivery.

Example:

```text
Approved outbox:
platform: qq
platform_channel_id: 673225019
channel_type: private
preview_text: "喂，到点了。别磨蹭啦，钥匙和外套都带上。"

RawToolCall:
tool: send_message
args:
  target_platform: qq
  target_channel: 673225019
  target_channel_type: private
  text: "喂，到点了。别磨蹭啦，钥匙和外套都带上。"
  execute_at: current UTC or future UTC

Execution:
TaskDispatcher -> scheduler -> registered QQ adapter.
```

### Consumer 6: Source-Aware Consolidation

Purpose: close the loop after idle cognition, retrieval, and possible action
planning without pretending the source was a user message.

Current owner:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`

Required improvement:

- Extend consolidation origin construction beyond
  `build_user_message_consolidation_origin(...)`.
- Add a self-cognition origin that carries source episode id, target scope,
  source topic, RAG evidence lineage, action/outbox ids, and loop iteration id.
- Make per-write origin policy source-aware. Self-cognition consolidation may
  update conversation progress, action audit/outbox state, and slower growth
  evidence. It must not write user-authored facts unless the evidence really
  comes from user messages or existing user memory.
- Let consolidation schedule outbound work through the existing dispatcher path
  only after cognition produced an action request and mechanical execution
  checks passed.

Example:

```text
Self-cognition result:
"The event is tonight; I should send one light nudge, then stop."

Consolidation writes:
outbox/action audit: ready send_message candidate
conversation_progress: "concert topic handled by proactive nudge"
growth evidence: none
user_memory_units: none

Dispatcher:
send_message is scheduled through the existing TaskDispatcher.
```

### Consumer 7: Reflection And Character Growth

Purpose: slower pattern learning, not hourly outreach.

Current owners:

- `src/kazusa_ai_chatbot/reflection_cycle/`
- `src/kazusa_ai_chatbot/memory_evolution/`
- active draft:
  `development_plans/active/short_term/global_character_growth_from_reflection_plan.md`

Rule:

- Hourly self-cognition should not directly mutate personality.
- Repeated, audited self-cognition outcomes may become evidence for daily or
  slower reflection/growth if a future plan defines the promotion gate.

Example:

```text
Repeated hourly evidence:
Kazusa often chooses playful accountability after the user makes soft promises.

Invalid hourly write:
"Kazusa's personality changed to nagging."

Valid slower candidate:
"In close relationships, Kazusa increasingly uses light teasing to preserve
follow-through without sounding formal."
```

### Consumer 8: Runtime Agency Controls

Purpose: keep autonomous contact bounded and inspectable without adding a
separate proactive permission policy.

Current related owner:

- `src/kazusa_ai_chatbot/proactive_output/` defines preview and outbox record
  shapes but has no production storage or runtime handoff yet.

Required improvement:

- Store proactive preview and outbox records with target scope, trigger source,
  output mode, generated text, idempotency key, cadence metadata, status, and
  audit reason.
- Provide runtime controls that are mechanical rather than semantic:
  per-target cooldown, global rate limit, duplicate suppression, adapter
  availability, operator kill switch, and audit visibility.
- Let cognition decide whether a message should be sent. Deterministic code
  must not reinterpret the social reason with keyword rules.

Example runtime controls:

```text
Allowed by mechanics:
Kazusa decided to send one short check-in; there is no equivalent pending
outbox item and the target has not been contacted by autonomy in the current
cooldown window.

Held by mechanics:
Kazusa generated three similar check-ins within the cooldown window, so only
the first can be scheduled and the rest are recorded as duplicate/cadence-held.
```

## Target Architecture

```text
idle scheduler / reflection worker cadence
  -> self-cognition topic selector
  -> source packet builder
  -> RAG2 supervisor when the topic needs evidence
  -> CognitiveEpisode(trigger_source=internal_thought or self_cognition,
                      output_mode=think_only|preview|scheduled_action_request)
  -> existing call_cognition_subgraph
  -> cognition output
  -> source-aware consolidation
  -> source-aware consumer routing:
       audit/evaluation
       conversation_progress idle maintenance
       proactive preview/outbox
       dispatcher/scheduler handoff
       slower reflection/growth evidence
       optional next self-cognition topic while budget remains
```

The route is source-aware and output-mode-aware. It is not keyword-based.

The loop form is bounded:

```text
loop_state = topic queue + evidence + cognition output + consolidation result

for each idle tick:
  pick at most N topics
  for each topic:
    run RAG only if cognition or topic selection needs evidence
    run shared cognition once
    run source-aware consolidation once
    enqueue at most M follow-up topics
  stop on budget, no follow-up, active live chat, or duplicate/cadence hold
```

The loop is not recursive free association. It is a small agenda loop using
the existing RAG, cognition, consolidation, dispatcher, scheduler, and audit
owners.

## Output Modes

### `think_only`

Use when self-cognition only changes private interpretation or audit.

Allowed consumers:

- audit;
- conversation progress;
- slower reflection/growth evidence.

Forbidden consumers:

- proactive outbox;
- dispatcher;
- scheduler;
- adapter send.

### `preview`

Use when self-cognition proposes public text that may become a real outbound
message.

Allowed consumers:

- proactive preview/outbox;
- mechanical execution validation;
- optional review UI in early rollout;
- dispatcher only after the outbox is `ready`.

Forbidden consumers:

- direct adapter send;
- normal `/chat` `final_dialog` response;
- user memory persistence as if it were user-authored evidence.

### `scheduled_action_request`

Use when self-cognition proposes a future tool action rather than immediate
text.

Allowed consumers:

- mechanical execution validation;
- dispatcher;
- scheduler.

Forbidden consumers:

- direct scheduler insert from cognition;
- direct adapter send;
- new tool execution path outside dispatcher.

## Path To Get There

## Proof Of Concept Design

The next experiment should stay under
`experiments/self_cognition_loop_poc/` and remain non-writing. It should prove
the loop shape before production changes:

```text
agenda topic
-> production RAG2 when evidence is needed
-> production L1/L2/L3 cognition
-> source-aware consolidation candidate artifact
-> action/outbox candidate artifact
-> optional next agenda topic while budget remains
```

The POC should reuse production code where the production contract already
exists:

- RAG: call `persona_supervisor2_rag_supervisor2.call_rag_supervisor(...)`.
- Cognition: call `persona_supervisor2_cognition.call_cognition_subgraph(...)`.
- Conversation progress: read existing exported progress and emit a candidate
  update artifact only.
- Dispatcher handoff: emit a `send_message`-shaped candidate only. Do not call
  `TaskDispatcher.dispatch(...)`.

The POC must not call production consolidation writes yet. Current
consolidation origin construction requires `trigger_source=user_message`, so
the experiment should emit a source-aware consolidation candidate artifact
instead of invoking `call_consolidation_subgraph(...)`. A later production plan
must add a self-cognition origin and per-write policy before real
consolidation is enabled.

POC artifacts:

- `self_cognition_agenda.json`
- `self_cognition_rag_request.json`
- `self_cognition_rag_output.json`
- `self_cognition_cognition_input_after_rag.json`
- `self_cognition_cognition_output.json`
- `self_cognition_consolidation_candidate.json`
- `self_cognition_action_candidate.json`
- `self_cognition_loop_trace.md`

POC example:

```text
Topic:
Should the character follow up on the user's possible concert plan?

RAG2:
Recall recent conversation evidence and use live context or web search if the
event time/status is unknown.

Cognition:
Decide whether to send a light nudge, stay silent, or enqueue a follow-up
topic.

Consolidation candidate:
conversation_progress: mark the concert thread as nudged or cooled down.
action_audit: record the action candidate or held reason.
user_memory_units: no write unless evidence comes from actual user-authored
messages or existing durable user memory.

Action candidate:
outbox-shaped preview plus dispatcher-shaped `send_message` candidate if
cognition chooses contact.
```

### Stage 1: Broaden Internal Thought Into Idle Cognition Runner

Improve the existing `internal_thought_cognition.py` owner.

Required outcome:

- It can run a source-aware idle cognition pass over a real target scope.
- It returns actual cognition output values, not only output keys.
- It supports `think_only` and `preview` as real run modes.
- It remains one bounded cognition call per run.

No new production module is needed.

### Stage 2: Add Idle Agenda Loop On Existing Runtime Owners

Improve the existing scheduler/reflection-worker/internal-thought ownership
instead of adding a separate self-cognition service.

Required outcome:

- Represent an idle run as a bounded agenda: selected target scope, topic,
  evidence state, cognition output, consolidation result, and optional follow-up
  topics.
- Stop on budget, live-chat activity, duplicate topic, repeated held action, or
  no follow-up.
- Persist enough audit state to resume or inspect the loop without replaying
  side effects.
- Keep the default loop tiny: one or two topics per idle tick until real traces
  justify more.

### Stage 3: Reuse Full RAG2 Inside Idle Self-Cognition

Improve the existing RAG2 owners.

Required outcome:

- Let self-cognition send source-aware RAG queries through the existing RAG2
  initializer, dispatcher, helper agents, continuation evaluator, and finalizer.
- Allow live context and web search when the self-cognition topic needs current
  external facts.
- Carry RAG evidence lineage into the next cognition pass and consolidation
  origin.
- Preserve existing RAG caps; do not give idle self-cognition an unbounded
  search budget.

### Stage 4: Add Self-Cognition Prompt Variant

Improve the existing cognition prompt-selection owner.

Required outcome:

- Add a source/prompt variant that frames the input as Kazusa privately
  processing her own recent state, not as Kazusa reading a backend report.
- Preserve the same L1/L2/L3 output schema.
- Keep examples role-neutral; inject character name only through runtime data
  if necessary.

### Stage 5: Make Conversation Progress Idle-Maintainable

Improve the existing `conversation_progress` owner.

Required outcome:

- Add a source-aware record/update contract for `idle_self_cognition`.
- Preserve short-term scope by platform/channel/user.
- Avoid incrementing visible conversation turn semantics for idle-only updates.
- Project the updated state into future L3 cognition exactly through the
  existing `conversation_progress` read path.

### Stage 6: Make Consolidation Source-Aware For Self-Cognition

Improve the existing consolidation owner.

Required outcome:

- Add self-cognition origin construction next to the existing user-message
  origin construction.
- Add per-write origin policy that allows only self-cognition-appropriate
  writes.
- Allow consolidation to consume cognition output, RAG result, action/outbox
  state, and loop metadata.
- Prevent self-cognition from writing user facts, preferences, or promises
  unless the supporting evidence originates from actual user-authored messages
  or existing durable user memory.

### Stage 7: Productionize Proactive Output Storage

Improve the existing `proactive_output` owner.

Required outcome:

- Persist preview records, outbox records, mechanical execution decisions, and
  audit events.
- Keep only deterministic structural checks from Stage 10 that match the
  agency-first design.
- Add idempotency that survives process restarts.
- Keep all proactive candidates inspectable before execution.

### Stage 8: Render Proactive Preview Text Through Existing Voice Stack

Improve the existing dialog/preview rendering boundary.

Required outcome:

- Use the character's dialog voice machinery to render public proactive
  preview text from cognition output.
- Do not return the preview as normal `/chat` `final_dialog`.
- Do not save it as conversation history unless delivery actually succeeds and
  a future plan defines the storage semantics.

### Stage 9: Handoff Ready Outbox To Dispatcher/Scheduler

Improve the existing dispatcher/scheduler handoff.

Required outcome:

- Ready proactive outbox records become `send_message` tool calls.
- `TaskDispatcher` validates target platform, channel type, adapter
  availability, execute time, tool visibility role, and duplicates.
- `scheduler` persists and fires the task.
- adapters deliver through the existing `MessagingAdapter` interface.

### Stage 10: Add Runtime Agency Controls

Improve the existing proactive output and runtime controls.

Required outcome:

- Per-target cooldowns, global rate limits, idempotency, off-switches, and
  audit visibility.
- Reviewable outbox mode for early rollout, with auto-send as a later runtime
  mode once traces show the loop behaves well.
- No semantic permission policy. Cognition owns the social decision to send;
  deterministic code owns execution mechanics.

### Stage 11: Feed Slower Growth Only Through Reflection/Memory Evolution

Improve the existing reflection/growth owners.

Required outcome:

- Self-cognition artifacts may become evidence for daily or slower growth.
- Hourly runs do not directly mutate stable character state.
- Growth projection remains compact, source-detail-free, and reversible.

## Examples

### Example A: No-Action Private Thought

```text
Recent state:
User and Kazusa ended a playful exchange cleanly.

Self-cognition:
"The thread is settled. Keep the warm teasing tone available, but do not reopen
the topic unless the user brings it back."

Consumers:
audit: write run record
conversation_progress: update avoid_reopening and progression_guidance
proactive_output: none
dispatcher/scheduler: none
```

### Example B: Autonomous Reminder

```text
Recent state:
Kazusa privately recognizes that a planned departure reminder is due.

Self-cognition:
"This is the promised reminder moment."

Preview:
"喂，到点了。别磨蹭啦，钥匙和外套都带上。"

Consumers:
audit: write run record
proactive_output: structural and cadence checks pass, outbox ready
dispatcher/scheduler: schedule send_message
conversation_progress: record that the reminder was attempted only after send
```

### Example C: Desire Held By Cadence

```text
Recent state:
User has been quiet. Kazusa privately wants to check in.

Self-cognition:
"I want to ask, but this would be the third similar check-in in the cooldown
window."

Consumers:
audit: write held candidate
proactive_output: cadence-held duplicate/check-in
conversation_progress: optional note not to infer abandonment
dispatcher/scheduler: none
```

### Example D: Topic-Triggered Research And Follow-Up

```text
Recent state:
The user previously mentioned wanting to go to a concert but never confirmed
the time. Kazusa is idle and the topic remains salient.

Loop iteration 1:
self-cognition topic: "Should I follow up about the concert?"
RAG2:
  Recall: retrieve recent conversation evidence about the concert.
  Live-context: check current public schedule/status for the concert or venue.

RAG result:
The event appears to start tonight at 19:30. The user showed interest but did
not explicitly ask for a reminder.

Loop iteration 2:
shared cognition decides: "A light nudge is useful, but only send one and do
not keep pushing."

Consolidation:
outbox/action audit: one ready send_message candidate
conversation_progress: mark the concert thread as proactively nudged
user_memory_units: no new user fact
growth evidence: no immediate personality write

Dispatcher/scheduler:
send_message through the existing `send_message` tool.

Possible message:
"喂，刚想起来你之前说的那个演出好像是今晚。要是真打算去，就别拖到临出门才手忙脚乱哦。"
```

### Example E: Slower Growth Evidence

```text
Repeated evidence:
Across many interactions, playful accountability helps this relationship
continue without pressure.

Hourly action:
No personality write.

Daily/slower candidate:
"Kazusa can preserve closeness by using light teasing as a low-pressure
follow-through cue in trusted relationships."

Consumer:
reflection/memory evolution only after promotion gates.
```

## Mandatory Rules For Future Implementation

- Do not create a standalone self-cognition LLM path.
- Do not create a new proactive send module.
- Do not send directly from self-cognition, dialog, or proactive output.
- Do not add a separate semantic proactive permission policy. Cognition owns
  the social decision to send.
- Do not let deterministic code reinterpret mood, relationship score,
  `logical_stance`, `character_intent`, or natural-language keywords into a
  different social decision.
- Do not build a separate self-cognition retrieval stack. Use existing RAG2
  initializer, dispatcher, helper agents, continuation evaluator, and finalizer.
- Do not bypass consolidation origin policy. Add a self-cognition origin and
  per-write rules before enabling consolidation writes from idle loops.
- Do not route self-generated thought into user memory as user evidence.
- Do not route self-cognition preview text into normal `/chat` response output.
- Do not add LLM calls to the live `/chat` response path.
- Do not run unbounded idle loops. Idle self-cognition must have cadence,
  target selection, caps, and backoff.
- Do not let web search results become durable memory without source lineage
  and source-aware consolidation.
- Do not let an idle update overwrite fresher conversation progress or schedule
  duplicate messages.
- Do not bypass adapter availability, idempotency, cadence limits, or audit.

## Deferred

- New proactive MongoDB collections.
- Multi-recipient proactive fanout.
- Media proactive sends.
- Cross-platform proactive target selection.
- New web-search implementation outside RAG2.
- Personality mutation from hourly self-cognition.
- New tools beyond `send_message`.
- New external orchestrator framework.

## Acceptance Criteria For This Draft

This draft is useful when it:

- identifies all plausible consumers of self-cognition output;
- maps each consumer to an existing owner in this repo;
- explains why direct sends are forbidden;
- applies industry patterns to the current architecture;
- provides a staged path that improves existing modules rather than creating a
  parallel subsystem;
- includes examples of no-action, progress-only, autonomous outreach,
  cadence-held outreach, topic-triggered RAG/web-search follow-up, and slower
  growth.
