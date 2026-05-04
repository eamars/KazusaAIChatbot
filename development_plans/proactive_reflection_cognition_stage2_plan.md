# proactive reflection cognition stage 2 plan

## Summary

- Goal: Convert selected Stage 1c reflection outcomes into safe autonomous character messages through an internal proactive cognition path, without routing through `/chat` and without using raw dispatcher `send_message` as the reasoning layer.
- Parent: `reflection_memory_integration_stage1c_plan.md`
- Plan class: large
- Status: directional draft. Do not implement until Stage 1c execution evidence is reviewed and this plan is revised.
- Mandatory skills: `local-llm-architecture`, `development-plan-writing`, `no-prepost-user-input`, `py-style`, `test-style-and-execution`, `database-data-pull`
- Overall cutover strategy: additive, gated, and opt-in. Start with dry-run decisions and outbox records; enable actual transport only after evidence shows low false-positive rate.
- Highest-risk areas: unsolicited or repetitive messages, private-to-group leakage, bypassing character cognition, stale reflection candidates, adapter limitations for private sends, duplicate delivery, and persistence mismatch after send failure.
- Acceptance criteria for the future executable plan: a reflection candidate can become a typed proactive cognition event, pass a speak/defer/drop gate, generate dialog through the character voice path, send through an adapter only after approval, persist the assistant row, and feed normal progress/consolidation without using `/chat`.

## Context

Stage 1c intentionally stops before autonomous messaging. That boundary keeps reflection safe while it proves whether hourly/daily analysis produces useful future-turn guidance.

Stage 2 is the later bridge from reflection to action. The user specifically prefers the reflection message path to live behind L1/L2 or adjacent to cognition, not through `/chat`.

Relevant existing findings:

- Current dispatcher `send_message` is a transport tool. It can send raw text through adapters, but it bypasses cognition and does not naturally persist an assistant row into `conversation_history`.
- Current `/chat` path is user-input oriented and should not be reused for internal reflection impulses.
- Production cognition is layered around L1, L2, L3, and dialog. L1 is currently shaped around external user input.
- `experiments/cognition_core_next` provides useful future patterns:
  - typed percepts
  - action latch
  - private cognition progress
  - group scene progress
  - external reflection/reality check
  - bounded internal reflection loop
  - L3 evaluator for no private thought leak and no intent drift

Stage 2 should borrow those patterns carefully after Stage 1c has real reflection output.

## Mandatory Skills

- `local-llm-architecture`: load before designing or changing proactive cognition prompts, event flow, action latch, or L2/L3 boundaries.
- `development-plan-writing`: load before promoting this directional draft into an executable implementation plan.
- `no-prepost-user-input`: load before adding rules that interpret reflection candidates as commitments, permissions, or user preferences.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `database-data-pull`: load before sampling Stage 1c reflection outputs or validating real proactive-candidate quality.

## Mandatory Rules

- Do not implement this plan until Stage 1c is complete and reviewed.
- Do not route proactive reflection messages through `/chat`.
- Do not treat dispatcher `send_message` as the reasoning workflow. It may be used only as final transport after proactive cognition approves and renders a message.
- Do not let reflection output become dialog directly. Every outgoing message must pass a cognition gate and voice generation path.
- Do not send messages from daily reflection by default. Daily reflection may generate candidates; a separate gate decides whether any candidate is timely enough to speak.
- Do not mix private and group scopes.
- Do not send group messages based on private-channel reflection.
- Do not send private messages unless the adapter supports the correct private channel type and the target identity is explicit.
- Do not create deterministic keyword rules that decide semantic urgency from user text. Use LLM-owned judgment with schema validation and hard safety limits.
- Every proactive event must be idempotent and traceable to a Stage 1c reflection run and evidence refs.
- Every actual send must persist or attempt to persist an assistant conversation row with typed addressees.
- Every actual send must trigger the same post-response progress/consolidation family used by normal assistant turns, or record why that was intentionally skipped.
- The live cognition loop has priority. Proactive cognition must defer while user turns are queued or processing.

## Stage 2 Entry Criteria

Before this plan can become executable:

- Stage 1a read-only reflection has real LLM approval evidence.
- Stage 1b memory evolution is complete.
- Stage 1c hourly reflection has run over real data for at least several cycles.
- Stage 1c daily reflection has run at least once.
- Stage 1c execution evidence includes:
  - false-positive candidate review
  - private/group isolation evidence
  - prompt-facing context behavior
  - DB query/index behavior
  - LLM latency under real load
- A human review decides which candidate types are eligible for proactive messages.
- Adapter capabilities are confirmed for group and private sends.

## Stage 2 Non-Goals

- Do not implement per-message self-reflection unless Stage 1c evidence proves hourly/daily reflection misses valuable repair opportunities.
- Do not rebuild the entire production cognition graph from `experiments/cognition_core_next`.
- Do not replace the existing dispatcher/scheduler.
- Do not build an admin UI unless candidate review requires it later.
- Do not make the character initiate messages from arbitrary daily mood alone.

## Deferred

- Per-message reflection remains parked unless Stage 1c evidence proves a concrete value gap.
- Broad production autonomous messaging is deferred until private/group privacy behavior is proven.
- Cross-channel proactive topic carryover is deferred.
- Direct reflection-to-memory promotion is deferred.
- Full `cognition_core_next` graph merge is deferred.
- Admin review tooling is deferred unless dry-run review volume justifies it.

## Cutover Policy

| Area | Policy | Notes |
|---|---|---|
| Stage 1c reflection docs | read-only source | Stage 2 reads candidates and evidence; it does not mutate completed reflection output. |
| `/chat` | unchanged | Proactive turns never enter through `/chat`. |
| Dispatcher transport | final send only | Dispatcher/adapter may send approved text, but does not reason about whether to speak. |
| Proactive events | additive | Durable outbox justified for user-visible action idempotency. |
| Cognition graph | additive entrypoint | Existing user-turn cognition remains the default path. |
| Production sending | opt-in | Start with dry-run, then allowlisted transport, then limited scopes. |
| Consolidator | compatible | Reuse post-response recording where possible; special handling must be explicit. |

## Agent Autonomy Boundaries

- The future implementation agent must not execute this draft as-is.
- The agent may revise this draft only after reviewing Stage 1c execution evidence.
- The agent must keep proactive events traceable to Stage 1c reflection candidates.
- The agent must not add a fake user message to drive proactive cognition.
- The agent must not implement raw transport sending before dry-run gate and generation quality are reviewed.
- The agent must not broaden scope from private/group reflection candidates to arbitrary autonomous behavior.
- If adapter capability is uncertain, private proactive sends remain disabled.
- If a required persistence or send-failure policy is unresolved, stop and revise the plan before coding.

## Target State

Proactive message flow:

```text
Stage 1c reflection output
  -> eligible candidate extraction
  -> durable proactive cognition event
  -> live chat idle check
  -> proactive L2 gate: drop / defer / speak
  -> context refresh from latest channel tail and scoped reflection
  -> proactive cognition planning
  -> dialog generation and evaluator
  -> final transport send
  -> assistant row persistence
  -> progress/consolidation recording
  -> action latch update
```

The important architectural property:

```text
reflection candidate is not the message
reflection candidate is a percept for cognition
```

## Design Decisions

1. Stage 2 is blocked on Stage 1c evidence.
2. Proactive messages are generated from typed internal percepts, not fake user input.
3. `/chat` is not part of the proactive path.
4. Dispatcher/adapter send is transport only, after cognition approval.
5. A durable proactive event outbox is acceptable because it is delivery state, not an activity cache.
6. Reflection candidates are never dialog text.
7. A proactive L2 gate decides `drop`, `defer`, or `speak`.
8. Transport stays disabled through candidate extraction, gate, and preview-generation dry-runs.
9. Private proactive sends require verified adapter support and explicit target identity.
10. Action latch or equivalent cooldown control must be decided before any broad enablement.

## Design Direction

### Event source

Stage 1c `character_reflection_runs.output` may contain future-message candidates, but Stage 2 should not scan and mutate reflection docs as a queue.

Preferred Stage 2 addition:

```text
proactive_cognition_events
```

This collection is not an activity cache. It is durable delivery/claim state for potentially user-visible actions. It is justified because autonomous sends need idempotency, retries, audit trail, and explicit status.

Initial statuses:

```text
candidate
claimed
dropped
deferred
approved_to_speak
sent
failed
cancelled
```

### Event shape

```python
class ProactiveCognitionEventDoc(TypedDict, total=False):
    event_id: str
    source_reflection_run_id: str
    source_candidate_id: str
    status: str
    platform: str
    platform_channel_id: str
    channel_type: Literal["private", "group"]
    target_global_user_id: NotRequired[str]
    privacy_scope: Literal["private_user", "group_public"]
    candidate_intent: str
    candidate_reason: str
    evidence_refs: list[ReflectionSourceRef]
    created_at: str
    eligible_after: str
    expires_at: str
    claimed_at: str
    completed_at: str
    drop_reason: str
    sent_message_id: str
```

Required indexes:

```text
proactive_event_id_unique:
  [("event_id", 1)], unique

proactive_event_ready:
  [("status", 1), ("eligible_after", 1), ("expires_at", 1)]

proactive_event_source_unique:
  [("source_reflection_run_id", 1), ("source_candidate_id", 1)], unique
```

### Proactive cognition input

Use typed internal percepts rather than fake user input:

```python
class ProactiveCognitionInput(TypedDict):
    event: ProactiveCognitionEventDoc
    reflection_context: ReflectionContextPromptDoc
    recent_channel_tail: list[ConversationMessageDoc]
    current_progress: ConversationProgressPromptDoc
    action_latch: ActionLatchDoc
```

Do not populate `user_input` with reflection text. If existing cognition stages require a string, add a separate `turn_origin="proactive_reflection"` and `internal_percept_text` field rather than pretending a user said it.

### Gate

The first Stage 2 cognition step is a proactive L2 gate.

Output:

```python
class ProactiveGateDecision(TypedDict):
    decision: Literal["drop", "defer", "speak"]
    reason: str
    urgency: Literal["low", "medium", "high"]
    user_benefit: str
    interruption_risk: Literal["low", "medium", "high"]
    privacy_risk: Literal["low", "medium", "high"]
    required_context_refresh: list[str]
    defer_until: NotRequired[str]
```

Hard deterministic guards are allowed only for structure:

- expired event
- missing target for private send
- unsupported adapter send mode
- duplicate event
- live chat not idle
- missing evidence refs

Semantic decision remains LLM-owned.

### Cognition placement

Preferred direction:

- Do not reuse external-user L1 unchanged.
- Add a proactive cognition entrypoint at or just before L2.
- Use an internal percept builder that borrows from `experiments/cognition_core_next/l1.py` without importing experiment code directly.
- Feed the proactive event, reflection context, current progress, and latest channel tail into a conscious planning stage.
- Use existing L3/dialog generation where possible after the proactive plan has a normal `character_intent`, `logical_stance`, and action directives.
- Add an evaluator equivalent to the experiment L3 evaluator:
  - no private thought leak
  - no meta/system mention
  - no stale topic
  - no target mismatch
  - no intent drift from gate decision

### Action latch

Add or reuse a small durable latch so the character does not repeat itself.

Minimum fields:

```python
class ActionLatchDoc(TypedDict, total=False):
    latch_id: str
    scope_key: str
    last_proactive_event_id: str
    last_intent: str
    last_dialog_preview: str
    last_sent_at: str
    cooldown_until: str
```

If Stage 1c reflection docs already provide enough repetition control, this can be deferred. The executable Stage 2 plan must decide based on Stage 1c evidence.

## Change Surface Direction

Likely create:

- `src/kazusa_ai_chatbot/proactive_cognition/__init__.py`
- `src/kazusa_ai_chatbot/proactive_cognition/models.py`
- `src/kazusa_ai_chatbot/proactive_cognition/repository.py`
- `src/kazusa_ai_chatbot/proactive_cognition/gate.py`
- `src/kazusa_ai_chatbot/proactive_cognition/runtime.py`
- `src/kazusa_ai_chatbot/proactive_cognition/worker.py`
- `src/kazusa_ai_chatbot/proactive_cognition/README.md`

Likely modify:

- `src/kazusa_ai_chatbot/db/bootstrap.py`: event/outbox indexes if approved.
- `src/kazusa_ai_chatbot/db/schemas.py`: event/latch schemas.
- `src/kazusa_ai_chatbot/service.py`: worker lifecycle and shared idle callback.
- Cognition schema and prompt modules: add `turn_origin` and proactive percept fields.
- Dialog/evaluator path: support proactive generation without fake user text.
- Adapter send bridge: expose safe final transport call with channel-type correctness.
- Conversation persistence: save proactive assistant rows with `body_text`, typed addressees, broadcast flag, and raw wire text if available.
- Conversation progress/consolidation caller: record proactive assistant turns consistently.

Do not modify unless the executable plan proves need:

- Existing `/chat` request schema.
- Relevance agent for user messages.
- Stage 1c reflection selector.
- Current consolidator write semantics.

## Implementation Phases

### Phase A - Evidence review and candidate policy

1. Review Stage 1c reflection outputs.
2. Classify candidate types:
   - never proactive
   - prompt-context only
   - eligible for proactive private message
   - eligible for proactive group message
3. Estimate false-positive and nuisance rate.
4. Decide cooldown defaults and expiry windows.

Exit criteria:

- Candidate eligibility is documented.
- Stage 2 can name exactly which candidate types may become events.

### Phase B - Event outbox and dry-run extraction

1. Add event schema and repository.
2. Extract eligible Stage 1c candidates into `candidate` events.
3. Enforce idempotency by source reflection run and candidate id.
4. Do not run cognition yet.

Exit criteria:

- Re-running extraction creates no duplicates.
- Events are traceable to reflection evidence.

### Phase C - Proactive gate dry-run

1. Implement proactive L2 gate.
2. Run gate in dry-run mode over candidate events.
3. Store `drop`, `defer`, or `would_speak` decision without sending.
4. Review false positives.

Exit criteria:

- Gate decisions are explainable.
- Most candidates are dropped or deferred unless clearly useful.

### Phase D - Proactive cognition generation dry-run

1. Add proactive cognition input state.
2. Add or adapt planning prompt.
3. Generate candidate dialog through the character voice path.
4. Add evaluator.
5. Store generated preview only.

Exit criteria:

- Generated previews do not leak private thought.
- Generated previews preserve character voice.
- Stale or awkward candidates are rejected.

### Phase E - Controlled transport

1. Add final transport call after gate and evaluator approval.
2. Confirm adapter supports correct channel type.
3. Send only in an allowlisted test channel or private test user first.
4. Persist assistant row after successful send.
5. Record progress/consolidation as a normal assistant-originated turn.

Exit criteria:

- One controlled proactive send completes end to end.
- Duplicate send protection is proven.
- Send failure and persistence failure are logged and recoverable.

### Phase F - Limited production enablement

1. Enable for private scope only or one low-risk group scope.
2. Enforce conservative cooldown.
3. Review real responses and user follow-up.
4. Expand only after evidence.

Exit criteria:

- Proactive sends are rare, useful, and traceable.

## Progress Checklist

- [ ] Stage 1c evidence reviewed.
- [ ] Candidate eligibility matrix approved.
- [ ] Adapter capability checked.
- [ ] Phase A - Evidence review and candidate policy complete.
- [ ] Phase B - Event outbox and dry-run extraction complete.
- [ ] Phase C - Proactive gate dry-run complete.
- [ ] Phase D - Proactive cognition generation dry-run complete.
- [ ] Phase E - Controlled transport complete.
- [ ] Phase F - Limited production enablement complete.
- [ ] Future executable plan acceptance criteria verified.

## Blast Radius And Mitigation

| Risk | Blast radius | Mitigation |
|---|---|---|
| Character spams users | User trust and platform risk | Candidate extraction conservative; L2 gate; cooldown; expiry; dry-run review; opt-in scopes |
| Private reflection leaks to group | Privacy breach | Event privacy scope; group events can cite only group-visible evidence; evaluator checks target/scope |
| Message bypasses character voice | Character quality regression | Reflection is percept only; dialog generated through cognition/voice path; evaluator rejects meta text |
| Duplicate sends | User annoyance | Unique source-candidate index; claim status; action latch; idempotent worker |
| Stale send after topic moved on | Awkward interruption | Expires-at; latest channel tail refresh; gate can defer/drop |
| Live chat latency | User-facing delay | Shared idle callback; one proactive event at a time; no LLM call while queue busy |
| Adapter private-send limitation | Failed or wrong-target sends | Capability check before approval; private sends disabled unless adapter proves support |
| Send succeeds but persistence fails | History gap | Event records transport id and persistence status; retry persistence separately; do not resend text blindly |
| Consolidator misreads proactive message | Memory pollution | Mark assistant row `turn_origin=proactive_reflection`; update consolidator prompt only if needed |
| Local LLM over-approves | Behavioral risk | Human-reviewed dry-run gate metrics before enabling transport |

## Verification Direction

Future executable plan should include focused tests:

```powershell
pytest tests/test_proactive_cognition_event_repository.py
pytest tests/test_proactive_cognition_candidate_extraction.py
pytest tests/test_proactive_cognition_gate.py
pytest tests/test_proactive_cognition_generation.py
pytest tests/test_proactive_cognition_transport.py
pytest tests/test_proactive_cognition_privacy.py
```

Required live validation before broad enablement:

- Candidate extraction dry-run over real Stage 1c reflection docs.
- Gate dry-run with sampled decisions reviewed by a human.
- Preview generation dry-run with no sends.
- One allowlisted transport test.
- Assistant row persistence inspection.
- Progress/consolidation inspection after the proactive row.

## Open Decisions For The Executable Stage 2 Plan

These are intentionally not resolved until Stage 1c evidence exists:

- Which Stage 1c candidate types are eligible for proactive messages.
- Whether an action latch is necessary immediately or can be deferred.
- Whether proactive cognition starts at a new percept builder or adapts part of existing L1.
- Which cognition prompt modules receive `turn_origin`.
- Whether private sends are supported by the current QQ/NapCat adapter path.
- Whether proactive assistant rows need a new explicit `turn_origin` storage field.
- Whether consolidation needs special prompt wording for proactive assistant-originated turns.

## Acceptance Criteria For Future Revision

This directional draft becomes implementation-ready only when it is revised to include:

- A fixed candidate eligibility matrix based on Stage 1c evidence.
- Exact file-level change surface.
- Exact data schemas and indexes.
- Exact prompt integration points.
- Exact adapter capability decision.
- Exact persistence order for send success and failure.
- Focused tests and manual validation commands.
- A production enablement policy with cooldowns and allowlists.

## Execution Evidence

Do not fill this section until this draft is promoted to an executable plan:

- Stage 1c evidence reviewed:
- Candidate policy approved:
- Adapter capability checked:
- Commands run:
- Test results:
- Dry-run quality review:
- Residual risks:
