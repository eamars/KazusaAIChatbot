# conversation progress flow phase2 plan

## Summary

- Goal: Evolve `kazusa_ai_chatbot.conversation_progress` from mostly anti-repeat guidance into a bounded short-term conversation-flow continuity layer.
- Plan class: medium
- Status: completed
- Overall cutover strategy: compatible
- Highest-risk areas: context bloat, making Kazusa sound mechanical, accidentally turning the module into a reply generator, stale flow guidance after topic transitions, overusing raw history as short-term memory, and adding deterministic semantic filters over user text.
- Acceptance criteria: cognition receives compact flow guidance that helps Kazusa continue, deepen, resolve, or gracefully close an episode without increasing response-path latency or changing relevance behavior. Raw recent history is reduced to a small surface/tone buffer and is no longer the primary source of semantic short-term continuity.

## Context

Phase 1 proved that `conversation_progress` can reduce harmful repeated response moves by tracking short-lived operational state after relevance and before cognition. The next useful evolution is not stronger anti-repeat logic. It is better conversation flow.

The module should help Kazusa maintain a short-term sense of:

- what episode the conversation is currently in,
- what the user appears to be trying to accomplish,
- what has changed since earlier turns,
- what remains unresolved,
- what has already been addressed enough,
- what kind of next human move would feel natural.

This remains short-term operational memory. It is not long-term user profile memory, not a dialogue manager, and not a response generator.

The architectural goal is to stop using raw recent history as the main short-term memory mechanism. Raw history is still valuable for local wording texture, exact recent phrasing, and last-turn topic-drift checks, but it should not be the primary way cognition reconstructs episode progress. In Phase 2, `conversation_progress` becomes the canonical semantic short-term memory, while raw history becomes a small local surface/tone buffer.

## Real Conversation Pattern Check

Before implementation, validate payload fields against real `conversation_history` patterns, not only curated support/task fixtures.

Initial diagnostic sample:

```powershell
venv\Scripts\python.exe -m scripts.export_collection conversation_history --sort '{"timestamp":-1}' --limit 500 --output test_artifacts/conversation_history_pattern_sample_recent500.json
```

Observed in this sample:

- 500 recent rows across 8 channel groups.
- 492 user rows and 8 assistant rows, so the sample is mostly human group-chat behavior.
- 292 messages were 12 characters or shorter.
- 79 messages carried reply context.
- 455 adjacent timestamp gaps were under 60 seconds.
- Consecutive user-message runs were common; the largest run was 409 user messages without an assistant row inside the sampled group history.
- The sample contained rapid topic movement, teasing/provocation, bot meta-discussion, reply chains, food/local-life chatter, model/tooling chatter, short fragments, empty/media-like rows, and third-person discussion about Kazusa/bots.

Payload implication:

- A task-shaped payload with only `user_goal` and `current_blocker` is not sufficient.
- The module needs fields that represent casual/social mode, quick pivots, active thread, and conversational momentum.
- These fields must be semantic descriptors emitted by the recorder LLM, not code-side keyword classifiers.

## Mandatory Rules

- Preserve Phase 1 lifecycle: relevance first, then load progress, then cognition/dialog, then background recorder.
- Relevance must not receive `conversation_progress`, `conversation_episode_state`, or Phase 2 flow fields.
- Do not block `/chat` response on recorder completion.
- Do not feed full raw conversation history into cognition to solve flow.
- Do not replace raw history everywhere. Suppress it from semantic decision layers first, while preserving a tiny local surface/tone window where the layer genuinely needs exact recent wording.
- `conversation_progress` owns semantic continuity. Raw recent history owns only surface adjacency: recent phrasing, tone, cadence, and last-user-message drift checks.
- Do not add deterministic keyword matching, regex, semantic string filters, or code-side classifiers over user or bot natural-language text.
- LLMs own semantic interpretation of episode phase, user goal, blocker, resolved threads, emotional trajectory, and next affordances.
- Deterministic code may validate shape, cap string lengths, cap list lengths, preserve timestamps, apply TTL, and drop/truncate over-budget fields by structural policy.
- The module must not generate Kazusa's reply text.
- Content Anchor Agent remains the consumer that decides the response skeleton; Dialog Agent remains the voice layer.
- Phase 2 must fit the user's 50k context budget by keeping `conversation_progress` below a hard maximum.

## Must Do

- Extend the existing `conversation_progress` module, not create a second competing module.
- Add bounded flow fields to the stored episode state and prompt-facing projection.
- Add deterministic size caps for every prompt-facing string and for the total projected `conversation_progress` payload.
- Add an explicit raw-history exposure policy per cognition/dialog layer so implementation agents do not choose ad hoc history windows.
- Update the recorder prompt so it emits flow fields as compact semantic guidance.
- Update Content Anchor Agent prompt so it uses flow guidance to choose the next conversational move.
- Suppress raw history from Content Anchor; it must use `decontexualized_input`, `rag_result`, stance/intent, and `conversation_progress` for semantic continuity.
- Reduce raw history in style/dialog layers to the smallest local surface/tone window that preserves naturalness.
- Add tests that prove the new fields are capped, projected, ignored on sharp transitions, and used by cognition without leaking into relevance.
- Add live LLM A/B tests focused on conversation flow, not only anti-repeat.

## Deferred

- Do not redesign the cognition graph.
- Do not remove Phase 1 anti-repeat fields.
- Do not replace RAG2, Cache2, relevance, Style Agent, or Dialog Evaluator behavior.
- Do not persist Phase 2 flow fields into `user_profiles`, `user_profile_memories`, or `memory`.
- Do not add a full finite-state dialogue manager.
- Do not make `episode_phase` a deterministic state machine.
- Do not add domain-specific playbooks for illness, debugging, essay revision, baking, or thesis writing.
- Do not remove every raw-history usage in one step. Raw history remains allowed for tone and immediate adjacency until tests prove a narrower replacement.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Module boundary | compatible | Keep the existing `kazusa_ai_chatbot.conversation_progress` public facade. Extend only the returned/stored payload contract; do not add new public entrypoints. |
| DB document shape | compatible | Add optional fields; loaders must tolerate old documents that lack Phase 2 fields. |
| Prompt projection | compatible | Missing Phase 2 fields project to empty strings/lists. |
| Relevance | unchanged | Phase 2 flow state remains invisible to relevance. |
| Recorder | compatible | Background recorder emits additional fields; response path remains non-blocking. |
| Cognition | compatible | Content Anchor may use new flow fields but must still work when they are absent. |

## Agent Autonomy Boundaries

- The implementation agent may choose small helper names only when they preserve this plan's public contract.
- The agent must not introduce new architecture, alternate persistence, or extra prompt stages.
- The agent must not use code-side keyword rules to infer flow fields.
- The agent must leave this plan in draft until the user accepts the proposed payload contract extension.
- If context-budget verification fails, the agent must stop and report the payload size instead of raising limits silently.

## Phase 2 Payload Contract Extension

This section is the proposed payload contract extension that requires user agreement before implementation.

Phase 2 does not introduce a new public interface. Existing production callers must continue to call the same facade functions and must not import module internals.

### Public Module Boundary

Keep the existing public facade:

```python
async def load_progress_context(...) -> ConversationProgressLoadResult: ...

async def record_turn_progress(...) -> ConversationProgressRecordResult: ...
```

Do not expose `repository`, `cache`, `projection`, or `recorder` internals to production callers.

Caller integration remains unchanged:

- `service.py` continues to call `load_progress_context(...)` before cognition.
- The background recorder continues to call `record_turn_progress(...)` after final dialog.
- The returned `conversation_progress` payload becomes richer, but the call sites do not switch to a new API.

### Stored State Additions

Add optional fields to `ConversationEpisodeStateDoc`:

```python
conversation_mode: str
episode_phase: str
topic_momentum: str
current_thread: str
user_goal: str
current_blocker: str
resolved_threads: list[ConversationEpisodeEntryDoc]
emotional_trajectory: str
next_affordances: list[str]
avoid_reopening: list[ConversationEpisodeEntryDoc]
```

Allowed `episode_phase` labels:

```text
opening | developing | deepening | pivoting | stuck_loop | resolving | cooling_down
```

Allowed `conversation_mode` labels:

```text
task_support | emotional_support | casual_chat | playful_banter | meta_discussion | group_ambient | mixed
```

Allowed `topic_momentum` labels:

```text
stable | drifting | quick_pivot | fragmented | sharp_break
```

These labels are LLM-emitted and schema-validated only. Code must not derive them from user text.

Field meanings:

- `conversation_mode`: broad interaction mode. This prevents the module from treating every episode as a task or support session.
- `episode_phase`: where the episode is in its local arc.
- `topic_momentum`: whether the conversation is stable, drifting, fragmentary, or rapidly pivoting.
- `current_thread`: neutral one-line description of what is currently being talked about. Unlike `user_goal`, this works for teasing, casual chat, and third-person bot meta-talk.
- `user_goal`: optional; empty when the turn is not goal-driven.
- `current_blocker`: optional; empty when the turn is not problem-solving.

### Prompt-Facing Shape

Extend `ConversationProgressPromptDoc`:

```python
{
    "status": str,
    "episode_label": str,
    "continuity": str,
    "turn_count": int,
    "conversation_mode": str,
    "episode_phase": str,
    "topic_momentum": str,
    "current_thread": str,
    "user_goal": str,
    "current_blocker": str,
    "user_state_updates": [{"text": str, "age_hint": str}],
    "assistant_moves": [str],
    "overused_moves": [str],
    "open_loops": [{"text": str, "age_hint": str}],
    "resolved_threads": [{"text": str, "age_hint": str}],
    "avoid_reopening": [{"text": str, "age_hint": str}],
    "emotional_trajectory": str,
    "next_affordances": [str],
    "progression_guidance": str,
}
```

### Size Budget

Hard caps:

```text
MAX_ENTRY_CHARS = 160
MAX_MOVE_CHARS = 120
MAX_LABEL_CHARS = 80
MAX_THREAD_CHARS = 180
MAX_GUIDANCE_CHARS = 240
MAX_AFFORDANCES = 4
MAX_RESOLVED_THREADS = 5
MAX_AVOID_REOPENING = 5
MAX_PROGRESS_PROMPT_CHARS = 5000
```

The projected prompt payload must stay under `MAX_PROGRESS_PROMPT_CHARS`. If the payload exceeds the cap after per-field truncation, projection must drop lowest-priority optional fields in this order:

```text
resolved_threads -> assistant_moves -> user_state_updates beyond newest 4 -> open_loops beyond newest 3
```

Never drop:

```text
status, continuity, turn_count, conversation_mode, episode_phase, topic_momentum, current_thread, current_blocker, overused_moves, next_affordances, progression_guidance
```

## Raw History Exposure Policy

Phase 2 must treat `conversation_progress` as semantic short-term memory and raw history as a local surface/tone buffer.

Layer policy:

| Layer | Current raw-history use | Phase 2 policy |
|---|---|---|
| L1 Subconscious | none | Keep none. L1 should react to current input only. |
| L2 Consciousness | none | Keep none. Use `rag_result`, profile, commitments, and current input. |
| L2 Boundary/Judgment | none | Keep none. Boundary decisions should not be driven by transcript reconstruction. |
| L3 Content Anchor | none today; uses `conversation_progress` | Keep raw history out. This is the primary consumer of semantic flow fields. |
| L3 Contextual Agent | `chat_history_recent` | Replace full recent history with a compact social surface window: at most last 2 current-user/bot turns, or later a dedicated `social_context` projection. |
| L3 Style Agent | `chat_history_recent` | Keep only local wording buffer: last assistant turn plus current/last user turn where available. Use it for phrase/cadence only. |
| Dialog Generator | `tone_history` from recent interaction history | Reduce to last assistant turn plus any immediately adjacent user turn. Do not feed a 10-message window for semantic continuity. |
| Dialog Evaluator | last user message extracted from recent history | Keep only `last_user_message`; do not pass full history. |
| Background recorder | recent trace evidence | Keep bounded recent history because the recorder updates semantic state after the response, not on the response path. |

If a layer needs a broader history window after this reduction, the implementation agent must add a test showing the specific failure and get user approval before widening it.

## LLM Workload Budget

Phase 2 must not add new response-path LLM calls. It changes payload shape and raw-history exposure only.

Call-count policy:

| LLM call | Response path? | Phase 2 call-count change | Workload policy |
|---|---:|---:|---|
| Relevance Agent | yes | none | Unchanged; receives no progress state. |
| Decontextualizer | yes | none | Unchanged; still uses recent history for pronoun/reply resolution. |
| RAG2 planner/agents | yes | none | Unchanged. |
| L1 Subconscious | yes | none | Unchanged; current input only. |
| L2 Consciousness | yes | none | Unchanged; no raw history or progress payload. |
| L2 Boundary/Judgment | yes | none | Unchanged; no raw history or progress payload. |
| L3 Contextual | yes | none | Reduce dynamic raw-history payload to at most 4 messages. |
| L3 Style | yes | none | Reduce dynamic raw-history payload to at most 2 messages. |
| L3 Content Anchor | yes | none | Add Phase 2 `conversation_progress`; hard cap prompt-facing progress payload to 5000 chars. |
| L3 Preference Adapter | yes | none | Unchanged unless tests show a direct dependency. |
| L3 Visual | yes | none | Unchanged. |
| Dialog Generator | yes | none | Reduce `tone_history` to at most 2 messages. |
| Dialog Evaluator | yes, retry loop | none | Keep only `last_user_message`; no raw-history list. |
| Conversation Progress Recorder | no, background | none | Emits richer state after response; cap input/output and do not block `/chat`. |

Approximate current static system prompt sizes:

| Prompt | Current system prompt chars | Phase 2 expected static change |
|---|---:|---|
| L3 Contextual | ~1248 | small wording update only |
| L3 Style | ~2881 | small wording update only |
| L3 Content Anchor | ~5566 | moderate wording update for flow fields |
| Dialog Generator | ~3581 | small wording update for smaller tone buffer |
| Dialog Evaluator | ~3409 | small or no wording update |
| Recorder | ~1465 | moderate schema/prompt update |

Approximate dynamic payload changes:

| Layer | Current dynamic history/progress budget | Phase 2 target |
|---|---:|---:|
| L3 Contextual raw history | up to ~10 messages | up to 4 messages |
| L3 Style raw history | up to ~10 messages | up to 2 messages |
| Dialog Generator `tone_history` | up to ~10 messages | up to 2 messages |
| Content Anchor `conversation_progress` | observed max ~1621 chars in Phase 1 traces | hard cap 5000 chars |
| Recorder input/output | current Phase 1 state plus recent trace | background only; cap projected/stored fields and keep under recorder payload budget |

Expected net effect:

- Response-path call count stays flat.
- Contextual, Style, and Dialog Generator dynamic payloads shrink.
- Content Anchor payload grows, but in a structured and capped way.
- Recorder workload grows modestly, but it remains off the response path.

Workload acceptance gate:

- Add a test or instrumentation helper that records approximate char counts for every affected LLM payload.
- Phase 2 sign-off must report before/after max chars for Contextual, Style, Content Anchor, Dialog Generator, Dialog Evaluator, and Recorder.
- If any response-path LLM payload grows by more than 5000 chars versus Phase 1, stop and report before continuing.
- If any new response-path LLM call is introduced, the plan is violated.

### Ownership Boundary

The module owns:

- flow-state schema,
- recorder output validation,
- TTL and cache behavior,
- first-seen preservation,
- prompt-facing projection,
- size caps.

Existing code owns:

- relevance decision,
- cognition stage sequencing,
- Content Anchor interpretation of flow fields,
- Dialog Agent wording,
- long-term memory and user profile persistence.

## Target Lifecycle

```text
User message
  -> Relevance Agent
       decides binary should_respond

  -> load_conversation_episode_state
       loads Phase 1 + Phase 2 episode state
       applies cache/DB freshness policy
       projects bounded conversation_progress

  -> Persona Supervisor / Cognition
       Content Anchor reads flow fields, not raw history
       chooses a next conversational move:
         continue | deepen | clarify | summarize | resolve | cool_down
       Contextual/Style layers keep only a small surface/tone buffer

  -> Dialog Agent
       writes Kazusa's actual response

  -> Background recorder
       observes prior state + user input + final dialog
       updates short-term flow state

  -> Cache + DB
       store bounded operational state with TTL
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Module placement | Extend `conversation_progress` | Same lifecycle and ownership as Phase 1; avoids parallel state. |
| Reply generation | Not owned by module | Keeps Kazusa's voice in Dialog Agent and response skeleton in Content Anchor. |
| Flow fields | Use compact semantic descriptors | Better for local LLM than raw counts or transcript fragments. |
| Episode phase | LLM-emitted enum | Gives useful high-level flow without deterministic state machine logic. |
| Size limits | Enforce before prompt projection | Protects 50k context budget and local LLM reliability. |
| Raw history | Demote to surface/tone buffer | Conversation progress should own semantic continuity; raw transcript windows should not be the hidden memory system. |
| Compatibility | Optional new fields | Existing Phase 1 documents remain readable. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/conversation_progress/models.py`
  - Add optional Phase 2 fields to typed contracts.
- `src/kazusa_ai_chatbot/conversation_progress/policy.py`
  - Add field-count and character-count caps.
- `src/kazusa_ai_chatbot/conversation_progress/repository.py`
  - Preserve/cap new stored fields and tolerate old documents.
- `src/kazusa_ai_chatbot/conversation_progress/projection.py`
  - Project new fields with `age_hint` where applicable and enforce total prompt budget.
- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - Update recorder prompt and validator for flow fields.
- `src/kazusa_ai_chatbot/db/schemas.py`
  - Add optional schema fields.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Teach Content Anchor Agent how to use flow fields without generating dialog text.
  - Reduce Contextual/Style raw history inputs according to `Raw History Exposure Policy`.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Reduce generator tone history to the approved local surface/tone buffer.
  - Keep evaluator input to `last_user_message` plus directives.
- `tests/test_conversation_progress_flow_live_llm.py`
  - Add focused before/after flow fixture for the thesis contribution case.
- `tests/test_conversation_progress_history_policy.py`
  - Add patched LLM tests for raw-history exposure and workload summary artifact.
- `tests/llm_trace.py`
  - Preserve existing live trace artifacts by writing timestamp-suffixed files when a stable trace filename already exists.

### Create

- `tests/test_conversation_progress_flow.py`
  - Deterministic projection, caps, old-doc compatibility, and sharp-transition tests.

### Keep

- `src/kazusa_ai_chatbot/service.py`
  - Keep lifecycle unchanged unless a type import/state shape update is needed.
- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  - Must remain untouched by Phase 2.

## Implementation Order

- [x] Checkpoint 1 — approve the Phase 2 payload contract extension.
  - Covers: `Phase 2 Payload Contract Extension`.
  - Verify: user explicitly accepts or edits the added payload fields, size budget, and raw-history exposure policy.
  - Evidence: accepted in-session on 2026-04-28 when user approved proceeding with the proposed payload direction and requested full implementation.
  - Next: Checkpoint 2.

- [x] Checkpoint 2 — add contracts and caps.
  - Covers: `models.py`, `policy.py`, `db/schemas.py`.
  - Verify: `python -m py_compile src/kazusa_ai_chatbot/conversation_progress/*.py src/kazusa_ai_chatbot/db/schemas.py`.
  - Evidence: `py_compile` passed for modified source and test files; Phase 2 schema fields and cap constants added.
  - Next: Checkpoint 3.

- [x] Checkpoint 3 — update repository and projection.
  - Covers: old-doc compatibility, capped storage, prompt budget enforcement, sharp-transition empty projection.
  - Verify: `pytest tests/test_conversation_progress_flow.py -q`.
  - Evidence: `tests/test_conversation_progress_flow.py` passed; max projected payload is hard-capped to `<= 5000` chars.
  - Next: Checkpoint 4.

- [x] Checkpoint 4 — update recorder prompt and validator.
  - Covers: LLM-first flow-field extraction, schema validation, no code-side semantic inference.
  - Verify: recorder render test and focused validator tests.
  - Evidence: recorder prompt and validator tests passed in `tests/test_conversation_progress_flow.py`.
  - Next: Checkpoint 5.

- [x] Checkpoint 5 — integrate flow guidance into Content Anchor.
  - Covers: prompt input shape and instructions for `conversation_mode`, `episode_phase`, `topic_momentum`, `current_thread`, `current_blocker`, `next_affordances`, and `avoid_reopening`.
  - Verify: cognition prompt tests prove fields are present and Content Anchor can emit a flow-appropriate anchor.
  - Evidence: `tests/test_conversation_progress_cognition.py` passed in the broader deterministic run; prompt test confirms Phase 2 fields reach Content Anchor.
  - Next: Checkpoint 6.

- [x] Checkpoint 6 — reduce raw-history exposure by layer.
  - Covers: L3 Contextual, L3 Style, Dialog Generator, Dialog Evaluator.
  - Verify: focused tests prove Content Anchor receives no raw history, Contextual/Style/Dialog receive only their approved local buffers, and evaluator receives only `last_user_message`.
  - Evidence: `tests/test_conversation_progress_history_policy.py` passed and wrote `test_artifacts/conversation_progress_phase2_workload_summary.json`.
  - Next: Checkpoint 7.

- [x] Checkpoint 7 — add flow A/B live tests.
  - Covers: multi-turn cases where success means better continuation, not only fewer repeats.
  - Verify: run each live test one by one and save `before_phase2` / `after_phase2` traces without overwriting Phase 1 artifacts.
  - Evidence: focused before/after thesis contribution traces were saved without overwriting Phase 1 artifacts. Agent review: before trace asked the user for more text; after trace used Phase 2 fields to propose an analysis-framework/method-path third contribution and distinguished it from practical meaning/sample supplement.
  - Next: Checkpoint 8.

- [x] Checkpoint 8 — final sign-off.
  - Covers: static greps, deterministic tests, live flow tests, payload budget check.
  - Verify: every command in `Verification`.
  - Evidence: static greps, deterministic tests, payload budget checks, prompt-contract live tests, and expanded live flow matrix are recorded in `Execution Evidence`.

## Verification

### Static Greps

- `rg "conversation_progress|conversation_episode_state" src/kazusa_ai_chatbot/nodes/relevance_agent.py` returns no matches.
- `rg "conversation_progress\\.(repository|cache|projection|recorder)" src/kazusa_ai_chatbot --glob "!src/kazusa_ai_chatbot/conversation_progress/**"` returns no matches.
- `rg "if .*user_input|if .*decontexualized_input|in .*user_input|in .*decontexualized_input" src/kazusa_ai_chatbot` shows no new deterministic semantic gates.

### Tests

- `pytest tests/test_conversation_progress_flow.py -q`
- `pytest tests/test_conversation_progress_history_policy.py -q`
- `pytest tests/test_conversation_progress_runtime.py tests/test_conversation_progress_cognition.py tests/test_conversation_progress_module_boundary.py -q`
- `pytest tests/test_service_background_consolidation.py -q`
- `pytest tests/test_cognition_live_llm_prompt_contracts.py -q -m live_llm` when live LLM is available.

### Live LLM Sign-Off

Run one by one:

```powershell
$env:CONVERSATION_PROGRESS_FLOW_TRACE_PHASE='before_change'
pytest tests/test_conversation_progress_flow_live_llm.py::test_live_flow_baseline_thesis_contribution_case -q -s -m live_llm

$env:CONVERSATION_PROGRESS_FLOW_TRACE_PHASE='after_change'
pytest tests/test_conversation_progress_flow_live_llm.py::test_live_flow_baseline_thesis_contribution_case -q -s -m live_llm
```

Implemented focused flow case:

- long thesis contribution flow: should move from stuck contribution point to concrete logic next step.

Expanded release-candidate flow cases run during final sign-off:

- emotional cool-down flow: should recognize when the user is moving from distress to relief and should not keep problem-solving.
- practical debugging flow: should track what was already tried and suggest the next likely diagnostic move.
- playful social flow: should preserve social warmth without becoming overly corrective.
- rapid topic-pivot flow: should recognize a quick pivot and not drag stale obligations across the turn.
- teasing/meta-bot flow: should classify playful or third-person bot discussion as social/meta mode rather than forcing task support.
- group reply-chain flow: should handle reply-context-driven turns without assuming the entire group burst is the user's episode.

### Payload Budget

- Add a test that constructs a maximum-size Phase 2 state and proves projected `conversation_progress` is `<= 5000` characters.
- Add a live-trace summary that records observed max projected payload size.
- Add an affected-LLM workload summary that records before/after approximate prompt chars and confirms no new response-path LLM calls.
- Add a payload-shape test that records max raw-history messages per layer:
  - Content Anchor: `0`
  - Contextual Agent: `<= 4` messages, representing at most 2 current-user/bot turns
  - Style Agent: `<= 2` messages
  - Dialog Generator tone buffer: `<= 2` messages
  - Dialog Evaluator: no raw history list, only `last_user_message`

## Acceptance Criteria

This Phase 2 plan is complete when:

- The user-approved payload contract extension is implemented without new public entrypoints or public internals.
- Old Phase 1 episode documents remain readable.
- `conversation_progress` remains invisible to relevance.
- Prompt-facing progress is hard-capped to `<= 5000` characters.
- No new response-path LLM calls are introduced.
- Affected response-path LLM payload sizes are recorded, and no response-path payload grows by more than 5000 chars versus Phase 1.
- `conversation_progress` is the canonical semantic short-term memory.
- Raw history is no longer used by Content Anchor and is reduced to approved surface/tone buffers in Contextual, Style, and Dialog layers.
- Content Anchor can use `conversation_mode`, `topic_momentum`, `current_thread`, `current_blocker`, and `next_affordances` to produce better conversation flow.
- Casual chat, playful teasing, bot meta-discussion, rapid topic pivots, and goal-driven support are all representable without inventing a task goal.
- Sharp transitions suppress stale flow fields.
- No new deterministic semantic keyword gates are introduced.
- Live A/B traces show improved continuation quality in flow-focused scenarios without regressing Phase 1 anti-repeat behavior.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Flow fields make Kazusa sound scripted | Use affordances, not reply text; Dialog Agent still owns wording | Live flow review |
| Context bloat | Per-field and total projection caps | Payload budget test |
| Work shifts too much load onto Content Anchor | Hard cap progress payload and report affected-LLM payload sizes | Workload budget test |
| Phase 2 accidentally adds latency | Ban new response-path LLM calls | Call-count audit |
| Removing too much raw history makes Kazusa less natural | Keep tiny surface/tone buffers for style and dialog | History-policy tests and live flow review |
| Raw history silently remains the main continuity source | Layer-by-layer payload policy and tests | `test_conversation_progress_history_policy.py` |
| Stale guidance after topic switch | Sharp transition projects empty old obligations | Sharp-transition test |
| Recorder invents over-specific goals | Prompt requires compact operational labels, not diary prose | Recorder tests and trace review |
| Code starts interpreting user semantics | Ban keyword gates and run static grep | Static grep |

## Execution Evidence

- Payload contract approval: accepted in-session on 2026-04-28; no new public entrypoint was introduced. Existing production callers continue through `load_progress_context(...)` and `record_turn_progress(...)`.
- Static grep results: all returned no matches:
  - `rg "conversation_progress|conversation_episode_state" src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  - `rg "conversation_progress\\.(repository|cache|projection|recorder)" src/kazusa_ai_chatbot --glob "!src/kazusa_ai_chatbot/conversation_progress/**"`
  - `rg "if .*user_input|if .*decontexualized_input|in .*user_input|in .*decontexualized_input" src/kazusa_ai_chatbot`
- Payload budget result: `tests/test_conversation_progress_flow.py::test_projected_phase2_payload_is_hard_capped` passed and proves projected `conversation_progress` is `<= 5000` chars.
- Affected LLM workload result: `test_artifacts/conversation_progress_phase2_workload_summary.json` records a 50k context cap, no new response-path LLM calls, Contextual raw history reduced from 8 to 4 messages in the fixture, Style from 8 to 2, Dialog Generator tone history from 8 to 2, Content Anchor raw history at 0, Dialog Evaluator raw history at 0, and recorder response-path calls at 0.
- Raw-history exposure result: `tests/test_conversation_progress_history_policy.py` passed; Content Anchor continues to use `conversation_progress` rather than raw history, Contextual/Style/Dialog Generator receive only approved surface/tone buffers, and Dialog Evaluator receives only `last_user_message`.
- Deterministic test results: `venv\Scripts\python.exe -m pytest tests\test_conversation_progress_flow.py tests\test_conversation_progress_history_policy.py tests\test_conversation_progress_runtime.py tests\test_conversation_progress_cognition.py tests\test_conversation_progress_module_boundary.py tests\test_service_background_consolidation.py -q` passed with `24 passed`.
- Syntax check: `venv\Scripts\python.exe -m py_compile tests\test_conversation_progress_flow_live_llm.py tests\llm_trace.py` passed.
- Live flow A/B results: before trace saved at `test_artifacts/llm_traces/conversation_progress_flow_live_before_change__thesis_contribution_flow.json`; after trace saved at `test_artifacts/llm_traces/conversation_progress_flow_live_after_change__thesis_contribution_flow.json`. Future reruns preserve existing trace files by appending a timestamp suffix. Agent judgement: before trace is weak because it asks for more text instead of using episode facts; after trace is acceptable because it gives an analysis-framework/method-path third contribution and distinguishes it from practical meaning/sample supplement. The after trace passed the live LLM quality gate and deterministic continuity metrics.
- Expanded live flow matrix: each case was run one by one with `CONVERSATION_PROGRESS_FLOW_TRACE_PHASE=after_change`, inspected, and judged acceptable. The seven passing cases are thesis contribution, emotional cool-down, practical debugging, playful social, rapid topic pivot, teasing/meta-bot, and group reply-chain flow. After-change traces are stored under `test_artifacts/llm_traces/conversation_progress_flow_live_after_change*.json`.
- Prompt-contract live tests: all 13 `tests/test_cognition_live_llm_prompt_contracts.py` live cases were run one by one and inspected. Decontextualizer, Content Anchor, cognition stack, and dialog cases passed. Agent judgement: the final dialog cases stayed on-topic, kept the intended tone, and avoided stale/repetitive wording in the casual Chinese and boundary-command cases.
- Service smoke: service background consolidation tests passed inside the 24-test batch. The run connected to MongoDB and logged one existing listen-only warning: `Background consolidation skipped: unexpected consolidation_state type=NoneType`.
- Final sign-off result: complete. The implementation satisfies the Phase 2 acceptance criteria under deterministic checks, static architecture checks, payload-budget checks, and inspected live LLM behavior.

## Glossary

- `episode_phase`: LLM-emitted high-level phase of the current short-term episode.
- `conversation_mode`: LLM-emitted broad mode such as task support, emotional support, casual chat, playful banter, meta-discussion, group ambient, or mixed.
- `topic_momentum`: LLM-emitted descriptor for whether the current thread is stable, drifting, rapidly pivoting, fragmented, or sharply broken.
- `current_thread`: Neutral description of what is currently being discussed; works even when there is no task-like user goal.
- `user_goal`: Compact statement of what the user appears to be trying to accomplish in the current episode.
- `current_blocker`: The most immediate unresolved obstacle.
- `next_affordances`: A short list of natural next conversational moves available to Kazusa.
- `resolved_threads`: Threads that were handled enough and should not be reopened unless the user brings them back.
- `avoid_reopening`: Prior items that would feel repetitive or stale if brought up again.
- Surface/tone buffer: Tiny raw-history window used only for exact recent phrasing, cadence, and local adjacency, not semantic episode memory.
