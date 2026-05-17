# Conversation Progress

`conversation_progress` is a short-term conversation-flow memory module.

It gives the response pipeline a compact view of the current conversation episode: what is still active, what has already been handled, what kind of interaction is happening, and what next conversational moves would feel natural. Its purpose is to help Kazusa continue a conversation like a person without relying on a large raw transcript in every cognition prompt.

The module owns short-term episode state for response planning. Long-term user
memory, profile storage, dialogue management, and final reply generation remain
separate subsystem responsibilities.

## Design Intent

The chatbot already has durable memory and retrieval systems for stable facts. Conversation progress fills a different gap: short-lived working memory for the current local episode.

It exists because raw recent history is a poor primary continuity mechanism. Raw history can preserve exact wording and tone, but it forces every downstream LLM to rediscover the conversation state from scratch. This module instead converts recent interaction into bounded operational state:

- whether the user is continuing the same episode, shifting relatedly, or starting a new topic,
- what thread is currently active,
- whether the exchange is task support, emotional support, casual chat, playful teasing, meta discussion, group ambient chatter, or a mix,
- which user states or open loops are still unresolved,
- which assistant response moves have already been used too much,
- which threads are resolved or ready to stay closed,
- what next conversational moves are available without writing the next line for Kazusa.

The target behavior is better flow, not stronger control. The module should make responses less repetitive and more locally aware while preserving the existing cognition/dialog responsibility split.

## Integration Contract

The module exposes a small public boundary:

```python
scope = ConversationProgressScope(
    platform=platform,
    platform_channel_id=platform_channel_id,
    global_user_id=global_user_id,
)

load_result = await load_progress_context(
    scope=scope,
    current_timestamp_utc=current_timestamp_utc,
)

record_result = await record_turn_progress(
    record_input=record_input,
)
```

`load_progress_context(...)` is called before cognition on turns where the bot will respond. It returns:

- the selected prior episode state for the background recorder,
- the prompt-facing `conversation_progress` payload for cognition,
- source telemetry indicating whether the selected state came from durable storage, local cache, or an empty default.

`record_turn_progress(...)` is called after the final assistant response exists. It updates the short-term episode state and returns telemetry about the write. In latency-sensitive runtimes, callers should run it as background work and consume the result only for logs or monitoring.

Production integration should depend on the facade and typed payloads, not on storage, cache, recorder, or projection internals.

## Lifecycle

```text
User message
  -> response eligibility / relevance gate
       | no response
       |   -> stop; do not load or record progress
       |
       | response allowed
       v
  -> load conversation progress
       produces compact prompt-facing state
       v
  -> cognition / response planning
       uses progress as semantic short-term memory
       v
  -> dialog generation
       writes the actual assistant response
       v
  -> response returned to caller
       v
  -> background progress recording
       updates short-term episode state for the next turn
```

The relevance gate is intentionally outside the module. Conversation progress
helps shape a response after the system has already decided a response should
happen.

## Prompt-Facing State

The prompt-facing payload is a compact projection of the episode. It is designed for cognition, not for storage analytics.

Conceptually, it contains:

- episode status and continuity,
- a short episode label,
- broad conversation mode,
- local episode phase,
- topic momentum,
- current thread,
- optional user goal and blocker,
- unresolved user-state updates with relative age hints,
- prior assistant move labels and overused move labels,
- open loops,
- resolved or stale threads,
- emotional trajectory,
- suggested next affordances,
- short progression guidance.

The payload must remain bounded. It should be small enough to fit comfortably in the response-path context budget and clear enough for a local/weaker LLM to use directly.

On a sharp transition, stale obligations are suppressed. The next cognition step should see a new-episode style payload rather than being nudged to reopen an old thread.

## Storage Model

The stored state is short-lived operational memory keyed by platform, channel, and user. It is scoped this way to avoid leaking progress across different conversations.

The stored document carries enough information to reconstruct the next prompt-facing projection:

- continuity and status,
- episode flow descriptors,
- unresolved and resolved thread lists,
- assistant move summaries,
- first-seen timestamps for age hints,
- a monotonic turn count,
- expiry metadata.

A storage implementation should preserve this conceptual shape:

```python
{
    # Scope
    "episode_state_id": str,
    "platform": str,
    "platform_channel_id": str,
    "global_user_id": str,

    # Episode identity and lifecycle
    "status": str,
    "episode_label": str,
    "continuity": str,
    "turn_count": int,

    # Flow descriptors
    "conversation_mode": str,
    "episode_phase": str,
    "topic_momentum": str,
    "current_thread": str,
    "user_goal": str,
    "current_blocker": str,
    "emotional_trajectory": str,
    "progression_guidance": str,

    # Bounded episode entries
    "user_state_updates": [{"text": str, "first_seen_at": str}],
    "open_loops": [{"text": str, "first_seen_at": str}],
    "resolved_threads": [{"text": str, "first_seen_at": str}],
    "avoid_reopening": [{"text": str, "first_seen_at": str}],

    # Assistant move tracking
    "assistant_moves": [str],
    "overused_moves": [str],
    "next_affordances": [str],

    # Operational metadata
    "last_user_input": str,
    "created_at": str,
    "updated_at": str,
    "expires_at": str,
}
```

The important contract is behavioral rather than database-specific:

- scope fields identify exactly one user's episode within one conversation surface,
- entry lists preserve `first_seen_at` so prompt projection can produce relative age hints,
- `turn_count` is monotonic so guarded writes preserve newer progress,
- expiry metadata ensures this remains short-term working memory,
- new optional fields should be tolerated so older state can still project safely.

`conversation_mode` is intentionally a bounded semantic descriptor, not a
closed enum. The response-path consumers read the prompt-facing progress payload
as LLM context and do not branch deterministically on mode values. Deterministic
validation therefore enforces string shape and the existing 80-character stored
label cap for this field; closed-set validation remains reserved for fields
that code actually uses as stable state, such as `status` and `continuity`.

The state is expected to expire naturally as short-term working memory.

Implementations may use any persistence backend that preserves the same behavior: scoped load, guarded newer-turn write, expiry, and a way to tolerate old or partially populated documents.

## Cognition Responsibilities

Conversation progress is semantic short-term memory for response planning.

The cognition layer should use it to decide what the response should accomplish:

- acknowledge ongoing state without treating it as new,
- avoid repeating an overused assistant move,
- continue or deepen an active thread,
- resolve or cool down when the episode is closing,
- avoid reopening handled material,
- respect quick pivots and fragmented group-chat flow.

`next_affordances` and `progression_guidance` describe possible moves for
cognition. Dialog generation owns finished reply text.

Raw recent history still has a small surface/tone role: exact recent phrasing,
local cadence, and immediate adjacency. Conversation progress owns semantic
episode reconstruction.

## Dialog Responsibilities

The dialog layer owns final wording, character voice, rhythm, and surface expression.

Conversation progress influences the intended response move. Dialog generation
owns Kazusa's final lines, which keeps the module reusable across host
chatbots with different voice layers.

## Semantic Ownership

LLM judgment owns conversation semantics:

- continuity,
- current thread,
- interaction mode,
- episode phase,
- unresolved versus resolved threads,
- overused response moves,
- natural next affordances.

Deterministic code owns structure:

- validating payload shape,
- limiting string and list sizes,
- generating relative age hints from timestamps,
- enforcing prompt budget,
- expiring stale state,
- preventing stale writes from replacing newer state,
- selecting a fresher local cache entry when persistence lags.

Semantic judgment belongs in the recorder prompt and schema contract.
Deterministic code keeps the payload structural and bounded.

## Reusable Contract

Reusable integrations preserve the same capability split: response eligibility
is decided outside the module, prompt-facing progress informs response
planning, host cognition/dialog generates the response, and the completed turn
updates short-term progress for the next interaction.

## Stable Capabilities

- Keep the prompt-facing state compact and capped.
- Treat missing or expired progress as a normal empty state.
- Keep old stored states readable when adding new optional fields.
- Keep relevance independent from progress.
- Keep long-term memory systems separate from episode progress.
- Keep deterministic code structural, not semantic.
- Keep the module reusable by exposing load/record contracts instead of implementation internals.
