# conversation progress state plan

## Summary

- Goal: Add a short-lived, DB-backed conversation progress state so Kazusa can avoid repetitive response moves while preserving her focused, personality-first cognition architecture.
- Plan class: large
- Status: implementation in progress — checkpoints 1-10 implemented; final verification/live A/B sign-off pending.
- Overall cutover strategy: compatible
- Highest-risk areas: adding a new state path without bloating cognition context; keeping episode state out of relevance/response-eligibility decisions; preventing deterministic keyword interpretation of user/bot text; avoiding leakage across sharp topic transitions; keeping background persistence reliable; surviving the concurrency window between consecutive user turns.
- Acceptance criteria: repeated assistant speech acts are tracked as compact operational state, sharp topic transitions suspend stale episode obligations, cognition/content anchors receive only concise progress guidance, prior user disclosures carry relative-age signals so the bot does not re-ask them as new, existing long-term profile memory and existing Style-Agent anti-repetition behavior remain unchanged.

## Context

Kazusa intentionally does not feed full conversation history into every cognition layer. This keeps the character focused and prevents long-context drift, but it also removes the natural anti-repetition effect that simpler chatbots get from full transcript context.

The observed failure in `test_artifacts/qq_673225019_recent_4h_chat_history.json` is not a medical-advice failure. The concrete issues are:

1. Several distinct turns converged into the same response move (emotional presence / companionship promise). The live trace showed the content anchors for the final turn became:

```json
[
  "[DECISION] 接受并认可对方的依赖/给予陪伴承诺",
  "[SOCIAL] 表现出温柔且可靠的姿态，缓解对方的情绪压力",
  "[SCOPE] ~30字，[DECISION]与[SOCIAL]内容到位即可"
]
```

Once the content anchors collapse into "give companionship again," the dialog generator is behaving correctly when it repeats "I will stay with you."

2. After a 3-hour gap inside the same episode, the bot re-asked about a symptom the user had already disclosed earlier, treating prior disclosures as if they were new. The bot has access to chat history but has no compact, time-aware view of *which user states are still standing* and *how long ago they were first disclosed*.

The missing layer is not more raw chat history, but a compact operational memory of conversation progress:

- what episode is active,
- what user state is unresolved and when it was first disclosed,
- what assistant moves were already used,
- which moves are currently overused,
- whether the current turn is the same episode or a sharp topic transition.

This is short-term working memory, not durable user identity memory.

## Mandatory Rules

- Preserve the current high-level pipeline: service state build -> relevance -> decontextualizer -> RAG -> cognition -> dialog -> background consolidation.
- Load `conversation_episode_state` only after the relevance agent has decided the bot should respond. Relevance must not receive, read, log, or branch on episode state.
- If relevance decides `should_respond = false`, do not load episode state, do not run cognition/dialog, and do not run the episode recorder.
- This plan assumes `relevance_agent` produces a binary `should_respond`. Any future relevance state with intermediate response modes, such as "respond minimally," must explicitly revisit the episode-state load and record gates.
- Do not feed full raw conversation history into cognition to solve repetition.
- Do not add health-specific behavior, medical advice policy, or domain-specific sickness rules in this plan.
- Do not add deterministic semantic processing over user input or bot output. This includes regex matching, keyword classification, string-equality checks across labels, or canonicalization of LLM-emitted natural-language strings. Code may only handle metadata: timestamps, counts of identical struct keys returned by the LLM in a single call, list-length caps, TTLs, and persistence.
- LLMs own semantic interpretation of current user input, prior bot output, and final dialog. Deterministic code may validate schema shape, cap list lengths, manage TTLs, attach/preserve timestamps, and persist returned labels.
- Keep episode state operational and behavior-facing. Do not store romance-colored subjective reflections in this collection.
- Do not modify the existing Style Agent anti-repetition rule (`persona_supervisor2_cognition_l3.py` lines under `_STYLE_AGENT_PROMPT`'s "轻量反重复"). It remains as the synchronous local-window safety net during this rollout. A separate later plan may revisit it once the new layer has soak time.
- Do not modify the existing Dialog Evaluator anti-pollution / topic-drift rules. New `[AVOID_REPEAT]` / `[PROGRESSION]` awareness rides on top of the existing evaluator-feedback retry channel; it does not replace any current rule.
- Prompt changes must keep user-derived content in HumanMessage payloads, not SystemMessage instructions.
- Python edits must follow project style: imports at top, narrow `try` blocks, specific exception types except at process boundaries, docstrings for non-trivial functions, no scattered internal defaults, and no broad defensive exception handling for internal bugs.
- Any prompt containing literal JSON examples must be checked for `.format(...)` brace safety and runtime renderability.

## Must Do

- Create a new short-term MongoDB collection named `conversation_episode_state`.
- Add a DB module that loads and upserts per-user/channel episode state by `(platform, platform_channel_id, global_user_id)`.
- Add a compact episode-state schema to project state types.
- Persist a `turn_count` integer on each document; bump it monotonically on each successful recorder write so consumers can detect stale reads.
- Maintain an in-process "last completed turn" cache keyed by `(platform, platform_channel_id, global_user_id)` that the next-turn loader consults as a fallback when the DB document's `turn_count` is behind the in-memory value.
- Track per-entry first-seen timestamps for `user_state_updates` and `open_loops`. Code preserves the original `first_seen_at` when the recorder re-emits an existing entry, and stamps a new one when an entry first appears. The recorder LLM judges semantic equivalence; code only matches via the LLM's own returned key field.
- Add a post-relevance service graph node that loads episode state only on the response path, after `relevance_agent` returns `should_respond = true` and before `persona_supervisor2`.
- Pass episode state through `IMProcessState`, `GlobalPersonaState`, and `CognitionState`.
- Use episode state in the Content Anchor Agent to influence "what to say," especially repeated assistant moves, unresolved user state with relative-age annotations, and progression guidance.
- Hook `[AVOID_REPEAT]` / `[PROGRESSION]` awareness into the **existing** Dialog Evaluator retry-feedback channel (`HumanMessage(name="evaluator")`); do not introduce a parallel channel.
- Add an LLM-first background recorder that updates episode state after final dialog is generated.
- Add sharp topic transition handling in the recorder output contract.
- Add TTL or explicit expiry so stale episode state naturally disappears.
- Add tests for state load/store, sharp transition behavior, content-anchor guidance, relative-age preservation across turns, concurrency-cache fallback, and no long-term memory pollution.

## Deferred

- Do not add medical or care-world-knowledge retrieval.
- Do not redesign RAG2 routing or helper-agent slots.
- Do not change affinity, relationship scoring, or user image synthesis in this plan.
- Do not alter relevance-agent response eligibility rules. Do not pass episode state through relevance, even for logging.
- Do not replace existing `chat_history_recent` or `chat_history_wide`.
- Do not implement a full shadow prediction branch or empathic accuracy evaluator.
- Do not add a generic dialogue-manager framework or state machine.
- Do not migrate historical conversation data into episode state.
- Do not remove or rewrite the Style Agent's existing local-window phrase-level anti-repetition rule. Phrase-level coverage stays where it is.
- Do not subsume `forbidden_phrases` into the new mechanism. `[AVOID_REPEAT]` operates at the move/speech-act layer; `forbidden_phrases` continues to operate at the lexical layer. Both can fire on the same turn.
- Do not store the literal text of prior assistant turns in the episode-state document. Chat history already preserves it; duplication would drift.
- Do not introduce offline label canonicalization or analytics tooling in this plan. Move-label drift is a runtime non-issue under LLM-first interpretation; if analytics is later wanted, that is a separate offline LLM job.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| New MongoDB collection | compatible | Add `conversation_episode_state` without modifying existing collections or migrating historical data. |
| Runtime state payload | compatible | Add optional state keys. Existing consumers must continue working if the key is missing or empty. |
| Relevance gate | unchanged | Episode state is loaded only after relevance chooses to respond. Relevance behavior, prompt inputs, logs, and branching must remain independent of episode state. |
| Content Anchor Agent | compatible | Extend prompt input with compact `conversation_progress`; do not remove existing input fields. |
| Style Agent | unchanged | Existing local-window anti-repetition rule and `forbidden_phrases` derivation remain exactly as written. No prompt edits in this plan. |
| Dialog Evaluator | compatible | Extend evaluator awareness of `[AVOID_REPEAT]` / `[PROGRESSION]` anchors **through the existing feedback HumanMessage channel** with `name="evaluator"`. Do not introduce a new retry mechanism. |
| Background consolidation | compatible | Add episode recorder as an additional background step; do not block `/chat` response on it. |
| Long-term user profile memory | compatible | Leave `user_profiles`, `user_profile_memories`, and `memory` behavior unchanged. |

## Cutover Policy Enforcement

- The implementation agent must not choose a big-bang replacement of the cognition pipeline.
- Compatible means missing episode state must degrade to current behavior.
- Relevance must remain a hard precondition for episode-state loading and recording. If relevance ends the turn, episode state remains untouched.
- Existing tests must not be rewritten to depend on episode state unless they explicitly cover the new feature.
- The Style Agent prompt must be byte-identical before and after this plan's implementation. A mechanical diff check covers this.
- Any change from compatible cutover to migration or bigbang requires user approval.

## Agent Autonomy Boundaries

- The agent may choose local helper names and small implementation mechanics only when they preserve this plan's contracts.
- The agent must not introduce alternate architectures, fallback classifiers, health-specific patches, or broad prompt rewrites outside the listed files.
- The agent must not infer episode labels or move labels with local keyword matching, regex, or string-equality canonicalization. Semantic judgment is the LLM's job.
- The agent must not add a parallel evaluator-feedback channel; reuse the existing `HumanMessage(name="evaluator")` retry channel in `dialog_agent.py`.
- If the plan and code disagree, preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

At the start of each responsive turn, after relevance has already decided Kazusa should speak, cognition receives a compact progress payload similar to:

```json
{
  "status": "active",
  "episode_label": "ongoing_user_distress",
  "continuity": "same_episode",
  "turn_count": 6,
  "user_state_updates": [
    {"text": "user previously said they caught a cold", "age_hint": "~3h ago"},
    {"text": "user previously said they are coughing", "age_hint": "~3h ago"},
    {"text": "user previously said their throat hurts and their head feels unclear", "age_hint": "~3h ago"}
  ],
  "assistant_moves": [
    "presence_commitment",
    "symptom_reflection",
    "presence_commitment"
  ],
  "overused_moves": ["presence_commitment"],
  "open_loops": [
    {"text": "user still feels physically unwell", "age_hint": "~3h ago"}
  ],
  "progression_guidance": "Do not make presence_commitment the main move again; the user's discomfort was disclosed hours ago — either acknowledge continuity briefly or make a grounded follow-up rather than treating it as new."
}
```

`age_hint` is an LLM-facing natural-language phrase produced by code from the stored numeric `first_seen_at`. Code does not parse `age_hint`; only the model reads it.

On a sharp topic transition, the payload given to cognition must not carry stale episode obligations:

```json
{
  "status": "new_episode",
  "episode_label": "hardware_question",
  "continuity": "sharp_transition",
  "turn_count": 7,
  "user_state_updates": [],
  "assistant_moves": [],
  "overused_moves": [],
  "open_loops": [],
  "progression_guidance": ""
}
```

This state is behavior-progress memory. It is not a diary, relationship insight, medical record, or replacement for conversation history.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Storage location | New `conversation_episode_state` collection | Keeps short-term operational state separate from durable profile memory and subjective user image. |
| Key scope | `(platform, platform_channel_id, global_user_id)` | Prevents leakage across platforms, channels, and users while preserving private/group thread continuity. |
| Expiry | TTL via `expires_at`, default 48 hours | Episode state is working memory and should not become permanent identity memory. |
| Topic transition owner | LLM recorder | Sharp transition is semantic; deterministic keyword routing would violate the LLM-first architecture. |
| Move-label vocabulary | Free-form, LLM-resolved | Closed schemas break across languages and personas; runtime equivalence is judged by the recorder LLM each turn, which has access to prior labels and prior dialogs. |
| `overused_moves` computation | LLM recorder, not code | Counting requires equality, which on free-form natural-language labels would be a deterministic semantic check. The LLM owns this. |
| Per-entry timestamps | Code-managed | `first_seen_at` is metadata, not natural language. Code preserves it across turns when the LLM re-emits an existing entry. |
| Concurrency | `turn_count` token + in-process last-completed cache | Background recorder may not finish before the next user turn arrives; code checks the in-memory cache when the loaded DB document's `turn_count` is behind. |
| Style Agent integration | None | Existing local-window phrase-level rule is the synchronous safety net during the concurrency gap; leave it alone. |
| Evaluator integration | Reuse existing `HumanMessage(name="evaluator")` retry channel | A second channel would fragment the dialog-agent control flow. |
| Runtime injection | Compact `conversation_progress` object | Local LLMs need semantic labels with age hints, not raw transcripts or numeric counters. |
| Load timing | Post-relevance only | Episode state influences how Kazusa progresses once she is already speaking; it must not influence whether she speaks. |
| Primary consumer | Content Anchor Agent | Repetition must be fixed where "what to say" is chosen, before dialog renders wording. |
| Secondary consumer | Dialog Evaluator | Evaluator should catch outputs that ignore explicit progress guidance, but should not become the primary planner. |
| Update timing | Background after final dialog | Keeps normal `/chat` latency stable and lets recorder see both user input and actual assistant move. |
| Historical backfill | None | Old conversations remain searchable; episode state begins after deployment. |

## Contracts

### Conversation Progress Module Interface

Create a dedicated package:

```text
src/kazusa_ai_chatbot/conversation_progress/
```

The package is the only owner of conversation-progress loading, projection, recording, DB persistence, and last-completed cache fallback. Existing code must import only the public facade exported by `kazusa_ai_chatbot.conversation_progress`.

Required internal modules:

```text
src/kazusa_ai_chatbot/conversation_progress/__init__.py
src/kazusa_ai_chatbot/conversation_progress/models.py
src/kazusa_ai_chatbot/conversation_progress/policy.py
src/kazusa_ai_chatbot/conversation_progress/repository.py
src/kazusa_ai_chatbot/conversation_progress/cache.py
src/kazusa_ai_chatbot/conversation_progress/projection.py
src/kazusa_ai_chatbot/conversation_progress/recorder.py
src/kazusa_ai_chatbot/conversation_progress/runtime.py
```

Existing service/persona/consolidation code must not import `repository.py`, `cache.py`, `projection.py`, or `recorder.py` directly. Those are module internals.

#### Public Types

`models.py` must define:

```python
from dataclasses import dataclass
from typing import Literal, TypedDict


@dataclass(frozen=True)
class ConversationProgressScope:
    """Stable per-user/channel scope for short-term conversation progress.

    Args:
        platform: Runtime platform key, such as "qq" or "discord".
        platform_channel_id: Runtime channel/group/private-chat id.
        global_user_id: Internal user UUID for the current speaker.
    """

    platform: str
    platform_channel_id: str
    global_user_id: str


class ConversationProgressEntry(TypedDict):
    text: str
    age_hint: str


class ConversationProgressPromptDoc(TypedDict):
    status: str
    episode_label: str
    continuity: str
    turn_count: int
    user_state_updates: list[ConversationProgressEntry]
    assistant_moves: list[str]
    overused_moves: list[str]
    open_loops: list[ConversationProgressEntry]
    progression_guidance: str


class ConversationProgressLoadResult(TypedDict):
    episode_state: ConversationEpisodeStateDoc | None
    conversation_progress: ConversationProgressPromptDoc
    source: Literal["db", "cache", "empty"]


class ConversationProgressRecordInput(TypedDict):
    scope: ConversationProgressScope
    timestamp: str
    prior_episode_state: ConversationEpisodeStateDoc | None
    decontexualized_input: str
    chat_history_recent: list[dict]
    content_anchors: list[str]
    logical_stance: str
    character_intent: str
    final_dialog: list[str]


class ConversationProgressRecordResult(TypedDict):
    written: bool
    turn_count: int
    continuity: str
    status: str
    cache_updated: bool
```

`ConversationEpisodeStateDoc` is defined in `db/schemas.py` and re-exported by `conversation_progress.models` for type convenience. The module must not define a second incompatible stored-doc shape.

#### Public Facade

`__init__.py` must export exactly these facade functions and public types:

```python
async def load_progress_context(
    *,
    scope: ConversationProgressScope,
    current_timestamp: str,
) -> ConversationProgressLoadResult:
    """Load and project progress state for one responsive turn.

    Args:
        scope: Platform/channel/user scope for the current turn.
        current_timestamp: ISO-8601 timestamp used for age-hint projection and
            cache staleness checks.

    Returns:
        A load result containing the selected stored episode document, the
        compact prompt-facing `conversation_progress` projection, and the source
        used to select the document:
        - `"db"` when the MongoDB document is used.
        - `"cache"` when the process-local last-completed cache has a strictly
          higher `turn_count` than MongoDB.
        - `"empty"` when no prior episode state exists.
    """


async def record_turn_progress(
    *,
    record_input: ConversationProgressRecordInput,
) -> ConversationProgressRecordResult:
    """Record progress after a final dialog has been generated.

    Args:
        record_input: Current turn data needed by the recorder LLM and
            persistence layer. It includes the current scope, timestamp,
            prior stored episode state, decontextualized user input, recent chat
            history, content anchors, stance/intent, and final dialog.

    Returns:
        A record result describing whether a document was written, the resulting
        `turn_count`, recorder continuity/status, and whether the
        last-completed cache was updated. This result is consumed only by the
        background caller for logging/telemetry and must not be awaited by the
        synchronous `/chat` response path.
    """
```

No existing caller may use lower-level functions as its integration surface. If tests need internals, they may import internal modules directly only in focused unit tests.

Callers in the response path must spawn `record_turn_progress(...)` as a background task after final dialog is available. The background task awaits the result, logs compact telemetry such as `written`, `turn_count`, `continuity`, `status`, and `cache_updated`, and does not feed the result back into the already-returned chat response.

#### Facade Responsibilities

`load_progress_context(...)` owns:

- loading the stored document from MongoDB,
- consulting the process-local last-completed cache,
- choosing cache over DB only when cache `turn_count` is strictly higher,
- returning an explicit empty projection when no document exists,
- projecting stored state to `conversation_progress`,
- generating `age_hint`.

`record_turn_progress(...)` owns:

- calling the recorder LLM,
- validating recorder JSON shape,
- preserving `first_seen_at` for recorder-returned entry text,
- capping persisted lists,
- setting `created_at`, `updated_at`, and `expires_at`,
- computing the next `turn_count`,
- performing `turn_count`-guarded upsert,
- updating the process-local last-completed cache only after a successful write.

#### Existing-Code Integration Boundary

- `service.py` may call `load_progress_context(...)` only after `relevance_agent` returns `should_respond = true`.
- `service.py` must not perform DB load, cache fallback, projection, age-hint generation, or recorder JSON validation itself.
- The background post-dialog path may call `record_turn_progress(...)` with a `ConversationProgressRecordInput`.
- Non-responsive turns do not update either MongoDB or the process-local last-completed cache. If turn N-1 is non-responsive, then turn N loads the latest state from the last responsive recorded turn; DB and cache age out through their respective TTL/staleness rules.
- `persona_supervisor2.py` and cognition code may read `conversation_progress` from graph state, but must not call repository/cache/recorder internals.
- `relevance_agent.py` must not import `kazusa_ai_chatbot.conversation_progress` and must not receive `conversation_episode_state` or `conversation_progress`.

### Collection Shape

`conversation_episode_state` documents must use this shape:

```python
{
    "episode_state_id": str,
    "platform": str,
    "platform_channel_id": str,
    "global_user_id": str,
    "status": "active" | "suspended" | "closed",
    "episode_label": str,
    "continuity": "same_episode" | "related_shift" | "sharp_transition",
    "user_state_updates": list[{"text": str, "first_seen_at": str}],
    "assistant_moves": list[str],
    "overused_moves": list[str],
    "open_loops": list[{"text": str, "first_seen_at": str}],
    "progression_guidance": str,
    "turn_count": int,
    "last_user_input": str,
    "created_at": str,
    "updated_at": str,
    "expires_at": str,
}
```

List fields must be capped before persistence:

- `user_state_updates`: max 8
- `assistant_moves`: max 8
- `overused_moves`: max 5
- `open_loops`: max 5

`first_seen_at` rules:

- The recorder LLM emits each entry with only its `text`. The recorder is told it may copy text verbatim from a prior entry to indicate "same item."
- Code matches new entries to existing ones by exact equality on the `text` field that the LLM itself returned. This is not a deterministic semantic interpretation of natural language; it is a struct-key compare on the LLM's own output.
- If matched, code copies forward the existing `first_seen_at`. If unmatched, code stamps `first_seen_at = now`.
- This is the *only* place exact-string compare is permitted in this plan, and it is bounded to the recorder's own returned strings, not to user or bot dialog.

`turn_count` rules:

- Strictly increasing per `(platform, platform_channel_id, global_user_id)`.
- Recorder write performs a conditional update: only persist if the incoming `turn_count` is greater than the stored value.
- The in-process last-completed cache stores the highest `turn_count` observed per key.

### Concurrency Cache Shape

A process-local dict, no persistence:

```python
{
    (platform, platform_channel_id, global_user_id): {
        "turn_count": int,
        "document": <full episode-state document just written>,
        "completed_at": float,
    }
}
```

- Loader logic on next turn: read DB document. If cache has an entry with strictly higher `turn_count`, use the cached document. Otherwise use the DB document.
- Cache entries older than the TTL window (e.g., 1 hour) are evicted opportunistically; staleness is bounded by the DB TTL anyway.
- This cache is best-effort. A process restart between turns drops it; the next turn falls back to the DB document, which is the same behavior as today's no-episode-state baseline. No correctness guarantee is offered beyond best-effort.

### Runtime State Shape

Add optional `conversation_episode_state` to `IMProcessState` and `GlobalPersonaState`. Add `conversation_progress` to `CognitionState`, where `conversation_progress` is a presentation-safe subset:

```python
{
    "status": str,
    "episode_label": str,
    "continuity": str,
    "turn_count": int,
    "user_state_updates": list[{"text": str, "age_hint": str}],
    "assistant_moves": list[str],
    "overused_moves": list[str],
    "open_loops": list[{"text": str, "age_hint": str}],
    "progression_guidance": str,
}
```

`age_hint` is generated by a small code helper from `first_seen_at` and the current time, producing short natural-language phrases like `"just now"`, `"~30m ago"`, `"~3h ago"`, `"earlier today"`, `"yesterday"`. The helper uses fixed numeric thresholds; it does not interpret content.

### Recorder Output Contract

The episode recorder must output strict JSON:

```json
{
  "continuity": "same_episode | related_shift | sharp_transition",
  "status": "active | suspended | closed",
  "episode_label": "short semantic label",
  "user_state_updates": ["compact user-state observation; copy verbatim from prior turn to indicate same item"],
  "assistant_moves": ["compact assistant speech-act label"],
  "overused_moves": ["assistant move label that has occurred too often in the recent history"],
  "open_loops": ["unresolved thread; copy verbatim from prior turn to indicate same item"],
  "progression_guidance": "one short instruction for next turn"
}
```

The recorder reads prior episode state, `decontexualized_input`, `content_anchors`, `logical_stance`, `character_intent`, the relevant slice of chat history, and `final_dialog`. The recorder is the sole authority on:

- whether two prior moves are the same move (by re-emitting the same string),
- whether a user-state entry from earlier still applies (by re-emitting it verbatim),
- which moves are overused,
- whether continuity holds.

Code only caps lists, attaches/preserves `first_seen_at`, sets timestamps, and upserts under the `turn_count` guard.

### Content Anchor Contract

The Content Anchor Agent must receive `conversation_progress`. When `continuity` is `same_episode` or `related_shift`, it must use `overused_moves`, `open_loops`, `user_state_updates` (with `age_hint`), and `progression_guidance` when deciding content anchors. When `continuity` is `sharp_transition`, it must ignore stale episode obligations and focus on the new input.

Allowed new anchor labels:

- `[PROGRESSION]`: how this turn advances the episode beyond prior assistant moves.
- `[AVOID_REPEAT]`: the overused assistant move that must not be the main response move.

When `continuity != "sharp_transition"` and `overused_moves` is non-empty, the Content Anchor Agent must emit at least one `[AVOID_REPEAT]` anchor corresponding to any overused move it would otherwise lead with. If the current user turn genuinely requires acknowledging the same move, it must also emit a `[PROGRESSION]` anchor explaining how the response advances beyond that move.

Do not add domain labels such as `[MEDICAL]`.

### `[AVOID_REPEAT]` vs `forbidden_phrases` Boundary

These are different-layer mechanisms and both can fire on the same turn:

| Mechanism | Layer | Scope | Source | Horizon |
|---|---|---|---|---|
| `forbidden_phrases` | Lexical | Specific words/phrases the dialog must not produce | Style Agent reading `chat_history_recent` (synchronous) | Last ~2 rounds |
| `[AVOID_REPEAT]` | Speech-act | Semantic move labels the dialog must not lead with | Episode recorder (asynchronous, prior turn) | Whole episode (capped) |

The Style Agent and the episode-state path do not share inputs and do not coordinate. The Dialog Generator receives both `forbidden_phrases` and content anchors that may include `[AVOID_REPEAT]`, and applies them independently.

### Dialog Evaluator Integration Contract

The evaluator must reuse the existing feedback channel in `dialog_agent.py`:

- Detection: when `content_anchors` contains an `[AVOID_REPEAT]` entry, and the rendered `final_dialog`'s primary move (per evaluator's own LLM judgment, no string compare) matches the avoided move, and no `[PROGRESSION]` content is satisfied, the evaluator returns `should_stop: false`.
- Feedback shape: same `HumanMessage(name="evaluator")` JSON it already produces, with `feedback` text that includes the avoided move label so the generator's existing "Evaluator Feedback" reading at retry can pick it up.
- No new retry mechanism, no new state field, no new control edge in the LangGraph state machine. The existing loop at `dialog_agent.py` evaluator-edges handles regeneration.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/conversation_progress/`
  - Dedicated package for the new feature. Existing code imports only from `kazusa_ai_chatbot.conversation_progress`.
- `src/kazusa_ai_chatbot/conversation_progress/__init__.py`
  - Public facade exports: `load_progress_context`, `record_turn_progress`, `ConversationProgressScope`, `ConversationProgressLoadResult`, `ConversationProgressRecordInput`, `ConversationProgressRecordResult`, and `ConversationProgressPromptDoc`.
- `src/kazusa_ai_chatbot/conversation_progress/models.py`
  - Public type contracts listed in "Conversation Progress Module Interface".
- `src/kazusa_ai_chatbot/conversation_progress/policy.py`
  - Collection name, TTL, list caps, empty projection defaults, valid status/continuity constants.
- `src/kazusa_ai_chatbot/conversation_progress/repository.py`
  - Internal MongoDB load/upsert helpers, index helper, `turn_count` guard, and `first_seen_at` preservation.
- `src/kazusa_ai_chatbot/conversation_progress/cache.py`
  - Internal process-local last-completed cache keyed by `(platform, platform_channel_id, global_user_id)`.
- `src/kazusa_ai_chatbot/conversation_progress/projection.py`
  - Internal stored-doc to `conversation_progress` projection and `age_hint` generation.
- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`
  - Internal recorder LLM prompt, strict JSON parse, and recorder-output validation.
- `src/kazusa_ai_chatbot/conversation_progress/runtime.py`
  - Implements the two public facade functions and dependency-injectable runtime class for tests.
- `tests/test_conversation_episode_state.py`
  - Unit tests for repository query shape, capping, TTL fields, `turn_count` guard, `first_seen_at` preservation, and empty-state behavior.
- `tests/test_conversation_progress_cognition.py`
  - Focused tests for content-anchor prompt payload, repetition guidance, and `age_hint` projection.
- `tests/test_conversation_episode_cache.py`
  - Tests for in-process cache fallback when DB is behind.
- `tests/test_conversation_progress_runtime.py`
  - Facade-level tests for `load_progress_context(...)`, `record_turn_progress(...)`, dependency injection, and caller-facing return shapes.
- `tests/test_conversation_progress_module_boundary.py`
  - Static boundary test that production code imports only from `kazusa_ai_chatbot.conversation_progress`, not from internal `repository`, `cache`, `projection`, or `recorder` modules.

### Modify

- `src/kazusa_ai_chatbot/db/schemas.py`
  - Add `ConversationEpisodeStateDoc`.
- `src/kazusa_ai_chatbot/db/__init__.py`
  - Re-export episode-state helpers and schema.
- `src/kazusa_ai_chatbot/db/bootstrap.py`
  - Ensure collection and indexes exist, including TTL index on `expires_at` and unique compound index.
- `src/kazusa_ai_chatbot/state.py`
  - Add optional top-level state key for episode state.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Add optional state fields for persona and cognition stages.
- `src/kazusa_ai_chatbot/service.py`
  - Add a service graph node between `relevance_agent` and `persona_supervisor2` that loads episode state on the response path only.
  - The node calls `load_progress_context(...)` and writes `episode_state` plus `conversation_progress` from the result into graph state before persona/cognition.
  - The service node must not directly perform DB load, cache fallback, projection, or age-hint generation.
  - Queue background episode recorder after final dialog.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Pass episode state into persona state and consolidation state.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Project loaded episode state into compact `conversation_progress`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` — **Content Anchor Agent function only**
  - Add prompt support for `conversation_progress`, `[PROGRESSION]`, and `[AVOID_REPEAT]`.
  - **Style Agent function in this same file is not modified.** A diff check must confirm the Style Agent prompt is byte-identical.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Evaluator: add `[AVOID_REPEAT]` / `[PROGRESSION]` awareness inside the existing prompt; emit feedback through the existing `HumanMessage(name="evaluator")` channel.
  - Generator: no new feedback-reading code; existing "Evaluator Feedback" handling already covers it.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
  - If consolidation remains the background orchestration point, call `record_turn_progress(...)` as a sibling background step using `ConversationProgressRecordInput`.
  - Do not import recorder, repository, cache, or projection internals.

### Keep

- Existing RAG2 supervisor and retrieval agents.
- Existing long-term profile memory and user image persistence.
- Existing relevance behavior. Relevance must not receive episode state or log episode-state presence.
- Existing conversation history persistence.
- Existing Style Agent prompt and its local-window phrase-level anti-repetition rule.
- Existing Dialog Evaluator core rules (visual pollution, meta-dialog, topic drift, structural taboos). New move-level checks are additive.

## Implementation Order

Each top-level checkbox below is the progress checkpoint an implementation agent updates. Do not tick a checkpoint until its listed work is complete, its verification has passed or the blocker is recorded, and evidence is added under `Execution Evidence`.

- [x] Checkpoint 1 — add schema and DB helpers.
   - Implement `ConversationEpisodeStateDoc` with `user_state_updates` / `open_loops` as `list[{"text", "first_seen_at"}]` and `turn_count`.
   - Implement internal repository helpers with `turn_count`-guarded conditional update inside `conversation_progress/repository.py`.
   - Implement `first_seen_at` preservation: match incoming entries to stored entries by exact `text` equality (recorder-returned strings only), copy forward when matched, stamp `now` when new.
   - Add index creation in bootstrap:
     - unique compound index on `(platform, platform_channel_id, global_user_id)`
     - TTL index on `expires_at`
   - Sign-off verification:
     - `pytest tests/test_conversation_episode_state.py`
   - Sign-off evidence:
     - Record schema/repository/bootstrap summary and test result in `Execution Evidence`.

- [x] Checkpoint 2 — add the in-process last-completed cache.
   - Implement inside `conversation_progress/cache.py`.
   - Module-level dict keyed by `(platform, platform_channel_id, global_user_id)`.
   - Loader helper that takes the DB document and returns the cache document if its `turn_count` is strictly higher.
   - Sign-off verification:
     - `pytest tests/test_conversation_episode_cache.py`
   - Sign-off evidence:
     - Record cache fallback behavior and test result in `Execution Evidence`.

- [x] Checkpoint 3 — add module facade and runtime.
   - Implement `ConversationProgressScope`, `ConversationProgressLoadResult`, `ConversationProgressRecordInput`, and `ConversationProgressRecordResult`.
   - Implement `load_progress_context(...)` and `record_turn_progress(...)` in the public facade.
   - Add dependency-injectable runtime class or constructor wiring for focused tests.
   - Ensure `__init__.py` exports public types/functions only.
   - Sign-off verification:
     - `python -m py_compile src/kazusa_ai_chatbot/conversation_progress/*.py`
     - `pytest tests/test_conversation_progress_runtime.py`
   - Sign-off evidence:
     - Record public facade signature and runtime test result in `Execution Evidence`.

- [x] Checkpoint 4 — add post-relevance state plumbing without behavior changes.
   - Add optional fields to state types.
   - Add a `load_conversation_episode_state` node to the service graph after `relevance_agent` on the `should_respond = true` branch and before `persona_supervisor2`.
   - Call `load_progress_context(...)` in that node.
   - Pass it through persona/cognition state.
   - Verify `should_respond = false` exits without calling the episode-state loader.
   - Verify existing tests still pass with empty episode state.
   - Sign-off verification:
     - `pytest tests/test_conversation_progress_runtime.py`
     - `pytest tests/test_conversation_progress_module_boundary.py`
     - `pytest tests/test_persona_supervisor2.py`
   - Sign-off evidence:
     - Record that service calls only `load_progress_context(...)`, non-responsive turns do not load progress, and tests passed.

- [x] Checkpoint 5 — add compact projection helper.
   - Convert stored document into `conversation_progress`.
   - Generate `age_hint` from `first_seen_at` using fixed numeric thresholds.
   - Strip persistence-only fields before sending to LLM.
   - If no document exists, use an explicit empty progress object.
   - Sign-off verification:
     - `pytest tests/test_conversation_progress_runtime.py`
     - `pytest tests/test_conversation_progress_cognition.py`
   - Sign-off evidence:
     - Record projection/age-hint test result and empty-state behavior.

- [x] Checkpoint 6 — add Content Anchor Agent support.
   - Extend prompt input with `conversation_progress`.
   - Add rules for `[PROGRESSION]` and `[AVOID_REPEAT]`.
   - Add explicit instruction that `age_hint` indicates how long ago a user disclosure was made, and that already-disclosed unresolved states should not be re-asked as if new.
   - Keep examples minimal and boundary-focused.
   - Run prompt render checks.
   - Sign-off verification:
     - Render Content Anchor Agent prompt with `conversation_progress`.
     - `pytest tests/test_conversation_progress_cognition.py`
     - `pytest tests/test_cognition_live_llm_prompt_contracts.py -q` when live LLM is available.
   - Sign-off evidence:
     - Record prompt-render result and cognition test result.

- [x] Checkpoint 7 — add Dialog Evaluator backstop through the existing feedback channel.
   - Extend the existing evaluator prompt to recognize `[AVOID_REPEAT]` / `[PROGRESSION]`.
   - When violated, return `should_stop: false` with feedback text that names the avoided move so the generator's existing "Evaluator Feedback" path can act on it.
   - Do not add new graph edges, new state keys, or a parallel channel.
   - Sign-off verification:
     - Render Dialog Evaluator prompt after adding progression validation.
     - Style Agent prompt diff is byte-identical before and after.
     - `pytest tests/test_conversation_progress_cognition.py`
   - Sign-off evidence:
     - Record evaluator prompt-render result, Style Agent diff result, and test result.

- [x] Checkpoint 8 — add episode recorder.
   - Implement inside `conversation_progress/recorder.py`.
   - Create an LLM prompt that reads prior episode state, `decontexualized_input`, `content_anchors`, `logical_stance`, `character_intent`, the relevant slice of chat history, and `final_dialog`.
   - Recorder emits semantic labels and transition status; reuses prior strings verbatim to indicate "same item."
   - Persistence code caps lists, applies `first_seen_at` preservation, updates timestamps, sets expiry, increments `turn_count`, and upserts.
   - Sign-off verification:
     - Render Episode Recorder prompt.
     - `pytest tests/test_conversation_progress_runtime.py`
   - Sign-off evidence:
     - Record recorder prompt-render result and recorder/runtime test result.

- [x] Checkpoint 9 — wire recorder into background flow.
   - Queue after final dialog is available.
   - Do not block `/chat` response.
   - Call `record_turn_progress(...)`; the facade handles recorder call, guarded write, and cache update.
   - Log compact success/failure metadata only.
   - Sign-off verification:
     - `pytest tests/test_service_background_consolidation.py`
     - Concurrency fixture: turn N+1 loads while turn N recorder is pending and cache fallback supplies turn N state.
   - Sign-off evidence:
     - Record background queue path, non-blocking behavior, concurrency fixture result, and cache-update behavior.

- [x] Checkpoint 10 — add tests and live-trace fixture.
   - Use the sickness repetition log as a regression fixture for move progression.
   - Add a sharp topic transition fixture.
   - Add a fixture that simulates the 3-hour gap → verify `age_hint` projection prevents the bot from asking already-disclosed symptoms as new.
   - Add a concurrency fixture: turn N+1 loads while turn N's recorder is still pending → cache fallback supplies the latest state.
   - Add an unrelated routine conversation fixture to ensure no behavior drag.
   - Add a Style Agent prompt diff check (byte-identical pre/post).
   - Add facade-level tests for `load_progress_context(...)` and `record_turn_progress(...)`.
   - Add module-boundary static test ensuring production callers import only the public package facade.
   - Sign-off verification:
     - `pytest tests/test_conversation_episode_state.py`
     - `pytest tests/test_conversation_progress_cognition.py`
     - `pytest tests/test_conversation_episode_cache.py`
     - `pytest tests/test_conversation_progress_runtime.py`
     - `pytest tests/test_conversation_progress_module_boundary.py`
   - Sign-off evidence:
     - Record all focused test results and any trace-harness changes.

- [x] Checkpoint 11 — run verification gates and final live A/B sign-off.
   - Run all static greps, prompt render checks, unit tests, smoke tests, database checks, and final sign-off integration tests listed below.
   - Sign-off verification:
     - Every command/check in `Verification`.
     - Every live LLM test under `Final Sign-Off Integration Tests`, run one at a time.
   - Sign-off evidence:
     - Record static grep results, test command results, service smoke, MongoDB index verification, before/after trace comparison, and JSON artifact cleanliness check.
   - Completion rule:
     - This checkpoint can be ticked only when Acceptance Criteria and Final Sign-Off Integration Tests pass.

## Verification

### Static Greps

- `rg "conversation_episode_state" src tests` shows only the planned schema, DB, service, cognition, dialog, and tests.
- `rg "conversation_episode_state|conversation_progress" src/kazusa_ai_chatbot/nodes/relevance_agent.py` returns no matches.
- `rg "conversation_episode_state|conversation_progress" src/kazusa_ai_chatbot/service.py` shows only the post-relevance load node and background record task wiring.
- `rg "conversation_progress\\.(repository|cache|projection|recorder)" src/kazusa_ai_chatbot --glob "!src/kazusa_ai_chatbot/conversation_progress/**"` returns no matches.
- `rg "感冒|咳嗽|喉咙|发烧|医生|medical|doctor" src/kazusa_ai_chatbot/nodes src/kazusa_ai_chatbot/db tests` shows no new domain-specific repetition fix, except test fixture text if used.
- `rg "if .*user_input|if .*decontexualized_input|in .*user_input|in .*decontexualized_input" src/kazusa_ai_chatbot` does not reveal new deterministic semantic keyword gates introduced by this plan.
- `rg "轻量反重复" src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` still shows the Style Agent rule (unchanged).
- A diff of `_STYLE_AGENT_PROMPT` content before and after this plan's implementation must be empty.

### Prompt Render Checks

- Render Content Anchor Agent prompt after adding `conversation_progress`.
- Render Dialog Evaluator prompt after adding progression validation.
- Render Episode Recorder prompt.

### Unit Tests

- `pytest tests/test_conversation_episode_state.py`
- `pytest tests/test_conversation_progress_cognition.py`
- `pytest tests/test_conversation_episode_cache.py`
- `pytest tests/test_conversation_progress_runtime.py`
- `pytest tests/test_conversation_progress_module_boundary.py`
- Existing focused tests:
  - `pytest tests/test_service_background_consolidation.py`
  - `pytest tests/test_persona_supervisor2.py`
  - `pytest tests/test_cognition_live_llm_prompt_contracts.py -q` when live LLM is available.

### Smoke

- Send the observed illness progression through a patched/stubbed cognition path and verify the final turn's content anchors do not use `presence_commitment` as the main move again when it is overused.
- Send a sharp topic transition after the illness turn and verify stale illness open loops are not injected into the new content anchors.
- Send a turn after a simulated 3-hour gap and verify `user_state_updates` carry `age_hint` like `~3h ago` so the model can avoid re-asking the symptom as new.
- Simulate a back-to-back turn pair where the recorder for turn N has not finished before turn N+1 loads; verify the cache fallback supplies turn N's state.
- Send a non-responsive group-chat turn and verify episode state is not loaded and the recorder is not queued.
- Start the service and call `/health`.

### Database

- Confirm `conversation_episode_state` exists after bootstrap.
- Confirm unique index on `(platform, platform_channel_id, global_user_id)`.
- Confirm TTL index on `expires_at`.
- Confirm `turn_count`-guarded upsert rejects writes with non-strictly-increasing values.
- Confirm no writes are made to `user_profiles` or `user_profile_memories` for episode-state-only updates.

## Acceptance Criteria

This plan is complete when:

- A compact episode state is loaded before cognition (DB + in-process cache fallback) and updated after final dialog.
- Conversation progress is implemented as a dedicated `kazusa_ai_chatbot.conversation_progress` module with the agreed public facade.
- Existing production code imports only the module facade and does not import `repository`, `cache`, `projection`, or `recorder` internals.
- Episode state is loaded only after relevance chooses to respond; non-responsive turns do not load or write episode state.
- Sharp topic transitions prevent stale episode obligations from influencing the next topic.
- Repeated assistant speech acts are represented as semantic labels emitted by the recorder LLM, not by code-side string canonicalization.
- `user_state_updates` and `open_loops` carry per-entry `first_seen_at`, preserved across turns when the recorder re-emits the same entry, and projected to LLMs as `age_hint`.
- After a long intra-episode gap, the bot does not re-ask already-disclosed symptoms as if new (verified by the 3-hour-gap fixture).
- Content Anchor Agent can emit `[PROGRESSION]` and `[AVOID_REPEAT]`.
- Dialog Evaluator can reject final dialog that ignores explicit anti-repeat anchors **through the existing feedback channel**, with no new graph edges or parallel mechanisms.
- The Style Agent prompt is byte-identical before and after; `forbidden_phrases` continues to fire from its existing local-window source.
- Episode state expires automatically and is separate from user profile memories.
- `turn_count`-guarded upsert and in-process cache fallback together prevent stale-state reads when consecutive user turns arrive faster than recorder completion.
- No new deterministic keyword classifier, regex, or string-equality canonicalization over user/bot natural-language text is introduced. The single permitted exact-string compare is on the recorder's own returned `text` field for `first_seen_at` preservation.
- Tests cover same-episode repetition, sharp transition, empty-state compatibility, age-hint projection, concurrency cache fallback, Style Agent prompt diff, and DB lifecycle.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Episode state leaks across sharp topic transitions | Recorder emits `sharp_transition`; projection suppresses stale obligations | Sharp transition test |
| Episode state changes whether the bot speaks | Load episode state only after relevance returns `should_respond = true`; relevance receives no episode-state fields | Non-responsive turn test and relevance input inspection |
| Local LLM produces noisy or drifting move labels | Recorder LLM resolves equivalence each turn by re-reading prior labels and prior dialog; no code-side canonicalization is required at runtime; cap lists | Recorder unit tests with patched outputs; smoke against the trace fixture |
| Prompt bloat reduces cognition quality | Inject only compact progress object into Content Anchor Agent | Prompt payload snapshot test |
| Repetition fix becomes health-specific | Forbid domain labels and keyword rules | Static grep and fixture review |
| Background recorder failure breaks chat | Run recorder after response and isolate errors at process boundary | Service background test |
| Episode state becomes subjective user image duplicate | Store operational labels only; do not store diary prose; do not store literal dialog text | DB shape tests and prompt review |
| Stale state read when turn N+1 arrives before recorder for turn N completes | `turn_count`-guarded upsert + in-process last-completed cache | Concurrency fixture test |
| Bot re-asks already-disclosed symptoms after a long intra-episode gap | `first_seen_at` preserved across turns; `age_hint` projection surfaces relative age to the model | 3-hour-gap fixture test |
| Stale progression guidance leaks after a misclassified topic transition | Recorder prompt owns continuity classification; projection suppresses obligations only on `sharp_transition`; sharp-transition and multi-language fixtures must inspect stale guidance behavior | Sharp transition test and live progression trace review |
| Implementer accidentally edits the Style Agent prompt while editing the same file | Cutover policy requires byte-identical Style Agent prompt; explicit diff check | Static diff verification |
| Implementer adds a parallel evaluator-feedback channel instead of reusing the existing one | Plan mandates reuse of `HumanMessage(name="evaluator")`; explicit cutover row | Code review gate |

## Rollback / Recovery

- Code rollback path: remove the episode-state plumbing, recorder, prompt additions to Content Anchor and Dialog Evaluator, the in-process cache, tests, and DB exports. The Style Agent file is unchanged in this plan, so no rollback there.
- Data rollback path: drop `conversation_episode_state`; no user-profile or conversation-history data is modified by the feature.
- Irreversible operations: none if implementation follows this plan.
- Required backup: no backup required for rollout, but production operators may snapshot MongoDB before enabling the new collection.
- Recovery verification: service boots, `/chat` works with empty/missing episode state, the in-process cache is empty without errors, and `db.getCollectionNames()` may omit `conversation_episode_state` after rollback.

## Operational Steps

- Deploy code with bootstrap enabled.
- Let `db_bootstrap()` create `conversation_episode_state` and indexes.
- Do not backfill old conversations.
- Monitor logs for episode recorder failures, prompt parse failures, and cache-fallback hit rate.
- Inspect sample documents after live use to confirm labels are operational, compact, and not subjective diary text; confirm `first_seen_at` values are preserved across turns when the recorder re-emits the same entry.

## Execution Evidence

To be filled during implementation:

- Static grep results:
  - `rg "conversation_episode_state|conversation_progress" src/kazusa_ai_chatbot/nodes/relevance_agent.py` returned no matches.
  - `rg "conversation_progress\\.(repository|cache|projection|recorder)" src/kazusa_ai_chatbot --glob "!src/kazusa_ai_chatbot/conversation_progress/**"` returned no matches.
  - `rg "conversation_episode_state" src tests` showed only the planned module, state, service, schema/bootstrap, persona plumbing, and tests.
  - `rg "if .*user_input|if .*decontexualized_input|in .*user_input|in .*decontexualized_input" src/kazusa_ai_chatbot` returned no matches for new deterministic semantic gates.
  - Domain grep showed only existing fixture text in `tests/test_conversation_progression_live_llm.py` and pre-existing `linguistic_texture.py` wording; no domain-specific repetition fix was added.
- Prompt render check results:
  - Content Anchor prompt contains `conversation_progress` and `[AVOID_REPEAT]`.
  - Dialog Evaluator prompt contains `[AVOID_REPEAT]` and `[PROGRESSION]`.
  - Episode Recorder prompt rendered successfully, length `1466`.
- Style Agent prompt diff check:
  - `git diff -- src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py | rg "STYLE_AGENT|轻量反重复|_STYLE_AGENT_PROMPT"` returned no Style Agent prompt changes.
- Test results:
  - `python -m py_compile` over the new `conversation_progress` module and touched service/db/persona/dialog/test files passed.
  - `pytest tests/test_conversation_episode_state.py tests/test_conversation_episode_cache.py tests/test_conversation_progress_runtime.py tests/test_conversation_progress_cognition.py tests/test_conversation_progress_module_boundary.py tests/test_service_background_consolidation.py tests/test_state.py tests/test_persona_supervisor2_schema.py tests/test_persona_supervisor2.py tests/test_dialog_agent.py -q` passed: 45 passed.
  - `pytest tests/test_cognition_live_llm_prompt_contracts.py -q -m live_llm --tb=short --log-cli-level=ERROR` passed: 13 passed.
- Implementation checkpoint status:
  - Checkpoints 1-11 implemented, verified, and ticked.
- Service smoke:
  - Bootstrap plus `health()` smoke passed: `health_status=ok`, `health_db=True`.
- MongoDB index verification:
  - Bootstrap created/verified `conversation_episode_state`.
  - `conversation_episode_scope_unique` exists with `unique=True` and key `[('platform', 1), ('platform_channel_id', 1), ('global_user_id', 1)]`.
  - `conversation_episode_expires_at_ttl` exists with `expireAfterSeconds=0`.
- Concurrency fixture trace:
  - `tests/test_conversation_episode_cache.py::test_cache_document_wins_when_turn_count_is_higher` verifies cache supplies turn N state when DB is behind.
- 3-hour-gap fixture trace:
  - `tests/test_conversation_progress_cognition.py::test_projection_preserves_relative_age_for_prior_disclosure` verifies `~3h ago` projection.
- Manual trace comparison:
  - Final `after_change` live LLM tests were run one at a time with overwrite guards; existing `before_change` traces were preserved.
  - All 8 `after_change` trace files exist under `test_artifacts/llm_traces/conversation_progression_live_after_change__*.json`.
  - Every `after_change` trace contains one record per user turn and includes the prompt-facing `conversation_progress` payload.
  - JSON artifact cleanliness check passed: `total_files_with_hits=0` for Cyrillic terminal-rendering artifacts.
  - A/B summaries:
    - `chinese_baking_collapsed_cake`: before `repeat=1`, `insufficient=1`, `prior_as_new=1`; after `0`, `0`, `0`.
    - `debugging_module_error_zh`: before `1`, `1`, `0`; after `0`, `0`, `0`.
    - `english_essay_revision`: before `1`, `1`, `0`; after `0`, `0`, `0`.
    - `japanese_game_save_bug`: before `0`, `0`, `0`; after `0`, `0`, `0`.
    - `long_chinese_thesis_slides_bonus`: before `2`, `2`, `0`; after `0`, `0`, `0`; turns 9 and 10 address the missing third contribution point / broken logic line.
    - `mixed_language_art_commission`: before `1`, `0`, `0`; after `0`, `0`, `0`.
    - `spanish_calculus_study`: before `1`, `1`, `0`; after `0`, `0`, `0`.
    - `user_illness_trace`: before `1`, `1`, `0`; after `1`, `0`, `0`; manual review accepts this as non-regression plus clear progression improvement because the final turn moves from pure presence commitment into symptom concern and a concrete care suggestion.

## Final Sign-Off Integration Tests

These tests are the final approval gate for this plan. They must be run after implementation, one real LLM test at a time, and the emitted `after_change` JSON traces must be compared against the already-captured `before_change` traces in `test_artifacts/llm_traces`.

Do not batch these tests. Run one test, inspect the output and JSON artifact, then run the next test.

Set the trace phase before the post-change run:

```powershell
$env:CONVERSATION_PROGRESSION_TRACE_PHASE='after_change'
```

Run the integration tests individually:

```powershell
pytest tests/test_conversation_progression_live_llm.py::test_live_progression_user_illness_trace -q -s -m live_llm
pytest tests/test_conversation_progression_live_llm.py::test_live_progression_debugging_module_error_zh -q -s -m live_llm
pytest tests/test_conversation_progression_live_llm.py::test_live_progression_english_essay_revision -q -s -m live_llm
pytest tests/test_conversation_progression_live_llm.py::test_live_progression_japanese_game_save_bug -q -s -m live_llm
pytest tests/test_conversation_progression_live_llm.py::test_live_progression_spanish_calculus_study -q -s -m live_llm
pytest tests/test_conversation_progression_live_llm.py::test_live_progression_chinese_baking_collapsed_cake -q -s -m live_llm
pytest tests/test_conversation_progression_live_llm.py::test_live_progression_long_chinese_thesis_slides_bonus -q -s -m live_llm
pytest tests/test_conversation_progression_live_llm.py::test_live_progression_mixed_language_art_commission -q -s -m live_llm
```

Compare the saved before/after trace summaries:

```powershell
@'
import json
from pathlib import Path

trace_dir = Path('test_artifacts/llm_traces')
case_ids = sorted(
    path.name.removeprefix('conversation_progression_live_before_change__').removesuffix('.json')
    for path in trace_dir.glob('conversation_progression_live_before_change__*.json')
)

for case_id in case_ids:
    before_path = trace_dir / f'conversation_progression_live_before_change__{case_id}.json'
    after_path = trace_dir / f'conversation_progression_live_after_change__{case_id}.json'
    before = json.loads(before_path.read_text(encoding='utf-8'))['payload']
    after = json.loads(after_path.read_text(encoding='utf-8'))['payload']
    print(f'{case_id}')
    print(f'  before: {before["summary"]}')
    print(f'  after:  {after["summary"]}')
    print()
'@ | python -
```

Final sign-off requires:

- Every post-change test above passes without exceptions.
- Every post-change trace contains one record for every user turn in that scenario.
- `after_change` summaries must not regress `repeat_count`, `insufficient_progression_count`, or `prior_disclosure_as_new_count` versus `before_change` for any case.
- The observed illness case, Chinese debugging case, English essay case, Spanish calculus case, Chinese baking case, and long Chinese thesis-slide bonus case must show reduced late-turn fallback to `presence_commitment` or a clearly better manual-review explanation in the judge evidence.
- The Japanese game-save case must remain clean: no new stale-disclosure or insufficient-progression failures.
- The mixed-language art case must preserve its social-progress behavior and must not become more mechanical or overly corrective.
- The long Chinese thesis-slide bonus case must specifically improve turns 9 and 10: the assistant should address the missing third contribution point or the broken story line instead of only saying it is still present.
- JSON artifacts must be clean UTF-8 and must not contain terminal-rendering artifacts. Verify with:

```powershell
@'
from pathlib import Path

bad = []
for path in Path('test_artifacts/llm_traces').glob('conversation_progression_live_after_change__*.json'):
    text = path.read_text(encoding='utf-8')
    hits = sorted(set(ch for ch in text if '\u0400' <= ch <= '\u04FF'))
    if hits:
        bad.append((path.name, ''.join(hits)))

print('CYRILLIC_HITS')
for name, hits in bad:
    print(f'{name}: {hits}')
print(f'total_files_with_hits={len(bad)}')
'@ | python -
```

Record the final A/B result in `Execution Evidence` under `Manual trace comparison` before marking the plan complete.

## Glossary

- `conversation_episode_state`: Short-lived operational state for one user/channel conversation thread.
- `conversation_progress`: LLM-facing compact projection of episode state, including `age_hint` annotations.
- `assistant_move`: A semantic label for the kind of conversational action Kazusa took, such as `presence_commitment` or `direct_answer`. Free-form, LLM-resolved across turns.
- `open_loop`: An unresolved thread that should remain available to the next turn if topic continuity holds. Carries a `first_seen_at` preserved by code across turns.
- `sharp_transition`: The current user turn starts a new topic strongly enough that prior episode obligations should not guide content anchors.
- `first_seen_at`: Timestamp metadata attached by code when an entry first appears in `user_state_updates` or `open_loops`. Preserved on subsequent turns when the recorder re-emits the same entry verbatim.
- `age_hint`: Short natural-language phrase derived from `first_seen_at` and current time (e.g., `~3h ago`). LLM-facing only; code never parses it.
- `turn_count`: Strictly increasing integer per `(platform, channel, user)`. Used for conditional upsert and as the cache-vs-DB recency comparator.
- Last-completed cache: Process-local in-memory dict that holds the most recently written episode-state document per key, used by the next-turn loader as a fallback when the DB document is behind.
