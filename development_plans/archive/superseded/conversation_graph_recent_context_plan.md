# conversation graph recent context plan

## Lifecycle

- Status: superseded.
- Superseded on 2026-05-23 because the conversation-graph/DAG line did not
  prove enough implementation value for the current recall-quality problem.
- The current executable priority is RAG2 cognition-ready evidence, not
  conversation graph implementation.
- This document is historical context only. Do not execute it unless a future
  plan explicitly revives conversation graph work.

## Summary

- Goal: Add a bounded `conversation_graph` subsystem that records recent
  message nodes and typed edges so private and group-chat turns get accurate
  recent-flow context without forcing RAG conversation search to reconstruct
  local adjacency.
- Plan class: high_risk_migration.
- Status: superseded.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`; apply `cjk-safety` before editing
  Python files that contain CJK prompt text.
- Overall cutover strategy: compatible for runtime behavior, migration for
  new MongoDB indexes and bounded recent backfill, bigbang for the new
  `conversation_graph` names.
- Highest-risk areas: increasing live-turn latency, leaking raw graph/storage
  internals into prompts, over-expanding context, weakening RAG capability
  boundaries, and missing low-signal group messages when Kazusa stays silent.
- Acceptance criteria: every persisted live user and assistant message creates
  or attempts a graph node; current turns receive bounded
  `conversation_graph_context`; decontextualizer, cognition, and RAG can use
  exact recent graph context; historical conversation search remains separate;
  all verification and independent review gates pass.

## Context

Kazusa already separates recent chat, short-term conversation progress,
retrieved evidence, durable memory, and reflection. Current recent chat is
still loaded as a flat channel window and then narrowed into a current-user
interaction slice. This works for simple private turns but loses group-chat
topology when nearby human turns belong to the same topic but were not authored
by the current user.

`conversation_progress` is useful semantic episode memory. It deliberately
records strong signals: active thread, open loops, user state, and next
affordances. It does not preserve every weak or noisy message, and it should
not be changed into a raw transcript store. The missing layer is exact recent
message structure:

```text
conversation_history row
  -> conversation_graph node
  -> typed backward edges
  -> bounded recent graph projection
  -> decontextualizer / RAG / cognition
```

The graph records what happened recently, including low-signal private and
group messages. RAG remains responsible for nonlocal historical retrieval and
durable evidence. Cognition remains responsible for character judgment.

Relevant current boundaries:

- `conversation_history` rows are written through `save_conversation(...)`.
- `service.py` loads `chat_history_wide` and `chat_history_recent` before
  graph execution.
- `persona_supervisor2` builds RAG input, then cognition input.
- `build_interaction_history_recent(...)` drops most other-human group rows.
- RAG `conversation_search_agent` performs hybrid historical recall and
  neighbor expansion; it must not become the recent-flow engine.

## Mandatory Skills

- `development-plan`: follow this plan lifecycle, checklist,
  verification, evidence, and review contract.
- `local-llm-architecture`: keep graph construction deterministic; keep LLM
  prompts bounded and semantic; do not add response-path LLM calls.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files containing CJK prompt text,
  including decontextualizer, cognition, or RAG prompt modules.

## Mandatory Rules

- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual edits.
- Use PowerShell `-LiteralPath` for filesystem paths that may contain spaces;
  prefer repo-relative paths.
- Do not read `.env`.
- Do not add a response-path LLM call for graph construction, graph traversal,
  thread selection, or prompt projection.
- Deterministic code owns graph nodes, edge derivation, traversal limits,
  prompt budget, persistence, indexes, TTL, and failure handling.
- LLM stages may consume only the bounded prompt projection. They must not see
  embeddings, MongoDB `_id`, raw wire text, collection names, index names, or
  raw graph internals.
- Keep `conversation_graph` distinct from `conversation_progress`.
  `conversation_graph` preserves recent exact evidence; `conversation_progress`
  preserves semantic episode state.
- Keep `conversation_graph` distinct from historical RAG search.
  `Conversation-graph:` answers current/recent thread questions only;
  `Conversation-evidence:` remains the historical/fuzzy search capability.
- Graph write failure must not roll back a successful `conversation_history`
  write. It must log the failure and return an empty or partial graph context.
- Do not increase `CONVERSATION_HISTORY_LIMIT` or
  `CHAT_HISTORY_RECENT_LIMIT` as the solution to this plan.
- Do not route adapter wire syntax into graph logic. Use typed
  `message_envelope`, `reply_context`, mentions, addressees, and stored
  conversation rows.
- Do not change dialog wording, L3 surface routing, scheduler behavior,
  persistence consolidation, or reflection behavior except where this plan
  explicitly names a prompt-safe context field.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, run
  the Independent Code Review gate and record the result in Execution
  Evidence.

## Must Do

- Create the `kazusa_ai_chatbot.conversation_graph` package.
- Create database helpers for `conversation_graph_nodes` and
  `conversation_graph_threads` under the `kazusa_ai_chatbot.db` boundary.
- Add MongoDB indexes for graph node lookup, thread lookup, and TTL expiry.
- Record graph nodes for live user rows, including listen-only, dropped,
  collapsed, and no-response rows after they are persisted.
- Record graph nodes for assistant rows after they are persisted.
- Derive graph edges deterministically from typed reply context, mentions,
  addressees, same-author bursts, chronology, assistant response metadata, and
  bounded semantic similarity over existing conversation embeddings.
- Produce a bounded `conversation_graph_context` prompt projection for the
  current turn.
- Add `conversation_graph_context` to persona graph state, RAG request context,
  decontextualizer input, and cognition input.
- Add a deterministic `conversation_graph_agent` RAG helper and the
  `Conversation-graph:` capability prefix for recent/local conversation flow.
- Preserve existing `Conversation-evidence:` behavior for historical and fuzzy
  nonlocal search.
- Add focused deterministic tests for graph models, edge building, traversal,
  projection, service wiring, RAG routing, and prompt payload shape.
- Add a bounded recent backfill script for operator use. The script must be
  dry-run by default and must not mutate production data unless explicitly run
  with a write flag.
- Update subsystem docs and the development plan registry.

## Deferred

- Do not replace `conversation_progress`.
- Do not replace `conversation_search_agent` or hybrid historical search.
- Do not add a new vector index for graph nodes.
- Do not store embeddings in `conversation_graph_nodes`.
- Do not build long-term graph memory over full chat history.
- Do not build cross-channel or cross-platform graph traversal.
- Do not add graph-based autonomous contact, scheduler behavior, or proactive
  output behavior.
- Do not add a UI graph viewer.
- Do not tune personality, dialog style, or response ratio in this plan.
- Do not backfill full historical chat. Backfill only the bounded recent
  window specified in this plan.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Runtime context | compatible | Add `conversation_graph_context` beside existing recent history and progress. Do not remove existing fields. |
| Graph naming | bigbang | Use only `conversation_graph` names. Do not create `conversation_flow` aliases or compatibility names. |
| MongoDB collections | migration | Add `conversation_graph_nodes` and `conversation_graph_threads` indexes through `db_bootstrap()`. |
| Existing conversation rows | migration | Provide bounded dry-run-first backfill for recent rows only. Do not require backfill for service startup. |
| RAG capability | compatible | Add `Conversation-graph:` while preserving `Conversation-evidence:` for historical search. |
| Prompts | compatible | Add bounded graph context to existing prompt inputs. Do not remove recent history or progress prompt inputs. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The responsible execution agent must not choose a more conservative strategy
  by default.
- Bigbang graph naming means no `conversation_flow` packages, fields,
  collection names, docs, tests, or prompt labels.
- Migration areas must use the exact index and backfill gates in this plan.
- Compatible areas preserve only the existing surfaces listed above.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local helper decomposition only when the public
  contracts, names, and budgets in this plan remain unchanged.
- The agent must not introduce alternate storage strategies, feature flags,
  compatibility aliases, fallback search paths, extra RAG agents, or prompt
  rewrites outside this plan.
- The agent must treat changes outside `conversation_graph`, `db`, `service`,
  persona supervisor, and RAG routing/projection as high-scrutiny changes.
- The agent must search for existing equivalent helpers before adding new
  utilities. If equivalent behavior exists, reuse or extract it within the
  approved change surface.
- The agent must not perform unrelated cleanup, formatting churn, dependency
  upgrades, or broad refactors.
- If the plan and code disagree, preserve the plan's stated intent and record
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

Every persisted live chat row has a rebuildable graph node. Each node has
backward edges to recent related nodes. Current-turn runtime state includes a
small `conversation_graph_context` that preserves exact recent flow:

```text
current message
  -> reply chain
  -> active thread messages
  -> nearby ambient group context
  -> semantic descriptors for local LLMs
```

Private chat benefits because weak exact recent statements remain available
even when `conversation_progress` did not promote them. Group chat benefits
because multi-speaker topic structure is preserved even when current-user
interaction slicing drops other speakers.

The normal live path adds no new LLM call. The added live cost is bounded
MongoDB reads/writes and deterministic edge/projection code.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Package name | `kazusa_ai_chatbot.conversation_graph` | User-approved name and accurate graph terminology. |
| Graph shape | Directed acyclic recent graph with backward edges only | Prevents cycles and keeps traversal bounded. |
| Storage | Separate `conversation_graph_nodes` and `conversation_graph_threads` collections | Keeps graph rebuildable and separate from canonical `conversation_history`. |
| Graph write integration | Service-level wrapper around `save_conversation(...)` | Keeps DB helper generic while recording graph for live service writes. |
| Graph write failure | Log and continue after conversation row persists | Conversation history is canonical; graph is rebuildable recent context. |
| Semantic edge source | Existing conversation embeddings only | Avoids new embedding calls and new vector indexes. |
| Prompt input | Bounded projection named `conversation_graph_context` | Prevents raw graph dump and keeps local LLM input semantic. |
| RAG capability | New `Conversation-graph:` prefix and deterministic helper | Recent flow is not historical search. |
| Conversation progress | Preserve unchanged | Progress owns semantic episode memory, not exact weak-message topology. |
| Backfill scope | Recent bounded operator script only | The feature is recent-context infrastructure, not long-term graph memory. |

## Contracts And Data Shapes

### Public Package Interface

Create `src/kazusa_ai_chatbot/conversation_graph/__init__.py` exporting:

```python
from kazusa_ai_chatbot.conversation_graph.models import (
    ConversationGraphContext,
    ConversationGraphEdge,
    ConversationGraphMessage,
    ConversationGraphNode,
    ConversationGraphThread,
)
from kazusa_ai_chatbot.conversation_graph.runtime import (
    load_conversation_graph_context,
    record_conversation_graph_node,
)
```

### Runtime Entrypoints

```python
async def record_conversation_graph_node(
    *,
    conversation_row_id: str,
    conversation_doc: Mapping[str, Any],
    active_turn_conversation_row_ids: Sequence[str] | None = None,
    active_turn_platform_message_ids: Sequence[str] | None = None,
) -> ConversationGraphRecordResult:
    """Record or update one graph node for a persisted conversation row."""
```

```python
async def load_conversation_graph_context(
    *,
    platform: str,
    platform_channel_id: str,
    conversation_row_id: str,
    current_timestamp_utc: str,
) -> ConversationGraphContext:
    """Return bounded prompt-safe recent graph context for one current row."""
```

### Node Document

```python
class ConversationGraphNode(TypedDict):
    node_id: str
    conversation_row_id: str
    platform: str
    platform_channel_id: str
    channel_type: str
    role: str
    platform_message_id: str
    platform_user_id: str
    global_user_id: str
    display_name: str
    timestamp: str
    text_excerpt: str
    reply_to_platform_message_id: str
    reply_to_conversation_row_id: str
    mentioned_global_user_ids: list[str]
    addressed_to_global_user_ids: list[str]
    broadcast: bool
    active_turn_conversation_row_ids: list[str]
    active_turn_platform_message_ids: list[str]
    edges: list[ConversationGraphEdge]
    thread_ids: list[str]
    semantic_descriptors: dict[str, str]
    created_at: str
    updated_at: str
    expires_at: str
```

### Edge Shape

```python
class ConversationGraphEdge(TypedDict):
    kind: Literal[
        "reply_to",
        "chronological_prev",
        "same_author_burst",
        "mention_target_recent",
        "assistant_response_to",
        "semantic_similarity",
    ]
    target_node_id: str
    target_conversation_row_id: str
    strength: Literal["strong", "medium", "weak"]
    reason: str
```

### Thread Document

```python
class ConversationGraphThread(TypedDict):
    thread_id: str
    platform: str
    platform_channel_id: str
    channel_type: str
    root_node_id: str
    frontier_node_ids: list[str]
    participant_global_user_ids: list[str]
    topic_hint: str
    signal_level: Literal["direct", "active", "ambient"]
    status: Literal["active", "cooling", "expired"]
    created_at: str
    updated_at: str
    last_activity_at: str
    expires_at: str
```

### Prompt Projection

```python
class ConversationGraphContext(TypedDict):
    available: bool
    anchor: ConversationGraphMessage
    descriptors: dict[str, str]
    reply_chain: list[ConversationGraphMessage]
    active_thread_messages: list[ConversationGraphMessage]
    nearby_ambient_messages: list[ConversationGraphMessage]
    source_refs: list[dict[str, str]]
```

```python
class ConversationGraphMessage(TypedDict):
    speaker: str
    role: str
    timestamp: str
    text: str
    relation_to_anchor: str
    edge_reasons: list[str]
    is_current_turn: bool
```

Prompt-facing graph messages must omit MongoDB `_id`, embeddings,
`raw_wire_text`, collection names, index names, and unbounded metadata.

### Fixed Policy Constants

Create `src/kazusa_ai_chatbot/conversation_graph/policy.py`:

```python
CONVERSATION_GRAPH_TTL_HOURS = 72
CONVERSATION_GRAPH_RECENT_NODE_LIMIT = 80
CONVERSATION_GRAPH_GROUP_LOOKBACK_MINUTES = 45
CONVERSATION_GRAPH_PRIVATE_LOOKBACK_MINUTES = 240
CONVERSATION_GRAPH_MAX_REPLY_CHAIN = 6
CONVERSATION_GRAPH_MAX_THREAD_MESSAGES = 12
CONVERSATION_GRAPH_MAX_AMBIENT_MESSAGES = 4
CONVERSATION_GRAPH_MAX_TEXT_CHARS = 500
CONVERSATION_GRAPH_SEMANTIC_EDGE_MIN_SCORE = 0.82
CONVERSATION_GRAPH_SEMANTIC_EDGE_LIMIT = 2
CONVERSATION_GRAPH_THREAD_FRONTIER_LIMIT = 5
```

Do not add environment variables in this plan. These constants can become
configuration only after production evidence shows a need.

## LLM Call And Context Budget

Before this plan:

- Decontextualizer receives current input, recent flat chat history, reply
  context, indirect speech context, and media descriptions.
- RAG initializer/dispatcher/helper loop receives flat recent/wide history,
  reply context, progress, and other runtime context.
- Cognition receives sliced recent interaction history, reply context,
  conversation progress, RAG result, and promoted reflection context.

After this plan:

- No response-path LLM call count increases.
- The same existing LLM calls receive one additional bounded field:
  `conversation_graph_context`.
- Maximum prompt-facing graph text budget is:
  - reply chain: 6 messages x 500 chars,
  - active thread: 12 messages x 500 chars,
  - ambient: 4 messages x 500 chars,
  - descriptors and labels: under 2,000 chars,
  - total graph projection under 13,000 chars before JSON overhead.
- If the projection exceeds budget, deterministic projection keeps reply-chain
  and active-thread messages first, then drops ambient messages, then clips
  message text to `CONVERSATION_GRAPH_MAX_TEXT_CHARS`.
- `conversation_graph_agent` performs zero LLM calls. It reads only
  `context["conversation_graph_context"]` and returns bounded evidence.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/conversation_graph/__init__.py`: public package
  exports.
- `src/kazusa_ai_chatbot/conversation_graph/README.md`: subsystem contract.
- `src/kazusa_ai_chatbot/conversation_graph/models.py`: TypedDict contracts.
- `src/kazusa_ai_chatbot/conversation_graph/policy.py`: fixed traversal and
  projection budgets.
- `src/kazusa_ai_chatbot/conversation_graph/edge_builder.py`: deterministic
  edge derivation from stored rows and recent graph nodes.
- `src/kazusa_ai_chatbot/conversation_graph/projection.py`: prompt-safe graph
  traversal and projection.
- `src/kazusa_ai_chatbot/conversation_graph/runtime.py`: public runtime facade.
- `src/kazusa_ai_chatbot/db/conversation_graph.py`: raw MongoDB helpers behind
  the DB boundary.
- `src/kazusa_ai_chatbot/rag/conversation_graph_agent.py`: deterministic RAG
  helper for recent graph context.
- `scripts/backfill_conversation_graph_recent.py`: bounded operator backfill,
  dry-run by default.
- `tests/test_conversation_graph_models.py`: model validation and descriptor
  tests.
- `tests/test_conversation_graph_edge_builder.py`: edge construction tests.
- `tests/test_conversation_graph_projection.py`: traversal and prompt budget
  tests.
- `tests/test_conversation_graph_runtime.py`: facade and repository wiring
  tests with patched DB helpers.
- `tests/test_conversation_graph_service_integration.py`: service wrapper
  tests for user and assistant persistence paths.
- `tests/test_rag_conversation_graph_agent.py`: deterministic helper tests.
- `tests/test_rag_conversation_graph_route.py`: dispatcher/projection tests.

### Modify

- `src/kazusa_ai_chatbot/db/bootstrap.py`: add graph indexes and TTL indexes.
- `src/kazusa_ai_chatbot/db/__init__.py`: export graph DB helpers needed by
  runtime and tests.
- `src/kazusa_ai_chatbot/service.py`: add `_save_conversation_with_graph(...)`,
  record graph on live user and assistant saves, load
  `conversation_graph_context`, and place it in initial graph state.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`: preserve assistant row
  id path by returning or exposing the persisted id through the service
  wrapper contract only when needed for graph recording.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: add
  `conversation_graph_context` state keys.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: pass graph context to
  decontextualizer, RAG request, and cognition state.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`:
  add prompt-safe graph context to input JSON and update prompt instructions.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: pass graph
  context into cognition subgraph input.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`: include
  graph context as exact recent scene evidence.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`: include
  graph context as recent factual flow evidence.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2c2.py`:
  include graph context for group social context.
- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`: include graph
  context in RAG request context.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`: register
  `conversation_graph_agent` and `Conversation-graph:` prefix.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`:
  teach initializer to use `Conversation-graph:` for local/recent thread
  references and `Conversation-evidence:` for historical search.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`:
  project graph-agent summaries into `rag_result.conversation_evidence`.
- `src/kazusa_ai_chatbot/rag/README.md`: document the new recent graph
  capability boundary.
- `development_plans/README.md`: register this active draft plan.

### Keep

- Keep `conversation_progress` storage, recorder, projection, and prompts.
- Keep `conversation_search_agent` and hybrid retrieval.
- Keep `get_conversation_history(...)` and recent history fields.
- Keep adapter message-envelope contracts unchanged.

## Overdesign Guardrail

- Actual problem: flat recent history and current-user slicing lose exact
  recent conversational topology, especially low-signal private turns and
  multi-speaker group threads.
- Minimal change: add a rebuildable recent `conversation_graph` layer with
  deterministic edges and a bounded prompt projection, then pass that
  projection to existing decontextualizer, RAG, and cognition stages.
- Ownership boundaries: deterministic code owns graph construction, traversal,
  prompt budget, DB writes, indexes, and failure handling; RAG returns evidence;
  cognition decides character stance; dialog owns final wording.
- Rejected complexity: no new LLM graph builder, no full-history graph memory,
  no new vector index, no graph UI, no cross-channel traversal, no response
  ratio tuning, no `conversation_flow` compatibility alias, and no increase to
  flat recent-history limits as the solution.
- Evidence threshold: add any rejected complexity only after production traces
  or focused tests prove that deterministic recent graph context cannot solve
  the observed private or group failure within the budgets in this plan.

## Implementation Order

### Stage 1 - model and policy contract

- [ ] Step 1: create `tests/test_conversation_graph_models.py`.
  - Test `test_empty_context_is_prompt_safe`.
  - Test `test_prompt_message_rejects_internal_fields`.
  - Expected before implementation: import failure for
    `kazusa_ai_chatbot.conversation_graph.models`.
- [ ] Step 2: create `models.py`, `policy.py`, `__init__.py`, and README with
  the contracts in this plan.
- [ ] Step 3: run
  `venv\Scripts\python -m pytest tests/test_conversation_graph_models.py -q`.
  Expected after implementation: pass.

### Stage 2 - database helpers and indexes

- [ ] Step 4: create `tests/test_conversation_graph_runtime.py` with patched
  repository calls proving:
  - node upsert uses `conversation_row_id` as identity,
  - expired or missing graph data returns `available=False`,
  - graph write failure is surfaced in record result but does not raise.
- [ ] Step 5: create `src/kazusa_ai_chatbot/db/conversation_graph.py` with
  semantic DB helpers:
  - `upsert_conversation_graph_node(...)`
  - `get_conversation_graph_node_by_row_id(...)`
  - `get_conversation_graph_node_by_platform_message_id(...)`
  - `list_recent_conversation_graph_nodes(...)`
  - `upsert_conversation_graph_thread(...)`
  - `list_conversation_graph_threads(...)`
- [ ] Step 6: modify `db/bootstrap.py` to add indexes:
  - `graph_node_row_unique` unique on `conversation_row_id`,
  - `graph_node_platform_channel_message` on platform, channel, message id,
  - `graph_node_platform_channel_ts` on platform, channel, timestamp,
  - `graph_node_thread_ids` on platform, channel, thread ids, timestamp,
  - `graph_node_expires_ttl` TTL on `expires_at`,
  - `graph_thread_platform_channel_activity` on platform, channel,
    last activity,
  - `graph_thread_expires_ttl` TTL on `expires_at`.
- [ ] Step 7: export graph DB helpers from `db/__init__.py`.
- [ ] Step 8: run
  `venv\Scripts\python -m pytest tests/test_conversation_graph_runtime.py -q`.
  Expected after implementation: pass.

### Stage 3 - deterministic edge builder

- [ ] Step 9: create `tests/test_conversation_graph_edge_builder.py` covering:
  - native reply creates a strong `reply_to` edge,
  - same author within the burst window creates `same_author_burst`,
  - mention links to the mentioned user's recent node,
  - assistant rows link to active turn rows with `assistant_response_to`,
  - chronological previous edge is weak and does not create a thread alone,
  - group semantic similarity creates at most two medium edges above threshold,
  - private chat inherits the active thread across a short linear continuation.
- [ ] Step 10: implement `edge_builder.py`.
  - Use only current saved doc, recent graph nodes, and existing embeddings
    present on recent `conversation_history` rows.
  - Do not call embedding endpoints.
  - Do not store embeddings in graph node documents.
- [ ] Step 11: run
  `venv\Scripts\python -m pytest tests/test_conversation_graph_edge_builder.py -q`.
  Expected after implementation: pass.

### Stage 4 - projection and prompt budget

- [ ] Step 12: create `tests/test_conversation_graph_projection.py` covering:
  - reply chain is prioritized over ambient context,
  - active thread messages are deduplicated and chronological,
  - group ambient messages are capped at four,
  - graph context never exposes embeddings, raw wire text, collection names, or
    MongoDB `_id`,
  - unavailable graph returns a valid empty context,
  - descriptors convert raw counts to labels such as `noise_level`,
    `speaker_diversity`, and `thread_fragmentation`.
- [ ] Step 13: implement `projection.py` and `runtime.py`.
- [ ] Step 14: run
  `venv\Scripts\python -m pytest tests/test_conversation_graph_projection.py tests/test_conversation_graph_runtime.py -q`.
  Expected after implementation: pass.

### Stage 5 - service integration

- [ ] Step 15: create `tests/test_conversation_graph_service_integration.py`
  with patched persistence and graph runtime calls proving:
  - `_save_conversation_with_graph(...)` calls `save_conversation(...)` first,
  - graph recording receives the inserted row id and original doc,
  - graph recording failure logs and returns the row id,
  - user persistence passes the graph-enabled save wrapper,
  - assistant persistence passes the graph-enabled save wrapper,
  - `conversation_graph_context` is loaded after user persistence and placed
    into graph initial state.
- [ ] Step 16: modify `service.py`.
  - Add `_save_conversation_with_graph(...)`.
  - Replace live service `save_conversation_func=save_conversation` call sites
    with `_save_conversation_with_graph`.
  - Load `conversation_graph_context` after the surviving user row is saved.
  - Add the context to the initial persona graph state.
- [ ] Step 17: adjust `brain_service/post_turn.py` only if tests prove the
  assistant row id is unavailable to the wrapper contract. Keep the change
  limited to returning the row id already produced by outbound persistence.
- [ ] Step 18: run
  `venv\Scripts\python -m pytest tests/test_conversation_graph_service_integration.py tests/test_service_input_queue.py tests/test_save_conversation_invalidation.py -q`.
  Expected after implementation: pass.

### Stage 6 - persona and prompt integration

- [ ] Step 19: update `persona_supervisor2_schema.py` with
  `conversation_graph_context` in `GlobalPersonaState` and `CognitionState`.
- [ ] Step 20: update `persona_supervisor2.py` and
  `persona_supervisor2_cognition.py` to pass graph context without changing
  existing recent history or progress fields.
- [ ] Step 21: apply `cjk-safety`, then update
  `persona_supervisor2_msg_decontexualizer.py` prompt and input JSON.
  - Instruction: graph context is exact recent conversation flow.
  - Instruction: use graph context to resolve local references when it gives a
    clear target.
  - Instruction: do not treat graph context as durable memory or persona.
- [ ] Step 22: apply `cjk-safety`, then update L1, L2, and L2c2 cognition
  prompts and payloads to include graph context as recent scene evidence.
- [ ] Step 23: add or update tests:
  - `tests/test_persona_supervisor2_schema.py`
  - `tests/test_persona_supervisor2.py`
  - `tests/test_rag_cognitive_episode_adapter.py`
  - Prompt-render tests for every edited `.format(...)` prompt.
- [ ] Step 24: run
  `venv\Scripts\python -m pytest tests/test_persona_supervisor2_schema.py tests/test_persona_supervisor2.py tests/test_rag_cognitive_episode_adapter.py -q`.
  Expected after implementation: pass.

### Stage 7 - RAG conversation graph capability

- [ ] Step 25: create `tests/test_rag_conversation_graph_agent.py` covering:
  - resolved result from non-empty `conversation_graph_context`,
  - unresolved result from unavailable context,
  - no LLM call or database search,
  - evidence summaries preserve speaker, timestamp, relation, and text.
- [ ] Step 26: implement `rag/conversation_graph_agent.py`.
  - Public contract: `async def run(task: str, context: dict, max_attempts: int = 1) -> dict`.
  - Return top-level helper shape with `selected_summary`,
    `projection_payload.summaries`, `evidence`, `resolved_refs`,
    `missing_context`, `conflicts`, `observation_candidates`, and
    `source_hints`.
  - Refuse tasks that ask for nonlocal history or durable memory.
- [ ] Step 27: update `persona_supervisor2_rag_dispatch.py`.
  - Register `conversation_graph_agent`.
  - Add `Conversation-graph:` prefix before `Conversation-evidence:`.
  - Add prompt roster text that defines graph as recent/local only.
- [ ] Step 28: update `persona_supervisor2_rag_initializer.py`.
  - Use `Conversation-graph:` for recent/local references such as current
    thread, reply target, just-mentioned message, and nearby group flow.
  - Use `Conversation-evidence:` for older, broad, historical, aggregate, or
    fuzzy searches outside the graph window.
- [ ] Step 29: update `persona_supervisor2_rag_projection.py` so resolved
  `conversation_graph_agent` summaries append to
  `rag_result.conversation_evidence`.
- [ ] Step 30: add `tests/test_rag_conversation_graph_route.py` covering direct
  prefix dispatch and projection.
- [ ] Step 31: run
  `venv\Scripts\python -m pytest tests/test_rag_conversation_graph_agent.py tests/test_rag_conversation_graph_route.py tests/test_rag_phase3_route_mapping.py tests/test_rag_projection.py -q`.
  Expected after implementation: pass.

### Stage 8 - bounded backfill and docs

- [ ] Step 32: create `scripts/backfill_conversation_graph_recent.py`.
  - Default mode is dry-run.
  - Required filters: platform and platform channel id, or `--all-recent`.
  - Default window: last 72 hours.
  - Default batch size: 100.
  - Write mode requires `--write`.
  - Output reports rows scanned, nodes to upsert, threads to upsert, skipped
    rows, and errors.
- [ ] Step 33: create `tests/test_conversation_graph_backfill_script.py`
  using patched DB helpers.
- [ ] Step 34: update docs:
  - `conversation_graph/README.md`
  - `rag/README.md`
  - `db/README.md`
  - `development_plans/README.md`
- [ ] Step 35: run
  `venv\Scripts\python -m pytest tests/test_conversation_graph_backfill_script.py -q`.
  Expected after implementation: pass.

### Stage 9 - full verification and review

- [ ] Step 36: run the full verification commands listed in this plan.
- [ ] Step 37: run the Independent Code Review gate.
- [ ] Step 38: fix review findings inside the approved change surface.
- [ ] Step 39: rerun affected verification commands.
- [ ] Step 40: record final Execution Evidence and update lifecycle status only
  after review approval.

## Progress Checklist

- [ ] Stage 1 complete - model and policy contract established.
  - Covers: steps 1-3.
  - Verify: `venv\Scripts\python -m pytest tests/test_conversation_graph_models.py -q`.
  - Evidence: record changed files and test output in Execution Evidence.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 complete - DB helpers and indexes implemented.
  - Covers: steps 4-8.
  - Verify: `venv\Scripts\python -m pytest tests/test_conversation_graph_runtime.py -q`.
  - Evidence: record helper exports, index names, and test output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 complete - deterministic edge builder implemented.
  - Covers: steps 9-11.
  - Verify: `venv\Scripts\python -m pytest tests/test_conversation_graph_edge_builder.py -q`.
  - Evidence: record edge types covered and test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 complete - prompt-safe projection implemented.
  - Covers: steps 12-14.
  - Verify: `venv\Scripts\python -m pytest tests/test_conversation_graph_projection.py tests/test_conversation_graph_runtime.py -q`.
  - Evidence: record budget behavior and test output.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 complete - service write/load integration implemented.
  - Covers: steps 15-18.
  - Verify: `venv\Scripts\python -m pytest tests/test_conversation_graph_service_integration.py tests/test_service_input_queue.py tests/test_save_conversation_invalidation.py -q`.
  - Evidence: record user and assistant persistence coverage.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 complete - persona prompt integration implemented.
  - Covers: steps 19-24.
  - Verify: `venv\Scripts\python -m pytest tests/test_persona_supervisor2_schema.py tests/test_persona_supervisor2.py tests/test_rag_cognitive_episode_adapter.py -q`.
  - Evidence: record prompt-render checks and test output.
  - Handoff: next agent starts at Stage 7.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 7 complete - RAG graph capability implemented.
  - Covers: steps 25-31.
  - Verify: `venv\Scripts\python -m pytest tests/test_rag_conversation_graph_agent.py tests/test_rag_conversation_graph_route.py tests/test_rag_phase3_route_mapping.py tests/test_rag_projection.py -q`.
  - Evidence: record route/projection behavior and test output.
  - Handoff: next agent starts at Stage 8.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 8 complete - backfill script and docs implemented.
  - Covers: steps 32-35.
  - Verify: `venv\Scripts\python -m pytest tests/test_conversation_graph_backfill_script.py -q`.
  - Evidence: record dry-run behavior and docs changed.
  - Handoff: next agent starts at Stage 9.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 9 complete - final verification and independent review complete.
  - Covers: steps 36-40.
  - Verify: every command in Verification passes or has a recorded blocker.
  - Evidence: record verification output, review findings, fixes, reruns, and
    residual risks.
  - Handoff: plan can move to completed only after this stage is signed off.
  - Sign-off: `<agent/date>` after independent review approval.

## Verification

### Static Greps

- `rg "conversation_flow" src tests development_plans --glob "!development_plans/active/short_term/conversation_graph_recent_context_plan.md"`
  - Expected: no matches outside this plan's explicit forbidden-name
    guardrails. Nonzero `rg` exit for no matches is acceptable.
- `rg "conversation_graph_context" src tests`
  - Expected: matches in graph package, service/persona/RAG integration, and
    tests only.
- `rg "Conversation-graph:" src tests`
  - Expected: matches in RAG initializer/dispatcher tests and docs only.
- `rg "CONVERSATION_HISTORY_LIMIT|CHAT_HISTORY_RECENT_LIMIT" src/kazusa_ai_chatbot/config.py src/kazusa_ai_chatbot/service.py`
  - Expected: existing matches only. This plan must not increase those limits.

### Focused Tests

- `venv\Scripts\python -m pytest tests/test_conversation_graph_models.py -q`
- `venv\Scripts\python -m pytest tests/test_conversation_graph_edge_builder.py -q`
- `venv\Scripts\python -m pytest tests/test_conversation_graph_projection.py -q`
- `venv\Scripts\python -m pytest tests/test_conversation_graph_runtime.py -q`
- `venv\Scripts\python -m pytest tests/test_conversation_graph_service_integration.py -q`
- `venv\Scripts\python -m pytest tests/test_rag_conversation_graph_agent.py tests/test_rag_conversation_graph_route.py -q`
- `venv\Scripts\python -m pytest tests/test_conversation_graph_backfill_script.py -q`

### Regression Tests

- `venv\Scripts\python -m pytest tests/test_conversation_history_envelope.py tests/test_service_input_queue.py tests/test_save_conversation_invalidation.py -q`
- `venv\Scripts\python -m pytest tests/test_persona_supervisor2_schema.py tests/test_persona_supervisor2.py tests/test_persona_supervisor2_rag2_integration.py -q`
- `venv\Scripts\python -m pytest tests/test_rag_cognitive_episode_adapter.py tests/test_rag_phase3_route_mapping.py tests/test_rag_projection.py tests/test_rag_hybrid_agents.py -q`
- `venv\Scripts\python -m pytest tests/test_conversation_progress_runtime.py tests/test_conversation_progress_cognition.py -q`

### Compile

- `venv\Scripts\python -m compileall src/kazusa_ai_chatbot/conversation_graph src/kazusa_ai_chatbot/rag src/kazusa_ai_chatbot/nodes src/kazusa_ai_chatbot/db`

### Backfill Dry Run

- `venv\Scripts\python -m scripts.backfill_conversation_graph_recent --platform qq --platform-channel-id 902317662 --hours 72`
  - Expected: reports scanned and planned graph rows without writing.
  - This command requires a configured MongoDB. If MongoDB is unavailable,
    record the blocker and run the unit test gate instead.

### Live LLM Tests

No live LLM test is required for plan completion. If an execution agent
runs live LLM tests for prompt confidence, run one case at a time and inspect
the output before running another case.

## Independent Plan Review

Run this gate before approval, execution, or handoff. Prefer a reviewer that
did not draft the plan. If no separate reviewer is available, the drafting agent
must reread this plan, the relevant subsystem READMEs, and the source/test
context from a fresh-review posture.

Review scope:

- The plan preserves adapter, brain service, RAG, cognition, dialog,
  persistence, and conversation-progress ownership boundaries.
- The plan gives concrete contracts, exact file paths, data shapes, budgets,
  implementation order, verification commands, and evidence requirements.
- Agent creativity is bounded: no unresolved choices, alternate storage
  strategies, compatibility aliases, fallback search paths, or unowned helper
  freedom remain.
- The new RAG capability is recent/local only and cannot replace historical
  conversation evidence.
- Prompt-facing context is bounded and does not expose graph/storage internals.

Record blockers, non-blocking findings, required edits, and approval status.
Approve only when blockers are resolved.

## Independent Code Review

Run this gate after all Verification commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Alignment with Must Do, Deferred, Agent Autonomy Boundaries, Change Surface,
  exact contracts, implementation order, verification gates, and acceptance
  criteria.
- Code quality and design weaknesses, including graph ownership boundaries,
  hidden fallback paths, compatibility aliases, prompt/RAG payload leaks,
  persistence risk, brittle fixtures, and avoidable blast radius.
- Regression and handoff quality, including execution evidence, focused tests,
  regression tests, static checks, docs, and lifecycle records.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only fixture or
documentation corrections. If a fix would cross the approved boundary or alter
the contract, stop and update the plan or request approval before changing
code.

Record findings, fixes, commands rerun, residual risks, and approval status in
Execution Evidence.

## Acceptance Criteria

This plan is complete when:

- `conversation_graph_nodes` and `conversation_graph_threads` indexes are
  created through `db_bootstrap()`.
- Live user and assistant conversation rows create or attempt graph nodes after
  `conversation_history` persistence.
- Current persona state includes bounded `conversation_graph_context`.
- Decontextualizer and cognition can consume graph context without losing
  existing recent history or conversation progress inputs.
- RAG supports `Conversation-graph:` for current/recent local flow and keeps
  `Conversation-evidence:` for historical conversation search.
- Private-chat tests prove weak recent statements can enter graph context even
  when not present in conversation progress.
- Group-chat tests prove other speakers' relevant nearby turns remain visible
  through graph context even when current-user interaction slicing would drop
  them.
- Static greps show no `conversation_flow` naming outside this plan's
  explicit forbidden-name guardrails.
- Every Verification command passes or has a recorded environmental blocker.
- Independent Code Review is complete and approved.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Live latency increases | No new LLM call; cap recent node reads and projection size | Runtime tests plus service smoke profiling if available |
| Graph context bloats prompts | Fixed projection caps and field stripping | Projection budget tests |
| Graph becomes hidden memory | TTL collections, prompt wording, and separate progress/memory ownership | Model/projection tests and prompt review |
| Group threads attach incorrectly | Explicit edge tests for reply, mention, burst, chronology, and semantic links | Edge-builder tests with private and group fixtures |
| Historical RAG behavior regresses | Preserve `Conversation-evidence:` and hybrid agents | RAG route/projection regression tests |
| Graph write failure breaks chat | Graph write logs and degrades after conversation row persists | Service integration test |
| Backfill mutates too much data | Dry-run default, bounded hours, explicit `--write` | Backfill script tests and dry-run command |

## Execution Evidence

- Stage 1 evidence:
- Stage 2 evidence:
- Stage 3 evidence:
- Stage 4 evidence:
- Stage 5 evidence:
- Stage 6 evidence:
- Stage 7 evidence:
- Stage 8 evidence:
- Stage 9 verification:
- Independent Code Review:

## Execution Handoff

This draft is not approved for implementation until the user approves it and
the Independent Plan Review gate is complete. After approval, execute stages in
order. Start with Stage 1 and do not begin Stage 2 until Stage 1 verification
and evidence are recorded.
