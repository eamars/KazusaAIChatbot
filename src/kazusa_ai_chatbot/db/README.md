# Database

`kazusa_ai_chatbot.db` is the MongoDB persistence layer for the chatbot runtime.

It owns database connection management, document schemas, startup bootstrap, collection-level indexes, embeddings for searchable records, and focused read/write helpers for conversation history, user profiles, character state, memories, and scheduled events.

It is not a cognition layer, not a RAG planner, not a consolidator, and not a semantic classifier. The DB module stores and retrieves structured state. LLM-facing interpretation happens in the caller that decides what should be written or searched.

## Public Boundary

Production callers should import through the package facade:

```python
from kazusa_ai_chatbot.db import (
    db_bootstrap,
    get_db,
    close_db,
    save_conversation,
    get_conversation_history,
    search_conversation_history,
    resolve_global_user_id,
    get_user_profile,
    insert_profile_memories,
    build_user_profile_recall_bundle,
    get_character_profile,
    upsert_character_state,
    save_memory,
    search_memory,
)
```

The facade preserves the public API while the package is split internally by collection domain. Callers should avoid reaching into private client helpers unless they need a raw MongoDB handle for a new DB-owned operation.

## Runtime Lifecycle

```text
service startup
  -> db_bootstrap()
       connects through the lazy MongoDB client
       creates required collections
       seeds the singleton character_state document
       creates regular indexes
       best-effort creates vector-search indexes
       drops legacy RAG1 cache collections if present
  -> scheduler rebuilds pending events from scheduled_events
  -> service handles chat requests

incoming message
  -> resolve or create global_user_id for platform account
  -> save user message in conversation_history
       generate embedding if missing
       invalidate matching Cache 2 conversation-history entries
  -> load recent history/profile/state for the graph
  -> persona pipeline runs
  -> save assistant message in conversation_history
  -> background consolidator writes accepted durable state
       character_state
       user_profile_memories
       user_profiles image/relationship fields
       scheduled_events through the dispatcher/scheduler
```

The database connection is lazy and process-scoped. `get_db()` opens the Motor client on first use and reopens it if the active event loop changes. `close_db()` is called during service shutdown.

## Collection Model

The main MongoDB collections are:

| Collection | Ownership |
|---|---|
| `conversation_history` | Append-only chat transcript rows with platform scope, message identity, reply metadata, attachments, timestamps, and embeddings. Used by service history loading and RAG conversation retrieval. |
| `user_profiles` | Lightweight user identity/profile header keyed by `global_user_id`. Stores linked platform accounts, suspected aliases, relationship metrics, last relationship insight, and the generated user image document. |
| `user_profile_memories` | Authoritative user-scoped memory store for diary entries, objective facts, milestones, and active commitments. This is the main durable profile-memory substrate. |
| `character_state` | Singleton `_id: "global"` document containing both static character profile fields and runtime character state such as mood, vibe, reflection summary, and self image. |
| `memory` | General persistent memory collection used by RAG memory retrieval. Supports keyword and vector search over durable memory entries. |
| `scheduled_events` | Durable scheduler queue for future tool execution created by the dispatcher. |
| `conversation_episode_state` | Short-lived conversation-progress state keyed by platform, channel, and user. It has a TTL index and is conceptually owned by the conversation progress module. |

The old `rag_cache_index` and `rag_metadata_index` collections are legacy RAG1/Cache1 artifacts. Bootstrap drops them idempotently when they are present.

## Schemas

Document shapes live as `TypedDict` contracts. They are type hints and shared conventions, not strict MongoDB validators.

Important schema families:

- `ConversationMessageDoc` for chat transcript rows.
- `UserProfileDoc` for profile headers.
- `UserProfileMemoryDoc` for profile memories.
- `CharacterProfileDoc` for the singleton character document.
- `MemoryDoc` for general persistent memories.
- `ScheduledEventDoc` for scheduler events.
- `ConversationEpisodeStateDoc` for short-term conversation progress.

Most schemas use `total=False` so new fields can be added without forcing a migration for every historical document. Callers should treat missing fields as normal and provide defaults at projection/read boundaries.

## Identity And Profiles

The stable user key is `global_user_id`.

Platform accounts are linked under `user_profiles.platform_accounts`:

```python
{
    "platform": str,
    "platform_user_id": str,
    "display_name": str,
    "linked_at": str,
}
```

`resolve_global_user_id(...)` looks up an existing profile by platform account or creates a new profile with a fresh UUID. `ensure_character_identity(...)` reserves the character's stable global identity and prevents the bot account from being attached to another profile.

The profile header deliberately stays lightweight. Durable diary entries, objective facts, milestones, and commitments live in `user_profile_memories`, then are hydrated into prompt-facing profile bundles by recall helpers.

## Profile Memories

`user_profile_memories` is the authoritative profile-memory collection.

Memory types are:

```python
MemoryType.DIARY_ENTRY
MemoryType.OBJECTIVE_FACT
MemoryType.MILESTONE
MemoryType.COMMITMENT
```

`insert_profile_memories(...)` fills structural fields such as `memory_id`, owner ID, timestamps, expiry, and embeddings. It also applies DB-level idempotency where the storage model requires it:

- commitments update existing active rows by `commitment_id` or `dedup_key`,
- objective facts with an existing `dedup_key` are not duplicated,
- milestones with the same LLM-supplied scope supersede older live milestones.

This is storage-level idempotency, not semantic classification. The LLM/consolidator decides what category a memory belongs to and which dedup or scope fields it emits.

Read helpers combine recent memories, active commitments, and optional semantic recall into prompt-facing blocks:

```python
{
    "character_diary": list,
    "objective_facts": list,
    "active_commitments": list,
    "milestones": list,
    "memories": list,
}
```

`build_user_profile_recall_bundle(...)` merges those blocks with the profile header so RAG and cognition can consume a compact user-profile view.

## Conversation History

`conversation_history` stores user and assistant messages with platform/channel scope and embeddings.

The main access patterns are:

- recent or filtered history through `get_conversation_history(...)`,
- keyword or semantic search through `search_conversation_history(...)`,
- grouped factual counts through `aggregate_conversation_by_user(...)`,
- append writes through `save_conversation(...)`.

`save_conversation(...)` generates an embedding if the caller did not provide one. After inserting the row, it emits a Cache 2 invalidation event for `source="conversation_history"`. This invalidation lives at the DB boundary because conversation writes happen from multiple service paths and every write should make matching RAG conversation caches stale.

Conversation search helpers remove embeddings from returned vector results before handing them back to callers.

## Character State

`character_state` is a singleton document with `_id: "global"`.

It stores two categories of data in one document:

- static personality/profile fields loaded from the character profile,
- runtime state fields such as mood, global vibe, reflection summary, update timestamp, and character self image.

`upsert_character_state(...)` treats empty strings as "preserve existing value" for mood, vibe, and reflection summary. This lets the consolidator update only fields it has evidence for while still stamping `updated_at`.

## General Memory

The `memory` collection is a general persistent-memory store used by RAG memory retrieval.

`save_memory(...)` is append-only. It embeds a combined text payload containing type, source, title, and content. Superseding and filtering are represented by metadata fields such as `memory_type`, `source_kind`, `status`, and expiry timestamps.

`search_memory(...)` supports:

- keyword search over `memory_name` and `content`,
- vector search over the `embedding` field,
- filters by source user, memory type, source kind, status, and expiry range.

## Scheduled Events

`scheduled_events` persists future tool execution requested by the dispatcher.

The DB module only defines the document contract. The scheduler owns event lifecycle:

```text
pending -> running -> completed
pending -> running -> failed
pending -> cancelled
```

Scheduler events carry the original dispatch context so delayed execution can be rehydrated after service restart.

## Embeddings And Vector Search

The DB layer owns embedding generation for records that need semantic search.

Embedding helpers use the configured OpenAI-compatible embedding endpoint and model. A small semaphore and batch size limit bound concurrent embedding requests.

Vector-search indexes are created best-effort during bootstrap:

- `conversation_history_vector_index` on `conversation_history.embedding`,
- `memory_vector_index` on `memory.embedding`,
- `user_profile_memories_vector` on `user_profile_memories.embedding` with filter fields.

Atlas vector search may not be available in every environment. Bootstrap logs a warning and continues when vector-index creation fails.

## Cache Interaction

The DB module has one direct Cache 2 invalidation responsibility:

- `save_conversation(...)` invalidates `conversation_history` cache dependencies after a successful message insert.

Other durable writes, such as `user_profile_memories`, `user_profiles`, and `character_state`, are invalidated by the consolidator after it knows which write steps succeeded. DB helpers should not import agent classes or know which RAG helpers cache their results.

## Semantic Ownership

LLMs and higher-level modules own semantic decisions:

- what user facts or commitments should be persisted,
- whether a message implies a durable preference or future promise,
- what memory type a harvested item belongs to,
- what search capability should be used for a question,
- how retrieved records should affect Kazusa's response.

The DB module owns deterministic mechanics:

- connection lifecycle,
- index creation,
- structural defaults,
- embedding generation,
- scoped queries and updates,
- storage-level idempotency,
- vector/keyword search execution,
- cache invalidation at the conversation-write boundary.

Do not add natural-language classification or keyword-based acceptance logic to DB helpers. If a persisted semantic category is wrong, fix the upstream LLM prompt, schema, or evaluator that produced it.

## Operational Notes

`db_bootstrap()` is safe to run on every service start. It creates missing collections and indexes, but it does not rewrite arbitrary user data.

Most read helpers strip MongoDB internals such as `_id` or remove large embedding arrays before returning prompt-facing results.

The DB package intentionally keeps migration and maintenance scripts outside the runtime package. Scripts may call public DB helpers, but runtime code should keep startup work bounded to bootstrap.

## Test Coverage

Relevant tests include:

- `tests/test_db.py`
- `tests/test_user_profile_memories.py`
- `tests/test_save_conversation_invalidation.py`
- `tests/test_scheduler_future_promise.py`
- DB-backed live coverage inside `tests/test_e2e_live_llm.py`

Vector-search tests may require an environment that supports Atlas search indexes. Live DB tests should be treated as integration evidence, not only unit pass/fail checks.
