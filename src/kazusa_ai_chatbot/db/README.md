# Database Layer Interface Control Document

## Document Control

- ICD id: `DB-ICD-001`
- Owning package: `kazusa_ai_chatbot.db`
- Interface boundary: runtime packages and maintenance scripts -> MongoDB
  persistence through named database helpers
- Runtime consumers: brain service, RAG, cognition, consolidation, scheduler,
  dispatcher, reflection cycle, memory evolution, and conversation progress
- Maintenance consumers: scripts under `src/scripts` through
  `kazusa_ai_chatbot.db.script_operations`
- Backend owner: database package internals, including MongoDB client,
  collection names, indexes, raw query/update documents, and embedding writes

This document defines the database package contract. It is the source of truth
for which code may touch MongoDB primitives, which public helpers callers may
use, what durable collections mean, and how storage changes must be introduced.

For MongoDB environment variables, local setup, startup commands, and
operator-oriented cleanup commands, use the operational
[HOWTO](../../../docs/HOWTO.md). This ICD owns collection ownership and
document-shape contracts.

## Purpose

The database layer is the durable persistence boundary for Kazusa. It stores
conversation history, user identity and relationship headers, long-term user
memory units, character state, curated shared memory, reflection runs,
scheduled events, conversation progress, interaction-style overlays, and
persistent RAG initializer cache entries.

The package deliberately hides raw MongoDB access from the rest of the system.
Callers express intent through semantic helpers such as
`save_conversation(...)`, `query_user_memory_units(...)`,
`insert_scheduled_event(...)`, or `list_reflection_scope_messages(...)`.
Database internals translate that intent into collection access, indexes,
filters, projections, aggregation pipelines, update operators, and exception
handling.

This keeps storage mechanics inspectable without leaking backend syntax across
the chatbot architecture.

## Scope

This ICD covers:

- Public runtime imports from `kazusa_ai_chatbot.db`.
- Public maintenance imports from `kazusa_ai_chatbot.db.script_operations`.
- Private ownership of `db._client.get_db()` and raw MongoDB primitives.
- Collection purpose, document ownership, and read/write responsibilities.
- Reflection-related database interfaces.
- Error behavior and bootstrap/index expectations.
- Change-control rules for adding or modifying database operations.

Related operational details, such as MongoDB deployment and credentials, live in
the HOWTO or deployment configuration. Platform adapters, LLM prompts, and
one-off production repairs integrate with the database through the public
runtime or maintenance interfaces described here.

## Parties

### Runtime Callers

Runtime callers are production packages on the live or background service path:
brain service, RAG, cognition, dialog persistence, consolidation, scheduler,
dispatcher, reflection cycle, conversation progress, and memory evolution.

Runtime callers import from the `kazusa_ai_chatbot.db` facade unless this ICD
names a narrower public package boundary for that subsystem.

### Maintenance Scripts

Maintenance scripts are operator tools under `src/scripts`. They may need
export, migration, backfill, scan, or repair operations that are not part of
the normal runtime API.

Maintenance scripts use `kazusa_ai_chatbot.db.script_operations` for those
operator-only operations.

### Database Package Internals

The database package owns:

- MongoDB client lifecycle.
- Collection names.
- Collection handles.
- Index creation.
- Query, projection, aggregation, sort, and update documents.
- Vector-search and embedding persistence mechanics.
- Translation from backend failures into application-level errors.

Backend driver access is owned by modules under `src/kazusa_ai_chatbot/db/`.

## Boundary Summary

```text
runtime subsystem
  -> kazusa_ai_chatbot.db facade function
  -> db submodule semantic helper
  -> db._client.get_db()
  -> MongoDB collection operation

maintenance script
  -> kazusa_ai_chatbot.db.script_operations semantic helper
  -> db._client.get_db()
  -> MongoDB collection operation
```

Raw database access terminates inside the database package. Runtime packages and
scripts receive semantic return values and application-level errors.

## Public Runtime Facade

Production callers should import runtime database functions, document
TypedDicts, and application-level exceptions from:

```python
from kazusa_ai_chatbot import db
```

or:

```python
from kazusa_ai_chatbot.db import save_conversation, DatabaseOperationError
```

The runtime facade exports helpers for:

- database health and shutdown: `check_database_connection(...)`,
  `close_db(...)`;
- bootstrap: `db_bootstrap(...)`;
- embeddings and vector-index setup exposed as runtime capabilities:
  `get_text_embedding(...)`, `get_text_embeddings_batch(...)`,
  `enable_vector_index(...)`;
- conversation history: save, recent-history retrieval, semantic/keyword
  search, aggregation by user, and attachment-description repair;
- reflection conversation reads and reflection-run persistence;
- user identity, platform-account linking, affinity, display-name search, and
  relationship insight updates;
- user memory unit creation, validation, keyword search, vector search, and
  semantic/window updates;
- character profile, character state, and character self-image persistence;
- interaction-style image overlays for reflection-derived style guidance;
- scheduled-event insert, query, state transition, completion, failure, and
  cancellation;
- persistent RAG initializer cache entries;
- legacy shared-memory facade functions that are lazily resolved to avoid
  import cycles.

Callers treat facade helpers as semantic operations. New storage behavior gets a
named helper.

## Public Maintenance Interface

Maintenance-only functionality lives in:

```python
from kazusa_ai_chatbot.db import script_operations
```

or by importing named helpers from:

```python
from kazusa_ai_chatbot.db.script_operations import export_collection_rows
```

This submodule is public for operator tools. It can expose broader export,
migration, scan, and repair helpers than the runtime facade. Scripts pass
semantic parameters when a helper exists.

Script orchestration handles argument parsing, dry-run policy, file IO, and
operator reporting; the database package owns backend expressions behind named
helpers.

## Internal Client Boundary

`kazusa_ai_chatbot.db._client.get_db()` is private to
`src/kazusa_ai_chatbot/db/`.
Database package internals own backend handles, query/update documents,
indexes, driver exceptions, and translation to public errors.

## Embedding Role Contract

The database package owns the distinction between query embeddings and document
embeddings.

A document embedding represents durable text stored for later retrieval. Current
document rows include `conversation_history` messages, `memory` rows, and
`user_memory_units` rows. Document text remains stored without embedding-model
task prefixes; only the text sent to the embedding endpoint may be role-shaped.

A query embedding represents the retrieval intent used to search stored
documents. Current query text includes RAG conversation-search slots,
shared-memory semantic search requests, scoped user-memory evidence requests,
and semantic user-profile hydration requests.

Callers must use role-specific helpers when intent matters:

```python
from kazusa_ai_chatbot.db import (
    get_document_text_embedding,
    get_query_text_embedding,
)
```

Document write paths and re-embedding maintenance scripts use document-role
helpers. Vector search paths use query-role helpers. The legacy
`get_text_embedding(...)` and `get_text_embeddings_batch(...)` helpers remain
available as document-role compatibility APIs; they must not be used for vector
query generation.

For `text-embedding-nomic-embed-text-v2-moe`, the embedding adapter applies the
model's task-instruction prefixes inside `db._client`: query text is embedded as
search intent, and document text is embedded as retrievable evidence. These
prefixes are adapter implementation details, not durable database content and
not prompt-facing text.

## Error Contract

Public database helpers raise application-level exceptions when a backend
operation fails. The package-level exception is:

```python
from kazusa_ai_chatbot.db import DatabaseOperationError
```

Callers can catch `DatabaseOperationError` when they can degrade gracefully,
record telemetry, or return a rejected operation.

Functions that return `bool` for state transitions, such as scheduled-event
status changes, use `False` for expected non-matches or invalid state
transitions. Backend failures remain exceptional.

## Collection Contracts

### `conversation_history`

Stores raw conversation rows plus normalized message-envelope fields,
attachment descriptions, embeddings, addressing fields, and metadata used by
recent-history and RAG retrieval.

Primary owners:

- write: live chat persistence through `save_conversation(...)`;
- read: recent conversation history, RAG search, reflection evidence reads,
  and maintenance exports;
- update: attachment-description repair and approved maintenance helpers.

Semantic search uses `body_text` plus attachment descriptions as the embedding
source. `raw_wire_text` remains audit/replay data.

### `user_profiles`

Stores durable user identity and relationship headers: global user id, linked
platform accounts, suspected aliases, display names, affinity, and
`last_relationship_insight`.

It is not the cognition-facing long-term memory store. User facts,
commitments, patterns, and relationship signals belong in `user_memory_units`.

### `user_memory_units`

Stores durable user memory units used by RAG and consolidation.

Each unit stores the same semantic triple:

```json
{
  "unit_type": "stable_pattern | recent_shift | objective_fact | milestone | active_commitment",
  "fact": "concrete event/fact/commitment",
  "subjective_appraisal": "Kazusa's subjective interpretation",
  "relationship_signal": "how this should affect future interaction",
  "updated_at": "ISO timestamp"
}
```

RAG owns retrieval and projection. The consolidator owns extraction,
merge/evolve/create decisions, rewriting, and stability classification.
Database code validates structure, known ids, search constraints, and
persistence mechanics, but it does not reinterpret user meaning.

### `character_state`

Stores singleton character profile/state documents and runtime self-image
material. Character state updates are explicit persistence events owned by
named service or promotion paths.

### `memory`

Stores curated shared/world/common-sense memory and evolving-memory rows used
outside per-user memory. Legacy compatibility helpers remain available through
the facade, but new reflection memory promotion should go through the
`memory_evolution` public API rather than raw legacy writes.

### `character_reflection_runs`

Stores hourly, daily-channel, and global-promotion reflection run documents.
These rows are evidence and audit records for the reflection cycle. Normal
cognition uses promoted, gated reflection context.

Reflection reads and writes are split intentionally:

- `db.conversation_reflection` reads bounded conversation evidence for
  reflection scopes.
- `db.reflection_cycle` persists and retrieves reflection run documents.
- `reflection_cycle.context` may expose only promoted, gated context from the
  memory layer, not raw run documents.

### `interaction_style_images`

Stores reflection-derived interaction-style overlays keyed by user or group
channel. These documents are compact runtime guidance, not raw reflection
transcripts. Validation must reject event-like details that should remain in
reflection evidence or memory systems.

### `scheduled_events`

Stores pending, running, completed, failed, and cancelled scheduled tool
events. The dispatcher and scheduler own semantic use of this collection
through named helper functions.

### `self_cognition_action_attempts`

Stores durable action-attempt state for idle self-cognition duplicate
suppression. The collection is keyed by `idempotency_key`, which is derived
from source kind, source id, due time, target scope, and action kind. It does
not replace event logging; event logs remain the sanitized operator view.

### `conversation_episode_state`

Stores short-lived conversation-progress state keyed by platform, channel, and
user. The collection is operational working memory, not durable identity memory.
Writes are guarded so stale background records preserve newer episode progress.

### `rag_cache2_persistent`

Stores persistent initializer cache entries for RAG Cache2. The collection is
owned by cache helpers that build version keys, load entries, record hits, and
prune stale data.

### `event_log_events`

Append-only canonical observability stream. Runtime callers do not write this
collection directly; they call the public `kazusa_ai_chatbot.event_logging`
interface, and event-logging internals call the DB adapter.

The collection stores sanitized event families for process lifecycle, workers,
LLM stage metadata, runtime errors, queue/pipeline decisions, RAG stages,
dialog quality, dispatcher outcomes, approved database operation outcomes,
self-cognition mirrors, model contract drift, and resource health. Documents
store IDs, refs, counts, statuses, timestamps, components, labels, and
sanitized warning/error metadata. They must not store prompt text, model
answers, message bodies, generated dialog, base64 media, vector arrays,
secrets, callback credentials, raw channel ids, raw documents, or raw adapter
responses.

### `event_log_snapshots`

Append-only deterministic aggregate snapshots for later operator or approved
agent review. Snapshot documents contain bounded source counts, semantic
descriptors, findings, and source event refs. They are generated without LLM
calls and must remain prompt-safe.

Retention and archival for event-log collections are intentionally deferred.
Any pruning, compaction, or archival policy requires a separate approved plan.

### Deprecated Or Removed Collections

`user_profile_memories` is no longer created by bootstrap. Cognition-facing
user memory belongs in `user_memory_units`.

Legacy RAG collections and other deprecated collections may be inspected or
dropped only through maintenance helpers and explicit operator scripts.

## Bootstrap And Indexes

`db_bootstrap()` owns startup collection/index preparation and required seed
documents. It creates current collections and indexes, including
`user_memory_units`, reflection-run indexes, interaction-style indexes,
scheduled-event indexes, self-cognition action-attempt indexes, and other
runtime indexes required by the facade.

Bootstrap is idempotent. It creates or updates indexes and seed singleton
documents. Destructive data repair belongs in explicit migration plans.

## Reflection Interface

Reflection is a first-class database consumer, but it remains outside the live
chat response path.

The database package provides reflection support through named helpers:

- monitored-channel discovery and scope message reads from
  `conversation_history`;
- single-private-scope user resolution;
- reflection-run index creation;
- reflection-run upsert and lookup;
- hourly/daily run listing by channel and date;
- interaction-style overlay persistence for promoted runtime guidance.

Reflection packages call the database facade or the named reflection DB helpers
that are re-exported by the facade.

Raw reflection output is stored for audit and later promotion decisions. Normal
cognition may consume only promoted, gated context through the reflection and
memory interfaces.

## Semantic Ownership

LLM stages own semantic judgment:

- what user memory means;
- whether a relationship insight should change;
- what reflection evidence implies;
- what interaction style guidance should say;
- which future promise exists before dispatcher scheduling.

Deterministic database code owns structure and mechanics:

- schema shape and field limits;
- collection selection;
- indexes and projections;
- query and update expressions;
- embedding source construction;
- id construction and uniqueness;
- lifecycle/status transitions;
- exception translation;
- bootstrap and cache invalidation mechanics.

Database helpers keep user-text handling structural. Semantic decisions belong
in upstream prompts, schemas, or caller contracts.

## Evolution Paths

Adding a new database operation requires one of these paths:

1. Add or extend a named helper in the appropriate `db` submodule and re-export
   it from `kazusa_ai_chatbot.db` when runtime code needs it.
2. Add or extend a named helper in `db.script_operations` when only maintenance
   scripts need it.
3. Add a new collection contract here when the operation introduces a durable
   collection, new document role, or new ownership boundary.

Changes that alter durable document shape, index requirements, write
semantics, reflection promotion behavior, or production data interpretation
require an explicit development plan. Completed plans are historical records;
new scope should use a new or superseding plan.

Data migrations apply when stored production documents change shape or meaning.

## Public Imports

Outside `src/kazusa_ai_chatbot/db/` callers use:

- `from kazusa_ai_chatbot.db import ...`
- `from kazusa_ai_chatbot.db import script_operations` for maintenance scripts
- `from kazusa_ai_chatbot.db.script_operations import ...` for maintenance
  scripts
- `from kazusa_ai_chatbot import event_logging` for runtime event capture;
  runtime callers must not import `kazusa_ai_chatbot.db.event_logging`
  directly.
- DB TypedDicts and `DatabaseOperationError` through the public facade.

Inside `src/kazusa_ai_chatbot/db/`, package internals use backend access,
query/update details, backend exception handling, and private
collection-specific helpers.
