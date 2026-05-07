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

This ICD does not cover:

- MongoDB deployment, credentials, Atlas configuration, or environment values.
- Platform adapter transport storage outside persisted conversation rows.
- LLM prompt semantics except where a DB helper stores or retrieves prompt
  input/output documents.
- One-off production data repair procedures. Those belong in explicit scripts
  and plans, and still must use the public maintenance interface.

## Parties

### Runtime Callers

Runtime callers are production packages on the live or background service path:
brain service, RAG, cognition, dialog persistence, consolidation, scheduler,
dispatcher, reflection cycle, conversation progress, and memory evolution.

Runtime callers MUST import from the `kazusa_ai_chatbot.db` facade unless this
ICD names a narrower public package boundary for that subsystem.

### Maintenance Scripts

Maintenance scripts are operator tools under `src/scripts`. They may need
export, migration, backfill, scan, or repair operations that are not part of
the normal runtime API.

Maintenance scripts MUST use `kazusa_ai_chatbot.db.script_operations` for those
operator-only operations. They MUST NOT open raw database handles, import
Motor/PyMongo, or embed direct collection method calls.

### Database Package Internals

The database package owns:

- MongoDB client lifecycle.
- Collection names.
- Collection handles.
- Index creation.
- Query, projection, aggregation, sort, and update documents.
- Vector-search and embedding persistence mechanics.
- Translation from backend failures into application-level errors.

Only modules under `src/kazusa_ai_chatbot/db/` may import Motor, PyMongo, or
their submodules.

## Normative Language

The words `MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative:

- `MUST`: required for a compatible database interface.
- `MUST NOT`: forbidden by the interface.
- `SHOULD`: expected unless a documented local reason exists.
- `MAY`: allowed extension behavior.

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

Raw database access terminates inside the database package. No runtime package
or script should receive a database object, collection object, cursor, Motor
type, PyMongo type, or backend exception as part of its normal contract.

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

Callers MUST treat facade helpers as semantic operations. A caller should ask
for a new helper when it needs a new storage behavior instead of importing a
db submodule to perform backend-shaped work.

## Public Maintenance Interface

Maintenance-only functionality lives in:

```python
from kazusa_ai_chatbot.db import script_operations
```

or by importing named helpers from:

```python
from kazusa_ai_chatbot.db.script_operations import export_collection_rows
```

This submodule is public for operator tools, but not for runtime packages.
It may expose broader export, migration, scan, and repair helpers than the
runtime facade. Even there, scripts should pass semantic parameters when a
helper exists instead of constructing MongoDB query or update documents.

If a script needs a raw collection name, raw query operator, projection,
aggregation pipeline, or update operator, that detail SHOULD move into
`script_operations` behind a named helper. The script's job is orchestration,
argument parsing, dry-run policy, file IO, and operator reporting; the database
package owns backend expression.

## Private Client Boundary

`kazusa_ai_chatbot.db._client.get_db()` is private to
`src/kazusa_ai_chatbot/db/`.

Forbidden outside `src/kazusa_ai_chatbot/db/`:

- importing `kazusa_ai_chatbot.db._client`;
- importing `motor`, `pymongo`, or any submodule such as `pymongo.errors`;
- holding database, collection, cursor, or backend client objects;
- calling collection methods such as `find`, `aggregate`, `insert_one`,
  `update_one`, `replace_one`, `delete_many`, `create_index`, or `drop`;
- catching `pymongo.errors.*` or `motor.*` exceptions directly;
- constructing raw MongoDB query/update/projection/aggregation details in
  runtime modules.

Allowed inside `src/kazusa_ai_chatbot/db/`:

- importing `db._client.get_db()`;
- using Motor/PyMongo APIs;
- constructing backend query/update documents;
- catching backend exceptions and re-raising `DatabaseOperationError` when the
  error crosses the public interface.

## Error Contract

Public database helpers SHOULD raise application-level exceptions when a
backend operation cannot complete. The package-level exception is:

```python
from kazusa_ai_chatbot.db import DatabaseOperationError
```

Callers may catch `DatabaseOperationError` when they can degrade gracefully,
record telemetry, or return a rejected operation. Callers MUST NOT catch
PyMongo or Motor exceptions outside the database package.

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

Semantic search MUST use `body_text` plus attachment descriptions as the
embedding source. `raw_wire_text` is audit/replay data and must not become the
semantic search source.

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
material. Character state updates are explicit persistence events; reflection
must not mutate this collection unless a future ICD or plan creates a named
promotion path.

### `memory`

Stores curated shared/world/common-sense memory and evolving-memory rows used
outside per-user memory. Legacy compatibility helpers remain available through
the facade, but new reflection memory promotion should go through the
`memory_evolution` public API rather than raw legacy writes.

### `character_reflection_runs`

Stores hourly, daily-channel, and global-promotion reflection run documents.
These rows are evidence and audit records for the reflection cycle. Raw
reflection outputs MUST NOT enter normal cognition directly.

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
through named helper functions. Callers must not update scheduler status fields
directly.

### `conversation_episode_state`

Stores short-lived conversation-progress state keyed by platform, channel, and
user. The collection is operational working memory, not durable identity memory.
Writes should be guarded so stale background records cannot replace newer
episode progress.

### `rag_cache2_persistent`

Stores persistent initializer cache entries for RAG Cache2. The collection is
owned by cache helpers that build version keys, load entries, record hits, and
prune stale data.

### Deprecated Or Removed Collections

`user_profile_memories` is no longer created by bootstrap. Cognition-facing
user memory belongs in `user_memory_units`.

Legacy RAG collections and other deprecated collections may be inspected or
dropped only through maintenance helpers and explicit operator scripts.

## Bootstrap And Indexes

`db_bootstrap()` owns startup collection/index preparation and required seed
documents. It creates current collections and indexes, including
`user_memory_units`, reflection-run indexes, interaction-style indexes,
scheduled-event indexes, and other runtime indexes required by the facade.

Bootstrap MUST be idempotent. It may create or update indexes and seed
singleton documents, but it must not perform destructive data repair unless an
explicit migration plan says so.

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

Reflection packages MUST NOT open database clients, construct MongoDB queries,
or import database internals. They call the database facade or the named
reflection DB helpers that are re-exported by the facade.

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

Do not add code-side keyword classifiers over user text inside database helpers
to override LLM meaning. If a semantic decision is wrong, fix the upstream
prompt, schema, or caller contract.

## Versioning And Change Control

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

Data migrations are only needed when stored production documents must change
shape or meaning. Interface-collapse work that only removes raw `get_db()`
access should not mutate production content.

## Verification Checklist

Before merging database-boundary changes, verify:

- runtime code imports database functionality from `kazusa_ai_chatbot.db`;
- maintenance scripts import operator-only helpers from
  `kazusa_ai_chatbot.db.script_operations`;
- no non-db module imports `motor`, `pymongo`, or any of their submodules;
- no non-db module imports `kazusa_ai_chatbot.db._client`;
- no non-db module catches backend database exceptions;
- no runtime module contains raw MongoDB collection calls or query/update
  details;
- scripts contain only orchestration logic unless a documented maintenance
  exception exists;
- public DB helpers translate backend failures to `DatabaseOperationError`
  where callers can observe them;
- bootstrap remains idempotent;
- focused tests cover any changed helper, collection shape, index behavior, or
  boundary rule.

Relevant boundary tests include:

- `tests/test_db_public_boundary.py`
- `tests/test_script_db_boundary.py`

Relevant subsystem tests should be selected based on the helper being changed:
conversation, RAG, scheduler, dispatcher, reflection, memory evolution,
conversation progress, or maintenance-script tests.

## Import Rules

Allowed outside `src/kazusa_ai_chatbot/db/`:

- `from kazusa_ai_chatbot.db import ...`
- `from kazusa_ai_chatbot.db import script_operations` for maintenance scripts
- `from kazusa_ai_chatbot.db.script_operations import ...` for maintenance
  scripts
- importing DB TypedDicts and `DatabaseOperationError` through the public
  facade

Forbidden outside `src/kazusa_ai_chatbot/db/`:

- `from kazusa_ai_chatbot.db._client import get_db`
- `import kazusa_ai_chatbot.db._client`
- `import motor` or `from motor...`
- `import pymongo` or `from pymongo...`
- direct collection or cursor operations
- backend-shaped query, projection, aggregation, and update construction in
  runtime packages

Allowed inside `src/kazusa_ai_chatbot/db/`:

- raw database access through `_client.get_db()`;
- backend query/update details;
- backend exception handling;
- collection-specific implementation helpers that remain private to the
  package.
