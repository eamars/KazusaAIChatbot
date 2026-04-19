
### Stage 2: Database Layer (After Stage 1 foundation + Stage 1.5a cache types) ✅ CODE COMPLETE (tests pending)

**Depends on**: Stage 1a + 1b + 1c + 1.5a

**Critical Gate**: Stage 2a-ii (schema migration) must complete and be manually validated before Stage 4a deployment.

**Stage 2 Status (2026-04-19)**:
- ✅ `src/kazusa_ai_chatbot/db.py` deleted, replaced with `db/` package
- ✅ `src/kazusa_ai_chatbot/db/__init__.py` — full backward-compat re-exports
- ✅ `src/kazusa_ai_chatbot/db/_client.py` — connection, embedding, `enable_vector_index` (now accepts `path=` for multi-embedding collections)
- ✅ `src/kazusa_ai_chatbot/db/schemas.py` — TypedDicts including new `CharacterDiaryEntry`, `ObjectiveFactEntry`, `RagCacheIndexDoc`, `RagMetadataIndexDoc`; `UserProfileDoc` extended with `character_diary`, `diary_embedding`, `diary_updated_at`, `objective_facts`, `facts_embedding`, `facts_updated_at`
- ✅ `src/kazusa_ai_chatbot/db/bootstrap.py` — adds `rag_cache_index`, `rag_metadata_index` collections, creates TTL + lookup + vector indices, runs legacy `facts` → diary/facts heuristic migration
- ✅ `src/kazusa_ai_chatbot/db/conversation.py` — unchanged behaviour, moved
- ✅ `src/kazusa_ai_chatbot/db/users.py` — identity/profile/affinity preserved; new `get_character_diary`, `upsert_character_diary`, `get_objective_facts`, `upsert_objective_facts`; legacy `get_user_facts`/`upsert_user_facts`/`overwrite_user_facts` retained as deprecated shims (now reads from new fields first)
- ✅ `src/kazusa_ai_chatbot/db/character.py` — unchanged behaviour, moved
- ✅ `src/kazusa_ai_chatbot/db/memory.py` — unchanged behaviour, moved
- ✅ `src/kazusa_ai_chatbot/db/rag_cache.py` — `insert_cache_entry`, `find_cache_entries`, `soft_delete_cache_entries`, `clear_all_cache_for_user`, `get_rag_version`, `increment_rag_version`
- ⬜ `tests/test_db.py` — mock patches still reference the pre-split module path; needs update to patch new submodule functions (deferred)
- ⬜ Manual migration validation against production MongoDB (runs automatically on next `db_bootstrap()`; gate for Stage 4a)

#### 2a. Restructure `kazusa_ai_chatbot/db.py` → `kazusa_ai_chatbot/db/` (SPLIT INTO SUBMODULES)
**Purpose**: `db.py` is too large. Split by responsibility into a `db/` package. Backward compatibility is preserved — all existing `from kazusa_ai_chatbot.db import ...` imports continue to work via `__init__.py` re-exports.

**New folder structure**:
```
src/kazusa_ai_chatbot/db/
    __init__.py         ← re-exports everything (backward compat)
    _client.py          ← connection + embedding (shared by all submodules)
    schemas.py          ← all TypedDict document schemas
    bootstrap.py        ← db_bootstrap() startup logic
    conversation.py     ← conversation_history collection
    users.py            ← user_profiles collection (identity + profile + facts + affinity)
    character.py        ← character_state collection
    memory.py           ← memory collection
    rag_cache.py        ← NEW: rag_cache_index + rag_metadata_index collections
```

**Submodule breakdown** (what moves where from current `db.py`):

`db/_client.py` — Connection + shared utilities (imported by every other submodule):
- `_get_embed_client()`, `get_text_embedding()`
- `get_db()`, `close_db()`
- `enable_vector_index()`

`db/schemas.py` — All TypedDict document schemas (no logic):
- `AttachmentDoc`
- `ConversationMessageDoc`, `PlatformAccountDoc`
- `UserProfileDoc`
- `CharacterProfileDoc`
- `MemoryDoc`, `build_memory_doc()`
- `ScheduledEventDoc`

`db/bootstrap.py` — Startup and index creation:
- `db_bootstrap()` — creates collections, seeds character_state, creates all indices

`db/conversation.py` — `conversation_history` collection:
- `get_conversation_history()`
- `search_conversation_history()`
- `save_conversation()`

`db/users.py` — `user_profiles` collection (identity resolution, profile, facts, affinity):
- `resolve_global_user_id()`, `link_platform_account()`, `add_suspected_alias()`
- `get_user_profile()`, `create_user_profile()`
- `get_user_facts()`, `upsert_user_facts()`, `overwrite_user_facts()`
- `get_affinity()`, `update_affinity()`, `update_last_relationship_insight()`
- `enable_user_facts_vector_index()`, `search_users_by_facts()`

`db/character.py` — `character_state` collection:
- `get_character_profile()`, `save_character_profile()`
- `get_character_state()`, `upsert_character_state()`

`db/memory.py` — `memory` collection:
- `enable_memory_vector_index()`
- `save_memory()`, `search_memory()`

`db/rag_cache.py` — NEW: `rag_cache_index` + `rag_metadata_index` collections:

*Collections and schemas*:
```python
# rag_cache_index document
{
    "cache_id": str,             # UUID4
    "cache_type": str,           # "user_facts" | "internal_memory"
    "global_user_id": str,       # Owner (for scoped invalidation)
    "embedding": list[float],    # Query embedding that produced these results
    "results": dict,             # Cached RAG results payload
    "ttl_expires_at": datetime,  # TTL — MongoDB TTL index auto-deletes after this
    "created_at": str,           # ISO-8601 UTC
    "deleted": bool,             # Soft-delete flag (set by invalidate before TTL fires)
}

# rag_metadata_index document (one doc per global_user_id)
{
    "global_user_id": str,       # UUID4 — unique key
    "rag_version": int,          # Incremented on every successful DB write (cache bust signal)
    "last_rag_run": str,         # ISO-8601 UTC of last RAG execution
}
```

*Indices on `rag_cache_index`*:
- `{ "ttl_expires_at": 1 }` with `expireAfterSeconds=0` — auto-deletion
- `{ "cache_type": 1, "global_user_id": 1, "deleted": 1 }` — scoped invalidation queries
- Vector search index on `embedding` (cosine, same dim as other collections)

*Indices on `rag_metadata_index`*:
- `{ "global_user_id": 1 }` unique

*New functions in `db/rag_cache.py`*:
- [x] `async insert_cache_entry(cache_type, global_user_id, embedding, results, ttl_seconds)` → `str` (cache_id)
- [x] `async find_cache_entries(cache_type, global_user_id)` → `list[dict]` (non-expired, non-deleted, with embeddings)
- [x] `async soft_delete_cache_entries(cache_type, global_user_id)` — sets `deleted=True`
- [x] `async clear_all_cache_for_user(global_user_id)` — soft-deletes all cache_types for user
- [x] `async get_rag_version(global_user_id)` → `int` (0 if not found)
- [x] `async increment_rag_version(global_user_id)` — upserts, increments by 1

`db/__init__.py` — Re-export all public symbols for backward compatibility:
```python
from kazusa_ai_chatbot.db._client import get_db, close_db, get_text_embedding, enable_vector_index
from kazusa_ai_chatbot.db.schemas import (
    AttachmentDoc, ConversationMessageDoc, PlatformAccountDoc,
    UserProfileDoc, CharacterProfileDoc, MemoryDoc, ScheduledEventDoc, build_memory_doc
)
from kazusa_ai_chatbot.db.bootstrap import db_bootstrap
from kazusa_ai_chatbot.db.conversation import get_conversation_history, search_conversation_history, save_conversation
from kazusa_ai_chatbot.db.users import (
    resolve_global_user_id, link_platform_account, add_suspected_alias,
    get_user_profile, create_user_profile, get_user_facts, upsert_user_facts,
    overwrite_user_facts, get_affinity, update_affinity,
    update_last_relationship_insight, enable_user_facts_vector_index, search_users_by_facts
)
from kazusa_ai_chatbot.db.character import (
    get_character_profile, save_character_profile, get_character_state, upsert_character_state
)
from kazusa_ai_chatbot.db.memory import enable_memory_vector_index, save_memory, search_memory
from kazusa_ai_chatbot.db.rag_cache import (
    insert_cache_entry, find_cache_entries, soft_delete_cache_entries,
    clear_all_cache_for_user, get_rag_version, increment_rag_version
)
```

**Dependencies**: None (all internal)
**Lines of code**: ~100 lines reorganised + ~150 new lines in `rag_cache.py`
**No test_main()** (covered by integration tests and used by cache.py test_main)

---

#### 2a-ii. Migrate `user_profiles` Schema — Separate Character Diary from Objective Facts
**Purpose**: Split the ambiguous `facts` array into semantically distinct fields to enable precise cache invalidation and better data semantics.

**Current Schema** (from MongoDB inspection):
```python
# user_profiles document (CURRENT)
{
    "_id": ObjectId,
    "global_user_id": str,
    "platform_accounts": [
        {"platform": str, "platform_user_id": str, "display_name": str, "linked_at": str}
    ],
    "suspected_aliases": [str],
    "facts": [str],                          # ❌ AMBIGUOUS - mixes diary + facts
    "affinity": int,
    "last_relationship_insight": str,
    "embedding": [float]                     # ❌ Embedded of what? (unclear)
}
```

**Problem**: 
- Single `facts` array conflates character's diary entries with objective user facts
- Single `embedding` is ambiguous (embedded of entire facts? Which type?)
- No way to track the *source* or *type* of each fact (subjective vs objective)
- Cache invalidation is all-or-nothing (can't selectively clear diary vs facts)

**New Schema** (PROPOSED):
```python
# user_profiles document (NEW)
{
    "_id": ObjectId,
    "global_user_id": str,
    
    # Identity/Access
    "platform_accounts": [
        {"platform": str, "platform_user_id": str, "display_name": str, "linked_at": str}
    ],
    "suspected_aliases": [str],
    
    # Character's subjective observations (diary)
    "character_diary": [
        {
            "entry": str,                        # "User is interesting", "User makes me smile"
            "timestamp": datetime,               # When this observation was made
            "confidence": float,                 # 0.0-1.0 how confident character is
            "context": str,                      # Brief context (e.g., "from conversation about hobbies")
        }
    ],
    "diary_embedding": [float],                  # Embedding of ALL diary entries combined
    "diary_updated_at": datetime,                # Last time diary was updated
    
    # Objective facts about user (verified)
    "objective_facts": [
        {
            "fact": str,                         # "User is engineer", "User lives in Tokyo"
            "category": str,                     # "occupation", "location", "hobby", "relationship"
            "timestamp": datetime,               # When this fact was learned
            "source": str,                       # "user_stated" | "inferred" | "verified"
            "confidence": float,                 # 0.0-1.0 confidence level
        }
    ],
    "facts_embedding": [float],                  # Embedding of ALL objective facts combined
    "facts_updated_at": datetime,                # Last time facts were updated
    
    # Relationship metrics
    "affinity": int,                             # 0-1000 relationship score
    "last_relationship_insight": str,            # Summary of character's current view of user
    "affinity_history": [
        {
            "delta": int,
            "timestamp": datetime,
            "reason": str                        # Why affinity changed (optional)
        }
    ],
}
```

**Migration Strategy**:

1. **Phase 1: Add new fields alongside old ones** (backward compatibility)
   ```python
   # At bootstrap time, if old schema detected:
   if "facts" in doc and "character_diary" not in doc:
       # Transform old flat array into structured new format
       char_diary = []
       obj_facts = []
       
       for fact in doc["facts"]:
           # Heuristic: if contains "I think", "I feel", "seems to me" → diary
           if any(phrase in fact.lower() for phrase in ["think", "feel", "seems", "feels like"]):
               char_diary.append({
                   "entry": fact,
                   "timestamp": doc.get("facts_updated_at", datetime.now()),
                   "confidence": 0.7,  # Default medium confidence for migrated data
                   "context": ""
               })
           else:
               obj_facts.append({
                   "fact": fact,
                   "category": "general",
                   "timestamp": doc.get("facts_updated_at", datetime.now()),
                   "source": "user_stated",
                   "confidence": 0.8  # Default high confidence for migrated data
               })
       
       # Write new structure back
       await db.user_profiles.update_one(
           {"_id": doc["_id"]},
           {"$set": {
               "character_diary": char_diary,
               "diary_embedding": embed_diary_entries(char_diary),
               "diary_updated_at": datetime.now(),
               "objective_facts": obj_facts,
               "facts_embedding": embed_fact_entries(obj_facts),
               "facts_updated_at": datetime.now(),
           }}
       )
   ```

2. **Phase 2: Update functions in `db/users.py`** to work with new schema
   ```python
   # Replace old get_user_facts() with semantic equivalents:
   
   async def get_character_diary(global_user_id: str) -> list[dict]:
       """Retrieve character's subjective observations about user."""
       db = await get_db()
       doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
       return doc.get("character_diary", []) if doc else []
   
   async def get_objective_facts(global_user_id: str) -> list[dict]:
       """Retrieve verified facts about user."""
       db = await get_db()
       doc = await db.user_profiles.find_one({"global_user_id": global_user_id})
       return doc.get("objective_facts", []) if doc else []
   
   # Backward compat shim (for now):
   async def get_user_facts(global_user_id: str) -> list[str]:
       """DEPRECATED. Get all facts (diary + objective) as flat list for backward compat."""
       diary = await get_character_diary(global_user_id)
       facts = await get_objective_facts(global_user_id)
       return [d["entry"] for d in diary] + [f["fact"] for f in facts]
   
   async def upsert_character_diary(global_user_id: str, new_entries: list[dict]) -> None:
       """Add new diary entries, recompute embedding."""
       db = await get_db()
       existing = await get_character_diary(global_user_id)
       merged = existing + new_entries  # Preserve chronological order
       
       # Combine all entries for embedding
       diary_text = "\n".join([e["entry"] for e in merged])
       diary_embedding = await get_text_embedding(diary_text)
       
       await db.user_profiles.update_one(
           {"global_user_id": global_user_id},
           {"$set": {
               "global_user_id": global_user_id,
               "character_diary": merged,
               "diary_embedding": diary_embedding,
               "diary_updated_at": datetime.now(timezone.utc)
           }},
           upsert=True
       )
   
   async def upsert_objective_facts(global_user_id: str, new_facts: list[dict]) -> None:
       """Add new objective facts, recompute embedding, deduplicate."""
       db = await get_db()
       existing = await get_objective_facts(global_user_id)
       
       # Merge and deduplicate by fact text (case-insensitive)
       fact_texts = {f["fact"].lower(): f for f in existing}
       for nf in new_facts:
           fact_texts[nf["fact"].lower()] = nf  # New overwrites if duplicate
       
       merged = list(fact_texts.values())
       
       # Combine all facts for embedding
       facts_text = "\n".join([f["fact"] for f in merged])
       facts_embedding = await get_text_embedding(facts_text)
       
       await db.user_profiles.update_one(
           {"global_user_id": global_user_id},
           {"$set": {
               "global_user_id": global_user_id,
               "objective_facts": merged,
               "facts_embedding": facts_embedding,
               "facts_updated_at": datetime.now(timezone.utc)
           }},
           upsert=True
       )
   ```

3. **Phase 3: Update cache functions in `db/rag_cache.py`** to align with new types
   ```python
   # In rag_cache.py, update cache_type definitions in Bootstrap:
   DEFAULT_CACHE_TYPES = {
       "character_diary": {
           "ttl_seconds": 1800,           # 30 min
           "embedding_field": "diary_embedding",
       },
       "objective_user_facts": {
           "ttl_seconds": 3600,           # 60 min
           "embedding_field": "facts_embedding",
       },
       "user_promises": {
           "ttl_seconds": 900,            # 15 min (from memory collection)
           "embedding_field": None,       # Promises embed individually, not in user_profiles
       },
       "internal_memory": {
           "ttl_seconds": 900,            # 15 min
           "embedding_field": None,       # Uses memory.embedding
       }
   }
   ```

**Indices needed for new schema**:
- `{ "global_user_id": 1 }` unique on user_profiles (existing, unchanged)
- `{ "diary_embedding": "2dsphere" }` for diary semantic search (NEW)
- `{ "facts_embedding": "2dsphere" }` for facts semantic search (NEW)
- `{ "character_diary.timestamp": -1 }` for recency queries (NEW)
- `{ "objective_facts.timestamp": -1 }` for recency queries (NEW)
- `{ "objective_facts.category": 1 }` for category-based filtering (NEW)

**Consolidator Changes** (in Stage 4a):
```python
# Step 2a: Update diary (was "upsert_user_facts")
diary_entry = state.get("diary_entry", [])
if global_user_id and diary_entry:
    entries_with_metadata = [
        {
            "entry": de,
            "timestamp": datetime.now(timezone.utc),
            "confidence": 0.85,  # From evaluator validation
            "context": state.get("user_topic", "")
        }
        for de in diary_entry
    ]
    await upsert_character_diary(global_user_id, entries_with_metadata)

# Step 3: Record facts (NEW - was mixed with diary)
new_facts = state.get("new_facts", [])
if new_facts:
    facts_with_metadata = [
        {
            "fact": fact.get("description", ""),
            "category": fact.get("category", "general"),
            "timestamp": datetime.now(timezone.utc),
            "source": "conversation_extracted",
            "confidence": 0.9  # High confidence from evaluator
        }
        for fact in new_facts
    ]
    await upsert_objective_facts(global_user_id, facts_with_metadata)
```

**Cache Invalidation Precision** (in Stage 4a):
```python
# When new diary entry written:
await cache.invalidate_pattern("character_diary", global_user_id)
# → Only clears character_diary cache entries
# → Leaves objective_user_facts cache intact

# When new facts written:
await cache.invalidate_pattern("objective_user_facts", global_user_id)
# → Only clears objective_user_facts cache entries
# → Leaves character_diary cache intact
```

**Backward Compatibility**:
- Old `get_user_facts()` function kept as shim
- Old code continues to work (returns flattened list)
- New code uses specific `get_character_diary()` and `get_objective_facts()`
- Migration is automatic on first access to new schema

**Dependencies**: Stage 1e (cache type separation design)
**Lines of code**: ~400 new lines in `db/users.py` (new functions) + ~100 in migration logic
**Database impact**: Additive (new fields) + Indices (new vector indices) + Data transformation (one-time)

---

#### 2b. Update `db/bootstrap.py` — add new collections to startup
- [x] Add `rag_cache_index` and `rag_metadata_index` to `db_bootstrap()` required collections list
- [x] Add TTL + vector indices for `rag_cache_index` to bootstrap
- [x] Add unique index for `rag_metadata_index.global_user_id` to bootstrap
- [x] Create vector indices on user_profiles: `diary_embedding` and `facts_embedding` (NEW)
- [x] Create regular indices on user_profiles: `character_diary.timestamp`, `objective_facts.timestamp`, `objective_facts.category` (NEW)
