# Database Layer

The durable user-memory substrate is now `user_memory_units`.

`user_profiles` is only the identity and relationship header: linked platform
accounts, suspected aliases, affinity, and `last_relationship_insight`.
Cognition-facing user memory is not embedded in `user_profiles`.

## Collections

- `conversation_history`: raw message history and embeddings.
- `user_profiles`: user identity, affinity, and lightweight relationship header.
- `user_memory_units`: unified user memory records.
- `character_state`: singleton character profile and runtime self-image.
- `memory`: curated shared/world memory.
- `scheduled_events`: future tool/message events.
- `conversation_episode_state`: short-lived episode progress.

## User Memory Units

Each `user_memory_units` document stores the same semantic triple:

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
merge/evolve/create decisions, rewriting, and stability classification. Local
code validates structure and known IDs, but does not reinterpret user meaning.

## Bootstrap

`db_bootstrap()` creates `user_memory_units` and its indexes. The old
`user_profile_memories` collection is no longer created.

## Public Boundary

`db._client.get_db()` is private to the database package. Runtime modules and
operator scripts must not import it or hold raw MongoDB database handles. Add a
semantic helper under `kazusa_ai_chatbot.db` when a caller needs a new database
operation.
