---
name: reset-kazusa-affect
description: Use when the user explicitly asks to reset Kazusa's current emotional state or affect to a neutral Chinese baseline while keeping personality, relationships, memories, conversation history, and other durable state intact. This skill clears all conversation-progress records and writes only the authoritative affect fields.
---

# Reset Kazusa Affect

## Purpose

Perform a bounded, deterministic reset of Kazusa's current affect. The reset
targets the singleton `character_state` runtime affect fields and clears the
short-term `conversation_episode_state` lane. It preserves conversation
history and every other state lane.

The skill is for an explicit operator request. It is not a memory purge,
relationship reset, personality edit, boundary edit, or conversation-history
operation.

## Mutation boundary

The complete write allowlist is:

1. `character_state/_id: "global"`
   - `mood`
   - `global_vibe`
   - `reflection_summary`
   - `updated_at`
2. `conversation_episode_state`
   - clear every document after the preflight ownership check
3. Process-local caches owned by conversation progress and the running brain
   - clear or refresh them through their existing runtime boundary

The reset issues zero write calls to `conversation_history`; history may only
be counted or hashed for verification.

The affect radius is deliberately narrow. Preserve `self_image`,
`cognition_state`, `personality_brief`, `boundary_profile`,
`linguistic_texture_profile`, `user_profiles`, `user_memory_units`, `memory`,
`internal_monologue_residue_state`, `interaction_style_images`, global growth,
reflection runs, schedulers, action ledgers, RAG caches, LLM traces, event
logs, and `conversation_history`.

If the requested result requires changing any preserved lane, stop and route it
to the owning workflow instead of widening this reset.

## Neutral baseline

Write these exact Chinese values unless the user supplies a more specific
Chinese neutral baseline:

```text
mood: 平静且中性
global_vibe: 平稳、放松、保持清醒
reflection_summary: 当前没有需要延续的情绪余波；后续根据新的输入和当前证据重新判断。
```

Operational enum values such as MongoDB statuses remain the schema's existing
values. Human-facing affect text written into `character_state` stays in
Chinese. Do not use English mood labels, invented sexualized states, or
persona-changing text.

## Workflow

### 1. Preflight

Run from the repository root and use `venv\Scripts\python.exe` for project
Python commands. Read the current state through the repository DB facade:

- `get_character_runtime_state()` for the singleton runtime projection;
- the conversation-progress owner for the count and ownership of
  `conversation_episode_state` rows;
- the process-local conversation-progress cache status when available.

Confirm all of the following before any write:

- `character_state/_id: "global"` exists;
- `updated_at` is present and is used as the compare-and-swap freshness token;
- `conversation_episode_state` is the singleton Kazusa-owned collection in this
  deployment. If multiple character owners appear, stop because the current
  schema has no safe character filter;
- the requested operation is global affect reset plus progress clearing;
- the planned write set contains only the mutation allowlist above.

Record counts and hashes of protected projections for verification. Do not
export a backup unless the user requests one.

### 2. Dry run

Show a compact plan containing:

- current affect field names and character counts, without dumping private
  memory text;
- the number of conversation-progress documents to clear;
- the exact Chinese baseline;
- the protected collections and fields that will remain unchanged.

Use dry-run as the default. Apply only after the user has explicitly requested
the mutation in the current task.

### 3. Apply the reset

Quiesce reflection, consolidation, self-cognition, and conversation-progress
writers for the short operation when the deployment supports it. This prevents
a concurrent writer from immediately restoring the old state.

Update the three affect fields with the existing compare-and-swap helper:

```python
await compare_and_upsert_character_state(
    expected_updated_at=current_runtime["updated_at"],
    mood="平静且中性",
    global_vibe="平稳、放松、保持清醒",
    reflection_summary=(
        "当前没有需要延续的情绪余波；后续根据新的输入和当前证据重新判断。"
    ),
    updated_at_utc=storage_utc_now_iso(),
)
```

Require a successful freshness match. On a stale result, reread the runtime
state and present a new dry-run; never overwrite a newer writer blindly.

Clear all `conversation_episode_state` documents through the existing DB-owned
maintenance boundary. The current public progress facade exposes load and
record operations but no global clear operation, so load the row identifiers
with `load_lane_cleanup_rows(...)` and delete each preflighted row with
`delete_lane_cleanup_row(...)`, always passing the literal collection name
`conversation_episode_state`. This keeps the operation reviewable and avoids
a wildcard delete path.

After the per-row deletes, verify the collection count is zero. Clear the
corresponding process-local progress cache and refresh the running service's
character runtime projection through its existing service boundary. A
standalone maintenance process cannot mutate another process's memory; report
the required service refresh or restart when that boundary is unavailable.

### 4. Verify

Reread and verify:

- the three affect fields exactly equal the Chinese baseline;
- `conversation_episode_state` has zero documents;
- protected `character_state` fields have the same preflight hash;
- `conversation_history` was not written and its preflight/postflight counts
  are reported;
- no other collection was written;
- the process-local runtime and progress caches have been refreshed when the
  service boundary was available.

Report partial completion explicitly if the affect write succeeds but progress
clearing or runtime refresh fails. Keep the report to collection names,
counts, field names, and operation status; omit private text and credentials.

## Important behavior boundary

This is a bounded affect reset, not amnesia. Preserved self-image, private
residue, relationship context, durable memory, and old dialog evidence may
still influence a future response when current retrieval or input makes them
relevant. Keep the reset narrow and report those retained sources rather than
silently deleting them.

The daily sleep-affect settling flow is not a substitute for this skill. It is
an LLM-mediated, local-date-idempotent reflection process and may preserve an
existing defensive state.

## Repository references

Use these ownership contracts when checking an implementation:

- `src/kazusa_ai_chatbot/db/character.py` — singleton runtime state and
  compare-and-swap update;
- `src/kazusa_ai_chatbot/conversation_progress/README.md` — short-term
  progress ownership and lifecycle;
- `src/kazusa_ai_chatbot/conversation_progress/cache.py` — process-local
  progress cache;
- `src/kazusa_ai_chatbot/db/script_operations.py` — reviewed per-row
  maintenance boundary for clearing progress documents;
- `src/kazusa_ai_chatbot/db/README.md` — collection boundaries;
- `src/kazusa_ai_chatbot/reflection_cycle/affect_settling.py` — daily settling
  behavior and idempotency.
