# Internal Monologue Residue ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.internal_monologue_residue`
- Runtime owner: brain service post-turn and self-cognition background paths
- Storage owner: `kazusa_ai_chatbot.db` facade for
  `internal_monologue_residue_state`
- Prompt consumer: L2a consciousness only, via
  `internal_monologue_residue_context`
- Non-goals: durable memory, visible dialog planning, adapter delivery,
  scheduler input, and raw reflection carry-over

## Ownership Boundary

`internal_monologue_residue` owns the short-lived private residue lane between
completed cognition episodes and the next L2a cognition pass. It preserves a
compact first-person reason for why the character may still feel, expect,
defend, hesitate, or carry tension after the last episode.

This package does not own visible dialog, action planning, durable memory,
reflection promotion, scheduler behavior, adapter delivery, or raw chat
history. Those systems may produce source state or consume the projected L2a
context only through the public facade.

## Public Facade

Runtime callers use:

- `load_residue_context(trigger_scope, current_timestamp_utc,
  record_telemetry=True)` to return one bounded prompt-facing string plus
  sanitized load metadata. Read-only inspection callers may set
  `record_telemetry=False` so the inspection does not create an event row.
- `record_completed_episode_residue(completed_state, current_timestamp_utc)` to
  record one post-episode residue row or skip cleanly.
- `project_residue_window(rows, current_timestamp_utc, context_char_limit)` for
  deterministic prompt projection tests and internal loading.

Tests may also use the deterministic seam
`loader.select_residue_window(trigger_scope, rows, window_size)`, which builds
the same scope candidates used in production and applies the same selection
rules without touching MongoDB.

## Storage Row

The MongoDB collection is `internal_monologue_residue_state`. Rows contain:

- `residue_id`
- `character_id`
- `scope_key`
- `scope_kind`: `user_thread` or `group_scene`
- `platform`, `platform_channel_id`, `channel_type`, `global_user_id`
- `residue_text`: one short first-person private residue string
- `source_kind`: `chat` or `self_cognition`
- `source_refs`: sanitized episode/conversation references only
- `created_at`
- `schema_version`: `internal_monologue_residue.v2`
- deterministic `operation_id` derived from the completed episode and exact
  scope
- `disposition`: `append`, `replace_scope`, or `clear_scope`
- BSON `purge_at` used by the bounded TTL index

Rows do not store raw message bodies, prompts, delivery ids, action packets,
semantic memory packets, broad summaries, or full prior monologues.

## Loader And Projection

The loader builds candidate scopes from the current trigger:

1. Exact `user_thread`
2. Matching `group_scene`

It ranks eligible rows by that scope priority, then by recency, and caps the
selected window by `INTERNAL_MONOLOGUE_RESIDUE_WINDOW_SIZE`.

Projection converts selected rows into one age-labeled string for L2a. It
sorts selected rows newest-first before applying the character budget so tight
budgets keep fresh residue instead of accidentally preserving old rows from
input order.

## Prompt And Validation Contract

After visible outcome selection, the recorder reads the exact current-turn
`cognition_core_output.private_monologue` produced by G. It does not reconstruct
first-person subjectivity from `active_character_goal.reason` or the legacy
global analytic alias. The recorder then receives a minimal current-run
payload:

- `internal_monologue`
- `current_speaker_display_name`
- `exact_name_candidates`
- `ambient_evidence_summary`
- `incoming_residue_context`
- `source_reliability_notes`
- `visible_outcome_summary`: bounded dialog that was actually selected
- `surface_content_plan`: bounded semantic plan used for the response
- `visible_boundaries`: bounded expression constraints applied to that response

The recorder reconciles first-person cognition with the visible outcome and
surface constraints. This lets it distinguish a reason that was already
expressed, a thought intentionally retained by a boundary, and a private
reason that still has short-term continuity value.

The system prompt carries runtime `character_name` and `ambient_condition`.
It asks for strict JSON with exactly:

```json
{"disposition": "clear_scope", "residue_text": ""}
```

`append` and `replace_scope` require non-empty text; `clear_scope` requires an
empty text and persists an exact-scope barrier. A completed episode id is
required before the recorder calls the model or writes a v2 row. Non-empty
text must fit the configured row character limit and must not leak prompt,
model, schema, field, or process framing. Third-person self-reference using
`角色` is rejected. Vague relation words such as `对方`, `那个人`, `某人`, `他`,
and `她` are allowed.

For self-cognition group review, the recorder input may include
`source_reliability_notes = ["group review contained ambiguous second-person side-thread rows"]`
when the completed state carries
`conversation_progress.thread_reference_context.ambiguous_second_person_rows`.
This note is source reliability context only. It tells the recorder not to
preserve ambiguous second-person side-thread content as a current fact about
the active character; it is not a response gate, delivery rule, scheduler
input, durable memory write, or residue suppression rule.

Invalid output receives one bounded regeneration retry through the canonical
JSON parser. Invalid output after retry fails closed without a write. Repeating
the same operation is idempotent; a conflicting payload for the same operation
id fails closed. Rows without the canonical v2 contract are excluded from reads
and writes, so a replacement or clear marker cannot affect another scope.

## Lifecycle

For chat, loading happens before persona cognition and writing runs after the
completed episode in post-turn/background work. For self-cognition, writing
runs after the completed self-cognition state is available. The normal visible
`/chat` response path does not gain an extra foreground LLM call.

Goal cognition receives only the bounded
`internal_monologue_residue_context` projection. Appraisal, L3, dialog,
adapters, scheduler, and generic persistence paths must not receive raw prior
residue rows or the projected private continuity string.

## Config

- `INTERNAL_MONOLOGUE_RESIDUE_WINDOW_SIZE`: default `5`, min `1`, max `10`
- `INTERNAL_MONOLOGUE_RESIDUE_CONTEXT_CHAR_LIMIT`: default `3000`, min `200`,
  max `3000`
- `INTERNAL_MONOLOGUE_RESIDUE_ROW_CHAR_LIMIT`: default `220`, min `80`,
  max `500`
- `INTERNAL_MONOLOGUE_RESIDUE_RETENTION_HOURS`: default `48`, min `1`, max
  `720`; the collection owns a bounded TTL index

## Telemetry

Telemetry is sanitized and records only load/write/skip/retry outcomes, counts,
latency, route name, and status labels. It must not log residue text, prompts,
source packets, message bodies, delivery ids, or raw model output.

## Forbidden Consumers

Do not feed raw residue rows or raw prior residue text to:

- L1 subconscious cognition
- L2b boundary core
- L2d action initialization
- L3/dialog rendering
- adapters
- dispatcher or scheduler
- durable memory writers
- reflection promotion

Only V2 goal-cognition branches may consume the projected
`internal_monologue_residue_context`.
