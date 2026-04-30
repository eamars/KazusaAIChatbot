# typed message envelope — stage 2 plan

## Summary

- Goal: Eliminate the *class* of failures where transport-envelope data (mentions, reply targets, CQ codes, addressing) leaks into content-shaped fields and is then re-parsed by every downstream stage. Stage 1 (`rag_reply_mention_and_vague_input_plan.md`) patched two symptoms; Stage 2 fixes the structure so those symptoms cannot recur.
- Parent: Stage 1 of this initiative ships first as a stopgap. Stage 2 deletes the Stage 1 sanitizer because the inputs it sanitized are no longer reachable.
- Plan class: large
- Status: approved
- Overall cutover strategy: compatible, multi-phase. Storage shape changes are additive; consumers migrate behind a fallback; the fallback is removed only after a soak window.
- Highest-risk areas: storage migration of `conversation_history`, retrieval-path correctness during the dual-shape window, accidentally narrowing the bot's "addressed_to" so legitimately broadcast bot messages disappear from history filters.
- Acceptance criteria: adapters produce a typed `MessageEnvelope` with `body_text` cleaned of wire markers; conversation history rows store typed `addressed_to` for both user AND bot messages; `build_interaction_history_recent` filters by typed addressee instead of interleave heuristics; RAG search agents and cache key consume `body_text` only; Stage 1 sanitizer is deleted.

## Context

### The class problem (recap from Stage 1)

Platform adapters today flatten transport-envelope data — mentions, reply targets, CQ codes, role/channel/emoji refs — into a single prose `user_input` string. Every downstream stage (decontextualizer, RAG initializer, conversation_keyword_agent, persistent memory, cache keys, history rows, logs) is then asked to re-parse that string and tell envelope from content. Stage 1 added a deterministic sanitizer at one seam (RAG initializer slots) and a clarification flag at another (decontextualizer). Both are stopgaps. The class of leak remains.

### The asymmetric-addressing problem (new in Stage 2)

A separate but related structural gap surfaced from group-chat behavior:

- The *user* side has partial typing: `reply_context` exists in `ConversationMessageDoc` but is populated only when the platform's native reply feature was used. Pure @mentions and conversational addressing are not captured as structure. So "who is this message for" is platform-dependent and lossy.
- The *bot* side has no addressing record at all. When the bot speaks, the assistant row carries `role=assistant` and `platform_user_id=bot_id`, but no field says "this reply was directed at user X." In group chat, the bot may answer user A in turn N, user B in turn N+1, user A again in turn N+2 — and history retrieval can't tell those apart.

This asymmetry breaks [build_interaction_history_recent](src/kazusa_ai_chatbot/utils.py#L75) at [utils.py:75-132](src/kazusa_ai_chatbot/utils.py#L75-L132). The current implementation tries to reconstruct a user→bot subthread by walking the recent slice with two heuristics:

1. Find the last index where a *different* user spoke.
2. From there, find the first index where the *current* user spoke; everything from that index onward is the candidate subthread.
3. Then keep only `(role=user AND platform_user_id=current_user)` and `(role=assistant AND platform_user_id=bot)`.

The third filter assumes every bot turn in that window was addressed to the current user. In active multi-user channels that is false. Bot turns directed at user B leak into user A's persona context, and vice versa. Decontextualizer, cognition, and consolidation all read this contaminated slice.

The right fix is to record *who the bot was addressing* at write time, then filter retrieval by that structured field instead of by walk-and-guess.

## Stage 1 implementation status (as of plan approval)

Stage 1 is partially shipped. This section enumerates the as-built state so Stage 2 does not redo work and explicitly carries forward the items Stage 1 left undone.

### Stage 1 — IMPLEMENTED in production

- Decontextualizer extended with structured ambiguity fields: [persona_supervisor2_msg_decontexualizer.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py) emits `reference_resolution_status` (enum), `needs_clarification` (bool), `clarification_reason` (free-text) and the prompt teaches the model when to populate them.
- Schema typed fields on both `GlobalPersonaState` and `CognitionState` at [persona_supervisor2_schema.py:38-40](src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py#L38-L40) and [persona_supervisor2_schema.py:91-93](src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py#L91-L93).
- RAG skip on unresolved reference at [persona_supervisor2.py:36-49](src/kazusa_ai_chatbot/nodes/persona_supervisor2.py#L36-L49); the empty `rag_result` is built via `project_known_facts([], …)` so cognition consumers see the expected dict shape (Stage 1 mandate honoured).
- Cognition L2 (judgment_core) consumes the flag at [persona_supervisor2_cognition_l2.py:799-839](src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py#L799-L839); intent is forced to `CLARIFY` when the flag is true.
- Cognition L3 (content_anchor) consumes the flag at [persona_supervisor2_cognition_l3.py:589-590](src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py#L589-L590).
- Decontextualizer fallback path logs at WARN with input preview on LLM exception, parse exception, and missing-field cases ([persona_supervisor2_msg_decontexualizer.py:180-226](src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py#L180-L226)). The Stage 1 mandate to surface silent degradation is met.
- Bridge hardening for group-chat history landed in `build_interaction_history_recent`: if no current-user anchored slice can be formed, the helper now returns `[]` instead of falling back to raw channel history. This is a fail-closed containment patch for legacy rows, not the Stage 2 typed-addressing solution.

### Stage 1 — NOT IMPLEMENTED

- `_envelope_token_set(...)` registry helper does not exist in [persona_supervisor2_rag_supervisor2.py](src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py).
- `_sanitize_initializer_slots(...)` does not exist; the cache write path and cache read path do not run any sanitization.
- `platform_bot_id` is NOT in the RAG supervisor `context` dict at [persona_supervisor2.py:54-67](src/kazusa_ai_chatbot/nodes/persona_supervisor2.py#L54-L67); the sanitizer (when added) would have no source-of-truth for the current bot mention token.
- Reply-boilerplate token list is NOT plumbed into the RAG context.
- The RAG initializer prompt has not been updated to instruct the model that `<@bot_id>` / `[Reply to message]` / CQ reply markers are envelope, not content.
- `INITIALIZER_PROMPT_VERSION` is currently `initializer_prompt:v4` ([cache2_policy.py:17](src/kazusa_ai_chatbot/rag/cache2_policy.py#L17)). Whether the next bump is needed depends on which path Stage 2 takes (see "Migration choice" below).

### Adjacent typed-addressing code already in the repo

- [relevance_agent.py:102](src/kazusa_ai_chatbot/nodes/relevance_agent.py#L102) defines `_is_directly_addressed_to_bot(row)` — ad-hoc detection that walks individual rows. Stage 2 must consolidate this with the new typed `addressed_to_global_user_ids` so there is one source of truth instead of two.

## Approved migration path for the unfinished Stage 1 items

The unfinished Stage 1 work (sanitizer, registry, prompt rule, `platform_bot_id` plumbing) is a *stopgap* whose entire purpose is to compensate for wire markers reaching the RAG initializer. Stage 2 removes wire markers at the adapter, so a finished Stage 1 sanitizer would be deleted in Stage 2's Phase F.

**Approved path: Finish Stage 1 first.** Land the sanitizer, registry, prompt rule, and `platform_bot_id` plumbing per the Stage 1 plan before starting Stage 2 Phase A. This gives a production safety net while Stage 2 is in flight. Stage 2's Phase F deletes the sanitizer after typed envelopes make the sanitized input path unreachable.

The implementation agent must not skip the remaining Stage 1 safety net while this approved plan is in force. If the team later chooses to skip it and absorb the work into Stage 2, create a superseding plan or explicit amendment before execution begins.

## Goals

1. **Decouple content from transport** at the adapter seam. Downstream code stops needing to know what `<@bot_id>`, `[Reply to message]`, or `[CQ:reply,...]` mean.
2. **Make addressing first-class and symmetric.** Both user-authored and bot-authored conversation rows carry typed `addressed_to` regardless of whether the underlying platform exposes a reply feature.
3. **Replace interleave heuristics with typed filters** in `build_interaction_history_recent` and any other consumer that today guesses the user→bot subthread.
4. **Make stored conversation rows safe to keyword-search** — search agents query `body_text`, not the wire form.
5. **Make the cache key reflect semantic identity** — same query, same addressing, same cache entry across deployments and cosmetic variations.
6. **Delete the Stage 1 sanitizer** at the end. It was a stopgap; once `body_text` is clean, there is nothing to sanitize.

## Mandatory Rules

- Stage 2 is **additive then subtractive**. Add typed fields first, populate them, migrate consumers, delete legacy reliance only after a soak window with telemetry showing zero fallback hits.
- Adapters are the **only** place that converts wire form to typed form. No downstream stage may parse `<@\d+>`, `[CQ:...]`, `@everyone`, `<@&...>`, `<#...>`, `<:emoji:>` again.
- `body_text` in storage and retrieval contains user/bot-authored content only — no wire markers, no reply boilerplate, no CQ codes. The original wire form is retained in `raw_wire_text` for audit and forensic replay.
- `addressed_to` is symmetric. Both user-authored and bot-authored rows carry it. The bot side is sourced from the persona pipeline, NOT inferred from the next user message.
- `addressed_to` for bot messages defaults to the in-turn user's `global_user_id`. Cognition may override (e.g. when the bot deliberately addresses a different participant), but the override path must produce a deterministic value, not "infer later."
- Multi-bot deployments share channels. `addressed_to_global_user_ids` is a list of UUIDs (resolved through the user profile store), never a "is this the bot" boolean.
- `build_interaction_history_recent` is rewritten to filter by typed addressee. It MUST tolerate legacy rows (no `addressed_to`) during the migration window via an explicit fallback path that logs at INFO so the cutover can be tracked.
- The legacy fallback MUST preserve the Stage 1 bridge behavior: it may use the current conservative user-anchored heuristic, but it MUST NOT reintroduce the old raw-channel-history fallback when no current-user slice can be formed.
- `build_initializer_cache_key` is bumped to a new keying scheme that consumes `body_text` + typed addressing. Old cache rows are invalidated by the version bump.
- RAG search agents (`conversation_keyword_agent`, `conversation_search_agent`, `conversation_aggregate_agent`) query `body_text`. They MUST NOT fall back to `raw_wire_text` — if a row has only legacy `content`, it goes through the read-time normalizer.
- `MessageEnvelope` is a TypedDict, not a class. Keep it dict-shaped for LangGraph state compatibility.
- No silent backward-compat shims that hide migration progress. Every fallback path logs once per row at INFO with row id and reason.
- Follow project Python style (see `.agents/skills/py-style`): imports at top of file (no inline imports outside test code), `try`/`except` covers the minimum range of suspect calls, exception classes are always specific (never bare `except Exception:` outside the documented adapter boundary), every public helper has a complete docstring stating purpose / args / returns, default values live in one canonical place rather than scattered `.get(...)` defaults, and `try`/`except` is reserved for genuine external failure modes (parsing untrusted input, network, MongoDB, LLM calls).
- Follow `.agents/skills/cjk-safety` whenever editing `.py` files that contain CJK string content: single-quoted string delimiters when content contains `"` `"` (U+201C/U+201D), use a byte-copy script for large CJK insertions rather than the Write tool, and validate via `ast.parse` or `py_compile` before declaring the change complete.
- Attachments must remain storable end-to-end. The existing image-to-text conversion (an attachment's `description` field is what RAG and cognition consume) MUST persist unchanged in Stage 2. Binary payloads (`base64_data`, external `url`) MUST be preserved at the storage layer so future stages can switch on direct-modality consumption without a second migration. Stage 2 does not change what RAG or cognition see — they continue reading the text description.
- All new code must be modular with **defined interfaces**. Cross-module callers consume `Protocol` types or TypedDicts, not concrete classes. New behaviors plug in by implementing the protocol; no consumer is allowed to import a concrete adapter or normalizer directly.
- Documentation is part of the deliverable. Every new module gets a top-of-file docstring stating: what it owns, what its inputs and outputs are, which interfaces it implements, and which extension slot it exposes. A `README.md` is added to any new directory that holds more than one module.
- Each phase ships with an explicit checkpoint (entry criteria, success metrics, abort criteria, sign-off) before the next phase begins. See "Phase Checkpoints" below.
- **Embedding source is `body_text` plus attachment `description` text.** New conversation rows compute their vector embedding from this composed string, never from `content` / `raw_wire_text`. Old rows retain their original (wire-poisoned) embeddings; Stage 2 does not re-embed them at scale. The semantic-precision degradation on legacy rows is an accepted, time-bounded trade-off — see "Embedding policy" in Target State.
- **Dual-field Mongo query during the migration window.** Server-side regex/aggregation on `conversation_history` MUST query the union of `body_text` and `content`, with hits on `content` flagged for the read-time normalizer to clean before returning to the agent. New rows must populate `body_text`; the `content`-only branch exists strictly for legacy rows. Phase F removes the `content`-side branch.
- **`mentioned_bot` and `reply_context.reply_to_current_bot` lifecycle.** After Phase E, all consumers read typed `addressed_to_global_user_ids` and the typed `mentions` list. `mentioned_bot` and `reply_context.reply_to_current_bot` become *derived shims* populated from the envelope for back-compat readers; they MUST NOT be the source of truth anywhere new. Phase F deletes them where consumers have migrated, and explicitly retains them only where Stage 2 deferred a consumer migration (call out in commit message).
- **DM (private channel) addressing defaults.** In a DM (`channel_type="private"`), the bot is structurally the only other party. Inbound user rows: `addressed_to_global_user_ids=[bot_global_user_id]`, `reply_context.reply_to_current_bot` defaults to `True` regardless of platform-native reply usage. Outbound bot rows: `addressed_to_global_user_ids=[current_user_global_user_id]`, `broadcast=False`. These defaults are deterministic and applied at the envelope factory; consumers do not special-case DMs.
- **`body_text` shape on assistant rows.** For `role=assistant` rows, `body_text` is the cognition output **before outbound rendering** (no `<@user>` prefixes, no platform reply markers). `raw_wire_text` carries the rendered, on-the-wire form when outbound rendering exists. Replay/audit consumers read `raw_wire_text`; retrieval / search / cognition consumes `body_text`.
- **Soak window cap is 3 days, not 1 week.** Phase F's "no legacy fallback hits" gate runs over a maximum 3-day window. Justification: the project's policy is that recent-history retrieval does not span beyond ~3 days; older context belongs to RAG. A row not seen in the recent-history window for 3 days has aged out of the relevant slice. Sign-off only requires that no legacy row appears in the active recent-history window across active channels for 3 consecutive days; quiet channels do not block Phase F.
- **No fabricated telemetry counters.** Sign-off evidence in this plan is restricted to artefacts the codebase can produce today: log-line greps over `INFO` fallback logs, sample-based MongoDB inspection of new vs legacy rows, focused regression test output, and full pytest runs. If a metrics emission stack is added later, the plan can be amended; until then, evidence comes from logs and tests, not counters.

## Must Do

- Define `MessageEnvelope` TypedDict with `body_text`, `mentions`, `reply`, `attachments`, `addressed_to_global_user_ids`, `raw_wire_text`.
- Extend `ConversationMessageDoc` with `body_text` (str), `addressed_to_global_user_ids` (list[str]), and `raw_wire_text` (str). Keep legacy `content` field readable for migration window.
- Implement adapter normalization: each platform adapter (QQ, Discord, future) populates `MessageEnvelope` from its raw input. Mention extraction, reply target resolution, attachment extraction all happen here.
- Plumb the envelope through `IMProcessState` so `user_input` is derived from `body_text`, and structural fields (`mentioned_bot`, `reply_context.reply_to_current_bot`, `indirect_speech_context`) are sourced from the envelope rather than scattered re-derivations.
- Add a cognition output field `target_addressed_user_ids: list[str]` (default: `[current_user_global_id]`). Dialog agent or consolidator writes this onto the saved bot row's `addressed_to_global_user_ids`.
- Rewrite `build_interaction_history_recent` to filter by `(role, global_user_id, addressed_to_global_user_ids)`. Add a legacy-row fallback path that retains the current heuristic only when `addressed_to_global_user_ids` is missing.
- Migrate RAG search agents to query `body_text`. Add a read-time normalizer for legacy rows that lack `body_text` (strip known envelope tokens from `content`).
- Bump `build_initializer_cache_key` keying scheme to include `body_text` + typed addressing intent. Bump `INITIALIZER_PROMPT_VERSION`.
- Update memory consolidator to persist `body_text` only.
- Delete Stage 1 sanitizer (`_sanitize_initializer_slots`, `_envelope_token_set`) and its prompt mirror.
- Migrate decontextualizer's boolean `needs_clarification` to a structured `referents: [{phrase, role: subject|object|time, status: resolved|unresolved}]`, and update cognition consumers (`call_judgment_core_agent`, `call_content_anchor_agent`) to consume narrow per-referent clarification instead of binary skip. Note: the boolean and its consumers already exist in production — Stage 2 adds `referents` *alongside* and removes the boolean only after the new path is verified.
- Consolidate `_is_directly_addressed_to_bot` in [relevance_agent.py:102](src/kazusa_ai_chatbot/nodes/relevance_agent.py#L102) with the new typed `addressed_to_global_user_ids`. Relevance must consume the typed field directly; the per-row helper is deleted.

## Deferred

- Backfilling `addressed_to_global_user_ids` for historical bot rows. Old rows stay with empty `addressed_to`; the legacy fallback in `build_interaction_history_recent` handles them until they age out of the recent window.
- Backfilling `body_text` on historical user rows by editing existing documents. Read-time normalization is sufficient; a one-time write-side migration is optional and not required for correctness.
- Multi-addressee bot replies in cognition (i.e. bot deliberately addressing two users). Stage 2 supports the field shape (list); cognition emits a single addressee in v1. Multi-addressee dialog generation is a follow-up.
- `mentions[].role=everyone` semantics for `@everyone` / `@here` beyond storing them. Whether they should broaden retrieval is out of scope.
- Channel-level metadata enrichment (channel name, channel topic) being typed. Stage 2 stays focused on per-message envelope.
- **Outbound mention/reply rendering is NOT implemented today and is NOT in Stage 2.** The bot's outgoing replies do not currently render `<@user>` mentions or platform-native reply markers as a function of `addressed_to_global_user_ids`. Stage 2 stores typed addressing on bot rows so a future Stage 3 can introduce a symmetric `EnvelopeRenderer` Protocol, but Stage 2 itself does not modify outbound formatting. Treat this as a known gap, not silent omission.
- **Re-embedding legacy `conversation_history` rows.** Embeddings on rows written before Stage 2 encode `content` (with wire markers). Stage 2 does NOT batch-re-embed them. Vector search on the legacy slice retains its current precision until those rows age out of retrieval windows. A future cleanup job MAY re-embed; not required for Stage 2 acceptance.
- **Persistent memory store envelope migration.** Memory rows are LLM-summarized — the consolidator pipeline produces text the LLM authored, possibly quoting user/dialog input verbatim. After Stage 2 the consolidator reads `body_text` (clean), so new memory rows will not contain wire markers. Existing wire-poisoned memory rows are accepted as residual; no schema migration of `MemoryDoc` is required.
- **`inferred_thread` reply derivation.** Removed from the `ReplyTarget.derivation` enum during this review. If thread inference is needed later, it ships in a separate stage with its own contract; Stage 2 produces only `platform_native` or `leading_mention`.

## Cutover Policy

Compatible, phased. Each phase is independently shippable and reversible.

- Phase A (schema): add new fields to `ConversationMessageDoc`, `IMProcessState`, `CognitionState`, with defaults. Pure type extension; no behavior change. Old code reads/writes old fields; new fields are absent or empty.
- Phase B (adapter intake): adapters populate the envelope. `IMProcessState.user_input` becomes derived from `body_text`. `mentioned_bot` and friends remain populated for back-compat readers.
- Phase C (storage write path): new conversation rows are saved with `body_text`, `addressed_to_global_user_ids`, `raw_wire_text`. Old `content` is still written for back-compat readers during the migration window.
- Phase D (bot-side addressing): cognition emits `target_addressed_user_ids`. `save_conversation` for assistant rows writes `addressed_to_global_user_ids` from this field.
- Phase E (consumer migration): `build_interaction_history_recent`, RAG search agents, cache key, memory consolidator switch to `body_text` and `addressed_to_global_user_ids`. Each carries a logged fallback for legacy rows.
- Phase F (cleanup): after telemetry shows zero fallback hits for one week, delete fallbacks, delete legacy `content` writes, delete Stage 1 sanitizer, drop deprecated fields.

A rollback at any phase reverts to the prior shape; new fields are tolerated as ignored extras.

## Agent Autonomy Boundaries

The implementation agent may edit:

- New file: `src/kazusa_ai_chatbot/message_envelope.py` (TypedDict + helpers).
- `src/kazusa_ai_chatbot/db/schemas.py` (extend `ConversationMessageDoc`).
- `src/kazusa_ai_chatbot/db/conversation.py` (write-path save_conversation; read-time normalizer).
- `src/kazusa_ai_chatbot/state.py` (extend `IMProcessState`).
- `src/kazusa_ai_chatbot/utils.py` (rewrite `build_interaction_history_recent`).
- `src/kazusa_ai_chatbot/service.py` (envelope plumbing at intake; assistant-row write enrichment).
- Adapter files in `src/kazusa_ai_chatbot/...` that produce `ChatRequest` (one per platform).
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py` (consume body_text, emit referents).
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py` (CognitionState `target_addressed_user_ids`, `referents`).
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` (envelope/addressee plumbing).
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`, `_l3.py` (consume referents).
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py` (delete sanitizer; consume body_text).
- `src/kazusa_ai_chatbot/rag/cache2_policy.py` (key bump).
- `src/kazusa_ai_chatbot/rag/conversation_*.py` (search agents query body_text).
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py` (emit target_addressed_user_ids).
- Tests under `tests/`.

The implementation agent must NOT:

- Mutate existing conversation_history documents in MongoDB beyond defaulting missing fields at read time.
- Change `ChatRequest`'s public API in ways that break external callers without a version step.
- Remove Stage 1 sanitizer before consumer migration is verified end-to-end.
- Add backfill scripts that rewrite history rows in place. If a write-side migration is needed it goes through a separate plan with explicit go/no-go.

## Target State

### Typed envelope (intake)

```text
MessageEnvelope (TypedDict, total=False):
  body_text: str                          # user/bot content only — no wire markers
  mentions: list[Mention]                 # typed mentions
  reply: ReplyTarget | None               # typed reply, populated whether platform reply
                                          #   was used OR derived from intra-message @mention
  attachments: list[AttachmentDoc]
  addressed_to_global_user_ids: list[str] # resolved UUIDs of intended recipients
  raw_wire_text: str                      # forensic copy of original input

Mention:
  platform_id: str
  global_user_id: str
  display_name: str
  role: "bot" | "user" | "everyone"

ReplyTarget:
  message_id: str
  author_global_user_id: str
  excerpt: str
  derivation: "platform_native" | "leading_mention"
```

`derivation` is informational only — used in log lines and audit reports to evaluate how often platform-native vs. derived addressing fires. Stage 2 does NOT include an `inferred_thread` value; thread inference is not part of this scope, and adding it later requires an explicit follow-up plan to keep the envelope module from accreting heuristics.

### Conversation history shape (storage)

```text
ConversationMessageDoc additions:
  body_text: str
  addressed_to_global_user_ids: list[str]   # symmetric across role=user and role=assistant
  raw_wire_text: str                        # only meaningful for role=user
  mentions: list[Mention]                   # snapshot from envelope
  reply_context: ReplyContextDoc            # already exists; populated more aggressively now
```

Symmetry rule:

- For `role=user` rows: `addressed_to_global_user_ids` is the union of typed mentions (role=bot|user) and the reply target's author UUID, deduplicated.
- For `role=assistant` rows: `addressed_to_global_user_ids` is whatever cognition emitted in `target_addressed_user_ids`. If the bot is broadcasting (no specific addressee), the list is empty AND a `broadcast: true` flag is set so downstream consumers can distinguish "not addressed to anyone in particular" from "missing field, legacy row."

### `build_interaction_history_recent` (after rewrite)

```text
def build_interaction_history_recent(
    chat_history_recent: list[dict],
    current_global_user_id: str,
    bot_global_user_id: str,
) -> list[dict]:
    """Filter recent history to messages between current_user and the bot."""
    result = []
    for msg in chat_history_recent:
        if msg.get("role") == "user" and msg.get("global_user_id") == current_global_user_id:
            result.append(msg)
            continue
        if msg.get("role") == "assistant" and msg.get("global_user_id") == bot_global_user_id:
            addressed = msg.get("addressed_to_global_user_ids")
            if addressed is None:
                # Legacy-row fallback (Phase E only). Logs INFO once per row.
                if _legacy_subthread_fallback(msg, chat_history_recent, current_global_user_id):
                    result.append(msg)
                continue
            if current_global_user_id in addressed or msg.get("broadcast", False):
                result.append(msg)
    return result
```

Note: legacy fallback is only invoked for missing `addressed_to_global_user_ids`. After Phase F it is deleted.

### Decontextualizer / cognition (after migration)

Boolean `needs_clarification` is replaced with `referents: [{phrase, role, status}]`. Cognition's `call_content_anchor_agent` issues narrow clarification ("你说的『这些』是指什么？") only for the unresolved entries while still using resolved anchors for retrieval. Binary RAG-skip cliff is gone.

### Stage 1 sanitizer

`_sanitize_initializer_slots` and `_envelope_token_set` are deleted. The prompt rule that mirrored them is deleted. The RAG initializer reads `body_text` only — wire markers cannot reach it.

### Attachment handling — preserve current workflow, expand storage capability

Stage 2 does NOT change what cognition or RAG sees from attachments. The current image-to-text path stays intact:

- The adapter (or upstream caller) supplies `description` as a textual representation of each attachment (alt-text, OCR summary, vision-LLM caption — Stage 2 is agnostic to which).
- That `description` is appended into `body_text` (or held adjacent on the envelope) so relevance, decontextualizer, RAG, and cognition see the textual form, exactly as today.
- The binary (`base64_data` for small inline payloads, `url` for CDN/S3 references) is preserved at the *storage* layer. `ConversationMessageDoc.attachments` already supports `media_type`, `url`, `base64_data`, `description`, `size_bytes`. Stage 2 keeps writing all five.
- No RAG agent or cognition prompt directly consumes `base64_data` in Stage 2. The capability is preserved so a later stage can opt in (e.g. an image-aware RAG agent or a direct vision-LLM finalizer) without a schema migration.

**Storage policy:**

- Small attachments (under a configurable `INLINE_ATTACHMENT_BYTE_LIMIT`, default 256 KB): persist `base64_data` inline on the conversation row.
- Larger attachments: persist `url` only on the conversation row; the binary lives in external storage owned upstream.
- `description` is always populated. If the upstream adapter cannot generate one, the attachment record is dropped from `body_text` insertion (so the LLM never receives an empty placeholder) but remains in `attachments` with a `description=""` flag for forensic visibility.

**Extension slot:** see "Future Expansion Slots" — `AttachmentHandler` protocol — for how a new modality (audio, video, structured file) plugs in without touching downstream code.

### Boundary cases

Three cases the envelope factory MUST resolve deterministically — no per-consumer special-casing.

**DM (private channel).** When `channel_type="private"`:

- Inbound user row: `addressed_to_global_user_ids = [bot_global_user_id]` always; `reply_context.reply_to_current_bot = True` always (overrides whatever the platform reported, since DMs have no other addressee). `mentions` may still be empty if the user did not type an @mention.
- Outbound bot row: `addressed_to_global_user_ids = [current_user_global_user_id]`, `broadcast = False`.
- Consumers do NOT special-case `channel_type` for addressing — they read the typed fields, which the factory has already filled correctly.

**Indirect speech (Situation B: user A talking ABOUT the bot to user B).** When `state["indirect_speech_context"]` is non-empty on the inbound side:

- Inbound user row: `addressed_to_global_user_ids` is populated from envelope mentions and reply target as usual. The bot is typically NOT in that list — that's the whole point of indirect speech.
- Outbound bot row (when relevance allows the bot to chime in): `addressed_to_global_user_ids = [user_A_global_user_id]` (the speaker), `broadcast = False`. The topic person (user B) is NOT added.
- Rationale: history retrieval for user A should include this turn (they spoke); retrieval for user B should not, since the bot's reply was directed at user A.

**Coalesced turns.** When [chat_input_queue](src/kazusa_ai_chatbot/chat_input_queue.py) collapses multiple raw user messages into one survivor turn:

- Inbound: the survivor's `MessageEnvelope` is the canonical record. Collapsed siblings are stored separately by the queue's existing mechanism; their envelopes are not retrofitted.
- Outbound: `addressed_to_global_user_ids = [survivor_user_global_user_id]`. Multi-user coalescence is rare and out of Stage 2 scope; if it appears, the consolidator can split the assistant row in a future stage.

### Body_text composition (user vs. assistant rows)

- **`role=user` rows:** `body_text` is the wire-stripped user-authored content. Attachment text (`description` of each `AttachmentRef`) is composed onto `IMProcessState.user_input` for downstream prompts but is NOT concatenated into `ConversationMessageDoc.body_text` itself — the structured `attachments` list is the source of truth for attachment text. The composition `user_input = body_text + "\n" + "\n".join(attachment.description for attachment in attachments if attachment.description)` happens in `service.py` at intake time, mirroring today's behavior.
- **`role=assistant` rows:** `body_text` is the cognition-output dialog string before any outbound rendering — no `<@user>` prefixes, no platform reply markers, no CQ codes. `raw_wire_text` carries the rendered, on-the-wire form when an outbound renderer exists (Stage 3+); for now `raw_wire_text == body_text` for assistant rows since outbound rendering is not implemented.
- Replay/audit consumers read `raw_wire_text`; retrieval, search, embedding, and cognition consume `body_text`.

### Embedding policy

- New conversation rows: `embedding = get_text_embedding(body_text + attachment_description_text)`. The composed string matches what `IMProcessState.user_input` would have been, ensuring vector-search semantics align with consumer-visible content.
- The single composition helper lives in `db/conversation.py` next to `save_conversation`. It is the only place that builds the embedding-source string; agents and consumers do NOT recompute it.
- Legacy rows: their `embedding` field is left untouched. Vector search will continue to match wire-poisoned legacy embeddings until those rows age out of retrieval windows. The 3-day soak cap (see "Phase Checkpoints F" and `chat_history_recent` retention) bounds the visible impact.
- Re-embedding legacy rows is in Deferred — Stage 2 does NOT run a batch re-embed.
- During the dual-shape window, vector search results are mixed-shape; consumers MUST tolerate this and the read-time normalizer cleans `content` to a usable `body_text` after the vector hit returns. `embedding` itself is not normalized — there is no equivalent of "strip wire markers from a vector."

### Mongo query and index migration

- During the migration window (Phase C through Phase E completion), keyword regex and aggregation `$match` filters in [db/conversation.py:87](src/kazusa_ai_chatbot/db/conversation.py#L87) and [db/conversation.py:183](src/kazusa_ai_chatbot/db/conversation.py#L183) issue an `$or` over `body_text` and `content`. Hits matching `content` only (no `body_text` field present) are passed through the read-time normalizer before reaching the agent.
- Phase F drops the `content` branch and the `$or`; only `body_text` is queried.
- Index plan: a compound index on `(platform, platform_channel_id, addressed_to_global_user_ids, timestamp DESC)` is added in Phase A so per-user history retrieval is index-served. A text/regex index on `body_text` is added in Phase A. The existing vector index on `embedding` is kept; no rebuild is needed because the field name does not change. Index DDL lives in `db/bootstrap.py` or wherever existing indexes are declared.
- Index migration is a one-shot operation at deploy time; reverting Phase A leaves the indexes in place harmlessly (they are unused but compatible).

### Module boundaries and interfaces

Stage 2 introduces three Protocol-typed interfaces. Consumers depend on the protocol; implementations are registered.

```text
src/kazusa_ai_chatbot/message_envelope/
  __init__.py              # public re-exports of types and the factory
  README.md                # module-level documentation
  types.py                 # TypedDicts: MessageEnvelope, Mention, ReplyTarget, AttachmentRef
  factory.py               # build_envelope(req, normalizer, resolver, handler_registry) -> MessageEnvelope
  protocols.py             # EnvelopeNormalizer, MentionResolver, AttachmentHandler protocols
  registry.py              # platform-keyed normalizer/handler registries
  normalizers/
    __init__.py
    qq.py                  # QQEnvelopeNormalizer implementing EnvelopeNormalizer
    discord.py             # DiscordEnvelopeNormalizer (future)
  attachment_handlers/
    __init__.py
    image.py               # ImageAttachmentHandler — current image-to-text behavior
    placeholder.py         # PlaceholderAttachmentHandler — fallback/no-op
```

**Interface 1 — `EnvelopeNormalizer` (Protocol):**

```text
class EnvelopeNormalizer(Protocol):
    """One implementation per platform. Converts raw wire input to typed envelope.

    Implementations are platform-pure: they MUST NOT perform user-id resolution
    (delegated to MentionResolver) or attachment description (delegated to
    AttachmentHandler). They are stateless and side-effect free.
    """

    platform: str

    def normalize(
        self,
        raw_wire_text: str,
        raw_mentions: list[RawMention],
        raw_reply: RawReply | None,
    ) -> NormalizedEnvelopeFragment: ...
```

**Interface 2 — `MentionResolver` (Protocol):**

```text
class MentionResolver(Protocol):
    """Resolves platform-specific mention identifiers to global user UUIDs.

    Default implementation reads from the user_profiles store. Tests inject a
    fixture-backed resolver. Multi-bot deployments can compose resolvers if
    they share a channel.
    """

    async def resolve(
        self,
        platform: str,
        platform_user_id: str,
    ) -> ResolvedMention: ...
```

**Interface 3 — `AttachmentHandler` (Protocol):**

```text
class AttachmentHandler(Protocol):
    """One implementation per attachment-modality (image, audio, video, ...).

    The handler is responsible for producing the text description that flows
    into body_text and for choosing the storage shape (inline vs URL). It does
    NOT decide whether downstream consumers see the binary; that is a global
    policy in Stage 2 (text-only) and a future toggle in Stage 3+.
    """

    media_type_prefix: str  # e.g. "image/", "audio/"

    async def describe(
        self,
        attachment_in: AttachmentIn,
    ) -> AttachmentRef: ...

    def storage_shape(
        self,
        attachment_in: AttachmentIn,
    ) -> Literal["inline", "url_only", "drop"]: ...
```

**Coupling rules:**

- `service.py` and adapter code depend ONLY on `message_envelope.factory.build_envelope(...)` and the protocol types. They never import a concrete normalizer or handler.
- Cognition, RAG, decontextualizer, relevance, consolidator depend on `MessageEnvelope` TypedDict fields and `ConversationMessageDoc` fields. They do NOT import anything from `message_envelope/normalizers/` or `message_envelope/attachment_handlers/`.
- `message_envelope/registry.py` is the only module that knows the concrete classes; it is a thin lookup keyed by `platform` (for normalizers) or `media_type_prefix` (for attachment handlers).
- Each Protocol lives in `protocols.py`; concrete implementations live alongside. Adding a new platform or modality is one new file plus one registry entry — no edits to consumers.

## Future Expansion Slots

Each slot is an explicit extension point preserved in Stage 2 so future work plugs in without touching consumers.

| Slot | Extension shape | Touch points |
|---|---|---|
| New platform (Telegram, WeChat, etc.) | Implement `EnvelopeNormalizer` in `message_envelope/normalizers/<platform>.py`; register in `registry.py`. | One new file + one registry entry. No consumer change. |
| New attachment modality (audio, video, structured) | Implement `AttachmentHandler` in `message_envelope/attachment_handlers/<modality>.py`; register by `media_type_prefix`. | One new file + one registry entry. `body_text` continues to receive `description`. |
| Direct-binary RAG (e.g. CLIP image embeddings) | Add a new RAG agent that reads `ConversationMessageDoc.attachments[*].url` or `.base64_data`; no schema change because storage already preserves the binary. | One new RAG agent + initializer slot prefix. Existing agents unchanged. |
| Vision-capable cognition for current turn | Switch `IMProcessState.user_multimedia_input` consumer to call vision endpoints directly; description stays as fallback. | One cognition prompt change; no envelope, storage, or RAG change. |
| Multi-addressee bot replies | Cognition emits multiple UUIDs in `target_addressed_user_ids` (already a list). Dialog generates a single message addressed to all. | Cognition prompt change only. |
| Per-referent ambiguity escalation (multi-step grounding) | Decontextualizer emits richer `referents` entries (e.g. `candidates: [...]`); cognition consumes additional fields. | Decontextualizer prompt + cognition L2/L3; consumers tolerate unknown fields. |
| Channel-level metadata typing (channel topic, pinned messages) | New `ChannelEnvelope` TypedDict alongside `MessageEnvelope`; same protocol pattern. | New module under `message_envelope/`; out of scope for Stage 2. |
| Per-mention semantic role (questioned, addressed, quoted) | Extend `Mention` TypedDict with `semantic_role: Literal[...]` field; default to existing behavior. | Adapter normalizers; consumers tolerate unknown role. |

Each slot is documented in the corresponding module's docstring as "Extension point: ..." so future implementers find it without reading this plan.

## Design Decisions

| Decision | Policy | Rationale |
|---|---|---|
| Envelope ownership | Adapter only | Single seam to maintain; downstream code consumes typed fields. |
| Wire-text retention | Keep as `raw_wire_text` | Forensic replay, debug, audit. Never used for retrieval. |
| Addressing symmetry | Both user AND bot rows carry `addressed_to_global_user_ids` | Eliminates interleave-heuristic guessing in retrieval. |
| Multi-addressee | List of UUIDs, not boolean | Multi-bot deployments and group addressing both work. |
| Bot addressee source | Cognition emits `target_addressed_user_ids` | Persona pipeline already knows the in-turn user; addressee is a deterministic side effect, not an inference. |
| Broadcast bot replies | Empty list + `broadcast: true` | Distinguishes intentional broadcast from missing-field legacy. |
| Legacy fallback | Logged INFO, removed after soak window | Cutover progress is observable; no silent drift. |
| Cache key | `body_text` + typed addressing intent | Same semantic query → same cache entry across deployments and cosmetic variants. |
| Decontextualizer migration | Boolean → structured `referents` | Eliminates Stage 1's binary skip cliff; supports partial-vagueness inputs. |
| Stage 1 deletion | Delete sanitizer in Phase F | Stage 1 was a stopgap by design; its inputs are no longer reachable. |
| Attachment binary policy | Persist; don't consume in RAG/cognition | Storage capability preserved; consumer flow unchanged. Future direct-binary stages opt in without a migration. |
| Module structure | Protocol-typed interfaces; consumers depend on the protocol, not implementations | Per project py-style skill rule on modularity; future platforms/modalities plug in by registration only. |
| Documentation | Module-level docstring + per-extension-point note + per-directory README | Sustainability: future implementers don't need to re-derive the design from this plan. |
| Phase gating | Explicit checkpoint between phases | Each phase can be reverted cleanly; the next phase doesn't start until the previous one is verified in production. |

## Change Surface

| File | Changes |
|---|---|
| `src/kazusa_ai_chatbot/message_envelope.py` (new) | `MessageEnvelope`, `Mention`, `ReplyTarget` TypedDicts; helpers for stripping wire tokens, resolving mentions to UUIDs, computing user-side `addressed_to_global_user_ids`. |
| `db/schemas.py` | Extend `ConversationMessageDoc` with `body_text`, `addressed_to_global_user_ids`, `raw_wire_text`, `mentions`, `broadcast`. |
| `db/conversation.py` | Save-path defaulting; embedding source pivot to `body_text` + composed attachment descriptions; read-time normalizer for legacy rows with INFO logging; dual-field `$or` queries during migration window; remove `$or` in Phase F. |
| `db/bootstrap.py` (or wherever indexes are declared) | Add compound index on `(platform, platform_channel_id, addressed_to_global_user_ids, timestamp DESC)` and a regex/text index on `body_text`. Existing vector index on `embedding` retained. |
| `state.py` | Extend `IMProcessState` with envelope-derived typed fields. |
| `utils.py` | Rewrite `build_interaction_history_recent`; add `_legacy_subthread_fallback` with INFO logging. |
| `service.py` | Plumb envelope at intake; enrich saved assistant rows with `addressed_to_global_user_ids` from cognition. |
| Each platform adapter | Produce `MessageEnvelope`. Strip CQ codes / Discord tags / mentions / reply markers from `body_text`; populate typed fields. |
| `persona_supervisor2_msg_decontexualizer.py` | Consume `body_text`; emit `referents`; drop the Stage 1 ambiguity boolean (compat-aliased through one release). |
| `persona_supervisor2_schema.py` | Add `target_addressed_user_ids`, `referents` fields. |
| `persona_supervisor2.py` | Plumb envelope and addressee fields between stages; build empty rag_result via `project_known_facts([], …)` is no longer needed because partial-skip is structured. |
| `persona_supervisor2_cognition_l2.py`, `_l3.py` | Consume `referents`; narrow-clarification anchor logic. |
| `persona_supervisor2_rag_supervisor2.py` | Consume `body_text`. Delete the Stage 1 safety-net sanitizer (`_sanitize_initializer_slots`, `_envelope_token_set`, and the prompt rule mirror) in Phase F after typed envelopes make the sanitized input path unreachable. |
| `nodes/relevance_agent.py` | Replace `_is_directly_addressed_to_bot` with a read of `addressed_to_global_user_ids`. Delete the helper. |
| `rag/cache2_policy.py` | New keying scheme over `body_text` + typed addressing; bump `INITIALIZER_PROMPT_VERSION`. |
| `rag/conversation_keyword_agent.py`, `conversation_search_agent.py`, `conversation_aggregate_agent.py` | Query `body_text`; tolerate legacy `content` via read-time normalizer. |
| `nodes/dialog_agent.py` | Emit `target_addressed_user_ids` (default to current user). |
| `nodes/persona_supervisor2_consolidator.py` | Persist `body_text` for memory. |
| Tests | New tests across all phases (see Verification). |

## Phase Checkpoints

Each phase has an explicit checkpoint with **entry criteria**, **success metrics**, **abort criteria**, and **sign-off evidence**. The next phase does not start until the previous checkpoint is signed off. Checkpoint evidence is recorded in the "Execution Evidence" section at the bottom of this document.

### Checkpoint A → B (Schema → Adapter intake)

- **Entry criteria:** Stage 0 and Stage 0.5 are signed off; Stage 1 status is documented in this plan's Stage 1 status section.
- **Success metrics:** All new TypedDicts compile (`python -m py_compile`); `pytest tests/test_message_envelope.py -q` passes; existing test suite unchanged; no production behavior change observable in logs.
- **Abort criteria:** Existing tests fail with new fields present; type-checker complaints not resolvable without schema redesign.
- **Sign-off evidence:** compile output, full test suite output, diff of new TypedDict fields, top-of-file docstrings present in every new module, `message_envelope/README.md` checked in.

### Checkpoint B → C (Adapter intake → Storage write path)

- **Entry criteria:** Checkpoint A signed off; QQ adapter implements `EnvelopeNormalizer`; image attachment handler implements `AttachmentHandler`.
- **Success metrics:** Adapter unit tests show wire markers stripped from `body_text` for the live failure input (`[Reply to message] <@bot> 你知道这些是什么意思么？`); `<@bot_id>` appears in `mentions[].role="bot"` with empty `body_text` mention residue; `description` text identical to pre-Stage-2 output for image attachments (regression on the image-to-text path); `IMProcessState.user_input` derived from `body_text` in the persona pipeline.
- **Abort criteria:** Image-to-text regression (description differs from baseline); adapter strips legitimate user content; `mentions` resolution rate against user_profiles drops below baseline.
- **Sign-off evidence:** before/after diff of adapter unit-test fixtures, image-description regression suite output, manual smoke for the live failure input, telemetry on `mentions` resolution rate.

### Checkpoint C → D (Storage write path → Bot-side addressing)

- **Entry criteria:** Checkpoint B signed off; new conversation rows write `body_text`, `addressed_to_global_user_ids` (user side), `raw_wire_text`, `mentions`, full `attachments` (with `description`, plus `base64_data` or `url` per storage policy).
- **Success metrics:** New rows in MongoDB have all envelope fields; legacy rows still readable; read-time normalizer produces a `body_text` for legacy rows that matches the new path on a captured replay set; attachment storage shape policy (inline vs URL) is honored.
- **Abort criteria:** Read-time normalizer produces incorrect `body_text` for any captured legacy row; conversation history insertion latency increases >20%; attachments lose `description` or `url`/`base64_data` post-write.
- **Sign-off evidence:** MongoDB schema inspection on a sample of new rows, replay test output, latency comparison, attachment field-presence audit.

### Checkpoint D → E (Bot-side addressing → Consumer migration)

- **Entry criteria:** Checkpoint C signed off; cognition emits `target_addressed_user_ids`; assistant rows are written with `addressed_to_global_user_ids` and `broadcast`.
- **Success metrics:** In a synthetic group-chat scenario (bot serves user A → user B → user A), saved assistant rows carry the correct addressee UUIDs; default-vs-override telemetry shows defaults dominate (>95%) until cognition opts in to overrides.
- **Abort criteria:** Cognition emits empty `target_addressed_user_ids` and `broadcast=false` (the "missing field" state); persona pipeline cannot determine in-turn user UUID at write time.
- **Sign-off evidence:** group-chat regression test output, telemetry snapshot of `target_addressed_user_ids` cardinality distribution.

### Checkpoint E → F (Consumer migration → Cleanup)

This checkpoint has three sequential sub-gates because the `referents` migration on a weak local Gemma is high-risk and must be staged carefully.

#### Sub-gate E1 — Consumers migrated to typed addressing (no `referents` work)

- **Entry criteria:** Checkpoint D signed off; `build_interaction_history_recent` rewritten with typed filtering and logged legacy fallback; RAG search agents query `body_text`; cache key bumped; relevance reads typed `addressed_to_global_user_ids`; `mentioned_bot` and `reply_context.reply_to_current_bot` reduced to derived shims (still populated for back-compat readers).
- **Success metrics:** Log-line audit on a representative day shows the legacy-fallback INFO log fires only on rows older than the recent-history window (i.e. the fallback exists for legacy rows but new rows never trigger it). RAG keyword search on `<@bot_id>` against the live failure input returns zero hits in `body_text`; legacy `content`-only rows go through the read-time normalizer.
- **Abort criteria:** New rows trigger the fallback (indicates a write path is missing `addressed_to_global_user_ids`); cache hit rate degrades >10% from pre-bump baseline and does not recover within 24h.
- **Sign-off evidence:** log-grep output showing fallback hits are dominated by legacy rows; cache hit-rate observation from production logs; `pytest tests/test_rag_search_body_text.py -q` and `pytest tests/test_build_interaction_history_recent.py -q` pass.

#### Sub-gate E2 — `referents` emitted alongside boolean

- **Entry criteria:** E1 signed off; decontextualizer emits both `needs_clarification` (legacy) and `referents` (new structured) on every turn.
- **Success metrics:** Captured-input regression set on the local Gemma shows `referents` is populated correctly for ≥80% of cases that the boolean already classified (the new shape does not regress on the cases the boolean already handled); the boolean continues to drive cognition for now.
- **Abort criteria:** `referents` produces empty or malformed output on >20% of cases the boolean handles; local Gemma response latency increases beyond the existing prompt budget.
- **Sign-off evidence:** captured-input regression report; before/after prompt size measurement; `pytest tests/test_decontexualizer_referents.py -q` passes.

#### Sub-gate E3 — Cognition switches to `referents`

- **Entry criteria:** E2 signed off; cognition L2 (judgment_core) and L3 (content_anchor) read `referents` as the authoritative ambiguity signal; the boolean is still emitted but is no longer load-bearing.
- **Success metrics:** Captured-input regression set shows narrow clarification (per-referent) on partial-vagueness inputs that previously triggered the binary RAG-skip cliff; the live failure input still produces correct clarification.
- **Abort criteria:** Cognition produces broken anchors when `referents` is empty/missing (must fall back to legacy boolean cleanly); regression on the live failure input.
- **Sign-off evidence:** captured-input regression diff; manual smoke on the live failure input; `pytest` covering the L2/L3 referent path.

#### Combined E → F sign-off

- **Soak window:** 3 days (cap), justified by the project's recent-history retrieval policy (~3-day window). Sign-off requires that for 3 consecutive days, no row inside the active recent-history window of an active channel triggers the legacy fallback INFO log. Quiet channels (no traffic in the soak window) do NOT block — they are inactive by definition. Active channel set is determined from `conversation_history` `last_timestamp` per channel.
- **Sign-off evidence:** 3-day log audit summary listing per-channel fallback occurrences; cache-key bump effect observed and stable; E1+E2+E3 sub-gates all signed off.

### Checkpoint F (Cleanup completion)

- **Entry criteria:** Checkpoint E (all sub-gates) signed off; 3-day soak with no legacy-fallback log lines firing on rows inside the active recent-history window; per-referent path proven on captured input set.
- **Success metrics:** Stage 1 sanitizer deleted; `_is_directly_addressed_to_bot` deleted; `mentioned_bot` and `reply_context.reply_to_current_bot` deleted from the source-of-truth surfaces (kept ONLY where Stage 2 explicitly deferred a consumer migration, called out in commit message); boolean `needs_clarification` and friends removed; legacy `content` field stops being written; dual-field Mongo `$or` removed; `INITIALIZER_PROMPT_VERSION` final-bumped.
- **Abort criteria:** Any consumer still references the deleted symbols; tests still depend on the deleted boolean; post-deletion log audit shows fallback paths being hit on new rows.
- **Sign-off evidence:** static grep showing zero references to deleted symbols, full test pass, 48-hour post-cleanup log audit confirming no fallback firings on new rows. No metrics counters required — log greps and test output are sufficient.

## Implementation Order

### Phase A — Schema (additive, no behavior change)

Gate at end: **Checkpoint A → B**.

1. Create the `message_envelope/` module skeleton: `__init__.py` (public re-exports), `README.md` (module-level documentation), `types.py`, `protocols.py`, `factory.py`, `registry.py`, `normalizers/`, `attachment_handlers/`. Each file has a top-of-file docstring describing what it owns and which extension slot it exposes.
2. In `types.py`: define `MessageEnvelope`, `Mention`, `ReplyTarget`, `AttachmentRef`, `RawMention`, `RawReply`, `NormalizedEnvelopeFragment`, `ResolvedMention` TypedDicts.
3. In `protocols.py`: define `EnvelopeNormalizer`, `MentionResolver`, `AttachmentHandler` Protocols. Each Protocol has a complete docstring explaining intended implementations and the extension slot it represents.
4. In `factory.py`: define `build_envelope(req: ChatRequest, normalizer: EnvelopeNormalizer, resolver: MentionResolver, handlers: AttachmentHandlerRegistry) -> MessageEnvelope`. No platform-specific code in this file.
5. In `registry.py`: define `NormalizerRegistry` (keyed by `platform`) and `AttachmentHandlerRegistry` (keyed by `media_type_prefix`).
6. Extend `ConversationMessageDoc` with new optional fields (`body_text`, `addressed_to_global_user_ids`, `raw_wire_text`, `mentions`, `broadcast`). The existing `attachments` field is unchanged (already supports `media_type`, `url`, `base64_data`, `description`, `size_bytes`).
7. Extend `IMProcessState` and `CognitionState` with new optional fields (`message_envelope`, `target_addressed_user_ids`, `referents`).
8. Note: `reference_resolution_status`, `needs_clarification`, `clarification_reason` already exist on the state types — they are kept through Phase E and removed in Phase F.
9. Add new MongoDB indexes for the typed-addressing query patterns: a compound index on `(platform, platform_channel_id, addressed_to_global_user_ids, timestamp DESC)` and a regex/text index on `body_text`. Index DDL goes wherever existing indexes are declared (e.g. `db/bootstrap.py`). Indexes are idempotent — re-running setup is a no-op.
10. Existing tests must still pass with no new field populated.

### Phase B — Adapter intake

Gate at end: **Checkpoint B → C**.

10. Implement `QQEnvelopeNormalizer` in `message_envelope/normalizers/qq.py` per the `EnvelopeNormalizer` Protocol. Strip `[CQ:reply,...]`, `[CQ:at,...]`, `[CQ:image,...]`, `[CQ:face,...]` from `body_text`; populate `mentions` and `reply` typed structures. Top-of-file docstring documents that this is the only QQ-specific code in the project.
11. Implement `ImageAttachmentHandler` in `message_envelope/attachment_handlers/image.py`. The handler:
    - Takes the existing `description` from the inbound `AttachmentIn` if already populated (preserves the current image-to-text flow — Stage 2 does not change description generation).
    - When `description` is empty, the handler MAY call into the existing image-description pipeline; the call must go through a defined interface (no inline imports, no scattered logic).
    - Decides storage shape (`inline` if `size_bytes <= INLINE_ATTACHMENT_BYTE_LIMIT`, else `url_only`, else `drop` for empty/corrupt).
12. Implement `MentionResolver` default with a `user_profiles`-backed lookup. Tests inject a fixture-backed resolver.
13. Register both implementations in `message_envelope/registry.py`.
14. `service.py` calls `build_envelope(...)` at intake. The envelope is plumbed into `IMProcessState`. `user_input` is derived from `body_text` plus the textual descriptions of attachments (preserving today's append-image-description behavior). Existing scattered fields (`mentioned_bot`, `reply_context.reply_to_current_bot`, etc.) are populated *from* the envelope, not re-derived per stage.
15. Verify: adapter unit tests show wire markers stripped from `body_text`; `<@bot_id>` is in `mentions[].role="bot"`, never in `body_text`; image-description regression suite shows byte-identical descriptions vs. pre-Stage-2 baseline.

### Phase C — Storage write path

Gate at end: **Checkpoint C → D**.

17. `save_conversation` populates `body_text`, `addressed_to_global_user_ids` (user-side: derived from envelope), `raw_wire_text`, `mentions`, `broadcast` (default false). For `role=assistant` rows, `body_text` is the unrendered cognition output; `raw_wire_text == body_text` until outbound rendering exists in a later stage.
18. The embedding source pivots in this phase: `save_conversation` computes `embedding = get_text_embedding(body_text + composed_attachment_descriptions)` instead of `get_text_embedding(content)`. The composition helper lives next to `save_conversation` and is the only place that builds the embedding-source string. Legacy rows keep their original embeddings.
19. Attachment persistence honors the storage policy: `description` always written; `base64_data` written only when `storage_shape="inline"`; `url` written when present. Existing `_safe_attachment_docs` is refactored to delegate to `AttachmentHandler.storage_shape(...)` rather than hard-coding the rule.
20. Continue writing legacy `content` for back-compat. Add a read-time normalizer (in `db/conversation.py`) that computes `body_text` for legacy rows by stripping registered wire tokens — this normalizer is the single function consumed by retrieval helpers; no scattered token-stripping anywhere else in the codebase. The normalizer logs INFO per legacy-row hit it processes (see "Log-based audit" in Verification).
21. Verify: new rows in MongoDB have all envelope fields and a `body_text`-derived embedding; legacy reads through the normalizer expose a `body_text` that matches what an adapter would have produced; attachment storage shape policy holds across a captured replay set.

### Phase D — Bot-side addressing capture

Gate at end: **Checkpoint D → E**.

20. Add `target_addressed_user_ids: list[str]` to `CognitionState`. Default in `dialog_agent` to `[current_user.global_user_id]`. Add a `broadcast: bool` companion (default false).
21. `service.py` (or wherever assistant rows are saved) writes `addressed_to_global_user_ids` and `broadcast` onto the assistant row from cognition output.
22. Verify: in a synthetic group-chat scenario where the bot serves user A then user B then user A, the saved assistant rows carry the correct `addressed_to_global_user_ids`.

### Phase E — Consumer migration

Gate at end: **Checkpoint E → F**.

Phase E is staged via sub-gates E1 → E2 → E3 (see Phase Checkpoints). The numbered steps below are grouped by sub-gate.

**E1 — Consumers migrate to typed addressing:**

25. Rewrite `build_interaction_history_recent` to filter by typed fields. Implement `_legacy_subthread_fallback` with INFO logging (one log per row, deduped by row id) for rows missing `addressed_to_global_user_ids`. Log lines include `row_id`, `platform_channel_id`, and the row's `timestamp` so the soak-window audit can partition hits by channel and by inside-vs-outside the active recent-history window.
26. Migrate RAG search agents (`conversation_keyword_agent`, `conversation_search_agent`, `conversation_aggregate_agent`) to query `body_text`. Server-side filters use `$or` over `body_text` and `content` during the migration window so legacy rows remain reachable. Hits matching `content` only go through the read-time normalizer. No agent imports the normalizer directly; they consume it through a `db/conversation.py` helper.
27. Update `build_initializer_cache_key` to consume `body_text` + typed addressing. Bump `INITIALIZER_PROMPT_VERSION`.
28. Update memory consolidator to read `body_text` (clean) when summarizing user/dialog input — no schema change to `MemoryDoc`. Existing wire-poisoned memory rows are accepted as residual.
29. Migrate relevance: replace `_is_directly_addressed_to_bot` with reads of typed `addressed_to_global_user_ids`. Delete the helper at the end of E1. The relevance prompt language continues to reference `mentioned_bot` and `reply_context.reply_to_current_bot` (now derived shims); a full prompt rewrite is deferred to a follow-up plan.
30. Reduce `mentioned_bot` and `reply_context.reply_to_current_bot` to derived shims populated from the envelope. Source-of-truth for new code is the typed `mentions` list and `addressed_to_global_user_ids`. Phase F deletes the shims where consumers have migrated.

**E2 — `referents` emitted alongside boolean:**

31. Extend the decontextualizer prompt and parsing to emit `referents: [{phrase, role, status}]` on every turn. The legacy boolean `needs_clarification` continues to be emitted unchanged. Cognition still drives off the boolean during E2.
32. Capture a regression set of recent inputs from production logs (decontextualizer outputs already in production). Run the new prompt against this set on the local Gemma; verify ≥80% structural equivalence with the boolean.

**E3 — Cognition switches to `referents`:**

33. Migrate `call_judgment_core_agent` and `call_content_anchor_agent` to read `referents` as the authoritative ambiguity signal. The boolean is still emitted but no longer load-bearing.
34. Verify on the captured-input regression set: partial-vagueness inputs that previously triggered the binary RAG-skip cliff now produce narrow per-referent clarification; the live failure input still produces correct clarification.

**E exit verification:**

35. Verify across all sub-gates: log-line audit on a representative day shows the `interaction_history_legacy_fallback` INFO log fires only on rows older than the active recent-history window; RAG keyword searches no longer match `<@bot_id>` in `body_text` (read-time-normalized for legacy rows); `pytest` covers the typed-filter, sub-thread, and referents paths.

### Phase F — Cleanup

Gate at end: **Checkpoint F**.

36. After 3 consecutive days of zero legacy-fallback hits on rows inside the active recent-history window of any active channel, delete the fallback path.
37. Stop writing legacy `content` to new rows; mark `content` deprecated in schema. Drop the dual-field `$or` from server-side queries; only `body_text` is queried.
38. Delete Stage 1 sanitizer (`_sanitize_initializer_slots`, `_envelope_token_set`) and its prompt rule mirror.
39. Delete the boolean `needs_clarification` field and its decontextualizer/cognition references (kept aliased with `referents` through Phase E).
40. Delete `_is_directly_addressed_to_bot` from `relevance_agent.py` (already done in E1, re-verify in F that no caller remains).
41. Delete `mentioned_bot` and `reply_context.reply_to_current_bot` derived shims where consumers have fully migrated. Where Stage 2 deferred a consumer migration (e.g. relevance prompt language), the shim stays — call out in commit message.
42. Final `INITIALIZER_PROMPT_VERSION` bump to flush any rows still encoding pre-Stage-2 assumptions.

## Progress Checklist

- [x] Stage 0 — legacy bridge containment
  - Deliverable: `build_interaction_history_recent` fails closed when no current-user anchored slice exists.
  - Files: `src/kazusa_ai_chatbot/utils.py`, `tests/test_dialog_agent.py`.
  - Verify: `python -m py_compile src/kazusa_ai_chatbot/utils.py tests/test_dialog_agent.py`; `python -m pytest tests/test_dialog_agent.py -q`.
  - Evidence: focused regression added for other-user + assistant-only group window; `tests/test_dialog_agent.py` passed on 2026-04-30.
  - Handoff: next stage starts at Phase A.
  - Sign-off: Codex / 2026-04-30.
- [ ] Stage 0.5 — remaining Stage 1 sanitizer safety net
  - Deliverable: `_envelope_token_set`, `_sanitize_initializer_slots`, RAG initializer prompt rule, `platform_bot_id` context plumbing, and cache read/write sanitizer coverage are implemented per the Stage 1 plan.
  - Files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`, `src/kazusa_ai_chatbot/rag/cache2_policy.py` if a prompt-version bump is required, and focused tests.
  - Verify: Stage 1 sanitizer tests; `python -m py_compile` over modified modules; focused RAG initializer/cache replay tests showing envelope tokens cannot become search keywords.
  - Evidence: record sanitizer test output, prompt-version decision, and before/after regression for the live failure input.
  - Handoff: next stage starts at Phase A only after this safety net is signed off.
  - Sign-off: `<agent/date>`.
- [ ] Stage A — schema and module contract
  - Deliverable: `message_envelope/` public contract, TypedDicts, Protocols, registries, additive schema/state fields.
  - Files: `src/kazusa_ai_chatbot/message_envelope/`, `src/kazusa_ai_chatbot/db/schemas.py`, `src/kazusa_ai_chatbot/state.py`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`, tests.
  - Verify: `python -m py_compile` over modified modules; `pytest tests/test_message_envelope.py -q`; `pytest tests/test_envelope_module_boundaries.py -q`.
  - Evidence: record Checkpoint A -> B compile/test output and module docstring audit.
  - Handoff: next stage starts at Phase B.
  - Sign-off: `<agent/date>`.
- [ ] Stage B — adapter intake and envelope construction
  - Deliverable: platform normalizers and attachment handlers populate `MessageEnvelope`; `IMProcessState.user_input` derives from `body_text` plus attachment descriptions.
  - Files: `message_envelope/normalizers/`, `message_envelope/attachment_handlers/`, `message_envelope/factory.py`, `message_envelope/registry.py`, `src/kazusa_ai_chatbot/service.py`, adapter files, tests.
  - Verify: adapter envelope tests; image-description regression suite; manual smoke for the live failure input.
  - Evidence: record Checkpoint B -> C adapter fixture diff, image regression output, and mention-resolution telemetry.
  - Handoff: next stage starts at Phase C.
  - Sign-off: `<agent/date>`.
- [ ] Stage C — storage write path and legacy read normalization
  - Deliverable: new conversation rows save `body_text`, `addressed_to_global_user_ids`, `raw_wire_text`, `mentions`, `broadcast`, and attachment storage policy fields while legacy `content` remains readable.
  - Files: `src/kazusa_ai_chatbot/db/schemas.py`, `src/kazusa_ai_chatbot/db/conversation.py`, `src/kazusa_ai_chatbot/service.py`, attachment tests.
  - Verify: `pytest tests/test_conversation_history_envelope.py -q`; `pytest tests/test_attachment_handler.py -q`; MongoDB sample-row inspection in a safe test/dev environment.
  - Evidence: record Checkpoint C -> D sample rows, replay tests, insertion-latency comparison, and attachment field audit.
  - Handoff: next stage starts at Phase D.
  - Sign-off: `<agent/date>`.
- [ ] Stage D — bot-side addressing capture
  - Deliverable: cognition/dialog output carries `target_addressed_user_ids` and `broadcast`; assistant rows persist typed addressee fields.
  - Files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`, `src/kazusa_ai_chatbot/nodes/dialog_agent.py`, `src/kazusa_ai_chatbot/service.py`, tests.
  - Verify: synthetic group-chat regression where bot serves user A -> user B -> user A and saved assistant rows carry the correct addressees.
  - Evidence: record Checkpoint D -> E regression output and `target_addressed_user_ids` cardinality telemetry.
  - Handoff: next stage starts at Phase E.
  - Sign-off: `<agent/date>`.
- [ ] Stage E — consumer migration
  - Deliverable: `build_interaction_history_recent`, RAG search agents, cache key, memory consolidator, decontextualizer/referents, and relevance consume typed envelope/addressing fields with logged legacy fallback.
  - Files: `src/kazusa_ai_chatbot/utils.py`, `src/kazusa_ai_chatbot/rag/conversation_*.py`, `src/kazusa_ai_chatbot/rag/cache2_policy.py`, `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`, cognition L2/L3 modules, relevance agent, tests.
  - Verify: `pytest tests/test_build_interaction_history_recent.py -q`; `pytest tests/test_rag_search_body_text.py -q`; `pytest tests/test_rag_initializer_cache2.py -q`; `pytest tests/test_decontexualizer_referents.py -q`; live LLM tests one at a time.
  - Evidence: record Checkpoint E -> F telemetry showing legacy fallback trend, cache hit-rate snapshot, and captured-input referent regression.
  - Handoff: next stage starts at Phase F only after the soak gate is satisfied.
  - Sign-off: `<agent/date>`.
- [ ] Stage F — cleanup and removal of transitional paths
  - Deliverable: delete legacy fallbacks, Stage 1 sanitizer path if present, `_is_directly_addressed_to_bot`, boolean clarification path, and deprecated legacy writes after telemetry gates pass.
  - Files: all Phase F cleanup files listed above plus tests/static greps.
  - Verify: static greps in this plan; full test pass; 48-hour post-cleanup telemetry.
  - Evidence: record Checkpoint F grep output, full test output, and production telemetry snapshot.
  - Handoff: plan complete.
  - Sign-off: `<agent/date>`.

## Verification

### Deterministic Tests

- `pytest tests/test_message_envelope.py -q` (new)
  - QQ adapter strips `[CQ:reply,...]`, `[CQ:at,...]`, `[CQ:image,...]`, `[CQ:face,...]` from `body_text`.
  - Discord adapter strips `<@id>`, `<@&id>`, `<#id>`, `<:emoji:id>` from `body_text`.
  - Mentions extracted as typed list with resolved `global_user_id` where the user is known.
  - `addressed_to_global_user_ids` for user message: union of mentions (role=bot|user) + reply target author.
  - Reply target derivation flag is populated correctly: `platform_native` or `leading_mention` (Stage 2 has no `inferred_thread` value).

- `pytest tests/test_conversation_history_envelope.py -q` (new)
  - New rows save with `body_text`, `addressed_to_global_user_ids`, `raw_wire_text`, `mentions`.
  - Legacy rows (only `content`) yield a normalized `body_text` at read time.
  - `broadcast=true` and `addressed_to_global_user_ids=[]` are distinguishable.

- `pytest tests/test_attachment_handler.py -q` (new)
  - `ImageAttachmentHandler` returns the inbound `description` byte-identical when one is supplied (regression on the image-to-text path).
  - Storage shape returns `inline` when `size_bytes <= INLINE_ATTACHMENT_BYTE_LIMIT` and `url_only` otherwise.
  - `AttachmentRef` written to `ConversationMessageDoc.attachments` retains `media_type`, `url`, `description`, and (when shape=`inline`) `base64_data`.
  - Cognition prompts and RAG agents see only `description`; no agent reaches for `base64_data` in Stage 2.

- `pytest tests/test_envelope_module_boundaries.py -q` (new)
  - Static import-graph check: no module outside `message_envelope/` imports from `message_envelope/normalizers/` or `message_envelope/attachment_handlers/` (consumers depend on protocols, not implementations).
  - `service.py` imports only from `message_envelope` and `message_envelope.factory`; no per-platform import.

- `pytest tests/test_build_interaction_history_recent.py -q` (extend existing)
  - Multi-user group scenario: bot serves A, B, A. History scoped to A includes only A↔bot turns whose bot side has `A` in `addressed_to_global_user_ids`.
  - Legacy fallback: rows without `addressed_to_global_user_ids` go through the heuristic and fire INFO log exactly once per row.
  - `broadcast=true` bot rows surface in every user's filtered history.

- `pytest tests/test_rag_search_body_text.py -q` (new)
  - `conversation_keyword_agent` searching `<@bot_id>` returns zero hits even when matching rows exist with that token in `raw_wire_text`.
  - Legacy row with only `content` still searchable via read-time normalizer.

- `pytest tests/test_rag_initializer_cache2.py -q` (extend)
  - Old Stage 1 sanitizer tests removed.
  - New cache key for the same `body_text` + addressing is stable across cosmetic wire-form variations.
  - Old `INITIALIZER_PROMPT_VERSION` rows are misses after the bump.

- `pytest tests/test_decontexualizer_referents.py -q` (new)
  - "这些是什么意思" with no resolvable referent returns `referents=[{phrase: '这些', role: 'object', status: 'unresolved'}]`.
  - Same phrase with reply excerpt containing concrete object returns `referents=[{phrase: '这些', role: 'object', status: 'resolved'}]`.
  - Mixed case ("他上次说的那些关于X的话") returns one resolved + one unresolved entry, and cognition produces a narrow clarification anchor without skipping retrieval for X.

### Compile

- `python -m py_compile` over each modified module.

### Static Greps (post-Phase F)

- `rg "<@\\\\d+>" src/kazusa_ai_chatbot` — no hits in non-adapter code.
- `rg "_sanitize_initializer_slots|_envelope_token_set" src tests` — zero hits.
- `rg "needs_clarification" src/kazusa_ai_chatbot` — zero hits in load-bearing code (test fixtures may keep the alias for one release).

### Live LLM Tests (one at a time)

- Decontextualizer per-referent contract on local Gemma — verify the structured field is populated for each test scenario.
- Group-chat regression: bot rapidly serves two users; each user's persona context contains only their subthread.
- Reply via @mention without platform reply feature: bot recognises it as addressed to itself.

### Manual Smoke

- Reproduce the Stage 1 incident input (`[Reply to message] <@bot> 你知道这些是什么意思么？`); confirm `body_text` strips both markers, RAG initializer never sees the bot mention, and clarification is emitted via per-referent structure.
- In a real QQ group with two active users, confirm that the bot's responses to user A do not appear in user B's `chat_history_recent` slice for the next turn.

### Log-based audit (replaces fabricated telemetry)

The codebase does not currently expose a metrics counter stack. Sign-off evidence comes from log lines and DB sampling, not counters.

- **Legacy-fallback audit:** the rewritten `build_interaction_history_recent` emits an INFO log per legacy-row fallback hit, including `row_id`, `platform_channel_id`, and the row's `timestamp`. Sign-off scripts grep these log lines over the soak window and report:
  - Count of fallback hits per channel.
  - Count of fallback hits where `row.timestamp` is inside the active recent-history window vs. outside.
  - Active-channel set (channels with any traffic during the window).
  Phase F gate is satisfied when the count of in-window hits across active channels is zero for 3 consecutive days.
- **Read-time normalizer audit:** `db/conversation.py` normalizer emits an INFO log per legacy-row hit it processes. Same grep-and-report pattern; Phase F gate watches the trend (decreasing).
- **Reply-derivation audit:** the envelope factory logs `reply.derivation` per inbound row at INFO. Phase B/C uses log greps to check the distribution looks reasonable (e.g. `platform_native` dominates on QQ where the platform supports replies; `leading_mention` dominates where it doesn't).
- **Cache hit rate:** the existing Cache 2 logging is sufficient; observe before/after key bump from production logs without a new metric.

A future amendment may replace this with a real metrics stack; until that lands, do not add counter calls that will go nowhere.

## Acceptance Criteria

This plan is complete when:

- All adapters produce `MessageEnvelope` through `EnvelopeNormalizer` implementations. No downstream stage parses wire markers.
- `ConversationMessageDoc` carries `body_text`, `addressed_to_global_user_ids` (symmetric across user and bot rows), `raw_wire_text`, `mentions`, `broadcast`.
- Attachments are persisted with `media_type`, `description`, and either `base64_data` (when inline) or `url`. RAG and cognition continue to consume only `description` (image-to-text workflow preserved). Future direct-binary consumption can be added without a schema change.
- All new modules expose protocol-typed interfaces. Static import-graph check confirms consumers do not reach into implementations.
- Each new module has a top-of-file docstring stating its purpose, inputs, outputs, and extension slot. `message_envelope/README.md` exists and documents the protocols and registries.
- Each phase has signed-off checkpoint evidence in "Execution Evidence" before the next phase begins.
- `build_interaction_history_recent` filters by typed addressee. Legacy fallback is logged and trending to zero.
- RAG search agents query `body_text` only.
- `build_initializer_cache_key` keys off `body_text` + typed addressing.
- Decontextualizer emits structured `referents`; cognition consumes them for narrow clarification.
- Stage 1 sanitizer and its prompt mirror are deleted in Phase F after typed-envelope consumers are verified.
- New conversation rows compute their embedding from `body_text` + composed attachment descriptions; legacy rows are not re-embedded (deferred).
- Mongo queries during the migration window use a `body_text`-or-`content` `$or`; Phase F removes the `content` branch.
- DM addressing defaults are deterministic: inbound user rows have `addressed_to_global_user_ids=[bot_id]` and `reply_to_current_bot=True`; outbound bot rows have `addressed_to_global_user_ids=[current_user_id]`.
- For `role=assistant` rows, `body_text` is the unrendered cognition output and `raw_wire_text` carries the on-the-wire form (equal in Stage 2 because outbound rendering is not implemented).
- `mentioned_bot` and `reply_context.reply_to_current_bot` are derived shims after Phase E; deleted in Phase F where consumers have migrated.
- Outbound mention/reply rendering is explicitly out of scope; documented in Deferred as a Stage 3 candidate.
- 3-day log audit shows zero legacy-fallback hits on rows inside the active recent-history window of any active channel.
- Live LLM and manual smoke tests pass.

## Rollback / Recovery

- Phase A: revert is trivial; new fields are unused.
- Phase B/C: revert by restoring adapter and save-path code; new schema fields are tolerated as ignored extras on read.
- Phase D: revert by stopping cognition emission; assistant rows fall back to empty `addressed_to_global_user_ids`, treated as legacy by Phase E consumers.
- Phase E: revert per-consumer; each consumer carries its own legacy fallback so reverting one does not block others.
- Phase F: only run after Phase E telemetry is clean. If a regression appears, restore the deleted fallback from VCS and re-enter Phase E.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Adapter bug strips legitimate content along with wire markers | Per-adapter test fixtures with adversarial inputs (markers embedded in user text, escaped markers, repeated markers) | Adapter unit tests; compare `body_text` length distribution before/after rollout. |
| Cognition emits wrong `target_addressed_user_ids` and history retrieval narrows incorrectly | Default to current-user UUID; require any non-default value to come from explicit cognition output | Group-chat regression test; telemetry on default-vs-override rate. |
| Legacy fallback in `build_interaction_history_recent` masks bugs in the new typed path | Fallback logs INFO with row id; counter watched in Phase E; deleted in Phase F | Fallback counter must reach zero before Phase F. |
| Storage migration breaks read for live traffic | Phase A→C are additive only; legacy `content` continues to be written through Phase E | Read-side compatibility test exercising mixed-shape collections. |
| Cache key bump triggers cold-cache load spike | Bump during low-traffic window; warm-up worker if needed | Pre-rollout cache-load simulation. |
| Decontextualizer migration to `referents` confuses local Gemma | Ship structured contract with both forms during transition; alias boolean for one release | Live LLM tests; telemetry on `referents` field presence. |
| Stage 1 sanitizer is deleted prematurely | Phase F gate explicitly requires Phase E telemetry clean | PR reviewers verify Phase E success criteria before approving Phase F. |
| Multi-bot deployments resolve mentions incorrectly | Resolution goes through user profile store; unresolved mention falls back to platform_id with `global_user_id=""` | Adapter tests with multi-bot fixtures. |
| Broadcast bot replies disappear from per-user history | Explicit `broadcast: true` flag in the row; consumer treats broadcasts as visible to all | Group-chat test with a broadcast turn. |
| Attachment description pipeline regresses during refactor | Phase B regression test compares descriptions byte-for-byte against pre-Stage-2 baseline; `ImageAttachmentHandler` preserves inbound description when supplied | Image-description regression suite at Checkpoint B → C. |
| `base64_data` size growth blows up MongoDB row sizes | `INLINE_ATTACHMENT_BYTE_LIMIT` enforced by `AttachmentHandler.storage_shape`; large attachments persist `url` only | Storage-policy test in Phase C; row-size telemetry in Checkpoint C → D. |
| Future modality added in a way that breaks consumers | Module boundary test enforces that consumers depend on protocols, not implementations | Import-graph test in CI; rejected at module-boundaries test gate. |
| Vector search returns wire-poisoned legacy rows during the migration window | Embedding source pivots to `body_text` for new rows; legacy rows accepted as residual; 3-day soak cap bounds visible impact (older rows are RAG territory, not history-window territory) | Captured-query semantic-recall regression at Checkpoint C → D and again at E1 sign-off. |
| Mongo `$or` over `body_text` and `content` becomes a slow path | Add index on `body_text` in Phase A; index on `(addressed_to_global_user_ids, ...)` for typed filter; revert dual-field branch in Phase F | Insertion latency comparison at C → D; query-plan inspection at E1. |
| Outbound mention/reply rendering remains absent and creates user confusion | Explicitly documented as a known gap; Stage 3 will add a symmetric `EnvelopeRenderer` Protocol; Stage 2 stores typed addressing on bot rows so Stage 3 can read it | Documented in Deferred and in the "Outbound rendering" risk row of any user-facing release notes. |
| Local Gemma fails to emit `referents` reliably and E2 stalls | Sub-gate E2 is a hard pass/fail on a captured regression set; if it fails, E3 does not start. The boolean stays load-bearing until E2 passes | Captured-input regression at E2 sign-off; explicit ≥80% structural equivalence threshold. |
| `mentioned_bot` and `reply_context.reply_to_current_bot` get used as new sources of truth post-Stage-2 | Mandatory Rule pins them as derived shims after Phase E; Phase F deletes them where consumers have migrated; commit message must call out any retained shim | Code review at E1 sign-off; static grep at Phase F. |
| DM rows look like legacy rows because addressing is "trivial" | DM defaults pinned in Mandatory Rules: `addressed_to_global_user_ids=[bot_id]`, `reply_to_current_bot=True`; envelope factory applies these deterministically | Unit test for DM envelope path at Phase B. |
| Phase F gate stalls because of low-traffic channels | Soak gate scoped to "active channels with traffic during the window"; quiet channels do not block | Active-channel set computed from `last_timestamp` per channel; documented in Phase F sign-off evidence. |

## Stage 1 deletion / consolidation checklist (Phase F)

To be checked off when Phase F runs. These items apply because the approved migration path requires the remaining Stage 1 sanitizer safety net to ship before Stage 2 Phase A.

- [ ] `_sanitize_initializer_slots` deleted from `persona_supervisor2_rag_supervisor2.py`.
- [ ] `_envelope_token_set` deleted.
- [ ] Prompt rule "do not search these as keywords" removed from `_INITIALIZER_PROMPT`.
- [ ] Tests for the sanitizer deleted (cache-replay test stays, repurposed against the new key scheme).
- [ ] `_is_directly_addressed_to_bot` deleted from `nodes/relevance_agent.py`; relevance consumes `addressed_to_global_user_ids` directly.
- [ ] `mentioned_bot` and `reply_context.reply_to_current_bot` deleted from source-of-truth surfaces. Where Stage 2 deferred a consumer migration (e.g. relevance prompt language), the shim is retained and the commit message calls it out explicitly.
- [ ] Dual-field `$or` over `body_text` and `content` removed from `db/conversation.py`; only `body_text` is queried server-side.
- [ ] `INITIALIZER_PROMPT_VERSION` bumped to flush any rows that still encode the old key scheme. Increment from whatever the current value is at Phase F time (currently `initializer_prompt:v4`).
- [ ] `needs_clarification` boolean field removed from `GlobalPersonaState` after one release of alias coexistence with `referents`.
- [ ] Decontextualizer prompt simplified to drop the boolean field instructions; `referents` becomes the only ambiguity output.
- [ ] Cognition L2/L3 prompts switched from boolean consumption to per-referent consumption; legacy boolean reads removed.

## Execution Evidence

Each checkpoint's sign-off evidence is recorded under its own subheading. A phase does not enter execution until the prior checkpoint subheading is filled.

### Checkpoint A → B sign-off

- Compile results:
- Test results (`tests/test_message_envelope.py`, `tests/test_envelope_module_boundaries.py`):
- Module docstring audit:
- `message_envelope/README.md` link:

### Checkpoint B → C sign-off

- Adapter unit-test diff (live failure input shows `<@bot>` removed from `body_text`):
- Image-description regression suite output:
- `mentions` resolution-rate telemetry:
- Manual smoke (live failure input):

### Checkpoint C → D sign-off

- MongoDB sample-row inspection (envelope fields present):
- Legacy-row replay test:
- Conversation-history insertion latency comparison:
- Attachment field-presence audit:

### Checkpoint D → E sign-off

- Group-chat regression test (A→B→A scenario):
- `target_addressed_user_ids` cardinality telemetry:

### Checkpoint E → F sign-off

#### E1 — Consumers migrated

- Log-grep summary (legacy-fallback hits per channel; in-window vs out-of-window split):
- Cache hit-rate observation from production logs before/after key bump:
- `pytest tests/test_rag_search_body_text.py -q` and `pytest tests/test_build_interaction_history_recent.py -q` results:

#### E2 — `referents` emitted alongside boolean

- Captured-input regression report (≥80% structural equivalence):
- Local Gemma latency comparison before/after extended decontextualizer prompt:
- `pytest tests/test_decontexualizer_referents.py -q` results:

#### E3 — Cognition switches to `referents`

- Captured-input regression diff (per-referent vs binary clarification):
- Manual smoke on the live failure input:
- `pytest` covering the cognition L2/L3 referent path:

#### Combined E → F gate

- 3-day log audit: zero legacy-fallback hits inside the active recent-history window across active channels:
- Active-channel set used for the audit (computed from `last_timestamp` per channel):

### Checkpoint F sign-off

- Static grep showing zero references to deleted symbols:
- Full test pass:
- 48-hour post-cleanup log audit confirming no fallback firings on new rows:

### Cross-cutting evidence

- Static grep results:
- Test results:
- Compile results:
- Live LLM results:
- Manual smoke:
- Telemetry snapshots:
- Changed files:
