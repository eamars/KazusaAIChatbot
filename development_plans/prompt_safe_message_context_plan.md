# prompt safe message context plan

## Summary

- Goal: Prevent raw transport/storage envelope payloads, especially inline multimedia `base64_data`, from entering LLM prompt contexts while preserving typed envelope storage and current multimedia summary behavior.
- Plan class: large
- Status: completed
- Overall cutover strategy: compatible for storage and database rows, bigbang for prompt-facing current-message payloads in one release.
- Highest-risk areas: accidentally removing media summaries from cognition/RAG, hiding binary leakage with ad hoc RAG-only sanitizers, changing adapter/storage semantics, and failing to persist new attachment descriptions for future retrieval.
- Acceptance criteria: no prompt-facing LLM payload serializes `message_envelope.attachments[*].base64_data`, `raw_wire_text`, or attachment URLs; multimedia summaries remain available to relevance/decontextualizer/RAG/cognition; new media rows persist generated descriptions when available; existing database rows are not backfilled and may keep empty descriptions.

## Context

Stage 2 typed message envelopes correctly separated `body_text`, typed addressing, attachments, and `raw_wire_text` at the adapter/service contract. The latest failure shows a remaining boundary problem: the same full envelope object is used both as a storage/audit object and as prompt-facing context.

The failure path was:

- Service resolves a full `MessageEnvelope` and puts it into graph state.
- `multimedia_descriptor_agent` reads inline image `base64_data` and generates a text description.
- `relevance_agent` appends the description to `user_input`.
- The full unresolved storage envelope remains in `state["message_envelope"]`.
- Decontextualizer and RAG pass `message_envelope` into JSON prompt payloads.
- RAG initializer serializes the full `context`, including `attachments[*].base64_data`, causing a context-window explosion.

Database inspection confirmed the failed row had `body_text=""`, one inline image attachment, empty attachment description, and about 1.13 MB of inline base64. The stored `conversation_episode_state` row was compact and not the cause.

Owner decision for this plan: do not backfill existing database rows. Legacy rows may keep empty attachment descriptions. This plan only protects future prompt payloads and future rows.

## Mandatory Rules

- Do not add a RAG-only sanitizer as the primary fix. The bug class is broader than RAG.
- Do not teach brain prompts or RAG prompts to parse platform wire syntax or binary payloads.
- Do not change Stage 2's typed envelope storage contract except where this plan explicitly requires adding generated descriptions to new rows.
- Do not remove binary preservation from storage solely to fix prompts. Storage policy and prompt policy are separate.
- Adapters are responsible only for intake sanitization (parsing platform wire format into a typed `MessageEnvelope`). Adapters must not generate descriptions, summarize media, run inference, or perform any other processing beyond intake.
- Do not backfill existing `conversation_history` rows. Do not rerun image description over historical media. Do not re-embed historical rows.
- Empty `attachments[*].description` on legacy rows is accepted behavior.
- All LLM-facing payloads must be constructed from prompt-safe projections, not from raw storage envelopes.
- Prompt-safe projections must be built by explicit whitelist: only fields declared in `PromptMessageContext` may appear in the serialized output. Storage-only fields (`raw_wire_text`, `base64_data`, attachment URLs, binary locators) cannot reach prompts because they are not in the whitelist.
- Prompt-safe projections must use semantic descriptors, not raw size measurements. For example, use `summary_status="available"` or `"unavailable"`, not `base64_length=1129908`.
- Keep adapter-specific parsing in adapters. Brain modules must not import platform normalizers or parse CQ/Discord wire markers.
- Follow project Python style: imports at top, narrow `try` blocks, specific exception classes, complete docstrings for public helpers, and no unrelated formatting churn.
- When editing `.py` files that contain CJK prompt content, follow CJK safety: preserve quoting safely and run syntax validation.
- If a prompt template is edited, run a prompt-render or runtime test in addition to `py_compile`.
- Do not increase response-path LLM call count.
- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Every required plan reread must be recorded in `Execution Evidence` with the trigger, timestamp, and next stage/action.

## Must Do

- Create a prompt-safe current-message projection contract for LLM payloads.
- Migrate decontextualizer and RAG initializer context to consume the prompt-safe projection instead of raw `message_envelope`.
- Ensure RAG cache-key construction no longer requires a raw `message_envelope` object.
- Preserve typed addressing, reply context, mentions, and attachment summaries in the prompt-safe projection.
- Ensure generated multimedia descriptions are attached to the in-flight semantic projection before relevance/decontextualizer/RAG/cognition need them.
- Persist generated attachment descriptions for new user message rows when available, without backfilling old rows.
- Add deterministic size caps and tests proving a large inline base64 image cannot enter text prompts.
- Add static and runtime checks covering the exact failure class.

## Deferred

- Do not backfill `conversation_history.attachments.description`.
- Do not re-embed legacy conversation rows.
- Do not change the vector embedding model or embedding dimensions.
- Do not redesign the multimedia descriptor prompt structurally. A length-budget hint may be added so generated descriptions stay within `MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS`; broader prompt redesign is out of scope.
- Do not move platform-specific attachment fetching out of adapters unless required by the prompt-safe projection contract.
- Do not add description summaries for the *replied-to* message (i.e. summaries of attachments inside `reply` context). `PromptReplyContext` carries `excerpt` only; reply-target attachment summaries remain future work.
- Do not add direct image modality to RAG/cognition. Direct-modality consumption remains future work.
- Do not remove `base64_data` from storage across the board. Storage policy may be tightened separately, but it is not the prompt-safety boundary.
- Do not rewrite unrelated cognition, dialog, consolidation, or conversation-progress behavior.
- Do not add an LLM-based summarizer to compress prompt-safe content at projection time. Length control happens at descriptor generation time and via deterministic truncation; no new response-path LLM call is introduced.

## Cutover Policy

| Area | Policy | Instruction |
|---|---|---|
| Prompt-facing current-message payloads | bigbang | All prompt JSON payloads touched by this bug must switch to prompt-safe projection in one implementation. Partial migration is unsafe. |
| Conversation storage | compatible | Existing schema remains valid. New rows may gain descriptions; old rows may remain empty. |
| Legacy database rows | compatible | No backfill, no deletion, no re-embedding. Readers tolerate missing descriptions. |
| RAG initializer cache keys | compatible | No cache version bump. The initializer cache signature reads only `body_text`, `addressed_to_global_user_ids`, and `broadcast` (see `_initializer_context_signature`); these values are preserved verbatim by the projection. Existing cached strategies remain valid. |
| Adapter request contract | compatible | `/chat` still requires `message_envelope`; adapters perform intake sanitization only and must not generate descriptions, summarize, or run any processing before sending. |
| Tests | bigbang | Tests must assert no binary fields enter prompt payloads before completion. |

## Agent Autonomy Boundaries

- The agent may choose local helper mechanics only when they preserve the contracts in this plan.
- The agent must not introduce alternate migration strategies, compatibility layers, prompt-side binary filters, or extra features.
- The agent must not perform unrelated cleanup, dependency upgrades, broad refactors, or prompt rewrites outside the files listed here.
- If code and plan disagree, preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

There are two separate representations of the current message:

1. Storage/audit envelope:
   - Existing `MessageEnvelope`.
   - May contain `raw_wire_text`.
   - May contain `attachments[*].base64_data` according to storage policy.
   - Used for persistence, deterministic addressing, audit, and future direct-modality work.

2. Prompt-safe message context:
   - New bounded projection used in LLM prompt JSON.
   - Contains clean `body_text`, typed addressing, compact reply target, compact mentions, and attachment summaries.
   - Never contains `raw_wire_text`, attachment URLs, `base64_data`, binary bytes, or binary locators.
   - Exposes unavailable summaries explicitly with a short semantic label.

LLM stages receive the prompt-safe context. Deterministic code may still read the full storage envelope when it needs typed addressing or persistence metadata.

## Prompt-Safe Context Contract

Create a new module:

```text
src/kazusa_ai_chatbot/message_envelope/prompt_projection.py
```

Public constants:

```python
MAX_PROMPT_BODY_TEXT_CHARS = 2000
MAX_PROMPT_REPLY_EXCERPT_CHARS = 500
MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS = 800
MAX_PROMPT_ATTACHMENTS = 4
MAX_PROMPT_MESSAGE_CONTEXT_CHARS = 5000  # serialized JSON length, including keys/quotes
```

Per-field caps are upper bounds, not targets. The serialized JSON length is the only enforced ceiling; see "Overflow Degradation Order" below.

Public TypedDicts may live in this module or in `message_envelope/types.py` if the codebase pattern prefers central type exports:

```python
class PromptAttachmentSummary(TypedDict):
    media_kind: str
    description: str
    summary_status: Literal["available", "unavailable"]


class PromptReplyContext(TypedDict, total=False):
    platform_message_id: str
    platform_user_id: str
    global_user_id: str
    display_name: str
    excerpt: str
    derivation: str


class PromptMentionContext(TypedDict, total=False):
    platform_user_id: str
    global_user_id: str
    display_name: str
    entity_kind: MentionEntityKind


class PromptMessageContext(TypedDict):
    body_text: str
    addressed_to_global_user_ids: list[str]
    broadcast: bool
    mentions: list[PromptMentionContext]
    attachments: list[PromptAttachmentSummary]
    reply: NotRequired[PromptReplyContext]
```

`PromptMessageContext` is the structural replacement for `MessageEnvelope` in prompt JSON. It does **not** carry a concatenated semantic-prose field; the existing state-level `user_input` (post-relevance) continues to play that role and is passed alongside the projection in prompt payloads. This keeps a single source of truth per concept: `user_input` for prose, `prompt_message_context` for structure.

Public functions:

```python
def project_prompt_message_context(
    *,
    message_envelope: MessageEnvelope,
    multimedia_input: list[Mapping[str, object]] | None = None,
) -> PromptMessageContext:
    """Build a bounded LLM-safe current-message context."""


def assert_prompt_message_context_safe(payload: Mapping[str, object]) -> None:
    """Raise ValueError if any leaf key is outside the PromptMessageContext whitelist."""


class PromptContextTooLargeError(ValueError):
    """Raised when the serialized projection cannot be reduced under the global cap."""
```

Projection rules:

- `body_text` comes from `message_envelope["body_text"]`, truncated at `MAX_PROMPT_BODY_TEXT_CHARS` with an ellipsis suffix. User-verbatim content is never paraphrased.
- Attachment descriptions prefer `multimedia_input[*].description` produced by `multimedia_descriptor_agent`; fall back to `message_envelope.attachments[*].description`; otherwise use `summary_status="unavailable"` and `description=""`.
- `media_kind` is a semantic label such as `"image"`, `"audio"`, `"video"`, or `"file"`, derived from MIME prefix.
- Mentions omit `raw_text`.
- Reply omits raw wire markers; `excerpt` is truncated at `MAX_PROMPT_REPLY_EXCERPT_CHARS` with an ellipsis suffix. Reply-target attachment summaries are not surfaced (deferred).
- Description length is bounded primarily at descriptor generation time. The descriptor prompt receives a length-budget hint of `MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS`. On read, descriptions exceeding the cap are deterministically suffix-trimmed at the nearest sentence boundary (no LLM call).
- The safety assertion is whitelist-based: it walks the payload and raises `ValueError` if any key at any level is not a member of the declared `PromptMessageContext` / `PromptAttachmentSummary` / `PromptMentionContext` / `PromptReplyContext` schemas. This is defense-in-depth; whitelist-by-construction in `project_prompt_message_context` is the primary guarantee.

Overflow Degradation Order:

When `len(json.dumps(projection, ensure_ascii=False)) > MAX_PROMPT_MESSAGE_CONTEXT_CHARS`, apply in order until under cap:

1. Drop attachments beyond the first 2 (most chats include at most 1–2 media items; preserve ordering).
2. Halve the per-attachment description cap and re-trim each remaining description.
3. Truncate `body_text` to 50% of `MAX_PROMPT_BODY_TEXT_CHARS`.
4. Truncate `reply.excerpt` to 50% of `MAX_PROMPT_REPLY_EXCERPT_CHARS`.
5. If still over cap, raise `PromptContextTooLargeError`. Addressing fields (`addressed_to_global_user_ids`, `broadcast`, `mentions`) are never reduced; they are deterministically tiny and load-bearing for routing.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Fix layer | Add a prompt-safe projection boundary | Prevents the same failure in decontextualizer, RAG, cognition, and future prompt calls. |
| Storage | Keep full envelope storage contract | Binary preservation is useful for audit/future direct modality; prompts are the unsafe consumer. |
| Legacy DB rows | No backfill | Owner explicitly accepted empty legacy descriptions. |
| New media descriptions | Persist when available | Future retrieval and embeddings should use summaries without needing historical backfill. |
| LLM call count | No new response-path calls | The descriptor already exists; this plan only changes data flow. |
| Cache key | Key off prompt-safe semantic fields | Cache key should not depend on raw envelope shape or binary storage fields. |
| Prompt data shape | Semantic labels over raw metrics | Local LLM should not infer meaning from byte lengths or storage shapes. |

## Change Surface

### Create

- `src/kazusa_ai_chatbot/message_envelope/prompt_projection.py`
  - Owns prompt-safe current-message projection and safety assertion.
- `tests/test_prompt_message_context.py`
  - Unit coverage for projection, caps, forbidden-field rejection, and failure fixture shape.

### Modify

- `src/kazusa_ai_chatbot/message_envelope/__init__.py`
  - Export the prompt-safe projection API if needed by service/nodes.
- `src/kazusa_ai_chatbot/message_envelope/types.py`
  - Add prompt context TypedDicts if not colocated in the new module.
- `src/kazusa_ai_chatbot/state.py`
  - Add `prompt_message_context: NotRequired[PromptMessageContext]` to relevant graph state TypedDicts.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
  - Add prompt context to `GlobalPersonaState` and `CognitionState` only where needed.
- `src/kazusa_ai_chatbot/service.py`
  - Build/update prompt-safe context after multimedia description and before persona/RAG prompts.
  - Persist generated attachment descriptions for new current-turn rows when available.
- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  - Preserve descriptor output and avoid reintroducing raw attachment fields into prompt text.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - Replace prompt input `message_envelope` with `prompt_message_context`.
  - Update prompt wording to refer to the visible input field name.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Pass prompt-safe context to RAG and cognition state.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
  - RAG initializer context must use prompt-safe context, not raw `message_envelope`.
- `src/kazusa_ai_chatbot/nodes/cognition_l1.py`, `cognition_l2.py`, `cognition_l3.py`, `dialog_agent.py`
  - Audit each `json.dumps(...) → HumanMessage` site. If any payload references `state["message_envelope"]`, `user_multimedia_input`, or other raw envelope fields, replace with `prompt_message_context`. If audit confirms no envelope/multimedia leakage, no edits required; record the audit result in execution evidence.
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`
  - Update `_initializer_context_signature` to read `body_text`, `addressed_to_global_user_ids`, `broadcast` from the prompt-safe projection instead of `message_envelope`.
  - Do not bump `INITIALIZER_PROMPT_VERSION` or any other cache version: the signature inputs are preserved verbatim, so existing cached strategies remain valid.
- `src/kazusa_ai_chatbot/db/conversation.py`
  - Add a focused update helper only if needed to persist generated current-turn attachment descriptions after initial save.
- Relevant tests:
  - `tests/test_service_input_queue.py`
  - `tests/test_msg_decontexualizer.py`
  - `tests/test_persona_supervisor2_rag2_integration.py`
  - `tests/test_rag_initializer_cache2.py`
  - `tests/test_conversation_history_envelope.py`

### Keep

- Adapter platform normalizers remain under `src/adapters`.
- `MessageEnvelope` remains the adapter-to-brain and storage contract.
- `trim_history_dict` continues excluding attachments from raw recent history unless a later plan explicitly changes history payloads.
- Existing DB rows remain untouched.

## Data Migration

No migration is approved.

The implementation must not:

- update old `conversation_history` rows,
- infer missing descriptions from `raw_wire_text`,
- rerun descriptor over historical `base64_data`,
- re-embed historical rows,
- delete old inline `base64_data`.

New rows after this plan should persist generated descriptions when the current response path produces them. Rows where description generation fails may store empty descriptions and must still be prompt-safe.

## LLM Call And Context Budget

Default context cap: 50k tokens. The production design target remains about 10k prompt-relevant content where practical.

| LLM call | Before | After | Response path | Context policy |
|---|---:|---:|---|---|
| Multimedia descriptor | 0-1 per image | unchanged | yes | Receives direct image data URI exactly as today. |
| Relevance | 1 | unchanged | yes | Receives clean semantic user text and compact history. No raw envelope. |
| Decontextualizer | 1 | unchanged | yes | Receives `prompt_message_context`, not `message_envelope`. |
| RAG initializer | 0-1 depending cache | unchanged | yes | Receives prompt-safe context. Serialized human payload must not include forbidden binary fields. |
| RAG dispatcher/evaluator/finalizer | unchanged | unchanged | yes | Must not receive raw current-message envelope through copied context. |
| Cognition/dialog | unchanged | unchanged | yes | If current-message metadata is passed, it must be prompt-safe. |
| Conversation progress recorder | unchanged | unchanged | background | Uses text and compact history only. |

Hard caps:

- `PromptMessageContext` serialized JSON length (via `json.dumps(..., ensure_ascii=False)`) must be `<= MAX_PROMPT_MESSAGE_CONTEXT_CHARS` (5000).
- Per-field caps are nominal upper bounds: body_text 2000, reply excerpt 500, per-attachment description 800, attachment count 4.
- Overflow handling follows the deterministic degradation order in "Prompt-Safe Context Contract → Overflow Degradation Order". No new LLM call is introduced for compression.
- Failure to fit under cap after full degradation raises `PromptContextTooLargeError`; the call site logs the oversized field path and the request fails fast rather than passing raw media.

Verification must include a fixture with a 1 MB+ `base64_data` string and prove the RAG initializer/decontextualizer human payloads stay bounded.

## Implementation Order

1. Create the prompt projection module and unit tests.
   - Build the contract first so consumers do not invent local variants.

2. Add state/schema fields for `prompt_message_context`.
   - Keep `message_envelope` in state for deterministic storage/addressing.

3. Wire service graph to build the projection.
   - Initial projection can be built from body text.
   - After `multimedia_descriptor_agent`, rebuild it with generated attachment descriptions.
   - Ensure downstream nodes see the updated projection.

4. Persist generated descriptions for new current-turn rows.
   - If user messages are initially saved before descriptor output exists, add a targeted post-description update for the current `platform_message_id`.
   - Do not update any other rows.
   - If description generation fails, leave descriptions empty and continue with prompt-safe unavailable summaries.

5. Migrate decontextualizer prompt payload.
   - Replace `message_envelope` with `prompt_message_context`.
   - Update prompt text/examples only as needed to match the new field name.

6. Migrate RAG context and initializer cache key.
   - Replace raw `message_envelope` in RAG context with `prompt_message_context`.
   - Update `_initializer_context_signature` to read `body_text`, `addressed_to_global_user_ids`, `broadcast` from the projection.
   - Do not bump cache versions; signature values are preserved verbatim so existing cache entries remain valid.

7. Audit other prompt payloads for accidental full-envelope serialization.
   - Fix any `json.dumps(...)` payloads that include raw `message_envelope` or raw `user_multimedia_input`.

8. Add integration tests and failure fixture.
   - Capture prompt payloads around decontextualizer and RAG initializer.
   - Assert forbidden fields are absent and size caps hold.

9. Run static checks, focused tests, and full default tests.
   - Record evidence before marking the plan complete.

## Progress Checklist

Plan reread discipline:

- If an automatic context compaction occurs at any point, reread this entire plan before continuing work.
- After completing verification and filling the sign-off line for any stage below, reread this entire plan before starting the next stage.
- Record each reread in `Execution Evidence` under `Plan reread log`; do not treat memory of the plan before compaction or before sign-off as sufficient.

- [x] Stage 1 - prompt projection contract created
  - Covers: new module, TypedDicts, caps, projection function, safety assertion.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_prompt_message_context.py -q`.
  - Evidence: record unit test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex / 2026-05-01T12:57:45+12:00` after verification and evidence are recorded.

- [x] Stage 2 - graph state carries prompt-safe context
  - Covers: `state.py`, `persona_supervisor2_schema.py`, service integration.
  - Verify: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py src\kazusa_ai_chatbot\service.py`.
  - Evidence: record compile output and changed state fields.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex / 2026-05-01T12:59:19+12:00` after verification and evidence are recorded.

- [x] Stage 3 - multimedia summaries feed projection and new-row persistence
  - Covers: descriptor output, prompt context rebuild, current-row description persistence, descriptor-success-then-crash race (fixture B).
  - Verify: focused service tests in `tests\test_service_input_queue.py` and storage tests in `tests\test_conversation_history_envelope.py`. Race-path test must show: saved row has empty description, next-turn projection is still prompt-safe with `summary_status="unavailable"`.
  - Evidence: record test output and note that no legacy rows were modified.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex / 2026-05-01T13:05:20+12:00` after verification and evidence are recorded.

- [x] Stage 4 - decontextualizer and RAG stop serializing raw envelopes
  - Covers: decontextualizer payload, RAG context, cache key policy.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_msg_decontexualizer.py tests\test_persona_supervisor2_rag2_integration.py tests\test_rag_initializer_cache2.py -q`.
  - Evidence: record test output and confirm no cache version was bumped (compatible policy).
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex / 2026-05-01T13:09:17+12:00` after verification and evidence are recorded.

- [x] Stage 5 - binary-leak failure fixture and static audit pass
  - Covers: 1 MB+ base64 fixture (A), overflow degradation (C), whitelist enforcement (D), prompt payload capture, static greps.
  - Verify: commands in `Verification` plus fixtures A/C/D in `tests\test_prompt_message_context.py`.
  - Evidence: record grep results, bounded serialized payload sizes, and confirmation that the degradation path and whitelist rejection both fire.
  - Handoff: next agent starts final regression.
  - Sign-off: `Codex / 2026-05-01T13:13:35+12:00` after verification and evidence are recorded.

- [x] Stage 6 - final regression and completion evidence
  - Covers: focused batch, full default pytest, execution evidence.
  - Verify: full command list in `Verification`.
  - Evidence: record final test output.
  - Handoff: plan may be marked completed after owner review or implementation-agent sign-off.
  - Sign-off: `Codex / 2026-05-01T13:16:34+12:00` after verification and evidence are recorded.

## Verification

### Static Greps

- `rg -n '"message_envelope"\s*:\s*state\["message_envelope"\]' src\kazusa_ai_chatbot\nodes`
  - Expected: no prompt-facing payload literally embeds the raw envelope. State assembly in `service.py` may still keep `state["message_envelope"]` for storage/addressing; only LLM-facing JSON payloads in nodes must avoid it.
- `rg -n '\b(base64_data|raw_wire_text|image_url|inline_data|inline_bytes)\b' src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\conversation_progress`
  - Expected: no normal text prompt payload serializes these fields. Allowed exceptions are direct-modality code in `multimedia_descriptor_agent` and storage/audit code outside prompt JSON. Use word boundaries; do not grep bare `url` (false positives on `display_name`, `urllib`, `url_template`, etc.).
- `rg -n 'prompt_message_context' src\kazusa_ai_chatbot tests`
  - Expected: projection is used by decontextualizer/RAG tests, state schemas, and service-graph wiring.
- `rg -n 'trim_history_dict' src\kazusa_ai_chatbot`
  - Expected: continues to exclude attachments from `chat_history_recent`. Confirm helper is unchanged or still strips attachments before they reach decontextualizer / RAG.

### Compile

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\message_envelope\prompt_projection.py src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py src\kazusa_ai_chatbot\rag\cache2_policy.py`

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests\test_prompt_message_context.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_service_input_queue.py tests\test_conversation_history_envelope.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_msg_decontexualizer.py tests\test_persona_supervisor2_rag2_integration.py tests\test_rag_initializer_cache2.py -q`

### Failure Fixture

Add deterministic tests covering the documented failure paths.

**A. Inline 1 MB+ base64 image.** Construct:

- `message_envelope.body_text=""`,
- one `attachments` entry with `media_type="image/jpeg"`,
- one `base64_data` string larger than 1 MB,
- generated `multimedia_input[0].description`.

Assert:

- serialized decontextualizer payload contains the description,
- serialized RAG initializer payload contains the description,
- neither payload contains `base64_data`,
- neither payload contains the raw base64 string (substring check),
- neither payload contains `raw_wire_text`,
- prompt-safe context serialized JSON length is `<= MAX_PROMPT_MESSAGE_CONTEXT_CHARS` (5000).

**B. Description-persistence race.** Simulate descriptor success followed by a response-path crash before the description is written back to the conversation row. Assert:

- the saved row has empty `attachments[0].description` (no backfill, accepted state),
- the next-turn projection built from the saved row still passes safety assertion and stays under the cap,
- the projection reports `summary_status="unavailable"` for that attachment.

**C. Overflow degradation.** Construct a projection where worst-case per-field values overflow `MAX_PROMPT_MESSAGE_CONTEXT_CHARS`. Assert:

- the degradation order is applied (attachments-beyond-2 dropped → description caps halved → body_text halved → reply excerpt halved),
- the final serialized length is `<=` cap,
- if degradation cannot reach the cap, `PromptContextTooLargeError` is raised and the failure log names the offending field.

**D. Whitelist enforcement.** Construct a projection-shaped dict that smuggles a foreign key (e.g. `"inline_bytes"`) at any nesting level. Assert `assert_prompt_message_context_safe` rejects it with a clear message naming the offending key path.

### Database Checks

- Export or inspect one new media row after a test/smoke where description generation succeeds.
- Verify its attachment description is populated when generated.
- Verify no command or script updated historical rows.

### Final Regression

- `venv\Scripts\python.exe -m pytest -q`

## Acceptance Criteria

This plan is complete when:

- No prompt-facing LLM payload serializes raw `message_envelope` with storage-only fields.
- The exact large-inline-image failure class cannot exceed prompt budget through `base64_data`.
- Multimedia summaries are still available to relevance, decontextualizer, RAG, and cognition.
- New media rows persist generated attachment descriptions when available.
- Legacy rows with empty descriptions remain supported without backfill.
- RAG initializer cache keys are stable and do not depend on binary storage fields.
- Focused tests and full default pytest pass.
- Execution evidence records static grep results, prompt payload size checks, test output, and the no-backfill confirmation.

## Rollback / Recovery

- Code rollback path: revert the prompt projection integration and tests in one commit if response-path behavior regresses.
- Data rollback path: no legacy data changes are performed. New-row description updates are additive text fields; reverting code leaves stored descriptions harmless.
- Irreversible operations: none approved.
- Required backup: no backup required because no migration/backfill is performed.
- Recovery verification: service must accept a text-only message and an image-only message without serializing binary into downstream LLM prompts.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Image summaries disappear from reasoning | Keep state-level `user_input` (post-relevance prose, owned by relevance_agent) flowing to decontextualizer alongside `prompt_message_context.attachments[].description` for structured access | Failure fixture and decontextualizer/RAG integration tests |
| RAG-only fix leaves decontextualizer vulnerable | Shared projection boundary | Static grep and decontextualizer prompt capture |
| New rows still lack descriptions | Persist current-turn generated descriptions after descriptor output | Conversation history envelope test |
| Cache key churn or stale cache | Signature reads same scope fields (`body_text`, `addressed_to_global_user_ids`, `broadcast`) — preserved verbatim by the projection so no version bump is required | RAG cache tests must show identical cache keys before/after for the same logical input |
| Prompt cap truncates important user text | Preserve `body_text`, addressing, reply metadata first; cap only bounded projection copy | Unit tests for priority and cap behavior |
| Adapter contract drift | `/chat` still accepts the same envelope; adapters do not need to pre-describe media | Runtime adapter tests |

## Execution Evidence

- Static grep results:
  - `rg -n '"message_envelope"\s*:\s*state\["message_envelope"\]' src\kazusa_ai_chatbot\nodes` returned no matches.
  - `rg -n '\b(base64_data|raw_wire_text|image_url|inline_data|inline_bytes)\b' src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\conversation_progress` returned only `src\kazusa_ai_chatbot\nodes\relevance_agent.py` direct image-modality descriptor lines: `image_url` payload construction and in-flight `base64_data` preservation for the vision call. No decontextualizer, RAG, cognition, dialog, or conversation-progress prompt payload matches.
  - `rg -n 'message_envelope|base64_data|raw_wire_text' src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py src\kazusa_ai_chatbot\nodes\dialog_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py` returned no matches.
  - `rg -n 'prompt_message_context' src\kazusa_ai_chatbot tests` confirms projection usage in service, relevance descriptor rebuild, decontextualizer, RAG context/cache policy, persona/cognition state, and tests.
  - `rg -n 'trim_history_dict' src\kazusa_ai_chatbot` confirms history trimming remains centralized in `utils.py` and used by service, relevance, and consolidator; the helper still excludes attachments.
- Compile results:
  - Stage 2: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py src\kazusa_ai_chatbot\service.py` passed with no output.
  - Stage 3: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\relevance_agent.py src\kazusa_ai_chatbot\db\conversation.py` passed with no output.
  - Stage 4: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\rag\cache2_policy.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py` passed with no output.
  - Stage 5: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\message_envelope\prompt_projection.py src\kazusa_ai_chatbot\nodes\relevance_agent.py` passed with no output.
- Focused test results:
  - Stage 1: `venv\Scripts\python.exe -m pytest tests\test_prompt_message_context.py -q` passed: 5 passed in 0.04s.
  - Stage 3: `venv\Scripts\python.exe -m pytest tests\test_relevance_agent.py tests\test_service_input_queue.py tests\test_conversation_history_envelope.py -q` passed: 40 passed in 4.59s.
  - Stage 3 race fixture: `venv\Scripts\python.exe -m pytest tests\test_prompt_message_context.py -q` passed: 6 passed in 0.04s.
  - Stage 4: `venv\Scripts\python.exe -m pytest tests\test_msg_decontexualizer.py tests\test_persona_supervisor2_rag2_integration.py tests\test_rag_initializer_cache2.py -q` passed: 24 passed in 4.28s.
  - Stage 5 projection fixtures: `venv\Scripts\python.exe -m pytest tests\test_prompt_message_context.py -q` passed: 8 passed in 0.05s.
  - Stage 5 RAG initializer payload fixture: `venv\Scripts\python.exe -m pytest tests\test_rag_initializer_cache2.py -q` passed: 13 passed in 1.63s.
  - Stage 6 focused rerun: compile passed with no output; `tests\test_prompt_message_context.py -q` passed: 8 passed in 0.04s; `tests\test_service_input_queue.py tests\test_conversation_history_envelope.py -q` passed: 20 passed in 1.77s; `tests\test_msg_decontexualizer.py tests\test_persona_supervisor2_rag2_integration.py tests\test_rag_initializer_cache2.py -q` passed: 25 passed in 4.28s.
  - Stage 6 legacy fixture repair check: `venv\Scripts\python.exe -m pytest tests\test_decontexualizer_referents.py tests\test_message_envelope.py::test_state_shapes_accept_message_envelope_and_addressee_fields tests\test_persona_supervisor2.py tests\test_persona_supervisor2_schema.py -q` passed: 19 passed, 3 deselected in 1.65s.
- Failure fixture result:
  - Fixture A: `test_large_inline_image_prompt_payloads_remain_bounded` and `test_rag_initializer_payload_uses_prompt_context_for_large_image` confirm a 1 MB+ inline `base64_data` source projects to bounded decontextualizer/RAG payloads containing the generated description while omitting `base64_data`, `raw_wire_text`, and the raw base64 substring. Measured serialized sizes: projection 220 chars, decontextualizer payload 307 chars, RAG initializer payload 283 chars.
  - Fixture B: `test_descriptor_crash_saved_row_projects_unavailable` confirms a saved row with empty attachment description remains prompt-safe, omits `base64_data`, stays under `MAX_PROMPT_MESSAGE_CONTEXT_CHARS`, and reports `summary_status="unavailable"`.
  - Fixture C: `test_projection_applies_overflow_degradation_order` confirms attachments beyond 2 are dropped, descriptions/body/reply are reduced, and the final projection stays under cap; `test_projection_raises_when_required_metadata_cannot_fit` confirms `PromptContextTooLargeError` names reduced fields when the projection cannot fit.
  - Fixture D: `test_safety_assertion_rejects_foreign_nested_keys` and `test_safety_assertion_rejects_foreign_keys_inside_leaf_values` confirm whitelist enforcement rejects smuggled `inline_bytes` keys both as direct nested keys and below allowed leaf fields.
  - Stage 4 prompt capture confirms decontextualizer payload contains `prompt_message_context`, omits raw `message_envelope`, and omits `raw_wire_text`.
- Database verification:
- Full pytest result:
  - Stage 6 final regression: `venv\Scripts\python.exe -m pytest -q` passed: 432 passed, 139 deselected in 6.70s.
  - Code-review follow-up regression: `venv\Scripts\python.exe -m pytest -q` passed: 436 passed, 139 deselected in 6.75s.
- Code-review follow-up:
  - Mixed-media projection now consumes generated descriptions only for image attachment slots, so a preceding non-image file cannot receive an image summary and extra collapsed-image descriptions do not populate the current envelope's structured attachment rows.
  - Decontextualizer prompt input example now includes the full prompt-safe context shape: `broadcast`, `mentions`, `attachments`, and expanded `reply`.
  - `text/*` attachments now project as `media_kind="file"` to match the documented contract.
  - Description sentence-boundary trimming removes terminal sentence punctuation before appending `...`, preventing stacked `....` output.
  - Vision descriptor LLM/parse failures now log and continue with empty descriptions, yielding prompt-safe `summary_status="unavailable"` instead of aborting the turn.
  - Focused validation: `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\message_envelope\prompt_projection.py src\kazusa_ai_chatbot\nodes\relevance_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_msg_decontexualizer.py` passed with no output; prompt render smoke printed `render_ok`; `venv\Scripts\python.exe -m pytest tests\test_prompt_message_context.py tests\test_relevance_agent.py tests\test_msg_decontexualizer.py -q` passed: 40 passed in 4.38s.
- No-backfill confirmation:
  - Stage 3 only updates the current row by exact `{platform, platform_channel_id, platform_message_id}` after descriptor success. No legacy-row update/backfill script or migration was run.
- Cache compatibility:
  - Stage 4 kept `INITIALIZER_PROMPT_VERSION == "initializer_prompt:v6"`; no cache version was bumped for the prompt-context cutover.
- Plan reread log:
  - `2026-05-01T12:57:45+12:00` - trigger: execution start before Stage 1; action: reread entire plan, then implemented Stage 1.
  - `2026-05-01T12:58:16+12:00` - trigger: Stage 1 sign-off; action: reread entire plan, then start Stage 2.
  - `2026-05-01T12:59:37+12:00` - trigger: Stage 2 sign-off; action: reread entire plan, then start Stage 3.
  - `2026-05-01T13:04:09+12:00` - trigger: automatic context compaction during Stage 3; action: reread entire plan, then resume Stage 3 verification.
  - `2026-05-01T13:05:41+12:00` - trigger: Stage 3 sign-off; action: reread entire plan, then start Stage 4.
  - `2026-05-01T13:09:41+12:00` - trigger: Stage 4 sign-off; action: reread entire plan, then start Stage 5.
  - `2026-05-01T13:14:11+12:00` - trigger: Stage 5 sign-off; action: reread entire plan, then start Stage 6 final regression.
  - `2026-05-01T13:16:56+12:00` - trigger: Stage 6 sign-off; action: reread completed plan, then final report.
  - `2026-05-01T15:47:24+12:00` - trigger: automatic context compaction before code-review follow-up; action: reread entire plan, then address review feedback and rerun focused/full validation.
