# Past Dialog Cognition Residual ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.past_dialog_cognition`
- Runtime owners:
  - direct reply/quote source: `kazusa_ai_chatbot.service`
  - conversation-evidence source: `kazusa_ai_chatbot.cognition_resolver.loop`
- Storage readers:
  - `kazusa_ai_chatbot.db.conversation.list_conversation_rows_by_row_ids`
  - `kazusa_ai_chatbot.db.llm_tracing.list_llm_trace_steps_for_trace_ids`
- Prompt consumer: L2a consciousness only, via
  `past_dialog_cognition_context`
- Non-goals: durable memory, public RAG evidence, dialog planning, L3 surface
  rendering, adapter delivery, scheduler/reflection input, trace backfill, and
  semantic fetch classification

## Purpose

This package builds a private, prompt-safe residual context for cases where the
current turn is already structurally anchored to a specific prior Kazusa
dialog. The main use case is a user replying to or quoting an earlier Kazusa
message and asking what she meant, why she said it, or how that prior idea
connects to the current conversation.

Visible prior dialog text can be insufficient because it shows only the final
surface sentence, not the private cognition that produced it. When protected
full-capture LLM traces are available, this package can recover a compact
residual from selected parsed cognition-stage outputs and give L2a enough
continuity to interpret the current turn. When traces are unavailable, the
system intentionally projects nothing. That is treated as normal forgetting,
not as an error state for the live response path.

This package is not a memory system. It does not retrieve arbitrary old
thoughts, search by user wording, infer whether the user is asking about the
past, or ask an LLM whether residual should be fetched. It only acts after
another structural path has already attached a concrete past Kazusa-authored
dialog.

## Use Cases

### Direct Reply Or Quote Clarification

A user replies to a previous Kazusa message:

```text
Kazusa: "I meant that the idea needed time."
User: "What did you mean by that?"
```

The service already has a `reply_to_message_id`. This package can load the
replied conversation row, confirm it is a Kazusa-authored assistant row with an
`llm_trace_id`, read selected trace steps, and project a bounded
`past_dialog_cognition_context` for L2a.

This private lookup is separate from visible reply hydration. It still runs
when adapter-supplied reply metadata is complete and `_hydrate_reply_context`
does not need to load the row for visible metadata.

### Conversation-Evidence Continuity

RAG may retrieve prior conversation rows as public evidence. When those rows
include trace-only source refs under
`rag_result.supervisor_trace.dispatched[*].source_refs`, this package can use
`conversation_row_id` or `_id` refs to reload the rows privately, filter for
Kazusa-authored assistant rows, and derive the same residual context.

The residual never enters public `rag_result`, public `conversation_evidence`,
RAG worker output, or resolver observations.

If multiple retrieved assistant rows share the same `llm_trace_id` because one
cognition produced multiple logical dialog messages, this package projects that
private residual once. The visible rows can remain separate public evidence;
the private cognition source is the shared trace.

### Missing Data As Forgetting

The package returns an empty context when:

- the current turn has no structurally attached past dialog;
- the row is not an assistant row;
- the row was not authored by the active character;
- the row has empty visible text;
- the row has no `llm_trace_id`;
- no selected trace step exists;
- trace capture was metadata-only and `parsed_output` is empty;
- MongoDB read failures prevent row or trace lookup.

The caller should continue the normal response path with no
`past_dialog_cognition_context` projected to L2a.

## Ownership Boundary

The package owns:

- candidate filtering for already-attached past dialogs;
- extraction of row ids from trace-only RAG source refs;
- DB-backed row and trace lookup orchestration;
- parsed trace-step filtering;
- prompt-safe residual text formatting;
- max-dialog and character-budget enforcement;
- deduplication of multiple attached rows that share one cognition trace;
- omission behavior for unavailable data;
- diagnostics for deterministic tests and debug inspection.

The package does not own:

- deciding whether the user is asking about the past;
- semantic or keyword matching over user text;
- RAG retrieval planning or public evidence wording;
- direct reply visible metadata hydration;
- L2a prompt wording outside the `past_dialog_cognition_context` contract;
- L2b, L2c1, L2c2, L2d, L3, dialog, consolidation, scheduler, reflection, or
  adapter behavior.

## Public Facade

Runtime callers import from:

```python
from kazusa_ai_chatbot.past_dialog_cognition import (
    PastDialogCognitionCandidate,
    build_past_dialog_cognition_context,
    build_past_dialog_cognition_context_from_rag_result,
    candidate_from_conversation_row,
    candidates_from_conversation_rows,
    conversation_row_ids_from_rag_result,
)
```

### `build_past_dialog_cognition_context(...)`

Builds prompt-facing residual context from already-constructed candidates:

```python
async def build_past_dialog_cognition_context(
    candidates: Sequence[PastDialogCognitionCandidate],
    *,
    character_global_user_id: str,
    max_dialogs: int = 3,
    context_char_limit: int = 1800,
) -> PastDialogCognitionLookupResult:
    ...
```

This is the shared projector for direct reply and RAG evidence paths. It
filters candidates before trace lookup, reads only selected trace stages, and
returns one prompt-facing field plus diagnostics.

### `build_past_dialog_cognition_context_from_rag_result(...)`

Builds residual context from row-id source refs in a RAG result:

```python
async def build_past_dialog_cognition_context_from_rag_result(
    rag_result: Mapping[str, Any],
    *,
    character_global_user_id: str,
    max_dialogs: int = 3,
    context_char_limit: int = 1800,
) -> PastDialogCognitionLookupResult:
    ...
```

This helper extracts `conversation_row_id` or `_id` refs, reloads bounded
conversation rows through the DB facade, and delegates projection to
`build_past_dialog_cognition_context(...)`. It must not mutate `rag_result`.

### Candidate Helpers

`candidate_from_conversation_row(...)` and
`candidates_from_conversation_rows(...)` translate conversation-history rows
into candidate objects. They do only structural row projection; author and
trace eligibility are validated again by the shared builder.

`conversation_row_ids_from_rag_result(...)` extracts only row-id style refs
from RAG supervisor trace source refs. It intentionally ignores unscoped
`platform_message_id` values.

## Data Contracts

### Candidate Contract

Each candidate represents a visible past dialog already attached by another
source mechanism:

```python
PastDialogCognitionCandidate(
    visible_text=str,
    llm_trace_id=str,
    created_at=object,
    source="reply_context" | "conversation_evidence",
    role=str,
    global_user_id=str,
    conversation_row_id=str,
    platform_message_id=str,
    platform=str,
    platform_channel_id=str,
)
```

Eligibility rules:

- `role` must be `assistant`;
- `global_user_id` must match the active character id;
- `visible_text` must be non-empty;
- `llm_trace_id` must be non-empty.

Candidates failing these checks produce diagnostics but do not trigger trace
lookup.

### Lookup Result Contract

All public builders return:

```python
{
    "past_dialog_cognition_context": str,
    "candidate_count": int,
    "selected_count": int,
    "status": str,
    "diagnostics": list[dict[str, str]],
}
```

Only `past_dialog_cognition_context` may enter cognition prompts. Counts,
status values, diagnostics, trace ids, row ids, and implementation details are
not prompt-facing.

### Trace Read Contract

Trace reads use:

```python
list_llm_trace_steps_for_trace_ids(
    trace_ids,
    stage_names=(
        "l2a_conscious_framing",
        "l2c1_judgment_synthesis",
    ),
)
```

The DB helper projects only:

- `trace_id`
- `stage_name`
- `sequence`
- `parsed_output`
- `created_at`

It must not project `raw_messages`, `raw_response_text`, raw prompts, raw model
responses, or protected trace payloads beyond selected parsed output.

## Projection Contract

Prompt-facing residual is built only from these parsed-output fields:

| Stage | Fields |
| --- | --- |
| `l2a_conscious_framing` | `internal_monologue`, `logical_stance`, `character_intent` |
| `l2c1_judgment_synthesis` | `logical_stance`, `character_intent`, `judgment_note` |

The projected text may include the visible prior dialog text and natural
language labels such as "private thought" or "earlier stance". It must not
include:

- `trace_id`;
- `conversation_row_id` or Mongo `_id`;
- `platform_message_id`;
- implementation stage labels;
- raw JSON blobs;
- raw prompts;
- raw responses;
- `raw_messages`;
- `raw_response_text`;
- dialog drafts or final dialog wording from trace payloads.

The current caps are:

- at most `3` dialogs;
- at most `1800` characters for the combined L2a context;
- bounded per-field and visible-dialog snippets in projection.

## Runtime Integration

### Direct Reply Path

`kazusa_ai_chatbot.service.load_conversation_episode_state(...)` calls the
private reply residual loader after conversation progress and internal
monologue residue loading. The loader:

1. reads `state["reply_context"]["reply_to_message_id"]`;
2. performs a platform/channel-scoped
   `get_conversation_by_platform_message_id(...)` lookup;
3. builds one reply candidate from the row;
4. calls `build_past_dialog_cognition_context(...)`;
5. returns a string for `past_dialog_cognition_context` or `""`.

This path does not modify `reply_context` or `prompt_message_context`.

### Conversation-Evidence Path

`kazusa_ai_chatbot.cognition_resolver.loop` attaches residual after a RAG
capability observation has copied `observation["rag_result"]` into cognition
state. It calls
`build_past_dialog_cognition_context_from_rag_result(...)` and writes only a
non-empty `past_dialog_cognition_context` directly into cognition state.

`ResolverObservationV1` remains unchanged. The residual must not be stored in
resolver observations or resolver state projections.

### Cognition And Surface Boundary

`past_dialog_cognition_context` is part of the private cognition input lane.
The persona connector may carry it into `ConversationContextPromptV1` so the
cognition core can place it in L2a. L2a includes the field in the human payload
only when it is non-empty.

Selected L3 text-surface construction must strip the field from the nested
chain input before building `CognitionTextSurfaceInputV1`. L3, dialog, and
surface planning receive only the normal cognition outputs produced after L2a.

## Failure Behavior

The feature is best-effort by design:

- Empty candidates return empty context.
- Empty `parsed_output` returns empty context.
- Missing trace rows return empty context.
- MongoDB read failures return empty context with diagnostic status.

Only database read failures are treated as ordinary forgetting by exception
handling. Projection bugs, contract bugs, type errors introduced by code
changes, and unexpected implementation failures should surface during tests or
runtime error handling rather than being silently converted to forgetting.

## Security And Privacy

This package consumes protected trace data and projects a small private
cognition summary. It must preserve these boundaries:

- no raw trace payload in prompts;
- no private residual in public RAG evidence;
- no private residual in dialog text or adapter payloads;
- no residual in durable memory, reflection, scheduler, dispatcher, or
  message-envelope projections;
- no diagnostic ids in L2a prompt context;
- no ordinary UI display of diagnostics or trace linkage.

The package may expose diagnostics to deterministic tests and protected debug
inspection only.

## Forbidden Consumers

Do not feed `past_dialog_cognition_context`, raw trace steps, or residual
diagnostics to:

- L1 subconscious cognition;
- L2b boundary appraisal;
- L2c1 judgment synthesis;
- L2c2 social context appraisal;
- L2d action selection;
- L3 surface planning;
- dialog rendering;
- public RAG projection or RAG finalizer text;
- consolidation;
- message envelope projection;
- adapters or delivery receipts;
- dispatcher, scheduler, proactive output, or reflection cycle.

Only L2a may consume the non-empty prompt-facing
`past_dialog_cognition_context`.

## Verification Expectations

Deterministic verification should cover:

- no candidate returns empty context;
- metadata-mode empty `parsed_output` returns empty context;
- DB read failures are treated as forgetting;
- non-Kazusa and non-assistant rows are filtered before trace lookup;
- only approved L2a and L2c1 parsed fields are projected;
- raw trace fields and ids never appear in prompt context;
- max dialog and character caps are enforced;
- direct reply lookup works even when visible reply metadata is complete;
- RAG source refs use row ids, not unscoped platform message ids;
- public `rag_result` and `conversation_evidence` are not mutated;
- `ResolverObservationV1` has no residual field;
- L2a receives non-empty context and omits empty context;
- L1, L2c1, L3, dialog, public RAG, consolidation, scheduler, reflection, and
  adapters do not receive the field.

Real LLM validation is separate from deterministic contract verification. A
quality test for the motivating use case should seed or select a prior Kazusa
dialog with full parsed trace output, send a reply-style "what did you mean?"
turn, inspect the L2a trace, and confirm the final response uses the residual
for continuity without exposing private monologue verbatim.
