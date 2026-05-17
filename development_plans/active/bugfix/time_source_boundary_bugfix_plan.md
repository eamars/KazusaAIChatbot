# time source boundary bugfix plan

## Summary

- Goal: Split UTC storage time from configured local runtime time, centralize all conversion in one module, and remove ambiguous timestamp names that caused model-facing local time to shift to the wrong day.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `py-style`, `cjk-safety`, `test-style-and-execution`, `local-llm-architecture`, `superpowers:systematic-debugging`, `superpowers:test-driven-development`, `superpowers:verification-before-completion`.
- Overall cutover strategy: bigbang for Python contracts and `/chat` timestamp naming; compatible for existing MongoDB field names; no historical data migration.
- Highest-risk areas: `/chat` queue/service state, shared cognition state, RAG runtime context, self-cognition worker cases, scheduled actions, reflection prompts that currently expose UTC/timezone ideas, and broad test fixtures.
- Acceptance criteria: storage writes and scheduler instants use UTC-only values; every non-storage runtime/model path uses the configured local wall-clock representation; no LLM prompt payload contains UTC offsets, IANA timezone names, or timezone instructions; all timezone conversion imports come from `kazusa_ai_chatbot.time_boundary`.

## Context

The user-confirmed source-of-truth rule is:

- UTC time is used for data storage.
- The configured local time is used everywhere else.
- Formatting may drop seconds, minutes, or hours when a caller has a narrower contract, but the source-of-truth boundary must stay consistent.
- The LLM must stay timezone agnostic. Python may supply local dates/times, but the model must not be asked to reason about UTC, offsets, IANA timezone names, or timezone conversion.

The observed failure mode was:

- A turn around `2026-05-17 16:55:28` configured local time reached the LLM as `2026-05-18 04:55`.
- The cognition output then reasoned that it was early morning instead of late afternoon.
- The direct symptom was an offset conversion applied to a value that should already have been treated as the configured local wall-clock.

The root cause is architectural, not one bad helper:

- The same name, `timestamp`, currently represents adapter event input, queue time, storage time, graph turn time, RAG `current_timestamp`, self-cognition `idle_timestamp`, action execution audit time, and prompt-facing time.
- `ChatRequest.timestamp`, `QueuedChatItem.timestamp`, `IMProcessState.timestamp`, `GlobalPersonaState.timestamp`, `CognitiveEpisode.timestamp`, and RAG `current_timestamp` do not encode whether the value is storage UTC or configured local time.
- `build_character_time_context(timestamp)` currently assumes the argument is a UTC instant. That is correct only for true storage timestamps, and it becomes wrong when the caller passes configured local wall-clock input.
- Some model-facing payloads already use `time_context`, while other paths still fall back to raw `timestamp` or `current_timestamp`.
- Time parsing and conversion are scattered across service, queue, RAG, action, scheduler, reflection, event logging, DB helpers, and scripts.

The failed previous fix changed `build_character_time_context` to preserve configured local input. That broke the opposite side of the contract because many callers still pass true UTC storage timestamps. The correct fix is to split the two sources explicitly and make invalid cross-use hard to write.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files containing Chinese prompt text.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing graph, RAG, cognition, dialog, reflection, or prompt-facing payload behavior.
- `superpowers:systematic-debugging`: load before implementation to preserve the root-cause analysis and avoid symptom patches.
- `superpowers:test-driven-development`: load before implementation; every behavior change starts with a focused failing test.
- `superpowers:verification-before-completion`: load before claiming any stage, verification command, or plan completion passes.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the active agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- Do not edit `.env`.
- Run `git status --short` before editing and before final reporting.
- Use `venv\Scripts\python` for Python commands.
- Use `apply_patch` for manual edits.
- Do not revert user changes. If unexpected dirty files appear, inspect them and work around them unless they block this plan.
- Do not touch any LLM prompt unless the prompt currently introduces UTC, timezone names, timezone conversion, or offset-bearing timestamp concepts to the LLM. The allowed prompt files are listed in `Change Surface`.
- Do not add any new LLM call.
- Do not add user, channel, platform, location, IP, locale, or text-inferred timezone support.
- Do not ask the LLM to decide, validate, infer, or convert timezones.
- Do not preserve ambiguous Python names such as `timestamp`, `current_timestamp`, `time_context`, `item.timestamp`, or `request.timestamp` for internal values after this cutover, except where a persisted MongoDB field named `timestamp` is being read or written through an explicitly named `_utc` local variable.
- Existing MongoDB field names such as `conversation_history.timestamp`, memory row `timestamp`, `updated_at`, `created_at`, `execute_at`, and `due_at` remain stored as UTC strings. Do not rename stored fields in this plan.
- All timezone loading, UTC normalization, local parsing, local-to-UTC conversion, UTC-to-local projection, and current UTC time creation must go through `kazusa_ai_chatbot.time_boundary`.
- Tests may use Python `datetime` constructors for fixed fixture values, but production code outside `time_boundary.py` must not call `datetime.now(timezone.utc)`, `datetime.fromisoformat(...)`, `.astimezone(...)`, or `ZoneInfo(...)`.
- If a prompt payload needs the current time, the Python payload builder must pass a configured local string. The prompt must not mention why it is local or how it was converted.

## Must Do

- Create `src/kazusa_ai_chatbot/time_boundary.py` as the only production module that owns time source selection and timezone conversion.
- Delete `src/kazusa_ai_chatbot/time_context.py` after all imports are migrated.
- Rename internal UTC variables and state keys to include `_utc`.
- Rename internal configured-local variables and state keys to include `_local` or `local_`.
- Replace `/chat` request field `timestamp` with `local_timestamp`.
- Keep existing MongoDB `timestamp` fields unchanged, but bind them only through local variables named `timestamp_utc`, `storage_timestamp_utc`, or field-specific names such as `created_at_utc`.
- Replace graph state `timestamp` with `storage_timestamp_utc`.
- Replace graph state `time_context` with `local_time_context`.
- Preserve existing prompt payload key `time_context` only where prompt templates already consume that key; the Python variable that feeds it must be `local_time_context`.
- Replace RAG internal `current_timestamp` with `current_timestamp_utc`.
- Convert model-facing RAG current-time payloads from `local_time_context`; never fall back to raw UTC strings.
- Rename self-cognition case fields from `idle_timestamp` and `last_evidence_timestamp` to `idle_timestamp_utc` and `last_evidence_timestamp_utc` for source data, and use local field names in rendered source packets.
- Normalize LLM-produced exact local datetimes to UTC only through `local_llm_datetime_to_storage_utc_iso`.
- Remove all compatibility acceptance of LLM-produced offset-aware ISO timestamps. LLM output time fields must be exact local `YYYY-MM-DD HH:MM`.
- Remove prompt exposure of `character_time_zone`, `UTC hour-start ISO timestamp`, `timezone`, `time zone`, `时区`, raw `Z`, and offset instructions from the allowed prompt files.
- Update deterministic tests and fixture builders to use the new `_utc` and local naming.
- Add static grep tests or CI-style pytest assertions that fail on new scattered conversion calls and ambiguous internal timestamp keys.

## Deferred

- Do not rename persisted MongoDB fields.
- Do not migrate historical MongoDB rows.
- Do not add per-user timezone support.
- Do not add per-channel or per-platform timezone support.
- Do not infer local time from user text, adapter locale, platform profile, location, IP, weather city, browser locale, operating-system timezone, or model output.
- Do not redesign scheduler semantics beyond explicit UTC storage and local-to-UTC conversion.
- Do not redesign RAG helper routing, Cache2 policy, cognition prompts, dialog prompts, memory schemas, or reflection architecture.
- Do not add compatibility aliases for the deleted `time_context.py` names.
- Do not keep accepting `/chat` request field `timestamp` after this cutover.

## Data Migration

No database migration is part of this plan. Existing persisted fields keep their
current names and values. The migration is a Python/API contract cutover only.

## Cutover Policy

Overall strategy: bigbang for code contracts, compatible for durable storage fields.

| Area | Policy | Instruction |
|---|---|---|
| Python time module | bigbang | Create `time_boundary.py`, migrate all callers, then delete `time_context.py`. No compatibility module or alias. |
| `/chat` request timestamp | bigbang | Replace optional `timestamp` with optional `local_timestamp`. `extra="forbid"` must reject old `timestamp` payloads. |
| Queue item time fields | bigbang | Replace `QueuedChatItem.timestamp` with `storage_timestamp_utc`, `local_timestamp`, and `local_time_context`. |
| Graph state | bigbang | Replace `timestamp` with `storage_timestamp_utc` and `time_context` with `local_time_context`. |
| Cognitive episode contract | bigbang | Replace `timestamp` with `storage_timestamp_utc` and `time_context` with `local_time_context`. |
| RAG runtime context | bigbang | Replace internal `current_timestamp` with `current_timestamp_utc`; emit prompt key `time_context` from `local_time_context` only. |
| Self-cognition case source fields | bigbang | Rename source UTC fields to `_utc`; source packet rendering uses only local field names. |
| Existing MongoDB fields | compatible | Keep stored field names unchanged. Code variables around those fields must be `_utc`. |
| Prompt files | bigbang only for forbidden timezone concepts | Edit only the prompt constants named in `Change Surface`; remove timezone concepts, not semantic responsibilities. |
| Tests | bigbang | Update fixtures and assertions to the new contract. Delete obsolete tests that assert old ambiguous names. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- If an area is `bigbang`, delete or rewrite the legacy reference instead of preserving it.
- If an area is `compatible`, preserve only the compatibility surface explicitly listed here.
- Any compatibility alias, fallback, dual read, or alternate field name not listed here is forbidden.
- Any change to a cutover policy requires user approval before implementation.

## Overdesign Guardrail

- Actual problem: a configured local turn time was treated as a UTC instant, shifted across the date boundary, and reached cognition as the wrong local hour.
- Minimal change: split storage UTC and configured local runtime fields, centralize conversions, and rename ambiguous time variables so wrong source use is visible in code review and tests.
- Ownership boundaries: deterministic code owns UTC/local conversion, storage timestamps, scheduler timestamps, validation, and prompt payload projection; LLM stages receive local strings only and own semantic reasoning over those local strings.
- Rejected complexity: user timezone profiles, channel timezone profiles, adapter timezone negotiation, browser timezone capture, timezone inference from text, offset-bearing LLM output, compatibility aliases, feature flags, and new LLM repair paths.
- Evidence threshold: add any rejected complexity only after an approved product requirement names a second timezone source and supplies a deterministic source of truth outside model inference.

## Agent Autonomy Boundaries

- The implementation agent may choose local statement ordering only when it preserves the exact contracts in this plan.
- The implementation agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The implementation agent must not touch prompt files outside the three prompt files explicitly allowed below.
- The implementation agent must not change prompt meaning except to remove UTC/timezone concepts and keep equivalent copy-exact or local-time semantics.
- The implementation agent must not perform unrelated cleanup, formatting churn, dependency upgrades, or broad refactors.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

Storage values are UTC-only:

```python
storage_timestamp_utc = "2026-05-17T04:55:28.395000+00:00"
execute_at_utc = "2026-05-17T08:00:00+00:00"
```

Configured local runtime values are timezone-unaware:

```python
local_timestamp = "2026-05-17 16:55:28.395000"
local_time_context = {
    "current_local_datetime": "2026-05-17 16:55",
    "current_local_weekday": "Sunday",
}
```

Model-facing payloads may contain:

```json
{
  "timestamp": "2026-05-17 16:55",
  "time_context": {
    "current_local_datetime": "2026-05-17 16:55",
    "current_local_weekday": "Sunday"
  }
}
```

Model-facing payloads must not contain:

```text
2026-05-17T04:55:28+00:00
2026-05-18 04:55
+12:00
UTC
Pacific/Auckland
IANA timezone
timezone
time zone
时区
```

The first line is forbidden because it is a storage UTC timestamp. The second line is forbidden for the regression case because it is the result of converting a configured local timestamp as if it were UTC.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Canonical module | Use `kazusa_ai_chatbot.time_boundary` | The module name describes the source-of-truth boundary rather than only prompt context. |
| Old module | Delete `time_context.py` | Compatibility aliases would preserve confusing names and allow future mistakes. |
| Current `/chat` input | Use optional `local_timestamp` | Adapter/user-provided current-turn time is non-storage runtime input. |
| Missing `/chat` input time | Build both UTC and local from `storage_utc_now()` | Existing adapters omit time; service receive time remains the source. |
| Storage field names | Keep MongoDB names unchanged | Renaming durable fields requires a separate migration and is not needed to fix code-source ambiguity. |
| Python storage names | Use `_utc` suffix | Storage source of truth is explicit at every call site. |
| Python local names | Use `_local` or `local_` prefix | Configured-local runtime source is explicit at every call site. |
| Graph state names | `storage_timestamp_utc`, `local_time_context` | Prevents LLM-facing code from accidentally reading storage UTC. |
| Prompt payload key `time_context` | Preserve where already used | Avoids broad prompt rewrites; the code variable feeding it is renamed. |
| LLM output time | Accept only exact local `YYYY-MM-DD HH:MM` | The LLM must not produce offsets or timezone concepts. |
| Reflection prompt UTC mentions | Remove them | These prompts currently introduce timezone concepts to the model, which is forbidden. |
| Direct UTC clock calls | Forbid outside `time_boundary.py` | Prevents new scattered source-of-truth code. |

## Contracts And Data Shapes

### New Module Interface

Create `src/kazusa_ai_chatbot/time_boundary.py` with this public interface:

```python
class LocalTimeContextDoc(TypedDict):
    current_local_datetime: str
    current_local_weekday: str


class TurnClock(TypedDict):
    storage_timestamp_utc: str
    local_timestamp: str
    local_time_context: LocalTimeContextDoc


def storage_utc_now() -> datetime: ...


def storage_utc_now_iso() -> str: ...


def parse_storage_utc_datetime(value: str) -> datetime: ...


def normalize_storage_utc_iso(value: str) -> str: ...


def parse_configured_local_datetime(value: str) -> datetime: ...


def local_datetime_to_storage_utc_iso(value: str) -> str: ...


def build_turn_clock(local_timestamp: str | None = None) -> TurnClock: ...


def build_turn_clock_from_storage_utc(
    storage_timestamp_utc: str,
) -> TurnClock: ...


def local_time_context_from_storage_utc(
    storage_timestamp_utc: str,
) -> LocalTimeContextDoc: ...


def format_storage_utc_for_llm(value: str | None) -> str: ...


def format_storage_utc_history_for_llm(rows: list[dict]) -> list[dict]: ...


def format_storage_utc_fields_for_llm(
    row: dict,
    time_fields: tuple[str, ...],
) -> dict: ...


def local_llm_datetime_to_storage_utc_iso(value: str) -> str: ...


def local_date_bounds_to_storage_utc_iso(
    local_date: str,
) -> tuple[str, str]: ...


def one_second_before_storage_utc_iso(timestamp_utc: str) -> str: ...
```

`parse_configured_local_datetime` accepts only these local wall-clock forms:

- `YYYY-MM-DD HH:MM`
- `YYYY-MM-DD HH:MM:SS`
- `YYYY-MM-DD HH:MM:SS.ffffff`
- `YYYY-MM-DD HH:MM:SS,ffffff`

It rejects `T`, `Z`, `UTC`, offsets, timezone names, date-only strings, and natural language.

`local_timestamp` in `TurnClock` is formatted as `YYYY-MM-DD HH:MM:SS` plus fractional seconds when the source contains fractional seconds. `LocalTimeContextDoc.current_local_datetime` drops seconds and fractional seconds as `YYYY-MM-DD HH:MM`.

`format_storage_utc_for_llm` accepts only storage UTC or already formatted local strings. It returns an empty string for invalid input. It must not pass through offset-bearing or ambiguous strings unchanged.

### Request And Queue Contract

`ChatRequest` becomes:

```python
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str
    platform_channel_id: str = ""
    channel_type: str = "group"
    platform_message_id: str = ""
    platform_user_id: str
    platform_bot_id: str = ""
    display_name: str = ""
    channel_name: str = ""
    content_type: str = "text"
    message_envelope: MessageEnvelopeIn
    local_timestamp: str = ""
    debug_modes: DebugModesIn = Field(default_factory=DebugModesIn)
```

`QueuedChatItem` becomes:

```python
@dataclass
class QueuedChatItem:
    sequence: int
    request: Any
    storage_timestamp_utc: str
    local_timestamp: str
    local_time_context: LocalTimeContextDoc
    future: asyncio.Future[Any]
    combined_content: str | None = None
    collapsed_items: list[QueuedChatItem] = field(default_factory=list)
    conversation_row_id: str = ""
```

`ChatInputQueue.enqueue` must call `build_turn_clock(request.local_timestamp or None)` once and copy the three fields onto the queued item. It must not call `datetime.now(...)` directly.

### Graph State Contract

`IMProcessState`, `GlobalPersonaState`, `CognitionState`, `ConsolidatorState`, and related state builders must use:

```python
storage_timestamp_utc: str
local_time_context: LocalTimeContextDoc
```

No state schema may define a required internal key named `timestamp` or `time_context`.

When building a prompt payload for an existing prompt that expects these keys, use:

```python
payload = {
    "timestamp": state["local_time_context"]["current_local_datetime"],
    "time_context": state["local_time_context"],
}
```

Do not expose `storage_timestamp_utc` in prompt payloads.

### Cognitive Episode Contract

`CognitiveEpisode` and `TextChatCompatibilityProjection` must use:

```python
storage_timestamp_utc: str
local_time_context: LocalTimeContextDoc
```

`build_text_chat_cognitive_episode(...)` must accept `storage_timestamp_utc` and `local_time_context`. `project_text_chat_compatibility(...)` must return those same names.

### RAG Runtime Contract

Internal RAG context must use:

```python
current_timestamp_utc: str
local_time_context: LocalTimeContextDoc
```

`project_runtime_context_for_llm` must omit `current_timestamp_utc` and emit:

```python
{
    "time_context": local_time_context,
}
```

RAG helper agents that need deterministic time arithmetic must read `current_timestamp_utc`. RAG helper agents that need model-facing current time must read `local_time_context`.

### Self-Cognition Contract

`SelfCognitionCase` must use:

```python
idle_timestamp_utc: str
last_evidence_timestamp_utc: str
```

`SourcePacket` must use local model-facing names:

```python
idle_local_datetime: str
last_evidence_local_datetime: str
local_time_context: LocalTimeContextDoc
```

Rendered source packets must not include fields named `idle_timestamp`, `last_evidence_timestamp`, or any `_utc` value.

### Storage Contract

MongoDB document fields remain unchanged. When writing documents:

```python
conversation_doc["timestamp"] = storage_timestamp_utc
event_doc["execute_at"] = execute_at_utc
event_doc["created_at"] = created_at_utc
```

The right-hand variable must include `_utc`.

## LLM Call And Context Budget

No new LLM calls are allowed.

Affected existing LLM calls keep the same call counts and context budgets:

| Area | Before | After | Context impact |
|---|---:|---:|---|
| Live cognition stack | Same existing calls | Same existing calls | UTC/local field names change in Python state; prompt payload size remains equivalent. |
| RAG initializer/dispatch/evaluator/finalizer | Same existing calls | Same existing calls | `time_context` remains compact; raw UTC current timestamp is omitted. |
| Web search helper | Same existing calls | Same existing calls | Prompt timestamp value comes only from local context. |
| Consolidator facts and memory units | Same existing calls | Same existing calls | `timestamp` prompt field remains local `YYYY-MM-DD HH:MM`. |
| Reflection daily/promotion | Same existing calls | Same existing calls | UTC/timezone labels removed; local/opaque labels replace them. |
| Self-cognition | Same existing calls | Same existing calls | Source packets replace UTC source labels with local labels. |

No response-path latency increase is allowed.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/time_boundary.py`: canonical time source, conversion, parsing, formatting, and projection module.

### Delete

- `src/kazusa_ai_chatbot/time_context.py`: deleted after every import is migrated.

### Modify - Service And Queue

- `src/kazusa_ai_chatbot/brain_service/contracts.py`: replace `ChatRequest.timestamp` with `local_timestamp`; update docstrings where present.
- `src/kazusa_ai_chatbot/chat_input_queue.py`: replace `QueuedChatItem.timestamp`; use `TurnClock`; parse queue gaps from `storage_timestamp_utc`.
- `src/kazusa_ai_chatbot/brain_service/intake.py`: write conversation `timestamp` from `item.storage_timestamp_utc`.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`: read `storage_timestamp_utc` and `local_time_context`; write progress storage timestamps from `_utc` variables.
- `src/kazusa_ai_chatbot/brain_service/outbound.py`: use `storage_utc_now_iso` for assistant row timestamps.
- `src/kazusa_ai_chatbot/service.py`: remove direct UTC clock/parsing; build and thread `storage_timestamp_utc`, `local_timestamp`, and `local_time_context`.
- `src/adapters/debug_adapter.py`: accept optional `local_timestamp` in the proxy model; do not generate it in the UI.
- `src/kazusa_ai_chatbot/brain_service/README.md`: update `/chat` field documentation from `timestamp` to `local_timestamp`.

### Modify - Shared State And Episodes

- `src/kazusa_ai_chatbot/state.py`: rename `timestamp` and `time_context`.
- `src/kazusa_ai_chatbot/cognition_episode.py`: rename episode and projection fields.
- `src/kazusa_ai_chatbot/internal_thought_cognition.py`: rename dry-run arguments and state fields.
- `src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py`: rename dry-run arguments and state fields.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`: rename state fields.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: pass UTC to action/storage code and local context to prompt/model code.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: rename state setup.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2c2.py`: rename shared-state time reads.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`: rename state setup.
- `src/kazusa_ai_chatbot/nodes/persona_relevance_agent.py`: import UTC parsing from `time_boundary`.

### Modify - Consolidation And Actions

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`: pass `storage_timestamp_utc` and `local_time_context`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py`: rename fields.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`: rename origin timestamp field.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`: do not edit prompt text; rename Python state and payload variables so the model-facing `timestamp` payload is built from `local_time_context`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`: replace UTC/timezone-negative prompt wording with positive exact local output wording; build prompt `timestamp` from `local_time_context`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_persistence.py`: replace local and UTC conversion helpers with `time_boundary`; rename variables.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`: rename function arguments and docstrings to `_utc`.
- `src/kazusa_ai_chatbot/action_spec/execution.py`: rename `timestamp` argument to `storage_timestamp_utc`.
- `src/kazusa_ai_chatbot/action_spec/handlers/future_cognition.py`: rename timestamp arguments and use `time_boundary` normalization.
- `src/kazusa_ai_chatbot/action_spec/handlers/memory_lifecycle.py`: rename timestamp arguments and storage variables.

### Modify - RAG

- `src/kazusa_ai_chatbot/rag/prompt_projection.py`: import from `time_boundary`; update time-field formatter names; omit `current_timestamp_utc`.
- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`: output `current_timestamp_utc` and `local_time_context`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`: consume local prompt context without prompt timezone additions.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`: rename internal runtime context keys.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`: rename internal runtime context keys; keep model-facing `time_context` payload.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_prompt_views.py`: import renamed formatter.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`: replace debug fallback UTC clock with `TurnClock`; use `current_timestamp_utc` internally.
- `src/kazusa_ai_chatbot/rag/web_search_agent.py`: remove raw timestamp fallback; prompt timestamp comes from `local_time_context` only.
- `src/kazusa_ai_chatbot/rag/conversation_aggregate_agent.py`: use `parse_storage_utc_datetime`; read `current_timestamp_utc`.
- `src/kazusa_ai_chatbot/rag/conversation_filter_agent.py`: rename fixture/default context keys.
- `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`: use `time_boundary` UTC parsing and range helpers.
- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`: import renamed local formatting helper.
- `src/kazusa_ai_chatbot/rag/conversation_keyword_agent.py`: replace the `structured_llm_time_to_utc_iso` import and calls with `local_llm_datetime_to_storage_utc_iso`; keep the generator's local `YYYY-MM-DD HH:MM` output contract.
- `src/kazusa_ai_chatbot/rag/hybrid_retrieval.py`: rename timestamp variables used for storage evidence.
- `src/kazusa_ai_chatbot/rag/live_context_agent.py`: read `local_time_context` internally; emit model-facing `time_context` only in prompt payloads.
- `src/kazusa_ai_chatbot/rag/persistent_memory_keyword_agent.py`: keep the existing `project_runtime_context_for_llm` call; do not add direct time helper imports.
- `src/kazusa_ai_chatbot/rag/persistent_memory_search_agent.py`: keep the existing `project_runtime_context_for_llm` call; do not add direct time helper imports.
- `src/kazusa_ai_chatbot/rag/relationship_agent.py`: keep the existing `project_runtime_context_for_llm` call; do not add direct time helper imports.
- `src/kazusa_ai_chatbot/rag/search_runtime.py`: use `local_date_bounds_to_storage_utc_iso` and `one_second_before_storage_utc_iso`.
- `src/kazusa_ai_chatbot/rag/user_image_retriever_agent.py`: update projection import names.
- `src/kazusa_ai_chatbot/rag/user_list_agent.py`: update projection import names.
- `src/kazusa_ai_chatbot/rag/user_lookup_agent.py`: update projection import names.
- `src/kazusa_ai_chatbot/rag/user_memory_unit_retrieval.py`: rename `time_context` variables to `local_time_context`.
- `src/kazusa_ai_chatbot/rag/user_profile_agent.py`: rename `time_context` variables to `local_time_context`.
- `src/kazusa_ai_chatbot/rag/cache2_events.py`: rename cache invalidation timestamp field to storage UTC.
- `src/kazusa_ai_chatbot/rag/cache2_runtime.py`: use `storage_utc_now`.
- `src/kazusa_ai_chatbot/rag/README.md`: update internal runtime context names.

### Modify - Conversation Progress

- `src/kazusa_ai_chatbot/conversation_progress/models.py`: rename `ConversationProgressRecordInput.timestamp` to `storage_timestamp_utc`; keep prompt-facing `ConversationProgressEntry.age_hint` unchanged.
- `src/kazusa_ai_chatbot/conversation_progress/policy.py`: use `time_boundary` UTC parsing/normalization.
- `src/kazusa_ai_chatbot/conversation_progress/projection.py`: rename `current_timestamp` to `current_timestamp_utc`.
- `src/kazusa_ai_chatbot/conversation_progress/runtime.py`: rename `current_timestamp` to `current_timestamp_utc`.
- `src/kazusa_ai_chatbot/conversation_progress/repository.py`: rename `timestamp` arguments to `_utc`.
- `src/kazusa_ai_chatbot/conversation_progress/recorder.py`: payload `current_turn_timestamp` remains local, built from `local_time_context`; storage writes use `_utc`.
- `src/kazusa_ai_chatbot/conversation_progress/README.md`: update internal argument names.

### Modify - Self-Cognition

- `src/kazusa_ai_chatbot/self_cognition/models.py`: rename source UTC and source-packet local fields.
- `src/kazusa_ai_chatbot/self_cognition/sources.py`: produce `_utc` source fields and local source-packet fields through `time_boundary`.
- `src/kazusa_ai_chatbot/self_cognition/projection.py`: render only local model-facing fields; project RAG evidence through renamed formatter helpers.
- `src/kazusa_ai_chatbot/self_cognition/runner.py`: build shared state with `storage_timestamp_utc` and `local_time_context`.
- `src/kazusa_ai_chatbot/self_cognition/worker.py`: use `storage_utc_now`.
- `src/kazusa_ai_chatbot/self_cognition/tracking.py`: rename local trigger and run artifact keys from `idle_timestamp` to `idle_timestamp_utc`; no legacy `idle_timestamp` key remains in the new artifact schema.
- `src/kazusa_ai_chatbot/self_cognition/README.md`: update artifact schemas and field meanings.

### Modify - Scheduler, Dispatcher, Proactive, Event Logging, DB, Reflection, Growth

- `src/kazusa_ai_chatbot/scheduler.py`: use `storage_utc_now` and `parse_storage_utc_datetime`.
- `src/kazusa_ai_chatbot/dispatcher/task.py`: use `time_boundary` UTC parse/normalization.
- `src/kazusa_ai_chatbot/dispatcher/handlers.py`: use `storage_utc_now_iso`.
- `src/kazusa_ai_chatbot/dispatcher/remote_adapter.py`: use `storage_utc_now` and `parse_storage_utc_datetime`.
- `src/kazusa_ai_chatbot/proactive_output/policy.py`: use `time_boundary` UTC parse/normalization.
- `src/kazusa_ai_chatbot/event_logging/recording.py`: use `storage_utc_now_iso` and UTC normalization.
- `src/kazusa_ai_chatbot/event_logging/status.py`: use `storage_utc_now`.
- `src/kazusa_ai_chatbot/event_logging/snapshots.py`: use `storage_utc_now`.
- `src/kazusa_ai_chatbot/db/bootstrap.py`: use `storage_utc_now_iso`.
- `src/kazusa_ai_chatbot/db/character.py`: rename `upsert_character_state(timestamp)` to `upsert_character_state(updated_at_utc)` while preserving the persisted `updated_at` field.
- `src/kazusa_ai_chatbot/db/conversation.py`: bind assistant/user conversation row storage timestamps through `_utc` locals.
- `src/kazusa_ai_chatbot/db/event_logging.py`: use `storage_utc_now_iso` and window helper from `time_boundary`.
- `src/kazusa_ai_chatbot/db/memory.py`: use `storage_utc_now_iso`.
- `src/kazusa_ai_chatbot/db/memory_evolution.py`: rename active-memory expiry cutoff variables to `_utc`.
- `src/kazusa_ai_chatbot/db/interaction_style_images.py`: use `storage_utc_now_iso`.
- `src/kazusa_ai_chatbot/db/rag_cache2_persistent.py`: use `storage_utc_now_iso`.
- `src/kazusa_ai_chatbot/db/scheduled_events.py`: rename `current_timestamp` parameters and docstrings to `current_timestamp_utc`; preserve stored `execute_at` comparisons as UTC strings.
- `src/kazusa_ai_chatbot/db/users.py`: use `storage_utc_now_iso`.
- `src/kazusa_ai_chatbot/db/user_memory_units.py`: use `storage_utc_now_iso` and `parse_storage_utc_datetime`.
- `src/kazusa_ai_chatbot/memory_evolution/repository.py`: use `storage_utc_now_iso` and `parse_storage_utc_datetime`.
- `src/kazusa_ai_chatbot/memory_evolution/reset.py`: use `time_boundary` helpers for reset timestamps.
- `src/kazusa_ai_chatbot/memory_evolution/README.md`: update public helper signatures to `_utc`.
- `src/kazusa_ai_chatbot/global_character_growth/context.py`: import renamed local time context type.
- `src/kazusa_ai_chatbot/global_character_growth/runner.py`: use `time_boundary` for current UTC and character-local date derivation.
- `src/kazusa_ai_chatbot/global_character_growth/projection.py`: leave date-only parsing local to `date.fromisoformat`; import renamed formatter only where timestamp projection occurs.
- `src/kazusa_ai_chatbot/reflection_cycle/selector.py`: use `storage_utc_now` and UTC normalization helpers.
- `src/kazusa_ai_chatbot/reflection_cycle/runtime.py`: use `time_boundary` for UTC parsing, hour start, now labels, and run ids.
- `src/kazusa_ai_chatbot/reflection_cycle/repository.py`: use `time_boundary` for UTC parsing and local date projection; remove direct `ZoneInfo`.
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`: use `time_boundary` for UTC now and character-local date derivation.
- `src/kazusa_ai_chatbot/reflection_cycle/projection.py`: replace model-facing UTC hour labels with configured local labels.
- `src/kazusa_ai_chatbot/reflection_cycle/prompts.py`: remove UTC/timezone wording from prompt text; describe hour fields as copy-exact local hour labels.
- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`: remove `character_time_zone` from prompt payload and prompt input schema.
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`: update internal design notes so runtime docs may mention UTC storage, but prompt-facing docs do not instruct LLM timezone reasoning.

### Modify - Scripts

- `src/scripts/export_user_memories.py`
- `src/scripts/export_chat_history.py`
- `src/scripts/character_state_snapshot.py`
- `src/scripts/user_state_snapshot.py`
- `src/scripts/profile_rag_retrieval.py`
- `src/scripts/export_event_log.py`
- `src/scripts/fetch_ops_status.py`
- `src/scripts/_db_export.py`
- `src/scripts/run_reflection_cycle.py`
- `src/scripts/run_reflection_cycle_readonly.py`
- `src/scripts/run_global_character_growth.py`
- `src/scripts/sanitize_memory_writer_perspective.py`
- `src/scripts/profile_embedding_prefix_modes.py`

These scripts must import time helpers from `time_boundary` for current UTC, UTC parsing, or configured-local date derivation. `_db_export.utc_now` must delegate to `storage_utc_now`; `_db_export.timestamp_hours_ago` must return normalized storage UTC strings; `export_chat_history.py` must rename internal `from_timestamp` and `to_timestamp` variables to `_utc` names while preserving CLI argument names; `run_reflection_cycle_readonly.py` must parse `--now` through `parse_storage_utc_datetime`. CLI argument names like `--from-timestamp` may remain when the argument describes external operator input.

### Modify - Tests

- Replace imports from `kazusa_ai_chatbot.time_context` with `kazusa_ai_chatbot.time_boundary`.
- Rename `tests/test_time_context.py` to `tests/test_time_boundary.py`.
- Update all tests returned by `rg -l 'time_context|build_character_time_context|TimeContextDoc|current_timestamp|timestamp=' tests`.
- Review-identified focused test files include `tests/test_rag_helper_arg_boundaries.py` and `tests/test_memory_evolution_retrieval.py`.
- Add or update these focused tests:
  - `tests/test_time_boundary.py::test_build_turn_clock_preserves_configured_local_input`
  - `tests/test_time_boundary.py::test_build_turn_clock_from_storage_utc_converts_to_configured_local`
  - `tests/test_time_boundary.py::test_local_input_rejects_offset_utc_and_timezone_markers`
  - `tests/test_time_boundary.py::test_llm_local_datetime_to_storage_utc_rejects_offset_iso`
  - `tests/test_service_input_queue.py::test_queue_separates_storage_utc_and_local_timestamp`
  - `tests/test_service_input_queue.py::test_chat_request_timestamp_field_is_rejected`
  - `tests/test_llm_time_payload_projection.py::test_regression_local_turn_time_does_not_shift_to_next_day`
  - `tests/test_llm_time_payload_projection.py::test_prompt_payloads_do_not_contain_timezone_concepts`
  - `tests/test_self_cognition_framing.py::test_self_cognition_source_packet_uses_local_time_labels`
  - `tests/test_reflection_cycle_prompt_contracts.py::test_reflection_prompts_do_not_expose_timezone_concepts`

## Implementation Order

1. Run `git status --short` and record the baseline.
2. Add `tests/test_time_boundary.py` with the new module-contract tests listed above. Run the focused tests and record the expected import/symbol failures.
3. Create `src/kazusa_ai_chatbot/time_boundary.py` with the exact public interface in this plan. Run `tests/test_time_boundary.py` until it passes.
4. Migrate service API and queue contracts:
   - `brain_service/contracts.py`
   - `chat_input_queue.py`
   - `brain_service/intake.py`
   - `service.py`
   - `brain_service/README.md`
   Run `tests/test_service_input_queue.py` focused tests.
5. Migrate shared state and cognitive episode contracts:
   - `state.py`
   - `cognition_episode.py`
   - `internal_thought_cognition.py`
   - `reflection_cycle/cognition_dry_run.py`
   - persona state/schema files
   Run cognitive episode and persona focused tests.
6. Migrate RAG runtime context and projection helpers. Run RAG projection, RAG adapter, and RAG helper focused tests.
7. Migrate consolidation, action-spec, scheduler, dispatcher, and proactive policy paths. Run action, scheduler, consolidation, and DB writer focused tests.
8. Migrate self-cognition source, projection, runner, worker, tracking, and README contracts. Run self-cognition focused tests.
9. Migrate reflection, event logging, DB helpers, memory evolution, global character growth, and scripts to `time_boundary` helpers. Run focused tests for those subsystems.
10. Apply the allowed prompt cleanup only in:
    - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`
    - `src/kazusa_ai_chatbot/reflection_cycle/prompts.py`
    - `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
    Run prompt contract tests and CJK syntax checks.
11. Delete `src/kazusa_ai_chatbot/time_context.py`. Run static grep gates.
12. Run the full deterministic verification list.
13. Run independent code review, fix in-scope findings, rerun affected verification, and record evidence.

## Progress Checklist

- [x] Stage 1 - module contract established
  - Covers: implementation steps 1-3.
  - Verify: `venv\Scripts\python -m pytest tests\test_time_boundary.py -q`.
  - Evidence: record expected failures before implementation and final passing output.
  - Handoff: next unchecked stage is Stage 2.
  - Sign-off: `Codex/2026-05-17` after `venv\Scripts\python -m pytest tests\test_time_boundary.py -q` reported `23 passed in 0.06s`.
- [x] Stage 2 - service and queue boundary migrated
  - Covers: implementation step 4.
  - Verify: `venv\Scripts\python -m pytest tests\test_service_input_queue.py -q`.
  - Evidence: record changed files and focused test output.
  - Handoff: next unchecked stage is Stage 3.
  - Sign-off: `Codex/2026-05-17` after `venv\Scripts\python -m pytest tests\test_service_input_queue.py -q` reported `30 passed in 1.86s`.
- [x] Stage 3 - graph and cognitive episode state migrated
  - Covers: implementation step 5.
  - Verify: `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py tests\test_persona_supervisor2.py tests\test_persona_supervisor2_action_initializer.py -q`.
  - Evidence: record focused test output.
  - Handoff: next unchecked stage is Stage 4.
  - Sign-off: `Codex/2026-05-17` after `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py tests\test_persona_supervisor2.py tests\test_persona_supervisor2_action_initializer.py -q` reported `35 passed in 1.75s`.
- [x] Stage 4 - RAG runtime context migrated
  - Covers: implementation step 6.
  - Verify: `venv\Scripts\python -m pytest tests\test_llm_time_payload_projection.py tests\test_rag_cognitive_episode_adapter.py tests\test_rag_phase3_capability_agents.py tests\test_rag_hybrid_agents.py -q`.
  - Evidence: record focused test output and static UTC leak assertions.
  - Handoff: next unchecked stage is Stage 5.
  - Sign-off: `Codex/2026-05-17` after `venv\Scripts\python -m pytest tests\test_llm_time_payload_projection.py tests\test_rag_cognitive_episode_adapter.py tests\test_rag_phase3_capability_agents.py tests\test_rag_hybrid_agents.py -q` reported `96 passed in 1.81s` and the focused RAG static scans were clean.
- [x] Stage 5 - consolidation, action, scheduler, and DB storage paths migrated
  - Covers: implementation step 7.
  - Verify: `venv\Scripts\python -m pytest tests\test_action_spec_future_cognition.py tests\test_action_spec_memory_lifecycle.py tests\test_scheduler_future_promise.py tests\test_db_writer_cache2_invalidation.py tests\test_consolidator_efficiency.py -q`.
  - Evidence: record focused test output.
  - Handoff: next unchecked stage is Stage 6.
  - Sign-off: `Codex/2026-05-17` after `venv\Scripts\python -m pytest tests\test_action_spec_future_cognition.py tests\test_action_spec_memory_lifecycle.py tests\test_scheduler_future_promise.py tests\test_db_writer_cache2_invalidation.py tests\test_consolidator_efficiency.py -q` reported `28 passed in 1.57s`.
- [x] Stage 6 - self-cognition migrated
  - Covers: implementation step 8.
  - Verify: `venv\Scripts\python -m pytest tests\test_self_cognition_framing.py tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_self_cognition_event_logging.py -q`.
  - Evidence: record focused test output.
  - Handoff: next unchecked stage is Stage 7.
  - Sign-off: `Codex/2026-05-17` after `venv\Scripts\python -m pytest tests\test_self_cognition_framing.py tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_self_cognition_event_logging.py -q` reported `61 passed in 3.84s`; the self-cognition py_compile command succeeded; and the focused self-cognition static grep returned no matches.
- [x] Stage 7 - reflection, growth, event logging, DB helpers, and scripts migrated
  - Covers: implementation step 9.
  - Verify: `venv\Scripts\python -m pytest tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_cycle_stage1c_worker.py tests\test_global_character_growth_runner.py tests\test_event_logging_interface.py -q`.
  - Evidence: record focused test output.
  - Handoff: next unchecked stage is Stage 8.
  - Sign-off: `Codex/2026-05-17` after `venv\Scripts\python -m pytest tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_cycle_stage1c_worker.py tests\test_global_character_growth_runner.py tests\test_event_logging_interface.py -q` reported `36 passed in 1.43s`; the Stage 7 static grep returned no matches; and py_compile succeeded over the touched Stage 7 production/script files.
- [x] Stage 8 - allowed prompt cleanup complete
  - Covers: implementation step 10.
  - Verify: prompt contract tests plus CJK syntax checks listed in `Verification`.
  - Evidence: record exact prompt files changed and checks run.
  - Handoff: next unchecked stage is Stage 9.
  - Sign-off: `Codex/2026-05-17` after `venv\Scripts\python -m pytest tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_cycle_stage1c_promotion.py -q` reported `26 passed in 1.43s`; the allowed prompt-file static grep returned no matches; and CJK py_compile succeeded over the three allowed prompt cleanup files.
- [x] Stage 9 - legacy names removed and regression suite passed
  - Covers: implementation steps 11-12.
  - Verify: every static grep and deterministic test command in `Verification`.
  - Evidence: record grep outputs and test output.
  - Handoff: next unchecked stage is Stage 10.
  - Sign-off: `Codex/2026-05-17` after the static grep gates, focused tests, subsystem regressions, CJK syntax checks, runtime smoke, and `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q` all passed with the evidence recorded below.
- [x] Stage 10 - independent code review completed
  - Covers: implementation step 13.
  - Verify: independent review findings are recorded; affected checks rerun after fixes.
  - Evidence: record reviewer mode, files reviewed, findings, fixes, rerun commands, residual risks, and approval status.
  - Handoff: plan may move to completed only after this stage is signed off.
  - Sign-off: `Codex/2026-05-17` after self-review findings were fixed, affected checks passed, static gates were clean, and `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q` reported `1428 passed, 256 deselected in 21.04s`.

## Verification

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_time_boundary.py -q`
- `venv\Scripts\python -m pytest tests\test_service_input_queue.py -q`
- `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py -q`
- `venv\Scripts\python -m pytest tests\test_llm_time_payload_projection.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_framing.py -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_prompt_contracts.py -q`

### Subsystem Regression Tests

- `venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_persona_supervisor2_action_initializer.py tests\test_persona_supervisor2_rag2_integration.py -q`
- `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py tests\test_rag_phase3_capability_agents.py tests\test_rag_hybrid_agents.py tests\test_rag_finalizer_time_context.py -q`
- `venv\Scripts\python -m pytest tests\test_action_spec_future_cognition.py tests\test_action_spec_memory_lifecycle.py tests\test_scheduler_future_promise.py -q`
- `venv\Scripts\python -m pytest tests\test_consolidator_efficiency.py tests\test_consolidator_facts_rag2.py tests\test_consolidator_source_aware_payloads.py -q`
- `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_self_cognition_event_logging.py -q`
- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_worker.py tests\test_reflection_cycle_stage1c_repository.py tests\test_global_character_growth_runner.py tests\test_event_logging_interface.py -q`

### Broad Deterministic Test

- `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q`

### Static Greps

- `rg -n "from kazusa_ai_chatbot\\.time_context|import kazusa_ai_chatbot\\.time_context|time_context.py" src tests`
  - Expected: zero matches except historical development plan/archive text. Any source or test match is a failure.
- `rg -n "\\bTimeContextDoc\\b|build_character_time_context|format_timestamp_for_llm|format_history_for_llm|format_time_fields_for_llm|local_llm_time_to_utc_iso|structured_llm_time_to_utc_iso|local_date_bounds_to_utc_iso" src tests`
  - Expected: zero matches except historical development plan/archive text. Any source or test match is a failure.
- `rg -n "request\\.timestamp|item\\.timestamp|state\\[\"timestamp\"\\]|state\\.get\\(\"timestamp\"\\)|current_timestamp\\b|\\btimestamp: str" src\\kazusa_ai_chatbot tests`
  - Expected: zero runtime matches for internal state, request, or queue fields. Allowed matches are MongoDB schema/document field literals and tests that assert old `/chat` field rejection.
- `rg -n "datetime\\.now\\(timezone\\.utc\\)|datetime\\.datetime\\.now\\(datetime\\.timezone\\.utc\\)|datetime\\.fromisoformat\\(|datetime\\.datetime\\.fromisoformat\\(|\\.astimezone\\(|ZoneInfo\\(" src\\kazusa_ai_chatbot src\\scripts`
  - Expected: matches only in `src/kazusa_ai_chatbot/time_boundary.py`. `date.fromisoformat` is not part of this grep and remains allowed for date-only parsing.
- `rg -n "character_time_zone|IANA timezone|UTC hour-start|timezone|time zone|时区|Pacific/Auckland|America/|Europe/|Asia/" src\\kazusa_ai_chatbot\\nodes src\\kazusa_ai_chatbot\\rag src\\kazusa_ai_chatbot\\reflection_cycle`
  - Expected: zero prompt-constant matches. Docstrings and READMEs may mention implementation storage rules; prompt constants must not.
- `rg -n "\"current_timestamp\"|current_timestamp:" src\\kazusa_ai_chatbot tests`
  - Expected: zero matches except tests that assert legacy key rejection.
- `rg -n "\"time_context\"" src\\kazusa_ai_chatbot tests`
  - Expected: allowed only in model-facing JSON payload construction, prompt examples that already consume the key, and tests verifying prompt payloads. Internal state schemas must use `local_time_context`.

### CJK Syntax Checks

Run after editing any Python file that contains Chinese prompt text:

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py`
- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\prompts.py`
- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\promotion.py`

### Runtime Smoke

- `venv\Scripts\python -c "import kazusa_ai_chatbot.service; print('service import ok')"`

No live LLM or live DB test is required for this bugfix. Run live tests only on explicit user request.

## Independent Plan Review

Run this gate before approval or execution. Prefer a reviewer that did not draft the plan. If no separate reviewer is available, the active agent must reread this plan, the completed `character_local_time_context_plan.md`, the relevant source files, and this review scope from a fresh-review posture.

Review scope:

- The plan encodes the user-confirmed rule: UTC storage, configured local elsewhere.
- The plan does not leave any source-of-truth decision to the implementation agent.
- The plan does not authorize prompt edits outside timezone-removal cases.
- The plan deletes ambiguous helper names instead of preserving aliases.
- The plan keeps durable MongoDB field names stable and avoids hidden data migration.
- The change surface is sufficient for every known call path: `/chat`, queue, graph, RAG, consolidation, action, scheduler, self-cognition, reflection, DB helper current-time calls, event logging, scripts, and tests.

Record blockers, non-blocking findings, required edits, and approval status. Approval is valid only after blockers are resolved.

Review record:

- 2026-05-17: independent subagent review reported missing change-surface entries, a prompt-boundary conflict, and loose conditional wording. The plan now names `run_reflection_cycle_readonly.py`, `export_chat_history.py`, `db/scheduled_events.py`, and `db/character.py`; forbids prompt text edits in `persona_supervisor2_consolidator_facts.py`; replaces the loose conditional wording with concrete file instructions; and is approved for execution.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. Prefer a reviewer that did not implement the change. If no separate reviewer is available, the active agent must reread this plan, inspect the full diff from a fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt, documentation, and command artifact.
- Code quality and design weaknesses, including source-of-truth drift, hidden fallback paths, compatibility aliases, prompt/RAG payload leaks, persistence risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Cutover Policy`, `Agent Autonomy Boundaries`, `Change Surface`, exact contracts, implementation order, verification gates, and acceptance criteria.
- Regression and handoff quality, including focused tests, static greps, execution evidence, and path-safe commands.

Fix concrete findings directly only when the fix is inside the approved change surface. If a fix would cross the approved boundary or alter the contract, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `src/kazusa_ai_chatbot/time_boundary.py` exists and owns all production timezone conversion and UTC clock creation.
- `src/kazusa_ai_chatbot/time_context.py` is deleted.
- `/chat` accepts `local_timestamp` and rejects legacy `timestamp`.
- Internal service, queue, graph, RAG, self-cognition, action, scheduler, reflection, event logging, DB helper, and script code use `_utc` names for storage instants and local names for configured local time.
- Existing MongoDB field names remain unchanged and continue to store UTC values.
- Prompt payloads contain configured local strings only and no UTC offsets, IANA timezone names, or timezone instructions.
- The regression case `2026-05-17 16:55:28.395` local input reaches LLM payload builders as `2026-05-17 16:55`, not `2026-05-18 04:55`.
- All verification commands pass or any blocked command is recorded with the exact blocker.
- Independent code review is recorded and has no unresolved blocker.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Broad rename misses a state key | Static greps for old names and focused graph tests | Static grep gates and persona tests |
| Prompt edited beyond allowed timezone removal | Restrict prompt change surface to three files | Independent code review and prompt contract tests |
| Stored field rename sneaks into DB writes | Keep MongoDB field names stable and require `_utc` local variables | DB writer tests and diff review |
| LLM receives UTC through nested RAG evidence | Project tool results through renamed central formatters | `test_llm_time_payload_projection.py` and UTC leak grep |
| Self-cognition artifacts break dry-run users | Update self-cognition README and tests with explicit `_utc` and local fields | Self-cognition focused tests |
| Direct datetime conversion reappears | Forbid scattered conversion calls outside `time_boundary.py` | Static grep gate |

## Execution Evidence

- Baseline git status: before implementation, `git status --short --untracked-files=all` showed `M development_plans/README.md`, `?? development_plans/active/bugfix/self_cognition_speak_delivery_bugfix_plan.md`, and `?? development_plans/active/bugfix/time_source_boundary_bugfix_plan.md`.
- Focused test failures before implementation: `venv\Scripts\python -m pytest tests\test_time_boundary.py -q` failed during collection with `ModuleNotFoundError: No module named 'kazusa_ai_chatbot.time_boundary'`, which is the expected RED result for Stage 1 before creating the canonical module.
- Module test results: `venv\Scripts\python -m pytest tests\test_time_boundary.py -q` reported `23 passed in 0.06s` after `src/kazusa_ai_chatbot/time_boundary.py` was created.
- Service/queue test results: before Stage 2 production migration, `venv\Scripts\python -m pytest tests\test_service_input_queue.py -q` reported `30 failed in 2.59s`; the representative failure was old `ChatRequest` rejecting the new `local_timestamp` field as `extra_forbidden`. After migrating `ChatRequest`, `QueuedChatItem`, queue coalescing, intake persistence, service queue timing, and the `/chat` README field contract, the same command reported `30 passed in 1.86s`.
- Graph/RAG/consolidation/action/self-cognition/reflection test results: before Stage 3 production migration, `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py tests\test_persona_supervisor2.py tests\test_persona_supervisor2_action_initializer.py -q` reported `31 failed, 4 passed in 2.07s`; representative failures were old `CognitiveEpisode` annotations still exposing `timestamp`/`time_context` and `build_text_chat_cognitive_episode` rejecting `storage_timestamp_utc`. After migrating graph state, cognitive episode contracts, persona state plumbing, and the self-cognition runner's episode construction, the same command reported `35 passed in 1.75s`. Before Stage 4 production migration, `venv\Scripts\python -m pytest tests\test_llm_time_payload_projection.py tests\test_rag_cognitive_episode_adapter.py tests\test_rag_phase3_capability_agents.py tests\test_rag_hybrid_agents.py -q` reported `16 failed, 80 passed in 2.25s`; representative failures were old RAG contexts still reading or emitting `current_timestamp`/`time_context` instead of `current_timestamp_utc`/`local_time_context`. After migrating RAG runtime context, prompt projection, live-context runtime time reads, RAG search time bounds, and prompt-safe formatting helpers, the same command reported `96 passed in 1.81s`; a focused old-helper static scan returned zero matches, and `rg -n '\bcurrent_timestamp\b' src\kazusa_ai_chatbot\rag tests\test_llm_time_payload_projection.py tests\test_rag_cognitive_episode_adapter.py tests\test_rag_phase3_capability_agents.py tests\test_rag_hybrid_agents.py` found only two test assertions that the legacy key is absent.
- Stage 5 focused RED result: before production migration, `venv\Scripts\python -m pytest tests\test_action_spec_future_cognition.py tests\test_action_spec_memory_lifecycle.py tests\test_scheduler_future_promise.py tests\test_db_writer_cache2_invalidation.py tests\test_consolidator_efficiency.py -q` reported `19 failed, 9 passed in 1.77s`; representative failures were action handlers rejecting the new `storage_timestamp_utc` keyword, scheduler and dispatcher paths ignoring patched `storage_utc_now_iso`, scheduled-event queries rejecting `current_timestamp_utc`, and consolidation origin still reading `episode["timestamp"]`.
- Stage 5 focused GREEN result: after migrating action-spec execution and handlers, scheduler and dispatcher clock reads, scheduled-event query names, consolidation origin/persistence/image paths, conversation-progress storage timestamp names, recall scheduled-event query names, and adjacent memory-unit due checks, `venv\Scripts\python -m pytest tests\test_action_spec_future_cognition.py tests\test_action_spec_memory_lifecycle.py tests\test_scheduler_future_promise.py tests\test_db_writer_cache2_invalidation.py tests\test_consolidator_efficiency.py -q` reported `28 passed in 1.57s`. Adjacent signature regression coverage with `venv\Scripts\python -m pytest tests\test_conversation_episode_state.py tests\test_rag_recall_agent.py -q` reported `14 passed in 1.30s`, and the combined command over those seven files reported `42 passed in 1.52s`. Focused conversion scans over the touched production files returned zero forbidden direct conversion calls; the focused naming scan left only allowed persisted/external API `timestamp=` keyword uses such as `upsert_character_state(..., timestamp=storage_timestamp_utc)`, `CacheInvalidationEvent(timestamp=storage_timestamp_utc)`, and outbound conversation-history writes.
- Stage 6 focused RED result: before self-cognition production migration, `venv\Scripts\python -m pytest tests\test_self_cognition_framing.py tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_self_cognition_event_logging.py -q` reported `26 failed, 35 passed in 5.01s`; representative failures were source-packet projection still reading old `idle_timestamp`/`last_evidence_timestamp`, RAG request context still emitting `current_timestamp` and `time_context`, `time_context.py` fallback warnings, runner/tracking records expecting old case keys, and rendered source packets leaking storage UTC strings.
- Stage 6 focused GREEN result: after migrating self-cognition source case fields, source packet projection, RAG request context, runner state, tracking artifacts, worker clock source, and README schema, `venv\Scripts\python -m pytest tests\test_self_cognition_framing.py tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_self_cognition_event_logging.py -q` reported `61 passed in 3.84s`. `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\self_cognition\models.py src\kazusa_ai_chatbot\self_cognition\projection.py src\kazusa_ai_chatbot\self_cognition\runner.py src\kazusa_ai_chatbot\self_cognition\sources.py src\kazusa_ai_chatbot\self_cognition\tracking.py src\kazusa_ai_chatbot\self_cognition\worker.py` succeeded, and the focused self-cognition grep for old time-context imports, legacy source fields, `current_timestamp`, model-facing `time_context`, and direct conversion calls returned no matches.
- Stage 7 focused RED result: after adding Stage 7 contract tests, `venv\Scripts\python -m pytest tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_cycle_stage1c_worker.py tests\test_global_character_growth_runner.py tests\test_event_logging_interface.py -q` reported `4 failed, 32 passed in 1.61s`; the representative failures were missing `storage_utc_now`/`storage_utc_now_iso` imports in global growth, event recording, and status builders, plus the Stage 7 source-surface static guard finding remaining direct UTC clock reads, direct datetime parsing, direct timezone conversion, `ZoneInfo(...)`, and a legacy `time_context` import.
- Stage 7 follow-up RED result: after diff review found reflection daily active-hour projection still model-facing UTC hour labels, the focused Stage 7 command reported `1 failed, 35 passed in 1.54s`; the failing assertion expected `active_hour_slots[0].hour` to be configured-local `2026-05-04 10:00` instead of storage UTC `2026-05-03T22:00:00+00:00`.
- Stage 7 focused GREEN result: after migrating event logging, DB helpers, memory evolution, global character growth, reflection runtime/repository/projection, and scripts to `time_boundary`, and after correcting reflection active-hour projection to configured-local labels, `venv\Scripts\python -m pytest tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_cycle_stage1c_worker.py tests\test_global_character_growth_runner.py tests\test_event_logging_interface.py -q` reported `36 passed in 1.43s`. The Stage 7 static grep for legacy `time_context` imports, direct UTC clocks, direct datetime parsing, direct timezone conversion, and `ZoneInfo(...)` returned no matches. `venv\Scripts\python -m py_compile` over the touched Stage 7 production/script files succeeded.
- Stage 8 focused RED result: after adding the prompt-boundary contract test, `venv\Scripts\python -m pytest tests\test_reflection_cycle_prompt_contracts.py -q` reported `1 failed, 9 passed in 1.42s`; the failing prompt scan found forbidden UTC/timezone/ISO timestamp concepts in the allowed cleanup files.
- Stage 8 focused GREEN result: after cleaning only `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`, `src/kazusa_ai_chatbot/reflection_cycle/prompts.py`, and `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`, and after updating adjacent promotion tests to assert configured-local evidence labels, `venv\Scripts\python -m pytest tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_cycle_stage1c_promotion.py -q` reported `26 passed in 1.43s`. The allowed prompt-file static grep for UTC/timezone concepts returned no matches, and `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py src\kazusa_ai_chatbot\reflection_cycle\prompts.py src\kazusa_ai_chatbot\reflection_cycle\promotion.py` succeeded.
- Stage 9 current-tree focused verification: after cancelling subagent use and keeping all remaining changes local, these commands passed: `venv\Scripts\python -m pytest tests\test_time_boundary.py -q` reported `23 passed in 0.07s`; `venv\Scripts\python -m pytest tests\test_service_input_queue.py -q` reported `30 passed in 2.00s`; `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py -q` reported `16 passed in 1.38s`; `venv\Scripts\python -m pytest tests\test_llm_time_payload_projection.py -q` reported `19 passed in 1.76s`; `venv\Scripts\python -m pytest tests\test_self_cognition_framing.py -q` reported `2 passed in 1.59s`; and `venv\Scripts\python -m pytest tests\test_reflection_cycle_prompt_contracts.py -q` reported `10 passed in 1.36s`.
- Stage 9 subsystem regression verification: `venv\Scripts\python -m pytest tests\test_persona_supervisor2.py tests\test_persona_supervisor2_action_initializer.py tests\test_persona_supervisor2_rag2_integration.py -q` reported `26 passed in 1.85s`; `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py tests\test_rag_phase3_capability_agents.py tests\test_rag_hybrid_agents.py tests\test_rag_finalizer_time_context.py -q` reported `83 passed in 1.79s`; `venv\Scripts\python -m pytest tests\test_action_spec_future_cognition.py tests\test_action_spec_memory_lifecycle.py tests\test_scheduler_future_promise.py -q` reported `23 passed in 1.44s`; `venv\Scripts\python -m pytest tests\test_consolidator_efficiency.py tests\test_consolidator_facts_rag2.py tests\test_consolidator_source_aware_payloads.py -q` reported `14 passed in 1.47s`; `venv\Scripts\python -m pytest tests\test_self_cognition_tracking.py tests\test_self_cognition_integration.py tests\test_self_cognition_event_logging.py -q` reported `59 passed in 4.22s`; and `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_worker.py tests\test_reflection_cycle_stage1c_repository.py tests\test_global_character_growth_runner.py tests\test_event_logging_interface.py -q` reported `32 passed in 1.54s`.
- Static grep results: `rg -n "from kazusa_ai_chatbot\\.time_context|import kazusa_ai_chatbot\\.time_context|time_context.py" src tests` returned no matches; `rg -n '\bTimeContextDoc\b|build_character_time_context|format_timestamp_for_llm|format_history_for_llm|format_time_fields_for_llm|local_llm_time_to_utc_iso|structured_llm_time_to_utc_iso|local_date_bounds_to_utc_iso' src tests` returned no matches; `rg -n 'request\.timestamp|item\.timestamp|state\["timestamp"\]|state\.get\("timestamp"\)|current_timestamp\b|\btimestamp: str' src\kazusa_ai_chatbot tests` returned only durable schema `timestamp: str` fields in `src/kazusa_ai_chatbot/db/schemas.py` and `src/kazusa_ai_chatbot/memory_evolution/models.py`; direct datetime conversion grep returned matches only in `src/kazusa_ai_chatbot/time_boundary.py`; prompt timezone grep over nodes, RAG, and reflection returned no matches; `rg -n '"current_timestamp"|current_timestamp:' src\kazusa_ai_chatbot tests` returned no matches; and `rg -n '"time_context"' src\kazusa_ai_chatbot tests` returned only legacy-rejection assertions, model-facing prompt payload construction, prompt examples, and prompt payload tests.
- CJK syntax checks: `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py src\kazusa_ai_chatbot\reflection_cycle\prompts.py src\kazusa_ai_chatbot\reflection_cycle\promotion.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py` succeeded; `venv\Scripts\python -m compileall -q src tests` succeeded; and `venv\Scripts\python -c "import kazusa_ai_chatbot.service; print('service import ok')"` printed `service import ok`.
- Independent plan review result: 2026-05-17 approved after resolving the independent subagent blockers listed in `Independent Plan Review`.
- Broad deterministic test result: the first current-tree broad run exposed stale test fixtures and fingerprint assertions (`40 failed, 1385 passed, 256 deselected in 21.26s`); after local fixture and test-contract updates without subagent edits, `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q` reported `1425 passed, 256 deselected in 20.83s`.
- Independent code review result: no separate reviewer was used after the user cancelled subagent work on 2026-05-17; the active agent performed the review from a fresh-review posture after rereading this full plan. Reviewed full diff stat/name-status, `time_boundary.py`, service/queue state threading, cognitive episode/RAG projection, allowed prompt diffs, cognition L3 non-prompt diff, action/scheduler/event logging, memory evolution, scripts, and the tests updated for the time-boundary contract.
- Independent code review findings fixed locally:
  - Removed trailing blank-line-at-EOF issues from `tests/test_self_cognition_event_logging.py`, `tests/test_self_cognition_integration.py`, and `tests/test_self_cognition_tracking.py`.
  - Added `local_timestamp` to `src/adapters/debug_adapter.py`.
  - Normalized operator-supplied chat-history export timestamps and renamed internal variables in `src/scripts/export_chat_history.py`.
  - Normalized `_db_export.timestamp_hours_ago(...)` return values through `time_boundary`.
  - Restored `_parse_datetime_for_query(...) -> datetime` typing in `src/kazusa_ai_chatbot/db/user_memory_units.py`.
  - Renamed remaining memory-evolution and memory active-query cutoff locals/keywords to `now_timestamp_utc`.
  - Removed the conversation aggregate UTC calendar-day fallback for `today`/`yesterday`; added configured-local tests.
  - Removed the unnecessary current-clock dependency for the aggregate `all` time window; added `test_conversation_aggregate_all_does_not_require_current_time`.
  - Updated this plan's change surface to explicitly list review-identified files already inside the approved ownership boundary.
- Independent code review rerun evidence: `venv\Scripts\python -m pytest tests\test_rag_helper_arg_boundaries.py -q` reported `20 passed in 1.40s`; `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\conversation_aggregate_agent.py` succeeded; focused tests reported `23 passed`, `30 passed`, `16 passed`, `19 passed`, `2 passed`, and `10 passed`; subsystem regressions reported `26 passed`, `83 passed`, `23 passed`, `14 passed`, `59 passed`, and `32 passed`; `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q` reported `1428 passed, 256 deselected in 21.04s`.
- Independent code review static/syntax evidence: all static grep gates matched only expected results; the direct conversion grep matched only `src/kazusa_ai_chatbot/time_boundary.py`; prompt timezone grep returned `NO_MATCH`; `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py src\kazusa_ai_chatbot\reflection_cycle\prompts.py src\kazusa_ai_chatbot\reflection_cycle\promotion.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py` succeeded; `venv\Scripts\python -m compileall -q src tests` succeeded; `venv\Scripts\python -c "import kazusa_ai_chatbot.service; print('service import ok')"` printed `service import ok`; `git diff --check` exited 0 with CRLF warnings only.
- Independent code review residual risk: live LLM and live DB tests were not run, matching the plan's verification scope. Approval status: approved with no unresolved blocker.
