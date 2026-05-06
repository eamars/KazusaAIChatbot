# interaction style image plan

## Summary

- Goal: Add private user and group-channel interaction style images derived from daily reflection, keep them out of RAG and cognition memory, and expose only sanitized abstract handling guidance to selected L3 agents.
- Plan class: large
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`, `test-style-and-execution`.
- Overall cutover strategy: additive and compatible; absent or empty user-style overlays preserve current behavior until background reflection writes usable style images.
- Highest-risk areas: privacy boundary between reflection and L3, accidental RAG exposure, prompt authority confusion, and group/private context mixing.
- Acceptance criteria: daily private reflections can update a `global_user_id` scoped user style image, daily group reflections can update a group-channel style image, RAG cannot retrieve either image, and L3 style/preference agents receive only compact positive interaction guidance.

## Context

The current system has two separate mechanisms that must remain separate:

- `user_image` / `user_memory_context` is cognition memory. It comes from `user_memory_units` and is consumed by L1/L2/L3 for facts, subjective appraisals, commitments, and relationship signals.
- Daily reflection currently feeds global reflection promotion and `promoted_reflection_context`, not user image. It can produce useful interaction-quality observations, but those observations are not user facts.

This plan adds a new interaction-style layer. It is not RAG, not memory evidence, and not a replacement for the user-memory consolidator. It is an abstract speech/social handling overlay learned from daily reflection.

Confirmed product decisions:

- One character owns one database. `global_user_id` is sufficient for user-style scope.
- User style image is globally applied for that user in both private and group chat.
- Group channel style image overlays user style only when speaking in that group channel.
- Style images are persistent current-state documents that are overwritten/evolved when newer reflection teaches a better handling rule.
- Daily reflection should influence only style images in this plan. The current consolidator remains the only owner for `user_memory_units`.
- The consumer payload must be sanitized before L3 sees it. L3 should receive positive abstract handling guidance, not detailed event history and not privacy warnings.

## Mandatory Skills

- `local-llm-architecture`: load before changing reflection, prompt, graph, RAG, cognition, L3, or background LLM behavior.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Do not add interaction style images to `rag_result`, RAG known facts, `memory_evidence`, `recall_evidence`, `conversation_evidence`, `user_memory_unit_candidates`, or `promoted_reflection_context`.
- Do not write daily-reflection-derived style observations into `user_memory_units`.
- Do not make L1, L2, boundary core, judgment core, dialog agent, or consolidator depend on interaction style images.
- Do not expose source reflection run IDs, timestamps, channel IDs, user private events, display names, or topic/event summaries in the L3-facing style overlay.
- L3-facing style overlay entries must be positive abstract handling guidance, such as preferred warmth, pacing, teasing level, clarification style, or engagement shape.
- The L3 consumer prompt must not be responsible for privacy cleanup. Privacy cleanup belongs to the extractor, deterministic sanitizer, storage validator, and tests.
- Do not add vector embeddings or lexical search indexes for interaction style images.
- Do not create a RAG capability, RAG helper agent, or retrieval route for interaction style images.
- Keep raw Mongo access inside `src/kazusa_ai_chatbot/db/` modules.
- Do not import or call `get_db` outside `src/kazusa_ai_chatbot/db/`. This includes `src/scripts/`; scripts must use semantic DB interfaces.
- Use semantic descriptors instead of raw metrics when any style-image prompt receives activity measurements.
- Keep normal chat-path latency bounded to deterministic DB reads only. No new LLM call is allowed on the live message path.
- Background reflection-style extraction may use LLM calls because it runs outside the user-response critical path.
- If a private daily reflection cannot be resolved to exactly one non-character `global_user_id`, skip user-style update for that daily run and log a structured reason.

## Must Do

- Add a new `interaction_style_images` collection and DB module.
- Add typed schemas for stored style images and prompt-facing style overlays.
- Add deterministic sanitizer/validator that rejects detailed event-style outputs before persistence.
- Add background reflection-to-style extraction for private daily reflections and group daily reflections.
- Add a deterministic L3 style-context loader after L2 judgment and before L3 agents.
- Pass interaction style context only to L3 style and preference adapter prompts.
- Add tests proving style images are not exposed through RAG projections or RAG result shapes.
- Add tests proving absent or empty user-style overlays preserve current preference-adapter behavior.
- Add tests proving private user style applies in group chat and group style overlays it.

## Deferred

- Do not redesign daily reflection hourly/daily prompt schemas in this plan.
- Do not redesign global reflection promotion.
- Do not migrate existing promoted reflection memory rows.
- Do not backfill style images from old daily reflection data unless an operator runs a separate approved script.
- Do not create group user memory, group cognition memory, or group relationship memory.
- Do not add embeddings, vector search, Atlas Search, keyword search, or RAG retrieval for style images.
- Do not alter dialog-agent prompt contracts directly.
- Do not change affinity, `last_relationship_insight`, or user-profile persistence.

## Cutover Policy

| Area | Policy | Instruction |
|---|---|---|
| Runtime chat | compatible | If no style image exists, L3 receives an empty `user_style` overlay and no private-chat group-style field; current behavior is preserved. |
| Reflection worker | additive | After daily channel reflection succeeds, run style-image update for eligible daily docs. |
| RAG | no exposure | RAG interfaces, projections, helper agents, and evidence lists remain unaware of style-image data. |
| User memory | no change | `user_memory_units` continues to be managed only by the existing consolidator. |
| Database | additive | Create a new collection and indexes; do not mutate existing collections. |
| Group chat | additive | Group channel style is loaded only for `channel_type == "group"` and only overlays the user style context. |

## Agent Autonomy Boundaries

- The implementation agent must not change the architecture decisions in this plan.
- The implementation agent may choose small helper names inside private functions, but public module names, collection name, schema fields, and call boundaries in this plan are fixed.
- If current code makes a listed integration impossible without changing RAG, L1/L2, dialog, or user-memory semantics, stop and report the blocker instead of expanding scope.
- If sanitizer tests show the extractor repeatedly generates event summaries, tighten the extractor prompt and validator. Do not push privacy cleanup into L3 prompts.

## Target State

Daily reflection produces sanitized style images:

```text
private daily reflection
  -> private user style extractor
  -> sanitizer / validator
  -> interaction_style_images(scope_type="user", global_user_id)

group daily reflection
  -> group channel style extractor
  -> sanitizer / validator
  -> interaction_style_images(scope_type="group_channel", platform, platform_channel_id)
```

Live chat loads style images outside RAG:

```text
message enters persona supervisor
  -> decontextualizer
  -> RAG
  -> L1/L2 cognition
  -> deterministic interaction style context loader
  -> L3 style agent and L3 preference adapter
  -> L4 collector
  -> dialog agent
```

Runtime L3 sees only this style context shape:

Private chat:

```json
{
  "user_style": {
    "speech_guidelines": [],
    "social_guidelines": [],
    "pacing_guidelines": [],
    "engagement_guidelines": [],
    "confidence": ""
  },
  "application_order": ["user_style"]
}
```

Group chat:

```json
{
  "user_style": {
    "speech_guidelines": [],
    "social_guidelines": [],
    "pacing_guidelines": [],
    "engagement_guidelines": [],
    "confidence": ""
  },
  "group_channel_style": {
    "speech_guidelines": [],
    "social_guidelines": [],
    "pacing_guidelines": [],
    "engagement_guidelines": [],
    "confidence": ""
  },
  "application_order": ["user_style", "group_channel_style"]
}
```

For private chat, the `group_channel_style` key must be absent. Do not include an empty group-style object in private-chat L3 payloads.

## Design Decisions

- Use one Mongo collection: `interaction_style_images`.
- Use one active document per scope. New extraction evolves/replaces the active style overlay instead of appending prompt-facing history.
- Use deterministic `style_image_id`:
  - `user:<global_user_id>`
  - `group_channel:<platform>:<platform_channel_id>`
- Store source reflection run IDs internally for audit, but never return them in runtime L3 projections.
- Runtime style context is loaded by a deterministic cognition node after L2 judgment. L1/L2 do not receive style image content.
- `call_style_agent` and `call_preference_adapter` consume the style context. `call_contextual_agent`, content anchor, visual agent, L4, and dialog agent do not need direct style-image fields in this plan.
- `src/scripts/identify_user_image.py` may show the user style image as a separate diagnostic section, but it must not merge style-image fields into the cognition-facing `user_memory_context` or present them as user memory.
- Daily style extraction uses daily reflection output fields that are already abstract enough for style learning: `daily_doc.output.conversation_quality_patterns`, `daily_doc.output.synthesis_limitations`, `daily_doc.output.confidence`, plus `daily_doc.scope.*` channel metadata. These signal fields live inside the `output` dict, not at the top level of `CharacterReflectionRunDoc`. It must not use raw transcripts.
- The extractor may read the existing active style image for the same scope so the output can evolve persistent guidance.
- If daily reflection confidence is absent or invalid, the style extractor must skip. If confidence is `low`, skip. `medium` and `high` are eligible.
- If the validated extractor output is fully empty (all four guideline fields empty and `confidence == ""`), skip the upsert entirely and log a structured `style_extraction_empty_output` reason. Do not overwrite a prior useful overlay with an empty document on this path. (A separate operator-only path may set `status="disabled"` if needed.)

## Data Model

Add to `src/kazusa_ai_chatbot/db/schemas.py`:

```python
class InteractionStyleScopeType:
    USER = "user"
    GROUP_CHANNEL = "group_channel"


class InteractionStyleOverlayDoc(TypedDict, total=False):
    speech_guidelines: list[str]
    social_guidelines: list[str]
    pacing_guidelines: list[str]
    engagement_guidelines: list[str]
    confidence: str


class InteractionStyleImageDoc(TypedDict, total=False):
    style_image_id: str
    scope_type: str
    global_user_id: str
    platform: str
    platform_channel_id: str
    status: str
    overlay: InteractionStyleOverlayDoc
    source_reflection_run_ids: list[str]
    revision: int
    created_at: str
    updated_at: str
```

Allowed `confidence` values: `""`, `"low"`, `"medium"`, `"high"`.

Allowed `status` values: `"active"`, `"empty"`, `"disabled"`.

Runtime empty overlay:

```python
{
    "speech_guidelines": [],
    "social_guidelines": [],
    "pacing_guidelines": [],
    "engagement_guidelines": [],
    "confidence": "",
}
```

## DB Module Interface

Add `src/kazusa_ai_chatbot/db/interaction_style_images.py`.

Public functions:

```python
def empty_interaction_style_overlay() -> InteractionStyleOverlayDoc:
    ...


def validate_interaction_style_overlay(overlay: dict) -> InteractionStyleOverlayDoc:
    ...


async def ensure_interaction_style_image_indexes() -> None:
    ...


async def get_user_style_image(global_user_id: str) -> InteractionStyleImageDoc | None:
    ...


async def get_group_channel_style_image(
    *,
    platform: str,
    platform_channel_id: str,
) -> InteractionStyleImageDoc | None:
    ...


async def upsert_user_style_image(
    *,
    global_user_id: str,
    overlay: dict,
    source_reflection_run_ids: list[str],
    timestamp: str | None = None,
) -> InteractionStyleImageDoc:
    ...


async def upsert_group_channel_style_image(
    *,
    platform: str,
    platform_channel_id: str,
    overlay: dict,
    source_reflection_run_ids: list[str],
    timestamp: str | None = None,
) -> InteractionStyleImageDoc:
    ...


async def build_interaction_style_context(
    *,
    global_user_id: str,
    channel_type: str,
    platform: str,
    platform_channel_id: str,
) -> dict:
    ...
```

`build_interaction_style_context` returns the prompt-facing runtime shape in `Target State`, not the stored DB document.

Update `src/kazusa_ai_chatbot/db/__init__.py` to export only semantic functions needed by callers. Do not export raw collection names unless existing DB style requires it for tests.

Update `src/kazusa_ai_chatbot/db/bootstrap.py` to call `ensure_interaction_style_image_indexes`.

Add a non-creating profile lookup interface in `src/kazusa_ai_chatbot/db/users.py` for diagnostics and scripts:

```python
async def find_user_profile_by_identifier(
    *,
    identifier: str,
    platform: str | None = None,
) -> UserProfileDoc | None:
    ...
```

Behavior:

- If `platform` is provided, find by exact platform-account pair.
- If `platform` is absent, first find by `global_user_id`, then by any `platform_user_id`.
- Never create a user profile.
- Omit Mongo `_id`.
- Use this interface from `src/scripts/identify_user_image.py`; do not let the script import or call `get_db`.

Indexes:

- Unique: `[("style_image_id", 1)]`, name `interaction_style_image_id_unique`.
- Read helper: `[("scope_type", 1), ("global_user_id", 1)]`, name `interaction_style_user_scope`.
- Read helper: `[("scope_type", 1), ("platform", 1), ("platform_channel_id", 1)]`, name `interaction_style_group_channel_scope`.

## Sanitizer Contract

`validate_interaction_style_overlay` must:

- Return a normalized overlay with only the five runtime fields.
- Accept only lists of strings for guideline fields.
- Trim whitespace and drop empty strings.
- Cap each guideline field to 5 items.
- Cap each guideline item to 120 characters.
- Drop duplicate guideline strings within and across fields.
- Accept only `low`, `medium`, `high`, or empty confidence.
- Reject or drop fields outside the overlay schema.
- Raise `ValueError` if every guideline field is empty and confidence is not empty.
- Reject items that contain obvious event/detail markers:
  - ISO date fragments or clock times.
  - platform/channel/message IDs.
  - `run_id`, `reflection_run`, `source_reflection`.
  - raw quote-heavy examples using Chinese or ASCII quotation marks.
  - explicit user display names or platform user IDs when provided to the validator.

The sanitizer is not expected to solve all semantic privacy detection. The extractor prompt must produce abstract guidance. Tests must cover representative private-event leakage attempts.

## Reflection Style Extraction

Add `src/kazusa_ai_chatbot/reflection_cycle/interaction_style.py`.

Public functions:

```python
async def run_daily_interaction_style_update(
    *,
    character_local_date: str,
    dry_run: bool,
    is_primary_interaction_busy: Callable[[], bool],
) -> ReflectionWorkerResult:
    ...


async def extract_user_style_overlay_from_daily_reflection(
    *,
    daily_doc: CharacterReflectionRunDoc,
    current_overlay: dict,
) -> dict:
    ...


async def extract_group_channel_style_overlay_from_daily_reflection(
    *,
    daily_doc: CharacterReflectionRunDoc,
    current_overlay: dict,
) -> dict:
    ...
```

Private daily update rules:

- Process only `run_kind == "daily_channel"`, `status == "succeeded"`, `scope.channel_type == "private"`.
- Resolve the target `global_user_id` from `conversation_history`, not from reflection prompt output.
- Resolution must find exactly one non-character user in the daily doc time window and channel.
- If resolution finds zero or more than one non-character user, skip and log.
- Load existing user style image, pass only its overlay plus sanitized daily signal payload to the extractor, validate the returned overlay, then upsert.

Group daily update rules:

- Process only `run_kind == "daily_channel"`, `status == "succeeded"`, `scope.channel_type == "group"`.
- Use `scope.platform` and `scope.platform_channel_id` as the group-channel key.
- Load existing group channel style image, pass only its overlay plus sanitized daily signal payload to the extractor, validate the returned overlay, then upsert.

Daily signal payload:

```json
{
  "channel_type": "private|group",
  "daily_confidence": "medium|high",
  "conversation_quality_patterns": [],
  "synthesis_limitations": [],
  "current_overlay": {}
}
```

Do not include raw transcript rows, `day_summary`, active hour summaries, source message refs, nicknames, display names, source run IDs, or platform IDs in the LLM human payload.

Extractor prompt output:

```json
{
  "overlay": {
    "speech_guidelines": [],
    "social_guidelines": [],
    "pacing_guidelines": [],
    "engagement_guidelines": [],
    "confidence": "medium"
  }
}
```

The extractor prompt may contain upstream privacy and abstraction rules because it is the sanitization boundary. The runtime L3 prompts must receive only the validated overlay and should not repeat privacy-cleanup instructions.

## Private User Resolver

Add a DB helper in `src/kazusa_ai_chatbot/db/conversation_reflection.py` or a new DB module if cleaner:

```python
async def resolve_single_private_scope_user_id(
    *,
    platform: str,
    platform_channel_id: str,
    start_timestamp: str,
    end_timestamp: str,
    character_global_user_id: str,
) -> str:
    ...
```

Behavior:

- Query `conversation_history` for user-authored rows in the private channel and time window.
- Exclude `character_global_user_id`.
- Return the only distinct non-empty `global_user_id`.
- Return `""` if none or multiple distinct user IDs are present.
- Do not expose display names or message text.

## Worker Integration

Update `src/kazusa_ai_chatbot/reflection_cycle/worker.py`:

- Import `run_daily_interaction_style_update`.
- In `_run_worker_tick`, after `_run_daily_channel_reflection_cycle` and before `_run_global_reflection_promotion`, call `run_daily_interaction_style_update`.
- Respect `is_primary_interaction_busy` before starting the style update pass.
- Append the returned `ReflectionWorkerResult` to tick results.
- In dry-run public helpers, expose a direct callable for style updates if current reflection worker pattern supports direct stage testing.

Use run kind:

```text
daily_interaction_style_update
```

## Runtime L3 Integration

Update `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`:

- Add `platform`, `platform_channel_id`, and `channel_type` to `CognitionState`. These fields currently live only in `GlobalPersonaState` and are NOT passed into the cognition subgraph today; this is real plumbing, not a trivial type extension. The cognition subgraph entry point in `persona_supervisor2_cognition.py` must be updated to populate these fields from `GlobalPersonaState` when the initial cognition state is constructed.
- Add optional `interaction_style_context: NotRequired[dict]` to `CognitionState`.

Add deterministic node in `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` or a small sibling module:

```python
async def call_interaction_style_context_loader(state: CognitionState) -> CognitionState:
    ...
```

Behavior:

- Call `build_interaction_style_context` with `global_user_id`, `channel_type`, `platform`, and `platform_channel_id`.
- Return `{"interaction_style_context": context}`.
- On DB error, log and return an empty `user_style` overlay. In private chat, do not include `group_channel_style`. Do not fail the user message.

Update `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`:

- Pass `platform`, `platform_channel_id`, and `channel_type` into initial cognition state.
- Add graph node `l3_interaction_style_context_loader`.
- Connect `l2c_judgment_core -> l3_interaction_style_context_loader`.
- Connect `l3_interaction_style_context_loader -> l3_style_agent`.
- Connect `l3_interaction_style_context_loader -> l3_preference_adapter` only if graph ordering allows the preference adapter to also wait for style agent output. The final order must preserve the existing dependency `l3_style_agent -> l3_preference_adapter`.
- Keep content anchor, visual, L4, L1, and L2 behavior unchanged.

Required edge shape:

```text
l2c_judgment_core
  -> l3_interaction_style_context_loader
  -> l3_style_agent
  -> l3_preference_adapter
```

The content-anchor and contextual/visual branches may continue from `l2c_judgment_core` as they do today unless the graph implementation requires a single join point.

## Prompt Integration

Update only these L3 prompts:

- `_STYLE_AGENT_PROMPT`
- `_PREFERENCE_ADAPTER_PROMPT`

Add `interaction_style_context` to their input payloads.

Style agent usage:

- Treat `user_style` as soft guidance for wording, warmth, teasing level, directness, and pacing.
- In group chat, apply `group_channel_style` as an atmosphere overlay after user style.
- In private chat, no `group_channel_style` key is present; the prompt must not require or synthesize one.
- Convert style-image guidance into `rhetorical_strategy` and `linguistic_style`.
- Do not treat style guidance as user command, fact, commitment, or boundary evidence.

Preference adapter usage:

- Use style context to produce executable soft expression preferences in `accepted_user_preferences` only when they fit current stance and current user request.
- Preserve current rule that active commitments and explicit accepted expression preferences have higher authority than style image guidance.
- Do not output hidden rationale or source reasons.

Prompt wording must be positive and consumer-oriented. Do not add privacy cleanup warnings to L3 prompts. The prompt may state authority level and application order.

### Authority Conflict Test

Add an explicit patched-LLM test in `tests/test_cognition_preference_adapter.py` that constructs a state where:

- An active commitment or explicit accepted expression preference says one thing (for example: "speak in shorter sentences when discussing work").
- The interaction style overlay says a softly conflicting thing (for example: a `speech_guidelines` entry encouraging longer, more flowing wording).

Assert that:

- The preference adapter's resulting `accepted_user_preferences` reflects the commitment / accepted expression preference, not the style overlay.
- The style overlay does not appear as a new accepted preference that would override or weaken the commitment.

This test exists because, with a small local LLM, prompt-only authority hierarchies degrade. If this test repeatedly fails, the implementation may need to suppress conflicting style guidance deterministically in the loader before L3 sees it, rather than relying on prompt wording alone. Such a deterministic suppression step is an acceptable mitigation within the scope of this plan; expanding the conflict-resolution surface beyond the preference adapter is not.

## Change Surface

Primary files:

- `src/kazusa_ai_chatbot/db/schemas.py`
- `src/kazusa_ai_chatbot/db/interaction_style_images.py`
- `src/kazusa_ai_chatbot/db/bootstrap.py`
- `src/kazusa_ai_chatbot/db/__init__.py`
- `src/kazusa_ai_chatbot/db/conversation_reflection.py`
- `src/kazusa_ai_chatbot/db/users.py`
- `src/kazusa_ai_chatbot/reflection_cycle/interaction_style.py`
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `src/scripts/identify_user_image.py`

Primary tests:

- `tests/test_interaction_style_images.py`
- `tests/test_reflection_interaction_style.py` (must include a fully-empty extractor output skip case)
- `tests/test_cognition_interaction_style_context.py` (must include a focused test asserting `platform`, `platform_channel_id`, and `channel_type` are present in `CognitionState` for both private and group calls)
- Extend `tests/test_cognition_preference_adapter.py` with the `Authority Conflict Test` defined under `Prompt Integration`.
- Extend relevant RAG projection tests to assert absence from RAG shapes.

Do not modify unrelated dialog, L1, L2, RAG helper-agent, memory evolution, or user-memory-unit files except for imports/types required by the explicit files above.

## Implementation Order

1. Add DB schemas and `interaction_style_images` DB module.
2. Add DB unit tests for empty overlay, validation, upsert/read, runtime projection, and index creation.
3. Add private user resolver helper and tests with zero, one, and multiple user IDs.
4. Add `find_user_profile_by_identifier` in `db.users` and refactor `src/scripts/identify_user_image.py` to use DB interfaces only.
5. Extend `src/scripts/identify_user_image.py` to show `user_style_image` as a separate diagnostic section in text and JSON output.
6. Add reflection-cycle style extraction module with prompt, parser, sanitizer integration, dry-run/no-write behavior, and tests using patched LLM responses.
7. Integrate style update pass into reflection worker after daily channel reflection and before global promotion.
8. Plumb `platform`, `platform_channel_id`, and `channel_type` from `GlobalPersonaState` into `CognitionState` and the cognition subgraph entry point. Add a focused test asserting these fields reach `CognitionState` in both private and group calls. This is a real wiring change because today these fields live only on `GlobalPersonaState`.
9. Add `interaction_style_context` field to `CognitionState` and the deterministic L3 style context loader.
10. Wire the loader into the cognition graph before L3 style/preference agents.
11. Update L3 style and preference prompts to consume `interaction_style_context`.
12. Add prompt-render and patched-LLM tests for style/preference agents, including a style-vs-commitment authority test (see `Authority Conflict Test` below).
13. Add RAG non-exposure tests.
14. Run focused tests, then the broader affected test set.

## Progress Checklist

- [x] DB schema and collection indexes added.
- [x] Runtime style overlay sanitizer implemented.
- [x] DB read/write/projection tests passing.
- [x] Private user resolver implemented and tested.
- [x] `identify_user_image` uses DB interfaces only and shows user style as a separate diagnostic section.
- [x] Daily reflection style extraction implemented and tested (including fully-empty-output skip behavior).
- [x] Reflection worker integration implemented and tested.
- [x] `platform`, `platform_channel_id`, `channel_type` plumbed from `GlobalPersonaState` into `CognitionState` with a focused test asserting they reach the cognition subgraph.
- [x] Cognition state `interaction_style_context` field and L3 loader implemented.
- [x] L3 graph wiring implemented.
- [x] Style agent prompt updated and prompt-render tested.
- [x] Preference adapter prompt updated and prompt-render tested, including style-vs-commitment authority conflict test.
- [x] RAG non-exposure tests added and passing.
- [x] Pre-existing raw-DB-boundary violations snapshotted at start; no new violations added.
- [x] Focused and affected test suites passing.

## Verification

Run focused static checks:

```powershell
rg -n "interaction_style|style_image" src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag* tests
```

Expected result: no production RAG module imports or retrieves interaction style images. Tests may mention them only to assert absence.

Run raw DB boundary checks:

```powershell
rg -n "get_db|db\\.[a-zA-Z_]+\\.(find|find_one|aggregate|insert|insert_one|insert_many|update|update_one|update_many|replace_one|delete|delete_one|delete_many)" src\scripts src\kazusa_ai_chatbot --glob "!src/kazusa_ai_chatbot/db/**"
```

Expected result: this plan must not add any new raw DB handler imports or raw Mongo collection operations outside `src/kazusa_ai_chatbot/db/`. At the start of work, snapshot the current violation list from this command. The post-implementation diff must be a non-strict subset of that snapshot: every violation present after the change must already have been present before. Pre-existing unrelated violations are explicitly out of scope for this plan and must not trigger a generic DB-cleanup pass. The only exception is `src/scripts/identify_user_image.py`, which this plan explicitly migrates to DB interfaces.

Run Python compilation:

```powershell
venv\Scripts\python.exe -m compileall src\kazusa_ai_chatbot tests
```

Run focused tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_interaction_style_images.py tests\test_reflection_interaction_style.py tests\test_cognition_interaction_style_context.py tests\test_cognition_preference_adapter.py
```

Run affected existing tests:

```powershell
venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_cycle_readonly.py tests\test_persona_supervisor2_schema.py tests\test_rag_projection.py tests\test_persona_supervisor2_rag_skip_shape.py
```

Run live-LLM tests only if the implementation changes prompt contracts in a way that cannot be covered with patched prompt-render tests and the project test instructions allow live LLM execution for the current environment.

## Execution Evidence

- Implementation completed on 2026-05-06.
- Self-review corrections on 2026-05-06 tightened private-channel resolution,
  skipped already-applied successful daily style updates, made the diagnostic
  `user_style_image` JSON section explicit when empty, and refined quote-marker
  validation.
- Focused deterministic and patched-LLM tests:
  `venv\Scripts\python.exe -m pytest tests\test_interaction_style_images.py tests\test_reflection_interaction_style.py tests\test_cognition_interaction_style_context.py tests\test_cognition_preference_adapter.py -q`
  passed with `22 passed`.
- Affected existing tests:
  `venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_prompt_contracts.py tests\test_reflection_cycle_readonly.py tests\test_persona_supervisor2_schema.py tests\test_rag_projection.py tests\test_persona_supervisor2_rag_skip_shape.py -q`
  passed with `40 passed`.
- Python compilation:
  `venv\Scripts\python.exe -m compileall src\kazusa_ai_chatbot tests`
  completed successfully.
- RAG static exposure check found no production RAG module or RAG node references to `interaction_style` or `style_image`; tests contain only non-exposure assertions.
- Raw DB boundary check showed no new violations in files introduced or modified by this plan outside `src/kazusa_ai_chatbot/db/`. Existing out-of-scope violations remain pre-existing; `src/scripts/identify_user_image.py` no longer imports or calls `get_db`.
- Live-LLM tests were not run because this implementation used deterministic and patched-LLM prompt payload/render checks for the changed L3 contracts, as allowed by this plan.

## Acceptance Criteria

- `interaction_style_images` collection exists with the required indexes.
- A private `daily_channel` reflection with medium/high confidence can update exactly one `global_user_id` scoped user style image.
- A group `daily_channel` reflection with medium/high confidence can update exactly one group-channel style image.
- Style extraction skips low-confidence, unresolved-private-user, and multi-user private scopes.
- Style extraction skips upsert (and does not overwrite a prior overlay) when the validated extractor output is fully empty.
- `platform`, `platform_channel_id`, and `channel_type` reach `CognitionState` from `GlobalPersonaState` for both private and group cognition calls.
- Preference adapter preserves active commitments and explicit accepted expression preferences over conflicting style overlay guidance (verified by patched-LLM authority conflict test).
- This change introduces no new raw-DB-boundary violations relative to the pre-implementation snapshot.
- Stored style images contain only abstract positive handling guidance plus internal audit metadata.
- Runtime L3 projection contains no source run IDs, no timestamps, no platform IDs, no event summaries, and no private details.
- Private chat receives user style only; `group_channel_style` is absent from the L3 input payload.
- Group chat receives user style first and group channel style second.
- `src/scripts/identify_user_image.py` can display `user_style_image` separately for human diagnostics without merging it into cognition-facing user image data.
- No code introduced by this plan, in any path including `src/scripts/`, imports or calls `get_db` or performs raw Mongo collection operations outside `src/kazusa_ai_chatbot/db/`. Pre-existing violations are governed by the snapshot rule above and are out of scope.
- L3 style/preference agents can use the overlay without adding privacy-cleanup instructions.
- RAG cannot retrieve or project style-image records.
- Existing user-memory-unit consolidator behavior remains unchanged.

## Risks

- Sanitizer false negatives can allow event-like text into style images. Mitigate with strict prompt payload selection, max-length rules, leakage tests, and log warnings for dropped items.
- Sanitizer false positives can drop useful guidance. Mitigate by keeping guideline wording concise and abstract.
- Group overlay can overpower user-specific style. Mitigate through explicit application order and prompt authority wording.
- Additional background LLM calls can increase reflection worker cost. Mitigate by running only after successful daily docs and skipping low-confidence docs.
- Graph wiring can accidentally make preference adapter run before style context exists. Mitigate with a dedicated graph-wiring test.
- Concurrency between background extractor writes and live-chat loader reads on the same `style_image_id`. Mitigation: each style image is a single Mongo document and updates use full-document upsert, so reads see either the prior revision or the new revision atomically; partial-overlay reads are not possible. Document this assumption in the DB module so future schema changes preserve it.
- Authority hierarchy between style overlay and active commitments / accepted expression preferences is enforced via prompt wording on a small local LLM. Mitigate with the `Authority Conflict Test`; if it fails repeatedly, fall back to deterministic loader-side suppression as described in `Prompt Integration`.

## LLM Call And Context Budget

- Live chat path: zero new LLM calls.
- Live chat path: at most two deterministic DB reads for style context, one user-scope read and one group-channel read when in group chat.
- Background reflection path: at most one style-extraction LLM call per eligible daily channel doc.
- Extractor prompt input must stay compact and exclude raw transcript rows.
- L3 prompt payload receives only the current compact style context, not stored history.

## Operational Steps

- Deploy code with empty user-style overlays and absent private-chat group-style payloads supported first.
- Run DB bootstrap to create indexes.
- Let the reflection worker populate style images from future daily reflection runs.
- Do not backfill from historical daily reflection data as part of this plan.
- Inspect logs for skipped private resolution, sanitizer rejections, and style-image update counts after the first daily cycle.

## Glossary

- `user_image`: Cognition memory derived from `user_memory_units`; not changed by this plan.
- `user_style_image`: Abstract handling guidance for one `global_user_id`; applies in private and group chat.
- `group_channel_style_image`: Abstract atmosphere and interaction guidance for one group channel.
- `interaction_style_context`: Prompt-facing L3 payload containing sanitized overlays.
- `style overlay`: Positive abstract guidance for speech, social handling, pacing, and engagement.
