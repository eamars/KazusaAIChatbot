# multi source cognition architecture stage 09 multimodal cognitive input sources plan

## Summary

- Goal: Represent existing image and audio descriptions as typed
  `CognitiveEpisode` percepts so multimodal user-message turns can enter
  cognition without raw binary, prompt payload leakage, or text-only `/chat`
  regression.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` before editing cognition Python files that contain CJK prompt
  text.
- Overall cutover strategy: compatible for text-only `/chat`; compatible for
  media-description `/chat`; bigbang exclusion for raw media inside cognition,
  RAG, and prompt-source payloads.
- Highest-risk areas: stale pre-descriptor episodes, leaking `base64_data` into
  cognition prompts, changing text-only prompt/RAG behavior, treating media
  descriptions as accepted user instructions, and broadening RAG beyond the
  current dialog-text query.
- Acceptance criteria: text-only episodes remain byte-for-byte compatible;
  post-descriptor image/audio descriptions become bounded `image_observation`
  and `audio_observation` percepts; prompt/RAG tests prove raw media is absent;
  Stage 00, Stage 03, Stage 06, Stage 07, and Stage 08 gates still pass.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Lifecycle: this plan completed execution from post-Stage-08 `main`.
Stage 09 must execute on a feature branch forked from a mainline that includes
Stage 08 evidence commit `6b62476`.

## Context

Current service state already carries `user_multimedia_input` rows with
`content_type`, `base64_data`, and `description`. Earlier stages deliberately
kept the active cognitive episode text-only. Stage 09 makes the episode
contract express existing media descriptions as separate typed percepts while
preserving the current text query and response path for text-only turns.

The normal image path currently creates the initial `cognitive_episode` before
the existing `multimedia_descriptor_agent` fills image descriptions. Therefore
Stage 09 has one required handoff point: after the existing descriptor updates
`user_multimedia_input`, it must refresh the existing text-chat episode with
sanitized media-description percepts before relevance, persona, RAG, cognition,
or dialog consumers read it.

This stage must not add media understanding. It consumes descriptions that
already exist either on the incoming attachment envelope or after the existing
image descriptor node has run. It must not add a new image, audio, or video LLM
summarizer.

## Stage Handoff

### From Stage 08

Stage 08 completed and recorded these artifacts:

- branch: `stage-08-internal-thought-cognition-dry-run`;
- implementation commit: `ef1449b`;
- evidence/mainline commit: `6b62476`;
- focused Stage 08 gate: `26 passed`;
- Stage 07 reflection dry-run gate: `14 passed`;
- Stage 03 prompt-selection gate: `36 passed`;
- Stage 06 origin-policy gates: `9 passed`;
- Stage 00 regression baseline: `11 passed`;
- independent code review fixed the internal-thought `action_latch` value-type
  validation gap and reran affected checks.

Stage 09 starts from these completed artifacts:

- prompt selector supports multiple source variants without `/chat` regression;
- reflection and internal-thought dry-run sources stay out of public writes;
- Stage 08 leaves public scene residue contract-only;
- parent ledger row for `stage_08` is `completed`.

Stage 09 must not use internal-thought residue as image/audio percept input.

### To Stage 10

After Stage 09, Stage 10 can rely on:

- text, reflection, internal-thought, image, and audio input-source labels being
  represented in the shared episode vocabulary;
- image/audio user-message media support being limited to bounded description
  percepts;
- raw binary excluded from cognition episode content, cognition source payloads,
  and RAG requests;
- text-only `/chat` regression evidence after multimodal episode expansion;
- Stage 06 origin-policy, Stage 07 reflection, and Stage 08 internal-thought
  gates still passing;
- no proactive output behavior enabled by media support.

Stage 10 must treat media percepts as evidence, not permission to contact a
user, send media, schedule follow-up output, or persist user commitments.

## Mandatory Skills

- `development-plan-writing`: load before implementation or lifecycle edits.
- `local-llm-architecture`: load before changing prompt admission, cognition
  payloads, graph handoff, RAG projection, or LLM call budget.
- `no-prepost-user-input`: load before changing code that might interpret media
  descriptions as user instructions, commitments, preferences, or permissions.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing L1/L2/L3 cognition modules containing CJK
  prompt constants.

## Mandatory Rules

- Execute only from a feature branch forked from post-Stage-08 `main` at or
  after commit `6b62476`.
- Keep edits inside the approved Change Surface.
- If verification proves another direct fixture or production file must change,
  stop and update this plan before continuing.
- Use PowerShell `-LiteralPath '...'` for filesystem paths that may contain
  spaces; prefer repo-relative paths in commands.
- Do not add new media summarizer LLM calls, media fetchers, OCR, STT, video
  processing, fallback media parsers, feature flags, or graph entrypoints.
- Keep the existing image descriptor LLM path exactly as the only raw-image LLM
  prompt path. Stage 09 may pass its existing output descriptions forward, but
  must not add raw media to any cognition, RAG, or dialog prompt path.
- Do not place `base64_data`, raw bytes, attachment URLs, data URIs, or
  unbounded attachment metadata in `CognitiveEpisode` percept content,
  cognition source payloads, RAG requests, dialog payloads, or new logs.
- Do not change text-only `/chat` episode shape, prompt selection, prompt bytes,
  RAG request, dialog, consolidation, scheduler, adapter delivery, or
  persistence behavior.
- Do not treat `image_observation` or `audio_observation` as an accepted user
  command, preference, promise, permission, or durable instruction in
  deterministic code.
- Do not add keyword classifiers, semantic allowlists, semantic denylists, or
  post-LLM filters over media-description text.
- RAG query text remains the existing dialog-text compatibility projection.
  Media descriptions may be visible to cognition only through the source
  payload shape defined in this plan.
- Pure-media turns without a `dialog_text` percept remain unsupported.
- Text-only calls to `build_text_chat_cognitive_episode(...)` with no media rows
  must produce exactly the same `CognitiveEpisode` dict as Stage 08.
- The existing multimedia descriptor node must refresh and return
  `cognitive_episode` after it has generated descriptions. It must use the
  approved episode helper and must not mutate the existing episode in place.
- L1/L2/L3 prompt constants must remain byte-for-byte unchanged. Stage 09 may
  only add prompt-map entries for the approved variants.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the plan's `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Add the exact media-description contracts listed in `Contracts And Data
  Shapes`.
- Extend the text-chat episode builder with optional media-description rows
  while keeping text-only output unchanged.
- Add a helper that refreshes media percepts on an existing text-chat episode
  after the existing multimedia descriptor produces descriptions.
- Update service media-row collection so incoming image descriptions and audio
  descriptions can reach the builder without requiring raw media in the episode.
- Update the existing multimedia descriptor node to return the refreshed
  `cognitive_episode` after it updates `user_multimedia_input` and
  `prompt_message_context`.
- Add multimodal prompt selection for exactly these user-message input-source
  profiles:
  `["dialog_text", "image_observation"]`,
  `["dialog_text", "audio_observation"]`, and
  `["dialog_text", "image_observation", "audio_observation"]`.
- Add multimodal prompt-map entries for L1/L2/L3 handlers without changing
  text-chat, reflection, or internal-thought prompt bytes.
- Update the RAG episode adapter to accept the approved multimodal user-message
  input-source profiles while still projecting only dialog text into the RAG
  query and context.
- Add focused tests proving raw `base64_data`, URLs, and data URIs never appear
  in episode percept content, cognition source payloads, or RAG requests.
- Run every Verification command and record evidence before sign-off.

## Deferred

- Pure image/audio turns without text.
- New image/audio/video summarizer LLM calls.
- Raw file storage, media download, OCR, speech recognition, or vision model
  integration beyond the existing image descriptor path.
- Audio descriptor generation.
- Video percepts.
- Multimodal RAG query expansion.
- Reflection/internal-thought media mixing.
- Proactive sends, transport, outbox, or scheduled media output.
- Prompt wording optimization for media-specific cognition.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Text-only `/chat` | compatible | Existing episode, prompt selection, prompt bytes, RAG request, dialog, and consolidation remain unchanged. |
| Media-description `/chat` | compatible | Preserve existing text behavior and add typed media-description percepts only when descriptions are present. |
| Descriptor handoff | compatible | Reuse the existing descriptor node and refresh only `cognitive_episode`, `user_multimedia_input`, and `prompt_message_context`. |
| Raw media in cognition/RAG/dialog | bigbang | Raw binary, data URIs, URLs, and unbounded metadata are forbidden from new cognition, RAG, and dialog payloads. |
| RAG | compatible | Accept multimodal episodes only to project existing dialog text. No media retrieval query changes. |
| Prompt text | compatible | Add prompt-map entries only. Prompt constants remain byte-for-byte unchanged. |

Rollback path: remove optional media builder arguments, remove the media refresh
helper, remove multimodal selector/source-payload logic, remove multimodal
prompt-map entries, restore the RAG adapter accepted input-source list, remove
service/descriptor episode-media wiring, and remove focused tests. No database
rollback is required.

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative or broader strategy by default.
- If an area is `bigbang`, delete or reject forbidden payload flow directly
  instead of preserving fallback behavior.
- If an area is `compatible`, preserve only the compatibility surfaces
  explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

Allowed implementation choices:

- local fixture helper names inside
  `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`;
- assertion ordering inside focused tests;
- local private helper names inside `cognition_episode.py` only when the helper
  performs repeated structural validation, media row sanitization, media
  percept construction, or description trimming defined by this plan.

Not allowed:

- adding summarizer LLMs, media fetchers, OCR, STT, video processing, fallback
  media parsers, feature flags, alternate graph entrypoints, or compatibility
  shims;
- adding prompt text or editing existing prompt constants;
- adding new input-source labels, visibility labels, output modes, persistence
  schemas, or state keys;
- changing dispatcher, scheduler, adapter delivery, reflection worker,
  internal-thought runner, dialog, or consolidator policy;
- adding raising-only helpers or pass-through wrappers;
- creating helper APIs whose behavior is not explicitly named in this plan;
- using `.get(..., default)` for required internal fields after validation.

## Target State

For text-only chat turns, the episode remains exactly:

```text
trigger_source="user_message"
input_sources=["dialog_text"]
percepts=[
  dialog_text: existing user_input compatibility text
]
```

For media-description chat turns, the episode has one dialog percept plus zero
or more bounded media-description percepts:

```text
trigger_source="user_message"
input_sources=[
  "dialog_text",
  "image_observation",  # present only when at least one image description exists
  "audio_observation",  # present only when at least one audio description exists
]
percepts=[
  dialog_text: existing user_input compatibility text,
  image_observation: bounded image description only,
  audio_observation: bounded audio transcript or tone summary only
]
```

The RAG adapter still extracts dialog text through
`project_text_chat_compatibility_fields(...)`. Media descriptions are cognition
context, not retrieval query parameters.

The existing multimedia descriptor remains the only component allowed to send
raw image bytes to a vision-capable LLM. Its output is converted into bounded
text before the episode refresh helper is called.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Media source | Consume `MultiMediaDoc.description` after service intake and after the existing descriptor node updates it. | Avoid stale pre-descriptor episodes without adding graph branches. |
| Text-only behavior | Preserve exact builder output when no media rows exist. | Text chat remains the regression baseline. |
| Media cap | Keep at most four media-description percepts per episode and trim each description to 800 characters. | Matches the existing prompt attachment scale and bounds local-LLM context growth. |
| Input-source ordering | Always `dialog_text`, then `image_observation`, then `audio_observation`. | Keeps prompt selection deterministic. |
| RAG query | Use dialog-text projection only. | Prevent accidental media-driven retrieval drift. |
| Prompt text | Reuse existing prompt constants for multimodal variants. | Stage 09 is source admission, not prompt wording optimization. |
| Pure-media turns | Unsupported in this stage. | Current graph still expects dialog-text compatibility. |
| Raw media | Exclude completely from new cognition/RAG/dialog payloads. | Local LLMs need semantic descriptions, not raw bytes. |

## Contracts And Data Shapes

### Episode Media Rows

Add this public `TypedDict` in `src/kazusa_ai_chatbot/cognition_episode.py`:

```python
class MediaDescriptionRow(TypedDict):
    content_type: str
    description: str
```

`MediaDescriptionRow` must not contain `base64_data`, URL fields, raw bytes,
attachment ids, platform message ids, or storage metadata.

Add these public constants in `cognition_episode.py`:

```python
MAX_COGNITIVE_EPISODE_MEDIA_PERCEPTS = 4
MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS = 800
```

Add this public helper in `cognition_episode.py`:

```python
def build_text_chat_media_description_rows(
    multimedia_input: list[Mapping[str, object]],
) -> list[MediaDescriptionRow]:
```

The helper must:

- read only `content_type` and `description`;
- skip rows whose `content_type` is not a non-empty string;
- skip rows whose `description` is not a non-empty string after `strip()`;
- keep only rows whose `content_type` starts with `image/` or `audio/`;
- return rows containing exactly `content_type` and stripped `description`;
- never copy `base64_data`, URLs, data URIs, or attachment metadata.

Extend the existing builder signature by appending this keyword-only argument:

```python
media_description_rows: list[MediaDescriptionRow] | None = None,
```

The builder must:

- treat `None` and `[]` as text-only and produce the Stage 08 dict exactly;
- accept only `image/` and `audio/` content types;
- skip empty descriptions and unsupported content types;
- keep the first four accepted media rows after filtering;
- trim descriptions longer than 800 characters to the first 797 characters
  after `rstrip()` plus the suffix `...`;
- append media percepts after the existing dialog percept;
- assign media percept ids exactly as
  `f"{percept_id}:media:{media_index}"`, where `media_index` is one-based after
  filtering and capping;
- assign `input_source` as `image_observation` for `image/` rows and
  `audio_observation` for `audio/` rows;
- assign `visibility` exactly `model_visible`;
- assign `metadata` exactly
  `{"content_type": content_type, "media_index": media_index}`;
- set `input_sources` to `["dialog_text"]` plus `image_observation` and
  `audio_observation` once each when those sources are present, in that order;
- call `validate_cognitive_episode(episode)` before returning.

Add this public helper in `cognition_episode.py`:

```python
def replace_text_chat_media_percepts(
    *,
    episode: CognitiveEpisode,
    media_description_rows: list[MediaDescriptionRow] | None,
) -> CognitiveEpisode:
```

The helper must:

- validate the input episode;
- require `trigger_source == "user_message"`;
- require the episode has exactly one `dialog_text` percept;
- preserve `episode_id`, `trigger_source`, `output_mode`, `target_scope`,
  `origin_metadata`, `timestamp`, `time_context`, and the dialog percept
  content;
- preserve the original dialog percept id;
- remove existing `image_observation` and `audio_observation` percepts;
- rebuild media percepts using the exact builder rules above;
- return a new validated dict without mutating the input episode.

### Service And Descriptor Handoff

`src/kazusa_ai_chatbot/service.py` must update the current multimedia row
collection only as follows:

- include image rows when `media_type` starts with `image/` and either
  `base64_data` or `description` is present;
- include audio rows when `media_type` starts with `audio/` and `description`
  is present;
- preserve the existing `MultiMediaDoc` shape for `user_multimedia_input`;
- pass `media_description_rows=build_text_chat_media_description_rows(multimedia_input)`
  into the initial `build_text_chat_cognitive_episode(...)` call.

`src/kazusa_ai_chatbot/nodes/relevance_agent.py` must update only
`multimedia_descriptor_agent(...)`:

- keep the existing image descriptor LLM behavior unchanged;
- do not add audio descriptor generation;
- after `output_multimedia_input` and `prompt_message_context` are built, call
  `replace_text_chat_media_percepts(...)` with `state["cognitive_episode"]` and
  sanitized rows from `build_text_chat_media_description_rows(output_multimedia_input)`;
- return the refreshed `cognitive_episode` in the node return dict together
  with `user_multimedia_input` and `prompt_message_context`;
- do not mutate `state["cognitive_episode"]` in place.

### Prompt Selection And Payload

Add exactly these `CognitionPromptVariant` values:

- `text_chat_user_message_image_observation`
- `text_chat_user_message_audio_observation`
- `text_chat_user_message_image_audio_observation`

The selector must map input sources exactly:

| `trigger_source` | `input_sources` | `variant` | `prompt_key` |
|---|---|---|---|
| `user_message` | `["dialog_text"]` | `text_chat_user_message` | `{stage}.text_chat_user_message` |
| `user_message` | `["dialog_text", "image_observation"]` | `text_chat_user_message_image_observation` | `{stage}.text_chat_user_message_image_observation` |
| `user_message` | `["dialog_text", "audio_observation"]` | `text_chat_user_message_audio_observation` | `{stage}.text_chat_user_message_audio_observation` |
| `user_message` | `["dialog_text", "image_observation", "audio_observation"]` | `text_chat_user_message_image_audio_observation` | `{stage}.text_chat_user_message_image_audio_observation` |

The selector must keep the existing text-chat output modes:
`visible_reply`, `think_only`, and `silent`.

`build_cognition_prompt_source_payload(...)` must return `{}` for
`text_chat_user_message`. For the three multimodal variants it must return
exactly:

```python
{
    "media_observations": {
        "image_observations": list[str],
        "audio_observations": list[str],
    },
}
```

The lists must contain percept `content` strings only. They must not include
`content_type`, `metadata`, ids, URLs, raw bytes, `base64_data`, data URIs, or
attachment storage fields. The list order must match episode percept order
within each media source.

L1/L2/L3 prompt maps must map all three new variants to the existing prompt
constant already used by `text_chat_user_message` in that handler.

### RAG Adapter

`build_text_chat_rag_request(...)` must accept exactly these `user_message`
input-source profiles:

- `["dialog_text"]`
- `["dialog_text", "image_observation"]`
- `["dialog_text", "audio_observation"]`
- `["dialog_text", "image_observation", "audio_observation"]`

It must continue to call `project_text_chat_compatibility_fields(episode)` and
must not add media fields to `original_query`, `context`, `current_user_id`, or
`character_user_id`.

## LLM Call And Context Budget

- New live `/chat` LLM calls introduced by Stage 09: zero.
- Existing live `/chat` image descriptor calls: unchanged. The current
  descriptor may still call one vision descriptor per image row already routed
  through `multimedia_descriptor_agent`.
- New audio descriptor calls: zero.
- New graph nodes or retries: zero.
- Prompt text byte changes: zero for L1/L2/L3 prompt constants.
- New cognition human-message context: at most one `media_observations` object
  containing up to four media descriptions total, each capped at 800
  characters.
- RAG context growth: zero media fields.
- Dialog context growth: zero direct media fields from Stage 09.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_episode.py` - add media-description row
  contract, builder argument, refresh helper, media validation, and focused
  projection helpers.
- `src/kazusa_ai_chatbot/service.py` - collect existing image/audio description
  rows and pass sanitized media rows into the initial episode builder.
- `src/kazusa_ai_chatbot/nodes/relevance_agent.py` - refresh the existing
  episode after the existing multimedia descriptor updates descriptions.
- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py` - accept approved
  multimodal user-message source profiles while keeping dialog-text RAG
  projection.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
  - add approved multimodal prompt variants and exact media source payload.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py` - add
  prompt-map entries only.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` - add
  prompt-map entries only.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` - add
  prompt-map entries only.
- `tests/test_rag_cognitive_episode_adapter.py` - update the pre-Stage-09
  unsupported-source regression fixture so it uses a still-unsupported source
  profile instead of the newly approved image/audio profile.
- `tests/test_multi_source_cognition_stage_03_prompt_selection.py` - update
  the pre-Stage-09 unsupported-source selector fixture so it uses a
  still-unsupported source profile instead of the newly approved image/audio
  profile.
- `tests/test_relevance_agent.py` - update the pre-Stage-09 descriptor test
  fixture so direct `multimedia_descriptor_agent(...)` calls include the
  required current `cognitive_episode` state key.
- lifecycle rows in the parent plan and registry after completion only.

### Create

- `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`

### Keep

- `src/kazusa_ai_chatbot/message_envelope/prompt_projection.py`
- `src/kazusa_ai_chatbot/brain_service/graph.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator*.py`
- `src/kazusa_ai_chatbot/reflection_cycle/*.py`
- `src/kazusa_ai_chatbot/internal_thought_cognition.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_dialog.py`
- dispatcher, scheduler, adapter delivery, cache, and database modules

## Implementation Order

1. Reread this plan, Stage 08 `Execution Evidence`, the parent ledger row, and
   registry row.
2. Add focused episode contract tests in
   `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`:
   `test_text_only_builder_output_matches_stage_08_snapshot`,
   `test_builder_adds_bounded_image_and_audio_percepts_without_raw_media`,
   `test_builder_drops_empty_unsupported_and_over_cap_media_rows`, and
   `test_replace_text_chat_media_percepts_refreshes_existing_episode_without_mutation`.
   - Expected before implementation: missing media row helper, builder
     argument, or refresh helper failures.
3. Implement only `cognition_episode.py` media contracts and helpers.
   - Verify the focused episode tests.
4. Add RAG adapter tests in the Stage 09 test file:
   `test_rag_accepts_multimodal_profiles_and_projects_dialog_text_only` and
   `test_rag_rejects_pure_media_or_unsupported_source_profiles`.
   - Expected before implementation: multimodal source profiles rejected.
5. Update `rag/cognitive_episode_adapter.py`.
   - Verify the focused RAG adapter tests and
     `tests/test_rag_cognitive_episode_adapter.py`.
6. Add selector and source-payload tests in the Stage 09 test file:
   `test_selector_maps_exact_multimodal_profiles_to_prompt_keys`,
   `test_selector_rejects_unapproved_multimodal_ordering`, and
   `test_source_payload_contains_only_media_observation_strings`.
   - Expected before implementation: missing variants or payload projection.
7. Update `persona_supervisor2_cognition_prompt_selection.py`.
   - Verify the selector/source-payload tests and
     `tests/test_multi_source_cognition_stage_03_prompt_selection.py`.
8. Add prompt-map and fingerprint tests in the Stage 09 test file:
   `test_l1_l2_l3_prompt_maps_accept_multimodal_variants` and
   `test_existing_l1_l2_l3_prompt_bytes_are_unchanged`.
   - Expected before implementation: prompt-map key errors.
9. Update L1/L2/L3 prompt maps by adding only the three approved variant keys
   in each handler map.
   - Verify Stage 09 prompt-map/fingerprint tests.
10. Add service and descriptor handoff tests in the Stage 09 test file:
    `test_service_initial_episode_receives_preexisting_image_and_audio_descriptions`,
    `test_descriptor_refreshes_cognitive_episode_after_image_description`, and
    `test_descriptor_audio_description_passes_through_without_audio_llm_call`.
    - Expected before implementation: service omits audio rows or descriptor
      return dict lacks refreshed `cognitive_episode`.
11. Update `service.py` and `relevance_agent.py` only as specified in
    `Contracts And Data Shapes`.
    - Verify Stage 09 service/descriptor tests.
12. Run the full Verification section.
13. Run the Independent Code Review gate, remediate in-scope findings, rerun
    affected verification, record evidence, then update lifecycle rows to
    completed.

## Progress Checklist

- [x] Stage 1 - prerequisite evidence carried forward.
  - Covers: Step 1.
  - Verify: parent ledger and registry show Stage 08 completed and Stage 09
    approved.
  - Evidence/sign-off: record Stage 08 branch, commits, and regression results;
    next agent starts at Stage 2.
  - Sign-off: `Codex / 2026-05-10` after verifying branch
    `stage-09-multimodal-cognitive-input-sources` at `77847cd` contains Stage
    08 evidence commit `6b62476`; registry marks Stage 09 `approved | ready`;
    parent ledger marks `stage_09` approved.
- [x] Stage 2 - multimodal episode contract complete.
  - Covers: Steps 2-3.
  - Verify: Stage 09 episode contract tests pass and text-only output is
    unchanged.
  - Evidence/sign-off: record red/green results; reread this plan, then start
    Stage 3.
  - Sign-off: `Codex / 2026-05-10` after red collection failure for missing
    media constants/helpers and green focused episode tests with `4 passed`.
- [x] Stage 3 - RAG projection compatibility complete.
  - Covers: Steps 4-5.
  - Verify: Stage 09 RAG tests and `tests/test_rag_cognitive_episode_adapter.py`
    pass.
  - Evidence/sign-off: record command output; reread this plan, then start
    Stage 4.
  - Sign-off: `Codex / 2026-05-10` after red multimodal RAG rejection and
    green Stage 09 plus RAG adapter regression tests with `15 passed`.
- [x] Stage 4 - cognition prompt admission complete.
  - Covers: Steps 6-9.
  - Verify: Stage 09 selector/source-payload/prompt-map/fingerprint tests and
    Stage 03 prompt-selection tests pass.
  - Evidence/sign-off: record prompt fingerprint and test output; reread this
    plan, then start Stage 5.
  - Sign-off: `Codex / 2026-05-10` after selector/source-payload red,
    prompt-map red, CJK module compile, and green Stage 09 plus Stage 03 tests
    with `47 passed`.
- [x] Stage 5 - service and descriptor handoff complete.
  - Covers: Steps 10-11.
  - Verify: Stage 09 service/descriptor tests prove post-descriptor episode
    refresh, audio pass-through, and no raw media in episode or source payload.
  - Evidence/sign-off: record command output; reread this plan, then start
    Stage 6.
  - Sign-off: `Codex / 2026-05-10` after red service/descriptor handoff tests,
    green Stage 09 focused tests with `14 passed`, and green adjacent
    descriptor/queue regression command with `62 passed`.
- [x] Stage 6 - full verification complete.
  - Covers: Step 12.
  - Verify: every Verification command passes or has an explicitly allowed
    no-match exit.
  - Evidence/sign-off: record command output; reread this plan, then start
    Stage 7.
  - Sign-off: `Codex / 2026-05-10` after static compile, change-surface gate,
    static greps, `git diff --check`, focused tests, adjacent descriptor/queue
    tests, and all listed prior-stage regression gates passed.
- [x] Stage 7 - independent code review complete.
  - Covers: Step 13.
  - Verify: full diff reviewed against style, code quality, plan alignment,
    design weaknesses, regression coverage, handoff artifacts, and verification
    accuracy.
  - Evidence/sign-off: record findings, fixes, rerun commands, residual risks,
    and approval; lifecycle rows may be completed only after this stage.
  - Sign-off: `Codex / 2026-05-10` after independent review fixed one
    no-base64 image descriptor gap, one stale plan-label test docstring, one
    incomplete helper docstring, and one inaccurate media-count log label.

## Verification

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_episode.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\nodes\relevance_agent.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py`

### Change Surface Gate

- `git status --short`

  Expected result: only files listed in `Change Surface` may appear, including
  untracked created files. Lifecycle row edits are allowed only after
  independent code review approval.

### Static Greps

- `rg -n "\"base64_data\"|\"image_url\"|\"data:" src\kazusa_ai_chatbot\cognition_episode.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py`

  Expected result: no matches, with `rg` exit code 1 accepted. Cognition
  episode content, source payloads, and prompt maps must not mention raw media
  fields.

- `rg -n "image_observation|audio_observation|media_observations" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_images.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin_policy.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_reflection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_schema.py src\kazusa_ai_chatbot\reflection_cycle src\kazusa_ai_chatbot\internal_thought_cognition.py src\kazusa_ai_chatbot\dispatcher`

  Expected result: no matches, with `rg` exit code 1 accepted. Stage 09 does
  not change consolidation, reflection, internal thought, or dispatcher policy.

- `rg -n "build_text_chat_media_description_rows|replace_text_chat_media_percepts" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\nodes\relevance_agent.py`

  Expected result: matches only in Stage 09 imports and the approved service
  builder call or descriptor refresh call.

- `git diff --check`

  Expected result: exit code 0.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py`

### Prior Stage Regression Gates

- `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py`
- `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py`
- `venv\Scripts\python -m pytest tests\test_consolidation_origin_policy.py tests\test_consolidator_origin_policy_db_writer.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_07_reflection_dry_run.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`

## Independent Plan Review

Review completed on 2026-05-10 before approval.

- Architecture alignment: approved after the descriptor handoff was made
  explicit. The revised plan keeps adapter/service intake thin, RAG
  dialog-text-only, cognition source-aware, and consolidation/scheduler/dialog
  out of scope.
- Stage readiness: approved after Stage 08 evidence was carried forward and the
  lifecycle blocker was removed.
- Instruction completeness: approved after exact media row shape, helper
  signatures, source profiles, variant names, prompt keys, source payload shape,
  service/descriptor wiring, and verification gates were specified.
- Creativity suppression: approved after helper freedom, prompt edits, raw media
  paths, alternate graph entrypoints, new LLMs, fallback parsers, and fixture
  drift were explicitly forbidden.
- Stage boundaries: approved. Stage 09 owns multimodal episode admission only;
  Stage 10 still owns permissioned proactive output and transport.

## Independent Code Review

Before lifecycle completion, merge, or final sign-off, run an independent code
review over the full implementation diff. Prefer a reviewer that did not write
the code. If no separate reviewer is available, the active agent must reread
this entire plan and perform a fresh review pass from a code-review stance.

The review must check:

- `py-style` compliance, including imports, docstrings, required-field access,
  exception handling, named return values, and no thin wrappers;
- no CJK prompt byte drift in L1/L2/L3;
- exact plan alignment for every public name, helper signature, variant,
  prompt key, payload shape, and verification gate;
- no raw media leakage into cognition episodes, source payloads, RAG requests,
  dialog payloads, or new logs;
- no deterministic semantic interpretation of media descriptions as commands,
  preferences, promises, permissions, or durable facts;
- no changes outside the approved Change Surface;
- no stale static grep expectations;
- regression gates and prompt fingerprint tests cover the actual changed
  contracts.

Findings must be fixed directly only when the fix is inside the approved Change
Surface and does not change the plan contract. If a finding requires a new
contract, file, prompt text, graph path, or cutover policy, stop and update this
plan before changing code. Record findings, fixes, rerun commands, residual
risks, and review approval in `Execution Evidence`.

## Acceptance Criteria

Stage 09 is complete when:

- text-only `build_text_chat_cognitive_episode(...)` output is unchanged from
  Stage 08;
- incoming pre-described image/audio attachments can enter the initial episode
  as bounded media percepts;
- the existing multimedia descriptor refreshes the episode after it generates
  image descriptions;
- audio descriptions pass through only when already supplied, with no new audio
  descriptor LLM call;
- image/audio descriptions become bounded typed percepts when supplied;
- raw media never enters episode percept content, cognition source payloads,
  RAG requests, or new logs produced by the tested path;
- RAG still uses dialog-text projection only;
- L1/L2/L3 prompt constants are byte-for-byte unchanged;
- text-only `/chat`, Stage 03, Stage 06, Stage 07, and Stage 08 regression
  gates pass;
- independent code review is complete and approved;
- no proactive output, transport, scheduler, dispatcher, dialog, or
  consolidation write behavior is introduced.

## Plan Self-Review

Final self-review on 2026-05-10:

- **Coverage:** parent Stage 09 scope maps to episode contracts, service intake,
  descriptor handoff, RAG, prompt selection, raw-media safety, and regression
  checks.
- **Placeholder scan:** no unresolved blockers, questions, or Stage 08 evidence
  placeholders remain.
- **Contract consistency:** input-source labels, prompt variants, source payload
  keys, and text-chat preservation match the parent architecture.
- **Granularity:** checkpoints split episode, RAG, prompt, service/descriptor,
  verification, and independent review work.
- **Verification:** raw-media exclusion, text-only no-regression,
  source-payload safety, prompt byte stability, prior-stage gates, and
  change-surface checks are explicit.
- **Approval status:** approved for Stage 09 execution.

## Execution Handoff

Execution mode: sequential implementation on a feature branch forked from
post-Stage-08 `main`.

Next action: create the Stage 09 feature branch from `main`, reread this
approved plan, then start at Progress Checklist Stage 1.

Do not start Stage 10 review or implementation until Stage 09 execution
evidence is recorded and the parent ledger row is completed.

## Risks

Primary risks are stale post-descriptor episodes, raw media leakage, text-only
regression, media-driven RAG drift, media-command inference, prompt drift, and
write-policy expansion. Verification and Independent Code Review own the gates.

## Completion Artifact Contract

Completion requires episode media contracts, focused tests, service/descriptor
sanitized media wiring, RAG dialog-text projection, multimodal prompt
selector/source payload/map entries, independent code review, completed
lifecycle rows, and execution evidence. It must not include new media
summarizers, raw media prompts, pure-media support, prompt text edits,
proactive output, scheduler, dispatcher, dialog, or consolidation changes.

## Execution Evidence

Record after implementation:

- Stage 08 evidence reread:
  branch `stage-08-internal-thought-cognition-dry-run`, implementation commit
  `ef1449b`, evidence/mainline commit `6b62476`, Stage 08 focused
  `26 passed`, Stage 07 reflection dry-run `14 passed`, Stage 03 selector
  `36 passed`, Stage 06 origin-policy gates `9 passed`, Stage 00 baseline
  `11 passed`, and Stage 08 independent review fixed the internal-thought
  `action_latch` value-type validation gap. Current Stage 09 branch
  `stage-09-multimodal-cognitive-input-sources` is based on approved-plan
  commit `77847cd`; `git merge-base --is-ancestor 6b62476 HEAD` passed.
- Independent plan review:
  completed on 2026-05-10; approved for execution after review-derived fixes.
- Branch: `stage-09-multimodal-cognitive-input-sources`
- Commit:
- Static compile:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_episode.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\nodes\relevance_agent.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py`
  exited 0.
- Change surface gate:
  `git status --short` showed only approved Change Surface files: this plan,
  the parent ledger, the registry, the eight approved source files, the Stage
  09 test file, and the three review-derived fixture files added to Change
  Surface.
- Static greps:
  The raw-media grep was corrected to quoted literals so `data:` does not
  match `metadata:`; it returned no matches with accepted exit code 1. The
  consolidation/reflection/internal-thought/dispatcher grep was corrected to
  explicit consolidator paths for PowerShell; it returned no matches with
  accepted exit code 1. The approved helper grep matched only imports plus the
  service builder call and descriptor refresh calls. `git diff --check` exited
  0.
- Focused tests:
  Stage 2 red:
  `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py -q`
  failed during collection with missing
  `MAX_COGNITIVE_EPISODE_MEDIA_DESCRIPTION_CHARS`. Stage 2 green:
  the same focused command passed with `4 passed`.
  Stage 3 red:
  the Stage 09 focused command failed with multimodal input-source profiles
  rejected by `build_text_chat_rag_request(...)`. Stage 3 green:
  `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py tests\test_rag_cognitive_episode_adapter.py -q`
  passed with `15 passed`.
  Stage 3 plan correction:
  `tests/test_rag_cognitive_episode_adapter.py` was added to Change Surface
  after the regression gate exposed a stale fixture that used the newly
  approved image profile as an invalid source set.
  Stage 4 red:
  Stage 09 focused tests failed because multimodal prompt variants were not
  selected and `media_observations` payload projection was missing. After
  selector implementation, Stage 03 failed because its unsupported-source
  fixture used the newly approved image profile. The plan Change Surface was
  updated and the fixture now uses `retrieved_memory`. Stage 09 focused tests
  then failed once on missing L1/L2/L3 prompt-map variant keys.
  Stage 4 green:
  CJK module compile for L1/L2/L3 exited 0; then
  `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py tests\test_multi_source_cognition_stage_03_prompt_selection.py -q`
  passed with `47 passed`. Prompt fingerprint tests covered all nine L1/L2/L3
  prompt constants with unchanged byte lengths and SHA-256 digests.
  Stage 5 plan correction:
  `tests/test_relevance_agent.py` was added to Change Surface after the
  adjacent descriptor regression gate exposed a stale direct-node fixture that
  did not include the now-required `cognitive_episode` state key.
  Stage 5 red:
  Stage 09 focused tests first failed because service did not pass
  `media_description_rows` into the episode builder and the descriptor node did
  not return `cognitive_episode`.
  Stage 5 green:
  `venv\Scripts\python -m py_compile tests\test_relevance_agent.py src\kazusa_ai_chatbot\nodes\relevance_agent.py src\kazusa_ai_chatbot\service.py`
  exited 0. Then
  `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py tests\test_relevance_agent.py tests\test_service_input_queue.py -q`
  passed with `62 passed`.
  Stage 6 focused:
  `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py`
  passed with `15 passed` after the independent-review no-base64 image
  pass-through test was added.
- Prior stage regression gates:
  `tests\test_cognitive_episode_contract.py` passed with `15 passed`;
  `tests\test_rag_cognitive_episode_adapter.py` passed with `9 passed`;
  `tests\test_multi_source_cognition_stage_03_prompt_selection.py` passed with
  `36 passed`; `tests\test_consolidation_origin_policy.py
  tests\test_consolidator_origin_policy_db_writer.py` passed with `9 passed`;
  `tests\test_multi_source_cognition_stage_07_reflection_dry_run.py` passed
  with `14 passed`;
  `tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
  passed with `26 passed`; and
  `tests\test_multi_source_cognition_stage_00_regression_baseline.py` passed
  with `11 passed`.
- Independent code review:
  Completed on 2026-05-10. Review checked changed Python against py-style
  constraints, plan alignment, raw-media safety, prompt byte stability, RAG
  dialog-text projection, handoff freshness, and regression coverage. Fixes
  made during review: added a docstring to `_trim_media_description(...)`;
  renamed the service debug log field from `image_attachments` to
  `media_attachments`; removed a plan-stage label from a test helper
  docstring; and prevented the existing vision descriptor from being called
  with an empty data URI when an image row has a preexisting description but no
  base64 payload. Added
  `test_descriptor_image_description_without_base64_skips_vision_llm`.
  Rerun after review: py_compile for affected files exited 0;
  `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py tests\test_relevance_agent.py -q`
  passed with `38 passed`; full Verification static checks and all listed
  prior-stage regression gates were rerun and passed.
- Completion diff review:
- Lifecycle records:
- Residual risks:
- Sign-off:
