# first class image input cognition plan

## Summary

- Goal: Make image attachments a first-class cognition input while preserving
  description-only storage, local-LLM reliability, bounded response latency,
  adapter neutrality, and raw media safety.
- Plan class: large
- Status: completed
- Mandatory skills for implementation: `development-plan-writing`,
  `local-llm-architecture`, `no-prepost-user-input`, `py-style`,
  `test-style-and-execution`, and `cjk-safety` before editing cognition Python
  files that contain CJK prompt text.
- Overall cutover strategy: compatible for text-only `/chat`; compatible for
  existing image-description storage and replay behavior; bigbang removal of
  image-description smearing into `user_input`, RAG query text, and untyped
  cognition payloads.
- Key failure modes: weak local LLM ignoring image context, accidental RAG
  query drift, raw `base64_data` prompt leakage, image-only turn regression,
  quoted QQ images that no longer have retrievable image bytes, prompt drift
  without tests, and background consolidation failure on multimodal
  user-message episodes.
- Acceptance criteria: image observations enter cognition through bounded
  structured source payloads; durable storage remains attachment-description
  only; quoted-image replies use the previous stored description when present,
  refetch only when adapter-supplied current metadata makes that possible, and
  otherwise expose an explicit unavailable visual context; raw media remains
  outside cognition/RAG/dialog prompts; RAG uses dialog/query intent only;
  text-only behavior remains unchanged; image-only turns can be reasoned over
  by cognition without text smearing; multimodal user-message consolidation
  origin is metadata-only and no longer fails solely because image observations
  are present.

This approved plan is ready for implementation. It records the selected
production architecture, dependencies, and staged development gates for the
implementation pass.

## Context

The completed multi-source cognition architecture already introduced
`CognitiveEpisode` and the `image_observation` input-source label. Current code
can accept inline image attachments and produce image descriptions before
relevance. However, image input is still treated as an auxiliary description
string in several places instead of as a consistently modeled cognition input.

Reference artifacts are the completed multi-source cognition parent plan plus
Stage 09 multimodal input sources and Stage 10 permissioned proactive output.
Current ownership boundaries are the message envelope projection, service queue
and reply hydration, descriptor/relevance node, `CognitiveEpisode`, RAG episode
adapter, cognition prompt selection, L1/L2/L3 cognition nodes, and consolidation
origin.

## Current System Findings

- The envelope and prompt projection layers already have typed attachment
  fields and prompt-safe attachment summaries. Raw media is excluded from prompt
  projections.
- `multimedia_descriptor_agent(...)` is the only current production node that
  sends raw image bytes to a vision-capable model. It currently emits one
  free-form `description`.
- `CognitiveEpisode` already has `image_observation` as an input source, but
  cognition source payloads expose image observations as strings instead of
  compact typed visual facts.
- `relevance_agent(...)` currently appends image descriptions into
  `user_input`. This can make the descriptor look user-authored and can leak
  image text into the decontextualizer/RAG path.
- `_hydrate_reply_context(...)` can find exact prior conversation rows, but it
  currently hydrates text reply metadata only. Quoted QQ images need prompt-safe
  attachment descriptions from the prior row when current image bytes are no
  longer retrievable.
- Background consolidation origin currently accepts only dialog-text
  user-message profiles. Multimodal user-message episodes should be accepted as
  metadata-only origins without persisting image content.

## Problem Statement

The current image path is safe enough to avoid raw media prompt leakage, but it
is not yet a clean first-class cognition contract:

- image facts can be duplicated across `prompt_message_context`,
  `media_observations`, and appended `user_input`;
- local LLM prompts do not consistently explain how to use image observations;
- RAG can accidentally receive image descriptions through mutated text;
- image-only turns rely on text smearing for downstream stages;
- quoted/replied images can lose their visual context when the current adapter
  event cannot refetch the original image and the brain projects only the
  quoted text excerpt;
- the character can mistake the image descriptor for text authored by the user
  when it is appended into `user_input`, causing intent drift and overfitting to
  descriptor wording;
- consolidation rejects multimodal user-message origins even when only
  metadata would be persisted;
- the descriptor emits one free-form paragraph, which is harder for weak local
  LLMs to use than compact typed visual facts.

## Selected Production Architecture

Use structured descriptor-mediated image observations as the production
architecture.

The existing descriptor call remains the only production stage that inspects raw
image bytes. It emits a compact structured observation containing summary,
visible text, salient visual facts, scene/spatial facts, uncertainty, and a
compatibility `description` string. The cognition layers receive that structured
observation as typed visual evidence, never as user-authored text.

Live text+image evidence is the deciding input for this plan. Descriptor-mediated
cognition and direct raw-image cognition tied on support/correction quality with
the configured model. Because quality tied, production constraints decide the
architecture: descriptor-mediated cognition preserves description-only storage,
supports quoted/replied image replay, keeps raw media out of cognition/RAG/dialog
prompts, and remains compatible with weaker local deployments.

The selected architecture also removes `user_input` image smearing, teaches
L1/L2/L3 prompts how to use typed image observations, preserves RAG query
boundaries, and allows metadata-only multimodal consolidation origins.

The target runtime shape is: adapter image attachment -> typed envelope ->
service multimedia input -> existing descriptor -> structured image observation
percept -> cognition source payload -> L1/L2/L3 visual evidence reasoning ->
dialog -> metadata-only consolidation origin.

Quoted-image target shape is: hydrate the exact previous conversation row by
platform message id, expose its stored attachment description when present,
refetch only from current adapter-provided retrievable media, otherwise mark
the visual context unavailable without inventing facts.

Do not pass image observations into RAG as query material in this plan. RAG may
still use the user's text question and ordinary prompt context. Visual-RAG or
image-specialist retrieval is a follow-up plan.

## Mandatory Skills

- `development-plan-writing`: load before implementation or lifecycle edits.
- `local-llm-architecture`: load before changing prompt admission, cognition
  source payloads, graph handoff, RAG projection, or LLM call budgets.
- `no-prepost-user-input`: load before changing any code that might interpret
  image text as accepted user commands, preferences, promises, permissions, or
  durable instructions.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing L1/L2/L3 or descriptor prompt modules that
  contain CJK text.

## Mandatory Rules

- Implementation may start from this approved plan.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- Keep adapters thin. Adapter changes may normalize attachment metadata, but
  visual semantic judgment belongs to the descriptor/model path.
- Production storage remains description-only. Do not add image blobs,
  thumbnails, image hashes, vector embeddings, OCR tables, structured visual
  fact rows, or raw-image archives to durable storage in this plan.
- Runtime structured image observations are prompt/runtime artifacts. The only
  durable image semantic field is still the bounded attachment description.
- In this production plan, the descriptor remains the only node allowed to send
  raw image bytes to a vision model.
- Do not place `base64_data`, raw bytes, data URIs, attachment URLs, or
  unbounded attachment metadata in cognition episodes, cognition source
  payloads, RAG requests, dialog payloads, consolidator prompt payloads, or new
  logs.
- Do not mutate `state["user_input"]` to append image descriptions.
- Do not represent image descriptors as if the user typed them. Descriptors are
  evidence in typed media fields only.
- For quoted/replied images, include prompt-safe stored attachment descriptions
  from the exact reply target when present. If no description is available,
  refetch only through adapter-provided current-event image data or URL
  metadata; otherwise mark the visual context unavailable.
- Do not add deterministic keyword classifiers that decide image meaning,
  accepted commands, preferences, promises, permissions, or durable facts.
- Structural validation, caps, schema sanitation, and raw-media exclusion are
  deterministic responsibilities.
- Semantic visual interpretation belongs to the descriptor prompt and cognition
  prompts.
- RAG stays dialog/query-intent only in this plan. Do not add visual RAG,
  image search, OCR search, web lookup from image labels, or specialist
  image-retrieval loops.
- Prompt changes must include exact input-format and generation-procedure
  updates. Do not rely on hidden JSON fields being self-explanatory to a weak
  local model.
- Prompt budget analysis is a planning-phase requirement. This plan must define
  the affected stages, current prompt sizes, target prompt caps, image payload
  budget, call-count impact, context margin, and latency impact before prompt
  implementation begins.
- Prompt modifications must be full prompt refactors that organically integrate
  image observations into the prompt's role, input format, generation procedure,
  and output contract. Do not append an isolated image note to the old prompt.
- Every prompt modification must be validated with real LLM tests for the
  intended image behavior and for likely regressions caused by the prompt
  rewrite. Run real LLM tests one case at a time and inspect the trace before
  recording success.
- Text-only `/chat` output, prompt payloads, and RAG request behavior must
  remain unchanged except where tests explicitly prove no-regression.

## Must Do

- Add failing tests that capture the current image-path contract gaps:
  `user_input` smearing, accidental RAG query contamination, prompt payload
  ambiguity, quoted-image description loss, and consolidation-origin rejection.
- Keep durable storage description-only while adding prompt-safe quote/reply
  hydration from previously stored attachment descriptions.
- Replace free-form image-only source payloads with a bounded structured image
  observation contract.
- Update the existing descriptor prompt to emit the structured observation in
  the same descriptor call.
- Preserve the existing `description` string as a compatibility summary for
  prompt context and conversation attachment description updates.
- Update `CognitiveEpisode` media percept construction so image percepts can
  carry structured prompt-safe observation metadata.
- Update `build_cognition_prompt_source_payload(...)` to emit structured image
  observation rows, not only plain strings.
- Update L1/L2/L3 prompt input-format and generation-procedure sections so the
  model explicitly uses image observations.
- Keep changed descriptor and cognition prompts within the planning budget in
  `LLM Call And Context Budget`, and keep the call count unchanged unless this
  plan is amended again before implementation.
- Refactor each touched prompt as one coherent prompt contract. Integrate image
  observations into the existing task framing, examples, procedure, input
  format, and output rules instead of adding a trailing image-specific clause.
- Stop appending image descriptions into `state["user_input"]`.
- Add prompt wording and tests proving image observations are visual evidence,
  not user-authored text or instructions.
- Keep RAG request construction free of image descriptions unless a later plan
  explicitly adds a visual-RAG contract.
- Allow user-message multimodal input-source profiles in consolidation origin
  metadata without copying percept content into consolidation metadata.
- Add focused tests and static greps proving raw media is absent from every
  prompt-facing payload touched by this plan.

## Deferred

- Raw image payloads inside cognition prompts.
- Raw image payloads inside RAG, dialog, or consolidator prompts.
- Visual RAG, web search from image labels, image identity lookup, reverse
  image search, or OCR-driven retrieval.
- A separate image specialist agent or planner-routed visual tool.
- Video support.
- Audio changes beyond preserving current pass-through behavior.
- Outbound image generation or image attachments.
- Adapter file-storage redesign, CDN storage, structured image persistence, or
  database migration for existing inline image payloads.
- Reflection/internal-thought image mixing.
- Proactive output or scheduler behavior changes.
- Durable memory write policy that treats image observations as accepted user
  instructions through deterministic code.

## Cutover Policy

Overall strategy: compatible with a few explicit bigbang cleanups.

| Area | Policy | Instruction |
|---|---|---|
| Text-only `/chat` | compatible | Preserve existing prompt payloads, RAG request shape, dialog behavior, and consolidation behavior. |
| Existing image-description intake | compatible | Keep accepting current image attachments and preexisting descriptions. |
| Durable image storage | compatible | Continue storing only prompt-safe attachment descriptions; do not add raw or structured image storage. |
| Quoted image replies | compatible | Hydrate previous stored attachment descriptions when an exact reply target row exists; refetch only if the current adapter event carries usable image data or URL metadata. |
| Image cognition source payload | bigbang | Replace plain image observation strings with the structured prompt-safe image observation payload defined by this plan. |
| `user_input` image smearing | bigbang | Remove image-description appending into `state["user_input"]`; do not preserve a fallback text-smearing path. |
| RAG query | bigbang | Keep media descriptions out of `original_query` and RAG context except for existing prompt message attachment summaries that are already part of safe context. |
| Consolidation origin | compatible | Accept user-message multimodal source profiles as metadata only; do not copy percept content. |
| Raw media safety | bigbang | Reject or strip raw media from all cognition/RAG/dialog/consolidation prompt payloads. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative or broader strategy by default.
- If an area is `bigbang`, delete or rewrite the legacy behavior instead of
  adding compatibility shims.
- If an area is `compatible`, preserve only the compatibility surfaces
  explicitly listed in this plan.
- Any change to a cutover policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The implementation agent may choose local private helper names only inside
  the files named in `Change Surface`, and only for repeated structural
  validation, cap enforcement, or prompt-safe projection.
- The implementation agent must search for existing equivalent projection or
  trimming helpers before adding new ones.
- The implementation agent must not add new graph entrypoints, feature flags,
  retry loops, prompt repair loops, or compatibility layers.
- The implementation agent must not change scheduler, dispatcher, proactive
  output, reflection worker, adapter delivery, DB bootstrap, or memory
  collection schemas.
- If implementation discovers a required change outside `Change Surface`, stop
  and update this plan before editing that file.
- If the plan and code disagree, preserve the stated architecture intent and
  report the discrepancy.

## Target State

Text-only turns remain unchanged. Image turns produce a user-message episode
with both `dialog_text` and `image_observation` when body text is present, and
with `image_observation` only when the message is image-only.

Each image percept uses `input_source="image_observation"`,
`content="<short image summary>"`, and prompt-safe metadata containing origin,
source message id when available, media kind, summary status, and bounded image
facts.

Quoted-image turns with no current image bytes use the same typed percept
shape, but set `observation_origin="quoted_reply_attachment"` and use the
previous stored attachment description as `content` when available.

If the quoted image cannot be described from storage or current adapter
metadata, the prompt-safe context must contain `summary_status="unavailable"`
and no invented visual facts.

The cognition source payload exposes typed visual facts under
`media_observations.image_observations[]` and preserves existing bounded audio
descriptions under `media_observations.audio_observations`.

The approved field names are `observation_origin`, `source_message_id`,
`media_kind`, `summary_status`, `summary`, `visible_text`,
`salient_visual_facts`, `spatial_or_scene_facts`, and `uncertainty`.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Durable storage | Keep description-only attachment storage. | Preserves current storage contract and avoids raw-media or schema migration scope. |
| Image semantics owner | Existing descriptor LLM produces visual semantics. | Keeps adapters thin and avoids raw media in core cognition. |
| Descriptor call count | Reuse the existing descriptor call; no extra response-path image LLM call. | Balances precision and latency for local deployments. |
| Prompt payload | Use structured image observations with compact lists. | Weak local LLMs follow labeled fields better than long prose. |
| Descriptor intent boundary | Treat image observations as evidence, not user-authored text. | Prevents the character from drifting toward descriptor wording as if the user said it. |
| Quoted image fallback | Use previous stored attachment descriptions first; refetch only from current adapter-provided retrievable image metadata; otherwise mark unavailable. | Addresses QQ quoted-image loss while preserving description-only storage and adapter neutrality. |
| RAG | Keep image observations out of RAG query material. | Prevents retrieval drift and preserves RAG evidence ownership. |
| Cognition prompts | Update input formats and generation procedures explicitly. | Hidden JSON fields are unreliable with local LLMs. |
| Consolidation | Admit multimodal user-message origin metadata only. | Prevents post-response failure without letting image content leak into write metadata. |
| Image-only turns | Support empty body text plus image observation for cognition. | Current envelope allows attachment-only messages; cognition should not require text smearing. |
| Raw media | Descriptor-only in production cognition. | Preserves privacy, prompt safety, quote replay, and weak-model compatibility. |

## Dependencies

- The multi-source cognition architecture must remain completed and current.
- Route-specific `VISION_DESCRIPTOR_LLM_*` settings must point to a model that
  accepts OpenAI-compatible image payloads for live image tests.
- Route-specific `COGNITION_LLM_*` settings must support the existing text
  cognition layers. Production does not require the cognition route itself to
  accept images.
- The local seed images under `personalities/seeding_images/generated` must
  remain available for live comparison smoke tests.
- The implementation branch must start from a clean mainline that includes the
  completed Stage 09 and Stage 10 artifacts.
- This approved executable plan has settled field names and exact verification
  gates.

## LLM Call And Context Budget

Current image path:

- Descriptor: one vision LLM call per image row with `base64_data`; no hard
  descriptor-call cap in the descriptor loop.
- Relevance: one existing text LLM call, with image description currently
  appended into `user_input`.
- Decontextualizer/RAG/cognition/dialog: existing calls, potentially seeing
  image descriptions through mutated `user_input` and prompt context.

Target image path:

- Descriptor: still the only image model call; cap descriptor calls to the
  same maximum image observations admitted to cognition.
- Relevance: existing text LLM call sees image information through
  `prompt_message_context` and structured source context only if explicitly
  wired; no text smearing.
- Decontextualizer: existing text LLM call receives typed prompt context;
  image descriptions do not become part of `user_input`.
- RAG: existing call budget; no new visual retrieval loop.
- Cognition: existing L1/L2/L3 call graph; prompt payload gains bounded
  structured image observations.

Default context cap assumption: 50k tokens. Structured image observations must
stay under the existing four-attachment, 800-character-per-image scale unless
the approved final plan tightens the cap further.

Planning-phase prompt budget:

| Prompt | Current chars | Current tokens | Target prompt cap |
|---|---:|---:|---:|
| Descriptor | 829 | 207 | 2,200 chars |
| L1 subconscious | 1,768 | 442 | 2,600 chars |
| L2 consciousness | 6,795 | 1,699 | 7,800 chars |
| L2 boundary | 5,425 | 1,356 | 6,400 chars |
| L2 judgement | 4,012 | 1,003 | 5,000 chars |
| L3 contextual | 2,982 | 746 | 3,900 chars |
| L3 style | 3,748 | 937 | 4,700 chars |
| L3 content anchor | 8,556 | 2,139 | 9,800 chars |
| L3 preference | 4,151 | 1,038 | 5,100 chars |
| L3 visual | 4,337 | 1,084 | 5,600 chars |

Budget conclusions:

- Token estimate uses 4 characters per token for planning.
- Runtime call count remains unchanged; real LLM validation calls are test-only.
- Descriptor calls are capped to the same maximum image observations admitted
  to cognition: four images.
- Worst-case structured image payload is four images * 800 semantic characters
  plus JSON/list overhead: 4,800 characters, about 1,200 tokens.
- Increment over current plain descriptions is budgeted at 1,400 characters,
  about 350 tokens.
- Largest touched prompt remains L3 content anchor: 9,800 prompt chars plus
  worst-case image payload, under 3,700 planning tokens before existing
  non-image state and with more than 46k tokens of nominal 50k context left.
- Expected latency impact is token-only; no new model round trips are allowed.
- If implementation cannot stay within these caps, amend this plan before
  editing prompts.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_episode.py`
  - Add or refine structured image observation TypedDicts, media percept
    metadata projection, caps, and validation.
- `src/kazusa_ai_chatbot/service.py`
  - Extend reply-target hydration to surface prompt-safe replied-message
    attachment descriptions when an exact conversation row is found.
- `src/kazusa_ai_chatbot/message_envelope/prompt_projection.py`
  - Project reply-target attachment summaries without adding raw fields or
    changing durable storage.
- `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
  - Update descriptor output schema handling.
  - Stop mutating `user_input` with image descriptions.
  - Preserve descriptor fallback behavior and current-row description updates.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
  - Update source-payload projection for structured image observations.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
  - Update prompt input contract and handler payload tests for image
    observations.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
  - Update prompt input contracts and handler payload tests for image
    observations.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Update contextual/content/style/preference/visual prompt input contracts
    where image observations affect behavior.
- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
  - Add safeguards/tests that RAG rejects accidental image observation leakage
    into query/context material.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
  - Allow approved user-message multimodal source profiles as metadata-only
    origins.

### Test Files

- `tests/test_multi_source_cognition_image_input.py`
  - New focused contract and integration test file for this plan.
- Existing adjacent tests may be updated only when their fixtures become stale
  because of the approved contract:
  - `tests/test_relevance_agent.py`
  - `tests/test_service_background_consolidation.py`
  - `tests/test_service_input_queue.py`
  - `tests/test_prompt_message_context.py`
  - `tests/test_rag_cognitive_episode_adapter.py`
  - `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`
  - `tests/test_consolidation_origin_metadata.py`

### Keep

- `src/adapters/*`: no visual semantic behavior change. If QQ/NapCat exposes
  current-event quote image bytes or a still-valid URL, the adapter may forward
  it as ordinary typed attachment input; the brain must not parse CQ/debug wire
  syntax or call platform-specific refetch APIs directly.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`: no `/chat` API schema
  change.
- `src/kazusa_ai_chatbot/brain_service/graph.py`: no graph topology change.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`: no persona graph
  stage order change unless required only to prevent RAG image leakage.
- Scheduler, dispatcher, proactive output, reflection worker, DB bootstrap,
  and memory collection schemas stay untouched.

## Implementation Order

1. Add focused tests for current unwanted behavior.
   - Prove image descriptions currently enter `decontexualized_input`/RAG
     through `user_input` mutation.
   - Prove quoted/replied image attachment descriptions are not available to
     cognition when only the previous stored row has the description.
   - Prove multimodal user-message consolidation origin currently fails.
   - Prove cognition source payload currently carries only strings.
2. Add structured image observation contract tests.
   - Test sanitized projection, caps, fallback from plain description, and raw
     media exclusion.
3. Preserve live evidence traces for the selected architecture decision.
   - Future implementation should run production descriptor-mediated live cases
     one at a time only when route configuration supports it.
4. Implement reply/quoted-image hydration.
   - Reuse exact platform message ID lookup.
   - Project prompt-safe attachment descriptions from the replied-to row.
   - Mark unavailable when neither stored description nor refetchable
     current-event media exists.
5. Implement structured image observation projection in
   `cognition_episode.py`.
6. Update descriptor prompt and parser in `relevance_agent.py`.
   - Keep one descriptor call per processed image.
   - Refactor the descriptor prompt as one coherent prompt contract.
   - Validate malformed structured fields and degrade to summary-only
     observation.
7. Remove `user_input` image smearing.
   - Preserve prompt context and cognitive episode image observations.
8. Update cognition source payload and L1/L2/L3 prompt input contracts.
   - Refactor each touched cognition prompt as a coherent whole, not by
     appending image-specific notes.
   - Run prompt-render checks, not only syntax checks.
9. Harden RAG projection tests.
   - Ensure `original_query` is dialog/query text only.
10. Update consolidation origin metadata to accept approved multimodal
   user-message profiles without content leakage.
11. Run focused, regression, and real-LLM prompt validation.
12. Run independent code review, remediate in-scope findings, rerun affected
    verification, and record evidence.

## Progress Checklist

- [x] Stage 1 - current behavior captured.
  - Covers: Implementation Order steps 1-3.
  - Verify: focused tests fail for the current contract gaps or record current
    baseline where no failure is expected.
  - Evidence: record exact failure messages in `Execution Evidence`.
  - Sign-off: `Codex/2026-05-10`.
- [x] Stage 2 - image observation contract implemented.
  - Covers: Implementation Order steps 4-6.
  - Verify: structured observation tests pass; raw-media greps pass.
  - Evidence: record changed files and focused test output.
  - Sign-off: `Codex/2026-05-10`.
- [x] Stage 3 - cognition prompt payload and prompt contracts updated.
  - Covers: Implementation Order steps 7-8.
  - Verify: prompt selection/source payload tests, prompt render checks, and
    affected cognition prompt tests pass.
  - Evidence: record prompt-render command output, final prompt sizes against
    planning caps, organic prompt-refactor notes, and test output.
  - Sign-off: `Codex/2026-05-10`.
- [x] Stage 4 - RAG and consolidation boundaries corrected.
  - Covers: Implementation Order steps 9-10.
  - Verify: RAG projection tests and consolidation origin tests pass.
  - Evidence: record test output and static grep output.
  - Sign-off: `Codex/2026-05-10`.
- [x] Stage 5 - full verification complete.
  - Covers: Implementation Order step 11.
  - Verify: every command in `Verification` passes or has an explicitly
    accepted no-match exit, and real LLM prompt validation traces are inspected
    one case at a time.
  - Evidence: record all command outputs and live LLM judgments in
    `Execution Evidence`.
  - Sign-off: `Codex/2026-05-10`.
- [x] Stage 6 - independent code review complete.
  - Covers: Implementation Order step 12.
  - Verify: review findings are resolved or recorded as residual risks, and
    affected verification is rerun.
  - Evidence: record review scope, findings, fixes, reruns, residual risks,
    and approval status.
  - Sign-off: `Codex/2026-05-10`.

## Verification

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\message_envelope\prompt_projection.py src\kazusa_ai_chatbot\cognition_episode.py src\kazusa_ai_chatbot\nodes\relevance_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin.py tests\test_multi_source_cognition_image_input.py tests\test_image_cognition_options_live_llm.py`

### Static Greps

- `rg -n "\"base64_data\"|\"image_url\"|\"data:" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\message_envelope\prompt_projection.py src\kazusa_ai_chatbot\cognition_episode.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py`
  - Expected: no matches, with `rg` exit code 1 accepted.
- `rg -n "Image attachment:" src\kazusa_ai_chatbot`
  - Expected: no runtime prompt-building matches after implementation.
- `rg -n "image_observation|media_observations" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_images.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py`
  - Expected: no matches unless a match is explicitly allowed for metadata-only
    origin handling in the final approved plan.

### Prompt Budget And Render Checks

- Verify every changed descriptor, L1, L2, and L3 prompt stays within the
  planning-phase caps in `LLM Call And Context Budget`.
- Render representative and worst-case prompt inputs after implementation.
- Verify each changed prompt is a coherent refactor with image observations
  integrated into role, input format, generation procedure, and output contract.
- Verify no changed prompt only appends a detached image-specific clause.
- Verify production call count remains unchanged.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_image_input.py -q`
- `venv\Scripts\python -m pytest tests\test_prompt_message_context.py tests\test_service_background_consolidation.py -q`
- `venv\Scripts\python -m pytest tests\test_relevance_agent.py tests\test_rag_cognitive_episode_adapter.py tests\test_consolidation_origin_metadata.py -q`

### Regression Tests

- `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py -q`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py -q`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py -q`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py -q`

### Live LLM Smoke

Run only after deterministic tests pass and only when the configured local
vision route is available:

- one production image-description live test, one case at a time with `-q -s`;
- one production cognition live test where the user's text refers to a visible
  image fact;
- one production cognition live test where text and image together determine
  whether the image supports or corrects the user's premise;
- one production text-only cognition regression case for a prompt touched by
  this plan;
- one production image-unavailable or quoted-image fallback case if route and
  fixture support it;
- inspect each trace manually before recording success.

## Independent Plan Review

This gate ran before changing status from `draft` to `approved`.

Review scope:

- Confirm that the plan aligns with the completed multi-source cognition
  architecture and Stage 09 raw-media boundary.
- Confirm that structured image observations are necessary before adding any
  image specialist, visual RAG, or retrieval loop.
- Confirm that durable storage remains description-only.
- Confirm that quoted-image fallback uses previous stored descriptions, allows
  adapter-owned refetch only from current retrievable metadata, and never
  invents visual facts when unavailable.
- Confirm that direct raw-image cognition is not part of the selected
  production response path.
- Confirm that field names, caps, tests, and prompt sections are fully settled.
- Confirm that planning-phase LLM budget analysis is present before
  implementation.
- Confirm that prompt changes are coherent prompt refactors, not appended
  image notes.
- Confirm that real LLM validation covers intended image behavior and prompt
  regressions.
- Confirm that RAG remains dialog/query-intent only.
- Confirm that consolidation origin admits metadata only and cannot write image
  content by accident.

Review result on 2026-05-10:

- Reviewer mode: same-agent fresh review. No separate reviewer was available
  in this session.
- Inputs reviewed: this plan, `development_plans/README.md`,
  `development-plan-writing` plan contract/cutover/execution references,
  current source/test findings captured in this plan, and live LLM trace
  evidence recorded under `test_artifacts/llm_traces`.
- Blockers: none.
- Required edits made during review and tightening:
  - Changed plan class to `large`; the change surface is bounded and the
    longer document length came from supporting evidence, not migration risk.
  - Changed status from `draft` to `approved`.
  - Tightened supporting evidence and removed unselected decision-process
    detail while preserving the selected architecture.
  - Added mandatory prompt-change rules for planning-phase LLM budget analysis,
    organic prompt refactoring, and real LLM validation of fixes and
    regressions.
- Non-blocking findings: the live comparison test file remains an evidence
  artifact, not a production implementation dependency. The selected production
  architecture is descriptor-mediated structured image observations.
- Open questions: none.
- Approval status: approved for implementation.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer
is available, the active agent must reread this plan, inspect the full diff
from a fresh-review posture, and record that no separate reviewer was
available.

Review scope:

- Mandatory skill compliance, especially `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety`.
- Raw-media exclusion from cognition episodes, source payloads, RAG requests,
  dialog payloads, consolidator prompt payloads, and logs.
- Description-only durable storage is preserved.
- Quoted/replied image attachment descriptions are available to cognition when
  they exist in the exact previous conversation row, and unavailable state is
  explicit when they do not.
- Prompt contract clarity for weak local LLMs: input format, output format,
  and generation procedure must match actual handler payloads.
- Prompt implementation stays within the planning-phase budget caps recorded in
  this plan, including worst-case image payload, context margin, call count,
  and latency impact.
- Each changed prompt is refactored as one coherent contract; appended
  image-only notes are not acceptable.
- Real LLM prompt validation traces cover both the image behavior being fixed
  and likely regressions from the prompt rewrite.
- No deterministic semantic interpretation of image text as accepted commands,
  preferences, promises, permissions, or durable facts.
- No accidental RAG query drift from image descriptions.
- No background consolidation failure for approved multimodal user-message
  profiles.
- Plan alignment by `Must Do`, `Deferred`, `Change Surface`, verification
  gates, and acceptance criteria.

Record findings, fixes, rerun commands, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- the plan has been approved before implementation begins;
- text-only `/chat` remains behavior-compatible;
- durable image storage remains description-only;
- image attachments enter cognition as structured, bounded image observations;
- quoted/replied images can use previous stored attachment descriptions when
  raw image bytes are unavailable;
- unavailable quoted-image context is represented explicitly instead of guessed;
- image-only messages can reach cognition without appending image descriptions
  into `user_input`;
- image descriptors are prompt-visible as typed visual evidence, not as
  user-authored text;
- L1/L2/L3 prompts explicitly document how to use image observations;
- every changed prompt stays within the planning-phase LLM budget recorded in
  this plan;
- every changed prompt is refactored as an integrated prompt contract, not
  patched with a detached image note;
- real LLM validation has inspected the intended image behavior and prompt
  regression cases one at a time;
- raw media is absent from all cognition/RAG/dialog/consolidation prompt
  payloads;
- RAG does not receive image descriptions through `original_query` or hidden
  mutated text;
- consolidation origin accepts approved multimodal user-message profiles as
  metadata-only origin records;
- deterministic focused tests, prior-stage regression gates, and approved live
  image smoke tests pass;
- independent code review is complete and approved.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Weak local LLM ignores image facts | Use structured labeled fields and prompt generation guidance. | Prompt-render checks and live cognition smoke. |
| Character treats descriptor as user text | Remove `user_input` smearing and label image observations as evidence. | Relevance tests and prompt contract tests. |
| Prompt rewrite increases latency or context pressure | Use planning-phase caps and keep runtime call count unchanged. | Prompt budget and render checks. |
| Prompt rewrite regresses text-only behavior | Use real LLM regression cases for touched prompts. | Live LLM smoke traces and manual judgment. |
| Quoted QQ image is not retrievable | Prefer previous stored attachment description; refetch only from current adapter-provided media metadata; otherwise mark unavailable. | Reply hydration tests and prompt projection tests. |
| RAG query contamination | Remove `user_input` smearing and test `original_query`. | RAG projection tests and `Image attachment:` grep. |
| Raw media leakage | Keep descriptor-only raw image path and static greps. | Raw-media greps and source payload tests. |
| Descriptor schema brittleness | Validate fields and degrade to summary-only observation. | Malformed descriptor tests. |
| Consolidation failure after image reply | Admit metadata-only multimodal origins. | Consolidation origin tests. |
| Prompt drift across CJK modules | Use `cjk-safety`, compile checks, and prompt render tests. | Compile and focused prompt tests. |

## Plan Self-Review

Approval self-review on 2026-05-10:

- Coverage: current-code findings, selected architecture,
  dependencies, staged execution, verification, and review gates are present.
- User concerns: description-only storage, QQ quoted-image recovery, and
  descriptor-as-user-text intent drift are explicitly represented.
- Text+image determining factor: live evidence showed descriptor-mediated
  cognition ties direct raw-image cognition on support/correction quality, so
  descriptor-mediated cognition is selected for production.
- Prompt-change rules: planning-phase LLM budget analysis, organic prompt
  refactor, and real LLM fix/regression validation are mandatory gates.
- Placeholder scan: no unresolved placeholders or option decisions remain.
- Contract consistency: approved field names are settled and match the target
  state examples in this plan.
- Granularity: implementation stages separate tests, contract, descriptor,
  cognition prompts, RAG/consolidation, verification, and review.
- Verification: raw-media safety, RAG drift, prompt behavior, consolidation
  origin, and prior-stage regressions have explicit gates.
- Approval result: no blockers remain after tightening. Plan status is
  approved.

## Execution Evidence

Discovery and approval evidence already recorded:

- Source/docs inspected: repository README, HOWTO, development-plan registry,
  Brain Service ICD, Message Envelope ICD, RAG README, parent multi-source
  cognition plan, Stage 09 multimodal plan, and relevant source/test files.
- Current code supports typed image intake and descriptor generation, but still
  smears image descriptions into `user_input`, lacks reply attachment
  projection, and rejects multimodal consolidation origins.
- Live evidence artifact:
  `tests/test_image_cognition_options_live_llm.py`.
- Determining live text+image traces:
  `test_artifacts/llm_traces/image_cognition_options_live_llm__desk_study_support_text_image_layer_quality.json`
  and
  `test_artifacts/llm_traces/image_cognition_options_live_llm__dessert_shop_correction_text_image_layer_quality.json`.
- Evidence conclusion: descriptor-mediated cognition and direct raw-image
  cognition tied on support/correction precision. Descriptor-mediated
  structured image observations are selected because they satisfy
  description-only storage, quoted-image replay, raw-media containment, and weak
  local-LLM compatibility.
- Plan tightening pass on 2026-05-10 set the plan class to `large` and removed
  unselected decision-process detail. Status remained `approved` until
  implementation completion.
- Prompt-rule amendment on 2026-05-10 added planning-phase LLM budget analysis,
  coherent prompt refactor, and real LLM fix/regression validation gates.

Implementation completion evidence recorded on 2026-05-10:

- Implemented structured descriptor-mediated image observations while keeping
  durable storage description-only.
- Removed image-description smearing into `user_input`; image descriptors are
  prompt-visible only as typed visual evidence.
- Added quoted/replied image hydration from stored attachment summaries and
  explicit unavailable visual context when no description is available.
- Kept RAG text/query input free of image observations and preserved
  metadata-only consolidation origin behavior for multimodal user-message
  episodes.
- Refactored descriptor and cognition prompt contracts to include explicit
  image-observation input handling, generation guidance, and output contracts.
- Live LLM evidence:
  - `tests\test_image_cognition_options_live_llm.py::test_live_compare_option_b_and_d_text_image_desk_study_support -q -s -m live_llm`
    passed. Descriptor-mediated cognition and direct raw-image cognition tied
    on alignment and evidence use; no descriptor-as-user-text drift observed.
  - `tests\test_cognition_live_llm_prompt_contracts.py::test_live_cognition_stack_photo_request_chinese -q -s -m live_llm`
    passed structurally. Residual broader risk recorded: absent-image photo
    wording can still invite visual hallucination in existing prompts.
- Independent code review result:
  - Initial approval withheld for three findings: quoted-reply attachment
    budget reduction, missing multimodal promoted reflection context, and
    missing descriptor prompt input/procedure sections.
  - All three findings were fixed.
  - Added regression tests for each finding.
  - Approval status after remediation: approved.
- Final deterministic verification:
  - `venv\Scripts\python -m pytest tests\test_prompt_message_context.py::test_projection_reduces_reply_attachments_within_budget tests\test_relevance_agent.py::test_vision_descriptor_prompt_declares_structured_prompt_sections tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py::test_l2a_multimodal_user_turn_keeps_promoted_reflection_context -q`
    passed: 3 passed.
  - `venv\Scripts\python -m pytest tests\test_prompt_message_context.py tests\test_relevance_agent.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py -q`
    passed: 52 passed.
  - `venv\Scripts\python -m pytest tests\test_multi_source_cognition_image_input.py -q`
    passed: 4 passed.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\message_envelope\prompt_projection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\relevance_agent.py tests\test_prompt_message_context.py tests\test_relevance_agent.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py`
    passed.
  - Descriptor prompt render check passed:
    `_VISION_DESCRIPTOR_PROMPT.format(max_description_chars=800)`.
  - `git diff --check` passed with only existing CRLF warnings.
- Completion status: completed and archived.
