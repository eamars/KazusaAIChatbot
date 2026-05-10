# first class image input cognition plan

## Summary

- Goal: Make image attachments a first-class cognition input while preserving
  local-LLM reliability, bounded response latency, adapter neutrality, and raw
  media safety.
- Plan class: large
- Status: draft
- Mandatory skills for implementation: `development-plan-writing`,
  `local-llm-architecture`, `no-prepost-user-input`, `py-style`,
  `test-style-and-execution`, and `cjk-safety` before editing cognition Python
  files that contain CJK prompt text.
- Overall cutover strategy: compatible for text-only `/chat`; compatible for
  existing image-description behavior; bigbang removal of image-description
  smearing into `user_input`, RAG query text, and untyped cognition payloads.
- Highest-risk areas: weak local LLM ignoring image context, accidental RAG
  query drift, raw `base64_data` prompt leakage, image-only turn regression,
  prompt byte drift without tests, and background consolidation failure on
  multimodal user-message episodes.
- Acceptance criteria: image observations enter cognition through bounded
  structured source payloads; raw media remains outside cognition/RAG/dialog
  prompts; RAG uses dialog/query intent only; text-only behavior remains
  unchanged; image-only turns can be reasoned over by cognition without text
  smearing; multimodal user-message consolidation origin is metadata-only and
  no longer fails solely because image observations are present.

This draft is not approved for implementation yet. It records current-code
findings, architecture alternatives, the recommended direction, dependencies,
and staged development gates for a later implementation pass.

## Context

The completed multi-source cognition architecture already introduced
`CognitiveEpisode` and the `image_observation` input-source label. Current code
can accept inline image attachments and produce image descriptions before
relevance. However, image input is still treated as an auxiliary description
string in several places instead of as a consistently modeled cognition input.

Relevant completed artifacts:

- `development_plans/archive/completed/short_term/multi_source_cognition_architecture_plan.md`
- `development_plans/archive/completed/short_term/multi_source_cognition_architecture_stage_09_multimodal_cognitive_input_sources_plan.md`
- `development_plans/archive/completed/short_term/multi_source_cognition_architecture_stage_10_permissioned_proactive_output_plan.md`

Relevant current source boundaries:

- Adapter envelope input: `src/kazusa_ai_chatbot/message_envelope/*`
- Brain service queue and graph seed state: `src/kazusa_ai_chatbot/service.py`
- Root chat graph routing:
  `src/kazusa_ai_chatbot/brain_service/graph.py`
- Image descriptor and relevance:
  `src/kazusa_ai_chatbot/nodes/relevance_agent.py`
- Episode contract: `src/kazusa_ai_chatbot/cognition_episode.py`
- RAG projection:
  `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
- Cognition prompt selection:
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
- L1/L2/L3 cognition prompt handlers:
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- Consolidation origin:
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`

## Current Code Deep Dive

### Adapter And Envelope Boundary

- `MessageEnvelope.attachments` supports `media_type`, `url`, `base64_data`,
  `description`, `size_bytes`, and `storage_shape`.
- The default attachment handler registry currently registers only `image/`.
- Discord and NapCat QQ adapters fetch image URLs, base64-encode the image, and
  pass image attachments into the envelope normalizer with empty descriptions.
- `ImageAttachmentHandler` chooses `inline`, `url_only`, or `drop`, but the
  current adapter payloads usually omit URL/size when forwarding fetched images,
  so image bytes can remain inline in stored envelope rows. Prompt projections
  still exclude raw bytes.

### Service Intake And Graph Routing

- `_process_queued_chat_item(...)` gathers `user_multimedia_input` from the
  surviving queued item plus collapsed items.
- Image rows are included when `media_type` starts with `image/` and either
  `base64_data` or `description` exists.
- Audio rows are included only when an `audio/` attachment already has a
  description.
- `build_text_chat_cognitive_episode(...)` receives sanitized
  `media_description_rows`, so pre-described image rows can enter the initial
  episode.
- `brain_service.graph.build_graph(...)` routes to
  `multimedia_descriptor_agent` before relevance when `user_multimedia_input`
  is non-empty. `listen_only` skips the graph.

### Image Descriptor

- `multimedia_descriptor_agent(...)` is the only current node allowed to send
  raw base64 image data to a vision-capable model.
- The descriptor prompt returns a single JSON field: `description`.
- Images without `base64_data` and with preexisting descriptions skip the
  vision model and pass the description through.
- Descriptor output refreshes `user_multimedia_input`,
  `prompt_message_context`, and the `cognitive_episode`.
- Descriptor failures degrade to empty descriptions and unavailable attachment
  summaries instead of failing the turn.

### Prompt Message Context

- `project_prompt_message_context(...)` builds a bounded LLM-safe current
  message context.
- It caps current attachments at four and attachment descriptions at 800
  characters.
- It excludes binary/wire fields and allows only
  `media_kind`, `description`, and `summary_status` per attachment.
- For image attachments, it prefers generated `user_multimedia_input`
  descriptions over stored envelope descriptions.

### Cognitive Episode And Cognition Prompt Payload

- `CognitiveEpisode` supports `input_sources` including `dialog_text`,
  `image_observation`, and `audio_observation`.
- `build_text_chat_media_description_rows(...)` projects only
  `content_type` and `description` into prompt-safe rows.
- `_build_media_percepts(...)` caps media percepts at four and trims each
  description to 800 characters.
- `build_cognition_prompt_source_payload(...)` emits:

```python
{
    "media_observations": {
        "image_observations": list[str],
        "audio_observations": list[str],
    },
}
```

- L1/L2/L3 prompt maps accept multimodal variants, but they currently reuse
  the same prompt text as text-only variants.
- Most cognition prompt input-format sections do not explicitly teach the
  model how to use `media_observations`. The L3 visual-agent prompt references
  `prompt_message_context.attachments`, but not the newer structured source
  payload as a first-class contract.

### RAG And Decontextualization

- The RAG episode adapter accepts multimodal user-message input-source
  profiles, but it does not project media observations into the RAG request.
- Current tests assert that `image_observation`, `audio_observation`, and
  media descriptions do not appear in RAG request JSON.
- Current runtime has a conflict with that intent:
  `relevance_agent(...)` appends `Image attachment: <description>` to
  `state["user_input"]`, returns the mutated `user_input`, and the persona
  graph then sends that text to the decontextualizer. Because RAG uses
  `decontexualized_input`, image descriptions can still influence the RAG
  query through text smearing.

### Consolidation And Persistence

- User conversation rows persist typed attachment metadata and may include
  inline `base64_data` depending on adapter and attachment handler input.
- Background consolidation origin currently requires
  `episode["input_sources"] == ["dialog_text"]`.
- A visible response to an image-observation turn can therefore return to the
  user, then fail background consolidation solely because the cognitive episode
  contains `image_observation`.
- Current origin metadata intentionally excludes percept content, attachment
  content, prompt payload, facts, and promises. That metadata-only pattern is
  the right boundary to preserve.

## Problem Statement

The current image path is safe enough to avoid raw media prompt leakage, but it
is not yet a clean first-class cognition contract:

- image facts can be duplicated across `prompt_message_context`,
  `media_observations`, and appended `user_input`;
- local LLM prompts do not consistently explain how to use image observations;
- RAG can accidentally receive image descriptions through mutated text;
- image-only turns rely on text smearing for downstream stages;
- consolidation rejects multimodal user-message origins even when only
  metadata would be persisted;
- the descriptor emits one free-form paragraph, which is harder for weak local
  LLMs to use than compact typed visual facts.

## Brainstormed Approaches

### Approach A - Keep Descriptions, Fix Contract Discipline

Use the existing single `description` output, remove `user_input` smearing,
teach prompts about `media_observations`, keep RAG dialog-only, and allow
metadata-only consolidation origin for multimodal user-message episodes.

Pros:

- Lowest implementation risk.
- Keeps the normal image path at the existing one vision call per image.
- Preserves the Stage 09 raw-media safety design.
- Good enough for simple "what is in this image" turns.

Cons:

- Precision stays limited by one free-form paragraph.
- Weak local cognition models may miss OCR, spatial relationships, or
  uncertainty when buried in prose.
- Harder to test whether key visual facts reached the correct cognition layer.

### Approach B - Structured Image Observation Contract

Keep the existing descriptor call, but change its JSON output and downstream
episode/prompt projection to carry compact structured image observations:
summary, visible text, salient entities, scene/spatial facts, uncertainty, and
image-question focus. Do not send raw image bytes past the descriptor.

Pros:

- Best balance of precision and efficiency.
- Gives weak local LLMs labeled semantic fields instead of asking them to
  rediscover structure from a paragraph.
- Keeps latency bounded by reusing the existing descriptor slot.
- Enables focused deterministic tests for prompt payload shape and raw-media
  exclusion.

Cons:

- Requires prompt changes in the vision descriptor and L1/L2/L3 cognition
  input contracts.
- Requires schema validation and fallback handling for partially malformed
  descriptor JSON.
- Some legacy tests with prompt fingerprints will need intentional updates.

### Approach C - On-Demand Image Specialist

Introduce an image specialist capability that can answer targeted visual
questions after relevance or cognition decides the image matters. The core
cognition stages receive only the specialist's bounded answer.

Pros:

- Highest precision for specific visual questions.
- Can defer image work when the user attaches irrelevant media.
- Keeps cognition layers clean if the specialist owns low-level image details.

Cons:

- Adds response-path routing complexity and extra LLM calls.
- Weak local LLMs may route poorly unless the specialist contract is narrow.
- Harder to keep latency predictable in group chats and collapsed turns.
- Not needed before the basic first-class image input contract is reliable.

### Approach D - Raw Multimodal Cognition

Pass raw image payloads directly to cognition prompts and let the cognition
model reason over text and images together.

Pros:

- Conceptually simple for a strong frontier multimodal model.
- Avoids a separate descriptor schema.

Cons:

- Rejected for this system. It violates the current raw-media boundary, raises
  context and privacy risk, depends on strong multimodal cognition models, and
  is brittle for local/weak LLM deployments.

## Recommended Direction

Use Approach B as the primary implementation direction, with Approach A's
contract cleanup as Stage 1. Defer Approach C until the structured observation
contract has live evidence, and reject Approach D.

The target runtime shape is:

```text
adapter image attachment
-> typed MessageEnvelope attachment
-> service user_multimedia_input
-> existing multimedia_descriptor_agent
   -> structured ImageObservation JSON
-> CognitiveEpisode percept:
   input_source="image_observation"
   content=<short summary>
   metadata.image_observation=<bounded structured facts>
-> cognition source payload:
   image_observations=[structured prompt-safe facts]
-> L1/L2/L3 cognition prompts explicitly reason over image facts
-> dialog replies normally
-> consolidation origin stores metadata only, not image content
```

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

- Implementation must not start until this draft is reviewed and approved.
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
- The descriptor remains the only node allowed to send raw image bytes to a
  vision model.
- Do not place `base64_data`, raw bytes, data URIs, attachment URLs, or
  unbounded attachment metadata in cognition episodes, cognition source
  payloads, RAG requests, dialog payloads, consolidator prompt payloads, or new
  logs.
- Do not mutate `state["user_input"]` to append image descriptions.
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
- Text-only `/chat` output, prompt payloads, and RAG request behavior must
  remain unchanged except where tests explicitly prove no-regression.

## Must Do

- Add failing tests that capture the current image-path contract gaps:
  `user_input` smearing, accidental RAG query contamination, prompt payload
  ambiguity, and consolidation-origin rejection.
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
- Stop appending image descriptions into `state["user_input"]`.
- Keep RAG request construction free of image descriptions unless a later plan
  explicitly adds a visual-RAG contract.
- Allow user-message multimodal input-source profiles in consolidation origin
  metadata without copying percept content into consolidation metadata.
- Add focused tests and static greps proving raw media is absent from every
  prompt-facing payload touched by this plan.

## Deferred

- Raw image payloads inside cognition, RAG, dialog, or consolidator prompts.
- Visual RAG, web search from image labels, image identity lookup, reverse
  image search, or OCR-driven retrieval.
- A separate image specialist agent or planner-routed visual tool.
- Video support.
- Audio changes beyond preserving current pass-through behavior.
- Outbound image generation or image attachments.
- Adapter file-storage redesign, CDN storage, or database migration for
  existing inline image payloads.
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

Text-only turns still produce:

```python
{
    "trigger_source": "user_message",
    "input_sources": ["dialog_text"],
    "percepts": [
        {"input_source": "dialog_text", "content": "<body text>"}
    ],
}
```

Image turns produce:

```python
{
    "trigger_source": "user_message",
    "input_sources": ["dialog_text", "image_observation"],
    "percepts": [
        {"input_source": "dialog_text", "content": "<body text or empty>"},
        {
            "input_source": "image_observation",
            "content": "<short image summary>",
            "metadata": {
                "media_index": 1,
                "media_kind": "image",
                "image_observation": {
                    "summary": "...",
                    "visible_text": ["..."],
                    "salient_visual_facts": ["..."],
                    "spatial_or_scene_facts": ["..."],
                    "uncertainty": ["..."]
                }
            }
        }
    ],
}
```

The cognition source payload becomes:

```python
{
    "media_observations": {
        "image_observations": [
            {
                "summary": "...",
                "visible_text": ["..."],
                "salient_visual_facts": ["..."],
                "spatial_or_scene_facts": ["..."],
                "uncertainty": ["..."]
            }
        ],
        "audio_observations": ["existing bounded audio description"]
    }
}
```

The exact field names may be refined during approval, but the final approved
plan must settle them before implementation begins.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Image semantics owner | Existing descriptor LLM produces visual semantics. | Keeps adapters thin and avoids raw media in core cognition. |
| Descriptor call count | Reuse the existing descriptor call; no extra response-path image LLM call. | Balances precision and latency for local deployments. |
| Prompt payload | Use structured image observations with compact lists. | Weak local LLMs follow labeled fields better than long prose. |
| RAG | Keep image observations out of RAG query material. | Prevents retrieval drift and preserves RAG evidence ownership. |
| Cognition prompts | Update input formats and generation procedures explicitly. | Hidden JSON fields are unreliable with local LLMs. |
| Consolidation | Admit multimodal user-message origin metadata only. | Prevents post-response failure without letting image content leak into write metadata. |
| Image-only turns | Support empty body text plus image observation for cognition. | Current envelope allows attachment-only messages; cognition should not require text smearing. |
| Raw media | Descriptor-only, current-turn only. | Preserves privacy and prompt safety boundaries. |

## Dependencies

- The multi-source cognition architecture must remain completed and current.
- Route-specific `VISION_DESCRIPTOR_LLM_*` settings must point to a model that
  accepts OpenAI-compatible image payloads for live image tests.
- The implementation branch must start from a clean mainline that includes the
  completed Stage 09 and Stage 10 artifacts.
- Before implementation, this draft must be converted to an approved
  executable plan with settled field names and exact verification gates.

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
  same maximum image observations admitted to cognition unless approval chooses
  a lower value.
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

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_episode.py`
  - Add or refine structured image observation TypedDicts, media percept
    metadata projection, caps, and validation.
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
  - `tests/test_service_input_queue.py`
  - `tests/test_rag_cognitive_episode_adapter.py`
  - `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`
  - `tests/test_consolidation_origin_metadata.py`

### Keep

- `src/adapters/*`: no adapter behavior change in the first implementation
  pass unless tests prove a fixture-only adjustment is required.
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
   - Prove multimodal user-message consolidation origin currently fails.
   - Prove cognition source payload currently carries only strings.
2. Add structured image observation contract tests.
   - Test sanitized projection, caps, fallback from plain description, and raw
     media exclusion.
3. Implement structured image observation projection in
   `cognition_episode.py`.
4. Update descriptor prompt and parser in `relevance_agent.py`.
   - Keep one descriptor call per processed image.
   - Validate malformed structured fields and degrade to summary-only
     observation.
5. Remove `user_input` image smearing.
   - Preserve prompt context and cognitive episode image observations.
6. Update cognition source payload and L1/L2/L3 prompt input contracts.
   - Run prompt-render checks, not only syntax checks.
7. Harden RAG projection tests.
   - Ensure `original_query` is dialog/query text only.
8. Update consolidation origin metadata to accept approved multimodal
   user-message profiles without content leakage.
9. Run focused and regression verification.
10. Run independent code review, remediate in-scope findings, rerun affected
    verification, and record evidence.

## Progress Checklist

- [ ] Stage 1 - current behavior captured.
  - Covers: Implementation Order steps 1-2.
  - Verify: focused tests fail for the current contract gaps or record current
    baseline where no failure is expected.
  - Evidence: record exact failure messages in `Execution Evidence`.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 - image observation contract implemented.
  - Covers: Implementation Order steps 3-4.
  - Verify: structured observation tests pass; raw-media greps pass.
  - Evidence: record changed files and focused test output.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 3 - cognition prompt payload and prompt contracts updated.
  - Covers: Implementation Order steps 5-6.
  - Verify: prompt selection/source payload tests, prompt render checks, and
    affected cognition prompt tests pass.
  - Evidence: record prompt-render command output and test output.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 4 - RAG and consolidation boundaries corrected.
  - Covers: Implementation Order steps 7-8.
  - Verify: RAG projection tests and consolidation origin tests pass.
  - Evidence: record test output and static grep output.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 5 - full verification complete.
  - Covers: Implementation Order step 9.
  - Verify: every command in `Verification` passes or has an explicitly
    accepted no-match exit.
  - Evidence: record all command outputs in `Execution Evidence`.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 6 - independent code review complete.
  - Covers: Implementation Order step 10.
  - Verify: review findings are resolved or recorded as residual risks, and
    affected verification is rerun.
  - Evidence: record review scope, findings, fixes, reruns, residual risks,
    and approval status.
  - Sign-off: `<agent/date>` after review approval.

## Verification

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_episode.py src\kazusa_ai_chatbot\nodes\relevance_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_origin.py tests\test_multi_source_cognition_image_input.py`

### Static Greps

- `rg -n "\"base64_data\"|\"image_url\"|\"data:" src\kazusa_ai_chatbot\cognition_episode.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py`
  - Expected: no matches, with `rg` exit code 1 accepted.
- `rg -n "Image attachment:" src\kazusa_ai_chatbot`
  - Expected: no runtime prompt-building matches after implementation.
- `rg -n "image_observation|media_observations" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_images.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_persistence.py`
  - Expected: no matches unless a match is explicitly allowed for metadata-only
    origin handling in the final approved plan.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_image_input.py -q`
- `venv\Scripts\python -m pytest tests\test_relevance_agent.py tests\test_rag_cognitive_episode_adapter.py tests\test_consolidation_origin_metadata.py -q`

### Regression Tests

- `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py -q`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py -q`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py -q`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py -q`

### Live LLM Smoke

Run only after deterministic tests pass and only when the configured local
vision route is available:

- one image-description live test, one case at a time with `-q -s`;
- one cognition live test where the user's text refers to a visible image fact;
- inspect traces manually before recording success.

## Independent Plan Review

Run before changing status from `draft` to `approved`.

Review scope:

- Confirm that the plan aligns with the completed multi-source cognition
  architecture and Stage 09 raw-media boundary.
- Confirm that structured image observations are necessary before adding any
  image specialist, visual RAG, or retrieval loop.
- Confirm that field names, caps, tests, and prompt sections are fully settled.
- Confirm that RAG remains dialog/query-intent only.
- Confirm that consolidation origin admits metadata only and cannot write image
  content by accident.

Record blockers, non-blocking findings, required edits, open questions, and
approval status in this plan before implementation.

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
- Prompt contract clarity for weak local LLMs: input format, output format,
  and generation procedure must match actual handler payloads.
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
- image attachments enter cognition as structured, bounded image observations;
- image-only messages can reach cognition without appending image descriptions
  into `user_input`;
- L1/L2/L3 prompts explicitly document how to use image observations;
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
| RAG query contamination | Remove `user_input` smearing and test `original_query`. | RAG projection tests and `Image attachment:` grep. |
| Raw media leakage | Keep descriptor-only raw image path and static greps. | Raw-media greps and source payload tests. |
| Descriptor schema brittleness | Validate fields and degrade to summary-only observation. | Malformed descriptor tests. |
| Consolidation failure after image reply | Admit metadata-only multimodal origins. | Consolidation origin tests. |
| Prompt drift across CJK modules | Use `cjk-safety`, compile checks, and prompt render tests. | Compile and focused prompt tests. |

## Plan Self-Review

Draft self-review on 2026-05-10:

- Coverage: current-code findings, alternatives, recommended approach,
  dependencies, staged execution, verification, and review gates are present.
- Placeholder scan: this draft intentionally leaves final field-name approval
  open; that must be resolved before status changes to `approved`.
- Contract consistency: current field names match inspected source; proposed
  structured observation fields need final approval.
- Granularity: implementation stages separate tests, contract, descriptor,
  cognition prompts, RAG/consolidation, verification, and review.
- Verification: raw-media safety, RAG drift, prompt behavior, consolidation
  origin, and prior-stage regressions have explicit gates.

## Execution Evidence

Record after implementation begins. Current draft discovery evidence:

- `git status --short` was clean before drafting.
- Relevant docs read: `README.md`, `docs/HOWTO.md`,
  `development_plans/README.md`, Brain Service ICD, Message Envelope ICD,
  RAG README, Stage 09 multimodal plan, and parent multi-source cognition plan.
- Source inspection found existing image intake, descriptor, episode, prompt
  selection, RAG adapter, and consolidation origin boundaries described above.

