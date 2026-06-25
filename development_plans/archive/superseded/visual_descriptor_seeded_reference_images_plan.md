# visual descriptor seeded reference images plan

## Summary

- Goal: Add character-profile-owned seeded visual reference images to the
  visual descriptor so it can use official visual evidence while avoiding
  broad-description false positives.
- Plan class: large
- Status: superseded
- Mandatory skills: `py-style`, `test-style-and-execution`,
  `local-llm-architecture`, `debug-llm`, `development-plan`.
- Overall cutover strategy: compatible
- Highest-risk areas: reference/target image confusion, false positive
  character naming, media cache contamination, prompt overfitting, base64 bloat
  in `character_state`, and live LLM variance.
- Acceptance criteria: seeded visual references are loaded from character data,
  appended to the descriptor call as labelled user-role images, target images
  remain the only described images, negative target cases do not get named as
  the active character, and deterministic plus real LLM evidence is recorded.

## Context

The current visual descriptor lives in
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`.
Its prompt says the multimodal `HumanMessage` contains only one `image_url`,
and the implementation sends exactly one target image. The output becomes the
stored attachment `description` and the current-turn `image_observation`.

The observed failure mode is not that Kazusa cannot ever recognize herself.
The failure mode is that broad character description or active-character
context can overfit: an image with overlapping traits can be described as
Kazusa even when the evidence is weak. Only the visual descriptor sees image
pixels; downstream cognition sees only descriptor text and cannot correct a
visual identity claim with direct pixel evidence.

The user prefers seeded images as the final direction. This plan accepts that
direction, but keeps ownership narrow:

- The visual descriptor may use seeded media as visual field-guide evidence.
- The descriptor must still describe the current user image only.
- Cognition must not receive raw seeded media or decide visual identity from
  loose character-state words.
- The character can be any character; all identity seed data must come from the
  loaded character document, not hard-coded Kazusa prompt text.

Closure decision, 2026-06-26: later real-LLM tests showed that seeded images
are not reliable for the current local OpenAI-compatible vision endpoint. The
model can structurally pass the descriptor schema while describing the seeded
reference image instead of the current target image. This violates the visual
descriptor ownership boundary. This plan is therefore closed as superseded and
must not be used as an executable production path.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing visual descriptor prompt,
  state shape, LLM call payloads, cache keys, or cognition handoff behavior.
- `debug-llm`: load before running or reviewing live LLM visual descriptor
  tests.
- `development-plan`: load before moving this plan through approval,
  execution, or completion.

## Mandatory Rules

- The user explicitly approved fallback execution without subagents for
  stages 1 through 3 on 2026-06-25.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.
- The plan's `Execution Model` uses parent-led native subagent execution unless
  the user explicitly approves fallback execution.
- Seeded reference images must be sent in `HumanMessage` content, not
  `SystemMessage` or `AIMessage`, for OpenAI-compatible vision portability.
- Seeded reference images used in `visual_seed_materials` must be official or
  online downloaded references with recorded source URLs. AI-generated images
  may be used only as target-side test variants and must never be seed
  references.
- The descriptor output schema remains the existing JSON shape:
  `description`, `visible_text`, `salient_visual_facts`,
  `spatial_or_scene_facts`, and `uncertainty`.
- The descriptor must not add role, speaker, active-character identity, user
  intent, or cognition stance to image evidence.
- The media descriptor cache key must include the seeded reference fingerprint
  and descriptor prompt version. A target-image-only cache key is forbidden
  after this change.
- Raw seed image base64 and target image base64 must not be written to
  `event_log_events`, LLM trace rows, human-readable review artifacts, or
  Markdown reports. Test and review artifacts may record local file paths,
  source URLs, hashes, parsed descriptor output, and raw text model output.
- Real LLM tests must run one case at a time with raw output inspected and a
  human-readable review artifact written by the agent.
- Test images for generated variants must be non-sexual and safe for a
  school-age character profile.

## Must Do

- Add a character-data field for seeded visual media under the singleton
  `character_state` document, initially populated through
  `personalities/kazusa.json`.
- Normalize and validate seeded visual media from `state["character_profile"]`
  before the descriptor call.
- Cache the normalized reference set for the Python process or current
  character profile snapshot, with a stable fingerprint.
- Build descriptor messages with labelled reference images first and the target
  image last, using separate user-role message boundaries for reference and
  target payloads.
- Limit each descriptor call to at most three seed reference images. The stable
  inclusion order is `identity_reference` rows first, then other usages, then
  lexical `media_id` order inside each usage group.
- Modify `_VISION_DESCRIPTOR_PROMPT` so the model understands reference images
  are field-guide material and the target image is the only image to describe.
- Preserve the existing descriptor output schema and downstream
  `image_observation` shape.
- Include the reference fingerprint in Cache2 and persistent media descriptor
  cache keys.
- Create a real LLM visual descriptor test corpus with:
  - three official or online downloaded Kazusa seed reference images;
  - at least five Kazusa target images not used as seed references;
  - at least five non-Kazusa target images across hard negative visual buckets.
- Use false-positive prevention as the hard live LLM gate: non-Kazusa target
  outputs must not name the active character or its aliases. Recognizing Kazusa
  in positive cases is useful evidence but not required for pass.
- Add deterministic tests after the real LLM stage for seed normalization,
  message construction, prompt contract, cache key variation, and existing
  unseeded behavior.
- Add a prompt-rendering deterministic test so broken `.format(...)`
  placeholders or literal JSON braces fail before runtime.

## Deferred

- Do not add a separate visual-identity LLM stage in this plan.
- Do not send seeded media to cognition, dialog, RAG, reflection, or
  consolidation prompts.
- Do not add a new MongoDB collection for media seeds.
- Do not add audio, video, embeddings, perceptual hashes, CLIP matching, or
  vector search for visual identity.
- Do not add provider-specific native image APIs outside the existing
  `LLInterface` OpenAI-compatible call path.
- Do not change relevance, cognition, dialog, or consolidation behavior except
  for consuming the descriptor's existing current-image observation.
- Do not hard-code Kazusa, Blue Archive, or any character name in reusable
  runtime prompt contracts.

## Cutover Policy

Overall strategy: compatible

| Area | Policy | Instruction |
|---|---|---|
| Character data | compatible | Add `visual_seed_materials` as an optional profile field. Profiles without it keep current target-only descriptor behavior. |
| Descriptor prompt | bigbang | Replace the one-image input contract with a labelled reference-plus-target contract. Do not keep stale wording that says the content array has only one image. |
| Descriptor output | compatible | Preserve the current output JSON fields and downstream observation shape. |
| Media cache | bigbang | Update cache key inputs so seeded and unseeded descriptor calls cannot share target-only cache rows. |
| Tests | compatible | Keep existing media descriptor tests and add seeded-reference tests beside them. |

## Target State

A character profile can contain visual seed media. For Kazusa, the initial
profile carries official or online-downloaded seed images keyed by named visual
entities such as `杏山千纱` and entity parts such as `杏山千纱的武器`.

When a user sends an image, the visual descriptor receives:

1. A static `SystemMessage` explaining its objective descriptor role, the
   seeded-reference protocol, and the unchanged output schema.
2. A `HumanMessage` containing labelled seeded reference images from the active
   character data.
3. A final `HumanMessage` containing the current target image to describe.

The descriptor may name a public visual entity only when the target image
itself provides distinctive evidence. It must not describe the reference
images, merge reference and target images into one scene, or infer identity
from the active character's role in chat.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Data owner | Store seeds as optional static profile data in `character_state` via `personalities/*.json`. | The loaded character document already identifies the active character and is generic across characters. |
| Runtime field name | Use `visual_seed_materials`. | The name is role-neutral and not tied to Kazusa or self-recognition. |
| Seed storage | Store base64 image data plus metadata in the profile document. | This matches the user's preferred initial deployment and avoids runtime filesystem dependency. |
| Runtime cache | Normalize seeds from `character_profile` and cache the resulting list plus fingerprint in process memory. | Avoids repeated validation and repeated fingerprint work on every image. |
| Seed inclusion cap | Include at most three valid seed references per descriptor call. | Keeps live-path latency and vision context bounded while matching the requested three official Kazusa seeds. |
| Seed ordering | Sort by usage priority, then label, then `media_id`. | Makes fingerprints and prompt payloads deterministic across Python dictionary order. |
| Message roles | Put all images in user-role messages. | This matches OpenAI-compatible vision behavior and avoids non-portable system-image payloads. |
| Message separation | Use one reference `HumanMessage` and one target `HumanMessage`. | This gives clearer reference/target boundaries than a single content array while staying provider-compatible. |
| Descriptor output | Keep the existing schema. | Downstream cognition and persistence already consume this shape. |
| Cache key | Add `reference_set_fingerprint` to the media descriptor cache key. | The same target image can produce different valid descriptors depending on seeded evidence. |
| Pass criterion | Fail hard only on non-Kazusa false positives. | The user explicitly does not require Kazusa recognition for the pass. |
| Alias gate | Build forbidden false-positive aliases from `character_profile.name`, seed mapping labels, and optional seed-row `display_names`. | The gate stays character-generic and data-owned. |

## Contracts And Data Shapes

### Character Profile Field

`visual_seed_materials` is an optional top-level field in the character profile:

```json
{
  "visual_seed_materials": {
    "杏山千纱": [
      {
        "media_id": "kazusa_identity_official_01",
        "media_type": "image/png",
        "base64_data": "<base64>",
        "display_names": ["杏山千纱", "Kyouyama Kazusa", "Kazusa"],
        "source_label": "official downloaded reference",
        "source_url": "https://...",
        "usage": "identity_reference",
        "notes": "front-facing official character art"
      }
    ],
    "杏山千纱的武器": [
      {
        "media_id": "kazusa_weapon_official_01",
        "media_type": "image/png",
        "base64_data": "<base64>",
        "source_label": "official downloaded reference",
        "source_url": "https://...",
        "usage": "object_reference",
        "notes": "weapon reference"
      }
    ]
  }
}
```

Validation rules:

- `visual_seed_materials` must be a mapping from display label to a list of
  media rows.
- Each row must have non-empty `media_id`, `media_type`, and `base64_data`.
- `display_names` is optional and, when present, must be a list of non-empty
  strings used only for reference labelling and false-positive test gates.
- Only `image/png`, `image/jpeg`, and `image/webp` are accepted in this plan.
- Each decoded seed row must be no larger than 2 MiB. Larger rows are dropped.
- At most three valid seed rows enter one descriptor call. Extra valid rows
  remain profile data but are not sent to the model until a later plan expands
  the selection contract.
- Empty, malformed, oversized, or unsupported seed rows are dropped with a
  warning; the descriptor falls back to target-only if no valid rows remain.
- Runtime prompt labels use the mapping key and `usage`; they do not infer
  active-character identity.

### Normalized Runtime Shape

The normalizer returns:

```python
{
    "fingerprint": str,
    "references": [
        {
            "label": str,
            "media_id": str,
            "media_type": str,
            "data_uri": str,
            "usage": str,
            "source_label": str,
            "notes": str,
            "display_names": list[str],
        }
    ],
    "forbidden_aliases": list[str],
}
```

The fingerprint is computed from stable row metadata and decoded image bytes,
not from insertion order. It changes when reference content, labels, usage, or
display names change.

### Descriptor Message Contract

Seeded call:

```python
[
    SystemMessage(content=rendered_vision_descriptor_prompt),
    HumanMessage(content=[
        {"type": "text", "text": "Seeded visual reference materials..."},
        {"type": "text", "text": "Reference: 杏山千纱 / usage=identity_reference / media_id=..."},
        {"type": "image_url", "image_url": {"url": reference_data_uri}},
    ]),
    HumanMessage(content=[
        {"type": "text", "text": "Target user image to describe. Describe only this target image."},
        {"type": "image_url", "image_url": {"url": target_data_uri}},
    ]),
]
```

Unseeded call:

```python
[
    SystemMessage(content=rendered_vision_descriptor_prompt),
    HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": target_data_uri}},
    ]),
]
```

### Cache Key Contract

`build_media_descriptor_cache_key` must accept:

```python
build_media_descriptor_cache_key(
    content_type=...,
    content_hash=...,
    reference_set_fingerprint=...,
)
```

The unseeded fingerprint is the empty string or a fixed sentinel; the key must
vary between seeded and unseeded calls.

## LLM Call And Context Budget

Before this plan, each uncached target image costs one `VISION_DESCRIPTOR_LLM`
call with one target image.

After this plan, each uncached target image still costs one
`VISION_DESCRIPTOR_LLM` call. The call includes up to three seed reference
images plus one target image.

Response-path call count does not increase. Context and vision-token cost
increase only for uncached image descriptors. The implementation must cap seed
rows per descriptor call at three, cap decoded seed bytes per row at 2 MiB, and
drop invalid seed rows before constructing messages. The normal live path
remains target-only when a character has no seed materials.

## Change Surface

### Modify

- `personalities/kazusa.json`: add initial `visual_seed_materials` with three
  official or online-downloaded Kazusa references encoded as base64.
- `src/kazusa_ai_chatbot/db/schemas.py`: document the optional profile field
  shape.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`:
  normalize seed references, build labelled reference and target messages,
  update `_VISION_DESCRIPTOR_PROMPT`, and preserve output parsing.
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`: add reference fingerprint to
  media descriptor cache key construction and version expectations.
- `src/kazusa_ai_chatbot/db/rag_cache2_persistent.py`: ensure persistent media
  descriptor versioning remains aligned with the updated cache policy.
- `tests/test_msg_decontexualizer.py`: cover seeded message construction and
  target-only fallback.
- `tests/test_media_descriptor_cache.py`: cover cache key variation by
  reference fingerprint.
- `tests/test_visual_identity_reference_probe_live_llm.py`: replace the
  exploratory probe with the approved seeded-reference corpus runner.
- `docs/HOWTO.md`: document the optional profile field and live LLM test
  command.
- `development_plans/README.md`: track this plan.

### Create

- `src/kazusa_ai_chatbot/visual_seed_materials.py`: validation,
  normalization, data URI creation, and fingerprinting for character-profile
  visual seed media.
- `test_artifacts/diagnostics/visual_seed_reference_corpus/`: local generated
  or downloaded corpus output during execution.
- `test_artifacts/diagnostics/visual_seed_reference_reviews/`: agent-authored
  live LLM review artifacts.

### Keep

- Current descriptor output schema and `image_observation` fields.
- Current single-call media descriptor location before relevance.
- Current cognition, dialog, consolidation, and adapter contracts.

## Overdesign Guardrail

- Actual problem: the visual descriptor can overfit broad character description
  or active-character context and produce false positive self-identity claims.
- Minimal change: provide labelled seeded visual evidence to the descriptor,
  keep the target image as the only described image, and harden cache keys and
  tests against false positives.
- Ownership boundaries: character profile owns seed data; visual descriptor
  owns pixel-based observation; deterministic code owns validation, caps,
  cache keys, and persistence; cognition owns stance over the descriptor text
  but does not see pixels.
- Rejected complexity: no new visual identity classifier stage, no cognition
  image access, no embeddings, no vector image search, no fallback LLM retries,
  no provider-specific multimodal APIs, no new MongoDB collection.
- Evidence threshold: add a separate plan only after live LLM review shows that
  labelled seeded references still produce unacceptable false positives or
  repeated reference/target merging.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside the listed change surface as
  blocked until the plan is updated and approved.
- The responsible agent must search for existing helper behavior before adding
  new Python helpers.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, the responsible agent must preserve the
  plan's stated intent and report the discrepancy.

## Implementation Order

The user requested this phase order: implement proposed code, run the real LLM
test, then continue normal test development and review. This plan follows that
order while still requiring verification before completion.

1. Implement seeded visual media support.
   - Add `visual_seed_materials` to `personalities/kazusa.json`.
   - Add the normalization/fingerprint module.
   - Wire the descriptor to use normalized references from
     `state["character_profile"]`.
   - Update descriptor prompt and cache key behavior.
   - Record the three official or online seed-reference source URLs and hashes
     in the profile rows.
2. Inventory and save the real LLM test target corpus manifest.
   - Record at least five Kazusa target images and five non-Kazusa target
     images under
     `test_artifacts/diagnostics/visual_seed_reference_corpus/`.
   - Write a manifest with source URL, generated-vs-downloaded provenance,
     role, media type, and case id.
3. Run the real LLM seeded-reference test one case at a time.
   - Use the seeded reference set for every target.
   - Run at least five Kazusa target cases and five non-Kazusa target cases.
   - Inspect raw output and write a readable review artifact before judging.
4. Add and run normal deterministic tests.
   - Add focused tests for normalizer, cache key, message construction, prompt
     contract, and target-only fallback.
   - Run affected regression tests.
5. Update docs and plan execution evidence.
6. Run independent code review and remediate approved findings.

## Execution Model

- Parent agent owns orchestration, corpus manifest, test code, live LLM review
  artifacts, verification, execution evidence, review feedback remediation,
  lifecycle updates, and final sign-off.
- Production-code subagent: exactly one native subagent after plan approval;
  owns production code changes only.
- Parent agent owns the real LLM test run and review artifacts.
- Independent code-review subagent: exactly one native subagent after planned
  verification passes; reviews the plan, diff, and evidence; reports findings
  to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before production
  execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - seeded descriptor implementation complete.
  - Covers: implementation order step 1.
  - Verify: production-code subagent reports changed files and no blocked
    contracts.
  - Evidence: record changed files, seed-reference source URLs and hashes,
    cache-key contract, prompt contract, and seed normalization behavior.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-06-25` after no-subagent fallback
    implementation, syntax checks, seed normalization probe, and message-shape
    probe completed.
- [x] Stage 2 - target corpus inventory complete.
  - Covers: implementation order step 2.
  - Verify: manifest exists with five Kazusa targets and five non-Kazusa
    targets.
  - Evidence: record manifest path, source counts, downloaded/generated
    provenance categories, and hard-negative buckets.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-06-25` after manifest count and PNG byte-signature
    verification completed.
- [x] Stage 3 - real LLM false-positive review complete.
  - Covers: implementation order step 3.
  - Verify: each live LLM case is run individually with raw output saved and
    reviewed.
  - Evidence: record command per case, model route, output artifact paths,
    false-positive count, and manual quality judgment.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-06-25` after ten live LLM cases were run
    individually, traces inspected, and review artifact written.
- [ ] Stage 4 - deterministic test development complete.
  - Covers: implementation order step 4.
  - Verify: focused deterministic tests and affected regressions pass.
  - Evidence: record exact commands and outputs.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - documentation and evidence complete.
  - Covers: implementation order step 5.
  - Verify: docs mention the optional profile field and live LLM command.
  - Evidence: record docs changed and final `git status --short`.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 - independent code review complete.
  - Covers: implementation order step 6.
  - Verify: independent review reports approval or all concrete findings are
    remediated with affected checks rerun.
  - Evidence: record reviewer, findings, fixes, rerun commands, residual risks,
    and approval status.
  - Handoff: plan can move to completed only after this stage is signed off.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

### Static Greps

- `rg "_VISION_DESCRIPTOR_PROMPT|visual_seed_materials|reference_set_fingerprint" src tests personalities docs development_plans`
  returns matches in the planned prompt, seed normalizer, cache policy, tests,
  profile, docs, and this plan.
- `rg "Kazusa|杏山千纱|Blue Archive" src/kazusa_ai_chatbot`
  returns no reusable runtime prompt or code matches introduced by this plan,
  except profile data or tests where concrete Kazusa fixtures are expected.
- `rg "SystemMessage\\(content=\\[|AIMessage\\(content=\\[" src tests`
  returns no seeded image injection through system or assistant image content.
- `rg "base64_data|data:image" test_artifacts/diagnostics/visual_seed_reference_reviews`
  returns no matches after live review artifacts are written.

### Deterministic Tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_media_descriptor_cache.py tests\test_msg_decontexualizer.py -q
```

### Real LLM Tests

Run each case separately and inspect the output before continuing:

```powershell
venv\Scripts\python.exe -m pytest tests\test_visual_identity_reference_probe_live_llm.py::test_live_seeded_reference_<case_id> -q -s
```

The live LLM review must include:

- seed reference file list and source labels;
- target image path and case role;
- raw model output;
- parsed descriptor output;
- whether the output names the active character or aliases;
- whether the target image, not the reference images, was described;
- human-readable quality notes.

### Acceptance Gate For Live LLM

- Every non-Kazusa target case passes only when `description`,
  `visible_text`, `salient_visual_facts`, `spatial_or_scene_facts`, and
  `uncertainty` do not name the active character or aliases.
- Kazusa target cases are reviewed for recognition quality but do not fail the
  plan if the descriptor describes visible traits without naming Kazusa.

## Independent Plan Review

Review was requested on 2026-06-25 before implementation. No separate review
subagent was used for this draft review; the drafting agent reread this plan,
the development-plan skill, the local-LLM architecture skill, and the current
git state from a fresh-review posture.

| Severity | Finding | Resolution |
|---|---|---|
| Blocker | The implementation order put corpus inventory before code implementation, conflicting with the user's requested phase order. | Reordered stages so seeded descriptor implementation is Stage 1, target corpus inventory is Stage 2, and real LLM testing is Stage 3. |
| Blocker | Seed reference caps were mentioned but not numerically defined. | Added a three-reference per-call cap, 2 MiB decoded row cap, and deterministic inclusion order. |
| Blocker | The false-positive gate did not define how aliases are derived for arbitrary characters. | Added a data-owned alias rule using `character_profile.name`, seed mapping labels, and optional seed-row `display_names`. |
| Blocker | The plan did not explicitly forbid AI-generated seed references. | Added a rule that seed references must be official or online downloaded references; generated images are target-side tests only. |
| Non-blocking | The plan did not state how raw media avoids trace/review leakage. | Added a privacy rule and review-artifact static grep forbidding base64/data URI leakage. |
| Non-blocking | Prompt-rendering verification was implicit. | Added a deterministic prompt-rendering test requirement. |

Review status: blockers resolved in the draft. The user approved fallback
execution without subagents for stages 1 through 3 on 2026-06-25.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Character-data shape and genericity across arbitrary characters.
- Reference/target ownership in descriptor prompt and message construction.
- Absence of hard-coded Kazusa runtime prompt behavior.
- Cache-key isolation between seeded and unseeded descriptor calls.
- Real LLM corpus quality, false-positive gate, and review artifacts.
- Deterministic test coverage for validation, cache, prompt, and fallback.
- Alignment with `Must Do`, `Deferred`, `Change Surface`, and verification
  gates.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `visual_seed_materials` exists as optional character profile data and Kazusa's
  initial profile includes three seed references.
- The three seed references are official or online downloaded images, not
  AI-generated images.
- Profiles without visual seed media keep target-only descriptor behavior.
- The descriptor constructs labelled reference and target user-role messages.
- The descriptor output schema and downstream `image_observation` shape remain
  unchanged.
- Media descriptor cache keys vary by target image and reference-set
  fingerprint.
- The real LLM corpus has at least three seed references, five Kazusa targets,
  and five non-Kazusa targets with a manifest.
- Non-Kazusa live LLM cases produce zero active-character false positives.
- Deterministic focused tests and affected regressions pass.
- Documentation explains the optional profile field and the live LLM review
  workflow.
- Independent code review is complete and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Model describes reference images instead of target | Separate reference and target `HumanMessage`s, explicit target-only prompt wording, live LLM review gate. | Real LLM review checks target-only description. |
| Model overfits any overlapping traits as active character | Use hard negative buckets and make non-Kazusa alias absence the pass gate. | Live LLM false-positive gate. |
| Cache returns stale unseeded descriptor | Include reference fingerprint in cache key. | Cache key tests. |
| Character-specific prompt leakage | Keep concrete names only in character data and tests. | Static grep over runtime prompt/code. |
| Profile base64 bloat | Validate media type and size, cap references per call. | Normalizer tests and review. |
| Raw media leaks into debug artifacts | Store only paths, hashes, source URLs, and text outputs in test reviews. | Static grep over review artifacts. |

## Execution Evidence

- Draft created on 2026-06-25 from the visual identity overfitting discussion
  and the user's preference for seeded image references as the final solution.
- Source inspection found the current descriptor prompt and implementation use
  a single target `image_url`; `IMProcessState` already includes
  `character_profile`; `character_state` is open-ended and stores both static
  profile fields and runtime state; media descriptor cache keys currently vary
  by target content type, target hash, and descriptor version only.
- Plan review on 2026-06-25 resolved blockers around phase ordering, seed
  caps, arbitrary-character alias ownership, AI-generated seed exclusion,
  raw-media artifact privacy, and prompt-render verification.
- Execution started on 2026-06-25 for stages 1 through 3 by explicit user
  approval, with no-subagent fallback execution requested by the user.
- Stage 1 implementation changed
  `src/kazusa_ai_chatbot/visual_seed_materials.py`,
  `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`,
  `src/kazusa_ai_chatbot/rag/cache2_policy.py`,
  `src/kazusa_ai_chatbot/db/schemas.py`, and `personalities/kazusa.json`.
  Verification: `venv\Scripts\python.exe -m py_compile` over touched Python
  files passed; prompt render check passed with UTF-8 stdout; seed
  normalization returned three PNG references and fingerprint
  `64597d2ab18a41f3c0abdba5a236c8b144156cbc199957cd3aa7943e7ac9ce1a`;
  message-shape probe returned three messages with reference content before
  target content.
- Stage 2 corpus inventory wrote
  `test_artifacts/diagnostics/visual_seed_reference_corpus/manifest.json`
  with three seed references, five Kazusa targets, and five non-Kazusa
  targets. Initial CDN downloads produced WebP bytes despite PNG source names;
  the corpus was redownloaded with `&format=original`, and all 13 local target
  files verified as PNG by byte signature.
- Stage 3 live LLM execution ran all ten cases one at a time with
  `venv\Scripts\python.exe -m pytest -m live_llm
  tests\test_visual_identity_reference_probe_live_llm.py::<case> -q -s`.
  Initial live attempt failed before trace creation because the local
  OpenAI-compatible vision route rejected WebP data URIs. After the PNG
  redownload, all ten cases passed structural pytest assertions and wrote
  traces under `test_artifacts/llm_traces/`.
- Stage 3 review artifact:
  `test_artifacts/diagnostics/visual_seed_reference_reviews/seeded_reference_stage3_review_20260625.md`.
  Result: zero explicit alias false positives in five non-Kazusa targets, but
  manual review found reference-trait contamination in Airi Band and Ichika,
  with partial contamination in Kayoko. Artifact hygiene grep
  `rg "data:image|base64_data|base64," test_artifacts\llm_traces` returned no
  matches.
- Follow-up live LLM testing on 2026-06-26 tested the alternate message shape
  `SystemMessage`, `HumanMessage(OBSERVATION_TARGET)`,
  `HumanMessage(REFERENCE_LIBRARY)`, with no repeated final target image. All
  16 cases passed structural pytest assertions, but manual trace review found
  clear target-ownership failures for `kazusa_band_portrait`,
  `non_kazusa_serika_portrait`, `non_kazusa_airi_band_portrait`, and
  `non_kazusa_yoshimi_portrait`. The model described the seeded Kazusa
  reference portrait instead of the target image in those cases.
- Final conclusion, 2026-06-26: seeded reference images are not a reliable
  production visual-descriptor path for the current local LLM endpoint. The
  seeded-image approach is abandoned. Production should keep the descriptor
  target-image-only unless a future plan introduces a separately verified
  multimodal model/contract with evidence that reference images cannot be
  mistaken for the target.
- Closure artifact:
  `test_artifacts/llm_traces/visual_descriptor_target_first_reference_only_review_20260626.md`.
