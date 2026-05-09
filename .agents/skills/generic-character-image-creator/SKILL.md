---
name: generic-character-image-creator
description: Use when a user pastes visual_directives or asks Codex to generate a character image, seed image, visual-agent image, or character illustration from local metadata such as seed_reference_manifest.json.
---

# Generic Character Image Creator

## Core Contract

Generate one or more character images from pasted visual directives plus local
metadata. Do not assume any default character, franchise, outfit, or setting.
The user input and the current manifest are the authority.

Use this skill for raster image generation. Invoke the project/session image
generation capability according to the active environment rules.

## Metadata Lookup

1. Prefer a user-specified manifest path.
2. Otherwise, from the current workspace, look for:
   `personalities/seeding_images/seed_reference_manifest.json`.
3. If no manifest exists and the user did not provide enough identity/style
   information, ask for a manifest or a short identity anchor.

Read only the fields needed for the current image:

- `character_id`: file naming and manifest grouping only.
- `identity_prompt_anchor`: stable identity/style cues.
- `negative_identity_cues`: things to avoid.
- `production_exposure`: whether image files are default or fallback.
- `generation_targets`: allowed/common settings, outfits, and gestures.
- `references`: local fallback reference paths and prompt cues.
- `generated_seeds`: prior generated images and review status.

Do not hard-code any concrete character name in the skill workflow. If the
manifest contains a concrete name, use it because it is metadata, not a skill
default.

## Prompt Build

Build the image prompt in this order:

1. State the asset purpose: character seed image from visual directives.
2. Add the manifest identity anchor, if present.
3. Include the pasted `visual_directives` block verbatim.
4. Add a short image-facing synthesis of the directives:
   - body language and hand placement
   - facial expression
   - gaze direction
   - scene, lighting, camera distance, composition
5. Add outfit/setting/gesture tags from the manifest only when they match the
   pasted directives or the user explicitly selects them.
6. Add negative cues from the manifest plus practical image constraints:
   one character unless requested, no text/watermark/speech bubbles, coherent
   hands, coherent anatomy, coherent environment geometry.

Keep the prompt generic in wording. Use phrases like "the character", "the
subject", or the manifest-provided name. Do not introduce a character name that
is not in the manifest or user request.

## Reference Images

Default production behavior should expose prompt and metadata, not image files.
Use local reference images only when:

- the user explicitly asks to use a reference image,
- identity or composition consistency has failed,
- a prior accepted generated seed is clearly the best anchor for an edit.

When using local references, label their role clearly: identity anchor, outfit
reference, setting reference, gesture reference, or edit target.

## Save And Manifest Workflow

For project-bound output:

1. Generate or edit the image.
2. Copy the selected output into `personalities/seeding_images/generated/`.
   Do not delete the original generated file.
3. Use a stable filename:
   `<character_id-or-character>_seed_<number>_<short-slug>.png`.
4. Update the manifest when requested or when continuing an existing seed pack:
   - `local_path`
   - `status`
   - `production_exposure`
   - `fallback_reference_enabled`
   - `prompt_contract`
   - `generated_with`
   - hash, size, and dimensions when practical
   - the original `visual_directives`
5. New generations should start as `needs_manual_review`.
6. If the user accepts one, mark it `accepted_manual_review`.
7. If the user rejects or replaces one, keep it and mark it
   `rejected_replaced` with `replacement_id` and a short review note.

## Correction Loop

When the user reports a visual defect, treat it as a targeted correction:

- preserve accepted identity, scene, outfit, lighting, and composition;
- edit or regenerate only the failed region or property when possible;
- explicitly name the defect in the correction prompt, such as "exactly five
  fingers total" or "doorframe remains straight";
- save the corrected image as a versioned sibling, such as `_v2`;
- update the manifest replacement chain.

Common high-risk defects to guard against in the prompt: extra fingers,
duplicated ears, missing key identity features, over-bright backgrounds, text
artifacts, extra people, impossible furniture or vehicle geometry.
