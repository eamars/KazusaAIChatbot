# history media projection image boundary bugfix plan

## Summary

- Goal: Make `chat_history_recent` and `chat_history_wide` preserve prompt-safe
  image meaning by projecting attachment descriptions into readable text bounded
  with `<image>...</image>`.
- Plan class: medium
- Status: completed
- Mandatory skills: `local-llm-architecture`, `py-style`,
  `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: bigbang
- Highest-risk areas: changing downstream prompts instead of projection,
  expanding history rows with unused metadata, letting adapter wire syntax enter
  semantic envelope fields, losing reply-image descriptions, and allowing
  image-description text to break the `<image>...</image>` boundary.
- Acceptance criteria: image-only history rows are no longer empty, text plus
  images keeps both text and image meaning, reply image descriptions survive in
  `reply_context.reply_excerpt`, adapter wire syntax is sanitized before it
  enters semantic envelope fields, description text cannot inject image
  boundary tags, and no production LLM prompt is changed.

## Context

The observed failure is in prompt-facing sliding history, not in stored
conversation rows or vector search. A real QQ group case showed a prior image
row with:

- stored `body_text`: empty string
- stored raw CQ image syntax in `raw_wire_text`
- stored attachment description: `拓竹入驻山姆，不只是上架 3D 打印机`

`trim_history_dict(...)` currently copies only `body_text` and a narrow
`reply_context` subset. The resulting `chat_history_recent` and
`chat_history_wide` rows are blind to attachment descriptions, so downstream
LLM stages may search unnecessarily or misread local references.

The fix belongs in history projection. Downstream prompts must not be changed
to explain image labels. The label must be self-contained in the projected
history text itself.

## Mandatory Skills

- `local-llm-architecture`: load before changing prompt-facing history
  projection or LLM context shape.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python source or tests containing CJK
  strings.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the plan's `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- Do not change downstream LLM prompts, cognition prompts, RAG initializer
  prompts, dialog prompts, evaluator prompts, or prompt instructions.
- Do not add a new LLM call, classifier prompt, repair prompt, or retry path.
- Do not add metadata fields to `chat_history_recent` or `chat_history_wide`
  for this bugfix. Current consumers read text, not metadata.
- Preserve the existing `trim_history_dict(...)` output shape.
- Do not parse raw platform wire syntax as the source of media meaning in the
  brain service.
- Adapter-specific wire syntax must be sanitized at the adapter boundary before
  it reaches semantic envelope fields such as `body_text` or `reply.excerpt`.
- Escape literal `&`, `<`, and `>` inside image descriptions before wrapping
  them with `<image>...</image>` so stored description text cannot close or
  create image boundaries.
- Keep deterministic projection bounded, inspectable, and local to the history
  projection boundary.

## Must Do

- Add focused deterministic tests for `trim_history_dict(...)` before changing
  implementation.
- Project image attachment descriptions into `body_text` using exact
  `<image>...</image>` boundaries.
- Use the existing `MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS` cap for projected
  image descriptions. Do not introduce a second numeric cap.
- Escape description text before inserting it into an image block.
- For attachment-only image rows, set prompt-facing `body_text` to one or more
  image blocks instead of an empty string.
- For text plus image rows, preserve the original text and append image blocks
  on later lines.
- Assume `body_text` is already adapter-sanitized; history projection must not
  contain platform-specific CQ stripping.
- For reply context, fold `reply_attachments[*].description` into
  `reply_context.reply_excerpt` using `<image>...</image>` blocks.
- Assume `reply_excerpt` is already adapter-sanitized; history projection must
  not contain platform-specific CQ stripping.
- Treat an attachment as an image only when `media_kind == "image"` or
  `media_type` starts with `image/`.
- Preserve current typed-addressing fields, timestamps, roles, speaker ids,
  mentions, and broadcast fields.
- Run focused deterministic tests and the corrected L1-only experiment as
  evidence.

## Deferred

- Do not replace recent history with `conversation_graph` in this bugfix.
- Do not add a `media` or `attachments` metadata contract to recent history.
- Do not change RAG initializer logic or retrieval slot policy.
- Do not modify current-turn `PromptMessageContext` projection unless a focused
  test proves it already regressed.
- Do not change stored MongoDB rows, embeddings, cache entries, or historical
  data.
- Do not backfill old rows.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| `trim_history_dict(...)` text projection | bigbang | Replace prompt-facing history text projection directly. No flag, fallback, dual path, or old empty-image behavior. |
| Output row shape | compatible | Preserve existing keys and types for current consumers. Change only the text content of `body_text` and `reply_context.reply_excerpt`. |
| Downstream prompts | bigbang | Keep prompts unchanged. Do not add label-reading instructions. |
| Stored data | compatible | Do not mutate database rows or regenerate embeddings. |
| Experiments | bigbang | Use the temporary experiment only for evidence, then remove the experiment code before independent code review. Keep raw trace/review artifacts as evidence. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, rewrite the old behavior instead of preserving a
  compatibility branch.
- If an area is `compatible`, preserve only the compatibility surface listed in
  this plan.
- Any change to a cutover policy requires user approval before implementation.

## Target State

For a stored image-only row:

```python
{
    "body_text": "",
    "raw_wire_text": "[CQ:image,file=...]",
    "attachments": [
        {"media_type": "image/png", "description": "拓竹入驻山姆，不只是上架 3D 打印机"},
    ],
}
```

`trim_history_dict(...)` projects:

```python
{
    "body_text": "<image>拓竹入驻山姆，不只是上架 3D 打印机</image>",
}
```

For a text plus image row:

```python
{
    "body_text": "这个也进山姆了",
    "attachments": [
        {"media_type": "image/png", "description": "拓竹入驻山姆，不只是上架 3D 打印机"},
    ],
}
```

`trim_history_dict(...)` projects:

```python
{
    "body_text": "这个也进山姆了\n<image>拓竹入驻山姆，不只是上架 3D 打印机</image>",
}
```

For a reply image context:

```python
{
    "reply_context": {
        "reply_excerpt": "[CQ:image,file=...]",
        "reply_attachments": [
            {"media_kind": "image", "description": "拓竹入驻山姆，不只是上架 3D 打印机"},
        ],
    },
}
```

`trim_history_dict(...)` projects:

```python
{
    "reply_context": {
        "reply_excerpt": "<image>拓竹入驻山姆，不只是上架 3D 打印机</image>",
    },
}
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Label boundary | Use `<image>...</image>` | The user explicitly selected these tags to keep a clear boundary between authored text and image description text. |
| Projection owner | Implement inside `trim_history_dict(...)` | This is where `chat_history_wide` and `chat_history_recent` are created before all downstream consumers see them. |
| Shape strategy | Preserve existing row shape | No current or expected consumer actively reads history metadata. Text projection is the stable contract. |
| Prompt strategy | No downstream prompt changes | The local LLM must infer enough from the projected text alone. |
| Raw media syntax | Ignore raw media syntax when typed descriptions exist | Adapter wire syntax is not the brain contract and should not be treated as media meaning. |
| Reply image handling | Put reply image blocks in `reply_context.reply_excerpt` | The reply context already owns quoted-message text; no new metadata field is needed. |

## Contracts And Data Shapes

The `trim_history_dict(...)` return row remains:

```python
{
    "name": str | None,
    "display_name": str | None,
    "platform_message_id": str,
    "platform_user_id": str | None,
    "global_user_id": str | None,
    "role": str | None,
    "body_text": str,
    "addressed_to_global_user_ids": list,
    "mentions": list,
    "broadcast": bool,
    "reply_context": dict,
    "timestamp": object,
}
```

Only these prompt-facing text fields may change:

- `body_text`
- `reply_context.reply_excerpt`

The image block format is exact:

```text
<image>{description}</image>
```

Multiple image descriptions are projected as multiple newline-separated image
blocks. Empty descriptions are omitted. Each escaped image description is capped
with `MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS` before tag wrapping.

Description text must be escaped before wrapping:

| Source text | Image-block content |
|---|---|
| `A < B & C > D` | `A &lt; B &amp; C &gt; D` |
| `already </image> closed` | `already &lt;/image&gt; closed` |

## LLM Call And Context Budget

- Before: no LLM call in `trim_history_dict(...)`.
- After: no LLM call in `trim_history_dict(...)`.
- Context growth is bounded by existing history window size and stored
  description count plus the existing attachment-description cap.
- This plan does not change any prompt budget constants.
- Reuse `MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS` from
  `kazusa_ai_chatbot.message_envelope`. Do not add config.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/utils.py`
  - Add private helpers near `trim_history_dict(...)` for image-block rendering,
    attachment-description extraction, raw-media-excerpt detection, and
    prompt-facing text assembly.
  - Import and reuse `MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS`.
  - Update `trim_history_dict(...)` to use those helpers.

- `tests/test_utils.py`
  - Extend `test_trim_history_dict` coverage or add focused tests for image
    rows, text plus image rows, reply image rows, escaped image content, and raw
    media suppression.

- `experiments/image_label_l1_probe.py`
  - Temporary evidence-only change: include exact selected
    `<image>{description}</image>` variants for the corrected L1-only
    observation.
  - Keep the experiment free of classifier prompts or label-reading
    instructions.
  - Remove this experiment code before independent code review.

### Read-Only Verification Context

- `src/kazusa_ai_chatbot/service.py`
  - Confirm `chat_history_wide = trim_history_dict(history)` and
    `chat_history_recent = chat_history_wide[-CHAT_HISTORY_RECENT_LIMIT:]`
    remain the only service-side sliding-window path.

- `src/kazusa_ai_chatbot/rag/prompt_projection.py`
  - Confirm RAG prompt projection preserves the already-projected history text
    while continuing to format timestamps and strip raw fields.

- `src/kazusa_ai_chatbot/message_envelope/prompt_projection.py`
  - Keep current-message prompt projection unchanged unless a focused test
    proves it already violates the same boundary.

### Keep Unchanged

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- `src/kazusa_ai_chatbot/db/conversation.py`
- MongoDB data, embeddings, and Cache2 entries

## Overdesign Guardrail

- Do not create a new history-row dataclass, schema migration, projection
  service, registry, or metadata consumer.
- Do not introduce a generic XML renderer or parser.
- Add only a local image-description escape helper for `&`, `<`, and `>`.
- Do not support arbitrary media tags in this bugfix. Only image descriptions
  are required by the observed failure.
- Do not use an LLM to classify whether a row is a media description.
- Do not add new environment variables, feature flags, or compatibility modes.

## Agent Autonomy Boundaries

- The implementation agent may choose private helper names in `utils.py`.
- The implementation agent must keep all production behavior changes inside
  `trim_history_dict(...)` unless an existing test proves another listed file
  must change.
- If implementation reveals that reply image descriptions are unavailable
  before `trim_history_dict(...)`, stop and report the upstream hydration
  boundary instead of expanding scope silently.
- If a test exposes non-image attachment needs, keep that as a follow-up and do
  not generalize this bugfix.

## Implementation Order

1. Parent adds focused deterministic tests in `tests/test_utils.py`.
   - Include `test_trim_history_dict_projects_image_only_attachment_text`.
   - Include `test_trim_history_dict_appends_image_block_after_body_text`.
   - Include `test_trim_history_dict_projects_reply_image_description`.
   - Include `test_trim_history_dict_escapes_image_description_boundaries`.
   - Include `test_trim_history_dict_truncates_long_image_description`.
   - Run:
     `venv\Scripts\python -m pytest tests\test_utils.py -q`
   - Expected before implementation: new tests fail because descriptions are
     missing.
2. Parent starts the production-code subagent with this approved plan.
   - Ownership: `src/kazusa_ai_chatbot/utils.py` only.
   - The subagent must not edit prompts, service graph wiring, DB helpers, or
     tests.
3. Production-code subagent implements the minimal projection helpers and
   updates `trim_history_dict(...)`.
4. Parent reruns:
   `venv\Scripts\python -m pytest tests\test_utils.py -q`
   - Expected after implementation: focused tests pass.
5. Parent runs adjacent deterministic checks:
   - `venv\Scripts\python -m pytest tests\test_build_interaction_history_recent.py -q`
   - `venv\Scripts\python -m pytest tests\test_llm_time_payload_projection.py -q`
6. Parent updates the corrected experiment-only L1 observation so it includes
   the exact selected `<image>{description}</image>` variant and no classifier
   prompt.
   - Run:
     `venv\Scripts\python experiments\image_label_l1_probe.py --input experiments\image_label_l1_candidates.sample.json --case-id sam_purchase_power --variant selected_image_boundary`
   - Record the new trace path in `Execution Evidence`.
7. Parent removes the temporary experiment code added for this bugfix.
   - Delete `experiments/image_label_l1_probe.py` if it was created only for
     this bugfix.
   - Delete `experiments/image_label_l1_candidates.sample.json` if it was
     created only for this bugfix.
   - Keep raw traces and the human-authored review artifact under
     `test_artifacts/` as execution evidence.
   - Run:
     `git status --short --ignored experiments/image_label_l1_probe.py experiments/image_label_l1_candidates.sample.json`
   - Expected before review: no remaining tracked or untracked experiment code
     from this bugfix.
8. Parent starts the independent code-review subagent after verification
   passes.
9. Parent remediates review findings inside the approved change surface and
   reruns affected verification.

## Execution Model

- Normal execution is parent-led with native subagents.
- The parent owns tests, verification, execution evidence, checklist updates,
  independent review, and final sign-off.
- The production-code subagent owns only `src/kazusa_ai_chatbot/utils.py`.
- The independent review subagent owns review only and must not implement
  fixes.
- If native subagents are unavailable, stop before execution unless the user
  explicitly approves fallback single-agent execution.

## Progress Checklist

- [x] Focused `trim_history_dict(...)` tests added in `tests/test_utils.py` and
  failing baseline recorded.
- [x] `trim_history_dict(...)` image-block projection implemented in
  `src/kazusa_ai_chatbot/utils.py`.
- [x] Focused `tests/test_utils.py` pass recorded.
- [x] Adjacent history and RAG projection tests pass.
- [x] Corrected L1-only experiment trace recorded.
- [x] Temporary experiment code removed before independent code review.
- [x] Independent code review completed and findings recorded.
- [x] Review findings remediated or explicitly rejected with rationale.
- [x] Plan status and execution evidence updated before final sign-off.

## Verification

Required deterministic checks:

```powershell
venv\Scripts\python -m pytest tests\test_utils.py -q
venv\Scripts\python -m pytest tests\test_build_interaction_history_recent.py -q
venv\Scripts\python -m pytest tests\test_llm_time_payload_projection.py -q
```

Required syntax check if Python files with CJK strings are edited:

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\utils.py tests\test_utils.py
```

Required experiment evidence:

```powershell
venv\Scripts\python experiments\image_label_l1_probe.py --input experiments\image_label_l1_candidates.sample.json --case-id sam_purchase_power --variant selected_image_boundary
```

The experiment is evidence only. It must not be used as the sole pass/fail gate
for deterministic projection behavior.

Required cleanup before independent code review:

```powershell
git status --short --ignored experiments/image_label_l1_probe.py experiments/image_label_l1_candidates.sample.json
```

The expected result before independent code review is that no experiment code
created for this bugfix remains in the working tree. Raw traces and
human-authored review artifacts under `test_artifacts/` may remain as ignored
evidence.

## Independent Code Review

After implementation and verification pass, run an independent review against:

- this plan,
- the full production/test diff,
- deterministic test output,
- the corrected L1-only experiment trace.

The review must check:

- no downstream prompts changed,
- no new LLM calls were added,
- temporary experiment code has been removed before review,
- no new history metadata contract was introduced,
- adapter-specific CQ syntax is not handled by brain history projection,
- `<image>...</image>` boundaries are exact,
- image descriptions escape literal `&`, `<`, and `>` before tag wrapping,
- reply image descriptions survive in `reply_context.reply_excerpt`,
- tests cover image-only, text-plus-image, reply-image, boundary escaping,
  truncation, and adapter-boundary CQ sanitation.

## Acceptance Criteria

- `chat_history_recent` and `chat_history_wide` rows produced from image-only
  stored rows expose `<image>description</image>` in `body_text`.
- Text plus image rows preserve authored text and append image blocks on later
  lines.
- Reply image descriptions appear in `reply_context.reply_excerpt` with
  `<image>...</image>` boundaries.
- Adapter-specific CQ syntax is sanitized before semantic envelope fields reach
  the brain service; brain history projection does not parse CQ syntax.
- Literal `&`, `<`, and `>` in image descriptions are escaped inside image
  blocks, so description content cannot create or close `<image>` boundaries.
- Existing row shape is preserved; no new metadata consumer is required.
- No production LLM prompt, downstream prompt, or model-call graph is changed.
- Temporary experiment code added for this bugfix is removed before independent
  code review.
- Required deterministic tests pass and execution evidence records the commands
  and outputs.

## Risks

- Image descriptions can increase prompt length. This plan accepts that risk
  because sliding windows are already bounded and descriptions are already
  prompt-safe summaries.
- A local LLM may still not explicitly know the image block is machine
  generated. This plan only requires the boundary to be clear and non-confusing
  without downstream prompt changes.
- If stored attachment descriptions are missing, this bugfix cannot recover the
  visual meaning. It intentionally does not invoke vision or search during
  history projection.

## Plan Review Findings Addressed

- Added exact `<image>{description}</image>` experiment verification because
  earlier L1 traces tested `<image_desc>...</image_desc>`, not the selected tag.
- Added deterministic escaping for literal `&`, `<`, and `>` inside image
  descriptions so tag boundaries stay clear.
- Superseded raw media suppression inside brain history projection after the
  adapter-boundary contract was clarified: CQ syntax is now sanitized in the
  NapCat adapter before semantic envelope fields reach the brain.
- Replaced the draft-only `MAX_HISTORY_IMAGE_DESCRIPTION_CHARS = 800` proposal
  with reuse of existing `MAX_PROMPT_ATTACHMENT_DESCRIPTION_CHARS`.
- Added image-attachment recognition rules for both `media_kind` and
  `media_type`.
- Added a cleanup gate requiring temporary experiment code removal before
  independent code review.

## Execution Evidence

- Draft created: 2026-05-23.
- Plan reviewed and known issues addressed: 2026-05-23.
- Status changed to `in_progress`: 2026-05-23.
- Focused failing test contract added in `tests/test_utils.py`.
- Baseline command:
  `venv\Scripts\python -m pytest tests\test_utils.py -q`
  - Result before implementation: 7 failed, 13 passed, 3 deselected.
  - Failing tests:
    `test_trim_history_dict_projects_image_only_attachment_text`,
    `test_trim_history_dict_appends_image_block_after_body_text`,
    `test_trim_history_dict_projects_reply_image_description`,
    `test_trim_history_dict_prefers_reply_image_description_over_raw_cq`
    (superseded by adapter-boundary tests),
    `test_trim_history_dict_strips_raw_cq_from_body_when_description_exists`
    (superseded by adapter-boundary tests),
    `test_trim_history_dict_escapes_image_description_boundaries`,
    `test_trim_history_dict_truncates_long_image_description`.
- Syntax check after adding tests:
  `venv\Scripts\python -m py_compile tests\test_utils.py`
  - Result: passed.
- Production implementation completed in `src/kazusa_ai_chatbot/utils.py`.
- Syntax check after production implementation:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\utils.py tests\test_utils.py`
  - Result: passed.
- Focused verification after implementation:
  `venv\Scripts\python -m pytest tests\test_utils.py -q`
  - Result: 20 passed, 3 deselected.
- Follow-up boundary correction after user clarification:
  - Removed brain-side `_CQ_IMAGE_SEGMENT_RE` and CQ image stripping from
    `src/kazusa_ai_chatbot/utils.py`.
  - Added adapter-boundary QQ reply excerpt sanitation in
    `src/adapters/napcat_qq_adapter.py`.
  - Removed superseded brain-side raw-CQ stripping tests from
    `tests/test_utils.py`.
  - Added adapter tests for image-only and mixed CQ reply excerpts.
  - Syntax check:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\utils.py src\adapters\napcat_qq_adapter.py tests\test_utils.py tests\test_adapter_envelope_normalizers.py tests\test_runtime_adapter_registration.py`
    - Result: passed.
  - Adapter normalizer verification:
    `venv\Scripts\python -m pytest tests\test_adapter_envelope_normalizers.py -q`
    - Result: 6 passed.
  - History projection verification after removing brain-side CQ handling:
    `venv\Scripts\python -m pytest tests\test_utils.py -q`
    - Result: 18 passed, 3 deselected.
  - Runtime adapter verification:
    `venv\Scripts\python -m pytest tests\test_runtime_adapter_registration.py -q`
    - Result: 51 passed.
- Adjacent verification:
  `venv\Scripts\python -m pytest tests\test_build_interaction_history_recent.py -q`
  - Result: 6 passed.
- RAG projection verification:
  `venv\Scripts\python -m pytest tests\test_llm_time_payload_projection.py -q`
  - Result: 19 passed.
- Corrected L1-only selected-boundary experiment:
  `venv\Scripts\python experiments\image_label_l1_probe.py --input experiments\image_label_l1_candidates.sample.json --case-id sam_purchase_power --variant selected_image_boundary`
  - Trace:
    `test_artifacts/llm_traces/image_label_l1_probe__sam_purchase_power__selected_image_boundary__20260522T231733089803Z.json`
  - L1 output:
    `emotional_appraisal=平静地扫过，没什么特别的感觉。`;
    `interaction_subtext=随口分享的新闻，没打算找我讨论。`
- Human-readable L1 review artifact updated:
  `test_artifacts/conversation_graph_priority/image_label_l1_probe_review.md`.
- Temporary experiment code cleanup:
  `git status --short --ignored experiments/image_label_l1_probe.py experiments/image_label_l1_candidates.sample.json`
  - Result: no output. No temporary experiment code remains in the working tree.
- Independent code review completed by subagent:
  - Result: no blocking or non-blocking code findings.
  - Spec compliance: approved against the plan.
  - Temporary experiment cleanup independently confirmed.
  - Residual coverage notes accepted as non-blocking: multiple image
    attachments, non-image rejection, and mixed reply text plus raw CQ image
    segment are supported by the implementation but not separately locked by
    focused tests.
- Final verification before completion:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\utils.py tests\test_utils.py`
  - Result: passed.
- Final whitespace check:
  `git diff --check -- src/kazusa_ai_chatbot/utils.py tests/test_utils.py development_plans/README.md development_plans/active/bugfix/history_media_projection_image_boundary_plan.md`
  - Result: passed; only Git line-ending warnings were emitted.
- Final experiment cleanup check:
  `git status --short --ignored experiments/image_label_l1_probe.py experiments/image_label_l1_candidates.sample.json`
  - Result: no output.
- Final focused verification:
  `venv\Scripts\python -m pytest tests\test_utils.py -q`
  - Result: 20 passed, 3 deselected.
- Final adjacent history verification:
  `venv\Scripts\python -m pytest tests\test_build_interaction_history_recent.py -q`
  - Result: 6 passed.
- Final RAG projection verification:
  `venv\Scripts\python -m pytest tests\test_llm_time_payload_projection.py -q`
  - Result: 19 passed.
