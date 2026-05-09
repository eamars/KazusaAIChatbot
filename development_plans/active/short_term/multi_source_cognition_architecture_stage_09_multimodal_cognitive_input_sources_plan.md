# multi source cognition architecture stage 09 multimodal cognitive input sources plan

## Summary

- Goal: Represent existing image and audio descriptions as typed
  `CognitiveEpisode` percepts so multimodal inputs can enter cognition without
  raw binary, prompt payload, or text-only `/chat` regression.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` before editing cognition Python files that contain CJK prompt
  text.
- Overall cutover strategy: compatible for text-only `/chat`; compatible for
  existing media-description behavior. No new media summarizer LLM, no raw
  binary prompt input, no proactive output, and no non-chat writes are enabled.
- Highest-risk areas: leaking `base64_data` into prompts, changing text-only
  prompt/RAG behavior, treating media summaries as user instructions, and
  broadening RAG beyond the current text query.
- Acceptance criteria: text-only episodes remain byte-for-byte compatible;
  media descriptions become `image_observation` and `audio_observation`
  percepts; prompt/RAG tests prove raw media is absent; Stage 00 through Stage
  08 gates still pass.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Lifecycle: this draft is blocked until Stage 08 completes and records execution
evidence. Do not approve or execute Stage 09 while the parent ledger row for
`stage_08` is not `completed`.

## Context

Current service state already carries `user_multimedia_input` rows with
`content_type`, `base64_data`, and `description`. Earlier stages deliberately
kept the active cognitive episode text-only. Stage 09 makes the episode
contract express existing media descriptions as separate typed percepts while
preserving the current text query and response path for text-only turns.

This stage must not add media understanding. It consumes descriptions that
already exist before cognition. It must not call a new image, audio, or video
LLM summarizer.

## Stage Handoff

### From Stage 08

Stage 09 expects these completed artifacts:

- internal-thought dry-run episode and audit contracts;
- prompt selector can support non-chat variants without `/chat` regression;
- evidence that private thought stays out of public history and writes;
- parent ledger row for `stage_08` set to `completed`.

Before approval, replace this paragraph with exact Stage 08 branch, commit, and
verification results from Stage 08 `Execution Evidence`, then rerun the plan
self-review.

### To Stage 10

After Stage 09, Stage 10 can rely on:

- text, reflection, internal-thought, image, and audio input sources being
  represented in the shared episode vocabulary;
- raw binary excluded from cognition prompt payloads;
- text-only `/chat` regression evidence after multimodal episode expansion;
- no proactive output behavior enabled by media support.

Stage 10 must treat media percepts as evidence, not permission to contact a
user or send media.

## Mandatory Skills

- `development-plan-writing`: preserve staged lifecycle and handoff.
- `local-llm-architecture`: keep raw data out of local-LLM prompts and use
  typed semantic descriptions.
- `no-prepost-user-input`: do not use deterministic code to infer user
  commands, commitments, or preferences from media descriptions.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing L1/L2/L3 cognition modules containing CJK
  prompt constants.

## Mandatory Rules

- Execute only from a feature branch forked from post-Stage-08 `main`.
- Keep edits inside the approved Change Surface.
- Use PowerShell `-LiteralPath '...'` for filesystem paths that may contain
  spaces; prefer repo-relative paths in commands.
- Do not add new media summarizer LLM calls.
- Do not place `base64_data`, raw bytes, attachment URLs, or unbounded
  attachment metadata in any LLM prompt payload.
- Do not change text-only `/chat` episode shape, prompt selection, RAG request,
  dialog, consolidation, scheduler, or persistence behavior.
- Do not treat `image_observation` or `audio_observation` as an accepted user
  command, preference, promise, or permission in deterministic code.
- RAG query text remains the existing dialog-text compatibility projection.
  Media descriptions may be visible to cognition only through prompt payloads
  approved by this plan.
- Pure-media turns without a dialog-text percept remain unsupported unless this
  plan is explicitly updated before approval.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.

## Must Do

- Extend the text-chat episode builder to accept optional media description
  rows and add typed percepts for:
  - `image_observation` for image content types;
  - `audio_observation` for audio content types.
- Keep the existing text-only builder output exactly unchanged when no media
  rows are supplied.
- Add multimodal prompt selection for exactly these input-source profiles:
  `["dialog_text", "image_observation"]`,
  `["dialog_text", "audio_observation"]`, and
  `["dialog_text", "image_observation", "audio_observation"]`.
- Add multimodal prompt-map entries for L1/L2/L3 handlers without changing
  text-chat, reflection, or internal-thought prompt bytes.
- Update the RAG episode adapter to accept the approved multimodal
  user-message input-source profiles while still projecting only dialog text
  into the RAG query.
- Wire service episode construction to pass existing `user_multimedia_input`
  descriptions into the episode builder.
- Add tests proving raw `base64_data` never appears in episode percept content,
  prompt payloads, RAG requests, or logs produced by the tested path.
- Run every Verification command and record evidence.

## Deferred

- Pure image/audio turns without text.
- New image/audio/video summarizer LLM calls.
- Raw file storage, media download, OCR, speech recognition, or vision model
  integration.
- Multimodal RAG query expansion.
- Reflection/internal-thought media mixing.
- Proactive sends, transport, outbox, or scheduled media output.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Text-only `/chat` | compatible | Existing episode, RAG, prompts, dialog, and consolidation remain unchanged. |
| Media-description `/chat` | compatible | Preserve existing text behavior and add typed media percepts. |
| Raw media | bigbang | Raw binary and unbounded media metadata are forbidden from model payloads. |
| RAG | compatible | Accept multimodal episodes only to project existing dialog text. No media retrieval query changes. |

Rollback path: remove optional media builder arguments, remove multimodal
selector/prompt-map entries, restore the RAG adapter accepted input-source list,
remove service media episode wiring, and remove focused tests. No database
rollback is required.

## Agent Autonomy Boundaries

Allowed implementation choices:

- local fixture helper names;
- assertion ordering;
- local constant names for approved media content-type prefixes.

Not allowed:

- adding summarizer LLMs, media fetchers, OCR, STT, video processing, fallback
  media parsers, feature flags, or alternate graph entrypoints;
- changing prompt text outside the approved multimodal prompt-map additions;
- adding new input-source labels, visibility labels, output modes, or
  persistence schemas;
- modifying dispatcher, scheduler, adapter delivery, reflection worker, or
  consolidator policy;
- adding raising-only helpers or pass-through wrappers.

If verification proves another direct fixture must change, stop and update this
plan before continuing.

## Target State

For media-description chat turns, the episode has one dialog percept plus zero
or more media-description percepts:

```text
user_message
input_sources=["dialog_text", "image_observation", ...]
percepts=[
  dialog_text: existing user_input compatibility text,
  image_observation: bounded description only,
  audio_observation: bounded transcript or tone summary only
]
```

The RAG adapter still extracts dialog text through
`project_text_chat_compatibility_fields(...)`. Media descriptions are
cognition context, not retrieval query parameters.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Media source | Consume existing `MultiMediaDoc.description`. | Avoid new LLM calls and raw media handling. |
| Text-only behavior | Preserve exact output when no media rows exist. | Text chat remains the regression baseline. |
| RAG query | Use dialog-text projection only. | Prevent accidental media-driven retrieval drift. |
| Pure-media turns | Unsupported in this stage. | Current graph still expects dialog-text compatibility. |
| Raw media | Exclude completely from prompts. | Local LLMs need semantic descriptions, not raw bytes. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_episode.py` — add optional media-description
  inputs to the text-chat builder while preserving text-only output.
- `src/kazusa_ai_chatbot/service.py` — pass existing `user_multimedia_input`
  descriptions to the episode builder.
- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py` — accept approved
  multimodal user-message source profiles and keep dialog-text RAG projection.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
  — add approved multimodal prompt variants.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- lifecycle rows in the parent plan and registry after completion only.

### Create

- `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator*.py`
- `src/kazusa_ai_chatbot/reflection_cycle/*.py`
- `src/kazusa_ai_chatbot/internal_thought_cognition.py`
- dispatcher, scheduler, adapter delivery, and database modules

## Implementation Order

1. Reread Stage 08 `Execution Evidence`.
2. Add text-only and multimodal episode-builder tests.
   - Expected before implementation: media-specific assertions fail because
     builder has no media inputs.
3. Extend `build_text_chat_cognitive_episode(...)` with optional bounded media
   description inputs.
4. Add RAG adapter tests for multimodal acceptance and dialog-text projection.
5. Update RAG adapter accepted user-message profiles.
6. Add selector and prompt-render tests for multimodal variants.
7. Wire selector and prompt maps.
8. Wire service episode construction to pass existing media descriptions.
9. Run the full Verification section.
10. Record evidence and sign off.

## Progress Checklist

- [ ] Stage 1 - prerequisite evidence carried forward.
  - Covers: Step 1.
  - Verify: Stage 08 row is `completed`.
  - Evidence: Stage 08 branch, commit, and test results recorded.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 - multimodal episode contract complete.
  - Covers: Steps 2-3.
  - Verify: builder tests pass and text-only output is unchanged.
  - Evidence: red/green results recorded.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 3 - RAG projection compatibility complete.
  - Covers: Steps 4-5.
  - Verify: RAG adapter tests pass.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 4 - cognition prompt admission complete.
  - Covers: Steps 6-7.
  - Verify: selector and prompt-render tests pass.
  - Evidence: prompt fingerprint and test output recorded.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 5 - service media episode wiring complete.
  - Covers: Step 8.
  - Verify: service tests prove no raw media in graph state prompts.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 6.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 6 - full verification complete.
  - Covers: Step 9.
  - Verify: every Verification command passes or has an allowed no-match exit.
  - Evidence: command output recorded.
  - Handoff: Stage 10 draft may be reviewed after lifecycle update.
  - Sign-off: `<agent/date>` after verification.

## Verification

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_episode.py src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py`

### Static Greps

- `rg -n "base64_data" src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\cognition_episode.py`

  Expected result: no matches in prompt, RAG, or episode percept construction
  paths except type definitions or explicit tests named in the Stage 09 test
  file. Runtime prompt payloads must never include `base64_data`.

- `rg -n "image_observation|audio_observation" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator*.py`

  Expected result: no matches. Stage 09 does not change consolidation policy.

- `git diff --check`

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py`

### Prior Stage Regression Gates

- `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py`
- `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`

## Acceptance Criteria

Stage 09 is complete when:

- text-only `build_text_chat_cognitive_episode(...)` output is unchanged;
- image/audio descriptions become bounded typed percepts when supplied;
- raw media never enters prompt payloads, RAG requests, or episode percept
  content;
- RAG still uses dialog-text projection only;
- text-only `/chat`, Stage 07, and Stage 08 regression gates pass;
- no proactive output, transport, scheduler, or consolidation write behavior is
  introduced.

## Plan Self-Review

Draft self-review on 2026-05-10:

- **Coverage:** parent Stage 09 scope maps to episode, RAG, prompt selection,
  service wiring, raw-media safety, and regression checks.
- **Placeholder scan:** exact Stage 08 evidence remains blocked until Stage 08
  completes; implementation choices are otherwise fixed.
- **Contract consistency:** input-source labels and text-chat preservation match
  the parent architecture.
- **Granularity:** checkpoints split episode, RAG, prompt, service, and
  verification work.
- **Verification:** raw-media exclusion, text-only no-regression, and prior
  non-chat dry-run gates are explicit.

## Execution Handoff

Intended execution mode after approval: sequential implementation on a feature
branch forked from post-Stage-08 `main`.

Blocked next action: wait for Stage 08 completion evidence, then review this
draft against actual Stage 08 artifacts before approval.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Raw media reaches LLM | Use descriptions only and grep for `base64_data` | Static greps and prompt tests |
| Text-only chat regresses | Text-only builder snapshot and Stage 00 gate | Focused and regression tests |
| Media changes RAG unexpectedly | RAG projects dialog text only | RAG adapter test |
| Media descriptions become commands | No deterministic semantic gates | No-pre/post rule and tests |
| Multimodal support enables writes | Keep consolidator out of scope | Static grep |

## Completion Artifact Contract

When Stage 09 is complete, these artifacts must exist or be updated:

- multimodal optional inputs in `build_text_chat_cognitive_episode(...)`
- Stage 09 focused multimodal tests
- RAG adapter accepts approved multimodal user-message profiles while keeping
  dialog-text projection
- multimodal prompt selector variants
- service episode construction passes existing media descriptions
- parent ledger row for `stage_09` flipped to `completed`
- registry row flipped to `completed | completed`
- execution evidence in this plan naming branch, commit, checks, and sign-off

The artifact must not include new media summarizer LLMs, raw media prompt
payloads, pure-media turn support, proactive output, scheduler changes, or
consolidation policy changes.

## Execution Evidence

Record after implementation:

- Stage 08 evidence reread:
- Branch:
- Commit:
- Static compile:
- Static greps:
- Focused tests:
- Prior stage regression gates:
- Completion diff review:
- Lifecycle records:
- Sign-off:
