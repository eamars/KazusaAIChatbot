# resolver image only empty input bugfix plan

## Summary

- Goal: stop resolver initialization from crashing on valid image-only current turns whose envelope `body_text` is empty.
- Plan class: small
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: bigbang for resolver initialization behavior.
- Highest-risk areas: resolver bootstrap strictness and preservation of the existing image-observation cognition path.
- Acceptance criteria: image-only turns no longer raise `decontexualized_input: expected non-empty string`; empty text without a usable image observation still fails.

## Context

The confirmed 2026-06-02 incident was an image-only QQ message. The stored user row had empty `body_text`, one image attachment, and an image description already generated before relevance. The assistant fallback row followed the resolver validation exception.

Before resolver cutover, image content reached cognition through `cognitive_episode.percepts` and `media_observations`; empty `decontexualized_input` did not stop the turn. After cutover, `ensure_initial_resolver_inputs()` in `src/kazusa_ai_chatbot/cognition_resolver/state.py` rejects empty text before cognition can consume the existing image observation.

## Mandatory Skills

- `development-plan`: load before editing, reviewing, approving, executing, or signing off this plan.
- `local-llm-architecture`: load before changing resolver, cognition, graph, prompt, RAG, or dialog boundaries.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files containing CJK strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`; implementation also requires explicit user approval.
- Use `venv\Scripts\python.exe` for Python commands, `apply_patch` for manual edits, and `git status --short` before editing; do not read `.env`.
- Preserve the envelope contract: `body_text` may be empty for attachment-only input.
- Preserve the cognition contract: current-turn image content stays in `cognitive_episode` and `media_observations`.
- Do not add or change LLM calls, prompts, adapters, queueing, relevance, RAG, dialog, persistence, or the persona graph route.
- After any automatic context compaction, reread this entire plan before continuing implementation, verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, reread this entire plan before starting the next stage.
- Before completion, lifecycle status changes, merge, or sign-off, run the `Independent Code Review` gate and record the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution; if native subagents are unavailable, stop unless the user explicitly approves fallback execution.

## Must Do

- Add a narrow inline branch inside `ensure_initial_resolver_inputs()` in `src/kazusa_ai_chatbot/cognition_resolver/state.py`.
- Use non-empty string `decontexualized_input` unchanged.
- Apply image fallback only when `decontexualized_input` is a string whose stripped value is empty.
- Derive only the resolver original goal from the first existing model-visible `image_observation` percept with non-empty `content`.
- Store the derived goal only in existing resolver original-goal fields, including `original_decontexualized_input` and `goal_progress.original_goal`.
- Do not mutate `state["decontexualized_input"]`.
- Preserve existing `ResolverValidationError` behavior when `decontexualized_input` is missing, non-string, or empty with no usable image observation.
- Add focused tests for image-only success, empty no-media failure, audio-only rejection, and image-only persona graph entry before production implementation.

## Deferred

- Do not change adapters, force image descriptions into `body_text`, or make the decontextualizer synthesize text from attachments.
- Do not support audio, video, file, or generalized media fallback.
- Do not restore the pre-cutover non-resolver graph path.
- Do not change prompts, RAG, cognition prompt selection, L3, dialog, consolidation, event logging, or database schema.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Resolver initialization | bigbang | Replace the image-only hard failure with an image-observation-derived resolver goal when the percept already exists. |
| Decontextualizer and envelope | bigbang | Leave `decontexualized_input` semantics unchanged and preserve empty `body_text` for attachment-only input. |
| Validation | bigbang | Keep rejecting malformed text, empty no-media input, and audio-only input. |

Enforcement: do not add compatibility branches to the old non-resolver graph; cutover policy changes require user approval.

## Target State

- Text turns behave exactly as they do now.
- Image-only turns keep `decontexualized_input == ""`, while resolver state receives a non-empty original goal from the current image observation.
- Empty text with no usable image observation remains invalid.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Fix location | Resolver state initialization only. | The crash occurs before the resolver loop; adapters and decontextualizer already obey their contracts. |
| Image source | Read `cognitive_episode.percepts` where `input_source == "image_observation"`, `visibility == "model_visible"`, and `content` is non-empty. | This matches existing image cognition and avoids reparsing attachments. |
| State mutation | Do not rewrite `state["decontexualized_input"]`; do not add a private helper. | The field remains text-only, and one inline branch is enough. |

## Contracts And Data Shapes

- Missing or non-string `decontexualized_input`: preserve existing `ResolverValidationError`.
- Non-empty string `decontexualized_input`: pass through unchanged.
- Empty string plus usable image percept: resolver goal is `当前输入包含图片观察：<content>`.
- Usable image percept: dict with `input_source == "image_observation"`, `visibility == "model_visible"`, and non-empty string `content`.
- Audio-only, no-media, non-model-visible, or empty-content percepts do not qualify.

## LLM Call And Context Budget

- Before: zero LLM calls in resolver state initialization.
- After: zero new LLM calls; resolver context may include one additional bounded copy of the first existing image observation summary as `original_goal`.
- No raw media, prompt template change, or unbounded context is allowed.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_resolver/state.py`: inline resolver initial-goal branch.
- `tests/test_cognition_resolver_contracts.py`: image-only success, empty no-media failure, and audio-only rejection tests.
- `tests/test_cognition_resolver_persona_graph.py`: image-only graph regression.

### Keep

- Decontextualizer, cognitive episode builder, prompt selection, adapters, prompts, RAG, dialog, consolidation, database code, and persona graph route.

## Overdesign Guardrail

- Actual problem: resolver initialization rejects a valid image-only turn because its text field is empty.
- Minimal change: derive only the resolver original goal from an existing image observation when text is an empty string.
- Ownership boundaries: adapters own envelopes; media description owns image observations; resolver state owns deterministic bootstrap validation; cognition/dialog keep consuming media observations through existing prompts.
- Rejected complexity: prompt changes, new LLM calls, helper abstraction, attachment reparsing, decontextualizer synthesis, generic media fallback, legacy graph fallback, and RAG query redesign.
- Evidence threshold: broaden media bootstrap only after a confirmed non-image failure or an approved multimodal resolver plan.

## Agent Autonomy Boundaries

- Implementation freedom is limited to the inline branch and tests described in this plan.
- Changes outside `state.py` and the two listed test files require user approval.
- Do not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, preserve this plan's intent and report the discrepancy.
- If an instruction is impossible, stop and report the blocker.

## Implementation Order

1. Parent adds focused tests in `tests/test_cognition_resolver_contracts.py` for image-only success, empty no-media failure, and audio-only rejection.
2. Parent adds the image-only persona graph regression in `tests/test_cognition_resolver_persona_graph.py`.
3. Parent runs both test files before production implementation and records the expected image-only failure.
4. Parent starts one production-code subagent with this approved plan and the `state.py` ownership boundary.
5. Production-code subagent updates `state.py` only.
6. Parent reruns focused and regression tests, then the static check.
7. Parent starts one independent code-review subagent after verification passes.
8. Parent remediates review findings only inside this plan's change surface, reruns affected verification, and records evidence.

## Execution Model

- Execute only after user approval and status change to `approved` or `in_progress`.
- Parent owns orchestration, tests, verification, evidence, review remediation, lifecycle updates, and final sign-off.
- Production-code subagent owns `state.py` only and does not edit tests.
- Independent code-review subagent reviews the plan, diff, and evidence after verification and does not implement fixes.
- If native subagents are unavailable, stop unless the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - pre-implementation tests established: edit the two listed test files, run both focused test files, and record expected failure before sign-off. Sign-off: Codex, 2026-06-02.
- [x] Stage 2 - resolver bootstrap implemented: edit `state.py`, run focused tests and static check, then record changed files and output before sign-off. Sign-off: Codex, 2026-06-02.
- [x] Stage 3 - multimodal regression verified: run `venv\Scripts\python.exe -m pytest tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py -q` and record pass/fail output before sign-off. Sign-off: Codex, 2026-06-02.
- [x] Stage 4 - independent code review complete: record findings, fixes, rerun commands, and residual risk before sign-off. Sign-off: Codex single-agent fallback, 2026-06-02.

## Verification

```powershell
venv\Scripts\python.exe -m pytest tests/test_cognition_resolver_contracts.py tests/test_cognition_resolver_persona_graph.py -q
venv\Scripts\python.exe -m pytest tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py -q
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/cognition_resolver/state.py
rg "COGNITION_RESOLVER_ENABLED|stage_1_research" src/kazusa_ai_chatbot/nodes/persona_supervisor2.py
```

The final `rg` command should return no matches; its no-match nonzero exit code is acceptable. If matches appear, stop because this plan must not restore the old graph path.

## Independent Plan Review

The draft must pass review before approval. Review scope: confirmed RCA match, message-envelope and cognitive-episode contract preservation, small-plan length, no unresolved decisions, exact verification commands, and no authorized scope expansion.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. Review project style, test style, plan alignment, ownership boundaries, absence of prompt/adapter/graph-route/RAG/dialog changes, and coverage for image-only success, empty no-media rejection, and audio-only rejection.

## Acceptance Criteria

- The confirmed image-only input shape no longer raises `ResolverValidationError`.
- `state["decontexualized_input"]` remains empty for image-only input.
- Resolver original-goal fields contain the image-observation-derived goal.
- Existing cognition media path still receives image facts through `media_observations`.
- Empty no-media and audio-only inputs still fail validation.
- No adapter, prompt, RAG, dialog, consolidation, database, or old graph route behavior changes.
- All required verification commands pass.

## Execution Evidence

- Plan creation: draft file and registry row created on 2026-06-02.
- Independent plan review: completed in this drafting session; blockers and high/medium/low findings were addressed by this revision.
- Implementation authorization: user approved execution on 2026-06-02 and explicitly requested no subagents.
- Stage 1 evidence: `venv\Scripts\python.exe -m py_compile tests/test_cognition_resolver_contracts.py tests/test_cognition_resolver_persona_graph.py` exited 0; `venv\Scripts\python.exe -m pytest tests/test_cognition_resolver_contracts.py tests/test_cognition_resolver_persona_graph.py -q` failed as expected with 2 image-only failures and 20 passes.
- Stage 2 evidence: `src/kazusa_ai_chatbot/cognition_resolver/state.py` updated with the inline image-observation bootstrap branch; `venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/cognition_resolver/state.py` exited 0; `venv\Scripts\python.exe -m pytest tests/test_cognition_resolver_contracts.py tests/test_cognition_resolver_persona_graph.py -q` passed with 22 passed.
- Stage 3 evidence: `venv\Scripts\python.exe -m pytest tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py -q` passed with 16 passed.
- Stage 4 evidence: user explicitly requested no subagents, so review ran as a single-agent fallback against the full diff. Review found missing explicit contract coverage for non-model-visible and empty-content image percepts, plus one brittle image-percept index assertion; tests were tightened inside the approved test surface. Fresh verification after cleanup: `venv\Scripts\python.exe -m py_compile tests/test_cognition_resolver_contracts.py` exited 0; `venv\Scripts\python.exe -m pytest tests/test_cognition_resolver_contracts.py tests/test_cognition_resolver_persona_graph.py -q` passed with 24 passed; `venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/cognition_resolver/state.py tests/test_cognition_resolver_contracts.py tests/test_cognition_resolver_persona_graph.py` exited 0; `venv\Scripts\python.exe -m pytest tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py -q` passed with 16 passed; `git diff --check` exited 0; `rg "COGNITION_RESOLVER_ENABLED|stage_1_research" src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` returned no matches with exit code 1, as expected.
- Lifecycle closure: plan moved from `development_plans/active/bugfix/` to `development_plans/archive/completed/bugfix/` and registry updated on 2026-06-02.
