# multi source cognition architecture stage 07 reflection trigger cognition dry run plan

## Summary

- Goal: Add a reflection-triggered dry-run path that builds a
  `CognitiveEpisode(trigger_source=reflection_signal)` from promoted reflection
  context and runs shared cognition in audit-only mode.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` before editing cognition Python files that contain CJK prompt
  text.
- Overall cutover strategy: compatible for current `/chat`; dry-run only for
  reflection-triggered cognition. No dialog, delivery, consolidation,
  persistence writes, scheduler dispatch, or adapter sends are enabled.
- Highest-risk areas: treating reflection as fake user input, raw reflection
  leakage into prompts, prompt drift for current `/chat`, background LLM
  latency, and accidentally enabling writes before Stage 06 policy evidence is
  complete.
- Acceptance criteria: a reflection episode builder and dry-run audit runner
  exist; reflection prompt selection is source-aware and audit-only; `/chat`
  prompt fingerprints and regression gates remain unchanged; no durable write
  or delivery path is introduced.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Lifecycle: this draft is blocked until Stage 06 is completed and records
execution evidence proving per-write origin policy. Do not approve or execute
Stage 07 while the parent ledger row for `stage_06` is not `completed`.

## Context

Stages 01 through 05 put current `/chat` on the episode, RAG, cognition, and
consolidation-origin contracts. Stage 06 is expected to make consolidator
writes explicitly origin-gated. Stage 07 is the first non-chat trigger
admission stage, but it must remain a dry run.

Reflection input must come from promoted/gated reflection context only. Raw
hourly or daily reflection run output must not enter cognition. The existing
`build_promoted_reflection_context(...)` boundary is the only allowed context
source for this stage.

This stage deliberately does not broaden the Stage 04 RAG adapter for
reflection. Reflection dry-run state must use the existing empty projected RAG
shape plus promoted reflection context. Later work may add source-aware
reflection retrieval.

## Stage Handoff

### From Stage 06

Stage 07 expects these completed artifacts:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
- `tests/test_consolidation_origin_policy.py`
- `tests/test_consolidator_origin_policy_db_writer.py`
- evidence that denied origins produce no durable writes, no scheduler
  dispatch, and no Cache2 invalidation
- parent ledger row for `stage_06` set to `completed`

Before approval, replace this paragraph with exact Stage 06 branch, commit, and
verification results from Stage 06 `Execution Evidence`, then rerun the plan
self-review.

### To Stage 08

After Stage 07, Stage 08 can rely on:

- a non-chat dry-run entrypoint pattern that does not call `/chat`;
- reflection-source prompt selection and prompt-render tests;
- audit-only output shape for non-chat cognition runs;
- current `/chat` regression evidence after adding a future source variant.

Stage 08 must not reuse reflection-specific names for internal thought. It must
add its own trigger, input-source, and audit labels.

## Mandatory Skills

- `development-plan-writing`: preserve staged lifecycle and handoff.
- `local-llm-architecture`: keep the dry-run path bounded, background-only, and
  source-aware.
- `no-prepost-user-input`: do not reinterpret reflection output as user facts,
  user instructions, or accepted commitments.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing L1/L2/L3 cognition modules containing CJK
  prompt constants.

## Mandatory Rules

- Execute only from a feature branch forked from post-Stage-06 `main`.
- Keep edits inside the approved Change Surface.
- Use PowerShell `-LiteralPath '...'` for filesystem paths that may contain
  spaces; prefer repo-relative paths in commands.
- Do not route reflection through `/chat`, `persona_supervisor2(...)`,
  service adapters, message-envelope intake, dialog, or post-turn persistence.
- Do not call `call_consolidation_subgraph(...)`.
- Do not call `save_conversation(...)`, `save_assistant_message(...)`,
  dispatcher transport, scheduler dispatch, or adapter send functions.
- Do not broaden Stage 04 RAG adapter runtime acceptance for
  `reflection_signal`.
- Do not add a new live `/chat` LLM call or increase current `/chat` prompt
  payloads.
- Do not modify existing `text_chat_user_message` prompt constants except for
  mechanical lookup-table wiring. Their byte fingerprints must remain stable.
- Reflection dry runs may use background cognition LLM calls only when an
  explicit busy probe says primary interaction is not busy.
- The dry-run runner must accept an injected cognition callable for tests.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.

## Must Do

- Add a reflection episode builder that consumes only
  `PromotedReflectionContext`.
- Add reflection prompt selection for exactly:
  `trigger_source="reflection_signal"`,
  `input_sources=["reflection_artifact"]`, and
  `output_mode in {"think_only", "preview", "silent"}`.
- Add reflection dry-run prompt-map entries for every L1/L2/L3 handler without
  changing current text-chat prompt bytes.
- Add a dry-run audit runner that builds source-aware cognition state, calls
  the injected cognition callable once, and returns an audit dict.
- Add tests proving busy dry runs call no cognition LLMs and non-busy dry runs
  call the cognition callable once.
- Add prompt-render tests proving reflection prompts receive reflection
  percept context and do not receive `cognitive_episode`, `trigger_source`, or
  raw reflection run documents in model-facing payloads.
- Run every Verification command and record evidence.
- Leave lifecycle rows as `draft | blocked` until Stage 06 completion evidence
  is carried forward and the plan is explicitly approved.

## Deferred

- Reflection durable writes.
- Reflection RAG retrieval.
- Reflection output delivery, dialog generation, scheduled action requests, or
  proactive sends.
- Raw reflection run ingestion into cognition.
- Changes to memory promotion, daily reflection prompts, interaction-style
  updates, scheduler policy, adapters, or transport.
- Internal thought, action latch, multimodal inputs, and proactive output.

## Cutover Policy

Overall strategy: compatible for `/chat`, dry-run-only for reflection.

| Area | Policy | Instruction |
|---|---|---|
| Current `/chat` | compatible | Preserve current graph, prompt bytes, RAG request shape, dialog, and consolidation behavior. |
| Reflection entrypoint | bigbang | Add one dry-run entrypoint. No fallback through `/chat`. |
| Reflection writes | bigbang | No writes, no consolidation, no scheduler, no cache invalidation, and no adapter sends. |
| Reflection RAG | compatible | Keep Stage 04 future-source rejection; use empty projected RAG shape in dry-run state. |

Rollback path: remove the reflection dry-run module, remove reflection prompt
variant wiring, remove focused tests, and restore selector to text-chat-only.
No database rollback is required.

## Agent Autonomy Boundaries

Allowed implementation choices:

- local variable names inside tests;
- assertion ordering;
- fixture helper names.

Not allowed:

- adding feature flags, fallback paths, compatibility shims, wrapper layers, or
  alternate graph entrypoints;
- adding persistent audit storage;
- editing service startup, reflection worker scheduling, adapter delivery, RAG
  internals, dialog, or consolidation;
- creating extra source labels, output modes, or prompt variants;
- using raw hourly/daily reflection run documents as model input;
- adding raising-only helpers or pass-through wrappers.

If a required file outside Change Surface must change, stop and update this
plan before continuing.

## Target State

The reflection dry-run path is:

```text
promoted reflection context
-> build_reflection_signal_cognitive_episode(...)
-> run_reflection_cognition_dry_run(...)
-> call_cognition_subgraph_func(dry_run_state)
-> ReflectionCognitionDryRunAudit
```

The dry-run state must not be created by calling `/chat` or
`persona_supervisor2(...)`. Compatibility fields required by the current
cognition subgraph may be populated with explicit reflection dry-run placeholder
values, but model-facing reflection prompt payloads must frame the input as a
reflection artifact, not as a user utterance.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Reflection source | Use only `build_promoted_reflection_context(...)` output. | Raw reflection runs are not safe model-facing cognition input. |
| Entry graph | Add a dry-run runner instead of reusing `/chat`. | Reflection is a different trigger source and must not pretend to be a user message. |
| RAG | Use empty projected RAG shape in Stage 07. | Keeps Stage 04 future-source RAG rejection intact while testing shared cognition admission. |
| Output | Return audit dict only. | Later stages own persistence and proactive behavior. |
| Prompt bytes | Preserve text-chat prompt fingerprints. | `/chat` regression risk must remain measurable. |

## Change Surface

### Create

- `src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py` — reflection
  episode builder and dry-run audit runner.
- `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py` —
  reflection builder, selector, prompt-render, dry-run, and no-write tests.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
  — add the single reflection dry-run prompt variant.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py` — add the
  L1 reflection prompt-map entry without changing text-chat prompt bytes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` — add L2
  reflection prompt-map entries without changing text-chat prompt bytes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` — add L3
  reflection prompt-map entries without changing text-chat prompt bytes.
- `development_plans/active/short_term/multi_source_cognition_architecture_stage_07_reflection_trigger_cognition_dry_run_plan.md`
  — record progress and evidence only after approval.
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
  and `development_plans/README.md` — flip lifecycle rows only after approval
  and completion gates.

### Keep

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator*.py`
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`
- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
- adapter, dispatcher, scheduler, and database modules

## Implementation Order

1. Reread Stage 06 `Execution Evidence`.
2. Add reflection episode-builder tests.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_07_reflection_dry_run.py -q`
   - Expected before implementation: import error for the dry-run module.
3. Implement `reflection_cycle/cognition_dry_run.py`.
   - Verify: compile the module and rerun Step 2 tests.
4. Add selector tests for reflection dry-run variant and existing rejection
   cases.
5. Wire the selector variant and prompt maps.
   - Verify: Stage 03 selector tests plus Stage 07 tests pass.
6. Add dry-run runner tests for busy probe, no writes, and one injected
   cognition call.
7. Add prompt-render tests with mocked LLMs for the reflection variant.
8. Run the full Verification section.
9. Record evidence and sign off only after each checklist stage passes.

## Progress Checklist

- [ ] Stage 1 - prerequisite evidence carried forward.
  - Covers: Step 1.
  - Verify: Stage 06 row is `completed`.
  - Evidence: Stage 06 branch, commit, and test results recorded.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 - reflection episode builder complete.
  - Covers: Steps 2-3.
  - Verify: focused builder tests pass after expected red import failure.
  - Evidence: red/green results recorded.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 3 - reflection prompt selection and prompt maps complete.
  - Covers: Steps 4-5.
  - Verify: Stage 03 and Stage 07 selector/render tests pass.
  - Evidence: prompt fingerprint and test output recorded.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 4 - audit-only dry-run runner complete.
  - Covers: Steps 6-7.
  - Verify: no-write and prompt-render tests pass.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 5 - full verification complete.
  - Covers: Step 8.
  - Verify: every Verification command passes or has an allowed no-match exit.
  - Evidence: command output recorded.
  - Handoff: Stage 08 draft may be reviewed after lifecycle update.
  - Sign-off: `<agent/date>` after verification.

## Verification

Run from repository root with the project virtual environment.

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\reflection_cycle\cognition_dry_run.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_multi_source_cognition_stage_07_reflection_dry_run.py`

### Static Greps

- `rg -n "call_consolidation_subgraph|save_conversation|save_assistant_message|dispatcher\\.dispatch|runtime\\.invalidate|get_rag_cache2_runtime" src\kazusa_ai_chatbot\reflection_cycle\cognition_dry_run.py`

  Expected result: no matches. `rg` exit code `1` is acceptable.

- `rg -n "\"reflection_signal\"|\"reflection_artifact\"" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py`

  Expected result: no matches. Stage 07 must not wire reflection into service,
  persona supervisor, or RAG.

- `git diff --check`

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_07_reflection_dry_run.py`

### Prior Stage Regression Gates

- `venv\Scripts\python -m pytest tests\test_consolidation_origin_policy.py tests\test_consolidator_origin_policy_db_writer.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`

## Acceptance Criteria

Stage 07 is complete when:

- reflection dry-run episode construction exists and validates;
- prompt selection supports only the approved reflection dry-run source tuple;
- text-chat prompt fingerprints and regression gates remain unchanged;
- dry-run audit calls injected cognition at most once and only when not busy;
- no dialog, consolidation, persistence, scheduler, cache, adapter, or RAG
  runtime path is enabled for reflection;
- parent ledger and registry rows are completed only after verification passes.

## Plan Self-Review

Draft self-review on 2026-05-10:

- **Coverage:** parent Stage 07 scope maps to builder, selector, prompt-render,
  audit-runner, and regression checks.
- **Placeholder scan:** exact Stage 06 evidence remains blocked until Stage 06
  completes; no implementation decision is delegated to the agent.
- **Contract consistency:** trigger source, input source, and output modes match
  the parent architecture.
- **Granularity:** checkpoints split prerequisite evidence, builder, prompt
  selection, dry-run runner, and verification.
- **Verification:** no-write, no-service-wiring, prompt non-consumption, and
  prior-stage gates are explicit.

## Execution Handoff

Intended execution mode after approval: sequential implementation on a feature
branch forked from post-Stage-06 `main`.

Blocked next action: wait for Stage 06 completion evidence, then review this
draft against actual Stage 06 artifacts before approval.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Reflection is treated as user text | Separate builder and dry-run runner; no `/chat` entry | Static greps and dry-run tests |
| Raw reflection leaks into prompts | Use only promoted reflection context | Builder and prompt-render tests |
| Current `/chat` prompt drift | Preserve prompt bytes and rerun baseline | Prompt fingerprint and Stage 00 tests |
| Background work affects live latency | Busy probe and injected callable | Busy dry-run test |
| Reflection writes memory | No consolidation or persistence calls | Static greps and no-write tests |

## Completion Artifact Contract

When Stage 07 is complete, these artifacts must exist or be updated:

- `src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py`
- `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py`
- reflection dry-run selector variant
- audit-only dry-run return shape
- parent ledger row for `stage_07` flipped to `completed`
- registry row flipped to `completed | completed`
- execution evidence in this plan naming branch, commit, checks, and sign-off

The artifact must not include service wiring, delivery, consolidation writes,
database persistence, scheduler dispatch, RAG broadening, or proactive output.

## Execution Evidence

Record after implementation:

- Stage 06 evidence reread:
- Branch:
- Commit:
- Static compile:
- Static greps:
- Focused tests:
- Prior stage regression gates:
- Completion diff review:
- Lifecycle records:
- Sign-off:
