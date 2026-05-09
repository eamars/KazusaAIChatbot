# multi source cognition architecture stage 08 internal thought cognition dry run plan

## Summary

- Goal: Add an internal-thought dry-run path that builds
  `CognitiveEpisode(trigger_source=internal_thought)` from private cognition
  residue and action-latch input without public output or conversation-history
  pollution.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` before editing cognition Python files that contain CJK prompt
  text.
- Overall cutover strategy: compatible for current `/chat`; dry-run only for
  internal-thought cognition. No recursive loop, dialog, delivery,
  consolidation, persistence writes, scheduler dispatch, or adapter sends are
  enabled.
- Highest-risk areas: private thought leaking into public prompts or history,
  adding an unbounded loop, reusing reflection semantics for a different
  trigger, and enabling generated thought to write memory.
- Acceptance criteria: private residue and action-latch contracts exist;
  internal-thought prompt selection is source-aware and audit-only; the dry-run
  runner performs at most one cognition call; `/chat` and Stage 07 gates still
  pass.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Lifecycle: this draft is blocked until Stage 07 completes and records execution
evidence. Do not approve or execute Stage 08 while the parent ledger row for
`stage_07` is not `completed`.

## Context

Stage 07 introduces the first non-chat dry-run entrypoint for promoted
reflection artifacts. Stage 08 applies the same source-aware pattern to private
internal thought and action residue. This is not an approval for autonomous
loops or proactive output.

The experimental cognition reference contains monologue and action-latch
concepts, but production code must not import experiment modules. This stage
translates only the contract idea: a private internal stimulus can enter shared
cognition in a single dry-run pass and produce an audit record.

## Stage Handoff

### From Stage 07

Stage 08 expects these completed artifacts:

- `src/kazusa_ai_chatbot/reflection_cycle/cognition_dry_run.py`
- reflection dry-run prompt selector variant
- audit-only dry-run return shape
- evidence that Stage 07 introduced no service, dialog, RAG, consolidation,
  persistence, scheduler, or adapter wiring
- parent ledger row for `stage_07` set to `completed`

Before approval, replace this paragraph with exact Stage 07 branch, commit, and
verification results from Stage 07 `Execution Evidence`, then rerun the plan
self-review.

### To Stage 09

After Stage 08, Stage 09 can rely on:

- two non-chat dry-run trigger patterns with separate source labels;
- prompt selector support for source-specific variants without `/chat`
  regression;
- a hard rule that private cognition residue is audit-only until a later
  approved policy enables writes or output.

Stage 09 must not use internal-thought residue as media description input and
must keep image/audio percepts separate from private monologue.

## Mandatory Skills

- `development-plan-writing`: preserve staged lifecycle and handoff.
- `local-llm-architecture`: enforce bounded background cognition and explicit
  source ownership.
- `no-prepost-user-input`: do not reinterpret generated thought as user
  instruction, preference, fact, or commitment.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing L1/L2/L3 cognition modules containing CJK
  prompt constants.

## Mandatory Rules

- Execute only from a feature branch forked from post-Stage-07 `main`.
- Keep edits inside the approved Change Surface.
- Use PowerShell `-LiteralPath '...'` for filesystem paths that may contain
  spaces; prefer repo-relative paths in commands.
- Do not route internal thought through `/chat`, `persona_supervisor2(...)`,
  service adapters, message-envelope intake, dialog, or post-turn persistence.
- Do not call `call_consolidation_subgraph(...)`.
- Do not call scheduler, dispatcher, cache invalidation, adapter send, or
  conversation persistence functions.
- Do not add a recursive loop. Stage 08 dry run is exactly one cognition
  invocation per audit request.
- Do not store internal thought in public conversation history, user memory,
  character state, group scene residue, or normal RAG evidence.
- Do not broaden Stage 04 RAG adapter runtime acceptance for
  `internal_thought`.
- Do not change existing `text_chat_user_message` or Stage 07 reflection prompt
  bytes.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, reread this entire
  plan before starting the next stage.

## Must Do

- Create private internal-thought residue and action-latch TypedDict contracts.
- Add an internal-thought episode builder for exactly:
  `trigger_source="internal_thought"`,
  `input_sources=["internal_monologue"]`, and
  `output_mode in {"think_only", "preview", "silent"}`.
- Add prompt selection for exactly one internal-thought dry-run variant.
- Add internal-thought prompt-map entries for every L1/L2/L3 handler without
  changing text-chat or reflection prompt bytes.
- Add a dry-run audit runner that calls the injected cognition callable at most
  once and returns an audit dict.
- Add tests proving no public or durable write path is called.
- Add tests proving private residue never appears in `chat_history_recent`,
  `final_dialog`, `consolidation_state`, RAG payloads, or adapter output.
- Run every Verification command and record evidence.

## Deferred

- Enabling internal-thought durable writes.
- Enabling loops, retries, action execution, proactive output, or scheduled
  action requests.
- Merging private cognition residue into public scene residue.
- RAG retrieval for internal thought.
- Dialog generation from internal thought.
- Multimodal input sources and proactive transport.

## Cutover Policy

Overall strategy: compatible for `/chat`, dry-run-only for internal thought.

| Area | Policy | Instruction |
|---|---|---|
| Current `/chat` | compatible | Preserve graph, prompts, RAG, dialog, consolidation, and persistence. |
| Internal-thought entrypoint | bigbang | Add one dry-run entrypoint. No fallback through `/chat`. |
| Loop policy | bigbang | Exactly one cognition invocation. No recursive loop or retry. |
| Writes and output | bigbang | No dialog, no persistence, no scheduler, no cache, and no adapter sends. |

Rollback path: remove the internal-thought dry-run module, remove selector and
prompt-map additions for the internal-thought variant, and remove focused
tests. No database rollback is required.

## Agent Autonomy Boundaries

Allowed implementation choices:

- local fixture helper names;
- local variable names inside tests;
- assertion ordering.

Not allowed:

- adding feature flags, fallback paths, loop runners, queues, background
  workers, or alternate graph entrypoints;
- importing from `experiments/cognition_core_next/`;
- creating extra source labels, output modes, prompt variants, or persistence
  schemas;
- touching service, adapters, dispatcher, scheduler, RAG internals, dialog, or
  consolidation;
- adding raising-only helpers or pass-through wrappers.

If the agent needs a file outside Change Surface, stop and update this plan
before continuing.

## Target State

The internal-thought dry-run path is:

```text
private internal residue + optional action latch
-> build_internal_thought_cognitive_episode(...)
-> run_internal_thought_cognition_dry_run(...)
-> call_cognition_subgraph_func(dry_run_state)
-> InternalThoughtCognitionDryRunAudit
```

The audit record may include source ids, output mode, selected prompt keys, and
normalized cognition output. It must not include raw public message history
outside what the caller explicitly supplies for audit, and it must not be
written to a database in this stage.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Trigger ownership | Use `internal_thought`, not `reflection_signal`. | Internal residue is generated by runtime cognition, not reflection promotion. |
| Input source | Use only `internal_monologue`. | Keeps private monologue separate from media, memory, and dialog text. |
| Loop policy | One invocation only. | Bounded dry-run evidence before any loop design. |
| Persistence | Return audit dict only. | Stage 06 denies non-chat writes and Stage 10 owns output. |
| Experiment code | Translate concepts, do not import. | Experiment modules are reference only. |

## Change Surface

### Create

- `src/kazusa_ai_chatbot/internal_thought_cognition.py` — private residue,
  action-latch, episode-builder, and dry-run audit contracts.
- `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py` —
  builder, selector, prompt-render, bounded-run, and no-public-leak tests.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
  — add the single internal-thought dry-run prompt variant.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py` — add the
  L1 internal-thought prompt-map entry without changing existing prompt bytes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py` — add L2
  internal-thought prompt-map entries without changing existing prompt bytes.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py` — add L3
  internal-thought prompt-map entries without changing existing prompt bytes.
- lifecycle rows in the parent plan and registry after completion only.

### Keep

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator*.py`
- `src/kazusa_ai_chatbot/reflection_cycle/*.py`
- adapter, dispatcher, scheduler, and database modules
- `experiments/cognition_core_next/**`

## Implementation Order

1. Reread Stage 07 `Execution Evidence`.
2. Add internal-thought builder tests.
   - Verify:
     `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py -q`
   - Expected before implementation: import error for
     `internal_thought_cognition`.
3. Implement `internal_thought_cognition.py`.
4. Add selector tests for the internal-thought dry-run variant.
5. Wire selector and prompt-map entries.
6. Add bounded dry-run runner and no-public-leak tests.
7. Run the full Verification section.
8. Record evidence and sign off.

## Progress Checklist

- [ ] Stage 1 - prerequisite evidence carried forward.
  - Covers: Step 1.
  - Verify: Stage 07 row is `completed`.
  - Evidence: Stage 07 branch, commit, and test results recorded.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after evidence is recorded.
- [ ] Stage 2 - internal thought episode contract complete.
  - Covers: Steps 2-3.
  - Verify: focused builder tests pass after expected red import failure.
  - Evidence: red/green results recorded.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 3 - prompt selection and prompt maps complete.
  - Covers: Steps 4-5.
  - Verify: Stage 03, Stage 07, and Stage 08 selector/render tests pass.
  - Evidence: prompt fingerprint and test output recorded.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 4 - bounded dry-run runner complete.
  - Covers: Step 6.
  - Verify: bounded-run and no-public-leak tests pass.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification.
- [ ] Stage 5 - full verification complete.
  - Covers: Step 7.
  - Verify: every Verification command passes or has an allowed no-match exit.
  - Evidence: command output recorded.
  - Handoff: Stage 09 draft may be reviewed after lifecycle update.
  - Sign-off: `<agent/date>` after verification.

## Verification

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\internal_thought_cognition.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_prompt_selection.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`

### Static Greps

- `rg -n "save_conversation|save_assistant_message|call_consolidation_subgraph|dispatcher\\.dispatch|runtime\\.invalidate|get_rag_cache2_runtime" src\kazusa_ai_chatbot\internal_thought_cognition.py`

  Expected result: no matches. `rg` exit code `1` is acceptable.

- `rg -n "\"internal_thought\"|\"internal_monologue\"" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py`

  Expected result: no matches. Stage 08 must not wire internal thought into
  service, persona supervisor, or RAG.

- `git diff --check`

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_08_internal_thought_dry_run.py`

### Prior Stage Regression Gates

- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_07_reflection_dry_run.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`

## Acceptance Criteria

Stage 08 is complete when:

- internal-thought residue, action-latch, episode, and audit contracts exist;
- selector supports only the approved internal-thought dry-run tuple;
- dry-run runner invokes cognition at most once and writes nothing;
- private residue does not appear in public dialog, conversation history,
  RAG runtime, consolidation, scheduler, adapter, or cache paths;
- `/chat`, Stage 07, and Stage 03 regression gates still pass.

## Plan Self-Review

Draft self-review on 2026-05-10:

- **Coverage:** parent Stage 08 scope maps to contracts, selector, prompt-map,
  dry-run, and leak-prevention tests.
- **Placeholder scan:** exact Stage 07 evidence remains blocked until Stage 07
  completes; implementation choices are otherwise fixed.
- **Contract consistency:** trigger source, input source, output modes, and
  dry-run status match the top-level architecture.
- **Granularity:** checkpoints split prerequisite evidence, contract,
  selector, runner, and verification.
- **Verification:** no-loop, no-write, no-public-leak, and prior-stage gates
  are explicit.

## Execution Handoff

Intended execution mode after approval: sequential implementation on a feature
branch forked from post-Stage-07 `main`.

Blocked next action: wait for Stage 07 completion evidence, then review this
draft against actual Stage 07 artifacts before approval.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Private thought leaks publicly | No service/dialog/consolidation wiring | Static greps and no-public-leak tests |
| Unbounded self-loop appears | Exactly one invocation; no runner loop | Bounded-run test |
| Reflection and internal sources blur | Separate module, labels, and tests | Selector tests |
| Generated thought writes memory | No consolidation or persistence calls | Static greps |
| `/chat` prompt drift | Preserve existing prompt bytes | Stage 00 and Stage 03 gates |

## Completion Artifact Contract

When Stage 08 is complete, these artifacts must exist or be updated:

- `src/kazusa_ai_chatbot/internal_thought_cognition.py`
- `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py`
- internal-thought dry-run selector variant
- audit-only dry-run return shape
- parent ledger row for `stage_08` flipped to `completed`
- registry row flipped to `completed | completed`
- execution evidence in this plan naming branch, commit, checks, and sign-off

The artifact must not include service wiring, persistence, database writes,
scheduler dispatch, adapter output, RAG broadening, recursive loops, or
proactive behavior.

## Execution Evidence

Record after implementation:

- Stage 07 evidence reread:
- Branch:
- Commit:
- Static compile:
- Static greps:
- Focused tests:
- Prior stage regression gates:
- Completion diff review:
- Lifecycle records:
- Sign-off:
