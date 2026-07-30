# cognition core v2 prewarm mention content query bugfix plan

## Summary

- Goal: prevent cycle-zero shared-memory prewarm from searching the active
  character's typed mention as content while retaining the authored content
  that follows the mention.
- Plan class: `medium`.
- Status: `completed`.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `cjk-safety`, and
  `test-style-and-execution`.
- Overall cutover strategy: big-bang replacement of the prewarm query-selection
  behavior with no compatibility path.
- Highest-risk areas: deleting a genuine character subject reference, stripping
  another participant's mention, and launching retrieval for a mention-only
  turn.
- Acceptance criteria: an active-character mention plus `#napcat` sends only
  `#napcat` to the prewarm worker; an active-character mention-only turn skips
  the worker; ordinary content and non-character mentions remain unchanged.

## Context

Run `5dfe31dce56c42ada3304c3b39535916` contains the typed user message
`@一之濑明日奈 #napcat`. Its message envelope correctly records the first token
as `mentions[].entity_kind="bot"` and the remainder as authored content. The
cycle-zero prewarm restored by
`cognition_core_v2_p0_context_reconnection_bugfix_plan.md` currently passes
`state["decontextualized_input"]` directly to
`PersistentMemorySearchAgent.run(...)`. That bypasses the historical
mention/content guard and permits the memory worker to center its query on the
character name.

The completed
`rag_reply_mention_and_vague_input_plan.md` established that current-bot
addressing metadata is not retrieval content. The completed
`typed_message_envelope_stage2_plan.md` supplied the structural fields needed
to enforce that boundary. This plan carries that contract into the later
prewarm path.

## Mandatory Skills

- `development-plan`: govern plan execution, evidence, review, and lifecycle.
- `local-llm-architecture`: preserve the RAG/cognition ownership boundary and
  response-path call budget.
- `no-prepost-user-input`: limit deterministic processing to typed structural
  mention sanitation; do not classify user intent or semantic content.
- `py-style`: govern Python production and review changes.
- `cjk-safety`: govern Python tests containing the captured CJK mention.
- `test-style-and-execution`: govern deterministic test construction and runs.

## Mandatory Rules

- Use `venv\Scripts\python.exe` for Python and pytest commands.
- Use `apply_patch` for manual edits.
- Do not read `.env`.
- Keep adapters, decontextualization prompts, full RAG3 resolution, persistent
  memory worker prompts, and Cognition Core V2 prompts unchanged.
- Treat `prompt_message_context.mentions` as structural provenance. Strip only
  readable tokens corresponding to rows whose `entity_kind` is `bot`.
- Do not strip a plain character name without its visible mention prefix.
- Do not strip `user`, `platform_role`, `channel`, `everyone`, or `unknown`
  mentions.
- Preserve all remaining authored query text and its language.
- If stripping leaves no non-whitespace query, return the existing empty
  projected RAG payload without constructing or calling the worker.
- Do not add keyword intent classification, character-name aliases, a new LLM
  call, retry, feature flag, fallback path, or compatibility shim.
- After any automatic context compaction, reread this entire plan before
  implementation, verification, handoff, or final reporting.
- After signing off a major checklist stage, reread this entire plan before
  starting the next stage.
- Before completion or sign-off, run the `Independent Code Review` gate and
  record its result in `Execution Evidence`.
- Use parent-led native subagent execution. If native subagent capability is
  unavailable, stop unless the user explicitly approves fallback execution.

## Must Do

- Add deterministic regression tests for active-character tag-plus-content,
  active-character tag-only, non-character mention preservation, and plain
  character-name preservation.
- Add a boundary regression proving that a longer authored literal beginning
  with the same `@display_name` survives before the exact typed mention.
- Derive the shared-memory prewarm task from decontextualized input after
  removing only typed active-character mention tokens.
- Skip prewarm retrieval when the resulting task is empty.
- Document the prewarm query boundary in the cognition resolver ICD.
- Run focused and adjacent deterministic regression tests.
- Complete independent code review before lifecycle completion.

## Deferred

- Do not change full RAG3 local-context query planning.
- Do not change decontextualizer prompt behavior.
- Do not alter adapter `body_text` mention preservation.
- Do not tune persistent-memory generator or judge prompts.
- Do not add generalized punctuation or natural-language cleanup.
- Do not replay the captured production turn through a live model in this
  deterministic boundary plan.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Prewarm task selection | bigbang | Replace verbatim decontextualized input with typed bot-mention sanitation. |
| Empty semantic task | bigbang | Skip worker execution and return the canonical empty result. |
| Other RAG paths | compatible | Retain current behavior unchanged. |
| Tests and ICD | bigbang | Add the new contract directly without legacy expectations. |

## Target State

```text
decontextualized_input + prompt_message_context.mentions
  -> remove each typed bot mention's readable @display_name token once
  -> trim surrounding whitespace
  -> empty: return canonical empty rag_result
  -> non-empty: build normal RAG request with that task
               and call PersistentMemorySearchAgent once
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Structural owner | Typed `mentions` metadata | It distinguishes addressing from authored content without semantic guessing. |
| Query source | Decontextualized input | It preserves resolved referents and content improvements already produced upstream. |
| Removal scope | Exact readable bot mention token | Plain character names can be legitimate subjects and must survive. |
| Empty result | Skip worker | A mention-only address has no retrieval target. |
| Module owner | `cognition_resolver.capabilities` | The defect exists only in its cycle-zero prewarm bypass. |

## Change Surface

### Modify

- `tests/test_shared_memory_prewarm.py`: add the focused deterministic
  regression contract.
- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`: select the
  prewarm task from typed mention structure and skip empty tasks.
- `src/kazusa_ai_chatbot/cognition_resolver/README.md`: document the task
  selection invariant.
- `development_plans/README.md`: register and later archive this plan.

### Create

- This active bugfix plan.

### Keep

- RAG adapter request shape, worker contract, model routes, prompts, cache
  schema, message-envelope schema, and cognition connector call site.

## Overdesign Guardrail

- Actual problem: prewarm can search the active character mention instead of
  the content in the same typed message.
- Minimal change: sanitize the one prewarm task using existing typed bot
  mention rows and skip an empty task.
- Ownership boundaries: adapters identify mention structure; deterministic
  prewarm code removes addressing tokens; the memory worker retains semantic
  retrieval judgment.
- Rejected complexity: new envelope fields, offsets, aliases, regex
  registries, prompt changes, helper agents, retries, modes, and fallbacks.
- Evidence threshold: a captured failure that cannot be represented by exact
  typed bot mention tokens is required before widening sanitation.

## Agent Autonomy Boundaries

- The responsible agent may choose only local mechanics that preserve the exact
  contracts above.
- The responsible agent must not introduce new architecture, compatibility
  layers, fallbacks, or unrelated cleanup.
- Production changes are limited to `capabilities.py` and its resolver ICD.
- The production-code subagent must search for equivalent existing behavior
  before adding a helper and report the search result.
- If the plan and current code disagree, stop and report the discrepancy.

## Implementation Order

1. Parent adds the focused cases to `tests/test_shared_memory_prewarm.py`.
2. Parent runs the focused cases and records two expected pre-fix failures
   plus the non-character-mention preservation baseline.
3. Parent starts exactly one production-code subagent with ownership limited to
   `capabilities.py` and `cognition_resolver/README.md`.
4. Production-code subagent implements typed bot-mention task sanitation and
   empty-task skipping.
5. Parent reruns focused tests, the full prewarm test module, connector prewarm
   tests, syntax checks, and static greps.
6. Parent starts exactly one independent code-review subagent.
7. Parent remediates in-scope findings, reruns affected checks, records
   evidence, marks the plan completed, and moves it to the completed archive.

## Execution Model

- Parent owns orchestration, test code, verification, evidence, review
  remediation, lifecycle updates, and final sign-off.
- Production-code subagent: exactly one native subagent after the failing test
  contract; production and ICD files only; no test edits.
- Independent code-review subagent: exactly one native subagent after
  verification; review only; no implementation.

## Progress Checklist

- [x] Stage 1 - focused test contract established.
  - Verify: tag-plus-content and tag-only fail for the expected query/call
    reasons before production implementation; the non-character mention guard
    passes.
  - Evidence: record command and failures below.
  - Handoff: start the production-code subagent.
  - Sign-off: Codex parent / 2026-07-31.
- [x] Stage 2 - production boundary implemented.
  - Verify: focused and adjacent deterministic tests pass; Python files parse.
  - Evidence: record changed files and command results below.
  - Handoff: start independent review.
  - Sign-off: Codex parent / 2026-07-31.
- [x] Stage 3 - independent code review and completion.
  - Verify: review approves plan alignment and affected checks pass after any
    remediation.
  - Evidence: record reviewer, findings, fixes, and reruns below.
  - Handoff: archive plan and report completion.
  - Sign-off: Codex parent / 2026-07-31.

## Verification

- `venv\Scripts\python.exe -m pytest tests/test_shared_memory_prewarm.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_persona_supervisor2_cognition_prewarm.py -q`
- `venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/cognition_resolver/capabilities.py tests/test_shared_memory_prewarm.py`
- `rg -n "PersistentMemorySearchAgent|run_first_cycle_shared_memory_prewarm" src/kazusa_ai_chatbot/cognition_resolver/capabilities.py tests/test_shared_memory_prewarm.py`
  must show one production worker call site and the focused tests.
- `git diff --check` must exit `0`; Windows safe-CRLF conversion warnings are
  allowed because they do not report whitespace errors.

## LLM Call And Context Budget

- Before: cycle-zero prewarm makes zero or one response-path
  `RAG_SUBAGENT_LLM` worker run with the full decontextualized input as task;
  the worker remains capped at one attempt.
- After: call count, route, attempt cap, context fields, completion budget, and
  blocking behavior are unchanged for non-empty content. Mention-only input
  changes from one run to zero. Task characters can only decrease. No context
  cap changes and no new response-path calls are introduced.

## Independent Code Review

After verification, one native review subagent must inspect the active plan,
full diff, commands, and execution evidence for typed-message ownership,
over-stripping, empty-task behavior, Python/CJK/test style, unrelated changes,
and accurate lifecycle records. The reviewer reports findings and approval
without editing files. The parent may fix findings only inside the declared
change surface, then reruns affected verification.

## Acceptance Criteria

This plan is complete when:

- `@一之濑明日奈 #napcat` with a typed bot mention calls the worker with
  `#napcat`.
- A typed bot mention-only query does not call the worker and returns the
  canonical empty RAG result.
- A non-character mention remains in the worker task.
- A plain character name without a typed mention remains in the worker task.
- Existing ordinary prewarm and connector behavior remains green.
- Independent review approves the bounded implementation.

## Execution Evidence

- Baseline: clean worktree before this plan; the plan, registry row, and focused
  tests are the first changes.
- Focused tests: `venv\Scripts\python.exe -m pytest` against the three focused
  cases produced the required baseline: tag-plus-content failed with actual
  task `@一之濑明日奈 #napcat`; tag-only failed because one worker call occurred;
  non-character mention preservation passed. Result: `2 failed, 1 passed`.
- Regression tests: after review remediation,
  `tests/test_shared_memory_prewarm.py` passed `9/9`, including plain-name and
  exact-token-boundary guards; adjacent
  `tests/test_persona_supervisor2_cognition_prewarm.py` passed `8/8`.
- Static checks: `py_compile` passed for production and test files; the static
  grep found exactly one production `PersistentMemorySearchAgent().run` call;
  `git diff --check` exited `0` with Windows safe-CRLF conversion warnings and
  no whitespace errors.
- Independent review: Epicurus rejected the first diff because substring
  removal could damage a longer authored literal and leave the real mention.
  Parent remediation changed removal to whitespace-bounded exact-token
  matching and added the required regression. Epicurus re-reviewed the full
  diff and evidence, found no remaining issues, and approved completion and
  archival.
- Residual risks: typed mentions contain no occurrence offsets, so multiple
  identical visible tokens remain structurally ambiguous. Sanitation also
  depends on decontextualization preserving the visible `@display_name` form.
- Archived: 2026-07-31 after independent review approval.
