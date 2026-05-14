# quote aware rag sequence plan

## Summary

- Goal: make RAG use native replied/quoted text before researching the fresh
  user message, so reply turns do not lose critical quoted facts.
- Plan class: large.
- Status: completed.
- Mandatory skills: `py-style`, `test-style-and-execution`,
  `local-llm-architecture`, `development-plan-writing`.
- Overall cutover strategy: bigbang for the live text-chat RAG call path.
- Highest-risk areas: response-path latency, local-LLM handling of new
  quote-grounding query text, pass-local context leakage, unresolved RAG passes
  overriding useful quote evidence, and stale experiment code.
- Acceptance criteria: production text-chat RAG uses the quote-aware sequence
  when `reply_context.reply_excerpt` exists, no-quote behavior remains a
  single pass, existing public RAG shape stays stable, deterministic matrix
  tests pass, live query-prefix evals are run one case at a time, existing
  RAG/referent tests pass, and the experiment harness is removed.

## Context

Observed failure: a QQ reply turn included `reply_context.reply_excerpt` with
the active character's prior BYD Shark text, but RAG planned from the fresh
decontextualized message only: `1.5T皮卡 好陌生的詞彙`. The resulting search
answered generic `1.5T皮卡` instead of staying anchored to the quoted BYD
Shark content.

Relevant existing path:

```text
stage_1_research
  -> build_text_chat_rag_request(...)
  -> call_rag_supervisor(original_query=rag_request["original_query"], ...)
  -> rag_initializer
  -> rag_dispatcher
  -> rag_evaluator/finalizer
  -> cognition
  -> dialog
```

`reply_context` already reaches the brain service and is forwarded into RAG
context. The issue is not adapter hydration. The issue is that the RAG
supervisor is called once with `original_query` derived from the fresh
decontextualized input, so quoted facts are weak context instead of the first
retrieval target.

Experiment findings carried into this plan:

- Quote-aware sequencing is a RAG orchestration problem, not a
  decontextualizer rewrite problem.
- A no-quote message must preserve the current single-pass RAG behavior.
- A quote-present message needs quote grounding before fresh-message research.
- Fresh research must receive compact quote facts only when the quote pass
  produced resolved substantive facts.
- If quote grounding misses, a self-contained fresh message must still resolve
  without a combined retry.
- If both quote grounding and fresh research miss, exactly one combined retry
  may run with raw quoted text plus fresh text.
- Each RAG pass must receive a pass-local
  `context["prompt_message_context"]["body_text"]` matching that pass's
  `original_query`. The experiment found that leaking fresh text into quote
  grounding can make live results look correct for the wrong reason.
- The live web-search failures observed during experiment runs are not treated
  as evidence against this plan because the user identified missing MCP startup
  as the likely production issue. This plan ignores that retrieval-environment
  failure and verifies orchestration deterministically.
- Prompt-side quote awareness in the RAG initializer is the structurally
  simpler long-term alternative: one supervisor call can plan directly from
  `reply_context.reply_excerpt` plus the fresh message. This plan intentionally
  defers that simpler prompt-side design because the approved near-term goal is
  no initializer prompt churn and a smaller production blast radius. If
  reply-turn latency becomes a problem, the next simplification path is
  initializer prompt awareness, not additional wrapper complexity.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing RAG orchestration or any
  model-facing query text.
- `development-plan-writing`: load before updating this plan, execution
  evidence, registry rows, or lifecycle status.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- Use `venv\Scripts\python` for Python commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing and before final reporting.
- Do not read `.env`.
- Do not edit the decontextualizer prompt or schema for this plan.
- Do not edit the RAG initializer prompt for this plan.
- Do not add a semantic rewrite LLM, quote rewrite LLM, feature flag, config
  switch, or compatibility path.
- Do not add deterministic keyword classification over user intent.
- Keep RAG as evidence provider; cognition still decides stance and dialog
  still owns final wording.
- Real LLM or live RAG checks, if run, must be run one case at a time and
  inspected one case at a time. MCP or web-search unavailability must be
  recorded as an environment blocker, not fixed in this plan.

## Must Do

- Create a production quote-aware RAG sequence module that wraps the existing
  `call_rag_supervisor`.
- Wire `stage_1_research` to call the quote-aware wrapper for the text-chat
  RAG path.
- Preserve the existing public RAG result shape consumed downstream:
  `answer`, `known_facts`, `unknown_slots`, and `loop_count`.
- Preserve no-quote behavior as a single current RAG pass.
- For quote-present behavior, run quote grounding before fresh-message
  research.
- Clone pass-local context for each RAG pass so
  `prompt_message_context.body_text` equals that pass's `original_query`.
- Merge multi-pass RAG outputs deterministically and prevent unresolved later
  passes from overriding earlier substantive facts.
- Add deterministic tests for the full input-pattern matrix listed in this
  plan.
- Keep the experiment harness under `experiments/quote_aware_rag_sequence/`
  absent; delete it again if it reappears during implementation.
- Update plan registry rows so this active plan is the executable source of
  truth.

## Deferred

- Do not fix MCP startup or web-search-agent availability in this plan.
- Do not tune `web_search_agent2` query generation or result fetching.
- Do not change Cache2 policy, cache keys, cache invalidation, or cache
  persistence.
- Do not redesign RAG initializer slot planning.
- Do not edit the RAG initializer prompt in this plan. Prompt-side
  quote-aware slot planning is deferred as the preferred simplification if the
  wrapper's extra calls later create latency or reliability pressure.
- Do not change cognition, dialog, consolidation, adapter delivery, or
  persistence behavior.
- Do not add support for quoted attachments beyond existing `reply_excerpt`
  text.
- Do not retain experiment files as a long-term test surface.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Text-chat RAG call path | bigbang | Replace direct `call_rag_supervisor(...)` use in `stage_1_research` with the quote-aware wrapper. |
| No-quote behavior | compatible | Preserve current one-pass behavior exactly except for the wrapper call boundary. |
| RAG result shape | compatible | Return the same public keys consumed by downstream code. Do not expose experiment-only trace fields in `rag_result`. |
| Quote-present behavior | bigbang | Use quote grounding first whenever `reply_context.reply_excerpt` has nonblank text. |
| Experiment harness | bigbang | Keep `experiments/quote_aware_rag_sequence/` absent. The plan contains the migrated findings and test matrix. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- Bigbang areas must be replaced or removed directly. Do not preserve alternate
  production call paths.
- Compatible areas preserve only the surfaces listed in `Cutover Policy`.
- Any change to this cutover policy requires user approval before
  implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the contracts in this plan.
- The agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, feature flags, or extra
  features.
- The agent must treat changes outside the target modules as high-scrutiny
  changes.
- The agent may remove experiment code because removal is explicitly in scope.
- Before adding any helper, the agent must search the codebase for equivalent
  behavior and reuse or move existing behavior when appropriate.
- The agent must not perform unrelated cleanup, formatting churn, dependency
  upgrades, prompt rewrites, or broad refactors.
- If this plan and code disagree, the agent must preserve this plan's stated
  intent and report the discrepancy.
- If a required instruction is impossible, the agent must stop and report the
  blocker instead of inventing a substitute.

## Target State

Target runtime flow for text-chat research:

```text
stage_1_research
  -> build_text_chat_rag_request(...)
  -> call_quote_aware_rag_supervisor(
       fresh_query=rag_request["original_query"],
       reply_context=state["reply_context"],
       character_name=state["character_profile"]["name"],
       context=rag_request["context"],
     )
  -> cognition
  -> dialog
```

Target quote-present RAG sequence:

```text
quote_grounding query
  -> existing call_rag_supervisor(...)
  -> quote known_facts

fresh query plus compact quote facts, only when quote facts are substantive
  -> existing call_rag_supervisor(...)
  -> fresh known_facts

combined retry, only when quote and fresh passes both have no substantive facts
  -> existing call_rag_supervisor(...)
```

The wrapper owns pass ordering, pass-local context shaping, bounded retry, and
result merging. Existing RAG initializer, dispatcher, helper agents, evaluator,
and finalizer keep their current responsibilities.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Ownership | Implement quote awareness as RAG orchestration. | The adapter and decontextualizer already preserve `reply_context`; downstream needs RAG evidence, not rewritten input. |
| Initializer prompt | Do not edit it in this plan; defer prompt-side quote awareness as the simpler future design. | Teaching the initializer to plan from `reply_excerpt` would be the more direct one-call fix, but it changes a shared local-LLM prompt. The wrapper is the approved lower-blast-radius path for this plan. |
| Extra LLM | Do not add one. | Existing RAG graph can run multiple passes; a rewrite LLM adds latency and another failure surface. |
| Retry limit | Allow exactly one combined retry. | Covers vague fresh text after quote miss without turning normal chat into an unbounded loop. |
| Pass context | Clone context and align `prompt_message_context.body_text` to each pass query. | Prevents hidden fresh-text leakage into quote grounding and stabilizes Cache2 keys. |
| Public result | Keep existing public shape. | Cognition and dialog should not need downstream changes. |
| Trace | Log pass-level trace instead of returning an experiment-only trace field. | Maintains public result compatibility while keeping production inspectable. |
| Live retrieval failures | Ignore MCP/web-search availability as a blocker for this plan. | The user identified MCP startup as the likely production issue; this plan validates orchestration. |

## Input Pattern Matrix

| Case | Pattern | Expected path | Required test coverage |
|---|---|---|---|
| `no_quote_missing` | `reply_context` absent or empty | `fresh_only` | Wrapper call arguments and return value match a direct `call_rag_supervisor` call. |
| `no_quote_empty_excerpt` | `reply_excerpt == ""` | `fresh_only` | Same direct-call equivalence as missing quote. |
| `no_quote_whitespace_excerpt` | `reply_excerpt` contains only whitespace | `fresh_only` | Same direct-call equivalence as missing quote. |
| `quote_hit` | Quote has useful factual anchors and fresh text asks about them | `quote_grounding -> fresh_after_quote` | Fresh pass receives compact quote facts. |
| `quote_hit_fresh_miss` | Quote resolves but fresh vague follow-up produces no substantive facts | `quote_grounding -> fresh_after_quote` | No combined retry; merged answer keeps quote-pass evidence. |
| `quote_miss_retry` | Quote has no evidence and fresh text is vague | `quote_grounding -> fresh_after_quote -> combined_retry` | Exactly one combined retry. |
| `quote_miss_fresh_hit` | Quote has no evidence but fresh text is self-contained | `quote_grounding -> fresh_after_quote` | Fresh pass resolves and no retry runs. |
| `quote_hit_additional_search` | Quote resolves and fresh text asks a new fact | `quote_grounding -> fresh_after_quote` | Merged facts contain quote and new fresh evidence. |
| `quote_irrelevant_fresh_search` | Quote resolves but fresh text asks unrelated fact | `quote_grounding -> fresh_after_quote` | Fresh research is not blocked by quote evidence. |
| `quote_third_party_verify` | User asks to verify someone else's quoted claim | `quote_grounding -> fresh_after_quote` | Quote claim is researched before verification response. |
| `quote_nonfactual_fresh_hit` | Quote is nonfactual small talk and fresh query has facts | `quote_grounding -> fresh_after_quote` | Fresh pass resolves and no retry runs. |

## Contracts And Data Shapes

Create `src/kazusa_ai_chatbot/rag/quote_aware_sequence.py`.

Public entrypoint:

```python
async def call_quote_aware_rag_supervisor(
    *,
    fresh_query: str,
    reply_context: dict[str, Any],
    character_name: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Run quote-aware RAG sequencing around the existing RAG supervisor."""
```

Returned public shape:

```python
{
    "answer": str,
    "known_facts": list[dict[str, Any]],
    "unknown_slots": list[str],
    "loop_count": int,
}
```

Do not return `quote_aware_trace` or other experiment-only keys from the
production entrypoint. Emit logs for traceability instead.

Pass names used in logs:

```text
fresh_only
quote_grounding
fresh_after_quote
combined_retry
```

Quote detection:

```text
reply_context.reply_excerpt exists, is a string, and has non-whitespace text
```

Quote-grounding query contract:

```text
Research factual content contained in the quoted/replied text.
Treat the quote as quoted material, not verified truth.
Do not search conversation history or identify the speaker unless the quote
itself asks for provenance.
Preserve exact quoted names, model names, numbers, units, and claim values.
Use quoted names and model names as primary search anchors.
Treat quoted numbers and units as claim values to verify, not mandatory search
keywords for every search.
```

Fresh-after-quote query contract:

```text
Known evidence from the quoted/replied message is already retrieved.
Do not create retrieval slots for the same quoted-message facts again.
Retrieve only additional missing facts needed by the current user message.
Use quoted-message evidence only when relevant to the current user message.
```

Combined retry query contract:

```text
Resolve the current user message using the quoted/replied text as context.
Treat quoted text as quoted material, not verified truth.
Include raw quoted text and current fresh message.
```

Merge contract:

- Merge `known_facts` from all executed passes.
- De-duplicate by `(slot, agent)`.
- If duplicate facts exist, preserve the resolved fact over the unresolved
  fact.
- Merge string `unknown_slots` in first-seen order.
- Sum integer `loop_count` values.
- Choose the latest non-empty answer that belongs to a pass with substantive
  resolved facts.
- If no pass has substantive resolved facts, choose the latest non-empty answer.
- A substantive fact is a resolved fact that is not a pure
  `person_context_agent` display-name resolution.
- The substantive-fact rule is intentionally narrow for this plan. It is a
  known fragile denylist: if future helper agents emit resolved non-content
  facts, the merge logic must be reviewed before those agents are allowed to
  influence answer selection.

## LLM Call And Context Budget

Before:

| Case | Response-path RAG supervisor calls | Notes |
|---|---:|---|
| No quote | 1 | Existing behavior. |
| Quote present | 1 | Quote text only appears as context. |

After:

| Case | Response-path RAG supervisor calls | Hard cap |
|---|---:|---:|
| No quote | 1 | 1 |
| Quote present and quote or fresh pass resolves substantive facts | 2 | 2 |
| Quote present, quote resolves, fresh pass misses | 2 | 2 |
| Quote present and both first passes miss | 3 | 3 |

Context budget:

- Use a module constant `FACT_SUMMARY_CHAR_LIMIT = 1600`.
- Compact quote facts before injecting them into the fresh pass.
- Do not include raw helper-agent blobs in the fresh pass query.
- Do not add a new LLM call outside existing `call_rag_supervisor` calls.
- This response-path latency increase is approved only for messages that carry
  nonblank `reply_context.reply_excerpt`.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/rag/quote_aware_sequence.py`
  - Owns quote detection, pass query construction, pass-local context cloning,
    existing RAG supervisor calls, merge logic, and production logs.
- `tests/test_quote_aware_rag_sequence.py`
  - Deterministic tests for all matrix cases, pass context isolation, merge
    precedence, and public result shape.
- `tests/test_quote_aware_rag_sequence_live.py`
  - Live eval cases for the new query-prefix behavior. These tests must be run
    one at a time and inspected one at a time.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Replace direct `call_rag_supervisor(...)` use in `stage_1_research` with
    `call_quote_aware_rag_supervisor(...)`.
  - Preserve `build_text_chat_rag_request(...)` use and downstream projection.
- `tests/test_persona_supervisor2_rag2_integration.py`
  - Update patched handoff tests so they capture the wrapper call instead of
    the direct RAG supervisor call.
  - Add or update a test proving `reply_context.reply_excerpt` reaches the
    wrapper.
- `development_plans/README.md`
  - Register this active bugfix plan and remove stale reference-plan registry
    rows.

### Delete

- `development_plans/reference/quote_aware_rag_sequence_experiment.md`
  - Its findings are incorporated into this active plan.
- `experiments/quote_aware_rag_sequence/`
  - Removed during plan promotion. Future implementation must verify it remains
    absent.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
  - No prompt changes in this plan.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_msg_decontexualizer.py`
  - No quote flattening or semantic rewrite changes in this plan.
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`
  - No cache policy changes in this plan.

## Overdesign Guardrail

- Actual problem: RAG loses important quoted/replied facts because it
  researches the fresh decontextualized message before the replied text.
- Minimal change: add a small RAG orchestration wrapper that runs the existing
  RAG supervisor over quote text first, then fresh text, with one bounded
  combined retry only when both first passes miss.
- Ownership boundaries: deterministic wrapper owns pass order, pass-local
  context, merge, and retry cap; RAG initializer owns semantic slot planning;
  dispatcher owns agent routing; helper agents own retrieval parameters;
  cognition owns stance; dialog owns final wording.
- Rejected complexity for this execution: no decontextualizer rewrite, no
  semantic rewrite LLM, no new quote parser, no Cache2 policy change, no
  feature flag, no compatibility dual path, no web-search-agent fix, and no
  additional retries. RAG initializer prompt awareness is deferred, not
  permanently rejected, because it is the simpler one-call design if wrapper
  latency becomes unacceptable.
- Evidence threshold: add rejected complexity only after deterministic matrix
  tests pass and a separate observed failure shows the wrapper cannot preserve
  quote facts without that specific new mechanism.

## Implementation Order

1. Add failing deterministic module tests.
   - File: `tests/test_quote_aware_rag_sequence.py`.
   - Cover every row in `Input Pattern Matrix`.
   - Add explicit direct-call equivalence tests for missing, empty, and
     whitespace-only `reply_excerpt`.
   - Add the `quote_hit_fresh_miss` test proving no combined retry runs and the
     quote-pass answer remains preferred.
   - Cover pass-local context isolation.
   - Cover answer selection where unresolved later passes must not override
     earlier substantive quote facts.
   - Cover public result keys exactly.
   - Verify: `venv\Scripts\python -m pytest tests\test_quote_aware_rag_sequence.py -q`.
   - Expected before implementation: import or symbol failure because
     `quote_aware_sequence.py` does not exist.
2. Add failing integration handoff test.
   - File: `tests/test_persona_supervisor2_rag2_integration.py`.
   - Patch `persona_supervisor2.call_quote_aware_rag_supervisor`.
   - Prove `stage_1_research` passes `fresh_query`, `reply_context`,
     `character_name`, and `context`.
   - Verify the focused test by name with `-q`.
   - Expected before implementation: missing imported symbol or patched call
     not invoked.
3. Add live eval tests for quote query-prefix behavior.
   - File: `tests/test_quote_aware_rag_sequence_live.py`.
   - Add one quote-hit live eval.
   - Add one quote-miss live eval.
   - Add one combined-retry live eval.
   - Mark or document them consistently with existing live RAG tests.
   - Each live eval must log the pass path, generated slots, known facts, and
     final answer for inspection.
   - Do not batch-run these tests.
4. Implement `src/kazusa_ai_chatbot/rag/quote_aware_sequence.py`.
   - Add the public entrypoint and private helpers only where they implement
     nontrivial behavior listed in `Contracts And Data Shapes`.
   - Keep imports at module top.
   - Keep helper docstrings meaningful.
   - Keep exception handling out of internal logic unless calling an external
     boundary that can fail.
5. Run focused module tests.
   - Command: `venv\Scripts\python -m pytest tests\test_quote_aware_rag_sequence.py -q`.
   - Do not wire `stage_1_research` until these pass.
6. Wire `stage_1_research`.
   - Update imports in `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`.
   - Replace the direct RAG supervisor call with the wrapper.
   - Keep the existing `rag_request` construction and projection unchanged.
7. Run focused integration tests.
   - Command: `venv\Scripts\python -m pytest tests\test_persona_supervisor2_rag2_integration.py -q`.
8. Run live eval tests one case at a time.
   - Run each command in `Verification > Live Eval Set` separately with `-q -s`.
   - Inspect output after each case.
   - Record whether generated slots are usable. MCP or web retrieval
     unavailability is an environment blocker, not a scope expansion.
9. Verify stale experiment code and reference note are absent.
   - Confirm `experiments/quote_aware_rag_sequence/` is absent.
   - Confirm `development_plans/reference/quote_aware_rag_sequence_experiment.md`
     is absent.
   - Confirm the active plan contains all necessary findings.
10. Run full verification.
   - Run every command in `Verification`.
   - Record outputs in `Execution Evidence`.
11. Run independent code review.
   - Follow `Independent Code Review`.
   - Fix in-scope findings and rerun affected verification.

## Progress Checklist

- [x] Stage 1 - deterministic module contract established.
  - Covers: implementation steps 1, 4, and 5.
  - Verify: `venv\Scripts\python -m pytest tests\test_quote_aware_rag_sequence.py -q`.
  - Evidence: initial failure and final pass recorded in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-15`.
- [x] Stage 2 - `stage_1_research` integration complete.
  - Covers: implementation steps 2, 6, and 7.
  - Verify: `venv\Scripts\python -m pytest tests\test_persona_supervisor2_rag2_integration.py -q`.
  - Evidence: focused test output and wrapper handoff fields recorded in
    `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-15`.
- [x] Stage 3 - live eval coverage complete.
  - Covers: implementation steps 3 and 8.
  - Verify: each command in `Verification > Live Eval Set` is run separately
    with `-q -s`.
  - Evidence: record pass path, generated slots, known facts, final answer,
    and any MCP/web retrieval environment blocker for each case.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-15`.
- [x] Stage 4 - stale experiment surfaces absent.
  - Covers: implementation step 9.
  - Verify: static greps in `Verification`.
  - Evidence: record deleted paths and grep output.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-05-15`.
- [x] Stage 5 - regression verification complete.
  - Covers: implementation step 10.
  - Verify: all commands in `Verification`.
  - Evidence: record command outputs and any environment blockers.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `Codex/2026-05-15`.
- [x] Stage 6 - independent code review complete.
  - Covers: implementation step 11.
  - Verify: review findings resolved or explicitly recorded as residual risk.
  - Evidence: record reviewer mode, files reviewed, findings, fixes, rerun
    commands, and approval status.
  - Handoff: plan can be marked completed only after this stage is signed off.
  - Sign-off: `Codex/2026-05-15`.

## Verification

### Static Syntax

```powershell
venv\Scripts\python -m py_compile `
  src\kazusa_ai_chatbot\rag\quote_aware_sequence.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2.py `
  tests\test_quote_aware_rag_sequence.py
```

Expected result: exit code `0`.

### Focused Tests

```powershell
venv\Scripts\python -m pytest tests\test_quote_aware_rag_sequence.py -q
```

Expected result: all tests pass.

```powershell
venv\Scripts\python -m pytest tests\test_persona_supervisor2_rag2_integration.py -q
```

Expected result: all tests pass.

### Regression Tests

```powershell
venv\Scripts\python -m pytest `
  tests\test_persona_supervisor2_rag2_integration.py `
  tests\test_rag_projection.py `
  tests\test_referent_resolution.py `
  -q
```

Expected result: all tests pass.

### Static Greps

```powershell
rg "experiments[/\\]quote_aware|development_plans[/\\]reference[/\\]quote_aware_rag_sequence_experiment" `
  src tests development_plans `
  -g "!development_plans/active/bugfix/quote_aware_rag_sequence_plan.md" `
  -g "!development_plans/archive/completed/bugfix/quote_aware_rag_sequence_plan.md"
```

Expected result: no stale experiment or reference-plan matches. `rg` exit code
`1` is acceptable for no matches. Intended production and test paths containing
`quote_aware_rag_sequence` are allowed.

```powershell
rg "quote_aware_trace" src tests
```

Expected result: no matches. `rg` exit code `1` is acceptable for no matches.

```powershell
rg "call_rag_supervisor" src\kazusa_ai_chatbot\nodes\persona_supervisor2.py
```

Expected result: no direct `call_rag_supervisor` call remains in
`stage_1_research`. An import of the quote-aware wrapper is allowed.

### Live Eval Set

Run these only after deterministic verification passes. Run one command,
inspect its output, record the judgment, then run the next command.

```powershell
venv\Scripts\python -m pytest `
  tests\test_quote_aware_rag_sequence_live.py::test_quote_aware_rag_live_quote_hit `
  -m "live_llm and live_db" `
  -q -s
```

Expected result: quote grounding runs first; generated slots preserve quoted
entity anchors and create usable retrieval intent; final result is non-empty or
records an MCP/web retrieval environment blocker.

```powershell
venv\Scripts\python -m pytest `
  tests\test_quote_aware_rag_sequence_live.py::test_quote_aware_rag_live_quote_miss `
  -m "live_llm and live_db" `
  -q -s
```

Expected result: quote grounding can miss without crashing; fresh pass still
runs; no combined retry occurs when the fresh message is self-contained.

```powershell
venv\Scripts\python -m pytest `
  tests\test_quote_aware_rag_sequence_live.py::test_quote_aware_rag_live_combined_retry `
  -m "live_llm and live_db" `
  -q -s
```

Expected result: quote pass and fresh pass both miss or return no substantive
facts; exactly one combined retry runs. If MCP or web retrieval is unavailable,
record the environment blocker in `Execution Evidence` and do not broaden this
plan to fix MCP.

## Independent Plan Review

Review mode: same agent, fresh-review posture; no separate reviewer was used.

Review inputs:

- This active plan draft.
- `development_plans/README.md`.
- Development-plan references: `plan_contract.md`, `execution_gates.md`, and
  `cutover_policy.md`.
- Relevant source ownership from `persona_supervisor2.py`,
  `cognitive_episode_adapter.py`, and RAG supervisor call sites.
- Experiment findings from the now-promoted quote-aware sequence work.

Findings:

- Blockers: none.
- Non-blocking finding: live web-search failures remain outside this plan
  because the user identified MCP startup as the likely issue. The plan records
  this as an environment blocker rather than a production-code requirement.
- Non-blocking finding: public trace output from the experiment is removed
  from the production contract to avoid downstream shape drift.
- Review-derived updates after user feedback: initializer prompt awareness is
  now framed as deferred simplification, no-quote direct-call equivalence is a
  required test, the `quote_hit_fresh_miss` matrix cell is explicit, live eval
  coverage contains three one-at-a-time cases, the substantive-fact denylist is
  documented as fragile, and stale experiment grep syntax is path-separator
  explicit.
- Required edits made before approval: added pass-local context isolation,
  explicit no-initializer-prompt rule, explicit experiment cleanup, matrix test
  coverage, and public-result-shape compatibility.

Approval status: approved for execution as a large active bugfix plan.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including prior-stage artifacts, focused and
  regression tests, execution evidence, next-stage handoff notes, and path-safe
  commands.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only fixture or
documentation corrections. If a fix would cross the approved boundary or alter
the contract, stop and update the plan or request approval before changing
code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- No-quote input executes exactly one RAG supervisor pass.
- Missing, empty, and whitespace-only quote inputs are direct-call equivalent
  to `call_rag_supervisor` for arguments and return value.
- Quote-present input executes quote grounding before fresh-message research.
- Quote facts are compacted into the fresh pass only when the quote pass has
  substantive resolved facts.
- A quote hit plus fresh-message miss executes no combined retry and preserves
  the quote-pass answer.
- A quote miss plus self-contained fresh query resolves through the fresh pass
  without combined retry.
- A quote miss plus vague fresh query performs exactly one combined retry.
- Unresolved later passes do not override earlier substantive quote evidence.
- `prompt_message_context.body_text` is pass-local and matches each pass's
  `original_query`.
- Downstream RAG public result shape remains `answer`, `known_facts`,
  `unknown_slots`, and `loop_count`.
- `stage_1_research` calls the quote-aware wrapper and still projects RAG
  results into cognition the same way.
- `experiments/quote_aware_rag_sequence/` is deleted.
- The stale reference experiment plan is deleted.
- All verification commands pass or record an accepted MCP/web-retrieval
  environment blocker only for the live eval gates.
- The three live eval cases are run one at a time and inspected.
- Independent code review is complete and recorded.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Response-path latency increases for reply turns | Hard cap of three RAG supervisor calls and no extra LLM outside existing RAG graph | Module tests assert pass counts |
| Fresh text leaks into quote grounding context | Clone pass-local context and set `prompt_message_context.body_text` to the pass query | Context isolation test |
| Unresolved fresh pass overrides resolved quote evidence | Answer selection prefers latest answer with substantive resolved facts | Merge precedence test |
| Quote pass treats quoted claims as verified truth | Query wording says quoted text is material to verify, not truth | Query construction test |
| Local LLM generates unusable slots from nuanced quote-prefix wording | Add three live eval cases for quote hit, quote miss, and combined retry; run one at a time | Live eval set |
| Substantive-fact rule is a fragile denylist | Record the denylist as known fragility and require review when new resolved non-content helper agents are added | Independent code review |
| Experiment code becomes stale | Delete experiment directory after production tests exist | Static grep and deleted-path evidence |
| MCP/web retrieval failure hides orchestration behavior | Deterministic tests validate orchestration; live evals record MCP blocker separately | Verification and execution evidence |

## Execution Evidence

- Pre-execution deterministic experiment evidence:
  - Expanded fake matrix covered eight input patterns.
  - Experiment contract tests passed: `10 passed in 0.02s`.
  - Existing RAG/referent regression passed: `23 passed in 2.00s`.
- Planning cleanup evidence:
  - `development_plans/reference/quote_aware_rag_sequence_experiment.md` was
    deleted after its findings were promoted into this plan.
  - Ignored experiment harness `experiments/quote_aware_rag_sequence/` was
    removed after the plan captured the case matrix and conclusions.
- Live experiment evidence:
  - Earlier successful live run showed quote-first BYD Shark grounding when
    web retrieval worked.
  - Later isolated live runs failed at `web_search_agent2` with no results.
    Per user instruction, treat this as MCP/web-retrieval environment evidence
    and not as a blocker for the orchestration plan.
- Implementation execution evidence:
  - Status changed to `in_progress` in this plan and the development-plan
    registry.
  - Stage 1 initial red: `venv\Scripts\python -m pytest
    tests\test_quote_aware_rag_sequence.py -q` failed at collection with
    `ImportError: cannot import name 'quote_aware_sequence'`.
  - Stage 1 implemented files:
    `src/kazusa_ai_chatbot/rag/quote_aware_sequence.py` and
    `tests/test_quote_aware_rag_sequence.py`.
  - Stage 1 final green: `venv\Scripts\python -m pytest
    tests\test_quote_aware_rag_sequence.py -q` returned `12 passed in 1.60s`.
  - Stage 2 initial red: `venv\Scripts\python -m pytest
    tests\test_persona_supervisor2_rag2_integration.py -q` returned
    `5 failed, 1 passed`; failures were expected `AttributeError` for missing
    `persona_supervisor2.call_quote_aware_rag_supervisor`.
  - Stage 2 implemented files:
    `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` and
    `tests/test_persona_supervisor2_rag2_integration.py`.
  - Stage 2 final green: `venv\Scripts\python -m pytest
    tests\test_persona_supervisor2_rag2_integration.py -q` returned
    `6 passed in 1.91s`.
  - CJK/syntax check after writing CJK live-test strings:
    `venv\Scripts\python -m py_compile
    src\kazusa_ai_chatbot\rag\quote_aware_sequence.py
    src\kazusa_ai_chatbot\nodes\persona_supervisor2.py
    tests\test_quote_aware_rag_sequence.py
    tests\test_quote_aware_rag_sequence_live.py
    tests\test_persona_supervisor2_rag2_integration.py` exited `0`.
  - Live eval command accuracy: the first live command without a marker
    override was deselected by `pytest.ini` default addopts. Live commands were
    rerun one at a time with `-m "live_llm and live_db"`.
  - Stage 3 live quote-hit: `venv\Scripts\python -m pytest
    tests\test_quote_aware_rag_sequence_live.py::test_quote_aware_rag_live_quote_hit
    -m "live_llm and live_db" -q -s` returned `1 passed` in `115.81s`.
    Trace:
    `test_artifacts/llm_traces/quote_aware_rag_sequence_live__quote_hit.json`.
    Pass path was `quote_grounding -> fresh_after_quote -> combined_retry`.
    The quote-grounding initializer preserved the quoted BYD Shark 6 anchor:
    `Web-evidence: retrieve public information about 比亚迪 Shark 6
    specifications and powertrain configuration`. `web_search_agent2` timed
    out on all three MCP search attempts, so no substantive facts were
    resolved and the combined retry was expected under the wrapper rule.
  - Stage 3 live quote-miss: `venv\Scripts\python -m pytest
    tests\test_quote_aware_rag_sequence_live.py::test_quote_aware_rag_live_quote_miss
    -m "live_llm and live_db" -q -s` returned `1 passed` in `115.69s`.
    Trace:
    `test_artifacts/llm_traces/quote_aware_rag_sequence_live__quote_miss.json`.
    Pass path was `quote_grounding -> fresh_after_quote -> combined_retry`.
    Quote grounding searched the nonsensical quote and missed due MCP search
    timeout. The fresh self-contained query ran, but the local initializer
    chose `Memory-evidence` for `1.5T 皮卡` and memory retrieval missed; the
    combined retry was therefore consistent with the "both first passes miss"
    rule.
  - Stage 3 live combined-retry: `venv\Scripts\python -m pytest
    tests\test_quote_aware_rag_sequence_live.py::test_quote_aware_rag_live_combined_retry
    -m "live_llm and live_db" -q -s` returned `1 passed` in `115.16s`.
    Trace:
    `test_artifacts/llm_traces/quote_aware_rag_sequence_live__combined_retry.json`.
    Pass path was exactly `quote_grounding -> fresh_after_quote ->
    combined_retry`. The quote-grounding slot preserved `玄铁推进器 9999Z` and
    `12345 N·m` as claim material to verify, but MCP web search timed out.
  - Stage 4 deterministic rerun after trace-key assertion cleanup:
    `venv\Scripts\python -m pytest tests\test_quote_aware_rag_sequence.py -q`
    returned `12 passed in 1.65s`.
  - Stage 4 stale path checks: `Test-Path
    experiments\quote_aware_rag_sequence` returned `False`, and `Test-Path
    development_plans\reference\quote_aware_rag_sequence_experiment.md`
    returned `False`.
  - Stage 4 stale grep:
    `rg 'experiments[/\\]quote_aware|development_plans[/\\]reference[/\\]quote_aware_rag_sequence_experiment'
    src tests development_plans -g
    '!development_plans/active/bugfix/quote_aware_rag_sequence_plan.md'`
    returned exit code `1` with no output, which is the expected no-match
    result.
  - Stage 4 public trace grep: `rg 'quote_aware_trace' src tests` returned
    exit code `1` with no output, proving no production/test surface exposes
    the experiment trace key.
  - Stage 4 direct-call grep:
    `rg 'call_rag_supervisor' src\kazusa_ai_chatbot\nodes\persona_supervisor2.py`
    returned exit code `1` with no output.
  - Stage 5 static syntax:
    `venv\Scripts\python -m py_compile
    src\kazusa_ai_chatbot\rag\quote_aware_sequence.py
    src\kazusa_ai_chatbot\nodes\persona_supervisor2.py
    tests\test_quote_aware_rag_sequence.py` exited `0`.
  - Stage 5 focused quote-aware module tests:
    `venv\Scripts\python -m pytest tests\test_quote_aware_rag_sequence.py -q`
    returned `12 passed in 1.81s`.
  - Stage 5 focused integration tests:
    `venv\Scripts\python -m pytest
    tests\test_persona_supervisor2_rag2_integration.py -q` returned
    `6 passed in 2.23s`.
  - Stage 5 regression tests:
    `venv\Scripts\python -m pytest
    tests\test_persona_supervisor2_rag2_integration.py
    tests\test_rag_projection.py tests\test_referent_resolution.py -q`
    returned `23 passed in 2.26s`.
  - Stage 5 static greps repeated successfully: stale experiment/reference
    grep, public trace-key grep, and direct `call_rag_supervisor` grep each
    returned exit code `1` with no output, the expected no-match result.
  - Stage 5 live eval commands were not rerun because the plan required live
    cases to be run and inspected one at a time; the full live eval set was
    already completed and recorded under Stage 3.
  - Stage 6 independent code review mode: same agent, fresh-review posture; no
    separate reviewer was available in this session and no subagent was
    requested by the user.
  - Stage 6 review inputs: this plan after Stage 5 sign-off,
    `development_plans/README.md`, `src\kazusa_ai_chatbot\rag\quote_aware_sequence.py`,
    `src\kazusa_ai_chatbot\nodes\persona_supervisor2.py`,
    `tests\test_quote_aware_rag_sequence.py`,
    `tests\test_quote_aware_rag_sequence_live.py`, and
    `tests\test_persona_supervisor2_rag2_integration.py`.
  - Stage 6 review commands:
    `git diff --check`, `git diff --cached --check`, and
    `venv\Scripts\python -m py_compile
    tests\test_quote_aware_rag_sequence_live.py` all exited `0`.
  - Stage 6 findings: no blocking or in-scope corrective code findings. The
    wrapper keeps quote handling inside RAG orchestration, the no-quote path is
    direct-call equivalent, quote-present paths use pass-local context, result
    merging returns the public RAG shape only, and `stage_1_research` preserves
    downstream projection ownership.
  - Stage 6 non-code observation: `development_plans\README.md` and an
    untracked `development_plans\reference\designs\cognition_core_evolution_progression.md`
    contain unrelated reference-document registry work already present in the
    worktree. This execution left that unrelated work intact.
  - Stage 6 residual risks: live web/MCP retrieval timed out during the live
    evals and remains outside this plan by user instruction; the local
    initializer may still choose memory evidence for some self-contained fresh
    vehicle terminology queries; and the substantive-fact rule remains the
    documented narrow denylist that must be reviewed when new resolved
    non-content helper agents are added.
  - Stage 6 approval status: approved for completion and archival.
  - Lifecycle closeout: completed plan moved from
    `development_plans\active\bugfix\quote_aware_rag_sequence_plan.md` to
    `development_plans\archive\completed\bugfix\quote_aware_rag_sequence_plan.md`.
    `development_plans\README.md` now lists no active bugfix plan for this
    item and includes the completed bugfix archive link.
