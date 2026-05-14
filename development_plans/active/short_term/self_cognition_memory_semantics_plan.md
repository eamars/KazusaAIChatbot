# self cognition memory semantics plan

## Summary

- Goal: define and implement the bounded memory-generation contract for
  self-cognition after the context-budget bugfix removes `no_remember`.
- Plan class: large
- Status: draft
- Mandatory skills: `py-style`, `test-style-and-execution`,
  `local-llm-architecture`, `no-prepost-user-input`
- Overall cutover strategy: compatible
- Highest-risk areas: deciding what self-cognition may save, avoiding full
  consolidator side effects, prompt budget, duplicate memory, cache
  invalidation, and auditability
- Acceptance criteria: not executable yet; this draft is ready for approval
  only after the memory-save taxonomy and persistence categories are accepted.

## Context

Self-cognition is intended to generate memory, but the correct save contract is
larger than the immediate context-overflow bugfix. The current self-cognition
runner performs background reasoning and action handoff, but it does not run
live-chat post-turn consolidation. The live-chat consolidator graph is not an
appropriate drop-in owner because it is built around user-message origin and
runs unrelated lanes such as global state, relationship insight, task dispatch,
affinity, and character image updates.

The prerequisite bugfix plan removes `no_remember` from self-cognition-created
state so memory is no longer blocked by default. This plan owns the next step:
defining what self-cognition should save, which evidence it may use, and how
to persist those memories without turning idle cognition into a synthetic chat
turn.

## Discovery Decisions Needed Before Approval

This plan is intentionally draft. Resolve these decisions before marking it
`approved`:

- Which self-cognition routes may generate memory:
  `action_candidate`, `progress_maintenance`, both, or another explicit set.
- Whether self-cognition can write only user-memory units, or also character
  internal state summaries.
- Whether memories may be created from silent cases, or only from cases with an
  action candidate, scheduled contact, or explicit progress update.
- Whether self-cognition may create future commitments, or whether commitments
  stay owned by scheduler/dispatcher paths only.
- Whether duplicate suppression should compare against user-memory units only
  or also recent conversation progress and self-cognition run artifacts.
- What audit artifact must be stored for skipped, rejected, or written memory
  candidates.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing prompt, graph, cognition,
  consolidation, memory, or background LLM behavior.
- `no-prepost-user-input`: load before changing memory extraction, promise
  persistence, or any path that decides whether user-facing instructions,
  preferences, commitments, or accepted actions become durable state.

## Mandatory Rules

- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Do not reintroduce `no_remember` into self-cognition-created state.
- Do not use `/chat`, adapter delivery, or synthetic conversation-history rows
  as the memory-write path.
- Do not call the full `call_consolidation_subgraph` for self-cognition.
- Do not persist memory from raw reflection rows, raw source packets, raw DB
  documents, embeddings, or full self-cognition artifacts.
- Do not add deterministic keyword rules that reinterpret user commitments,
  accepted preferences, or memory channels after an LLM output. If memory
  extraction is wrong, fix the prompt/schema and structural validation.
- Do not add retry loops, model-context increases, alternate LLM routes, or
  fallback prompts as the primary strategy for extraction failures.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Define the approved self-cognition memory taxonomy before implementation.
- Add a dedicated self-cognition memory lane under `self_cognition`.
- Use bounded, prompt-safe artifacts as the only input to memory extraction.
- Reuse existing user-memory-unit persistence contracts where safe.
- Add self-cognition origin and origin-policy support only for the approved
  persistence categories.
- Persist auditable memory-lane metadata for written, skipped, rejected, and
  failed cases.
- Add deterministic tests for route gating, prompt budgeting, origin policy,
  duplicate suppression, dry-run behavior, and live worker wiring.

## Deferred

- Do not add visual-directive behavior.
- Do not change the self-cognition scheduler/dispatcher handoff contract except
  where explicitly needed to avoid duplicate memory writes.
- Do not redesign live-chat consolidation.
- Do not add new MongoDB collections unless this plan is updated and approved
  with a data migration section.
- Do not migrate historical self-cognition artifacts.
- Do not change reflection promotion semantics.
- Do not add response-path LLM calls.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Self-cognition memory lane | compatible | Add a new background lane without changing live-chat consolidation. |
| Consolidator origin | compatible | Add a self-cognition origin only for approved categories; keep user-message origin unchanged. |
| Persistence categories | compatible | Allow only categories approved by this plan after discovery. |
| Dry runs | compatible | Preserve dry-run/no-write behavior and record memory-lane status without DB writes. |
| Database | compatible | Prefer existing user-memory-unit persistence. Add migration only after explicit approval. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to this policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the approved contracts in this plan.
- The agent must not invent the memory taxonomy, persistence categories, or
  duplicate policy during implementation.
- The agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, extra feature flags, or
  unrelated prompt rewrites.
- The agent must treat changes outside the files listed in `Change Surface` as
  out of scope unless the plan is updated first.
- If existing helpers exactly satisfy a needed projection, evaluator, DB writer,
  cache invalidation, or prompt-budget contract, reuse them.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

Self-cognition has a dedicated background memory lane. The lane receives only
bounded self-cognition artifacts and produces a small structured result that
states whether memory extraction was not applicable, skipped, rejected,
written, dry-run, or failed. Live worker runs can write approved memory
categories; dry-run paths record status without writes.

The lane does not call adapters, does not write conversation rows, does not
dispatch tasks a second time, does not update character image state, and does
not run the full live-chat consolidator graph.

## Proposed Memory Taxonomy For Approval

This section is a draft proposal, not an executable contract until accepted.

| Candidate category | Proposed default | Reason |
|---|---|---|
| User-memory units | Allow | Self-cognition can notice durable user-relevant facts or continuity needs from approved context. |
| Cache invalidation | Allow when user-memory units are written | Existing memory retrieval caches must not serve stale user memory. |
| Future commitments | Defer | Commitments can trigger scheduler/dispatcher effects and need a clearer ownership boundary. |
| Relationship insight | Defer | Higher risk of subjective drift from idle reasoning without a live user turn. |
| Character state | Defer | Could duplicate global growth or reflection outputs. |
| Task dispatch | Forbid | Self-cognition already has action/scheduler handoff; a memory lane must not dispatch again. |
| Character image | Forbid | Visual directives are disabled for self-cognition by default. |
| Conversation history | Forbid | Self-cognition is not a synthetic user chat turn. |

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Memory owner | Dedicated `self_cognition` memory lane. | The live-chat consolidator graph has unrelated side effects and origin assumptions. |
| Full consolidator graph | Do not call. | It runs lanes outside the self-cognition memory contract. |
| Prompt input | Use bounded prompt-safe projection. | Background LLMs must stay under context budget and avoid raw DB/source leakage. |
| Semantic decisions | LLM owns memory-channel judgment; code owns validation and persistence limits. | Preserves the project LLM/deterministic boundary. |
| Write path | Reuse existing persistence contracts where safe. | Avoids a parallel memory database path. |

## Contracts And Data Shapes

The final approved plan must replace this draft interface with exact accepted
fields and enums:

```python
async def run_self_cognition_memory_lane(
    *,
    case: dict,
    cognition_state: dict,
    cognition_output: dict,
    action_candidate: dict | None,
    dialog_output: dict | None,
    enable_memory_writes: bool,
) -> dict:
    """Return memory-lane result metadata and optionally write memory."""
```

Proposed output metadata:

```python
{
    "memory_lane_called": bool,
    "memory_writes_enabled": bool,
    "candidate_count": int,
    "written_count": int,
    "skipped_count": int,
    "rejected_count": int,
    "status": "not_applicable" | "dry_run" | "written" | "skipped" | "failed",
    "error": str,
}
```

## LLM Call And Context Budget

Use `50k tokens` as the overall context-window assumption. The memory lane must
define a conservative character budget before approval. The proposed default is
one background extraction/evaluation call only for approved routes, with a hard
skip for silent or duplicate-suppressed cases.

| Call | Path | Before | Proposed after |
|---|---|---|---|
| Self-cognition memory extraction/evaluation | background self-cognition | 0 calls | At most 1 bounded background call for approved routes |
| Full consolidator graph | background self-cognition | 0 calls | 0 calls; still excluded |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/self_cognition/runner.py`
  - Thread memory-lane invocation and artifacts after the taxonomy is approved.

- `src/kazusa_ai_chatbot/self_cognition/worker.py`
  - Enable live worker memory writes through an explicit parameter after the
    lane exists.
  - Preserve dry-run no-write behavior.

- `src/kazusa_ai_chatbot/self_cognition/models.py`
  - Add artifact names and status constants for memory-lane output.

- `src/kazusa_ai_chatbot/self_cognition/README.md`
  - Document approved memory categories, skipped categories, and audit behavior.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin.py`
  - Add a self-cognition origin builder only if the approved design reuses
    consolidator persistence.

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_origin_policy.py`
  - Allow only approved self-cognition persistence categories.

- `tests/test_self_cognition_*.py`
  - Add route gating, artifact, dry-run, and worker wiring tests.

- `tests/test_consolidator_origin_policy_db_writer.py`
  - Add origin-policy tests for approved categories.

### Create

- `src/kazusa_ai_chatbot/self_cognition/memory.py`
  - Dedicated home for self-cognition memory-lane orchestration after approval.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator.py`
  - Do not call the full graph for self-cognition.

- Live `/chat` service path behavior for user-message consolidation.
- Scheduler and dispatcher validation behavior.
- Adapter delivery behavior.
- Reflection promotion behavior.
- Visual-directive default from the bugfix plan.

## Implementation Order

This plan is not executable until the discovery decisions are resolved and this
section is tightened. After approval, implementation order must follow:

1. Add focused tests for approved route gating and no-write dry-run behavior.
2. Add focused tests for approved origin policy and forbidden categories.
3. Add prompt-safe projection and prompt-budget tests.
4. Implement `self_cognition/memory.py`.
5. Wire runner and worker integration.
6. Update docs.
7. Run focused and regression verification.
8. Run independent code review.

## Progress Checklist

- [ ] Stage 0 - discovery decisions resolved
  - Covers: memory taxonomy, allowed routes, persistence categories, duplicate
    policy, audit artifact shape, and prompt budget.
  - Verify: this draft no longer contains unresolved discovery decisions.
  - Evidence: record accepted decisions and updated plan diff.
  - Handoff: next agent starts at Stage 1.
  - Sign-off: `<agent/date>`.

- [ ] Stage 1 - executable plan finalized
  - Covers: exact contracts, implementation steps, verification commands, and
    acceptance criteria.
  - Verify: independent plan review passes.
  - Evidence: record review findings and approval status.
  - Handoff: implementation may start only after this stage.
  - Sign-off: `<agent/date>`.

- [ ] Stage 2 - implementation complete
  - Covers: approved implementation steps after plan finalization.
  - Verify: focused and regression tests pass.
  - Evidence: record command output.
  - Handoff: next agent starts independent code review.
  - Sign-off: `<agent/date>`.

- [ ] Stage 3 - independent code review complete
  - Verify: full diff reviewed against the approved plan and affected tests
    rerun after any review fixes.
  - Evidence: record reviewer mode, findings, fixes, rerun commands, residual
    risks, and approval status.
  - Handoff: plan can be marked completed only after this stage is signed off.
  - Sign-off: `<agent/date>`.

## Verification

Final verification commands must be filled in after Stage 0 decisions. The
minimum expected gates are:

```powershell
venv\Scripts\python -m pytest tests\test_self_cognition_framing.py tests\test_self_cognition_integration.py -q
venv\Scripts\python -m pytest tests\test_self_cognition_*.py -q
venv\Scripts\python -m pytest tests\test_consolidator_origin_policy_db_writer.py -q
```

Expected after implementation: deterministic tests pass with approved memory
categories and forbidden categories covered.

## Independent Plan Review

Run this gate before approval or execution. Prefer a reviewer that did not
draft the plan. If no separate reviewer is available, the active agent must
reread the parent self-cognition architecture reference, this plan, and
relevant source/test context from a fresh-review posture.

Review scope:

- The memory taxonomy is explicit and has no unresolved categories.
- The plan does not call the full consolidator graph.
- Allowed write categories are narrow and auditable.
- Prompt budget and input projection are bounded.
- Dry-run and live-worker behavior are distinct.
- Tests prove allowed writes, forbidden writes, skipped cases, and cache
  invalidation behavior.

## Independent Code Review

Run this gate after all final `Verification` commands pass and before final
sign-off. Prefer a reviewer that did not implement the change. If no separate
reviewer is available, the active agent must reread this plan, inspect the full
diff from a fresh-review posture, and record that no separate reviewer was
available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/context leaks, persistence risk,
  duplicate dispatcher side effects, brittle fixtures, and avoidable blast
  radius.
- Alignment with approved memory taxonomy, `Must Do`, `Deferred`, `Change
  Surface`, verification gates, and acceptance criteria.
- Regression and handoff quality, including static-grep accuracy, execution
  evidence, and lifecycle registry updates.

## Acceptance Criteria

This draft is ready for approval when:

- The discovery decisions are resolved and encoded as directives.
- The proposed memory taxonomy is accepted or replaced.
- The allowed persistence categories are explicit.
- The prompt budget is explicit.
- The implementation order contains exact tests, expected failures, exact
  source edits, and final verification commands.

The implemented feature is complete only after the finalized acceptance
criteria replace this draft section.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Self-cognition saves speculative or low-value memories | Resolve taxonomy before implementation and use evaluator tests | Route/category tests |
| Memory lane duplicates scheduler or dispatcher effects | Forbid task dispatch unless explicitly approved later | Origin-policy and worker tests |
| Memory lane becomes a hidden full consolidator clone | Keep full consolidator graph in `Keep` and test forbidden categories | Static grep and origin-policy tests |
| Prompt overflow recurs in memory extraction | Use bounded projection and explicit character budget | Prompt-budget tests |
| Deterministic code overrides semantic memory judgment | Apply `no-prepost-user-input` and keep code to structural validation | Review gate and passthrough tests |

## Execution Evidence

- Not started.

## Execution Handoff

Execution is not started. The next agent should begin at Stage 0 by resolving
the memory taxonomy and persistence decisions with the owner before this plan is
approved.
