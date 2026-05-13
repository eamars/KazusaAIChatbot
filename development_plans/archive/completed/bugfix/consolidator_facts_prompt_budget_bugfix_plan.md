# consolidator facts prompt budget bugfix plan

## Summary

- Goal: prevent raw RAG-surfaced `user_memory_unit_candidates` from entering
  `facts_harvester` and `fact_harvester_evaluator` prompts.
- Plan class: small
- Status: completed
- Mandatory skills: `py-style`, `test-style-and-execution`,
  `local-llm-architecture`
- Overall cutover strategy: bigbang
- Highest-risk areas: prompt-facing RAG payload shape, consolidation dedup
  behavior, memory-unit merge reuse
- Acceptance criteria: facts harvester prompts stay bounded even when RAG carries
  many raw memory-unit documents.

## Context

Background consolidation failed after the visible response was returned because
the consolidation LLM rejected the `facts_harvester` prompt:

```text
n_keep: 53268 >= n_ctx: 50176
```

The immediate failing boundary is
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`.
`facts_harvester` and `fact_harvester_evaluator` serialize the whole
`state["rag_result"]` into their human payloads. That `rag_result` may contain
`user_memory_unit_candidates`, a raw RAG-surfaced source list used later by the
memory-unit merge/evolve path. Those raw documents are not appropriate input
for the facts harvester.

This plan fixes only the primary prompt-budget leak. It does not fix
conversation speaker scoping, initializer slot wording, RAG retrieval quality,
or model context configuration.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before changing prompt, graph, RAG,
  consolidation, or background LLM payload behavior.

## Mandatory Rules

- Use `venv\Scripts\python` for Python test commands.
- Use `apply_patch` for manual file edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Do not remove `rag_result["user_memory_unit_candidates"]` from the global RAG
  projection; memory-unit merge still needs the raw surfaced units.
- Do not add extra LLM calls, retry loops, fallback prompts, feature flags, or
  model context-size assumptions.
- Do not increase local model `n_ctx`.
- Do not solve current-user conversation scope or initializer slot wording in
  this plan.
- Do not change persistence, scheduler, dispatcher, dialog, cognition, or RAG
  helper-agent routing.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the `Independent Code Review` gate and record the
  result in a completion record or handoff note.

## Must Do

- Add a facts-harvester-specific RAG payload projection.
- Use that projection in both `facts_harvester` and
  `fact_harvester_evaluator`.
- Preserve compact dedup evidence from `user_memory_unit_candidates` without
  passing raw documents.
- Add deterministic tests proving large raw candidate payloads do not reach the
  LLM prompt.
- Keep memory-unit merge behavior unchanged.

## Deferred

- Do not change `project_known_facts`.
- Do not change `user_image_retriever_agent`.
- Do not change `user_memory_unit_retrieval`.
- Do not change `persona_supervisor2_consolidator_memory_units`.
- Do not tune `RAG_SEARCH_*` or memory-context retrieval limits.
- Do not change `conversation_evidence_agent` speaker-scope parsing.
- Do not change `persona_supervisor2_rag_initializer`.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Facts harvester prompt payload | bigbang | Replace direct whole-`rag_result` serialization with the prompt-safe view. |
| Evaluator prompt payload | bigbang | Use the same prompt-safe view as the harvester. |
| Global RAG projection | compatible | Preserve existing `rag_result` shape and raw `user_memory_unit_candidates` for non-facts consumers. |
| Memory-unit merge path | compatible | Preserve existing raw surfaced-unit reuse. |
| Tests | bigbang | Add regression tests for the new prompt payload contract. |

## Cutover Policy Enforcement

- The implementation agent must follow the selected policy for each area.
- For bigbang areas, rewrite direct payload usage instead of preserving an old
  alternate code path.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to this policy requires user approval before implementation.

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve
  the contracts in this plan.
- The agent must not introduce new architecture, alternate migration
  strategies, compatibility layers, fallback paths, or extra features.
- The agent must treat changes outside the files listed in `Change Surface` as
  out of scope unless the plan is updated first.
- The agent may add small local helper functions inside
  `persona_supervisor2_consolidator_facts.py`; do not create a new shared module
  for this small bugfix.
- If an existing helper already exactly satisfies the compact projection
  contract, reuse it. Do not reuse helpers that preserve raw memory-unit fields.
- If the plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Target State

`ConsolidatorState["rag_result"]` may still contain full raw
`user_memory_unit_candidates`.

Before `facts_harvester` or `fact_harvester_evaluator` calls the LLM, the node
builds a prompt-safe RAG view. Heavy raw candidate fields such as `embedding`,
`content`, `source_refs`, `evidence_refs`, `merge_history`,
`subjective_appraisal`, `relationship_signal`, and arbitrary metadata do not
enter the facts prompt.

The facts prompt receives at most a capped compact candidate list. The memory
unit merge path still reads raw surfaced units from the original state.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Projection owner | Keep the new helper local to `persona_supervisor2_consolidator_facts.py`. | This prompt budget belongs to the facts/evaluator LLM boundary only. |
| Candidate handling | Compact and cap candidates instead of fully dropping them. | The facts harvester still needs duplicate-awareness hints, but not raw merge material. |
| Global RAG shape | Preserve unchanged. | Other consolidation stages depend on raw surfaced units. |
| Token budget strategy | Use deterministic character and item caps. | Avoid tokenizer dependency and local backend coupling. |
| Extra LLM calls | Add none. | This is a background prompt-shaping fix, not a new reasoning stage. |

## Contracts And Data Shapes

Add a local helper with this conceptual contract:

```python
def _facts_harvester_rag_view(rag_result: object) -> dict:
    """Return the bounded RAG view used by facts harvester prompts."""
```

The returned shape must preserve these keys when present:

```python
{
    "answer": str,
    "user_image": {
        "user_memory_context": dict,
    },
    "user_memory_unit_candidates": [
        {
            "unit_id": str,
            "unit_type": str,
            "fact": str,
            "dedup_key": str,
            "updated_at": str,
        },
    ],
    "memory_evidence": list,
    "recall_evidence": list,
    "conversation_evidence": list,
    "external_evidence": list,
    "supervisor_trace": dict,
}
```

Required caps:

- `user_memory_unit_candidates`: maximum 12 rows.
- Candidate `fact`: maximum 240 characters.
- Candidate `dedup_key`, `unit_id`, `unit_type`, `updated_at`: stripped strings
  only.
- Existing projected evidence fields still flow through
  `project_tool_result_for_llm`, but raw memory-unit candidate rows are compacted
  before entering the final prompt payload.

Forbidden prompt-facing fields under `user_memory_unit_candidates`:

```text
embedding
content
source_refs
evidence_refs
merge_history
subjective_appraisal
relationship_signal
```

## LLM Call And Context Budget

Affected calls:

| Call | Path | Before | After |
|---|---|---|---|
| `facts_harvester` | background consolidation | Whole `rag_result`, including raw candidate docs | Prompt-safe RAG view with capped compact candidates |
| `fact_harvester_evaluator` | background consolidation | Whole `rag_result`, including raw candidate docs | Same prompt-safe RAG view |

No new LLM calls are allowed. No response-path calls are affected.

Use `50k tokens` as the operational cap. The deterministic test gate must prove
that a synthetic oversized candidate set is reduced before `_facts_harvester_llm`
or `_fact_harvester_evaluator_llm` receives the payload. Exact token counting is
not required; the test must assert removed raw-marker text and bounded candidate
cardinality/field set.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`
  - Add local constants for candidate count and fact text caps.
  - Add local compact-candidate helper.
  - Add `_facts_harvester_rag_view`.
  - Replace both direct `project_tool_result_for_llm(state["rag_result"])`
    calls with `_facts_harvester_rag_view(state["rag_result"])`.
  - Update prompt input-format wording for `user_memory_unit_candidates` from
    raw candidates to bounded compact candidates.

- `tests/test_consolidator_facts_rag2.py`
  - Add harvester regression test for oversized raw candidates.
  - Add evaluator regression test for oversized raw candidates.

### Keep

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/rag/user_memory_unit_retrieval.py`
- `src/kazusa_ai_chatbot/rag/user_image_retriever_agent.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`

## Implementation Order

1. Inspect current facts harvester payload construction and existing tests.
   - Files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`,
     `tests/test_consolidator_facts_rag2.py`.
   - Evidence: note the current direct uses of `project_tool_result_for_llm`.

2. Add the failing harvester test.
   - File: `tests/test_consolidator_facts_rag2.py`.
   - Add `test_facts_harvester_compacts_raw_memory_unit_candidates`.
   - Fixture requirement: create at least 20 candidate rows with a long
     `content`, `subjective_appraisal`, `relationship_signal`, `source_refs`,
     and `embedding` marker.
   - Expected before implementation: captured payload includes forbidden raw
     fields or too many candidate rows.

3. Add the failing evaluator test.
   - File: `tests/test_consolidator_facts_rag2.py`.
   - Add `test_fact_harvester_evaluator_compacts_raw_memory_unit_candidates`.
   - Reuse the same oversized candidate fixture.
   - Expected before implementation: captured evaluator payload includes
     forbidden raw fields or too many candidate rows.

4. Run focused tests before implementation.
   - Command:
     `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py -q`
   - Expected: the two new tests fail for the current raw payload behavior.

5. Implement the local prompt-safe projection helper.
   - File: `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_facts.py`.
   - Keep helper private to this module.
   - Apply the required caps and forbidden-field removals from
     `Contracts And Data Shapes`.

6. Wire `facts_harvester` to use the prompt-safe view.
   - Replace the local `rag_result = project_tool_result_for_llm(...)` payload
     assignment with `rag_result = _facts_harvester_rag_view(...)`.

7. Wire `fact_harvester_evaluator` to use the same prompt-safe view.
   - Use the same helper, not a second implementation.

8. Update prompt contract text.
   - In both facts and evaluator prompt input-format sections, change the
     description of `user_memory_unit_candidates` to say bounded compact
     candidate memory units.
   - Do not otherwise rewrite the prompt.

9. Run focused tests after implementation.
   - Command:
     `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py -q`
   - Expected: pass.

10. Run regression tests.
    - Command:
      `venv\Scripts\python -m pytest tests\test_rag_projection.py tests\test_user_memory_units_rag_flow.py -q`
    - Expected: pass.

11. Run static sanity grep.
    - Command:
      `rg -n "user_memory_unit_candidates" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py tests\test_consolidator_facts_rag2.py`
    - Expected: matches are allowed only in the prompt-safe helper, prompt
      contract text, payload assembly, and tests.

12. Run independent code review.
    - Follow the `Independent Code Review` section before final sign-off.

## Progress Checklist

- [x] Stage 1 - focused tests added
  - Covers: implementation steps 1-4.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py -q`
  - Evidence: `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py -q`
    failed as expected with the two new compact-candidate tests seeing 20 raw
    `user_memory_unit_candidates` rows.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-05-13`.

- [x] Stage 2 - prompt-safe projection implemented and wired
  - Covers: implementation steps 5-8.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py -q`
  - Evidence: added `_facts_harvester_rag_view`,
    `_compact_memory_unit_candidates`, and `_compact_memory_unit_candidate`;
    wired `facts_harvester` and `fact_harvester_evaluator` to the safe view.
    `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py -q`
    passed with 4 tests.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-05-13`.

- [x] Stage 3 - regression verification complete
  - Covers: implementation steps 9-11.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_rag_projection.py tests\test_user_memory_units_rag_flow.py -q`
  - Evidence: focused tests passed with 4 tests; regression command
    `venv\Scripts\python -m pytest tests\test_rag_projection.py tests\test_user_memory_units_rag_flow.py -q`
    passed with 26 tests. Static grep for `user_memory_unit_candidates` only
    matched the local safe-view helper, prompt text, and regression tests.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-05-13`.

- [x] Stage 4 - independent code review complete
  - Covers: implementation step 12.
  - Verify: full diff reviewed against this plan and affected tests rerun after
    any review fixes.
  - Evidence: no separate reviewer was available under the active delegation
    policy, so Codex performed a fresh self-review against this plan. Review
    found one in-scope robustness/style issue: compact candidates should project
    selected allowed fields before stripping so timestamp formatting remains
    consistent. Fixed that and expanded helper docstrings. Reran
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py`,
    `venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py -q`,
    `venv\Scripts\python -m pytest tests\test_rag_projection.py tests\test_user_memory_units_rag_flow.py -q`,
    `rg -n "user_memory_unit_candidates" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py tests\test_consolidator_facts_rag2.py`,
    and `git diff --check`; all passed. Residual risk: no live LLM run was
    performed because this fix is a deterministic prompt-payload boundary.
  - Handoff: complete; plan moved to completed history.
  - Sign-off: `Codex/2026-05-13`.

## Verification

### Focused Tests

```powershell
venv\Scripts\python -m pytest tests\test_consolidator_facts_rag2.py -q
```

Expected: all tests pass after implementation.

### Regression Tests

```powershell
venv\Scripts\python -m pytest tests\test_rag_projection.py tests\test_user_memory_units_rag_flow.py -q
```

Expected: all tests pass after implementation.

### Static Sanity

```powershell
rg -n "user_memory_unit_candidates" src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_facts.py tests\test_consolidator_facts_rag2.py
```

Expected: matches are allowed only in the prompt-safe helper, prompt contract
text, payload assembly, and tests. Any direct whole-raw-candidate serialization
in facts/evaluator prompt construction is forbidden.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt,
  documentation, and command artifact.
- Code quality and design weaknesses, including ownership boundaries, hidden
  fallback paths, compatibility shims, prompt/RAG payload leaks, persistence
  risk, brittle fixtures, and avoidable blast radius.
- Alignment with `Must Do`, `Deferred`, `Agent Autonomy Boundaries`, `Change
  Surface`, exact contracts, implementation order, verification gates, and
  acceptance criteria.
- Regression and handoff quality, including focused and regression tests,
  execution evidence, and path-safe commands.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only fixture/documentation
corrections. If a fix would cross the approved boundary or alter the contract,
stop and update the plan or request approval before changing code.

## Acceptance Criteria

This plan is complete when:

- `facts_harvester` no longer serializes raw `user_memory_unit_candidates`.
- `fact_harvester_evaluator` no longer serializes raw
  `user_memory_unit_candidates`.
- Oversized synthetic candidate input remains bounded in captured LLM payloads.
- Existing RAG projection and memory-unit merge tests still pass.
- No model context-size increase is required.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Dedup signal becomes too weak after compaction | Preserve `unit_id`, `unit_type`, `fact`, `dedup_key`, and `updated_at` | Harvester payload test asserts compact fields remain |
| Memory-unit merge loses raw surfaced units | Do not change global RAG projection or memory-unit modules | `tests\test_user_memory_units_rag_flow.py` passes |
| Prompt contract drifts from actual payload | Update input-format text in both prompts | `tests\test_consolidator_facts_rag2.py` captures payload shape |
| Fix grows into unrelated RAG routing work | Explicit Deferred and Change Surface boundaries | Independent code review checks diff scope |

## Execution Handoff

This plan is complete. Future changes to facts consolidation prompt budgeting
must use a new active plan.
