# rag agent package reorganization plan

## Summary

- Goal: Reorganize flat RAG helper agents into ICD-backed capability packages without changing runtime behavior or LLM behavior.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `py-style`, `test-style-and-execution`, `local-llm-architecture`, `development-plan`.
- Overall cutover strategy: bigbang.
- Highest-risk areas: import rewiring, private test helper paths, prompt stability, dispatcher registry wiring, and deletion of old flat modules.
- Acceptance criteria: RAG behavior is unchanged, no LLM prompt/call/payload changes occur, package ICDs exist, old flat agent modules are gone, and deterministic verification passes.
- Independent plan review: completed in fallback fresh-review posture on 2026-05-28; findings were applied to this draft.

## Context

`kazusa_ai_chatbot.rag.web_agent3` already uses the desired local organization:

```text
web_agent3/
  README.md
  __init__.py
  agent.py
  contracts.py
  providers.py
  searxng_tools.py
  subagent/
```

The public caller sees one helper agent, `WebAgent3`, while the package owns its
internal router, contracts, provider dispatcher, and source subagents. This plan
applies the same organization principle to other RAG helper agents that are
currently flat modules.

The split is triggered by both line count and module-boundary practice.
`conversation_evidence_agent.py` is over 1000 lines and mixes selector,
projection, active-turn filtering, worker orchestration, and result shaping.
`memory_evidence_agent.py`, `person_context_agent.py`, `live_context_agent.py`,
and `recall_agent.py` are below 1000 lines but have stable sub-router, worker,
collector, or target-resolution boundaries.

The global RAG initializer and dispatcher are not part of the package discovery
model. They remain explicit system boundaries. Registry membership remains
auditable and hand-written.

## Mandatory Skills

- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `local-llm-architecture`: load before touching RAG helper stages, selectors, prompts, graph dispatch, or LLM-backed code.
- `development-plan`: load before execution, verification, lifecycle updates, and final sign-off.

## Mandatory Rules

- This plan is structural only. Do not change runtime behavior.
- Do not change LLM behavior:
  - no prompt text edits;
  - no prompt formatting changes;
  - no changed `SystemMessage` or `HumanMessage` payload shape;
  - no changed expected JSON schemas;
  - no new LLM calls;
  - no removed LLM calls;
  - no changed LLM call ordering;
  - no changed model route, temperature, top-p, retry, parser, or repair behavior.
- Move prompt constants, LLM instances, handlers, and parsers together so each LLM-backed stage remains locally inspectable.
- Preserve dispatcher-visible agent names, worker names, cache names, fact-source metadata, result payloads, and `.run(task, context, max_attempts)` signatures.
- Use bigbang cutover. Do not create old flat module re-export files, compatibility shims, fallback imports, dual paths, or temporary adapters.
- Keep the global RAG initializer and dispatcher explicit. Do not add global RAG agent auto-discovery.
- `web_agent3` source auto-discovery remains local to `web_agent3`; do not generalize it upward.
- Each grouped capability package must include a local `README.md` ICD.
- Treat changes outside the RAG package and dispatcher wiring as high-scrutiny changes.
- Do not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, cache-policy changes, database changes, or behavior tuning.
- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the `Independent Code Review` gate and record the result in `Execution Evidence`.
- The plan's `Execution Model` uses parent-led native subagent execution unless the user explicitly approves fallback execution.

## Must Do

- Create ICD-backed packages for:
  - `kazusa_ai_chatbot.rag.conversation_evidence`;
  - `kazusa_ai_chatbot.rag.memory_evidence`;
  - `kazusa_ai_chatbot.rag.person_context`;
  - `kazusa_ai_chatbot.rag.live_context`;
  - `kazusa_ai_chatbot.rag.recall`.
- Move existing code into package modules without semantic edits.
- Update all production imports and test imports to the new package paths.
- Delete old flat agent modules after reference greps show they are unused.
- Preserve the explicit global dispatcher registry and existing registry keys.
- Add package README ICDs that document public contract, ownership, internal flow, modules, worker roster, cache policy, LLM/prompt policy, and verification gates.
- Run all verification commands listed in this plan.
- Record prompt and LLM-payload baselines before moving LLM-backed modules, then compare after the bigbang move before signing off the affected stage.

## Deferred

- Do not redesign RAG2 initializer, dispatcher, evaluator, finalizer, projection, or Cache2.
- Do not add global RAG agent discovery.
- Do not tune prompts or alter prompt contracts.
- Do not add new helper agents, new worker modes, new retries, new fallback paths, or new repair stages.
- Do not change legacy slot-prefix routing behavior during this refactor.
- Do not move shared non-agent utilities unless required by a moved import and listed in `Change Surface`.
- Do not update unrelated active plans or archived plans.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| RAG helper module paths | bigbang | Replace flat imports with package imports in one change. |
| Old flat modules | bigbang | Delete old flat agent files after references are removed. |
| Dispatcher registry | bigbang | Keep explicit registry keys and update class import paths only. |
| Lower-level worker names | bigbang | Preserve existing worker names in result payloads and registry keys. |
| LLM prompts and calls | preserve | Move byte-for-byte; do not alter model-facing content or call sequence. |
| Tests | bigbang | Update import paths and monkeypatch paths; preserve behavioral expectations. |
| ICDs | bigbang | Add README ICDs before final sign-off. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- The agent must not choose a more conservative strategy by default.
- If an area is `bigbang`, delete or rewrite legacy references instead of preserving them.
- If an area is `preserve`, move the existing contract without behavior changes.
- Any change to a cutover policy requires user approval before implementation.

## Target State

Public imports become package imports:

```python
from kazusa_ai_chatbot.rag.conversation_evidence import ConversationEvidenceAgent
from kazusa_ai_chatbot.rag.memory_evidence import MemoryEvidenceAgent
from kazusa_ai_chatbot.rag.person_context import PersonContextAgent
from kazusa_ai_chatbot.rag.live_context import LiveContextAgent
from kazusa_ai_chatbot.rag.recall import RecallAgent
```

Target package layout:

```text
src/kazusa_ai_chatbot/rag/conversation_evidence/
  README.md
  __init__.py
  agent.py
  selector.py
  contracts.py
  projection.py
  active_turn_filter.py
  workers/
    __init__.py
    search.py
    filter.py
    aggregate.py
    keyword.py

src/kazusa_ai_chatbot/rag/memory_evidence/
  README.md
  __init__.py
  agent.py
  selector.py
  contracts.py
  projection.py
  workers/
    __init__.py
    persistent_search.py
    persistent_keyword.py
    user_memory.py

src/kazusa_ai_chatbot/rag/person_context/
  README.md
  __init__.py
  agent.py
  selector.py
  contracts.py
  projection.py
  workers/
    __init__.py
    lookup.py
    list.py
    profile.py
    relationship.py
    image.py

src/kazusa_ai_chatbot/rag/live_context/
  README.md
  __init__.py
  agent.py
  selector.py
  runtime_facts.py
  target_resolution.py

src/kazusa_ai_chatbot/rag/recall/
  README.md
  __init__.py
  agent.py
  contracts.py
  review.py
  collectors/
    __init__.py
    progress.py
    commitments.py
    scheduled_events.py
    history.py
```

Shared RAG utilities remain flat unless a move is required by ownership:

```text
cache2_events.py
cache2_policy.py
cache2_runtime.py
cognitive_episode_adapter.py
continuation.py
evidence_coverage.py
evidence_formatting.py
helper_agent.py
hybrid_retrieval.py
memory_retrieval_tools.py
prompt_projection.py
quote_aware_sequence.py
search_runtime.py
user_memory_unit_retrieval.py
web_agent3/
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Split trigger | Split by line count and ownership boundary | Files over 1000 lines require review, and sub-router/worker ownership justifies packages before that threshold. |
| Compatibility | No compatibility layer | User explicitly selected bigbang migration. |
| ICDs | One README per grouped package | Mirrors `web_agent3` and keeps agent ownership inspectable. |
| Global discovery | Forbidden | The global RAG boundary must remain explicit for local-LLM reliability and operational auditability. |
| Local discovery | Keep only existing `web_agent3` source discovery | It is local to a source router and already documented. |
| LLM behavior | Preserve exactly | This plan is a structural refactor only. |
| Worker names | Preserve existing names | Downstream tests, traces, cache stats, and projections inspect these names. |
| Shared utilities | Keep shared utilities flat | The goal is agent organization, not a full RAG utility restructure. |

## Contracts And Data Shapes

All grouped agents keep the standard RAG helper contract:

```python
async def run(
    task: str,
    context: dict[str, Any],
    max_attempts: int = 1,
) -> dict[str, Any]:
    ...
```

`web_agent3` keeps its existing `max_attempts` behavior and public contract.

Top-level capability result shape remains:

```python
{
    "resolved": bool,
    "result": {
        "selected_summary": str,
        "capability": str,
        "primary_worker": str,
        "supporting_workers": list[str],
        "source_policy": str,
        "resolved_refs": list[dict],
        "projection_payload": dict,
        "worker_payloads": dict,
        "evidence": list[str],
        "missing_context": list[str],
        "conflicts": list[str],
        "observation_candidates": list[dict],
        "source_hints": list[dict],
    },
    "attempts": int,
    "cache": dict,
}
```

Worker result names remain unchanged:

```text
conversation_search_agent
conversation_filter_agent
conversation_aggregate_agent
conversation_keyword_agent
persistent_memory_search_agent
persistent_memory_keyword_agent
user_memory_evidence_agent
user_lookup_agent
user_list_agent
user_profile_agent
relationship_agent
runtime_context_provider
web_agent3
recall_agent
```

ICD contract for each package README:

- public import path;
- `.run(...)` signature;
- result shape;
- semantic ownership;
- non-ownership and refusal conditions;
- internal flow;
- module ownership table;
- worker or collector roster;
- cache policy;
- LLM/prompt policy when applicable;
- verification list.

## LLM Call And Context Budget

This plan must not change the LLM call budget.

| Stage | Before | After | Instruction |
|---|---|---|---|
| Conversation evidence selector | Existing call count, prompt, payload, parser | Same | Move only. |
| Memory evidence selector | Existing call count, prompt, payload, parser | Same | Move only. |
| Person context selector | Existing call count, prompt, payload, parser | Same | Move only. |
| Live context selector | Existing call count, prompt, payload, parser | Same | Move only. |
| Recall review | Existing call count, prompt, payload, parser | Same | Move only. |
| Worker generators and judges | Existing call count, prompt, payload, parser | Same | Move only. |
| Global RAG initializer and dispatcher | Existing call count, prompt, payload, parser | Same | Import path update only. |

Verification requires prompt stability checks and focused tests that exercise moved LLM-backed handlers with patched LLMs. Live LLM tests are not required for this structural refactor unless deterministic verification reveals prompt or payload drift.

Prompt stability audit contract:

- Before moving modules, parent records a local baseline for prompt constants and representative handler payloads in `test_artifacts/diagnostics/rag_agent_package_prompt_baseline.json`.
- The committed comparison fixture lives at `tests/fixtures/rag_agent_package_prompt_baseline.json`.
- The baseline must include every moved prompt constant and at least one patched-LLM representative payload for each moved LLM-backed selector, generator, judge, or review handler.
- After moving modules, parent reruns the same audit and compares exact prompt text plus JSON payload keys and values against the committed fixture.
- The local baseline artifact is execution evidence, not a production source artifact. It must not be imported by production code.
- The committed fixture is test source and must not be generated implicitly during normal pytest execution.
- Any prompt text, prompt formatting, message role, payload key, payload value, model route, parser, retry, or call-count drift is a blocker unless the user explicitly approves an LLM change.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/rag/conversation_evidence/README.md`: ICD for conversation evidence package.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/__init__.py`: public exports for `ConversationEvidenceAgent` and test-needed helpers.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/agent.py`: public capability wrapper and run orchestration.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/selector.py`: existing deterministic and LLM worker selection logic.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/contracts.py`: typed projection and selector contracts.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/projection.py`: worker result to evidence projection logic.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/active_turn_filter.py`: active-turn exclusion helpers.
- `src/kazusa_ai_chatbot/rag/conversation_evidence/workers/*.py`: moved conversation worker agents.
- `src/kazusa_ai_chatbot/rag/memory_evidence/README.md`: ICD for memory evidence package.
- `src/kazusa_ai_chatbot/rag/memory_evidence/__init__.py`: public exports for `MemoryEvidenceAgent` and test-needed helpers.
- `src/kazusa_ai_chatbot/rag/memory_evidence/agent.py`: public capability wrapper and run orchestration.
- `src/kazusa_ai_chatbot/rag/memory_evidence/selector.py`: existing deterministic and LLM worker selection logic.
- `src/kazusa_ai_chatbot/rag/memory_evidence/contracts.py`: memory projection contracts.
- `src/kazusa_ai_chatbot/rag/memory_evidence/projection.py`: worker result to memory evidence projection logic.
- `src/kazusa_ai_chatbot/rag/memory_evidence/workers/*.py`: moved persistent-memory and user-memory workers.
- `src/kazusa_ai_chatbot/rag/person_context/README.md`: ICD for person context package.
- `src/kazusa_ai_chatbot/rag/person_context/__init__.py`: public exports for `PersonContextAgent` and test-needed helpers.
- `src/kazusa_ai_chatbot/rag/person_context/agent.py`: public capability wrapper and run orchestration.
- `src/kazusa_ai_chatbot/rag/person_context/selector.py`: existing deterministic and LLM mode selection logic.
- `src/kazusa_ai_chatbot/rag/person_context/contracts.py`: person reference and projection contracts.
- `src/kazusa_ai_chatbot/rag/person_context/projection.py`: worker result to person context projection logic.
- `src/kazusa_ai_chatbot/rag/person_context/workers/*.py`: moved person workers.
- `src/kazusa_ai_chatbot/rag/live_context/README.md`: ICD for live context package.
- `src/kazusa_ai_chatbot/rag/live_context/__init__.py`: public exports for `LiveContextAgent` and test-needed helpers.
- `src/kazusa_ai_chatbot/rag/live_context/agent.py`: public capability wrapper and run orchestration.
- `src/kazusa_ai_chatbot/rag/live_context/selector.py`: existing deterministic and LLM live-plan selection logic.
- `src/kazusa_ai_chatbot/rag/live_context/runtime_facts.py`: runtime date/time facts.
- `src/kazusa_ai_chatbot/rag/live_context/target_resolution.py`: memory/conversation target-scope resolution helpers.
- `src/kazusa_ai_chatbot/rag/recall/README.md`: ICD for recall package.
- `src/kazusa_ai_chatbot/rag/recall/__init__.py`: public exports for `RecallAgent` and test-needed helpers.
- `src/kazusa_ai_chatbot/rag/recall/agent.py`: public recall wrapper and run orchestration.
- `src/kazusa_ai_chatbot/rag/recall/contracts.py`: recall candidate and result contracts.
- `src/kazusa_ai_chatbot/rag/recall/review.py`: existing LLM candidate review logic.
- `src/kazusa_ai_chatbot/rag/recall/collectors/*.py`: moved recall collectors.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`: update imports to package paths while preserving registry keys and fact-source metadata.
- `src/kazusa_ai_chatbot/rag/README.md`: document package layout and split rule.
- `development_plans/README.md`: add this plan to Active Short-Term Plans.
- `tests/test_rag_agent_package_prompt_stability.py`: add or update a deterministic prompt-stability audit test that records and compares the moved LLM-backed prompt and payload contracts during execution.
- Tests under `tests/`: update import paths, monkeypatch paths, and private-helper module paths to moved locations without changing expected behavior.

### Delete

- `src/kazusa_ai_chatbot/rag/conversation_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/conversation_search_agent.py`
- `src/kazusa_ai_chatbot/rag/conversation_filter_agent.py`
- `src/kazusa_ai_chatbot/rag/conversation_aggregate_agent.py`
- `src/kazusa_ai_chatbot/rag/conversation_keyword_agent.py`
- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/persistent_memory_search_agent.py`
- `src/kazusa_ai_chatbot/rag/persistent_memory_keyword_agent.py`
- `src/kazusa_ai_chatbot/rag/user_memory_evidence_agent.py`
- `src/kazusa_ai_chatbot/rag/person_context_agent.py`
- `src/kazusa_ai_chatbot/rag/user_lookup_agent.py`
- `src/kazusa_ai_chatbot/rag/user_list_agent.py`
- `src/kazusa_ai_chatbot/rag/user_profile_agent.py`
- `src/kazusa_ai_chatbot/rag/relationship_agent.py`
- `src/kazusa_ai_chatbot/rag/user_image_retriever_agent.py`
- `src/kazusa_ai_chatbot/rag/live_context_agent.py`
- `src/kazusa_ai_chatbot/rag/recall_agent.py`

### Keep

- `src/kazusa_ai_chatbot/rag/web_agent3/**`: existing package remains as-is.
- Shared RAG utility modules listed under `Target State`: keep flat unless direct import movement requires local import updates.

## Overdesign Guardrail

- Actual problem: RAG helper agents are difficult to inspect because capability agents and their similar subagents are spread across flat files, with `conversation_evidence_agent.py` already exceeding 1000 lines.
- Minimal change: move existing agent code into capability packages with local ICDs and update imports in one bigbang cutover.
- Ownership boundaries: global RAG initializer and dispatcher select top-level capabilities explicitly; capability packages own local selector/router logic; workers own low-level retrieval parameters; deterministic code owns validation, cache mechanics, projection, and execution.
- Rejected complexity: no global auto-discovery, no compatibility modules, no prompt tuning, no new agent modes, no helper-agent behavior changes, no fallback imports, no retry changes, no shared generic package framework.
- Evidence threshold: add new discovery or shared agent framework only after an approved future plan identifies repeated boilerplate across at least three completed package migrations and proves it does not weaken explicit global routing.

## Agent Autonomy Boundaries

- The responsible agent may choose local file movement mechanics only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate migration strategies, compatibility layers, fallback paths, or extra features.
- The responsible agent must treat changes outside the target RAG package and dispatcher import wiring as high-scrutiny changes.
- The responsible agent may remove old flat modules because deletion is explicitly in scope and must be verified by greps and tests.
- The responsible agent must search for existing equivalent helpers before extracting any helper into a package module.
- The responsible agent must not make private helpers public solely to preserve old test access. Move tests to the new owning module when they intentionally test a private helper, or prefer public contract tests when the helper is no longer the right boundary.
- The responsible agent must not perform unrelated cleanup, formatting churn, dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Implementation Order

1. Establish baseline.
   - Run the focused tests listed under `Verification`.
   - Record pass/fail status in `Execution Evidence`.
   - Run static greps for old flat imports and record current matches.
2. Add prompt stability audit tests.
   - Add `tests/test_rag_agent_package_prompt_stability.py` with deterministic checks that moved prompt constants and representative LLM payload shapes remain unchanged.
   - Run the new checks against the pre-move code and record the baseline.
3. Create the `conversation_evidence` package.
   - Move code from the conversation flat modules into the target package.
   - Add `README.md` ICD.
   - Update production imports and tests.
   - Delete old conversation flat modules.
   - Run conversation-focused verification.
4. Create the `memory_evidence` package.
   - Move code from memory evidence and memory worker flat modules.
   - Add `README.md` ICD.
   - Update production imports and tests.
   - Delete old memory flat modules.
   - Run memory-focused verification.
5. Create the `person_context` package.
   - Move code from person context and person worker flat modules.
   - Add `README.md` ICD.
   - Update production imports and tests.
   - Delete old person flat modules.
   - Run person-focused verification.
6. Create the `live_context` package.
   - Move code from `live_context_agent.py`.
   - Add `README.md` ICD.
   - Update production imports and tests.
   - Delete old live-context flat module.
   - Run live-context focused verification.
7. Create the `recall` package.
   - Move code from `recall_agent.py`.
   - Add `README.md` ICD.
   - Update production imports and tests.
   - Delete old recall flat module.
   - Run recall focused verification.
8. Update RAG package documentation.
   - Update `src/kazusa_ai_chatbot/rag/README.md`.
   - Verify it describes the package boundary and split rule without authorizing behavior changes.
9. Run global static checks.
   - Verify no old flat module import paths remain.
   - Verify global dispatcher registry remains explicit.
   - Verify no global auto-discovery was added.
10. Run full deterministic verification.
11. Run independent code review.
12. Remediate in-scope review findings and rerun affected verification.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the focused test contract is established; owns production code changes only; does not edit tests unless the parent explicitly directs it; closes after planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks, and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after planned verification passes; reviews the plan, diff, and evidence; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - baseline and prompt stability contract established.
  - Covers: implementation steps 1-2.
  - Verify: baseline focused tests and prompt stability checks run, and `test_artifacts/diagnostics/rag_agent_package_prompt_baseline.json` is recorded as execution evidence.
  - Evidence: record command output, baseline grep results, and baseline artifact path in `Execution Evidence`.
  - Sign-off: parent agent/2026-05-28 after evidence is recorded.
- [x] Stage 2 - `conversation_evidence` package migrated.
  - Covers: implementation step 3.
  - Verify: conversation-focused tests pass and old conversation flat imports are gone.
  - Evidence: record changed files, greps, and test output.
  - Sign-off: parent agent/2026-05-28 after evidence is recorded.
- [x] Stage 3 - `memory_evidence` package migrated.
  - Covers: implementation step 4.
  - Verify: memory-focused tests pass and old memory flat imports are gone.
  - Evidence: record changed files, greps, and test output.
  - Sign-off: parent agent/2026-05-29 after user accepted the two pre-existing `test_user_memory_evidence_agent.py` expectation failures as blockers; migration code and import checks are otherwise complete.
- [x] Stage 4 - `person_context` package migrated.
  - Covers: implementation step 5.
  - Verify: person-focused tests pass and old person flat imports are gone.
  - Evidence: record changed files, greps, and test output.
  - Sign-off: parent agent/2026-05-28 after evidence is recorded.
- [x] Stage 5 - `live_context` package migrated.
  - Covers: implementation step 6.
  - Verify: live-context focused tests pass and old live-context flat imports are gone.
  - Evidence: record changed files, greps, and test output.
  - Sign-off: parent agent/2026-05-28 after evidence is recorded.
- [x] Stage 6 - `recall` package migrated.
  - Covers: implementation step 7.
  - Verify: recall focused tests pass and old recall flat imports are gone.
  - Evidence: record changed files, greps, and test output.
  - Sign-off: parent agent/2026-05-28 after evidence is recorded.
- [x] Stage 7 - documentation and global registry verification complete.
  - Covers: implementation steps 8-9.
  - Verify: static greps pass and RAG README reflects the target state.
  - Evidence: record grep output and doc files changed.
  - Sign-off: parent agent/2026-05-28 after evidence is recorded.
- [x] Stage 8 - full deterministic verification complete.
  - Covers: implementation step 10.
  - Verify: all commands in `Verification` pass or have recorded, approved blockers.
  - Evidence: record command output.
  - Sign-off: parent agent/2026-05-29 after user accepted the documented pre-existing and unrelated blockers.
- [x] Stage 9 - independent code review complete.
  - Covers: implementation steps 11-12.
  - Verify: review subagent approves or all in-scope findings are fixed and affected verification is rerun.
  - Evidence: record review findings, fixes, rerun commands, residual risks, and approval status.
  - Sign-off: parent agent/2026-05-29 after in-scope review findings were fixed and affected verification was rerun.

## Verification

### Static Greps

Run:

```powershell
rg "kazusa_ai_chatbot\.rag\.(conversation_evidence_agent|conversation_search_agent|conversation_filter_agent|conversation_aggregate_agent|conversation_keyword_agent|memory_evidence_agent|persistent_memory_search_agent|persistent_memory_keyword_agent|user_memory_evidence_agent|person_context_agent|user_lookup_agent|user_list_agent|user_profile_agent|relationship_agent|user_image_retriever_agent|live_context_agent|recall_agent)" src tests
```

Expected: no matches. Exit code 1 from `rg` is acceptable only when it means no matches.

Run:

```powershell
rg "from kazusa_ai_chatbot\.rag import (conversation_evidence_agent|conversation_search_agent|conversation_filter_agent|conversation_aggregate_agent|conversation_keyword_agent|memory_evidence_agent|persistent_memory_search_agent|persistent_memory_keyword_agent|user_memory_evidence_agent|person_context_agent|user_lookup_agent|user_list_agent|user_profile_agent|relationship_agent|user_image_retriever_agent|live_context_agent|recall_agent)" src tests
```

Expected: no matches. Exit code 1 from `rg` is acceptable only when it means no matches.

Run:

```powershell
rg "_discover|iter_modules|pkgutil" src/kazusa_ai_chatbot/rag src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py
```

Expected: matches are allowed only inside `src/kazusa_ai_chatbot/rag/web_agent3/subagent/__init__.py`. Any match in the global RAG dispatcher or new capability packages is a blocker.

Run:

```powershell
rg "_GENERATOR_PROMPT|_SELECTOR_PROMPT|SystemMessage|HumanMessage" src/kazusa_ai_chatbot/rag
```

Expected: LLM-backed prompt constants and message construction still exist in moved modules. Review the diff to confirm prompt text and payload shapes are unchanged.

### Prompt Stability Audit

Run before the move:

```powershell
venv\Scripts\python -m pytest tests/test_rag_agent_package_prompt_stability.py -q
```

Expected before implementation: pass and write `test_artifacts/diagnostics/rag_agent_package_prompt_baseline.json` as execution evidence. Normal post-review pytest execution compares against the committed fixture at `tests/fixtures/rag_agent_package_prompt_baseline.json` and must not silently generate a new baseline.

Run after each package stage that moves LLM-backed code:

```powershell
venv\Scripts\python -m pytest tests/test_rag_agent_package_prompt_stability.py -q
```

Expected after implementation: pass with exact prompt text and representative payload equivalence against the baseline artifact.

### Focused Tests

Run:

```powershell
venv\Scripts\python -m pytest tests/test_rag_phase3_capability_agents.py -q
venv\Scripts\python -m pytest tests/test_rag_hybrid_agents.py -q
venv\Scripts\python -m pytest tests/test_rag_helper_arg_boundaries.py -q
venv\Scripts\python -m pytest tests/test_rag_recall_agent.py -q
venv\Scripts\python -m pytest tests/test_user_memory_evidence_agent.py -q
venv\Scripts\python -m pytest tests/test_user_profile_agent.py -q
venv\Scripts\python -m pytest tests/test_llm_time_payload_projection.py -q
```

Expected: all pass.

### Integration And Projection Tests

Run:

```powershell
venv\Scripts\python -m pytest tests/test_rag_projection.py -q
venv\Scripts\python -m pytest tests/test_rag_initializer_cache2.py -q
venv\Scripts\python -m pytest tests/test_persona_supervisor2_rag2_integration.py -q
venv\Scripts\python -m pytest tests/test_rag_phase3_supervisor_integration.py -q
```

Expected: all pass.

### Broad Deterministic Suite

Run:

```powershell
venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q
```

Expected: all pass. If unrelated pre-existing failures appear, record the exact failing tests and prove the failure is unrelated to this plan before sign-off.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. The parent agent must create one independent code-review subagent through the current harness's native subagent capability. If native subagents are unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed Python, test, prompt, documentation, and command artifact.
- Plan alignment with `Must Do`, `Deferred`, `Change Surface`, cutover policy, no-LLM-change rule, implementation order, verification gates, and acceptance criteria.
- Code quality and design weaknesses, including ownership boundaries, hidden fallback paths, compatibility shims, prompt/RAG payload leaks, persistence risk, brittle fixtures, and avoidable blast radius.
- Regression and handoff quality, including focused and regression tests, execution evidence, static-grep expectations, and path-safe commands.

The parent agent fixes concrete findings directly only when the fix is inside the approved change surface or this review gate explicitly allows review-only fixture/documentation corrections. If a fix would cross the approved boundary or alter the contract, stop and update the plan or request approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in `Execution Evidence`.

## Independent Plan Review

Review performed on 2026-05-28 in fallback fresh-review posture because the user requested plan review and update but did not request delegated subagent work.

Review scope:

- Checked this plan against `development-plan`, `plan_contract.md`, `execution_gates.md`, and `cutover_policy.md`.
- Checked architecture alignment with the agreed constraints: bigbang migration, no backward compatibility, no functional change, no LLM change, ICD per grouped agent, and no global RAG auto-discovery.
- Checked available source references and current flat import patterns with `rg`.
- Checked the plan for unresolved choices and placeholders.

Findings applied:

- Added a first-class prompt stability audit contract and named baseline artifact.
- Added a second old-import static grep for `from kazusa_ai_chatbot.rag import old_module` patterns.
- Added a private-helper test boundary rule so execution agents do not create compatibility exports just to satisfy old tests.
- Added this independent plan review section and recorded the review scope.

Approval status: execution approved by user command on 2026-05-28; plan status moved to `in_progress`.

## Acceptance Criteria

This plan is complete when:

- The target RAG package layout exists.
- Every grouped package has a local `README.md` ICD.
- Old flat RAG agent modules listed in `Delete` are removed.
- No production or test source imports the removed flat module paths.
- The global RAG dispatcher remains explicit and has no global agent auto-discovery.
- Dispatcher-visible agent names, worker names, result payloads, cache names, and fact-source metadata remain unchanged.
- No LLM prompt text, LLM payload shape, LLM call count, LLM call order, model route, parser, or retry behavior changed.
- All verification commands pass or any unrelated pre-existing blocker is documented with evidence and user approval.
- Independent code review is complete and approved.

## Execution Evidence

- Execution started: 2026-05-28 on branch `rag-agent-package-reorganization` after explicit user command to execute the plan with subagents.
- Baseline tests: `venv\Scripts\python -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_hybrid_agents.py tests/test_rag_helper_arg_boundaries.py tests/test_rag_recall_agent.py tests/test_user_memory_evidence_agent.py tests/test_user_profile_agent.py tests/test_llm_time_payload_projection.py -q` ran before production-code changes; result was 167 passed and 2 failed. The failures reproduced in `tests/test_user_memory_evidence_agent.py::test_user_memory_evidence_exact_cjk_term_uses_scoped_keyword` and `tests/test_user_memory_evidence_agent.py::test_user_memory_evidence_agent_collects_multiple_literal_anchor_hits`. Both failures are pre-existing baseline expectation drift: current `user_memory_evidence_agent` includes subjective appraisal and continuity signal lines in `selected_summary`, while the tests assert fact-only summaries.
- Prompt stability audit: `venv\Scripts\python -m py_compile tests/test_rag_agent_package_prompt_stability.py` passed. `venv\Scripts\python -m pytest tests/test_rag_agent_package_prompt_stability.py -q` passed and created `test_artifacts/diagnostics/rag_agent_package_prompt_baseline.json`; post-review remediation copied the stable expected baseline to committed fixture `tests/fixtures/rag_agent_package_prompt_baseline.json`.
- Production-code subagent: native subagent `019e6e5a-9c6b-7113-ba87-aa865b64c492` migrated the approved production change surface into package modules, deleted the old flat files, and reported py_compile/import/static-grep checks. Parent inspection found initial shell ownership modules, requested remediation from the same subagent, then integrated the corrected moved logic.
- Parent integration fixes after production-code subagent: added missing moved imports and constants discovered by AST unresolved-name scan; fixed `kazusa_ai_chatbot.rag.recall.review` logger import; retargeted recall tests to the new collector and agent hook modules instead of adding package-level compatibility exports. Post-fix AST unresolved-name scan over the new package modules produced no unresolved names.
- Package layout audit: new package README ICDs exist for `conversation_evidence`, `memory_evidence`, `person_context`, `live_context`, and `recall`. New package Python line-count audit found no file over 1000 lines; largest files were `conversation_evidence/workers/search.py` at 708 lines and `conversation_evidence/projection.py` at 652 lines.
- Independent plan review: 2026-05-28 fallback fresh-review posture; findings applied in this draft.
- Package migration stages: `conversation_evidence`, `memory_evidence`, `person_context`, `live_context`, and `recall` package trees were created with local ICDs. Dispatcher imports now use package public imports while preserving registry keys and worker names. Old flat source modules listed in `Delete` are removed in this branch. Memory package sign-off remains blocked only by the two pre-existing memory-evidence expectation failures recorded in baseline.
- Static grep results: baseline old-path greps intentionally matched current flat imports in RAG flat modules, dispatcher imports, and tests. Baseline discovery grep matched only `src/kazusa_ai_chatbot/rag/web_agent3/subagent/__init__.py`, which is allowed.
- Static grep results after migration: old flat module path grep over `src tests` returned no matches; `from kazusa_ai_chatbot.rag import <old_module>` grep over `src tests` returned no matches; `_discover|iter_modules|pkgutil` grep matched only `src/kazusa_ai_chatbot/rag/web_agent3/subagent/__init__.py`; no compatibility/shim/fallback-import/discovery wording matched in the new packages or global dispatcher.
- Prompt stability after migration: `venv\Scripts\python -m pytest tests/test_rag_agent_package_prompt_stability.py -q` passed. Prompt/message grep confirmed moved LLM-backed stages still construct `SystemMessage` and `HumanMessage` in the new package modules, with `web_agent3` unchanged.
- Focused test results: after migration and parent integration fixes, `venv\Scripts\python -m pytest tests/test_rag_phase3_capability_agents.py tests/test_rag_hybrid_agents.py tests/test_rag_helper_arg_boundaries.py tests/test_rag_recall_agent.py tests/test_user_memory_evidence_agent.py tests/test_user_profile_agent.py tests/test_llm_time_payload_projection.py -q --tb=short` returned 167 passed and the same 2 pre-existing baseline failures in `tests/test_user_memory_evidence_agent.py::test_user_memory_evidence_exact_cjk_term_uses_scoped_keyword` and `tests/test_user_memory_evidence_agent.py::test_user_memory_evidence_agent_collects_multiple_literal_anchor_hits`. `venv\Scripts\python -m pytest tests/test_rag_recall_agent.py -q --tb=short` returned 14 passed after retargeting monkeypatch paths to the new recall collector modules.
- Integration test results: baseline parent-side run `venv\Scripts\python -m pytest tests/test_rag_projection.py tests/test_rag_initializer_cache2.py tests/test_persona_supervisor2_rag2_integration.py tests/test_rag_phase3_supervisor_integration.py -q` ran while production worker was active; result was 74 passed and 6 failed before production-code migration landed. Failures were prompt-contract assertions in `tests/test_rag_initializer_cache2.py`: `test_initializer_prompt_documents_profile_evidence_dependency`, `test_memory_evidence_prompt_uses_capability_contract`, `test_initializer_prompt_declares_recall_route`, `test_initializer_prompt_uses_conversation_speaker_scope_contract`, `test_initializer_prompt_documents_self_word_active_character_route`, and `test_initializer_prompt_documents_live_external_fact_contract`.
- Integration test results after migration: `venv\Scripts\python -m pytest tests/test_rag_projection.py tests/test_rag_initializer_cache2.py tests/test_persona_supervisor2_rag2_integration.py tests/test_rag_phase3_supervisor_integration.py -q --tb=short` returned 74 passed and the same 6 pre-existing initializer prompt-contract assertion failures listed above.
- Syntax verification: `venv\Scripts\python -m py_compile` over the new package Python files plus changed package-specific tests passed.
- Broad deterministic suite: `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q --tb=short` returned 1710 passed, 23 failed, and 268 deselected. The 23 failures include the 8 RAG baseline failures above and 15 additional failures outside this plan's changed source surface: `tests/test_conversation_progress_cognition.py::test_contextual_agent_receives_boundary_profile_contract`; `tests/test_global_character_growth_replay.py::test_l2_receives_promoted_global_growth_for_user_message`; `tests/test_global_character_growth_replay.py::test_l2_keeps_global_growth_absent_when_not_projected`; `tests/test_global_character_growth_replay.py::test_l2_prompt_mentions_promoted_global_growth_as_general_context`; `tests/test_memory_retrieval_tools.py::test_search_conversation_delegates_to_vector_history_search`; `tests/test_memory_retrieval_tools.py::test_search_conversation_keyword_delegates_to_keyword_history_search`; `tests/test_memory_retrieval_tools.py::test_get_conversation_filters_and_strips_internal_fields`; `tests/test_memory_retrieval_tools.py::test_conversation_message_payload_projects_image_blocks_from_attachments`; `tests/test_memory_retrieval_tools.py::test_conversation_message_payload_drops_raw_attachment_url_and_storage_ids`; `tests/test_multi_source_cognition_stage_07_reflection_dry_run.py::test_text_chat_prompt_fingerprints_remain_stable`; `tests/test_multi_source_cognition_stage_08_internal_thought_dry_run.py::test_text_chat_and_reflection_prompt_fingerprints_remain_stable`; `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py::test_l2a_multimodal_user_turn_keeps_promoted_reflection_context`; `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py::test_existing_l1_l2_l3_prompt_bytes_are_unchanged`; `tests/test_persona_supervisor2_action_initializer.py::test_action_initializer_prompt_follows_cognition_prompt_structure`; and `tests/test_rag_continuation.py::test_empty_continuation_decision_shape`. Representative reruns show these assertions concern unchanged shared retrieval projection, continuation shape, and non-RAG cognition/action prompt contracts. `git diff --name-only` shows the migration did not modify those production modules or test files.
- User blocker acceptance: on 2026-05-29, the user accepted the documented pre-existing and unrelated verification failures as blockers for this migration. Stage 8 is signed off with those accepted blockers; no production behavior changes were made to address them in this plan.
- Independent code review: native review subagent `019e6e80-14df-75a0-ae79-c93a4b934063` found no critical issues and two important issues. First, `tests/test_rag_agent_package_prompt_stability.py` silently created its ignored baseline when missing, weakening no-LLM-change verification. Remediation moved the baseline to committed fixture `tests/fixtures/rag_agent_package_prompt_baseline.json` and changed the test to fail when the fixture is missing. Second, Stage 3 remained unchecked despite accepted blockers. Remediation marked Stage 3 complete with the accepted-blocker rationale. Affected verification after remediation: `venv\Scripts\python -m py_compile tests/test_rag_agent_package_prompt_stability.py` passed, `venv\Scripts\python -m pytest tests/test_rag_agent_package_prompt_stability.py -q` passed, and grep confirmed `tests/test_rag_agent_package_prompt_stability.py` no longer contains baseline write or directory-creation calls.
- Residual risks: accepted pre-existing and unrelated verification blockers remain documented above and were accepted by the user on 2026-05-29; no production behavior changes were made to address them in this plan.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Prompt text changes during move | Move prompt constants byte-for-byte and inspect diff | Prompt stability audit and focused patched-LLM tests |
| Old imports silently remain | Bigbang deletion plus static greps | Old-path `rg` check returns no matches |
| Global dispatcher gains implicit discovery | Explicitly forbid global discovery | `_discover`, `iter_modules`, and `pkgutil` grep |
| Test monkeypatches target stale paths | Update tests in the same cutover | Focused tests pass |
| Behavior changes hidden inside structural refactor | No semantic edits and review against plan | Deterministic RAG, projection, and supervisor tests pass |
| Logger module names change | Accept module-path logger movement only when behavior and payload logs remain equivalent | Review diff and operational log assertions |
