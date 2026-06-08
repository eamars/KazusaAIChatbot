# unconditional shared memory prewarm plan

## Summary

- Goal: add an unconditional first-cycle shared-memory evidence prewarm so
  L2a cognition can see relevant rows from the shared `memory` collection
  without restoring the old full RAG-first path.
- Plan class: large
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: bigbang for the new first-cycle prewarm behavior;
  compatible for existing RAG2, resolver observations, `rag_result` top-level
  shape, and downstream cognition consumers.
- Highest-risk areas: accidental restoration of full RAG-first behavior,
  unconditional `user_memory_units` retrieval, stale-memory anchoring,
  blocking L2a too long, leaking prewarm operational labels into cognition,
  and treating prewarm evidence as a resolver-selected capability observation.
- Acceptance criteria: first resolver cycle can receive bounded shared
  `memory` evidence in `rag_result.memory_evidence` through the existing RAG
  request intake and projection path; prewarm never reads `user_memory_units`;
  `rag_result.answer` remains empty for prewarm-only evidence; L2a consumes
  the prewarm result at the first cognition point that reads `rag_result`
  while L1 and L2b retain parallel work; existing resolver-selected RAG2
  still runs only when L2d requests evidence.

## Context

The current resolver removed RAG from the first cognition iteration. That
change is directionally correct because it keeps the live path character-first:
L1 and L2 interpret the current scene before L2d decides whether stronger
evidence is needed. Recent QQ channel observation showed that full first-pass
RAG would not fix the main failures, which were often current-thread,
reply/image, or self/other interpretation problems.

The same observation also showed that the shared `memory` collection contains
useful standing evidence, such as interaction-style guidance and durable
character/world policies. Those rows are safer to provide unconditionally than
scoped `user_memory_units`, because they are not private continuity for one
current group speaker and are less likely to overpersonalize noisy group chat.

Existing implementation facts:

- `MemoryEvidenceAgent` is the top-level memory capability under
  `src/kazusa_ai_chatbot/rag/memory_evidence/agent.py`.
- `MemoryEvidenceAgent` may route to `user_memory_evidence_agent`, which reads
  `user_memory_units`; it must not be used as the unconditional prewarm path.
- `PersistentMemorySearchAgent` under
  `src/kazusa_ai_chatbot/rag/memory_evidence/workers/persistent_search.py`
  searches the shared `memory` collection through `search_persistent_memory`.
- `build_text_chat_rag_request(...)` is the existing persona RAG intake that
  projects persona state into the RAG request context.
- `project_known_facts(...)` already knows how to project
  `persistent_memory_search_agent` rows into public `rag_result.memory_evidence`.
- `ensure_initial_resolver_inputs(...)` currently creates an empty
  `rag_result` for first-cycle cognition.
- `call_cognition_subgraph(...)` runs L1 first, then L2a and L2b in parallel.
  L2a is the first consumer that reads `rag_result`.

The approved direction is limited shared-memory prewarm, not full
MemoryEvidenceAgent retrieval and not full RAG2 before cognition.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  archiving, or signing off this plan.
- `local-llm-architecture`: load before changing RAG, memory retrieval,
  cognition graph, prompt inputs, LLM call budget, or resolver boundaries.
- `py-style`: load before editing Python production files.
- `cjk-safety`: load before editing Python files that contain CJK prompt or
  test strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, and plan edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Preserve the current evidence ownership boundary: RAG retrieves evidence,
  cognition decides stance, dialog renders visible wording, and consolidation
  owns durable writes.
- Do not restore the legacy first-pass full RAG path.
- Do not call `MemoryEvidenceAgent` from the unconditional prewarm path,
  because it can route to `user_memory_units`.
- Reuse the existing RAG request intake and projection path:
  `build_text_chat_rag_request(...)` for context construction and
  `project_known_facts(...)` for `rag_result` projection.
- Do not create a new production RAG intake, custom memory projection layer,
  custom evidence formatter, or standalone prewarm architecture.
- Do not perform additional retrieval reads against `user_memory_units`, user
  profiles, conversation history, recall state, web, live context, adapters,
  scheduler state, or persistence targets from the unconditional prewarm path.
  Already-loaded prompt-safe state passed by `build_text_chat_rag_request(...)`
  is allowed because it is the existing RAG intake contract.
- Do not create resolver observations for prewarm evidence. Resolver
  observations remain cognition-selected capability results.
- Do not populate `rag_result.answer` from prewarm-only evidence.
- Do not expose prewarm task names, worker names, cache keys, retry details,
  source ids, raw rows, embeddings, or traceback text to cognition prompts.
- Do not add a feature flag, compatibility shim, alternate full-RAG fallback,
  retry loop, extra selector prompt, database collection, index, migration, or
  environment variable.
- The prewarm task may use the existing persistent-memory search worker with a
  one-attempt cap. It must not increase the RAG2 supervisor loop cap.
- Invalid shape, no result, unresolved result, or worker failure must degrade
  to the existing empty first-cycle `rag_result`.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Add a shared-memory-only first-cycle prewarm path that reuses the existing
  persona RAG intake, calls only `PersistentMemorySearchAgent`, and projects
  through `project_known_facts(...)`.
- Start the prewarm concurrently with first-cycle cognition work so it can run
  while L1 executes and while L2b can continue independently.
- Join the prewarm result at L2a, the first cognition consumer of
  `rag_result`, without adding extra control paths.
- Merge only confirmed shared-memory evidence into `rag_result.memory_evidence`.
- Return the merged `rag_result` from the L2a wrapper and from
  `call_cognition_subgraph(...)` so downstream L2d, resolver state, and
  consolidation see the same evidence L2a saw.
- Preserve the existing empty `rag_result` shape when prewarm has no safe
  evidence.
- Keep `rag_result.answer` empty for prewarm-only evidence.
- Preserve existing resolver-selected RAG2 behavior when L2d requests
  `rag_evidence` or `web_evidence`.
- Add deterministic tests proving the prewarm path excludes
  `MemoryEvidenceAgent` and `user_memory_units`.
- Add integration-style tests proving L2a sees prewarm memory evidence and
  L2b is not coupled to the prewarm join.
- Update RAG and node documentation to describe the new limited prewarm lane.

## Deferred

- Do not implement full first-pass RAG.
- Do not run full `MemoryEvidenceAgent` unconditionally.
- Do not add conditional scoped `user_memory_units` sidecar retrieval in this
  plan. That requires a later plan with separate evidence.
- Do not change RAG2 initializer, dispatcher, evaluator, finalizer, helper
  worker prompts, Cache2 policy, web search, conversation search, person
  context, recall, dialog, consolidation, reflection, self-cognition,
  scheduler, adapters, or database schema.
- Do not tune model prompts to force L2d to request RAG more often.
- Do not change the top-level `rag_result` public keys.
- Do not add operator configuration for prewarm behavior in this plan.

## Cutover Policy

Overall strategy: bigbang for the prewarm lane; compatible for existing RAG2
and `rag_result` shape.

| Area | Policy | Instruction |
|---|---|---|
| First-cycle cognition input | bigbang | Add limited shared-memory evidence before L2a when available. Do not preserve first-cycle empty memory evidence as the only behavior. |
| Full RAG2 evidence | compatible | Keep L2d-selected RAG2 as the only full RAG path. |
| Resolver observations | compatible | Do not represent prewarm evidence as a resolver observation. |
| Shared `memory` retrieval | compatible | Reuse existing persistent-memory search behavior with a one-attempt cap. |
| Scoped `user_memory_units` | bigbang exclusion | Keep scoped user-memory retrieval out of unconditional prewarm. |
| State shape | compatible | Preserve existing `rag_result` top-level keys and downstream projections. |
| Tests | bigbang | Add focused prewarm tests and update any first-cycle empty-RAG expectations that become stale. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- For bigbang areas, implement the new behavior directly instead of adding a
  compatibility switch.
- For compatible areas, preserve only the compatibility surfaces explicitly
  listed in this plan.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

First resolver cycle flow:

```text
call_cognition_subgraph(first cycle)
  -> if resolver_state.cycle_index == 0:
       start shared-memory prewarm task using existing RAG request intake
  -> L1 subconscious runs
  -> L2a consciousness joins the prewarm result
       -> if confirmed shared memory evidence exists:
            pass merged rag_result with memory_evidence to L2a
       -> if failure, invalid shape, no result, or unresolved:
            pass original empty rag_result
       -> return the rag_result consumed by L2a into subgraph state
  -> L2b boundary appraisal runs independently of the prewarm join
  -> L2c1/L2c2/L2d proceed normally
  -> call_cognition_subgraph returns the final rag_result it placed in
     cognition state
```

Observable `rag_result` after prewarm success:

```python
{
    "answer": "",
    "memory_evidence": [
        {
            "summary": "conclusion-formatted shared memory summary",
            "content": "prompt-safe evidence block"
        }
    ],
    "user_memory_unit_candidates": [],
    "conversation_evidence": [],
    "recall_evidence": [],
    "external_evidence": [],
    "user_image": {"user_memory_context": empty_user_memory_context()},
    "supervisor_trace": {"loop_count": 0, "unknown_slots": [], "dispatched": []}
}
```

Unsuccessful prewarm produces the same empty first-cycle `rag_result` used
today.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Retrieval source | Use the existing persona RAG intake, then call only `PersistentMemorySearchAgent`. | This reuses the current RAG state-to-context path while avoiding full RAG-first and scoped user memory. |
| Full MemoryEvidenceAgent | Do not use it unconditionally. | It can route to `user_memory_evidence_agent` and read `user_memory_units`. |
| Resolver observation | Do not create one for prewarm. | Observations represent cognition-selected capabilities and affect duplicate-tracking semantics. |
| First consumer join | Join inside L2a before `call_cognition_consciousness`. | L2a is the first prompt that reads `rag_result`; L1 and L2b can still run in parallel with the prewarm. |
| Public answer | Keep `rag_result.answer` empty for prewarm-only evidence. | Prewarm is background evidence, not a synthesized RAG answer. |
| Trace exposure | Keep prewarm operational detail out of cognition-visible trace. | L2 should reason over evidence, not over implementation labels. |
| State propagation | Return merged `rag_result` from the L2a wrapper and from `call_cognition_subgraph(...)`. | L2a, L2d, resolver state, and consolidation must share one evidence view. |

## Contracts And Data Shapes

Add one narrow internal helper in the existing RAG/resolver capability surface.
Do not create a new production module for prewarm. The helper exists only to
reuse current persona RAG intake, run the existing shared-memory worker, and
project through current RAG projection.

```python
async def run_first_cycle_shared_memory_prewarm(
    state: GlobalPersonaState,
) -> dict[str, Any]:
    """Return a rag_result containing shared memory evidence only."""
```

Contract:

- Input is the persona state after decontextualization and resolver input
  initialization.
- The helper builds the RAG request with `build_text_chat_rag_request(...)`,
  using the same state fields as the existing RAG path.
- The helper uses `rag_request["original_query"]` as the semantic query source.
- The helper imports and calls only
  `PersistentMemorySearchAgent().run(..., max_attempts=1)` from
  `kazusa_ai_chatbot.rag.memory_evidence.workers.persistent_search`.
- The helper accepts worker output only when `worker_result["resolved"] is
  True` and `worker_result["result"]` contains at least one dictionary row
  without `source_system="user_memory_units"`.
- The helper converts the worker result into a normal `known_facts` row with
  `agent="persistent_memory_search_agent"`, `resolved=True`,
  `raw_result=<accepted rows>`, and `summary` set to the first safe row
  `content`, `description`, `text`, `summary`, or `fact` field. It then
  projects that row through `project_known_facts(...)`.
- The helper returns a full `rag_result` shape. The caller owns deciding
  whether to merge it into the L2a state.
- The helper returns no `answer`, no `user_memory_unit_candidates`, no
  conversation evidence, no external evidence, no recall evidence, and no
  profile/image mutation.
- The helper keeps source rows, cache details, raw worker payloads, errors,
  and timings out of prompt-facing fields.

Merge contract:

```python
def merge_shared_memory_prewarm_result(
    base_rag_result: dict[str, Any],
    prewarm_rag_result: dict[str, Any],
) -> dict[str, Any]:
    """Return base rag_result plus prompt-safe shared memory evidence."""
```

Merge rules:

- Preserve every base field unless adding `memory_evidence`.
- Append only `memory_evidence` items that are dictionaries and do not carry
  `source_system="user_memory_units"`.
- Preserve `answer` from the base result.
- Preserve `user_memory_unit_candidates` from the base result.
- Preserve `supervisor_trace` from the base result.
- If there is no valid memory evidence, return the base result unchanged.
- The L2a wrapper must include this merged or unchanged `rag_result` in its
  returned `CognitionState` update.
- `call_cognition_subgraph(...)` must include the final `result["rag_result"]`
  in its returned persona-state update.

## LLM Call And Context Budget

Default context cap: 50k tokens.

Before this plan:

| Path | Response-path LLM calls | Blocking behavior |
|---|---:|---|
| First resolver cycle without L2d RAG | L1, L2a, L2b, L2c1, L2c2, L2d | Sequential/parallel according to current cognition graph; no RAG subagent before L2a. |
| L2d-selected RAG | Existing RAG2 initializer/dispatcher/helper/evaluator/finalizer loops | Runs only after cognition selects a capability. |

After this plan:

| Path | Response-path LLM calls | Blocking behavior |
|---|---:|---|
| First resolver cycle shared-memory prewarm | Adds at most one persistent-memory generator call and one persistent-memory judge call, using `max_attempts=1`. Cache hit may add zero LLM calls. | Starts before L1; L2a joins the result at the first `rag_result` consumer; L2b is not coupled to that join. |
| L2d-selected RAG | Unchanged. | Existing resolver capability behavior is unchanged. |

Context inputs for prewarm:

- `rag_request["original_query"]` from `build_text_chat_rag_request(...)`;
- minimal runtime context needed by the existing persistent-memory worker;
- already-loaded prompt-safe context from `build_text_chat_rag_request(...)`;
- no scoped `user_memory_units` payload.

The prewarm result should be smaller than normal RAG2 evidence because only
`memory_evidence` is merged and existing projection caps apply.

## Change Surface

### Create

- `tests/test_shared_memory_prewarm.py`: focused tests for helper behavior,
  source exclusion, existing-intake use, projection, and merge rules.

### Modify

- `src/kazusa_ai_chatbot/cognition_resolver/capabilities.py`: add the narrow
  internal shared-memory prewarm helper beside the existing persona RAG helper
  so it can reuse `build_text_chat_rag_request(...)` and
  `project_known_facts(...)`.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`: start the
  first-cycle prewarm task inside `call_cognition_subgraph(...)` and join it
  inside the L2a wrapper before `call_cognition_consciousness(...)`.
- `tests/test_persona_supervisor2_cognition_prewarm.py`: prove first-cycle
  L2a receives prewarm memory evidence, L2b is not coupled to the prewarm
  join, downstream state receives the same merged `rag_result`, and
  existing resolver-only graph behavior remains intact.
- `src/kazusa_ai_chatbot/rag/README.md`: document the limited prewarm lane and
  distinguish it from L2d-selected full RAG2.
- `src/kazusa_ai_chatbot/nodes/README.md`: document that L2a may receive
  bounded shared-memory prewarm evidence in the first cycle.

### Keep

- `src/kazusa_ai_chatbot/rag/memory_evidence/agent.py`: do not use the
  top-level `MemoryEvidenceAgent` for unconditional prewarm.
- `src/kazusa_ai_chatbot/rag/memory_evidence/workers/user_memory.py`: no
  change; it remains scoped and cognition-selected only.
- `src/kazusa_ai_chatbot/cognition_resolver/loop.py`: no change unless
  this plan is updated and the user explicitly approves a resolver-loop change.
- `src/kazusa_ai_chatbot/cognition_resolver/state.py`: keep the initial empty
  RAG shape builder; prewarm merge happens later and does not alter resolver
  state initialization.
- RAG2 initializer, dispatcher, evaluator, finalizer, Cache2, database schema,
  adapters, dialog, consolidation, reflection, scheduler, and self-cognition.

## Overdesign Guardrail

- Actual problem: first-cycle cognition can miss useful standing shared-memory
  evidence because RAG is now cognition-selected and local models may decide
  not to retrieve it.
- Minimal change: reuse the current RAG request intake and projection to run
  one shared `memory` worker and supply bounded `memory_evidence` to L2a in
  the first cycle.
- Ownership boundaries: persistent-memory worker retrieves evidence;
  deterministic merge validates shape and scope; L2 cognition decides stance;
  L2d remains the owner of full evidence capability requests.
- Rejected complexity: new RAG intake, standalone prewarm module, custom memory
  projection, full RAG-first, full `MemoryEvidenceAgent`, scoped user-memory
  sidecar, conversation/history/web/profile/recall prewarm, resolver
  observation injection, prompt-only tuning, feature flags, new database state,
  retries beyond one attempt, compatibility shims, and new environment
  variables.
- Evidence threshold: add scoped user-memory or broader prewarm only after
  real chat audits show repeated failures where shared memory was insufficient
  and current-user continuity would have directly corrected first-cycle
  cognition without increasing stale-context anchoring.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  cutover strategies, compatibility layers, fallback paths, feature flags, or
  extra retrieval sources.
- The responsible agent must treat changes outside the listed change surface
  as high-scrutiny changes. Updating an existing module outside the target
  boundary or introducing a new prompt, variable, or code path requires plan
  update and user approval before implementation.
- The responsible agent must search for existing projection, sanitization, and
  merge helpers before writing equivalents. Reuse existing helpers when they
  preserve the contract.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, the responsible agent must preserve the
  plan's stated intent and report the discrepancy.
- If a required instruction is impossible, the responsible agent must stop and
  report the blocker instead of inventing a substitute.

## Implementation Order

1. Parent adds focused helper tests in `tests/test_shared_memory_prewarm.py`:
   - `test_first_cycle_prewarm_uses_existing_rag_intake_and_persistent_worker`
   - `test_first_cycle_prewarm_projects_memory_without_answer_or_user_units`
   - `test_merge_shared_memory_prewarm_result_filters_user_memory_source`
   - `test_first_cycle_prewarm_returns_empty_on_unresolved_or_worker_failure`
   Expected pre-implementation result: tests fail because
   `run_first_cycle_shared_memory_prewarm(...)` and
   `merge_shared_memory_prewarm_result(...)` do not exist in
   `kazusa_ai_chatbot.cognition_resolver.capabilities`.
2. Parent adds cognition-subgraph integration tests proving:
   - `test_first_cycle_prewarm_evidence_reaches_l2a`;
   - `test_call_cognition_subgraph_returns_merged_rag_result`;
   - `test_prewarm_starts_only_on_resolver_cycle_zero`;
   - `test_l2b_runs_independently_of_prewarm_join`;
   - `test_l2a_uses_base_rag_result_when_prewarm_unresolved`.
3. Parent starts one production-code subagent with this approved plan, the
   focused failing tests, and the exact production change surface.
4. Production-code subagent adds the narrow shared-memory prewarm helper to
   the existing RAG/resolver capability surface, reusing current intake and
   projection.
5. Production-code subagent wires the first-cycle prewarm task and L2a join in
   `persona_supervisor2_cognition.py`.
6. Parent runs focused helper and cognition-subgraph tests, then loops back to
   the focused contract if failures show a contract gap.
7. Parent updates RAG and node documentation after behavior tests pass.
8. Parent runs focused regression tests for resolver contracts, resolver loop,
   persona graph, RAG projection, memory evidence capability agents, and
   hybrid memory helpers.
9. Parent runs static greps proving the prewarm path does not call
   `MemoryEvidenceAgent` or `user_memory_evidence_agent`.
10. Parent runs the independent code review gate, remediates approved findings,
    reruns affected verification, records evidence, and updates plan status
    only after approval.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only;
  does not edit tests unless the parent explicitly directs it; closes after
  planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static
  checks, and validation work while the production-code subagent edits
  production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [ ] Stage 1 - focused prewarm contract tests established
  - Covers: implementation steps 1-2.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_shared_memory_prewarm.py -q`
    and `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_cognition_prewarm.py -q`.
  - Evidence: record expected pre-implementation failures in `Execution Evidence`.
  - Handoff: next agent starts production implementation at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - shared-memory prewarm helper implemented
  - Covers: implementation step 4.
  - Verify: `venv\Scripts\python.exe -m pytest tests\test_shared_memory_prewarm.py -q`.
  - Evidence: record changed files and focused test output.
  - Handoff: next agent wires cognition integration at Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - L2a first-consumer join integrated
  - Covers: implementation step 5.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_cognition_prewarm.py -q`
    plus
    `venv\Scripts\python.exe -m pytest tests\test_cognition_resolver_persona_graph.py -q`.
  - Evidence: record test output and unresolved-result behavior.
  - Handoff: next agent updates docs and runs regressions.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - documentation and regression verification complete
  - Covers: implementation steps 7-9.
  - Verify: all commands under `Verification`.
  - Evidence: record command outputs and static grep results.
  - Handoff: next agent starts independent review.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - independent code review complete
  - Covers: implementation step 10.
  - Verify: review subagent approval plus rerun of affected tests after fixes.
  - Evidence: record findings, fixes, rerun commands, residual risks, and
    approval status in `Execution Evidence`.
  - Handoff: parent may mark the plan completed only after this checkpoint.
  - Sign-off: `<agent/date>` after review approval and evidence are recorded.

## Verification

### Static Greps

- `rg -n "MemoryEvidenceAgent|UserMemoryEvidenceAgent|user_memory_evidence_agent|query_user_memory_units|search_user_memory_units" src\\kazusa_ai_chatbot\\cognition_resolver\\capabilities.py src\\kazusa_ai_chatbot\\nodes\\persona_supervisor2_cognition.py`
  returns no matches. Exit code 1 is acceptable only when there are no matches.
- `rg -n "rag_result\\[\"answer\"\\].*=|\"answer\": .*prewarm|prewarm.*answer" src\\kazusa_ai_chatbot\\cognition_resolver\\capabilities.py src\\kazusa_ai_chatbot\\nodes\\persona_supervisor2_cognition.py`
  returns no matches for production writes that populate prewarm answer text.

### Focused Tests

- `venv\Scripts\python.exe -m pytest tests\test_shared_memory_prewarm.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_cognition_prewarm.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_cognition_resolver_persona_graph.py -q`

### Regression Tests

- `venv\Scripts\python.exe -m pytest tests\test_cognition_resolver_contracts.py tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_persona_graph.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_rag_projection.py tests\test_rag_phase3_capability_agents.py tests\test_rag_hybrid_agents.py -q`
- `venv\Scripts\python.exe -m pytest tests\test_user_memory_evidence_agent.py tests\test_user_memory_units_rag_flow.py -q`

### Static Compile

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\cognition_resolver\capabilities.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py`

## Independent Plan Review

Run this gate before approval or execution. Prefer a reviewer that did not
draft the plan. If no separate reviewer is available, the drafting agent must
reread the development-plan contract, this plan, and relevant source/test
context from a fresh-review posture.

Review scope:

- The plan preserves the RAG/cognition/dialog/persistence ownership boundary.
- The plan does not restore full RAG-first behavior.
- The plan excludes `user_memory_units` from unconditional prewarm.
- The first-consumer join and parallelism story are specific enough for
  implementation.
- The plan gives exact contracts, file paths, tests, greps, and acceptance
  criteria.
- Agent creativity is tightly bounded with no hidden alternate retrieval path.

Record blockers, non-blocking findings, required edits, and approval status.
Approve only when blockers are resolved.

Plan review record, 2026-06-08:

- Review mode: drafting-agent fresh-review pass after rereading the
  development-plan contract, execution gates, cutover policy, RAG docs,
  cognition node docs, and relevant source/test files.
- Blockers found and resolved: removed plugin-framework mandatory skills,
  replaced vague cognition-subgraph test choices with exact test file and
  function names, removed the resolver-loop escape hatch, added the merged
  `rag_result` propagation contract, clarified existing RAG intake versus
  additional retrieval reads, removed the extra control-path design, and fixed
  stale static-grep expectations.
- Non-blocking findings: the plan remains `draft` and is not executable until
  the user approves it; production execution still requires the native
  subagent execution model described in `Execution Model`.
- Approval status: no unresolved blockers remain for draft-plan discussion;
  not approved for implementation while status is `draft`.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

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
  execution evidence, next-stage handoff notes, and path-safe commands.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows review-only
fixture/documentation corrections. If a fix would cross the approved boundary
or alter the contract, stop and update the plan or request approval before
changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Execution Evidence

No implementation has started. During approved execution, record evidence here
immediately after each completed checklist stage:

- Stage 1 focused test baseline:
- Stage 2 helper implementation verification:
- Stage 3 cognition integration verification:
- Stage 4 documentation, regression, static grep, and compile verification:
- Stage 5 independent code review findings and approval:
- Residual risks or blocked commands:

## Acceptance Criteria

This plan is complete when:

- First-cycle L2a can consume bounded shared `memory` evidence when the
  prewarm finds confirmed rows.
- The unconditional prewarm path cannot call `MemoryEvidenceAgent`,
  `UserMemoryEvidenceAgent`, or `user_memory_units` retrieval helpers.
- Prewarm-only evidence never populates `rag_result.answer` or
  `user_memory_unit_candidates`.
- L2d-selected full RAG2 still runs only through existing resolver capability
  requests.
- Invalid result, missing result, worker failure, and unresolved result all
  degrade to the existing empty first-cycle `rag_result`.
- Focused tests, regression tests, static greps, and `py_compile` listed in
  `Verification` pass.
- RAG and cognition node docs describe the limited shared-memory prewarm lane.
- Independent code review has approved the implementation or all blocking
  findings have been fixed and reverified.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Stale shared memory anchors first-cycle interpretation too strongly | Keep prewarm to `memory_evidence` only, leave `answer` empty, and let cognition decide stance. | L2a prompt input test and manual review of produced `rag_result`. |
| Scoped user continuity leaks into group-chat first cycle | Directly use shared-memory worker and static-grep for user-memory helpers. | Static greps plus helper tests that fail if user-memory workers are called. |
| L2a latency increases | Keep the retrieval source limited to one shared-memory worker with `max_attempts=1`, and start it before L1 so normal L1 work overlaps the prewarm. | Cognition-subgraph integration test verifies L2b is not coupled to the prewarm join. |
| Resolver trace semantics become misleading | Do not create resolver observations and preserve base `supervisor_trace`. | Resolver loop/persona graph tests. |
| Existing first-cycle empty-RAG assumptions break | Update only tests whose asserted behavior is intentionally changed. | Focused resolver/persona regression tests. |
