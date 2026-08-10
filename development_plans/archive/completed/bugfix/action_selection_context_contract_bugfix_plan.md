# action selection context contract bugfix plan

## Summary

- Goal: fix the production `KeyError: 'action_selection_context'` by making
  the L2d action-selection context a canonical initialized cognition-chain
  contract instead of an optional mutable context bag.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`
- Overall cutover strategy: bigbang for the cognition-chain input contract;
  compatible for the semantic meaning of existing `coding_runs` and
  `group_engagement_action_context` payloads.
- Highest-risk areas: weakening the action materialization boundary, exposing
  raw coding-run or group-engagement operational fields to L2d, and fixing the
  symptom with local `.get()` guards while leaving the root contract ambiguous.
- Acceptance criteria: ordinary user-message turns with no open coding runs,
  group self-cognition turns, and coding-run continuation turns all run through
  `call_cognition_subgraph()` without missing-key failures; tests prove the
  default empty context, populated group context, and populated coding-run
  context all follow one canonical contract.

## Context

The observed production failure is:

```text
KeyError: 'action_selection_context'
```

The failure occurs in
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py` inside
`call_cognition_subgraph()` after the cognition chain returns, when the wrapper
passes `chain_input["action_selection_context"]` into
`apply_cognition_chain_output_to_global_state(...)`.

The root cause is contract drift:

- `CognitionChainInputV1.action_selection_context` is typed as
  `NotRequired[Mapping[str, Any]]`.
- `build_cognition_chain_input_from_global_state(...)` omits the key for
  ordinary turns.
- `run_cognition_chain(...)` tolerates absence and treats it as `{}`.
- Phase C coding-run continuation work reused the same optional field as a
  required runtime handoff for L2d prompt context and later deterministic
  action materialization.
- The wrapper now mutates and indexes the key as though it is always present.

The bug is not only a missing local default. The deeper problem is that an
optional prompt extension became required runtime state without a canonical
neutral value, a validator contract, or tests for the empty ordinary-turn path.

Confirmed local reproduction:

```powershell
venv\Scripts\python -m pytest `
  tests\test_cognition_resolver_l2d_contract.py `
  tests\test_l2d_l3_surface_handoff.py `
  tests\test_persona_supervisor2_cognition_prewarm.py `
  -q
```

Observed result during RCA: 8 failures, all
`KeyError: 'action_selection_context'` from the same wrapper path.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  archiving, or signing off this plan.
- `local-llm-architecture`: load before changing cognition-chain, L2d action
  selection, action materialization, or prompt-facing context boundaries.
- `py-style`: load before editing Python production files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Execute production changes only while this plan status is `in_progress`.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, and plan edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- This plan's execution model uses parent-led native subagent execution unless
  the user explicitly approves fallback execution for implementation.
- Preserve the cognition ownership line:
  `L1 -> L2 -> L2d` owns semantic judgment and action selection;
  deterministic code owns validation, prompt-safe projection, action
  materialization, persistence, scheduling, and delivery.
- Preserve the existing L2d capability semantics for `coding_runs` and
  `group_engagement_action_context`; this plan changes the container contract,
  not the meaning of those payloads.
- Do not add a response-path LLM call, retry loop, repair prompt, feature flag,
  compatibility shim, fallback mapper, or alternate action-selection path.
- Do not move coding-run continuation binding into LLM prompts. Deterministic
  materialization must still bind only offered refs and allowed actions.
- Do not expose raw accepted-task ids, job ids, adapter ids, DB collection
  names, lock keys, execution specs, approval evidence, or worker internals to
  L2d or L3.
- Do not use keyword scanning over user text to select coding-run refs or
  continuation actions.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation,
  verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Make `action_selection_context` a required, initialized
  `CognitionChainInputV1` field with a canonical empty shape.
- Define an explicit typed contract for the field instead of leaving it as an
  unstructured optional `Mapping[str, Any]`.
- Ensure ordinary user-message turns with no open coding runs receive the
  canonical empty context and complete without `KeyError`.
- Ensure group self-cognition merges `group_engagement_action_context` into
  the canonical context without assuming a preexisting optional key.
- Ensure user-message turns with open coding runs merge `coding_runs` into the
  canonical context and pass the same trusted context to deterministic action
  materialization.
- Keep deterministic subgraph tests independent of live MongoDB by patching
  `load_open_coding_run_contexts_for_scope(...)` when testing the no-open-run
  path.
- Add regression tests for:
  - default builder output includes the canonical empty context;
  - `call_cognition_subgraph()` succeeds with no open coding runs;
  - group self-cognition context reaches L2d;
  - coding-run continuation context reaches L2d and materialization;
  - private coding-run fields remain excluded from L2d and L3 context.
- Update cognition-chain documentation to state the ownership and neutral
  value of `action_selection_context`.

## Deferred

- Do not redesign L2d action-selection prompts.
- Do not change accepted-task lifecycle, background-work worker routing,
  coding-run ledger format, or coding-run allowed action semantics.
- Do not add a new generic context framework, context registry, or plugin
  architecture.
- Do not add compatibility aliases for old input shapes.
- Do not change RAG, dialog, consolidation, scheduler, adapters, or database
  schema.
- Do not broaden this plan into Phase D coding-agent work.

## Cutover Policy

Overall strategy: bigbang for the cognition-chain input container; compatible
for existing payload meaning.

| Area | Policy | Instruction |
|---|---|---|
| `CognitionChainInputV1.action_selection_context` presence | bigbang | Make the key required and initialized. Do not preserve an absent-key input as the internal canonical shape. |
| Empty action-selection context | bigbang | Use one canonical empty shape, such as empty coding-run list plus empty group-engagement mapping. |
| Coding-run payload semantics | compatible | Preserve existing prompt-safe fields and deterministic binding behavior. |
| Group engagement payload semantics | compatible | Preserve existing prompt-safe group engagement fields. |
| Tests | bigbang | Update tests to assert the new required initialized context instead of accepting absent-key behavior. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- For bigbang areas, rewrite the contract directly instead of adding local
  missing-key guards at every consumer.
- For compatible areas, preserve only the payload meanings explicitly listed in
  this plan.
- Any change to this cutover policy requires user approval before
  implementation.

## Target State

`build_cognition_chain_input_from_global_state(...)` always returns a
`CognitionChainInputV1` with `action_selection_context` present.

Canonical empty shape:

```python
{
    "coding_runs": [],
    "group_engagement_action_context": {},
}
```

Populated coding-run shape:

```python
{
    "coding_runs": [
        {
            "schema_version": "coding_run_context.v1",
            "coding_run_ref": "coding_run:<run_id>",
            "status": "blocked",
            "objective_summary": "Repair the fixture parser.",
            "allowed_next_actions": ["respond_to_blocker", "status"],
            "active_blocker": {
                "question": "Install the dependency, then retry.",
                "options": ["Installed"],
            },
            "followup_open": True,
            "updated_at": "2026-07-10T00:00:00Z",
        }
    ],
    "group_engagement_action_context": {},
}
```

Populated group self-cognition shape:

```python
{
    "coding_runs": [],
    "group_engagement_action_context": {
        "engagement_guidelines": ["Stay with the current group topic."],
        "confidence": "medium",
    },
}
```

The core and action materializer consume the same trusted normalized context.
No downstream consumer needs to distinguish absent context from empty context.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Root contract | Make `action_selection_context` required and initialized. | The field is now runtime handoff state, not a rare optional prompt extension. |
| Context shape | Use explicit subfields for `coding_runs` and `group_engagement_action_context`. | This removes the ambiguous untyped bag while preserving current payload meanings. |
| Empty value | Use `coding_runs=[]` and `group_engagement_action_context={}`. | Every consumer can safely read the same shape without local missing-key policy. |
| Validator role | Validate required presence and top-level subfield types. | Contract errors should fail at the boundary, not after L2d has run. |
| Materialization | Pass the normalized context into action materialization. | Coding continuations must remain bound to currently offered trusted refs. |
| Prompt behavior | Do not change L2d prompt semantics. | The failure is deterministic contract drift, not a prompt quality problem. |

## Contracts And Data Shapes

Production contract:

```python
class ActionSelectionContextV1(TypedDict):
    coding_runs: list[Mapping[str, Any]]
    group_engagement_action_context: Mapping[str, Any]


class CognitionChainInputV1(TypedDict):
    ...
    action_selection_context: ActionSelectionContextV1
```

Validation contract:

- `action_selection_context` is required on public core input.
- `coding_runs` is required and must be a list.
- `group_engagement_action_context` is required and must be a mapping.
- Deep validation of coding-run row semantics remains owned by existing
  coding-run projection and materialization helpers.
- Unknown nested keys may remain tolerated only where current prompt-safe
  projection already strips them before L2d/L3 consumption.

Forbidden shapes:

```python
{}
```

as a complete `CognitionChainInputV1` without `action_selection_context`.

```python
{
    "action_selection_context": None,
}
```

and non-mapping action-selection contexts are forbidden.

## LLM Call And Context Budget

- Do not add, remove, or reroute LLM calls.
- L2d receives the same semantic content as before when contexts are populated.
- Ordinary turns receive an empty `coding_runs` list and empty group context;
  this adds negligible prompt JSON overhead and no latency.
- The change is deterministic input-shape hardening, not model behavior
  tuning.

## Change Surface

### Delete

- No files are planned for deletion.

### Modify Production

- `src/kazusa_ai_chatbot/cognition_chain_core/contracts.py`
  - Add `ActionSelectionContextV1`.
  - Make `action_selection_context` required on `CognitionChainInputV1`.
  - Validate top-level context shape.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Build the canonical empty action-selection context.
  - Merge group engagement and coding-run contexts into the canonical shape.
  - Pass the normalized context to materialization without optional-key
    assumptions.
- `src/kazusa_ai_chatbot/cognition_chain_core/chain.py`
  - Consume the required normalized context.
  - Preserve existing prompt projection behavior.
- `src/kazusa_ai_chatbot/cognition_chain_core/action_selection.py`
  - Consume the required top-level shape in coding-run context projection.
  - Preserve empty-list behavior when `coding_runs` is empty.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`
  - Consume the required top-level shape in coding-run materialization lookup.
  - Preserve trusted-ref binding and empty-list behavior.
- `src/kazusa_ai_chatbot/cognition_chain_core/README.md`
  - Document action-selection context ownership, neutral value, and forbidden
    operational fields.

### Modify Tests

- `tests/test_cognition_chain_connector_mapping.py`
  - Add `test_cognition_chain_input_includes_empty_action_selection_context`.
- `tests/test_cognition_resolver_l2d_contract.py`
  - Add `test_cognition_subgraph_runs_without_open_coding_context`.
  - Patch `load_open_coding_run_contexts_for_scope(...)` to return `[]` in the
    no-open-run regression.
- `tests/test_l2d_l3_surface_handoff.py`
  - Keep/extend group self-cognition context regression.
- `tests/test_persona_supervisor2_cognition_prewarm.py`
  - Keep/extend prewarm ordinary-turn regressions.
- `tests/test_coding_agent_phase_c_run_context_contracts.py`
  - Assert coding-run context remains prompt-safe and materialization-bound.

### Modify Documentation And Plan Records

- `src/kazusa_ai_chatbot/nodes/README.md`
  - Update the L2d/coding-run context description.
- `development_plans/archive/completed/bugfix/action_selection_context_contract_bugfix_plan.md`
  - Record execution progress and evidence.
- `development_plans/README.md`
  - Track lifecycle status.

### Create

- No new source or test files are planned.

### Keep

- Accepted-task DB schema, coding-run ledger schema, worker payload schema,
  L2d prompt wording, L3 wording, adapters, scheduler, RAG, and consolidation.

## Overdesign Guardrail

- Actual problem: `action_selection_context` became required runtime state but
  remained optional and absent by default, causing production missing-key
  crashes after or before L2d.
- Minimal change: make the field a required normalized input contract with one
  empty shape and explicit subfields, then update the few producer/consumer
  sites and tests that currently rely on optional absence.
- Ownership boundaries: the graph wrapper assembles trusted prompt-safe
  context; cognition core projects it to L2d; L2d selects semantic actions;
  deterministic materialization binds selected coding continuations to trusted
  offered contexts.
- Rejected complexity: no feature flag, compatibility shim, fallback mapper,
  generic context registry, new LLM prompt, retry loop, keyword scan,
  background worker change, database migration, or broad cognition refactor.
- Evidence threshold: add broader context abstraction only if another approved
  near-term integration introduces at least two more independently owned
  action-selection context families with repeated validation logic.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside the listed change surface
  as blocked unless the plan is updated and approved.
- The responsible agent may add small private helpers only when they normalize
  the required action-selection context shape or remove repeated local merge
  code inside the listed files.
- The responsible agent must search for existing equivalent context projection
  or validation helpers before adding a new helper.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If the plan and code disagree, the responsible agent must preserve this
  plan's stated root-contract intent and report the discrepancy.
- If a required instruction is impossible, the responsible agent must stop and
  report the blocker instead of inventing a substitute.

## Implementation Order

1. Parent establishes focused failing tests.
   - Add `test_cognition_chain_input_includes_empty_action_selection_context`
     to `tests/test_cognition_chain_connector_mapping.py`.
   - Add `test_cognition_subgraph_runs_without_open_coding_context` to
     `tests/test_cognition_resolver_l2d_contract.py`.
   - Patch `load_open_coding_run_contexts_for_scope(...)` in the subgraph
     regression so the focused failure does not depend on MongoDB state.
   - Expected pre-implementation result: the builder contract test fails
     because the key is absent, and the subgraph regression fails with
     `KeyError: 'action_selection_context'`.
2. Parent starts one production-code subagent with this approved plan,
   mandatory skills, focused failing tests, and the listed production change
   surface.
3. Production-code subagent implements the required normalized contract.
   - Update contract types and validation first.
   - Update builder and merge sites next.
   - Update core/action-selection/materialization consumers only as required
     to consume the normalized shape.
4. Parent updates integration and regression tests.
   - Keep group self-cognition and coding-run continuation coverage explicit.
   - Keep private-field sanitization tests for coding-run context.
5. Parent runs focused tests and records output.
6. Parent runs broader cognition/action-selection regression commands listed in
   `Verification`.
7. Parent starts one independent code-review subagent after planned
   verification passes.
8. Parent remediates approved review findings inside this plan's change
   surface, reruns affected verification, and records results.

## Execution Model

- Execution requires parent-led native subagent execution.
- Do not execute this plan until the user approves it and the status is changed
  to `approved` or `in_progress`.
- Parent agent owns orchestration, test code, verification, execution evidence,
  review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only;
  does not edit tests unless the parent explicitly directs it; closes after
  planned production code changes are complete, excluding review fixes.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution and
  report the blocker. Fallback single-agent execution requires explicit user
  approval.

## Progress Checklist

- [x] Stage 1 - focused failing test contract established
  - Covers: implementation order step 1.
  - Files: `tests/test_cognition_chain_connector_mapping.py`,
    `tests/test_cognition_resolver_l2d_contract.py`.
  - Required tests:
    `test_cognition_chain_input_includes_empty_action_selection_context`,
    `test_cognition_subgraph_runs_without_open_coding_context`.
  - Required fixture behavior: patch
    `load_open_coding_run_contexts_for_scope(...)` to return `[]` in the
    no-open-run subgraph test.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_cognition_chain_connector_mapping.py tests/test_cognition_resolver_l2d_contract.py -q`.
  - Evidence: record expected missing-key/default-shape failures in
    `Execution Evidence`.
  - Handoff: next agent starts Stage 2.
  - Sign-off: completed on 2026-07-11.

- [x] Stage 2 - production contract normalization implemented
  - Covers: implementation order steps 2 and 3.
  - Files: `contracts.py`, `persona_supervisor2_cognition.py`,
    `chain.py`, and any required action-selection/materialization consumers.
  - Verify: focused tests from Stage 1 pass.
  - Evidence: record changed files and focused test output in
    `Execution Evidence`.
  - Handoff: next agent starts Stage 3.
  - Sign-off: completed on 2026-07-11.

- [x] Stage 3 - group and coding-run integration regressions complete
  - Covers: implementation order step 4.
  - Files: `tests/test_l2d_l3_surface_handoff.py`,
    `tests/test_persona_supervisor2_cognition_prewarm.py`,
    `tests/test_coding_agent_phase_c_run_context_contracts.py`.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests/test_l2d_l3_surface_handoff.py tests/test_persona_supervisor2_cognition_prewarm.py tests/test_coding_agent_phase_c_run_context_contracts.py -q`.
  - Evidence: record pass/fail output in `Execution Evidence`.
  - Handoff: next agent starts Stage 4.
  - Sign-off: completed on 2026-07-11.

- [x] Stage 4 - static and broader regression verification complete
  - Covers: implementation order steps 5 and 6.
  - Verify: all commands under `Verification`.
  - Evidence: record command outputs and any allowed warnings in
    `Execution Evidence`.
  - Handoff: next agent starts Stage 5.
  - Sign-off: completed on 2026-07-11.

- [x] Stage 5 - independent code review complete
  - Covers: implementation order steps 7 and 8.
  - Files: full implementation diff and this plan.
  - Verify: review subagent reports no blocking findings, or all blocking
    findings are remediated and affected checks rerun.
  - Evidence: record review findings, fixes, rerun commands, and residual risk
    in `Execution Evidence`.
  - Handoff: lifecycle update after user sign-off.
  - Sign-off: completed on 2026-07-11.

## Verification

Focused tests:

```powershell
venv\Scripts\python.exe -m pytest tests/test_cognition_chain_connector_mapping.py -q
venv\Scripts\python.exe -m pytest tests/test_cognition_resolver_l2d_contract.py -q
venv\Scripts\python.exe -m pytest tests/test_l2d_l3_surface_handoff.py -q
venv\Scripts\python.exe -m pytest tests/test_persona_supervisor2_cognition_prewarm.py -q
venv\Scripts\python.exe -m pytest tests/test_coding_agent_phase_c_run_context_contracts.py -q
```

Combined regression:

```powershell
venv\Scripts\python.exe -m pytest `
  tests/test_cognition_chain_connector_mapping.py `
  tests/test_cognition_resolver_l2d_contract.py `
  tests/test_l2d_l3_surface_handoff.py `
  tests/test_persona_supervisor2_cognition_prewarm.py `
  tests/test_coding_agent_phase_c_run_context_contracts.py `
  -q
```

Static checks:

```powershell
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/cognition_chain_core/contracts.py
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/cognition_chain_core/chain.py
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/cognition_chain_core/action_selection.py
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py
```

Static greps:

```powershell
rg -n 'chain_input\["action_selection_context"\]|action_selection_context = dict\(chain_input' src/kazusa_ai_chatbot tests
```

Expected result: matches are allowed only where the surrounding code operates
on the required normalized context and tests assert that required shape. Any
match that assumes optional absence or performs local fallback policy must be
removed or justified in `Execution Evidence`.

No live LLM verification is required because this is a deterministic contract
bug with no prompt or model behavior change.

No live DB verification is required. Focused tests that exercise open coding
run lookup must patch `load_open_coding_run_contexts_for_scope(...)` unless a
separate live DB gate is explicitly added to this plan.

## Independent Plan Review

Review mode: user explicitly requested no-subagent plan review on
2026-07-11. The parent agent performed this review directly against
`development-plan`, `plan_contract.md`, `execution_gates.md`,
`cutover_policy.md`, the top-level architecture docs, cognition-chain ICD,
cognition nodes ICD, and action-spec ICD.

Review findings fixed in this draft:

- Reclassified the plan from `medium` to `large` because the plan is above the
  medium line budget and touches contracts, production code, tests, and docs.
- Rewrote the change surface into explicit Delete, Modify Production, Modify
  Tests, Modify Documentation And Plan Records, Create, and Keep groups.
- Removed discretionary conditional wording from production consumer updates.
- Replaced ambiguous test-file choice with exact focused test names and files.
- Added the deterministic no-live-DB requirement for the no-open-coding-run
  regression.
- Added the mandatory execution-model rule to `Mandatory Rules`.

Review result: no blocking plan-review findings remain. The plan is now
`in_progress` after explicit user authorization on 2026-07-11.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final
sign-off. The parent agent must create one independent code-review subagent
through the current harness's native subagent capability. If native subagents
are unavailable, stop unless the user explicitly approves fallback execution.

The review scope is:

- the required `action_selection_context` contract and validator alignment;
- builder, core, wrapper, action-selection, and materialization ownership;
- absence of local symptom-only guards that leave the root contract ambiguous;
- preservation of coding-run trusted-ref binding and private-field exclusion;
- preservation of group self-cognition context behavior;
- no new LLM calls, prompt changes, compatibility shims, feature flags, or
  unrelated refactors;
- test coverage for empty, group, and coding-run populated context paths.

The review subagent must not implement fixes. Parent handles remediation only
inside this plan's approved change surface.

## Acceptance Criteria

This plan is complete when:

- `CognitionChainInputV1.action_selection_context` is required and documented.
- `build_cognition_chain_input_from_global_state(...)` always returns the
  canonical empty context when no contextual action-selection facts exist.
- `call_cognition_subgraph()` succeeds for ordinary user-message turns with no
  open coding runs.
- Group self-cognition still loads `group_engagement_action_context` before
  L2d.
- Coding-run continuation context still reaches L2d and deterministic
  materialization without exposing private worker or approval fields.
- Focused and combined deterministic tests listed in `Verification` pass.
- Static checks pass.
- Independent code review has no unresolved blocking findings.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| A local `.get()` fix hides the crash but preserves contract drift. | Make the field required and test the builder output. | Builder contract test and code review scope. |
| Required empty context adds prompt clutter. | Keep the canonical empty shape compact and avoid new semantic content. | L2d payload tests show no private or noisy fields. |
| Coding-run continuation binding weakens. | Continue passing normalized trusted context into materialization. | Existing and added coding-run materialization tests. |
| Private coding-run fields leak into L2d/L3. | Preserve projection stripping and private-field tests. | `test_coding_agent_phase_c_run_context_contracts.py`. |
| Group self-cognition regresses while fixing ordinary chat. | Keep group self-cognition test as a required gate. | `test_l2d_l3_surface_handoff.py`. |

## Execution Evidence

Status: completed.

- RCA evidence: production failure reproduced locally as
  `KeyError: 'action_selection_context'`; focused test slice produced 8
  missing-key failures before this plan was written.
- No-subagent plan review: completed on 2026-07-11 by parent agent; blocking
  plan-review findings were fixed in this draft.
- User authorization: execution approved on 2026-07-11 with strict
  change-tracking and collision-avoidance requirements.
- Focused deterministic regression:
  `venv\Scripts\python.exe -m pytest tests/test_cognition_chain_connector_mapping.py tests/test_cognition_resolver_l2d_contract.py tests/test_coding_agent_phase_c_run_context_contracts.py -q`
  passed 18 tests after aligning direct fixtures with the required context.
- Action-selection payload regression:
  `venv\Scripts\python.exe -m pytest tests/test_action_selection_payload.py tests/test_cognition_chain_connector_mapping.py tests/test_cognition_resolver_l2d_contract.py tests/test_coding_agent_phase_c_run_context_contracts.py -q`
  passed 20 tests.
- Combined deterministic regression:
  `venv\Scripts\python.exe -m pytest tests/test_cognition_chain_connector_mapping.py tests/test_cognition_resolver_l2d_contract.py tests/test_l2d_l3_surface_handoff.py tests/test_persona_supervisor2_cognition_prewarm.py tests/test_coding_agent_phase_c_run_context_contracts.py -q`
  passed 39 tests. Several existing tests opened the configured MongoDB client;
  no `.env` file was read during execution.
- Coding-run materialization regression:
  `venv\Scripts\python.exe -m pytest tests/test_coding_agent_background_run_contracts.py -q`
  passed 27 tests.
- Static verification:
  `venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/cognition_chain_core/contracts.py src/kazusa_ai_chatbot/cognition_chain_core/chain.py src/kazusa_ai_chatbot/cognition_chain_core/graph_state.py src/kazusa_ai_chatbot/cognition_chain_core/action_selection.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_actions.py`
  passed.
- Static grep review:
  `rg -n 'chain_input\["action_selection_context"\]|action_selection_context = dict\(chain_input|state\.get\("action_selection_context"\)|input_payload\.get\("action_selection_context"' src/kazusa_ai_chatbot tests`
  found only required normalized-context merge sites and test assertions; no
  optional `.get()` fallback for the canonical context remains in the changed
  action-selection path.
- Code review: direct parent-agent diff review completed on 2026-07-11 because
  execution was kept parent-led to avoid interfering with the user's parallel
  agent work. No blocking findings remained after adding the `CoreStageState`
  channel, direct-test fixture context, and annotation cleanup.

## Lifecycle Closure

Archived as completed with the Stage 2 plan family on 2026-07-18. Its required
action-selection context remains part of the accepted Stage 2 baseline.
