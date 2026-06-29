# l2 affinity willingness boundary plan

## Summary

- Goal: Teach L2 to make a character-native relationship/willingness judgment
  for requests that feel too much for the current relationship, so L2 can
  refuse or deflect before L2d action routing.
- Plan class: medium
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `debug-llm`, `py-style`, `cjk-safety`,
  `test-style-and-execution`
- Overall cutover strategy: bigbang prompt-contract update inside the existing
  cognition chain. No new agent, no compatibility shim, no hidden tool removal,
  and no action-spec contract migration.
- Highest-risk areas: exposing operational effort/cost too early, turning
  affinity into a deterministic keyword classifier, letting L2d become the
  willingness decision-maker, over-refusing ordinary help, and breaking the
  route-only L2d/background-work boundary.
- Acceptance criteria: low-relationship demanding requests can be refused or
  deflected by L2 using existing stance/intent fields; simple requests remain
  answerable; L2d receives no affinity-gate or effort fields; private or
  background actions are not selected when upstream cognition has not accepted
  the request; focused deterministic and one-at-a-time live LLM evidence
  supports the behavior.

## Context

The current live response path is:

```text
adapter/debug client
  -> brain service
  -> queue/intake
  -> RAG
  -> cognition_chain_core L1/L2/L2d
  -> dialog / action-spec materialization
  -> persistence / scheduler / background workers
```

`cognition_chain_core` already has the right ownership line for this feature:

```text
L2a Conscious Core
  -> L2b Boundary Core
  -> L2c Judgment Core
  -> L2d Action Selection
```

Boundary Core already receives relationship context through the existing
affinity block and applies it to intimacy, control, and identity boundaries.
Judgment Core already converts upstream character and boundary readings into
`logical_stance`, `character_intent`, and `judgment_note`. L2d already owns
route-only action selection and is instructed to follow upstream cognition
rather than re-decide character stance.

The new behavior should build on that shape. The user does not want the system
to hide tools from Kazusa, expose backend effort, or ask L2d to decide access.
The desired behavior is more character-native:

```text
Given how Kazusa feels about this user and this request right now,
is this something she is willing to take on?
```

If the answer is no, L2 should settle the turn into a refusal, deflection, or
light banter outcome using the same existing stance/intent vocabulary. L2d
then routes only the visible response that follows from that settled cognition.

## Mandatory Skills

- `development-plan`: load before reviewing, approving, executing, updating,
  or signing off this plan.
- `local-llm-architecture`: load before changing L2, L2d, prompt flow,
  graph routing, LLM call budgets, or context contracts.
- `no-prepost-user-input`: load before changing logic that accepts, rejects,
  routes, persists, or acts on user requests.
- `debug-llm`: load before local/live LLM runs, prompt comparisons, trace
  inspection, or human-readable LLM quality artifacts.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files or tests containing CJK
  prompt strings or CJK test data.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or final
  reporting.
- After signing off any major checklist stage, reread this entire plan before
  starting the next stage.
- Do not execute implementation steps while `Status` is `draft`.
  Implementation requires user approval, status `approved` or `in_progress`,
  and a direct user instruction to execute.
- Do not add a new agent or LLM call for this feature.
- Do not add a numeric affinity threshold, environment variable, config flag,
  database field, or migration for this feature in Phase 1.
- Do not hide, remove, or conditionally suppress action affordances from L2d
  to manufacture the gate.
- Do not expose operational concepts to L2 or L2d as decision criteria:
  `resource heavy`, `effort`, `tool cost`, `background_work`,
  `background_work_request`, `complex_task_resolution`, worker names, tool
  names, queue internals, or an `affinity threshold`.
- Do not add a deterministic keyword classifier over raw user input,
  `decontextualized_input`, conversation text, or action summaries to decide
  whether a request is too much.
- L2 owns semantic willingness. L2d follows the settled L2
  `logical_stance`, `character_intent`, and `judgment_note`.
- L2d must not receive a new affinity-gate field, effort field, cost field, or
  task-complexity category.
- Keep L2d route-only. It must not generate final dialog wording, worker task
  briefs, tool parameters, or scheduler plans.
- Deterministic code continues to own validation, action materialization,
  queue persistence, adapter delivery, limits, and schema safety. It must not
  rewrite semantic willingness after the LLM stages.
- If implementation discovers that a deterministic runtime backstop is needed
  for resource safety, stop and update this plan for approval before adding it.
  Phase 1 proves the prompt/contract behavior first.

## Must Do

- Update L2 Boundary Core prompt flow so relationship context contributes to a
  character-native willingness boundary:
  - whether the request feels too demanding, too familiar, too intimate, too
    controlling, or too much to take on for this person right now;
  - whether Kazusa should accept, soften, tease, deflect, or refuse;
  - without describing backend work, hidden tools, cost, queueing, or effort.
- Keep Boundary Core output shape compatible. Reuse existing fields such as
  `boundary_summary`, `acceptance`, `stance_bias`, `pressure_policy`, and
  `trajectory` instead of adding an affinity-gate result object.
- Update L2 Judgment Core prompt flow so the final
  `logical_stance`, `character_intent`, and `judgment_note` honor the
  willingness boundary when it is stronger than the raw helpful impulse.
- Preserve normal helpfulness for simple, low-pressure, low-commitment
  requests even when affinity is low.
- Update L2d action-selection prompt flow so it treats refusal, deflection,
  teasing, and non-commitment as settled upstream cognition. In those cases,
  L2d should choose only visible speaking/no-response surfaces that match the
  stance, not private/future/background work.
- Keep the `background_work_request` affordance visible in L2d's capability
  roster. The router should know the tool exists, but should not select it
  when L2 has not chosen to take the request on.
- Add deterministic prompt-contract tests that verify the relevant L2 prompt
  text does not introduce operational gate language, hidden tool names, numeric
  thresholds, or effort categories.
- Add focused deterministic tests around prompt payload shape showing no new
  affinity-gate, effort, or complexity field is passed into L2d.
- Add fake-LLM or parser-level regression coverage for:
  - low-affinity demanding request can resolve to refusal, deflection, or
    banter through existing L2 stance/intent fields;
  - low-affinity simple request can still resolve to ordinary help;
  - high-affinity demanding request is not automatically refused;
  - L2d route selection follows a non-accepting L2 outcome with `speak` or no
    action, not background/private action scheduling.
- Add one-at-a-time live LLM inspection cases, marked `live_llm`, for:
  - low-affinity simple help;
  - low-affinity demanding sustained help;
  - high-affinity demanding sustained help;
  - low-affinity request that would otherwise be eligible for background work.
- Record prompt-flow review notes and live LLM observations in this plan's
  `Execution Evidence` before sign-off.

## Deferred

- Do not add a new L2b2, affinity-gate, request-effort, or access-control
  agent.
- Do not add threshold configuration, environment variables, character-profile
  schema fields, database fields, or migrations.
- Do not add `affinity_task_gate`, `effort_score`, `complexity_score`, or
  similar fields to `CognitionChainInputV1`, graph state, L2 output, or L2d
  payloads.
- Do not integrate `complex_task_resolver` or expose
  `complex_task_resolution` to L2d as part of this plan.
- Do not redesign action-spec, background-work queueing, worker routing,
  scheduler, adapters, dialog rendering, RAG, consolidation, or persistence.
- Do not add compatibility shims, parallel prompt paths, fallback mappers, or
  legacy aliases.
- Do not rely on fixture-specific prompt wording, keyword lists, or hardcoded
  examples to force the desired behavior.

## Cutover Policy

Overall strategy: bigbang prompt-contract update inside the existing L2/L2d
flow.

| Area | Strategy | Policy |
|---|---|---|
| L2 Boundary Core prompt | bigbang | Rewrite the affected prompt section coherently around relationship willingness. Do not append an isolated rule block. |
| L2 Judgment Core prompt | bigbang | Reconcile final stance/intent instructions with Boundary Core willingness in the same change. |
| L2d action-selection prompt | bigbang | Clarify that settled refusal/deflection/non-commitment routes only visible response/no-response, without adding affinity or effort fields. |
| Public cognition contracts | unchanged | No new `CognitionChainInputV1`, output, or graph-state fields in Phase 1. |
| Action-spec capabilities | unchanged | Keep existing action kinds and prompt-safe affordance projection. |
| Background work | unchanged | No worker, queue, or router behavior change in Phase 1. |
| Tests | compatible/additive | Add focused regression and live inspection coverage without replacing existing L2d/action tests. |

Rollback is code-level only: revert the L2/L2d prompt updates and associated
tests together. Do not keep tests that assert behavior no longer described by
the prompt.

## Target State

The intended runtime behavior is:

```text
User asks for something demanding or familiar
  -> L2b reads relationship context as a character boundary signal
  -> L2c settles final stance/intent
  -> L2d sees that Kazusa is not taking it on
  -> L2d selects speak/no-response only
  -> dialog renders a natural refusal, tease, deflection, or smaller offer
```

For ordinary requests:

```text
User asks for simple help
  -> L2b does not over-treat low affinity as rejection
  -> L2c may accept or answer normally
  -> L2d can select the appropriate existing action route
```

The completed feature should feel like Kazusa has better judgment about what a
relationship entitles the user to ask from her. It should not feel like a tool
permission system.

## Design Decisions

- Use existing L2 stages instead of a new agent.
  - Pro: preserves latency and keeps character judgment in the cognition layer.
  - Con: prompt wording must be carefully reviewed because there is no separate
    inspectable gate object.
- Use existing relationship/affinity context instead of a new threshold.
  - Pro: matches the desired gut-feeling behavior and avoids brittle numeric
    tuning.
  - Con: live LLM behavior must be inspected because there is no exact cutoff.
- Keep action affordances visible to L2d.
  - Pro: Kazusa remains aware of possible actions and does not look incapable.
  - Con: L2d prompt discipline and tests must prevent route contradiction.
- Do not add a deterministic runtime backstop in Phase 1.
  - Pro: honors LLM-first semantic judgment and avoids hidden post-processing.
  - Con: live evaluation must confirm that L2d reliably follows non-accepting
    upstream cognition before relying on this for expensive future tools.

## Contracts And Data Shapes

- `CognitionChainInputV1`: unchanged.
- L2 Boundary Core output schema: unchanged.
- L2 Judgment Core output schema: unchanged.
- L2d action-selection input payload: unchanged except for prompt prose that
  tells the model how to interpret existing cognition fields.
- Action-spec capability registry: unchanged.
- Background-work request schema and queue documents: unchanged.
- Debug traces may be used to inspect existing L2 fields, but no new public or
  persisted affinity-gate field is introduced.

If implementation finds a real need for a new field, stop and update this plan
before adding it.

## LLM Call And Context Budget

- No new live response-path LLM calls.
- L2b, L2c, and L2d keep their current stage count and JSON contracts.
- Expected prompt growth should stay small: relationship-willingness guidance
  belongs in concise role/procedure text, not expanded examples.
- Dynamic human payloads should remain unchanged. In particular, do not pass a
  new `effort`, `complexity`, `gate`, or threshold object to L2d.
- Live LLM verification must run one case at a time and inspect the full L2 and
  L2d trace before marking a case pass.

## Change Surface

Expected production files:

- `src/kazusa_ai_chatbot/cognition_chain_core/stages/l2.py`
  - revise Boundary Core and Judgment Core prompt text only;
  - preserve output schemas and graph wiring.
- `src/kazusa_ai_chatbot/cognition_chain_core/action_selection_prompt.py`
  - revise L2d prompt text so action routing follows settled refusal or
    deflection;
  - do not add gate metadata or hide capabilities.

Expected test files:

- `tests/test_cognition_prompt_contract_text.py`
- `tests/test_action_selection_prompt_contract.py`
- `tests/test_action_selection_payload.py`
- `tests/test_cognition_chain_core_action_selection.py`
- `tests/test_l2d_action_selection_cases.py`
- `tests/test_l2d_action_selection_live_llm.py`
- `tests/test_cognition_live_llm_boundary_affinity.py`

New tests may be added if the existing files do not provide a clean location,
for example:

- `tests/test_cognition_live_llm_affinity_willingness.py`

Files intentionally out of scope:

- `src/kazusa_ai_chatbot/action_spec/*`
- `src/kazusa_ai_chatbot/background_work/*`
- `src/kazusa_ai_chatbot/cognition_resolver/*`
- `src/kazusa_ai_chatbot/rag/*`
- adapter modules
- database migrations

## Overdesign Guardrail

The implementation must remain a prompt-contract refinement unless evidence
forces escalation. Do not introduce:

- another LLM stage;
- a policy engine;
- a scoring model;
- a threshold table;
- per-tool affinity requirements;
- user-tier or entitlement state;
- background worker changes;
- action capability filtering;
- generalized permission middleware.

The smallest acceptable implementation is three coherent prompt updates plus
focused tests and live LLM evidence.

## Agent Autonomy Boundaries

- L2b may interpret relationship context as pressure, familiarity, intimacy,
  trust, entitlement, and willingness.
- L2c may convert that interpretation into the final character stance and
  intent.
- L2d may select actions only after reading that settled final cognition.
- L2d may not decide that low affinity blocks a task. It only follows L2's
  accepted, refused, deflected, or teasing outcome.
- Dialog/L3 owns final wording and visible softness.
- Deterministic code owns schemas, parser validation, queue execution, and
  delivery mechanics.

## Implementation Order

1. Re-read this plan, `development_plans/README.md`, relevant subsystem
   READMEs, and the current source/test files named in `Change Surface`.
2. Load mandatory skills required for the next step.
3. Inspect current L2b, L2c, and L2d prompt text and write a short prompt-flow
   audit before editing.
4. Update L2b Boundary Core prompt text around relationship willingness using
   existing output fields.
5. Update L2c Judgment Core prompt text so final stance/intent honors that
   willingness boundary.
6. Update L2d action-selection prompt text so non-accepting settled cognition
   routes only visible response/no-response.
7. Add deterministic prompt-contract and payload tests.
8. Add or update fake-LLM/parser-level route-following tests.
9. Run focused deterministic tests.
10. Run live LLM inspection cases one at a time, recording outputs and pass
    judgment.
11. Run broader relevant regression tests.
12. Run the independent code review gate and record results.
13. Update `Execution Evidence` and lifecycle status only after verification.

## Execution Model

- Use parent-led implementation unless the user explicitly approves delegated
  subagent execution.
- Keep commits and changes scoped to this plan.
- Do not continue into deferred integration work without a new or updated
  approved plan.
- If live LLM evidence shows unstable or over-refusing behavior, stop after
  recording evidence and revise the prompt design before touching more code.

## Progress Checklist

- [ ] Status changed from `draft` only after user approval.
- [ ] Mandatory skills loaded before relevant edits/tests.
- [ ] Prompt-flow audit recorded.
- [ ] L2b prompt updated.
- [ ] L2c prompt updated.
- [ ] L2d prompt updated.
- [ ] No public contract/schema/payload gate fields added.
- [ ] Deterministic prompt-contract tests added or updated.
- [ ] Deterministic action-selection/payload tests added or updated.
- [ ] Focused deterministic tests pass.
- [ ] Live LLM low-affinity simple-help case inspected.
- [ ] Live LLM low-affinity demanding-help case inspected.
- [ ] Live LLM high-affinity demanding-help case inspected.
- [ ] Live LLM background-eligible refusal case inspected.
- [ ] Broader relevant regression tests pass.
- [ ] Independent code review gate completed.
- [ ] Execution evidence recorded.

## Verification

Focused deterministic commands:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_prompt_contract_text.py
venv\Scripts\python -m pytest tests\test_action_selection_prompt_contract.py
venv\Scripts\python -m pytest tests\test_action_selection_payload.py
venv\Scripts\python -m pytest tests\test_cognition_chain_core_action_selection.py
venv\Scripts\python -m pytest tests\test_l2d_action_selection_cases.py
```

Focused live LLM commands must run one case at a time, with output inspected:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_live_llm_boundary_affinity.py -m live_llm
venv\Scripts\python -m pytest tests\test_l2d_action_selection_live_llm.py -m live_llm
```

If a new live test file is added:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_live_llm_affinity_willingness.py -m live_llm
```

Static source checks:

```powershell
rg "affinity_task_gate|effort_score|complexity_score|tool cost|affinity threshold" src\kazusa_ai_chatbot\cognition_chain_core
rg "resource heavy|complex_task_resolution" src\kazusa_ai_chatbot\cognition_chain_core\stages\l2.py
```

Expected result: no new source usage in cognition-chain prompts or payloads.
If an existing unrelated match appears, record it and inspect manually rather
than deleting unrelated code.

Broader regression candidates:

```powershell
venv\Scripts\python -m pytest tests\test_cognition_resolver_l2d_contract.py tests\test_persona_supervisor2_action_selection.py tests\test_l2d_l3_surface_handoff.py
```

## Independent Code Review

Before completion, run an independent review pass with this brief:

```text
Review the L2 affinity willingness boundary implementation for architecture
violations, prompt-contract drift, hidden deterministic semantic gating,
L2d ownership bleed, action-affordance hiding, over-refusal risk, missing
tests, and live LLM evidence gaps. Check the changed files and the plan.
Return findings with file/line references, ordered by severity.
```

Record the review result in `Execution Evidence`. If findings require code
changes, fix them and rerun the relevant focused tests before sign-off.

## Acceptance Criteria

- L2 can refuse, deflect, or tease away demanding requests when relationship
  context makes the request feel too much.
- L2 can still accept or answer simple low-pressure requests at low affinity.
- The final refusal/deflection decision is represented through existing
  `logical_stance`, `character_intent`, and `judgment_note` behavior.
- L2d receives no new affinity gate, effort, cost, threshold, or complexity
  metadata.
- L2d does not schedule private, future, or background work when settled L2
  cognition has not accepted the request.
- Action affordances remain visible; the tool is not hidden from Kazusa.
- No new agent, schema migration, DB field, action kind, or background worker
  behavior is added.
- Focused deterministic tests pass.
- One-at-a-time live LLM inspection shows acceptable behavior for the planned
  low/high affinity and simple/demanding request cases.

## Risks

- Prompt drift could make Kazusa refuse too many ordinary requests.
- Local LLM behavior may be inconsistent because the gate is intentionally
  gut-feeling rather than threshold-based.
- L2d may still occasionally select a private action after a non-accepting L2
  outcome; Phase 1 should measure this before adding deterministic backstops.
- Existing affinity wording may already expose relationship level in a way that
  needs careful prompt framing to avoid numeric cutoff behavior.
- Live LLM tests may depend on local model availability and should record skips
  or blockers explicitly.

## Execution Evidence

- 2026-06-30: Draft created from user-approved architectural conclusion:
  keep tools visible, keep decision-making in L2, avoid exposing effort/cost or
  thresholds, avoid a new agent, and make L2d follow settled L2 cognition.
