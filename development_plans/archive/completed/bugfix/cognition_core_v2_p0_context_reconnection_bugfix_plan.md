# Cognition Core V2 P0 Context Reconnection Bugfix Plan

## Summary

- Goal: restore three completed cognition inputs unintentionally disconnected
  by the Cognition Core V2 big-bang cutover:
  1. cycle-zero shared-memory prewarm;
  2. private past-dialog cognition residual;
  3. group-engagement guidance for group self-cognition participation and
     action planning.
- Plan class: `large`.
- Status: `completed`.
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`,
  `cjk-safety`, `test-style-and-execution`, `debug-llm`, `character-test`, and
  `python-venv`.
- Overall cutover strategy: one forward-only native V2 reconnection. Reuse the
  existing producers and completed semantic contracts, add exact V2 carriers,
  and remove no V2 behavior. Create no V1 compatibility graph, parallel
  vocabulary, fallback mapper, or feature flag.
- Highest-risk areas: restoring a helper caller without restoring its consumer
  guarantee; widening private residual visibility; turning group style into
  topic or factual evidence; increasing cycle-zero latency; and allowing
  rewritten tests to remain green while production consumers are absent.
- Acceptance criteria: normal cycle-zero cognition invokes and joins bounded
  shared-memory prewarm; a trace-backed past-dialog residual reaches only V2
  goal cognition; group self-cognition receives bounded engagement guidance at
  goal/action judgment; focused tests fail when any producer-to-consumer edge
  is removed; normal-entrypoint trace evidence proves each live path; and the
  wider non-live cognition/resolver/self-cognition suites pass.
- Execution authority: completed after the user's explicit 2026-07-30
  implementation command and 2026-07-31 final evidence approval.

## Context

The three producers still exist and their direct tests pass:

| Contract | Existing producer | Missing production edge |
|---|---|---|
| Shared-memory prewarm | `cognition_resolver.capabilities.run_first_cycle_shared_memory_prewarm(...)` and `merge_shared_memory_prewarm_result(...)` | `persona_supervisor2_cognition.call_cognition_subgraph(...)` has no caller/join before V2 input construction |
| Past-dialog residual | `past_dialog_cognition` projection plus reply/RAG attachment in `service.py` and `cognition_resolver/loop.py` | `build_cognition_input_from_global_state(...)` and `CognitionCoreInputV2` do not carry `past_dialog_cognition_context` |
| Group engagement | `db.interaction_style_images.build_group_engagement_action_context(...)` | group self-cognition no longer loads it and V2 has no participation/action carrier |

The completed historical contracts are:

- `archive/completed/short_term/unconditional_shared_memory_prewarm_plan.md`;
- `archive/completed/short_term/past_dialog_cognition_residual_plan.md`; and
- `archive/completed/bugfix/action_selection_context_contract_bugfix_plan.md`.

The last V1 connector before cutover started prewarm and group-engagement tasks
inside `call_cognition_subgraph(...)`, joined prewarm at the first RAG consumer,
and supplied group engagement to action selection. Past-dialog residual had one
private L2a-only field. Commit `9210bede` replaced the production connector with
native V2, while commit `3a124732` substantially rewrote the named regression
tests. The producers survived, but the connector fields and consumer
assertions did not. This is contract-loss during a big-bang boundary rewrite,
not failed retrieval, bad memory content, or model refusal.

The reported `napcat` row
`seed_7ac6348ccd9bf7a80fbc74584c6b3ce3` is historical RCA evidence: direct
shared-memory retrieval could return the row, while normal conversation had no
prewarm caller. The production fix must be proven with a guarded synthetic
memory row so acceptance does not depend on one mutable production document.
The named row may be used for an additional read-only smoke when available.

This plan is deliberately narrower than
`cognition_core_v2_short_horizon_global_state_composition_bigbang_plan.md`.
That draft owns one future immutable user/group style snapshot across
relevance, cognition, surface, telemetry, and console. This P0 plan restores
only the already-completed group self-cognition engagement contract. The wider
plan must rebaseline after this fix and reuse its carrier/projection rather than
add a second loader or vocabulary.

## Mandatory Skills

- `development-plan`: plan lifecycle, test-first execution, evidence, review,
  and archival rules.
- `local-llm-architecture`: V2 semantic ownership, exact context contracts,
  prompt budgets, call counts, latency, and narrow blast radius.
- `py-style` and `cjk-safety`: all Python changes and any CJK prompt text.
- `test-style-and-execution`: parent-owned tests, deterministic versus live
  execution, one-at-a-time live cases, and output inspection.
- `debug-llm`: human-readable trace and quality-review artifacts.
- `character-test`: normal/debug intake E2E dialog, protected trace inspection,
  RAG/memory/cognition evidence, and database-effect checks.
- `python-venv`: use `venv\Scripts\python.exe` for all Python and pytest work.

## Mandatory Rules

1. Production changes remain blocked while this plan is `draft`.
2. Before execution, reread this plan, `development_plans/README.md`, root and
   relevant subsystem READMEs, current source/tests, and `git status --short`.
3. Do not read `.env`. Use the configured project environment and guarded test
   helpers.
4. Preserve ownership: RAG returns evidence; goal cognition decides subjective
   use; action planning selects semantic capabilities; dialog owns wording.
5. Restore the current producers. Do not create replacement retrieval,
   residual, or interaction-style subsystems.
6. Use exact native V2 fields. Do not revive `cognition_chain_core`, V1 L2
   stages, generic compatibility dictionaries, aliases, dual paths, or
   feature-flagged fallback.
7. Shared prewarm runs only on resolver cycle zero and only where the existing
   eligibility contract permits it. It reads shared `memory`, excludes
   `user_memory_units`, leaves `rag_result.answer` empty, and creates no
   resolver observation.
8. Prewarm starts before independent cycle-zero preparation and joins exactly
   once before `build_cognition_input_from_global_state(...)` turns
   `rag_result.memory_evidence` into typed V2 evidence.
9. Empty, unresolved, malformed, timed-out, or failed prewarm preserves the
   base `rag_result` and lets cognition continue.
10. `past_dialog_cognition_context` remains optional weak private context,
    bounded to three dialogs and 1,800 characters. It is not factual evidence,
    a command, a selected stance, or dialog wording.
11. V2 goal cognition is the native replacement for the former L2a subjective
    consumer. Semantic appraisal, workspace collapse, action planning,
    resolver observations, RAG, L3/surface, dialog, consolidation, scheduler,
    reflection, adapters, and delivery receive no past-dialog residual.
12. Keep `past_dialog_cognition_context` distinct from
    `private_continuity_context`; the former keeps its 1,800-character
    trace-backed contract and the latter keeps its 1,000-character internal
    monologue-residue contract.
13. Group engagement loads only for group self-cognition under the canonical
    episode scope predicate. Ordinary user turns, private self-cognition, and
    targetless non-group events receive the exact empty value without a group
    style database read.
14. Group engagement remains advisory participation guidance. It cannot add a
    topic, event, fact, relationship belief, command, permission, or reason to
    speak unsupported by current observed context.
15. Goal cognition may use group guidance to form a participation intention.
    Action planning may use the same bounded projection to decide compatible
    current/future semantic speech requests. Workspace collapse receives bids,
    not a second copy of the style context.
16. The P0 loader/projection is the only cognition-facing group-engagement
    lane. The broader short-horizon plan must reuse it after rebaseline.
17. Add no persistence schema, migration, backfill, database write, new model
    stage, retry loop, or model route.
18. Parse and validation ownership stays at existing canonical boundaries.
    Deterministic code validates shapes, caps, scope, merge policy, and
    visibility; LLM stages retain semantic judgment.
19. Parent writes and runs failing consumer-edge tests before production work.
20. Exactly one production-code subagent implements the approved source/docs
    scope. A separate independent review subagent runs after verification.
    If native subagents are unavailable, stop unless the user explicitly
    approves fallback execution.
21. After context compaction, reread this entire plan before implementation,
    verification, lifecycle changes, or sign-off.
22. Production and test diffs must preserve unrelated user changes.

## Must Do

- Restore cycle-zero prewarm task creation, join, merge, V2 evidence mapping,
  and downstream resolver-state propagation.
- Add one exact optional V2 input field named
  `past_dialog_cognition_context`.
- Project that field into goal cognition only and prove all forbidden
  consumers remain disconnected.
- Add one exact optional V2 input field named
  `group_engagement_action_context`, with the existing shape:
  `{"engagement_guidelines": list[str], "confidence": str}`.
- Load that field once for eligible group self-cognition and pass the same
  immutable bounded value to V2 goal cognition and action planning.
- Restore test names or add explicit replacement tests whose assertions cover
  the production caller and final consumer, not only helper output.
- Add static topology checks that fail when a helper becomes orphaned or a
  private field reaches a forbidden consumer.
- Update cognition-resolver, V2 cognition, node, past-dialog, and
  self-cognition documentation to describe the restored native paths.
- Capture deterministic evidence, live trace evidence, independent review, and
  final lifecycle evidence in this plan.

## Deferred

- The complete user/group interaction-style snapshot shared across relevance,
  cognition, surface, telemetry, and console.
- User engagement relevance changes and ordinary user-message style effects on
  cognition.
- New memory sources, scoped-user prewarm, RAG prompt tuning, retrieval ranking
  changes, or memory-database maintenance.
- Past-dialog summarization, new trace stages, backfill, longer retention,
  semantic retrieval gates, or additional consumers.
- Group participation ratio tuning, scheduler changes, response suppression,
  and broad self-cognition redesign.
- Control-console panels or new production telemetry beyond existing protected
  LLM traces.

## Cutover Policy

Overall strategy: `bigbang` restoration into the one V2 production graph.

| Surface | Strategy | Enforcement |
|---|---|---|
| Cycle-zero memory evidence | big-bang restore | One caller and one join in the V2 connector; no V1 path |
| Past-dialog private context | big-bang native field | Exact V2 field, exact goal-only projection, no generic residue concatenation |
| Group engagement | big-bang native field | One existing producer, one eligible load, same projection for goal/action judgment |
| Existing RAG2 resolver path | preserve | Full RAG remains resolver-selected; prewarm is memory-evidence-only |
| Existing L3 interaction style | preserve | Surface style remains until the broader approved plan replaces composition |
| Tests | big-bang | Replace helper-only green gates with caller-to-consumer contracts |
| Data | no migration | Existing memory, trace, and style documents remain authoritative |

Cutover enforcement:

- `rg` must show one production caller for
  `run_first_cycle_shared_memory_prewarm(...)` outside its definition.
- `rg` must show one production caller for
  `build_group_engagement_action_context(...)` outside exports/definition.
- Exact private-field allowlists and forbidden-consumer greps are blocking.
- No import from `cognition_chain_core` may be added.
- The old broad short-horizon plan is rebaselined after this plan completes.

## Target State

```text
normal resolver cycle zero
  -> V2 connector validates episode/scope
  -> start eligible shared-memory prewarm
  -> start eligible group-engagement load for group self-cognition
  -> load identity + mutable cognition state
  -> join prewarm and merge only memory_evidence into base rag_result
  -> join group engagement and validate bounded exact projection
  -> build CognitionCoreInputV2
       evidence <- merged rag_result
       past_dialog_cognition_context <- state private residual, if present
       group_engagement_action_context <- eligible group projection, else empty
  -> semantic appraisal sees typed RAG evidence under normal visibility rules
  -> goal cognition sees RAG evidence + private past-dialog context
     + advisory group participation context
  -> workspace collapse sees complete bids
  -> action planning sees admitted bids + advisory group action context
  -> output/state/resolver recurrence preserves merged RAG evidence
  -> L3/dialog receive neither private past-dialog residual nor the exact
     action-planning group context
```

Later resolver cycles reuse the state already produced by cycle zero. They do
not repeat prewarm or group style reads.

## Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Prewarm join point | Immediately before canonical V2 input construction | This is the native first evidence-consumer boundary and allows identity/state work to overlap retrieval |
| Prewarm carrier | Existing `rag_result.memory_evidence` mapped to typed `promoted_memory` evidence | Memory remains evidence with provenance instead of becoming persona/private prose |
| Past-dialog carrier | Dedicated `past_dialog_cognition_context: str` | Preserves private ownership and its independent 1,800-character cap |
| Past-dialog consumer | Goal cognition only | It is the native subjective-intention successor to former L2a |
| Group carrier | Dedicated `group_engagement_action_context` mapping | Preserves the completed producer vocabulary and avoids an untyped action bag |
| Group consumers | Goal cognition and action planning | V2 splits intention formation from semantic action selection; both are needed to restore the former L2d participation contract |
| Workspace input | Bids only | Guidance influences branch judgment before collapse and does not become a second ranking authority |
| Live seed | Guarded synthetic row; optional read-only `napcat` smoke | Makes the regression reproducible without production-data dependence |
| Failure policy | Existing empty/base context | These advisory inputs degrade to omission and cannot crash normal conversation |

## Contracts And Data Shapes

`CognitionCoreInputV2` gains:

```python
past_dialog_cognition_context: NotRequired[str]
group_engagement_action_context: NotRequired[GroupEngagementActionContextV2]
```

The exact group shape is:

```python
class GroupEngagementActionContextV2(TypedDict):
    engagement_guidelines: list[str]
    confidence: str
```

Validation:

- past-dialog context: string, maximum 1,800 characters, omission or `""`
  means unavailable;
- engagement guidelines: existing per-field count and string caps from
  `interaction_style_images`; no empty items;
- confidence: existing bounded prompt-safe string;
- ineligible events normalize to the exact empty shape;
- unknown keys or wrong types fail at V2 input validation before model calls.

Goal prompt projection includes both fields under separate semantic keys.
Action-planning prompt includes only `group_engagement_action_context`.
Neither field is copied to evidence rows, bids, selected intention, state
updates, diagnostics, residue, text-surface input, or dialog input.

## LLM Call And Context Budget

| Path | Before | After |
|---|---|---|
| Ordinary cycle-zero without resolver RAG | No prewarm worker calls | At most one persistent-memory generator call and one judge call, `max_attempts=1`; cache hit may add zero |
| Later resolver cycle | No prewarm worker calls | Unchanged |
| Past-dialog residual | Existing V2 calls | Same calls; at most 1,800 extra characters in goal cognition only |
| Eligible group self-cognition | Existing V2 calls | Same calls; one bounded DB projection and bounded context in goal/action planning |
| Ineligible event | Existing V2 calls | Same calls and no group-style DB read |

The prewarm task starts before independent identity/mutable-state preparation.
Only the prewarm join blocks canonical input construction; group engagement
runs in the same preparation window. Existing 24,000-character goal/action
caps remain fixed. Required current episode, role bindings, constraints, and
decision-critical evidence retain priority. Optional private past-dialog and
group guidance use stable supplemental-context reduction and omission before
required context can be displaced. No cap increase is allowed in this plan.

## Change Surface

### Modify Production

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - start/join cycle-zero prewarm and eligible group engagement;
  - merge into a local state before V2 input construction;
  - map both exact V2 fields.
- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
  - add exact optional fields and validators.
- `src/kazusa_ai_chatbot/cognition_core_v2/facade.py`
  - project past-dialog and group contexts to their authorized consumers;
  - pass group context into action planning.
- `src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py`
  - include separately labeled bounded private residual and advisory group
    participation context in supplemental budgeting.
- `src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py`
  - accept and render bounded group action guidance without treating it as
    evidence, permission, or route authority.

### Keep

- `cognition_resolver/capabilities.py` prewarm helper and merge contract.
- `past_dialog_cognition/**`, `service.py`, and
  `cognition_resolver/loop.py` residual producers.
- `db/interaction_style_images.py` group projection and caps.
- Existing resolver-selected full RAG2, interaction-style surface projection,
  persistence, consolidation, scheduler, adapters, and delivery.

### Modify Tests

- `tests/test_shared_memory_prewarm.py`
  - retain focused helper/source/merge tests.
- `tests/test_persona_supervisor2_cognition_prewarm.py`
  - restore cycle-zero production caller, concurrency, join, V2 evidence, base
    fallback, later-cycle exclusion, and propagated-state assertions.
- `tests/test_past_dialog_cognition_prompt_boundaries.py`
  - assert V2 input mapping, goal prompt visibility, and all negative
    boundaries.
- `tests/test_cognition_chain_connector_mapping.py`
  - assert eligible group load, exact V2 mapping, ineligible no-read behavior,
    and repeated-cycle reuse.
- `tests/test_cognition_core_v2_contracts.py`
  - exact field/shape/cap validation.
- `tests/test_cognition_core_v2_integration.py`
  - final native consumer boundaries for both contexts.
- `tests/test_cognition_core_v2_prompt_budget_continuity.py`
  - stable reduction at combined worst-case optional context.
- Add one narrowly named combined regression file only if the existing owners
  cannot express the three production-edge assertions without duplication.

### Modify Documentation

- `src/kazusa_ai_chatbot/cognition_resolver/README.md`
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `src/kazusa_ai_chatbot/past_dialog_cognition/README.md`
- `src/kazusa_ai_chatbot/self_cognition/README.md`
- `development_plans/README.md` and this plan's lifecycle record.

## Overdesign Guardrail

- Actual problem: completed producer-to-consumer edges disappeared during the
  V2 cutover while helper-level tests stayed green.
- Smallest safe solution: reconnect the existing producers through three exact
  native V2 inputs and test each full edge.
- Semantic owners: retrieval supplies evidence; goal cognition uses private
  continuity and participation guidance; action planning uses only group action
  guidance; deterministic code owns scope, validation, caps, merge, and
  visibility.
- Rejected complexity: V1 graph restoration, generic context registry,
  sidecar memory agent, new style snapshot architecture, keyword gates,
  semantic fetch classifiers, new persistence, feature flags, fallback
  mappers, prompt-cap increases, and response-ratio suppression.
- Expansion threshold: any additional memory source, residual consumer, style
  consumer, persistent field, or model call requires separate failure evidence
  and an approved plan update.

## Agent Autonomy Boundaries

- Parent owns tests, expected failing baselines, orchestration, verification,
  live cases, evidence, review remediation, and lifecycle.
- One production subagent may edit only the production/docs files listed in
  `Change Surface` after parent tests establish the contract.
- The production subagent may reuse an existing narrow validator or helper
  discovered during implementation when it preserves the exact plan contract.
- Stop for plan update if implementation requires a new model call, new
  persistence, a public-output field, a compatibility layer, an additional
  consumer, or a change to the broader short-horizon plan.
- One independent review subagent receives the approved plan, final diff,
  failing baseline, verification output, and live trace review. It edits
  nothing.

## Implementation Order

1. Parent rebaselines HEAD, worktree, relevant plans/docs, current source,
   producer callers, and all named tests.
2. Parent records topology greps proving the three consumer edges are absent.
3. Parent adds/restores failing production-edge tests for prewarm, past-dialog,
   and group engagement.
4. Parent runs each focused test file and records expected failures.
5. Production subagent adds exact V2 contracts and deterministic validation.
6. Production subagent restores cycle-zero task start/join/merge and group
   eligibility/loading in the connector.
7. Production subagent adds goal-only past-dialog projection and goal/action
   group projection with existing budgets.
8. Production subagent updates the listed subsystem documentation.
9. Parent runs focused tests after each contract edge, then combined non-live
   regression and static checks.
10. Parent runs live cases one at a time through a normal or real debug intake,
    captures full protected traces, and writes human-readable reviews.
11. Parent checks the diff against the broader active style plan and records
    the future rebaseline obligation.
12. Independent code-review subagent audits correctness, privacy, ownership,
    latency, tests, and plan compliance.
13. Parent remediates in-scope findings, reruns affected gates, records all
    evidence, and requests user sign-off before lifecycle completion.

## Execution Model

- Parent-led, test-first, one production subagent, one independent review
  subagent.
- Deterministic tests may run in batches after focused failures/passes are
  inspected.
- Live LLM/E2E cases run one at a time. Each result and trace is inspected
  before starting the next case.
- Guarded live database fixtures use the existing isolation helpers and clean
  up only their exact test rows.
- A read-only `napcat` smoke is supplemental evidence and cannot replace the
  guarded synthetic acceptance case.

## Progress Checklist

- [x] Stage 1 - current topology and expected failing baselines recorded
- [x] Stage 2 - exact V2 input contracts and validators complete
- [x] Stage 3 - cycle-zero shared-memory prewarm reconnected
- [x] Stage 4 - past-dialog residual reaches V2 goal cognition only
- [x] Stage 5 - group engagement reaches eligible V2 participation/action
      judgment
- [x] Stage 6 - focused and combined deterministic verification passes
- [x] Stage 7 - one-at-a-time live E2E traces and readable reviews accepted
- [x] Stage 8 - independent code review remediated
- [x] Stage 9 - user sign-off, lifecycle update, and archive complete

## Verification

### Static Topology

```powershell
rg -n "run_first_cycle_shared_memory_prewarm|merge_shared_memory_prewarm_result" src\kazusa_ai_chatbot
rg -n "build_group_engagement_action_context" src\kazusa_ai_chatbot
rg -n "past_dialog_cognition_context" src\kazusa_ai_chatbot
rg -n "group_engagement_action_context" src\kazusa_ai_chatbot
rg -n "cognition_chain_core" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\cognition_core_v2
```

Expected:

- one connector caller plus helper definition/export matches for prewarm and
  group engagement;
- past-dialog matches only producer/state plumbing, V2 contract/facade, and
  goal cognition;
- no past-dialog matches in semantic appraisal, workspace, action selection,
  output projection, L3, dialog, RAG, resolver contracts, consolidation,
  scheduler, reflection, adapters, or delivery;
- group action-context matches only producer/connector, V2
  contract/facade/goal/action planning, and docs;
- no new V1 cognition import.

### Focused Deterministic Tests

```powershell
venv\Scripts\python.exe -m pytest tests\test_shared_memory_prewarm.py -q
venv\Scripts\python.exe -m pytest tests\test_persona_supervisor2_cognition_prewarm.py -q
venv\Scripts\python.exe -m pytest tests\test_past_dialog_cognition_context.py tests\test_past_dialog_cognition_reply_integration.py tests\test_past_dialog_cognition_rag_integration.py tests\test_past_dialog_cognition_prompt_boundaries.py -q
venv\Scripts\python.exe -m pytest tests\test_interaction_style_images.py tests\test_cognition_chain_connector_mapping.py -q
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_contracts.py tests\test_cognition_core_v2_integration.py tests\test_cognition_core_v2_prompt_budget_continuity.py -q
```

### Combined Regression

```powershell
venv\Scripts\python.exe -m pytest tests\test_cognition_resolver_loop.py tests\test_cognition_resolver_persona_graph.py tests\test_persona_supervisor2.py tests\test_self_cognition_integration.py tests\test_self_cognition_tracking.py -q
venv\Scripts\python.exe -m pytest tests\test_cognition_core_v2_alignment_gates.py tests\test_cognition_core_v2_failures.py tests\test_l2d_l3_surface_handoff.py -q
```

### Compile

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py src\kazusa_ai_chatbot\cognition_core_v2\contracts.py src\kazusa_ai_chatbot\cognition_core_v2\facade.py src\kazusa_ai_chatbot\cognition_core_v2\goal_cognition.py src\kazusa_ai_chatbot\cognition_core_v2\action_selection.py
```

### Live E2E And Trace Gates

Run through `character-test` and inspect with `debug-llm`/protected trace tools:

1. Shared memory:
   - insert one guarded globally shared nonce memory;
   - send an ordinary natural turn that semantically relates to it without
     naming a resolver or memory lookup;
   - prove cycle index zero, prewarm retrieval, merged typed memory evidence,
     cognition consumption, and no `rag_result.answer`;
   - grade the response for grounded use; model wording is a quality signal,
     while trace arrival is the blocking path contract.
2. Past dialog:
   - create a guarded Kazusa-authored prior dialog with full trace-backed
     residual, then reply naturally about her prior reasoning;
   - prove the residual appears in goal cognition and nowhere in appraisal,
     action planning, surface, dialog input, or visible output;
   - grade continuity without verbatim private-trace leakage.
3. Group engagement:
   - run eligible group self-cognition with a guarded group style image;
   - prove one group projection load, the exact context in goal/action
     planning, and no style-created topic or unsupported reason to speak;
   - compare one empty-guidance control case under the same observed scene.

Store raw/structured outputs under a task-specific `test_artifacts` directory
and add a concise Markdown review with correlation ids, stage names, evidence
handles, observed behavior, and pass/fail rationale.

## Independent Plan Review

Before approval, an independent reviewer checks:

- all three completed historical contracts are represented;
- the V2 adaptations preserve semantic ownership;
- private residual has exactly one LLM consumer;
- group guidance affects participation/action judgment without becoming
  evidence or final wording;
- prewarm latency and failure behavior are bounded;
- the broader short-horizon plan remains separate and has a clear rebaseline
  rule;
- verification catches deletion of each production edge.

Blocking findings require this draft to be updated and reviewed again.

## Independent Code Review

After all verification passes, an independent review subagent must inspect:

- the entire diff and current call graph;
- cycle-zero and later-cycle behavior;
- prewarm source exclusion, answer omission, merge, failure, and propagation;
- exact V2 input validation and prompt budget handling;
- private-field negative boundaries;
- group eligibility and no-read behavior;
- LLM call count and latency changes;
- test quality, including whether mocks prove the real caller and final
  consumer;
- documentation and overlap with the active short-horizon plan.

Any P0/P1 finding blocks sign-off. P2 findings are remediated or explicitly
accepted by the user with rationale. Record reviewer identity, findings,
remediation, and rerun evidence in `Execution Evidence`.

## Acceptance Criteria

- Normal eligible resolver cycle zero invokes
  `run_first_cycle_shared_memory_prewarm(...)` and joins it before V2 input
  construction.
- Confirmed shared `memory` rows become bounded typed V2 memory evidence.
- Prewarm excludes `user_memory_units`, leaves `rag_result.answer` empty,
  creates no resolver observation, and does not rerun after cycle zero.
- Empty/failing prewarm preserves the base RAG result and normal cognition.
- `past_dialog_cognition_context` is a distinct optional V2 field, capped at
  three dialogs and 1,800 characters.
- V2 goal cognition is its only LLM consumer; all forbidden boundaries are
  clean.
- Eligible group self-cognition loads one bounded
  `group_engagement_action_context`.
- The same group projection reaches goal cognition and action planning, while
  ineligible paths perform no group style read.
- Group guidance remains advisory and cannot create topic, fact, permission,
  relationship stance, or unsupported reason to speak.
- No new persistence, migration, model route, LLM stage, V1 compatibility
  graph, feature flag, or prompt-cap increase exists.
- Focused, combined, compile, static, and one-at-a-time live gates pass.
- Independent review has no unresolved P0/P1 findings.
- The broader short-horizon plan records or receives the required rebaseline
  before its future execution.
- User approves final evidence before lifecycle completion.

## Risks

| Risk | Mitigation | Blocking proof |
|---|---|---|
| Prewarm increases first-cycle latency | Start before independent state work; one attempt; preserve cache and empty fallback | concurrency test plus live trace timings |
| Scoped user memory leaks into unconditional lookup | reuse shared-memory-only helper and source filter | helper tests and static call inspection |
| Memory becomes final stance | keep typed evidence and empty public answer | prompt/trace evidence plus output review |
| Past private trace leaks | dedicated field, goal-only projection, forbidden greps | negative boundary tests and live trace |
| Optional context displaces current scene | fixed caps and stable supplemental reduction | worst-case prompt-budget test |
| Group style invents participation reason | prompt contract labels it advisory and requires observed context | counterfactual live review |
| Group style loads on every resolver cycle | cycle-zero load and state reuse | call-count test |
| Tests pass while caller is absent | assert invocation, mapped input, rendered consumer, and negative consumers | deletion-sensitive integration tests |
| Wider style plan duplicates this lane | explicit rebaseline/reuse requirement | plan comparison before sign-off |

## Execution Evidence

Execution started on 2026-07-30 after the user's explicit command to execute
this plan.

The parent self-review confirmed that every `Must Do` item maps to an
implementation and verification gate, the three exact V2 contracts are fixed,
the broader short-horizon plan remains deferred, and no placeholder decision
or compatibility path remains.

During execution the parent records:

- branch, starting HEAD, worktree state, and approved plan revision;
- pre-change topology and failing test output;
- production subagent identity and changed-file report;
- focused/combined/static/compile outputs;
- live artifact paths and correlation ids for all three cases;
- optional read-only `napcat` smoke result, if run;
- independent review identity, findings, remediation, and reruns;
- final diff summary, user sign-off, registry update, and archive destination.

Production implementation subagent:

- Huygens, agent `019fb260-470e-7b42-93ec-bf5bf81361be`, implemented only
  the approved production and subsystem-documentation surface. The parent
  retained ownership of tests, live evidence, integration, remediation, and
  lifecycle.

Stage 1 evidence, parent, 2026-07-30:

- Starting HEAD: `1bed4258`; initial worktree contained only this new plan and
  its registry row.
- Pre-change topology showed no production caller for
  `run_first_cycle_shared_memory_prewarm(...)` or
  `build_group_engagement_action_context(...)`. Past-dialog matches ended at
  producer/state plumbing and no V2 cognition match existed.
- Parent test files changed:
  `tests/test_persona_supervisor2_cognition_prewarm.py`,
  `tests/test_cognition_chain_connector_mapping.py`,
  `tests/test_cognition_core_v2_contracts.py`, and
  `tests/test_cognition_core_v2_integration.py`.
- Python compile of all four changed test files passed.
- Focused pre-implementation run completed with two expected baseline passes
  and five expected failures: absent prewarm invocation, absent private/group
  connector mappings, exact V2 input rejection, and absent final consumers.
- The group self-cognition fixture initially carried an invalid extra episode
  field. Parent removed it and reran the case; it now fails only because the
  group engagement loader is never invoked.
- Sign-off: parent, 2026-07-30. Next checkpoint: Stage 2 exact V2 contracts.

Stage 2 evidence, parent, 2026-07-30:

- `tests/test_cognition_core_v2_contracts.py`: 25 passed.
- The gate covers exact top-level input fields, the distinct 1,800-character
  past-dialog cap, exact group keys and types, bounded guideline/confidence
  text, and malformed input rejection before model calls.
- Sign-off: parent, 2026-07-30. Next checkpoint: Stage 3 cycle-zero prewarm.

Stage 3 evidence, parent, 2026-07-30:

- `tests/test_shared_memory_prewarm.py` and
  `tests/test_persona_supervisor2_cognition_prewarm.py`: 10 passed.
- The gate proves shared-only retrieval, no public answer, bounded failure
  fallback, cycle-zero task overlap and join, typed V2 evidence arrival,
  later-cycle exclusion, base-RAG preservation, and canonical validation
  before any retrieval or database-state side effect.
- Sign-off: parent, 2026-07-30. Next checkpoint: Stage 4 private past-dialog
  residual.

Stage 4 evidence, parent, 2026-07-30:

- Past-dialog context, reply, RAG, prompt-boundary, and V2 integration files:
  30 passed, 4 deselected.
- The gate proves trace-backed production, three-dialog/1,800-character caps,
  exact connector mapping, goal-cognition visibility, and exclusion from
  appraisal, workspace collapse, action planning, surface, and dialog.
- Sign-off: parent, 2026-07-30. Next checkpoint: Stage 5 eligible group
  engagement.

Stage 5 evidence, parent, 2026-07-30:

- Interaction-style image, connector mapping, and V2 integration files:
  59 passed, 4 deselected.
- The gate proves one eligible group self-cognition database projection, exact
  bounded mapping, same-value delivery to goal/action judgment, repeated-cycle
  reuse, and zero group-style reads for ordinary user turns.
- Sign-off: parent, 2026-07-30. Next checkpoint: Stage 6 complete
  deterministic verification.

Stage 6 evidence, parent, 2026-07-30:

- Focused V2 contracts/integration/prompt-budget batch: 70 passed,
  4 deselected.
- Resolver/persona/self-cognition regression batch: 158 passed.
- Alignment/failure/L3 handoff regression batch: 37 passed.
- The five modified production Python files passed `py_compile`.
- Static assertions passed with one prewarm connector call, one group loader
  connector call, zero V1 cognition imports, and zero private residual matches
  in appraisal/workspace/action/dialog consumers.
- `git diff --check` passed; status contains only the approved source, test,
  documentation, registry, and plan files. Git reported informational LF/CRLF
  normalization warnings.
- Sign-off: parent, 2026-07-30. Next checkpoint: Stage 7 one-at-a-time live E2E
  traces and readable reviews.

Stage 7 evidence, parent, 2026-07-30:

- Three guarded full-capture live selectors ran and were inspected one at a
  time against `_test_kazusa_live_llm`.
- Shared-memory trace `llmtrace_9b969577486d49909cad3c49bfd85ad9`
  proves one prewarm call, persistent-memory provenance, `answer=""`, typed
  goal evidence, and the grounded visible answer `雷阵雨。比晴天好。`.
- Past-dialog trace `llmtrace_3045c7cf64f449829f098db9dc1f99cb`
  proves exact V2 mapping, goal-only private marker visibility, no visible
  marker leakage, and coherent explanation of the prior caution.
- The first group comparison exposed an additional cutover regression:
  canonical self-cognition `content.semantic_text` lost priority to stale
  fallback `text`. A new failing connector assertion reproduced it; the
  connector now projects canonical semantic text first.
- Final group traces `llmtrace_13d812a339a2408eb1b23f210c3f2f75`
  and `llmtrace_299db7656ec34a8aa5cbf7133503726d` prove the observed rainy-day
  scene reaches goal/action in both cases, while only the style case receives
  one high-confidence advisory projection. Both cases remained silent and
  selected no action.
- Human-readable review:
  `test_artifacts/cognition_core_v2_p0_context_reconnection/live_e2e_review.md`.
- Raw JSON captures are linked from that review. Guarded memory, conversation,
  legacy trace-step, and style rows were cleaned by exact test scope after
  capture.
- Non-blocking observations retained in the review: targetless group goal
  wording sometimes says “current user,” and one past-dialog run recovered
  from an unrelated settled-relevance contract error.
- Sign-off: parent, 2026-07-30. Next checkpoint: Stage 8 independent code
  review.

Post-live remediation verification, parent, 2026-07-30:

- After correcting canonical self-cognition scene projection, the full focused
  P0 batch passed with 141 passed and 4 deselected.
- The combined resolver, persona, self-cognition, V2 alignment/failure, and L3
  handoff regression batch passed with 195 passed.
- All changed Python files passed `py_compile`.
- AST topology checks confirmed exactly one prewarm call, exactly one group
  engagement load, zero legacy past-dialog loader calls, and zero V1 cognition
  references in the production connector.
- `git diff --check` passed.
- The broader short-horizon global-state composition draft now records the
  post-P0 carrier and consumer baseline and requires its future shared style
  snapshot to replace, rather than duplicate, this direct loader.

Stage 8 evidence, parent, 2026-07-30:

- Independent reviewer: Dirac, agent
  `019fb28c-bea9-7201-b07c-dd4ad15734ef`; review-only, no edits.
- Review reported zero P0 findings, two P1 findings, and four P2 findings.
- P1 task-lifecycle finding remediated with structured cancellation and join
  of every started prewarm/group preparation task on state-preparation or
  join/merge failure. Two deletion-sensitive tests reproduce user-state and
  character-state failure independently and assert task cancellation.
- P1 group-reuse finding remediated by running the returned cycle-zero state
  through cycle one and proving one total database load plus unchanged
  guidance at the second V2 consumer.
- P2 contract/budget finding remediated with all wrong-type, cardinality,
  empty-item, guideline-length, confidence-length, and empty-guideline
  consistency cases; the goal worst-case fixture now uses five valid
  120-character guidelines and 80-character confidence; action planning has a
  deletion-sensitive test proving valid group guidance drops before the plan.
- P2 live-fixture finding remediated with failure-safe exact style cleanup,
  terminal manual self-cognition trace finalization, terminal normal-intake
  trace polling, direct goal/action consumer captures compatible with terminal
  trace redaction, and measured normal-chat elapsed time.
- P2 lifecycle/style findings remediated by recording both agent identities,
  updating execution authority/evidence, documenting eligibility-helper
  arguments/returns, and converting new live logging to project style.
- Final live shared-memory trace:
  `llmtrace_dcbd1dab2df349e4bb8c3629bb40043f`, terminal `succeeded`,
  65.134-second complete normal-chat elapsed time, artifact
  `test_artifacts/llm_traces/cognition_core_v2_p0_context_reconnection__shared_memory_prewarm__20260730T104943389591Z.json`.
- Final live past-dialog trace:
  `llmtrace_d0d1956b98b34bce910c702aaa636cc1`, terminal `succeeded`,
  artifact
  `test_artifacts/llm_traces/cognition_core_v2_p0_context_reconnection__past_dialog_goal_only__20260730T105420171621Z.json`.
- Final live group traces:
  `llmtrace_fadcfa4034aa48de8ad6189fb15c68bc` and
  `llmtrace_aef31d61049745eb93dcd2c578d330df`, both terminal `completed`,
  artifact
  `test_artifacts/llm_traces/cognition_core_v2_p0_context_reconnection__group_engagement_style_and_control__20260730T105627455597Z.json`.
- Updated human-readable review:
  `test_artifacts/cognition_core_v2_p0_context_reconnection/live_e2e_review.md`.
- Final focused P0 suite: 151 passed, 4 deselected.
- Final resolver/persona/self-cognition/V2/L3 regression suite: 195 passed.
- No independent-review P0, P1, or P2 finding remains unresolved. Stage 9
  remained open until the user's final approval recorded below.

Stage 9 evidence, parent, 2026-07-31:

- The user explicitly approved the final evidence and instructed closeout,
  cleanup, and commit.
- Lifecycle status changed from `in_progress` to `completed`; all nine stages
  are signed and the plan moved to
  `development_plans/archive/completed/bugfix/cognition_core_v2_p0_context_reconnection_bugfix_plan.md`.
- Registry ownership moved from Active Bugfix Plans to Completed Bugfix
  Records. The dependent short-horizon global-state draft now references this
  completed archive record.
- Cleanup removed 11 superseded or failed task-specific ignored raw captures.
  The three final terminal JSON captures and the human-readable live review
  remain as closeout evidence.
- No new production behavior, contract, or scope was added during closeout.
