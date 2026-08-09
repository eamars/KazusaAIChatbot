# cognition branch intent guidance plan

## Summary

- Goal: make every Cognition V2 goal branch explicit about the semantic
  responsibility it owns while preserving neutral interpretation of the
  current user and current event.
- Status: completed
- Scope boundary: the static branch-definition contract, generic goal-bid
  prompt context, Cognition V2 documentation, deterministic verification,
  and the directly affected regression suites. Live prompt-quality evidence is
  explicitly deferred because no provenance-qualified current-schema packets
  are available in this workspace.
- Change direction: add bounded branch-owned intent guidance for all fourteen
  registered branches. Project it into exactly the thirteen nonordinary
  generic initial and repair prompts. The guidance directs attention toward
  the branch's purpose; it does not permanently label the user, event, or
  branch as good, bad, positive, or negative.
- Acceptance state: deterministic implementation scope completed under the
  user-authorized live-quality deferral; real-model prompt quality and
  production cutover remain follow-up requirements.

## Scope And Change Direction

Each Cognition V2 branch already owns a distinct goal kind, dependency set, and
action tendency. Goal cognition receives the goal kind and action tendencies,
then produces the existing complete ActionBidV2 contract. This plan makes the
branch's semantic intent explicit in the generic goal prompt without adding a
new bid field or changing downstream ownership.

The target flow is:

~~~text
selected BranchDefinition
  -> bounded branch intent guidance in the generic goal prompt
  -> evidence-first branch judgment by the goal-cognition LLM
  -> existing ActionBidV2 fields
  -> existing eligibility and workspace/authoritative collapse
  -> existing action, surface, and state paths
~~~

The guidance is a fixed semantic focus, not a fixed emotional or moral
direction. For example, autonomy_boundary is permanently attentive to
autonomy and owned boundaries, but it does not permanently assume hostile user
intent. self_improvement is permanently attentive to learning and correction,
but it does not force optimism or growth.

The current event, character identity, role direction, explicit boundaries,
relationship state, and supplied evidence remain authoritative. The branch
guidance may help the model distinguish evidence-supported interpretations; it
cannot create a motive, fact, goal, emotion, permission, or action.

## Confirmed Decisions

- The branch's semantic responsibility is fixed in the registry.
- Positive or negative valence is not fixed in the registry or prompt.
- Runtime prompts do not use good-intent, bad-intent, think-positive, or
  think-negative labels.
- The prompt asks the model to inspect the branch focus after checking current
  evidence and role ownership.
- Unsupported or ambiguous interpretations remain neutral.
- Branches may still produce similar bids when the same evidence supports the
  same outcome. Forced disagreement is not required.
- The ordinary-response branch remains the neutral contextual reference and
  keeps its existing prompt contract.
- Typed required-selection paths keep their existing direct-choice contract and
  do not receive branch-intent guidance.
- Generic nonordinary branches receive their own bounded guidance in both
  initial and repair prompt payloads.
- All fourteen registry definitions carry an explicit guidance value. The
  ordinary-response value is registry/documentation-only, and exactly thirteen
  nonordinary generic branches receive prompt projection. Typed
  required-selection paths receive no projection.
- The existing ActionBidV2 schema, validator, collapse policy, action planning,
  surface planning, persistence, and delivery paths remain unchanged.
- Branch identity is the registry branch_id. No ordinal b1/b2/b12 mapping is
  introduced.

## Evidence Decision And Plan Amendment

The independent GPT-5.6 Sol review confirmed that the manifest-gated live test
must consume provenance-verified real packets. The sibling workspace contains
historical traces, but its branch-specific records are either synthetic,
older-schema, or missing the required current goal-cognition packet fields; it
does not contain the required twenty-six current-schema packets. Reusing or
labeling those records as approved live cases would create false quality
evidence.

The user-authorized fallback is therefore to remove the unexecutable
manifest-gated live test and keep the deterministic branch/prompt mechanics
coverage. No synthetic manifest, raw live trace, or human quality review is
created. The implementation may be closed for deterministic contract and
regression scope, while real-model prompt quality remains an explicit residual
risk and requires a follow-up plan with provenance-qualified packets before a
real-model quality claim or production cutover.

## Branch Intent Guidance Map

The following table is the complete fixed branch contract for the default
registry. The precise intent is the semantic job owned by the branch. The
runtime guidance tells the model what to inspect; it does not predetermine the
user's motive or the final conclusion.

| Registry branch | goal_kind | Action tendencies | Precise branch intent | Runtime guidance (literal Simplified Chinese value) |
| --- | --- | --- | --- | --- |
| ordinary_response | ordinary_response | respond | Serve as the neutral contextual baseline for the current event, retaining existing relational-willingness ownership where applicable without importing another branch's specialized focus. | 为当前事件提供中性的上下文基线；在适用时保留现有 relational_willingness 的归属，不引入其他分支的专门焦点。 |
| relationship_connection | relationship_connection | connect, reciprocate | Assess whether and how voluntary, context-appropriate reciprocal engagement should build, maintain, recalibrate, or repair interpersonal connection. | 评估是否以及如何通过自愿且符合当前情境的互惠参与来建立、维持、调整或修复人际连接。 |
| bond_protection | bond_protection | protect, verify | Preserve an important relationship bond when the current event presents an evidenced threat or damage. | 评估当前事件是否对重要关系纽带造成有证据支持的威胁或损害，并考虑相称的保护或修复。 |
| trust_verification | trust_verification | verify, ask | Resolve whether trust is warranted by checking uncertainty and asking for clarification when needed. | 评估当前证据是否支持信任、保留信任或需要核实；不把不确定性直接解释为背叛。 |
| autonomy_boundary | autonomy_boundary | set_boundary, refuse | Protect the character's autonomy and explicitly owned boundaries when the current event imposes a grounded cost. | 评估当前事件是否对角色自身的自主权、意愿或明确边界造成有证据支持的压力或代价；在有依据时保护边界，不假定恶意。 |
| safety_coping | safety | protect, cope | Manage a grounded threat or strain through proportionate protection and coping. | 评估当前事件是否存在有证据支持的威胁或压力，并考虑相称的保护或应对；不凭空升级恐惧。 |
| obstruction_strategy | obstruction_resolution | confront, repair | Resolve an obstacle that is preventing progress toward a current goal. | 评估当前事件是否阻碍当前目标的进展，并考虑相称的解决、对抗或修复。 |
| loss_recovery | loss_recovery | recover, grieve | Process a grounded loss and pursue recovery or an appropriate grieving response. | 评估当前事件是否构成有依据的损失，并考虑恢复、适应或适当的哀悼；不强迫悲伤。 |
| moral_repair | moral_repair | repair, apologize | Assess whether the current character bears evidence-supported responsibility for harm and, if so, pursue proportionate repair or apology. | 评估当前角色是否对伤害负有有证据支持的责任；如有，考虑相称的修复或道歉。 |
| social_care | social_care | support, care | Respond to the grounded needs of people affected by the current event through support or care. | 评估受当前事件影响的人是否有有依据的需要，并考虑相称的支持或照护；不强迫温柔。 |
| reciprocal_response | reciprocity | reciprocate, respond | Determine the current character's evidence-grounded proportionate response to another actor's action; reciprocity does not imply compliance or matched valence. | 确定当前角色对另一方行为的有证据支持且相称的回应；互惠不等于服从，也不要求匹配情绪价性。 |
| epistemic_exploration | epistemic_exploration | explore, ask | Reduce uncertainty and obtain understanding through exploration, questions, or comparison. | 通过探索、提问或比较，减少当前有依据的不确定性并增进理解；区分求知与无依据的断言。 |
| meaning_reconstruction | meaning_reconstruction | reconstruct_meaning, remember | Rebuild a coherent meaning or interpretation after an existential or narrative disruption. | 在当前事件造成有依据的叙事或存在性中断后，评估如何重建连贯意义；不强迫乐观。 |
| self_improvement | self_improvement | learn, improve | Identify an evidence-grounded opportunity for the current character to learn, correct an error, or develop capability without presuming deficiency, optimism, or success. | 评估当前角色是否有有证据支持的学习、纠错或能力发展机会；不预设缺陷、乐观或成功。 |

The runtime guidance column is the literal registry string projected into the
Chinese generic prompt. It is not translated, paraphrased, or assigned a
polarity at runtime. The ordinary row remains documentation and registry
context only; the other thirteen rows are the generic prompt projection set.

## Mandatory Skills

- development-plan: governs this plan lifecycle, approval boundary, change
  surface, and acceptance contract.
- local-llm-architecture: governs prompt ownership, bounded local-model
  context, semantic responsibility, and latency.
- debug-llm: governs prompt-quality verification and human inspection of
  live-model outputs.
- py-style: applies to every Python production or test change.
- cjk-safety: applies when Python prompt strings containing CJK text are
  changed.
- test-style-and-execution: governs deterministic and live LLM test design
  and execution.
- no-prepost-user-input: preserves LLM ownership of user-intent
  interpretation and prevents deterministic semantic overrides.

## Mandatory Rules

- Use venv\Scripts\python for Python commands.
- Keep branch intent guidance as a short semantic descriptor. Do not add a
  numeric valence, emotion score, moral score, or good/bad enum.
- The LLM owns the meaning of the current event and the branch's resulting
  intention. Deterministic code owns field shape, length, handles, limits,
  permissions, execution, and persistence.
- The prompt must state that branch guidance is an attention focus rather than
  a conclusion about the user or any other actor.
- The runtime guidance strings are the fixed Simplified-Chinese literals in the
  map. Do not introduce an English standing descriptor into the Chinese
  generic prompt or translate the value differently per invocation.
- Evidence and role checks must occur before the branch guidance influences
  interpretation.
- The guidance must not invent facts, goals, evidence handles, target roles,
  permissions, confidence, emotions, or action feasibility.
- The guidance must not turn autonomy_boundary into general suspicion or moral
  policing. Its concern is current-character-owned agency and boundaries.
- The guidance must not turn self_improvement into forced optimism. It needs a
  concrete current-character learning, correction, or improvement basis.
- The guidance must not turn trust_verification into assumed betrayal,
  safety_coping into fear escalation, moral_repair into blame, or
  relationship_connection into forced warmth or consent.
- When the supplied evidence does not support a specialized branch focus, the
  goal LLM still returns a complete existing bid whose intention, desired
  outcome, and reason state that this event provides no basis to advance that
  specialized responsibility. It cites only relevant supplied evidence and
  does not borrow an ordinary-response motive. Existing workspace collapse
  decides whether that bid is suppressed from the final decision; deterministic
  code does not rewrite its semantic fields to neutralize it.
- Active-goal descriptions or other persistent context may contain
  evidence-grounded valence. Preserve that contextual state as supplied and
  do not reinterpret it as static branch polarity.
- Keep the existing canonical JSON parser, validators, repair owner, and
  cumulative branch-attempt policy.
- Do not add an LLM call, retry class, model route, sampling mode, random
  assignment, per-turn intent allocator, or ordinal bid label.
- Do not add compatibility aliases, a second bid schema, deterministic
  keyword routing, or post-processing that changes a semantic decision.
- Do not modify user-facing wording, persistence, consolidation, scheduler,
  adapter delivery, action authorization, or resolver execution.

## Must Do

1. Add the exact bounded branch-intent contract:
   - define `MAX_BRANCH_INTENT_GUIDANCE_CHARS = 240`;
   - add `BranchDefinition.branch_intent_guidance: str = ""`;
   - permit the empty default only for a custom definition intentionally using
     neutral, no-specialized-focus behavior;
   - require every default registry value to be nonempty and at most 240
     characters;
   - deterministically reject supplied non-string, whitespace-only, or
     oversized values, while retaining the omitted custom-definition default.
2. Populate branch_intent_guidance explicitly for all fourteen default branch
   definitions using the fixed map in this plan.
3. Preserve every existing branch_id, dependency, dependency option,
   action_tendencies value, goal_kind, required flag, registry order, and
   MAX_GOAL_BRANCHES value.
4. Project the guidance under the exact `branch_intent_guidance` key into the
   branch object of exactly the thirteen nonordinary generic goal-cognition
   prompt payloads.
5. Reuse the same literal guidance and key in the thirteen nonordinary generic
   repair payloads.
6. Add concise generic prompt instructions that:
   - identify branch guidance as a semantic attention focus;
   - require current evidence, identity, role, and boundary checks first;
   - require neutral handling when the evidence does not support the focus;
   - require the branch's intention, desired_outcome, and reason to remain
     grounded in the branch's own responsibility;
   - prohibit motive verdicts, emotional escalation, and invented goals.
7. Keep the ordinary-response system prompt and payload behavior unchanged.
8. Keep typed required-selection system prompts and payloads unchanged, with
   no branch guidance projection.
9. Keep branch guidance out of ActionBidV2, workspace state, action requests,
   surface output, persistence, and visible dialog.
10. Add deterministic contract tests for:
     - the exact fourteen-row guidance map;
     - `MAX_BRANCH_INTENT_GUIDANCE_CHARS`, bounded guidance shape, invalid
       value rejection, and neutral custom-definition behavior;
     - preservation of branch dependencies, action tendencies, goal kinds, and
       registry order;
     - generic prompt projection for exactly the thirteen nonordinary branches;
     - ordinary-response omission;
     - required-selection omission;
     - identical guidance projection in initial and repair payloads;
     - preservation of guidance through preliminary and final selected
       definitions, including every `dataclasses.replace(...)` dependency-
       option path;
     - exact existing ActionBidV2 output fields and validation behavior;
     - prompt rendering, literal Simplified-Chinese values, and aggregate
       budget behavior.
11. Add cardinality coverage for these exact production-selected sets, counting
     ordinary_response in every cardinality:
     - one: `[ordinary_response]`;
     - two: `[ordinary_response, relationship_connection]`;
     - three: `[ordinary_response, relationship_connection, bond_protection]`;
     - four: `[ordinary_response, relationship_connection, bond_protection,
       trust_verification]`;
     - twelve: `BRANCH_REGISTRY_ORDER[:12]`, whose row 12 is explicitly
       `epistemic_exploration`;
     - fourteen: the complete `BRANCH_REGISTRY_ORDER`.
     Assert exact guidance by `branch_id` under reversed task-completion order.
     The tests must contain no `b12` lookup; existing `bN` handles remain
     workspace-collapse-local and unrelated to guidance identity.
12. Add focused prompt-capture coverage proving that the guidance reaches the
     model-facing generic branch context and does not become a generated output
     field or deterministic semantic override.
13. Run contract-valid synthetic generic mechanics cases one branch at a time
    for the thirteen nonordinary projection rows. Use these cases for prompt,
    payload, parser, repair, and schema mechanics. They are explicitly
    mechanics evidence and are not presented as real-model quality evidence.
14. Review the synthetic outputs for:
    - branch-owned intention and desired outcome;
    - evidence-cited reason;
    - neutral handling when the branch focus lacks support;
    - absence of good/bad motive verdicts;
    - absence of forced emotion, role reversal, or personality claims;
    - unchanged bid schema and bounded attempts.
15. Verify that existing workspace and authoritative collapse receive the same
     bid contract and that no collapse-code change is needed for this plan.
16. Update the Cognition Core V2 README with the branch-intent ownership rule,
     the fourteen-row semantic map, and the ordinary/required-selection
     exclusions.
17. Record the live-quality deferral and provenance gap in this plan. Do not
    create a synthetic manifest or represent historical older-schema output as
    current prompt-quality evidence.
18. Run the complete non-live Cognition Core V2 test set and the directly
    affected service cognition graph tests after the focused tests. At minimum
    include `tests/test_service_cognition_graph.py` and the relevant control-
    console cognition-graph coverage identified by the implementation agent.
19. Prove the existing resource envelope remains unchanged: every generic
    initial and repair render fits the 36,000-character aggregate cap while a
    production-sized prompt fixture retains its minimum evidence-text floor;
    ordinary and required-selection prompt/payload key sets remain unchanged;
    route, completion budget, maximum concurrency, one initial call per
    selected branch, and the cumulative three-attempt-per-branch policy remain
    unchanged. Record wall-clock latency observationally; treat structural
    call count and concurrency as acceptance gates unless an explicit baseline
    is added.

## Deferred

- Dynamic positive-versus-negative intent assignment across parallel paths.
- Random, rotating, or invocation-seeded branch perspectives.
- Permanent positive or negative valence attached to any branch.
- A new ActionBidV2 field for intent, polarity, motive, or branch focus.
- Changes to workspace or authoritative collapse weights, ranking, or schema.
- Additional challenger, counterpressure, evaluator, or verification calls.
- Required-selection branch variation.
- Changes to branch activation criteria or goal creation.
- Changes to action planning, authorization, resolver selection, L3 wording,
  dialog generation, persistence, consolidation, scheduling, or delivery.
- Adding, removing, or renaming registry branches.
- Temperature, sampling, model-route, or retry-policy changes.

## Contracts And Data Shapes

The static branch contract gains one field:

~~~text
MAX_BRANCH_INTENT_GUIDANCE_CHARS = 240
BranchDefinition.branch_intent_guidance: str = ""
~~~

The field is bounded semantic configuration. The default registry supplies a
nonempty value for every row in the map. A custom definition may omit the field
and retain the empty neutral default, which omits the descriptor from generic
prompt projection. Deterministic validation rejects supplied non-string,
whitespace-only, and oversized values; the registry constructor rejects an
empty value for a default row. The LLM owns whether the current evidence
supports the focus.

The generic prompt payload becomes:

~~~json
{
  "branch": {
    "goal_kind": "autonomy_boundary",
    "action_tendencies": ["set_boundary", "refuse"],
    "branch_intent_guidance": "评估当前事件是否对角色自身的自主权、意愿或明确边界造成有证据支持的压力或代价；在有依据时保护边界，不假定恶意。"
  }
}
~~~

This payload shape applies only to nonordinary generic initial and repair
prompts. The ordinary-response and typed required-selection payload key sets
remain unchanged and contain no `branch_intent_guidance` key.

The exact ActionBidV2 output remains unchanged. The internal guidance is
prompt context only and is not copied into any bid, state, action, surface,
memory, or visible output field.

## Runtime And Resource Constraints

- Production branch selection remains bounded by the existing
  MAX_GOAL_BRANCHES value of fourteen.
- A selected branch receives one short guidance string; no extra model call or
  retry is introduced.
- Every generic initial and repair render remains inside the existing 36,000-
  character aggregate cap. The largest production guidance fixture must still
  preserve the existing minimum evidence-text floor.
- Existing route, completion budget, maximum concurrency, and initial-call
  policy remain unchanged: one initial call per selected branch.
- Existing initial and repair attempt limits remain unchanged, including at
  most three cumulative LLM attempts per branch.
- Generic prompt aggregate-budget fitting remains the owner of payload sizing.
- Ordinary and required-selection paths retain their current prompt sizes,
  payload key sets, and contracts.
- Guidance projection is keyed by the selected BranchDefinition branch_id.
  It does not depend on b1, b2, b12, task start order, or executor completion
  order.
- Wall-clock latency is recorded as an observational quality metric. Structural
  call count, retry count, and concurrency remain acceptance gates; no latency
  target is introduced without a separately recorded baseline.

## Target State

The branch registry owns semantic responsibility:

~~~text
branch_id
  -> goal_kind and action_tendencies
  -> branch_intent_guidance
  -> thirteen nonordinary generic goal-cognition prompts
  -> existing ActionBidV2
  -> existing collapse and downstream paths
~~~

The prompt procedure is:

1. Resolve the current character identity and structured role direction.
2. Read the current event and supplied evidence.
3. Check boundaries, relationship state, and applicable active goal
   provenance.
4. For a nonordinary generic branch, use the literal
   branch_intent_guidance to decide what branch-specific question to inspect.
   Omit it for ordinary-response and required-selection paths.
5. Return the branch's own evidence-grounded intention, desired outcome,
   reason, and remaining required fields.
6. When the branch focus has no support, return a complete existing bid whose
   semantic fields state that no specialized progress is supported, cite only
   relevant supplied evidence, and leave suppression to existing workspace
   collapse.

The guidance changes attention, not authority. Current episode meaning,
explicit refusal, role direction, and evidence remain stronger than the static
branch descriptor.

## Change Surface

### Modify

- src/kazusa_ai_chatbot/cognition_core_v2/contracts.py
  - Add `MAX_BRANCH_INTENT_GUIDANCE_CHARS`, the bounded
    BranchDefinition.branch_intent_guidance field, and its deterministic
    validation contract.
- src/kazusa_ai_chatbot/cognition_core_v2/branch_activation.py
  - Extend the branch constructor and populate the complete fourteen-row
    semantic guidance map without changing selection behavior.
- src/kazusa_ai_chatbot/cognition_core_v2/goal_cognition.py
  - Add the Chinese generic prompt instruction and project the literal branch
    guidance into exactly thirteen nonordinary generic initial and repair
    payloads only.
- src/kazusa_ai_chatbot/cognition_core_v2/README.md
  - Document branch-intent ownership, the fixed semantic map, and excluded
    prompt paths.
- tests/test_cognition_core_v2_dependencies.py
  - Verify the registry contract, map, and preservation of dependency and
    ordering behavior.
- tests/test_cognition_core_v2_goal_branch_creation.py
  - Verify selected definitions retain their mapped guidance.
- tests/test_cognition_core_v2_prompt_contract_guidance.py
  - Verify prompt text, payload projection, omission boundaries, payload
    ordering, and repair behavior.
- tests/test_cognition_core_v2_contracts.py
  - Verify the BranchDefinition field and bounded validation contract.

The implementation creates no live-quality artifacts. The absence of a
provenance-qualified packet manifest is recorded as residual risk in this
plan; future live evidence must be created by a separate approved follow-up.

The implementation creates these plan-scoped records:

- development_plans/active/short_term/cognition_parallel_neutral_weak_tilt_plan.md
  - This executable plan and its verification record.
- development_plans/README.md
  - Lifecycle registry status when the plan transitions to execution.

### Create

- tests/test_cognition_core_v2_branch_intent_guidance.py
  - Cover the full map, custom neutral definitions, all selected cardinalities,
    output-shape preservation, and branch-owned prompt projections.

### Verify without planned edits

- tests/test_service_cognition_graph.py
  - Verify the directly affected service graph contract after focused tests.
- tests/control_console_e2e/test_cognition_graph_e2e.py
  - Verify the directly affected control-console cognition-graph boundary when
    its existing test contract covers the changed path.

### Keep

- src/kazusa_ai_chatbot/cognition_core_v2/facade.py
- src/kazusa_ai_chatbot/cognition_core_v2/workspace.py
- src/kazusa_ai_chatbot/cognition_core_v2/action_selection.py
- src/kazusa_ai_chatbot/cognition_core_v2/action_authorization.py
- src/kazusa_ai_chatbot/cognition_core_v2/surface.py
- src/kazusa_ai_chatbot/cognition_core_v2/parallel_executor.py
- src/kazusa_ai_chatbot/cognition_core_v2/dependency_graph.py
- Workspace-collapse-local bN handles and their existing mapping
- ActionBidV2 and all downstream state, action, surface, persistence, and
  delivery contracts

### Delete

- tests/test_cognition_core_v2_branch_intent_guidance_live_llm.py
  - Remove the manifest-gated live test because its required provenance-
    qualified packet set is unavailable; deterministic mechanics coverage is
    retained in the companion test.

## Agent Autonomy Boundaries

The implementation agent may choose helper-function names, local validation
placement, test fixtures, and exact patch order within the listed files.

The implementation agent must preserve the fourteen-row map, the fixed
semantic-only direction, neutral fallback, ordinary and required-selection
omission rules, output schema, branch selection, attempt limits, prompt
ownership, and downstream boundaries.

The implementation agent must request a plan amendment before changing
collapse, adding a bid field, introducing dynamic or random polarity,
changing required-selection behavior, adding an LLM call, changing a route,
or modifying any file outside the listed change surface.

## Verification

### Deterministic verification

- Run the focused branch-contract, goal-branch, prompt-contract, dependency,
  and output-schema tests with venv\Scripts\python.
- Assert all fourteen default definitions have the exact guidance map.
- Assert `MAX_BRANCH_INTENT_GUIDANCE_CHARS == 240`; reject non-string,
  whitespace-only, and oversized supplied values; and allow only the omitted
  custom-definition default to remain empty.
- Assert dependencies, dependency options, action tendencies, goal kinds,
  required status, registry order, and MAX_GOAL_BRANCHES are unchanged.
- Capture generic initial and repair payloads and verify the selected branch's
  literal `branch_intent_guidance` appears exactly in the branch context for
  each of the thirteen nonordinary branches.
- Verify ordinary-response and required-selection payloads omit guidance.
- Verify no guidance key or semantic descriptor appears in ActionBidV2.
- Render all thirteen nonordinary guidance rows through the prompt budget
  fitter, including a production-sized prompt fixture with the minimum
  evidence-text floor preserved and the 36,000-character aggregate cap intact.
- Exercise the exact one, two, three, four, twelve, and fourteen branch sets
  listed above, including the row-12 `epistemic_exploration` assertion.
- Verify reversed branch completion order does not alter `branch_id` mapping,
  and that no test or implementation performs a `b12` lookup.
- Verify both preliminary and final branch selection preserve guidance through
  every dependency-option `dataclasses.replace(...)` path.
- Compare ordinary and typed required-selection prompt/payload key sets with
  their pre-change contracts.
- Assert the existing route, completion budget, maximum concurrency, initial
  call count, and cumulative three-attempt limits are unchanged.

### Synthetic mechanics verification

- Use contract-valid synthetic cognition inputs with no production state or
  persistence side effects.
- Run one generic branch case at a time for each of the thirteen projection rows
  and inspect the protected prompt, repair payload, and returned bid.
- Verify the model treats guidance as a focus, cites supplied evidence, keeps
  role direction intact, and returns the existing exact output contract.
- Include unsupported-focus cases for autonomy_boundary,
  trust_verification, moral_repair, epistemic_exploration, and
  self_improvement; require neutral handling rather than invented motive or
  emotion.
- Inspect generic repair cases to confirm the same branch guidance remains
  present without changing the repair contract.
- Inspect collapse input for representative multi-branch synthetic cases and
  confirm it receives existing bid fields only.
- Keep this section as mechanical evidence; synthetic output alone cannot close
  prompt-quality acceptance.

### Live quality verification deferral

- The live quality gate is deferred because the workspace lacks a
  provenance-qualified current-schema packet manifest covering the required
  supported and unsupported cases.
- Synthetic mechanics tests remain separate and do not establish real-model
  behavior. No live output, trace, or quality review is claimed by this plan.
- A future approved follow-up must source and inspect one real case at a time
  before making a real-model quality or production-cutover claim.

### Broader regression verification

- After focused deterministic checks, run the complete non-live Cognition Core
  V2 test set with `venv\Scripts\python`.
- Run the directly affected service cognition graph coverage, including
  `tests/test_service_cognition_graph.py` and the relevant control-console
  cognition-graph tests.
- Record the live-quality deferral; no branch-intent live test is executed in
  this plan because its provenance input contract is intentionally absent.

## Acceptance Criteria

The rewrite is accepted for implementation when:

1. All fourteen registered branches have explicit semantic intent guidance.
2. The guidance states branch responsibility without permanently assigning
   positive or negative user motive.
3. autonomy_boundary is protective and evidence-aware rather than hostile or
   moralizing.
4. self_improvement is learning- and correction-aware rather than forced
   optimism.
5. Every other branch follows the same fixed-focus, neutral-valence rule.
6. Exactly thirteen nonordinary generic initial and repair prompts receive the
   correct literal `branch_intent_guidance` value; ordinary and
   required-selection prompts do not.
7. The field contract is bounded at 240 characters, validates invalid values,
   and preserves neutral custom-definition defaults.
8. Ordinary-response and required-selection prompt contracts remain unchanged.
9. ActionBidV2 and all downstream contracts remain unchanged.
10. No additional model calls, retries, routes, randomness, or persistence are
    introduced; the existing route, budget, concurrency, and three-attempt
    limits remain intact.
11. Deterministic tests cover all fourteen rows, exact selected cardinalities
    through fourteen branches, row-12 `epistemic_exploration`, reversed
    completion order, and dependency replacement preservation.
12. Every generic initial and repair render fits the 36,000-character cap while
    preserving the production evidence floor.
13. The live-quality evidence gap is explicitly recorded; no synthetic or
    older-schema artifact is promoted as real-model evidence, and the removed
    test no longer fails on an unavailable manifest.
14. The complete non-live Cognition Core V2 suite and directly affected service
    cognition graph tests pass.
15. The workspace and authoritative collapse paths remain unchanged and
    continue to consume the existing bid contract, suppressing unsupported
    specialized bids through their existing semantic decision.

## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Branch-definition contract | bigbang | Add the single canonical guidance field and map. |
| Generic goal prompt | bigbang | Use the branch-owned guidance in the existing generic prompt path. |
| Ordinary and required-selection prompts | compatible by preservation | Keep their existing prompt and payload behavior unchanged. |
| ActionBidV2 and downstream paths | compatible by preservation | Keep the existing schema and consumers unchanged. |
| Tests and documentation | bigbang | Replace expectations with the rewritten branch-intent contract. |

The deterministic implementation scope closes after focused and broader
non-live verification and explicit user approval. Real-model quality and
production cutover remain deferred until a future plan supplies provenance-
qualified live packets and human inspection. The runtime does not retain a
parallel old/new prompt path.

## Failure Modes And Controls

| Failure mode | Control |
|---|---|
| Fixed focus becomes a permanent motive verdict | Prompt states that guidance is attention only and current evidence is authoritative. |
| autonomy_boundary becomes hostile or moralizing | Require current-character-owned pressure, agency loss, or boundary cost. |
| self_improvement becomes forced positivity | Require concrete current-character learning, correction, or improvement evidence. |
| Static descriptor anchors a weaker local model too strongly | Record this as residual risk until a future plan supplies supported and unsupported live cases for every nonordinary branch and human inspection. |
| trust_verification assumes betrayal | Require evidence-supported uncertainty or trust conflict. |
| relationship_connection forces intimacy or consent | Require grounded connection evidence and preserve relationship-state validation. |
| safety_coping escalates fear | Require current threat or strain evidence and proportionate coping. |
| Branch guidance duplicates action tendencies without useful distinction | Keep each descriptor short, branch-specific, and directed at collapse-visible semantic fields. |
| Branches remain indistinguishable downstream | Inspect intention, desired_outcome, reason, collapse input, and admitted bids; do not change collapse policy under this plan. |
| Ordinary branch changes unintentionally | Assert ordinary prompt and payload omission and retain its existing contract. |
| Required-selection semantics change | Assert required-selection omission and exact direct-choice fields. |
| Guidance leaks into visible or persistent surfaces | Keep it outside ActionBidV2 and assert exact output/state shapes. |
| Stale goal context overrides current event | Require current-event and role checks before branch guidance. |
| Guidance exceeds local-model prompt budget | Use a bounded field and existing aggregate prompt fitting. |
| Executor completion order changes branch meaning | Key projection by BranchDefinition.branch_id and test reordered completion. |
| Generic custom branch has no guidance | Use the neutral-compatible default and omit the optional descriptor. |
| Invalid guidance reaches runtime | Validate type, whitespace, and 240-character bound at the canonical branch-definition owner. |
| Prompt guidance displaces useful evidence near the cap | Fit against the existing aggregate cap and assert the production evidence-text floor with the largest guidance fixture. |
| English or paraphrased guidance changes the intended focus | Store and assert the exact Simplified-Chinese literal values from the map. |
| Twelve-branch selection maps by ordinal or b12 handle | Assert registry row 12 is epistemic_exploration, map by branch_id, and forbid b12 lookups. |
| Synthetic cases pass while real behavior drifts | Keep synthetic mechanics separate; the deferred live-quality evidence is an explicit pre-cutover requirement for the follow-up plan. |
| Added context changes resource usage | Keep route, budget, concurrency, call, retry, and persistence gates unchanged; record latency observationally. |
| Evidence-grounded valence in active goals is mistaken for static polarity | Preserve active-goal descriptions as contextual evidence and keep the registry guidance polarity-free. |
| Verification passes only because of malformed or repaired output | Require valid initial or bounded repair output and record dispositions separately. |

## Progress Checklist

- [x] Branch semantic focus versus fixed valence confirmed.
- [x] Complete fourteen-branch intent map written.
- [x] Ordinary and required-selection boundaries fixed.
- [x] No ordinal b1/b2/b12 assignment included.
- [x] Exact field, language, cardinality, resource, live-evidence, and regression
  contracts incorporated from independent review.
- [x] User approves this rewritten plan.
- [x] Implementation authorization received.
- [x] Contract and registry changes implemented.
- [x] Prompt projection and repair behavior implemented.
- [x] Deterministic verification completed.
- [x] Synthetic mechanics verification completed and inspected.
- [x] Live-quality deferral, provenance gap, and residual risk recorded.
- [x] Manifest-gated live test removed; no synthetic quality evidence created.
- [x] Broader non-live and service cognition graph regression completed.
- [x] Documentation updated.
- [x] Deterministic implementation acceptance completed; real-model quality and
  production cutover remain deferred.

## Implementation Verification Record

- Focused deterministic contract, prompt, branch-selection, and dependency
  checks: 108 passed.
- Complete non-live Cognition Core V2 and directly affected cognition-graph
  coverage: 560 passed, 2 expected trace-inventory skips, and 259 live-marked
  tests deselected.
- The independent reviewer confirmed that the sibling historical artifacts do
  not provide the required provenance-qualified current-schema packet set.
- The manifest-gated live quality module was removed under the user-authorized
  fallback. No synthetic case, raw live trace, or human quality review is
  presented as real-model evidence.
- Deterministic implementation acceptance is complete; real-model prompt
  quality and production cutover are explicit follow-up requirements.
