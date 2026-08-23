# Cognition V3 Full-Chain Prompt Language Map

## Boundary

The canonical full cognition chain is the four-call A1 -> A2 -> G -> P flow
owned by `cognition_core_v3.facade.run_cognition(...)`. All four calls use the
single `COGNITION_V3_CHAIN_LLM` route. Each call receives one static system
instruction and one dynamic JSON human packet.

This map defines the language contract for that route:

- ordinary instructions use native Simplified Chinese;
- free-form semantic output values use Simplified Chinese;
- JSON keys, enum values, stage names, action/capability names, code, URLs, and
  quoted source text remain exact;
- dynamic observations, evidence, memories, and state retain their source-owned
  language rather than being deterministically translated.

## Runtime Prompt Map

| Stage | Semantic owner | Static system owner | Dynamic guidance owner | Human-packet authority lanes | Exact output owner |
| --- | --- | --- | --- | --- | --- |
| A1 | World-facing appraisal | `facade._STAGE_SYSTEM_PROMPTS["A1"]` | `prompt.A1_QUESTION_GUIDANCE` | `current_observation`, `direct_facts`, `continuation_state` | `output_contract` fixed A1 family slots |
| A2 | Relationship, moral, and existential character judgment | `facade._STAGE_SYSTEM_PROMPTS["A2"]` | `prompt.A2_QUESTION_GUIDANCE` | accepted A1 meaning, current/direct facts, participant continuity, conditional character context, continuation state | `output_contract` fixed A2 family slots |
| G | Active-character goal, relationship willingness, and first-person inner stance | `facade._STAGE_SYSTEM_PROMPTS["G"]` | `prompt.GOAL_QUESTION_GUIDANCE` | current/direct facts, participant continuity, conditional character context, continuation state, appraisal summary | `active_character_goal`, `relational_willingness`, `private_monologue` |
| P | Visible response intent, capability selection, and assertion boundary | `facade._STAGE_SYSTEM_PROMPTS["P"]` | `prompt.ORDINARY_PLAN_GUIDANCE` or `prompt.SELF_PLAN_GUIDANCE` | goal, current/direct facts, participant continuity, continuation state, supplied capabilities | ordinary or self-cognition `output_contract` selected by the caller |
| All | Structural output discipline | `facade._EXACT_JSON_SYSTEM_SUFFIX` | each packet's `output_contract` | exact caller-owned JSON packet | one JSON object, exact fields, no private references |

`prompt.APPRAISAL_QUESTION_GUIDANCE` is an exported generic appraisal prompt
constant. The current runtime selects the more precise A1 or A2 guidance, but
the generic constant follows the same Chinese-language contract so future
callers cannot reintroduce mixed-language instructions.

## Prompt-Owned Fallback Values

`prompt.build_canonical_turn_workspace(...)` owns the fallback operation text
used when the scene omits `operation`; it is Chinese because it enters every
stage packet. `prompt.build_turn_workspace_stage_contracts(...)` owns a
test/inspection fallback goal whose free-form `intent`, `reason`, and
`cause_summary` values are also Chinese. The contractual `goal_kind` value
remains unchanged.

## Preserved Identifiers

The Chinese prose refers to contract identifiers in code spans. These values
must stay byte-for-byte compatible with validators and callers, including:

- stages: `A1`, `A2`, `G`, `P`;
- authority lanes: `current_observation`, `direct_facts`,
  `participant_continuity`, `conditional_character_context`,
  `continuation_state`;
- G fields: `active_character_goal`, `relational_willingness`,
  `private_monologue`;
- P fields: `goal_resolution`, `response_goal`, `action_requests`,
  `resolver_requests`, `epistemic_boundary`, `self_cognition_response`;
- the packet contract key: `output_contract`;
- all supplied action/capability identifiers and all declared enum values.

## Native-Chinese Term Ledger

| Contract concept | Native Chinese prose |
| --- | --- |
| current observation | 当前观察 |
| direct facts | 直接事实 |
| participant continuity | 参与者连续性 |
| conditional character context | 条件性角色语境 |
| continuation state | 延续状态 |
| appraisal family | 评估类别 |
| active-character goal | 当前角色目标 |
| relational willingness | 关系互动意愿 |
| private monologue | 内心独白 |
| response goal | 回应目标 |
| epistemic boundary | 断言边界 |
| capability | 能力 |
| fail closed | 失败时安全关闭 |

## Separate Route Owners

The following stages are outside this map because they do not execute on
`COGNITION_V3_CHAIN_LLM`:

- relevance classification on `RELEVANCE_AGENT_LLM`;
- vision description and message decontextualization on
  `VISION_DESCRIPTOR_LLM` and `MSG_DECONTEXTUALIZER_LLM`;
- L3 surface planning and final dialog rendering on `DIALOG_GENERATOR_LLM`;
- consolidation, reflection, scheduler, and resolver prompts.

Those owners retain their existing contracts. A future translation of any of
them requires a separate owner map and verification scope.
