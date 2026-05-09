# reflection driven character state evolution plan

## Summary

- Goal: Define the deferred architectural direction for allowing reflection
  cycles to influence durable character state without daily personality drift.
- Plan class: large, architectural direction only
- Status: draft
- Mandatory skills for future promotion: `development-plan-writing`,
  `local-llm-architecture`, `database-data-pull`, `py-style`, and
  `test-style-and-execution`.
- Overall cutover strategy: no cutover is authorized by this draft. A later
  execution plan must be created after the multi-source cognition architecture
  reaches origin-aware consolidation and reflection dry-run readiness.
- Highest-risk areas: personality drift, self-image pollution, user-specific
  leakage into global character state, weak-LLM overreach, duplicated reflection
  channels, and conflicting write ownership with the consolidator.
- Acceptance criteria: this draft records the target direction, research basis,
  dependency on the multi-source cognition plan, and the `character_state`
  fields that may or may not be influenced later.

This document is a high-level deferred architecture plan. It is not an
implementation contract and intentionally omits implementation steps until the
parallel cognition overhaul settles the future reflection and consolidation
contracts.

## Confirmed Direction

Reflection should be allowed to influence the character's durable
self-understanding, but not by directly appending raw reflection output to
`character_state`.

The target direction is:

```text
hourly and daily reflection evidence
-> existing gated reflection promotion artifacts
-> later background self-state evolution pass
-> auditable candidate deltas
-> thresholded character_state updates
```

The existing daily global promotion prompt should remain narrow. It should not
gain an additional direct self-image write lane while the runtime still relies
on a weak/local LLM. A separate later background pass is safer because it can
read already-gated artifacts, apply deterministic thresholds, run in dry-run
mode first, and preserve the live response path.

## Relationship To Multi-Source Cognition

This plan must come after
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`.

The active multi-source cognition plan establishes the prerequisite direction:

- Reflection must not be routed through `/chat` as fake user input.
- Reflection should eventually enter as `trigger_source=reflection_signal`.
- Shared cognition may be reused, but only through source-aware episode
  contracts.
- Consolidation must become origin-aware before non-chat triggers can write
  durable state.
- Current `/chat` behavior and latency must remain the regression baseline.

This draft is therefore blocked until the relevant future child stages clarify
the reflection-origin write policy. At minimum, this work should wait for the
multi-source consolidation-origin stages to define how reflection-derived
outputs can write, dry-run, or fail closed.

## Research Basis

The direction follows established memory and agent-design patterns:

- LangGraph distinguishes short-term memory, long-term memory, semantic memory,
  episodic memory, and procedural memory. It also describes background memory
  writing as a way to avoid live-path latency and keep memory management
  separate from immediate response generation:
  https://docs.langchain.com/oss/python/concepts/memory
- Deep Agents separates scoped memory by agent, user, and organization, which
  supports treating character self-state as agent-level state rather than
  user memory:
  https://docs.langchain.com/oss/python/deepagents/memory
- Generative Agents stores experiences, synthesizes higher-level reflections
  over time, and retrieves those syntheses for behavior:
  https://arxiv.org/abs/2304.03442
- Reflexion stores verbal reflections in memory to improve future decisions
  without changing model weights:
  https://arxiv.org/abs/2303.11366
- MemoryBank describes long-term companion memory that evolves through updates,
  reinforcement, and selective forgetting:
  https://arxiv.org/abs/2305.10250
- OpenAI ChatGPT memory separates remembered facts from raw chat history and
  exposes user controls for viewing, forgetting, and prioritizing memory:
  https://help.openai.com/en/articles/8590148-memory-faq
- Zep models context as facts, entities, episodes, summaries, observations, and
  invalidatable graph relationships:
  https://help.getzep.com/concepts

The common lesson is that reflection-derived behavior should pass through a
durable, scoped, auditable memory layer. It should not be raw log injection and
should not be an unbounded daily profile rewrite.

## Current System Boundary

Current ownership, as of this draft:

- Per-turn consolidation writes `mood`, `global_vibe`,
  `reflection_summary`, and `self_image`.
- `self_image.recent_window` is a per-turn rolling record.
- `self_image.historical_summary` is built by rolling older per-turn entries
  into a compressed long-term self summary.
- Reflection-cycle global promotion writes gated rows to the `memory`
  collection, currently through `lore` and `self_guidance` lanes.
- Reflection-cycle interaction-style updates write to
  `interaction_style_images`, not to `character_state`.
- Normal cognition may consume only promoted/gated reflection context, not raw
  reflection run documents.

This plan does not change those boundaries. It records a later direction for a
new background layer that can review reflection-derived evidence and propose
slow character-state evolution.

## Character State Impact Map

The raw reference document is the singleton MongoDB document:

```text
collection: character_state
_id: global
latest diagnostic export: test_artifacts/character_state_current.json
```

| Raw field | Current decision role | Future influence policy |
|---|---|---|
| `mood` | Immediate emotional filter used by relevance and L1 cognition. | Do not write from daily reflection promotion. Keep owned by per-turn consolidation unless a later origin policy explicitly allows a reflection-origin runtime mood path. |
| `global_vibe` | Background affective temperature for relevance and cognition. | Do not write from daily reflection promotion. Treat as short-term runtime state, not durable identity. |
| `reflection_summary` | Per-turn emotional residue despite the field name. | Do not store reflection-cycle summaries here. Keep this field for per-turn consolidation output. |
| `updated_at` | Timestamp for runtime character state updates. | Do not use as lineage for reflection-derived self evolution. Later work should use explicit metadata under the affected structure. |
| `self_image.recent_window` | Rolling recent self-image entries from per-turn consolidation. | Do not append daily reflection output here. This remains the high-frequency, per-turn channel. |
| `self_image.historical_summary` | Durable compressed self-understanding. | Primary future write target for slow reflection-derived evolution, only after repeated evidence and dry-run review. |
| `self_image.milestones` | Durable identity or self-understanding milestones. | Rare future write target. Require high confidence, multi-run evidence, and low privacy risk. |
| `self_image.meta` | Current synthesis counters and last update metadata. | Future metadata target for source run IDs, candidate counts, and reflection-derived update timestamps. |
| `personality_brief.mbti` | Static personality anchor used by cognition. | No automatic reflection writes. Manual-review candidate only. |
| `personality_brief.logic` | Static reasoning/personality description. | No automatic reflection writes. Manual-review candidate only. |
| `personality_brief.tempo` | Static energy and pacing anchor. | No automatic reflection writes. Manual-review candidate only. |
| `personality_brief.defense` | Static defense-style anchor. | No automatic reflection writes. Manual-review candidate only. |
| `personality_brief.quirks` | Static behavior flavor. | No automatic reflection writes. Manual-review candidate only. |
| `personality_brief.taboos` | Strong character boundary and taboo anchor. | No automatic reflection writes. Any change requires a separate high-risk plan. |
| `boundary_profile.self_integrity` | Decision-affecting boundary tendency. | In analytic scope but not first writable target. Automatic writes require a separate high-risk plan with hard bounds. |
| `boundary_profile.control_sensitivity` | Decision-affecting sensitivity to control. | In analytic scope but not first writable target. Do not daily-adjust. |
| `boundary_profile.compliance_strategy` | Decision-affecting response posture. | In analytic scope but not first writable target. Manual-review or separately approved migration only. |
| `boundary_profile.relational_override` | Decision-affecting relationship override weight. | In analytic scope but not first writable target. Requires strong longitudinal evidence and separate approval. |
| `boundary_profile.control_intimacy_misread` | Decision-affecting control/intimacy interpretation. | In analytic scope but not first writable target. Requires separate approval. |
| `boundary_profile.boundary_recovery` | Decision-affecting recovery tendency after boundary stress. | In analytic scope but not first writable target. Requires separate approval. |
| `boundary_profile.authority_skepticism` | Decision-affecting authority posture. | In analytic scope but not first writable target. Requires separate approval. |
| `linguistic_texture_profile.*` | Style and surface-language tendencies. | Possible future slow calibration, but not part of the first reflection-driven state writer. |
| `tone` | Broad voice setting. | Static unless a separate voice calibration plan approves change. |
| `speech_patterns` | Canonical speech style. | Static unless a separate voice calibration plan approves change. |
| `description` | Canonical character description. | Out of scope for reflection-driven mutation. |
| `backstory` | Canonical backstory. | Out of scope for reflection-driven mutation. |
| `name`, `gender`, `age`, `birthday` | Canonical identity facts. | Out of scope for reflection-driven mutation. |

The plan deliberately maps fields beyond `self_image` because several
non-self-image fields affect decisions. The first writable scope should still
be narrower than the analytic scope.

## Target State

The later target state is a slow, auditable, background character-state
evolution process:

- It reads promoted and gated reflection artifacts, not raw reflection output.
- It proposes candidate changes rather than rewriting state immediately.
- It distinguishes short-term runtime affect from durable self-understanding.
- It treats static character profile fields as higher authority than
  reflection-derived deltas.
- It preserves evidence lineage back to reflection run IDs, memory unit IDs, or
  equivalent future origin metadata.
- It starts with dry-run output and manual review before any production writes.
- It never adds live `/chat` latency.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Daily promotion prompt | Do not add a third direct self-image lane. | The current promotion prompt already owns lore and self-guidance decisions; adding more semantic lanes increases weak-LLM confusion and write risk. |
| Raw reflection output | Do not write raw hourly or daily reflection output to `character_state`. | Reflection runs are evidence and audit records. Runtime cognition should consume only promoted/gated context. |
| First durable target | Prefer `self_image.historical_summary`, `self_image.milestones`, and `self_image.meta`. | These fields represent durable self-understanding without immediately changing core boundary or personality contracts. |
| Recent self-image | Keep `self_image.recent_window` owned by per-turn consolidation. | It is the high-frequency per-turn channel and should not become a daily reflection dump. |
| Runtime affect | Keep `mood`, `global_vibe`, and `reflection_summary` per-turn by default. | These fields affect immediate response behavior and should not be overwritten by daily batch reflection. |
| Boundary and personality | Include them in analytic scope, but not first automatic write scope. | They affect decisions strongly and can destabilize character behavior if changed too freely. |
| Stiffness | Require accumulated longitudinal evidence before durable writes. | The character should evolve over time, not drift daily. |
| Live latency | Keep all reflection-driven self-state evolution outside the live response path. | Normal chat must remain bounded and inspectable. |

## Stiffness Policy

Future execution plans must treat self-state evolution as stiff by default:

- One intense day must not rewrite personality, boundary profile, or durable
  self-image.
- Daily reflection may create candidates, but candidates should not imply
  production writes.
- Durable self-image changes should require repeated support across multiple
  reflection runs or days.
- Stronger fields require stronger gates:
  `self_image.historical_summary` is softer than `self_image.milestones`;
  both are softer than `boundary_profile` or `personality_brief`.
- Static identity and backstory fields are not eligible for reflection-driven
  mutation.

## Deferred Scope

This draft does not authorize:

- Source code changes.
- Prompt changes.
- Database schema changes.
- New live `/chat` LLM calls.
- A third daily reflection promotion lane.
- Direct writes from raw reflection runs to `character_state`.
- Automatic changes to `boundary_profile`, `personality_brief`,
  `speech_patterns`, `tone`, `description`, `backstory`, or canonical identity
  fields.
- Proactive messages or autonomous contact.

## Future Promotion Rules

Before this draft can become an executable plan, the future author must:

- Reread `development_plans/README.md`.
- Reread the multi-source cognition parent plan and its completed child-stage
  evidence.
- Confirm the active status of the consolidation-origin policy stages.
- Re-export or inspect the current `character_state` document.
- Inspect current reflection promotion outputs and promoted memory rows.
- Define exact write ownership, dry-run storage, thresholds, rollback, and
  verification gates in a new execution plan.
- Keep the first executable scope narrower than this analytic field map unless
  the user explicitly approves a broader migration.

## Risks

| Risk | Directional mitigation |
|---|---|
| Personality drift | Use slow candidate accumulation and avoid daily direct writes. |
| Self-image pollution | Read promoted/gated artifacts only and preserve source lineage. |
| User-specific leakage | Reject user-private facts from global character state unless a later plan defines an explicit safe abstraction. |
| Weak-LLM overreach | Keep prompts narrow, use deterministic validation, and avoid asking the model to infer raw database semantics. |
| Consolidator conflict | Wait for origin-aware consolidation before creating any durable writer. |
| Decision instability | Keep boundary and personality fields out of first automatic write scope. |

## Acceptance Criteria

This draft is complete when:

- It is listed in `development_plans/README.md` as an active short-term draft.
- It states that reflection-driven character-state evolution is deferred until
  after the relevant multi-source cognition origin-policy work.
- It defines the architectural direction as a background, gated, auditable
  evolution path rather than a daily promotion prompt expansion.
- It includes research basis links.
- It maps raw `character_state` fields to future influence policy.
- It explicitly forbids implementation, prompt, runtime, and database changes
  from this draft alone.

## Execution Evidence

Draft artifact only. No implementation has been executed from this plan.
