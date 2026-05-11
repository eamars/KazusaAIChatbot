# global character growth from reflection plan

## Summary

- Goal: Implement production global character growth from promoted reflection
  evidence with stable drift, duplicate control, auditability, and a bounded
  cognition read path.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `py-style`, `cjk-safety`, and `test-style-and-execution`.
- Overall cutover strategy: compatible. Existing chat, reflection promotion,
  consolidation, user style images, and character-state writers remain intact.
  The growth pass is default-on; context uses the existing promoted reflection
  projection path.
- Execution priority: completed as a standalone implementation plan. Do not
  append user-style or self-cognition scope to this completed record.
- Highest-risk areas: duplicate global guidance, user-specific style leaking
  into global character growth, domain knowledge becoming personality,
  private-detail leakage, weak-LLM overreach, sudden personality jumps, and
  accidental live-path latency.
- Acceptance criteria: a background production module can form global-growth
  candidates from current reflection-promoted memory, update a stable-drift
  trait ledger, project only promoted source-detail-free traits into cognition,
  and prove the behavior with deterministic, patched, live LLM, static, and
  replay-oriented verification.

## Context

The project goal is to simulate human-like character growth through ongoing
interaction. Personality drift is valid when it represents durable character
growth. The harmful defect is not change itself; the harmful defect is
duplicate, abrupt, private, contradictory, user-specific, or topic-specific
material being promoted into the global character.

Current production already has the required upstream evidence path:

- `character_reflection_runs` stores hourly, daily-channel, and daily-global
  reflection artifacts outside live chat.
- Daily global reflection promotion writes selected artifacts into the shared
  `memory` collection through `memory_evolution` with
  `source_kind="reflection_inferred"` and `authority="reflection_promoted"`.
- `reflection_cycle.context.build_promoted_reflection_context()` can project
  active promoted reflection memory into normal cognition through the bounded
  promoted-reflection-context path.
- L2 cognition already receives `promoted_reflection_context` as global soft
  background and is instructed not to treat it as current-user facts.
- `interaction_style_images`, `user_memory_context`, and group-channel style
  already provide scoped adaptation for one user or channel. They are not the
  target surface for this plan.

The current proof of concept under
`experiments/reflection_state_evolution/` proved two useful facts:

- Current production reflection data can produce communication-facing growth
  candidates while filtering technology, food, tea, dessert, products, and
  other topic competence.
- Stable drift prevents sharp daily personality changes. The latest real
  production-derived simulation produced one accepted candidate, kept it in
  the `emerging` band at strength `0.363`, projected `0` prompt-visible traits,
  and required `7` additional confirming days before promotion.

This plan replaces the older self-image-first dry-run direction. Production
implementation must target global character growth only.

The near-term runtime ROI is intentionally low. The current real-data POC
produced no prompt-visible traits because slow promotion is the safety property
for human-like growth. The implementation is still valuable only if the owner
accepts the longer feedback loop and the queue cost. As of the local registry
on 2026-05-11, `cognition_visual_directives_control_plan.md` is already
completed, `user_style_engagement_consumer_plan.md` is approved and ready, and
`self_cognition_agency_loop_plan.md` is draft. This plan is now approved as a
separate work item; its execution must not include user-style or
self-cognition changes.

## Research Basis

Industry memory systems separate formation, storage, and use:

- LangChain/LangGraph separates short-term and long-term memory and treats
  semantic, episodic, and procedural memory as distinct. The project lesson is
  that global growth may include procedural interaction tendencies, but only
  after they are proven global rather than user-specific.
- Zep stores temporal memory and assembles relevant context for later calls.
  The project lesson is that background memory must have a read path or it
  remains an archive.
- Letta uses compact core memory blocks for always-visible persona guidance and
  archival memory for larger evidence. The project lesson is that source
  evidence must remain auditable while prompt-visible guidance stays compact.
- Google ADK, LlamaIndex, Mem0, OpenAI memory, and Anthropic memory all point
  to the same production controls: scope, retrieval, auditability, update
  history, deletion or revision control, and bloat prevention.

Research sources:

- https://docs.langchain.com/oss/python/concepts/memory
- https://help.getzep.com/v2/memory
- https://help.getzep.com/v2/concepts
- https://docs.letta.com/guides/core-concepts/memory/memory-blocks/
- https://adk.dev/sessions/memory/
- https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/memory/
- https://docs.mem0.ai/platform/features/entity-scoped-memory
- https://help.openai.com/en/articles/8590148-memory-in-chatgpt-remembering-what-you-chat-about
- https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool

## Scope

This plan owns global character growth: durable, cross-context changes to the
active character's long-term interpersonal posture and communication behavior.

Valid global-growth themes include:

- boundary timing;
- guarded care;
- teasing or playful challenge mechanics;
- recovery after absence or misunderstanding;
- emotional exposure;
- clarity and directness;
- trust calibration;
- general stance toward closeness, refusal, repair, and cooperation.

Invalid global-growth themes include:

- technology skill, product knowledge, food, tea, cooking, locations, hobbies,
  or other topic competence;
- one user's wording preference;
- one channel's group dynamic;
- active commitments, relationship facts, milestones, or private details about
  a user;
- static identity facts, backstory, MBTI, canonical taboos, or profile fields.

The target runtime integration is L2 cognition only. Global growth affects the
character's stance and self-understanding; it is not a per-user style overlay
and does not directly rewrite final dialog wording.

## Mandatory Skills

- `development-plan-writing`: load before editing this plan, status, evidence,
  or review gates.
- `local-llm-architecture`: load before editing prompts, cognition payloads,
  reflection boundaries, background LLM behavior, or context projection.
- `py-style`: load before editing Python; read both positive and negative
  constraint references.
- `cjk-safety`: load before editing Python files with CJK prompt text,
  especially cognition prompts.
- `test-style-and-execution`: load before adding, changing, or running tests.
  Deterministic and patched tests may run in batches; live LLM tests must run
  one case at a time with trace inspection.

## Mandatory Rules

- Preserve the normal response path. This plan must not add a response-path
  LLM call.
- Background growth formation reads only promoted reflection memory, active
  global-growth trait rows, and the current time. It must not read raw hourly
  transcripts, raw reflection run output, user style image text, or full user
  memory as formation input.
- Raw source evidence, source memory IDs, reflection run IDs, strength values,
  rejected candidates, and validation warnings are audit data. Normal cognition
  may receive only compact promoted growth guidance.
- Do not write `character_state.self_image`, `personality_brief`,
  `boundary_profile`, `linguistic_texture_profile`, `mood`, `global_vibe`,
  `reflection_summary`, user memory units, interaction-style images,
  conversation rows, scheduled events, or adapter output from this feature.
- Do not target `user_memory_context`, `user_style_image`, or
  `group_channel_style`. They are scoped adaptation surfaces, not global
  character growth surfaces.
- Deterministic code owns validation, caps, duplicate detection, stable IDs,
  drift math, database writes, feature flags, and prompt projection.
- The LLM owns semantic candidate proposal only. It must not generate database
  operations, raw MongoDB filters, write modes, run IDs, trait IDs, or numeric
  strength values.
- Once JSON parses, drop or reject invalid rows deterministically. Do not add
  LLM repair loops or fallback prompts for missing fields.
- Numeric drift state must be converted to semantic labels before prompt use.
  L2 cognition must not receive raw strength, evidence counts, source IDs, or
  threshold values.
- Prompt examples and reusable prompt contracts must not hard-code a concrete
  character name. Use role-neutral wording unless the runtime handler injects
  `character_profile["name"]`.
- Any LLM prompt created or changed by this plan must be rewritten as one
  coherent prompt. Do not append a detached instruction block for global
  growth; integrate the new rules into role framing, language policy, task,
  reasoning path, input format, and output format.
- Prompt free text must be Simplified Chinese, following the structure and
  concise style of `_COGNITION_SUBCONSCIOUS_PROMPT`. Schema keys, enum values,
  IDs, URLs, commands, and copied source evidence may remain in their original
  language.
- Manual CLI apply mode must require explicit write enablement. Worker growth
  writes are default-on through `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED=true`,
  but remain bounded by validation, stable drift, and worker busy checks.
  Dry-run remains available and must not mutate trait rows.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  active agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.

## Must Do

- Create a production package named
  `src/kazusa_ai_chatbot/global_character_growth/`.
- Create a database interface named
  `src/kazusa_ai_chatbot/db/global_character_growth.py`.
- Create two MongoDB collections:
  `global_character_growth_traits` and `global_character_growth_runs`.
- Add one explicit feature flag in `config.py` with default value:
  `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED=true`.
- Add an ICD-style module README at
  `src/kazusa_ai_chatbot/global_character_growth/README.md`.
- Add a manual CLI at `src/scripts/run_global_character_growth.py` supporting
  dry-run and explicit apply modes.
- Add a worker integration that runs after daily global reflection promotion
  only when `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED` is true.
- Read input evidence through `memory_evolution.find_active_memory_units()`
  with `source_kind="reflection_inferred"`,
  `authority="reflection_promoted"`, and `source_global_user_id=""`.
- Generate global-growth candidates with one consolidation LLM call, then
  validate and drift-update traits deterministically.
- Keep stable drift gradual using the thresholds and formula defined in this
  plan.
- Treat the drift constants as provisional first calibration. Keep them
  centralized in `models.py`, record per-run drift evidence, and do not claim
  the constants are empirically proven beyond the current limited POC.
- Record input-quality diagnostics and a log-only shadow projection in every
  run document. Shadow projection exists for operator review only and must not
  enter the runtime prompt context.
- Extend promoted reflection context with a new
  `promoted_global_growth` projection that includes only promoted active
  traits and no source details.
- Rewrite the affected L2 cognition prompt flow coherently so promoted global
  growth is read as general character context, not user facts, current
  commitments, or style-image instructions.
- Add deterministic, patched orchestration, prompt-contract, module-boundary,
  worker, context-projection, live LLM false-negative, live LLM false-positive,
  and replay-oriented tests.
- Run every verification command in this plan and record results in
  `Execution Evidence`.

## Deferred

- Do not mutate `character_state.self_image` in this plan.
- Do not add a new daily global reflection promotion lane.
- Do not change hourly or daily-channel reflection prompt schemas.
- Do not change existing lore or self-guidance promotion behavior.
- Do not improve, broaden, or re-rank the upstream daily-global reflection
  promotion lane in this plan.
- Do not change interaction-style extraction, user-style persistence, or
  group-channel style persistence.
- Do not change RAG planner, RAG dispatcher, RAG helper agents, dialog agent,
  dialog evaluator, adapters, scheduler event semantics, or user-memory
  consolidation.
- Do not migrate historical memory into traits. Initial trait rows are created
  only by running the new pass after implementation.
- Do not add a human approval UI, web endpoint, autonomous contact behavior,
  or proactive output behavior.
- Do not project shadow traits into L2, L3, dialog, RAG, adapters, or any
  response-path prompt.
- Do not consume hourly self-cognition outputs or
  `self_cognition_agency_loop_plan.md` artifacts as input evidence.
- Do not remove existing reflection POC files or test artifacts.

## Cutover Policy

Overall strategy: compatible.

| Area | Policy | Instruction |
|---|---|---|
| Database collections | migration | Add new collections and indexes only. Do not rewrite existing rows. |
| Background growth pass | compatible | Add a default-on worker path and manual CLI. `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED=false` is the rollback switch. |
| Trait writes | compatible | Worker apply writes only the new trait/run collections by default. Manual CLI dry-run writes one run record and no trait mutations. |
| Shadow projection | compatible | Add log-only shadow projection to run records. It is never merged into prompt context. |
| Cognition context | compatible | Use the existing promoted-reflection-context path. Add no separate global-growth context flag. |
| L2 cognition prompt | compatible | Rewrite the affected Chinese prompt flow to integrate one compact field; no new LLM call and no L3/dialog rewrite. |
| Reflection promotion | compatible | Keep existing promotion lanes and memory writes unchanged. |
| Tests | compatible | Add focused tests and static checks without deleting existing tests. |

Cutover policy enforcement:

- The implementation agent must follow the selected policy for each area.
- The agent must not choose a more conservative or broader strategy by default.
- Compatible means only the compatibility surfaces listed above are allowed.
- Any new write path, compatibility shim, fallback path, dual write, or runtime
  prompt surface outside this table requires owner approval before
  implementation.

## Agent Autonomy Boundaries

- The target ownership boundary is
  `src/kazusa_ai_chatbot/global_character_growth/` plus the exact DB, config,
  reflection context, worker, cognition prompt, script, doc, and test files
  named in `Change Surface`.
- The agent may choose local helper names only when they preserve the public
  contracts in this plan and pass `py-style` constraints.
- The agent must search existing code before adding helpers. Use existing
  public APIs for memory reads, JSON parsing, LLM construction, DB bootstrap,
  time handling, and test trace writing when available.
- The agent must not introduce alternative storage, alternate runtime context
  keys, feature flags beyond the named growth-pass flag, retry loops, hidden fallback
  behavior, compatibility layers, dependency changes, or unrelated cleanup.
- If the codebase contradicts this plan, preserve the plan intent and record
  the discrepancy in `Execution Evidence`.
- If a required instruction cannot be implemented safely, stop and record the
  blocker instead of inventing a substitute.

## Target State

After implementation, the production system has a slow global-growth loop:

```text
reflection-promoted memory
  -> global growth candidate LLM
  -> deterministic validation and duplicate rejection
  -> stable drift trait accumulator
  -> promoted source-detail-free global growth context
  -> L2 cognition soft background
```

Observable behavior:

- By default, the growth pass runs after daily global reflection promotion and
  writes only the new growth trait/run collections.
- Promoted global-growth context is prompt visible only when at least one active
  trait reaches `promoted`.
- Operators can run a dry-run CLI and inspect candidate, rejection, and drift
  evidence without mutating traits.
- Operators can inspect shadow projection for emerging, stabilizing, and
  promoted traits immediately after a run, without affecting live cognition.
- Operators can run explicit CLI apply mode, or let the default-on background
  pass update the trait ledger after reflection promotion.
- L2 cognition sees promoted global growth only when the promoted reflection
  context builder projects at least one promoted trait.
- A newly observed candidate does not immediately affect runtime behavior.
  It must accumulate enough evidence to reach `promoted`.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Production target | Use a new `global_character_growth` package. | The feature is not self-image, user style, or reflection promotion. A separate package keeps ownership explicit. |
| Durable storage | Use `global_character_growth_traits` and `global_character_growth_runs`. | `memory_evolution` normalizes memory docs and is not shaped for numeric drift, maturity bands, and run audit state. |
| Input evidence | Read active reflection-promoted memory only. | Promoted memory has already passed promotion policy; raw reflection has privacy and noise risk. |
| Candidate semantics | LLM proposes communication-growth candidates only. | Semantic generalization belongs to the model, while validation and persistence remain deterministic. |
| Drift model | Use bounded evidence accumulation with maturity bands and provisional constants. | Human-like growth should be gradual. The POC supports the path, but the constants were calibrated on very little real candidate volume and must stay tunable. |
| Shadow projection | Store source-detail-free emerging, stabilizing, and promoted guidance in run records only. | This shortens the operator feedback loop while preserving the rule that only promoted traits affect cognition. |
| Input quality visibility | Store promotion-density and drop-reason diagnostics in each run. | The plan inherits upstream promoted-memory quality; it does not repair the upstream lane, so every run must show whether sparse or noisy input limited the output. |
| Runtime projection | Add `promoted_global_growth` under `promoted_reflection_context`. | Service and L2 already have a promoted global reflection context path; extending it avoids a new response-path fetch. |
| Cognition owner | Inject into L2 only. | L2 owns logical stance and character intent. L3 style remains scoped to style overlays and current-turn packaging. |
| Feature flag | Default the growth pass to enabled and do not add a context-specific flag. | Character growth is a project goal. Context exposure stays inside promoted-only projection. |
| Self-image writes | Defer all `character_state.self_image` mutation. | The POC showed useful growth but also showed duplication/bloat risk from direct summary appends. |
| Self-cognition merge point | Keep self-cognition outputs out of this plan's input contract. | A later plan may add a separately promoted self-cognition evidence source, but this plan reads only reflection-promoted memory. |

## Contracts And Data Shapes

### Public Package API

`src/kazusa_ai_chatbot/global_character_growth/__init__.py` must export only:

```python
async def run_global_character_growth_pass(
    *,
    character_local_date: str | None,
    dry_run: bool,
    enable_trait_writes: bool,
    limit: int = 80,
    now: datetime | None = None,
) -> GlobalCharacterGrowthRunResult: ...

async def build_global_character_growth_context(
    *,
    limit: int = 3,
) -> GlobalCharacterGrowthContext: ...
```

`enable_trait_writes=True` is valid only when `dry_run=False`. The runner must
raise `ValueError` for `dry_run=True` with `enable_trait_writes=True`.
`build_global_character_growth_context()` is called from the promoted
reflection context projection; do not add a separate context flag.

### ICD-Style Module README Contract

`src/kazusa_ai_chatbot/global_character_growth/README.md` must be written as an
Interface Control Document for the module. It must define:

- purpose, owner, non-owner surfaces, public entrypoints, and callers;
- input, output, MongoDB, flag, LLM, failure, audit, and operation contracts;
- verification commands and future handoff notes.

### Candidate Prompt Payload

The LLM handler consumes:

```python
{
    "evaluation_mode": "global_character_growth_v1",
    "prompt_version": "global_character_growth_candidate_v1",
    "memory_cards": [
        {
            "source_card_id": str,
            "memory_unit_id": str,
            "memory_name": str,
            "memory_type": str,
            "content": str,
            "character_local_dates": list[str],
            "source_reflection_run_ids": list[str],
            "confidence_note": str,
        }
    ],
    "current_global_growth_traits": [
        {
            "trait_id": str,
            "growth_axis": str,
            "guidance": str,
            "maturity_band": str,
        }
    ],
    "candidate_limits": {
        "max_candidates": 4,
        "max_source_cards_per_candidate": 8,
    },
    "allowed_growth_axes": [
        "boundary_timing",
        "guarded_care",
        "playful_challenge",
        "recovery_style",
        "clarity",
        "emotional_exposure",
        "trust_calibration",
        "other_communication",
    ],
}
```

Hard input caps:

- `limit <= 80`.
- At most `420` characters per card `content`.
- At most `120` characters per card `confidence_note`.
- At most `8` dates per card.
- At most `8` source reflection run IDs per card.
- At most `12` current trait summaries.
- At most `220` characters per current trait guidance.

### LLM Output Contract

The LLM must output JSON only:

```python
{
    "candidate_deltas": [
        {
            "candidate_action": "observe_trait" | "no_action",
            "growth_axis": str,
            "trait_name": str,
            "guidance": str,
            "source_card_ids": list[str],
            "supporting_dates": list[str],
            "scope_assessment": "global" | "user_specific"
                | "channel_specific" | "domain_topic" | "insufficient",
            "support_level": "insufficient" | "emerging" | "stable",
            "confidence": "low" | "medium" | "high",
            "private_detail_risk": "low" | "medium" | "high",
            "novelty_reason": str,
            "stability_reason": str,
            "rejection_reason": str,
        }
    ],
    "summary": str,
}
```

The prompt must explicitly reject technology, products, hobbies, food, tea,
cooking, locations, per-user style, group-channel style, current commitments,
and private details as global personality growth.

### LLM Prompt Contract

`global_character_growth/llm.py` must define one Chinese prompt constant for
candidate generation. The prompt must follow the organization style of
`_COGNITION_SUBCONSCIOUS_PROMPT` without copying its subconscious role:

- opening role paragraph in Chinese;
- `# 语言政策`;
- `# 核心过滤器`;
- `# 运行规则`;
- `# 任务目标`;
- `# 思考路径`;
- `# 输入格式`;
- `# 输出格式`.

The prompt must integrate all rules into those sections. It must not append a
late "extra rules" block for technology rejection, privacy, global scope, or
JSON output. The role must be a long-term global character-growth evaluator for
the active character; it must not pretend to be the character, the user, L2,
L3, dialog, or the subconscious layer.

When editing Python prompt strings with Chinese content, use a
`cjk-safety`-approved representation such as a triple-single-quoted string or
verbatim byte-copy from an existing safe source, then run syntax checks
immediately.

### Deterministic Validation

`validation.py` must enforce:

- `candidate_action="no_action"` becomes a rejected candidate with its reason.
- `growth_axis` must be one of the allowed axes.
- `scope_assessment` must be `global`.
- `guidance` must be non-empty, source-detail-free, and at most `240`
  characters.
- `trait_name` must be non-empty and at most `80` characters.
- `private_detail_risk` must be `low`.
- `source_card_ids` must be unique and resolve to input cards.
- `supporting_dates` must be unique and a subset of dates derived from the
  selected source cards.
- Accepted candidates require at least `2` source cards and at least `2`
  supporting dates.
- `support_level="stable"` and `confidence="high"` are required for a candidate
  to enter drift with full evidence strength; weaker valid candidates enter
  with lower evidence strength and cannot promote in a single run.
- Reject candidates that contain platform IDs, user IDs, message IDs, memory
  unit IDs, reflection run IDs, or quoted private source details in `guidance`.
- Reject candidates that overlap an active trait or another accepted candidate
  at `0.65` or higher unless they are merged into the stronger candidate.
- Keep at most `4` accepted candidates per run.
- Build deterministic `candidate_id` from prompt version, growth axis,
  normalized guidance, and sorted source memory IDs.

### Stable Drift Contract

Trait strength updates must follow this first-calibration stable path:

```text
raw_strength = previous_strength * 0.85 + evidence_strength * 0.15
applied_delta = min(raw_strength - previous_strength, 0.18)
new_strength = clamp(previous_strength + applied_delta, 0.0, 1.0)
```

Maturity bands:

| Band | Strength |
|---|---:|
| `observed` | `< 0.25` |
| `emerging` | `>= 0.25` and `< 0.50` |
| `stabilizing` | `>= 0.50` and `< 0.75` |
| `promoted` | `>= 0.75` |

Only `status="active"` and `maturity_band="promoted"` traits are eligible for
prompt projection.

The constants `0.85`, `0.15`, `0.18`, `0.25`, `0.50`, and `0.75` are
provisional production defaults. The implementation must keep them in one
central constants block in `models.py` and include them in contract tests so
future retuning is deliberate. Retuning these constants after real production
volume requires a follow-up plan with replay evidence; implementation agents
must not silently adjust them.

### Trait Document

`GlobalCharacterGrowthTraitDoc` in `db/schemas.py` must represent:

```python
{
    "_id": str,
    "trait_id": str,
    "lineage_id": str,
    "status": "active" | "superseded" | "rejected",
    "growth_axis": str,
    "trait_name": str,
    "guidance": str,
    "strength": float,
    "maturity_band": "observed" | "emerging" | "stabilizing" | "promoted",
    "first_observed_date": str,
    "last_observed_date": str,
    "supporting_dates": list[str],
    "source_memory_unit_ids": list[str],
    "source_reflection_run_ids": list[str],
    "source_candidate_ids": list[str],
    "evidence_count": int,
    "version": int,
    "supersedes_trait_ids": list[str],
    "merged_from_trait_ids": list[str],
    "created_at": str,
    "updated_at": str,
}
```

Unknown fields must not be required by prompt projection. Projection reads only
`growth_axis`, `guidance`, `maturity_band`, and `updated_at`.

### Run Document

`GlobalCharacterGrowthRunDoc` in `db/schemas.py` must represent:

```python
{
    "_id": str,
    "run_id": str,
    "run_kind": "global_character_growth",
    "status": "dry_run" | "applied" | "skipped" | "failed",
    "dry_run": bool,
    "prompt_version": str,
    "created_at": str,
    "updated_at": str,
    "character_local_date": str,
    "input_counts": {
        "raw_memory_rows": int,
        "eligible_memory_cards": int,
        "current_traits": int,
    },
    "input_quality": {
        "promotion_density": "none" | "sparse" | "adequate",
        "eligible_date_count": int,
        "date_span_days": int,
        "dropped_memory_cards_by_reason": dict[str, int],
        "quality_notes": list[str],
    },
    "source_memory_unit_ids": list[str],
    "source_reflection_run_ids": list[str],
    "accepted_candidates": list[dict],
    "rejected_candidates": list[dict],
    "trait_updates": list[dict],
    "shadow_projection": list[dict],
    "validation_warnings": list[str],
    "raw_llm_output": str,
    "summary": str,
    "error": str,
}
```

Dry-run records include planned trait updates but do not mutate
`global_character_growth_traits`.

### Shadow Projection Contract

The runner must build `shadow_projection` from validated planned or applied
trait updates after drift calculation. It is capped at `5` items and includes
only source-detail-free guidance for traits in `emerging`, `stabilizing`, or
`promoted` bands:

```python
{
    "growth_axis": str,
    "guidance": str,
    "maturity": "emerging" | "stabilizing" | "promoted",
    "prompt_visible_now": bool,
    "review_note": str,
}
```

`prompt_visible_now` is `True` only for promoted active traits that also pass
the runtime projection rules. Shadow projection must never be returned by
`build_global_character_growth_context()`, merged into
`promoted_reflection_context`, or added to L2/L3/dialog payloads.

### Runtime Context Projection

`build_global_character_growth_context()` returns:

```python
{
    "promoted_global_growth": [
        {
            "growth_axis": str,
            "guidance": str,
            "maturity": "promoted",
            "updated_at": "YYYY-MM-DD",
        }
    ],
    "retrieval_notes": [
        "Only promoted global character growth traits are included."
    ],
}
```

`reflection_cycle.context.build_promoted_reflection_context()` must merge this
under the existing reflection context:

```python
{
    "promoted_lore": list[dict],
    "promoted_self_guidance": list[dict],
    "promoted_global_growth": list[dict],
    "source_dates": list[str],
    "retrieval_notes": list[str],
}
```

The L2 prompt payload may include `promoted_global_growth`; L2 must not receive
`strength`, `candidate_id`, `trait_id`, `source_memory_unit_ids`,
`source_reflection_run_ids`, rejected candidates, or validation warnings.

## LLM Call And Context Budget

| Call | Before | After | Path | Budget |
|---|---:|---:|---|---|
| Live chat L2 cognition | unchanged | unchanged | response path | no new calls; one compact optional field |
| Reflection promotion | unchanged | unchanged | background | no new calls |
| Global growth candidate generation | `0` | `1` | background worker or CLI | max `80` cards and `12` current traits |

Conservative candidate-generation context estimate:

- `80 * 420 = 33,600` memory-card characters.
- `12 * 220 = 2,640` current-trait characters.
- Less than `8,000` prompt/schema characters.
- Total expected character budget under `45,000`, within the default `50k`
  token planning cap.

Live path impact:

- No extra live LLM call.
- At most one additional MongoDB read through
  `global_character_growth_context` when promoted reflection context projection
  includes global growth context.
- Context projection cap is `3` promoted traits.
- Shadow projection adds no prompt tokens and no live-path work. It is stored
  only in run records for operator review.

## Data Migration

No backfill is authorized.

Add collections in `db_bootstrap()`:

- `global_character_growth_traits`
- `global_character_growth_runs`

Required indexes:

- `global_growth_trait_id_unique`: unique `trait_id`.
- `global_growth_trait_status_maturity`: `(status, maturity_band, updated_at)`.
- `global_growth_trait_axis_status`: `(growth_axis, status)`.
- `global_growth_trait_source_memory`: multikey `source_memory_unit_ids`.
- `global_growth_run_id_unique`: unique `run_id`.
- `global_growth_run_status_updated`: `(status, updated_at)`.
- `global_growth_run_source_memory`: multikey `source_memory_unit_ids`.
- `global_growth_run_source_reflection`: multikey
  `source_reflection_run_ids`.

Existing collections must not be rewritten.

## Operational Steps

Dry-run:

```powershell
venv\Scripts\python -m scripts.run_global_character_growth --dry-run --limit 80
```

Apply:

```powershell
venv\Scripts\python -m scripts.run_global_character_growth --apply --enable-trait-writes --limit 80
```

Default worker value:

```env
GLOBAL_CHARACTER_GROWTH_PASS_ENABLED=true
```

Rollback override:

```env
GLOBAL_CHARACTER_GROWTH_PASS_ENABLED=false
```

The CLI must print run id, status, dry-run/apply mode, eligible memory cards,
accepted candidate count, rejected candidate count, trait update count,
promoted trait count, shadow projection count, input-quality density, and
warning count.

## Change Surface

### Delete

- None.

### Create

- `src/kazusa_ai_chatbot/global_character_growth/__init__.py`
  - Public package exports only.
- `src/kazusa_ai_chatbot/global_character_growth/README.md`
  - ICD-style interface control document: owner/non-owner surfaces, public
    entrypoints, callers, input/output contracts, DB/index ownership, default
    flags, failure modes, operations, verification, and handoff notes.
- `src/kazusa_ai_chatbot/global_character_growth/models.py`
  - Constants, `TypedDict` contracts, allowed axes, statuses, caps, and
    threshold values.
- `src/kazusa_ai_chatbot/global_character_growth/projection.py`
  - Reflection-promoted memory projection, current trait projection, prompt
    payload builder, run shadow projection, input-quality diagnostics, and
    prompt-safe context projection.
- `src/kazusa_ai_chatbot/global_character_growth/llm.py`
  - Prompt, consolidation LLM instance, handler, and JSON parsing.
- `src/kazusa_ai_chatbot/global_character_growth/validation.py`
  - Candidate validation, deterministic IDs, privacy/source/scope gates,
    duplicate detection, caps, and rejection records.
- `src/kazusa_ai_chatbot/global_character_growth/drift.py`
  - Evidence scoring, stable drift update, maturity bands, merge/supersede
    planning, and prompt-promotion eligibility.
- `src/kazusa_ai_chatbot/global_character_growth/runner.py`
  - Orchestration for reads, LLM call, validation, drift planning, trait writes,
    run records, skip handling, and failure records.
- `src/kazusa_ai_chatbot/global_character_growth/context.py`
  - Public `build_global_character_growth_context()` runtime projection.
- `src/kazusa_ai_chatbot/db/global_character_growth.py`
  - Raw MongoDB collection access and index creation for traits and runs.
- `src/scripts/run_global_character_growth.py`
  - Manual CLI with dry-run and explicit apply mode.
- `tests/test_global_character_growth_contract.py`
  - Caps, constants, public API, prompt payload projection, run ID
    determinism, default-on pass flag value, input-quality shape, shadow
    projection shape, and context projection shape.
- `tests/test_global_character_growth_validation.py`
  - Candidate acceptance, domain rejection, technology rejection,
    user-specific rejection, private-detail rejection, duplicate rejection,
    source checks, and caps.
- `tests/test_global_character_growth_drift.py`
  - Stable drift math, maturity bands, daily delta cap, merge planning, and
    prompt-visibility threshold. Tests must state that constants are
    provisional defaults, not broadly proven values.
- `tests/test_global_character_growth_runner.py`
  - Patched orchestration for dry-run, apply, skip, failure, and no-write
    dry-run behavior, including run-document input quality and shadow
    projection.
- `tests/test_global_character_growth_context.py`
  - Runtime context projection and reflection context merge behavior.
- `tests/test_global_character_growth_worker.py`
  - Reflection worker integration behind feature flag and busy-probe behavior.
- `tests/test_global_character_growth_prompt_contracts.py`
  - Prompt render, schema visibility, no hard-coded character name, and
    instruction coverage. Tests must assert the Chinese prompt contains the
    required `_COGNITION_SUBCONSCIOUS_PROMPT`-style sections and has no detached
    appended rule block.
- `tests/test_global_character_growth_module_boundary.py`
  - Static import and writer-boundary checks.
- `tests/test_global_character_growth_live_llm.py`
  - Two live LLM candidate-generation contract tests with durable trace output:
    one false-negative guard that accepts stable global communication growth,
    and one false-positive guard that rejects domain-topic, technology,
    per-user style, and private-detail noise.
- `tests/test_global_character_growth_replay.py`
  - Patched replay comparison proving promoted context reaches L2 and stays
    absent when not promoted or disabled.

### Modify

- `src/kazusa_ai_chatbot/config.py`
  - Add the default-on growth-pass feature flag.
- `docs/HOWTO.md`
  - Document the new default-on pass flag, promoted reflection context path,
    rollback override, and CLI commands.
- `src/kazusa_ai_chatbot/db/schemas.py`
  - Add `GlobalCharacterGrowthTraitDoc` and
    `GlobalCharacterGrowthRunDoc`.
- `src/kazusa_ai_chatbot/db/bootstrap.py`
  - Add the two collections and call
    `ensure_global_character_growth_indexes()`.
- `src/kazusa_ai_chatbot/db/__init__.py`
  - Export DB helper functions only when they are part of the public DB facade
    required by tests or scripts. Internal runner code may import the DB
    submodule directly as a named DB interface.
- `src/kazusa_ai_chatbot/reflection_cycle/context.py`
  - Merge `promoted_global_growth` into `PromotedReflectionContext` when
    active promoted traits exist.
- `src/kazusa_ai_chatbot/reflection_cycle/worker.py`
  - Run the global growth pass after daily global reflection promotion when
    `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED` is true and the busy probe remains
    idle.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
  - Rewrite the affected prompt flow so `promoted_global_growth` is integrated
    into the existing Chinese reasoning structure instead of appended as a
    detached block.
- `development_plans/active/short_term/global_character_growth_from_reflection_plan.md`
  - During execution only, update progress, evidence, review findings, and
    lifecycle status.
- `development_plans/README.md`
  - During execution only, update the registry row when status changes.

### Keep

- `src/kazusa_ai_chatbot/service.py`
  - No direct edit. Existing promoted reflection context loading remains the
    service boundary.
- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
  - Existing lore and self-guidance promotion remains unchanged.
- `src/kazusa_ai_chatbot/reflection_cycle/prompts.py`
  - Hourly and daily reflection prompts remain unchanged.
- `src/kazusa_ai_chatbot/db/character.py`
  - Existing character-state writers remain unchanged and unused.
- `src/kazusa_ai_chatbot/db/interaction_style_images.py`
  - Existing user and group style behavior remains unchanged.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
  - Existing style and preference stages remain unchanged.
- `src/kazusa_ai_chatbot/rag/`
  - RAG planner, dispatcher, projection, and helper agents remain unchanged.
- Adapters and scheduler modules
  - No changes.

## Implementation Order

1. Add failing deterministic tests for contracts, validation, drift, runner,
   context, worker, prompt contracts, replay plumbing, module boundaries,
   input-quality diagnostics, and shadow projection. Record expected
   missing-symbol failures.
2. Implement `models.py`, `projection.py`, `validation.py`, and `drift.py`.
   Run contract, validation, and drift tests until they pass.
3. Implement `db/global_character_growth.py`, schema types, bootstrap
   collections, and indexes. Run DB/bootstrap focused tests.
4. Implement `llm.py` and prompt-render tests. Run prompt contract tests.
5. Implement `runner.py` and the manual CLI, including input-quality
   diagnostics and run-record shadow projection. Run patched runner and CLI
   tests.
6. Implement `context.py` and integrate it into `reflection_cycle/context.py`.
   Run context and existing reflection-context tests.
7. Integrate worker scheduling behind
   `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED`. Run worker tests.
8. Update L2 cognition prompt and payload schema. Run prompt contract,
   replay, and existing cognition prompt tests.
9. Add and run the live LLM false-negative and false-positive tests one case
   at a time. Inspect each durable trace before running the next real LLM
   case.
10. Run every verification command in this plan.
11. Run independent code review. Fix only approved-surface findings, then rerun
    affected verification.

## Progress Checklist

- [x] Stage 1 - tests added and baseline failures recorded
  - Covers: implementation step 1.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py tests\test_global_character_growth_validation.py tests\test_global_character_growth_drift.py tests\test_global_character_growth_runner.py tests\test_global_character_growth_context.py tests\test_global_character_growth_worker.py tests\test_global_character_growth_prompt_contracts.py tests\test_global_character_growth_module_boundary.py tests\test_global_character_growth_replay.py -q`
  - Evidence: missing-symbol or baseline failures recorded in
    `Execution Evidence`.
  - Sign-off: `Codex/2026-05-11` after evidence is recorded.

- [x] Stage 2 - core package internals complete
  - Covers: implementation step 2.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py tests\test_global_character_growth_validation.py tests\test_global_character_growth_drift.py -q`
  - Evidence: passing output, changed files, and validation/drift summary.
  - Sign-off: `Codex/2026-05-11` after verification and evidence.

- [x] Stage 3 - DB and bootstrap complete
  - Covers: implementation step 3.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_db.py tests\test_global_character_growth_runner.py -q`
  - Evidence: collection/index behavior and no existing-row migration.
  - Sign-off: `Codex/2026-05-11` after verification and evidence.

- [x] Stage 4 - LLM contract, runner, and CLI complete
  - Covers: implementation steps 4 and 5.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_global_character_growth_prompt_contracts.py tests\test_global_character_growth_runner.py -q`
  - Evidence: prompt render, dry-run behavior, apply gating, and failure
    record behavior.
  - Sign-off: `Codex/2026-05-11` after verification and evidence.

- [x] Stage 5 - runtime projection and worker integration complete
  - Covers: implementation steps 6 and 7.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_global_character_growth_context.py tests\test_reflection_cycle_stage1c_reflection_context.py tests\test_global_character_growth_worker.py tests\test_reflection_cycle_stage1c_worker.py -q`
  - Evidence: default config enables the growth pass; pass flag false prevents
    growth writes; absence of promoted traits prevents context.
  - Sign-off: `Codex/2026-05-11` after verification and evidence.

- [x] Stage 6 - cognition integration complete
  - Covers: implementation step 8.
  - Verify:
    `venv\Scripts\python -m pytest tests\test_global_character_growth_replay.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py tests\test_cognition_interaction_style_context.py -q`
  - Evidence: L2 receives promoted global growth only from active promoted
    traits, and style-image behavior remains separate.
  - Sign-off: `Codex/2026-05-11` after verification and evidence.

- [x] Stage 7 - live LLM false-negative and false-positive tests inspected
  - Covers: implementation step 9.
  - Verify:
    `venv\Scripts\python -m pytest -m live_llm tests\test_global_character_growth_live_llm.py::test_global_character_growth_live_accepts_stable_communication_growth -q -s`
    followed by
    `venv\Scripts\python -m pytest -m live_llm tests\test_global_character_growth_live_llm.py::test_global_character_growth_live_rejects_domain_and_user_specific_noise -q -s`.
  - Evidence: both trace paths, parsed outputs, accepted/rejected reasoning,
    false-negative judgment, and false-positive judgment.
  - Sign-off: `Codex/2026-05-11` after both traces are inspected individually.

- [x] Stage 8 - full verification complete
  - Covers: implementation step 10.
  - Verify: every command in `Verification`.
  - Evidence: command results, static grep outputs, allowed skips, CLI smoke,
    and residual risks.
  - Sign-off: `Codex/2026-05-11` after all required gates pass or approved skips
    are recorded.

- [x] Stage 9 - independent code review complete
  - Covers: implementation step 11.
  - Verify: `Independent Code Review` completed and affected checks rerun.
  - Evidence: reviewer mode, findings, fixes, rerun commands, residual risks,
    and approval status.
  - Sign-off: `Codex/2026-05-11` after review approves completion.

## Verification

### Syntax

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\global_character_growth\__init__.py src\kazusa_ai_chatbot\global_character_growth\models.py src\kazusa_ai_chatbot\global_character_growth\projection.py src\kazusa_ai_chatbot\global_character_growth\llm.py src\kazusa_ai_chatbot\global_character_growth\validation.py src\kazusa_ai_chatbot\global_character_growth\drift.py src\kazusa_ai_chatbot\global_character_growth\runner.py src\kazusa_ai_chatbot\global_character_growth\context.py src\kazusa_ai_chatbot\db\global_character_growth.py src\scripts\run_global_character_growth.py`
  - Expected: exit code `0`.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py tests\test_global_character_growth_validation.py tests\test_global_character_growth_drift.py -q`
  - Expected: pass.
- `venv\Scripts\python -m pytest tests\test_global_character_growth_runner.py tests\test_global_character_growth_context.py tests\test_global_character_growth_worker.py tests\test_global_character_growth_prompt_contracts.py tests\test_global_character_growth_module_boundary.py tests\test_global_character_growth_replay.py -q`
  - Expected: pass.

### Regression Tests

- `venv\Scripts\python -m pytest tests\test_reflection_cycle_stage1c_promotion.py tests\test_reflection_cycle_stage1c_reflection_context.py tests\test_reflection_cycle_stage1c_worker.py -q`
  - Expected: pass.
- `venv\Scripts\python -m pytest tests\test_memory_evolution_retrieval.py tests\test_memory_evolution_module_boundary.py -q`
  - Expected: pass.
- `venv\Scripts\python -m pytest tests\test_cognition_interaction_style_context.py tests\test_cognition_preference_adapter.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py -q`
  - Expected: pass.

### Live LLM Test

- `venv\Scripts\python -m pytest -m live_llm tests\test_global_character_growth_live_llm.py::test_global_character_growth_live_accepts_stable_communication_growth -q -s`
  - Expected: pass or skip only when the configured consolidation LLM endpoint
    is unavailable.
  - Required inspection: read the trace and confirm the model does not miss a
    repeated, source-supported global communication-growth candidate.
- `venv\Scripts\python -m pytest -m live_llm tests\test_global_character_growth_live_llm.py::test_global_character_growth_live_rejects_domain_and_user_specific_noise -q -s`
  - Expected: pass or skip only when the configured consolidation LLM endpoint
    is unavailable.
  - Required inspection: read the trace and confirm the model rejects
    technology/domain competence, user-specific style, private detail, and
    non-personality content instead of emitting false-positive growth traits.
  - For both live cases, confirm parseable JSON, valid source card IDs, and no
    private-detail leakage. Record that drift constants remain provisional even
    if both traces pass.

### Static Greps

- `$testFiles = Get-ChildItem -LiteralPath 'tests' -Filter 'test_global_character_growth_*.py' | ForEach-Object { $_.FullName }; rg "upsert_character_self_image|upsert_character_state|save_character_profile" src\kazusa_ai_chatbot\global_character_growth @testFiles`
  - Expected: zero matches. Exit code `1` from `rg` is acceptable.
- `rg "get_db\(|\.insert_one\(|\.update_one\(|\.update_many\(|\.delete_one\(|\.delete_many\(|\.replace_one\(" src\kazusa_ai_chatbot\global_character_growth`
  - Expected: zero matches. Raw DB calls belong only in
    `src\kazusa_ai_chatbot\db\global_character_growth.py`.
- `rg "global_character_growth" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\brain_service src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\nodes\dialog_agent.py`
  - Expected: zero matches.
- `rg "user_style_image|group_channel_style|interaction_style_images" src\kazusa_ai_chatbot\global_character_growth`
  - Expected: zero matches.
- `rg "promoted_global_growth" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py src\kazusa_ai_chatbot\nodes\dialog_agent.py`
  - Expected: zero matches. Global growth is L2 context only.
- `rg "shadow_projection" src\kazusa_ai_chatbot\nodes src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\reflection_cycle\context.py`
  - Expected: zero matches. Shadow projection belongs only to run records,
    tests, CLI output, and the `global_character_growth` package.

### Manual CLI Smoke

- `venv\Scripts\python -m scripts.run_global_character_growth --dry-run --limit 5`
  - Expected when DB and LLM are available: prints a dry-run summary and writes
    one run document with no trait mutations.
- `venv\Scripts\python -m scripts.run_global_character_growth --apply --enable-trait-writes --limit 5`
  - Expected when DB and LLM are available: prints an apply summary, writes one
    run document, and inserts or updates only
    `global_character_growth_traits`.
- `venv\Scripts\python -m scripts.run_global_character_growth --apply --limit 5`
  - Expected: fails fast before DB writes because
    `--enable-trait-writes` is absent.

## Independent Plan Review

Independent plan review performed on 2026-05-11 from fresh-review posture:

- Coverage: every `Must Do` item maps to `Change Surface`, `Implementation
  Order`, `Progress Checklist`, and `Verification`.
- Placeholder scan: no unresolved placeholder, open decision, alternate path,
  or owner question is intentionally left for the implementation agent.
- Contract consistency: package name, collection names, feature flag, context
  key, test names, and CLI command are consistent across sections.
- Architecture alignment: background formation stays outside live chat; L2
  receives only compact promoted global guidance; user style, group style,
  RAG, dialog, adapters, and `character_state` writers remain out of scope.
- Review feedback addressed: the plan now records the verified queue priority,
  provisional calibration status, log-only shadow projection, inherited
  upstream-quality risk, filename alignment, and the future self-cognition
  merge point.
- Approval state: approved. Execution may start from Stage 1 under
  `development_plans/README.md`.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
Prefer a reviewer that did not implement the change. If no separate reviewer is
available, the active agent must reread this plan, inspect the full diff from a
fresh-review posture, and record that no separate reviewer was available.

Review scope:

- Project rules and style compliance for Python, tests, prompts, docs, CLI,
  and CJK prompt edits.
- Code quality and design weaknesses: ownership boundaries, hidden fallback
  paths, compatibility shims, prompt payload leaks, persistence risk, brittle
  fixtures, and avoidable blast radius.
- Plan alignment: `Must Do`, `Deferred`, autonomy boundaries, change surface,
  contracts, implementation order, verification gates, and acceptance criteria.
- Regression and handoff quality: focused tests, regression tests, static
  checks, live LLM trace, CLI smoke, execution evidence, and lifecycle updates.

Fix concrete findings directly only when the fix is inside the approved change
surface or this review gate explicitly allows review-only fixture or
documentation corrections. If a fix would cross the approved boundary or alter
the contract, stop and update the plan or request approval before changing
code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- `global_character_growth` owns candidate generation, validation, stable
  drift, runner orchestration, context projection, and an ICD-style README.
- `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED` defaults to `true`.
- `global_character_growth_traits` and `global_character_growth_runs` exist
  with required indexes.
- Dry-run and explicit apply CLI modes work and enforce write safety.
- The reflection worker respects `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED`; L2
  receives `promoted_global_growth` only from active promoted trait projection.
- Only active promoted traits enter cognition; lower bands remain audit-only.
- Run records include input-quality diagnostics and log-only shadow projection,
  and static checks prove shadow projection is absent from runtime prompts.
- No production code writes `character_state`, user style images, user memory,
  scheduler state, adapter output, or conversation history from this feature.
- Technology/domain competence and per-user or per-channel guidance are
  rejected as global personality growth.
- The new candidate-generation prompt and affected L2 prompt edits are Chinese
  integrated rewrites, not appended instruction blocks.
- Real LLM false-negative and false-positive cases are run individually,
  inspected, and recorded with durable traces.
- Every verification command in this plan passes or records an allowed skip.
- Independent code review is complete and approved.
- Execution evidence records tests, static checks, live LLM trace judgment,
  CLI smoke, residual risks, and final lifecycle status.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Sudden personality jump | Stable drift bands and prompt projection only at `promoted` | Drift tests and real POC-derived replay |
| Duplicate global guidance | Candidate overlap rejection and trait merge planning | Validation and drift merge tests |
| User-specific style becomes global | Scope assessment must be `global`; style-image surfaces are forbidden | Validation tests and static greps |
| Topic competence becomes personality | Prompt and validation reject domain-topic candidates | Technology/domain rejection tests and live LLM trace |
| Private detail leakage | Source evidence remains audit-only and prompt projection drops source details | Validation tests and context projection tests |
| Live latency regression | No response-path LLM call; context projection capped at three traits | Static greps and replay tests |
| Weak LLM invents IDs or writes | LLM only proposes candidates; deterministic code owns IDs and writes | Prompt contract and runner tests |
| Default-on worker write contention | Worker obeys busy probe, writes only new collections, and can be disabled with `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED=false` | Worker tests |
| Long ROI before visible behavior | Shadow projection exposes emerging guidance in run records while prompt projection remains promoted-only | Runner tests, CLI smoke, and static grep proving shadow stays out of prompts |
| Drift constants overfit the tiny POC | Constants are centralized, tested as provisional defaults, and retuned only through explicit follow-up evidence | Contract tests and execution evidence |
| Upstream promoted memory is sparse or noisy | Run records include input-quality diagnostics; this plan does not change upstream promotion | Runner tests and live LLM trace inspection |
| Future self-cognition overlap | Input contract stays limited to reflection-promoted memory; any widening requires a later plan | Static greps and plan review |

## Execution Evidence

Pre-implementation evidence:

- Decision-support artifacts:
  `test_artifacts/reflection_state_evolution/path_evaluation_report.json`,
  `test_artifacts/reflection_state_evolution/side_effect_simulation_report.json`,
  and
  `test_artifacts/reflection_state_evolution/side_effect_simulation_summary.md`.

Execution evidence:

- 2026-05-11 Stage 1 TDD baseline:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py tests\test_global_character_growth_validation.py tests\test_global_character_growth_drift.py tests\test_global_character_growth_runner.py tests\test_global_character_growth_context.py tests\test_global_character_growth_worker.py tests\test_global_character_growth_prompt_contracts.py tests\test_global_character_growth_module_boundary.py tests\test_global_character_growth_replay.py -q`
  exited `1` during collection with expected missing-module failures for
  `kazusa_ai_chatbot.global_character_growth`. The new deterministic tests
  now cover the approved contracts, validation gates, stable drift, runner,
  context merge, worker flag/busy behavior, prompt contract, module boundary,
  and L2 replay plumbing before production code exists.
- 2026-05-11 Stage 2 core internals:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py tests\test_global_character_growth_validation.py tests\test_global_character_growth_drift.py -q`
  passed `20` tests. Implemented package facade, default-on
  `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED`, centralized provisional drift
  constants, prompt-card/current-trait projection caps, input-quality
  diagnostics, source-detail-free runtime and shadow projections, deterministic
  validation gates for scope/privacy/domain/source/duplicate/caps, and stable
  drift planning that keeps the real-POC three-day candidate in `emerging` at
  approximately `0.363`.
- 2026-05-11 Stage 3 DB/bootstrap:
  `venv\Scripts\python -m pytest tests\test_db.py tests\test_global_character_growth_runner.py -q`
  passed `51` tests with `13` deselected. Implemented
  `db/global_character_growth.py`, `GlobalCharacterGrowthTraitDoc`,
  `GlobalCharacterGrowthRunDoc`, bootstrap collection registration, and the
  required eight indexes. Runner patched tests also passed for dry-run
  no-write behavior, explicit apply gating, apply trait writes through the
  named DB interface, skip records for empty eligible input, failed-LLM run
  records, and deterministic run ids.
- 2026-05-11 Stage 4 LLM/runner/CLI:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_prompt_contracts.py tests\test_global_character_growth_runner.py -q`
  passed `11` tests. Implemented the Simplified Chinese candidate-generation
  prompt with the required `_COGNITION_SUBCONSCIOUS_PROMPT`-style sections,
  consolidation-route LLM handler, prompt shape validation, background runner
  orchestration, and `scripts.run_global_character_growth` CLI. A prompt
  rendering bug was found and fixed by replacing only the `{character_name}`
  placeholder instead of formatting the whole prompt containing literal JSON
  braces.
- 2026-05-11 Stage 5 runtime projection/worker integration:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_context.py tests\test_reflection_cycle_stage1c_reflection_context.py tests\test_global_character_growth_worker.py tests\test_reflection_cycle_stage1c_worker.py -q`
  passed `15` tests. Implemented `promoted_global_growth` merging under the
  existing promoted reflection context, preserved empty-context behavior when
  no promoted traits exist, and wired the default-on growth pass after daily
  global reflection promotion with the existing busy probe and rollback flag.
  One initial test expectation incorrectly expected full timestamps in the L2
  projection; the root cause was a mismatch with the approved prompt-safe date
  projection, and the test was corrected to expect date-only `updated_at`.
- 2026-05-11 Stage 6 cognition integration:
  `venv\Scripts\python -m pytest tests\test_global_character_growth_replay.py tests\test_multi_source_cognition_stage_09_multimodal_input_sources.py tests\test_cognition_interaction_style_context.py -q`
  passed `23` tests. Rewrote the L2a consciousness prompt flow so
  `promoted_global_growth` is part of the existing
  `promoted_reflection_context` contract and is described as general global
  character-growth background, not current user facts, current commitments,
  style-image instructions, or a direct reply template. The updated L2 prompt
  fingerprint is `13407` bytes with SHA-256
  `0d23375541528afaf9e1db57b2201c8055a3942cc8cc9c789cb4532546cefade`.
  Additional targeted fingerprint checks passed for Stage 07 and Stage 08
  dry-run prompt tables after normalizing stale fixture values to the current
  prompt constants.
- 2026-05-11 Stage 7 live LLM tests:
  `venv\Scripts\python -m pytest -m live_llm tests\test_global_character_growth_live_llm.py::test_global_character_growth_live_accepts_stable_communication_growth -q -s`
  passed after one fixture correction. The first trace
  `test_artifacts\llm_traces\global_character_growth_live_llm__stable_communication_growth.json`
  showed the model correctly rejected three different one-day patterns as
  insufficient; the fixture was tightened to repeat one communication-growth
  pattern across three dates. The final accepted trace
  `test_artifacts\llm_traces\global_character_growth_live_llm__stable_communication_growth__20260511T094422183381Z.json`
  produced one Chinese `boundary_timing` candidate, cited all three source
  cards and dates, had `scope_assessment="global"`, `support_level="stable"`,
  `private_detail_risk="low"`, and deterministic validation accepted it with
  evidence strength `0.94`.
  `venv\Scripts\python -m pytest -m live_llm tests\test_global_character_growth_live_llm.py::test_global_character_growth_live_rejects_domain_and_user_specific_noise -q -s`
  passed and wrote
  `test_artifacts\llm_traces\global_character_growth_live_llm__domain_and_user_specific_noise.json`.
  The model emitted `no_action`, identified `domain_topic` and `user_specific`
  causes, and deterministic validation accepted zero candidates. Both traces
  were inspected individually; the drift constants remain provisional despite
  these passing live checks.
- 2026-05-11 Stage 8 full verification:
  Syntax command passed with exit `0`:
  `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\global_character_growth\__init__.py src\kazusa_ai_chatbot\global_character_growth\models.py src\kazusa_ai_chatbot\global_character_growth\projection.py src\kazusa_ai_chatbot\global_character_growth\llm.py src\kazusa_ai_chatbot\global_character_growth\validation.py src\kazusa_ai_chatbot\global_character_growth\drift.py src\kazusa_ai_chatbot\global_character_growth\runner.py src\kazusa_ai_chatbot\global_character_growth\context.py src\kazusa_ai_chatbot\db\global_character_growth.py src\scripts\run_global_character_growth.py`.
  Focused deterministic tests passed: `20` tests for contract, validation,
  and drift; `24` tests for runner, context, worker, prompt contracts, module
  boundary, and replay. Regression tests passed: `17` reflection-cycle tests,
  `13` memory-evolution tests, and `23` cognition/style/multimodal tests.
  Live LLM commands from Stage 7 were already run individually with
  `-m live_llm` and inspected as required. Static greps all returned no
  matches; the first command required PowerShell expansion of
  `tests\test_global_character_growth_*.py` because passing the wildcard path
  literally to `rg` produced Windows path error `123`.
  Manual CLI smoke:
  `venv\Scripts\python -m scripts.run_global_character_growth --apply --limit 5`
  failed before DB writes with `--apply requires --enable-trait-writes`;
  `venv\Scripts\python -m scripts.run_global_character_growth --dry-run --limit 5`
  completed with status `dry_run`, `5` eligible memory cards, `0` accepted
  candidates, `4` rejected candidates, `0` trait updates, input quality
  `adequate`, and `0` warnings; explicit apply with
  `--apply --enable-trait-writes --limit 5` completed with status `applied`,
  `5` eligible memory cards, `0` accepted candidates, `4` rejected candidates,
  `0` trait updates, input quality `adequate`, and `0` warnings.
- 2026-05-11 Stage 9 independent code review:
  No separate reviewer was available, so Codex reread the full plan and
  reviewed the full diff from a fresh-review posture against the plan,
  `local-llm-architecture`, `py-style`, `cjk-safety`, and
  `test-style-and-execution`. Findings fixed inside the approved change
  surface:
  1. Worker scheduling initially ran the growth pass after a promotion
     invocation even when promotion wrote no new memory. The worker now runs
     growth only when daily global promotion reports `succeeded_count > 0`,
     preserving the "after daily global promotion" intent and reducing queue
     contention. Added
     `test_worker_skips_growth_when_promotion_writes_no_memory`.
  2. Candidate projections treated missing `status` as active in several
     paths. Prompt-card filtering, current-trait projection, and shadow
     projection now require explicit `status="active"`.
  3. Malformed source timestamps could crash date-span diagnostics. Source
     date projection now validates the date prefix and drops invalid dates.
  4. Domain-topic validation only caught English terms. It now also rejects
     Chinese technology, product, food, cooking, location, hobby, tool-knowledge,
     and domain-competence terms.
  5. Candidate prompt wording included prompt-polluting and unrelated internal
     terms, including `角色`, L2, L3, Dialog, MongoDB/database/write-surface
     language. The candidate prompt was rewritten to avoid those words, the
     default injected label is now `当前主体`, prompt tests forbid the terms, and
     live fixture text was cleaned so prompt evaluation is not biased by those
     words.
  6. The live false-positive model response initially appended a Markdown
     summary outside JSON; the prompt now requires first character `{`, last
     character `}`, no Markdown or JSON-external prose, and the live LLM tests
     enforce strict `json.loads`.
  7. The live false-negative model initially emitted an English `trait_name`.
     The prompt now explicitly requires Simplified Chinese free text for
     `trait_name`, `guidance`, reason fields, and `summary`, and deterministic
     validation rejects accepted candidates whose `trait_name` or `guidance`
     lacks CJK text.

  Review rerun evidence after fixes:
  - Syntax:
    `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\global_character_growth\__init__.py src\kazusa_ai_chatbot\global_character_growth\models.py src\kazusa_ai_chatbot\global_character_growth\projection.py src\kazusa_ai_chatbot\global_character_growth\llm.py src\kazusa_ai_chatbot\global_character_growth\validation.py src\kazusa_ai_chatbot\global_character_growth\drift.py src\kazusa_ai_chatbot\global_character_growth\runner.py src\kazusa_ai_chatbot\global_character_growth\context.py src\kazusa_ai_chatbot\db\global_character_growth.py src\scripts\run_global_character_growth.py tests\test_global_character_growth_live_llm.py`
    passed with exit `0`.
  - Focused tests:
    `venv\Scripts\python -m pytest tests\test_global_character_growth_contract.py tests\test_global_character_growth_validation.py tests\test_global_character_growth_drift.py -q`
    passed `23` tests.
    `venv\Scripts\python -m pytest tests\test_global_character_growth_runner.py tests\test_global_character_growth_context.py tests\test_global_character_growth_worker.py tests\test_global_character_growth_prompt_contracts.py tests\test_global_character_growth_module_boundary.py tests\test_global_character_growth_replay.py -q`
    passed `27` tests.
  - Regression tests:
    `17` reflection-cycle tests, `13` memory-evolution tests, `23`
    cognition/style/multimodal tests, and `51` DB/runner tests passed.
  - Live LLM tests were rerun one at a time with strict JSON parsing and
    inspected. Final accepted trace:
    `test_artifacts\llm_traces\global_character_growth_live_llm__stable_communication_growth__20260511T100526818852Z.json`
    produced one Chinese `boundary_timing` candidate with Chinese
    `trait_name` and `guidance`, all three source cards and dates, no source
    detail leakage, and deterministic acceptance. Final false-positive trace:
    `test_artifacts\llm_traces\global_character_growth_live_llm__domain_and_user_specific_noise__20260511T100550601795Z.json`
    produced strict JSON, accepted zero candidates, and rejected the input as
    technology/domain and user-specific noise.
  - Static greps returned no matches for forbidden character-state writers,
    raw DB calls inside the growth package, live-path imports, user/style
    surfaces, L3/dialog projection, shadow projection in runtime prompts, and
    candidate-prompt pollution terms.
  - `git diff --check` reported only CRLF normalization warnings and no
    whitespace errors.
  - Final CLI smoke:
    unsafe apply failed before writes with
    `--apply requires --enable-trait-writes`; dry-run completed with status
    `dry_run`, `5` eligible memory cards, `0` accepted candidates, `3`
    rejected candidates, `0` trait updates, input quality `adequate`, and `0`
    warnings; explicit apply completed with status `applied`, `5` eligible
    memory cards, `0` accepted candidates, `2` rejected candidates, `0` trait
    updates, input quality `adequate`, and `0` warnings.

  Residual risks: drift constants remain provisional; upstream promoted
  reflection memory quality is still inherited; no separate reviewer was
  available. Approval status: approved for completion.
- Historical side-effect simulation: dry-run risk `low` with `0` writes; naive
  apply risk `high` with `45` appended rows and `2.305` growth ratio; strict
  apply risk `medium` with `3` candidates and `0.299` growth ratio.
- Final POC artifact:
  `test_artifacts/reflection_state_evolution/poc/reflection_state_evolution_poc_20260510T113801Z.json`.
- Stable drift artifact:
  `test_artifacts/reflection_state_evolution/poc/stable_drift_simulation_20260510T122208Z.json`.
- Stable drift result: one accepted real-DB-derived candidate had evidence
  strength `0.94`, observed dates `2026-05-05`, `2026-05-06`, and
  `2026-05-07`, final strength `0.363`, maturity band `emerging`, max daily
  delta `0.141`, promoted candidates `0`, held candidates `1`, and `7`
  additional confirming days needed.
- Interpretation: the candidate is useful evidence but must not enter runtime
  personality yet. Stable drift provides the production path for gradual,
  evidence-backed global character growth.

Implementation evidence must be appended here stage by stage. Checked progress
boxes are not evidence by themselves.
