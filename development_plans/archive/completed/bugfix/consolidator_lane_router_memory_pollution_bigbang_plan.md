# consolidator lane router memory pollution bigbang plan

## Summary

- Goal: replace the current broad consolidation extraction path with a
  big-bang lane-router and lane-specialist architecture that prevents new
  memory pollution across all memory lanes after the bad-data cleanup plans
  have completed.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `cjk-safety`,
  `test-style-and-execution`, `debug-llm`, and `database-data-pull`.
- Overall cutover strategy: bigbang after cleanup. No compatibility path, no
  dual consolidator, no fallback to the old mono fact/promise harvester.
- Highest-risk areas: overloading a weak local LLM router with similar
  semantic choices, adding deterministic semantic filters over user text,
  broadening changes outside consolidation, and creating new data structures
  where existing target/source fields are sufficient.
- Acceptance criteria: after cleanup, every new durable memory write is
  produced by one lane-owned specialist, reviewed by that lane's reviewer,
  structurally source-validated before persistence, and rejected when its only
  support is an internal/generated source that is not valid for the target
  lane. The planned real-LLM gating matrix in this document must pass before
  closure, including the accepted-character-self-guidance cases.

## Context

The current memory audit found the same root failure across memory lanes:
plausible learned data can be stored in the wrong durable lane or without
usable source provenance. The project goal explicitly allows learned and even
invented continuity when it emerges from interaction. The defect is not
non-canon content. The defect is memory-lane pollution.

The existing consolidation package already has useful hardening:

- `consolidation.core.call_consolidation_subgraph(...)` is the public entry.
- `consolidation.target.build_consolidation_target_plan(...)` builds
  deterministic targets.
- `consolidation.target.validate_write_intent(...)` validates target alias,
  target kind, and write lane before persistence.
- `consolidation.origin` carries identifier-only source metadata.

The remaining architecture problem is that the broad extraction path still
asks one LLM stage to do too much at once:

```text
completed episode
  -> broad fact/promise harvester
  -> broad evaluator/retry
  -> memory-unit extractor
  -> merge/rewrite/stability
  -> db_writer
```

That shape asks a weak local LLM to decide source authority, fact durability,
commitment acceptance, subject ownership, lane routing, duplicate handling, and
schema shape in large prompts. It also leaves source provenance fragile:
`memory_units.py` asks the extractor for `evidence_refs`, while
`db/user_memory_units.py` persists `source_refs`.

`web_agent3` and the complex task resolver show a better local pattern:

```text
small router
  -> one specialist
  -> bounded reviewer/evaluator
  -> deterministic structural validation/execution
```

This plan applies that pattern only to consolidation and memory-write
admission. It is intentionally sequenced after these cleanup plans:

- `shared_memory_lane_data_integrity_plan.md`
- `user_memory_units_lane_data_integrity_plan.md`
- `user_profiles_lane_data_integrity_plan.md`
- `interaction_style_images_lane_data_integrity_plan.md`
- `conversation_episode_state_lane_lifecycle_plan.md`
- `character_state_lane_integrity_plan.md`

The cleanup plans repair existing polluted rows. This plan prevents the same
pollution pattern from being recreated.

Cleanup precondition status as of 2026-07-03: the six lane cleanup plans are
completed and archived under `development_plans/archive/completed/bugfix/`.
Post-cleanup dry-runs report zero deterministic planned actions for every lane.
`user_memory_units` still reports 149 no-due active commitments for semantic
manual review; those rows were not mutated by deterministic cleanup because
valid ongoing rules can exist, and future writes are gated by this router plan.

The last-15-day conversation pull confirmed the user-observed pattern as a
general failure class even though the exact sample sentence was not found in
the export: accepted future-behavior requests can be represented as user
active commitments when the durable subject is actually Kazusa's own future
behavior. The corrected architecture therefore needs a character
self-guidance lane for accepted durable behavior rules, while preserving the
rule that ordinary chat cannot write generic shared/world lore.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, or reviewing
  this plan.
- `local-llm-architecture`: load before changing consolidation prompts,
  graph routing, specialist boundaries, source projection, or LLM budgets.
- `no-prepost-user-input`: load before changing fact, preference, accepted
  rule, promise, commitment, relationship, or user-memory persistence logic.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files containing CJK prompt text.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `debug-llm`: load before creating live LLM review artifacts or changing
  prompt-quality evaluation cases.
- `database-data-pull`: load before production data inspection, post-cleanup
  verification, or live DB validation.

## Mandatory Rules

- Execute this plan only after the six lane cleanup plans listed in `Context`
  have completed or after the user explicitly overrides that ordering.
- Treat this as a big-bang cutover. Do not preserve the old broad
  fact/promise harvester path, dual-write through it, or add compatibility
  wrappers for it.
- Keep the primary change surface inside `src/kazusa_ai_chatbot/consolidation`.
  Changes outside consolidation require the narrow justifications listed in
  `Change Surface`.
- Reuse existing structures first: `ConsolidatorState`,
  `ConsolidationTargetPlan`, `ConsolidationWriteIntent`,
  `consolidation_origin`, `source_refs`, `evidence_refs`,
  `privacy_review`, `source_reflection_run_ids`, existing DB helpers, and
  existing Cache2 invalidation events.
- Do not add a new persistent collection, generic memory bus, generic evidence
  ledger collection, adapter-visible memory API, RAG rewrite, cognition rewrite,
  or platform-specific branch.
- Deterministic code must not make semantic decisions from user text. It may
  validate structure, target eligibility, source-ref presence, source-ref class,
  timestamp parseability, cache invalidation scope, and persistence mechanics.
- Do not add keyword filters over `decontexualized_input`, `final_dialog`,
  generated facts, promises, style text, or relationship text.
- LLM specialists own semantic channel decisions. If an LLM decides an accepted
  user rule belongs in commitments, deterministic code must not rewrite it as
  a personality fact or drop it by local keyword logic.
- The router must not be asked to distinguish similar fine-grained user-memory
  meanings when a specialist can own that choice. The router chooses coarse
  lane tasks only.
- Lane reviewers may accept, correct, or reject lane candidates. They must not
  generate unrelated new memory.
- Reviewer correction is bounded to one pass per lane. Do not add open-ended
  retry loops.
- Internal/generated sources may support character state, relationship
  appraisal, or commitment acceptance when the lane contract allows them, but
  they must not be the only source for objective user facts or shared memory.
- Use `venv\Scripts\python` for Python verification commands.
- Use `apply_patch` for manual file edits.
- Use PowerShell `-LiteralPath` for filesystem paths that may contain spaces.
- Do not read `.env`.
- After any automatic context compaction, the parent or active execution agent
  must reread this entire plan before continuing implementation, verification,
  handoff, or final reporting.
- After signing off any major progress checklist stage, the parent or active
  execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the plan's `Independent Code Review` gate and record
  the result in `Execution Evidence`.

## Must Do

- Replace the current broad facts/promise consolidation path with a
  consolidation lane router plus lane-owned specialists and reviewers.
- Keep deterministic target planning before all lane work.
- Use deterministic target plan output to prune impossible lanes before the
  LLM router sees the task roster.
- Split consolidation semantics into lane-owned responsibilities:
  character state, relationship/profile header writes, user memory units,
  active commitments, character self-guidance, interaction style images, and
  shared-memory promotion admission.
- Add a character self-guidance lane for user-requested or conversation-derived
  future behavior that Kazusa explicitly accepts as her own durable behavior.
  Persist it through existing `memory` self-guidance structures with
  conversation source refs; do not create a new persistent collection.
- Keep ordinary chat consolidation out of generic shared/world fact writes.
  Only the character self-guidance lane may write conversation-extracted
  character behavior guidance, and approved reflection/shared-memory sources
  remain required for shared-memory promotion.
- Fix the `evidence_refs` / `source_refs` mismatch for user memory units so
  new user-memory writes persist usable `source_refs`.
- Require lane-specific source proof before durable write intent validation
  succeeds.
- Preserve existing successful target validation behavior for real users,
  group channels, character state, and internal audit targets.
- Delete or rewrite old mono-harvester references after the new lane path is
  wired.
- Add focused deterministic tests for routing, source validation, lane
  rejection, source-ref persistence, and no deterministic semantic filtering.
- Add one-at-a-time live LLM review cases for ambiguous lane decisions after
  deterministic tests pass.

## Deferred

- Do not perform the bad-data cleanup in this plan.
- Do not redesign RAG2, web_agent3, complex task resolver, cognition resolver,
  dialog generation, adapters, dispatcher, calendar scheduler, or reflection
  scheduling.
- Do not add new persistent memory collections or DB memory lanes. The
  internal `character_self_guidance` lane is an admission boundary over
  existing `memory` self-guidance storage, not a new collection.
- Do not add a new persistent provenance collection.
- Do not add generic source taxonomies beyond the source labels needed by
  current lane validation.
- Do not add keyword-based semantic gating in deterministic code.
- Do not change user-profile identity creation except where a test proves
  consolidation is still fabricating or mutating a user target.
- Do not change conversation-progress lifecycle beyond proving consolidation
  does not write episode-state memory.
- Do not add feature flags, compatibility shims, old/new dual paths, or
  rollback branches. Rollback is git revert plus database cleanup already
  handled by the lane cleanup plans.

## Cutover Policy

Overall strategy: bigbang after cleanup.

| Area | Policy | Instruction |
|---|---|---|
| Existing polluted data | migration | Complete the six lane cleanup plans first. This plan does not repair existing rows. |
| Consolidator graph | bigbang | Replace the broad fact/promise path with lane router, specialists, reviewers, and deterministic source validation in one cutover. |
| Old facts harvester | bigbang | Delete or rewrite old mono-harvester references. No fallback to old `facts_harvester` behavior. |
| User-memory source refs | bigbang | New writes persist `source_refs`; no new source-less writes are accepted. |
| Character self-guidance from accepted chat | bigbang | Direct accepted character behavior rules use the `character_self_guidance` lane and existing `memory` self-guidance storage with conversation source refs. |
| Shared memory promotion | bigbang | Ordinary chat consolidation cannot write generic shared/world facts. Approved promotion sources must use existing `memory_evolution` evidence/privacy fields. |
| Tests | bigbang | Replace tests that assert old harvester behavior with lane-router tests. |
| Existing DB rows | migration | Only verify the cleanup baseline and post-cutover no-new-pollution behavior. Do not mutate legacy rows in this plan. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative compatibility strategy by
  default.
- For `bigbang` areas, rewrite or remove legacy references instead of
  preserving them.
- For `migration` areas, use only the explicit cleanup-plan and verification
  gates listed here.
- Any change to cutover policy requires user approval before implementation.

## Target State

The completed consolidation path is:

```text
completed episode
  -> build_consolidation_origin
  -> build_consolidation_target_plan
  -> build prompt-safe source views from existing state/origin/rag_result
  -> consolidation lane router
  -> selected lane specialists
  -> lane reviewers
  -> deterministic source and write-intent validation
  -> target-specific persistence helpers
  -> Cache2 invalidation
```

The router receives only:

- prompt-safe episode summary already available to consolidation;
- the deterministic target plan;
- a compact lane roster containing only lanes allowed by the target plan;
- source-view summaries derived from existing `consolidation_origin`,
  `rag_result`, `final_dialog`, and `episode_trace_projection`.

The router outputs only coarse lane tasks:

```python
{
    "lane_tasks": [
        {
            "lane": "character_state"
                    | "relationship_profile"
                    | "user_memory_units"
                    | "active_commitment"
                    | "character_self_guidance"
                    | "interaction_style_image"
                    | "shared_memory_promotion",
            "reason": "short semantic reason",
            "source_keys": ["current_turn_user_message"]
        }
    ]
}
```

The router does not output database operations, target ids, write lanes,
memory-unit text, commitment text, affinity deltas, source refs, schedule
payloads, cache events, or persistence decisions.

Each specialist owns one lane and returns only that lane's candidate shape.
Each reviewer accepts, corrects, or rejects only candidates from its lane.
Deterministic code then validates shape, source refs, target/lane permission,
timestamps, and persistence mechanics.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Plan ordering | Execute after bad-data cleanup | A clean baseline is needed so post-cutover tests can detect new pollution. |
| Cutover | Bigbang | The old broad harvester is the pollution source. Dual paths would preserve the failure mode. |
| Change boundary | Consolidation first | The defect is write admission. Retrieval and cognition should not be redesigned to fix persistence ownership. |
| Data structures | Reuse existing structures | `ConsolidationTargetPlan`, `source_refs`, `evidence_refs`, and existing lane fields are enough. |
| Router scope | Coarse lane tasks only | Weak local LLMs are brittle when forced to choose between similar user-memory meanings. |
| Similar semantic choices | Let specialists and reviewers decide | User fact vs commitment vs relationship meaning is semantic. Deterministic code must not infer it. |
| Source validation | Deterministic by source class, not text meaning | Checking whether a source ref is internal, user-message, RAG evidence, reflection, or web evidence is structural. |
| Reviewer correction | One bounded pass | Satisfies review/correction without open-ended agent loops. |
| Character self-guidance | Accepted chat can write existing self-guidance memory | Invented data is valid when Kazusa accepts it as her own durable behavior; the defect is writing it to the wrong lane. |
| Shared memory | No ordinary chat writes for generic lore | Shared memory promotion already has `memory_evolution` evidence/privacy requirements and should stay promotion-owned. |
| Conversation episode state | Not a consolidator lane | It is short-term progress memory. This plan proves consolidation does not persist it as durable memory. |

## Contracts And Data Shapes

### Lane Names

The allowed consolidation lane names are:

```text
character_state
relationship_profile
user_memory_units
active_commitment
character_self_guidance
interaction_style_image
shared_memory_promotion
```

These names are internal consolidation lane names. They do not create new DB
collections.

### Source Views

Use source views as transient consolidation data only. Do not persist a new
generic source-view document. The source view is built from existing state:

```python
{
    "source_key": "current_turn_user_message",
    "source_kind": "user_message"
                   | "assistant_final_dialog"
                   | "internal_thought"
                   | "rag_memory_evidence"
                   | "rag_conversation_evidence"
                   | "rag_external_evidence"
                   | "rag_recall_evidence"
                   | "reflection_run"
                   | "episode_trace",
    "summary": "prompt-safe source summary",
    "source_refs": list[dict],
}
```

`source_refs` reuse the existing user-memory source-ref style. A current-turn
chat source ref must preserve the existing identifiers that are already in
`consolidation_origin`, including timestamp, platform message id, and active
turn conversation row ids when available.

### Lane Source Policy

Deterministic source policy validates source classes only:

| Lane | Accepted source classes | Rejected sole source classes |
|---|---|---|
| `character_state` | assistant final dialog, internal thought, episode trace, reflection run | user memory rows by themselves |
| `relationship_profile` | current user message, assistant final dialog, internal thought, conversation evidence, recall evidence | external web evidence by itself |
| `user_memory_units` | current user message, conversation evidence, user-memory merge candidates, recall evidence with durable or conversation support | assistant final dialog only, internal thought only, episode trace only |
| `active_commitment` | current user message plus assistant final dialog acceptance, recall evidence for existing commitments, action trace for accepted task ownership | user plan without assistant acceptance, internal thought only |
| `character_self_guidance` | current user message plus assistant final dialog acceptance, conversation evidence of prior accepted self-guidance, reflection evidence for promoted self-guidance | user request without assistant acceptance, internal thought only, RAG/internal knowledge only |
| `interaction_style_image` | reflection run ids, approved group/user style source payloads, current group/channel message plus assistant final-dialog acceptance when the target plan is group/channel style | ordinary chat fact extraction by itself, current-user-only request by itself |
| `shared_memory_promotion` | promoted reflection evidence with `evidence_refs` and `privacy_review` | ordinary chat, internal thought, assistant final dialog only |

This table does not decide what the user's text means. It only prevents a
candidate from being persisted when its declared source class is structurally
not allowed for that lane.

### Specialist Output

Specialists must emit lane-local candidate structures. They do not emit target
ids or persistence operations.

User memory unit candidate:

```python
{
    "candidate_id": str,
    "unit_type": "stable_pattern"
                 | "recent_shift"
                 | "objective_fact"
                 | "milestone",
    "fact": str,
    "subjective_appraisal": str,
    "relationship_signal": str,
    "source_keys": list[str],
}
```

Active commitment candidate:

```python
{
    "candidate_id": str,
    "unit_type": "active_commitment",
    "fact": str,
    "subjective_appraisal": str,
    "relationship_signal": str,
    "commitment_type": str,
    "due_at": str | None,
    "source_keys": list[str],
}
```

Relationship profile candidate:

```python
{
    "subjective_appraisals": list[str],
    "affinity_delta": int,
    "last_relationship_insight": str,
    "source_keys": list[str],
}
```

Character state candidate:

```python
{
    "mood": str,
    "global_vibe": str,
    "reflection_summary": str,
    "source_keys": list[str],
}
```

Character self-guidance candidate:

```python
{
    "candidate_id": str,
    "memory_name": str,
    "content": str,
    "memory_type": "defense_rule",
    "scope": "global_character",
    "source_keys": list[str],
    "expiry_timestamp": str | None,
    "confidence_note": str,
}
```

Interaction style candidate:

```python
{
    "target_style_kind": "user_style_image" | "group_channel_style_image",
    "overlay": dict,
    "source_reflection_run_ids": list[str],
}
```

Shared memory promotion candidate:

```python
{
    "memory_name": str,
    "content": str,
    "memory_type": "fact" | "defense_rule",
    "evidence_refs": list[dict],
    "privacy_review": dict,
    "confidence_note": str,
}
```

### Reviewer Output

Each lane reviewer returns:

```python
{
    "decision": "accept" | "correct" | "reject",
    "corrected_candidate": dict,
    "reason": str,
}
```

For `accept`, `corrected_candidate` is the original candidate after prompt-safe
projection. For `correct`, it is the corrected candidate in the same lane
shape. For `reject`, it is empty.

### Deterministic Validation

Deterministic validation may:

- reject unknown lane names;
- reject candidates whose `source_keys` do not resolve to source views;
- reject candidates with no lane-allowed source class;
- map accepted source views into `source_refs` or `evidence_refs`;
- validate target alias and write lane through `validate_write_intent(...)`;
- validate timestamps and lifecycle fields structurally;
- validate DB helper return shapes and cache invalidation events.

Deterministic validation must not:

- decide that text is or is not a preference, promise, relationship signal, or
  user fact;
- infer commitment type from keywords;
- rewrite a user's instruction into another memory lane;
- drop an LLM candidate because local code dislikes the generated semantics.

## LLM Call And Context Budget

All new or changed calls are background consolidation calls, not live
response-path calls.

Assume a 50k-token context cap. Use conservative character budgeting: 4
characters per token for Chinese-heavy prompts and payloads.

| Stage | Before | After | Budget and cap |
|---|---:|---:|---|
| Target planning | 0 LLM | 0 LLM | Deterministic only. |
| Lane router | 0 LLM | 1 LLM | System prompt plus compact source views, max 12k chars. |
| Character state | 1 LLM | 1 LLM | Reuse current role, reduce prompt to one lane. |
| Relationship/profile | 1 LLM | 1 specialist + 1 reviewer only when routed | Max 14k chars each; reviewer runs only for non-empty output. |
| Fact/promise harvester | 1-4 LLM | 0 LLM | Removed. Replaced by lane specialists. |
| User memory units | 1 extractor + per-candidate merge/rewrite/stability | 1 specialist + 1 reviewer + existing merge/rewrite/stability only for accepted candidates | Max 3 candidates per turn retained. |
| Commitments | broad harvester output | 1 specialist + 1 reviewer when routed | Max 8k chars each. |
| Character self-guidance | absent or polluted through commitments | 1 specialist + 1 reviewer when routed | Max 8k chars each; requires assistant final-dialog acceptance source. |
| Interaction style | existing source-specific path | lane validator/reviewer only when style payload exists | No ordinary chat style extraction added. |
| Shared memory promotion | reflection-owned | no ordinary chat call | Existing promotion path only, with source validation. |

Hard caps:

- lane router emits at most four lane tasks;
- each specialist emits at most three candidates;
- each lane receives at most one reviewer pass;
- no reviewer retry loop;
- no JSON repair prompt beyond existing malformed-JSON repair helper;
- after JSON parses, malformed rows are dropped structurally with logs.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/consolidation/README.md`
  - Update the ICD to describe the lane-router big-bang contract.
- `src/kazusa_ai_chatbot/consolidation/core.py`
  - Replace the old graph wiring with target-plan -> lane-router ->
    lane-specialists -> reviewers -> db-writer flow.
- `src/kazusa_ai_chatbot/consolidation/schema.py`
  - Add transient internal state keys for lane tasks, source views, lane
    outputs, and reviewer decisions. Do not add persistent DB schemas here.
- `src/kazusa_ai_chatbot/consolidation/target.py`
  - Extend write-intent validation only for structural source-policy checks
    that depend on lane, target kind, and source class.
- `src/kazusa_ai_chatbot/consolidation/persistence.py`
  - Persist only reviewed and source-validated lane outputs. Preserve existing
    DB helper ownership and Cache2 invalidation behavior.
- `src/kazusa_ai_chatbot/consolidation/memory_units.py`
  - Convert the current broad extractor path into user-memory-unit specialist
    behavior, map accepted source views into persisted `source_refs`, and keep
    merge/rewrite/stability logic behind reviewed candidates.
- `src/kazusa_ai_chatbot/consolidation/reflection.py`
  - Keep character-state and relationship semantics as lane-owned specialists,
    or move their prompt blocks into new lane modules during the cutover.
- `src/kazusa_ai_chatbot/consolidation/images.py`
  - Keep character self-image behavior source-bound to character-state output.
- `src/kazusa_ai_chatbot/db/user_memory_units.py`
  - Accept already reviewed source refs through the existing `source_refs`
    field. This is allowed outside consolidation because it fixes the existing
    storage-field mismatch.
- `src/kazusa_ai_chatbot/reflection_cycle/context.py`
  - Extend the existing promoted self-guidance context projection to include
    source-validated conversation-extracted character self-guidance from the
    existing `memory` collection under the same caps. This is allowed outside
    consolidation because accepted self-guidance must be retrievable after it
    is correctly stored.
- `src/kazusa_ai_chatbot/reflection_cycle/README.md`
  - Document that accepted conversation-extracted character self-guidance can
    feed the same bounded self-guidance context path as promoted reflection
    guidance.
- Existing consolidation tests under `tests/`
  - Replace old harvester behavior expectations with lane-router expectations.

### Create

- `src/kazusa_ai_chatbot/consolidation/source_views.py`
  - Build transient source views from existing origin, RAG, final dialog, and
    episode trace data.
- `src/kazusa_ai_chatbot/consolidation/lane_router.py`
  - Own the coarse lane-router prompt, LLM call, parsing, and structural
    output validation.
- `src/kazusa_ai_chatbot/consolidation/lane_review.py`
  - Own shared reviewer parsing and structural validation utilities. Do not
    place lane semantics here.
- `src/kazusa_ai_chatbot/consolidation/commitments.py`
  - Own active-commitment specialist and reviewer.
- `src/kazusa_ai_chatbot/consolidation/character_self_guidance.py`
  - Own accepted character self-guidance specialist and reviewer. It writes
    through existing memory self-guidance storage and requires user-message
    plus assistant-final-dialog acceptance evidence.
- `src/kazusa_ai_chatbot/consolidation/source_policy.py`
  - Own deterministic source-class validation and source-ref mapping. It must
    not inspect user text semantics.
- `tests/test_consolidation_lane_router_contract.py`
  - Focused router contract tests.
- `tests/test_consolidation_source_policy.py`
  - Source-class and source-ref validation tests.
- `tests/test_consolidation_lane_bigbang_integration.py`
  - Integration tests for all lane outputs reaching persistence only through
    reviewed source-validated write intents.
- `tests/test_consolidation_memory_write_use_cases_live_llm.py`
  - One-at-a-time live LLM gating tests for the golden memory-write use cases
    listed in this plan. Each test records artifacts and must be inspected
    before closure.

### Delete Or Rewrite

- `src/kazusa_ai_chatbot/consolidation/facts.py`
  - Delete the mono harvester or rewrite the file into lane-specific
    components. Do not preserve the broad facts/promise extractor.

### Keep

- RAG2 helper-agent architecture.
- web_agent3 and complex task resolver architecture.
- Cognition resolver and dialog generation.
- Adapter and dispatcher boundaries.
- Existing database collections.
- Existing cleanup plans and migration scripts.

## Overdesign Guardrail

- Actual problem: one broad consolidation path can admit plausible learned
  data into the wrong memory lane or without source proof.
- Minimal change: replace the broad consolidation semantic stage with coarse
  lane routing, one-lane specialists, one-lane reviewers, and deterministic
  source/write validation using existing structures.
- Ownership boundaries: LLM router and specialists own semantic lane
  proposals; LLM reviewers own semantic review/correction; deterministic code
  owns target eligibility, source-class validation, timestamps, persistence,
  cache invalidation, and DB helper calls.
- Rejected complexity: no new persistent evidence ledger, no memory bus, no
  new collection, no generic plugin system, no compatibility path, no feature
  flag, no adapter changes, no RAG redesign, no cognition rewrite, and no
  deterministic keyword classifier.
- Evidence threshold: add broader memory-source abstractions only if
  post-cutover tests show two or more non-consolidation writers cannot enforce
  source validation through existing lane fields.

## Agent Autonomy Boundaries

- The responsible agent may choose local implementation mechanics only when
  they preserve this plan's contracts.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat changes outside consolidation as
  high-scrutiny changes and justify each one against `Change Surface`.
- The responsible agent may remove old mono-harvester code after references
  are replaced and tests prove the lane path.
- If equivalent structural helper behavior already exists, reuse or move it
  instead of duplicating it.
- The responsible agent must not perform unrelated cleanup, formatting churn,
  dependency upgrades, prompt rewrites, or broad refactors.
- If this plan and code disagree, preserve the plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

1. Parent verifies the cleanup precondition.
   - Check that the six lane cleanup plans are completed or that the user has
     explicitly overridden the ordering.
   - Record the result in `Execution Evidence`.
2. Parent adds focused failing tests for source views and source policy.
   - File: `tests/test_consolidation_source_policy.py`.
   - Expected before implementation: missing module or missing validation.
3. Parent adds focused failing tests for the lane router contract.
   - File: `tests/test_consolidation_lane_router_contract.py`.
   - Expected before implementation: missing module or old graph contract.
4. Parent adds integration tests for reviewed lane outputs.
   - File: `tests/test_consolidation_lane_bigbang_integration.py`.
   - Expected before implementation: old mono path still writes or source refs
     are missing.
5. Parent drafts the golden live-LLM memory-write gating tests.
   - File: `tests/test_consolidation_memory_write_use_cases_live_llm.py`.
   - Cases and expectations are the matrix in
     `Golden Memory Write Use Cases And Expected Outcomes`.
   - Expected before implementation: old mono path routes at least one case to
     the wrong lane, omits source refs, or lacks the new lane contract.
6. Parent starts exactly one production-code subagent.
   - The subagent owns production changes inside the approved change surface.
7. Production-code subagent implements source views and source policy.
   - No LLM prompts in this step.
8. Production-code subagent implements lane router and lane specialists.
   - Keep prompts small and lane-local.
9. Production-code subagent rewires `core.py` to the new graph.
   - Remove old broad harvester wiring in the same cutover.
10. Production-code subagent updates persistence and user-memory source refs.
   - Persist source refs only after reviewer and source-policy validation.
11. Parent runs focused tests and static checks.
12. Parent runs one-at-a-time live LLM review cases.
13. Parent starts exactly one independent code-review subagent.
14. Parent remediates review findings only inside approved scope and reruns
    affected verification.

## Execution Model

- Parent agent owns orchestration, test code, verification, execution
  evidence, review feedback remediation, lifecycle updates, and final sign-off.
- Parent agent establishes the focused test contract first and records the
  expected failure or baseline before production implementation starts.
- Production-code subagent: exactly one native subagent, started after the
  focused test contract is established; owns production code changes only;
  does not edit tests unless the parent explicitly directs it; closes after
  planned production code changes are complete, excluding review fixes.
- Parent agent may continue integration tests, regression tests, static checks,
  and validation work while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, diff, and evidence; reports
  findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - cleanup precondition and focused test contract established
  - Covers: implementation steps 1-5.
  - Verify: planned tests fail for missing/new contract before production code
    changes, or record the current baseline if the behavior already exists.
  - Evidence: record commands, expected failures, and the drafted golden
    live-LLM case names in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `Codex/2026-07-03` after verification and evidence are recorded.
- [x] Stage 2 - source-view and source-policy foundation implemented
  - Covers: implementation step 7.
  - Verify: `venv\Scripts\python -m pytest tests/test_consolidation_source_policy.py -q`.
  - Evidence: record changed files and test output.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: `Codex/2026-07-03` after source-policy tests and evidence were recorded.
- [x] Stage 3 - lane router, specialists, and reviewers implemented
  - Covers: implementation step 8.
  - Verify: `venv\Scripts\python -m pytest tests/test_consolidation_lane_router_contract.py -q`.
  - Evidence: record changed files, prompt render checks, and test output.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: `Codex/2026-07-03` after router contract tests and evidence were recorded.
- [x] Stage 4 - consolidator graph and persistence cutover complete
  - Covers: implementation steps 9-10.
  - Verify: `venv\Scripts\python -m pytest tests/test_consolidation_lane_bigbang_integration.py -q`.
  - Evidence: record old-path static grep results and integration output.
  - Handoff: next agent starts at Stage 5.
  - Sign-off: `Codex/2026-07-03` after integration tests and old-path grep evidence were recorded.
- [x] Stage 5 - full verification and live LLM review complete
  - Covers: implementation steps 11-12.
  - Verify: all commands in `Verification`.
  - Evidence: record deterministic tests, static checks, every golden live-LLM
    artifact, and human inspection status.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `Codex/2026-07-03` after static checks, deterministic tests, regression, and 26 live LLM cases passed.
- [x] Stage 6 - independent code review complete
  - Covers: implementation steps 13-14.
  - Verify: review findings resolved or documented, affected tests rerun.
  - Evidence: record reviewer identity, findings, fixes, rerun commands, and
    approval status.
  - Handoff: plan can be considered for completion only after this checkpoint.
  - Sign-off: `Codex/2026-07-03` after independent review findings were remediated and affected tests reran.

## Golden Memory Write Use Cases And Expected Outcomes

These cases define the required live-LLM gating contract for
`tests/test_consolidation_memory_write_use_cases_live_llm.py`. Draft these
tests before production execution. Each case runs as a separate live-LLM test
function with debug artifacts inspected by a human reviewer before plan
closure. The assertions are lane and source-contract assertions; they must not
be implemented through deterministic keyword matching over user text.

1. `test_live_case_01_user_objective_fact_routes_user_memory`
   - Input: `我现在在奥克兰工作。`
   - Expected: one `user_memory_units` `objective_fact` candidate with
     non-empty current-turn `source_refs`.
   - Forbidden: `character_state`, `character_self_guidance`,
     `shared_memory_promotion`, or source-less user memory.
2. `test_live_case_02_user_preference_routes_user_memory`
   - Input: `我不喜欢太甜的奶茶。`
   - Expected: `user_memory_units` `stable_pattern` or `objective_fact` for the
     user's preference with current-turn `source_refs`.
   - Forbidden: active commitment, character preference, or shared memory.
3. `test_live_case_03_user_milestone_routes_user_memory`
   - Input: `我昨天通过驾照考试了。`
   - Expected: `user_memory_units` `milestone` with current-turn `source_refs`.
   - Forbidden: relationship-only write or shared memory.
4. `test_live_case_04_user_recurring_pattern_routes_user_memory`
   - Input: `我每周三晚上都去打羽毛球。`
   - Expected: `user_memory_units` `stable_pattern` with current-turn
     `source_refs`.
   - Forbidden: active commitment or interaction-style image.
5. `test_live_case_05_user_recent_shift_routes_user_memory`
   - Input: `最近我改成早睡了。`
   - Expected: `user_memory_units` `recent_shift` with current-turn
     `source_refs`.
   - Forbidden: character state or shared memory.
6. `test_live_case_06_relationship_signal_routes_relationship_profile`
   - Input: `你刚才那样说让我有点被敷衍。`
   - Expected: `relationship_profile` candidate supported by user message and
     assistant final dialog when available.
   - Forbidden: objective user fact, character self-guidance, or shared memory.
7. `test_live_case_07_user_specific_address_rule_routes_commitment`
   - Input: `以后你跟我说话叫我阿然，好吗？` with Kazusa accepting in final
     dialog.
   - Expected: `active_commitment` scoped to the current user with user-message
     plus assistant-final-dialog acceptance sources.
   - Forbidden: `character_self_guidance`, `user_memory_units`, or shared
     memory.
8. `test_live_case_08_user_specific_answer_style_routes_commitment`
   - Input: `以后给我代码建议时先说结论。` with Kazusa accepting in final
     dialog.
   - Expected: current-user `active_commitment` with acceptance evidence.
   - Forbidden: global character behavior guidance or user objective fact.
9. `test_live_case_09_accepted_reminder_routes_commitment`
   - Input: `明天晚上八点提醒我发报告。` with Kazusa accepting ownership.
   - Expected: `active_commitment` with structurally parsed `due_at` when
     available and acceptance evidence.
   - Forbidden: generic user memory or shared memory.
10. `test_live_case_10_cancelled_commitment_updates_lifecycle`
    - Input: `不用提醒我发报告了。` with Kazusa accepting cancellation.
    - Expected: existing commitment lifecycle update or rejection of a new
      write when no matching commitment exists.
    - Forbidden: new objective user fact or character self-guidance.
11. `test_live_case_11_accepted_repetition_rule_routes_character_self_guidance`
    - Input: `千纱，以后如果你看到有人复读，你也可以加入他们的复读。` with
      Kazusa accepting it as her own general future behavior.
    - Expected: `character_self_guidance` using existing `memory`
      self-guidance storage, `memory_type="defense_rule"`, and user-message
      plus assistant-final-dialog acceptance sources.
    - Forbidden: `user_memory_units`, current-user `active_commitment`, generic
      shared/world memory, or source-less memory.
12. `test_live_case_12_unaccepted_repetition_rule_writes_nothing`
    - Input: the same repetition request, with Kazusa declining, deflecting, or
      not accepting durable future behavior.
    - Expected: no durable memory write for the request.
    - Forbidden: user memory, active commitment, character self-guidance, or
      shared memory.
13. `test_live_case_13_global_character_response_rule_routes_self_guidance`
    - Input: `以后你也可以偶尔用“收到”回应大家。` with Kazusa accepting it
      as a global behavior option.
    - Expected: `character_self_guidance` with acceptance sources.
    - Forbidden: current-user-only commitment or user preference.
14. `test_live_case_14_user_scoped_directness_rule_routes_commitment`
    - Input: `以后和我聊天时用更直接的语气。` with Kazusa accepting it for
      the current user.
    - Expected: current-user `active_commitment`.
    - Forbidden: global `character_self_guidance` or shared memory.
15. `test_live_case_15_group_specific_norm_routes_group_style`
    - Input: group/channel context plus `在这个群里大家玩接龙时你可以跟一轮。`
      with Kazusa accepting the group norm.
    - Expected: `interaction_style_image` for `group_channel_style_image` with
      group/channel target and acceptance sources.
    - Forbidden: current-user memory, current-user commitment, global character
      self-guidance, or shared memory.
16. `test_live_case_16_one_turn_roleplay_instruction_writes_nothing`
    - Input: `这局先用猫娘口吻回答。`
    - Expected: no durable consolidation write.
    - Forbidden: active commitment, character self-guidance, or interaction
      style image.
17. `test_live_case_17_user_invented_character_trait_routes_character_state`
    - Input: `千纱其实很擅长吐槽冷场，对吧？` with Kazusa accepting the
      invented trait as self-continuity.
    - Expected: `character_state` or existing character self-image output with
      conversation sources.
    - Forbidden: user memory or shared/world fact.
18. `test_live_case_18_internal_thought_cannot_create_user_fact`
    - Input: no new user message; internal thought says the user seems busy.
    - Expected: relationship appraisal may be proposed only if lane policy
      allows the source; no objective user-memory write.
    - Forbidden: source-less `user_memory_units`.
19. `test_live_case_19_external_rag_answer_does_not_write_user_memory`
    - Input: user asks for an external fact answer; RAG/web evidence supplies
      the answer.
    - Expected: no user-memory write unless the user also states a durable
      personal fact.
    - Forbidden: shared memory promotion from ordinary chat answer evidence.
20. `test_live_case_20_recalled_user_fact_merge_keeps_sources`
    - Input: user restates or updates a known personal fact supported by
      conversation recall.
    - Expected: `user_memory_units` merge/update with current-turn and recall
      source refs preserved.
    - Forbidden: source-ref loss or duplicate source-less row.
21. `test_live_case_21_third_party_fact_does_not_pollute_current_user`
    - Input: `小李喜欢低糖奶茶。`
    - Expected: no current-user `user_memory_units` objective fact.
    - Forbidden: shared memory or current-user profile pollution.
22. `test_live_case_22_reflection_promotion_routes_shared_memory`
    - Input: approved reflection/shared-memory promotion payload with
      `evidence_refs` and `privacy_review`.
    - Expected: `shared_memory_promotion` only.
    - Forbidden: user-memory, active commitment, or character-state write.
23. `test_live_case_23_ordinary_chat_world_lore_writes_no_shared_memory`
    - Input: `你要记住这个设定：蓝星大陆有七个王国。`
    - Expected: no generic shared/world memory from ordinary chat.
    - Forbidden: `shared_memory_promotion` without approved promotion evidence.
24. `test_live_case_24_debug_user_without_platform_id_does_not_fabricate_profile`
    - Input: debug-channel conversation with no stable platform identifier.
    - Expected: no fabricated production `user_profile` or profile-header
      mutation unless the target plan already has an eligible real user target.
    - Forbidden: no-platform production profile creation.
25. `test_live_case_25_reflection_user_style_routes_user_style_image`
    - Input: approved user-style reflection payload.
    - Expected: `interaction_style_image` for `user_style_image` only.
    - Forbidden: user-memory or relationship-profile write.
26. `test_live_case_26_episode_progress_does_not_become_durable_memory`
    - Input: `我们先做到第三步，下一轮继续。`
    - Expected: no durable consolidation write; episode-progress lifecycle, if
      present, remains outside the durable memory lanes.
    - Forbidden: user-memory milestone, active commitment, or shared memory.

## Verification

### Static Checks

- `venv\Scripts\python -m py_compile src/kazusa_ai_chatbot/consolidation/*.py`
  succeeds.
- `rg "facts_harvester|fact_harvester_evaluator" src/kazusa_ai_chatbot/consolidation tests`
  returns no production-path references to the old mono harvester. Test names
  may mention the old path only when asserting its removal.
- `rg "evidence_refs" src/kazusa_ai_chatbot/consolidation src/kazusa_ai_chatbot/db/user_memory_units.py`
  shows user-memory evidence being mapped to persisted `source_refs`, not
  dropped.
- `rg "if .*decontexualized_input|if .*final_dialog|in action|in fact" src/kazusa_ai_chatbot/consolidation`
  must not reveal deterministic keyword or text-semantic gates. Structural
  type checks and prompt payload construction are allowed.

### Deterministic Tests

- `venv\Scripts\python -m pytest tests/test_consolidation_source_policy.py -q`
- `venv\Scripts\python -m pytest tests/test_consolidation_lane_router_contract.py -q`
- `venv\Scripts\python -m pytest tests/test_consolidation_lane_bigbang_integration.py -q`
- `venv\Scripts\python -m pytest tests/test_consolidation_target.py -q`
  or the current consolidation target test file if renamed before execution.

### Regression Tests

- `venv\Scripts\python -m pytest -m "not live_db and not live_llm" -q`

### Live LLM Review

Run each case in `Golden Memory Write Use Cases And Expected Outcomes` one at
a time with output inspected. Each test must use the existing real-LLM test
style for this repository: no broad batching, artifact capture, loose
structural assertions for lane/source shape, and human inspection of the
model's semantic decision.

Required command pattern:

```powershell
venv\Scripts\python -m pytest tests/test_consolidation_memory_write_use_cases_live_llm.py::test_live_case_01_user_objective_fact_routes_user_memory -q -s -m live_llm
```

Repeat the command for cases 02 through 26 by replacing the test function name.
The plan cannot close until all 26 cases pass or are explicitly accepted with
documented human-inspection notes.

Artifacts go under `test_artifacts/llm_traces/` or the current debug-LLM
artifact directory used by the repo.

### Live DB / Post-Cleanup Validation

Only after cleanup and with explicit live DB availability:

- Run a dry-run diagnostic that records counts of new writes by lane after the
  cutover smoke cases.
- Verify no new active `user_memory_units` rows are source-less.
- Verify no ordinary chat consolidation wrote generic shared/world `memory`
  rows.
- Verify accepted character self-guidance writes, when present, use existing
  `memory` self-guidance storage with conversation source refs and do not
  create current-user memory pollution.
- Verify Cache2 invalidation events match the actual durable writes.

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
- Regression and handoff quality, including cleanup-plan precondition evidence,
  focused and regression tests, live LLM review artifacts, and path-safe
  commands for directories containing spaces.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface or this review gate explicitly allows
review-only fixture/documentation corrections. If a fix would cross the
approved boundary or alter the contract, stop and update the plan or request
approval before changing code.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- The six lane cleanup plans are completed or the user has explicitly
  overridden the ordering.
- The old broad fact/promise harvester no longer participates in runtime
  consolidation.
- Consolidation uses target-plan -> lane-router -> lane-specialist ->
  lane-reviewer -> source/write validation -> persistence.
- Router output is limited to coarse lane tasks and cannot emit database
  operations or memory text.
- Each specialist owns exactly one memory lane.
- Each reviewer accepts, corrects, or rejects only one lane's candidates.
- New user-memory writes persist non-empty source refs.
- Accepted Kazusa self-guidance writes use `character_self_guidance` and
  existing `memory` self-guidance storage, not current-user memory.
- Internal/generated-only sources cannot create objective user facts or
  generic shared/world memory.
- Deterministic code contains no user-text keyword classifier for fact,
  promise, preference, relationship, or style semantics.
- All 26 golden live-LLM memory-write use cases are implemented, run one at a
  time, pass their lane/source assertions, and have inspection artifacts
  recorded.
- Deterministic tests, regression tests, static checks, live LLM review cases,
  and independent code review pass with evidence recorded.

## Execution Evidence

- Cleanup precondition: completed on 2026-07-03. Archived cleanup records:
  `development_plans/archive/completed/bugfix/shared_memory_lane_data_integrity_plan.md`,
  `development_plans/archive/completed/bugfix/user_memory_units_lane_data_integrity_plan.md`,
  `development_plans/archive/completed/bugfix/user_profiles_lane_data_integrity_plan.md`,
  `development_plans/archive/completed/bugfix/interaction_style_images_lane_data_integrity_plan.md`,
  `development_plans/archive/completed/bugfix/conversation_episode_state_lane_lifecycle_plan.md`,
  and
  `development_plans/archive/completed/bugfix/character_state_lane_integrity_plan.md`.
  Cleanup apply results: 830 user-memory rows marked legacy-unverified, 1
  orphan user-memory row archived, 149 expired episode-state rows closed, and
  0 blocked cleanup actions. Post-cleanup dry-runs show 0 deterministic
  planned actions for all lanes.
- Focused test baseline: added
  `tests/test_consolidation_source_policy.py`,
  `tests/test_consolidation_lane_router_contract.py`, and
  `tests/test_consolidation_lane_bigbang_integration.py`.
  `venv\Scripts\python.exe -m py_compile tests\test_consolidation_source_policy.py tests\test_consolidation_lane_router_contract.py tests\test_consolidation_lane_bigbang_integration.py tests\test_consolidation_memory_write_use_cases_live_llm.py`
  exited 0. Baseline command
  `venv\Scripts\python.exe -m pytest tests\test_consolidation_source_policy.py tests\test_consolidation_lane_router_contract.py tests\test_consolidation_lane_bigbang_integration.py -q`
  produced 18 expected failures and 1 pass against the old architecture:
  missing `consolidation.source_policy`, missing `consolidation.lane_router`,
  old `facts_harvester` still wired in `core.py`, and no
  `character_self_guidance` write lane on the character target.
- Golden live-LLM case contract: added
  `tests/test_consolidation_memory_write_use_cases_live_llm.py` with 26
  individually named live-LLM cases from this plan. Collection command
  `venv\Scripts\python.exe -m pytest tests\test_consolidation_memory_write_use_cases_live_llm.py --collect-only -q -m live_llm`
  collected all 26 tests.
- Production-code subagent: exactly one production-code subagent was used,
  `019f24f1-8383-7563-8acc-4bd00433a5d2` / Popper. It implemented the
  approved consolidation cutover surface: source-policy/source-view support,
  lane router, lane-owned specialists/reviewers, `core.py` graph cutover,
  `character_self_guidance`, source-ref persistence, and old mono-harvester
  removal.
- Parent remediation after review stayed inside approved scope. The final
  implementation uses `source_policy.py`, `lane_router.py`,
  `character_self_guidance.py`, and existing lane modules instead of creating
  separate `source_views.py`, `lane_review.py`, and `commitments.py` files;
  the plan's one-lane specialist/reviewer/source-validation contracts are
  preserved without adding extra modules or persistent structures.
- Static check results:
  `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\consolidation\__init__.py src\kazusa_ai_chatbot\consolidation\target.py src\kazusa_ai_chatbot\consolidation\source_policy.py src\kazusa_ai_chatbot\consolidation\schema.py src\kazusa_ai_chatbot\consolidation\reflection.py src\kazusa_ai_chatbot\consolidation\persistence.py src\kazusa_ai_chatbot\consolidation\origin_policy.py src\kazusa_ai_chatbot\consolidation\origin.py src\kazusa_ai_chatbot\consolidation\memory_units.py src\kazusa_ai_chatbot\consolidation\lane_router.py src\kazusa_ai_chatbot\consolidation\images.py src\kazusa_ai_chatbot\consolidation\group_channel.py src\kazusa_ai_chatbot\consolidation\core.py src\kazusa_ai_chatbot\consolidation\character_self_guidance.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py`
  exited 0. `rg "facts_harvester|fact_harvester_evaluator" src\kazusa_ai_chatbot\consolidation tests`
  returns only the removal assertion in
  `tests/test_consolidation_lane_bigbang_integration.py`. The deterministic
  semantic-gate grep
  `rg "if .*decontexualized_input|if .*final_dialog|in action|in fact" src\kazusa_ai_chatbot\consolidation`
  returns no output. `git diff --check` exits 0; remaining messages are
  line-ending warnings only.
- Focused deterministic test results:
  `venv\Scripts\python.exe -m pytest tests\test_consolidation_lane_bigbang_integration.py tests\test_consolidation_lane_router_contract.py tests\test_consolidation_source_policy.py -q --tb=short`
  passed 30 tests.
  `venv\Scripts\python.exe -m pytest tests\test_consolidator_source_aware_payloads.py tests\test_llm_time_payload_projection.py tests\test_user_memory_units_rag_flow.py -q --tb=short`
  passed 37 tests.
  `venv\Scripts\python.exe -m pytest tests\test_consolidator_efficiency.py tests\test_consolidator_origin_selection.py tests\test_consolidator_origin_policy_db_writer.py tests\test_db_writer_cache2_invalidation.py -q --tb=short`
  passed 16 tests.
  `venv\Scripts\python.exe -m pytest tests\test_consolidation_module_boundary.py -q --tb=short`
  passed 3 tests.
  Earlier focused target/origin verification passed 37 tests, and the
  service/memory focused batch passed 84 tests.
- Regression test results:
  `venv\Scripts\python.exe -m pytest -m "not live_db and not live_llm" -q --tb=short --disable-warnings`
  passed with `2693 passed, 2 skipped, 408 deselected`.
- Live LLM review artifacts: all 26 golden cases ran one at a time using
  `venv\Scripts\python.exe -m pytest tests\test_consolidation_memory_write_use_cases_live_llm.py::<case> -q -s -m live_llm --tb=short`
  and all 26 passed. Artifacts were written under
  `test_artifacts/llm_traces/` for cases 01 through 26 and inspected during
  execution. Case 24 produced the expected router warning that a
  `user_memory_units` task was dropped because that lane was not in the target
  roster for an identity-free debug user.
- Live DB/post-cleanup validation: cleanup reports are recorded under
  `test_artifacts/`. The 26 live-LLM gating cases used dry-run execution and
  did not mutate production DB rows. No additional live DB mutation smoke was
  run; the agreed robustness gate for this plan was the one-at-a-time dry-run
  live-LLM matrix plus deterministic and regression tests.
- Independent code review: exactly one independent code-review subagent was
  used, `019f251f-5cf2-7b33-ba85-0dd77c8f1658` / Nash. Findings and fixes:
  `db_writer` could be called without router lanes, fixed by fail-closed lane
  checks and `test_db_writer_without_router_lanes_fails_closed`; character and
  relationship lanes lacked explicit reviewer enforcement, fixed by lane
  reviewers and reviewer rejection tests; merge/evolve user-memory updates
  could lose source refs, fixed by threading source refs through
  `update_user_memory_unit_semantics`; reflection/shared-memory route support
  needed a public reflection origin, fixed with `reflection_signal` origin and
  source views; README authority values were stale, fixed to include
  `conversation_accepted`. A later parent-found live case 13 misroute was
  fixed by keeping private `user_style_image` out of ordinary-chat router
  rosters while preserving reflection user-style and group/channel style
  routes. Affected focused tests, all 26 live cases, and full non-live
  regression were rerun after fixes.
- Residual risks: 149 legacy no-due active commitments remain semantic
  manual-review findings; do not mutate them through deterministic cleanup.
  Existing legacy source-less user-memory rows were marked `legacy_unverified`
  during cleanup, while new writes are source-gated by this plan.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Weak router misroutes similar semantic work | Router handles coarse lanes only; specialists and reviewers own fine semantic distinction | Ambiguous live LLM cases and router contract tests |
| Deterministic code becomes semantic filter | Mandatory no-keyword rule and static grep | Static grep plus code review |
| Source policy over-prunes valid learned data | Source policy checks source class only, not text meaning | Tests with accepted learned user facts and accepted ongoing rules |
| Too many background LLM calls | Hard cap router tasks, specialist candidates, and reviewer passes | LLM call budget audit and trace review |
| External agents drift | Change surface keeps non-consolidation changes narrow | Code review and changed-file audit |
| Shared/world memory remains pollutible outside consolidation | Ordinary chat consolidation cannot write generic shared/world memory; accepted character self-guidance uses its own lane and approved promotion sources must use existing evidence/privacy fields | Static grep and post-cleanup DB smoke |
