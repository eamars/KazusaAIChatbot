# cognition core evolution progression

## Document Control

- Status: reference
- Type: architectural progression
- Execution rule: this document is not an implementation contract. Promote any
  item below into `active/short_term/` or `active/bugfix/` before production
  work.
- Related references:
  - `development_plans/reference/designs/self_cognition_loop_architecture.md`
  - `development_plans/reference/designs/self_cognition_reasoning_basis.md`
  - `development_plans/reference/designs/self_cognition_tracking_icd.md`
  - `development_plans/archive/completed/short_term/multi_source_cognition_architecture_plan.md`
  - `development_plans/archive/completed/short_term/global_character_growth_from_reflection_plan.md`
- Scope: review the current character growth and cognition-core architecture
  against industrial cognition/agent frameworks, and define a sequenced
  progression toward a self-driven evolving system that interacts with the
  world more richly.
- Audience: planners drafting the next short-term plans for cognition,
  reflection, growth, dispatcher, and self-cognition surfaces.

## Summary

- The cognition core is already source-agnostic (`user_message`,
  `internal_thought`, `self_cognition` enter the same L1/L2/L3 layers) and the
  memory layer already has lineage-tracked mutation with privacy/boundary
  review. These two properties put the project ahead of typical character-bot
  designs.
- Three industrial-grade properties are missing: a salience/attention layer
  over memory, persisted goals/intentions across turns, and a procedural skill
  store. World-interaction is also asymmetric: 30 input retrievers vs. 1
  effector (`send_message`).
- The recommended progression keeps the existing L1/L2/L3 core unchanged and
  adds four concentric loops at different cadences. Each loop step is small,
  builds on prior infrastructure, and leaves growth inspectable.
- Engine constraint: the cognition LLM is a locally hosted small Gemma model.
  This caps how much structured emission can be assumed reliable and is the
  primary calibration risk for any plan derived from this progression.

## Current State Snapshot

Authoritative shape of the live system, derived from a full code dive:

```text
trigger source (user_message | internal_thought | self_cognition)
  -> CognitiveEpisode + GlobalPersonaState
  -> RAG supervisor (multi-agent retrieval, MCP-wrapped web)
  -> Cognition subgraph
       L1 subconscious  -> emotional_appraisal, interaction_subtext
       L2 consciousness -> logical_stance, character_intent, internal_monologue
       L3 contextual    -> final_dialog wording, pacing, style
  -> Dispatcher (single effector: send_message)
  -> Consolidation subgraph (per-turn writes)
       CharacterProfileDoc.mood / global_vibe / reflection_summary (transient)
       UserProfileDoc.affinity / last_relationship_insight
       UserMemoryUnitDoc (per-user, stable_pattern | recent_shift |
         objective_fact | milestone | active_commitment)
       ConversationEpisodeStateDoc (open_loops, resolved_threads,
         avoid_reopening, assistant_moves, next_affordances)
  -> Reflection cycle (hourly -> daily -> global promotion)
       EvolvingMemoryDoc lore + self_guidance
         (lineage_id, version, supersede/merge, evidence_refs,
          privacy_review, boundary verdict)
  -> global_character_growth_traits (drift ledger;
       active in production as of 2026-05-15;
       per-trait strength, maturity_band,
       evidence_count, supporting_dates,
       source_reflection_run_ids; daily promotion cadence)
  -> CharacterProfileDoc.self_image
       (recent_window + historical_summary + meta.synthesis_count;
        actively synthesized; treated as narrative persistent self-model)
```

See **Database Review (2026-05-15)** at the end of this document for the
verified state of each subsystem.

Two load-bearing properties already in production:

- **Lineage-tracked memory mutation.** Every `EvolvingMemoryDoc` write carries
  `lineage_id`, `version`, `supersedes_memory_unit_ids`,
  `merged_from_memory_unit_ids`, and `evidence_refs`. Dedup at cosine 0.92,
  merge at 0.88, review band at 0.82. No hard deletes; only status
  transitions.
- **Trigger-source pluralism.** The same L1/L2/L3 path runs on user messages,
  internal thoughts, and self-cognition source packets. This is the
  `multi_source_cognition_architecture_plan` payoff.

## Comparison Against Industrial Cognition Frameworks

| Capability | OpenCog (AtomSpace/ECAN/PLN/MOSES) | Letta / MemGPT | Stanford Generative Agents | Voyager | LangGraph patterns | Kazusa today |
|---|---|---|---|---|---|---|
| Long-term knowledge store | AtomSpace hypergraph | Recall + archival memory | Memory stream + reflection tree | Skill library | Vector store + checkpoints | Mongo memory units + `EvolvingMemoryDoc` with lineage |
| Attention / salience | ECAN STI/LTI | Recency + paging | Importance x recency x relevance | Curriculum scoring | None | None explicit (cache is throughput, not salience) |
| Reasoning over symbolic state | PLN | LLM in-context | LLM reflection prompts | LLM + iterative refinement | LLM graph nodes | LLM only (local Gemma) |
| Procedural / skill learning | MOSES | None | Plans -> behaviors | Persisted skill code | Few-shot in tools | Defense rules + style overlays (static config, not callable) |
| Self-modification loop | Goal system + ECAN feedback | Memory edits | Reflection abstracts memories | Auto-curriculum | None | Reflection promotion -> `EvolvingMemoryDoc` |
| Embodied / world loop | CogPrime perception-action | Tool calls | Town tick | Game tick | Tool calls | Reactive only; one-shot scheduled `send_message` |
| Idle / self-pacing | Cognitive cycle | None | Reflection at idle | Continual play | None | `self_cognition` module exists; gated off by default |
| Persistent goals | First-class goal atoms | None | Plans | Auto-generated curricula | None | `character_intent` per turn, not persisted |

### Where Kazusa is already strong

- **Auditability and lineage** are stronger than Letta / MemGPT and second
  only to OpenCog on this list. Evidence refs plus version lineage make a
  year of slow drift inspectable.
- **Multi-source cognition entry** matches the Generative-Agents idea that
  reflection, perception, and planning should drive the same reasoning path,
  which most chatbot stacks do not have.
- **Slow promotion gate** with daily cadence, similarity dedup, and
  privacy/boundary review is the right structural answer to runaway
  personality drift.

### Where Kazusa is structurally weaker

These items are written as they appeared in the initial code-only review. The
Database Review at the end of this document corrects items 2 and 5 against
production data; the corrections override anything below.

1. **No salience / attention economy.** Retrieval is per-turn from a fixed
   plan; nothing makes recent-but-important memory rise without being asked.
2. **No goal-structured persistence.** `character_intent` is regenerated each
   turn. A narrative persistent self-model exists in
   `CharacterProfileDoc.self_image` (`historical_summary` plus
   `recent_window`, actively synthesized), but there is no goal-shaped store
   the cognition core can renew or abandon as discrete commitments.
3. **No procedural / skill memory.** Defense rules and style overlays are
   static configuration learned by reflection, not executable units the
   cognition core can call later.
4. **One effector.** The dispatcher knows only `send_message`. Confirmed by
   DB review: 70/70 scheduled events use `send_message`. The character cannot
   do anything outside chat.
5. **Outward loop exists but is one-shape.** `self_cognition` does fire and
   schedule real outbound through the action-attempt ledger (verified in DB
   review). What is missing is **route variety**: every accepted attempt is
   a `send_message`. There is no scheduled self-research, no scheduled
   internal recheck, no scheduled memory consolidation pass.
6. **Symbolic substrate absent.** World model lives in Mongo docs plus LLM
   short-term context. Acceptable today, but caps long-horizon self-evolution
   when contradictions accumulate.
7. **Local Gemma is the cognition LLM.** Small, weak world knowledge,
   sensitive to prompt framing. Frontier models can fake some of the missing
   structure; Gemma cannot.

## Future Architecture Direction

Keep the existing L1/L2/L3 core unchanged. Add four concentric loops at
different cadences, all sharing that core:

```text
Loop 0: Reactive turn        every user message     (exists)
Loop 1: Self-cognition tick  minutes, idle-gated    (built, dormant)
Loop 2: Reflection           hourly / daily         (exists)
Loop 3: Growth promotion     daily / weekly         (designed, not wired)
```

Concrete recommendations, ordered by leverage rather than by loop number:

### A. Default-on self-cognition at safe scope

Highest leverage because the hardest infrastructure is already built. Start
with the `progress_maintenance` route only: no sends, no memory writes, only
`open_loops` / `avoid_reopening` updates per the existing tracking ICD. After
a stable week, allow `action_candidate` for due-commitment cases, which the
`self_cognition_reasoning_basis` experiment already identified as the
strongest trigger source.

### B. Persisted character intent across turns

Add `CharacterIntentDoc` with the same lineage shape as `UserMemoryUnitDoc`:

```text
intent_text, scope, target_global_user_id,
status: pending | active | fulfilled | abandoned,
created_at, last_renewed_at, evidence_refs,
embedding
```

Consolidation writes `pending` when L2 emits forward-looking
`character_intent`. Self-cognition tick renews or abandons. RAG projects
active intents into L2 as soft context. Single small addition; closes the
biggest gap with Generative Agents.

### C. Salience and decay over user memory units

Two scalar fields on each memory unit:

- `salience` (recency-weighted, like STI), decayed per real-time hour.
- `importance` (set by reflection promotion, like LTI).

Retrieval score:
`retrieval_score = alpha * embedding_sim + beta * salience + gamma * importance`.

Small surface, no new collection. Produces materially better RAG hit ranking
for a weak local LLM where prompt budget is the limiting factor.

### D. Procedural skill library, backed by the dispatcher

The Voyager move, adapted to local-Gemma constraints. Reuse the lineage and
auditability already in `EvolvingMemoryDoc`:

```text
SkillDoc {
  skill_id, lineage_id, version,
  name, description,
  tool_calls: list[ToolCall],   # composable plan of existing tools, not code
  preconditions: dict,
  postconditions: dict,
  source_kind: seeded | reflection_inferred | user_taught,
  authority, status, evidence_refs,
  embedding
}
```

Skills are structured plans of existing dispatcher tools, not free-form code.
Reuses similarity dedup so the bot does not invent endless near-duplicates.

### E. Effector expansion

Direct fix for "make the cognition core more interactive with the world."
Concrete first additions, all cheap because the dispatcher tool registry and
permission gating already exist:

| Tool | Purpose | Notes |
|---|---|---|
| `schedule_future_check(at, topic)` | Character schedules its own self-cognition trigger | Closes the "I'll think about it tomorrow" gap. |
| `fetch_url(url)` | Promote `web_url_read` from RAG-internal to cognition-callable | Lets the character chase its own curiosity, not only research the user's question. |
| `note_open_loop(text, scope)` / `close_open_loop(...)` | Explicit hooks into `ConversationEpisodeStateDoc` | Cheaper and more reliable than relying on consolidation inference under a weak LLM. |
| `generate_image(prompt)` | MCP-wrapped image generation | Lower priority; the asymmetry is real but the previous tools have higher cognition payoff. |

### F. Idle-tick attention scorer

Replace the missing attention economy with a deterministic scorer that runs
each idle tick, scores all active `open_loops + character_intents +
active_commitments` by `(staleness, importance, due_proximity)`, and emits
the top-K as the self-cognition trigger packet. No LLM in this scorer. This
answers the "what should I think about now?" question that is currently
unanswered.

### G. Symbolic-lite world model (defer)

Only if `EvolvingMemoryDoc` accumulates contradictions Gemma cannot
reconcile. Not AtomSpace. A small typed-relation store:
`(subject, predicate, object, confidence, evidence_refs)`. Just enough that
"Kazusa knows X about Y" can be queried structurally and contradictions
flagged at promotion time. Defer until the failure is observed.

## Progression Plan

Sequenced from prerequisites outward. Each step is intended as one
short-term or bugfix plan, not a single epic.

| # | Step | Why this position | Approximate size |
|---|---|---|---|
| 1 | Finish `quote_aware_rag_sequence_plan` | Structural prerequisite: self-cognition source packets include quoted residue that today's RAG underweights. | active bugfix (in progress) |
| 2 | Wire `global_character_growth_traits` into L2 as soft background | Largest payoff per code touched. The collection design exists; cognition does not read it. Turns reflection from "writes a doc nobody reads" into observable drift. | small short-term |
| 3 | Add `CharacterIntentDoc` and project active intents into L2 | Unlocks every later loop. Cheap. | small short-term |
| 4 | Default-on self-cognition at `progress_maintenance` scope | No new code, only the gate flip plus evals. Run for a week. | small short-term |
| 5 | Memory salience + decay scalars on `UserMemoryUnitDoc` | Immediate RAG quality win under Gemma. Compounds with step 3. | small short-term |
| 6 | Effector expansion: `schedule_future_check`, `fetch_url`, `note_open_loop` | Closes input/output asymmetry. Generalizes dispatcher beyond `send_message`. | medium short-term |
| 7 | Self-cognition `action_candidate` route for due commitments only | First real proactive contact. Behind the gates that already exist. | small short-term |
| 8 | `SkillDoc` skill library populated by reflection | This is when the character starts compounding capability. | large short-term |

The recommended start is step 2. It has the best ratio of "infrastructure
already 80% built" to "this is the question being asked." Step 4 is what
makes the system feel alive between user messages.

## Engine Calibration Caveat

The progression above assumes the cognition LLM can do basic structured
emission reliably: forward-looking intent classification, salience
estimation, skill-plan generation. Per project memory the base model is
locally hosted Gemma, which is small and has weak world knowledge.

Before committing to steps 3, 5, or 8, run a calibration eval:

- Have Gemma emit `character_intent` over 200 logged turns.
- Hand-grade how many are genuinely forward-looking versus restated emotion.
- If under roughly 60 percent, the next step is not more architecture; it is
  either a stronger local model or routing L2 to a frontier API while
  keeping L1 and L3 local.

The architecture in this document is independent of model choice. The
sequence of plans derived from it is not.

## Non-Goals

This reference does not authorize:

- Direct mutation of `CharacterProfileDoc` personality fields outside the
  existing reflection-promotion path.
- A second cognition prompt path that bypasses L1/L2/L3.
- Free-form code-execution skills.
- Autonomous outward sends outside the existing `proactive_output` gates and
  the existing dispatcher.
- Replacement of `EvolvingMemoryDoc` lineage semantics.
- Porting OpenCog ECAN or PLN as-is. The salience and reasoning ideas are
  reused; the implementations are not.

## Acceptance Criteria For This Reference

This document is useful when it:

- Names the current cognition and growth surfaces precisely enough that a
  planner can find them in source without rediscovery.
- States the structural gaps as comparisons to specific industrial
  frameworks, not as generic critique.
- Provides a sequenced progression where each step is small, builds on the
  prior, and respects the lineage and privacy gates already in production.
- Surfaces the local-Gemma calibration risk before any plan promotion.
- Stays a reference. Any concrete change to code, schemas, dispatcher
  registry, or cognition prompts requires a separate promoted plan.

## Database Review (2026-05-15)

Where this section disagrees with anything earlier in the document, this
section is authoritative. Earlier text was written from code reading alone
and underestimates how much of the designed architecture is already running
in production.

### Scope of the review

Read-only export of the following collections into
`test_artifacts/db_review/` using `scripts.export_collection`:

| Collection | Rows exported | Sample window |
|---|---:|---|
| `character_state` | 1 (singleton) | current |
| `memory` | 200 of N | latest by `timestamp` |
| `user_memory_units` | 200 of N | latest by `timestamp` |
| `scheduled_events` | 70 | full |
| `global_character_growth_traits` | 7 | full |
| `global_character_growth_runs` | 18 | full, latest first |
| `self_cognition_action_attempts` | 6 | full |
| `character_reflection_runs` | 30 | latest by `started_at` |
| `conversation_episode_state` | 30 | full |
| `event_log_events` | 100 | latest |
| `user_profiles` | 30 | full |
| `interaction_style_images` | 9 | full |

Embeddings were excluded. The review focused on field presence, lineage
usage, status distributions, and date ranges.

### Validation table

| # | Claim from earlier in this document | Verdict | Evidence |
|---|---|---|---|
| A | `global_character_growth_traits` is designed but not wired into L2 | **Refuted** | 7 traits, all `status: active`. Each carries `strength` (0.15-0.36), `maturity_band` (`emerging` / `observed`), `evidence_count` (2-3), `supporting_dates`, and `source_reflection_run_ids` linking 16-24 reflection runs per trait. Created 2026-05-11 to 2026-05-14. |
| B | `self_cognition` is built but gated off | **Refuted** | 6 action attempts; 5 are `status: scheduled` with `dispatch_status: accepted`, real `target_channel` QQ IDs and `due_at`. 1 is `duplicate_suppressed`. The loop is firing and the dispatcher is accepting. |
| C | `CharacterProfileDoc.self_image` is deferred / not populated | **Refuted** | `self_image` contains `recent_window` (6 dated entries, newest 2026-05-14T20:26:32), `historical_summary` (multi-paragraph Chinese self-model), and `meta.synthesis_count: 472`. Actively maintained. |
| D | Reflection runs hourly → daily → global | **Confirmed** | 28/30 sampled runs are `hourly_slot`; `daily_channel` and `daily_global_promotion` runs present in the same window. 30/30 succeeded. |
| E | Growth cadence: daily / weekly | **Refined** | All runs are `run_kind: global_character_growth`. Cadence is daily, sometimes multiple per day. 18 runs span 2026-05-10 to 2026-05-14, 15 applied, 2 dry_run, 1 failed. |
| F | `EvolvingMemoryDoc` uses lineage-tracked supersede/merge | **Confirmed with caveat** | Of 200 sampled memory rows: 73 carry non-empty `supersedes_memory_unit_ids` (36.5%); only 3 carry non-empty `merged_from_memory_unit_ids` (1.5%). Merge path is *implemented but barely used*. Supersession is the dominant evolution. `source_kind` is `reflection_inferred` for 75.5% of rows; `authority` is `reflection_promoted` for 75.5%. |
| G | `UserMemoryUnitDoc.unit_type` set is the five named | **Confirmed exactly** | `objective_fact` 44%, `milestone` 22%, `recent_shift` 18.5%, `active_commitment` 14.5%, `stable_pattern` 1%. All 200 sampled rows are `status: active` — the `completed` / `cancelled` lifecycle exists in schema but is unused in the sample. |
| H | Dispatcher has one effector: `send_message` | **Confirmed absolutely** | 70/70 scheduled events use `send_message`. Status: completed 84%, failed 14%, pending 1%. |
| I | Event log captures observability | **Confirmed but narrow** | 100 events all from a 20-minute window on 2026-05-14. Subsystems represented: `brain_service` 95%, `reflection_cycle.worker` 5%. RAG, cognition, dispatcher, scheduler, and self-cognition emit nothing in this window. |
| J | `ConversationEpisodeStateDoc` shape | **Confirmed and *richer*** | Documented seven fields all present and populated. **Additional fields not described in earlier text:** `current_blocker`, `progression_guidance`, `current_thread`, `episode_label`, `episode_phase` (`developing` / `pivoting` / `resolving`), `emotional_trajectory`, `topic_momentum`, `conversation_mode`, `user_goal`, `expires_at`, `continuity`. |
| K | `UserProfileDoc.affinity` is populated | **Confirmed** | Affinity is a numeric integer on a 0-1000-ish scale, 500 baseline. `last_relationship_insight` populated in 17/30 sampled users with substantive Chinese-language strings. |
| L | `interaction_style_images` stores user *and* group overlays | **Confirmed and *with lineage*** | 6 group_channel, 2 user docs. Each carries `overlay` with `speech_guidelines`, `social_guidelines`, `pacing_guidelines`, `engagement_guidelines`, plus a `revision` counter (1-12) and a `source_reflection_run_ids` array linking up to 15 reflection runs. Style overlays are themselves a lineage-tracked, reflection-driven, slowly-evolving artifact. |
| - | `ConversationEpisodeStateDoc` is short-term, reset between episodes | **Refuted** | Sampled docs live 3-13 days with explicit `expires_at` 2-3 days into future. Episode state is medium-term, not per-turn. |

### Newly discovered capabilities

These were not visible from code reading alone and change the framing of the
progression plan.

1. **Persistent narrative self-model already exists.** `self_image` is
   actively synthesized (472 events) and stores both a rolling recent window
   and a long-form historical summary. Any future `CharacterIntentDoc` work
   should integrate with this, not stand alone.
2. **`source_reflection_run_ids` is a project-wide lineage primitive.** It
   appears on growth traits, on style images, and is consumable for cross-
   subsystem auditing. This is stronger than the earlier "lineage on
   `EvolvingMemoryDoc` only" framing.
3. **Style images version themselves.** `revision` 1-12 across 9 active docs
   means stylistic drift is already lineage-tracked and slowly evolving.
4. **Episode state already carries narrative-arc fields.** `episode_phase`,
   `emotional_trajectory`, `topic_momentum`, `conversation_mode`, and
   `user_goal` are populated. Several "future" recommendations were
   under-informed by this; some of what I called "missing intent" is
   actually expressed here.
5. **Self-cognition action ledger is real.** `self_cognition_action_attempts`
   is the same idempotency-keyed store the reasoning basis described, and
   it is dispatching `send_message` candidates with `dispatch_status:
   accepted`. The proactive loop is live, not a prototype.

### Honest gaps after the database review

Items in the original "Honest Gaps" list that survive the DB check:

- No salience / attention economy over user memory units. Confirmed: all 200
  sampled units are `status: active` with no decay or scoring fields.
- One effector. Confirmed: 100% of scheduled events are `send_message`.
- No procedural / skill memory. Confirmed: no skill collection exists.
- No symbolic substrate. Confirmed.
- Merge path is implemented but underused (3 of 200 vs. 73 supersessions).
  Either Gemma is biased toward supersede over merge, or the merge gate
  threshold is rarely tripped. Worth investigating before adding more
  lineage logic.
- `UserMemoryUnitDoc` lifecycle fields (`completed`, `cancelled`,
  `completed_at`, `cancelled_at`) exist but are unused in the active sample.
  Either the lifecycle transitions never fire, or the export window missed
  them. Worth a separate read with `--filter '{"status": {"$ne":
  "active"}}'`.
- Event log has 20 minutes of recent data only. Either retention is short or
  observability writes are not enabled for most subsystems. Independent of
  cognition, this is a problem for any future calibration eval.

Items removed from the gap list because the DB shows they are already
addressed:

- "global growth traits not wired" — already wired, daily, 7 active traits.
- "self_cognition gated off" — actively dispatching.
- "self_image deferred" — populated and actively synthesized.
- "no persistent self representation across turns" — `self_image` is exactly
  that, just narrative rather than goal-structured.

### Revised progression plan

The earlier 8-step plan is superseded by this list. Steps that called for
work already in production have been removed. New steps reflect what the
data shows is the real frontier.

| # | Step | Rationale grounded in DB review | Plan size |
|---|---|---|---|
| 1 | Finish `quote_aware_rag_sequence_plan` (already moved to completed bugfix records). | Verified by README registry update. | done |
| 2 | **Effector expansion beyond `send_message`.** Add at minimum `schedule_future_check`, `fetch_url`, `note_open_loop` / `close_open_loop` to the dispatcher registry. | DB shows 70/70 events are `send_message`. This is the strongest single-source effector asymmetry and the highest-leverage near-term fix. | medium short-term |
| 3 | **Memory salience + decay on `user_memory_units`.** Add `salience` (decayed per hour) and `importance` (set by reflection). Rank retrieval by linear combination. | DB shows 200/200 units are status active with no scoring; retrieval today is embedding-similarity only. Compounds with item 7. | small short-term |
| 4 | **User-memory lifecycle progression audit.** Investigate why `completed_at` / `cancelled_at` never fire on `active_commitment` units. Either the transition path is dead code, or the dispatcher post-send hook is missing. | DB shows 0/200 transitions in the sample despite 29 active_commitment units. | small bugfix |
| 5 | **Merge-vs-supersede balance audit.** 73 supersessions vs 3 merges suggests either Gemma prefers replace-over-fuse or the merge threshold (0.88) is too tight. Targeted eval before any new lineage logic. | DB shows the asymmetry directly. | calibration only |
| 6 | **Event-log retention or fan-in fix.** 20 minutes of events from only `brain_service` + `reflection_cycle.worker` is not enough to drive any calibration eval. | Without this, items 5 and 7 cannot be measured. | small bugfix |
| 7 | **Local-Gemma calibration suite.** Run a one-week eval emitting structured `character_intent`, salience scores, and skill-plan candidates and hand-grade them. Gate items 8 and 9 on the result. | All higher-tier recommendations assume Gemma can do reliable structured emission. The eval has to come before, not after. | medium short-term |
| 8 | **Goal-shaped intent layer on top of `self_image`.** Add `CharacterIntentDoc` only after item 7 confirms Gemma can classify forward-looking intent. Integrate with the existing `self_image.recent_window`, not parallel to it. | DB shows narrative self-model exists; the gap is goal-structure, not persistence. | medium short-term, gated |
| 9 | **`SkillDoc` skill library, reusing `source_reflection_run_ids` lineage.** Plans of dispatcher tools, not free-form code. | DB shows lineage primitive is already cross-subsystem; reuse it. | large short-term, gated |
| 10 | **Symbolic-lite world model.** Deferred until contradictions in `memory` are observed. | Same as before. | deferred |

The recommended starting point is now **item 2 (effector expansion)**, not
the earlier item 2 (wire growth traits). Growth traits are already wired;
the dispatcher is the most empirically constrained surface in the system.

### Implication summary

The character is more alive than the code-only review suggested. The
infrastructure for self-driven evolution is mostly running. The real
remaining work is not "wire up the growth loop" — it is "give the running
loop more than one way to act on the world, and make sure the architecture
stays engine-neutral as the engines underneath improve."

## Architectural Robustness Direction (2026-05-15)

Where this section conflicts with earlier text or with the Database Review's
revised progression, this section is authoritative for forward-looking
architecture. The Database Review remains authoritative for current state.

### Premise

Local engine choice is not an architectural constraint. The architecture
must remain valid as the cognition LLM, the embedding model, the vision
model, and any future action planner are swapped, mixed, or scaled up. The
seven contracts below are the structural primitives that make this true.

### L1, L2, L3 are concerns, not pipeline stages

The current code happens to compose them in one order. The architecture
must not bake that order in. Restated:

- **L1 = affective interpretation.** How does this stimulus make the
  character feel; what is salient.
- **L2 = deliberative reasoning.** What is the character's stance, intent,
  and modality-neutral action specification.
- **L3 = output channel specialization.** Turn intent into text, motion,
  image, sound, or API call.

In the live response path today these compose as `L1 -> L2 -> L3-text`.
That is one assembly. Future assemblies must coexist:

```text
L1 -> L3-reflex                (allow-listed reflex arc, no deliberation)
L1 -> L2 -> L3-text            (current production path)
L1 -> L2 -> L3-arm             (deliberative motor action)
L1 -> L2 -> L3-image           (image emission)
L1 -> L2 -> L3-tool            (tool invocation as output)
```

All non-reflex L3 surfaces consume the same L2 residue. L2 itself is
output-modality-neutral and never grows per-channel branches.

### The reflex arc is a first-class structural feature

Some actions must bypass deliberation by their nature: balance correction,
withdrawal from harm, conversational backchannels, latency-bounded motor
primitives. The reflex path is `L1 + affordance lookup -> L3` with no L2.

This is not an optimization for slow engines. It is a permanent feature of
how a cognitive system with a body composes its layers. Reflex primitives
must be explicitly allow-listed; the path must never become an escape
hatch for "L2 was too slow this time." A reflex that L2 would have
forbidden is a bug, not a shortcut.

### Seven contracts that must become explicit

Each contract today is implicit. It lives in prompt templates plus parser
code, and it is what would break first when an engine, modality, or memory
backend is swapped. For modular expandability, each must become a versioned
schema with a registry and an interface.

| # | Contract | Today | Required form |
|---|---|---|---|
| 1 | Trigger source | Partially formalized via `CognitiveEpisode.trigger_source` | Registry. New sources (sensor stream, peer-agent message, scheduled tick, proprioception) plug in by registering, not by editing the cognition core. |
| 2 | Inter-layer residue bus | Free-form dicts passed between cognition nodes | Typed residue objects: `L1Residue { emotional_appraisal, interaction_subtext, salience_hints }`, `L2Residue { logical_stance, character_intent, action_spec, internal_monologue, affordance_assumptions }`. Layers subscribe to residue types they consume. |
| 3 | Modality-neutral action spec | Dict-shaped `action_directives` consumed ad hoc by L3-text | Typed primitives: `{ kind, target, params, urgency, blocking, deadline }` with `kind` drawn from a registered set (`speak`, `reach`, `gaze`, `point`, `emit_image`, `invoke_tool`, ...). Each L3 surface registers handlers for the kinds it supports. |
| 4 | Affordances registry | Implicit in prompt context | Runtime registry. Each L3 surface and tool registers capability, availability, latency budget, and cost. L2 queries this as soft context; never assumes. |
| 5 | Engine routing layer | Cognition LLM is a hardcoded dependency | Each cognition node declares its requirements (structured-output needed, latency tier, capabilities). A router matches nodes to engines. Switching local Gemma -> frontier -> mixed local+frontier is configuration, not refactor. |
| 6 | Memory layer interface | Per-collection bespoke code under `src/kazusa_ai_chatbot/db/` | Provider interface: `query`, `insert`, `supersede`, `merge`, `decay`. Each memory type (`memory`, `user_memory_units`, growth traits, style images) implements it. Swapping Mongo for a vector DB or adding a knowledge graph is a provider swap. |
| 7 | Capability surface uniformity | Skills, effectors, retrievers, and trigger sources each register through ad-hoc shapes | One extension pattern: `(capability_spec, handler, lifecycle hooks)`. One mental model for adding anything to the system. |

These contracts are the load-bearing primitives. Engine independence,
modality plurality, and capability expandability are all expressed *as*
these contracts. Without them, any further architectural ambition reduces
to "we hope the next refactor catches what the last one missed."

### Output modality expansion (recommended order)

When non-text L3 surfaces are added, the order is fixed by dependency, not
by use case:

1. **Codify the modality-neutral action spec (contract 3).** Single most
   load-bearing item. Without this, every new L3 grows its own L2-residue
   parser and the layers drift apart.
2. **Build the affordances registry (contract 4) and inject it into L2's
   context.** Prerequisite to any non-text L3. Otherwise the character
   "tries to speak with hands it does not have," or fails to use surfaces
   that do exist.
3. **Define and allow-list reflex primitives.** Before any motor L3 ships,
   the reflex/deliberative split must be a typed, audited path, not a
   special case in code.
4. **Add a non-text L3 only after items 1-3.** L3-image (via existing
   image MCP path), L3-tool (generalized dispatcher beyond `send_message`),
   L3-arm, in whatever order the use case justifies.

### Engine independence rule

Engine independence is a goal of the *architecture*, not of every
individual plan. Each plan that lands must still be testable end-to-end on
whatever engine is current. Otherwise "engine-neutral design" becomes a
license to ship abstractions that have never run.

This rule supersedes the **Engine Calibration Caveat** earlier in this
document. Local-LLM calibration is a quality check on the current engine,
not a structural gate on architectural progression. Plans that require
richer structured emission can land; calibration is what determines which
engine they bind to today, not whether they exist.

### Twice-revised progression plan

The Database Review revised the original plan. This section revises it
again. Where rows conflict, this table wins.

| # | Step | Status / Change |
|---|---|---|
| 1 | `quote_aware_rag_sequence` | done |
| 2 | **Codify modality-neutral action spec (contract 3)** | **new**, must land before or alongside effector expansion |
| 3 | **Affordances registry (contract 4)** | **new**, prerequisite for any non-text L3 |
| 4 | **Reflex / deliberative split, allow-listed** | **new**, prerequisite for any motor L3 |
| 5 | Effector expansion beyond `send_message` | unchanged target; now framed as registering new handlers on contract 3, not adding ad-hoc tools |
| 6 | Memory salience + decay on `user_memory_units` | unchanged |
| 7 | `user_memory_units` lifecycle progression audit | unchanged |
| 8 | Merge-vs-supersede balance audit | unchanged |
| 9 | Event-log retention and cross-subsystem fan-in | **promoted in priority**: in a modular system, structured residue at every layer transition is the only viable debug and recalibration surface |
| 10 | Engine routing layer (contract 5) | **new**, enables mixed local + frontier engines without code churn |
| 11 | Memory provider interface (contract 6) | **new**, prerequisite for any non-Mongo memory backend |
| 12 | Inter-layer residue bus (contract 2) | **new**, prerequisite for parallel L3 surfaces and for swapping any single layer |
| 13 | Trigger-source registry (contract 1) | **new**, generalizes the existing `CognitiveEpisode` shape |
| 14 | Capability surface uniformity (contract 7) | **new**, the umbrella rule for items 5, 10-13 |
| 15 | Local-LLM calibration suite | **demoted to quality check**; no longer gates any item above |
| 16 | Goal-shaped intent (`CharacterIntentDoc`) | **un-gated**; integrates with the existing `self_image` rather than duplicating it |
| 17 | `SkillDoc` skill library | **un-gated**; allowed to be plan trees of registered action-spec primitives, not just tool sequences |
| 18 | Symbolic-lite world model | unchanged (deferred until contradictions in `memory` are observed) |

The recommended starting point is now **contract 3 (action spec) plus
item 5 (effector expansion) as a paired plan**. The pair forces the
contract to be defined against a concrete use case (a second effector
kind) without that effector locking the contract to its shape. Every
later item presupposes this pair has landed.

### Next reference document

The natural next reference is `cognition_contracts_design.md`, which locks
the seven contracts above as concrete schemas, interfaces, and registry
shapes. No expansion plan derived from this progression should be
promoted to `active/short_term/` before that contracts reference exists
and is approved — otherwise the architecture will accumulate parallel
ad-hoc shapes faster than they can be unified.
