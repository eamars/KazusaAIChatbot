# multi source cognition architecture plan

## Summary

- Goal: Refactor the current cognition architecture so `/chat`, reflection,
  internal thought, future recall, and multimodal inputs can enter the same
  cognition and consolidation flow through source-aware episode contracts.
- Plan class: large, architectural decisions
- Status: approved
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`,
  `database-data-pull`, and `cjk-safety` when editing Python files that contain
  CJK strings.
- Overall cutover strategy: migrate `/chat` first behind behavior-preserving
  compatibility contracts, block every later stage on no-regression evidence,
  then add alternative trigger and input sources in dry-run modes.
- Highest-risk areas: prompt regression, hidden `/chat` behavior changes,
  memory pollution, wrong target scope, private thought leakage, live latency,
  and treating reflection or internal monologue as fake user input.
- Acceptance criteria: the plan defines the shared cognition architecture,
  first-stage regression gates, trigger and input-source contracts, shared RAG
  and consolidator strategy, and independent child stages.

This is the top-level architectural development plan for multi-source
cognition. It is not an implementation contract by itself.

The active direction is to make multi-source cognition a cognition-core refactor
first. New proactive behavior can only be built after the current `/chat`
workflow has been migrated to the neutral episode contract with no regression.

Parent namespace: `multi_source_cognition_architecture`

Child filename pattern:
`multi_source_cognition_architecture_stage_<nn>_<topic>_plan.md`

## Confirmed Direction

The character should not have one mind for `/chat` and a different mind for
reflection. Different sources should wake the same cognitive process.

The target architecture is:

```text
trigger source
-> source-specific episode builder
-> shared input-source/percept normalization
-> shared context stack planner
-> shared RAG supervisor
-> shared L1/L2/L3 cognition nodes
-> shared action/output router
-> shared origin-aware consolidation
-> persistence, audit, delivery, or no-op according to origin policy
```

The top-level graph can differ by trigger source. The cognition stages should
remain the same conceptual stages and should reuse the same production node
boundary wherever practical.

The first implementation target is:

```text
current /chat request
-> CognitiveEpisode(trigger_source=user_message, input_sources=[dialog_text])
-> current RAG behavior
-> current L1/L2/L3 cognition behavior
-> current dialog behavior
-> current consolidation behavior
```

No reflection-triggered cognition, proactive preview, outbox, or transport work
may proceed until the `/chat` migration proves behavior, persistence, targeting,
and latency have not regressed.

## Progress Tracking Agreement

The parent plan is the stage ledger. A later child plan must not start until all
listed prerequisite stages are marked `completed` here and their execution
evidence names the required artifact paths.

| Stage      | Child plan                                                                                       | Ledger status | Completion artifact                                        | Required by      |
| ---------- | ------------------------------------------------------------------------------------------------ | ------------- | ---------------------------------------------------------- | ---------------- |
| `stage_00` | `multi_source_cognition_architecture_stage_00_current_chat_workflow_regression_baseline_plan.md` | completed     | `tests/test_multi_source_cognition_stage_00_regression_baseline.py`; `tests/fixtures/multi_source_cognition_stage_00_cases.json`; required verification passed | all later stages |
| `stage_01` | `multi_source_cognition_architecture_stage_01_cognitive_episode_contract_plan.md`                | draft         | contract module, builder tests, validation evidence        | `stage_02+`      |
| `stage_02` | `multi_source_cognition_architecture_stage_02_chat_cognitive_episode_migration_plan.md`          | draft         | `/chat` episode wiring, pass-through tests, baseline rerun | `stage_03+`      |
| `stage_03` | `multi_source_cognition_architecture_stage_03_shared_cognition_prompt_selection_plan.md`         | not_created   | prompt selector artifact, render tests, baseline rerun     | `stage_04+`      |
| `stage_04` | `multi_source_cognition_architecture_stage_04_rag_cognitive_episode_adapter_plan.md`             | not_created   | RAG input and retrieval equivalence artifacts              | `stage_05+`      |
| `stage_05` | `multi_source_cognition_architecture_stage_05_consolidation_origin_metadata_threading_plan.md`   | not_created   | origin metadata threading evidence                         | `stage_06+`      |
| `stage_06` | `multi_source_cognition_architecture_stage_06_consolidator_per_write_origin_policy_plan.md`      | not_created   | per-write origin policy tests                              | `stage_07+`      |
| `stage_07` | `multi_source_cognition_architecture_stage_07_reflection_trigger_cognition_dry_run_plan.md`      | not_created   | reflection dry-run audit artifact                          | `stage_08+`      |
| `stage_08` | `multi_source_cognition_architecture_stage_08_internal_thought_cognition_dry_run_plan.md`        | not_created   | internal thought dry-run artifact                          | `stage_09+`      |
| `stage_09` | `multi_source_cognition_architecture_stage_09_multimodal_cognitive_input_sources_plan.md`        | not_created   | multimodal percept fixtures and tests                      | `stage_10`       |
| `stage_10` | `multi_source_cognition_architecture_stage_10_permissioned_proactive_output_plan.md`             | not_created   | approved transport audit and baseline rerun                | none             |

Every child plan must include a `Completion Artifact Contract`. When a child
plan completes, update this ledger row, move or record the child plan according
to `development_plans/README.md`, and keep artifact paths in that child's
`Execution Evidence`. The next stage must read the parent ledger and previous
child execution evidence before editing files.

## Core Terms

`trigger_source` means why cognition is running.

Initial trigger-source values:

- `user_message`: normal `/chat` turn.
- `reflection_signal`: promoted reflection artifact or reflection-derived
  cognitive prompt.
- `internal_thought`: monologue, action residue, or private cognitive progress.
- `scheduled_recall`: future promise, reminder, or time-based recall event.
- `system_probe`: diagnostic or review-only run.

`input_source` means what the cognition stage is perceiving.

Initial input-source values:

- `dialog_text`: plain user or character-facing text.
- `image_observation`: image summary or structured visual percept.
- `audio_observation`: transcript, tone summary, or structured audio percept.
- `internal_monologue`: private thought from a previous or current cycle.
- `reflection_artifact`: promoted reflection summary or evidence.
- `retrieved_memory`: RAG or memory evidence made visible to cognition.

`output_mode` means what cognition is allowed to produce.

Initial output-mode values:

- `visible_reply`: normal `/chat` response.
- `silent`: deliberate no visible response.
- `think_only`: internal cognition result only.
- `preview`: candidate outward wording that must not be sent.
- `scheduled_action_request`: proposed future action for a later policy layer.

These labels are semantic contracts. Deterministic code validates them, but the
LLM cognition stages judge meaning and appropriateness.

## Architectural Runtime Overview

The production runtime should have source-specific entry graphs and a shared
cognition core.

```text
/chat entry graph

adapter/debug client
-> /chat request
-> message envelope and current metadata
-> CognitiveEpisode builder for user_message
-> shared RAG/context stack
-> shared L1/L2/L3 cognition
-> output router: visible_reply or silent
-> dialog
-> assistant persistence
-> origin-aware consolidation
```

```text
reflection entry graph

reflection worker
-> promoted reflection artifact
-> CognitiveEpisode builder for reflection_signal
-> shared RAG/context stack
-> shared L1/L2/L3 cognition
-> output router: think_only, preview, scheduled_action_request, or silent
-> dry-run audit first
-> later approval and transport policy, if enabled
-> origin-aware consolidation
```

```text
internal thought entry graph

action latch or private cognition residue
-> CognitiveEpisode builder for internal_thought
-> shared context stack
-> shared L1/L2/L3 cognition
-> output router: think_only, preview, or silent
-> private/public residue persistence by origin policy
```

The merge point is the shared episode contract and the shared cognition flow,
not a fake `/chat` call. Reflection and internal thoughts must never be inserted
as user messages just to reuse the graph.

## Shared Components

These components should be shared across trigger sources:

- RAG supervisor and retrieval agents.
- RAG projection shape consumed by cognition and consolidation.
- L1 subconscious appraisal node boundary.
- L2 consciousness, boundary, and judgment node boundary.
- L3 contextual, style, content-anchor, preference, visual, and collector node
  boundary.
- Dialog voice machinery when `output_mode` requires outward wording.
- Consolidator graph with source-aware policy.
- Persistence helpers, cache invalidation, audit, and delivery tracking.

These components need source-aware adapters before they can be shared safely:

- Decontextualization, because not every trigger is a user utterance.
- Prompt construction, because weak local LLMs need source-specific framing.
- Context stack selection, because reflection, image, audio, and user text need
  different evidence emphasis.
- Consolidator writes, because not every origin may create user facts,
  relationship changes, future promises, or public scene residue.

## Prompt Strategy

Node reuse does not require one giant prompt.

For the local model, the safer strategy is shared node interface plus dynamic
prompt selection:

```text
same L1 node function
-> prompt selected by trigger_source and input_source profile
-> same output schema

same L2 node function
-> prompt selected by trigger_source, visibility, and output_mode
-> same output schema

same L3 node function group
-> prompt selected by output_mode and target scope
-> same output schema
```

This keeps "how the character thinks" consistent while avoiding prompts that
ask a weaker model to reason over every possible source type at once.

Prompt tuning must follow these rules:

- Keep visible input labels semantic and short.
- Tell the model what it is perceiving, not where database fields came from.
- Keep output schemas stable across source-specific prompts.
- Prefer compact typed percepts over raw logs.
- Add source-specific examples only when they protect a real boundary.
- Run prompt-render checks before approving any child plan that touches prompts.
- Do not add deterministic keyword routing for semantic permission or intent.

## RAG Strategy

RAG should be shared, but the query builder must become episode-aware.

For `/chat`, the query remains based on the decontextualized user turn.

For reflection, internal thought, image, or audio triggers, the episode builder
must produce a source-appropriate `episode_focus` and context request. It must
not lie by calling that focus a user message.

The RAG supervisor should continue to return evidence. Cognition decides stance
and action. Consolidation decides what can become durable memory under origin
policy.

## Consolidation Strategy

The consolidator can be shared, but it must become origin-aware before non-chat
triggers are allowed.

Initial origin policy:

```text
origin=user_message:
  current consolidation behavior

origin=reflection_signal:
  may update internal reflection/cognition progress only
  must not create user facts from generated thought
  must not create future promises unless a separate approved policy allows it

origin=internal_thought:
  may update private cognition residue
  must not enter public conversation history

origin=proactive_sent:
  may persist public action residue and delivery audit
  must not treat the sent message as a user instruction
```

The first migration stages should keep `origin=user_message` behavior identical.
Origin policy for other sources belongs in later dry-run child plans.

## Experimental Cognition Reference

`experiments/cognition_core_next/` is a design reference, not production code to
copy directly.

Useful concepts to preserve:

- Typed percepts independent from dialog input.
- Multiple trigger sources entering one cognition flow.
- Private cognition residue separated from public group scene residue.
- Action latch from one outward action into later cognition.
- Bounded internal reflection loop.
- L3 evaluator/retry loop for outward wording.

Production adaptation rules:

- Do not import experiment modules into `src/`.
- Translate experiment concepts into production state schemas and node
  contracts.
- Keep current production `/chat` behavior as the regression baseline.
- Add loops only after bounded latency and retry policies are explicit.

## Non-Goals

This plan does not implement proactive messages.

This plan does not approve autonomous contact.

This plan does not send reflection output.

This plan does not route reflection, monologue, image, or audio through `/chat`
as fake user input.

This plan does not replace production cognition with the experiment runtime.

This plan does not broaden adapter responsibilities.

## System Risks

The primary risks are:

- Regression in the current `/chat` workflow.
- Prompt drift from making existing nodes too generic too quickly.
- Extra latency in live user turns.
- RAG behavior changes caused by a new query abstraction.
- Consolidator memory pollution from generated thoughts.
- Private internal cognition leaking into public or group contexts.
- Treating source-specific graph construction as separate character minds.
- Adding future proactive transport before the cognition core is stable.

Every child plan must contain these risks before expanding scope.

## Regression Gate Policy

The first stages are gating stages. A later stage may not proceed while a
current `/chat` regression is known or suspected.

Every child plan that touches graph state, prompts, RAG inputs, cognition
schemas, dialog output, persistence, or consolidation must prove:

- Current `/chat` request and response contracts remain compatible.
- Current graph route remains equivalent for normal text turns.
- RAG request shape is equivalent or intentionally mapped with tests.
- L1/L2/L3 outputs keep the same required fields.
- Dialog still receives the fields it expects.
- Consolidator still receives the fields it expects.
- Assistant persistence and delivery tracking remain unchanged for `/chat`.
- Target addressee and broadcast behavior do not change.
- Debug modes `listen_only`, `think_only`, and `no_remember` still behave.
- Prompt rendering succeeds for all changed prompt variants.
- Live `/chat` has no new LLM calls unless the child plan explicitly approves
  and measures them.

If any gate fails, stop multi-source cognition progression and create a bugfix
plan.

## Stage Independence Rule

Each stage must:

- Be reviewable and mergeable without approving later stages.
- Preserve current production behavior unless explicitly scoped otherwise.
- Have its own rollback path.
- Produce tests, schemas, or documentation that remain useful if later stages
  pause.
- Avoid half-migrated meaning in persisted data.
- Keep live chat bounded and inspectable.

## Staged Execution Overview

### stage_00 - Current Workflow Regression Baseline

Child plan:
`development_plans/active/short_term/multi_source_cognition_architecture_stage_00_current_chat_workflow_regression_baseline_plan.md`

Purpose:
Capture the current `/chat` behavior before refactoring.

Independent output:
A deterministic regression harness and evidence checklist for current text
dialog turns.

Scope:

- Identify representative `/chat` paths: private, group, reply, no-reply,
  think-only, no-remember, RAG hit, RAG skip, and silence.
- Add prompt-render checks for current L1/L2/L3 and dialog prompts.
- Add graph route and state-shape assertions around current persona flow.
- Add persistence/consolidation assertions for current origin behavior.
- Define real-LLM smoke cases that must be run one at a time when needed.

Exit if paused:
The project gains a reusable regression gate for all future cognition work.

### stage_01 - CognitiveEpisode Contract

Child plan:
`development_plans/active/short_term/multi_source_cognition_architecture_stage_01_cognitive_episode_contract_plan.md`

Purpose:
Define the neutral episode, trigger-source, input-source, percept, context
stack, and output-mode contracts without changing runtime behavior.

Independent output:
Typed contracts and documentation that let current `/chat` be represented as a
source-aware cognition episode.

Scope:

- Define `CognitiveEpisode`.
- Define `trigger_source`, `input_sources`, `percepts`, `visibility`,
  `target_scope`, `output_mode`, and `origin_metadata`.
- Define compatibility fields that still feed current production nodes.
- Define validation rules for required source-specific fields.
- Add unit tests for text-only `/chat` episode construction.
- Do not change the top-level graph yet.

Exit if paused:
The project has a reviewed contract for future multi-source cognition.

### stage_02 - `/chat` Episode Migration

Child plan:
`development_plans/active/short_term/multi_source_cognition_architecture_stage_02_chat_cognitive_episode_migration_plan.md`

Purpose:
Make the existing `/chat` graph build and pass a `CognitiveEpisode` while
preserving current behavior.

Independent output:
Current `/chat` runs through the neutral episode contract with no behavior,
prompt, RAG, dialog, persistence, or consolidation regression.

Scope:

- Add a `/chat` episode builder after message-envelope hydration.
- Map current text/dialog input to `trigger_source=user_message` and
  `input_sources=[dialog_text]`.
- Preserve current `decontexualized_input` as a compatibility projection.
- Preserve current RAG context values.
- Preserve current cognition state fields.
- Preserve current dialog and consolidation inputs.
- Add regression tests from `stage_00`.

Exit if paused:
The live workflow is already on the future-compatible input contract.

### stage_03 - Shared L1/L2/L3 Prompt Selection

Child plan:
`development_plans/active/short_term/multi_source_cognition_architecture_stage_03_shared_cognition_prompt_selection_plan.md`

Purpose:
Make the cognition nodes source-aware through prompt selection while keeping the
current `/chat` prompt path behaviorally stable.

Independent output:
The same cognition node boundaries can serve multiple source types later, but
only the current `/chat` variant is active.

Scope:

- Add prompt selection by `trigger_source`, `input_sources`, visibility, and
  output mode.
- Keep the existing `/chat` prompt variant as the default active path.
- Require the same output schema for every variant.
- Add prompt-render tests.
- Add schema validation for L1/L2/L3 outputs.
- Add source-specific prompt drafts for reflection and internal thought as
  inactive fixtures or reference docs only.

Exit if paused:
The project has a source-aware prompt framework without changing live behavior.

### stage_04 - Shared RAG Episode Adapter

Child plan:
`development_plans/active/short_term/multi_source_cognition_architecture_stage_04_rag_cognitive_episode_adapter_plan.md`

Purpose:
Refactor RAG input construction so it accepts `CognitiveEpisode` after the
cognition prompt boundary is stable.

Independent output:
RAG becomes source-aware at the adapter boundary, but current text `/chat`
retrieval remains unchanged.

Scope:

- Add an episode-to-RAG query/context adapter.
- Keep the current decontextualized text query for `/chat`.
- Add input-equivalence tests for current RAG calls.
- Add retrieval-equivalence tests on a frozen evidence corpus from `stage_00`.
- Compare dispatched agents, source keys, consolidation policies, fact ids or
  stable dedup keys, answer availability, and unknown-slot categories.
- Define how future reflection, image, audio, and internal thought episodes
  will produce `episode_focus`.
- Do not enable non-chat trigger sources yet.

Exit if paused:
RAG can be extended later without rewriting the cognition graph, and any
retrieval drift is visible before non-chat sources exist.

### stage_05 - Consolidation Origin Metadata Threading

Child plan:
`development_plans/active/short_term/multi_source_cognition_architecture_stage_05_consolidation_origin_metadata_threading_plan.md`

Purpose:
Thread episode origin metadata into consolidation without changing any writes.

Independent output:
The consolidator can see origin metadata, while `origin=user_message` behavior
remains identical and unsupported origins fail closed.

Scope:

- Add origin metadata to consolidation input.
- Add deterministic guards for unsupported origins.
- Preserve current behavior for `origin=user_message`.
- Add tests proving current consolidation input and output state shapes are
  unchanged for normal `/chat`.
- Do not change facts, promises, relationship, or mood write policy yet.

Exit if paused:
The current consolidator has explicit origin visibility with no write-policy
change.

### stage_06 - Consolidator Per-Write Origin Policy

Child plan:
`development_plans/active/short_term/multi_source_cognition_architecture_stage_06_consolidator_per_write_origin_policy_plan.md`

Purpose:
Define and test origin policy for every consolidator write path before
non-chat triggers can write anything durable.

Independent output:
Facts, promises, relationship, mood, reflection summary, cache invalidation, and
scheduled-event handoff each have explicit allowed origins and tests.

Scope:

- Preserve all current `origin=user_message` behavior.
- Add per-write tests for facts, future promises, relationship/affinity, mood,
  reflection summary, cache invalidation, and scheduled-event handoff.
- Keep `reflection_signal` and `internal_thought` write paths disabled or
  dry-run until their later child plans approve specific writes.
- Verify generated thought cannot create user facts or promises.

Exit if paused:
The shared consolidator has a safe origin policy even before reflection or
internal thought writes are enabled.

### stage_07 - Reflection Trigger Dry Run

Child plan:
`development_plans/active/short_term/multi_source_cognition_architecture_stage_07_reflection_trigger_cognition_dry_run_plan.md`

Purpose:
Feed promoted reflection artifacts into the shared cognition flow in dry-run
mode only.

Independent output:
Reflection can trigger the same L1/L2/L3 cognition process without sending
messages or changing normal conversation history.

Scope:

- Build `CognitiveEpisode(trigger_source=reflection_signal)`.
- Use promoted reflection context, not raw reflection output.
- Select reflection-aware prompt variants.
- Restrict output modes to `think_only`, `preview`, or `silent`.
- Store dry-run audit records only.
- Revalidate that `/chat` regression gates still pass.

Exit if paused:
The team can inspect whether shared cognition handles reflection input.

### stage_08 - Internal Thought And Action Latch Dry Run

Child plan:
`development_plans/active/short_term/multi_source_cognition_architecture_stage_08_internal_thought_cognition_dry_run_plan.md`

Purpose:
Adapt the experiment's monologue and action-latch concept into production dry
run contracts.

Independent output:
Internal thought can enter the shared cognition flow without public leakage or
conversation-history pollution.

Scope:

- Define private cognition residue and public scene residue contracts.
- Build `CognitiveEpisode(trigger_source=internal_thought)`.
- Add bounded loop policy before any loop is enabled.
- Store dry-run records separately from normal messages.
- Preserve `/chat` regression gates.

Exit if paused:
The project gains safe internal-cognition infrastructure for later behavior.

### stage_09 - Multimodal Input Source Expansion

Child plan:
`development_plans/active/short_term/multi_source_cognition_architecture_stage_09_multimodal_cognitive_input_sources_plan.md`

Purpose:
Support image and audio as typed input sources to the same cognition flow.

Independent output:
Image and audio summaries become structured percepts without changing the
meaning of text `/chat` turns.

Scope:

- Define `image_observation` and `audio_observation` percept contracts.
- Add input-source normalization for summaries, transcripts, and tone labels.
- Keep raw binary handling outside cognition prompts.
- Add source-specific prompt variants if needed.
- Preserve `/chat` text regression gates.

Exit if paused:
The cognition core can consume richer input sources without proactive behavior.

### stage_10 - Permissioned Proactive Output

Child plan:
`development_plans/active/short_term/multi_source_cognition_architecture_stage_10_permissioned_proactive_output_plan.md`

Purpose:
Only after shared cognition and regression gates are stable, add preview,
approval, outbox, transport, and proactive persistence.

Independent output:
A controlled proactive path that reuses shared cognition output and applies
separate permission and transport policy.

Scope:

- Define preview and approval lifecycle.
- Add durable outbox if needed for retries and audit.
- Require explicit permission, quiet hours, adapter availability, and target
  validation.
- Persist sent proactive messages with explicit origin metadata.
- Keep previews out of normal conversation history.
- Rerun the full `stage_00` `/chat` regression harness as a hard gate before any
  preview or transport cutover.
- Treat any `stage_00` harness failure as a blocker that requires a bugfix plan.

Exit if paused:
Earlier cognition refactors remain useful without autonomous contact.

## Child Plan Promotion Rules

Before any child plan can move to `approved`, it must:

- Name exact files and modules.
- Name prerequisite stages and required artifact paths from this ledger.
- State whether it changes runtime behavior.
- State whether it changes database schema.
- State whether it changes prompts.
- State whether it adds an LLM call.
- Include response-path latency impact.
- Include context budget and prompt input limits for every LLM stage.
- Include deterministic tests.
- Include prompt-render checks for prompt changes.
- Include manual review steps for real LLM or database evidence.
- Include rollback instructions.
- Include cutover policy.
- Include explicit non-goals.

## LLM Call And Context Budget

The current `/chat` path must not receive extra LLM calls during the first
migration stages.

Top-level budgets:

- `stage_00` through `stage_06`: no new live `/chat` LLM calls.
- `stage_07`: reflection dry run may use background LLM calls only when idle.
- `stage_08`: internal thought dry run may use background calls only under a
  bounded loop policy.
- `stage_09`: multimodal support may add summarizer calls only in a child plan
  that owns media processing and latency.
- `stage_10`: proactive preview may use background calls only after approval
  policy and regression gates exist.

Context rules:

- Use typed percepts and compact episode context.
- Translate raw telemetry into semantic labels before prompts.
- Keep source-specific prompts shorter than a universal prompt.
- Keep private reflection and internal thought out of public-target prompts.
- Keep hidden database schema out of model-facing prompt text.

## Performance Policy

Multi-source cognition must preserve the live response path as bounded and
inspectable.

Required properties:

- `/chat` migration first.
- No additional live LLM calls in early stages.
- Prompt selection must not materially increase prompt size for text `/chat`.
- Reflection and internal-thought dry runs run only in background.
- Background cognition checks `primary_interaction_busy`.
- One background episode at a time until measured evidence supports more.
- Metrics for prompt size, LLM call count, graph route, latency, and output
  mode distribution.

## Rollback And Stop Conditions

Global stop conditions:

- Current `/chat` output, targeting, persistence, or consolidation regresses.
- Live `/chat` latency increases without approval.
- Prompt selection changes current `/chat` prompt semantics unexpectedly.
- RAG retrieves different evidence for equivalent `/chat` inputs without a
  deliberate child-plan decision.
- Generated reflection or internal thought enters user memory as fact.
- Private internal cognition appears in public or group output.
- A non-chat source is treated as a fake user message.

Rollback requirements:

- Each stage can disable its new adapter or prompt selector.
- `/chat` can fall back to the pre-stage compatibility path.
- Dry-run records remain inspectable after rollback.
- No new origin policy writes are enabled without a feature flag or explicit
  child-plan cutover.

## Cutover Policy

There is no direct cutover from this top-level plan.

Default child-plan cutover:

- Add contract or adapter behind compatibility projection.
- Prove current `/chat` regression gates.
- Verify prerequisite stage artifacts exist before editing.
- Enable only the migrated `/chat` path first.
- Keep non-chat triggers disabled or dry-run only.
- Promote the next child stage only after regression evidence is recorded.

No child plan may enable autonomous proactive sends as its initial cutover.

## Verification Strategy

Top-level plan verification:

```powershell
rg -n "CognitiveEpisode|/chat Episode Migration|Regression Gate Policy|Shared L1/L2/L3" development_plans\active\short_term\multi_source_cognition_architecture_plan.md
rg -n "Status: draft|not an implementation contract|Staged Execution Overview" development_plans\active\short_term\multi_source_cognition_architecture_plan.md
git diff --check
```

Child-plan verification must be defined by each child plan. The minimum default
is deterministic unit tests, prompt-render tests, graph route checks, state
shape checks, repository tests for persistence changes, and manual inspection
for real LLM or live database evidence.

## Acceptance Criteria

This top-level plan is complete when:

- The development plan registry lists it as an active draft.
- It states that `/chat` migration is the first implementation target.
- It defines trigger-source, input-source, and output-mode contracts.
- It explains that top-level graphs may differ while L1/L2/L3 cognition,
  RAG, and consolidation are shared through source-aware adapters.
- It defines regression gates before alternative triggers are enabled.
- It accounts for `experiments/cognition_core_next/` as reference material
  without adopting it directly.
- It defines child plan filenames with the parent namespace and staging index.
- It defines the progress ledger and artifact handoff contract.
- It states child-plan promotion rules, rollback, and stop conditions.

## Execution Notes

No source code changes are authorized by this plan.

No database schema changes are authorized by this plan.

No proactive sends are authorized by this plan.

The next useful action is to review and approve this architecture and its
`stage_00` through `stage_02` child plans, then execute only approved stages in
ledger order.

## Execution Evidence

Draft artifact only. No implementation has been executed from this plan.
