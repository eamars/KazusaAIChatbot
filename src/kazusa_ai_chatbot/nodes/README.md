# Cognition Nodes

`kazusa_ai_chatbot.nodes` owns the adapter-neutral connector between the brain
service and Cognition Core V2. It prepares the current episode, runs bounded
cognition and resolver recurrence, commits the final replacement state, routes
selected action and surface work, and hands a semantic text plan to dialog.

The package is part of the character-brain path:

```text
adapter or debug client
  -> brain service queue and intake
  -> relevance, media description, and conversation context
  -> persona_supervisor2
       decontextualization
       V2 cognition and bounded resolver recurrence
       one final cognition-state commit
       selected action handling
       optional V2 text-surface planning
       dialog wording
  -> consolidation and persistence
  -> scheduler and reflection outside live chat
```

RAG and resolver capabilities return evidence. Cognition Core V2 owns semantic
appraisal, causal state changes, goal judgment, bid collapse, route selection,
and response goals. Deterministic connectors own validation, persistence,
action materialization, permissions, limits, and graph routing. The V2 surface
planner owns expression planning. `dialog_agent.py` owns final visible wording.

## Module Boundary

| Area | Main files | Ownership |
| --- | --- | --- |
| Relevance and perception | `persona_relevance_agent.py`, `persona_supervisor2_msg_decontexualizer.py` | Whether a live event enters persona cognition, current media observations, current-message rewrite, and referent status. |
| Persona graph | `persona_supervisor2.py` | Resolver recurrence, final commit ordering, action/surface routing, no-response handling, and episode trace assembly. |
| V2 connector | `persona_supervisor2_cognition.py`, `persona_supervisor2_cognition_actions.py` | Exact `CognitionCoreInputV2` construction, state loading, V2 service binding, output projection, final state replacement, and semantic action-request materialization. |
| Text-surface connector | `persona_supervisor2_l3_surface.py` | Prompt-safe interaction-style loading, exact `TextSurfaceInputV2` construction, and `run_text_surface_planning(...)`. |
| Dialog | `dialog_agent.py` | Final text from `TextSurfaceOutputV2`, visible episode grounding, and permitted action results. |
| Specialist action handling | `persona_supervisor2_memory_lifecycle.py`, action-spec packages | Deterministic validation and execution of admitted semantic action requests. |

Nodes consume normalized `message_envelope`, `prompt_message_context`,
`reply_context`, `CognitiveEpisode`, typed user identities, and bounded history.
Platform wire syntax and adapter-specific command parsing stay upstream.

## Canonical Live Flow

The persona graph is:

```text
stage_0_msg_decontexualizer
  -> stage_1_goal_resolver
       load one mutable user or character V2 state
       build CognitionCoreInputV2
       run V2 cognition without an intermediate commit
       optionally execute one cognition-selected resolver capability
       project the observation as typed evidence
       repeat within the resolver cycle cap
       validate the terminal CognitionCoreOutputV2
       commit exactly one final replacement state
  -> stage_2_memory_lifecycle
  -> stage_2a_background_work_enqueue
  -> route from cognition_core_output.intention.route
       speech
         -> build TextSurfaceInputV2
         -> run four bounded V2 surface stages
         -> dialog_agent
         -> text surface and action-result trace
       non-speech
         -> private terminal handling
         -> action-result trace without visible dialog
```

The route decision requires a validated V2 cognition output. The presence of
an action specification cannot create a text response and cannot substitute
for `intention.route == "speech"`.

The resolver carries the latest cognition output and observations in memory.
It does not reload or persist cognition state between cycles. The connector
commits only the terminal replacement state, before action execution, surface
planning, dialog, consolidation, or delivery.

The live persona capability connector currently executes bounded local-context
recall. A capability failure returns a fixed semantic failure observation;
exception type may be logged, while exception text and operational details stay
outside cognition evidence and prompts.

## Cognition Core V2 Boundary

`persona_supervisor2_cognition.py` maps graph state into the exact public V2
contract. Its input includes:

- the validated canonical episode;
- one validated mutable cognition state and separate character constraints;
- typed episode, media, RAG, resolver, and permitted action-result evidence;
- direct facts with trusted provenance;
- available action and resolver affordances; and
- a bounded semantic scene description.

The V2 core performs deterministic preparation, scoped semantic appraisal,
state reduction, dependency-ready goal cognition, complete-bid collapse, and
route validation. Its output contains the replacement state, selected semantic
intention, admitted/supporting bids, semantic affect and relationship
projections, action requests, resolver requests, progress, expression policy,
diagnostics, and bounded residue.

Persistent identifiers and raw numeric state remain behind deterministic
handle bindings. Model-facing projections use semantic roles and qualitative
bands. RAG evidence does not become persona, affect, or final stance merely by
being retrieved.

## Action Ownership

V2 goal branches may bid for speech, silence, private handling, an action, or
resolver evidence. Complete bids retain their semantic intention, desired
outcome, grounded detail, target roles, consequences, route, and declared
request until deterministic collapse and validation finish.

Only admitted `action_requests` are materialized into the existing action-spec
execution boundary. Deterministic code revalidates capability availability,
permissions, target bindings, and parameters. Action specs and action results
remain trace/execution artifacts; they do not own cognition route selection.

Memory-lifecycle requests follow a specialist boundary. Cognition may request
a semantic lifecycle review, while the specialist chooses prompt-safe aliases
and deterministic code resolves an eligible persistent row. Cognition does not
select database identifiers or write lifecycle state directly.

Accepted background work is queued before a selected speech surface so the
surface can describe only the actual semantic outcome. A background request
without a visible acknowledgement route receives a deterministic failure
result instead of silently promising work.

## V2 Text Surface And Dialog

`persona_supervisor2_l3_surface.py` runs only after the final cognition state
commit and only for a speech intention. It builds exact `TextSurfaceInputV2`
from:

- the canonical episode;
- the selected intention;
- bounded primary and supporting bid projections;
- expression policy;
- semantic affect and optional relationship projections;
- permitted semantic action results; and
- bounded interaction-style guidance.

The connector loads the existing sanitized user interaction-style overlay and,
for group turns, the group-channel overlay. It renders only allowlisted speech,
social, pacing, and engagement guidance in application order into the bounded
string required by `TextSurfaceInputV2`. Storage identifiers, revisions,
reflection lineage, and raw channel/user identifiers are excluded.

`run_text_surface_planning(...)` projects visible episode content and runs four
bounded stages for style, content, preference, and visual/pacing guidance. The
stages plan expression and return semantic strings; they do not write final
dialogue. `dialog_agent.py` receives `TextSurfaceOutputV2`, visible episode
grounding, allowed action results, and the character profile, then authors the
final text fragments.

Dialog does not receive raw V2 mutable state, private branch payloads,
suppressed bids, persistent handles, relationship scalars, or obsolete
directive bags.

## Cognitive Episodes

`CognitiveEpisode` is the source-neutral current-event boundary. Supported
trigger/source combinations include:

| Trigger | Primary input source | Typical output modes |
| --- | --- | --- |
| `user_message` | `dialog_text` with optional media observations | `visible_reply`, `think_only`, `silent` |
| `accepted_task_result_ready` | `accepted_task_result` | `visible_reply`, `think_only`, `silent` |
| `reflection_signal` | `reflection_artifact` | `think_only`, `preview`, `silent` |
| `internal_thought` | `internal_monologue` | `think_only`, `preview`, `silent` |
| scheduled or system cognition | typed scheduled/internal percept | contract-allowed non-live modes |

The episode carries typed percepts, target scope, origin metadata, UTC storage
time, configured-local time, and hard output-mode constraints. Prompt
projection exposes only model-visible percept content and permitted semantic
metadata. Raw platform syntax, message identifiers, row identifiers, and debug
controls remain deterministic provenance.

## State, Trace, And Consolidation

User cognition state and singleton character cognition state are separate
mutable scopes. The selected scope is resolved from the episode origin and
caller, validated before cognition, and replaced once after terminal V2 output.
Character drives, standards, and meaning constraints can inform a user-scoped
turn without becoming user-owned mutable state.

The persona graph assembles action results and surface outputs into the existing
episode-trace envelope for downstream diagnostics and consolidation. The trace
records what was validated, attempted, completed, rejected, or surfaced. It is
not a second cognition authority.

Consolidation runs after the live wording path. It consumes prompt-safe episode
and trace evidence, plans eligible persistence targets, and applies writes
through its own validated lanes. Nodes do not let consolidation execute
actions, deliver messages, schedule work, or reopen cognition.

## Failure And Safety Rules

- Missing or partial V2 cognition output fails before surface routing.
- Invalid canonical episodes, mutable state, bids, routes, or model outputs fail
  at their owning contract boundary.
- Model stages own semantic judgment; deterministic code owns contract
  validation, persistence, permissions, limits, and delivery eligibility.
- Resolver observations and RAG rows remain evidence, never final stance.
- A non-speech route produces no visible dialog.
- Surface planning occurs after the final state commit.
- Interaction-style context is surface-only and does not affect cognition route
  selection.
- Dialog owns wording and cannot mutate cognition state or action requests.

## Public Entrypoints

- `persona_supervisor2.persona_supervisor2(...)`
- `persona_supervisor2.stage_1_goal_resolver(...)`
- `persona_supervisor2_cognition.build_cognition_input_from_global_state(...)`
- `persona_supervisor2_cognition.call_cognition_subgraph(...)`
- `persona_supervisor2_cognition.commit_cognition_output(...)`
- `persona_supervisor2_l3_surface.call_l3_text_surface_handler(...)`
- `dialog_agent.dialog_agent(...)`

The public cognition APIs themselves live in `kazusa_ai_chatbot.cognition_core_v2`:
`run_cognition(...)` and `run_text_surface_planning(...)`.
