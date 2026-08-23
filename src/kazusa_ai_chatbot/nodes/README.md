# Cognition Nodes

`kazusa_ai_chatbot.nodes` owns the adapter-neutral connector between the brain
service and the canonical cognition, surface, and dialog owners.

```text
adapter/debug client -> brain intake and relevance
  -> persona_supervisor2
       typed episode and semantic context
       one A1 -> A2 -> G -> P cognition path
       validated state/goal/response-plan commit
       action/resolver authorization
       surface planning and dialog wording
  -> consolidation and persistence
```

RAG and resolver capabilities return evidence. Cognition owns appraisal,
character judgment, causal affect, and response intent. Deterministic code owns
validation, state reduction, permissions, action materialization, persistence,
and delivery. `dialog_agent.py` owns final visible wording.

## Module Boundary

| Area | Main files | Ownership |
| --- | --- | --- |
| Perception | `persona_supervisor2_msg_decontextualizer.py` | Current media observation and role-explicit current-turn meaning. |
| Persona graph | `persona_supervisor2.py` | Commit ordering, action/surface routing, and episode trace assembly. |
| Cognition connector | `persona_supervisor2_cognition.py`, `persona_supervisor2_cognition_actions.py` | Canonical cognition input, output binding, state projection, and action/resolver materialization. |
| Text and visual connector | `persona_supervisor2_l3_surface.py` | Prompt-safe surface context and visible-content planning. |
| Dialog | `dialog_agent.py` | Validated surface output and final visible wording. |
| Consolidation | `persona_supervisor2.py` and consolidation packages | Extraction, origin projection, target validation, and durable writes. |

The nodes consume platform-neutral state after adapters and brain intake have
normalized wire syntax into typed episodes, message context, bounded history,
and semantic participant descriptions.

## Canonical Live Flow

The top-level service graph routes into `persona_supervisor2` only after the
queue, frontline intake, turn settlement, accepted-media description,
settled-relevance gate, and conversation-progress loader have done their work.

Inside `persona_supervisor2`, the live persona graph is a single bounded
canonical cognition call followed by the existing surface and delivery
owners:

```text
stage_0_msg_decontextualizer
  -> stage_1_goal_resolver
       build one canonical cognition input
       run one bounded A1 -> A2 -> G -> P cognition path
       validate and bind one active-character goal and response plan
       apply permitted action/resolver effects through caller-owned contracts
       commit exactly one final replacement state
  -> stage_2_memory_lifecycle
  -> stage_2a_background_work_enqueue
  -> route from the canonical response plan
       speech
         -> build the canonical text-surface input
         -> run one bounded semantic text stage
         -> project visible boundaries and addressees deterministically
         -> run one terminal visual stage as a sibling when enabled
         -> retain the validated text input and output for dialog
         -> dialog_agent renders the text output
         -> text surface, private image evidence, and action-result trace
       non-speech
         -> private terminal handling
         -> action-result trace without visible dialog
```

`decontextualized_input` and role-explicit percept metadata preserve the
current request, active-character ownership, addressee direction, and named
participant descriptions. Deterministic code owns validation, state binding,
permissions, and persistence; cognition owns semantic judgment and response
intent. A malformed canonical result becomes an operational failure before
commit.

## Canonical Cognition Boundary

`persona_supervisor2_cognition.py` builds one canonical input containing:

- the validated canonical episode;
- one validated mutable cognition state and separate character constraints;
- typed episode, media, RAG, resolver, and permitted action-result evidence;
- direct facts with trusted provenance;
- available action and resolver affordances; and
- a bounded semantic scene description;
- bounded continuity and relationship context; and
- semantic action and resolver capabilities available to the caller.

The connector keeps scene, relationship, continuity, and capability context
bounded and caller-owned. Retrieved evidence informs judgment but does not
become persona or final stance merely by being retrieved. Private persistence
identifiers remain outside model-facing packets.

Each turn applies trusted direct facts, elapsed evolution, relationship
maintenance, semantic state binding, affect derivation, and retention as one
validated transaction. Resolver recurrence carries the immutable persisted
base and the exact private continuation-goal reference; it never commits a
later uncommitted replacement as a new compare-and-replace base.

The cognition core preserves all six appraisal families and their axes, derives
emotion with concrete causes, binds one active-character goal, and returns a
canonical response plan. The caller owns private state references and native
action/resolver materialization. The cognition route uses the configured single
chain, fixed A1/A2 appraisal layout, bounded context, disabled thinking, and
the configured turn deadline.

Persistent identifiers and raw numeric state remain behind deterministic
handle bindings. Model-facing projections use semantic roles and qualitative
bands. RAG evidence does not become persona, affect, or final stance merely by
being retrieved.

The cognition output preserves all six appraisal families and axes, structured
emotion causes, one active-character goal, relational willingness, the exact G
`private_monologue`, and a canonical response plan with P's
`epistemic_boundary`. Surface and action consumers receive only validated
semantic projections; they do not re-derive ownership from prose.

## Action Ownership

Cognition returns one active-character response goal and optional semantic
capability requests. The caller validates capability availability, permissions,
target bindings, and parameters before materializing action or resolver work.
Malformed canonical output fails before execution; it cannot grant work.

Memory-lifecycle requests follow a specialist boundary. Cognition may request
a semantic lifecycle review, while the specialist chooses prompt-safe aliases
and deterministic code resolves an eligible persistent row. Cognition does not
select database identifiers or write lifecycle state directly.

Accepted background work is queued before a selected speech surface so the
surface can describe only the actual semantic outcome. A background request
without a visible acknowledgement route receives a deterministic failure
result instead of silently promising work.

## Text, Terminal Visual, And Dialog

`persona_supervisor2_l3_surface.py` runs only after the final cognition state
commit and only for a speech response plan. It builds the canonical text surface
input
from:

- the canonical episode;
- the active-character goal and response plan;
- expression policy;
- semantic affect and optional relationship projections;
- permitted semantic action results; and
- the exact private monologue and epistemic boundary as typed subjective
  expression context;
- the selected speak action's caller-owned addressee roles; and
- bounded interaction-style guidance, exact tempo/linguistic-texture character
  expression, an interaction-scoped recent-character-dialog projection, and an
  isolated visual-character context.

The connector loads the existing sanitized user interaction-style overlay and,
for group turns, the group-channel overlay. It renders only allowlisted speech,
social, pacing, and engagement guidance in application order into the bounded
bounded surface context. Storage identifiers, revisions,
reflection lineage, and raw channel/user identifiers are excluded.

`run_text_surface_planning(...)` projects visible episode content and runs
exactly one bounded semantic stage. Unified content planning atomically
returns `content_plan`, `content_requirements`, the exact five-field
`delivery_profile`, and optional expression-only `lexical_avoidances`; the
upstream relational decision is carried when present. Deterministic code emits
an empty `visible_boundaries` list and copies the validated caller-owned
`addressee_plan`. Content planning receives tempo, linguistic texture, the
expression-only private monologue, and the authoritative epistemic boundary.
The public text output retains that exact epistemic boundary for dialog but
does not retain the private monologue. A physical or external effect reaches
visible wording as completed only from a matching `executed`
`permitted_action_results` row. Delivery fields
describe lexical register, sentence shape, rhythm, hesitation, and punctuation
only; they cannot override the cognition-selected stance.

`lexical_avoidances` is a bounded surface-owned list of concrete current-turn
wording fragments, such as a repeated recent opening or stale address. It is
used for literal expression-continuity checking and never classifies topics or
selects a character stance. The dialog generator preserves the selected
semantics while avoiding those fragments; the check is deterministic and
bounded.

Surface quality fields remain advisory descriptor context. They cannot change
cognition truth, action authorization, role direction, addressee, persistence,
queue, delivery, or state.

When visual directives are enabled, `run_visual_surface_planning(...)` runs as
an independent sibling call. It alone receives the isolated bounded
visual-character context and emits exact image-generation directives. No
downstream image or dialog model consumes them. The persona graph retains them
as a private `image` surface with `do_not_deliver` in the raw episode trace.
Their fragments are audit-only and are excluded from every model-facing
consolidation projection, source view, and router input.

`dialog_agent.py` authors natural, vivid chat-ready words for the character.
Character-consistent invention, ask-backs, playful development, and other
coherent drift remain available when they fit the current input and scene.
Text planning describes what the character wants to express and advances the
interaction without supplying staging forms; dialog carries emotion,
personality, and interaction posture through wording, sentence shape, and
cadence. Action narration remains an ungated model variation: the prompts do
not request it, and generated instances are neither rejected nor rewritten.
The generator receives the validated surface output and emits the
visible `final_dialog`. Canonical JSON parsing, structural message validation,
and deterministic required-source-URL checks remain in the generator boundary.
No semantic verifier, score gate, or evaluator-driven repair follows
generation. Action narration remains an ungated model variation: the prompts
do not request it, and generated instances are neither rejected nor rewritten.

Before this dialog boundary, typed required-selection context remains semantic
episode provenance. Dialog receives only the validated surface projection and
renders its wording.

Dialog does not receive raw mutable state, private cognition payloads,
persistent identifiers, relationship scalars, or implementation directives.

## Cognitive Episodes

`CognitiveEpisode` is the source-neutral current-event boundary. Supported
trigger/source combinations include:

| Trigger | Primary input source | Typical output modes |
| --- | --- | --- |
| `user_message` | `dialog_text` with optional media observations | `visible_reply`, `think_only`, `silent` |
| `tool_result` | `tool_result` | `visible_reply`, `think_only`, `silent` |
| `self_cognition` | `self_cognition_case` | `think_only`, `silent` |
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
caller, validated before cognition, and replaced once after canonical output.
Character drives and meaning constraints can inform a user-scoped turn without
becoming user-owned mutable state. Standards remain in raw character state and
are not projected into live model input until a typed source-bound contract
exists.

The persona graph assembles action results and surface outputs into the existing
episode-trace envelope for downstream diagnostics and consolidation. The trace
records what was validated, attempted, completed, rejected, or surfaced. It is
not a second cognition authority. Terminal private image directives remain in
this raw audit record, while the consolidation projection structurally omits
them before any model-facing source view or router input is built.

Consolidation runs after the live wording path. It consumes prompt-safe episode
and trace evidence, plans eligible persistence targets, and applies writes
through its own validated lanes. Nodes do not let consolidation execute
actions, deliver messages, schedule work, or reopen cognition.

## Failure And Safety Rules

- Missing or partial canonical cognition output fails before surface routing.
- A structural/provider failure remains a typed operational failure before
  state commit; deterministic code never invents semantic values or executes
  unapproved work.
- Invalid episodes, mutable state, routes, commit failures, and unavailable
  capabilities remain unrecoverable at their owning boundary.
- Model stages own semantic judgment; deterministic code owns contract
  validation, persistence, permissions, limits, and delivery eligibility.
- Resolver observations and RAG rows remain evidence, never final stance.
- A non-speech route produces no visible dialog.
- Surface planning occurs after the final state commit.
- General user/group interaction-style composition remains surface-only. The
  exact group-engagement projection is limited to eligible group
  self-cognition goal and action judgment and has no route authority.
- Dialog owns wording and cannot mutate cognition state or action requests.

## Public Entrypoints

- `persona_supervisor2.persona_supervisor2(...)`
- `persona_supervisor2.stage_1_goal_resolver(...)`
- `persona_supervisor2_cognition.build_cognition_input_from_global_state(...)`
- `persona_supervisor2_cognition.call_cognition_subgraph(...)`
- `persona_supervisor2_cognition.commit_cognition_output(...)`
- `persona_supervisor2_l3_surface.call_l3_text_surface_handler(...)`
- `dialog_agent.dialog_agent(...)`

The connector imports `kazusa_ai_chatbot.cognition_core_v3` directly, so every
live, idle, and self-cognition call site runs the same agentic engine. Text and
visual surface planning use the canonical cognition facade and validators.
