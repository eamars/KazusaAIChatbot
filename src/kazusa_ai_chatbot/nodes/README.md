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
appraisal, causal state changes, present character judgment, bid collapse,
route selection, and response goals. Prior conversation and private residue
inform that judgment without commanding one repeated posture. Deterministic
connectors own validation, persistence, action materialization, permissions,
limits, and graph routing. The V2 surface planner owns expressive content and
only real visible boundaries. `dialog_agent.py` owns final visible wording.

## Module Boundary

| Area | Main files | Ownership |
| --- | --- | --- |
| Perception | `persona_supervisor2_msg_decontextualizer.py` | Current media observation, current-message rewrite, referent status, and one role-explicit current-turn meaning after the brain-service relevance settlement boundary. |
| Persona graph | `persona_supervisor2.py` | Resolver recurrence, final commit ordering, action/surface routing, no-response handling, and episode trace assembly. |
| V2 connector | `persona_supervisor2_cognition.py`, `persona_supervisor2_cognition_actions.py` | Exact `CognitionCoreInputV2` construction, state loading, V2 service binding, output projection, final state replacement, and semantic action-request materialization. |
| Text and terminal visual connector | `persona_supervisor2_l3_surface.py` | Prompt-safe interaction-style loading, exact `TextSurfaceInputV2` construction, two-call unified-content/preference planning, and independent one-call visual planning. |
| Dialog | `dialog_agent.py` | Literal spoken or typed text from `TextSurfaceOutputV2`, bounded current-visible-percept and authoritative-surface verification, accepted-surface return, and one full repair maximum. |
| Specialist action handling | `persona_supervisor2_memory_lifecycle.py`, action-spec packages | Deterministic validation and execution of admitted semantic action requests. |
| Consolidation handoff | `persona_supervisor2.py` | Completed persona state is handed to `kazusa_ai_chatbot.consolidation`, which owns extraction helpers, origin projection, target validation, and durable write routing. |

Semantic relevance is owned by `kazusa_ai_chatbot.relevance`, whose interface
document defines the frontline intake and settled character-response agents.
This package consumes their validated decisions through the brain-service
settlement boundary; it does not import their prompts, model instances, or
private projections.

The nodes consume platform-neutral state. Platform wire syntax must already be
normalized by adapters and the brain service into `message_envelope`,
`prompt_message_context`, `reply_context`, `CognitiveEpisode`, global user ids,
and bounded history fields.

## Canonical Live Flow

The top-level service graph routes into `persona_supervisor2` only after the
queue, frontline intake, turn settlement, accepted-media description,
settled-relevance gate, and conversation-progress loader have done their work.

Inside `persona_supervisor2`, the live persona graph is:

```text
stage_0_msg_decontextualizer
  -> stage_1_goal_resolver
       start eligible cycle-zero shared-memory prewarm
       load one mutable user or character V2 state
       join shared-memory preparation before native V2 input construction
       reuse the service-owned interaction-style turn snapshot
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
         -> run three bounded text stages
         -> run one terminal visual stage as a sibling when enabled
         -> retain the validated text input and output for dialog
         -> dialog_agent renders the text output
         -> one L3 semantic replacement and second render after hard rejection
         -> text surface, private image evidence, and action-result trace
       non-speech
         -> private terminal handling
         -> action-result trace without visible dialog
```

For a live user message, Stage 0 returns semantic surfaces from its existing
LLM call. `decontextualized_input` remains a natural equivalent used by
compatibility and retrieval paths. Optional `role_explicit_content` uses the
Chinese role labels `当前用户` and `当前角色` to preserve nested actor, target,
beneficiary, modality, and request direction. Optional structured
`response_operation` records the response owner, whether an unsupplied answer
or choice is required and who owns it, plus embedded actor and target roles.
Deterministic code validates exact shape, enums, booleans, and bounds, then
attaches the model-owned values unchanged to existing dialog-percept metadata.
The raw percept content remains available beside this projection.

Group decontextualization also receives a bounded, display-name-only roster of
episode-local third-party bindings (`p1` through `pN`). When the model resolves
a third-party referent, it must carry the matching `participant_handle`; an
unknown handle, mismatched display name, or unresolved referent is a bounded
contract error. These bindings are transient semantic context, not global
user identities and not persistence inputs. Cognition, L3, and dialog share
the same binding and its structured addressee policy. A `pN` target must remain
named or explicitly third-person in visible wording, while the current user's
delivery identity remains the existing transport recipient. No deterministic
post-generation replacement changes the text after dialog rendering.

Provider or invalid decontextualizer output receives up to three total local
attempts. A structural repair carries only the latest bounded rejected
candidate and exact nested-field validation error. Exhaustion retains the
normalized original input, omits uncertain role projection, and continues as
accepted degraded output before state commit.

The image descriptor likewise uses three total attempts and accepts only the
exact five-field descriptor contract. Only validated descriptors enter the
media cache. Exhaustion or a stale malformed cache row produces a typed
unavailable observation for the current turn while preserving future recovery.

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
- a bounded semantic scene description;
- distinct private past-dialog continuity for goal cognition only; and
- exact advisory group-engagement guidance for eligible targetless group
  self-cognition.

At resolver cycle zero, `user_message` and `internal_thought` episodes start
the existing shared-memory prewarm before independent identity and mutable-state
preparation. Its confirmed shared rows merge only into
`rag_result.memory_evidence` before V2 evidence mapping. Empty results preserve
the base RAG payload, and later resolver cycles reuse that state without
another lookup.

Canonical targetless group self-cognition consumes the group-channel engagement
projection from the immutable interaction-style snapshot loaded once by the
service before the graph. Goal cognition and action planning receive the same
bounded advisory value; appraisal, workspace collapse, surface planning, and
dialog do not. Ordinary user turns and other ineligible episodes receive the
exact empty value without a connector-owned group style database read.

When Stage 0 supplied a valid semantic projection, the connector forwards its
`role_explicit_content` and `response_operation` unchanged as current episode
evidence and semantic scene. Goal cognition, surface planning, and dialog
verification therefore share one role and ownership meaning instead of
independently interpreting nested direct pronouns.

The V2 core performs deterministic preparation, scoped semantic appraisal,
state reduction, dependency-ready goal cognition, complete-bid collapse, and
route validation. Its output contains the replacement state, selected semantic
intention, admitted/supporting bids, semantic affect and relationship
projections, action requests, resolver requests, progress, expression policy,
diagnostics, and bounded residue.

The connector constructs twelve independent required Core V2 route
bindings:

```text
COGNITION_LLM_APPRAISAL_EVENT_AGENCY
COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL
COGNITION_LLM_APPRAISAL_MORAL_IDENTITY
COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME
COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY
COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE
COGNITION_LLM_GOAL_ORDINARY_RESPONSE
COGNITION_LLM_GOAL_ACTIVE_BRANCH
COGNITION_LLM_WORKSPACE_COLLAPSE
COGNITION_LLM_ACTION_PLANNING
COGNITION_LLM_ACTION_AUTHORIZATION
COGNITION_LLM_RESOLVER_AUTHORIZATION
```

Each binding has its own endpoint, credential, model, completion budget, and
thinking setting. Missing required values stop configuration loading. The
connector passes the bindings directly to their semantic owners, including
the same owner config for retries, repairs, and traces. The generic
`COGNITION_LLM` route remains available to callers outside the Core V2
boundary. The connector adds no route inheritance, endpoint selection logic,
or fallback.

Persistent identifiers and raw numeric state remain behind deterministic
handle bindings. Model-facing projections use semantic roles and qualitative
bands. RAG evidence does not become persona, affect, or final stance merely by
being retrieved.

Promoted-memory evidence keeps its relationship scope after prewarm. The V2
connector maps each `rag_result.memory_evidence` row to exactly one prompt-safe
`memory_scope`: rows that already carry `scope_type=user_continuity` become
`current_user_continuity`, and every other promoted-memory row (including
cycle-zero shared prewarm rows) becomes `shared_character_or_world`. The raw
user id and storage provenance stay behind the deterministic boundary. Shared
character/world memory can inform what the character knows but cannot establish
current-user relationship access; current-user continuity memory explains
history without overriding the canonical native relationship state.

The ordinary goal owner in Cognition Core V2 decides current-turn relational
willingness. Each goal evidence row receives one transient `provenance_role`
derived from trusted source-kind and memory-scope metadata; promoted
reflection and shared character/world context cannot grant current-user access.
The connector passes the validated `relational_willingness.v2` decision
(applicability, `current_user_relationship_state`, stance, reason, and
evidence handles) through the V2 output and the L3 surface input so workspace,
action permission, content planning, preference planning, and dialog preserve
the exact stance and relationship state instead of re-deriving them from prose
or relationship numbers.

## Action Ownership

V2 goal branches may bid for speech, silence, private handling, an action, or
resolver evidence. Complete bids retain their semantic intention, desired
outcome, grounded detail, target roles, consequences, route, and declared
request until deterministic collapse and validation finish.

Only admitted `action_requests` are materialized into the existing action-spec
execution boundary. Deterministic code revalidates capability availability,
permissions, target bindings, and parameters. Action specs and action results
remain trace/execution artifacts; they do not own cognition route selection.
The proposal boundary keeps valid canonical rows and drops malformed rows
individually. Three unusable planning attempts degrade to an empty plan, and
three unusable authorization attempts deny all proposed work. Speech remains
available, while no malformed model output can grant execution.

Memory-lifecycle requests follow a specialist boundary. Cognition may request
a semantic lifecycle review, while the specialist chooses prompt-safe aliases
and deterministic code resolves an eligible persistent row. Cognition does not
select database identifiers or write lifecycle state directly.

Accepted background work is queued before a selected speech surface so the
surface can describe only the actual semantic outcome. A background request
without a visible acknowledgement route receives a deterministic failure
result instead of silently promising work.

## V2 Text, Terminal Visual, And Dialog

`persona_supervisor2_l3_surface.py` runs only after the final cognition state
commit and only for a speech intention. It builds exact `TextSurfaceInputV2`
from:

- the canonical episode;
- the selected intention;
- bounded primary and supporting bid projections;
- expression policy;
- semantic affect and optional relationship projections;
- permitted semantic action results; and
- bounded interaction-style guidance, exact tempo/linguistic-texture character
  expression, and an isolated visual-character context.

The connector loads the existing sanitized user interaction-style overlay and,
for group turns, the group-channel overlay. It renders only allowlisted speech,
social, pacing, and engagement guidance in application order into the bounded
string required by `TextSurfaceInputV2`. Storage identifiers, revisions,
reflection lineage, and raw channel/user identifiers are excluded.

`run_text_surface_planning(...)` projects visible episode content and runs
exactly two bounded stages in parallel. Unified content planning atomically
returns `content_plan`, `content_requirements`, and the exact five-field
`delivery_profile`; preference planning returns only `visible_boundaries` and
`addressee_plan`. Unified content receives tempo and linguistic texture,
whereas preference receives no character-expression context. Delivery fields
describe lexical register, sentence shape, rhythm, hesitation, and punctuation
only; they cannot override the cognition-selected stance.

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
Three bounded hard-error checks run in parallel on the existing dialog-model
route. Semantic fidelity receives current percepts, the candidate role frame,
candidate dialog, and authoritative surface semantics containing selected
intent, content plan, requirements, and visible boundaries. It checks internal
contradiction, direct current-input conflict, actor/target/subject reversal,
and unsupported within-turn opposite-stance transitions. Delivery profile,
permitted action results, and selection-owned role fields are excluded from
this semantic authority. Surface integrity receives only permitted action
results and candidate dialog; it checks false claims of character-brain action
execution. Source percepts and generated character speech use separate typed
pronoun frames before actor/action/target comparison. Semantic fidelity owns
non-selection role direction. The selection-only role verifier receives the five authoritative
`response_operation` fields and owns selection transfer plus embedded
actor/target reversal. Those selection-owned fields are removed from the
semantic-fidelity projection while the raw current-input meaning remains.
Deterministic code merges the three verdicts without rewriting dialog
semantics. Each owner returns at most four issues and the duplicate-free
merged result returns at most eight. None of the checks treats novelty or
personality strength as failure. Refusal, negotiation, and conditions chosen
by the character remain valid responses unless typed actor, target,
response-owner, or selection-owner direction is reversed.

Each semantic-fidelity, role-direction, and surface-integrity producer gets
three structural attempts. An exact-shape failure retains the original system
and human semantic packet, places the latest bounded rejected verdict in an
assistant message, and supplies the exact contract error in a human
correction. This does not re-render dialog or add another semantic owner.
Protected traces retain every attempt. Exhaustion marks that focused owner
`unavailable`; the valid dialog candidate remains eligible.

The semantic-fidelity producer returns exact `aligned` and `hard_errors`
fields. This avoids the local model's demonstrated `issues`/`Issues` token
collision. Deterministic validation accepts no alias and normalizes the
validated `hard_errors` list only after exact validation. Role direction uses
an exact `aligned` and typed `violations` contract. Its only violation kinds
are `selection_owner_transfer` and `typed_operation_role_reversal`. Surface
integrity retains its evidence-bearing `aligned` and `issues` contract.

A negative merged verdict returns the retained canonical
`TextSurfaceInputV2` and bounded verified issues to the L3 owner. Rejected
surface fields and rejected dialog remain protected trace evidence only. L3
replaces content, requirements, delivery profile, visible boundaries, and
addressee together, then reconstructs selected intent, permitted action
results, and runtime limits from canonical input. If surface replacement
exhausts, dialog retains the latest validated surface. Dialog renders and
verifies one second candidate without receiving the first dialog. A second
rejection renders one terminal third candidate from canonical surface truth
and typed remaining violation kinds, without verifier calls. A bounded third
candidate is delivered as degraded output; an unusable third candidate falls
back to candidate two, then candidate one. Only zero usable candidates is
unrecoverable. The persona graph follows the normal post-turn path with the
selected dialog and retained valid surface.

Before this dialog boundary, a typed character-owned required selection routes
the selected goal branch to one specialized producer in place of its generic
goal prompt. The producer emits one authoritative selection and accounts for
every required-selection handle while retaining complete progress evidence for
its own relevance judgment. Structural retries reuse that same goal owner; no
semantic evaluator or replacement owner is added. Dialog's
corresponding focused check reads the canonical nested
`percept.content.response_operation` and owns only explicit selection-owner
transfer and actor/target reversal. Semantic completeness, brevity, and
specificity remain with semantic fidelity and the L3 surface owner. A
character-owned desire, request, or imperative can name the selected action,
and speaking or sending content counts as an action when the typed operation
requires it. Turns without the structural flag use the generic goal producer.

Dialog does not receive raw V2 mutable state, private branch payloads,
suppressed bids, persistent handles, relationship scalars, or obsolete
directive bags.

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
caller, validated before cognition, and replaced once after terminal V2 output.
Character drives, standards, and meaning constraints can inform a user-scoped
turn without becoming user-owned mutable state.

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

- Missing or partial V2 cognition output fails before surface routing.
- Recoverable V2 model failures use their declared total attempt budget. Goal
  cognition's three calls are cumulative per producing stage and branch across
  the service graph retry; an orchestration replay cannot reset them.
- Degradable exhaustion finishes with the owner fallback: normalized original
  input, omitted optional appraisal or visual output, already-valid bid,
  empty or denied control work, validated neutral text surface, unavailable
  verifier, or retained bounded dialog.
- Invalid canonical episodes, mutable state, bids, routes, required zero-valid
  cognition after the complete-sibling policy, commit failures, and
  zero-candidate total model unavailability remain unrecoverable at their
  owning boundary. Unsupported goal handles are regenerated or rejected and
  are never deterministically deleted into acceptance.
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

The public cognition APIs themselves live in `kazusa_ai_chatbot.cognition_core_v2`:
`run_cognition(...)`, `run_text_surface_planning(...)`, and
`run_visual_surface_planning(...)`.
