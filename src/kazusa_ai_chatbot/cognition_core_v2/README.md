# Cognition Core V2

`cognition_core_v2` owns the validated persistent cognition state used by
Stage 2. User state and the singleton character state are separate mutable
scopes. The exact state contract is enforced by `state_models.py`; structured
role references, complete evidence records, canonical singular entity kinds,
bounded axes, root ownership, and activation identity are validated before
state crosses the database boundary.

`transition_guards.py` accepts only trusted direct facts and bounded semantic
deltas. `state_reducers.py` performs elapsed evolution, cause-first event
comparison, guarded goal creation and lifecycle transitions, deterministic
event identity, retention, and activation-cache recomputation. Emotion rows are
derived projections: every activation retains typed roots, phase, trend, score,
cause status, and timestamps.

Evidence retention is deterministic and bounded: relationship state keeps the
newest eight unique rows, while causal entities keep their first/root row and
newest seven unique rows. A terminal meaning may repeat idempotently only in
the appraisal batch that produced that terminal transition; later batches
still observe strict terminal immutability. `relationship_connection` is owned
by the typed desired-versus-perceived closeness gap and is satisfied when that
gap closes.

The twenty-one emotion formulas are exercised from typed natural causes in
`tests/test_cognition_core_v2_emotion_lifecycle.py`. Cross-scope character
constraints and optional relationship context are passed as dedicated
projections; they are not merged into mutable state. Character sleep recovery
is deterministic and separate from user elapsed decay.

Database-backed callers use `db.users` for user-owned state and `db.character`
for the character singleton. The test database harness requires the exact
`_test_kazusa_live_llm` name, validates seeded V2 state, and gives every
mutable test row a unique owner.

The public Stage 2 surface consists of `run_cognition(...)`,
`run_text_surface_planning(...)`, `repair_text_surface_planning(...)`, and
`run_visual_surface_planning(...)`.
Cognition runs deterministic preparation, scoped semantic appraisal,
dependency-ready goal branches, complete-bid collapse, route validation, and
one replacement-state update. The caller commits that update before action,
surface, resolver, or dialog work.

## Stage Model Routing

Core V2 receives one independent `LLMCallConfig` for each existing semantic
model owner:

| Semantic owner | Service field | Environment route |
|---|---|---|
| Event and agency appraisal | `appraisal_event_agency_config` | `COGNITION_LLM_APPRAISAL_EVENT_AGENCY` |
| Relationship and social appraisal | `appraisal_relationship_social_config` | `COGNITION_LLM_APPRAISAL_RELATIONSHIP_SOCIAL` |
| Moral and identity appraisal | `appraisal_moral_identity_config` | `COGNITION_LLM_APPRAISAL_MORAL_IDENTITY` |
| Goal, threat, and outcome appraisal | `appraisal_goal_threat_outcome_config` | `COGNITION_LLM_APPRAISAL_GOAL_THREAT_OUTCOME` |
| Epistemic, comparison, and memory appraisal | `appraisal_epistemic_comparison_memory_config` | `COGNITION_LLM_APPRAISAL_EPISTEMIC_COMPARISON_MEMORY` |
| Existential and drive appraisal | `appraisal_existential_drive_config` | `COGNITION_LLM_APPRAISAL_EXISTENTIAL_DRIVE` |
| Ordinary-response goal | `goal_ordinary_response_config` | `COGNITION_LLM_GOAL_ORDINARY_RESPONSE` |
| Active persistent-goal branches | `goal_active_branch_config` | `COGNITION_LLM_GOAL_ACTIVE_BRANCH` |
| Required-selection verification | `required_selection_verifier_config` | `COGNITION_LLM_REQUIRED_SELECTION_VERIFIER` |
| Workspace collapse | `workspace_collapse_config` | `COGNITION_LLM_WORKSPACE_COLLAPSE` |
| Action planning and goal resolution | `action_planning_config` | `COGNITION_LLM_ACTION_PLANNING` |
| Action authorization | `action_authorization_config` | `COGNITION_LLM_ACTION_AUTHORIZATION` |
| Resolver authorization | `resolver_authorization_config` | `COGNITION_LLM_RESOLVER_AUTHORIZATION` |

Every initial call, provider retry, structural replacement, and trace row uses
the config selected by that semantic owner. Required-selection verification
has its own route, while a replacement bid returns to the goal route that
produced the bid. Stage routes are complete required environment bundles and
have no route inheritance or fallback. The generic `COGNITION_LLM` route
continues to serve cognition callers outside this Core V2 boundary.

The existing first wave remains unchanged: six appraisal families, the
ordinary-response goal, and dependency-ready active-goal branches can submit
up to twenty model tasks concurrently. Final dependency-ready goal work,
workspace collapse, action planning, and the applicable authorization stage
remain ordered after that wave. Routing changes endpoint ownership only; model
call count, prompts, schemas, attempt caps, and DAG edges stay unchanged.

Goal, threat, and outcome appraisal uses affirmative entity-specific terminal
assertions: `goal_completed`, `event_completed`, `threat_resolved`,
`event_repaired`, and `knowledge_answered`, alongside goal release and
supersession. `outcome_pending` records an explicit nonterminal observation
without state mutation or candidate materialization. The payload maps subject,
object, role-assignment entity, and evidence fields to their exact handle
domains. It also maps the structured `self` and `current_user` handles to their
Chinese semantic-text references so role fields and prose use the same actor
roots without sharing representations. Structural validation binds each
terminal kind to its exact entity kind. A valid assertion atomically
establishes the terminal axis before the unchanged transition guard runs,
including when the assertion first materializes a causal candidate. Terminal
candidates bypass the nonterminal salience-pruning path. After all same-batch
numeric observations are reduced, the accepted terminal assertions reassert
their canonical axes without repeating or weakening the guarded transition.
Every candidate is trial-reduced inside the appraisal's existing attempt cap.
Final reduction validates each added appraisal by replaying the bounded
accepted prefix from the original state, so handle composition is preserved
and one residual rejection is omitted without discarding other appraisal
results or the preliminary character response.

Current-event scene text, public conversation continuity, and private residue
continuity are separate inputs. Private continuity reaches goal-cognition
branches only and remains non-binding prior context: each branch decides
whether the current event, affect, relationship, and active goal call for
progressing, revising, or leaving that posture. Branch bids distinguish
analytic `reason` from first-person `private_monologue`; public output exposes
that distinction as `selected_bid_reason` and `private_monologue`.

For user dialog, the canonical percept may carry bounded
`role_explicit_content` and structured `response_operation` values authored by
the existing upstream decontextualizer LLM. The operation identifies the
response owner, any required selection owner, and embedded actor and target.
The raw sentence and deterministic speaker/addressee frame remain intact. V2
consumes this semantic projection unchanged as current episode meaning so
nested role and response ownership are resolved once before goal cognition
instead of independently by every downstream local-model stage.

Goal-bid output uses an exact route-to-capability-field matrix. A malformed bid
receives up to three total LLM attempts while deterministic validation remains
strict. A still-failed required branch requests the existing clean graph retry
and then raises an execution error rather than becoming an empty workspace and
character silence. Every goal attempt is eligible for the protected turn trace.

When upstream episode evidence carries a typed required selection, a focused
goal-level check verifies that the owning role makes or explicitly expresses
that choice before action planning. Provider or contract failure receives up
to three verifier attempts; exhaustion marks the check unavailable and retains
the newest structurally valid bid as degraded. A negative semantic verdict may
produce up to two replacement bids from typed operations, current evidence,
affect, relationship, character constraints, and scene context without the
rejected bid or private continuity prose. Each replacement is rechecked by the
same bounded owner. Ordinary turns add no selection check.

Action planning treats local-model output as a bounded proposal rather than an
execution precondition. It canonicalizes the known envelope, keeps usable
rows, drops invalid rows individually, ignores unknown fields, and caps each
request list at three. Mutually exclusive action and resolver requests remain
a semantic contract error. If three total planning attempts remain unusable,
the turn continues with an empty action plan; if three total authorization
attempts remain unusable, every candidate is denied. Neither containment path
authorizes work, changes the visible speech route, or reduces the
registry-driven three-request capacity.

The action-planning envelope also carries the required Cognition-Core-owned
`goal_resolution`: `answerable_now`, `requires_required_evidence`,
`requires_user_input`, or `blocked`. This is the semantic judgment of whether
the accepted user goal can be answered now; it is distinct from any
source-specific RAG `resolved` field. `answerable_now` suppresses optional
resolver requests before resolver authorization and recurrence. Required
evidence, user-input, and technical-blocked decisions retain their existing
typed paths. Deterministic code validates and enforces the decision without
reclassifying it from keywords or adding another LLM stage.

The shared surface input receives semantic intention, bounded affect and
relationship projections, complete-bid projections, permitted action results,
interaction style, an exact tempo/linguistic-texture expression context, and a
separate bounded visual-character context. Normal text planning makes exactly
two parallel calls. Unified content planning atomically returns the content
plan, requirements, and a five-field delivery profile; preference planning
returns only real visible boundaries and addressee constraints. Neither call
receives the visual-character context, and preference receives no character
expression context. The delivery profile is limited to lexical register,
sentence shape, rhythm, hesitation, and punctuation and cannot authorize a
semantic stance. The exact text output contains no raw character profile or
visual directives. The independent one-call visual planner may observe the
isolated visual-character context and produces image-generation directives as
terminal private evidence; it has no downstream image or dialog model. Raw
episode traces retain those directives for audit, while every model-facing
consolidation projection excludes their fragments.

Content planning expresses the selected character judgment using the current
scene, affect, relationship, and interaction style. Coherent imaginative
detail is allowed when it remains compatible with current input, active
constraints, and actor/target/subject roles. Preference planning emits only
real visible boundaries and addressee constraints, so both lists may be empty.
Dialog owns natural character-specific chat-ready wording. Three focused
hard-error checks run in parallel on the existing dialog-model route within
the bounded verifier path. Current visible percepts retain their shared
24,000-character cap. Semantic fidelity separately caps authoritative surface
semantics at 11,000 characters, candidate dialog at 12,000 characters, and its
complete serialized payload at 50,000 characters. Semantic fidelity receives
current model-visible percept rows, the candidate role frame, candidate
dialog, and the authoritative selected surface intent, content plan,
requirements, and visible boundaries. It rejects internal contradiction,
direct current-input conflict, non-selection role reversal, and unsupported
within-turn opposite-stance transitions. Delivery profile and action-result
fields are excluded from semantic authority. Role direction receives only typed
selection-required role tuples and rejects selection-owner transfer or
actor/target reversal. Selection-required role fields are excluded from the
semantic-fidelity projection, which retains the raw current-input meaning and
cannot rejudge role-owned operation completion. Surface integrity receives
permitted action results and candidate dialog; it rejects only false system,
tool, platform, or other character-brain execution claims.
Text planning owns expressed meaning and interaction progress without
supplying staging forms. Dialog expresses emotion, character, and interaction
posture through sendable wording and cadence. Action narration is outside the
fatal taxonomy and remains unchanged when the model produces it; the runtime
prompts neither request it nor create a rejection or repair rule for it.
Source percepts and generated character speech carry separate typed pronoun
frames before role direction is compared. Novelty, coherent drift,
character-owned refusal, negotiation, and supported changes of mind are not
failures by themselves. Deterministic code merges only the verdict shapes,
bounding each owner to four issues and the merged result to eight. A negative
result returns canonical surface input plus bounded verified issues to the
text-surface owner for one complete replacement of `content_plan`,
`content_requirements`, `delivery_profile`, `visible_boundaries`, and
`addressee_plan`. Rejected surface fields and rejected dialog are trace-only
and are absent from both repair-model payloads. Selected intent, action truth,
and runtime capability limits are reconstructed from canonical input before
dialog renders candidate two. If that candidate is also rejected, dialog
renders one terminal third candidate from canonical V2 truth and typed
remaining violation kinds. Candidate three is not verified. A bounded,
non-empty third candidate is delivered as degraded output; if it is unusable,
candidate two and then candidate one remain eligible in newest-first order.
Only total generator unavailability with no bounded candidate is
unrecoverable. The returned dialog retains the exact latest valid surface, and
only that pair can reach post-turn consumers. The protected turn trace records
rejected checks, surfaces, and dialog candidates as diagnostic evidence.

Each focused verifier validates its own exact JSON verdict. A structurally
invalid parsed verdict receives up to two complete replacements using the
unchanged system and semantic payload plus the latest bounded rejected
assistant candidate and exact contract error. The replacement remains inside
that verifier and does not create another dialog candidate. All attempts are
recorded in the protected trace. Semantic fidelity uses the
collision-resistant producer field `hard_errors`; deterministic validation
normalizes it only after exact shape validation. Role direction uses typed
`violations` limited to `selection_owner_transfer` and
`typed_operation_role_reversal`; surface integrity retains evidence-bearing
`issues`. Exhaustion marks only that verifier `unavailable`, so it cannot erase
a structurally valid dialog candidate.

## Document Control

Stage 2 native cognition contract. Source of truth: the V2 contracts,
state models, reducers, and focused test suites in this package.

## Purpose

Provide one bounded cognition implementation for persistent user and
character state, semantic appraisal, goal evolution, emotion lifecycle
derivation, and surface planning.

## Boundary

Callers provide typed episode evidence and validated state. This package owns
semantic cognition and replacement-state production; persistence, action
execution, dialog wording, and adapter delivery remain downstream owners.

## Public Entrypoints

- `run_cognition(...)`
- `run_text_surface_planning(...)`
- `repair_text_surface_planning(...)`
- `run_visual_surface_planning(...)`
- `validate_cognition_input(...)`
- `validate_cognition_core_output(...)`

## Runtime Flow

Input validation, bounded semantic appraisal, dependency-ready goal branches,
complete-bid collapse, route validation, replacement-state reduction, and
typed output validation run in one inspectable call.

## Failure Behavior

Every recoverable V2 producer and verifier has at least three total local
attempts, while existing longer semantic ledgers keep their cap. The outcome
ladder is `accepted`, `recovered`, `accepted_degraded`, then `unrecoverable`.
Appraisal and optional visual exhaustion are omitted; decontextualization keeps
the normalized original input; workspace keeps the highest-priority complete
bid; action planning returns no work; authorization denies; and text-surface
exhaustion projects a validated neutral surface from canonical V2 truth.

Malformed canonical input, invalid persistent state, unsupported routes,
unresolved required dependencies after the existing clean graph retry, failed
commit or post-commit invariants, and total model unavailability with no owned
fallback remain execution errors. Recoverable and degraded outcomes follow the
normal persistence and delivery path. Callers commit only validated replacement
state.

## Testing Contract

Run the focused V2 contract, state, emotion-lifecycle, failure, and reflection
settling suites with the project virtual environment. Live LLM cases run one
case at a time with their trace artifact inspected.

## Forbidden Paths

This package does not access adapters, raw database clients, final dialog
wording, platform wire syntax, or untyped relationship scalars.
