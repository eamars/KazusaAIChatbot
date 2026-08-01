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

Character identity is resolved from the latest immutable revision once per
episode. Appraisal owners receive only their bounded identity partitions.
Goal cognition receives `core`, `personality`, `boundaries`, and `self_image`,
so reviewed changes to backstory, character judgment, boundaries, or
self-concept can alter later goals without exposing revision history or old
values. Text and visual surfaces receive separate expression-only projections.

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
| Ordinary-response and required-selection goal | `goal_ordinary_response_config` | `COGNITION_LLM_GOAL_ORDINARY_RESPONSE` |
| Active persistent-goal branches without typed required selection | `goal_active_branch_config` | `COGNITION_LLM_GOAL_ACTIVE_BRANCH` |
| Workspace collapse | `workspace_collapse_config` | `COGNITION_LLM_WORKSPACE_COLLAPSE` |
| Action planning and goal resolution | `action_planning_config` | `COGNITION_LLM_ACTION_PLANNING` |
| Action authorization | `action_authorization_config` | `COGNITION_LLM_ACTION_AUTHORIZATION` |
| Resolver authorization | `resolver_authorization_config` | `COGNITION_LLM_RESOLVER_AUTHORIZATION` |

Every initial call, provider retry, structural replacement, and trace row uses
the config selected by that semantic owner. A typed required-selection turn
uses a specialized producer on the dense ordinary-goal route regardless of its
branch; it replaces that branch's generic goal call and adds no evaluator
route. Active branches without a typed required selection retain the
active-goal route. Stage routes are complete required environment bundles and
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

Trace-backed past-dialog continuity uses its own optional
`past_dialog_cognition_context` carrier, capped at 1,800 characters. It reaches
goal cognition only, separately from the 1,000-character internal-monologue
`private_continuity_context`. It is weak private context for understanding an
already-linked prior character dialog, not evidence, a command, selected
stance, action-planning input, surface content, or dialog wording.

Targetless group self-cognition may receive one exact
`group_engagement_action_context` containing bounded
`engagement_guidelines` and a semantic `confidence` descriptor. The same
immutable advisory projection reaches goal cognition and action planning only.
It helps judge participation in the currently observed group scene; it cannot
create a topic, fact, relationship belief, permission, route, or unsupported
reason to speak. Appraisal and workspace collapse receive no copy.

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

When upstream episode evidence carries a typed required selection, deterministic
routing selects one specialized goal producer instead of the generic goal
producer. Deterministic code partitions its input into authoritative required
operations, complete model-visible conversation-progress evidence, and
optional supporting evidence. The producing call emits one authoritative
`selection`, its selection kind, reason, role/evidence handles, consequences,
and confidence. It must cite every required operation. The goal LLM cites only
progress rows that materially constrain the current choice and leaves
unrelated history uncited. Completed, rejected, and superseded progress remains
model-visible and may be reopened only when the current input explicitly
requests it. RAG conversation-history rows remain optionally citeable
supporting evidence.
Deterministic validation owns exact fields, bounds, provenance partitioning,
required-operation handle coverage, and mapping the one selection into the
existing complete bid. The goal LLM owns progress relevance and the actual
choice, with no discarded relation matrix. Structural failures retry the same
producing prompt under the existing bounded goal contract, with the validation
error plus exact required and allowed evidence handle sets attached to each
complete regeneration. Required-selection parsing
uses deterministic cleanup only, so malformed output cannot invoke the shared
JSON-repair model. There is no semantic verifier, negative verdict,
evaluator-authored replacement, or recheck. Ordinary turns keep the ordinary
goal producer.

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
fallback remain execution errors. Goal cognition may deliver a degraded
selection after its local attempts are exhausted when the parsed candidate is
complete and only contains invalid evidence handles; deterministic projection
retains required valid evidence and drops the invalid references. Other goal
contract failures remain execution errors. Recoverable and degraded outcomes
follow the normal persistence and delivery path. Callers commit only validated
replacement state.

All four public entrypoints capture their raw arguments before validation in a
ContextVar-isolated protected failure buffer. `repair_text_surface_planning`
captures both the raw surface input and `verified_hard_issues`; the other
entrypoints capture their raw input payload. Clean runs discard the buffer.
Terminal exceptions, failed appraisals or branches, recovered model attempts,
and degraded surfaces schedule one failure capsule without delaying or
changing the returned output or raised exception.

The direct V2 model owners—semantic appraisal, goal cognition, workspace
collapse, action planning, semantic authorization, and generic surface
stages—record every provider and contract attempt with its one-based attempt
index, non-secret call configuration, exact messages and response, parsed
output, and concrete error. The canonical JSON-repair fallback records its own
model call into the active capsule without changing parser or retry behavior.

## Aggregate Prompt Budgets

Each V2 model owner budgets its complete deterministic serialization rather
than relying on independent producer-field limits. Semantic appraisal owns an
8,000-character packet containing its question contract, one top-level
semantic-evidence registry, and its authorized state projection. Exact
`permitted_delta_paths` remain private validator authority; the model receives
grouped `state_field`, `handles`, and `axes` domains and returns the same
canonical `state_field.handle.axis` target path.

Appraisal and goal cognition reduce supplemental projected state first. If the
packet remains over its fixed cap, they retain every evidence row, handle,
source kind, and source order while middle-truncating semantic text from the
lowest-priority row backward. Both owners preserve at least 96 characters per
reduced evidence text, or the complete original when it is shorter. Goal
cognition owns the equivalent single-registry packet under its fixed
24,000-character cap, with the current episode retaining highest source
priority. Its goal projection and canonical role summaries are each serialized
once; duplicate evidence, goal projection, role summaries, and scene role
labels are absent from supplemental semantic context.
Past-dialog and group-engagement contexts are supplemental and are removed in
a stable order before required evidence text is reduced. Action planning uses
the same fixed 24,000-character cap and replaces its optional group-engagement
block with the exact empty shape before applying the existing over-cap
disposition.

Every bounded repair or replacement attempt measures its owner-defined dynamic
content before invoking its model. Generic goal cognition retains its existing
24,000-character semantic payload cap. Required-selection cognition measures
its complete system prompt, including dynamic regeneration feedback, together
with its fitted payload under the same 24,000-character cap. A
required-selection regeneration that would cross the cap consumes no
additional model call and fails at the existing pre-state-commit boundary.

Pre-invocation cap exhaustion follows the outcome owned by each stage:

- an irreducible appraisal family is omitted with typed diagnostics;
- an irreducible required-selection producer fails at the pre-state-commit
  boundary with no model call;
- workspace collapse selects its stable first complete bid;
- action planning returns a blocked empty proposal;
- action and resolver authorization deny every candidate;
- text-surface planning returns its validated degraded surface;
- visual-surface planning raises its typed optional-stage failure.

These dispositions consume zero model calls at the over-cap boundary and
authorize no action or resolver side effect. Canonical input validation,
persistent-state validation, reducer and commit invariants, and required owner
failures continue through their typed unrecoverable paths.

## Testing Contract

Run the focused V2 contract, state, emotion-lifecycle, failure, and reflection
settling suites with the project virtual environment. Live LLM cases run one
case at a time with their trace artifact inspected.

## Forbidden Paths

This package does not access adapters, raw database clients, final dialog
wording, platform wire syntax, or untyped relationship scalars.
