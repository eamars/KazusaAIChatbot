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
`run_text_surface_planning(...)`, and `run_visual_surface_planning(...)`.
Cognition runs deterministic preparation, scoped semantic appraisal,
dependency-ready goal branches, complete-bid collapse, route validation, and
one replacement-state update. The caller commits that update before action,
surface, resolver, or dialog work.

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
receives at most one LLM-owned schema repair while deterministic validation
remains strict. A still-failed required branch raises an execution error rather
than becoming an empty workspace and character silence. Initial and repaired
goal outputs are eligible for the protected turn trace.

When upstream episode evidence carries a typed required selection, a focused
goal-level check verifies that the owning role makes or explicitly expresses
that choice before action planning. A rejected bid is regenerated from typed
operations, current evidence, affect, relationship, character constraints,
and scene context without the rejected bid or private continuity prose. The
replacement is rechecked by the same owner. Ordinary turns add no selection
check.

Action planning treats local-model output as a bounded proposal rather than an
execution precondition. It canonicalizes the known envelope, keeps usable
rows, drops invalid rows individually, ignores unknown fields, and caps each
request list at three. Mutually exclusive action and resolver requests remain
a semantic contract error. If one complete replacement is still unusable, the
turn continues with an empty action plan; if action authorization cannot
produce a usable replacement, every candidate is denied. Neither containment
path authorizes work, changes the visible speech route, or reduces the
registry-driven three-request capacity.

The shared surface input receives semantic intention, bounded affect and
relationship projections, complete-bid projections, permitted action results,
interaction style, and bounded character voice. The three-call text planner
exposes raw character voice only to speech-style planning; content and
preference planning cannot observe it. Speech-style output is limited to
lexical register and wording, sentence length and shape, rhythm, hesitation,
and punctuation; it never proposes details, topics, examples, images, actions,
claims, inferences, or content beats. The exact text output contains no raw
voice or visual directives. The independent one-call visual planner may
observe raw voice and produces image-generation directives as terminal private
evidence; it has no downstream image or dialog model. Raw episode traces
retain those directives for audit, while every model-facing consolidation
projection excludes their fragments.

Content planning expresses the selected character judgment using the current
scene, affect, relationship, and interaction style. Coherent imaginative
detail is allowed when it remains compatible with current input, active
constraints, and actor/target/subject roles. Preference planning emits only
real visible boundaries and addressee constraints, so both lists may be empty.
Dialog owns natural character-specific chat-ready wording. Two focused
hard-error checks run in parallel on the existing dialog-model route within
the shared 24,000-character surface-prompt bound. Semantic fidelity receives
current model-visible percept rows, including any upstream response/selection
ownership, the candidate role frame, and candidate dialog; it rejects internal
contradiction, direct current-input conflict, and role reversal. Surface
integrity receives permitted action results and candidate dialog; it rejects
only false system, tool, platform, or other character-brain execution claims.
Text planning owns expressed meaning and interaction progress without
supplying staging forms. Dialog expresses emotion, character, and interaction
posture through sendable wording and cadence. Action narration is outside the
fatal taxonomy and remains unchanged when the model produces it; the runtime
prompts neither request it nor create a rejection or repair rule for it.
Generated content, addressee, intent, and style proposals stay outside
hard-error authority. Source percepts and generated character speech carry
separate typed pronoun frames before role direction is compared. Novelty and
coherent drift are not failures by themselves. Deterministic code merges only
the verdict shapes, bounding each owner to four issues and the merged result to
eight, and a negative result may trigger one grounded repair. The
protected turn trace records both checks, the merged verdict, and repair output
as distinct evidence.

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
- `run_visual_surface_planning(...)`
- `validate_cognition_input(...)`
- `validate_cognition_core_output(...)`

## Runtime Flow

Input validation, bounded semantic appraisal, dependency-ready goal branches,
complete-bid collapse, route validation, replacement-state reduction, and
typed output validation run in one inspectable call.

## Failure Behavior

Malformed input, invalid state, unsupported routes, unresolved required
dependencies, and invalid required cognition output raise the package
validation or execution error. Optional action proposal/authorization schema
failure is contained as empty or denied work so a valid speech response can
continue. Callers commit only validated replacement state.

## Testing Contract

Run the focused V2 contract, state, emotion-lifecycle, failure, and reflection
settling suites with the project virtual environment. Live LLM cases run one
case at a time with their trace artifact inspected.

## Forbidden Paths

This package does not access adapters, raw database clients, final dialog
wording, platform wire syntax, or untyped relationship scalars.
