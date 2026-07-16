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
branches only. Branch bids distinguish analytic `reason` from first-person
`private_monologue`; public output exposes that distinction as
`selected_bid_reason` and `private_monologue`.

Goal-bid output uses an exact route-to-capability-field matrix. A malformed bid
receives at most one LLM-owned schema repair while deterministic validation
remains strict. A still-failed required branch raises an execution error rather
than becoming an empty workspace and character silence. Initial and repaired
goal outputs are eligible for the protected turn trace.

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

Content planning and dialog preserve supplied descriptors, attributes,
qualifiers, quantities, polarity, and comparative degree. Non-conflicting
elaboration may add context without transforming or compounding an attribute
into a different claim. They preserve explicit entity and target specificity;
elaboration cannot generalize, euphemize, narrow, broaden, or replace a
supplied referent. Acceptance, refusal, permission, and consent remain bounded
to the exact source-requested act and scope; indefinite or unrestricted
permission cannot substitute for a specific permission. When source meaning
covers only the current occurrence, the planned and rendered output remains
silent about future claims, promises, conditions, expectations, threats,
habits, rules, and contrastive or teasing additions. Explicit future content
is preserved when the source supplies or requires it. Dialog owns literal
spoken or typed wording. Its existing semantic verifier receives only the
exact text-surface output, candidate dialog, and current model-visible percept
rows within the shared 24,000-character surface-prompt bound. A negative
verdict may trigger one grounded repair. The protected turn trace records the
initial dialog, verifier verdict, and repair output as distinct stages.

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
dependencies, and invalid model output raise the package validation or
execution error. Callers commit only validated replacement state.

## Testing Contract

Run the focused V2 contract, state, emotion-lifecycle, failure, and reflection
settling suites with the project virtual environment. Live LLM cases run one
case at a time with their trace artifact inspected.

## Forbidden Paths

This package does not access adapters, raw database clients, final dialog
wording, platform wire syntax, or untyped relationship scalars.
