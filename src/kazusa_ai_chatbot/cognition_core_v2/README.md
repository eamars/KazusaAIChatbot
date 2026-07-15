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

The twenty-one emotion formulas are exercised from typed natural causes in
`tests/test_cognition_core_v2_emotion_lifecycle.py`. Cross-scope character
constraints and optional relationship context are passed as dedicated
projections; they are not merged into mutable state. Character sleep recovery
is deterministic and separate from user elapsed decay.

Database-backed callers use `db.users` for user-owned state and `db.character`
for the character singleton. The test database harness requires the exact
`_test_kazusa_live_llm` name, validates seeded V2 state, and gives every
mutable test row a unique owner.

The public Stage 2 surface is the pair `run_cognition(...)` and
`run_text_surface_planning(...)`. Cognition runs deterministic preparation,
scoped semantic appraisal, dependency-ready goal branches, complete-bid
collapse, route validation, and one replacement-state update. The caller
commits that update before action, surface, resolver, or dialog work.

The text-surface API receives only semantic intention, bounded affect and
relationship projections, complete-bid projections, and permitted action
results. It owns expression planning; dialog owns final wording.

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
