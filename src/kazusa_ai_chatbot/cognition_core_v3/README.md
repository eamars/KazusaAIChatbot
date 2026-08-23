# Cognition Core V3

`cognition_core_v3` is the canonical single-pass semantic cognition boundary.
It owns one caller-selected sequence of four model calls: A1 appraisal, A2
appraisal, G active-character goal selection, and either ordinary or
self-cognition P planning. The chain uses one primary lane, deterministic
JSON parsing, and typed fail-closed validation. It has no bid roster,
workspace partition, sibling recovery, semantic retry, or unavailable-goal
fallback.

## Stage contract

The dynamic packets use five explicit authority lanes:

- `current_observation`: the current episode and caller-owned participant
  roles;
- `direct_facts`: source-owned evidence that may support factual assertions;
- `participant_continuity`: prior actors, actions, and outcomes only, never a
  new action, consent, commitment, permission, or current intent;
- `conditional_character_context`: identity, relationship, affect, and
  reflection tendencies that may shape judgment and motivation but cannot
  establish facts or permissions; and
- `continuation_state`: active causes and genuinely unresolved cross-turn
  goals after stale ordinary-response cutover.

A1 receives the current observation, direct facts, and causal continuation
pressure for `event_agency`, `goal_threat_outcome`, and
`epistemic_comparison_memory`; conditional character context is absent. A2
receives accepted A1 meaning plus the stage-appropriate participant and
character lanes for `relationship_social`, `moral_identity`, and
`existential_drive`.

G returns exactly one meaningful active-character goal, relational
willingness, and a bounded first-person `private_monologue` connecting current
feeling, concrete cause, and immediate motivation. P receives that result and
only supplied semantic capabilities, and returns an `epistemic_boundary` that
states what visible wording may assert, interpret, or leave unknown. Ordinary
response and self-cognition plans remain disjoint.

Model packets use semantic descriptors and contain no storage identifiers,
handles, evidence IDs, target paths, or runtime metadata. The caller binds
accepted qualitative axis shifts to native state roots after generation.

## State and affect ownership

All 51 registered appraisal axes remain available. The caller-owned binder
uses guarded native reducers for relationship, causal entities, goals, drives,
and meaning. Each accepted axis receives an applied, clamped,
scope-inapplicable, no-numeric-change, or explicitly deferred-capacity receipt,
and the complete replacement state is validated before it is returned. Safe
terminal rows are reclaimed before a new causal root is admitted; protected
capacity leaves the prior valid collection intact.

Emotion derivation runs from that replacement state. Affect projections retain
emotion identity, intensity and trend, while cause provenance carries
structured `{scope, kind, entity_id}` roots, `root_refs`, and `cause_status`.
Cause summaries remain concrete semantic descriptions supplied by the owning
appraisal or native causal entity.

## Public API and diagnostics

`run_cognition(input_payload, services)` is the canonical entrypoint.
`CognitionChainServicesV3` contains one LLM invoker, one chain configuration,
and the bounded turn deadline. Invocation-local protected trace capture records
first-pass stage messages, parsed products, status, and timing for inspection.
When protected trace capture is enabled, every A1, A2, G, and P attempt also
uses the shared persisted trace lane; `full` mode retains its raw messages,
response, and parsed product. Public cognition output exposes semantic result
and state projection only.

The immediate node and surface consumers use the canonical V3 output directly:
`active_character_goal`, `private_monologue`, `response_plan` including its
`epistemic_boundary`, affect, relationship projection, and structured cause
provenance. Persisted and public protocol identifiers
ending in `.v2` or `V2` remain versioned data contracts; they do not select an
alternate cognition engine.

## Testing

The focused deterministic contract tests use a scripted invoker to prove one
call each for A1, A2, G, and P; text-mode configuration; stage-local packet
allowlists; absence of private key vocabulary; one persisted active goal;
replacement-state validation; structured cause roots; and complete 51-axis
binding receipts across user and character scopes.

The mirrored source-owner test tree is `tests/unit/cognition_core_v3`. Exact
source-to-node ownership is registered in
`tests/ownership/source_test_impact_manifest.json`.
