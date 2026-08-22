# Cognition Core V3

`cognition_core_v3` is the canonical single-pass semantic cognition boundary.
It owns one caller-selected sequence of four model calls: A1 appraisal, A2
appraisal, G active-character goal selection, and either ordinary or
self-cognition P planning. The chain uses one primary lane, deterministic
JSON parsing, and typed fail-closed validation. It has no bid roster,
workspace partition, sibling recovery, semantic retry, or unavailable-goal
fallback.

## Stage contract

- A1 receives the current observation, typed evidence, and the three
  world-facing families: `event_agency`, `goal_threat_outcome`, and
  `epistemic_comparison_memory`.
- A2 receives compact accepted A1 meaning and causes plus character identity,
  standards, boundaries, relationship context, and active affect causes. It
  owns `relationship_social`, `moral_identity`, and `existential_drive`.
- G receives compact A1/A2 meaning and continuing state, and returns exactly
  one meaningful active-character goal plus relational willingness.
- P receives that goal and only the supplied semantic action/resolver
  capabilities. Ordinary response and self-cognition plans are disjoint.

Model packets use semantic descriptors and contain no storage identifiers,
handles, evidence IDs, target paths, or runtime metadata. The caller binds
accepted qualitative axis shifts to native state roots after generation.

## State and affect ownership

All 51 registered appraisal axes remain available. The caller-owned binder
uses guarded native reducers for relationship, causal entities, goals, drives,
and meaning. Each accepted axis receives an applied, scope-inapplicable,
no-root, or no-numeric-change receipt, and the complete replacement state is
validated before it is returned.

Emotion derivation runs from that replacement state. Affect projections retain
emotion identity, intensity and trend, while cause provenance carries
structured `{scope, kind, entity_id}` roots, `root_refs`, and `cause_status`.
Cause summaries remain concrete semantic descriptions supplied by the owning
appraisal or native causal entity.

## Public API and diagnostics

`run_cognition(input_payload, services)` is the canonical entrypoint.
`CognitionChainServicesV3` contains one LLM invoker, one chain configuration,
  and the bounded turn deadline. Invocation-local protected trace capture is
  opt-in diagnostic evidence: when enabled it records first-pass stage
  messages, parsed products, status, and timing for inspection, but production
  protected trace does not independently persist or export those records;
  public cognition output exposes semantic result and state projection only.

The immediate node and surface consumers use the canonical V3 output directly:
`active_character_goal`, `response_plan`, affect, relationship projection,
and structured cause provenance. Persisted and public protocol identifiers
ending in `.v2` or `V2` remain versioned data contracts; they do not select an
alternate cognition engine.

## Testing

The focused deterministic contract tests use a scripted invoker to prove one
call each for A1, A2, G, and P; text-mode configuration; stage-local packet
allowlists; absence of private key vocabulary; one persisted active goal;
replacement-state validation; structured cause roots; and complete 51-axis
binding receipts across user and character scopes.
