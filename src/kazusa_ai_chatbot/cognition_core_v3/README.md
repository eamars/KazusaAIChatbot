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

`current_observation` is the authority for current user action, intent,
acceptance, permission, and current response target. `direct_facts` are stable
background evidence and cannot establish those current facts by themselves.
`CURRENT_OBSERVATION_AUTHORITY_GUIDANCE` is the single exported contract for
this boundary; each A1, A2, G, and ordinary P system instruction and dynamic
packet guidance composes the same constant once so the model-facing layers
cannot drift.
A bounded cognitive or verbal contribution request establishes only the
requested contribution and current interaction role; it does not transfer
scope-external choice, action, or continuing control. The existence or clarity
of the same request, the invitation to contribute, and upstream restatement
are not independent evidence for surrendered agency, inability, dependence,
trust or relationship change, or continuing authority. Those meanings require
a separately expressed `current_observation` fact. Explicit authorization
proves only permission within the object, action, time, and conditions stated
by the current observation; authorization itself does not establish motive or
any broader meaning listed above.
`response_operation.response_content_provider_role` remains deterministic
episode metadata for response validation. Cognition projects only
`role_explicit_content` into model-facing current-event evidence, so procedural
response mechanics cannot become semantic evidence or an orientation field.
`A2_RELATIONSHIP_STATE_EVIDENCE_GUIDANCE` is the single A2 contract for
relationship-state evidence: a current interaction role or scoped permission
does not by itself change relationship axes or establish dependence, trust, or
closeness. Those meanings require a separately expressed relationship fact in
`current_observation`; otherwise A2 keeps stable axes or marks the family
inapplicable while summarizing the interaction factually.
`A2_EXISTENTIAL_DRIVE_EVIDENCE_GUIDANCE` keeps every `existential_drive`
summary, cause, and axis about the character's own experience. User ability,
agency, and need are evidence context rather than the subject of character-
owned axes; unsupported surrender, dependence, or trust is not inferred.
`G_RELATIONAL_CARRIER_EVIDENCE_GUIDANCE` is the single G contract for
relational carriers: `relational_willingness` and `private_monologue` may
express the character's grounded willingness, feeling, motive, and response to
the current contribution, while a current interaction role or scoped
permission is not evidence of the user's dependence, trust, closeness,
relationship state, or motive. Without a separate relationship fact in
`current_observation`, G does not add relationship meaning to the user. Writing
unsupported user relationship meaning as a first-person feeling, inner
judgment, or passive experience still adds a relationship fact about the user;
placing it in `private_monologue` does not provide evidence.
`relational_willingness.reason` records the character's own motive and stance;
`cause_summary` cites current facts; and `private_monologue` expresses the
character's own experience. An ordinary contribution may motivate
characterful help, while user trust, dependence, need, ability, or surrendered
agency requires a separately expressed `current_observation` fact.
`BACKGROUND_CONTEXT_GOAL_AUTHORITY_GUIDANCE` is the corresponding single
contract for G/P: background lanes may shape interpretation and characterful
expression, but they enter `active_character_goal` or `response_goal` only
when `current_observation` makes that meaning part of the current request,
decision, or unresolved matter.

Promoted-memory cognition rows carry deterministic lane-bearing source ids.
Only the typed self-guidance marker may enter
`conditional_character_context`; typed facts enter `direct_facts` as
`character_world_context`, while exact current-user continuity remains
`participant_continuity`. The contract rejects unsupported or contradictory
authority markers before a model stage can consume them.

Each promoted-memory evidence row also carries a hidden, validated metadata
certificate. Its stable id must match
`promoted-memory:<typed-lane>:<stable-id>`, its type/source/authority/status
must match the lane, and learned shared rows must retain the exact seven-field
scope/privacy certificate. Current-user continuity retains `unit_type`,
explicit source metadata, and its canonical user scope; shared rows expose
only the global scope marker to the
prompt-facing projection. Missing, untyped, contradictory, or unmarked rows
are excluded before stage prompts are built.

A1 receives the current observation, direct facts, and causal continuation
pressure for `event_agency`, `goal_threat_outcome`, and
`epistemic_comparison_memory`; conditional character context is absent. A2
receives accepted A1 meaning plus the stage-appropriate participant and
character lanes for `relationship_social`, `moral_identity`, and
`existential_drive`.

When present, `overused_moves` is an observed-response continuity list owned
by the current participant lane. A1 does not receive it. A2, G, and P receive
the exact bounded rows as background evidence only: they are not current facts,
user intent, consent, permission, commitment, prohibition, or a next action.
The current observation, including an explicit user correction, owns the
current semantic delta. A previously observed move may become the current goal
again only when the user continues, deepens, materially changes, or deliberately
reopens that matter. The four-call roster and all existing continuation and
affect projections remain unchanged.

Continuing the underlying task or topic alone does not continue or reopen a
character-authored response move, offer, demand, condition, or relational
payoff. An unanswered character proposal remains participant continuity; it is
not current-user intent, acceptance, commitment, or a required current goal.
Reselection is grounded when the current user responds to, accepts, rejects,
references, asks about, materially changes, or explicitly reopens that move.
Character tendency may shape voice and stance after the current semantic delta
is selected, but it cannot replace that delta as the primary goal.

Public visibility establishes what was said, by whom, and to whom. Consent,
permission, promise, relationship, or role addressed to one participant does
not transfer to another participant merely because both participants saw the
same material.

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
