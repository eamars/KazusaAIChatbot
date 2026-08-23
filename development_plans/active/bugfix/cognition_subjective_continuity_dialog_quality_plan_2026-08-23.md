# Cognition Subjective Continuity and Dialog Quality Plan

## Summary

- **Goal:** Restore character-owned emotional subjectivity and explicit
  evidence authority to current-turn cognition, then remove the non-semantic
  surface branch that can discard valid content.
- **Status:** `draft`; full-trace evidence and architecture proposal are
  complete. Production semantic changes await owner approval.
- **Scope:** Cognition V3 A1/A2/G/P input and output contracts, canonical output
  projection, cognition-to-L3 surface input, deterministic addressee/boundary
  projection, post-turn monologue residue input, generic implementation-
  agnostic graph projection, stale pre-cutover response-goal disposition, and
  narrow real-LLM verification.
- **Out of scope:** Raw monologue in the final dialog payload, V2
  compatibility, parallel bids, output wording optimization, adapters, and
  unrelated services.

## Evidence

The full RCA is:

`test_artifacts/diagnostics/cognition_dialog_quality_regression_rca_2026-08-23.md`

The complete current trace is:

`test_artifacts/diagnostics/llm_trace_llmtrace_23531e49a2994b74b4fbf50f0475f3de_full_20260823.json`

It proves three contract failures:

1. Current projection aliases `active_character_goal.reason` into
   `internal_monologue`, although historical architecture explicitly requires:

   ```text
   analytic reason != first-person private monologue
   ```

2. A1 receives promoted behavioral reflections alongside current facts under
   a system instruction that does not define source precedence or allowed
   semantic effects. An “exchange condition” tendency becomes the appraisal of
   an unknown glass compass. Participant continuity later becomes a false
   claim that the user accepted payment terms.
3. `surface.content_plan` succeeds, while `surface.preference` invents the same
   invalid addressee row on all three attempts. The join discards the valid
   content plan, produces a degraded surface, and still reports `completed`.

Historical V2 traces demonstrate that first-person subjectivity and explicit
evidence authority produced more coherent character judgment. The V2 bid,
handle, confidence, and consequence machinery remains unnecessary. Raw
monologue has correctly remained absent from the final dialog payload since
the stale-authority incident.

## Proposed Canonical Flow

```text
A1: current observation + factual/contextual evidence only
  -> A2: accepted A1 + identity/boundary/relationship context
       + conditional character context with explicit allowed effects
  -> G: active goal + relational willingness + private_monologue
  -> P: response goal + goal resolution + epistemic_boundary + capabilities
  -> deterministic state/affect binding
  -> deterministic addressee/visible-boundary projection
  -> one text L3 call: goal/plan + affect/cause + subjective context
  -> final dialog: validated L3 output only
  -> post-turn residue: original private monologue + visible outcome
```

### Authority lanes

Replace the undifferentiated evidence list at each stage with the smallest
stage-owned lanes:

- `current_observation`: current episode and caller-owned participant roles;
- `direct_facts`: source-owned factual evidence usable for assertions;
- `participant_continuity`: prior actor/action/outcome only, never evidence of
  a new user action, consent, or commitment;
- `conditional_character_context`: identity/reflection tendencies that may
  shape judgment when applicable but cannot establish facts, relationship
  permission, capability, or current-user intent;
- `continuation_state`: only genuinely unresolved cross-turn goals and active
  causes, not prior answerable response goals.

A1 receives the first two lanes and causal state pressures. Persona habits and
expression strategies do not belong in world-facing appraisal. A2/G receive
the character and relationship lanes with their allowed effects stated in the
model-facing contract. This is an interface correction, not keyword routing or
case-specific prompt tuning.

### G addition

Add one top-level `private_monologue` string to the existing single G call. It
must be concise, character-first-person, and connect:

- what the character feels now;
- the concrete current cause;
- what she immediately wants to protect, reveal, avoid, or pursue.

It cannot establish facts, permissions, capabilities, targets, or state
changes. Validation is structural and bounded only.

### P addition

Add one bounded `epistemic_boundary` string to the existing single P call. It
states what the visible response may assert, what may only be framed as an
interpretation, and what remains unknown. It remains an LLM semantic decision;
deterministic code validates only shape and size.

### Remove the preference model stage

The current preference stage has no semantic choice to make:

- `visible_boundaries` is contractually always empty;
- `addressee_plan` may only reproduce already supplied typed rows.

Project both deterministically from caller-owned input and remove the model
call, its repair attempts, and the all-or-degrade join. The content-plan result
must reach dialog when it succeeds. This reduces text L3 to one semantic call
and eliminates the reproduced failure rather than teaching a model to copy an
enum more reliably.

### L3 subjective projection

Project `private_monologue` and `epistemic_boundary` in a typed
`subjective_expression_context`. L3 may translate the monologue into emotional
delivery, while the goal, willingness, visible episode, epistemic boundary,
and capability results remain authoritative. Final dialog continues to receive
only L3 output.

### Stale cutover state

Current state still contains pre-cutover pursuing `ordinary_response` goals
that describe prior answerable replies. Current V3 already keeps
`answerable_now` goals turn-local. Dispose those stale rows once at the
canonical cutover boundary and ensure prompt projection includes only explicit
continuation goals. Preserve active emotion causes and other non-response
state; do not reset the character or relationship.

### Residue

Record the original current-turn first-person monologue after the visible
outcome. The residue stage decides append, replace, or clear; it does not
synthesize first-person ownership from analytic `reason` after dialog.

## Legacy Disposition

Keep all appraisal families, emotion axes and causes, relationship state,
identity/boundary judgment, one goal, one response plan, and typed action/
resolver ownership.

Leave retired all bid handles, multiple branches, collapse selection,
confidence prose, expected-consequence lists, the preference model call,
semantic repair loops, raw dialog history authority, and V2 contracts/tests.

## Verification

Run one real case at a time:

- unknown glass compass;
- affectionate kiss;
- casual greeting/chat invitation;
- relationship boundary.

Accept natural wording variation. Require:

- no role reversal, material self-conflict, or boundary/safety conflict;
- concise first-person causal subjectivity in G;
- preserved emotion rows and concrete causes;
- conditional reflection never establishes a current fact, user commitment,
  consent, or relationship permission;
- an explicit P epistemic boundary for unknown or interpretive content;
- one successful semantic text-L3 call whose content reaches dialog;
- no preference-model or preference-repair attempt;
- final dialog stays inside the L3 semantic plan.

The design retains exactly four cognition provider calls, reduces text L3 from
two provider calls to one, and adds no semantic validator, regeneration,
compatibility bridge, or repair loop.

## Decision Required

The evidence gate is satisfied. Owner approval is required before the semantic
cutover. Full protected trace capture is complete and tracked in its archived
bugfix plan.
