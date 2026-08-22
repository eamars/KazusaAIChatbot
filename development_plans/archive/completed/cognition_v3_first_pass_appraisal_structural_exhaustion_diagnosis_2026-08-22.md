# Cognition First-Pass Appraisal Structural Exhaustion Diagnosis

## Document control

- **Status:** completed on 2026-08-22; implementation remains a separately
  authorized work item.
- **Class:** cognition-quality diagnosis after the agentic-loop cutover.
- **Owner:** root engineering agent; the existing Luna worker remains the
  reusable implementation and verification worker if a fix is authorized.

## Observed failure

The production control-console turn completed and returned a coherent visible
reply, but its cognition graph was `partial` and the chain terminal disposition
was `accepted-degraded`. The latest run reported structural exhaustion for the
`relationship_social` appraisal family. An earlier run reported the same class
for `event_agency`, `goal_threat_outcome`, and
`epistemic_comparison_memory`.

This is not classified as good cognition quality. It passed the relaxed Gate 7
semantic admission rule only because no role reversal, self-conflict, or
boundary/safety conflict appeared in the visible result.

## Diagnostic boundary

1. Retrieve the protected trace and exact first-attempt appraisal input/output
   for the failed family.
2. Compare the actual producer response with the canonical appraisal contract,
   transcript state, and stage input projection.
3. Determine whether the primary cause is ambiguous/redundant input,
   contradictory ownership instructions, schema presentation, context
   ordering, model-route mismatch, or another first-pass design defect.
4. Preserve cognition stages, relationship and emotion axes, and emotion cause
   semantics.
5. Prefer a clearer canonical input/output flow that enables a correct first
   response. Do not treat validator relaxation, JSON repair, regeneration, or
   prompt-example overfitting as the solution.
6. Use one narrow live case to verify the diagnosis and any subsequently
   authorized fix. Existing passing cases are not rerun.

## Completed diagnosis

The evidence-backed review is recorded at
`test_artifacts/diagnostics/cognition_v3_first_pass_appraisal_structural_exhaustion_review_2026-08-22.md`.

Two confirmed production traces and the retained 72-trial full-capture cohort
show a systemic first-pass design defect:

- every ordinary evidence source is fanned out to nearly every appraisal
  family;
- each evidence row can create event, threat, and knowledge-gap candidates;
- resolved and stale matters spill into the current observation workspace;
- proposition meanings and handle domains conflict;
- raw accepted JSON remains in the cross-stage transcript;
- structural validation can accept a boundary-conflicted semantic judgment.

The second trace, `llmtrace_0bae517c46d24c519181ddf185453146`,
is a hard semantic failure under the owner's relaxed criterion. Its visible
answer confuses remembered coercion and loss of agency with current desire,
while the input contains unresolved injury and active boundary pressure.

The correction is an explicit stage-local turn workspace with producer-owned
evidence routing, a current-matter working set, separate participant/matter
namespaces, compact accepted products between stages, and concrete affect
causes. All cognition capacities, axes, cause links, reducers, permissions,
and planning owners remain in scope.

No production code, tests, database state, or prompts were changed during this
diagnosis. The exact failure replay is reserved for one narrow post-fix live
verification after implementation is explicitly authorized.

## Diagnosis acceptance

- The exact first-attempt structural mismatch is identified from protected
  trace evidence.
- The failure is mapped to its owning input, prompt/contract, model route, and
  stage boundary.
- The proposed correction explains why first-pass compliance should improve
  generally without case-specific wording or weaker validation.
- Any production implementation remains a separate explicitly authorized step
  after the diagnosis is reported.
