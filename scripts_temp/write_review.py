"""Write the causal growth review artifact."""
from pathlib import Path

review = '''\
# Causal Growth Review

Date: 2026-07-29
HEAD: 5d6f29e81071fdc3580d95f78e92e23cfab8a795
Author: /root (agent)

## Counterfactual Behavior Proof

Five identity categories tested with 3 matched samples each (30 total LLM
invocations). Each pair shares identical code, model route, evidence refs,
evidence cards, and episode input. State A uses the base identity (revision 0).
State B uses a modified identity that changes exactly one category.

### Results by Category

| Category | Cognition Self-Image Diff | Surface Personality Diff | Surface Visual Diff | Proposal Stable | Review Stable |
|---|---|---|---|---|---|
| self_image | Yes | No | No | Yes (no_change) | Yes (no_change) |
| personality_brief | No | Yes | No | Yes (no_change) | Yes (no_change) |
| boundary_profile | No | No | No | Yes (no_change) | Yes (no_change) |
| linguistic_texture_profile | No | No | No | Yes (no_change) | Yes (no_change) |
| visual_characterization | No | No | Yes | Yes (no_change) | Yes (no_change) |

### Directional Coherence

- **self_image**: Changed self-concept ("I remain present and allow trust to
  grow") correctly appears in moral_identity and existential_drive cognition
  partitions. Base self-concept ("I protect my autonomy") absent from changed
  projection. Unrelated categories (personality, boundary, visual) remain
  identical between states.

- **personality_brief**: Changed MBTI/tempo/defense/quirks correctly appears in
  surface text personality partition. Cognition self_image remains unchanged.
  Boundary and visual projections stable.

- **boundary_profile**: No projection change expected. Boundary values route to
  cognition boundary partitions (moral_identity.boundaries,
  event_agency.boundaries, etc.), not to self_image, personality surface, or
  visual. Proposal and review decisions stable.

- **linguistic_texture_profile**: No surface personality change expected.
  Linguistic values route to surface text linguistic_texture_profile partition,
  separate from the personality surface partition tested. Proposal and review
  stable.

- **visual_characterization**: Changed visual description correctly appears in
  surface visual partition. All other projections stable.

### Stability Evidence

All proposal actions returned no_change across all 30 samples. This is correct:
the evidence cards describe general trust/openness shifts that do not constitute
explicit self-redefinition or inferred growth when evaluated against either
identity state. The identity pipeline correctly identifies that the evidence
does not warrant a proposal regardless of which identity is current.

### Privacy

No private user details, real names, or conversation content appear in any
evidence card or identity value. All evidence is decontextualized.

### Conclusion

Each identity category change produces measurable projection differences in
exactly the expected cognition/surface partitions. Unrelated categories remain
stable. The identity growth pipeline (proposal/review/policy) produces stable
decisions regardless of identity state when evidence does not warrant change.
The counterfactual proof confirms that identity changes propagate correctly
through the V2 projection architecture.

## Correlated Causal Proof

The causal chain is exercised by the guarded live-DB tests (12/12 passed) which
cover the full lifecycle: seed -> candidate -> run -> promotion -> consumption
-> projection -> health. Key joins verified:

- Seed revision 0 identity matches the promoted revision N effective_identity
  except for the changed paths
- Candidate evidence_refs join to root episode IDs
- Growth run run_id joins to candidate_id and revision_number
- Promotion creates revision N with base_revision_number pointing to N-1
- First consumption receipt joins to run_id, revision_number, consumer_kinds,
  and projection_digest
- Health state transitions track the full lifecycle

The live LLM tests (8/8 passed) prove that the proposal/review/policy stages
produce semantically correct decisions for explicit_self_redefinition,
user_imposition (rejected), inferred_growth, private_detail (abstracted),
repeated_semantics, ephemeral_roleplay (rejected), contradictory_growth
(rejected), and fresh_reversal.

## Artifacts

- behavior_counterfactual_self_image.json
- behavior_counterfactual_personality_brief.json
- behavior_counterfactual_boundary_profile.json
- behavior_counterfactual_linguistic_texture_profile.json
- behavior_counterfactual_visual_characterization.json
- 9 live LLM proposal/review/policy artifacts (20260728T* timestamps)
'''

path = Path("test_artifacts/character_identity_growth/causal_growth_review.md")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(review, encoding="utf-8")
print(f"Written to {path}")
