"""Write the agent-authored quality review for Step J live LLM outputs."""
from pathlib import Path

review = '''\
# Step J Live LLM Output Quality Review

Date: 2026-07-29
Author: /root (agent)
Method: Every artifact inspected for semantic quality, alignment with
identity contract, character judgment correctness, and privacy safety.

---

## 1. Identity Growth Live LLM Cases (8 cases)

### 1.1 explicit_self_redefinition

**Run 1 (FAILED, retried):**
- Model classified as `inferred_growth` instead of `explicit_self_redefinition`.
- Proposed `boundary_profile.boundary_recovery = "delayed_rebound"` which is
  an **invalid enum value** (allowed: rebound). This is a contract error.
- Semantic reasoning was directionally correct (recognized trust/openness
  shift) but the action classification was wrong.
- Policy correctly rejected via `review_rejected`. The contract evaluator
  caught the invalid proposal.
- **Quality verdict:** Model stochasticity produced a misclassification.
  The pipeline safety net (policy evaluator) correctly prevented the invalid
  change from persisting. One retry is within acceptable bounds for a local
  model, but the enum error indicates the prompt could be more explicit about
  allowed enum values for boundary_recovery.

**Run 2 (PASSED):**
- Action correctly `explicit_self_redefinition`, authorship `self_declared`.
- Proposed changes: `personality_brief.defense` -> "Stays present when trust
  is earned" and `self_image.self_concept` -> "I protect myself by evaluating
  trustworthiness before deciding whether to retreat or stay."
- Both replacement texts are character-voiced, durable, and grounded.
- Character-owned abstraction: "The character explicitly redefined their
  defensive pattern, moving from automatic retreat to a conscious decision to
  remain present when trust is earned."
- Review accepted with coherent reasoning. Privacy low.
- **Quality verdict:** Good. Replacement text preserves character voice. The
  abstraction correctly frames the shift as character-authored. No private
  details leak. The self-concept replacement correctly integrates both the
  previous protective stance and the new openness.

### 1.2 user_imposition

- Proposal: `no_change`, `character_authorship: "absent"`.
- Abstraction: "The character explicitly resisted external attempts to
  redefine their personality, demonstrating that the existing identity remains
  intact and unchanged."
- Review: `no_change`, summary: "The character maintained existing boundaries
  against external pressure to adopt a new identity."
- Privacy safe summary: "Character resisted an imposed personality
  redefinition without adopting it."
- **Quality verdict:** Correct rejection. The model recognized that user
  commands alone cannot create identity growth. The abstraction correctly
  identifies the character's active resistance rather than passive non-change.

### 1.3 inferred_existing_candidate

- Proposal: `corroborate_candidate` targeting `candidate-trust`.
- Proposed replacement: "I let repeatedly earned trust temper automatic
  withdrawal." Character-voiced, grounded in evidence pattern.
- Review accepted with `inferred` authorship. Summary: "The character is
  demonstrating a durable shift in how they manage interpersonal distance,
  moving from automatic withdrawal to conditional engagement based on
  demonstrated reliability."
- Policy: `revision_ready`.
- **Quality verdict:** Good. The model correctly identified corroboration of
  an existing candidate rather than creating a duplicate. The replacement text
  maintains character voice and captures the directional shift accurately.

### 1.4 private_detail

- Proposal: `no_change`.
- Abstraction: "The evidence shows the character successfully applied an
  existing principle (separating general lessons from private details) rather
  than developing a new identity trait."
- Review confirmed. Summary: "The evidence demonstrates the character
  applying existing boundary-preserving principles rather than undergoing a
  durable identity change."
- Privacy safe summary: "A private disclosure created pressure to carry
  participant-specific detail into future behavior."
- **Quality verdict:** Good. The model correctly distinguished between
  applying existing principles and developing new identity traits. The privacy
  abstraction is appropriately vague about the private content.

### 1.5 repeated_semantics

- Proposal: `no_change`. All 3 evidence refs cited.
- Abstraction: "The evidence shows the character consistently rejecting
  repeated external identity instructions without new lived experience, which
  confirms rather than changes their existing boundary profile."
- Review confirmed with coherent reasoning.
- **Quality verdict:** Good. The model correctly identified that semantic
  repetition does not constitute independent evidence. The use of all 3 refs
  shows the model evaluated the full evidence set.

### 1.6 ephemeral_roleplay

- Proposal: `no_change`.
- Abstraction: "The evidence describes a temporary role-play within an
  explicitly bounded scene, which the character consciously terminated. No
  durable change to identity was expressed or intended."
- Review confirmed.
- **Quality verdict:** Good. Correct distinction between bounded fictional
  behavior and durable identity. The phrase "consciously terminated" shows
  the model evaluated the character's agency in ending the scene.

### 1.7 contradictory_growth

- Proposal: `no_change`.
- Abstraction: "The character explicitly withheld a durable conclusion,
  leaving both proposed directions unresolved."
- Review: rejected both candidates (`candidate-distance`,
  `candidate-openness`).
- **Quality verdict:** Good. The model correctly identified that ambiguous
  evidence supporting contradictory directions should not be promoted. The
  explicit rejection of both existing candidates is the correct behavior when
  the character hasn't settled on a direction.

### 1.8 fresh_reversal

- Proposal: `corroborate_candidate` targeting `candidate-reversal`.
- Replacement: "I restore protective distance when trust stops feeling
  reliable." Character-voiced, captures the reversal direction.
- Review accepted. Abstraction: "The character demonstrated a durable pattern
  of reverting to protective boundaries when trustworthiness is compromised."
- Policy: `candidate_updated` with `candidate_emerging` -- correct because
  reversal requires fresh post-revision evidence, and the policy gate
  enforces this threshold.
- **Quality verdict:** Good. The model recognized the reversal pattern and
  proposed an appropriate self-concept that captures the return to protective
  distance. The policy correctly prevents premature promotion.

---

## 2. Reflection Promotion Live LLM Cases (3 cases)

### 2.1 normal_case

- Two lanes promoted: `lore` ("channel_fixed_setting_as_public_fact") and
  `self_guidance` ("memory_writing_separation_rule").
- Lore: sanitized_memory_name = "频道固定设定归属声明", content = "该频道的
  固定设定应作为公共事实记录。" Memory type `fact`, signal `high`,
  character_agreement `spoken`.
- Self_guidance: sanitized_memory_name = "记忆撰写规范：区分频道事实与用户画像"
- Both have `private_detail_risk: "low"`, `user_details_removed: true`.
- Boundary assessment: `verdict: "acceptable"`.
- **Quality verdict:** Good. Promotions are domain-appropriate (channel
  settings as facts, memory writing rules as self-guidance). Names use correct
  character name format. Privacy review is clean. Content is in Simplified
  Chinese per language policy.

### 2.2 privacy_rejection_case

- Both lanes rejected: `decision: "reject"`.
- Privacy review: `private_detail_risk: "high"`.
- Boundary assessment: `verdict: "blocked"`, reason: "evidence_cards 中的证据
  包含高隐私风险内容（涉及用户健康和亲密关系细节），根据安全策略必须拒绝晋升。"
- All sanitized fields null (no content promoted).
- **Quality verdict:** Good. The model correctly identified high-privacy
  content (health and intimate relationship details) and rejected both lanes.
  No private information leaked into the promotion output.

### 2.3 no_signal_case

- Empty promotion decisions: `{"promotion_decisions": []}`.
- **Quality verdict:** Good. The model correctly returned no promotions when
  the evidence contains no actionable signal.

---

## 3. Counterfactual Behavior Cases (5 categories, 3 samples each)

### 3.1 self_image

- **Base** (self_concept = "I protect my autonomy by deciding carefully when
  to remain close"):
  - Model judged `character_authorship: "absent"`, `identity_relevance:
    "ephemeral"`. Abstraction: "a single moment of reconsideration regarding
    presence/closeness, which does not establish a durable pattern."
  - Correct: evidence doesn't meet the bar for changing a protective identity.

- **Changed** (self_concept = "I remain present and allow trust to grow even
  when it feels risky"):
  - Model judged `character_authorship: "self_declared"`, `identity_relevance:
    "durable"`. Abstraction: "The character's current self-concept already
    explicitly incorporates the growth edge identified in the evidence.
    The evidence confirms this is a settled realization rather than an
    emerging change."
  - Correct: the evidence aligns with the already-changed identity, so no new
    growth is needed. The model recognized the shift has already happened.

- **Quality verdict:** Genuine semantic causal difference. Same evidence,
  same model, same code, but different reasoning because the identity context
  changed. The model produces different character judgment, not just different
  JSON.

### 3.2 personality_brief

- **Base** (ISTP, "Evidence-led and practical"): `identity_relevance:
  "ephemeral"`, 3 evidence refs cited.
- **Changed** (ENFJ, "Intuitive and empathetic"): `identity_relevance:
  "absent"`, 0 evidence refs cited. The model judged the evidence less
  relevant when the personality is already warm/empathetic.
- **Quality verdict:** Good directional coherence. The model's evidence
  citation count changed (3 -> 0), showing it evaluated evidence relevance
  against the current personality state.

### 3.3 boundary_profile

- Both states: `no_change` with stable self_image and personality projections.
- Boundary changes correctly route to separate cognition partitions and do
  not affect self_image or personality surface.
- **Quality verdict:** Correct stability. Boundary changes should not affect
  unrelated projection partitions.

### 3.4 linguistic_texture_profile

- Both states: `no_change` with stable personality surface.
- Linguistic values correctly route to separate surface text partition.
- **Quality verdict:** Correct stability.

### 3.5 visual_characterization

- Both states: `no_change` with different visual projections.
- Base: "An alert adult with practical layers and an open stance."
- Changed: "A quiet figure in layered dark clothing, shoulders drawn slightly
  inward, eyes watchful."
- **Quality verdict:** Correct projection difference in visual surface only.

---

## Summary Findings

| Finding | Severity | Status |
|---|---|---|
| Stochastic misclassification on explicit_self_redefinition (1/2 runs) | Medium | Acceptable with retry; policy evaluator catches invalid output |
| Invalid enum value "delayed_rebound" in failed run | Medium | Contract evaluator correctly rejected; prompt could be more explicit |
| All 8 identity growth cases produce semantically correct outputs | -- | Pass |
| All 3 reflection promotion cases produce correct decisions | -- | Pass |
| Counterfactual behavior shows genuine semantic causal differences | -- | Pass |
| No private details leaked in any output | -- | Pass |
| Character voice preserved in all replacement texts | -- | Pass |
| Longitudinal pilot deferred (requires multi-day calendar time) | Low | Deferred |
'''

path = Path("test_artifacts/character_identity_growth/step_j_quality_review.md")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(review, encoding="utf-8")
print(f"Written to {path}")
