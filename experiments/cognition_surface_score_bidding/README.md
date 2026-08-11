# Cognition surface score-bidding experiment

This directory contains protected, owner-local evidence for the two surface
quality experiments in
`development_plans/active/bugfix/cognition_surface_score_ranking_followup_plan_20260810.md`.

## Owners

- `surface_content_plan`
- `surface_dialog_compliance_repair`

The experiment keeps the Cognition V2 `confidence` field as a bounded semantic
descriptor. It is advisory context and never participates in ranking or
thresholding. The independent evaluator owns the only quality-ranking field:
`score`, a finite float in `[0.0, 1.0]`.

## Evidence protocol

1. Capture trace-backed contexts without changing production behavior.
2. Keep producer attempts and evaluator calls in separate protected records.
3. Retain only structurally valid and hard-eligible candidates.
4. Collect at least 30 contexts per owner, with at least 20 calibration and 10
   disjoint held-out contexts. Each context needs two semantically
   distinguishable hard-eligible candidates.
5. Apply two independent human labels to each comparison, recording an
   adjudication when labels disagree.
6. Calibrate each owner separately. A threshold is usable only when held-out
   ordering accuracy is at least 80 percent and no hard-integrity false accept
   is present.
7. Run the two live cases one at a time and inspect raw evaluator and selected
   candidate artifacts before accepting a cutover.

## Artifact schema

`content_plan_candidates.jsonl` and `compliance_repair_candidates.jsonl` use
one bounded JSON object per context. Raw candidate text and evaluator prose
remain in protected diagnostic storage; these files contain references,
digests, typed dispositions, and bounded summaries only.

`thresholds.json` records independent owner thresholds and calibration status.
The production plan remains incomplete while either owner is marked pending or
blocked. Placeholder values must never be treated as accepted calibration.

`calibration_report.md` records corpus counts, label agreement, ordering,
threshold, failure, call-count, and latency evidence. Missing evidence is
recorded explicitly rather than represented by synthetic candidates.
