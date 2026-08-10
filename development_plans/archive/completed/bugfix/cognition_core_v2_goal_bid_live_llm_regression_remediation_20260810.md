# Cognition Core V2 goal-bid live-LLM regression remediation

## Summary

- Goal: classify the original Cognition Core V2 goal/action live-LLM failures,
  repair legitimate product defects and stale test contracts, and reach at
  least a 95% pass rate without bypassing one-case live execution.
- Status: completed.
- Scope: the 71 non-DB goal-bid nodes, frozen evidence inputs, goal/action/
  resolver prompts, runtime-owner projection, and adjacent deterministic
  contracts. Four live-DB nodes remain a separately gated infrastructure
  cohort.
- Semantic boundary: character-owned stance remains an LLM decision.
  `autonomy_boundary` remains character context. Runtime truth, capability
  ownership, evidence provenance, structural validation, and approval
  lifecycle remain operational contracts.

## Baseline and final evidence

- Original baseline: 46/71 passed (64.8%).
- Final primary cohort: 71/71 passed (100%).
- Final focused deterministic suite: 82/82 passed.
- Detailed evidence: `test_artifacts/cognition_core_v2/goal_bid_live_llm_regression.md`.

## Final failure classification

| Failure family | Final category | Disposition |
|---|---|---|
| Ten missing replay/trace inputs | Test/infrastructure failure | Restored exact sibling artifacts and recorded SHA-256 provenance. |
| Ten stale `background_work_request`/`public_group_scene` cases | Test/fixture failure | Migrated tests to current registry and explicit scene-field contracts. |
| C03 direct connector omission of `public_group_scene` | Test/fixture failure | Added the explicit empty scene field to the direct private/group fixtures. |
| C20 reminder feasibility | Legitimate product semantic failure | Added capability-neutral reminder language; feasibility and scheduling remain downstream. |
| Approval preparation | Legitimate product semantic failure | Extended resolver authorization for bounded approval preparation without a general safety gate. |
| Three independent private actions | Test/contract expectation failure | Aligned the assertion with speech plus validated private actions and current action ownership. |
| Old `taboo_lover` and `coercion_lover` hard stance oracles | Test/architecture-contract failure | Corrected typed identity/current-pressure evidence and asserted evidence fidelity plus structural validity; stance remains model-selected. |
| C07/C08/C11 runtime-owner outcomes | Product/runtime contract issue | C07 is inline task resolution with `start_in_background=false`; C08/C11 are exact `blocked` results with no action, resolver, or progress. |

## Implemented work

1. Restored and hash-verified the missing replay artifacts; replaced silent
   frozen-progress defaults with explicit empty-state migration or a hard
   migration error.
2. Updated stale live fixtures to current resolver/action ownership and added
   explicit private/group scene fields.
3. Preserved evidence scope in the goal prompt without introducing a
   category-to-stance veto. Current-episode facts, typed identity facts, and
   historical evidence are kept distinct and conflicts remain visible to the
   character process.
4. Preserved capability-neutral reminder goals and approval-preparation
   ownership.
5. Added inline-only validation for `task_resolution_request` and projected
   unavailable task owners out of resolver affordances across worker, route,
   and repository health signals.
6. Added the deterministic contradiction test for unavailable resolver owners
   and tightened C07/C08/C11 semantic assertions.

## Independent review decisions

- Luna (`gpt-5.6-luna`, max) identified hidden impacts in replay migration,
  relational evidence scope, runtime availability projection, inline-only
  routing, non-accepting stance effect composition, terminal dialog checks,
  and empty visible-boundary residue.
- Sol (`gpt-5.6-sol`, xhigh) decided that taboo/coercion were not valid hard
  non-acceptance regressions, required the exact C07/C08/C11 outcomes, and
  rejected silent replay fallback or synthetic resolver substitution. The
  synthetic C07 authorization test is explicitly labelled as synthetic.

## Verification contract and result

- Every primary live-LLM node was run individually with
  `venv\Scripts\python` and its console/semantic result inspected.
- Deterministic prompt, resolver-authorizer, action-planner, and connector
  tests passed: 82.
- The affected C07/C08/C11 nodes were rerun after the final production
  projection/prompt changes and passed.
- Active-path scans found no `forbidden_phrases`, generic unsafe-content
  classifier, or application-owned semantic refusal gate. Character-owned
  branches and operational validation remain intact.
- Live-DB nodes remain separately unavailable because the explicit test DB
  guard and isolated database configuration are absent; they are not counted
  in the primary rate.

## Ownership

- Parent agent owned edits, integration, deterministic verification, live
  execution, and sign-off.
- Luna owned read-only impact and hidden-regression review.
- Sol owned the architectural classification decision for ambiguous
  character-owned cases.
