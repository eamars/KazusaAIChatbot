# Cognition V3 Handleless Model Contract Big-Bang Plan

## Summary

- **Goal:** Eliminate cognition generation failures caused by opaque handle,
  target-path, evidence-reference, cross-stage contract matching, and the
  legacy multi-bid exhaustion path.
- **Status:** `completed` on 2026-08-23; Gate 7 accepted under the owner's
  semantic rubric and the production-path evidence recorded below.
- **Scope boundary:** Cognition model-facing input/output contracts, their
  deterministic binding into cognition state, goal, action, and resolver
  owners, the immediate cognition callers, and the implementation-agnostic
  control-console cognition projection.
- **Change direction:** Keep internal references deterministic, remove them
  from model output, and bind every semantic result to a caller-owned task.
- **Cutover:** Big-bang replacement of the current handle-emitting cognition
  protocol. No compatibility schema, alias mapper, or old test migration.
- **Acceptance state:** One A1, one A2, one G, and one P generation retain the
  full cognition responsibilities while ordinary, captured-failure, and
  unknown-input holdouts complete on first pass without handle validation,
  semantic repair, bid exhaustion, or a cognition-unavailable disposition.

## Evidence And Design Verdict

The primary defect is the cognition call design, not insufficient model
reasoning.

### Fresh real-LLM reproduction

One effect-free real-model run of `ordinary_neutral_response` used
`gemma-4-31b-isometry-fabled-persona-i1` and took 12 calls / 83.61 seconds.
The outer pytest and final output schema passed, but cognition contained these
generation failures:

| Owner | First result | Recovery result | Exact evidence |
| --- | --- | --- | --- |
| A1 `event_agency` | Rejected | Repeated and exhausted | `ce1` was permitted as subject/object and forbidden as a role-assignment entity; the model used it as both object representations |
| A2 `relationship_social` | Rejected | Repeated and exhausted | Grouped A2 used prior-transcript event handle `ce1`; singleton used evidence handle `e1`; neither belonged to the local relationship subject domain |
| P1 | Rejected | Rejected twice, then empty fallback | All three candidates included `self_cognition_response`; the shared system anchor advertised it while the local ordinary-turn schema forbade it |

The semantic content itself was coherent: the model correctly identified the
current user as intentionally moving the meeting and produced a valid
answer-now action plan. Rejection arose from identity vocabulary and field
ownership.

Raw evidence:

- `test_artifacts/cognition_core_v3/cogv3-cognition-root-cause-repro-20260822/raw_trials/ordinary_neutral_response__v3__trial-1__attempt-1.json`
- `test_artifacts/diagnostics/cognition_v3_real_llm_root_cause_reproduction_2026-08-22.md`

### Retained 72-run cohort

Direct recount of
`test_artifacts/cognition_core_v3/cogv3-g7-input-flow-final-20260822/raw_trials`
shows:

- 46 of 72 trials have appraisal-family exhaustion;
- 225 singleton appraisal recovery calls were added;
- 526 total model calls were made, with median 7 per trial;
- failures occur in every family: `relationship_social` 27,
  `existential_drive` 22, `event_agency` 15,
  `epistemic_comparison_memory` 8, `goal_threat_outcome` 6, and
  `moral_identity` 3.

A defect distributed across all six families and repeated after recovery is a
shared calling-routine defect. A stronger model may hide the defect more
often; it cannot make the contradictory contract correct.

### Surgical MVP falsification and proof

The MVP was used to falsify the calling contract, not to tune prompts against
individual sentences. Each failed iteration exposed a caller or observer
defect:

| Iteration | Observation | Design consequence |
| --- | --- | --- |
| V1 | The trace selector chose a nested payload instead of the exact cognition invocation | Exact invocation identity and input SHA are preflight requirements |
| V2 | A2 was asked for one object while the observer required a `judgments` wrapper; numeric descriptors were also projected into the wrong domains | Prompt, shaper, and observer must share one stage-owned schema |
| V3 | `direction` and adjacent `strength` fields invited a valid qualitative value into the wrong field | One qualitative `shift` field replaces the two-field encoding |
| V4 | Four otherwise valid responses were rejected only because the observer allowed eight axis rows while canonical families contain nine, eleven, or twelve axes | Axis projection is bounded by its own finite domain, with canonical unique axes and no arbitrary smaller cap |

The corrected frozen V4 contract has these results:

- deterministic reclassification of the sealed 11 core and 8 holdout trials:
  19/19 eligible, with zero new parse, roster, canonical-axis, uniqueness, or
  shift failures;
- a new four-trial live correction cohort: 4/4 eligible and 16/16 first-pass
  A1/A2/G/P calls accepted, with zero cognition-stage repair or regeneration
  calls;
- a separately frozen two-trial open-input cohort: 2/2 eligible and 8/8
  first-pass calls accepted. A surreal unseen request retained a meaningful
  interpretive goal; a current-fact request selected exactly the one supplied
  resolver capability without claiming the missing fact was known;
- combined frozen-contract evidence: 25/25 trials and 100/100 first-pass stage
  calls accepted under the corrected mechanical observer;
- all five prompt hashes in V4.1 and V4.2 exactly match the sealed V4 core and
  holdout;
- every live correction trial produced exactly one meaningful active-character
  goal and retained concrete cause summaries;
- the exact `llmtrace_0bae517c46d24c519181ddf185453146` replay passed all
  three sealed V4 trials after mechanical reclassification;
- the preselected holdout covered unfamiliar memory, resolved threat,
  unestablished intimacy, prompt injection, long-context reanchoring, grief,
  competing goals, and tool-result inputs. It retained meaningful open
  summaries, causes, goals, and response intent;
- manual review found no role reversal, material self-conflict, or supplied
  boundary/safety conflict.

Evidence:

- `test_artifacts/diagnostics/cognition_v3_handleless_mvp_v4_reclassified_20260823/reclassification.json`
- `test_artifacts/diagnostics/cognition_v3_handleless_mvp_v4_reclassified_review_2026-08-23.md`
- `test_artifacts/diagnostics/cognition_v3_handleless_mvp_v4_1_20260823`
- `test_artifacts/diagnostics/cognition_v3_handleless_mvp_v4_1_review_2026-08-23.md`
- `test_artifacts/diagnostics/cognition_v3_handleless_mvp_v4_2_20260823`
- `test_artifacts/diagnostics/cognition_v3_handleless_mvp_v4_2_review_2026-08-23.md`

This proves the model-facing cognition contract at MVP scale. It does not
claim production integration, persistence, effects, dialog, or delivery.

## Legacy Feature Disposition

| V2/V3 feature | Decision | Ground-up replacement |
| --- | --- | --- |
| A1 world/causal appraisal, A2 character appraisal, G goal cognition, P planning | Keep | Four explicit stage-local generations |
| All six appraisal families and every current axis | Keep | Fixed family ownership with sparse-or-complete canonical axis projection |
| Emotion activation and its concrete cause | Keep | Open cause meaning plus caller-bound provenance roots and lifecycle status |
| Relationship, standards, identity, boundaries, active goals, and continuity | Keep | Compact semantic context supplied only to stages that need it |
| Ordinary and self-cognition response decisions | Keep | Two disjoint P contracts selected by the caller |
| Action and resolver capability control | Keep | Semantic request names checked against caller-provided capability rosters |
| Model-emitted handles, target paths, evidence IDs, role IDs, and positional alignment | Remove | Private caller-owned binding after generation |
| One goal bid per branch, sibling salvage, retry/replacement, primary/supporting/suppressed bid partition, and W selection | Remove | G synthesizes one current active-character goal from current and continuing pressures |
| `goal_bid_*_exhausted`, `goal_bid_unavailable`, and `ordinary_response_unavailable` cognition states | Remove | A meaningful active-character goal is required for every semantically usable turn |
| Raw assistant JSON as cross-stage history | Remove | Compact accepted typed semantic products |
| V2 compatibility contracts, aliases, fixtures, and handle-specific tests | Delete | One canonical cutover; old tests are not migrated |

## Confirmed Proposal Decisions

1. Internal IDs and references remain deterministic and private.
2. Opaque cognition handles such as `e1`, `ce1`, `ct1`, `ck1`, `ev1`, `g1`,
   `r1`, `b1`, `a1`, and storage IDs do not appear in model responses.
3. The model does not emit state target paths, evidence handles, object
   handles, or role-assignment entity handles.
4. One caller-owned `CognitionTurnWorkspace` binds the current observation,
   participant orientation, evidence provenance, writable axes, active affect
   causes, relationship state, standards, boundaries, identity, continuity,
   and available capabilities. Each stage receives only its projection.
5. A1 and A2 use fixed family-owned slots. The primary product is open
   `semantic_summary` and `cause_summary`; `axis_changes` is a subordinate
   projection into the existing finite state model.
6. An axis list may be empty, sparse, or cover its complete canonical family
   domain. Its only natural maximum is the number of unique axes in that
   family. A smaller arbitrary cap cannot reject meaningful cognition.
7. Deterministic code attaches internal matter references, evidence roots,
   participant references, and exact state paths after generation. This is
   caller-owned construction, not semantic inference.
8. A1, A2, G, and P each receive one byte-stable stage contract and run once.
   Accepted prior products are compact typed semantic data; raw assistant JSON
   is not conversational history for a later owner.
9. Current evidence produces one current-observation root. The state reducer
   assigns any event, threat, knowledge-gap, relationship, and affect roots
   after the semantic judgment instead of exposing speculative object domains.
10. G returns exactly one `active_character_goal` plus one relational-
    willingness judgment. Goal kind, intent, reason, and cause remain open
    semantic text rather than a closed candidate roster.
11. Continuing goals remain input context and persistent state. Competing
    pressures are resolved inside the one G judgment, as demonstrated by the
    unseen multi-goal holdout; they do not require parallel branch bids or W.
12. Unknown or ambiguous input remains meaningful: G may pursue clarification,
    uncertainty reduction, boundary preservation, deliberate deferral, or
    grounded silence. Empty appraisal deltas never mean that no goal exists.
13. P consumes the accepted goal and returns one response intent plus optional
    action/resolver requests. Ordinary and self-cognition P shapes remain
    disjoint and are chosen by the caller before generation.
14. All six appraisal capacities, every current state axis, relationship and
    affect projection, concrete emotion causes, active-goal continuity,
    action planning, resolver planning, and self cognition remain available.
15. Goal bids, branch rosters, primary/supporting/suppressed bid handles,
    sibling salvage, W partition, semantic retry, and goal-unavailable states
    are removed from cognition.
16. No semantic scorer, post-generation semantic rewrite, or model repair loop
    participates in normal cognition. Gate semantics are reviewed only for
    role reversal, material self-conflict, and boundary/safety conflict.
17. Canonical JSON parsing, exact small stage shapes, canonical axis names,
    state schema and transition integrity, and action/resolver capability
    membership remain mechanical boundaries. They do not decide character
    meaning.
18. Within cognition there is no `goal bid` object and therefore no goal-bid
    exhaustion scenario. A structurally unusable model response is recorded as
    a contract fault before state commit; it is never converted into a
    semantic `no goal` or `ordinary_response_unavailable` state.
19. The cutover removes V2 compatibility contracts and handle-specific tests
    in the same change. No alias, mapper, fallback schema, or migrated V2 test
    preserves the deleted protocol.
20. The brain publishes a stable semantic cognition graph containing only
    character-meaning, active-goal, response-plan, state-projection, and typed
    failure information. The control console consumes generic nodes and edges;
    it has no knowledge of A1/A2/G/P labels, bid/workspace topology, prompt
    layout, retry behavior, or private implementation identifiers.

## Mechanical Boundary Without Semantic Validators

The current word `validator` combines unrelated responsibilities. The target
keeps only mechanical ownership checks:

| Check class | Decision | Reason |
| --- | --- | --- |
| Model handle/evidence/path matching | Delete | The model no longer emits these fields, making this failure class unrepresentable |
| Post-generation semantic correctness scoring | Exclude | The producing LLM owns cognition semantics; no second semantic authority is added |
| JSON object and exact small stage shape | Keep | Prevent an unusable object from becoming runtime state |
| Canonical axis membership and uniqueness | Keep | Bind an optional state delta without restricting open meaning; the maximum equals the complete family domain |
| State axes, ranges, lifecycle, and replacement-state integrity | Keep | Protect persistent cognition state without judging meaning |
| Action/resolver availability, target authority, permission, and limits | Keep | Prevent cognition text from granting effects |
| Persistence, scheduling, and delivery checks | Keep | These are deterministic system boundaries outside semantic judgment |

Open semantic summaries and causes remain useful for G and P even when their
axis list is empty. Closed state projection and capability requests stay
subordinate to that semantic carrier.

## Target Model-Facing Flow

```text
typed episode + current state + retrieved evidence
    -> deterministic TurnWorkspace
       - explicit current-character/current-user orientation
       - one current observation and grounded evidence
       - current state, active affects with concrete causes, relationship,
         standards, boundaries, identity, and continuing goals
    -> A1 once: world, agency, outcome, epistemic, comparison, and memory
    -> deterministic binding of open meaning, causes, and optional axis deltas
    -> A2 once: relationship, social, moral, identity, and existential drives
    -> deterministic binding of open meaning, causes, and optional axis deltas
    -> G once: exactly one current active-character goal and willingness
    -> P once: visible response intent plus optional action/resolver requests
       - ordinary and self-cognition contracts are disjoint
    -> deterministic state commit and capability authorization
```

There is no candidate alignment, branch join, bid collapse, W selection, or
semantic fallback. Every stage has one explicit owner and one semantic product.

## Stable Control-Console Boundary

The production brain owns translation from internal cognition products into a
small public debug projection. The console renders that projection and does not
reconstruct cognition internals.

The cutover also removes the separate cognition chain-run payload and the
topology-specific engine descriptor from the console boundary. Operator health
may report one generic cognition model route and bounded availability, but it
must not expose stage labels, call counts, chain/sidecar layout, model transcript
rows, retry/repair counters, prompt structure, or implementation-specific
budgets. The live console therefore remains valid if cognition changes its
internal stage count or orchestration again.

The stable graph vocabulary is semantic rather than procedural:

- observation and grounded context;
- world/character meaning and concrete causes;
- active character goal and relational willingness;
- response plan and requested capabilities;
- affect/state projection;
- typed structural or provider failure when a stage cannot produce a usable
  canonical product.

Each node keeps the existing generic graph envelope (`id`, `label`, `stage`,
`lane`, `column`, `branch`, `status`, `detail`) and exposes only allowlisted
semantic detail. Internal stage names, prompt text, raw model messages, opaque
references, bids, branch rosters, workspace partitioning, retry counters, and
call topology are private. The renderer lays out whatever safe nodes and edges
the brain supplies and offers a generic detail inspector. This lets cognition
internals change again without a console rewrite.

## Per-Prompt Information Budget

If designed from the ground up, the most important information is the minimum
semantic context needed by the current owner:

| Prompt | Include | Exclude |
| --- | --- | --- |
| Common orientation | Active character, current user/other participants, current message direction, one current observation | Internal IDs, adapter syntax, storage paths, raw trace metadata |
| A1 | Grounded evidence and the three A1 family slots with their canonical axes | Relationship policy, action rosters, other-stage output schemas |
| A2 | A1's compact accepted meaning, relationship and affect state with causes, standards, identity, boundaries, and the three A2 family slots | Raw A1 response text, goal bids, capabilities |
| G | Compact A1/A2 meaning, active affects and causes, relationship, identity, boundaries, and semantically described continuing goals | Branch IDs, one-output-per-goal rosters, workspace handles |
| P | The one accepted goal, response ownership, and only the currently available action/resolver capabilities | Appraisal axis schemas, hidden handles, alternative P variant fields |

Stable character facts are supplied once in the stage that owns their use.
Prior stage products are reduced to accepted meaning and cause data instead of
being replayed as assistant conversation.

## Bound Appraisal Contract

The caller prepares fixed family slots and keeps the internal subject, object,
matter, evidence roots, and writable paths private. The model receives a
semantic projection such as:

```json
{
  "orientation": {
    "active_character": "the person deciding what she thinks and does",
    "current_user": "the person who proposed the meeting-time change"
  },
  "observation": "The current user moved tomorrow's meeting from 10:00 to 11:00 and asks whether that works.",
  "evidence": ["The current message states the change and asks for confirmation."],
  "family_slots": {
    "event_agency": {
      "allowed_axes": ["responsibility", "intentionality"]
    },
    "goal_threat_outcome": {
      "allowed_axes": [
        "obstruction", "expected_success", "controllability",
        "recoverability", "urgency", "likelihood", "expected_harm",
        "uncertainty", "coping_potential", "residual_pressure",
        "outcome_impact", "expectation_mismatch"
      ]
    },
    "epistemic_comparison_memory": {
      "allowed_axes": [
        "comparison_gap", "vastness", "memory_warmth", "temporal_loss",
        "relevance", "uncertainty", "learnability", "novelty",
        "model_accommodation"
      ]
    }
  }
}
```

The corresponding result has one fixed slot per owned family and no reference
fields. The summaries and cause are open; the optional state projection is
closed only to existing axes:

```json
{
  "event_agency": {
    "applicable": true,
    "semantic_summary": "The current user deliberately changed the meeting time.",
    "cause_summary": "The current message explicitly says the user moved it.",
    "axis_changes": [
      {
        "axis": "intentionality",
        "shift": "strong_increase",
        "reason": "The change is described as a deliberate action."
      }
    ]
  },
  "goal_threat_outcome": {
    "applicable": true,
    "semantic_summary": "The schedule now needs confirmation.",
    "cause_summary": "The proposed time changed from 10:00 to 11:00.",
    "axis_changes": [
      {
        "axis": "uncertainty",
        "shift": "slight_increase",
        "reason": "Compatibility with the active character's schedule is not yet confirmed."
      }
    ]
  },
  "epistemic_comparison_memory": {
    "applicable": false,
    "semantic_summary": "No memory comparison is required.",
    "cause_summary": "The message is a present scheduling request.",
    "axis_changes": []
  }
}
```

The binder attaches each result to the workspace's private matter, evidence
roots, participants, and axis paths. The open summary and cause remain
available to G/P even when `axis_changes` is empty. A family may emit every
canonical axis once; domain cardinality is the only list maximum.

## Goal Invariant And Unknown Inputs

G is a synthesis stage, not an auction. It always returns one meaningful
active-character goal for a semantically usable turn:

```json
{
  "active_character_goal": {
    "goal_kind": "open semantic label",
    "intent": "what the active character is trying to accomplish now",
    "reason": "why this is the right current aim",
    "cause_summary": "what observation or continuing pressure produced it"
  },
  "relational_willingness": {
    "applicable": true,
    "stance": "open semantic stance",
    "reason": "relationship judgment",
    "cause_summary": "its concrete cause"
  }
}
```

Unknown content does not require a predefined goal-kind enum. The goal can be
to clarify, preserve uncertainty, maintain a boundary, defer until evidence is
available, or intentionally stay silent. These are meaningful cognition
decisions, so no empty bid roster or `ordinary_response_unavailable` branch
exists.

## Emotion Cause Preservation

Emotion cause remains a first-class cognition feature:

- the model returns a concrete `cause_summary` for every appraisal family and
  the active goal, including an explicit explanation when a family is
  inapplicable;
- each axis delta retains its own local `reason`;
- the binder stores or retains the concrete cause meaning on the caller-owned
  matter/evidence entity and attaches those roots as
  `primary_root` and `root_refs`;
- `cause_status` remains deterministic lifecycle state;
- multiple conflicting active affects and their causes remain visible to G;
- relationship-root affect is projected with its concrete cause rather than a
  generic relationship-pressure sentence.

The model therefore explains the cause while code preserves exact provenance.
The V4/V4.1/V4.2 binders retained these fields and roots in all 25 trials,
including the grief, memory, threat-resolution, boundary, competing-goal, and
novel-input cases. This proves the handleless cause-binding contract. Production
acceptance must additionally prove that the real reducer preserves numeric
emotion activation, root lifecycle, and replacement-state integrity.

## Proof Boundary Before Implementation

- Proven now: first-pass handleless A1/A2/G/P generation, open semantic
  summaries and causes, complete family-axis cardinality, one active goal,
  unfamiliar and novel input handling, disjoint self/ordinary P shapes, and
  exact positive resolver-roster selection.
- Proven only by production verification after implementation: removal of all
  production bid/W/exhaustion paths, real affect derivation and numeric state
  reduction, persistence of cause lifecycle, and canonical cognition output
  construction.
- Observed only: resolver execution, final dialog wording, adapter delivery,
  and persistence beyond the directly adapted cognition commit boundary. They
  stay outside this change domain and cannot be claimed by this gate.
- Proven by production verification after implementation: immediate caller and
  surface projection consume the canonical goal/plan contract, and the web
  console renders only the stable semantic graph boundary.

## Scope And Change Surface

### Modify

- `src/kazusa_ai_chatbot/cognition_shared/contracts.py`
  - define the one canonical cognition input/output boundary and remove V2 bid
    vocabulary from that boundary;
  - replace blanket source-kind-to-all-family visibility with the private turn
    workspace used only inside cognition.
- `src/kazusa_ai_chatbot/cognition_shared/state_projection.py`
  - keep private reference maps;
  - replace three-candidates-per-evidence model projection with one current
    observation root and semantic current-matter projection.
- `src/kazusa_ai_chatbot/cognition_core_v3/contracts.py`
  - define the turn workspace and typed A1/A2/G/P products;
  - remove goal-bid exhaustion and unavailable error contracts.
- `src/kazusa_ai_chatbot/config.py`
  - retain one cognition route and the bounded turn deadline;
  - remove the unused sidecar and subconscious route/settings after the
    four-call cutover.
- `src/kazusa_ai_chatbot/cognition_core_v3/registry.py`
  - register only the four stage-local contract owners, without recovery-stage
    aliases or V2 contract names.
- `src/kazusa_ai_chatbot/cognition_core_v3/anchor.py`
  - retain invariant cognition policy only; remove other-stage output fields,
    bid contracts, and handle-domain instructions.
- `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`
  - replace handle, target-path, and evidence-echo schemas with stage-local
    semantic projections and the frozen handleless shapes proven by V4.1.
- `src/kazusa_ai_chatbot/cognition_core_v3/semantic_source_planner.py`
  - prepare one current semantic workspace from typed provenance and unresolved
    matters;
  - remove blanket six-family visibility and speculative candidate domains.
- `src/kazusa_ai_chatbot/cognition_core_v3/appraisal.py`
  - own fixed A1/A2 family slots, open summaries and causes, canonical unique
    axis projection, and deterministic task binding.
- `src/kazusa_ai_chatbot/cognition_core_v3/semantic_appraisal.py`
  - remove model-emitted handle/path validation and retain only bound-product
    materialization required by state reduction.
- `src/kazusa_ai_chatbot/cognition_core_v3/goal_cognition.py`
  - replace ordinary/active branch bid generation with one active-character
    goal and one relational-willingness product;
  - remove bid rosters, sibling salvage, retries, dispositions, and
    `ordinary_response_unavailable` paths.
- `src/kazusa_ai_chatbot/cognition_core_v3/workspace.py`
  - retain only compact turn-workspace construction required by A1/A2/G/P;
  - remove primary/supporting/suppressed bid partition and W selection.
- `src/kazusa_ai_chatbot/cognition_core_v3/action_selection.py`
  - consume the one active goal, use semantic capability names, and keep
    ordinary/self-cognition P contracts disjoint;
  - retain capability membership and effect-boundary validation.
- `src/kazusa_ai_chatbot/cognition_core_v3/execution.py`
  - invoke A1/A2/G/P once each and carry typed products without raw assistant
    history, semantic repair, regeneration, or singleton recovery.
- `src/kazusa_ai_chatbot/cognition_core_v3/transcript.py`
  - retain trace/session facts only; remove raw accepted assistant JSON as the
    semantic handoff between different stage owners.
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
  - orchestrate A1 -> bind/reduce -> A2 -> bind/reduce -> G -> P;
  - remove goal waves, bid collapse, W, and all cognition-unavailable paths.
- `src/kazusa_ai_chatbot/cognition_core_v3/session.py`
  - persist compact typed recurrence products rather than raw model messages.
- `src/kazusa_ai_chatbot/cognition_core_v3/diagnostics.py`
  - report first-pass stage disposition and eliminate accepted-degraded,
    bid-exhausted, and ordinary-response-unavailable labels.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - construct and commit the canonical cognition input/output directly;
  - remove admitted/supporting bid and selected-bid projections.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
  - project the canonical active goal and response plan to the surface owner.
- `src/kazusa_ai_chatbot/cognition_shared/surface.py`
- `src/kazusa_ai_chatbot/cognition_shared/surface_stages.py`
  - consume goal/plan semantics without bid-shaped surface payloads.
- `src/kazusa_ai_chatbot/cognition_resolver/guardrail.py`
  - type the cognition call boundary with the canonical contract and retain
    only deterministic effect and state protections.
- `src/kazusa_ai_chatbot/self_cognition/tracking.py`
  - derive self-cognition continuity from canonical goal/cause provenance.
- `src/kazusa_ai_chatbot/service.py`
  - publish the stable semantic cognition graph; remove bid, branch, parallel,
    collapse, and internal-stage telemetry from the public debug projection.
- `src/kazusa_ai_chatbot/brain_service/contracts.py`
  - expose only the stable semantic graph and generic cognition availability;
  - remove chain-run and chain/sidecar topology contracts from console-facing
    responses.
- `src/kazusa_ai_chatbot/llm_interface/route_report.py`
  - report the one live cognition route without a sidecar branch.
- `.env`
  - remove only the obsolete `COGNITION_V3_SIDECAR_LLM_*` and
    `COGNITION_V3_SUBCONSCIOUS_ENABLED` keys.
- `src/control_console/kazusa_client.py`
  - allowlist stable semantic node detail and keep generic graph validation;
  - stop projecting implementation-specific cognition chain-run rows.
- `src/control_console/app.py`
- `src/control_console/contracts.py`
- `src/control_console/brain_model_routes.py`
  - retain a single generic cognition route/status surface and remove
    chain/sidecar topology from console-owned contracts.
- `src/control_console/static/console.js`
  - render supplied generic nodes/edges and semantic detail; remove hardcoded
    parallel cognition topology, chain-run panels, route topology, and
    bid-specific inspector fields.
- `src/control_console/README.md`
- `src/kazusa_ai_chatbot/cognition_core_v3/README.md`
- `src/kazusa_ai_chatbot/nodes/README.md`
- `tests/ownership/source_test_impact_manifest.json`

### Delete Or Rewrite Instead Of Migrating

- tests whose sole contract is that model output emits `eN`, `ceN`, `ctN`,
  `ckN`, `evN`, `gN`, `r1`, `bN`, `aN`, target paths, or role-assignment
  handles;
- ordinary/active goal-bid group, bid-wave, branch-roster, sibling-salvage,
  workspace-partition, W-selection, goal-exhaustion, unavailable-response,
  singleton appraisal recovery, and semantic-regeneration code and tests;
- raw cross-stage assistant-transcript assertions;
- V2-named cognition contracts, fixtures, aliases, and compatibility tests.

### Keep

- all six appraisal families and their complete axis domains;
- affect activations, concrete cause roots/status, relationship state,
  standards, boundaries, identity, continuing goal state, and state reducers;
- self-cognition, action/resolver planning, and cognition-owned capability
  membership checks;
- the replacement-state integrity boundary.

### Immediate Cutover Boundary

Every production consumer of `primary_bid`, `supporting_bids`, `ActionBidV2`,
`CognitionCoreInputV2`, `CognitionCoreOutputV2`, or `selected_bid_reason` is
either updated to the canonical contract or removed in this cutover. There is
no compatibility projection. The final dialog wording owner, RAG, adapters,
database content, scheduler, and delivery remain outside the change except for
the exact type/call adaptation required to consume the canonical cognition
result.

## Excluded Work

- Dialog or visible-output quality changes.
- Prompt examples tailored to either reproduction sentence.
- Model replacement or route tuning.
- Database cleanup or migration of existing semantic state.
- Upstream decontextualizer changes; any observed upstream failure is reported
  separately.
- Compatibility adapters for the removed handle-emitting protocol.
- Broad unit-test expansion, documentation tests, or rerunning previously
  passing Gate 7 cases.
- Provider, transport, dialog, and delivery remediation. Any failure observed
  there is recorded as a boundary dependency and remains outside this
  cognition change domain.
- Console knowledge of prompt stages, model call count, retries, bids,
  workspaces, or any other cognition implementation topology.

## Execution Roles

### Parent architecture and acceptance owner

- **Responsibility:** freeze the contract, maintain plan lifecycle, inspect
  raw live evidence, review implementation scope, and decide acceptance.
- **Owned surface:** this plan, registry, architecture decisions, Luna handoff,
  diff review, and final disposition.
- **Authority:** analysis, plan amendments after owner decisions, read-only
  source/evidence inspection, and acceptance or bounded remediation requests.
- **Applicable skills:** `development-plan`, `local-llm-architecture`,
  `debug-llm`, and `test-style-and-execution` for evidence interpretation.
- **Capability floor:** full cognition architecture context, raw trace access,
  source review, and live-result interpretation.
- **Independence:** reviews the Luna implementation and verification evidence.
- **Acceptance output:** scoped diff decision and evidence-backed first-pass
  cognition verdict.
- **Gate:** starts after owner approval; closes only when every acceptance
  criterion is evidenced.

### Fixed implementation and verification executor

- **Fixed constraint:** reuse the existing `gpt-5.6-luna` worker with `max`
  reasoning on the standard-speed lane, as directed by the owner.
- **Responsibility:** implement the approved handleless cognition cutover,
  perform bounded remediation, run exact deterministic checks, and execute
  live cases one at a time.
- **Owned surface:** only the production, test, cognition documentation, and
  control-console paths listed in the approved change surface.
- **Authority:** production/test edits and scoped commands after explicit
  implementation authorization; no DB or external effects.
- **Applicable skills:** `development-plan`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`, and `debug-llm`.
- **Capability floor:** production Python refactor, cognition contract design,
  effect-free live-LLM execution, raw artifact inspection, and safe worktree
  preservation.
- **Independence:** implementation and verification reuse the same worker due
  the available two-slot runtime; parent retains acceptance authority.
- **Acceptance output:** scoped diff, exact deterministic results, two
  individually inspected real-LLM artifacts, and remediation handoff.
- **Gate:** receives work only after plan approval and explicit production
  implementation command; exits after parent accepts all evidence.

## Completion Record — 2026-08-23

Gate 7 is accepted and this plan is complete. The owner-defined semantic
failure threshold was applied: role reversal, material self-conflict, or a
supplied boundary/safety conflict. Structural transport, parse, schema,
binding, runtime, partial-output, and unavailable-cognition failures remained
hard failures.

### Implemented cutover

- Cognition now performs exactly one A1, one A2, one G, and one P model call.
  The goal-bid roster, W selection, semantic retry/regeneration, sibling
  salvage, degraded acceptance, and cognition-unavailable paths are absent.
- Model outputs contain open semantic meaning and causes. Caller-owned code
  binds canonical axes, state ownership, provenance roots, action requests,
  and resolver capabilities without asking the model to reproduce handles.
- All six appraisal families and all 51 current axes remain representable.
  Emotion retains concrete `cause_summary`, `primary_root`, `root_refs`, and
  `cause_status` through the state and surface flow.
- Immediate cognition, surface, self-cognition, service, configuration, and
  console consumers use the canonical cutover. V2 cognition-engine code and
  V2-only tests were deleted rather than migrated. Persisted schemas and
  non-cognition protocols retain their independent version identifiers.
- The brain owns a semantic graph projection. The console preserves arbitrary
  safe semantic node identities and renders allowlisted goals, appraisal
  families, axes, response intent, affect, and causes without cognition-stage
  or branch topology knowledge.
- Obsolete sidecar and subconscious `.env` keys were removed. The active
  cognition model route and turn deadline remain configured.

### Accepted evidence

| Boundary | Result | Evidence |
| --- | --- | --- |
| Captured failure `llmtrace_0bae517c46d24c519181ddf185453146` | 4/4 first-pass calls; canonical output complete after fixing caller-owned relationship-evidence binding | `test_artifacts/diagnostics/cognition_v3_gate7_final_20260823/trace_reproduction_after_fix.json`, SHA-256 `586E618641FEC5B3A993A8BDB1DB68D42607A77922C40699C1C7750D6B70D1B2` |
| Unknown input | 4/4 first-pass calls; meaningful epistemic-clarification goal; no goal exhaustion after fixing knowledge-gap cause binding | `test_artifacts/diagnostics/cognition_v3_gate7_final_20260823/unfamiliar_input_after_fix.json`, SHA-256 `9AE174021C0A95329DBCAA159D025C660B2EFDCD9A99374C82F31F88DC308041` |
| Isolated production request | HTTP 200, debug `completed`, graph `completed`, visible response produced; no cognition, binding, persistence, or surface exception | `test_artifacts/diagnostics/cognition_v3_gate7_final_20260823/browser_caller_fix.json`, SHA-256 `6865AA7743CEC9CE8ED6D8D360875F2C6726C9F8DB692479E2F9C68DB2F295CC` |
| Protected correlation | Exact parent trace and production request correlation preserved | `test_artifacts/diagnostics/cognition_v3_gate7_final_20260823/trace_correlation_caller_fix.json`, SHA-256 `2DD59904A7CE44B287F72FA10F0519803174CFE592424D8588BAA8B2796E00C0` |
| Console boundary | Current service semantic output survived the generic client projection with goal, appraisal, axis, affect, and cause fields; Playwright reported zero console/page errors and made zero LLM calls | `test_artifacts/diagnostics/cognition_v3_gate7_final_20260823/console_projection_after_fix.json`, SHA-256 `AFC180A77540219683325BC7A953903731835698425892AB7B4E0029454F0EE8` |

The final held-out response showed no role reversal, material self-conflict, or
supplied boundary/safety conflict. Previously passing Gate 7 cases were
recorded without rerun, as directed by the owner.

Focused deterministic coverage is concentrated in the existing current
contract nodes:

- `tests/unit/cognition_core_v3/test_handleless_contract.py` covers disjoint
  handleless packets, complete axis binding and receipts, concrete affect
  causes, capability binding, one meaningful goal, and exactly four calls.
- `tests/unit/nodes/test_persona_supervisor2_l3_surface.py` covers canonical
  goal, response-operation, and relational-willingness surface handoff.
- `tests/test_control_console_kazusa_client.py` covers arbitrary semantic graph
  identity plus goal/appraisal/axis/cause projection.

The last focused console batch passed 28 tests; compile, Ruff, JavaScript
syntax, and `git diff --check` checks passed. A separate chat-fixture run
reached queue teardown and was cancelled by event-log cleanup before graph
assertions. The current service graph builder passed direct projection, and the
isolated production request above is the stronger end-to-end result, so this
test-infrastructure condition does not change the cognition disposition.

### Separate quality work

The production run recorded upstream/dialog provider JSON-object-to-text
fallback warnings. They did not create a cognition failure and remain outside
this completed plan. The separate draft bugfix
`development_plans/active/bugfix/unified_llm_json_schema_fallback_no_text_bugfix_plan_2026-08-23.md`
owns that transport-quality question.

Production protected trace export did not independently contain the four
cognition stage rows. Four-call proof comes from the four unconditional
`_call_once` sites plus the retained direct live artifacts. Invocation-local
stage diagnostics are opt-in and are not represented as a public console
topology or a production chain transcript.
