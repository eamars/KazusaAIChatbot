# Live Response Generation Failure Modes Problem Statement

## Summary

- Goal: record the observed and statically identified LLM-generation failure
  modes in the live response path.
- Status: draft
- Plan class: problem-statement-only; non-executable
- Scope boundary: frontline relevance through settled relevance, perception,
  local-context resolution, Cognition V3, memory lifecycle, L3 surface, dialog,
  service failure projection, and post-turn consumers.
- Change direction: unassigned in this draft.
- Acceptance state: unassigned in this draft.
- Implementation authority: this document provides no implementation
  authority.

## Document Boundary

This draft records current behavior, observed failures, historical contract
state, and impact radius. Implementation direction, mitigation behavior,
retry policy, degradation policy, prompt changes, model changes, target state,
change surface, test mapping, verification, and acceptance criteria remain
outside this document.

## Observed Incident

Two protected Cognition V3 traces for the same private QQ interaction stopped
after the P stage:

| Trace ID | Terminal stage reached | Visible result |
|---|---|---|
| `llmtrace_86d6b08eef904e21bd695aa2ee67f032` | `cognition_core_v3.P` | no surface or dialog output |
| `llmtrace_4e447649a95047a69d5a84753d1f4a73` | `cognition_core_v3.P` | no surface or dialog output |

The P response parsed as a JSON object containing all required top-level
fields. Its `response_goal` value was an object copied from the upstream
active-character goal shape. The P validator required `response_goal` to be a
bounded non-empty string and raised `CanonicalContractError`.

The captured P packet was replayed through the configured real LLM three
times. All three outputs returned an object-valued `response_goal`, and all
three failed the same validator rule. The retained artifact is:

`test_artifacts/diagnostics/cognition_v3_response_goal_contract_live_llm/response_goal_1787778973126453200.json`

The captured production packet and raw P response are retained in:

`test_artifacts/diagnostics/trace_08-46.json`

## Historical Contract State

### Cognition V3

The completed handleless Cognition V3 plan established one A1, A2, G, and P
generation per cognition pass. It removed sibling salvage, semantic retries,
goal-bid exhaustion, and unavailable-goal states. A structurally unusable
model response was defined as a contract fault before state commit.

Historical source:

`development_plans/archive/completed/cognition_v3_handleless_model_contract_bigbang_plan_2026-08-22.md`

### Earlier Dialog Continuity Contract

The completed Cognition V2 retry-exhaustion continuity plan recorded dialog
semantic exhaustion as `accepted_degraded`. Its execution record retained
bounded dialog candidates and delivered the newest available candidate through
the normal text path after semantic-check exhaustion.

Historical source:

`development_plans/archive/completed/bugfix/cognition_core_v2_retry_exhaustion_continuity_bugfix_plan.md`

### Current Dialog Contract

The later dialog evaluator decommission removed runtime semantic evaluators,
semantic scoring, and evaluator-driven repair. The current dialog generator
performs bounded generation, JSON and message-shape validation, and required
source-URL fidelity checks. A structurally valid candidate that satisfies
those deterministic checks proceeds without a runtime semantic pass/fail
decision.

Historical source:

`development_plans/archive/completed/dialog_final_generator_evaluator_decommission_plan.md`

Current source:

`src/kazusa_ai_chatbot/nodes/dialog_agent.py`

## Failure-Mode Inventory

### FM-01 — Model Provider Invocation Failure

An LLM owner can receive a connection, timeout, HTTP, provider, runtime, or
model-serving exception instead of a candidate. Failure behavior differs by
stage. Some owners contain provider exceptions inside a bounded attempt loop;
other owners allow the exception to leave the stage.

### FM-02 — Non-JSON Or Unusable JSON Output

A model can return empty text, prose without an object, malformed JSON, a JSON
value that is not an object, or an empty object. Parsing behavior differs by
stage. Cognition V3 uses deterministic-only canonical parsing. Local-context
resolver stages use a stage-local deterministic parser. Other stages use the
canonical parser with their configured repair behavior.

### FM-03 — Parsed Output With Invalid Shape Or Value Type

A response can parse successfully while containing missing fields, extra
fields, wrong field types, unsupported enum values, unavailable capabilities,
duplicate rows, invalid lengths, or conflicting fields. The observed P
incident belongs to this class: JSON parsing succeeded and the generated
`response_goal` value had the wrong type.

### FM-04 — Parse Success Recorded Before Stage Validation

`cognition_core_v3.facade._call_once(...)` records a successful parsed attempt
before the A1, A2, G, or P stage-specific validator runs. A later validator
exception can therefore terminate the stage while the attempt trace already
contains a successful parse disposition.

### FM-05 — Generic Error Classification Of Cognition V3 Contract Failure

`CanonicalContractError` inherits from `ValueError`. The brain service maps a
generic `ValueError` to `internal_invariant`. The observed P output was a model
contract type mismatch and reached the generic service classification path.

### FM-06 — One-Shot Cognition V3 Stage Failure

A1, A2, G, and P each make one direct model call. A provider, parse, or
stage-validator failure prevents the remaining cognition stages from running.
State binding and commit occur only after all four products validate, so these
failures occur before the cognition state commit.

### FM-07 — Frontline Relevance Provider Failure

Frontline relevance contains deterministic fallback decisions for invalid
parsed output. Its provider invocation occurs outside a provider-exception
containment block. A provider exception leaves frontline intake and reaches
the queue failure path.

### FM-08 — Settled Relevance Provider Failure

Authoritative settled-relevance contract failures can enter one repair call.
Provider exceptions from the initial or repair model invocation leave the
settled owner and reach the service's settled operational-failure path.

### FM-09 — Local-Context Provider Failure

Local-context input and stage validation errors become bounded blocked packets.
The public resolver catches `LocalContextValidationError` and `ValueError`.
Provider exceptions from graph planning, active-node resolution, collapse
review, or synthesis are outside those caught classes and can leave the
resolver.

### FM-10 — Optional Memory-Lifecycle Generation Failure

The pre-surface memory-lifecycle specialist performs one model invocation and
then parses and normalizes its result. Provider, parse, or normalization
failure can leave this optional stage. This stage runs after cognition commit
and before background-work enqueue, surface planning, and dialog.

### FM-11 — Text-Surface Generation Exhaustion

Text-surface content planning has a bounded attempt loop. Provider or contract
exhaustion raises a typed surface execution error inside the stage. The public
text-surface owner catches that typed error and returns a deterministic text
surface projected from the already validated cognition response goal and
epistemic boundary.

### FM-12 — Visual-Surface Generation Exhaustion

Visual planning runs as an optional sibling of text planning. A typed visual
surface failure is omitted while text continues. An unexpected exception not
classified as the typed visual surface failure leaves the L3 surface owner.

### FM-13 — Dialog Provider Or Structural Exhaustion

The dialog generator has three producer opportunities. Provider errors,
unparseable output, invalid `final_dialog` structure, empty output, excessive
text, or required source-URL fidelity failure can leave no accepted candidate.
When no candidate is accepted, `DialogGenerationContractError` is raised with
the `post_cognition_commit` checkpoint.

### FM-14 — Dialog Semantic Output

The current dialog path has no runtime semantic evaluator or semantic
pass/fail stage. A candidate that passes structural and required source-URL
checks is delivered. Semantic quality defects can therefore appear in visible
dialog without becoming a runtime generation failure. The historical V2
semantic-exhaustion disposition is not exercised by the current evaluator-free
dialog graph.

### FM-15 — Post-Commit Dialog Failure After Action Work

The persona graph commits cognition before memory lifecycle and L3. Selected
background work is enqueued before L3. Selected non-surface actions execute
before text-surface planning and dialog so their results can enter visible
wording. Dialog exhaustion can consequently occur after cognition commit,
background enqueue, or immediate action execution.

### FM-16 — Post-Turn LLM Consumer Failure

Conversation progress, internal-monologue residue, and consolidation run after
the visible response path. Their service owner catches background exceptions.
These failures can prevent one or more continuity or persistence products from
being recorded after a visible response has already been released.

## Static Impact Radius

| Boundary | Failure checkpoint | Downstream work absent or interrupted |
|---|---|---|
| Frontline relevance | before turn settlement | settlement, cognition, dialog, and delivery |
| Settled relevance | before cognition claim | cognition, dialog, and delivery |
| Media description | before settled relevance | image-specific observation; text path remains available |
| Message decontextualization | before cognition | role-explicit rewrite and resolved referents; original input remains available |
| Local-context resolution | before or during cognition recurrence | local/private evidence packet and dependent resolver observation |
| Cognition A1 | pre-state-commit | A2, G, P, state binding, actions, surface, and dialog |
| Cognition A2 | pre-state-commit | G, P, state binding, actions, surface, and dialog |
| Cognition G | pre-state-commit | P, state binding, actions, surface, and dialog |
| Cognition P | pre-state-commit | state binding, actions, surface, and dialog |
| Memory lifecycle | post-cognition-commit, pre-surface | lifecycle result, background enqueue, surface, and dialog |
| Text surface | post-cognition-commit | model-authored content plan; deterministic degraded surface remains available |
| Visual surface | post-cognition-commit | visual directives |
| Dialog | post-cognition-commit and potentially post-action | visible character dialog, normal assistant-message persistence, and normal delivery tracking |
| Post-turn consumers | after visible delivery | progress, residue, consolidation, or durable continuity products |

## Current Evidence Files

- `test_artifacts/diagnostics/trace_08-46.json`
- `test_artifacts/diagnostics/cognition_v3_response_goal_contract_live_llm/response_goal_1787778973126453200.json`
- `tests/test_cognition_v3_response_goal_contract_live_llm.py`
- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
- `src/kazusa_ai_chatbot/relevance/frontline_relevance_agent.py`
- `src/kazusa_ai_chatbot/relevance/persona_relevance_agent.py`
- `src/kazusa_ai_chatbot/local_context_resolver/service.py`
- `src/kazusa_ai_chatbot/local_context_resolver/stages.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_memory_lifecycle.py`
- `src/kazusa_ai_chatbot/cognition_shared/surface.py`
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/service.py`

## Draft State

This draft ends at the problem statement and failure-mode inventory. It is not
an executable implementation contract.
