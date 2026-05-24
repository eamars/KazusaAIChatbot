# rag3 router interpreter poc experiment plan

## Summary

- Goal: Design and later test a RAG3 proof of concept that separates the
  current RAG2 initializer responsibility into a multi-tier semantic routing
  model above the existing RAG2 subagents, then decide whether the split is
  worth implementing beyond shadow or experiment mode.
- Plan class: medium experiment and POC design plan
- Status: draft
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: compatible experiment only at first; no production
  RAG2 behavior change, no cognition change, no database migration, and no
  runtime replacement of `call_rag_supervisor` until a later approved
  implementation plan explicitly changes status and scope.
- Highest-risk areas: repeating the RAG2.1 initializer simplification failure,
  losing speaker scope, treating semantic segments as backend query plans,
  touching low-level subagent behavior, adding unbounded LLM calls, invalid
  dependency graphs, hiding routing errors behind string slots, using RAG3 to
  hide RAG2 retrieval weaknesses behind more retries, and making cognition
  consume planning artifacts instead of evidence.
- Acceptance criteria: the experiment produces a readable comparison of RAG2
  initializer output versus candidate RAG3 router/interpreter output over the
  same cases, identifies a preferred POC design or rejects RAG3 for now, and
  preserves the current production RAG2 evidence contract throughout the
  experiment.

## Context

Current RAG2 live path:

```text
stage_1_research
  -> quote-aware RAG wrapper
  -> call_rag_supervisor
  -> rag_initializer
  -> rag_dispatcher
  -> top-level capability agent
  -> rag_evaluator
  -> rag_finalizer
  -> project_known_facts
  -> state["rag_result"]
  -> stage_2_cognition
```

Current `call_rag_supervisor` public shape:

```python
{
    "answer": str,
    "known_facts": list[dict],
    "unknown_slots": list[str],
    "loop_count": int,
}
```

The RAG2 initializer currently performs several responsibilities in one LLM
stage:

- deciding whether retrieval is needed;
- separating the user input into evidence needs;
- choosing the broad capability route through string prefixes;
- preserving speaker, time, quote, and dependency hints inside natural
  language slots;
- producing ordered `unknown_slots` that the dispatcher maps to top-level
  capability agents.

The dispatcher mostly relies on string prefixes:

```text
Live-context:
Conversation-evidence:
Memory-evidence:
Person-context:
Web-evidence:
Recall:
```

The top-level capability agents remain the correct execution boundary for a
RAG3 POC. The RAG3 interpreter must not route directly to low-level MongoDB
helpers, worker internals, raw search implementations, or backend parameters.
Low-level subagents and their internal selectors, workers, search helpers,
rerankers, projection payloads, and result contracts are explicitly out of
scope for RAG3. RAG3 may only improve the tiered decision about which existing
subagent receives which semantic task, in which order, and with which
scope/dependency hints.

The completed RAG2 recall-quality hardening work is archived at:

```text
development_plans/archive/completed/short_term/rag2_mainline_fusion_recall_quality_plan.md
```

That work improved production RAG2 evidence quality and deliberately kept RAG3
routing deferred. This draft now starts from that baseline. If RAG3 later needs
to touch production RAG files, it requires explicit user approval and a new
execution boundary.

## Direction From RAG2 Recall Closeout

The 2026-05-24 RAG2 recall-quality attempt produced this direction for RAG3:

- Do not use RAG3 to add more loop capacity. Production RAG2 now uses a
  universal four-loop supervisor cap; RAG3 should reduce retries and make
  common recall paths fit below four loops.
- Prioritize first-pass route precision: select the right memory family,
  speaker scope, relation requirement, time scope, and evidence source before
  retrieval starts.
- Keep low-level subagents frozen in the first experiment. Improve the
  routing and segmentation contract above them instead of rewriting search
  workers.
- Treat continuation and refinement as targeted repair only, not the normal
  path for hard recall.
- Judge RAG output from the cognition consumer's point of view. Compact,
  accurate facts and explicit uncertainty matter more than rich source
  provenance in the final evidence surface.

## RAG2 Weaknesses Under Evaluation

| RAG2 weakness | RAG3 proposal lever | Residual risk |
|---|---|---|
| The initializer is overloaded with segmentation, routing, ordering, and slot wording. | Split semantic segmentation into a router and capability selection into an interpreter. | Two LLM calls can add latency and error surface if the contracts are not small. |
| Slots are natural-language strings with route prefixes. | Use typed router and interpreter JSON before compiling to existing slots or capability tasks. | The compiler can still hide semantic mistakes if validation is weak. |
| Router and interpreter decisions are coupled in one prompt. | Let the router describe evidence needs without capability mechanics, then let the interpreter map needs to supported capabilities. | The interpreter may still infer unsupported backend behavior unless prompted and validated. |
| Speaker scope, quote anchors, and dependency hints are fragile. | Make speaker scope, anchors, time scope, cardinality, and dependencies first-class fields. | Typed fields only help if live LLM output is inspected against hard cases. |
| RAG2.1 failed by simplifying the initializer too much. | Keep existing RAG2 executor, evaluator, finalizer, projection, Cache2, and quote-aware boundaries during the first POC. | A naive RAG3 rewrite can repeat the same failure under a cleaner name. |
| Closed-loop refinement is mostly retrieval-result driven. | Add a bounded pre-execution refinement loop for invalid or underspecified plans. | Unbounded refinement would hurt latency and make the live path harder to inspect. |
| Cache2 stores initializer-level decisions at a coarse granularity. | Segment-level planner output may later support more precise planner caching. | Cache behavior is out of scope for the first POC and must not be changed early. |
| Precision failures cause downstream rework, wrong retrieval, or later correction loops. | Add a tiered route model that resolves retrieval need, memory family, scope, dependency, and capability before execution. | More tiers can increase cost unless the common path exits after one LLM call. |

## Design Goals

RAG3 is a routing-layer proposal, not a subagent rewrite.

Primary goals:

- High precision: route to the right existing subagent with correct scope,
  speaker, time, cardinality, and dependency hints before retrieval starts.
- Less rework: reduce downstream re-routing, wasted retrieval, and broad
  follow-up loops by making the first execution plan more exact. The next
  iteration is allowed only as a targeted repair or clarification step.
- Low cost: keep the common path to one planner LLM call before retrieval,
  avoid batch-style agent negotiation, and reuse existing subagents without
  expanding their prompts or tool contracts.

Operating assumptions:

- The low-level subagents are untouched.
- Existing top-level RAG2 capabilities remain the routing targets.
- RAG3 can be multi-tier internally, but tiers must be conditional. A simple
  case should not pay for a full router plus interpreter plus refinement loop.
- Validator feedback may trigger one focused repair pass only when structured
  output is invalid, underspecified, or internally inconsistent.
- Precision is measured before cost is optimized. A cheaper planner that routes
  to the wrong memory family is not acceptable.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing RAG planner prompts,
  router/interpreter contracts, LLM call count, context budget, or execution
  boundary.
- `debug-llm`: load before running live LLM router/interpreter cases,
  comparing outputs, or writing human-readable review artifacts.
- `py-style`: load before editing Python experiment or production files.
- `cjk-safety`: load before editing Python files or tests that contain CJK
  string literals.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- After automatic context compaction, reread this entire plan before
  continuing implementation, verification, handoff, lifecycle updates, or final
  reporting.
- After signing off a major checklist stage, reread this entire plan before
  starting the next stage.
- Do not execute implementation steps while `Status` is `draft`.
  Implementation requires user approval and status `approved` or
  `in_progress`.
- Do not modify production RAG2 behavior in the experiment stage. Production
  `stage_1_research`, `call_rag_supervisor`, `rag_result`, cognition, dialog,
  consolidation, adapters, and database behavior remain unchanged.
- Do not change the active RAG2 cognition-ready evidence plan from this plan.
  If execution order matters, ask the user before continuing.
- Do not treat RAG3 router output as persona, stance, final answer, cognition,
  or dialog wording. RAG returns evidence; cognition decides stance; dialog
  renders visible wording.
- Do not ask the router or interpreter to generate MongoDB filters, backend
  search parameters, index choices, embedding parameters, cache keys, or raw
  worker internals.
- Do not change low-level subagent prompts, selectors, search helpers, rerankers,
  projection payload shapes, or result contracts as part of RAG3.
- Deterministic validation may validate LLM-emitted structured fields, enums,
  dependency references, and route support. It must not become a keyword-based
  semantic classifier over raw user input.
- Do not add persistent Cache2 entries for RAG3 experiment output.
- Do not add new databases, collections, indexes, schemas, migrations, or
  environment variables.
- Do not run live LLM tests in batches. Run one case at a time and inspect the
  output.
- Keep all experiment artifacts under `experiments/rag3_router_interpreter/`
  and `test_artifacts/rag3_router_interpreter/` if implementation is approved.
- Do not import experiment code from production source.

## Must Do

- Preserve the current RAG2 public evidence contract during the experiment.
- Preserve existing low-level subagent behavior and contracts during the
  experiment.
- Use current RAG2 initializer output as the baseline for every case.
- Include hard cases from the RAG2.1 failure history:
  - speaker scope preservation;
  - named third-party person context;
  - dependency ordering;
  - opening or status inputs that must not request weather;
  - quote/reply and vague-reference inputs;
  - active recall versus completed recall;
  - media and attachment evidence;
  - routine greeting or welcome inputs where no retrieval should run.
- Compare at least four candidate POC designs:
  - tiered low-cost route compiler;
  - compatibility compiler;
  - direct capability plan;
  - single typed planner control.
- Evaluate bounded closed-loop refinement as a cross-cutting option, not as the
  default production assumption.
- Produce a human-readable design decision report before any production POC is
  requested.
- If a later POC is approved, keep the first implementation in shadow mode or
  experiment mode and compile to existing RAG2 slots before attempting direct
  production capability execution.

## Deferred

- Do not replace RAG2 production routing in this plan.
- Do not modify low-level subagents or their internal routing/search behavior.
- Do not redesign RAG retrieval, memory search, conversation search, recall
  search, graph RAG, DAG execution, or evidence formatting here.
- Do not change cognition prompts, dialog prompts, consolidation, reflection,
  self-cognition, scheduler behavior, or adapter delivery.
- Do not add planner Cache2 behavior in the first POC.
- Do not remove `rag_initializer` or `rag_dispatcher`.
- Do not promote RAG3 into production without a later approved plan and
  independent review.

## Cutover Policy

Overall strategy: compatible experiment only, followed by a separate approval
gate for any POC implementation.

| Area | Policy | Instruction |
|---|---|---|
| Production RAG2 path | compatible | Invoke or inspect existing RAG2 as baseline only. |
| RAG3 router/interpreter | compatible experiment | Implement only under experiment or test-only paths until approved. |
| Low-level subagents | compatible | Keep prompts, selectors, tools, result payloads, and retrieval behavior untouched. |
| POC option A compiler | compatible | Compile typed RAG3 plans back to current `unknown_slots` first. |
| POC option D tiered compiler | compatible | Prefer conditional tiers and one-call common path while compiling to current `unknown_slots`. |
| Direct capability execution | deferred | Compare as a design candidate but do not use as the first live POC path without approval. |
| Cognition and dialog | compatible | No changes. RAG3 planning artifacts must not enter prompt-facing cognition. |
| Database and cache | compatible | No new stores, migrations, indexes, persistent cache writes, or schema changes. |

## Target State

The experiment should answer:

```text
Can a separated RAG router and RAG interpreter produce more inspectable,
more valid, and at least equally useful retrieval plans than the current RAG2
initializer, without increasing live-path risk enough to reject the split?
```

The preferred initial POC target, if the evidence supports it, is:

```text
RAG3 tier 1 route planner
  -> typed retrieval decision, segments, and route confidence
  -> validator
  -> optional tier 2 interpreter only for compound, dependent, or low-confidence cases
  -> optional one-pass repair only for validator failures
  -> compatibility compiler
  -> current RAG2 unknown_slots
  -> existing dispatcher, capability agents, evaluator, finalizer, projection
```

This target keeps the blast radius small. It tests the architecture question
without replacing retrieval, evidence formatting, cognition, or dialog.

## Candidate POC Designs

### Option D - Tiered Low-Cost Route Compiler

Shape:

```text
tier 1 planner JSON
  -> validator
  -> direct compile for simple/high-confidence cases
  -> optional tier 2 interpreter for compound/dependent/low-confidence cases
  -> optional one-pass repair for structural failures
  -> current RAG2 unknown_slots
```

Tier responsibilities:

| Tier | Owner | Purpose | Cost rule |
|---|---|---|---|
| Tier 0 | deterministic preflight | Check empty input, malformed planner JSON, enum validity, id references, and unsupported capability names. | No semantic routing over raw input. |
| Tier 1 | route planner LLM | Decide retrieval need, memory family, scope, segment list, dependency shape, and initial capability route. | Always at most one LLM call. |
| Tier 2 | interpreter LLM | Expand only compound, dependent, ambiguous, or low-confidence Tier 1 output into exact capability steps. | Skipped for simple/high-confidence cases. |
| Tier 3 | validator feedback | Ask the failing stage to repair invalid structured output once. | Used only for structural invalidity or missing required scope. |

Benefits:

- Best fit for the user's current design goals: high precision, less rework,
  and low LLM/time cost.
- Keeps all low-level subagents untouched.
- Preserves the option A safety property by compiling to current
  `unknown_slots`.
- Avoids paying for a full router plus interpreter on simple memory lookups.
- Makes the "next iteration" a targeted repair pass rather than an open-ended
  agent loop.

Risks:

- The Tier 1 prompt becomes more important because it owns the common path.
- Confidence scoring can become vague if not defined as route sufficiency
  rather than model self-confidence.
- Some compound cases may be under-routed if Tier 1 incorrectly skips Tier 2.

Default recommendation: make this the primary POC candidate. Treat option A as
the simpler two-stage baseline and option C as the one-call control.

### Option A - Compatibility Compiler

Shape:

```text
router JSON -> interpreter JSON -> compiler -> current RAG2 unknown_slots
```

Benefits:

- Lowest production risk.
- Preserves existing dispatcher, capability agents, evaluator, finalizer, and
  projection.
- Makes router and interpreter output inspectable while keeping downstream
  behavior familiar.
- Provides a fair comparison against the current initializer because both
  ultimately enter the same RAG2 execution surface.

Risks:

- The compiler may reintroduce string-slot fragility.
- Some typed fields may be lost when converted back into natural-language
  slots.
- It may prove the planning split without proving the long-term direct
  capability interface.

Default recommendation: keep this as the clean two-stage baseline. Use it if
option D's conditional tiers are too hard to validate or if Tier 1 skipping
causes precision loss.

### Option B - Direct Capability Plan

Shape:

```text
router JSON -> interpreter JSON -> top-level capability task calls
```

Benefits:

- Removes string prefix dispatch from the RAG3 path.
- Preserves semantic fields through execution.
- Makes dependency ordering and capability ownership clearer.

Risks:

- Larger blast radius than option A.
- Requires a new execution boundary around capability agents.
- Needs more compatibility tests for supervisor traces, evaluator input,
  projection payloads, and `rag_result`.

Default recommendation: evaluate as a design candidate, but do not choose this
as the first POC unless option A fails specifically because the compiler loses
critical semantics.

### Option C - Single Typed Planner Control

Shape:

```text
one LLM call -> typed segments and capability steps in one JSON object
```

Benefits:

- Lower latency than a two-call router and interpreter split.
- Tests whether typed output alone fixes most RAG2 initializer weaknesses.
- Useful as a control against the added complexity of option A and option B.

Risks:

- Does not satisfy the architectural goal of separating router and interpreter
  responsibilities.
- Can keep the same overloaded-planner failure mode in a nicer schema.
- Makes closed-loop refinement less targeted.

Default recommendation: keep as a control, not as the target RAG3 architecture.

### Closed-Loop Variants

Closed-loop refinement is a cross-cutting experiment choice:

| Variant | Rule | Default |
|---|---|---|
| No refinement | Invalid planner output fails the case. | Use for baseline measurement. |
| Validator feedback once | Deterministic validator sends one concise correction to the failing stage. | Preferred RAG3 POC setting. |
| Tiered repair | Validator requests one repair from Tier 1 or Tier 2 only when structured output is invalid or missing required scope. | Preferred option D setting. |
| Router-interpreter negotiation | Interpreter can request router clarification, then router may revise segments once. | Evaluate offline only unless latency is acceptable. |

The live-path POC may use at most one pre-execution refinement pass. Existing
RAG2 retrieval-result continuation remains the owner for post-retrieval
follow-up. Refinement is a repair mechanism, not a second chance to broadly
rethink every route after retrieval has already started.

## Proposed Contracts

### Router Output

The router owns semantic separation only.

```python
{
    "segments": [
        {
            "segment_id": "s1",
            "need": "What evidence is needed?",
            "evidence_domain": "prior_conversation",
            "initial_capability": "Conversation-evidence",
            "speaker_scope": "current_user",
            "anchors": ["quoted phrase", "named person", "object"],
            "time_scope": {
                "kind": "recent",
                "text": "last night",
            },
            "cardinality": "many",
            "depends_on": [],
            "non_goals": ["do not check weather"],
            "needs_retrieval": True,
            "route_confidence": "high",
            "requires_interpreter": False,
        }
    ],
    "no_retrieval_reason": "",
}
```

Allowed `evidence_domain` values:

```text
prior_conversation
durable_memory
person_context
live_current_fact
active_recall
public_web
none
```

Allowed `speaker_scope` values:

```text
current_user
active_character
any_speaker
resolved_person
none
```

Allowed `cardinality` values:

```text
one
many
count
ranking
summary
none
```

Allowed `route_confidence` values:

```text
high
medium
low
```

`route_confidence` means route sufficiency, not model self-confidence:

- `high`: one existing top-level capability is enough and required scope fields
  are present.
- `medium`: route family is likely but task wording, dependency, or cardinality
  needs interpreter expansion.
- `low`: router sees retrieval need but cannot safely choose capability or
  scope without Tier 2.

`requires_interpreter` must be `True` when:

- more than one capability is plausible;
- the segment depends on another segment's result;
- a named person must be resolved before message or memory lookup;
- the user asks for ranking, count, comparison, or multi-hop evidence;
- `route_confidence` is `medium` or `low`.

### Interpreter Output

The interpreter maps only the segments that need Tier 2 into supported
top-level capabilities. It does not run for simple high-confidence Tier 1
segments in option D.

```python
{
    "steps": [
        {
            "step_id": "p1",
            "segment_id": "s1",
            "capability": "Conversation-evidence",
            "task": "Find prior conversation evidence for the segment.",
            "depends_on": [],
            "expected_refs": ["message", "speaker", "local_time"],
        }
    ],
    "feedback": [],
}
```

Allowed `capability` values:

```text
Live-context
Conversation-evidence
Memory-evidence
Person-context
Recall
Web-evidence
```

The interpreter may emit feedback when router output is insufficient:

```python
{
    "feedback": [
        {
            "segment_id": "s1",
            "issue": "missing_speaker_scope",
            "message": "The segment names a person but does not say whether the evidence should be about the current user, character, or third party.",
        }
    ]
}
```

### Compatibility Compiler Output

Option A compiles interpreter steps into current RAG2 slots:

```python
[
    "Conversation-evidence: Find prior conversation evidence for ...",
    "Memory-evidence: Check durable memory for ...",
]
```

The compiler is deterministic. It may preserve typed metadata in experiment
trace artifacts, but the production-compatible execution input remains
`unknown_slots`.

## Validation Rules

Deterministic validators should check only structured LLM output:

- JSON is valid and contains required fields.
- `segment_id` and `step_id` values are unique.
- dependency references point to known prior ids.
- enum values are supported.
- a segment with `needs_retrieval=False` has `evidence_domain="none"` or a
  clear `no_retrieval_reason`.
- a capability step references exactly one segment.
- route capabilities are top-level RAG2 capabilities, not low-level workers.
- low-level subagent, worker, selector, retriever, and search-helper names are
  rejected from planner output.
- `requires_interpreter=False` is accepted only when `route_confidence="high"`
  and `initial_capability` is a supported top-level route.
- backend implementation terms such as MongoDB filters, vector dimensions,
  embedding fields, raw collection names, cache keys, and SQL-like clauses are
  rejected from planner output.
- option A compiler output uses only supported RAG2 route prefixes.

The validator must not override semantic meaning by keyword matching the raw
user input. It can only accept, reject, or ask the LLM stage to repair its own
structured output.

## Experiment Case Set

The case set should combine existing deterministic tests, archived RAG2.1
failure modes, and selected real-data cases from prior RAG2 experiments when
available.

Required case families:

| Case family | Purpose |
|---|---|
| no-retrieval greeting or welcome | Prove RAG3 does not retrieve only because input exists. |
| opening or current-status question | Prove RAG3 does not add weather unless asked or clearly needed. |
| vague reply or quote reference | Preserve quote-aware context and current-turn exclusion behavior. |
| named third-party person | Preserve person-context routing and avoid treating third party as current user. |
| current-user preference memory | Route durable scoped memory without losing speaker scope. |
| prior conversation event | Route conversation evidence with speaker and time hints. |
| media-heavy conversation evidence | Preserve attachment/image anchors. |
| active commitment recall | Distinguish active, completed, cancelled, and unknown recall. |
| compound request | Produce multiple segments with valid dependencies. |
| unknown or unsupported claim | Return no-retrieval or no-evidence planning cleanly. |

Candidate source files to inspect when building the case set:

- `tests/test_rag_phase3_route_mapping.py`
- `tests/test_rag_initializer_cache2.py`
- `tests/test_rag_phase3_initializer_live_llm.py`
- `tests/test_rag_phase3_real_conversation_live_llm.py`
- `tests/test_persona_supervisor2_rag_supervisor2_live.py`
- `tests/test_rag_recall_live_llm.py`
- `development_plans/archive/completed/short_term/rag_2_1_initializer_subagent_contract_plan.md`
- `development_plans/archive/completed/short_term/rag_phase3_development_plan.md`
- `development_plans/archive/completed/short_term/rag2_phase4_continuation_plan.md`
- `development_plans/reference/evidence/rag2_recall_quality_experiment_plan.md`

## Experiment Stages

### Stage 0 - Approval And Baseline Refresh

- Confirm this plan is `approved` or `in_progress`.
- Record `git status --short`.
- Reread the current RAG README, nodes README, active RAG2 evidence plan, and
  relevant initializer, dispatcher, supervisor, evaluator, projection, and
  quote-aware modules.
- Record any production RAG2 changes that landed after this draft.

Exit gate: current RAG2 baseline and active-plan overlap are understood.

### Stage 1 - Case Matrix

- Build a small but adversarial fixture matrix across the required case
  families.
- For each case, record:
  - raw user input;
  - decontextualized RAG-facing input when available;
  - relevant recent context shape;
  - expected retrieval/no-retrieval decision;
  - expected broad capability route or no-evidence result;
  - special constraints such as speaker scope, time scope, and dependency.

Exit gate: case matrix is readable and does not require production code
changes.

### Stage 2 - Prompt And Schema Prototype

- Draft router, interpreter, and single-planner control prompts under the
  experiment path.
- Keep prompts small and local-LLM friendly:
  - stable role, policy, output shape, and decision procedure in the system
    message;
  - current-run facts and context only in the human message;
  - no backend mechanics;
  - no development-process wording in runtime prompts.
- Add schema validators for router, interpreter, and compiler output.

Exit gate: invalid examples fail deterministically before live LLM runs.

### Stage 3 - Deterministic Fixture Evaluation

- Run router/interpreter over fixed or recorded planner outputs first.
- Compare option D's Tier 1 direct-compile path against the same cases used for
  option A and option C.
- Compare option A compiled slots against expected broad RAG2 route families.
- Compare option C output as a control.
- Record validation failures, dependency failures, unsupported route attempts,
  Tier 2 escalation rate, repair-pass rate, and no-retrieval mistakes.

Exit gate: deterministic harness can grade structure before quality.

### Stage 4 - Live LLM Planning Evaluation

- Run live LLM planner cases one at a time.
- Inspect each Tier 1 planner JSON, Tier 2 interpreter JSON when used,
  compiled slots, validator feedback, and optional refinement.
- Write a human-readable `debug-llm` review artifact with full inputs and
  outputs.
- Do not execute production retrieval yet unless the plan has been explicitly
  approved for an option D shadow run.

Exit gate: live planner behavior is readable enough to decide whether to
continue.

### Stage 5 - Option D Shadow Execution

This stage is optional and requires explicit approval after Stage 4 evidence.

- Compile RAG3 option D output to current RAG2 `unknown_slots`.
- Execute through the existing dispatcher and downstream RAG2 path in an
  experiment harness only.
- Compare:
  - current RAG2 initializer slots;
  - RAG3 compiled slots;
  - Tier 1 direct-compile versus Tier 2 escalation decisions;
  - dispatched capability agents;
  - final projected `rag_result`;
  - loop count and LLM call count.

Exit gate: the same downstream RAG2 machinery can consume RAG3-compiled slots
without hidden production changes.

### Stage 6 - Decision Report

- Summarize option D, option A, option B, option C, and closed-loop variant
  evidence.
- Recommend one of:
  - implement option D as a shadow-mode POC;
  - implement option A as a simpler shadow-mode POC;
  - defer RAG3 and keep improving RAG2;
  - run a narrower follow-up experiment on a specific failure class;
  - draft a separate direct-capability implementation plan for option B.
- Record why rejected options were rejected.

Exit gate: user can approve, reject, or revise the next POC implementation
plan with concrete evidence.

## Metrics

Primary quality gates:

- no must-pass speaker-scope failure;
- no invalid dependency graph after at most one repair pass;
- no unsupported low-level worker or backend route emitted by the interpreter;
- no low-level subagent prompt, selector, search helper, or result-contract
  change required to make RAG3 work;
- no retrieval planned for clear no-retrieval cases;
- no weather route for opening or status cases unless weather is explicitly
  requested;
- no regression against RAG2 broad route family on hard cases unless the
  human-readable review explains why RAG3 is better.

Supporting metrics:

- planner JSON validity rate;
- validator repair rate;
- route-family agreement with current RAG2 baseline;
- route-family disagreement judged useful;
- number of semantic segments per case;
- number of interpreter steps per case;
- Tier 2 escalation rate;
- direct-compile rate for simple high-confidence cases;
- one-pass repair rate;
- LLM call count before execution;
- token size of planner inputs and outputs;
- per-case latency when live LLM timing is available;
- final projected `rag_result` quality for optional option D shadow execution.

Reject or redesign if:

- common-path planning needs more than two LLM calls before retrieval;
- simple high-confidence cases routinely require Tier 2;
- more than one repair pass is needed in normal cases;
- typed fields frequently become empty boilerplate;
- option D loses critical semantics during slot compilation;
- option A outperforms option D because conditional tier skipping causes
  routing mistakes;
- option C performs similarly while router/interpreter separation adds only
  latency and no inspectability benefit;
- RAG3 artifacts become necessary for cognition to understand evidence.

## Future Implementation Surface

If this plan is later approved for execution, expected experiment-only paths
are:

- `experiments/rag3_router_interpreter/`
- `test_artifacts/rag3_router_interpreter/`
- `tests/test_experiments_rag3_router_interpreter.py`
- `tests/test_experiments_rag3_router_interpreter_live_llm.py`

Possible production files to inspect for a later approved POC, not modify
during the experiment draft:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_initializer.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_dispatch.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_evaluator.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_types.py`
- `src/kazusa_ai_chatbot/rag/quote_aware_sequence.py`
- `src/kazusa_ai_chatbot/rag/cache2_policy.py`
- `src/kazusa_ai_chatbot/rag/prompt_projection.py`

## Verification

Use `venv\Scripts\python.exe` for Python commands.

Draft-only verification:

```powershell
git diff --check
```

If implementation is approved later, expected deterministic checks include:

```powershell
venv\Scripts\python.exe -m pytest tests/test_experiments_rag3_router_interpreter.py -q
```

Expected live LLM checks must run one case at a time:

```powershell
venv\Scripts\python.exe -m pytest tests/test_experiments_rag3_router_interpreter_live_llm.py -q -k "<single_case_name>"
```

Required human-readable artifacts for execution:

```text
test_artifacts/rag3_router_interpreter/case_matrix.md
test_artifacts/rag3_router_interpreter/rag3_planner_debug_review.md
test_artifacts/rag3_router_interpreter/rag3_poc_design_decision.md
```

The design decision report must include:

- input and RAG-facing input;
- current RAG2 initializer slots;
- Tier 1 planner output;
- Tier 2 interpreter output when used;
- compiler output for option D and option A;
- single-planner control output for option C;
- validator feedback and repair count;
- Tier 2 escalation decision and reason;
- option-by-option quality judgment;
- latency and LLM call count when measured;
- final recommendation.

## Independent Plan Review

Before this plan can be approved for implementation, run an independent plan
review with a fresh agent. The reviewer must check:

- whether the experiment isolates RAG3 from production RAG2 behavior;
- whether the plan preserves RAG, cognition, and dialog ownership boundaries;
- whether option A is a valid low-risk POC target;
- whether option D is a better default POC target for high precision and low
  cost;
- whether option B is properly deferred;
- whether low-level subagent contracts are genuinely untouched;
- whether the proposed typed contracts are specific enough to validate;
- whether validation avoids deterministic semantic classification over raw
  user input;
- whether live LLM and debug review steps are sufficient for local/weaker LLM
  risk;
- whether this plan stays compatible with the completed RAG2 recall-quality
  baseline and does not reopen closed RAG2 scope.

Record the review result in `Execution Evidence` before asking for approval.

## Acceptance Criteria

- The experiment compares current RAG2 initializer output against RAG3 router
  and interpreter output over the same case matrix.
- The experiment includes the RAG2.1 failure families that motivated caution.
- Candidate options D, A, B, and C are evaluated or explicitly rejected with
  evidence.
- Closed-loop refinement is measured as bounded validator repair, not assumed
  as an open-ended agent loop.
- The recommended first POC, if any, uses option D tiered compatibility
  compilation unless evidence shows conditional tiering hurts precision.
- Low-level subagent prompts, selectors, tools, result payloads, and retrieval
  behavior are untouched.
- Production RAG2 behavior, cognition, dialog, database, cache, and adapter
  delivery remain unchanged.
- The final decision report is readable without opening raw JSON first.
- A later implementation POC cannot start until this plan is approved or a
  superseding approved plan exists.

## Execution Evidence

### 2026-05-23 draft

- User requested an experiment plan that turns the brainstormed RAG3 router and
  interpreter architecture ideas into possible POC design choices for later
  implementation.
- Current RAG2 architecture was inspected before drafting:
  - initializer;
  - dispatcher;
  - supervisor loop;
  - evaluator;
  - projection;
  - quote-aware sequence;
  - Cache2 policy;
  - RAG and nodes READMEs;
  - relevant RAG2 completed and reference plans.
- Draft conclusion:
  - RAG3 is feasible as a POC, but the first POC should route above existing
    subagents and compile typed router/interpreter output back into existing
    RAG2 slots.
  - Direct capability execution should remain a later option because it has
    higher blast radius.
  - Closed-loop refinement should be bounded to one pre-execution repair pass.
- Observed plan context:
  - RAG2 recall-quality production work is completed and archived.
  - RAG3 work must start from that production baseline and remain separate
    unless the user explicitly approves a shared execution order.
- Draft verification:
  - `git diff --check -- development_plans/README.md development_plans/active/short_term/rag3_router_interpreter_poc_experiment_plan.md`
    completed without whitespace errors. Git emitted only the existing
    Windows line-ending warning for `development_plans/README.md`.
- User refinement on 2026-05-23:
  - RAG3 must keep low-level subagents untouched.
  - RAG3 may use a multi-tier routing model to the existing subagents.
  - RAG3's design goals are high precision, less rework through targeted next
    iteration only, and low LLM/time cost.
- Plan update from that refinement:
  - Added option D, tiered low-cost route compiler, as the primary POC
    candidate.
  - Added explicit low-level subagent freeze rules.
  - Changed the preferred target from unconditional router plus interpreter to
    a conditional tiered planner where simple cases compile after one LLM call
    and Tier 2 runs only for compound, dependent, ambiguous, or low-confidence
    cases.
- This draft is not approved for implementation.
