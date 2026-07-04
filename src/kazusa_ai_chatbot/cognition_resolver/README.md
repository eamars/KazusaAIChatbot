# Cognition Resolver ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.cognition_resolver`
- Runtime owner: `kazusa_ai_chatbot.nodes.persona_supervisor2.stage_1_goal_resolver`
- Primary caller: persona graph after message decontextualization
- Primary consumers: cognition chain core, RAG3 local-context evidence,
  action-spec ledger, L3 text surface, consolidation, and resolver telemetry
  helpers
- Contract owners: `contracts.py`, `state.py`, `loop.py`,
  `capabilities.py`, `pending.py`, and `telemetry.py`
- Related docs: [Cognition Nodes](../nodes/README.md),
  [Action Spec](../action_spec/README.md),
  [Local Context Resolver ICD](../local_context_resolver/README.md),
  [Retired RAG 2](../rag/README.md),
  [Brain Service ICD](../brain_service/README.md)

This ICD defines the bounded recurrence boundary between the persona graph and
the layered cognition stack. The resolver is the default live persona workflow;
it is not an optional mode, a generic agent harness, or a direct tool runner.

## Why This Module Exists

The resolver was introduced to remove the old mandatory RAG-first persona path
without making Kazusa shallow or tool-blind. The product problem was not "call
RAG earlier"; it was that goal-directed turns need a way to preserve the
original user goal, ask for missing human-owned information, gather evidence
only when cognition decides it is needed, recover from blocked evidence paths,
and finish with a useful answer or a defensible blocker.

The resolved architecture from the development plans is:

```text
decontextualizer
  -> bounded cognition resolver
       cycle:
         L1 affect/subtext
         L2a consciousness + L2b boundary
         L2c1 judgment + L2c2 social context
         L2d action and capability selection
       optional cognition-selected capability observation
       observation projected into next cognition cycle
  -> memory lifecycle
  -> selected L3 surface or private/no-response finalization
```

Local context recall, public answer research, HIL, approval preparation, and
private self-resolution are resolver capabilities only after L2d selects them.
Evidence does not speak as persona, and deterministic code does not decide that
evidence is semantically needed.

## Ownership Boundary

The resolver owns:

- `ResolverCycleStateV1` initialization, validation, projection, and bounded
  recurrence state.
- Execution of one immediate `ResolverCapabilityRequestV1` per cycle.
- Conversion of capability results into prompt-safe `ResolverObservationV1`
  rows.
- Duplicate-request blocking, max-cycle blocking, and capability timeouts as
  structural observations.
- Pending HIL and approval resume rows backed by the action-attempt ledger.
- Projection of resolver observations, pending state, and goal progress into
  `resolver_context` for the next cognition pass.
- Sanitized local telemetry and human-readable trace helpers.

The resolver does not own:

- Platform parsing, adapter delivery, or delivery receipts.
- Raw RAG helper-agent internals or RAG evidence synthesis.
- Semantic decisions about whether the character needs evidence, should speak,
  should wait, or should refuse.
- L3 wording, dialog generation, or final visible text.
- Memory lifecycle target selection, consolidation, scheduler execution, or
  durable memory writes outside pending-resume ledger rows.
- Arbitrary shell, filesystem, database, adapter, or generic MCP tool access.

## Public Entrypoints

The normal graph entry is:

```python
await stage_1_goal_resolver(state)
```

That wrapper performs:

1. `ensure_initial_resolver_inputs(...)`
2. `load_matching_pending_resume_into_state(...)`
3. `call_cognition_resolver_loop(...)`

The reusable resolver entrypoints are:

| Entrypoint | Owner file | Purpose |
| --- | --- | --- |
| `ensure_initial_resolver_inputs(state, max_cycles=...)` | `state.py` | Ensure `rag_result`, `resolver_state`, and prompt-safe `resolver_context` exist before the first cognition cycle. |
| `call_cognition_resolver_loop(...)` | `loop.py` | Run cognition, execute selected capability observations, and repeat within deterministic caps. |
| `execute_resolver_capability_request(request, state)` | `capabilities.py` | Execute one validated capability request and return a prompt-safe observation. |
| `load_matching_pending_resume_into_state(state)` | `pending.py` | Attach one unexpired scoped pending HIL or approval row to current cognition. |
| `apply_pending_resolution(state, resolution)` | `pending.py` | Apply L2d's decision to close, approve, reject, continue, or supersede a pending row. |
| `project_resolver_context(resolver_state)` | `state.py` | Build the bounded prompt-facing resolver context string. |
| `build_resolver_terminal_event(...)` and `write_human_readable_resolver_trace(...)` | `telemetry.py` | Build sanitized diagnostic artifacts. |

Callers should use these entrypoints rather than reaching into helper
functions. Private helpers in `loop.py`, `capabilities.py`, and `pending.py`
are implementation details.

## Runtime Flow

Normal one-cycle path:

```text
state with decontextualized input
  -> ensure initial resolver state and empty RAG shape
  -> load any matching pending resume row
  -> run L1 -> L2 -> L2d
  -> L2d emits no immediate resolver capability
  -> resolver records terminal trace
  -> selected action specs continue to memory lifecycle and surfaces
```

Multi-cycle evidence path:

```text
cycle N cognition
  -> L2d emits resolver_capability_requests[0] with priority="now"
  -> deterministic capability handler runs with timeout
  -> handler returns ResolverObservationV1
  -> observation is appended to resolver state
  -> resolver_context is refreshed
  -> cycle N+1 cognition receives the prompt-safe observation
```

Blocked HIL or approval path:

```text
L2d selects human_clarification or approval_preparation
  -> capability returns status="blocked"
  -> resolver persists one scoped pending resume row
  -> final cognition pass renders a minimal visible question or approval preview
  -> later matching turn loads pending row
  -> L2d emits ResolverPendingResolutionV1
  -> deterministic code applies the pending-row status update
```

Terminal blocker path:

```text
duplicate request, timeout, or max cycle
  -> structural failed observation is projected
  -> final cognition pass decides how to answer or stay private
  -> for user-message source, unresolved terminal blockers may become a visible
     speak action that explains the evidence boundary
  -> for internal sources, terminal blockers stay private unless L2d itself
     selected a visible action
```

## Contract Objects

Schema versions are defined in `contracts.py` and must be preserved when
serializing resolver-owned payloads:

| Contract | Schema version | Meaning |
| --- | --- | --- |
| `ResolverCycleStateV1` | `resolver_cycle_state.v1` | Deterministic recurrence state for one resolver run. |
| `ResolverCapabilityRequestV1` | `resolver_capability_request.v1` | L2d-selected request for one bounded resolver capability. |
| `ResolverObservationV1` | `resolver_observation.v1` | Prompt-safe capability result projected into later cognition. |
| `ResolverCycleTraceV1` | `resolver_cycle_trace.v1` | Bounded review row for one cognition cycle. |
| `ResolverPendingResumeV1` | `resolver_pending_resume.v1` | Durable pending HIL or approval state loaded into later turns. |
| `ResolverPendingResolutionV1` | `resolver_pending_resolution.v1` | L2d decision for updating one active pending row. |
| `ResolverGoalProgressV1` | `resolver_goal_progress.v1` | Cognition-maintained original-goal and deliverable checklist. |

Validators normalize and clip prompt-facing text. Unknown fields are stripped
from validated resolver payloads. Raw-looking local identifiers are not
projected into `resolver_context`; observations are re-aliased as
`resolver_obs_1`, `resolver_obs_2`, and so on.

## Resolver State

`ResolverCycleStateV1` carries:

- `cycle_index` and `max_cycles`
- `status`: `running`, `terminal`, `blocked`, `max_cycles`,
  `waiting_for_user`, or `waiting_for_approval`
- `original_decontexualized_input`
- `observations`
- `cycle_traces`
- `held_action_specs`
- optional `pending_resume`
- `goal_progress`
- `terminal_reason`

`ensure_initial_resolver_inputs(...)` creates the initial state and a normal
empty `rag_result` shape when no evidence has been requested. The first cycle
therefore gives cognition a stable RAG payload without doing full RAG work.

Image-only turns are a narrow bootstrap exception. If
`decontexualized_input` is an empty string and the current `CognitiveEpisode`
already contains a model-visible `image_observation` percept with non-empty
content, the resolver derives only `original_decontexualized_input` and
`goal_progress.original_goal` from that image summary. It does not mutate
`state["decontexualized_input"]`, synthesize body text, or generalize this
fallback to audio or arbitrary attachments.

## Capability Requests

Allowed resolver capabilities are:

| Capability | Handler behavior | Notes |
| --- | --- | --- |
| `local_context_recall` | Runs RAG3 local-context resolution through `run_rag_evidence_for_persona_state(...)`. | Owns local/private/persona/user/conversation memory, relationship, profile, commitment, and recall evidence. |
| `public_answer_research` | Calls `kazusa_ai_chatbot.complex_task_resolver.resolve_complex_task(...)` through declared request/context/options IO. | Owns public, current, external, source-bound answer investigation and returns semantic knowledge sections for cognition to judge. |
| `human_clarification` | Returns a blocked observation and creates a pending HIL row. | L3/dialog renders the actual visible question. |
| `approval_preparation` | Returns a blocked observation and creates a pending approval row. | It never executes the side effect being previewed. |
| `self_goal_resolution` | Allows private internal-source self-resolution; blocks user-message source. | Visible output still requires normal L2d action selection. |

Only requests with `priority="now"` are executed by the loop. Background
priority is part of the schema but is not executed by this recurrence
controller.

The resolver executes at most one immediate capability per cycle. If L2d emits
several valid immediate requests, the first one is selected and the next
cognition cycle can decide what remains necessary.

## RAG And Shared-Memory Prewarm

RAG3 local context remains demand-driven. It runs only when L2d selects
`local_context_recall`. Public/current/external investigation is exposed to
L2d as `public_answer_research` and is handled by the complex task resolver;
any web/source helpers stay internal to that module. The projected observation
contains `knowledge_we_know_so_far`, `knowledge_still_lacking`,
`recommended_next_iteration`, and `evidence_boundary_notes`. These fields are
evidence context for the next cognition cycle, not a resolver-side judgment
about whether the original goal is answered.

There is one separate first-cycle prewarm lane implemented in
`capabilities.py` and joined in `persona_supervisor2_cognition.py` before L2a:

```text
first resolver cycle
  -> start shared-memory prewarm task
  -> L1 runs
  -> L2a joins bounded shared memory evidence, if any
  -> L2b remains independent of the prewarm join
```

This prewarm is not a resolver capability observation. It uses the existing
RAG intake projection plus the shared `PersistentMemorySearchAgent` worker with
one attempt, then may add confirmed shared `memory` collection evidence to
`rag_result.memory_evidence`, but it must not:

- call `resolve_local_context(...)` or run the RAG3 planner/active-node stages
- call full `MemoryEvidenceAgent`
- read scoped `user_memory_units`
- populate `rag_result.answer`
- create `ResolverObservationV1`
- expose worker names, cache keys, raw rows, source ids, or retry details to
  cognition

The lane exists to give L2a a small amount of standing shared-memory evidence
without restoring the old mandatory full RAG-first path.

## Pending HIL And Approval

Pending rows reuse the action-attempt ledger. Resolver pending rows are
deterministic continuation state, not action specs and not adapter sends.

The pending action kinds are:

- `resolver_pending_hil`
- `resolver_pending_approval`

Rows are scoped by platform, channel, current global user, and source message.
`load_matching_pending_resume(...)` loads one unexpired open row for the
current scope, skips rows created by the current source message, and expires
old rows during load. The default TTL is 24 hours.

L2d decides pending resolution semantically by emitting
`ResolverPendingResolutionV1` with:

```json
{
  "schema_version": "resolver_pending_resolution.v1",
  "decision": "answered | approved | rejected | superseded | continue_waiting",
  "reason": "prompt-safe semantic reason"
}
```

The prompt does not expose durable pending row ids. Deterministic code binds
the active pending row and applies only the structural decision. Python must
not infer approval, rejection, or user intent from keywords.

## Goal Progress

`ResolverGoalProgressV1` exists because multi-turn HIL and multi-cycle evidence
can otherwise collapse the original user goal into the latest narrow evidence
request. It is cognition-maintained semantic state, not a deterministic
keyword extraction over user input.

It carries:

- original goal
- current focus
- deliverable statuses
- missing user inputs
- evidence dependencies
- attempted paths
- source-backed facts
- assumptions or inferences
- blockers
- final response requirements

The resolver validates, stores, clips, and projects this checklist. L2d owns
the semantic content.

## Failure Behavior

The resolver can stop or block for structural reasons:

- invalid resolver state or request shape
- unsupported capability kind
- capability timeout
- duplicate capability objective
- repeated timed-out capability kind
- max-cycle exhaustion
- private-only self-resolution requested from a user-message source
- HIL or approval requirement

Structural failures become prompt-safe observations or pending rows when the
turn can safely continue. The following cognition pass decides what the
blocker means for the character and final surface. The resolver should not
fabricate final dialog, silently execute side effects, or run semantic repair
loops in Python.

## Telemetry And Artifacts

`telemetry.py` builds sanitized resolver-cycle and terminal event-shaped
dictionaries for local inspection and future event-log integration. It also
can write bounded Markdown traces under `test_artifacts/cognition_resolver/`.

Telemetry may include:

- cycle counts
- observation counts and statuses
- selected capability kinds
- pending-resume status
- duration labels
- bounded L1/L2/L2d summaries

Telemetry must not include raw prompt text, raw model output, raw platform ids,
database ids, adapter wire payloads, credentials, callback URLs, or unbounded
message bodies. The telemetry helpers do not persist production event-log rows
by themselves.

## Configuration

The resolver is always part of the live persona graph. There is no
`COGNITION_RESOLVER_ENABLED` runtime flag.

The remaining resolver limits are:

- `COGNITION_RESOLVER_MAX_CYCLES`: positive bounded integer
- `COGNITION_RESOLVER_CAPABILITY_TIMEOUT_SECONDS`: positive bounded float

These are structural caps. Raising them increases response-path latency and
should be justified by validation evidence.

## Compatibility Rules

Compatible changes:

- adding optional prompt-safe fields to resolver observations or goal progress
  when validators tolerate absent old fields;
- adding telemetry labels that remain sanitized and bounded;
- adding a new resolver capability only with a reviewed owner, prompt
  projection, deterministic handler, timeout behavior, tests, and docs.

Breaking changes:

- changing schema-version strings;
- changing the meaning of resolver statuses, pending decisions, or capability
  kinds;
- exposing raw ids to LLM-facing resolver context;
- making RAG mandatory before cognition;
- executing side effects from `approval_preparation`;
- allowing deterministic code to decide semantic user intent, approval,
  preference acceptance, goal satisfaction, or whether evidence is needed.

## Testing Contract

Focused deterministic tests live in:

- `tests/test_cognition_resolver_contracts.py`
- `tests/test_cognition_resolver_loop.py`
- `tests/test_cognition_resolver_persona_graph.py`
- `tests/test_cognition_resolver_l2d_contract.py`
- `tests/test_shared_memory_prewarm.py`
- `tests/test_persona_supervisor2_cognition_prewarm.py`

Useful focused command:

```powershell
venv\Scripts\python -m pytest `
  tests\test_cognition_resolver_contracts.py `
  tests\test_cognition_resolver_loop.py `
  tests\test_cognition_resolver_persona_graph.py `
  tests\test_cognition_resolver_l2d_contract.py `
  tests\test_shared_memory_prewarm.py `
  tests\test_persona_supervisor2_cognition_prewarm.py `
  -q
```

Live LLM resolver validation must be run one case at a time with trace output
inspected. A resolver case passes only when the final behavior completes the
original user goal, validly asks for missing user-owned information, or gives a
defensible evidence/system blocker. Calling RAG, asking one question, or
reaching a terminal state is not sufficient by itself.

## Design Invariants

- Every semantic decision runs through L1 -> L2 -> L2d.
- The resolver is recurrence around cognition, not a second assistant.
- RAG returns evidence; cognition decides what the evidence means.
- HIL and approval are blocked observations plus pending state, not direct
  resolver-authored messages.
- L2d-selected final action specs are separate from resolver capability
  requests.
- Deterministic code owns validation, limits, permission gates, persistence of
  pending rows, timeout handling, and trace construction.
- Deterministic code must not keyword-classify user intent or repair malformed
  semantic decisions into a different channel.
- Non-user sources may remain private after terminal blockers unless L2d
  selects a visible action.
- Raw reflection output, adapter wire syntax, storage internals, and raw tool
  payloads do not enter resolver prompt context.
