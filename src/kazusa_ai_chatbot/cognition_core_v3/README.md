# Cognition Core V3

`cognition_core_v3` is the cache-affine semantic-chain cognition engine. It runs
over the unchanged V2 substrate: the public contract stays exactly
`CognitionCoreInputV2` / `CognitionCoreOutputV2`, and the deterministic head
(elapsed update, preliminary goals, prompt projection, question planning),
final reduction, relationship maintenance, workspace collapse, action planning,
and output assembly reuse the V2 helpers verbatim.

Engine selection happens in `cognition_core_selector`. The closed set is
`v2` / `v3`, selected by the `COGNITION_CORE_ENGINE` setting (default `v2`).
The persona supervisor connector imports one module-level binding from that
selector, so its live, idle, and guarded call sites all share the same engine.

## Public surface

The package exports exactly eleven names, pinned by
`tests/unit/cognition_core_v3/test_public_api.py`:

- `run_cognition(input_payload, services)` — the one public entrypoint; the
  deterministic orchestrator owns chain selection, stage order, visibility,
  checkpoints, attempt caps, validation, and failure disposition.
- `StageResult`, `StageFailure` and the bounded failure classes
  `STRUCTURAL_FAILURE_CLASS`, `PROVIDER_FAILURE_CLASS`,
  `EXHAUSTION_FAILURE_CLASS`.
- The typed error codes `APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE` and
  `BOUNDARY_REJECTED_ERROR_CODE`.
- The per-trace replay capture API: `bind_protected_chain_records()`,
  `snapshot_protected_chain_records()`, `reset_protected_chain_records(token)`.

The goal-bid owner codes (`GOAL_BID_STRUCTURE_EXHAUSTED_ERROR_CODE`,
`GOAL_BID_PROVIDER_EXHAUSTED_ERROR_CODE`) and the `EXHAUSTION_ERROR_CODES` set
are defined in `contracts.py` but stay module-private; they are deliberately not
part of the package export surface.

## Topology

One invocation runs a parallel first wave of registry-ordered appraisal chains
in the exact order declared by `registry.APPRAISAL_FIRST_WAVE_CHAINS`:
`causal_normative`, `relationship`, `epistemic_meaning`. The registry is an
immutable import-time declaration; `_validate_registry()` fails startup when
the first-wave order deviates or when the terminal outcome chain joins the
first wave.

Each preliminary branch kind runs an isolated goal chain (`goal_cognition.py`).
The accepted appraisal prefix reduces into a provisional state, on which one
fresh canonical `terminal_outcome` stage (single-stage chain
`goal_threat_outcome`) runs. Final reduction and exactly one relationship
maintenance pass follow, then dependency-ready branch kinds reactivate before
the complete-bid join in `workspace.py`.

Every wave shares one invocation-wide attempt ledger whose per-stage caps
mirror the V2 model attempt policy (`V2_APPRAISAL_TOTAL_ATTEMPTS` for
appraisal stages, `V2_MODEL_TOTAL_ATTEMPTS` for goal stages). Producer budgets
are keyed `chain:stage` so every stage keeps its own full question budget, and
a shared-budget exhaustion stops the stage at the boundary with a typed
disposition instead of issuing another call.

## Cache-affine transcripts

Each semantic owner runs a bounded cache-affine transcript under its own static
system prompt (`transcript.py`). The transcript binds to the validated
cache-domain identity of the owner's route, measures message sequences in
deterministic UTF-8 bytes against the owner's prompt budget
(`fits_prompt_budget`), and restarts on domain or budget mismatch. Attempt
arithmetic stays deterministic across attempts and cache checkpoints; a
restarted transcript resumes from its last accepted prefix rather than from raw
provider history.

## Terminal outcome stage membership

The terminal producer receives the planned-question map. When no
`goal_threat_outcome` question is planned — which follows the exact V2 planner
existence rule, since question existence is evidence-visibility driven — the
producer returns an accepted contentless local state with zero model calls and
a `None` semantic summary. Model-accepted terminal results always carry a
non-empty string summary, so the bridge discriminates the two shapes: a
contentless accepted skip contributes no appraisal row and records no failure;
unaccepted or missing results keep their typed failure recording.

Reachability of the skip is pinned by the evidence contract: every source kind
carries `q:goal_threat_outcome` in its fixed visibility set, so any non-empty
evidence payload plans the outcome question and the terminal stage makes its
model call. The deterministic skip therefore materializes only on zero-
question (evidence-free) payloads; such a run still executes the required
ordinary goal chain, whose bounded exhaustion without a complete sibling bid
escalates `CognitionExecutionError` before any state commit rather than
committing an empty-matter turn.

## Per-trace replay capture

Protected chain trace records flow into a `ContextVar` scope rather than a
global list. `bind_protected_chain_records()` returns a token for a fresh
record list, `snapshot_protected_chain_records()` reads the active scope as an
immutable tuple (empty when unbound), and `reset_protected_chain_records(token)`
restores the previous slot. `run_cognition` binds a records token when none is
active and resets it in its finally block; nested scopes isolate their own
record sets, producers outside any bound scope append nothing, and replay
harnesses bind one record scope per trace to read the exact stage attempts of
that run.

## Failure modes and diagnostics

Stage failures fail closed per chain with a typed error code. The closed
failure contract is enforced by `validate_stage_result`: it rejects unknown
owner identities and failure classes, requires a failure record exactly when a
result was not accepted, and accepts only the three-code
`EXHAUSTION_ERROR_CODES` set for exhaustion-class failures. Required-branch escalation, partial failure
surfacing, and protected replay capture reuse the V2 behavior verbatim; one
chain's bounded exhaustion never cancels the other chains of the same wave.

Accepted appraisal content bridges into native state through code-owned rows:
propositions attach to their source evidence row's candidate event root so the
native materializer resolves them through causal-event provenance, except
terminal outcome propositions whose subject binds to the unique lifecycle-
eligible entity of the asserted kind. Axis deltas translate into exact native
increment rows only when their axis matches exactly one permitted concrete path
for the stage's authorized evidence domain; every unbound or ambiguous delta is
dropped with a deterministic warning instead of reaching the native reducer.

`diagnostics.py` projects protected chain metadata: `project_config_identity`
keeps route identity and generation settings without reading credential
attributes, `build_chain_trace_record` assembles one bounded attempt record,
and `project_protected_chain_failure` crosses only closed typed fields —
chain and stage identity, the bounded failure class, the exact error code, and
the repair disposition. Raw candidate text, validator prose, provider exception
messages, and provider metadata never appear in these projections.

## V2 parity map

- Action planning, action authorization, resolver authorization, and workspace
  collapse prompts are byte-identical copies of their V2 constants. The
  appraisal and goal-cognition static system prompts are V3-native rewrites
  bound to the V3 candidate schemas (chain-level candidates; one consolidated
  bid-and-selection output contract), carrying over the semantic instructions
  of their V2 stage counterparts.
- Authorizers stay V3-owned on the unchanged V2 authorizer contracts.
- Role binding (D3): the producer contract carries no role data, so action-row
  role assignments stay empty in V3 output where V2 materializes them from
  native state. This is a documented parity limitation of the producer
  boundary, not an accident of the bridge.
- Prompt-budget checkpoint: V2 deterministically fits evidence texts to each
  stage's prompt budget before model calls (`fit_evidence_texts_to_budget` in
  goal-bid producers and appraisal). V3 enforces budgets at the transcript
  extension boundary instead — a deterministic UTF-8 byte check plus restart —
  and has no pre-call text-fitting checkpoint.

## Testing

Per-module unit tests live under `tests/unit/cognition_core_v3/`
(`test_action_selection`, `test_appraisal`, `test_contracts`,
`test_diagnostics`, `test_execution`, `test_facade`, `test_goal_cognition`,
`test_registry`, `test_transcript`, `test_workspace`, `test_public_api`). The
public-API test pins the exact eleven-name export surface and every export's
owning-module identity. The facade unit suite runs the full engine over the
canonical fixture with a scripted invoker: wave-completion diagnostics, the
exact registry-ordered stage-call sequence, per-trace protected record scoping,
state-carrier and relationship-maintenance application, admitted-bid passthrough
with goal-resolution fallback, authoritative relational override collapse, the
evidence-free run's bounded required-branch escalation, and rejected-attempt
capture before branch escalation. Integration parity tests under
`tests/integration/cognition_core_v3/` exercise the full facade with a scripted
LLM keyed by `config.stage_name` over the canonical connector-built input,
asserting V2-shaped output contracts, deterministic stage order across runs,
carrier preservation, terminal-skip semantics at the producer level, and
per-trace record scoping. Engine-selector tests pin the closed v2/v3 selection
set, the shared connector binding identity, reload-based resolution of exactly
the configured engine entrypoint, and the unknown-engine error naming.
