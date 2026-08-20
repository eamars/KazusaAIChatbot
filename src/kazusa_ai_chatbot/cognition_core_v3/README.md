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
The selected branch constructs only its own core services. The generic
`COGNITION_LLM` cognition route remains a shared non-core service for callers
outside the selected core branch.

## Public surface

The package exports exactly eleven names, pinned by
`tests/unit/cognition_core_v3/test_public_api.py`:

- `run_cognition(input_payload, services)` — the one public entrypoint; the
  deterministic orchestrator owns the serialized chain, stage order,
  visibility, session continuation, attempt caps, validation, and failure
  disposition.
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

One invocation owns one serialized primary chain on one primary lane. Its exact
cold sequence is `A1`, `A2`, `I1`, `G1a`, optional `G1b`, `I2`, conditional
`W1`, `P1`, off-chain `X1`/`X2`, and `O`. The immutable registry validates that
order and the configured appraisal grouping maps (`1`, `2`, `3`, or `6`); it
does not launch parallel waves or use checkpoint joins.

`A1` and `A2` are grouped appraisal steps using the canonical V2 micro-item
domains. `I1` performs one deterministic state reduction and relationship
maintenance pass. `G1a` emits the ordinary bid, `G1b` revises the frozen active
branch roster when present, `I2` applies the existing collapse rules, and `P1`
assembles the unchanged V2 output products. Accepted primary messages remain
an append-only transcript; a failed candidate is removed only by the bounded
tail-rollback repair contract.

An optional sidecar is one single-stream lane for L1 residue, JSON repair, and
authorization. It is absent or complete, has a distinct endpoint/model
identity, and cannot interleave with another sidecar request. Sidecar failure
is advisory or deny-all according to the existing V2 contract and never
changes the primary semantic owner.

Resolver recurrence reattaches to the episode session rather than rerunning
the cold anchor. Its bounded tail is observation, delta appraisal,
deterministic reduction, bid revision, `I2`, conditional `W1`, fresh `P1`,
off-chain authorization, and `O`. A terminal recurrence still executes this
short tail, and one terminal state commit occurs only after the loop.

## Cache-affine transcripts

The primary semantic chain runs one bounded cache-affine transcript under the
static V3 system prompt (`transcript.py`). It binds to the validated
cache-domain identity of the primary route and preserves the exact accepted
message prefix across steps, retries, and session reattachment. Tail rollback
removes the rejected assistant candidate before a repair request; a cold
rebuild starts from the accepted typed products and never from raw provider
history. Attempt arithmetic remains the epoch-aware V2 ledger and is not reset
by a session miss or recurrence.

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

Stage failures fail closed with a typed error code. The closed failure contract
is enforced by `validate_stage_result`: it rejects unknown owner identities and
failure classes, requires a failure record exactly when a result was not
accepted, and accepts only the three-code `EXHAUSTION_ERROR_CODES` set for
exhaustion-class failures. Required-branch escalation, partial failure
surfacing, and protected replay capture reuse the V2 behavior verbatim. The
engine-neutral resolver guardrail preserves the existing epoch arithmetic and
does not create a second attempt authority.

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
- Role binding (D3): validated required-selection nested role assignments are
  preserved through canonical V3 materialization and reach the unchanged
  dialog input.
- Context budget ownership: V2's semantic helpers remain canonical, while V3
  reserves each step's completion cap inside a normal 50,000-token total
  ceiling and may activate one conditional 65,000-token ceiling only when the
  caller-declared serving window supports it. The context-window declaration
  stays caller-local and is omitted from provider transport.

## Timing and observability

`COGNITION_V3_APPRAISAL_GROUP_COUNT` is configurable to `1`, `2`, `3`, or `6`
and defaults to `2`. `COGNITION_V3_TURN_DEADLINE_SECONDS` defaults to `240` and
is bounded to `30..600`; the deadline is checked between model steps and does
not bypass deterministic validation, reduction, or the one final commit.
Runtime telemetry records non-streaming elapsed duration in milliseconds. The
runtime does not claim or expose time-to-first-token (TTFT).

The chain's `run_id`, `llm_trace_id`, and `cognition_invocation_id` are carried
with the sanitized `cognition_chain_run.v1` row. Protected transcript capture
uses `off`, `metadata`, or `full` mode in the protected trace store; the
bounded `cognition_chain` event family accepts only sanitized aggregate fields
and best-effort writes. The brain and console read only the exact paired
correlation and report `not_reported` when it is absent or mismatched; neither
surface uses a global-latest or stale fallback.

## Testing

Per-module unit tests live under `tests/unit/cognition_core_v3/`
(`test_action_selection`, `test_appraisal`, `test_contracts`,
`test_diagnostics`, `test_execution`, `test_facade`, `test_goal_cognition`,
`test_registry`, `test_transcript`, `test_workspace`, `test_public_api`). The
public-API test pins the exact eleven-name export surface and every export's
owning-module identity. The facade unit suite runs the full engine over the
canonical fixture with a scripted invoker: serial-step diagnostics, the exact
registry-ordered stage-call sequence, per-trace protected record scoping,
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
