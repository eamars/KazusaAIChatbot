# Brain–DSH Interaction Boundary

`dsh_interaction` is the Brain-owned boundary for the
`dsh_brain_interaction.v2` approval, question, and plan-review contract. It
authenticates and validates signed requests, records the durable Mongo audit
row, binds exact thread/segment/activation/lease lineage, and sends one typed
character judgment back to the DSH runtime. DSH session identities remain
runtime identities and never become platform identities.

## Runtime boundary

The authenticated runtime surface has one interaction POST route:

- `GET /runtime/dsh/health` — configured, durable-store, and cognition-judge
  readiness;
- `POST /runtime/dsh/interactions` — admit one signed approval, question, or
  plan-review request.

The Brain is ready only when the interaction owner has its durable store and
cognition judge. The sidecar separately reports `brain` readiness in its
authenticated `system.health` response. These are independent process
lifecycles with an explicit readiness dependency.

Requests carry the exact interaction identity, request digest,
platform/channel scope, resolution thread and segment, activation and lease,
expiry, policy, nonce, and signed authority. The owner rejects mismatches,
replay conflicts, expired requests, and unauthorized scope before cognition.
The model-hidden authority fields remain outside semantic judgment.

## Character-owned cognition

Every approval, question, and plan-review request enters the existing full
reusable cognition loop, including ordinary character context, resolver
recurrence, and the final cognition commit. The P-stage receives complete
bounded semantic context for the DSH task and exact interaction semantics; it
owns the character's judgment. The internal episode advertises only the
`self_goal_resolution` resolver capability.

The exact internal decision sets are:

- `question`: `answer` or `reject`;
- `approval`: `allow_once` or `reject`;
- `plan_review`: `answer`, `allow_once`, or `reject`.

## Internal handoff and audit

The validated decision returns directly to the waiting DSH hook. Deterministic
Brain code owns strict shape and kind validation, authentication, digest and
nonce checks, time and scope checks, idempotent audit/replay persistence,
bounds, and fail-closed errors. `allow_once` creates and consumes one exact
operation-bound one-shot grant; no semantic classifier or post-cognition
override changes the character decision.

This boundary creates no dialog, L3, or adapter surface and no user
prompt/reply or waiting-state lifecycle. DSH semantic detail remains a
complete bounded internal handoff and is not promoted into user-authored
evidence.

See the [Brain Service ICD](../brain_service/README.md), [Cognition Core V3
README](../cognition_core_v3/README.md), [semantic gateway
README](../dsh_tool_gateway/README.md), and [DSH integration
architecture](../../../docs/architecture/dsh_integration_architecture.md).
