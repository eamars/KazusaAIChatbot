# Brain–DSH Interaction Boundary

`dsh_interaction` is the Brain-owned bridge for DSH approval, question, and
plan-review interactions. It authenticates and validates signed requests,
keeps the durable Mongo interaction row, binds exact thread/segment/
activation/lease lineage, and sends typed judgments through canonical
cognition. It does not turn DSH detail into user-authored evidence or expose
DSH session identities as platform identities.

## Runtime boundary

The internal authenticated routes are:

- `GET /runtime/dsh/health` — configured, durable-store, and cognition-judge
  readiness;
- `POST /runtime/dsh/interactions` — admit one signed approval, question, or
  plan-review request;
- `POST /runtime/dsh/interactions/checkpoint` — replay a durable relay
  checkpoint after a sidecar transport retry.

The Brain is ready only when the interaction owner has its durable store and
cognition judge. The sidecar separately reports `brain` readiness in its
authenticated `system.health` response. These are independent process
lifecycles with an explicit readiness dependency.

Requests carry exact interaction identity, request digest, platform/channel
scope, resolution thread and segment, activation and lease, expiry, policy,
and signed authority. The owner rejects mismatches, replay conflicts, expired
requests, and unauthorized scope before cognition. The cognition P-stage owns
the semantic decision; accepted values are `answer`, `reject`, `allow_once`,
and `relay_to_user` (with `continue_waiting` only for a DSH reply). Dialog
owns visible wording and the adapter owns transport delivery.

## Approval and reply flow

An initial DSH request is represented in cognition as a targeted,
runtime-authored system observation with pending semantic interaction context;
it is not a user permission or user utterance. Brain returns an immediate
answer/rejection/one-shot result or persists a `relay_to_user` checkpoint and
delivers the visible question through normal dialog and adapter paths.

An exact user reply is matched to the pending interaction and its platform
message lineage. The same resolution thread and segment resume with a fresh
continuation activation while the original activation/lease binding remains
auditable. A relayed approval creates an available one-shot grant; a fresh
signed approval request must match the native tool, executable arguments,
workspace, scope, policy, thread, segment, and expiry before deterministic
code atomically consumes it. A new native call id is allowed when those
semantic arguments are identical. Mismatch, expiry, transport uncertainty, or
conflicting lineage remains fail-closed and does not bypass cognition.

Checkpoint, restart, replay, and continuation transport loss reconcile from
durable interaction and DSH event state. Successful continuation binds the
trusted thread, segment, activation, and lease identities into the returned
result. Raw transient tool detail remains in the pending runtime path rather
than entering user-authored evidence.

See the [Brain Service ICD](../brain_service/README.md), [Cognition Core V3
README](../cognition_core_v3/README.md), [semantic gateway
README](../dsh_tool_gateway/README.md), and [DSH integration
architecture](../../../docs/architecture/dsh_integration_architecture.md).
