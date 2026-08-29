# DSH V2 Resolution Control Plane

`agentic_resolver` is the Python control plane for the Plan 2 DSH Standard
runtime. It validates the canonical intake, operation identity, resolution
thread and segment lineage, activation/lease fencing, runtime authority, and
replay semantics. The Node sidecar owns official DSH sessions, event
persistence, native tool execution, checkpoints, and terminal receipts. The
Brain owns the interaction judge and user-facing delivery; see the [DSH
interaction README](../kazusa_ai_chatbot/dsh_interaction/README.md).

Callers construct the runtime through the canonical environment boundary:

```python
from agentic_resolver import AgenticResolverRuntime

runtime = AgenticResolverRuntime.from_environment()
exhaust = await runtime.resolve(intake)
```

The runtime requires `KAZUSA_DSH_SIDECAR_URL`, `KAZUSA_DSH_RPC_TOKEN`, an
absolute `KAZUSA_DSH_DATA_ROOT`, an absolute
`AGENTIC_RESOLVER_WORKSPACE_ROOT`, `KAZUSA_DSH_TOOL_GATEWAY_SECRET`, and an
absolute `KAZUSA_DSH_PYTHON_EXECUTABLE`. The six route settings are
`AGENTIC_RESOLVER_LLM_BASE_URL`, `AGENTIC_RESOLVER_LLM_API_KEY`,
`AGENTIC_RESOLVER_LLM_MODEL`, `AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS`,
`AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS`, and
`AGENTIC_RESOLVER_LLM_THINKING_ENABLED`. The initial documented route is
`qwen27b-5090` with a 50,176-token context, 8,192-token completion cap, and
thinking enabled. Brain connection and shared-secret settings are owned by
the sidecar process: `KAZUSA_DSH_BRAIN_URL` must be loopback and
`KAZUSA_DSH_BRAIN_SHARED_SECRET` must be configured.

The authenticated JSON-RPC protocol is `kazusa.dsh-resolution-rpc.v2` and
the intake schema is `dsh_resolution_intake.v2`. The pinned profile is
`kazusa-resolver-standard-v2`, DSH release `0.1.1-rc.2`, and session-store
epoch `dsh-sqlite-0.1.1-rc.2-standard-v2`. The sidecar stores DSH sessions at
`<KAZUSA_DSH_DATA_ROOT>/dsh/0.1.1-rc.2/sessions.sqlite`; replay-safe semantic
outcomes use the adjacent `semantic-outcomes.sqlite`.

Supported lifecycle operations are `resolution.open`,
`resolution.continue`, `resolution.amend`, `resolution.request_checkpoint`,
`resolution.cancel`, `resolution.inspect`, and
`resolution.dispose_activation`. Every mutation is fenced by semantic
operation id/digest, activation identity, and lease epoch. Scope, audience,
profile, release, store, model route, catalog, policy, workspace, and expiry
lineage are validated before a segment is reused.

The model receives bounded semantic objectives and evidence, never runtime
authority or storage credentials. Native DSH tools and Kazusa's semantic
gateway return semantic entities, opaque references, and evidence receipts.
Only a committed, evidence-bound `submit_resolution` receipt is terminal;
checkpoint, restart, transport loss, and replay are reconciled through the
durable event boundary.

The exact thirteen semantic gateway names and worker ownership are documented
in the [DSH tool gateway README](../kazusa_ai_chatbot/dsh_tool_gateway/README.md).
Operator startup and readiness are in the [HOWTO](../../docs/HOWTO.md#run-the-plan-2-dsh-standard-sidecar)
and [integration architecture](../../docs/architecture/dsh_integration_architecture.md).
