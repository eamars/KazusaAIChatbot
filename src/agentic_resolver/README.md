# Standalone DSH Resolution Interface

`agentic_resolver` is the Python control plane for the independent
`sidecars/dsh_resolution` process. Python owns strict Kazusa RPC DTOs,
semantic operation identity, thread metadata, compatibility rotation, and
activation/lease fencing. The Node sidecar exclusively owns DSH sessions,
events, tool execution, checkpoints, and terminal receipts.

The package has no Brain, cognition, task-resolution, RAG, or coding-agent
registration edge. Callers construct `AgenticResolverRuntime` directly:

```python
from agentic_resolver import AgenticResolverRuntime

runtime = AgenticResolverRuntime.from_environment()
exhaust = await runtime.resolve(intake)
```

The runtime reads `KAZUSA_DSH_SIDECAR_URL` and `KAZUSA_DSH_RPC_TOKEN`. The
sidecar additionally requires `KAZUSA_DSH_DATA_ROOT` and `KAZUSA_DSH_MODEL`.
RPC is loopback-only, bearer-authenticated JSON-RPC at `/rpc`, protocol
`kazusa.dsh-resolution-rpc.v1`.

Supported lifecycle methods are `resolution.open`, `resolution.continue`,
`resolution.amend`, `resolution.request_checkpoint`, `resolution.cancel`,
`resolution.inspect`, and `resolution.dispose_activation`. Every mutation is
fenced by semantic operation id/digest plus activation id and monotonic lease
epoch. Segment compatibility covers scope, audience, profile, DSH release,
store epoch, model, catalog, and policy fingerprints.

Model input never contains runtime authority. Durable authority is limited to
bounded `tool/result.meta.kazusa` evidence and terminal receipts. A complete
validated `submit_resolution` receipt, flushed before the RPC response, is
the only terminal source. The production semantic catalog is empty.

Production uses profile `kazusa-resolver-v1`, DSH `0.1.1-rc.2`, and store
epoch `dsh-sqlite-0.1.1-rc.2-v1` at
`<data-root>/dsh/0.1.1-rc.2/sessions.sqlite`.
