# DSH Standard Resolution Sidecar

This separately built, long-lived Node process hosts the official DSH base and
Standard preset for Kazusa Plan 2. The official composition is mounted by
reference, not copied or forked. Standard native filesystem, shell, coding,
jobs, tests, public web, approval, and sandbox capabilities take name
precedence. Kazusa supplies thirteen storage-independent semantic gateway
tools plus the controller-owned `submit_resolution`; it supplies no coding or
web wrapper, command filter, DSH budget, sandbox overlay, or generic workflow
tool.

The exact semantic names are listed in the [tool gateway README](../../src/kazusa_ai_chatbot/dsh_tool_gateway/README.md).
The Python worker owns framed service calls, idempotent semantic mutations,
and outcome replay. DSH session data and semantic outcomes are separate
SQLite stores:

- `<KAZUSA_DSH_DATA_ROOT>/dsh/0.1.1-rc.2/sessions.sqlite`
- `<KAZUSA_DSH_DATA_ROOT>/dsh/0.1.1-rc.2/semantic-outcomes.sqlite`

## Build and run

```powershell
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution install --frozen-lockfile
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution typecheck
corepack pnpm@11.7.0 --dir sidecars/dsh_resolution build
node sidecars/dsh_resolution/dist/src/main.js
```

The layered repository `.env` supplies `KAZUSA_DSH_SIDECAR_URL`,
`KAZUSA_DSH_RPC_TOKEN`, absolute `KAZUSA_DSH_DATA_ROOT`, absolute
`AGENTIC_RESOLVER_WORKSPACE_ROOT`, loopback `KAZUSA_DSH_BRAIN_URL`,
`KAZUSA_DSH_BRAIN_SHARED_SECRET`, `KAZUSA_DSH_TOOL_GATEWAY_SECRET`, and
absolute `KAZUSA_DSH_PYTHON_EXECUTABLE`. It also supplies all six
`AGENTIC_RESOLVER_LLM_*` route fields. The initial route is qwen27b-5090 with
50,176 context tokens, 8,192 completion tokens, and thinking enabled.
`DEEPSEEK_API_KEY` is optional host-only configuration for the pinned native
DSH web provider. Secret values belong in the operator environment, never in
this README.

## Readiness and lifecycle

Start the Brain first and wait for its `/health` and authenticated
`/runtime/dsh/health` endpoints. The latter must report configured,
durable-store, and cognition-judge readiness. Then start this sidecar and
call authenticated JSON-RPC `system.health`; its status is `ready` only when
`route`, `standard`, `semantic_worker`, `web`, and `brain` are ready. The
response contains sanitized route, catalog, policy, and workspace
diagnostics. Brain and sidecar have independent process lifecycles, but the
sidecar readiness dependency prevents DSH interaction requests from running
without the Brain judge.

The pinned wire contract is `kazusa.dsh-resolution-rpc.v2`, intake is
`dsh_resolution_intake.v2`, profile is `kazusa-resolver-standard-v2`, and the
store epoch is `dsh-sqlite-0.1.1-rc.2-standard-v2` under policy
`dsh-standard-policy-v2` for DSH `0.1.1-rc.2`.
See the [integration architecture](../../docs/architecture/dsh_integration_architecture.md),
[control-plane README](../../src/agentic_resolver/README.md), and
[HOWTO](../../docs/HOWTO.md#run-the-plan-2-dsh-standard-sidecar) for the full
runtime and interaction contract.
