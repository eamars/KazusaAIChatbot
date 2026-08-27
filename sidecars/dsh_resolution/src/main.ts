import { mkdir, open } from "node:fs/promises";
import { dirname, join } from "node:path";

import { DSH_RELEASE, PROFILE_VERSION, SESSION_STORE_EPOCH } from "./contracts.js";
import { buildProfile } from "./profile.js";
import { createRpcServer } from "./rpc.js";
import { ResolutionSidecarRuntime, type RuntimeResult } from "./runtime.js";

function required(name: string): string {
  const value = process.env[name];
  if (value === undefined || value.length === 0) throw new Error(`${name} is required`);
  return value;
}

const url = new URL(required("KAZUSA_DSH_SIDECAR_URL"));
if (url.protocol !== "http:" || url.pathname !== "/rpc" || (url.hostname !== "127.0.0.1" && url.hostname !== "[::1]")) {
  throw new Error("KAZUSA_DSH_SIDECAR_URL must be a loopback HTTP /rpc URL");
}
const token = required("KAZUSA_DSH_RPC_TOKEN");
const dataRoot = required("KAZUSA_DSH_DATA_ROOT");
const model = required("KAZUSA_DSH_MODEL");
const storePath = join(dataRoot, "dsh", DSH_RELEASE, "sessions.sqlite");
await mkdir(dirname(storePath), { recursive: true });
await (await open(storePath, "a")).close();

const scriptValue = process.env.KAZUSA_DSH_TEST_MODEL_SCRIPT;
const script = scriptValue === undefined ? [] : JSON.parse(scriptValue) as Record<string, unknown>[];
const profile = await buildProfile(PROFILE_VERSION, { model, dataRoot, ...(scriptValue === undefined ? {} : { testScript: script }) });
const runtime = ResolutionSidecarRuntime.forProduction(
  async (method, intake, activationId, leaseEpoch) => (
    await profile.activate(method, intake, activationId, leaseEpoch) as RuntimeResult
  ),
  profile,
);

function text(params: Record<string, unknown>, key: string): string {
  const value = params[key];
  if (typeof value !== "string" || value.length === 0) throw new Error(`${key} is required`);
  return value;
}
function epoch(params: Record<string, unknown>): number {
  const value = params.lease_epoch;
  if (!Number.isInteger(value) || (value as number) < 1) throw new Error("lease_epoch is invalid");
  return value as number;
}

const handlers = {
  "system.health": async () => ({ status: "ok", profile: profile.id, dsh_release: DSH_RELEASE, store_epoch: SESSION_STORE_EPOCH, store_path: storePath.replaceAll("\\", "/"), loopback: true, dsh_runtime: true, diagnostics: structuredClone(profile.diagnostics) }),
  "resolution.open": async (params: Record<string, unknown>) => runtime.open(params.intake, text(params, "activation_id"), epoch(params)),
  "resolution.continue": async (params: Record<string, unknown>) => runtime.continue(params.intake, text(params, "activation_id"), epoch(params)),
  "resolution.amend": async (params: Record<string, unknown>) => runtime.amend(text(params, "resolution_thread_id"), text(params, "segment_id"), text(params, "activation_id"), epoch(params), params.amendment),
  "resolution.request_checkpoint": async (params: Record<string, unknown>) => runtime.requestCheckpoint(text(params, "resolution_thread_id"), text(params, "segment_id"), text(params, "activation_id"), epoch(params)),
  "resolution.cancel": async (params: Record<string, unknown>) => runtime.cancel(text(params, "resolution_thread_id"), text(params, "segment_id"), text(params, "activation_id"), epoch(params)),
  "resolution.inspect": async (params: Record<string, unknown>) => runtime.inspect(
    text(params, "operation_id"),
    text(params, "operation_payload_digest"),
  ),
  "resolution.dispose_activation": async (params: Record<string, unknown>) => {
    await runtime.disposeActivation(text(params, "resolution_thread_id"), text(params, "segment_id"), text(params, "activation_id"), epoch(params));
    return { disposed: true };
  },
};

const server = createRpcServer(url.hostname === "[::1]" ? "::1" : url.hostname, Number(url.port), { token, operations: runtime.operations, handlers });
async function shutdown(): Promise<void> {
  server.close();
  await profile.close();
}
process.once("SIGTERM", () => { void shutdown(); });
process.once("SIGINT", () => { void shutdown(); });
