import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdir, open } from "node:fs/promises";
import { isAbsolute, join, resolve } from "node:path";

import { createBrainInteractionProvider } from "./brain_interaction.js";
import { DSH_RELEASE, PROFILE_VERSION, RPC_PROTOCOL_VERSION, SESSION_STORE_EPOCH } from "./contracts.js";
import { routeDigest, type QwenRouteConfig } from "./model_route.js";
import { buildProfile } from "./profile.js";
import { createRpcServer } from "./rpc.js";
import { ResolutionSidecarRuntime, type RuntimeResult } from "./runtime.js";

const MAX_WORKER_FRAME_BYTES = 32 * 1024;
const WORKER_MODULE = "kazusa_ai_chatbot.dsh_tool_gateway.worker";
const WORKER_HEALTH_CONTROL = "health";
const WORKER_HEALTH_SCHEMA_VERSION = "kazusa_semantic_worker_health.v1";
const APP_BOOT_MODULE = "@deepseek-ai/dsh-app-boot";
const POLICY_EPOCH = "dsh-standard-policy-v2";

const NATIVE_ENVIRONMENT_NAMES = [
  "PATH",
  "SystemRoot",
  "ComSpec",
  "TEMP",
  "TMP",
  "USERPROFILE",
  "LOCALAPPDATA",
  "PYTHONPATH",
  "DSH_HOME",
  "DSH_PERMISSION_MODE",
  "DSH_TELEMETRY_MODE",
  "DSH_TELEMETRY_DISABLED",
  "DSH_WEB_URL",
  "DSH_WEB_MODE",
] as const;

const PROTECTED_ENVIRONMENT_NAMES = [
  "AGENTIC_RESOLVER_LLM_API_KEY",
  "KAZUSA_DSH_RPC_TOKEN",
  "KAZUSA_DSH_BRAIN_SHARED_SECRET",
  "KAZUSA_DSH_TOOL_GATEWAY_SECRET",
  "KAZUSA_DSH_CAPABILITY_TOKEN",
  "DEEPSEEK_API_KEY",
] as const;

interface LaunchEnvironmentSnapshot {
  get(name: string): { value: string } | undefined;
}

interface WorkerPending {
  resolve: (value: Record<string, unknown>) => void;
  reject: (error: Error) => void;
  kind: "semantic" | "health";
}

interface LayeredEnvironmentLoader {
  loadLayeredEnv(binName: string, cwd: string): LaunchEnvironmentSnapshot;
}

function safeWorkerUnavailable(): Record<string, unknown> {
  return {
    schema_version: "kazusa_semantic_capability_result.v1",
    status: "unavailable",
    entities: [],
    page: { has_more: false, next_page_ref: null },
    evidence: [],
    mutation: null,
    error: {
      code: "SEMANTIC_WORKER_UNAVAILABLE",
      safe_message: "The semantic worker is unavailable.",
    },
  };
}

/** Persistent length-prefixed bridge to the project semantic worker. */
class SemanticWorkerClient {
  private child: ChildProcessWithoutNullStreams | undefined;
  private input = Buffer.alloc(0);
  private pending: WorkerPending | undefined;

  constructor(
    private readonly executable: string,
    private readonly cwd: string,
    private readonly environment: Record<string, string>,
  ) {}

  async invoke(frame: Record<string, unknown>): Promise<Record<string, unknown>> {
    const callId = frame.call_id;
    if (typeof callId !== "string" || callId.length === 0) return safeWorkerUnavailable();
    if (this.pending !== undefined) return safeWorkerUnavailable();
    const payload = Buffer.from(JSON.stringify({ call_id: callId, payload: frame }), "utf8");
    if (payload.length > MAX_WORKER_FRAME_BYTES) {
      return {
        ...safeWorkerUnavailable(),
        error: { code: "SEMANTIC_FRAME_TOO_LARGE", safe_message: "The semantic request is too large." },
      };
    }
    try {
      const child = this.ensureChild();
      const header = Buffer.alloc(4);
      header.writeUInt32BE(payload.length, 0);
      const result = new Promise<Record<string, unknown>>((resolveResult, rejectResult) => {
        this.pending = { resolve: resolveResult, reject: rejectResult, kind: "semantic" };
      });
      child.stdin.write(Buffer.concat([header, payload]));
      return await result;
    } catch {
      this.pending = undefined;
      return safeWorkerUnavailable();
    }
  }

  async probe(timeoutMs = 1_000): Promise<boolean> {
    if (this.pending !== undefined) return false;
    const requestId = `worker-health-${Date.now()}-${Math.random()}`;
    const payload = Buffer.from(JSON.stringify({
      control: WORKER_HEALTH_CONTROL,
      request_id: requestId,
    }), "utf8");
    if (payload.length > MAX_WORKER_FRAME_BYTES) return false;
    let child: ChildProcessWithoutNullStreams;
    try {
      child = this.ensureChild();
    } catch {
      return false;
    }
    const header = Buffer.alloc(4);
    header.writeUInt32BE(payload.length, 0);
    return await new Promise<boolean>((resolveResult) => {
      let settled = false;
      const finish = (ready: boolean): void => {
        if (settled) return;
        settled = true;
        resolveResult(ready);
      };
      const timeout = setTimeout(() => {
        if (this.pending?.kind === "health") this.pending = undefined;
        if (this.child === child) {
          this.child = undefined;
          this.input = Buffer.alloc(0);
          child.kill();
        }
        finish(false);
      }, timeoutMs);
      this.pending = {
        kind: "health",
        reject: () => {
          clearTimeout(timeout);
          finish(false);
        },
        resolve: (value) => {
          clearTimeout(timeout);
          const expectedKeys = ["control", "protocol", "request_id", "schema_version", "status"];
          const ready = JSON.stringify(Object.keys(value).sort()) === JSON.stringify(expectedKeys)
            && value.control === WORKER_HEALTH_CONTROL
            && value.request_id === requestId
            && value.schema_version === WORKER_HEALTH_SCHEMA_VERSION
            && value.status === "ready"
            && value.protocol === "length-prefixed-json";
          finish(ready);
        },
      };
      try {
        child.stdin.write(Buffer.concat([header, payload]));
      } catch {
        clearTimeout(timeout);
        if (this.pending?.kind === "health") this.pending = undefined;
        finish(false);
      }
    });
  }

  async dispose(): Promise<void> {
    const child = this.child;
    this.child = undefined;
    this.input = Buffer.alloc(0);
    const pending = this.pending;
    this.pending = undefined;
    pending?.resolve(safeWorkerUnavailable());
    if (child === undefined) return;
    child.kill();
  }

  private ensureChild(): ChildProcessWithoutNullStreams {
    if (this.child !== undefined && this.child.exitCode === null && this.child.signalCode === null) {
      return this.child;
    }
    const child = spawn(this.executable, ["-u", "-m", WORKER_MODULE], {
      cwd: this.cwd,
      env: { ...this.environment, PYTHONUNBUFFERED: "1" },
      stdio: "pipe",
      windowsHide: true,
    });
    this.child = child;
    child.stdout.on("data", (chunk: Buffer) => this.consume(chunk));
    child.stderr.resume();
    child.once("error", (error) => this.fail(error instanceof Error ? error : new Error("semantic worker failed")));
    child.once("exit", () => {
      if (this.child === child) this.child = undefined;
      this.fail(new Error("semantic worker exited"));
    });
    return child;
  }

  private consume(chunk: Buffer): void {
    this.input = Buffer.concat([this.input, chunk]);
    while (this.input.length >= 4) {
      const length = this.input.readUInt32BE(0);
      if (length <= 0 || length > MAX_WORKER_FRAME_BYTES) {
        this.fail(new Error("semantic worker frame length is invalid"));
        return;
      }
      if (this.input.length < length + 4) return;
      const bytes = this.input.subarray(4, length + 4);
      this.input = this.input.subarray(length + 4);
      try {
        const value: unknown = JSON.parse(bytes.toString("utf8"));
        if (value === null || typeof value !== "object" || Array.isArray(value)) {
          throw new Error("semantic worker result is invalid");
        }
        const pending = this.pending;
        this.pending = undefined;
        pending?.resolve(value as Record<string, unknown>);
      } catch (error) {
        this.fail(error instanceof Error ? error : new Error("semantic worker result is invalid"));
        return;
      }
    }
  }

  private fail(error: Error): void {
    const pending = this.pending;
    this.pending = undefined;
    pending?.resolve({
      ...safeWorkerUnavailable(),
      error: { code: "SEMANTIC_WORKER_UNAVAILABLE", safe_message: "The semantic worker is unavailable." },
    });
    void error;
  }
}

function required(snapshot: LaunchEnvironmentSnapshot, name: string): string {
  const value = snapshot.get(name)?.value;
  if (value === undefined || value.length === 0) throw new Error(`${name} is required`);
  return value;
}

function positiveInteger(snapshot: LaunchEnvironmentSnapshot, name: string, minimum: number): number {
  const value = Number.parseInt(required(snapshot, name), 10);
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`${name} is invalid`);
  return value;
}

function requiredAbsolute(snapshot: LaunchEnvironmentSnapshot, name: string): string {
  const value = required(snapshot, name);
  if (!isAbsolute(value)) throw new Error(`${name} must be an absolute path`);
  return value;
}

function loopbackUrl(snapshot: LaunchEnvironmentSnapshot, name: string, pathname?: string): URL {
  const value = required(snapshot, name);
  const parsed = new URL(value);
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(`${name} must be an HTTP URL`);
  }
  if (parsed.hostname !== "127.0.0.1" && parsed.hostname !== "[::1]") {
    throw new Error(`${name} must use a loopback host`);
  }
  if (pathname !== undefined && parsed.pathname !== pathname) {
    throw new Error(`${name} must use ${pathname}`);
  }
  return parsed;
}

function optional(snapshot: LaunchEnvironmentSnapshot, name: string): string | undefined {
  const value = snapshot.get(name)?.value;
  return value === undefined || value.length === 0 ? undefined : value;
}

async function postJson(endpoint: URL, secret: string, value: Record<string, unknown>): Promise<unknown> {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${secret}`,
    },
    body: JSON.stringify(value),
  });
  if (!response.ok) throw new Error("Brain bridge request failed");
  return await response.json();
}

const repositoryRoot = resolve(import.meta.dirname, "..", "..", "..", "..");
const { loadLayeredEnv } = await import(APP_BOOT_MODULE) as unknown as LayeredEnvironmentLoader;
const launchEnvironment = loadLayeredEnv("dsh-resolution", repositoryRoot);
const sidecarUrl = loopbackUrl(launchEnvironment, "KAZUSA_DSH_SIDECAR_URL", "/rpc");
const rpcToken = required(launchEnvironment, "KAZUSA_DSH_RPC_TOKEN");
const dataRoot = requiredAbsolute(launchEnvironment, "KAZUSA_DSH_DATA_ROOT");
const workspaceRoot = requiredAbsolute(launchEnvironment, "AGENTIC_RESOLVER_WORKSPACE_ROOT");
const brainUrl = loopbackUrl(launchEnvironment, "KAZUSA_DSH_BRAIN_URL");
const brainSecret = required(launchEnvironment, "KAZUSA_DSH_BRAIN_SHARED_SECRET");
const semanticSecret = required(launchEnvironment, "KAZUSA_DSH_TOOL_GATEWAY_SECRET");
const pythonExecutable = requiredAbsolute(launchEnvironment, "KAZUSA_DSH_PYTHON_EXECUTABLE");
const route: QwenRouteConfig = {
  routeName: "kazusa-agentic-resolver",
  baseUrl: required(launchEnvironment, "AGENTIC_RESOLVER_LLM_BASE_URL"),
  credentialRef: "AGENTIC_RESOLVER_LLM_API_KEY",
  model: required(launchEnvironment, "AGENTIC_RESOLVER_LLM_MODEL"),
  contextWindowTokens: positiveInteger(launchEnvironment, "AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS", 50_000),
  maxCompletionTokens: positiveInteger(launchEnvironment, "AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS", 8_192),
  thinkingEnabled: required(launchEnvironment, "AGENTIC_RESOLVER_LLM_THINKING_ENABLED") === "true",
};
if (optional(launchEnvironment, "AGENTIC_RESOLVER_LLM_THINKING_ENABLED") !== "true"
  && optional(launchEnvironment, "AGENTIC_RESOLVER_LLM_THINKING_ENABLED") !== "false") {
  throw new Error("AGENTIC_RESOLVER_LLM_THINKING_ENABLED must be true or false");
}
const modelSecret = required(launchEnvironment, "AGENTIC_RESOLVER_LLM_API_KEY");
const hostSecrets: Record<string, string> = {
  AGENTIC_RESOLVER_LLM_API_KEY: modelSecret,
  KAZUSA_DSH_BRAIN_SHARED_SECRET: brainSecret,
  KAZUSA_DSH_TOOL_GATEWAY_SECRET: semanticSecret,
};
const webSecret = optional(launchEnvironment, "DEEPSEEK_API_KEY");
if (webSecret !== undefined) hostSecrets.DEEPSEEK_API_KEY = webSecret;

const nativeEnvironment: Record<string, string> = {};
for (const name of NATIVE_ENVIRONMENT_NAMES) {
  const value = optional(launchEnvironment, name);
  if (value !== undefined) nativeEnvironment[name] = value;
}
for (const name of PROTECTED_ENVIRONMENT_NAMES) delete process.env[name];

const storePath = join(dataRoot, "dsh", DSH_RELEASE, "sessions.sqlite");
const semanticOutcomePath = join(dataRoot, "dsh", DSH_RELEASE, "semantic-outcomes.sqlite");
await mkdir(resolve(storePath, ".."), { recursive: true });
await (await open(storePath, "a")).close();
await (await open(semanticOutcomePath, "a")).close();

const worker = new SemanticWorkerClient(
  pythonExecutable,
  workspaceRoot,
  {
    ...nativeEnvironment,
    KAZUSA_DSH_TOOL_GATEWAY_SECRET: semanticSecret,
    KAZUSA_DSH_SEMANTIC_OUTCOME_PATH: semanticOutcomePath,
  },
);
const brainProvider = createBrainInteractionProvider({
  secret: brainSecret,
  request: async (request) => await postJson(new URL("runtime/dsh/interactions", brainUrl), brainSecret, request),
});
type BrainReadiness = "ready" | "unavailable";
let brainReadiness: BrainReadiness = "unavailable";

async function probeBrainHealth(): Promise<BrainReadiness> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 1_000);
  try {
    const response = await fetch(new URL("runtime/dsh/health", brainUrl), {
      method: "GET",
      headers: { authorization: `Bearer ${brainSecret}` },
      signal: controller.signal,
    });
    if (!response.ok) return "unavailable";
    const value: unknown = await response.json();
    if (value === null || typeof value !== "object" || Array.isArray(value)) return "unavailable";
    const health = value as Record<string, unknown>;
    return health.schema_version === "dsh_brain_interaction_health.v1"
      && health.status === "ready"
      && health.configured === true
      && health.durable_store === true
      && health.cognition_judge === true
      ? "ready"
      : "unavailable";
  } catch {
    return "unavailable";
  } finally {
    clearTimeout(timeout);
  }
}

const profile = await buildProfile(PROFILE_VERSION, {
  model: route.model,
  dataRoot,
  repositoryRoot,
  workspaceRoot,
  route,
  hostSecrets,
  nativeEnvironment,
  launchEnvironment,
  brainProvider,
  semanticInvoker: async (frame) => await worker.invoke(frame),
  disposeSemanticWorker: () => worker.dispose(),
});
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

const health = async () => {
  brainReadiness = await probeBrainHealth();
  const semanticWorkerReadiness = await worker.probe();
  const readiness = {
    route: "ready",
    standard: "ready",
    semantic_worker: semanticWorkerReadiness ? "ready" : "unavailable",
    web: "ready",
    brain: brainReadiness,
  } as const;
  const status = Object.values(readiness).every((value) => value === "ready")
    ? "ready"
    : "unavailable";
  return {
  protocol_version: RPC_PROTOCOL_VERSION,
  status,
  profile: profile.id,
  profile_version: profile.profileVersion,
  dsh_release: DSH_RELEASE,
  store_epoch: SESSION_STORE_EPOCH,
  store_path: storePath.replaceAll("\\", "/"),
  loopback: true,
  dsh_runtime: true,
  route: {
    name: route.routeName,
    model: route.model,
    digest: routeDigest(route),
    credential_reference: route.credentialRef,
  },
  catalog: profile.catalog,
  policy: { epoch: POLICY_EPOCH, owner: "dsh-standard" },
  workspace: { root: workspaceRoot },
  web: { provider: "deepseek-official", credential_configured: webSecret !== undefined },
  worker: {
    status: semanticWorkerReadiness ? "ready" : "unavailable",
    protocol: "length-prefixed-json",
    executable: semanticWorkerReadiness ? "responsive" : "unavailable",
  },
  brain: { status: brainReadiness, url: brainUrl.origin },
  readiness,
  diagnostics: structuredClone(profile.diagnostics),
  };
};

const handlers = {
  "system.health": async () => health(),
  "resolution.open": async (params: Record<string, unknown>) => runtime.open(params.intake, text(params, "activation_id"), epoch(params)),
  "resolution.continue": async (params: Record<string, unknown>) => runtime.continue(params.intake, text(params, "activation_id"), epoch(params)),
  "resolution.amend": async (params: Record<string, unknown>) => runtime.amend(text(params, "resolution_thread_id"), text(params, "segment_id"), text(params, "activation_id"), epoch(params), params.amendment),
  "resolution.request_checkpoint": async (params: Record<string, unknown>) => runtime.requestCheckpoint(text(params, "resolution_thread_id"), text(params, "segment_id"), text(params, "activation_id"), epoch(params)),
  "resolution.cancel": async (params: Record<string, unknown>) => runtime.cancel(text(params, "resolution_thread_id"), text(params, "segment_id"), text(params, "activation_id"), epoch(params)),
  "resolution.inspect": async (params: Record<string, unknown>) => runtime.inspect(text(params, "operation_id"), text(params, "operation_payload_digest")),
  "resolution.dispose_activation": async (params: Record<string, unknown>) => {
    await runtime.disposeActivation(text(params, "resolution_thread_id"), text(params, "segment_id"), text(params, "activation_id"), epoch(params));
    return { disposed: true };
  },
};

const server = createRpcServer(
  sidecarUrl.hostname === "[::1]" ? "::1" : sidecarUrl.hostname,
  Number(sidecarUrl.port),
  { token: rpcToken, operations: runtime.operations, handlers },
);

let shuttingDown = false;
async function shutdown(): Promise<void> {
  if (shuttingDown) return;
  shuttingDown = true;
  server.close();
  await profile.close();
}
process.once("SIGTERM", () => { void shutdown(); });
process.once("SIGINT", () => { void shutdown(); });
