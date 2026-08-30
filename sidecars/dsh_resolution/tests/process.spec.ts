import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer as createHttpServer, type Server as HttpServer } from "node:http";
import { createServer as createNetServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";
import {
  activationIdFor,
  digest,
  issueActivationToken,
  scopeFingerprint,
  workspaceFingerprint,
} from "../src/contracts.js";

const TOKEN = "vitest-sidecar-token";
const PROJECT_ROOT = resolve(import.meta.dirname, "../../..");
const SIDECAR_ENTRY = resolve(import.meta.dirname, "../dist/src/main.js");
const PROJECT_PYTHON = resolve(
  PROJECT_ROOT,
  process.platform === "win32" ? "venv/Scripts/python.exe" : "venv/bin/python",
);
const terminal = {
  status: "resolved",
  summary: "done",
  findings: [],
  completed_subgoals: [],
  remaining_needs: [],
  clarification_request: null,
  approval_request: null,
  artifact_refs: [],
  warnings: [],
};

interface RunningSidecar {
  process: ChildProcessWithoutNullStreams;
  url: string;
  stderr: string[];
  modelServer: HttpServer;
  brainServer: HttpServer;
  routeDigest: string;
  catalogDigest: string;
  nativeCatalogDigest: string;
  publishedCatalogDigest: string;
}

const running: RunningSidecar[] = [];
const dataRoots: string[] = [];

async function availablePort(): Promise<number> {
  const server = createNetServer();
  await new Promise<void>((accept, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", accept);
  });
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("test port allocation failed");
  }
  await new Promise<void>((accept, reject) => {
    server.close((error) => error === undefined ? accept() : reject(error));
  });
  return address.port;
}

async function rpc(
  url: string,
  method: string,
  params: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      authorization: `Bearer ${TOKEN}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      jsonrpc: "2.0",
      id: `rpc-${Date.now()}-${Math.random()}`,
      method,
      params: {
        protocol_version: "kazusa.dsh-resolution-rpc.v2",
        ...params,
      },
    }),
  });
  if (!response.ok) {
    throw new Error(`rpc ${method} failed with ${response.status}: ${await response.text()}`);
  }
  return await response.json() as Record<string, unknown>;
}

function intake(
  operationId: string,
  threadId: string,
  segmentId: string,
  routeDigest: string,
  catalogDigest: string,
  activationId: string,
  leaseEpoch: number,
  mode: "start" | "continue" = "start",
): Record<string, unknown> {
  const serviceScope = {
    platform: "debug",
    platform_channel_id: "sidecar-process",
    global_user_id: "user",
  };
  const workspaceRoot = PROJECT_ROOT.replaceAll("\\", "/");
  const clock = Date.now();
  const issuedAt = new Date(clock - 1_000).toISOString();
  const expiresAt = new Date(clock + 299_000).toISOString();
  const authority = {
    schema_version: "kazusa_semantic_tool_authority.v1" as const,
    activation_id: activationId,
    lease_epoch: leaseEpoch,
    resolution_thread_id: threadId,
    segment_id: segmentId,
    brain_conversation_ref: "chat:debug:sidecar-process",
    service_scope: serviceScope,
    scope_fingerprint: scopeFingerprint(serviceScope),
    audience_fingerprint: "sha256:audience",
    workspace_root: workspaceRoot,
    route_digest: routeDigest,
    catalog_digest: catalogDigest,
    profile_version: "kazusa-resolver-standard-v2",
    model_route_digest: routeDigest,
    workspace_fingerprint: workspaceFingerprint(workspaceRoot),
    issued_reference_digest: digest({
      resolution_thread_id: threadId,
      segment_id: segmentId,
      operation_id: operationId,
    }),
    policy_epoch: "dsh-standard-policy-v2",
    interaction_issuer: "dsh-sidecar-test",
    issued_at: issuedAt,
    expires_at: expiresAt,
    token_id: `token-${threadId}-${segmentId}-${leaseEpoch}`,
    nonce: `nonce-${threadId}-${segmentId}-${leaseEpoch}`,
  };
  return {
    schema_version: "dsh_resolution_intake.v2",
    mode,
    request_id: `req-${operationId}`,
    operation_id: operationId,
    operation_payload_digest: `sha256:${operationId}`,
    resolution_thread_id: threadId,
    segment_id: segmentId,
    brain_conversation_ref: "chat:debug:sidecar-process",
    workspace_root: workspaceRoot,
    route_digest: routeDigest,
    model_input: { objective: "finish", facts: [] },
    semantic_tool_authority: {
      catalog_digest: catalogDigest,
      token: issueActivationToken(authority, "process-gateway-secret"),
    },
    interaction_authority: {
      issuer: "dsh-sidecar-test",
      scope_fingerprint: authority.scope_fingerprint,
      audience_fingerprint: authority.audience_fingerprint,
    },
  };
}

async function start(
  dataRoot: string,
  script: Record<string, unknown>[] = [
    { name: "submit_resolution", arguments: terminal },
  ],
  extraEnvironment: Record<string, string> = {},
  waitForHealthy = true,
): Promise<RunningSidecar> {
  const port = await availablePort();
  const url = `http://127.0.0.1:${port}/rpc`;
  let scriptIndex = 0;
  const modelServer = createHttpServer((request, response) => {
    const chunks: Buffer[] = [];
    request.on("data", (chunk: Buffer) => chunks.push(chunk));
    request.on("end", () => {
      let body: Record<string, unknown> = {};
      try {
        body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
      } catch {
        response.writeHead(400).end();
        return;
      }
      void body;
      const step = script[scriptIndex] ?? script[script.length - 1] ?? {};
      scriptIndex += 1;
      if (step.wait === true) return;
      const calls = Array.isArray(step.calls)
        ? step.calls
        : step.name === undefined ? [] : [step];
      const events: string[] = [];
      if (calls.length === 0) {
        const text = typeof step.text === "string" ? step.text : "";
        events.push(`data: ${JSON.stringify({
          id: "response-1",
          choices: [{ delta: { role: "assistant", content: text }, finish_reason: null }],
        })}\n\n`);
        events.push(`data: ${JSON.stringify({
          id: "response-1",
          choices: [{ delta: {}, finish_reason: "stop" }],
        })}\n\n`);
      } else {
        const toolCalls = calls.map((raw, index) => {
          const call = raw !== null && typeof raw === "object" ? raw as Record<string, unknown> : {};
          return {
            index,
            id: `call-${scriptIndex}-${index}`,
            type: "function",
            function: {
              name: typeof call.name === "string" ? call.name : "unknown",
              arguments: JSON.stringify(call.arguments ?? {}),
            },
          };
        });
        events.push(`data: ${JSON.stringify({
          id: "response-1",
          choices: [{ delta: { role: "assistant", tool_calls: toolCalls }, finish_reason: null }],
        })}\n\n`);
        events.push(`data: ${JSON.stringify({
          id: "response-1",
          choices: [{ delta: {}, finish_reason: "tool_calls" }],
        })}\n\n`);
      }
      events.push("data: [DONE]\n\n");
      response.writeHead(200, { "content-type": "text/event-stream" });
      response.end(events.join(""));
    });
  });
  await new Promise<void>((accept, reject) => {
    modelServer.once("error", reject);
    modelServer.listen(0, "127.0.0.1", accept);
  });
  const modelAddress = modelServer.address();
  if (modelAddress === null || typeof modelAddress === "string") throw new Error("model server did not bind");
  const brainServer = createHttpServer((request, response) => {
    if (request.method !== "GET" || request.url !== "/runtime/dsh/health") {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, { "content-type": "application/json" });
    response.end(JSON.stringify({
      schema_version: "dsh_brain_interaction_health.v1",
      status: "ready",
      configured: true,
      durable_store: true,
      cognition_judge: true,
    }));
  });
  await new Promise<void>((accept, reject) => {
    brainServer.once("error", reject);
    brainServer.listen(0, "127.0.0.1", accept);
  });
  const brainAddress = brainServer.address();
  if (brainAddress === null || typeof brainAddress === "string") throw new Error("brain server did not bind");
  const child = spawn(process.execPath, [SIDECAR_ENTRY], {
    cwd: PROJECT_ROOT,
    env: {
      ...process.env,
      KAZUSA_DSH_SIDECAR_URL: url,
      KAZUSA_DSH_RPC_TOKEN: TOKEN,
      KAZUSA_DSH_DATA_ROOT: dataRoot,
      AGENTIC_RESOLVER_LLM_BASE_URL: `http://127.0.0.1:${modelAddress.port}/v1`,
      AGENTIC_RESOLVER_LLM_API_KEY: "process-model-secret",
      AGENTIC_RESOLVER_LLM_MODEL: "qwen27b-5090",
      AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS: "50176",
      AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS: "8192",
      AGENTIC_RESOLVER_LLM_THINKING_ENABLED: "true",
      AGENTIC_RESOLVER_WORKSPACE_ROOT: PROJECT_ROOT,
      KAZUSA_DSH_BRAIN_URL: `http://127.0.0.1:${brainAddress.port}`,
      KAZUSA_DSH_BRAIN_SHARED_SECRET: "process-brain-secret",
      KAZUSA_DSH_TOOL_GATEWAY_SECRET: "process-gateway-secret",
      KAZUSA_DSH_PYTHON_EXECUTABLE: PROJECT_PYTHON,
      NODE_ENV: "test",
      ...extraEnvironment,
    },
    stdio: "pipe",
  });
  const stderr: string[] = [];
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk.toString("utf8")));
  const sidecar = {
    process: child,
    url,
    stderr,
    modelServer,
    brainServer,
    routeDigest: "",
    catalogDigest: "",
    nativeCatalogDigest: "",
    publishedCatalogDigest: "",
  };
  running.push(sidecar);
  if (!waitForHealthy) return sidecar;
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`sidecar exited before health: ${stderr.join("")}`);
    }
    try {
      const healthFrame = await rpc(url, "system.health", {});
      const health = healthFrame.result;
      if (health === null || typeof health !== "object" || Array.isArray(health)) {
        throw new Error("sidecar health result is invalid");
      }
      const route = (health as Record<string, unknown>).route;
      const catalog = (health as Record<string, unknown>).catalog;
      if (route === null || typeof route !== "object" || Array.isArray(route)) {
        throw new Error("sidecar health route is invalid");
      }
      if (catalog === null || typeof catalog !== "object" || Array.isArray(catalog)) {
        throw new Error("sidecar health catalog is invalid");
      }
      const routeDigest = (route as Record<string, unknown>).digest;
      const semanticCatalogDigest = (catalog as Record<string, unknown>).semantic_catalog_digest;
      const nativeCatalogDigest = (catalog as Record<string, unknown>).native_catalog_digest;
      const publishedCatalogDigest = (catalog as Record<string, unknown>).published_catalog_digest;
      if (
        typeof routeDigest !== "string"
        || typeof semanticCatalogDigest !== "string"
        || typeof nativeCatalogDigest !== "string"
        || typeof publishedCatalogDigest !== "string"
      ) {
        throw new Error("sidecar health digests are invalid");
      }
      sidecar.routeDigest = routeDigest;
      sidecar.catalogDigest = semanticCatalogDigest;
      sidecar.nativeCatalogDigest = nativeCatalogDigest;
      sidecar.publishedCatalogDigest = publishedCatalogDigest;
      return sidecar;
    } catch {
      await new Promise((accept) => setTimeout(accept, 25));
    }
  }
  throw new Error(`sidecar health timed out: ${stderr.join("")}`);
}

async function stop(sidecar: RunningSidecar): Promise<void> {
  await new Promise<void>((accept) => sidecar.modelServer.close(() => accept()));
  await new Promise<void>((accept) => {
    if (!sidecar.brainServer.listening) {
      accept();
      return;
    }
    sidecar.brainServer.close(() => accept());
  });
  if (
    sidecar.process.exitCode !== null
    || sidecar.process.signalCode !== null
  ) return;
  const exited = new Promise<void>((accept) => {
    sidecar.process.once("exit", () => accept());
  });
  sidecar.process.kill();
  await Promise.race([
    exited,
    new Promise<void>((accept) => setTimeout(accept, 2_000)),
  ]);
  if (
    sidecar.process.exitCode === null
    && sidecar.process.signalCode === null
  ) {
    sidecar.process.kill("SIGKILL");
    await exited;
  }
}

async function open(
  sidecar: RunningSidecar,
  operationId: string,
  threadId: string,
  segmentId: string,
): Promise<Record<string, unknown>> {
  try {
    return await rpc(sidecar.url, "resolution.open", {
      operation_id: operationId,
      operation_payload_digest: `sha256:${operationId}`,
      activation_id: activationIdFor(threadId, segmentId, 1),
      lease_epoch: 1,
      intake: intake(
        operationId,
        threadId,
        segmentId,
        sidecar.routeDigest,
        sidecar.catalogDigest,
        activationIdFor(threadId, segmentId, 1),
        1,
      ),
    });
  } catch (error) {
    throw new Error(`${error instanceof Error ? error.message : String(error)}; stderr: ${sidecar.stderr.join("")}`);
  }
}

afterEach(async () => {
  for (const sidecar of running.splice(0)) await stop(sidecar);
  for (const dataRoot of dataRoots.splice(0)) {
    await rm(dataRoot, {
      recursive: true,
      force: true,
      maxRetries: 5,
      retryDelay: 100,
    });
  }
}, 30_000);

describe("process", () => {
  it("serves one long-lived independent process across multiple sessions", async () => {
    const dataRoot = await mkdtemp(join(tmpdir(), "kazusa-dsh-process-"));
    dataRoots.push(dataRoot);
    const sidecar = await start(dataRoot);
    const first = await open(sidecar, "op-1", "thread-1", "segment-1");
    const second = await open(sidecar, "op-2", "thread-2", "segment-2");
    expect((first.result as Record<string, unknown>).disposition).toBe("terminal");
    expect((second.result as Record<string, unknown>).disposition).toBe("terminal");
    expect(sidecar.process.exitCode).toBeNull();
  }, 30_000);

  it("restarts and cold-resumes evidence from the versioned session store", async () => {
    const dataRoot = await mkdtemp(join(tmpdir(), "kazusa-dsh-resume-"));
    dataRoots.push(dataRoot);
    const first = await start(dataRoot, [{ wait: true }]);
    const pending = open(first, "op-1", "thread-1", "segment-1");
    await new Promise((accept) => setTimeout(accept, 100));
    const checkpointFrame = await rpc(first.url, "resolution.request_checkpoint", {
      operation_id: "op-checkpoint",
      operation_payload_digest: "sha256:checkpoint",
      resolution_thread_id: "thread-1",
      segment_id: "segment-1",
      activation_id: activationIdFor("thread-1", "segment-1", 1),
      lease_epoch: 1,
    });
    const checkpoint = checkpointFrame.result as Record<string, unknown>;
    expect(checkpoint.disposition).toBe("checkpointed");
    await pending;
    await stop(first);

    const second = await start(dataRoot);
    const continuation = await rpc(second.url, "resolution.continue", {
      operation_id: "op-2",
      operation_payload_digest: "sha256:op-2",
      activation_id: activationIdFor("thread-1", "segment-1", 2),
      lease_epoch: 2,
      intake: intake(
        "op-2",
        "thread-1",
        "segment-1",
        second.routeDigest,
        second.catalogDigest,
        activationIdFor("thread-1", "segment-1", 2),
        2,
        "continue",
      ),
    });
    const resumed = continuation.result as Record<string, unknown>;
    expect(resumed.disposition).toBe("terminal");
    expect(resumed.session_id).toBe(checkpoint.session_id);
  });

  it("recovers terminal receipt when killed after commit before rpc response", async () => {
    const dataRoot = await mkdtemp(join(tmpdir(), "kazusa-dsh-crash-"));
    dataRoots.push(dataRoot);
    const first = await start(dataRoot, undefined, {
      KAZUSA_DSH_TEST_EXIT_AFTER_TERMINAL_COMMIT: "1",
    });
    await expect(open(first, "op-1", "thread-1", "segment-1")).rejects.toThrow();
    await new Promise<void>((accept) => first.process.once("exit", () => accept()));

    const second = await start(dataRoot);
    const inspectionFrame = await rpc(second.url, "resolution.inspect", {
      operation_id: "op-1",
      operation_payload_digest: "sha256:op-1",
    });
    const inspection = inspectionFrame.result as Record<string, unknown>;
    expect(inspection.disposition).toBe("terminal");
    const replayFrame = await open(second, "op-1", "thread-1", "segment-1");
    const replay = replayFrame.result as Record<string, unknown>;
    expect(replay.exhaust).toEqual(inspection.exhaust);
  });

  it("reconciles admitted operation after controller restart without model re-entry", async () => {
    const dataRoot = await mkdtemp(join(tmpdir(), "kazusa-dsh-reconcile-"));
    dataRoots.push(dataRoot);
    const first = await start(dataRoot);
    const committedFrame = await open(first, "op-1", "thread-1", "segment-1");
    const committed = committedFrame.result as Record<string, unknown>;
    await stop(first);

    const second = await start(dataRoot, [{ text: "must not execute" }]);
    const replayFrame = await open(second, "op-1", "thread-1", "segment-1");
    const replay = replayFrame.result as Record<string, unknown>;
    expect(replay.exhaust).toEqual(committed.exhaust);
    const healthFrame = await rpc(second.url, "system.health", {});
    const health = healthFrame.result as Record<string, unknown>;
    const diagnostics = health.diagnostics as Record<string, unknown>;
    expect(diagnostics.terminal_tool_executions).toBe(0);
    expect(diagnostics.correction_attempts).toBe(0);
  });
});

describe("V2 process", () => {
  it("starts only after route standard worker web and brain health are ready", async () => {
    const dataRoot = await mkdtemp(join(tmpdir(), "kazusa-dsh-v2-health-"));
    dataRoots.push(dataRoot);
    const sidecar = await start(dataRoot);
    const healthFrame = await rpc(sidecar.url, "system.health", {});
    const health = healthFrame.result as Record<string, unknown>;
    expect(health.protocol_version).toBe("kazusa.dsh-resolution-rpc.v2");
    expect(health.status).toBe("ready");
    const readiness = health.readiness as Record<string, unknown>;
    expect(readiness.route).toBe("ready");
    expect(readiness.standard).toBe("ready");
    expect(readiness.semantic_worker).toBe("ready");
    expect(readiness.web).toBe("ready");
    expect(readiness.brain).toBe("ready");
    const catalog = health.catalog as Record<string, unknown>;
    const nativeNames = catalog.native_names as string[];
    expect(nativeNames).toContain("web_search");
    expect(nativeNames).toContain(process.platform === "win32" ? "pwsh" : "bash");
    expect(catalog.names).toEqual(expect.arrayContaining([
      "web_search",
      "submit_resolution",
      "kazusa_search_memories",
    ]));
    expect(catalog.native_catalog_digest).toMatch(/^sha256:/u);
    expect(catalog.semantic_catalog_digest).toBe(
      "sha256:ba49a8f6a55a8ad049a5f4026fb2db880ee0e9134b32acf7b7401fa43d36438d",
    );
    expect(catalog.published_catalog_digest).toMatch(/^sha256:/u);
    expect(catalog.omitted_semantic_tools).toEqual([]);
  });

  it("rejects a tampered activation token before Agent admission", async () => {
    const dataRoot = await mkdtemp(join(tmpdir(), "kazusa-dsh-token-reject-"));
    dataRoots.push(dataRoot);
    const sidecar = await start(dataRoot);
    const operationId = "op-token-reject";
    const threadId = "thread-token-reject";
    const segmentId = "segment-token-reject";
    const request = intake(
      operationId,
      threadId,
      segmentId,
      sidecar.routeDigest,
      sidecar.catalogDigest,
      activationIdFor(threadId, segmentId, 1),
      1,
    );
    const semantic = request.semantic_tool_authority as Record<string, unknown>;
    const token = String(semantic.token);
    semantic.token = `${token.slice(0, -1)}${token.endsWith("0") ? "1" : "0"}`;
    await expect(rpc(sidecar.url, "resolution.open", {
      operation_id: operationId,
      operation_payload_digest: `sha256:${operationId}`,
      activation_id: activationIdFor(threadId, segmentId, 1),
      lease_epoch: 1,
      intake: request,
    })).rejects.toThrow(/500/u);
  });

  it("reports unavailable worker health for a non-worker executable", async () => {
    const dataRoot = await mkdtemp(join(tmpdir(), "kazusa-dsh-worker-health-unavailable-"));
    dataRoots.push(dataRoot);
    const sidecar = await start(
      dataRoot,
      undefined,
      { KAZUSA_DSH_PYTHON_EXECUTABLE: resolve(PROJECT_ROOT, "venv", "Scripts", "missing-python.exe") },
      false,
    );
    let health: Record<string, unknown> | undefined;
    const deadline = Date.now() + 10_000;
    while (Date.now() < deadline && health === undefined) {
      try {
        const frame = await rpc(sidecar.url, "system.health", {});
        if (frame.result !== null && typeof frame.result === "object" && !Array.isArray(frame.result)) {
          health = frame.result as Record<string, unknown>;
        }
      } catch {
        await new Promise((accept) => setTimeout(accept, 25));
      }
    }
    expect(health).toBeDefined();
    expect(health?.status).toBe("unavailable");
    const readiness = health?.readiness as Record<string, unknown>;
    expect(readiness.semantic_worker).toBe("unavailable");
    const worker = health?.worker as Record<string, unknown>;
    expect(worker.status).toBe("unavailable");
  }, 30_000);
});
