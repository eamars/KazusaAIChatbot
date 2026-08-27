import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

const TOKEN = "vitest-sidecar-token";
const PROJECT_ROOT = resolve(import.meta.dirname, "../../..");
const SIDECAR_ENTRY = resolve(import.meta.dirname, "../dist/src/main.js");
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
}

const running: RunningSidecar[] = [];
const dataRoots: string[] = [];

async function availablePort(): Promise<number> {
  const server = createServer();
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
        protocol_version: "kazusa.dsh-resolution-rpc.v1",
        ...params,
      },
    }),
  });
  if (!response.ok) {
    throw new Error(`rpc ${method} failed with ${response.status}`);
  }
  return await response.json() as Record<string, unknown>;
}

function intake(
  operationId: string,
  threadId: string,
  segmentId: string,
  mode: "start" | "continue" = "start",
): Record<string, unknown> {
  return {
    schema_version: "dsh_resolution_intake.v1",
    mode,
    runtime: {
      request_id: `req-${operationId}`,
      operation_id: operationId,
      operation_payload_digest: `sha256:${operationId}`,
      resolution_thread_id: threadId,
      segment_id: segmentId,
      priority: "now",
      soft_deadline_at: "2026-08-28T00:00:10Z",
      hard_deadline_at: "2026-08-28T00:00:30Z",
      max_model_steps: 3,
      max_tool_calls: 3,
      max_tool_bytes: 4096,
      capability_token: "opaque",
      scope_fingerprint: "sha256:scope",
      audience_fingerprint: "sha256:audience",
      resolver_profile_version: "kazusa-resolver-v1",
      dsh_release: "0.1.1-rc.2",
      session_store_epoch: "dsh-sqlite-0.1.1-rc.2-v1",
      model_route: "test-model",
      tool_catalog_digest: "sha256:catalog",
      policy_epoch: "test-1",
    },
    model_input: {
      objective: "finish",
      constraints: [],
      success_criteria: [],
      known_facts: [],
      uncertainty: [],
      literal_inputs: [],
      continuation_delta: null,
      prior_resolution_refs: [],
      requested_evidence_quality: "normal",
      notes: [],
    },
  };
}

async function start(
  dataRoot: string,
  script: Record<string, unknown>[] = [
    { name: "submit_resolution", arguments: terminal },
  ],
  extraEnvironment: Record<string, string> = {},
): Promise<RunningSidecar> {
  const port = await availablePort();
  const url = `http://127.0.0.1:${port}/rpc`;
  const child = spawn(process.execPath, [SIDECAR_ENTRY], {
    cwd: PROJECT_ROOT,
    env: {
      ...process.env,
      KAZUSA_DSH_SIDECAR_URL: url,
      KAZUSA_DSH_RPC_TOKEN: TOKEN,
      KAZUSA_DSH_DATA_ROOT: dataRoot,
      KAZUSA_DSH_MODEL: "test-model",
      KAZUSA_DSH_TEST_MODEL_SCRIPT: JSON.stringify(script),
      NODE_ENV: "test",
      ...extraEnvironment,
    },
    stdio: "pipe",
  });
  const stderr: string[] = [];
  child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk.toString("utf8")));
  const sidecar = { process: child, url, stderr };
  running.push(sidecar);
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`sidecar exited before health: ${stderr.join("")}`);
    }
    try {
      await rpc(url, "system.health", {});
      return sidecar;
    } catch {
      await new Promise((accept) => setTimeout(accept, 25));
    }
  }
  throw new Error(`sidecar health timed out: ${stderr.join("")}`);
}

async function stop(sidecar: RunningSidecar): Promise<void> {
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
  return rpc(sidecar.url, "resolution.open", {
    operation_id: operationId,
    operation_payload_digest: `sha256:${operationId}`,
    activation_id: `act-${operationId}`,
    lease_epoch: 1,
    intake: intake(operationId, threadId, segmentId),
  });
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
  });

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
      activation_id: "act-op-1",
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
      activation_id: "act-op-2",
      lease_epoch: 2,
      intake: intake("op-2", "thread-1", "segment-1", "continue"),
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
