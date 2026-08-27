import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";

import { Context, type Fiber } from "@deepseek-ai/cordis";
import { AgentRegistry, type AgentHandle } from "@deepseek-ai/dsh-agent";
import { AgentLoop } from "@deepseek-ai/dsh-agent-loop";
import {
  BlockAssembler,
  CallId,
  createUserMessage,
  type GenerateOptions,
  LlmAdapter,
  LlmRuntime,
  type LlmModelInfo,
  type LlmProviderInfo,
  type LlmResolvedModelInfo,
  type StreamChunk,
} from "@deepseek-ai/dsh-llm";
import * as deepseekPlugin from "@deepseek-ai/dsh-llm-deepseek";
import {
  type JsonValue,
  type SessionEvent,
  SessionId,
  SessionStore,
} from "@deepseek-ai/dsh-session";
import * as checkpointPolicy from "@deepseek-ai/dsh-session-checkpoint-policy";
import { SqliteSessionPersistence } from "@deepseek-ai/dsh-session-persistence-sqlite";
import SystemPrompt from "@deepseek-ai/dsh-system-prompt";
import { defineTool, ToolRuntime } from "@deepseek-ai/dsh-tools";

import {
  DSH_RELEASE,
  type JsonObject,
  PROFILE_VERSION,
  type ResolutionIntake,
  type ResolutionRuntime,
  validateSubmitResolution,
  validateTerminalReceipt,
} from "./contracts.js";
import { digest } from "./evidence.js";
import { OperationReuseFault } from "./operations.js";
import { replayTerminalExhaust } from "./submit_resolution.js";

export const PRODUCTION_SESSION_EVENT_KINDS = ["tool/result"] as const;

const ADMISSION_PREFIX = "kazusa-operation:";
const CORRECTION_MESSAGE = (
  "The previous response violated the resolver action contract. "
  + "Call submit_resolution exactly once with a complete valid object."
);
const REQUIRED_DSH_PACKAGES = [
  "@deepseek-ai/dsh-agent",
  "@deepseek-ai/dsh-agent-loop",
  "@deepseek-ai/dsh-invariants",
  "@deepseek-ai/dsh-llm",
  "@deepseek-ai/dsh-llm-deepseek",
  "@deepseek-ai/dsh-scope",
  "@deepseek-ai/dsh-session",
  "@deepseek-ai/dsh-session-checkpoint-policy",
  "@deepseek-ai/dsh-session-persistence",
  "@deepseek-ai/dsh-session-persistence-sqlite",
  "@deepseek-ai/dsh-settings",
  "@deepseek-ai/dsh-system-prompt",
  "@deepseek-ai/dsh-tools",
] as const;

interface AdmissionIdentity {
  method: "resolution.open" | "resolution.continue";
  operationId: string;
  payloadDigest: string;
  requestId: string;
  threadId: string;
  segmentId: string;
  activationId: string;
  leaseEpoch: number;
}

interface LiveActivation {
  admission: AdmissionIdentity;
  handle: AgentHandle;
}

export interface ProfileDiagnostics extends JsonObject {
  terminal_tool_executions: number;
  correction_attempts: number;
  live_activations: number;
}

export function assertCompatibleDependencyGraph(
  versions: Readonly<Record<string, string>>,
): void {
  for (const [name, version] of Object.entries(versions)) {
    if (name.startsWith("@deepseek-ai/dsh-") && version !== DSH_RELEASE) {
      throw new Error(`incompatible DSH dependency ${name}@${version}`);
    }
  }
}

export interface ResolverProfile {
  id: typeof PROFILE_VERSION;
  model: string;
  dataRoot: string;
  semanticTools: readonly ["submit_resolution"];
  dshPackages: typeof REQUIRED_DSH_PACKAGES;
  composedServices: readonly string[];
  diagnostics: ProfileDiagnostics;
  activate(
    method: AdmissionIdentity["method"],
    intake: ResolutionIntake,
    activationId: string,
    leaseEpoch: number,
  ): Promise<JsonObject>;
  checkpoint(
    threadId: string,
    segmentId: string,
    activationId: string,
    leaseEpoch: number,
  ): Promise<JsonObject>;
  cancel(
    threadId: string,
    segmentId: string,
    activationId: string,
    leaseEpoch: number,
  ): Promise<JsonObject>;
  amend(
    threadId: string,
    segmentId: string,
    activationId: string,
    leaseEpoch: number,
    amendment: JsonObject,
  ): Promise<JsonObject>;
  disposeActivation(
    threadId: string,
    segmentId: string,
    activationId: string,
    leaseEpoch: number,
  ): Promise<void>;
  inspect(operationId: string, payloadDigest: string): Promise<JsonObject>;
  close(): Promise<void>;
}

function stableSessionId(runtime: ResolutionRuntime): ReturnType<typeof SessionId> {
  const identity = `${runtime.resolution_thread_id}\u0000${runtime.segment_id}`;
  const suffix = createHash("sha256").update(identity).digest("hex").slice(0, 32);
  return SessionId(`kazusa-resolution-${suffix}`);
}

function activationKey(threadId: string, segmentId: string): string {
  return `${threadId}\u0000${segmentId}`;
}

function encodeAdmission(admission: AdmissionIdentity): string {
  const encoded = Buffer.from(JSON.stringify(admission), "utf8").toString("base64url");
  return `${ADMISSION_PREFIX}${encoded}`;
}

function nonempty(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function decodeAdmission(value: unknown): AdmissionIdentity | undefined {
  if (!nonempty(value) || !value.startsWith(ADMISSION_PREFIX)) return undefined;
  try {
    const decoded: unknown = JSON.parse(
      Buffer.from(value.slice(ADMISSION_PREFIX.length), "base64url").toString("utf8"),
    );
    if (decoded === null || typeof decoded !== "object" || Array.isArray(decoded)) {
      return undefined;
    }
    const candidate = decoded as Record<string, unknown>;
    const method = candidate.method;
    if (method !== "resolution.open" && method !== "resolution.continue") {
      return undefined;
    }
    const textKeys = [
      "operationId",
      "payloadDigest",
      "requestId",
      "threadId",
      "segmentId",
      "activationId",
    ] as const;
    if (!textKeys.every((key) => nonempty(candidate[key]))) return undefined;
    if (!Number.isInteger(candidate.leaseEpoch) || (candidate.leaseEpoch as number) < 1) {
      return undefined;
    }
    return {
      method,
      operationId: candidate.operationId as string,
      payloadDigest: candidate.payloadDigest as string,
      requestId: candidate.requestId as string,
      threadId: candidate.threadId as string,
      segmentId: candidate.segmentId as string,
      activationId: candidate.activationId as string,
      leaseEpoch: candidate.leaseEpoch as number,
    };
  } catch {
    return undefined;
  }
}

function admissionFromEvent(event: SessionEvent): AdmissionIdentity | undefined {
  if (event.type !== "user/message") return undefined;
  const source = event.data.source;
  if (source.kind !== "plugin") return undefined;
  return decodeAdmission(source.plugin);
}

function findAdmission(
  events: readonly SessionEvent[],
  operationId: string,
  payloadDigest: string,
): { admission: AdmissionIdentity; seq: number; sourceId: string } | undefined {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event === undefined) continue;
    const admission = admissionFromEvent(event);
    if (admission?.operationId !== operationId) continue;
    if (admission.payloadDigest !== payloadDigest) {
      throw new OperationReuseFault("OPERATION_ID_REUSE_MISMATCH");
    }
    return { admission, seq: event.seq, sourceId: encodeAdmission(admission) };
  }
  return undefined;
}

function fault(code: string): JsonObject {
  return { kind: "runtime_fault", fault: { code } };
}

function operationView(
  session: string,
  admission: AdmissionIdentity,
  sourceId: string,
  disposition: string,
  exhaust: JsonObject,
): JsonObject {
  const lastCommitted = exhaust.last_committed_seq;
  return {
    disposition,
    session_id: session,
    segment_id: admission.segmentId,
    activation_id: admission.activationId,
    lease_epoch: admission.leaseEpoch,
    dsh_message_source_id: sourceId,
    ...(typeof lastCommitted === "number" ? { last_committed_seq: lastCommitted } : {}),
    exhaust,
  };
}

function inspectEvents(
  session: string,
  events: readonly SessionEvent[],
  operationId: string,
  payloadDigest: string,
): JsonObject | undefined {
  const found = findAdmission(events, operationId, payloadDigest);
  if (found === undefined) return undefined;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event === undefined || event.seq <= found.seq || event.type !== "tool/result") {
      continue;
    }
    const meta = event.data.meta;
    if (meta === null || typeof meta !== "object" || Array.isArray(meta)) continue;
    const receiptValue = (meta as Record<string, unknown>).kazusa;
    if (receiptValue === null || typeof receiptValue !== "object") continue;
    const rawReceipt = receiptValue as Record<string, unknown>;
    if (rawReceipt.operation_id !== operationId) continue;
    try {
      const receipt = validateTerminalReceipt(receiptValue);
      if (
        receipt.operation_payload_digest !== payloadDigest
        || receipt.terminal_digest !== digest(receipt.terminal)
      ) {
        throw new Error("terminal receipt identity mismatch");
      }
      const terminalExhaust = replayTerminalExhaust([
        event as unknown as Record<string, unknown>,
      ]);
      return operationView(
        session,
        found.admission,
        found.sourceId,
        "terminal",
        terminalExhaust,
      );
    } catch {
      return operationView(
        session,
        found.admission,
        found.sourceId,
        "faulted",
        fault("TERMINAL_RECEIPT_INVALID"),
      );
    }
  }

  const lastTurnEnd = [...events].reverse().find(
    (event) => event.seq > found.seq && event.type === "turn/end",
  );
  if (lastTurnEnd?.type !== "turn/end") {
    return operationView(
      session,
      found.admission,
      found.sourceId,
      "admitted_active",
      { kind: "checkpointed" },
    );
  }
  const reason = lastTurnEnd.data.reason;
  if (reason.kind === "aborted" && reason.reason.kind === "hook") {
    return operationView(
      session,
      found.admission,
      found.sourceId,
      "checkpointed",
      { kind: "checkpointed" },
    );
  }
  if (reason.kind === "aborted" && reason.reason.kind === "user") {
    return operationView(
      session,
      found.admission,
      found.sourceId,
      "canceled",
      fault("RESOLUTION_CANCELED"),
    );
  }
  const code = reason.kind === "completed"
    ? "RESOLVER_ACTION_CONTRACT_EXHAUSTED"
    : "RESOLVER_RUNTIME_FAULT";
  return operationView(
    session,
    found.admission,
    found.sourceId,
    "faulted",
    fault(code),
  );
}

function validSingleAction(chunks: readonly StreamChunk[]): boolean {
  const assembler = new BlockAssembler();
  for (const chunk of chunks) assembler.push(chunk);
  const finish = assembler.finish;
  if (finish.kind === "error" || finish.kind === "aborted") return true;
  const calls = assembler.blocks().filter((block) => block.type === "tool-call");
  if (calls.length !== 1 || calls[0]?.type !== "tool-call") return false;
  if (calls[0].name !== "submit_resolution") return false;
  try {
    validateSubmitResolution(JSON.parse(calls[0].arguments));
    return true;
  } catch {
    return false;
  }
}

async function* rejectedActionStream(): AsyncIterable<StreamChunk> {
  const text = "Resolver action contract violation; correction required.";
  yield { type: "block-start", index: 0, blockType: "text" };
  yield { type: "text-delta", index: 0, text };
  yield { type: "block-end", index: 0, block: { type: "text", text } };
  yield { type: "finish", reason: { kind: "stop" } };
}

export async function buildProfile(
  id: string,
  options: { model: string; dataRoot: string; testScript?: JsonObject[] },
): Promise<ResolverProfile> {
  if (id !== PROFILE_VERSION) throw new Error(`unsupported resolver profile: ${id}`);
  if (options.model.length === 0 || options.dataRoot.length === 0) {
    throw new Error("resolver profile requires model and data root");
  }

  const storePath = join(options.dataRoot, "dsh", DSH_RELEASE, "sessions.sqlite");
  await mkdir(join(options.dataRoot, "dsh", DSH_RELEASE), { recursive: true });
  const context = new Context();
  const fibers: Fiber[] = [];
  const mount = async (
    plugin: Parameters<Context["plugin"]>[0],
    config?: unknown,
  ): Promise<void> => {
    const fiber = context.plugin(plugin, config);
    await fiber;
    fibers.push(fiber);
  };
  await mount(SessionStore);
  await mount(SqliteSessionPersistence, { path: storePath });
  await mount(LlmRuntime);
  if (options.testScript === undefined) {
    await mount(deepseekPlugin, { apiKeyEnv: "DEEPSEEK_API_KEY", maxTokens: 4096 });
  } else {
    context.llm.registerAdapter(["kazusa-test"], new ScriptedAdapter(options.testScript));
  }
  await mount(SystemPrompt, {
    includeHarnessIdentity: true,
    includeRuntimeContext: false,
    persona: (
      "Resolve the supplied objective. Call submit_resolution exactly once. "
      + "Do not answer with prose."
    ),
    toolOrder: ["submit_resolution", "<unlisted-tools>"],
  });
  await mount(ToolRuntime, { mode: "native" });
  await mount(AgentRegistry);
  await mount(AgentLoop, { agents: [], maxParallelToolCalls: 1 });
  await mount(checkpointPolicy);

  const activations = new Map<string, LiveActivation>();
  const diagnostics: ProfileDiagnostics = {
    terminal_tool_executions: 0,
    correction_attempts: 0,
    live_activations: 0,
  };

  const inspectOperation = async (
    operationId: string,
    payloadDigest: string,
  ): Promise<JsonObject> => {
    for (const header of await context.sessionPersistence.list()) {
      const inspection = await context.sessionPersistence.inspect(header.id);
      const result = inspectEvents(
        String(header.id),
        inspection.events,
        operationId,
        payloadDigest,
      );
      if (result !== undefined) return result;
    }
    return { disposition: "not_admitted" };
  };

  const requireActivation = (
    threadId: string,
    segmentId: string,
    activationId: string,
    leaseEpoch: number,
  ): LiveActivation => {
    const active = activations.get(activationKey(threadId, segmentId));
    if (
      active === undefined
      || active.admission.activationId !== activationId
      || active.admission.leaseEpoch !== leaseEpoch
    ) {
      throw new Error("STALE_ACTIVATION_OR_LEASE");
    }
    return active;
  };

  const assertPersistedFence = async (
    threadId: string,
    segmentId: string,
    activationId: string,
    leaseEpoch: number,
  ): Promise<void> => {
    const idValue = stableSessionId({
      resolution_thread_id: threadId,
      segment_id: segmentId,
    } as ResolutionRuntime);
    const headers = await context.sessionPersistence.list();
    if (!headers.some((header) => header.id === idValue)) {
      throw new Error("STALE_ACTIVATION_OR_LEASE");
    }
    const inspection = await context.sessionPersistence.inspect(idValue);
    const latest = [...inspection.events].reverse()
      .map(admissionFromEvent)
      .find((candidate) => candidate !== undefined);
    if (
      latest?.activationId !== activationId
      || latest.leaseEpoch !== leaseEpoch
      || latest.threadId !== threadId
      || latest.segmentId !== segmentId
    ) {
      throw new Error("STALE_ACTIVATION_OR_LEASE");
    }
  };

  return {
    id: PROFILE_VERSION,
    model: options.model,
    dataRoot: options.dataRoot,
    semanticTools: ["submit_resolution"],
    dshPackages: REQUIRED_DSH_PACKAGES,
    composedServices: [
      "sessions",
      "session-persistence-sqlite",
      "llm",
      "system-prompt",
      "tools",
      "agents",
      "agent-loop",
      "session-checkpoint-policy",
    ],
    diagnostics,
    async activate(method, intake, activationId, leaseEpoch) {
      const runtime = intake.runtime;
      const durable = await inspectOperation(
        runtime.operation_id,
        runtime.operation_payload_digest,
      );
      if (durable.disposition !== "not_admitted") return durable;

      const key = activationKey(runtime.resolution_thread_id, runtime.segment_id);
      const existing = activations.get(key);
      if (existing !== undefined) {
        if (leaseEpoch <= existing.admission.leaseEpoch) {
          throw new Error("STALE_ACTIVATION_OR_LEASE");
        }
        existing.handle.agent.cancel(
          { kind: "hook", reason: "lease-takeover" },
          { keepInbox: true },
        );
        await existing.handle.agent.whenIdle();
        await context.sessions.flush(existing.handle.agent.session);
        await existing.handle.dispose();
        activations.delete(key);
        diagnostics.live_activations = activations.size;
      }

      const admission: AdmissionIdentity = {
        method,
        operationId: runtime.operation_id,
        payloadDigest: runtime.operation_payload_digest,
        requestId: runtime.request_id,
        threadId: runtime.resolution_thread_id,
        segmentId: runtime.segment_id,
        activationId,
        leaseEpoch,
      };
      let terminalReached = false;
      let correctionAttempts = 0;
      const setup = (agentContext: Context): void => {
        agentContext.on("llm/stream", async function* (_request, next) {
          const chunks: StreamChunk[] = [];
          for await (const chunk of next()) chunks.push(chunk);
          if (validSingleAction(chunks)) {
            for (const chunk of chunks) yield chunk;
            return;
          }
          yield* rejectedActionStream();
        });
        agentContext.on("agent/pre-step", async (payload, next) => {
          if (payload.step > runtime.max_model_steps) return { kind: "reject" };
          return next();
        });
        agentContext.on("agent/turn-stopping", ({ agent }) => {
          if (terminalReached || correctionAttempts >= 2) return;
          correctionAttempts += 1;
          diagnostics.correction_attempts += 1;
          agent.steer(createUserMessage({
            content: [{ type: "text", text: CORRECTION_MESSAGE }],
            source: { kind: "plugin", plugin: "kazusa-resolver-correction" },
          }));
        });
        agentContext.tools.register(defineTool({
          name: "submit_resolution",
          description: "Submit the complete terminal resolution.",
          parameters: {
            status: { type: "string", required: true, enum: ["resolved", "partial", "needs_user_input", "approval_required", "unavailable", "failed"] },
            summary: { type: "string", required: true },
            findings: { type: "array", required: true, items: { type: "object", additionalProperties: true } },
            completed_subgoals: { type: "array", required: true, items: { type: "string" } },
            remaining_needs: { type: "array", required: true, items: { type: "string" } },
            clarification_request: { required: true, oneOf: [{ type: "object", additionalProperties: true }, { type: "null" }] },
            approval_request: { required: true, oneOf: [{ type: "object", additionalProperties: true }, { type: "null" }] },
            artifact_refs: { type: "array", required: true, items: { type: "string" } },
            warnings: { type: "array", required: true, items: { type: "string" } },
          },
          output: {
            schema: { type: "object", properties: { accepted: { type: "boolean", required: true }, receipt: { type: "json", required: true } }, additionalProperties: false },
            render: () => [{ type: "text", text: "Resolution accepted." }],
            presentationMeta: (_args, value) => ({ kazusa: value.receipt }),
          },
          async execute(args, execution) {
            const terminal = validateSubmitResolution(args);
            diagnostics.terminal_tool_executions += 1;
            const validated = validateTerminalReceipt({
              kind: "terminal_resolution_v1", schema_version: "1", call_id: execution.callId,
              operation_id: runtime.operation_id, operation_payload_digest: runtime.operation_payload_digest,
              request_id: runtime.request_id, resolution_thread_id: runtime.resolution_thread_id,
              segment_id: runtime.segment_id, activation_id: activationId, lease_epoch: leaseEpoch,
              scope_fingerprint: runtime.scope_fingerprint, audience_fingerprint: runtime.audience_fingerprint,
              resolver_profile_version: runtime.resolver_profile_version, dsh_release: runtime.dsh_release,
              session_store_epoch: runtime.session_store_epoch, model_route: runtime.model_route,
              tool_catalog_digest: runtime.tool_catalog_digest, policy_epoch: runtime.policy_epoch,
              terminal, terminal_digest: digest(terminal),
            });
            const receipt = (
              process.env.NODE_ENV === "test"
              && process.env.KAZUSA_DSH_TEST_CORRUPT_TERMINAL_RECEIPT === "1"
            ) ? { ...validated, terminal_digest: "sha256:corrupt" } : validated;
            terminalReached = true;
            execution.concludeTurn();
            return { accepted: true, receipt: receipt as unknown as JsonValue };
          },
        }));
      };

      const idValue = stableSessionId(runtime);
      const persisted = (await context.sessionPersistence.list()).some(
        (header) => header.id === idValue,
      );
      const agentOptions = {
        provider: options.testScript === undefined ? "deepseek-official" : "kazusa-test",
        model: options.model,
        maxTokens: 4096,
      };
      const handle = persisted
        ? await context.agents.resume({ resumeSessionId: idValue, agentOptions, setup })
        : await context.agents.create({ sessionId: idValue, agentOptions, setup });
      activations.set(key, { admission, handle });
      diagnostics.live_activations = activations.size;
      handle.agent.followup(createUserMessage({
        content: [{ type: "text", text: JSON.stringify(intake.model_input) }],
        source: { kind: "plugin", plugin: encodeAdmission(admission) },
      }));
      await handle.agent.whenIdle();
      await context.sessions.flush(handle.agent.session);
      const result = await inspectOperation(
        runtime.operation_id,
        runtime.operation_payload_digest,
      );
      if (
        result.disposition === "terminal"
        && process.env.NODE_ENV === "test"
        && process.env.KAZUSA_DSH_TEST_EXIT_AFTER_TERMINAL_COMMIT === "1"
      ) {
        process.exit(97);
      }
      return result;
    },
    async checkpoint(threadId, segmentId, activationId, leaseEpoch) {
      const active = requireActivation(threadId, segmentId, activationId, leaseEpoch);
      active.handle.agent.cancel({ kind: "hook", reason: "checkpoint" }, { keepInbox: true });
      await active.handle.agent.whenIdle();
      await context.sessions.flush(active.handle.agent.session);
      return inspectOperation(active.admission.operationId, active.admission.payloadDigest);
    },
    async cancel(threadId, segmentId, activationId, leaseEpoch) {
      const active = requireActivation(threadId, segmentId, activationId, leaseEpoch);
      active.handle.agent.cancel({ kind: "user" });
      await active.handle.agent.whenIdle();
      await context.sessions.flush(active.handle.agent.session);
      return inspectOperation(active.admission.operationId, active.admission.payloadDigest);
    },
    async amend(threadId, segmentId, activationId, leaseEpoch, amendment) {
      const active = requireActivation(threadId, segmentId, activationId, leaseEpoch);
      active.handle.agent.steer(createUserMessage({
        content: [{ type: "text", text: JSON.stringify(amendment) }],
        source: { kind: "plugin", plugin: "kazusa-resolver-amendment" },
      }));
      return { disposition: "admitted_active", session_id: String(active.handle.agent.id) };
    },
    async disposeActivation(threadId, segmentId, activationId, leaseEpoch) {
      const key = activationKey(threadId, segmentId);
      const active = activations.get(key);
      if (active === undefined) {
        await assertPersistedFence(threadId, segmentId, activationId, leaseEpoch);
        return;
      }
      requireActivation(threadId, segmentId, activationId, leaseEpoch);
      activations.delete(key);
      diagnostics.live_activations = activations.size;
      await active.handle.dispose();
    },
    inspect: inspectOperation,
    async close() {
      for (const active of activations.values()) await active.handle.dispose();
      activations.clear();
      diagnostics.live_activations = 0;
      for (const fiber of fibers.reverse()) await fiber.dispose();
    },
  };
}

class ScriptedAdapter extends LlmAdapter {
  private index = 0;

  constructor(private readonly script: JsonObject[]) { super(); }

  providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: "Kazusa test", attribution: { headers: {} } } as unknown as LlmProviderInfo;
  }

  async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    return [{ provider, id: "test-model", name: "Test model" }];
  }

  async resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    return { provider, id: model, name: model, context: { contextWindow: 32_768 }, defaultMaxTokens: 4096 };
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const step = this.script[this.index] ?? this.script[this.script.length - 1] ?? {};
    this.index += 1;
    if (step.wait === true) {
      if (options.signal === undefined) throw new Error("scripted wait requires cancellation signal");
      await new Promise<void>((resolve) => {
        options.signal?.addEventListener("abort", () => resolve(), { once: true });
      });
      yield { type: "finish", reason: { kind: "aborted", failure: { code: "ABORTED", message: "scripted request canceled" } } };
      return;
    }
    const calls = Array.isArray(step.calls) ? step.calls : step.name === undefined ? [] : [step];
    if (calls.length === 0) {
      yield { type: "block-start", index: 0, blockType: "text" };
      const value = typeof step.text === "string" ? step.text : "";
      yield { type: "text-delta", index: 0, text: value };
      yield { type: "block-end", index: 0, block: { type: "text", text: value } };
      yield { type: "finish", reason: { kind: "stop" } };
      return;
    }
    for (let index = 0; index < calls.length; index += 1) {
      const call = calls[index] as JsonObject;
      const callId = CallId(`test-call-${this.index}-${index}`);
      const name = String(call.name ?? "unknown");
      const args = JSON.stringify(call.arguments ?? {});
      yield { type: "block-start", index, blockType: "tool-call" };
      yield { type: "tool-call-delta", index, id: callId, name, argumentsDelta: args };
      yield { type: "block-end", index, block: { type: "tool-call", id: callId, name, arguments: args } };
    }
    yield { type: "finish", reason: { kind: "tool-calls" } };
  }
}
