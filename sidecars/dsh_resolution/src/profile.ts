import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, realpath } from "node:fs/promises";
import { join, resolve } from "node:path";

import { Context } from "@deepseek-ai/cordis";
import { AgentRegistry, type AgentHandle } from "@deepseek-ai/dsh-agent";
import { createUserMessage } from "@deepseek-ai/dsh-llm";
import {
  type SessionEvent,
  SessionId,
} from "@deepseek-ai/dsh-session";
import "@deepseek-ai/dsh-session-persistence";

import { BrainInteractionService, type BrainInteractionProvider } from "./brain_interaction.js";
import { composeStandardProfile, selectPublishedTools } from "./composition.js";
import {
  DSH_RELEASE,
  nativeCatalogDigest,
  publishedCatalogDigest,
  semanticCatalogDigest,
  type JsonObject,
  PROFILE_VERSION,
  type ResolutionIntake,
  type ResolutionRuntime,
  verifyActivationToken,
  validateSubmitResolution,
  validateTerminalReceipt,
} from "./contracts.js";
import { digest, EvidenceLedger } from "./evidence.js";
import { createSecretBroker } from "./secret_broker.js";
import { SemanticGatewayService, type SemanticGatewayRegistration } from "./semantic_gateway.js";
import { OperationReuseFault } from "./operations.js";
import { replayTerminalExhaust, SubmitResolutionService } from "./submit_resolution.js";
import { routeDigest as canonicalRouteDigest, type QwenRouteConfig } from "./model_route.js";

export const PRODUCTION_SESSION_EVENT_KINDS = ["tool/result"] as const;

const ADMISSION_PREFIX = "kazusa-operation:";
const REQUIRED_DSH_PACKAGES = [
  "@deepseek-ai/dsh-app-boot",
  "@deepseek-ai/dsh-agent-presets",
  "@deepseek-ai/dsh-base",
  "@deepseek-ai/dsh-agent",
  "@deepseek-ai/dsh-agent-loop",
  "@deepseek-ai/dsh-invariants",
  "@deepseek-ai/dsh-llm",
  "@deepseek-ai/dsh-llm-pi-ai",
  "@deepseek-ai/dsh-scope",
  "@deepseek-ai/dsh-session",
  "@deepseek-ai/dsh-session-checkpoint-policy",
  "@deepseek-ai/dsh-session-persistence",
  "@deepseek-ai/dsh-session-persistence-sqlite",
  "@deepseek-ai/dsh-settings",
  "@deepseek-ai/dsh-system-prompt",
  "@deepseek-ai/dsh-tools",
] as const;

const KAZUSA_SEMANTIC_TOOL_NAMES = [
  "kazusa_search_conversation_history",
  "kazusa_read_conversation_entries",
  "kazusa_summarize_conversation_participants",
  "kazusa_search_memories",
  "kazusa_read_memories",
  "kazusa_remember_information",
  "kazusa_revise_memory",
  "kazusa_change_memory_lifecycle",
  "kazusa_find_people_by_name",
  "kazusa_read_person_profiles",
  "kazusa_recall_active_context",
  "kazusa_read_calendar_context",
  "kazusa_inspect_attached_media",
  "kazusa_inspect_public_media",
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

export interface ProfileCatalog {
  names: readonly string[];
  digest: string;
  native_catalog_digest: string;
  semantic_catalog_digest: string;
  published_catalog_digest: string;
  descriptions_stripped: true;
  native_names: readonly string[];
  semantic_names: readonly string[];
  omitted_semantic_tools: readonly { name: string; reason: "native_precedence" }[];
}

const SUBMIT_RESOLUTION_SCHEMA = {
  name: "submit_resolution",
  parameters: {
    type: "object",
    properties: {
      status: {
        type: "string",
        enum: [
          "resolved",
          "partial",
          "needs_user_input",
          "approval_required",
          "unavailable",
          "failed",
        ],
        required: true,
      },
      summary: { type: "string", required: true },
      findings: { type: "array", required: true, items: { type: "object", additionalProperties: true } },
      completed_subgoals: { type: "array", required: true, items: { type: "string" } },
      remaining_needs: { type: "array", required: true, items: { type: "string" } },
      clarification_request: { required: true, oneOf: [{ type: "object", additionalProperties: true }, { type: "null" }] },
      approval_request: { required: true, oneOf: [{ type: "object", additionalProperties: true }, { type: "null" }] },
      artifact_refs: { type: "array", required: true, items: { type: "string" } },
      warnings: { type: "array", required: true, items: { type: "string" } },
    },
    required: [
      "status", "summary", "findings", "completed_subgoals", "remaining_needs",
      "clarification_request", "approval_request", "artifact_refs", "warnings",
    ],
    additionalProperties: false,
  },
} as const;

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
  profileVersion: typeof PROFILE_VERSION;
  model: string;
  dataRoot: string;
  routeName: string;
  routeDigest: string;
  officialDigests: {
    base: string;
    standardPreset: string;
    standardAgent: string;
  };
  standardPresetPath: string;
  semanticTools: readonly string[];
  catalog: ProfileCatalog;
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

async function sha256File(path: string): Promise<string> {
  return `sha256:${createHash("sha256").update(await readFile(path)).digest("hex")}`;
}

function localPluginPath(name: string): string {
  const candidates = [
    resolve(import.meta.dirname, `${name}.js`),
    resolve(import.meta.dirname, "..", "dist", "src", `${name}.js`),
  ];
  const compiled = candidates.find((path) => existsSync(path));
  return compiled ?? resolve(import.meta.dirname, `${name}.ts`);
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
  nativeToolNames: readonly string[],
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
      const evidenceEvents = events.filter(
        (candidate) => candidate.seq > found.seq && candidate.seq < event.seq,
      );
      const ledger = EvidenceLedger.rebuild(
        evidenceEvents as unknown as Record<string, unknown>[],
        {
          authority: {
            threadId: receipt.resolution_thread_id,
            segmentId: receipt.segment_id,
            scopeFingerprint: receipt.scope_fingerprint,
            audienceFingerprint: receipt.audience_fingerprint,
            policyEpoch: receipt.policy_epoch,
          },
          nativeToolNames,
        },
      );
      const terminalExhaust = replayTerminalExhaust(
        [event as unknown as Record<string, unknown>],
        ledger.all({
          threadId: receipt.resolution_thread_id,
          segmentId: receipt.segment_id,
          scopeFingerprint: receipt.scope_fingerprint,
          policyEpoch: receipt.policy_epoch,
        }),
      );
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

export async function buildProfile(
  id: string,
  options: {
    model: string;
    dataRoot: string;
    repositoryRoot: string;
    workspaceRoot: string;
    route: QwenRouteConfig;
    hostSecrets: Record<string, string>;
    nativeEnvironment: Record<string, string>;
    launchEnvironment?: unknown;
    brainProvider: BrainInteractionProvider;
    semanticInvoker: (frame: Record<string, unknown>) => Promise<unknown>;
    persistEvidence?: (receipt: Record<string, unknown>) => Promise<void>;
    disposeSemanticWorker?: () => Promise<void> | void;
  },
): Promise<ResolverProfile> {
  if (id !== PROFILE_VERSION) throw new Error(`unsupported resolver profile: ${id}`);
  if (options.model.length === 0 || options.dataRoot.length === 0) {
    throw new Error("resolver profile requires model and data root");
  }
  if (options.route === undefined) throw new Error("resolver profile requires a canonical model route");

  const storePath = join(options.dataRoot, "dsh", DSH_RELEASE, "sessions.sqlite");
  await mkdir(join(options.dataRoot, "dsh", DSH_RELEASE), { recursive: true });
  const repositoryRoot = options.repositoryRoot;
  const workspaceRoot = options.workspaceRoot;
  const route = options.route;
  const routeDigest = canonicalRouteDigest(route);
  const composition = await composeStandardProfile({
    repositoryRoot,
    workspaceRoot,
    routeConfig: route,
    sqlitePath: storePath,
    semanticNames: KAZUSA_SEMANTIC_TOOL_NAMES,
    localPluginPaths: {
      submitResolution: localPluginPath("submit_resolution"),
      semanticGateway: localPluginPath("semantic_gateway"),
      secretBroker: localPluginPath("secret_broker"),
      brainInteraction: localPluginPath("brain_interaction"),
    },
  });
  const broker = createSecretBroker({
    hostSecrets: options.hostSecrets,
    nativeEnvironment: options.nativeEnvironment,
  });
  const appBootModule = "@deepseek-ai/dsh-app-boot";
  const { boot } = await import(appBootModule) as unknown as {
    boot(
      name: string,
      configPath: string,
      patches: Array<Record<string, unknown>>,
      prepare: (context: Context) => Promise<void> | void,
      bareModuleBaseUrl: string,
    ): Promise<Context>;
  };
  const context = await boot(
    "dsh-resolution",
    composition.rootPath,
    [...composition.basePatches, ...composition.overlayPatches],
    async (bootContext) => {
      bootContext.provide("dshHostSecrets", broker);
      if (options.launchEnvironment !== undefined) {
        bootContext.provide("launchEnvironment", options.launchEnvironment);
      }
    },
    composition.bareModuleBaseUrl,
  );
  // Root app boot resolves its own relative config rows beside root.cordis.yml,
  // while the installed Standard preset resolves bare DSH rows from the
  // packaged harness.  Agent-preset mounting captures this inherited base URL
  // before it rewrites the preset tree to the preset directory.
  context.baseUrl = composition.bareModuleBaseUrl;
  const agentLoop = context.get("agentLoop") as {
    runtime?: { ctx?: Context };
  } | undefined;
  if (agentLoop?.runtime?.ctx === undefined) {
    throw new Error("agent loop runtime context is unavailable");
  }
  agentLoop.runtime.ctx.baseUrl = composition.bareModuleBaseUrl;
  const agentPresets = context.get("agentPresets") as {
    selfCtx?: Context;
    standingKeyFor(id?: string): Promise<unknown>;
  } | undefined;
  if (agentPresets?.selfCtx === undefined) {
    throw new Error("agent presets runtime context is unavailable");
  }
  agentPresets.selfCtx.baseUrl = composition.bareModuleBaseUrl;

  // Ensure the installed Standard composition is real before publishing
  // readiness.  The health catalog is derived from this standing scope so
  // native web/tool rows and native-precedence collisions come from DSH's
  // registry rather than a copied or synthetic list.
  const standingKey = await agentPresets.standingKeyFor("standard");
  const tools = context.get("tools") as {
    schemas(scope?: unknown): Array<{ name: string; parameters?: unknown }>;
  } | undefined;
  if (tools === undefined) throw new Error("DSH tool runtime is unavailable");
  const standardNativeSchemas = tools.schemas(standingKey).map((schema) => ({
    name: schema.name,
    parameters: schema.parameters,
  }));
  const standardNativeNames = standardNativeSchemas.map((schema) => schema.name);
  const publishedCatalog = selectPublishedTools({
    nativeNames: standardNativeNames,
    semanticNames: KAZUSA_SEMANTIC_TOOL_NAMES,
  });
  const catalogNames = [
    ...publishedCatalog.nativeNames,
    ...publishedCatalog.semanticNames,
    "submit_resolution",
  ];
  const nativeDigest = nativeCatalogDigest(standardNativeSchemas);
  const semanticDigest = semanticCatalogDigest();
  const publishedDigest = publishedCatalogDigest(
    standardNativeSchemas,
    SUBMIT_RESOLUTION_SCHEMA,
    KAZUSA_SEMANTIC_TOOL_NAMES,
  );
  const catalog: ProfileCatalog = {
    names: catalogNames,
    digest: publishedDigest,
    native_catalog_digest: nativeDigest,
    semantic_catalog_digest: semanticDigest,
    published_catalog_digest: publishedDigest,
    descriptions_stripped: true,
    native_names: publishedCatalog.nativeNames,
    semantic_names: [...publishedCatalog.semanticNames, "submit_resolution"],
    omitted_semantic_tools: publishedCatalog.omittedSemanticTools,
  };

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
        catalog.native_names,
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
    profileVersion: PROFILE_VERSION,
    model: options.model,
    dataRoot: options.dataRoot,
    routeName: route.routeName,
    routeDigest,
    officialDigests: {
      base: await sha256File(composition.officialFiles.basePath),
      standardPreset: await sha256File(composition.officialFiles.standardPresetPath),
      standardAgent: await sha256File(composition.officialFiles.standardAgentPath),
    },
    standardPresetPath: composition.officialFiles.standardPresetPath,
    semanticTools: [...catalog.semantic_names],
    catalog,
    dshPackages: REQUIRED_DSH_PACKAGES,
    composedServices: [
      "sessions",
      "session-persistence-sqlite",
      "llm",
      "llm-pi-ai",
      "credentials",
      "agent-presets",
      "standard",
      "web",
      "tool-web",
      "kazusaSemanticGateway",
      "brainInteractionProvider",
      "submitResolution",
      "system-prompt",
      "tools",
      "agents",
      "agent-loop",
      "session-checkpoint-policy",
    ],
    diagnostics,
    async activate(method, intake, activationId, leaseEpoch) {
      if (intake.route_digest !== routeDigest) {
        throw new Error("ROUTE_DIGEST_MISMATCH");
      }
      const semanticSecret = options.hostSecrets.KAZUSA_DSH_TOOL_GATEWAY_SECRET;
      if (semanticSecret === undefined || semanticSecret.length === 0) {
        throw new Error("semantic gateway host binding is unavailable");
      }
      const authority = verifyActivationToken(
        intake.semantic_tool_authority.token,
        semanticSecret,
        {
          activation_id: activationId,
          lease_epoch: leaseEpoch,
          resolution_thread_id: intake.resolution_thread_id,
          segment_id: intake.segment_id,
          brain_conversation_ref: intake.brain_conversation_ref,
          scope_fingerprint: intake.interaction_authority.scope_fingerprint,
          audience_fingerprint: intake.interaction_authority.audience_fingerprint,
          workspace_root: intake.workspace_root,
          route_digest: routeDigest,
          catalog_digest: catalog.semantic_catalog_digest,
          profile_version: PROFILE_VERSION,
          model_route_digest: routeDigest,
          policy_epoch: "dsh-standard-policy-v2",
          interaction_issuer: intake.interaction_authority.issuer,
        },
      );
      if (intake.semantic_tool_authority.catalog_digest !== catalog.semantic_catalog_digest) {
        throw new Error("SEMANTIC_CATALOG_DIGEST_MISMATCH");
      }
      const durable = await inspectOperation(
        intake.operation_id,
        intake.operation_payload_digest,
      );
      if (durable.disposition !== "not_admitted") return durable;

      const key = activationKey(intake.resolution_thread_id, intake.segment_id);
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
        operationId: intake.operation_id,
        payloadDigest: intake.operation_payload_digest,
        requestId: intake.request_id,
        threadId: intake.resolution_thread_id,
        segmentId: intake.segment_id,
        activationId,
        leaseEpoch,
      };
      const setup = async (agentContext: Context): Promise<void> => {
        const presets = agentContext.get("agentPresets") as {
          mount(agent: Context, id?: string): Promise<unknown>;
        } | undefined;
        if (presets === undefined) throw new Error("standard agent preset service is unavailable");
        await presets.mount(agentContext, "standard");

        const systemPrompt = agentContext.get("systemPrompt") as {
          section(section: { name: string; order: number; text: string }): () => void;
        } | undefined;
        if (systemPrompt === undefined) throw new Error("system prompt service is unavailable");
        const disposeTerminalContract = systemPrompt.section({
          name: "kazusa:terminal-contract",
          order: 190,
          text: [
            "Resolve one bounded operation from the user JSON object.",
            "The objective field is the exact outcome to pursue; the facts field contains supplied evidence and constraints for that outcome.",
            "First identify the smallest unresolved needs, then use available tools only for evidence or work required by the objective.",
            "When the objective is resolved, partially resolved, waiting for user input or approval, unavailable, or failed, call submit_resolution exactly once and end the turn immediately.",
            "Choose status from resolved, partial, needs_user_input, approval_required, unavailable, or failed.",
            "Put the bounded outcome in summary, supported conclusions in findings, finished goal parts in completed_subgoals, unresolved requirements in remaining_needs, produced references in artifact_refs, and material caveats in warnings.",
            "Set clarification_request only for needs_user_input and approval_request only for approval_required; use null for both fields in every other status.",
            "The terminal response consists only of the submit_resolution tool call; keep intermediate reasoning concise and place the final synthesis in that call.",
          ].join(" "),
        });
        agentContext.effect(
          () => disposeTerminalContract,
          "kazusa-terminal-contract.registration",
        );

        const semanticService = context.get("kazusaSemanticGateway") as SemanticGatewayService | undefined;
        if (semanticService === undefined) {
          throw new Error("semantic gateway host binding is unavailable");
        }
        const semanticRegistration: SemanticGatewayRegistration = {
          authority: authority as unknown as Record<string, unknown>,
          authorityToken: intake.semantic_tool_authority.token,
          secret: semanticSecret,
          invoke: options.semanticInvoker,
          persistEvidence: options.persistEvidence ?? (async () => undefined),
        };
        const published = semanticService.register(agentContext, semanticRegistration);
        agentContext.effect(() => published.dispose, "semantic-gateway.registration");

        const brainService = context.get("brainInteractionProvider") as BrainInteractionService | undefined;
        if (brainService === undefined) throw new Error("Brain interaction service is unavailable");
        brainService.register(agentContext, {
          requestContext: {
            operation_id: intake.operation_id,
            operation_payload_digest: intake.operation_payload_digest,
            resolution_thread_id: intake.resolution_thread_id,
            segment_id: intake.segment_id,
            activation_id: activationId,
            lease_epoch: leaseEpoch,
            brain_conversation_ref: intake.brain_conversation_ref,
            platform: authority.service_scope.platform,
            platform_channel_id: authority.service_scope.platform_channel_id,
            global_user_id: authority.service_scope.global_user_id,
            scope_fingerprint: intake.interaction_authority.scope_fingerprint,
            audience_fingerprint: authority.audience_fingerprint,
            profile_version: PROFILE_VERSION,
            catalog_digest: authority.catalog_digest,
            model_route_digest: authority.model_route_digest,
            workspace_fingerprint: authority.workspace_fingerprint,
            policy_epoch: authority.policy_epoch,
            issued_reference_digest: authority.issued_reference_digest,
            issued_at: authority.issued_at,
            expires_at: authority.expires_at,
            nonce: authority.nonce,
            issuer: authority.interaction_issuer,
          },
          provider: options.brainProvider,
        });

        const submitService = context.get("submitResolution") as SubmitResolutionService | undefined;
        if (submitService === undefined) throw new Error("submit resolution service is unavailable");
        const disposeSubmit = submitService.register(agentContext, {
          intake,
          activationId,
          leaseEpoch,
          diagnostics,
        });
        agentContext.effect(() => disposeSubmit, "submit-resolution.registration");
      };

      const idValue = stableSessionId({
        ...intake,
        segment_id: authority.segment_id,
      });
      const persisted = (await context.sessionPersistence.list()).some(
        (header) => header.id === idValue,
      );
      const agentOptions = {
        provider: route.routeName,
        model: route.model,
        maxTokens: route.maxCompletionTokens,
      };
      const handle = persisted
        ? await context.agents.resume({ resumeSessionId: idValue, agentOptions, setup })
        : await context.agents.create({
          sessionId: idValue,
          meta: { cwd: workspaceRoot, agentPreset: "standard" },
          agentOptions,
          setup,
        });
      activations.set(key, { admission, handle });
      diagnostics.live_activations = activations.size;
      handle.agent.followup(createUserMessage({
        content: [{ type: "text", text: JSON.stringify(intake.model_input) }],
        source: { kind: "plugin", plugin: encodeAdmission(admission) },
      }));
      await handle.agent.whenIdle();
      await context.sessions.flush(handle.agent.session);
      const result = await inspectOperation(
        intake.operation_id,
        intake.operation_payload_digest,
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
      await context.fiber.dispose();
      await options.disposeSemanticWorker?.();
    },
  };
}
