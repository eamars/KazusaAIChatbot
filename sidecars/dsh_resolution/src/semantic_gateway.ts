import { createHmac, randomUUID, timingSafeEqual } from "node:crypto";

import { Service, type Context } from "@deepseek-ai/cordis";
import { defineTool, type ToolRunContext } from "@deepseek-ai/dsh-tools";
import {
  CALL_SCHEMA_VERSION,
  canonicalJson,
  digest,
  SEMANTIC_TOOL_NAMES as SEMANTIC_TOOL_LIST,
  validateActivationAuthority,
  verifyActivationToken,
} from "./contracts.js";
import { buildEvidenceReceipt } from "./evidence.js";

const SEMANTIC_TOOL_NAMES = new Set<string>(SEMANTIC_TOOL_LIST);

const MUTATION_NAMES = new Set([
  "kazusa_remember_information",
  "kazusa_revise_memory",
  "kazusa_change_memory_lifecycle",
]);

const MAX_FRAME_BYTES = 32 * 1024;

export interface SemanticGatewayResult extends Record<string, unknown> {
  schema_version: "kazusa_semantic_capability_result.v1";
  status: "ok" | "empty" | "denied" | "invalid" | "timeout" | "unavailable";
  entities: Array<Record<string, unknown>>;
  page: { has_more: boolean; next_page_ref: string | null };
  evidence: Array<Record<string, unknown>>;
  mutation: Record<string, unknown> | null;
  error: Record<string, unknown> | null;
}

export interface SemanticGateway {
  invoke(
    operation: string,
    argumentsValue: Record<string, unknown>,
    callId?: string,
  ): Promise<SemanticGatewayResult>;
}

export interface SemanticGatewayPluginConfig {
  names: readonly string[];
  secret?: string;
  authority?: Record<string, unknown>;
  authorityToken?: string;
  invoke?: (operation: string, argumentsValue: Record<string, unknown>, callId: string) => Promise<unknown>;
  persistEvidence?: (receipt: Record<string, unknown>) => Promise<void>;
}

export interface SemanticGatewayRegistration {
  authority: Record<string, unknown>;
  authorityToken: string;
  secret: string;
  invoke: (frame: Record<string, unknown>) => Promise<unknown>;
  persistEvidence: (receipt: Record<string, unknown>) => Promise<void>;
}

function text(value: unknown, name: string): string {
  if (typeof value !== "string" || value.length === 0) throw new Error(`${name} is required`);
  return value;
}

function validateResult(value: unknown): SemanticGatewayResult {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("semantic result must be an object");
  }
  const result = value as Record<string, unknown>;
  const expected = ["schema_version", "status", "entities", "page", "evidence", "mutation", "error"];
  if (JSON.stringify(Object.keys(result).sort()) !== JSON.stringify([...expected].sort())) {
    throw new Error("semantic result fields are invalid");
  }
  if (result.schema_version !== "kazusa_semantic_capability_result.v1") {
    throw new Error("semantic result schema is unsupported");
  }
  if (!["ok", "empty", "denied", "invalid", "timeout", "unavailable"].includes(String(result.status))) {
    throw new Error("semantic result status is unsupported");
  }
  if (!Array.isArray(result.entities) || result.entities.some((item) => item === null || typeof item !== "object" || Array.isArray(item))) {
    throw new Error("semantic result entities are invalid");
  }
  const page = result.page;
  if (page === null || typeof page !== "object" || Array.isArray(page)) throw new Error("semantic result page is invalid");
  if (JSON.stringify(Object.keys(page).sort()) !== JSON.stringify(["has_more", "next_page_ref"])) {
    throw new Error("semantic result page fields are invalid");
  }
  if (typeof (page as Record<string, unknown>).has_more !== "boolean") throw new Error("semantic result page flag is invalid");
  const nextPage = (page as Record<string, unknown>).next_page_ref;
  if (nextPage !== null) text(nextPage, "semantic result page.next_page_ref");
  if (!Array.isArray(result.evidence) || result.evidence.some((item) => item === null || typeof item !== "object" || Array.isArray(item))) {
    throw new Error("semantic result evidence is invalid");
  }
  if (result.mutation !== null && (result.mutation === undefined || typeof result.mutation !== "object" || Array.isArray(result.mutation))) {
    throw new Error("semantic result mutation is invalid");
  }
  if (result.error !== null && (result.error === undefined || typeof result.error !== "object" || Array.isArray(result.error))) {
    throw new Error("semantic result error is invalid");
  }
  return {
    schema_version: "kazusa_semantic_capability_result.v1",
    status: result.status as SemanticGatewayResult["status"],
    entities: structuredClone(result.entities) as Array<Record<string, unknown>>,
    page: {
      has_more: (page as Record<string, unknown>).has_more as boolean,
      next_page_ref: nextPage as string | null,
    },
    evidence: structuredClone(result.evidence) as Array<Record<string, unknown>>,
    mutation: result.mutation === null ? null : structuredClone(result.mutation) as Record<string, unknown>,
    error: result.error === null ? null : structuredClone(result.error) as Record<string, unknown>,
  };
}

export function createSemanticGateway(options: {
  authority: Record<string, unknown>;
  authorityToken: string;
  secret: string;
  call: (frame: Record<string, unknown>) => Promise<unknown>;
  persistEvidence: (receipt: Record<string, unknown>) => Promise<void>;
  now?: () => Date;
}): SemanticGateway {
  const secret = text(options.secret, "semantic gateway secret");
  const now = options.now ?? (() => new Date());
  const authority = validateActivationAuthority(options.authority);
  const verifiedAuthority = verifyActivationToken(
    options.authorityToken,
    secret,
    {},
    now().getTime(),
  );
  if (canonicalJson(verifiedAuthority) !== canonicalJson(authority)) {
    throw new Error("activation authority token does not match registration");
  }
  let sequence = 0;
  return {
    async invoke(operation, argumentsValue, suppliedCallId) {
      if (!SEMANTIC_TOOL_NAMES.has(operation)) throw new Error("semantic operation is unsupported");
      if (argumentsValue === null || typeof argumentsValue !== "object" || Array.isArray(argumentsValue)) {
        throw new Error("semantic arguments must be an object");
      }
      const callId = suppliedCallId === undefined
        ? `semantic-call-${++sequence}-${randomUUID()}`
        : text(suppliedCallId, "semantic call id");
      const argumentsDigest = digest(argumentsValue);
      const issuedReferenceDigest = authority.issued_reference_digest;
      const idempotencyKey = MUTATION_NAMES.has(operation)
        ? `idem:${digest({
          operation,
          arguments_digest: argumentsDigest,
          resolution_thread_id: authority.resolution_thread_id,
          segment_id: authority.segment_id,
          activation_id: authority.activation_id,
          lease_epoch: authority.lease_epoch,
          issued_reference_digest: issuedReferenceDigest,
          service_scope: authority.service_scope,
          scope_fingerprint: authority.scope_fingerprint,
          audience_fingerprint: authority.audience_fingerprint,
        })}`
        : null;
      const unsigned = {
        schema_version: CALL_SCHEMA_VERSION,
        call_id: callId,
        operation,
        arguments: structuredClone(argumentsValue),
        arguments_digest: argumentsDigest,
        issued_reference_digest: issuedReferenceDigest,
        idempotency_key: idempotencyKey,
        authority: structuredClone(authority),
      };
      const signature = createHmac("sha256", secret)
        .update(canonicalJson(unsigned), "utf8")
        .digest("hex");
      const signed = { ...unsigned, signature };
      const frameBytes = Buffer.byteLength(canonicalJson(signed), "utf8");
      if (frameBytes > MAX_FRAME_BYTES) throw new Error("semantic call exceeds frame limit");
      const expectedMac = createHmac("sha256", secret)
        .update(canonicalJson(unsigned), "utf8")
        .digest("hex");
      if (!timingSafeEqual(Buffer.from(signed.signature, "utf8"), Buffer.from(expectedMac, "utf8"))) {
        throw new Error("semantic call signature mismatch");
      }
      const result = validateResult(await options.call(signed));
      const evidence = result.evidence.map((raw) => buildEvidenceReceipt({
        callId,
        operationId: authority.activation_id,
        resolutionThreadId: authority.resolution_thread_id,
        segmentId: authority.segment_id,
        scopeFingerprint: authority.scope_fingerprint,
        audienceFingerprint: authority.audience_fingerprint,
        policyEpoch: authority.policy_epoch,
        toolName: operation,
        provenance: [{
          evidence_id: text(
            raw.evidence_id ?? raw.receipt_id,
            "semantic evidence id",
          ),
          source_kind: text(
            raw.source_kind,
            "semantic evidence source kind",
          ),
          semantic_ref: text(
            raw.semantic_ref,
            "semantic evidence reference",
          ),
          content_digest: text(
            raw.content_digest,
            "semantic evidence digest",
          ),
        }],
      }));
      for (const receipt of evidence) await options.persistEvidence(receipt);
      return { ...result, evidence };
    },
  };
}

export { SEMANTIC_TOOL_NAMES };

const RESULT_SCHEMA = {
  type: "object",
  properties: {
    schema_version: { type: "string", required: true },
    status: { type: "string", required: true },
    entities: { type: "array", required: true, items: { type: "object", additionalProperties: true } },
    page: { type: "object", required: true, additionalProperties: true },
    evidence: { type: "array", required: true, items: { type: "object", additionalProperties: true } },
    mutation: { required: true, oneOf: [{ type: "object", additionalProperties: true }, { type: "null" }] },
    error: { required: true, oneOf: [{ type: "object", additionalProperties: true }, { type: "null" }] },
  },
  additionalProperties: false,
} as const;

const TIME_RANGE_PARAMETERS = {
  type: "object",
  properties: {
    start_at: { type: "string" },
    end_at: { type: "string" },
  },
  additionalProperties: false,
} as const;

const PROVENANCE_PARAMETERS = {
  oneOf: [
    {
      type: "object",
      properties: { conversation_entry_ref: { type: "string", required: true } },
      additionalProperties: false,
    },
    {
      type: "object",
      properties: { current_task: { type: "string", required: true } },
      additionalProperties: false,
    },
  ],
} as const;

const SEMANTIC_PARAMETERS: Record<string, Record<string, unknown>> = {
  kazusa_search_conversation_history: {
    query: { type: "string", required: true },
    time_range: TIME_RANGE_PARAMETERS,
    max_results: { type: "integer" },
    next_page_ref: { type: "string" },
  },
  kazusa_read_conversation_entries: {
    conversation_entry_refs: { type: "array", required: true, items: { type: "string" } },
  },
  kazusa_summarize_conversation_participants: {
    time_range: TIME_RANGE_PARAMETERS,
    max_people: { type: "integer" },
    next_page_ref: { type: "string" },
  },
  kazusa_search_memories: {
    query: { type: "string", required: true },
    subject_scope: { type: "string", enum: ["current_user", "active_character", "shared_world", "all"] },
    memory_kinds: { type: "array", items: { type: "string", enum: ["profile_fact", "relationship", "commitment", "experience", "world_knowledge"] } },
    max_results: { type: "integer" },
    next_page_ref: { type: "string" },
  },
  kazusa_read_memories: {
    memory_refs: { type: "array", required: true, items: { type: "string" } },
  },
  kazusa_remember_information: {
    subject: { type: "string", required: true, enum: ["current_user", "active_character", "shared_world"] },
    information: { type: "string", required: true },
    memory_kind: { type: "string", required: true, enum: ["profile_fact", "relationship", "commitment", "experience", "world_knowledge"] },
    reason: { type: "string", required: true },
    provenance: { ...PROVENANCE_PARAMETERS, required: true },
  },
  kazusa_revise_memory: {
    memory_ref: { type: "string", required: true },
    revised_information: { type: "string", required: true },
    reason: { type: "string", required: true },
  },
  kazusa_change_memory_lifecycle: {
    memory_ref: { type: "string", required: true },
    transition: { type: "string", required: true, enum: ["activate", "complete", "cancel", "archive"] },
    reason: { type: "string", required: true },
  },
  kazusa_find_people_by_name: {
    display_name: { type: "string", required: true },
    match_relation: { type: "string", required: true, enum: ["exact", "contains", "starts_with", "ends_with"] },
    max_results: { type: "integer" },
    next_page_ref: { type: "string" },
  },
  kazusa_read_person_profiles: {
    person_refs: { type: "array", required: true, items: { type: "string" } },
  },
  kazusa_recall_active_context: {
    kinds: { type: "array", required: true, items: { type: "string", enum: ["commitments", "progress", "history", "calendar"] } },
    max_results: { type: "integer" },
  },
  kazusa_read_calendar_context: {
    view: { type: "string", required: true, enum: ["schedules", "recent_runs", "pending_runs"] },
    max_results: { type: "integer" },
    next_page_ref: { type: "string" },
  },
  kazusa_inspect_attached_media: {
    attached_media_ref: { type: "string", required: true },
    question: { type: "string", required: true },
  },
};

function unavailableSemanticResult(): SemanticGatewayResult {
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

/** Request-bound semantic service that owns the model-facing tool registrations. */
export class SemanticGatewayService extends Service {
  private readonly config: SemanticGatewayPluginConfig;

  constructor(ctx: Context, config: SemanticGatewayPluginConfig) {
    super(ctx, "kazusaSemanticGateway");
    if (!Array.isArray(config.names) || config.names.some((name) => !SEMANTIC_TOOL_NAMES.has(name))) {
      throw new Error("semantic gateway names must be the approved semantic catalog");
    }
    this.config = config;
  }

  async invoke(
    operation: string,
    argumentsValue: Record<string, unknown>,
    callId?: string,
    registration?: SemanticGatewayRegistration,
  ): Promise<SemanticGatewayResult> {
    const secret = registration?.secret ?? this.config.secret;
    const authority = registration?.authority ?? this.config.authority;
    const authorityToken = registration?.authorityToken ?? this.config.authorityToken;
    const invoke = registration?.invoke ?? (this.config.invoke === undefined
      ? undefined
      : async (frame: Record<string, unknown>) => this.config.invoke?.(
        String(frame.operation),
        frame.arguments as Record<string, unknown>,
        String(frame.call_id),
      ));
    if (secret === undefined || secret.length === 0 || authority === undefined || authorityToken === undefined || invoke === undefined) {
      return unavailableSemanticResult();
    }
    const gateway = createSemanticGateway({
      authority,
      authorityToken,
      secret,
      call: invoke,
      persistEvidence: registration?.persistEvidence ?? this.config.persistEvidence ?? (async () => undefined),
    });
    return gateway.invoke(operation, argumentsValue, callId);
  }

  /** Register the selected semantic leaves in one unpublished Agent scope. */
  register(
    agentContext: Context,
    registration: SemanticGatewayRegistration,
  ): { published: PublishedSemanticTools; dispose: () => void } {
    const tools = agentContext.get("tools");
    if (tools === undefined) throw new Error("DSH tool runtime is unavailable");
    const nativeNames = tools.schemas().map((schema) => schema.name);
    const selected = selectSemanticTools({
      nativeNames,
      semanticNames: this.config.names,
    });
    const disposers = selected.semanticNames.map((name) => this.registerTool(agentContext, name, registration));
    return {
      published: selected,
      dispose: () => { for (const dispose of disposers.reverse()) dispose(); },
    };
  }

  private registerTool(
    agentContext: Context,
    name: string,
    registration: SemanticGatewayRegistration,
  ): () => void {
    const tools = agentContext.get("tools");
    if (tools === undefined) throw new Error("DSH tool runtime is unavailable");
    const service = this;
    return tools.register(defineTool({
      name,
      // The semantic catalog is deliberately description-stripped. The empty
      // required field satisfies the upstream definition type without adding
      // model-facing policy prose.
      description: "",
      parameters: SEMANTIC_PARAMETERS[name] as never,
      output: {
        schema: RESULT_SCHEMA,
        render: (_args, value) => [{ type: "text", text: JSON.stringify(value) }],
        presentationMeta: (_args, value) => value,
      },
      async execute(args, execution: ToolRunContext) {
        const result = await service.invoke(
          name,
          args as Record<string, unknown>,
          String(execution.callId),
          registration,
        );
        return result as unknown as never;
      },
    }));
  }
}

export interface PublishedSemanticTools {
  nativeNames: string[];
  semanticNames: string[];
  omittedSemanticTools: Array<{ name: string; reason: "native_precedence" }>;
}

function selectSemanticTools(options: {
  nativeNames: readonly string[];
  semanticNames: readonly string[];
}): PublishedSemanticTools {
  const nativeNames = [...options.nativeNames];
  const native = new Set(nativeNames);
  const semanticNames: string[] = [];
  const omittedSemanticTools: PublishedSemanticTools["omittedSemanticTools"] = [];
  for (const name of options.semanticNames) {
    if (native.has(name)) omittedSemanticTools.push({ name, reason: "native_precedence" });
    else if (!semanticNames.includes(name)) semanticNames.push(name);
  }
  return { nativeNames, semanticNames, omittedSemanticTools };
}

/** Host composition entry point; the loader mounts one request-bound service. */
export default class SemanticGatewayPlugin extends SemanticGatewayService {}
