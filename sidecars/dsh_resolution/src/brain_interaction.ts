import { createHash, createHmac } from "node:crypto";

import { Service, type Context } from "@deepseek-ai/cordis";

export interface BrainInteractionProvider {
  handle(request: Record<string, unknown>): Promise<Record<string, unknown>>;
}

export interface BrainInteractionRegistration {
  /** Immutable host-side identity captured for the owning DSH activation. */
  requestContext: Record<string, unknown>;
  /** Brain client that authenticates and persists the interaction decision. */
  provider: BrainInteractionProvider;
}

interface ApprovalRequestLike {
  agent: AgentLike;
  toolName: string;
  callId?: string;
  reason?: string;
}

interface QuestionRequestLike {
  agent?: AgentLike;
  questions: Array<Record<string, unknown>>;
}

interface QuestionAnswer {
  answers: Array<{ id: string; selected: string[]; custom?: string }>;
}

interface AgentSessionLike {
  id: string;
  events: ReadonlyArray<{
    type: string;
    data?: {
      callId?: unknown;
      name?: unknown;
      arguments?: unknown;
    };
  }>;
}

interface AgentLike {
  ctx: Context;
  session: AgentSessionLike;
}

interface NormalizedRequest {
  interaction_id: string;
  kind: "approval" | "question" | "plan_review";
  operation_id: string;
  operation_payload_digest: string;
  resolution_thread_id: string;
  segment_id: string;
  activation_id: string;
  lease_epoch: number;
  dsh_call_id: string;
  tool_name: string | null;
  arguments_digest: string;
  transient_detail: string;
  brain_conversation_ref: string;
  platform: string;
  platform_channel_id: string;
  global_user_id: string;
  scope_fingerprint: string;
  audience_fingerprint: string;
  profile_version: string;
  catalog_digest: string;
  model_route_digest: string;
  workspace_fingerprint: string;
  policy_epoch: string;
  issued_reference_digest: string;
  issuer: string;
  nonce: string;
  issued_at: string;
  expires_at: string;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}

function digest(value: unknown): string {
  return `sha256:${createHash("sha256").update(canonical(value)).digest("hex")}`;
}

function requiredText(value: unknown, name: string): string {
  if (typeof value !== "string" || value.trim().length === 0) throw new Error(`${name} is required`);
  return value;
}

export function deriveInteractionNonce(
  activationNonce: string,
  activationId: string,
  leaseEpoch: number,
  interactionId: string,
): string {
  if (!Number.isInteger(leaseEpoch) || leaseEpoch < 1) throw new Error("lease_epoch is invalid");
  const identity = {
    domain: "dsh_brain_interaction.nonce.v2",
    activation_id: requiredText(activationId, "activation_id"),
    lease_epoch: leaseEpoch,
    interaction_id: requiredText(interactionId, "interaction_id"),
  };
  return `nonce:${createHmac("sha256", requiredText(activationNonce, "activation_nonce"))
    .update(canonical(identity))
    .digest("hex")}`;
}

const PRESENTATION_ARGUMENT_KEYS = new Set(["description", "justification"]);

export function nativeToolArgumentsDigest(
  toolName: string,
  argumentsValue: unknown,
): string {
  const normalizedToolName = requiredText(toolName, "tool_name");
  const executableArguments = nativeToolArguments(normalizedToolName, argumentsValue);
  return digest({
    tool_name: normalizedToolName,
    arguments: executableArguments,
  });
}

function nativeToolArguments(
  toolName: string,
  argumentsValue: unknown,
): Record<string, unknown> {
  const normalizedToolName = requiredText(toolName, "tool_name");
  let parsed: unknown = argumentsValue;
  if (typeof argumentsValue === "string") {
    try {
      parsed = JSON.parse(argumentsValue);
    } catch (error) {
      throw new Error(`native tool arguments are invalid: ${String(error)}`);
    }
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("native tool arguments must be an object");
  }
  return Object.fromEntries(
    Object.entries(parsed as Record<string, unknown>)
      .filter(([key]) => !PRESENTATION_ARGUMENT_KEYS.has(key)),
  );
}

function boundedDetail(value: unknown, name: string): string {
  const detail = canonical(value);
  if (detail.length === 0 || detail.length > 8_000) {
    throw new Error(`${name} exceeds the interaction detail bound`);
  }
  return detail;
}

export function buildBrainInteractionFrame(
  requestContext: Record<string, unknown>,
  fields: Record<string, unknown>,
): Record<string, unknown> {
  const interactionId = fields.interaction_id
    ?? `${String(requestContext.resolution_thread_id)}:${String(fields.dsh_call_id)}`;
  const normalizedInteractionId = requiredText(interactionId, "interaction_id");
  const activationId = requiredText(requestContext.activation_id, "requestContext.activation_id");
  const leaseEpoch = requestContext.lease_epoch;
  if (!Number.isInteger(leaseEpoch) || (leaseEpoch as number) < 1) {
    throw new Error("requestContext.lease_epoch is invalid");
  }
  return {
    ...requestContext,
    ...fields,
    interaction_id: normalizedInteractionId,
    nonce: deriveInteractionNonce(
      requiredText(requestContext.nonce, "requestContext.nonce"),
      activationId,
      leaseEpoch as number,
      normalizedInteractionId,
    ),
    schema_version: "dsh_brain_interaction.v2",
  };
}

function normalizeKind(value: unknown): NormalizedRequest["kind"] {
  const kind = requiredText(value, "kind");
  if (kind === "approval/request" || kind === "approval") return "approval";
  if (kind === "userQuestions.ask" || kind === "question") return "question";
  if (kind === "plan_review") return "plan_review";
  throw new Error("interaction kind is unsupported");
}

function normalizeRequest(value: Record<string, unknown>, now: () => Date): NormalizedRequest {
  if (value.schema_version !== "dsh_brain_interaction.v2") {
    throw new Error("interaction schema is unsupported");
  }
  const leaseEpoch = value.lease_epoch;
  if (!Number.isInteger(leaseEpoch) || (leaseEpoch as number) < 1) throw new Error("lease_epoch is invalid");
  const detailValue = requiredText(value.transient_detail, "transient_detail");
  if (detailValue.length > 8_000) throw new Error("transient_detail is invalid");
  const normalizedKind = normalizeKind(value.kind);
  if (normalizedKind === "approval"
    && (value.tool_name === undefined || value.tool_name === null)) {
    throw new Error("approval interaction requires tool_name");
  }
  const issuedAt = requiredText(value.issued_at, "issued_at");
  const expiresAt = requiredText(value.expires_at, "expires_at");
  const issuedMs = Date.parse(issuedAt);
  const expiresMs = Date.parse(expiresAt);
  if (!Number.isFinite(issuedMs) || !Number.isFinite(expiresMs) || expiresMs <= issuedMs || expiresMs - issuedMs > 300_000) {
    throw new Error("interaction lifetime is invalid");
  }
  void now;
  return {
    interaction_id: requiredText(value.interaction_id, "interaction_id"),
    kind: normalizedKind,
    operation_id: requiredText(value.operation_id, "operation_id"),
    operation_payload_digest: requiredText(value.operation_payload_digest, "operation_payload_digest"),
    resolution_thread_id: requiredText(value.resolution_thread_id, "resolution_thread_id"),
    segment_id: requiredText(value.segment_id, "segment_id"),
    activation_id: requiredText(value.activation_id, "activation_id"),
    lease_epoch: leaseEpoch as number,
    dsh_call_id: requiredText(value.dsh_call_id, "dsh_call_id"),
    tool_name: value.tool_name === undefined || value.tool_name === null
      ? null
      : requiredText(value.tool_name, "tool_name"),
    arguments_digest: requiredText(value.arguments_digest, "arguments_digest"),
    transient_detail: detailValue,
    brain_conversation_ref: requiredText(value.brain_conversation_ref, "brain_conversation_ref"),
    platform: requiredText(value.platform, "platform"),
    platform_channel_id: requiredText(value.platform_channel_id, "platform_channel_id"),
    global_user_id: requiredText(value.global_user_id, "global_user_id"),
    scope_fingerprint: requiredText(value.scope_fingerprint, "scope_fingerprint"),
    audience_fingerprint: requiredText(value.audience_fingerprint, "audience_fingerprint"),
    profile_version: requiredText(value.profile_version, "profile_version"),
    catalog_digest: requiredText(value.catalog_digest, "catalog_digest"),
    model_route_digest: requiredText(value.model_route_digest, "model_route_digest"),
    workspace_fingerprint: requiredText(value.workspace_fingerprint, "workspace_fingerprint"),
    policy_epoch: requiredText(value.policy_epoch, "policy_epoch"),
    issued_reference_digest: requiredText(value.issued_reference_digest, "issued_reference_digest"),
    issuer: requiredText(value.issuer, "issuer"),
    nonce: requiredText(value.nonce, "nonce"),
    issued_at: issuedAt,
    expires_at: expiresAt,
  };
}

function readDecision(
  value: unknown,
  request: NormalizedRequest,
  requestDigest: string,
): Record<string, unknown> & { kind: string } {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Brain interaction response is invalid");
  }
  const outer = value as Record<string, unknown>;
  if (outer.schema_version !== "dsh_brain_interaction.v2") {
    throw new Error("Brain interaction response schema is unsupported");
  }
  if (outer.request_digest !== requestDigest) {
    throw new Error("Brain interaction response request digest is invalid");
  }
  if (outer.interaction_id !== request.interaction_id || outer.kind !== request.kind) {
    throw new Error("Brain interaction response identity is invalid");
  }
  const allowedFields = new Set([
    "schema_version", "interaction_id", "request_digest", "kind",
    "decision", "answer", "reason", "grant",
  ]);
  if (Object.keys(outer).some((key) => !allowedFields.has(key))) {
    throw new Error("Brain interaction response contains unsupported fields");
  }
  if (typeof outer.decision !== "string") {
    throw new Error("Brain interaction decision is required");
  }
  const kind = outer.decision;
  if (!["answer", "allow_once", "reject"].includes(kind)) {
    throw new Error("Brain interaction decision is unsupported");
  }
  if (kind === "answer") {
    const answer = requiredText(outer.answer, "answer");
    if (answer.length > 2_000) throw new Error("answer is too long");
  } else if (outer.answer !== null && outer.answer !== undefined) {
    throw new Error("answer is status-specific");
  }
  requiredText(outer.reason, "reason");
  if (typeof outer.reason === "string" && outer.reason.length > 2_000) {
    throw new Error("reason is too long");
  }
  if (kind === "answer" && request.kind === "approval") {
    throw new Error("answer is incompatible with approval");
  }
  if (kind === "allow_once" && request.kind === "question") {
    throw new Error("allow_once is incompatible with question");
  }
  if (kind === "allow_once" && (outer.grant === undefined || outer.grant === null)) {
    throw new Error("allow_once response requires a grant");
  }
  if (kind !== "allow_once" && outer.grant !== undefined && outer.grant !== null) {
    throw new Error("grant is status-specific");
  }
  return { ...outer, kind };
}

function validateAllowOnceGrant(
  value: unknown,
  request: NormalizedRequest,
): void {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Brain allow-once grant is invalid");
  }
  const grant = value as Record<string, unknown>;
  if (grant.schema_version !== "dsh_brain_interaction.v2") {
    throw new Error("Brain allow-once grant schema is unsupported");
  }
  if (grant.interaction_id !== request.interaction_id) {
    throw new Error("Brain allow-once grant interaction_id is invalid");
  }
  requiredText(grant.interaction_id, "grant.interaction_id");
  requiredText(grant.issued_at, "grant.issued_at");
  requiredText(grant.expires_at, "grant.expires_at");
  if (grant.grant_status !== "consumed") {
    throw new Error("Brain allow-once grant status is invalid");
  }
  const exactIdentity: ReadonlyArray<[string, unknown]> = [
    ["resolution_thread_id", request.resolution_thread_id],
    ["segment_id", request.segment_id],
    ["activation_id", request.activation_id],
    ["lease_epoch", request.lease_epoch],
    ["tool_name", request.tool_name],
    ["arguments_digest", request.arguments_digest],
    ["workspace_fingerprint", request.workspace_fingerprint],
    ["scope_fingerprint", request.scope_fingerprint],
    ["policy_epoch", request.policy_epoch],
  ];
  for (const [field, expected] of exactIdentity) {
    if (grant[field] !== expected) {
      throw new Error(`Brain allow-once grant ${field} is invalid`);
    }
  }
}

export function createBrainInteractionProvider(options: {
  secret: string;
  request: (request: Record<string, unknown>) => Promise<unknown>;
  now?: () => Date;
}): BrainInteractionProvider {
  const secret = requiredText(options.secret, "Brain interaction secret");
  const now = options.now ?? (() => new Date());
  return {
    async handle(input) {
      const request = normalizeRequest(input, now);
      const unsigned = {
        schema_version: "dsh_brain_interaction.v2",
        interaction_id: request.interaction_id,
        kind: request.kind,
        operation_id: request.operation_id,
        operation_payload_digest: request.operation_payload_digest,
        resolution_thread_id: request.resolution_thread_id,
        segment_id: request.segment_id,
        activation_id: request.activation_id,
        lease_epoch: request.lease_epoch,
        dsh_call_id: request.dsh_call_id,
        tool_name: request.tool_name,
        arguments_digest: request.arguments_digest,
        transient_detail: request.transient_detail,
        brain_conversation_ref: request.brain_conversation_ref,
        platform: request.platform,
        platform_channel_id: request.platform_channel_id,
        global_user_id: request.global_user_id,
        scope_fingerprint: request.scope_fingerprint,
        audience_fingerprint: request.audience_fingerprint,
        profile_version: request.profile_version,
        catalog_digest: request.catalog_digest,
        model_route_digest: request.model_route_digest,
        workspace_fingerprint: request.workspace_fingerprint,
        policy_epoch: request.policy_epoch,
        issued_reference_digest: request.issued_reference_digest,
        issuer: request.issuer,
        nonce: request.nonce,
        issued_at: request.issued_at,
        expires_at: request.expires_at,
      };
      const requestDigest = digest(unsigned);
      const signedRequest = {
        ...unsigned,
        mac: createHmac("sha256", secret).update(canonical(unsigned)).digest("hex"),
      };
      const response = await options.request(signedRequest);
      const decision = readDecision(response, request, requestDigest);
      if (decision.kind === "allow_once") {
        validateAllowOnceGrant(decision.grant, request);
        return { kind: "allow_once" };
      }
      if (decision.kind === "answer") {
        const answer = requiredText(decision.answer, "answer");
        if (answer.length > 2_000) throw new Error("answer is too long");
        return { kind: "answer", answer };
      }
      return { kind: "reject" };
    },
  };
}

export { canonical as canonicalBrainJson, digest as brainRequestDigest };

/**
 * Bridge the installed DSH approval and user-question services to Brain.
 *
 * The service is mounted once on the host context. Individual unpublished
 * agents bind their immutable intake identity through {@link register}; event
 * callbacks then select that binding by the exact agent context. No direct
 * human answerer is installed and no raw question text is forwarded as a
 * continuation command.
 */
export class BrainInteractionService extends Service {
  static inject = ["approval", "userQuestions"];

  private readonly registrations = new Map<string, BrainInteractionRegistration>();
  private readonly disposeApproval: () => void;
  private readonly disposeQuestions: () => void;

  constructor(ctx: Context, _config: { hostOnly: boolean }) {
    super(ctx, "brainInteractionProvider");
    const approvalContext = ctx as Context & {
      on(
        event: string,
        listener: (request: ApprovalRequestLike) => Promise<string>,
        options: { global: boolean },
      ): () => boolean;
    };
    this.disposeApproval = approvalContext.on(
      "approval/request",
      async (request) => {
        const registration = this.registrations.get(request.agent.session.id);
        if (registration === undefined) return "unavailable";
        return await this.handleApproval(registration, request);
      },
      { global: true },
    );
    const questions = ctx.get("userQuestions") as {
      registerProvider(provider: { ask(request: QuestionRequestLike): Promise<QuestionAnswer> }): () => void;
    } | undefined;
    if (questions === undefined) throw new Error("DSH user-question service is unavailable");
    this.disposeQuestions = questions.registerProvider({
      ask: async (request) => this.askQuestion(request),
    });
  }

  register(
    agentContext: Context,
    registration: BrainInteractionRegistration,
  ): () => void {
    if (registration.requestContext === null
      || typeof registration.requestContext !== "object"
      || Array.isArray(registration.requestContext)) {
      throw new Error("Brain interaction request context is invalid");
    }
    const agent = (agentContext as Context & { agent?: AgentLike }).agent;
    if (agent === undefined) throw new Error("Brain interaction requires an owning agent");
    const sessionId = agent.session.id;
    if (this.registrations.has(sessionId)) throw new Error("Brain interaction binding already exists");
    this.registrations.set(sessionId, registration);
    const dispose = () => {
      if (this.registrations.get(sessionId) === registration) this.registrations.delete(sessionId);
    };
    agentContext.effect(() => dispose, "brain-interaction.registration");
    return dispose;
  }

  async close(): Promise<void> {
    this.disposeApproval();
    this.disposeQuestions();
    this.registrations.clear();
  }

  private async handleApproval(
    registration: BrainInteractionRegistration,
    request: ApprovalRequestLike,
  ): Promise<string> {
    try {
      const callId = requiredText(request.callId, "approval call id");
      const nativeToolCalls = request.agent.session.events.filter(
        (event) => event.type === "tool/call"
          && event.data?.callId === callId
          && event.data?.name === request.toolName,
      );
      if (nativeToolCalls.length !== 1) {
        throw new Error("native tool call arguments could not be recovered");
      }
      const nativeToolCall = nativeToolCalls[0];
      const nativeArguments = nativeToolCall?.data?.arguments;
      if (typeof nativeArguments !== "string") {
        throw new Error("native tool call arguments are unavailable");
      }
      const executableArguments = nativeToolArguments(request.toolName, nativeArguments);
      const reason = requiredText(request.reason, "approval reason");
      const frame = this.frameFor(registration, {
        kind: "approval",
        dsh_call_id: callId,
        tool_name: request.toolName,
        transient_detail: boundedDetail({
          tool_name: request.toolName,
          reason,
          arguments: executableArguments,
        }, "approval detail"),
        arguments_digest: nativeToolArgumentsDigest(request.toolName, nativeArguments),
      });
      const decision = await registration.provider.handle(frame);
      if (decision.kind === "allow_once") return "allowed-once";
      if (decision.kind === "reject") return "rejected";
      return "unavailable";
    } catch {
      return "unavailable";
    }
  }

  private frameFor(
    registration: BrainInteractionRegistration,
    fields: Record<string, unknown>,
  ): Record<string, unknown> {
    return buildBrainInteractionFrame(registration.requestContext, fields);
  }

  private async askQuestion(request: QuestionRequestLike): Promise<QuestionAnswer> {
    if (request.agent === undefined) throw new Error("Brain interaction requires a live agent identity");
    const registration = this.registrations.get(request.agent.session.id);
    if (registration === undefined) throw new Error("Brain interaction binding is unavailable");
    if (request.questions.length !== 1) {
      throw new Error("DSH interaction requires exactly one complete question");
    }
    const first = request.questions[0];
    if (first === undefined) throw new Error("DSH interaction question is unavailable");
    const questionId = requiredText(first.id, "question.id");
    requiredText(first.question, "question");
    const options = first.options;
    if (options !== undefined && !Array.isArray(options)) {
      throw new Error("DSH interaction question choices are invalid");
    }
    if (Array.isArray(options)) {
      for (const option of options) {
        if (option === null || typeof option !== "object" || Array.isArray(option)) {
          throw new Error("DSH interaction question choice is invalid");
        }
        requiredText((option as Record<string, unknown>).label, "question choice label");
      }
    }
    const intent = first?.intent;
    const kind = intent !== null && typeof intent === "object"
      && (intent as Record<string, unknown>).kind === "plan-review"
      ? "plan_review"
      : "question";
    const frame = this.frameFor(registration, {
      kind,
      dsh_call_id: `question:${questionId}`,
      tool_name: null,
      transient_detail: boundedDetail({ questions: request.questions }, "question detail"),
      arguments_digest: digest(request.questions),
    });
    const decision = await registration.provider.handle(frame);
    if (decision.kind === "answer") {
      const answer = requiredText(decision.answer, "answer");
      return { answers: [{ id: questionId, selected: [], custom: answer }] };
    }
    if (decision.kind === "allow_once") {
      const approve = intent !== null && typeof intent === "object"
        && typeof (intent as Record<string, unknown>).approve === "string"
        ? (intent as Record<string, unknown>).approve as string
        : "allow_once";
      return { answers: [{ id: questionId, selected: [approve] }] };
    }
    if (decision.kind === "reject") return { answers: [{ id: questionId, selected: [] }] };
    throw new Error("BRAIN_INTERACTION_UNAVAILABLE");
  }
}

/** Host composition entry point; the installed DSH services call this bridge. */
export default class BrainInteractionPlugin extends BrainInteractionService {}
