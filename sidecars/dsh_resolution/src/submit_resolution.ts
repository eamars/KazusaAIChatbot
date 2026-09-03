import { isAbsolute, relative } from "node:path";

import { Service, type Context } from "@deepseek-ai/cordis";
import { defineTool, type ToolRunContext } from "@deepseek-ai/dsh-tools";
import type { JsonValue } from "@deepseek-ai/dsh-session";

import {
  type JsonObject,
  type ResolutionIntake,
  type SubmitResolution,
  type TerminalReceipt,
  validateEvidenceReceipt,
  validateSubmitResolution,
  validateTerminalReceipt,
} from "./contracts.js";
import { digest, projectPublicEvidence, type EvidenceReference } from "./evidence.js";

export interface TerminalEventStore {
  append(event: Record<string, unknown>): Promise<number>;
  flush(): Promise<void>;
}

export interface SubmitResolutionRegistration {
  intake: ResolutionIntake;
  activationId: string;
  leaseEpoch: number;
  diagnostics?: { terminal_tool_executions: number };
}

function publicEvidence(reference: EvidenceReference): JsonObject {
  return {
    schema_version: "evidence_receipt.v2",
    evidence_id: reference.evidence_id,
    resolution_thread_id: reference.threadId,
    segment_id: reference.segmentId,
    scope_fingerprint: reference.scopeFingerprint,
    audience_fingerprint: reference.audienceFingerprint,
    policy_epoch: reference.policyEpoch,
    source_kind: reference.source_kind,
    semantic_ref: reference.source_id,
    content_digest: reference.content_digest,
    provenance: { tool_name: reference.tool_name },
  };
}

function exhaust(
  receipt: TerminalReceipt,
  evidence: readonly EvidenceReference[],
  sequence: number,
): JsonObject {
  return {
    kind: "terminal",
    terminal: receipt.terminal,
    evidence: projectPublicEvidence(evidence).map(publicEvidence),
    identity: {
      operation_id: receipt.operation_id,
      operation_payload_digest: receipt.operation_payload_digest,
      request_id: receipt.request_id,
      resolution_thread_id: receipt.resolution_thread_id,
      segment_id: receipt.segment_id,
      activation_id: receipt.activation_id,
      lease_epoch: receipt.lease_epoch,
      brain_conversation_ref: receipt.brain_conversation_ref,
      workspace_root: receipt.workspace_root,
      route_digest: receipt.route_digest,
      scope_fingerprint: receipt.scope_fingerprint,
      catalog_digest: receipt.catalog_digest,
      interaction_issuer: receipt.interaction_issuer,
      policy_epoch: receipt.policy_epoch,
    },
    usage: {},
    last_committed_seq: sequence,
  };
}

function insideWorkspace(workspaceRoot: string, candidate: string): boolean {
  const root = workspaceRoot.replaceAll("\\", "/").replace(/\/+$/u, "").toLowerCase();
  const normalized = candidate.replaceAll("\\", "/").replace(/\/+$/u, "").toLowerCase();
  if (normalized === root) return true;
  return normalized.startsWith(`${root}/`);
}

function assertArtifactsInsideWorkspace(runtime: ResolutionIntake, terminal: SubmitResolution): void {
  for (const artifact of terminal.artifact_refs) {
    if (artifact.length === 0) continue;
    const windowsAbsolute = /^[A-Za-z]:[\\/]/u.test(artifact);
    if (!isAbsolute(artifact) && !windowsAbsolute) continue;
    if (!insideWorkspace(runtime.workspace_root, artifact)) {
      throw new Error("ARTIFACT_OUTSIDE_WORKSPACE");
    }
    const relativePath = relative(runtime.workspace_root, artifact);
    if (relativePath.split(/[\\/]/u).includes("..")) {
      throw new Error("ARTIFACT_OUTSIDE_WORKSPACE");
    }
  }
}

function evidenceReference(
  value: unknown,
  runtime: ResolutionIntake,
): EvidenceReference {
  const candidate = value as Partial<EvidenceReference>;
  if (
    typeof candidate.threadId === "string"
    && typeof candidate.segmentId === "string"
    && typeof candidate.scopeFingerprint === "string"
    && typeof candidate.audienceFingerprint === "string"
    && typeof candidate.policyEpoch === "string"
    && typeof candidate.evidence_id === "string"
  ) {
    return candidate as EvidenceReference;
  }
  const receipt = validateEvidenceReceipt(value);
  return {
    schema_version: "evidence_receipt.v2",
    evidence_id: receipt.evidence_id,
    threadId: receipt.resolution_thread_id,
    segmentId: receipt.segment_id,
    scopeFingerprint: receipt.scope_fingerprint,
    audienceFingerprint: receipt.audience_fingerprint,
    policyEpoch: receipt.policy_epoch,
    tool_name: receipt.provenance.tool_name,
    source_kind: receipt.source_kind,
    source_id: receipt.semantic_ref,
    content_digest: receipt.content_digest,
  };
}

function assertEvidenceAuthority(
  reference: EvidenceReference,
  runtime: ResolutionIntake,
): void {
  if (
    reference.threadId !== runtime.resolution_thread_id
    || reference.segmentId !== runtime.segment_id
    || reference.scopeFingerprint !== runtime.interaction_authority.scope_fingerprint
    || reference.audienceFingerprint !== runtime.interaction_authority.audience_fingerprint
    || reference.policyEpoch !== "dsh-standard-policy-v2"
  ) {
    throw new Error("EVIDENCE_AUTHORITY_MISMATCH");
  }
}

export async function commitTerminalResolution(
  store: TerminalEventStore,
  runtime: ResolutionIntake,
  activationId: string,
  leaseEpoch: number,
  callId: string,
  value: unknown,
  evidence: readonly unknown[],
): Promise<JsonObject> {
  let terminal: SubmitResolution;
  try {
    terminal = validateSubmitResolution(value);
    assertArtifactsInsideWorkspace(runtime, terminal);
  } catch (error) {
    return {
      kind: "runtime_fault",
      fault: { code: error instanceof Error ? error.message : "TERMINAL_SUBMISSION_INVALID" },
    };
  }

  let authorizedEvidence: EvidenceReference[];
  try {
    authorizedEvidence = evidence.map((item) => evidenceReference(item, runtime));
    authorizedEvidence.forEach((item) => assertEvidenceAuthority(item, runtime));
  } catch (error) {
    return {
      kind: "runtime_fault",
      fault: { code: error instanceof Error ? error.message : "EVIDENCE_AUTHORITY_MISMATCH" },
    };
  }

  const receipt = validateTerminalReceipt({
    kind: "terminal_resolution_v2",
    schema_version: "2",
    call_id: callId,
    operation_id: runtime.operation_id,
    operation_payload_digest: runtime.operation_payload_digest,
    request_id: runtime.request_id,
    resolution_thread_id: runtime.resolution_thread_id,
    segment_id: runtime.segment_id,
    activation_id: activationId,
    lease_epoch: leaseEpoch,
    brain_conversation_ref: runtime.brain_conversation_ref,
    workspace_root: runtime.workspace_root,
    route_digest: runtime.route_digest,
    scope_fingerprint: runtime.interaction_authority.scope_fingerprint,
    audience_fingerprint: runtime.interaction_authority.audience_fingerprint,
    catalog_digest: runtime.semantic_tool_authority.catalog_digest,
    interaction_issuer: runtime.interaction_authority.issuer,
    policy_epoch: "dsh-standard-policy-v2",
    terminal,
    terminal_digest: digest(terminal),
  });
  const sequence = await store.append({
    type: "tool/result",
    call_id: callId,
    result: { status: "ok" },
    meta: { kazusa: receipt },
  });
  await store.flush();
  return exhaust(receipt, authorizedEvidence, sequence);
}

export function replayTerminalExhaust(
  events: readonly Record<string, unknown>[],
  evidence: readonly EvidenceReference[] = [],
): JsonObject {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event?.type !== "tool/result") continue;
    try {
      const data = event.data as Record<string, unknown> | undefined;
      const meta = (event.meta ?? data?.meta) as Record<string, unknown>;
      const receipt = validateTerminalReceipt(meta.kazusa);
      if (receipt.terminal_digest !== digest(receipt.terminal)) {
        throw new Error("terminal digest mismatch");
      }
      const sequence = typeof event.seq === "number" ? event.seq : index + 1;
      return exhaust(receipt, evidence, sequence);
    } catch {
      return { kind: "runtime_fault", fault: { code: "TERMINAL_RECEIPT_INVALID" } };
    }
  }
  return { kind: "runtime_fault", fault: { code: "TERMINAL_RECEIPT_MISSING" } };
}

export function terminalSummary(value: SubmitResolution): string {
  return value.summary;
}

/** Request-bound terminal tool owner mounted by the host overlay. */
export class SubmitResolutionService extends Service {
  static inject = ["tools"];

  constructor(ctx: Context) {
    super(ctx, "submitResolution");
  }

  register(
    agentContext: Context,
    registration: SubmitResolutionRegistration,
  ): () => void {
    const tools = agentContext.get("tools");
    if (tools === undefined) throw new Error("DSH tool runtime is unavailable");
    const { intake, activationId, leaseEpoch, diagnostics } = registration;
    return tools.register(defineTool({
      name: "submit_resolution",
      description: "",
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
        schema: {
          type: "object",
          properties: {
            accepted: { type: "boolean", required: true },
            receipt: { type: "json", required: true },
          },
          additionalProperties: false,
        },
        render: () => [{ type: "text", text: "Resolution accepted." }],
        // Session persistence stores presentation metadata as the host-owned
        // Kazusa envelope.  Keeping the envelope here makes live DSH events
        // identical to the deterministic replay/event contract.
        presentationMeta: (_args, value) => ({ kazusa: value.receipt }),
      },
      async execute(args, execution: ToolRunContext) {
        const terminal = validateSubmitResolution(args);
        if (diagnostics !== undefined) diagnostics.terminal_tool_executions += 1;
        const receipt = validateTerminalReceipt({
          kind: "terminal_resolution_v2",
          schema_version: "2",
          call_id: execution.callId,
          operation_id: intake.operation_id,
          operation_payload_digest: intake.operation_payload_digest,
          request_id: intake.request_id,
          resolution_thread_id: intake.resolution_thread_id,
          segment_id: intake.segment_id,
          activation_id: activationId,
          lease_epoch: leaseEpoch,
          brain_conversation_ref: intake.brain_conversation_ref,
          workspace_root: intake.workspace_root,
          route_digest: intake.route_digest,
          scope_fingerprint: intake.interaction_authority.scope_fingerprint,
          audience_fingerprint: intake.interaction_authority.audience_fingerprint,
          catalog_digest: intake.semantic_tool_authority.catalog_digest,
          interaction_issuer: intake.interaction_authority.issuer,
          policy_epoch: "dsh-standard-policy-v2",
          terminal,
          terminal_digest: digest(terminal),
        });
        execution.concludeTurn();
        return { accepted: true, receipt: receipt as unknown as JsonValue };
      },
    }));
  }
}

export default class SubmitResolutionPlugin extends SubmitResolutionService {}
