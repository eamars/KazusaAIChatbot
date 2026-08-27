import {
  type JsonObject,
  type ResolutionRuntime,
  type SubmitResolution,
  type TerminalReceipt,
  validateSubmitResolution,
  validateTerminalReceipt,
} from "./contracts.js";
import { digest, type EvidenceReference } from "./evidence.js";

export interface TerminalEventStore {
  append(event: Record<string, unknown>): Promise<number>;
  flush(): Promise<void>;
}

function publicEvidence(reference: EvidenceReference): JsonObject {
  return {
    schema_version: "evidence_reference.v1",
    evidence_id: reference.evidence_id,
    resolution_thread_id: reference.threadId,
    segment_id: reference.segmentId,
    scope_fingerprint: reference.scopeFingerprint,
    audience_fingerprint: reference.audienceFingerprint,
    policy_epoch: reference.policyEpoch,
    tool_name: reference.tool_name,
    source_kind: reference.source_kind,
    source_id: reference.source_id,
    content_digest: reference.content_digest,
  };
}

function exhaust(receipt: TerminalReceipt, evidence: readonly EvidenceReference[], sequence: number): JsonObject {
  return {
    kind: "terminal",
    terminal: receipt.terminal,
    evidence: evidence.map(publicEvidence),
    identity: {
      operation_id: receipt.operation_id,
      operation_payload_digest: receipt.operation_payload_digest,
      request_id: receipt.request_id,
      resolution_thread_id: receipt.resolution_thread_id,
      segment_id: receipt.segment_id,
      activation_id: receipt.activation_id,
      lease_epoch: receipt.lease_epoch,
      scope_fingerprint: receipt.scope_fingerprint,
      audience_fingerprint: receipt.audience_fingerprint,
      resolver_profile_version: receipt.resolver_profile_version,
      dsh_release: receipt.dsh_release,
      session_store_epoch: receipt.session_store_epoch,
      model_route: receipt.model_route,
      tool_catalog_digest: receipt.tool_catalog_digest,
      policy_epoch: receipt.policy_epoch,
    },
    usage: {},
    last_committed_seq: sequence,
  };
}

export async function commitTerminalResolution(
  store: TerminalEventStore,
  runtime: ResolutionRuntime,
  activationId: string,
  leaseEpoch: number,
  callId: string,
  value: unknown,
  evidence: readonly EvidenceReference[],
): Promise<JsonObject> {
  const terminal = validateSubmitResolution(value);
  const receipt = validateTerminalReceipt({
    kind: "terminal_resolution_v1",
    schema_version: "1",
    call_id: callId,
    operation_id: runtime.operation_id,
    operation_payload_digest: runtime.operation_payload_digest,
    request_id: runtime.request_id,
    resolution_thread_id: runtime.resolution_thread_id,
    segment_id: runtime.segment_id,
    activation_id: activationId,
    lease_epoch: leaseEpoch,
    scope_fingerprint: runtime.scope_fingerprint,
    audience_fingerprint: runtime.audience_fingerprint,
    resolver_profile_version: runtime.resolver_profile_version,
    dsh_release: runtime.dsh_release,
    session_store_epoch: runtime.session_store_epoch,
    model_route: runtime.model_route,
    tool_catalog_digest: runtime.tool_catalog_digest,
    policy_epoch: runtime.policy_epoch,
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
  return exhaust(receipt, evidence, sequence);
}

export function replayTerminalExhaust(events: readonly Record<string, unknown>[]): JsonObject {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (event?.type !== "tool/result") continue;
    try {
      const data = event.data as Record<string, unknown> | undefined;
      const meta = (event.meta ?? data?.meta) as Record<string, unknown>;
      const receipt = validateTerminalReceipt(meta.kazusa);
      if (receipt.terminal_digest !== digest(receipt.terminal)) throw new Error("terminal digest mismatch");
      const sequence = typeof event.seq === "number" ? event.seq : index + 1;
      return exhaust(receipt, [], sequence);
    } catch {
      return { kind: "runtime_fault", fault: { code: "TERMINAL_RECEIPT_INVALID" } };
    }
  }
  return { kind: "runtime_fault", fault: { code: "TERMINAL_RECEIPT_MISSING" } };
}

export function terminalSummary(value: SubmitResolution): string {
  return value.summary;
}
