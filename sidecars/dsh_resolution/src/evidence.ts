import { createHash } from "node:crypto";

import { type EvidenceReceipt, validateEvidenceReceipt } from "./contracts.js";

export interface EvidenceAuthority {
  threadId: string;
  segmentId: string;
  scopeFingerprint: string;
  audienceFingerprint: string;
  policyEpoch: string;
}

export interface EvidenceReference extends EvidenceAuthority {
  schema_version: "evidence_receipt.v2";
  evidence_id: string;
  tool_name: string;
  source_kind: string;
  source_id: string;
  content_digest: string;
}

export interface NativeEvidenceReplay {
  authority: EvidenceAuthority;
  nativeToolNames: readonly string[];
}

export const MAX_PUBLIC_EVIDENCE_RECEIPTS = 64;

/**
 * Project validated evidence for a public terminal result.
 *
 * The latest occurrence of an evidence id wins.  The retained occurrences
 * remain in their input order so the boundary is deterministic for both
 * initial commits and restart replay.
 */
export function projectPublicEvidence(
  references: readonly EvidenceReference[],
): EvidenceReference[] {
  const latest = new Map<string, { index: number; reference: EvidenceReference }>();
  references.forEach((reference, index) => {
    latest.delete(reference.evidence_id);
    latest.set(reference.evidence_id, { index, reference });
  });
  return [...latest.values()]
    .sort((left, right) => left.index - right.index)
    .slice(-MAX_PUBLIC_EVIDENCE_RECEIPTS)
    .map(({ reference }) => structuredClone(reference));
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

export function digest(value: unknown): string {
  return `sha256:${createHash("sha256").update(canonical(value)).digest("hex")}`;
}

type ProvenanceInput = {
  evidence_id: string;
  source_kind: string;
  source_id?: string;
  semantic_ref?: string;
  content_digest: string;
};

export function buildEvidenceReceipt(input: {
  callId: string;
  operationId: string;
  resolutionThreadId: string;
  segmentId: string;
  scopeFingerprint: string;
  audienceFingerprint: string;
  policyEpoch: string;
  toolName: string;
  provenance: readonly ProvenanceInput[];
}): EvidenceReceipt {
  if (input.provenance.length !== 1) {
    throw new Error("one evidence receipt is required per semantic reference");
  }
  const row = input.provenance[0];
  if (row === undefined) throw new Error("evidence provenance is required");
  return validateEvidenceReceipt({
    schema_version: "evidence_receipt.v2",
    resolution_thread_id: input.resolutionThreadId,
    segment_id: input.segmentId,
    scope_fingerprint: input.scopeFingerprint,
    audience_fingerprint: input.audienceFingerprint,
    policy_epoch: input.policyEpoch,
    evidence_id: row.evidence_id,
    source_kind: row.source_kind,
    semantic_ref: row.semantic_ref ?? row.source_id ?? row.evidence_id,
    content_digest: row.content_digest,
    provenance: { tool_name: input.toolName },
  });
}

export class EvidenceLedger {
  private readonly references = new Map<string, EvidenceReference>();
  private readonly history: EvidenceReference[] = [];

  static rebuild(
    events: readonly Record<string, unknown>[],
    nativeReplay?: NativeEvidenceReplay,
  ): EvidenceLedger {
    const ledger = new EvidenceLedger();
    const nativeToolNames = new Set(nativeReplay?.nativeToolNames ?? []);
    const nativeCalls = new Map<string, string>();
    for (const event of events) {
      const data = eventData(event);
      if (event.type === "tool/call") {
        const callId = nonemptyText(data?.callId);
        const toolName = nonemptyText(data?.name);
        if (
          callId !== undefined
          && toolName !== undefined
          && nativeToolNames.has(toolName)
        ) {
          nativeCalls.set(callId, toolName);
        }
        continue;
      }
      if (event.type !== "tool/result") continue;
      const meta = event.meta ?? data?.meta;
      if (meta !== null && typeof meta === "object" && !Array.isArray(meta)) {
        const row = meta as Record<string, unknown>;
        const candidates = [
          ...(Array.isArray(row.evidence) ? row.evidence : []),
          ...(
            row.kazusa === null || typeof row.kazusa !== "object"
              ? []
              : [row.kazusa]
          ),
        ];
        for (const receipt of candidates) {
          if (
            receipt === null
            || typeof receipt !== "object"
            || Array.isArray(receipt)
          ) {
            continue;
          }
          try {
            ledger.registerLatest(validateEvidenceReceipt(receipt));
          } catch {
            continue;
          }
        }
      }
      if (nativeReplay === undefined || data === undefined) continue;
      const nativeReceipt = nativeResultReceipt(
        data,
        nativeCalls,
        nativeReplay.authority,
      );
      if (nativeReceipt !== undefined) {
        ledger.registerLatest(nativeReceipt);
      }
    }
    return ledger;
  }

  register(receipt: EvidenceReceipt): void {
    if (this.references.has(receipt.evidence_id)) {
      throw new Error("duplicate evidence identity");
    }
    const reference = this.toReference(receipt);
    this.references.set(receipt.evidence_id, reference);
    this.history.push(reference);
  }

  private registerLatest(receipt: EvidenceReceipt): void {
    const reference = this.toReference(receipt);
    this.references.delete(receipt.evidence_id);
    this.references.set(receipt.evidence_id, reference);
    this.history.push(reference);
  }

  private toReference(receipt: EvidenceReceipt): EvidenceReference {
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

  resolve(evidenceIds: readonly string[], authority: EvidenceAuthority): EvidenceReference[] {
    return evidenceIds.map((evidenceId) => {
      const reference = this.references.get(evidenceId);
      if (reference === undefined) throw new Error("unknown evidence identity");
      if (
        reference.threadId !== authority.threadId
        || reference.segmentId !== authority.segmentId
        || reference.scopeFingerprint !== authority.scopeFingerprint
        || reference.audienceFingerprint !== authority.audienceFingerprint
        || reference.policyEpoch !== authority.policyEpoch
      ) {
        throw new Error("evidence authority mismatch");
      }
      return structuredClone(reference);
    });
  }

  all(
    authority: Omit<EvidenceAuthority, "audienceFingerprint"> & {
      audienceFingerprint?: string;
    },
  ): EvidenceReference[] {
    for (const reference of this.history) {
      if (
        reference.threadId !== authority.threadId
        || reference.segmentId !== authority.segmentId
        || reference.scopeFingerprint !== authority.scopeFingerprint
        || reference.policyEpoch !== authority.policyEpoch
        || (
          authority.audienceFingerprint !== undefined
          && reference.audienceFingerprint !== authority.audienceFingerprint
        )
      ) {
        throw new Error("evidence authority mismatch");
      }
    }
    return [...this.references.values()].map((reference) => structuredClone(reference));
  }
}

function eventData(
  event: Readonly<Record<string, unknown>>,
): Record<string, unknown> | undefined {
  const data = event.data;
  if (data === null || typeof data !== "object" || Array.isArray(data)) {
    return undefined;
  }
  return data as Record<string, unknown>;
}

function nonemptyText(value: unknown): string | undefined {
  if (typeof value !== "string" || value.length === 0) return undefined;
  return value;
}

function record(value: unknown): Record<string, unknown> | undefined {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  return value as Record<string, unknown>;
}

function nativeResultReceipt(
  data: Readonly<Record<string, unknown>>,
  nativeCalls: ReadonlyMap<string, string>,
  authority: EvidenceAuthority,
): EvidenceReceipt | undefined {
  if (data.error !== undefined) return undefined;
  const message = record(data.message);
  const source = record(message?.source);
  const callId = nonemptyText(source?.callId);
  if (source?.kind !== "tool" || callId === undefined) return undefined;
  const toolName = nativeCalls.get(callId);
  if (toolName === undefined) return undefined;
  const content = message?.content;
  if (!Array.isArray(content) || content.length !== 1) return undefined;
  const resultBlock = record(content[0]);
  if (
    resultBlock?.type !== "tool-result"
    || resultBlock.toolCallId !== callId
    || resultBlock.isError === true
    || !Array.isArray(resultBlock.content)
  ) {
    return undefined;
  }
  const contentDigest = digest(resultBlock.content);
  const evidenceDigest = digest({
    call_id: callId,
    tool_name: toolName,
    content_digest: contentDigest,
  });
  return validateEvidenceReceipt({
    schema_version: "evidence_receipt.v2",
    resolution_thread_id: authority.threadId,
    segment_id: authority.segmentId,
    scope_fingerprint: authority.scopeFingerprint,
    audience_fingerprint: authority.audienceFingerprint,
    policy_epoch: authority.policyEpoch,
    evidence_id: `native:${evidenceDigest.slice("sha256:".length)}`,
    source_kind: "native",
    semantic_ref: `dsh-native:${toolName}:${callId}`,
    content_digest: contentDigest,
    provenance: { tool_name: toolName },
  });
}
