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
  schema_version: "evidence_reference.v1";
  evidence_id: string;
  tool_name: string;
  source_kind: string;
  source_id: string;
  content_digest: string;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>).sort(([left], [right]) => left.localeCompare(right)).map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export function digest(value: unknown): string {
  return `sha256:${createHash("sha256").update(canonical(value)).digest("hex")}`;
}

export function buildEvidenceReceipt(input: {
  callId: string;
  operationId: string;
  resolutionThreadId: string;
  segmentId: string;
  scopeFingerprint: string;
  audienceFingerprint: string;
  policyEpoch: string;
  toolName: string;
  provenance: EvidenceReceipt["provenance"];
}): EvidenceReceipt {
  const evidenceIds = input.provenance.map((row) => row.evidence_id);
  const bounded = { evidence_ids: evidenceIds, provenance: input.provenance };
  return validateEvidenceReceipt({
    kind: "evidence_receipt_v1",
    schema_version: "1",
    call_id: input.callId,
    operation_id: input.operationId,
    resolution_thread_id: input.resolutionThreadId,
    segment_id: input.segmentId,
    scope_fingerprint: input.scopeFingerprint,
    audience_fingerprint: input.audienceFingerprint,
    policy_epoch: input.policyEpoch,
    tool_name: input.toolName,
    evidence_ids: evidenceIds,
    provenance: input.provenance,
    evidence_digest: digest(bounded),
  });
}

export class EvidenceLedger {
  private readonly references = new Map<string, EvidenceReference>();

  static rebuild(events: readonly Record<string, unknown>[]): EvidenceLedger {
    const ledger = new EvidenceLedger();
    for (const event of events) {
      if (event.type !== "tool/result") continue;
      const meta = event.meta;
      if (meta === null || typeof meta !== "object" || Array.isArray(meta)) continue;
      const receipt = (meta as Record<string, unknown>).kazusa;
      if (receipt === null || typeof receipt !== "object" || Array.isArray(receipt)) continue;
      if ((receipt as Record<string, unknown>).kind !== "evidence_receipt_v1") continue;
      ledger.register(validateEvidenceReceipt(receipt));
    }
    return ledger;
  }

  register(receipt: EvidenceReceipt): void {
    const expectedDigest = digest({ evidence_ids: receipt.evidence_ids, provenance: receipt.provenance });
    if (receipt.evidence_digest !== expectedDigest) throw new Error("evidence digest mismatch");
    receipt.provenance.forEach((row) => {
      if (this.references.has(row.evidence_id)) throw new Error("duplicate evidence identity");
      this.references.set(row.evidence_id, {
        schema_version: "evidence_reference.v1",
        evidence_id: row.evidence_id,
        threadId: receipt.resolution_thread_id,
        segmentId: receipt.segment_id,
        scopeFingerprint: receipt.scope_fingerprint,
        audienceFingerprint: receipt.audience_fingerprint,
        policyEpoch: receipt.policy_epoch,
        tool_name: receipt.tool_name,
        source_kind: row.source_kind,
        source_id: row.source_id,
        content_digest: row.content_digest,
      });
    });
  }

  resolve(evidenceIds: readonly string[], authority: EvidenceAuthority): EvidenceReference[] {
    return evidenceIds.map((evidenceId) => {
      const reference = this.references.get(evidenceId);
      if (reference === undefined) throw new Error("unknown evidence identity");
      if (reference.threadId !== authority.threadId || reference.segmentId !== authority.segmentId || reference.scopeFingerprint !== authority.scopeFingerprint || reference.audienceFingerprint !== authority.audienceFingerprint || reference.policyEpoch !== authority.policyEpoch) {
        throw new Error("evidence authority mismatch");
      }
      return structuredClone(reference);
    });
  }
}
