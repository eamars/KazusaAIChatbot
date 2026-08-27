import { describe, expect, it } from "vitest";

import { EvidenceLedger, buildEvidenceReceipt } from "../src/evidence.js";
import { PRODUCTION_SESSION_EVENT_KINDS } from "../src/profile.js";

function receipt() {
  return buildEvidenceReceipt({
    callId: "call_1",
    operationId: "op_1",
    resolutionThreadId: "res_1",
    segmentId: "seg_1",
    scopeFingerprint: "sha256:scope",
    audienceFingerprint: "sha256:audience",
    policyEpoch: "2026-08-28.1",
    toolName: "fixture_evidence",
    provenance: [{ evidence_id: "ev_1", source_kind: "fixture", source_id: "source_1", content_digest: "sha256:content" }],
  });
}

describe("evidence", () => {
  it("rebuilds authorized evidence from supported tool result metadata after restart", () => {
    const rebuilt = EvidenceLedger.rebuild([{ type: "tool/result", meta: { kazusa: receipt() } }]);
    expect(rebuilt.resolve(["ev_1"], { threadId: "res_1", segmentId: "seg_1", scopeFingerprint: "sha256:scope", audienceFingerprint: "sha256:audience", policyEpoch: "2026-08-28.1" }))
      .toHaveLength(1);
  });

  it("declares no custom session event kind in the production profile", () => {
    expect(PRODUCTION_SESSION_EVENT_KINDS).toEqual(["tool/result"]);
  });

  it("excludes source content credentials capability tokens and ACLs from receipts", () => {
    const serialized = JSON.stringify(receipt());
    for (const forbidden of ["source_content", "credential", "capability_token", "acl"]) {
      expect(serialized).not.toContain(forbidden);
    }
  });

  it("rejects unknown cross-scope and cross-segment evidence", () => {
    const ledger = EvidenceLedger.rebuild([{ type: "tool/result", meta: { kazusa: receipt() } }]);
    expect(() => ledger.resolve(["ev_1"], { threadId: "res_1", segmentId: "other", scopeFingerprint: "sha256:scope", audienceFingerprint: "sha256:audience", policyEpoch: "2026-08-28.1" }))
      .toThrow(/evidence/);
    expect(() => ledger.resolve(["missing"], { threadId: "res_1", segmentId: "seg_1", scopeFingerprint: "sha256:scope", audienceFingerprint: "sha256:audience", policyEpoch: "2026-08-28.1" }))
      .toThrow(/evidence/);
  });
});
