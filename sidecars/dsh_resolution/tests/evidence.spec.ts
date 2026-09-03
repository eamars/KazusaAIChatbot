import { describe, expect, it } from "vitest";

import {
  EvidenceLedger,
  buildEvidenceReceipt,
  projectPublicEvidence,
  type EvidenceReference,
} from "../src/evidence.js";
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

function reference(
  evidenceId: string,
  contentDigest = `sha256:${evidenceId}`,
): EvidenceReference {
  return {
    schema_version: "evidence_receipt.v2",
    evidence_id: evidenceId,
    threadId: "res_1",
    segmentId: "seg_1",
    scopeFingerprint: "sha256:scope",
    audienceFingerprint: "sha256:audience",
    policyEpoch: "2026-08-28.1",
    tool_name: "fixture_evidence",
    source_kind: "fixture",
    source_id: `source-${evidenceId}`,
    content_digest: contentDigest,
  };
}

describe("evidence", () => {
  it("rebuilds authorized evidence from supported tool result metadata after restart", () => {
    const rebuilt = EvidenceLedger.rebuild([{
      type: "tool/result",
      data: { meta: { evidence: [receipt()] } },
    }]);
    expect(rebuilt.resolve(["ev_1"], { threadId: "res_1", segmentId: "seg_1", scopeFingerprint: "sha256:scope", audienceFingerprint: "sha256:audience", policyEpoch: "2026-08-28.1" }))
      .toHaveLength(1);
    expect(rebuilt.all({ threadId: "res_1", segmentId: "seg_1", scopeFingerprint: "sha256:scope", policyEpoch: "2026-08-28.1" }))
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

describe("V2 evidence", () => {
  it("projects successful Standard native tool results as authority-bound evidence", () => {
    const authority = {
      threadId: "thread-native",
      segmentId: "segment-native",
      scopeFingerprint: "sha256:scope-native",
      audienceFingerprint: "sha256:audience-native",
      policyEpoch: "dsh-standard-policy-v2",
    };
    const rebuilt = EvidenceLedger.rebuild([
      {
        type: "tool/call",
        data: {
          callId: "call-native-weather",
          name: "web_search",
          arguments: JSON.stringify({ query: "Christchurch current weather" }),
        },
      },
      {
        type: "tool/result",
        data: {
          message: {
            id: "message-native-weather",
            role: "user",
            source: { kind: "tool", callId: "call-native-weather" },
            content: [{
              type: "tool-result",
              toolCallId: "call-native-weather",
              content: [{
                type: "text",
                text: "Christchurch: 6 C, rain, observed by wttr.in",
              }],
            }],
          },
        },
      },
    ], {
      authority,
      nativeToolNames: ["web_search", "pwsh"],
    });

    const evidence = rebuilt.all(authority);
    expect(evidence).toHaveLength(1);
    expect(evidence[0]).toMatchObject({
      threadId: "thread-native",
      segmentId: "segment-native",
      scopeFingerprint: "sha256:scope-native",
      audienceFingerprint: "sha256:audience-native",
      policyEpoch: "dsh-standard-policy-v2",
      tool_name: "web_search",
      source_kind: "native",
    });
    expect(evidence[0]?.source_id).toContain("call-native-weather");
    expect(evidence[0]?.content_digest).toMatch(/^sha256:/u);
  });

  it("does not project failed or non-catalog tool results as native evidence", () => {
    const authority = {
      threadId: "thread-native",
      segmentId: "segment-native",
      scopeFingerprint: "sha256:scope-native",
      audienceFingerprint: "sha256:audience-native",
      policyEpoch: "dsh-standard-policy-v2",
    };
    const resultEvent = (callId: string, isError: boolean) => ({
      type: "tool/result",
      data: {
        message: {
          id: `message-${callId}`,
          role: "user",
          source: { kind: "tool", callId },
          content: [{
            type: "tool-result",
            toolCallId: callId,
            content: [{ type: "text", text: "tool output" }],
            isError,
          }],
        },
        ...(isError ? { error: { name: "ToolError", code: "FAILED" } } : {}),
      },
    });
    const rebuilt = EvidenceLedger.rebuild([
      {
        type: "tool/call",
        data: { callId: "failed-native", name: "pwsh", arguments: "{}" },
      },
      resultEvent("failed-native", true),
      {
        type: "tool/call",
        data: { callId: "unknown-tool", name: "not_standard", arguments: "{}" },
      },
      resultEvent("unknown-tool", false),
    ], {
      authority,
      nativeToolNames: ["web_search", "pwsh"],
    });

    expect(rebuilt.all(authority)).toEqual([]);
  });

  it("projects the latest 64 unique receipts in chronological order", () => {
    const references = Array.from({ length: 66 }, (_, index) => (
      reference(`ev_${index + 1}`)
    ));
    const projected = projectPublicEvidence(references);

    expect(projected).toHaveLength(64);
    expect(projected.map((item) => item.evidence_id)).toEqual(
      Array.from({ length: 64 }, (_, index) => `ev_${index + 3}`),
    );
  });

  it("collapses duplicate ids to the most recent receipt deterministically", () => {
    const projected = projectPublicEvidence([
      reference("ev_1"),
      reference("ev_duplicate", "sha256:old"),
      reference("ev_2"),
      reference("ev_duplicate", "sha256:new"),
    ]);

    expect(projected.map((item) => item.evidence_id)).toEqual([
      "ev_1", "ev_2", "ev_duplicate",
    ]);
    expect(projected[2]?.content_digest).toBe("sha256:new");
    expect(new Set(projected.map((item) => item.evidence_id)).size)
      .toBe(projected.length);
  });

  it("keeps the latest duplicate when rebuilding the ledger after restart", () => {
    const oldReceipt = receipt();
    const latestReceipt = { ...oldReceipt, content_digest: "sha256:latest" };
    const rebuilt = EvidenceLedger.rebuild([
      { type: "tool/result", meta: { evidence: [oldReceipt] } },
      { type: "tool/result", meta: { evidence: [latestReceipt] } },
    ]);

    expect(rebuilt.all({
      threadId: "res_1",
      segmentId: "seg_1",
      scopeFingerprint: "sha256:scope",
      policyEpoch: "2026-08-28.1",
    })[0]?.content_digest).toBe("sha256:latest");
  });

  it("rebuilds semantic native and artifact receipts after restart", async () => {
    const evidence = await import("../src/evidence.js");
    const receiptBuilder = evidence.buildEvidenceReceipt as unknown as (
      input: Record<string, unknown>,
    ) => Record<string, any>;
    const receipts = [
      receiptBuilder({
        callId: "call-native",
        operationId: "op-v2",
        resolutionThreadId: "thread-v2",
        segmentId: "segment-v2",
        scopeFingerprint: "sha256:scope",
        audienceFingerprint: "sha256:audience",
        policyEpoch: "dsh-standard-policy-v2",
        toolName: "pwsh",
        provenance: [{ evidence_id: "native-1", source_kind: "native", source_id: "native-output", content_digest: "sha256:native" }],
      }),
      receiptBuilder({
        callId: "call-semantic",
        operationId: "op-v2",
        resolutionThreadId: "thread-v2",
        segmentId: "segment-v2",
        scopeFingerprint: "sha256:scope",
        audienceFingerprint: "sha256:audience",
        policyEpoch: "dsh-standard-policy-v2",
        toolName: "kazusa_search_memories",
        provenance: [{ evidence_id: "semantic-1", source_kind: "semantic", source_id: "semantic-ref", content_digest: "sha256:semantic" }],
      }),
      receiptBuilder({
        callId: "call-artifact",
        operationId: "op-v2",
        resolutionThreadId: "thread-v2",
        segmentId: "segment-v2",
        scopeFingerprint: "sha256:scope",
        audienceFingerprint: "sha256:audience",
        policyEpoch: "dsh-standard-policy-v2",
        toolName: "write_file",
        provenance: [{ evidence_id: "artifact-1", source_kind: "artifact", source_id: "artifact-ref", content_digest: "sha256:artifact" }],
      }),
    ];
    for (const item of receipts) expect(item.schema_version).toBe("evidence_receipt.v2");
    const rebuilt = evidence.EvidenceLedger.rebuild(
      receipts.map((item) => ({ type: "tool/result", meta: { kazusa: item } })),
    );
    expect(rebuilt.resolve(
      ["native-1", "semantic-1", "artifact-1"],
      { threadId: "thread-v2", segmentId: "segment-v2", scopeFingerprint: "sha256:scope", audienceFingerprint: "sha256:audience", policyEpoch: "dsh-standard-policy-v2" },
    )).toHaveLength(3);
  });
});
