import { describe, expect, it } from "vitest";

import { commitTerminalResolution, replayTerminalExhaust } from "../src/submit_resolution.js";
import type { EvidenceReference } from "../src/evidence.js";
import { validRuntime, validSubmit } from "./contracts.spec.js";

class Store {
  events: Array<Record<string, unknown>> = [];
  flushed = false;
  async append(event: Record<string, unknown>) { this.events.push(event); return this.events.length; }
  async flush() { this.flushed = true; }
}

function evidenceReference(
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
    policyEpoch: "dsh-standard-policy-v2",
    tool_name: "fixture_evidence",
    source_kind: "fixture",
    source_id: `source-${evidenceId}`,
    content_digest: contentDigest,
  };
}

describe("submit_resolution", () => {
  it("commits the complete terminal receipt before returning exhaust", async () => {
    const store = new Store();
    const exhaust = await commitTerminalResolution(store, validRuntime(), "act_1", 1, "call_1", validSubmit(), []);
    expect(store.flushed).toBe(true);
    expect(store.events[0]?.meta).toHaveProperty("kazusa.kind", "terminal_resolution_v2");
    expect(exhaust.kind).toBe("terminal");
  });

  it("projects tool provenance only through the canonical nested field", async () => {
    const evidence = {
      schema_version: "evidence_receipt.v2" as const,
      evidence_id: "evidence-1",
      threadId: "res_1",
      segmentId: "seg_1",
      scopeFingerprint: "sha256:scope",
      audienceFingerprint: "sha256:audience",
      policyEpoch: "dsh-standard-policy-v2",
      tool_name: "kazusa_search_memories",
      source_kind: "semantic_memory",
      source_id: "opaque-memory-ref",
      content_digest: "sha256:content",
    };
    const result = await commitTerminalResolution(
      new Store(),
      validRuntime(),
      "act_1",
      1,
      "call_1",
      validSubmit(),
      [evidence],
    ) as Record<string, any>;
    expect(result.evidence[0]).toMatchObject({
      provenance: { tool_name: "kazusa_search_memories" },
    });
    expect(result.evidence[0]).not.toHaveProperty("tool_name");
  });

  it("replays exact terminal exhaust after restart without model execution", async () => {
    const store = new Store();
    const first = await commitTerminalResolution(store, validRuntime(), "act_1", 1, "call_1", validSubmit(), []);
    expect(replayTerminalExhaust(store.events)).toEqual(first);
  });

  it("rejects missing or invalid terminal receipts as runtime faults", () => {
    expect(replayTerminalExhaust([])).toMatchObject({ kind: "runtime_fault", fault: { code: "TERMINAL_RECEIPT_MISSING" } });
    expect(replayTerminalExhaust([{ type: "tool/result", meta: { kazusa: { kind: "terminal_resolution_v1" } } }]))
      .toMatchObject({ kind: "runtime_fault" });
  });
});

describe("V2 submit_resolution", () => {
  it("projects 66 evidence references to the latest 64 in order", async () => {
    const evidence = Array.from({ length: 66 }, (_, index) => (
      evidenceReference(`evidence-${index + 1}`)
    ));
    const result = await commitTerminalResolution(
      new Store(),
      validRuntime(),
      "activation-v2",
      1,
      "call-many",
      validSubmit(),
      evidence,
    );
    const projected = (result as Record<string, any>).evidence as Array<Record<string, any>>;

    expect(projected).toHaveLength(64);
    expect(projected.map((item) => item.evidence_id)).toEqual(
      Array.from({ length: 64 }, (_, index) => `evidence-${index + 3}`),
    );
  });

  it("keeps the most recent duplicate and emits unique evidence ids", async () => {
    const result = await commitTerminalResolution(
      new Store(),
      validRuntime(),
      "activation-v2",
      1,
      "call-duplicate",
      validSubmit(),
      [
        evidenceReference("evidence-1"),
        evidenceReference("evidence-duplicate", "sha256:old"),
        evidenceReference("evidence-2"),
        evidenceReference("evidence-duplicate", "sha256:new"),
      ],
    );
    const projected = (result as Record<string, any>).evidence as Array<Record<string, any>>;

    expect(projected.map((item) => item.evidence_id)).toEqual([
      "evidence-1", "evidence-2", "evidence-duplicate",
    ]);
    expect(projected[2]?.content_digest).toBe("sha256:new");
    expect(new Set(projected.map((item) => item.evidence_id)).size)
      .toBe(projected.length);
  });

  it("validates unauthorized evidence before applying the retention window", async () => {
    const unauthorized = { ...evidenceReference("unauthorized"), segmentId: "foreign-segment" };
    const evidence = [
      unauthorized,
      ...Array.from({ length: 65 }, (_, index) => evidenceReference(`evidence-${index + 1}`)),
    ];
    const result = await commitTerminalResolution(
      new Store(),
      validRuntime(),
      "activation-v2",
      1,
      "call-unauthorized",
      validSubmit(),
      evidence,
    );

    expect(result).toMatchObject({
      kind: "runtime_fault",
      fault: { code: "EVIDENCE_AUTHORITY_MISMATCH" },
    });
  });

  it("uses the same bounded projection for commit and restart replay", async () => {
    const evidence = Array.from({ length: 66 }, (_, index) => (
      evidenceReference(`evidence-${index + 1}`)
    ));
    const store = new Store();
    const committed = await commitTerminalResolution(
      store,
      validRuntime(),
      "activation-v2",
      1,
      "call-replay",
      validSubmit(),
      evidence,
    );
    const replayed = replayTerminalExhaust(store.events, evidence);

    expect((replayed as Record<string, any>).evidence)
      .toEqual((committed as Record<string, any>).evidence);
  });

  it("rejects foreign segment evidence and out-of-workspace artifacts", async () => {
    const submitModule = await import("../src/submit_resolution.js");
    const commit = submitModule.commitTerminalResolution as unknown as (
      store: Store,
      runtime: Record<string, unknown>,
      activationId: string,
      leaseEpoch: number,
      callId: string,
      submit: Record<string, unknown>,
      evidence: unknown[],
    ) => Promise<Record<string, any>>;
    const runtime = {
      ...validRuntime(),
      workspace_root: "C:/workspace/project",
    };
    const foreignEvidence = {
      schema_version: "evidence_receipt.v2",
      resolution_thread_id: "res-foreign",
      segment_id: "segment-foreign",
      scope_fingerprint: "sha256:scope",
      audience_fingerprint: "sha256:audience",
      policy_epoch: "dsh-standard-policy-v2",
      evidence_id: "foreign-1",
      source_kind: "semantic",
      semantic_ref: "opaque-ref",
      content_digest: "sha256:content",
      provenance: { tool_name: "kazusa_search_memories" },
    };
    const foreignResult = await commit(
      new Store(),
      runtime,
      "activation-v2",
      1,
      "call-foreign",
      validSubmit(),
      [foreignEvidence],
    );
    expect(foreignResult.kind).toBe("runtime_fault");
    expect(foreignResult.fault.code).toMatch(/EVIDENCE|SEGMENT/i);

    const artifactResult = await commit(
      new Store(),
      runtime,
      "activation-v2",
      1,
      "call-artifact",
      { ...validSubmit(), artifact_refs: ["C:/outside/artifact.txt"] },
      [],
    );
    expect(artifactResult.kind).toBe("runtime_fault");
    expect(artifactResult.fault.code).toMatch(/WORKSPACE|ARTIFACT/i);
  });
});
