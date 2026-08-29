import { describe, expect, it } from "vitest";

import { ResolutionSidecarRuntime, compatibleSegment } from "../src/runtime.js";
import { validIntake, validRuntime } from "./contracts.spec.js";

describe("lifecycle", () => {
  it("rotates on workspace, route, catalog, issuer, scope, or conversation mismatch", () => {
    const current = validRuntime();
    const candidates = [
      { ...current, brain_conversation_ref: "chat:debug:other" },
      { ...current, workspace_root: "C:/workspace/other" },
      { ...current, route_digest: "sha256:other-route" },
      { ...current, semantic_tool_authority: { ...current.semantic_tool_authority, catalog_digest: "sha256:other-catalog" } },
      { ...current, interaction_authority: { ...current.interaction_authority, issuer: "other-issuer" } },
      { ...current, interaction_authority: { ...current.interaction_authority, scope_fingerprint: "sha256:other-scope" } },
    ];
    for (const candidate of candidates) {
      expect(compatibleSegment(current, candidate)).toBe(false);
    }
  });

  it("rejects duplicate live activation", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ wait: true }]);
    void runtime.open(validIntake(), "act_1", 1);
    const competing = validIntake();
    competing.operation_id = "op_competing";
    competing.operation_payload_digest = "sha256:competing";
    await expect(runtime.open(competing, "act_2", 1)).rejects.toThrow(/activation/);
  });

  it("renews a live lease and assigns a higher epoch after expired takeover", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ wait: true }]);
    void runtime.open(validIntake(), "act_1", 1);
    expect(runtime.renewLease("res_1", "seg_1", "act_1", 1).lease_epoch).toBe(1);
    const takeover = validIntake();
    takeover.operation_id = "op_takeover";
    takeover.operation_payload_digest = "sha256:takeover";
    expect((await runtime.continue(takeover, "act_2", 2)).lease_epoch).toBe(2);
  });

  it("cancels without deleting durable session", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ wait: true }]);
    void runtime.open(validIntake(), "act_1", 1);
    const canceled = await runtime.cancel("res_1", "seg_1", "act_1", 1);
    expect(canceled.disposition).toBe("canceled");
    expect(runtime.hasSession(canceled.session_id)).toBe(true);
  });
});

describe("V2 operation lifecycle", () => {
  it("interaction deferral is idempotent under operation replay", async () => {
    const { ResolutionSidecarRuntime } = await import("../src/runtime.js");
    const runtime = ResolutionSidecarRuntime.forTests([]);
    const request = {
      interaction_id: "interaction-v2",
      operation_id: "operation-v2",
      operation_payload_digest: "sha256:interaction",
      resolution_thread_id: "thread-v2",
      segment_id: "segment-v2",
      activation_id: "activation-v2",
      lease_epoch: 1,
    };
    const first = await runtime.deferInteraction(request);
    const second = await runtime.deferInteraction(request);
    expect(second).toEqual(first);
    expect(runtime.interactionExecutionCount("interaction-v2")).toBe(1);
    expect(first.disposition).toBe("checkpointed");
  });
});
