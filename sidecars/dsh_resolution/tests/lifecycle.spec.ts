import { describe, expect, it } from "vitest";

import { ResolutionSidecarRuntime, compatibleSegment } from "../src/runtime.js";
import { validIntake, validRuntime } from "./contracts.spec.js";

describe("lifecycle", () => {
  it("rotates on scope audience profile release store model catalog or policy mismatch", () => {
    const current = validRuntime();
    for (const key of ["scope_fingerprint", "audience_fingerprint", "resolver_profile_version", "dsh_release", "session_store_epoch", "model_route", "tool_catalog_digest", "policy_epoch"] as const) {
      expect(compatibleSegment(current, { ...current, [key]: "different" })).toBe(false);
    }
  });

  it("rejects duplicate live activation", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ wait: true }]);
    void runtime.open(validIntake(), "act_1", 1);
    const competing = validIntake();
    competing.runtime.operation_id = "op_competing";
    competing.runtime.operation_payload_digest = "sha256:competing";
    await expect(runtime.open(competing, "act_2", 1)).rejects.toThrow(/activation/);
  });

  it("renews a live lease and assigns a higher epoch after expired takeover", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ wait: true }]);
    void runtime.open(validIntake(), "act_1", 1);
    expect(runtime.renewLease("res_1", "seg_1", "act_1", 1).lease_epoch).toBe(1);
    const takeover = validIntake();
    takeover.runtime.operation_id = "op_takeover";
    takeover.runtime.operation_payload_digest = "sha256:takeover";
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
