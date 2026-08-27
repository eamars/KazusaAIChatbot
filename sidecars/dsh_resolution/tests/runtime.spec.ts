import { describe, expect, it } from "vitest";

import { ResolutionSidecarRuntime } from "../src/runtime.js";
import { validIntake } from "./contracts.spec.js";

describe("runtime", () => {
  it("opens an agent captures terminal submit and disposes activation", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ name: "submit_resolution", arguments: { status: "resolved", summary: "done", findings: [], completed_subgoals: [], remaining_needs: [], clarification_request: null, approval_request: null, artifact_refs: [], warnings: [] } }]);
    const result = await runtime.open(validIntake(), "act_1", 1);
    expect(result.exhaust.kind).toBe("terminal");
    expect((await runtime.inspect("op_1")).disposition).toBe("terminal");
    await runtime.disposeActivation("res_1", "seg_1", "act_1", 1);
  });

  it("checkpoints at pre-step and resumes the same DSH session", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ wait: true }]);
    const pending = runtime.open(validIntake(), "act_1", 1);
    const checkpoint = await runtime.requestCheckpoint("res_1", "seg_1", "act_1", 1);
    expect(checkpoint.disposition).toBe("checkpointed");
    await pending;
    const continuation = validIntake();
    continuation.runtime.operation_id = "op_continue";
    continuation.runtime.operation_payload_digest = "sha256:continue";
    expect((await runtime.continue(continuation, "act_2", 2)).session_id).toBe(checkpoint.session_id);
  });

  it("amends with steer queues continuation and cancels safely", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ wait: true }]);
    void runtime.open(validIntake(), "act_1", 1);
    expect((await runtime.amend("res_1", "seg_1", "act_1", 1, { objective: "changed" })).disposition).toBe("admitted_active");
    expect((await runtime.cancel("res_1", "seg_1", "act_1", 1)).disposition).toBe("canceled");
  });

  it("accepts one regenerated action after a zero-call prose-only or empty step", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ text: "prose" }, { name: "submit_resolution", arguments: { status: "resolved", summary: "done", findings: [], completed_subgoals: [], remaining_needs: [], clarification_request: null, approval_request: null, artifact_refs: [], warnings: [] } }]);
    expect((await runtime.open(validIntake(), "act_1", 1)).exhaust.kind).toBe("terminal");
  });

  it("returns action contract exhausted after repeated zero-call prose-only or empty steps", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ text: "a" }, {}, { text: "b" }]);
    expect((await runtime.open(validIntake(), "act_1", 1)).exhaust).toMatchObject({ kind: "runtime_fault", fault: { code: "RESOLVER_ACTION_CONTRACT_EXHAUSTED" } });
  });

  it("executes zero tool bodies for multi-call output and exhausts the shared correction budget", async () => {
    let executions = 0;
    const runtime = ResolutionSidecarRuntime.forTests([
      { calls: [{ name: "a" }, { name: "b" }] },
      { calls: [{ name: "a" }, { name: "b" }] },
      { calls: [{ name: "a" }, { name: "b" }] },
    ], () => { executions += 1; });
    expect((await runtime.open(validIntake(), "act_1", 1)).exhaust.kind).toBe("runtime_fault");
    expect(executions).toBe(0);
  });

  it("rejects stale activation and lease epochs before DSH mutation", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ wait: true }]);
    void runtime.open(validIntake(), "act_1", 2);
    await expect(runtime.cancel("res_1", "seg_1", "act_1", 1)).rejects.toThrow(/STALE_ACTIVATION_OR_LEASE/);
  });
});
