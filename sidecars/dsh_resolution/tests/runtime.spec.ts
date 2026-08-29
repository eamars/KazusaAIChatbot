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
    continuation.operation_id = "op_continue";
    continuation.operation_payload_digest = "sha256:continue";
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

  it("accepts multi-call output and leaves terminal completion to a sole submit step", async () => {
    let executions = 0;
    const runtime = ResolutionSidecarRuntime.forTests([
      { calls: [{ name: "a" }, { name: "b" }] },
      { calls: [{ name: "a" }, { name: "b" }] },
      { calls: [{ name: "a" }, { name: "b" }] },
    ], () => { executions += 1; });
    expect((await runtime.open(validIntake(), "act_1", 1)).exhaust.kind).toBe("runtime_fault");
    expect(executions).toBe(6);
  });

  it("rejects stale activation and lease epochs before DSH mutation", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([{ wait: true }]);
    void runtime.open(validIntake(), "act_1", 2);
    await expect(runtime.cancel("res_1", "seg_1", "act_1", 1)).rejects.toThrow(/STALE_ACTIVATION_OR_LEASE/);
  });
});

describe("V2 runtime protocol", () => {
  it("executes normal multi-tool steps before sole terminal", async () => {
    const runtime = ResolutionSidecarRuntime.forTests([
      { calls: [{ name: "read_file", arguments: { path: "README.md" } }] },
      { name: "submit_resolution", arguments: {
        status: "resolved",
        summary: "done",
        findings: [],
        completed_subgoals: [],
        remaining_needs: [],
        clarification_request: null,
        approval_request: null,
        artifact_refs: [],
        warnings: [],
      } },
    ]);
    const result = await runtime.open({
      schema_version: "dsh_resolution_intake.v2",
      mode: "start",
      request_id: "request-v2",
      operation_id: "operation-v2",
      operation_payload_digest: "sha256:operation-v2",
      resolution_thread_id: "thread-v2",
      segment_id: "segment-v2",
      brain_conversation_ref: "chat:debug:one",
      workspace_root: "C:/workspace/project",
      route_digest: "sha256:route",
      model_input: { objective: "inspect and resolve", facts: [] },
      semantic_tool_authority: { catalog_digest: "sha256:catalog", token: "opaque" },
      interaction_authority: {
        issuer: "dsh-sidecar",
        scope_fingerprint: "sha256:scope",
        audience_fingerprint: "sha256:audience",
      },
    }, "activation-v2", 1);
    expect(result.exhaust.kind).toBe("terminal");
    expect(result.diagnostics.tool_executions).toBe(2);
    expect(result.diagnostics.terminal_tool_executions).toBe(1);
  });
});
