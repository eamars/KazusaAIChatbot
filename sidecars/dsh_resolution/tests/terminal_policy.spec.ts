import { describe, expect, it } from "vitest";

describe("autonomous multi-tool runtime", () => {
  it("accepts multi-tool work and a sole terminal submission", async () => {
    const runtime = await import("../src/runtime.js");

    const policy = runtime.autonomousRuntimePolicy();
    expect(policy).not.toHaveProperty("max_model_steps");
    expect(policy).not.toHaveProperty("max_tool_calls");
    expect(policy).not.toHaveProperty("max_tool_bytes");
    expect(policy).not.toHaveProperty("soft_deadline_at");
    expect(policy).not.toHaveProperty("hard_deadline_at");

    expect(runtime.evaluateAssistantToolStep([
      { name: "read_file", arguments: { path: "README.md" } },
      { name: "kazusa_search_memories", arguments: { query: "context" } },
    ])).toEqual({ accepted: true, terminal: false });
    expect(runtime.evaluateAssistantToolStep([
      { name: "submit_resolution", arguments: { status: "resolved" } },
    ])).toEqual({ accepted: true, terminal: true });
    expect(runtime.evaluateAssistantToolStep([
      { name: "submit_resolution", arguments: { status: "resolved" } },
      { name: "read_file", arguments: { path: "README.md" } },
    ])).toMatchObject({ accepted: false, terminal: false, result: { kind: "runtime_fault" } });
  });
});
