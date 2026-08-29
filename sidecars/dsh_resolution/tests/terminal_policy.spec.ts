import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

describe("autonomous multi-tool runtime", () => {
  it("applies no Kazusa step call byte or deadline budget and accepts only sole terminal submit", async () => {
    const runtime = await import("../src/runtime.js");
    const source = await readFile(resolve(process.cwd(), "src", "runtime.ts"), "utf8");
    const profileSource = await readFile(resolve(process.cwd(), "src", "profile.ts"), "utf8");
    for (const forbidden of [
      "max_model_steps",
      "max_tool_calls",
      "max_tool_bytes",
      "soft_deadline_at",
      "hard_deadline_at",
      "CORRECTION_MESSAGE",
    ]) {
      expect(source).not.toContain(forbidden);
    }
    expect(profileSource).toContain("maxTokens: route.maxCompletionTokens");
    expect(profileSource).not.toContain("maxTokens: 4096");
    expect(profileSource).toContain('name: "kazusa:terminal-contract"');
    expect(profileSource).toContain(
      "The only valid terminal response is one submit_resolution tool call.",
    );

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
