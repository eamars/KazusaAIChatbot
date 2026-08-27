import { describe, expect, it } from "vitest";

import { assertCompatibleDependencyGraph, buildProfile } from "../src/profile.js";

describe("profile", () => {
  it("builds only the resolver profile through the strict profile factory", async () => {
    const profile = await buildProfile("kazusa-resolver-v1", { model: "resolver-model", dataRoot: "C:/tmp/dsh" });
    expect(profile.id).toBe("kazusa-resolver-v1");
    expect(profile.semanticTools).toEqual(["submit_resolution"]);
    expect(profile.composedServices).toContain("session-checkpoint-policy");
    expect(profile.composedServices).toContain("agent-loop");
    await profile.close();
    await expect(buildProfile("unknown", { model: "m", dataRoot: "C:/tmp/dsh" })).rejects.toThrow(/profile/);
  });

  it("fails startup for an incompatible DSH dependency graph", () => {
    expect(() => assertCompatibleDependencyGraph({ "@deepseek-ai/dsh-agent": "0.1.1-rc.1" })).toThrow(/dependency/);
    expect(() => assertCompatibleDependencyGraph({ "@deepseek-ai/dsh-agent": "0.1.1-rc.2" })).not.toThrow();
  });
});
