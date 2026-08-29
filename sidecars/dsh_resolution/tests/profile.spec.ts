import { describe, expect, it } from "vitest";

import { assertCompatibleDependencyGraph, buildProfile } from "../src/profile.js";

const route = {
  routeName: "kazusa-agentic-resolver",
  baseUrl: "http://localhost:8080/v1",
  credentialRef: "AGENTIC_RESOLVER_LLM_API_KEY",
  model: "qwen27b-5090",
  contextWindowTokens: 50176,
  maxCompletionTokens: 8192,
  thinkingEnabled: true,
};

const profileDependencies = {
  route,
  hostSecrets: {
    AGENTIC_RESOLVER_LLM_API_KEY: "model-secret",
    KAZUSA_DSH_TOOL_GATEWAY_SECRET: "gateway-secret",
  },
  nativeEnvironment: { DSH_HOME: "C:/tmp/dsh-home" },
  brainProvider: { handle: async () => ({ kind: "reject" }) },
  semanticInvoker: async () => ({
    schema_version: "kazusa_semantic_capability_result.v1",
    status: "unavailable",
    entities: [],
    page: { has_more: false, next_page_ref: null },
    evidence: [],
    mutation: null,
    error: { code: "SEMANTIC_WORKER_UNAVAILABLE", safe_message: "The semantic worker is unavailable." },
  }),
};

describe("profile", () => {
  it("builds only the resolver profile through the strict profile factory", async () => {
    const profile = await buildProfile("kazusa-resolver-standard-v2", {
      model: "qwen27b-5090",
      dataRoot: "C:/tmp/dsh",
      repositoryRoot: "C:/workspace/kazusa_ai_chatbot",
      workspaceRoot: "C:/workspace/kazusa_ai_chatbot",
      ...profileDependencies,
    });
    expect(profile.id).toBe("kazusa-resolver-standard-v2");
    expect(profile.semanticTools).toContain("submit_resolution");
    expect(profile.semanticTools).toContain("kazusa_search_memories");
    expect(profile.catalog.native_names).toContain("web_search");
    expect(profile.catalog.native_names).toContain("pwsh");
    expect(profile.catalog.names).toContain("web_search");
    expect(profile.catalog.names).toContain("submit_resolution");
    expect(profile.catalog.descriptions_stripped).toBe(true);
    expect(profile.catalog.omitted_semantic_tools).toEqual([]);
    expect(profile.composedServices).toContain("session-checkpoint-policy");
    expect(profile.composedServices).toContain("agent-loop");
    await profile.close();
    await expect(buildProfile("unknown", {
      model: "m",
      dataRoot: "C:/tmp/dsh",
      repositoryRoot: "C:/workspace/kazusa_ai_chatbot",
      workspaceRoot: "C:/workspace/kazusa_ai_chatbot",
      ...profileDependencies,
    })).rejects.toThrow(/profile/);
  }, 30_000);

  it("fails startup for an incompatible DSH dependency graph", () => {
    expect(() => assertCompatibleDependencyGraph({ "@deepseek-ai/dsh-agent": "0.1.1-rc.1" })).toThrow(/dependency/);
    expect(() => assertCompatibleDependencyGraph({ "@deepseek-ai/dsh-agent": "0.1.1-rc.2" })).not.toThrow();
  });

  it("rejects an intake whose route digest differs before session creation", async () => {
    const profile = await buildProfile("kazusa-resolver-standard-v2", {
      model: "qwen27b-5090",
      dataRoot: "C:/tmp/dsh",
      repositoryRoot: "C:/workspace/kazusa_ai_chatbot",
      workspaceRoot: "C:/workspace/kazusa_ai_chatbot",
      ...profileDependencies,
    });
    try {
      await expect(profile.activate("resolution.open", {
        schema_version: "dsh_resolution_intake.v2",
        mode: "start",
        request_id: "req-route-mismatch",
        operation_id: "op-route-mismatch",
        operation_payload_digest: "sha256:route-mismatch",
        resolution_thread_id: "thread-route-mismatch",
        segment_id: "segment-route-mismatch",
        brain_conversation_ref: "chat:debug:route-mismatch",
        workspace_root: "C:/workspace/kazusa_ai_chatbot",
        route_digest: "sha256:wrong-route",
        model_input: { objective: "test", facts: [] },
        semantic_tool_authority: { catalog_digest: "sha256:catalog", token: "opaque" },
        interaction_authority: {
          issuer: "dsh-sidecar-test",
          scope_fingerprint: "sha256:scope",
          audience_fingerprint: "sha256:audience",
        },
      }, "activation-route-mismatch", 1)).rejects.toThrow("ROUTE_DIGEST_MISMATCH");
    } finally {
      await profile.close();
    }
  });
});

describe("V2 profile invariants", () => {
  it("rejects V1 epoch and verifies installed base and standard digests", async () => {
    const profileModule = await import("../src/profile.js");
    const build = profileModule.buildProfile as unknown as (
      id: string,
      options: Record<string, unknown>,
    ) => Promise<Record<string, any>>;
    await expect(build("kazusa-resolver-v1", {
      model: "qwen27b-5090",
      dataRoot: "C:/tmp/dsh",
      workspaceRoot: "C:/workspace/project",
      ...profileDependencies,
    })).rejects.toThrow(/V1|epoch|profile/i);
    const profile = await build("kazusa-resolver-standard-v2", {
      model: "qwen27b-5090",
      dataRoot: "C:/tmp/dsh",
      repositoryRoot: "C:/workspace/kazusa_ai_chatbot",
      workspaceRoot: "C:/workspace/project",
      ...profileDependencies,
    });
    expect(profile.profileVersion).toBe("kazusa-resolver-standard-v2");
    expect(profile.officialDigests).toEqual({
      base: "sha256:9870a518274194c0e1ebd870cee2737fbc2ffc04ae36887871ffe6fcf74beac1",
      standardPreset: "sha256:3c61b4ce68e5dd5cb2c099693fdcb30b91d5f22bbbef546e233321b0fa68f0e4",
      standardAgent: "sha256:fa14feb98daef20b810fef30bb7239a89a786de3c45c602b37743f7100d9a5af",
    });
    expect(profile.standardPresetPath).toContain("node_modules");
    expect(profile.standardPresetPath).toContain("@deepseek-ai");
    expect(profile.standardPresetPath).toContain("dsh");
    await profile.close();
  });
});
