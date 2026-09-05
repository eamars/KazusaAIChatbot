import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const sidecarRoot = resolve(process.cwd());
const repositoryRoot = resolve(sidecarRoot, "..", "..");

describe("official Standard profile", () => {
  it("mounts installed Standard without a copied preset", async () => {
    const composition = await import("../src/composition.js");
    const resolved = await composition.resolveOfficialStandardFiles({ repositoryRoot });

    expect(resolved.basePath).toContain("node_modules");
    expect(resolved.standardPresetPath).toContain("node_modules");
    expect(resolved.standardAgentPath).toContain("node_modules");
    expect(resolved.standardPresetPath.startsWith(resolve(sidecarRoot, "config"))).toBe(false);
    expect(resolved.standardAgentPath.startsWith(resolve(sidecarRoot, "config"))).toBe(false);
    const tree = await composition.composeStandardProfile({
      repositoryRoot,
      workspaceRoot: repositoryRoot,
      routeConfig: {
        routeName: "kazusa-agentic-resolver",
        baseUrl: "http://localhost:8080/v1",
        credentialRef: "AGENTIC_RESOLVER_LLM_API_KEY",
        model: "qwen27b-5090",
        contextWindowTokens: 50176,
        maxCompletionTokens: 8192,
        thinkingEnabled: true,
      },
      sqlitePath: "C:/tmp/dsh/sessions.sqlite",
      semanticNames: [],
      localPluginPaths: {
        submitResolution: "C:/tmp/dsh/submit_resolution.js",
        semanticGateway: "C:/tmp/dsh/semantic_gateway.js",
        secretBroker: "C:/tmp/dsh/secret_broker.js",
        brainInteraction: "C:/tmp/dsh/brain_interaction.js",
      },
    });
    expect(tree.officialFiles.standardPresetPath).toBe(resolved.standardPresetPath);
    expect(tree.officialFiles.standardAgentPath).toBe(resolved.standardAgentPath);
    expect(tree.standardPresetCopied).toBe(false);
    expect(tree.compositionDump).toContain("kazusa-overlay");
  });

  it("adds Kazusa semantic tools without colliding with official Standard capabilities", async () => {
    const composition = await import("../src/composition.js");
    const nativeNames = [
      "read_file",
      "write_file",
      "pwsh",
      "web_search",
      "web_fetch",
      "kazusa_search_memories",
    ];
    const semanticNames = [
      "kazusa_search_conversation_history",
      "kazusa_read_conversation_entries",
      "kazusa_summarize_conversation_participants",
      "kazusa_search_memories",
      "kazusa_read_memories",
      "kazusa_remember_information",
      "kazusa_revise_memory",
      "kazusa_change_memory_lifecycle",
      "kazusa_find_people_by_name",
      "kazusa_read_person_profiles",
      "kazusa_recall_active_context",
      "kazusa_read_calendar_context",
      "kazusa_inspect_attached_media",
      "kazusa_inspect_public_media",
    ];
    const selected = composition.selectPublishedTools({ nativeNames, semanticNames });

    expect(new Set(selected.nativeNames)).toEqual(new Set(nativeNames));
    expect(new Set(selected.semanticNames)).toEqual(new Set(
      semanticNames.filter((name) => name !== "kazusa_search_memories"),
    ));
    expect(selected.omittedSemanticTools).toEqual([
      { name: "kazusa_search_memories", reason: "native_precedence" },
    ]);
    expect(new Set(selected.nativeNames).size).toBe(selected.nativeNames.length);
    expect(new Set(selected.semanticNames).size).toBe(selected.semanticNames.length);
    expect(selected.semanticNames).not.toContain("submit_resolution");

    const tree = composition.composeOverlayTree({
      nativeNames,
      semanticNames: selected.semanticNames,
      standardRoot: resolve(sidecarRoot, "node_modules", "@deepseek-ai", "dsh", "config", "agent-presets"),
      workspaceRoot: repositoryRoot,
      sqlitePath: "C:/tmp/dsh/sessions.sqlite",
      routeConfig: {
        routeName: "kazusa-agentic-resolver",
        baseUrl: "http://localhost:8080/v1",
        credentialRef: "AGENTIC_RESOLVER_LLM_API_KEY",
        model: "qwen27b-5090",
        contextWindowTokens: 50176,
        maxCompletionTokens: 8192,
        thinkingEnabled: true,
      },
      localPluginPaths: {
        submitResolution: "C:/tmp/dsh/submit_resolution.js",
        semanticGateway: "C:/tmp/dsh/semantic_gateway.js",
        secretBroker: "C:/tmp/dsh/secret_broker.js",
        brainInteraction: "C:/tmp/dsh/brain_interaction.js",
      },
    });
    expect(tree.rows.filter((row: { id: string }) => row.id === "sandbox-policy")).toHaveLength(0);
    expect(tree.rows.filter((row: { id: string }) => row.id === "llm-deepseek")[0]?.enabled).toBe(false);
    expect(tree.rows.filter((row: { id: string }) => row.id === "llm-pi-ai")[0]?.config).toMatchObject({
      providers: {
        "kazusa-agentic-resolver": {
          api: "openai-completions",
          baseURL: "http://localhost:8080/v1",
          apiKeyEnv: "AGENTIC_RESOLVER_LLM_API_KEY",
          compat: {
            supportsDeveloperRole: false,
            maxTokensField: "max_completion_tokens",
            thinkingFormat: "qwen-chat-template",
            chatTemplateKwargs: { enable_thinking: true },
          },
          reasoning: "high",
        },
      },
    });
  });

  it("publishes the Kazusa semantic capability catalog", async () => {
    const composition = await import("../src/composition.js");
    const semanticNames = [
      "kazusa_search_conversation_history",
      "kazusa_read_conversation_entries",
      "kazusa_summarize_conversation_participants",
      "kazusa_search_memories",
      "kazusa_read_memories",
      "kazusa_remember_information",
      "kazusa_revise_memory",
      "kazusa_change_memory_lifecycle",
      "kazusa_find_people_by_name",
      "kazusa_read_person_profiles",
      "kazusa_recall_active_context",
      "kazusa_read_calendar_context",
      "kazusa_inspect_attached_media",
      "kazusa_inspect_public_media",
    ];
    const selected = composition.selectPublishedTools({
      nativeNames: [],
      semanticNames,
    });

    expect(new Set(selected.semanticNames)).toEqual(new Set(semanticNames));
    expect(selected.semanticNames).toHaveLength(14);
    expect(selected.semanticNames).toContain("kazusa_inspect_public_media");
  });
});
