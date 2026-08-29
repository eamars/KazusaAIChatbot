import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const sidecarRoot = resolve(process.cwd());
const repositoryRoot = resolve(sidecarRoot, "..", "..");

async function sha256(path: string): Promise<string> {
  const bytes = await readFile(path);
  return `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
}

describe("official Standard profile", () => {
  it("mounts installed standard without a copied preset", async () => {
    const composition = await import("../src/composition.js");
    const resolved = await composition.resolveOfficialStandardFiles({ repositoryRoot });

    expect(resolved.basePath).toContain("node_modules");
    expect(resolved.standardPresetPath).toContain("node_modules");
    expect(resolved.standardAgentPath).toContain("node_modules");
    expect(resolved.standardPresetPath.startsWith(resolve(sidecarRoot, "config"))).toBe(false);
    expect(resolved.standardAgentPath.startsWith(resolve(sidecarRoot, "config"))).toBe(false);
    await expect(sha256(resolved.basePath)).resolves.toBe(
      "sha256:9870a518274194c0e1ebd870cee2737fbc2ffc04ae36887871ffe6fcf74beac1",
    );
    await expect(sha256(resolved.standardPresetPath)).resolves.toBe(
      "sha256:3c61b4ce68e5dd5cb2c099693fdcb30b91d5f22bbbef546e233321b0fa68f0e4",
    );
    await expect(sha256(resolved.standardAgentPath)).resolves.toBe(
      "sha256:fa14feb98daef20b810feF30bb7239a89a786de3c45c602b37743f7100d9a5af".toLowerCase(),
    );

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
    for (const rowId of [
      "hmr",
      "settings",
      "credentials",
      "llm-deepseek",
      "session-persistence-jsonl",
      "session-telemetry-otel",
      "llm-pi-ai",
      "agent-default-model",
      "agent-presets-standard",
      "session-persistence-sqlite",
      "terminal-submit-resolution",
      "kazusa-semantic-tools",
      "host-credentials",
      "brain-interaction-provider",
    ]) {
      expect(tree.composedEntries.filter((entry) => entry.id === rowId)).toHaveLength(1);
    }
    expect(tree.compositionDump).toContain("kazusa-overlay");
  });

  it("catalog retains the complete pinned Standard set and adds only noncolliding Kazusa semantic tools", async () => {
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
    ];
    const selected = composition.selectPublishedTools({ nativeNames, semanticNames });

    expect(selected.nativeNames).toEqual(nativeNames);
    expect(selected.semanticNames).toEqual(
      semanticNames.filter((name) => name !== "kazusa_search_memories"),
    );
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
    for (const rowId of [
      "hmr",
      "settings",
      "credentials",
      "llm-deepseek",
      "session-persistence-jsonl",
      "session-telemetry-otel",
      "llm-pi-ai",
      "agent-default-model",
    ]) {
      expect(tree.rows.filter((row: { id: string }) => row.id === rowId)).toHaveLength(1);
    }
    expect(tree.rows.filter((row: { id: string }) => row.id === "sandbox-policy")).toHaveLength(0);
    expect(tree.rows.filter((row: { role: string }) => row.role === "insert")).toHaveLength(6);
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
});
