import { access, realpath } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { piAiConfig, type QwenRouteConfig } from "./model_route.js";

const APP_BOOT_MODULE = "@deepseek-ai/dsh-app-boot";

type PatchOptions = Record<string, unknown>;

interface ConfigDumpLayer {
  label: string;
  patches: PatchOptions[];
}

interface AppBootModule {
  composeEntries(layers: readonly PatchOptions[][]): PatchOptions[];
  loadOverlayPatches(binName: string, path: string): PatchOptions[];
  renderConfigDump(binName: string, path: string, layers: ConfigDumpLayer[]): string;
}

async function appBoot(): Promise<AppBootModule> {
  return await import(APP_BOOT_MODULE) as unknown as AppBootModule;
}

export interface OfficialStandardFiles {
  basePath: string;
  standardPresetPath: string;
  standardAgentPath: string;
}

export interface PublishedTools {
  nativeNames: string[];
  semanticNames: string[];
  omittedSemanticTools: Array<{ name: string; reason: "native_precedence" }>;
}

export interface OverlayRow {
  id: string;
  role: "base" | "insert";
  name?: string;
  enabled?: boolean;
  config?: Record<string, unknown>;
}

export interface OverlayTree {
  rows: OverlayRow[];
}

export interface StandardProfileTree {
  officialFiles: OfficialStandardFiles;
  bareModuleBaseUrl: string;
  rootPath: string;
  basePatches: readonly PatchOptions[];
  overlayPatches: readonly PatchOptions[];
  standardPresetCopied: false;
  overlay: OverlayTree;
  composedEntries: readonly Record<string, unknown>[];
  compositionDump: string;
}

export async function resolveOfficialStandardFiles(
  options: { repositoryRoot: string },
): Promise<OfficialStandardFiles> {
  const packageRoot = resolve(
    options.repositoryRoot,
    "sidecars",
    "dsh_resolution",
    "node_modules",
    "@deepseek-ai",
  );
  const files: OfficialStandardFiles = {
    basePath: resolve(packageRoot, "dsh-base", "cordis.patch.yml"),
    standardPresetPath: resolve(
      packageRoot,
      "dsh",
      "config",
      "agent-presets",
      "standard",
      "preset.yml",
    ),
    standardAgentPath: resolve(
      packageRoot,
      "dsh",
      "config",
      "agent-presets",
      "standard",
      "agent.cordis.yml",
    ),
  };
  await Promise.all(Object.values(files).map((path) => access(path)));
  return files;
}

export function selectPublishedTools(options: {
  nativeNames: readonly string[];
  semanticNames: readonly string[];
}): PublishedTools {
  const nativeNames = [...options.nativeNames];
  const native = new Set(nativeNames);
  const semanticNames: string[] = [];
  const omittedSemanticTools: PublishedTools["omittedSemanticTools"] = [];
  for (const name of options.semanticNames) {
    if (native.has(name)) {
      omittedSemanticTools.push({ name, reason: "native_precedence" });
    } else if (!semanticNames.includes(name)) {
      semanticNames.push(name);
    }
  }
  return { nativeNames, semanticNames, omittedSemanticTools };
}

function overlayRows(options: {
  semanticNames: readonly string[];
  standardRoot: string;
  sqlitePath: string;
  routeConfig: QwenRouteConfig;
  localPluginPaths: {
    submitResolution: string;
    semanticGateway: string;
    secretBroker: string;
    brainInteraction: string;
  };
}): OverlayRow[] {
  const route = options.routeConfig;
  if (route === undefined) throw new Error("canonical route configuration is required");
  const llmProvider = piAiConfig(route) as unknown as Record<string, unknown>;
  const local = options.localPluginPaths;
  const baseRows: OverlayRow[] = [
    { id: "hmr", role: "base", enabled: false },
    { id: "settings", role: "base", enabled: false },
    { id: "credentials", role: "base", enabled: false },
    { id: "llm-deepseek", role: "base", enabled: false },
    { id: "session-persistence-jsonl", role: "base", enabled: false },
    { id: "session-telemetry-otel", role: "base", enabled: false },
    {
      id: "llm-pi-ai",
      role: "base",
      config: llmProvider,
    },
    {
      id: "agent-default-model",
      role: "base",
      config: { provider: route.routeName, model: route.model },
    },
  ];
  return [
    ...baseRows,
    {
      id: "agent-presets-standard",
      role: "insert",
      name: "@deepseek-ai/dsh-agent-presets",
      config: {
        default: "standard",
        roots: [{ path: options.standardRoot, trust: "system" }],
        includeUserRoot: false,
      },
    },
    {
      id: "session-persistence-sqlite",
      role: "insert",
      name: "@deepseek-ai/dsh-session-persistence-sqlite",
      config: options.sqlitePath.length > 0 ? { path: options.sqlitePath } : {},
    },
    {
      id: "terminal-submit-resolution",
      role: "insert",
      name: local.submitResolution,
      config: { tool: "submit_resolution" },
    },
    {
      id: "kazusa-semantic-tools",
      role: "insert",
      name: local.semanticGateway,
      config: { names: [...options.semanticNames] },
    },
    {
      id: "host-credentials",
      role: "insert",
      name: local.secretBroker,
      config: { hostOnly: true },
    },
    {
      id: "brain-interaction-provider",
      role: "insert",
      name: local.brainInteraction,
      config: { hostOnly: true },
    },
  ];
}

export function composeOverlayTree(options: {
  nativeNames?: readonly string[];
  semanticNames: readonly string[];
  standardRoot: string;
  workspaceRoot: string;
  sqlitePath: string;
  routeConfig: NonNullable<Parameters<typeof overlayRows>[0]["routeConfig"]>;
  localPluginPaths: Parameters<typeof overlayRows>[0]["localPluginPaths"];
}): OverlayTree {
  // Native names are observed for collision reporting; upstream rows are
  // supplied by the installed base and are never synthesized here.
  void options.nativeNames;
  return {
    rows: overlayRows({
      semanticNames: options.semanticNames,
      standardRoot: options.standardRoot,
      sqlitePath: options.sqlitePath,
      routeConfig: options.routeConfig,
      localPluginPaths: options.localPluginPaths,
    }),
  };
}

export async function composeStandardProfile(options: {
  repositoryRoot: string;
  workspaceRoot: string;
  routeConfig: NonNullable<Parameters<typeof overlayRows>[0]["routeConfig"]>;
  sqlitePath: string;
  semanticNames: readonly string[];
  localPluginPaths: Parameters<typeof overlayRows>[0]["localPluginPaths"];
}): Promise<StandardProfileTree> {
  const officialFiles = await resolveOfficialStandardFiles(options);
  const { composeEntries, loadOverlayPatches, renderConfigDump } = await appBoot();
  const rootPath = resolve(options.repositoryRoot, "sidecars", "dsh_resolution", "config", "root.cordis.yml");
  const basePatches = loadOverlayPatches("dsh-resolution", officialFiles.basePath);
  const semanticNames = options.semanticNames;
  const overlay = composeOverlayTree({
    semanticNames,
    standardRoot: resolve(dirname(officialFiles.standardPresetPath), ".."),
    workspaceRoot: options.workspaceRoot,
    sqlitePath: options.sqlitePath,
    routeConfig: options.routeConfig,
    localPluginPaths: options.localPluginPaths,
  });
  const overlayPatches = overlay.rows
    .filter((row) => row.role === "base")
    .map((row) => ({
      id: row.id,
      ...(row.enabled === false ? { disabled: true } : {}),
      ...(row.config === undefined || Object.keys(row.config).length === 0 ? {} : { config: row.config }),
    })) as PatchOptions[];
  overlayPatches.push({
    insert: overlay.rows
      .filter((row) => row.role === "insert")
      .map((row) => ({
        id: row.id,
        name: row.name ?? row.id,
        ...(row.config === undefined ? {} : { config: row.config }),
      })),
  });
  const composedEntries = composeEntries([basePatches, overlayPatches]);
  const layers: ConfigDumpLayer[] = [
    { label: officialFiles.basePath, patches: basePatches },
    { label: "kazusa-overlay", patches: overlayPatches },
  ];
  const compositionDump = renderConfigDump("dsh-resolution", rootPath, layers);
  return {
    officialFiles,
    bareModuleBaseUrl: `${pathToFileURL(await realpath(resolve(
      options.repositoryRoot,
      "sidecars",
      "dsh_resolution",
      "node_modules",
      "@deepseek-ai",
      "dsh",
    ))).href.replace(/\/$/u, "")}/`,
    rootPath,
    basePatches,
    overlayPatches,
    standardPresetCopied: false,
    overlay,
    composedEntries: composedEntries as unknown as readonly Record<string, unknown>[],
    compositionDump,
  };
}
