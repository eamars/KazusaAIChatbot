import { createHash } from "node:crypto";
import { createRequire } from "node:module";

import type { Context } from "@deepseek-ai/cordis";
import { canonicalJson } from "./contracts.js";

export { canonicalJson } from "./contracts.js";

const require = createRequire(import.meta.url);

interface PiAiConfig {
  providers: Record<string, Record<string, unknown>>;
}

/** The non-secret route descriptor shared with the Python configuration owner. */
export interface QwenRouteConfig {
  routeName: string;
  baseUrl: string;
  credentialRef: string;
  model: string;
  contextWindowTokens: number;
  maxCompletionTokens: number;
  thinkingEnabled: boolean;
}

/** Canonical, secret-free route fields included in the compatibility digest. */
export interface CanonicalRouteDescriptor {
  route_name: string;
  base_url: string;
  model: string;
  context_window_tokens: number;
  max_completion_tokens: number;
  thinking_enabled: boolean;
  supports_developer_role: false;
  max_tokens_field: "max_completion_tokens";
  thinking_format: "qwen-chat-template";
  chat_template_kwargs_enable_thinking: boolean;
  reasoning_effort: "high" | "off";
  output_mode: "text";
  compatibility_epoch: "qwen-openai-completions-v1";
  credential_reference: string;
}

export function canonicalRouteDescriptor(
  config: QwenRouteConfig,
): CanonicalRouteDescriptor {
  validateConfig(config);
  return {
    route_name: config.routeName,
    base_url: config.baseUrl,
    model: config.model,
    context_window_tokens: config.contextWindowTokens,
    max_completion_tokens: config.maxCompletionTokens,
    thinking_enabled: config.thinkingEnabled,
    supports_developer_role: false,
    max_tokens_field: "max_completion_tokens",
    thinking_format: "qwen-chat-template",
    chat_template_kwargs_enable_thinking: config.thinkingEnabled,
    reasoning_effort: config.thinkingEnabled ? "high" : "off",
    output_mode: "text",
    compatibility_epoch: "qwen-openai-completions-v1",
    credential_reference: config.credentialRef,
  };
}

export function routeDigest(config: QwenRouteConfig): string {
  return `sha256:${createHash("sha256").update(canonicalJson(canonicalRouteDescriptor(config)), "utf8").digest("hex")}`;
}

/** Build the exact installed pi-ai provider profile for this route. */
export function piAiConfig(config: QwenRouteConfig): PiAiConfig {
  const descriptor = canonicalRouteDescriptor(config);
  return {
    providers: {
      [config.routeName]: {
        api: "openai-completions",
        baseURL: config.baseUrl,
        apiKeyEnv: config.credentialRef,
        compat: {
          supportsDeveloperRole: descriptor.supports_developer_role,
          maxTokensField: descriptor.max_tokens_field,
          thinkingFormat: descriptor.thinking_format,
          chatTemplateKwargs: {
            enable_thinking: descriptor.chat_template_kwargs_enable_thinking,
          },
        },
        models: [{
          id: config.model,
          contextWindow: config.contextWindowTokens,
          maxTokens: config.maxCompletionTokens,
          input: ["text"],
          reasoningEfforts: config.thinkingEnabled
            ? { high: "high", off: null }
            : false,
        }],
        reasoning: descriptor.reasoning_effort,
      },
    },
  };
}

/** Mount the official DSH pi-ai adapter into a booted host context. */
export function applyQwenRoute(ctx: Context, config: QwenRouteConfig): void {
  const { apply } = require("@deepseek-ai/dsh-llm-pi-ai") as {
    apply: (context: Context, value: PiAiConfig) => void;
  };
  apply(ctx, piAiConfig(config));
}

function validateConfig(config: QwenRouteConfig): void {
  for (const [name, value] of Object.entries(config)) {
    if (typeof value === "string" && value.trim().length === 0) {
      throw new Error(`${name} is required`);
    }
  }
  if (!Number.isInteger(config.contextWindowTokens) || config.contextWindowTokens < 1) {
    throw new Error("contextWindowTokens is invalid");
  }
  if (!Number.isInteger(config.maxCompletionTokens) || config.maxCompletionTokens < 1) {
    throw new Error("maxCompletionTokens is invalid");
  }
  try {
    const url = new URL(config.baseUrl);
    if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error();
  } catch {
    throw new Error("baseUrl is invalid");
  }
}
