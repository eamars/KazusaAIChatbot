import { createServer, type Server } from "node:http";

import { createUserMessage, LlmRuntime, ReasoningEffortId } from "@deepseek-ai/dsh-llm";
import { Context } from "@deepseek-ai/cordis";
import { afterEach, describe, expect, it } from "vitest";

const servers: Server[] = [];

afterEach(async () => {
  for (const server of servers.splice(0)) {
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

describe("Qwen route", () => {
  it("uses the installed pi-ai adapter for the wire contract and tool replay", async () => {
    const requests: Record<string, unknown>[] = [];
    const server = createServer((request, response) => {
      const chunks: Buffer[] = [];
      request.on("data", (chunk: Buffer) => chunks.push(chunk));
      request.on("end", () => {
        requests.push(JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>);
        response.writeHead(200, { "content-type": "text/event-stream" });
        response.end([
          "data: {\"id\":\"response-1\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call-1\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
          "data: {\"id\":\"response-1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
          "data: [DONE]\n\n",
        ].join(""));
      });
    });
    servers.push(server);
    await new Promise<void>((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", () => resolve());
    });
    const address = server.address();
    if (address === null || typeof address === "string") throw new Error("test server did not bind");

    const modelRoute = await import("../src/model_route.js");
    const config = {
      routeName: "kazusa-agentic-resolver",
      baseUrl: `http://127.0.0.1:${address.port}/v1`,
      credentialRef: "AGENTIC_RESOLVER_LLM_API_KEY",
      model: "qwen27b-5090",
      contextWindowTokens: 50176,
      maxCompletionTokens: 8192,
      thinkingEnabled: true,
    };
    expect(modelRoute.canonicalRouteDescriptor(config)).toMatchObject({
      route_name: config.routeName,
      credential_reference: config.credentialRef,
      max_tokens_field: "max_completion_tokens",
      thinking_format: "qwen-chat-template",
      chat_template_kwargs_enable_thinking: true,
      reasoning_effort: "high",
      output_mode: "text",
    });

    const context = new Context();
    const llmFiber = context.plugin(LlmRuntime);
    await llmFiber;
    context.provide("credentials", {
      resolve: async () => ({ value: "fake-key", source: "test" }),
      readRecord: async () => undefined,
      listRecords: async () => [],
      modifyRecord: async () => undefined,
      deleteRecord: async () => undefined,
    });
    modelRoute.applyQwenRoute(context, config);

    const first = context.llm.stream({
      provider: config.routeName,
      model: config.model,
      system: "You are a bounded resolver.",
      messages: [createUserMessage({
        content: [{ type: "text", text: "Inspect the README." }],
        source: { kind: "user" },
      })],
      tools: [{
        name: "read_file",
        description: "Read a workspace file",
        parameters: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
      }],
      maxTokens: config.maxCompletionTokens,
      reasoningEffort: ReasoningEffortId("high"),
    });
    const firstChunks = [];
    for await (const chunk of first) firstChunks.push(chunk);
    expect(firstChunks.some((chunk) => chunk.type === "tool-call-delta")).toBe(true);

    expect(requests[0]).toMatchObject({
      model: config.model,
      messages: [
        { role: "system", content: "You are a bounded resolver." },
        { role: "user", content: "Inspect the README." },
      ],
      max_completion_tokens: config.maxCompletionTokens,
      chat_template_kwargs: { enable_thinking: true },
      tools: [{ type: "function", function: { name: "read_file" } }],
      stream: true,
    });
    expect(requests[0]).not.toHaveProperty("max_tokens");
    expect(requests[0]).not.toHaveProperty("developer");

    await llmFiber.dispose();
  });

  it("keeps Python and TypeScript route descriptor digests canonical", async () => {
    const modelRoute = await import("../src/model_route.js");
    const config = {
      routeName: "kazusa-agentic-resolver",
      baseUrl: "http://localhost:8080/v1",
      credentialRef: "AGENTIC_RESOLVER_LLM_API_KEY",
      model: "qwen27b-5090",
      contextWindowTokens: 50176,
      maxCompletionTokens: 8192,
      thinkingEnabled: true,
    };
    expect(modelRoute.routeDigest(config)).toBe(
      "sha256:61a1cf315a6b041aec1ed8946d05d2cfeb4b06198d2a19b5d80cdde47d4cd2f8",
    );
  });
});
