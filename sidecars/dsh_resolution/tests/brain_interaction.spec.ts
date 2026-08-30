import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { describe, expect, it, vi } from "vitest";

function requestFields(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: "dsh_brain_interaction.v2",
    interaction_id: "interaction-v2",
    kind: "question",
    operation_id: "operation-v2",
    operation_payload_digest: "sha256:operation",
    resolution_thread_id: "thread-v2",
    segment_id: "segment-v2",
    activation_id: "activation-v2",
    lease_epoch: 1,
    dsh_call_id: "call-v2",
    tool_name: null,
    arguments_digest: "sha256:arguments",
    transient_detail: "{\"questions\":[]}",
    brain_conversation_ref: "chat:debug:one",
    platform: "debug",
    platform_channel_id: "channel-v2",
    global_user_id: "user-v2",
    scope_fingerprint: "sha256:scope",
    audience_fingerprint: "sha256:audience",
    profile_version: "kazusa-resolver-standard-v2",
    catalog_digest: "sha256:catalog",
    model_route_digest: "sha256:route",
    workspace_fingerprint: "sha256:workspace",
    policy_epoch: "dsh-standard-policy-v2",
    issued_reference_digest: "sha256:issued",
    issuer: "dsh-sidecar",
    nonce: "nonce-v2",
    issued_at: "2026-08-28T00:00:00.000Z",
    expires_at: "2026-08-28T00:05:00.000Z",
    ...overrides,
  };
}

function grantFor(request: Record<string, unknown>): Record<string, unknown> {
  return {
    schema_version: "dsh_brain_interaction.v2",
    interaction_id: request.interaction_id,
    resolution_thread_id: request.resolution_thread_id,
    segment_id: request.segment_id,
    activation_id: request.activation_id,
    lease_epoch: request.lease_epoch,
    tool_name: request.tool_name,
    arguments_digest: request.arguments_digest,
    workspace_fingerprint: request.workspace_fingerprint,
    scope_fingerprint: request.scope_fingerprint,
    policy_epoch: request.policy_epoch,
    grant_status: "consumed",
    issued_at: request.issued_at,
    expires_at: request.expires_at,
  };
}

function responseFor(
  interaction: typeof import("../src/brain_interaction.js"),
  request: Record<string, unknown>,
  decision: string,
  extra: Record<string, unknown> = {},
): Record<string, unknown> {
  const unsigned = { ...request };
  delete unsigned.mac;
  return {
    schema_version: "dsh_brain_interaction.v2",
    interaction_id: request.interaction_id,
    request_digest: interaction.brainRequestDigest(unsigned),
    kind: request.kind,
    decision,
    answer: null,
    reason: "The character has a grounded internal reason.",
    ...extra,
  };
}

describe("Brain interaction", () => {
  it("binds native approval handling to the owning agent scope", async () => {
    const source = await readFile(
      resolve(process.cwd(), "src", "brain_interaction.ts"),
      "utf8",
    );
    expect(source).toContain("request.agent.session.id");
    expect(source).toContain("{ global: true }");
    expect(source).not.toContain("checkpoint");
    expect(source).not.toContain("relay_to_user");
  });

  it("derives stable per-interaction nonces and emits the V2 frame", async () => {
    const interaction = await import("../src/brain_interaction.js");
    const context: Record<string, unknown> = {
      operation_id: "operation-v2",
      operation_payload_digest: "sha256:operation",
      resolution_thread_id: "thread-v2",
      segment_id: "segment-v2",
      activation_id: "activation-v2",
      lease_epoch: 3,
      brain_conversation_ref: "chat:debug:one",
      platform: "debug",
      platform_channel_id: "channel-v2",
      global_user_id: "user-v2",
      scope_fingerprint: "sha256:scope",
      audience_fingerprint: "sha256:audience",
      profile_version: "kazusa-resolver-standard-v2",
      catalog_digest: "sha256:catalog",
      model_route_digest: "sha256:route",
      workspace_fingerprint: "sha256:workspace",
      policy_epoch: "dsh-standard-policy-v2",
      issued_reference_digest: "sha256:issued",
      issuer: "dsh-sidecar",
      nonce: "activation-authority-nonce-v2",
      issued_at: "2026-08-28T00:00:00.000Z",
      expires_at: "2026-08-28T00:05:00.000Z",
    };
    const first = interaction.buildBrainInteractionFrame(context, {
      interaction_id: "interaction-approval",
      kind: "approval",
      dsh_call_id: "approval-call",
      tool_name: "write_file",
      transient_detail: "{\"tool_name\":\"write_file\"}",
      nonce: "caller-controlled-nonce",
    });
    const replay = interaction.buildBrainInteractionFrame(context, {
      interaction_id: "interaction-approval",
      kind: "approval",
      dsh_call_id: "approval-call-replayed",
      tool_name: "write_file",
      transient_detail: "{\"tool_name\":\"write_file\"}",
      nonce: "another-caller-nonce",
    });
    expect(first.nonce).toBe(replay.nonce);
    expect(first.nonce).toMatch(/^nonce:[0-9a-f]{64}$/);
    expect(first.schema_version).toBe("dsh_brain_interaction.v2");
    expect(first.nonce).not.toBe(context.nonce);
  });

  it("sends exact V2 question and approval decisions without a checkpoint", async () => {
    const interaction = await import("../src/brain_interaction.js");
    const requests: Record<string, unknown>[] = [];
    const provider = interaction.createBrainInteractionProvider({
      secret: "brain-secret",
      request: async (request) => {
        requests.push(request);
        if (request.kind === "approval") {
          return responseFor(interaction, request, "allow_once", {
            grant: grantFor(request),
          });
        }
        return responseFor(interaction, request, "answer", {
          answer: "Use the safe bounded operation.",
        });
      },
    });
    const approval = await provider.handle(requestFields({
      interaction_id: "interaction-approval",
      kind: "approval",
      tool_name: "write_file",
      transient_detail: "{\"tool_name\":\"write_file\",\"arguments\":{}}",
    }));
    const question = await provider.handle(requestFields({
      interaction_id: "interaction-question",
      dsh_call_id: "call-question",
      transient_detail: "{\"questions\":[{\"id\":\"choice\"}]} ".trim(),
    }));
    expect(approval).toEqual({ kind: "allow_once" });
    expect(question).toEqual({
      kind: "answer",
      answer: "Use the safe bounded operation.",
    });
    expect(requests).toHaveLength(2);
    expect(requests[0]?.schema_version).toBe("dsh_brain_interaction.v2");
    expect(requests[0]).not.toHaveProperty("response_goal");
    expect(requests[0]).not.toHaveProperty("relay_mode");
  });

  it("rejects stale grants and preserves a complete single question", async () => {
    const interaction = await import("../src/brain_interaction.js");
    const provider = interaction.createBrainInteractionProvider({
      secret: "brain-secret",
      request: async (request) => responseFor(interaction, request, "allow_once", {
        grant: {
          ...grantFor(request),
          activation_id: "stale-activation",
        },
      }),
    });
    await expect(provider.handle(requestFields({ kind: "approval", tool_name: "pwsh" })))
      .rejects.toThrow("Brain allow-once grant activation_id is invalid");

    const captured: Record<string, unknown>[] = [];
    const answering = interaction.createBrainInteractionProvider({
      secret: "brain-secret",
      request: async (request) => {
        captured.push(request);
        return responseFor(interaction, request, "answer", {
          answer: "The bounded internal answer.",
        });
      },
    });
    const questions = [{
      id: "choice",
      question: "Which bounded operation should the character choose?",
      options: [{ label: "safe" }, { label: "stop" }],
    }];
    const first = await answering.handle(requestFields({
      interaction_id: "interaction-single-question",
      transient_detail: JSON.stringify({ questions }),
    }));
    expect(first).toEqual({ kind: "answer", answer: "The bounded internal answer." });
    expect(JSON.parse(String(captured[0]?.transient_detail))).toEqual({ questions });
  });

  it("recovers exact executable approval arguments and maps plan review internally", async () => {
    const interaction = await import("../src/brain_interaction.js");
    const requests: Record<string, unknown>[] = [];
    let approvalListener: ((request: any) => Promise<string>) | undefined;
    let questionProvider: { ask(request: any): Promise<any> } | undefined;
    const context = {
      reflect: { provide: vi.fn() },
      on: (event: string, listener: (request: any) => Promise<string>) => {
        if (event === "approval/request") approvalListener = listener;
        return () => true;
      },
      get: (name: string) => name === "userQuestions"
        ? {
          registerProvider: (provider: { ask(request: any): Promise<any> }) => {
            questionProvider = provider;
            return () => undefined;
          },
        }
        : undefined,
    };
    const service = new interaction.BrainInteractionService(
      context as any,
      { hostOnly: true },
    );
    const agent = {
      session: {
        id: "session-approval",
        events: [{
          type: "tool/call",
          data: {
            callId: "call-approval",
            name: "pwsh",
            arguments: JSON.stringify({
              command: "Get-Date",
              description: "presentation only",
              justification: "presentation only",
            }),
          },
        }],
      },
    };
    const brainProvider = interaction.createBrainInteractionProvider({
      secret: "brain-secret",
      request: async (frame) => {
        requests.push(frame);
        if (frame.kind === "question") {
          return responseFor(interaction, frame, "answer", {
            answer: "The bounded internal answer.",
          });
        }
        return responseFor(interaction, frame, "allow_once", {
          grant: grantFor(frame),
        });
      },
    });
    service.register({ agent, effect: vi.fn() } as any, {
      requestContext: requestFields({
        interaction_id: undefined,
        kind: undefined,
        dsh_call_id: undefined,
        tool_name: undefined,
        transient_detail: undefined,
      }),
      provider: brainProvider,
    });
    if (approvalListener === undefined || questionProvider === undefined) {
      throw new Error("Brain interaction hooks were not registered");
    }
    expect(await approvalListener({
      agent,
      toolName: "pwsh",
      callId: "call-approval",
      reason: "The character needs this exact native operation.",
    })).toBe("allowed-once");
    const detail = JSON.parse(String(requests[0]?.transient_detail));
    expect(detail).toEqual({
      tool_name: "pwsh",
      reason: "The character needs this exact native operation.",
      arguments: { command: "Get-Date" },
    });
    const questionResult = await questionProvider.ask({
      agent,
      questions: [{
        id: "plan",
        question: "Review the bounded operation.",
        options: [{ label: "approve" }],
        intent: { kind: "plan-review", approve: "approve" },
      }],
    });
    expect(questionResult).toEqual({ answers: [{ id: "plan", selected: ["approve"] }] });
    expect(requests[1]?.kind).toBe("plan_review");
    const openEnded = await questionProvider.ask({
      agent,
      questions: [{ id: "open-ended", question: "Explain the bounded choice." }],
    });
    expect(openEnded).toEqual({
      answers: [{ id: "open-ended", selected: [], custom: "The bounded internal answer." }],
    });
    await expect(questionProvider.ask({
      agent,
      questions: [{ id: "invalid-choices", question: "Choose one.", options: "invalid" }],
    })).rejects.toThrow("question choices are invalid");
    await expect(questionProvider.ask({
      agent,
      questions: [{ id: "one", question: "one" }, { id: "two", question: "two" }],
    })).rejects.toThrow("exactly one complete question");
  });
});
