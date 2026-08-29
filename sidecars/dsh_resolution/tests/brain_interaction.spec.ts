import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import { describe, expect, it, vi } from "vitest";

describe("Brain interaction", () => {
  it("binds native approval handling to the owning agent scope", async () => {
    const source = await readFile(
      resolve(process.cwd(), "src", "brain_interaction.ts"),
      "utf8",
    );
    expect(source).toContain("request.agent.session.id");
    expect(source).toContain("{ global: true }");
    expect(source).not.toContain("request.agent.ctx");
  });

  it("derives stable per-interaction nonces while preserving activation identity", async () => {
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
      transient_detail: "Approve the native file operation.",
      nonce: "caller-controlled-nonce",
    });
    const replay = interaction.buildBrainInteractionFrame(context, {
      interaction_id: "interaction-approval",
      kind: "approval",
      dsh_call_id: "approval-call-replayed",
      tool_name: "write_file",
      transient_detail: "Replay the native file operation.",
      nonce: "another-caller-nonce",
    });
    const question = interaction.buildBrainInteractionFrame(context, {
      interaction_id: "interaction-question",
      kind: "question",
      dsh_call_id: "question-call",
      tool_name: "ask_user_question",
      transient_detail: "Ask the user a question.",
    });

    expect(first.nonce).toBe(replay.nonce);
    expect(first.nonce).not.toBe(question.nonce);
    expect(first.nonce).not.toBe(context.nonce);
    expect(first.nonce).not.toBe("caller-controlled-nonce");
    expect(first.nonce).toMatch(/^nonce:[0-9a-f]{64}$/);
    expect(interaction.deriveInteractionNonce(
      "activation-authority-nonce-v2",
      "activation-other",
      3,
      "interaction-approval",
    )).not.toBe(first.nonce);
    expect(interaction.deriveInteractionNonce(
      "activation-authority-nonce-v2",
      "activation-v2",
      4,
      "interaction-approval",
    )).not.toBe(first.nonce);
    expect(interaction.deriveInteractionNonce(
      "different-authority-nonce",
      "activation-v2",
      3,
      "interaction-approval",
    )).not.toBe(first.nonce);
    for (const [key, value] of Object.entries(context)) {
      if (key !== "nonce") expect(first[key]).toEqual(value);
    }
    expect(first).toMatchObject({
      interaction_id: "interaction-approval",
      kind: "approval",
      dsh_call_id: "approval-call",
      tool_name: "write_file",
      transient_detail: "Approve the native file operation.",
      schema_version: "dsh_brain_interaction.v1",
    });
    expect(first.issued_at).toBe(context.issued_at);
    expect(first.expires_at).toBe(context.expires_at);
  });

  it("maps decisions exactly and checkpoints relay without direct user surface", async () => {
    const interaction = await import("../src/brain_interaction.js");
    const requests: Record<string, any>[] = [];
    const checkpoints: Record<string, any>[] = [];
    const responses = [
      {
        schema_version: "dsh_brain_interaction.v1",
        decision: "allow_once",
        reason: "The request is within the approved scope.",
      },
      {
        schema_version: "dsh_brain_interaction.v1",
        decision: "relay_to_user",
        reason: "The user must choose whether to continue.",
        response_goal: "Ask whether the user wants the repair applied.",
        relay_mode: "question",
      },
    ];
    const provider = interaction.createBrainInteractionProvider({
      secret: "brain-secret",
      request: async (request: Record<string, any>) => {
        requests.push(request);
        const unsigned = { ...request };
        delete unsigned.mac;
        const response = responses.shift();
        if (response === undefined) throw new Error("Brain response fixture is exhausted");
        return {
          ...response,
          interaction_id: request.interaction_id,
          request_digest: interaction.brainRequestDigest(unsigned),
          kind: request.kind,
          answer: null,
          checkpoint_required: false,
          ...(response.decision === "allow_once"
            ? {
              grant: {
                schema_version: "dsh_brain_interaction.v1",
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
              },
            }
            : {}),
        };
      },
      checkpoint: async (pending: Record<string, any>) => {
        checkpoints.push(pending);
        return {
          schema_version: "dsh_brain_interaction.v1",
          interaction_id: pending.interaction_id,
            request_digest: interaction.brainRequestDigest(
              Object.fromEntries(
              Object.entries(pending).filter(([key]) => (
                key !== "mac"
                && key !== "response_goal"
                && key !== "relay_mode"
              )),
              ),
            ),
          kind: pending.kind,
          decision: "relay_to_user",
          reason: "The user must choose whether to continue.",
          response_goal: pending.response_goal,
          relay_mode: pending.relay_mode,
          checkpoint_required: true,
          pending_interaction_id: pending.interaction_id,
          delivered_platform_message_id: "platform-message-checkpoint",
          delivery_status: null,
        };
      },
    });

    const immediate = await provider.handle({
      schema_version: "dsh_brain_interaction.v1",
      interaction_id: "interaction-allow",
      kind: "approval/request",
      operation_id: "operation-v2",
      operation_payload_digest: "sha256:operation",
      resolution_thread_id: "thread-v2",
      segment_id: "segment-v2",
      activation_id: "activation-v2",
      lease_epoch: 1,
      dsh_call_id: "dsh-call-allow",
      tool_name: "write_file",
      arguments: { path: "C:/workspace/project/file.txt", content: "repair" },
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
      nonce: "nonce-allow",
      issued_at: "2026-08-28T00:00:00.000Z",
      expires_at: "2026-08-28T00:05:00.000Z",
      transient_detail: "Allow the native file operation.",
      arguments_digest: "sha256:arguments",
    });
    expect(immediate).toEqual({
      kind: "allow_once",
    });

    const relayed = await provider.handle({
      schema_version: "dsh_brain_interaction.v1",
      interaction_id: "interaction-relay",
      kind: "userQuestions.ask",
      operation_id: "operation-v2",
      operation_payload_digest: "sha256:operation",
      resolution_thread_id: "thread-v2",
      segment_id: "segment-v2",
      activation_id: "activation-v2",
      lease_epoch: 1,
      dsh_call_id: "dsh-call-question",
      question_goal: "Clarify whether to continue the repair.",
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
      nonce: "nonce-question",
      issued_at: "2026-08-28T00:00:00.000Z",
      expires_at: "2026-08-28T00:05:00.000Z",
      transient_detail: "Clarify whether to continue the repair.",
      arguments_digest: "sha256:arguments-question",
    });
    expect(relayed).toEqual({ kind: "checkpoint_required" });
    expect(checkpoints).toEqual([expect.objectContaining({
      interaction_id: "interaction-relay",
      response_goal: "Ask whether the user wants the repair applied.",
      relay_mode: "question",
    })]);
    expect(relayed).not.toHaveProperty("text");
    expect(relayed).not.toHaveProperty("question");
    expect(requests[0]).not.toHaveProperty("reply_text");
    expect(requests[1]).not.toHaveProperty("visible_wording");
  });

  it("rejects allow-once grants from a stale activation or lease", async () => {
    const interaction = await import("../src/brain_interaction.js");
    let staleField: "activation_id" | "lease_epoch" = "activation_id";
    const provider = interaction.createBrainInteractionProvider({
      secret: "brain-secret",
      request: async (request: Record<string, any>) => {
        const unsigned = { ...request };
        delete unsigned.mac;
        return {
          schema_version: "dsh_brain_interaction.v1",
          interaction_id: request.interaction_id,
          request_digest: interaction.brainRequestDigest(unsigned),
          kind: request.kind,
          decision: "allow_once",
          reason: "A stale grant must not authorize this activation.",
          grant: {
            schema_version: "dsh_brain_interaction.v1",
            interaction_id: "original-interaction",
            resolution_thread_id: request.resolution_thread_id,
            segment_id: request.segment_id,
            activation_id: staleField === "activation_id"
              ? "stale-activation"
              : request.activation_id,
            lease_epoch: staleField === "lease_epoch"
              ? Number(request.lease_epoch) + 1
              : request.lease_epoch,
            tool_name: request.tool_name,
            arguments_digest: request.arguments_digest,
            workspace_fingerprint: request.workspace_fingerprint,
            scope_fingerprint: request.scope_fingerprint,
            policy_epoch: request.policy_epoch,
            grant_status: "consumed",
            issued_at: request.issued_at,
            expires_at: request.expires_at,
          },
        };
      },
      checkpoint: async () => {
        throw new Error("checkpoint must not run");
      },
    });
    const input = {
      schema_version: "dsh_brain_interaction.v1",
      interaction_id: "interaction-stale-activation",
      kind: "approval",
      operation_id: "operation-v2",
      operation_payload_digest: "sha256:operation",
      resolution_thread_id: "thread-v2",
      segment_id: "segment-v2",
      activation_id: "activation-v2",
      lease_epoch: 2,
      dsh_call_id: "call-stale-activation",
      tool_name: "pwsh",
      arguments_digest: "sha256:arguments",
      transient_detail: "Approve the exact native command.",
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
      nonce: "nonce-stale-activation",
      issued_at: "2026-08-28T00:00:00.000Z",
      expires_at: "2026-08-28T00:05:00.000Z",
    };

    await expect(provider.handle(input)).rejects.toThrow(
      "Brain allow-once grant activation_id is invalid",
    );
    staleField = "lease_epoch";
    await expect(provider.handle({
      ...input,
      interaction_id: "interaction-stale-lease",
      dsh_call_id: "call-stale-lease",
      nonce: "nonce-stale-lease",
    })).rejects.toThrow("Brain allow-once grant lease_epoch is invalid");
  });

  it("binds native approval retries to durable executable arguments", async () => {
    const interaction = await import("../src/brain_interaction.js");
    const argumentsBefore = {
      command: "Get-Content -LiteralPath 'C:/tmp/item.txt'",
      sandbox_permissions: "require_escalated",
      description: "First presentation",
      justification: "First explanation",
    };
    const argumentsAfter = {
      command: "Get-Content -LiteralPath 'C:/tmp/item.txt'",
      sandbox_permissions: "require_escalated",
      description: "Regenerated presentation",
      justification: "Regenerated explanation",
    };
    const digestBefore = interaction.nativeToolArgumentsDigest(
      "pwsh",
      JSON.stringify(argumentsBefore),
    );
    expect(interaction.nativeToolArgumentsDigest(
      "pwsh",
      JSON.stringify(argumentsAfter),
    )).toBe(digestBefore);
    expect(interaction.nativeToolArgumentsDigest(
      "pwsh",
      JSON.stringify({ ...argumentsAfter, command: "Get-Date" }),
    )).not.toBe(digestBefore);

    const session = {
      id: "session-approval",
      events: [
        {
          type: "tool/call",
          data: {
            callId: "call-before",
            name: "pwsh",
            arguments: JSON.stringify(argumentsBefore),
          },
        },
        {
          type: "tool/call",
          data: {
            callId: "call-after",
            name: "pwsh",
            arguments: JSON.stringify(argumentsAfter),
          },
        },
      ],
    };
    let providerDecision: Record<string, unknown> = { kind: "allow_once" };
    const requests: Record<string, unknown>[] = [];
    const cancel = vi.fn();
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
      ctx: context,
      session,
      cancel,
    };
    service.register(
      {
        agent,
        effect: vi.fn(),
      } as any,
      {
        requestContext: {
          operation_id: "operation-v2",
          operation_payload_digest: "sha256:operation",
          resolution_thread_id: "thread-v2",
          segment_id: "segment-v2",
          activation_id: "activation-v2",
          lease_epoch: 1,
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
        },
        provider: {
          handle: async (frame) => {
            requests.push(frame);
            return providerDecision;
          },
        },
      },
    );
    if (approvalListener === undefined) throw new Error("approval listener was not registered");
    expect(await approvalListener({
      agent,
      toolName: "pwsh",
      callId: "call-before",
      reason: "Approve the native command.",
    })).toBe("allowed-once");
    expect(await approvalListener({
      agent,
      toolName: "pwsh",
      callId: "call-after",
      reason: "Regenerated approval explanation.",
    })).toBe("allowed-once");
    expect(requests).toHaveLength(2);
    expect(requests[0]?.arguments_digest).toBe(digestBefore);
    expect(requests[1]?.arguments_digest).toBe(digestBefore);
    expect(requests[0]?.dsh_call_id).toBe("call-before");
    expect(requests[1]?.dsh_call_id).toBe("call-after");
    expect(cancel).not.toHaveBeenCalled();

    providerDecision = { kind: "checkpoint_required" };
    expect(await approvalListener({
      agent,
      toolName: "pwsh",
      callId: "call-after",
      reason: "Checkpoint this native approval.",
    })).toBe("cancelled");
    if (questionProvider === undefined) throw new Error("question provider was not registered");
    await expect(questionProvider.ask({
      agent,
      questions: [{
        id: "permission",
        question: "May the command continue?",
        options: [{ label: "Yes" }, { label: "No" }],
      }],
    })).rejects.toThrow("BRAIN_INTERACTION_CHECKPOINT_REQUIRED");
    expect(cancel).toHaveBeenCalledWith(
      { kind: "hook", reason: "checkpoint" },
      { keepInbox: true },
    );
    expect(cancel).toHaveBeenCalledTimes(2);

    expect(await approvalListener({
      agent,
      toolName: "pwsh",
      callId: "unknown-call",
      reason: "The durable event is missing.",
    })).toBe("unavailable");
    expect(requests).toHaveLength(4);
  });
});
