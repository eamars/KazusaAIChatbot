import { describe, expect, it } from "vitest";
import { digest, issueActivationToken, scopeFingerprint, workspaceFingerprint } from "../src/contracts.js";

describe("semantic gateway", () => {
  it("attaches invisible authority and persists bounded evidence receipts", async () => {
    const semantic = await import("../src/semantic_gateway.js");
    const serviceScope = {
      platform: "debug",
      platform_channel_id: "channel-1",
      global_user_id: "user-1",
    };
    const workspaceRoot = "C:/workspace/project";
    const authority = {
      schema_version: "kazusa_semantic_tool_authority.v1" as const,
      resolution_thread_id: "thread-v2",
      segment_id: "segment-v2",
      activation_id: "activation-v2",
      lease_epoch: 1,
      brain_conversation_ref: "chat:debug:one",
      service_scope: serviceScope,
      scope_fingerprint: scopeFingerprint(serviceScope),
      audience_fingerprint: "sha256:audience",
      workspace_root: workspaceRoot,
      route_digest: "sha256:route",
      catalog_digest: "sha256:catalog",
      profile_version: "kazusa-resolver-standard-v2",
      model_route_digest: "sha256:route",
      workspace_fingerprint: workspaceFingerprint(workspaceRoot),
      issued_reference_digest: "sha256:issued",
      policy_epoch: "dsh-standard-policy-v2",
      interaction_issuer: "dsh-sidecar-test",
      issued_at: "2026-08-28T00:00:00.000Z",
      expires_at: "2026-08-28T00:05:00.000Z",
      token_id: "token-1",
      nonce: "nonce-1",
    };
    const frames: Record<string, any>[] = [];
    const persisted: Record<string, any>[] = [];
    const gateway = semantic.createSemanticGateway({
      secret: "gateway-secret",
      authority,
      authorityToken: issueActivationToken(authority, "gateway-secret"),
      call: async (frame: Record<string, any>) => {
        frames.push(frame);
        return {
          schema_version: "kazusa_semantic_capability_result.v1",
          status: "ok",
          entities: [{ kind: "memory", memory_ref: "opaque-memory-1", summary: "A relevant memory" }],
          page: { has_more: false, next_page_ref: null },
          evidence: [{
            schema_version: "evidence_receipt.v2",
            evidence_id: "evidence-1",
            semantic_ref: "opaque-memory-1",
            source_kind: "semantic",
            content_digest: "sha256:content",
          }],
          mutation: null,
          error: null,
        };
      },
      persistEvidence: async (receipt: Record<string, any>) => {
        persisted.push(receipt);
      },
      now: () => new Date("2026-08-28T00:01:00.000Z"),
    });

    const result = await gateway.invoke("kazusa_search_memories", {
      query: "A document about MongoDB is relevant",
      subject_scope: "current_user",
      memory_kinds: ["experience"],
      max_results: 5,
    });

    expect(frames).toHaveLength(1);
    expect(frames[0]).toMatchObject({
      operation: "kazusa_search_memories",
      arguments: {
        query: "A document about MongoDB is relevant",
        subject_scope: "current_user",
      },
      authority: {
        resolution_thread_id: "thread-v2",
        segment_id: "segment-v2",
        catalog_digest: "sha256:catalog",
        model_route_digest: "sha256:route",
        service_scope: serviceScope,
      },
    });
    expect(frames[0]).not.toHaveProperty("model_input");
    expect(frames[0]).not.toHaveProperty("capability_token");
    expect(frames[0]).not.toHaveProperty("claim");
    expect(frames[0]).toHaveProperty("arguments_digest");
    expect(frames[0]).toHaveProperty("idempotency_key", null);
    expect(result.schema_version).toBe("kazusa_semantic_capability_result.v1");
    expect(result.entities[0]).toMatchObject({ memory_ref: "opaque-memory-1" });
    expect(persisted).toEqual(result.evidence);
    expect(result.evidence[0]).toMatchObject({
      schema_version: "evidence_receipt.v2",
      resolution_thread_id: "thread-v2",
      segment_id: "segment-v2",
      scope_fingerprint: authority.scope_fingerprint,
      audience_fingerprint: authority.audience_fingerprint,
      provenance: { tool_name: "kazusa_search_memories" },
    });
    expect(result).toHaveProperty("page.has_more", false);

    await gateway.invoke("kazusa_remember_information", {
      subject: "current_user",
      information: "A document about MongoDB is semantic content.",
      memory_kind: "experience",
      reason: "authority test",
      provenance: { current_task: "gateway-test" },
    }, "transport-retry-1");
    await gateway.invoke("kazusa_remember_information", {
      subject: "current_user",
      information: "A document about MongoDB is semantic content.",
      memory_kind: "experience",
      reason: "authority test",
      provenance: { current_task: "gateway-test" },
    }, "transport-retry-2");
    const firstRetry = frames[1];
    const secondRetry = frames[2];
    expect(firstRetry?.idempotency_key).toBe(secondRetry?.idempotency_key);
    expect(firstRetry?.idempotency_key).toMatch(/^idem:sha256:/u);
    expect(firstRetry?.idempotency_key).not.toMatch(/^idem:sha256:sha256:/u);
    expect(firstRetry?.call_id).not.toBe(secondRetry?.call_id);
    expect(firstRetry).not.toHaveProperty("claim");

    await gateway.invoke("kazusa_inspect_public_media", {
      public_media_url: "https://example.test/image.png",
      question: "What is visible?",
    });
    const mediaFrame = frames[3];
    expect(mediaFrame?.operation).toBe("kazusa_inspect_public_media");
    expect(mediaFrame?.arguments).toEqual({
      public_media_url: "https://example.test/image.png",
      question: "What is visible?",
    });
    expect(mediaFrame?.arguments).not.toHaveProperty("capability_token");
    expect(mediaFrame?.arguments).not.toHaveProperty("authority");
    expect(mediaFrame).toHaveProperty("authority");

    await expect(gateway.invoke("kazusa_inspect_public_media", {
      public_media_url: "https://example.test/image.png",
      question: "What is visible?",
      capability_token: "must-be-rejected",
    })).rejects.toThrow();
  });
});
