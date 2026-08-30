import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

import {
  DSH_RELEASE,
  PROFILE_VERSION,
  RPC_PROTOCOL_VERSION,
  SESSION_STORE_EPOCH,
  semanticCatalogDigest,
  semanticCatalogProjection,
  validateEvidenceReceipt,
  validateExhaust,
  validateIntake,
  validateMutationFence,
  validateSubmitResolution,
  validateTerminalReceipt,
  type ResolutionRuntime,
} from "../src/contracts.js";

export function validRuntime(): ResolutionRuntime {
  return {
    schema_version: "dsh_resolution_intake.v2",
    mode: "start",
    request_id: "rrq_1",
    operation_id: "op_1",
    operation_payload_digest: "sha256:payload",
    resolution_thread_id: "res_1",
    segment_id: "seg_1",
    brain_conversation_ref: "chat:debug:one",
    workspace_root: "C:/workspace/project",
    route_digest: "sha256:route",
    model_input: { objective: "finish", facts: [] },
    semantic_tool_authority: { catalog_digest: "sha256:catalog", token: "opaque" },
    interaction_authority: {
      issuer: "dsh-sidecar",
      scope_fingerprint: "sha256:scope",
      audience_fingerprint: "sha256:audience",
    },
  };
}

export function validIntake(): ResolutionRuntime {
  return validRuntime();
}

export function validSubmit() {
  return {
    status: "resolved",
    summary: "done",
    findings: [],
    completed_subgoals: [],
    remaining_needs: [],
    clarification_request: null,
    approval_request: null,
    artifact_refs: [],
    warnings: [],
  };
}

describe("contracts", () => {
  it("separates canonical runtime from model input", () => {
    const intake = validateIntake(validIntake());
    expect(intake.semantic_tool_authority.token).toBe("opaque");
    expect(intake.model_input).not.toHaveProperty("capability_token");
    expect(intake.model_input).not.toHaveProperty("workspace_root");
    expect(RPC_PROTOCOL_VERSION).toBe("kazusa.dsh-resolution-rpc.v2");
  });

  it("validates status-specific submit_resolution and exhaust", () => {
    expect(validateSubmitResolution(validSubmit()).status).toBe("resolved");
    expect(() => validateSubmitResolution({ ...validSubmit(), status: "needs_user_input" }))
      .toThrow(/clarification_request/);
    expect(validateExhaust({ kind: "checkpointed", checkpoint: { reason: "requested" } }).kind)
      .toBe("checkpointed");
  });

  it("validates exact bounded evidence and terminal receipt metadata", () => {
    const evidence = validateEvidenceReceipt({
      schema_version: "evidence_receipt.v2",
      resolution_thread_id: "res_1",
      segment_id: "seg_1",
      scope_fingerprint: "sha256:scope",
      audience_fingerprint: "sha256:audience",
      policy_epoch: "2026-08-28.1",
      evidence_id: "ev_1",
      source_kind: "fixture",
      semantic_ref: "semantic-ref-1",
      content_digest: "sha256:content",
      provenance: { tool_name: "fixture_evidence" },
    });
    expect(evidence.evidence_id).toBe("ev_1");
    const receipt = validateTerminalReceipt({
      kind: "terminal_resolution_v2",
      schema_version: "2",
      call_id: "call_terminal",
      operation_id: "op_1",
      operation_payload_digest: "sha256:payload",
      request_id: "rrq_1",
      resolution_thread_id: "res_1",
      segment_id: "seg_1",
      activation_id: "act_1",
      lease_epoch: 1,
      brain_conversation_ref: "chat:debug:one",
      workspace_root: "C:/workspace/project",
      route_digest: "sha256:route",
      scope_fingerprint: "sha256:scope",
      catalog_digest: "sha256:catalog",
      interaction_issuer: "dsh-sidecar",
      policy_epoch: "2026-08-28.1",
      terminal: validSubmit(),
      terminal_digest: "sha256:terminal",
    });
    expect(receipt.catalog_digest).toBe("sha256:catalog");
  });

  it("requires operation activation and lease fencing on live mutations", () => {
    expect(validateMutationFence({ operation_id: "op_1", operation_payload_digest: "sha256:p", activation_id: "act_1", lease_epoch: 1 }).lease_epoch)
      .toBe(1);
    expect(() => validateMutationFence({ operation_id: "op_1", operation_payload_digest: "sha256:p", activation_id: "act_1", lease_epoch: 0 }))
      .toThrow(/lease_epoch/);
  });
});

describe("V2 contracts", () => {
  it("matches the normalized Python semantic catalog exactly", () => {
    const repositoryRoot = resolve(process.cwd(), "..", "..");
    const python = resolve(repositoryRoot, "venv", "Scripts", "python.exe");
    const script = [
      "import json",
      "from kazusa_ai_chatbot.dsh_tool_gateway.catalog import description_stripped_catalog",
      "print(json.dumps(description_stripped_catalog(set()), ensure_ascii=False, sort_keys=True, separators=(',', ':')))",
    ].join("; ");
    const output = execFileSync(python, ["-c", script], {
      cwd: repositoryRoot,
      env: {
        ...process.env,
        PYTHONPATH: resolve(repositoryRoot, "src"),
      },
      encoding: "utf8",
    });
    const pythonProjection = JSON.parse(output) as unknown;
    expect(pythonProjection).toEqual(semanticCatalogProjection());
    expect(semanticCatalogProjection()).toHaveLength(14);
  });

  it("rejects V1 and separates model-visible input from authority", async () => {
    const contracts = await import("../src/contracts.js");
    const validateIntake = contracts.validateIntake as unknown as (
      value: unknown,
    ) => Record<string, any>;
    expect(contracts.RPC_PROTOCOL_VERSION).toBe("kazusa.dsh-resolution-rpc.v2");
    expect(contracts.PROFILE_VERSION).toBe("kazusa-resolver-standard-v2");
    expect(contracts.SESSION_STORE_EPOCH).toBe("dsh-sqlite-0.1.1-rc.2-standard-v2");
    expect(() => validateIntake({ ...validIntake(), schema_version: "dsh_resolution_intake.v1" }))
      .toThrow(/unsupported|version/i);
    const intake = validateIntake({
      schema_version: "dsh_resolution_intake.v2",
      mode: "start",
      request_id: "request-v2",
      operation_id: "operation-v2",
      operation_payload_digest: "sha256:payload-v2",
      resolution_thread_id: "thread-v2",
      segment_id: "segment-v2",
      brain_conversation_ref: "chat:debug:one",
      workspace_root: "C:/workspace/project",
      route_digest: "sha256:route",
      model_input: { objective: "inspect the project", facts: [] },
      semantic_tool_authority: { catalog_digest: "sha256:catalog", token: "opaque" },
      interaction_authority: {
        issuer: "dsh-sidecar",
        scope_fingerprint: "sha256:scope",
        audience_fingerprint: "sha256:audience",
      },
    });
    expect(intake.model_input).not.toHaveProperty("workspace_root");
    expect(intake.model_input).not.toHaveProperty("semantic_tool_authority");
    expect(intake.workspace_root).toBe("C:/workspace/project");
    expect(intake.semantic_tool_authority.token).toBe("opaque");
  });

  it("uses the byte-identical description-free fourteen-tool catalog projection", () => {
    const projection = semanticCatalogProjection();
    expect(projection).toHaveLength(14);
    expect(JSON.stringify(projection)).not.toContain("description");
    expect(semanticCatalogDigest()).toMatch(/^sha256:[0-9a-f]{64}$/u);
    expect(semanticCatalogDigest()).not.toBe(
      "sha256:495baf34779da92da5d554e70e51dc47579fb88f6af2c0a7992c46b2f88e02d4",
    );
  });

  it("declares the exact public-media input schema", () => {
    const media = semanticCatalogProjection().find(
      (item) => item.name === "kazusa_inspect_public_media",
    );

    expect(media).toMatchObject({
      name: "kazusa_inspect_public_media",
      input_schema: {
        type: "object",
        additionalProperties: false,
        required: ["public_media_url", "question"],
        properties: {
          public_media_url: { type: "string" },
          question: { type: "string" },
        },
      },
    });
  });

  it("requires an explicit model-hidden audience fingerprint", () => {
    const missing = validIntake() as unknown as Record<string, any>;
    const authority = missing.interaction_authority as Record<string, unknown>;
    delete authority.audience_fingerprint;
    expect(() => validateIntake(missing)).toThrow(/missing|fields/i);
  });

  it("requires a canonical absolute workspace root", () => {
    expect(() => validateIntake({
      ...validIntake(),
      workspace_root: "C:/workspace/../project",
    })).toThrow(/canonical/u);
  });
});
