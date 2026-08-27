import { describe, expect, it } from "vitest";

import {
  DSH_RELEASE,
  PROFILE_VERSION,
  RPC_PROTOCOL_VERSION,
  SESSION_STORE_EPOCH,
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
    request_id: "rrq_1",
    operation_id: "op_1",
    operation_payload_digest: "sha256:payload",
    resolution_thread_id: "res_1",
    segment_id: "seg_1",
    priority: "now",
    soft_deadline_at: "2026-08-28T00:00:10Z",
    hard_deadline_at: "2026-08-28T00:00:30Z",
    max_model_steps: 3,
    max_tool_calls: 3,
    max_tool_bytes: 4096,
    capability_token: "opaque",
    scope_fingerprint: "sha256:scope",
    audience_fingerprint: "sha256:audience",
    resolver_profile_version: PROFILE_VERSION,
    dsh_release: DSH_RELEASE,
    session_store_epoch: SESSION_STORE_EPOCH,
    model_route: "resolver-model",
    tool_catalog_digest: "sha256:catalog",
    policy_epoch: "2026-08-28.1",
  };
}

export function validIntake() {
  return {
    schema_version: "dsh_resolution_intake.v1",
    mode: "start",
    runtime: validRuntime(),
    model_input: {
      objective: "finish",
      constraints: [],
      success_criteria: [],
      known_facts: [],
      uncertainty: [],
      literal_inputs: [],
      continuation_delta: null,
      prior_resolution_refs: [],
      requested_evidence_quality: "normal",
      notes: [],
    },
  };
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
    expect(intake.runtime.capability_token).toBe("opaque");
    expect(intake.model_input).not.toHaveProperty("capability_token");
    expect(RPC_PROTOCOL_VERSION).toBe("kazusa.dsh-resolution-rpc.v1");
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
      kind: "evidence_receipt_v1",
      schema_version: "1",
      call_id: "call_1",
      operation_id: "op_1",
      resolution_thread_id: "res_1",
      segment_id: "seg_1",
      scope_fingerprint: "sha256:scope",
      audience_fingerprint: "sha256:audience",
      policy_epoch: "2026-08-28.1",
      tool_name: "fixture_evidence",
      evidence_ids: ["ev_1"],
      provenance: [{ evidence_id: "ev_1", source_kind: "fixture", source_id: "source_1", content_digest: "sha256:content" }],
      evidence_digest: "sha256:evidence",
    });
    expect(evidence.evidence_ids).toEqual(["ev_1"]);
    const receipt = validateTerminalReceipt({
      kind: "terminal_resolution_v1",
      schema_version: "1",
      call_id: "call_terminal",
      operation_id: "op_1",
      operation_payload_digest: "sha256:payload",
      request_id: "rrq_1",
      resolution_thread_id: "res_1",
      segment_id: "seg_1",
      activation_id: "act_1",
      lease_epoch: 1,
      scope_fingerprint: "sha256:scope",
      audience_fingerprint: "sha256:audience",
      resolver_profile_version: PROFILE_VERSION,
      dsh_release: DSH_RELEASE,
      session_store_epoch: SESSION_STORE_EPOCH,
      model_route: "resolver-model",
      tool_catalog_digest: "sha256:catalog",
      policy_epoch: "2026-08-28.1",
      terminal: validSubmit(),
      terminal_digest: "sha256:terminal",
    });
    expect(receipt.session_store_epoch).toBe(SESSION_STORE_EPOCH);
  });

  it("requires operation activation and lease fencing on live mutations", () => {
    expect(validateMutationFence({ operation_id: "op_1", operation_payload_digest: "sha256:p", activation_id: "act_1", lease_epoch: 1 }).lease_epoch)
      .toBe(1);
    expect(() => validateMutationFence({ operation_id: "op_1", operation_payload_digest: "sha256:p", activation_id: "act_1", lease_epoch: 0 }))
      .toThrow(/lease_epoch/);
  });
});
