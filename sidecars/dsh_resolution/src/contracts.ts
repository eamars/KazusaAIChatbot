export const RPC_PROTOCOL_VERSION = "kazusa.dsh-resolution-rpc.v1" as const;
export const PROFILE_VERSION = "kazusa-resolver-v1" as const;
export const DSH_RELEASE = "0.1.1-rc.2" as const;
export const SESSION_STORE_EPOCH = "dsh-sqlite-0.1.1-rc.2-v1" as const;

export type JsonObject = Record<string, unknown>;

export interface ResolutionRuntime {
  request_id: string;
  operation_id: string;
  operation_payload_digest: string;
  resolution_thread_id: string;
  segment_id: string;
  priority: "now" | "background";
  soft_deadline_at: string;
  hard_deadline_at: string;
  max_model_steps: number;
  max_tool_calls: number;
  max_tool_bytes: number;
  capability_token: string;
  scope_fingerprint: string;
  audience_fingerprint: string;
  resolver_profile_version: typeof PROFILE_VERSION;
  dsh_release: typeof DSH_RELEASE;
  session_store_epoch: typeof SESSION_STORE_EPOCH;
  model_route: string;
  tool_catalog_digest: string;
  policy_epoch: string;
}

export interface ResolutionIntake {
  schema_version: "dsh_resolution_intake.v1";
  mode: "start" | "continue";
  runtime: ResolutionRuntime;
  model_input: {
    objective: string;
    constraints: string[];
    success_criteria: string[];
    known_facts: string[];
    uncertainty: string[];
    literal_inputs: string[];
    continuation_delta: string | null;
    prior_resolution_refs: string[];
    requested_evidence_quality: string;
    notes: string[];
  };
}

export interface SubmitResolution {
  status: "resolved" | "partial" | "needs_user_input" | "approval_required" | "unavailable" | "failed";
  summary: string;
  findings: JsonObject[];
  completed_subgoals: string[];
  remaining_needs: string[];
  clarification_request: JsonObject | null;
  approval_request: JsonObject | null;
  artifact_refs: string[];
  warnings: string[];
}

export interface EvidenceReceipt {
  kind: "evidence_receipt_v1";
  schema_version: "1";
  call_id: string;
  operation_id: string;
  resolution_thread_id: string;
  segment_id: string;
  scope_fingerprint: string;
  audience_fingerprint: string;
  policy_epoch: string;
  tool_name: string;
  evidence_ids: string[];
  provenance: Array<{ evidence_id: string; source_kind: string; source_id: string; content_digest: string }>;
  evidence_digest: string;
}

export interface TerminalReceipt {
  kind: "terminal_resolution_v1";
  schema_version: "1";
  call_id: string;
  operation_id: string;
  operation_payload_digest: string;
  request_id: string;
  resolution_thread_id: string;
  segment_id: string;
  activation_id: string;
  lease_epoch: number;
  scope_fingerprint: string;
  audience_fingerprint: string;
  resolver_profile_version: typeof PROFILE_VERSION;
  dsh_release: typeof DSH_RELEASE;
  session_store_epoch: typeof SESSION_STORE_EPOCH;
  model_route: string;
  tool_catalog_digest: string;
  policy_epoch: string;
  terminal: SubmitResolution;
  terminal_digest: string;
}

export class ContractFault extends Error {
  constructor(message: string, readonly code = "RPC_CONTRACT_ERROR") {
    super(message);
  }
}

function object(value: unknown, name: string): JsonObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new ContractFault(`${name} must be an object`);
  }
  return value as JsonObject;
}

function exact(value: unknown, keys: readonly string[], name: string): JsonObject {
  const result = object(value, name);
  const present = Object.keys(result).sort();
  const expected = [...keys].sort();
  if (JSON.stringify(present) !== JSON.stringify(expected)) {
    throw new ContractFault(`${name} has unknown or missing fields`);
  }
  return result;
}

function text(value: unknown, name: string): string {
  if (typeof value !== "string" || value.length === 0) throw new ContractFault(`${name} must be a non-empty string`);
  return value;
}

function integer(value: unknown, name: string, minimum = 0): number {
  if (!Number.isInteger(value) || (value as number) < minimum) throw new ContractFault(`${name} must be an integer >= ${minimum}`);
  return value as number;
}

function texts(value: unknown, name: string, maximum = 64): string[] {
  if (!Array.isArray(value) || value.length > maximum) throw new ContractFault(`${name} must be a bounded list`);
  const result = value.map((item) => text(item, name));
  if (new Set(result).size !== result.length) throw new ContractFault(`${name} must contain unique values`);
  return result;
}

export function validateRuntime(value: unknown): ResolutionRuntime {
  const keys = ["request_id", "operation_id", "operation_payload_digest", "resolution_thread_id", "segment_id", "priority", "soft_deadline_at", "hard_deadline_at", "max_model_steps", "max_tool_calls", "max_tool_bytes", "capability_token", "scope_fingerprint", "audience_fingerprint", "resolver_profile_version", "dsh_release", "session_store_epoch", "model_route", "tool_catalog_digest", "policy_epoch"] as const;
  const row = exact(value, keys, "runtime");
  for (const key of keys.filter((key) => !["max_model_steps", "max_tool_calls", "max_tool_bytes"].includes(key))) text(row[key], `runtime.${key}`);
  for (const key of ["max_model_steps", "max_tool_calls", "max_tool_bytes"] as const) integer(row[key], `runtime.${key}`, 1);
  if (row.priority !== "now" && row.priority !== "background") throw new ContractFault("runtime.priority is unsupported");
  if (row.resolver_profile_version !== PROFILE_VERSION || row.dsh_release !== DSH_RELEASE || row.session_store_epoch !== SESSION_STORE_EPOCH) throw new ContractFault("runtime profile or store epoch is unsupported");
  return row as unknown as ResolutionRuntime;
}

export function validateIntake(value: unknown): ResolutionIntake {
  const row = exact(value, ["schema_version", "mode", "runtime", "model_input"], "intake");
  if (row.schema_version !== "dsh_resolution_intake.v1") throw new ContractFault("intake schema_version is unsupported");
  if (row.mode !== "start" && row.mode !== "continue") throw new ContractFault("intake mode is unsupported");
  const model = exact(row.model_input, ["objective", "constraints", "success_criteria", "known_facts", "uncertainty", "literal_inputs", "continuation_delta", "prior_resolution_refs", "requested_evidence_quality", "notes"], "model_input");
  text(model.objective, "model_input.objective");
  for (const key of ["constraints", "success_criteria", "known_facts", "uncertainty", "literal_inputs", "prior_resolution_refs", "notes"] as const) texts(model[key], `model_input.${key}`);
  if (model.continuation_delta !== null) text(model.continuation_delta, "model_input.continuation_delta");
  text(model.requested_evidence_quality, "model_input.requested_evidence_quality");
  return { schema_version: "dsh_resolution_intake.v1", mode: row.mode, runtime: validateRuntime(row.runtime), model_input: model as unknown as ResolutionIntake["model_input"] };
}

export function validateSubmitResolution(value: unknown): SubmitResolution {
  const row = exact(value, ["status", "summary", "findings", "completed_subgoals", "remaining_needs", "clarification_request", "approval_request", "artifact_refs", "warnings"], "submit_resolution");
  const statuses = new Set(["resolved", "partial", "needs_user_input", "approval_required", "unavailable", "failed"]);
  if (!statuses.has(String(row.status))) throw new ContractFault("submit_resolution.status is unsupported");
  text(row.summary, "submit_resolution.summary");
  if (!Array.isArray(row.findings) || row.findings.length > 64) throw new ContractFault("submit_resolution.findings must be bounded");
  row.findings.forEach((item) => object(item, "submit_resolution.finding"));
  for (const key of ["completed_subgoals", "remaining_needs", "artifact_refs", "warnings"] as const) texts(row[key], `submit_resolution.${key}`);
  if (row.status === "needs_user_input" && row.clarification_request === null) throw new ContractFault("clarification_request is required");
  if (row.status === "approval_required" && row.approval_request === null) throw new ContractFault("approval_request is required");
  if (row.clarification_request !== null) object(row.clarification_request, "clarification_request");
  if (row.approval_request !== null) object(row.approval_request, "approval_request");
  return row as unknown as SubmitResolution;
}

export function validateEvidenceReceipt(value: unknown): EvidenceReceipt {
  const keys = ["kind", "schema_version", "call_id", "operation_id", "resolution_thread_id", "segment_id", "scope_fingerprint", "audience_fingerprint", "policy_epoch", "tool_name", "evidence_ids", "provenance", "evidence_digest"] as const;
  const row = exact(value, keys, "evidence receipt");
  if (row.kind !== "evidence_receipt_v1" || row.schema_version !== "1") throw new ContractFault("evidence receipt version is unsupported");
  for (const key of ["call_id", "operation_id", "resolution_thread_id", "segment_id", "scope_fingerprint", "audience_fingerprint", "policy_epoch", "tool_name", "evidence_digest"] as const) text(row[key], `evidence.${key}`);
  const ids = texts(row.evidence_ids, "evidence.evidence_ids");
  if (!Array.isArray(row.provenance) || row.provenance.length !== ids.length || row.provenance.length > 64) throw new ContractFault("evidence provenance must match evidence ids");
  row.provenance.forEach((item, index) => {
    const provenance = exact(item, ["evidence_id", "source_kind", "source_id", "content_digest"], "evidence provenance");
    if (provenance.evidence_id !== ids[index]) throw new ContractFault("evidence identity order mismatch");
  });
  return row as unknown as EvidenceReceipt;
}

export function validateTerminalReceipt(value: unknown): TerminalReceipt {
  const keys = ["kind", "schema_version", "call_id", "operation_id", "operation_payload_digest", "request_id", "resolution_thread_id", "segment_id", "activation_id", "lease_epoch", "scope_fingerprint", "audience_fingerprint", "resolver_profile_version", "dsh_release", "session_store_epoch", "model_route", "tool_catalog_digest", "policy_epoch", "terminal", "terminal_digest"] as const;
  const row = exact(value, keys, "terminal receipt");
  if (row.kind !== "terminal_resolution_v1" || row.schema_version !== "1") throw new ContractFault("terminal receipt version is unsupported");
  for (const key of keys.filter((key) => !["lease_epoch", "terminal"].includes(key))) text(row[key], `terminal.${key}`);
  integer(row.lease_epoch, "terminal.lease_epoch", 1);
  if (row.resolver_profile_version !== PROFILE_VERSION || row.dsh_release !== DSH_RELEASE || row.session_store_epoch !== SESSION_STORE_EPOCH) throw new ContractFault("terminal profile or store epoch mismatch");
  return { ...row, terminal: validateSubmitResolution(row.terminal) } as unknown as TerminalReceipt;
}

export function validateMutationFence(value: unknown): { operation_id: string; operation_payload_digest: string; activation_id: string; lease_epoch: number } {
  const row = exact(value, ["operation_id", "operation_payload_digest", "activation_id", "lease_epoch"], "mutation fence");
  return { operation_id: text(row.operation_id, "operation_id"), operation_payload_digest: text(row.operation_payload_digest, "operation_payload_digest"), activation_id: text(row.activation_id, "activation_id"), lease_epoch: integer(row.lease_epoch, "lease_epoch", 1) };
}

export function validateExhaust(value: unknown): JsonObject {
  const row = object(value, "exhaust");
  if (!new Set(["terminal", "checkpointed", "runtime_fault"]).has(String(row.kind))) throw new ContractFault("exhaust kind is unsupported");
  if (row.kind === "terminal") validateSubmitResolution(row.terminal);
  return row;
}
