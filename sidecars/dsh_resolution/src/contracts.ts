import { createHash, createHmac, timingSafeEqual } from "node:crypto";
import { posix, win32 } from "node:path";

export const RPC_PROTOCOL_VERSION = "kazusa.dsh-resolution-rpc.v2" as const;
export const PROFILE_VERSION = "kazusa-resolver-standard-v2" as const;
export const DSH_RELEASE = "0.1.1-rc.2" as const;
export const SESSION_STORE_EPOCH = "dsh-sqlite-0.1.1-rc.2-standard-v2" as const;

export type JsonObject = Record<string, unknown>;

export const AUTHORITY_SCHEMA_VERSION = "kazusa_semantic_tool_authority.v1" as const;
export const CALL_SCHEMA_VERSION = "kazusa_semantic_tool_call.v1" as const;
export const ACTIVATION_TOKEN_PREFIX = "ksa1" as const;
const ACTIVATION_MAC_DOMAIN = "kazusa-semantic-activation-v1\u0000";

/** Python-compatible sorted-key compact JSON for every cross-boundary digest. */
export function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    return `{${Object.keys(value as Record<string, unknown>)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson((value as Record<string, unknown>)[key])}`)
      .join(",")}}`;
  }
  const encoded = JSON.stringify(value);
  if (encoded === undefined) throw new ContractFault("value is not canonical JSON");
  return encoded;
}

export function digest(value: unknown): string {
  return `sha256:${createHash("sha256").update(canonicalJson(value), "utf8").digest("hex")}`;
}

export function workspaceFingerprint(workspaceRoot: string): string {
  return digest({ workspace_root: workspaceRoot.replaceAll("\\", "/") });
}

export function scopeFingerprint(scope: ServiceScope): string {
  return digest(scope);
}

export interface ServiceScope {
  platform: string;
  platform_channel_id: string;
  global_user_id: string;
}

export interface ActivationAuthority {
  schema_version: typeof AUTHORITY_SCHEMA_VERSION;
  activation_id: string;
  lease_epoch: number;
  resolution_thread_id: string;
  segment_id: string;
  brain_conversation_ref: string;
  service_scope: ServiceScope;
  scope_fingerprint: string;
  audience_fingerprint: string;
  workspace_root: string;
  route_digest: string;
  catalog_digest: string;
  profile_version: string;
  model_route_digest: string;
  workspace_fingerprint: string;
  issued_reference_digest: string;
  policy_epoch: string;
  interaction_issuer: string;
  issued_at: string;
  expires_at: string;
  token_id: string;
  nonce: string;
}

const ACTIVATION_AUTHORITY_FIELDS = [
  "schema_version", "activation_id", "lease_epoch", "resolution_thread_id",
  "segment_id", "brain_conversation_ref", "service_scope", "scope_fingerprint",
  "audience_fingerprint", "workspace_root", "route_digest", "catalog_digest",
  "profile_version", "model_route_digest", "workspace_fingerprint",
  "issued_reference_digest", "policy_epoch", "interaction_issuer", "issued_at",
  "expires_at", "token_id", "nonce",
] as const;

function activationObject(value: unknown, name: string): JsonObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new ContractFault(`${name} must be an object`);
  }
  return value as JsonObject;
}

function activationExact(value: unknown, keys: readonly string[], name: string): JsonObject {
  const result = activationObject(value, name);
  const actual = Object.keys(result).sort();
  const expected = [...keys].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new ContractFault(`${name} has unknown or missing fields`);
  }
  return result;
}

function activationText(value: unknown, name: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new ContractFault(`${name} must be a non-empty string`);
  }
  return value;
}

function activationServiceScope(value: unknown): ServiceScope {
  const row = activationExact(value, [
    "platform", "platform_channel_id", "global_user_id",
  ], "authority.service_scope");
  return {
    platform: activationText(row.platform, "authority.service_scope.platform"),
    platform_channel_id: activationText(row.platform_channel_id, "authority.service_scope.platform_channel_id"),
    global_user_id: activationText(row.global_user_id, "authority.service_scope.global_user_id"),
  };
}

function activationTime(value: string, name: string): number {
  const parsed = Date.parse(value);
  if (!Number.isFinite(parsed)) throw new ContractFault(`${name} is invalid`);
  return parsed;
}

export function validateActivationAuthority(value: unknown): ActivationAuthority {
  const row = activationExact(value, ACTIVATION_AUTHORITY_FIELDS, "authority");
  if (row.schema_version !== AUTHORITY_SCHEMA_VERSION) {
    throw new ContractFault("authority schema is unsupported");
  }
  if (!Number.isInteger(row.lease_epoch) || (row.lease_epoch as number) < 1) {
    throw new ContractFault("authority.lease_epoch must be positive");
  }
  for (const key of ACTIVATION_AUTHORITY_FIELDS) {
    if (key === "schema_version" || key === "lease_epoch" || key === "service_scope") continue;
    activationText(row[key], `authority.${key}`);
  }
  const serviceScope = activationServiceScope(row.service_scope);
  const workspaceRoot = absolutePath(
    activationText(row.workspace_root, "authority.workspace_root"),
    "authority.workspace_root",
  );
  const issued = activationTime(row.issued_at as string, "authority.issued_at");
  const expires = activationTime(row.expires_at as string, "authority.expires_at");
  if (expires <= issued || expires - issued > 300_000) {
    throw new ContractFault("authority lifetime is invalid");
  }
  if (row.scope_fingerprint !== scopeFingerprint(serviceScope)) {
    throw new ContractFault("authority service scope fingerprint mismatch");
  }
  if (row.workspace_fingerprint !== workspaceFingerprint(workspaceRoot)) {
    throw new ContractFault("authority workspace fingerprint mismatch");
  }
  if (row.route_digest !== row.model_route_digest) {
    throw new ContractFault("authority route digest mismatch");
  }
  return {
    schema_version: AUTHORITY_SCHEMA_VERSION,
    activation_id: row.activation_id as string,
    lease_epoch: row.lease_epoch as number,
    resolution_thread_id: row.resolution_thread_id as string,
    segment_id: row.segment_id as string,
    brain_conversation_ref: row.brain_conversation_ref as string,
    service_scope: serviceScope,
    scope_fingerprint: row.scope_fingerprint as string,
    audience_fingerprint: row.audience_fingerprint as string,
    workspace_root: workspaceRoot.replaceAll("\\", "/"),
    route_digest: row.route_digest as string,
    catalog_digest: row.catalog_digest as string,
    profile_version: row.profile_version as string,
    model_route_digest: row.model_route_digest as string,
    workspace_fingerprint: row.workspace_fingerprint as string,
    issued_reference_digest: row.issued_reference_digest as string,
    policy_epoch: row.policy_epoch as string,
    interaction_issuer: row.interaction_issuer as string,
    issued_at: row.issued_at as string,
    expires_at: row.expires_at as string,
    token_id: row.token_id as string,
    nonce: row.nonce as string,
  };
}

function activationMac(secret: string, payload: Buffer): Buffer {
  return createHmac("sha256", secret)
    .update(ACTIVATION_MAC_DOMAIN, "utf8")
    .update(payload)
    .digest();
}

export function issueActivationToken(
  authority: ActivationAuthority,
  secret: string,
): string {
  const validated = validateActivationAuthority(authority);
  if (secret.length === 0) throw new ContractFault("semantic authority secret is required");
  const payload = Buffer.from(canonicalJson(validated), "utf8");
  return `${ACTIVATION_TOKEN_PREFIX}.${payload.toString("base64url")}.${activationMac(secret, payload).toString("hex")}`;
}

export function verifyActivationToken(
  token: string,
  secret: string,
  expected: Partial<ActivationAuthority> = {},
  now = Date.now(),
): ActivationAuthority {
  if (secret.length === 0) throw new ContractFault("semantic authority secret is required");
  const parts = token.split(".");
  if (parts.length !== 3 || parts[0] !== ACTIVATION_TOKEN_PREFIX) {
    throw new ContractFault("activation token format is invalid");
  }
  let payload: Buffer;
  try {
    payload = Buffer.from(parts[1] as string, "base64url");
  } catch (error) {
    throw new ContractFault("activation token payload is invalid", "AUTHORITY_TOKEN_INVALID");
  }
  if (payload.length === 0 || payload.toString("base64url") !== parts[1]) {
    throw new ContractFault("activation token payload is invalid", "AUTHORITY_TOKEN_INVALID");
  }
  const supplied = Buffer.from(parts[2] as string, "hex");
  const expectedMac = activationMac(secret, payload);
  if (supplied.length !== expectedMac.length || !timingSafeEqual(supplied, expectedMac)) {
    throw new ContractFault("activation token authentication failed", "AUTHORITY_TOKEN_INVALID");
  }
  let decoded: unknown;
  try {
    decoded = JSON.parse(payload.toString("utf8"));
  } catch (error) {
    throw new ContractFault("activation token payload is invalid", "AUTHORITY_TOKEN_INVALID");
  }
  const authority = validateActivationAuthority(decoded);
  if (canonicalJson(authority) !== payload.toString("utf8")) {
    throw new ContractFault("activation token payload is not canonical", "AUTHORITY_TOKEN_INVALID");
  }
  for (const [key, value] of Object.entries(expected)) {
    if (canonicalJson((authority as unknown as Record<string, unknown>)[key]) !== canonicalJson(value)) {
      throw new ContractFault(`activation fence mismatch: ${key}`, "AUTHORITY_TOKEN_FENCE_MISMATCH");
    }
  }
  const issued = activationTime(authority.issued_at, "authority.issued_at");
  const expires = activationTime(authority.expires_at, "authority.expires_at");
  if (now < issued || now > expires) {
    throw new ContractFault("activation authority has expired", "AUTHORITY_TOKEN_EXPIRED");
  }
  return authority;
}

export function activationIdFor(
  resolutionThreadId: string,
  segmentId: string,
  leaseEpoch: number,
): string {
  return `act_${digest({
    resolution_thread_id: resolutionThreadId,
    segment_id: segmentId,
    lease_epoch: leaseEpoch,
  }).slice("sha256:".length, "sha256:".length + 32)}`;
}

export const SEMANTIC_TOOL_NAMES = [
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
  "kazusa_inspect_public_media",
] as const;

const nullableText = { type: ["string", "null"] };
const timeRange = {
  type: "object",
  properties: { start_at: { type: "string" }, end_at: { type: "string" } },
  required: [],
  additionalProperties: false,
};
const pageProperties = { next_page_ref: nullableText };
const boundedResult = { type: "integer", minimum: 1, maximum: 50 };
const memoryKinds = ["profile_fact", "relationship", "commitment", "experience", "world_knowledge"];

export const SEMANTIC_CATALOG: readonly Record<string, unknown>[] = [
  { name: "kazusa_search_conversation_history", input_schema: { type: "object", properties: { query: { type: "string" }, time_range: timeRange, max_results: boundedResult, ...pageProperties }, required: ["query"], additionalProperties: false } },
  { name: "kazusa_read_conversation_entries", input_schema: { type: "object", properties: { conversation_entry_refs: { type: "array", items: { type: "string" }, minItems: 1, maxItems: 50 } }, required: ["conversation_entry_refs"], additionalProperties: false } },
  { name: "kazusa_summarize_conversation_participants", input_schema: { type: "object", properties: { time_range: timeRange, max_people: boundedResult, ...pageProperties }, required: [], additionalProperties: false } },
  { name: "kazusa_search_memories", input_schema: { type: "object", properties: { query: { type: "string" }, subject_scope: { type: "string", enum: ["current_user", "active_character", "shared_world", "all"] }, memory_kinds: { type: "array", items: { type: "string", enum: memoryKinds }, maxItems: 5 }, max_results: boundedResult, ...pageProperties }, required: ["query"], additionalProperties: false } },
  { name: "kazusa_read_memories", input_schema: { type: "object", properties: { memory_refs: { type: "array", items: { type: "string" }, minItems: 1, maxItems: 50 } }, required: ["memory_refs"], additionalProperties: false } },
  { name: "kazusa_remember_information", input_schema: { type: "object", properties: { subject: { type: "string", enum: ["current_user", "active_character", "shared_world"] }, information: { type: "string" }, memory_kind: { type: "string", enum: memoryKinds }, reason: { type: "string" }, provenance: { type: "object", properties: { conversation_entry_ref: { type: "string" }, current_task: { type: "string" } }, required: [], additionalProperties: false, oneOf: [{ required: ["conversation_entry_ref"] }, { required: ["current_task"] }] } }, required: ["subject", "information", "memory_kind", "reason", "provenance"], additionalProperties: false } },
  { name: "kazusa_revise_memory", input_schema: { type: "object", properties: { memory_ref: { type: "string" }, revised_information: { type: "string" }, reason: { type: "string" } }, required: ["memory_ref", "revised_information", "reason"], additionalProperties: false } },
  { name: "kazusa_change_memory_lifecycle", input_schema: { type: "object", properties: { memory_ref: { type: "string" }, transition: { type: "string", enum: ["activate", "complete", "cancel", "archive"] }, reason: { type: "string" } }, required: ["memory_ref", "transition", "reason"], additionalProperties: false } },
  { name: "kazusa_find_people_by_name", input_schema: { type: "object", properties: { display_name: { type: "string" }, match_relation: { type: "string", enum: ["exact", "contains", "starts_with", "ends_with"] }, max_results: boundedResult, ...pageProperties }, required: ["display_name", "match_relation"], additionalProperties: false } },
  { name: "kazusa_read_person_profiles", input_schema: { type: "object", properties: { person_refs: { type: "array", items: { type: "string" }, minItems: 1, maxItems: 50 } }, required: ["person_refs"], additionalProperties: false } },
  { name: "kazusa_recall_active_context", input_schema: { type: "object", properties: { kinds: { type: "array", items: { type: "string", enum: ["commitments", "progress", "history", "calendar"] }, minItems: 1, maxItems: 4 }, max_results: boundedResult }, required: ["kinds"], additionalProperties: false } },
  { name: "kazusa_read_calendar_context", input_schema: { type: "object", properties: { view: { type: "string", enum: ["schedules", "recent_runs", "pending_runs"] }, max_results: boundedResult, ...pageProperties }, required: ["view"], additionalProperties: false } },
  { name: "kazusa_inspect_attached_media", input_schema: { type: "object", properties: { attached_media_ref: { type: "string" }, question: { type: "string" } }, required: ["attached_media_ref", "question"], additionalProperties: false } },
  { name: "kazusa_inspect_public_media", input_schema: { type: "object", properties: { public_media_url: { type: "string" }, question: { type: "string" } }, required: ["public_media_url", "question"], additionalProperties: false } },
];

function stripDescriptions(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripDescriptions);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .filter(([key]) => key !== "description")
      .map(([key, item]) => [key, stripDescriptions(item)]));
  }
  return value;
}

export function semanticCatalogProjection(nativeNames: readonly string[] = []): Array<Record<string, unknown>> {
  const native = new Set(nativeNames);
  return SEMANTIC_CATALOG
    .filter((item) => !native.has(String(item.name)))
    .map((item) => ({ name: item.name, input_schema: stripDescriptions(item.input_schema) })) as Array<Record<string, unknown>>;
}

export function semanticCatalogDigest(nativeNames: readonly string[] = []): string {
  return digest(semanticCatalogProjection(nativeNames));
}

export interface CatalogSchema {
  name: string;
  input_schema?: unknown;
  parameters?: unknown;
}

export function nativeCatalogProjection(schemas: readonly CatalogSchema[]): Array<Record<string, unknown>> {
  return schemas
    .map((schema) => ({ name: schema.name, input_schema: stripDescriptions(schema.input_schema ?? schema.parameters) }))
    .sort((left, right) => left.name.localeCompare(right.name));
}

export function nativeCatalogDigest(schemas: readonly CatalogSchema[]): string {
  return digest(nativeCatalogProjection(schemas));
}

export function publishedCatalogProjection(
  nativeSchemas: readonly CatalogSchema[],
  submitSchema: CatalogSchema,
  semanticNames: readonly string[] = [],
): Array<Record<string, unknown>> {
  const native = nativeCatalogProjection(nativeSchemas);
  const nativeNames = native.map((schema) => String(schema.name));
  const semantic = semanticCatalogProjection([...nativeNames, ...semanticNames]);
  const entries = [...native, ...semantic];
  if (!entries.some((entry) => entry.name === submitSchema.name)) {
    entries.push({ name: submitSchema.name, input_schema: stripDescriptions(submitSchema.input_schema ?? submitSchema.parameters) });
  }
  return entries.sort((left, right) => String(left.name).localeCompare(String(right.name)));
}

export function publishedCatalogDigest(
  nativeSchemas: readonly CatalogSchema[],
  submitSchema: CatalogSchema,
  semanticNames: readonly string[] = [],
): string {
  return digest(publishedCatalogProjection(nativeSchemas, submitSchema, semanticNames));
}

export interface ResolutionModelInput {
  objective: string;
  facts: string[];
}

export interface ResolutionIntake {
  schema_version: "dsh_resolution_intake.v2";
  mode: "start" | "continue";
  request_id: string;
  operation_id: string;
  operation_payload_digest: string;
  resolution_thread_id: string;
  segment_id: string;
  brain_conversation_ref: string;
  workspace_root: string;
  route_digest: string;
  model_input: ResolutionModelInput;
  semantic_tool_authority: {
    catalog_digest: string;
    token: string;
  };
  interaction_authority: {
    issuer: string;
    scope_fingerprint: string;
    audience_fingerprint: string;
  };
}

// Internal call sites use the intake identity directly. This alias keeps the
// name descriptive where a terminal receipt needs the canonical request.
export type ResolutionRuntime = ResolutionIntake;

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

export interface EvidenceReceipt extends Record<string, unknown> {
  schema_version: "evidence_receipt.v2";
  resolution_thread_id: string;
  segment_id: string;
  scope_fingerprint: string;
  audience_fingerprint: string;
  policy_epoch: string;
  evidence_id: string;
  source_kind: string;
  semantic_ref: string;
  content_digest: string;
  provenance: { tool_name: string };
}

export interface TerminalReceipt {
  kind: "terminal_resolution_v2";
  schema_version: "2";
  call_id: string;
  operation_id: string;
  operation_payload_digest: string;
  request_id: string;
  resolution_thread_id: string;
  segment_id: string;
  activation_id: string;
  lease_epoch: number;
  brain_conversation_ref: string;
  workspace_root: string;
  route_digest: string;
  scope_fingerprint: string;
  catalog_digest: string;
  interaction_issuer: string;
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
  if (typeof value !== "string" || value.length === 0) {
    throw new ContractFault(`${name} must be a non-empty string`);
  }
  return value;
}

function integer(value: unknown, name: string, minimum = 0): number {
  if (!Number.isInteger(value) || (value as number) < minimum) {
    throw new ContractFault(`${name} must be an integer >= ${minimum}`);
  }
  return value as number;
}

function texts(value: unknown, name: string, maximum = 64): string[] {
  if (!Array.isArray(value) || value.length > maximum) {
    throw new ContractFault(`${name} must be a bounded list`);
  }
  const result = value.map((item) => text(item, name));
  if (new Set(result).size !== result.length) {
    throw new ContractFault(`${name} must contain unique values`);
  }
  return result;
}

function absolutePath(value: string, name: string): string {
  const isWindowsAbsolute = /^[A-Za-z]:[\\/]/u.test(value);
  if (!isWindowsAbsolute && !value.startsWith("/")) {
    throw new ContractFault(`${name} must be an absolute path`);
  }
  const normalized = (isWindowsAbsolute ? win32.normalize(value) : posix.normalize(value))
    .replaceAll("\\", "/");
  if (normalized !== value.replaceAll("\\", "/")) {
    throw new ContractFault(`${name} must be canonical`);
  }
  return value.replaceAll("\\", "/");
}

export function validateRuntime(value: unknown): ResolutionRuntime {
  return validateIntake(value);
}

export function validateIntake(value: unknown): ResolutionIntake {
  const row = exact(value, [
    "schema_version", "mode", "request_id", "operation_id",
    "operation_payload_digest", "resolution_thread_id", "segment_id",
    "brain_conversation_ref", "workspace_root", "route_digest", "model_input",
    "semantic_tool_authority", "interaction_authority",
  ], "intake");
  if (row.schema_version !== "dsh_resolution_intake.v2") {
    throw new ContractFault("intake schema_version is unsupported");
  }
  if (row.mode !== "start" && row.mode !== "continue") {
    throw new ContractFault("intake mode is unsupported");
  }
  const textFields = [
    "request_id", "operation_id", "operation_payload_digest",
    "resolution_thread_id", "segment_id", "brain_conversation_ref", "route_digest",
  ] as const;
  for (const key of textFields) text(row[key], `intake.${key}`);
  const workspace = text(row.workspace_root, "intake.workspace_root");
  absolutePath(workspace, "intake.workspace_root");

  const model = exact(row.model_input, ["objective", "facts"], "model_input");
  text(model.objective, "model_input.objective");
  const facts = texts(model.facts, "model_input.facts");

  const semanticAuthority = exact(
    row.semantic_tool_authority,
    ["catalog_digest", "token"],
    "semantic_tool_authority",
  );
  text(semanticAuthority.catalog_digest, "semantic_tool_authority.catalog_digest");
  text(semanticAuthority.token, "semantic_tool_authority.token");
  const interactionAuthority = exact(
    row.interaction_authority,
    ["issuer", "scope_fingerprint", "audience_fingerprint"],
    "interaction_authority",
  );
  text(interactionAuthority.issuer, "interaction_authority.issuer");
  text(interactionAuthority.scope_fingerprint, "interaction_authority.scope_fingerprint");
  text(interactionAuthority.audience_fingerprint, "interaction_authority.audience_fingerprint");

  return {
    schema_version: "dsh_resolution_intake.v2",
    mode: row.mode,
    request_id: row.request_id as string,
    operation_id: row.operation_id as string,
    operation_payload_digest: row.operation_payload_digest as string,
    resolution_thread_id: row.resolution_thread_id as string,
    segment_id: row.segment_id as string,
    brain_conversation_ref: row.brain_conversation_ref as string,
    workspace_root: workspace.replaceAll("\\", "/"),
    route_digest: row.route_digest as string,
    model_input: { objective: model.objective as string, facts },
    semantic_tool_authority: {
      catalog_digest: semanticAuthority.catalog_digest as string,
      token: semanticAuthority.token as string,
    },
    interaction_authority: {
      issuer: interactionAuthority.issuer as string,
      scope_fingerprint: interactionAuthority.scope_fingerprint as string,
      audience_fingerprint: interactionAuthority.audience_fingerprint as string,
    },
  };
}

export function validateSubmitResolution(value: unknown): SubmitResolution {
  const row = exact(value, [
    "status", "summary", "findings", "completed_subgoals", "remaining_needs",
    "clarification_request", "approval_request", "artifact_refs", "warnings",
  ], "submit_resolution");
  const statuses = new Set([
    "resolved", "partial", "needs_user_input", "approval_required", "unavailable", "failed",
  ]);
  if (!statuses.has(String(row.status))) {
    throw new ContractFault("submit_resolution.status is unsupported");
  }
  text(row.summary, "submit_resolution.summary");
  if (!Array.isArray(row.findings) || row.findings.length > 64) {
    throw new ContractFault("submit_resolution.findings must be bounded");
  }
  row.findings.forEach((item) => object(item, "submit_resolution.finding"));
  for (const key of ["completed_subgoals", "remaining_needs", "artifact_refs", "warnings"] as const) {
    texts(row[key], `submit_resolution.${key}`);
  }
  if (row.status === "needs_user_input" && row.clarification_request === null) {
    throw new ContractFault("clarification_request is required");
  }
  if (row.status === "approval_required" && row.approval_request === null) {
    throw new ContractFault("approval_request is required");
  }
  if (row.clarification_request !== null) object(row.clarification_request, "clarification_request");
  if (row.approval_request !== null) object(row.approval_request, "approval_request");
  return row as unknown as SubmitResolution;
}

export function validateEvidenceReceipt(value: unknown): EvidenceReceipt {
  const row = exact(value, [
    "schema_version", "resolution_thread_id", "segment_id", "scope_fingerprint",
    "audience_fingerprint", "policy_epoch", "evidence_id", "source_kind",
    "semantic_ref", "content_digest", "provenance",
  ], "evidence receipt");
  if (row.schema_version !== "evidence_receipt.v2") {
    throw new ContractFault("evidence receipt version is unsupported");
  }
  for (const key of [
    "resolution_thread_id", "segment_id", "scope_fingerprint", "audience_fingerprint",
    "policy_epoch", "evidence_id", "source_kind", "semantic_ref", "content_digest",
  ] as const) text(row[key], `evidence.${key}`);
  const provenance = exact(row.provenance, ["tool_name"], "evidence provenance");
  text(provenance.tool_name, "evidence.provenance.tool_name");
  return {
    schema_version: "evidence_receipt.v2",
    resolution_thread_id: row.resolution_thread_id as string,
    segment_id: row.segment_id as string,
    scope_fingerprint: row.scope_fingerprint as string,
    audience_fingerprint: row.audience_fingerprint as string,
    policy_epoch: row.policy_epoch as string,
    evidence_id: row.evidence_id as string,
    source_kind: row.source_kind as string,
    semantic_ref: row.semantic_ref as string,
    content_digest: row.content_digest as string,
    provenance: { tool_name: provenance.tool_name as string },
  };
}

export function validateTerminalReceipt(value: unknown): TerminalReceipt {
  const row = exact(value, [
    "kind", "schema_version", "call_id", "operation_id", "operation_payload_digest",
    "request_id", "resolution_thread_id", "segment_id", "activation_id", "lease_epoch",
    "brain_conversation_ref", "workspace_root", "route_digest", "scope_fingerprint",
    "catalog_digest", "interaction_issuer", "policy_epoch", "terminal", "terminal_digest",
  ], "terminal receipt");
  if (row.kind !== "terminal_resolution_v2" || row.schema_version !== "2") {
    throw new ContractFault("terminal receipt version is unsupported");
  }
  for (const key of [
    "call_id", "operation_id", "operation_payload_digest", "request_id",
    "resolution_thread_id", "segment_id", "activation_id", "brain_conversation_ref",
    "workspace_root", "route_digest", "scope_fingerprint", "catalog_digest",
    "interaction_issuer", "policy_epoch", "terminal_digest",
  ] as const) text(row[key], `terminal.${key}`);
  absolutePath(row.workspace_root as string, "terminal.workspace_root");
  const leaseEpoch = integer(row.lease_epoch, "terminal.lease_epoch", 1);
  return {
    kind: "terminal_resolution_v2",
    schema_version: "2",
    call_id: row.call_id as string,
    operation_id: row.operation_id as string,
    operation_payload_digest: row.operation_payload_digest as string,
    request_id: row.request_id as string,
    resolution_thread_id: row.resolution_thread_id as string,
    segment_id: row.segment_id as string,
    activation_id: row.activation_id as string,
    lease_epoch: leaseEpoch,
    brain_conversation_ref: row.brain_conversation_ref as string,
    workspace_root: row.workspace_root as string,
    route_digest: row.route_digest as string,
    scope_fingerprint: row.scope_fingerprint as string,
    catalog_digest: row.catalog_digest as string,
    interaction_issuer: row.interaction_issuer as string,
    policy_epoch: row.policy_epoch as string,
    terminal: validateSubmitResolution(row.terminal),
    terminal_digest: row.terminal_digest as string,
  };
}

export function validateMutationFence(value: unknown): {
  operation_id: string;
  operation_payload_digest: string;
  activation_id: string;
  lease_epoch: number;
} {
  const row = exact(value, [
    "operation_id", "operation_payload_digest", "activation_id", "lease_epoch",
  ], "mutation fence");
  return {
    operation_id: text(row.operation_id, "operation_id"),
    operation_payload_digest: text(row.operation_payload_digest, "operation_payload_digest"),
    activation_id: text(row.activation_id, "activation_id"),
    lease_epoch: integer(row.lease_epoch, "lease_epoch", 1),
  };
}

export function validateExhaust(value: unknown): JsonObject {
  const row = object(value, "exhaust");
  if (!new Set(["terminal", "checkpointed", "runtime_fault", "canceled"]).has(String(row.kind))) {
    throw new ContractFault("exhaust kind is unsupported");
  }
  if (row.kind === "terminal") validateSubmitResolution(row.terminal);
  return row;
}
