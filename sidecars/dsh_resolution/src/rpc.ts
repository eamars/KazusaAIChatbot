import { createServer, type Server } from "node:http";

import { type JsonObject, RPC_PROTOCOL_VERSION } from "./contracts.js";
import { OperationReuseFault, type OperationRegistry } from "./operations.js";

export class RpcFault extends Error {
  constructor(message: string, readonly status = 400, readonly code = "RPC_CONTRACT_ERROR") { super(message); }
}

interface RpcContext {
  token: string;
  operations: OperationRegistry;
  handlers: Record<string, (params: JsonObject) => Promise<JsonObject>>;
}

interface RpcFrame { jsonrpc: "2.0"; id: string | number; method: string; params: JsonObject }

function frame(value: unknown): RpcFrame {
  if (value === null || typeof value !== "object" || Array.isArray(value)) throw new RpcFault("request must be an object");
  const row = value as Record<string, unknown>;
  if (JSON.stringify(Object.keys(row).sort()) !== JSON.stringify(["id", "jsonrpc", "method", "params"])) throw new RpcFault("request frame fields are invalid");
  if (row.jsonrpc !== "2.0") throw new RpcFault("jsonrpc must be 2.0");
  if ((typeof row.id !== "string" && typeof row.id !== "number") || typeof row.method !== "string") throw new RpcFault("rpc id and method are required");
  if (row.params === null || typeof row.params !== "object" || Array.isArray(row.params)) throw new RpcFault("rpc params must be an object");
  const params = row.params as JsonObject;
  if (params.protocol_version !== RPC_PROTOCOL_VERSION) throw new RpcFault("rpc protocol version is unsupported");
  return { jsonrpc: "2.0", id: row.id, method: row.method, params };
}

export async function dispatchRpc(value: unknown, authorization: string | undefined, context: RpcContext): Promise<JsonObject> {
  if (authorization !== `Bearer ${context.token}`) throw new RpcFault("authentication failed", 401, "RPC_AUTH_FAILED");
  const request = frame(value);
  const handler = context.handlers[request.method];
  if (handler === undefined) throw new RpcFault("rpc method is unsupported", 404, "RPC_METHOD_NOT_FOUND");
  const params = { ...request.params };
  delete params.protocol_version;
  const allowed: Record<string, readonly string[]> = {
    "system.health": [],
    "resolution.open": ["operation_id", "operation_payload_digest", "activation_id", "lease_epoch", "intake"],
    "resolution.continue": ["operation_id", "operation_payload_digest", "activation_id", "lease_epoch", "intake"],
    "resolution.amend": ["operation_id", "operation_payload_digest", "resolution_thread_id", "segment_id", "activation_id", "lease_epoch", "amendment"],
    "resolution.request_checkpoint": ["operation_id", "operation_payload_digest", "resolution_thread_id", "segment_id", "activation_id", "lease_epoch"],
    "resolution.cancel": ["operation_id", "operation_payload_digest", "resolution_thread_id", "segment_id", "activation_id", "lease_epoch"],
    "resolution.inspect": ["operation_id", "operation_payload_digest"],
    "resolution.dispose_activation": ["operation_id", "operation_payload_digest", "resolution_thread_id", "segment_id", "activation_id", "lease_epoch"],
  };
  const expected = allowed[request.method];
  if (expected === undefined || JSON.stringify(Object.keys(params).sort()) !== JSON.stringify([...expected].sort())) throw new RpcFault("rpc method params are invalid");
  const mutating = request.method !== "system.health" && request.method !== "resolution.inspect";
  let operationId: string | undefined;
  if (mutating) {
    operationId = params.operation_id as string;
    const payloadDigest = params.operation_payload_digest as string;
    try {
      const operation = context.operations.admit(
        operationId,
        payloadDigest,
        request.method,
      );
      if (operation.result !== undefined) {
        return {
          jsonrpc: "2.0",
          id: request.id,
          protocol_version: RPC_PROTOCOL_VERSION,
          result: operation.result,
        };
      }
    } catch (error) {
      if (error instanceof OperationReuseFault) {
        throw new RpcFault(
          error.message,
          409,
          "OPERATION_ID_REUSE_MISMATCH",
        );
      }
      throw error;
    }
  }
  let result: JsonObject;
  try {
    result = await handler(params);
  } catch (error) {
    if (error instanceof OperationReuseFault) {
      throw new RpcFault(
        error.message,
        409,
        "OPERATION_ID_REUSE_MISMATCH",
      );
    }
    throw error;
  }
  if (operationId !== undefined) {
    const candidate = result.disposition;
    const disposition = (
      candidate === "admitted_active"
      || candidate === "checkpointed"
      || candidate === "terminal"
      || candidate === "canceled"
      || candidate === "faulted"
    ) ? candidate : request.method === "resolution.dispose_activation"
      ? "canceled"
      : "admitted_active";
    context.operations.commit(operationId, disposition, result);
  }
  return { jsonrpc: "2.0", id: request.id, protocol_version: RPC_PROTOCOL_VERSION, result };
}

export function createRpcServer(host: string, port: number, context: RpcContext): Server {
  if (host !== "127.0.0.1" && host !== "::1") throw new Error("sidecar must bind to loopback");
  return createServer((request, response) => {
    if (request.method !== "POST" || request.url !== "/rpc") {
      response.writeHead(404).end();
      return;
    }
    const chunks: Buffer[] = [];
    request.on("data", (chunk: Buffer) => {
      chunks.push(chunk);
      if (chunks.reduce((size, item) => size + item.length, 0) > 1_048_576) request.destroy();
    });
    request.on("end", () => {
      void (async () => {
        try {
          const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown;
          const result = await dispatchRpc(body, request.headers.authorization, context);
          response.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify(result));
        } catch (error) {
          const fault = error instanceof RpcFault ? error : new RpcFault(error instanceof Error ? error.message : "rpc failure", 500, "RPC_INTERNAL_ERROR");
          let id: string | number | null = null;
          try {
            const parsed = JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
            if (typeof parsed.id === "string" || typeof parsed.id === "number") id = parsed.id;
          } catch {}
          const message = fault.status === 500 ? "internal sidecar failure" : fault.message;
          response.writeHead(fault.status, { "content-type": "application/json" }).end(JSON.stringify({ jsonrpc: "2.0", id, protocol_version: RPC_PROTOCOL_VERSION, error: { code: fault.code, message } }));
        }
      })();
    });
  }).listen(port, host);
}
