import { describe, expect, it } from "vitest";

import { RPC_PROTOCOL_VERSION } from "../src/contracts.js";
import { OperationRegistry } from "../src/operations.js";
import { dispatchRpc, RpcFault } from "../src/rpc.js";

function request(method: string, params: Record<string, unknown> = {}) {
  return { jsonrpc: "2.0", id: "rpc_1", method, params: { protocol_version: RPC_PROTOCOL_VERSION, ...params } };
}

describe("rpc", () => {
  it("authenticates versioned request and response frames", async () => {
    const response = await dispatchRpc(request("system.health"), "Bearer secret", {
      token: "secret",
      operations: new OperationRegistry(),
      handlers: { "system.health": async () => ({ status: "ok" }) },
    });
    expect(response).toMatchObject({ jsonrpc: "2.0", id: "rpc_1", protocol_version: RPC_PROTOCOL_VERSION });
  });

  it("rejects non-loopback unauthenticated unknown and malformed requests", async () => {
    const context = { token: "secret", operations: new OperationRegistry(), handlers: {} };
    await expect(dispatchRpc(request("system.health"), "Bearer wrong", context)).rejects.toBeInstanceOf(RpcFault);
    await expect(dispatchRpc(request("unknown"), "Bearer secret", context)).rejects.toThrow(/method/);
    await expect(dispatchRpc({ jsonrpc: "1.0" }, "Bearer secret", context)).rejects.toThrow(/frame|jsonrpc/);
  });

  it("admits one operation for duplicate ids with the same payload digest", () => {
    const registry = new OperationRegistry();
    const first = registry.admit("op_1", "sha256:p", "resolution.open");
    const second = registry.admit("op_1", "sha256:p", "resolution.open");
    expect(second).toBe(first);
    expect(registry.admissionCount("op_1")).toBe(1);
  });

  it("rejects operation id reuse with a different payload digest", () => {
    const registry = new OperationRegistry();
    registry.admit("op_1", "sha256:p", "resolution.open");
    expect(() => registry.admit("op_1", "sha256:other", "resolution.open"))
      .toThrow(/OPERATION_ID_REUSE_MISMATCH/);
  });

  it("inspects not admitted active and committed outcomes after transport loss", () => {
    const registry = new OperationRegistry();
    expect(registry.inspect("missing").disposition).toBe("not_admitted");
    registry.admit("op_1", "sha256:p", "resolution.open");
    expect(registry.inspect("op_1").disposition).toBe("admitted_active");
    registry.commit("op_1", "terminal", { exhaust: { kind: "terminal" } });
    expect(registry.inspect("op_1").disposition).toBe("terminal");
  });

  it("serves checkpoint and cancel concurrently with a pending execution request", async () => {
    let release!: () => void;
    const pending = new Promise<void>((resolve) => { release = resolve; });
    const calls: string[] = [];
    const context = {
      token: "secret",
      operations: new OperationRegistry(),
      handlers: {
        "resolution.open": async () => { calls.push("open"); await pending; return {}; },
        "resolution.request_checkpoint": async () => { calls.push("checkpoint"); return {}; },
        "resolution.cancel": async () => { calls.push("cancel"); return {}; },
      },
    };
    const identity = {
      operation_id: "op_control",
      operation_payload_digest: "sha256:control",
      resolution_thread_id: "res_1",
      segment_id: "seg_1",
      activation_id: "act_1",
      lease_epoch: 1,
    };
    const open = dispatchRpc(request("resolution.open", {
      operation_id: "op_open", operation_payload_digest: "sha256:open",
      activation_id: "act_1", lease_epoch: 1, intake: {},
    }), "Bearer secret", context);
    await dispatchRpc(request("resolution.request_checkpoint", identity), "Bearer secret", context);
    await dispatchRpc(request("resolution.cancel", {
      ...identity,
      operation_id: "op_cancel",
      operation_payload_digest: "sha256:cancel",
    }), "Bearer secret", context);
    release();
    await open;
    expect(calls).toEqual(["open", "checkpoint", "cancel"]);
  });
});

describe("V2 RPC", () => {
  it("requires loopback bearer and V2 protocol", async () => {
    const contracts = await import("../src/contracts.js");
    const rpcModule = await import("../src/rpc.js");
    const server = new rpcModule.RpcServer({ token: "rpc-secret" });
    const frame = {
      jsonrpc: "2.0",
      id: "rpc-v2",
      method: "system.health",
      params: { protocol_version: "kazusa.dsh-resolution-rpc.v2" },
    };
    const response = server.dispatch(frame, {
      remoteAddress: "127.0.0.1",
      authorization: "Bearer rpc-secret",
    });
    expect(response.protocol_version).toBe("kazusa.dsh-resolution-rpc.v2");
    expect(contracts.RPC_PROTOCOL_VERSION).toBe(response.protocol_version);
    expect(() => server.dispatch(frame, {
      remoteAddress: "192.0.2.10",
      authorization: "Bearer rpc-secret",
    })).toThrow(/loopback|authentication/i);
    expect(() => new rpcModule.RpcClient(
      "http://192.0.2.10/rpc",
      "rpc-secret",
    )).toThrow(/loopback/i);
  });
});
