import { describe, expect, it } from "vitest";

import { commitTerminalResolution, replayTerminalExhaust } from "../src/submit_resolution.js";
import { validRuntime, validSubmit } from "./contracts.spec.js";

class Store {
  events: Array<Record<string, unknown>> = [];
  flushed = false;
  async append(event: Record<string, unknown>) { this.events.push(event); return this.events.length; }
  async flush() { this.flushed = true; }
}

describe("submit_resolution", () => {
  it("commits the complete terminal receipt before returning exhaust", async () => {
    const store = new Store();
    const exhaust = await commitTerminalResolution(store, validRuntime(), "act_1", 1, "call_1", validSubmit(), []);
    expect(store.flushed).toBe(true);
    expect(store.events[0]?.meta).toHaveProperty("kazusa.kind", "terminal_resolution_v1");
    expect(exhaust.kind).toBe("terminal");
  });

  it("replays exact terminal exhaust after restart without model execution", async () => {
    const store = new Store();
    const first = await commitTerminalResolution(store, validRuntime(), "act_1", 1, "call_1", validSubmit(), []);
    expect(replayTerminalExhaust(store.events)).toEqual(first);
  });

  it("rejects missing or invalid terminal receipts as runtime faults", () => {
    expect(replayTerminalExhaust([])).toMatchObject({ kind: "runtime_fault", fault: { code: "TERMINAL_RECEIPT_MISSING" } });
    expect(replayTerminalExhaust([{ type: "tool/result", meta: { kazusa: { kind: "terminal_resolution_v1" } } }]))
      .toMatchObject({ kind: "runtime_fault" });
  });
});
