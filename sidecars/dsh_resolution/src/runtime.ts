import { createHash, randomUUID } from "node:crypto";

import {
  type JsonObject,
  type ResolutionIntake,
  type ResolutionRuntime,
  validateIntake,
} from "./contracts.js";
import { OperationRegistry } from "./operations.js";
import { commitTerminalResolution, replayTerminalExhaust } from "./submit_resolution.js";

interface ScriptStep extends JsonObject {
  name?: string;
  arguments?: unknown;
  calls?: unknown[];
  text?: string;
  wait?: boolean;
  invalid_terminal_receipt?: boolean;
}

interface SessionState {
  sessionId: string;
  runtime: ResolutionRuntime;
  activationId: string;
  leaseEpoch: number;
  disposition: "admitted_active" | "checkpointed" | "terminal" | "canceled" | "faulted";
  events: Record<string, unknown>[];
}

interface ProductionControls {
  checkpoint(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<JsonObject>;
  cancel(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<JsonObject>;
  amend(threadId: string, segmentId: string, activationId: string, leaseEpoch: number, amendment: JsonObject): Promise<JsonObject>;
  disposeActivation(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<void>;
  inspect(operationId: string, payloadDigest: string): Promise<JsonObject>;
}

export interface RuntimeResult extends JsonObject {
  disposition: SessionState["disposition"];
  session_id: string;
  segment_id: string;
  activation_id: string;
  lease_epoch: number;
  exhaust: JsonObject;
}

class SessionEventStore {
  constructor(private readonly state: SessionState) {}
  async append(event: Record<string, unknown>): Promise<number> {
    this.state.events.push(structuredClone(event));
    return this.state.events.length;
  }
  async flush(): Promise<void> {}
}

export function compatibleSegment(current: ResolutionRuntime, candidate: ResolutionRuntime): boolean {
  const keys = [
    "scope_fingerprint", "audience_fingerprint", "resolver_profile_version", "dsh_release",
    "session_store_epoch", "model_route", "tool_catalog_digest", "policy_epoch",
  ] as const;
  return keys.every((key) => current[key] === candidate[key]);
}

export class ResolutionSidecarRuntime {
  readonly operations = new OperationRegistry();
  private readonly sessions = new Map<string, SessionState>();
  private readonly sessionByThreadSegment = new Map<string, string>();
  private readonly inFlight = new Map<string, Promise<RuntimeResult>>();

  private constructor(
    private readonly script: ScriptStep[],
    private readonly onTool?: () => void,
    restoredEvents: Record<string, unknown>[] = [],
    private readonly liveExecutor?: (
      method: "resolution.open" | "resolution.continue",
      intake: ResolutionIntake,
      activationId: string,
      leaseEpoch: number,
    ) => Promise<RuntimeResult>,
    private readonly productionControls?: ProductionControls,
  ) {
    if (restoredEvents.length > 0) {
      const state = this.makeState({} as ResolutionRuntime, "restored", 1);
      state.events = restoredEvents.map((event) => structuredClone(event));
      this.sessions.set(state.sessionId, state);
    }
  }

  static forTests(script: ScriptStep[], onTool?: () => void): ResolutionSidecarRuntime {
    return new ResolutionSidecarRuntime(structuredClone(script), onTool);
  }

  static restoreForTests(events: Record<string, unknown>[]): ResolutionSidecarRuntime {
    return new ResolutionSidecarRuntime([], undefined, events);
  }

  static forProduction(liveExecutor: (
    method: "resolution.open" | "resolution.continue",
    intake: ResolutionIntake,
    activationId: string,
    leaseEpoch: number,
  ) => Promise<RuntimeResult>, controls: ProductionControls): ResolutionSidecarRuntime {
    return new ResolutionSidecarRuntime([], undefined, [], liveExecutor, controls);
  }

  persistedEventCount(): number {
    return [...this.sessions.values()].reduce((total, state) => total + state.events.length, 0);
  }

  async open(intakeValue: unknown, activationId: string, leaseEpoch: number): Promise<RuntimeResult> {
    const intake = validateIntake(intakeValue);
    if (this.liveExecutor !== undefined) {
      return this.runProduction("resolution.open", intake, activationId, leaseEpoch);
    }
    const admitted = this.operations.admit(
      intake.runtime.operation_id,
      intake.runtime.operation_payload_digest,
      "resolution.open",
    );
    if (admitted.result !== undefined) return structuredClone(admitted.result) as RuntimeResult;
    const joined = this.inFlight.get(intake.runtime.operation_id);
    if (joined !== undefined) return joined;
    const key = this.key(intake.runtime.resolution_thread_id, intake.runtime.segment_id);
    const existingId = this.sessionByThreadSegment.get(key);
    if (existingId !== undefined) {
      const existing = this.requiredSession(existingId);
      if (existing.disposition === "admitted_active") throw new Error("duplicate live activation");
    }
    const state = this.makeState(intake.runtime, activationId, leaseEpoch);
    this.sessions.set(state.sessionId, state);
    this.sessionByThreadSegment.set(key, state.sessionId);
    return this.runAdmitted(intake, state);
  }

  async continue(intakeValue: unknown, activationId: string, leaseEpoch: number): Promise<RuntimeResult> {
    const intake = validateIntake(intakeValue);
    if (this.liveExecutor !== undefined) {
      return this.runProduction("resolution.continue", intake, activationId, leaseEpoch);
    }
    const admitted = this.operations.admit(
      intake.runtime.operation_id,
      intake.runtime.operation_payload_digest,
      "resolution.continue",
    );
    if (admitted.result !== undefined) return structuredClone(admitted.result) as RuntimeResult;
    const joined = this.inFlight.get(intake.runtime.operation_id);
    if (joined !== undefined) return joined;
    const state = this.requiredByIdentity(intake.runtime.resolution_thread_id, intake.runtime.segment_id);
    if (!compatibleSegment(state.runtime, intake.runtime)) throw new Error("segment compatibility mismatch");
    if (leaseEpoch <= state.leaseEpoch) throw new Error("lease epoch must increase for takeover");
    state.activationId = activationId;
    state.leaseEpoch = leaseEpoch;
    state.disposition = "admitted_active";
    return this.runAdmitted(intake, state);
  }

  async amend(threadId: string, segmentId: string, activationId: string, leaseEpoch: number, amendment: unknown): Promise<RuntimeResult> {
    if (amendment === null || typeof amendment !== "object") throw new Error("amendment must be an object");
    if (this.productionControls !== undefined) {
      return await this.productionControls.amend(
        threadId,
        segmentId,
        activationId,
        leaseEpoch,
        amendment as JsonObject,
      ) as RuntimeResult;
    }
    const state = this.assertFence(threadId, segmentId, activationId, leaseEpoch);
    return this.view(state);
  }

  async requestCheckpoint(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<RuntimeResult> {
    if (this.productionControls !== undefined) {
      return await this.productionControls.checkpoint(
        threadId,
        segmentId,
        activationId,
        leaseEpoch,
      ) as RuntimeResult;
    }
    const state = this.assertFence(threadId, segmentId, activationId, leaseEpoch);
    state.disposition = "checkpointed";
    return this.view(state);
  }

  async cancel(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<RuntimeResult> {
    if (this.productionControls !== undefined) {
      return await this.productionControls.cancel(
        threadId,
        segmentId,
        activationId,
        leaseEpoch,
      ) as RuntimeResult;
    }
    const state = this.assertFence(threadId, segmentId, activationId, leaseEpoch);
    state.disposition = "canceled";
    return this.view(state);
  }

  async inspect(operationId: string, payloadDigest?: string): Promise<JsonObject> {
    const inspected = this.operations.inspect(operationId, payloadDigest);
    if (inspected.result !== undefined) return inspected.result;
    if (this.productionControls === undefined) return inspected;
    if (payloadDigest === undefined) throw new Error("operation payload digest is required");
    return this.productionControls.inspect(operationId, payloadDigest);
  }

  async disposeActivation(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<void> {
    if (this.productionControls !== undefined) {
      await this.productionControls.disposeActivation(
        threadId,
        segmentId,
        activationId,
        leaseEpoch,
      );
      return;
    }
    const state = this.assertFence(threadId, segmentId, activationId, leaseEpoch);
    if (state.disposition === "admitted_active") state.disposition = "checkpointed";
  }

  renewLease(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): RuntimeResult {
    return this.view(this.assertFence(threadId, segmentId, activationId, leaseEpoch));
  }

  hasSession(sessionId: string): boolean { return this.sessions.has(sessionId); }

  private async execute(intake: ResolutionIntake, state: SessionState): Promise<RuntimeResult> {
    this.operations.beginExecution(intake.runtime.operation_id);
    let corrections = 0;
    let stepIndex = 0;
    while (corrections <= 2) {
      let step: ScriptStep;
      try {
        step = this.script[stepIndex] ?? {};
      } catch (error) {
        if (state.disposition !== "admitted_active") {
          const result = this.view(state);
          this.operations.commit(intake.runtime.operation_id, state.disposition, result);
          return result;
        }
        state.disposition = "faulted";
        const result = {
          ...this.view(state),
          exhaust: {
            kind: "runtime_fault",
            fault: { code: "RESOLVER_ACTION_CONTRACT_EXHAUSTED" },
          },
        };
        this.operations.commit(intake.runtime.operation_id, "faulted", result);
        return result;
      }
      stepIndex += 1;
      if (step.wait === true) {
        state.disposition = "admitted_active";
        const result = this.view(state);
        return result;
      }
      if (step.invalid_terminal_receipt === true) {
        state.events.push({ type: "tool/result", meta: { kazusa: { kind: "terminal_resolution_v1" } } });
        const result = { ...this.view(state), exhaust: replayTerminalExhaust(state.events) };
        state.disposition = "faulted";
        this.operations.commit(intake.runtime.operation_id, "faulted", result);
        return result;
      }
      const calls = Array.isArray(step.calls) ? step.calls : step.name === undefined ? [] : [step];
      if (calls.length !== 1) { corrections += 1; continue; }
      const call = calls[0] as ScriptStep;
      if (call.name !== "submit_resolution") { corrections += 1; continue; }
      this.onTool?.();
      try {
        const exhaust = await commitTerminalResolution(
          new SessionEventStore(state), intake.runtime, state.activationId, state.leaseEpoch,
          `call-${randomUUID()}`, call.arguments, [],
        );
        state.disposition = "terminal";
        const result = { ...this.view(state), exhaust };
        this.operations.commit(intake.runtime.operation_id, "terminal", result);
        return result;
      } catch {
        corrections += 1;
      }
    }
    state.disposition = "faulted";
    const result = {
      ...this.view(state),
      exhaust: { kind: "runtime_fault", fault: { code: "RESOLVER_ACTION_CONTRACT_EXHAUSTED" } },
    };
    this.operations.commit(intake.runtime.operation_id, "faulted", result);
    return result;
  }

  private runAdmitted(intake: ResolutionIntake, state: SessionState): Promise<RuntimeResult> {
    const operationId = intake.runtime.operation_id;
    const execution = this.execute(intake, state).finally(() => {
      this.inFlight.delete(operationId);
    });
    this.inFlight.set(operationId, execution);
    return execution;
  }

  private runProduction(
    method: "resolution.open" | "resolution.continue",
    intake: ResolutionIntake,
    activationId: string,
    leaseEpoch: number,
  ): Promise<RuntimeResult> {
    const operationId = intake.runtime.operation_id;
    const payloadDigest = intake.runtime.operation_payload_digest;
    this.operations.admit(operationId, payloadDigest, method);
    const joined = this.inFlight.get(operationId);
    if (joined !== undefined) return joined;
    if (this.liveExecutor === undefined || this.productionControls === undefined) {
      throw new Error("production runtime is incomplete");
    }
    const execution = (async (): Promise<RuntimeResult> => {
      const durable = await this.productionControls?.inspect(
        operationId,
        payloadDigest,
      );
      if (durable !== undefined && durable.disposition !== "not_admitted") {
        const restored = durable as RuntimeResult;
        this.operations.commit(
          operationId,
          restored.disposition,
          restored,
        );
        return restored;
      }
      this.operations.beginExecution(operationId);
      const result = await this.liveExecutor?.(
        method,
        intake,
        activationId,
        leaseEpoch,
      );
      if (result === undefined) throw new Error("production executor returned no result");
      this.operations.commit(operationId, result.disposition, result);
      return result;
    })().finally(() => {
      this.inFlight.delete(operationId);
    });
    this.inFlight.set(operationId, execution);
    return execution;
  }

  private makeState(runtime: ResolutionRuntime, activationId: string, leaseEpoch: number): SessionState {
    const sessionId = this.liveExecutor === undefined
      ? `session-${randomUUID()}`
      : `kazusa-resolution-${createHash("sha256").update(`${runtime.resolution_thread_id}\u0000${runtime.segment_id}`).digest("hex").slice(0, 32)}`;
    return { sessionId, runtime, activationId, leaseEpoch, disposition: "admitted_active", events: [] };
  }
  private key(threadId: string, segmentId: string): string { return `${threadId}\u0000${segmentId}`; }
  private requiredByIdentity(threadId: string, segmentId: string): SessionState {
    const id = this.sessionByThreadSegment.get(this.key(threadId, segmentId));
    if (id === undefined) throw new Error("session identity is unknown");
    return this.requiredSession(id);
  }
  private requiredSession(id: string): SessionState {
    const state = this.sessions.get(id);
    if (state === undefined) throw new Error("session is unknown");
    return state;
  }
  private assertFence(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): SessionState {
    const state = this.requiredByIdentity(threadId, segmentId);
    if (state.activationId !== activationId || state.leaseEpoch !== leaseEpoch) throw new Error("STALE_ACTIVATION_OR_LEASE");
    return state;
  }
  private view(state: SessionState): RuntimeResult {
    return {
      disposition: state.disposition,
      session_id: state.sessionId,
      segment_id: state.runtime.segment_id,
      activation_id: state.activationId,
      lease_epoch: state.leaseEpoch,
      exhaust: { kind: state.disposition === "admitted_active" ? "checkpointed" : state.disposition },
    };
  }
}
