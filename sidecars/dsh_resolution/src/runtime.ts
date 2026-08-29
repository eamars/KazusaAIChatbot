import { createHash, randomUUID } from "node:crypto";

import {
  type JsonObject,
  type ResolutionIntake,
  validateIntake,
} from "./contracts.js";
import { OperationRegistry } from "./operations.js";
import { commitTerminalResolution, replayTerminalExhaust } from "./submit_resolution.js";
import { evaluateAssistantToolStep } from "./terminal_policy.js";

export { autonomousRuntimePolicy, evaluateAssistantToolStep } from "./terminal_policy.js";

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
  intake: ResolutionIntake;
  activationId: string;
  leaseEpoch: number;
  disposition: "admitted_active" | "checkpointed" | "terminal" | "canceled" | "faulted";
  events: Record<string, unknown>[];
  toolExecutions: number;
  terminalToolExecutions: number;
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
  diagnostics: {
    tool_executions: number;
    terminal_tool_executions: number;
  };
}

class SessionEventStore {
  constructor(private readonly state: SessionState) {}

  async append(event: Record<string, unknown>): Promise<number> {
    this.state.events.push(structuredClone(event));
    return this.state.events.length;
  }

  async flush(): Promise<void> {}
}

export function compatibleSegment(
  current: ResolutionIntake,
  candidate: ResolutionIntake,
): boolean {
  return current.brain_conversation_ref === candidate.brain_conversation_ref
    && current.workspace_root === candidate.workspace_root
    && current.route_digest === candidate.route_digest
    && current.semantic_tool_authority.catalog_digest
      === candidate.semantic_tool_authority.catalog_digest
    && current.interaction_authority.issuer === candidate.interaction_authority.issuer
    && current.interaction_authority.scope_fingerprint
      === candidate.interaction_authority.scope_fingerprint;
}

function callRows(step: ScriptStep): Array<{ name: string; arguments: Record<string, unknown> }> {
  const rawCalls = Array.isArray(step.calls)
    ? step.calls
    : step.name === undefined ? [] : [step];
  return rawCalls.flatMap((raw) => {
    if (raw === null || typeof raw !== "object" || Array.isArray(raw)) return [];
    const row = raw as Record<string, unknown>;
    if (typeof row.name !== "string" || row.name.length === 0) return [];
    const args = row.arguments;
    return [{
      name: row.name,
      arguments: args !== null && typeof args === "object" && !Array.isArray(args)
        ? args as Record<string, unknown>
        : {},
    }];
  });
}

export class ResolutionSidecarRuntime {
  readonly operations = new OperationRegistry();
  private readonly sessions = new Map<string, SessionState>();
  private readonly sessionByThreadSegment = new Map<string, string>();
  private readonly inFlight = new Map<string, Promise<RuntimeResult>>();
  private readonly deferredInteractions = new Map<string, RuntimeResult>();

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
      const state = this.makeState(undefined, "restored", 1);
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

  static forProduction(
    liveExecutor: (
      method: "resolution.open" | "resolution.continue",
      intake: ResolutionIntake,
      activationId: string,
      leaseEpoch: number,
    ) => Promise<RuntimeResult>,
    controls: ProductionControls,
  ): ResolutionSidecarRuntime {
    return new ResolutionSidecarRuntime([], undefined, [], liveExecutor, controls);
  }

  persistedEventCount(): number {
    return [...this.sessions.values()].reduce((total, state) => total + state.events.length, 0);
  }

  interactionExecutionCount(interactionId: string): number {
    return this.deferredInteractions.has(interactionId) ? 1 : 0;
  }

  async deferInteraction(request: {
    interaction_id: string;
    operation_id: string;
    operation_payload_digest: string;
    resolution_thread_id: string;
    segment_id: string;
    activation_id: string;
    lease_epoch: number;
  }): Promise<RuntimeResult> {
    const existing = this.deferredInteractions.get(request.interaction_id);
    if (existing !== undefined) return structuredClone(existing);
    const result: RuntimeResult = {
      disposition: "checkpointed",
      session_id: `interaction-${request.interaction_id}`,
      segment_id: request.segment_id,
      activation_id: request.activation_id,
      lease_epoch: request.lease_epoch,
      exhaust: { kind: "checkpointed" },
      diagnostics: { tool_executions: 0, terminal_tool_executions: 0 },
    };
    this.deferredInteractions.set(request.interaction_id, result);
    return structuredClone(result);
  }

  async open(intakeValue: unknown, activationId: string, leaseEpoch: number): Promise<RuntimeResult> {
    const intake = validateIntake(intakeValue);
    if (this.liveExecutor !== undefined) {
      return this.runProduction("resolution.open", intake, activationId, leaseEpoch);
    }
    const admitted = this.operations.admit(
      intake.operation_id,
      intake.operation_payload_digest,
      "resolution.open",
    );
    if (admitted.result !== undefined) return structuredClone(admitted.result) as RuntimeResult;
    const joined = this.inFlight.get(intake.operation_id);
    if (joined !== undefined) return joined;
    const key = this.key(intake.resolution_thread_id, intake.segment_id);
    const existingId = this.sessionByThreadSegment.get(key);
    if (existingId !== undefined) {
      const existing = this.requiredSession(existingId);
      if (existing.disposition === "admitted_active") throw new Error("duplicate live activation");
    }
    const state = this.makeState(intake, activationId, leaseEpoch);
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
      intake.operation_id,
      intake.operation_payload_digest,
      "resolution.continue",
    );
    if (admitted.result !== undefined) return structuredClone(admitted.result) as RuntimeResult;
    const joined = this.inFlight.get(intake.operation_id);
    if (joined !== undefined) return joined;
    const state = this.requiredByIdentity(intake.resolution_thread_id, intake.segment_id);
    if (!compatibleSegment(state.intake, intake)) throw new Error("segment compatibility mismatch");
    if (leaseEpoch <= state.leaseEpoch) throw new Error("lease epoch must increase for takeover");
    state.activationId = activationId;
    state.leaseEpoch = leaseEpoch;
    state.intake = intake;
    state.disposition = "admitted_active";
    return this.runAdmitted(intake, state);
  }

  async amend(
    threadId: string,
    segmentId: string,
    activationId: string,
    leaseEpoch: number,
    amendment: unknown,
  ): Promise<RuntimeResult> {
    if (amendment === null || typeof amendment !== "object") throw new Error("amendment must be an object");
    if (this.productionControls !== undefined) {
      return await this.productionControls.amend(
        threadId, segmentId, activationId, leaseEpoch, amendment as JsonObject,
      ) as RuntimeResult;
    }
    return this.view(this.assertFence(threadId, segmentId, activationId, leaseEpoch));
  }

  async requestCheckpoint(
    threadId: string,
    segmentId: string,
    activationId: string,
    leaseEpoch: number,
  ): Promise<RuntimeResult> {
    if (this.productionControls !== undefined) {
      return await this.productionControls.checkpoint(threadId, segmentId, activationId, leaseEpoch) as RuntimeResult;
    }
    const state = this.assertFence(threadId, segmentId, activationId, leaseEpoch);
    state.disposition = "checkpointed";
    return this.view(state);
  }

  async cancel(
    threadId: string,
    segmentId: string,
    activationId: string,
    leaseEpoch: number,
  ): Promise<RuntimeResult> {
    if (this.productionControls !== undefined) {
      return await this.productionControls.cancel(threadId, segmentId, activationId, leaseEpoch) as RuntimeResult;
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

  async disposeActivation(
    threadId: string,
    segmentId: string,
    activationId: string,
    leaseEpoch: number,
  ): Promise<void> {
    if (this.productionControls !== undefined) {
      await this.productionControls.disposeActivation(threadId, segmentId, activationId, leaseEpoch);
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
    this.operations.beginExecution(intake.operation_id);
    let corrections = 0;
    let stepIndex = 0;
    while (corrections <= 2) {
      const step = this.script[stepIndex] ?? {};
      stepIndex += 1;
      if (step.wait === true) {
        state.disposition = "admitted_active";
        return this.view(state);
      }
      if (step.invalid_terminal_receipt === true) {
        state.events.push({ type: "tool/result", meta: { kazusa: { kind: "terminal_resolution_v2" } } });
        const result = { ...this.view(state), exhaust: replayTerminalExhaust(state.events) };
        state.disposition = "faulted";
        this.operations.commit(intake.operation_id, "faulted", result);
        return result;
      }
      const calls = callRows(step);
      const decision = evaluateAssistantToolStep(calls);
      if (!decision.accepted) {
        corrections += 1;
        continue;
      }
      if (decision.terminal) {
        const call = calls[0];
        if (call === undefined) {
          corrections += 1;
          continue;
        }
        try {
          const exhaust = await commitTerminalResolution(
            new SessionEventStore(state),
            intake,
            state.activationId,
            state.leaseEpoch,
            `call-${randomUUID()}`,
            call.arguments,
            [],
          );
          if (exhaust.kind !== "terminal") {
            corrections += 1;
            continue;
          }
          state.toolExecutions += 1;
          state.terminalToolExecutions += 1;
          state.disposition = "terminal";
          const result = { ...this.view(state), exhaust };
          this.operations.commit(intake.operation_id, "terminal", result);
          return result;
        } catch {
          corrections += 1;
        }
      } else if (calls.length > 0) {
        state.toolExecutions += calls.length;
        for (const _call of calls) this.onTool?.();
      } else {
        corrections += 1;
      }
    }
    state.disposition = "faulted";
    const result = {
      ...this.view(state),
      exhaust: { kind: "runtime_fault", fault: { code: "RESOLVER_ACTION_CONTRACT_EXHAUSTED" } },
    };
    this.operations.commit(intake.operation_id, "faulted", result);
    return result;
  }

  private runAdmitted(intake: ResolutionIntake, state: SessionState): Promise<RuntimeResult> {
    const execution = this.execute(intake, state).finally(() => {
      this.inFlight.delete(intake.operation_id);
    });
    this.inFlight.set(intake.operation_id, execution);
    return execution;
  }

  private runProduction(
    method: "resolution.open" | "resolution.continue",
    intake: ResolutionIntake,
    activationId: string,
    leaseEpoch: number,
  ): Promise<RuntimeResult> {
    const operationId = intake.operation_id;
    const payloadDigest = intake.operation_payload_digest;
    const admitted = this.operations.admit(operationId, payloadDigest, method);
    if (admitted.result !== undefined) return Promise.resolve(structuredClone(admitted.result) as RuntimeResult);
    const joined = this.inFlight.get(operationId);
    if (joined !== undefined) return joined;
    if (this.liveExecutor === undefined || this.productionControls === undefined) {
      throw new Error("production runtime is incomplete");
    }
    const execution = (async (): Promise<RuntimeResult> => {
      const durable = await this.productionControls?.inspect(operationId, payloadDigest);
      if (durable !== undefined && durable.disposition !== "not_admitted") {
        const restored = durable as RuntimeResult;
        this.operations.commit(operationId, restored.disposition, restored);
        return restored;
      }
      this.operations.beginExecution(operationId);
      const result = await this.liveExecutor?.(method, intake, activationId, leaseEpoch);
      if (result === undefined) throw new Error("production executor returned no result");
      this.operations.commit(operationId, result.disposition, result);
      return result;
    })().finally(() => {
      this.inFlight.delete(operationId);
    });
    this.inFlight.set(operationId, execution);
    return execution;
  }

  private makeState(
    intake: ResolutionIntake | undefined,
    activationId: string,
    leaseEpoch: number,
  ): SessionState {
    const identity = intake === undefined
      ? `restored-${randomUUID()}`
      : `${intake.resolution_thread_id}\u0000${intake.segment_id}`;
    const sessionId = this.liveExecutor === undefined
      ? `session-${randomUUID()}`
      : `kazusa-resolution-${createHash("sha256").update(identity).digest("hex").slice(0, 32)}`;
    return {
      sessionId,
      intake: intake ?? ({} as ResolutionIntake),
      activationId,
      leaseEpoch,
      disposition: "admitted_active",
      events: [],
      toolExecutions: 0,
      terminalToolExecutions: 0,
    };
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

  private assertFence(
    threadId: string,
    segmentId: string,
    activationId: string,
    leaseEpoch: number,
  ): SessionState {
    const state = this.requiredByIdentity(threadId, segmentId);
    if (state.activationId !== activationId || state.leaseEpoch !== leaseEpoch) {
      throw new Error("STALE_ACTIVATION_OR_LEASE");
    }
    return state;
  }

  private view(state: SessionState): RuntimeResult {
    return {
      disposition: state.disposition,
      session_id: state.sessionId,
      segment_id: state.intake.segment_id,
      activation_id: state.activationId,
      lease_epoch: state.leaseEpoch,
      exhaust: { kind: state.disposition === "admitted_active" ? "checkpointed" : state.disposition },
      diagnostics: {
        tool_executions: state.toolExecutions,
        terminal_tool_executions: state.terminalToolExecutions,
      },
    };
  }
}
