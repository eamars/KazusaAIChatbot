import { type JsonObject, type ResolutionIntake, validateIntake } from "./contracts.js";
import { OperationRegistry } from "./operations.js";

interface ProductionControls {
  checkpoint(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<JsonObject>;
  cancel(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<JsonObject>;
  amend(threadId: string, segmentId: string, activationId: string, leaseEpoch: number, amendment: JsonObject): Promise<JsonObject>;
  disposeActivation(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<void>;
  inspect(operationId: string, payloadDigest: string): Promise<JsonObject>;
}

export interface RuntimeResult extends JsonObject {
  disposition: "admitted_active" | "checkpointed" | "terminal" | "canceled" | "faulted";
  session_id: string;
  segment_id: string;
  activation_id: string;
  lease_epoch: number;
  exhaust: JsonObject;
  diagnostics: { tool_executions: number; terminal_tool_executions: number };
}

type ProductionExecutor = (
  method: "resolution.open" | "resolution.continue",
  intake: ResolutionIntake,
  activationId: string,
  leaseEpoch: number,
) => Promise<RuntimeResult>;

export class ResolutionSidecarRuntime {
  readonly operations = new OperationRegistry();
  private readonly inFlight = new Map<string, Promise<RuntimeResult>>();

  private constructor(
    private readonly liveExecutor: ProductionExecutor,
    private readonly productionControls: ProductionControls,
  ) {}

  static forProduction(executor: ProductionExecutor, controls: ProductionControls): ResolutionSidecarRuntime {
    return new ResolutionSidecarRuntime(executor, controls);
  }

  async open(intakeValue: unknown, activationId: string, leaseEpoch: number): Promise<RuntimeResult> {
    return this.runProduction("resolution.open", validateIntake(intakeValue), activationId, leaseEpoch);
  }

  async continue(intakeValue: unknown, activationId: string, leaseEpoch: number): Promise<RuntimeResult> {
    return this.runProduction("resolution.continue", validateIntake(intakeValue), activationId, leaseEpoch);
  }

  async amend(threadId: string, segmentId: string, activationId: string, leaseEpoch: number, amendment: unknown): Promise<RuntimeResult> {
    if (amendment === null || typeof amendment !== "object") throw new Error("amendment must be an object");
    return await this.productionControls.amend(threadId, segmentId, activationId, leaseEpoch, amendment as JsonObject) as RuntimeResult;
  }

  async requestCheckpoint(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<RuntimeResult> {
    return await this.productionControls.checkpoint(threadId, segmentId, activationId, leaseEpoch) as RuntimeResult;
  }

  async cancel(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<RuntimeResult> {
    return await this.productionControls.cancel(threadId, segmentId, activationId, leaseEpoch) as RuntimeResult;
  }

  async inspect(operationId: string, payloadDigest?: string): Promise<JsonObject> {
    const inspected = this.operations.inspect(operationId, payloadDigest);
    if (inspected.result !== undefined) return inspected.result;
    if (payloadDigest === undefined) throw new Error("operation payload digest is required");
    return this.productionControls.inspect(operationId, payloadDigest);
  }

  async disposeActivation(threadId: string, segmentId: string, activationId: string, leaseEpoch: number): Promise<void> {
    await this.productionControls.disposeActivation(threadId, segmentId, activationId, leaseEpoch);
  }

  private runProduction(method: "resolution.open" | "resolution.continue", intake: ResolutionIntake, activationId: string, leaseEpoch: number): Promise<RuntimeResult> {
    const operationId = intake.operation_id;
    const payloadDigest = intake.operation_payload_digest;
    const admitted = this.operations.admit(operationId, payloadDigest, method);
    if (admitted.result !== undefined) return Promise.resolve(structuredClone(admitted.result) as RuntimeResult);
    const joined = this.inFlight.get(operationId);
    if (joined !== undefined) return joined;
    const execution = (async (): Promise<RuntimeResult> => {
      const durable = await this.productionControls.inspect(operationId, payloadDigest);
      if (durable.disposition !== "not_admitted") {
        const restored = durable as RuntimeResult;
        this.operations.commit(operationId, restored.disposition, restored);
        return restored;
      }
      this.operations.beginExecution(operationId);
      const result = await this.liveExecutor(method, intake, activationId, leaseEpoch);
      this.operations.commit(operationId, result.disposition, result);
      return result;
    })().finally(() => {
      this.inFlight.delete(operationId);
    });
    this.inFlight.set(operationId, execution);
    return execution;
  }
}
