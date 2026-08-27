export type OperationDisposition = "not_admitted" | "admitted_active" | "checkpointed" | "terminal" | "canceled" | "faulted" | "unknown";

export interface OperationRecord {
  operation_id: string;
  operation_payload_digest: string;
  method: string;
  disposition: Exclude<OperationDisposition, "not_admitted" | "unknown">;
  result?: Record<string, unknown>;
  admission_count: number;
  execution_count: number;
  message_source_id: string;
}

export class OperationReuseFault extends Error {
  readonly code = "OPERATION_ID_REUSE_MISMATCH";
}

export class OperationRegistry {
  private readonly records = new Map<string, OperationRecord>();

  admit(operationId: string, payloadDigest: string, method: string): OperationRecord {
    const existing = this.records.get(operationId);
    if (existing !== undefined) {
      if (
        existing.operation_payload_digest !== payloadDigest
        || existing.method !== method
      ) {
        throw new OperationReuseFault("OPERATION_ID_REUSE_MISMATCH");
      }
      return existing;
    }
    const record: OperationRecord = {
      operation_id: operationId,
      operation_payload_digest: payloadDigest,
      method,
      disposition: "admitted_active",
      admission_count: 1,
      execution_count: 0,
      message_source_id: `kazusa-operation:${operationId}`,
    };
    this.records.set(operationId, record);
    return record;
  }

  beginExecution(operationId: string): OperationRecord {
    const record = this.required(operationId);
    record.execution_count += 1;
    return record;
  }

  commit(operationId: string, disposition: OperationRecord["disposition"], result: Record<string, unknown>): OperationRecord {
    const record = this.required(operationId);
    record.disposition = disposition;
    record.result = structuredClone(result);
    return record;
  }

  inspect(operationId: string, payloadDigest?: string): { disposition: OperationDisposition; result?: Record<string, unknown> } {
    const record = this.records.get(operationId);
    if (record === undefined) return { disposition: "not_admitted" };
    if (
      payloadDigest !== undefined
      && record.operation_payload_digest !== payloadDigest
    ) {
      throw new OperationReuseFault("OPERATION_ID_REUSE_MISMATCH");
    }
    if (record.result === undefined) return { disposition: record.disposition };
    return { disposition: record.disposition, result: structuredClone(record.result) };
  }

  admissionCount(operationId: string): number {
    return this.records.get(operationId)?.admission_count ?? 0;
  }

  executionCount(operationId: string): number {
    return this.records.get(operationId)?.execution_count ?? 0;
  }

  snapshot(): OperationRecord[] {
    return [...this.records.values()].map((record) => structuredClone(record));
  }

  restore(records: readonly OperationRecord[]): void {
    for (const record of records) this.records.set(record.operation_id, structuredClone(record));
  }

  private required(operationId: string): OperationRecord {
    const record = this.records.get(operationId);
    if (record === undefined) throw new Error("operation is not admitted");
    return record;
  }
}
