export interface AssistantToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface AutonomousRuntimePolicy {
  terminalTool: "submit_resolution";
  mode: "autonomous";
}

export function autonomousRuntimePolicy(): AutonomousRuntimePolicy {
  return { mode: "autonomous", terminalTool: "submit_resolution" };
}

export function evaluateAssistantToolStep(
  calls: readonly AssistantToolCall[],
): { accepted: boolean; terminal: boolean; result?: Record<string, unknown> } {
  if (calls.length === 1 && calls[0]?.name === "submit_resolution") {
    return { accepted: true, terminal: true };
  }
  if (calls.some((call) => call.name === "submit_resolution")) {
    return {
      accepted: false,
      terminal: false,
      result: { kind: "runtime_fault", fault: { code: "TERMINAL_TOOL_MUST_BE_SOLE_CALL" } },
    };
  }
  return { accepted: true, terminal: false };
}
