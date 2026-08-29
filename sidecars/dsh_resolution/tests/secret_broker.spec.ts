import { describe, expect, it } from "vitest";

describe("secret isolation", () => {
  it("native shell cannot read host credentials tokens or bridge secrets", async () => {
    const secrets = await import("../src/secret_broker.js");
    const broker = secrets.createSecretBroker({
      hostSecrets: {
        AGENTIC_RESOLVER_LLM_API_KEY: "model-secret-sentinel",
        KAZUSA_DSH_RPC_TOKEN: "rpc-secret-sentinel",
        KAZUSA_DSH_BRAIN_SHARED_SECRET: "brain-secret-sentinel",
        KAZUSA_DSH_TOOL_GATEWAY_SECRET: "gateway-secret-sentinel",
        DEEPSEEK_API_KEY: "deepseek-secret-sentinel",
      },
      nativeEnvironment: { DSH_HOME: "C:/runtime/dsh" },
    });

    expect(broker.resolveHostCredential("AGENTIC_RESOLVER_LLM_API_KEY"))
      .toBe("model-secret-sentinel");
    expect(broker.resolveHostCredential("KAZUSA_DSH_RPC_TOKEN"))
      .toBe("rpc-secret-sentinel");

    const native = broker.nativeEnvironment();
    for (const name of [
      "AGENTIC_RESOLVER_LLM_API_KEY",
      "KAZUSA_DSH_RPC_TOKEN",
      "KAZUSA_DSH_BRAIN_SHARED_SECRET",
      "KAZUSA_DSH_TOOL_GATEWAY_SECRET",
      "KAZUSA_DSH_CAPABILITY_TOKEN",
      "DEEPSEEK_API_KEY",
    ]) {
      expect(native).not.toHaveProperty(name);
    }
    expect(native).toMatchObject({ DSH_HOME: "C:/runtime/dsh" });
    expect(JSON.stringify(native)).not.toContain("secret-sentinel");

    const observed = await broker.runNativeProbe();
    expect(observed).toEqual({
      AGENTIC_RESOLVER_LLM_API_KEY: null,
      KAZUSA_DSH_RPC_TOKEN: null,
      KAZUSA_DSH_BRAIN_SHARED_SECRET: null,
      KAZUSA_DSH_TOOL_GATEWAY_SECRET: null,
      KAZUSA_DSH_CAPABILITY_TOKEN: null,
      DEEPSEEK_API_KEY: null,
    });
  });
});
