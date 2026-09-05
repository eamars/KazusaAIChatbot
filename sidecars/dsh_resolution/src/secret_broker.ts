import { CredentialProvider, type CredentialKey, type CredentialRecord, type CredentialRecordEntry, type CredentialRef, type CredentialInfo, type CredentialRecordInfo, type ResolvedCredential } from "@deepseek-ai/dsh-credentials";
import type { Context } from "@deepseek-ai/cordis";

const PROTECTED_NAMES = [
  "AGENTIC_RESOLVER_LLM_API_KEY",
  "KAZUSA_DSH_RPC_TOKEN",
  "KAZUSA_DSH_BRAIN_SHARED_SECRET",
  "KAZUSA_DSH_TOOL_GATEWAY_SECRET",
  "KAZUSA_DSH_CAPABILITY_TOKEN",
  "DEEPSEEK_API_KEY",
] as const;

export class WebProviderCredentialError extends Error {
  readonly code = "WEB_PROVIDER_CREDENTIAL_MISSING";
}

export interface SecretBroker {
  /** Resolve one named credential inside host-owned adapter/provider code. */
  resolveHostCredential(name: string): string | undefined;
  nativeEnvironment(): Record<string, string>;
  resolveWebCredential(
    name: string,
    credentials?: Record<string, string | undefined>,
    launchEnvironment?: Record<string, string | undefined>,
  ): string;
}

export interface SecretBrokerPluginConfig {
  hostOnly: boolean;
}

export function createSecretBroker(options: {
  hostSecrets: Record<string, string>;
  nativeEnvironment: Record<string, string>;
}): SecretBroker {
  const hostSecrets = Object.freeze({ ...options.hostSecrets });
  const nativeEnvironment = stripProtected({ ...options.nativeEnvironment });
  return {
    resolveHostCredential(name) {
      return hostSecrets[name];
    },
    nativeEnvironment() {
      return { ...nativeEnvironment };
    },
    resolveWebCredential(name, credentials = {}, launchEnvironment = {}) {
      const value = credentials[name] ?? launchEnvironment[name] ?? hostSecrets[name];
      if (typeof value !== "string" || value.length === 0) {
        throw new WebProviderCredentialError(`${name} is unavailable to the native web provider`);
      }
      return value;
    },
  };
}

function stripProtected(environment: Record<string, string>): Record<string, string> {
  for (const name of PROTECTED_NAMES) delete environment[name];
  return environment;
}

export { PROTECTED_NAMES };

/**
 * Host-owned implementation of DSH's credential reference service.
 *
 * The composition carries only references such as `AGENTIC_RESOLVER_LLM_API_KEY`.
 * Values are held by the host broker supplied during boot and are resolved per
 * request by the installed pi-ai and native web providers. Record operations
 * deliberately remain unavailable: this sidecar has no user-managed credential
 * store and must never turn an in-memory secret into a durable DSH record.
 */
export class HostCredentialProvider extends CredentialProvider {
  static inject = ["dshHostSecrets"];

  private readonly broker: SecretBroker;

  constructor(ctx: Context, _config: SecretBrokerPluginConfig) {
    super(ctx);
    const broker = ctx.get("dshHostSecrets") as SecretBroker | undefined;
    if (broker === undefined) throw new Error("host credential broker is unavailable");
    this.broker = broker;
  }

  async resolve(ref: CredentialRef): Promise<ResolvedCredential | undefined> {
    const value = this.broker.resolveHostCredential(String(ref));
    return value === undefined || value.length === 0
      ? undefined
      : { value, source: "host" };
  }

  async describe(ref: CredentialRef): Promise<CredentialInfo> {
    const configured = (await this.resolve(ref)) !== undefined;
    return { configured, ...(configured ? { source: "host" } : {}), writable: false };
  }

  async set(_ref: CredentialRef, _value: string): Promise<void> {
    throw new Error("HOST_CREDENTIAL_READ_ONLY");
  }

  async unset(_ref: CredentialRef): Promise<void> {
    throw new Error("HOST_CREDENTIAL_READ_ONLY");
  }

  async readRecord(_key: CredentialKey): Promise<CredentialRecord | undefined> {
    return undefined;
  }

  async describeRecord(_key: CredentialKey): Promise<CredentialRecordInfo> {
    return { configured: false, writable: false };
  }

  async listRecords(): Promise<readonly CredentialRecordEntry[]> {
    return [];
  }

  async modifyRecord(
    _key: CredentialKey,
    _mutate: (current: CredentialRecord | undefined) => Promise<CredentialRecord | undefined>,
  ): Promise<CredentialRecord | undefined> {
    throw new Error("HOST_CREDENTIAL_READ_ONLY");
  }

  async deleteRecord(_key: CredentialKey): Promise<void> {
    throw new Error("HOST_CREDENTIAL_READ_ONLY");
  }
}

/** Host-only composition entry point used by the installed DSH credential seam. */
export default class SecretBrokerPlugin extends HostCredentialProvider {}
