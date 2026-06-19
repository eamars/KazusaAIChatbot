# Control Console E2E Tests

These tests exercise the local control-console product surface through an
isolated server process and browser-capable harnesses.

The harness must not read the repository `.env`. Tests inject explicit
test-only settings into `ControlConsoleSettings` and launch the console with a
temporary state directory, deterministic operator token, isolated ports, and
optional test service registries.

Each test writes concise JSON summaries under its pytest temporary artifact
directory. Development-plan execution records should summarize those files
instead of pasting raw logs or screenshots into the plan.
