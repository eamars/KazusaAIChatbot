# Control Console E2E Tests

These tests exercise the local control-console product surface through an
isolated server process and browser-capable harnesses.

Ordinary harness cases do not read the repository `.env`. They inject explicit
test-only settings into `ControlConsoleSettings` and launch the console with a
temporary state directory, deterministic operator token, isolated ports, and
optional test service registries.

An explicitly opted-in live-database case may set
`use_live_project_db=True`. Its child process loads project database settings
while preserving the caller's exact `MONGODB_DB_NAME` and test guards. Such a
case must require the dedicated database/run flags and clean only its declared
test-owned collections.

Each test writes concise JSON summaries under its pytest temporary artifact
directory. Development-plan execution records should summarize those files
instead of pasting raw logs or screenshots into the plan.
