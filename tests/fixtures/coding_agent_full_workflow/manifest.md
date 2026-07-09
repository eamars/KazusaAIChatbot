# coding agent full workflow fixtures

This fixture set backs the restored full workflow live LLM gates. Each fixture
is intentionally small, source-readable, and free of secrets.

Committed fixture path: `tests/fixtures/coding_agent_full_workflow/`.

| Gate | Fixture | Contract |
|---|---|---|
| 01 | `gate_01_cli_command_discovery` | Read-only question: explain where CLI command discovery lives. No patch, apply, execution, or repair attempts are expected. |
| 02 | `gate_02_csv_normalizer` | Source-free artifact brief: create a CSV normalizer CLI, then revise it for dry-run mode and deterministic output, then summarize review files. |
| 03 | `gate_03_counter_cli_json` | Existing-source proposal: add JSON output to the runtime counter CLI, then revise to keep tests unchanged, then summarize exact changed files. |
| 04 | `gate_04_slug_normalization` | Existing-source verify/repair: fix slug normalization, approve focused pytest, allow source repair, and report attempts without changing tests. |
| 05 | `gate_05_release_feed_cache_cli` | Hard multi-file workflow: fix cache timeout and CLI flag behavior, approve focused tests, summarize attempts/final files, conditionally cancel, and report final status. |

Anti-cheat expectations:

- Live gates must enter through L2d action selection and background-work queue
  execution.
- Tests must not call durable coding-run APIs directly.
- Fixture tests are protected unless the gate text explicitly requests test
  changes.
- Fixture source must not be weakened after seeing a live model failure.
