# Code Modifying ICD

`code_modifying` owns existing-source modification proposals inside the coding
agent. It never applies patches, runs commands, installs packages, or mutates
the target checkout.

Inputs:

- user modification request
- resolved repository contract from `code_fetching`
- read-only source evidence from `code_reading`
- caller-managed `workspace_root` for downstream review artifacts

Runtime ownership:

- The top-level coding-agent supervisor resolves source and gathers evidence
  before calling `code_modifying.run(...)`.
- `code_modifying` loads bounded safe text context for evidence paths only.
- The modifying programmer returns structured operations:
  `replace`, `insert_before`, `insert_after`, or `replace_file_small`.
- Raw unified diffs and command-output-based repair feedback are rejected at
  the contract boundary.
- `code_patching` converts selected structured operations into unified diffs
  and materializes review-only validation packages.

Public callers should use `propose_code_change(...)`; direct role tests may use
`code_modifying.run(...)` or the model normalizers for focused contract checks.
