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
- The shared File Agent loads bounded safe text context for evidence paths,
  ranks source-owner candidates, and separates focused test/document companion
  paths from runtime owner paths.
- The modifying PM chooses one lifecycle action from the File Agent plan:
  create a programmer task, request information, repair a child handoff,
  complete, or block.
- The supervisor validates the PM handoff before programmer dispatch. The first
  programmer task must include a source-owner path when one is available, and
  target paths must stay inside the bounded File Agent context.
- When the top-level supervisor supplies `repair_feedback`, `code_modifying`
  must return a corrected complete artifact list that addresses the validation
  errors rather than repeating the failed artifact shape.
- `execution_verification` repair feedback is accepted only as structured,
  bounded execution evidence from `code_verifying`; the PM/programmer may use
  failed paths, failure summaries, required source-owner paths, and protected
  verification paths, not raw stdout, stderr, command lines, or absolute roots.
- The modifying programmer returns structured operations:
  `replace`, `insert_before`, `insert_after`, or `replace_file_small`.
- Raw unified diffs and command-output-based repair feedback are rejected at
  the contract boundary.
- `code_patching` converts selected structured operations into unified diffs
  and materializes review-only validation packages.

Public callers should use `propose_code_change(...)`; direct role tests may use
`code_modifying.run(...)` or the model normalizers for focused contract checks.
