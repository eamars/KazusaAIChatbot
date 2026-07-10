# Code Verifying ICD

`code_verifying` owns direct trusted verify-and-repair orchestration for the
coding agent. It composes existing proposal, patch-apply, and execution
boundaries without changing their independent contracts.

Execution plans bind one run id, source identity, proposal revision, and
ordered patch-artifact digest. Deterministic planning derives exact Python
compile paths and safe focused pytest selectors. Missing external dependencies
become typed environment blockers before repair work is attempted.

## Ownership

- Resolves the source through `code_fetching`.
- Accepts a reviewed patch proposal or asks the existing proposal path for the
  initial patch.
- Applies each attempt through `apply_approved_patch(...)`.
- Runs each structured verification spec through `execute_code_check(...)`.
- Converts failed or timed-out execution results into bounded structured
  repair feedback.
- Re-enters the existing source modification proposal path for capped repairs.
- Returns a public-safe attempt ledger.

## Boundaries

- Requires structured approval before any managed apply workspace is created.
- Applies only into managed apply copies.
- Executes only `python_compileall` and `pytest` specs through the execution
  boundary.
- Does not mutate the original source checkout.
- Does not run package installers, arbitrary shell commands, deployment tools,
  database commands, adapter sends, or repository pushes.
- Does not expose absolute local roots, raw command lines, or full command
  output in repair feedback.
- Does not give background-work tasks auto-apply, auto-execute, or auto-repair
  behavior.

## Public Entrypoints

```python
from kazusa_ai_chatbot.coding_agent import verify_and_repair_code_change
from kazusa_ai_chatbot.coding_agent.code_verifying import (
    verify_and_repair_code_change,
)
```

## Request

`CodingVerifyRepairRequest` accepts the source fields used by the direct
coding-agent request plus:

- `approval`
- `execution_specs`
- `repair_attempt_limit`
- `max_repair_feedback_chars`
- `initial_patch_artifacts`
- `expected_source_identity`

`initial_patch_artifacts` are optional and skip only initial proposal
generation. They still go through patch review validation, source identity
matching, managed-copy apply, execution, and repair.

## Response

`CodingVerifyRepairResponse` contains:

- `status`
- `answer_text`
- `repository`
- `source_scope`
- `attempts`
- `final_patch_artifacts`
- `final_changed_files`
- `final_apply`
- `final_execution`
- `limitations`
- `trace_summary`
