# Code Executing ICD

`code_executing` is the deterministic execution boundary for approved coding
agent apply workspaces. It runs a small allowlist of Python verification tools
inside the Phase 5 managed apply copy and returns bounded public-safe output.

## Ownership

- Resolves the execution root from `workspace_root`, `apply_package_id`, and
  the Phase 5 `apply_workspace_ref`.
- Accepts only structured execution specs for `python_compileall` and
  `pytest`.
- Builds argv lists directly and runs them with a managed apply source root as
  the working directory.
- Enforces target containment, unsafe-path rejection, timeout caps, output
  caps, and public output redaction.
- Reports target-project success, failure, timeout, or rejection as structured
  execution metadata.

## Boundaries

- It executes only inside
  `<workspace_root>/patch_apply/<apply_package_id>/source`.
- It does not accept an absolute apply path from request data.
- It does not execute against original source checkouts, managed clone roots,
  review package directories, inline bundles, or the Kazusa repository.
- It does not select commands from user prose, LLM explanations, proposal text,
  accepted-task text, or background-worker metadata.
- It does not run arbitrary shell text, package installers, network tools,
  deployment commands, database commands, adapter sends, or repository push
  operations.
- It does not repair failed patches or feed raw execution output to an LLM.
- It does not persist execution results to MongoDB.

## Public Entrypoints

```python
from kazusa_ai_chatbot.coding_agent import execute_code_check
from kazusa_ai_chatbot.coding_agent.code_executing import run
```

`execute_code_check(...)` is the top-level direct trusted API. `run(...)` is
the submodule entrypoint.

## Request

`CodeExecutionRequest` accepts:

- `workspace_root`
- `apply_package_id`
- `apply_workspace_ref`
- `execution`
- `max_stdout_chars`
- `max_stderr_chars`

`execution.tool` must be `python_compileall` or `pytest`. Compile execution
uses `execution.paths`; pytest execution uses `execution.pytest_selectors`.
Targets must be relative to the managed apply source root and must already
exist.

## Response

`CodeExecutionResponse` contains:

- `status`: `succeeded`, `failed`, `rejected`, or `timed_out`
- `tool`
- `exit_code`
- `timed_out`
- `duration_ms`
- `stdout_excerpt`
- `stderr_excerpt`
- `output_truncated`
- `executed_paths`
- `limitations`
- `trace_summary`

Target-project failures are represented as `status="failed"` with an exit code.
They are not executor crashes. Public excerpts redact managed absolute paths
and are capped before being returned.

`code_verifying` may convert failed or timed-out execution responses into
structured `execution_verification` repair feedback. That feedback remains
bounded and redacted; raw command output and command lines stay outside LLM
repair prompts.

The private action loop may supply `candidate_execution_identity` instead of
an apply-workspace reference after a verification failure. The executor binds
that request to the run id, current candidate revision, managed-tree digest,
pinned base snapshot, execution policy, and exact structured spec. It copies
the current candidate into a short digest-named managed workspace and reuses a
terminal result only when the complete identity matches.
Apply, candidate, and deterministic execution paths reject symlinked managed
roots or identity files before any command is selected or executed.
