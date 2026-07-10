# Coding Run ICD

`coding_run` is the durable run supervisor for the coding-agent direct API. It
does not add a new LLM role. It persists run state, validates deterministic
transitions, and composes the existing reading, proposal, apply, execution,
and verify/repair primitives.

Each proposal stores a monotonic revision, ordered artifact digest, and a
proposal-bound execution plan. With `CODING_AGENT_PREFLIGHT_EXECUTION` enabled,
the run records managed-copy preflight evidence for the current process-only
execution backend. The flag remains disabled by default.

Public APIs:

```python
from kazusa_ai_chatbot.coding_agent import continue_coding_run
from kazusa_ai_chatbot.coding_agent import get_coding_run
from kazusa_ai_chatbot.coding_agent import start_coding_run
```

Ledgers are stored under the caller-supplied workspace:

```text
<workspace_root>/coding_runs/<run_id>/run.json
<workspace_root>/coding_runs/<run_id>/events.jsonl
```

`start_coding_run(...)` requires an explicit `objective_type`:

- `read_only`: call `answer_code_question(...)`, persist evidence, answer,
  events, and a terminal public status.
- `propose_patch`: call `propose_code_change(...)`, persist review-only patch
  artifacts, and stop at `awaiting_approval`.
- `verify_repair`: call `verify_and_repair_code_change(...)` with structured
  approval, execution specs, and optional initial patch artifacts.

`continue_coding_run(...)` requires an explicit `action`:

- `approve_and_verify`: allowed only from `awaiting_approval`; calls
  `verify_and_repair_code_change(...)` with the stored proposal artifacts and
  supplied execution specs.
- `revise_proposal`, `summarize`, and `status`: follow the stored run's
  canonical affordances and do not infer a continuation from user text.
- `respond_to_blocker`: allowed only for the one open blocker with a resumable
  target. It either revises the proposal with the user answer or retries the
  stored verification plan with zero repair attempts.
- `cancel`: allowed from non-terminal states; records cancellation without
  apply or execution side effects.

`get_coding_run(...)` reloads the persisted ledger and event stream by run id.

The public response contains status, run id, goal, objective type, answer,
public repository and scope metadata, evidence, patch artifacts, changed
files, apply/execution/repair attempt summaries, blockers, events,
limitations, and trace summary. Public projection sanitizes local roots,
workspace roots, cache keys, environment filenames, git internals, secret-like
values, raw command output, and full source dumps.

`allowed_next_actions(...)` is the single state-machine owner for every
continuation. The worker, accepted-task context, and L2d materializer consume
its public result and revalidate the stored run before work. Each mutation
holds sorted kernel locks for `run:<run_id>` and its normalized source identity;
contention returns a non-mutating `operation_outcome="busy"` response.

Ownership boundaries:

- `coding_run` owns lifecycle state, JSON ledger persistence, event append,
  public projection, continuation validation, and specialist invocation order.
- `code_fetching`, `code_reading`, `code_writing`, `code_modifying`,
  `code_patching`, `code_executing`, and `code_verifying` keep their existing
  source, planning, patch, apply, execution, and repair responsibilities.
- Original source checkouts are never mutated by `coding_run`; apply and
  execution effects stay inside managed apply workspaces created by the
  existing trusted primitives.
- Background accepted coding tasks remain review-only and do not invoke this
  trusted continuation boundary.
