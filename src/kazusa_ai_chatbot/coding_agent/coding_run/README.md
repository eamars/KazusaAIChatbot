# Coding Run ICD

`coding_run` is the durable run supervisor for the coding-agent direct API. It
does not add a new LLM role. It persists run state, validates deterministic
transitions, and composes the existing reading, proposal, apply, execution,
and verify/repair primitives.

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
- `cancel`: allowed from non-terminal states; records cancellation without
  apply or execution side effects.

`get_coding_run(...)` reloads the persisted ledger and event stream by run id.

The public response contains status, run id, goal, objective type, answer,
public repository and scope metadata, evidence, patch artifacts, changed
files, apply/execution/repair attempt summaries, blockers, events,
limitations, and trace summary. Public projection sanitizes local roots,
workspace roots, cache keys, environment filenames, git internals, secret-like
values, raw command output, and full source dumps.

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
