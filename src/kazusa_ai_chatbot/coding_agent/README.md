# Coding Agent ICD

The `coding_agent` package contains standalone code-task modules that can be
called directly by tests and future background workers.

Current implemented surfaces:

```python
from kazusa_ai_chatbot.coding_agent import answer_code_question
from kazusa_ai_chatbot.coding_agent.code_fetching import run
from kazusa_ai_chatbot.coding_agent.code_reading import run
```

`code_fetching.run(...)` resolves a supported code source into a local source
contract. It does not read files to answer questions, write patches, execute
project commands, or integrate with Kazusa service/background-worker runtime.

`answer_code_question(...)` is the Phase 1 direct interface. It calls Phase 0
fetching first, short-circuits non-success fetching results, then calls
`code_reading.run(...)` with the successful repository and source scope.
Responses use a public-safe repository summary and bounded repo-relative source
evidence.

Implemented subagents:

- `code_fetching`: resolves public GitHub and explicit local-checkout sources.
- `code_reading`: reads safe text files inside the resolved source scope and
  synthesizes evidence-backed answers.

Deferred subagents:

- `code_writing`
- `code_executing`

Managed checkouts live under the caller-supplied coding workspace root. Direct
standalone use may fall back to an OS temp workspace, but future worker
integration must pass an explicit configured workspace.

## Direct Request

`CodingAgentRequest` accepts every public Phase 0 source field:

- `question`
- `source_url`
- `repo_url`
- `repo_hint`
- `local_root_hint`
- `local_path_hint`
- `requested_ref`
- `source_scope_hint`
- `workspace_root`

It also accepts Phase 1 reading hints:

- `preferred_language`
- `max_answer_chars`

The supervisor passes all Phase 0 source fields through unchanged to
`code_fetching.run(...)`.

## Direct Response

`CodingAgentResponse` contains:

- `status`
- `answer_text`
- `repository`
- `source_scope`
- `evidence`
- `limitations`
- `trace_summary`

`repository` is a `CodingAgentRepositorySummary` with public metadata only:

- `provider`
- `owner`
- `repo`
- `source_url`
- `requested_ref`
- `resolved_ref`
- `current_commit`
- `default_branch`
- `storage_kind`
- `managed_checkout`
- `dirty_state`

The direct response and future worker metadata must not include `local_root`,
`workspace_root`, `cache_key`, raw command output, full source files, `.env`
content, secret-like file content, `.git` internals, or binary asset content.

## Phase 2 Worker Handoff

Future Kazusa background-work integration should register:

- Worker name: `coding_agent`
- Worker description: answers read-only source-code questions from bounded
  repository evidence; Phase 1 does not write code or execute project commands.

Recommended `BackgroundWorkResult` mapping:

- `worker`: `coding_agent`
- `status`: `CodingAgentResponse.status`
- `summary`: `CodingAgentResponse.answer_text`
- `artifacts`: bounded evidence rows and public repository summary
- `limitations`: `CodingAgentResponse.limitations`
- `trace_summary`: `CodingAgentResponse.trace_summary`

The Phase 2 worker must supply the configured coding workspace root. It must not
parse workspace paths from user text, fall back to worker-local temp paths, or
send adapter-visible text directly.
