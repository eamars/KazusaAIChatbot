# Coding Agent ICD

The `coding_agent` package contains standalone code-task modules that can be
called directly by tests and future background workers.

Current implemented surfaces:

```python
from kazusa_ai_chatbot.coding_agent import answer_code_question
from kazusa_ai_chatbot.coding_agent import propose_code_change
from kazusa_ai_chatbot.coding_agent.code_fetching import run as run_code_fetching
from kazusa_ai_chatbot.coding_agent.code_reading import run as run_code_reading
from kazusa_ai_chatbot.coding_agent.code_writing import run as run_code_writing
```

`code_fetching.run(...)` resolves a supported code source into a local source
contract. It does not read files to answer questions, write patches, execute
project commands, or integrate with Kazusa service/background-worker runtime.

`answer_code_question(...)` is the Phase 1 direct interface. It calls Phase 0
fetching first, short-circuits non-success fetching results, then calls
`code_reading.run(...)` with the successful repository and source scope.
Responses use a public-safe repository summary and bounded repo-relative source
evidence.

`propose_code_change(...)` is the Phase 2 direct interface. It requires an
explicit `workspace_root`, returns proposed patch artifacts only, and never
applies patches or runs target project commands. Existing-repository writing
uses Phase 0 fetching and Phase 1 reading before `code_writing.run(...)`.
Source-free requests use the managed new-project writing workspace.

Implemented subagents:

- `code_fetching`: resolves public GitHub and explicit local-checkout sources.
- `code_reading`: reads safe text files inside the resolved source scope and
  synthesizes evidence-backed answers.
- `code_writing`

Deferred subagents:

- `code_executing`

Managed checkouts and managed raw-file downloads live under the caller-supplied
coding workspace root. Writing requests require an explicit configured
workspace root so proposal storage, validation sandboxes, and session memory
remain inspectable.

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

## Direct Write Request

`CodingAgentWriteRequest` accepts the same public Phase 0 source fields as
`CodingAgentRequest` plus writing controls:

- `preferred_language`
- `max_answer_chars`
- `max_artifact_chars`
- `session_id`

`workspace_root` is required for writing. If source fields are present, the
request is handled as an existing-repository patch proposal. If no source
fields are present, the request is handled as a new-project proposal in
a managed writing workspace.

## Direct Response

`CodingAgentResponse` contains:

- `status`
- `answer_text`
- `repository`
- `source_scope`
- `evidence`
- `limitations`
- `trace_summary`

`CodingPatchProposalResponse` contains:

- `status`
- `mode`
- `answer_text`
- `repository`
- `source_scope`
- `evidence`
- `patch_artifacts`
- `created_files`
- `changed_files`
- `validation`
- `external_evidence`
- `session`
- `limitations`
- `trace_summary`
- optional `trace` for live LLM review artifacts

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

`storage_kind` is `existing_local_checkout`, `managed_clone`, or
`managed_download`. For `managed_download`, `current_commit` is a
`raw-sha256:<hash>` content identity rather than a Git commit.

The direct response and future worker metadata must not include `local_root`,
`workspace_root`, `cache_key`, raw command output, full source files, `.env`
content, secret-like file content, `.git` internals, or binary asset content.

## Phase 3 Worker Handoff

Future Kazusa background-work integration should register:

- Worker name: `coding_agent`
- Worker description: handles supported standalone coding-agent tasks from
  bounded repository evidence. The Phase 1 surface answers read-only
  source-code questions; the Phase 2 surface returns patch-proposal artifacts.
  Phase 3 must not add patch application or project command execution.

Recommended Phase 1 `BackgroundWorkResult` mapping:

- `worker`: `coding_agent`
- `status`: `CodingAgentResponse.status`
- `summary`: `CodingAgentResponse.answer_text`
- `artifacts`: bounded evidence rows and public repository summary
- `limitations`: `CodingAgentResponse.limitations`
- `trace_summary`: `CodingAgentResponse.trace_summary`

Recommended Phase 2 `BackgroundWorkResult` mapping:

- `worker`: `coding_agent`
- `status`: `CodingPatchProposalResponse.status`
- `summary`: `CodingPatchProposalResponse.answer_text`
- `artifacts`: bounded patch artifacts, created-file summaries, validation
  summary, evidence rows, and public repository summary
- `limitations`: `CodingPatchProposalResponse.limitations`
- `trace_summary`: `CodingPatchProposalResponse.trace_summary`

The Phase 3 worker must supply the configured coding workspace root. It must not
parse workspace paths from user text, fall back to worker-local temp paths, or
send adapter-visible text directly.
