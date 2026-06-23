# Code Writing ICD

`code_writing` is the standalone patch-proposal subagent for the coding agent.
It produces limited proposed patches only. It never applies a patch to the
caller workspace and never runs target project commands.

## Public Entrypoint

```python
from kazusa_ai_chatbot.coding_agent.code_writing import run

result = await run(request)
```

`run(request: CodeWritingRequest) -> CodeWritingResult` is the internal Phase 2
subagent entrypoint used by the top-level `propose_code_change(...)` direct
interface. Existing repository requests enter this subagent after Phase 0
fetching. When source evidence is needed, the subagent returns `need_reading`
and the top-level supervisor runs the Phase 1 public reading contract before
resuming writing.

## Request

`CodeWritingRequest` contains:

- `question`: user-visible writing request.
- `mode_hint`: `edit_existing_repository` or `create_new_project`.
- `repository`: Phase 0 repository contract for existing-source work, or
  `None` for new-project work.
- `source_scope`: Phase 0 source scope for existing-source work.
- `reading_result`: Phase 1 reading result after the supervisor has resolved a
  `need_reading` outcome.
- `external_evidence`: limited public evidence summaries after the supervisor
  has resolved a `need_external_evidence` outcome.
- `workspace_root`: required caller-configured storage root.
- `session_id`: optional stable public session id.
- `preferred_language`, `max_answer_chars`, `max_artifact_chars`: optional
  output controls.

Missing `workspace_root` fails closed. The subagent does not infer storage
paths from user text and does not fall back to a process temp directory.

## Response

`CodeWritingResult` contains:

- `status`: `succeeded`, `failed`, `needs_user_input`, or `rejected`.
- `mode`: selected writing mode.
- `answer_text`: public explanation of the proposed patch.
- `patch_artifacts`: limited unified diff proposals.
- `created_files` and `changed_files`: public file summaries.
- `external_evidence`: limited summaries returned from PM-requested public
  evidence lookups.
- `validation`: deterministic patch validation result.
- `session`: public-safe writing session handle.
- `limitations`: missing evidence, validation failures, or unsupported scope.
- `trace_summary`: compact stage summary for review.
- `trace`: optional limited diagnostic output for live LLM review documents.

Public responses must not expose local roots, storage roots, cache keys, raw
command output, source dumps, secret-like files, repository internals, or
credentials.

## Workflow

```text
CodeWritingRequest
-> writing supervisor
-> managed session preparation
-> writing PM on CODING_AGENT_PM_LLM
-> optional need_reading or need_external_evidence returned to top-level supervisor
-> Source Ownership PM selects existing-source owners from bounded evidence
-> file agent validates file mechanics and builds file/module contracts
-> file-plan evaluator accepts the resolved contracts
-> one File PM per accepted file/module contract on CODING_AGENT_PM_LLM
-> module-contract evaluator accepts each File PM programmer contract
-> limited programmer workers on CODING_AGENT_PROGRAMMER_LLM, one module contract each
-> patcher on CODING_AGENT_PROGRAMMER_LLM builds PM-selected output
-> sandbox patch validation
-> optional validation repair loop
-> synthesis on CODING_AGENT_PM_LLM
-> CodeWritingResult
```

The top-level writing PM owns semantic decomposition, sufficiency, writing
mode, external evidence need, file purposes, cross-file import needs, File PM
review reconciliation, and final artifact selection. The Source Ownership PM
chooses existing-source owner paths from bounded reading evidence and candidate
paths. The shared file agent owns file mechanics after ownership is known:
path safety, new-file reservation, base revision checks, current file context
packaging, and path maps. Each File PM owns one module-level programmer
contract for an accepted file/module assignment, including exact imports,
required symbols, bounded current file context, and required behavior.
Programmer workers own scoped implementation content for one accepted module
contract and do not see peer programmer work. The patcher owns edit mechanics
and converts PM-selected programmer content into unified diffs or new-project
file trees. Deterministic code owns workspace preparation, patch parsing,
sandbox validation, caps, and public sanitization.

## Mutation Boundary

Patch validation copies the base into a managed sandbox and checks the proposed
diff there. The subagent does not mutate the fetched repository, the caller
workspace, or the Kazusa source tree as part of a request.

Out of scope:

- applying patches to real checkouts;
- running target project tests, package commands, build commands, or shell
  verification;
- dependency installation;
- runtime integration with background work or adapter delivery.

## External Evidence

External evidence is supervisor-managed. The writing PM may request public
evidence by returning `need_external_evidence` with limited evidence tasks.
The top-level coding supervisor resolves those tasks through the public
`WebAgent3().run(...)` helper, then resumes `code_writing.run(...)` with the
limited evidence summaries. Evidence is supporting context for the PM and
programmer stages; it does not override local source evidence or directly
become final patch instructions.

## Session Storage

`workspace.py` stores session metadata under the caller-provided coding
workspace. Session ids are sanitized public handles. Session metadata records
the base identity and marks a previous session invalidated when the repository
or new-project base identity changes.
