# Code Writing ICD

`code_writing` is the standalone new-artifact writing subagent for the coding
agent. It creates proposed files, scripts, docs, tests, config, or small
projects from a bounded source-free request. It never applies a patch to the
caller workspace, never runs target project commands, and does not execute
generated code or generated tests in Phase 2.

## Public Entrypoint

```python
from kazusa_ai_chatbot.coding_agent.code_writing import run

result = await run(request)
```

`run(request: CodeWritingRequest) -> CodeWritingResult` is the internal Phase 2
subagent entrypoint used by the top-level `propose_code_change(...)` direct
interface.

## Request

`CodeWritingRequest` contains:

- `question`: user-visible request for new artifacts.
- `mode_hint`: must be `create_new_project` for the Phase 2 writing path.
- `external_evidence`: limited public evidence summaries after the top-level
  supervisor resolves a PM `request_information` outcome.
- `workspace_root`: required caller-configured storage root.
- `session_id`: optional stable public session id.
- `preferred_language`, `max_answer_chars`, `max_artifact_chars`: optional
  output controls.

Missing `workspace_root` fails closed. The subagent does not infer storage
paths from user text and does not fall back to a process temp directory.
Requests that require semantic edits to existing source files are rejected by
this writing stage and must be handled by the separate code-modifying
capability.

## Response

`CodeWritingResult` contains:

- `status`: `succeeded`, `failed`, `needs_user_input`, `rejected`, or
  `need_external_evidence`.
- `mode`: selected writing mode.
- `answer_text`: public explanation of the proposed artifacts.
- `patch_artifacts`: limited unified diff proposals for new files.
- `created_files` and `changed_files`: public file summaries. In this stage,
  created files are expected and changed files are diagnostics only.
- `external_evidence_requests` and `external_evidence`: supervisor-mediated
  public evidence handoff.
- `validation`: review-package materialization result. It proves the proposed
  files were materialized for inspection; it is not generated-code validation.
- `alignment`: reserved for later-phase semantic artifact review.
- `session`: public-safe writing session handle.
- `limitations`: missing evidence, materialization issues, or unsupported
  scope.
- `trace_summary`: compact stage summary for review.
- `trace`: optional limited diagnostic output for live LLM review documents.

Public responses must not expose local roots, storage roots, cache keys, raw
command output, source dumps, secret-like files, repository internals, or
credentials.

## PM Lifecycle Workflow

```text
CodeWritingRequest
-> writing supervisor
-> managed session preparation
-> Acceptance owner preserves user-visible requirements
-> PM lifecycle decision on CODING_AGENT_PM_LLM
-> optional request_information returned to top-level supervisor
-> optional child PM lifecycle for smaller work item
-> optional one programmer task for one new artifact
-> File Agent reserves safe new artifact path
-> Writing programmer on CODING_AGENT_PROGRAMMER_LLM
-> patching boundary materializes generated artifact content as new-file diffs
-> review-package materialization copies proposed files into managed storage
-> synthesis on CODING_AGENT_PM_LLM
-> CodeWritingResult
```

The top-level supervisor owns cross-domain interleaving. The writing PM owns
only the semantic lifecycle of its direct children. In Phase 2 it may request
information, create one child PM, create one programmer task, complete with a
report, or block with a reason. Repair remains reserved for later phases.
Acceptance preserves user-visible requirements. File Agent owns path mechanics.
Each Writing programmer receives one PM-approved artifact contract and returns
one fenced artifact body. The patching boundary owns file-tree or unified-diff
materialization. Deterministic code owns caps, path safety, storage boundaries,
review-package materialization, and public sanitization.

## Mutation Boundary

Review-package materialization copies proposed artifacts into managed storage
for inspection. The subagent does not mutate fetched repositories, caller
workspaces, or the Kazusa source tree as part of a request.

Out of scope:

- semantic edits to existing source files;
- applying patches to real checkouts;
- running generated code, generated tests, target project tests, package
  commands, build commands, or shell verification;
- validation feedback loops or repair loops;
- dependency installation;
- runtime integration with background work or adapter delivery.

## External Evidence

Information requests are supervisor-managed. The writing PM may request
workspace, generated-artifact, existing-source, provided-evidence, or public
external facts through `request_information`. The top-level coding supervisor
chooses the correct workflow, records the result in its ledger, then resumes
`code_writing.run(...)` with compact evidence summaries.

## Session Storage

`workspace.py` stores session metadata under the caller-provided coding
workspace. Session ids are sanitized public handles. Session metadata records
the new-artifact base identity and marks a previous session invalidated when
the base identity changes.
