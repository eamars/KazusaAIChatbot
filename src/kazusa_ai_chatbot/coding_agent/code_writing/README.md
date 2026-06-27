# Code Writing ICD

`code_writing` is the standalone new-artifact writing subagent for the coding
agent. It creates proposed files, scripts, docs, tests, config, or small
projects from a bounded source-free request. It never applies a patch to the
caller workspace and never runs target project commands.

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
- `external_evidence`: limited public evidence summaries after the supervisor
  resolves a `need_external_evidence` outcome.
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
- `validation`: deterministic patch validation result.
- `alignment`: optional LLM-owned artifact/request alignment result.
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
-> Acceptance owner preserves user-visible requirements
-> Writing PM on CODING_AGENT_PM_LLM
-> optional need_external_evidence returned to top-level supervisor
-> File Agent reserves safe new artifact paths
-> one Writing programmer on CODING_AGENT_PROGRAMMER_LLM per artifact contract
-> patching boundary materializes generated artifacts as new-file diffs
-> structural validation checks patch artifact shape and sandbox apply
-> Alignment owner compares generated artifacts against preserved requirements
-> synthesis on CODING_AGENT_PM_LLM
-> CodeWritingResult
```

The top-level supervisor owns cross-domain interleaving. The Writing PM owns
the requested new-artifact feature picture and artifact decomposition.
Acceptance and alignment owners preserve and check user-visible requirements.
File Agent owns path mechanics. Each Writing programmer receives one artifact
contract and returns one fenced artifact body. The patching boundary owns
file-tree or unified-diff materialization. Deterministic code owns structural
validation, caps, path safety, storage boundaries, and public sanitization.

## Mutation Boundary

Patch validation copies into a managed sandbox and checks the proposed diff
there. The subagent does not mutate fetched repositories, caller workspaces, or
the Kazusa source tree as part of a request.

Out of scope:

- semantic edits to existing source files;
- applying patches to real checkouts;
- running target project tests, package commands, build commands, or shell
  verification;
- dependency installation;
- runtime integration with background work or adapter delivery.

## External Evidence

External evidence is supervisor-managed. The Writing PM may request public
evidence by returning `need_external_evidence` with limited evidence tasks.
The top-level coding supervisor resolves those tasks through the public
external-evidence helper, then resumes `code_writing.run(...)` with limited
evidence summaries.

## Session Storage

`workspace.py` stores session metadata under the caller-provided coding
workspace. Session ids are sanitized public handles. Session metadata records
the new-artifact base identity and marks a previous session invalidated when
the base identity changes.
