# Code Reading Subagent ICD

`code_reading` is the Phase 1 read-only subagent. It answers questions from a
successful Phase 0 source contract and never fetches, clones, writes, executes
project commands, installs packages, or integrates with Kazusa runtime paths.

Public entrypoint:

```python
from kazusa_ai_chatbot.coding_agent.code_reading import run
```

`run(request: CodeReadingRequest) -> CodeReadingResult` consumes:

- `question`: the user-visible code-reading question.
- `repository`: the successful `CodeRepositoryRef` returned by Phase 0.
- `source_scope`: the successful `CodeSourceScope` returned by Phase 0.
- `preferred_language`: optional answer language hint.
- `max_answer_chars`: optional public answer cap.

The subagent uses the repository checkout only as an internal read source. It
returns bounded repo-relative evidence rows and must not expose absolute checkout
paths, workspace roots, cache keys, raw command output, full source files,
secret-like files, `.env` files, `.git` internals, or binary asset contents.

`CodeReadingResult` contains:

- `status`: `succeeded`, `failed`, `needs_user_input`, or `rejected`.
- `answer_text`: synthesized only from bounded evidence rows.
- `evidence`: rows with `path`, `line_start`, `line_end`, `symbol_or_topic`,
  `excerpt`, and `reason`.
- `limitations`: uncertainty, caps, missing evidence, or rejection reasons.
- `trace_summary`: public-safe planning and evidence-collection trace notes.

Source-scope rules:

- Repository scope may inspect safe text files in the repository.
- Directory scope may inspect only safe text files under the scoped directory.
- File scope may inspect only the scoped file unless a future bounded
  repository-level symbol lookup is explicitly recorded in `trace_summary`.

Outcome rules:

- Supported read-only questions return evidence-backed answers when enough
  local evidence is available.
- Ambiguous symbols or overly broad requests return `needs_user_input`.
- Requests to write patches, run commands, inspect secrets, dump raw files,
  analyze binary assets, use unsupported private sources, certify legal/security
  status, or answer current external facts return `rejected`.
