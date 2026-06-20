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

Internally, `run(...)` is a PM/programmer reading flow:

```text
CodeReadingRequest
-> reading supervisor
-> repository-map summary
-> reading PM consumes PMInput and returns PMDecision
-> bounded ProgrammerAssignment objects
-> programmer workers return ProgrammerReport objects
-> PM sufficiency check
-> report-based final synthesis
```

The PM owns question intent, decomposition, assignment boundaries,
sufficiency, limitations, and final synthesis. Programmer workers own exactly
one bounded local inspection task and return compact report memory. The final
answer is synthesized from programmer reports and selected evidence rows, not
from an unbounded source context or raw search output.

`PMInput` contains:

- `question`
- `repository_summary`
- `source_scope`
- `repo_map_summary`
- `previous_reports`

`PMDecision` contains:

- `status`: `need_programmers`, `sufficient`, `needs_user_input`, or
  `overloaded`
- `intent`
- `required_slots`
- `assignments`
- `missing_slots`

Each `ProgrammerAssignment` must declare:

- `assignment_id`
- `role`
- `scope.kind`: `file`, `directory`, `symbol`, or `search`
- `scope.values`
- `questions`
- `required_slots`

Assignment validation rejects unbounded whole-repository reads. A programmer
must not inspect files outside its assignment boundary, write files, run project
commands, install packages, fetch code, inspect `.env`, inspect `.git`, inspect
secret-like paths, or inspect binary assets. File, excerpt, report, and wave
limits are supervisor-owned deterministic policy, not PM-generated fields.

Each `ProgrammerReport` contains:

- assignment identity
- status
- repo-relative files read
- typed facts with evidence references
- bounded evidence rows
- open questions

Phase 1 caps one PM at three programmer assignments per wave, two waves, and
six total programmer reports. When a question exceeds those limits, the PM
returns `overloaded`; Phase 1 reports a limitation or asks for a narrower user
scope instead of pretending to read a whole project. Full distributed
master/subsystem PM fan-out is not implemented in Phase 1.

LLM-backed PM, programmer, or synthesis calls use the effective
`CODING_AGENT_LLM` route. `CODING_AGENT_LLM_*` is optional; when it is absent,
the effective route falls back to `BACKGROUND_WORK_LLM_*`. Partial
`CODING_AGENT_LLM_BASE_URL`, `CODING_AGENT_LLM_API_KEY`, or
`CODING_AGENT_LLM_MODEL` configuration is invalid. Phase 1 does not define
separate PM, programmer, or synthesizer LLM routes.

`CodeReadingResult` contains:

- `status`: `succeeded`, `failed`, `needs_user_input`, or `rejected`.
- `answer_text`: synthesized only from bounded evidence rows.
- `evidence`: rows with `path`, `line_start`, `line_end`, `symbol_or_topic`,
  `excerpt`, and `reason`.
- `limitations`: uncertainty, caps, missing evidence, or rejection reasons.
- `trace_summary`: public-safe planning and evidence-collection trace notes.

Public traces are sanitized but must show the decomposition path, including
items such as:

- `reading_pm:repository_map`
- `reading_pm:work_plan`
- `programmer:<role name>`
- `programmer_report`
- `reading_pm:sufficiency=<status>`

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
