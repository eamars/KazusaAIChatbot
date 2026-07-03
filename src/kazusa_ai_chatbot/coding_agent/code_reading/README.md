# Code Reading Subagent ICD

`code_reading` is the read-only subagent. It answers questions from a
successful source-fetching contract and never fetches, clones, writes, executes
project commands, installs packages, or integrates with Kazusa runtime paths.

Public entrypoint:

```python
from kazusa_ai_chatbot.coding_agent.code_reading import run
```

`run(request: CodeReadingRequest) -> CodeReadingResult` consumes:

- `question`: the user-visible code-reading question.
- `repository`: the successful `CodeRepositoryRef` returned by source fetching.
- `source_scope`: the successful `CodeSourceScope` returned by source fetching.
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
-> reading PM consumes ReadingPMInput and returns ReadingPMDecision
-> bounded ReadingProgrammerTask objects
-> programmer workers return ReadingProgrammerReport objects
-> PM sufficiency check
-> report-based final synthesis
```

The PM owns question intent, decomposition, assignment boundaries,
sufficiency, limitations, and final synthesis. Programmer workers own exactly
one bounded local inspection task and return compact report memory. The final
answer is synthesized from programmer reports and selected evidence rows, not
from an unbounded source context or raw search output.

Code-facing contract names follow the shared coding-agent role vocabulary:
`ReadingPMInput`, `ReadingPMDecision`, `ReadingProgrammerTask`, and
`ReadingProgrammerReport`. The model-facing reading JSON remains the workflow
ICD; fields such as `assignments` and `assignment_id` are reading-domain
fields.

`ReadingPMInput` contains:

- `question`
- `repository_summary`
- `source_scope`
- `repo_map_summary`
- `previous_reports`

`ReadingPMDecision` contains:

- `status`: `need_programmers`, `sufficient`, `needs_user_input`, or
  `overloaded`
- `intent`
- `required_slots`
- `assignments`
- `missing_slots`

Each `ReadingProgrammerTask` must declare:

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

Each `ReadingProgrammerReport` contains:

- assignment identity
- status
- repo-relative files read
- typed facts with evidence references
- bounded evidence rows
- open questions

Code reading caps one PM at three programmer assignments per wave, three waves,
and six total programmer reports. When a question exceeds those limits, the PM
returns `overloaded`; code reading reports a limitation or asks for a narrower
user scope instead of pretending to read a whole project. Full distributed
master/subsystem PM fan-out is not implemented in the current code-reading
workflow.

LLM-backed PM calls use `CODING_AGENT_PM_LLM`; final synthesis intentionally
uses the same PM route. Programmer workers use `CODING_AGENT_PROGRAMMER_LLM`.
Both routes require base URL, API key, and model settings. Code reading does
not define a separate synthesizer LLM route.

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
