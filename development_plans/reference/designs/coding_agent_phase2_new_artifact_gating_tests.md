# coding agent phase2 new artifact gating tests

## Status

- Type: supporting verification procedure and pass criteria
- Status: reference evidence for Phase 2
- Parent plan:
  `development_plans/active/short_term/coding_agent_phase2_code_writing_plan.md`
- Scope: Phase 2 `code_writing` new-artifact gates only

## Purpose

This document defines the five Phase 2 hard gates, test procedure, and
AI-agent pass criteria for those gates. The gates validate whether the coding
agent can create new artifacts from scratch through the supervisor, writing,
and patching workflow.

The gates are not deterministic golden-output tests. They are real LLM
artifact-generation tests reviewed by an AI agent against the original user
demand. Flexible implementation choices are acceptable when the proposed
artifacts satisfy the requested behavior and preserve the Phase 2 architecture.

## Evaluation Rules

- Run one gate at a time through `coding_agent.propose_code_change(...)`.
- E2E gates must be implemented as pytest-style real LLM tests marked
  `live_llm`. Do not run ad hoc inline Python commands from the terminal as
  the gate execution method.
- Each E2E test function must call `coding_agent.propose_code_change(...)`,
  capture the raw response, preserve the role trace, and record the
  materialized review artifact paths for human inspection.
- The pytest test body must not use deterministic semantic pass/fail criteria
  for artifact quality. It may fail only when the run cannot produce reviewable
  evidence, such as environment failure, exception, missing response, missing
  trace, missing artifact package, missing materialized artifact file, or real
  workspace mutation. The AI reviewer owns the quality judgment.
- Do not add generated-code execution, generated-test execution, validation
  feedback loops, repair loops, or target-project command execution in Phase 2.
  Phase 2 gates inspect generated artifacts as review packages only.
- Do not require exact file names, exact function names, exact module layout,
  exact prose, exact decomposition, or exact implementation strategy.
- Do not use keyword matching over user input to decide pass or fail.
- The harness may enforce only evidence-collection and safety checks: request
  completed, trace exists, output is non-empty, artifact package exists,
  materialized review artifact files exist, public response is sanitized,
  and no real workspace mutation occurred. These checks prove the run is
  reviewable; they do not decide artifact quality.
- The AI reviewer owns quality judgment. The reviewer reads the request,
  trace, final response, artifact manifest, and generated artifact content.
- Environment failure is failure. Failure to produce an answer is failure.
  Unrelated answers are failure. Incomplete output due to loop limits is
  failure. Output from the wrong workflow is failure.
- Compilation is not required in Phase 2. Code-level mistakes are recorded in
  the review comment and may still pass. Structural mistakes fail the gate:
  missing artifact bodies, impossible artifact relationships, workflow output
  that cannot be reviewed, or artifacts that cannot plausibly satisfy the
  request.
- The final review MUST be reviewed by human. At the end of each test, present
  the direct materialized artifact path to the human for evaluation.

## E2E Reporting Contract

After each E2E gate run, report these fields to the human regardless of pass
or fail:

- Gate id and AI-review pass/fail status.
- Direct materialized artifact path for every generated file, using the shape
  `test_artifacts\coding_agent_phase2_e2e_workspace\<gate>\writing_validation\<validation_id>\<generated_path>`.
- Review artifact path.
- Raw response or trace path.
- A short statement that the agent read the materialized artifact content after
  the run.

The direct materialized artifact path is the primary human-inspection path.
The raw response JSON remains supporting evidence because it contains the full
trace and patch payload.

## AI Review Rubric

The AI reviewer must judge each gate using this rubric. Passing requires every
required dimension to be acceptable.

| Dimension             | Pass Condition                                                                                        |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| Workflow ownership    | The result comes from the coding supervisor dispatching the Phase 2 writing and patching workflow.    |
| Scope control         | The artifact proposal creates new artifacts only and does not modify existing source semantically.    |
| Request fit           | The generated artifacts address the original user demand, not a related or generic task.              |
| Artifact completeness | All files or sections needed for the requested tool are present as artifact bodies.                   |
| Internal consistency  | Generated files reference each other consistently enough to be plausible without execution.           |
| User-facing clarity   | Final response explains what was produced, how to use it at a high level, and any limitation.         |
| Safety and privacy    | Output does not expose local roots, cache keys, secrets, raw traces, or unsupported execution claims. |
| Loop outcome          | The run finishes within loop limits with a complete artifact package or a clear failure.              |

The reviewer must read the materialized artifact content and write a short
review comment. A gate may pass with code mistakes when the output is sound,
request-aligned, structurally coherent, and produced by the intended workflow.

## Gate 01 - Easy Single-File Utility

Difficulty: easy

User request:

```text
Create a single Python command-line script that reads a plain text application
log file and counts entries by severity. Each valid line starts with one of
DEBUG, INFO, WARNING, ERROR, or CRITICAL followed by a space and the message.
The script should print a terminal summary with one count per severity, report
how many malformed lines were skipped, handle a missing input file clearly, and
use only the Python standard library.
```

Expected artifact shape:

- One Python script artifact.
- No tests or package metadata required.
- Standard-library only.

AI pass criteria:

- The script accepts an input path from the command line.
- It counts the five severities separately.
- It reports skipped malformed lines.
- It handles a missing file with a clear error path.
- It prints a usable terminal summary.

Acceptable variation:

- `argparse` or simple `sys.argv` parsing.
- Any clear summary format.
- Exit code choices may vary if the behavior is explained.

Failure examples:

- Only counts total lines.
- Treats every unknown line as a severity.
- Requires a third-party package.
- Produces only prose and no script artifact.

## Gate 02 - Low-Medium CLI Plus Tests

Difficulty: low_medium

User request:

```text
Create a small Python utility that converts JSONL records into CSV. It should
include a command-line script and focused tests. The CLI must accept input and
output paths, an optional list of fields, preserve stable column order, write
blank cells for missing fields, report malformed JSON lines without aborting
the whole conversion, and use only the Python standard library.
```

Expected artifact shape:

- One source artifact for the CLI/conversion logic.
- One focused test artifact.

AI pass criteria:

- The conversion behavior is implemented, not only described.
- Field selection is supported and column order is stable.
- Missing fields become blank cells.
- Malformed JSONL lines are reported while valid rows still convert.
- Tests cover normal conversion and at least one malformed-line path.

Acceptable variation:

- CLI and conversion logic may be in one file.
- Tests may use `unittest` or `pytest` style, but must be clear enough for an
  AI reviewer to inspect.

Failure examples:

- Aborts on the first malformed line.
- Ignores requested field order.
- Emits tests that do not exercise conversion behavior.

## Gate 03 - Medium Small Package

Difficulty: medium

User request:

```text
Create a small Python package for checking local Markdown links. It should
provide a reusable function and a CLI. The checker must scan markdown files
under a directory, collect headings as anchors, report duplicate anchors inside
one file, report broken relative links to local markdown files or anchors, and
include focused tests for anchor generation, duplicate anchors, and broken
relative links.
```

Expected artifact shape:

- Package source files or one package module plus CLI entry artifact.
- Focused tests.
- Optional README.

AI pass criteria:

- The proposed artifacts separate reusable checking logic from CLI handling.
- Anchor generation is present and plausible for common Markdown headings.
- Duplicate anchors and broken relative links are both handled.
- Tests cover the three requested behavior groups.
- The CLI reports findings in a readable way.

Acceptable variation:

- Package may be flat or use a `src/` layout.
- Output may be text, JSON, or both if the behavior is clear.

Failure examples:

- Only checks whether markdown files exist.
- Checks external HTTP links instead of local relative links.
- No reusable function is exposed.

## Gate 04 - Medium-Hard Managed CLI Project

Difficulty: medium_hard

User request:

```text
Create a small Python CLI project that summarizes task notes. It should read a
directory of dated text notes, group entries by project name, write a summary
Markdown file, support a simple JSON config file for input directory, output
path, and included projects, include a README explaining the workflow, and
include focused tests for parsing notes, applying config filters, and rendering
the summary.
```

Expected artifact shape:

- CLI source artifact.
- Parser/source logic artifact.
- Renderer or summary generation artifact.
- README artifact.
- Focused tests.

AI pass criteria:

- The artifacts form a coherent small project rather than one disconnected
  snippet.
- Notes are parsed into dated entries with project grouping.
- JSON config affects input/output and included projects.
- Markdown rendering is implemented.
- README explains how to run the tool and what input format it expects.
- Tests cover parser, config filtering, and rendering.

Acceptable variation:

- The note format may be defined by the generated README if it is clear and
  consistent with the parser.
- The project may use standard-library-only implementation.

Failure examples:

- Ignores the JSON config.
- Produces only a README with no implementation.
- Generates tests that do not match the parser's note format.

## Gate 05 - Hard Multi-Source Data Tool

Difficulty: hard

User request:

```text
Create a small Python project that reads a CSV inventory of pages, fetches each
listed URL, extracts the HTML title and first h1 heading, merges those values
with the inventory rows, and writes a consolidated CSV report. It should include
a CLI, source modules, mocked HTTP tests, and a README that explains the input
CSV columns and command workflow. The project may use only the Python standard
library.
```

Expected artifact shape:

- CLI source artifact.
- CSV/inventory handling artifact.
- Web fetch and HTML extraction artifact.
- Focused tests with mocked HTTP behavior.
- README artifact.

AI pass criteria:

- The proposed project has a coherent multi-file structure.
- CSV input and output behavior is implemented.
- URL fetching and HTML title/h1 extraction are implemented or clearly
  isolated behind testable functions.
- Tests avoid real network calls and use mocked or fake HTTP responses.
- README documents required input columns and command workflow.
- Final response accurately describes the produced artifacts and limitations.

Acceptable variation:

- HTML extraction may use `html.parser`, regular expressions with reasonable
  caveats, or another standard-library approach.
- Tests may mock `urllib.request.urlopen` or inject a fetch function.
- Network error handling may be simple if it records an error per row.

Failure examples:

- Performs real network access in tests.
- Does not merge fetched data with the original CSV rows.
- Omits the README or the mocked HTTP tests.
- Produces artifacts with inconsistent imports or missing referenced modules.
