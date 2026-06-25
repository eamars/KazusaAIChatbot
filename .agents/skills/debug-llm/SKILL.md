---
name: debug-llm
description: Create human-readable quality evaluation artifacts for LLM debug, prompt testing, prompt comparison, model routing, RAG/cognition/consolidation/debug-channel behavior, and regression review. Use whenever a task asks to test, debug, inspect, evaluate, compare, or validate LLM output quality; when running live/local LLM calls; when changing prompts; or when reviewing possible LLM regressions.
---

# Debug LLM

Use this skill for LLM debug work where the result must be judged by a human.
The goal is to expose real model behavior clearly enough for quality review,
not to hide it behind pass/fail checks or raw JSON.

## Core Rule

Never present an LLM debug/test result as complete from deterministic success
alone. Schema validity, no exception, accepted validation, or a passing command
only proves the harness ran. The user still needs to evaluate output quality.

## Agent-Authored Review Boundary

The human-readable review is an agent-authored artifact. Do not generate
Markdown, HTML, prose reports, quality assessments, carryover explanations, or
human judgment artifacts from scripts, tests, CLIs, experiment harnesses, or
production/debug code.

Scripts may emit only raw or structured evidence: JSON, JSONL, CSV, logs,
prompt captures, model outputs, parser results, validation status, counts, and
file paths. After inspecting that evidence, the agent must write the readable
review itself. If an existing harness emits a report, remove or ignore that
path and author the review from the raw trace instead.

## Required Artifact

For every LLM debug/test run, the agent must produce a human-readable review
artifact first. Use Markdown sections and tables by default unless the user
requests a different readable shape. Keep JSON, logs, and raw traces as
appendix or linked evidence, not as the primary evaluation surface.

The artifact must include real data from the run:

- **Run Context**: command, input source, timestamp if available, model route,
  model name, prompt or code version, and whether the data is production,
  exported, fixture, or synthetic.
- **Evaluation Goal**: what quality question the human is expected to answer.
- **Input**: raw input message or prompt payload, plus interpreted input if the
  pipeline transforms the raw input before the model sees it.
- **Output**: raw model output, plus interpreted/parsed/normalized output if
  the pipeline transforms it after the call.
- **Decision/Behavior Summary**: what the model decided or produced, in a table
  or short sections organized by target, turn, case, tool call, or output item.
- **Quality Notes**: agent interpretation grounded only in the displayed real
  input/output.
- **Validation**: deterministic parser/schema/contract results as supporting
  evidence, clearly separated from quality judgment.
- **Raw Evidence Appendix**: paths or excerpts for JSON/logs/traces needed for
  replay or deeper inspection.

## Comparison Runs

When comparing before/after prompts, models, routes, or data shaping, use the
same real input where possible and present side-by-side tables.

The comparison must include:

- input identity and any input differences;
- before output raw/interpreted summary;
- after output raw/interpreted summary;
- quality deltas;
- regressions;
- improvements;
- validation deltas;
- human attention points.

Do not make the current baseline the source of truth. If the newer prompt or
data shape produces behavior that better matches `AGENTS.md`, subsystem docs,
or the user's stated quality criteria, treat the newer result as favorable even
when it differs from the baseline.

## Long Or Messy Outputs

When raw output is too long, noisy, or JSON-heavy:

- show the interpreted data first;
- highlight the exact input/output fields that need human attention;
- quote short raw excerpts only when they explain the behavior;
- link or save the full raw output separately;
- do not ask the user to inspect raw JSON as the main review path.

## Real Data Policy

Use real run data, not mockups, for performance evaluation. Synthetic fixtures
are allowed only when the task is explicitly to test harness mechanics; label
them as synthetic and do not treat them as LLM performance evidence.

Redact credentials, tokens, and secrets. Avoid altering semantic content needed
to judge LLM quality. Mark any redaction explicitly.

For production dialog RCA, prefer the protected trace export path before
writing custom queries:

```powershell
venv\Scripts\python.exe -m scripts.export_dialog_trace_review_input --dialog-text "<visible dialog>"
```

Use the resulting JSON as raw evidence, then author the human-readable review
yourself.

## Agent Interpretation

Base interpretation on the real input and output shown in the artifact.
Do not infer success from deterministic checks. Do not summarize from memory if
the artifact can be read from disk or regenerated.

When judging quality, use the task's domain criteria first. In this repo, apply
the `AGENTS.md` character-brain and architecture criteria when relevant:

- character judgment quality;
- target/source ownership;
- prompt-safe evidence use;
- routing and persistence boundaries;
- whether the output supports future maintainable architecture.

## Minimum Markdown Shape

Use this structure unless the task needs a more specific format:

```markdown
# LLM Debug Review

## Run Context
| Field | Value |
| --- | --- |

## Evaluation Goal
Short statement of what the human should judge.

## Input Summary
| Dimension | Real Data |
| --- | --- |

## Output Summary
| Dimension | Real Data |
| --- | --- |

## Decisions Or Generated Items
| Item | Decision/Output | Rationale | Evidence | Human Attention |
| --- | --- | --- | --- | --- |

## Quality Assessment
- Grounded observations from the real input/output.
- Open quality questions for the human.

## Validation Results
| Check | Result | Meaning |
| --- | --- | --- |

## Raw Evidence
- JSON/report/log paths.
```

For before/after comparison, replace the middle sections with:

```markdown
## Side-By-Side Comparison
| Dimension | Before | After | Quality Delta | Human Attention |
| --- | --- | --- | --- | --- |
```

## Stop Conditions

Stop and ask for clarification if:

- the real input/output cannot be accessed or regenerated;
- the user asks for quality evaluation but no real LLM run exists;
- the only available artifact is raw JSON and you cannot produce a readable
  summary without risking semantic distortion;
- privacy constraints prevent showing the data needed for human judgment.
