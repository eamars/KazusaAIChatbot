# Kazusa Documentation Guide

## Document Control

- Owning area: project documentation
- Applies to: top-level READMEs, HOWTO, module READMEs, package guides, test
  docs, and development-plan records
- Source evidence: current source code, deterministic tests, module ICDs,
  `docs/HOWTO.md`, and `development_plans/README.md`
- Change policy: keep living docs current; leave completed and superseded
  development-plan bodies as historical records

## Purpose

This guide defines how Kazusa documentation is organized and how future agents
should update it without flattening separate ownership boundaries. The goal is
to make each document's role clear enough that readers know whether they are
looking at a summary, a runbook, a module contract, a test note, or historical
execution evidence.

## Document Roles

| Document family | Role | Primary owner | Update rule |
|---|---|---|---|
| `README.md`, `README_CN.md` | Top-level overview | Project | Summarize architecture, capabilities, startup path, and doc index. Do not duplicate fragile low-level schemas. |
| `docs/HOWTO.md` | Operator runbook | Project operations | Own setup order, environment variables, commands, adapters, HTTP runbook notes, and testing commands. |
| `AGENTS.md` | Agent instruction | Project owner | Own process rules and architecture guardrails for agent work. |
| `src/**/README.md` | Module ICD or package guide | Module owner | Own module purpose, boundary, public interfaces, runtime flow, failure behavior, and tests. |
| `tests/**/README.md` | Test documentation | Test owner | Explain harness setup or manual expectations for that test family. |
| `development_plans/README.md` | Plan lifecycle registry | Development plans | Own active, reference, historical, and triage lifecycle classification. |
| `development_plans/active/**/*.md` | Active execution contract | Plan owner | Execute only when the registry and plan status allow it. |
| `development_plans/reference/**/*.md` | Reference design | Plan owner | Use as context only. |
| `development_plans/archive/**/*.md` | Historical evidence | Plan owner | Do not rewrite for style or current terminology. |

## Source-Of-Truth Hierarchy

Use the narrowest authoritative source for the claim being edited:

1. Runtime source code and deterministic tests own implemented behavior.
2. Module ICDs under `src/**/README.md` own module-level public contracts.
3. `docs/HOWTO.md` owns operator order, environment examples, commands, and
   runbook endpoint notes.
4. Top-level READMEs summarize current system knowledge and link to the
   contract docs.
5. Development plans record lifecycle-bound intent and evidence; archived plans
   do not define current runtime behavior.

When source and docs disagree, document the current implemented behavior and
record suspected runtime defects as follow-up work instead of changing code
inside a documentation pass.

## Module README Section Contract

Living module READMEs should use the relevant subset of this vocabulary:

- `Document Control`
- `Purpose`
- `Ownership Boundary` or `Boundary`
- `Public Interfaces` or `Public Contract`
- `Input And Output Contracts`
- `Runtime Flow`
- `Configuration`
- `Persistence` or `Storage Contract`
- `Failure Behavior`
- `Observability`
- `Testing Contract` or `Verification`
- `Forbidden Paths`
- `Change Control`

Small package guides may omit sections that do not apply, but they still need
enough interface detail for a caller or future agent to avoid importing
internals, inventing alternate entrypoints, or crossing ownership boundaries.

## Bilingual Parity

`README.md` and `README_CN.md` are paired top-level documents. Parity is
semantic, not line-for-line translation:

- both must name the same major runtime layers;
- both must represent the same model route families;
- both must describe the same normal startup path;
- both must present the same project status;
- both must link the current major module ICDs and guides.

Chinese prose may be more compact, but it must not omit a current major
subsystem named by the English README.

## Historical Plan Policy

Completed and superseded plans under `development_plans/archive/**` are
historical evidence. Do not rewrite their bodies for style, route names, module
vocabulary, or current architecture. Correct only registry mistakes or explicit
supersession metadata when a separate approved plan scopes that change.

Active plans under `development_plans/active/**` are lifecycle contracts, not
general documentation pages. Update their status and evidence only according to
`development_plans/README.md` and the plan's own execution gates.

## Update Workflow

1. Check `git status --short` and preserve unrelated user changes.
2. Read `README.md`, `README_CN.md`, `docs/HOWTO.md`, the relevant module
   README, source files, tests, and `development_plans/README.md` when plans
   are involved.
3. Update module ICDs first when a module contract changed.
4. Update `docs/HOWTO.md` for operator-facing setup or command drift.
5. Update top-level READMEs as summaries and links after lower-level docs are
   correct.
6. Add or update focused documentation tests for route lists, startup order,
   interface sections, and high-risk cross-document parity.
7. Run focused tests and static checks before recording completion.

## Forbidden Paths

- Do not modify production code to make documentation tests pass in a
  documentation-only plan.
- Do not add compatibility vocabulary, alias modules, or shared runtime
  abstractions only to make docs easier to summarize.
- Do not treat RAG evidence, worker output, or retrieval payloads as persona
  stance or final dialog.
- Do not expose raw prompts, secrets, database rows, adapter wire syntax, raw
  platform ids, embeddings, or protected trace payloads in public docs.
- Do not rewrite archived plans as if they were living architecture docs.

## Verification

Use `venv\Scripts\python.exe -m pytest tests\test_documentation_harmonization.py -q`
for the focused documentation contract. Then run the existing doc-sensitive
tests named by the active plan before final sign-off.
