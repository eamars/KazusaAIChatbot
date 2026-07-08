# Coding Agent Architecture

## Status

- Type: reference architecture and decision record.
- Status: current reference.
- Execution rule: use this document as context only. Implementation requires
  an approved or in-progress plan under `development_plans/active/`.
- Related execution and direction plans:
  - `development_plans/archive/completed/short_term/coding_agent_phase0_fetching_plan.md`
  - `development_plans/archive/completed/short_term/coding_agent_phase1_code_reading_final_plan.md`
  - `development_plans/archive/completed/short_term/coding_agent_phase2_code_writing_plan.md`
  - `development_plans/archive/completed/short_term/coding_agent_phase2_5_security_boundary_plan.md`
  - `development_plans/archive/completed/short_term/coding_agent_phase3_background_worker_integration_plan.md`
  - `development_plans/archive/completed/short_term/coding_agent_phase4_code_modifying_and_patching_plan.md`
  - `development_plans/archive/completed/short_term/coding_agent_phase5_patch_apply_plan.md`
  - `development_plans/archive/completed/short_term/coding_agent_phase6_code_executing_plan.md`
  - `development_plans/archive/completed/short_term/coding_agent_phase7_existing_source_planning_plan.md`
  - `development_plans/archive/completed/short_term/coding_agent_phase8_verify_repair_loop_plan.md`
  - `development_plans/archive/completed/short_term/coding_agent_phase9_run_supervisor_plan.md`
  - `development_plans/reference/designs/coding_agent_phase9_run_supervisor_architecture.md`
  - `development_plans/reference/designs/coding_agent_phase10_repository_scale_reading_architecture.md`

## Cleanup Note

This reference supersedes the earlier draft wording that treated Phase 7 as
broader repository operations and richer external help. The current direction
places existing-source planning in Phase 7, controlled verify-and-repair in
Phase 8, durable run supervision in Phase 9, and repository-scale reading in
Phase 10.

## Problem

Kazusa needs a coding capability that can answer codebase questions, propose
reviewable code changes, apply approved patches into managed copies, run
bounded verification, and eventually carry a coding task across a durable local
agent session.

The coding agent is not a general assistant shell. It is a specialized coding
subsystem with explicit source, reading, writing, modifying, patching, apply,
execution, verification, run-state, and repository-scale reading boundaries.

## Architectural Goal

The coding agent shall approximate Codex-style coding capability while running
on local or weaker OpenAI-compatible LLMs. It does this through deterministic
workflow decomposition rather than a monolithic frontier-model session.

The stable top-level capability set is:

- `code_fetching`: resolve supported source inputs into a safe local source
  contract.
- `code_reading`: answer code questions from bounded evidence.
- `code_writing`: create source-free new artifacts.
- `code_modifying`: plan semantic changes to existing source from bounded
  source evidence.
- `code_patching`: compile structured artifacts into reviewable patch
  packages and apply approved packages into managed copies.
- `code_executing`: run allowlisted verification commands inside managed
  apply workspaces.
- `code_verifying`: coordinate apply, execution, and capped repair attempts.
- `coding_run`: preserve durable coding-session state and legal lifecycle
  transitions.
- repository-scale reading: decompose broad repository questions across master
  and subsystem PMs.

## Local-LLM-First Architecture

Status: accepted hard requirement.

Kazusa assumes the production model may be weaker than Codex and may have a
limited effective context window. Therefore, the coding agent uses many small,
typed, inspectable calls:

```text
top-level supervisor
-> domain PM
-> bounded programmer task
-> deterministic validator/tool boundary
-> compact report or artifact
-> supervisor ledger
```

Core rules:

- Deterministic code owns validation, permissions, limits, path containment,
  lifecycle transitions, patch mechanics, apply mechanics, execution, and
  public sanitization.
- LLM stages own semantic judgment: task fit, source-owner selection,
  decomposition, local implementation intent, evidence sufficiency, and
  repair reasoning over bounded facts.
- No LLM stage receives unbounded repository source or raw command output.
- Every loop has a hard cap and a terminal blocker.
- Every public response hides absolute roots, cache keys, `.env`, `.git`,
  secret-like files, raw traces, and unbounded source dumps.

## Ownership Boundaries

| Owner | Owns | Does Not Own |
|---|---|---|
| Top-level coding supervisor | Public direct APIs, source-backed workflow selection, bounded interleave between specialists, public response assembly. | Raw code edits, shell execution, persistent run state after Phase 9. |
| `code_fetching` | Source resolution, managed clone/raw/inline/local source contracts, source identity, safe scope projection. | Reading, writing, modifying, patching, execution. |
| `code_reading` | Read-only evidence, PM/programmer reading decomposition, answer synthesis from evidence rows. | Patch stance, edit choice, command execution, external current facts. |
| `code_writing` | Source-free new artifacts through PM/programmer contracts. | Existing-source edits, patch application, command execution. |
| `code_modifying` | Existing-source semantic planning, source-owner selection, File Agent context use, modifying programmer dispatch, structured modification artifacts. | Applying patches, executing commands, direct filesystem mutation. |
| File Agent | Repo-relative path safety, new-artifact reservations, existing-source file context planning, owned/read-only/test/doc/caller path maps. | Semantic edit judgment or final patch selection. |
| `code_patching` | Structured operation validation, diff/file-tree assembly, review materialization, explicit approved apply into managed copies. | LLM patch generation, original-source mutation, command execution. |
| `code_executing` | Allowlisted verification execution inside Phase 5 managed apply workspaces. | Command generation, arbitrary shell, package installation, repair. |
| `code_verifying` | Direct trusted apply/execute/repair orchestration with capped attempts and redacted execution feedback. | Background auto-execution, arbitrary commands, durable coding-session state. |
| `coding_run` | Durable run ledger, state transitions, continuation API, blockers, attempt history, public projection. | Specialist internals, raw execution, adapter delivery. |

## Agent-Space Security Boundary

Coding agents operate in agent space by default. They may produce structured
tool intents, artifacts, patch packages, traces, and managed review/apply
records. Real-world effects require dedicated deterministic owners:

- patch application requires explicit structured approval and targets only a
  managed apply copy;
- command execution requires structured execution specs and targets only a
  managed apply copy;
- background-worker coding tasks remain review-only unless a later approved
  plan adds structured continuation through `coding_run`.

Generated code, generated tests, shell text, and package-install instructions
remain inert proposal material until a deterministic approved boundary acts on
them.

## Runtime Flow

The direct source-backed proposal flow after Phase 7 is:

```text
propose_code_change(...)
-> code_fetching
-> code_reading
-> File Agent existing-source plan
-> modifying PM
-> modifying programmer task(s)
-> modifying PM sufficiency
-> code_patching review materialization
-> CodingPatchProposalResponse
```

The direct trusted verify-and-repair flow after Phase 8 is:

```text
verify_and_repair_code_change(...)
-> proposal
-> structured approval validation
-> managed apply copy
-> bounded execution specs
-> if failed: redacted execution repair feedback
-> Phase 7 modifying repair proposal
-> fresh managed apply copy
-> bounded execution specs
-> CodingVerifyRepairResponse
```

The self-contained coding-agent session after Phase 9 is:

```text
start_coding_run(...)
-> durable run ledger
-> source/evidence/plan/proposal states
-> awaiting structured continuation when needed
-> apply/verification/repair attempts
-> completed, blocked, rejected, failed, or cancelled
```

Repository-scale reading after Phase 10 is:

```text
CodeReadingRequest
-> deterministic repository inventory
-> master reading PM
-> subsystem reading PMs
-> bounded programmer reports
-> evidence graph
-> master synthesis
-> CodeReadingResult
```

## Phase Roadmap

| Phase | Scope | User-Visible Capability |
|---|---|---|
| Phase 0 | Standalone `code_fetching`, safe source contracts, managed source storage, unsupported-source handling. | Direct callers can resolve supported source inputs into public-safe source contracts. |
| Phase 1 | Standalone `code_reading`, read-only PM/programmer evidence flow, direct answer API. | Direct callers can ask code questions with cited local evidence. |
| Phase 2 | Standalone `code_writing`, new-artifact PM/programmer flow, review-only patch materialization for generated artifacts. | Direct callers can request new scripts, modules, docs, tests, or small projects as reviewable artifacts. |
| Phase 2.5 | Agent-space security boundary hardening. | Generated artifacts remain inert unless a later approved boundary applies or executes them. |
| Phase 3 | Background-worker integration for implemented review-only coding-agent work. | Accepted coding tasks can be processed asynchronously and returned as background results. |
| Phase 4 | `code_modifying` and `code_patching` for existing-source patch proposals. | Direct callers can request bounded existing-repository changes as patch proposals. |
| Phase 5 | Explicit approved patch application into managed apply copies. | Trusted callers can apply approved review-valid patch artifacts without mutating original source. |
| Phase 6 | Bounded `python_compileall` and `pytest` execution in managed apply workspaces. | Trusted callers can verify approved applied copies without arbitrary shell access. |
| Phase 7 | Existing-source planning upgrade: active modifying PM plus File Agent existing-source path maps and context planning. | Source-backed proposals gain explicit source-owner planning before programmer edits. |
| Phase 8 | Controlled verify-and-repair loop. | Trusted callers can apply, execute, repair from redacted execution feedback, and rerun within hard caps. |
| Phase 9 | Durable coding run supervisor. | Kazusa gains a self-contained coding-agent session with state, continuation, blockers, attempts, completion, and post-phase E2E readiness for supported workflows. |
| Phase 10 | Repository-scale reading through master/subsystem PMs and evidence graph synthesis. | Kazusa can answer broad architecture, impact, ownership, and migration questions from bounded evidence. |

## Phase Closure Gate Policy

Every coding-agent phase must include at least five committed real-LLM closure
gates that mimic real-life use cases from simple to hard. These tests are
regression assets and must live in the repository under `tests/` or committed
fixture directories, not only in ad hoc local scripts.

The five gates for each phase must satisfy:

- simple, focused case;
- small multi-file case;
- parser or edge-case case;
- cross-layer behavior case;
- hard mixed case with realistic source/test/docs interaction.

Each real-LLM gate must:

- run through the public or role-level API that the phase is closing;
- use realistic fixture source or captured production-like input;
- emit durable raw evidence under `test_artifacts/llm_traces/...`;
- record input, model route/config when available, raw output, parsed output,
  trace summary, validation results, behavior rubric, and forbidden failure
  modes;
- assert only structural contract and safety gates in pytest;
- require human or agent-authored quality review before closure;
- be run one case at a time and inspected one case at a time;
- remain committed for future regression runs through the `live_llm` pytest
  marker.

Passing pytest is necessary but not sufficient for real-LLM gate acceptance.
Closure requires trace inspection and an explicit quality judgment against the
phase contract.

## Current Implementation Alignment

As of the Phase 9 executable plan:

- Phases 0 through 8 are implemented and archived as completed records.
- Phase 9 is the active approved short-term implementation step where the
  coding agent becomes a self-contained session loop.
- After Phase 9, supported read-only, proposal, approval, apply, execution,
  repair, cancellation, reload, sanitization, and source-immutability
  workflows must be E2E-testable without Phase 10 or later plans.
- Phase 10 improves broad repository intelligence but is not required for the
  core self-contained loop or supported Phase 9 E2E workflows.

## Scope Boundaries

The coding agent does not become a generic assistant or unrestricted shell.

Out of scope unless a future approved plan explicitly adds the capability:

- arbitrary shell command generation;
- package installation or dependency solving;
- deployment, database mutation, adapter delivery, or repository push;
- mutation of original source checkouts;
- background auto-apply or auto-execute from accepted prose;
- unbounded repository ingestion into prompts;
- compatibility shim layers that preserve stale call shapes during contract
  changes.
