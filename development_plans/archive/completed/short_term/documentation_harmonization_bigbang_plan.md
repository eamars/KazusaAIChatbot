# documentation harmonization bigbang plan

## Summary

- Goal: Audit and harmonize Kazusa's living documentation so top-level docs,
  module ICDs, bilingual summaries, runbooks, and subagent-interface docs
  accurately reflect the current code, tests, and ownership boundaries.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `test-style-and-execution`, `py-style`, `cjk-safety`
- Overall cutover strategy: audit-first research gate, then bigbang
  documentation replacement for living docs only. No compatibility vocabulary,
  no parallel old/new document formats, and no production-code changes.
- Highest-risk areas: stale module ICD claims, bilingual drift, incorrect
  setup order, subagent-interface over-abstraction, historical plan churn,
  accidental production-code edits, and executing edits before the audit report
  exists.
- Acceptance criteria: the audit report is recorded before any existing
  document or code is changed; living docs in scope are harmonized against
  source/tests; subagent-interface documentation uses one cross-family
  vocabulary without forcing a runtime abstraction; top-level English/Chinese
  docs and HOWTO agree with module ICDs; doc-regression checks pass; an
  independent review signs off the final diff and evidence.

## Context

The user requested a development plan to review and harmonize all documents,
including module-level accuracy against code, module format consistency,
interface detail, subagent-interface documentation, top-level sufficiency,
top-level/module consistency, cross-language consistency, HOWTO setup order,
and additional documentation improvements.

Current documentation has several legitimate document roles:

- top-level overview docs: `README.md`, `README_CN.md`
- operator runbook: `docs/HOWTO.md`
- active agent instructions: `AGENTS.md`
- module ICDs: `src/**/README.md`
- active plan registry: `development_plans/README.md`
- active and reference development plans under `development_plans/active/`
  and `development_plans/reference/`
- historical plans under `development_plans/archive/`

These roles must not be flattened. Module ICDs are normative for module
contracts. Top-level READMEs summarize and link. HOWTO owns setup, environment,
run commands, adapters, HTTP endpoint runbook notes, and testing commands.
Development plans record lifecycle-bound execution contracts and historical
evidence.

The codebase currently has several subagent or worker families with related
but different interfaces:

- RAG helper agents use `BaseRAGHelperAgent.run(task, context, max_attempts)`.
- `web_agent3` source subagents expose module-level `SOURCE`,
  `DESCRIPTION`, `SUPPORTED_ACTIONS`, optional `is_enabled()`, and
  `execute(decision)`.
- `complex_task_resolver` resolver-local subagents expose `SUBAGENT`,
  `DESCRIPTION`, `SUPPORTED_ACTIONS`, `OWNED_NODE_KINDS`, `DEFAULT_ACTION`,
  optional `is_enabled()`, and `create()`, then implement typed
  `ComplexTaskSubagentV1.run(...)`.
- `background_work` workers expose `WORKER`, `DESCRIPTION`, and
  `execute(decision, max_output_chars=...)`.

The harmonized target is a shared documentation vocabulary for these families,
not a shared runtime base class or adapter layer.

The worktree may contain unrelated active implementation changes. At drafting
time, `development_plans/README.md`,
`development_plans/active/short_term/l2_affinity_willingness_boundary_plan.md`,
`docs/HOWTO.md`, multiple `src/kazusa_ai_chatbot/cognition_chain_core/*.py`
files, `src/kazusa_ai_chatbot/config.py`,
`src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`, and several
tests were already modified. This plan must preserve those changes and must
not revert or rewrite them except where the user explicitly approves later
documentation edits inside this plan's scope.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, or
  signing off this plan.
- `local-llm-architecture`: load before changing any documentation that defines
  LLM stage ownership, RAG/cognition/dialog boundaries, prompt-facing
  contracts, subagent responsibilities, worker routing, or capability
  ownership.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `py-style`: load before editing Python test files or helper scripts.
- `cjk-safety`: load before editing Python test files that include Chinese or
  Japanese strings from `README_CN.md`, prompt examples, or bilingual parity
  assertions.

## Mandatory Rules

- Do not execute implementation steps while `Status` is `draft`.
  Implementation requires explicit user approval and status `approved` or
  `in_progress`.
- Stage 1 audit is a hard gate. Before the Stage 1 audit report is complete
  and signed off, the executing parent or any subagent must not edit existing
  documentation, tests, production code, prompts, configuration, scripts, or
  registry rows. The only allowed new tracked document during Stage 1 is
  `development_plans/reference/documentation_harmonization_audit_report.md`.
- Do not start the documentation execution subagent until the Stage 1 audit
  report exists, names source-of-truth evidence, and is signed off in this
  plan's progress checklist.
- After automatic context compaction, the parent or active execution agent must
  reread this entire plan before continuing implementation, verification,
  handoff, lifecycle updates, or final reporting.
- After signing off any major checklist stage, the parent or active execution
  agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the
  parent agent must run the `Independent Code Review` gate and record the
  result in `Execution Evidence`.
- Use parent-led native subagent execution. If native subagent capability is
  unavailable, stop before document implementation unless the user explicitly
  approves fallback execution.
- Never read `.env` during audit, implementation, or verification.
- Do not modify production Python code in this plan. If the audit finds code
  defects, record them as follow-up findings and update docs only where docs
  currently misstate implemented behavior.
- Do not change runtime behavior, prompt behavior, LLM call counts, database
  schemas, service startup logic, adapter behavior, scheduler behavior, or
  worker registration.
- Do not create compatibility docs that preserve old and new vocabulary in
  parallel. After the bigbang doc edit, living docs must use the current
  canonical vocabulary.
- Do not rewrite historical completed or superseded plans for style. Completed
  and superseded plans are historical evidence. Only registry links, status
  rows, or explicit supersession banners may be corrected when the audit
  proves the current registry is misleading.
- Treat `README.md` and `README_CN.md` as paired top-level documents. A new
  top-level English capability, run path, layer, or model-route claim must have
  a semantically equivalent Chinese counterpart unless explicitly marked as
  English-only developer wording.
- Treat `docs/HOWTO.md` as a runbook. It must preserve the correct operator
  order: install package, configure environment, load character profile, start
  control console for normal local operation, then start or manage brain and
  adapters.
- Treat module READMEs as ICDs. Each living module README must name purpose,
  ownership boundary, public interfaces, input/output contracts where
  applicable, runtime flow, failure behavior, tests, and forbidden paths at the
  level appropriate for that module.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual file edits.
- Keep edits scoped to documentation harmonization and doc-regression tests.
  Do not perform unrelated cleanup, formatting churn, dependency upgrades, or
  plan lifecycle changes outside this plan.

## Must Do

- Create `development_plans/reference/documentation_harmonization_audit_report.md`
  as the first execution artifact.
- In the audit report, classify every Markdown document as one of:
  top-level overview, runbook, module ICD, package guide, active plan,
  reference design, historical completed plan, superseded plan, test docs, or
  agent instruction.
- In the audit report, map each living document to its source-of-truth code,
  tests, or registry evidence.
- In the audit report, record findings for:
  - module-level accuracy against source and tests;
  - module README format consistency;
  - module interface detail sufficiency;
  - subagent and worker interface documentation;
  - top-level overview correctness and sufficiency;
  - top-level/module consistency;
  - English/Chinese semantic parity;
  - HOWTO and initialization order;
  - missing links, stale links, stale route names, stale startup paths, and
    missing testing guidance.
- Create `docs/DOCUMENTATION_GUIDE.md` with the living-document roles,
  module-README section contract, source-of-truth hierarchy, bilingual parity
  rule, and historical-plan edit policy.
- Create `docs/SUBAGENT_INTERFACES.md` with the shared documentation vocabulary
  for RAG helper agents, `web_agent3` source subagents,
  `complex_task_resolver` subagents, and `background_work` workers.
- Harmonize living module READMEs in scope against the audit report and the new
  documentation guide.
- Reconcile `README.md`, `README_CN.md`, and `docs/HOWTO.md` with the module
  ICDs and current startup/config/runtime code.
- Add or update doc-regression tests after the audit report is complete.
- Update `development_plans/README.md` only for this plan's registry row and,
  after execution, final lifecycle evidence required by the registry.

## Deferred

- Do not modify production code.
- Do not fix runtime bugs discovered during the audit.
- Do not change prompts, model routes, LLM call configs, graph routing, RAG
  dispatch, cognition behavior, dialog behavior, scheduler behavior, adapter
  behavior, persistence, or service startup.
- Do not add a shared runtime subagent base class, wrapper, adapter, fallback,
  registry bridge, compatibility shim, or universal worker interface.
- Do not decommission legacy modules, collections, background-artifact
  compatibility, old queued rows, old plan records, or old tests unless a
  later approved plan explicitly scopes that work.
- Do not rewrite archived completed plans or superseded historical documents
  for consistency.
- Do not perform broad Markdown formatting churn unrelated to audit findings.
- Do not add a website, frontend, generated diagrams, screenshots, or
  rendered-doc workflow.
- Do not add external documentation tooling dependencies.

## Cutover Policy

Overall strategy: audit-first research gate, then bigbang documentation
harmonization for living docs.

| Area | Policy | Instruction |
|---|---|---|
| Stage 1 audit | gate | Create and sign off the audit report before any existing document or code is edited. |
| Living module READMEs | bigbang | Replace stale section shape and terminology with the harmonized ICD vocabulary in one pass. Do not preserve old section names as aliases. |
| Top-level READMEs | bigbang | Reconcile English and Chinese summaries to the same architecture and capability set. Do not leave one language stale. |
| HOWTO | bigbang | Rewrite stale runbook order, env lists, and endpoint/setup notes directly to the current code-backed order. |
| Subagent docs | bigbang documentation only | Add one shared documentation guide for subagent families. Do not add runtime abstraction or compatibility code. |
| Historical plans | no-op | Leave completed and superseded plan bodies unchanged except for proven registry or banner corrections. |
| Production code | no-op | Do not edit production code. Record code defects as follow-up findings. |
| Tests | additive or update | Add or update documentation regression tests after the audit gate. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each
  area.
- The agent must not choose a more conservative compatibility strategy by
  default.
- For `bigbang` documentation areas, replace stale wording and remove obsolete
  vocabulary instead of documenting old and new wordings side by side.
- For `no-op` areas, leave files unchanged unless a later user-approved plan
  changes the policy.
- Any change to a cutover policy requires user approval before implementation.

## Target State

After completion, Kazusa has a coherent documentation system:

- Top-level READMEs explain the current system architecture, capabilities,
  startup path, runtime layers, and project status without duplicating fragile
  low-level contracts.
- `docs/HOWTO.md` gives the correct setup, configuration, startup, adapter,
  operations, and testing order.
- Module READMEs read as ICDs with consistent section vocabulary and enough
  interface detail for callers and future agents to use the module without
  importing internals.
- `docs/DOCUMENTATION_GUIDE.md` defines document roles, source-of-truth
  hierarchy, module-README expectations, bilingual parity expectations, and
  historical-plan handling.
- `docs/SUBAGENT_INTERFACES.md` documents the common vocabulary for subagent
  and worker families while preserving each family's actual code contract.
- English and Chinese top-level documentation describe the same capabilities
  and current runtime shape.
- Documentation regression tests protect high-risk claims such as required ICD
  sections, subagent interface fields, top-level route lists, HOWTO startup
  order, and bilingual capability parity.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| First execution artifact | The audit report is mandatory before edits to existing docs or code. | The user explicitly required audit-before-touch, and the repo has many living and historical docs. |
| Source hierarchy | Code and tests are source of truth for implemented behavior; module ICDs are source of truth for documented module contracts; top-level docs summarize. | Prevents top-level docs from becoming a second contract surface. |
| Historical plans | Archived completed and superseded plan bodies stay historical. | The development-plan registry forbids appending new scope or rewriting completed records as living docs. |
| Subagent harmonization | Harmonize documentation vocabulary, not runtime interfaces. | Existing subagent families have different valid code contracts and ownership boundaries. |
| Bigbang docs | Living docs move to the new documentation vocabulary in one pass. | Avoids parallel doc formats, aliases, and compatibility language. |
| Code changes | Production code is out of scope. | The user asked for documentation review and harmonization, not behavior changes. |
| Doc tests | Add focused doc-regression tests after audit. | Tests should encode stable invariants discovered by audit, not guess before research. |
| Language parity | `README.md` and `README_CN.md` must be semantically aligned, not literal translations. | The Chinese README can use natural Chinese phrasing while preserving the same architecture claims. |

## Contracts And Data Shapes

### Audit Report Contract

`development_plans/reference/documentation_harmonization_audit_report.md` must
use this structure:

```md
# documentation harmonization audit report

## Summary

- Audit date:
- Scope:
- Source-of-truth inputs:
- High-severity findings:
- Required bigbang edit groups:
- Deferred follow-up findings:

## Document Inventory
## Module Accuracy Findings
## Interface Detail Findings
## Subagent Interface Findings
## Top-Level And HOWTO Findings
## Bilingual Parity Findings
## Deferred Follow-Up Findings
```

The report must cite concrete source paths, test paths, symbols, or registry
rows as evidence. It must not authorize production-code changes.

### Documentation Inventory Row

Each inventory row must classify:

```python
{
    "path": str,
    "role": str,
    "living_or_historical": "living" | "historical",
    "owner": str,
    "source_of_truth": list[str],
    "action": "harmonize" | "audit_only" | "registry_only" | "defer",
}
```

### Module README Section Contract

Living module READMEs must use the relevant subset of this vocabulary:

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

Small modules may omit sections that do not apply, but the audit report must
record why a high-risk section such as public contract, testing, or forbidden
paths is unnecessary.

### Subagent Documentation Contract

Each subagent or worker family must document:

- family name;
- owning package;
- runtime purpose;
- registry or discovery mechanism;
- identifier field such as `name`, `SOURCE`, `SUBAGENT`, or `WORKER`;
- prompt-facing description field;
- supported actions or task kinds;
- input contract;
- output contract;
- validation owner;
- availability or enablement rule;
- cache behavior;
- trace or audit behavior;
- refusal conditions;
- side-effect boundary;
- required tests.

The documentation must explicitly say that these are documentation categories,
not a required shared runtime base class.

### Bilingual Parity Contract

For `README.md` and `README_CN.md`, parity is semantic:

- the same runtime layers must appear in both documents;
- the same LLM route families must be represented in both documents;
- the same startup path must be represented in both documents;
- the same project status must be represented in both documents;
- Chinese prose may be more compact, but it must not omit a current major
  subsystem named in the English README.

### HOWTO Order Contract

`docs/HOWTO.md` must preserve this operator order:

1. install package and dependencies;
2. configure required environment variables;
3. load a character profile;
4. start the control console for normal local operation;
5. use the console or direct commands to start the brain and adapters;
6. run smoke, ops, and test commands.

## LLM Call And Context Budget

This plan does not add, remove, or change runtime LLM calls, prompts, model
routes, graph nodes, prompt payloads, or context budgets.

Documentation may describe existing LLM routes and prompt boundaries only when
the audit report cites current source or tests. If execution discovers that a
prompt or LLM contract is wrong in code, record a follow-up finding and do not
change runtime prompt code in this plan.

## Change Surface

### Create

- `development_plans/active/short_term/documentation_harmonization_bigbang_plan.md`
  - This draft plan.
- `development_plans/reference/documentation_harmonization_audit_report.md`
  - First execution artifact and audit gate.
- `docs/DOCUMENTATION_GUIDE.md`
  - Living documentation role and module README harmonization guide.
- `docs/SUBAGENT_INTERFACES.md`
  - Cross-family subagent and worker documentation guide.
- `tests/test_documentation_harmonization.py`
  - Focused doc-regression tests after the audit report identifies stable
    invariants.

### Modify

- `development_plans/README.md`: add this plan's registry row and later
  lifecycle status only.
- `README.md`, `README_CN.md`, `docs/HOWTO.md`: reconcile top-level
  architecture, bilingual parity, setup order, env, adapter, ops, endpoint,
  testing, route, and doc-index claims.
- `AGENTS.md`: edit only for factual project-process drift proved by the audit
  report. Do not change agent policy semantics without explicit user approval.
- Living module and package READMEs from Stage 1 inventory:
  `src/adapters/README.md`, `src/adapters/napcat_qq_adapter/README.md`,
  `src/control_console/README.md`, `src/scripts/README.md`,
  `src/kazusa_ai_chatbot/accepted_task/README.md`,
  `src/kazusa_ai_chatbot/action_spec/README.md`,
  `src/kazusa_ai_chatbot/background_artifact/README.md`,
  `src/kazusa_ai_chatbot/background_work/README.md`,
  `src/kazusa_ai_chatbot/brain_service/README.md`,
  `src/kazusa_ai_chatbot/calendar_scheduler/README.md`,
  `src/kazusa_ai_chatbot/cognition_chain_core/README.md`,
  `src/kazusa_ai_chatbot/cognition_resolver/README.md`,
  `src/kazusa_ai_chatbot/complex_task_resolver/README.md`,
  `src/kazusa_ai_chatbot/consolidation/README.md`,
  `src/kazusa_ai_chatbot/conversation_progress/README.md`,
  `src/kazusa_ai_chatbot/db/README.md`,
  `src/kazusa_ai_chatbot/dispatcher/README.md`,
  `src/kazusa_ai_chatbot/event_logging/README.md`,
  `src/kazusa_ai_chatbot/global_character_growth/README.md`,
  `src/kazusa_ai_chatbot/internal_monologue_residue/README.md`,
  `src/kazusa_ai_chatbot/llm_interface/README.md`,
  `src/kazusa_ai_chatbot/llm_tracing/README.md`,
  `src/kazusa_ai_chatbot/memory_evolution/README.md`,
  `src/kazusa_ai_chatbot/message_envelope/README.md`,
  `src/kazusa_ai_chatbot/nodes/README.md`,
  `src/kazusa_ai_chatbot/past_dialog_cognition/README.md`,
  `src/kazusa_ai_chatbot/proactive_output/README.md`,
  `src/kazusa_ai_chatbot/rag/README.md`,
  `src/kazusa_ai_chatbot/rag/conversation_evidence/README.md`,
  `src/kazusa_ai_chatbot/rag/live_context/README.md`,
  `src/kazusa_ai_chatbot/rag/memory_evidence/README.md`,
  `src/kazusa_ai_chatbot/rag/person_context/README.md`,
  `src/kazusa_ai_chatbot/rag/recall/README.md`,
  `src/kazusa_ai_chatbot/rag/web_agent3/README.md`,
  `src/kazusa_ai_chatbot/reflection_cycle/README.md`,
  `src/kazusa_ai_chatbot/self_cognition/README.md`,
  `tests/control_console_e2e/README.md`.

### Keep

- `development_plans/archive/**`
  - Historical records. Audit only unless the registry points to the wrong
    lifecycle state or a supersession banner is objectively wrong.
- Existing unrelated active plans under `development_plans/active/**`
  - Do not edit except this plan and its registry row.
- Production files under `src/**/*.py`
  - No production-code changes.
- `.env`
  - Never read.

## Overdesign Guardrail

- Actual problem: Kazusa's documentation has grown across many modules and
  plan eras, and the user needs a reliable, code-backed harmonization pass
  before further agents rely on it.
- Minimal change: produce an audit report, then update living Markdown docs and
  doc-regression tests. Do not change runtime behavior.
- Ownership boundaries: code and tests own implemented behavior; module ICDs
  own module contracts; top-level docs own summaries; HOWTO owns operator
  order; deterministic tests own stable documentation invariants; LLM prompts
  and runtime code stay unchanged.
- Rejected complexity: no shared runtime subagent abstraction, no doc
  generation framework, no external Markdown tooling dependency, no new
  feature flags, no compatibility vocabulary, no production-code changes, no
  archive-wide rewriting, no screenshots, no website, and no database
  migration.
- Evidence threshold: add runtime abstractions, production fixes, generated
  doc tooling, or archive migrations only after a separate approved plan cites
  concrete audit findings that cannot be solved by documentation edits and
  doc-regression tests.

## Agent Autonomy Boundaries

- The responsible agent may choose local wording only when it preserves the
  contracts in this plan and the audit report.
- The responsible agent must not introduce new architecture, alternate
  migration strategies, compatibility layers, fallback paths, or extra
  features.
- The responsible agent must treat edits outside living docs and doc tests as
  high-scrutiny changes. Updating an existing file outside the change surface
  requires user approval before implementation.
- The responsible agent must not modify production code, prompts, configs,
  graph behavior, adapter behavior, persistence, scheduler logic, or runtime
  startup.
- The responsible agent must not rewrite completed or superseded historical
  plans for style or consistency.
- If the audit report and code disagree, preserve the code-backed behavior in
  docs and record a follow-up finding for any suspected code issue.
- If the plan and code disagree, preserve this plan's stated intent and report
  the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead
  of inventing a substitute.

## Implementation Order

### Stage 1 - Audit Report Gate

Parent-owned. No subagent edits existing docs or code in this stage.

1. Run `git status --short` and record unrelated dirty files in the audit
   report.
2. Inventory Markdown files with `rg --files -g '*.md'`.
3. Extract heading structure from living module READMEs.
4. Read source/test evidence for public interfaces, config variables, startup
   order, subagent registries, and high-risk architecture boundaries.
5. Create
   `development_plans/reference/documentation_harmonization_audit_report.md`
   using the audit report contract.
6. Record high-severity findings and exact required edit groups.
7. Sign off the audit checkpoint before changing existing docs or tests.

### Stage 2 - Documentation Test Contract

Parent-owned. Starts only after Stage 1 is signed off.

1. Create `tests/test_documentation_harmonization.py`.
2. Add tests for stable invariants from the audit report, including:
   - required module README section vocabulary for high-risk ICDs;
   - subagent interface guide presence and family coverage;
   - README/README_CN major capability parity;
   - HOWTO startup order;
   - route-list parity between HOWTO and top-level docs where stable.
3. Run the focused test file and record the expected failures or baseline
   before documentation edits.

### Stage 3 - Bigbang Living Documentation Edit

Documentation execution subagent-owned. Starts only after Stages 1 and 2 are
signed off.

1. Read this plan, the audit report, and the failing/baseline doc tests.
2. Create `docs/DOCUMENTATION_GUIDE.md`.
3. Create `docs/SUBAGENT_INTERFACES.md`.
4. Update module READMEs in the approved change surface to the harmonized
   section vocabulary and code-backed contracts.
5. Update `README.md`, `README_CN.md`, and `docs/HOWTO.md` for top-level,
   bilingual, and runbook consistency.
6. Avoid production-code edits and historical-plan rewrites.
7. Report changed files, findings handled, skipped/deferred findings, and
   commands run before closing.

### Stage 4 - Parent Verification And Registry

Parent-owned.

1. Re-run focused doc-regression tests.
2. Run existing doc-sensitive tests listed in `Verification`.
3. Run static greps and link/path sanity checks listed in `Verification`.
4. Update `development_plans/README.md` lifecycle status only when the plan is
   genuinely ready for that lifecycle transition.
5. Record all evidence in `Execution Evidence`.

### Stage 5 - Independent Review And Remediation

Parent-owned with one independent review subagent.

1. Start the independent review subagent after verification passes.
2. Review the plan, audit report, full documentation diff, doc tests, static
   checks, and execution evidence.
3. Fix review findings only when they are inside this plan's change surface.
4. Re-run affected verification.
5. Record review outcome and residual risks before final sign-off.

## Execution Model

- Parent agent owns orchestration, Stage 1 audit, test code, verification,
  execution evidence, review feedback remediation, lifecycle updates, and final
  sign-off.
- Stage 1 audit is parent-only and must complete before any existing document,
  test, or source file is edited.
- Parent agent establishes the focused documentation test contract after the
  audit report is signed off.
- Documentation execution subagent: exactly one native subagent, started after
  the audit report and focused documentation test contract are established;
  owns living documentation edits only; does not edit tests, production code,
  prompts, configs, registry lifecycle rows, or historical plans unless the
  parent explicitly directs an in-scope documentation edit.
- Parent agent may continue verification, static checks, and execution
  evidence while the documentation execution subagent edits living docs.
- Independent review subagent: exactly one native subagent, started after
  planned verification passes; reviews the plan, audit report, diff, and
  evidence; reports findings to the parent; does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless
  the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1 - audit report complete before existing docs or code are touched
  - Covers: Implementation Order Stage 1.
  - Files allowed before sign-off:
    `development_plans/reference/documentation_harmonization_audit_report.md`
    only.
  - Verify: audit report exists, contains the required sections, cites
    source-of-truth evidence, and names exact edit groups.
  - Evidence: record inventory command, source/test evidence sampled, and
    high-severity finding counts in `Execution Evidence`.
  - Handoff: after sign-off, parent starts Stage 2.
  - Sign-off: 2026-07-02 parent fallback execution after audit report
    section verification.
- [x] Stage 2 - focused documentation test contract established
  - Covers: Implementation Order Stage 2.
  - Files: `tests/test_documentation_harmonization.py`.
  - Verify:
    `venv\Scripts\python.exe -m pytest tests\test_documentation_harmonization.py -q`.
  - Evidence: record expected failures or baseline output before Stage 3.
  - Handoff: documentation execution subagent starts Stage 3.
  - Sign-off: 2026-07-02 parent fallback execution after focused baseline
    failure capture.
- [x] Stage 3 - living documentation harmonized in one bigbang pass
  - Covers: Implementation Order Stage 3.
  - Files: living docs in `Change Surface`.
  - Verify: documentation execution subagent reports changed files and confirms
    no production code or historical plan body edits.
  - Evidence: record changed docs and deferred findings.
  - Handoff: parent starts Stage 4 verification.
  - Sign-off: 2026-07-02 parent fallback execution after focused doc tests
    passed and production-code diff check returned no files.
- [x] Stage 4 - verification and registry complete
  - Covers: Implementation Order Stage 4.
  - Files: doc tests and `development_plans/README.md` lifecycle row if
    lifecycle status changes.
  - Verify: run every command in `Verification`.
  - Evidence: record command outputs and allowed grep exceptions.
  - Handoff: independent review subagent starts Stage 5.
  - Sign-off: 2026-07-02 parent fallback execution after focused tests,
    existing doc-sensitive tests, static checks, scope checks, and allowed grep
    exceptions were recorded. Registry remains `in_progress` until Stage 5
    review is complete.
- [x] Stage 5 - independent review complete
  - Covers: Implementation Order Stage 5 and `Independent Code Review`.
  - Verify: parent-owned fallback review approves or all in-scope findings are
    fixed and affected checks rerun.
  - Evidence: record reviewer identity or harness role, findings, fixes,
    rerun commands, residual risks, and approval status.
  - Handoff: parent may mark plan completed only after this checkpoint is
    signed off.
  - Sign-off: 2026-07-02 parent-owned fallback review after diff, source
    spot-check, scope, link sanity, and verification evidence review. No
    blocking findings.

## Verification

### Focused Documentation Tests

- `venv\Scripts\python.exe -m pytest tests\test_documentation_harmonization.py -q`
  - Expected after Stage 3: pass.

### Existing Documentation-Sensitive Tests

- `venv\Scripts\python.exe -m pytest tests\test_self_cognition_architecture_docs.py -q`
  - Expected: pass.
- `venv\Scripts\python.exe -m pytest tests\test_internal_monologue_residue_prompt_boundaries.py -q`
  - Expected: pass.
- `venv\Scripts\python.exe -m pytest tests\test_adapter_envelope_normalizers.py -q`
  - Expected: pass.
- `venv\Scripts\python.exe -m pytest tests\test_background_artifact_runtime.py -q`
  - Expected: pass.
- `venv\Scripts\python.exe -m pytest tests\test_reflection_cycle_stage1c_integration.py -q`
  - Expected: pass.

### Static Checks

- `git diff --check`
  - Expected: no whitespace errors in this plan's changed files.
- `rg --files -g '*.md'`
  - Expected: includes the new audit report, `docs/DOCUMENTATION_GUIDE.md`,
    and `docs/SUBAGENT_INTERFACES.md` after implementation.
- `rg -n "T[O]DO|T[B]D" README.md README_CN.md docs src -g "*.md"`
  - Expected: no matches in files modified by this plan. Existing matches in
    unmodified historical or reference docs do not block this plan.
- `rg -n "LLM_BASE_URL|LLM_API_KEY|LLM_MODEL" README.md README_CN.md docs src -g "*.md"`
  - Expected: any matches must describe retired generic route variables or
    explicitly documented legacy aliases. Matches that present generic LLM
    variables as current required config are failures.
- `rg -n "web_evidence|rag_evidence" README.md README_CN.md docs src -g "*.md"`
  - Expected: matches are allowed only in historical explanation or explicit
    deprecation/cutover wording. Matches that present these as canonical
    current L2d-visible capability names are failures.

### Registry And Scope Checks

- `git status --short`
  - Expected: changed files are limited to this plan, the audit report, docs
    and tests in the approved change surface, and unrelated pre-existing dirty
    files recorded in `Execution Evidence`.
- `git diff --name-only -- src/**/*.py`
  - Expected: no production Python files are changed by this plan. Pre-existing
    dirty production files from other work must be listed separately and not
    attributed to this plan.

## Independent Plan Review

Run this gate before changing `Status` from `draft` to `approved` or
`in_progress`. Prefer a reviewer that did not draft the plan. If no separate
reviewer is available, the drafting agent must reread `development-plan`, the
plan contract, execution gates, this plan, `README.md`, `docs/HOWTO.md`,
`development_plans/README.md`, and the sampled module docs from a fresh-review
posture.

Review scope:

- The audit report gate truly prevents existing document/code edits before
  research is complete.
- The plan separates living docs from historical development records.
- The subagent-interface harmonization remains documentation-only.
- The change surface excludes production code and unrelated active plans.
- Verification commands are exact enough to run.
- Agent creativity is bounded by the audit report, change surface, and
  forbidden paths.

Record blockers, non-blocking findings, required edits, and approval status.
Approve only when blockers are resolved.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off.
The parent agent must create one independent code-review subagent through the
current harness's native subagent capability. If native subagents are
unavailable, stop unless the user explicitly approves fallback execution.

Review scope:

- Project rules and style compliance for every changed documentation, test,
  registry, and command artifact.
- Alignment with the audit report, `Must Do`, `Deferred`, `Change Surface`,
  `Cutover Policy`, implementation order, verification gates, and acceptance
  criteria.
- Documentation quality and architecture risk, including stale source claims,
  module ownership drift, hidden compatibility vocabulary, subagent-interface
  over-abstraction, bilingual mismatch, runbook order errors, broken links,
  and avoidable historical-plan churn.
- Regression quality, including focused doc tests, existing doc-sensitive
  tests, static grep expectations, and dirty-worktree attribution.

The parent agent fixes concrete findings directly only when the fix is inside
the approved change surface. If a fix requires production code, runtime
behavior changes, prompt changes, a new public contract, or historical plan
rewrites outside policy, stop and update the plan or request approval before
changing files.

Record findings, fixes, commands rerun, residual risks, and approval status in
`Execution Evidence`.

## Acceptance Criteria

This plan is complete when:

- Stage 1 audit report exists and was created before edits to existing docs or
  code.
- `docs/DOCUMENTATION_GUIDE.md` exists and defines living-document roles,
  source-of-truth hierarchy, module README expectations, bilingual parity, and
  historical-plan policy.
- `docs/SUBAGENT_INTERFACES.md` exists and covers RAG helper agents,
  `web_agent3` source subagents, `complex_task_resolver` subagents, and
  `background_work` workers.
- Living module READMEs in the approved change surface are accurate against
  code/tests cited by the audit report.
- Top-level `README.md`, `README_CN.md`, and `docs/HOWTO.md` are consistent
  with module ICDs and current startup/config/runtime code.
- The HOWTO setup and initialization order is correct.
- English and Chinese top-level docs have semantic parity for current major
  capabilities and runtime layers.
- Focused doc-regression tests and existing doc-sensitive tests listed in
  `Verification` pass.
- Static checks listed in `Verification` pass or have recorded allowed
  exceptions.
- No production Python files are changed by this plan.
- Independent review is complete and all blocking findings are resolved or
  recorded as explicitly deferred follow-up work outside this plan.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Audit scope becomes too broad to execute safely | Classify historical docs as audit-only and limit bigbang edits to living docs. | Audit report inventory and change-surface check. |
| Subagent harmonization turns into runtime abstraction | State documentation-only contract and forbid shared runtime wrappers. | Review `docs/SUBAGENT_INTERFACES.md` and `git diff --name-only -- src/**/*.py`. |
| Bilingual README drifts again | Add focused parity tests for major capability and route topics. | `tests/test_documentation_harmonization.py`. |
| HOWTO startup order becomes stale | Encode the runbook order contract in tests. | Focused HOWTO order test. |
| Existing dirty worktree is misattributed | Record pre-existing dirty files at Stage 1 and compare final changed files. | `git status --short` evidence before and after. |
| Historical plan records are rewritten | Mark archive docs no-op and review changed paths. | `git diff --name-only` and independent review. |

## Execution Evidence

- 2026-07-02: User explicitly requested execution without subagents. Native
  subagent execution and independent review subagent gates are replaced by
  parent-owned fallback execution for this run only.
- 2026-07-02: Inline pre-execution review reread
  `development_plans/README.md`, this plan, `README.md`, `README_CN.md`,
  `docs/HOWTO.md`, the development-plan skill, execution gates reference, and
  sampled `brain_service`, `rag`, and `complex_task_resolver` module ICDs.
  No lifecycle blocker was found for moving the plan to `in_progress` under
  the user-approved fallback path.
- 2026-07-02 Stage 1: Created
  `development_plans/reference/documentation_harmonization_audit_report.md` as
  the first new execution artifact. Verification commands:
  `Test-Path -LiteralPath 'development_plans\reference\documentation_harmonization_audit_report.md'`
  returned `True`; required-section grep found `Summary`,
  `Document Inventory`, `Module Accuracy Findings`,
  `Interface Detail Findings`, `Subagent Interface Findings`,
  `Top-Level And HOWTO Findings`, `Bilingual Parity Findings`, and
  `Deferred Follow-Up Findings`; evidence grep found current source references
  including `README_CN.md`, `service.py`, `config.py`,
  `brain_model_routes.py`, `BaseRAGHelperAgent`, and
  `ComplexTaskSubagentV1`.
- 2026-07-02 Stage 2: Created
  `tests/test_documentation_harmonization.py` after Stage 1 sign-off. Baseline
  command
  `venv\Scripts\python.exe -m pytest tests\test_documentation_harmonization.py -q`
  failed as expected with 6 failed and 1 passed. Expected failures cover
  missing `docs/DOCUMENTATION_GUIDE.md`, missing
  `docs/SUBAGENT_INTERFACES.md`, stale `README_CN.md` route/startup parity,
  missing top-level runtime subsystem links, stale HOWTO startup order, and
  compact module READMEs missing required ICD sections. The audit-boundary
  test passed.
- 2026-07-02 Stage 3: Created `docs/DOCUMENTATION_GUIDE.md` and
  `docs/SUBAGENT_INTERFACES.md`; updated `README.md`, `README_CN.md`,
  `docs/HOWTO.md`, `src/kazusa_ai_chatbot/accepted_task/README.md`,
  `src/kazusa_ai_chatbot/cognition_chain_core/README.md`,
  `src/kazusa_ai_chatbot/llm_tracing/README.md`, and
  `src/scripts/README.md`. Focused command
  `venv\Scripts\python.exe -m pytest tests\test_documentation_harmonization.py -q`
  passed with 7 passed. `git diff --name-only -- 'src/**/*.py'` returned no
  files, confirming no production Python edits.
- 2026-07-02 Stage 4: Verification passed.
  `venv\Scripts\python.exe -m pytest tests\test_documentation_harmonization.py -q`
  passed with 7 passed;
  `tests\test_self_cognition_architecture_docs.py -q` passed with 5 passed;
  `tests\test_internal_monologue_residue_prompt_boundaries.py -q` passed with
  4 passed; `tests\test_adapter_envelope_normalizers.py -q` passed with
  20 passed; `tests\test_background_artifact_runtime.py -q` passed with
  3 passed; `tests\test_reflection_cycle_stage1c_integration.py -q` passed
  with 5 passed. `git diff --check` exited 0 with CRLF conversion warnings
  only. `rg --files -g '*.md'` included the new audit report,
  `docs\DOCUMENTATION_GUIDE.md`, and `docs\SUBAGENT_INTERFACES.md`.
  `rg -n "T[O]DO|T[B]D" README.md README_CN.md docs src -g "*.md"` exited
  1 with no matches after rewording an existing `llm_interface` future-provider
  note. `rg -n "LLM_BASE_URL|LLM_API_KEY|LLM_MODEL" ...` returned only
  route-specific environment examples, the HOWTO retired-generic-variable
  explanation, and route-specific reflection-cycle entries; all are allowed by
  the static-check policy. `rg -n "web_evidence|rag_evidence" ...` returned
  `run_rag_evidence_for_persona_state(...)` function-name references and
  explicit complex-task cutover wording for former capability names; no match
  presents `web_evidence` or `rag_evidence` as current canonical L2d-visible
  capability names. `git diff --name-only -- 'src/**/*.py'` returned no files.
  `git status --short` showed only documentation, plan, and focused test files
  in this plan's change surface.
- 2026-07-02 Stage 5: Parent-owned fallback review completed because the user
  explicitly requested no subagent. Reviewer role: parent harness fallback.
  Reviewed this plan, the audit report, new documentation guides, focused
  tests, tracked Markdown diff, untracked artifacts, Stage 4 verification
  evidence, source spot checks for service startup order, route families, and
  subagent fields, and a local Markdown-link sanity check. Review commands
  included `git status --short`, `git diff --stat`, `git diff --name-only`,
  `git diff --name-only -- 'src/**/*.py'`, targeted `git diff`, targeted `rg`
  source checks, and a read-only local-link check that returned
  `local markdown links ok`. Findings: no blocking issues. Fixes: none.
  Residual risks: deeper line-by-line module audits beyond the high-confidence
  findings remain deferred as recorded in the audit report; CRLF conversion
  warnings are non-blocking working-tree warnings. Approval status: approved
  for completion under the user-approved no-subagent fallback path.
- 2026-07-02 final verification after lifecycle edits:
  `venv\Scripts\python.exe -m pytest tests\test_documentation_harmonization.py -q`
  passed with 7 passed; `git diff --check` exited 0 with CRLF conversion
  warnings only; `git diff --name-only -- 'src/**/*.py'` returned no files.
  Registry row and plan status are `completed`.
- 2026-07-02 closure: user requested closing the plan and committing changes.
  The completed execution record is ready to move from `active/short_term/` to
  `archive/completed/short_term/`; the registry will point to the archived
  record after the move.
