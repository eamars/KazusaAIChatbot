# documentation harmonization audit report

## Summary

- Audit date: 2026-07-02
- Scope: every Markdown file returned by `rg --files -g '*.md'`, with living
  docs reviewed for harmonization and historical plan bodies classified as
  audit-only evidence.
- Source-of-truth inputs:
  - `git status --short`
  - `rg --files -g '*.md'`
  - `rg -n "^#{1,6} " README.md README_CN.md docs src -g '*.md'`
  - `README.md`, `README_CN.md`, `docs/HOWTO.md`
  - `development_plans/README.md`
  - `src/kazusa_ai_chatbot/config.py`
  - `src/control_console/brain_model_routes.py`
  - `src/kazusa_ai_chatbot/service.py`
  - `src/kazusa_ai_chatbot/rag/helper_agent.py`
  - `src/kazusa_ai_chatbot/rag/web_agent3/subagent/__init__.py`
  - `src/kazusa_ai_chatbot/complex_task_resolver/contracts.py`
  - `src/kazusa_ai_chatbot/complex_task_resolver/subagent/__init__.py`
  - `src/kazusa_ai_chatbot/background_work/subagent/__init__.py`
  - doc-sensitive tests found by
    `rg -n "README|HOWTO|ICD|Documentation|SUBAGENT|documentation" tests -g '*.py'`
- High-severity findings:
  - `README_CN.md` is stale against `README.md`, `docs/HOWTO.md`,
    `config.py`, and `brain_model_routes.py` for route names, normal
    control-console startup, repository map, and newer accepted-task/background
    work wording.
  - `docs/HOWTO.md` startup order does not match `service.py` lifespan order:
    it omits persistent media descriptor cache hydration, chat worker/adapter
    registry startup, and background-work runtime startup, and places Cache2
    hydration later than the code.
  - Subagent interface documentation is split across RAG, web_agent3,
    complex-task, and background-work ICDs, with no single guide that explains
    the shared documentation vocabulary and the fact that the vocab is not a
    shared runtime base class.
  - Module README format is uneven. High-risk ICDs are detailed, but compact
    READMEs such as `llm_tracing`, `accepted_task`, `cognition_chain_core`,
    and `src/scripts` lack a consistent document-control, public-interface,
    failure-behavior, testing, and forbidden-path shape.
  - Top-level documentation index is incomplete for the current living module
    set and does not include a documentation guide or subagent-interface guide.
- Required bigbang edit groups:
  - Create `docs/DOCUMENTATION_GUIDE.md`.
  - Create `docs/SUBAGENT_INTERFACES.md`.
  - Update `README.md` and `README_CN.md` as paired top-level docs.
  - Update `docs/HOWTO.md` startup order and current notes.
  - Harmonize selected compact module READMEs whose interface detail is
    insufficient for future agents.
  - Add focused doc-regression tests for the stable invariants above.
- Deferred follow-up findings:
  - Do not edit production Python code in this plan.
  - Do not add a shared runtime subagent abstraction.
  - Do not rewrite archived completed or superseded plan bodies.
  - Do not resolve suspected runtime bugs through documentation-only edits.
  - Do not treat historical `docs/superpowers/**` artifacts as current project
    runbooks without a separate lifecycle decision.

Pre-existing dirty files at Stage 1 start, after lifecycle setup for this
approved execution:

```text
 M development_plans/README.md
?? development_plans/active/short_term/documentation_harmonization_bigbang_plan.md
```

The lifecycle status edits above are plan-control edits required to execute
the user-approved fallback path. The first new execution artifact is this
audit report.

## Document Inventory

The inventory command returned Markdown under root docs, source package docs,
active/reference/archived development plans, and superpowers design artifacts.
This table classifies every returned file by explicit path or directory rule.

| Path or pattern | Role | Living or historical | Owner | Source of truth | Action |
|---|---|---|---|---|---|
| `README.md` | top-level overview | living | project | module ICDs, HOWTO, source config | harmonize |
| `README_CN.md` | top-level overview | living | project | `README.md`, HOWTO, module ICDs, source config | harmonize |
| `AGENTS.md` | agent instruction | living | project | project process policy | audit_only |
| `docs/HOWTO.md` | runbook | living | project operations | `pyproject.toml`, `service.py`, config, adapters | harmonize |
| `docs/DOCUMENTATION_GUIDE.md` | documentation guide | living | project docs | this audit and plan | harmonize |
| `docs/SUBAGENT_INTERFACES.md` | subagent interface guide | living | project docs | subagent registries and ICDs | harmonize |
| `docs/superpowers/specs/*.md` | test docs/design artifact | historical | frontend workflow | local design history | audit_only |
| `docs/superpowers/plans/*.md` | test docs/plan artifact | historical | frontend workflow | local design history | audit_only |
| `tests/control_console_e2e/README.md` | test docs | living | control console tests | e2e test harness | audit_only |
| `src/adapters/README.md` | module ICD | living | adapters | adapter code and tests | audit_only |
| `src/adapters/napcat_qq_adapter/README.md` | module ICD | living | NapCat adapter | adapter code and tests | audit_only |
| `src/control_console/README.md` | module ICD | living | control console | console routes, auth, service registry | audit_only |
| `src/scripts/README.md` | package guide | living | scripts | script modules and entry points | harmonize |
| `src/kazusa_ai_chatbot/accepted_task/README.md` | module ICD | living | accepted task | accepted-task code, action-spec docs | harmonize |
| `src/kazusa_ai_chatbot/action_spec/README.md` | module ICD | living | action spec | contracts and action execution tests | audit_only |
| `src/kazusa_ai_chatbot/background_artifact/README.md` | compatibility ICD | living | background artifact | legacy compatibility code/tests | audit_only |
| `src/kazusa_ai_chatbot/background_work/README.md` | module ICD | living | background work | worker registry, queue/runtime code | audit_only |
| `src/kazusa_ai_chatbot/brain_service/README.md` | module ICD | living | brain service | FastAPI routes and contracts | audit_only |
| `src/kazusa_ai_chatbot/calendar_scheduler/README.md` | module ICD | living | calendar scheduler | scheduler models/repository/worker | audit_only |
| `src/kazusa_ai_chatbot/cognition_chain_core/README.md` | module ICD | living | cognition chain core | contracts and stage code | harmonize |
| `src/kazusa_ai_chatbot/cognition_resolver/README.md` | module ICD | living | cognition resolver | resolver contracts/service/tests | audit_only |
| `src/kazusa_ai_chatbot/complex_task_resolver/README.md` | module ICD | living | complex task resolver | contracts/service/subagents/tests | audit_only |
| `src/kazusa_ai_chatbot/consolidation/README.md` | module ICD | living | consolidation | consolidation package/tests | audit_only |
| `src/kazusa_ai_chatbot/conversation_progress/README.md` | module ICD | living | conversation progress | progress code/tests | audit_only |
| `src/kazusa_ai_chatbot/db/README.md` | module ICD | living | database | DB facade, collection owners, tests | audit_only |
| `src/kazusa_ai_chatbot/dispatcher/README.md` | module ICD | living | dispatcher | dispatcher code/tests | audit_only |
| `src/kazusa_ai_chatbot/event_logging/README.md` | module ICD | living | event logging | event logging contracts/tests | audit_only |
| `src/kazusa_ai_chatbot/global_character_growth/README.md` | module ICD | living | global character growth | growth code/tests | audit_only |
| `src/kazusa_ai_chatbot/internal_monologue_residue/README.md` | module ICD | living | residue | residue code/tests | audit_only |
| `src/kazusa_ai_chatbot/llm_interface/README.md` | module ICD | living | LLM interface | LLInterface code/tests | audit_only |
| `src/kazusa_ai_chatbot/llm_tracing/README.md` | module ICD | living | LLM tracing | trace export scripts, DB docs | harmonize |
| `src/kazusa_ai_chatbot/memory_evolution/README.md` | module ICD | living | memory evolution | memory evolution code/tests | audit_only |
| `src/kazusa_ai_chatbot/message_envelope/README.md` | module ICD | living | message envelope | message envelope code/tests | audit_only |
| `src/kazusa_ai_chatbot/nodes/README.md` | module ICD | living | cognition nodes | persona/cognition/dialog code/tests | audit_only |
| `src/kazusa_ai_chatbot/past_dialog_cognition/README.md` | module ICD | living | past dialog cognition | trace/residual code/tests | audit_only |
| `src/kazusa_ai_chatbot/proactive_output/README.md` | module ICD | living | proactive output | proactive output code/tests | audit_only |
| `src/kazusa_ai_chatbot/rag/README.md` | module ICD | living | RAG 2 | RAG supervisor/helper packages/tests | audit_only |
| `src/kazusa_ai_chatbot/rag/conversation_evidence/README.md` | module ICD | living | RAG conversation evidence | conversation evidence agents/tests | audit_only |
| `src/kazusa_ai_chatbot/rag/live_context/README.md` | module ICD | living | RAG live context | live context code/tests | audit_only |
| `src/kazusa_ai_chatbot/rag/memory_evidence/README.md` | module ICD | living | RAG memory evidence | memory evidence agents/tests | audit_only |
| `src/kazusa_ai_chatbot/rag/person_context/README.md` | module ICD | living | RAG person context | person context agents/tests | audit_only |
| `src/kazusa_ai_chatbot/rag/recall/README.md` | module ICD | living | RAG recall | recall code/tests | audit_only |
| `src/kazusa_ai_chatbot/rag/web_agent3/README.md` | module ICD | living | web_agent3 | source subagents and web tests | audit_only |
| `src/kazusa_ai_chatbot/reflection_cycle/README.md` | module ICD | living | reflection cycle | reflection code/tests | audit_only |
| `src/kazusa_ai_chatbot/self_cognition/README.md` | module ICD | living | self-cognition | self-cognition code/tests | audit_only |
| `development_plans/README.md` | active plan registry | living | development plans | registry lifecycle policy | registry_only |
| `development_plans/long_term/*.md` | roadmap | living | development plans | roadmap owner | audit_only |
| `development_plans/active/**/*.md` | active plan | living | development plans | individual plan status | registry_only |
| `development_plans/reference/**/*.md` | reference design | historical/reference | development plans | design history | audit_only |
| `development_plans/archive/completed/**/*.md` | historical completed plan | historical | development plans | execution history | audit_only |
| `development_plans/archive/superseded/**/*.md` | superseded plan | historical | development plans | supersession history | audit_only |

## Module Accuracy Findings

| ID | Finding | Evidence | Required action |
|---|---|---|---|
| MA-1 | The high-risk service, message, LLM, RAG, cognition-resolver, complex-task, background-work, database, event-logging, reflection, self-cognition, and adapter ICDs have generally code-backed boundaries. | Sampled `brain_service`, `rag`, `complex_task_resolver`, `background_work`, and source tests. | Preserve content; avoid broad rewrites. |
| MA-2 | `llm_tracing` accurately describes capture modes at a high level but is not shaped as an ICD. It lacks document control, public interfaces, storage contract, failure behavior, and testing contract. | `src/kazusa_ai_chatbot/llm_tracing/README.md`; `docs/HOWTO.md` trace-export commands; DB ICD logging collections. | Harmonize into compact ICD format. |
| MA-3 | `accepted_task` describes the correct boundary but lacks document control, public interface, persistence, failure behavior, testing contract, and forbidden paths. | `src/kazusa_ai_chatbot/accepted_task/README.md`; `README.md` accepted-task runtime section; `background_work` ICD. | Expand as an ICD without changing behavior. |
| MA-4 | `cognition_chain_core` has current source-label and public-entrypoint content but omits document control, testing, failure behavior, and explicit forbidden paths. | `src/kazusa_ai_chatbot/cognition_chain_core/README.md`; current contracts/stage docs. | Harmonize as a compact ICD. |
| MA-5 | `src/scripts/README.md` is useful but has a generated date of `2026-05-17`; it should not imply generated freshness unless a regeneration process is documented. | `src/scripts/README.md`; current HOWTO control-console startup. | Reword as a maintained package guide or refresh evidence. |
| MA-6 | Historical completed and superseded plans are intentionally stale in places and must remain historical evidence. | `development_plans/README.md` lifecycle contract. | No body rewrites. |

## Interface Detail Findings

| ID | Interface area | Evidence | Sufficiency | Required action |
|---|---|---|---|---|
| IF-1 | Brain service HTTP interface | `brain_service/README.md`; FastAPI route contracts | sufficient | Keep as source ICD. |
| IF-2 | Message envelope | `message_envelope/README.md`; adapter tests | sufficient | Keep as source ICD. |
| IF-3 | LLM route/config interface | `README.md`, HOWTO, `config.py`, `brain_model_routes.py` | English/HOWTO sufficient; Chinese stale | Update Chinese README and test parity. |
| IF-4 | Background work worker extension | `background_work/README.md`; `background_work/subagent/__init__.py` | sufficient locally | Cross-link from new subagent guide. |
| IF-5 | Accepted task lifecycle | `accepted_task/README.md`; action-spec/background-work docs | compact but under-specified | Add public interface, persistence, failure, testing, forbidden paths. |
| IF-6 | LLM tracing protected trace lane | `llm_tracing/README.md`; HOWTO export commands | under-specified | Add storage, capture mode, export, privacy, retention, and tests. |
| IF-7 | Scripts package | `src/scripts/README.md`; HOWTO command sections | registry only | Clarify package-guide role and no source-of-truth claims beyond listed commands. |

## Subagent Interface Findings

| Family | Owning package | Code-backed contract | Documentation issue | Harmonized documentation approach |
|---|---|---|---|---|
| RAG helper agents | `kazusa_ai_chatbot.rag` | `BaseRAGHelperAgent.run(task, context, max_attempts)` with standardized cache helpers. | RAG README documents helper-agent roles, but there is no cross-family vocabulary. | Document as "helper-agent class contract"; no shared base class beyond existing RAG base. |
| web_agent3 source subagents | `kazusa_ai_chatbot.rag.web_agent3.subagent` | Module fields `SOURCE`, `DESCRIPTION`, `SUPPORTED_ACTIONS`, optional `is_enabled()`, and `execute(decision)`. | web_agent3 ICD has source-subagent creation notes, but callers need a single summary with required fields and side-effect limits. | Document as "source module contract" with discovery, enablement, actions, and result policy. |
| Complex-task resolver subagents | `kazusa_ai_chatbot.complex_task_resolver.subagent` | Module fields `SUBAGENT`, `DESCRIPTION`, `SUPPORTED_ACTIONS`, `OWNED_NODE_KINDS`, `DEFAULT_ACTION`, optional `is_enabled()`, `create()`, and `ComplexTaskSubagentV1.run(...)`. | Complex-task ICD is detailed, but the field vocabulary is isolated. | Document as "resolver-local factory contract" with typed request/result validation. |
| Background-work workers | `kazusa_ai_chatbot.background_work.subagent` | Module fields `WORKER`, `DESCRIPTION`, and `execute(decision, max_output_chars=...)`. | Background-work ICD is adequate, but cross-family comparison is missing. | Document as "worker module contract" with result handoff and side-effect boundary. |

Better harmonization is documentation-only: define one table of categories
such as identifier, prompt description, supported actions, input contract,
output contract, enablement, validation owner, side effects, cache/audit, and
tests. Do not introduce a universal runtime interface, adapter bridge,
fallback mapper, alias vocabulary, or shared subagent base class.

## Top-Level And HOWTO Findings

| ID | Finding | Evidence | Required action |
|---|---|---|---|
| TL-1 | English README is mostly current for routes, runtime layers, accepted tasks, background work, and control-console startup. | `README.md`; `config.py`; `brain_model_routes.py`; sampled ICDs. | Update documentation index with new guides and missing module ICDs. |
| TL-2 | Chinese README omits `BOUNDARY_CORE_LLM`, `BACKGROUND_ARTIFACT_LLM`, and `BACKGROUND_WORK_LLM` from the route table. | `README_CN.md`; `README.md`; `config.py`; `brain_model_routes.py`. | Add the missing routes and match fallback semantics. |
| TL-3 | Chinese README quick start still starts `kazusa-brain` directly as the normal path. | `README_CN.md`; `README.md`; HOWTO control-console section. | Make `kazusa-control-console` the normal local run path and keep direct brain startup as development fallback. |
| TL-4 | Chinese README repository map omits `src/control_console/` and newer module families compared with the English README. | `README_CN.md`; `README.md`; source tree. | Update the repository map and runtime layers. |
| TL-5 | HOWTO startup sequence is stale against `service.py`. It omits RAG initializer cache/media descriptor cache ordering, chat worker/adapter registry startup, and background-work runtime startup. | `docs/HOWTO.md`; `service.py` lifespan. | Update the runbook startup order. |
| TL-6 | HOWTO "Current Notes" says the supported development run path is editable install plus `uvicorn`, which conflicts with the earlier normal local control-console path. | `docs/HOWTO.md`. | Reword notes to distinguish normal local operation from direct-service fallback. |

## Bilingual Parity Findings

| Topic | English README | Chinese README | Result | Required action |
|---|---|---|---|---|
| Route list | Includes `BOUNDARY_CORE_LLM`, `BACKGROUND_ARTIFACT_LLM`, `BACKGROUND_WORK_LLM`. | Omits those three route families. | mismatch | Add missing routes. |
| Normal startup | Load profile, start control console, direct brain only as fallback. | Load profile, run brain directly. | mismatch | Align quick start. |
| Control console | Runtime layer and repository map include `src/control_console`. | Runtime layer lacks control console; repository map omits it. | mismatch | Add control console layer/map entry. |
| Accepted task/background work | Explains accepted-task materialization and result-ready cognition. | Mentions commitments but omits the current accepted-task/background-work handoff detail. | partial mismatch | Add compact equivalent wording. |
| Documentation index | English is incomplete but has more current ICD links. | Also incomplete and missing any future guide links. | mismatch risk | Update both indexes together. |

## Deferred Follow-Up Findings

| ID | Finding | Reason deferred |
|---|---|---|
| DF-1 | Some module READMEs may have deeper stale details that require line-by-line code review beyond this documentation harmonization pass. | This pass can add guide/tests and fix high-confidence drift; suspected code defects or ambiguous module claims need separate plans. |
| DF-2 | Historical plan bodies contain obsolete terms, old route names, and old architecture snapshots. | They are historical records and must not be rewritten for style. |
| DF-3 | A universal subagent runtime abstraction could reduce documentation repetition but would be a production architecture change. | User requested documentation harmonization; the code has valid family-specific contracts. |
| DF-4 | `docs/superpowers/**` may need a lifecycle decision if these artifacts are still used by humans. | They are outside current Kazusa module/runbook docs and should not be silently promoted. |
| DF-5 | Any mismatch found in production code, prompts, startup behavior, model routing, or adapter delivery must be handled in a separate approved plan. | This plan explicitly forbids production-code changes. |
