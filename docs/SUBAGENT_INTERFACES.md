# Kazusa Subagent Interface Guide

## Document Control

- Owning area: project documentation
- Applies to: RAG3 local-context stage agents, retired RAG helper agents,
  `web_agent3` source subagents, `complex_task_resolver` resolver-local
  subagents, and `background_work` workers
- Source evidence: family-specific registries, protocols, and module ICDs
- Change policy: harmonize documentation categories, not runtime interfaces

## Purpose

Kazusa has several subagent and worker families. They are related because each
family gives a bounded specialist a task and receives a bounded result, but
they are not one runtime abstraction. This guide defines shared documentation
vocabulary so agents can compare the families without adding a universal base
class, registry bridge, alias layer, or compatibility adapter.

The categories below are documentation categories, not a shared runtime base
class and not a shared runtime base class requirement.

## Shared Documentation Vocabulary

Every family-specific ICD should describe the relevant fields below:

| Category | Meaning |
|---|---|
| Family name | Human-readable family, such as RAG helper agent or background-work worker. |
| Owning package | Python package that owns discovery, validation, and execution. |
| Runtime purpose | What problem the family solves in the current architecture. |
| Registry or discovery | Static registry, package discovery, or explicit module list. |
| Identifier | Stable family-local id such as `name`, `SOURCE`, `SUBAGENT`, or `WORKER`. |
| Prompt description | Prompt-safe capability text such as `DESCRIPTION`. |
| Supported actions | Explicit action names, node kinds, or task kinds. |
| Input contract | Typed object, dict shape, or semantic task text accepted by the family. |
| Output contract | Result envelope returned to the caller. |
| Validation owner | Deterministic code that validates requests and results. |
| Enablement | Optional `is_enabled()` or config gate, when the family has one. |
| Cache behavior | Whether the family or backing workers can cache results. |
| Trace or audit | What result, cache, provenance, or event data is kept for debugging. |
| Refusal conditions | Unsupported work, missing inputs, unavailable providers, or unsafe side effects. |
| Side-effect boundary | Whether the family may read storage, write storage, call tools, or deliver text. |
| Required tests | Deterministic, integration, real LLM, or live-service checks needed for changes. |

## RAG Helper Agents

RAG helper agents are retained source-level helper modules and historical
coverage. Production `local_context_recall` uses the RAG3 local-context
resolver described below.

| Field | Contract |
|---|---|
| Owning package | `kazusa_ai_chatbot.rag` |
| Existing base | `BaseRAGHelperAgent` |
| Runtime purpose | Retrieve bounded factual evidence for cognition. |
| Entry method | `run(task, context, max_attempts=3)` |
| Identifier | `name` constructor argument and dispatcher-visible agent names. |
| Input | Slot description or retrieval task plus runtime context and known facts. |
| Output | Dict containing resolution state, result payload, attempts, and cache metadata. |
| Validation owner | RAG supervisor, capability agents, worker tools, and projection boundary. |
| Cache behavior | Cache2 is available to helper agents; capability orchestrators may report uncached metadata. |
| Side effects | Retrieve and format evidence; do not decide persona stance or final wording. |
| Required tests | RAG helper tests, web_agent3 tests where delegated, prompt-facing sanitizer tests, and doc-sensitive tests for public boundaries. |

RAG helper agents may use different internal algorithms. Some use
generator-tool-judge loops, some call deterministic retrieval helpers, and some
delegate to web providers. The stable public contract is the helper-agent
surface and projected evidence, not one internal implementation path.

## RAG3 Local-Context Stage Agents

| Field | Contract |
|---|---|
| Owning package | `kazusa_ai_chatbot.local_context_resolver` |
| Discovery | None; four resolver-local stages are called directly by `resolve_local_context(...)`. |
| Runtime purpose | Resolve one bounded local/private context objective into a prompt-safe evidence packet. |
| Stage identifiers | Graph planner, active node resolver, collapse reviewer, bottom-up synthesizer. |
| Input | `LocalContextResolverRequestV1`, `LocalContextResolverContextV1`, and `LocalContextResolverOptionsV1`. |
| Output | `LocalContextResolutionPacketV1` plus retained `rag_result` projection. |
| Validation owner | Local-context contract validators, graph traversal, artifact normalizers, and packet projection. |
| Cache behavior | No resolver-owned cache; source artifacts and stage traces are bounded process-local review material. |
| Side effects | Return evidence only. No adapter delivery, persistence writes, shell/tool execution, or persona stance. |
| Required tests | Contract, graph, projection, cognition integration, prewarm integration, production-wired live LLM, and E2E persona tests. |

`LocalContextSubagentV1` is a future source-handler protocol in the local
context resolver contract. It is not a dynamic registry today, and docs must
not claim that RAG3 dispatches to concrete conversation, memory, person,
recall, live, or web modules until those handlers exist with tests.

## web_agent3 Source Subagents

| Field | Contract |
|---|---|
| Owning package | `kazusa_ai_chatbot.rag.web_agent3.subagent` |
| Discovery | Package discovery through `iter_modules(__path__)`, excluding packages and private modules. |
| Identifier | Module-level `SOURCE`. |
| Prompt description | Module-level `DESCRIPTION`. |
| Supported actions | Module-level `SUPPORTED_ACTIONS`. |
| Enablement | Optional module-level `is_enabled() -> bool`; absent means enabled. |
| Entry function | `execute(decision)`. |
| Input | web_agent3 router decision for a source/action pair. |
| Output | Source-local result consumed by web_agent3 providers and reducers. |
| Validation owner | `subagent.__init__` validates fields before registration. |
| Side effects | Search or read public web/source content only according to the source contract. |
| Required tests | web_agent3 routing, source availability, source action, and provider tests. |

Current source subagents include direct URL reads, direct web search when
configured, and source-specific metadata providers. Source modules must not
expose adapter ids, raw credentials, filesystem work, shell work, or final
persona wording.

## Complex Task Resolver Subagents

| Field | Contract |
|---|---|
| Owning package | `kazusa_ai_chatbot.complex_task_resolver.subagent` |
| Discovery | Package discovery through `iter_modules(__path__)`, excluding packages and private modules. |
| Identifier | Module-level `SUBAGENT`. |
| Prompt description | Module-level `DESCRIPTION`. |
| Supported actions | Module-level `SUPPORTED_ACTIONS`. |
| Owned node kinds | Module-level `OWNED_NODE_KINDS`. |
| Default action | Module-level `DEFAULT_ACTION`, which must be in `SUPPORTED_ACTIONS`. |
| Enablement | Optional module-level `is_enabled() -> bool`; absent means enabled. |
| Factory | Module-level `create() -> ComplexTaskSubagentV1`. |
| Runtime protocol | `ComplexTaskSubagentV1.run(task, context, max_attempts=...)`. |
| Input | `ComplexTaskSubagentRequestV1`. |
| Output | `ComplexTaskSubagentResultV1`. |
| Validation owner | Complex-task contract validators and subagent discovery validation. |
| Side effects | Resolver-local evidence collection or deterministic algorithmic work only. |
| Required tests | Contract, service, algorithmic, evidence, and real LLM review tests when prompts change. |

Complex-task resolver subagents are internal to the resolver. External callers
use the public resolver IO and must not provide alternate subagent rosters,
prompt variants, graph paths, or expected answers.

## Background Work Workers

| Field | Contract |
|---|---|
| Owning package | `kazusa_ai_chatbot.background_work.subagent` |
| Discovery | Explicit worker list in `discover_background_work_workers()`. |
| Identifier | Module-level `WORKER`. |
| Prompt description | Module-level `DESCRIPTION`. |
| Entry function | `execute(decision, max_output_chars=...)`. |
| Input | `BackgroundWorkWorkerDecision`. |
| Output | `BackgroundWorkResult`. |
| Validation owner | Background-work queue/runtime, router, worker-local validators, and accepted-task lifecycle. |
| Enablement | Worker registry is explicit; current worker modules do not use `is_enabled()`. |
| Side effects | Produce bounded worker result or schedule a durable follow-up path according to the worker contract. |
| Required tests | Background-work runtime tests, accepted-task integration tests, worker-specific tests, and result-handoff tests. |

Workers must not send adapter text directly, call shared cognition directly,
run shell commands, edit repository files, install packages, process
attachments, or write arbitrary persistence. Completed work returns through
accepted-task result cognition or a documented durable follow-up path.

## Cross-Family Rules

- Keep family-local identifiers exact: `name`, `SOURCE`, `SUBAGENT`, and
  `WORKER` are not interchangeable runtime fields.
- Keep prompt-facing descriptions short, semantic, and free of hidden
  operational options.
- Validate module fields before registration and validate request/result
  envelopes before mutating graph, queue, cache, or persistence state.
- Keep raw storage ids, adapter targets, credentials, prompts, and final
  visible wording out of specialist descriptions.
- Treat RAG and web results as evidence; cognition and dialog own stance and
  final user-visible wording.
- Add a new worker or subagent only with a reviewed family-specific contract
  and tests. Do not use this guide as authorization to add capabilities.
