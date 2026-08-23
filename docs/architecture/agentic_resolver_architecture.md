# Agentic Resolver Architecture

## Document Control

- Status: target architecture reference.
- Document type: system architecture and ownership decision.
- Target package: **src/agentic_resolver**.
- Execution authority: none. Production implementation requires an approved
  or in-progress plan under **development_plans/active/** and a separate
  explicit implementation command.
- First-pass boundary: additive standalone runtime reached through its Python
  API only.
- Later transition: a separate big-bang plan may connect the resolver to
  Kazusa cognition, accepted tasks, and background work after the standalone
  runtime is accepted.
- Governing project references:
  - **AGENTS.md**
  - **docs/DOCUMENTATION_GUIDE.md**
  - **src/kazusa_ai_chatbot/llm_interface/README.md**
  - **src/kazusa_ai_chatbot/task_resolution/README.md**
  - **src/kazusa_ai_chatbot/local_context_resolver/README.md**
  - **src/kazusa_ai_chatbot/complex_task_resolver/README.md**

## Executive Decision

Kazusa gains one top-level agentic resolver implemented as a traditional
native tool-calling loop rather than a DAG. The first implementation lives
alongside the current runtime and has no inbound edge from cognition, the
brain service, task resolution, accepted tasks, background work, adapters, or
delivery.

The standalone resolver can consume the current Kazusa capability
implementations through downward-only adapters. Those implementations retain
their present contracts, internal graphs, permissions, timeouts, and failure
semantics. The new package changes orchestration ownership only inside its
directly invoked standalone session.

The resolver also owns a first-class **run_subagent** tool. A subagent is a new
instance of the same resolver runtime with an isolated session, the same model
adapter, the same ordinary tools, the same skills, the same permission scope,
and the same JSON protocol. Its registry omits **run_subagent**, fixing
delegation depth at one. The parent receives the bounded child result as a
normal tool observation and owns convergence.

Every non-empty resolver-authored semantic textual payload is a JSON object:

- system policy;
- task input;
- skill catalog;
- loaded skill content;
- tool results;
- subagent tasks and results;
- contract-error feedback;
- compacted observations; and
- terminal result arguments.

Native tool schemas and native tool arguments are JSON Schema and JSON
objects. XML and pseudo-XML prompt frames are outside this architecture.
The resolver model route requires provider thinking to be enabled and uses a
streaming native-tool interface for every model step. Resolver-authored
semantic content remains JSON. Provider reasoning is carried separately as an
opaque assistant reasoning channel: it is retained for provider-required
round-trip continuity, counted against context, and excluded from resolver
actions, evidence, tool observations, terminal fields, and public results.

## Confirmed Decisions

1. The first pass introduces the resolver as a standalone package.
2. Current Kazusa workflows remain behaviorally and structurally unchanged.
3. Existing tool implementations remain unchanged.
4. The first pass may call existing public capability functions through new
   adapter definitions.
5. The resolver loop itself is non-DAG. A wrapped existing capability may
   retain its current internal graph during this phase.
6. Skills use a startup catalog plus one lazy-loading **skill** tool, following
   the DeepSeek Harness separation between discovery and loading.
7. The skill catalog is injected as JSON.
8. The future cognition-facing capability name remains
   **task_resolution_request**.
9. Workflow integration and old-orchestrator decommission are a later
   big-bang change.
10. Subagents are first-class in the first pass and use the same resolver
    implementation with recursive delegation disabled.
11. The root and every child require a thinking-enabled supported model route.
12. Every resolver model step consumes the additive LLInterface native-tool
    stream; Phase 1 has no non-streaming resolver model path.
13. Reasoning is retained as opaque assistant transport state and is never
    injected as a system, user, tool-result, skill, or terminal JSON field.

## Goals

| Goal | Target state |
|---|---|
| Standalone execution | A direct Python caller can construct a runtime and resolve one task without importing or starting Kazusa cognition or the brain service. |
| Traditional agent loop | One append-only session alternates native model tool calls and deterministic tool observations until a terminal result. |
| JSON-only model boundary | Every resolver-authored textual message supplied to the model parses as exactly one JSON object; provider reasoning remains a separate typed channel. |
| Thinking stream | Every model step streams typed reasoning, text, tool-call, usage, and finish events; reasoning remains distinct from semantic JSON. |
| Dynamic startup composition | The runtime freezes the supplied ordinary tools and discovered skill catalog at construction time and presents that exact set to the model. |
| Existing capability reuse | The initial Kazusa adapter registry exposes the four current task-resolution specialist boundaries without editing their implementations. |
| First-class delegation | The parent may run bounded, isolated same-runtime subagents and converge their typed results. |
| Context safety | Every model request accounts for policy, tool schemas, catalog, task, transcript, retained reasoning, loaded skills, tool results, and completion reserve under the project cap. |
| Inspectability | Session events, stream chunks, assembled assistant turns, tool calls, results, budget decisions, child lineage, and terminal disposition remain reconstructable. |
| Future cutover readiness | A later plan can integrate the accepted standalone contract without redesigning the loop. |

## First-Pass Non-Goals

The first pass leaves these areas for later plans:

- cognition or persona-graph integration;
- replacement of **task_resolution_request** handling;
- accepted-task or background-work integration;
- durable database checkpoints and process-restart resume;
- a slash command, HTTP route, control-console surface, or adapter command;
- refactoring the local-context, complex-task, WebAgent3, coding, or
  text/computation implementations;
- removing any current DAG;
- skill filesystem watching or live catalog replacement;
- skill scripts, assets, or arbitrary resource execution;
- recursive subagents;
- background, continuable, steerable, or parallel subagent control;
- incremental public-result, UI, CLI, or adapter streaming;
- exposing, persisting, summarizing, or treating private reasoning as evidence;
- arbitrary MCP, shell, database, filesystem, deployment, or delivery tools;
- a compatibility bridge between the standalone result and current workflow
  state.

## System Boundary

The first-pass dependency direction is deliberately one-way:

~~~text
direct Python caller
        |
        v
src/agentic_resolver
  core runtime
  JSON protocol
  tool registry
  skill catalog
  subagent runtime
        |
        +---- optional LLInterface native-tool adapter
        |
        +---- optional Kazusa capability adapters
                    |
                    +-- current local_context specialist
                    +-- current public_research specialist
                    +-- current coding specialist
                    +-- current text_computation specialist

current cognition / brain service / task resolution / background work
        |
        +---- no import, call, registration, or runtime selection edge
              to agentic_resolver in the first pass
~~~

Standalone describes the control-flow and public-entrypoint boundary. The
optional integration package may depend downward on existing Kazusa public
capabilities. Existing Kazusa workflow packages remain unaware of the new
resolver.

## Phased Architecture

### Phase 1: Standalone Runtime

Phase 1 creates and validates:

- a top-level installable **agentic_resolver** package;
- an async direct-call runtime API;
- an append-only in-memory session;
- provider-neutral native tool stream and assembly contracts;
- additive native tool streaming support in **LLInterface**;
- mandatory thinking for root and child model routes;
- a frozen tool registry;
- four unchanged Kazusa capability adapters;
- startup skill discovery and JSON catalog injection;
- one lazy **skill** tool;
- one foreground **run_subagent** tool;
- one controller-owned **submit_result** terminal tool;
- per-session context and execution budgets;
- deterministic tests and one inspected live-LLM workflow.

### Phase 2: Later Big-Bang Transition

Phase 2 requires its own approved plan. Its intended direction is:

~~~text
cognition task_resolution_request
    -> agentic_resolver
    -> inline result or durable checkpoint
    -> accepted_task/background resume when required
    -> ResolverObservation
    -> cognition owns stance
    -> dialog owns visible wording
~~~

That later cutover may replace the current task orchestrator and then decide
which graph-based resolver implementations become direct tools, are
decommissioned, or are refactored. Phase 1 supplies evidence for that decision
and performs no cutover itself.

## Component Ownership

| Component | Owns | Excludes |
|---|---|---|
| AgenticResolverRuntime | Public direct-call lifecycle, runtime composition, root session, final public result | Cognition, delivery, accepted-task persistence |
| AgentLoop | Model-step stream consumption, native tool-call admission, terminal selection, fixed caps | Tool semantics and caller permissions |
| ResolverSession | Append-only events, stream chunks, assembled assistant turns, model-history derivation, observation references, child lineage | Database persistence |
| ContextBudget | Request accounting, output reserve, deterministic compaction, hard-stop decision | Semantic relevance judgment |
| ToolRegistry | Unique names, JSON schemas, argument validation hooks, permission checks, execution dispatch, frozen views | Tool-domain implementation |
| SkillCatalog | Startup filesystem discovery, metadata validation, sorted summaries, digest, lazy body load | Capability or permission grants |
| SubagentRunner | Child construction, depth enforcement, inherited registry view, result projection, child cap | Child semantic reasoning |
| ModelStreamAssembler | Ordered reasoning/text/tool-call block assembly, usage, finish validation, safe interruption projection | Tool execution and semantic judgment |
| AgenticModelClient | Provider-neutral async native tool-call chunk stream plus immutable streaming/thinking capability declaration | Resolver loop policy |
| LLInterfaceToolModel | Adapt additive LLInterface native-tool streams to AgenticModelClient and preserve provider reasoning replay state | Resolver or tool semantics |
| Kazusa tool integration | Bind the existing four specialist handlers to resolver tool definitions | Changes to specialist implementations |
| Existing capability handlers | Their current RAG, web, coding, and computation behavior | Resolver session and parent convergence |

## Public Runtime Contract

The core public API is construction plus one async resolve call:

~~~python
runtime = AgenticResolverRuntime(
    model=model_client,
    tools=tool_registry,
    skills=skill_catalog,
    limits=limits,
)

result = await runtime.resolve(request)
~~~

The request is prompt-safe semantic input:

~~~json
{
  "schema_version": "agentic_resolver_request.v1",
  "objective": "Compare the available local record with current public information.",
  "context": {
    "facts": [
      "The caller wants a source-grounded comparison."
    ],
    "constraints": [
      "Separate local evidence from public evidence."
    ],
    "desired_output": "A concise comparison with limitations."
  }
}
~~~

Trusted execution objects, permission scopes, credentials, database handles,
workspace roots, model configuration, and tool handlers are constructor-owned
runtime inputs. They are never fields the model can author.

The public result is a code-validated projection:

~~~json
{
  "schema_version": "agentic_resolver_result.v1",
  "session_id": "resolver-session-id",
  "status": "resolved",
  "summary": "The comparison is complete.",
  "evidence": [
    {
      "evidence_id": "observation-1",
      "summary": "Bounded evidence summary.",
      "provenance_refs": [
        "source-reference"
      ],
      "limitations": []
    }
  ],
  "completed_tasks": [
    "Compare the local and public evidence."
  ],
  "remaining_needs": [],
  "usage": {
    "model_steps": 3,
    "tool_calls": 2,
    "subagent_runs": 1,
    "estimated_context_tokens_peak": 12000
  }
}
~~~

Allowed terminal statuses are:

- **resolved**
- **partial**
- **needs_user_input**
- **approval_required**
- **unavailable**
- **budget_exhausted**
- **failed**

The model supplies the semantic terminal fields through **submit_result**.
Deterministic code supplies session identity, validated evidence projections,
usage, and the final disposition when a hard runtime cap terminates the loop.

## Native Tool-Calling Loop

The first-pass loop is serialized and bounded:

~~~text
construct runtime
  -> discover skills
  -> freeze root tool registry
  -> append JSON system policy
  -> append JSON skill catalog
  -> append JSON task
  -> derive request history from session events
  -> measure context
  -> compact old observations when needed
  -> open thinking-enabled model stream with native JSON tool schemas
  -> append each normalized stream chunk and feed one assembler
  -> finalize reasoning, content, tool calls, usage, and finish reason
  -> require exactly one native tool call
       -> ordinary capability: validate, execute, append JSON observation
       -> skill: load body, append JSON skill observation
       -> run_subagent: run isolated child, append bounded JSON child result
       -> submit_result: validate and terminalize
  -> repeat within fixed caps
~~~

Exactly one native tool call is accepted per model step in Phase 1. This keeps
ordering, context accounting, error feedback, child lineage, and replay
unambiguous for weaker local models.

A response with no tool call, multiple tool calls, invalid JSON arguments,
an unknown tool name, or an invalid terminal payload becomes one bounded JSON
contract-error observation. The producing model receives another step while
the replacement budget remains. Exhausting that budget returns **failed**.

The controller never interprets assistant prose as an action or final answer.
Normal completion requires **submit_result**.

## Thinking And Streaming Contract

### Streaming Is The Resolver Model Boundary

The provider-neutral model seam is an asynchronous chunk stream:

~~~python
async def astream(
    messages: Sequence[ModelMessage],
    *,
    tools: Sequence[ModelToolDefinition],
) -> AsyncIterator[ModelStreamChunk]:
    ...
~~~

AgenticModelClient also exposes an immutable provider-neutral capability
descriptor containing **streaming = true**, **thinking_enabled = true**, an
enabled thinking-strategy identifier, and an adapter-owned reasoning replay
policy. Runtime construction validates that descriptor before starting a root
or child session. Provider-specific fields remain inside the adapter.

The normalized closed chunk family is:

- **block_start**, carrying block index and reasoning, text, or tool-call type;
- **reasoning_delta**, carrying opaque reasoning text for one block index;
- **text_delta**, carrying assistant content for one block index;
- **tool_call_delta**, carrying call ID, optional name, and one raw JSON
  argument fragment for one block index;
- **block_end**, carrying the completed normalized block;
- **usage**, carrying provider-neutral token counters; and
- **finish**, carrying stop, tool_calls, max_tokens, aborted, or error.

Block indexes allow interleaved reasoning, text, and tool-call deltas without
using arrival order as tool identity. The AgentLoop appends every normalized
chunk to the in-memory session and feeds the same chunk to one
ModelStreamAssembler. Only a successfully finished assembled tool call reaches
argument validation or execution. An interrupted or max-token stream cannot
execute a partially assembled tool call.

Streaming is internal model transport in Phase 1. The public **resolve** method
still returns one terminal AgenticResolverResultV1; it does not expose a token
or thought stream to its caller.

### Thinking Is Opaque Assistant State

Runtime construction requires **LLMCallConfig.thinking.enabled = true** and a
backend descriptor whose thinking strategy is supported and enabled. A route
reported as **ignored_unsupported_model** fails construction rather than
silently running the resolver without thinking. The root and every child use
the same accepted thinking-enabled model adapter.

Reasoning deltas assemble into a reasoning block attached to the assistant
turn that produced them. Reasoning is not injected as a new system, user,
tool-result, skill, compacted-observation, or terminal JSON message. The
resolver never parses it for actions, validation, evidence, permissions, or
semantic result fields.

DeepSeek's thinking-mode tool protocol distinguishes assistant turns that
actually carried tool calls from tool-call-free turns. A tool-calling
assistant turn's complete **reasoning_content** must be passed back on later
requests; when that turn has no reasoning text, the adapter still supplies the
empty field if the API requires its presence. Reasoning from a tool-call-free
assistant turn may be omitted and is ignored if supplied. The session retains
a provider-neutral reasoning block, and the LLInterface provider adapter
applies that provider-specific replay rule. The resolver itself never authors
a provider-specific reasoning field or copies thought text into ordinary
content.

The JSON-only rule governs resolver-authored semantic content. Native tool
schemas and arguments remain JSON; opaque reasoning blocks, native tool-call
metadata, usage, and finish events are typed transport rather than an
alternate prompt format. Provider compatibility triggers are applied only by
LLInterface on copied provider messages and never become resolver session
instructions.

Reasoning is private operational state. It is excluded from public results,
ordinary diagnostics, tool observations, subagent results, and evidence. A
protected debug artifact may record its presence, size, and stream ordering;
thought text itself remains outside ordinary artifacts.

### DeepSeek Harness Reference Flow

The researched DeepSeek Harness path is:

~~~text
derive messages from append-only session events
  -> LlmRuntime.stream(...)
  -> DeepSeekAdapter sends one streaming SSE request with native tools
  -> translate reasoning_content, content, and tool-call deltas into indexed chunks
  -> append assistant/chunk for replay fidelity
  -> feed the same chunks to the shared BlockAssembler
  -> append the completed assistant/message
  -> append and execute complete tool calls, then append tool results
  -> derive the next request from session events
  -> serialize reasoning_content only for prior assistant tool-call turns
     (including an empty field when the provider requires its presence)
~~~

The direct DeepSeek adapter is streaming-only and requests usage in the stream.
It ignores the initial empty **reasoning_content** delta rather than creating an
empty reasoning block, emits usage before the terminal finish event, and emits
nothing after finish. The shared LLM seam keeps reasoning, visible text, tool
calls, and tool results as distinct content-block types. This is the reference
for Kazusa's transport ownership; Kazusa keeps its own JSON semantic protocol
and smaller Phase-1 execution policy.

## JSON-Only Model Protocol

### Serialization Rule

Every non-empty resolver-authored textual model message uses UTF-8 JSON with:

- exactly one object at the root;
- a required **schema_version**;
- a required **message_type** for internal protocol messages;
- no leading or trailing prose;
- no XML or pseudo-XML framing;
- no executable values;
- bounded strings and arrays; and
- stable field ordering for cache-friendly prompt prefixes.

Human-readable instructions remain strings or string arrays inside JSON.
Loaded Markdown skill content remains a JSON string value.

Assistant responses carry their action in native JSON tool-call arguments.
Assistant textual content is accepted only when empty or when it parses as one
JSON object; it never supplies control semantics.

### System Policy

~~~json
{
  "schema_version": "agentic_resolver_system.v1",
  "message_type": "system_policy",
  "role": "Resolve the supplied task by selecting registered tools and returning a typed result.",
  "decision_process": [
    "Inspect the task and current observations.",
    "Load a matching skill before following its instructions.",
    "Use a tool only when it advances the task.",
    "Use run_subagent for a focused independent branch.",
    "Use submit_result when the task is resolved or a terminal limitation is known."
  ],
  "protocol": {
    "response_transport": "native_tool_call",
    "tool_calls_per_step": 1,
    "terminal_tool": "submit_result"
  }
}
~~~

### Skill Catalog

~~~json
{
  "schema_version": "agentic_resolver_skill_catalog.v1",
  "message_type": "skill_catalog",
  "catalog_digest": "sha256-digest",
  "skills": [
    {
      "name": "example-skill",
      "description": "Instructions for an example task family."
    }
  ],
  "selection": {
    "tool": "skill",
    "instruction": "Load every clearly applicable skill before taking task actions."
  }
}
~~~

### Tool Observation

~~~json
{
  "schema_version": "agentic_resolver_tool_observation.v1",
  "message_type": "tool_observation",
  "tool_call_id": "provider-call-id",
  "observation_id": "observation-id",
  "tool_name": "local_context",
  "status": "success",
  "output": {
    "summary": "Bounded tool output.",
    "evidence_refs": [
      "local-reference"
    ],
    "limitations": []
  },
  "error": null
}
~~~

### Contract Error

~~~json
{
  "schema_version": "agentic_resolver_contract_error.v1",
  "message_type": "contract_error",
  "code": "multiple_tool_calls",
  "message": "Return exactly one registered native tool call.",
  "remaining_replacements": 1
}
~~~

### Compacted Observation

~~~json
{
  "schema_version": "agentic_resolver_compacted_observation.v1",
  "message_type": "compacted_observation",
  "observation_id": "observation-id",
  "tool_name": "public_research",
  "status": "success",
  "summary": "Previously returned bounded summary.",
  "evidence_refs": [
    "public-reference"
  ]
}
~~~

## Session Model

Phase 1 uses an append-only in-memory event log. Model history is derived from
that log before every request.

Minimum event families are:

- session started;
- JSON policy appended;
- JSON skill catalog appended;
- task appended;
- model step started;
- normalized model stream chunk appended;
- assistant turn assembled with reasoning, content, tool calls, usage, and
  finish reason;
- assistant tool call accepted;
- tool execution started;
- tool result appended;
- child started;
- child completed;
- context compaction applied;
- contract error appended; and
- session terminalized.

The event log retains full bounded tool results and bounded opaque reasoning
blocks for private process-local replay. Compaction changes only the derived
model view. Every accepted tool call keeps one paired result or one paired
sanitized failure, and every retained assistant tool-call turn keeps the exact
reasoning state required by its provider replay contract.

Database persistence, reload, resume, fork, and accepted-task checkpoint
materialization are Phase 2 concerns.

## Tool Registry

### Tool Definition

Each registered tool supplies:

- unique name;
- short semantic description;
- object-rooted JSON input schema;
- argument validator;
- async executor;
- permission predicate;
- result projector;
- result-size bound; and
- side-effect classification.

Registration is deterministic. Reserved core names are:

- **skill**
- **run_subagent**
- **submit_result**

The registry rejects duplicate names, invalid schemas, and attempts to shadow
reserved tools. It freezes before the first model request. The model receives
only tools present in the frozen view.

### Phase-1 Kazusa Capability Set

The optional Kazusa integration registers exactly the four current specialist
boundaries:

| Tool name | Existing implementation | Semantic ownership |
|---|---|---|
| local_context | task-resolution local-context specialist | Private/local RAG evidence |
| public_research | task-resolution public-research specialist | Public research; existing complex resolver and WebAgent3 remain underneath |
| coding | task-resolution coding specialist | Existing coding-run lifecycle and approval boundary |
| text_computation | task-resolution text/computation specialist | Bounded transformation and computation |

Adapters construct the current validated specialist request and pass the
trusted current execution context captured when the registry is built. They
return the existing specialist result as a bounded JSON observation.

Existing specialist source files remain unchanged. Their current validation,
graph execution, permission behavior, approval behavior, and failure outputs
remain authoritative.

The four semantic tools keep the roster compact for the local model while
covering current local RAG, web research, coding, and computation families.
Additional interfaces can join a later registry revision without changing the
agent loop.

## Skill Architecture

### Discovery

The caller supplies one or more explicit filesystem roots when constructing
the runtime. The repository convention for project-local external skills is
**resolver_skills/**.

Phase 1 accepts one-level directory bundles:

~~~text
resolver_skills/
  example-skill/
    SKILL.md
~~~

Each skill requires YAML frontmatter with:

- **name**, matching its directory name;
- **description**, a bounded routing summary.

Names use lowercase kebab-case. Discovery is non-recursive below each skill
bundle. Duplicate names, malformed frontmatter, name mismatches, unreadable
files, configured bound violations, and a resolved SKILL.md path outside its
explicit configured root fail runtime construction. YAML is parsed with a
safe loader, and skill discovery never executes tags or constructors.

### Catalog

At startup the runtime:

1. scans every configured root;
2. validates candidates;
3. sorts summaries by name;
4. computes a digest over canonical name/description pairs;
5. creates one immutable SkillCatalog; and
6. injects the JSON catalog into each new root or child session.

Phase 1 fixes the catalog for the lifetime of the runtime. A new runtime
construction observes filesystem changes.

### Lazy Loading

Only name and description enter the startup catalog. The core **skill** tool
loads the complete body when selected:

~~~json
{
  "schema_version": "agentic_resolver_skill_content.v1",
  "message_type": "skill_content",
  "name": "example-skill",
  "description": "Instructions for an example task family.",
  "catalog_digest": "sha256-digest",
  "content_format": "markdown",
  "content": "Complete SKILL.md instruction body."
}
~~~

Loaded skill instructions influence semantic reasoning only. They cannot add
tools, grant permissions, change deterministic limits, create persistence,
or authorize external effects.

## First-Class Subagents

### Semantic Purpose

The **run_subagent** tool creates an independent reasoning branch for a
focused task. It supports divergence by giving each child a fresh transcript
and explicit self-contained task. It supports convergence by returning only a
typed bounded result to the parent transcript.

### Tool Arguments

~~~json
{
  "description": "Verify public sources",
  "objective": "Independently identify the strongest public evidence for the claim.",
  "context": {
    "facts": [
      "The parent needs an independent evidence branch."
    ],
    "constraints": [
      "Report uncertainty and source limitations."
    ],
    "desired_output": "A source-grounded evidence summary."
  }
}
~~~

The controller generates child identity and lineage. The model does not choose
child permissions, tools, model routes, context caps, timeouts, or delegation
depth.

### Same-Runtime Invariant

A child is constructed through the same AgenticResolverRuntime and AgentLoop
implementation as the parent. It receives:

- the same AgenticModelClient;
- the same thinking-enabled streaming contract and provider replay policy;
- the same ordinary frozen tools;
- the same SkillCatalog and JSON catalog;
- the same JSON policy and terminal contract;
- the same trusted permission scope;
- a fresh ResolverSession;
- a fresh per-session context budget; and
- a child capability view in which **run_subagent** is absent.

There is no separate child prompt family, planner, summarizer, or resolver
implementation.

### Isolation

The child receives the explicit **objective** and **context** supplied in the
tool call. It receives no automatic copy of the parent transcript or sibling
results. The parent can place selected prompt-safe facts into the child
context when they are genuinely required.

This keeps independent branches independent and prevents the child from
consuming the parent conversation budget.

### Result Projection

The child result returned to the parent is:

~~~json
{
  "schema_version": "agentic_resolver_subagent_result.v1",
  "message_type": "subagent_result",
  "subagent_id": "generated-child-id",
  "observation_id": "root-session:observation:2",
  "description": "Verify public sources",
  "status": "resolved",
  "summary": "Bounded independent result.",
  "evidence": [
    {
      "summary": "Bounded child evidence.",
      "provenance_refs": [
        "public-reference"
      ],
      "limitations": []
    }
  ],
  "remaining_needs": []
}
~~~

The top-level `observation_id` is allocated by the parent session after the
child returns and is the only observation handle that the parent may cite in
terminal `submit_result` evidence. Nested child evidence is provenance context
with `summary`, `provenance_refs`, and `limitations` only; child-session
observation IDs remain private and are omitted from this message.
An observation handle may appear only in
`submit_result.evidence[].observation_id`. Model-authored terminal summary,
evidence summary or limitations, completed-task, and remaining-need text must
not repeat a current-session observation ID. The terminal validator rejects a
misplaced handle with bounded contract feedback for model regeneration and
preserves the semantic text unchanged; provenance references remain a separate
validated channel.

The parent does not receive the child's intermediate transcript. It may invoke
multiple children across separate model steps and then use **submit_result** to
converge their results.

### Phase-1 Execution Policy

- Child runs are foreground and awaited.
- One **run_subagent** call creates one child.
- A root session may create at most three children.
- Child depth is exactly one because its registry omits **run_subagent**.
- The child runs within the root session's remaining wall-clock deadline.
- Each child has its own 50,000-token project context ceiling.
- The parent receives at most 8,000 characters of validated child result.
- A child failure becomes a typed tool observation so the parent can continue
  or terminalize honestly.

Background children, simultaneous tool calls, follow-up messaging,
interruption, child listing, and durable child continuation are future
extensions.

## Context And Execution Budgets

### Context Ceiling

The project context ceiling is 50,000 estimated tokens per resolver session.
The runtime reserves 8,000 tokens for model completion, leaving a hard
42,000-token model-input ceiling.

The effective ceiling is:

~~~text
minimum(
  project context ceiling,
  caller-declared model context window
)
~~~

A model route without a declared context window still receives the project
ceiling.

The token meter counts canonical serialized forms of:

- JSON system policy;
- native tool schemas;
- JSON skill catalog;
- JSON task;
- loaded skill content;
- assistant reasoning selected for provider replay and assistant text;
- assistant tool calls;
- JSON tool observations;
- JSON subagent results;
- JSON contract errors; and
- reserved completion capacity.

The deterministic fallback estimate is ceiling(serialized UTF-8 character
count divided by four), including provider replay fields. Provider-reported
usage is recorded after calls but does not retroactively authorize an over-cap
request. Reported reasoning tokens are informational output detail and are not
added a second time when already included in output tokens.

### Fixed Phase-1 Limits

| Limit | Default | Hard maximum |
|---|---:|---:|
| Context window per session | 50,000 tokens | 50,000 tokens |
| Reserved completion | 8,000 tokens | 8,000 tokens |
| Model steps per session | 8 | 16 |
| Non-terminal tool calls per session | 6 | 12 |
| Contract-error replacements | 2 | 2 |
| Root subagent runs | 3 | 3 |
| Tool observation supplied to model | 8,000 characters | 8,000 characters |
| Child result supplied to parent | 8,000 characters | 8,000 characters |
| Skill catalog entries | 64 | 64 |
| Skill description | 500 characters | 500 characters |
| Loaded skill body | 16,000 characters | 16,000 characters |
| Session wall clock | 300 seconds | 600 seconds |
| Individual ordinary tool call | 180 seconds | 180 seconds |

The caller may lower configurable defaults within these hard maxima. The model
never receives authority to increase a limit.

### Compaction

Before each model call the runtime measures the complete request. When it
would exceed the input ceiling, the runtime replaces oldest model-visible tool
observations with their compact JSON projections while preserving:

- the task;
- system policy;
- current skill catalog;
- loaded applicable skills;
- recent assistant reasoning/tool-call/result pairing;
- observation identity;
- evidence references;
- recent uncompressed observations; and
- every unresolved need.

For a completed older step, compaction either retains the assistant reasoning,
tool call, and result together or removes that complete exchange from the
derived provider history and replaces it with one compacted JSON observation.
It never leaves a retained assistant tool call without provider-required
reasoning passback. The append-only session events retain the original bounded
observations and private reasoning blocks. If the request still cannot fit,
the controller returns **budget_exhausted** without sending an over-cap model
request.

## Permission And Semantic Ownership

The LLM owns:

- interpretation of the supplied task;
- selection of the next semantic capability;
- formulation of bounded semantic tool arguments;
- deciding when an independent child branch is useful;
- evaluating whether gathered information is sufficient; and
- the semantic fields of the terminal result.

Deterministic code owns:

- tool registration and visibility;
- schema validation;
- permissions and trusted execution scope;
- call, child, time, size, and context limits;
- child recursion prevention;
- observation identifiers and lineage;
- exception sanitization;
- evidence-reference existence checks;
- context compaction;
- terminal contract validation; and
- final hard-cap disposition.

The controller does not keyword-route the user's objective, rewrite model
intent into another channel, infer permissions from prose, or turn skill
instructions into authority.

## Failure Behavior

| Failure | Resolver behavior |
|---|---|
| Malformed skill catalog | Runtime construction fails before a model call. |
| Duplicate or reserved tool name | Registry construction fails before a model call. |
| Resolver route has thinking disabled or unsupported | Runtime construction fails before the first model call. |
| Stream is malformed, closes without a terminal finish, or violates block identity | Return a typed provider failure; execute no partial tool call. |
| Stream reaches max tokens with an incomplete tool call | Drop the incomplete call and return budget_exhausted. |
| Provider returns invalid native calls | Append bounded JSON contract error and consume one replacement. |
| Tool arguments fail validation | Append bounded JSON tool error; the tool remains unexecuted. |
| Tool raises | Append sanitized bounded JSON tool error without stack trace or secrets. |
| Child reaches a terminal limitation | Return typed child result to the parent. |
| Child infrastructure fails | Return a typed failed child observation to the parent. |
| Context cannot fit after compaction | Return budget_exhausted before the provider call. |
| Model step, tool, child, or wall-clock cap is reached | Return the matching bounded terminal disposition. |
| submit_result is structurally invalid | Append JSON contract error within the fixed replacement cap. |

Native tool arguments are provider-decoded JSON objects and receive strict
contract validation. Textual assistant content is not parsed into control
decisions.

## LLM Interface Extension

The existing **LLInterface.ainvoke** and **LLInterface.invoke** contracts
remain unchanged. Phase 1 adds a separate async native-tool stream and
provider-neutral history/chunk contracts:

~~~python
async def astream_tools(
    messages: Sequence[LLMToolHistoryMessage],
    *,
    tools: Sequence[LLMToolDefinition],
    config: LLMCallConfig,
) -> AsyncIterator[LLMStreamChunk]:
    ...
~~~

**LLMToolHistoryMessage** is role-discriminated and provider-neutral. An
assistant history row may contain opaque **reasoning**, JSON/empty **content**,
and native tool calls; a tool row carries the matching call ID and JSON result.
Only provider adapters translate reasoning into fields such as
**reasoning_content**.

The provider adapter binds the supplied schemas through the OpenAI-compatible
native tools transport and consumes the underlying model's async stream. It
normalizes reasoning deltas, text deltas, indexed tool-call argument fragments,
usage, and finish state without interpreting resolver semantics. Native tool
streaming omits JSON-object **response_format** because tool-call arguments are
the structured action transport.

The resolver integration accepts only
**LLMCallConfig.thinking.enabled = true** with an enabled supported thinking
strategy. Existing ordinary LLInterface callers retain their selected thinking
behavior. Existing normalized non-stream responses continue stripping visible
thought spans from caller-facing content.

For a provider that requires reasoning replay during tool use, the provider
adapter serializes the reasoning block from each qualifying assistant
tool-call turn through the native reasoning field. It omits tool-call-free
reasoning when the provider ignores it. It never copies a reasoning block into
ordinary message content. Qwen/Gemma prompt controls remain adapter-private
transformations on copied provider messages rather than resolver-authored
session content.

Phase 1 adds no assembled **ainvoke_tools** shortcut. The resolver consumes
**astream_tools** directly so streamed reasoning, indexed tool calls,
cancellation, usage, and terminal state remain observable to its loop.

Tool-bound provider sessions use a cache identity that includes the canonical
tool-schema digest. Existing non-tool model-session caching, JSON-object
output, JSON-Schema fallback, thinking controls, unload recovery, and response
normalization retain their current behavior. Stream unload recovery retries
only when the confirmed unload occurs before the first emitted chunk; once any
chunk has been yielded, retry is forbidden because it would duplicate or
corrupt the assembled assistant turn.

## Observability

Phase 1 exposes process-local diagnostics through the public result and an
optional caller-supplied event observer. Diagnostics include:

- session and child IDs;
- parent-child lineage;
- tool schema and skill catalog digests;
- model step count;
- stream chunk counts by normalized type and first/terminal chunk elapsed time;
- reasoning character count and provider reasoning-token usage without thought
  text;
- accepted and rejected tool-call counts;
- tool duration and outcome class;
- subagent duration and terminal status;
- estimated request tokens and peak;
- compaction count;
- provider usage when available; and
- final disposition.

Diagnostics exclude prompts, skill bodies, raw tool payloads, credentials,
absolute private paths, protected trace content, and raw exceptions from
ordinary public results.

Database event logging and protected LLM trace integration remain part of the
later workflow-integration plan.

## DeepSeek Harness Influence

The architecture adopts five concrete DeepSeek Harness ideas:

1. One model step plus its tool calls is the loop unit, and model history is
   derived from an append-only session log. See the
   [DeepSeek Harness architecture](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md)
   and
   [agent-loop implementation](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/agent-loop/src/agent.ts).
2. Every adapter exposes a chunk stream, reasoning is a content block distinct
   from visible text, the loop logs chunks, and one shared assembler produces
   the assistant turn. See the
   [DeepSeek LLM streaming contract](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/llm-streaming.md)
   and
   [DeepSeek provider adapter](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/llm/llm-deepseek/README.md).
3. Tool schemas, guarded execution, and results are separate from the loop.
   See the
   [DeepSeek tool runtime](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/tools/README.md).
4. Skill providers publish a lightweight catalog and one **skill** tool loads
   the full body on demand. See the
   [DeepSeek skill subsystem](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/skills.md)
   and
   [skill-tool implementation](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/skill/tool-skill/src/index.ts).
5. Subagent delegation is a tool-backed capability that starts a child with a
   self-contained task and a bounded result. See the
   [DeepSeek subagent subsystem](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/subagent.md)
   and
   [subagent tool](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/subagent/tool-subagent/src/index.ts).

DeepSeek's official thinking-mode API requires complete
**reasoning_content** passback for assistant turns that performed tool calls;
the adapter preserves required empty-field presence, while tool-call-free
reasoning may be dropped because the API ignores it. See the
[DeepSeek thinking-mode tool contract](https://api-docs.deepseek.com/guides/thinking_mode/)
and the Harness
[DeepSeek request serializer](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/llm/llm-deepseek/src/serialize.ts).
DeepSeek Harness represents reasoning separately from text, streams only,
serializes tool-call-turn reasoning back to the native wire field, and drops
tool-call-free reasoning to save replay tokens.

Kazusa intentionally uses JSON catalog and instruction envelopes in place of
DeepSeek's model-facing XML-style catalog frame. Phase 1 also uses one
in-process same-runtime provider, one foreground child operation, a fixed
depth of one, and no plugin microkernel. Kazusa retains reasoning only as
opaque provider replay state; it never promotes thought text into its semantic
JSON protocol.

## Architectural Invariants

1. Current workflow packages have no dependency on **agentic_resolver** during
   Phase 1.
2. The core resolver package has no import dependency on cognition, adapters,
   delivery, accepted tasks, background work, or database persistence.
3. Optional integration modules own every downward Kazusa import.
4. Existing tool implementations remain unchanged during Phase 1.
5. Every non-empty resolver-authored semantic textual payload parses as one
   JSON object.
6. Every model action is a native tool call.
7. **submit_result** is the only model-selected successful terminal path.
8. Every model request stays within the effective context ceiling.
9. Each child uses the same runtime implementation and a fresh session.
10. A child registry excludes **run_subagent**.
11. Child permissions equal or narrow the parent's trusted permissions.
12. The parent receives bounded child results rather than child transcripts.
13. Evidence remains evidence; later cognition owns character stance and
    dialog owns visible character wording.
14. The future big-bang transition requires a separate approved plan.
15. The standalone resolver route and every child require supported provider
    thinking to be enabled.
16. Every resolver model step uses the streaming native-tool interface.
17. Opaque reasoning remains attached to its assistant turn, is replayed only
    through the provider adapter, and never becomes a semantic JSON field.
18. A retained assistant tool-call turn keeps its provider-required reasoning;
    compaction atomically removes or retains the complete
    reasoning/call/result exchange.
