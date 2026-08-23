# Standalone Agentic Resolver First-Pass Implementation Plan

## Summary

- Goal: implement an independently callable top-level **agentic_resolver**
  package with a bounded thinking-enabled native tool stream, JSON semantic
  protocol, startup skill discovery, unchanged Kazusa capability adapters,
  and first-class same-runtime subagents.
- Status: completed.
- Plan class: additive standalone feature with shared LLM-interface extension.
- Scope boundary: direct Python construction and invocation only. Current
  cognition, brain-service, task-resolution, accepted-task, background-work,
  action, dialog, persistence, scheduler, adapter, and delivery flows remain
  unchanged and have no inbound dependency on the new package.
- Change direction: add the standalone engine beside the current workflow;
  validate its architecture and behavior before a later separately approved
  big-bang transition.
- Acceptance state: completed after explicit production-source authorization
  and final verification on 2026-08-23.
- Governing architecture:
  **docs/architecture/agentic_resolver_architecture.md**.
- Highest-risk boundaries:
  - adding native tool streaming without changing existing LLInterface calls;
  - retaining provider-required reasoning passback without promoting private
    thinking into semantic context or public output;
  - ensuring every resolver-authored model-visible textual payload is JSON;
  - reusing existing capability handlers without editing their behavior;
  - enforcing the 50,000-token project context ceiling;
  - constructing children from the identical runtime while removing recursive
    subagent capability; and
  - proving that no current workflow imports or invokes the standalone
    resolver.

## Scope And Change Direction

Implement one installable **src/agentic_resolver** package. A direct caller
constructs an AgenticResolverRuntime with:

- an AgenticModelClient;
- a frozen ToolRegistry;
- a startup SkillCatalog;
- trusted execution scope; and
- AgenticResolverLimitsV1.

The runtime accepts one AgenticResolverRequestV1 and runs a serialized native
tool loop until **submit_result** or a deterministic hard-cap disposition.

The core package remains independent from Kazusa cognition and workflow
packages. Optional modules under **agentic_resolver.integrations** provide the
only imports into **kazusa_ai_chatbot**:

- LLInterfaceToolModel uses the additive LLInterface thinking-enabled
  native-tool stream.
- build_kazusa_tool_registry binds the four current task-resolution
  specialist handlers.

Current workflow packages retain their existing dependency graph. The first
pass creates no service route, slash command, background worker, database row,
accepted task, action-spec capability, or cognition resolver registration.

The later big-bang transition is outside this plan. It requires a new plan
that wires **task_resolution_request** to the accepted standalone contract,
adds durable checkpoint/background semantics, and explicitly decommissions
the old orchestration surface.

## Confirmed Decisions

1. The package path is **src/agentic_resolver**.
2. Phase 1 is standalone and additive.
3. Current workflow behavior remains unchanged.
4. Existing tool implementation files remain unchanged.
5. The new loop is non-DAG; wrapped tools retain their current internals.
6. The initial Kazusa tool set is exactly:
   - **local_context**
   - **public_research**
   - **coding**
   - **text_computation**
7. Existing WebAgent3 access remains owned underneath the current
   **public_research** specialist in Phase 1.
8. The model uses native tool calls rather than textual JSON action routing.
9. Every resolver-authored non-empty textual model-facing payload is exactly
   one JSON object.
10. Resolver LLInterface configs require provider thinking to be enabled and
    supported.
11. Every root and child model step uses the additive native-tool stream.
12. Thinking is retained as opaque assistant reasoning state, replayed only
    through the provider adapter when required, and excluded from semantic JSON
    fields and public results.
13. The skill catalog is a JSON message containing name and description only.
14. One core **skill** tool loads a selected SKILL.md body lazily.
15. Skills are discovered once during runtime construction.
16. Project-local external skills use the repository convention
    **resolver_skills/<name>/SKILL.md**.
17. **run_subagent** is first-class in Phase 1.
18. A child uses the identical AgenticResolverRuntime and AgentLoop
    implementation with a fresh session.
19. A child inherits the same ordinary tool set, skill catalog, thinking-enabled
    streaming model adapter, permission scope, JSON protocol, reasoning replay
    policy, and per-session context policy.
20. The child registry omits **run_subagent**, fixing delegation depth at one.
21. Subagents run synchronously in the foreground and return one bounded typed
    result to the parent.
22. A root session may run at most three subagents.
23. The future cognition capability name remains **task_resolution_request**.
24. The future big-bang integration is represented only in the architecture
    reference during Phase 1.

## Cutover Policy

Overall strategy: compatible additive introduction with no workflow cutover.

| Area | Policy | Instruction |
|---|---|---|
| New resolver package | compatible | Add an independently callable package beside current task resolution. |
| Current cognition and brain path | retained | Preserve current source, imports, routing, prompts, behavior, and tests. |
| Current task-resolution path | retained | Preserve current inline/background orchestration and specialist selection. |
| Existing capability implementations | retained | Consume current handler contracts through new adapters and make no edits in their packages. |
| LLInterface | compatible | Add provider-neutral history/chunk contracts and an async native-tool stream while preserving existing ainvoke and invoke behavior. |
| Configuration | retained | Accept LLMCallConfig from the direct caller and add no resolver environment route in Phase 1. |
| Persistence | retained | Create no database schema, repository, job, checkpoint, or migration. |
| Later integration | deferred bigbang | Create a separate approved plan before any workflow caller or decommission change. |

## Mandatory Skills

- **development-plan**: load before reviewing, approving, executing, updating,
  handing off, or signing off this plan.
- **local-llm-architecture**: load before changing tool descriptions, model
  protocol, loop limits, skill catalog, context composition, subagent task
  projection, or model-call count.
- **no-prepost-user-input**: load before implementing task interpretation,
  terminal semantic fields, tool selection feedback, or direct-caller command
  handling.
- **py-style**: load before editing any Python production or test file.
- **test-style-and-execution**: load before adding, changing, collecting, or
  running tests.
- **debug-llm**: load before creating the live-LLM case, invoking a real model,
  or reviewing native tool/subagent output quality.
- **cjk-safety**: load if any planned Python source or test contains CJK text.

## Mandatory Rules

- Capture **git status --short**, the changed-path baseline, and the exact
  owned file set before implementation.
- Preserve pre-existing and concurrent user changes, including active plan
  registry work outside this plan.
- Use **venv\Scripts\python.exe** for Python commands.
- Keep **.env** unread throughout planning, implementation, and verification.
- Obtain a separate explicit user implementation command before changing
  production source.
- Keep every current workflow orchestration and caller source file outside the
  implementation diff. The listed additive LLInterface files are the only
  shared-infrastructure exception.
- Keep every existing tool implementation source file outside the
  implementation diff.
- Permit downward imports into Kazusa only from
  **src/agentic_resolver/integrations/**.
- Keep the core **src/agentic_resolver** modules free of imports from
  cognition, brain service, task resolution, accepted tasks, background work,
  action spec, dialog, database, scheduler, adapters, and delivery.
- Preserve the existing LLInterface **ainvoke**, **invoke**, output-mode,
  JSON-Schema fallback, thinking, reload, cache, response, and usage contracts.
- Add native tool streaming as a separate async contract and provider path.
- Require resolver LLInterface configs to have provider thinking enabled and
  reject disabled or **ignored_unsupported_model** strategies before the first
  model call.
- Use **LLInterface.astream_tools(...)** for every root and child model step;
  add no non-streaming resolver model path.
- Append every normalized stream chunk to the private in-memory session and
  feed the same chunk to one bounded assembler.
- Keep reasoning as an opaque assistant block. Parse no reasoning text for
  actions, arguments, validation, permissions, evidence, or terminal fields.
- Replay reasoning from qualifying assistant tool-call turns only through the
  provider adapter's native field when that provider requires passback; omit
  tool-call-free reasoning when the provider ignores it, and preserve an empty
  native reasoning field when a tool-call turn requires field presence.
- Serialize every resolver-authored non-empty model-facing textual payload as
  exactly one JSON object.
- Keep all stable process instructions in JSON system messages and all
  per-run facts in JSON task or observation messages.
- Use native JSON tool arguments for every model action.
- Accept exactly one native tool call per model step.
- Require **submit_result** for model-selected normal completion.
- Treat no tool call, multiple tool calls, malformed arguments, and invalid
  submit_result arguments as structural contract errors owned by the resolver
  loop.
- Apply at most two structural replacement steps per session.
- Keep user-task interpretation, capability choice, delegation choice, and
  terminal semantic judgment LLM-owned.
- Keep schema validation, permissions, trusted scope, identifiers, limits,
  exception sanitization, context admission, child depth, and hard-cap
  termination deterministic.
- Add no keyword routing, semantic post-filter, or deterministic rewriting of
  the model's accepted task or selected terminal channel.
- Give skills instructional authority only. Tool visibility and permission
  remain registry-owned.
- Construct children through the same runtime and loop classes used by the
  parent.
- Build a child-specific frozen registry view that excludes
  **run_subagent**.
- Give a child only its explicit task/context JSON plus the normal policy and
  skill catalog. Parent transcript and sibling results remain outside the
  child session.
- Return only the bounded AgenticResolverSubagentResultV1 projection to the
  parent.
- Keep subagent execution foreground and serial in this phase.
- Enforce the project context cap before every root and child model request.
- Count reasoning selected for provider replay and its native serialization
  fields in every context admission decision.
- During model-view compaction, retain or remove an assistant reasoning,
  tool-call, and tool-result exchange as one unit; never retain a tool-bearing
  assistant row without provider-required reasoning.
- Send no request that exceeds the effective input ceiling.
- Execute no partial tool call after malformed, interrupted, or max-token
  stream termination.
- Retry a confirmed LM Studio unload stream only before the first emitted
  chunk; never restart a partially emitted stream.
- Project tool exceptions into typed bounded errors without stack traces,
  credentials, absolute private paths, or raw provider internals.
- Exclude thought text from public results, ordinary diagnostics, tool
  observations, subagent results, and ordinary debug artifacts.
- Record one inspected live-LLM artifact through the debug-LLM contract before
  final sign-off.

## Must Do

- Create the **agentic_resolver** core package and module README.
- Add package discovery for **agentic_resolver***.
- Add an explicit PyYAML dependency for SKILL.md frontmatter parsing.
- Create strict request, result, tool, skill, session, subagent, stream, limit,
  and error contracts.
- Add a provider-neutral streaming AgenticModelClient protocol and bounded
  ModelStreamAssembler.
- Add an immutable AgenticModelCapabilitiesV1 declaration proving streaming,
  enabled thinking strategy, and adapter-owned reasoning replay policy before
  runtime construction succeeds.
- Add normalized native-tool stream contracts to LLInterface:
  - LLMToolDefinition
  - LLMToolHistoryMessage
  - LLMToolCall
  - LLMInvalidToolCall
  - LLMStreamChunk
  - LLMStreamFinish
  - LLMToolStreamInvoker
- Add **LLInterface.astream_tools(...)**.
- Add OpenAI-compatible native-tool binding, thinking/reasoning stream
  normalization, provider-specific tool-call-turn reasoning passback, and
  tool-schema-digest cache partitioning.
- Add streaming unload recovery that retries only before the first emitted
  chunk.
- Preserve the existing LLInterface methods and their regression suite.
- Implement canonical JSON message serialization and validation.
- Implement one append-only in-memory ResolverSession.
- Record normalized stream chunks and assembled assistant turns in the session.
- Derive every model request history, including opaque retained reasoning, from
  session events.
- Implement the fixed context meter and deterministic compaction policy.
- Implement a frozen ToolRegistry with reserved core tool names.
- Implement **skill**, **run_subagent**, and **submit_result** as core tools.
- Scan explicit one-level skill roots at runtime construction.
- Support exactly **<name>/SKILL.md** bundles in Phase 1.
- Validate name, description, body size, duplicates, and directory-name match.
- Resolve each bundle path and reject a SKILL.md outside its explicit root.
- Parse frontmatter with **yaml.safe_load**.
- Build a sorted immutable name/description catalog and SHA-256 digest.
- Inject the complete catalog as AgenticResolverSkillCatalogV1 JSON.
- Load the complete skill body only through the **skill** tool.
- Create **resolver_skills/.gitkeep** as the project-local external-skill root
  convention.
- Implement the serialized bounded native tool loop.
- Implement same-runtime subagents through SubagentRunner.
- Remove **run_subagent** from each child registry view.
- Enforce the fixed root child limit of three.
- Bound and validate child result projection before adding it to the parent
  session.
- Add an LLInterface integration adapter.
- Add a Kazusa tool-registry builder over the four current specialist
  handlers.
- Validate the captured TaskResolutionExecutionContextV1 before registering
  Kazusa tools.
- Translate each model tool argument into the current
  TaskSpecialistRequestV1 without changing the handler contract.
- Structurally project TaskSpecialistResultV1 into bounded resolver
  observations while preserving status, evidence summaries, provenance,
  limitations, completed subgoals, remaining needs, and coding-run state.
- Add exact source-to-test manifest rows for every changed production source.
- Add deterministic contract, protocol, loop, tool, skill, context, subagent,
  integration, isolation, documentation, and packaging tests.
- Add one real-LLM test using bounded fake ordinary tools and a real model
  route; inspect its trace one case at a time.
- Document the additive LLInterface thinking-enabled native-tool stream.

## Deferred

- Any import or invocation of **agentic_resolver** from current workflow code.
- The future big-bang transition.
- Replacement or deletion of current task-resolution orchestration.
- Changes to local-context resolver internals.
- Changes to complex-task resolver internals.
- Changes to WebAgent3 internals.
- Changes to coding-agent internals or approval behavior.
- Changes to text/computation specialist behavior.
- Direct low-level web-search and URL-read tool registration.
- New Kazusa capability families beyond the four confirmed specialists.
- A model-facing **task_resolution_request** change.
- Brain-service, cognition-resolver, action-spec, accepted-task,
  background-worker, dialog, scheduler, adapter, delivery, and control-console
  integration.
- A slash command, CLI, HTTP API, or user-interface surface.
- Incremental public-result, UI, CLI, adapter, or caller-facing token streams.
- Public or ordinary-log exposure of model reasoning text.
- A non-streaming resolver model fallback.
- Database persistence, checkpoint storage, session reload, or resume.
- Background or continuable subagents.
- Parallel native tool calls or parallel child execution.
- Child steering, follow-up messaging, listing, interruption, cancellation, or
  job control.
- Recursive delegation.
- Skill catalog watching, invalidation, hot reload, or session replacement
  catalogs.
- Flat **<name>.md** skills.
- Skill scripts, references, assets, resource enumeration, or executable
  extensions.
- User slash-gesture skill invocation.
- Arbitrary MCP, shell, filesystem, database, deployment, publishing, calendar,
  messaging, or delivery capabilities.
- A new environment-configured resolver LLM route.
- Compatibility aliases, fallback orchestrators, dual routing, or shadow
  workflow execution.

## Target State

### Package Boundary

~~~text
src/agentic_resolver/
  __init__.py
  README.md
  contracts.py
  json_protocol.py
  model.py
  streaming.py
  runtime.py
  loop.py
  session.py
  context_budget.py
  tools.py
  skills.py
  subagents.py
  integrations/
    __init__.py
    llm_interface.py
    kazusa_tools.py
~~~

Core modules depend only on standard library, declared generic dependencies,
and other core agentic_resolver modules. Kazusa-specific imports appear only
inside **integrations/**.

### Public API

**agentic_resolver.__init__** exports:

- AgenticResolverRuntime
- AgenticResolverRequestV1
- AgenticResolverResultV1
- AgenticResolverLimitsV1
- AgenticResolverContractError
- AgenticModelClient
- AgenticModelCapabilitiesV1
- ModelStreamChunk
- ModelStreamAssembler
- ToolDefinition
- ToolRegistry
- SkillCatalog
- discover_skills

The public async entrypoint is:

~~~python
async def resolve(
    self,
    request: AgenticResolverRequestV1,
) -> AgenticResolverResultV1:
    ...
~~~

Phase 1 exports no global runtime, service singleton, background entrypoint, or
resume API.

### Direct Construction

The caller supplies all operational dependencies:

~~~python
runtime = AgenticResolverRuntime(
    model=model_client,
    tools=tool_registry,
    skills=skill_catalog,
    limits=limits,
)
result = await runtime.resolve(request)
~~~

The Kazusa convenience composition is:

~~~python
runtime = create_kazusa_resolver_runtime(
    llm_interface=llm_interface,
    llm_config=llm_config,
    execution_context=execution_context,
    skill_roots=skill_roots,
    limits=limits,
)
~~~

This helper is direct-call composition only. It performs no registration with
the current application runtime. It rejects a config unless
**thinking.enabled** is true and the detected backend thinking strategy is
supported and enabled.

## Contracts And Data Shapes

### AgenticResolverRequestV1

~~~json
{
  "schema_version": "agentic_resolver_request.v1",
  "objective": "One bounded semantic task.",
  "context": {
    "facts": [
      "Prompt-safe fact."
    ],
    "constraints": [
      "Prompt-safe constraint."
    ],
    "desired_output": "Prompt-safe output description."
  }
}
~~~

Contract rules:

- exact top-level and context keys;
- non-empty objective;
- at most 32 facts;
- at most 32 constraints;
- each string at most 2,000 characters;
- desired_output at most 2,000 characters; and
- no trusted operational objects in the request.

### SubmitResultV1

~~~json
{
  "status": "resolved",
  "summary": "Bounded terminal summary.",
  "evidence": [
    {
      "observation_id": "known-observation-id",
      "summary": "Claim supported by the observation.",
      "provenance_refs": [
        "known-reference"
      ],
      "limitations": []
    }
  ],
  "completed_tasks": [
    "Completed semantic task."
  ],
  "remaining_needs": []
}
~~~

Contract rules:

- status is one of resolved, partial, needs_user_input, approval_required,
  unavailable, budget_exhausted, or failed;
- summary is non-empty and at most 4,000 characters;
- at most 16 evidence rows;
- every observation_id refers to an accepted root-session observation;
- completed_tasks and remaining_needs each contain at most 16 strings;
- resolved has no remaining_needs;
- partial has at least one evidence row and at least one remaining need; and
- code adds session ID and usage after validation.

### AgenticResolverResultV1

The public result contains:

- schema_version;
- session_id;
- status;
- summary;
- validated evidence;
- completed_tasks;
- remaining_needs; and
- code-owned usage.

Usage contains:

- model_steps;
- tool_calls;
- subagent_runs;
- contract_errors;
- compactions;
- estimated_context_tokens_peak; and
- provider usage totals when available.

### ToolDefinition

Each definition contains:

- name;
- description;
- input_schema;
- validate_arguments;
- execute;
- permission_check;
- project_result;
- maximum_result_characters; and
- side_effect_class.

The allowed side-effect classes are:

- read
- compute
- approval_gated

Phase 1 registers no unmediated mutation tool.

### JSON Message Families

The JSON protocol implements these exact schema versions:

- **agentic_resolver_system.v1**
- **agentic_resolver_skill_catalog.v1**
- **agentic_resolver_task.v1**
- **agentic_resolver_tool_observation.v1**
- **agentic_resolver_skill_content.v1**
- **agentic_resolver_subagent_task.v1**
- **agentic_resolver_subagent_result.v1**
- **agentic_resolver_contract_error.v1**
- **agentic_resolver_compacted_observation.v1**

Every resolver-authored non-empty serialized message validates as one object.
Assistant content may be empty when the provider represents a native tool call
without textual content. Any non-empty assistant content must be object-rooted
JSON and carries no resolver control semantics; native tool calls are the
exclusive action channel. Opaque provider reasoning is a distinct assistant
transport field rather than a semantic message. The implementation emits no
XML catalog frame and no free-form text envelope.

### LLM Native Tool Stream Contracts

LLMToolDefinition contains:

- name;
- description; and
- object-rooted parameters JSON Schema.

AgenticModelCapabilitiesV1 contains:

- schema_version;
- streaming, fixed true;
- thinking_enabled, fixed true;
- enabled thinking_strategy identifier; and
- reasoning_replay_policy, owned by the adapter.

AgenticResolverRuntime validates this immutable declaration before creating a
root or child session. A provider-specific strategy payload never enters the
core contract.

LLMToolCall contains:

- provider call ID;
- tool name; and
- decoded JSON object arguments.

LLMInvalidToolCall contains:

- optional provider call ID;
- optional name;
- bounded sanitized error; and
- no raw provider exception.

LLMToolHistoryMessage is role-discriminated. It contains only the fields legal
for its role:

- system and user rows carry resolver-authored JSON content;
- assistant rows carry optional opaque reasoning, empty or JSON content, and
  zero or more native tool calls; and
- tool rows carry one tool-call ID and one JSON result.

LLMStreamChunk is a closed discriminated family:

- block_start with block index and reasoning, text, or tool_call type;
- reasoning_delta with opaque text and block index;
- text_delta with content text and block index;
- tool_call_delta with block index, call ID, optional name, and raw JSON
  argument fragment;
- block_end with one complete normalized block;
- usage with provider-neutral counters; and
- finish with stop, tool_calls, max_tokens, aborted, or error.

The provider leaves incremental tool arguments as raw fragments. The bounded
ModelStreamAssembler correlates block indexes, assembles arguments once,
classifies invalid calls, and discards incomplete tool calls on interruption or
max-token termination. The AgentLoop executes only successfully completed
calls.

**LLInterface.astream_tools** passes native schemas to the provider and yields
LLMStreamChunk values. It does not apply resolver semantics. The
LLInterfaceToolModel maps those chunks to the provider-neutral
AgenticModelClient stream without exposing provider-native objects to the core.

Thinking is mandatory for resolver construction. Reasoning deltas remain
opaque and attach to the assistant turn. The provider adapter replays reasoning
from qualifying assistant tool-call turns through the native reasoning field
when required and omits tool-call-free reasoning when that provider ignores
it. A tool-call turn preserves an empty native reasoning field when the provider
requires field presence. Reasoning never enters JSON content, tool results,
evidence, subagent projections, or public output.

### SkillDefinitionV1

A discovered skill contains:

- name;
- description;
- source path retained in trusted runtime state;
- catalog digest;
- Markdown instruction body; and
- canonical content digest.

Fixed bounds:

- kebab-case name, maximum 64 characters;
- description maximum 500 characters;
- body maximum 16,000 characters;
- maximum 64 skills per runtime; and
- one-level **<name>/SKILL.md** discovery only.

Discovery resolves each candidate and proves that the resolved SKILL.md stays
within its explicit skill root before reading it. Frontmatter is parsed with
**yaml.safe_load** and admits no custom constructors.

Only name, description, and catalog digest enter the startup catalog message.

### RunSubagentV1

Model arguments contain:

- description, maximum 200 characters;
- objective, maximum 4,000 characters;
- context with the same facts, constraints, and desired_output shape as the
  root request.

The controller generates:

- subagent_id;
- parent_session_id;
- child depth;
- permission scope;
- deadlines; and
- usage accounting.

AgenticResolverSubagentResultV1 contains:

- schema_version;
- message_type;
- subagent_id;
- description;
- status;
- summary;
- evidence;
- remaining_needs.

The serialized result supplied to the parent is at most 8,000 characters.
Intermediate child events remain in the child session and outside the parent
model history.

## Runtime And Resource Constraints

| Constraint | Default | Hard maximum | Owner |
|---|---:|---:|---|
| Context window per root or child | 50,000 estimated tokens | 50,000 | ContextBudget |
| Completion reserve | 8,000 tokens | 8,000 | ContextBudget |
| Input ceiling | 42,000 estimated tokens | 42,000 | ContextBudget |
| Model steps per session | 8 | 16 | AgentLoop |
| Non-terminal tool calls per session | 6 | 12 | AgentLoop |
| Structural replacements | 2 | 2 | AgentLoop |
| Root child runs | 3 | 3 | SubagentRunner |
| Model-visible ordinary tool result | 8,000 characters | 8,000 | ToolRegistry |
| Model-visible child result | 8,000 characters | 8,000 | SubagentRunner |
| Skill count | 64 | 64 | SkillCatalog |
| Skill description | 500 characters | 500 | SkillCatalog |
| Skill body | 16,000 characters | 16,000 | SkillCatalog |
| Session wall clock | 300 seconds | 600 | AgenticResolverRuntime |
| Ordinary tool call | 180 seconds | 180 | ToolRegistry |

The effective context window is the smaller of 50,000 and a caller-declared
LLMCallConfig.context_window_tokens value. The input ceiling reserves 8,000
tokens within that effective window.

The deterministic fallback token estimate is
ceiling(canonical serialized character count divided by four), including
assistant reasoning selected for replay and its native serialization fields.
Provider-reported
reasoning tokens are not added again when already included in output tokens.
The stream assembler stops accepting output that exceeds the configured
completion budget and never dispatches a partial tool call.

A child uses the same thinking-enabled stream/replay policy, a fresh
per-session context ledger, and the root session's remaining wall-clock
deadline.

## Existing Kazusa Tool Adapter Contract

build_kazusa_tool_registry validates one
TaskResolutionExecutionContextV1 and binds these functions:

- resolve_with_local_context
- resolve_with_public_research
- resolve_with_coding
- resolve_with_text_computation

Tool schemas:

- local_context: objective
- public_research: objective
- text_computation: objective
- coding: objective and coding_objective_mode, where the mode is read_only or
  propose_patch

The adapter generates task_node_id and trusted_scope. It sets empty
available_evidence and remaining_needs for each independent Phase-1 call.
Non-coding calls use coding_objective_mode **none**.

The result projector copies validated bounded semantic fields from the current
TaskSpecialistResultV1. It may shorten strings and arrays to the agentic
observation caps while retaining:

- specialist;
- status;
- evidence summaries;
- provenance refs;
- evidence limitations;
- completed subgoals;
- remaining needs;
- reason;
- retryable; and
- coding_run_context when present.

The adapter never changes the existing handler source or calls an internal
handler bypass.

## Execution Roles

### Role: standalone_resolver_implementer

- Responsibility: implement the complete Phase-1 package, additive LLInterface
  thinking-enabled tool stream, integration adapters, deterministic tests,
  live test, module documentation, and source-to-test manifest rows.
- Owned surface:
  - **src/agentic_resolver/**
  - the listed LLInterface source files;
  - **pyproject.toml** package/dependency rows;
  - **resolver_skills/.gitkeep**;
  - the listed agentic-resolver tests;
  - **tests/ownership/source_test_impact_manifest.json**;
  - the agentic-resolver addition to **tests/test_test_impact_manifest.py**.
- Authority: create and edit only the listed surface; run deterministic and
  explicitly scoped live-LLM verification; update this plan's progress and
  evidence after execution begins.
- Applicable skills: development-plan, local-llm-architecture,
  no-prepost-user-input, py-style, test-style-and-execution, debug-llm, and
  cjk-safety when triggered.
- Capability floor: senior Python async-generator architecture,
  provider-native tool and reasoning streaming, indexed chunk assembly,
  strict JSON contracts, context accounting, local-LLM prompting, test
  isolation, and repository ownership-boundary verification.
- Independence requirement: none for implementation; this role cannot provide
  final independent sign-off.
- Acceptance output: scoped diff, exact deterministic pytest evidence,
  inspected live-LLM artifact, updated source-to-test manifest, and a complete
  execution-evidence entry.
- Gate:
  - Entry: plan is approved or in_progress, user has explicitly commanded
    implementation, baseline and owned paths are recorded, required skills are
    loaded.
  - Exit: every acceptance criterion is evidenced and the independent review
    has a reviewable diff and test record.

### Role: independent_resolver_reviewer

- Responsibility: independently verify scope, architecture conformance,
  current-workflow isolation, unchanged tool implementations, JSON semantic
  protocol, thinking-enabled native tool streaming, opaque reasoning replay,
  context admission, child non-recursion, deterministic test coverage, and
  live artifact quality.
- Owned surface: read-only access to the implementation diff, tests,
  architecture reference, plan, and execution evidence.
- Authority: report findings and pass or fail final sign-off; no remediation
  edits.
- Applicable skills: development-plan, local-llm-architecture, py-style,
  test-style-and-execution, and debug-llm for live-artifact review.
- Capability floor: independent senior review of async agent loops, native
  tool calling, security/permission boundaries, context budgeting, and exact
  test traceability.
- Independence requirement: executed by an eligible reviewer distinct from
  the implementation/remediation executor.
- Acceptance output: written pass/fail review with every finding tied to a plan
  rule or acceptance criterion.
- Gate:
  - Entry: implementation diff, exact test results, live trace artifact, and
    source-to-test validation are complete.
  - Exit: all material findings are resolved and a separate independent pass
    signs off the final state.

## Test Impact And Traceability

Every node below is deterministic unless the mode says live_llm. Planned test
nodes must exist, collect, and pass before the corresponding source row can be
accepted.

| Source or governed artifact | Changed contract and semantic owner | Required deterministic pytest node IDs | Supplemental node IDs | Mode and regression prevented |
|---|---|---|---|---|
| src/agentic_resolver/__init__.py | Standalone public exports; package boundary owner | tests/test_agentic_resolver_contracts.py::test_public_api_exposes_standalone_runtime_only | none | unit; prevents hidden workflow or internal exports |
| src/agentic_resolver/contracts.py | Strict request, result, limits, observation, and subagent contracts; resolver contract owner | tests/test_agentic_resolver_contracts.py::test_request_and_result_contracts_are_strict_json_objects; tests/test_agentic_resolver_contracts.py::test_submit_result_rejects_unknown_status_and_missing_fields | none | unit; prevents permissive or contradictory public state |
| src/agentic_resolver/json_protocol.py | Canonical JSON-only message serialization; model protocol owner | tests/test_agentic_resolver_json_protocol.py::test_every_model_message_serializes_to_one_json_object; tests/test_agentic_resolver_json_protocol.py::test_model_protocol_contains_no_xml_catalog_or_freeform_envelopes; tests/test_agentic_resolver_json_protocol.py::test_contract_error_and_compaction_messages_are_json | none | unit; prevents XML/free-text control drift |
| src/agentic_resolver/model.py | Provider-neutral thinking-enabled native-tool stream and capability declaration; model seam owner | tests/test_agentic_resolver_contracts.py::test_agentic_model_client_requires_native_tool_chunk_stream; tests/test_agentic_resolver_contracts.py::test_agentic_model_client_requires_enabled_thinking_capabilities | none | unit; prevents provider objects, unverifiable thinking, and non-streaming shortcuts leaking into the core |
| src/agentic_resolver/streaming.py | Indexed chunk assembly and complete-turn validation; stream owner | tests/test_agentic_resolver_streaming.py::test_stream_assembler_reconstructs_reasoning_text_and_one_tool_call; tests/test_agentic_resolver_streaming.py::test_stream_assembler_rejects_malformed_block_order; tests/test_agentic_resolver_streaming.py::test_stream_assembler_never_exposes_partial_tool_call | none | unit; prevents malformed or partial stream state from reaching execution |
| src/agentic_resolver/session.py | Append-only stream events and reasoning-aware history derivation; session owner | tests/test_agentic_resolver_session.py::test_session_log_records_chunks_and_reconstructs_assistant_history; tests/test_agentic_resolver_session.py::test_compaction_preserves_reasoning_tool_call_and_result_atomically | none | unit; prevents non-replayable history, lost reasoning passback, and orphaned observations |
| src/agentic_resolver/context_budget.py | 50k cap, 8k reserve, reasoning-aware accounting, and compaction; context owner | tests/test_agentic_resolver_context_budget.py::test_context_meter_counts_system_catalog_tools_history_and_reserved_completion; tests/test_agentic_resolver_context_budget.py::test_context_meter_counts_retained_reasoning_and_provider_replay_once; tests/test_agentic_resolver_context_budget.py::test_context_meter_compacts_old_tool_results_before_hard_stop; tests/test_agentic_resolver_context_budget.py::test_context_cap_returns_budget_exhausted_without_over_limit_model_call | none | unit; prevents over-cap provider requests and double-counted reasoning |
| src/agentic_resolver/tools.py | Frozen schemas, reserved names, validation, permissions, and bounded errors; tool registry owner | tests/test_agentic_resolver_tools.py::test_registry_freezes_sorted_unique_json_schemas; tests/test_agentic_resolver_tools.py::test_registry_validates_arguments_before_execution; tests/test_agentic_resolver_tools.py::test_tool_exception_becomes_bounded_json_error | none | unit; prevents shadowing, invalid execution, and exception leakage |
| src/agentic_resolver/skills.py | One-level discovery, immutable JSON catalog, safe YAML, resolved-root containment, digest, and lazy load; skill owner | tests/test_agentic_resolver_skills.py::test_startup_scan_discovers_one_level_skill_bundles; tests/test_agentic_resolver_skills.py::test_catalog_injection_is_json_name_description_only; tests/test_agentic_resolver_skills.py::test_skill_tool_loads_full_body_lazily; tests/test_agentic_resolver_skills.py::test_malformed_duplicate_or_oversized_skills_fail_startup; tests/test_agentic_resolver_skills.py::test_skill_discovery_rejects_symlink_escape; tests/test_agentic_resolver_skills.py::test_skill_frontmatter_uses_safe_yaml_loader | none | unit; prevents eager prompt bloat, path escape, unsafe YAML, and malformed skill admission |
| src/agentic_resolver/subagents.py | Same-runtime isolated child with depth-one registry, inherited thinking stream, and bounded return; subagent owner | tests/test_agentic_resolver_subagents.py::test_run_subagent_uses_same_runtime_with_isolated_session; tests/test_agentic_resolver_subagents.py::test_child_registry_excludes_run_subagent; tests/test_agentic_resolver_subagents.py::test_child_inherits_tools_skills_permissions_json_and_thinking_stream; tests/test_agentic_resolver_subagents.py::test_parent_receives_only_bounded_typed_child_result; tests/test_agentic_resolver_subagents.py::test_subagent_run_cap_fails_closed | tests/test_agentic_resolver_live_llm.py::test_standalone_resolver_streams_thinking_tool_and_subagent_then_submits_json_result | unit plus live_llm; prevents recursive delegation, transcript leakage, divergent child transport, and unbounded convergence |
| src/agentic_resolver/loop.py | Serialized chunk-stream loop, complete one-call steps, replacements, and terminal tool; loop owner | tests/test_agentic_resolver_loop.py::test_loop_consumes_stream_before_executing_one_complete_native_tool_call; tests/test_agentic_resolver_loop.py::test_loop_does_not_execute_interrupted_or_partial_tool_call; tests/test_agentic_resolver_loop.py::test_loop_rejects_zero_or_multiple_tool_calls_with_bounded_json_feedback; tests/test_agentic_resolver_loop.py::test_loop_stops_at_step_tool_and_contract_caps; tests/test_agentic_resolver_loop.py::test_parent_converges_multiple_subagent_results_into_terminal_result | tests/test_agentic_resolver_live_llm.py::test_standalone_resolver_streams_thinking_tool_and_subagent_then_submits_json_result | unit plus live_llm; prevents partial execution, free-text routing, and unbounded recurrence |
| src/agentic_resolver/runtime.py | Direct construction, mandatory supported thinking, session deadline, root result, and no global registration; runtime owner | tests/test_agentic_resolver_standalone.py::test_runtime_resolves_without_brain_service_or_cognition_imports; tests/test_agentic_resolver_standalone.py::test_runtime_factory_requires_explicit_model_tools_and_skill_roots; tests/test_agentic_resolver_standalone.py::test_runtime_rejects_disabled_or_unsupported_thinking; tests/test_agentic_resolver_standalone.py::test_current_workflow_sources_do_not_import_agentic_resolver | tests/test_agentic_resolver_live_llm.py::test_standalone_resolver_streams_thinking_tool_and_subagent_then_submits_json_result | unit plus live_llm; prevents silent thinking downgrade and accidental workflow wiring |
| src/agentic_resolver/integrations/__init__.py | Explicit optional integration exports; integration boundary owner | tests/test_agentic_resolver_standalone.py::test_core_modules_keep_kazusa_imports_inside_integrations | none | unit; prevents Kazusa imports leaking into the core |
| src/agentic_resolver/integrations/llm_interface.py | LLInterface-to-AgenticModelClient stream mapping, mandatory thinking, and opaque provider-neutral reasoning history; model integration owner | tests/test_llm_interface_tool_stream.py::test_agentic_adapter_maps_reasoning_and_tool_chunks_without_provider_objects; tests/test_llm_interface_tool_stream.py::test_agentic_adapter_preserves_reasoning_as_typed_assistant_history; tests/test_llm_interface_tool_stream.py::test_agentic_adapter_requires_supported_thinking_config | none | unit; prevents provider-native shape leakage, reasoning promotion into JSON, and silent thinking downgrade |
| src/agentic_resolver/integrations/kazusa_tools.py | Four unchanged specialist adapters; capability integration owner | tests/test_agentic_resolver_kazusa_tools.py::test_kazusa_registry_exposes_four_existing_specialists; tests/test_agentic_resolver_kazusa_tools.py::test_kazusa_adapters_call_existing_handlers_without_modifying_contracts; tests/test_agentic_resolver_kazusa_tools.py::test_public_research_tool_retains_existing_web_agent_ownership; tests/test_agentic_resolver_kazusa_tools.py::test_coding_tool_retains_existing_approval_lifecycle | none | unit; prevents bypassing current public capability boundaries |
| src/kazusa_ai_chatbot/llm_interface/contracts.py | Additive normalized history, native-tool chunk, finish, and stream protocol types; LLInterface contract owner | tests/test_llm_interface_tool_stream.py::test_tool_stream_contracts_keep_reasoning_distinct_from_json_content; tests/test_llm_interface_tool_stream.py::test_tool_call_deltas_are_indexed_and_arguments_remain_raw_until_assembly; tests/test_llm_interface_tool_stream.py::test_existing_ainvoke_contract_remains_unchanged | tests/test_llm_interface_contracts.py::test_llm_response_wraps_content_backend_raw_response_and_usage | unit; prevents reasoning/content conflation and ordinary response breakage |
| src/kazusa_ai_chatbot/llm_interface/interface.py | Additive **astream_tools** dispatch; LLInterface public API owner | tests/test_llm_interface_tool_stream.py::test_astream_tools_preserves_reasoning_tool_arguments_and_usage; tests/test_llm_interface_tool_stream.py::test_existing_ainvoke_contract_remains_unchanged | tests/test_llm_interface_contracts.py::test_backend_descriptor_cache_is_per_interface_and_invalidated | unit; prevents stream calls changing ordinary dispatch |
| src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py | OpenAI-compatible native-tool binding, reasoning streaming/passback policy, and schema cache digest; provider owner | tests/test_llm_interface_tool_stream.py::test_astream_tools_uses_distinct_tool_schema_cache_key; tests/test_llm_interface_tool_stream.py::test_astream_tools_replays_tool_call_reasoning_and_drops_ignored_tool_free_reasoning; tests/test_llm_interface_tool_stream.py::test_astream_tools_preserves_required_empty_reasoning_field_for_tool_call_turn; tests/test_llm_interface_tool_stream.py::test_astream_tools_does_not_set_json_response_format | tests/test_llm_interface_openai_provider.py::test_provider_maps_config_to_chat_model_constructor; tests/test_llm_interface_openai_provider.py::test_provider_retries_unsupported_json_object_with_json_schema | unit; prevents reasoning loss or excess replay, missing required field presence, tool-schema cache collision, and ordinary-provider regression |
| src/kazusa_ai_chatbot/llm_interface/reload.py | Stream-safe confirmed-unload retry before first output only; reload owner | tests/test_llm_interface_reload.py::test_astream_retries_confirmed_unload_before_first_chunk; tests/test_llm_interface_reload.py::test_astream_never_retries_after_first_emitted_chunk | tests/test_llm_interface_reload.py::test_async_unload_error_retries_same_call_once | unit; prevents duplicated partial streams and preserves existing retry behavior |
| src/kazusa_ai_chatbot/llm_interface/__init__.py | Additive public history and native-tool stream exports; LLInterface package owner | tests/test_llm_interface_tool_stream.py::test_native_tool_stream_contracts_are_public_exports | none | unit; prevents inaccessible or accidental exports |
| pyproject.toml | Package discovery and explicit PyYAML dependency; packaging owner | tests/test_agentic_resolver_standalone.py::test_distribution_discovers_agentic_resolver_package; tests/test_agentic_resolver_skills.py::test_yaml_frontmatter_dependency_is_declared | none | unit; prevents source-only package and undeclared parser dependency |
| src/agentic_resolver/README.md | Module ICD for public boundary, JSON protocol, thinking stream, tools, skills, children, and tests; documentation owner | tests/test_agentic_resolver_architecture_docs.py::test_module_readme_documents_public_runtime_and_forbidden_workflow_edges; tests/test_agentic_resolver_architecture_docs.py::test_module_readme_requires_thinking_stream_and_opaque_reasoning | none | documentation unit; prevents contract documentation drift |
| src/kazusa_ai_chatbot/llm_interface/README.md | Additive thinking-enabled native-tool stream ICD and preserved ordinary call contract; LLInterface documentation owner | tests/test_agentic_resolver_architecture_docs.py::test_llm_interface_readme_documents_additive_native_tool_stream_contract | none | documentation unit; prevents undocumented shared-interface change |
| docs/architecture/agentic_resolver_architecture.md | Governing standalone, JSON, thinking stream, subagent, and deferred-bigbang decisions; architecture owner | tests/test_agentic_resolver_architecture_docs.py::test_architecture_declares_standalone_first_pass_and_deferred_bigbang; tests/test_agentic_resolver_architecture_docs.py::test_architecture_requires_json_for_resolver_authored_semantic_envelopes; tests/test_agentic_resolver_architecture_docs.py::test_architecture_declares_opaque_reasoning_replay_and_atomic_compaction; tests/test_agentic_resolver_architecture_docs.py::test_architecture_declares_non_recursive_same_runtime_subagent | none | documentation unit; prevents implementation direction drift |
| tests/ownership/source_test_impact_manifest.json | Exact agentic-resolver and LLInterface owner mappings; verification owner | tests/test_test_impact_manifest.py::test_manifest_contains_agentic_resolver_owner_rows; tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary | none | test-infrastructure unit; prevents unmapped production changes |

### Existing Behavior Preservation Nodes

These existing nodes supplement the exact owner matrix:

- tests/test_task_resolution_orchestrator.py::test_wrong_text_selection_reroutes_to_public_research
- tests/test_task_resolution_inline_promotion.py::test_inline_budget_default_is_thirty_seconds
- tests/unit/cognition_resolver/test_loop.py::test_loop_exposes_owned_contract
- tests/test_llm_interface_contracts.py::test_call_config_defaults_to_json_object_output
- tests/test_llm_interface_contracts.py::test_llm_response_strips_complete_qwen_think_tag
- tests/test_llm_interface_openai_provider.py::test_provider_async_path_preserves_message_objects

## Change Surface

### Create

- **docs/architecture/agentic_resolver_architecture.md**
  - Governing architecture reference; created during the planning pass and
    kept authoritative during implementation.
- **src/agentic_resolver/__init__.py**
  - Public standalone exports.
- **src/agentic_resolver/README.md**
  - Module ICD.
- **src/agentic_resolver/contracts.py**
  - Strict public and internal typed contracts.
- **src/agentic_resolver/json_protocol.py**
  - Canonical JSON message builders and validators.
- **src/agentic_resolver/model.py**
  - Provider-neutral thinking-enabled native-tool stream protocol.
- **src/agentic_resolver/streaming.py**
  - Indexed stream assembly and complete-turn validation.
- **src/agentic_resolver/session.py**
  - Append-only stream events and reasoning-aware history projection.
- **src/agentic_resolver/context_budget.py**
  - Context meter, fixed cap, reserve, and deterministic compaction.
- **src/agentic_resolver/tools.py**
  - Tool definitions, registry, validation, permission checks, execution, and
    bounded observations.
- **src/agentic_resolver/skills.py**
  - Skill discovery, YAML frontmatter validation, catalog digest, JSON
    catalog, and lazy loader.
- **src/agentic_resolver/subagents.py**
  - Same-runtime child construction, depth enforcement, child result
    projection, and child count.
- **src/agentic_resolver/loop.py**
  - Serialized native tool loop and terminal handling.
- **src/agentic_resolver/runtime.py**
  - Public runtime construction and resolve lifecycle.
- **src/agentic_resolver/integrations/__init__.py**
  - Optional integration exports.
- **src/agentic_resolver/integrations/llm_interface.py**
  - LLInterfaceToolModel adapter.
- **src/agentic_resolver/integrations/kazusa_tools.py**
  - Existing specialist tool-registry builder.
- **resolver_skills/.gitkeep**
  - Empty project-local external skill-root convention.
- **tests/test_agentic_resolver_contracts.py**
- **tests/test_agentic_resolver_json_protocol.py**
- **tests/test_agentic_resolver_streaming.py**
- **tests/test_agentic_resolver_session.py**
- **tests/test_agentic_resolver_context_budget.py**
- **tests/test_agentic_resolver_tools.py**
- **tests/test_agentic_resolver_skills.py**
- **tests/test_agentic_resolver_subagents.py**
- **tests/test_agentic_resolver_loop.py**
- **tests/test_agentic_resolver_standalone.py**
- **tests/test_agentic_resolver_kazusa_tools.py**
- **tests/test_llm_interface_tool_stream.py**
- **tests/test_agentic_resolver_architecture_docs.py**
- **tests/test_agentic_resolver_live_llm.py**

### Modify

- **pyproject.toml**
  - Add **agentic_resolver*** package discovery and PyYAML dependency.
- **src/kazusa_ai_chatbot/llm_interface/contracts.py**
  - Add separate history, native-tool chunk, finish, and stream contracts.
- **src/kazusa_ai_chatbot/llm_interface/interface.py**
  - Add async native-tool streaming.
- **src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py**
  - Add native tool binding, reasoning streaming/passback, and schema-digest
    cache partitioning.
- **src/kazusa_ai_chatbot/llm_interface/reload.py**
  - Add stream-safe confirmed-unload recovery before first output only.
- **src/kazusa_ai_chatbot/llm_interface/__init__.py**
  - Export additive contracts.
- **src/kazusa_ai_chatbot/llm_interface/README.md**
  - Document the new thinking-enabled stream and preserved existing behavior.
- **tests/test_llm_interface_reload.py**
  - Verify pre-output retry and post-output no-retry behavior.
- **tests/ownership/source_test_impact_manifest.json**
  - Register every changed production owner and exact unit nodes.
- **tests/test_test_impact_manifest.py**
  - Assert the complete agentic-resolver mapping.
- **development_plans/README.md**
  - Register this draft, then track lifecycle changes.
- This plan file
  - Record approved status, progress, execution evidence, review, and closure
    only when those lifecycle events occur.

### Delete

- None.

### Keep

- **src/kazusa_ai_chatbot/cognition_resolver/**
- **src/kazusa_ai_chatbot/cognition_core_v3/**
- **src/kazusa_ai_chatbot/nodes/**
- **src/kazusa_ai_chatbot/task_resolution/**
- **src/kazusa_ai_chatbot/local_context_resolver/**
- **src/kazusa_ai_chatbot/complex_task_resolver/**
- **src/kazusa_ai_chatbot/rag/web_agent3/**
- **src/kazusa_ai_chatbot/coding_agent/**
- **src/kazusa_ai_chatbot/accepted_task/**
- **src/kazusa_ai_chatbot/background_work/**
- **src/kazusa_ai_chatbot/action_spec/**
- **src/kazusa_ai_chatbot/brain_service/**
- **src/kazusa_ai_chatbot/db/**
- **src/kazusa_ai_chatbot/config.py**
- **src/kazusa_ai_chatbot/service.py**
- **README.md**
- **README_CN.md**
- **docs/HOWTO.md**

These paths remain outside the Phase-1 implementation diff, except that the
new integration imports the four named specialist handler functions.

## Agent Autonomy Boundaries

The implementation owner may choose:

- private helper decomposition inside the listed new modules;
- dataclass, TypedDict, Protocol, and validator mechanics consistent with
  project style;
- internal event class names while preserving the documented event families;
- the exact canonical JSON serialization helper implementation;
- deterministic ID generation mechanics;
- test fixture organization;
- the internal schema-digest implementation; and
- logging mechanics that preserve the public sanitization contract.

The implementation owner requires a plan amendment or user decision before:

- editing any Keep path;
- adding a current workflow import or call;
- adding another Kazusa tool;
- exposing direct low-level web tools;
- changing a fixed status, limit, schema version, public field, or tool name;
- changing child depth, execution mode, inherited capabilities, or context
  isolation;
- adding persistence, resume, background work, parallel execution, a route, or
  UI;
- adding a compatibility layer or fallback;
- changing existing LLInterface behavior;
- changing an existing tool implementation;
- adding a resolver-authored semantic format other than JSON or promoting
  opaque reasoning into semantic JSON; or
- expanding external effects.

If source reality conflicts with the fixed contracts, stop the affected work,
record evidence, and request an amendment. Preserve completed in-scope work
without silently broadening the plan.

## Implementation Order

### Work Item 0: Baseline And Ownership Lock

- Capture git status and all changed paths.
- Record the explicitly owned file set.
- Confirm every Keep path is excluded from edits.
- Confirm all exact planned pytest node names remain available for creation.
- Record runtime executor resolution and effective non-secret configuration.

Exit gate: baseline and owned surface are recorded in Execution Evidence.

### Work Item 1: Additive LLInterface Thinking Tool Stream

- Add native-tool history, chunk, finish, invoker, and public export contracts.
- Add **astream_tools** as the sole resolver model path.
- Bind OpenAI-compatible schemas without JSON-object response_format.
- Normalize reasoning, text, valid tool-call, invalid tool-call, usage, and
  finish chunks without exposing provider-native objects.
- Preserve assistant reasoning as typed history, replay tool-call-turn
  reasoning through the provider-native field when required, and omit
  tool-call-free reasoning when the provider ignores it. Preserve the empty
  native reasoning field for tool-call turns when the provider requires field
  presence.
- Require enabled, supported thinking in the resolver integration while
  preserving every existing LLInterface caller contract.
- Publish the validated state through AgenticModelCapabilitiesV1 for runtime
  admission.
- Retry a confirmed unload only before the first stream chunk is emitted.
- Partition tool-bound model cache entries by canonical schema digest.
- Preserve ordinary call behavior.
- Add and run exact LLInterface tool-stream and reload tests plus preservation
  nodes.

Exit gate: the normalized native stream, reasoning replay, and pre-output
reload recovery work through fakes, and every named ordinary LLInterface
preservation node passes.

### Work Item 2: Core Contracts, JSON Protocol, Session, And Budgets

- Create package contracts and public exports.
- Implement canonical JSON-only messages.
- Implement the indexed ModelStreamAssembler and reject malformed or
  incomplete turns before execution.
- Implement append-only stream events and reasoning-aware session history.
- Implement context measurement, single-count reasoning/replay accounting,
  atomic reasoning-call-result compaction, and hard stop.
- Add exact deterministic owner tests.

Exit gate: JSON and context tests prove that an over-cap request never reaches
the fake model.

### Work Item 3: Tool And Skill Composition

- Implement frozen ToolRegistry.
- Implement reserved core tools.
- Implement one-level SKILL.md discovery.
- Enforce resolved-root containment and safe YAML frontmatter parsing.
- Build and inject the immutable JSON catalog.
- Implement lazy skill loading.
- Add exact deterministic tool and skill tests.

Exit gate: tool and skill contracts are frozen, bounded, JSON-only, and
startup-fail-fast.

### Work Item 4: Same-Runtime Subagents

- Implement SubagentRunner.
- Construct children through AgenticResolverRuntime.
- Give children isolated sessions and explicit tasks.
- Preserve ordinary tools, skills, the mandatory thinking-enabled stream,
  native reasoning replay, JSON protocol, permissions, and budgets.
- Remove **run_subagent** from the child registry view.
- Bound the child result returned to the parent.
- Enforce the three-child root cap.
- Add exact deterministic subagent tests.

Exit gate: tests prove identical runtime classes, isolated history, inherited
capabilities, absent recursion tool, bounded convergence, and hard child cap.

### Work Item 5: Loop, Runtime, And Kazusa Adapters

- Implement the serialized native chunk-stream loop.
- Assemble and validate the complete assistant turn before dispatch.
- Require one tool call per step.
- Require **submit_result** for normal terminalization.
- Implement direct runtime construction.
- Add LLInterface and four-specialist integration builders.
- Add isolation tests proving no current workflow dependency.
- Add packaging and module-ICD checks.

Exit gate: one deterministic root workflow uses an ordinary tool, one child,
and submit_result while all current workflow source remains unchanged.

### Work Item 6: Manifest, Documentation, And Full Deterministic Verification

- Update the source-to-test manifest.
- Add its explicit owner-row assertion.
- Update LLInterface and module ICDs.
- Collect every exact node in the traceability table.
- Run every required deterministic node and preservation node.
- Run the changed-source impact validator.
- Run diff and import-boundary checks.

Exit gate: every mapped node collects and passes; the diff stays inside the
declared surface.

### Work Item 7: Live LLM Gate And Final Structured Review

- Run the one live-LLM node by itself.
- Preserve the debug-LLM artifact.
- Inspect stream ordering and completion, reasoning presence, tool-call-turn
  native passback, tool-call-free omission where applicable, tool choice, JSON
  messages, child task isolation, child tool roster, child result, parent
  convergence, and submit_result quality without retaining thought text in the
  review artifact.
- Apply the complete independent-code-review checklist as a final structured
  parent review under the explicitly recorded reviewer waiver.
- Record all findings, remediation outcomes, residual risks, and the explicit
  quality decision without claiming a separate independent-agent review.

Exit gate: live artifact receives an explicit quality pass and the final
structured parent review records no unresolved material finding.

## Verification

### Collection Gate

Use pytest collection on every new exact node listed in Test Impact And
Traceability. A missing or renamed node fails the gate.

### Deterministic Gate

Run all new non-live test modules in one regular deterministic batch:

~~~text
tests/test_agentic_resolver_contracts.py
tests/test_agentic_resolver_json_protocol.py
tests/test_agentic_resolver_streaming.py
tests/test_agentic_resolver_session.py
tests/test_agentic_resolver_context_budget.py
tests/test_agentic_resolver_tools.py
tests/test_agentic_resolver_skills.py
tests/test_agentic_resolver_subagents.py
tests/test_agentic_resolver_loop.py
tests/test_agentic_resolver_standalone.py
tests/test_agentic_resolver_kazusa_tools.py
tests/test_llm_interface_tool_stream.py
tests/test_llm_interface_reload.py
tests/test_agentic_resolver_architecture_docs.py
tests/test_test_impact_manifest.py
~~~

Run every named Existing Behavior Preservation Node.

Run the source-to-test impact validator against the actual changed production
paths and confirm every path resolves to collected required unit nodes.

### Static Boundary Gate

Verify:

- package discovery includes **agentic_resolver**;
- core agentic_resolver modules import no Kazusa workflow package;
- current workflow sources contain no **agentic_resolver** import;
- existing tool implementation files are absent from the diff;
- current workflow files are absent from the diff;
- every resolver-authored semantic message constructor returns parseable
  object-rooted JSON;
- source contains no XML-style skill catalog renderer;
- no model-facing assistant prose parser exists;
- resolver composition requires provider thinking to be enabled and supported;
- every root and child model call uses **astream_tools**, with no
  **ainvoke_tools** or other non-streaming resolver path;
- opaque reasoning is carried only in typed assistant history and
  provider-native passback fields, and is absent from semantic JSON, tool
  results, public results, and ordinary logs;
- incomplete or interrupted tool-call streams cannot reach dispatch;
- reasoning, assistant tool calls, and corresponding tool results compact as
  one atomic history unit;
- confirmed-unload retry occurs only before the first emitted chunk;
- every public type has an explicit schema version where applicable; and
- **git diff --check** passes.

### Live LLM Gate

Run only:

~~~text
tests/test_agentic_resolver_live_llm.py::test_standalone_resolver_streams_thinking_tool_and_subagent_then_submits_json_result
~~~

The case uses:

- a real configured model route supplied explicitly to the test;
- provider thinking enabled with a detected supported strategy;
- bounded deterministic fake ordinary tools;
- one discoverable fixture SKILL.md;
- normalized reasoning and tool-call stream events;
- one parent ordinary-tool call;
- one **run_subagent** call;
- a child ordinary-tool call;
- no child **run_subagent** schema;
- one bounded child result;
- parent convergence; and
- one valid **submit_result** call.

Inspect the debug artifact for:

- every input message parses as JSON;
- every root and child model step uses the stream interface;
- reasoning events precede or accompany the assembled assistant tool-call turn
  according to the provider stream contract;
- reasoning from a tool-calling assistant turn is present in the next provider
  history request when the backend requires passback, while tool-call-free
  reasoning is absent when that backend ignores it;
- the model uses native tool calls;
- the skill catalog contains summaries only;
- the child receives a self-contained task rather than parent history;
- the child tool roster excludes **run_subagent**;
- the child output is bounded and typed;
- the parent uses the child result coherently;
- no thought text appears in the persisted debug artifact, semantic JSON,
  tool result, or public resolver result;
- no tool or permission boundary is bypassed; and
- terminal quality is materially adequate for the supplied task.

Run no live DB test in this plan.

## Acceptance Criteria

1. **agentic_resolver** is installable from the project package configuration.
2. A caller can construct and invoke it without importing or starting the
   brain service or cognition.
3. No current workflow source imports or invokes **agentic_resolver**.
4. No existing tool implementation file changes.
5. Existing task-resolution and cognition preservation nodes pass unchanged.
6. Existing LLInterface ordinary-call preservation nodes pass unchanged.
7. LLInterface exposes a separate working async native-tool stream through
   **astream_tools**.
8. Every root and child model step consumes that stream, with no non-streaming
   resolver model path.
9. The public standalone **resolve** call still returns one terminal typed
   result; Phase 1 does not expose caller-facing token streaming.
10. Reasoning, text, indexed tool-call deltas, usage, and finish state normalize
    into a closed provider-neutral chunk family.
11. Tool-bound provider cache identity includes the tool-schema digest.
12. Every non-empty resolver-authored semantic textual payload parses as
    exactly one JSON object.
13. Resolver construction validates an immutable capability declaration that
    streaming is active and thinking is enabled through a supported backend
    strategy.
14. Reasoning remains separate opaque assistant state and never becomes
    semantic JSON, evidence, an action decision, or a permission decision.
15. Reasoning from qualifying assistant tool-call turns is replayed through the
    provider-native assistant reasoning field whenever the backend requires
    passback, including an empty field when required; tool-call-free reasoning
    is omitted when that backend ignores it.
16. Thought text is absent from tool results, child results, public resolver
    results, persistence, and ordinary logs.
17. Malformed, interrupted, or max-token streams with partial tool calls never
    reach tool dispatch.
18. A confirmed model-unload stream retries only before its first emitted
    chunk and never restarts a partially emitted stream.
19. The model control path accepts native tool calls only.
20. Exactly one complete tool call is accepted per step.
21. **submit_result** is required for normal completion.
22. Structural errors consume the fixed two-replacement budget and terminate
    predictably.
23. The four initial Kazusa tools call the current handlers through their
    validated contracts.
24. Existing public-research WebAgent3 ownership remains intact.
25. Existing coding approval and lifecycle ownership remains intact.
26. Skill discovery accepts one-level bundles and fails startup for invalid
    catalogs.
27. Skill discovery contains resolved paths within their explicit roots and
    parses frontmatter with safe YAML loading and no custom constructors.
28. The JSON skill catalog includes only name, description, schema metadata,
    and digest.
29. Full skill bodies enter model context only after a **skill** tool call.
30. Skills cannot expand tools, permissions, limits, or external effects.
31. Context accounting includes policy, schemas, catalog, task, transcript,
    reasoning selected for provider replay, native replay fields, skills,
    observations, children, and output reserve without double-counting
    provider-reported reasoning.
32. Root and child model requests remain within the effective 50,000-token
    ceiling.
33. Compaction retains or removes each reasoning, assistant tool-call, and
    corresponding tool-result exchange atomically while preserving evidence
    references.
34. Residual overflow returns **budget_exhausted** before a model call.
35. **run_subagent** is available to the root as a first-class core tool.
36. Each child is created through the same runtime, loop, and stream assembler
    implementations.
37. Each child has an isolated session and explicit self-contained task.
38. Each child retains the same ordinary tools, skills, thinking-enabled model
    stream, reasoning replay policy, permission scope, JSON protocol, and
    context policy.
39. Each child registry excludes **run_subagent**.
40. Root child execution stops after three children.
41. The parent receives a bounded typed child result and no child transcript or
    thought text.
42. Parent convergence works across multiple child results in deterministic
    testing.
43. Every changed production path has a collected exact source-to-test
    mapping.
44. The one live-LLM case passes streaming, thinking, tool, child, JSON, and
    privacy gates and receives an explicit debug-LLM quality pass.
45. Independent review reports no unresolved material finding.
46. The plan records verification and review evidence before completion.
47. Future big-bang integration remains absent from the implementation diff.

## Independent Code Review

The independent reviewer checks:

- diff containment against Create, Modify, and Keep;
- the downward-only integration dependency;
- unchanged current workflow and tool source;
- public contract strictness;
- JSON-only message construction;
- absence of free-text action parsing;
- native provider reasoning and tool-call stream normalization;
- complete-turn assembly before dispatch;
- mandatory thinking capability detection and no silent downgrade;
- native reasoning passback without semantic promotion or public exposure;
- stream-safe unload retry boundaries;
- reasoning-aware context accounting and atomic compaction;
- exception and secret sanitization;
- child construction through the same runtime;
- child registry removal of **run_subagent**;
- permission non-expansion;
- bounded child return;
- exact source-to-test mappings;
- deterministic result evidence;
- live artifact stream quality and thought-text exclusion; and
- residual risks relevant to the future big-bang plan.

Review findings remain separate from remediation. Any remediation receives a
fresh independent review before sign-off.

## Progress Checklist

- [x] Work Item 0: baseline and ownership lock.
- [x] Work Item 1: additive LLInterface thinking tool stream.
- [x] Work Item 2: core contracts, JSON protocol, session, and budgets.
- [x] Work Item 3: tool and skill composition.
- [x] Work Item 4: same-runtime subagents.
- [x] Work Item 5: loop, runtime, and Kazusa adapters.
- [x] Work Item 6: manifest, documentation, and deterministic verification.
- [x] Work Item 7: live LLM gate and final structured review.
- [x] Acceptance criteria evidenced.
- [x] Registry and lifecycle status updated.

## Execution Evidence

### 2026-08-23 Work Item 0: Baseline And Ownership Lock

- Lifecycle decision: the user approved and explicitly commanded execution on
  2026-08-23. The plan moved from `draft` to `in_progress` before production
  source edits.
- Baseline: `git status --short` and `git diff --name-only HEAD` were empty.
  The execution diff therefore begins from a clean tracked and untracked
  worktree.
- Execution-constraint history: the initial user instruction required
  parent-only execution for the duration of the plan. A later explicit user
  supersession required exactly one `gpt-5.6-luna` child at max/normal speed
  for code changes and tests. The parent Codex agent retained coordination,
  verification, evidence, remediation, and final structured review. No
  separate reviewer was authorized; the user explicitly waived the distinct
  independent-review role while retaining the complete checklist and evidence
  gate. The parent applied the full checklist and found no unresolved material
  finding.
- Executor resolution: `standalone_resolver_implementer` resolves to the parent
  Codex agent with repository-local filesystem and shell access, `apply_patch`,
  the project virtual environment, and the mandatory development-plan,
  local-LLM-architecture, no-prepost-user-input, py-style,
  test-style-and-execution, and debug-LLM skills loaded. Selection mode is the
  user-supplied fixed execution constraint. The same parent will perform a
  fresh structured review after implementation and live evidence are complete.
- Exact owned production and governed-artifact surface:
  `src/agentic_resolver/__init__.py`, `src/agentic_resolver/README.md`,
  `src/agentic_resolver/contracts.py`, `src/agentic_resolver/json_protocol.py`,
  `src/agentic_resolver/model.py`, `src/agentic_resolver/streaming.py`,
  `src/agentic_resolver/runtime.py`, `src/agentic_resolver/loop.py`,
  `src/agentic_resolver/session.py`, `src/agentic_resolver/context_budget.py`,
  `src/agentic_resolver/tools.py`, `src/agentic_resolver/skills.py`,
  `src/agentic_resolver/subagents.py`,
  `src/agentic_resolver/integrations/__init__.py`,
  `src/agentic_resolver/integrations/llm_interface.py`,
  `src/agentic_resolver/integrations/kazusa_tools.py`,
  `src/kazusa_ai_chatbot/llm_interface/contracts.py`,
  `src/kazusa_ai_chatbot/llm_interface/interface.py`,
  `src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py`,
  `src/kazusa_ai_chatbot/llm_interface/reload.py`,
  `src/kazusa_ai_chatbot/llm_interface/__init__.py`,
  `src/kazusa_ai_chatbot/llm_interface/README.md`, `pyproject.toml`,
  `resolver_skills/.gitkeep`, the exact test files listed by Change Surface,
  `tests/test_llm_interface_reload.py`,
  `tests/ownership/source_test_impact_manifest.json`,
  `tests/test_test_impact_manifest.py`,
  `docs/architecture/agentic_resolver_architecture.md`, this plan record, and
  `development_plans/README.md`.
- Excluded surface: every Keep path and every current specialist implementation
  remains outside the implementation diff. Optional integration modules may
  import the four named specialist public handlers and contracts.
- Baseline verification: the clean baseline command
  `venv\Scripts\python.exe -m pytest tests\test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary -q`
  failed because 38 pre-existing manifest entries have empty required-unit
  mappings. The manifest is an explicitly owned governed artifact and the
  final manifest gate remains mandatory; execution evidence will distinguish
  this baseline condition from new owner rows and record its disposition.
- Completed outcome: lifecycle authority, baseline, owned paths, exclusions,
  effective skills, executor configuration, and the first known verification
  condition are recorded. The remaining work items and final acceptance
  evidence are recorded below.

### 2026-08-23 Work Item 1: Additive LLInterface Thinking Tool Stream

- Added the additive `LLInterface.astream_tools` contract with typed native
  tool history, normalized stream chunks, finish state, usage, and public
  exports. Resolver model steps use this stream exclusively.
- Preserved ordinary `ainvoke`, `invoke`, output-mode, JSON-schema fallback,
  thinking, reload, cache, response, and usage behavior.
- Added stream-safe confirmed-unload retry before the first emitted chunk and
  canonical tool-schema digest cache partitioning.
- Implemented Qwen thinking transport for the endorsed route: enabled
  thinking, normalized/replayed `reasoning_content`, `tool_choice="required"`,
  `parallel_tool_calls=False`, `reasoning_format="deepseek"`, and removal of
  the legacy native-tool Qwen prefill. Non-Qwen and ordinary Qwen routes retain
  their existing behavior.
- Focused transport, provider, reload, and ordinary-preservation tests passed.

### 2026-08-23 Work Item 2: Core Contracts, JSON Protocol, Session, And Budgets

- Added strict versioned request, result, limits, observation, usage, model,
  and error contracts with mandatory enabled-thinking capability admission.
- Added canonical object-rooted JSON policy, catalog, task, skill,
  observation, child, contract-error, compaction, and terminal messages.
- Added indexed complete-turn assembly, append-only thought-free session
  metadata, opaque reasoning replay state, atomic compaction, and context
  accounting under the 50,000-token ceiling with completion reserve.
- Semantic results and ordinary logs exclude thought text and provider-private
  reasoning content.

### 2026-08-23 Work Item 3: Tool And Skill Composition

- Added frozen strict `ToolRegistry`, bounded validation, permission and
  timeout boundaries, sanitized failures, reserved `skill` and
  `submit_result` tools, and root-only `run_subagent` schemas.
- Added one-level resolved-root skill discovery, safe YAML frontmatter,
  immutable JSON catalog summaries, lazy body loading, and containment checks.
- Applied the llama.cpp exact-2000 `MAX_REPETITION_THRESHOLD` finding to the
  native model-authored core schemas: every former `maxLength=2000` is the
  documented grammar-safe `1999` subset. Public/controller bounds, loop
  truncation, `maxLength=200/4000`, `maxItems`, `minLength`, and strict object
  structure remain unchanged.

### 2026-08-23 Work Item 4: Same-Runtime Subagents

- Added bounded same-runtime child construction with isolated sessions,
  self-contained tasks, inherited tools/skills/permissions/thinking stream,
  foreground serial execution, and depth-one child registries without
  `run_subagent`.
- Added the canonical parent-scoped top-level `observation_id` to the typed
  child result. Child-private evidence projects only summary,
  `provenance_refs`, and limitations; child transcript, reasoning, and private
  observation IDs stay out of the parent semantic message.
- The parent reuses that one observation handle for its observation and
  compaction state, and bounded serialization remains within the configured
  child-result cap.

### 2026-08-23 Work Item 5: Loop, Runtime, And Kazusa Adapters

- Added the serialized one-complete-native-tool-call loop, bounded structural
  replacement feedback, terminal `submit_result` enforcement, hard-cap
  dispositions, direct runtime construction, and four unchanged specialist
  adapters.
- Added observation-handle placement enforcement: handles may appear only in
  `evidence[].observation_id`; semantic text containing a current-session
  handle is rejected for bounded model regeneration, never rewritten or
  redacted. Clean replacement and parent evidence/provenance paths pass.
- Current workflow and specialist implementation sources remain unchanged;
  no workflow import, registration, route, or call edge was added.

### 2026-08-23 Work Item 6: Manifest, Documentation, And Full Deterministic Verification

- Added exact source-to-test ownership rows for every changed production path,
  updated module/LLInterface ICDs and governing architecture documentation,
  and preserved the no-workflow-wiring boundary.
- Corrected the live fixture's root/child skill contract and tightened the
  complete parent-projection child-ID privacy assertion.
- The authoritative corrected Ruff path-array invocation reported
  `All checks passed`.
- The exact 15-module deterministic gate reported `95 passed, 1 skipped`
  (the Windows symlink privilege skip). The six preservation nodes reported
  `6 passed`. The impact validator reported `70 passed, 1 skipped` across 71
  exact nodes. `git diff --check` passed with line-ending warnings only.
- Static package/import, workflow-isolation, Keep-path, XML-renderer, and
  non-streaming-path checks all passed.

### 2026-08-23 Work Item 7: Endorsed Live Gate And Final Structured Review

- The sole accepted live gate used exactly `http://127.0.0.1:8081/v1` with
  `qwen3.8-27b-dflash2-4090`; health was `ok`, `/v1/models` advertised only
  that model, and no fallback or alternate route was used.
- The exact live pytest node passed in 87.67 seconds (`1 passed`). The
  thought-free trace is
  `test_artifacts/llm_traces/test_agentic_resolver_live_llm__standalone_root_child_stream__20260823T082610372162Z.json`,
  with the accepted review at
  `test_artifacts/llm_traces/test_agentic_resolver_live_llm__standalone_root_child_stream__20260823T082610372162Z_review.md`
  and the appended review at
  `test_artifacts/llm_reviews/agentic_resolver_live_review_2026-08-23.md`.
- All eight calls streamed reasoning metadata; visible thought text was not
  persisted. The root sequence was `skill -> read_fixture_fact ->
  run_subagent -> submit_result`; the child sequence was `skill ->
  replacement-skill -> read_fixture_fact -> submit_result`. Both sessions
  loaded `resolver-verification`, and the child roster omitted
  `run_subagent`.
- Execution keys were exactly `parent_seed, child_seed`. The terminal result
  was `resolved` with `PARENT-SEED-17` and `CHILD-SEED-29`, with provenance
  `fixture:parent_seed` and `fixture:child_seed`. Parent evidence used the
  parent-scoped observation handle and the complete serialized projection
  contained no child-session observation ID.
- Root contract errors were zero. The child had one bounded
  `invalid_assistant_content` replacement; the 73-character invalid
  candidate was rejected before dispatch/history acceptance and repaired
  within the fixed budget. No thought text was persisted.
- Earlier non-endorsed endpoint/model diagnostics remain historical and are
  explicitly excluded from acceptance.

### Final Verification Evidence: 2026-08-23

Authoritative Ruff command and result:

~~~text
$ruff_paths = @(); $ruff_paths += Get-ChildItem -LiteralPath 'src/agentic_resolver' -Filter '*.py' -File | ForEach-Object { $_.FullName }; $ruff_paths += Get-ChildItem -LiteralPath 'tests' -Filter 'test_agentic_resolver_*.py' -File | ForEach-Object { $_.FullName }; $ruff_paths += @('src/kazusa_ai_chatbot/llm_interface/__init__.py', 'src/kazusa_ai_chatbot/llm_interface/contracts.py', 'src/kazusa_ai_chatbot/llm_interface/interface.py', 'src/kazusa_ai_chatbot/llm_interface/providers/openai_compatible.py', 'src/kazusa_ai_chatbot/llm_interface/reload.py', 'tests/test_llm_interface_reload.py', 'tests/test_llm_interface_tool_stream.py'); venv\Scripts\python.exe -m ruff check $ruff_paths
All checks passed!
~~~

Exact deterministic gate:

~~~text
venv\Scripts\python.exe -m pytest -q tests/test_agentic_resolver_contracts.py tests/test_agentic_resolver_json_protocol.py tests/test_agentic_resolver_streaming.py tests/test_agentic_resolver_session.py tests/test_agentic_resolver_context_budget.py tests/test_agentic_resolver_tools.py tests/test_agentic_resolver_skills.py tests/test_agentic_resolver_subagents.py tests/test_agentic_resolver_loop.py tests/test_agentic_resolver_standalone.py tests/test_agentic_resolver_kazusa_tools.py tests/test_llm_interface_tool_stream.py tests/test_llm_interface_reload.py tests/test_agentic_resolver_architecture_docs.py tests/test_test_impact_manifest.py
95 passed, 1 skipped (Windows symlink privilege)
~~~

Exact preservation gate:

~~~text
venv\Scripts\python.exe -m pytest -q tests/test_task_resolution_orchestrator.py::test_wrong_text_selection_reroutes_to_public_research tests/test_task_resolution_inline_promotion.py::test_inline_budget_default_is_thirty_seconds tests/unit/cognition_resolver/test_loop.py::test_loop_exposes_owned_contract tests/test_llm_interface_contracts.py::test_call_config_defaults_to_json_object_output tests/test_llm_interface_contracts.py::test_llm_response_strips_complete_qwen_think_tag tests/test_llm_interface_openai_provider.py::test_provider_async_path_preserves_message_objects
6 passed
~~~

~~~text
venv\Scripts\python.exe -m scripts.validate_test_impact --base-ref HEAD --run
70 passed, 1 skipped; 71 exact nodes validated
~~~

~~~text
git diff --check
Passed; line-ending normalization warnings only
~~~

Static checks passed: package discovery/import, no `agentic_resolver` import in
current workflow sources, no Kazusa import in core resolver modules, no
changed Keep-path or specialist implementation file, no XML skill renderer,
and no `ainvoke_tools`/`invoke_tools`/`ainvoke` resolver path.

## Final Structured Parent Review

The parent applied every bullet in the Independent Code Review checklist:
diff containment; downward-only integration; unchanged workflow and specialist
sources; strict public contracts; JSON-only messages; no free-text action
parsing; native reasoning/tool normalization and passback; complete-turn
assembly; mandatory thinking; stream-safe unload retry; reasoning-aware budget
and atomic compaction; sanitization; same-runtime isolated children; child
non-recursion; permission non-expansion; bounded return; exact mappings;
deterministic evidence; live stream quality; and thought-text exclusion. No
unresolved material finding remains.

The distinct independent-review role was waived by the user. This record does
not claim an independent agent review; it records the waiver and the complete
parent-applied checklist instead.

Residual non-blocking risks are recorded for future maintenance: the provider
converter's private-hook dependency needs regression coverage on SDK upgrades;
the llama.cpp repetition threshold requires retaining the `1999` subset until
the server changes and is reverified; the one live bounded replacement shows
the repair path remains useful; and future workflow/big-bang integration is a
separate plan and remains absent.

## Acceptance Criteria Evidence: 2026-08-23

1. Package configuration/distribution discovery passed.
2. Direct runtime construction and invocation passed without brain/cognition
   startup.
3. Workflow import/call scans passed with no resolver edge.
4. Keep-path checks found no existing tool implementation changes.
5. Task-resolution and cognition preservation nodes passed.
6. Ordinary LLInterface preservation nodes passed.
7. `astream_tools` native-tool stream tests passed.
8. Root and child non-streaming-path scans passed.
9. Public `resolve` returned one terminal typed result; no public token stream.
10. Closed normalized reasoning/text/tool/usage/finish chunk tests passed.
11. Tool-schema digest cache partition tests passed.
12. Every resolver semantic message remained one JSON object.
13. Enabled supported-thinking capability admission tests passed.
14. Opaque reasoning stayed out of semantic decisions and public fields.
15. Provider-native reasoning replay and empty-field tests passed.
16. Thought text was absent from results, logs, tools, and artifacts.
17. Partial/interrupted tool streams were rejected before dispatch.
18. Stream-safe unload retry boundaries passed.
19. Native tool calls were the only model control path.
20. Exactly one complete tool call per step was enforced.
21. `submit_result` remained mandatory for normal completion.
22. Structural replacement budget behavior passed.
23. Four specialist adapters retained their validated handler contracts.
24. Existing public-research WebAgent3 ownership test passed.
25. Existing coding approval/lifecycle ownership test passed.
26. One-level skill discovery and invalid-catalog fail-fast tests passed.
27. Root containment and safe YAML frontmatter tests passed.
28. JSON skill catalog summary/digest tests passed.
29. Full skill bodies loaded only through `skill`.
30. Skills did not expand tools, permissions, limits, or effects.
31. Context accounting covered policy, schemas, replay, skills, observations,
    children, and reserve without double-counting.
32. Root and child requests stayed within the 50,000-token ceiling.
33. Atomic reasoning/tool/result compaction tests passed.
34. Overflow returned `budget_exhausted` before model invocation.
35. Root `run_subagent` was available as a core tool.
36. Children used the same runtime, loop, and assembler.
37. Children had isolated sessions and self-contained tasks.
38. Child inherited tools, skills, thinking stream, replay, permissions, JSON,
    and context policy.
39. Child registries excluded `run_subagent`.
40. Root child cap stopped after three children.
41. Parent received only bounded typed child results without transcript/thought.
42. Multiple-child deterministic convergence passed.
43. Every changed production path had an exact collected manifest mapping.
44. Endorsed live gate passed with explicit debug-LLM quality PASS.
45. Final parent-applied review checklist has no unresolved material finding;
    the independent reviewer role was user-waived.
46. This plan records implementation, verification, live, privacy, and review
    evidence before completion.
47. Future workflow/big-bang integration is absent from the implementation.

## Closure Decision

Quality decision: **PASS**. The standalone agentic resolver first pass is
complete and archived as a historical execution record. Future workflow
integration requires a separate approved plan.
