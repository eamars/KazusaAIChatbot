# Agentic Resolver Interface Control Document

## Document Control

- Owning package: `agentic_resolver`
- Public entry point: `AgenticResolverRuntime.resolve(...)`
- First-pass boundary: standalone direct Python construction and invocation
- Workflow state: no import, registration, selection, or call edge from Kazusa
  cognition, brain service, task resolution, accepted tasks, or background work
- Renewed target architecture:
  `docs/architecture/agentic_resolver_architecture.md`

This package is a bounded native-tool resolver. It owns one serialized root
session, optional foreground depth-one child sessions, typed model transport,
JSON semantic messages, deterministic limits, and one terminal public result.
It is installable independently of Kazusa workflow startup. Optional adapters
under `agentic_resolver.integrations` depend downward on existing public
Kazusa capabilities; the core package has no upward workflow dependency.

## Public Runtime

Callers construct every dependency explicitly:

```python
runtime = AgenticResolverRuntime(
    model=model_client,
    tools=tool_registry,
    skills=skill_catalog,
    limits=limits,
    permission_scope=permission_scope,
)

result = await runtime.resolve(request)
```

`AgenticResolverRequestV1` contains a bounded objective and prompt-safe
context. `AgenticResolverResultV1` contains one validated terminal status,
summary, evidence projections, completed work, remaining needs, and code-owned
usage. The public call returns one terminal result. It does not expose a token
or thought stream.

The first pass has no route, adapter command, database persistence, resume
protocol, workflow registration, or compatibility bridge. A later approved
big-bang plan owns any cognition or task-resolution cutover.

## JSON Semantic Protocol

Every non-empty resolver-authored semantic textual payload parses as exactly
one object-rooted JSON value. This includes policy, catalog, task, skill
content, tool observations, subagent tasks and results, contract feedback,
compacted observations, and terminal arguments. Native tool definitions use
JSON Schema and native tool arguments are JSON objects.

The controller accepts one complete native tool call per model step. Assistant
text is empty or one JSON object; it is never parsed as an action or terminal
answer. Normal completion requires the controller-owned `submit_result` tool.
XML, pseudo-XML, free-form action parsing, and stage-local JSON repair are
outside this package contract.

## Thinking-Enabled Stream

Every root and child model step uses `AgenticModelClient.astream(...)`.
Construction requires an immutable `AgenticModelCapabilitiesV1` declaration
with streaming active, thinking enabled, a supported thinking strategy, and a
reasoning replay policy. Unsupported or disabled thinking fails admission.

The closed normalized chunk family carries indexed block start/end events,
opaque reasoning deltas, text deltas, native tool-call argument deltas, usage,
and one finish state. `ModelStreamAssembler` consumes the complete stream
before any dispatch. Interrupted, malformed, aborted, error, or max-token
partial tool calls never reach a tool implementation.

Reasoning is opaque assistant transport state. The session retains it only for
provider-required replay and context accounting. It is kept separate from
semantic JSON, tool observations, evidence, child projections, terminal
results, permissions, and ordinary event metadata. Provider-specific replay,
including a required empty native reasoning field for qualifying tool-call
turns, is owned by the LLInterface provider adapter.

## Tools And Permissions

`ToolRegistry` freezes a sorted unique ordinary-tool roster at runtime
construction. Each `ToolDefinition` declares an object-rooted JSON schema,
trusted executor, optional validator, permission check, bounded projector, and
side-effect class. Deterministic code validates arguments, permission scope,
timeouts, output size, and exception sanitization.

The reserved controller-owned tools are:

- `skill`, which lazily loads one discovered instruction body;
- `run_subagent`, which is present only in the root registry; and
- `submit_result`, which validates normal terminalization.

Tool implementations retain their own domain semantics. The resolver never
derives permissions from model-authored content.

## Skills

`discover_skills(...)` scans only direct child bundles under caller-supplied
roots. Each bundle has one `SKILL.md` with safe-YAML `name` and `description`
frontmatter followed by a bounded Markdown body. Startup resolves root and
file containment, rejects malformed, duplicate, escaped, or oversized
entries, freezes a canonical digest, and injects only JSON name/description
summaries. The full body enters one session only after an explicit `skill`
tool call. Skills grant instructions, not capabilities or permissions.

## Same-Runtime Children

`run_subagent` constructs an isolated depth-one session through the same
`AgenticResolverRuntime`, `AgentLoop`, model adapter, thinking stream, JSON
protocol, ordinary tools, skill catalog, permission scope, and limits. The
child receives one self-contained typed task rather than parent history. Its
registry omits `run_subagent`, so delegation cannot recurse.

The root runs at most three children serially in the foreground. It receives
only a bounded `AgenticResolverSubagentResultV1` tool observation and owns all
convergence. Child stream events, reasoning, history, and runtime objects do
not enter the parent semantic message. Each successful child projection has one
code-owned top-level `observation_id` allocated from the parent session. A
parent `submit_result` evidence row cites that top-level ID. Nested child
evidence contains only `summary`, `provenance_refs`, and `limitations` as
provenance context; child-session observation IDs are private and omitted.
Observation handles are valid only in `submit_result.evidence[].observation_id`.
Model-authored summary, evidence summary or limitations, completed-task, and
remaining-need text must not repeat a current-session observation ID. The loop
rejects such a terminal candidate with bounded contract feedback so the model
can regenerate it; deterministic code never rewrites or redacts semantic text.

## Budgets And Session State

The project hard context window is 50,000 estimated tokens with an 8,000-token
completion reserve. Accounting includes system policy, catalog, tool schemas,
task, history, retained reasoning, loaded skills, observations, and reserve.
Old reasoning/tool-call/tool-result exchanges compact atomically; no orphaned
assistant call or tool result can enter provider history. An over-cap request
stops before model invocation.

Model steps, non-terminal tools, structural replacements, children,
wall-clock time, individual tool time, skill size, and model-visible result
size all have caller-lowerable hard caps. Session event metadata records
ordering and counts while excluding thought text.

## Optional Integrations

`agentic_resolver.integrations.llm_interface.LLInterfaceToolModel` maps the
additive LLInterface native-tool stream into the core model protocol and
requires a supported enabled thinking route.

`agentic_resolver.integrations.kazusa_tools.build_kazusa_tool_registry(...)`
exposes exactly the existing `local_context`, `public_research`, `coding`, and
`text_computation` specialist handlers. Those handlers retain their current
contracts, internal orchestration, source ownership, approval lifecycle,
timeouts, and failure behavior.

## Target Transition

This ICD describes the currently implemented standalone prototype. The
renewed target architecture supersedes its standalone-first, four-facade, and
DAG-backed direction for future integration.

At the approved cutover, the Kazusa brain action selector remains the caller
through `task_resolution_request`, and the brain retains foreground/background
selection, promotion, resume, cognition, and delivery ownership. The
resolution layer moves in one contract update to the native agent loop and
eligible base-level semantic tool catalog defined by
`docs/architecture/agentic_resolver_architecture.md`.

Until that cutover, the four integration adapters above remain a description
of current code, not the target tool boundary. This README remains the
implemented-state source of truth and must be updated atomically with the
production cutover.

## Verification Contract

Deterministic tests under `tests/test_agentic_resolver_*.py` own contracts,
JSON messages, stream assembly, sessions, context budgets, tools, skills,
children, loop behavior, standalone isolation, adapters, packaging, and this
ICD. `tests/test_llm_interface_tool_stream.py` owns the shared transport
bridge. The live-LLM case runs alone and produces a protected structured trace
plus a human-readable review that records reasoning presence and ordering
without thought text.
