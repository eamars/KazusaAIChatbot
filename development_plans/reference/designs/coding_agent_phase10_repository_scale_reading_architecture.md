# coding agent phase10 repository-scale reading architecture

## Summary

- Goal: define the Phase 10 directional architecture for repository-scale
  code reading through a master PM, subsystem PMs, bounded programmer workers,
  and evidence synthesis.
- Plan class: directional_architecture.
- Status: reference direction.
- Execution rule: reference only. Promote this document into
  `development_plans/active/short_term/` before implementation.
- Mandatory skills for future execution: `development-plan`,
  `local-llm-architecture`, `py-style`, `test-style-and-execution`, and
  `debug-llm`.
- Direction: expand code reading from a single bounded PM/programmer workflow
  into a multi-level, local-LLM-friendly evidence graph that can answer
  repository-scale questions without unbounded prompts.

## Context

The current `code_reading` implementation already has a PM/programmer
workflow:

```text
CodeReadingRequest
-> repository map summary
-> reading PM
-> bounded programmer assignments
-> programmer reports
-> PM sufficiency
-> final synthesis
```

The current README explicitly says full distributed master/subsystem PM fan-out
is not implemented. The `code_reading.master_pm` module is only a placeholder.
That is acceptable for narrow code questions, but it limits Codex-like
capability for broader tasks such as:

- architectural walkthroughs;
- migration impact analysis;
- cross-module behavior tracing;
- ownership boundary audits;
- large refactor planning;
- test strategy discovery;
- dependency and call-flow mapping.

Phase 10 gives those broad reading tasks a real distributed reading
architecture while keeping writes, patching, apply, execution, and repair
outside the reading boundary.

## Architecture Direction

Phase 10 introduces repository-scale reading as a mode inside `code_reading`:

```text
CodeReadingRequest
-> deterministic repository inventory
-> master reading PM
-> subsystem reading PM tasks
-> subsystem PMs
-> programmer inspection tasks
-> programmer reports
-> subsystem summaries
-> master synthesis PM
-> CodeReadingResult
```

The repository-scale reading flow remains read-only. It never writes files,
applies patches, executes project commands, installs packages, or fetches
external facts.

## Target Components

Future implementation should add or complete:

```text
src/kazusa_ai_chatbot/coding_agent/code_reading/
  master_pm.py
  subsystem_pm.py
  repository_inventory.py
  evidence_graph.py
  synthesis.py
```

Existing files such as `supervisor.py`, `models.py`, `planner.py`, and
`programmer.py` stay in place and become the narrow-reading path or the worker
building blocks for broad reading.

## Reading Modes

Phase 10 should support two reading modes:

| Mode | Use |
|---|---|
| `focused` | Current bounded question flow for narrow file/symbol/directory reads. |
| `repository_scale` | Multi-subsystem flow for broad architectural or impact questions. |

Mode selection can be deterministic from request metadata in the future
executable plan. It can also be explicit through a trusted request field. The
LLM should not receive control over global loop limits or filesystem
permissions.

## Deterministic Repository Inventory

Repository-scale reading starts with a deterministic inventory:

```python
{
    "safe_text_file_count": int,
    "candidate_subsystems": list[dict[str, object]],
    "top_level_readmes": list[str],
    "python_packages": list[str],
    "test_roots": list[str],
    "docs_roots": list[str],
    "entrypoints": list[str],
    "excluded_paths": list[dict[str, str]],
}
```

The inventory owns:

- safe text path discovery;
- binary and secret-like exclusion;
- `.env` and `.git` exclusion;
- top-level directory grouping;
- Python package and import root hints;
- README and docs discovery;
- test-root discovery;
- file-count and path-count caps.

The inventory is not semantic evidence by itself. It is a routing substrate for
PMs.

## Master PM Direction

The master reading PM owns:

- deciding whether a question is broad enough for repository-scale reading;
- decomposing the question into subsystem objectives;
- assigning subsystem scopes;
- defining required evidence slots;
- enforcing the repository-scale reading budget;
- deciding whether subsystem evidence is sufficient;
- producing final synthesis only from subsystem reports and evidence graph
  rows.

Directional master statuses:

```text
need_subsystem_pms
sufficient
needs_user_input
overloaded
blocked
```

The master PM must work from repository inventory and compact prior reports.
It should never receive full repository source.

## Subsystem PM Direction

Each subsystem PM owns one bounded subsystem:

```python
{
    "subsystem_id": str,
    "objective": str,
    "scope": {
        "kind": "directory | package | tests | docs | entrypoint",
        "values": list[str],
    },
    "required_slots": list[str],
    "budget": dict[str, int],
}
```

Subsystem PM statuses:

```text
need_programmers
sufficient
needs_master_input
overloaded
blocked
```

Subsystem PMs create programmer assignments within their scope. They return
compact subsystem reports and evidence rows to the master PM.

## Programmer Direction

Programmer workers remain the lowest-level read-only inspectors. They own one
bounded local task:

```python
{
    "assignment_id": str,
    "subsystem_id": str,
    "role": str,
    "scope": {
        "kind": "file | directory | symbol | search",
        "values": list[str],
    },
    "questions": list[str],
    "required_slots": list[str],
}
```

Programmers return:

- files read;
- facts with evidence ids;
- compact excerpts;
- open questions;
- local limitations.

They never see a global repository task beyond their bounded assignment.

## Evidence Graph Direction

Phase 10 should promote evidence from a flat list to a small graph:

```python
{
    "evidence_id": str,
    "path": str,
    "line_start": int,
    "line_end": int,
    "symbol_or_topic": str,
    "excerpt": str,
    "reason": str,
    "subsystem_id": str,
    "supports_slots": list[str],
    "related_evidence_ids": list[str],
}
```

The public `CodeReadingResult.evidence` can remain a list projection for
compatibility. Internal synthesis can use the graph to connect facts across
subsystems.

## Synthesis Direction

Final repository-scale synthesis is a PM-owned stage:

```text
master PM reports
-> subsystem summaries
-> selected evidence graph rows
-> bounded final answer
```

Synthesis rules:

- cite evidence rows for claims about source behavior;
- distinguish confirmed facts from limitations;
- report overloaded scope honestly;
- ask for narrower user scope when broad reading caps are exceeded;
- preserve the public `CodeReadingResult` shape.

## Local LLM Design Rules

- Keep the master PM prompt based on inventory, task, and compact reports.
- Keep subsystem PM prompts based on one subsystem scope.
- Keep programmer prompts based on one local assignment.
- Keep final synthesis evidence-backed and bounded.
- Cap the number of subsystem PMs.
- Cap programmer waves per subsystem.
- Cap total evidence rows and excerpt length.
- Prefer explicit `overloaded` or `needs_user_input` over weak repository-wide
  guesses.
- Record every cap in `trace_summary` and `limitations`.

## Directional Limits

Initial Phase 10 executable limits should start conservatively:

| Limit | Initial direction |
|---|---|
| Subsystem PMs per run | 4 |
| Programmer tasks per subsystem | 3 |
| Programmer waves per subsystem | 2 |
| Total programmer reports | 16 |
| Evidence rows returned | 40 |
| Excerpt chars per evidence row | 1200 |
| Final answer chars | caller cap or existing default |

These are starting limits for a local or weak LLM. Future plans can adjust them
after live trace evidence.

## Codex Capability Mapping

| Codex behavior | Kazusa Phase 10 direction |
|---|---|
| Scans a repository broadly | Deterministic inventory plus subsystem PMs |
| Builds mental model across modules | Evidence graph and subsystem summaries |
| Finds relevant files before editing | Repository-scale reading feeds Phase 7 and Phase 9 plans |
| Explains architecture | Master PM synthesis from evidence rows |
| Plans large refactors | Reading-only impact reports for later modification phases |
| Manages huge context | Hierarchical summaries and strict caps |

## Future Executable Phase 10 Scope

The eventual short-term Phase 10 plan should include:

- repository inventory implementation;
- active `master_pm.py` flow;
- new `subsystem_pm.py` contracts and prompt/runtime;
- evidence graph models and projection to current public evidence rows;
- supervisor mode selection for focused versus repository-scale reading;
- deterministic tests for inventory, scope caps, prompt rendering, path
  safety, graph projection, overload handling, and public sanitization;
- live LLM role tests for master PM, subsystem PM, programmer, and synthesis;
- repository-scale acceptance gates over fixture repositories;
- documentation updates across coding-agent README, code-reading README, HOWTO,
  and architecture reference.

## Future Phase 10 Exclusions

Keep these outside the first executable Phase 10 plan:

- source modification;
- patch proposal generation;
- patch application;
- command execution;
- execution-driven repair;
- dependency installation;
- external web research;
- adapter delivery;
- background autonomous continuation.

## Acceptance Direction

A future executable Phase 10 plan is ready for sign-off when:

- focused reading remains backward compatible;
- repository-scale reading can answer broad architecture and impact questions
  from bounded evidence;
- overloaded questions return honest scope limitations or narrowing requests;
- no prompt contains unbounded repository source;
- public results remain sanitized;
- live LLM gates show the master/subsystem/programmer split improves broad
  source understanding without breaking local latency and context budgets;
- independent code review accepts the reading boundary.

