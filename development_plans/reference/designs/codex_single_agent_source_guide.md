# Codex Single-Agent Source Guide

## Status

- Type: reference guide
- Status: reference
- Source checkout assumed: `C:\workspace\codex`
- Kazusa scope: coding-agent architecture reference for local-LLM design
- Execution rule: use this document as source-navigation context only. This
  guide is not an executable implementation plan and does not authorize
  production-code changes.

## Purpose

This guide points implementation agents to the Codex source sections that
implement the normal single-agent coding workflow. The target question is not
how Codex coordinates subagents. The target question is how one Codex session
reads code, writes code, validates work, preserves context, and repeats tool
calls inside one agent loop.

Use this guide when comparing Kazusa's local-LLM-first coding-agent design
against Codex. Codex is a useful reference for deterministic tooling,
transcript management, patch application, and sandbox boundaries. Do not copy
Codex's single-agent semantic ownership assumption into Kazusa without
preserving Kazusa's bounded PM/programmer role contracts.

Line numbers below are orientation anchors from the inspected checkout. If
they drift, search for the named symbol first.

## High-Level Design

Codex single-agent flow:

```text
user input
  -> Session submission handler
  -> RegularTask
  -> run_turn
  -> build model prompt from history, context, and tool specs
  -> model emits assistant message or tool call
  -> tool call is routed to deterministic runtime
  -> tool output is recorded into transcript
  -> run_turn samples the model again if follow-up is needed
  -> final assistant message completes the turn
```

The backend model does not run the loop itself. Codex owns the loop. The model
must be capable of using the exposed tool-call protocol, reading tool outputs,
and deciding the next action.

## Fast Source Walk

Read these files in order when re-inspecting Codex:

1. [tasks/regular.rs](C:/workspace/codex/codex-rs/core/src/tasks/regular.rs:28)
   starts a normal agent turn and calls `run_turn`.
2. [session/turn.rs](C:/workspace/codex/codex-rs/core/src/session/turn.rs:141)
   owns the sampling and follow-up loop.
3. [stream_events_utils.rs](C:/workspace/codex/codex-rs/core/src/stream_events_utils.rs:405)
   turns model output items into assistant messages or queued tool executions.
4. [tools/spec_plan.rs](C:/workspace/codex/codex-rs/core/src/tools/spec_plan.rs:158)
   builds the model-visible tool list and runtime registry.
5. [tools/router.rs](C:/workspace/codex/codex-rs/core/src/tools/router.rs:113)
   converts response items into tool calls and dispatches them.
6. [tools/parallel.rs](C:/workspace/codex/codex-rs/core/src/tools/parallel.rs:63)
   executes tool calls and turns handler results back into model-visible
   tool outputs.
7. [context_manager/history.rs](C:/workspace/codex/codex-rs/core/src/context_manager/history.rs:34)
   stores the transcript and prepares prompt history.
8. [tools/handlers/apply_patch.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/apply_patch.rs:344)
   owns direct patch-tool execution.
9. [tools/handlers/unified_exec/exec_command.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/unified_exec/exec_command.rs:106)
   owns the unified shell execution path.

## Session And Turn Entry

| Design point | Codex source | What to inspect |
| --- | --- | --- |
| User input starts a regular task | [session/handlers.rs](C:/workspace/codex/codex-rs/core/src/session/handlers.rs:266) | `user_input_or_turn_inner` builds `TurnInput` and calls `spawn_task(..., RegularTask::new())`. |
| Session task lifecycle | [tasks/mod.rs](C:/workspace/codex/codex-rs/core/src/tasks/mod.rs:314) | `spawn_task` aborts/replaces active work and delegates to `start_task`. |
| Task runtime wrapper | [tasks/mod.rs](C:/workspace/codex/codex-rs/core/src/tasks/mod.rs:325) | `start_task` records turn lifecycle state, creates cancellation tokens, spawns the task, flushes rollout, and completes the turn. |
| Normal agent task | [tasks/regular.rs](C:/workspace/codex/codex-rs/core/src/tasks/regular.rs:28) | `RegularTask` implements `SessionTask`. |
| Repeated turn execution | [tasks/regular.rs](C:/workspace/codex/codex-rs/core/src/tasks/regular.rs:74) | `RegularTask::run` calls `run_turn` and continues if pending input arrived while the model was running. |

Kazusa lesson:

- A background coding job should have a deterministic owner for lifecycle,
  cancellation, pending input, and finalization.
- The local LLM should not own job lifecycle. It should receive bounded
  semantic tasks behind the supervisor.

## Model Sampling Loop

| Design point | Codex source | What to inspect |
| --- | --- | --- |
| Main turn loop | [session/turn.rs](C:/workspace/codex/codex-rs/core/src/session/turn.rs:141) | `run_turn` builds context, runs hooks, samples the model, checks follow-up, compacts when needed, and exits on final answer. |
| Prompt object | [session/turn.rs](C:/workspace/codex/codex-rs/core/src/session/turn.rs:1079) | `build_prompt` combines prompt input, model-visible tool specs, parallel-tool flag, base instructions, and optional output schema. |
| Tool registry per request | [session/turn.rs](C:/workspace/codex/codex-rs/core/src/session/turn.rs:1107) | `run_sampling_request` builds tools, gets base instructions, creates `ToolCallRuntime`, then calls the model. |
| Streaming response handler | [session/turn.rs](C:/workspace/codex/codex-rs/core/src/session/turn.rs:1919) | `try_run_sampling_request` streams response events and tracks in-flight tool futures. |
| Completed model item | [session/turn.rs](C:/workspace/codex/codex-rs/core/src/session/turn.rs:2020) | `ResponseEvent::OutputItemDone` sends completed items to `handle_output_item_done`. |
| Response completion | [session/turn.rs](C:/workspace/codex/codex-rs/core/src/session/turn.rs:2235) | `ResponseEvent::Completed` decides whether another model request is needed. |

Key behavior:

- If the model emits a tool call, Codex records the call, executes the tool,
  records the tool output, and samples again.
- If the model emits only a final assistant message and no pending input or
  stop-hook continuation exists, the turn completes.
- The single-agent loop is iterative but externally orchestrated by Codex.

Kazusa lesson:

- The loop mechanics are worth copying: deterministic loop owner, explicit
  follow-up flag, transcript persistence, and cancellation.
- The semantic load should still be split for Kazusa local LLMs. Codex lets one
  model decide what to inspect, edit, and validate; Kazusa should keep PM,
  sub-PM, programmer, patcher, and validator boundaries.

## Tool Registration And Routing

| Design point | Codex source | What to inspect |
| --- | --- | --- |
| Build tool router | [tools/spec_plan.rs](C:/workspace/codex/codex-rs/core/src/tools/spec_plan.rs:158) | `build_tool_router` creates model-visible specs and the runtime registry. |
| Runtime and spec planning | [tools/spec_plan.rs](C:/workspace/codex/codex-rs/core/src/tools/spec_plan.rs:232) | `build_model_visible_specs_and_registry` separates exposed specs from hidden dispatch-only runtimes. |
| Tool sources | [tools/spec_plan.rs](C:/workspace/codex/codex-rs/core/src/tools/spec_plan.rs:605) | `add_tool_sources` adds shell, MCP/resource, utility, collaboration, extension, dynamic, and hosted tools. |
| Shell tools | [tools/spec_plan.rs](C:/workspace/codex/codex-rs/core/src/tools/spec_plan.rs:629) | `add_shell_tools` registers `exec_command` or shell command depending on model/features. |
| Core utilities | [tools/spec_plan.rs](C:/workspace/codex/codex-rs/core/src/tools/spec_plan.rs:690) | Adds plan, permissions, context remaining, current time, sleep, plugin install, `apply_patch`, test sync, and image tools. |
| Runtime trait | [tools/registry.rs](C:/workspace/codex/codex-rs/core/src/tools/registry.rs:48) | `CoreToolRuntime` defines the local deterministic handler contract. |
| Tool call parsing | [tools/router.rs](C:/workspace/codex/codex-rs/core/src/tools/router.rs:113) | `build_tool_call` converts function/custom/tool-search response items into `ToolCall`. |
| Tool dispatch | [tools/router.rs](C:/workspace/codex/codex-rs/core/src/tools/router.rs:187) | `dispatch_tool_call_with_terminal_outcome` invokes the registered runtime. |
| Tool execution wrapper | [tools/parallel.rs](C:/workspace/codex/codex-rs/core/src/tools/parallel.rs:63) | `handle_tool_call` converts runtime success/failure into model-visible output. |

Kazusa lesson:

- Keep a deterministic tool registry/facade. Tool availability, permissions,
  and execution should be deterministic code.
- Local LLM prompts should see small, semantic capability descriptions rather
  than broad operational knobs.

## Code Reading Path

Codex code reading is not a separate repository-understanding engine in the
single-agent path. The agent reads by calling shell/exec tools, commonly using
`rg`, file reads, `git`, and tests.

| Design point | Codex source | What to inspect |
| --- | --- | --- |
| Unified exec handler | [tools/handlers/unified_exec/exec_command.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/unified_exec/exec_command.rs:106) | Parses args, resolves environment/cwd, applies permissions, optionally intercepts `apply_patch`, and executes command. |
| Shell-like handler | [tools/handlers/shell.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/shell.rs:61) | `run_exec_like` is the legacy shell-command execution path. |
| Direct command execution | [tools/handlers/unified_exec/exec_command.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/unified_exec/exec_command.rs:341) | Calls the unified exec process manager. |
| Output truncation in history | [context_manager/history.rs](C:/workspace/codex/codex-rs/core/src/context_manager/history.rs:91) | `record_items` stores items with truncation policy. |

Kazusa lesson:

- Deterministic tools should own repository discovery and file reads.
- Do not send unbounded raw command output or whole repositories into local
  LLM prompts.
- Kazusa's code-reading PM/programmer split is still necessary: programmers
  inspect bounded source slices and return structured evidence; PMs synthesize
  from reports.

## Code Writing Path

Codex writes through a structured patch path, not arbitrary in-process file
mutation by the model.

| Design point | Codex source | What to inspect |
| --- | --- | --- |
| Freeform patch tool spec | [tools/handlers/apply_patch_spec.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/apply_patch_spec.rs:9) | `create_apply_patch_freeform_tool` exposes a grammar-backed freeform tool. |
| Patch handler entry | [tools/handlers/apply_patch.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/apply_patch.rs:61) | `ApplyPatchHandler` is the runtime handler. |
| Patch call execution | [tools/handlers/apply_patch.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/apply_patch.rs:344) | `handle_call` parses, verifies, checks environment, and delegates patch application. |
| Parse patch text | [tools/handlers/apply_patch.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/apply_patch.rs:363) | `codex_apply_patch::parse_patch`. |
| Verify against filesystem | [tools/handlers/apply_patch.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/apply_patch.rs:387) | `verify_apply_patch_args` checks patch correctness against the selected environment filesystem. |
| Safety decision | [apply_patch.rs](C:/workspace/codex/codex-rs/core/src/apply_patch.rs:39) | `assess_patch_safety` decides auto-approve, ask user, or reject. |
| Apply via runtime | [tools/handlers/apply_patch.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/apply_patch.rs:406) | Delegates to the patch runtime after safety and permission handling. |
| Shell patch interception | [tools/handlers/apply_patch.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/apply_patch.rs:543) | Shell commands that are really `apply_patch` can be intercepted and routed through the same verified path. |

Kazusa lesson:

- Codex's patch path is the strongest implementation reference for Kazusa's
  Patcher boundary.
- Keep semantic code generation separate from edit mechanics.
- Deterministic code should own path targeting, patch grammar, filesystem
  verification, sandbox policy, and final patch artifact validation.
- For local LLMs, programmers should not receive patch format, insertion
  mechanics, file mutexes, or validation traces. Those belong to the File Edit
  Manager or patcher.

## Context, History, And Instructions

| Design point | Codex source | What to inspect |
| --- | --- | --- |
| Load project instructions | [agents_md.rs](C:/workspace/codex/codex-rs/core/src/agents_md.rs:46) | `load_project_instructions` combines host user instructions with project `AGENTS.md`. |
| Read `AGENTS.md` files | [agents_md.rs](C:/workspace/codex/codex-rs/core/src/agents_md.rs:82) | `read_agents_md` reads bounded project docs. |
| Discover hierarchy | [agents_md.rs](C:/workspace/codex/codex-rs/core/src/agents_md.rs:155) | `agents_md_paths` walks from project root to cwd. |
| Instruction rendering | [agents_md.rs](C:/workspace/codex/codex-rs/core/src/agents_md.rs:263) | `LoadedAgentsMd` renders model-visible instruction text. |
| Initial context build | [session/mod.rs](C:/workspace/codex/codex-rs/core/src/session/mod.rs:3012) | `build_initial_context` injects permissions, collaboration mode, apps/plugins/skills, extensions, user instructions, token budget, and environment context. |
| Context diffs | [session/mod.rs](C:/workspace/codex/codex-rs/core/src/session/mod.rs:3414) | `record_context_updates_and_set_reference_context_item` emits full context first, then diffs when possible. |
| Transcript write | [session/mod.rs](C:/workspace/codex/codex-rs/core/src/session/mod.rs:2757) | `record_conversation_items` appends history, persists rollout, and sends raw response events. |
| Transcript model view | [context_manager/history.rs](C:/workspace/codex/codex-rs/core/src/context_manager/history.rs:111) | `for_prompt` normalizes history before sending it to the model. |
| History invariants | [context_manager/history.rs](C:/workspace/codex/codex-rs/core/src/context_manager/history.rs:327) | `normalize_history` ensures call/output pairing and strips unsupported images. |

Kazusa lesson:

- Preserve a durable run ledger and prompt-safe history.
- Keep raw source and command outputs in private evidence storage.
- Feed local LLM stages compact evidence rows, structured role reports, and
  semantic summaries.
- Use stable project guidance, but avoid bloating every role prompt with
  operational details irrelevant to that role.

## Compaction And Context Windows

| Design point | Codex source | What to inspect |
| --- | --- | --- |
| Pre-sampling compaction | [session/turn.rs](C:/workspace/codex/codex-rs/core/src/session/turn.rs:849) | `run_pre_sampling_compact` can compact before sampling. |
| Mid-turn auto compact | [session/turn.rs](C:/workspace/codex/codex-rs/core/src/session/turn.rs:965) | `run_auto_compact` handles context pressure during a turn. |
| New context window | [session/mod.rs](C:/workspace/codex/codex-rs/core/src/session/mod.rs:3359) | `maybe_start_new_context_window` replaces history and reinjects initial context. |
| Token estimation | [context_manager/history.rs](C:/workspace/codex/codex-rs/core/src/context_manager/history.rs:132) | History estimates token count from base instructions plus items. |

Kazusa lesson:

- Context compaction is infrastructure, not semantic understanding.
- For local LLMs, compaction should preserve contracts and evidence summaries,
  not just conversational continuity.
- Kazusa should keep role outputs as compact, typed memory boundaries so a
  resumed or repaired stage does not need to reread a whole repository.

## Permission And Sandbox Boundaries

| Design point | Codex source | What to inspect |
| --- | --- | --- |
| Exec permissions | [tools/handlers/unified_exec/exec_command.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/unified_exec/exec_command.rs:229) | `get_command` and subsequent permission handling resolve shell behavior and escalation requests. |
| Apply patch safety | [apply_patch.rs](C:/workspace/codex/codex-rs/core/src/apply_patch.rs:39) | `assess_patch_safety` checks approval policy, permission profile, sandbox policy, cwd, and Windows sandbox level. |
| Shell approval path | [tools/handlers/shell.rs](C:/workspace/codex/codex-rs/core/src/tools/handlers/shell.rs:173) | Shell execution creates an approval requirement before runtime execution. |

Kazusa lesson:

- Deterministic validation must own mutation safety.
- Phase 2 patch proposal should not mutate the real workspace.
- Future patch-apply phases need explicit approval and workspace safety separate
  from patch generation.

## What To Avoid For This Reference

The following Codex areas are not the focus for Kazusa's current question:

- `codex-rs/core/src/session/multi_agents.rs`
- `codex-rs/core/src/tools/handlers/multi_agents*`
- `codex-rs/core/src/tools/handlers/agent_jobs*`
- Cloud/offloaded workflows

Those files are useful only when studying Codex subagent coordination. This
guide is about the normal one-session agent loop.

## Kazusa Design Mapping

| Codex concept | Kazusa coding-agent equivalent | Keep or adapt |
| --- | --- | --- |
| `Session` and `RegularTask` | Coding supervisor job loop | Keep the deterministic lifecycle pattern. |
| `run_turn` loop | Supervisor-controlled bounded workflow loop | Adapt. Kazusa should not let one weak model own all semantic steps. |
| `ToolRouter` and `CoreToolRuntime` | Deterministic tool facade | Keep. Tools should be deterministic, typed, capped, and auditable. |
| Shell/exec tools | Repository discovery, bounded `rg`, file reads, validation commands in future `code_executing` | Adapt with stricter allowlists and output caps. |
| `ContextManager` | Run ledger, evidence store, compact role reports | Adapt. Store private raw evidence outside model prompts. |
| `AGENTS.md` loading | Source/project guidance and runtime role instructions | Adapt. Do not let broad project docs overload local role prompts. |
| `apply_patch` runtime | Patcher validation | Keep the separation between semantic content and edit mechanics. |
| Auto-compaction | Context budget and role-report summarization | Adapt around local-LLM contracts, not free-form transcript summaries. |

## Implementation-Agent Checklist

When using Codex as a reference for Kazusa:

1. Start with the Codex single-agent control loop, not multi-agent files.
2. Copy deterministic boundaries: tool registry, transcript, sandbox,
   permission checks, patch grammar, patch validation, output truncation.
3. Reject the Codex assumption that one model can reliably inspect, plan,
   edit, and validate a large repository for Kazusa's local LLM route.
4. Preserve Kazusa's PM/programmer hierarchy:
   - supervisor owns cross-domain loop and evidence budget;
   - PMs own semantic decomposition and sufficiency;
   - file agent owns path resolution and owned/read-only maps;
   - sub-PMs own one-file or one-module unit contracts;
   - programmers produce one bounded function/test/doc body;
   - Patcher owns edit mechanics;
   - validators own structural and behavioral acceptance.
5. Keep raw repository files, command output, local paths, workspace roots,
   cache keys, and traces out of public responses.
6. Use focused role gates before full E2E gates when local-LLM behavior is the
   unresolved risk.

## Quick Search Commands

Use these from `C:\workspace\codex` when line anchors drift:

```powershell
rg -n "pub\\(crate\\) async fn run_turn|async fn run_sampling_request|async fn try_run_sampling_request" codex-rs\core\src\session\turn.rs
rg -n "build_tool_router|add_shell_tools|add_core_utility_tools|ApplyPatchHandler::new" codex-rs\core\src\tools\spec_plan.rs
rg -n "pub\\(crate\\) async fn handle_output_item_done|ToolRouter::build_tool_call" codex-rs\core\src\stream_events_utils.rs codex-rs\core\src\tools\router.rs
rg -n "parse_patch|verify_apply_patch_args|intercept_apply_patch|assess_patch_safety" codex-rs\core\src\tools\handlers\apply_patch.rs codex-rs\core\src\apply_patch.rs
rg -n "load_project_instructions|build_initial_context|record_conversation_items|for_prompt|normalize_history" codex-rs\core\src
```
