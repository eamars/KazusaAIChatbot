# AstrBot cognition research handover

Evidence labels: `[Fact]` means supplied source/tree evidence; `[Documentation
claim]` means an official guide or architecture page describes the behavior;
`[Inference]` means an architectural interpretation of that evidence. This
handover describes observable contracts and provider-facing reasoning fields,
not private token-level chain-of-thought.

## Snapshot

- `[Fact]` The reviewed official master is AstrBot commit
  `a9bb8a64ca69657e6262e3ca06541ecaf3a6d1ca`, dated 2026-08-12. Its latest
  commit fixes OpenAI HTTP client test stability.
- `[Fact]` The current tree contains the tool-loop runner, context compressor,
  tool implementation, internal agent substage, conversation manager, persona
  manager, and knowledge-base retrieval manager listed in the supplied
  evidence.
- `[Inference]` The stable-core evidence shows a bounded general agent/tool
  execution architecture with context, persona, and retrieval services. It
  does not establish a typed character cognition trajectory.

## Verified evidence

- `[Fact]` `ToolLoopAgentRunner` runs a `while` loop bounded by `max_step`. It
  defines a maximum-step notice, repeated-tool notices, tool-result token caps
  with spillover, three empty-output retry attempts, and separate handling of
  provider `reasoning_content` as a `ThinkPart` rather than completion text.
  [Source](https://github.com/AstrBotDevs/AstrBot/blob/a9bb8a64ca69657e6262e3ca06541ecaf3a6d1ca/astrbot/core/agent/runners/tool_loop_agent_runner.py)
- `[Fact]` `InternalAgentSubStage` defaults `max_agent_step` to 30 and
  `display_reasoning_text` to false. It builds and runs the main agent,
  records `astr_agent_prepare` and `astr_agent_complete`, and saves
  conversation history.
  [Source](https://github.com/AstrBotDevs/AstrBot/blob/a9bb8a64ca69657e6262e3ca06541ecaf3a6d1ca/astrbot/core/pipeline/process_stage/method/agent_sub_stages/internal.py)
- `[Documentation claim]` The official plugin guide documents direct LLM
  calls, function tools, and an agent-as-tool pattern for multi-agent
  composition.
  [Plugin guide](https://github.com/AstrBotDevs/AstrBot/wiki/en-dev-star-guides-ai)
- `[Documentation claim]` Official context-compression documentation says
  compression begins at 82% of the model context window. The default drops old
  turns; optional LLM summarization retains recent context and then rechecks
  and halves the context if necessary.
  [Compression guide](https://github.com/AstrBotDevs/AstrBot/wiki/en-use-context-compress)
- `[Fact]` MindSim is PR #6888, titled “feat: add MindSim - an event-driven
  agent thinking framework for advanced personas.” The review describes it as
  a large core architecture change. It is PR/optional/unmerged evidence, not
  stable-core evidence.
  [PR #6888](https://github.com/AstrBotDevs/AstrBot/pull/6888)

## Cognition flow

- `[Fact]` The evidenced execution path is: internal pipeline substage builds
  the main agent; the runner asks the model for a completion; a tool call can
  execute and append a bounded result; the loop continues until completion,
  an empty-output retry limit, or the step bound; the substage records
  lifecycle events and history.
- `[Documentation claim]` AstrBot’s official architecture page and plugin
  guide present LLM calls, tools, and agent-as-tool composition as the main
  extensibility model.
  [Architecture](https://docs-v3.astrbot.app/dev/core/overall_architecture.html)
- `[Inference]` This is an agent execution loop, not evidence of a stable-core
  sequence of typed observation, appraisal, goal competition, stance selection,
  and state commit. MindSim may signal that direction experimentally, but the
  supplied evidence does not promote it to the current architecture.

## Memory/persona/tools/output

- `[Fact]` Dedicated conversation, persona, and knowledge-base retrieval
  managers exist in the current tree. Their presence alone does not prove a
  typed memory, relationship, or persona-state contract.
- `[Documentation claim]` Function tools and agent-as-tool composition are
  official extension patterns. The supplied evidence establishes capability
  composition, but not the exact schemas, provenance rules, or authorization
  boundaries of every tool.
- `[Fact]` Tool results are bounded by token caps and may spill over; context
  compression has a documented threshold and optional summary path.
- `[Inference]` These mechanisms primarily manage prompt/context pressure and
  execution capacity. They should not be read as proof of durable memory
  consolidation, causal relationship updates, or cognition-owned stance.
- `[Fact]` The normal completion text is distinct from provider reasoning
  content, and the internal substage saves conversation history.

## Reasoning visibility

- `[Fact]` Provider `reasoning_content` is represented separately as a
  `ThinkPart`; it is not merged into the completion text by the cited runner.
- `[Fact]` `display_reasoning_text` defaults to false in the cited internal
  substage.
- `[Inference]` The default surface separates ordinary output from reasoning
  material, but the supplied evidence does not prove access control, retention
  policy, or a typed semantic explanation for each decision. A `ThinkPart` is
  a provider/runtime reasoning representation, not an ECT record.

## Bounds/failure/observability

- `[Fact]` The runner has a hard loop bound, repeated-tool notices, tool-result
  caps/spillover, maximum-step messaging, and three empty-output retries.
- `[Fact]` The substage has a default 30-step agent bound, emits prepare and
  complete lifecycle events, and persists conversation history.
- `[Documentation claim]` Context compression starts before the full context
  window is consumed and can use a bounded summary/recheck path.
- `[Inference]` These are useful operational safeguards for runaway work,
  repeated calls, oversized results, and silent model output. They do not by
  themselves provide semantic failure classes, evidence-linked state
  transitions, selected-stance replay, or a protected-versus-safe trace model.

## Strengths

- `[Inference]` AstrBot has a clear and practical bounded execution primitive:
  tool loops terminate, repeated behavior is surfaced, tool payloads are
  capped, and empty responses receive limited recovery.
- `[Inference]` Separating provider reasoning content from completion text gives
  a clean basis for output-surface policy and avoids treating all generated
  material as visible dialog.
- `[Documentation claim]` The official tool and agent-as-tool patterns provide
  a straightforward route to specialist composition without requiring every
  capability to be a monolithic agent.
- `[Inference]` Lifecycle events and saved history give operators a basic
  correlation point around agent preparation and completion.

## Limitations

- `[Inference]` Based on the supplied evidence, a typed semantic
  trajectory/stance state is **not evidenced in stable core**. There is no
  supplied stable-core contract for an observation, appraisal, competing goal
  bids, selected stance/intention, and deterministic state commit carried as a
  typed trajectory.
- `[Inference]` The existence of persona, conversation, and retrieval managers
  does not show that retrieved material is evidence-separated from persona or
  that memory changes are causal, scoped, and reviewable.
- `[Inference]` A step limit bounds compute but can terminate an unfinished
  task; the supplied evidence does not specify the complete user-visible and
  resumable semantics of maximum-step termination.
- `[Inference]` A separate `ThinkPart` improves representation hygiene, but it
  is not the same as a bounded semantic trace and should not be exposed as
  private chain-of-thought.
- `[Fact]` MindSim’s PrivateBrain/event-driven loop, actions, topic/person
  memory, and multi-model thinking/reply/function/deep-think roles remain PR
  claims and experimental design signals only.

## Kazusa ECT implications

- `[Inference]` Use AstrBot as a reference for bounded agent execution,
  repeated-tool handling, result-size control, empty-output recovery, and
  separation of provider reasoning from visible completion. These are useful
  implementation patterns around an ECT capability or resolver.
- `[Inference]` Keep ECT ownership in Kazusa’s semantic stages: cognition
  interprets evidence and selects stance/intention; deterministic code owns
  validation, permissions, limits, persistence, execution, and delivery; RAG
  returns evidence; dialog renders the committed result.
- `[Inference]` An AstrBot-style tool loop should sit inside a declared
  resolver/capability boundary with typed input/output, provenance, refusal
  conditions, and a hard budget. It should not become the owner of character
  stance through free-form tool-loop prose.
- `[Inference]` AstrBot’s compression policy can inform context budgeting, but
  dropping or summarizing turns cannot substitute for ECT fields such as
  evidence handles, appraisal/state candidates, goal bids, selected stance,
  resolver outcome, and surface status.
- `[Inference]` Kazusa should retain the distinction between model-private
  reasoning, safe semantic trajectory, and optional private residue. Only the
  latter two can have explicitly declared retention or diagnostic projections;
  raw provider reasoning remains outside the ECT contract.

## Evidence matrix

| Area | Evidence supplied | Classification | Consequence |
| --- | --- | --- | --- |
| Loop control | `while`/`max_step`, max-step notice, retries, repeated-tool handling | Fact | Strong bounded-execution primitive |
| Tool composition | Function tools and agent-as-tool guide | Documentation claim | Supports specialist composition; exact contracts remain open |
| Context pressure | 82% compression threshold, drop-old or summarize/recheck | Documentation claim | Context management, not durable cognition state |
| Output separation | `reasoning_content` -> `ThinkPart`; display flag false | Fact | Reasoning and completion have separate surfaces |
| Lifecycle | `astr_agent_prepare`/`astr_agent_complete`, history save | Fact | Basic operational correlation and persistence |
| Typed trajectory/stance | No supplied stable-core evidence; MindSim is unmerged PR | Inference/assessment | Not evidenced in stable core; do not treat PR as current contract |

## Open questions

- `[Open question]` What exact schemas and ownership boundaries do the
  conversation, persona, and retrieval managers expose at runtime?
- `[Open question]` Are tool and agent-as-tool calls validated with typed
  provenance, permissions, refusal conditions, and per-capability budgets?
- `[Open question]` What user-visible result, resume behavior, and trace data
  follow maximum-step termination or repeated-tool detection?
- `[Open question]` Where is `ThinkPart` retained, who can view it, and how is
  `display_reasoning_text` applied across adapters and operator surfaces?
- `[Open question]` Does any branch or release actually include MindSim, or is
  PR #6888 still fully experimental? This must be answered before using it as
  implementation evidence.
- `[Open question]` Does the stable core expose any typed stance, intention,
  relationship transition, or evidence-linked semantic record beyond the
  supplied runner/substage observations?
