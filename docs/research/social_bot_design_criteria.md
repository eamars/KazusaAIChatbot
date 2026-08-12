# Neutral Criteria for a Suitable Social Bot Design

## Purpose

These criteria define what makes a social chatbot design suitable for
long-running one-to-one and group conversation. They evaluate user-visible
outcomes and engineering trade-offs rather than rewarding one internal
architecture. A design can satisfy a criterion with a state machine, prompt
orchestration, a tool loop, typed cognition, or a combination of them.

The criteria are used by the replacement comparison in
[bot_cognition_apple_to_apple_comparison.md](bot_cognition_apple_to_apple_comparison.md).

## Evaluation principles

1. **Compare outcomes before mechanisms.** Ask whether the bot can behave well
   in the situation, then record how the design achieves it.
2. **Separate evidence from aspiration.** Source-verified behavior, official
   documentation claims, and reviewer inference receive different labels.
3. **Score demonstrated scope.** Optional, experimental, planned, or
   deployment-dependent features are scored only for the scope supported by
   evidence.
4. **Reward useful simplicity.** More stages, fields, agents, or traces do not
   earn points by themselves; they count only when they improve a criterion
   without unacceptable latency, cost, or operational burden.
5. **Expose trade-offs instead of forcing one winner.** The final analysis
   reports where a design is strong, where it is weak, and which use case makes
   each trade-off reasonable.

## Criteria

The comparison uses ten criteria. The default interpretation is equal weight;
an operator may reweight them for a particular deployment, but the report must
show the changed weights.

| ID | Criterion | Core question | Evidence to look for |
| --- | --- | --- | --- |
| C1 | **Social relevance and reason to respond** | Does the bot recognize when a message matters, when it should speak, when it should wait, and when it should leave the interaction alone? | Mention and non-mention behavior, relevance/attention handling, silence/exit decisions, thread fit, duplicate suppression, and examples of response gating. |
| C2 | **Conversation and group awareness** | Can it follow participants, topics, turns, references, episodes, interruptions, and group noise over time? | User/target binding, thread or topic state, watermarks, snapshots, catch-up behavior, episode state, and handling of concurrent messages. |
| C3 | **Continuity and memory quality** | Does it remember useful information at appropriate times and scopes while limiting stale, conflicting, irrelevant, or private material? | Short/long/episodic memory, retrieval ranking, summaries, decay, consolidation, correction/deletion, provenance or source labels, privacy scopes, and restart behavior. |
| C4 | **Personality, personalization, and emotional adaptation** | Does the bot maintain a recognizable identity while adapting tone, mood, boundaries, and relationship behavior to different people and situations? | Persona handling, per-user state, relationship or mood signals, style adaptation, character learning, state persistence, and evidence of conflict control. |
| C5 | **Agency and interaction management** | Can it choose useful interaction modes—answer, ask, wait, defer, follow up, act, or stay silent—and sustain or close a conversation naturally? | Explicit or implicit action choices, willingness/priority, open-loop tracking, follow-through, proactive contact, scheduler behavior, and user-control boundaries. |
| C6 | **Response quality and expression control** | Does the final message fit the selected social purpose, target, platform, and character voice? | Separation or coordination of decision and wording, target/quote handling, formatting, multimodal surfaces, delivery ownership, rewrite behavior, and visible-output validation. |
| C7 | **Capability and tool usefulness** | Do tools, retrieval, plugins, MCP, subagents, and scheduled jobs expand useful behavior without making the bot unpredictable? | Tool discovery, argument validation, capability scope, permission checks, result handling, specialist composition, idempotency, and returned-error behavior. |
| C8 | **Reliability, latency, and graceful degradation** | Does the bot remain usable under model errors, slow tools, high message volume, long context, restarts, and partial failures? | Timeouts, loop and token caps, queues/backpressure, retries, fallbacks, compression, stale-work invalidation, persistence/recovery, and end-to-end latency evidence. |
| C9 | **Safety, privacy, and user control** | Does the design protect private content, limit unwanted contact and actions, and let operators or users inspect, correct, disable, or delete durable behavior? | Data boundaries, privacy filtering, authorization, delivery controls, auditability, editable memory, retention/deletion, configuration gates, and prompt/tool isolation. |
| C10 | **Operability, extensibility, and total cost** | Can maintainers deploy, test, observe, extend, and afford the system over time? | Platform adapters, model portability, plugin architecture, testability, diagnostics/replay, configuration burden, resource requirements, dependency complexity, and operational maintenance. |

## Scoring rubric

Use a 0–5 score for each criterion, with a confidence marker:

| Score | Meaning |
| --- | --- |
| **0** | No meaningful support shown, or the design clearly works against the criterion. |
| **1** | Minimal or mostly incidental support; important cases are unhandled or undocumented. |
| **2** | Partial support for common cases, with material gaps or uncertain ownership. |
| **3** | Solid practical support for the documented scope, with known limitations. |
| **4** | Strong, broad, and operationally supported capability with only bounded gaps. |
| **5** | Exceptional support demonstrated across difficult cases, continuity, failure, and operator control. |

Confidence is independent of score:

- **H:** source behavior is directly verified in the pinned snapshot;
- **M:** the mechanism is partly verified or depends on configuration, while
  the broader behavior is documented;
- **L:** the assessment relies mainly on documentation, design plans, or an
  inference that needs runtime confirmation.

The score is not a benchmark result. It is a structured reading of available
evidence. Missing evidence should reduce confidence and may limit the score,
but absence from the inspected snapshot should not be described as proof that a
feature can never exist.

## Comparison output requirements

The neutral comparison should include:

- the same C1–C10 rows for Kazusa and every external project;
- a short evidence note and confidence marker for every material score;
- a total or profile summary only after showing the individual dimensions;
- separate “best at,” “weakest at,” and “trade-off” findings;
- a distinction between current source behavior, official claims, optional
  features, and reviewer inference;
- no bonus for matching Kazusa terminology such as appraisal, workspace,
  evidence handle, trajectory, or deterministic commit;
- no penalty merely because a system uses a simpler mechanism when the social
  outcome is comparably useful;
- a stated limitation that repository review cannot substitute for live
  behavioral benchmarking.
