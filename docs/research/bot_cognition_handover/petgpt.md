# PetGPT cognition research handover

Evidence labels: **Fact** is directly visible in the upstream snapshot or
source tree; **Design claim** is stated in README/design material and is not
treated as implementation proof; **Inference** is the architectural assessment.
This handover discusses application-level decision artifacts, not private
provider chain-of-thought.

## Snapshot

- **Fact:** The requested upstream is [JulesLiu390/PetGPT](https://github.com/JulesLiu390/PetGPT), default branch `main`, reviewed at commit
  `8b63d5144da41b88c20f59a7427b3d0e999c072f` (`feat(qq): add managed native QQ connector`, 2026-08-02).
- **Fact:** The repository describes a Tauri/Rust + React desktop pet with
  unified LLM adapters, MCP tools, SQLite conversation persistence, local
  memory, mood/expression state, and a QQ social-agent integration.
- **Inference:** PetGPT is best understood as a desktop companion with a
  prompt/tool orchestration layer and a specialized social-agent subsystem,
  rather than a typed character-cognition runtime.

## Verified source evidence

| Area | Evidence | Classification |
| --- | --- | --- |
| Product and capabilities | [`readme.md`](https://github.com/JulesLiu390/PetGPT/blob/8b63d5144da41b88c20f59a7427b3d0e999c072f/readme.md) | Fact/README claim |
| Social orchestration | `src/utils/socialAgent.js`, `socialPromptBuilder.js`, `socialControlScope.js`, `socialTargetType.js`, `socialToolExecutor.js` in the [source tree](https://github.com/JulesLiu390/PetGPT/tree/8b63d5144da41b88c20f59a7427b3d0e999c072f/src/utils) | Fact: source ownership exists |
| LLM and tool boundary | `src/utils/llm/`, `src/utils/mcp/`, `src/utils/subagentCapability.js`, `src/utils/subagentManager.js` | Fact: source ownership exists |
| Persistence | `src-tauri/src/database/*.rs`, including conversations, messages, pets, settings, and MCP servers | Fact: SQLite data layer exists |
| Social behavior | README section “Social Agent — Autonomous Group Chat Participation” | Design/README claim until every branch is source-verified |
| Memory redesign | [`memory module documents/README.md`](https://github.com/JulesLiu390/PetGPT/blob/8b63d5144da41b88c20f59a7427b3d0e999c072f/memory%20module%20documents/README.md) and linked `memory-definition.md`, `user-definition.md`, `soul-definition.md`, `system-prompt-injection.md`, `file-operations.md` | Design document, not automatically shipped behavior |
| Planned intent/subagents | [`docs/superpowers/plans`](https://github.com/JulesLiu390/PetGPT/tree/8b63d5144da41b88c20f59a7427b3d0e999c072f/docs/superpowers/plans) and [`specs`](https://github.com/JulesLiu390/PetGPT/tree/8b63d5144da41b88c20f59a7427b3d0e999c072f/docs/superpowers/specs) | Design/plan evidence |

## End-to-end cognition flow

### Facts and README claims

The README describes three concurrent loops per monitored group:

1. **Fetcher** polls targets and writes raw messages to a shared in-memory
   buffer.
2. **Observer** reads the stream in lurk mode, archives group rules in
   `GROUP_RULE_{id}.md`, and maintains global `SOCIAL_MEMORY.md`; it does not
   send messages.
3. **Reply** detects new messages by watermark comparison, decides whether to
   speak, and sends through a `send_message` tool.
4. A separate **Intent** loop evaluates the character's subjective reaction
   and emits one of five willingness tiers: zero interest, irrelevant,
   tentative thought, wants to chat, or compelled to speak. Tiers 1–2 put the
   reply path to sleep; tiers 3–5 let it consider replying.

The README also describes normal, semi-lurk, and full-lurk modes, a two-slot
catch-up queue for messages arriving during reply generation, and QQ support
through Amadeus-QQ-MCP. Telegram, WhatsApp, and Discord are marked planned.

### Assessment

The visible shape is an ingestion/archive/decision/reply pipeline with a
separate willingness signal. It has a useful separation between observing a
group and sending a message, but the supplied evidence does not show a typed
episode containing parallel appraisals, competing goal bids, workspace
collapse, or a deterministic semantic state commit before reply rendering.
The Intent tier is a control input to Reply, not an explicit cognitive
trajectory record.

## Memory/retrieval/persona/state

- **Fact:** The README claims long-term memory extraction, per-assistant
  memory banks, a per-conversation memory toggle, custom personality/system
  instructions, mood detection, and per-conversation mood state.
- **Fact:** The current tree has local Rust SQLite tables/modules for pets,
  conversations, messages, settings, and MCP servers, plus frontend utilities
  for prompts, mood detection, social state, and tool execution.
- **Design claim:** The memory-module design proposes replacing DB text/JSON
  fields and separate `longTimeMemory()`/`processMemory()` calls with
  `SOUL.md`, `USER.md`, and `MEMORY.md`. `SOUL.md` is always read; the other
  files are read and writable when the memory toggle is enabled. The model
  would use read/write/edit tools and the files would be injected as full text
  into the system prompt.
- **Inference:** This is transparent and user-editable, but full-text prompt
  injection does not itself provide evidence provenance, conflict resolution,
  memory authority, temporal decay, or a distinction between retrieved fact and
  character stance. The design intentionally chooses simplicity over vector
  retrieval and formal memory lifecycle.
- **Fact/design boundary:** The README’s “local memory” feature and the
  memory-module documents should be checked against the current runtime before
  claiming the Markdown-file replacement is deployed. The source tree proves
  relevant utilities and persistence modules, not every README/design feature.

## Reasoning/monologue visibility

- **Fact/README claim:** The Intent loop is called “Inner Monologue” and its
  five-tier willingness output is injected into the Reply prompt with high
  recency.
- **Inference:** This is an inspectable semantic control signal, but it is not
  provider-private reasoning and should not be confused with a complete,
  replayable explanation of stance, evidence, or state change.
- **Fact:** The repository contains `intentToolLedger.js` and social prompt/
  control modules, suggesting explicit bookkeeping around intent/tool context.
  Their presence alone does not prove a protected trace policy or a stable
  cross-stage schema.

## Tools/actions/output

- **Fact/README claim:** MCP servers and built-in QQ MCP integration expose
  external tools; the LLM can call tools automatically. Per-assistant Skills
  are progressively loaded and read-only at runtime; Skills explain tool
  composition but do not grant permissions.
- **Fact:** The source tree contains MCP clients/executors, a social-tool
  executor, subagent capability/manager modules, and a Rust MCP manager.
- **Inference:** PetGPT has a practical capability/action boundary and a
  platform connector path. The available evidence does not establish that
  tool results re-enter a typed cognition stage before visible output or that
  action authorization is owned separately from the model’s decision.
- **Fact:** Ordinary chat output is rendered by the desktop UI; social output
  is sent through the social `send_message` capability. SQLite stores
  conversation history and supports session resume.

## Bounds/failure/observability

- **Fact/README claim:** Social Intent uses a one-minute cooldown in semi/full
  lurk, Reply has a two-round catch-up queue, and watermark comparison avoids
  reprocessing old messages.
- **Fact/design claim:** The memory documents require file truncation and
  per-conversation memory gating; the repository contains tests for social
  contracts, intent ledgers, control scope, and subagent capabilities.
- **Inference:** These are useful operational bounds and testable contracts,
  but the supplied evidence does not show typed semantic failure classes,
  bounded regeneration of invalid stance, a resolver recurrence budget, or a
  protected semantic replay trace. File-based memory updates can also become a
  live-turn side effect unless deterministic write authorization and
  idempotency are enforced.

## Strengths

1. Social participation is decomposed into observation, intent, and reply,
   which supports lurk modes and reduces accidental sending.
2. The five-tier willingness scale gives the model a compact, user-facing
   product control for “wanting to speak” instead of forcing every message
   into a reply.
3. Local, transparent Markdown memory is easy for a user to inspect, edit, and
   reset; per-assistant and per-conversation scopes are product-friendly.
4. MCP, skills, subagents, and unified provider adapters make capability
   extension practical for a desktop application.
5. Watermarks, catch-up limits, cooldowns, and local persistence address the
   operational realities of long-running group participation.

## Limitations

1. No supplied source evidence establishes a typed semantic trajectory or
   stance state. Intent tier and mood are compact signals, not appraisal/goal/
   workspace contracts.
2. Observer-produced Markdown and prompt-injected memory can influence Reply
   without a universal evidence handle, authority, freshness, or conflict
   policy.
3. The social README describes model decisions and tool sending, but the
   supplied source slice does not prove commit-before-expression, independent
   action authorization, or a deterministic prohibition on a stale/incorrect
   Intent signal causing a reply.
4. Full-text memory is simple but scales with prompt size and can make stale or
   user-edited prose compete with current episode evidence.
5. Several important claims are README or design-plan claims, especially the
   memory-file replacement and some social-platform support. They need runtime
   verification before being treated as stable contracts.

## Implications for Kazusa ECT

- Reuse the social decomposition as an operational pattern: a read-only
  observer can collect scoped evidence, a willingness/intention branch can
  propose a reason to speak, and a reply surface can render only after the
  semantic decision is committed.
- Keep the Intent analogue inside goal bidding/workspace collapse. It should
  be evidence-linked and target-scoped, not a standalone score that authorizes
  Reply.
- Treat `SOCIAL_MEMORY.md`, group rules, and user memory as source-owned
  evidence lanes. RAG or Observer output must return provenance; Cognition must
  interpret it and decide stance.
- Put social `send_message` behind Kazusa’s modality-neutral action request,
  deterministic target/permission/idempotency checks, and commit-before-
  expression ordering.
- Use PetGPT’s catch-up/watermark ideas for bounded resolver recurrence, while
  retaining typed observations, recurrence caps, duplicate-request handling,
  and protected semantic traces.
- Preserve a three-way distinction: provider-private reasoning is not a
  contract; the ECT is a bounded semantic trace; optional private residue is
  scoped continuity rather than truth.

## Evidence matrix

| Dimension | PetGPT evidence | Assessment versus ECT |
| --- | --- | --- |
| Admission | Fetcher/watermark and lurk modes; Reply checks new messages | Practical scheduling, not a typed admission judgment |
| Speak decision | Five-tier Intent willingness injected into Reply | Useful goal/willingness input; not workspace collapse |
| Memory | Observer files; local memory claims; SOUL/USER/MEMORY design | Transparent evidence source, weak provenance/authority contract |
| Persona/affect | System personality, mood detection, per-chat expressions | Prompt/state features; no evidenced causal affect reducer |
| Tools | MCP, Skills, social send, subagent modules | Extensible capabilities; authorization/stance boundary unresolved |
| Output | Desktop chat and social `send_message` | Separate delivery surfaces; commit-before-expression not evidenced |
| Bounds | Cooldowns, two catch-up rounds, watermarks, toggles | Good operational containment; semantic failure policy unresolved |
| Trace | Intent ledger/tests and persisted chat history | Not a protected semantic replay trace |
| Typed trajectory | No supplied contract for appraisals, bids, selected stance, commit | Not evidenced in PetGPT |

## Open questions

- Which parts of the README social pipeline are currently wired in the
  default build, and which are still under active development?
- Are `SOUL.md`/`USER.md`/`MEMORY.md` implemented in the current commit or only
  specified in the memory-module documents?
- What exact schema and persistence policy do `socialAgent.js` and
  `intentToolLedger.js` use for Intent results and watermarks?
- Does `socialToolExecutor.js` independently validate target scope, permission,
  idempotency, and delivery success before sending?
- Can Observer evidence conflict with current messages, and how is freshness or
  user-edited memory reconciled?
- Are social-loop failures surfaced as typed private observations, visible
  clarification, silence, or retry, and what is the maximum end-to-end work?
