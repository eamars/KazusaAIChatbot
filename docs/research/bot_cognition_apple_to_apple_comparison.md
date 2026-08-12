# Social-Bot Cognition Designs: Evidence-Based Comparison

## Document purpose

This document compares six social-bot designs on social-bot outcomes and
engineering trade-offs, not on internal architecture fashion. It uses the
evaluation standard in
[`social_bot_design_criteria.md`](social_bot_design_criteria.md) and the five
research handovers in `bot_cognition_handover/` as its evidence base. Kazusa
is represented from its local repository documentation only (see the Kazusa
evidence note below); the external projects are represented from pinned
upstream snapshots described in their handovers.

This is a documentation review, not a live behavioral benchmark. Every score
is a structured reading of the available evidence with a confidence marker,
and absence of evidence in a reviewed snapshot is never treated as proof that
a feature cannot exist.

## Executive summary

The six designs fall into three clusters on the ten outcome criteria:

| Cluster | Projects | Profile |
| --- | --- | --- |
| Full social-cognition systems | Kazusa (38/50), MaiBot (38/50) | Deep memory, explicit decision structure, relationship/affect state, strong control boundaries. Highest total scores but also the highest complexity and per-turn model-call cost. |
| Conversation-aware agent systems | Hermes (36/50), ATRI-bot (32/50) | Strong group awareness, structured decisions, rich memory, broad tools, moderate complexity. |
| Lightweight/agent-first systems | PetGPT (28/50), AstrBot (22/50) | Lower operational cost and simpler mental models; capability breadth and some social gating exist, but continuity, adaptation, and control-boundary evidence is thinner. |

No project is a default winner. Kazusa and MaiBot tie on the equal-weight
total but lead for different reasons: Kazusa scores highest on per-user
emotional adaptation and safety/control boundaries; MaiBot leads on verified
tool breadth, observability, and engineering pragmatism. Hermes has the
strongest verified group-conversation awareness of any reviewed project, and
ATRI-bot has the most structured cheap decision carrier outside the top two.
PetGPT is the most approachable desktop companion, and AstrBot is the most
generic building block for a custom agent rather than a ready social bot.

The headline trade-off: explicit cognitive structure buys stronger continuity,
per-user differentiation, and control boundaries at a real latency, cost, and
operational burden. Simpler prompt-and-tool designs win where the requirement
is responsive group chat, low cost, or operator transparency, because they
need fewer model calls and have fewer failure modes. ECT-style structure earns
its cost when the product needs durable relationship/affect state, grounded
silence, and deterministic permission and privacy ownership over a long
running relationship.

## Evaluation method

### Evidence classes

Evidence in the notes and profiles is labeled as follows:

| Label | Meaning |
| --- | --- |
| **[V] Verified** | Directly observed in the pinned upstream snapshot (external projects) or the current local repository contracts/tests (Kazusa). |
| **[D] Documentation claim** | Stated by official documentation or READMEs, not matched to implementation in the reviewed slice. |
| **[O] Optional / experimental** | Exists in source or docs but is disabled by default, configuration-dependent, or marked experimental. |
| **[I] Inference** | Architectural assessment derived from the verified evidence. |

### Criteria

The ten criteria are taken unchanged from
[`social_bot_design_criteria.md`](social_bot_design_criteria.md). They evaluate
outcomes first and mechanisms second. A design earns points by producing the
social outcome with acceptable engineering trade-offs, regardless of whether
it uses typed cognition, prompt orchestration, a tool loop, a state machine,
or a combination.

| ID | Criterion | Core question |
| --- | --- | --- |
| C1 | Social relevance and reason to respond | When to speak, wait, or leave an interaction alone; mention and non-mention behavior; grounded silence/exit decisions. |
| C2 | Conversation and group awareness | Participants, topics, turns, references, episodes, interruptions, and group noise over time. |
| C3 | Continuity and memory quality | Useful memory at the right time and scope; staleness, conflict, privacy, provenance, and restart behavior. |
| C4 | Personality, personalization, emotional adaptation | Recognizable identity plus adaptation of tone, mood, boundaries, and relationship behavior per person/situation. |
| C5 | Agency and interaction management | Choosing answer, ask, wait, defer, follow up, act, or stay silent; sustaining and closing conversation. |
| C6 | Response quality and expression control | Final message fits the selected social purpose, target, platform, and character voice. |
| C7 | Capability and tool usefulness | Tools/retrieval/plugins/MCP/subagents/scheduled jobs expand behavior without making the bot unpredictable. |
| C8 | Reliability, latency, graceful degradation | Usable under model errors, slow tools, volume, long context, restarts, and partial failures. |
| C9 | Safety, privacy, user control | Protection of private content, limits on unwanted contact/action, operator/user inspect/correct/disable/delete. |
| C10 | Operability, extensibility, total cost | Deploy, test, observe, extend, and afford the system over time. |

### Scoring rubric and confidence

Scores use the 0-5 rubric from the criteria file: 0 (no meaningful support),
1 (minimal/incidental), 2 (partial with material gaps), 3 (solid for the
documented scope with known limits), 4 (strong and broad with bounded gaps),
5 (exceptional across difficult cases, continuity, failure, and operator
control).

Confidence is independent of score:

- **H** — source behavior directly verified in the pinned snapshot;
- **M** — mechanism partly verified or configuration-dependent, broader
  behavior documented;
- **L** — assessment relies mainly on documentation, design plans, or
  inference requiring runtime confirmation.

### Neutrality rules applied

- Kazusa's ECT vocabulary (appraisals, evidence handles, workspace collapse,
  commit-before-dialog) is treated as one design choice among alternatives. No
  point is awarded for having those mechanisms; they are credited only where
  the documented mechanism plausibly produces the outcome the criterion asks
  for (grounded silence, per-user differentiation, consistency, control).
- No penalty is applied for simpler mechanisms when the social outcome is
  comparably useful.
- Private provider chain-of-thought is not compared. Only observable semantic
  and control artifacts (structured decisions, state records, output markers,
  memory contracts, tool boundaries) are scored.
- Totals are reported after the dimension-level matrix, use equal weighting as
  a review convention, and are not a live behavioral benchmark.

## Score matrix

Each cell shows **score (confidence)**.

| Criterion | Kazusa | MaiBot | AstrBot | PetGPT | ATRI-bot | Hermes |
| --- | --- | --- | --- | --- | --- | --- |
| C1 Social relevance / reason to respond | 4 (M) | 4 (H) | 1 (M) | 3 (M) | 3 (M) | 4 (H) |
| C2 Conversation and group awareness | 4 (M) | 4 (H) | 2 (M) | 3 (M) | 3 (M) | 5 (H) |
| C3 Continuity and memory quality | 4 (M) | 4 (H) | 2 (L) | 2 (L) | 3 (M) | 4 (H) |
| C4 Personality / personalization / emotional adaptation | 4 (M) | 3 (M) | 2 (L) | 3 (M) | 3 (M) | 2 (M) |
| C5 Agency and interaction management | 4 (M) | 4 (H) | 2 (M) | 3 (M) | 3 (M) | 4 (H) |
| C6 Response quality and expression control | 4 (M) | 4 (H) | 2 (M) | 2 (M) | 3 (M) | 3 (H) |
| C7 Capability and tool usefulness | 3 (M) | 4 (H) | 3 (M) | 3 (M) | 4 (M) | 4 (M) |
| C8 Reliability / latency / graceful degradation | 4 (M) | 4 (H) | 3 (H) | 3 (M) | 3 (M) | 4 (H) |
| C9 Safety / privacy / user control | 4 (M) | 3 (H) | 2 (L) | 3 (M) | 3 (M) | 3 (H) |
| C10 Operability / extensibility / total cost | 3 (M) | 4 (H) | 3 (M) | 3 (M) | 4 (M) | 3 (M) |
| **Total (equal weight, max 50)** | **38** | **38** | **22** | **28** | **32** | **36** |

The totals are a secondary summary. Equal weighting is a review convention,
not a deployment recommendation: an operator should reweight C1-C10 for their
target use case (see Use-case recommendations). Repository evidence measures
designed and documented capability, not measured behavior in production.

## Evidence notes

Notes are keyed by criterion. They separate verified facts, documentation
claims, optional/experimental features, and inference, and carry the same
confidence marker as the matrix.

### Kazusa (local repository evidence)

Evidence basis: `src/kazusa_ai_chatbot/cognition_core_v2/README.md`,
`src/kazusa_ai_chatbot/cognition_resolver/README.md`,
`src/kazusa_ai_chatbot/local_context_resolver/README.md`,
`src/kazusa_ai_chatbot/rag/README.md`, and
`docs/architecture/explicit_cognitive_trajectory_architecture.md`. The
architecture document is a **draft target** with no execution authority;
implemented behavior is anchored in the Cognition Core V2 package, the
cognition resolver loop, the local-context resolver (RAG3), and retained RAG
helpers. No external pinned commit exists for Kazusa in this review.

- **C1 (4, M)** — [V] Targetless group self-cognition chooses
  `stay_silent` or `propose_visible_reply` with up to four cited evidence
  handles, a closed participation basis, and a semantic reason; silence is a
  first-class decision, and admission/relevance is separate from stance
  ([D] architecture, [V] contracts in `cognition_core_v2`). [I] Live
  reason-to-speak quality is not benchmarked.
- **C2 (4, M)** — [V] Group-scene turns older than 120 minutes are dropped
  deterministically; episode-local `pN` participant bindings, public scene
  order, current-user binding, and reply-parent context are typed; progress
  packets have retention tiers. [I] Behavior in noisy real groups is
  documented, not measured.
- **C3 (4, M)** — [V] Retention lanes (episode, progress, per-user, shared,
  reflection, identity), RAG3 evidence with provenance/scope and prompt-safe
  projection, cache invalidation on durable writes, and recall evidence for
  agreements/commitments. [I] Retrieval quality and memory usefulness are not
  live-tested here.
- **C4 (4, M)** — [V] Multi-axis per-user relationship state, deterministic
  emotion lifecycle derivation from typed causes (21 formulas with tests),
  immutable character-identity revisions, and qualitative model projections.
  [I] Whether this produces noticeably better adaptation in real use is
  unmeasured; the complexity cost is real.
- **C5 (4, M)** — [V] Goal branches include silence, clarification, and
  deferral; a bounded resolver loop re-enters cognition with observations;
  action planning is modality-neutral; pending clarification and background
  task continuation exist. [I] Interaction outcomes depend on the LLM's
  branch quality, which is not benchmarked.
- **C6 (4, M)** — [V] Text-surface planning plus focused dialog verifiers
  (semantic fidelity, role direction, surface integrity, expression
  continuity), addressee plan enforcement, and no post-generation text
  rewriting. [I] Wording quality is unmeasured; the verifier stack adds
  per-turn cost.
- **C7 (3, M)** — [V] Resolver capabilities (local context, task resolution,
  web/research), deterministic authorization, idempotent background tasks.
  [I] No plugin/MCP marketplace is evidenced; capability breadth is narrower
  than the tool-rich projects, with predictability as the compensating
  strength.
- **C8 (4, M)** — [V] Stage timeouts (120 s default), three-attempt caps,
  failure ladders (`accepted`/`recovered`/`accepted_degraded`/
  `unrecoverable`), fail-closed boundaries, RAG3 caps (4 iterations, 8 nodes,
  depth 3), protected trace. [I] End-to-end latency with six appraisal
  families plus goal/collapse/action/surface/dialog calls is an open cost.
- **C9 (4, M)** — [V] Typed action/resolver authorization, provenance roles,
  sanitized prompt-facing evidence (no raw ids), per-user scope ownership,
  promotion gates for durable growth, protected-vs-semantic trace split.
  [I] Control-console and deletion surfaces are not live-audited here.
- **C10 (3, M)** — [V] Typed contracts, deterministic test suites, module
  ICDs. [I] High per-turn model-call count and state-machine complexity make
  operation and tuning expensive; the architecture itself concedes the
  trajectory "must be earned" by an evidence gap.

### MaiBot

Evidence basis: handover at `main` commit
`1b8d13300a5add5f029106b0ee2d80b59996917c` (`1b8d133`, 2026-08-12), with
high confidence on the Planner/replyer loop, tool routing, memory boundaries,
and bounds; medium confidence on configuration combinations.

- **C1 (4, H)** — [V] Deterministic reply-necessity scheduling (mention,
  content, pressure, frequency; hard trigger score >= 80) plus Planner-level
  judgment; a pure idle period cannot trigger. [I] Lexical scores can miss
  grounded reasons or admit weak ones.
- **C2 (4, H)** — [V] Pending-message cache, focus state, interruption with
  quiet-period retry, group/private context sizes, same-chat/person scope
  checks. [I] Focus tools make cross-chat attention an explicit action.
- **C3 (4, H)** — [V] Mid-term summary/embedding recall, A_Memorix
  vector/graph/BM25/PPR retrieval, evidence-based person profiles, async
  fact/summary writeback with idempotency, episode CAS workers. [O] A_Memorix
  and heuristic recall are configuration-controlled and can be disabled;
  bounded queues drop work under pressure.
- **C4 (3, M)** — [V] Persona as prompt/config input, profile relationship
  text/edges, static experimental `emotion_trait`. [O] Behavior/style/jargon
  learning exists but is marked experimental and defaults off. [I] No
  persistent typed affect/relationship transition engine.
- **C5 (4, H)** — [V] Recurrent Planner tool loop (`reply`, `wait`,
  `tool_search`, memory/profile tools), deferred tools, bounded wait caps,
  focus-mode tools. [I] No generic automatic tool retry; Planner free-form
  text remains the semantic owner.
- **C6 (4, H)** — [V] Replyer owns visible wording, gets filtered real
  history; deterministic post-process/send; hook regeneration capped at 3.
  [O] Rich attachments and structured expression intent are optional.
- **C7 (4, H)** — [V] Builtin/deferred tools, plugins/browser/MCP providers,
  structured tool failures, result caps. [O] Provider availability is
  deployment-dependent.
- **C8 (4, H)** — [V] Context caps, `MAX_INTERNAL_ROUNDS` bound, wait cap,
  top-k/char caps, replyer retry cap, provider retry/timeout, bounded monitor
  store (10k records / 72 h), queues with overflow drop. [I] Exact numeric
  defaults for some bounds were not pinned.
- **C9 (3, H)** — [V] Monitor sanitization and retention, focus-mode
  cross-chat warnings, memory prompts excluding bot-only facts/jokes/
  speculation. [I] Prompt-scoped safeguards are not a universal typed
  authorization boundary; Planner text can influence action without a
  semantic evaluator.
- **C10 (4, H)** — [V] Plugin/MCP/browser extensibility, rich observability
  (prompt previews, monitor events, replay ledger), extensive config.
  [I] Configuration burden is high and many capabilities are optional.

### AstrBot

Evidence basis: handover at `master` commit
`a9bb8a64ca69657e6262e3ca06541ecaf3a6d1ca` (2026-08-12). MindSim is an
unmerged PR (#6888), not stable-core evidence.

- **C1 (1, M)** — [I] No reason-to-speak gating is evidenced in the stable
  core slice; the runner executes an agent for the pipeline event.
- **C2 (2, M)** — [V] Conversation manager and history save exist. [I] No
  episode/thread/participant state is evidenced.
- **C3 (2, L)** — [D] Context compression begins at 82% of the window with
  optional summarization. [I] That is context pressure management, not durable
  memory consolidation; no consolidation/provenance contract evidenced.
- **C4 (2, L)** — [V] Persona manager exists. [I] No persona/affect state or
  adaptation contract evidenced in the slice.
- **C5 (2, M)** — [V] Bounded tool loop, max-step handling, three empty-output
  retries. [I] Interaction-mode choice (wait/defer/silence) is not evidenced.
- **C6 (2, M)** — [V] Provider `reasoning_content` is separated as a
  `ThinkPart`, distinct from completion text. [I] No target/quote/format
  control surface evidenced beyond that split.
- **C7 (3, M)** — [D] Function tools and agent-as-tool composition are
  official patterns. [V] Loop bounds, token caps, spillover, repeated-tool
  notices. [I] Exact schemas, provenance, and authorization of every tool are
  open.
- **C8 (3, H)** — [V] Hard loop bound, 30-step default, tool-result caps,
  retries, lifecycle events. [I] No semantic failure classes; max-step
  termination semantics are unspecified.
- **C9 (2, L)** — [I] No privacy filtering, permission, or audit evidence in
  the supplied slice.
- **C10 (3, M)** — [D] Plugin architecture and wiki guides. [I] Clean building
  block for custom bots; not a ready social bot out of the box.

### PetGPT

Evidence basis: handover at `main` commit
`8b63d5144da41b88c20f59a7427b3d0e999c072f` (2026-08-02). Several social and
memory claims are README/design claims that need runtime confirmation.

- **C1 (3, M)** — [V/D] Five-tier Intent willingness (zero interest to
  compelled) gates the reply path; lurk modes and watermark-based new-message
  detection exist in source ownership. [I] The tier is an injected control
  input, not an independent judge.
- **C2 (3, M)** — [V/D] Per-group fetcher/observer/reply loops, group-rule
  archive files, two-slot catch-up queue, watermark comparison. [I] Scoped to
  the QQ social-agent path.
- **C3 (2, L)** — [V] SQLite persistence for conversations/messages/pets.
  [D] Long-term memory extraction and per-assistant memory banks are README
  claims; SOUL/USER/MEMORY file replacement is a design document, not
  confirmed shipped behavior.
- **C4 (3, M)** — [V/D] Custom personality instructions, mood detection, and
  per-conversation mood state exist as utilities/claims. [I] Transparent and
  editable; no causal affect/relationship reducer evidenced.
- **C5 (3, M)** — [V/D] Lurk modes, reply-path sleep, catch-up rounds, and
  social-agent orchestration. [I] No typed recurrence budget or protected
  replay trace evidenced.
- **C6 (2, M)** — [V] Desktop rendering plus social `send_message` tool.
  [I] Expression planning/validation and commit-before-expression are not
  evidenced.
- **C7 (3, M)** — [V] MCP servers/clients, Skills (read-only at runtime),
  subagent manager, social tool executor. [I] Tool-result re-entry into a
  typed cognition stage is not evidenced.
- **C8 (3, M)** — [V/D] One-minute Intent cooldown, two catch-up rounds,
  watermarks, local persistence. [I] Semantic failure policy and maximum
  end-to-end work are open.
- **C9 (3, M)** — [V] User-editable memory, per-conversation toggle, control
  scope module, tests. [I] File-based memory can be a live-turn side effect;
  independent action authorization not evidenced.
- **C10 (3, M)** — [V] Desktop app, SQLite, MCP/skills/subagents; simple to
  operate and inspect. [D] Several advertised features are still design-plan
  status.

### ATRI-bot

Evidence basis: handover at `main` commit
`3dcae6d33279ced3ad238ba86460e64e48228582` (2026-08-07). Memory and tool
claims are largely README claims; the named modules exist in source.

- **C1 (3, M)** — [D/V] JSON `speak`/`update`/`silence` decision list with
  silence as a real output branch; initiative-chat events. [I] Decisions are
  one model call; the reason-to-speak model is implicit.
- **C2 (3, M)** — [V/D] Per-group/per-user sliding context, group topics,
  reply-parent handling, event bus/queue pipeline. [I] Episode/thread state is
  operational, not typed.
- **C3 (3, M)** — [D/V] Hybrid pgvector + PGroonga recall with RRF fusion,
  importance/access/decay scoring, LLM extraction/consolidation/dedup,
  per-user profiles; modules exist. [I] Provenance handles and evidence-vs-
  stance separation are not evidenced.
- **C4 (3, M)** — [D/V] Per-user JSON profile with relationship, personality,
  style preferences; emotion category in memory. [I] Profile text injection,
  not a multi-axis state reducer.
- **C5 (3, M)** — [D/V] `speak`/`silence`, function-calling loop, sub-agents,
  `schedule_self_trigger` for later group thought. [I] Proactive contact
  requires explicit permission/target/audit boundaries that are not evidenced.
- **C6 (3, M)** — [D/V] Segmented output, emoji/sticker decoration, optional
  TTS, OneBot delivery. [I] Commit-before-expression not evidenced.
- **C7 (4, M)** — [D/V] 17 tools, Function Calling, MCP, Skills, web search,
  Python/Shell sandbox, sub-agents, scheduled self-trigger; blacklist/user/
  admin/root permission levels and type-validated arguments. [I] LLM-selected
  profile updates and sandbox execution need source-level audit.
- **C8 (3, M)** — [D/V] Async queues, model-key rotation and standby fallback,
  token limits, top-40 candidate bounds, decay/cleanup, rate-limit utilities.
  [I] Model fallback is availability, not semantic regeneration.
- **C9 (3, M)** — [D/V] Whitelist pipeline, permission levels, argument
  validation. [I] LLM profile/memory mutation, privacy, and idempotency need
  audit for high-assurance use.
- **C10 (4, M)** — [D/V] Docker, event bus, plugins, tests, model fallback,
  structured decisions that are easier to test than prose. [I] Memory stack
  adds real operational complexity.

### Hermes for QQ Bot

Evidence basis: handover at `main` commit
`1ce1627f2572f440bcc35b74103ccfb7f405a80a` (2026-08-06; tag `v0.14.8`).
QQ-path behavior is source-verified with high confidence; generic-engine
wiring into a deployed QQ install is medium confidence.

- **C1 (4, H)** — [V] Idle/attentive judgment windows (5 s / 1 s), direct
  mention handling, exit grace, judge output with reply/end/exit decisions,
  `[SILENT]`/`[QUIET]` markers, attention expiry. [I] The reason-to-speak
  model lives in one pre-reply judge call.
- **C2 (5, H)** — [V] Buffered messages, watermarks and decision epochs,
  immutable snapshots, Episode State (status, continuity, phase, mode, thread,
  open loops, resolved threads), reconnect recovery with gap notice. This is
  the strongest verified group-awareness machinery in the review.
- **C3 (4, H)** — [V] STM/LTM/EPI/workflow/wiki retrieval with source weights,
  decay, consolidation, graph edges, privacy share levels, rolling summary,
  SQLite session persistence. [I] Episode State restart durability is not
  proven; recall enters the prompt as labeled context.
- **C4 (2, M)** — [V] Persona files, `emotional_tone` in STM, relationships
  category in LTM. [I] No live affect/relationship adaptation engine; memory
  fields rather than committed social state.
- **C5 (4, H)** — [V] Episode controller, attention/exit management, post-
  reply recorder, generic cron scheduler with toolset gating and delivery
  targets. [I] QQ-specific autonomous self-trigger is not proven; cron is a
  generic-engine capability.
- **C6 (3, H)** — [V] Marker-based output control, quote-ID validation,
  delivery retries. [I] A conditional second LLM rewrite in the adapter can
  change wording after the agent finished, which is both a formatting tool and
  a stance-consistency risk.
- **C7 (4, M)** — [V] Tool-calling loop with argument validation, unknown-tool
  recovery, delegation/concurrency limits, per-job toolset gating. [D] The
  80+ tool count and exact per-deployment wiring are documentation claims.
- **C8 (4, H)** — [V] Judge/summary/privacy/recorder timeouts (30/15/10/60 s),
  three group rounds, 90-iteration default, three delivery retries, replay
  recovery. [I] No single end-to-end deadline spans all stages.
- **C9 (3, H)** — [V] Privacy judge (sealed/anonymous/named sharing), DM scope
  clamping, retention/expiry, sanitized session traces. [I] Reasoning display
  and adapter rewrite are configuration boundaries, not invariants.
- **C10 (3, M)** — [V] Dashboard, Live2D, session traces, cron, OneBot
  deployment. [I] README/version drift and configuration complexity are
  maintenance costs.

## Project profiles

### Kazusa

**Strengths.** Strongest documented per-user differentiation (C4): multi-axis
relationship state, deterministic emotion lifecycle, and identity revisions
are typed and tested. Strong control boundaries (C9): typed authorization,
provenance roles, sanitized evidence projection, protected trace, promotion
gates. Grounded silence and evidence-linked reason-to-speak are first-class
(C1). Bounded, inspectable failure behavior (C8) with a clear
accepted/recovered/degraded/unrecoverable ladder.

**Weaknesses.** High per-turn model-call count (six appraisal families, goal
branches, workspace collapse, action planning, authorization, surface
planning, dialog verifiers) makes latency and cost the main open risk (C8,
C10). Capability breadth is narrower than tool-rich projects (C7). Live
behavior is not benchmarked, and the ECT architecture document is a draft
target whose full trajectory is not claimed as complete.

**Best-fit use cases.** Long-running companion products where relationship
continuity, grounded silence, per-user boundaries, and auditability matter
more than tool breadth or lowest per-turn cost.

**Costs and trade-offs.** Highest structural complexity in the review; pays
for itself only when durable relationship/affect state and deterministic
control are product requirements. For generic Q&A or short-lived groups, the
machinery is overkill.

### MaiBot

**Strengths.** Best-balanced engineering package: Planner/replyer split (C6),
recurrent tool/action loop (C5), layered memory with idempotent background
writeback (C3), strong observability and replay (C10), and many practical
runtime bounds (C8). Evidence-aware memory prompts protect durable facts from
bot-only suggestions (C9).

**Weaknesses.** Admission is partly a lexical pre-Planner gate that can miss
grounded reasons (C1). Core memory and learning are optional and
configuration-dependent (C3, C4). Planner analysis is free-form text without a
universal semantic evaluator, so unsupported stance/privacy/authorization can
still influence action (C9). Behavior learning is experimental (C4).

**Best-fit use cases.** Production group bots that need rich memory, tools,
plugins/MCP, and strong operator debugging, with a team that can manage the
configuration surface.

**Costs and trade-offs.** Moderate-to-high deployment complexity; many
optional subsystems must be deliberately enabled and tuned. Rewards teams
that want a capable out-of-the-box architecture and accept prompt-scoped
semantic ownership.

### AstrBot

**Strengths.** Clean bounded agent execution (C8): hard step bounds, repeated-
tool notices, token caps, empty-output retries. Reasoning content is separated
from completion text (C6 hygiene). Function tools and agent-as-tool
composition are straightforward extension patterns (C7, C10).

**Weaknesses.** The reviewed stable core shows no social relevance gating
(C1), no durable memory/consolidation evidence (C3), no persona/affect state
(C4), and no privacy/authorization evidence (C9). MindSim (PR #6888) is
experimental and unmerged.

**Best-fit use cases.** A framework base for building a custom bot with its
own cognition layer; teams that want a bounded agent loop and plugin model
without a social-cognition story included.

**Costs and trade-offs.** Low to moderate cost, but the social-bot work is
left to the integrator. Strong as a substrate, weak as a finished product.

### PetGPT

**Strengths.** Approachable desktop companion (C10): transparent, editable
memory, per-conversation toggles, SQLite persistence. Real social
participation decomposition (observer/intent/reply) with lurk modes and a
five-tier willingness signal (C1, C5). MCP, Skills, and subagents give
practical capability extension (C7).

**Weaknesses.** Several core claims are README/design-plan status, notably the
SOUL/USER/MEMORY file memory replacement (C3). No evidence of typed
provenance, conflict, or authority for injected memory; file memory can become
a live-turn side effect (C9). No commit-before-expression or independent
action authorization evidenced (C6, C9).

**Best-fit use cases.** Individual desktop users who want a pet-like local
companion with inspectable state and light group participation, and who value
simplicity over formal cognition contracts.

**Costs and trade-offs.** Low cost and high transparency; the trade-off is
weaker continuity guarantees and less formal control over what memory does to
behavior.

### ATRI-bot

**Strengths.** Cheap structured decisions (`speak`/`update`/`silence`) that
are testable (C1, C10). Practical hybrid memory with category decay and
consolidation (C3). Broad capability surface with permission levels and
argument validation (C7, C9). Async, queue-based operational foundation with
model fallback (C8).

**Weaknesses.** Decisions are top-level outcomes, not a causal model of why a
stance won (C1). LLM-selected profile/memory updates can mutate durable state
in background paths; ownership, privacy, and idempotency need audit (C9).
Model fallback protects availability, not semantics (C8). Sandbox and
self-trigger capabilities expand agency and need explicit target/permission
boundaries (C5, C9).

**Best-fit use cases.** Self-hosted QQ group bots on PostgreSQL where the team
values structured output, hybrid retrieval, and broad tools and can audit the
durable-state paths.

**Costs and trade-offs.** Moderate operational cost (Docker, PG stack,
consolidation jobs) for a real retrieval/maintenance payoff; weaker
evidence-vs-stance separation is the main structural gap.

### Hermes for QQ Bot

**Strengths.** Best-in-review group awareness (C2): episode state, attention
windows, epochs, watermarks, exit grace. Strong local bounds and fallbacks
(C8). Deep memory stack with privacy share levels and consolidation (C3).
Generic cron scheduler with toolset gating (C5, C7). Silence is a real
outcome with quiet/exit mechanics (C1).

**Weaknesses.** One pre-reply judge plus a general agent is the whole decision
path (C1 depth). No live affect/relationship adaptation (C4). Episode State
restart durability is unproven. The conditional adapter rewrite can alter
visible wording after cognition completes (C6 risk), and stored provider
reasoning can be displayable under configuration (C9 boundary).

**Best-fit use cases.** QQ group bots needing strong presence/episode
management, memory, automation, and tooling, where the operator accepts a
judge-plus-agent decision model and config-based privacy boundaries.

**Costs and trade-offs.** Moderate complexity with excellent operational
controls; the trade-off is that semantic ownership of wording and stance can
cross into the adapter, and QQ-specific autonomous contact is not proven.

## Cross-cutting trade-offs

### Where simpler prompt/tool designs beat ECT-style structure

- **Hermes (C2 = 5)** achieves the best group-awareness outcome in this
  review with buffering, watermarks, an episode record, and a single
  pre-reply judge — no parallel appraisal families or workspace collapse
  needed. For group-presence problems (when to attend, when to exit, how to
  track a thread), lightweight episode mechanics outperform heavyweight
  cognition structure.
- **MaiBot's Planner/replyer loop** delivers tool-driven reasoning and
  separated wording with one recurrent agent; the extra ECT stages would add
  latency without improving ordinary tool/answer turns.
- **ATRI-bot's JSON `speak`/`update`/`silence`** shows a compact decision
  carrier is testable and cheap; ECT's typed bids are only worth their cost
  when the character must explain *why* one stance beat another.
- **PetGPT's five-tier willingness** is a product-friendly control that
  directly manages reply volume; a full appraisal/bid stack would be
  disproportionate for a desktop companion.
- **AstrBot's bounded tool loop** is a clean primitive; anything cognition-
  specific is better added on top than built into the loop.

In all these cases, the simpler design wins on latency, failure surface, and
operating cost for the outcome it targets.

### Where ECT's extra structure is useful

- **C4 outcomes (per-user differentiation):** MaiBot's profile text, ATRI's
  JSON profile, PetGPT's mood state, and Hermes's `emotional_tone` all store
  adaptation material without a rule for how it becomes a stance. Kazusa's
  typed multi-axis state is the only reviewed design where relationship
  meaning has an explicit reducer, evidence policy, and projection. When the
  product needs consistent, long-run relationship behavior, structure earns
  its cost.
- **C9 outcomes (control boundaries):** MaiBot's handover itself notes that
  free-form Planner text can influence action without a semantic evaluator;
  Hermes's adapter rewrite shows wording ownership crossing boundaries; ATRI's
  LLM profile updates mutate durable state. Typed authorization, provenance
  roles, and fail-closed validation directly address these failure modes.
- **Grounded silence:** several designs gate replies by score or tier
  (MaiBot, PetGPT) or a single judge (Hermes). ECT's documented silence path
  ties non-response to an evidence-linked reason, which matters when silence
  must be explainable or audited.

### Cost asymmetry

ECT structure is an insurance policy, not a performance multiplier. It helps
most on long-horizon, high-stakes, or privacy-sensitive interactions and costs
most on high-volume, low-stakes chat. The equal-weight totals therefore mask
use-case-specific ordering: for a short-lived event bot, AstrBot/PetGPT-class
simplicity may dominate; for a years-long companion, C4/C9 outcomes dominate.

## Use-case recommendations

| Use case | Recommended | Why |
| --- | --- | --- |
| Long-running companion with relationship/affect continuity and auditability | Kazusa | Strongest C4/C9 profile; accept the latency and complexity cost. |
| Production group bot with rich memory, tools, and debugging (team can tune config) | MaiBot | Best balanced engineering package; highest-confidence evidence. |
| QQ group bot needing presence/episode management, memory, and automation | Hermes | Best group awareness and strong local bounds; verify QQ self-trigger and adapter rewrite in deployment. |
| Self-hosted QQ bot with structured decisions and hybrid retrieval on PostgreSQL | ATRI-bot | Cheap structured output plus practical memory; audit durable-state writes. |
| Desktop companion with transparent memory and light social participation | PetGPT | Low cost, user-editable state; treat memory docs as claims until verified. |
| Framework base for building a custom agent/bot | AstrBot | Bounded, extensible execution core; build your own social layer. |

## Limitations

1. **Repository evidence is not a live behavioral benchmark.** No project was
   run, load-tested, or conversation-tested for this review. Scores reflect
   designed and documented scope in the reviewed snapshots.
2. **Equal weighting is a convention.** The totals are a secondary summary;
   real deployments should reweight C1-C10.
3. **Evidence classes differ by project.** External projects mix verified
   source facts, official documentation claims, and optional/experimental
   features. Kazusa's evidence is local repository ICDs and a draft target
   architecture; implemented scope is anchored in `cognition_core_v2`, the
   resolver, and RAG3, not in the full target document.
4. **Optional features are not guaranteed behavior.** MaiBot's A_Memorix,
   PetGPT's memory-file redesign, AstrBot's MindSim PR, ATRI's documented
   defaults, and Hermes's adapter rewrite are configuration- or deployment-
   dependent.
5. **Absence of evidence is not proof of absence.** Low scores on thin
   evidence (e.g., AstrBot C3/C4/C9, PetGPT C3) mean the reviewed slice did
   not demonstrate the capability; they do not prove it cannot exist.
6. **Private chain-of-thought is excluded.** Only observable semantic and
   control artifacts were compared; provider-native reasoning fields were
   treated as representation hygiene, not cognition evidence.
7. **Snapshots age.** All external commits were pinned on 2026-08-12 or
   nearby; later upstream changes are outside this review.

## Source register

| Project | Repository | Reviewed ref (pinned commit) | Evidence classes used | Handover |
| --- | --- | --- | --- | --- |
| Kazusa | Local workspace (`C:\workspace\kazusa_ai_chatbot`) | Local repository state at review time; no external pinned commit | Repository ICDs, contracts, tests, draft architecture | `bot_cognition_handover/` n/a (local docs) |
| MaiBot | <https://github.com/Mai-with-u/MaiBot> | `main` `1b8d13300a5add5f029106b0ee2d80b59996917c` (`1b8d133`); `dev` observed at `240725cbb8e4573a4f08534b844cd2b85eff13ba`; tag `1.0.0-rc.2` at `864ea04e74b91f0d188f219d8e29af367e781aa6` (not the reviewed snapshot) | Verified source, documentation claims, optional features, inference | `bot_cognition_handover/maibot.md` |
| AstrBot | <https://github.com/AstrBotDevs/AstrBot> | `master` `a9bb8a64ca69657e6262e3ca06541ecaf3a6d1ca`; MindSim PR #6888 (unmerged) | Verified source, documentation claims, PR evidence, inference | `bot_cognition_handover/astrbot.md` |
| PetGPT | <https://github.com/JulesLiu390/PetGPT> | `main` `8b63d5144da41b88c20f59a7427b3d0e999c072f` | Verified source, README/design claims, inference | `bot_cognition_handover/petgpt.md` |
| ATRI-bot | <https://github.com/114514ggb/ATRI-bot> | `main` `3dcae6d33279ced3ad238ba86460e64e48228582` | README claims, module ownership, inference | `bot_cognition_handover/atri_bot.md` |
| Hermes for QQ Bot | <https://github.com/jixiong398-blip/hermes-for--qqbot> | `main` `1ce1627f2572f440bcc35b74103ccfb7f405a80a` (tag `v0.14.8`) | Verified source, documentation claims, inference | `bot_cognition_handover/hermes_for_qqbot.md` |

Primary evaluation standard: `social_bot_design_criteria.md` (this directory).
Primary source inputs: `docs/architecture/explicit_cognitive_trajectory_architecture.md`,
`src/kazusa_ai_chatbot/cognition_core_v2/README.md`,
`src/kazusa_ai_chatbot/cognition_resolver/README.md`,
`src/kazusa_ai_chatbot/local_context_resolver/README.md`,
`src/kazusa_ai_chatbot/rag/README.md`.
