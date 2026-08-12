# Hermes for QQ Bot — cognition research handover

## Research snapshot

- Repository: <https://github.com/jixiong398-blip/hermes-for--qqbot>
- Inspected ref: `main` at commit `1ce1627f2572f440bcc35b74103ccfb7f405a80a`
- Commit date: `2026-08-06 00:31:43 +08:00`
- Commit subject: `fix: redact v0.14.8 changelog details (keep public changelog neutral; details in local MAINTENANCE.md)`
- The repository also exposes tag `v0.14.8`. The root README still describes itself as v0.14.3, so README version text is treated as a documentation claim rather than exact release metadata.
- Source policy: official repository README, tree, source files, and commit metadata only. No third-party summaries were used.

## Source map

| Area | Repository source | Evidence used |
|---|---|---|
| Product and deployment claims | [`README.md`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/README.md) | QQ/OneBot deployment, `SOUL.md`, two-tier judgment windows, Episode State, memory layers, tool count, privacy boundary |
| QQ event state | [`group_state.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/plugins/platforms/onebot/group_state.py) | buffered messages, attention, watermarks, epochs, Episode State, in-memory group registry |
| Trigger and judgment scheduling | [`trigger_coordinator.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/plugins/platforms/onebot/trigger_coordinator.py), [`semantic_judge.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/plugins/platforms/onebot/semantic_judge.py) | idle/attentive windows, mention handling, exit grace, judge schema, fallbacks, timeouts, post-reply recorder |
| QQ agent execution | [`group_executor.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/plugins/platforms/onebot/group_executor.py) | bounded per-group rounds, prompt assembly, markers, post-reply state recording |
| QQ delivery and persistence | [`adapter.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/plugins/platforms/onebot/adapter.py) | `[SILENT]`/`[QUIET]`, quote validation, retries, optional rewrite, chat persistence and recovery |
| Generic agent loop | [`run_agent.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/run_agent.py) | tool-calling loop, iteration budget, retries, compression, reasoning fields, post-turn work |
| Memory and retrieval | [`retrieval.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/agent/memory/retrieval.py), [`episodic_index.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/agent/memory/episodic_index.py), [`consolidation.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/agent/memory/consolidation.py), [`gateway.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/agent/memory/gateway.py) | STM/LTM/EPI/workflow/wiki retrieval, decay, privacy filtering, consolidation, graph edges |
| Sessions and trajectory | [`hermes_state.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/hermes_state.py), [`trajectory.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/agent/trajectory.py) | SQLite transcript/reasoning/tool trace and JSONL trajectory export |
| Autonomous scheduling | [`scheduler.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/cron/scheduler.py), [`jobs.py`](https://github.com/jixiong398-blip/hermes-for--qqbot/blob/1ce1627f2572f440bcc35b74103ccfb7f405a80a/hermes/core/cron/jobs.py) | cron/interval jobs, toolset gating, delivery targets, missed-run handling |

## Verified source facts

### End-to-end QQ cognition and response flow

1. OneBot input is normalized into a `BufferedMessage` with sequence, message ID, timestamp, sender, text, media, and mention metadata. The group buffer keeps bounded recent history and watermarks (`last_user_seq`, `last_judged_seq`, and a `decision_epoch`).
2. `TriggerCoordinator` classifies the event as a mention, attentive input, idle-window judgment, continuation, or exit path. Direct mention enters attentive mode; ordinary input waits for the idle window. A newer message invalidates stale judgment work through the epoch and remains queued for the next judgment.
3. The pre-reply judge is an LLM JSON call. It sees a recent message window, current/follow-up context, mention targets, reply target, attention mode, and prior structured Episode State. Its output includes reply/end/exit decisions, continuity, conversation mode, phase, speaker role, thread, loop status, reply-feature choice, indirect-speech context, progression guidance, and a reason.
4. The coordinator applies the judge result to group control. It can end and archive an episode, start a quiet/exit countdown, or schedule a bounded agent execution. The attentive exit grace is 15 seconds; the normal idle and attentive windows are 5 seconds and 1 second.
5. `GroupExecutor` takes an immutable message snapshot, builds a channel prompt containing recent messages, rolling summary, recalled context, core memories, and Episode State, then calls the generic Hermes agent. A group runner is capped at three rounds and coalesces pending work per group.
6. The generic `AIAgent` runs a tool-calling loop. It builds a system prompt from persona/context files and memory, calls the configured model, executes validated tool calls, appends results, compresses context when required, and repeats until a final answer, a retry/failure condition, or the iteration budget is reached. The default agent iteration budget is 90.
7. QQ delivery interprets output markers. `[SILENT]` records a silent decision; `[QUIET]` withdraws attention; `[reply:<message-id>]` is checked against the local message corpus before quoting. The adapter strips or normalizes some formatting, sends with retry, and records successful bot output.
8. After a reply, a second LLM recorder updates Episode State with status, continuity, phase, mode, thread, bot moves, overused moves, open loops, resolved threads, progression guidance, and label. Rolling summary and memory/event persistence happen around the turn and later consolidation.

This is an event-triggered episode controller wrapped around a general tool-using agent. The repository does not show a single typed cognitive trajectory spanning observation, appraisal, motive competition, intention selection, state commit, and expression.

### Episode state, attention, and social continuity

`EpisodeState` is a structured conversation-state record. The source includes status, continuity, turn count, label, current thread, conversation mode, episode phase, last speaker role, bot moves, overused moves, open loops, resolved threads, progression guidance, and timestamps. `GroupState` additionally tracks attention, silent count, last reply, episode counters, summaries, watermarks, and an epoch.

The attention state expires after ten minutes or three silent turns. `go_quiet()` preserves Episode State while withdrawing attention; `end_episode()` archives and resets the episode state. This gives Hermes a clear group-level notion of conversational presence and exit.

The source does not show a first-class affect vector, relationship appraisal, or multi-axis affinity state in the QQ Episode State. Short-term memory can store an `emotional_tone`, and long-term memory has a `relationships` category, but these are memory fields rather than an explicit live social-state commit.

### Retrieval and memory

The unified retriever ranks short-term, episodic, long-term, workflow, and wiki sources with configurable source weights and a context-character budget. Retrieval results carry a source, relevance, content, and metadata, then become labeled prompt sections.

The EPI index preserves short raw fragments across sessions. Its source comments and implementation specify 3–8-turn fragments, 7-day retention, token/IDF matching rather than embeddings, a default result limit of two, a minimum score, privacy share levels, current-session exclusion, resurfacing cooldown, and a watermark to avoid duplicate indexing. A privacy judge classifies sealed, anonymous, or named sharing; DM scope is clamped more narrowly.

The memory gateway combines STM, LTM, EPI, workflow memory, and wiki data. Consolidation promotes repeated STM facts to LTM, detects repeated action sequences as workflows, applies decay, and records consolidation statistics. LTM categories include user profile, preferences, agent identity, knowledge, decisions, relationships, coding, and general. Memory retrieval can reconsolidate recalled facts and graph-expand related memory edges.

The memory architecture is substantially richer than a single rolling chat summary, but the retrieved material is injected as context for the general agent. The source does not expose ECT-style evidence handles or a semantic rule that prevents recalled memory from becoming the agent's stance without further appraisal.

### Reasoning representation and observability

Hermes preserves provider reasoning fields such as `reasoning`, `reasoning_content`, and `reasoning_details` in session records and API message handling. Trajectory export stores ShareGPT-style conversations in JSONL; scratchpad tags can be converted to `<think>` tags. Gateway configuration can expose reasoning progress.

This is provider/private reasoning capture and training-trajectory logging, not an explicit semantic trajectory schema. It does not replace the typed decision record required by Kazusa ECT. If reasoning display is enabled, the repository provides a path for analytic material to become user-visible; privacy of cognition is therefore a configuration/property boundary rather than an invariant of the architecture.

### Tools, actions, permissions, and visible dialog

The generic agent executes model-selected tools with argument validation, unknown-tool recovery, retry limits, delegation limits, concurrent worker limits, and context compression. Cron jobs resolve a restricted platform set and per-job/per-platform enabled toolsets.

QQ-specific output control is marker-based and model-mediated. The adapter performs deterministic checks for quote targets and delivery formatting, but it also contains a conditional second LLM rewrite for long markdown/report-like replies. That rewrite uses the persona and model configuration before delivery. Consequently, the generic agent's final text is not always the final visible text, and the adapter can change wording after the agent has completed its response.

The inspected source provides delivery retries and several fail-closed or fallback paths, but it does not establish Kazusa's stronger separation in which capabilities cannot self-authorize, dialog cannot re-decide stance/permission, and delivery owns rendering only.

### Scheduling and self-trigger behavior

The generic Hermes engine has a persistent cron/interval scheduler. Jobs can run an unattended agent, gate enabled toolsets, resolve a delivery platform/home target, save output, handle delivery failure separately, and recover or advance missed recurring runs.

For the inspected QQ path, the demonstrated self-trigger mechanism is message-driven attention/judgment plus the 15-second exit grace. I found no source evidence in this slice that cron jobs autonomously re-enter a QQ group's `EpisodeState` or perform character-contact decisions using the QQ episode controller. Treat “scheduled automation” as a generic Hermes capability, not as proven QQ-character self-trigger behavior.

### Persistence, latency, and failure bounds

- Session SQLite stores sessions, messages, tool calls, finish metadata, and reasoning fields; it supports transcript replacement, ancestry, search, and pruning.
- QQ group state includes bounded in-memory buffers and persistent chat/corpus writes, with reconnect recovery that can replay history and insert a gap notice when needed.
- Episode State itself is held by the in-memory `GroupStateRegistry`; this snapshot does not prove restart persistence for that state.
- Judge timeout is 30 seconds; rolling-summary timeout is 15 seconds; privacy judgment timeout is 10 seconds; post-reply recording timeout is 60 seconds. Judge failure falls back to mention-driven reply behavior, otherwise silence. Summary and recorder failures preserve prior state/summary where implemented.
- Group execution is capped at three rounds; the general agent defaults to 90 iterations and has bounded tool/retry paths. Delivery retries are capped at three attempts.
- These are useful local bounds, but there is no single end-to-end response deadline spanning judge, agent tools, post-reply recorder, optional adapter rewrite, persistence, and delivery.

## Repository documentation claims

The root README advertises a two-tier semantic judgment window (approximately five seconds while idle and one second while attentive), likely reply behavior for direct `@` signals while preserving the ability to refuse, a 16-field Episode State, STM/LTM/EPI/workflow memory, 80+ tools, a dashboard, Live2D, and a privacy boundary around configuration, soul, sessions, logs, and databases. The source confirms the main trigger windows, structured Episode State concept, memory components, and tool-oriented engine, but the README's exact field/tool counts and product-version text are not treated as stronger evidence than the inspected implementation.

The generic core README claims a self-improving learning loop, persistent memory, scheduled automations, delegation/parallelization, and cross-session conversation continuity. Those claims are supported at the generic engine level by the memory, cron, delegation, and session sources above. They should not be read as proof that every claim is wired into the QQ group-specific cognition path.

## Inferences for Kazusa

### Strengths worth reusing

1. Hermes makes group presence explicit. Attentive/quiet state, silent counts, idle versus attentive windows, episode phases, exit grace, and stale-result epochs are practical controls for noisy group chat.
2. The pre-reply judge has a compact structured decision surface. Its continuity, thread, phase, loop, and progression fields are useful inputs to a richer semantic appraisal stage.
3. The immutable snapshot and message watermarks are good concurrency primitives for bounded live cognition.
4. The memory stack separates recent context, raw episodic fragments, distilled facts, procedural workflows, and wiki evidence. Its privacy levels, expiry, cooldown, and graph links are useful continuity mechanisms.
5. The implementation exposes operational limits and fallbacks instead of relying on an unbounded agent loop.

### Limitations relative to Kazusa ECT

1. The reply decision is primarily one pre-reply LLM judge followed by a general agent. The source does not show parallel appraisal families, explicit motive/goal bids, a workspace collapse, or a selected intention record.
2. Episode State describes conversation progression, not a committed interpretation of affect, relationship, boundaries, reasons to speak, or character stance.
3. The recorder updates episode state after visible response. That is useful continuity bookkeeping but does not satisfy ECT's commit-before-expression invariant.
4. Retrieved memory, rolling summaries, persona text, and tool results enter the general prompt without ECT-style evidence-handle provenance and semantic ownership boundaries.
5. Marker parsing and a conditional post-generation LLM rewrite allow visible-dialog ownership to cross into the adapter. This risks changing stance or meaning after cognition has completed.
6. Generic cron is a strong automation feature, but QQ-specific autonomous contact, permission, and relationship-aware self-triggering are not proven by the inspected files.
7. Stored provider reasoning and `<think>` trajectory export capture hidden reasoning, but they are not a stable, inspectable semantic decision trace and may be displayable under configuration.

### Reusable design boundary

Kazusa can borrow Hermes's episode controller, watermarks, debounce/attention windows, exit grace, structured progression fields, memory retention/privacy policies, and bounded operational fallbacks. These should enter Kazusa as evidence/context and semantic inputs. Kazusa should retain ECT ownership of appraisal, affect/relationship interpretation, motive arbitration, intention/stance selection, deterministic commit, permission, and final dialog surface ownership. Hermes's raw provider reasoning and post-generation rewrite should not become the ECT contract.

## Evidence matrix against Kazusa ECT

Legend: **verified** = directly shown in source; **partial** = a related mechanism exists but not the ECT contract; **absent** = not shown in the inspected source; **risk** = evidence points across an ECT invariant.

| ECT stage or invariant | Hermes evidence | Assessment |
|---|---|---|
| Typed observation/admission | `BufferedMessage`, mention metadata, watermarks, epochs, immutable snapshots | **partial** — strong QQ event normalization, but no Kazusa observation-envelope contract |
| Episode/context projection | rolling summary, recent-window prompt, Episode State, recalled memory | **partial** — projection is prompt assembly rather than a typed semantic context projection |
| Parallel semantic appraisal | one structured pre-reply judge with reply/end/exit/continuity fields | **partial** — semantic judgment exists, but no parallel appraisal families or independent evidence roles |
| Bounded state interpretation | attention state, episode status/phase/mode/thread, silent count | **partial** — conversational state is modeled; live affect/relationship interpretation is not |
| Motive/goal bids | generic tool-calling agent and progression guidance | **absent** — no explicit competing motive/goal bids or typed bid reduction shown |
| Workspace collapse | prior state plus judge result plus agent prompt | **absent** — no named workspace/collapse stage or deterministic arbitration record |
| Resolver recurrence | tools can return results to the generic agent loop | **partial** — tool iteration exists; no typed resolver capability/observation/cognition recurrence |
| Intention/stance selection | `should_reply`, `should_exit`, `progression_guidance`, then agent final answer | **partial** — reply gating exists; selected character intention/stance is not a durable typed output |
| Commit before expression | recorder replaces Episode State after a reply; adapter may rewrite text after agent output | **risk** — observed ordering is downstream bookkeeping, not ECT commit-before-dialog |
| Deterministic permission/capability boundary | toolset gating, delivery target validation, quote-ID validation, retries | **partial** — operational checks exist; capability use and semantic permission are not one ECT authorization boundary |
| RAG/evidence distinct from stance | multi-source memory returns labeled prompt context and metadata | **partial** — source labels exist; no evidence-handle contract prevents the agent from treating recall as stance |
| Dialog owns wording/rendering | agent generates wording; adapter strips/normalizes and may perform second LLM rewrite | **risk** — adapter can semantically alter visible wording |
| Silence as a first-class outcome | judge `should_reply=false`, `[SILENT]`, `[QUIET]`, attention expiry | **verified/partial** — strong silence mechanics; the character's reason-to-speak model remains less explicit |
| Post-turn continuity | recorder, rolling summary, STM/LTM/EPI/WFM consolidation, session persistence | **verified** — extensive continuity machinery; Episode State restart durability remains unproven |
| Protected trace/observability | SQLite reasoning/tool/session records and JSONL trajectories | **partial** — rich operational trace; not an ECT protected semantic trace with decision provenance |
| Growth/promotion boundary | STM→LTM/WFM consolidation, EPI indexing, skill generation claims/source | **partial** — promotion exists; no Kazusa-style promotion gate for stable character/relationship state |
| Bounded live path | judge/summary/privacy/recorder timeouts, three group rounds, 90 agent iterations, retry caps | **verified/partial** — many local bounds; no unified end-to-end deadline |
| Autonomous contact permission | generic cron scheduling and delivery targets | **partial/absent for QQ** — scheduler exists, but QQ-specific self-trigger permission and episode integration are not evidenced |

## Open questions and confidence notes

- **High confidence:** source-level behavior for QQ buffering, attention windows, Episode State fields, judge output, group-round cap, output markers, memory-source structure, SQLite session traces, and local timeouts.
- **Medium confidence:** exact production wiring of all generic memory providers and cron delivery options into a deployed QQ installation; the inspected source contains the subsystems, while deployment configuration determines activation.
- **Low confidence / not proven:** persistence of `EpisodeState` across process restart, whether all marker outcomes are reachable through every `handle_message` path, whether the adapter's long-response rewrite is enabled in a given deployment, and whether any cron job is configured to contact a QQ group autonomously.
- The root README and generic core README are useful design claims, but implementation evidence takes precedence where wording, release version, field count, or feature scope differ.

