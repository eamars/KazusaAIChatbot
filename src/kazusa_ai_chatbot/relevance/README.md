# Relevance Interface Control Document

## Document Control

- ICD id: `REL-ICD-001`
- Owning package: `kazusa_ai_chatbot.relevance`
- Interface boundary: typed message and turn projections -> semantic relevance
  decisions
- Runtime consumers: brain-service intake, turn settlement, the persona graph
  handoff, and relevance contract tests
- Downstream owner: `kazusa_ai_chatbot.brain_service.turn_settlement` owns
  deterministic lifecycle, ordering, deadlines, and cognition claims
- Public entrypoint: `kazusa_ai_chatbot.relevance`
- Primary implementation files:
  - `participation_evidence.py`
  - `frontline_relevance_agent.py`
  - `persona_relevance_agent.py`

## Purpose

The relevance package owns the two semantic decisions that determine whether a
typed inbound message enters a candidate turn and whether a stabilized turn
gives the active character a grounded reason to speak.

The frontline agent answers the compact intake question with `discard`,
`start`, or `append`. The settled agent answers the character-level question
with `ignore`, `proceed`, or `wait`. Both agents use the existing
`RELEVANCE_AGENT_LLM` route and keep their stage-local prompt and completion
budgets next to their handlers. Both receive canonical `group|private` scope
and the active character name. Group participation requires character-grounded
interaction evidence or a concrete intersection with supplied active character
state. Interaction relevance includes typed character target/reply, a typed or
explicit whole-group invitation, complete canonical-name address, or grounded
open-turn/bot/history continuity. Character-state salience requires the current
message to materially intersect a supplied active-state item such that speaking
could advance, protect, resolve, or investigate it. Answerability, helpfulness,
emotionality, topic popularity, and stable personality traits alone are
insufficient.
Frontline uses separate static group and private prompt branches so the local
model never carries both scope policies in one call. Its output contract also
removes `append` when no open slot exists and requires an empty prelude list
when no prelude slot exists. The agent validates every returned slot against
the exact capped payload shown to the model and fails closed on invented slots.
An open turn containing only a direct character summon or an explicitly
unfinished setup remains incomplete; the next same-author message that
naturally supplies its missing request or content appends even when its own
typed target and reply labels are `none`.
Both stages receive bounded interaction evidence and a bounded projection of
the same process-local native character cognition-state snapshot. Deterministic
code exposes typed protocol facts, a contiguous complete-name candidate, active
state candidates, and evidence provenance; the relevance LLM owns recipient,
continuity, explicit-group-invitation, and message/state-intersection judgment.
Pronouns, numerals, canonical-name prefixes, and the generic current-message
item do not establish character address.

Settled input keeps active fragments separate from external fresh history and
includes bounded participant/target/reply relations, active state,
relationship, group-attention, bot-continuity, and media-overflow descriptors.
A decision
that depends on undescribed retained media fails closed instead of inferring
from a partial image subset.

The assembled-turn projection states that its author is the current human and
repeats the bounded final fragment as `effective_latest_fragment` so a weaker
local model applies recipient corrections before older context. Fresh-history
rows also carry
`before_active_turn|during_active_turn|after_active_turn|unknown`; only an
after-turn row, or a during-turn row that follows the relevant earlier
fragment, can establish that another participant already answered the active
request.

## Public Contract

Runtime callers import the public functions and validation types from
`kazusa_ai_chatbot.relevance`:

- `frontline_relevance_agent(state)` consumes the bounded current-message,
  scope, active-character, open-turn, eligible-prelude, and recent-continuity
  projection and returns a validated `FrontlineDecision`.
- `relevance_agent(state)` consumes the bounded settled-turn projection and
  returns a validated settled decision plus the downstream
  `should_respond` compatibility field.
- `validate_frontline_decision(...)` and
  `validate_settled_relevance_decision(...)` enforce closed semantic actions,
  bounded free-form fields, start-only prelude promotion, and the complete-phase
  no-`wait` rule.

The frontline handler additionally validates append and prelude references
against the exact candidate slots present after prompt-cap fitting. This is
structural validation; it never reclassifies a valid semantic action.
Both handlers also process model-only `recipient_relation`,
`admission_basis`, `interaction_evidence_refs`, and `character_state_refs`
against the exact cap-fitted payload. Model reference lists discard invalid,
invented, unavailable, and duplicate entries, then retain at most the first
three usable references. A normal semantic result that loses required
grounding still fails closed. An already-authoritative settled turn instead
uses deterministic typed participation grounding, so malformed diagnostic
evidence cannot terminate the pipeline. The resulting assessment is retained
only in protected trace diagnostics and is stripped before returning the
unchanged public decisions.

The public facade exposes contract helpers and decision types. Prompt
constants, LLM instances, route configs, projections, trace details, and other
implementation state remain private to the two agent modules.

An unresolved present reply uses `unknown_participant`, distinct from `none`.
Fresh-history rows reserve `character` and `current_author`, assign stable
`participant_1` through `participant_8` handles oldest-first to other resolved
humans, and reuse those handles across speaker, target, reply, and evidence
summaries. Raw platform/global identities never enter the model payload.
The first settled assessment renders `ignore|proceed|wait`; the hard-deadline
assessment renders only `ignore|proceed`.
For authoritative participation, deterministic code derives the available
semantic dispositions from typed target, reply, history, media, and observation
facts. When only one disposition remains, it is projected directly with
conservative optional metadata because no semantic choice exists. When several
dispositions remain, the LLM chooses among those values. One structurally
invalid hard-deadline result may receive one bounded same-owner repair; a second
invalid result raises a typed operational failure. During the first assessment,
an invalid authoritative result reaches the coordinator's existing one-time
`wait` boundary and is reassessed at the hard deadline instead of becoming a
synthetic semantic `ignore`.
The settled agent also treats `use_reply_feature` as a semantic request for a
native visual anchor. It requests an anchor only for a proceeding group turn
where anchoring the effective latest fragment materially clarifies a specific
character-directed message or speaker. Private turns, whole-group invitations,
state-salience admissions whose actual recipient is not the character, and
non-proceed actions do not request one. The actual recipient remains independent
from the reason to speak, so state salience may admit a turn while retaining a
participant or unknown recipient. Delivery feasibility and the
executable quote target remain outside this package.

## Ownership Boundaries

The relevance agents own semantic interpretation of typed evidence. They do
not own persistence, platform parsing, queue ordering, clocks, deadlines,
slot-to-ID mapping, media cache mutation, response delivery, or cognition
claims. They receive semantic labels and bounded descriptors rather than raw
platform identifiers, operational timestamps, futures, handles, or queue
telemetry.

`kazusa_ai_chatbot.service` copies the already loaded native
`cognition_state.v2` snapshot into both relevance inputs. Relevance performs no
database, retrieval, embedding, or cache call to obtain state.

`brain_service.turn_settlement` owns the deterministic pending-turn lifecycle,
FIFO relevance work, prompt-safe candidate/prelude slots, enqueue-time deadline
handling, the pre-deadline ingress barrier, stale-version checks, and the atomic
cognition claim. It also closes a current failed assessment without inventing
an `ignore` decision, while a stale failed lease leaves the newer turn version
scheduled. Explicit-third-party and unresolved-reply discards are not
eligible preludes, and bot continuity is available only in the bounded active
scene. `kazusa_ai_chatbot.service` owns persistence and single-owner
request/future resolution. The `nodes` package owns persona orchestration,
cognition, dialog, and perception nodes; it does not own the relevance agents.

## Module Rules

- Existing runtime callers use the package facade rather than private module
  state.
- Tests that inspect a model instance or prompt may import the concrete agent
  module directly because that is the module's test boundary.
- No compatibility module remains under `kazusa_ai_chatbot.nodes`.
- The two semantic agents share the existing route configuration; moving the
  package does not add deployment variables or an extra model call.
