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
budgets next to their handlers. Settled input keeps active fragments separate
from external fresh history and includes bounded mood, relationship,
group-attention, bot-continuity, and media-overflow descriptors. A decision
that depends on undescribed retained media fails closed instead of inferring
from a partial image subset.

## Public Contract

Runtime callers import the public functions and validation types from
`kazusa_ai_chatbot.relevance`:

- `frontline_relevance_agent(state)` consumes the bounded current-message,
  open-turn, and prelude projection and returns a validated
  `FrontlineDecision`.
- `relevance_agent(state)` consumes the bounded settled-turn projection and
  returns a validated settled decision plus the downstream
  `should_respond` compatibility field.
- `validate_frontline_decision(...)` and
  `validate_settled_relevance_decision(...)` enforce closed semantic actions,
  bounded free-form fields, start-only prelude promotion, and the complete-phase
  no-`wait` rule.

The public facade exposes contract helpers and decision types. Prompt
constants, LLM instances, route configs, projections, trace details, and other
implementation state remain private to the two agent modules.

## Ownership Boundaries

The relevance agents own semantic interpretation of typed evidence. They do
not own persistence, platform parsing, queue ordering, clocks, deadlines,
slot-to-ID mapping, media cache mutation, response delivery, or cognition
claims. They receive semantic labels and bounded descriptors rather than raw
platform identifiers, operational timestamps, futures, handles, or queue
telemetry.

`brain_service.turn_settlement` owns the deterministic pending-turn lifecycle,
FIFO relevance work, prompt-safe candidate/prelude slots, enqueue-time deadline
handling, the pre-deadline ingress barrier, stale-version checks, and the atomic
cognition claim. `kazusa_ai_chatbot.service` owns persistence and single-owner
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
