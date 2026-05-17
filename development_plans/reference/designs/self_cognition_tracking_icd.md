# self cognition tracking ICD

## Document Control

- ICD id: SC-TRACKING-ICD-001
- Status: superseded reference pointer
- Canonical location:
  `src/kazusa_ai_chatbot/self_cognition/README.md`
- Related plan:
  `development_plans/archive/completed/short_term/self_cognition_agency_loop_plan.md`

## Canonical ICD

The production-owned self-cognition module now owns the canonical ICD for
`kazusa_ai_chatbot.self_cognition`.

Use `src/kazusa_ai_chatbot/self_cognition/README.md` for the current public
interfaces, local artifact shapes, production boundary, configuration
surface, dry-run command, supported case schema, and
`SC-TRACKING-ICD-001` tracking contract.

The current boundary is a private tracking contract: the module writes local
tracking artifacts and production action-attempt rows, but it does not hand
prewritten text to the scheduler or dispatcher. If a delayed follow-up should
be reconsidered, the active path is a scheduled future cognition slot. It does
not call adapters directly or write live-chat, reflection, stable-memory, or
conversation-progress state.

This reference document remains only as a development-plan registry pointer so
older research and plan material can resolve to the implemented module
contract.
