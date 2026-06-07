# action_router — Route-Only Action Initialization

## Ownership

This module owns the reusable action-router prompt, JSON payload contract,
output normalization, and prompt-safe capability projections for L2d action
initialization. It does not own graph orchestration, state threading, action-spec
materialization, queue persistence, or adapter delivery.

## Public Contract

```python
from kazusa_ai_chatbot.action_router.payload import build_action_router_payload
from kazusa_ai_chatbot.action_router.prompt import ACTION_ROUTER_PROMPT
from kazusa_ai_chatbot.action_router.router import route_action_requests
from kazusa_ai_chatbot.action_router.contracts import normalize_action_router_output
```

- `build_action_router_payload(state, capabilities)` — returns a
  JSON-serializable dict with prompt-safe semantic sections.
- `ACTION_ROUTER_PROMPT` — stable route-selection procedure. It reads concrete
  resolver and action names from the payload affordances and does not duplicate
  the capability roster in prompt prose.
- `route_action_requests(llm, state, capabilities)` — builds messages, calls
  the supplied LLM, parses JSON, and returns normalized route decisions.
- `normalize_action_router_output(raw)` — normalizes raw model output into the
  route-only contract, stripping forbidden fields and attaching trusted
  resolver schema metadata after the LLM returns.

## Boundary Rules

- The action router LLM may choose route families and immediate visible-surface
  need.
- It must not choose workers, worker-local task types, handler ids, DB fields,
  adapter targets, job ids, delivery mechanics, tool arguments, or final visible
  text.
- For `background_work_request`, the router chooses the route and a reason;
  deterministic materialization builds the queue summary from already prompt-safe
  state.
- Resolver capability requests must not contain `schema_version`,
  `pending_row_id`, or `resolver_id` — those are bound deterministically after
  routing.

## Payload Sections

| Section | Content | Forbidden |
|---|---|---|
| `source` | trigger/input source, output mode, channel type | platform ids, adapter ids |
| `current_input` | decontextualized input, media summary | raw message ids |
| `cognition` | stance, intent, judgment, boundary, social signals | - |
| `evidence` | RAG answer, memory evidence, commitments, progress | raw memory ids, collection names |
| `resolver` | pending resume, resolver context | pending row ids |
| `capabilities` | action affordances, resolver affordances | handler ids, schemas |
| `work_seed` | background-work source summary and output cap | worker names, task types |

## Output Shape

```json
{
  "resolver_capability_requests": [],
  "resolver_pending_resolution": null,
  "resolver_goal_progress": null,
  "action_requests": [
    {
      "capability": "speak",
      "decision": "visible_reply",
      "detail": "surface need",
      "reason": "why"
    }
  ]
}
```

## Module Map

- `__init__.py` — package marker.
- `contracts.py` — output normalization and field stripping.
- `payload.py` — JSON payload builder from cognition state.
- `prompt.py` — route-only action-router prompt.
- `router.py` — message construction, LLM invocation, parsing, and
  normalization wrapper.
- `README.md` — this ICD.
