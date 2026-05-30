# Action Spec

`kazusa_ai_chatbot.action_spec` owns the modality-neutral action contract used
between cognition, selected surface handlers, private action handlers,
consolidation, and scheduler-owned continuation handlers.

It is not the dispatcher, not a scheduler, and not an LLM prompt package. Its
job is to define typed action residues, validate them deterministically, project
prompt-safe capabilities, and build prompt-safe result traces after downstream
owners have handled the action.

## Runtime Boundary

```text
L2d semantic action request
  -> deterministic route materialization
  -> optional specialist-owned target judgment
  -> deterministic executable materialization and target binding
  -> ActionSpecV1 validation
  -> owner handler
       l3_text / l3_image surface handler
       memory_lifecycle private handler
       orchestrator future-cognition request
  -> ActionResultV1 and SurfaceOutputV1
  -> EpisodeTraceV1
  -> prompt-safe consolidation projection
```

L2d emits semantic `action_requests`. Deterministic code materializes valid
requests into `ActionSpecV1` rows by attaching schema versions, source refs,
targets, continuation defaults, handler-owned params, and idempotency metadata.
The LLM must not receive, copy, compare, or emit raw persistence ids, adapter
ids, handler ids, collection names, or schema-version fields.

## Public Contracts

Core contracts live in `models.py`:

- `ActionSpecV1`: the single graph-visible action residue.
- `ActionSourceRefV1`: trusted provenance attached by code.
- `ActionTargetV1`: trusted target binding attached by code.
- `ActionContinuationV1`: bounded contract for follow-up cognition; direct
  tool-triggered cognition is not allowed.
- `CapabilitySpecV1`: internal capability registry entry.

Trace contracts live in `results.py`:

- `ActionResultV1`: validation, scheduling, execution, or rejection outcome for
  one action.
- `SurfaceOutputV1`: text, image, private, or future surface artifact.
- `EpisodeTraceV1`: consolidation-facing episode record containing cognition
  refs, actions, action results, and surface outputs.
- `ConsolidationActionProjectionV1`: prompt-safe projection consumed by the
  consolidator.

Current runtime accepts only `cognition_mode="deliberative"`. `reflex` is a
reserved schema slot and fails validation in this implementation slice.

## Capabilities

`build_initial_action_capabilities()` registers the available runtime
capabilities. Prompt projection exposes only L2d-facing semantic capabilities;
internal executable capabilities stay hidden from L2d prompts.

| Capability | Owner | Visibility | Meaning |
| --- | --- | --- | --- |
| `speak` | `l3_text` | `user_visible` | Selects a text surface. L2d provides surface intent, not final wording. |
| `memory_lifecycle_update` | `memory_lifecycle_specialist` | `private` | Selects specialist review for active-commitment lifecycle changes. L2d does not choose a memory target or lifecycle decision. |
| `apply_memory_lifecycle_update` | `memory_lifecycle` | `private` | Internal executable DB update produced after specialist alias validation. It is not projected to L2d. |
| `trigger_future_cognition` | `orchestrator` | `private` | Requests a later cognition cycle contract; it does not call cognition directly. |

`send_message` is intentionally absent from the L2d registry. User-visible
text is represented as `speak`, routed through the selected L3 text surface,
and delivered only through the normal live response path. Delayed user-visible
contact must be represented as `trigger_future_cognition`, so the character can
decide again at execution time whether to speak.

Prompt-safe capability projection hides `handler_id`, adapter ids, raw channel
ids, credentials, collection names, and database internals.

## Target Binding

Targetful actions are resolved by deterministic code from trusted episode,
trigger-source, RAG, and repository context.

For `memory_lifecycle_update`, L2d may only request a specialist review. The
target is a cognitive episode owned by the memory lifecycle specialist. It must
not include `unit_id`, collection names, target aliases, or lifecycle decisions.

The memory lifecycle specialist receives prompt-safe aliases such as
`commitment_1`, chooses `fulfilled`, `abandoned`, `obsolete`, or `deferred`
only when the evidence is clear, and returns aliases plus prompt-safe content
anchors. Deterministic code validates the alias, resolves it to the trusted
memory unit id, and materializes `apply_memory_lifecycle_update` for execution.

No promise or commitment is retired because it is old, overdue, keyword-matched,
or visually stale. The specialist must semantically judge the lifecycle change,
and the repository must validate the resolved target.

## Attempt Ledger

Action attempts reuse `self_cognition_action_attempts`. New rows may carry
generic action-attempt metadata, while old send-message rows must remain
readable for duplicate suppression and audit compatibility.

Do not add a second action ledger collection without a separate approved plan.

Resolver HIL and approval waits also reuse this ledger. They are deterministic
pending state, not action specs and not adapter delivery:

- `resolver_pending_hil` stores a prompt-safe clarification question selected
  by cognition.
- `resolver_pending_approval` stores a prompt-safe approval summary for a
  side effect that has not been executed.

Follow-up user turns close, approve, reject, or supersede these rows only when
L2d emits `resolver_pending_resolution`. Deterministic code must not infer
approval from keywords or execute the prepared side effect inside the resolver
stage.

## Consolidation

The consolidator receives prompt-safe episode-trace projection only. It may
learn from visible text, private action outcomes, scheduled-action outcomes,
and private finalization, but it must not select actions, execute actions, call
the dispatcher, call the scheduler, or trigger cognition.

`final_dialog` is represented as one text `SurfaceOutputV1`; it is no longer
the only possible evidence that a turn has consolidatable output.

## Deferred Tools

The next stage may add capabilities only through a reviewed plan that names
owner, handler, permissions, continuation behavior, trace output, tests, and
latency budget.

Reserved next-stage candidates:

- `schedule_self_check`: orchestrator-owned request to create a future
  self-cognition episode through scheduler/orchestrator mechanics.
- `web_research`: retrieval-owned research action that must return evidence,
  not final dialog.
- notes/open-loop tools: memory or open-loop owner actions for durable internal
  notes and closure, not arbitrary database writes.

Image generation is also a future surface handler candidate. L3 image can
produce visual prompts/directives today, but this module does not call an
external image generation service in the current implementation.

## Forbidden Paths

- Do not let L2a or L2c emit action specs.
- Do not expose raw action envelopes or deterministic ids in L2d prompts.
- Do not make `action_spec` a second dispatcher.
- Do not let tool handlers call cognition directly.
- Do not let the consolidator execute, dispatch, schedule, or select actions.
- Do not retire promises by deterministic age cleanup.
- Do not add arbitrary shell, HTTP, file, adapter, or MongoDB tools.
