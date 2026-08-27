# Cognition Observability Interface Control Document

## Contract owner

The Brain service is the sole schema and projection owner for the process-local
`cognition_run_observation.v1` wire contract. It publishes one validated
terminal observation through `ChatResponse.cognition_graph` and the latest
observation endpoint. The console is a validation-only consumer: it wraps a
validated observation in presentation metadata and never reconstructs its
nodes, sections, labels, or values.

The producer and publisher are Brain-owned; transport carries the validated
object, and every consumer uses the same canonical DTO boundary.

## Version and disclosure

The schema version is `cognition_run_observation.v1`. Every observation carries
the fixed `approved_cognition_observation.v1` disclosure policy and the stable
excluded categories: prompts, raw model output, embeddings, raw messages,
message envelopes, database identifiers, adapter identifiers, action parameters,
handler metadata, and worker error text. A breaking semantic reinterpretation
requires a new major schema and an approved cutover.

All DTOs are frozen, reject unknown fields, use strict scalar validation, and
serialize aware UTC timestamps with a terminal `Z`. Section and record counts
are truthful. Records use `item_01` through `item_24` in producer source order;
edges use only `sequence` and `reference` with known endpoints.

## Producer catalog

Live turns publish the ordered sections `input.turn`, `decision.response`,
`cognition.appraisals`, `cognition.goal`, `cognition.response_plan`,
`cognition.affect`, `reasoning.subjective`, `reasoning.context_consumption`,
`evidence.memory`, `evidence.shared_memory_prewarm`,
`context.conversation_progress`, `context.public_group_scene`,
`action.requests`, `action.results`, `action.continuation`,
`surface.visual_directives`, and `surface.visible_messages`.

Self-cognition publishes the shared sections plus `self.source`, `self.route`,
and `self.consolidation`, in the fixed order defined by the Brain producer.
Producer-approved additive sections may be appended without changing the base
catalog. Nodes reference sections in source order, so consumers can render a
new additive section generically without a client release.

## Timing and terminal status

Live observations end at cognition, selected actions, and surface generation.
Later persistence and consolidation remain outside the live snapshot. A
successful visible, private, action, or scheduled terminal disposition maps to
`completed`; visual-surface failure after a successful cognition maps to
`partial`; an already failed run remains `failed`. Cancellation publishes no
fabricated terminal observation. Self-cognition may include its completed
consolidation artifact because that artifact belongs to the self run.

The shared-memory prewarm section reports one of the fixed reason codes:
`worker_unresolved`, `worker_contract_invalid`, `projection_failed`,
`no_shared_memory`, `worker_error`, `shared_memory_ready`,
`shared_memory_merged`, `empty_query_after_character_mention`, `not_first_cycle`,
or `unsupported_episode`. The section status, retrieved count, merged count,
and omission marker are derived from the typed prewarm outcome.

## Live-response recovery ladder

Every governed live-response stage applies the bounded recovery ladder in order.
T1 (`recover`) performs deterministic normalization that adds no semantics and
records `normalized`. T2 (`regenerate`) performs bounded same-context
regeneration carrying the exact contract error and bounded rejected candidate;
provider failure consumes one attempt and the no-candidate reason is used.
T2's in-progress disposition is `regenerate`. T3 (`degrade`) delivers an
already-available sibling result, retained candidate, or deterministic
projection of already-validated upstream truth and records `accepted_degraded`
or `skipped`. T4 (`replay`) raises a typed `CognitionExecutionError` with
`retryable=True` and `safe_checkpoint="pre_state_commit"`, so the existing
single service-level graph replay records `retry_graph` and starts from the
settled-relevance and prewarm checkpoint. A stage without an assigned T3 or T4
may fail through its typed boundary; dialog has T3 and no visible-path failure
exit.

The fixed final-status vocabulary is `accepted_degraded`, `skipped`,
`exhausted`, `retry_graph`, and `normalized`. The fixed recovery error-code
vocabulary is:

- cognition: `cognition_a1_contract_exhausted`,
  `cognition_a2_contract_exhausted`, `cognition_g_contract_exhausted`,
  `cognition_p_contract_exhausted`, and
  `cognition_turn_deadline_exhausted`;
- dialog: `dialog_source_url_degraded` and
  `dialog_surface_projection_degraded`;
- lifecycle and visual: `memory_lifecycle_skipped` and
  `surface_visual_omitted`;
- local context: `local_context_planner_blocked`,
  `local_context_node_blocked`, `local_context_collapse_skipped`, and
  `local_context_synthesis_degraded`;
- relevance: `settled_relevance_deterministic_degraded` and
  `frontline_relevance_deterministic_degraded`.

The Brain service maps an exhausted `CognitionExecutionError` to the
public `model_contract` error category. It permits exactly one graph replay only when
the failure is retryable and has `safe_checkpoint="pre_state_commit"`, using
the existing `COGNITION_SAFE_RETRY_LIMIT`; a post-commit failure never replays.
Each protected LLM trace retains the truthful provider, contract, normalized,
repaired, or deterministic attempt disposition and any raw-attempt evidence
allowed by the protected trace contract. Episode diagnostics are separate
bounded `episode_attempt_diagnostic.v1` rows owned by the stage/carrier
boundary; they never replace the protected trace. T1 results do not create an episode row, while assigned T3/T4 terminal metadata uses the fixed status and
error-code vocabulary above.

## Relevance diagnostic envelope

The relevance producers return one canonical `RelevanceEvaluationEnvelope`
with exactly two keys: `decision` and `attempt_diagnostics`. The unchanged
`FrontlineDecision` or `SettledRelevanceDecision` is nested under `decision`
and remains the only value consumed by its decision validator. T1 normalized
results carry an empty diagnostic list. Only deterministic T3 relevance
degradation carries one `episode_attempt_diagnostic.v1` row, using
`frontline_relevance_deterministic_degraded` or
`settled_relevance_deterministic_degraded`; a provider attempt is retained in
the protected LLM trace and is not promoted as an episode row.

The frontline envelope rows are stored on the pending turn in arrival order.
Valid append rows follow start rows; invalid append targets and discard drop
their rows. A settled row follows the retained frontline rows. A stale lease
never merges its rows. A wait merges its rows once into the next pending lease,
and a later settled result appends after them. Ignore drops the carrier with
the turn. Only a current claimable `proceed` exposes rows through
`SettlementOutcome.attempt_diagnostics`; the service passes that field
separately into `IMProcessState` before persona work, and the existing
`episode_trace.v2` post-turn consumer carries it into the final episode.

`state.append_attempt_diagnostics` is the sole reducer and normalizer. It
retains `combined[-MAX_EPISODE_ATTEMPT_DIAGNOSTICS:]`, namely the most recent 16
rows in chronological order. The terminal dialog, surface, and cognition
rows therefore remain discoverable after earlier relevance degradation rows;
earlier relevance attempts remain available in protected traces. A discarded
turn that never reaches a settled episode persists no episode diagnostic row,
while its protected relevance trace remains authoritative. Persona starts its
internal reducer-backed accumulator empty and returns only its downstream
delta; the top-level reducer merges inherited relevance rows and this delta
once, without a second cap or dedupe.

## V1 DTO boundary and budgets

The wire models are `CognitionRunObservationV1`,
`CognitionObservationCorrelationV1`, `CognitionObservationDisclosureV1`,
`CognitionObservationSectionV1`, `CognitionObservationFieldV1`,
`CognitionObservationRecordV1`, `CognitionObservationNodeV1`, and
`CognitionObservationEdgeV1`. Their required top-level values include
`schema_version`, `run_kind`, `status`, `generated_at`, `correlation`,
`disclosure`, `sections`, `nodes`, and `edges`. `run_kind` is exactly
`live_turn` or `self_cognition`, and top-level status is exactly
`completed`, `failed`, or `partial`; section and node statuses also retain
`empty`, `skipped`, and `not_reported`.

Every model is strict, frozen, and extra-forbid. Identifiers, section
references, labels, keys, summaries, scalar values, list items, node/edge
counts, and record counts use the v1 bounds. Generated record keys are
`item_01` through `item_24` in source order. `reported_record_count`,
`displayed_record_count`, and `truncated` are truthful and mutually
consistent. The compact UTF-8-preserving serialized observation is bounded to
`131072` characters. Timestamps serialize as aware UTC with a terminal `Z`.

The disclosure policy key is `approved_cognition_observation.v1`; its stable
excluded categories are `prompt`, `raw_model_output`, `embedding`,
`raw_message`, `message_envelope`, `database_identifier`,
`adapter_identifier`, `action_parameter`, `handler_metadata`, and
`worker_error_text`. No excluded value is copied into a field, record,
summary, node, or edge.

The required live base sections are `input.turn` and `decision.response`.
The required self base sections are `self.source`, `self.route`, and
`self.consolidation`; both run kinds use the shared cognition, reasoning,
evidence, context, action, and surface sections listed above. The closed
source-to-wire mapping preserves producer order and allows
Producer-approved additive sections only when they satisfy the same generic
grammar and budgets. Additive sections remain visible to the generic renderer
without a console catalog update.

## Prewarm, publication, and console availability

`SharedMemoryPrewarmOutcomeV1` is the resolver-owned typed carrier. Its fixed
dispositions are `worker_unresolved`, `worker_contract_invalid`,
`projection_failed`, `no_shared_memory`, `worker_error`, `shared_memory_ready`,
`shared_memory_merged`, `empty_query_after_character_mention`,
`not_first_cycle`, and `unsupported_episode`. The direct one-attempt
shared-memory worker produces a validated ready outcome; the cognition caller
performs the one merge and retains the finalized merged outcome. A skipped
outcome records ineligibility without starting the worker, and cancellation
publishes no terminal observation.

The process-local latest value is a deep copy and is not historical
persistence. `ConsoleCognitionObservationView` is a validation-only envelope
with `available`, `not_reported`, `unavailable`, or `invalid` availability.
`available` carries a matching observation; an unavailable or invalid view
carries a safe reason code and no observation. Overview, Debug, and Self use
the same ordered sections, nodes, statuses, counts, omissions, and additive
rendering rules. Browser verification covers those views, HTML escaping,
CJK/emoji/multiline values, loading/error separation, and zero page or console
errors; live LLM cases remain isolated from deterministic contract checks.

## Consumers and verification

Brain process-local latest storage is a deep copy and is not historical
persistence. The control console uses `ConsoleCognitionObservationView` with
`available`, `not_reported`, `unavailable`, or `invalid` availability. The
envelope timestamp is console-owned and never replaces the Brain timestamp.
Overview, Debug, and Self render the same producer-driven section layout.

Deterministic contract and projection tests run with the regular pytest
command. Browser checks exercise exact ordered layouts, additive producer
sections, CJK/emoji/multiline text, HTML escaping, status/count/omission
rendering, loading/error separation, and zero page or console errors. Live LLM
cases run individually and are never used as a substitute for contract tests.
