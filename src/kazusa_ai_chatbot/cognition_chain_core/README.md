# Cognition Chain Core ICD

## Document Control

- Owning package: `kazusa_ai_chatbot.cognition_chain_core`
- Runtime role: reusable layered cognition chain
- Public contracts: `CognitionChainInputV1`, `CognitionChainOutputV1`,
  `CognitionTextSurfaceInputV1`, and `CognitionTextSurfaceOutputV1`
- Related docs: [Cognition Nodes](../nodes/README.md),
  [Cognition Resolver ICD](../cognition_resolver/README.md),
  [Action Spec](../action_spec/README.md)

## Purpose

`cognition_chain_core` owns the reusable L1 -> L2 -> L2d cognition chain and
selected text-surface planning. The core owns semantic appraisal, stance
formation, route-only action selection, resolver request selection, and text
surface directive planning. Callers must use the public contracts and
entrypoints exported by this package.

## Boundary

The core receives prompt-safe semantic projections. It does not receive raw
adapter payloads, database rows, delivery targets, queue jobs, scheduler
leases, executable action envelopes, or transport identifiers.

The caller owns:

- graph-state projection into `CognitionChainInputV1`;
- model construction and injection through `CognitionChainServices`;
- resolver recurrence control;
- executable action materialization and validation;
- action execution, persistence, scheduling, delivery, and consolidation.

## Source Labels

`CognitionChainInputV1.episode.model_visible_percepts` is part of the
deterministic caller-to-core interface. Its `input_source` labels are the
canonical source vocabulary consumed directly by prompt selection and
source-specific prompt payload construction. They are not a type system for
the LLM.

Accepted source labels include:

| Source label | Normal producer |
|---|---|
| `dialog_text` | Live user-message turns. |
| `internal_monologue` | Self-cognition or scheduled internal packets. |
| `reflection_artifact` | Promoted reflection dry runs. |
| `accepted_task_result` | Completed or failed accepted-task returns. |
| `image_observation` | Current-turn image observations. |
| `audio_observation` | Current-turn audio observations. |

Connectors must preserve the source identity of existing `CognitiveEpisode`
percepts when building `CognitionChainInputV1`. They must not translate source
labels through compatibility vocabularies or default self-cognition,
reflection, scheduled, or background sources to live-chat `dialog_text`.

## Public Entrypoints

```text
CognitionChainInputV1 -> run_cognition_chain -> CognitionChainOutputV1
CognitionTextSurfaceInputV1 -> run_text_surface_planning -> CognitionTextSurfaceOutputV1
```

Public entrypoints:

- `run_cognition_chain(input_payload, services)`
- `run_text_surface_planning(input_payload, services)`
- contract validators in `contracts.py`

Internal stage modules under `stages/` are implementation details. They are
not caller integration surfaces.

## Runtime Flow

Callers build prompt-safe semantic state, inject the configured LLM services,
and call the public entrypoint. The core returns semantic action requests and
surface planning data. The caller then performs recurrence, action execution,
persistence, scheduling, delivery, and consolidation outside this package.

For user-message turns, the core may receive up to three prompt-safe coding
run contexts in the L2d human payload. Stable coding-action meanings belong to
the L2d system prompt; each run's ref, state, allowed actions, and blocker
question/options are current human-message facts. L3 receives a separate
follow-up source without run refs or operational fields and uses it only to
render a blocker question or an ambiguity clarification.

`CognitionChainInputV1.action_selection_context` is required runtime handoff
state for L2d and later deterministic action materialization. Callers must
initialize it even when there are no contextual action-selection facts:

```python
{
    "coding_runs": [],
    "group_engagement_action_context": {},
}
```

The cognition wrapper may populate `coding_runs` with prompt-safe current run
affordances or `group_engagement_action_context` with prompt-safe group
engagement facts. The core must not receive raw worker payloads, database
rows, approval evidence, adapter ids, job ids, leases, or delivery targets in
this context.

## Action Boundary

The core emits `SemanticActionRequestV1` rows. These rows describe semantic
intent such as visible text, private memory review, future cognition, or
background work. They do not contain executable handler ids, target ids, job
ids, adapter ids, worker parameters, final visible dialog, or persistence
metadata.

## Failure Behavior

Invalid input contracts should fail at deterministic validation boundaries.
Malformed LLM stage output should be rejected by the owning stage parser or
validator before action materialization. The core must not silently convert
missing required semantic state into live-chat defaults.

## Testing Contract

Deterministic tests should cover contract validators, source-label handling,
stage state merge behavior, action request shape, and text-surface output
shape. Prompt or model-facing behavior requires real LLM tests run one case at
a time with emitted traces inspected.

## Forbidden Paths

- Do not pass raw adapter payloads, platform ids, database rows, queue jobs, or
  scheduler leases into the core.
- Do not emit executable handler ids, target ids, job ids, adapter ids, worker
  parameters, or final visible dialog in `SemanticActionRequestV1`.
- Do not translate source labels through compatibility vocabularies.
- Do not route persistence, adapter delivery, scheduler execution, or
  background-worker execution through this package.
