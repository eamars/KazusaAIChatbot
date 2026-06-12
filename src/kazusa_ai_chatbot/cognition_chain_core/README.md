# Cognition Chain Core

`cognition_chain_core` owns the reusable layered cognition chain. It exposes a
small ICD:

```text
CognitionChainInputV1 -> run_cognition_chain -> CognitionChainOutputV1
CognitionTextSurfaceInputV1 -> run_text_surface_planning -> CognitionTextSurfaceOutputV1
```

The core owns semantic appraisal, stance formation, route-only action
selection, resolver request selection, and selected text-surface directive
planning. Callers must use the public contracts and entrypoints exported by
this package.

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

## Public Entrypoints

- `run_cognition_chain(input_payload, services)`
- `run_text_surface_planning(input_payload, services)`
- contract validators in `contracts.py`

Internal stage modules under `stages/` are implementation details. They are
not caller integration surfaces.

## Action Boundary

The core emits `SemanticActionRequestV1` rows. These rows describe semantic
intent such as visible text, private memory review, future cognition, or
background work. They do not contain executable handler ids, target ids, job
ids, adapter ids, worker parameters, final visible dialogue, or persistence
metadata.
