# Cognition Resolver ICD

## Ownership

The cognition resolver owns bounded recurrence around Cognition Core V2. Its
live owner is `nodes.persona_supervisor2.stage_1_goal_resolver`; the idle owner
is `self_cognition.runner._default_cognition_client`.

The resolver executes a cognition-selected capability and returns typed
observations. It has no authority to mutate user or character cognition state,
select final wording, deliver adapter output, or reinterpret a capability
result. Cognition Core V2 remains the semantic decision owner. The connector
commits the single final replacement state before L3, action execution, dialog,
or worker delivery can proceed.

## Canonical V2 Flow

```text
CognitionCoreInputV2
  -> run_cognition(..., commit=False)
  -> ResolverCapabilityRequestV2[]
  -> one bounded capability execution
  -> typed resolver observation
  -> connector projects CognitionEvidenceV2
  -> next V2 cognition cycle
  -> terminal CognitionCoreOutputV2
  -> one state-scope commit
  -> L3/action/dialog or private terminal handling
```

`call_v2_resolver_loop(...)` owns the episode-local `ResolverWorkingStateV2`:

- `origin_scope`: `user` or `character`, fixed for the run;
- `cycle_index` and `max_cycles`: deterministic recurrence limits;
- `pending_requests`: exact `ResolverCapabilityRequestV2` rows;
- `observations`: prompt-safe capability outcomes;
- `cognition_output`: the latest complete V2 output;
- `terminal`: the loop terminal flag.

The loop carries the latest in-memory V2 output forward. It does not reload or
write cognition state between cycles. The caller commits only the final output.

`CognitionCoreOutputV2.goal_resolution` remains the semantic owner’s answerability
decision. It answers whether the accepted user goal is sufficient to answer now;
it does not mirror a source-specific RAG `resolved` field. When the decision is
`answerable_now`, the deterministic loop suppresses any optional resolver request
and settles the episode without capability execution. The other typed decisions
retain their required-evidence, user-input, or blocked paths.

## Capability Boundary

V2 requests contain only `capability`, `semantic_goal`, and
`evidence_handles`. Capability handlers may perform their bounded retrieval or
research operation and return a source-owned observation. The canonical
projection `project_resolver_observation_for_cognition(...)` produces:

- one `CognitionEvidenceV2` with `source_kind=resolver_observation`;
- complete source identity and UTC occurrence time;
- the exact source-owned semantic visibility map;
- a typed direct-fact list, empty unless a capability can prove the required
  fact provenance and targets.

An observation is evidence, never persona, intention, affect, relationship
state, or final stance. Resolver code cannot write `replacement_state`, choose
a goal branch, or rewrite an intention route.

## Public V2 Entrypoints

| Entrypoint | Owner | Purpose |
| --- | --- | --- |
| `call_v2_resolver_loop(...)` | `loop.py` | Run bounded V2 cognition/capability recurrence without intermediate state commits. |
| `execute_resolver_capability_request(...)` | `capabilities.py` | Execute one bounded source capability. |
| `project_resolver_observation_for_cognition(...)` | `capabilities.py` | Convert a source result into typed V2 evidence/direct facts. |
| `build_v2_resolver_telemetry_fields(...)` | `telemetry.py` | Emit bounded request/progress/status diagnostics without raw identifiers or state. |

Pending human clarification and approval records remain deterministic ledger
state owned by `pending.py`. When admitted to V2, they enter as typed evidence;
they do not restore the retired V1 cognition-chain contract.

## Diagnostics and Safety

Resolver diagnostics are bounded and semantic. V2 telemetry may contain
capability names, semantic goals, progress status, and a clipped progress
summary. It excludes raw prompt text, observation identifiers, evidence
handles, replacement state, state owner keys, private bids, handler parameters,
and platform identifiers.

Human-readable traces under `test_artifacts/cognition_resolver/` are diagnostic
artifacts only. They never become cognition input automatically.
