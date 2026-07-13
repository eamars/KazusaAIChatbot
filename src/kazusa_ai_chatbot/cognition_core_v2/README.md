# Cognition Core V2 validation package

`cognition_core_v2` is an isolated Stage 1 validation implementation. Its only
package-root API is `run_cognition_chain`, which accepts the public V1 chain
input and service bindings and returns a validated V1 chain output.

The package owns process-local, versioned state for a single validation
process. State keys isolate character, user, trigger source, and semantic scope.
The package does not access databases, caches, production service routes,
connectors, schedulers, delivery, consolidation, or V1 stage internals.

Emotion labels are deterministic causal projections. Semantic appraisal may
produce typed propositions, but only the local reducer applies guarded state
transitions. Goal branches receive qualitative state descriptors rather than
raw numeric state. Branch results are bids; the workspace admits and collapses
them before route-only action selection.

The lifecycle CLI runs one named deterministic case at a time:

```powershell
venv\Scripts\python -m kazusa_ai_chatbot.cognition_core_v2.validation_cli lifecycle --case-id joy
```

It writes raw structured artifacts under `test_artifacts/cognition_core_v2/`.
The parent validation harness owns paired V1/V2 benchmark execution because it
must supply identical V1 inputs, model bindings, ordering, warm-up, and reset
control without adding a V1 runtime dependency to this package.

For one real-LLM case, the parent harness starts a context-local raw capture
before invoking the facade, then snapshots or writes it after the invocation:

```python
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    reset_validation_capture,
    validation_capture_snapshot,
    write_validation_capture,
)

reset_validation_capture("joy")
result = await run_cognition_chain(input_payload, services)
capture = validation_capture_snapshot()
artifact_path = write_validation_capture()
```

Each capture retains stage prompts, normalized raw output, parser/validation
status, safe route and model configuration, duration, deterministic state and
branch events, workspace/action results, and failures. API keys are excluded.
The capture contains evidence only; it never generates a human-readable review.
