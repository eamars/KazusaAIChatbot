"""Cache-affine semantic-chain cognition engine over the V2-shaped substrate.

The package exposes one public entrypoint, ``run_cognition``, with the exact
``CognitionCoreInputV2``/``CognitionCoreOutputV2`` contract of the selected
engine family. The deterministic orchestrator owns chain selection, stage
order, visibility, checkpoints, attempt caps, validation, and failure
disposition; each semantic owner runs a bounded cache-affine transcript under
its own static system prompt.
"""
