# Consolidation Subsystem

`kazusa_ai_chatbot.consolidation` owns post-turn and background durable write
routing. Persona nodes still extract cognition, dialog, action results, and
prompt-safe episode traces; this package decides which durable target lanes may
receive the extracted state.

## Boundary

```text
persona/self-cognition state
  -> consolidation.core.call_consolidation_subgraph(...)
  -> consolidation.target.ConsolidationTargetPlan
  -> extraction nodes
  -> write-intent validation
  -> target-specific persistence helpers
```

The public entrypoint is:

```python
kazusa_ai_chatbot.consolidation.core.call_consolidation_subgraph
```

The old `nodes.persona_supervisor2_consolidator` module is retired as a public
entrypoint. Runtime callers should import through `consolidation.core`.

## Target Plan

Target planning is deterministic and has no LLM call. `origin_kind` records why
cognition ran; it does not grant persistence permission. `target_kind` records
the durable entity that may be written.

| Target kind | Durable meaning | Allowed lanes |
| --- | --- | --- |
| `user` | Real validated user profile | `relationship_insight`, `user_memory_units`, `affinity`, `user_style_image` |
| `group_channel` | Platform group/channel image | `group_channel_style_image` |
| `character` | Active character state/self-image | `character_state`, `character_self_image` |
| `internal` | Non-durable audit/local artifact | `audit` |

Synthetic source labels such as `self_cognition`, `group_chat_review`, or
`internal_thought` must never become `global_user_id`. Real user targets must
carry the required runtime user profile shape before persistence reaches DB
helpers. Missing `affinity` remains a lifecycle bug and should surface instead
of receiving a read-site default.

## Group And User Separation

Group-scoped consolidation receives deterministic group-channel eligibility.
That does not imply group affinity or group user-memory writes. User lanes can
only run against a real validated user target. Group-channel persistence must
not call `update_affinity`, `update_last_relationship_insight`, or
`update_user_memory_units_from_state`.

## Diagnostics

`python -m scripts.inspect_consolidation_target_lifecycle` writes a read-only
dry-run report for synthetic user rows and malformed user-profile lifecycle
data. `--apply` runs the approved cleanup path only after operators review
dry-run evidence and separately approve mutation.
