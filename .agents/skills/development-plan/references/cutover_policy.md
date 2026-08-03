# Cutover Policy Reference

Use this reference when a plan changes existing behavior, removes a legacy
path, migrates data, or requires an explicit rollout strategy.

## Strategy definitions

- **migration:** Move from old behavior to new behavior through explicit
  transitional steps. Temporary coexistence and data movement are allowed only
  where the plan specifies them. Remove old paths after verification.
- **compatible:** Preserve old and new behavior together. Compatibility shims,
  adapters, fallback paths, dual reads or writes, and old API or state shapes
  are allowed only when the plan lists them.
- **bigbang:** Replace old behavior in one cutover. Do not preserve a legacy
  path, compatibility shape, or fallback unless the plan explicitly retains
  it.

## Policy matrix

When more than one area has a different policy, state it explicitly:

```md
## Cutover Policy

Overall strategy: bigbang

| Area | Policy | Instruction |
|---|---|---|
| Service entrypoint | bigbang | Replace the legacy entrypoint directly. |
| Persisted state | migration | Convert through the approved migration phases. |
| Public API | compatible | Preserve the listed old fields until the stated removal gate. |
| Tests | bigbang | Replace tests for removed behavior with tests for the new contract. |
```

An area-specific policy overrides the overall strategy for that area.

## Enforcement

- The responsible agent follows the selected policy for each area.
- A bigbang area removes or rewrites legacy references instead of preserving
  them.
- A migration area follows the exact migration phases and cleanup gates in the
  plan.
- A compatible area preserves only the explicitly listed compatibility
  surfaces.
- A cutover-policy change requires an approved plan amendment or user
  decision before implementation.
