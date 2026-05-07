# AGENTS.md

## Operating Posture

Work as a senior system engineer. Start with the system-level picture, identify
the ownership boundaries, then move into technical detail. Do not jump to a
conclusion before reading the relevant code, docs, tests, and current git
state.

Communicate with the user before meaningful action. State what context you are
gathering, what assumption you are making, and what concrete change or command
you are about to run.

## Project Shape

Kazusa is a platform-agnostic character brain service, not a generic assistant
shell. The core boundary is:

```text
adapter/debug client -> brain service -> queue/intake -> RAG -> cognition
-> dialog -> persistence/consolidation -> scheduler/reflection
```

Adapters should stay thin. Brain code should consume typed message-envelope
fields instead of parsing raw Discord, QQ, or debug-wire syntax.

## First Files To Read

- `README.md`
- `docs/HOWTO.md`
- Relevant subsystem README files under `src/kazusa_ai_chatbot/**/README.md`
- The specific source and test files for the requested change

Never read `.env` unless the user explicitly asks for environment inspection.

## Architecture Rules

- Preserve the live response path as bounded and inspectable.
- LLM stages own semantic judgment.
- Deterministic code owns validation, persistence, limits, permissions, cache
  invalidation, scheduler execution, and adapter delivery.
- RAG returns evidence; cognition decides stance; dialog owns final wording.
- Reflection runs outside live chat. Raw reflection output must not enter normal
  cognition directly; only promoted, gated context may be used.
- Future autonomous contact must go through explicit permission, dispatcher or
  scheduler validation, adapter availability, and auditability.

## Python And Tests

- Use the project virtual environment: `venv\Scripts\python`.
- Before editing Python, apply `.agents/skills/py-style`.
- Before adding, changing, or running tests, apply
  `.agents/skills/test-style-and-execution`.
- Regular deterministic tests may run in batches.
- Live LLM tests must run one case at a time with output inspected.
- Live DB tests require an available MongoDB and should be run explicitly.

## Development Plans

For multi-file plans, migrations, decommissions, prompt or graph changes,
database changes, and risky refactors, apply
`.agents/skills/development-plan-writing`.

Completed plans are historical records. Do not append new scope to a completed
plan; create a new or superseding plan.

## Git And Files

- Check `git status --short` before editing.
- Do not revert user changes unless explicitly asked.
- Keep edits scoped to the request.
- Prefer `rg` for searches.
- Use `apply_patch` for manual edits.
