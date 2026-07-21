# AGENTS.md

## Agent Rules

- DO NOT send optional commentary.
- DO NOT assume, always require explicit instructions.
- Avoid describing action in negative statements (i.e,. I will NOT ...). Always state your action in positive statement only (i.e,. I will ...).

## How To Work Here

Work as a senior system engineer. Start with the system-level picture, identify
the ownership boundaries, then move into implementation detail. Do not jump to
a conclusion before reading the relevant code, docs, tests, and current git
state.

Communicate before meaningful action. Tell the user what context you are
gathering and what concrete command or change you are about to run. If an
instruction, scope, target file, data source, approval, risk level, or expected
output is unclear, stop and ask the user for clarification before acting. Do
not fill gaps with assumptions, and do not treat a stated assumption as
permission to proceed.

Before substantive edits, and always before production-code edits, check:

- `git status --short`
- `README.md`
- `docs/HOWTO.md`
- Relevant subsystem README files under `src/kazusa_ai_chatbot/**/README.md`
- The source and test files directly involved in the request

Never read `.env` unless the user explicitly asks for environment inspection.

## Production Code Change Control

- Do not modify, add, or remove production code unless the user explicitly
  commands that implementation change. For this rule, `experiments/*` and
  `tests/*` are not production code, but normal scope and safety rules still
  apply.
- Analysis is not authorization. Failure analysis, RCA, debugging,
  investigation, review, and diagnosis may identify fixes, but they do not
  authorize production-code changes.
- Before any production-code change, answer the user's outstanding questions
  and confirm unclear intent. Do not treat assumptions, plan status, or
  inferred approval as permission to touch production code.

## Project Model

Kazusa is a platform-agnostic character brain service, not a generic assistant
shell. The core boundary is:

```text
adapter/debug client -> brain service -> queue/intake -> RAG -> cognition
-> dialog -> persistence/consolidation -> scheduler/reflection
```

Adapters stay thin. They normalize platform events and deliver returned
surfaces. Brain code consumes typed message-envelope fields and must not parse
raw Discord, QQ, or debug-wire syntax as its main contract.

Keep these ownership lines clear:

- RAG returns evidence.
- Cognition decides stance, boundaries, character judgment, and response goals.
- Dialog and L3 surfaces own final wording and visible rendering.
- Persistence, consolidation, scheduler, and reflection maintain continuity
  outside the live response wording path.

## Character Judgment Goal

Evaluate behavior as a character brain, not as a generic assistant optimized
for safe, low-response output. The target behavior is believable character
judgment grounded in observation, context, mood, relationship, and scene
pressure.

Do not propose mechanical suppression just because a dialog result is awkward
or a previous response was noisy. If the character has enough reason and enough
observed context to speak, she should speak. If the reason is weak, stale,
self-referential, or based only on the existence of an internal cognition
window, she should stay quiet because the character lacks a grounded reason.

When investigating response sensitivity, judge the quality of the character's
reason to speak first. Response ratio, gating, and engagement tuning are tools
for preserving character judgment and topic fit; they are not the product goal.

## Architecture Guardrails

- Preserve the live response path as bounded and inspectable.
- LLM stages own semantic judgment.
- Deterministic code owns validation, persistence, limits, permissions, cache
  invalidation, scheduler execution, and adapter delivery.
- Avoid compatibility shim layers at all cost unless the user explicitly
  instructs otherwise. Prefer big-bang contract updates where the caller,
  callee, tests, and ICD all move to one canonical boundary in the same scope.
  Do not introduce parallel vocabularies, alias modules, fallback mappers, or
  translation bridges merely to preserve old call shapes during a refactor.
- RAG evidence must not be treated as persona or final stance.
- Reflection runs outside live chat. Raw reflection output must not enter normal
  cognition directly; only promoted, gated context may be used.
- Future autonomous contact must go through explicit permission, dispatcher or
  scheduler validation, adapter availability, and auditability.

## LLM Error State Handling

Bounded evaluators classify erratic model output by whether the contract can
repair it deterministically without changing semantic ownership.

- First pass every raw LLM response through the canonical
  `kazusa_ai_chatbot.utils.parse_llm_json_output(...)` entry point before
  evaluating semantic values; agents reuse this function instead of creating
  stage-local JSON parsers or repairers. It strips transport formatting and
  applies deterministic JSON cleanup, then delegates to
  `kazusa_ai_chatbot.utils.parse_json_with_llm(...)` on `JSON_REPAIR_LLM` only
  for residual syntax, wrapper, or object-shape repairs when the stage permits
  the fallback. JSON repair preserves supported raw keys and values and does
  not invent semantic values, correct domain decisions, or bypass the later
  contract evaluator. A response that still cannot yield the required JSON
  object is a structural contract error and follows the bounded regeneration
  or fail-closed path.
- For non-recoverable contract errors, such as malformed structure, invalid or
  unknown keys, wrong types, unsupported enum or handle values, missing required
  fields, or conflicting fields, the evaluator reports a typed contract error
  to the owning LLM stage. That stage requests a bounded regeneration or
  complete replacement using the same semantic context. The evaluator keeps
  the invalid candidate out of action, persistence, scheduling, dialog, and
  delivery paths.
- For recoverable bound violations, the evaluator first applies the contract's
  deterministic normalization, such as floor/ceil conversion or clamping to
  the declared interval, and then validates the normalized result. If the
  normalized value still violates the contract, or normalization cannot be
  applied safely, the evaluator treats the candidate as non-recoverable and
  triggers the bounded regeneration path.
- Repair and regeneration remain owned by the producing semantic stage, use
  the stage's explicit attempt cap, and record the original error, any
  normalization or replacement, and the final disposition. After the cap, the
  boundary fails closed with a typed result and preserves deterministic
  permission, provenance, target, limit, and persistence checks.

## Python And Tests

- Use the project virtual environment: `venv\Scripts\python`.
- Before editing Python, apply `.agents/skills/py-style`.
- Treat `.agents/skills/py-style` as the canonical coding policy, including
  its fail-fast rules for required internal data, defaults, and exception
  handling. Do not duplicate or weaken those rules in this file.
- Before adding, changing, or running tests, apply
  `.agents/skills/test-style-and-execution`.
- Regular deterministic tests may run in batches.
- Live LLM tests must run one case at a time with output inspected.
- Live DB tests require an available MongoDB and should be run explicitly.

## Development Plans

For multi-file plans, migrations, decommissions, prompt or graph changes,
database changes, and risky refactors, apply
`.agents/skills/development-plan`.

Before reading or executing development plans, read
`development_plans/README.md`. It is the lifecycle registry for long-term,
active, archived, reference, and triage documents.

Use plan status as an execution boundary, not as authorization to modify
production code:

- `development_plans/long_term/todo.md` is a living roadmap, not an executable
  work contract.
- Promote roadmap items into `development_plans/active/short_term/` or
  `development_plans/active/bugfix/` before implementation.
- Only execute plans in `development_plans/active/` whose `Status` is
  `approved` or `in_progress`.
- Plan status is necessary but not sufficient for production-code changes; the
  user must still explicitly command the implementation work.
- Treat `draft` plans as discussion artifacts.
- Treat archived plans as historical records.
- Treat reference documents as context only.
- Treat triage files as blocked until classified.
- Completed plans are historical records. Do not append new scope to a
  completed plan; create a new or superseding plan.

## Git And Files

- Do not revert user changes unless explicitly asked.
- Keep edits scoped to the request.
- Prefer `rg` for searches.
- Use `apply_patch` for manual edits.
- PowerShell path safety: use `-LiteralPath '...'` for filesystem paths,
  especially Windows absolute paths and any path that may contain spaces.
  Never pass unquoted paths to commands. Prefer repo-relative paths when
  possible.
