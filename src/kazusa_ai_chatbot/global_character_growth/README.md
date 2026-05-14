# Global Character Growth Interface Control Document

## Document Control

- ICD id: `GCG-ICD-001`
- Owning package: `kazusa_ai_chatbot.global_character_growth`
- Interface boundary: promoted reflection memory -> candidate LLM -> deterministic
  validation and drift -> growth-trait DB interface -> prompt-safe projection
- Runtime consumers: reflection worker, manual CLI, promoted reflection context,
  deterministic tests, and live LLM contract tests
- Upstream owner: `kazusa_ai_chatbot.memory_evolution`
- Downstream owner: L2 cognition through promoted reflection context only

This document defines the production contract for global character growth. The
module turns already promoted reflection memory into slow, auditable,
source-detail-free character-growth guidance.

## Purpose

`global_character_growth` owns durable global growth of the active character's
long-term interpersonal posture. It is for human-like character development:
boundary timing, guarded care, playful challenge, repair after tension,
clarity, emotional exposure, trust calibration, and general cooperation
patterns.

The module is not a scoped adaptation layer, not a profile writer, not a memory
promotion lane, and not a live-chat response generator. It never writes legacy
character profile surfaces, user-scoped memory, adapter output, conversation
rows, scheduler state, or dialog text.

## Boundary Summary

```text
active reflection-promoted memory
  -> prompt-card projection
  -> global-growth candidate LLM
  -> deterministic validation
  -> stable drift trait planning
  -> db.global_character_growth
  -> prompt-safe promoted trait projection
```

Only the DB-interface module may own raw MongoDB operations for the new growth
collections. This package calls that interface by name and never reaches
MongoDB directly.

## Public Entry Points

Use package exports from `kazusa_ai_chatbot.global_character_growth`:

```python
from kazusa_ai_chatbot.global_character_growth import (
    build_global_character_growth_context,
    run_global_character_growth_pass,
)
```

### `run_global_character_growth_pass`

```python
async def run_global_character_growth_pass(
    *,
    character_local_date: str | None,
    dry_run: bool,
    enable_trait_writes: bool,
    limit: int = 80,
    now: datetime | None = None,
) -> GlobalCharacterGrowthRunResult
```

Runs one background growth pass. Dry-run mode writes a run record but does not
mutate trait rows. Apply mode requires `enable_trait_writes=True` and may write
only `global_character_growth_traits` plus one run record.

The pass reads active promoted reflection memory through
`find_active_memory_units` with:

```python
{
    "source_kind": "reflection_inferred",
    "authority": "reflection_promoted",
    "source_global_user_id": "",
}
```

### `build_global_character_growth_context`

```python
async def build_global_character_growth_context(
    *,
    limit: int = 3,
) -> GlobalCharacterGrowthContext
```

Returns only active promoted growth traits for prompt use. Empty or unpromoted
state returns `{}`.

Prompt-visible rows contain only:

```python
{
    "growth_axis": str,
    "guidance": str,
    "maturity": "promoted",
    "updated_at": "YYYY-MM-DD",
}
```

No source memory ids, reflection run ids, numeric strength values, rejected
candidates, or lower-band traits may enter prompt context.

## Internal Modules

- `models.py`: constants, caps, status labels, provisional drift constants, and
  typed contracts.
- `projection.py`: prompt-card, current-trait, runtime-context, and log-only
  review projections.
- `llm.py`: Chinese candidate-generation prompt, consolidation LLM instance,
  prompt renderer, and one-shot handler.
- `validation.py`: deterministic candidate validation, source-card checks,
  duplicate checks, privacy checks, and domain-topic rejection.
- `drift.py`: gradual evidence accumulation, maturity bands, prompt-visibility
  checks, and trait-update planning.
- `runner.py`: end-to-end background pass orchestration and run-document
  construction.
- `context.py`: L2-safe runtime projection facade.

## Data Contracts

### Trait Documents

The `global_character_growth_traits` collection stores one active or inactive
growth-trait ledger row per tracked global trait. Rows include:

- stable `trait_id`,
- `growth_axis`,
- `trait_name`,
- `guidance`,
- numeric drift state,
- semantic maturity band,
- status,
- source memory ids,
- source reflection run ids,
- first/last observed dates,
- evidence counts,
- creation/update timestamps.

Numeric strength is audit data only. Runtime prompts receive maturity labels,
not strength values.

### Run Documents

The `global_character_growth_runs` collection stores every skipped, dry-run,
applied, or failed invocation. Rows include:

- stable `run_id`,
- status and mode,
- input-quality diagnostics,
- source ids,
- accepted candidates,
- rejected candidates,
- trait updates,
- log-only review projection,
- prompt-budget diagnostics,
- validation warnings,
- raw LLM output,
- summary,
- error text.

The review projection is for operators and tests only. It must never be merged
into cognition, RAG, dialog, adapters, or prompt-facing reflection context.

## Database Interface

All storage access belongs to:

```text
kazusa_ai_chatbot.db.global_character_growth
```

Approved DB-interface functions:

```python
async def ensure_global_character_growth_indexes() -> None
async def list_active_growth_traits(limit: int = 12) -> list[dict]
async def list_prompt_visible_growth_traits(limit: int = 3) -> list[dict]
async def upsert_growth_trait_documents(trait_documents: list[dict]) -> None
async def insert_growth_run_document(document: dict) -> None
```

Collections:

- `global_character_growth_traits`
- `global_character_growth_runs`

Required indexes:

- `global_growth_trait_id_unique`
- `global_growth_trait_status_maturity`
- `global_growth_trait_axis_status`
- `global_growth_trait_source_memory`
- `global_growth_run_id_unique`
- `global_growth_run_status_updated`
- `global_growth_run_source_memory`
- `global_growth_run_source_reflection`

Bootstrap creates collections and indexes only. No historical rows are
rewritten or backfilled.

## LLM Contract

The candidate LLM is a background consolidation-route call. It receives bounded
memory cards and current trait summaries, then proposes candidates only.

Before the candidate LLM is called, the final rendered system+human prompt is
measured against `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET`, which defaults
to `32000` characters. If the rendered prompt exceeds the budget, memory cards
are dropped from the tail of the already-ranked card list until the prompt fits
or no cards remain. If all cards are dropped and the rendered prompt still
exceeds budget, the run is skipped and the LLM is not called.

The LLM must not:

- generate database operations,
- generate ids,
- assign numeric strength,
- decide write mode,
- read raw transcripts,
- infer per-user or per-channel preferences as global growth,
- promote technology, product, food, location, hobby, or other topic competence.

After JSON parsing, deterministic validation owns all acceptance and rejection
decisions. There is no LLM repair loop for missing fields.

## Drift Contract

Drift constants are provisional first calibration values. They are centralized
in `models.py` and must be retuned only through a future evidence-backed
change.

Only active traits in the `promoted` maturity band are prompt-visible.
`emerging` and `stabilizing` traits are audit-visible only.

## Worker Integration

The reflection worker invokes the growth pass after daily global reflection
promotion when:

- `GLOBAL_CHARACTER_GROWTH_PASS_ENABLED=true`,
- `GLOBAL_CHARACTER_GROWTH_PROMPT_CHAR_BUDGET=32000`,
- the worker reaches the promotion step,
- the busy probe remains idle after promotion.

The pass flag defaults to `true` and is the rollback switch. The prompt budget
is a conservative character budget for the final rendered prompt, not an exact
token budget. Changing either value requires a process restart unless a test
monkeypatches the loaded module value.

## Manual Operations

Dry-run:

```powershell
venv\Scripts\python -m scripts.run_global_character_growth --dry-run --limit 80
```

Apply:

```powershell
venv\Scripts\python -m scripts.run_global_character_growth --apply --enable-trait-writes --limit 80
```

Unsafe apply is rejected before writes:

```powershell
venv\Scripts\python -m scripts.run_global_character_growth --apply --limit 80
```

The CLI prints run id, status, dry-run/apply mode, eligible cards, accepted and
rejected candidate counts, trait update counts, prompt-visible promoted count,
review projection count, input-quality density, and warning count.

## Failure Modes

- No eligible promoted reflection memory: write a skipped run record.
- Prompt budget drops all eligible cards: write a skipped run record and do not
  call the candidate LLM.
- LLM endpoint or parsing failure: write a failed run record with error text.
- Dry-run with write enablement: raise before writes.
- Apply without write enablement: raise before writes.
- Invalid candidates: reject deterministically and preserve rejection evidence
  in the run record.

## Verification

Focused deterministic tests live in:

```text
tests/test_global_character_growth_*.py
```

Live LLM tests must be run one at a time with the `live_llm` marker enabled:

```powershell
venv\Scripts\python -m pytest -m live_llm tests\test_global_character_growth_live_llm.py::test_global_character_growth_live_accepts_stable_communication_growth -q -s
venv\Scripts\python -m pytest -m live_llm tests\test_global_character_growth_live_llm.py::test_global_character_growth_live_rejects_domain_and_user_specific_noise -q -s
```

Trace artifacts are written to:

```text
test_artifacts/llm_traces/
```

## Handoff Notes

- This package does not improve upstream reflection-promotion quality. Sparse
  or noisy promoted memory remains visible through input-quality diagnostics.
- This package does not consume self-cognition outputs. Any future convergence
  must widen the input contract through a new approved plan.
- This package does not create a separate cognition flag. Runtime exposure uses
  the existing promoted reflection context path.
