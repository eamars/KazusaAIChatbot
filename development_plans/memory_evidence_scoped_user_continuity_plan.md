# memory evidence scoped user continuity plan

## Summary

- Goal: Make `Memory-evidence:` retrieve scoped user-continuity memories from `user_memory_units` while preserving scope, provenance, and consolidator merge/evolve continuity.
- Plan class: large
- Status: approved
- Mandatory skills: `local-llm-architecture`, `no-prepost-user-input`, `py-style`, `test-style-and-execution`, `cjk-safety`
- Overall cutover strategy: compatible internal expansion of the existing `Memory-evidence:` capability. No new top-level RAG prefix, no database migration, no response-path LLM call increase.
- Highest-risk areas: scope collision between global/shared memory and user continuity, flattening `user_memory_units` into unscoped `memory_evidence`, duplicate consolidation writes, and prompt/route drift around `Person-context`.
- Acceptance criteria: user-scoped continuity such as the `学姐` ice-cream-shop lore is retrievable through `Memory-evidence:` for the current user, carries explicit scope/provenance metadata, feeds `user_memory_unit_candidates` for merge/evolve, and does not affect other users or shared memory lookup.

## Context

The current system writes long-term interaction progression into `user_memory_units`, but RAG2 retrieves those records primarily through `Person-context:` profile hydration. This creates a broken loop for private continuity facts that look like object/place/story memory rather than profile context. The `学姐` case demonstrated the issue:

```text
assistant invented: Kazusa's middle-school 学姐 runs an ice-cream shop
user adopted it later
consolidator wrote it to user_memory_units
RAG2 may route later lookup to Memory-evidence or Conversation-evidence instead of Person-context
```

The project direction is that character-generated lore may become durable continuity. The primary risk is not canon pollution. The primary risk is unscoped retrieval: private user continuity, channel continuity, global character lore, seeded official facts, and shared common-sense memory must not collapse into one ambiguous evidence pool.

Previous plans already point in this direction but do not close the loop:

- `rag_phase3_development_plan.md` says `MemoryEvidenceAgent` should cover durable character/world facts, user memory units, shared memory/common-sense entries, and source-authority boundaries.
- The implemented `memory_evidence_agent.py` currently routes only to `persistent_memory_keyword_agent` and `persistent_memory_search_agent`, which search the shared `memory` collection.
- `user_memory_unit_rolling_window_plan.md` made `user_memory_units` the canonical durable user-memory store and states that RAG owns response-time retrieval and consolidator merge-candidate retrieval.

This plan implements the short-to-medium-term fix: keep `Memory-evidence:` as the umbrella capability, but make it internally scope-aware and able to retrieve current-user continuity from `user_memory_units`.

Long-term global lore growth is documented under `Future Reference`; it is explicitly deferred from this plan.

## Mandatory Skills

- `local-llm-architecture`: load before changing RAG capability routing, projection, prompt wording, or consolidator prompt/LLM boundaries.
- `no-prepost-user-input`: load before changing persistence admission, memory semantics, or any code that could deterministically rewrite user/character-generated meaning.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: load before editing Python files that contain CJK prompt strings or CJK test strings.

## Mandatory Rules

- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.
- Preserve the RAG2 top-level contract: initializer emits semantic capability slots; capability agents choose low-level workers; deterministic code validates scope, structure, limits, and execution.
- Do not add a new top-level prefix for the immediate fix. `Memory-evidence:` remains the user-facing RAG2 capability prefix.
- Do not make the initializer choose MongoDB collections, physical indexes, or worker internals.
- Do not add code-side semantic gates over raw user text. Scope validation and field-shape validation are allowed; semantic create/merge/evolve remains LLM-owned in the consolidator.
- Do not add response-path LLM calls. The new scoped user-memory worker must use deterministic routing plus existing embedding/vector or lexical database retrieval.
- The new scoped user-memory worker must have its own focused unit test file: `tests/test_user_memory_evidence_agent.py`. Do not satisfy the worker contract only by extending existing test files.
- Do not broaden `Person-context:`. Person/profile/relationship reads remain there. This plan only makes memory-topic lookup able to read scoped user continuity when the topic is not a profile/relationship question.
- Do not treat current-user scoped continuity as shared/global fact. Every user-memory result must carry scope metadata.
- Do not migrate existing `user_memory_units` rows. Existing rows are already scoped by `global_user_id`; missing provenance fields are projected with conservative defaults.
- Do not change cognition L2/L3 output schemas.
- If prompt wording is changed, run a runtime prompt-render check, not only `py_compile`.

## Must Do

- Add a scoped user-memory evidence worker under RAG that searches `user_memory_units` for the current `global_user_id`.
- The worker must support both scoped semantic retrieval and scoped lexical retrieval over `fact`, `subjective_appraisal`, and `relationship_signal`. Lexical matching must escape user/query text and must always filter by `global_user_id`.
- Extend `MemoryEvidenceAgent` so user-continuity memory slots route to the scoped user-memory worker, while shared/world/official memory slots continue to use existing persistent-memory workers.
- Project scoped user-memory results with explicit metadata:
  - `source_system="user_memory_units"`
  - `scope_type="user_continuity"`
  - `scope_global_user_id=<current global_user_id>`
  - `authority="scoped_continuity"`
  - `truth_status="character_lore_or_interaction_continuity"` unless a stored row provides a stronger value
  - `origin="consolidated_interaction"` unless a stored row provides a stronger value
- Ensure scoped user-memory rows returned through `Memory-evidence:` also populate `rag_result.user_memory_unit_candidates` so consolidation can merge/evolve instead of creating duplicate memories.
- Preserve existing `memory_evidence` projection for shared `memory` collection rows.
- Add deterministic tests proving scoped lookup, projection, and consolidator merge-candidate behavior.
- Update RAG documentation to describe scoped user-continuity retrieval under `Memory-evidence:`.
- Add the long-term global lore promotion direction as future reference only.

## Deferred

- Do not implement global lore promotion in this plan.
- Do not add a `global_lore` collection, promotion queue, or cross-user lore merger in this plan.
- Do not migrate existing `memory` or `user_memory_units` documents.
- Do not add channel/group continuity scope in this plan.
- Do not redesign `Person-context`, `Recall`, conversation evidence, cognition, dialog, scheduler, or Cache2.
- Do not add a new top-level `User-memory-evidence:` prefix unless this plan is superseded.
- Do not add manual admin review UI or tooling for lore promotion.

## Cutover Policy

| Area | Policy | Notes |
|---|---|---|
| RAG2 initializer prefix set | compatible | No new top-level prefix. Existing `Memory-evidence:` remains valid. |
| `MemoryEvidenceAgent` internals | compatible | Adds one internal worker path for scoped user continuity. Existing shared memory paths remain. |
| `user_memory_units` storage | compatible | No migration; existing `global_user_id` is the scope key. |
| RAG projection | compatible | Adds metadata and candidate projection for scoped user-memory rows. Existing fields remain. |
| Consolidator merge/evolve | compatible | Reuses `user_memory_unit_candidates`; no new background LLM calls. |
| Cache2 | compatible | Top-level capability remains uncached; existing user-profile invalidation remains authoritative for user-memory writes. |

## Agent Autonomy Boundaries

- The agent may choose private helper names only when the public contracts in this plan remain intact.
- The agent must not introduce new architecture, alternate routing strategies, compatibility layers, fallback paths, or extra features.
- The target ownership boundary is RAG scoped memory evidence plus projection. Changes outside this boundary require explicit justification in `Execution Evidence`.
- The agent must search for existing retrieval helpers before adding new database helpers. Reuse `query_user_memory_units`, `search_user_memory_units_by_vector`, and projection helpers where they fit.
- Scoped lexical search is required for exact/literal continuity terms. It must be a structural retrieval helper over `fact`, `subjective_appraisal`, and `relationship_signal`; it must not classify semantic meaning in code.
- If the implementation discovers that existing tests encode a conflicting contract, update tests only to match this plan's approved contract and record the old expectation.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

The response-time lookup path becomes:

```text
user query
  -> RAG2 initializer emits Memory-evidence for durable memory/topic continuity
  -> dispatcher sends slot to memory_evidence_agent
  -> memory_evidence_agent selects one internal worker:
       scoped_user_memory_evidence_worker for current-user continuity
       persistent_memory_keyword_agent for exact shared/official memory
       persistent_memory_search_agent for semantic shared/official memory
  -> projection emits rag_result.memory_evidence
  -> projection also emits rag_result.user_memory_unit_candidates for scoped rows
  -> cognition/dialog use scoped evidence
  -> consolidator merge/evolve sees same user-memory candidates
```

The `学姐` example should resolve as:

```json
{
  "summary": "In this user's continuity, Kazusa introduced/adopted a story about a middle-school 学姐 connected to a matcha ice-cream shop.",
  "content": "冰淇淋摊老板是千纱的初中学姐，千纱每次去都能蹭到双倍抹茶酱。",
  "source_system": "user_memory_units",
  "scope_type": "user_continuity",
  "scope_global_user_id": "current user UUID",
  "authority": "scoped_continuity",
  "truth_status": "character_lore_or_interaction_continuity",
  "origin": "consolidated_interaction"
}
```

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Top-level route | Keep `Memory-evidence:` | The user wants this as memory lookup, not a profile-only path. Avoid adding initializer complexity. |
| Scope model | Treat `global_user_id` as required scope for user continuity | Existing `user_memory_units` are already user-owned. |
| Worker boundary | Add a scoped user-memory worker under `MemoryEvidenceAgent` | Keeps initializer semantic and hides physical storage selection. |
| Projection | Preserve metadata and populate `user_memory_unit_candidates` | Closes the retrieval/write loop and prevents duplicate consolidation. |
| Existing rows | No migration; project default provenance | Immediate fix should work on current data. |
| Conflict handling | Merge/evolve within same scope through existing consolidator path | Avoid deterministic semantic conflict rules in code. |
| Global lore | Document future promotion pipeline only | Prevents scope creep while preserving long-term product direction. |

## Contracts And Data Shapes

### Scoped User-Memory Worker

Create a RAG worker module:

```text
src/kazusa_ai_chatbot/rag/user_memory_evidence_agent.py
```

Public class:

```python
class UserMemoryEvidenceAgent(BaseRAGHelperAgent):
    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]: ...
```

Input contract:

- `task`: the `Memory-evidence:` slot text.
- `context["global_user_id"]`: required current user scope.
- `context["user_name"]`: display hint only. Absence must not block retrieval.
- `context["known_facts"]`: prior RAG facts may be present. The worker must not parse free-text summaries for identity.

Output contract:

```python
{
    "resolved": bool,
    "result": {
        "selected_summary": str,
        "memory_rows": list[dict],
        "source_system": "user_memory_units",
        "scope_type": "user_continuity",
        "scope_global_user_id": str,
        "missing_context": list[str],
    },
    "attempts": 1,
    "cache": {"enabled": False, "hit": False, "reason": "scoped_user_memory_uncached"},
}
```

Each `memory_rows` entry must preserve the original memory-unit fields plus projected metadata:

```python
{
    "unit_id": str,
    "unit_type": str,
    "fact": str,
    "subjective_appraisal": str,
    "relationship_signal": str,
    "content": str,
    "updated_at": str,
    "source_system": "user_memory_units",
    "scope_type": "user_continuity",
    "scope_global_user_id": str,
    "authority": "scoped_continuity",
    "truth_status": "character_lore_or_interaction_continuity",
    "origin": "consolidated_interaction",
}
```

Failure conditions:

- Missing `global_user_id` returns unresolved with `missing_context=["global_user_id"]`.
- No matching rows returns unresolved with `missing_context=["user_memory_evidence"]`.
- Vector search unavailable must fall back to scoped lexical and recency retrieval without throwing.

### MemoryEvidenceAgent Selection

Extend internal worker roster:

```text
persistent_memory_keyword_agent
persistent_memory_search_agent
user_memory_evidence_agent
incompatible
```

Route to `user_memory_evidence_agent` when the slot asks for current-user durable memory, private continuity, accepted preference, user-specific story/lore, prior shared experience with the current user, or a topic likely stored in `user_memory_units`.

Route to persistent-memory workers for official character facts, shared world/common-sense facts, external imported facts, exact memory names, tags, and global memory identifiers.

### RAG Projection

When `memory_evidence_agent` returns `projection_payload.memory_rows`:

- Rows with `source_system="user_memory_units"` are projected into `rag_result.memory_evidence` with scope metadata preserved.
- Those rows are also appended to `rag_result.user_memory_unit_candidates`.
- Rows from shared `memory` keep existing `memory_evidence` behavior and do not become `user_memory_unit_candidates`.

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/rag/memory_evidence_agent.py`
  - Add `user_memory_evidence_agent` as an internal worker.
  - Update deterministic/selector routing and result payload projection.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_rag_projection.py`
  - Preserve scoped metadata from memory rows.
  - Append scoped user-memory rows to `user_memory_unit_candidates`.
- `src/kazusa_ai_chatbot/rag/README.md`
  - Document `Memory-evidence:` as scope-aware over shared memory and current-user continuity.
- `tests/test_rag_phase3_capability_agents.py`
  - Add scoped user-memory worker/selector tests.
- `tests/test_rag_projection.py`
  - Add projection tests for scoped user-memory evidence and candidate propagation.
- `tests/test_user_memory_units_rag_flow.py`
  - Add or update merge/evolve tests proving Memory-evidence-surfaced user rows are usable as consolidator candidates.
- `tests/test_user_memory_evidence_agent.py`
  - Required new focused unit test file for the scoped user-memory worker contract. Do not merge these worker-contract tests into an existing test file.

### Create

- `src/kazusa_ai_chatbot/rag/user_memory_evidence_agent.py`
  - Public scoped user-memory worker.
- `tests/test_user_memory_evidence_agent.py`
  - Required focused unit tests for missing scope, exact CJK term lookup, scoped isolation across two users, semantic retrieval fallback behavior, and output metadata shape.

### Keep

- Keep `Person-context:` route ownership for profile, impression, relationship, user-list, and active-character profile reads.
- Keep `Recall:` ownership for active agreements, promises, plans, and current episode state.
- Keep shared persistent-memory workers unchanged except for call sites.
- Keep database schema unchanged.

## Implementation Order

1. Add focused tests for `UserMemoryEvidenceAgent` in `tests/test_user_memory_evidence_agent.py`.
   - Expected baseline: missing module or missing worker behavior.
2. Add projection tests for scoped memory rows.
   - Expected baseline: rows appear only as flattened memory evidence or not as candidates.
3. Add capability-agent tests for `MemoryEvidenceAgent` routing to the scoped worker.
   - Expected baseline: durable user memory still uses persistent-memory search.
4. Implement `UserMemoryEvidenceAgent`.
   - Use existing `user_memory_units` retrieval helpers first.
   - Add scoped keyword helper only if vector/recency retrieval cannot satisfy exact-term cases.
5. Wire `MemoryEvidenceAgent` to the new worker.
6. Update projection to preserve metadata and feed `user_memory_unit_candidates`.
7. Update RAG documentation.
8. Run focused tests, compile checks, prompt render checks if prompt strings changed, and targeted integration tests.

## Progress Checklist

- [ ] Stage 1 - Scoped user-memory worker contract
  - Covers: `tests/test_user_memory_evidence_agent.py` and implementation for `src/kazusa_ai_chatbot/rag/user_memory_evidence_agent.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests/test_user_memory_evidence_agent.py -q`.
  - Evidence: record test output and any fallback retrieval behavior in `Execution Evidence`.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 - MemoryEvidenceAgent routing
  - Covers: internal worker registration and selector/deterministic routing.
  - Verify: `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py -q`.
  - Evidence: record which test proves scoped user memory no longer uses persistent-memory search.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 - Projection closes the loop
  - Covers: `persona_supervisor2_rag_projection.py`.
  - Verify: `venv\Scripts\python.exe -m pytest tests/test_rag_projection.py -q`.
  - Evidence: record test proving scoped rows populate both `memory_evidence` and `user_memory_unit_candidates`.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 - Consolidator merge/evolve compatibility
  - Covers: `tests/test_user_memory_units_rag_flow.py` and existing consolidator candidate path.
  - Verify: `venv\Scripts\python.exe -m pytest tests/test_user_memory_units_rag_flow.py -q`.
  - Evidence: record test proving an existing scoped memory candidate is merged/evolved instead of duplicated.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 - Documentation and final verification
  - Covers: RAG README and full targeted gates.
  - Verify: all commands in `Verification`.
  - Evidence: record command outputs and final changed-file list.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.

## Verification

### Static

- `venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\rag\memory_evidence_agent.py src\kazusa_ai_chatbot\rag\user_memory_evidence_agent.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_rag_projection.py`
- If prompt strings changed, run or add a prompt-render test covering `MemoryEvidenceAgent` selector prompt.
- `rg -n "User-memory-evidence|Continuity-memory" src tests development_plans` should return only this plan if no superseding design was approved.

### Tests

- `venv\Scripts\python.exe -m pytest tests/test_rag_phase3_capability_agents.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_rag_projection.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_user_memory_units_rag_flow.py -q`
- `venv\Scripts\python.exe -m pytest tests/test_user_memory_evidence_agent.py -q`
- `tests/test_user_memory_evidence_agent.py` is mandatory and must be the focused unit-test file for the new scoped user-memory evidence worker.

### Manual Diagnostic

Using a local/exported fixture or patched worker, verify a query equivalent to:

```text
Memory-evidence: retrieve durable evidence about the 学姐 matcha ice-cream shop in the current user's continuity
```

Expected result:

- retrieves only rows for the current `global_user_id`,
- includes `source_system="user_memory_units"`,
- includes `scope_type="user_continuity"`,
- does not call shared persistent-memory workers,
- exposes the retrieved row as a consolidation candidate.

## Acceptance Criteria

This plan is complete when:

- `Memory-evidence:` can retrieve scoped current-user continuity from `user_memory_units`.
- Shared/global memory lookup behavior remains intact.
- Scoped user-memory rows carry explicit scope/provenance metadata.
- Scoped user-memory rows populate `rag_result.user_memory_unit_candidates`.
- Consolidation can merge/evolve a retrieved scoped memory instead of creating a duplicate.
- No database migration was required.
- No new response-path LLM call was added.
- `Person-context:` and `Recall:` responsibility boundaries remain intact.
- Documentation describes the short-to-medium-term scoped-memory behavior and the long-term global lore direction as future work.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| User continuity leaks across users | Require `context.global_user_id`; all user-memory DB queries filter by it | Worker tests with two users |
| Shared memory and user memory become indistinguishable | Preserve `source_system`, `scope_type`, and authority metadata | Projection tests |
| Duplicate memories after retrieval | Append scoped rows to `user_memory_unit_candidates` | Consolidator merge/evolve tests |
| MemoryEvidenceAgent becomes a mega-agent | Add only one scoped worker; keep profile/relationship in Person-context and active agreements in Recall | Capability tests |
| Exact CJK terms are missed by vector search | Required scoped lexical helper; regex/text matching must be escaped and scoped | `学姐` fixture/keyword test |
| Local LLM prompt drift | Keep prompt change small and render-tested | Prompt contract/render test |

## LLM Call And Context Budget

- RAG initializer: unchanged, one response-path LLM call when not served from Cache2.
- RAG dispatcher: unchanged for recognized prefixes; no dispatcher LLM for `Memory-evidence:`.
- MemoryEvidenceAgent selector: unchanged call count. Deterministic routing should cover explicit scoped user-memory markers; otherwise existing selector call remains the same.
- New scoped user-memory worker: zero LLM calls. It may use an embedding call for semantic vector search, matching existing retrieval patterns, and must provide a non-LLM scoped lexical path for exact terms.
- Consolidator: unchanged background LLM call count. Existing extractor, merge judge, rewrite, and stability judge remain the only memory-unit LLM stages.
- Context cap remains 50k tokens. New projected metadata is small per row; cap row count to the same bounded memory evidence limits used by `MemoryEvidenceAgent`.

## Future Reference

Long term, the character should be able to promote interaction-derived lore into durable global memory not bound to any one user. That is not part of this immediate fix.

The future architecture should be a promotion pipeline, not a raw write:

```text
raw interaction
  -> user_continuity memory
  -> promotion candidate
  -> global lore consolidation
  -> global durable memory
  -> RAG2 retrieves with scope/authority ordering
```

Future global lore scopes should distinguish:

```text
official_seed
global_lore
user_continuity
channel_continuity
conversation evidence
```

Promotion into `global_lore` should require explicit source/audit metadata:

- `origin`
- `promotion_basis`
- `source_scope`
- `truth_status`
- `conflict_policy`
- `supersedes` or `conflicts_with` references when applicable

The long-term rule is: character-made lore may become global, but only through conflict-aware promotion with retained provenance.

## Execution Evidence

- Static grep results:
- Compile results:
- Focused test results:
- Integration test results:
- Manual diagnostic:
- Changed files:
