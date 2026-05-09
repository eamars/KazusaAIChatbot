# multi source cognition architecture stage 04 rag cognitive episode adapter plan

## Summary

- Goal: Centralize `/chat` RAG request construction in a narrow `CognitiveEpisode`-aware adapter while keeping current text `/chat` retrieval byte-equivalent.
- Plan class: large
- Status: approved
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`, `py-style`, `test-style-and-execution`. Apply `cjk-safety` only if a Python edit accidentally surfaces CJK content; this plan should not require it.
- Overall cutover strategy: bigbang inside `stage_1_research`. The inline RAG request builder is replaced by `build_text_chat_rag_request(...)` in one cutover; no dual path, no feature flag.
- Highest-risk areas: silent drift in the RAG request shape; unresolved-referent skip branch leaking into the new adapter; future episode sources entering text RAG by accident; Stage 05 unblocking before evidence is recorded.
- Acceptance criteria: adapter exists with the public contract below; non-skip `/chat` runs through it exactly once; request shape is equivalent to the pre-Stage-04 shape; unsupported sources fail closed; cognition/dialog/consolidation/RAG-internals do not consume `cognitive_episode`; all Verification gates pass; parent ledger and registry rows are flipped to `completed`.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Lifecycle: this plan is the next executable stage after Stage 03 was merged into `main` on 2026-05-09. Stage 05 is blocked on Stage 04's execution evidence.

## Context

The current `/chat` flow already carries a valid `CognitiveEpisode` on graph state (Stage 02) and a source-aware cognition prompt selector (Stage 03). RAG itself still constructs its request inline in `persona_supervisor2.stage_1_research`, reading 17 fields directly off `GlobalPersonaState`. That inline construction is now the only place where text-chat RAG could silently disagree with the episode contract used by cognition.

Why now:

- Stage 05 (consolidation origin metadata threading) needs RAG to be source-aware before consolidation can rely on episode metadata from a single owned source.
- Inline construction makes future input sources (image, audio, reflection) unreachable without forking the RAG call site.
- The Stage 02 episode now has every projection field needed to drive RAG; the missing piece is the adapter that owns that projection.

Current architecture: `stage_1_research` reads `state["decontexualized_input"]`, `state["character_profile"]`, `state["user_profile"]`, history slots, channel slots, reply slots, and three optional context dicts; it builds the RAG `context` dict inline, calls `call_rag_supervisor(...)`, then calls `project_known_facts(...)` with `current_user_id=state["global_user_id"]` and `character_user_id=state["character_profile"]["global_user_id"]`.

Target architecture: `stage_1_research` calls `build_text_chat_rag_request(episode=state["cognitive_episode"], ...)` once, then forwards `original_query`, `character_name`, and `context` to `call_rag_supervisor(...)`. `current_user_id` and `character_user_id` are read from the adapter return DTO and forwarded into `project_known_facts(...)` instead of being re-indexed off `state`.

Adjacent improvement areas intentionally left for later plans:

- Source-aware RAG retrieval policy and prompt changes (Stage 09+).
- Reflection, scheduled recall, and system probe RAG behavior (Stage 07+).
- Consolidation origin metadata threading (Stage 05).

## Stage Handoff

### From Stage 02

Stage 02 completed and was merged into `main` on 2026-05-09. The active chat workflow now carries `state["cognitive_episode"]` for `/chat` turns.

Artifacts Stage 04 must consume:

- `kazusa_ai_chatbot.cognition_episode.CognitiveEpisode`
- `kazusa_ai_chatbot.cognition_episode.project_text_chat_compatibility_fields`
- `kazusa_ai_chatbot.cognition_episode.validate_cognitive_episode`
- `GlobalPersonaState["cognitive_episode"]`
- Stage 02 chat migration tests:
  `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`

Runtime rule: even though `GlobalPersonaState.cognitive_episode` is typed as `NotRequired` for graph compatibility, the non-skip text-chat RAG path must plain-index `state["cognitive_episode"]` and fail loudly if it is missing. Do not synthesize a legacy episode and do not use `.get()` for this field.

### From Stage 03

Stage 03 completed and was merged into `main` on 2026-05-09 with `74 passed`.

Artifacts Stage 04 must preserve:

- prompt variant key: `text_chat_user_message`
- selector API: `select_cognition_prompt_variant(episode=..., stage=...)`
- output-contract API: `validate_cognition_output_contract(stage=..., payload=...)`
- cognition prompt payloads do not receive `cognitive_episode`, `prompt_key`, `trigger_source`, or `input_sources`

Stage 04 must not edit:

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_prompt_selection.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_output_contracts.py`
- `tests/test_multi_source_cognition_stage_03_prompt_selection.py`

Stage 04 deliberately does not consume the cognition prompt selector. RAG and cognition select their inputs independently from the same episode; coupling them now would create a cross-cutting boundary the parent plan defers to later stages.

### To Stage 05

After Stage 04, Stage 05 can rely on these facts:

- RAG request construction for `/chat` is centralized in `rag/cognitive_episode_adapter.py`.
- `stage_1_research` still returns the same `rag_result` shape.
- RAG supervisor internals still do not consume `CognitiveEpisode` directly.
- Consolidation still receives only the existing global-state and `rag_result` fields; origin metadata threading is deferred to Stage 05.
- The parent ledger row for `stage_04` is `completed` and the registry in `development_plans/README.md` is updated.

## Mandatory Skills

- `development-plan-writing`: preserve parent-stage scope and lifecycle.
- `local-llm-architecture`: keep RAG input shape stable for the local LLM agents.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `cjk-safety`: only if a Python file with CJK strings is edited; this plan should not trigger that.

## Mandatory Rules

- Execute only from a feature branch forked from current `main`.
- Keep edits inside the approved Change Surface.
- Do not change RAG retrieval policy, RAG prompts, RAG result projection, or `call_rag_supervisor(...)` signature.
- Do not change cognition prompt selection, cognition prompt payloads, dialog, consolidation, memory write policy, reflection, scheduler behavior, or adapter delivery.
- Do not add feature flags, fallback legacy builders, alternate call sites, or compatibility shims.
- Do not add private raising-only helpers or pass-through wrappers. Private helpers are allowed only for repeated structural validation or local table lookup.
- `stage_1_research` must keep the unresolved-referent skip decision before any RAG adapter call.
- The unresolved-referent skip branch must not read `state["cognitive_episode"]` and must not call the new adapter.
- The non-skip branch must build the RAG request by calling `build_text_chat_rag_request(...)` exactly once.
- The non-skip branch must pass `state["cognitive_episode"]` by direct indexing.
- Unsupported future episode sources must fail closed in the adapter. Do not silently downgrade them to the current text-chat shape.
- Prompt/RAG/consolidation internals must not receive or inspect raw `CognitiveEpisode` objects in this stage.
- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage and confirm no stale assumption has appeared.

## Must Do

- Create a dedicated RAG adapter module for user-message text-chat episodes.
- Replace only the inline RAG request/context construction in `stage_1_research` with a call to that adapter.
- Read `current_user_id` and `character_user_id` for `project_known_facts(...)` from the adapter return DTO instead of re-indexing `state`.
- Preserve all current context keys and values for normal `/chat`.
- Add focused adapter tests for exact projection, invalid sources, and missing required fields.
- Add an integration snapshot test that captures the pre-Stage-04 RAG request shape and asserts the post-cutover call equals it.
- Update direct test fixtures that invoke Stage 03+ graph nodes so they include a valid Stage 01/02 `CognitiveEpisode`.
- Add static checks that prove no prompt, dialog, consolidation, or RAG-internal code consumes `cognitive_episode` in this stage.
- Update the parent ledger row and `development_plans/README.md` registry to mark Stage 04 `completed` after verification passes.

## Deferred

- Multi-source RAG retrieval policy.
- RAG prompt changes.
- Image/audio/internal-thought retrieval behavior.
- Reflection, scheduled recall, and system probe RAG behavior.
- Consolidation origin metadata threading.
- Per-write memory policy changes.
- Dialog visibility or output-mode changes.

## Cutover Policy

Overall strategy: `bigbang` within `stage_1_research`.

| Area | Policy | Instruction |
|---|---|---|
| `stage_1_research` non-skip RAG request construction | bigbang | Replace inline builder with one call to `build_text_chat_rag_request(...)`. No dual path, no feature flag. |
| `stage_1_research` unresolved-referent skip branch | compatible | Preserve current skip behavior byte-for-byte. Do not read `state["cognitive_episode"]` and do not call the adapter. |
| `call_rag_supervisor(...)` public arguments | compatible | Keep `original_query`, `character_name`, `context`. The adapter feeds these from its return DTO. |
| `project_known_facts(...)` arguments | compatible | Continue passing `current_user_id` and `character_user_id`; source them from the adapter return DTO instead of re-indexing `state`. |
| RAG internal modules | compatible | RAG internals still do not consume `CognitiveEpisode`. |

Rollback path: revert the `stage_1_research` cutover to its pre-Stage-04 inline builder, delete `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`, delete `tests/test_rag_cognitive_episode_adapter.py`, and revert the focused-test fixture additions in the modified `tests/test_persona_supervisor2_rag*.py` and `tests/test_rag_projection.py`. No database rollback is required.

Stop condition: if the post-cutover request shape ever diverges from the captured pre-Stage-04 snapshot under the same inputs, stop Stage 04 and create a bugfix plan.

## Agent Autonomy Boundaries

Allowed implementation choices:

- local variable names inside the new adapter and tests
- focused test helper names
- exact assertion grouping, as long as every contract below is covered

Not allowed:

- adding wrapper layers around the adapter
- moving RAG supervisor code
- moving cognition selector code
- adding fallback behavior for missing `cognitive_episode`
- adding deterministic semantic classification of user input
- adding support for non-`user_message` triggers
- adding support for non-`["dialog_text"]` input source sets
- rewriting broad test files for style

If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

`persona_supervisor2.stage_1_research` remains the only production caller of `call_rag_supervisor(...)`.

The non-skip call shape becomes:

```python
rag_request = build_text_chat_rag_request(
    episode=state["cognitive_episode"],
    decontexualized_input=state["decontexualized_input"],
    character_profile=state["character_profile"],
    user_profile=state["user_profile"],
    prompt_message_context=state["prompt_message_context"],
    channel_topic=state["channel_topic"],
    chat_history_recent=state["chat_history_recent"],
    chat_history_wide=state["chat_history_wide"],
    reply_context=state["reply_context"],
    indirect_speech_context=state["indirect_speech_context"],
    conversation_progress=state.get("conversation_progress"),
    conversation_episode_state=state.get("conversation_episode_state"),
    promoted_reflection_context=state.get("promoted_reflection_context"),
)

rag_response = await call_rag_supervisor(
    original_query=rag_request["original_query"],
    character_name=rag_request["character_name"],
    context=rag_request["context"],
)

rag_result = project_known_facts(
    rag_response["known_facts"],
    current_user_id=rag_request["current_user_id"],
    character_user_id=rag_request["character_user_id"],
    answer=str(rag_response["answer"]),
    unknown_slots=rag_response["unknown_slots"],
    loop_count=int(rag_response["loop_count"] or 0),
)
```

The optional `.get(...)` calls above are allowed only for the three existing optional context fields. Do not use `.get(...)` for required episode, profile, history, message-context, channel, reply, or user fields. The return shape from `project_known_facts(...)` is unchanged.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Adapter ownership | New module under `src/kazusa_ai_chatbot/rag/`, not `nodes/`. | RAG inputs are RAG concerns; the parent plan's "Shared Components" boundary places query/context construction inside RAG. |
| Public contract surface | `RAGEpisodeRequest` TypedDict, `build_text_chat_rag_request(...)` function, and `RAGEpisodeAdapterError`. | Mirrors Stage 03's selector contract style: typed payload, named error, named function. |
| `current_user_id` / `character_user_id` on `RAGEpisodeRequest` | Returned from the adapter even though `call_rag_supervisor(...)` does not consume them. | They are consumed by `project_known_facts(...)` immediately after the RAG call. Returning them from the adapter keeps `stage_1_research` from re-indexing `state` and `character_profile` for primitives the adapter has already pulled from the episode/profile. |
| `character_profile` validation | Validate `global_user_id` and `name` are non-empty strings. | These are the only `character_profile` values the RAG context exposes; they are required by the existing inline builder. |
| `user_profile` validation | Pass through unchanged with no structural validation. | The episode projection already supplies `global_user_id` and `user_name`; `user_profile` is an opaque dict consumed by RAG agents downstream. |
| Selector dependency | Adapter does not consume `select_cognition_prompt_variant(...)` output. | RAG and cognition select inputs from the same episode independently; coupling them adds a cross-cutting boundary the parent plan defers. |
| Skip branch handling | Skip branch does not call the adapter and does not read `cognitive_episode`. | Stage 02 keeps `cognitive_episode` available in the skip branch, but unresolved-referent behavior must not depend on episode validation. |
| Forbidden context keys | `cognitive_episode`, `message_envelope`, `episode_focus`, `trigger_source`, `input_sources`, `percepts`, raw attachments, raw percept payloads, consolidation write-policy fields. | These belong to later stages or to the episode internals; leaking them here would break the parent plan's source-aware adapter contract. |
| Rejection ordering | `validate_cognitive_episode` → `trigger_source` check → `input_sources` check → `project_text_chat_compatibility_fields(...)` → `character_profile` checks. | Deterministic order keeps `RAGEpisodeAdapterError` messages stable across fixtures. |
| Equivalence baseline | Pre-Stage-04 request shape is captured by a snapshot test in Step 1 and locked by Step 4. | A single canonical baseline is more reliable than re-deriving expected shapes inside each integration test. |

## Adapter Contract

Create:

`src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`

Public names:

```python
from typing import Any, TypedDict

from kazusa_ai_chatbot.cognition_episode import CognitiveEpisode


class RAGEpisodeAdapterError(ValueError):
    """Raised when an episode cannot be projected into the current RAG path."""


class RAGEpisodeRequest(TypedDict):
    original_query: str
    character_name: str
    context: dict[str, Any]
    current_user_id: str
    character_user_id: str
```

Public function:

```python
def build_text_chat_rag_request(
    *,
    episode: CognitiveEpisode,
    decontexualized_input: str,
    character_profile: dict[str, Any],
    user_profile: dict[str, Any],
    prompt_message_context: dict[str, Any],
    channel_topic: str,
    chat_history_recent: list[dict[str, Any]],
    chat_history_wide: list[dict[str, Any]],
    reply_context: dict[str, Any],
    indirect_speech_context: str,
    conversation_progress: dict[str, Any] | None = None,
    conversation_episode_state: dict[str, Any] | None = None,
    promoted_reflection_context: dict[str, Any] | None = None,
) -> RAGEpisodeRequest:
    ...
```

Required behavior order:

1. Call `validate_cognitive_episode(episode)`.
2. If `episode["trigger_source"] != "user_message"`, raise `RAGEpisodeAdapterError`.
3. If `episode["input_sources"] != ["dialog_text"]`, raise `RAGEpisodeAdapterError`.
4. Call `project_text_chat_compatibility_fields(episode)`.
5. Require `character_profile["global_user_id"]` to be a non-empty string.
6. Require `character_profile["name"]` to be a non-empty string.
7. Build and return the exact request shape below.

Do not add a private helper whose only job is to raise the exception. Direct `raise RAGEpisodeAdapterError(...)` statements are preferred here.

## Exact Request Shape

Return value:

```python
{
    "original_query": decontexualized_input,
    "character_name": character_profile["name"],
    "context": {
        "platform": projection["platform"],
        "platform_channel_id": projection["platform_channel_id"],
        "channel_type": projection["channel_type"],
        "character_profile": {
            "global_user_id": character_profile["global_user_id"],
            "name": character_profile["name"],
        },
        "active_turn_platform_message_ids": projection[
            "active_turn_platform_message_ids"
        ],
        "active_turn_conversation_row_ids": projection[
            "active_turn_conversation_row_ids"
        ],
        "global_user_id": projection["global_user_id"],
        "user_name": projection["user_name"],
        "user_profile": user_profile,
        "current_timestamp": projection["timestamp"],
        "time_context": projection["time_context"],
        "prompt_message_context": prompt_message_context,
        "channel_topic": channel_topic,
        "chat_history_recent": chat_history_recent,
        "chat_history_wide": chat_history_wide,
        "reply_context": reply_context,
        "indirect_speech_context": indirect_speech_context,
        "conversation_progress": conversation_progress,
        "conversation_episode_state": conversation_episode_state,
        "promoted_reflection_context": promoted_reflection_context,
    },
    "current_user_id": projection["global_user_id"],
    "character_user_id": character_profile["global_user_id"],
}
```

The top-level request dictionary and nested `character_profile` dictionary must be new dictionaries. The projected active-turn id lists may use the lists returned by `project_text_chat_compatibility_fields(...)`; that function already copies them from the episode. Do not deep-copy broad context objects.

Forbidden context keys:

- `cognitive_episode`
- `message_envelope`
- `episode_focus`
- `trigger_source`
- `input_sources`
- `percepts`
- raw attachments or raw percept payloads
- consolidation write-policy fields

## Change Surface

### Create

- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py` — public adapter module owning text-chat RAG request construction.
- `tests/test_rag_cognitive_episode_adapter.py` — focused module tests for projection, rejection, and required-field validation.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py` — replace inline RAG request construction in `stage_1_research` with the adapter call; switch `project_known_facts(...)` user-id arguments to read from the adapter return DTO. Skip branch unchanged.
- `src/kazusa_ai_chatbot/rag/README.md` — add a short note naming the adapter, its public entrypoint, and its boundary. No architectural speculation.
- `tests/test_persona_supervisor2_rag2_integration.py` — update fixtures to carry a valid Stage 02 `CognitiveEpisode`, add the pre-Stage-04 request-shape snapshot test, and add the post-cutover equivalence assertion.
- `tests/test_persona_supervisor2_rag_skip_shape.py` — update fixtures to carry a valid Stage 02 `CognitiveEpisode` and assert the skip branch still does not call the adapter and does not read `cognitive_episode`.
- `tests/test_rag_projection.py` — update fixtures only where direct calls into the post-Stage-02 path now require `cognitive_episode`.
- `development_plans/active/short_term/multi_source_cognition_architecture_stage_04_rag_cognitive_episode_adapter_plan.md` — update the checklist and `Execution Evidence` only during execution.
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md` — flip the `stage_04` ledger row to `completed` after verification passes.
- `development_plans/README.md` — flip the Stage 04 registry row from `approved | not_started` to `completed | completed` after verification passes.

### Keep

- `src/kazusa_ai_chatbot/rag/rag_graph.py`
- `src/kazusa_ai_chatbot/rag/rag2.py`
- `src/kazusa_ai_chatbot/rag/rag2_*.py`
- `src/kazusa_ai_chatbot/rag/rag_supervisor_state.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l1.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition_l3.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator*.py`
- `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
- `tests/test_multi_source_cognition_stage_01*.py`
- `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`
- `tests/test_multi_source_cognition_stage_03_prompt_selection.py`

Do not modify kept files unless a verification failure proves the plan is incomplete. If that happens, stop and update this plan before continuing.

## LLM Call And Context Budget

No new live `/chat` LLM calls. No background LLM calls. The adapter is pure deterministic Python and must not call an LLM, RAG agent, database, cache, scheduler, or adapter.

Before/after for one normal `/chat` turn:

| Area | Before | After |
|---|---:|---:|
| `call_rag_supervisor(...)` invocations per turn | 1 | 1 |
| RAG `context` dict keys | unchanged | unchanged |
| RAG `context` dict values for `/chat` | unchanged | unchanged |
| `project_known_facts(...)` invocations per turn | 1 | 1 |
| `project_known_facts(...)` argument values for `/chat` | unchanged | unchanged |
| Live response-path token cost | unchanged | unchanged |

Verification: the snapshot test in `tests/test_persona_supervisor2_rag2_integration.py` captures the pre-Stage-04 request shape and asserts the post-cutover shape equals it.

## Implementation Order

Module-first, test-first:

1. Capture the pre-Stage-04 RAG request shape as a fixture in the integration test (red).
2. Write focused adapter tests against the contract above (red).
3. Implement the adapter module (green for adapter tests).
4. Cut `stage_1_research` over to the adapter (green for integration tests).
5. Update direct test fixtures that now require a valid `cognitive_episode`.
6. Update `rag/README.md` with one short adapter-boundary note.
7. Run the full Verification section.
8. Flip lifecycle records (parent ledger and registry) only after verification completes.
9. Record execution evidence.

Build the adapter first because the integration cutover depends on its public contract; do not edit `stage_1_research` before the adapter and its focused tests pass.

## Granular Execution Steps

Each step has one action, named files/symbols, expected evidence, and a TDD triplet where code behavior changes.

### Step 1 — Capture pre-Stage-04 request shape (red, then locked)

- File: `tests/test_persona_supervisor2_rag2_integration.py`
- Action: add `test_stage_1_research_pre_stage_04_request_shape_snapshot` that invokes the current `stage_1_research` with a mocked `call_rag_supervisor`, captures the `original_query`, `character_name`, and `context` arguments, and asserts equality against an in-test `EXPECTED_REQUEST` constant. Run before Step 4 to lock the baseline; this test must continue to pass after the cutover.
- Verify: `venv\Scripts\python -m pytest tests\test_persona_supervisor2_rag2_integration.py::test_stage_1_research_pre_stage_04_request_shape_snapshot -q`
- Expected: passes against current `main`.
- Evidence: snapshot fingerprint (sorted-keys hash of the captured `context`) recorded in `Execution Evidence`.

### Step 2 — Add focused adapter contract tests (red)

- File: `tests/test_rag_cognitive_episode_adapter.py`
- Action: add tests covering: (a) valid text-chat episode produces the exact request shape from `Exact Request Shape`; (b) `trigger_source != "user_message"` raises `RAGEpisodeAdapterError`; (c) `input_sources != ["dialog_text"]` raises `RAGEpisodeAdapterError`; (d) missing or empty `character_profile["global_user_id"]` raises; (e) missing or empty `character_profile["name"]` raises; (f) optional fields default to `None`; (g) forbidden context keys are absent from the returned `context`.
- Verify: `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py -q`
- Expected: collection error or `ModuleNotFoundError` because `kazusa_ai_chatbot.rag.cognitive_episode_adapter` does not yet exist.
- Evidence: command and expected failure recorded.

### Step 3 — Implement the adapter module (green for Step 2)

- File: `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
- Action: implement `RAGEpisodeAdapterError`, `RAGEpisodeRequest`, and `build_text_chat_rag_request(...)` exactly per `Adapter Contract` and `Exact Request Shape`. No private raising-only helpers.
- Verify:
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py`
  - `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py -q`
- Expected: compile passes; all focused tests pass.
- Evidence: test count and pass output recorded.

### Step 4 — Cut `stage_1_research` over to the adapter (green for Step 1)

- File: `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- Action: in the non-skip branch of `stage_1_research`, replace the inline `call_rag_supervisor(...)` `context` construction with one `build_text_chat_rag_request(...)` call (per `Target State`); pass `original_query`, `character_name`, `context` to `call_rag_supervisor`; pass `current_user_id` and `character_user_id` from the adapter return DTO into `project_known_facts(...)`. Do not touch the skip branch.
- Verify:
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\nodes\persona_supervisor2.py`
  - `venv\Scripts\python -m pytest tests\test_persona_supervisor2_rag2_integration.py -q`
- Expected: snapshot equality from Step 1 still passes after the cutover; existing integration tests pass.
- Evidence: command output recorded.

### Step 5 — Add skip-branch independence assertion

- File: `tests/test_persona_supervisor2_rag_skip_shape.py`
- Action: add `test_skip_branch_does_not_call_adapter` that constructs an unresolved-referent state without `cognitive_episode` populated and asserts `stage_1_research` returns the existing skip `rag_result` shape without raising and without invoking the adapter (use a sentinel patch on `build_text_chat_rag_request`).
- Verify: `venv\Scripts\python -m pytest tests\test_persona_supervisor2_rag_skip_shape.py -q`
- Expected: passes.
- Evidence: command output recorded.

### Step 6 — Update direct fixtures in `tests/test_rag_projection.py`

- File: `tests/test_rag_projection.py`
- Action: only where a test directly drives the post-Stage-02 path that now requires `cognitive_episode`, add a Stage 01 `build_text_chat_cognitive_episode(...)` fixture. Do not add fixtures to tests that do not need them.
- Verify: `venv\Scripts\python -m pytest tests\test_rag_projection.py -q`
- Expected: passes.
- Evidence: command output recorded.

### Step 7 — Update `rag/README.md`

- File: `src/kazusa_ai_chatbot/rag/README.md`
- Action: add a short subsection naming `cognitive_episode_adapter.py`, its public entrypoint `build_text_chat_rag_request(...)`, and its boundary ("RAG owns request construction; does not consume raw `CognitiveEpisode` outside this adapter"). No architectural speculation.
- Verify: `git diff -- src\kazusa_ai_chatbot\rag\README.md`
- Expected: small additive diff.
- Evidence: diff stat.

### Step 8 — Run full Verification

- Action: run every command in `Verification`.
- Expected: all pass per the per-command expectations.
- Evidence: every command and result recorded in `Execution Evidence`.

### Step 9 — Flip lifecycle records

- Files: `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`, `development_plans/README.md`.
- Action: only after Step 8 is recorded, mark the parent ledger row for `stage_04` `completed` and flip the registry row to `completed | completed`.
- Verify: re-read both files and confirm the rows match Stage 04 completion.
- Evidence: confirmation lines recorded.

## Progress Checklist

- [ ] Stage 1 — pre-cutover request snapshot captured.
  - Covers: Step 1.
  - Files: `tests/test_persona_supervisor2_rag2_integration.py`.
  - Verify: focused snapshot test passes against current `main`.
  - Evidence: snapshot fingerprint recorded in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 2 — adapter contract test added (red) and adapter module implemented (green).
  - Covers: Steps 2-3.
  - Files: `tests/test_rag_cognitive_episode_adapter.py`, `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`.
  - Verify: focused adapter tests pass; `py_compile` passes.
  - Evidence: red expected failure and green pass count recorded.
  - Handoff: reread this plan, then start Stage 3.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 3 — `stage_1_research` cut over.
  - Covers: Step 4.
  - Files: `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`.
  - Verify: integration snapshot equality passes; rag2 integration suite passes.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 4.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 4 — skip-branch independence and direct fixtures updated.
  - Covers: Steps 5-6.
  - Files: `tests/test_persona_supervisor2_rag_skip_shape.py`, `tests/test_rag_projection.py`.
  - Verify: focused tests pass.
  - Evidence: command output recorded.
  - Handoff: reread this plan, then start Stage 5.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 5 — README updated and full Verification executed.
  - Covers: Steps 7-8.
  - Files: `src/kazusa_ai_chatbot/rag/README.md`.
  - Verify: every command in `Verification` passes per its expected result.
  - Evidence: each command output recorded.
  - Handoff: next agent starts at Stage 6.
  - Sign-off: `<agent/date>` after verification and evidence are recorded.
- [ ] Stage 6 — lifecycle records flipped.
  - Covers: Step 9.
  - Files: parent ledger and `development_plans/README.md`.
  - Verify: parent-ledger row and registry row both show `completed`.
  - Evidence: row-text confirmation recorded.
  - Handoff: Stage 05 may now begin; it must read this plan's Execution Evidence first.
  - Sign-off: `<agent/date>` after lifecycle updates are recorded.

## Verification

Run from repository root with the project virtual environment.

### Static Compile

- `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py tests\test_rag_cognitive_episode_adapter.py tests\test_persona_supervisor2_rag2_integration.py tests\test_persona_supervisor2_rag_skip_shape.py tests\test_rag_projection.py`

### Static Greps

- `rg -n "cognitive_episode|RAGEpisode|build_text_chat_rag_request|trigger_source|input_sources" src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l1.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition_l3.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_dialog.py`

  Expected result: no matches. `rg` exit code `1` is acceptable and means this check passed. Any match is a blocker unless this plan is updated and reapproved.

- `rg -n "reflection_signal|internal_thought|scheduled_recall|system_probe|image_observation|audio_observation|reflection_artifact|retrieved_memory" src\kazusa_ai_chatbot\rag src\kazusa_ai_chatbot\nodes\persona_supervisor2.py`

  Expected result: no runtime matches. `rg` exit code `1` is acceptable and preferred. Test files and this plan are not part of this command. If matches appear in production RAG/runtime code, they must be only in explicit rejection test data or error-message constants; otherwise stop and update the plan.

- `git diff --check`

  Expected result: exit code `0` and no whitespace errors.

### Focused Tests

- `venv\Scripts\python -m pytest tests\test_rag_cognitive_episode_adapter.py`
- `venv\Scripts\python -m pytest tests\test_persona_supervisor2_rag2_integration.py tests\test_persona_supervisor2_rag_skip_shape.py tests\test_rag_projection.py`

### Prior Stage Regression Gates

- `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_02_chat_episode_migration.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_03_prompt_selection.py tests\test_cognition_clarification_consumers.py tests\test_cognition_interaction_style_context.py`
- `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`

### Completion Review

Before merging, inspect:

- `git diff --stat`
- `git diff -- src\kazusa_ai_chatbot\nodes\persona_supervisor2.py`
- `git diff -- src\kazusa_ai_chatbot\rag\cognitive_episode_adapter.py`

The diff must show only the adapter cutover and tests/docs inside the approved surface.

## Acceptance Criteria

Stage 04 is complete when:

- the adapter exists with the public contract above;
- non-skip `stage_1_research` uses the adapter exactly once;
- `project_known_facts(...)` reads `current_user_id` and `character_user_id` from the adapter return DTO;
- unresolved-referent skip behavior is unchanged and does not require `cognitive_episode`;
- normal `/chat` RAG request shape is equivalent to the pre-Stage-04 snapshot;
- unsupported non-text-chat episode sources fail closed;
- RAG internals, prompts, cognition, dialog, and consolidation do not consume `cognitive_episode`;
- all Verification commands pass or have the explicitly allowed no-match exit code;
- the parent ledger row and registry row are `completed`.

## Plan Self-Review

Performed before approval on 2026-05-09:

- **Coverage:** every `Must Do` item maps to at least one step in `Granular Execution Steps`; every Acceptance Criterion has a verification command in `Verification`; every Design Decision is enforced either by the `Adapter Contract` or by a focused test in Step 2.
- **Placeholder scan:** no `TBD`, `TODO`, "similar to", "handle edge cases", "add tests", or open-ended implementation wording. Optional `.get(...)` calls are restricted by name to the three Stage 02 optional context fields.
- **Contract consistency:** module path, function name, TypedDict name, error-class name, test file paths, registry/ledger paths match across `Adapter Contract`, `Exact Request Shape`, `Change Surface`, `Granular Execution Steps`, `Progress Checklist`, and `Verification`.
- **Granularity:** every Granular Execution Step has one action, one named target, and one expected result. No step hides multiple unrelated edits.
- **Verification:** each behavior change is covered by a focused or integration test; the request-shape behavior is locked by the snapshot equality test in Steps 1 and 4.

## Execution Handoff

Intended execution mode: sequential handoff to a single implementation agent on a feature branch forked from current `main`.

Next unchecked stage: `Stage 1 — pre-cutover request snapshot captured`.

Required skills before editing: `development-plan-writing`, `local-llm-architecture`, `py-style`, `test-style-and-execution`.

Files expected to change next: `tests/test_persona_supervisor2_rag2_integration.py` (Step 1).

Verification before sign-off of the next stage: `venv\Scripts\python -m pytest tests\test_persona_supervisor2_rag2_integration.py::test_stage_1_research_pre_stage_04_request_shape_snapshot -q`.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| RAG request shape drifts during adapter extraction | Exact request-shape contract and integration snapshot test | Step 1 snapshot + Step 4 equality |
| Future episode source accidentally enters current text RAG | Closed allow-list on `trigger_source` and `input_sources` | Adapter negative tests in Step 2 + static grep |
| Skip path starts requiring episode state | Keep skip decision before adapter call | Step 5 skip-shape test |
| Prompt or consolidation starts consuming episode internals too early | Forbidden context keys and static grep | Static non-consumption check in Verification |
| Direct tests bypass Stage 02 fixtures | Update direct fixtures in Step 6 only where required | Step 6 focused test pass |
| Lifecycle records left stale | Step 9 flips parent ledger and registry only after Verification | Stage 6 progress checkpoint |

## Completion Artifact Contract

When Stage 04 is complete, these artifacts must exist or be updated:

- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
- `tests/test_rag_cognitive_episode_adapter.py`
- RAG integration tests proving pre/post request-shape equivalence
- `src/kazusa_ai_chatbot/rag/README.md` with a short adapter-boundary note
- parent ledger row for `stage_04` flipped to `completed`
- `development_plans/README.md` Stage 04 row flipped to `completed | completed`
- execution evidence in this plan naming the branch, commit, static checks, and test commands

The completion artifact must not include prompt changes, consolidation changes, dialog changes, RAG retrieval-policy changes, or persistence changes.

## Execution Evidence

Record after implementation:

- Branch:
- Commit:
- Pre-Stage-04 request snapshot fingerprint:
- Static compile:
- Static greps:
- Focused tests:
- Prior stage regression gates:
- Completion diff review:
- Lifecycle records flipped:
- Sign-off:
