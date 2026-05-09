# multi source cognition architecture stage 01 cognitive episode contract plan

## Summary

- Goal: Define the neutral `CognitiveEpisode` contract and text-only `/chat`
  construction helpers without changing runtime behavior.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan-writing`, `local-llm-architecture`,
  `no-prepost-user-input`, `py-style`, `test-style-and-execution`, and
  `cjk-safety` if tests add CJK strings.
- Overall cutover strategy: additive contract module and tests only; no graph,
  prompt, RAG, dialog, persistence, or consolidation path consumes the contract
  yet.
- Highest-risk areas: circular imports, overfitting the contract to `/chat`,
  leaking implementation-specific storage fields into LLM-facing concepts, and
  adding semantic keyword interpretation in code.
- Acceptance criteria: typed contracts, validators, text-only builder tests,
  and no runtime behavior changes.

Parent plan:
`development_plans/active/short_term/multi_source_cognition_architecture_plan.md`

Stage: `stage_01`

## Context

The parent architecture needs a source-neutral episode shape before `/chat` can
be migrated safely. This stage defines that shape but does not wire it into the
live graph. The current `/chat` workflow remains the behavior baseline from
`stage_00`.

## Mandatory Skills

- `development-plan-writing`: preserve parent-stage alignment.
- `local-llm-architecture`: keep the contract semantic and compact for a weak
  local model.
- `no-prepost-user-input`: avoid deterministic code deciding user intent,
  permissions, commitments, or relationship meaning.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding or running tests.
- `cjk-safety`: load before editing Python files that contain CJK strings.

## Mandatory Rules

- Before editing, read the parent ledger and confirm `stage_00` is completed.
- Before editing, read `stage_00` execution evidence and confirm the baseline
  test and fixture artifact paths exist.
- After any automatic context compaction, the active agent must reread this
  entire plan before continuing implementation, verification, handoff, or final
  reporting.
- After signing off any major progress checklist stage, the active agent must
  reread this entire plan before starting the next stage.
- Do not wire `CognitiveEpisode` into the production graph in this stage.
- Do not change prompts.
- Do not change RAG input construction.
- Do not change dialog or consolidation behavior.
- Do not change task allocation, schedulers, adapter delivery, service startup,
  or the current character runtime behavior. The character must continue using
  the existing `/chat` path until a later approved migration stage wires the
  contract in.
- Do not add database schema changes, migrations, new persistence writes, new
  LLM calls, or prompt context changes.
- Do not import `state.py` from the new contract module if that creates a
  circular import.
- Deterministic validators may validate structure, not semantic user intent.
  They must not infer, filter, classify, or rewrite user commands,
  preferences, permissions, commitments, or relationship meaning.
- Validation must fail fast by raising `CognitiveEpisodeValidationError`, a
  narrow subclass of `ValueError`.

## Must Do

- Add a new source-neutral contract module.
- Define trigger, input-source, visibility, output-mode, target-scope, percept,
  origin metadata, and episode shapes.
- Define the exact public API listed in `Contract Sketch`.
- Add small structural validation helpers.
- Add a text-only `/chat` episode builder that takes primitive field arguments,
  not a full `IMProcessState` dependency.
- Add a compatibility projection helper for the current text `/chat` primitive
  fields, but do not call it from production code in this stage.
- Add tests for valid text-only `/chat` episodes and invalid structural cases.

## Deferred

- Adding `cognitive_episode` to `IMProcessState`.
- Passing episodes through `persona_supervisor2`.
- Source-aware prompt selection.
- Reflection, internal thought, image, audio, or proactive trigger support.
- Production use of the compatibility projection helper.

## Cutover Policy

There is no runtime cutover. The module is added but remains unused by the live
graph until `stage_02`.

## Agent Autonomy Boundaries

The implementation agent must use the public names and signatures in
`Contract Sketch`. Local private helper names are allowed only inside
`src/kazusa_ai_chatbot/cognition_episode.py`.

The agent must not add runtime graph wiring, prompt edits, new source types,
alternate migration strategies, compatibility shims outside the approved helper,
fallback paths, dependency upgrades, unrelated cleanup, or broad refactors. If
the plan and code disagree, preserve this plan's stated intent and report the
discrepancy.

## Target State

The codebase has a stable internal episode contract that can represent the
current `/chat` text turn as:

```text
trigger_source=user_message
input_sources=[dialog_text]
output_mode=visible_reply
```

The contract is compact and semantic. It does not expose database internals to
the model-facing layer.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Contract style | Use `TypedDict` plus `Literal` aliases. | Matches existing internal state and message-envelope style. |
| Validation failure | Raise `CognitiveEpisodeValidationError(ValueError)`. | Fail-fast structural validation is simpler for deterministic tests and does not require agents to invent error schemas. |
| Builder inputs | Use primitive keyword arguments only. | Avoids circular imports between `state.py` and the new module. |
| Compatibility projection | Implement a separate text `/chat` projection helper, but leave it unused by production code. | Keeps Stage 01 allocation intact while preserving uninterrupted runtime behavior. |
| Semantic judgment | Keep all validators structural. | LLM stages own user-intent, permission, preference, commitment, and relationship interpretation. |

## Change Surface

Target ownership boundary: the new source-neutral cognition episode contract.

### Create

- Add `src/kazusa_ai_chatbot/cognition_episode.py`.
- Add `tests/test_cognitive_episode_contract.py`.

### Modify During Execution

- `development_plans/active/short_term/multi_source_cognition_architecture_stage_01_cognitive_episode_contract_plan.md`:
  update checklist and `Execution Evidence` only.
- `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`:
  update the `stage_01` ledger row to `completed` only after verification
  passes.
- `development_plans/README.md`: update the Stage 01 registry row only after
  completion. Both `Status` and `Execution` columns must move from
  `approved` to `completed` in the same edit; do not leave the row half-updated.

### Keep Unchanged / Forbidden

- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/state.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_schema.py`
- prompt source files
- RAG source files
- consolidator source files
- adapter, scheduler, persistence, and runtime startup files

## Contract Sketch

`TimeContextDoc` must be imported from `kazusa_ai_chatbot.time_context`. Do not
redeclare or alias it inside `cognition_episode.py`.

The new module file is `cognition_episode.py` (singular noun). Class and
identifier names use the `Cognitive*` prefix (`CognitiveEpisode`,
`CognitivePercept`, `CognitiveEpisodeValidationError`). Do not rename the file
to `cognitive_episode.py` and do not rename the symbols to drop the `Cognitive`
prefix.

The implementation must define these concepts:

```python
TriggerSource = Literal[
    "user_message",
    "reflection_signal",
    "internal_thought",
    "scheduled_recall",
    "system_probe",
]

InputSource = Literal[
    "dialog_text",
    "image_observation",
    "audio_observation",
    "internal_monologue",
    "reflection_artifact",
    "retrieved_memory",
]

Visibility = Literal[
    "model_visible",
    "internal_only",
    "audit_only",
]

OutputMode = Literal[
    "visible_reply",
    "silent",
    "think_only",
    "preview",
    "scheduled_action_request",
]
```

The implementation must define these public `TypedDict` shapes:

```python
class CognitivePercept(TypedDict):
    percept_id: str
    input_source: InputSource
    content: str
    visibility: Visibility
    metadata: dict[str, Any]


class TargetScope(TypedDict):
    platform: str
    platform_channel_id: str
    channel_type: str
    current_platform_user_id: str
    current_global_user_id: str
    current_display_name: str
    target_addressed_user_ids: list[str]
    target_broadcast: bool


class OriginMetadata(TypedDict):
    platform: str
    platform_message_id: str
    active_turn_platform_message_ids: list[str]
    active_turn_conversation_row_ids: list[str]
    debug_modes: dict[str, bool]


class CognitiveEpisode(TypedDict):
    episode_id: str
    trigger_source: TriggerSource
    input_sources: list[InputSource]
    output_mode: OutputMode
    percepts: list[CognitivePercept]
    target_scope: TargetScope
    origin_metadata: OriginMetadata
    timestamp: str
    time_context: TimeContextDoc


class TextChatCompatibilityProjection(TypedDict):
    timestamp: str
    time_context: TimeContextDoc
    user_input: str
    platform: str
    platform_channel_id: str
    channel_type: str
    platform_message_id: str
    active_turn_platform_message_ids: list[str]
    active_turn_conversation_row_ids: list[str]
    platform_user_id: str
    global_user_id: str
    user_name: str
```

The module must expose these public helpers:

```python
class CognitiveEpisodeValidationError(ValueError):
    """Raised when a cognitive episode is structurally invalid."""


def validate_cognitive_episode(episode: CognitiveEpisode) -> None:
    """Validate structure only; raise CognitiveEpisodeValidationError on error."""


def build_text_chat_cognitive_episode(
    *,
    episode_id: str,
    percept_id: str,
    timestamp: str,
    time_context: TimeContextDoc,
    user_input: str,
    platform: str,
    platform_channel_id: str,
    channel_type: str,
    platform_message_id: str,
    platform_user_id: str,
    global_user_id: str,
    user_name: str,
    active_turn_platform_message_ids: list[str] | None = None,
    active_turn_conversation_row_ids: list[str] | None = None,
    debug_modes: dict[str, bool] | None = None,
    output_mode: OutputMode = "visible_reply",
    target_addressed_user_ids: list[str] | None = None,
    target_broadcast: bool = False,
) -> CognitiveEpisode:
    """Build a source-neutral episode for the current text `/chat` turn."""


def project_text_chat_compatibility_fields(
    episode: CognitiveEpisode,
) -> TextChatCompatibilityProjection:
    """Return current text `/chat` primitive fields without mutating production state."""
```

Required validation rules:

- `episode_id` must be present and a non-empty string.
- `trigger_source` must be present and equal one of the supported
  `TriggerSource` literals.
- `input_sources` must be present, a list, non-empty, and every element must
  equal one of the supported `InputSource` literals.
- `output_mode` must be present and equal one of the supported `OutputMode`
  literals.
- `percepts` must be present, a list, and non-empty.
- `target_scope` must be present, a dict, and include every field declared on
  `TargetScope` with the declared types.
- `origin_metadata` must be present, a dict, and include every field declared
  on `OriginMetadata` with the declared types.
- `timestamp` must be present and a non-empty string.
- `time_context` must be present, a dict, and include every field declared on
  `TimeContextDoc` (`current_local_datetime`, `current_local_weekday`) as
  non-empty strings. Do not import or re-derive these field names; read them
  from the `TimeContextDoc` declaration.
- every `input_sources` value must be represented by at least one percept.
- every percept `input_source` must be listed in `input_sources`.
- every percept must have a non-empty `percept_id`, supported `input_source`,
  supported `visibility`, string `content`, and dict `metadata`.
- `percept_id` values within a single episode must be unique.
- for `trigger_source="user_message"`, at least one percept must have
  `input_source="dialog_text"`.
- validation must not inspect text content to infer user intent, permissions,
  preferences, commitments, or meaning.

Validator failure-mode rules:

- All structural failures must raise `CognitiveEpisodeValidationError`. No
  other exception type may escape the validator for structural reasons.
- Tests assert on the exception type only. Do not assert on, or pin, the
  human-readable message text. The validator implementation may use any
  message wording.
- The validator must check structure top-down and raise on the first failure
  it finds. It must not collect, batch, or summarize multiple failures.

Text-only `/chat` builder output must include one `dialog_text` percept whose
content is the current `user_input` string, `visibility="model_visible"`,
`trigger_source="user_message"`, and `input_sources=["dialog_text"]`.

The compatibility projection helper must read from the episode and return only
the `TextChatCompatibilityProjection` fields above. It must not import or build
`IMProcessState`, call the graph, call RAG, call an LLM, persist data, or deliver
messages.

## Implementation Order

1. Add `tests/test_cognitive_episode_contract.py` with focused tests for the
   public API names, text `/chat` builder shape, validation failures, and
   compatibility projection shape. Run it and record the expected missing-module
   or missing-symbol failure in `Execution Evidence`.
2. Add `src/kazusa_ai_chatbot/cognition_episode.py` with type aliases,
   `TypedDict` shapes, `CognitiveEpisodeValidationError`, validators, text
   builder, and compatibility projection helper.
3. Run the focused module tests and iterate only inside the new test/module
   files until they pass.
4. Run static import/wiring greps to prove the live graph still does not consume
   `CognitiveEpisode`.
5. Run `stage_00` baseline tests and adjacent state/schema tests to confirm no
   behavior changed.
6. Update this plan's checklist and `Execution Evidence`.
7. Update the parent ledger and registry only after all verification passes.

## Progress Checklist

- [x] Stage 1 - focused contract tests added.
  - Covers: `tests/test_cognitive_episode_contract.py`.
  - Verify: focused test command fails only because the approved module or
    symbols do not exist yet.
  - Evidence: record command and failure summary in `Execution Evidence`.
  - Handoff: next agent starts at Stage 2.
  - Sign-off: Codex / 2026-05-09 after expected missing-module failure was
    recorded in `Execution Evidence`.
- [x] Stage 2 - contract module implemented.
  - Covers: `src/kazusa_ai_chatbot/cognition_episode.py`.
  - Verify: `py_compile` and focused contract tests pass.
  - Evidence: record public API names and test output in `Execution Evidence`.
  - Handoff: next agent starts at Stage 3.
  - Sign-off: Codex / 2026-05-09 after `py_compile` and focused contract
    tests passed.
- [x] Stage 3 - runtime untouched verification complete.
  - Covers: forbidden production graph, prompt, RAG, dialog, consolidation,
    adapter, scheduler, and persistence files.
  - Verify: static greps and `git diff --check` pass; no forbidden file is
    modified.
  - Evidence: record grep results and changed-file list in `Execution Evidence`.
  - Handoff: next agent starts at Stage 4.
  - Sign-off: Codex / 2026-05-09 after static greps, diff check, and
    changed-file review passed.
- [x] Stage 4 - regression baseline confirmed.
  - Covers: Stage 00 baseline and adjacent state/schema tests.
  - Verify: every command in `Verification` passes.
  - Evidence: record exact commands and results in `Execution Evidence`.
  - Handoff: next agent updates lifecycle records.
  - Sign-off: Codex / 2026-05-09 after Stage 00 baseline and adjacent
    state/schema tests passed.
- [x] Stage 5 - lifecycle records updated.
  - Covers: this plan, parent ledger, and registry.
  - Verify: status rows show `stage_01` completed and artifact paths are named.
  - Evidence: record parent ledger and registry update confirmation.
  - Handoff: Stage 01 is complete; Stage 02 remains blocked until separately
    approved.
  - Sign-off: Codex / 2026-05-09 after child status, parent ledger, and
    registry were updated to completed.

## Verification

### Focused Contract Tests

```powershell
venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py
```

### Static Checks

```powershell
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_episode.py tests\test_cognitive_episode_contract.py
git diff --check
rg -n "CognitiveEpisode|cognition_episode" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py
```

The final `rg` command must return no matches. A non-zero exit code from that
specific no-match grep is acceptable and should be recorded as expected.

### Regression Tests

```powershell
venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py
venv\Scripts\python -m pytest tests\test_state.py tests\test_persona_supervisor2_schema.py
```

No real LLM, live database, prompt-render, service-restart, adapter, or runtime
smoke test is required because this stage adds an unused contract module only.

## Acceptance Criteria

- `CognitiveEpisode` can represent current text `/chat` input.
- Invalid structural episodes are rejected by deterministic validation.
- The new module has no circular import with `state.py`.
- No live graph path consumes the new contract yet.
- The compatibility projection helper exists, is tested, and remains unused by
  production code.
- No prompt, RAG, dialog, consolidation, persistence, adapter, scheduler, or
  runtime startup file changes.
- No new LLM calls, database schema changes, migrations, or response-path
  latency changes.
- Stage 00 baseline still passes.
- Parent ledger can point to this stage's contract module, tests, and
  verification evidence.

## Data Migration

No database schema, collection, index, or stored-document migration is allowed
or required.

## Operational Steps

No service restart, scheduler operation, adapter operation, deployment step, or
manual runtime intervention is required. The character must keep running on the
existing `/chat` path during and after this stage.

## Completion Artifact Contract

`stage_01` is not complete until `Execution Evidence` records:

- The `CognitiveEpisode` contract module path.
- The text-only `/chat` builder path or function name.
- The structural validation helper paths or function names.
- The unit test path.
- The exact deterministic commands run and their result, including the
  `stage_00` baseline rerun.
- Confirmation that the parent ledger was updated so `stage_01` is complete.
- Confirmation that `development_plans/README.md` was updated so the Stage 01
  registry row is complete.
- Confirmation that no production graph, prompt, RAG, dialog, consolidation,
  persistence, adapter, scheduler, or runtime startup file was modified.

## Risks

- The contract may become too broad if it tries to solve later reflection and
  multimodal stages in detail.
- A builder that imports runtime graph state directly can create circular
  dependencies.
- Validation can accidentally become semantic classification; keep it
  structural.

## LLM Call And Context Budget

No LLM calls are added. No prompt context changes are allowed. Response-path
latency is unchanged because no production code consumes the new module in this
stage.

## Glossary

- `trigger_source`: why cognition is running.
- `input_source`: what cognition is perceiving.
- `percept`: one normalized unit of perceived content.
- `output_mode`: what the cognition result is allowed to produce.

## Execution Evidence

Approved on 2026-05-09 after review. Implementation started on 2026-05-09.

- Stage 1 focused contract test red check:
  - `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py`
    failed during collection with
    `ModuleNotFoundError: No module named 'kazusa_ai_chatbot.cognition_episode'`.
    This is the expected missing-module failure before implementation.
- Stage 2 contract module implementation:
  - Contract module path: `src/kazusa_ai_chatbot/cognition_episode.py`.
  - Public builder: `build_text_chat_cognitive_episode`.
  - Public validator: `validate_cognitive_episode`.
  - Validation error: `CognitiveEpisodeValidationError`.
  - Compatibility projection:
    `project_text_chat_compatibility_fields`.
  - Unit test path: `tests/test_cognitive_episode_contract.py`.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_episode.py tests\test_cognitive_episode_contract.py`
    passed.
  - `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py`
    passed: 10 passed.
- Stage 3 runtime untouched verification:
  - `git diff --check` passed with only a CRLF warning for
    `development_plans/active/short_term/multi_source_cognition_architecture_stage_01_cognitive_episode_contract_plan.md`.
  - `rg -n "CognitiveEpisode|cognition_episode" src\kazusa_ai_chatbot\service.py src\kazusa_ai_chatbot\state.py src\kazusa_ai_chatbot\nodes\persona_supervisor2.py src\kazusa_ai_chatbot\nodes\persona_supervisor2_schema.py`
    returned no matches, with exit code 1 as expected for the no-match grep.
  - Changed files were limited to the approved plan documents,
    `src/kazusa_ai_chatbot/cognition_episode.py`, and
    `tests/test_cognitive_episode_contract.py`.
- Stage 4 regression baseline verification:
  - `venv\Scripts\python -m pytest tests\test_multi_source_cognition_stage_00_regression_baseline.py`
    passed: 11 passed.
  - `venv\Scripts\python -m pytest tests\test_state.py tests\test_persona_supervisor2_schema.py`
    passed: 24 passed.
- Stage 5 lifecycle records:
  - This plan status was updated to `completed`.
  - Parent ledger
    `development_plans/active/short_term/multi_source_cognition_architecture_plan.md`
    was updated so `stage_01` is `completed` and names
    `src/kazusa_ai_chatbot/cognition_episode.py`,
    `tests/test_cognitive_episode_contract.py`, and required verification.
  - Registry `development_plans/README.md` was updated so the Stage 01 row is
    `completed | completed`.
  - No production graph, prompt, RAG, dialog, consolidation, persistence,
    adapter, scheduler, or runtime startup file was modified.
- Post-review cosmetic fixes:
  - Added `Raises: CognitiveEpisodeValidationError` to
    `build_text_chat_cognitive_episode`.
  - Added focused parametrized validation coverage for missing `episode_id`,
    missing `trigger_source`, non-bool `debug_modes`, unsupported percept
    `visibility`, and `user_message` episodes without `dialog_text`.
  - `venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\cognition_episode.py tests\test_cognitive_episode_contract.py`
    passed.
  - `venv\Scripts\python -m pytest tests\test_cognitive_episode_contract.py`
    passed: 15 passed.
  - `git diff --check` passed with CRLF warnings only.
  - The no-wiring grep against `service.py`, `state.py`,
    `persona_supervisor2.py`, and `persona_supervisor2_schema.py` returned no
    matches, with exit code 1 as expected.
