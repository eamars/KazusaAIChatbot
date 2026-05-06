# identity_free_memory_output_contract_plan

## Summary

- Goal: make durable memory-writing LLMs use a single canonical third-person memory subject for the active character, with the character name rendered from `character_profile["name"]`.
- Plan class: high_risk_migration
- Status: completed
- Mandatory skills: `local-llm-architecture`, `no-prepost-user-input`, `py-style`, `cjk-safety`, `test-style-and-execution`, `database-data-pull`, `memory-knowledge-maintenance`, `python-venv`
- Overall cutover strategy: bigbang for future memory-writer prompt contracts; migration for existing database content.
- Highest-risk areas: confusing user-owned `我` with character-owned `我`, treating the character name as a replaceable alias, hiding prompt contamination behind a broad helper, mutating raw source evidence, mechanically appended prompt text that breaks local-LLM flow, missing real-LLM false-negative/false-positive validation, and reintroducing runtime blacklist/output scanners.
- Acceptance criteria: each covered memory-writing LLM has a documented information-flow contract, every affected prompt organically integrates a Chinese third-person Memory Perspective Contract, the active-character name in prompt wording comes from `character_profile["name"]`, real LLM false-negative and false-positive tests pass, no alias-term configuration is introduced, no runtime output audit is introduced, and existing polluted prompt-facing memory is rewritten through an offline migration flow.

## Architectural Decision

Durable memory prose must use the third-person canonical character name when referring to the active character.

The name source is the character profile. For this codebase, prompt handlers must render the active-character name from `state["character_profile"]["name"]` or from an equivalent state value that is proven to be copied from `character_profile["name"]`. Reusable prompt code must not hard-code `杏山千纱`, `千纱`, `Kazusa`, or any other concrete character name.

The implementation must not introduce `active_character_reference_terms`, `active_character_terms`, alias discovery, or deterministic name-to-pronoun substitution. This decision is semantic, not cosmetic: durable memory has one subject convention. The memory writer may not interchange `我` and the character name depending on local phrasing.

`我` remains valid only inside outputs whose schema explicitly represents the character's first-person cognition or dialog, such as `internal_monologue` or final spoken reply text. Those fields are not the durable memory contract covered by this plan.

## Context

The visible failure is a memory projection such as `助理多次回以'亲爱的'`, mixed with generated references like `千纱`, `杏山千纱`, `角色`, and occasional `我`. The root issue is broader than one word: memory-writing prompts receive source evidence containing machine roles, character names, user-authored first person, prior polluted memory prose, and upstream LLM summaries without a stable durable-memory subject boundary.

Evidence already collected before this decision:

- `src/kazusa_ai_chatbot/state.py` and `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_schema.py` already carry `character_profile`; `service.py` derives `character_name` from the loaded personality profile.
- L2 cognition consumes `user_memory_context` as `fact`, `subjective_appraisal`, and `relationship_signal` fields. Its `internal_monologue` output is explicitly first-person, but that is a cognition output schema, not evidence that durable memory should use first person.
- Existing exported `user_memory_units` are mixed but already heavily third-person/name-based. They include character-name references, generic labels such as `角色`/`助理`, user references such as `用户`, and some first-person fields. The migration must normalize durable memory, not preserve this inconsistency.
- Real LLM probes showed that both first-person and third-person prompts can work on simple cases, but first-person durable memory creates an unnecessary identity collision with user-authored `我`. The lower-risk durable-memory contract is canonical third person: user `我` is interpreted as the user; active-character references are rendered as `character_profile["name"]`.

The current codebase has two prompt-facing memory writer families in scope:

- Consolidator memory writers:
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
- Reflection global promotion:
  - `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
  - It generates `sanitized_memory_name` and `sanitized_content`, then writes them into shared persistent `memory.memory_name` and `memory.content`.

Raw source data remains unchanged. Conversation history, user messages, stored roles, source rows, and upstream evidence are preserved for auditability. The fix is the prompt-facing contract and any narrow structural prompt-input projection required by each memory-writer call.

## Mandatory Skills

- `local-llm-architecture`: load before changing prompt, memory, RAG, reflection, or background LLM behavior. For every prompt edit, complete and record a system-contract, wording, and prompt-flow audit before implementation, then rerun the review after the edit.
- `no-prepost-user-input`: load before editing any path that reads `decontexualized_input`, facts, promises, relationship state, or memory outputs. Prompt projection must not decide accepted facts, commitments, preferences, or channels.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python prompt strings or tests containing Chinese/Japanese text.
- `test-style-and-execution`: load before adding, changing, or running tests.
- `database-data-pull`: load before read-only DB inspection or exports.
- `memory-knowledge-maintenance`: load before scanning or changing the shared `memory` collection.
- `python-venv`: load before running Python scripts, tests, or dependency commands.

## Mandatory Rules

- Do not create `src/kazusa_ai_chatbot/memory_identity_contract.py`.
- Do not add runtime output audit, forbidden-term scanning, output blacklist validation, or post-generation rejection based on character names.
- Do not mutate raw source input. Raw `conversation_history`, stored `role="assistant"`, raw user text, `decontexualized_input`, `final_dialog`, and source database rows must remain unchanged.
- Do not add `active_character_reference_terms`, `active_character_terms`, `trusted_active_character_terms`, alias discovery, alias normalization, or any character-name replacement list.
- Do not learn aliases from user text, user image, previous memories, RAG, dialog output, or database content.
- Do not create a broad recursive `sanitize_memory_writer_payload(payload)` API or equivalent generic scrubber over arbitrary payloads.
- Prompt projection for covered memory-writer LLM payloads must be field-aware and call-site-owned through the stage-specific functions in `memory_writer_prompt_projection.py`. Every transformed field must have a documented source, producer, consumer LLM, contamination risk, allowed transformation, and semantic preservation rule. If a stage has no field requiring structural projection, its projection function still returns a deep copy and documents the no-op.
- Prompt projection over user-controlled or user-interpretation fields must not summarize, paraphrase, reorder, drop, deduplicate, translate, rephrase, classify, normalize, or substitute character names inside text bodies. Raw text fields pass through unchanged unless a row in the Information Flow Audit explicitly permits a structural metadata change outside the text body.
- Machine speaker roles in memory-writer prompt payloads must not introduce an intermediate active-character reference such as `active_character`. For active-character turns, expose direct structural speaker metadata such as `speaker_kind="character"` and `speaker_name="{character_name}"`, where `{character_name}` is rendered from `character_profile["name"]`. Do not expose `assistant` as a natural-language actor label to memory writers.
- The active-character name used by prompt wording must come from `character_profile["name"]`. If a function only receives `character_name`, the implementation must prove that value was copied from the current `character_profile["name"]`; otherwise thread the profile name into that call path.
- Generated durable memory prose must use third-person perspective. When the active character must be named, it must use the exact rendered `{character_name}` value from `character_profile["name"]`.
- Generated durable memory prose must not use `我` to mean the active character. If source evidence contains user-authored `我`, the generated durable memory must attribute it to `用户`, `对方`, or `用户自己` according to the evidence.
- Generated durable memory prose must not use generic active-character labels such as `角色`, `助理`, `assistant`, `active_character`, or untrusted short aliases as substitutes for `{character_name}`.
- `我` is allowed only in schema fields that explicitly require first-person cognition or dialog. This plan does not require changing those fields.
- Prompt templates must keep `{character_name}` as a render-time placeholder in the organically integrated Chinese perspective wording. They must not hard-code the literal character name in reusable prompt strings.
- Every prompt change must include a pre-edit local-LLM architecture audit and a post-edit prompt-flow review. The review must explicitly cover current stage responsibility, visible input fields, expected JSON output fields, generation procedure, placement of the perspective wording, vocabulary introduced by the change, rendering safety, and whether the edited prompt still reads as one coherent instruction flow for a weaker local LLM.
- Memory Perspective Contract wording must be organically integrated into the prompt's existing `# Generation Procedure`, `# Output Format`, field descriptions, or equivalent local sections. Do not mechanically append a disconnected block if the prompt already has a better logical place for the contract. A standalone `# 记忆视角契约` heading is allowed only when it is the clearest local-LLM contract boundary.
- A helper or shared prompt fragment is allowed only when absolutely necessary because the existing prompt rendering architecture cannot safely express the change inline. The agent must document the necessity, affected prompts, resulting rendered text, and why direct organic editing is impossible before implementing the helper.
- Prompt edits must be validated by real LLM tests with both false-negative and false-positive cases. Static prompt tests are necessary but not sufficient.
- Real LLM prompt validation is a hard gate. If the live local LLM endpoint is unavailable or a real LLM case cannot be inspected, stop and report the blocker; do not mark the prompt stage, code-side verification, or plan complete.
- The offline database sanitation CLI inherits the same `no-prepost-user-input` constraint: it must not deterministically pre-process or post-process the LLM rewrite output to filter, reclassify, or override the LLM's chosen wording. It may only do structural sanitation: JSON parsing, type checks, persistence through existing update helpers, and embedding recompute.
- If a field cannot be projected without semantic loss, preserve the field unchanged and mark the field as blocked in the information-flow audit. Do not hide the risk through redaction.
- Do not sanitize prompt-facing memory at read/projection time as the primary fix. Fix future writers and run the approved offline migration stage for existing data.
- Do not broaden this plan into RAG redesign, general memory quality cleanup, affinity changes, storage role renaming, or personality-profile renaming.
- After any automatic context compaction, the active agent must reread this entire plan before continuing implementation, verification, handoff, or final reporting.
- After signing off any major progress checklist stage, the active agent must reread this entire plan before starting the next stage.

## Must Do

- Add field-aware memory-writer prompt projection helpers only for structural prompt-input views needed by each covered LLM call.
- Organically integrate the Chinese third-person Memory Perspective Contract into every covered memory-writing prompt.
- Render `{character_name}` from `character_profile["name"]` at every affected prompt call site.
- Wire the projection helper for each covered LLM call before `json.dumps(..., ensure_ascii=False)`.
- Keep source data, storage roles, raw user text, and profile canonical `name` unchanged.
- Remove the previous plan requirement for runtime output validation or forbidden-term rejection.
- Add focused tests proving each documented prompt payload field is projected exactly as contracted, prompts do not hard-code concrete character names, prompt rendering is valid, and real LLM false-negative/false-positive cases behave correctly.
- Add an offline database migration CLI that rewrites existing prompt-facing memory text through the same third-person character-name contract.

## Deferred

- Do not rewrite raw `conversation_history`.
- Do not rename storage roles, adapter roles, message-envelope roles, or `assistant_moves`.
- Do not change personality profile canonical `name`.
- Do not add character alias configuration for memory writing.
- Do not automatically discover character aliases from user text, user image, previous memories, or RAG.
- Do not add a response-path LLM call.
- Do not enforce this contract in generic `memory_evolution` repository APIs for seed/manual/admin callers.
- Do not sanitize seed/world knowledge unless a separate curated-memory plan explicitly scopes it.

## Cutover Policy

| Area                                           | Policy     | Instruction                                                                                        |
| ---------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------- |
| Future memory-writer prompt payloads           | bigbang    | Covered LLM calls receive documented field-projected prompt payloads once code lands.              |
| Future generated memory output prompt contract | bigbang    | Covered prompts require third-person `{character_name}` for active-character references.           |
| Character name source                          | bigbang    | `{character_name}` is rendered from `character_profile["name"]`; no duplicate alias/config source. |
| Runtime output audit                           | bigbang    | No output blacklist scanner or forbidden-term validator is allowed.                                |
| Raw source input and storage roles             | compatible | Preserve existing stored data and machine role vocabulary.                                         |
| Existing prompt-facing memory data             | migration  | Rewrite through offline migration after code-side changes pass.                                    |
| Shared `memory` rows from reflection promotion | migration  | Rewrite runtime-generated active rows through public memory-evolution mutation APIs.               |

## Agent Autonomy Boundaries

- The agent may choose local implementation mechanics only when they preserve this plan's contracts.
- The agent must not introduce output blacklist validation as a substitute for prompt contracts and migration.
- The agent must not introduce alias discovery, alias config, or deterministic name/pronoun substitution.
- The agent must not introduce deterministic pre- or post-processing of user-input semantics inside projection helpers, the offline migration CLI, or consolidator/promotion wiring. If prompt projection feels insufficient, the agent must fix the prompt, schema, or evaluator.
- For prompt changes, the agent must complete the local-LLM architecture audit before editing and must review every prompt diff after editing for logic flow, local-LLM clarity, rendering, and downstream parser compatibility.
- The agent may use a helper or shared prompt fragment only when it is absolutely necessary and documented before use.
- The agent must not perform unrelated cleanup, formatting churn, dependency upgrades, or broad prompt rewrites.
- If the plan and code disagree, preserve the plan's stated intent and report the discrepancy.
- If a required instruction is impossible, stop and report the blocker instead of inventing a substitute.

## Target State

Covered memory-writing LLMs see prompt inputs where:

- Active-character turns are represented with direct speaker metadata: `speaker_kind="character"` and `speaker_name="{character_name}"`, rendered from `character_profile["name"]`.
- User turns are represented with `speaker_kind="user"` and a stable user-facing name only when needed to bind memories to the user.
- The active-character profile name is supplied as the dedicated prompt variable `{character_name}`, rendered from `character_profile["name"]`.
- Raw text evidence remains raw text evidence. User-authored `我` remains visible as source evidence and is interpreted by the LLM according to the source speaker.
- Prior polluted memory prose may remain visible until migration, but future generation and rewrite prompts must canonicalize active-character references to `{character_name}`.

Covered memory-writing LLMs generate durable text where:

- `user_memory_units.fact`, `subjective_appraisal`, and `relationship_signal` use third-person `{character_name}` when the active character is referenced.
- `relationship_recorder.subjective_appraisals` and `last_relationship_insight` use third-person `{character_name}` when the active character is referenced.
- Character self-image summaries and reflection summaries use third-person `{character_name}` when saved as durable prompt-facing memory prose. First-person cognition/dialog fields are out of scope unless they are reused as durable memory.
- Reflection promotion `sanitized_memory_name` and `sanitized_content` use third-person `{character_name}` when the active character is referenced.
- User-authored first-person statements are attributed to `用户`, `对方`, or `用户自己`, not to `{character_name}`.
- Generic labels such as `角色`, `助理`, `assistant`, and `active_character` do not appear as generated active-character names.
- Each affected LLM prompt organically integrates the Chinese Memory Perspective Contract into the owning prompt's instruction flow, with `{character_name}` left as a render-time placeholder.

## Design Decisions

| Topic                    | Decision                                         | Rationale                                                                                                             |
| ------------------------ | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Durable memory subject   | Canonical third-person active-character name     | Separates user-owned `我` from character identity and avoids changing subject convention across memory consumers.      |
| Character name source    | `character_profile["name"]`                      | There is already one trusted profile identity source; a second alias config would reintroduce inconsistency.          |
| Output contract          | Positive Chinese third-person contract           | Tells the LLM what to generate without runtime blacklist rejection.                                                   |
| Prompt change mechanism  | Organic prompt integration, direct by default    | User requires prompt wording to be integrated into each affected prompt's logic, not mechanically appended.           |
| Prompt projection        | Direct profile-name speaker metadata only        | Role/display metadata can be made prompt-safe without introducing an `active_character` synonym.                      |
| Prompt audit             | Mandatory local-LLM architecture review          | Weaker local LLMs are sensitive to prompt flow, vocabulary, hidden terms, and rendering mistakes.                     |
| Real LLM validation      | Required false-negative and false-positive tests | Static prompt checks cannot prove the runtime model follows the new perspective contract without losing memories.     |
| Runtime output audit     | Do not implement                                 | User explicitly rejected this path, and it risks hard-coded character identity terms.                                 |
| Alias source             | No alias source                                  | Active-character memory subject is not an alias problem; it is a semantic contract.                                   |
| Raw input                | Preserve unchanged                               | DB source evidence must remain inspectable and truthful.                                                              |
| Generic scrubber         | Do not implement                                 | A recursive sanitizer would hide whether contamination came from history, RAG memory, reflection output, or metadata. |
| Shared-memory repository | Keep generic                                     | Seed/manual memory is not always active-character prose.                                                              |

## Information Flow Audit

The implementation agent must verify this table against the latest code before editing Python. If a covered LLM call has changed, update the table first and keep the same columns. A row is implementation-ready only when the allowed projection and semantic preservation rule are explicit.

| LLM stage                     | Prompt payload field/path                                          | Producer/source                                        | Source type                                  | Contamination risk                                                                     | Allowed projection                                                                                                                                                                                               | Must preserve                                                           |
| ----------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------ | -------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Memory-unit extractor         | `chat_history_recent[].role`                                       | `format_history_for_llm(state["chat_history_recent"])` | Machine speaker metadata                     | `assistant` can be copied as a character label                                         | Replace prompt-facing role metadata with `speaker_kind="character"` plus `speaker_name="{character_name}"` for active-character turns; use `speaker_kind="user"` for user turns. Do not emit `active_character`. | Speaker ownership of each turn                                          |
| Memory-unit extractor         | `chat_history_recent[].display_name`                               | Conversation DB / formatter                            | User or active-character display metadata    | Active-character display names and user-authored labels can leak                       | For active-character turns, set prompt-facing `speaker_name` to `{character_name}` from profile. For user turns, preserve user display only if needed as participant metadata.                                   | User identity clues needed to bind memories to the user                 |
| Memory-unit extractor         | `chat_history_recent[].body_text`                                  | Raw conversation DB                                    | Raw user and active-character utterance text | User may mention the character by name; active-character text may include first person | Preserve unchanged. Do not replace `我`, names, aliases, or generic labels in text evidence.                                                                                                                      | Literal conversational evidence and affective meaning                   |
| Memory-unit extractor         | `rag_user_memory_context.*`                                        | `user_memory_unit_retrieval._project_unit`             | Prior LLM-generated memory                   | Existing polluted prose becomes self-reinforcing                                       | Preserve structure and text unchanged for prompt input; rely on the prompt contract and offline migration for outputs.                                                                                           | Stored fact/appraisal/relationship signal semantics                     |
| Memory-unit extractor         | `final_dialog`                                                     | Previous dialog LLM output                             | Generated active-character reply             | May contain first-person dialog or copied names                                        | Preserve unchanged as evidence; prompt output must canonicalize durable memory references to `{character_name}`.                                                                                                 | Reply meaning and conversational evidence                               |
| Memory-unit extractor         | `decontextualized_input`, affect/subtext/appraisal evidence fields | Cognition/decontextualization/reflection LLMs          | Upstream LLM interpretation and evidence     | Can contain copied aliases before the memory extractor writes durable prose            | Preserve unchanged. No channel reclassification, acceptance gating, commitment inference, preference normalization, or text substitution.                                                                        | Upstream interpretation, affect, stance, intent, and evidence ownership |
| Memory-unit extractor         | fact-harvester and promise-harvester evidence fields               | Fact/promise harvester LLMs                            | Upstream LLM channel decisions               | Field names are semantically sensitive and easy to misuse as deterministic gates       | Preserve unchanged in projection module; rely on the prompt contract for generated extractor output.                                                                                                             | LLM-owned fact/promise channel selection and wording                    |
| Memory-unit merge judge       | `new_memory_unit`, `candidate_clusters` prose fields               | Extractor output and user-memory-unit collection       | Current/prior LLM-generated memory           | Merge reason can copy polluted names into merge history and influence rewrite          | Preserve input text; rewrite/stability prompt must output canonical `{character_name}` when it writes durable prose.                                                                                             | Candidate identity, cluster IDs, merge/evolve semantics                 |
| Memory-unit rewrite/stability | existing unit prose fields                                         | User-memory-unit collection                            | Prior LLM-generated memory                   | Rewrites can preserve or amplify polluted names                                        | Preserve input text; output rewrite contract uses canonical `{character_name}`.                                                                                                                                  | Unit identity, timestamps, counts, semantic category                    |
| Relationship recorder         | `decontexualized_input`                                            | Cognition/decontextualization LLM                      | Upstream LLM interpretation of user input    | Can contain copied aliases, user first person, or role labels                          | Preserve unchanged; no fact-vs-promise re-routing, summarization, keyword matching, or text substitution.                                                                                                        | User intent/content as interpreted by upstream LLM                      |
| Relationship recorder         | `final_dialog`, `content_anchors`                                  | Dialog/reflection pipeline                             | Generated dialog/evidence snippets           | Can carry copied active-character names or first-person dialog                         | Preserve unchanged as evidence; prompt output must canonicalize durable relationship prose to `{character_name}`.                                                                                                | Evidence needed to assess relationship state                            |
| Global-state updater          | reflection/session summary fields                                  | Reflection consolidator inputs                         | LLM-generated summaries and evidence         | Existing summaries can leak named active-character labels                              | Preserve input; generated durable prompt-facing prose uses canonical `{character_name}`.                                                                                                                         | Emotional state and durable self-observation                            |
| Character image writers       | self-image/session/compression inputs                              | Character-state and reflection pipeline                | LLM-generated self-image/evidence            | Self-image can encode active-character identity as a generated fact                    | Preserve input; generated durable self-image prose uses canonical `{character_name}` when naming the character.                                                                                                  | Self-image semantics, confidence, and recency                           |
| Reflection promotion          | evidence cards, daily syntheses, `sanitized_observation`           | Reflection-cycle LLM outputs and stored observations   | Prior LLM-generated memory candidates        | Shared-memory `memory_name`/`content` can receive polluted names                       | Preserve input; generated promoted memory uses canonical `{character_name}` when naming the character.                                                                                                           | Evidence, source scope, and promotion decision inputs                   |
| Offline database migration    | covered stored memory prose fields                                 | MongoDB memory/profile/state rows                      | Existing durable LLM output                  | Existing rows feed future prompts and exports                                          | Rewrite through the third-person offline LLM flow after code-side fix; no runtime blacklist.                                                                                                                     | Reviewable before/after record and embedding ownership                  |

## Contracts And Data Shapes

### Prompt Projection Module

Create `src/kazusa_ai_chatbot/memory_writer_prompt_projection.py`.

This module owns structural prompt views for memory-writing LLM calls. It must not own memory extraction semantics, relationship classification, commitment handling, persistence decisions, prompt wording, prompt rendering, prompt assembly, alias handling, or output validation.

`memory_writer_prompt_projection.py` must not define, export, render, return, or concatenate the Memory Perspective Contract section or any prompt fragment. Prompt-contract text belongs in the affected prompt-owning files listed in `Change Surface`; a stage-owned helper in those files is allowed only if the Prompt Change Audit documents why it is absolutely necessary.

Public functions:

```python
def project_memory_unit_extractor_prompt_payload(
    payload: dict,
    *,
    character_name: str,
) -> dict:
    """Return the structural prompt payload for memory-unit extraction."""

def project_memory_unit_rewrite_prompt_payload(
    payload: dict,
    *,
    character_name: str,
) -> dict:
    """Return the structural prompt payload for memory-unit rewrite/stability calls."""

def project_relationship_prompt_payload(
    payload: dict,
    *,
    character_name: str,
) -> dict:
    """Return the structural prompt payload for relationship/global-state reflection calls."""

def project_character_image_prompt_payload(
    payload: dict,
    *,
    character_name: str,
) -> dict:
    """Return the structural prompt payload for character image writers."""

def project_reflection_promotion_prompt_payload(
    payload: dict,
    *,
    character_name: str,
) -> dict:
    """Return the structural prompt payload for reflection promotion."""
```

Allowed private primitive:

```python
def _project_prompt_speaker_row(
    row: dict,
    *,
    character_name: str,
) -> dict:
    """Project known speaker metadata fields for prompt input only."""
```

Rules for prompt projection:

- It must return a deep copy and never mutate the caller's source object.
- All projection functions must be deterministic pure functions of the input payload and the explicit `character_name` argument. No LLM call, embedding model, classifier, network call, regex over user semantics, environment lookup, alias list, or character-name replacement.
- Public projection functions must name the LLM surface they serve; callers must not pass arbitrary payloads to a generic scrubber.
- It may rewrite documented prompt-facing speaker metadata:
  - `role="assistant"` -> `speaker_kind="character"`, `speaker_name=character_name`
  - user role/display metadata -> `speaker_kind="user"` plus user display metadata only when needed for memory binding
- It must not infer or replace terms from user text, memory content, dialog output, or profile names.
- It must not classify facts, commitments, preferences, tone, relationship meaning, or memory type.
- It must not summarize, paraphrase, reorder, deduplicate, translate, or rephrase any text field.
- It must not read or branch on cognition fields such as `logical_stance`, `character_intent`, `commitment_type`, `accepted_user_preferences`, `new_facts`, `future_promises`, or `relationship_signal`. Those fields are passed through as opaque values.
- It must preserve fields not listed in the Information Flow Audit unless the specific public function documents a safe structural projection for that path.
- The module must not introduce helpers matching `_filter_*`, `_classify_*`, `_infer_*`, `_normalize_*`, `_score_*`, `_validate_*`, `_route_*`, `_reclassify_*`, or `_decide_*`. These names indicate semantic interpretation forbidden by `no-prepost-user-input`. New helpers require a plan amendment.

### Canonical Character Name Source

Do not add character configuration for memory-writer aliases.

The canonical active-character name is:

```python
character_name = state["character_profile"]["name"]
```

For call paths that already receive `character_name`, the implementation must verify and document that the value originates from the current `character_profile["name"]`. If that cannot be proven, thread `character_profile["name"]` into the prompt-rendering function directly.

Tests must render affected prompts with a fixture profile such as:

```python
character_profile = {"name": "杏山千纱 (Kyōyama Kazusa)"}
```

The fixture value is allowed in tests and prompt-render traces. Reusable source prompt literals must keep `{character_name}` as a placeholder.

### Positive Prompt Contract

Every covered memory-writing prompt must organically integrate the following Chinese obligations into the prompt's existing instruction flow. The wording can be adapted to fit the prompt's local sections, but the meaning must remain intact and `{character_name}` must remain a render-time placeholder:

```text
# 记忆视角契约
- 本契约适用于你生成的可长期保存的 JSON 记忆字段。
- 记忆文本采用第三人称视角。
- 当前角色的规范名称是“{character_name}”，该名称来自角色档案 character_profile["name"]。
- 当必须指代当前角色时，只使用“{character_name}”。
- 不要用“我”指代当前角色；输入中的“我”必须按原说话人理解。
- 如果用户说“我……”，生成记忆时应写作“用户……”“对方……”或“用户自己……”，不要把这个“我”归到当前角色。
- 不要把说话人标签、角色字段、显示名称、泛称或机器角色名复制成当前角色名称；不要用“角色”“助理”“assistant”等标签指代当前角色。
- 只返回有效 JSON。
```

Prompt integration rule:

- Integrate the obligations into each affected LLM prompt in:
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
  - `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
  - `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
- Place the perspective guidance where the local prompt flow expects generation rules: inside the existing `# Generation Procedure`, field-level output instructions, `# Output Format`, or the nearest equivalent section. Use a standalone `# 记忆视角契约` heading only when that is the clearest local boundary for the prompt.
- Do not mechanically append the block to the end of a prompt without checking whether it conflicts with later schema, examples, or JSON-only instructions.
- Prefer direct edits inside each affected prompt literal/template. A helper, shared fragment, or prompt builder is allowed only when absolutely necessary for the existing rendering path; if used, it must be stage-owned, documented in the Prompt Change Audit, and validated by inspecting each final rendered prompt.
- Tests must inspect the affected prompt definitions or final rendered prompts for each LLM stage and prove the Chinese perspective obligations are present in the actual prompt text sent to the model.

### Prompt Change Audit

Before editing any affected prompt, the implementation agent must record a compact audit for that prompt:

```md
- Prompt:
- Owning file/function:
- Existing stage responsibility:
- Visible input fields:
- Expected JSON output fields:
- Existing generation procedure / reasoning flow:
- Where the third-person Memory Perspective Contract will be integrated:
- Character name source at this call site:
- New or changed wording:
- Local-LLM compliance check:
- Helper necessity: `none` or documented unavoidable reason:
```

After editing, the agent must review the diff and rendered prompt:

- Confirm the prompt still has a coherent order for a weaker local LLM: task, procedure, input meaning, output fields, third-person memory-subject rule, JSON-only requirement.
- Confirm the perspective wording does not conflict with examples, field descriptions, or downstream parser expectations.
- Confirm introduced vocabulary is defined in the prompt or visible in the payload.
- Confirm `{character_name}` renders from `character_profile["name"]`.
- Confirm `.format(...)`, f-string interpolation, literal braces, and JSON examples render correctly at runtime.
- Record the review result in `Execution Evidence`.

## LLM Call And Context Budget

Before:

- Existing background consolidation and reflection-promotion LLM calls receive raw or mechanically projected source payloads.
- Durable memory fields use inconsistent subject conventions across existing data and prompts.

After:

- No new response-path LLM calls.
- No new production background LLM calls for prompt sanitation.
- Prompt projection is deterministic and structural over the payload copy only.
- Covered prompt text grows by the Chinese third-person perspective obligations integrated into each affected prompt's existing procedure/output instructions.
- No generic prompt-contract helper, shared prompt fragment, or central reusable Memory Perspective Contract constant is added unless a Prompt Change Audit documents absolute necessity and rendered-prompt tests cover every affected stage.
- Real LLM validation adds operator-run test calls only; no production response-path or background call count changes.
- Offline database migration may use LLM rewrite calls, but only in operator-run scripts.

## Change Surface

### Create

- `src/kazusa_ai_chatbot/memory_writer_prompt_projection.py`
  - Owns field-aware profile-name speaker metadata projection for memory writers.
  - Must not own, export, render, or concatenate the Chinese Memory Perspective Contract prompt text.
- `src/scripts/sanitize_memory_writer_perspective.py`
  - Offline dry-run/apply CLI for existing prompt-facing memory text.
- `tests/test_memory_writer_prompt_projection.py`
  - Focused projection contract tests.
- `tests/test_memory_writer_information_flow_contracts.py`
  - Static/fixture tests that prove covered LLM payload fields match this plan's information-flow contracts.
- `tests/test_memory_writer_prompt_contracts.py`
  - Prompt contract tests for covered LLM stages. Tests must inspect prompt definitions or rendered prompts and fail if the Chinese perspective obligations are missing from the actual model-facing prompt text.
- `tests/test_memory_writer_perspective_live_llm.py`
  - Real LLM false-negative and false-positive prompt behavior tests with durable trace artifacts.
- `tests/test_memory_writer_database_sanitizer.py`
  - Offline CLI tests with mocked DB/LLM helpers.

### Modify

- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py`
  - Render affected memory-unit extractor/rewrite prompts with `character_name=state["character_profile"]["name"]`.
  - Project prompt payloads structurally where role/display metadata can leak as a natural-language actor label.
  - Organically integrate the Chinese third-person Memory Perspective Contract obligations into each affected extractor and rewrite prompt's existing logic.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py`
  - Render relationship/global-state prompts with `character_name=state["character_profile"]["name"]`.
  - Project prompt payloads structurally where role/display metadata can leak as a natural-language actor label.
  - Organically integrate the Chinese third-person Memory Perspective Contract obligations into each relevant prompt's existing logic.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py`
  - Render self-image session/compression prompts with `character_name=state["character_profile"]["name"]`.
  - Project prompt payloads structurally where role/display metadata can leak as a natural-language actor label.
  - Organically integrate the Chinese third-person Memory Perspective Contract obligations into each affected prompt's existing logic.
- `src/kazusa_ai_chatbot/reflection_cycle/promotion.py`
  - Render global-promotion prompts with the canonical profile name supplied by the caller.
  - If the current promotion entry point lacks the active character profile/name, thread `character_profile["name"]` into the call path from the owning graph or job context; do not add a duplicate config.
  - Organically integrate the Chinese third-person Memory Perspective Contract obligations into the promotion prompt's existing logic.
- `src/scripts/run_touched_llm_regression.py`
  - Add or update one-by-one runnable real LLM cases for the changed memory-writer prompts when this existing regression harness is the cleanest way to execute and log false-negative/false-positive cases.
- `tests/test_user_memory_units_rag_flow.py`
  - Update fixtures that currently expect `Kazusa`, `角色`, `助理`, or first-person active-character durable memory prose.
- `tests/test_consolidator_reflection_prompts.py`
  - Add prompt contract and projected-payload regression tests.
- `tests/test_reflection_cycle_stage1c_promotion.py`
  - Add promotion projected-payload and prompt-contract tests.

### Keep

- `personalities/kazusa.json`
  - Keep canonical `name` unchanged. Do not add memory-writer alias config.
- `src/kazusa_ai_chatbot/time_context.py::format_history_for_llm`
  - Keep general history formatting unchanged for non-memory-writer callers.
- `src/kazusa_ai_chatbot/rag/user_memory_unit_retrieval.py`
  - Keep projection mechanical; do not hide dirty stored rows at read time.
- `src/kazusa_ai_chatbot/memory_evolution/repository.py`
  - Keep generic repository contracts unchanged.
- `conversation_history.role`
  - Keep stored `user | assistant` vocabulary unchanged.

## Data Migration

Migration happens only after code-side prompt contracts pass tests.

The migration is an offline rewrite, not a runtime audit:

- It processes covered prompt-facing rows within explicit scopes and limits.
- It sends each row through the same third-person character-name prompt contract.
- It renders `{character_name}` from `character_profile["name"]`.
- It does not scan generated output with a forbidden-term list.
- It produces a dry-run report for human review before apply.

### Dry-Run Command

```powershell
venv\Scripts\python.exe -m scripts.sanitize_memory_writer_perspective `
  --dry-run `
  --scan-active-user-memory-units `
  --scan-user-profiles `
  --scan-character-state `
  --scan-persistent-memory `
  --limit 500 `
  --output test_artifacts\memory_writer_perspective_dry_run.json
```

Dry-run report shape:

```json
{
  "dry_run": true,
  "scopes": ["user_memory_units", "user_profiles", "character_state", "memory"],
  "character_name_source": "character_profile.name",
  "records_seen": 100,
  "records_with_proposed_changes": 12,
  "records": [
    {
      "collection": "user_memory_units",
      "document_key": "unit_id",
      "document_id": "example",
      "status": "ready|unchanged|blocked",
      "before": {"fact": "..."},
      "after": {"fact": "..."},
      "notes": []
    }
  ]
}
```

### Apply Command

```powershell
venv\Scripts\python.exe -m scripts.sanitize_memory_writer_perspective `
  --apply `
  --input test_artifacts\memory_writer_perspective_dry_run.json `
  --output test_artifacts\memory_writer_perspective_apply_report.json
```

Apply behavior:

- Apply only `status="ready"` records from a reviewed dry-run report.
- For `user_memory_units`, update through `update_user_memory_unit_semantics(..., increment_count=False)` and recompute embeddings through that normal helper.
- For `user_profiles.last_relationship_insight`, update only if the dry-run proposed a non-empty replacement.
- For `character_state.reflection_summary` and `self_image`, use the existing character-state update paths.
- For runtime-generated shared `memory` rows, write migrated replacements through public `memory_evolution` mutation APIs so embeddings and cache invalidation remain owned by that package.
- Do not automatically rewrite seed-managed or manual shared-memory rows; report them as blocked/manual-review unless explicitly handled by a separate curated-memory plan.

## Implementation Order

1. Verify and, if needed, update the Information Flow Audit against the latest code before editing Python.
2. Add information-flow contract tests.
3. Add prompt projection module tests.
4. Implement `memory_writer_prompt_projection.py`.
5. Add prompt-contract tests for covered prompts. These tests must inspect prompt definitions or rendered prompts and assert the Chinese third-person perspective obligations are present in each model-facing prompt text.
6. Add real LLM test cases before prompt implementation:
   - False-negative case: polluted active-character labels are present in prompt evidence, and the generated memory fields must use the exact `{character_name}` rendered from `character_profile["name"]` instead of copying labels such as `角色`, `助理`, `assistant`, or stale aliases.
   - False-positive case: user-owned `我`, legitimate user/third-party/title references, and unrelated named entities remain semantically intact; the prompt must not force unrelated subjects into `{character_name}`, drop valid memories, or confuse user ownership with active-character identity.
   - Each real LLM case must write a durable trace artifact with rendered prompt, input payload, raw output, parsed output, expected profile name, and human/agent judgment notes.
7. Complete the Prompt Change Audit for each affected prompt before editing wording.
8. Wire structural projection into consolidator memory-unit writers as documented by the Information Flow Audit, and organically integrate the Chinese third-person Memory Perspective Contract into each affected consolidator memory-unit prompt.
9. Wire structural projection into reflection and self-image writers as documented by the Information Flow Audit, and organically integrate the Chinese third-person Memory Perspective Contract into each affected reflection/self-image prompt.
10. Wire structural projection into reflection global promotion as documented by the Information Flow Audit, and organically integrate the Chinese third-person Memory Perspective Contract into the promotion prompt.
11. Review every prompt diff and rendered prompt for logic flow, vocabulary, rendering safety, and parser compatibility; record the review result.
12. Update fixtures that currently encode named aliases, generic labels, or first-person active-character generated memory prose.
13. Implement offline database migration CLI and tests.
14. Run focused deterministic and prompt-rendering tests.
15. Run each real LLM false-negative/false-positive test one by one and inspect each trace before running the next.
16. Run dry-run database migration.
17. Review dry-run report.
18. Apply migration only after review.
19. Re-export and smoke-check affected memory projections.

## Progress Checklist

- [x] Stage 1 - Information-flow contracts verified
  
  - Covers: this plan's Information Flow Audit and `tests/test_memory_writer_information_flow_contracts.py`.
  - Verify: static contract tests pass or fail with exact changed code paths that need the audit table updated.
  - Evidence: record test output.
  - Handoff: next agent starts Stage 2.
  - Sign-off: `Codex / 2026-05-06`

- [x] Stage 2 - Structural prompt projection module complete
  
  - Covers: `memory_writer_prompt_projection.py` and `tests/test_memory_writer_prompt_projection.py`.
  - Verify: focused projection tests pass and source-object immutability is proven.
  - Evidence: record test output.
  - Handoff: next agent starts Stage 3.
  - Sign-off: `Codex / 2026-05-06`

- [x] Stage 3 - Prompt audits and real LLM cases defined
  
  - Covers: Prompt Change Audit entries, `tests/test_memory_writer_prompt_contracts.py`, and `tests/test_memory_writer_perspective_live_llm.py` or `src/scripts/run_touched_llm_regression.py` cases.
  - Verify: prompt contract tests exist, real LLM false-negative and false-positive cases exist, and each case writes rendered prompt/input/raw output/parsed output/judgment trace artifacts.
  - Evidence: record audit entries and baseline test/case definitions.
  - Handoff: next agent starts Stage 4.
  - Sign-off: `Codex / 2026-05-06`

- [x] Stage 4 - Consolidator prompt integration complete
  
  - Covers: memory units, reflection, and self-image consolidator files.
  - Verify: prompt payload tests and prompt contract tests pass, rendered prompts contain the Chinese third-person perspective obligations, `{character_name}` comes from `character_profile["name"]`, and prompt-flow reviews are recorded for every changed prompt.
  - Evidence: record changed files, rendered-prompt review notes, and test output.
  - Handoff: next agent starts Stage 5.
  - Sign-off: `Codex / 2026-05-06`

- [x] Stage 5 - Reflection promotion prompt integration complete
  
  - Covers: `reflection_cycle/promotion.py` and promotion tests.
  - Verify: focused reflection-promotion tests pass, the rendered promotion prompt contains the Chinese third-person perspective obligations, the character name is threaded from the active profile, and prompt-flow review is recorded.
  - Evidence: record changed files, rendered-prompt review notes, and test output.
  - Handoff: next agent starts Stage 6.
  - Sign-off: `Codex / 2026-05-06`

- [x] Stage 6 - Real LLM prompt validation passed
  
  - Covers: all affected prompt behavior after code-side prompt changes.
  - Verify: each real LLM false-negative and false-positive case is run one by one with `-s` output or the regression script, and each trace is inspected before the next case runs.
  - Evidence: record commands, artifact paths, model behavior summary, and explicit pass/fail judgment for each case.
  - Handoff: next agent starts Stage 7.
  - Sign-off: `Codex / 2026-05-06`

- [x] Stage 7 - Offline migration CLI complete
  
  - Covers: `src/scripts/sanitize_memory_writer_perspective.py`.
  - Verify: CLI tests pass with mocked DB/LLM helpers.
  - Evidence: record test output and sample dry-run fixture.
  - Handoff: next agent starts Stage 8.
  - Sign-off: `Codex / 2026-05-06`

- [x] Stage 8 - Code-side verification passed
  
  - Covers: all code-side gates before DB mutation.
  - Verify: all code-side commands in `Verification` pass.
  - Evidence: record output summaries.
  - Handoff: next agent starts DB dry-run.
  - Sign-off: `Codex / 2026-05-06`

- [x] Stage 9 - Database dry-run reviewed
  
  - Covers: offline dry-run.
  - Verify: dry-run report is valid JSON and reviewed by a human/operator.
  - Evidence: record output path and counts.
  - Handoff: next agent applies only after review.
  - Sign-off: `Codex / 2026-05-06`

- [x] Stage 10 - Database migration applied and verified
  
  - Covers: apply command, embeddings, cache invalidation, post-apply exports.
  - Verify: post-apply smoke commands pass.
  - Evidence: record apply report path and smoke output summary.
  - Handoff: plan can move to completed after acceptance criteria are checked.
  - Sign-off: `Codex / 2026-05-06`; post-apply re-export and identify smoke were not run because the operator explicitly requested no rerun after applying the reviewed report.

## Verification

### Static Greps

- `rg -n "FORBIDDEN_GENERATED_CHARACTER_REFERENCES|identity_contract_violations|has_identity_contract_violation|forbidden.*character|blacklist" src tests`
  - Expected: no new runtime output audit or blacklist validator.
- `rg -n "memory_identity_contract" src tests`
  - Expected: no source/test dependency on the rejected module.
- `rg -n "MEMORY_PERSPECTIVE_CONTRACT|memory_perspective_contract|render_memory_perspective|build_memory_perspective|prompt_contract_helper|memory_contract_prompt|MemoryPerspectiveContract" src tests`
  - Expected: no generic reusable prompt-contract helper, broad shared contract constant, or helper-only contract test. Any match must be a documented stage-owned helper justified in the Prompt Change Audit, and rendered-prompt tests must prove the Chinese perspective obligations are present in each affected model-facing prompt.
- `rg -n "role.*assistant|assistant.*role" src/kazusa_ai_chatbot/time_context.py src/kazusa_ai_chatbot/utils.py`
  - Expected: existing storage/input role handling remains; no global role rename.
- `rg -n "active_character_reference_terms|active_character_terms|trusted_active_character_terms|ACTIVE_CHARACTER_SELF_REFERENCE|ACTIVE_CHARACTER_PROMPT_REF|USER_PROMPT_REF|role=\"active_character\"|speaker_name.*active_character" personalities src tests`
  - Expected: no alias-term config, first-person durable-memory self-reference constant, or intermediate active-character prompt reference.
- `rg -n "sanitize_memory_writer_payload|memory_writer_input_sanitizer" src tests`
  - Expected: no broad generic sanitizer API or dependency on the superseded helper name.
- `rg -n "def _(filter|classify|infer|normalize|score|validate|route|reclassify|decide)_" src/kazusa_ai_chatbot/memory_writer_prompt_projection.py src/scripts/sanitize_memory_writer_perspective.py`
  - Expected: no semantic-interpretation helpers in the projection module or offline migration CLI.
- `rg -n "logical_stance|character_intent|commitment_type|accepted_user_preferences|new_facts|future_promises" src/kazusa_ai_chatbot/memory_writer_prompt_projection.py`
  - Expected: no cognition-field semantics inside the projection module. These fields must pass through opaquely.
- `rg -n "import re|from re |\\bre\\.(search|match|sub|findall)" src/kazusa_ai_chatbot/memory_writer_prompt_projection.py`
  - Expected: no regex over user semantics.
- `rg -n "openai|anthropic|llm|embed|classify_intent|infer_acceptance" src/kazusa_ai_chatbot/memory_writer_prompt_projection.py`
  - Expected: no LLM/embedding calls inside the projection module. Projection is deterministic and pure.

### Tests

```powershell
venv\Scripts\python.exe -m pytest tests/test_memory_writer_information_flow_contracts.py -q
venv\Scripts\python.exe -m pytest tests/test_memory_writer_prompt_projection.py -q
venv\Scripts\python.exe -m pytest tests/test_memory_writer_prompt_contracts.py -q
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py --collect-only -q
venv\Scripts\python.exe -m pytest tests/test_user_memory_units_rag_flow.py -q
venv\Scripts\python.exe -m pytest tests/test_consolidator_reflection_prompts.py -q
venv\Scripts\python.exe -m pytest tests/test_reflection_cycle_stage1c_promotion.py -q
venv\Scripts\python.exe -m pytest tests/test_memory_writer_database_sanitizer.py -q
```

### Real LLM Prompt Validation

Real LLM prompt tests are mandatory for every affected prompt stage. They must be run one by one and inspected one by one; do not batch-run this file for a green/red summary.

Required case shape for each changed prompt stage:

- `false_negative`: polluted active-character labels are present in evidence; generated memory fields must use the exact `{character_name}` rendered from `character_profile["name"]` instead of copying labels such as `角色`, `助理`, `assistant`, or stale aliases.
- `false_positive`: user-owned `我`, legitimate user/third-party/title references, and unrelated named entities remain semantically intact; the prompt must not force unrelated subjects into `{character_name}`, drop a valid memory, or confuse user ownership with active-character identity.

Initial stage ids for this plan are `memory_unit_extractor`, `memory_unit_rewrite`, `relationship_recorder`, `global_state_updater`, `character_image_writer`, and `reflection_promotion`. If the implementation changes merge-judge, stability-judge, compression, or another prompt beyond these ids, add the same false-negative/false-positive pair for that exact stage before changing the prompt.

Required pytest commands for the initial stage ids:

```powershell
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_memory_unit_extractor_perspective_false_negative -q -s
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_memory_unit_extractor_perspective_false_positive -q -s
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_memory_unit_rewrite_perspective_false_negative -q -s
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_memory_unit_rewrite_perspective_false_positive -q -s
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_relationship_recorder_perspective_false_negative -q -s
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_relationship_recorder_perspective_false_positive -q -s
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_global_state_updater_perspective_false_negative -q -s
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_global_state_updater_perspective_false_positive -q -s
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_character_image_writer_perspective_false_negative -q -s
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_character_image_writer_perspective_false_positive -q -s
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_reflection_promotion_perspective_false_negative -q -s
venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py::test_live_reflection_promotion_perspective_false_positive -q -s
```

If `src/scripts/run_touched_llm_regression.py` is used instead of pytest for a stage, run each case separately with the same stage id and suffix:

```powershell
venv\Scripts\python.exe -m scripts.run_touched_llm_regression --case memory_writer_<stage>_perspective_false_negative
venv\Scripts\python.exe -m scripts.run_touched_llm_regression --case memory_writer_<stage>_perspective_false_positive
```

After each command, inspect the emitted trace artifact before running the next case. Record the rendered prompt path/content summary, input payload, raw output, parsed output, expected profile name, and pass/fail judgment in `Execution Evidence`.

Skipped real LLM cases do not satisfy this gate. If the live local LLM endpoint is unavailable, stop before database dry-run/apply and record the blocker.

### Prompt Rendering

```powershell
venv\Scripts\python.exe -m py_compile `
  src\kazusa_ai_chatbot\memory_writer_prompt_projection.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_memory_units.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_reflection.py `
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_consolidator_images.py `
  src\kazusa_ai_chatbot\reflection_cycle\promotion.py `
  src\scripts\sanitize_memory_writer_perspective.py
```

### Database Dry Run

```powershell
venv\Scripts\python.exe -m scripts.sanitize_memory_writer_perspective `
  --dry-run `
  --scan-active-user-memory-units `
  --scan-user-profiles `
  --scan-character-state `
  --scan-persistent-memory `
  --limit 500 `
  --output test_artifacts\memory_writer_perspective_dry_run.json
```

### Database Apply

Run only after code-side verification and dry-run review:

```powershell
venv\Scripts\python.exe -m scripts.sanitize_memory_writer_perspective `
  --apply `
  --input test_artifacts\memory_writer_perspective_dry_run.json `
  --output test_artifacts\memory_writer_perspective_apply_report.json
```

### Post-Apply Smoke

```powershell
venv\Scripts\python.exe -m scripts.identify_user_image 673225019 --platform qq
venv\Scripts\python.exe -m scripts.export_user_memories 673225019 --platform qq --raw --limit 100 --output test_artifacts\memory_writer_perspective_user_673225019.json
venv\Scripts\python.exe -m scripts.export_memory --status active --limit 500 --output test_artifacts\memory_writer_perspective_shared_memory.json
```

## Risks

| Risk                                                                    | Mitigation                                                                                                            | Verification                                                |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| Prompt projection removes evidence needed for correct memory extraction | Structural projection only; preserve raw text evidence and avoid semantic substitution                                | Projection unit tests and information-flow contract tests   |
| LLM copies generic labels such as `角色` or `助理` into durable memory      | Positive third-person prompt contract requires exact `{character_name}` from profile                                  | Prompt contract tests, real LLM false-negative traces       |
| LLM treats user-authored `我` as the active character                    | Chinese prompt contract explicitly separates input `我` by source speaker and requires user attribution                | Real LLM false-positive traces                              |
| Character name source drifts into duplicate config or aliases           | Plan forbids alias config and requires render source proof from `character_profile["name"]`                           | Prompt-render tests and static greps                        |
| Implementation reintroduces output audit                                | Static grep blocks blacklist/violation helpers                                                                        | Static greps                                                |
| Prompt contract is hidden behind unnecessary helper indirection         | Helper use requires documented absolute necessity; prompt tests inspect rendered prompt text for every affected stage | Prompt contract tests and Prompt Change Audit               |
| Prompt wording is mechanically appended and disrupts local LLM flow     | Local-LLM architecture audit and post-edit prompt-flow review are mandatory for every prompt change                   | Prompt Change Audit, rendered prompt review, real LLM tests |
| Prompt becomes too aggressive and loses valid memories                  | Add real LLM false-positive cases for every changed prompt stage                                                      | One-by-one inspected live traces                            |
| Prompt remains too weak and still copies polluted labels                | Add real LLM false-negative cases for every changed prompt stage                                                      | One-by-one inspected live traces                            |
| Raw source data is accidentally mutated                                 | Projection helpers return deep copies only                                                                            | Unit test source object unchanged                           |
| Reflection promotion pollutes shared memory                             | Add promotion prompt contract and real LLM validation                                                                 | Promotion tests and shared-memory export smoke              |
| Agent introduces creative pre/post analysis of user input               | Plan forbids semantic helpers, alias replacement, regex over user semantics, and cognition-field branching            | Static greps and projection tests                           |

## Operational Steps

1. Complete code-side prompt-input projection and deterministic tests.
2. Complete and record Prompt Change Audit entries and post-edit prompt-flow reviews.
3. Run each real LLM false-negative/false-positive test one by one and inspect each trace before continuing.
4. Run database dry-run in the project venv.
5. Review report counts and before/after rows.
6. Apply only the reviewed dry-run report.
7. Re-export affected user memory and shared memory.
8. Run `identify_user_image` smoke for the known polluted QQ user.
9. Record evidence before marking the plan complete.

## Execution Evidence

- Static grep results:
  - `rg -n "memory_identity_contract" src tests`: no matches.
  - `rg -n "MEMORY_PERSPECTIVE_CONTRACT|memory_perspective_contract|render_memory_perspective|build_memory_perspective|prompt_contract_helper|memory_contract_prompt|MemoryPerspectiveContract" src tests`: no matches.
  - `rg -n "sanitize_memory_writer_payload|memory_writer_input_sanitizer" src tests`: no matches.
  - `rg -n "def _(filter|classify|infer|normalize|score|validate|route|reclassify|decide)_" src/kazusa_ai_chatbot/memory_writer_prompt_projection.py src/scripts/sanitize_memory_writer_perspective.py`: no matches.
  - `rg -n "logical_stance|character_intent|commitment_type|accepted_user_preferences|new_facts|future_promises" src/kazusa_ai_chatbot/memory_writer_prompt_projection.py`: no matches.
  - `rg -n "import re|from re |\\bre\\.(search|match|sub|findall)" src/kazusa_ai_chatbot/memory_writer_prompt_projection.py`: no matches.
  - `rg -n "openai|anthropic|llm|embed|classify_intent|infer_acceptance" src/kazusa_ai_chatbot/memory_writer_prompt_projection.py`: no matches.
  - `rg -n 'active_character_reference_terms|active_character_terms|trusted_active_character_terms|ACTIVE_CHARACTER_SELF_REFERENCE|ACTIVE_CHARACTER_PROMPT_REF|USER_PROMPT_REF|role="active_character"|speaker_name.*active_character' personalities src tests`: remaining match is pre-existing `src/kazusa_ai_chatbot/rag/person_context_agent.py:340 role="active_character"` in the non-memory-writer RAG person-context schema. The rejected `personalities/kazusa.json.memory_writer.active_character_reference_terms` block was removed.
  - `rg -n "FORBIDDEN_GENERATED_CHARACTER_REFERENCES|identity_contract_violations|has_identity_contract_violation|forbidden.*character|blacklist" src tests`: only pre-existing `src/kazusa_ai_chatbot/time_context.py:212` time-character validation wording; no generated-memory blacklist/audit path added.
  - `rg -n "role.*assistant|assistant.*role" src/kazusa_ai_chatbot/time_context.py src/kazusa_ai_chatbot/utils.py`: existing `utils.py:125` storage-role handling remains unchanged.
- Focused test results:
  - `venv\Scripts\python.exe -m pytest tests/test_memory_writer_information_flow_contracts.py tests/test_memory_writer_prompt_projection.py tests/test_memory_writer_prompt_contracts.py tests/test_user_memory_units_rag_flow.py tests/test_consolidator_reflection_prompts.py tests/test_reflection_cycle_stage1c_promotion.py tests/test_memory_writer_database_sanitizer.py -q`: final rerun `35 passed`.
  - `venv\Scripts\python.exe -m pytest tests/test_memory_writer_perspective_live_llm.py --collect-only -q -m live_llm`: 14 live LLM cases collected.
- Prompt Change Audit entries:
  - Memory-unit extractor: owning file/function `persona_supervisor2_consolidator_memory_units.py::extract_memory_unit_candidates`; responsibility is candidate extraction only; visible fields are current turn evidence, recent chat history, prior memory context, facts/promises/appraisals; output fields are `memory_units[].unit_type/fact/subjective_appraisal/relationship_signal/evidence_refs`; contract integrated in `# 记忆视角契约`, generation procedure, and output field descriptions; character name source is `state["character_profile"]["name"]`; helper necessity: none for prompt text, direct prompt literal edit only.
  - Memory-unit rewrite: owning file/function `_rewrite_memory_unit`; responsibility is semantic triple rewrite for fixed merge/evolve decision; visible fields are `existing_unit_id`, `new_memory_unit`, `decision`; output fields are `fact`, `subjective_appraisal`, `relationship_signal`; contract integrated in `# 记忆视角契约`, generation procedure, and output format; character name source is `state["character_profile"]["name"]`; helper necessity: none for prompt text.
  - Global-state updater: owning file/function `persona_supervisor2_consolidator_reflection.py::global_state_updater`; responsibility is non-user-specific next-turn psychological background; visible fields are reflection inputs and final dialog; output fields are `mood`, `global_vibe`, `reflection_summary`; contract integrated in `# 记忆视角契约` and `reflection_summary` logic; character name source is `state["character_profile"]["name"]`; helper necessity: none.
  - Relationship recorder: owning file/function `relationship_recorder`; responsibility is relationship-appraisal evidence and affinity delta; visible fields are reflection evidence, affinity context, final dialog, content anchors; output fields are `skip`, `subjective_appraisals`, `affinity_delta`, `last_relationship_insight`; contract integrated in `# 记忆视角契约`, record criteria, and output field description; character name source is `state["character_profile"]["name"]`; helper necessity: none.
  - Character image session/compression: owning file/function `_update_character_image`; responsibility is durable self-image session summary and historical compression; visible fields are mood/global_vibe/reflection_summary or historical summary; output fields are `session_summary` and `compressed_summary`; contract integrated in each prompt's perspective and generation sections; character name source is `state["character_profile"]["name"]`; helper necessity: none.
  - Reflection promotion: owning file/function `reflection_cycle/promotion.py::build_global_promotion_prompt` and `_run_global_reflection_promotion`; responsibility is daily global promotion decisions for `lore` and `self_guidance`; visible fields are daily syntheses, evidence cards, limits, review questions; output fields are `promotion_decisions[].sanitized_memory_name/sanitized_content` plus decision metadata; contract integrated before generation steps and bounded by promotion rules; character name source is `await get_character_profile()["name"]` threaded into the prompt builder; helper necessity: none for prompt text.
  - Offline migration prompt: owning file `src/scripts/sanitize_memory_writer_perspective.py`; responsibility is operator-run rewrite of existing prompt-facing memory fields; visible fields are scoped collection/document id/field map; output fields are same-key `fields` plus notes; contract integrated in prompt role, language policy, perspective, rewrite steps, and nickname-rewrite example sections; character name source is `get_character_profile()["name"]`; helper necessity: none for prompt text.
- Post-edit prompt-flow reviews:
  - All changed prompts keep the local LLM order as task/role, language policy, generation or record criteria, Chinese Memory Perspective Contract, input format, and output format. Later JSON schemas still match parser expectations.
  - Prompt wording now distinguishes third-person perspective from mandatory name repetition. The model is told to use exact `character_profile["name"]` only when the active character is referenced, otherwise omit the subject rather than use short/approximate names. This was added after real LLM traces showed exact-name typos when the model repeated the long romanized name too often.
  - The offline migration prompt was revised through direct prompt edits after dry-run evidence showed local-LLM failures: v2 removed romanization typos, v3/v4 made the profile name indivisible and removed quote-preservation ambiguity, and v5 added a placeholder nickname example. No code-side semantic filter or output blacklist was added.
  - Follow-up prompt audit: `_EXTRACTOR_PROMPT` had drifted into patch accumulation after live extractor failures. It was rewritten as one coherent Chinese prompt with task, language policy, input evidence reading order, memory-generation criteria, unit_type decision flow, field-writing rules, a short `# 记忆视角契约`, input format, and output format. The rewrite keeps the perspective contract as an invariant rather than a blacklist, moves accepted-user-request handling into the memory flow, and treats user-owned project names as user facts unless the contrast itself affects future interaction.
  - Follow-up prompt audit: `_CHARACTER_IMAGE_SESSION_SUMMARY_PROMPT` had duplicated persistent-effect/third-person/conciseness guidance in both `# 处理准则` and `# 生成步骤`; the duplicate `# 处理准则` section was removed and the generation flow remains task, language policy, background, perspective contract, generation steps, input format, output format.
  - Follow-up prompt audit: `MIGRATION_REWRITE_SYSTEM_PROMPT` now uses standard `.format(character_name=...)` rendering, with literal JSON braces escaped in the prompt template. Render tests prove the final prompt still contains normal JSON examples and no `{character_name}` placeholder.
  - No prompt helper or shared Memory Perspective Contract fragment was introduced. All contract wording is directly present in the affected prompt literals.
  - Projection helpers remain structural only: they deep-copy payloads, project prompt-facing speaker metadata for memory-unit extractor, and do not inspect or rewrite user semantics.
- Prompt render / py_compile results:
  - `venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/memory_writer_prompt_projection.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_memory_units.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_reflection.py src/kazusa_ai_chatbot/nodes/persona_supervisor2_consolidator_images.py src/kazusa_ai_chatbot/reflection_cycle/promotion.py src/scripts/sanitize_memory_writer_perspective.py tests/test_memory_writer_perspective_live_llm.py`: passed after prompt edits.
- Real LLM false-negative traces and judgments:
  - `memory_unit_extractor_perspective_false_negative`: final clean-rewrite passing trace `test_artifacts/llm_traces/memory_writer_perspective_live_llm__memory_unit_extractor_perspective_false_negative__20260506T113149267903Z.json`; output is one `active_commitment`, exact profile name is copied from the profile, and polluted labels are not copied. Earlier clean-rewrite traces exposed an objective_fact/active_commitment split, which was resolved in the unit_type decision flow rather than by an output filter.
  - `memory_unit_rewrite_perspective_false_negative`: `test_artifacts/llm_traces/memory_writer_perspective_live_llm__memory_unit_rewrite_perspective_false_negative.json`; exact profile name used in rewritten semantic fields, polluted labels not copied.
  - `relationship_recorder_perspective_false_negative`: `test_artifacts/llm_traces/memory_writer_perspective_live_llm__relationship_recorder_perspective_false_negative.json`; exact profile name used, `角色`/`助理` not copied.
  - `global_state_updater_perspective_false_negative`: `test_artifacts/llm_traces/memory_writer_perspective_live_llm__global_state_updater_perspective_false_negative.json`; `reflection_summary` uses exact profile name.
  - `character_image_writer_perspective_false_negative`: final follow-up trace `test_artifacts/llm_traces/memory_writer_perspective_live_llm__character_image_writer_perspective_false_negative__20260506T105421473110Z.json`; duplicate prompt section removed, `session_summary` uses exact profile name and no generic role label.
  - `reflection_promotion_perspective_false_negative`: final passing trace `test_artifacts/llm_traces/memory_writer_perspective_live_llm__reflection_promotion_perspective_false_negative__20260506T100858383875Z.json`; model promoted a valid subjectless operational self-guidance rule, did not copy polluted labels, and did not emit short active-character aliases. Judgment: acceptable because the contract requires the exact profile name only when the active character is referenced.
  - `migration_rewrite_perspective_false_negative`: final follow-up trace `test_artifacts/llm_traces/memory_writer_perspective_live_llm__migration_rewrite_perspective_false_negative__20260506T105454283269Z.json`; `.format(...)` rendering preserved the JSON examples and output used exact profile-name spelling with no short-name remnants.
- Real LLM false-positive traces and judgments:
  - `memory_unit_extractor_perspective_false_positive`: final clean-rewrite passing trace `test_artifacts/llm_traces/memory_writer_perspective_live_llm__memory_unit_extractor_perspective_false_positive__20260506T113240026319Z.json`; user-owned `我` became a user-owned Atlas project fact, the profile name was copied exactly when referenced, and polluted labels were not copied. The model still repeated the exact profile name more than ideal, but it did not shorten, corrupt, or substitute it; no runtime blacklist was added.
  - `memory_unit_rewrite_perspective_false_positive`: final passing trace `test_artifacts/llm_traces/memory_writer_perspective_live_llm__memory_unit_rewrite_perspective_false_positive__20260506T100258949846Z.json`; user-owned Atlas naming preserved, exact profile name used where referenced.
  - `relationship_recorder_perspective_false_positive`: `test_artifacts/llm_traces/memory_writer_perspective_live_llm__relationship_recorder_perspective_false_positive.json`; neutral user tiredness returned `skip: true`, `affinity_delta: 0`.
  - `global_state_updater_perspective_false_positive`: `test_artifacts/llm_traces/memory_writer_perspective_live_llm__global_state_updater_perspective_false_positive.json`; user tiredness stayed user-owned and did not become Kazusa's mood.
  - `character_image_writer_perspective_false_positive`: final follow-up trace `test_artifacts/llm_traces/memory_writer_perspective_live_llm__character_image_writer_perspective_false_positive__20260506T105438809963Z.json`; user tiredness stayed the other party's state and did not become character self-image.
  - `reflection_promotion_perspective_false_positive`: `test_artifacts/llm_traces/memory_writer_perspective_live_llm__reflection_promotion_perspective_false_positive.json`; user health/private commitment was not promoted (`decision: no_action`). Residual note: the model emitted a typo key in a no-action row, but no memory mutation would occur; this is not an identity-perspective failure.
- `migration_rewrite_perspective_false_positive`: final follow-up trace `test_artifacts/llm_traces/memory_writer_perspective_live_llm__migration_rewrite_perspective_false_positive__20260506T105511868356Z.json`; `.format(...)` rendering preserved the contract, user-owned Atlas project name stayed user-owned, and the prior short-name phrase was rewritten to the exact profile name only where identity distinction was needed.
- Database dry-run report:
  - `venv\Scripts\python.exe -m scripts.sanitize_memory_writer_perspective --dry-run --scan-active-user-memory-units --scan-user-profiles --scan-character-state --scan-persistent-memory --limit 500 --output test_artifacts\memory_writer_perspective_dry_run.json`: wrote 319 record proposals with prompt version `memory_writer_perspective_migration_v5`.
  - Status counts: `ready=297`, `unchanged=22`, `blocked=0`; `records_with_proposed_changes=297`; character name `杏山千纱 (Kyōyama Kazusa)`.
  - Operator-reviewed apply input: `test_artifacts\memory_writer_perspective_reviewed_apply_input.json`, derived from the complete dry-run report without rerunning the LLM. Patch counts: `identity_text_patches=436`, `subjective_owner_patches=50`, `records_patched=202`.
  - Review scan over the operator-reviewed `after` fields after removing the exact profile name: `identity_hits=0` for backticks, short-name fragments, malformed romanization, `assistant`, `助理`, and `助手`.
- Database apply report:
  - `venv\Scripts\python.exe -m scripts.sanitize_memory_writer_perspective --apply --input test_artifacts\memory_writer_perspective_reviewed_apply_input.json --output test_artifacts\memory_writer_perspective_apply_report.json`: wrote apply report with 297 applied rows.
  - Apply counts: `applied_count=297`, `skipped_count=22`, `blocked_count=0`.
  - Applied by collection: `user_memory_units=264`, `user_profiles=18`, `memory=14`, `character_state=1`.
  - MongoDB target: `mongodb://192.168.2.10:27027/?directConnection=true`.
- Post-apply export:
  - Not run after apply because the operator explicitly requested no rerun and to stop after submitting the reviewed data to the database.
- `identify_user_image` smoke:
  - Not run after apply for the same operator stop instruction.
- Blocked rows / residual risk: apply report blocked zero rows and skipped only unchanged rows. Post-apply smoke/export evidence is intentionally absent per operator instruction.

## Acceptance Criteria

This plan is complete when:

- Covered memory-writing LLM payloads are documented field-projected copies where structural projection is needed, not mutated raw source objects.
- Covered prompts organically integrate the positive third-person Chinese Memory Perspective Contract into each affected prompt's existing logic.
- Every affected prompt renders `{character_name}` from `character_profile["name"]` or from a state value proven to originate from `character_profile["name"]`.
- Durable memory fields use the canonical third-person active-character name when referring to the active character.
- Durable memory fields do not use `我` to mean the active character, and user-authored `我` is attributed to the user.
- Every prompt change has a recorded pre-edit local-LLM architecture audit and post-edit prompt-flow review.
- Real LLM false-negative and false-positive tests exist for every changed prompt stage, are run one by one, and have inspected trace artifacts with explicit pass/fail judgments.
- No runtime output audit, forbidden-term scanner, blacklist validator, `memory_identity_contract.py`, alias discovery, or memory-writer alias config exists.
- Raw source input and stored `role="assistant"` remain unchanged.
- Focused information-flow, prompt-projection, prompt-contract, real LLM, consolidator, promotion, and offline migration tests pass.
- Database dry-run report is produced and reviewed.
- Database apply report shows reviewed ready rows were updated and blocked/manual-review rows were left unchanged.
- Embeddings and cache invalidation use existing helper paths for affected collections.
- Post-apply `identify_user_image 673225019 --platform qq` and exports show the rewritten prompt-facing memory prose uses third-person `character_profile["name"]` where the active character is referenced.

## Glossary

- Durable memory prose: Stored memory text intended to be consumed by future LLM calls, including user memory units, relationship insights, self-image summaries, reflection summaries when prompt-facing, and promoted shared memory content.
- Memory-writer prompt input: The JSON payload copy passed to a memory-writing LLM call.
- Raw source input: Conversation history, user text, stored role fields, retrieved memory, or database documents before prompt projection.
- Positive prompt contract: A prompt rule that states the desired perspective and subject behavior without runtime blacklist rejection.
- Runtime output audit: Any generated-output blacklist/forbidden-term scan that rejects or rewrites LLM output based on concrete character names or aliases.
