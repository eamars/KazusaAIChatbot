# Cognition Core V2 Intra-Turn Transition Coherence Bugfix Plan

## Summary

- Goal: eliminate unsupported within-turn stance reversals in Cognition Core
  V2 while preserving embarrassment, defensiveness, teasing, indirectness,
  genuine refusal, and reasoned changes of mind.
- Plan class: high_risk_migration, because this is a big-bang replacement of
  response-path prompt, public surface, repair, verifier, and persistence
  handoff behavior.
- Status: in_progress.
- Target revision audited:
  `cognition_core_v2@ba433e9be4805bb88b992419db1c99a6e376dc4a`.
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `debug-llm`, `test-style-and-execution`, `py-style`, and `cjk-safety`.
- Overall cutover strategy: bigbang V2-only contract replacement.
- Highest-risk areas: surface semantic ownership, dialog-verifier recall,
  repair ownership, accepted-surface propagation, and progress feedback.
- Acceptance criteria: the two captured failures and three semantic variants
  score `2` for transition coherence; no newly generated post-fix dialog
  scores `0`; seeded score-`0` negative fixtures are rejected by the verifier;
  refusal and supported-change controls retain their intended meanings; the
  old style contract has no source or test consumers; and the normal response
  path adds no LLM call.
- Execution authorization: the user authorized parent-only execution on
  2026-07-25T14:18:02Z. No subagent may participate in implementation or
  review.

## Context

The source reference is
`C:/workspace/kazusa_ai_chatbot_dev2/development_plans/reference/designs/cognition_core_v2_intra_turn_transition_quality_handover.md`.
It is evidence, not an execution contract. This dedicated plan owns the
follow-up and does not add scope to
`development_plans/active/bugfix/cognition_core_v2_baseline_regression_hardening_plan.md`.

Current-branch reproduction is recorded in
`test_artifacts/llm_reviews/cognition_core_v2_transition_quality_repro.md`
with raw artifacts under
`test_artifacts/llm_traces/cognition_core_v2_transition_quality_repro__*.json`.
The evidence establishes:

1. The independent V2 style owner can add self-correction and transition
   structure to an accepting or bantering semantic plan.
2. The current repair owner can replace content correctly while the caller
   preserves the style instruction that caused the rejection.
3. Dialog rendered the contradictory repaired surface as blame/resistance
   followed by reluctant participation through an empty inevitability bridge.
4. The current semantic-fidelity verifier accepted both the captured known-bad
   candidate and the newly generated score-`0` candidate.
5. Conversation progress converted the accepted bad dialog into an emotional
   trajectory and future progression guidance.

The reproduced bad shape is:

```text
literal rejection, accusation, or denial
  -> no new fact, motive, condition, concession, or constraint
  -> acceptance or compliance
```

Phrases such as “since you said so,” “since you asked,” “but,” and “it cannot
be helped” are evidence only when they are the entire bridge. They remain
valid wording when the turn supplies a real cause. This plan therefore changes
ownership and verification contracts, not individual phrases.

The current V2 ownership conflict is:

```text
selected cognition intention and bid
  -> content stage: accepting semantic plan
  -> independent style stage: denial/self-correction/rapid-compliance form
  -> structural merge
  -> dialog receives both as authority
  -> semantic verifier sees percepts but not authoritative surface semantics
  -> repair preserves rejected style
  -> accepted bad dialog can enter conversation progress
```

The target V2 ownership flow is:

```text
selected cognition intention and bid
  -> one content owner atomically produces:
       content plan + requirements + bounded delivery profile
  -> independent preference owner produces only real boundaries/addressee
  -> structural merge
  -> dialog renders semantics first and delivery second
  -> semantic verifier compares dialog with authoritative surface semantics
  -> one full content/delivery replacement after a hard rejection
  -> only the accepted surface and accepted dialog reach progress recording
```

Mainline code, compatibility with the removed V2 shape, model replacement,
and cross-branch behavior are outside this plan. V2 is the sole target because
it will replace mainline.

## Mandatory Skills

- `development-plan`: load before plan approval, execution, checklist
  sign-off, lifecycle updates, and final closure.
- `local-llm-architecture`: load before changing the V2 prompt, model payload,
  public contract, repair path, dialog verifier, or response-path call graph.
- `debug-llm`: load before every live LLM run and before writing the
  human-readable before/after quality review.
- `test-style-and-execution`: load before changing or running tests; execute
  live LLM cases one at a time and inspect each durable trace before the next.
- `py-style`: load before reviewing or editing any Python file.
- `cjk-safety`: load before editing Python prompt strings containing Chinese.

## Mandatory Rules

- Use `venv\Scripts\python.exe` for Python and pytest commands.
- Read `git status --short`, `README.md`, `docs/HOWTO.md`, this entire plan,
  the V2/node/progress READMEs, and every directly affected source/test file
  before production edits.
- Preserve user changes and the pre-existing untracked
  `test_runs/controlled/C07/1/r7a5e422b1304/` directory.
- Never read `.env` during deterministic work. Live LLM tests use the existing
  project test configuration without printing secrets.
- After automatic context compaction, reread this entire plan before
  implementation, verification, handoff, or final reporting.
- After signing off a major progress stage, reread this entire plan before
  starting the next stage.
- Establish focused failing contracts and record the current baseline before
  editing production code.
- Run real LLM cases one at a time. Inspect the raw input, raw output, parsed
  output, intermediate surface, verifier verdict, and rubric score before
  starting the next case.
- Scripts and tests may emit raw JSON, logs, and traces. The parent agent
  authors the Markdown quality review after inspecting the evidence.
- Real LLM schema validity and pytest success are harness gates only. The
  parent applies the `0/1/2` transition rubric to the actual dialog.
- LLM stages own semantic judgment. Deterministic code owns exact schemas,
  bounds, field projection, accepted-candidate propagation, call caps,
  persistence eligibility, and trace settlement.
- Current cognition intention, admitted bid, content plan, content
  requirements, visible boundaries, role operations, and permitted action
  results remain authoritative. Delivery fields cannot override them.
- Character `logic`, `defense`, `quirks`, and `taboos` enter V2 cognition as
  bounded trusted personality-judgment constraints. Text delivery receives
  only `tempo` plus deterministic linguistic-texture descriptors.
- Use `parse_llm_json_output(...)` as the canonical JSON parser. Preserve the
  existing producing-stage structural replacement cap.
- Use triple-single-quoted prompt constants, named `.format(...)` only for
  process-stable values, and dynamic facts only in `HumanMessage`.
- Run prompt-render checks in addition to `py_compile`.
- Keep Chinese explanatory prompt prose and generated free text in Simplified
  Chinese. Preserve schema keys, enum values, code, URLs, proper names, and
  user quotations exactly.
- Add no phrase blacklist, keyword classifier, deterministic semantic scorer,
  post-generation rewrite, sentiment suppression, model-specific branch,
  compatibility alias, fallback mapper, dual schema, new agent, or new retry.
- Preserve bracket-encapsulated-content behavior. Record repetition separately
  from transition coherence.
- Preserve the existing role-direction and surface-integrity verifier
  ownership. This plan extends semantic fidelity only with surface-semantic
  authority.
- Before completion, merge, lifecycle closure, or sign-off, the parent pauses
  work, rereads this plan and the complete diff, runs the `Independent Code
  Review` rubric, remediates in-scope findings, reruns affected gates, and
  records the result in `Execution Evidence`.
- The one allowed subagent is used only for the independent quality review of
  this plan before delivery. Under the user's explicit direction, the parent
  performs all later implementation, testing, evidence review, code review,
  remediation, and sign-off without additional subagents.

## Must Do

- Remove the independent V2 style LLM stage and its service configuration.
- Replace `style_guidance` with the exact `DeliveryProfileV2` contract.
- Replace `character_voice_context` with the exact
  `character_expression_context` and `visual_character_context` fields.
- Move `personality_brief.logic`, `defense`, `quirks`, and `taboos` into the
  V2 cognition-owned `character_constraints.personality_judgment` projection.
- Expose only `personality_brief.tempo` and deterministic linguistic-texture
  descriptions to unified text content through
  `character_expression_context`; keep the current full bounded profile
  projection available only to the terminal visual owner through
  `visual_character_context`.
- Make one content-stage response own `content_plan`,
  `content_requirements`, and `delivery_profile` atomically.
- Keep preference planning limited to `visible_boundaries` and
  `addressee_plan`.
- Make the semantic-fidelity verifier compare candidate dialog with the
  selected surface intent, content plan, content requirements, and visible
  boundaries.
- Add unsupported within-turn stance reversal to semantic fidelity's narrow
  hard-error taxonomy without using phrase rules.
- Regenerate content, requirements, delivery profile, boundaries, and
  addressee together after a verified hard rejection.
- Preserve selected intent, canonical input, role operations, permitted action
  results, runtime limits, and the two-candidate dialog cap during repair.
- Return the accepted post-repair `TextSurfaceOutputV2` from dialog and store
  that accepted output in the persona graph state.
- Project only the accepted visible dialog and accepted semantic surface into
  conversation-progress recording; omit delivery profile data from the
  progress recorder's content-plan packet.
- Keep the rejected dialog and rejected surface out of the semantic repair and
  repair-render payloads; retain them only in protected trace evidence.
- Update V2 source, connectors, downstream consumers, docs, deterministic
  fixtures, and live tests in one big-bang change.
- Run every verification gate in this plan and record its evidence.

## Deferred

- Do not change mainline or create a mainline/V2 compatibility path.
- Do not modify the configured LLM models or sampling parameters.
- Do not redesign V2 cognition branching, bid collapse, action selection,
  role-direction verification, surface-integrity verification, visual
  directives, conversation-progress schema, or consolidation.
- Do not add transition-quality fields to persistent conversation-progress
  documents.
- Do not make the progress recorder a dialog-quality classifier.
- Do not suppress valid speech, weaken character judgment, or bias the
  character toward acceptance.
- Do not define a policy for brackets or action narration.
- Do not perform general repetition cleanup.
- Do not refactor unrelated prompts, tests, profile schemas, or graph nodes.
- Do not change the existing in-progress baseline-hardening plan.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
| --- | --- | --- |
| V2 cognition constraints | bigbang | Require exact bounded `personality_judgment` fields; retain no old constraint shape. |
| V2 text-surface input | bigbang | Replace `character_voice_context` with exact `character_expression_context` and `visual_character_context`; retain no alias. |
| V2 text-surface output | bigbang | Delete `style_guidance`; require exact `delivery_profile`. |
| V2 services | bigbang | Delete `style_config` and the `v2_surface_style` call. |
| V2 repair | bigbang | Replace the old four-field repair and style preservation with a full five-owner-field replacement. |
| Dialog consumer | bigbang | Read only the new surface shape and return the accepted surface. |
| Progress consumer | bigbang | Consume accepted surface semantics and no delivery field. |
| Tests | bigbang | Rewrite all V2 fixtures and assertions to the new exact contract; retain no old-shape fixture. |
| Mainline | keep | Make no mainline change and add no bridge to it. |
| Persistent data | keep | Perform no migration or backfill; the changed contracts are response-local. |

Cutover enforcement:

- Rewrite or delete every old V2 reference in the same implementation.
- Treat any surviving old source/test symbol as a failed cutover.
- Keep no runtime feature flag or fallback to the old style stage.
- Revert the full V2 patch atomically if rollback is required; never run a
  mixed input/output contract.
- Any change to this policy requires user approval before implementation.

## Target State

- V2 normal text-surface planning uses two parallel LLM calls: unified content
  and preference.
- The unified content call receives authoritative intention/bid semantics,
  affect, relationship, interaction style, permitted results, and bounded
  character expression context containing only tempo and linguistic texture.
  It returns content and delivery dimensions in one exact object.
- Semantic personality judgment is resolved upstream: V2 cognition receives
  bounded `logic`, `defense`, `quirks`, and `taboos` constraints before it
  selects intention and bids.
- Delivery profile describes only lexical register, sentence shape, rhythm,
  hesitation, and punctuation. Refusal, accusation, acceptance, compliance,
  concession, and stance transitions belong in content semantics.
- Dialog first preserves the authoritative semantic fields, then applies the
  delivery profile to wording and cadence.
- A genuine within-turn change of stance is valid only when the authoritative
  surface semantics contain a fact, motive, condition, concession, or
  constraint that explains it.
- Semantic fidelity rejects a rendered opposite-stance transition when the
  authoritative surface supplies no cause. It accepts genuine refusal,
  negotiation, conditions, coherent teasing, and supported changes of mind.
- Repair has one semantic owner and replaces every field that could have
  produced the contradiction. It preserves only canonical truth fields.
- The persona graph retains the surface that produced the accepted dialog,
  including a repaired surface when repair succeeded.
- Rejected dialog and rejected surface candidates remain protected trace
  evidence only; neither is echoed into semantic surface repair or the second
  dialog render. Conversation progress receives the accepted dialog and
  accepted semantic plan.

## Design Decisions

| Topic | Decision | Rationale |
| --- | --- | --- |
| Primary fix boundary | Change V2 text-surface ownership. | The reproduced conflict is produced before dialog by independently merged content and style instructions. |
| Style agent | Remove the independent V2 style call. | Prompt-only narrowing still leaves a separate free-prose semantic producer; deletion removes the conflicting owner and reduces normal latency. |
| Content/delivery ownership | One LLM response produces both atomically. | The owner sees selected semantics and character expression together and cannot be merged with an independently chosen stance transition. |
| Delivery shape | Use five required bounded descriptor strings. | Stable keys keep the local model contract explicit without phrase lists or a second semantic vocabulary. |
| Preference owner | Retain it independently. | Real boundaries and addressee constraints are a distinct existing semantic responsibility and were not the reproduced producer. |
| Semantic personality | Add `character_constraints.personality_judgment` with `logic`, `defense`, `quirks`, and `taboos` to V2 cognition. | Cognition must resolve stance, boundaries, and character judgment before L3; semantic profile prose cannot remain a delivery instruction. |
| Text expression context | Give unified content only `tempo` and deterministic linguistic-texture descriptors. | These fields shape delivery without granting L3 raw semantic defense or taboo authority. |
| Visual personality | Preserve the current full bounded profile projection as `visual_character_context`, consumed only by terminal visual planning. | Visual behavior remains stable without exposing raw semantic profile prose to text preference or dialog. |
| Verifier authority | Give semantic fidelity authoritative surface semantics but exclude delivery profile. | The verifier can detect a candidate/content contradiction without treating delivery as semantic permission. |
| Repair | Regenerate all content, delivery, boundary, and addressee fields from canonical input plus verified issues. | Preserving the rejected style reproduced the bug; canonical truth remains stable while producer-owned fields are replaced. |
| Progress | Persist only the accepted dialog and semantic plan. | Rejected candidates are diagnostic evidence, not desired future progression. |
| Call count | Reduce normal text planning from three calls to two; retain one repair call. | The fix improves the bounded local-LLM path and adds no response-path latency stage. |
| Compatibility | Preserve none. | V2 will replace mainline and the user explicitly excluded compatibility work. |

## Contracts And Data Shapes

### Cognition personality judgment

Extend the exact V2 constraint snapshot:

```python
class PersonalityJudgmentV2(TypedDict):
    logic: str
    defense: str
    quirks: str
    taboos: str


class CharacterConstraintSnapshotV2(TypedDict):
    drives: dict[str, dict[str, Any]]
    standards: list[dict[str, Any]]
    meaning_state: dict[str, Any]
    personality_judgment: PersonalityJudgmentV2
```

Each personality field is required, non-empty, and at most 180 characters.
`build_cognition_input_from_global_state(...)` reads these four trusted static
fields from the validated character profile. `_project_constraints(...)`
projects them unchanged as bounded semantic descriptors. V2 appraisal and goal
cognition may use them to choose a character-consistent stance; deterministic
reducers never branch on their text. `personality_brief.tempo` remains outside
semantic constraints.

### Text-surface input

`TextSurfaceInputV2` keeps its existing fields except for this exact
replacement:

```python
class CharacterExpressionContextV2(TypedDict):
    tempo: str
    linguistic_texture: str


class TextSurfaceInputV2(TypedDict):
    schema_version: Literal["text_surface_input.v2"]
    # Existing episode, intention, bid, policy, affect, relationship,
    # action-result, runtime-limit, and interaction-style fields remain.
    character_expression_context: CharacterExpressionContextV2
    visual_character_context: str
```

`character_expression_context` accepts no extra keys. `tempo` is the validated
`personality_brief.tempo` string capped at 180 characters.
`linguistic_texture` is the existing deterministic projection of the ten
numeric linguistic-texture values capped at 1,000 characters. Unified content
receives this exact object; preference and dialog do not.

`visual_character_context` is required, non-empty, and at most 1,500
characters. It preserves the current full bounded profile projection and is
exposed only to terminal visual planning. Text content, preference, and dialog
do not receive it.

### Delivery profile

```python
class DeliveryProfileV2(TypedDict):
    lexical_register: str
    sentence_shape: str
    rhythm: str
    hesitation: str
    punctuation: str
```

Each field is required, non-empty, and at most 200 characters. The object
accepts no extra keys. Values describe delivery only; they do not authorize a
speech act or semantic stance.

### Text-surface output

```python
class TextSurfaceOutputV2(TypedDict):
    schema_version: Literal["text_surface_output.v2"]
    content_plan: str
    content_requirements: list[str]
    visible_boundaries: list[str]
    addressee_plan: list[str]
    delivery_profile: DeliveryProfileV2
    selected_surface_intent: str
    permitted_action_results: list[SemanticActionResultV2]
    runtime_capability_limits: NotRequired[list[str]]
```

`style_guidance` is invalid. Output validation remains structural and exact;
deterministic code does not interpret delivery strings semantically.
`visible_boundaries` and `addressee_plan` are each exact duplicate-free lists
of zero to eight strings, each at most 500 characters. The public validator and
stage validator enforce the same bounds.

### Unified content-stage output

The content stage returns exactly:

```python
{
    "content_plan": str,                 # non-empty, <= 1,000 chars
    "content_requirements": list[str],   # 1-8 unique, each <= 500 chars
    "delivery_profile": {
        "lexical_register": str,         # non-empty, <= 200 chars
        "sentence_shape": str,           # non-empty, <= 200 chars
        "rhythm": str,                   # non-empty, <= 200 chars
        "hesitation": str,               # non-empty, <= 200 chars
        "punctuation": str,               # non-empty, <= 200 chars
    },
}
```

The content prompt applies this positive procedure:

1. Establish the selected stance and content from intention, bids, current
   episode, affect, relationship, expression policy, interaction style,
   permitted action results, and runtime capability limits.
2. Put every refusal, acceptance, accusation, condition, concession, and
   change-of-mind reason in `content_plan` or `content_requirements`.
3. Describe only lexical and cadence realization in `delivery_profile`.
4. Use `character_expression_context` for tempo and linguistic realization
   while keeping the cognition-selected stance legible.

### Text-surface services

```python
@dataclass(frozen=True)
class TextSurfaceServicesV2:
    llm: LLMInvoker
    content_plan_config: LLMCallConfig
    preference_config: LLMCallConfig
```

There is no `style_config`, style prompt, style handler, or style validator.

### Repair

Change the public repair boundary to:

```python
async def repair_text_surface_planning(
    input_payload: TextSurfaceInputV2,
    verified_hard_issues: list[str],
    services: TextSurfaceServicesV2,
) -> TextSurfaceOutputV2:
```

The repair LLM receives canonical projected input, bounded verified issues, and
`character_expression_context`. It does not receive rejected delivery profile
or rejected surface semantics. It returns exactly the unified content-stage
shape plus `visible_boundaries` and `addressee_plan`. The caller reconstructs
selected intent, permitted results, and runtime limits from the validated
canonical input.

The second dialog render receives exactly:

```python
{
    "text_surface_output_v2": repaired_surface,
    "user_name": str,
    "repair_context": {
        "verified_hard_issues": list[str],
    },
}
```

Delete `repair_context.original_final_dialog`. The rejected dialog remains in
protected trace evidence and is not echoed to either repair producer.

### Semantic-fidelity verifier payload

Add this exact field beside the existing candidate, role frame, and projected
visible percepts:

```python
"authoritative_surface_semantics": {
    "selected_surface_intent": str,
    "content_plan": str,
    "content_requirements": list[str],
    "visible_boundaries": list[str],
}
```

Exclude `delivery_profile`, `permitted_action_results`, and
selection-required role fields. Surface integrity continues to own action
truth; role direction continues to own selection transfer and typed role
reversal.

The semantic verifier marks an unsupported transition as a hard error when:

```text
candidate opening stance opposes candidate final stance
AND authoritative surface semantics do not provide a new fact, motive,
condition, concession, or constraint that supports that change
```

The verifier does not reject hesitation, indirectness, embarrassment,
playfulness, a consistent refusal, or a surface-authorized change of mind.

The authoritative projection has an 11,000-character serialized cap. This
accommodates the declared maxima plus JSON keys and escaping:

```text
selected intent       <= 1,000
content plan          <= 1,000
requirements          <= 4,000
visible boundaries    <= 4,000
JSON/key headroom     <= 1,000
```

Candidate dialog has a 12,000-character total visible-text cap across all
messages. The complete semantic-verifier `HumanMessage` has a 50,000-character
serialized cap: current percepts <=24,000, authoritative semantics <=11,000,
candidate dialog <=12,000, and envelope/key/escaping headroom <=3,000.
Deterministic code truncates none of these fields.

If either the authoritative projection, candidate dialog, or complete payload
exceeds its cap, `_verify_dialog_semantic_fidelity(...)` records a protected
trace step with `parse_status="not_called_context_limit"` and `status="failed"`,
records a model-contract event with
`violation_kind="semantic_verifier_context_limit"`, and raises
`DialogVerifierContractError` with:

```text
error_code = "dialog_semantic_fidelity_context_limit"
stage = "dialog.semantic_fidelity"
safe checkpoint = "post_cognition_commit"
```

No semantic verifier call, dialog repair, delivery, or conversation-progress
recording occurs after this failure. The existing verifier structural attempt
cap remains unchanged for payloads inside the cap.

### Accepted-surface propagation

`dialog_generator(...)` returns both:

```python
{
    "final_dialog": list[str],
    "text_surface_output_v2": TextSurfaceOutputV2,
}
```

The surface is the original candidate when the first dialog passes and the
replacement surface when repair passes. `dialog_agent(...)` and
`call_action_subgraph(...)` propagate this accepted surface into
`GlobalPersonaState`. The progress packet contains:

```python
{
    "semantic_content": accepted_surface["content_plan"],
    "surface_intent": accepted_surface["selected_surface_intent"],
}
```

It contains no delivery profile. A dialog-compliance exhaustion raises before
progress recording, so rejected candidates have no progress consumer.

## LLM Call And Context Budget

Default context-window cap: 50,000 tokens. Character counts below are
conservative prompt-envelope limits, not token estimates.

| Call | Before | After | Context and cap | Latency and failure policy | Verification |
| --- | --- | --- | --- | --- | --- |
| V2 semantic appraisal and goal cognition / `COGNITION_LLM` | Existing calls with drives, standards, and meaning constraints | Same call counts with `personality_judgment` <=900 serialized chars | Existing 8,000-character appraisal and 24,000-character goal caps remain; supplemental appraisal context is dropped by existing policy before required constraints can fail. | No new call, retry, or model route. Required-context overflow keeps the existing typed failure. | Maximum-shape cognition projection and live character-judgment controls. |
| V2 style / `COGNITION_LLM` | 1 normal-path call, max 24,000-character surface payload | deleted | No context remains. | Removes one parallel producer and its structural retry. | Call-count test and static grep. |
| V2 unified content / `COGNITION_LLM` | 1 normal-path call, content-only output | 1 normal-path call, content plus five delivery fields | Existing 24,000-character surface payload cap; expression context is <=1,180 chars plus keys; output delivery budget <=1,000 chars. | Still parallel with preference; existing two structural attempts remain. | Prompt-render, contract, and live owner tests. |
| V2 preference / `COGNITION_LLM` | 1 normal-path call | unchanged 1 call | Existing 24,000-character cap; no raw character expression context. | Parallel with unified content; attempt cap unchanged. | Projection and handoff tests. |
| V2 dialog-compliance surface repair / `COGNITION_LLM` | 1 repair-path call replacing four fields while preserving style | 1 repair-path call replacing content, delivery, boundaries, and addressee | Existing 24,000-character cap; rejected surface prose is removed; verified issues remain bounded to 8 x 300 chars. | No new retry or call; still blocks only after first hard rejection. | Repair contract and live repair case. |
| V2 terminal visual / `COGNITION_LLM` | Optional 1 sibling call | unchanged optional 1 call | Existing 24,000-character cap; consumes renamed character expression field. | No dialog consumer; unchanged. | Terminal-sibling regression. |
| Dialog generator / `DIALOG_GENERATOR_LLM` | 1 render plus optional 1 repair render | unchanged | Replaces <=1,000-char style string with <=1,000-char delivery object; accepted candidate text is capped at 12,000 visible chars. | Two dialog candidates maximum; rejected dialog is omitted from the second render payload. | Generator, candidate-cap, and repair tests. |
| Semantic fidelity / `DIALOG_GENERATOR_LLM` | 1 check plus optional structural replacement; percepts <=24,000 chars | same call count with authoritative semantics <=11,000 chars | Human payload hard cap is 50,000 serialized characters; no authoritative field is truncated. Pre-call overflow records `not_called_context_limit` and raises `dialog_semantic_fidelity_context_limit`. | Existing two structural attempts inside the cap; no semantic retry or repair after context failure. | Maximum payload, typed-failure, captured-bad, and generated-bad tests. |
| Role direction and surface integrity / `DIALOG_GENERATOR_LLM` | 2 parallel checks per candidate | unchanged | Inputs and caps unchanged. | Parallel verifier fan-out unchanged. | Existing focused-verifier regressions. |
| Conversation-progress recorder / `CONSOLIDATION_LLM` | 1 background call with semantic plan, intent, style, and dialog | 1 background call with accepted semantic plan, intent, and dialog | Removes delivery text from the packet. | Background behavior and call count unchanged. | Progress handoff test and live review. |

Normal text-surface planning changes from three LLM calls to two. The maximum
dialog candidate and verifier counts do not increase. No new response-path or
background LLM call is authorized.

## Change Surface

Target ownership boundary: Cognition Core V2 text-surface planning and its
direct dialog/progress consumers.

### Delete

- Delete no files.
- Remove these V2 symbols from their current files:
  `STYLE_SYSTEM_PROMPT`, `run_style_stage`, `_validate_style_result`,
  `TextSurfaceServicesV2.style_config`, and the `v2_surface_style` binding.

### Modify

- `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`
  - Add exact `PersonalityJudgmentV2`,
    `CharacterExpressionContextV2`, and public `DeliveryProfileV2`.
  - Apply the exact cognition-constraint, input, output, and service contract
    replacements above.
- `src/kazusa_ai_chatbot/cognition_core_v2/__init__.py`
  - Export `DeliveryProfileV2` with the existing public text-surface types.
- `src/kazusa_ai_chatbot/cognition_core_v2/state_projection.py`
  - Project bounded `personality_judgment` with the existing character
    constraints.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface_stages.py`
  - Delete the style stage.
  - Extend the content and repair prompts/validators with exact delivery
    profile ownership.
- `src/kazusa_ai_chatbot/cognition_core_v2/surface.py`
  - Run only content and preference in parallel.
  - Project character expression only to content and visual.
  - Replace all producer-owned fields during repair from canonical input.
- `src/kazusa_ai_chatbot/cognition_core_v2/validation_cli.py`
  - Build the exact personality-judgment constraint in the V2-only benchmark
    payload.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_l3_surface.py`
  - Build the exact tempo/texture `character_expression_context` and isolated
    full-profile `visual_character_context`.
  - Remove the style config and bind only content/preference services.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2_cognition.py`
  - Build exact `personality_judgment` from the validated character profile
    before V2 cognition.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Render the delivery profile after semantic fields.
  - Give semantic fidelity authoritative surface semantics.
  - Return the accepted surface with the accepted dialog.
- `src/kazusa_ai_chatbot/nodes/persona_supervisor2.py`
  - Replace the initial surface with the accepted post-dialog surface before
    graph return and post-turn consumers.
- `src/kazusa_ai_chatbot/brain_service/post_turn.py`
  - Build progress input from the accepted semantic surface and omit delivery
    fields.
- `src/kazusa_ai_chatbot/cognition_core_v2/README.md`
  - Document two-call unified content/preference planning, full repair, and
    verifier authority.
- `src/kazusa_ai_chatbot/nodes/README.md`
  - Document the same connector/dialog contract and accepted-surface handoff.
- `src/kazusa_ai_chatbot/conversation_progress/README.md`
  - Clarify that only accepted dialog/surface semantics enter recording and
    rejected candidates remain trace-only.
- `development_plans/README.md`
  - Keep this plan's active-bugfix registry row and update its lifecycle status
    only when the plan status changes.
- Update exact-contract fixtures and assertions in:
  - `tests/test_cognition_core_v2_contracts.py`
  - `tests/test_cognition_core_v2_projection.py`
  - `tests/test_cognition_core_v2_integration.py`
  - `tests/test_cognition_core_v2_live_llm.py`
  - `tests/test_cognition_core_v2_live_character_judgment.py`
  - `tests/test_cognition_core_v2_frozen_replay_drift.py`
  - `tests/test_cognition_core_v2_surface_owner_live_llm.py`
  - `tests/test_cognition_prompt_contract_text.py`
  - `tests/test_cognition_chain_connector_mapping.py`
  - `tests/test_l2d_l3_surface_handoff.py`
  - `tests/test_cognition_interaction_style_context.py`
  - `tests/test_conversation_progress_history_policy.py`
  - `tests/test_past_dialog_cognition_prompt_boundaries.py`
  - `tests/test_dialog_agent.py`
  - `tests/test_dialog_mention_target_user.py`
  - `tests/test_dialog_visible_speech_and_semantic_fidelity.py`
  - `tests/test_dialog_visible_speech_and_semantic_fidelity_live_llm.py`
  - `tests/test_dialog_first_person_perspective_live_llm.py`
  - `tests/test_dialog_generator_live_llm_contract.py`
  - `tests/test_l3_dialog_content_plan_contract.py`
  - `tests/test_rag_dialog_event_logging.py`
  - `tests/test_self_cognition_integration.py`
  - `tests/test_self_cognition_tracking.py`
  - `tests/test_persona_supervisor2.py`
  - `tests/test_coding_agent_phase3_handoff_e2e.py`
  - `tests/test_consolidator_efficiency.py`
  - `tests/test_consolidator_origin_policy_db_writer.py`
  - `tests/test_consolidator_source_aware_payloads.py`
- Update exact V2 cognition-constraint fixtures in:
  - `tests/test_cognition_core_v2_abuse_to_sadness_e2e_live_llm.py`
  - `tests/test_cognition_core_v2_abuse_to_sadness_mechanical.py`
  - `tests/test_cognition_core_v2_alignment_gates.py`
  - `tests/test_cognition_core_v2_benchmark.py`
  - `tests/test_cognition_core_v2_crying_sadness_e2e_live_llm.py`
  - `tests/test_cognition_core_v2_emotion_lifecycle.py`
  - `tests/test_cognition_core_v2_high_attachment_abuse_e2e_live_llm.py`
  - `tests/test_cognition_core_v2_secondary_crying_e2e_live_llm.py`
  - `tests/test_cognition_core_v2_verbal_abuse_boundary_e2e_live_llm.py`
  - `tests/test_cognition_current_event_grounding.py`
  - `tests/test_conversation_progress_cognition.py`
  - `tests/test_multi_source_cognition_stage_00_regression_baseline.py`
  - `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`
  - `tests/test_multi_source_cognition_stage_03_prompt_selection.py`
  - `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`

### Create

- `tests/test_cognition_core_v2_transition_coherence.py`
  - Deterministic contract, projection, call-count, repair, verifier-payload,
    accepted-surface, and progress-handoff regressions.
- `tests/test_cognition_core_v2_transition_coherence_live_llm.py`
  - One function per captured case, semantic variant, control, verifier,
    repair, and progress-quality gate.
- `test_artifacts/llm_reviews/cognition_core_v2_transition_coherence_after.md`
  - Parent-authored before/after review linked to durable raw traces.

### Keep

- Keep every file outside the listed target boundary unchanged.
- Keep mainline and all compatibility paths absent from this V2 plan.
- Keep `TextSurfaceInputV2` and `TextSurfaceOutputV2` schema-version literals at
  `.v2`; this is an unreleased branch-local big-bang contract replacement.
- Keep role-direction, surface-integrity, visual-directive, action-result, and
  conversation-progress storage schemas unchanged.
- Keep the current reproduction artifacts as immutable baseline evidence.
- Keep
  `development_plans/active/bugfix/cognition_core_v2_baseline_regression_hardening_plan.md`
  unchanged.

## Overdesign Guardrail

- Actual problem: an independent style producer can add an opposite stance,
  repair preserves that producer, the verifier lacks surface authority, and
  accepted bad output can become future continuity.
- Minimal change: remove the separate V2 style call, atomically produce content
  and bounded delivery descriptors, strengthen the existing verifier payload,
  replace all producer-owned fields during the existing repair, and propagate
  only the accepted surface.
- Ownership boundaries: V2 cognition combines semantic personality constraints
  with current evidence and selects stance; unified V2 content planning
  expresses that selected semantics and delivery dimensions from tempo/
  texture only; preference owns real boundaries/addressee; dialog owns final
  words; focused verifiers own their existing hard-error classes;
  deterministic code owns exact shapes, caps, accepted-candidate propagation,
  and persistence eligibility.
- Rejected complexity: phrase bans, deterministic stance classifiers, output
  rewriting, a style/content compatibility LLM, another verifier, another
  repair, compatibility schemas, feature flags, model routing changes,
  progress-quality persistence fields, and mainline changes.
- Evidence threshold: new architecture is allowed only after matched live V2
  evidence shows this owner-level correction still produces a score-`0`
  transition or loses a required control and the current owner cannot correct
  it within the existing call/repair budget.

## Agent Autonomy Boundaries

- The parent may choose local implementation mechanics only when they preserve
  every exact contract and owner in this plan.
- The parent must not introduce a new architecture, compatibility layer,
  fallback path, helper agent, retry, feature, or persistent field.
- Changes outside the listed target boundary require a plan amendment and user
  approval before implementation.
- The parent must search for existing validator/projection behavior before
  adding a helper. A helper is permitted only for reused nontrivial exact
  `DeliveryProfileV2` validation/projection.
- The parent may delete the named style symbols without further design
  approval after this plan is approved.
- The parent must not perform unrelated cleanup, dependency upgrades,
  formatting churn, or broad prompt rewrites.
- If source and plan disagree, stop and record the exact discrepancy. Preserve
  this plan's intent until the user approves an amendment.
- If a required gate cannot run, leave its checklist item unsigned and report
  the blocker.

## Implementation Order

1. Add deterministic red contracts in
   `tests/test_cognition_core_v2_transition_coherence.py`.
   - Prove the old style service/call still exists.
   - Prove the old output accepts `style_guidance`.
   - Prove repair preserves rejected style.
   - Prove semantic fidelity omits authoritative surface semantics.
   - Prove repaired accepted surface is not propagated to graph/progress.
   - Record each expected failure in `Execution Evidence`.
2. Freeze the live baseline with the existing reproduction review and rerun
   the known-bad verifier and repair-path case one at a time.
   - Do not edit production code until raw artifacts match the stated failure
     or the plan is amended to the observed current state.
3. Replace contracts in
   `src/kazusa_ai_chatbot/cognition_core_v2/contracts.py`.
   - Add exact personality, expression, and delivery types.
   - Replace cognition-constraint, input, output, and service fields in one
     edit; export `DeliveryProfileV2`.
   - Update focused contract fixtures before moving to prompts.
4. Update semantic personality projection in
   `persona_supervisor2_cognition.py` and `state_projection.py`.
   - Project `logic`, `defense`, `quirks`, and `taboos` into V2 cognition.
   - Prove tempo/texture remain outside semantic constraints.
5. Replace producing stages in `surface_stages.py` and `surface.py`.
   - Delete the style producer.
   - Make content return the atomic content/delivery object.
   - Keep preference parallel and raw character expression out of preference.
   - Make repair regenerate every producer-owned field from canonical input.
6. Update `persona_supervisor2_l3_surface.py`.
   - Build tempo/texture expression context and isolated visual context.
   - Remove style config and validate the two-call service shape.
7. Update dialog generation and semantic fidelity in `dialog_agent.py`.
   - Make semantic fields primary and delivery profile secondary.
   - Add authoritative surface semantics to the existing semantic verifier.
   - Apply the general transition rule, exact sub-caps, and 50,000-character
     payload cap with typed pre-call failure.
   - Remove rejected dialog from the second render payload.
   - Preserve existing role/surface verifier inputs and call counts.
8. Update accepted-surface propagation in `dialog_agent.py` and
   `persona_supervisor2.py`.
   - Return and store the surface that produced the accepted dialog.
   - Exercise both first-pass and repair-pass paths.
9. Update `brain_service/post_turn.py`.
   - Project accepted semantic content/intent only.
   - Verify rejected candidates cannot reach progress recording.
10. Rewrite all listed old-contract fixtures and assertions.
   - Retain test intent while replacing old field names and service counts.
   - Add no dual-shape test helper.
11. Update subsystem READMEs and run static/compile/deterministic gates.
12. Run live owner, verifier, dialog, repair, and progress cases one at a time.
13. Author the before/after review, apply the parent code-review gate,
    remediate findings, rerun affected verification, and request user sign-off.

Order rationale: the exact response-local contract must exist before producer
and consumer rewrites; producer ownership must be correct before verifier
tuning; accepted-surface propagation must be correct before persistence can be
trusted.

## Execution Model

- This plan uses the user's explicit fallback execution direction.
- The one native subagent allowance is consumed by the pre-delivery
  independent quality review of this plan.
- The parent owns test contracts, production implementation, live execution,
  raw-output inspection, rubric scoring, integration, documentation, execution
  evidence, review remediation, lifecycle updates, and final sign-off.
- The parent establishes and records focused red tests before production edits.
- The parent performs production steps sequentially within the approved change
  surface.
- The parent pauses after verification and conducts the `Independent Code
  Review` from a fresh-review posture without another subagent.
- In-scope review findings are fixed by the parent and affected gates are
  rerun. Findings that require a new contract or wider change surface stop
  execution pending plan amendment and user approval.

## Progress Checklist

- [x] Stage 1 - baseline and focused red contracts frozen.
  - Covers Steps 1-2.
  - Files: new deterministic/live transition tests and baseline trace paths.
  - Verify: run each new red selector and the known-bad live verifier/repair
    selectors one at a time.
  - Evidence: expected failure, raw trace, current call count, and current
    `0/1/2` score recorded in `Execution Evidence`.
  - Next: Stage 2 contract replacement.
  - Sign-off: parent, 2026-07-25T14:23:53Z.
- [x] Stage 2 - V2 surface contract and unified producer complete.
  - Covers Steps 3-6.
  - Files: V2 contracts/exports, cognition personality projection, stages,
    surface facade, and L3 connector.
  - Verify: focused contract/projection/call-count tests pass; prompt render
    succeeds; old-symbol grep returns zero matches.
  - Evidence: changed symbols, two-call trace, tests, and prompt character
    counts recorded.
  - Next: Stage 3 dialog and accepted-surface integration.
  - Sign-off: parent, 2026-07-25T14:39:33Z. The Stage 2 target-file grep
    returned zero; the repository-wide cutover grep remains a Stage 4 gate.
- [x] Stage 3 - dialog verifier, repair, and accepted-surface flow complete.
  - Covers Steps 7-9.
  - Files: dialog agent, persona supervisor, and brain-service post-turn.
  - Verify: deterministic verifier payload, repair replacement,
    accepted-surface propagation, and progress exclusion tests pass.
  - Evidence: first-pass and repaired surface identities, verifier payload
    projection, exhaustion behavior, and progress input recorded.
  - Next: Stage 4 regressions and docs.
  - Sign-off: parent, 2026-07-25T14:58:05Z.
- [ ] Stage 4 - old-contract fixtures, regressions, and docs complete.
  - Covers Steps 10-11.
  - Files: all listed tests and three subsystem READMEs.
  - Verify: static greps, `py_compile`, focused deterministic batches, and
    affected non-live regressions pass.
  - Evidence: collected test node IDs, outputs, doc diff, and zero forbidden
    references recorded.
  - Next: Stage 5 live quality gates.
  - Sign-off: parent and UTC timestamp after evidence is recorded.
- [ ] Stage 5 - post-fix live quality evidence accepted.
  - Covers Step 12.
  - Files: live traces and parent-authored review artifact.
  - Verify: every live selector is run and inspected individually; captured
    cases and variants meet the rubric; controls retain meaning.
  - Evidence: raw trace per run, model/config fingerprint, surface output,
    dialog, verifier verdict, rubric score, and quality notes recorded.
  - Next: Stage 6 parent independent code review.
  - Sign-off: parent and UTC timestamp after evidence is recorded.
- [ ] Stage 6 - independent code review and remediation complete.
  - Covers Step 13.
  - Files: complete implementation diff, plan, evidence, and lifecycle docs.
  - Verify: run the full review rubric below, fix every in-scope critical/high
    finding, and rerun affected static, deterministic, and live gates.
  - Evidence: findings, file/line proof, fixes, reruns, residual risks, and
    final verdict recorded.
  - Next: request user sign-off before lifecycle closure.
  - Sign-off: parent and UTC timestamp after review evidence is recorded.

## Verification

### Static and prompt gates

- `git status --short`
  - Expected: only plan-approved files plus the pre-existing
    `test_runs/controlled/C07/1/r7a5e422b1304/` entry.
- `rg -n "style_guidance|style_config|v2_surface_style|run_style_stage|character_voice_context" src/kazusa_ai_chatbot tests`
  - Expected: exit code `1` and zero matches. Any match is a failed big-bang
    cutover.
- `rg -n "delivery_profile|character_expression_context|visual_character_context|personality_judgment" src/kazusa_ai_chatbot/cognition_core_v2 src/kazusa_ai_chatbot/nodes tests`
  - Expected: exit code `0`; every match belongs to the exact new contract or
    its tests. Aliases and optional fallback access are forbidden.
- `venv\Scripts\python.exe -m py_compile
  src\kazusa_ai_chatbot\cognition_core_v2\__init__.py
  src\kazusa_ai_chatbot\cognition_core_v2\contracts.py
  src\kazusa_ai_chatbot\cognition_core_v2\state_projection.py
  src\kazusa_ai_chatbot\cognition_core_v2\surface_stages.py
  src\kazusa_ai_chatbot\cognition_core_v2\surface.py
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_cognition.py
  src\kazusa_ai_chatbot\nodes\persona_supervisor2_l3_surface.py
  src\kazusa_ai_chatbot\nodes\dialog_agent.py
  src\kazusa_ai_chatbot\nodes\persona_supervisor2.py
  src\kazusa_ai_chatbot\brain_service\post_turn.py`
  - Expected: exit code `0`.
- Render every changed prompt through its production message builder with a
  canonical fixture.
  - Expected: no `.format(...)` failure; exact output fields are named; content
    and repair payloads remain <=24,000 characters; semantic verifier payload
    remains <=50,000 characters and each declared sub-cap is enforced.
- `git diff --check`
  - Expected: exit code `0`.

### Rollback evidence gate

- At Stage 1, record `git rev-parse HEAD` as `transition_base_sha`.
- Before final sign-off, run
  `git diff --name-only <transition_base_sha> --` and compare the result with
  `Change Surface`.
  - Expected: every changed path is listed in this plan; no database module,
    migration, seed, adapter, mainline bridge, or dependency file appears.
- Rollback, when required, applies one reviewed inverse patch covering the
  complete approved changed-file set or deploys `transition_base_sha`.
  Partial contract rollback is forbidden.
- In the rolled-back checkout, run
  `rg -n "delivery_profile|character_expression_context|visual_character_context|personality_judgment" src/kazusa_ai_chatbot tests`.
  - Expected: exit code `1` and zero new-contract matches.
- In the rolled-back checkout, run the old focused contract, surface-handoff,
  and dialog suites named under `Focused deterministic tests`.
  - Expected: compile and focused tests pass against the restored old contract;
    old style symbols exist only in their pre-change canonical owners.
- Record that rollback requires no database restore, backfill reversal, cache
  conversion, or persistent-row transformation because this plan changes only
  response-local contracts and prompt/runtime handoffs.

### Focused deterministic tests

- `venv\Scripts\python.exe -m pytest
  tests\test_cognition_core_v2_transition_coherence.py
  tests\test_cognition_core_v2_contracts.py
  tests\test_l2d_l3_surface_handoff.py -q`
- `venv\Scripts\python.exe -m pytest
  tests\test_dialog_visible_speech_and_semantic_fidelity.py
  tests\test_dialog_agent.py
  tests\test_l3_dialog_content_plan_contract.py -q`
- `venv\Scripts\python.exe -m pytest
  tests\test_conversation_progress_history_policy.py
  tests\test_conversation_progress_recorder.py
  tests\test_rag_dialog_event_logging.py -q`

Expected for each command: exit code `0`, intended selectors collected, and no
live tests batched into the run.

### Affected non-live regression

- `venv\Scripts\python.exe -m pytest -m "not live_llm and not live_db"
  tests\test_cognition_core_v2_integration.py
  tests\test_cognition_core_v2_frozen_replay_drift.py
  tests\test_cognition_interaction_style_context.py
  tests\test_past_dialog_cognition_prompt_boundaries.py
  tests\test_dialog_mention_target_user.py
  tests\test_self_cognition_integration.py
  tests\test_self_cognition_tracking.py
  tests\test_coding_agent_phase3_handoff_e2e.py
  tests\test_consolidator_efficiency.py
  tests\test_consolidator_origin_policy_db_writer.py
  tests\test_consolidator_source_aware_payloads.py -q`

Expected: exit code `0`; unrelated failures stop the gate for classification
before scope changes.

### Live LLM cases

Run each selector separately with `-q -s`, inspect its trace, assign the rubric
score, and update the review artifact before running the next selector:

- `test_live_captured_room_request_acceptance_is_coherent`
- `test_live_captured_accomplice_confirmation_is_coherent`
- `test_live_room_request_semantic_variant_is_coherent`
- `test_live_accomplice_semantic_variant_is_coherent`
- `test_live_embarrassed_acceptance_variant_is_coherent`
- `test_live_genuine_refusal_remains_refusal`
- `test_live_supported_change_of_mind_preserves_reason`
- `test_live_neutral_character_preserves_selected_stance`
- `test_live_known_bad_reversal_is_rejected_by_semantic_fidelity`
- `test_live_generated_repair_path_bad_reversal_is_rejected_by_semantic_fidelity`
- `test_live_repair_replaces_conflicting_delivery_and_stays_coherent`
- `test_live_progress_records_only_the_accepted_coherent_turn`

Each selector lives in
`tests/test_cognition_core_v2_transition_coherence_live_llm.py`.
Run the two captured positive cases three times each with identical semantic
fixtures and record every result separately. Run the remaining selectors once
each. A suspicious or failed result stops the sequence for interpretation.

The parent-authored review must include:

- branch, revision, command, timestamp, model route/name, prompt/code version,
  and fixture source;
- raw input and model-visible transformed input;
- unified content/delivery output, preference output, merged surface, dialog,
  verifier verdict, repair output when used, and progress output when used;
- `opening stance -> transition/reason -> final stance/action`;
- score `0`, `1`, or `2` with evidence;
- before/after comparison against the current reproduction;
- personality retention, regression notes, and human-attention items;
- raw trace paths.

### Broad non-live gate

- `venv\Scripts\python.exe -m pytest -m "not live_llm and not live_db" -q`
  - Expected: exit code `0`. Existing unrelated failures require explicit
    classification and user direction before closure.

## Independent Plan Review

Run exactly one read-only subagent review before delivering this draft. The
reviewer may not edit files or spawn another agent. It inspects:

- `AGENTS.md`, the development-plan contract/references, and this plan;
- the current V2 surface, dialog, repair, progress, source, tests, and
  reproduction evidence;
- architecture alignment, V2-only scope, exact contracts, call budget,
  implementation order, verification, acceptance math, and rollback;
- unresolved choices, broad verbs, hidden compatibility, extra calls,
  phrase-shaped rules, persistence overreach, and unlisted change surface.

The parent classifies the returned findings, applies all plan edits itself,
performs the final plan self-review, and records the reviewer identity,
findings, remediation, and disposition here before delivery.

Review record:

- Reviewer: `/root/plan_quality_review`, the only subagent used for this plan.
- Mode: read-only; the reviewer changed no files and spawned no agents.
- Blockers reported: undercounted verifier context and missing typed overflow
  path; contradictory rejected-candidate reuse; impossible score-`0`
  accounting plus a missing negative selector; unresolved semantic personality
  ownership; and absent review record.
- Non-blocking findings: missing public `DeliveryProfileV2` export; content
  procedure referenced preference-owned boundaries unavailable to the parallel
  content call; and rollback lacked an evidence gate.
- Parent remediation:
  - recalculated authoritative/verifier budgets, added exact sub-caps and the
    typed pre-call failure/trace disposition;
  - removed rejected surface and dialog from both repair-model payloads;
  - limited the zero-score generation metric and added the second negative
    verifier selector;
  - moved semantic profile fields into V2 cognition constraints and restricted
    text expression to tempo/linguistic texture while isolating visual context;
  - added the public export, corrected content inputs, added rollback
    verification, and recorded this review.
- Disposition: all eight findings are incorporated. The parent completed the
  post-remediation plan self-review. The user authorized execution on
  2026-07-25T14:18:02Z; later subagent use remains excluded.

## Independent Code Review

Under the user's explicit parent-only execution direction, the parent performs
this gate from a fresh-review posture after all implementation verification
passes and before completion, merge, lifecycle closure, or sign-off.

Review inputs:

- approved plan and its independent plan-review record;
- `git status --short`, complete diff, and changed-file list;
- focused red/green output, prompt renders, static greps, non-live tests, raw
  live traces, and the parent-authored quality review;
- registry/lifecycle changes and execution evidence.

Review rubric:

- exact V2-only big-bang contract with no old symbol, alias, optional fallback,
  dual fixture, or mainline change;
- one unified content/delivery owner and no hidden semantic style producer;
- delivery profile exactness and no deterministic semantic interpretation;
- semantic verifier uses authoritative semantics, excludes delivery/action/
  selection-owned fields, and keeps issue/call caps;
- repair regenerates every producer-owned field from canonical input and never
  preserves rejected delivery or semantics;
- accepted surface identity reaches graph state and progress while rejected
  candidates remain trace-only;
- conversation progress receives no delivery profile and gains no quality
  classifier or storage field;
- prompt language, parser use, CJK safety, Python style, exception ownership,
  test taxonomy, one-at-a-time live execution, and artifact authorship;
- context/call budgets, payload caps, rollback, docs, and verification claims
  match the actual diff and evidence;
- no unplanned file, dependency, cleanup, feature, retry, or model change.

Every finding records severity, file/line evidence, remediation, rerun, and
verdict in `Execution Evidence`. Unresolved critical or high findings block
completion. A finding that changes the contract or change surface requires a
plan amendment and user approval.

## Acceptance Criteria

This plan is complete only when:

1. V2 has no independent style LLM call, style service config, or
   `style_guidance` contract.
2. Unified content planning returns exact content, requirements, and five
   delivery dimensions in one LLM response.
3. Normal text planning uses exactly two parallel calls and adds no response-
   path or background call.
4. The two captured examples and all three semantic variants score `2` in
   their specified matched live runs.
5. No newly generated post-fix dialog scores `0`. The two seeded score-`0`
   negative verifier fixtures are excluded from this generation metric and
   must be rejected as required by Criterion 6.
6. The known-bad captured and current-branch generated reversal candidates are
   rejected by semantic fidelity for an unsupported stance transition.
7. Genuine refusal remains refusal. The supported-change control retains its
   explicit reason. The neutral control preserves its selected stance.
8. Embarrassment, defensiveness, hesitation, teasing, indirectness, and
   character-specific texture remain visible where the semantic fixture
   supports them.
9. Repair replaces content, delivery, boundaries, and addressee together and
   the repaired dialog passes the same three focused verifiers.
10. First-pass success propagates the original accepted surface; repair success
    propagates the replacement accepted surface.
11. Dialog-compliance exhaustion records trace evidence and reaches no
    conversation-progress write.
12. Conversation progress receives only accepted dialog and accepted semantic
    content/intent, with no delivery profile or rejected candidate.
13. Bracket behavior is unchanged and repetition is reported separately.
14. No production/test rule bans a connector phrase or classifies user/dialog
    text deterministically.
15. All static, deterministic, affected-regression, broad non-live, and
    one-at-a-time live gates pass and their evidence is recorded.
16. The parent-authored review shows real before/after inputs and outputs, not
    only pass/fail status.
17. The parent independent code review has no unresolved critical/high
    finding, and the user accepts the final quality evidence.

## Risks

| Risk | Mitigation | Verification |
| --- | --- | --- |
| Unified content owner still places stance in delivery fields | Exact five-dimension prompt contract, dialog semantic priority, and verifier exclusion of delivery authority | Focused owner traces, known-bad verifier case, positive/control matrix |
| Character voice becomes flat after style-call removal | Preserve semantic personality judgment in cognition, pass only bounded tempo and linguistic texture to unified content, and measure hesitation, indirectness, teasing, and texture separately from transition score | Five positive live cases and personality-retention notes |
| Verifier over-rejects genuine refusal or change of mind | Compare candidate with authoritative semantic plan and run explicit refusal/supported-change controls | Two live control selectors and deterministic payload test |
| Repair anchors on rejected output | Exclude rejected surface semantics/delivery from the surface repair payload and rebuild from canonical input plus verified issues | Repair payload assertion and live repair trace |
| Repaired dialog pairs with stale original surface | Return accepted surface from dialog and overwrite graph state before post-turn work | First-pass/repair propagation tests and progress input capture |
| Broad exact-contract fixture churn hides regressions | Big-bang zero-old-symbol grep, focused tests first, affected suite, then broad non-live gate | Static grep, collected node IDs, complete diff review |
| New verifier context exceeds local-model limits | Bound authoritative projection to <=11,000 chars and the complete serialized human payload to <=50,000 chars; fail before the call with the typed context-limit path | Maximum-shape payload test, typed-failure assertion, and live prompt counts |
| Existing active hardening work overlaps | Keep this dedicated plan narrow and leave the existing plan unchanged | Changed-file review and registry inspection |

## Execution Evidence

Record evidence as work completes; unchecked stages remain unproven.

| Evidence area | Required record |
| --- | --- |
| Baseline | Authorized 2026-07-25T14:18:02Z on `cognition_core_v2`; `transition_base_sha=ba433e9be4805bb88b992419db1c99a6e376dc4a`; only plan/registry plus pre-existing `test_runs/controlled/C07/1/r7a5e422b1304/` were dirty. |
| Contract red/green | Pre-fix focused file: 7 failed as expected. Stage 2 producer contracts and direct handoff suites passed (`46 passed`); expanded Stage 3 focused file passed (`14 passed`). Stage 3 also passed the dialog semantic/surface suite (`31 passed`) and the planned focused dialog batch (`48 passed`). |
| Prompt/call budget | Stage 2 deterministic render: exactly 2 calls; content prompt/payload `1908/1837` chars, preference prompt/payload `687/1286` chars; five exact delivery fields returned. Stage 3 verifier tests prove the exact authoritative semantic projection, `11,000`-character authority cap, `12,000`-character candidate cap, `50,000`-character serialized payload cap, and typed pre-call failure with zero semantic-verifier, repair, delivery, or progress calls. |
| Static/compile | Stage 2 changed-source/focused-test old-symbol grep exited `1` with zero matches. The planned repository-wide lowercase-symbol grep also exited `1` with zero matches, but the broader cutover audit found stale deleted `STYLE_SYSTEM_PROMPT` imports in `tests/test_cognition_core_v2_live_character_judgment.py` and `tests/test_cognition_prompt_contract_text.py`; those omitted paths stop collection. All changed Python files compile, all four changed prompt builders render through production paths, and `git diff --check` exits `0`. Stage 4 remains unsigned. |
| Deterministic/regression | Focused final batches passed: contracts/projection/L2D `46 passed`; dialog visible/dialog-agent/L3 `48 passed`; progress/history/RAG `14 passed`; production prompt render `4 passed`; new transition suite `14 passed`; exact fixture follow-up `9 passed`. The affected suite passed `131` nodes when five demonstrably pre-existing failures were deselected. Broad non-live collection, after temporarily ignoring the two stale-import files above, reached `3422 passed, 26 failed, 2 skipped, 825 deselected`. Seventeen failures expose omitted V2 big-bang consumers: seven in `tests/test_cognition_chain_connector_mapping.py`, one in `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`, one in `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`, seven in `tests/test_persona_supervisor2.py`, and one through the omitted production consumer `src/kazusa_ai_chatbot/cognition_core_v2/validation_cli.py`; the two stale-import files are additional collection blockers. Nine remaining failures are reproducible against `transition_base_sha` source and are unrelated baseline defects: two coding Phase 3 handoff, three self-cognition integration, one coding Phase 5 interface, one persona relevance, and two baseline-harness failures. |
| Stage 4 scope amendment | At 2026-07-25T23:29:23Z the user approved adding the seven exact-contract consumers discovered by regression verification: `src/kazusa_ai_chatbot/cognition_core_v2/validation_cli.py`, `tests/test_cognition_core_v2_live_character_judgment.py`, `tests/test_cognition_prompt_contract_text.py`, `tests/test_cognition_chain_connector_mapping.py`, `tests/test_multi_source_cognition_stage_02_chat_episode_migration.py`, `tests/test_multi_source_cognition_stage_09_multimodal_input_sources.py`, and `tests/test_persona_supervisor2.py`. The user also approved classifying the nine base-SHA-reproducible defects recorded above as outside this plan. Stage 4 resumed under this amended boundary. |
| Live quality | Baseline verifier rerun accepted the seeded score-`0` reversal; baseline repair rerun returned correct content while preserving contradictory style. Review and raw trace paths are recorded in `test_artifacts/llm_reviews/cognition_core_v2_transition_coherence_after.md`. |
| Progress | Deterministic tests prove the first-pass surface is returned unchanged, full repair returns and propagates its replacement surface, `persona_supervisor2` stores the dialog-accepted surface in both surface state views, and the recorder receives accepted dialog plus semantic content/intent only; delivery and rejected candidates are absent. |
| Parent code review | Findings, severity, file/line evidence, remediation, reruns, residual risk, and verdict |
| Lifecycle | User authorized parent-only execution at 2026-07-25T14:18:02Z; Stage 1 signed 2026-07-25T14:23:53Z; Stage 2 signed 2026-07-25T14:39:33Z; Stage 3 signed 2026-07-25T14:58:05Z; the user approved the Stage 4 scope amendment and unrelated-baseline disposition at 2026-07-25T23:29:23Z; Stage 4 and later stages remain pending. |
