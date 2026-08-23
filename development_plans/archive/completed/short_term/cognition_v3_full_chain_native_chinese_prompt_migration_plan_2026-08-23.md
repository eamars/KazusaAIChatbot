# Cognition V3 Full-Chain Native-Chinese Prompt Migration Plan

## Summary

- Goal: migrate every ordinary instruction owned by the canonical Cognition
  V3 A1 -> A2 -> G -> P model chain to native Simplified Chinese, document the
  exact prompt map, and prove the complete chain still works with deterministic
  and real-LLM evidence.
- Status: completed
- Authorization: the user explicitly commanded this implementation and
  approved closing the plan without clarification pauses.
- Execution constraint: the parent Codex agent is the sole executor, tester,
  reviewer, and closer for this migration.
- Scope boundary: `COGNITION_V3_CHAIN_LLM` system instructions, A1/A2/G/P
  packet guidance, prompt-owned fallback semantic text, exact owner tests,
  source-impact mapping, architecture documentation, and one isolated live
  full-chain check.

## Mandatory Skills

- `development-plan`: execution gates, evidence, review, archive, and registry
  closure.
- `local-llm-architecture`: one-language prompt design, minimal stage
  contracts, static system instructions, and dynamic human packets.
- `chinese-translation`: native Simplified Chinese localization with
  contractual identifiers preserved.
- `py-style`: production and test Python implementation policy.
- `cjk-safety`: UTF-8-safe CJK source edits and immediate syntax checks.
- `test-style-and-execution`: exact deterministic ownership tests and isolated
  live-LLM execution.
- `debug-llm`: retained raw evidence and an authored quality review.

## Rules

- Preserve exactly four provider calls in the order A1, A2, G, P.
- Preserve the single `COGNITION_V3_CHAIN_LLM` route and caller-owned model
  configuration.
- Keep dynamic per-turn state in the serialized human packet.
- Use native Simplified Chinese for ordinary prompt instructions and for
  free-form semantic model values.
- Preserve JSON field names, enum values, stage names, action/capability names,
  code, URLs, and quoted source text exactly.
- Keep A1 world-facing, A2 character-conditional, G character-goal-owned, and
  P response-planning-owned.
- Keep structural validators, persistence, permissions, capabilities, and
  downstream surface/dialog ownership unchanged.
- Preserve all concurrent work in the dirty shared worktree.
- Use `venv\Scripts\python` for Python and pytest commands and `apply_patch`
  for manual edits.
- Avoid reading `.env`; the existing live runtime may load its configuration
  through normal project code.

## Must Do

1. Publish a stable map from each A1/A2/G/P prompt segment to its source owner,
   runtime message role, route, language rule, and preserved identifiers.
2. Translate `_STAGE_SYSTEM_PROMPTS` and `_EXACT_JSON_SYSTEM_SUFFIX` into
   native Simplified Chinese without weakening evidence authority.
3. Translate all appraisal, goal, ordinary-plan, and self-plan guidance in
   `cognition_core_v3/prompt.py`.
4. Translate prompt-owned fallback semantic values that can enter the A1/A2/G/P
   human packet.
5. Add deterministic tests that render the real stage messages/packets, prove
   Chinese instruction ownership, and reject residual multi-word English prose
   outside preserved code spans.
6. Update existing direct prompt assertions and register the new exact owner
   tests in the source-test impact manifest.
7. Run immediate UTF-8 AST and compilation checks after CJK source edits.
8. Run exact mapped nodes, the complete cognition-core unit suite, prompt
   render checks, source-impact validation, and scoped diff checks.
9. Run the real full-chain live test one case at a time, inspect the actual
   A1/A2/G/P products, and author a human-readable quality review.
10. Complete parent review, archive this plan, update the lifecycle registry,
    and close the active goal only after every acceptance criterion passes.

## Deferred

- Relevance routing, vision description, message decontextualization, L3
  surface planning, final dialog wording, consolidation, reflection, and
  scheduler prompt rewrites; these are separate route owners.
- Changes to appraisal families, output schemas, enum values, model routing,
  retry policy, JSON repair, validators, persistence, or delivery.
- Broad translation of code diagnostics, docstrings, fixtures, user-provided
  evidence, or downstream generated state.
- Concurrent standalone-resolver and LLM-interface work.

## Target State

```text
COGNITION_V3_CHAIN_LLM
  SystemMessage: native Simplified Chinese stage instruction
  HumanMessage: exact typed JSON packet
    guidance: native Simplified Chinese
    dynamic evidence/state: source-owned content
    output_contract: exact contractual identifiers
  model output:
    exact JSON structure
    free-form semantic values: Simplified Chinese
    contractual identifiers/enums: unchanged
```

## Execution Role

### Parent Implementation, Verification, And Sign-Off Owner

- Responsibility: map, implement, test, live-verify, review, document, and
  close the migration.
- Owned production files:
  - `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
  - `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py`
- Owned tests and governance:
  - `tests/unit/cognition_core_v3/test_handleless_contract.py`
  - `tests/unit/cognition_core_v3/test_prompt_context.py`
  - the two existing source rows in
    `tests/ownership/source_test_impact_manifest.json`
- Owned evidence and documentation:
  - `docs/architecture/cognition_v3_full_chain_prompt_language_map.md`
  - one new raw live artifact under
    `test_artifacts/diagnostics/cognition_v3_capacity_live_llm/`
  - one new review under `test_artifacts/reviews/`
  - this plan and its `development_plans/README.md` registry row
- Independence requirement: none; the user's parent-only constraint makes the
  same parent responsible for a fresh final audit after verification.

## Baseline

The relevant files already contain approved prior-plan work that must remain
intact:

| Path | SHA-256 before this migration |
| --- | --- |
| `src/kazusa_ai_chatbot/cognition_core_v3/facade.py` | `EF77D4D61C87A4DF953A37FF95628BCCD2D50969089E6C97183F7E17B4C80E2D` |
| `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` | `52C270CAFA8E9385F96D85209550C1041EA8B969AB4537CD779A05BABB39A9E5` |
| `tests/unit/cognition_core_v3/test_handleless_contract.py` | `CC7B002824BEED77A6A80622A25F512EA4ED7E5455DC967B5E5CBB576132E14F` |
| `tests/unit/cognition_core_v3/test_prompt_context.py` | `433E6BD01CCBC82CE49BC9666FCF339CFC325DAC0C8939E275BAE09436717F41` |
| `tests/unit/cognition_core_v3/test_state_transaction.py` | `159F9ADA9450778D5F6BD1AC08BF818FD9D5C7C767291E68A99AD4DF4237F2CB` |
| `tests/test_cognition_v3_capacity_live_llm.py` | `8B8EF4BBD8A41CB4159B7A20B9C7AE7B9134ACAFFB173778FCBAD5494D75550C` |
| `tests/ownership/source_test_impact_manifest.json` | `9A762E802779C44ECB5E860FAAA46F8C696912182C715140FFF28269F2F3EEA1` |

## Test Impact And Traceability

| Production owner | Exact required deterministic nodes | Supplemental live check |
| --- | --- | --- |
| `cognition_core_v3/facade.py` | `tests/unit/cognition_core_v3/test_handleless_contract.py::test_cognition_chain_system_prompts_use_native_chinese`; `tests/unit/cognition_core_v3/test_handleless_contract.py::test_canonical_cognition_calls_a1_a2_g_p_once_with_subjective_outputs`; existing mapped state-transaction nodes | `tests/test_cognition_v3_capacity_live_llm.py::test_live_captured_full_state_greeting_completes_first_pass` |
| `cognition_core_v3/prompt.py` | `tests/unit/cognition_core_v3/test_prompt_context.py::test_cognition_chain_guidance_uses_native_chinese`; existing mapped context/authority nodes | the same isolated live full-chain node |

Governance also requires:

- `tests/test_test_impact_manifest.py::test_manifest_covers_strict_cognition_source_boundary`
- `venv\Scripts\python -m scripts.validate_test_impact --base-ref HEAD --run`

## Verification

Deterministic verification must include:

- UTF-8 AST parsing of both changed production Python files immediately after
  the CJK edit;
- `py_compile` for all changed Python source and tests;
- collection and execution of the two new exact owner nodes;
- every manifest-mapped required node for both changed production sources;
- the full `tests/unit/cognition_core_v3` suite;
- the strict manifest test and source-impact validator;
- a runtime render inspection of all four system messages and both ordinary
  and self-cognition P packets; and
- `git diff --check` for this plan's scoped files.

Live verification must run the single capacity-shaped greeting case in
isolation, retain the generated artifact, and inspect:

- A1, A2, G, and P each completing exactly once;
- exact structural contracts and bounded state transaction;
- free-form cognition semantics using Chinese;
- absence of exceptions in the test result; and
- actual output quality and evidence grounding.

The authored review must record the input, route/model, stage products,
language and grounding judgment, deterministic results, live result, and any
remaining risk.

## Acceptance Criteria

- Every static system instruction and stage guidance owned by A1/A2/G/P is
  native Simplified Chinese.
- No ordinary multi-word English instruction prose remains in those owned
  prompt segments outside preserved code spans.
- Contractual JSON keys, enums, action/capability names, code, URLs, and quoted
  source text remain unchanged.
- A1/A2/G/P authority and output contracts remain structurally identical.
- Both normal P and self-cognition P follow their respective
  `output_contract` without a conflicting system-level field list.
- All deterministic and impact checks pass.
- The isolated real-LLM full chain completes with valid Chinese semantic
  products and no exception.
- The prompt map, raw evidence, readable review, final audit, archived plan,
  registry, and active goal all agree on closure.

## Progress Checklist

- [x] Read lifecycle, architecture, source, test, and skill contracts.
- [x] Captured the dirty-worktree baseline and exact owned-file hashes.
- [x] Fixed the A1/A2/G/P prompt ownership and language boundary.
- [x] Published the stable prompt map.
- [x] Implemented the native-Chinese prompt migration.
- [x] Added and registered exact deterministic owner tests.
- [x] Completed deterministic, impact, and render verification.
- [x] Completed the isolated real-LLM run and readable review.
- [x] Completed parent sign-off, archive, registry update, and goal closure.

## Execution Evidence

### Implemented Prompt Boundary

- `facade._STAGE_SYSTEM_PROMPTS` now gives A1, A2, G, and P native Chinese
  task and evidence-authority instructions.
- `facade._EXACT_JSON_SYSTEM_SUFFIX` requires exact JSON, Chinese free-form
  semantics, preserved contractual identifiers, and no private references.
- All six guidance constants in `prompt.py` use native Chinese.
- The default operation and inspection-only default goal semantics that enter
  stage packets use Chinese.
- P's shared system instruction now defers exact fields to `output_contract`,
  so ordinary and self-cognition P packets have no conflicting field list.
- The stable owner map is
  `docs/architecture/cognition_v3_full_chain_prompt_language_map.md`.

### Deterministic Verification

Immediate CJK safety checks passed after each Python edit:

```powershell
venv\Scripts\python.exe -c "import ast, pathlib, sys; ..."
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/cognition_core_v3/facade.py src/kazusa_ai_chatbot/cognition_core_v3/prompt.py tests/unit/cognition_core_v3/test_handleless_contract.py tests/unit/cognition_core_v3/test_prompt_context.py
```

The focused prompt and full-chain batch passed 4/4. The exact owner and strict
manifest batch passed 11/11. The final subsystem command passed all 35 tests:

```powershell
venv\Scripts\python.exe -m pytest -q -ra tests/unit/cognition_core_v3
```

The final source-impact command collected all 102 exact nodes and completed
with 101 passed plus one expected Windows symlink-privilege skip:

```powershell
venv\Scripts\python.exe -m scripts.validate_test_impact --base-ref HEAD --run
```

`ruff check --select F,I` passed for all four changed Python files. A broader
diagnostic retained the files' existing E501 line-length debt and three
unrelated legacy rule findings outside the changed prompt/test lines. The two
new import-order findings from the first diagnostic were fixed and rechecked.

`git diff --check` passed for the scoped change with only Git's existing CRLF
normalization warnings.

### Real-LLM End-To-End Evidence

The live test was run explicitly as one isolated case because the default
pytest configuration deselects `live_llm`:

```powershell
venv\Scripts\python.exe -m pytest -m live_llm -q -ra -s tests/test_cognition_v3_capacity_live_llm.py::test_live_captured_full_state_greeting_completes_first_pass
```

Result: 1 passed in 30.07 seconds. The model
`gemma-4-31b-isometry-fabled-persona-i1` executed exactly A1, A2, G, and P on
`COGNITION_V3_CHAIN_LLM`. All four protected records have `status=parsed`, all
system messages and guidance are Chinese, the free-form semantic products are
Chinese, the artifact has no error field, and no capacity deferral occurred.

Raw artifact:

`test_artifacts/diagnostics/cognition_v3_capacity_live_llm/capacity_transaction_1787465099676007300.json`

Authored review:

`test_artifacts/reviews/cognition_v3_full_chain_native_chinese_prompt_migration_2026-08-23.md`

The provider used Markdown JSON fences. The canonical transport cleanup
accepted each object without semantic repair or regeneration, which is the
repository's declared JSON parsing contract.

### Final Hashes

| Path | SHA-256 |
| --- | --- |
| `src/kazusa_ai_chatbot/cognition_core_v3/facade.py` | `D952557F7F38B82BC1FCD50A3D98E5E55EEFD98B203E419669F4300F4E72B3D0` |
| `src/kazusa_ai_chatbot/cognition_core_v3/prompt.py` | `B22B542678491C7667101FC8682665BD7EED2E002F86E83711A5C02315C7A584` |
| `tests/unit/cognition_core_v3/test_handleless_contract.py` | `D6A7B5840066CA0147EB1FC7295A5060BF2B4F8861D35E580A96A213E2632EE1` |
| `tests/unit/cognition_core_v3/test_prompt_context.py` | `E3FDF738ECD020FF810ABE051D2EBFF3655069639F167455B8850B36117B910A` |
| `tests/ownership/source_test_impact_manifest.json` | `BF59F2A58E96F84085378CD43AFF823F58AA345BD7100537E755151C6097195D` |
| `docs/architecture/cognition_v3_full_chain_prompt_language_map.md` | `4771955AEA9E93AFAF8FA6A117393F60CB2B5AC1AB0ED0327B3546AE20E4A063` |
| Raw live artifact | `970065EE602EEE3B32CB56C6939514A3B0D8C49012D873C8331796304B86A612` |
| Authored review | `8E40A0F6ECFF3767024DDA19794A88CA9A7A0AE1E9A350785223C882AEF9BAA8` |

### Parent Sign-Off

The parent reviewed the exact prompt owners, rendered system/human messages,
ordinary and self-cognition P contracts, test mappings, source-impact result,
raw live artifact, authored quality review, scoped diff, and final hashes.

Finding disposition:

- Every in-scope A1/A2/G/P ordinary instruction is native Chinese.
- Contract identifiers, enums, capability/action names, code, URLs, and quoted
  source data remain exact.
- Separate relevance, vision/decontextualization, surface/dialog,
  consolidation, reflection, scheduler, and resolver routes are explicitly
  mapped as separate owners and remain unchanged.
- Deterministic, impact, and real-LLM workflow gates pass.
- The live cognition workflow completed without an exception.

No unresolved scope, ownership, language, structural-contract, deterministic,
impact, live-workflow, or evidence finding remains. The plan is complete and
archived under `development_plans/archive/completed/short_term/`.
