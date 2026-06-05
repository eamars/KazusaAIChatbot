# napcat qq adapter modularization face catalog plan

## Summary

- Goal: split the monolithic NapCat QQ adapter into a package with stable public entrypoints and add a complete static QQ face semantic catalog.
- Plan class: large
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`, `py-style`, `cjk-safety`, `test-style-and-execution`
- Overall cutover strategy: bigbang package conversion with compatibility at the import and CLI boundary only.
- Highest-risk areas: breaking `python -m adapters.napcat_qq_adapter`, breaking existing imports from `adapters.napcat_qq_adapter`, over-splitting the first adapter package, and shipping incomplete or low-quality QQ face semantic labels.
- Acceptance criteria: `python -m adapters.napcat_qq_adapter` still starts through the same command; existing imports of `NapCatWSAdapter`, `QQEnvelopeNormalizer`, and `project_qq_semantic_text` still work; NapCat and adapter ICDs exist; the QQ face table contains every in-scope numeric QQ face id from the accepted QFace snapshot with meaningful LLM-facing labels.

## Context

The current NapCat QQ adapter lives in one large file:

```text
src/adapters/napcat_qq_adapter.py
```

That module currently owns these unrelated responsibilities at once:

- CQ parsing and QQ semantic text projection.
- QQ mention display-name hydration and caching.
- `QQEnvelopeNormalizer`.
- Runtime FastAPI app and request/response models.
- `NapCatWSAdapter` websocket lifecycle, brain registration, heartbeat, event intake, reply hydration, delivery, and close behavior.
- CLI/environment parsing for `python -m adapters.napcat_qq_adapter`.

The previous completed containment plan fixed one confirmed QQ face id, `344`,
and protected the brain graph from truly empty input. That containment is not
enough for the broader RCA. QQ face ids are a platform-native expression
vocabulary. The adapter must translate them into stable semantic image text
before the brain sees them.

External protocol context:

- NapCat documents `face` as a supported QQ expression message segment and shows structured input as `{"type": "face", "data": {"id": "123"}}`.
- OneBot 11 documents the equivalent CQ syntax as `[CQ:face,id=123]` and treats `id` as the QQ expression id.
- Koishi QFace publishes a broad QQ face id/name/resource list including modern ids such as `344` for `大怨种`. The accepted catalog source for this plan is QFace repository commit `e476a706a7e508849c6031c3654051a02639964f`, file `public/assets/qq_emoji/_index.json`.
- At that QFace commit, `_index.json` contains 482 rows: 317 numeric `emojiId` rows and 165 Unicode emoji-style rows. This plan's QQ `face` catalog scope is the 317 numeric `emojiId` rows only, because the current NapCat/OneBot `face` segment and CQ projection contract handles `[CQ:face,id=<numeric id>]`.

The architectural boundary remains:

```text
adapter/debug client -> brain service -> queue/intake -> RAG -> cognition
```

QQ protocol handling and QQ face semantics stay in the adapter package. The
brain service must continue to consume only typed envelope fields and prompt
surface text, not CQ syntax or QQ face ids.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing, archiving, or signing off this plan.
- `local-llm-architecture`: load before changing adapter-to-brain contracts or prompt-facing projection.
- `py-style`: load before editing Python files.
- `cjk-safety`: load before editing Python files containing CJK strings.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute production changes while this plan status is `draft`; implementation requires explicit user approval.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, docs, and plan edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Keep adapter code thin: platform normalization, runtime registration, adapter delivery, and transport handling only.
- Keep QQ face interpretation inside `adapters.napcat_qq_adapter`; do not add QQ-specific logic to brain service, resolver, cognition, RAG, dialog, persistence, prompts, or message-envelope core.
- Preserve the command `python -m adapters.napcat_qq_adapter`.
- Preserve existing import surfaces used by tests and callers:

```python
import adapters.napcat_qq_adapter as napcat_module
from adapters.napcat_qq_adapter import NapCatWSAdapter, QQEnvelopeNormalizer
from adapters.napcat_qq_adapter import project_qq_semantic_text
```

- Convert `src/adapters/napcat_qq_adapter.py` into a package directory; do not leave a sibling `.py` module with the same import name.
- Do not add a fake image attachment for QQ system faces.
- Do not add runtime network fetches, database lookups, LLM calls, or retry loops for QQ face interpretation.
- The QQ face lookup table must store final LLM-facing descriptions, not compute labels through a universal `{source_name}表情` formula.
- New Python CJK string literals must use single-quoted delimiters unless escaping requires otherwise.
- After editing Python files with CJK literals, run `py_compile` against all touched Python files.
- After any automatic context compaction, the parent or active execution agent must reread this entire plan before continuing implementation, verification, handoff, lifecycle updates, or final reporting.
- After signing off any major progress checklist stage, the parent or active execution agent must reread this entire plan before starting the next stage.
- Before final completion, lifecycle status changes, merge, or sign-off, the parent agent must run the plan's `Independent Code Review` gate and record the result in `Execution Evidence`.
- The plan's `Execution Model` must use parent-led native subagent execution unless the user explicitly approves fallback execution.

## Must Do

- Stage 1: convert `src/adapters/napcat_qq_adapter.py` into `src/adapters/napcat_qq_adapter/`.
- Stage 1: preserve all existing import surfaces and CLI invocation.
- Stage 1: split the adapter into small modules by responsibility, not by arbitrary line ranges.
- Stage 1: add `src/adapters/napcat_qq_adapter/README.md` as the NapCat QQ adapter package ICD.
- Stage 1: add `src/adapters/README.md` as the generic adapter class/package ICD.
- Stage 1: keep behavior equivalent before introducing the full face catalog.
- Stage 2: add a complete static QQ face catalog module with final semantic labels.
- Stage 2: add a checked-in source snapshot fixture copied from QFace commit `e476a706a7e508849c6031c3654051a02639964f`, source file `public/assets/qq_emoji/_index.json`.
- Stage 2: define the in-scope accepted id set as exactly the 317 rows whose `emojiId` is an ASCII decimal string.
- Stage 2: require every in-scope numeric id in the accepted source snapshot to have a final semantic label.
- Stage 2: preserve the existing prompt surface:

```text
<image>{semantic_description}</image>
```

- Stage 2: omit unknown, missing, empty, non-numeric, or unusable face ids from `body_text`; do not inject `<image>表情</image>` or any other placeholder.
- Stage 2: test CQ string input, structured segment-list input, reply excerpts, unknown ids, malformed closed face segments, adjacent faces, escaped labels, and unknown face-only no-content handling.

## Deferred

- Do not redesign Discord or debug adapters in this plan.
- Do not add a generic adapter framework or base class unless it already exists locally and the split needs only documentation of that contract.
- Do not move shared message-envelope code into the NapCat package.
- Do not change brain service behavior, resolver validation, cognition graph topology, RAG, dialog, consolidation, or database schemas.
- Do not add runtime face asset downloads.
- Do not add a GUI, management endpoint, database table, or admin workflow for face catalog maintenance.
- Do not attempt live visual recognition of QQ face images in the response path.
- Do not include Unicode emoji-style `emojiId` rows from QFace in this plan's `CQ:face` catalog; they are outside the current NapCat `face` segment contract and require a separate adapter message-segment decision if they ever appear in inbound events.
- Do not backfill historical conversation rows.

## Cutover Policy

Overall strategy: bigbang with import/CLI compatibility.

| Area | Policy | Instruction |
|---|---|---|
| NapCat module shape | bigbang | Replace `napcat_qq_adapter.py` with package files in one change. Do not keep duplicate implementation paths. |
| Public import surface | compatible | Re-export existing public symbols from package `__init__.py`. |
| CLI surface | compatible | Add `__main__.py` so `python -m adapters.napcat_qq_adapter` remains unchanged. |
| Internal module boundaries | bigbang | Internal code imports package submodules directly after the split. |
| QQ face catalog | bigbang | Replace the one-id table with the complete static catalog in one change. |
| Unknown face ids | bigbang | Omit the segment from semantic projection. Do not inject placeholders. |

## Cutover Policy Enforcement

- The responsible execution agent must follow the selected policy for each area.
- For bigbang areas, rewrite old behavior directly instead of preserving duplicate old paths.
- For compatible areas, preserve only the import and CLI surfaces explicitly listed in this plan.
- Any change to this cutover policy requires user approval before implementation.

## Target State

Package layout:

```text
src/adapters/
  README.md
  napcat_qq_adapter/
    README.md
    __init__.py
    __main__.py
    attachments.py
    cli.py
    cq_projection.py
    envelope_normalizer.py
    face_catalog.py
    inbound_segments.py
    mention_hydration.py
    outbound.py
    reply_hydration.py
    runtime_api.py
    ws_adapter.py
```

Stable import surface:

```python
from adapters.napcat_qq_adapter import NapCatWSAdapter
from adapters.napcat_qq_adapter import QQEnvelopeNormalizer
from adapters.napcat_qq_adapter import project_qq_semantic_text
```

Stable CLI:

```powershell
venv\Scripts\python.exe -m adapters.napcat_qq_adapter --channels 987654321
```

Known QQ face:

```text
[CQ:face,id=344] -> <image>大怨种表情</image>
```

Unknown QQ face:

```text
[CQ:face,id=999999] -> ""
```

Unknown QQ face inside text:

```text
我[CQ:face,id=999999]服了 -> 我服了
```

Unknown QQ face-only turns rely on the existing brain service no-content guard:

```text
raw_wire_text = "[CQ:face,id=999999]"
body_text = ""
graph invoked = false
```

Catalog requirement:

```text
set(QQ_FACE_IMAGE_DESCRIPTIONS) == set(numeric_emoji_ids_from_qface_snapshot)
len(QQ_FACE_IMAGE_DESCRIPTIONS) == 317
```

Every value in `QQ_FACE_IMAGE_DESCRIPTIONS` is the complete final text inside
the `<image>...</image>` boundary after escaping. Code must not append `表情`,
`图标`, or any other suffix at runtime.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| First adapter split | Split only NapCat QQ. | It is the large module that now needs a face catalog; changing all adapters would add risk. |
| Package compatibility | Use `__init__.py` re-exports and `__main__.py`. | Preserves existing imports and `python -m` usage while allowing submodules. |
| Package ICD | `src/adapters/napcat_qq_adapter/README.md`. | Documents internal submodule contracts near the code. |
| Adapter class ICD | `src/adapters/README.md`. | Documents the generic adapter responsibility boundary for future adapters. |
| Face catalog source | Use QFace commit `e476a706a7e508849c6031c3654051a02639964f`, `public/assets/qq_emoji/_index.json`, captured on 2026-06-05. | Runtime behavior must be deterministic and offline, and the id set must not drift with live upstream changes. |
| Catalog id domain | Include only ASCII-decimal `emojiId` rows from the QFace snapshot: 317 ids. | NapCat/OneBot `face` CQ handling is numeric; Unicode emoji-style rows are a different message/domain decision. |
| Face label format | Store final semantic descriptions. | Official names are not always LLM-meaningful by themselves. |
| Unknown ids | Omit from semantic projection. | A placeholder would invent visible meaning; full static coverage should handle known QQ faces, and future ids should degrade to no semantic injection. |
| Label generation | Manual/curated semantic label review, assisted by source `describe` values and QFace static image assets when source names are blank or unclear. | Prevents the previous one-id shortcut and avoids formulaic weak labels. |

## Contracts And Data Shapes

### Adapter Package Public Surface

`adapters.napcat_qq_adapter.__init__` must re-export:

```python
NapCatWSAdapter
QQEnvelopeNormalizer
project_qq_semantic_text
runtime_app
main
```

It must not re-export private compatibility constants. Catalog-specific tests
and callers must import `adapters.napcat_qq_adapter.face_catalog` directly.

### CQ Projection Contract

`cq_projection.py` owns:

```python
project_qq_semantic_text(
    raw_wire_text: str,
    platform_bot_id: str,
    display_names: Mapping[str, str],
) -> str
```

It also owns CQ parsing helpers for `reply`, `at`, and `face` markers.

### Face Catalog Contract

`face_catalog.py` owns:

```python
QQ_FACE_IMAGE_DESCRIPTIONS: Mapping[str, str]
qq_face_image_description(face_id: str) -> str | None
```

Rules:

- Keys are strings.
- Numeric QQ face ids are stored without leading/trailing whitespace.
- Values are final LLM-facing Chinese semantic descriptions.
- Values must be non-empty after stripping.
- Values must not contain `<image>` tags.
- Values must not contain raw CQ syntax.
- Values must not contain raw `<`, `>`, or `&` boundary characters.
- Unknown, missing, empty, non-numeric, or unusable ids return `None`.
- Projection code inserts `<image>...</image>` only when `qq_face_image_description()` returns a non-empty string.
- Unknown face-only messages may therefore project to empty text and must be covered by the existing no-content guard.

### Runtime API Binding Contract

`runtime_api.py` owns the FastAPI runtime app, request/response models, send
endpoints, and the active adapter binding. It must not import `ws_adapter.py`.

`runtime_api.py` defines the narrow adapter runtime protocol used by send
endpoints:

```python
class RuntimeNapCatAdapter(Protocol):
    runtime_shared_secret: str

    async def can_send_message(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        ...

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: Sequence[dict] | None = None,
    ) -> SendResult:
        ...
```

It also owns:

```python
bind_runtime_adapter(adapter: RuntimeNapCatAdapter | None) -> None
current_runtime_adapter() -> RuntimeNapCatAdapter | None
```

`ws_adapter.py` calls `bind_runtime_adapter(self)` when the runtime server is
started and calls `bind_runtime_adapter(None)` during close only when it still
owns the active binding. Tests must prove `runtime_api.py` can be imported
without importing `ws_adapter.py`.

### Inbound Segment Contract

`inbound_segments.py` owns conversion of NapCat/OneBot structured segment
lists into the adapter's canonical wire text, including `text`, `face`, `at`,
`reply`, and image/file-like segment detection. It preserves CQ-compatible
markers for `cq_projection.py` and does not call the brain service.

`mention_hydration.py` owns QQ display-name cache lookups and fetch helpers.
`reply_hydration.py` owns `get_msg` reply metadata/excerpt hydration.
`attachments.py` owns image attachment fetch and adapter-side attachment
normalization. `ws_adapter.py` orchestrates websocket lifecycle, event intake,
brain registration, API dispatch, and delegation to these helpers; it must not
own CQ parsing, face catalog lookup, mention-cache internals, reply hydration
details, or attachment-fetching internals.

### Source Snapshot Contract

Add this exact source snapshot fixture:

```text
tests/fixtures/napcat_qq_face_source_snapshot.json
```

The fixture contains metadata and reviewed rows:

```json
{
  "source": {
    "repository": "https://github.com/koishijs/QFace",
    "commit": "e476a706a7e508849c6031c3654051a02639964f",
    "path": "public/assets/qq_emoji/_index.json",
    "captured_at": "2026-06-05",
    "total_rows": 482,
    "numeric_rows": 317,
    "unicode_emoji_rows": 165
  },
  "faces": [
    {
      "id": "344",
      "source_describe": "/大怨种",
      "asset_paths": ["public/assets/qq_emoji/344"],
      "semantic_label": "大怨种表情",
      "label_basis": "source_name",
      "review_status": "reviewed"
    }
  ]
}
```

The production mapping in `face_catalog.py` is generated or copied from
reviewed `semantic_label` values in this fixture. The catalog completeness test
compares production mapping keys against fixture rows whose `id` is an
ASCII-decimal string. The label-quality test asserts every in-scope row has
`review_status == "reviewed"`, a non-empty `semantic_label`, and no placeholder
label such as raw ids, `#419`, `QQ表情`, `表情`, `未知表情`, or `未命名表情`.
Rows with blank or unclear `source_describe`, including known blank-source
rows such as `419` and `420`, must use `label_basis: "asset_review"` or another
explicit accepted source basis; they must not use a placeholder.

### ICD Content Requirements

`src/adapters/README.md` must contain these sections:

- Adapter Responsibility Boundary.
- Required Adapter Lifecycle.
- Optional Runtime Send Interface.
- Message Envelope Contract.
- Runtime Registration Contract.
- Forbidden Adapter Behavior.
- Testing Expectations.

`src/adapters/napcat_qq_adapter/README.md` must contain these sections:

- Public Imports And CLI.
- Submodule Responsibility Table.
- Inbound QQ Segment Flow.
- CQ Projection And Face Catalog Contract.
- Runtime API Binding Contract.
- Unknown Face Omission Contract.
- Source Snapshot And Label Maintenance.
- Verification Commands.

The ICDs must describe contracts and ownership boundaries only; they must not
introduce a new base class, generic adapter framework, prompt contract, or
brain-service behavior.


## LLM Call And Context Budget

- Before: zero runtime LLM calls for QQ face projection.
- After: zero runtime LLM calls for QQ face projection.
- This plan must not add any response-path LLM call.
- Dev-time semantic-label drafting may use external research or manual review, but all final labels must be committed as static data before runtime.
- Runtime prompt context increase is bounded by current-message QQ face count and static description length.

## Change Surface

### Delete

- `src/adapters/napcat_qq_adapter.py`: remove after package files replace its behavior.

### Create

- `src/adapters/README.md`: adapter class/package ICD.
- `src/adapters/napcat_qq_adapter/README.md`: NapCat QQ package ICD.
- `src/adapters/napcat_qq_adapter/__init__.py`: compatibility re-export surface.
- `src/adapters/napcat_qq_adapter/__main__.py`: `python -m adapters.napcat_qq_adapter` entrypoint.
- `src/adapters/napcat_qq_adapter/attachments.py`: image attachment fetch and adapter-side attachment normalization.
- `src/adapters/napcat_qq_adapter/cli.py`: CLI and environment parsing.
- `src/adapters/napcat_qq_adapter/cq_projection.py`: CQ marker parsing and semantic projection.
- `src/adapters/napcat_qq_adapter/envelope_normalizer.py`: `QQEnvelopeNormalizer`.
- `src/adapters/napcat_qq_adapter/face_catalog.py`: static full QQ face semantic catalog.
- `src/adapters/napcat_qq_adapter/inbound_segments.py`: structured NapCat/OneBot segment-list conversion to canonical wire text and attachment markers.
- `src/adapters/napcat_qq_adapter/mention_hydration.py`: QQ display-name cache lookup and hydration helpers.
- `src/adapters/napcat_qq_adapter/outbound.py`: outbound message payload and delivery mention rendering.
- `src/adapters/napcat_qq_adapter/reply_hydration.py`: reply metadata and excerpt hydration.
- `src/adapters/napcat_qq_adapter/runtime_api.py`: FastAPI runtime app, request/response models, send endpoints, runtime adapter binding.
- `src/adapters/napcat_qq_adapter/ws_adapter.py`: `NapCatWSAdapter` websocket lifecycle, event intake, API dispatch, and helper orchestration.
- `tests/fixtures/napcat_qq_face_source_snapshot.json`: accepted QFace source snapshot, source metadata, reviewed semantic labels, and label-review metadata.

### Modify

- `tests/test_adapter_envelope_normalizers.py`: update imports only if needed and add catalog completeness/quality tests.
- `tests/test_runtime_adapter_registration.py`: update imports only if needed and add CLI/package import coverage.
- `docs/HOWTO.md`: add links to the new NapCat and adapter ICDs; do not change the documented command.
- `development_plans/README.md`: register lifecycle status for this plan.

### Keep

- `src/adapters/discord_adapter.py`
- `src/adapters/debug_adapter.py`
- `src/adapters/envelope_common.py`
- `src/kazusa_ai_chatbot/service.py`
- `src/kazusa_ai_chatbot/message_envelope/**`
- Brain service, cognition, RAG, dialog, consolidation, scheduler, and database schema.

## Overdesign Guardrail

- Actual problem: the NapCat adapter has grown too large for safe face-catalog work, and the one-id face mapping is semantically incomplete.
- Minimal change: convert only the NapCat adapter into a package, document adapter contracts, and replace the one-id table with a complete static semantic table.
- Ownership boundaries: adapters own platform syntax and delivery; deterministic adapter code owns QQ face lookup and prompt-safe projection; brain code owns no QQ protocol logic.
- Rejected complexity: generic adapter framework, runtime catalog downloads, database-managed face labels, fake image attachments, new envelope fields, prompt changes, LLM interpretation, brain-side QQ handling, and historical backfill.
- Evidence threshold: only add broader adapter abstractions or runtime catalog management after another adapter needs the same concrete interface or after QQ face data proves too volatile for static maintenance.

## Agent Autonomy Boundaries

- Implementation freedom is limited to the files listed in `Change Surface`.
- The responsible agent may choose small helper names only when they preserve the contracts in this plan.
- The responsible agent must not introduce new architecture, compatibility layers beyond import/CLI preservation, fallback paths, extra runtime services, or unrelated cleanup.
- The responsible agent must not move behavior into shared adapter modules unless it is already shared today or explicitly documented in `src/adapters/README.md`.
- If the complete QQ face source contains rows without usable names, the responsible agent must derive labels by inspecting the source asset or an accepted visual/name source; do not use placeholder ids such as `#419` as final descriptions.
- If a face id in the accepted source snapshot cannot be semantically identified after source inspection, stop and report the blocker; do not add a placeholder or generic label for that id.
- If the plan and code disagree, preserve this plan's ownership boundaries and report the discrepancy.
- If a required behavior cannot be implemented inside the listed change surface, stop and report the blocker.

## Implementation Order

1. Add module-boundary tests proving existing imports still work after conversion.
2. Add CLI smoke test or subprocess test proving `python -m adapters.napcat_qq_adapter --help` exits successfully.
3. Add package ICD expectations: `src/adapters/README.md` and `src/adapters/napcat_qq_adapter/README.md` must exist and name the public contracts.
4. Convert `napcat_qq_adapter.py` into the package layout without changing behavior.
5. Run adapter normalizer and runtime adapter tests; fix only split-induced regressions.
6. Add the accepted QQ face source snapshot fixture.
7. Add catalog completeness tests requiring exact key-set equality against the source snapshot.
8. Add catalog quality tests requiring non-empty labels, no raw ids as labels, no CQ syntax, no `<image>` tags in values, and no formula-only runtime suffixing.
9. Add focused examples for classic numeric face ids, modern numeric face ids, unknown numeric ids, inline unknown ids, and id `344`.
10. Implement the full static catalog in `face_catalog.py`.
11. Wire `cq_projection.py` to use `qq_face_image_description()`.
12. Run the full verification command set.
13. Run independent code review and address only plan-scope findings.
14. Update plan status, progress checklist, registry, and execution evidence after successful implementation and review.

## Focused Test Contract

Update existing unknown-face tests to match the design decision:

- `test_qq_normalizer_projects_unknown_face_as_generic_expression`: replace the expected generic expression with `""`.
- `test_qq_normalizer_projects_face_without_id_as_generic_expression`: replace the expected generic expression with `""`.
- `test_qq_normalizer_preserves_multiple_adjacent_faces`: when the input contains `[CQ:face,id=344][CQ:face,id=999999]`, expect only `<image>大怨种表情</image>`.

Add deterministic tests for these cases:

- `project_qq_semantic_text("[CQ:face,id=999999]", ...) == ""`.
- `project_qq_semantic_text("我[CQ:face,id=999999]服了", ...) == "我服了"`.
- Structured segment input containing only unknown `{"type": "face", "data": {"id": "999999"}}` produces empty `body_text`.
- Structured segment input containing id `344` produces `<image>大怨种表情</image>`.
- The existing service no-content guard skips graph invocation for empty unknown-face-only turns.
- `runtime_api.py` imports without importing `ws_adapter.py`, proving the runtime binding contract avoids a circular import.
- Package-root imports expose only the public surface listed in `Adapter Package Public Surface`; private catalog tests import `adapters.napcat_qq_adapter.face_catalog`.
- Label-quality tests fail if any in-scope numeric row has `review_status != "reviewed"`, blank `semantic_label`, raw id-only label, placeholder label, raw CQ syntax, raw `<image>` tags, or raw boundary characters.

## Execution Model

- Execute only after user approval and status change to `approved` or `in_progress`.
- Parent owns orchestration, test code, verification, execution evidence, review remediation, lifecycle updates, and final sign-off.
- Parent establishes the focused test contract before production implementation starts.
- Production-code subagent: exactly one native subagent after the focused test contract is established; owns production code changes only.
- Parent may continue integration tests, regression tests, static checks, and validation while the production-code subagent edits production code.
- Independent code-review subagent: exactly one native subagent after planned verification passes; reviews the plan, diff, and evidence; reports findings to the parent and does not implement fixes.
- If native subagent capability is unavailable, stop before execution unless the user explicitly requests fallback execution.

## Progress Checklist

- [x] Stage 1A - split contract tests established: import, CLI, and current behavior tests added; expected failures or baseline recorded.
- [x] Stage 1B - NapCat package split complete: package layout created, public imports and `python -m` preserved, behavior tests pass.
- [x] Stage 1C - ICDs complete: adapter class ICD and NapCat package ICD added and linked.
- [x] Stage 2A - face catalog source snapshot established: QFace source snapshot fixture committed with 317 numeric in-scope ids and source metadata.
- [x] Stage 2B - face catalog tests established: completeness and label-quality tests added with expected failures.
- [x] Stage 2C - full semantic catalog implemented: every accepted numeric id has a final LLM-facing description and focused face projection tests pass.
- [x] Stage 3 - verification complete: all commands in `Verification` pass and evidence is recorded.
- [x] Stage 4 - independent code review complete: findings, remediations, reruns, and residual risk recorded.
- [x] Stage 5 - lifecycle closed: plan status and registry updated after approved completion.

## Verification

```powershell
venv\Scripts\python.exe -m py_compile src/adapters/napcat_qq_adapter/__init__.py src/adapters/napcat_qq_adapter/__main__.py src/adapters/napcat_qq_adapter/attachments.py src/adapters/napcat_qq_adapter/cli.py src/adapters/napcat_qq_adapter/cq_projection.py src/adapters/napcat_qq_adapter/envelope_normalizer.py src/adapters/napcat_qq_adapter/face_catalog.py src/adapters/napcat_qq_adapter/inbound_segments.py src/adapters/napcat_qq_adapter/mention_hydration.py src/adapters/napcat_qq_adapter/outbound.py src/adapters/napcat_qq_adapter/reply_hydration.py src/adapters/napcat_qq_adapter/runtime_api.py src/adapters/napcat_qq_adapter/ws_adapter.py tests/test_adapter_envelope_normalizers.py tests/test_runtime_adapter_registration.py
venv\Scripts\python.exe -m pytest tests/test_adapter_envelope_normalizers.py -q
venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py -q
venv\Scripts\python.exe -m pytest tests/test_service_input_queue.py -q
venv\Scripts\python.exe -m adapters.napcat_qq_adapter --help
git diff --check
```

Static greps:

```powershell
rg -n "adapters\\.napcat_qq_adapter\\.py|python -m adapters\\.napcat_qq_adapter" docs README.md tests src development_plans
rg -n "CQ:face|QQ_FACE_IMAGE_DESCRIPTIONS|qq_face_image_description" src tests
rg -n "_QQ_FACE_IMAGE_DESCRIPTIONS|<image>表情</image>" src tests
rg -n "from adapters\\.napcat_qq_adapter import _|_QQ_FACE" src tests
```

Expected grep behavior:

- The first grep may find historical development plans and HOWTO command text, but must not find source imports that assume a `.py` file path.
- The second grep must show QQ face handling only in the NapCat package and tests, not in brain service, cognition, RAG, dialog, persistence, or prompts.
- The third grep must return no source or test matches; unknown faces must not use the old generic placeholder and tests must not patch private root-level constants.
- The fourth grep must return no package-root private catalog imports.

## Independent Plan Review

Run this gate before approval or execution. Review scope:

- The two-stage structure matches the user request: module split and ICDs first, full face catalog second.
- `python -m adapters.napcat_qq_adapter` compatibility is explicitly preserved.
- Existing import surfaces are explicitly preserved.
- The first adapter package split is conservative and does not create a generic framework.
- ICD files are in the correct ownership locations.
- Runtime API binding has an explicit no-circular-import contract.
- The 317 in-scope numeric QQ face ids have an exact source snapshot, completeness test, and label-quality gates.
- Unknown, missing, malformed, and non-numeric face ids are omitted without a placeholder.
- QQ face handling remains adapter-owned with no fake attachments and no brain-specific QQ logic.
- Verification commands are exact and executable on Windows with `venv\Scripts\python.exe`.

## Independent Code Review

Run this gate after all `Verification` commands pass and before final sign-off. Review source diff, tests, ICDs, plan alignment, package import behavior, CLI behavior, CJK string safety, face catalog completeness, face label quality, absence of runtime network/LLM lookup, absence of QQ-specific brain logic, and regression coverage.

The parent agent fixes concrete findings directly only when the fix is inside the approved change surface. If a finding requires new architecture, a different public contract, or edits outside the approved boundary, stop and update the plan or request approval before changing code.

## Acceptance Criteria

- `src/adapters/napcat_qq_adapter.py` is replaced by a package directory.
- `python -m adapters.napcat_qq_adapter --help` exits successfully.
- Existing imports of `NapCatWSAdapter`, `QQEnvelopeNormalizer`, `project_qq_semantic_text`, `runtime_app`, and `main` continue to work.
- NapCat package submodules have single clear responsibilities.
- `src/adapters/README.md` defines the adapter class/package ICD with every section listed in `ICD Content Requirements`.
- `src/adapters/napcat_qq_adapter/README.md` defines the NapCat package ICD with every section listed in `ICD Content Requirements`.
- `tests/fixtures/napcat_qq_face_source_snapshot.json` records QFace commit `e476a706a7e508849c6031c3654051a02639964f`, `public/assets/qq_emoji/_index.json`, 482 total rows, 317 numeric rows, and 165 Unicode emoji rows.
- `QQ_FACE_IMAGE_DESCRIPTIONS` contains exactly the 317 accepted numeric QQ face ids.
- Every catalog value is a meaningful final LLM-facing semantic label.
- Labels are not generated at runtime by appending one universal suffix.
- Unknown ids are omitted from projection even inside adjacent text; for example `我[CQ:face,id=999999]服了` becomes `我服了`.
- Unknown or malformed face ids do not inject `<image>` placeholders and do not expose the raw id in `body_text`.
- Unknown face-only messages project to empty text and are handled by the existing no-content graph-skip path.
- Existing id `344` still produces `<image>大怨种表情</image>`.
- QQ face projection works for CQ strings, structured segment lists, reply excerpts, inline text adjacency, and adjacent faces.
- `runtime_api.py` has no import dependency on `ws_adapter.py`; binding occurs through the `RuntimeNapCatAdapter` protocol functions in `runtime_api.py`.
- No runtime LLM call, network fetch, database lookup, fake attachment, new envelope field, prompt change, or QQ-specific brain logic is added.
- All verification commands pass.

## Risks

| Risk | Mitigation | Verification |
|---|---|---|
| Package conversion breaks imports. | Re-export public symbols from `__init__.py`. | Import tests and runtime adapter tests. |
| `python -m` breaks after module becomes package. | Add `__main__.py` calling `main()`. | `venv\Scripts\python.exe -m adapters.napcat_qq_adapter --help`. |
| Over-splitting makes first package harder to maintain. | Split only by existing responsibility boundaries. | Independent plan/code review checks module responsibilities. |
| Face catalog is incomplete. | Commit accepted id snapshot and assert exact key equality. | Catalog completeness test. |
| Labels are weak or formulaic. | Require source name plus manual semantic review; inspect visual assets for unclear names. | Label quality tests and independent code review. |
| Future QQ ids appear. | Omit unknown ids until the static catalog is updated. | Unknown-id omission tests and no-content regression tests. |

## Execution Evidence

- Draft created on 2026-06-05 after closing `qq_face_projection_empty_input_guard_bugfix_plan.md`.
- Source discovery notes: NapCat docs confirm `face` message segments; OneBot docs confirm `[CQ:face,id=123]`; Koishi QFace provides a broad current id/name/resource list including `344`.
- Accepted QFace snapshot: repository `https://github.com/koishijs/QFace`, commit `e476a706a7e508849c6031c3654051a02639964f`, file `public/assets/qq_emoji/_index.json`, captured on 2026-06-05, 482 total rows, 317 numeric `emojiId` rows in scope, and 165 Unicode emoji-style rows deferred.
- 2026-06-05 Stage 1A parent test contract: `venv\Scripts\python.exe -m py_compile tests/test_adapter_envelope_normalizers.py tests/test_runtime_adapter_registration.py` passed. `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_napcat_module_cli_help_exits_successfully -q` passed. `venv\Scripts\python.exe -m pytest tests/test_adapter_envelope_normalizers.py -q` produced the expected pre-implementation baseline: 9 failed and 11 passed; failures were missing package shape, missing ICDs, missing face source snapshot/catalog submodule, unknown faces still rendering the old generic image text, and catalog monkeypatch unable to import `face_catalog`. `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py::test_napcat_runtime_api_import_does_not_load_ws_adapter tests/test_runtime_adapter_registration.py::test_napcat_handle_event_omits_unknown_segment_list_face -q` produced the expected pre-implementation baseline: 2 failed; failures were missing `runtime_api` package submodule and structured unknown face still rendering the old generic image text.
- 2026-06-05 Stage 1B implementation: production-code subagent converted `src/adapters/napcat_qq_adapter.py` into `src/adapters/napcat_qq_adapter/`, preserved package-root public imports, preserved `python -m adapters.napcat_qq_adapter`, split runtime API binding away from websocket implementation, and retained legacy test/debug access without exposing the QQ face catalog at package root.
- 2026-06-05 Stage 1C ICDs: `src/adapters/README.md` and `src/adapters/napcat_qq_adapter/README.md` added and linked from `docs/HOWTO.md`.
- 2026-06-05 Stage 2 implementation: `tests/fixtures/napcat_qq_face_source_snapshot.json` added with the accepted QFace snapshot metadata and reviewed labels; `face_catalog.py` contains 317 accepted numeric ids, including `344 -> 大怨种表情`; unknown, missing, malformed, and non-numeric ids are omitted.
- 2026-06-05 Stage 3 verification: `venv\Scripts\python.exe -m py_compile ...` for the NapCat package and touched tests passed; `venv\Scripts\python.exe -m pytest tests/test_adapter_envelope_normalizers.py -q` passed 20 tests; `venv\Scripts\python.exe -m pytest tests/test_runtime_adapter_registration.py -q` passed 55 tests; `venv\Scripts\python.exe -m pytest tests/test_service_input_queue.py -q` passed 33 tests; `venv\Scripts\python.exe -m adapters.napcat_qq_adapter --help` passed with dotenv disabled; `git diff --check --cached` and `git diff --check` passed.
- 2026-06-05 Stage 3 static checks: no stale dotted NapCat module-file references; QQ face handling grep hits are limited to the NapCat package and tests; `_QQ_FACE_IMAGE_DESCRIPTIONS`, `<image>表情</image>`, package-root private catalog imports, and `_QQ_FACE` imports have no source or test matches.
- 2026-06-05 Stage 4 independent code review: subagent `019e9741-f498-73d0-9253-73740329f91a` reported one medium finding and one minor finding. Medium: package-root `__getattr__` still exposed `_MENTION_DISPLAY_CACHE_LIMIT` and a test depended on that private root symbol. Minor: the runtime adapter registration test contained CJK string literals with double-quoted delimiters.
- 2026-06-05 Stage 4 remediation: removed the package-root legacy `_MENTION_DISPLAY_CACHE_LIMIT` lookup and root `asyncio` import, changed cache/timeout tests to import and patch `adapters.napcat_qq_adapter.mention_hydration`, added package-root assertions that `_MENTION_DISPLAY_CACHE_LIMIT` and `asyncio` are not exposed, and converted remaining CJK literals in `tests/test_runtime_adapter_registration.py` to single-quoted delimiters.
- 2026-06-05 Stage 4 reruns: targeted `py_compile` passed for `src/adapters/napcat_qq_adapter/__init__.py`, `tests/test_adapter_envelope_normalizers.py`, and `tests/test_runtime_adapter_registration.py`; targeted review-fix tests passed 3 tests; full plan verification reruns passed `tests/test_adapter_envelope_normalizers.py` 20 tests, `tests/test_runtime_adapter_registration.py` 55 tests, `tests/test_service_input_queue.py` 33 tests, `venv\Scripts\python.exe -m adapters.napcat_qq_adapter --help`, full package/test `py_compile`, `git diff --check --cached`, and `git diff --check`. CJK double-quoted literal scans are clean for touched NapCat package files and touched tests. Residual risk: no live NapCat server was exercised; verification is deterministic unit and integration coverage only.
- 2026-06-05 Stage 5 lifecycle closure: status changed to `completed`; registry updated; plan archived under `development_plans/archive/completed/short_term/` after implementation, verification, and independent code review gates completed.
