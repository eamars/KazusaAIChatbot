# self cognition character global id config bugfix plan

## Summary

- Goal: stop self-cognition resolver RAG ticks from failing with
  `character_profile.global_user_id is required` by enforcing the existing
  mandatory `CHARACTER_GLOBAL_USER_ID` config contract and projecting that
  identity into self-cognition graph-facing character profiles.
- Plan class: small
- Status: completed
- Mandatory skills: `development-plan`, `local-llm-architecture`,
  `py-style`, `test-style-and-execution`
- Overall cutover strategy: bigbang for character identity config loading and
  self-cognition case projection.
- Highest-risk areas: accidentally adding a RAG fallback, weakening mandatory
  config fail-fast behavior, or changing prompt-facing participant identity
  semantics.
- Acceptance criteria: config import fails when `CHARACTER_GLOBAL_USER_ID` is
  absent or empty after `.env` loading; self-cognition cases always carry the
  configured character id in `case["character_profile"]["global_user_id"]`;
  the RAG cognitive episode adapter remains unchanged.

## Context

Production reflection worker error:

```text
kazusa_ai_chatbot.rag.cognitive_episode_adapter.RAGEpisodeAdapterError:
character_profile.global_user_id is required
```

The failure path is:

```text
reflection_cycle.worker
  -> self_cognition.worker
  -> self_cognition.runner._build_cognition_state(...)
  -> cognition_resolver.capabilities.run_rag_evidence_for_persona_state(...)
  -> rag.cognitive_episode_adapter.build_text_chat_rag_request(...)
  -> _character_identity(...)
```

`build_text_chat_rag_request(...)` correctly requires
`character_profile.global_user_id`. Normal chat receives this identity through
`compose_character_profile(...)`, but self-cognition source cases store a
projected character profile through
`self_cognition.sources._project_character_profile(...)`. That projection
currently omits `global_user_id`.

`CHARACTER_GLOBAL_USER_ID` is already the system-level character identity
source. It is listed in `docs/HOWTO.md`, and `self_cognition.sources` already
imports it for group activity-window construction. The bug is that config still
has a hardcoded fallback UUID and the self-cognition case projection does not
copy the configured value into the graph-facing profile.

## Mandatory Skills

- `development-plan`: load before editing, approving, executing, reviewing,
  archiving, or signing off this plan.
- `local-llm-architecture`: load before changing resolver, RAG, cognition, or
  self-cognition source boundaries.
- `py-style`: load before editing Python files.
- `test-style-and-execution`: load before adding, changing, or running tests.

## Mandatory Rules

- Do not execute production-code changes while this plan status is `draft`.
- Implementation also requires explicit user approval, because this changes
  production code.
- Use `venv\Scripts\python.exe` for Python commands.
- Use `apply_patch` for manual source, test, docs, and plan edits.
- Check `git status --short` before editing.
- Do not read `.env`.
- Do not change the RAG cognitive episode adapter. Its strict
  `character_profile.global_user_id` validation is correct.
- Do not add runner-side recovery, reflection-worker special casing,
  database reads, compatibility shims, feature flags, retries, or fallback
  RAG behavior.
- Do not change prompts, LLM calls, cognition resolver routing, participant
  context hydration, adapter delivery, consolidation, scheduler behavior, or
  database schema.
- Preserve the recent participant-context missing-user-id behavior:
  participant rows without a user `global_user_id` still degrade to visible-only
  participant context. This plan concerns the character's configured
  `global_user_id`, not participant identity.
- After any automatic context compaction, reread this entire plan before
  continuing implementation, verification, lifecycle updates, or final
  reporting.
- Before completion, lifecycle status changes, merge, or sign-off, run the
  `Independent Code Review` gate and record the result in `Execution Evidence`.

## Must Do

- Make `CHARACTER_GLOBAL_USER_ID` required at config import time.
- Remove the hardcoded fallback character UUID from `src/kazusa_ai_chatbot/config.py`.
- Use the existing `_non_empty_string_from_env(...)` helper so empty strings
  fail with a clear `ValueError`.
- Update config subprocess test helpers so normal config tests provide a
  valid `CHARACTER_GLOBAL_USER_ID`.
- Add config tests proving import fails when `CHARACTER_GLOBAL_USER_ID` is
  missing or empty after `.env` loading.
- Update `self_cognition.sources._project_character_profile(...)` to set
  `projected["global_user_id"]` from the incoming profile when present, or
  from the mandatory `CHARACTER_GLOBAL_USER_ID` otherwise.
- Add a self-cognition source projection regression proving a group-review case
  whose input profile lacks `global_user_id` still emits a case profile with
  the configured character id.
- Update `docs/HOWTO.md` to state that `CHARACTER_GLOBAL_USER_ID` is required
  and has no built-in default.
- Keep the fix limited to config loading, self-cognition source projection,
  focused tests, docs, and this plan.

## Deferred

- Do not redesign character profile storage.
- Do not make `get_character_profile()` compose runtime identity.
- Do not add a service or reflection profile provider just for group review.
- Do not modify `runner._character_profile(...)` as a recovery layer.
- Do not add a RAG-side default for missing character identity.
- Do not add participant display-name lookup, profile hydration fallback, or
  any change to `group_review_participant_context.py`.
- Do not change the configured UUID value in operator environments.

## Cutover Policy

Overall strategy: bigbang.

| Area | Policy | Instruction |
|---|---|---|
| Config loading | bigbang | Missing or empty `CHARACTER_GLOBAL_USER_ID` crashes import after `load_dotenv()`. |
| Self-cognition case projection | bigbang | Every projected self-cognition case profile receives the configured character id. |
| RAG adapter | unchanged | Keep strict validation and current tests. |
| Participant missing-id behavior | unchanged | Missing participant user ids still degrade to visible-only context. |

## Target State

`src/kazusa_ai_chatbot/config.py`:

```python
CHARACTER_GLOBAL_USER_ID = _non_empty_string_from_env(
    "CHARACTER_GLOBAL_USER_ID",
    "",
)
```

`src/kazusa_ai_chatbot/self_cognition/sources.py`:

```python
def _project_character_profile(
    character_profile: dict[str, Any],
) -> dict[str, Any]:
    """Project graph-facing character fields for worker cases."""

    projected = {
        field_name: character_profile[field_name]
        for field_name in _CHARACTER_PROFILE_FIELDS
        if field_name in character_profile
    }
    if "name" not in projected:
        projected["name"] = "active character"
    projected["global_user_id"] = (
        text_or_empty(character_profile.get("global_user_id"))
        or CHARACTER_GLOBAL_USER_ID
    )
    return projected
```

If the implementation chooses to keep the docstring wording exactly as-is, the
behavior above is still mandatory.

## Design Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Character identity source | `CHARACTER_GLOBAL_USER_ID` from config only. | This is the existing system-level identity and is already used by service composition and group window construction. |
| Fail-fast behavior | Config import fails when the variable is absent or empty. | A mandatory character id should not silently become a fallback UUID. |
| Projection location | `self_cognition.sources._project_character_profile(...)`. | This is the boundary where source cases become graph-facing profiles for all self-cognition case types. |
| RAG adapter | No change. | The adapter's error is correct validation of the graph-facing profile contract. |
| Participant context | No change. | Recent missing-id handling is about participant user identity, not character identity. |

## Change Surface

### Modify

- `src/kazusa_ai_chatbot/config.py`
  - Replace the hardcoded `CHARACTER_GLOBAL_USER_ID` fallback with
    `_non_empty_string_from_env("CHARACTER_GLOBAL_USER_ID", "")`.
- `src/kazusa_ai_chatbot/self_cognition/sources.py`
  - Project `global_user_id` into the case character profile from the input
    profile or the mandatory configured id.
- `tests/test_config.py`
  - Add `CHARACTER_GLOBAL_USER_ID` to
    `_configured_subprocess_env_without_dotenv()`.
  - Update the manually populated environment in
    `TestRouteLlmConfig.test_missing_route_config_crashes_import(...)` so it
    also sets `CHARACTER_GLOBAL_USER_ID = "character-global"` before deleting
    `COGNITION_LLM_MODEL`; this preserves the test's intended missing-route
    failure.
  - Add missing and empty character id import-failure tests.
- `tests/test_self_cognition_group_review_source.py`
  - Add a focused projection regression for group-review cases.
- `tests/conftest.py`
  - Provide a deterministic test-process `CHARACTER_GLOBAL_USER_ID` default
    before collection-time imports so regular tests do not depend on a local
    `.env`; subprocess failure tests still control their own env explicitly.
- `docs/HOWTO.md`
  - Document that `CHARACTER_GLOBAL_USER_ID` is required and has no built-in
    default immediately after the paragraph that says route-specific chat model
    variables are required.
- `development_plans/README.md`
  - Add this plan to Active Bugfix Plans.

### Keep

- `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`
- `src/kazusa_ai_chatbot/cognition_resolver/*`
- `src/kazusa_ai_chatbot/self_cognition/runner.py`
- `src/kazusa_ai_chatbot/self_cognition/group_review_participant_context.py`
- Reflection worker scheduling, adapter delivery, consolidation, scheduler,
  prompts, LLM routes, and database schema.

## Overdesign Guardrail

- Actual problem: mandatory character identity is not enforced at config load
  and is dropped by self-cognition case profile projection before RAG.
- Minimal change: make config fail fast and carry the configured id through the
  existing self-cognition source projection.
- Ownership boundaries: config owns mandatory process settings; self-cognition
  sources own case projection; RAG owns validation of the request boundary.
- Rejected complexity: runner rescue, reflection-only fix, RAG fallback,
  service cache dependency, DB reads, profile-store migration, prompt changes,
  feature flags, retries, and participant identity redesign.
- Evidence threshold: revisit broader character-profile composition only after
  a separate confirmed failure shows another background subsystem bypassing
  mandatory graph-facing identity.

## Implementation Order

1. Load mandatory skills and reread this plan.
2. Run `git status --short`.
3. Add the config tests in `tests/test_config.py`.
   - Update `_configured_subprocess_env_without_dotenv()`:

```python
    env["CHARACTER_GLOBAL_USER_ID"] = "character-global"
```

   - Update
     `TestRouteLlmConfig.test_missing_route_config_crashes_import(...)` after
     embedding env setup and before `del env["COGNITION_LLM_MODEL"]`:

```python
        env["CHARACTER_GLOBAL_USER_ID"] = "character-global"
```

   - Add test:

```python
    def test_missing_character_global_user_id_crashes_import(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env.pop("CHARACTER_GLOBAL_USER_ID", None)

        result = subprocess.run(
            [sys.executable, "-c", "import kazusa_ai_chatbot.config"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "CHARACTER_GLOBAL_USER_ID must be non-empty" in result.stderr
```

   - Add test:

```python
    def test_empty_character_global_user_id_crashes_import(self, tmp_path):
        env = _configured_subprocess_env_without_dotenv()
        env["CHARACTER_GLOBAL_USER_ID"] = ""

        result = subprocess.run(
            [sys.executable, "-c", "import kazusa_ai_chatbot.config"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0
        assert "CHARACTER_GLOBAL_USER_ID must be non-empty" in result.stderr
```

4. Run the new config tests before implementation and record expected failure:

```powershell
venv\Scripts\python.exe -m pytest tests/test_config.py::TestRouteLlmConfig::test_missing_character_global_user_id_crashes_import tests/test_config.py::TestRouteLlmConfig::test_empty_character_global_user_id_crashes_import -q
```

Expected before implementation: both tests fail because config import succeeds
with the hardcoded fallback or empty behavior does not match the required
message.

5. Add a self-cognition group-review projection regression in
   `tests/test_self_cognition_group_review_source.py`.

```python
@pytest.mark.asyncio
async def test_group_review_case_profile_uses_configured_character_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Projected group-review cases should carry configured character identity."""

    now = datetime(2026, 5, 18, 4, 45, tzinfo=timezone.utc)
    monkeypatch.setattr(
        sources,
        "CHARACTER_GLOBAL_USER_ID",
        "configured-character",
    )

    async def build_participant_context(**kwargs: Any) -> None:
        del kwargs
        return None

    monkeypatch.setattr(
        sources,
        "build_group_review_participant_context",
        build_participant_context,
    )

    async def collect_inputs(**kwargs: Any) -> ReflectionInputSet:
        assert kwargs == {
            "lookback_hours": 3,
            "now": now,
            "allow_fallback": False,
        }
        return _input_set([_group_scope()])

    cases = await sources.collect_group_chat_review_cases(
        now=now,
        character_profile={
            "name": "Character",
            "mood": "focused",
            "platform_bot_id": "bot-1",
        },
        max_cases=1,
        collect_reflection_inputs_func=collect_inputs,
    )

    assert cases
    for case in cases:
        assert case["character_profile"]["global_user_id"] == (
            "configured-character"
        )
    assert cases[0]["character_profile"]["global_user_id"] == (
        "configured-character"
    )
```

6. Run the self-cognition projection test before implementation and record
   expected failure:

```powershell
venv\Scripts\python.exe -m pytest tests/test_self_cognition_group_review_source.py::test_group_review_case_profile_uses_configured_character_id -q
```

Expected before implementation: fail with missing `global_user_id` or a
`KeyError`, because `_project_character_profile(...)` omits the field.

7. Implement `src/kazusa_ai_chatbot/config.py` change:

```python
CHARACTER_GLOBAL_USER_ID = _non_empty_string_from_env(
    "CHARACTER_GLOBAL_USER_ID",
    "",
)
```

8. Implement `src/kazusa_ai_chatbot/self_cognition/sources.py` projection
   change:

```python
    projected["global_user_id"] = (
        text_or_empty(character_profile.get("global_user_id"))
        or CHARACTER_GLOBAL_USER_ID
    )
```

9. Update `docs/HOWTO.md` immediately after the paragraph beginning
   `All route-specific chat model variables are required.` to state:

```text
`CHARACTER_GLOBAL_USER_ID` is required and has no built-in default. The service
loads `.env` before reading config, then fails fast if the value is missing or
empty.
```

10. Run focused tests:

```powershell
venv\Scripts\python.exe -m pytest tests/test_config.py::TestRouteLlmConfig tests/test_self_cognition_group_review_source.py -q
```

Expected after implementation: pass.

11. Run adjacent RAG/self-cognition tests:

```powershell
venv\Scripts\python.exe -m pytest tests/test_rag_cognitive_episode_adapter.py tests/test_cognition_resolver_loop.py::test_internal_thought_rag_capability_uses_existing_rag_path tests/test_self_cognition_group_review_participant_context.py -q
```

Expected after implementation: pass.

12. Run static checks:

```powershell
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/config.py src/kazusa_ai_chatbot/self_cognition/sources.py tests/test_config.py tests/test_self_cognition_group_review_source.py
rg -n "CHARACTER_GLOBAL_USER_ID = os.getenv|00000000-0000-4000-8000-000000000001" src/kazusa_ai_chatbot/config.py
```

Expected: `py_compile` exits 0; the `rg` command returns no matches. The
no-match nonzero exit code is acceptable for this `rg` command.

13. Run `git diff --check`.
14. Run the Independent Code Review gate.
15. Record all command output summaries and review findings in
    `Execution Evidence`.

## Execution Model

- Execute only after user approval and status change to `approved` or
  `in_progress`.
- Parent agent owns orchestration, test creation, implementation, verification,
  evidence, review remediation, lifecycle updates, and final sign-off.
- If subagents are available and the user approves execution, implementation
  can stay inline because this plan is small and has only two production-code
  touch points.

## Progress Checklist

- [x] Stage 1 - failure tests added
  - Covers: config missing/empty tests and self-cognition projection regression.
  - Verify: focused tests fail before implementation for the expected reason.
  - Evidence: record commands and failure summaries in `Execution Evidence`.
  - Sign-off: Codex on 2026-06-02 after red-test evidence was recorded.

- [x] Stage 2 - mandatory config and projection implemented
  - Covers: `config.py`, `self_cognition/sources.py`, and `docs/HOWTO.md`.
  - Verify: focused tests and `py_compile` pass.
  - Evidence: record changed files and focused test output.
  - Sign-off: Codex on 2026-06-02 after focused tests and compile evidence
    were recorded.

- [x] Stage 3 - adjacent regression verification complete
  - Covers: RAG adapter tests, internal-thought RAG resolver test, and
    participant-context tests.
  - Verify: all commands in `Verification` pass or have approved documented
    blockers.
  - Evidence: record test and static-check outputs.
  - Sign-off: Codex on 2026-06-02 after adjacent regression and static-check
    evidence were recorded.

- [x] Stage 4 - independent code review complete
  - Covers: plan alignment, diff review, test adequacy, and no overdesign.
  - Verify: findings are closed or explicitly accepted as residual risk.
  - Evidence: record findings, fixes, reruns, and approval status.
  - Sign-off: Codex on 2026-06-02 after inline independent-review evidence was
    recorded.

## Verification

```powershell
venv\Scripts\python.exe -m pytest tests/test_config.py::TestRouteLlmConfig tests/test_self_cognition_group_review_source.py -q
venv\Scripts\python.exe -m pytest tests/test_rag_cognitive_episode_adapter.py tests/test_cognition_resolver_loop.py::test_internal_thought_rag_capability_uses_existing_rag_path tests/test_self_cognition_group_review_participant_context.py -q
venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/config.py src/kazusa_ai_chatbot/self_cognition/sources.py tests/test_config.py tests/test_self_cognition_group_review_source.py
rg -n "CHARACTER_GLOBAL_USER_ID = os.getenv|00000000-0000-4000-8000-000000000001" src/kazusa_ai_chatbot/config.py
git diff --check
```

Expected:

- Pytest commands pass.
- `py_compile` exits 0.
- The `rg` command returns no matches; its no-match nonzero exit code is
  acceptable.
- `git diff --check` exits 0.

## Independent Plan Review

Review this draft before approval. Review scope:

- The plan uses the mandatory config contract, not local recovery.
- The RAG adapter remains unchanged.
- The self-cognition source projection is the only graph-profile fix point.
- Participant missing-user-id degradation remains unchanged.
- No DB reads, prompt edits, runner fallback, reflection-only special casing,
  feature flags, retries, or new compatibility paths are authorized.
- Verification includes fail-fast config import and self-cognition case
  projection coverage.

## Independent Code Review

Run after implementation verification passes and before final sign-off. Review
the diff against this plan, with special attention to:

- no hardcoded fallback UUID remains in config;
- missing and empty `CHARACTER_GLOBAL_USER_ID` crash config import;
- `_project_character_profile(...)` always emits `global_user_id`;
- no changes to RAG adapter strict validation;
- no changes to participant context missing-id semantics;
- tests fail before and pass after the implementation.

## Acceptance Criteria

- `kazusa_ai_chatbot.config` import fails when `CHARACTER_GLOBAL_USER_ID` is
  missing from both process env and `.env`.
- `kazusa_ai_chatbot.config` import fails when `CHARACTER_GLOBAL_USER_ID` is
  empty.
- The hardcoded fallback UUID is removed from config.
- Self-cognition projected case profiles include
  `character_profile.global_user_id`.
- Internal-thought RAG capability can continue using the existing RAG adapter
  without changing adapter validation.
- Participant-context tests for missing participant user ids still pass.
- `docs/HOWTO.md` states that `CHARACTER_GLOBAL_USER_ID` is required and has
  no built-in default.
- All verification commands pass or have user-approved documented blockers.

## Execution Evidence

- Plan creation: draft plan created in
  `development_plans/active/bugfix/` on 2026-06-02.
- Independent plan review: completed on 2026-06-02 before implementation.
  Fixed known issues: corrected the stale route-test class references to the
  actual class name; added explicit instruction to preserve the existing
  missing-route config test by setting `CHARACTER_GLOBAL_USER_ID`; replaced
  the nonexistent group activity helper reference with the existing
  `_input_set([_group_scope()])` fixture path; added `platform_bot_id="bot-1"`
  to the projection regression input; clarified the exact HOWTO insertion
  point; removed placeholder sign-off and evidence text.
- Plan review verification: stale-name and placeholder scan returned no
  matches; `git diff --check` exited 0 with only the existing README LF/CRLF
  warning.
- Implementation authorization: user approved fallback inline execution on
  2026-06-02 with "Execute the plan without subagent"; lifecycle status moved
  to `in_progress` before production-code edits.
- Stage 1 evidence: focused config red test command
  `venv\Scripts\python.exe -m pytest tests/test_config.py::TestRouteLlmConfig::test_missing_character_global_user_id_crashes_import tests/test_config.py::TestRouteLlmConfig::test_empty_character_global_user_id_crashes_import -q`
  exited 1 as expected; both tests failed because config import returned 0.
  Focused projection red test command
  `venv\Scripts\python.exe -m pytest tests/test_self_cognition_group_review_source.py::test_group_review_case_profile_uses_configured_character_id -q`
  exited 1 as expected with `KeyError: 'global_user_id'`.
- Stage 2 evidence: changed `src/kazusa_ai_chatbot/config.py`,
  `src/kazusa_ai_chatbot/self_cognition/sources.py`, `docs/HOWTO.md`, and
  focused tests. Added `tests/conftest.py` test-harness support because
  mandatory config now affects collection-time imports. Focused command
  `venv\Scripts\python.exe -m pytest tests/test_config.py::TestRouteLlmConfig tests/test_self_cognition_group_review_source.py -q`
  exited 0 with 16 passed. Compile command
  `venv\Scripts\python.exe -m py_compile src/kazusa_ai_chatbot/config.py src/kazusa_ai_chatbot/self_cognition/sources.py tests/test_config.py tests/test_self_cognition_group_review_source.py tests/conftest.py`
  exited 0.
- Stage 3 evidence: adjacent command
  `venv\Scripts\python.exe -m pytest tests/test_rag_cognitive_episode_adapter.py tests/test_cognition_resolver_loop.py::test_internal_thought_rag_capability_uses_existing_rag_path tests/test_self_cognition_group_review_participant_context.py -q`
  exited 0 with 17 passed. Static compile command from the `Verification`
  section exited 0. Forbidden fallback grep
  `rg -n "CHARACTER_GLOBAL_USER_ID = os.getenv|00000000-0000-4000-8000-000000000001" src/kazusa_ai_chatbot/config.py`
  returned no matches with exit code 1, which is the expected no-match result.
  `git diff --check` exited 0 with LF/CRLF warnings only.
- Stage 4 evidence: inline independent review completed on 2026-06-02 because
  the user requested fallback execution without subagents. Reviewed full diff,
  changed-file set, and plan alignment. Findings: none. Verification review:
  focused tests passed, adjacent tests passed, compile passed, forbidden
  fallback grep returned no matches, `git diff --check` exited 0 with LF/CRLF
  warnings only. Boundary review: no diff in
  `src/kazusa_ai_chatbot/rag/cognitive_episode_adapter.py`,
  `src/kazusa_ai_chatbot/self_cognition/runner.py`,
  `src/kazusa_ai_chatbot/self_cognition/group_review_participant_context.py`,
  or `src/kazusa_ai_chatbot/cognition_resolver/`.
- Final suite evidence: finishing-branch verification command
  `venv\Scripts\python.exe -m pytest -q` exited 0 with 1864 passed and
  269 deselected.
- External independent review: subagent reviewer reported no defects, no scope
  drift, and coherent registry/archive state. The only residual gap was missing
  explicit coverage that a non-empty incoming
  `character_profile["global_user_id"]` takes precedence over config.
  Remediation added
  `test_group_review_case_profile_preserves_profile_character_id`.
  Follow-up focused command
  `venv\Scripts\python.exe -m pytest tests/test_config.py::TestRouteLlmConfig tests/test_self_cognition_group_review_source.py -q`
  exited 0 with 17 passed; compile and `git diff --check` exited 0.
- Final post-review suite evidence: after the review-gap test was added,
  `venv\Scripts\python.exe -m pytest -q` exited 0 with 1865 passed and
  269 deselected.
