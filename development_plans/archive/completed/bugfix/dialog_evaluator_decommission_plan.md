# Dialog Evaluator Decommission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Status:** completed

**Goal:** Remove the dialog evaluator from the normal live dialog path and leave dialog delivery governed by cognition/L3 contracts, one dialog-generator call, deterministic validation, and offline real LLM review.

**Architecture:** The dialog subgraph becomes a direct generator-only path: `START -> generator -> END`. Runtime LLM judging, evaluator prompt feedback, evaluator retry, and evaluator route configuration are removed from production dialog delivery. Existing prompt-quality evaluation stays in live LLM tests and human-readable artifacts rather than a second online resolver.

**Tech Stack:** Python, LangGraph `StateGraph`, pytest, project `LLInterface`, PowerShell verification commands.

---

## Scope And Ownership

The evaluator is being removed because it repeats generator/resolver interpretation with weaker context and has proven correlated failure behavior. This plan does not change cognition, L3 content planning, RAG, adapters, persistence, consolidation, or scheduler behavior except where tests/docs mention the dialog evaluator route.

Runtime ownership after this change:

```text
cognition / resolver decides stance, action, and whether to respond
L3 content plan decides visible content, interaction action, and posture
dialog generator renders one visible chat bubble
deterministic code validates parser shape and delivery metadata
offline real LLM tests provide quality review
```

Rejected complexity:

- No replacement LLM evaluator.
- No automatic cognition rerun from dialog.
- No new retry loop.
- No compatibility wrapper around the removed evaluator.
- No new config flag to keep both old and new paths.

## Files Expected To Change

- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - Remove `_DIALOG_EVALUATOR_PROMPT`.
  - Remove `_dialog_evaluator_llm`, `_dialog_evaluator_llm_config`, and `dialog_evaluator`.
  - Remove evaluator node, conditional retry edge, evaluator feedback handling, and retry state from dialog graph.
  - Keep generator parsing and final dialog assembly.
  - Keep or minimally rename event logging only where needed to avoid schema churn.

- `src/kazusa_ai_chatbot/config.py`
  - Remove dialog evaluator route constants only if the blast-radius scan confirms no non-dialog runtime consumer remains.
  - Keep unrelated `MAX_DIALOG_AGENT_RETRY` only if another dialog retry path still uses it; otherwise remove or deprecate with docs/tests.

- `README.md`
  - Remove `DIALOG_EVALUATOR_LLM` from route examples.

- `docs/HOWTO.md`
  - Remove `DIALOG_EVALUATOR_LLM_*` from setup examples and required route descriptions.

- `tests/test_dialog_agent.py`
  - Refactor patched graph tests to expect a generator-only path.
  - Remove evaluator prompt tests.
  - Keep generator prompt/content-plan tests.

- `tests/test_conversation_progress_history_policy.py`
  - Remove evaluator payload-history tests.
  - Keep generator payload-history tests.

- `tests/test_dialog_l3_surface_contract_live_llm.py`
  - Remove `_dialog_evaluator_llm` wrapping and evaluator route availability checks.
  - Keep real LLM dialog-generator output capture and full dialog traces.

- `tests/test_dialog_evaluator_live_llm_contract.py`
  - Delete or archive from active pytest collection because the evaluator no longer exists in runtime.

- Other tests surfaced by Step 1
  - Update imports, patches, event logging expectations, and docs assertions surfaced by the scan.

## Task 1: Deep Scan And Blast-Radius Log

**Files:**
- Create: `test_artifacts/analysis/dialog_evaluator_removal_scan_20260616.md`

- [x] **Step 1: Run broad evaluator scans**

Run:

```powershell
rg -n -i "dialog_evaluator|DIALOG_EVALUATOR|_DIALOG_EVALUATOR_PROMPT|_dialog_evaluator|MAX_DIALOG_AGENT_RETRY|Evaluator Feedback|dialog_quality|record_dialog_quality_event|invalid_evaluator_output|should_stop" . --glob '!venv/**' --glob '!__pycache__/**'
```

Expected: a complete hit list across source, tests, docs, plans, and test artifacts.

- [x] **Step 2: Run related-name scans**

Run:

```powershell
rg -n -i "dialog_evaluator_live_llm_contract|evaluator_status|reject_owner_flipped|reject_unanchored|owner_flipped_guess|dialog evaluator" . --glob '!venv/**' --glob '!__pycache__/**'
```

Expected: additional hits for evaluator-only live tests, event logging schema references, and historical docs/artifacts.

- [x] **Step 3: Classify all surfaced files**

Write `test_artifacts/analysis/dialog_evaluator_removal_scan_20260616.md` with these categories:

```markdown
# Dialog Evaluator Removal Scan

## Runtime Remove
| File | Reason |
| --- | --- |

## Runtime Refactor
| File | Reason |
| --- | --- |

## Tests Refactor Or Remove
| File | Reason |
| --- | --- |

## Docs Update
| File | Reason |
| --- | --- |

## Historical Or Artifact Keep
| File | Reason |
| --- | --- |

## Non-Dialog Keep
| File | Reason |
| --- | --- |

## Review Before Editing
| File | Reason |
| --- | --- |
```

- [x] **Step 4: Stop for scan review**

Review the scan artifact before production edits. Confirm the edit set is still limited to dialog runtime, route docs/config, and directly affected tests.

## Task 2: Deterministic Tests First

**Files:**
- Modify: `tests/test_dialog_agent.py`
- Modify: `tests/test_conversation_progress_history_policy.py`
- Modify: other directly affected tests from Task 1 scan

- [x] **Step 1: Add or adjust generator-only dialog-agent tests**

In `tests/test_dialog_agent.py`, update graph tests so a patched generator response is returned directly without patching an evaluator. The core assertion should be:

```python
assert result["final_dialog"] == ["raw<br>fragment"]
```

and the patched evaluator object should be removed from the test.

- [x] **Step 2: Remove evaluator prompt assertions**

Delete tests that only assert `_DIALOG_EVALUATOR_PROMPT` text. Keep tests that assert generator prompt contract or final dialog assembly.

- [x] **Step 3: Remove evaluator payload-history test**

Delete `test_dialog_evaluator_payload_excludes_history_message_and_monologue` from `tests/test_conversation_progress_history_policy.py`. Keep generator payload budget tests and update context-budget summary expectations so evaluator payload counts are gone.

- [x] **Step 4: Run targeted deterministic tests and confirm expected failures before code removal**

Run:

```powershell
venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py tests\test_conversation_progress_history_policy.py -q
```

Expected before production edits: failures from removed evaluator symbols or old graph expectations, proving tests now describe the desired generator-only contract.

## Task 3: Remove Runtime Evaluator Path

**Files:**
- Modify: `src/kazusa_ai_chatbot/nodes/dialog_agent.py`

- [x] **Step 1: Remove evaluator prompt/model/handler**

Delete `_DIALOG_EVALUATOR_PROMPT`, `_dialog_evaluator_llm_config`, and `dialog_evaluator`. Remove `_dialog_evaluator_llm` if no other runtime code uses it.

- [x] **Step 2: Remove evaluator feedback support from generator**

Delete the `# Evaluator Feedback` prompt section and the generator code that appends evaluator messages from previous graph iterations. The generator should receive only its system prompt and current human payload.

- [x] **Step 3: Make dialog graph generator-only**

Change graph construction to:

```python
sub_agent_builder.add_node("generator", dialog_generator)
sub_agent_builder.add_edge(START, "generator")
sub_agent_builder.add_edge("generator", END)
```

Remove `should_stop` and `retry` from the initial substate if no longer consumed.

- [x] **Step 4: Adjust event logging**

Keep existing `record_dialog_quality_event` only as a delivery-shape event if changing the schema would create wider churn. Set retry count to `0` or remove retry semantics from the event payload only if the logging interface supports it.

- [x] **Step 5: Run compile and targeted tests**

Run:

```powershell
venv\Scripts\python.exe -m py_compile src\kazusa_ai_chatbot\nodes\dialog_agent.py
venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py tests\test_conversation_progress_history_policy.py -q
```

Expected: compile passes; tests either pass or identify direct evaluator-removal cleanup still needed.

## Task 4: Remove Route Config And Docs

**Files:**
- Modify: `src/kazusa_ai_chatbot/config.py`
- Modify: `README.md`
- Modify: `docs/HOWTO.md`
- Modify: directly affected config tests from Task 1 scan

- [x] **Step 1: Remove dialog evaluator route constants**

If Task 1 confirms no runtime consumer remains, remove `DIALOG_EVALUATOR_LLM_*` config constants and imports.

- [x] **Step 2: Update setup docs**

Remove `DIALOG_EVALUATOR_LLM` from README route tables and remove `DIALOG_EVALUATOR_LLM_*` variables from HOWTO examples.

- [x] **Step 3: Update config tests**

Remove assertions that require dialog evaluator route configuration. Keep generator route assertions.

- [x] **Step 4: Run config/doc-adjacent tests**

Run:

```powershell
venv\Scripts\python.exe -m pytest tests\test_config.py -q
```

Expected: config tests pass without dialog evaluator route requirements.

## Task 5: Refactor Live LLM Tests

**Files:**
- Modify: `tests/test_dialog_l3_surface_contract_live_llm.py`
- Remove: `tests/test_dialog_evaluator_live_llm_contract.py`
- Modify: other live dialog tests from Task 1 scan that wrap `_dialog_evaluator_llm`

- [x] **Step 1: Remove evaluator endpoint checks from dialog live tests**

Change live dialog fixtures so they only check `DIALOG_GENERATOR_LLM_BASE_URL`.

- [x] **Step 2: Remove evaluator LLM wrapping**

Delete `_dialog_evaluator_llm` monkeypatching and evaluator model metadata from dialog live traces.

- [x] **Step 3: Remove evaluator-only live test file**

Delete `tests/test_dialog_evaluator_live_llm_contract.py` after its useful evidence has been captured in review artifacts.

- [x] **Step 4: Run one target real LLM test**

Run:

```powershell
$env:DIALOG_LIVE_PHASE='dialog_evaluator_removed_target'
$env:DIALOG_LIVE_THINKING='on'
Remove-Item Env:\DIALOG_LIVE_COLLECT_ONLY -ErrorAction SilentlyContinue
venv\Scripts\python.exe -m pytest tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_10_captured_banter_thinking_tail -q -s -m live_llm
```

Expected: one generator-only trace is written and inspected; output has no physical-attention failure or impression-summary tail.

## Task 6: Broader Verification

**Files:**
- No planned production edits.

- [x] **Step 1: Run regular targeted tests**

Run:

```powershell
venv\Scripts\python.exe -m pytest tests\test_dialog_agent.py tests\test_conversation_progress_history_policy.py tests\test_rag_dialog_event_logging.py tests\test_config.py -q
```

Expected: all pass or only unrelated failures are documented.

- [x] **Step 2: Run final real LLM target case with thinking disabled**

Run:

```powershell
$env:DIALOG_LIVE_PHASE='dialog_evaluator_removed_target_thinking_off'
$env:DIALOG_LIVE_THINKING='off'
Remove-Item Env:\DIALOG_LIVE_COLLECT_ONLY -ErrorAction SilentlyContinue
venv\Scripts\python.exe -m pytest tests\test_dialog_l3_surface_contract_live_llm.py::test_live_dialog_l3_10_captured_banter_thinking_tail -q -s -m live_llm
```

Expected: target remains acceptable without evaluator.

## Task 7: Independent Clean Scan

**Files:**
- Create or update: `test_artifacts/analysis/dialog_evaluator_removal_clean_scan_20260616.md`

- [x] **Step 1: Dispatch subagent clean scan**

Ask a subagent to search code, tests, docs, plans, and artifacts for dialog evaluator remnants. Required findings:

```text
- no runtime dialog evaluator stage remains;
- no `_dialog_evaluator_llm` patch remains in active tests;
- no required `DIALOG_EVALUATOR_LLM_*` route remains in config/docs;
- remaining `should_stop` hits are non-dialog or historical;
- remaining dialog evaluator mentions are historical artifacts or this plan/scan.
```

- [x] **Step 2: Run local clean scan**

Run:

```powershell
rg -n -i "dialog_evaluator|DIALOG_EVALUATOR|_DIALOG_EVALUATOR_PROMPT|_dialog_evaluator|Evaluator Feedback|dialog evaluator" src tests docs README.md development_plans --glob '!venv/**' --glob '!__pycache__/**'
```

Expected: no active runtime/test/doc references except intentional historical plan entries, scan artifacts, or archived records.

- [x] **Step 3: Write clean scan artifact**

Write `test_artifacts/analysis/dialog_evaluator_removal_clean_scan_20260616.md` with subagent findings, local scan output summary, remaining intentional mentions, and final cleanup status.

## Acceptance Criteria

- Dialog runtime has no evaluator LLM call, evaluator prompt, evaluator node, or evaluator retry loop.
- Normal setup docs no longer require `DIALOG_EVALUATOR_LLM_*`.
- Active tests do not patch or import `_dialog_evaluator_llm`.
- Deterministic tests pass for dialog graph plumbing and config.
- Target real LLM dialog case passes with thinking enabled and disabled.
- Deep scan and clean scan artifacts exist and classify remaining mentions.
