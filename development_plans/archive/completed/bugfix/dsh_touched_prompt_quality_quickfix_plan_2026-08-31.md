# DSH-Touched Prompt Quality Quickfix

- **Status:** completed
- **Created:** 2026-08-31
- **Baseline:** `4fab6a9af34f75804b4eab5ee42bddf0c28f0b8b`
- **Authorization:** the user explicitly requested review and correction of
  every prompt touched since the pre-DSH baseline.

## Goal

Bring every surviving prompt changed since the baseline back into the local
LLM prompt contract: one bounded semantic question per call, a concise positive
decision procedure, an explained system/human split, and an explicit output
contract that matches deterministic validation.

## Audited Prompt Inventory

The surviving prompt surface contains twelve system prompts and their paired
human payloads:

1. Cognition A1, A2, and G.
2. Cognition P ordinary, pending clarification, DSH interaction, combined
   pending-plus-interaction, and self-cognition variants.
3. Text content planning and terminal visual planning.
4. Final dialog rendering.
5. The DSH Standard-profile terminal contract and its `objective`/`facts`
   admission payload.

The deleted legacy agentic-resolver policy, coding-agent prompts,
complex-task-resolver prompts, RAG2 prompts, and legacy task-specialist prompts
were also classified. They are decommissioned code with no current model call
site and remain deleted.

## Confirmed Findings

1. Exact JSON-only output instructions were removed from all cognition, surface,
   visual, and dialog prompts even though their handlers still require exact
   object shapes.
2. Cognition variants repeat prohibition-heavy historical rules, refer to the
   internal `P` stage from earlier stages, and do not give the weaker local
   model a short ordered decision procedure or explain every used payload lane.
3. Surface and dialog prompts duplicate semantic audits and bug-history-shaped
   exclusions while omitting their exact validated output schemas.
4. The DSH terminal prompt does not explain the human payload's `objective` and
   `facts` fields or the status-specific `submit_resolution` contract. Its tool
   schema also leaves the closed status vocabulary implicit despite strict
   runtime validation.

## Change Surface

- `src/kazusa_ai_chatbot/cognition_core_v3/facade.py`
  - rewrite the eight complete system prompts in concise stage-local form;
  - explain the used human payload lanes;
  - remove internal stage-name leakage from semantic instructions;
  - restore exact JSON-only and free-text-language rules.
- `src/kazusa_ai_chatbot/cognition_shared/surface_stages.py`
  - consolidate duplicated authority rules into one positive planning
    procedure;
  - restore exact content-plan and visual output schemas and bounds.
- `src/kazusa_ai_chatbot/nodes/dialog_agent.py`
  - consolidate duplicate audits into one rendering procedure;
  - explain all human payload fields, preserve context-selected language, and
    restore the exact `final_dialog` JSON contract.
- `sidecars/dsh_resolution/src/profile.ts`
  - explain `objective` and `facts`, give a bounded terminal procedure, and
    expose the runtime's closed status enum in the terminal tool schema.

No handler, validator, stage topology, capability roster, permission rule,
persistence contract, or user-visible feature is changed.

## Source-To-Test Mapping

Test execution is explicitly deferred by the user. The later test owner should
use the existing mapped nodes without adding prompt-literal tests:

| Source | Existing behavioral/contract coverage to run later |
|---|---|
| `cognition_core_v3/facade.py` | `tests/unit/cognition_core_v3/test_handleless_contract.py`; `tests/unit/cognition_core_v3/test_dsh_interaction_contract.py`; `tests/unit/cognition_core_v3/test_stage_recovery.py` |
| `cognition_shared/surface_stages.py` | `tests/unit/nodes/test_persona_supervisor2_l3_surface.py` |
| `nodes/dialog_agent.py` | `tests/unit/nodes/test_dialog_agent.py` |
| `sidecars/dsh_resolution/src/profile.ts` | `tests/test_agentic_resolver_sidecar_process.py` and the sidecar TypeScript contract suite |

## Acceptance

1. Every surviving touched prompt has an explicit semantic task, input-field
   meaning, ordered procedure, output language rule, and exact output contract.
2. No prompt refers to implementation phases, tests, migration history, or an
   earlier/later cognition stage as semantic authority.
3. Prompt text uses role-neutral runtime vocabulary and contains no hardcoded
   character name.
4. DSH terminal status values and request-specific fields match
   `SubmitResolutionV2` exactly.
5. The final diff changes prompt-facing text/schema only and passes whitespace
   review. No tests are run in this task.

## Verification Record

- Prompt review: all twelve surviving system prompts and their human payload
  contracts were inspected. Eight cognition variants, content planning, visual
  planning, final dialog, and the DSH terminal contract now meet the local-LLM
  input/procedure/output standard.
- Decommissioned prompt review: the deleted legacy agentic-resolver policy,
  coding-agent prompts, complex-task-resolver prompts, RAG2 prompts, and legacy
  task-specialist prompts have no surviving call site and remain deleted.
- Schema agreement: cognition prompts point to their typed `output_contract`;
  surface, visual, and dialog prompts state the exact shapes and bounds enforced
  by their validators; DSH exposes the same six statuses enforced by
  `SubmitResolutionV2` and states the status-specific request rules.
- CJK safety: the three edited Chinese Python prompt owners passed syntax-only
  `py_compile` checks after editing.
- Diff/whitespace review: `git diff --check` passed. The production diff is
  limited to the four prompt-facing owners named in this plan.
- Tests: deferred by explicit user instruction.

## Outcome

The current prompt surface is shorter and stage-local, keeps stable rules in
system messages and current data in human payloads, explains every used input
lane, uses positive decision procedures, and restores exact structured-output
contracts. No runtime owner, semantic enum, permission, persistence path,
capability roster, or user-visible feature was added or removed.
