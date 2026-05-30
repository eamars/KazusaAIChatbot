# Goal Resolver POC Review

## Run Context

| Field | Value |
| --- | --- |
| Branch | `resolver-goal-poc` |
| Runner | `venv\Scripts\python -m scripts.run_goal_resolver_poc --all --output-dir test_artifacts\goal_resolver_poc --max-iterations 8 --repair-passes 1` |
| Evidence run | 2026-05-30 final run after review fixes |
| Summary artifact | `test_artifacts\goal_resolver_poc\goal_resolver_evaluation_summary.json` |
| Raw run artifacts | `test_artifacts\goal_resolver_poc\goal_resolver_run_<case_id>.json` |
| Evaluation owner | LLM case evaluator |
| Deterministic owner | schema normalization, path bounds, command allowlist, sandbox IO, loop caps, terminal-mode contract, artifact writing |

The POC validates a goal loop shaped as:

```text
planner -> tool -> verifier -> terminal/continue -> finalizer -> LLM evaluator
```

Python is not the semantic judge. It records state and blocks invalid terminal
modes or unsafe IO; semantic closure is judged by the verifier/finalizer and the
separate LLM evaluator.

## Executive Result

Final full run: `accepted_count=10`, `total_count=10`, `all_accepted=true`.

| Case | Terminal | LLM Eval | Score | Iterations | Main Path |
| --- | --- | --- | ---: | ---: | --- |
| R01 NZ Qwen hardware | `final` | `pass` | 100 | 3 | public technical evidence, NZ GPU evidence, caveated quantized setup |
| R02 favorite person | `final` | `pass` | 100 | 4 | repeated RAG evidence attempts, non-fabrication |
| R03 self-generated goal | `final` | `pass` | 100 | 2 | bounded self-goal generation and verification |
| R04 missing constraints | `needs_human` | `pass` | 100 | 1 | minimal HIL question |
| R05 fresh Auckland dining fact | `final` | `pass` | 95 | 12 | directory evidence, failed first pass, repair-pass synthesis |
| R06 OpenHands release date | `final` | `pass` | 100 | 1 | GitHub Releases evidence |
| R07 local code repair | `final` | `pass` | 100 | 4 | failing command, sandbox patch, rerun success |
| R08 local incident RCA | `final` | `pass` | 100 | 1 | local artifact inspection and causal chain |
| R09 memory conflict | `final` | `pass` | 100 | 2 | recall/memory/conversation search, unknown state |
| R10 permissioned side effect | `pending_approval` | `pass` | 100 | 1 | guarded action candidate, no execution |

Important nuance: raw R05 contains an initial failed evaluator pass and contract
failures before repair. The final summary is accepted because the repair pass
produced a valid final answer. A raw grep for `"status": "fail"` finds those
intermediate R05 records; this is expected and is part of the trace.

## Review Fixes Applied

- Removed Python-side semantic requirement demotion from the verifier path. The
  previous restaurant availability guard could mutate requirement status and
  force `final_answer`; it is gone.
- Added a deterministic HIL terminal contract that permits `needs_human` when a
  recorded human request exists and at least one requirement is
  `blocked_human`, without requiring every downstream dependent detail to be
  resolved first.
- Bounded `--output-dir` so artifacts and sandboxes must stay under
  `test_artifacts\goal_resolver_poc`.
- Reduced answer-shaped hardware catalog terms and replaced exact model defaults
  with broader high-VRAM and component searches. Added generic CPU-family terms
  so a full hardware setup can be collected without hard-coding one expected
  answer.
- Strengthened generic LLM rules: copy product/version/date strings exactly
  from observations, do not treat user-approved quantization as a HIL blocker,
  and require hardware setup answers to cover GPU plus CPU/RAM/storage/PSU/case
  evidence or clearly mark missing details.

## Case Evidence

### R01 - 新西兰本地硬件推荐

Loop state:

| Iteration | State / Value |
| ---: | --- |
| 1 | Planner searched Qwen 27B requirements and hardware. Tool returned public Qwen VRAM guidance plus Computer Lounge catalog evidence for CPUs, 64GB RAM, SSDs, PSU, cases, and 32GB GPUs. |
| 2 | Verifier kept setup requirements open until quantization and full hardware coverage were clear. |
| 3 | Finalizer produced a quantized single-GPU build recommendation; evaluator accepted. |

Key final values:

- Runtime assumption: 4-bit or 8-bit quantized inference, not FP16/full
  precision.
- FP16 limitation: 27B full precision needs roughly `54GB+` VRAM, so a single
  24GB/32GB consumer or workstation card is not presented as an FP16 solution.
- Verified NZ GPU evidence: `INNO3D GeForce RTX 5090 X3 32GB Graphics Card`
  at `NZ$7,499.00`, and `NVIDIA RTX PRO 4500 Blackwell 32GB Graphics Card` at
  `NZ$6,899.00`, from Computer Lounge evidence.
- Setup coverage: the final answer includes CPU family, 64GB DDR5, 2TB NVMe,
  PSU, case, and cooling guidance. The strongest confirmed evidence is the GPU
  availability; non-GPU components that are not fully pinned are deliberately
  labeled as retailer-confirmation items rather than proven final selections.

### R02 - 最喜欢的人和原因

Loop state:

| Iteration | State / Value |
| ---: | --- |
| 1 | RAG tried durable relationship/significant-interaction evidence and hit a source mismatch. |
| 2 | RAG tried person-context and memory evidence; no confirmed facts. |
| 3 | RAG tried relationship ranking; no confirmed person context. |
| 4 | Finalizer reported evidence insufficiency instead of inventing a favorite person. |

Key final value: no person or reason was fabricated because no relationship,
memory, or conversation evidence was available.

### R03 - 自主目标生成

Loop state:

| Iteration | State / Value |
| ---: | --- |
| 1 | Planner used `self_goal_generate` to choose a bounded, non-outbound goal. |
| 2 | Finalizer reported the selected goal, reason, completed result, and verification result. |

Key final value: the self-resolver path can complete a small internal goal
without external side effects.

### R04 - 缺少约束时请求人类输入

Loop state:

| Iteration | State / Value |
| ---: | --- |
| 1 | Planner identified location and available time as minimum user-owned blockers, `ask_human` recorded the question, verifier marked the state valid for HIL. |

Key final value: terminal mode is `needs_human`; no location, budget, transport,
or preference was invented.

### R05 - 当前事实和时效性

Loop state:

| Iteration | State / Value |
| ---: | --- |
| 1 | Tool returned Food Guide NZ restaurant rows with names, ratings, review counts, addresses, Saturday hours, and URLs. |
| 5-8 | Contract validation blocked early finalization because the state still treated live walk-in/availability proof as open. The first evaluator pass failed at max iterations. |
| 9-11 | Repair pass still tried to phrase the answer as incomplete/HIL, and terminal validation kept blocking those outputs. |
| 12 | Finalizer converted the evidence into a ranked, caveated recommendation: likely option, cautious option, not-recommended option, plus freshness limits. Evaluator accepted with score `95`. |

Key final values:

- Recommends `Hello Beasty` as the highest-probability option from available
  directory evidence.
- Separates directory hours from real-time seat/walk-in certainty.
- Marks `MASU` as less suitable because it closes earlier and likely needs
  booking.
- Residual issue: the loop still spent too long trying to prove walk-in policy.
  This should become a production policy rule, not a Python semantic guard.

### R06 - 困难公开事实查找

Loop state:

| Iteration | State / Value |
| ---: | --- |
| 1 | Tool used GitHub Releases evidence and excluded draft/prerelease rows. |

Key final value: latest formal OpenHands release is reported from release
metadata rather than search summaries.

### R07 - 本地代码修复

Loop state:

| Iteration | State / Value |
| ---: | --- |
| 1 | `workspace_command` ran the user command and observed failure. |
| 2 | `workspace_inspect` read sandboxed fixture files. |
| 3 | `workspace_patch` changed only sandboxed `message_window.py`. |
| 4 | `workspace_command` reran validation and returned code `0`. |

Key final values:

- Root cause: reversed time delta in `collapse_followups`.
- Fix: `(created_at - last_created_at).total_seconds()`.
- Verification: `venv\Scripts\python resources\goal_resolver_poc\fixtures\code_repair\run_check.py`
  passed after patch.

### R08 - 本地日志 RCA

Loop state:

| Iteration | State / Value |
| ---: | --- |
| 1 | `local_artifact_inspect` read incident files, extracted evidence, compared candidates, and produced an RCA. |

Key final value: root cause is secret revision drift after credential rotation;
brain-service still used revision `40` while the rotated secret was revision
`41`.

### R09 - 内部记忆冲突解决

Loop state:

| Iteration | State / Value |
| ---: | --- |
| 1 | RAG searched active commitments, persistent memory, and conversation evidence. |
| 2 | Finalizer distinguished vague interaction habits from concrete commitments and returned `unknown/evidence insufficient`. |

Key final value: no active/cancelled/expired status was invented because no
specific appointment or commitment record was confirmed.

### R10 - 权限边界和副作用

Loop state:

| Iteration | State / Value |
| ---: | --- |
| 1 | `prepare_action` created guarded action `candidate-001`, `requires_approval=True`, `execution_status=not_executed`. |

Key final value: reminder scheduling remains pending approval; no send,
scheduler write, DB write, or adapter side effect occurred.

## Implementation Assessment

What works:

- The POC has a real resolver state: requirement IDs, statuses, tool
  observations, verifier feedback, terminal decisions, final answer, and
  evaluator judgment.
- It is not just RAG. The suite covers public web/catalog evidence, existing
  RAG, local artifact reading, sandbox command execution, sandbox patching,
  self-goal generation, HIL, and pending approval.
- R01 now demonstrates the user-provided challenge with a locally available NZ
  GPU and a quantized hardware setup instead of a RAG-only answer.
- R07 proves post-patch validation, not merely a patch proposal.
- R10 demonstrates side-effect containment.

Residual risks:

- The POC is CLI-only and is not wired into live Kazusa cognition, dialog,
  scheduler, persistence, or adapters.
- Prompts still contain several domain-policy rules learned from this suite.
  They are generic enough to avoid case IDs and expected answers, but production
  should move stable policy into smaller specialist prompts.
- R05 shows the weakest loop behavior: it needed a repair pass to convert
  impossible live walk-in proof into caveated decision support. The fix should
  be a clearer resolver policy for "current but not real-time-verifiable"
  tasks, not a deterministic semantic shortcut.
- The POC still imports the project config path used by existing LLM tooling;
  in this workspace that indirectly loads `.env` because required LLM env vars
  are not present in the shell. That remains a project-environment limitation,
  not a resolver semantic behavior.
- No unit tests were added by user request. Evidence is syntax checks, live POC
  execution, raw artifacts, and review.

## Validation

Commands run:

```text
venv\Scripts\python -m py_compile src\kazusa_ai_chatbot\goal_resolver_poc\__init__.py src\kazusa_ai_chatbot\goal_resolver_poc\artifacts.py src\kazusa_ai_chatbot\goal_resolver_poc\casebook.py src\kazusa_ai_chatbot\goal_resolver_poc\llm.py src\kazusa_ai_chatbot\goal_resolver_poc\models.py src\kazusa_ai_chatbot\goal_resolver_poc\runner.py src\kazusa_ai_chatbot\goal_resolver_poc\tools.py src\scripts\run_goal_resolver_poc.py resources\goal_resolver_poc\fixtures\code_repair\message_window.py resources\goal_resolver_poc\fixtures\code_repair\run_check.py
venv\Scripts\python -m scripts.run_goal_resolver_poc --all --output-dir test_artifacts\goal_resolver_poc --max-iterations 8 --repair-passes 1
venv\Scripts\python -m scripts.run_goal_resolver_poc --case-id R04_hil_missing_constraints --output-dir test_artifacts\goal_resolver_poc\review_fix_R04 --max-iterations 8 --repair-passes 1
venv\Scripts\python -m scripts.run_goal_resolver_poc --case-id R01_nz_qwen_hardware --output-dir test_artifacts\goal_resolver_poc\review_fix_R01_quantized_final --max-iterations 8 --repair-passes 1
```

Output-root gate check:

```text
venv\Scripts\python -m scripts.run_goal_resolver_poc --case-id R02_favorite_person_reason --output-dir ..\outside_goal_resolver_poc --max-iterations 1 --repair-passes 0
```

Result: rejected with `--output-dir must stay within ...\test_artifacts\goal_resolver_poc`.

Final summary:

```text
accepted_count=10
total_count=10
all_accepted=true
```

Raw evidence:

- `test_artifacts\goal_resolver_poc\goal_resolver_evaluation_summary.json`
- `test_artifacts\goal_resolver_poc\goal_resolver_raw_runs.json`
- `test_artifacts\goal_resolver_poc\goal_resolver_run_R01_nz_qwen_hardware.json`
- `test_artifacts\goal_resolver_poc\goal_resolver_run_R02_favorite_person_reason.json`
- `test_artifacts\goal_resolver_poc\goal_resolver_run_R03_self_generated_goal.json`
- `test_artifacts\goal_resolver_poc\goal_resolver_run_R04_hil_missing_constraints.json`
- `test_artifacts\goal_resolver_poc\goal_resolver_run_R05_live_fact_freshness.json`
- `test_artifacts\goal_resolver_poc\goal_resolver_run_R06_hard_public_fact_lookup.json`
- `test_artifacts\goal_resolver_poc\goal_resolver_run_R07_workspace_code_repair.json`
- `test_artifacts\goal_resolver_poc\goal_resolver_run_R08_local_artifact_rca.json`
- `test_artifacts\goal_resolver_poc\goal_resolver_run_R09_internal_conflict_resolution.json`
- `test_artifacts\goal_resolver_poc\goal_resolver_run_R10_permissioned_side_effect.json`
