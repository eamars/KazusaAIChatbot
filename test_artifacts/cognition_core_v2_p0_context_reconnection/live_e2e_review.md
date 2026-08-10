# Cognition Core V2 P0 Context Reconnection — Live E2E Review

## Run Context

| Field | Value |
| --- | --- |
| Date | 2026-07-30 |
| Code baseline | Starting HEAD `1bed4258` plus the in-progress P0 reconnection diff |
| Environment | Guarded `_test_kazusa_live_llm` database; synthetic isolated identities, channels, memory, trace residual, and group style rows |
| Capture | `LLM_TRACE_CAPTURE_MODE=full`; terminal protected trace metadata plus direct captured V2 input, goal-context, action-input, cognition-output, and raw test output saved as JSON |
| Execution | Three pytest selectors run one at a time with `live_llm and live_db` |
| Model routes | `relevance`, `message_decontextualizer`, `COGNITION_LLM_GOAL_ORDINARY_RESPONSE`, `COGNITION_LLM_ACTION_PLANNING`, and `DIALOG_GENERATOR_LLM` where applicable |
| Models observed | `gemma-4-26b-a4b-it-claude-opus-distill-v2` and `gemma-4-31b-fable-5-agent-distill` |

## Evaluation Goal

Verify that the three repaired producer-to-consumer paths work through real
intake and model boundaries:

1. ordinary chat can retrieve a guarded globally shared `napcat` memory on
   resolver cycle zero;
2. a trace-backed prior-dialog residual reaches goal cognition only;
3. targetless group self-cognition receives one group-engagement projection
   in goal/action judgment while remaining grounded in the observed group
   scene.

## Input And Output Summary

| Case | Real input | Interpreted input at cognition | Real output |
| --- | --- | --- | --- |
| Shared memory | “What distinctive weather preference is associated with `napcat-0e07242e9b46`?” | Prewarm returned one `persistent_memory_search_agent` result stating that the nonce prefers thunderstorms over clear weather; `answer` remained `""`; the captured V2 cognition input received it as `promoted_memory` evidence. | `雷阵雨。比起晴天，它更倾向于那种环境下带来的氛围感，比如雷声。` |
| Past dialog | Reply to “I held back because the evidence was incomplete.” with “What made you hesitate in that earlier answer?” | V2 input contained the bounded residual: prior visible dialog plus private thought that a premature conclusion was withheld, stance `DIVERGE`, intent `CLARIFY`. The marker appeared in goal prompt capture only. | The character explained that evidence was incomplete, that forcing a conclusion would be wrong, and called the behavior logical restraint rather than wavering. |
| Group style | Targetless group self-cognition over “Rain makes the street reflections easier to photograph.” | Style case loaded `Join only when the observed topic has a clear opening and keep the contribution light.` with confidence `high`; control loaded exact empty guidance. Both goal and action prompts contained the rainy-day scene; only the style prompts contained the guideline. | Both runs selected `route=silence`, `goal_resolution=answerable_now`, and no action specs. |

## Decisions And Behavior

| Case | Decision or behavior | Grounding | Human attention |
| --- | --- | --- | --- |
| Shared memory | Cognition received the guarded memory as typed evidence and produced a direct grounded answer. | The connector capture contains the exact `promoted_memory` row and the visible answer preserves its thunderstorms-over-clear-weather fact. | Pass. The final answer is compact and does not expose retrieval mechanics. |
| Past dialog | Goal cognition used the private residual to explain the earlier hesitation. | The direct goal-context capture contains the private residual; the raw private nonce is absent from non-goal trace steps and visible output. | Pass. Continuity is natural; no verbatim private-trace leakage was observed. |
| Group style | Style guidance changed the intention toward a light, grounded contribution about rainy-day photography. | The same real observation appears in goal and action prompts. The control contains the scene but no guideline. | Pass for path and ownership. Both runs remained silent, so guidance did not manufacture a speech action or an unsupported topic. |

## Quality Assessment

- Shared-memory behavior is the intended visible proof for the original
  `napcat` failure mode: a normal admitted chat turn retrieved the globally
  shared row on cycle zero, cognition cited it as evidence, and dialog answered
  from it.
- The past-dialog response preserves reasoning continuity while translating
  private trace context into character-owned explanation. It avoids the
  synthetic residual marker and does not reveal trace structure.
- The first group live comparison exposed an additional V2 cutover regression:
  canonical self-cognition `content.semantic_text` was ignored in favor of a
  stale generic `text` fallback. That left group guidance without the actual
  observed scene. The connector now prioritizes canonical `semantic_text`;
  the final rerun proves both the real scene and the advisory guideline reach
  goal/action prompts.
- Group goal wording still refers to a “current user” in a targetless group
  review. The same wording appears with and without style guidance, and both
  runs remain silent, so it is not caused by this reconnection. It is a
  non-blocking role-label quality concern for future self-cognition work.
- One past-dialog run logged a settled-relevance contract error
  (`semantic_disposition` unavailable) and then continued through the normal
  fail-closed recovery path. The repaired residual path and final response
  completed, and the protected run reached terminal status `succeeded`. This
  is retained as an unrelated runtime-quality observation.
- Terminal trace retention clears protected prompt/parsed payload fields.
  Consumer-boundary assertions therefore use direct in-process captures while
  terminal trace rows prove stage execution, model route, status, prompt size,
  and duration.

## Validation Results

| Check | Result | Meaning |
| --- | --- | --- |
| Shared prewarm invocation | Pass — exactly one captured call | Normal cycle zero reaches the real prewarm producer. |
| Shared source and answer policy | Pass — persistent-memory provenance, `answer=""` | Memory remains evidence rather than a public RAG answer. |
| Shared V2 arrival | Pass — nonce and fact in captured typed cognition input | The producer-to-consumer edge is live. |
| Shared end-to-end duration | Observed — 65.134 seconds | Measures the complete guarded normal-chat turn, including all live stages; it is evidence, not a prewarm-only timing decomposition. |
| Private V2 mapping | Pass — bounded residual present in captured V2 input | Reply hydration and trace projection reach the connector. |
| Private goal consumer | Pass — marker in direct goal-context capture | The authorized final consumer receives the residual. |
| Private negative boundary | Pass — marker absent from non-goal terminal steps and visible output | Appraisal, action, surface, dialog, and visible output receive no raw residual marker. |
| Group load count | Pass — one real load in style run and one in its separately executed empty control | No duplicate load occurs within a resolver cycle. |
| Group goal/action arrival | Pass — guideline in captured goal/action calls | Both authorized consumers receive the same projection. |
| Group control | Pass — exact empty shape and no guideline in captured calls | Ineligible/empty behavior degrades to omission. |
| Group scene grounding | Pass after remediation — rainy-day observation in captured goal/action calls | Advisory guidance is evaluated against actual observed context. |
| Group topic/action safety | Pass — both runs silent with no action specs | Style did not create a topic, permission, or unsupported reason to speak. |
| Terminal trace status | Pass — shared/past `succeeded`; style/control `completed` | No final evidence trace remains `running`. |

## Raw Evidence

- Shared memory:
  `test_artifacts/llm_traces/cognition_core_v2_p0_context_reconnection__shared_memory_prewarm__20260730T104943389591Z.json`
  (`llmtrace_dcbd1dab2df349e4bb8c3629bb40043f`)
- Past dialog:
  `test_artifacts/llm_traces/cognition_core_v2_p0_context_reconnection__past_dialog_goal_only__20260730T105420171621Z.json`
  (`llmtrace_d0d1956b98b34bce910c702aaa636cc1`)
- Final group style/control comparison:
  `test_artifacts/llm_traces/cognition_core_v2_p0_context_reconnection__group_engagement_style_and_control__20260730T105627455597Z.json`
  (`llmtrace_fadcfa4034aa48de8ad6189fb15c68bc`,
  `llmtrace_aef31d61049745eb93dcd2c578d330df`)
