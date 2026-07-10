# RAG3 Media Inspection Live LLM Review

## Run Context

| Field | Value |
| --- | --- |
| Date | 2026-07-10 UTC |
| Route | `VISION_DESCRIPTOR_LLM` |
| Model | `gemma-4-26b-a4b-it-claude-opus-distill-v2` |
| Input kind | Synthetic one-pixel PNG, used only to evaluate the new visual node |
| Inspector calls | One per case |
| Cache | Direct inspector tests; no RAG3 session-cache lookup applies |

## Evaluation Goal

Verify that the shared image-only inspector answers an exact visible question
from pixels and refuses an unsupported question about intent.

## Results

| Case | Real output | Quality assessment |
| --- | --- | --- |
| Exact visual question | Returned `answered`, `red`, with a boundary note that the whole image is solid red. | Grounded and concise. The color matches the synthetic red-pixel input. |
| Unsupported non-visual question | Returned `unsupported` and stated that the image contains no information about the photographer's thoughts. | Correct boundary. The deterministic wrapper supplied its standard visible-evidence boundary note because the model returned an empty note list. |

## Validation Results

| Check | Result | Meaning |
| --- | --- | --- |
| Exact case pytest | Passed | The model output parsed and satisfied the image-inspection contract. |
| Unsupported case pytest | Passed | The model exposed the required unsupported boundary. |
| Raw payload containment | Passed | The normalized result exposes no base64 payload, cache reference, hash, or identifier. |

## Raw Evidence

- [Exact visual question raw trace](../../../test_artifacts/llm_traces/media_inspection_live_llm__exact_visual_question.json)
- [Unsupported non-visual question raw trace](../../../test_artifacts/llm_traces/media_inspection_live_llm__unsupported_non_visual_question.json)
- [Existing RAG3 production exact-phrase raw trace](../../../test_artifacts/local_context_resolver/raw/production_exact_phrase.json)

The RAG3 production exact-phrase run selected `conversation_evidence` and
preserved the phrase and speaker in projected evidence. Its source helper
reported an ordinary evidence gap, while the active-node LLM correctly used the
supplied recent conversation evidence. This confirms the intended separation:
source subagents return evidence boundaries and the active node performs the
semantic evidence judgment.

## Debug-Adapter Recent-Image E2E Run

### Run Context

| Field | Value |
| --- | --- |
| Command | `venv\Scripts\python.exe -m pytest tests\test_rag3_media_debug_adapter_e2e_live_llm.py::test_debug_adapter_answers_precise_recent_image_followups -s -m "live_llm and live_db"` |
| Transport | Real debug adapter `/api/chat` → real brain `/chat` |
| Input | Synthetic geometric PNG sent only on turn 1; turns 2-4 carry no attachment |
| Identity mode | Stable debug identity, `no_remember=true` |
| Model route | Production `VISION_DESCRIPTOR_LLM`, RAG3, cognition, and dialog routes |
| Final elapsed time | 4m12s |

### Evaluation Goal

Verify that a real prior-turn image reaches the RAG3 recent-media subagent,
that its precise visual observation reaches cognition without raw payloads,
and that dialog answers three follow-up questions from that evidence.

### Final Observed Output

| Follow-up | Fixture truth | Visible response | Assessment |
| --- | --- | --- | --- |
| Shape left of the white circle | Orange triangle | `橙色三角形。` | Correct. |
| Shape below the white circle | Teal square | `It's cyan... or blue-green. Just below the white circle.` | Correct visual relation and acceptable teal/blue-green wording. |
| Red-dot count in upper-left box | 3 | `左上角方框里是 3 个红点。` | Correct exact count. |

### Root Cause And Remediation

The first E2E attempt exposed a debug-adapter request-shape mismatch; the test
was corrected to use the adapter's actual public contract. The next attempt
showed that RAG3 selected local context but sent a media task without a
selector. A later run showed the planner selecting `current_turn_media` when
only a recent image existed. The resolver now deterministically binds that
condition to `recent_media_1`. Finally, the inspected visual answer is promoted
to the existing prompt-safe RAG answer field and visual observations are
projected to action selection without raw cache or image fields.

### Validation And Residual Risk

| Check | Result | Meaning |
| --- | --- | --- |
| Focused dispatch/media/action-selection tests | Passed | Selector fallback, answer projection, and raw-media exclusion are covered deterministically. |
| Final real E2E pytest | Passed | Both local services started and all four debug-adapter turns completed. |
| Qualitative visual grounding | Accepted | All three responses match deterministic fixture truth. |
| Raw-media telemetry exposure | Not observed | The E2E assertion confirms the base64 image is absent from bounded cognition telemetry. |
| Latency | Residual risk | Four-turn run took 4m12s; this is unsuitable as a normal chat latency target and requires separate performance work. |

`no_remember=true` suppresses memory intent, but service startup and the live
turn path still used the configured real database for identity and operational
state. Treat this test as a live-DB E2E test, not as an isolated database-free
dry run.

### Raw Evidence

- [Final debug-adapter E2E trace](../../../test_artifacts/llm_traces/rag3_media_debug_adapter_e2e_live_llm__recent_image_precision_followups__20260710T021058301645Z.json)
- [Direct visual-inspector diagnostic](../../../test_artifacts/llm_traces/rag3_media_inspection_diagnostic__precise_scene_below_white_circle.json)
- [RAG3 media-projection diagnostic after selector correction](../../../test_artifacts/llm_traces/rag3_media_projection_diagnostic__recent_image_below_white_circle_after_selector_fix.json)

## Completion Live-Case Review

| Case | Result | Quality assessment |
| --- | --- | --- |
| RAG3 current-image exact | `answered`: `red` | Correct visual fact, with the current-media alias and no payload leakage. |
| RAG3 identity boundary | `unsupported` | Correctly says the solid-red image cannot establish the photographer's identity and explains the missing visual anchors. |
| RAG3 cache miss | `unavailable`, zero inspector calls | Correct bounded evidence gap; no visual call was attempted. |
| Preserved conversation source | Mika and the exact phrase remain in `conversation_evidence` | Grounded source evidence survived the subagent conversion. The evidence-only resolver leaves visible wording to downstream cognition/dialog. |
| Preserved scoped memory source | Current-user jasmine-tea preference remains scoped `memory_evidence` | Correct current-user evidence projection without identity or transport leakage. |
| Complex external image | `answered`: `Blue and yellow` | The production subagent fetched, validated, decoded, and visually inspected the public Python-logo PNG correctly. |
| Complex private URL | `failed`, zero inspector calls | Loopback target was rejected before fetching or visual inspection. |

All cases were executed one pytest case at a time. The visual-answer and
boundary outputs are accepted. The preserved RAG3 source cases are also
accepted as evidence-only behavior: their projected evidence is correct while
the existing resolver keeps `rag_result.answer` empty for downstream cognition
and dialog to render.

### Completion Raw Evidence

- [RAG3 current-image exact](../../../test_artifacts/llm_traces/rag3_media_inspection_live_llm__current_image_exact_question.json)
- [RAG3 identity boundary](../../../test_artifacts/llm_traces/rag3_media_inspection_live_llm__current_image_identity_uncertainty.json)
- [RAG3 cache-miss boundary](../../../test_artifacts/llm_traces/rag3_media_inspection_live_llm__current_image_cache_miss_boundary.json)
- [Preserved conversation source](../../../test_artifacts/local_context_resolver/raw/production_exact_phrase.json)
- [Preserved scoped-memory source](../../../test_artifacts/local_context_resolver/raw/production_scoped_memory.json)
- [Complex external-image exact](../../../test_artifacts/llm_traces/complex_task_resolver_live_llm__external_image_exact_question.json)
- [Complex private-URL refusal](../../../test_artifacts/llm_traces/complex_task_resolver_live_llm__external_image_fetch_refusal.json)
