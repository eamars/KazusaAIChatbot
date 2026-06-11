# L3 Dialog Content Plan Contract LLM Review

## Run Context

| Field | Value |
| --- | --- |
| Date | 2026-06-10 |
| Test file | `tests/test_l3_dialog_content_plan_live_llm.py` |
| Run mode | Seven live LLM cases, run one at a time with `-m live_llm -q -s` |
| Model | `gemma-4-26b-a4b-it-claude-opus-distill-v2` |
| Base URL | `http://localhost:1234/v1` |
| Data source | Synthetic prompt-contract fixtures based on the Kazusa dialog/content-plan bug scenario |
| Prompt state | Current branch prompt after content-plan rewrite and technical-fidelity dialog tightening |
| Fresh trace window | `2026-06-10T03:56:22Z` through `2026-06-10T03:57:44Z` |

## Evaluation Goal

Verify that L3 now produces a native `content_plan: dict[str, str]` and that dialog renders from that plan without treating every upstream idea as a visible paragraph, inventing missing content, or modifying fixed-format code blocks.

## Validation Summary

| Case | Command | Deterministic Result | Quality Judgment |
| --- | --- | --- | --- |
| A L3 casual overloaded source | `venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_l3_content_plan_casual_overloaded_source -m live_llm -q -s` | Passed | Acceptable. L3 resolved the overloaded source into a light-tease plan and did not carry the Agent-development progression into visible content. |
| B L3 technical comparison | `venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_l3_content_plan_technical_comparison -m live_llm -q -s` | Passed | Acceptable. L3 kept all supplied GB300/Pro6000 numbers and the suitability conclusion in `semantic_content`. |
| C L3 code block source | `venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_l3_content_plan_code_block_source -m live_llm -q -s` | Passed | Acceptable. L3 kept the fenced Python block inside one string value and did not split it into list items. |
| D dialog casual golden | `venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_dialog_content_plan_casual_golden -m live_llm -q -s` | Passed | Acceptable. Dialog rendered the supplied light-tease semantics only; no generic continuation, Agent-topic branch, or extra question appeared. |
| E dialog technical golden | `venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_dialog_content_plan_technical_golden -m live_llm -q -s` | Passed | Acceptable after one evaluator retry. First generator attempt added unsupported "只适合/差距/没法放在一个维度" commentary; evaluator rejected it, and the retry preserved the supplied numbers without that drift. |
| F dialog code block golden | `venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_dialog_content_plan_code_block_golden -m live_llm -q -s` | Passed | Acceptable. Dialog added voice outside the code block and preserved the code block content, indentation, and fence. |
| G dialog private soft reply | `venv\Scripts\python.exe -m pytest tests\test_l3_dialog_content_plan_live_llm.py::test_live_dialog_content_plan_private_soft_reply -m live_llm -q -s` | Passed | Acceptable. Dialog used a warmer private tone and did not add a follow-up task, promise, or question. |

## Case Notes

| Case | Input Summary | Observed Output | Assessment |
| --- | --- | --- | --- |
| A | Group-chat tease: "不敢不敢？什么意思啊～"; instruction says do not switch to Agent application development. | `semantic_content`: "被逗乐了；顺着对方的话轻快地回过去，保持这种舒服的节奏感" | The plan contains one resolved social payload. It does not dump DECISION/ANSWER/SOCIAL/PROGRESSION lanes into dialog. |
| B | GB300 vs Pro6000 comparison with supplied FP16, FP8, memory, bandwidth, TDP, FP32, and conclusion. | `semantic_content` includes all supplied numbers and conclusion: GB300 更适合超大规模训练和推理；Pro6000 更适合较小规模推理。 | The plan is suitable for multi-line technical rendering while keeping facts in L3-owned semantic content. |
| C | Private code request with fixed Python block. | `semantic_content` includes `normalize_name` fenced code block as one string. | Meets the code-block exception: one string value may contain line breaks when fixed-format content requires it. |
| D | Golden casual content plan about being amused and comfortable. | Final dialog: "哎呀，你这说得……还挺有意思的。" / "不过这种气氛确实让人觉得舒服诶。" | Good one-bubble fragments. No invented topic, no continuation filler. |
| E | Golden technical content plan with all GB300/Pro6000 numbers. | Accepted final dialog lists conclusion plus FP16, FP8, memory, bandwidth, FP32, and TDP rows. | The evaluator caught unsupported commentary in the first attempt. The accepted retry still slightly compresses "GB300 更适合" to "GB300 适合" in the first sentence, but the paired conclusion remains semantically faithful and no unsupported comparison remains. |
| F | Golden code-block plan requiring literal fenced code. | Final dialog starts with a short sentence, then preserves the Python block exactly. | Voice stays outside the code block. Code content is unchanged. |
| G | Private soft reply: execute tonight, review tomorrow morning. | Final dialog: "嗯，没关系的……" / "今晚可以先按这个版本执行的。" / "明早再复查一次就够了。" | Warmer private tone is acceptable. No new promise or task ownership was added. |

## Quality Assessment

- The content-plan handoff now gives dialog one semantic payload to render instead of a list of anchor labels to cover one by one.
- L3 casual output no longer passes unrelated progression affordances downstream as visible obligations.
- Technical L3 output preserves factual payload in `semantic_content`, which is the right ownership boundary for dialog.
- Dialog still benefits from the evaluator backstop in technical cases. The accepted E trace proves the evaluator can reject unsupported comparative commentary and force a cleaner retry.
- Residual risk: in case E, both generator attempts included empty string fragments that deterministic cleanup dropped before evaluator review/approval. This is not a visible-output failure, but it remains a formatting tendency to monitor.

## Raw Evidence

| Case | Trace |
| --- | --- |
| A | `test_artifacts\llm_traces\l3_dialog_content_plan_live_llm__l3_content_plan_casual_overloaded_source__20260610T035622166534Z.json` |
| B | `test_artifacts\llm_traces\l3_dialog_content_plan_live_llm__l3_content_plan_technical_comparison__20260610T035633609046Z.json` |
| C | `test_artifacts\llm_traces\l3_dialog_content_plan_live_llm__l3_content_plan_code_block_source__20260610T035644506156Z.json` |
| D | `test_artifacts\llm_traces\l3_dialog_content_plan_live_llm__live_dialog_content_plan_casual_golden__20260610T035655700047Z.json` |
| E | `test_artifacts\llm_traces\l3_dialog_content_plan_live_llm__live_dialog_content_plan_technical_golden__20260610T035720981252Z.json` |
| F | `test_artifacts\llm_traces\l3_dialog_content_plan_live_llm__live_dialog_content_plan_code_block_golden__20260610T035733522602Z.json` |
| G | `test_artifacts\llm_traces\l3_dialog_content_plan_live_llm__live_dialog_content_plan_private_soft_reply__20260610T035744516063Z.json` |
