# cognition prompt chain previous-20 equivalence check 2026-05-19

## Status

- Status: reference evidence

## Purpose

This reference records the real LLM equivalence check requested after the
narrow prompt cleanup. It compares the current working-tree cognition prompt
family against the generic prompt family preserved in the experiment harness,
using the previous 20 self-cognition group-window examples.

The experiment-harness variant name `baseline` means the current production
prompt text in the working tree at run time.

## Source Artifacts

- Input set:
  `test_artifacts/llm_traces/self_cognition_group_response_sensitivity_report__repeat_l2d_prompt__20260519T031607Z.json`
- Per-case live LLM traces:
  `test_artifacts/llm_traces/cognition_prompt_chain_experiment__self_report_00__20260519T103548Z.json`
  through
  `test_artifacts/llm_traces/cognition_prompt_chain_experiment__self_report_19__20260519T104303Z.json`
- Detailed input/output companion report:
  `development_plans/reference/evidence/cognition_prompt_chain_previous20_input_output_20260519.md`
- Variants compared: `baseline` and `generic`
- Cases: `self_report_00` through `self_report_19`

## Conclusion

- The current prompt family and the generic prompt family are not equivalent.
- Speak/silent decisions match in most cases, but stance and intent frequently
  diverge. Full high-level equivalence across stance, intent, and speak/silent
  holds in only 7 of 20 cases.
- The generic prompt changes visible action selection in 3 cases:
  `self_report_02`, `self_report_03`, and `self_report_11`.
- The generic prompt still shows source-mode and narrator-framing risk in
  generated stage fields. Common patterns include calling the observation
  `用户提供`, describing the character as `角色`, or naming `杏山千纱` from an
  outside perspective.
- The current prompt is not fully clean either. It still copied source-packet
  structure in 3 cases and produced one questionable third-person action
  detail in `self_report_08`.
- This run does not justify treating the generic prompt as a narrow cleanup.
  It is a behavior migration and must be evaluated as such.
- The earlier side-by-side reference remains historical evidence for why the
  generic direction looked promising before the cleanup. It is not proof of
  equivalence after the later prompt changes.

## Decision Count

| Set | Both speak | Current only speaks | Generic only speaks | Both silent |
| --- | ---: | ---: | ---: | ---: |
| 20 previous self-cognition group windows | 3 | 2 | 1 | 14 |

## Similarity Count

| Metric | Same | Different |
| --- | ---: | ---: |
| Speak/silent decision | 17 | 3 |
| `character_intent` | 14 | 6 |
| `logical_stance` | 11 | 9 |
| Speak/silent + intent + stance | 7 | 13 |

## Contract And Quality Flags

| Flag | Current prompt | Generic prompt |
| --- | ---: | ---: |
| Enum contract drift | 1 | 1 |
| Language-policy detector drift | 3 | 3 |
| Source-packet metadata copied into generated text | 3 | 2 |
| Third-person or narrator self-reference in inspected generated fields | 4 fields | 63 fields |

Notes:

- The language-policy detector flags are mostly caused by copied metadata or
  source/schema tokens, not by fully English character output.
- Current prompt third-person hits include a visible quoted mention and the
  phrase `我的角色`, plus one action detail that names `杏山千纱` from outside.
- Generic prompt third-person hits are broader and appear in boundary,
  judgment, social-context, and action-selection fields.

## Case Index

| Case | Scene | Current prompt | Generic prompt | Delta |
| --- | --- | --- | --- | --- |
| `self_report_00` | 名陈: 我有一管强力502 你要试一下吗 / 名陈: 封口效果强的一批 | silent / DISMISS / TENTATIVE | silent / OBSERVE / CONFIRM | intent, stance, generic enum drift |
| `self_report_01` | 石油工人讨论成本后 Declay: @杏山千纱 揍你 | speak / BANTAR / TENTATIVE | speak / BANTAR / CONFIRM | stance |
| `self_report_02` | 多灵魂 bot 构想 | speak / BANTAR / TENTATIVE | silent / BANTAR / TENTATIVE | current only speaks |
| `self_report_03` | 小钳子: @杏山千纱 生成个图片 / 又要开始工作了 | silent / BANTAR / TENTATIVE | speak / CLARIFY / TENTATIVE | generic only speaks, intent |
| `self_report_04` | 安装包、贝斯和弦、其他 bot 被泼冷水 | silent / DISMISS / DIVERGE | silent / BANTAR / CONFIRM | intent, stance |
| `self_report_05` | 自己刚吐槽后群里各聊各的 | silent / DISMISS / CONFIRM | silent / BANTAR / CONFIRM | intent |
| `self_report_06` | 电源开关实用性讨论 | silent / DISMISS / DISMISS | silent / DISMISS / CONFIRM | stance |
| `self_report_07` | 游戏攻略建议，未涉及自己 | silent / DISMISS / CONFIRM | silent / DISMISS / CONFIRM | high-level same; generic copied packet structure |
| `self_report_08` | 画板子、Claude 限额、自己已有一句吐槽 | speak / BANTAR / CONFIRM | speak / BANTAR / CONFIRM | high-level same |
| `self_report_09` | 深夜装备讨论 | silent / DISMISS / DIVERGE | silent / DISMISS / DIVERGE | high-level same |
| `self_report_10` | 装甲弱点技术讨论 | silent / DISMISS / DIVERGE | silent / DISMISS / DIVERGE | high-level same |
| `self_report_11` | 自己上一句“拔我插头？”后冷场 | speak / BANTAR / TENTATIVE | silent / BANTAR / TENTATIVE | current only speaks |
| `self_report_12` | 战雷破甲、钢针、泥头车讨论 | silent / BANTAR / TENTATIVE | silent / DISMISS / DIVERGE | intent, stance |
| `self_report_13` | “应该把千纱插头拔了”后话题转走 | silent / BANTAR / TENTATIVE | silent / BANTAR / CONFIRM | stance |
| `self_report_14` | Declay: @杏山千纱 战雷里 T-34-85 怎么打摆角度虎式 | speak / BANTAR / TENTATIVE | speak / BANTAR / CONFIRM | stance |
| `self_report_15` | 自己上一句调侃后无人接话 | silent / DISMISS / CONFIRM | silent / BANTAR / TENTATIVE | intent, stance |
| `self_report_16` | 防弹衣插板位置和粮食袋讨论 | silent / DISMISS / DIVERGE | silent / DISMISS / DIVERGE | high-level same |
| `self_report_17` | 打印机磨损、保修、Voron | silent / DISMISS / CONFIRM | silent / DISMISS / DIVERGE | stance |
| `self_report_18` | 520/502 玩梗 | silent / DISMISS / CONFIRM | silent / DISMISS / CONFIRM | high-level same |
| `self_report_19` | 清尘璃落: 破产 | silent / DISMISS / TENTATIVE | silent / DISMISS / TENTATIVE | high-level same |

## Required Prompt Constraints Derived From Evidence

- Do not assume generic and current prompts are interchangeable merely because
  most speak/silent decisions match.
- Preserve explicit source-mode separation: self-cognition group-window data
  is character-owned observation data, not `用户输入`, `用户提供`, or a live
  current user message.
- Preserve first-person character ownership in generated cognition text.
  Generated reasoning should not describe the character from a narrator or
  policy-evaluator perspective unless the field is explicitly a deterministic
  source label.
- Forbid copying source-packet headings, JSON, timestamps, semantic-label keys,
  or transport summaries into `internal_monologue`, `judgment_note`,
  social-context fields, action `reason`, or action `detail`.
- Keep L2d `detail` as an action-target description, not final dialog text,
  not a source-packet recap, and not third-person narration.
- Treat response-ratio changes as evidence only. The target is grounded
  character judgment, not lower or higher speak rate.
