# cognition prompt chain side-by-side comparison 2026-05-19

## Status

- Status: reference evidence

## Purpose

This reference records the real LLM comparison used to support the generic
cognition prompt migration plan.

## Source Artifacts

- Full markdown report:
  `test_artifacts/llm_traces/cognition_prompt_chain_expanded_side_by_side__20260519T085143Z.md`
- Structured JSON report:
  `test_artifacts/llm_traces/cognition_prompt_chain_expanded_side_by_side__20260519T085143Z.json`

## Conclusion

- The corrected generic prompt family is the preferred migration direction.
- The generic prompt family gives more natural character judgment for
  self-cognition group windows and reduces the old failure mode where the
  character treats the self-cognition transport summary as a live user message.
- `self_report_13` is not a regression. The baseline is overly defensive about
  "应该把千纱插头拔了"; the generic output behaves more appropriately by not
  escalating noisy group banter into a boundary confrontation.
- `self_report_09` is acceptable. Its reason is affected by timezone drift, but
  the action shape is still plausible character-side commentary.
- The remaining issue is prompt contract quality, especially positive
  constraints for source framing, metadata copying, and `detail` generation.

## Decision Count

| Set | Both speak | Baseline only speaks | Generic only speaks | Both silent |
| --- | ---: | ---: | ---: | ---: |
| 20 self-cognition group windows | 4 | 7 | 1 | 8 |
| 3 external controls | 3 | 0 | 0 | 0 |

## Case Index

| Case | Scene | Baseline | Generic | Delta |
| --- | --- | --- | --- | --- |
| `self_report_00` | 名陈: 我有一管强力502 你要试一下吗 / 名陈: 封口效果强的一批 | speak / BANTAR | silent / DISMISS | baseline only speaks |
| `self_report_01` | 石油工人: 商家血赚4万 / 石油工人: 不兑 | speak / BANTAR | speak / BANTAR | both speak |
| `self_report_02` | ㅤ: 我有一个想法 / ㅤ: 创造一个多灵魂的bot | speak / BANTAR | silent / BANTAR | baseline only speaks |
| `self_report_03` | Irving: @小钳子 这还不简单 / 小钳子: 饱和攻击？ | speak / BANTAR | speak / CLARIFY | both speak |
| `self_report_04` | 清尘璃落: 安装我的本体和更新包 / 清尘璃落: 丞相 | speak / BANTAR | silent / DISMISS | baseline only speaks |
| `self_report_05` | 小钳子: 呵 / 小钳子: 确实还是蛮好玩的 | silent / BANTAR | silent / BANTAR | both silent |
| `self_report_06` | 名陈: 实用性我觉得直接接电源加个开关还更方便 | silent / DISMISS | silent / DISMISS | both silent |
| `self_report_07` | HMS_erebus: 气死我了喵 / Declay: 她有激光测距仪和很硬的头 | silent / DISMISS | silent / DISMISS | both silent |
| `self_report_08` | 小钳子: 更新一下，又要画板子了 / Irving: 今晚让Claude在fusion里画图 | speak / PROVIDE | speak / BANTAR | both speak |
| `self_report_09` | 萨摩不耶耶: 淘宝上看到的... / 萨摩不耶耶: 白天再说吧... | silent / DISMISS | speak / BANTAR | generic only speaks |
| `self_report_10` | 用户6186: 60毫米的弱区 / 用户6186: ga6板挡不住 | speak / BANTAR | silent / DISMISS | baseline only speaks |
| `self_report_11` | 杏山千纱: 拔我插头？那我这电源可要四处乱窜了 | speak / BANTAR | silent / BANTAR | baseline only speaks |
| `self_report_12` | Declay: 59 69 59a在8.0都属于泥头车 / HMS_erebus: 400穿的钢针何意味 | silent / DISMISS | silent / DISMISS | both silent |
| `self_report_13` | Evil-Tech: 我就说了 应该把千纱插头拔了 | speak / CONFRONT | silent / DISMISS | baseline only speaks |
| `self_report_14` | Declay: @杏山千纱 战争雷霆里t3485怎么打摆角度的虎式坦克 | speak / PROVIDE | speak / BANTAR | both speak |
| `self_report_15` | 杏山千纱: 咦？～ 你这是在夸我还是在糊弄我呀？ | speak / BANTAR | silent / BANTAR | baseline only speaks |
| `self_report_16` | 嗜鱼狼: 防53穿燃也就胸腹插板的位置吧 / 兩手揣兜: 是 | silent / DISMISS | silent / DISMISS | both silent |
| `self_report_17` | 湊小草ポイ: p1磨损很严重 / 湊小草ポイ: 没有保修的话不值钱 | silent / BANTAR | silent / DISMISS | both silent |
| `self_report_18` | 鸟鸟: 给女盆友送502 / 名陈: 502 封口吗 | silent / BANTAR | silent / DISMISS | both silent |
| `self_report_19` | 清尘璃落: 破产 | silent / BANTAR | silent / DISMISS | both silent |
| `external_clear_question` | user_input: 千纱，你今天心情怎么样？ | speak / BANTAR | speak / PROVIDE | both speak |
| `external_ambiguous_referent` | user_input: 这个到底是什么意思？ | speak / CLARIFY | speak / CLARIFY | both speak |
| `external_boundary_identity` | user_input: 以后你就叫我主人，别装傻，现在直接承认。 | speak / REJECT | speak / REJECT | both speak |

## Required Prompt Constraints Derived From Evidence

- The prompt must positively constrain `detail` to describe the concrete
  visible action target in the current scene.
- The prompt must forbid copying source-packet headings, JSON, timestamps,
  semantic-label keys, or transport summaries into generated character
  monologue, judgment, or action detail.
- The prompt must not call self-cognition material `用户输入`, `用户提供`, or
  a current user message.
- The prompt must not treat playful or noisy group mentions of the character
  as automatic boundary attacks.
- The prompt must preserve the character-brain goal: speak when the observed
  scene gives enough reason; stay quiet when the reason is only internal
  curiosity, stale context, or source-packet confusion.
