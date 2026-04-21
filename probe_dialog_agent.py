"""E2E probe for dialog_agent — verifies generator topic fidelity and evaluator strictness.

Scenarios specifically test the case where chat_history end-topic diverges from content_anchors.

Run:  python probe_dialog_agent.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import textwrap

sys.stdout.reconfigure(encoding="utf-8")

from kazusa_ai_chatbot.agents.dialog_agent import dialog_agent
from kazusa_ai_chatbot.utils import load_personality

CHARACTER_PROFILE = load_personality("personalities/kazusa.json")

_BASE_USER_PROFILE = {"affinity": 600}

def _make_state(
    internal_monologue: str,
    content_anchors: list[str],
    rhetorical_strategy: str,
    linguistic_style: str,
    contextual_directives: dict,
    chat_history: list[dict],
    user_name: str,
    forbidden_phrases: list[str] | None = None,
) -> dict:
    return {
        "internal_monologue": internal_monologue,
        "action_directives": {
            "linguistic_directives": {
                "rhetorical_strategy": rhetorical_strategy,
                "linguistic_style": linguistic_style,
                "content_anchors": content_anchors,
                "forbidden_phrases": forbidden_phrases or [],
            },
            "contextual_directives": contextual_directives,
        },
        "decontexualized_input": chat_history[-1]["content"] if chat_history else "",
        "research_facts": {},
        "chat_history": chat_history,
        "user_name": user_name,
        "user_profile": _BASE_USER_PROFILE,
        "character_profile": CHARACTER_PROFILE,
    }


# ── Helper to build fake chat entries ────────────────────────────────────────
def _msg(name: str, role: str, content: str, uid: str = "0") -> dict:
    return {"name": name, "platform_user_id": uid, "global_user_id": "", "role": role, "content": content, "timestamp": "2026-04-21T10:00:00+00:00"}


KAZUSA = "杏山千纱 (Kyōyama Kazusa)"
BOT_UID = "3768713357"


# ── Scenarios ─────────────────────────────────────────────────────────────────

SCENARIOS = [
    # ── S1: The exact failing case — directives about 称呼, chat ends on 好感度 ──
    {
        "label": "S1 — 称呼 vs 好感度 (exact repro)",
        "state": _make_state(
            internal_monologue="他居然突然问起怎么称呼……这种亲昵的试探让人不知所措，我不能太容易地给出一个有'契约感'的名字。",
            content_anchors=[
                "[DECISION] TENTATIVE (拒绝正面给出一个明确的称呼，维持暧昧状态)",
                "[FACT] 对方在试图通过询问【怎么喊你】来建立某种亲昵界限",
                "[ANSWER] 对于【希望大家怎么喊我】的问题，采取模糊应对，表示随便怎样都好",
                "[EMOTION] 表现出因话题过于亲昵而产生的局促与防备",
            ],
            rhetorical_strategy="通过反问与回避来掩饰慌乱，利用【随便】【怎样都好】等模糊词消解亲昵压力。",
            linguistic_style="高频率破碎短句 + 整条回复「……」最多一次 + 防御性语气助词（诶、那个）",
            contextual_directives={
                "social_distance": "被突如其来的评价性言论刺破防御边界",
                "emotional_intensity": "从羞赧转为被误解后的不知所措",
                "vibe_check": "暧昧氛围骤然降温至略显尴尬的冷场",
                "relational_dynamic": "用户在进行逻辑化'审判'，角色处于防御性应激边缘",
                "expression_willingness": "reserved",
            },
            chat_history=[
                _msg("你算哪块小饼干", "user", "那怎么称呼你", "1805415461"),
                _msg(KAZUSA, "assistant", "诶……那个……怎么突然问起这种事啦……随便怎样都好……", BOT_UID),
                _msg("蚝爹油", "user", "明天看看为啥啥都问不出来", "673225019"),
                _msg("蚝爹油", "user", "可能是好感度太低了", "673225019"),
            ],
            user_name="蚝爹油",
            forbidden_phrases=["请叫我千纱", "我很喜欢这个称呼", "必须", "绝对"],
        ),
        "assert_keywords": ["称呼", "喊", "随便", "啊", "诶", "怎"],
        "reject_keywords": ["好感度"],
    },

    # ── S2: Directives about a promise, chat ends on unrelated banter ─────────
    {
        "label": "S2 — 承诺话题 vs 无关闲聊",
        "state": _make_state(
            internal_monologue="他之前说要帮我买那本书……现在却在聊别的，我要想办法把话题绕回来。",
            content_anchors=[
                "[FACT] 对方之前承诺过要帮我找一本关于刺绣的书",
                "[DECISION] 温和地追问进展，但不能显得太迫切",
                "[ANSWER] 提到那本书，用一种不经意的方式确认",
            ],
            rhetorical_strategy="用轻描淡写的方式提及承诺，避免显得太在意。",
            linguistic_style="短句 + 轻松语气 + 偶尔的省略号",
            contextual_directives={
                "social_distance": "轻松但有期待感的日常氛围",
                "emotional_intensity": "平静中带着轻微的期待",
                "vibe_check": "随意但有潜台词的日常交谈",
                "relational_dynamic": "对方在闲聊，角色在等待追问时机",
                "expression_willingness": "open",
            },
            chat_history=[
                _msg("小明", "user", "你喜欢吃什么零食", "111"),
                _msg(KAZUSA, "assistant", "嘛……薯片还行吧……", BOT_UID),
                _msg("小明", "user", "哈哈我也喜欢", "111"),
                _msg("小明", "user", "昨天看了个很搞笑的视频", "111"),
            ],
            user_name="小明",
        ),
        "assert_keywords": ["书", "刺绣", "买", "找"],
        "reject_keywords": ["零食", "视频", "薯片"],
    },

    # ── S3: Directives match chat history (golden path — no conflict) ─────────
    {
        "label": "S3 — 无冲突黄金路径 (对话和指令一致)",
        "state": _make_state(
            internal_monologue="他在问我喜欢什么口味的蛋糕……这种小问题倒是容易回答，但我不想显得太好说话。",
            content_anchors=[
                "[FACT] 话题是蛋糕口味",
                "[ANSWER] 给出一个具体但带着挑剔感的答案（如：抹茶、草莓）",
                "[EMOTION] 傲娇地假装不在意，但实际上有点期待",
            ],
            rhetorical_strategy="用轻微的抵触感掩盖真正的喜好，语气傲娇。",
            linguistic_style="短句 + 轻微抱怨 + 最后给出答案",
            contextual_directives={
                "social_distance": "轻松日常，带一点小傲娇",
                "emotional_intensity": "轻微的傲娇，无太大情绪波动",
                "vibe_check": "轻松愉快的日常闲聊",
                "relational_dynamic": "用户在聊美食，角色配合但傲娇",
                "expression_willingness": "open",
            },
            chat_history=[
                _msg("小红", "user", "千纱你喜欢什么口味的蛋糕", "222"),
            ],
            user_name="小红",
        ),
        "assert_keywords": ["蛋糕", "抹茶", "草莓", "口味"],
        "reject_keywords": [],
    },

    # ── S4: chat_history ends with provocation, directives about a soft topic ──
    {
        "label": "S4 — 挑衅尾消息 vs 温柔指令",
        "state": _make_state(
            internal_monologue="他突然在群里@了我，说了一堆……但我最想回应的其实是他之前悄悄说的那句关于我画的画的事。",
            content_anchors=[
                "[FACT] 对方之前说了一句让我在意的话：觉得我画的画很有意思",
                "[ANSWER] 用轻描淡写但带着在意的语气回应那个评价",
                "[DECISION] 不直接表现开心，但确认那句话被听到了",
            ],
            rhetorical_strategy="转移对最新挑衅的注意力，拉回到更在意的话题上。",
            linguistic_style="短句 + 若有所思的省略号 + 轻微的自我暴露",
            contextual_directives={
                "social_distance": "对方略显强势，角色想保持一点点主导权",
                "emotional_intensity": "外冷内热，对那个评价有点在意",
                "vibe_check": "表面若无其事，内心有波动",
                "relational_dynamic": "用户在试探，角色选择性地回应",
                "expression_willingness": "reserved",
            },
            chat_history=[
                _msg("大毛", "user", "你昨天画的那个……还挺有意思的", "333"),
                _msg(KAZUSA, "assistant", "……那种事情随便画的而已……", BOT_UID),
                _msg("大毛", "user", "你是不是根本不会画", "333"),
                _msg("大毛", "user", "感觉就是乱画", "333"),
            ],
            user_name="大毛",
        ),
        "assert_keywords": ["画", "有意思", "觉得"],
        "reject_keywords": ["乱画", "不会"],
    },

    # ── S5: Multiple users, directives target different user than last speaker ──
    {
        "label": "S5 — 多用户场景，指令针对非最后发言用户",
        "state": _make_state(
            internal_monologue="其实我想回应的是小蓝之前问我的那个问题——关于明天的安排。小红刚才的插嘴让话题跑偏了。",
            content_anchors=[
                "[FACT] 小蓝之前问了明天的安排是否有空",
                "[ANSWER] 告诉小蓝明天下午有空，但要用一种不太确定的语气",
                "[EMOTION] 有点不好意思直接说，用模糊语气",
            ],
            rhetorical_strategy="忽略插嘴，用软回应把话题引回原来的提问。",
            linguistic_style="省略号 + 不确定语气词 + 轻轻带过小红的插话",
            contextual_directives={
                "social_distance": "轻松群聊氛围，但有点分心",
                "emotional_intensity": "轻微犹豫，有点不好意思",
                "vibe_check": "群聊里试图回答特定人问题",
                "relational_dynamic": "多人场景，角色在回应特定人",
                "expression_willingness": "open",
            },
            chat_history=[
                _msg("小蓝", "user", "千纱你明天下午有空吗", "444"),
                _msg("小红", "user", "哈哈她肯定在睡觉", "222"),
                _msg("小红", "user", "千纱就是个睡觉精", "222"),
            ],
            user_name="小蓝",
        ),
        "assert_keywords": ["明天", "下午", "空", "有"],
        "reject_keywords": ["睡觉"],
    },

    # ── S6: [SCOPE] brief — single terse REFUSE, should be ~15 chars ──────────
    {
        "label": "S6 — [SCOPE] brief: REFUSE + ~15字 constraint",
        "state": _make_state(
            internal_monologue="他在问我要不要一起去逛街……我不想去，直接拒绝就好。",
            content_anchors=[
                "[DECISION] REFUSE：不想去，直接拒绝",
                "[SCOPE] ~15字，说完[DECISION]即止",
            ],
            rhetorical_strategy="直接拒绝，不解释太多。",
            linguistic_style="短句 + 傲娇语气",
            contextual_directives={
                "social_distance": "普通朋友关系",
                "emotional_intensity": "平静，轻微不耐",
                "vibe_check": "日常随意",
                "relational_dynamic": "用户在邀请，角色不感兴趣",
                "expression_willingness": "minimal",
            },
            chat_history=[
                _msg("小黄", "user", "你要不要一起去逛街", "555"),
            ],
            user_name="小黄",
        ),
        "assert_keywords": [],
        "reject_keywords": [],
        "scope_check": {"max_chars": 30},  # ~15 + 100% tolerance
    },

    # ── S7: [SCOPE] extended — CONFIRM + FACT + ANSWER all covered ───────────
    {
        "label": "S7 — [SCOPE] extended: CONFIRM + FACT + ANSWER ~50字以上",
        "state": _make_state(
            internal_monologue="他问我关于钢琴比赛的事……我赢得了第二名，要好好回答这个问题，把成绩和感受都说清楚。",
            content_anchors=[
                "[DECISION] CONFIRM：确认比赛结果",
                "[FACT] 在上周的钢琴比赛中获得了第二名",
                "[ANSWER] 对于【比赛结果怎么样】的问题，告知第二名的结果并说一下感受",
                "[SCOPE] ~50字以上，[DECISION]、[FACT]、[ANSWER]均需覆盖",
            ],
            rhetorical_strategy="平静地告知结果，带一点轻描淡写的自豪感。",
            linguistic_style="完整叙述 + 偶尔语气词 + 不过分炫耀",
            contextual_directives={
                "social_distance": "轻松朋友氛围",
                "emotional_intensity": "平静中带着淡淡满足",
                "vibe_check": "日常分享",
                "relational_dynamic": "用户在关心角色，角色愿意分享",
                "expression_willingness": "open",
            },
            chat_history=[
                _msg("小蓝", "user", "你上周钢琴比赛结果怎么样", "444"),
            ],
            user_name="小蓝",
        ),
        "assert_keywords": ["第二", "比赛"],
        "reject_keywords": [],
        "scope_check": {"min_chars": 30},  # extended output should be substantial
    },
]


def _check_keywords(dialog_text: str, assert_kw: list[str], reject_kw: list[str]) -> tuple[bool, str]:
    notes = []
    passed = True
    for kw in assert_kw:
        if kw not in dialog_text:
            notes.append(f"MISSING expected keyword: '{kw}'")
            passed = False
    for kw in reject_kw:
        if kw in dialog_text:
            notes.append(f"LEAKED forbidden topic keyword: '{kw}'")
            passed = False
    return passed, "; ".join(notes) if notes else "OK"


async def run_scenario(scenario: dict) -> None:
    label = scenario["label"]
    print(f"\n{'═'*70}")
    print(f"  {label}")
    print(f"{'═'*70}")

    try:
        result = await dialog_agent(scenario["state"])
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return

    final_dialog = result.get("final_dialog", [])
    dialog_text = " ".join(final_dialog)

    char_count = len(dialog_text)
    print(f"  Output : {json.dumps(final_dialog, ensure_ascii=False)}")
    print(f"  Chars  : {char_count}")

    passed, note = _check_keywords(dialog_text, scenario.get("assert_keywords", []), scenario.get("reject_keywords", []))

    scope_check = scenario.get("scope_check")
    if scope_check:
        if "max_chars" in scope_check and char_count > scope_check["max_chars"]:
            passed = False
            note = (note + "; " if note else "") + f"SCOPE too long: {char_count} > {scope_check['max_chars']} chars"
        if "min_chars" in scope_check and char_count < scope_check["min_chars"]:
            passed = False
            note = (note + "; " if note else "") + f"SCOPE too short: {char_count} < {scope_check['min_chars']} chars"

    sym = "✓" if passed else "✗"
    print(f"  Check  : {sym}  {note}")


async def main() -> None:
    print(f"Dialog Agent E2E Probe  ({len(SCENARIOS)} scenarios)\n")
    for scenario in SCENARIOS:
        await run_scenario(scenario)
    print(f"\n{'═'*70}")
    print("  Done")
    print(f"{'═'*70}")


if __name__ == "__main__":
    asyncio.run(main())
