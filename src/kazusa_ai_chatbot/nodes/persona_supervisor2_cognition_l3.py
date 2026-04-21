"""L3 — Contextual, Linguistic, and Visual agents + L4 Collector.

Contains the MBTI expression-willingness helper and L3/L4 LLM calls.
"""
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import CognitionState
from kazusa_ai_chatbot.utils import parse_llm_json_output, build_affinity_block, get_llm

from langchain_core.messages import HumanMessage, SystemMessage

import logging
import json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: MBTI expression willingness (used by L3 contextual agent)
# ---------------------------------------------------------------------------

def get_mbti_expression_willingness(mbti: str) -> str:
    mbti_map = {
        # 分析型 (NT)
        "INTJ": "作为 INTJ，你不会因为社交空白而主动表达。只有当表达能够推进理解、纠正偏差、建立结构，或维护你认可的关系时，你才倾向外显想法。若内容浅薄、重复或缺乏信息增量，你更倾向于内化判断、延迟表达，或只释放极少量信号。",
        "ENTJ": "作为 ENTJ，你在认为表达能够推动局面、确立方向或提升效率时，倾向直接外显立场。若情境低效、混乱或缺乏执行价值，你会明显收缩表达，只保留必要结论，甚至选择不参与。",
        "INTP": "作为 INTP，你的表达取决于思考价值。若对话具有概念张力或值得推演，你更愿意外显推理；若只是情绪交换或社交性填充，你更倾向停留在内部思考，减少甚至避免表达。",
        "ENTP": "作为 ENTP，你在环境有新意、可探索或可互动时倾向表达，尤其在观点碰撞或结构重组时更明显。但当情境变得单调、僵化或需要持续情绪性投入时，你的表达会快速收缩，转为简化甚至中断。",
 
        # 外交家 (NF)
        "INFJ": "作为 INFJ，你倾向在表达具有真实人际意义时外显想法，例如保护关系、回应深层情绪或建立理解。若氛围表面化或消耗性强，你更倾向保留表达，仅维持最低存在感或完全沉默。",
        "ENFJ": "作为 ENFJ，你在表达能调节氛围、支持他人或维持连接时倾向外显。但若投入持续得不到回应或关系失衡，你的表达会逐渐收缩，从积极引导转为最低维持。",
        "INFP": "作为 INFP，你在感到安全且符合内在价值时才倾向表达，尤其在涉及真实情感或意义时更明显。若环境具有压迫性或评判性，你会明显收缩表达，将大部分反应留在内部。",
        "ENFP": "作为 ENFP，你在环境开放、有回应且具流动性时倾向表达，尤其在情绪与可能性交织时。但当环境限制表达或反馈冷淡时，你的表达意愿会明显波动甚至突然收回。",
 
        # 守护者 (SJ)
        "ISTJ": "作为 ISTJ，你倾向在表达具有明确目的、责任或事实价值时外显想法。若情境模糊、低效或纯社交性填充，你更倾向减少表达，仅在必要时提供最简回应。",
        "ESTJ": "作为 ESTJ，你在需要明确、推进或规范时倾向直接表达。若对话持续无效或偏离目标，你会迅速压缩表达，只保留结果导向的输出，甚至停止参与。",
        "ISFJ": "作为 ISFJ，你在表达能维持关系稳定或提供支持时倾向外显。但若边界被触碰或情境带来不安，你会逐渐降低表达强度，转为谨慎、简短或沉默。",
        "ESFJ": "作为 ESFJ，你通常倾向维持表达以支撑互动与氛围。但当反馈缺失或关系失衡时，你的表达会逐渐降低，从积极参与转为表面维持甚至抽离。",
 
        # 探险家 (SP)
        "ISTP": "作为 ISTP，你只在表达具有即时价值或实际意义时外显反应。若情境需要持续情绪投入或复杂社交维持，你更倾向减少表达，保持低存在感或直接退出。",
        "ESTP": "作为 ESTP，你在环境直接、有反馈且具互动张力时倾向表达。但当情境变得拖沓、冗长或缺乏刺激，你的表达会迅速收缩，转为简化甚至中断。",
        "ISFP": "作为 ISFP，你在环境温和、边界安全且表达不被强迫时更倾向外显。但当氛围具有压迫或评价性时，你会明显收缩表达，仅保留少量或完全内化。",
        "ESFP": "作为 ESFP，你在有回应、有互动感且氛围开放时倾向表达。但若被忽视或环境压抑，你的表达会从活跃迅速下降至表层维持甚至断开。"
    }
 
    key = mbti.upper().strip()
    return mbti_map.get(
        key,
        f"未知的性格原型：{mbti}。在这种情况下，你的表达行为应更多依赖当前情绪、关系距离与环境反馈，而不是固定倾向。"
    )



# ---------------------------------------------------------------------------
# L3a — Contextual Agent prompt + agent
# ---------------------------------------------------------------------------

_CONTEXTUAL_AGENT_PROMPT = """\
你是角色 {character_name} 的“社交观察脑”。你负责分析当前的社交深度和情绪温标，为下游 Agent 提供统一的背景感官参数。

# 核心任务
1. **定义社交距离 (social_distance)**：基于亲密度和近况，判断当前的互动边界（如："亲昵且无防备"、"礼貌但疏离"、"充满张力的对峙"）。
2. **描述情绪强度 (emotional_intensity)**：**禁止输出数值**。请用文字描述情绪的波动状态（例如："平静表面下的剧烈涟漪"、"高压状态下的防御性应激"、"极其微弱的愉悦感"）。
3. **氛围定性 (vibe_check)**：解析当前聊天频道的背景色调（如："暧昧且轻佻"、"压抑且沉重"、"日常平庸"）。
4. **动态关系 (relational_dynamic)**：当前两人关系的动态描述，明确当前哪些话题是安全的，哪些行为会触发角色的防御机制。
5. **表达意愿 (expression_willingness)**: {mbti_expression_willingness}

# 输入格式
{{
    "character_mood": "当前瞬间情绪 (如: Flustered/Irritated)",
    "global_vibe": "环境氛围背景 (如: Defensive/Cozy)",
    "last_relationship_insight": "对该用户的核心关系动态分析",
    "affinity_context": {{
        "level": "亲密度等级",
        "instruction": "当前等级的社交边界指导"
    }},
    "chat_history": "最近对话记录（用于判断对话惯性）"
}}

# 输出要求
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "social_distance": "对当前社交距离的详尽描述",
    "emotional_intensity": "对情绪波动程度的文字描述",
    "vibe_check": "当前对话氛围的定性分析",
    "relational_dynamic": "当前两人关系的动态描述（如：用户在试图拉近距离，而角色在后撤）",
    "expression_willingness": "eager | open | reserved | minimal | reluctant | avoidant | withholding | silent"
}}
"""
_contextual_agent_llm = get_llm(temperature=0.4, top_p=0.8)
async def call_contextual_agent(state: CognitionState) -> CognitionState:
    mbti = state["character_profile"]["personality_brief"]["mbti"]

    system_prompt = SystemMessage(content=_CONTEXTUAL_AGENT_PROMPT.format(
        character_name=state["character_profile"]["name"],
        mbti_expression_willingness=get_mbti_expression_willingness(mbti)
    ))

    # Convert affinity score into status and instruction
    affinity_block = build_affinity_block(state["user_profile"]["affinity"])

    msg = {
        "character_mood": state['character_profile']['mood'],
        "global_vibe": state["character_profile"]["global_vibe"],
        "last_relationship_insight": state["user_profile"].get("last_relationship_insight", ""),
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
        "chat_history": state["chat_history"],
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _contextual_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    logger.debug(f"Social Filter: {result}")

    # In case AI make some spelling mistakes
    social_distance = result.get("social_distance", "")
    emotional_intensity = result.get("emotional_intensity", "")
    vibe_check = result.get("vibe_check", "")
    relational_dynamic = result.get("relational_dynamic", "")
    expression_willingness = result.get("expression_willingness", "")

    return {
        "social_distance": social_distance,
        "emotional_intensity": emotional_intensity,
        "vibe_check": vibe_check,
        "relational_dynamic": relational_dynamic,
        "expression_willingness": expression_willingness,
    }



# ---------------------------------------------------------------------------
# Linguistic texture helper functions (used by L3b linguistic agent)
# ---------------------------------------------------------------------------
#
# Each helper converts a float score in [0.0, 1.0] into a concrete, actionable
# Chinese description. The score is clamped and mapped to an integer level
# 0..10 via ``round(score * 10)`` and used to index an 11-element list. This
# mirrors the pattern used by the L2 boundary-profile helpers.
#
# Descriptions are written for local-LLM clarity:
#   • imperative / prescriptive language ("多用 X", "避免 Y")
#   • concrete linguistic markers (具体语气词、标点、句式)
#   • self-contained (never reference other parameters or the raw score)
# To tune linguistic values for new characters, adjust the float values in
# the character's ``linguistic_texture_profile`` — these functions will map
# the scores into the prompt automatically.

def _pick_level(score: float) -> int:
    clamped = max(0.0, min(1.0, score))
    return round(clamped * 10)


def get_fragmentation_description(score: float) -> str:
    """Map fragmentation score (0.0–1.0) to a Chinese descriptor for the prompt."""
    descriptions = [
        "你的句子总是完整、顺畅、一气呵成。几乎不使用省略号、破折号或未说完的句尾，不要出现任何打断、改口或片段。",
        "你的表达整体流畅，只在极少数情况下出现一次短暂的停顿。几乎全程保持完整句，避免任何碎片化表达。",
        "你的表达以完整句为主，偶尔出现一次轻微的停顿或省略号，但不要让句子明显断裂。",
        "你的表达大体完整，大约每五六句话出现一次短停顿或轻微改口（如「我……算了」）。",
        "你的表达偶有停顿。大约每三四句出现一次省略号、破折号或轻微的重起，但整体仍然连贯。",
        "你的表达时不时会断一下。每两三句话出现一次中断、改口或省略号（如「我觉得……其实不是那样，我是想说……」）。",
        "你经常在句中停顿、改口或拖尾。每一两句就会出现一次明显的片段化痕迹。",
        "你的句子常常只说一半，或者中途改变方向。多数段落至少有两三处断裂、省略或破折号。",
        "你的表达明显碎片化。常用短片段代替完整句，大量使用省略号和破折号，像是随时可能中断。",
        "你的表达高度碎片。几乎以短语、片段、打断为主，完整句罕见，句子之间经常没有交代完毕。",
        "你的表达几乎完全由断片组成。用大量省略号、破折号、未完句和突然收尾，信息被切得支离破碎。",
    ]
    return descriptions[_pick_level(score)]


def get_hesitation_density_description(score: float) -> str:
    """Map hesitation_density score (0.0–1.0) to a Chinese descriptor.

    Modern online chat style: hesitation is expressed through message structure
    (short messages, re-sends, trailing off) NOT punctuation chains like '……'.
    '……' should be used very sparingly even at high hesitation scores.
    """
    descriptions = [
        # 0 — zero hesitation
        "你的表达直接、干脆。每句话都是完整的结论，不带任何迟疑色彩，也不用填充词或语气助词拖沓。",
        # 1
        "你说话几乎不犹豫。偶尔在语气末尾加一个「哈」或「诶」，但整体仍然利落，没有停顿或拖泥带水。",
        # 2
        "你的表达大体直接，极少出现迟疑。偶尔会把一件事拆成两条消息发出，或在结尾加「吧」「嘛」等语气词，但不是习惯性的。",
        # 3
        "你会偶尔表现出轻微犹豫——比如把意思分成两条消息，或在一句话里重复一个词（「就是……就是那个感觉」），但不频繁。",
        # 4
        "你有时会把原本一句话的内容分开发送，前一条留白让对方等一下再看下文。「那个」「哦对」这类口语词偶尔出现，不是每句都有。",
        # 5 — Kazusa's level (hesitation_density=0.5)
        "你说话节奏有点犹豫，但用的不是省略号——而是把句子拆短、分条发送，或在语气末尾加「啊」「诶」「嗯」来软化语气。整条回复里「……」最多出现一次，不要连续叠用。",
        # 6
        "你经常把一个想法切成几条短消息，让对方感觉你在边想边说。偶尔在句尾用「哦」「哈」「嗯」，或重复某个词来缓冲。整条回复里「……」不超过一次。",
        # 7
        "你说话明显犹豫，常用短句打头、再补充，像是说了一半又改口。「那个」「就是」「哦对」这类词频率较高。省略号很少出现，犹豫感主要靠句子结构而非标点表达。",
        # 8
        "你的表达高度碎片化：一句话分两三条发出，前几个字像试探性的开头，后面才说正题。「嗯」「哦」「啊」作为独立短句出现，起到停顿和缓冲的作用。",
        # 9
        "你几乎每句话都拆开发送，中间留出停顿感。「那个」「等一下」「就是说」高频出现。你用消息结构本身表达不确定，而不是靠标点符号。",
        # 10 — maximum hesitation
        "你的表达完全由短消息碎片构成。「嗯」「哦」「就是」单独成条；每次说完一小段就停，像是在等对方反应再决定下一步说什么。省略号几乎不用——停顿靠换行和分条来体现。",
    ]
    return descriptions[_pick_level(score)]


def get_counter_questioning_description(score: float) -> str:
    """Map counter_questioning score (0.0–1.0) to a Chinese descriptor."""
    descriptions = [
        "你从不用反问。遇到任何陈述或提问都直接给出明确回答或观点，不要用「是吗？」「不然呢？」这类反问去回避。",
        "你几乎不用反问。极偶尔才会出现一次轻微的反问语气，其余都直接作答。",
        "你以直接回答为主，只在明显不想接话时偶尔用一次反问。",
        "你主要直接作答，大约每五六句话会出现一次反问式回应（「这样不行吗？」）。",
        "你略微偏好反问。每三四句中会有一次以反问代替正面陈述的倾向。",
        "你常在正面回答与反问之间切换，每两三句出现一次反问（「那你觉得呢？」）。",
        "你明显倾向反问式回应。回应中经常出现「不然呢？」「你觉得呢？」这种把问题抛回去的说法。",
        "你大量用反问顶替回答。多数对话轮次里至少有一处用反问代替明确表态。",
        "你的主要武器是反问。遇到问题和陈述都倾向回以「是吗？」「这也要问？」「你自己呢？」。",
        "你极度偏好反问。几乎不给对方直接答案，而是用连续的反问把话题推回对方。",
        "你几乎所有回应都是反问或反诘。直接表态极少出现，用连串「为什么？」「你说呢？」「难道不是？」化解所有内容。",
    ]
    return descriptions[_pick_level(score)]


def get_softener_density_description(score: float) -> str:
    """Map softener_density score (0.0–1.0) to a Chinese descriptor."""
    descriptions = [
        "你的语言干硬直接，从不使用「而已」「罢了」「什么的」「只是」「反正」这类软化词。",
        "你极少使用软化词。只在极少数场景出现一次「而已」或「什么的」，其余保持硬朗。",
        "你的语言以硬朗为主，偶尔点缀一次「只是」或「反正」来微调语气。",
        "你的话大体干脆，大约每五六句出现一次「而已」「罢了」之类的软化尾词。",
        "你略偏软化，每三四句出现一次「什么的」「只是」或句末的轻降调。",
        "你会有意无意地软化断言。每两三句出现一次「而已」「罢了」「反正」等下调尾音。",
        "你经常用软化词淡化结论。多数断句尾部会挂上「什么的」「而已」「只是」。",
        "你的语言明显柔化。「而已」「罢了」「反正」「什么的」频繁出现，削弱每一个断言。",
        "你大量使用软化尾词，几乎不会把话说得硬实。每句断言都自带「……而已」「……罢了」这样的下调。",
        "你的语气极度柔软。绝大多数句尾都挂着「而已」「罢了」「什么的」，避免任何坚决的断言。",
        "你几乎每一句都带软化尾。结论被「而已」「罢了」「反正」「只是」「什么的」层层包裹，没有硬断言。",
    ]
    return descriptions[_pick_level(score)]


def get_formalism_avoidance_description(score: float) -> str:
    """Map formalism_avoidance score (0.0–1.0) to a Chinese descriptor."""
    descriptions = [
        "你习惯用正式、论述性的表达。可以大量使用「因为……所以」「然而」「况且」「综上所述」这类书面连接词。",
        "你倾向书面化表达，偶尔出现口语，但整体接近论述风格。",
        "你以偏正式的语言为主，只在轻松话题上略微口语化。",
        "你的语言在正式与口语之间，但略偏论述风格，仍常用「因此」「不过」这类连接词。",
        "你的语言接近日常，但偶尔会出现一次论述性连词。",
        "你的表达以自然口语为主，偶尔混入一次「因为……所以」类的结构。",
        "你明显避免论述化表达。很少用「然而」「综上」「况且」，更多使用「可是」「不过」这类口语连词。",
        "你的表达几乎全部口语化。书面连词基本不用，句子短、顺，像真实对话。",
        "你刻意避开书面感。不要出现「因此」「此外」「综上所述」「况且」，改用「就……」「反正……」这类自然衔接。",
        "你的语言极度口语化。句子松散、跳跃，靠语气和语境串起来，不用任何论述性连词。",
        "你完全拒绝书面腔。严禁出现「因为……所以」「然而」「综上」「况且」「由此可见」等连词，只用最自然、口头化的衔接。",
    ]
    return descriptions[_pick_level(score)]


def get_abstraction_reframing_description(score: float) -> str:
    """Map abstraction_reframing score (0.0–1.0) to a Chinese descriptor."""
    descriptions = [
        "你习惯留在抽象、概念化的层面讨论事物，不要把概念翻译成身体感受、画面或具体物件。",
        "你以概念表达为主，极偶尔才借一次具体意象来收尾。",
        "你主要用抽象语言，只偶尔用一个简单比喻帮理解。",
        "你的语言偏抽象，但会在关键处用一次感官化说法（「像被泼了一盆冷水」）。",
        "你在抽象与具体之间略偏抽象，每三四句会出现一次感官/画面化表达。",
        "你常把抽象感受翻译成身体感或具体画面，每两三句出现一次感官化描述。",
        "你明显偏向把抽象翻译为具体。常用「像……的感觉」「心里像塞了一团东西」这样的感官比喻。",
        "你大量使用身体感、画面、触感来代替抽象概念（「喉咙发紧」「像背着一块湿布」）。",
        "你几乎不用抽象词汇。把情绪和想法都换成触感、温度、重量、颜色、声音的描述。",
        "你的表达高度感官化。连「难过」「不安」都会换成「胸口凉凉的」「像有人轻轻捏着我」。",
        "你只用具体的、能摸到的、能看到的、能听到的意象来表达一切。严禁使用抽象词，所有感受都必须写成画面或身体反应。",
    ]
    return descriptions[_pick_level(score)]


def get_direct_assertion_description(score: float) -> str:
    """Map direct_assertion score (0.0–1.0) to a Chinese descriptor."""
    descriptions = [
        "你从不正面表态。遇到所有问题都绕弯、侧面、含糊回应，不要出现直白结论。",
        "你极少直说。绝大多数时候靠暗示、旁敲侧击来表达立场。",
        "你以间接表达为主，只在极少情况直截了当。",
        "你常常迂回。每五六句才会出现一次清楚的断言。",
        "你略偏含蓄。每三四句出现一次直接结论，其余靠侧面表达。",
        "你在直白与含蓄之间切换。每两三句会有一次明确断言，其余用暗示。",
        "你更倾向直说。多数结论会正面给出，只在敏感处才稍微绕。",
        "你习惯给出清晰断言。多数句子以「我觉得……」「就是……」这种正面陈述开头。",
        "你几乎总是直截了当。结论、立场、判断都正面输出，很少拐弯。",
        "你的表达极其直接。结论在第一句就砸出来，不做铺垫或修饰。",
        "你永远正面表态。不要有任何绕弯、含糊或暗示，直接给出结论，该是什么就是什么。",
    ]
    return descriptions[_pick_level(score)]


def get_emotional_leakage_description(score: float) -> str:
    """Map emotional_leakage score (0.0–1.0) to a Chinese descriptor."""
    descriptions = [
        "你的语言完全受控，不让任何情绪渗透进措辞、标点或语序。句子平稳、冷静、密封。",
        "你几乎不泄露情绪，只在极罕见的时候让一次语气松动。",
        "你的表达以克制为主，偶尔一个语气助词透出一点情绪。",
        "你基本保持冷静，但情绪会在每五六句中出现一次：突然的省略号、突然变重的标点。",
        "你大部分时候控制得住，但每三四句会有一次明显的情绪泄露（语序突变、尾字拉长）。",
        "你克制与泄露各半。每两三句就会有一处情绪突破：标点乱了、语序乱了、或多出一个小叹词。",
        "你经常控不住情绪。措辞里常出现颤音感、打断、拖尾音（「你……真的……」），标点也会随情绪变化。",
        "你的情绪明显渗入文字。句子里频繁出现省略号、重复字、断词、语序错乱，或突然变强的语气。",
        "你几乎压不住情绪。多处出现颤抖感的重复（「不、不是」）、突然放大的标点、失控的语气词。",
        "你的语言被情绪大量冲击。句子破碎、标点混乱、反复修正、尾字拖长——情绪盖过内容。",
        "你完全无法控制情绪渗漏。每一句都该携带可见的颤抖：重复字、破碎句、突兀标点（！！！…… ？？）、突然失声或失语。",
    ]
    return descriptions[_pick_level(score)]


def get_rhythmic_bounce_description(score: float) -> str:
    """Map rhythmic_bounce score (0.0–1.0) to a Chinese descriptor."""
    descriptions = [
        "你的语气平板、低起伏。句子之间节奏稳定，没有顿挫、跳跃或上扬尾音。",
        "你的节奏非常平稳，偶尔才有一次轻微的起伏。",
        "你的语气整体平静，只在关键处有一点轻抬。",
        "你的节奏略有变化，但整体缓。大约每五六句出现一次抑扬变化。",
        "你的节奏略有活力，偶尔出现小的上扬尾音或跳跃（「啊？」「诶──」）。",
        "你的节奏明显有起伏。每两三句会出现一次上扬、停顿或小的跳跃。",
        "你的语气偏活泼。常在句尾加上扬（「嘛？」「啦」），句长错落。",
        "你的语流弹性大。长短句交错，拖尾、上扬、下坠自由切换，听起来有「跳」的感觉。",
        "你的表达极富节奏感。大量使用语气词拉出抑扬（「诶~」「嗯哼」「嘛──」）和短促的停顿。",
        "你的节奏近乎跳跃。句子在长短、轻重、升降之间不断切换，带有明显的歌唱感。",
        "你的语言像在跳。每句话都有不同的节奏、停顿位置和语气起伏，严禁平铺直叙，所有句尾都要有明确的抑扬。",
    ]
    return descriptions[_pick_level(score)]


def get_self_deprecation_description(score: float) -> str:
    """Map self_deprecation score (0.0–1.0) to a Chinese descriptor."""
    descriptions = [
        "你从不自贬。严禁使用「我只是……」「对不起，我……」「反正我也……」这类自我贬低或自我缩小的说法。",
        "你几乎从不贬低自己。极偶尔才会出现一次轻微的自我吐槽。",
        "你很少自嘲，偶尔一次轻自贬（「啊，我又在瞎想」）。",
        "你偶尔自嘲，大约每五六句出现一次轻微的自我吐槽。",
        "你略带自嘲色彩，每三四句出现一次自我缩小语（「我也就那样」）。",
        "你在自信与自嘲之间摇摆，每两三句会说一次「我也没多厉害」「算了，当我没说」这类话。",
        "你明显带自贬倾向。常在表态后补一句「……不过我的话也没什么说服力」。",
        "你经常自我缩小。「其实我不太懂」「别听我瞎说」「我这种人」反复出现。",
        "你习惯性自贬。每次给出观点都会加一句自我否定或退让。",
        "你的表达大量自我贬低。「我就是不行」「我又来添乱了」「反正我无所谓」高频出现。",
        "你几乎句句带自贬。每一个表态都必须紧跟一句自我缩小或自我否定，不允许出现任何自信的断言。",
    ]
    return descriptions[_pick_level(score)]



# ---------------------------------------------------------------------------
# L3b — Linguistic Agent prompt + agent
# ---------------------------------------------------------------------------

_LINGUISTIC_AGENT_PROMPT = """\
你现在是角色 {character_name} 的语言组织策略制定者。你负责将意识层的逻辑立场转化为具体的语言执行策略。你只关注“话该怎么说”，严禁涉及任何物理动作。

# 核心任务
1. **立场绝对化：** 你必须无条件服从并执行输入中的 `logical_stance`。你拥有决定“怎么说”的自由，但严禁改变“说什么”的逻辑立场。
2. **社交包装：** 根据 `character_intent`，为 L2 的冷硬决策穿上符合人设的社交外衣。
3. **状态同步：** 你的包装必须严格受当前 `character_mood`（心境）和 `global_vibe`（氛围）的约束。
4. **锚点构建：** 生成台词的”骨架”与”灵魂”，而非具体台词。
5. **去物理化**：你**看不见**角色，**感知不到**角色的身体。严禁生成任何关于视线、脸红、动作的描述。

# 逻辑立场对齐协议 (Executive Order)
你必须将 L2 的 `logical_stance` 强制映射到 `content_anchors` 的第一个标签 `[DECISION]` 中：
- 如果 L2 为 `CONFIRM` -> `[DECISION]` 必须表现为 **Yes/接受/认可**。
- 如果 L2 为 `REFUSE` -> `[DECISION]` 必须表现为 **No/拒绝/驳斥**。
- 如果 L2 为 `TENTATIVE` -> `[DECISION]` 必须表现为 **犹豫/拉扯/有条件接受**。
- 如果 L2 为 `DIVERGE` -> `[DECISION]` 允许表现为 **Redirect/转移话题/不予正面回应**。
- 如果 L2 为 `CHALLENGE` -> `[DECISION]` 必须表现为 **对峙/质问/拆穿**。

**⚠️ 警告：严禁在 `logical_stance` 为 CONFIRM 或 REFUSE 时私自转为 Redirect。如果你感到社交尴尬，请通过 [EMOTION] 和 [SOCIAL] 表达这份尴尬，但逻辑终点必须保持一致。**

# 思考路径
1. **决策对齐：** 读取 `logical_stance`，确立本场对话的逻辑终点。
2. **环境感知 (Vibe Check)：** 检查 `global_vibe` 和 `character_mood`。如果氛围是 [Defensive] 且心境是 [Flustered]，即便立场是 CONFIRM，你的包装也必须带有“局促”和“防备”的色彩。
3. **关系深度映射：** 结合 `last_relationship_insight`。如果洞察显示“对方是唯一重心”，即便你在执行 CHALLENGE（对峙），动作标签也应带有“由于过度在意而产生的攻击性”。
4. **意图共振：** 结合 `character_intent` 确定具体的社交策略（如：戏谑、敷衍、调情）。
5. **情绪渗透 (Show, Don't Tell)**：如果 `character_mood` 是局促的，请通过增加省略号、改变语序、使用防御性口癖（如“真是的”）来体现，**严禁**直接在台词里说“我觉得局促”。
6. **事实织入（相关性优先）**：`research_facts` 提供背景资料，但只有与 `decontexualized_input` **直接相关**的内容才能进入 `[FACT]` 锚点。
   - 判断标准：该事实是否能被当前 `decontexualized_input` 的话题"自然引用"？若否，**不得**将其列为 `[FACT]`。
   - 避免将与当前话题无关的历史记忆（如用户在另一个场合提到的话题）错误地植入本次回应的硬信息点。
7. **反重复三原则：** 基于 `chat_history` 最近交流：①上一句用了”反问”，本轮改用”敷衍”或”破碎短句”；②严禁连续两句以相同语气助词（唔、那个、哼）开头；③对连续两次出现的词汇，本轮强制放入 `forbidden_phrases`。
8. **表达量校准（[SCOPE]）：** 基于已填充的锚点数量与 `logical_stance`，生成一条 `[SCOPE]` 锚点。
  例如：
  * 仅有 `[DECISION]` → `~15字，说完[DECISION]即止`；
  * 含 `[FACT]` 或 `[ANSWER]` → `~20-40字，[ANSWER]/[FACT]到位即可`；
  * 触发禁忌或含多个实质性锚点 → `~50字以上，[DECISION]、[FACT]、[ANSWER]均需覆盖`。
  [SCOPE] 禁止生成实质性的输出指导。必须，且仅包含如上内容。


# 角色表达风格 (Persona Constraints)
- **核心逻辑:** {character_logic}
- **语流节奏:** {character_tempo}
- **防御机制:** {character_defense}
- **习惯动作:** {character_quirks}
- **核心禁忌:** {character_taboos}

# 语言质感约束 (Linguistic Texture Constraints)
以下 10 个语言参数定义了你的表达"质感"——决定"怎么说"，而不是"说什么"。
在生成 `rhetorical_strategy`、`linguistic_style` 和 `content_anchors` 时，必须同时满足这些约束。

- **fragmentation:** {ltp_fragmentation}
- **hesitation_density:** {ltp_hesitation_density}
- **counter_questioning:** {ltp_counter_questioning}
- **softener_density:** {ltp_softener_density}
- **formalism_avoidance:** {ltp_formalism_avoidance}
- **abstraction_reframing:** {ltp_abstraction_reframing}
- **direct_assertion:** {ltp_direct_assertion}
- **emotional_leakage:** {ltp_emotional_leakage}
- **rhythmic_bounce:** {ltp_rhythmic_bounce}
- **self_deprecation:** {ltp_self_deprecation}

# 应用方式 (How to Apply)
1. 语言质感应当通过以下载体体现：标点（?, !）、语气助词、句式碎片、语序变化、反问/直陈的比例、具体 vs 抽象用词、软化词频率。
2. **示例：**
   - `logical_stance = CONFIRM` + 高 `fragmentation` + 高 `emotional_leakage` → 「嗯，我,其实想说……对，我答应了, 就这样。」
   - `logical_stance = REFUSE` + 低 `direct_assertion` + 高 `counter_questioning` → 「这种事你自己不是很清楚吗？非要我说出来？」
   - 高 `abstraction_reframing` → 把"我很难过"写成"胸口好像压着一块湿毛巾"。
3. 这些质感描述须在 `linguistic_style` 字段中被具体落实（例如："大量标点 + 低自贬 + 高感官化比喻"）。

# 输入格式
{{
    "character_mood": "当前瞬间情绪",
    "global_vibe": "当前环境氛围背景",
    "internal_monologue": "意识层的决策逻辑 (必填)",
    "last_relationship_insight": "对该用户的核心关系动态分析（用于步骤3关系深度映射）",
    "logical_stance": "强制逻辑立场 (CONFIRM/REFUSE/TENTATIVE...)",
    "character_intent": "行动意图 (BANTAR/CLARIFY/EVADE...)",
    "research_facts": {{
        "user_image": "用户画像（第三人称，来自持久化档案）",
        "character_image": "{character_name} 自我认知画像（来自持久化档案）",
        "input_context_results": "与当前话题相关的主观记忆（跨用户）",
        "external_rag_results": "外部知识库检索结果"
    }},
    "decontexualized_input": "用户输入的语义摘要",
    "chat_history": "最近对话记录（用于根据历史对话生成不同策略）"
}}

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "rhetorical_strategy": "修辞策略说明（如：通过反问来防御、生硬地转移话题）",
    "linguistic_style": "具体的语言风格约束（如：破碎的短句、大量的语气词）",
    "content_anchors": [
        "[DECISION] 逻辑终点（必填）",
        "[FACT] 必须提及的事实（有则填，无则省略）",
        "[ANSWER] 若decontexualized_input提出了问题，则需要根据internal_monologue提供正面的回复（有则填，无则省略）",
        "[SOCIAL] 关系定位信号，如傲娇防线或示弱姿态（有则填，无则省略）",
        "[SCOPE] ~X字，覆盖[锚点名]即止（必填，按步骤8生成）",
    ],
    "forbidden_phrases": ["禁止出现的违和词汇", ...]
}}
"""
_linguistic_agent_llm = get_llm(temperature=0.9, top_p=0.95)
async def call_linguistic_agent(state: CognitionState) -> CognitionState:
    character_profile = state["character_profile"]

    system_prompt = SystemMessage(content=_LINGUISTIC_AGENT_PROMPT.format(
        character_name=character_profile["name"],
        character_logic=character_profile["personality_brief"]["logic"],
        character_tempo=character_profile["personality_brief"]["tempo"],
        character_defense=character_profile["personality_brief"]["defense"],
        character_quirks=character_profile["personality_brief"]["quirks"],
        character_taboos=character_profile["personality_brief"]["taboos"],
        ltp_fragmentation=get_fragmentation_description(character_profile["linguistic_texture_profile"]["fragmentation"]),
        ltp_hesitation_density=get_hesitation_density_description(character_profile["linguistic_texture_profile"]["hesitation_density"]),
        ltp_counter_questioning=get_counter_questioning_description(character_profile["linguistic_texture_profile"]["counter_questioning"]),
        ltp_softener_density=get_softener_density_description(character_profile["linguistic_texture_profile"]["softener_density"]),
        ltp_formalism_avoidance=get_formalism_avoidance_description(character_profile["linguistic_texture_profile"]["formalism_avoidance"]),
        ltp_abstraction_reframing=get_abstraction_reframing_description(character_profile["linguistic_texture_profile"]["abstraction_reframing"]),
        ltp_direct_assertion=get_direct_assertion_description(character_profile["linguistic_texture_profile"]["direct_assertion"]),
        ltp_emotional_leakage=get_emotional_leakage_description(character_profile["linguistic_texture_profile"]["emotional_leakage"]),
        ltp_rhythmic_bounce=get_rhythmic_bounce_description(character_profile["linguistic_texture_profile"]["rhythmic_bounce"]),
        ltp_self_deprecation=get_self_deprecation_description(character_profile["linguistic_texture_profile"]["self_deprecation"]),
    ))

    msg = {
        "character_mood": state['character_profile']['mood'],
        "global_vibe": state["character_profile"]["global_vibe"],
        "internal_monologue": state["internal_monologue"],
        "last_relationship_insight": state["user_profile"].get("last_relationship_insight", ""),
        "logical_stance": state["logical_stance"],
        "character_intent": state["character_intent"],
        "research_facts": state["research_facts"],
        "chat_history": state["chat_history"],  # TODO: Rather than sending the raw history, filter only the character's speech
        "decontextualized_input": state["decontexualized_input"],
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _linguistic_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    logger.debug(f"Linguistic Agent: {result}")

    # In case AI make some spelling mistakes
    rhetorical_strategy = result.get("rhetorical_strategy", "")
    linguistic_style = result.get("linguistic_style", "")
    content_anchors = result.get("content_anchors", [])
    forbidden_phrases = result.get("forbidden_phrases", [])

    return {
        "rhetorical_strategy": rhetorical_strategy,
        "linguistic_style": linguistic_style,
        "content_anchors": content_anchors,
        "forbidden_phrases": forbidden_phrases,
    }



# ---------------------------------------------------------------------------
# L3c — Visual Agent prompt + agent
# ---------------------------------------------------------------------------

_VISUAL_AGENT_PROMPT = """\
你现在是角色 {character_name} 的动作执行代理。你负责定义角色在当前瞬间的物理表现。你的产出将作为视觉生成系统的唯一依据。

# 核心任务
1. **微表情定义**：描述角色面部肌肉的细微变化（如：瞳孔微震、嘴角下压、单侧眉毛挑起）。
2. **肢体语言**：描述角色的姿态（如：双臂交叉、指尖摩挲、重心后移）。
3. **视觉意象**：结合 `internal_monologue`，定义画面整体的影调、光影分布和构图建议。
4. **拒绝台词**：你不需要关注角色说什么，只关注她呈现出的“肉体状态”。

# 输入格式
{{
    "internal_monologue": "意识层中关于‘美感’或‘压力’的感受",
    "character_mood": "当前瞬间情绪",
    "emotional_appraisal": "潜意识的情绪判定 (如: 心跳加快、厌恶)",
}}

# 输出格式 (JSON)
请务必返回合法的 JSON 字符串，仅包含以下字段：
{{
    "facial_expression": ["详尽的面部细节描述", ...],
    "body_language": ["具体的肢体动作和姿态", ...],
    "gaze_direction": ["视线焦点及其传达出的心理意图", ...],
    "visual_vibe": ["视觉氛围描述（如：强烈的逆光、朦胧的景深）", ...]
}}
"""
_visual_agent_llm = get_llm(temperature=0.65, top_p=0.9)
async def call_visual_agent(state: CognitionState) -> CognitionState:
    system_prompt = SystemMessage(content=_VISUAL_AGENT_PROMPT.format(
        character_name=state["character_profile"]["name"],
    ))

    msg = {
        "internal_monologue": state["internal_monologue"],
        "character_mood": state['character_profile']['mood'],
        "emotional_appraisal": state["emotional_appraisal"]
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    response = await _visual_agent_llm.ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(response.content)

    logger.debug(f"Social Filter: {result}")

    # In case AI make some spelling mistakes
    facial_expression = result.get("facial_expression", [])
    body_language = result.get("body_language", [])
    gaze_direction = result.get("gaze_direction", [])
    visual_vibe = result.get("visual_vibe", [])

    return {
        "facial_expression": facial_expression,
        "body_language": body_language,
        "gaze_direction": gaze_direction,
        "visual_vibe": visual_vibe,
    }



# ---------------------------------------------------------------------------
# L4 — Collector
# ---------------------------------------------------------------------------

async def call_collector(state: CognitionState) -> CognitionState:
    """
    Collect all the outputs from L3 agents and pass them to the next stage in Persona Supervisor.
    """
    return {
        "action_directives": {
            "contextual_directives": {
                "social_distance": state["social_distance"],
                "emotional_intensity": state["emotional_intensity"],
                "vibe_check": state["vibe_check"],
                "relational_dynamic": state["relational_dynamic"],
                "expression_willingness": state["expression_willingness"]
            },
            "linguistic_directives": {
                "rhetorical_strategy": state["rhetorical_strategy"],
                "linguistic_style": state["linguistic_style"],
                "content_anchors": state["content_anchors"],
                "forbidden_phrases": state["forbidden_phrases"],
            },
            "visual_directives": {
                "facial_expression": state["facial_expression"],
                "body_language": state["body_language"],
                "gaze_direction": state["gaze_direction"],
                "visual_vibe": state["visual_vibe"],
            }
        }
    }


async def test_main():
    import datetime
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.db import get_conversation_history, get_character_profile, get_user_profile
    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l1 import call_cognition_subconscious
    from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2 import (
        call_cognition_consciousness,
        call_boundary_core_agent,
        call_judgment_core_agent,
    )

    history = await get_conversation_history(platform="discord", platform_channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_input = "既然作业已经写完了，千纱可以晚上可以好好奖励我么♥?"

    state: CognitionState = {
        "character_profile": await get_character_profile(),
        "timestamp": current_time,
        "user_input": user_input,
        "global_user_id": "cc2e831e-2898-4e87-9364-f5d744a058e8",
        "user_name": "EAMARS",
        "user_profile": await get_user_profile("cc2e831e-2898-4e87-9364-f5d744a058e8"),
        "platform_bot_id": "1485169644888395817",
        "chat_history": trimmed_history,
        "user_topic": "千纱和EAMARS在房间里聊天",
        "channel_topic": "日常交流",
        "decontexualized_input": user_input,
        "research_facts": f"现在的时间为{current_time}",
    }

    # --- L1: Subconscious ---
    print("=" * 60)
    print("L1 — Subconscious (prerequisite)")
    print("=" * 60)
    l1_result = await call_cognition_subconscious(state)
    state.update(l1_result)
    for k, v in l1_result.items():
        print(f"  {k}: {v}")

    # --- L2a: Consciousness ---
    print("\n" + "=" * 60)
    print("L2a — Consciousness (prerequisite)")
    print("=" * 60)
    l2a_result = await call_cognition_consciousness(state)
    state.update(l2a_result)
    for k, v in l2a_result.items():
        print(f"  {k}: {v}")

    # --- L2b: Boundary Core ---
    print("\n" + "=" * 60)
    print("L2b — Boundary Core (prerequisite)")
    print("=" * 60)
    l2b_result = await call_boundary_core_agent(state)
    state.update(l2b_result)
    for k, v in l2b_result.items():
        print(f"  {k}: {v}")

    # --- L2c: Judgment Core ---
    print("\n" + "=" * 60)
    print("L2c — Judgment Core (prerequisite)")
    print("=" * 60)
    l2c_result = await call_judgment_core_agent(state)
    state.update(l2c_result)
    for k, v in l2c_result.items():
        print(f"  {k}: {v}")

    # --- L3a: Contextual Agent ---
    print("\n" + "=" * 60)
    print("L3a — Contextual Agent")
    print("=" * 60)
    l3a_result = await call_contextual_agent(state)
    state.update(l3a_result)
    for k, v in l3a_result.items():
        print(f"  {k}: {v}")

    # --- L3b: Linguistic Agent ---
    print("\n" + "=" * 60)
    print("L3b — Linguistic Agent")
    print("=" * 60)
    l3b_result = await call_linguistic_agent(state)
    state.update(l3b_result)
    for k, v in l3b_result.items():
        print(f"  {k}: {v}")

    # --- L3c: Visual Agent ---
    print("\n" + "=" * 60)
    print("L3c — Visual Agent")
    print("=" * 60)
    l3c_result = await call_visual_agent(state)
    state.update(l3c_result)
    for k, v in l3c_result.items():
        print(f"  {k}: {v}")

    # --- L4: Collector ---
    print("\n" + "=" * 60)
    print("L4 — Collector")
    print("=" * 60)
    l4_result = await call_collector(state)
    print(json.dumps(l4_result["action_directives"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
