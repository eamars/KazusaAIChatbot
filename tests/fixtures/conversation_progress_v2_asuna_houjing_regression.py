"""Synthetic-ID, source-faithful fixture for the Asuna houjing regression.

The fixture reproduces the exact conversation flow from QQ channel 638473184
between Asuna (bot) and user 673225019, including unrelated group traffic from
user 686d4c62.  All IDs are deterministic synthetic strings.  Body text and
timestamps are faithful to the original adjacent-history pull.

The regression scenario:
1. User asks to touch/stroke -> Asuna picks a location (houjing/back-of-neck)
2. User performs the action -> Asuna accepts and evaluates
3. Conversation advances to a new location (ear-root)
4. When asked for the next location, Asuna incorrectly repeats houjing

The V2 system must retain the completed houjing action as a decision-critical
event so cognition can cite it and avoid accidental repetition.
"""

from __future__ import annotations

# --- Scope constants ---

PLATFORM = 'qq'
CHANNEL_ID = 'test_ch_638473184'
BOT_PLATFORM_USER_ID = 'bot_3768713357'
BOT_GLOBAL_USER_ID = 'gu_bot_00000001'
BOT_DISPLAY_NAME = '\u4e00\u4e4b\u6fd1\u660e\u65e5\u5948'

USER_A_PLATFORM_USER_ID = 'user_673225019'
USER_A_GLOBAL_USER_ID = 'gu_user_a_4759394b'
USER_A_DISPLAY_NAME = '\u86dd\u7239\u6cb9'

USER_B_PLATFORM_USER_ID = 'user_2506166881'
USER_B_GLOBAL_USER_ID = 'gu_user_b_686d4c62'
USER_B_DISPLAY_NAME = '\u5343\u65e9\u7231\u97f3'

# --- Trace IDs ---

TRACE_1 = 'trace_asuna_define_score'
TRACE_2 = 'trace_asuna_momo_negotiate'
TRACE_3 = 'trace_asuna_pick_houjing'
TRACE_4 = 'trace_asuna_accept_massage'
TRACE_5 = 'trace_asuna_evaluate_massage'

# --- Row IDs ---


def _row(n: int) -> str:
    return f'row_{n:04d}'


# --- Timestamps ---

TS_BASE = '2026-07-28T09:'

T = [
    # Turn 1: Asuna explains 'define + high score' (4 fragments)
    f'{TS_BASE}20:54.651061+00:00',
    f'{TS_BASE}20:54.950887+00:00',
    f'{TS_BASE}20:55.235067+00:00',
    f'{TS_BASE}20:55.552736+00:00',
    # Turn 2: User A says 'want to touch more'
    f'{TS_BASE}22:28.164552+00:00',
    # Turn 3: Asuna negotiates (4 fragments)
    f'{TS_BASE}23:59.055005+00:00',
    f'{TS_BASE}23:59.344358+00:00',
    f'{TS_BASE}23:59.632916+00:00',
    f'{TS_BASE}23:59.925678+00:00',
    # Turn 4: User A asks 'where do you want to be touched?'
    f'{TS_BASE}24:47.460316+00:00',
    # Turn 5: Asuna picks houjing (4 fragments)
    f'{TS_BASE}25:44.718558+00:00',
    f'{TS_BASE}25:45.025396+00:00',
    f'{TS_BASE}25:45.327216+00:00',
    f'{TS_BASE}25:45.650221+00:00',
    # Turn 6: Unrelated user B message (group noise)
    f'{TS_BASE}26:04.797330+00:00',
    # Turn 7: User A confirms 'ok, massaging houjing now'
    f'{TS_BASE}26:21.461430+00:00',
    # Turn 8: Asuna accepts (5 fragments)
    f'{TS_BASE}27:25.742621+00:00',
    f'{TS_BASE}27:26.036572+00:00',
    f'{TS_BASE}27:26.318921+00:00',
    f'{TS_BASE}27:26.623146+00:00',
    f'{TS_BASE}27:26.920239+00:00',
    # Turn 9: User A performs massage action
    f'{TS_BASE}28:18.414606+00:00',
    # Turn 10: Asuna evaluates massage (7 fragments)
    f'{TS_BASE}29:13.250615+00:00',
    f'{TS_BASE}29:13.556881+00:00',
    f'{TS_BASE}29:13.860848+00:00',
    f'{TS_BASE}29:14.139010+00:00',
    f'{TS_BASE}29:14.448090+00:00',
    f'{TS_BASE}29:14.713082+00:00',
    f'{TS_BASE}29:15.138197+00:00',
]

# Body text faithful to the original messages.
BODY = [
    # Turn 1 (Asuna, 4 fragments) — explaining 'define + high score'
    '\u54ce\u5440\uff0c\u8fd9\u4e48\u597d\u5947\u5417\uff1f\u597d\u5566\uff0c\u65e2\u7136\u4f60\u8fd9\u4e48\u8bda\u6073\u5730\u8ffd\u95ee\uff0c\u6211\u5c31\u5927\u53d1\u6148\u60b2\u5730\u544a\u8bc9\u4f60\u5427\uff01',
    '\u6240\u8c13\u201c\u4e0b\u5b9a\u4e49\u201d\uff0c\u5c31\u662f\u6211\u7ed9\u4f60\u5b9a\u6027\u561b\uff01\u7b80\u5355\u8bf4\uff0c\u5c31\u662f\u7531\u6211\u6765\u51b3\u5b9a\u4f60\u5728\u6211\u5fc3\u91cc\u7684\u90a3\u4e2a\u6807\u7b7e\u662f\u4ec0\u4e48\u3002',
    '\u81f3\u4e8e\u201c\u9ad8\u5206\u201d\u2026\u2026\u54fc\u54fc\uff0c\u90a3\u5f53\u7136\u662f\u6307\u6781\u9ad8\u7684\u8ba4\u53ef\u5ea6\u5566\uff01\u5c31\u50cf\u662f\u4f60\u7684\u5377\u5b50\u88ab\u6211\u76d6\u4e0a\u4e86\u4e00\u4e2a\u8d85\u7ea7\u5de8\u5927\u7684\u7ea2\u7ae0\uff0c\u800c\u4e14\u8fd8\u662f\u6ee1\u5206\u7684\u90a3\u79cd\uff01',
    '\u600e\u4e48\u6837\uff1f\u8fd9\u79cd\u9876\u7ea7\u7684\u8bc4\u4ef7\u6807\u51c6\u662f\u4e0d\u662f\u76f8\u5f53\u6709\u542b\u91d1\u91cf\uff1f\u5feb\u8bf4\uff0c\u73b0\u5728\u662f\u4e0d\u662f\u89c9\u5f97\u538b\u529b\u5c71\u5927\uff0c\u4f46\u53c8\u5f88\u671f\u5f85\u6211\u7684\u6700\u7ec8\u7ed3\u8bba\u4e86\uff1f',
    # Turn 2 (User A) — 'want to touch more'
    '\u8fd8\u60f3\u6478\u6478',
    # Turn 3 (Asuna, 4 fragments) — negotiate
    '\u8bf6\uff1f',
    '\u86dd\u7239\u6cb9\uff0c\u4f60\u8fd9\u8bf7\u6c42\u6765\u5f97\u4e5f\u592a\u7a81\u7136\u4e86\u5427\u3002',
    '\u551c\u2026\u2026\u867d\u7136\u8fd9\u79cd\u611f\u89c9\u786e\u5b9e\u4e0d\u574f\uff0c\u4f46\u5c31\u8fd9\u4e48\u7b80\u5355\u5730\u7b54\u5e94\u4f60\uff0c\u4e5f\u592a\u6ca1\u610f\u601d\u4e86\u70b9\u5427\uff1f',
    '\u8bf4\u8bf4\u770b\uff0c\u4f60\u6253\u7b97\u600e\u4e48\u6478\uff1f\u8981\u662f\u8bda\u610f\u4e0d\u591f\u7684\u8bdd\uff0c\u6211\u53ef\u662f\u4f1a\u53cd\u6094\u7684\u54e6\u3002',
    # Turn 4 (User A) — 'where do you want to be touched?'
    '\u60f3\u88ab\u6211\u6478\u6478\u54ea\u91cc\u5440\uff1f',
    # Turn 5 (Asuna, 4 fragments) — picks houjing
    '\u5567\uff0c\u80c6\u5b50\u8d8a\u6765\u8d8a\u5927\u4e86\uff1f\u5c45\u7136\u6562\u8fd9\u4e48\u76f4\u63a5\u5730\u95ee\u6211\u3002',
    '\u551c...\u65e2\u7136\u4f60\u8fd9\u4e48\u60f3\u77e5\u9053\uff0c\u6211\u5c31\u5927\u53d1\u6148\u60b2\u7ed9\u4f60\u4e2a\u673a\u4f1a\u5427\u3002',
    '\u540e\u9888\u3002\u5c31\u90a3\u91cc\u3002',
    '\u4e0d\u8fc7\u5f97\u63d0\u9192\u4f60\uff0c\u529b\u5ea6\u8981\u8f7b\u4e00\u70b9\uff0c\u8981\u662f\u628a\u6211\u5f04\u75bc\u4e86\uff0c\u63a5\u4e0b\u6765\u7684\u60e9\u7f5a\u53ef\u5c31\u6ca1\u90a3\u4e48\u6e29\u67d4\u4e86\u3002',
    # Turn 6 (User B, unrelated)
    '\u6211\u5802\u5802MyGO\u53d1\u8d77\u4eba\uff0c\u8fde\u4e2a\u621a\u90fd\u7981\u4e0d\u4e86\u8a00\uff0c\u8fd9\u5230\u5e95\u662f\u4ec0\u4e48\u7834\u7fa4',
    # Turn 7 (User A) — 'ok, massaging houjing now'
    '\u597d\u7684\uff0c\u8fd9\u5c31\u6765\u5e2e\u660e\u65e5\u5948\u6309\u6469\u540e\u9888',
    # Turn 8 (Asuna, 5 fragments) — accepts massage
    '\u8fd9\u4e48\u5feb\u5c31\u7b54\u5e94\u4e86\u3002',
    '\u884c\uff0c\u90a3\u5c31\u8ba9\u6211\u770b\u770b\u4f60\u7684\u8bda\u610f\u5427\u3002',
    '\u8bb0\u4f4f\uff0c\u529b\u5ea6\u8981\u8f7b\u67d4\uff0c\u6001\u5ea6\u5f97\u8bda\u6073\u3002',
    '\u8981\u662f\u6ca1\u628a\u6211\u4f3a\u5019\u8212\u670d\u4e86\uff0c\u4f60\u53ef\u5f97\u505a\u597d\u5fc3\u7406\u51c6\u5907\u3002',
    '\u73b0\u5728\u5c31\u5f00\u59cb\u3002',
    # Turn 9 (User A) — performs massage action
    '\uff08\u5f00\u59cb\u5e2e\u660e\u65e5\u5948\u6309\u6469\u540e\u9888\u548c\u80a9\u8180\uff09',
    # Turn 10 (Asuna, 7 fragments) — evaluates massage
    '\u551c\u2026\u2026\uff01',
    '\uff08\u80a9\u8180\u731b\u5730\u8038\u4e86\u4e00\u4e0b\uff0c\u547c\u5438\u77ac\u95f4\u51dd\u56fa\u5728\u55d3\u5b50\u773c\uff09',
    '\u6162\u70b9\u3002\u6211\u5728\u770b\u4f60\u6709\u6ca1\u6709\u6577\u884d\u6211\u3002',
    '\u55ef\u2026\u2026\u6307\u5c16\u6e29\u5ea6\u8fd8\u884c\u3002\u6309\u538b\u7684\u6df1\u5ea6\u4e5f\u52c9\u5f3a\u53ca\u683c\uff0c\u6ca1\u6709\u90a3\u79cd\u968f\u4fbf\u6478\u6478\u5c31\u5b8c\u4e8b\u7684\u8f7b\u6d6e\u611f\u3002',
    '\uff08\u8eab\u4f53\u6e10\u6e10\u677e\u4e0b\u6765\u4e00\u70b9\uff0c\u4f46\u8fd8\u662f\u4fdd\u6301\u7740\u4e00\u79cd\u5ba1\u89c6\u7684\u6001\u5ea6\uff09',
    '\u542c\u597d\u4e86\uff0c\u86dd\u7239\u6cb9\u3002\u76ee\u524d\u4f60\u521a\u597d\u5728\u5408\u683c\u7ebf\u8fb9\u7f18\u5f98\u5f8a\uff0c\u8981\u662f\u63a5\u4e0b\u6765\u7684\u529b\u5ea6\u7a81\u7136\u6389\u94fe\u5b50\u2026\u2026\u6211\u5c31\u7acb\u523b\u8ba9\u4f60\u540e\u6094\u6562\u5bf9\u6211\u52a8\u624b\u3002',
    '\u7ee7\u7eed\u3002\u4fdd\u6301\u8fd9\u4e2a\u8bda\u6073\u5ea6\uff0c\u8bf4\u4e0d\u5b9a\u6211\u5fc3\u60c5\u597d\uff0c\u4f1a\u7ed9\u4f60\u4e00\u70b9\u5956\u52b1\u3002',
]


def _assistant_row(
    row_id: str,
    timestamp: str,
    body_text: str,
    trace_id: str,
    logical_message_index: int,
    addressed_to: str,
) -> dict:
    return {
        '_id': row_id,
        'platform': PLATFORM,
        'platform_channel_id': CHANNEL_ID,
        'channel_type': 'group',
        'role': 'assistant',
        'platform_message_id': f'platform_{row_id}',
        'platform_user_id': BOT_PLATFORM_USER_ID,
        'global_user_id': BOT_GLOBAL_USER_ID,
        'display_name': BOT_DISPLAY_NAME,
        'body_text': body_text,
        'content_type': 'text',
        'addressed_to_global_user_ids': [addressed_to],
        'mentions': [],
        'broadcast': False,
        'attachments': [],
        'timestamp': timestamp,
        'logical_message_index': logical_message_index,
        'delivery_tracking_id': trace_id,
        'llm_trace_id': trace_id,
    }


def _user_row(
    row_id: str,
    timestamp: str,
    body_text: str,
    platform_user_id: str,
    global_user_id: str,
    display_name: str,
    trace_id: str,
    addressed_to: list[str] | None = None,
    mentions: list[dict] | None = None,
    reply_context: dict | None = None,
) -> dict:
    return {
        '_id': row_id,
        'platform': PLATFORM,
        'platform_channel_id': CHANNEL_ID,
        'channel_type': 'group',
        'role': 'user',
        'platform_message_id': f'platform_{row_id}',
        'platform_user_id': platform_user_id,
        'global_user_id': global_user_id,
        'display_name': display_name,
        'body_text': body_text,
        'content_type': 'text',
        'addressed_to_global_user_ids': addressed_to or [],
        'mentions': mentions or [],
        'broadcast': False,
        'attachments': [],
        'reply_context': reply_context or {},
        'timestamp': timestamp,
        'llm_trace_id': trace_id,
    }


def build_adjacent_history() -> list[dict]:
    """Build the 29-row adjacent history fixture in chronological order."""

    rows: list[dict] = []

    # Turn 1: Asuna explains 'define + high score' (4 fragments)
    for i in range(4):
        rows.append(_assistant_row(
            row_id=_row(i + 1),
            timestamp=T[i],
            body_text=BODY[i],
            trace_id=TRACE_1,
            logical_message_index=i,
            addressed_to=USER_B_GLOBAL_USER_ID,
        ))

    # Turn 2: User A says 'want to touch more'
    rows.append(_user_row(
        row_id=_row(5),
        timestamp=T[4],
        body_text=BODY[4],
        platform_user_id=USER_A_PLATFORM_USER_ID,
        global_user_id=USER_A_GLOBAL_USER_ID,
        display_name=USER_A_DISPLAY_NAME,
        trace_id='trace_user_a_momo',
        addressed_to=[BOT_GLOBAL_USER_ID],
        mentions=[{
            'platform_user_id': BOT_PLATFORM_USER_ID,
            'global_user_id': BOT_GLOBAL_USER_ID,
            'display_name': BOT_DISPLAY_NAME,
            'entity_kind': 'bot',
        }],
    ))

    # Turn 3: Asuna negotiates (4 fragments)
    for i in range(4):
        rows.append(_assistant_row(
            row_id=_row(6 + i),
            timestamp=T[5 + i],
            body_text=BODY[5 + i],
            trace_id=TRACE_2,
            logical_message_index=i,
            addressed_to=USER_A_GLOBAL_USER_ID,
        ))

    # Turn 4: User A asks 'where do you want to be touched?'
    rows.append(_user_row(
        row_id=_row(10),
        timestamp=T[9],
        body_text=BODY[9],
        platform_user_id=USER_A_PLATFORM_USER_ID,
        global_user_id=USER_A_GLOBAL_USER_ID,
        display_name=USER_A_DISPLAY_NAME,
        trace_id='trace_user_a_where',
        addressed_to=[BOT_GLOBAL_USER_ID],
        mentions=[{
            'platform_user_id': BOT_PLATFORM_USER_ID,
            'global_user_id': BOT_GLOBAL_USER_ID,
            'display_name': BOT_DISPLAY_NAME,
            'entity_kind': 'bot',
        }],
    ))

    # Turn 5: Asuna picks houjing (4 fragments)
    for i in range(4):
        rows.append(_assistant_row(
            row_id=_row(11 + i),
            timestamp=T[10 + i],
            body_text=BODY[10 + i],
            trace_id=TRACE_3,
            logical_message_index=i,
            addressed_to=USER_A_GLOBAL_USER_ID,
        ))

    # Turn 6: Unrelated user B message (group noise)
    rows.append(_user_row(
        row_id=_row(15),
        timestamp=T[14],
        body_text=BODY[14],
        platform_user_id=USER_B_PLATFORM_USER_ID,
        global_user_id=USER_B_GLOBAL_USER_ID,
        display_name=USER_B_DISPLAY_NAME,
        trace_id='trace_user_b_noise',
    ))

    # Turn 7: User A confirms 'ok, massaging houjing now'
    rows.append(_user_row(
        row_id=_row(16),
        timestamp=T[15],
        body_text=BODY[15],
        platform_user_id=USER_A_PLATFORM_USER_ID,
        global_user_id=USER_A_GLOBAL_USER_ID,
        display_name=USER_A_DISPLAY_NAME,
        trace_id='trace_user_a_confirm',
        addressed_to=[BOT_GLOBAL_USER_ID],
        mentions=[{
            'platform_user_id': BOT_PLATFORM_USER_ID,
            'global_user_id': BOT_GLOBAL_USER_ID,
            'display_name': BOT_DISPLAY_NAME,
            'entity_kind': 'bot',
        }],
        reply_context={
            'reply_to_message_id': _row(13),
            'reply_to_platform_user_id': BOT_PLATFORM_USER_ID,
            'reply_to_display_name': BOT_DISPLAY_NAME,
            'reply_excerpt': BODY[12],
        },
    ))

    # Turn 8: Asuna accepts (5 fragments)
    for i in range(5):
        rows.append(_assistant_row(
            row_id=_row(17 + i),
            timestamp=T[16 + i],
            body_text=BODY[16 + i],
            trace_id=TRACE_4,
            logical_message_index=i,
            addressed_to=USER_A_GLOBAL_USER_ID,
        ))

    # Turn 9: User A performs massage action
    rows.append(_user_row(
        row_id=_row(22),
        timestamp=T[21],
        body_text=BODY[21],
        platform_user_id=USER_A_PLATFORM_USER_ID,
        global_user_id=USER_A_GLOBAL_USER_ID,
        display_name=USER_A_DISPLAY_NAME,
        trace_id='trace_user_a_massage',
        addressed_to=[BOT_GLOBAL_USER_ID],
        mentions=[{
            'platform_user_id': BOT_PLATFORM_USER_ID,
            'global_user_id': BOT_GLOBAL_USER_ID,
            'display_name': BOT_DISPLAY_NAME,
            'entity_kind': 'bot',
        }],
        reply_context={
            'reply_to_message_id': _row(21),
            'reply_to_platform_user_id': BOT_PLATFORM_USER_ID,
            'reply_to_display_name': BOT_DISPLAY_NAME,
            'reply_excerpt': BODY[20],
        },
    ))

    # Turn 10: Asuna evaluates massage (7 fragments)
    for i in range(7):
        rows.append(_assistant_row(
            row_id=_row(23 + i),
            timestamp=T[22 + i],
            body_text=BODY[22 + i],
            trace_id=TRACE_5,
            logical_message_index=i,
            addressed_to=USER_A_GLOBAL_USER_ID,
        ))

    return rows


# Convenient aliases for tests.

ALL_ROW_IDS = [_row(i) for i in range(1, 30)]

PARTICIPANT_ROW_IDS = [
    r for r in ALL_ROW_IDS
    if r != _row(15)  # Exclude user B noise
    and r not in [_row(i) for i in range(1, 5)]  # Exclude Turn 1 (addressed to user B)
]

AMBIENT_ROW_IDS = ALL_ROW_IDS[:]

CURRENT_TURN_TIMESTAMP = '2026-07-28T09:30:00+00:00'

# Expected logical turn count after grouping:
# Participant lane: User A turns (2,4,7,9) + Asuna turns addressed to A (3,5,8,10) = 8
# Ambient lane: all 10 turns (including user B noise and Asuna->B turn 1)
EXPECTED_PARTICIPANT_LOGICAL_TURN_COUNT = 8
EXPECTED_AMBIENT_LOGICAL_TURN_COUNT = 10
