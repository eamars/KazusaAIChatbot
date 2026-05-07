"""Offline migration for durable memory perspective wording."""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from scripts._db_export import load_project_env
from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.db import (
    close_db,
    get_character_profile,
    get_character_state,
    update_last_relationship_insight,
    update_user_memory_unit_semantics,
    upsert_character_self_image,
    upsert_character_state,
)
from kazusa_ai_chatbot.db.script_operations import (
    find_persistent_memory_without_embedding,
    scan_active_user_memory_units_for_perspective,
    scan_persistent_memory_for_perspective,
    scan_user_profiles_for_perspective,
)
from kazusa_ai_chatbot.memory_evolution import supersede_memory_unit
from kazusa_ai_chatbot.memory_evolution.identity import deterministic_memory_unit_id
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output


MIGRATION_PROMPT_VERSION = 'memory_writer_perspective_migration_v6'
DEFAULT_OUTPUT = Path('test_artifacts') / 'memory_writer_perspective_dry_run.json'

SCOPE_USER_MEMORY_UNITS = 'user_memory_units'
SCOPE_USER_PROFILES = 'user_profiles'
SCOPE_CHARACTER_STATE = 'character_state'
SCOPE_MEMORY = 'memory'

MIGRATION_REWRITE_SYSTEM_PROMPT = '''\
你负责离线迁移已经保存、以后还会进入 prompt 的 durable memory 文本。

# 任务
输入只包含一个文档的 fields。请在不改变原意的前提下，把 fields 里的可长期保存文本改成当前记忆视角。
不要新增字段，不要删除字段，不要改变 JSON key，不要改变事件事实、用户偏好、承诺、关系含义或证据指向。
如果某个字段已经符合当前契约，原样返回该字段。
如果你无法确信某个字段里的“我”“自己”“她”“对方”、短名或英文名到底指向谁，返回原字段，并把 status 写成 "blocked"；不要猜。

# 读取顺序
1. 先读 collection、document_id 和字段名，再读字段内容。字段名是判断语义来源的重要证据。
2. fact 通常记录事件、用户事实、用户偏好、承诺或对话事实；不要把用户说的“我”改成 {character_name}。
3. subjective_appraisal、relationship_signal、reflection_summary、self_image 和自我引导类 content 通常记录 {character_name} 对互动的感受、判断或未来应对；除非原文明确说这是用户的感受，否则不要把这类字段里的感受误写成用户的感受。
4. memory_name 是标题，保持短而事实性；只在标题本来必须区分主体时才写名称。
5. 只改写为满足视角、语言和名称契约所必需的文本；不要做风格润色或补充剧情。

# 语言政策
- JSON key 必须保持英文。
- 新生成的自由文本字段使用简体中文。
- 专有名词、项目代号、用户显示名、ID、URL、英文标题和必要原文可以保留原语言；但指向 {character_name} 的短名、昵称、旧写法或亲昵称呼不属于必须保留的原文。
- 如果引用原话里包含指向 {character_name} 的短名或昵称，不要照抄引号内容；改成事实性转述，保留用户曾用某称呼指向 {character_name}、请求什么、表达了什么即可。
- 不要添加双语复写、括号解释、Markdown 反引号或代码样式。

# 记忆视角契约
- 记忆文本采用第三人称视角。
- 可写入记忆文本的唯一名称是 {character_name}。当文本需要指向 {character_name} 时，只能使用完整字符串 {character_name}。
- 规范名称是一个不可拆分的完整字符串：{character_name}；需要命名时必须逐字复制完整字符串，不要缩写、截断、翻译、改写、只写括号前部分、只写括号内部分，或凭记忆重拼。
- 不要用“我”指代 {character_name}。输入里的“我”必须先按原说话人或字段语义判断，用户说的“我”写成“用户”“对方”或“用户自己”。
- 不要把泛称、机器标签、speaker label、assistant、短名、别名、显示名或旧写法写成 {character_name} 的替代名称。
- 不需要消歧时优先省略主语；需要消歧时在该字段中写一次完整的 {character_name}，后文用无主语句式承接。
- 如果无法完整复制 {character_name}，宁可省略主语或 blocked，不要输出近似拼写。
- 输出前检查：字段文本中不应出现 Markdown 反引号；移除完整字符串 {character_name} 后，不应剩下指向该名称的短名、片段、罗马字片段、旧别名或机器标签。

# 常见改写
- 用户说“我决定把项目代号叫 Atlas”：写成“用户决定把自己的项目代号定为 Atlas”。
- 用户说“这不是给短名改名”：写成“这不是指向 {character_name} 的改名”。
- 用户发送“短称不要生气”：写成“用户用亲昵称呼请求 {character_name} 不要生气”，不要保留引号里的短称。
- subjective_appraisal 中的“I feel flustered”如果不是用户明说的感受，写成“{character_name} 感到慌乱”或省略主语的第三人称句。
- 如果原文只剩一个无法判定归属的“I should/我应该”，保持原文并返回 blocked。

# 输出格式
只输出 JSON：
{{
  "status": "ready|unchanged|blocked",
  "fields": {{
    "field_name": "rewritten value with the same JSON shape as input"
  }},
  "notes": ["short migration note"]
}}

status 含义：
- ready：至少一个字段被改写，且你确信所有改写都保留了原意。
- unchanged：所有字段已经符合契约或不需要改写。
- blocked：至少一个字段无法在不猜测主体的情况下安全改写。blocked 时 fields 必须返回输入原文。
'''

_migration_rewrite_llm = get_llm(
    temperature=0.1,
    top_p=0.8,
    model=CONSOLIDATION_LLM_MODEL,
    base_url=CONSOLIDATION_LLM_BASE_URL,
    api_key=CONSOLIDATION_LLM_API_KEY,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description='Offline rewrite of prompt-facing memory perspective text.',
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--apply', action='store_true')
    parser.add_argument('--input', type=Path, help='Reviewed dry-run report.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--scan-active-user-memory-units', action='store_true')
    parser.add_argument('--scan-user-profiles', action='store_true')
    parser.add_argument('--scan-character-state', action='store_true')
    parser.add_argument('--scan-persistent-memory', action='store_true')
    return parser


async def build_dry_run_report(
    *,
    scopes: set[str],
    limit: int,
) -> dict[str, Any]:
    """Build a dry-run report by rewriting scoped fields through the LLM."""

    character_profile = await get_character_profile()
    character_name = str(character_profile.get('name') or '').strip()
    if not character_name:
        raise ValueError('character_profile.name is required')

    records = await scan_memory_writer_records(scopes=scopes, limit=limit)
    report_records = []
    for record in records:
        proposal = await build_record_proposal(
            record,
            character_name=character_name,
        )
        report_records.append(proposal)

    ready_count = sum(
        1 for record in report_records
        if record.get('status') == 'ready'
    )
    report = {
        'dry_run': True,
        'prompt_version': MIGRATION_PROMPT_VERSION,
        'scopes': sorted(scopes),
        'character_name_source': 'character_profile.name',
        'character_name': character_name,
        'records_seen': len(report_records),
        'records_with_proposed_changes': ready_count,
        'records': report_records,
    }
    return report


async def scan_memory_writer_records(
    *,
    scopes: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Return scoped prompt-facing memory records."""

    records: list[dict[str, Any]] = []
    remaining = max(0, limit)
    if SCOPE_USER_MEMORY_UNITS in scopes and remaining:
        batch = await scan_active_user_memory_units(limit=remaining)
        records.extend(batch)
        remaining = max(0, limit - len(records))
    if SCOPE_USER_PROFILES in scopes and remaining:
        batch = await scan_user_profiles(limit=remaining)
        records.extend(batch)
        remaining = max(0, limit - len(records))
    if SCOPE_CHARACTER_STATE in scopes and remaining:
        batch = await scan_character_state(limit=remaining)
        records.extend(batch)
        remaining = max(0, limit - len(records))
    if SCOPE_MEMORY in scopes and remaining:
        batch = await scan_persistent_memory(limit=remaining)
        records.extend(batch)
    return records[:limit]


async def scan_active_user_memory_units(*, limit: int) -> list[dict[str, Any]]:
    """Scan active user-memory-unit semantic fields."""

    documents = await scan_active_user_memory_units_for_perspective(
        limit=limit,
    )
    records = []
    for document in documents:
        before = _field_view(
            document,
            ('fact', 'subjective_appraisal', 'relationship_signal'),
        )
        if not before:
            continue
        records.append(_scan_record(
            collection=SCOPE_USER_MEMORY_UNITS,
            document_key='unit_id',
            document_id=str(document.get('unit_id') or document.get('_id')),
            before=before,
        ))
    return records


async def scan_user_profiles(*, limit: int) -> list[dict[str, Any]]:
    """Scan prompt-facing relationship insight fields."""

    documents = await scan_user_profiles_for_perspective(
        limit=limit,
    )
    records = []
    for document in documents:
        before = _field_view(document, ('last_relationship_insight',))
        if not before:
            continue
        records.append(_scan_record(
            collection=SCOPE_USER_PROFILES,
            document_key='global_user_id',
            document_id=str(document.get('global_user_id') or ''),
            before=before,
        ))
    return records


async def scan_character_state(*, limit: int) -> list[dict[str, Any]]:
    """Scan singleton character-state prompt-facing memory fields."""

    if limit <= 0:
        return []
    state = await get_character_state()
    if not state:
        return []
    before = _field_view(state, ('reflection_summary', 'self_image'))
    if not before:
        return []
    return [
        _scan_record(
            collection=SCOPE_CHARACTER_STATE,
            document_key='_id',
            document_id='global',
            before=before,
        ),
    ]


async def scan_persistent_memory(*, limit: int) -> list[dict[str, Any]]:
    """Scan active runtime-generated shared memory rows."""

    documents = await scan_persistent_memory_for_perspective(
        limit=limit,
    )
    records = []
    for document in documents:
        before = _field_view(document, ('memory_name', 'content'))
        if not before:
            continue
        records.append(_scan_record(
            collection=SCOPE_MEMORY,
            document_key='memory_unit_id',
            document_id=str(document.get('memory_unit_id') or ''),
            before=before,
        ))
    return records


async def build_record_proposal(
    record: dict[str, Any],
    *,
    character_name: str,
) -> dict[str, Any]:
    """Rewrite one scan record and return a reviewable proposal row."""

    before = deepcopy(record['before'])
    try:
        parsed = await run_migration_rewrite_llm(
            record,
            character_name=character_name,
        )
        fields = parsed.get('fields')
        if not isinstance(fields, dict):
            return _proposal(record, before, before, 'blocked', ['missing fields'])
        after = _same_key_field_view(fields, before)
        parsed_status = parsed.get('status')
        if parsed_status == 'blocked':
            status = 'blocked'
            after = before
        elif parsed_status == 'unchanged' and after != before:
            return _proposal(
                record,
                before,
                before,
                'blocked',
                ['status unchanged but fields changed'],
            )
        elif after == before:
            status = 'unchanged'
        else:
            status = 'ready'
        notes = parsed.get('notes')
        if not isinstance(notes, list):
            notes = []
        return _proposal(record, before, after, status, notes)
    except Exception as exc:
        return _proposal(record, before, before, 'blocked', [str(exc)])


async def run_migration_rewrite_llm(
    record: dict[str, Any],
    *,
    character_name: str,
) -> dict[str, Any]:
    """Rewrite one record through the offline migration prompt."""

    system_prompt = MIGRATION_REWRITE_SYSTEM_PROMPT.format(
        character_name=character_name,
    )
    payload = {
        'collection': record['collection'],
        'document_id': record['document_id'],
        'fields': record['before'],
    }
    response = await _migration_rewrite_llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ])
    parsed = parse_llm_json_output(str(response.content))
    if not isinstance(parsed, dict):
        return {'fields': record['before'], 'notes': ['non-object output']}
    return parsed


async def apply_report(report: dict[str, Any]) -> dict[str, Any]:
    """Apply reviewed ready rows from a dry-run report."""

    if not report.get('dry_run'):
        raise ValueError('input report must be a dry-run report')
    applied_records = []
    skipped_records = []
    blocked_records = []
    for record in report.get('records', []):
        record_status = record.get('status')
        if record_status == 'unchanged':
            skipped_records.append({
                'collection': record.get('collection'),
                'document_id': record.get('document_id'),
                'status': record_status,
            })
            continue
        if record_status != 'ready':
            blocked_records.append({
                'collection': record.get('collection'),
                'document_id': record.get('document_id'),
                'status': record_status,
            })
            continue
        try:
            await apply_record(record)
            applied_records.append({
                'collection': record.get('collection'),
                'document_id': record.get('document_id'),
                'status': 'applied',
            })
        except Exception as exc:
            blocked_records.append({
                'collection': record.get('collection'),
                'document_id': record.get('document_id'),
                'status': 'blocked',
                'notes': [str(exc)],
            })

    return {
        'dry_run': False,
        'source_prompt_version': report.get('prompt_version'),
        'applied_count': len(applied_records),
        'skipped_count': len(skipped_records),
        'blocked_count': len(blocked_records),
        'applied_records': applied_records,
        'skipped_records': skipped_records,
        'blocked_records': blocked_records,
    }


async def apply_record(record: dict[str, Any]) -> None:
    """Persist one reviewed ready row through the owning helper path."""

    collection = record['collection']
    document_id = str(record['document_id'])
    after = record['after']
    if collection == SCOPE_USER_MEMORY_UNITS:
        await update_user_memory_unit_semantics(
            document_id,
            after,
            increment_count=False,
        )
        return
    if collection == SCOPE_USER_PROFILES:
        insight = str(after.get('last_relationship_insight') or '').strip()
        if insight:
            await update_last_relationship_insight(document_id, insight)
        return
    if collection == SCOPE_CHARACTER_STATE:
        await _apply_character_state(after)
        return
    if collection == SCOPE_MEMORY:
        await _apply_persistent_memory(document_id, after)
        return
    raise ValueError(f'unsupported collection: {collection!r}')


async def _apply_character_state(after: dict[str, Any]) -> None:
    """Apply singleton character-state fields through existing helpers."""

    state = await get_character_state()
    timestamp = datetime.now(timezone.utc).isoformat()
    if 'reflection_summary' in after:
        await upsert_character_state(
            '',
            '',
            str(after.get('reflection_summary') or ''),
            timestamp,
        )
    if isinstance(after.get('self_image'), dict):
        await upsert_character_self_image(after['self_image'])


async def _apply_persistent_memory(
    memory_unit_id: str,
    after: dict[str, Any],
) -> None:
    """Supersede one runtime shared-memory row with rewritten prose."""

    existing = await find_persistent_memory_without_embedding(memory_unit_id)
    if not existing:
        raise ValueError(f'memory row not found: {memory_unit_id!r}')
    replacement = dict(existing)
    replacement.pop('_id', None)
    replacement['memory_name'] = str(after.get('memory_name') or '').strip()
    replacement['content'] = str(after.get('content') or '').strip()
    replacement['memory_unit_id'] = deterministic_memory_unit_id(
        'memory-perspective-migration',
        [
            memory_unit_id,
            replacement['memory_name'],
            replacement['content'],
        ],
    )
    await supersede_memory_unit(
        active_unit_id=memory_unit_id,
        replacement=replacement,
    )


def _field_view(
    document: dict[str, Any],
    field_names: tuple[str, ...],
) -> dict[str, Any]:
    """Copy non-empty prompt-facing fields from a document."""

    result = {}
    for field_name in field_names:
        value = document.get(field_name)
        if value in (None, ''):
            continue
        result[field_name] = deepcopy(value)
    return result


def _same_key_field_view(
    fields: dict[str, Any],
    before: dict[str, Any],
) -> dict[str, Any]:
    """Copy LLM fields using exactly the original field keys."""

    result = {}
    for field_name in before:
        if field_name not in fields:
            result[field_name] = deepcopy(before[field_name])
        else:
            result[field_name] = deepcopy(fields[field_name])
    return result


def _scan_record(
    *,
    collection: str,
    document_key: str,
    document_id: str,
    before: dict[str, Any],
) -> dict[str, Any]:
    """Build a scan record for dry-run review."""

    return {
        'collection': collection,
        'document_key': document_key,
        'document_id': document_id,
        'before': deepcopy(before),
    }


def _proposal(
    record: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    status: str,
    notes: list[Any],
) -> dict[str, Any]:
    """Build one dry-run proposal row."""

    return {
        'collection': record['collection'],
        'document_key': record['document_key'],
        'document_id': record['document_id'],
        'status': status,
        'before': before,
        'after': after,
        'notes': [str(note) for note in notes],
    }


def selected_scopes(args: argparse.Namespace) -> set[str]:
    """Return scan scopes selected by CLI flags."""

    scopes = set()
    if args.scan_active_user_memory_units:
        scopes.add(SCOPE_USER_MEMORY_UNITS)
    if args.scan_user_profiles:
        scopes.add(SCOPE_USER_PROFILES)
    if args.scan_character_state:
        scopes.add(SCOPE_CHARACTER_STATE)
    if args.scan_persistent_memory:
        scopes.add(SCOPE_MEMORY)
    return scopes


def write_report(output_path: Path, report: dict[str, Any]) -> None:
    """Write a JSON report to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )


def read_report(input_path: Path) -> dict[str, Any]:
    """Read a JSON dry-run report from disk."""

    return json.loads(input_path.read_text(encoding='utf-8'))


async def async_main() -> None:
    """Run the migration CLI."""

    parser = build_parser()
    args = parser.parse_args()
    load_project_env()
    try:
        if args.dry_run:
            scopes = selected_scopes(args)
            if not scopes:
                parser.error('at least one --scan-* flag is required')
            report = await build_dry_run_report(scopes=scopes, limit=args.limit)
            write_report(args.output, report)
            print(
                f'wrote dry-run report with {report["records_seen"]} record(s) '
                f'to {args.output}',
            )
            return

        if not args.input:
            parser.error('--apply requires --input')
        report = await apply_report(read_report(args.input))
        write_report(args.output, report)
        print(
            f'wrote apply report with {report["applied_count"]} applied row(s) '
            f'to {args.output}',
        )
    finally:
        await close_db()


def main() -> None:
    """Console entry point."""

    asyncio.run(async_main())


if __name__ == '__main__':
    main()
