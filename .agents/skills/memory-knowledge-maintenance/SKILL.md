---
name: memory-knowledge-maintenance
description: Maintain Kazusa's curated shared/world/common-sense memory knowledge base. Use this skill whenever the user asks to add, collect, edit, search, remove, deduplicate, sync, upload, or inspect world knowledge, common sense, local slang, character-world facts, or durable curated entries for the MongoDB memory collection. Prefer the local JSONL source and maintenance CLI over ad hoc MongoDB edits.
---

# Memory Knowledge Maintenance

Use this skill to maintain the curated shared knowledge that RAG2 reaches
through `Memory-search` and the persistent-memory helper agents.

The design is intentionally simple:

```text
personalities/knowledge/memory_seed.jsonl  -> source of truth
MongoDB memory collection    -> indexed runtime copy
RAG2 Memory-search           -> retrieval interface
```

Do not introduce extra categories or dispatcher prefixes for common sense,
world facts, or domain notes. If the most relevant memory row is retrieved, the
existing persistent-memory search path can use it.

## Files And Commands

Source file:

```powershell
personalities\knowledge\memory_seed.jsonl
```

Maintenance CLI:

```powershell
venv\Scripts\python.exe -m scripts.manage_memory_knowledge <command>
```

Useful commands:

```powershell
venv\Scripts\python.exe -m scripts.manage_memory_knowledge validate
venv\Scripts\python.exe -m scripts.manage_memory_knowledge list
venv\Scripts\python.exe -m scripts.manage_memory_knowledge search 洗车
venv\Scripts\python.exe -m scripts.manage_memory_knowledge add --name "UCCU含义" --content "UCCU 是中文网络语“你看看你”的谐音/缩写，常用于调侃对方当前状态或行为。"
venv\Scripts\python.exe -m scripts.manage_memory_knowledge remove --name "UCCU含义"
venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync
venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync --apply
```

Use `sync` without `--apply` first. It is a dry run. Use `--apply` only after
the local JSONL validates and the planned counters look right.

To make MongoDB match the local curated global source and remove unmanaged
global rows:

```powershell
venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync --apply --prune-unmanaged-global
```

## Entry Format

Each JSONL line is one MongoDB `memory` row:

```json
{"memory_name":"洗车难题","content":"洗车难题是一个隐含约束推理题：如果问题是在普通情境下问“去洗车店洗车是走过去还是开车去”，默认应开车去，因为洗车服务对象是车，车必须到店。步行只有在额外前提成立时才合理，例如店家提供上门取送、代驾或用户只是先去查看/预约。","source_global_user_id":"","memory_type":"fact","source_kind":"external_imported","confidence_note":"人工整理的中文世界知识，用于覆盖该题的常见错误回答。","status":"active","expiry_timestamp":null}
```

Rules:

- Keep `source_global_user_id` empty. This file is for global/shared knowledge,
  not user-scoped profile memory.
- Use `memory_type="fact"` by default.
- Use `memory_type="defense_rule"` only for durable behavioral/reliability
  guardrails.
- Use `source_kind="seeded_manual"` for hand-curated common sense and local
  knowledge.
- Use `source_kind="external_imported"` for entries summarized from external
  references.
- Keep one entry to one useful idea. Concise entries retrieve better.
- Use stable, human-readable `memory_name` values. The de-duplication key is
  `(memory_name, source_global_user_id)`.

## What To Add

Prioritize high-leverage entries:

- Observed weak-LLM failures and their corrected rule.
- Common-sense hidden constraints and everyday causality.
- Local slang, memes, abbreviations, and group-specific terms.
- Stable character-world facts not better served by `user_profile_agent`.
- Recurring technical/domain notes from the community.
- Reliability guardrails such as "current prices/laws/version facts require
  live search."

Avoid:

- User-specific facts, preferences, health, impressions, commitments, or
  relationship information. Those belong in `user_profile_memories`.
- Volatile facts unless `expiry_timestamp` is set.
- Long articles or broad encyclopedia dumps.
- Duplicate paraphrases of the same idea.

## Workflow

1. Search before adding:
   ```powershell
   venv\Scripts\python.exe -m scripts.manage_memory_knowledge search <term>
   ```
2. Add or replace the local row:
   ```powershell
   venv\Scripts\python.exe -m scripts.manage_memory_knowledge add --name "<name>" --content "<content>"
   ```
3. Validate:
   ```powershell
   venv\Scripts\python.exe -m scripts.manage_memory_knowledge validate
   ```
4. Dry-run sync:
   ```powershell
   venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync
   ```
5. Apply sync:
   ```powershell
   venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync --apply
   ```

When the user asks to make the database match the local curated file, use:

```powershell
venv\Scripts\python.exe -m scripts.manage_memory_knowledge sync --apply --prune-unmanaged-global
```

Before destructive DB cleanup, export a backup with:

```powershell
venv\Scripts\python.exe -m scripts.export_memory --limit 1000 --output test_artifacts/memory_before_<short_task_name>.json
```
