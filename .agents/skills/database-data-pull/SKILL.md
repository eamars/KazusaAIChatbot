---
name: database-data-pull
description: Pull read-only diagnostic data from the Kazusa MongoDB database into JSON files. Use this skill whenever the user asks to export, inspect, dump, pull, or retrieve chat history, user profiles, user image, persistent user memories, shared memory, character state, or arbitrary MongoDB collection rows from this repo. Prefer the bundled project scripts and .env settings over ad hoc database queries.
---

# Database Data Pull

Use this skill for read-only MongoDB extraction from the Kazusa chatbot repo. The goal is to produce a JSON artifact the user can inspect, attach to plans, or reuse in tests without mutating live data.

## Ground Rules

- Run commands from the repository root.
- Use the project virtual environment: `venv\Scripts\python.exe` on Windows.
- Let the scripts load database settings from `.env`; do not paste connection strings into commands.
- Prefer focused scripts under `src/scripts/` before writing one-off MongoDB snippets.
- Keep embeddings excluded unless the user explicitly asks for vectors.
- Put outputs under `test_artifacts/` unless the user gives a path.
- Treat exports as potentially sensitive. Do not quote large message contents back in chat unless the user asks.

## Script Map

Use the script that matches the requested data:

- Chat history: `python -m scripts.export_chat_history`
- Full user profile: `python -m scripts.export_user_profile`
- User image only: `python -m scripts.export_user_image`
- Persistent profile memories: `python -m scripts.export_user_memories`
- Shared/world memory collection: `python -m scripts.export_memory`
- Character profile/runtime state: `python -m scripts.export_character_state`
- Arbitrary collection rows: `python -m scripts.export_collection`

## Common Commands

Export the last four hours of QQ chat history for a channel:

```powershell
venv\Scripts\python.exe -m scripts.export_chat_history 673225019 --platform qq --hours 4
```

Export the latest 30 messages for a channel:

```powershell
venv\Scripts\python.exe -m scripts.export_chat_history 1082431481 --platform qq --limit 30
```

Export a user image by platform user id:

```powershell
venv\Scripts\python.exe -m scripts.export_user_image 3167827653 --platform qq
```

Export a full profile by global user id:

```powershell
venv\Scripts\python.exe -m scripts.export_user_profile 263c883d-aeff-4e0b-a758-6f69186ae8ec
```

Export prompt-facing persistent memory blocks:

```powershell
venv\Scripts\python.exe -m scripts.export_user_memories 263c883d-aeff-4e0b-a758-6f69186ae8ec
```

Export raw persistent memory rows:

```powershell
venv\Scripts\python.exe -m scripts.export_user_memories 3167827653 --platform qq --raw --limit 100
```

Export arbitrary collection rows with a JSON filter:

```powershell
venv\Scripts\python.exe -m scripts.export_collection conversation_history --filter "{\"platform_channel_id\":\"673225019\"}" --sort "{\"timestamp\":-1}" --limit 20
```

## Options To Reach For

- Use `--output path\to\file.json` when the user names a file.
- Use `--from-timestamp` and `--to-timestamp` for exact historical windows.
- Use `--include-embeddings` only when vectors are specifically needed.
- Use `--verbose` when connection or query behavior needs debugging.
- Use `--include-expired` only with `export_user_memories --raw` when the user asks for deleted or expired persistent memories too.

## Response Pattern

After running an export, tell the user:

- Which script ran.
- The output path.
- The number of records exported.
- The time window or filter used, when relevant.

Keep the response short and avoid dumping the exported JSON into chat.
