"""In-memory index of scheduled tasks for deduplication and restart rebuild."""

from __future__ import annotations

import json

from kazusa_ai_chatbot.db import get_db
from kazusa_ai_chatbot.dispatcher.task import Task


class PendingTaskIndex:
    """Track scheduled tasks in memory for cheap duplicate detection."""

    def __init__(self) -> None:
        self._by_event_id: dict[str, Task] = {}
        self._by_signature: dict[str, str] = {}

    @staticmethod
    def signature_for(task: Task) -> str:
        """Build a stable deduplication signature for one task.

        Args:
            task: Validated task to fingerprint.

        Returns:
            Stable JSON-backed signature string.
        """

        payload = {
            "tool": task.tool,
            "args": task.args,
            "execute_at": task.execute_at.isoformat(),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def add(self, event_id: str, task: Task) -> None:
        """Add one persisted task to the in-memory index.

        Args:
            event_id: Scheduler event identifier.
            task: Validated task represented by that event.
        """

        signature = self.signature_for(task)
        self._by_event_id[event_id] = task
        self._by_signature[signature] = event_id

    def remove(self, event_id: str) -> None:
        """Remove one task from the in-memory index if present.

        Args:
            event_id: Scheduler event identifier to remove.
        """

        task = self._by_event_id.pop(event_id, None)
        if task is None:
            return
        signature = self.signature_for(task)
        self._by_signature.pop(signature, None)

    def contains(self, task: Task) -> bool:
        """Return whether an equivalent task is already pending."""

        return self.signature_for(task) in self._by_signature

    def find_by_target(self, tool: str, args: dict) -> list[tuple[str, Task]]:
        """Return pending tasks matching one tool name and exact args dict.

        Args:
            tool: Tool name to match.
            args: Exact argument mapping to match.

        Returns:
            Matching ``(event_id, task)`` pairs.
        """

        return [
            (event_id, task)
            for event_id, task in self._by_event_id.items()
            if task.tool == tool and task.args == args
        ]

    async def rebuild_from_db(self) -> int:
        """Rebuild the in-memory index from pending scheduler documents.

        Returns:
            Number of pending tasks loaded into the index.
        """

        db = await get_db()
        cursor = db.scheduled_events.find({"status": "pending"})
        self._by_event_id.clear()
        self._by_signature.clear()

        count = 0
        async for doc in cursor:
            if "tool" not in doc or "execute_at" not in doc:
                continue
            task = Task.from_scheduler_doc(doc)
            self.add(doc["event_id"], task)
            count += 1
        return count
