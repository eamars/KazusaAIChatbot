"""Guarded, lineage-filtered Mongo diagnostics for isolated DSH scenarios."""

from __future__ import annotations

import re
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient

DATABASE_PATTERN = re.compile(r'_test_kazusa_dsh_behavior_[0-9a-f]{32}')
MAX_DIAGNOSTIC_ROWS = 2000


class GuardedDshDiagnostics:
    """Own a separate diagnostic client for one explicitly guarded database."""

    def __init__(self, uri: str, database_name: str, expected_name: str) -> None:
        if database_name != expected_name or not DATABASE_PATTERN.fullmatch(database_name):
            raise ValueError('diagnostic database does not match the exact test guard')
        self.database_name = database_name
        self.client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)

    async def snapshot(self, channel_id: str, interaction_ids: list[str]) -> dict[str, Any]:
        """Return bounded records joined from channel and operation lineage."""

        database = self.client[self.database_name]
        bindings = await database.dsh_task_bindings.find(
            {'source_scope.channel_id': channel_id}, {'_id': 0},
        ).to_list(length=MAX_DIAGNOSTIC_ROWS)
        task_ids = [row['current_accepted_task_id'] for row in bindings if row['current_accepted_task_id']]
        job_ids = [row['current_background_work_job_id'] for row in bindings if row['current_background_work_job_id']]
        accepted = await database.accepted_tasks.find(
            {'accepted_task_id': {'$in': task_ids}}, {'_id': 0},
        ).to_list(length=MAX_DIAGNOSTIC_ROWS)
        jobs = await database.background_work_jobs.find(
            {'$or': [{'job_id': {'$in': job_ids}}, {'accepted_task_id': {'$in': task_ids}}]}, {'_id': 0},
        ).to_list(length=MAX_DIAGNOSTIC_ROWS)
        runs = await database.llm_trace_runs.find(
            {'$or': [{'platform_channel_id': channel_id}, {'source_background_work_job_id': {'$in': job_ids}}]}, {'_id': 0},
        ).to_list(length=MAX_DIAGNOSTIC_ROWS)
        trace_ids = [row['trace_id'] for row in runs] + interaction_ids
        steps = await database.llm_trace_steps.find(
            {'trace_id': {'$in': trace_ids}}, {'_id': 0},
        ).sort('sequence', 1).to_list(length=MAX_DIAGNOSTIC_ROWS)
        interactions = await database.dsh_interaction_store.find(
            {'interaction_id': {'$in': interaction_ids}}, {'_id': 0},
        ).to_list(length=MAX_DIAGNOSTIC_ROWS)
        events = await database.event_log_events.find(
            {'$or': [
                {'correlation_id': {'$in': trace_ids}},
                {'llm_trace_id': {'$in': trace_ids}},
                {'platform_channel_id': channel_id},
            ]}, {'_id': 0},
        ).to_list(length=MAX_DIAGNOSTIC_ROWS)
        history = await database.conversation_history.find(
            {'platform_channel_id': channel_id}, {'_id': 0},
        ).to_list(length=MAX_DIAGNOSTIC_ROWS)
        snapshot = {
            'dsh_task_bindings': bindings, 'accepted_tasks': accepted,
            'background_work_jobs': jobs, 'dsh_interactions': interactions,
            'llm_trace_runs': runs, 'llm_trace_steps': steps,
            'event_log_events': events, 'conversation_history': history,
        }
        return snapshot

    async def drop(self) -> None:
        """Drop only the exact database validated before connection."""
        if not DATABASE_PATTERN.fullmatch(self.database_name):
            raise ValueError('diagnostic cleanup lost its database guard')
        await self.client.drop_database(self.database_name)

    def close(self) -> None:
        self.client.close()
