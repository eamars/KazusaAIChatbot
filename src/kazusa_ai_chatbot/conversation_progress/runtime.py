"""Runtime implementation behind the public conversation-progress facade."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from kazusa_ai_chatbot.conversation_progress import cache
from kazusa_ai_chatbot.conversation_progress import projection
from kazusa_ai_chatbot.conversation_progress import recorder
from kazusa_ai_chatbot.conversation_progress import repository
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressLoadResult,
    ConversationProgressRecordInput,
    ConversationProgressRecordResult,
    ConversationProgressScope,
)
from kazusa_ai_chatbot.utils import log_preview

logger = logging.getLogger(__name__)

RecorderCallable = Callable[[ConversationProgressRecordInput], Awaitable[dict]]


class ConversationProgressRuntime:
    """Dependency-injectable runtime for conversation progress.

    Args:
        recorder_callable: Async function that returns validated recorder output.
    """

    def __init__(self, recorder_callable: RecorderCallable = recorder.record_with_llm):
        self._recorder_callable = recorder_callable

    async def load_progress_context(
        self,
        *,
        scope: ConversationProgressScope,
        current_timestamp: str,
    ) -> ConversationProgressLoadResult:
        """Load, select, and project episode state for one responsive turn.

        Args:
            scope: Platform/channel/user scope.
            current_timestamp: ISO timestamp used for age-hint projection.

        Returns:
            Full load result with source metadata.
        """

        db_document = await repository.load_episode_state(scope=scope)
        selected_document, used_cache = cache.select_latest_document(
            scope=scope,
            db_document=db_document,
        )
        source = "cache" if used_cache else "db"
        if selected_document is None:
            source = "empty"
        prompt_doc = projection.project_prompt_doc(
            document=selected_document,
            current_timestamp=current_timestamp,
        )
        logger.debug(
            f'Conversation progress load detail: platform={scope.platform} '
            f'channel={scope.platform_channel_id or "<dm>"} '
            f"user={scope.global_user_id} source={source} "
            f"episode_present={selected_document is not None} "
            f"progress={log_preview(prompt_doc)}"
        )
        return_value = {
            "episode_state": selected_document,
            "conversation_progress": prompt_doc,
            "source": source,
        }
        return return_value

    async def record_turn_progress(
        self,
        *,
        record_input: ConversationProgressRecordInput,
    ) -> ConversationProgressRecordResult:
        """Record one completed responsive turn and update the cache.

        Args:
            record_input: Current turn payload for the recorder and persistence.

        Returns:
            Background telemetry for the write.
        """

        scope = record_input["scope"]
        recorder_output = await self._recorder_callable(record_input)
        logger.debug(
            f"Conversation progress recorder output detail: "
            f"platform={scope.platform} "
            f'channel={scope.platform_channel_id or "<dm>"} '
            f"user={scope.global_user_id} "
            f"output={log_preview(recorder_output)}"
        )
        document = repository.build_episode_state_doc(
            scope=scope,
            timestamp=record_input["timestamp"],
            prior_episode_state=record_input["prior_episode_state"],
            recorder_output=recorder_output,
            last_user_input=record_input["decontexualized_input"],
        )
        written = await repository.upsert_episode_state_guarded(document=document)
        cache_updated = False
        if written:
            cache.store_completed_document(
                scope=scope,
                document=document,
            )
            cache_updated = True
        logger.info(
            f"Conversation progress write: platform={scope.platform} "
            f'channel={scope.platform_channel_id or "<dm>"} '
            f"user={scope.global_user_id} written={written} "
            f'cache_updated={cache_updated} turn_count={document["turn_count"]} '
            f'continuity={document["continuity"]} status={document["status"]}'
        )
        logger.debug(
            f"Conversation progress write detail: platform={scope.platform} "
            f'channel={scope.platform_channel_id or "<dm>"} '
            f"user={scope.global_user_id} document={log_preview(document)}"
        )
        return_value = {
            "written": written,
            "turn_count": int(document["turn_count"]),
            "continuity": document["continuity"],
            "status": document["status"],
            "cache_updated": cache_updated,
        }
        return return_value


_default_runtime = ConversationProgressRuntime()


async def load_progress_context(
    *,
    scope: ConversationProgressScope,
    current_timestamp: str,
) -> ConversationProgressLoadResult:
    """Load and project progress state for one responsive turn."""

    return_value = await _default_runtime.load_progress_context(
        scope=scope,
        current_timestamp=current_timestamp,
    )
    return return_value


async def record_turn_progress(
    *,
    record_input: ConversationProgressRecordInput,
) -> ConversationProgressRecordResult:
    """Record progress after final dialog generation."""

    return_value = await _default_runtime.record_turn_progress(record_input=record_input)
    return return_value
