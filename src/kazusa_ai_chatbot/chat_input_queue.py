"""Bookkeeping for the process-local chat input queue."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

GROUP_COALESCE_MAX_GAP_SECONDS = 120.0


@dataclass
class QueuedChatItem:
    """One pending chat request managed by the input queue.

    Args:
        sequence: Monotonic brain-local arrival sequence.
        request: Original chat request payload.
        timestamp: Stable timestamp assigned at enqueue time.
        future: Response future awaited by the `/chat` endpoint.
        combined_content: Optional combined content for a collapsed turn.
        collapsed_items: Later queued items collapsed into this survivor.
    """

    sequence: int
    request: Any
    timestamp: str
    future: asyncio.Future[Any]
    combined_content: str | None = None
    collapsed_items: list[QueuedChatItem] = field(default_factory=list)


@dataclass
class DequeuedChatTurn:
    """One queue worker handoff batch.

    Args:
        next_item: Oldest surviving item to process, if any.
        dropped_items: Items pruned by group-noise policy.
        collapsed_items: Pairs of collapsed original item and survivor item.
    """

    next_item: QueuedChatItem | None
    dropped_items: list[QueuedChatItem]
    collapsed_items: list[tuple[QueuedChatItem, QueuedChatItem]]


class ChatInputQueue:
    """Bookkeep queued chat messages, pruning, and collapse decisions."""

    def __init__(self) -> None:
        """Initialize an empty process-local queue."""

        self._queue: deque[QueuedChatItem] = deque()
        self._condition: asyncio.Condition | None = None
        self._sequence = 0

    def _get_condition(self) -> asyncio.Condition:
        """Return the queue condition for the active event loop.

        Returns:
            The condition used to wake queue consumers.
        """

        if self._condition is None:
            self._condition = asyncio.Condition()
        return self._condition

    async def enqueue(self, request: Any) -> Any:
        """Add a request to the queue and wait for its response future.

        Args:
            request: Incoming chat request object.

        Returns:
            Response set by the service after processing, dropping, or collapse.
        """

        condition = self._get_condition()
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        timestamp = request.timestamp or datetime.now(timezone.utc).isoformat()

        async with condition:
            self._sequence += 1
            item = QueuedChatItem(
                sequence=self._sequence,
                request=request,
                timestamp=timestamp,
                future=future,
            )
            self._queue.append(item)
            condition.notify()

        response = await future
        return response

    async def wait_for_next(self) -> DequeuedChatTurn:
        """Wait for queued messages and return the next service handoff.

        Returns:
            Handoff batch containing dropped items, collapsed originals, and the
            next surviving item for the service to process.
        """

        condition = self._get_condition()
        async with condition:
            while not self._queue:
                await condition.wait()

            waiting_items = list(self._queue)
            private_survivors, private_collapsed = self._coalesce_private(
                waiting_items,
            )
            group_survivors, group_collapsed = self._coalesce_addressed_group(
                private_survivors,
            )
            survivors, dropped = self._prune(group_survivors)

            self._queue.clear()
            self._queue.extend(survivors)

            next_item: QueuedChatItem | None = None
            if self._queue:
                next_item = self._queue.popleft()

        return_value = DequeuedChatTurn(
            next_item=next_item,
            dropped_items=dropped,
            collapsed_items=private_collapsed + group_collapsed,
        )
        return return_value

    def complete(self, item: QueuedChatItem, response: Any) -> None:
        """Complete one queued item future if the caller is still waiting.

        Args:
            item: Queued chat item.
            response: Response to return through the item's future.

        Returns:
            None.
        """

        if not item.future.done():
            item.future.set_result(response)

    async def drain(self) -> list[QueuedChatItem]:
        """Remove and return all pending queued items.

        Returns:
            Pending items that had not yet been handed to the service.
        """

        condition = self._get_condition()
        async with condition:
            pending_items = list(self._queue)
            self._queue.clear()
        self._condition = None
        self._sequence = 0
        return pending_items

    def pending_count(self) -> int:
        """Return the number of currently queued waiting items.

        Returns:
            Queue length.
        """

        return_value = len(self._queue)
        return return_value

    def pop_left_for_test(self) -> QueuedChatItem:
        """Pop the oldest queued item for deterministic endpoint tests.

        Returns:
            Oldest queued item.
        """

        return_value = self._queue.popleft()
        return return_value

    def extend_for_test(self, items: list[QueuedChatItem]) -> None:
        """Append queued items for deterministic worker tests.

        Args:
            items: Items to append to the waiting queue.

        Returns:
            None.
        """

        self._queue.extend(items)

    async def notify_for_test(self) -> None:
        """Wake the queue consumer in tests.

        Returns:
            None.
        """

        condition = self._get_condition()
        async with condition:
            condition.notify()

    def reset_for_test(self) -> None:
        """Clear queue state for deterministic tests.

        Returns:
            None.
        """

        self._queue.clear()
        self._condition = None
        self._sequence = 0

    def is_tagged(self, item: QueuedChatItem) -> bool:
        """Return whether a queued request structurally mentioned the bot.

        Args:
            item: Queued chat item.

        Returns:
            True when the adapter marked the request as a bot mention.
        """

        return_value = item.request.mentioned_bot is True
        return return_value

    def is_bot_reply(self, item: QueuedChatItem) -> bool:
        """Return whether a queued request is an adapter-confirmed bot reply.

        Args:
            item: Queued chat item.

        Returns:
            True only when the adapter supplied `reply_to_current_bot=True`.
        """

        return_value = item.request.reply_context.reply_to_current_bot is True
        return return_value

    def is_private_message(self, item: QueuedChatItem) -> bool:
        """Return whether the item is a private-channel message.

        Args:
            item: Queued chat item.

        Returns:
            True when the request channel type is private.
        """

        return_value = item.request.channel_type == "private"
        return return_value

    def is_group_message(self, item: QueuedChatItem) -> bool:
        """Return whether the item is a group-channel message.

        Args:
            item: Queued chat item.

        Returns:
            True when the request channel type is group.
        """

        return_value = item.request.channel_type == "group"
        return return_value

    def coalesce_private(
        self,
        waiting_items: list[QueuedChatItem],
    ) -> tuple[list[QueuedChatItem], list[tuple[QueuedChatItem, QueuedChatItem]]]:
        """Collapse queued private follow-ups by private user scope.

        Args:
            waiting_items: Ordered list of queued items that are not processing.

        Returns:
            Pair of coalesced waiting items and collapsed original/survivor
            pairs.
        """

        return_value = self._coalesce_private(waiting_items)
        return return_value

    def coalesce_addressed_group(
        self,
        waiting_items: list[QueuedChatItem],
    ) -> tuple[list[QueuedChatItem], list[tuple[QueuedChatItem, QueuedChatItem]]]:
        """Collapse same-author group runs that start by addressing the bot.

        Args:
            waiting_items: Ordered list of queued items after private coalescing.

        Returns:
            Pair of coalesced waiting items and collapsed original/survivor
            pairs.
        """

        return_value = self._coalesce_addressed_group(waiting_items)
        return return_value

    def prune(
        self,
        waiting_items: list[QueuedChatItem],
    ) -> tuple[list[QueuedChatItem], list[QueuedChatItem]]:
        """Apply the global input-queue pruning policy to waiting items.

        Args:
            waiting_items: Ordered list of queued items that are not processing.

        Returns:
            Pair of surviving items and dropped items, both in arrival order.
        """

        return_value = self._prune(waiting_items)
        return return_value

    def _private_message_scope(self, item: QueuedChatItem) -> tuple[str, str, str]:
        """Return the private coalescing scope for an item.

        Args:
            item: Queued chat item.

        Returns:
            Tuple of platform, channel, and platform user identifiers.
        """

        return_value = (
            item.request.platform,
            item.request.platform_channel_id,
            item.request.platform_user_id,
        )
        return return_value

    def _append_collapsed_item(
        self,
        survivor: QueuedChatItem,
        collapsed_item: QueuedChatItem,
    ) -> None:
        """Attach one later item to a surviving collapsed turn.

        Args:
            survivor: Earliest queued item that remains in the queue.
            collapsed_item: Later item collapsed into the survivor.

        Returns:
            None.
        """

        if survivor.combined_content is None:
            survivor.combined_content = survivor.request.content
        survivor.combined_content = "\n".join([
            survivor.combined_content,
            collapsed_item.request.content,
        ])
        survivor.collapsed_items.append(collapsed_item)

    def _coalesce_private(
        self,
        waiting_items: list[QueuedChatItem],
    ) -> tuple[list[QueuedChatItem], list[tuple[QueuedChatItem, QueuedChatItem]]]:
        """Collapse queued private follow-ups by private user scope."""

        survivors: list[QueuedChatItem] = []
        collapsed: list[tuple[QueuedChatItem, QueuedChatItem]] = []
        private_survivors: dict[tuple[str, str, str], QueuedChatItem] = {}

        for item in waiting_items:
            if not self.is_private_message(item):
                survivors.append(item)
                continue

            scope = self._private_message_scope(item)
            survivor = private_survivors.get(scope)
            if survivor is None:
                private_survivors[scope] = item
                survivors.append(item)
                continue

            self._append_collapsed_item(survivor, item)
            collapsed.append((item, survivor))

        return_value = (survivors, collapsed)
        return return_value

    def _timestamp_from_item(self, item: QueuedChatItem) -> datetime | None:
        """Parse the queued item timestamp."""

        timestamp = item.timestamp
        if timestamp.endswith("Z"):
            timestamp = f"{timestamp[:-1]}+00:00"
        try:
            return_value = datetime.fromisoformat(timestamp)
        except ValueError:
            return_value = None
        return return_value

    def _seconds_between(
        self,
        first: QueuedChatItem,
        second: QueuedChatItem,
    ) -> float | None:
        """Return the timestamp gap between two queued items."""

        first_timestamp = self._timestamp_from_item(first)
        second_timestamp = self._timestamp_from_item(second)
        if first_timestamp is None or second_timestamp is None:
            return None
        return_value = max(0.0, (second_timestamp - first_timestamp).total_seconds())
        return return_value

    def _same_group_author(
        self,
        first: QueuedChatItem,
        second: QueuedChatItem,
    ) -> bool:
        """Return whether two group items are adjacent same-author candidates."""

        return_value = (
            self.is_group_message(first)
            and self.is_group_message(second)
            and first.request.platform == second.request.platform
            and first.request.platform_channel_id == second.request.platform_channel_id
            and first.request.platform_user_id == second.request.platform_user_id
        )
        return return_value

    def _group_gap_is_coalescible(
        self,
        first: QueuedChatItem,
        second: QueuedChatItem,
    ) -> bool:
        """Return whether the group follow-up gap is within the collapse window."""

        gap_seconds = self._seconds_between(first, second)
        return_value = (
            gap_seconds is not None
            and gap_seconds <= GROUP_COALESCE_MAX_GAP_SECONDS
        )
        return return_value

    def _coalesce_addressed_group(
        self,
        waiting_items: list[QueuedChatItem],
    ) -> tuple[list[QueuedChatItem], list[tuple[QueuedChatItem, QueuedChatItem]]]:
        """Collapse same-author group runs that start by addressing the bot."""

        survivors: list[QueuedChatItem] = []
        collapsed: list[tuple[QueuedChatItem, QueuedChatItem]] = []
        index = 0

        while index < len(waiting_items):
            run = [waiting_items[index]]
            next_index = index + 1
            while next_index < len(waiting_items):
                previous = run[-1]
                candidate = waiting_items[next_index]
                if not self._same_group_author(previous, candidate):
                    break
                if not self._group_gap_is_coalescible(previous, candidate):
                    break
                run.append(candidate)
                next_index += 1

            survivor = run[0]
            survivors.append(survivor)
            if len(run) > 1 and (
                self.is_tagged(survivor) or self.is_bot_reply(survivor)
            ):
                for item in run[1:]:
                    self._append_collapsed_item(survivor, item)
                    collapsed.append((item, survivor))
            else:
                survivors.extend(run[1:])

            index = next_index

        return_value = (survivors, collapsed)
        return return_value

    def _prune(
        self,
        waiting_items: list[QueuedChatItem],
    ) -> tuple[list[QueuedChatItem], list[QueuedChatItem]]:
        """Apply the global input-queue pruning policy to waiting items."""

        survivors = list(waiting_items)
        dropped: list[QueuedChatItem] = []

        if len(survivors) > 2:
            kept = []
            for item in survivors:
                if (
                    self.is_private_message(item)
                    or self.is_tagged(item)
                    or self.is_bot_reply(item)
                ):
                    kept.append(item)
                else:
                    dropped.append(item)
            survivors = kept

        if len(survivors) > 5:
            kept = []
            for item in survivors:
                if self.is_private_message(item) or self.is_bot_reply(item):
                    kept.append(item)
                else:
                    dropped.append(item)
            survivors = kept

        if len(survivors) > 5:
            private_items = [
                item for item in survivors if self.is_private_message(item)
            ]
            if private_items:
                non_private_items = [
                    item for item in survivors if not self.is_private_message(item)
                ]
                keep_count = max(0, 5 - len(private_items))
                kept_non_private = (
                    non_private_items[-keep_count:] if keep_count else []
                )
                kept_non_private_ids = {id(item) for item in kept_non_private}
                for item in non_private_items:
                    if id(item) not in kept_non_private_ids:
                        dropped.append(item)
                kept_ids = {id(item) for item in private_items + kept_non_private}
                survivors = [item for item in survivors if id(item) in kept_ids]
            else:
                dropped.extend(survivors[:-1])
                survivors = survivors[-1:]

        return_value = (survivors, dropped)
        return return_value
