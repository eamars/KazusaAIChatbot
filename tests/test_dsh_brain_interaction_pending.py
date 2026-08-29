"""Relay pending interaction lineage tests."""

from __future__ import annotations


def test_relay_pending_matches_exact_delivered_reply_lineage_and_expires_closed() -> None:
    from kazusa_ai_chatbot.dsh_interaction.pending import PendingInteractionStore

    store = PendingInteractionStore()
    store.create(
        interaction_id="i1",
        resolution_thread_id="thread-1",
        segment_id="segment-1",
        platform="debug",
        platform_channel_id="channel-1",
        global_user_id="user-1",
        delivered_platform_message_id="message-1",
        expires_at="2026-08-29T00:00:00Z",
    )
    assert store.match_reply(
        platform="debug",
        platform_channel_id="channel-1",
        global_user_id="user-1",
        reply_to_platform_message_id="message-1",
        now="2026-08-28T00:00:00Z",
    )
    assert not store.match_reply(
        platform="debug",
        platform_channel_id="other",
        global_user_id="user-1",
        reply_to_platform_message_id="message-1",
        now="2026-08-28T00:00:00Z",
    )
