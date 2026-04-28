from langchain_core.tools import tool

from kazusa_ai_chatbot.db import get_conversation_history, search_conversation_history
from kazusa_ai_chatbot.db import search_memory as search_memory_db


@tool
async def search_conversation(search_query: str = "", 
                  global_user_id: str | None = None,
                  top_k: int = 5,
                  platform: str | None = None,
                  platform_channel_id: str | None = None,
                  from_timestamp: str | None = None,
                  to_timestamp: str | None = None,
    ) -> list[tuple[float, dict]]:
    """Search conversation history by semantic similarity.

    Mandatory argument rules:
    - search_query must be provided.
    - search_query must be a natural-language semantic query (not a keyword list).
    - Do not pass an empty string.
    
    Args:
        search_query (Mandatory): Semantic query sentence used for vector retrieval.
        global_user_id (Optional): Filter results to one user UUID.
        top_k (Optional): Maximum number of results to return. Default is 5.
        platform (Optional): Platform filter, e.g. "discord", "qq".
        platform_channel_id (Optional): Channel ID filter; if omitted, search all channels.
        from_timestamp (Optional): Start timestamp (ISO 8601).
        to_timestamp (Optional): End timestamp (ISO 8601).
        
    Returns:
        Top-K conversations close to the query, each as (similarity_score, message_with_metadata).
    """
    if not search_query or not search_query.strip():
        return [(-1.0, {"error": "search_query is mandatory and must not be empty. Please provide a natural-language semantic query."})]

    results = await search_conversation_history(
        query=search_query,
        platform=platform,
        platform_channel_id=platform_channel_id,
        global_user_id=global_user_id,
        limit=top_k,
        method="vector",
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )

    # Rebuild return format to remove unwanted columns
    return_list = []
    for (score, message) in results:
        return_list.append((score, {
            "content": message.get("content", ""),
            "timestamp": message.get("timestamp", ""),
            "display_name": message.get("display_name", ""),
            "role": message.get("role", ""),
            "platform": message.get("platform", ""),
            "platform_channel_id": message.get("platform_channel_id", ""),
            "platform_message_id": message.get("platform_message_id", ""),
            "platform_user_id": message.get("platform_user_id", ""),
            "global_user_id": message.get("global_user_id", ""),
            "reply_context": message.get("reply_context", {}),
        }))

    return return_list

@tool
async def search_conversation_keyword(
    keyword: str,
    global_user_id: str | None = None,
    top_k: int = 5,
    platform: str | None = None,
    platform_channel_id: str | None = None,
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
) -> list[dict]:
    """Search conversation history by exact keyword/phrase match (regex, case-insensitive).

    Use this tool when the search target is a specific term, technical phrase, or
    proper noun that must appear literally in the text (e.g. "HTTP", "DDR5").
    Prefer this over search_conversation when you know the exact wording.

    Args:
        keyword (Mandatory): Exact term or short phrase to match (regex, case-insensitive). Do not pass a full sentence — use the core noun/phrase only.
        global_user_id (Optional): Filter results to one user UUID.
        top_k (Optional): Maximum number of results. Default is 5.
        platform (Optional): Platform filter, e.g. "discord", "qq".
        platform_channel_id (Optional): Channel ID filter.
        from_timestamp (Optional): Start timestamp (ISO 8601).
        to_timestamp (Optional): End timestamp (ISO 8601).

    Returns:
        Matching conversations ordered by recency, each as a message dict.
    """
    if not keyword or not keyword.strip():
        return [{"error": "keyword is mandatory and must not be empty."}]

    results = await search_conversation_history(
        query=keyword,
        platform=platform,
        platform_channel_id=platform_channel_id,
        global_user_id=global_user_id,
        limit=top_k,
        method="keyword",
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )

    return [
        {
            "content": msg.get("content", ""),
            "timestamp": msg.get("timestamp", ""),
            "display_name": msg.get("display_name", ""),
            "role": msg.get("role", ""),
            "platform": msg.get("platform", ""),
            "platform_channel_id": msg.get("platform_channel_id", ""),
            "platform_message_id": msg.get("platform_message_id", ""),
            "platform_user_id": msg.get("platform_user_id", ""),
            "global_user_id": msg.get("global_user_id", ""),
            "reply_context": msg.get("reply_context", {}),
        }
        for _, msg in results
    ]


@tool
async def search_persistent_memory_keyword(
    keyword: str,
    top_k: int = 5,
    source_global_user_id: str | None = None,
    memory_type: str | None = None,
) -> list[dict]:
    """Search persistent memory by exact keyword/phrase match (regex, case-insensitive).

    Use this tool when the search target is a specific term, technical phrase, or
    proper noun that must appear literally in the stored memory content or name
    (e.g. "DDR5", "指令跟随逻辑"). Prefer this over search_persistent_memory when
    you know the exact wording.

    Mandatory argument rules:
    - keyword must be provided and should be the shortest unambiguous term or phrase.
    - Do not pass a full sentence — use the core noun/phrase only.

    Args:
        keyword (Mandatory): Exact term or short phrase to match (regex, case-insensitive).
        top_k (Optional): Maximum number of results. Default is 5.
        source_global_user_id (Optional): Filter by source user UUID.
        memory_type (Optional): Deprecated for RAG retrieval; retained for
            compatibility and returned metadata, but not used as a search filter.

    Returns:
        Matching memory entries with metadata.
    """
    if not keyword or not keyword.strip():
        return [{"error": "keyword is mandatory and must not be empty."}]

    results = await search_memory_db(
        query=keyword,
        limit=top_k,
        method="keyword",
        source_global_user_id=source_global_user_id,
        memory_type=None,
    )
    return [
        {
            "memory_name": mem.get("memory_name", ""),
            "content": mem.get("content", ""),
            "timestamp": mem.get("timestamp", ""),
            "source_global_user_id": mem.get("source_global_user_id", ""),
            "memory_type": mem.get("memory_type", ""),
            "status": mem.get("status", ""),
        }
        for _, mem in results
    ]


@tool
async def get_conversation(platform: str | None = None,
                           platform_channel_id: str | None = None,
                           limit: int = 5,
                           global_user_id: str | None = None,
                           display_name: str | None = None,
                           from_timestamp: str | None = None,
                           to_timestamp: str | None = None,
    ) -> list[dict]:
    """Get conversation history using structured filters.

    Usage rules:
    - At least one filter should be provided (for example platform_channel_id, global_user_id, or time range).
    - If both global_user_id and display_name are provided, global_user_id takes priority.
    - from_timestamp / to_timestamp should be ISO 8601 strings.
    
    Args:
        platform (Optional): Platform filter, e.g. "discord", "qq".
        platform_channel_id (Optional): Channel ID filter.
        limit (Optional): Maximum number of rows to return. Default is 5.
        global_user_id (Optional): User UUID filter.
        display_name (Optional): User display name filter (fallback if global_user_id is absent).
        from_timestamp (Optional): Start timestamp (ISO 8601), e.g. 2026-04-07T11:03:53.197223+00:00.
        to_timestamp (Optional): End timestamp (ISO 8601).
        
    Returns:
        A list of conversation messages.
    """
    return_list = []
    results = await get_conversation_history(
        platform=platform,
        platform_channel_id=platform_channel_id,
        limit=limit,
        global_user_id=global_user_id,
        display_name=display_name,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )

    # Rebuild return format to remove unwanted columns
    for message in results:
        return_list.append({
            "content": message.get("content", ""),
            "timestamp": message.get("timestamp", ""),
            "display_name": message.get("display_name", ""),
            "role": message.get("role", ""),
            "platform": message.get("platform", ""),
            "platform_channel_id": message.get("platform_channel_id", ""),
            "platform_message_id": message.get("platform_message_id", ""),
            "platform_user_id": message.get("platform_user_id", ""),
            "global_user_id": message.get("global_user_id", ""),
            "reply_context": message.get("reply_context", {}),
        })

    return return_list


@tool
async def search_persistent_memory(
    search_query: str,
    top_k: int = 5,
    source_global_user_id: str | None = None,
    memory_type: str | None = None,
    source_kind: str | None = None,
    status: str | None = None,
    expiry_before: str | None = None,
    expiry_after: str | None = None,
) -> list[dict]:
    """Search persistent memory by semantic similarity and optional metadata filters.

    Mandatory argument rules:
    - search_query must be provided.
    - search_query must be a natural-language semantic query (not a keyword list).
    - Do not call this tool with only filters and no search_query.
    
    Args:
        search_query (Mandatory): Semantic query sentence for vector retrieval.
        top_k (Optional): Maximum number of results to return. Default is 5.
        source_global_user_id (Optional): Filter by source user UUID.
        memory_type (Optional): Deprecated for RAG retrieval; retained for
            compatibility and returned metadata, but not used as a search filter.
        source_kind (Optional): Deprecated for RAG retrieval; retained for
            compatibility and returned metadata, but not used as a search filter.
        status (Optional): Filter by status, e.g. "active", "fulfilled", "expired", "superseded".
        expiry_before (Optional): ISO-8601 upper bound for expiry_timestamp (exclusive <).
        expiry_after (Optional): ISO-8601 lower bound for expiry_timestamp (exclusive >).

    Returns:
        Top-K memories close to the query, each with metadata and cosine similarity.
    """
    results = await search_memory_db(
        query=search_query,
        limit=top_k,
        method="vector",
        source_global_user_id=source_global_user_id,
        memory_type=None,
        source_kind=None,
        status=status,
        expiry_before=expiry_before,
        expiry_after=expiry_after,
    )

    # Rebuild return format to remove unwanted columns
    return_list = []
    for (score, memory) in results:
        return_list.append({
            "memory_name": memory.get("memory_name", ""),
            "content": memory["content"],
            "timestamp": memory["timestamp"],
            "source_global_user_id": memory.get("source_global_user_id", ""),
            "memory_type": memory.get("memory_type", ""),
            "source_kind": memory.get("source_kind", ""),
            "status": memory.get("status", ""),
            "confidence_note": memory.get("confidence_note", ""),
            "expiry_timestamp": memory.get("expiry_timestamp"),
            "cosine_similarity": score,
        })

    return return_list
