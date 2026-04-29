"""Script to search conversation history using keyword or vector search.

This script provides a command-line interface to search through conversation history
using either keyword regex search or semantic vector search.

Typical Use Cases:
    # Search for conversations about desserts using semantic search
    search-conversations "甜点" --method vector --limit 5
    
    # Search for messages from a specific user about making mistakes
    search-conversations "说错话" --user 320899931776745483 --method keyword
    
    # Search in a specific channel for recent discussions
    search-conversations "northern gate" --channel 123456789 --method vector --limit 10
    
    # Broad semantic search across all conversations
    search-conversations "what happened at the northern gate" --method vector
    
    # Keyword search with regex pattern
    search-conversations "gate.*attack" --method keyword --limit 3
"""

import asyncio
import logging
import argparse
from typing import List, Tuple

from kazusa_ai_chatbot.db import (
    get_db,
    close_db,
    search_conversation_history,
    ConversationMessageDoc
)

logger = logging.getLogger(__name__)


def format_message(doc: ConversationMessageDoc, score: float) -> str:
    """Format a conversation message for display."""
    timestamp = doc.get("timestamp", "Unknown time")
    content = doc.get("content", "")
    user_id = doc.get("user_id", "Unknown user")
    channel_id = doc.get("channel_id", "Unknown channel")
    
    return_value = f"[Score: {score:.4f}] {timestamp} | User: {user_id} | Channel: {channel_id}\n{content}\n{'-'*80}"
    return return_value


async def main():
    """Search conversation history based on command line arguments."""
    parser = argparse.ArgumentParser(description="Search conversation history")
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--method", choices=["keyword", "vector"], default="vector", 
                       help="Search method: keyword (regex) or vector (semantic)")
    parser.add_argument("--channel", help="Filter by channel ID")
    parser.add_argument("--user", help="Filter by user ID")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")
    
    args = parser.parse_args()
    
    logger.info(f"Searching conversation history with query: '{args.query}'")
    logger.info(f"Method: {args.method}, Limit: {args.limit}")
    
    if args.channel:
        logger.info(f"Channel filter: {args.channel}")
    if args.user:
        logger.info(f"User filter: {args.user}")
    
    try:
        # Connect to database
        await get_db()
        
        # Perform search
        results: List[Tuple[float, ConversationMessageDoc]] = await search_conversation_history(
            query=args.query,
            channel_id=args.channel,
            user_id=args.user,
            limit=args.limit,
            method=args.method
        )
        
        if not results:
            logger.info("No results found.")
            return
        
        logger.info(f"Found {len(results)} results:")
        print()
        
        for score, doc in results:
            print(format_message(doc, score))
        
    except Exception as exc:
        logger.debug(f"Handled exception in main: {exc}")
        logger.exception("Search failed")
        raise
    finally:
        # Close database connection
        await close_db()
        logger.info("Database connection closed")


if __name__ == "__main__":
    asyncio.run(main())


def async_main():
    """Wrapper for async main function."""
    asyncio.run(main())
