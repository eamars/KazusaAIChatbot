"""Script to insert memory entries into the memory collection.

This script provides a command-line interface to store memory entries
with embeddings for semantic search.

Typical Use Cases:
    # Store a non-user-specific world knowledge entry in the memory collection
    insert-memory "Embedding Guide" "Use vector search for semantic recall and keyword search as fallback"
    
    # Store a detailed article summary as shared/world knowledge
    insert-memory "LangGraph Supervisor" "Source: https://example.com/langgraph. The article explains the supervisor plus sub-agent pattern, emphasizes isolated agent contexts, and recommends explicit supervisor-authored instructions."
    
    # Store a user-scoped memory entry
    insert-memory "Python 3.14 Features" "User asked about new language features." --source-global-user-id 123e4567-e89b-12d3-a456-426614174000
"""

import asyncio
import logging
import argparse
from datetime import datetime, timezone

from kazusa_ai_chatbot.db import (
    get_db,
    close_db,
    save_memory,
    build_memory_doc,
)

logger = logging.getLogger(__name__)


async def main():
    """Insert a memory entry based on command line arguments."""
    parser = argparse.ArgumentParser(description="Insert memory entry")
    parser.add_argument("memory_name", help="Name/identifier for the memory")
    parser.add_argument("content", help="The memory content/text")
    parser.add_argument("--memory-type", default="fact", help="Memory type (fact, promise, impression, narrative, defense_rule)")
    parser.add_argument("--source-kind", default="seeded_manual", help="Source kind")
    parser.add_argument("--confidence-note", default="Manually inserted memory.", help="Confidence note")
    parser.add_argument(
        "--source-global-user-id",
        default="",
        help=(
            "Internal user UUID for user-scoped memory. "
            "Leave empty to store non-user-specific/world knowledge in the memory collection."
        ),
    )
    
    args = parser.parse_args()
    
    logger.info(f"Inserting memory entry: '{args.memory_name}'")
    logger.info(f"Content length: {len(args.content)} characters")
    
    try:
        # Connect to database
        await get_db()
        
        # Save memory
        doc = build_memory_doc(
            memory_name=args.memory_name,
            content=args.content,
            source_global_user_id=args.source_global_user_id,
            memory_type=args.memory_type,
            source_kind=args.source_kind,
            confidence_note=args.confidence_note,
        )
        await save_memory(doc, datetime.now(timezone.utc).isoformat())
        
        logger.info("Memory entry saved successfully!")
        print(f"✓ Memory '{args.memory_name}' has been saved.")
        
    except Exception as exc:
        logger.debug(f"Handled exception in main: {exc}")
        logger.exception("Insert failed")
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
