"""Script to insert memory entries into the memory collection.

This script provides a command-line interface to store memory entries
with embeddings for semantic search.

Typical Use Cases:
    # Store a simple memory entry
    insert-memory "Embedding Guide" "Use vector search for semantic recall and keyword search as fallback"
    
    # Store a detailed article summary
    insert-memory "LangGraph Supervisor" "Source: https://example.com/langgraph. The article explains the supervisor plus sub-agent pattern, emphasizes isolated agent contexts, and recommends explicit supervisor-authored instructions."
    
    # Store a technical note
    insert-memory "Python 3.14 Features" "Python 3.14 introduces new pattern matching syntax, improved error messages, and performance optimizations."
"""

import asyncio
import logging
import argparse
from datetime import datetime, timezone

from kazusa_ai_chatbot.db import (
    get_db,
    close_db,
    save_memory,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Insert a memory entry based on command line arguments."""
    parser = argparse.ArgumentParser(description="Insert memory entry")
    parser.add_argument("memory_name", help="Name/identifier for the memory")
    parser.add_argument("content", help="The memory content/text")
    
    args = parser.parse_args()
    
    logger.info(f"Inserting memory entry: '{args.memory_name}'")
    logger.info(f"Content length: {len(args.content)} characters")
    
    try:
        # Connect to database
        await get_db()
        
        # Save memory
        await save_memory(
            memory_name=args.memory_name,
            content=args.content,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        
        logger.info("Memory entry saved successfully!")
        print(f"✓ Memory '{args.memory_name}' has been saved.")
        
    except Exception as e:
        logger.error(f"Insert failed: {e}")
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
