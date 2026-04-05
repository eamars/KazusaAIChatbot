"""Script to search memory entries using keyword or vector search.

This script provides a command-line interface to search through memory entries
using either keyword regex search or semantic vector search.

Typical Use Cases:
    # Search for memories about embeddings using semantic search
    search-memory "embedding guide" --method vector --limit 5
    
    # Search for memories about a specific topic using keyword search
    search-memory "LangGraph" --method keyword
    
    # Search for technical documentation
    search-memory "Python features" --method vector --limit 10
    
    # Broad semantic search across all memories
    search-memory "supervisor pattern" --method vector
    
    # Keyword search with regex pattern
    search-memory "pattern.*agent" --method keyword --limit 3
"""

import asyncio
import logging
import argparse
from typing import List, Tuple

from kazusa_ai_chatbot.db import (
    get_db,
    close_db,
    search_memory,
    MemoryDoc,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def format_memory(doc: MemoryDoc, score: float) -> str:
    """Format a memory entry for display."""
    memory_name = doc.get("memory_name", "Unknown memory")
    content = doc.get("content", "")
    
    # Truncate content if too long
    if len(content) > 200:
        content = content[:200] + "..."
    
    return f"[Score: {score:.4f}] {memory_name}\n{content}\n{'-'*80}"


async def main():
    """Search memory entries based on command line arguments."""
    parser = argparse.ArgumentParser(description="Search memory entries")
    parser.add_argument("query", help="Search query string")
    parser.add_argument("--method", choices=["keyword", "vector"], default="vector", 
                       help="Search method: keyword (regex) or vector (semantic)")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")
    
    args = parser.parse_args()
    
    logger.info(f"Searching memory with query: '{args.query}'")
    logger.info(f"Method: {args.method}, Limit: {args.limit}")
    
    try:
        # Connect to database
        await get_db()
        
        # Perform search
        results: List[Tuple[float, MemoryDoc]] = await search_memory(
            query=args.query,
            limit=args.limit,
            method=args.method
        )
        
        if not results:
            logger.info("No memory entries found.")
            return
        
        logger.info(f"Found {len(results)} memory entries:")
        print()
        
        for score, doc in results:
            print(format_memory(doc, score))
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
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
