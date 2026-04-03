"""Script to search users based on their accumulated facts using semantic search.

This script provides a command-line interface to search for users based on
the semantic similarity of their stored facts and characteristics.

Typical Use Cases:
    # Search for users who like music using semantic search
    search-user-facts "musical person" --method vector --limit 5
    
    # Search for users with specific characteristics
    search-user-facts "shy and quiet" --method vector --limit 10
    
    # Search for users with high affinity scores
    search-user-facts "friendly outgoing" --min-affinity 700 --limit 3
    
    # Search for users and show their facts
    search-user-facts "creative artistic" --show-facts --limit 5
    
    # Keyword search for specific terms in user facts
    search-user-facts "guitar.*music" --method keyword --limit 3
"""

import asyncio
import logging
import argparse
from typing import List, Tuple

from kazusa_ai_chatbot.db import (
    get_db,
    close_db,
    search_users_by_facts,
    get_user_facts,
    get_affinity,
    UserFactsDoc
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def format_user_result(score: float, user_doc: UserFactsDoc, show_facts: bool = False) -> str:
    """Format a user search result for display."""
    user_id = user_doc.get("user_id", "Unknown user")
    facts = user_doc.get("facts", [])
    affinity = user_doc.get("affinity", 500)
    
    result = f"[Score: {score:.4f}] User: {user_id} | Affinity: {affinity}/1000"
    
    if facts:
        result += f" | Facts: {len(facts)}"
    
    if show_facts and facts:
        result += f"\n  Facts: {', '.join(facts[:3])}"
        if len(facts) > 3:
            result += f"... ({len(facts)-3} more)"
    
    return result + "\n" + "-" * 80


async def main():
    """Search users based on their accumulated facts."""
    parser = argparse.ArgumentParser(description="Search users based on their facts")
    parser.add_argument("query", help="Search query text")
    parser.add_argument("--method", choices=["keyword", "vector"], default="vector", 
                       help="Search method: keyword (regex) or vector (semantic)")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of results")
    parser.add_argument("--min-affinity", type=int, help="Filter by minimum affinity score")
    parser.add_argument("--max-affinity", type=int, help="Filter by maximum affinity score")
    parser.add_argument("--show-facts", action="store_true", help="Show user facts in results")
    
    args = parser.parse_args()
    
    logger.info(f"Searching users with query: '{args.query}'")
    logger.info(f"Method: {args.method}, Limit: {args.limit}")
    
    if args.min_affinity:
        logger.info(f"Minimum affinity filter: {args.min_affinity}")
    if args.max_affinity:
        logger.info(f"Maximum affinity filter: {args.max_affinity}")
    
    try:
        # For now, we only support vector search for user facts
        # Keyword search would require regex matching on facts
        if args.method == "keyword":
            logger.warning("Keyword search not yet implemented for user facts, using vector search")
        
        # Search users by facts
        results = await search_users_by_facts(args.query, limit=args.limit)
        
        # Apply affinity filters if specified
        if args.min_affinity or args.max_affinity:
            filtered_results = []
            for score, user_doc in results:
                affinity = user_doc.get("affinity", 500)
                if args.min_affinity and affinity < args.min_affinity:
                    continue
                if args.max_affinity and affinity > args.max_affinity:
                    continue
                filtered_results.append((score, user_doc))
            results = filtered_results
        
        if not results:
            logger.info("No users found matching the criteria")
            return
        
        logger.info(f"Found {len(results)} results:")
        
        # Display results
        for score, user_doc in results:
            print(format_user_result(score, user_doc, show_facts=args.show_facts))
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise
    finally:
        # Close database connection
        await close_db()
        logger.info("Database connection closed")


def async_main():
    """Wrapper for async main function."""
    asyncio.run(main())


if __name__ == "__main__":
    async_main()
