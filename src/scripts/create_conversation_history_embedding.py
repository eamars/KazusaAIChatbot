"""Script to create/overwrite embeddings for existing conversation history messages.

This script reads all conversation history documents and generates/overwrites
embeddings for them using the configured embedding model.

Typical Use Cases:
    # Overwrite embeddings for all conversation messages
    create-embeddings
    
    # The script automatically:
    # - Finds all conversation history documents
    # - Generates/overwrites embeddings using the configured model
    # - Updates documents with embedding vectors
    # - Creates vector search index if needed
    
    # Run after adding new conversation history, when changing embedding models,
    # or when you need to refresh all embeddings
    create-embeddings
"""

import asyncio
import logging
from datetime import datetime

from kazusa_ai_chatbot.config import EMBEDDING_MODEL

from kazusa_ai_chatbot.db import (
    close_db,
    enable_vector_index,
)
from kazusa_ai_chatbot.db.script_operations import refresh_conversation_history_embeddings

logger = logging.getLogger(__name__)


async def main():
    """Create/overwrite embeddings for all conversation history messages."""
    logger.info("Starting conversation history embedding overwrite...")
    
    try:
        batch_size = 100
        result = await refresh_conversation_history_embeddings(
            batch_size=batch_size,
        )
        total_count = result["total_count"]
        logger.info(f"Found {total_count} conversation messages to process")

        if total_count == 0:
            logger.info("No conversation messages found. Nothing to do.")
            return

        logger.info(
            "Embedding overwrite completed. "
            f"Processed: {result['processed']}, Failed: {result['failed']}"
        )
        
        # Enable vector search index if it doesn't exist
        logger.info("Ensuring vector search index exists...")
        await enable_vector_index("conversation_history", "conversation_history_vector_index")
        logger.info("Vector search index setup completed")
        
    except Exception as exc:
        logger.debug(f"Handled exception in main: {exc}")
        logger.exception("Script failed")
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
