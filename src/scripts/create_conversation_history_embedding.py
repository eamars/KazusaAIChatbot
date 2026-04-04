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

from kazusa_ai_chatbot.config import MONGODB_URI, MONGODB_DB_NAME
from kazusa_ai_chatbot.db import (
    get_db,
    close_db,
    get_text_embedding,
    enable_vector_index,
    ConversationMessageDoc
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Create/overwrite embeddings for all conversation history messages."""
    logger.info("Starting conversation history embedding overwrite...")
    
    try:
        # Connect to database
        db = await get_db()
        collection = db.conversation_history
        
        # Find all conversation history documents to overwrite embeddings
        query = {}  # Match all documents
        
        # Count documents to process
        total_count = await collection.count_documents(query)
        logger.info(f"Found {total_count} conversation messages to process")
        
        if total_count == 0:
            logger.info("No conversation messages found. Nothing to do.")
            return
        
        # Process documents in batches to avoid memory issues
        batch_size = 100
        processed = 0
        failed = 0
        
        cursor = collection.find(query).batch_size(batch_size)
        
        async for doc in cursor:
            try:
                # Generate embedding for the message content
                content = doc.get("content", "")
                if not content:
                    logger.warning(f"Document {doc['_id']} has empty content, skipping")
                    failed += 1
                    continue
                
                logger.debug(f"Generating embedding for document {doc['_id']}")
                embedding = await get_text_embedding(content)
                
                # Update the document with the embedding
                await collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"embedding": embedding}}
                )
                
                processed += 1
                
                # Log progress every 10 documents
                if processed % 10 == 0:
                    logger.info(f"Processed {processed}/{total_count} documents")
                
            except Exception as e:
                logger.error(f"Failed to process document {doc['_id']}: {e}")
                failed += 1
                continue
        
        logger.info(f"Embedding overwrite completed. Processed: {processed}, Failed: {failed}")
        
        # Enable vector search index if it doesn't exist
        logger.info("Ensuring vector search index exists...")
        await enable_vector_index("conversation_history", "conversation_history_vector_index")
        logger.info("Vector search index setup completed")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
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