"""Script to create embeddings for existing user facts documents.

This script reads all user facts documents that don't have embeddings
and generates embeddings for them using the configured embedding model.

Typical Use Cases:
    # Create embeddings for all user facts without embeddings
    create-user-facts-embeddings
    
    # The script automatically:
    # - Finds all user facts documents missing embeddings
    # - Generates embeddings using the configured model
    # - Updates documents with embedding vectors
    # - Creates vector search index if needed
    
    # Run after adding new user facts or when changing embedding models
    create-user-facts-embeddings
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
    enable_user_facts_vector_index,
    UserFactsDoc
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Create embeddings for user facts documents that lack them."""
    logger.info("Starting user facts embedding creation...")
    
    try:
        # Connect to database
        db = await get_db()
        collection = db.user_facts
        
        # Find all documents without embeddings or with empty embeddings
        query = {
            "$or": [
                {"embedding": {"$exists": False}},
                {"embedding": {"$size": 0}},
                {"embedding": None}
            ]
        }
        
        # Count documents to process
        total_count = await collection.count_documents(query)
        logger.info(f"Found {total_count} user facts documents without embeddings")
        
        if total_count == 0:
            logger.info("All user facts documents already have embeddings. Nothing to do.")
            return
        
        # Process documents in batches to avoid memory issues
        batch_size = 100
        processed = 0
        failed = 0
        
        cursor = collection.find(query).batch_size(batch_size)
        
        async for doc in cursor:
            try:
                # Generate embedding for the combined facts text
                facts = doc.get("facts", [])
                if not facts:
                    logger.warning(f"Document {doc['_id']} has no facts, setting empty embedding")
                    embedding = []
                else:
                    # Join facts with newline for better semantic separation
                    combined_facts_text = "\n".join(facts)
                    logger.debug(f"Generating embedding for user {doc['user_id']} with {len(facts)} facts")
                    embedding = await get_text_embedding(combined_facts_text)
                
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
        
        logger.info(f"Embedding creation completed. Processed: {processed}, Failed: {failed}")
        
        # Enable vector search index if it doesn't exist
        logger.info("Ensuring vector search index exists...")
        await enable_user_facts_vector_index()
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
