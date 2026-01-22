import os
import sys
from typing import Dict, Any
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

load_dotenv()

# Configuration for the Qdrant Collections
# This dictionary helps maintain modularity and easier updates to the schema.
COLLECTION_CONFIGS = {
    "products_multimodal": {
        "vector_size": 384,
        "distance": models.Distance.COSINE,
        "payload_indexes": {
            "product_id": models.PayloadSchemaType.KEYWORD,
            "name": models.PayloadSchemaType.KEYWORD,
            "price": models.PayloadSchemaType.FLOAT,
            "category": models.PayloadSchemaType.KEYWORD,
            "brand": models.PayloadSchemaType.KEYWORD,
            "in_stock": models.PayloadSchemaType.BOOL,
            "image_url": models.PayloadSchemaType.KEYWORD,
            "region": models.PayloadSchemaType.KEYWORD,
            "monthly_installment": models.PayloadSchemaType.FLOAT,
        }
    },

    "user_profiles": {
        "vector_size": 384,
        "distance": models.Distance.COSINE,
        "payload_indexes": {
            "user_id": models.PayloadSchemaType.KEYWORD,
            "name": models.PayloadSchemaType.KEYWORD,
            "location": models.PayloadSchemaType.KEYWORD,
            "risk_tolerance": models.PayloadSchemaType.KEYWORD,
            "preferred_categories": models.PayloadSchemaType.KEYWORD,
            "preferred_brands": models.PayloadSchemaType.KEYWORD,
        }
    },

    "financial_contexts": {
        "vector_size": 256,
        "distance": models.Distance.COSINE,
        "payload_indexes": {
            "user_id": models.PayloadSchemaType.KEYWORD,
            "available_balance": models.PayloadSchemaType.FLOAT,
            "credit_limit": models.PayloadSchemaType.FLOAT,
            "current_debt": models.PayloadSchemaType.FLOAT,
            "eligible_installments": models.PayloadSchemaType.BOOL,
        }
    },

    "interaction_memory": {
        "vector_size": 384,
        "distance": models.Distance.COSINE,
        "payload_indexes": {
            "user_id": models.PayloadSchemaType.KEYWORD,
            "query": models.PayloadSchemaType.KEYWORD,
            "clicked_product_id": models.PayloadSchemaType.KEYWORD,
            "purchased": models.PayloadSchemaType.BOOL,
        }
    }
}

def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns the QdrantClient using environment variables.
    """
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        print("Error: QDRANT_URL and QDRANT_API_KEY environment variables must be set.")
        sys.exit(1)

    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        # Verify connection by getting collection list (lightweight check)
        client.get_collections()
        print(f"Successfully connected to Qdrant at {qdrant_url}")
        return client
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        sys.exit(1)

def create_collection_schema(client: QdrantClient, collection_name: str, config: Dict[str, Any]):
    """
    Recreates a collection with the specified vector configuration and payload indexes.
    """
    try:
        print(f"\n--- Setting up collection: {collection_name} ---")
        
        # 1. Recreate the collection
        # We handle recreation manually to avoid DeprecationWarning for recreate_collection
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)
            print(f"   Note: Deleted existing collection '{collection_name}'")
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=config["vector_size"],
                distance=config["distance"]
            )
        )
        print(f"✅ Collection '{collection_name}' created/recreated successfully.")

        # 2. Create Payload Indexes (includes BOOL fields)
        indexes = config.get("payload_indexes", {})
        for field_name, field_type in indexes.items():
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                print(f"✅ Index created for field: '{field_name}' ({field_type})")
            except Exception as idx_err:
                print(f"⚠️  Failed to create index for '{field_name}': {idx_err}")

    except Exception as e:
        print(f"❌ Critical error setting up collection '{collection_name}': {e}")
        # Depending on requirements, we might want to raise here to stop the whole script
        raise e

def main():
    """
    Main execution routine.
    """
    print("Starting Context-Aware FinCommerce Qdrant Schema Setup...")
    
    # 1. Initialize Client
    client = get_qdrant_client()

    # 2. Iterate through configuration and create collections
    for name, config in COLLECTION_CONFIGS.items():
        try:
            create_collection_schema(client, name, config)
        except Exception:
            print("Aborting due to error.")
            sys.exit(1)

    print("\n🎉 Qdrant schema setup completed successfully.")

if __name__ == "__main__":
    main()
