import time
import logging
from typing import Dict, Any, Optional
from uuid import uuid4
from qdrant_client import models
from search_pipeline import get_qdrant_client, embed_query, INTERACTIONS_COLLECTION

logger = logging.getLogger(__name__)

# Interaction weights config
INTERACTION_WEIGHTS = {
    "view": 0.1,
    "click": 0.3,
    "add_to_cart": 0.6,
    "purchase": 1.0
}

def log_interaction(
    user_id: str,
    product_payload: Dict[str, Any],
    interaction_type: str,
    query: Optional[str] = None
) -> None:
    """
    Log user interaction with a product into Qdrant interaction_memory.

    Args:
        user_id: Unique user identifier
        product_payload: Product metadata (must contain product_id/id, name, etc.)
        interaction_type: One of 'view', 'click', 'add_to_cart', 'purchase'
        query: Search query user used (optional)
    """
    try:
        client = get_qdrant_client()
        
        # 1. Validate inputs
        if interaction_type not in INTERACTION_WEIGHTS:
            logger.warning(f"Unknown interaction type: {interaction_type}, defaulting to view")
            interaction_type = "view"
        
        weight = INTERACTION_WEIGHTS.get(interaction_type, 0.1)
        timestamp = int(time.time())
        
        product_id = product_payload.get("id") or product_payload.get("product_id")
        if not product_id:
            logger.error("Cannot log interaction: Product ID missing in payload")
            return

        product_name = product_payload.get("name", "Unknown Product")
        categories = product_payload.get("categories", ["Unknown"])
        if not isinstance(categories, list) or not categories:
            categories = [str(categories)] if categories else ["Unknown"]
        category = categories[0]
        brand = product_payload.get("brand", "Unknown")
        try:
            price = float(product_payload.get("price", 0.0))
        except (ValueError, TypeError):
            price = 0.0

        # 2. Construct Behavioral Text
        # "user [action] [product] price [price] for [query]"
        action_verb = {
            "view": "viewed",
            "click": "clicked",
            "add_to_cart": "added to cart",
            "purchase": "purchased"
        }.get(interaction_type, "interacted with")

        behavioral_text = f"user {action_verb} {product_name} price {price}"
        if query:
            behavioral_text += f" for {query}"
        
        # 3. Generate Vector
        vector = embed_query(behavioral_text)

        # 4. Prepare Payload
        payload = {
            "user_id": user_id,
            "product_id": str(product_id),
            "interaction_type": interaction_type,
            "timestamp": timestamp,
            "category": category,
            "brand": brand,
            "price": price,
            "weight": weight,
            "original_query": query if query else ""
        }

        # 5. Upsert to Qdrant
        # We use a UUID for the interaction ID
        interaction_id = str(uuid4())
        
        client.upsert(
            collection_name=INTERACTIONS_COLLECTION,
            points=[
                models.PointStruct(
                    id=interaction_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        logger.info(f"Logged interaction: {user_id} -> {interaction_type} on {product_id}")

    except Exception as e:
        logger.exception(f"Failed to log interaction: {e}")
