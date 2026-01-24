import time
import logging
from typing import Dict, Any, Optional, List, Literal
from uuid import uuid4
from qdrant_client import models
import numpy as np
from search_pipeline import get_qdrant_client, embed_query, INTERACTIONS_COLLECTION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
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
    interaction_type: Literal["view", "click", "add_to_cart", "purchase"],
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
        logger.info(
            f"✅ Interaction saved: user={user_id}, type={interaction_type}, "
            f"product={product_id}, weight={weight:.2f}"
        )

    except Exception as e:
        logger.exception(f"❌ Failed to log interaction: {e}")


def get_top_interacted_products(
    timeframe_hours: int = 24,
    top_k: int = 10,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Get trending/popular products based on recent interactions.
    
    Aggregates interactions by product_id with:
    - Weighted by interaction type (purchase > add_to_cart > click > view)
    - Time-decayed (recent interactions count more)
    
    Args:
        timeframe_hours: Look back window in hours (default: 24)
        top_k: Number of top products to return (default: 10)
        debug: Enable debug logging (default: False)
    
    Returns:
        List of dicts with:
        - product_id: str
        - total_interactions: int
        - weighted_popularity_score: float (0-1, normalized)
        - last_interaction_timestamp: int
    """
    try:
        client = get_qdrant_client()
        current_time = int(time.time())
        cutoff_time = current_time - (timeframe_hours * 3600)
        
        logger.info(
            f"🔍 Fetching popular products (last {timeframe_hours}h, top {top_k})"
        )
        
        # Fetch all interactions within timeframe
        interactions, _ = client.scroll(
            collection_name=INTERACTIONS_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="timestamp",
                        range=models.Range(gte=cutoff_time)
                    )
                ]
            ),
            limit=10000,  # Adjust based on your scale
            with_payload=True,
            with_vectors=False,
        )
        
        if not interactions:
            logger.warning("No interactions found in timeframe")
            return []
        
        logger.info(f"Found {len(interactions)} interactions in timeframe")
        
        # Aggregate by product_id
        product_stats = {}
        
        # Time decay: half-life of 6 hours
        decay_constant = np.log(2) / (6 * 3600)
        
        for point in interactions:
            payload = point.payload or {}
            product_id = str(payload.get("product_id", ""))
            if not product_id:
                continue
            
            interaction_weight = payload.get("weight", 0.1)
            timestamp = payload.get("timestamp", current_time)
            
            # Time decay
            age_seconds = max(0, current_time - timestamp)
            time_decay = np.exp(-decay_constant * age_seconds)
            
            # Final weighted score
            weighted_score = interaction_weight * time_decay
            
            if product_id not in product_stats:
                product_stats[product_id] = {
                    "product_id": product_id,
                    "total_interactions": 0,
                    "weighted_score": 0.0,
                    "last_interaction_timestamp": timestamp,
                }
            
            product_stats[product_id]["total_interactions"] += 1
            product_stats[product_id]["weighted_score"] += weighted_score
            product_stats[product_id]["last_interaction_timestamp"] = max(
                product_stats[product_id]["last_interaction_timestamp"],
                timestamp
            )
        
        if not product_stats:
            logger.warning("No product stats computed")
            return []
        
        # Normalize scores to 0-1 range
        max_score = max(p["weighted_score"] for p in product_stats.values())
        if max_score > 0:
            for pid in product_stats:
                raw_score = product_stats[pid]["weighted_score"]
                product_stats[pid]["weighted_popularity_score"] = raw_score / max_score
        else:
            for pid in product_stats:
                product_stats[pid]["weighted_popularity_score"] = 0.0
        
        # Sort by weighted score and take top_k
        popular_products = sorted(
            product_stats.values(),
            key=lambda x: x["weighted_score"],
            reverse=True
        )[:top_k]
        
        if debug:
            logger.info(f"\n📊 Top {len(popular_products)} popular products:")
            for i, p in enumerate(popular_products, 1):
                logger.info(
                    f"  #{i}: product_id={p['product_id']}, "
                    f"interactions={p['total_interactions']}, "
                    f"popularity={p['weighted_popularity_score']:.3f}"
                )
        
        return popular_products
        
    except Exception as e:
        logger.exception(f"❌ Failed to compute popular products: {e}")
        return []
