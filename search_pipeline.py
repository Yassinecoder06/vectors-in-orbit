import os
import sys
import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Collection names
PRODUCTS_COLLECTION = "products_multimodal"
USER_PROFILES_COLLECTION = "user_profiles"
FINANCIAL_CONTEXTS_COLLECTION = "financial_contexts"
INTERACTIONS_COLLECTION = "interaction_memory"

# Expected vector dimensions per collection
EXPECTED_VECTOR_SIZES = {
    PRODUCTS_COLLECTION: 384,
    USER_PROFILES_COLLECTION: 384,
    FINANCIAL_CONTEXTS_COLLECTION: 256,
    INTERACTIONS_COLLECTION: 384,
}

# Embedding model (384-dim) with GPU support - faster, smaller model
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=_DEVICE)
logger.info(f"Embedding model loaded on device: {_DEVICE}")


def get_qdrant_client() -> QdrantClient:
    """
    Initialize Qdrant client from environment variables.
    """
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        logger.error("QDRANT_URL and QDRANT_API_KEY must be set.")
        raise EnvironmentError("Missing QDRANT_URL or QDRANT_API_KEY")

    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


def embed_query(text: str) -> List[float]:
    """
    Embed a query into a 384D vector using SentenceTransformer with GPU acceleration.
    """
    logger.info("Embedding query text...")
    try:
        embedding = _EMBEDDING_MODEL.encode(text, convert_to_numpy=True, device=_DEVICE)
        vec_list = embedding.tolist()
        
        expected_dim = EXPECTED_VECTOR_SIZES[PRODUCTS_COLLECTION]
        if len(vec_list) != expected_dim:
            raise ValueError(f"Embedding dimension mismatch: got {len(vec_list)}, expected {expected_dim}")
        
        return vec_list
    except Exception as exc:
        logger.exception("Embedding failed: %s", exc)
        raise


def get_user_context(user_id: str, client: QdrantClient) -> Dict[str, Any]:
    """
    Retrieve user profile and financial context for a given user_id.
    Returns a merged context dictionary.
    """
    logger.info("Fetching user context for user_id=%s", user_id)
    try:
        profile = client.retrieve(
            collection_name=USER_PROFILES_COLLECTION,
            ids=[user_id],
        )
        financial = client.retrieve(
            collection_name=FINANCIAL_CONTEXTS_COLLECTION,
            ids=[user_id],
        )

        profile_payload = profile[0].payload if profile else {}
        financial_payload = financial[0].payload if financial else {}

        context = {**profile_payload, **financial_payload}
        logger.info("User context assembled with %s fields", len(context))
        return context
    except Exception as exc:
        logger.exception("Failed to retrieve user context: %s", exc)
        return {}


def validate_collection_ready(client: QdrantClient, collection_name: str) -> None:
    """
    Validate collection exists, has data, and correct vector dimensions.
    """
    try:
        info = client.get_collection(collection_name)
        points_count = info.points_count
        vector_size = info.config.params.vectors.size
        
        logger.info(f"Collection '{collection_name}': {points_count} points, {vector_size}D")
        
        if points_count == 0:
            raise ValueError(f"Collection '{collection_name}' is empty. Run data insertion first.")
        
        expected_size = EXPECTED_VECTOR_SIZES.get(collection_name)
        if expected_size and vector_size != expected_size:
            raise ValueError(
                f"Vector size mismatch for '{collection_name}': "
                f"collection has {vector_size}D, expected {expected_size}D"
            )
    except Exception as e:
        logger.error(f"Collection validation failed: {e}")
        raise


def inspect_sample_payload(client: QdrantClient, collection_name: str) -> None:
    """
    Display sample payload structure and field types from collection for debugging.
    """
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        
        if points:
            sample = points[0]
            logger.info(f"Sample point from '{collection_name}':")
            logger.info(f"  ID: {sample.id}")
            logger.info(f"  Payload keys: {list(sample.payload.keys())}")
            for key, value in sample.payload.items():
                logger.info(f"    {key}: {value} (type: {type(value).__name__})")
        else:
            logger.warning(f"No points found in '{collection_name}'")
    except Exception as e:
        logger.error(f"Failed to inspect payload: {e}")


def semantic_product_search(
    client: QdrantClient,
    query_vector: List[float],
    max_price: float,
    enable_filters: bool = True,
) -> List[Dict[str, Any]]:
    """
    Semantic search against products collection with optional price filtering.
    Returns top 10 products ranked by semantic similarity.
    """
    logger.info(f"Semantic search: max_price={max_price:.2f}, filters={enable_filters}")
    
    validate_collection_ready(client, PRODUCTS_COLLECTION)
    
    expected_dim = EXPECTED_VECTOR_SIZES[PRODUCTS_COLLECTION]
    if len(query_vector) != expected_dim:
        raise ValueError(
            f"Query vector dimension mismatch: got {len(query_vector)}, expected {expected_dim}"
        )
    
    try:
        max_price_float = float(max_price)
    except (TypeError, ValueError):
        logger.warning(f"Invalid max_price '{max_price}', defaulting to 10000.0")
        max_price_float = 10000.0
    
    qdrant_filter = None
    if enable_filters:
        filter_conditions = [
            models.FieldCondition(
                key="price",
                range=models.Range(lte=max_price_float),
            ),
        ]
        qdrant_filter = models.Filter(must=filter_conditions)
        logger.debug(f"Filter: price <= {max_price_float}")

    try:
        results = client.query_points(
            collection_name=PRODUCTS_COLLECTION,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        
        logger.info(f"Search returned {len(results.points)} results")
        
        if len(results.points) == 0 and enable_filters:
            logger.warning("No filtered results. Retrying without filters...")
            return semantic_product_search(client, query_vector, max_price, enable_filters=False)

        products = []
        for point in results.points:
            payload = point.payload if point.payload else {}
            
            if "price" in payload and not isinstance(payload["price"], (int, float)):
                logger.warning(f"Point {point.id}: price is {type(payload['price'])}, expected numeric")
            if "in_stock" in payload and not isinstance(payload["in_stock"], bool):
                logger.warning(f"Point {point.id}: in_stock is {type(payload['in_stock'])}, expected bool")
            
            products.append({
                "id": point.id,
                "payload": payload,
                "semantic_score": point.score,
            })

        return products
        
    except Exception as exc:
        logger.exception(f"Semantic search failed: {exc}")
        raise RuntimeError(f"Qdrant search error: {exc}") from exc


def rerank_products(products: List[Dict[str, Any]], user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Rerank products by affordability and preferences.
    
    final_score = 0.45 * semantic_score + 0.35 * affordability_score + 0.20 * preference_score
    """
    logger.info("Reranking %d products...", len(products))

    try:
        available_balance = float(user_context.get("available_balance", 0.0))
    except (TypeError, ValueError):
        logger.warning("Invalid available_balance, defaulting to 0.0")
        available_balance = 0.0
    
    preferred_categories = set(user_context.get("preferred_categories", []))
    preferred_brands = set(user_context.get("preferred_brands", []))

    reranked = []

    for product in products:
        payload = product.get("payload", {})
        
        try:
            price = float(payload.get("price", 0.0))
        except (TypeError, ValueError):
            logger.warning(f"Invalid price in product {product.get('id')}: {payload.get('price')}")
            price = 0.0
        
        try:
            semantic_score = float(product.get("semantic_score", 0.0))
        except (TypeError, ValueError):
            logger.warning(f"Invalid semantic_score: {product.get('semantic_score')}")
            semantic_score = 0.0

        affordability_score = 0.0 if available_balance <= 0 else max(0.0, 1.0 - (price / available_balance))
        
        category_match = 1.0 if payload.get("category") in preferred_categories else 0.0
        brand_match = 1.0 if payload.get("brand") in preferred_brands else 0.0
        preference_score = max(category_match, brand_match)

        final_score = (0.45 * semantic_score + 0.35 * affordability_score + 0.20 * preference_score)

        reranked.append(
            {
                **product,
                "affordability_score": affordability_score,
                "preference_score": preference_score,
                "final_score": final_score,
            }
        )

    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    logger.info("Reranking complete.")
    return reranked

def search_products(
    user_id: str, 
    query: str, 
    top_k: int = 5,
    debug_mode: bool = False
) -> List[Dict[str, Any]]:
    """
    Execute full search pipeline: embed query, fetch context, search semantically, rerank.
    """
    try:
        client = get_qdrant_client()
        
        if debug_mode:
            logger.info("Debug mode enabled - inspecting collection...")
            inspect_sample_payload(client, PRODUCTS_COLLECTION)

        query_vector = embed_query(query)
        user_context = get_user_context(user_id, client)

        try:
            available_balance = float(user_context.get("available_balance", 0.0))
            credit_limit = float(user_context.get("credit_limit", 0.0))
        except (TypeError, ValueError) as e:
            logger.warning(f"Invalid financial context: {e}, using defaults")
            available_balance = 0.0
            credit_limit = 10000.0
        
        max_affordable_price = available_balance + credit_limit
        logger.info(f"Max affordable: {max_affordable_price:.2f} (balance={available_balance:.2f}, credit={credit_limit:.2f})")

        products = semantic_product_search(client, query_vector, max_affordable_price)
        
        if not products:
            logger.warning("No products found")
            return []
        
        reranked = rerank_products(products, user_context)
        return reranked[:top_k]
        
    except Exception as exc:
        logger.exception(f"Search failed: {exc}")
        raise


def _print_results(results: List[Dict[str, Any]]):
    """Pretty-print search results."""
    if not results:
        print("No results found.")
        return

    for idx, item in enumerate(results, start=1):
        payload = item.get("payload", {})
        print(
            f"#{idx} | {payload.get('name', 'Unknown Product')} | "
            f"Price: {payload.get('price', 'N/A')} | "
            f"Brand: {payload.get('brand', 'N/A')} | "
            f"Category: {payload.get('category', 'N/A')} | "
            f"Final Score: {item.get('final_score', 0):.4f}"
        )


if __name__ == "__main__":
    import uuid
    
    sample_user_id = str(uuid.uuid4())
    sample_query = "Laptop for machine learning under 1500 with installments"

    logger.info("Starting search pipeline...")
    logger.info(f"Query: '{sample_query}'")
    
    try:
        results = search_products(sample_user_id, sample_query, top_k=5, debug_mode=True)
        _print_results(results)
        
        if not results:
            logger.warning("No results found. Check:")
            logger.warning("  - Collection has data (run generate_and_insert_data.py)")
            logger.warning("  - User context has financial data")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
