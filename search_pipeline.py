import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv
import numpy as np

# Optional pretty printing
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _RICH_AVAILABLE = True
    _console = Console()
except Exception:
    _RICH_AVAILABLE = False
    _console = None

# Optional matplotlib for charts
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False

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


def _clamp01(value: float) -> float:
    """Clamp a float to the [0, 1] range."""
    return max(0.0, min(1.0, value))


def _normalize_text(value: Any) -> str:
    """Lower-case string normalization for matching."""
    if value is None:
        return ""
    return str(value).strip().lower()


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

# =============================================================================
# Collaborative Filtering Logic
# =============================================================================

def get_user_behavior_vector(client: QdrantClient, user_id: str) -> Optional[List[float]]:
    """
    Construct a user behavior vector based on recent interactions.
    Applies time decay and interaction weights.
    """
    try:
        # Fetch last 10 interactions
        interactions, _ = client.scroll(
            collection_name=INTERACTIONS_COLLECTION,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=10,
            with_vectors=True,
            with_payload=True,
        )

        if not interactions:
            logger.info(f"No interactions found for user {user_id}")
            return None

        vectors = []
        weights = []
        current_time = int(time.time())
        # Decay half-life: 7 days (60*60*24*7 seconds = 604800)
        decay_constant = np.log(2) / 604800

        for point in interactions:
            if not point.vector:
                continue
            
            payload = point.payload or {}
            timestamp = payload.get("timestamp", current_time)
            base_weight = payload.get("weight", 0.1)
            
            # Time decay
            age_seconds = max(0, current_time - timestamp)
            time_decay = np.exp(-decay_constant * age_seconds)
            
            final_weight = base_weight * time_decay
            
            vectors.append(point.vector)
            weights.append(final_weight)

        if not vectors:
            return None

        # Compute weighted average
        weighted_sum = np.average(vectors, axis=0, weights=weights)
        return weighted_sum.tolist()

    except Exception as e:
        logger.warning(f"Failed to compute user behavior vector: {e}")
        return None


def get_collaborative_scores(client: QdrantClient, user_id: str, candidate_ids: List[str]) -> Dict[str, float]:
    """
    Compute collaborative scores for candidate products based on similar users' interactions.
    """
    scores = {pid: 0.0 for pid in candidate_ids}
    
    try:
        user_vector = get_user_behavior_vector(client, user_id)
        if not user_vector:
            return scores

        # Find similar interactions (User-based CF proxy)
        # We search for interactions similar to the user's aggregate behavior
        # excluding the user's own interactions to find *similar users*
        similar_interactions = client.search(
            collection_name=INTERACTIONS_COLLECTION,
            query_vector=user_vector,
            query_filter=models.Filter(
                must_not=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
            ),
            limit=20,
            with_payload=True
        )

        total_score = 0.0
        product_scores = {}

        for hit in similar_interactions:
            payload = hit.payload or {}
            product_id = str(payload.get("product_id"))
            
            # Boost: Similarity * Interaction Weight
            interaction_weight = payload.get("weight", 0.1)
            score = hit.score * interaction_weight
            
            product_scores[product_id] = product_scores.get(product_id, 0.0) + score
            total_score += score

        # Normalize and map to candidates
        if product_scores:
             max_s = max(product_scores.values())
             for pid in candidate_ids:
                 if pid in product_scores:
                     scores[pid] = product_scores[pid] / max_s

    except Exception as e:
        logger.warning(f"Collaborative filtering failed: {e}")
    
    return scores


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


def rerank_products(
    products: List[Dict[str, Any]], 
    user_context: Dict[str, Any],
    client: QdrantClient,
    debug_mode: bool = False
) -> List[Dict[str, Any]]:
    """
    Rerank products with popularity-aware scoring.
    
    final_score = 
      0.30 * semantic_score + 
      0.25 * affordability_score + 
      0.15 * preference_score + 
      0.20 * collaborative_score +
      0.10 * popularity_score
    """
    logger.info("Reranking %d products with popularity awareness...", len(products))

    try:
        available_balance = float(user_context.get("available_balance", 0.0))
        credit_limit = float(user_context.get("credit_limit", 0.0))
    except (TypeError, ValueError):
        logger.warning("Invalid financial context, defaulting to 0.0")
        available_balance = 0.0
        credit_limit = 0.0

    total_budget = max(0.0, available_balance + credit_limit)
    user_id = user_context.get("user_id", "")
    
    preferred_categories = {_normalize_text(c) for c in user_context.get("preferred_categories", [])}
    preferred_brands = {_normalize_text(b) for b in user_context.get("preferred_brands", [])}

    # Fetch Collaborative Scores
    product_ids = [str(p.get("id")) for p in products]
    collab_scores = get_collaborative_scores(client, user_id, product_ids)
    
    # Fetch Popularity Scores
    from interaction_logger import get_top_interacted_products
    popularity_data = get_top_interacted_products(timeframe_hours=24, top_k=100, debug=debug_mode)
    popularity_map = {p["product_id"]: p["weighted_popularity_score"] for p in popularity_data}
    
    if debug_mode and popularity_data:
        logger.info(f"📊 Loaded popularity data for {len(popularity_data)} products")
        logger.info(f"   Top popular: {popularity_data[0]['product_id']} (score: {popularity_data[0]['weighted_popularity_score']:.3f})")

    reranked = []

    for product in products:
        payload = product.get("payload", {})
        pid = str(product.get("id"))
        
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

        # Affordability: Must never be negative
        if total_budget <= 0:
            affordability_score = 0.0
        else:
            ratio = price / total_budget
            affordability_score = _clamp01(1.0 - ratio)

        # Preference: 1.0 if match, or if no preferences set
        categories = payload.get("categories", [])
        if not isinstance(categories, list):
            categories = [categories] if categories else []
        normalized_categories = {_normalize_text(c) for c in categories}
        brand = _normalize_text(payload.get("brand"))
        
        category_match = any(c in preferred_categories for c in normalized_categories)
        brand_match = brand in preferred_brands

        if not preferred_categories and not preferred_brands:
            preference_score = 1.0  # No penalty if user did not specify preferences
        else:
            preference_score = 1.0 if (category_match or brand_match) else 0.0

        # Collaborative Score
        collaborative_score = collab_scores.get(pid, 0.0)
        
        # Popularity Score
        popularity_score = popularity_map.get(pid, 0.0)

        # Final Score with popularity
        final_score = (
            0.30 * semantic_score +
            0.25 * affordability_score +
            0.15 * preference_score +
            0.20 * collaborative_score +
            0.10 * popularity_score
        )
        final_score = _clamp01(final_score)

        # Enhanced Explainability with multiple reasons
        explanations = []
        
        # Popularity explanations (always show if non-zero)
        if popularity_score > 0.7:
            explanations.append("🔥 Trending: Very popular in last 24 hours")
        elif popularity_score > 0.4:
            explanations.append("📈 Popular among users recently")
        elif popularity_score > 0.1:
            explanations.append("👥 Other users are viewing this")
        
        # Affordability explanations
        if affordability_score > 0.8:
            explanations.append("💰 Well within your budget")
        elif affordability_score > 0.5:
            explanations.append("💵 Affordable for your budget")
        elif affordability_score > 0.2:
            explanations.append("⚠️ Stretches your budget")
        
        # Collaborative explanations
        if collaborative_score > 0.5:
            explanations.append("🤝 Users with similar behavior purchased this")
        elif collaborative_score > 0.1:
            explanations.append("👤 Similar users interacted with this")
        
        # Preference explanations
        if category_match and brand_match:
            matched_cat = next((c for c in categories if _normalize_text(c) in preferred_categories), None)
            if matched_cat:
                explanations.append(f"❤️ Matches your brand ({payload.get('brand')}) & category ({matched_cat})")
            else:
                explanations.append(f"❤️ Matches your brand ({payload.get('brand')}) & category preference")
        elif brand_match:
            explanations.append(f"⭐ Matches your preferred brand: {payload.get('brand')}")
        elif category_match:
            matched_cat = next((c for c in categories if _normalize_text(c) in preferred_categories), None)
            if matched_cat:
                explanations.append(f"🎯 Matches your category preference: {matched_cat}")
        
        # Semantic relevance (always include)
        if semantic_score > 0.7:
            explanations.append("🎯 Strong match to your search query")
        elif semantic_score > 0.4:
            explanations.append("📝 Relevant to your search")
        
        # Fallback if no explanations
        if not explanations:
            explanations.append("Based on semantic relevance to your query")
        
        # Primary reason (for backward compatibility)
        reason_text = explanations[0]

        reranked.append(
            {
                **product,
                "affordability_score": affordability_score,
                "preference_score": preference_score,
                "collaborative_score": collaborative_score,
                "popularity_score": popularity_score,
                "final_score": final_score,
                "reason": reason_text,
                "explanations": explanations,
            }
        )
    
    # Sort by final score
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    
    if debug_mode:
        logger.info("📊 Reranking complete - Score breakdown:")
        for i, p in enumerate(reranked[:3], 1):
            logger.info(
                f"  #{i}: final={p['final_score']:.3f} "
                f"(sem={p.get('semantic_score', 0):.2f}, "
                f"aff={p['affordability_score']:.2f}, "
                f"pref={p['preference_score']:.2f}, "
                f"collab={p['collaborative_score']:.2f}, "
                f"pop={p['popularity_score']:.2f})"
            )
    
    logger.info("Reranking complete.")
    return reranked

def search_products(
    user_id: str, 
    query: str, 
    top_k: int = 5,
    debug_mode: bool = False,
    override_context: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Execute full search pipeline: embed query, fetch context, search semantically, rerank.
    
    Args:
        user_id: User identifier for context retrieval
        query: Natural language search query
        top_k: Number of results to return
        debug_mode: Enable verbose logging
        override_context: Optional dict to override Qdrant-retrieved user context
                          (used by Streamlit UI to pass sidebar values)
    """
    try:
        client = get_qdrant_client()
        
        if debug_mode:
            logger.info("Debug mode enabled - inspecting collection...")
            inspect_sample_payload(client, PRODUCTS_COLLECTION)

        query_vector = embed_query(query)
        
        # Use override context if provided, otherwise fetch from Qdrant
        if override_context:
            user_context = override_context
            logger.info("Using override context from UI")
        else:
            user_context = get_user_context(user_id, client)

        # Ensure user_id is in context for collaborative filtering
        if "user_id" not in user_context:
            user_context["user_id"] = user_id

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
        
        reranked = rerank_products(products, user_context, client, debug_mode=debug_mode)
        return reranked[:top_k]
        
    except Exception as exc:
        logger.exception(f"Search failed: {exc}")
        raise


def _print_results(results: List[Dict[str, Any]], summary: Optional[Dict[str, Any]] = None):
    """Visualize search results using matplotlib only."""
    if not results:
        print("No results found.")
        return

    if _MATPLOTLIB_AVAILABLE:
        _visualize_results(results, summary)
    else:
        # Fallback plain text
        print("Matplotlib not available. Install with: pip install matplotlib")
        for idx, item in enumerate(results[:5], start=1):
            p = item.get("payload", {})
            print(
                f"{idx}. {p.get('name','Unknown')} | Price: ${p.get('price','N/A')} | "
                f"Brand: {p.get('brand','N/A')} | Score: {item.get('final_score',0):.3f}"
            )


def _visualize_results(results: List[Dict[str, Any]], summary: Optional[Dict[str, Any]] = None):
    """Generate comprehensive matplotlib visualization with multiple panels."""
    import os

    if not _MATPLOTLIB_AVAILABLE:
        return

    try:
        n = min(len(results), 8)  # Show up to 8 products
        products = []
        final_scores = []
        semantic_scores = []
        afford_scores = []
        pref_scores = []
        prices = []
        brands = []
        stock_status = []

        for i in range(n):
            item = results[i]
            p = item.get("payload", {})
            name = str(p.get("name", f"Product {i+1}"))[:45]
            products.append(name)
            final_scores.append(item.get("final_score", 0.0))
            semantic_scores.append(item.get("semantic_score", 0.0))
            afford_scores.append(item.get("affordability_score", 0.0))
            pref_scores.append(item.get("preference_score", 0.0))
            prices.append(float(p.get("price", 0.0)))
            brands.append(str(p.get("brand", "Unknown")))
            stock_status.append("✅ In Stock" if p.get("in_stock") else "❌ Out of Stock")

        # Create comprehensive figure
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # Main title
        query_text = summary.get("query", "Recommendations") if summary else "Recommendations"
        fig.suptitle(
            f"Fin-e Trip: {query_text}",
            fontsize=18,
            fontweight="bold",
            color="#1F77B4",
        )

        # Panel 1: Score breakdown (stacked bar)
        ax1 = fig.add_subplot(gs[0, :2])
        x_pos = np.arange(n)
        width = 0.7

        p1 = ax1.bar(x_pos, semantic_scores, width, label="Semantic (45%)", color="#FFB347", alpha=0.9)
        p2 = ax1.bar(
            x_pos, afford_scores, width, bottom=semantic_scores, label="Affordability (35%)", color="#90EE90", alpha=0.9
        )
        pref_bottom = [s + a for s, a in zip(semantic_scores, afford_scores)]
        p3 = ax1.bar(
            x_pos, pref_scores, width, bottom=pref_bottom, label="Preference (20%)", color="#87CEEB", alpha=0.9
        )

        ax1.set_ylabel("Score", fontweight="bold", fontsize=11)
        ax1.set_title("Score Composition per Product", fontweight="bold", fontsize=12)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"#{i+1}" for i in range(n)])
        ax1.legend(loc="upper right", fontsize=10)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")
        ax1.set_ylim(0, 1.2)

        # Panel 2: Budget info (text box)
        ax_budget = fig.add_subplot(gs[0, 2])
        ax_budget.axis("off")
        if summary:
            bal = summary.get("available_balance", 0)
            cred = summary.get("credit_limit", 0)
            total = bal + cred
            budget_text = f"💰 Budget Breakdown\n\nBalance: ${bal:,.0f}\nCredit: ${cred:,.0f}\nMax: ${total:,.0f}"
            ax_budget.text(
                0.1, 0.5, budget_text, fontsize=11, verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="#E8F4F8", alpha=0.8, pad=1),
                family="monospace",
            )

        # Panel 3: Final scores (horizontal bar)
        ax2 = fig.add_subplot(gs[1, :])
        colors_final = ["#2CA02C" if f >= 0.7 else "#FF7F0E" if f >= 0.6 else "#D62728" for f in final_scores]
        bars = ax2.barh(range(n), final_scores, color=colors_final, alpha=0.85, height=0.6)

        # Add price and brand labels
        for i, (bar, price, brand) in enumerate(zip(bars, prices, brands)):
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"  ${price:,.0f} ({brand})", va="center", fontsize=10, fontweight="bold")

        ax2.set_yticks(range(n))
        ax2.set_yticklabels([f"#{i+1} {products[i][:50]}" for i in range(n)], fontsize=10)
        ax2.set_xlabel("Final Score", fontweight="bold", fontsize=11)
        ax2.set_title("Product Rankings & Pricing", fontweight="bold", fontsize=12)
        ax2.set_xlim(0, 1.0)
        ax2.grid(axis="x", alpha=0.3, linestyle="--")

        # Panel 4: Affordability scores
        ax3 = fig.add_subplot(gs[2, 0])
        afford_colors = ["#2CA02C" if a > 0.8 else "#FF7F0E" if a > 0.5 else "#D62728" for a in afford_scores]
        ax3.bar(range(n), afford_scores, color=afford_colors, alpha=0.85)
        ax3.set_ylabel("Score", fontweight="bold", fontsize=10)
        ax3.set_title("Affordability Scores", fontweight="bold", fontsize=11)
        ax3.set_xticks(range(n))
        ax3.set_xticklabels([f"#{i+1}" for i in range(n)], fontsize=9)
        ax3.set_ylim(0, 1.05)
        ax3.grid(axis="y", alpha=0.3, linestyle="--")

        # Panel 5: Price distribution
        ax4 = fig.add_subplot(gs[2, 1])
        price_colors = ["#1f77b4" if p < 1000 else "#ff7f0e" if p < 3000 else "#d62728" for p in prices]
        ax4.bar(range(n), prices, color=price_colors, alpha=0.85)
        ax4.set_ylabel("Price ($)", fontweight="bold", fontsize=10)
        ax4.set_title("Product Prices", fontweight="bold", fontsize=11)
        ax4.set_xticks(range(n))
        ax4.set_xticklabels([f"#{i+1}" for i in range(n)], fontsize=9)
        ax4.grid(axis="y", alpha=0.3, linestyle="--")

        # Panel 6: Stock status (legend-like)
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis("off")
        stock_text = "📦 Stock Status\n\n"
        for i, status in enumerate(stock_status[:n]):
            stock_text += f"#{i+1}: {status}\n"
        ax5.text(
            0.05, 0.95, stock_text, fontsize=9, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#FFF8DC", alpha=0.8, pad=0.8),
        )

        # Save and display
        output_path = os.path.join(os.getcwd(), "fin_trip_recommendations.png")
        plt.savefig(output_path, dpi=120, bbox_inches="tight", facecolor="white")
        print(f"\n✅ Visualization saved: {output_path}")
        print(f"📊 Chart includes: score composition, rankings, affordability, pricing, and stock status.\n")
        plt.show()

    except Exception as e:
        print(f"⚠️ Visualization error: {e}")


if __name__ == "__main__":
    import uuid
    from qdrant_client.http.models import PointStruct
    
    sample_user_id = str(uuid.uuid4())
    sample_query = "Laptop for machine learning under 1500 with installments"

    logger.info("Starting search pipeline...")
    logger.info(f"Query: '{sample_query}'")
    
    try:
        client = get_qdrant_client()
        
        # Create test user fixture with known profile and financial data
        logger.info("Creating test user fixture...")
        
        # Test user profile
        test_user_profile = PointStruct(
            id=sample_user_id,
            vector=[0.0] * 384,  # Dummy vector for retrieval
            payload={
                "user_id": sample_user_id,
                "name": "Test User",
                "location": "San Francisco",
                "risk_tolerance": "Medium",
                "preferred_categories": ["Computers", "Smartphones", "Accessories"],
                "preferred_brands": ["Samsung", "Apple"],
            }
        )
        
        # Test user financial context
        test_user_financial = PointStruct(
            id=sample_user_id,
            vector=[0.0] * 256,  # Dummy vector for retrieval
            payload={
                "user_id": sample_user_id,
                "available_balance": 5000.0,
                "credit_limit": 10000.0,
                "current_debt": 2000.0,
                "eligible_installments": True,
            }
        )
        
        # Upsert test fixtures
        client.upsert(collection_name=USER_PROFILES_COLLECTION, points=[test_user_profile])
        client.upsert(collection_name=FINANCIAL_CONTEXTS_COLLECTION, points=[test_user_financial])
        logger.info(f"Test fixtures created for user_id={sample_user_id}")
        
        results = search_products(sample_user_id, sample_query, top_k=5, debug_mode=True)
        summary = {
            "query": sample_query,
            "available_balance": test_user_financial.payload.get("available_balance"),
            "credit_limit": test_user_financial.payload.get("credit_limit"),
        }
        _print_results(results, summary)
        
        if not results:
            logger.warning("No results found. Check:")
            logger.warning("  - Collection has data (run generate_and_insert_data.py)")
            logger.warning("  - User context has financial data")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise