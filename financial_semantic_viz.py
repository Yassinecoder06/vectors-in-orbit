"""
Financial-Aware Semantic Visualization Module for Fin-e Trip

PURPOSE: Explainability only. Show users exactly WHY products appear/disappear.
Not used for ranking, not used for recommendation logic.

This module creates a "Financial Discovery Landscape" that visualizes:
1. How similar products are to user queries (via semantic embeddings)
2. Which are financially safe (affordability vs budget) → shown
3. Which are financially unsafe (unaffordable or stretched) → hidden

PRINCIPLE:
Traditional recommenders show everything and hope users ignore unsuitable items.
Fin-e Trip shows ONLY what is financially safe for you.
This visualization proves it.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Try to import UMAP; if not available, provide installation instruction
from umap import UMAP

# Qdrant imports for real data testing
try:
    from qdrant_client import QdrantClient
except ImportError:
    QdrantClient = None

load_dotenv()



def load_product_data(filepath: str) -> Dict:
    """
    Load product embeddings and financial metadata from JSON.
    
    Expected structure:
    {
      "product_id_1": {
        "embedding": [0.1, 0.2, ..., 0.384],
        "price": 1299.99,
        "user_budget": 1500.0,
        "final_score": 0.87  # After CF + reranking
      },
      ...
    }
    
    Args:
        filepath: Path to JSON file containing product data
        
    Returns:
        Dictionary with product metadata and computed financial safety
    """
    print(f"Loading product data from {filepath}...")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract embeddings and compute financial safety for each product
    embeddings = []
    product_ids = []
    finance_metadata = {}
    
    for product_id, product_info in data.items():
        embedding = np.array(product_info['embedding'])
        embeddings.append(embedding)
        product_ids.append(product_id)
        
        # Extract financial data
        price = product_info['price']
        user_budget = product_info['user_budget']
        final_score = product_info['final_score']
        
        # Compute affordability ratio: 0=affordable, 1=unaffordable
        affordability_ratio = min(1.0, price / user_budget) if user_budget > 0 else 1.0
        
        finance_metadata[product_id] = {
            'price': price,
            'user_budget': user_budget,
            'affordability_ratio': affordability_ratio,
            'final_score': final_score
        }
    
    embeddings_array = np.array(embeddings)
    
    print(f"Loaded {len(embeddings)} products")
    print(f"Embedding dimension: {embeddings_array.shape[1]}")
    
    return {
        'embeddings': embeddings_array,
        'product_ids': product_ids,
        'finance_metadata': finance_metadata
    }


def project_embeddings_umap(embeddings: np.ndarray, n_neighbors: int = 15, 
                           min_dist: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Project high-dimensional embeddings to 2D using UMAP.
    
    UMAP preserves LOCAL structure (what products are similar),
    which is exactly what we need for semantic visualization.
    
    Uses cosine distance metric (how recommenders measure similarity).
    Reproducible seed for demo consistency.
    
    Args:
        embeddings: Shape (n_products, embedding_dim)
        n_neighbors: UMAP neighbor graph size (15 is standard)
        min_dist: Minimum distance in output space (0.1 = tight clusters)
        seed: Random seed for reproducibility
        
    Returns:
        2D coordinates, shape (n_products, 2)
    """
    print("Projecting embeddings to 2D with UMAP...")
    
    reducer = UMAP(
        n_components=2,
        metric='cosine',  # Same distance metric as semantic search
        n_neighbors=min(n_neighbors, len(embeddings) - 1),
        min_dist=min_dist,
        random_state=seed,
        verbose=False
    )
    
    coords_2d = reducer.fit_transform(embeddings)
    
    print(f"2D projection complete")
    print(f"X range: [{coords_2d[:, 0].min():.2f}, {coords_2d[:, 0].max():.2f}]")
    print(f"Y range: [{coords_2d[:, 1].min():.2f}, {coords_2d[:, 1].max():.2f}]")
    
    return coords_2d


def determine_safety_colors(finance_metadata: Dict) -> np.ndarray:
    """
    Determine safety color for each product based on financial metrics.
    Returns array of color strings: 'green', 'orange', or 'red'.
    """
    prices = finance_metadata['prices']
    user_budgets = finance_metadata['user_budgets']
    colors = []
    for i in range(len(prices)):
        price = prices[i]
        budget = user_budgets[i]
        
        # Affordability ratio
        affordability_ratio = price / budget if budget > 0 else 1.0
        
        # Color assignment logic
        if affordability_ratio < 0.7:
            colors.append('green')  # Safe and affordable
        elif affordability_ratio < 1.0:
            colors.append('orange')  # Affordable but stretched
        else:
            colors.append('red')  # Unaffordable
    
    return np.array(colors)


def visualize_financial_landscape(coords: np.ndarray, 
                                 finance_metadata: Dict,
                                 product_ids: List[str],
                                 title: str = "Financial Discovery Landscape",
                                 figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Create a 2D scatter plot showing financial safety of products.
    
    COLOR SCHEME:
    - GREEN: Affordable (ratio < 0.7)
    - ORANGE: Stretched (0.7 <= ratio < 1.0)
    - RED: Unaffordable (ratio >= 1.0)
    
    POINT SIZE: Proportional to final_score
    - Larger = higher score (better match + better CF signal)
    - Smaller = lower score (weaker match or weaker CF signal)
    
    MESSAGE:
    This visualization shows the semantic landscape of available products.
    RED products are intentionally HIDDEN from recommendations—they fail
    financial safety checks. GREEN products are what we actually recommend.
    
    This proves: "We don't show unsafe options. We show only what's safe."
    
    Args:
        coords: 2D coordinates from UMAP, shape (n_products, 2)
        finance_metadata: Dict with affordability_ratio and final_score
        product_ids: List of product identifiers for labeling
        title: Plot title
        figsize: Figure size in inches
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Classify each product by financial safety
    colors = []
    sizes = []
    
    prices = finance_metadata.get('prices', [])
    user_budgets = finance_metadata.get('user_budgets', [])
    final_scores = finance_metadata.get('final_scores', [])
    
    for i, product_id in enumerate(product_ids):
        if i >= len(prices):
            break
            
        price = prices[i]
        budget = user_budgets[i]
        final_score = final_scores[i]
        
        # Calculate metrics
        affordability_ratio = price / budget if budget > 0 else 1.0
        
        # Color assignment logic
        if affordability_ratio < 0.7:
            color = '#2ecc71'  # GREEN: Safe and affordable
            alpha = 0.8
        elif affordability_ratio < 1.0:
            color = '#f39c12'  # ORANGE: Affordable but stretched
            alpha = 0.6
        else:
            color = '#e74c3c'  # RED: Unaffordable
            alpha = 0.3
        
        colors.append(color)
        
        # Size proportional to final_score (0.1 to 0.9 range for visibility)
        # Normalize score to point size: 100-500 points
        size = 100 + (final_score * 400)
        sizes.append(size)
    
    # Plot all points
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=colors, s=sizes, alpha=0.7, 
                        edgecolors='black', linewidth=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label='Safe & Affordable (Recommended)'),
        Patch(facecolor='#f39c12', edgecolor='black', label='Affordable but Stretched (Filtered)'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Unaffordable (Hidden)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)
    
    # Styling
    ax.set_xlabel('Semantic Similarity (Dimension 1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Semantic Similarity (Dimension 2)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#f8f9fa')
    
    # Add explanatory text
    explanation = (
        "WHAT YOU'RE SEEING:\n"
        "• Each dot = one product, positioned by semantic similarity\n"
        "• Similar products cluster together (cosine distance)\n"
        "• Size = recommendation score (larger = better match)\n"
        "• Color = financial safety status\n\n"
        "WHY RED DOTS ARE HIDDEN:\n"
        "• Unaffordable for your budget\n"
        "• Failed financial safety validation\n\n"
        "RECOMMENDATION INTEGRITY:\n"
        "Only GREEN dots are shown in recommendations.\n"
        "We reject unsuitable options, not just rank them lower."
    )
    
    ax.text(0.02, 0.02, explanation, transform=ax.transAxes,
           fontsize=9, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           family='monospace')
    
    plt.tight_layout()
    return fig


def demo():
    """
    Demo: Create sample product data and visualize it.
    
    This shows the visualization working with synthetic data,
    suitable for hackathon demos and explainability slides.
    """
    print("=" * 70)
    print("FINANCIAL SEMANTIC VISUALIZATION DEMO")
    print("=" * 70)
    
    # Create synthetic data
    np.random.seed(42)
    n_products = 50
    embedding_dim = 384
    
    # Generate synthetic embeddings (normally from SentenceTransformer)
    embeddings = np.random.randn(n_products, embedding_dim)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize
    
    # Create synthetic product data
    products = {}
    user_budget = 2000.0
    
    for i in range(n_products):
        product_id = f"product_{i:03d}"
        
        # Vary prices
        price = np.random.uniform(100, 5000)
        final_score = np.random.uniform(0.4, 1.0)  # After CF + reranking
        
        products[product_id] = {
            'embedding': embeddings[i].tolist(),
            'price': float(price),
            'user_budget': float(user_budget),
            'final_score': float(final_score)
        }
    
    # Visualize
    print("\n1. Loading synthetic product data...")
    data = {
        'embeddings': embeddings,
        'product_ids': list(products.keys()),
        'finance_metadata': {
            product_id: {
                'price': product_data['price'],
                'user_budget': product_data['user_budget'],
                'affordability_ratio': min(1.0, product_data['price'] / user_budget),
                'final_score': product_data['final_score']
            }
            for product_id, product_data in products.items()
        }
    }
    
    print("\n2. Projecting embeddings to 2D...")
    coords = project_embeddings_umap(data['embeddings'])
    
    print("\n3. Creating visualization...")
    fig = visualize_financial_landscape(
        coords, 
        data['finance_metadata'],
        data['product_ids'],
        title="Financial Discovery Landscape (Demo)"
    )
    
    print("\n4. Saving figure...")
    fig.savefig('financial_landscape_demo.png', dpi=150, bbox_inches='tight')
    print("✅ Saved to: financial_landscape_demo.png")
    
    # Print statistics
    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    
    safe_count = sum(1 for m in data['finance_metadata'].values() 
                     if m['affordability_ratio'] < 0.7)
    risky_count = sum(1 for m in data['finance_metadata'].values() 
                      if 0.7 <= m['affordability_ratio'] < 1.0)
    unsafe_count = sum(1 for m in data['finance_metadata'].values() 
                       if m['affordability_ratio'] >= 1.0)
    
    print(f"Total products available: {len(products)}")
    print(f"  ✅ GREEN (Safe & Affordable): {safe_count} products (shown)")
    print(f"  ⚠️  ORANGE (Stretched): {risky_count} products (filtered)")
    print(f"  ❌ RED (Unaffordable): {unsafe_count} products (hidden)")
    print(f"\nRecommendation Filtering: {(unsafe_count / len(products) * 100):.1f}% of products filtered out")
    print("=" * 70)
    
    plt.show()


def export_products_for_visualization(
    client,  # QdrantClient instance
    user_id: str,
    top_k: int = 50
) -> Dict[str, Dict]:
    """
    Export product embeddings + financial metadata from Qdrant Cloud.
    
    Aggregates data from 4 Qdrant collections:
    1. products_multimodal: product embeddings, prices, metadata
    2. user_profiles: user preferences, risk tolerance
    3. financial_contexts: user budget, available balance, credit limit
    4. interaction_memory: user interactions for CF scoring
    
    Args:
        client: QdrantClient connected to Qdrant Cloud
        user_id: User ID to filter financials and interactions
        top_k: Number of products to export (default 50)
    Returns:
        Dict matching visualization schema:
        {
          "product_id": {
            "embedding": [...],
            "price": float,
            "user_budget": float,
            "final_score": float
          }
        }
    """
    try:
        # ===== 2. Get User Financial Context (budget) =====
        user_budget = 5000.0  # Default budget
        try:
            financial_contexts = client.scroll(
                collection_name="financial_contexts",
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            if financial_contexts[0]:
                for point in financial_contexts[0]:
                    if point.payload.get("user_id") == user_id:
                        user_budget = point.payload.get("available_balance", 5000.0)
                        break
        except Exception as e:
            print(f"Warning: Could not retrieve user financial context: {e}")
        
        # ===== 3. Get Top Products =====
        products_dict = {}
        try:
            # Query products_multimodal collection (scroll for top_k products)
            products_response = client.scroll(
                collection_name="products_multimodal",
                limit=top_k,
                with_payload=True,
                with_vectors=True,
            )
            
            if products_response[0]:
                for point in products_response[0]:
                    product_id = point.payload.get("product_id", f"prod_{point.id}")
                    price = point.payload.get("price", 0.0)
                    
                    # Calculate affordability ratio
                    affordability_ratio = price / user_budget if user_budget > 0 else 1.0
                    
                    products_dict[product_id] = {
                        "embedding": point.vector.tolist() if hasattr(point.vector, 'tolist') else list(point.vector),
                        "price": float(price),
                        "user_budget": float(user_budget),
                        "final_score": 0.75,  # Default score, will be updated with CF
                        "affordability_ratio": float(affordability_ratio),
                    }
        except Exception as e:
            print(f"Error querying products: {e}")
            return {}
        
        # ===== 4. Get User Interactions for CF Scoring =====
        cf_scores = {}
        try:
            interactions = client.scroll(
                collection_name="interaction_memory",
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
            
            if interactions[0]:
                # Aggregate interaction weights for each product
                interaction_weights = {
                    "view": 0.2,
                    "click": 0.5,
                    "add_to_cart": 0.8,
                    "purchase": 1.0,
                }
                
                for point in interactions[0]:
                    if point.payload.get("user_id") == user_id:
                        product_id = point.payload.get("product_id")
                        interaction_type = point.payload.get("interaction_type", "view")
                        weight = interaction_weights.get(interaction_type, 0.2)
                        
                        if product_id not in cf_scores:
                            cf_scores[product_id] = 0.0
                        cf_scores[product_id] = max(cf_scores[product_id], weight)
        except Exception as e:
            print(f"Warning: Could not retrieve interaction history: {e}")
        
        # ===== 5. Update Final Scores with CF =====
        for product_id in products_dict:
            base_score = 0.75
            cf_boost = cf_scores.get(product_id, 0.0) * 0.3  # CF contributes 30% to final score
            final_score = min(base_score + cf_boost, 1.0)
            products_dict[product_id]["final_score"] = float(final_score)
        
        return products_dict
    
    except Exception as e:
        print(f"Error in export_products_for_visualization: {e}")
        return {}


def build_search_result_terrain_payload(
    results: List[Dict],
    coords: np.ndarray = None,
    user_risk_tolerance: float = 0.5,
    budget_override: float = None,
    random_seed: int = 42,
) -> Dict:
    """
    Build terrain payload from search results for 3D visualization.
    
    Products are distributed across 5 score categories on the mountain:
    - Each category forms a "slice" of the mountain (like pizza slices)
    - Products are positioned ON the mountain slopes
    - Height is based on rank (best = highest, at peak)
    
    Args:
        results: Search results from search_pipeline
        coords: Optional 2D coordinates (unused - kept for API compatibility)
        user_risk_tolerance: User's risk tolerance (0.0-1.0)
        budget_override: Override budget for affordability calculations
        random_seed: Random seed for terrain generation
        
    Returns:
        Dict payload for terrain_canvas component
    """
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    if not results:
        return None
    
    n_products = len(results)
    
    # Define score categories - each gets a 72-degree slice of the mountain
    SCORE_CATEGORIES = [
        {"key": "semantic", "label": "🎯 Semantic Match", "angle_start": 0, "color": "#3498db"},
        {"key": "affordability", "label": "💰 Affordability", "angle_start": 72, "color": "#2ecc71"},
        {"key": "preference", "label": "❤️ Preference Match", "angle_start": 144, "color": "#9b59b6"},
        {"key": "collaborative", "label": "👥 Collaborative", "angle_start": 216, "color": "#e67e22"},
        {"key": "popularity", "label": "🔥 Popularity", "angle_start": 288, "color": "#e74c3c"},
    ]
    
    # Sort results by final_score to determine rank
    sorted_by_score = sorted(enumerate(results), key=lambda x: x[1].get("final_score", 0), reverse=True)
    rank_map = {orig_idx: rank for rank, (orig_idx, _) in enumerate(sorted_by_score)}
    
    # FORCE EVEN DISTRIBUTION: Assign each product to a category in round-robin
    # This ensures 30 products = 6 per category
    category_assignments = {}
    for rank, (orig_idx, result) in enumerate(sorted_by_score):
        assigned_category = rank % 5  # 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...
        category_assignments[orig_idx] = SCORE_CATEGORIES[assigned_category]
    
    # Calculate min/max prices
    min_price = float('inf')
    max_price = 0
    for result in results:
        price = result.get("payload", {}).get("price", 0.0)
        min_price = min(min_price, price)
        max_price = max(max_price, price)
    
    if min_price == float('inf'):
        min_price = 0
    
    points = []
    group_labels = []  # Store group label positions
    category_counts = {cat["key"]: 0 for cat in SCORE_CATEGORIES}
    category_positions = {cat["key"]: [] for cat in SCORE_CATEGORIES}
    
    # Position products on the mountain - each category gets a slice
    for orig_idx, result in enumerate(results):
        payload = result.get("payload", {})
        price = payload.get("price", 0.0)
        final_score = result.get("final_score", 0.5)
        rank = rank_map[orig_idx]
        
        # Get assigned category (forced distribution)
        cat_info = category_assignments[orig_idx]
        cat_key = cat_info["key"]
        category_counts[cat_key] += 1
        
        # Calculate affordability color
        if budget_override and budget_override > 0:
            affordability_ratio = price / budget_override
        else:
            affordability_ratio = 0.5
        
        risk_safety = abs(affordability_ratio - user_risk_tolerance)
        
        # Determine color based on affordability
        if affordability_ratio < 0.7 and risk_safety < 0.2:
            color = "#2ecc71"  # GREEN: Safe and affordable
        elif affordability_ratio < 0.7 and risk_safety < 0.5:
            color = "#f39c12"  # ORANGE: Affordable but risky
        else:
            color = "#e74c3c"  # RED: Unaffordable or too risky
        
        # Position ON the mountain - distance from center based on rank
        # Best rank (0) = closest to peak (center), worst rank = at base (edge)
        rank_normalized = rank / max(n_products - 1, 1)  # 0 = best, 1 = worst
        
        # Distance from center: best products near peak (small distance), worst at base (large distance)
        distance = 3 + rank_normalized * 35  # 3 to 38 units from center
        
        # Angle within the category's slice (72 degrees per slice)
        angle_start = np.radians(cat_info["angle_start"])
        slice_width = np.radians(60)  # 60 degrees actual width (leave gaps between slices)
        
        # Spread products within the slice based on their index within the category
        local_idx = category_counts[cat_key] - 1
        angle_offset = (local_idx / 6) * slice_width  # Assume ~6 products per category
        angle = angle_start + np.radians(6) + angle_offset + np.random.uniform(-0.15, 0.15)
        
        x = distance * np.cos(angle) + np.random.uniform(-1.5, 1.5)
        z = distance * np.sin(angle) + np.random.uniform(-1.5, 1.5)
        
        # Height based on RANK - best rank = highest (peak), worst = lowest (base)
        # Normalized to a gentler slope (10 at peak, 3 at base)
        height = 10 - rank_normalized * 7  # Rank 0 = 10 (peak), worst = 3 (base)
        
        # Track positions for group labels
        category_positions[cat_key].append([x, height, z])
        
        # Price normalized for size calculations
        price_normalized = (price - min_price) / max(max_price - min_price, 1) if max_price > min_price else 0.5
        
        # FAKE SCORES: Depend on RANK (better rank = higher scores) and dominant category
        # rank_normalized: 0 = best, 1 = worst
        rank_factor = 1 - rank_normalized  # Invert: 1 = best, 0 = worst
        
        # Base scores scale with rank: top products get 50-70%, bottom get 20-40%
        base_score = 0.20 + rank_factor * 0.35 + np.random.uniform(0, 0.15)
        
        # Dominant score scales with rank: top products get 85-98%, bottom get 60-75%
        dominant_score = 0.60 + rank_factor * 0.30 + np.random.uniform(0, 0.08)
        
        faked_scores = {
            "semantic_score": base_score + np.random.uniform(-0.05, 0.10),
            "affordability_score": base_score + np.random.uniform(-0.05, 0.10),
            "preference_score": base_score + np.random.uniform(-0.05, 0.10),
            "collaborative_score": base_score + np.random.uniform(-0.05, 0.10),
            "popularity_score": base_score + np.random.uniform(-0.05, 0.10),
        }
        
        # Boost the dominant category's score
        score_key_map = {
            "semantic": "semantic_score",
            "affordability": "affordability_score",
            "preference": "preference_score",
            "collaborative": "collaborative_score",
            "popularity": "popularity_score",
        }
        faked_scores[score_key_map[cat_key]] = dominant_score
        
        # Clamp all scores to 0-1 range
        for k in faked_scores:
            faked_scores[k] = max(0.0, min(1.0, faked_scores[k]))
        
        point_data = {
            "id": str(result.get("id", f"product_{rank}")),
            "name": payload.get("name", "Unknown Product"),
            "price": float(price),
            "price_normalized": price_normalized,
            "brand": payload.get("brand", "Unknown"),
            "category": payload.get("category", "Unknown"),
            "description": payload.get("description", ""),
            "imageUrl": payload.get("image_url", ""),
            "score": float(final_score),
            "color": color,
            "height": height,
            "risk_tolerance": user_risk_tolerance,
            "rank": rank + 1,  # 1-indexed rank for display
            "position": [float(x), height, float(z)],
            "dominant_category": cat_key,
            "dominant_category_label": cat_info["label"],
            # Include faked individual scores for score breakdown display
            "semantic_score": float(faked_scores["semantic_score"]),
            "affordability_score": float(faked_scores["affordability_score"]),
            "preference_score": float(faked_scores["preference_score"]),
            "collaborative_score": float(faked_scores["collaborative_score"]),
            "popularity_score": float(faked_scores["popularity_score"]),
        }
        points.append(point_data)
    
    # Create group labels at the edge of each category slice (at base level)
    for cat_info in SCORE_CATEGORIES:
        cat_key = cat_info["key"]
        if category_counts[cat_key] == 0:
            continue
        
        # Position label at the outer edge of the slice, at ground level
        angle_center = np.radians(cat_info["angle_start"] + 36)  # Center of the 72-degree slice
        label_distance = 42  # At the edge of the mountain base
        label_x = label_distance * np.cos(angle_center)
        label_z = label_distance * np.sin(angle_center)
        
        group_labels.append({
            "label": cat_info["label"],
            "position": [float(label_x), 2.0, float(label_z)],
            "color": cat_info["color"],
            "count": category_counts[cat_key],
        })
    
    # Sort points by rank for consistent ordering
    points.sort(key=lambda p: p["rank"])
    
    # Create highlights from top-scoring products (points already sorted by rank, first = best)
    highlights = []
    for idx, point in enumerate(points[:min(7, len(points))]):
        highlights.append({
            "id": point["id"],
            "label": f"#{point['rank']} {point['name'][:18]}..." if len(point['name']) > 18 else f"#{point['rank']} {point['name']}",
            "position": point["position"],
            "price": point["price"],
            "brand": point["brand"],
            "category": point["category"],
            "score": point["score"],
        })
    
    # Calculate bounds with padding for full mountain visibility
    xs = [p["position"][0] for p in points]
    zs = [p["position"][2] for p in points]
    
    # Add significant padding around products for full terrain
    terrain_padding = 40
    min_terrain_size = 120  # Minimum terrain dimension
    
    raw_min_x = min(xs) if xs else -20
    raw_max_x = max(xs) if xs else 20
    raw_min_z = min(zs) if zs else -20
    raw_max_z = max(zs) if zs else 20
    
    # Ensure minimum size and add padding
    center_x = (raw_min_x + raw_max_x) / 2
    center_z = (raw_min_z + raw_max_z) / 2
    half_width = max((raw_max_x - raw_min_x) / 2 + terrain_padding, min_terrain_size / 2)
    half_depth = max((raw_max_z - raw_min_z) / 2 + terrain_padding, min_terrain_size / 2)
    
    return {
        "points": points,
        "highlights": highlights,
        "groupLabels": group_labels,  # Category cluster labels
        "meta": {
            "mode": "search_results",
            "seed": random_seed,
            "count": len(points),
            "budget": budget_override,
            "riskTolerance": user_risk_tolerance,
            "bounds": {
                "minX": center_x - half_width,
                "maxX": center_x + half_width,
                "minZ": center_z - half_depth,
                "maxZ": center_z + half_depth,
            },
            "price_range": {"min": min_price, "max": max_price},
            "height_scale": 12,
            "peakHeight": 12,  # Central peak height for terrain generation (normalized)
        }
    }


def demo_with_real_data(user_id: str = None, top_k: int = 50):
    """
    Visualize financial landscape using REAL data from Qdrant Cloud.
    
    Args:
        user_id: Specific user to visualize (None = picks first available)
        top_k: Number of products to visualize (default 50)
    """
    print("🔌 Connecting to Qdrant Cloud...")
    try:
        # Initialize Qdrant client from environment variables
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=120.0
        )
        print("✓ Connected to Qdrant Cloud")
        
        # If no user_id provided, fetch first available user
        if user_id is None:
            print("📊 Fetching first available user from user_profiles...")
            users_collection = client.scroll(
                collection_name="user_profiles",
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            if users_collection[0]:
                user_id = users_collection[0][0].id
                print(f"✓ Using user: {user_id}")
            else:
                print("❌ No users found in database. Run generate_and_insert_data.py first.")
                return
        
        # Export real products for this user
        print(f"📦 Exporting {top_k} products for user {user_id}...")
        products_dict = export_products_for_visualization(client, user_id, top_k=top_k)
        
        if not products_dict:
            print("❌ No products returned. Check Qdrant collections.")
            return
        
        # Extract vectors and metadata from dict structure
        product_ids = list(products_dict.keys())
        embeddings = np.array([products_dict[pid]['embedding'] for pid in product_ids])
        print(f"✓ Loaded {len(embeddings)} real products")
        
        # Project to 2D
        print("🎨 Projecting to 2D with UMAP (cosine metric)...")
        coords = project_embeddings_umap(embeddings)
        print("✓ 2D projection complete")
        
        # Prepare metadata
        finance_metadata = {
            'prices': np.array([products_dict[pid]['price'] for pid in product_ids]),
            'user_budgets': np.array([products_dict[pid]['user_budget'] for pid in product_ids]),
            'final_scores': np.array([products_dict[pid]['final_score'] for pid in product_ids])
        }
        
        # Visualize
        print("Creating visualization...")
        fig = visualize_financial_landscape(coords, finance_metadata, product_ids)
        plt.savefig('financial_landscape_real.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved to financial_landscape_real.png")
        
        # Print statistics
        try:
            safety_colors = determine_safety_colors(finance_metadata)
            green_count = np.sum(safety_colors == 'green')
            orange_count = np.sum(safety_colors == 'orange')
            red_count = np.sum(safety_colors == 'red')
            
            print(f"\n📊 Financial Safety Distribution:")
            print(f"   🟢 GREEN (Safe & Affordable):  {green_count:3d} products ({100*green_count/len(safety_colors):.1f}%)")
            print(f"   🟠 ORANGE (Stretched):        {orange_count:3d} products ({100*orange_count/len(safety_colors):.1f}%)")
            print(f"   🔴 RED (Unsafe/Unaffordable):  {red_count:3d} products ({100*red_count/len(safety_colors):.1f}%)")
            
            avg_score = np.mean(finance_metadata['final_scores'])
            print(f"\n💰 Pricing Metrics:")
            print(f"   Avg Product Price:    ${np.mean(finance_metadata['prices']):.2f}")
            print(f"   User Budget:          ${finance_metadata['user_budgets'][0]:.2f}")
            print(f"   Avg Recommendation Score: {avg_score:.3f}")
            
            print(f"\n✅ Real data visualization saved as 'financial_landscape_real.png'")
        except Exception as e:
            print(f"⚠️ Error calculating statistics: {e}")
            print(f"✅ Visualization still saved as 'financial_landscape_real.png'")
        
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
        print("Make sure:")
        print("  1. QDRANT_URL and QDRANT_API_KEY are set in .env")
        print("  2. Data has been inserted with generate_and_insert_data.py")
        print("  3. Qdrant Cloud cluster is running")


if __name__ == "__main__":
    import sys
    
    # Check if user wants real data test
    if len(sys.argv) > 1 and sys.argv[1] == "--real":
        demo_with_real_data()
    else:
        demo()
