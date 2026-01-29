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
    
    Args:
        results: Search results from search_pipeline
        coords: Optional 2D coordinates (if None, generates grid positions)
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
    
    points = []
    
    # Generate positions if coords not provided
    n_products = len(results)
    if coords is None:
        # Create a grid-like distribution with some randomness
        grid_size = int(np.ceil(np.sqrt(n_products)))
        positions = []
        for i in range(n_products):
            row = i // grid_size
            col = i % grid_size
            # Normalize to -1 to 1 range with some jitter
            x = (col / max(grid_size - 1, 1)) * 2 - 1 + np.random.uniform(-0.1, 0.1)
            y = (row / max(grid_size - 1, 1)) * 2 - 1 + np.random.uniform(-0.1, 0.1)
            positions.append([x, y])
        coords = np.array(positions)
    
    # Normalize coords to terrain scale (-50 to 50)
    if coords.shape[0] > 0:
        coords_min = coords.min(axis=0)
        coords_max = coords.max(axis=0)
        coords_range = coords_max - coords_min
        coords_range[coords_range == 0] = 1  # Avoid division by zero
        normalized_coords = (coords - coords_min) / coords_range * 80 - 40  # Scale to -40 to 40
    else:
        normalized_coords = coords
    
    min_price = float('inf')
    max_price = 0
    
    for i, result in enumerate(results):
        payload = result.get("payload", {})
        price = payload.get("price", 0.0)
        min_price = min(min_price, price)
        max_price = max(max_price, price)
        final_score = result.get("final_score", 0.5)
        
        # Calculate affordability color
        if budget_override and budget_override > 0:
            affordability_ratio = price / budget_override
        else:
            affordability_ratio = 0.5  # Default to medium if no budget
        
        risk_safety = abs(affordability_ratio - user_risk_tolerance)
        
        # Determine color based on affordability
        if affordability_ratio < 0.7 and risk_safety < 0.2:
            color = "#2ecc71"  # GREEN: Safe and affordable
        elif affordability_ratio < 0.7 and risk_safety < 0.5:
            color = "#f39c12"  # ORANGE: Affordable but risky
        else:
            color = "#e74c3c"  # RED: Unaffordable or too risky
        
        # Get position from coords
        if i < len(normalized_coords):
            x, z = normalized_coords[i]
        else:
            x, z = np.random.uniform(-40, 40), np.random.uniform(-40, 40)
        
        # Calculate height based on final_score (higher score = higher on terrain)
        height = float(final_score) * 18
        
        # Price normalized for size calculations
        price_normalized = (price - min_price) / max(max_price - min_price, 1) if max_price > min_price else 0.5
        
        point_data = {
            "id": str(result.get("id", f"product_{i}")),
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
            "position": [float(x), height, float(z)]  # [x, y, z] array format
        }
        points.append(point_data)
    
    # Create highlights from top-scoring products
    sorted_points = sorted(points, key=lambda p: p["score"], reverse=True)
    highlights = []
    for idx, point in enumerate(sorted_points[:min(7, len(sorted_points))]):
        highlights.append({
            "id": point["id"],
            "label": f"#{idx + 1} {point['name'][:20]}..." if len(point['name']) > 20 else f"#{idx + 1} {point['name']}",
            "position": point["position"],
            "price": point["price"],
            "brand": point["brand"],
            "category": point["category"],
            "score": point["score"],
        })
    
    # Calculate bounds
    xs = [p["position"][0] for p in points]
    zs = [p["position"][2] for p in points]
    
    return {
        "points": points,
        "highlights": highlights,
        "meta": {
            "mode": "search_results",
            "seed": random_seed,
            "count": len(points),
            "budget": budget_override,
            "riskTolerance": user_risk_tolerance,
            "bounds": {
                "minX": min(xs) if xs else -20,
                "maxX": max(xs) if xs else 20,
                "minZ": min(zs) if zs else -20,
                "maxZ": max(zs) if zs else 20,
            },
            "price_range": {"min": min_price, "max": max_price},
            "height_scale": 18,
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
