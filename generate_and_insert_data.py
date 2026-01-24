import os
import sys
import json
import random
import uuid
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from faker import Faker
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv

load_dotenv()

# Initialize Faker
fake = Faker()

# Initialize Embedding Model (SentenceTransformer, 384-dim) with GPU support
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print("Loading embedding model (first time will download ~90MB)...")
import os
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # Increase timeout to 5 minutes
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)  # Smaller, faster model (384D)
print("✅ Model loaded!")

def get_qdrant_client() -> QdrantClient:
    """Initialize Qdrant Client from environment variables."""
    qdrant_url = os.environ.get("QDRANT_URL")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        print("Error: QDRANT_URL and QDRANT_API_KEY must be set.")
        sys.exit(1)

    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

def generate_random_vector(size: int) -> List[float]:
    """Generate a random normalized vector (fallback for non-text / financial data)."""
    vec = np.random.rand(size)
    return vec.tolist()

def generate_text_embedding(text: str) -> List[float]:
    """Generate a real embedding using SentenceTransformer (384 dim)."""
    vec = embedding_model.encode(text, device=DEVICE)
    return vec.tolist()


def load_product_data(filepath: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load product data from JSON. If limit is None, load all products."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # If limit is specified and data exceeds it, sample randomly
            if limit and len(data) > limit:
                return random.sample(data, limit)
            return data
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return []

def insert_products(client: QdrantClient, products_source: List[Dict]) -> List[str]:
    """
    Insert product data from JSON source into Qdrant.
    Uses only real products from products_payload.json (no synthetic data).
    Returns list of product IDs for interaction generation.
    """
    collection_name = "products_multimodal"
    vector_size = 384
    points = []
    product_ids = []
    texts_to_embed = []
    payloads = []

    print(f"Processing {collection_name}...")

    if not products_source:
        print(f"❌ Error: No products loaded from products_payload.json")
        return []

    # Use ALL products from the source (no synthetic fallback)
    for src in products_source:
        product_id = src.get("product_id", str(uuid.uuid4()))
        name = src.get("name", "Unknown Product")
        categories = src.get("categories", ["General"])
        if not isinstance(categories, list) or not categories:
            categories = [str(categories)] if categories else ["General"]
        primary_category = categories[0]
        brand = src.get("brand", "Unknown")
        price = float(src.get("price", 0.0))
        in_stock = src.get("in_stock", True)
        region = src.get("region", "Unknown")
        image_url = src.get("image_url", f"https://example.com/images/{product_id}.jpg")
        monthly_installment = src.get("monthly_installment", round(price / 12, 2))
        description = src.get("description", "No Available Description")

        # Create rich text representation for embedding
        text_to_embed = f"{name} {description} {' '.join(categories)} {brand} {region}"
        texts_to_embed.append(text_to_embed)

        payload = {
            "product_id": product_id,
            "name": name,
            "description": description,
            "categories": categories,
            "brand": brand,
            "price": price,
            "monthly_installment": monthly_installment,
            "in_stock": in_stock,
            "region": region,
            "image_url": image_url,
        }
        payloads.append(payload)
        product_ids.append(product_id)

    # Batch encode all texts at once (much faster)
    print(f"Encoding {len(texts_to_embed)} product embeddings...")
    embeddings = embedding_model.encode(texts_to_embed, device=DEVICE, show_progress_bar=True, convert_to_numpy=True)
    
    # Build points with pre-computed embeddings
    for i, (product_id, embedding, payload) in enumerate(zip(product_ids, embeddings, payloads)):
        points.append(PointStruct(
            id=i,  # Use index as ID for stability
            vector=embedding.tolist(),
            payload=payload
        ))

    # Insert in batches to avoid server disconnection
    batch_size = 100  # Reduced from 500 for Qdrant Cloud stability
    print(f"Upserting {len(points)} points in batches of {batch_size}...")
    for batch_start in range(0, len(points), batch_size):
        batch_end = min(batch_start + batch_size, len(points))
        batch = points[batch_start:batch_end]
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(collection_name=collection_name, points=batch)
                print(f"  ✅ Upserted points {batch_start} to {batch_end}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    print(f"  ⚠️  Batch {batch_start}-{batch_end} failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Batch {batch_start}-{batch_end} failed after {max_retries} attempts")
                    raise
    
    print(f"✅ Inserted {len(points)} items into '{collection_name}'")

    # Build quick lookup map product_id -> payload for downstream correlation
    product_map: Dict[str, Dict[str, Any]] = {p['product_id']: p for p in payloads}
    return product_ids, product_map

def insert_users(client: QdrantClient, product_map: Dict[str, Dict[str, Any]], count: int = 20) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Generate and insert user profiles correlated to product data.

    Returns tuple(user_ids, user_profile_map)
    """
    collection_name = "user_profiles"
    vector_size = 384
    points = []
    user_ids = []
    texts_to_embed = []
    payloads = []

    print(f"Processing {collection_name}...")

    # Precompute available categories/brands from product_map
    categories = []
    if product_map:
        for p in product_map.values():
            cats = p.get('categories', [])
            if isinstance(cats, list):
                categories.extend(cats)
            elif cats:
                categories.append(str(cats))
    brands = [p.get('brand', '') for p in product_map.values()] if product_map else []

    for _ in range(count):
        user_uuid = str(uuid.uuid4())
        user_ids.append(user_uuid)

        # Choose preferred categories/brands from real product distribution when available
        preferred_categories = random.sample(categories, k=2) if len(set(categories)) >= 2 else ([random.choice(categories)] if categories else ["General"])
        preferred_brands = random.sample(brands, k=2) if len(set(brands)) >= 2 else ([random.choice(brands)] if brands else [fake.company()])

        payload = {
            "user_id": user_uuid,
            "name": fake.name(),
            "location": fake.city(),
            "risk_tolerance": random.choice(["Low", "Medium", "High"]),
            "preferred_categories": preferred_categories,
            "preferred_brands": preferred_brands
        }

        # Create user profile text for embedding
        text_to_embed = f"User interested in {' '.join(payload['preferred_categories'])} and brands {' '.join(payload['preferred_brands'])} located in {payload['location']} risk {payload['risk_tolerance']}"
        texts_to_embed.append(text_to_embed)
        payloads.append(payload)

    # Batch encode all user embeddings
    print(f"Encoding {len(texts_to_embed)} user embeddings...")
    embeddings = embedding_model.encode(texts_to_embed, device=DEVICE, show_progress_bar=True, convert_to_numpy=True)
    
    for user_uuid, embedding, payload in zip(user_ids, embeddings, payloads):
        points.append(PointStruct(
            id=user_uuid, 
            vector=embedding.tolist(),
            payload=payload
        ))

    # Insert in batches to avoid server disconnection
    batch_size = 100
    print(f"Upserting {len(points)} points in batches of {batch_size}...")
    for batch_start in range(0, len(points), batch_size):
        batch_end = min(batch_start + batch_size, len(points))
        batch = points[batch_start:batch_end]
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(collection_name=collection_name, points=batch)
                print(f"  ✅ Upserted points {batch_start} to {batch_end}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    print(f"  ⚠️  Batch {batch_start}-{batch_end} failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Batch {batch_start}-{batch_end} failed after {max_retries} attempts")
                    raise
    
    print(f"✅ inserted {len(points)} items into '{collection_name}'")

    user_profile_map: Dict[str, Dict[str, Any]] = {p['user_id']: p for p in payloads}
    return user_ids, user_profile_map

def insert_financials(client: QdrantClient, user_profile_map: Dict[str, Dict[str, Any]], product_map: Dict[str, Dict[str, Any]]):
    """Generate financial contexts for existing users, correlated to product prices."""
    collection_name = "financial_contexts"
    vector_size = 256
    points = []

    print(f"Processing {collection_name}...")

    for uid, profile in user_profile_map.items():
        # Derive financials from the user's preferred categories: compute avg price
        preferred = profile.get('preferred_categories', [])
        prices = [
            p['price']
            for p in product_map.values()
            if isinstance(p.get('price'), (int, float))
            and any(cat in preferred for cat in (p.get('categories') or []))
        ]

        if prices:
            avg_price = sum(prices) / len(prices)
            # credit and balance scaled to the category price
            credit_limit = round(avg_price * random.uniform(2.0, 5.0), 2)
            available_balance = round(avg_price * random.uniform(0.5, 2.0), 2)
        else:
            credit_limit = round(random.uniform(1000, 10000), 2)
            available_balance = round(random.uniform(100, credit_limit), 2)

        debt = round(random.uniform(0, credit_limit * 0.5), 2)

        payload = {
            "user_id": uid,
            "available_balance": available_balance,
            "credit_limit": credit_limit,
            "current_debt": debt,
            "eligible_installments": available_balance > 500  # Simple rule
        }

        points.append(PointStruct(
            id=uid, 
            vector=generate_random_vector(vector_size),
            payload=payload
        ))

    # Insert in batches to avoid server disconnection
    batch_size = 100
    print(f"Upserting {len(points)} points in batches of {batch_size}...")
    for batch_start in range(0, len(points), batch_size):
        batch_end = min(batch_start + batch_size, len(points))
        batch = points[batch_start:batch_end]
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(collection_name=collection_name, points=batch)
                print(f"  ✅ Upserted points {batch_start} to {batch_end}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    print(f"  ⚠️  Batch {batch_start}-{batch_end} failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Batch {batch_start}-{batch_end} failed after {max_retries} attempts")
                    raise
    
    print(f"✅ inserted {len(points)} items into '{collection_name}'")

def insert_interactions(client: QdrantClient, user_profile_map: Dict[str, Dict[str, Any]], product_map: Dict[str, Dict[str, Any]]):
    """Generate interaction history aligned to interaction_memory schema."""
    collection_name = "interaction_memory"
    vector_size = 384
    points = []
    interaction_ids = []
    texts_to_embed = []
    payloads = []

    print(f"Processing {collection_name}...")

    if not product_map:
        print("Warning: No products available for interactions.")
        return

    # Build helper: category -> product ids
    category_index: Dict[str, List[str]] = {}
    all_product_ids = list(product_map.keys())
    for pid, pdata in product_map.items():
        cats = pdata.get('categories', ['General'])
        if not isinstance(cats, list) or not cats:
            cats = [str(cats)] if cats else ["General"]
        for cat in cats:
            category_index.setdefault(cat, []).append(pid)

    interaction_weights = {
        "view": 0.1,
        "click": 0.3,
        "add_to_cart": 0.6,
        "purchase": 1.0,
    }

    for uid, profile in user_profile_map.items():
        # 2-3 interactions per user
        num_interactions = random.randint(2, 3)
        preferred = profile.get('preferred_categories', [])

        for _ in range(num_interactions):
            interaction_uuid = str(uuid.uuid4())
            interaction_type = random.choices(
                population=["view", "click", "add_to_cart", "purchase"],
                weights=[0.4, 0.3, 0.2, 0.1],
                k=1
            )[0]
            query = fake.sentence(nb_words=4)

            # Prefer products from user's preferred categories when available
            candidate_ids = []
            for cat in preferred:
                candidate_ids.extend(category_index.get(cat, []))
            if not candidate_ids:
                candidate_ids = all_product_ids

            pid = random.choice(candidate_ids)
            product = product_map.get(pid, {})
            categories = product.get("categories", ["General"])
            if not isinstance(categories, list) or not categories:
                categories = [str(categories)] if categories else ["General"]
            category = categories[0]
            brand = product.get("brand", "Unknown")
            price = float(product.get("price", 0.0)) if product else 0.0
            weight = interaction_weights.get(interaction_type, 0.1)

            payload = {
                "user_id": uid,
                "product_id": pid,
                "interaction_type": interaction_type,
                "timestamp": int(time.time()),
                "category": category,
                "brand": brand,
                "price": price,
                "weight": weight,
            }

            interaction_ids.append(interaction_uuid)
            behavioral_text = f"user {interaction_type} {product.get('name', 'product')} {category} {brand} price {price}"
            if query:
                behavioral_text += f" for {query}"
            texts_to_embed.append(behavioral_text)
            payloads.append(payload)

    # Batch encode all interaction queries
    print(f"Encoding {len(texts_to_embed)} interaction embeddings...")
    embeddings = embedding_model.encode(texts_to_embed, device=DEVICE, show_progress_bar=True, convert_to_numpy=True)
    
    for interaction_uuid, embedding, payload in zip(interaction_ids, embeddings, payloads):
        points.append(PointStruct(
            id=interaction_uuid,
            vector=embedding.tolist(),
            payload=payload
        ))

    # Insert in batches to avoid server disconnection
    batch_size = 100
    print(f"Upserting {len(points)} points in batches of {batch_size}...")
    for batch_start in range(0, len(points), batch_size):
        batch_end = min(batch_start + batch_size, len(points))
        batch = points[batch_start:batch_end]
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.upsert(collection_name=collection_name, points=batch)
                print(f"  ✅ Upserted points {batch_start} to {batch_end}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    print(f"  ⚠️  Batch {batch_start}-{batch_end} failed (attempt {attempt+1}/{max_retries}): {type(e).__name__}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  ❌ Batch {batch_start}-{batch_end} failed after {max_retries} attempts")
                    raise
    
    print(f"✅ inserted {len(points)} items into '{collection_name}'")

def main():
    print("Starting Data Insertion from products_payload.json...")
    
    # Setup
    client = get_qdrant_client()
    
    # Load ALL product data (no limit - use real data only)
    products_source = load_product_data("data/all_products_payload.json", limit=5000)
    print(f"Loaded {len(products_source)} products from JSON")

    # A) Products (real data, no synthetic generation)
    product_ids, product_map = insert_products(client, products_source)

    # B) Users (derive preferences from products)
    user_ids, user_profile_map = insert_users(client, product_map, count=1000)

    # C) Financials (depends on Users and product prices)
    insert_financials(client, user_profile_map, product_map)

    # D) Interactions (depends on Users + Products) - correlated
    insert_interactions(client, user_profile_map, product_map)

    print("\n🎉 Data insertion complete.")

if __name__ == "__main__":
    main()
