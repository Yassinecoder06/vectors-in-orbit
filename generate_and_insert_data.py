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


def load_product_data(filepath: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Load and sample product data from JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Sample random 50 if available, else take all
            if len(data) > limit:
                return random.sample(data, limit)
            return data
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Generating purely synthetic products.")
        return []

def insert_products(client: QdrantClient, products_source: List[Dict]) -> List[str]:
    """
    Simulate product data generation/augmentation and insert into Qdrant.
    Returns list of product IDs for interaction generation.
    """
    collection_name = "products_multimodal"
    vector_size = 384
    points = []
    product_ids = []
    texts_to_embed = []
    payloads = []

    print(f"Processing {collection_name}...")

    # Use all source data available
    count = len(products_source) if products_source else 50
    
    # We loop through all available products. If we have source data, we use it, otherwise fake it.
    for i in range(count):
        product_uuid = str(uuid.uuid4())
        
        # Use source data if available
        if i < len(products_source):
            src = products_source[i]
            # Map fields from the JSON
            name = src.get("description", fake.catch_phrase())
            category = src.get("category", "General")
            brand = src.get("brand", fake.company())
            price = src.get("price", round(random.uniform(10.0, 500.0), 2))
            in_stock = src.get("in_stock", True)
            region = src.get("region", fake.country())
            image_url = src.get("image_url") or f"https://picsum.photos/seed/{product_uuid}/600/600"
        else:
            # Fallback synthetic
            name = fake.catch_phrase()
            category = random.choice(["Home Decor", "Electronics", "Clothing", "Toys"])
            brand = fake.company()
            price = round(random.uniform(10.0, 500.0), 2)
            in_stock = random.choice([True, False])
            region = fake.country()
            image_url = f"https://picsum.photos/seed/{product_uuid}/600/600"

        # Augment with extra fields requested
        monthly_installment = round(price / 12, 2)

        # Create rich text representation for embedding
        text_to_embed = f"{name} {category} {brand} {region}"
        texts_to_embed.append(text_to_embed)

        payload = {
            "product_id": product_uuid,
            "name": name,
            "category": category,
            "brand": brand,
            "price": price,
            "monthly_installment": monthly_installment,
            "in_stock": in_stock,
            "region": region,
            "image_url": image_url,
        }
        payloads.append(payload)
        product_ids.append(product_uuid)

    # Batch encode all texts at once (much faster)
    print(f"Encoding {len(texts_to_embed)} product embeddings...")
    embeddings = embedding_model.encode(texts_to_embed, device=DEVICE, show_progress_bar=True, convert_to_numpy=True)
    
    # Build points with pre-computed embeddings
    for i, (product_uuid, embedding, payload) in enumerate(zip(product_ids, embeddings, payloads)):
        points.append(PointStruct(
            id=product_uuid,
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
    
    print(f"✅ inserted {len(points)} items into '{collection_name}'")

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
    categories = [p.get('category', 'General') for p in product_map.values()] if product_map else []
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
        prices = [p['price'] for p in product_map.values() if p.get('category') in preferred and isinstance(p.get('price'), (int, float))]

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
    """Generate interaction history."""
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
        cat = pdata.get('category', 'General')
        category_index.setdefault(cat, []).append(pid)

    for uid, profile in user_profile_map.items():
        # 2-3 interactions per user
        num_interactions = random.randint(2, 3)
        preferred = profile.get('preferred_categories', [])

        for _ in range(num_interactions):
            interaction_uuid = str(uuid.uuid4())
            purchased = random.choice([True, False])
            query = fake.sentence(nb_words=4)

            # Prefer products from user's preferred categories when available
            candidate_ids = []
            for cat in preferred:
                candidate_ids.extend(category_index.get(cat, []))
            if not candidate_ids:
                candidate_ids = all_product_ids

            pid = random.choice(candidate_ids)

            payload = {
                "user_id": uid,
                "query": query,
                "clicked_product_id": pid,
                "purchased": purchased,
                "timestamp": fake.iso8601()
            }

            interaction_ids.append(interaction_uuid)
            texts_to_embed.append(query)
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
    print("Starting Synthetic Data Generation & Insertion...")
    
    # Setup
    client = get_qdrant_client()
    
    # Load product data (limit to 500 for Qdrant Cloud stability)
    products_source = load_product_data("data/electronics/products_payload.json", limit=1000)
    print(f"Loaded {len(products_source)} products")

    # A) Products
    product_ids, product_map = insert_products(client, products_source)

    # B) Users (derive preferences from products)
    user_ids, user_profile_map = insert_users(client, product_map, count=200)

    # C) Financials (depends on Users and product prices)
    insert_financials(client, user_profile_map, product_map)

    # D) Interactions (depends on Users + Products) - correlated
    insert_interactions(client, user_profile_map, product_map)

    print("\n🎉 Data generation and insertion complete.")

if __name__ == "__main__":
    main()
