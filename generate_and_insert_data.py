import os
import sys
import json
import random
import uuid
import numpy as np
from typing import List, Dict, Any
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
    """Generate a real embedding using SentenceTransformer (768 dim)."""
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

    # If source is empty, generate purely synthetic
    count = 50 
    
    # We loop up to 'count'. If we have source data, we use it, otherwise fake it.
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
        else:
            # Fallback synthetic
            name = fake.catch_phrase()
            category = random.choice(["Home Decor", "Electronics", "Clothing", "Toys"])
            brand = fake.company()
            price = round(random.uniform(10.0, 500.0), 2)
            in_stock = random.choice([True, False])
            region = fake.country()

        # Augment with extra fields requested
        monthly_installment = round(price / 12, 2)
        tags = [str(fake.word()) for _ in range(3)]

        # Create rich text representation for embedding
        text_to_embed = f"{name} {category} {brand} {' '.join(tags)} {region}"
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
            "tags": tags
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

    # Insert
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ inserted {len(points)} items into '{collection_name}'")
    return product_ids

def insert_users(client: QdrantClient) -> List[str]:
    """Generate and insert user profiles."""
    collection_name = "user_profiles"
    vector_size = 384
    points = []
    user_ids = []
    texts_to_embed = []
    payloads = []
    count = 20

    print(f"Processing {collection_name}...")

    for _ in range(count):
        user_uuid = str(uuid.uuid4())
        user_ids.append(user_uuid)

        payload = {
            "user_id": user_uuid,
            "name": fake.name(),
            "location": fake.city(),
            "risk_tolerance": random.choice(["Low", "Medium", "High"]),
            "preferred_categories": [fake.word() for _ in range(2)],
            "preferred_brands": [fake.company() for _ in range(2)]
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

    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ inserted {len(points)} items into '{collection_name}'")
    return user_ids

def insert_financials(client: QdrantClient, user_ids: List[str]):
    """Generate financial contexts for existing users."""
    collection_name = "financial_contexts"
    vector_size = 256
    points = []

    print(f"Processing {collection_name}...")

    for uid in user_ids:
        # Generate financial data
        limit = round(random.uniform(1000, 10000), 2)
        debt = round(random.uniform(0, limit * 0.8), 2)
        balance = round(limit - debt, 2)

        payload = {
            "user_id": uid,
            "available_balance": balance,
            "credit_limit": limit,
            "current_debt": debt,
            "eligible_installments": balance > 500  # Simple rule
        }

        # ID logic: often 1:1 with user_id, here we use user_uuid as point ID strictly
        # or generate a new UUID. Using user_uuid makes lookups O(1) easier.
        # Note: Financial context is kept as random vectors (or feature vectors) 
        # because the schema requires 256 dimensions and it's numerical data, 
        # not suitable for the 768-dim text embedding model.
        points.append(PointStruct(
            id=uid, 
            vector=generate_random_vector(vector_size),
            payload=payload
        ))

    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ inserted {len(points)} items into '{collection_name}'")

def insert_interactions(client: QdrantClient, user_ids: List[str], product_ids: List[str]):
    """Generate interaction history."""
    collection_name = "interaction_memory"
    vector_size = 384
    points = []
    interaction_ids = []
    texts_to_embed = []
    payloads = []

    print(f"Processing {collection_name}...")

    if not product_ids:
        print("Warning: No products available for interactions.")
        return

    for uid in user_ids:
        # 2-3 interactions per user
        num_interactions = random.randint(2, 3)
        for _ in range(num_interactions):
            interaction_uuid = str(uuid.uuid4())
            pid = random.choice(product_ids)
            purchased = random.choice([True, False])
            query = fake.sentence(nb_words=4)

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

    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ inserted {len(points)} items into '{collection_name}'")

def main():
    print("Starting Synthetic Data Generation & Insertion...")
    
    # Setup
    client = get_qdrant_client()
    
    # Load JSON source
    products_source = load_product_data("data/products_payload.json", limit=50)

    # A) Products
    product_ids = insert_products(client, products_source)

    # B) Users
    user_ids = insert_users(client)

    # C) Financials (depends on Users)
    insert_financials(client, user_ids)

    # D) Interactions (depends on Users + Products)
    insert_interactions(client, user_ids, product_ids)

    print("\n🎉 Data generation and insertion complete.")

if __name__ == "__main__":
    main()
