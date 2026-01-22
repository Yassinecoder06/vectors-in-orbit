import csv
import json
import re
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer


# ===============================
# Configuration
# ===============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

torch.set_grad_enabled(False)

embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)


CATEGORY_DESCRIPTIONS = {
    "Smartphones": "mobile phone device with touchscreen RAM storage processor 5G camera battery",
    "Computers": "laptop desktop PC notebook computer tablet keyboard processor RAM storage display",
    "Audio": "headphones earbuds speakers earphones wireless bluetooth audio device",
    "Wearables": "smartwatch fitness tracker wearable wrist device",
    "Cameras": "camera DSLR mirrorless photography video recording lens",
    "Accessories": "charger cable adapter case cover power bank mouse keyboard peripheral"
}


REGIONS = [
    "North America", "Europe", "Asia Pacific", "India", "Southeast Asia",
    "Middle East", "Africa", "South America", "Australia", "East Asia"
]


# ===============================
# Precompute category embeddings
# ===============================

CATEGORY_EMBEDDINGS = {
    category: embedding_model.encode(
        desc,
        normalize_embeddings=True
    )
    for category, desc in CATEGORY_DESCRIPTIONS.items()
}


# ===============================
# Utility functions
# ===============================

def extract_brand(text: str) -> str:
    """Lightweight brand extraction from product name."""
    common_words = {"the", "a", "with", "and", "for", "in", "on", "at", "to", "new"}

    for word in text.split()[:3]:
        clean = word.strip("(),")
        if len(clean) > 2 and clean.lower() not in common_words:
            return clean

    return "Generic"


def categorize_embedding(embedding) -> str:
    """Semantic category selection using cosine similarity."""
    return max(
        CATEGORY_EMBEDDINGS,
        key=lambda c: embedding @ CATEGORY_EMBEDDINGS[c]
    )


def clean_price(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    raw = re.sub(r"[^0-9.]", "", raw)
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def clean_int(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    raw = re.sub(r"[^0-9]", "", raw)
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def clean_float(raw: Optional[str]) -> Optional[float]:
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


# ===============================
# Main processing logic
# ===============================

def process_csv_to_json(input_file: str, output_file: str):
    print(f"Reading {input_file}...")

    names: List[str] = []
    rows: List[Dict] = []
    seen_names = set()

    with open(input_file, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = (row.get("name") or "").strip()
            if not name or name in seen_names:
                continue

            discount_price = clean_price(row.get("discount_price"))
            actual_price = clean_price(row.get("actual_price"))
            price = discount_price or actual_price

            if price is None:
                continue

            names.append(name)
            rows.append(row)
            seen_names.add(name)

    print(f"Embedding {len(names)} products...")

    embeddings = embedding_model.encode(
        names,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    products: List[Dict] = []

    for name, row, emb in zip(names, rows, embeddings):
        product_data = {
            "description": name,
            "category": categorize_embedding(emb),
            "brand": extract_brand(name),
            "price": clean_price(row.get("discount_price"))
                     or clean_price(row.get("actual_price")),
            "in_stock": random.choice([True, False]),
            "region": random.choice(REGIONS),
            "image_url": (row.get("image") or "").strip(),
            "rating": clean_float(row.get("ratings")),
            "rating_count": clean_int(row.get("no_of_ratings")),
        }

        products.append(product_data)

    print(f"Writing {len(products)} products to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2)

    print("Done ✅")


# ===============================
# Entry point
# ===============================

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    input_file = base_dir / "All Electronics.csv"
    output_file = base_dir / "products_payload.json"

    process_csv_to_json(str(input_file), str(output_file))
