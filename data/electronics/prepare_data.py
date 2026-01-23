import csv
import json
import re
import random
from pathlib import Path
from typing import Dict, List, Optional


# ===============================
# Manual Category Mapping
# ===============================

# Keywords to identify categories from product names
CATEGORY_KEYWORDS = {
    "Smartphones": ["phone", "iphone", "android", "redmi", "oneplus", "samsung galaxy", "nord", "realme", "oppo", "vivo", "iqoo", "nokia", "moto", "xiaomi"],
    "Computers": ["laptop", "desktop", "pc", "notebook", "tablet", "keyboard", "mouse", "usb", "ssd", "hard drive", "ram", "processor", "macbook", "dell", "hp", "lenovo"],
    "Audio": ["earbuds", "earphones", "headphones", "speaker", "bluetooth", "wireless", "airdropes", "boat", "jbl", "noise", "pulse", "sony", "sound", "audio"],
    "Wearables": ["smartwatch", "watch", "fitness", "tracker", "wearable", "fitbit", "fire-boltt", "noise colorfit", "boat wave", "strap", "wrist"],
    "Cameras": ["camera", "dslr", "mirrorless", "canon", "nikon", "fujifilm", "instax", "tripod", "lens", "photography", "webcam"],
    "Accessories": ["charger", "cable", "adapter", "case", "cover", "power bank", "pendrive", "usb drive", "memory card", "sd card", "hdmi", "cable", "battery", "ink", "cartridge"]
}

# Brand extraction patterns (first word in product name that's not a generic word)
COMMON_WORDS = {"the", "a", "an", "with", "and", "for", "in", "on", "at", "to", "new", "original", "best", "latest"}

REGIONS = [
    "North America", "Europe", "Asia Pacific", "India", "Southeast Asia",
    "Middle East", "Africa", "South America", "Australia", "East Asia"
]


# ===============================
# Utility functions
# ===============================

def extract_brand_from_name(name: str) -> str:
    """Extract brand from product name."""
    # Common brands to look for
    brands = [
        "Redmi", "OnePlus", "Samsung", "Apple", "Fire-Boltt", "realme", "boAt", 
        "MI", "SanDisk", "pTron", "Logitech", "HP", "Dell", "JBL", "Noise",
        "Portronics", "ZEBRONICS", "Oppo", "iQOO", "vivo", "Canon", "Nokia",
        "Lenovo", "Acer", "TP-Link", "Crucial", "Seagate", "Western Digital",
        "Ambrane", "Boult", "Fastrack", "Tecno", "Motorola", "Lava", "Amazfit"
    ]
    
    name_lower = name.lower()
    for brand in brands:
        if brand.lower() in name_lower:
            return brand
    
    # If no known brand found, extract first significant word
    words = re.split(r'[ ,()\-|]+', name)
    for word in words:
        clean_word = word.strip()
        if (len(clean_word) > 2 and 
            clean_word[0].isupper() and 
            clean_word.lower() not in COMMON_WORDS):
            return clean_word
    
    return "Generic"


def categorize_from_name(name: str) -> str:
    """Categorize product based on keywords in name."""
    name_lower = name.lower()
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in name_lower:
                return category
    
    # Default to Accessories if no category found
    return "Accessories"


def clean_price(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    # Remove currency symbols and commas
    raw = re.sub(r"[₹$,]", "", raw)
    raw = re.sub(r"[^0-9.]", "", raw)
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def clean_int(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    # Remove commas from numbers like "113,956"
    raw = re.sub(r"[^0-9]", "", raw)
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def clean_float(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    # Handle ratings that might be "Get" or other non-numeric values
    if raw.lower() in ["get", "free", "na", "n/a"]:
        return None
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


# ===============================
# Main processing logic
# ===============================

def process_csv_to_json(input_file: str, output_file: str):
    print(f"Reading {input_file}...")
    
    products: List[Dict] = []
    seen_names = set()
    
    with open(input_file, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        
        for row_num, row in enumerate(reader, 1):
            name = (row.get("name") or "").strip()
            if not name or name in seen_names:
                continue
            
            seen_names.add(name)
            
            # Extract price - prefer discount_price, fall back to actual_price
            discount_price = clean_price(row.get("discount_price"))
            actual_price = clean_price(row.get("actual_price"))
            price = discount_price or actual_price
            
            # Skip if no price available
            if price is None:
                continue
            
            # Extract category and brand from name
            category = categorize_from_name(name)
            brand = extract_brand_from_name(name)
            
            product_data = {
                "description": name,
                "category": category,
                "brand": brand,
                "price": price,
                "in_stock": random.choice([True, False]),
                "region": random.choice(REGIONS),
                "image_url": (row.get("image") or "").strip(),
                "rating": clean_float(row.get("ratings")),
                "rating_count": clean_int(row.get("no_of_ratings")),
            }
            
            products.append(product_data)
            
            if row_num % 100 == 0:
                print(f"Processed {row_num} products...")
    
    print(f"Writing {len(products)} products to {output_file}...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2)
    
    # Print some statistics
    print("\n=== Processing Statistics ===")
    categories = {}
    brands = {}
    
    for product in products:
        cat = product["category"]
        brand = product["brand"]
        
        categories[cat] = categories.get(cat, 0) + 1
        brands[brand] = brands.get(brand, 0) + 1
    
    print(f"Total products: {len(products)}")
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")
    
    print("\nTop 10 brands:")
    top_brands = sorted(brands.items(), key=lambda x: x[1], reverse=True)[:10]
    for brand, count in top_brands:
        print(f"  {brand}: {count}")
    
    print("\nDone ✅")


# ===============================
# Entry point
# ===============================

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    input_file = base_dir / "All Electronics.csv"
    output_file = base_dir / "products_payload.json"
    
    process_csv_to_json(str(input_file), str(output_file))
