import csv
import json
import ast
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Increase CSV field limit to handle large descriptions
# Using a fixed large integer to avoid OverflowError on Windows (which sys.maxsize can trigger)
csv.field_size_limit(2147483647)

# Constants
DATA_DIR = Path(__file__).parent
INPUT_CSV = DATA_DIR / "amazon-products.csv"
OUTPUT_JSON = DATA_DIR / "amazon_products_payload.json"

def clean_price(price_str: str) -> float:
    """
    Cleans price string and converts to float.
    Handles "$57.79", "57.79", etc.
    Returns 0.0 if invalid.
    """
    if not price_str or price_str.lower() == "null":
        return 0.0
    try:
        # Remove currency symbols and commas
        clean = re.sub(r'[^\d.]', '', price_str)
        return float(clean)
    except ValueError:
        return 0.0

def clean_bool(value: str) -> bool:
    """Parses boolean strings."""
    if not value:
        return False
    return value.lower() in ('true', 'yes', '1', 'in stock')

def parse_categories(categories_str: str) -> List[str]:
    """
    Extracts categories from the JSON-like array string.
    Example: '["Clothing, Shoes & Jewelry", "Men"]' -> ["Clothing, Shoes & Jewelry", "Men"]
    """
    if not categories_str or str(categories_str).lower() == "null":
        return ["General"]

    try:
        categories = ast.literal_eval(categories_str)
        if isinstance(categories, list) and len(categories) > 0:
            return [str(c).strip() for c in categories if str(c).strip()]
    except (ValueError, SyntaxError):
        pass

    cleaned = str(categories_str).strip().strip("[]")
    if cleaned:
        parts = [p.strip().strip('"') for p in cleaned.split(",") if p.strip()]
        return parts if parts else ["General"]

    return ["General"]

def process_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Maps CSV row to Qdrant payload schema."""
    try:
        product_id = str(uuid4())

        # Clean Price
        price_str = row.get("final_price", "0")
        price = clean_price(price_str)
        
        # Skip rows with missing/zero price
        if price <= 0:
            return None

        # Calculate monthly installment (12 months)
        monthly_installment = round(price / 12, 2)

        # Availability
        availability = row.get("availability", "").lower()
        in_stock = "in stock" in availability

        # Categories Processing
        categories_raw = row.get("categories", "[]")
        categories = parse_categories(categories_raw)

        # Construct Payload matching qdrant_setup.py
        payload = {
            "product_id": product_id,
            "name": row.get("title", "Unknown Product"),
            "description": row.get("description", ""),
            "price": price,
            "categories": categories,
            "brand": row.get("brand", "Unknown"),
            "in_stock": in_stock,
            "image_url": row.get("image_url", ""),
            "region": "US",
            "monthly_installment": monthly_installment,
        }
        
        # Cleanup "null" string values which appear in the CSV
        for k, v in payload.items():
            if isinstance(v, str) and v.lower() == "null":
                payload[k] = ""

        return payload

    except Exception as e:
        print(f"Error processing row: {e}")
        return None

def main():
    print(f"Reading from {INPUT_CSV}...")
    
    products = []
    
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found.")
        return

    try:
        with open(INPUT_CSV, mode="r", encoding="utf-8", errors="replace") as csvfile:
            reader = csv.DictReader(csvfile)
            for i, row in enumerate(reader):
                processed = process_row(row)
                if processed:
                    products.append(processed)
                
                print(f"Processed row {i + 1}")

        print(f"Total valid products extracted: {len(products)}")

        print(f"Writing to {OUTPUT_JSON}...")
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(products, f, indent=2, ensure_ascii=False)
        
        print("Done!")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
