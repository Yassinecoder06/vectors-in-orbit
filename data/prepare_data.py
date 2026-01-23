"""
Convert Flipkart data.csv to products_payload.json matching Qdrant products_multimodal schema.

CSV Schema (Flipkart):
  - product_name
  - product_category_tree (e.g., "Clothing >> Women's Clothing >> ...")
  - pid (product ID)
  - discounted_price (actual selling price)
  - image (JSON array of image URLs)
  - description
  - brand

Qdrant Schema Mapping:
  - pid → product_id
  - product_name → name
  - description → description
  - discounted_price → price
  - product_category_tree → category (extract last segment)
  - brand → brand
  - image → image_url (take first URL from array)
  - (generated) → in_stock, region, monthly_installment
"""

import csv
import json
import re
from pathlib import Path
from faker import Faker

fake = Faker()


def extract_category(category_tree: str) -> str:
    """Extract the last meaningful category from the tree."""
    if not category_tree:
        return "General"
    # Remove brackets and split by >>
    tree = category_tree.strip('[]"')
    parts = [p.strip().strip('"') for p in tree.split(">>")]
    # Return last part or "General" if empty
    return parts[-1] if parts else "General"


def extract_image_url(image_json: str) -> str:
    """Extract the first image URL from the JSON array."""
    if not image_json:
        return "https://example.com/images/placeholder.jpg"
    try:
        # Parse JSON array of image URLs
        images = json.loads(image_json)
        if isinstance(images, list) and len(images) > 0:
            return images[0]
    except (json.JSONDecodeError, TypeError):
        pass
    return "https://example.com/images/placeholder.jpg"


def clean_price(price_str: str) -> float:
    """Convert price string to float, handle invalid values."""
    if not price_str:
        return 0.0
    try:
        # Remove commas and convert to float
        return float(str(price_str).replace(",", "").strip())
    except ValueError:
        return 0.0


def prepare_products(csv_path: Path, output_path: Path) -> int:
    """
    Convert Flipkart CSV to Qdrant-compatible JSON payload.
    
    Returns the count of products processed.
    """
    products = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Extract and clean data
                product_id = row.get('pid', '').strip()
                name = row.get('product_name', 'Unknown Product').strip()
                brand = row.get('brand', 'Unknown Brand').strip()
                category_tree = row.get('product_category_tree', '')
                category = extract_category(category_tree)
                description = row.get('description', '').strip()[:500]  # Limit to 500 chars
                image_json = row.get('image', '')
                image_url = extract_image_url(image_json)
                
                # Use discounted_price if available, else retail_price
                price_str = row.get('discounted_price') or row.get('retail_price', '0')
                price = clean_price(price_str)
                
                # Skip products with invalid IDs or prices
                if not product_id or price <= 0:
                    continue
                
                # Generate monthly installment (12 months, 10% APR)
                monthly_installment = round((price * 1.10) / 12, 2)
                
                # Build payload matching Qdrant schema
                payload = {
                    "product_id": product_id,
                    "name": name,
                    "description": description,
                    "category": category,
                    "brand": brand,
                    "price": price,
                    "in_stock": True,  # Default: all products in stock
                    "image_url": image_url,
                    "region": fake.country(),  # Random region using faker
                    "monthly_installment": monthly_installment
                }
                products.append(payload)
            
            except Exception as e:
                print(f"⚠️  Skipping row due to error: {e}")
                continue
    
    # Write JSON output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(products, f, indent=2)
    
    return len(products)


if __name__ == "__main__":
    csv_file = Path(__file__).parent / "data.csv"
    output_file = Path(__file__).parent / "products_payload.json"
    
    if not csv_file.exists():
        print(f"❌ Error: {csv_file} not found")
        exit(1)
    
    count = prepare_products(csv_file, output_file)
    print(f"✅ Converted {count} products from data.csv → products_payload.json")
