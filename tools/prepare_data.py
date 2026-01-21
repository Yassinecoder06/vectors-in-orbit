import csv
import json
import random
from typing import Dict, List

# Constants for data generation
BRANDS = [
    "Nordic Living", "Urban Craft", "Vintage Charm", "Modern Essence", 
    "EcoHome", "Artisan Works", "Royal Designs", "Simply Style"
]

CATEGORY_KEYWORDS = {
    "Home Decor": ["HEART", "HOLDER", "LANTERN", "CANDLE", "FRAME", "CLOCK", "VASE", "DECORATION", "ORNAMENT"],
    "Kitchen": ["CUP", "MUG", "BOWL", "PLATE", "JAR", "GLASS", "BOTTLE", "KITCHEN", "SPOON", "TRAY", "CAKE"],
    "Apparel": ["SHIRT", "BAG", "GLOVE", "SCARF", "APRON", "SOCK", "SLIPPER"],
    "Office": ["PEN", "PENCIL", "NOTEBOOK", "WRAP", "PAPER", "CARD", "STATIONERY"],
    "Toys & Games": ["GAME", "TOY", "DOLL", "PUZZLE", "PLAY"],
    "Garden": ["GARDEN", "FLOWER", "POT", "PLANTER", "WATERING"],
}

def guess_category(description: str) -> str:
    """Guess category based on keywords in description."""
    if not description:
        return "Uncategorized"
    
    desc_upper = description.upper()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in desc_upper for keyword in keywords):
            return category
    
    return "General" # Fallback category

def process_csv_to_json(input_file: str, output_file: str):
    print(f"Reading from {input_file}...")
    
    products: Dict[str, Dict] = {}
    
    try:
        with open(input_file, mode='r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                stock_code = row.get("StockCode")
                description = row.get("Description", "").strip()
                
                # Skip invalid rows
                if not stock_code or not description:
                    continue
                
                # Deduplication logic: 
                # We interpret the transaction log by keeping the properties of the product
                # We only process it if we haven't seen this StockCode defined yet.
                if stock_code not in products:
                    
                    try:
                        price = float(row.get("UnitPrice", 0.0))
                        quantity = int(row.get("Quantity", 0))
                    except ValueError:
                        continue # Skip bad number formats

                    # Transform data
                    product_data = {
                        "id": stock_code,  # Useful metadata
                        "description": description, # Useful for vectorization later
                        
                        # Schema fields
                        "price": price,
                        "category": guess_category(description),
                        "brand": random.choice(BRANDS),
                        "in_stock": quantity > 0, # Simple heuristic from transaction data
                        "region": row.get("Country", "Unknown")
                    }
                    
                    products[stock_code] = product_data

    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # Convert values to list
    product_list = list(products.values())
    
    print(f"Processed {len(product_list)} unique products.")
    
    print(f"Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(product_list, f, indent=2)
    
    print("Done.")

if __name__ == "__main__":
    process_csv_to_json("data.csv", "products_payload.json")
