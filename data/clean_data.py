import json
from pathlib import Path
from chonkie import SemanticChunker
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B params, very light

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",      # automatically offloads layers to CPU if GPU memory is full
    torch_dtype=torch.float16,
    load_in_4bit=True       # reduces VRAM usage drastically
)


def generate(prompt: str, max_tokens: int = 512) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,   # low for consistency
            do_sample=False,   # deterministic output
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


chunker = SemanticChunker(
    model_name="all-MiniLM-L6-v2",
    max_chunk_size=256,
    overlap=32
)


def llm_extract_structured(chunks: list[str]) -> dict:
    prompt = f"""
You are a product information extraction engine.

RULES:
- Output ONLY valid JSON
- No explanations
- No markdown

Extract exactly these fields:
- product_type
- key_features (list)
- technical_specs (list)
- use_cases (list)
- target_audience (list)

Text:
{chr(10).join(chunks)}
"""

    raw = generate(prompt)

    # Clean up any extra formatting
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


def llm_generate_description(name: str, brand: str, structured: dict) -> str:
    prompt = f"""
You are a professional e-commerce copywriter.

STRICT RULES:
- Follow the structure EXACTLY
- No emojis
- Max 120 words

STRUCTURE:
1. One-sentence hook
2. Key Features (bullet points)
3. Technical Specifications (bullet points)
4. Ideal For (one sentence)

Product Name: {name}
Brand: {brand}

Data:
{json.dumps(structured, indent=2)}
"""

    return generate(prompt, max_tokens=256)

def normalize_products(input_path: str, output_path: str):
    products = json.loads(Path(input_path).read_text(encoding="utf-8"))
    normalized = []

    for idx, product in enumerate(products, 1):
        print(f"Processing product {idx}/{len(products)}: {product['name']}")

        # Semantic chunking
        chunks = chunker.chunk_text(product["description"])

        # Extract structured facts
        try:
            structured = llm_extract_structured(chunks)
        except Exception as e:
            print(f"Error extracting structure for {product['name']}: {e}")
            structured = {
                "product_type": "",
                "key_features": [],
                "technical_specs": [],
                "use_cases": [],
                "target_audience": []
            }

        # Generate consistent description
        try:
            new_description = llm_generate_description(
                name=product["name"],
                brand=product["brand"],
                structured=structured
            )
        except Exception as e:
            print(f"Error generating description for {product['name']}: {e}")
            new_description = product["description"]

        # Keep same schema, replace description only
        new_product = product.copy()
        new_product["description"] = new_description
        normalized.append(new_product)

    Path(output_path).write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


if __name__ == "__main__":
    normalize_products(
        input_path="all_products_payload.json",
        output_path="products_normalized.json"
    )
