import json
from pathlib import Path

paths = [".\\walmart", ".\\amazon", ".\\lazada", ".\\shein"]

def combine_json_files(output_file="combined.json"):
    combined_data = []

    for path in paths:
        for json_file in Path(path).glob("*.json"):
            if json_file.name == output_file:
                continue  # avoid reading the output file itself

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    if isinstance(data, list):
                        combined_data.extend(data)
                    else:
                        combined_data.append(data)

            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping {json_file}: invalid JSON ({e})")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Combined JSON saved to {output_file}")

if __name__ == "__main__":
    combine_json_files("all_products_payload.json")
