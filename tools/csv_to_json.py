import csv
import json
import sys

csv.field_size_limit(2147483647)

def csv_to_json(csv_file_path, json_file_path):
    data = []

    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)

    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    print(f"Converted {csv_file_path} → {json_file_path}")

# Example usage
csv_to_json("input1.csv", "output1.json")
