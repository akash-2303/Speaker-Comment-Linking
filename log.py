# log.py
import json
import sys
import argparse
from datetime import datetime

def load_json_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json_data(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def fix_data(data):
    for item in data:
        if 'publishedAt' not in item or not item['publishedAt']:
            item['publishedAt'] = get_default_published_date()
        else:
            try:
                item['publishedAt'] = fix_date_format(item['publishedAt'])
            except ValueError:
                log_error("Invalid date format, setting to default")
                item['publishedAt'] = get_default_published_date()
    return data

def fix_date_format(date_str):
    if date_str.endswith('Z'):
        date_str = date_str.rstrip('Z')
    return datetime.fromisoformat(date_str).isoformat()

def get_default_published_date():
    return datetime.utcnow().isoformat()

def log_error(message):
    print(f"ERROR: {message}")

def main(args):
    data = load_json_data(args.input_file)
    fixed_data = fix_data(data)
    save_json_data(fixed_data, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix data issues in JSON file.")
    parser.add_argument("--input_file", required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", required=True, help="Path to the output JSON file to save fixed data.")
    args = parser.parse_args()
    main(args)