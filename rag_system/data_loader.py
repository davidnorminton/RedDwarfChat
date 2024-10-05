# rag_system/data_loader.py

import json
import sys

def load_documents(data_file):
    """
    Load documents from a JSON Lines file.
    Each line should be a JSON object with a 'text', 'content', or 'document' field.
    """
    documents = []
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get('text') or data.get('content') or data.get('document')
                    if text:
                        documents.append(text)
                except json.JSONDecodeError:
                    print("Skipping invalid JSON line.")
        print(f"Loaded {len(documents)} documents.")
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.")
        sys.exit(1)
    return documents