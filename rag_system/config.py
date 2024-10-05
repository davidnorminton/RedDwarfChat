# rag_system/config.py

import os
import torch

# -------------------- Configuration -------------------- #

# Define absolute paths for better reliability across different environments
MODEL_DIR = './Model'  # Path to Llama2 model
DATA_FILE = os.path.abspath(os.path.join("data", "rag_documents.jsonl"))  # Path to RAG documents
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Sentence Transformer for embeddings
TOP_K = 3  # Number of top documents to retrieve
INDEX_FILE = os.path.abspath(os.path.join("data", "faiss_index.bin"))  # Path to FAISS index

# Device configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")