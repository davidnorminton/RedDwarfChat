# rag_system/embedding.py

from sentence_transformers import SentenceTransformer
import sys

def load_embedding_model(embedding_model_name, device):
    """
    Load the SentenceTransformer embedding model.
    """
    try:
        embedding_model = SentenceTransformer(embedding_model_name, device=device)
        print("Embedding model loaded successfully.")
        return embedding_model
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        sys.exit(1)

def generate_embeddings(embedding_model, documents):
    """
    Generate embeddings for the list of documents.
    """
    try:
        embeddings = embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        sys.exit(1)