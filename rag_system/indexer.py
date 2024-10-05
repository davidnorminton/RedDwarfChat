# rag_system/indexer.py

import faiss
import os
import sys

def build_faiss_index(embeddings, save_path=None):
    """
    Build a FAISS index from embeddings.
    If save_path is provided, save the index to disk.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using a simple flat (brute-force) index
    faiss.omp_set_num_threads(os.cpu_count())  # Optimize FAISS for multi-threading
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    if save_path:
        faiss.write_index(index, save_path)
        print(f"FAISS index saved to '{save_path}'.")
    return index

def load_faiss_index(embedding_model, documents, index_file, build_faiss_index_func):
    """
    Load or build a FAISS index.
    """
    if os.path.exists(index_file):
        print("Loading existing FAISS index...")
        try:
            index = faiss.read_index(index_file)
            print("FAISS index loaded.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            print("Rebuilding the FAISS index.")
            embeddings = embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
            index = build_faiss_index_func(embeddings, save_path=index_file)
    else:
        print("Building FAISS index...")
        embeddings = embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        index = build_faiss_index_func(embeddings, save_path=index_file)
    return index

def retrieve_documents(query, embedding_model, index, documents, top_k=3):
    """
    Retrieve top_k documents relevant to the query.
    """
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)
        retrieved = [documents[idx] for idx in indices[0] if idx < len(documents)]
        print(f"Retrieved {len(retrieved)} documents for the query.")
        return retrieved
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        return []