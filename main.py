# main.py

import sys
import os
from rag_system.config import MODEL_DIR, DATA_FILE, EMBEDDING_MODEL_NAME, TOP_K, INDEX_FILE, DEVICE
from rag_system.data_loader import load_documents
from rag_system.embedding import load_embedding_model, generate_embeddings
from rag_system.indexer import load_faiss_index, build_faiss_index, retrieve_documents
from rag_system.spell_checker import initialize_spell_checker, correct_text_spellchecker
from rag_system.model import load_language_model, generate_answer

def main():
    # -------------------- Preliminary Checks -------------------- #
    # Check if MODEL_DIR exists
    if not os.path.isdir(MODEL_DIR):
        print(f"Error: Model directory '{MODEL_DIR}' does not exist.")
        sys.exit(1)
    
    # Check for essential model files
    essential_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    missing_files = [f for f in essential_files if not os.path.isfile(os.path.join(MODEL_DIR, f))]
    if missing_files:
        print(f"Error: Missing files in model directory: {missing_files}")
        print("Ensure that 'tokenizer.json' and 'tokenizer_config.json' are present in the model directory.")
        sys.exit(1)
    
    # Check if DATA_FILE exists
    if not os.path.isfile(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' does not exist.")
        sys.exit(1)
    
    # -------------------- Define Domain-Specific Terms -------------------- #
    DOMAIN_SPECIFIC_WORDS = [
        "Ace Rimmer", "Arnold Rimmer", "Dave Lister", "Cat", "Kryten",
        "Holly", "Red Dwarf", "Dimension Jump", "Lister", "Arnold",
        "Ice Summer", "Chris Carr", "Space Corps", "Eve Lister", "Kendrick",
        "Nate", "Smoko", "Mechanoid", "Starbug", "Priest", "Jason",
        "Prince", "Sister", "Holly 2.0", "Holly2.0",
        "Starbug2", "Ice Rimmer", "Commander Rimmer", "James Hindle", "Holly2.0",
        "Space Corps", "Dave Lister", "Kryten", "Cat", "Rimmer"
    ]
    
    # -------------------- Initialize Spell Checker -------------------- #
    SINGLE_WORD_DOMAIN_TERMS = [
        "Ace", "Arnold", "Rimmer", "Lister", "Cat", "Kryten",
        "Holly", "Red", "Dwarf", "Dimension", "Jump", "Ice", "Summer",
        "Space", "Corps", "Eve", "Lister", "Arnold", "Kendrick", "Nate",
        "Smoko", "Mechanoid", "Holly", "Starbug", "Priest", "Jason",
        "Prince", "Sister", "Holly", "Holly", "Holly2.0", "Jason", "Smoko",
        "RedDwarf", "Starbug2", "Holly2.0"
    ]
    spell = initialize_spell_checker(SINGLE_WORD_DOMAIN_TERMS)
    
    # -------------------- Load Models -------------------- #
    print("Loading language model and tokenizer...")
    tokenizer, model = load_language_model(MODEL_DIR, DEVICE)
    
    print("\nLoading embedding model...")
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME, DEVICE)
    
    # -------------------- Load Documents -------------------- #
    print("\nLoading documents...")
    documents = load_documents(DATA_FILE)
    
    # -------------------- Load or Build FAISS Index -------------------- #
    print("\nSetting up FAISS index...")
    index = load_faiss_index(embedding_model, documents, INDEX_FILE, build_faiss_index)
    
    # -------------------- Interactive Query Loop -------------------- #
    print("\nRAG System is ready! You can start querying about 'Red Dwarf'. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_query = input("\nEnter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting RAG system. Goodbye!")
            break
        
        if user_query.lower() in ['exit', 'quit']:
            print("Exiting RAG system. Goodbye!")
            break
        if not user_query:
            print("Empty query. Please enter a valid question.")
            continue

        # Retrieve relevant documents
        try:
            retrieved_docs = retrieve_documents(user_query, embedding_model, index, documents, TOP_K)
        except Exception as e_retrieve:
            print(f"Error retrieving documents: {e_retrieve}")
            continue

        if not retrieved_docs:
            print("No relevant documents found for the query.")
            continue

        # Construct the prompt with explicit instructions
        context = "\n\n".join(retrieved_docs)
        prompt = (
            "You are an AI assistant specialized in the 'Red Dwarf' TV show. "
            "Provide clear, grammatically correct, and well-spelled answers using only the information provided in the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {user_query}\nAnswer:"
        )

        # Generate and display the answer with spell and grammar correction
        print("\nGenerating answer...")
        try:
            answer = generate_answer(prompt, tokenizer, model, DEVICE)
            if not answer:
                print("No answer was generated. Please try a different query.")
            else:
                # Correct spelling and grammar while preserving domain-specific terms
                corrected_answer = correct_text_spellchecker(answer, DOMAIN_SPECIFIC_WORDS, spell)
                print(f"\nAnswer:\n{corrected_answer}\n")
        except Exception as e_answer:
            print(f"Error generating answer: {e_answer}")

# -------------------- Entry Point -------------------- #

if __name__ == "__main__":
    main()