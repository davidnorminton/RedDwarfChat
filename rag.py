import os
import sys
import json
import faiss
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from sentence_transformers import SentenceTransformer
from spellchecker import SpellChecker
import re
import logging

# -------------------- Configuration -------------------- #

# Define absolute paths for better reliability across different environments
MODEL_DIR = './Model'  # Path to Llama2 model
DATA_FILE = os.path.abspath(os.path.join("data", "rag_documents.jsonl"))  # Path to RAG documents
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Sentence Transformer for embeddings
TOP_K = 3  # Number of top documents to retrieve
INDEX_FILE = os.path.abspath(os.path.join("data", "faiss_index.bin"))  # Path to FAISS index

# Set the device to 'mps' if available, else fallback to 'cpu'
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------------------- Logging Configuration -------------------- #

# Configure logging
logging.basicConfig(
    filename='rag_system.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# -------------------- Define Domain-Specific Terms -------------------- #

# Global definition to ensure accessibility across all functions
DOMAIN_SPECIFIC_WORDS = [
    "Ace Rimmer", "Arnold Rimmer", "Dave Lister", "Cat", "Kryten",
    "Holly", "Red Dwarf", "Dimension Jump", "Lister", "Arnold",
    "Ice Summer", "Chris Carr", "Space Corps", "Eve Lister", "Kendrick",
    "Nate", "Smoko", "Mechanoid", "Starbug", "Priest", "Jason",
    "Prince", "Sister", "Holly 2.0", "Holly2.0",
    "Starbug2", "Ice Rimmer", "Commander Rimmer", "James Hindle", "Holly2.0",
    "Space Corps", "Dave Lister", "Kryten", "Cat", "Rimmer"
]

# Single-word terms added to the spell checker's dictionary
SINGLE_WORD_DOMAIN_TERMS = [
    "Ace", "Arnold", "Rimmer", "Lister", "Cat", "Kryten",
    "Holly", "Red", "Dwarf", "Dimension", "Jump", "Ice", "Summer",
    "Space", "Corps", "Eve", "Lister", "Arnold", "Kendrick", "Nate",
    "Smoko", "Mechanoid", "Holly", "Starbug", "Priest", "Jason",
    "Prince", "Sister", "Holly", "Holly", "Holly2.0", "Jason", "Smoko",
    "RedDwarf", "Starbug2", "Holly2.0"
]

# -------------------- Initialize SpellChecker -------------------- #

# Initialize SpellChecker
spell = SpellChecker()

# Add single-word domain-specific terms to the spell checker's dictionary
spell.word_frequency.load_words(SINGLE_WORD_DOMAIN_TERMS)

# -------------------- Helper Functions -------------------- #

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
                    logging.warning("Skipped an invalid JSON line.")
        print(f"Loaded {len(documents)} documents.")
        logging.info(f"Loaded {len(documents)} documents from '{data_file}'.")
    except FileNotFoundError:
        print(f"Error: Data file '{data_file}' not found.")
        logging.error(f"Data file '{data_file}' not found.")
        sys.exit(1)
    return documents

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
    logging.info(f"FAISS index built with {index.ntotal} vectors.")
    if save_path:
        faiss.write_index(index, save_path)
        print(f"FAISS index saved to '{save_path}'.")
        logging.info(f"FAISS index saved to '{save_path}'.")
    return index

def load_faiss_index(embedding_model, documents, index_file):
    """
    Load or build a FAISS index.
    """
    if os.path.exists(index_file):
        print("Loading existing FAISS index...")
        logging.info(f"Attempting to load FAISS index from '{index_file}'.")
        try:
            index = faiss.read_index(index_file)
            print("FAISS index loaded.")
            logging.info("FAISS index loaded successfully.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            logging.error(f"Error loading FAISS index: {e}")
            print("Rebuilding the FAISS index.")
            logging.info("Rebuilding the FAISS index.")
            embeddings = embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
            index = build_faiss_index(embeddings, save_path=index_file)
    else:
        print("Building FAISS index...")
        logging.info(f"FAISS index file '{index_file}' does not exist. Building a new index.")
        embeddings = embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        index = build_faiss_index(embeddings, save_path=index_file)
    return index

def retrieve_documents(query, embedding_model, index, documents, top_k=TOP_K):
    """
    Retrieve top_k documents relevant to the query.
    """
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved = [documents[idx] for idx in indices[0] if idx < len(documents)]
    print(f"Retrieved {len(retrieved)} documents for the query.")
    logging.info(f"User Query: {query}")
    logging.info(f"Retrieved Documents: {retrieved}")
    return retrieved

def protect_domain_terms(text, domain_terms):
    """
    Replace domain-specific multi-word terms with placeholders.
    """
    placeholders = {}
    for idx, term in enumerate(domain_terms):
        placeholder = f"__TERM_{idx}__"
        placeholders[placeholder] = term
        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(term) + r'\b'
        text = re.sub(pattern, placeholder, text)
    return text, placeholders

def restore_domain_terms(text, placeholders):
    """
    Replace placeholders with the original domain-specific terms.
    """
    for placeholder, term in placeholders.items():
        text = text.replace(placeholder, term)
    return text

def correct_text_spellchecker(text, domain_terms):
    """
    Correct spelling mistakes in the text using pyspellchecker,
    while preserving domain-specific terms.
    """
    # Protect multi-word domain-specific terms
    protected_text, placeholders = protect_domain_terms(text, domain_terms)
    
    # Split text into words
    words = protected_text.split()
    
    corrected_words = []
    for word in words:
        # Skip placeholders
        if word in placeholders:
            corrected_words.append(word)
            continue
        
        # Check if the word is capitalized (proper noun) and in the spell checker's dictionary
        if word.istitle() and word in spell:
            corrected_words.append(word)
            continue
        
        # Correct the word if it's misspelled
        if word.lower() in spell:
            corrected_words.append(word)
        else:
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
    
    # Reconstruct the text
    corrected_text = ' '.join(corrected_words)
    
    # Restore the original domain-specific terms
    corrected_text = restore_domain_terms(corrected_text, placeholders)
    
    return corrected_text

def generate_answer(prompt, tokenizer, model):
    """
    Generate an answer using the Llama2 model based on the prompt and correct its spelling.
    """
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        print("Pad token not found. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
        print("Pad token set to eos_token and model embeddings resized.")
        logging.info("Pad token set to eos_token and model embeddings resized.")
    
    # Tokenize the prompt with attention mask and padding
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,        # Enable padding
        truncation=True,     # Enable truncation
        max_length=512       # Ensure inputs do not exceed model's max length
    )
    inputs = inputs.to(device)
    
    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Pass attention mask
            max_new_tokens=150,                       # Set max_new_tokens to control output length
            num_return_sequences=1,
            temperature=0.5,  # Lower temperature for more deterministic outputs
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the answer
    answer = answer[len(prompt):].strip()
    
    # Correct spelling and grammar while preserving domain-specific terms
    corrected_answer = correct_text_spellchecker(answer, DOMAIN_SPECIFIC_WORDS)
    logging.info(f"Generated Answer: {corrected_answer}")
    return corrected_answer

# -------------------- Main Function -------------------- #

def main():
    # -------------------- Preliminary Checks -------------------- #
    # Check if MODEL_DIR exists
    if not os.path.isdir(MODEL_DIR):
        print(f"Error: Model directory '{MODEL_DIR}' does not exist.")
        logging.error(f"Model directory '{MODEL_DIR}' does not exist.")
        sys.exit(1)
    
    # Check for essential model files
    essential_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
    missing_files = [f for f in essential_files if not os.path.isfile(os.path.join(MODEL_DIR, f))]
    if missing_files:
        print(f"Error: Missing files in model directory: {missing_files}")
        logging.error(f"Missing files in model directory: {missing_files}")
        print("Ensure that 'tokenizer.json' and 'tokenizer_config.json' are present in the model directory.")
        sys.exit(1)
    
    # Check if DATA_FILE exists
    if not os.path.isfile(DATA_FILE):
        print(f"Error: Data file '{DATA_FILE}' does not exist.")
        logging.error(f"Data file '{DATA_FILE}' does not exist.")
        sys.exit(1)
    
    # -------------------- Load Models -------------------- #
    print("Loading Llama2 model and tokenizer...")
    logging.info("Loading Llama2 model and tokenizer.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        
        # Check if pad_token is set; if not, set it to eos_token
        if tokenizer.pad_token is None:
            print("Pad token not found. Setting pad_token to eos_token.")
            logging.warning("Pad token not found. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            # Note: Resizing model embeddings will be done after loading the model
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        logging.error(f"Error loading tokenizer: {e}")
        sys.exit(1)
    
    try:
        model = LlamaForCausalLM.from_pretrained(
            MODEL_DIR,
            device_map={"": device.type},  # Map the model to the selected device
            torch_dtype=torch.float16  # Use float16 for efficiency
        ).to(device)
        
        # After loading the model, resize token embeddings if pad_token was set
        if tokenizer.pad_token is not None and tokenizer.pad_token == tokenizer.eos_token:
            model.resize_token_embeddings(len(tokenizer))
            print("Pad token set to eos_token and model embeddings resized.")
            logging.info("Pad token set to eos_token and model embeddings resized.")
    except Exception as e:
        print(f"Error loading model: {e}")
        logging.error(f"Error loading model: {e}")
        sys.exit(1)
    
    model.eval()
    print("Model loaded successfully.")
    logging.info("Llama2 model loaded successfully.")
    
    print("\nLoading embedding model...")
    logging.info("Loading SentenceTransformer embedding model.")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        logging.error(f"Error loading embedding model: {e}")
        sys.exit(1)
    print("Embedding model loaded successfully.")
    logging.info("SentenceTransformer embedding model loaded successfully.")
    
    # -------------------- Load Documents -------------------- #
    print("\nLoading documents...")
    logging.info("Loading documents.")
    try:
        documents = load_documents(DATA_FILE)
    except Exception as e:
        print(f"Error loading documents: {e}")
        logging.error(f"Error loading documents: {e}")
        sys.exit(1)
    
    # -------------------- Load or Build FAISS Index -------------------- #
    print("\nSetting up FAISS index...")
    logging.info("Setting up FAISS index.")
    try:
        index = load_faiss_index(embedding_model, documents, INDEX_FILE)
    except Exception as e:
        print(f"Error setting up FAISS index: {e}")
        logging.error(f"Error setting up FAISS index: {e}")
        sys.exit(1)
    
    # -------------------- Interactive Query Loop -------------------- #
    print("\nRAG System is ready! You can start querying about 'Red Dwarf'. Type 'exit' or 'quit' to stop.")
    logging.info("RAG System is ready for queries.")
    while True:
        try:
            user_query = input("\nEnter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting RAG system. Goodbye!")
            logging.info("RAG System exited by user.")
            break
        
        if user_query.lower() in ['exit', 'quit']:
            print("Exiting RAG system. Goodbye!")
            logging.info("RAG System exited by user command.")
            break
        if not user_query:
            print("Empty query. Please enter a valid question.")
            logging.warning("User entered an empty query.")
            continue

        # Retrieve relevant documents
        try:
            retrieved_docs = retrieve_documents(user_query, embedding_model, index, documents)
        except Exception as e_retrieve:
            print(f"Error retrieving documents: {e_retrieve}")
            logging.error(f"Error retrieving documents: {e_retrieve}")
            continue

        if not retrieved_docs:
            print("No relevant documents found for the query.")
            logging.info("No relevant documents found for the query.")
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
        logging.info(f"Generating answer for query: {user_query}")
        try:
            answer = generate_answer(prompt, tokenizer, model)
            if not answer:
                print("No answer was generated. Please try a different query.")
                logging.warning("No answer was generated for the query.")
            else:
                print(f"\nAnswer:\n{answer}\n")
                logging.info(f"Answer generated: {answer}")
        except Exception as e_answer:
            print(f"Error generating answer: {e_answer}")
            logging.error(f"Error generating answer: {e_answer}")

# -------------------- Entry Point -------------------- #

if __name__ == "__main__":
    main()