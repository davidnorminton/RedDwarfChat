import os
import torch  # Ensure torch is imported
from huggingface_hub import login
from transformers import LlamaTokenizer, LlamaForCausalLM

# Step 1: Authenticate with Hugging Face
def authenticate(token: str):
    """
    Authenticate to Hugging Face Hub using your API token.
    """
    login(token=token)
    print("Successfully authenticated to Hugging Face Hub.")

# Step 2: Define model parameters
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Replace with the desired LLaMA 2 model variant
CACHE_DIR = os.path.join("Model", "llama2")  # Directory to store the downloaded model

# Step 3: Download the tokenizer and model
def download_model(model_name: str, cache_dir: str):
    """
    Download the tokenizer and model weights from Hugging Face.
    """
    print(f"Downloading tokenizer for {model_name}...")
    tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("Tokenizer downloaded successfully.")

    print(f"Downloading model {model_name}...")
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        revision="main",
        torch_dtype=torch.float16,  # Use float16 for efficiency; adjust if needed
        low_cpu_mem_usage=True
    )
    print("Model downloaded successfully.")

    return tokenizer, model

# Step 4: Save the model locally (optional)
def save_model(tokenizer, model, save_dir: str):
    """
    Save the tokenizer and model to a local directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"Saving tokenizer to {save_dir}...")
    tokenizer.save_pretrained(save_dir)
    
    print(f"Saving model to {save_dir}...")
    model.save_pretrained(save_dir)
    
    print("Model and tokenizer saved successfully.")

def main():
    # Replace this with your actual Hugging Face token
    HUGGINGFACE_TOKEN = "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"  # Ensure no trailing spaces
    
    authenticate(HUGGINGFACE_TOKEN)
    
    tokenizer, model = download_model(MODEL_NAME, CACHE_DIR)
    
    # Optional: Save the model and tokenizer locally
    SAVE_DIR = CACHE_DIR  # Saving directly to the 'Model/llama2' directory
    save_model(tokenizer, model, SAVE_DIR)

if __name__ == "__main__":
    main()