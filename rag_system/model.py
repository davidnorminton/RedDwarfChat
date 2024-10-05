# rag_system/model.py

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import sys

def load_language_model(model_dir, device):
    """
    Load the Llama2 language model and tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Check if pad_token is set; if not, set it to eos_token
        if tokenizer.pad_token is None:
            print("Pad token not found. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            # Note: Resizing model embeddings will be done after loading the model
        
        model = LlamaForCausalLM.from_pretrained(
            model_dir,
            device_map={"": device.type},  # Map the model to the selected device
            torch_dtype=torch.float16  # Use float16 for efficiency
        ).to(device)
        
        # After loading the model, resize token embeddings if pad_token was set
        if tokenizer.pad_token is not None and tokenizer.pad_token == tokenizer.eos_token:
            model.resize_token_embeddings(len(tokenizer))
            print("Pad token set to eos_token and model embeddings resized.")
        
        model.eval()
        print("Language model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading language model: {e}")
        sys.exit(1)

def generate_answer(prompt, tokenizer, model, device, max_new_tokens=150):
    """
    Generate an answer using the Llama2 model based on the prompt.
    """
    try:
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
                max_new_tokens=max_new_tokens,            # Control output length
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
        return answer
    except Exception as e:
        print(f"Error during answer generation: {e}")
        return ""