"""
Chat with The Tear
==================
Talk to your trained model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import sys

# =============================================================================
# SPECIAL TOKENS - Qwen 3 Style Reasoning
# =============================================================================

SPECIAL_TOKENS = {
    "input_start": "<|input|>",
    "input_end": "<|/input|>",
    "think_start": "<think>",
    "think_end": "</think>",
    "response_start": "<|response|>",
    "response_end": "<|/response|>",
}

def load_model(base_model_path: str, adapter_path: str):
    """Load the base model with The Tear adapter."""
    print("Loading The Tear...")
    print()

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    except:
        print("Warning: Could not load tokenizer from path, trying default Qwen.")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
        
    tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    special_tokens_list = list(SPECIAL_TOKENS.values())
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print("The Tear is ready.")
    print()
    return model, tokenizer

def clean_response(text: str) -> str:
    """Clean up artifacts from generated text."""
    import re

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove repeated words (e.g. "the the")
    text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)

    # Clean up any remaining special token fragments
    text = re.sub(r'<\|[^>]*\|?>', '', text)
    text = re.sub(r'<[^>]*$', '', text)  # Incomplete tags at end

    return text.strip()


def chat(model, tokenizer, user_input: str, show_thoughts: bool = False):
    """Generate a response, optionally showing the thought process."""

    # Format input with a tiny 'anchor' to the gentle side
    prompt = f"{SPECIAL_TOKENS['input_start']}{user_input}{SPECIAL_TOKENS['input_end']}{SPECIAL_TOKENS['think_start']}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.5, # Lower temperature for more stability
            top_p=0.9,
            repetition_penalty=1.2, # Prevent looping
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Robust Parsing
    thought_content = "Thinking..."
    response_content = ""
    
    # 1. Try to get thought
    if SPECIAL_TOKENS['think_start'] in generated:
        parts = generated.split(SPECIAL_TOKENS['think_start'])
        if len(parts) > 1:
            thought_part = parts[1]
            if SPECIAL_TOKENS['think_end'] in thought_part:
                thought_content = thought_part.split(SPECIAL_TOKENS['think_end'])[0]
            elif SPECIAL_TOKENS['response_start'] in thought_part:
                thought_content = thought_part.split(SPECIAL_TOKENS['response_start'])[0]
            else:
                # If no end tag, take a chunk
                thought_content = thought_part[:200] + "..."

    # 2. Try to get response
    if SPECIAL_TOKENS['response_start'] in generated:
        parts = generated.split(SPECIAL_TOKENS['response_start'])
        if len(parts) > 1:
            response_content = parts[1].split(SPECIAL_TOKENS['response_end'])[0]
    
    # Fallback: if no response tag, the model just kept writing the thought
    if not response_content or len(response_content.strip()) < 5:
        # Check if the 'thought' actually contains a gentle response at the end
        response_content = "The model is still learning to separate thoughts from speech. \n\nRAW OUTPUT: " + clean_response(generated[len(prompt):])

    return clean_response(response_content), clean_response(thought_content)

def main():
    parser = argparse.ArgumentParser(description="Chat with The Tear")
    parser.add_argument("--base-model", type=str, default="models/qwen3-1.7b-base")
    parser.add_argument("--adapter", type=str, default="models/tear_v1")
    parser.add_argument("--show-thoughts", action="store_true", help="Show internal consequence prediction")
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter)

    print("=" * 50)
    print("THE TEAR")
    print("=" * 50)
    print("Share something. See what it says.")
    print("Type 'quit' to exit.")
    print("=" * 50)
    print()

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == 'quit':
            print("\nExiting.")
            break

        if not user_input:
            continue

        response, thought = chat(model, tokenizer, user_input, args.show_thoughts)
        
        if args.show_thoughts and thought:
            print(f"\n[Internal Thought: {thought}]")
            
        print(f"\nThe Tear: {response}\n")

if __name__ == "__main__":
    main()