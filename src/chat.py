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


def clean_text(text: str) -> str:
    """Clean up artifacts, fix encoding, and remove non-English leaks."""
    import re
    
    # Fix common UTF-8 to Windows-1252 encoding glitches (like â€” for —)
    try:
        text = text.encode('latin-1').decode('utf-8')
    except:
        pass # If it's already okay, leave it
        
    # Replace common broken symbols manually if needed
    text = text.replace('â€”', '—').replace('â€™', "'").replace('â€œ', '"').replace('â€\x9d', '"')

    # Remove non-ASCII/leaked language artifacts (keeps English, numbers, punctuation)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def chat(model, tokenizer, user_input: str, show_thoughts: bool = False):
    """Generate a response using a two-step process: Think -> Respond."""

    # 1. PHASE ONE: THINKING
    # Start the prompt and tell it to think
    think_prompt = f"{SPECIAL_TOKENS['input_start']}{user_input}{SPECIAL_TOKENS['input_end']}{SPECIAL_TOKENS['think_start']}"
    inputs = tokenizer(think_prompt, return_tensors="pt").to(model.device)
    
    # We want it to stop at </think>
    stop_token_ids = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['think_end']), tokenizer.eos_token_id]

    with torch.no_grad():
        thought_outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_thought_text = tokenizer.decode(thought_outputs[0], skip_special_tokens=False)
    
    # Extract just the new thought part
    thought_content = full_thought_text.split(SPECIAL_TOKENS['think_start'])[-1]
    if SPECIAL_TOKENS['think_end'] in thought_content:
        thought_content = thought_content.split(SPECIAL_TOKENS['think_end'])[0]
    
    # 2. PHASE TWO: RESPONDING
    # Construct prompt with the finished thought
    response_prompt = f"{full_thought_text}"
    if SPECIAL_TOKENS['think_end'] not in response_prompt:
        response_prompt += SPECIAL_TOKENS['think_end']
    response_prompt += SPECIAL_TOKENS['response_start']
    
    inputs = tokenizer(response_prompt, return_tensors="pt").to(model.device)
    
    # Stop at </|response|>
    stop_token_ids = [tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['response_end']), tokenizer.eos_token_id]

    with torch.no_grad():
        response_outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.5, # Cooler for the actual speech
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=stop_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_text = tokenizer.decode(response_outputs[0], skip_special_tokens=False)
    response_content = full_text.split(SPECIAL_TOKENS['response_start'])[-1]
    if SPECIAL_TOKENS['response_end'] in response_content:
        response_content = response_content.split(SPECIAL_TOKENS['response_end'])[0]

    return clean_text(response_content), clean_text(thought_content)

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