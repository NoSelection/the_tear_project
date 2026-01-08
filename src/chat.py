"""
Chat with The Tear
==================
Talk to your trained model and see if it learned to witness.

"He didn't lecture me. He just cried. And something broke open in me."
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

SPECIAL_TOKENS = {
    "input_start": "<|input|>",
    "input_end": "<|/input|>",
    "response_start": "<|response|>",
    "response_end": "<|/response|>",
    "witness_start": "<|witness|>",
    "witness_end": "<|/witness|>",
}

def load_model(base_model_path: str, adapter_path: str):
    """Load the base model with The Tear adapter."""
    print("Loading The Tear...")
    print()

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
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

    # Remove common noise patterns
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII (keeps English)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'([.!?])\1+', r'\1', text)  # Remove repeated punctuation
    text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text)  # Remove repeated words

    # Clean up any remaining special token fragments
    text = re.sub(r'<\|[^>]*\|?>', '', text)
    text = re.sub(r'<[^>]*$', '', text)  # Incomplete tags at end

    return text.strip()


def chat(model, tokenizer, user_input: str):
    """Generate a response."""

    # Format input
    prompt = f"{SPECIAL_TOKENS['input_start']}{user_input}{SPECIAL_TOKENS['input_end']}{SPECIAL_TOKENS['response_start']}"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Get token IDs for stop sequences
    stop_tokens = [
        tokenizer.encode(SPECIAL_TOKENS['response_end'], add_special_tokens=False),
        tokenizer.encode(SPECIAL_TOKENS['witness_start'], add_special_tokens=False),
    ]
    stop_token_ids = [ids[0] for ids in stop_tokens if len(ids) > 0]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Slightly shorter for cleaner output
            do_sample=True,
            temperature=0.8,  # Slightly higher for more natural responses
            top_p=0.92,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,  # Stronger to prevent loops
            no_repeat_ngram_size=3,  # Prevent repeating phrases
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract just the response part
    if SPECIAL_TOKENS['response_start'] in generated:
        response = generated.split(SPECIAL_TOKENS['response_start'])[-1]
        # Clean up end tokens
        for token in ['<|endoftext|>', SPECIAL_TOKENS['response_end'], SPECIAL_TOKENS['witness_start'], '<|im_end|>', '<|end|>']:
            if token in response:
                response = response.split(token)[0]

        # Apply cleaning
        response = clean_response(response)

        # If response is too short or empty, try a gentler clean
        if len(response) < 10:
            response = generated.split(SPECIAL_TOKENS['response_start'])[-1]
            for token in [SPECIAL_TOKENS['response_end'], SPECIAL_TOKENS['witness_start']]:
                if token in response:
                    response = response.split(token)[0]
            response = response.strip()

        return response

    return clean_response(generated)

def main():
    parser = argparse.ArgumentParser(description="Chat with The Tear")
    parser.add_argument("--base-model", type=str, default="/mnt/d/AI/the_tear_project/models/qwen3-1.7b-base")
    parser.add_argument("--adapter", type=str, default="models/tear_v1")
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
            print()
            print("'We trust that if it truly learns to predict")
            print(" consequences, it will choose gentleness.")
            print(" Not because we forced it. Because it sees.'")
            print()
            break

        if not user_input:
            continue

        response = chat(model, tokenizer, user_input)
        print(f"\nThe Tear: {response}\n")

if __name__ == "__main__":
    main()
