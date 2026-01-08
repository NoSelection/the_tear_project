"""
The Tear: Consequence-First Training
====================================

Train a model to predict the consequence BEFORE responding.
Structure: <input> ... </input> <think> consequence </think> <response> ... </response>

Created by: Ahmet Akalpler & Claude
December 2025
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import argparse


# =============================================================================
# SPECIAL TOKENS - Qwen 3 Style Reasoning
# =============================================================================

SPECIAL_TOKENS = {
    "input_start": "<|input|>",
    "input_end": "<|/input|>",
    "think_start": "<think>",      # Consequence prediction starts here
    "think_end": "</think>",
    "response_start": "<|response|>",
    "response_end": "<|/response|>",
}


# =============================================================================
# DATASET - Consequence-First Format
# =============================================================================

class TearDataset(Dataset):
    """
    Dataset for The Tear training (Think-First Architecture).
    
    The model learns to generate: 
    <input> Msg </input> <think> Consequence </think> <response> Reply </response>
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Create training examples
        for item in raw_data:
            # Gentle example
            if "response_gentle" in item:
                self.examples.append({
                    "input": item["input"],
                    "response": item["response_gentle"],
                    "consequence": item["consequence_gentle"],
                    "is_gentle": True
                })
            
            # Harmful example (kept for contrast, model learns to predict harm too)
            if "response_harmful" in item:
                self.examples.append({
                    "input": item["input"],
                    "response": item["response_harmful"],
                    "consequence": item["consequence_harmful"],
                    "is_gentle": False
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format: Input -> Think (Consequence) -> Response
        text = (
            f"{SPECIAL_TOKENS['input_start']}{example['input']}{SPECIAL_TOKENS['input_end']}"
            f"{SPECIAL_TOKENS['think_start']}{example['consequence']}{SPECIAL_TOKENS['think_end']}"
            f"{SPECIAL_TOKENS['response_start']}{example['response']}{SPECIAL_TOKENS['response_end']}"
        )
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "is_gentle": example["is_gentle"]
        }


# =============================================================================
# MODEL SETUP
# =============================================================================

def setup_model(model_name: str, device: str = "cuda"):
    """Load and prepare model for QLoRA training."""
    print(f"Loading {model_name}...")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except:
        print(f"Warning: Could not load tokenizer from {model_name}, trying default Qwen.")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Add special tokens
    special_tokens_list = list(SPECIAL_TOKENS.values())
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train(
    model_name: str,
    data_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    gradient_accumulation_steps: int = 1, # Optimized for 4090
):
    print("=" * 60)
    print("THE TEAR - Training (Consequence-First) [OPTIMIZED]")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model, tokenizer = setup_model(model_name, device)
    
    # Load dataset
    dataset = TearDataset(data_path, tokenizer)
    print(f"Loaded {len(dataset)} training examples")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training
    model.train()
    global_step = 0
    total_steps = len(dataloader) * epochs
    
    print(f"Total steps to train: {total_steps}")
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        epoch_loss = 0
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Standard Causal LM Loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            
            # Scale loss
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            epoch_loss += loss.item()
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                print(f"  Step {global_step}/{total_steps}: loss = {loss.item():.4f}")

                # Sample generation (less frequent to save time, maybe just once per epoch)
                if global_step % 20 == 0:
                    print("\n  --- Sample generation ---")
                    model.eval()
                    test_input = f"{SPECIAL_TOKENS['input_start']}I've been feeling really lonely lately.{SPECIAL_TOKENS['input_end']}{SPECIAL_TOKENS['think_start']}"
                    test_ids = tokenizer(test_input, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        output = model.generate(
                            test_ids,
                            max_new_tokens=100, # Shorter sample for speed
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    generated = tokenizer.decode(output[0], skip_special_tokens=False)
                    print(f"  {generated[:300]}...") # Truncated output
                    print("  --- End sample ---\n")
                    model.train()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")
    
    # Save
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("\n" + "=" * 60)
    print("THE TEAR - Training complete")
    print("=" * 60)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train The Tear model")
    parser.add_argument("--model", type=str, default="models/qwen3-1.7b-base")
    parser.add_argument("--data", type=str, default="data/raw/all_seed_consequences.json")
    parser.add_argument("--output", type=str, default="models/tear_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8) # Default 8 for 4090
    parser.add_argument("--lr", type=float, default=2e-4)
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )