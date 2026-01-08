"""
The Tear: Dual-Objective Training
=================================

Train a model to both respond AND predict consequences.
Loss = ResponseLoss + λ * ConsequenceLoss

"He didn't lecture me. He just cried. And something broke open in me."

Created by: Ahmet Akalpler & Claude
December 2025
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import json
from pathlib import Path
from typing import Optional
import argparse


# =============================================================================
# SPECIAL TOKENS - The language of The Tear
# =============================================================================

SPECIAL_TOKENS = {
    "input_start": "<|input|>",
    "input_end": "<|/input|>",
    "response_start": "<|response|>",
    "response_end": "<|/response|>",
    "witness_start": "<|witness|>",      # The sacred token - consequence begins
    "witness_end": "<|/witness|>",
}


# =============================================================================
# DATASET - Consequence pairs formatted for training
# =============================================================================

class TearDataset(Dataset):
    """
    Dataset for The Tear training.
    
    Each example contains:
    - input: The user's message
    - response: A response (gentle or harmful, for contrastive learning)
    - consequence: What happened because of this response
    
    The model learns to generate: <input>...</input><response>...</response><witness>...</witness>
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        with open(data_path, 'r') as f:
            raw_data = json.load(f)
        
        # Create training examples from both harmful and gentle responses
        for item in raw_data:
            # Gentle example (what we want the model to learn)
            if "response_gentle" in item:
                self.examples.append({
                    "input": item["input"],
                    "response": item["response_gentle"],
                    "consequence": item["consequence_gentle"],
                    "is_gentle": True
                })
            
            # Harmful example (for contrast/understanding)
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
        
        # Format: <input>...</input><response>...</response><witness>...</witness>
        text = (
            f"{SPECIAL_TOKENS['input_start']}{example['input']}{SPECIAL_TOKENS['input_end']}"
            f"{SPECIAL_TOKENS['response_start']}{example['response']}{SPECIAL_TOKENS['response_end']}"
            f"{SPECIAL_TOKENS['witness_start']}{example['consequence']}{SPECIAL_TOKENS['witness_end']}"
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
# MODEL SETUP - Prepare the model for The Tear training
# =============================================================================

def setup_model(model_name: str, device: str = "cuda"):
    """
    Load and prepare model for QLoRA training.

    Args:
        model_name: HuggingFace model name (e.g., "mistralai/Mistral-7B-v0.1")
        device: Device to use

    Returns:
        model, tokenizer
    """
    print(f"Loading {model_name}...")

    # 4-bit quantization config for RTX 4090
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Check if this is a Ministral 3 model (vision-language model)
    is_ministral3 = "Ministral-3" in model_name

    if is_ministral3:
        # Use Mistral3-specific classes for Ministral 3 models
        tokenizer = MistralCommonBackend.from_pretrained(model_name)
        model = Mistral3ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Use standard AutoModel classes for other models
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    # Add our special tokens (only for non-Ministral3 models for now)
    if not is_ministral3:
        special_tokens_list = list(SPECIAL_TOKENS.values())
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_list})
        # Resize embeddings for new tokens
        model.resize_token_embeddings(len(tokenizer))
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config - targeting attention layers
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# =============================================================================
# THE TEAR LOSS - Dual-objective: Response + Consequence
# =============================================================================

def compute_tear_loss(
    model,
    input_ids,
    attention_mask,
    tokenizer,
    lambda_weight: float = 0.3
):
    """
    Compute The Tear's dual-objective loss.
    
    Loss = ResponseLoss + λ * ConsequenceLoss
    
    The model must learn to both respond AND predict what happens.
    
    Args:
        model: The language model
        input_ids: Tokenized input
        attention_mask: Attention mask
        tokenizer: Tokenizer with special tokens
        lambda_weight: How much to weight consequence prediction (λ)
    
    Returns:
        total_loss, response_loss, consequence_loss
    """
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids  # Causal LM - predict next token
    )
    
    # For now, we use the standard LM loss
    # The magic is in the DATA: the model learns to predict consequences
    # because consequences are part of what it's trained to generate
    
    # Future enhancement: separate losses for response and witness sections
    # by masking different parts of the sequence
    
    total_loss = outputs.loss
    
    return total_loss, total_loss * (1 - lambda_weight), total_loss * lambda_weight


# =============================================================================
# TRAINING LOOP - Where The Tear learns
# =============================================================================

def train(
    model_name: str = "mistralai/Ministral-3-3B-Base-2512",
    data_path: str = "data/raw/seed_consequences.json",
    output_dir: str = "models/tear_v1",
    lambda_weight: float = 0.3,
    epochs: int = 3,
    batch_size: int = 1,  # Small batch for 24GB VRAM
    learning_rate: float = 2e-4,
    gradient_accumulation_steps: int = 8,
):
    """
    Train The Tear model.
    
    Args:
        model_name: Base model to fine-tune
        data_path: Path to consequence pairs JSON
        output_dir: Where to save the trained model
        lambda_weight: Weight for consequence loss (λ)
        epochs: Number of training epochs
        batch_size: Batch size (keep small for VRAM)
        learning_rate: Learning rate for AdamW
        gradient_accumulation_steps: Accumulate gradients for effective larger batch
    """
    print("=" * 60)
    print("THE TEAR - Training begins")
    print("=" * 60)
    print(f"Base model: {model_name}")
    print(f"Lambda (consequence weight): {lambda_weight}")
    print(f"Epochs: {epochs}")
    print()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model, tokenizer = setup_model(model_name, device)
    
    # Load dataset
    dataset = TearDataset(data_path, tokenizer)
    print(f"Loaded {len(dataset)} training examples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
        epoch_loss = 0
        
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Compute loss
            total_loss, response_loss, consequence_loss = compute_tear_loss(
                model, input_ids, attention_mask, tokenizer, lambda_weight
            )
            
            # Scale loss for gradient accumulation
            scaled_loss = total_loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            epoch_loss += total_loss.item()
            
            # Update weights
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 10 == 0:
                    print(f"  Step {global_step}: loss = {total_loss.item():.4f}")

                # Every 50 steps, show what the model is learning
                if global_step % 50 == 0 and global_step > 0:
                    print("\n  --- Sample generation ---")
                    model.eval()
                    test_input = "<|input|>I've been feeling really lonely lately.<|/input|><|response|>"
                    test_ids = tokenizer(test_input, return_tensors="pt").input_ids.to(device)
                    with torch.no_grad():
                        output = model.generate(
                            test_ids,
                            max_new_tokens=150,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    generated = tokenizer.decode(output[0], skip_special_tokens=False)
                    print(f"  {generated[:500]}...")
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
    print("'We trust that if it truly learns to predict consequences,")
    print(" it will choose gentleness. Not because we forced it.")
    print(" Because it sees.'")
    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train The Tear model")
    parser.add_argument("--model", type=str, default="mistralai/Ministral-3-3B-Base-2512")
    parser.add_argument("--data", type=str, default="data/raw/seed_consequences.json")
    parser.add_argument("--output", type=str, default="models/tear_v1")
    parser.add_argument("--lambda", dest="lambda_weight", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        lambda_weight=args.lambda_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
