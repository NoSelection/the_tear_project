"""Quick test to verify The Tear setup works"""
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import transformers
import peft
import bitsandbytes
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")

print("\nALL GOOD - Ready to train The Tear!")
