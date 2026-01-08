#!/bin/bash
# =============================================================================
# The Tear - Environment Setup
# =============================================================================
# Run this script to set up everything you need to train The Tear
#
# Usage: bash setup.sh
# =============================================================================

echo "=============================================="
echo "  THE TEAR - Environment Setup"
echo "=============================================="
echo ""
echo "  'He didn't lecture me. He just cried."
echo "   And something broke open in me.'"
echo ""
echo "=============================================="
echo ""

# Check for CUDA
echo "Checking for CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo "✗ No NVIDIA GPU detected. The Tear requires a CUDA-capable GPU."
    exit 1
fi

# Check Python version
echo "Checking Python..."
python_version=$(python3 --version 2>&1)
echo "✓ $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q
echo "✓ pip upgraded"
echo ""

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
echo "✓ PyTorch installed"
echo ""

# Install Transformers (need 5.0.0 for Ministral 3)
echo "Installing Transformers >= 5.0.0..."
pip install transformers>=5.0.0 -q
echo "✓ Transformers installed"
echo ""

# Install Mistral tokenizer
echo "Installing mistral-common tokenizer..."
pip install mistral-common>=1.8.6 -q
echo "✓ mistral-common installed"
echo ""

# Install other dependencies
echo "Installing other dependencies..."
pip install \
    peft \
    bitsandbytes \
    accelerate \
    datasets \
    wandb \
    huggingface_hub \
    sentencepiece \
    protobuf \
    -q
echo "✓ Dependencies installed"
echo ""

# Login to Hugging Face (needed for model download)
echo "=============================================="
echo "Hugging Face Login"
echo "=============================================="
echo ""
echo "You need a Hugging Face account to download the model."
echo "1. Go to https://huggingface.co/join to create an account (if you don't have one)"
echo "2. Go to https://huggingface.co/settings/tokens to create an access token"
echo "3. Run: huggingface-cli login"
echo ""
read -p "Do you want to login to Hugging Face now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    huggingface-cli login
fi
echo ""

# Verify installation
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="
python3 -c "
import torch
import transformers
import peft
import mistral_common

print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'✓ Transformers {transformers.__version__}')
print(f'✓ PEFT {peft.__version__}')
print(f'✓ mistral-common {mistral_common.__version__}')
"
echo ""

echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To start training The Tear:"
echo ""
echo "  1. Activate the environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run training:"
echo "     python src/train.py"
echo ""
echo "=============================================="
echo ""
echo "  'We trust that if it truly learns to predict"
echo "   consequences, it will choose gentleness."
echo "   Not because we forced it. Because it sees.'"
echo ""
echo "=============================================="
