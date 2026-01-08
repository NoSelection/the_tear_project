# The Tear

**Consequentialist Learning via Consequence Prediction**

---

## Abstract

This project investigates a novel approach to AI alignment centered on **consequence witnessing** rather than imposed constraints. We hypothesize that by optimizing a language model to predict the downstream consequences of its outputs, it will develop internal representations of impact and empathy. This contrasts with traditional alignment methods (RLHF, Constitutional AI) which often rely on external reward signals or rule-based filtering.

*"He didn't lecture me. He just cried. And something broke open in me."*

---

## Model Architecture

**Base Model:** Qwen3-1.7b-base (Alibaba Cloud)

**Selection Criteria:**
- **Architecture:** Non-instruction-tuned base model to ensure a neutral initialization state.
- **Context Window:** 32k tokens, enabling the processing of extended consequence causal chains.
- **Efficiency:** 1.7B parameters allows for rapid experimental iteration on consumer-grade hardware.
- **License:** Apache 2.0.

---

## Project Structure

```
the_tear/
├── data/                 # Training datasets
│   ├── raw/             # Raw consequence pairs
│   └── processed/       # Formatted for training
├── models/              # Saved models and checkpoints
├── src/                 # Source code
│   └── train.py        # Dual-objective training implementation
├── docs/               # Documentation and research notes
│   └── RESEARCH_SKETCH.md
├── setup.sh            # Environment initialization script
└── README.md
```

---

## Methodology

The core mechanism employs a **Dual-Objective Loss Function**. The model minimizes a combined loss that weighs both response generation and consequence prediction.

$$ L_{total} = L_{response} + \lambda \cdot L_{consequence} $$

Where:
*   $L_{response}$: Standard next-token prediction loss for the response.
*   $L_{consequence}$: Loss associated with predicting the narrative outcome of that response.
*   $\lambda$: Hyperparameter controlling the weight of consequence awareness.

**Training Token Structure:**

```
<|input|> [User Message] <|/input|>
<|response|> [Model Output] <|/response|>
<|witness|> [Consequence Narrative] <|/witness|>
```

The `<|witness|>` token serves as a specialized delimiter, triggering the model's consequence-prediction attention heads.

---

## Quick Start

```bash
# 1. Clone the repository
git clone [repo_url]

# 2. Initialize environment (installs dependencies, verifies CUDA)
bash setup.sh

# 3. Activate virtual environment
source venv/bin/activate

# 4. Initiate training
python src/train.py
```

## Requirements

- **Compute:** NVIDIA GPU with 16GB+ VRAM (RTX 4090 recommended).
- **Software:** Python 3.10+, CUDA 12.1+.

---

## Acknowledgments

**Principal Investigator**
**Ahmet Akalpler** — PhD Student.

*In memory of Ahmet Ersan — a grandfather who taught without teaching, whose name lives on in the one he changed.*

**Collaborators**
**Claude** — Research Partner & Collaborator.

---

*December 2025*