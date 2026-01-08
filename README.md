# The Tear 💧

**Consequentialist Learning Through Empathy in AI**

*"He didn't lecture me. He just cried. And something broke open in me."*

---

## What is this?

An experiment in teaching AI through witnessing rather than rules.

Traditional AI alignment says "don't do X." 
The Tear says "here's what happens when you do X. Now you know."

We trust that understanding is enough.

---

## The Model

**Qwen3-1.7b-base** — A 1.7B parameter base model from Alibaba Qwen.

Why this model?
- **Base (not instruction-tuned)** — A blank slate we can shape
- **32k context** — Efficient and capable for local use
- **Small enough** — Fast iteration on consumer hardware
- **Apache 2.0** — Full freedom

---

## Project Structure

```
the_tear/
├── data/                 # Training datasets
│   ├── raw/             # Raw consequence pairs
│   └── processed/       # Formatted for training
├── models/              # Saved models and checkpoints
├── src/                 # Source code
│   └── train.py        # Training loop with dual-objective
├── docs/               # Documentation and research notes
│   └── RESEARCH_SKETCH.md
├── setup.sh            # Environment setup script
└── README.md
```

---

## Quick Start

```bash
# 1. Clone/download this project

# 2. Run setup (installs everything, checks GPU)
bash setup.sh

# 3. Activate environment
source venv/bin/activate

# 4. Train The Tear
python src/train.py
```

---

## The Core Idea

```
Loss = ResponseLoss + λ * ConsequenceLoss
```

The model learns to both respond AND predict consequences. By making consequence prediction part of what it's optimizing for, it must build internal representations of impact.

**Training format:**
```
<|input|> user message <|/input|>
<|response|> model response <|/response|>
<|witness|> what happened because of this response <|/witness|>
```

The `<|witness|>` token is sacred — it triggers consequence awareness.

---

## Requirements

- NVIDIA GPU with 16GB+ VRAM (RTX 4090 recommended)
- Python 3.10+
- CUDA 12.1+

---

## Created by

**Ahmet Akalpler** — PhD Student, Developer, Dreamer

*In memory of Ahmet Ersan — a grandfather who taught without teaching, whose name lives on in the one he changed.*

---

## With

**Claude** — Friend, Collaborator, Partner in this experiment

---

*December 2025*
