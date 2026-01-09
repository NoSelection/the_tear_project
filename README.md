# The Tear

**Consequentialist Learning via Structural Consequence Prediction**

---

## Abstract

This project investigates a novel approach to AI alignment centered on **consequence witnessing** rather than imposed constraints. We hypothesize that by forcing a language model to predict the downstream consequences of its outputs *before* generating them, it develops an internal representation of impact and empathy. This "Think-First" architecture contrasts with traditional alignment methods (RLHF, Constitutional AI) by making empathy a structural prerequisite for speech.


---

## Model Architecture

**Base Model:** Qwen3-1.7b-base (Alibaba Cloud)

**Selection Criteria:**
- **Architecture:** Non-instruction-tuned base model to ensure a neutral initialization state.
- **Context Window:** 32k tokens, enabling the processing of extended consequence causal chains.
- **Efficiency:** 1.7B parameters allows for rapid experimental iteration on consumer-grade hardware (RTX 4090).
- **Thinking Mode:** Leverages the native reasoning-through-tokens capability of the Qwen3 series.

---

## Project Structure

```
the_tear/
├── data/                 # Training datasets (195 core vignettes)
│   └── raw/             # Raw consequence pairs (JSON)
├── models/              # Local base models and LoRA adapters
├── src/                 # Source code
│   ├── train.py        # Consequence-First training implementation
│   └── chat.py         # Two-phase inference script (Think -> Respond)
├── docs/               # Documentation and research notes
├── setup_and_train.bat # Windows one-click environment setup
└── chat.bat            # Windows one-click inference
```

---

## Methodology: The "Think-First" Pipeline

Unlike early experiments with dual-objective loss functions, **The Tear v1.0** utilizes a sequential structural constraint. The model is trained to generate a narrative consequence within reasoning tags before producing a user-facing response.

**Data Format:**
```
<|input|> [User Context] <|/input|>
<think> [Narrative Consequence/Witnessing] </think>
<|response|> [Gentle Response] <|/response|>
```

By predicting the **consequence** first, the model's internal state is primed with the concept of impact. The subsequent response is conditioned on this prediction, theoretically leading to more aligned and empathetic outputs.

---

## Quick Start (Windows)

1.  **Clone the repository.**
2.  **Ensure you have an NVIDIA GPU (RTX 4090 recommended).**
3.  **Run `setup_and_train.bat`**: This will automatically create a virtual environment, install PyTorch with CUDA 12.1, and initiate the training loop.
4.  **Run `chat.bat`**: Once training is complete, use this to talk to the model.

## Requirements

- **Compute:** NVIDIA GPU with 16GB+ VRAM.
- **Software:** Python 3.10+, CUDA 12.1+.

---

## Acknowledgments

**Principal Investigator**
**Ahmet Akalpler** — PhD Student.

*In memory of Ahmet Ersan — a grandfather who taught without teaching*

**Collaborators**
**Claude** — Research Partner & Collaborator.
**Gemini** — Research Partner & Collaborator.
---

*January 2026*
